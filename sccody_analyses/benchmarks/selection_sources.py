"""Prepare comparator donor/source metadata for localization reports.

Use the fixed source ages recorded in configs/aida.json:

  fit<k>_internal_fixed_sources_bidirectional.json   scNODE, ARTEMIS
  fit<k>_internal_fixed_source_age19_forward.json    scDiffEq (forward-only SDE)

    python -m sccody_analyses.benchmarks.selection_sources --h5ad data/aida_age_holdout_train.h5ad --full-h5ad data/aida_counts_hvg.h5ad
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import pandas as pd

DESIGNS = {
    "internal_fixed_sources_bidirectional": ("bidirectional_source_cohort", "bidirectional", None),
    "internal_fixed_source_age19_forward": ("full_forward_cohort", "earlier", [19.0]),
}


def manifest(obs, fit, holdout, design, direction, sources, h5ad_name, holdout_path, *, target_ages=None):
    holdout_donors = set(map(str, holdout["holdout_donors"]))
    donors = obs.groupby("donor_id", observed=True).agg(age=("age", "first"), sex=("sex", "first"), cells=("age", "size"))
    train = donors[~donors.index.isin(holdout_donors)]
    targets = donors[donors.index.isin(holdout_donors)]
    if set(targets.index) != holdout_donors:
        raise ValueError(f"fit {fit}: holdout donors are missing from {h5ad_name}")
    if set(train.index) != set(map(str, holdout["train_donors"])):
        raise ValueError(f"fit {fit}: training donors differ from {holdout_path}")
    if target_ages is not None:
        targets = targets[targets["age"].isin(target_ages)]
    groups = []
    for (age, sex), rows in targets.groupby(["age", "sex"], observed=True):
        available = set(train.loc[train["sex"] == sex, "age"].astype(float))
        groups.append({"id": f"age={age:g}|sex={sex}", "target_age": float(age), "match": {"sex": str(sex)},
                       "target_donors": sorted(rows.index.astype(str)), "target_cells": int(rows["cells"].sum()),
                       "source_ages": list(sources), "source_direction": direction,
                       "selected_source_age_count": len(sources), "supported": all(a in available for a in sources)})
    if not all(g["supported"] for g in groups):
        raise RuntimeError(f"fit {fit}: a fixed source age has no training donors of the target's sex")
    return {"schema": "matched_fixed_source_ages", "design": design, "age_key": "age", "donor_key": "donor_id",
            "match_by": ["sex"], "min_target_donors": 1, "min_source_donors": 1,
            "min_source_ages": len(sources), "max_source_ages": len(sources),
            "min_source_age": float(min(sources)), "max_source_age": float(max(sources)), "selection_policy": "spread",
            "fixed_source_ages": list(sources), "source_direction": direction, "seed": 42 + fit, "h5ad": h5ad_name,
            "holdout_manifest": str(holdout_path),
            "target_ages": sorted(float(a) for a in targets["age"].unique()), "groups": groups}


def read_obs(path):
    adata = ad.read_h5ad(path, backed="r")
    try:
        obs = adata.obs[["donor_id", "age", "sex"]].copy()
    finally:
        adata.file.close()
    obs["donor_id"] = obs["donor_id"].astype(str)
    obs["sex"] = obs["sex"].astype(str)
    obs["age"] = pd.to_numeric(obs["age"]).astype(float)
    if (obs.groupby("donor_id", observed=True)[["age", "sex"]].nunique() != 1).any().any():
        raise ValueError(f"Donors must have one age and sex in {path}")
    return obs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--full-h5ad", type=Path, default=Path("data/aida_counts_hvg.h5ad"))
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    args = parser.parse_args()
    obs = read_obs(args.h5ad)
    full_obs = read_obs(args.full_h5ad)
    metadata_dir = args.configs / "benchmarks/metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    tasks = json.loads((args.configs / "aida.json").read_text())["tasks"]
    for fit in range(3):
        holdout = json.loads((args.configs / f"donors/internal_{fit}.json").read_text())
        source_times = next(t["source_times"] for t in tasks if t["arm"] == "flowmap_global" and t["fit"] == fit)
        for name, (design, direction, fixed) in DESIGNS.items():
            sources = [float(a) for a in (fixed or source_times)]
            payload = manifest(obs, fit, holdout, design, direction, sources, args.h5ad.name,
                               args.configs / f"donors/internal_{fit}.json")
            path = metadata_dir / f"fit{fit}_{name}.json"
            path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            print(f"[saved] {path} groups={len(payload['groups'])} sources={sources}")
        report_path = args.configs / f"donors/external_{fit}.json"
        report_holdout = json.loads(report_path.read_text())
        if set(report_holdout["holdout_donors"]) != set(holdout["holdout_donors"]) | set(
                json.loads((args.configs / "aida.json").read_text())["external_samples"]):
            raise ValueError(f"fit {fit}: report split must exclude internal and external holdout donors")
        for name, (design, direction, fixed) in DESIGNS.items():
            sources = [float(a) for a in (fixed or source_times)]
            payload = manifest(full_obs, fit, report_holdout, design, direction, sources,
                               args.full_h5ad.name, report_path, target_ages=report_holdout["external_report_ages"])
            path = metadata_dir / f"fit{fit}_{name.removeprefix('internal_')}.json"
            path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            print(f"[saved] {path} groups={len(payload['groups'])} sources={sources}")


if __name__ == "__main__":
    main()
