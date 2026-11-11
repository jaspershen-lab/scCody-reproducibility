"""Localize disease and healthy-reference donors using the existing study weights."""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from sccody import ScCody
from sccody_analyses.preprocessing.prepare_covid import output_path
from sccody_analyses.orthogonal_variance.summarize import COHORT_GROUPS
from .tables import AD_GROUPS, COHORTS


def read_cells(path, model, donors=None):
    obj = ad.read_h5ad(path, backed="r")
    rows = np.arange(obj.n_obs) if donors is None else np.flatnonzero(obj.obs.donor_id.astype(str).isin(donors))
    result = ad.AnnData(obj[rows].X, obs=obj.obs.iloc[rows].copy(), var=obj.var.copy())
    obj.file.close()
    return result


def localize(model, target, *, source_times, grid, scope, cohort, fold):
    result = model.localize(target, grid, source_times=source_times, population_scope=scope,
                            min_cells=20, max_cells=2000, balanced_cells=True).predictions
    d = model.config.data
    result = result.rename(columns={d.sample_key: "donor_id", d.cell_type_key: "population", "predicted_time": "pred_age"})
    if scope == "global":
        result["population"] = "Global"
    meta = target.obs[["donor_id", "age", "condition"]].groupby("donor_id", as_index=False, observed=True).first()
    return result[["donor_id", "population", "pred_age"]].merge(meta, on="donor_id").assign(
        cohort=cohort, scope=scope, fold=fold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=["AD", *COHORTS], required=True)
    parser.add_argument("--scope", choices=["global", "cell_type"], required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--study", type=Path, default=Path("configs/aida.json"))
    parser.add_argument("--covid-input-dir", type=Path, default=Path("data/covid"))
    parser.add_argument("--ad-input", type=Path, default=Path("data/psychad/ad_case_control.h5ad"))
    parser.add_argument("--ad-model-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--ad-sources", type=Path, default=Path("configs/figure4/psychad_sources.json"))
    parser.add_argument("--ad-holdout", type=Path, default=Path("configs/donors/psychad_healthy_holdout.json"))
    parser.add_argument("--outdir", type=Path, default=Path("artifacts/figure4/projections"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.cohort == "AD":
        variant = "G" if args.scope == "global" else "CT"
        model = ScCody.load(args.ad_model_dir / f"PsychAD_scCody-{variant}_fit0.pt", device=args.device)
        input_path = reference_path = args.ad_input
        sources = json.loads(args.ad_sources.read_text())
        source_times = sorted({age for spec in sources.values() for ages in spec.values() for age in ages})
        grid = range(19, 90)
    else:
        study = json.loads(args.study.read_text())
        task = next(t for t in study["tasks"] if t["arm"] == "flowmap_" + args.scope and t["fit"] == args.fold)
        model = ScCody.load(task["canonical_checkpoint"], device=args.device)
        input_path = output_path(args.covid_input_dir, COHORTS[args.cohort])
        reference_path, source_times, grid = study["train_input"], task["source_times"], task["candidate_times"]
    training = model.split_table.loc[model.split_table.split.eq("train"), model.config.data.sample_key].astype(str)
    reference = read_cells(reference_path, model, training)
    reference = reference[reference.obs.age.isin(source_times)].copy()
    model.set_reference(reference)
    del reference
    target = read_cells(input_path, model)
    if args.cohort == "AD":
        holdouts = json.loads(args.ad_holdout.read_text())["holdout_donors"]
        target = target[target.obs.condition.eq("case") | target.obs.donor_id.astype(str).isin(holdouts)].copy()
        frames = []
        for sex, spec in sources.items():
            groups = ["Global"] if args.scope == "global" else AD_GROUPS[1:]
            for group in groups:
                mask = target.obs.sex.astype(str).eq(sex)
                if args.scope == "cell_type":
                    mask &= target.obs[model.config.data.cell_type_key].astype(str).eq(group)
                if not mask.any():
                    continue
                frames.append(localize(model, target[mask].copy(), source_times=spec.get(group, spec["default"]),
                    grid=grid, scope=args.scope, cohort="PsychAD_AD", fold=0))
        result = pd.concat(frames, ignore_index=True)
    else:
        if args.scope == "cell_type":
            target = target[target.obs[model.config.data.cell_type_key].astype(str).isin(COHORT_GROUPS[args.cohort])].copy()
        result = localize(model, target, source_times=source_times, grid=grid, scope=args.scope,
                          cohort="COVID_" + args.cohort, fold=args.fold)
    args.outdir.mkdir(parents=True, exist_ok=True)
    output = args.outdir / f"{args.cohort}_{args.scope}_fit{args.fold}.csv"
    result.to_csv(output, index=False)
    print(f"[project] {output}: {len(result)} donor populations", flush=True)


if __name__ == "__main__":
    main()
