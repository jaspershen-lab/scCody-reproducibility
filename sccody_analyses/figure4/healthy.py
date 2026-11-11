"""Compute PsychAD healthy-pair moments at exact observed gaps of 1–15 years."""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from sccody import ScCody
from sccody.decomposition import exact_assignment_columns
from sccody_analyses.extract_residual_fields import stable_rng
from .tables import AD_GROUPS, HEALTHY_COLUMNS


def pair_plan(metadata, holdouts):
    donors = metadata.loc[metadata.donor_id.astype(str).isin(holdouts), ["donor_id", "age", "sex"]].copy()
    donors["donor_id"] = donors.donor_id.astype(str)
    donors = donors.groupby("donor_id", as_index=False, observed=True).first().sort_values(["age", "donor_id"])
    pairs = donors.merge(donors, on="sex", suffixes=("_source", "_target"))
    pairs["age_gap"] = pairs.age_target - pairs.age_source
    pairs = pairs.loc[pairs.age_gap.isin(range(1, 16))]
    return pairs.rename(columns={"donor_id_source": "source_donor", "donor_id_target": "target_donor",
                                 "age_source": "source_age", "age_target": "target_age"})


def pair_moments(model, latent, codes, source, target, source_age, target_age):
    columns = exact_assignment_columns(latent[source], latent[target])
    disease = torch.from_numpy(latent[target[columns]] - latent[source]).float().to(model.device)
    aged = model.transport_latent(latent[source], source_age, target_age,
        cell_type_codes=codes[source] if model.config.model.scope == "cell_type" else None)
    aging = torch.from_numpy(aged - latent[source]).float().to(model.device)
    inner = float((disease * aging).sum(dim=1).mean())
    energy = float(aging.square().sum(dim=1).mean())
    total = float(disease.square().sum(dim=1).mean())
    perpendicular = max(total - inner * inner / energy, 0.) if energy > 1e-12 else np.nan
    return dict(inner_product=inner, aging_energy=energy, perpendicular_energy=perpendicular)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/psychad/ad_case_control.h5ad"))
    parser.add_argument("--holdout", type=Path, default=Path("configs/donors/psychad_healthy_holdout.json"))
    parser.add_argument("--global-model", type=Path, default=Path("checkpoints/PsychAD_scCody-G_fit0.pt"))
    parser.add_argument("--ct-model", type=Path, default=Path("checkpoints/PsychAD_scCody-CT_fit0.pt"))
    parser.add_argument("--outdir", type=Path, default=Path("artifacts/figure4/healthy"))
    parser.add_argument("--groups", nargs="+", default=AD_GROUPS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-ot-cells", type=int, default=256)
    parser.add_argument("--min-cells", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    holdouts = json.loads(args.holdout.read_text())["holdout_donors"]
    backed = ad.read_h5ad(args.input, backed="r")
    selected = np.flatnonzero(backed.obs.donor_id.astype(str).isin(holdouts))
    data = ad.AnnData(backed[selected].X, obs=backed.obs.iloc[selected].copy(), var=backed.var.copy())
    backed.file.close()
    plan = pair_plan(data.obs, holdouts)
    donors = data.obs.donor_id.astype(str).to_numpy()
    types = data.obs["class"].astype(str).to_numpy()
    args.outdir.mkdir(parents=True, exist_ok=True)
    for scope, checkpoint in [("global", args.global_model), ("cell_type", args.ct_model)]:
        groups = [g for g in args.groups if (g == "Global") == (scope == "global")]
        if not groups:
            continue
        model = ScCody.load(checkpoint, device=args.device)
        latent = model.encode(data)
        codes = pd.Categorical(types, categories=model.cell_types).codes
        for group in groups:
            rows = []
            label = "Global population" if group == "Global" else group
            for pair in plan.itertuples():
                source = np.flatnonzero((donors == pair.source_donor) & ((types == group) if scope == "cell_type" else True))
                target = np.flatnonzero((donors == pair.target_donor) & ((types == group) if scope == "cell_type" else True))
                n = min(len(source), len(target), args.max_ot_cells)
                if n < args.min_cells:
                    continue
                shift = f"plus_{int(pair.age_gap)}y"
                rng = stable_rng(args.seed, scope, 0, shift, pair.source_donor, pair.target_donor, label, "ot")
                source = source[np.sort(rng.choice(len(source), n, replace=False))]
                target = target[np.sort(rng.choice(len(target), n, replace=False))]
                moments = pair_moments(model, latent, codes, source, target, pair.source_age, pair.target_age)
                rows.append(dict(scope=scope, group=group, source_donor=pair.source_donor,
                                 target_donor=pair.target_donor, source_age=pair.source_age,
                                 target_age=pair.target_age, age_gap=pair.age_gap, **moments))
            pd.DataFrame(rows, columns=HEALTHY_COLUMNS).to_csv(args.outdir / f"{group}_healthy_pairs.tsv", sep="\t", index=False)
            print(f"[healthy] {group}: {len(rows)} donor pairs", flush=True)


if __name__ == "__main__":
    main()
