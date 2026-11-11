#!/usr/bin/env python3
"""Save per-cell disease, aging and residual displacement fields for one group.

Replays the recorded donor pairs with their sampling seeds (disease mode:
matched control -> case; healthy mode: control -> older control, the
matching-noise reference) and stores, for every OT-matched source / target
cell pair,

    z_source, z_target (matched), aging displacement a_i, pair alpha,

plus the h5ad row indices of both cells, so that decoded expression effects
and observed target-cell expression can be joined back to cell annotations.

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import hashlib

from sccody import ScCody, decompose_displacements
from sccody.decomposition import exact_assignment_columns

def stable_rng(seed: int, *labels: object) -> np.random.Generator:
    token = "\x1f".join([str(seed), *(str(label) for label in labels)]).encode()
    digest = hashlib.sha256(token).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("disease", "healthy"), default="disease")
    parser.add_argument("--shift-label", default="plus_10y")
    parser.add_argument(
        "--target-donor-manifest", type=Path, default=None,
        help="Healthy mode: JSON manifest whose holdout_donors are the retained targets",
    )
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target-h5ad", type=Path, required=True)
    parser.add_argument("--scope", choices=("global", "cell_type"), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--age-key", default="age")
    parser.add_argument("--cell-type-key", default="Annotation_Level1")
    parser.add_argument("--donor-key", default="donor_id")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-ot-cells", type=int, default=512)
    parser.add_argument("--min-cells", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(args.pairs, sep="\t")
    if args.mode == "disease":
        pairs = pairs.loc[pairs["target_kind"].eq("disease")].copy()
    else:
        pairs = pairs.loc[pairs["shift_label"].eq(args.shift_label)].copy()
        pairs["reference_donor"] = pairs["source_donor"]
    if args.target_donor_manifest is not None:
        keep = {str(d) for d in json.loads(args.target_donor_manifest.read_text())["holdout_donors"]}
        pairs = pairs.loc[pairs["target_donor"].astype(str).isin(keep)].copy()
    model = ScCody.load(args.model, device=args.device)
    model.config.training.representation_batch_size = args.batch_size
    if model.config.model.scope != args.scope:
        raise ValueError("Requested scope differs from model")
    data = ad.read_h5ad(args.target_h5ad)
    ages = pd.to_numeric(data.obs[args.age_key], errors="coerce").to_numpy()
    codes = pd.Categorical(data.obs[args.cell_type_key].astype(str), categories=model.cell_types).codes
    valid = np.isfinite(ages)
    if args.scope == "cell_type": valid &= codes >= 0
    rows_all = np.flatnonzero(valid)
    selected = data[valid].copy()
    # Column names must match the explicit native checkpoint data settings.
    donors = selected.obs[args.donor_key].astype(str).to_numpy()
    encoded_groups = selected.obs[args.cell_type_key].astype(str).to_numpy()
    latents = model.encode(selected)
    codes = np.where(codes[valid] >= 0, codes[valid], 0).astype(np.int64)

    for group in args.groups:
        group_pairs = pairs.loc[pairs["group"].eq(group)].reset_index(drop=True)
        meta_rows: list[dict[str, object]] = []
        fields: dict[str, list[np.ndarray]] = {
            key: [] for key in ("pair_id", "source_row", "target_row", "z_source", "z_target", "aging")
        }
        for pair_index, pair in group_pairs.iterrows():
            target_donor = str(pair["target_donor"])
            reference_donor = str(pair["reference_donor"])
            target_mask = donors == target_donor
            source_mask = donors == reference_donor
            if args.scope == "cell_type":
                target_mask &= encoded_groups == group
                source_mask &= encoded_groups == group
            source_index = np.flatnonzero(source_mask)
            target_index = np.flatnonzero(target_mask)
            n_cells = min(len(source_index), len(target_index), args.max_ot_cells)
            if n_cells < args.min_cells:
                continue
            if args.mode == "disease":
                rng = stable_rng(
                    args.seed, args.scope, args.fold, group, pair["target_kind"],
                    target_donor, reference_donor,
                )
            else:
                rng = stable_rng(
                    args.seed, args.scope, args.fold, pair["shift_label"],
                    reference_donor, target_donor, group, "ot",
                )
            source_index = source_index[np.sort(rng.choice(len(source_index), size=n_cells, replace=False))]
            target_index = target_index[np.sort(rng.choice(len(target_index), size=n_cells, replace=False))]
            source_latent = latents[source_index]
            columns = exact_assignment_columns(source_latent, latents[target_index])
            target_index = target_index[columns]
            target_latent = latents[target_index]

            if args.mode == "disease":
                source_age = float(pair["reference_t_hat"])
                transport_age = float(pair.get("transport_target_age_effective", np.nan))
                if not np.isfinite(transport_age):
                    transport_age = float(pair["transport_target_age"])
            else:
                source_age, transport_age = float(pair["source_age"]), float(pair["target_age"])
            aged = model.transport_latent(source_latent, source_age, transport_age,
                                           cell_type_codes=codes[source_index] if args.scope == "cell_type" else None)
            aging = (aged - source_latent).astype(np.float32)
            disease = (target_latent - source_latent).astype(np.float32)
            terms = decompose_displacements(disease, aging)
            summary = terms["summary"]
            total = summary["orthogonal_energy"]
            mean_shift = summary["mean_shift_energy"]
            within = summary["within_donor_energy"]
            pair_metrics = terms["pairs"].iloc[0]
            alpha, inner, aging_energy = pair_metrics["alpha"], pair_metrics["inner_product"], pair_metrics["aging_energy"]

            meta_rows.append({
                "pair_id": len(meta_rows), "pair_index": int(pair_index), "group": group,
                "fold": args.fold, "target_donor": target_donor,
                "reference_donor": reference_donor, "n_cells": int(n_cells),
                "alpha": alpha, "inner_product": inner, "aging_energy": aging_energy,
                "target_age": float(pair["target_age"]), "source_age": source_age,
                "orthogonal_energy": total, "pair_mean_shift_energy": mean_shift,
                "pair_within_energy": within,
            })
            fields["pair_id"].append(np.full(n_cells, len(meta_rows) - 1, dtype=np.int32))
            fields["source_row"].append(rows_all[source_index].astype(np.int64))
            fields["target_row"].append(rows_all[target_index].astype(np.int64))
            fields["z_source"].append(source_latent.astype(np.float32))
            fields["z_target"].append(target_latent.astype(np.float32))
            fields["aging"].append(aging)
            if (pair_index + 1) % 50 == 0 or pair_index + 1 == len(group_pairs):
                print(f"[extract] group={group} {pair_index + 1}/{len(group_pairs)} pairs", flush=True)

        if not meta_rows:
            print(f"[skip] group={group}: no pairs with enough cells", flush=True)
            continue
        meta = pd.DataFrame(meta_rows)
        output_group = ("Global" if group == "Global population" else group) + ("_healthy" if args.mode == "healthy" else "")
        meta.to_csv(args.outdir / f"{output_group}_pairs.tsv", sep="\t", index=False)
        np.savez_compressed(
            args.outdir / f"{output_group}_percell_fields.npz",
            **{key: np.concatenate(value) for key, value in fields.items()},
        )



if __name__ == "__main__":
    main()
