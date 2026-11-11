#!/usr/bin/env python3
"""Orthogonal residual variance of disease and +10 y healthy displacement.

Recipes in ``configs/orthogonal_variance.json`` replay the recorded AD (PsychAD
epoch 5) and COVID (AIDA fits 0-2) tangent-alignment pairs. ``extract``
writes per-cell fields with ``extract_residual_fields.py``; ``decompose`` turns
them into the per-arm artifacts read by ``orthogonal_variance/summarize.py`` and
``orthogonal_variance/panel_f.py``:

    <root>/AD/<group>/{disease,healthy}/
    <root>/COVID/<cohort>/fold<k>/<group>/{disease,healthy}/

For each OT-matched pair the residual is ``disease - alpha * aging``. A pair
contributes its mean residual vector and its within-pair energy; donors average
their pairs with equal weight, and the fold-specific between-pair spread is
added to the donor within term.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def recipe_groups(recipe):
    return ["Global" if g == "Global population" else g for g in recipe["groups"]]


def arm_dir(root, recipe, group):
    if recipe["cohort"] == "AD":
        return root / "AD" / group / recipe["mode"]
    return root / "COVID" / recipe["cohort"] / f"fold{recipe['fold']}" / group / recipe["mode"]


def extract(recipe, device):
    command = [sys.executable, "-m", "sccody_analyses.extract_residual_fields",
               "--mode", recipe["mode"], "--pairs", recipe["pairs"], "--model", recipe["model"],
               "--target-h5ad", recipe["target_h5ad"], "--scope", recipe["scope"],
               "--fold", str(recipe["fold"]), "--groups", *recipe["groups"],
               "--outdir", recipe["outdir"], "--cell-type-key", recipe["cell_type_key"],
               "--max-ot-cells", str(recipe["max_ot_cells"]), "--min-cells", str(recipe["min_cells"]),
               "--seed", str(recipe["seed"]), "--device", device]
    if recipe["shift_label"]:
        command += ["--shift-label", recipe["shift_label"]]
    if recipe["target_donor_manifest"]:
        command += ["--target-donor-manifest", recipe["target_donor_manifest"]]
    subprocess.run(command, check=True)


def decompose_arm(fields, pairs):
    """Pair and donor residual moments from saved per-cell fields."""
    pair_id = fields["pair_id"]
    alpha = pairs.set_index("pair_id").loc[pair_id, "alpha"].to_numpy(float)
    residual = (fields["z_target"].astype(np.float64) - fields["z_source"].astype(np.float64)
                - alpha[:, None] * fields["aging"].astype(np.float64))
    vectors = np.stack([residual[pair_id == p].mean(axis=0) for p in pairs["pair_id"]])
    pair_frame = pairs.copy()
    pair_frame["pair_within_energy"] = [
        float(np.mean(np.sum((residual[pair_id == p] - v) ** 2, axis=1)))
        for p, v in zip(pairs["pair_id"], vectors)]
    pair_frame["pair_mean_shift_energy"] = np.sum(vectors ** 2, axis=1)
    pair_frame["orthogonal_energy"] = pair_frame["pair_mean_shift_energy"] + pair_frame["pair_within_energy"]

    rows, donor_vectors = [], []
    for donor, index in pair_frame.groupby("target_donor", sort=True).groups.items():
        idx = np.asarray(list(index), dtype=int)
        donor_mean = vectors[idx].mean(axis=0)
        within = float(np.mean(pair_frame.loc[idx, "pair_within_energy"].to_numpy(float)
                               + np.sum((vectors[idx] - donor_mean) ** 2, axis=1)))
        shift = float(np.dot(donor_mean, donor_mean))
        rows.append({"group": str(pair_frame.loc[idx[0], "group"]), "target_donor": donor,
                     "n_pair_fields": int(len(idx)), "mean_shift_energy": shift,
                     "within_donor_energy": within, "orthogonal_energy": shift + within})
        donor_vectors.append(donor_mean)
    return pair_frame, pd.DataFrame(rows), vectors, np.stack(donor_vectors)


def decompose(recipe, root):
    fields_dir = Path(recipe["outdir"])
    suffix = "_healthy" if recipe["mode"] == "healthy" else ""
    for group in recipe_groups(recipe):
        stem = fields_dir / f"{group}{suffix}"
        if not Path(f"{stem}_percell_fields.npz").exists():
            print(f"[skip] {stem}: no extracted fields", flush=True)
            continue
        pairs = pd.read_csv(f"{stem}_pairs.tsv", sep="\t")
        pair_frame, donor_frame, pair_vectors, donor_vectors = decompose_arm(
            np.load(f"{stem}_percell_fields.npz"), pairs)
        outdir = arm_dir(root, recipe, group)
        outdir.mkdir(parents=True, exist_ok=True)
        donor_frame.insert(0, "scope", recipe["scope"])
        donor_frame.insert(0, "mode", recipe["mode"])
        pair_frame.to_csv(outdir / "pair_residual_decomposition.tsv", sep="\t", index=False)
        donor_frame.to_csv(outdir / "donor_residual_decomposition.tsv", sep="\t", index=False)
        np.savez_compressed(outdir / "residual_mean_vectors.npz", pair_mean_vectors=pair_vectors,
                            donor_mean_vectors=donor_vectors,
                            pair_index=pair_frame["pair_index"].to_numpy(int))
        print(f"[done] {outdir} donors={len(donor_frame)} pairs={len(pair_frame)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["extract", "decompose"])
    parser.add_argument("--recipes", type=Path, default=Path("configs/orthogonal_variance.json"))
    parser.add_argument("--task", type=int, help="Recipe index; all recipes when omitted")
    parser.add_argument("--root", type=Path, default=Path("artifacts/orthogonal_residual_variance"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    recipes = json.loads(args.recipes.read_text())
    for recipe in recipes if args.task is None else [recipes[args.task]]:
        if args.phase == "extract":
            extract(recipe, args.device)
        else:
            decompose(recipe, args.root)


if __name__ == "__main__":
    main()
