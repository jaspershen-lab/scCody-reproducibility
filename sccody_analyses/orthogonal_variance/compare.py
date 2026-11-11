#!/usr/bin/env python3
"""Compare donor-level orthogonal-residual decompositions D versus H10."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--healthy-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def load_arm(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if len(paths) == 0:
        raise ValueError("At least one arm directory is required")
    vectors = []
    within = []
    donor_frames = []
    for path in paths:
        donor_path = path / "donor_residual_decomposition.tsv"
        vector_path = path / "residual_mean_vectors.npz"
        if not donor_path.exists() or not vector_path.exists():
            raise FileNotFoundError(f"Missing decomposition artifacts under {path}")
        donor = pd.read_csv(donor_path, sep="\t")
        saved = np.load(vector_path)
        donor_vectors = np.asarray(saved["donor_mean_vectors"], dtype=float)
        if len(donor) != len(donor_vectors):
            raise ValueError(f"Donor/vector length mismatch under {path}")
        vectors.append(donor_vectors)
        within.append(donor["within_donor_energy"].to_numpy(float))
        cohort_tag = path.parts[-4] if len(path.parts) >= 4 else str(path.parent)
        donor_frames.append(donor.assign(cohort_tag=cohort_tag))

    # A donor may be represented once per CV fold.  Collapse those fold
    # replicas before the donor bootstrap.  The fold-level within term is
    # augmented by the variance of the fold donor means, preserving the exact
    # variance identity after collapse.
    all_donors = pd.concat(donor_frames, ignore_index=True)
    all_vectors = np.vstack(vectors)
    all_within = np.concatenate(within)
    keys = all_donors["cohort_tag"].astype(str) + "::" + all_donors["target_donor"].astype(str)
    collapsed_vectors = []
    collapsed_within = []
    collapsed_rows = []
    for key, indices in keys.groupby(keys).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        donor_mean = all_vectors[idx].mean(axis=0)
        donor_within = float(np.mean(
            all_within[idx] + np.sum((all_vectors[idx] - donor_mean) ** 2, axis=1)
        ))
        row = all_donors.iloc[idx[0]].copy()
        row["target_donor"] = str(key)
        row["n_pair_fields"] = int(all_donors.iloc[idx]["n_pair_fields"].sum())
        row["n_fold_replicates"] = int(len(idx))
        row["within_donor_energy"] = donor_within
        row["orthogonal_energy"] = float(np.dot(donor_mean, donor_mean) + donor_within)
        collapsed_rows.append(row)
        collapsed_vectors.append(donor_mean)
        collapsed_within.append(donor_within)
    return (
        np.vstack(collapsed_vectors), np.asarray(collapsed_within, dtype=float),
        pd.DataFrame(collapsed_rows).reset_index(drop=True),
    )


def decompose(vectors: np.ndarray, within: np.ndarray) -> dict[str, float]:
    grand_mean = vectors.mean(axis=0)
    mean_shift = float(np.dot(grand_mean, grand_mean))
    between = float(np.mean(np.sum((vectors - grand_mean) ** 2, axis=1)))
    within_energy = float(np.mean(within))
    orthogonal = mean_shift + between + within_energy
    return {
        "mean_shift_energy": mean_shift,
        "between_donor_energy": between,
        "within_donor_energy": within_energy,
        "orthogonal_energy": orthogonal,
        "mean_shift_fraction": mean_shift / orthogonal if orthogonal > 0 else np.nan,
        "between_donor_fraction": between / orthogonal if orthogonal > 0 else np.nan,
        "within_donor_fraction": within_energy / orthogonal if orthogonal > 0 else np.nan,
        "orthogonal_rms": float(np.sqrt(max(orthogonal, 0.0))),
    }


def main() -> None:
    args = parse_args()
    if len(args.disease_dirs) != len(args.healthy_dirs):
        raise ValueError("--disease-dirs and --healthy-dirs must have equal lengths")
    args.outdir.mkdir(parents=True, exist_ok=True)
    disease_vectors, disease_within, disease_donors = load_arm(args.disease_dirs)
    healthy_vectors, healthy_within, healthy_donors = load_arm(args.healthy_dirs)
    disease = decompose(disease_vectors, disease_within)
    healthy = decompose(healthy_vectors, healthy_within)
    row: dict[str, object] = {
        "dataset": args.dataset, "scope": args.scope, "group": args.group,
        "n_disease_donors": int(len(disease_vectors)),
        "n_healthy_donors": int(len(healthy_vectors)),
    }
    for key in disease:
        row[f"D_{key}"] = disease[key]
        row[f"H10_{key}"] = healthy[key]
        if key.endswith("_energy") or key.endswith("_rms"):
            row[f"D_H10_ratio_{key}"] = disease[key] / healthy[key] if healthy[key] > 0 else np.nan
            row[f"D_H10_percent_change_{key}"] = (
                100.0 * (disease[key] / healthy[key] - 1.0)
                if healthy[key] > 0 else np.nan
            )
        if key.endswith("_fraction"):
            row[f"D_minus_H10_{key}"] = disease[key] - healthy[key]

    rng = np.random.default_rng(args.seed)
    bootstrap: dict[str, list[float]] = {}
    for key in disease:
        if key.endswith("_energy") or key.endswith("_rms"):
            bootstrap[f"ratio_{key}"] = []
        elif key.endswith("_fraction"):
            bootstrap[f"difference_{key}"] = []
    for _ in range(args.bootstrap_repeats):
        d_idx = rng.integers(0, len(disease_vectors), size=len(disease_vectors))
        h_idx = rng.integers(0, len(healthy_vectors), size=len(healthy_vectors))
        d = decompose(disease_vectors[d_idx], disease_within[d_idx])
        h = decompose(healthy_vectors[h_idx], healthy_within[h_idx])
        for key in disease:
            if key.endswith("_energy") or key.endswith("_rms"):
                bootstrap[f"ratio_{key}"].append(d[key] / h[key] if h[key] > 0 else np.nan)
            elif key.endswith("_fraction"):
                bootstrap[f"difference_{key}"].append(d[key] - h[key])
    for key, values in bootstrap.items():
        low, median, high = np.nanpercentile(values, [2.5, 50, 97.5])
        row[f"{key}_ci_low"] = low
        row[f"{key}_bootstrap_median"] = median
        row[f"{key}_ci_high"] = high

    pd.DataFrame([row]).to_csv(args.outdir / "orthogonal_residual_D_H10_comparison.tsv", sep="\t", index=False)
    donor = pd.concat([
        disease_donors.assign(arm="D"), healthy_donors.assign(arm="H10"),
    ], ignore_index=True)
    donor.to_csv(args.outdir / "orthogonal_residual_D_H10_donors.tsv", sep="\t", index=False)
    print(pd.DataFrame([row]).T.to_string(header=False))


if __name__ == "__main__":
    main()
