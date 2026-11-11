#!/usr/bin/env python3
"""Build AD/COVID donor-level D/H10 orthogonal-residual tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sccody_analyses.orthogonal_variance.compare import decompose, load_arm


AD_GROUPS = ["Global", "Astro", "EN", "Endo", "IN", "Immune", "Mural", "OPC", "Oligo"]
# Keep the recorded bootstrap stream for each retained population.
COVID_GROUPS = {"Global": 0, "B": 1, "Myeloid": 3, "NK": 4, "Platelet": 5, "T": 6}
COHORTS = ["Ren", "COMBAT", "Stephenson"]
COHORT_GROUPS = {
    "Ren": COVID_GROUPS,
    "COMBAT": {"Global": 0, "B": 1, "Myeloid": 3, "NK": 4, "T": 5},
    "Stephenson": COVID_GROUPS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ad-root", type=Path, required=True)
    parser.add_argument("--covid-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def compare_arrays(
    d_vectors: np.ndarray,
    d_within: np.ndarray,
    h_vectors: np.ndarray,
    h_within: np.ndarray,
    *,
    dataset: str,
    scope: str,
    group: str,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    d = decompose(d_vectors, d_within)
    h = decompose(h_vectors, h_within)
    row: dict[str, object] = {
        "dataset": dataset, "scope": scope, "group": group,
        "n_disease_donors": len(d_vectors), "n_healthy_donors": len(h_vectors),
    }
    for key in d:
        row[f"D_{key}"] = d[key]
        row[f"H10_{key}"] = h[key]
        if key.endswith("_energy") or key.endswith("_rms"):
            row[f"D_H10_ratio_{key}"] = d[key] / h[key] if h[key] > 0 else np.nan
            row[f"D_H10_percent_change_{key}"] = (
                100.0 * (d[key] / h[key] - 1.0) if h[key] > 0 else np.nan
            )
        if key.endswith("_fraction"):
            row[f"D_minus_H10_{key}"] = d[key] - h[key]

    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for key in d:
        if key.endswith("_energy") or key.endswith("_rms"):
            draws[f"ratio_{key}"] = []
        elif key.endswith("_fraction"):
            draws[f"difference_{key}"] = []
    for _ in range(repeats):
        d_idx = rng.integers(0, len(d_vectors), size=len(d_vectors))
        h_idx = rng.integers(0, len(h_vectors), size=len(h_vectors))
        d_boot = decompose(d_vectors[d_idx], d_within[d_idx])
        h_boot = decompose(h_vectors[h_idx], h_within[h_idx])
        for key in d:
            if key.endswith("_energy") or key.endswith("_rms"):
                draws[f"ratio_{key}"].append(
                    d_boot[key] / h_boot[key] if h_boot[key] > 0 else np.nan
                )
            elif key.endswith("_fraction"):
                draws[f"difference_{key}"].append(d_boot[key] - h_boot[key])
    for key, values in draws.items():
        low, median, high = np.nanpercentile(values, [2.5, 50, 97.5])
        row[f"{key}_ci_low"] = low
        row[f"{key}_bootstrap_median"] = median
        row[f"{key}_ci_high"] = high
    return row


def aggregate_decompositions(parts: list[dict[str, float]]) -> dict[str, float]:
    """Average decomposition energies across cohorts, then rederive fractions."""
    mean_shift = float(np.mean([part["mean_shift_energy"] for part in parts]))
    between = float(np.mean([part["between_donor_energy"] for part in parts]))
    within = float(np.mean([part["within_donor_energy"] for part in parts]))
    orthogonal = mean_shift + between + within
    return {
        "mean_shift_energy": mean_shift,
        "between_donor_energy": between,
        "within_donor_energy": within,
        "orthogonal_energy": orthogonal,
        "mean_shift_fraction": mean_shift / orthogonal if orthogonal > 0 else np.nan,
        "between_donor_fraction": between / orthogonal if orthogonal > 0 else np.nan,
        "within_donor_fraction": within / orthogonal if orthogonal > 0 else np.nan,
        "orthogonal_rms": float(np.sqrt(max(orthogonal, 0.0))),
    }


def compare_cohort_balanced(
    d_arms: dict[str, tuple[np.ndarray, np.ndarray]],
    h_arms: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    dataset: str,
    scope: str,
    group: str,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    """Compare D and H10 after equal-weight, within-cohort decomposition.

    Each cohort contributes one decomposition.  Donor bootstrap draws are made
    independently within each cohort and the resulting component energies are
    averaged equally across cohorts.  This avoids treating cohort shifts as
    donor variation.
    """
    cohorts = list(d_arms)
    if set(cohorts) != set(h_arms):
        raise ValueError("Disease and healthy cohort sets do not match")
    d_parts = [decompose(*d_arms[cohort]) for cohort in cohorts]
    h_parts = [decompose(*h_arms[cohort]) for cohort in cohorts]
    disease = aggregate_decompositions(d_parts)
    healthy = aggregate_decompositions(h_parts)
    row: dict[str, object] = {
        "dataset": dataset, "scope": scope, "group": group,
        "n_disease_donors": int(sum(len(d_arms[c][0]) for c in cohorts)),
        "n_healthy_donors": int(sum(len(h_arms[c][0]) for c in cohorts)),
        "cohorts_used": ";".join(cohorts),
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

    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for key in disease:
        if key.endswith("_energy") or key.endswith("_rms"):
            draws[f"ratio_{key}"] = []
        elif key.endswith("_fraction"):
            draws[f"difference_{key}"] = []
    for _ in range(repeats):
        d_boot_parts = []
        h_boot_parts = []
        for cohort in cohorts:
            d_vectors, d_within = d_arms[cohort]
            h_vectors, h_within = h_arms[cohort]
            d_idx = rng.integers(0, len(d_vectors), size=len(d_vectors))
            h_idx = rng.integers(0, len(h_vectors), size=len(h_vectors))
            d_boot_parts.append(decompose(d_vectors[d_idx], d_within[d_idx]))
            h_boot_parts.append(decompose(h_vectors[h_idx], h_within[h_idx]))
        d_boot = aggregate_decompositions(d_boot_parts)
        h_boot = aggregate_decompositions(h_boot_parts)
        for key in disease:
            if key.endswith("_energy") or key.endswith("_rms"):
                draws[f"ratio_{key}"].append(
                    d_boot[key] / h_boot[key] if h_boot[key] > 0 else np.nan
                )
            elif key.endswith("_fraction"):
                draws[f"difference_{key}"].append(d_boot[key] - h_boot[key])
    for key, values in draws.items():
        low, median, high = np.nanpercentile(values, [2.5, 50, 97.5])
        row[f"{key}_ci_low"] = low
        row[f"{key}_bootstrap_median"] = median
        row[f"{key}_ci_high"] = high
    return row


def arm_arrays(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    vectors, within, _ = load_arm(paths)
    return vectors, within


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    def add_comparison(
        dataset: str, scope: str, group: str,
        disease_dirs: list[Path], healthy_dirs: list[Path], seed_offset: int,
    ) -> None:
        needed = [*disease_dirs, *healthy_dirs]
        missing = [str(path) for path in needed if not (
            (path / "donor_residual_decomposition.tsv").exists()
            and (path / "residual_mean_vectors.npz").exists()
        )]
        if missing:
            print(f"[skip] {dataset}/{group}: missing {missing[0]}")
            return
        dv, dw = arm_arrays(disease_dirs)
        hv, hw = arm_arrays(healthy_dirs)
        rows.append(compare_arrays(
            dv, dw, hv, hw, dataset=dataset, scope=scope, group=group,
            repeats=args.bootstrap_repeats, seed=args.seed + seed_offset,
        ))

    def add_cohort_balanced_comparison(
        dataset: str, scope: str, group: str,
        disease_dirs_by_cohort: dict[str, list[Path]],
        healthy_dirs_by_cohort: dict[str, list[Path]], seed_offset: int,
    ) -> None:
        needed = [
            path
            for cohort in disease_dirs_by_cohort
            for path in [*disease_dirs_by_cohort[cohort], *healthy_dirs_by_cohort[cohort]]
        ]
        missing = [str(path) for path in needed if not (
            (path / "donor_residual_decomposition.tsv").exists()
            and (path / "residual_mean_vectors.npz").exists()
        )]
        if missing:
            print(f"[skip] {dataset}/{group}: missing {missing[0]}")
            return
        d_arms = {
            cohort: arm_arrays(disease_dirs_by_cohort[cohort])
            for cohort in disease_dirs_by_cohort
        }
        h_arms = {
            cohort: arm_arrays(healthy_dirs_by_cohort[cohort])
            for cohort in healthy_dirs_by_cohort
        }
        rows.append(compare_cohort_balanced(
            d_arms, h_arms, dataset=dataset, scope=scope, group=group,
            repeats=args.bootstrap_repeats, seed=args.seed + seed_offset,
        ))

    for index, group in enumerate(AD_GROUPS):
        d = args.ad_root / group / "disease"
        h = args.ad_root / group / "healthy"
        add_comparison("AD", "global" if group == "Global" else "cell_type", group, [d], [h], index)

    for cohort_index, cohort in enumerate(COHORTS):
        for group, group_index in COHORT_GROUPS[cohort].items():
            d_dirs = [args.covid_root / cohort / f"fold{fold}" / group / "disease" for fold in range(3)]
            h_dirs = [args.covid_root / cohort / f"fold{fold}" / group / "healthy" for fold in range(3)]
            add_comparison(
                cohort, "global" if group == "Global" else "cell_type", group,
                d_dirs, h_dirs, 1000 + cohort_index * 100 + group_index,
            )

    for group, group_index in COVID_GROUPS.items():
        pooled_cohorts = [cohort for cohort in COHORTS if group in COHORT_GROUPS[cohort]]
        d_dirs_by_cohort = {
            cohort: [
                args.covid_root / cohort / f"fold{fold}" / group / "disease"
                for fold in range(3)
            ]
            for cohort in pooled_cohorts
        }
        h_dirs_by_cohort = {
            cohort: [
                args.covid_root / cohort / f"fold{fold}" / group / "healthy"
                for fold in range(3)
            ]
            for cohort in pooled_cohorts
        }
        add_cohort_balanced_comparison(
            "COVID_cohort_balanced", "global" if group == "Global" else "cell_type", group,
            d_dirs_by_cohort, h_dirs_by_cohort, 2000 + group_index,
        )

    result = pd.DataFrame(rows)
    result.to_csv(args.outdir / "orthogonal_residual_variance_D_H10_all.csv", index=False)


if __name__ == "__main__":
    main()
