#!/usr/bin/env python3
"""Build Figure 4 panel-f data from donor decomposition artifacts.

For component k, the plotted value is

    100 * (D_k - H10_k) / H10_total.

Bootstrap intervals resample target donors. Disease and healthy draws are
independent, but their numerator and denominator are evaluated in the same
bootstrap replicate. COVID is decomposed within cohort before equal-weight
averaging across cohorts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sccody_analyses.orthogonal_variance.compare import decompose, load_arm
from sccody_analyses.orthogonal_variance.summarize import COHORT_GROUPS


AD_ORDER = ["Global", "OPC", "Astro", "EN", "Oligo", "Endo", "Immune"]
COVID_ORDER = ["Global", "B", "Myeloid", "NK", "Platelet", "T"]
COHORTS = ["Ren", "COMBAT", "Stephenson"]
COMPONENTS = [
    ("shared", "mean_shift_energy", "Shared shift"),
    ("between", "between_donor_energy", "Between donors"),
    ("within", "within_donor_energy", "Within donors"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ad-root", type=Path, required=True)
    parser.add_argument("--covid-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def available_inputs(paths: list[Path]) -> bool:
    missing = [
        path for path in paths
        if not (path / "donor_residual_decomposition.tsv").exists()
        or not (path / "residual_mean_vectors.npz").exists()
    ]
    if missing:
        print(f"[skip] Missing decomposition artifacts: {missing[0]}")
    return not missing


def aggregate(parts: list[dict[str, float]]) -> dict[str, float]:
    """Equal-weight average of cohort component energies."""
    mean_shift = float(np.mean([part["mean_shift_energy"] for part in parts]))
    between = float(np.mean([part["between_donor_energy"] for part in parts]))
    within = float(np.mean([part["within_donor_energy"] for part in parts]))
    return {
        "mean_shift_energy": mean_shift,
        "between_donor_energy": between,
        "within_donor_energy": within,
        "orthogonal_energy": mean_shift + between + within,
    }


def draw_one(
    vectors: np.ndarray,
    within: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    index = rng.integers(0, len(vectors), size=len(vectors))
    return decompose(vectors[index], within[index])


def calculate_row(
    *,
    disease_arms: dict[str, tuple[np.ndarray, np.ndarray]],
    healthy_arms: dict[str, tuple[np.ndarray, np.ndarray]],
    disease: str,
    cell_type: str,
    cohorts: list[str],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    disease_point = aggregate([
        decompose(*disease_arms[cohort]) for cohort in cohorts
    ])
    healthy_point = aggregate([
        decompose(*healthy_arms[cohort]) for cohort in cohorts
    ])

    rng = np.random.default_rng(seed)
    draws = {name: [] for name, _, _ in COMPONENTS}
    for _ in range(repeats):
        d = aggregate([
            draw_one(*disease_arms[cohort], rng) for cohort in cohorts
        ])
        h = aggregate([
            draw_one(*healthy_arms[cohort], rng) for cohort in cohorts
        ])
        values = {
            name: 100.0 * (d[key] - h[key]) / h["orthogonal_energy"]
            for name, key, _ in COMPONENTS
        }
        for name in draws:
            draws[name].append(values[name])

    point_values = {
        name: 100.0 * (disease_point[key] - healthy_point[key])
        / healthy_point["orthogonal_energy"]
        for name, key, _ in COMPONENTS
    }

    rows = []
    n_disease = int(sum(len(disease_arms[c][0]) for c in cohorts))
    n_healthy = int(sum(len(healthy_arms[c][0]) for c in cohorts))
    for name, key, label in COMPONENTS:
        low, median, high = np.percentile(draws[name], [2.5, 50.0, 97.5])
        rows.append({
            "dataset": disease,
            "cell_type": cell_type,
            "component": name,
            "component_label": label,
            "estimate_percent": point_values[name],
            "ci_low_percent": low,
            "bootstrap_median_percent": median,
            "ci_high_percent": high,
            "D_energy": disease_point[key],
            "H10_energy": healthy_point[key],
            "H10_total_energy": healthy_point["orthogonal_energy"],
            "n_disease_donors": n_disease,
            "n_healthy_donors": n_healthy,
            "cohorts_used": ";".join(cohorts),
            "bootstrap_repeats": repeats,
            "bootstrap_seed": seed,
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    plotted = []
    for disease, order in (("AD", AD_ORDER), ("COVID", COVID_ORDER)):
        for group_index, group in enumerate(order):
            if disease == "AD":
                d_dir = args.ad_root / group / "disease"
                h_dir = args.ad_root / group / "healthy"
                if not available_inputs([d_dir, h_dir]):
                    continue
                disease_arms = {"AD": load_arm([d_dir])[:2]}
                healthy_arms = {"AD": load_arm([h_dir])[:2]}
                cohorts = ["AD"]
            else:
                cohorts = [
                    cohort for cohort in COHORTS
                    if group in COHORT_GROUPS[cohort]
                ]
                disease_arms = {}
                healthy_arms = {}
                paths = {
                    cohort: {
                        arm: [args.covid_root / cohort / f"fold{fold}" / group / arm for fold in range(3)]
                        for arm in ("disease", "healthy")
                    }
                    for cohort in cohorts
                }
                if not available_inputs([path for arms in paths.values() for arm in arms.values() for path in arm]):
                    continue
                for cohort in cohorts:
                    disease_arms[cohort] = load_arm(paths[cohort]["disease"])[:2]
                    healthy_arms[cohort] = load_arm(paths[cohort]["healthy"])[:2]

            frame = calculate_row(
                disease=disease,
                cell_type=group,
                disease_arms=disease_arms,
                healthy_arms=healthy_arms,
                cohorts=cohorts,
                repeats=args.bootstrap_repeats,
                seed=args.seed + (0 if disease == "AD" else 1000) + group_index,
            )
            plotted.append(frame)

    data = pd.concat(plotted, ignore_index=True) if plotted else pd.DataFrame()

    data.to_csv(args.outdir / "orthogonal_residual_variance_panel_f_data.csv", index=False)
    print(data.reindex(columns=["dataset", "cell_type", "component", "estimate_percent", "ci_low_percent", "ci_high_percent"]).to_string(index=False))


if __name__ == "__main__":
    main()
