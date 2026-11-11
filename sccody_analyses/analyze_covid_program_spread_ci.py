#!/usr/bin/env python3
"""Disease/healthy ratio of within-patient spread of decoded program shifts, with CIs.

For each cell, ``dec_within__<module>`` is the decoded within-donor residual
step averaged over the module genes (``decode_residual_components.py``).  The
within-patient spread of a module is the per-donor SD of that value.  Donor
SDs are averaged over the folds in which the donor is a target, then

    ratio = mean_disease_donors(SD) / mean_healthy_donors(SD)

with an independent donor bootstrap per mode and a donor-label permutation
p-value (one-sided, ratio > 1).

Output: program_spread_ratio_ci.tsv (one row per cohort x group x module).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODULES = ["isg_core", "ifn_alpha", "ifn_gamma", "inflammatory_response", "inflammatory_cytokine",
           "tnfa_nfkb", "il6_jak_stat3", "mhc_class_ii", "alarmin_emergency", "cd16_identity"]


def donor_sd(decoded_root: Path, cohort: str, group: str, healthy: bool, min_cells: int) -> pd.DataFrame:
    parts = []
    for fold_dir in sorted((decoded_root / cohort).glob("fold*")):
        path = fold_dir / (group + ("_healthy" if healthy else "")) / "percell_components.tsv.gz"
        if not path.exists():
            continue
        cols = ["target_donor"] + [f"dec_within__{m}" for m in MODULES]
        frame = pd.read_csv(path, sep="\t", usecols=lambda c: c in cols)
        counts = frame.groupby("target_donor").size()
        frame = frame[frame["target_donor"].isin(counts.index[counts >= min_cells])]
        sd = frame.groupby("target_donor").std()
        sd.columns = [c.replace("dec_within__", "") for c in sd.columns]
        parts.append(sd.assign(fold=fold_dir.name))
    return pd.concat(parts).groupby(level=0).mean(numeric_only=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoded-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--min-cells", type=int, default=30)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows = []
    for cohort in ("Ren", "COMBAT", "Stephenson"):
        for group in ("Myeloid", "DC"):
            dis = donor_sd(args.decoded_root, cohort, group, False, args.min_cells)
            hea = donor_sd(args.decoded_root, cohort, group, True, args.min_cells)
            for module in MODULES:
                if module not in dis or module not in hea:
                    continue
                d, h = dis[module].dropna().to_numpy(), hea[module].dropna().to_numpy()
                ratio = d.mean() / h.mean()
                boot = np.asarray([d[rng.integers(0, len(d), len(d))].mean() / h[rng.integers(0, len(h), len(h))].mean()
                                   for _ in range(args.bootstrap)])
                pooled = np.concatenate([d, h])
                perm = np.empty(args.permutations)
                for i in range(args.permutations):
                    idx = rng.permutation(len(pooled))
                    perm[i] = pooled[idx[:len(d)]].mean() / pooled[idx[len(d):]].mean()
                rows.append({
                    "cohort": cohort, "group": group, "module": module,
                    "n_disease_donors": len(d), "n_healthy_donors": len(h),
                    "disease_mean_sd": d.mean(), "healthy_mean_sd": h.mean(), "ratio": ratio,
                    "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)),
                    "perm_p_greater": float((np.sum(perm >= ratio) + 1) / (args.permutations + 1)),
                })
            print(f"[done] {cohort} {group}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.outdir / "program_spread_ratio_ci.tsv", sep="\t", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
