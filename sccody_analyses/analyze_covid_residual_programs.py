#!/usr/bin/env python3
"""Cellular programs and subpopulations behind within-patient residual variation (COVID).

Uses ``decode_residual_components.py`` per-cell output for one cohort and group
(all folds), in two modes that went through identical coarse-group OT:

  disease   matched control (source) -> COVID donor (target)
  healthy   control (source) -> older control (+10y; target); the reference for
            OT matching noise and normal heterogeneity

Within-patient residual energy per cell is ||r_i - r_D||^2 (latent).  Fine
labels are attached to cells after matching; they never constrain OT.
Observed module scores use measured expression of the matched cells (genes
mean-imputed in any cohort are excluded), standardised per gene over all
scored cells of both modes.

Outputs (folds kept separate)
  module_association.tsv       per-donor Spearman(within_norm, module score) summarised
                               across donors, vs a size-matched random-gene null; energy
                               share of the module-high quartile; both modes
  decoder_module_check.tsv     per-cell agreement of decoded vs observed module shifts
  subpopulation_contribution.tsv
                               per-label within-energy ratio and cell/energy share; both modes
  subpopulation_excess.tsv     per-label disease - healthy within energy and the label's
                               share of the total disease excess (independent donor bootstrap)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

LABEL_KEYS = {"Ren": "celltype", "COMBAT": "Annotation_major_subset", "Stephenson": "initial_clustering"}
MODES = ("disease", "healthy")


def harmonise(cohort: str, label: str) -> str:
    if cohort == "Ren":
        if label.startswith("Mono_c4"):
            return "Intermediate_Mono"
        if label.startswith("Mono_c5"):
            return "CD16_Mono"
        if label.startswith("Mono_"):
            return "CD14_Mono"
        if label.startswith("Macro"):
            return "Macrophage"
        if label.startswith("DC_c4"):
            return "pDC"
        if label.startswith("DC_"):
            return "cDC"
    if cohort == "COMBAT":
        return {"cMono": "CD14_Mono", "ncMono": "CD16_Mono", "DC": "DC_all"}.get(label, label)
    if cohort == "Stephenson":
        return {"CD14": "CD14_Mono", "CD16": "CD16_Mono", "Mono_prolif": "Prolif_Mono",
                "DCs": "cDC", "pDC": "pDC"}.get(label, label)
    return label


def spearman_by_donor(donor_index: dict, values: np.ndarray, y: np.ndarray, min_cells: int = 30) -> np.ndarray:
    return np.asarray([stats.spearmanr(values[idx], y[idx]).statistic
                       for idx in donor_index.values() if len(idx) >= min_cells])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, choices=sorted(LABEL_KEYS))
    parser.add_argument("--group", required=True)
    parser.add_argument("--decoded-root", type=Path, required=True, help="decoded/<cohort>")
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--modules", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--null-draws", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--min-label-cells", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    panel = pd.read_csv(args.panel, sep="\t")
    observed_cols = np.flatnonzero(panel["observed"].to_numpy(bool))
    modules = {k: np.asarray(v) for k, v in json.loads(args.modules.read_text()).items()}
    position = {g: i for i, g in enumerate(observed_cols)}
    module_pos = {k: np.asarray([position[g] for g in v]) for k, v in modules.items()}

    cells: dict[tuple[str, str], pd.DataFrame] = {}
    for fold_dir in sorted(args.decoded_root.glob("fold*")):
        for mode in MODES:
            path = fold_dir / (args.group + ("_healthy" if mode == "healthy" else "")) / "percell_components.tsv.gz"
            if path.exists():
                cells[(mode, fold_dir.name)] = pd.read_csv(path, sep="\t")
    print(f"[load] {args.cohort} {args.group}: " + ", ".join(f"{m}/{f}={len(c)}" for (m, f), c in cells.items()))

    rows = np.unique(np.concatenate([np.concatenate([c["source_row"], c["target_row"]]) for c in cells.values()]))
    obj = ad.read_h5ad(args.h5ad, backed="r")
    labels = obj.obs[LABEL_KEYS[args.cohort]].astype(str).to_numpy()
    blocks = []
    for start in range(0, len(rows), 50000):
        block = obj.X[rows[start:start + 50000]]
        block = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
        blocks.append(block[:, observed_cols].astype(np.float32))
    obj.file.close()
    expr = np.concatenate(blocks)
    mu, sd = expr.mean(axis=0), expr.std(axis=0)
    usable = sd > 1e-6
    Z = np.zeros_like(expr)
    Z[:, usable] = (expr[:, usable] - mu[usable]) / sd[usable]
    del expr
    row_pos = pd.Series(np.arange(len(rows)), index=rows)
    usable_pos = np.flatnonzero(usable)
    print(f"[expr] {len(rows)} cells x {len(observed_cols)} observed genes")

    def score(index: np.ndarray, which: np.ndarray) -> np.ndarray:
        return Z[np.ix_(which, index)].mean(axis=1)

    assoc_rows, decoder_rows, contrib_rows = [], [], []
    per_donor_tables: dict[tuple[str, str], pd.DataFrame] = {}
    for (mode, fold), frame in cells.items():
        t_idx = row_pos.loc[frame["target_row"]].to_numpy()
        s_idx = row_pos.loc[frame["source_row"]].to_numpy()
        energy = frame["within_sq"].to_numpy()
        within_norm = frame["within_norm"].to_numpy()
        donor_index = frame.groupby("target_donor").indices
        for name, index in module_pos.items():
            target_score = score(index, t_idx)
            delta_score = target_score - score(index, s_idx)
            for kind, y in (("target_score", target_score), ("target_minus_source", delta_score)):
                rho = spearman_by_donor(donor_index, within_norm, y)
                if len(rho) == 0:
                    continue
                null_medians = []
                for _ in range(args.null_draws):
                    rand = rng.choice(usable_pos, len(index), replace=False)
                    y_null = score(rand, t_idx) if kind == "target_score" else score(rand, t_idx) - score(rand, s_idx)
                    null_medians.append(np.median(spearman_by_donor(donor_index, within_norm, y_null)))
                null_medians = np.asarray(null_medians)
                shares = []
                for idx in donor_index.values():
                    if len(idx) >= 30:
                        high = y[idx] >= np.quantile(y[idx], 0.75)
                        shares.append(energy[idx][high].sum() / energy[idx].sum())
                assoc_rows.append({
                    "cohort": args.cohort, "group": args.group, "mode": mode, "fold": fold, "module": name,
                    "n_genes": len(index), "score": kind, "n_donors": len(rho),
                    "median_donor_rho": float(np.median(rho)), "frac_donors_rho_pos": float(np.mean(rho > 0)),
                    "wilcoxon_p": float(stats.wilcoxon(rho).pvalue) if len(rho) > 5 else np.nan,
                    "null_median_rho_mean": float(null_medians.mean()),
                    "null_median_rho_sd": float(null_medians.std()),
                    "null_z": float((np.median(rho) - null_medians.mean()) / null_medians.std()),
                    "null_empirical_p_greater": float((np.sum(null_medians >= np.median(rho)) + 1) / (len(null_medians) + 1)),
                    "high_quartile_energy_share": float(np.mean(shares)),
                })
            decoder_rows.append({
                "cohort": args.cohort, "group": args.group, "mode": mode, "fold": fold, "module": name,
                "spearman_dec_total_vs_observed_delta": float(stats.spearmanr(frame[f"dec_total__{name}"], delta_score).statistic),
                "spearman_dec_within_vs_observed_delta": float(stats.spearmanr(frame[f"dec_within__{name}"], delta_score).statistic),
            })

        # Per-donor, per-label within energy (labels attached post hoc).
        raw_t = labels[frame["target_row"].to_numpy()]
        raw_s = labels[frame["source_row"].to_numpy()]
        for level, lab_t, lab_s in (
            ("harmonised", np.asarray([harmonise(args.cohort, x) for x in raw_t]),
             np.asarray([harmonise(args.cohort, x) for x in raw_s])),
            ("fine", raw_t, raw_s),
        ):
            recs = []
            for donor, idx in donor_index.items():
                for value in np.unique(lab_t[idx]):
                    m = lab_t[idx] == value
                    if m.sum() >= args.min_label_cells:
                        recs.append({"donor": donor, "label": value, "n_cells": int(m.sum()),
                                     "cell_share": m.mean(), "energy_share": energy[idx][m].sum() / energy[idx].sum(),
                                     "mean_within_sq": energy[idx][m].mean(),
                                     "donor_mean_within_sq": energy[idx].mean(),
                                     "cross_label_match_rate": float(np.mean(lab_s[idx][m] != value))})
            table = pd.DataFrame(recs)
            per_donor_tables[(mode, fold, level)] = table
            for value, sub in table.groupby("label"):
                ratio = (sub["mean_within_sq"] / sub["donor_mean_within_sq"]).to_numpy()
                boot = [np.mean(ratio[rng.integers(0, len(ratio), len(ratio))]) for _ in range(args.bootstrap)]
                contrib_rows.append({
                    "cohort": args.cohort, "group": args.group, "mode": mode, "fold": fold, "level": level,
                    "label": value, "n_donors": len(sub), "mean_cell_share": sub["cell_share"].mean(),
                    "mean_energy_share": sub["energy_share"].mean(),
                    "mean_within_sq": sub["mean_within_sq"].mean(),
                    "per_cell_energy_ratio": float(ratio.mean()),
                    "ratio_ci_low": float(np.percentile(boot, 2.5)), "ratio_ci_high": float(np.percentile(boot, 97.5)),
                    "cross_label_match_rate": float(sub["cross_label_match_rate"].mean()),
                })
        print(f"[done] {mode}/{fold}", flush=True)

    # Disease - healthy excess by label.  Total excess E = mean_d - mean_h of the
    # donor mean within energy; label contribution c_l = s_d,l * m_d,l - s_h,l * m_h,l
    # (s = cell share, m = mean within energy of label cells; donor-averaged),
    # which sums to E over labels present in both modes (approximately: labels
    # with < min-label-cells in a donor are dropped for that donor).  Donors of the two
    # modes are resampled independently.
    excess_rows = []
    folds = sorted({f for (_, f) in cells})
    for fold in folds:
        for level in ("harmonised", "fine"):
            dis, hea = per_donor_tables.get(("disease", fold, level)), per_donor_tables.get(("healthy", fold, level))
            if dis is None or hea is None or dis.empty or hea.empty:
                continue

            def summarise(table: pd.DataFrame, donors: np.ndarray) -> tuple[dict, float]:
                sub = table.set_index("donor").loc[donors].reset_index()
                n = len(donors)
                total = table.groupby("donor")["donor_mean_within_sq"].first().loc[donors].mean()
                weighted = (sub["cell_share"] * sub["mean_within_sq"]).groupby(sub["label"]).sum() / n
                mean_m = sub.groupby("label")["mean_within_sq"].mean()
                share = sub.groupby("label")["cell_share"].sum() / n
                return {"weighted": weighted, "mean": mean_m, "share": share}, float(total)

            d_donors, h_donors = dis["donor"].unique(), hea["donor"].unique()
            d_point, d_total = summarise(dis, d_donors)
            h_point, h_total = summarise(hea, h_donors)
            shared_labels = sorted(set(d_point["mean"].index) & set(h_point["mean"].index))
            draws = {lab: [] for lab in shared_labels}
            frac_draws = {lab: [] for lab in shared_labels}
            for _ in range(args.bootstrap):
                d_b, d_t = summarise(dis, rng.choice(d_donors, len(d_donors)))
                h_b, h_t = summarise(hea, rng.choice(h_donors, len(h_donors)))
                for lab in shared_labels:
                    diff = d_b["mean"].get(lab, np.nan) - h_b["mean"].get(lab, np.nan)
                    draws[lab].append(diff)
                    contrib = d_b["weighted"].get(lab, 0.0) - h_b["weighted"].get(lab, 0.0)
                    frac_draws[lab].append(contrib / (d_t - h_t) if abs(d_t - h_t) > 1e-9 else np.nan)
            for lab in shared_labels:
                contrib = d_point["weighted"][lab] - h_point["weighted"][lab]
                excess_rows.append({
                    "cohort": args.cohort, "group": args.group, "fold": fold, "level": level, "label": lab,
                    "disease_mean_within_sq": d_point["mean"][lab], "healthy_mean_within_sq": h_point["mean"][lab],
                    "excess_within_sq": d_point["mean"][lab] - h_point["mean"][lab],
                    "excess_ci_low": float(np.nanpercentile(draws[lab], 2.5)),
                    "excess_ci_high": float(np.nanpercentile(draws[lab], 97.5)),
                    "disease_cell_share": d_point["share"][lab], "healthy_cell_share": h_point["share"][lab],
                    "group_total_excess": d_total - h_total,
                    "contribution_to_total_excess": contrib,
                    "fraction_of_total_excess": contrib / (d_total - h_total) if abs(d_total - h_total) > 1e-9 else np.nan,
                    "fraction_ci_low": float(np.nanpercentile(frac_draws[lab], 2.5)),
                    "fraction_ci_high": float(np.nanpercentile(frac_draws[lab], 97.5)),
                    "n_disease_donors": int(dis.loc[dis["label"].eq(lab), "donor"].nunique()),
                    "n_healthy_donors": int(hea.loc[hea["label"].eq(lab), "donor"].nunique()),
                })

    for name, table in (("module_association", assoc_rows), ("decoder_module_check", decoder_rows),
                        ("subpopulation_contribution", contrib_rows), ("subpopulation_excess", excess_rows)):
        pd.DataFrame(table).to_csv(args.outdir / f"{name}.tsv", sep="\t", index=False)
    print("[done]")


if __name__ == "__main__":
    main()
