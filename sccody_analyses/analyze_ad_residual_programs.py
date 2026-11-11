#!/usr/bin/env python3
"""Expression programs of the aging-aligned and shared residual components (PsychAD AD, excitatory neurons).

Uses ``decode_residual_components.py`` output.  All gene-level statistics treat
the target (AD) donor as the unit: per-donor decoded effects are averaged over
that donor's reference pairs, and inference is across donors.

Outputs
  nonlinearity_check.tsv     chained vs reverse-order effects (path dependence)
  component_summary.tsv      gene-space magnitude of each component
  signature_similarity.tsv   aligned vs shared signature correlation (donor bootstrap CI)
  observed_concordance.tsv   correlation of each signature with observed pseudobulk
                             AD-vs-control DE and observed control aging slope
  gene_component_stats.tsv   per-gene mean / SE / t for each component
  module_effects.tsv         module mean effect vs size-matched random-gene null
                             (disease and, with --healthy-decoded, the healthy +10y reference)
  healthy_reference.tsv      disease vs healthy-reference signature correlations and
                             gene-space magnitudes (same coarse-group OT, holdout controls)
  gsea_<component>.tsv       preranked GSEA on donor t-statistics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import gseapy as gp
import numpy as np
import pandas as pd
from scipy import stats
import json


def gene_t(effects: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = effects.mean(axis=0)
    se = effects.std(axis=0, ddof=1) / np.sqrt(len(effects))
    return mean, se, mean / np.where(se > 0, se, np.nan)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoded", type=Path, required=True)
    parser.add_argument("--healthy-decoded", type=Path, default=None)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--modules", type=Path, required=True)
    parser.add_argument("--gmt-dir", type=Path, required=True)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--group-key", default="class")
    parser.add_argument("--group", default="EN")
    parser.add_argument("--donor-key", default="donor_id")
    parser.add_argument("--age-key", default="age")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--null-draws", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    data = np.load(args.decoded / "donor_component_effects.npz")
    effects = data["effects"].astype(np.float64)  # donor x component x gene
    components = [str(c) for c in data["components"]]
    case_donors = [str(d) for d in data["donors"]]
    C = {c: effects[:, i, :] for i, c in enumerate(components)}
    panel = pd.read_csv(args.panel, sep="\t")
    symbols = panel["symbol"].astype(str).to_numpy()
    n_donors = len(case_donors)
    print(f"[load] {n_donors} AD donors, {len(symbols)} genes")

    # 1. Path dependence of the decoder.
    rows = []
    for name, alt in (("aligned", "aligned_alt"), ("shared", "shared_alt")):
        a, b = C[name].mean(axis=0), C[alt].mean(axis=0)
        per_donor = [corr(C[name][i], C[alt][i]) for i in range(n_donors)]
        rows.append({
            "component": name, "pearson_grand_mean": corr(a, b),
            "relative_l2_difference": float(np.linalg.norm(a - b) / np.linalg.norm(a)),
            "median_donor_pearson": float(np.median(per_donor)),
        })
    total = C["total"].mean(axis=0)
    chain = sum(C[c].mean(axis=0) for c in ("aligned", "shared", "between", "within"))
    rows.append({"component": "chain_sum_vs_total", "pearson_grand_mean": corr(total, chain),
                 "relative_l2_difference": float(np.linalg.norm(total - chain) / np.linalg.norm(total)),
                 "median_donor_pearson": np.nan})
    pd.DataFrame(rows).to_csv(args.outdir / "nonlinearity_check.tsv", sep="\t", index=False)

    # 2. Gene-space magnitude (grand-mean signature RMS and mean per-donor RMS).
    rows = []
    for c in ("aligned", "shared", "between", "within", "total"):
        rows.append({"component": c,
                     "grand_mean_rms": float(np.sqrt(np.mean(C[c].mean(axis=0) ** 2))),
                     "mean_donor_rms": float(np.mean(np.sqrt(np.mean(C[c] ** 2, axis=1))))})
    pd.DataFrame(rows).to_csv(args.outdir / "component_summary.tsv", sep="\t", index=False)

    # 3. Gene statistics.
    gene_rows = pd.DataFrame({"symbol": symbols, "ensembl": panel["ensembl"]})
    for c in ("aligned", "shared", "total", "shared_alt", "aligned_alt"):
        mean, se, t = gene_t(C[c])
        gene_rows[f"{c}_mean"], gene_rows[f"{c}_se"], gene_rows[f"{c}_t"] = mean, se, t
    gene_rows["base_mean"] = C["base"].mean(axis=0)

    # 4. Observed pseudobulk: AD vs control DE and control aging slope.
    pairs = pd.read_csv(args.pairs, sep="\t")
    control_donors = sorted(set(pairs["reference_donor"].astype(str)))
    obj = ad.read_h5ad(args.h5ad, backed="r")
    obs = obj.obs[[args.group_key, args.donor_key, args.age_key]].copy()
    obs[args.donor_key] = obs[args.donor_key].astype(str)
    keep = (obs[args.group_key].astype(str) == args.group) & obs[args.donor_key].isin(set(case_donors) | set(control_donors))
    rows_idx = np.flatnonzero(keep.to_numpy())
    pseudo, ages = {}, {}
    sub_obs = obs.iloc[rows_idx]
    for start in range(0, len(rows_idx), 50000):
        block = obj.X[rows_idx[start:start + 50000]]
        block = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
        donors_block = sub_obs[args.donor_key].to_numpy()[start:start + 50000]
        for d in np.unique(donors_block):
            m = donors_block == d
            s, n = pseudo.get(d, (0.0, 0))
            pseudo[d] = (s + block[m].sum(axis=0), n + int(m.sum()))
    obj.file.close()
    for d, age in sub_obs.groupby(args.donor_key, observed=True)[args.age_key].median().items():
        ages[d] = float(age)
    mean_expr = {d: s / n for d, (s, n) in pseudo.items()}
    case = np.stack([mean_expr[d] for d in case_donors if d in mean_expr])
    ctrl_ids = [d for d in control_donors if d in mean_expr]
    ctrl = np.stack([mean_expr[d] for d in ctrl_ids])
    observed_de = case.mean(axis=0) - ctrl.mean(axis=0)
    ctrl_age = np.asarray([ages[d] for d in ctrl_ids])
    X = np.column_stack([np.ones_like(ctrl_age), ctrl_age])
    observed_aging_slope = np.linalg.lstsq(X, ctrl, rcond=None)[0][1] * 10.0  # per decade
    gene_rows["observed_ad_de"] = observed_de
    gene_rows["observed_ctrl_aging_per_decade"] = observed_aging_slope
    gene_rows.to_csv(args.outdir / "gene_component_stats.tsv", sep="\t", index=False)

    def boot_corr(x_donor: np.ndarray, y: np.ndarray | None, y_donor: np.ndarray | None = None) -> tuple[float, float, float]:
        point = corr(x_donor.mean(axis=0), y if y is not None else y_donor.mean(axis=0))
        draws = []
        for _ in range(args.bootstrap):
            idx = rng.integers(0, n_donors, n_donors)
            draws.append(corr(x_donor[idx].mean(axis=0), y if y is not None else y_donor[idx].mean(axis=0)))
        low, high = np.percentile(draws, [2.5, 97.5])
        return point, float(low), float(high)

    rows = []
    for c in ("aligned", "shared", "between", "total"):
        for target_name, target in (("observed_ad_de", observed_de), ("observed_ctrl_aging", observed_aging_slope)):
            if c == "between":
                continue
            r, lo, hi = boot_corr(C[c], target)
            rows.append({"signature": c, "compared_to": target_name, "pearson": r, "ci_low": lo, "ci_high": hi})
    # Joint regression of observed DE on aligned + shared signatures (standardised).
    A = np.column_stack([C["aligned"].mean(axis=0), C["shared"].mean(axis=0)])
    A = (A - A.mean(axis=0)) / A.std(axis=0)
    y = (observed_de - observed_de.mean()) / observed_de.std()
    beta = np.linalg.lstsq(np.column_stack([np.ones(len(y)), A]), y, rcond=None)[0]
    fitted = np.column_stack([np.ones(len(y)), A]) @ beta
    rows.append({"signature": "aligned+shared joint", "compared_to": "observed_ad_de",
                 "pearson": corr(fitted, y), "ci_low": np.nan, "ci_high": np.nan,
                 "beta_aligned": beta[1], "beta_shared": beta[2]})
    pd.DataFrame(rows).to_csv(args.outdir / "observed_concordance.tsv", sep="\t", index=False)

    r, lo, hi = boot_corr(C["aligned"], None, C["shared"])
    r_alt, lo_alt, hi_alt = boot_corr(C["aligned_alt"], None, C["shared_alt"])
    pd.DataFrame([
        {"comparison": "aligned_vs_shared (chained)", "pearson": r, "ci_low": lo, "ci_high": hi},
        {"comparison": "aligned_vs_shared (reverse order)", "pearson": r_alt, "ci_low": lo_alt, "ci_high": hi_alt},
    ]).to_csv(args.outdir / "signature_similarity.tsv", sep="\t", index=False)

    # 5. Module effects with a size-matched random-gene null and donor bootstrap CI.
    modules = json.loads(args.modules.read_text())
    rows = []
    n_genes = len(symbols)
    H = None
    if args.healthy_decoded is not None:
        hdata = np.load(args.healthy_decoded / "donor_component_effects.npz")
        H = {str(c): hdata["effects"].astype(np.float64)[:, i, :] for i, c in enumerate(hdata["components"])}
    mode_sets = [("disease", C)] + ([("healthy", H)] if H is not None else [])
    for (mode, source), c in [(m, c) for m in mode_sets for c in ("aligned", "shared", "total")]:
        donor_effect = source[c]
        n_donors_mode = len(donor_effect)
        grand = donor_effect.mean(axis=0)
        for name, index in modules.items():
            index = np.asarray(index)
            value = float(grand[index].mean())
            null = np.asarray([grand[rng.choice(n_genes, len(index), replace=False)].mean() for _ in range(args.null_draws)])
            donor_vals = donor_effect[:, index].mean(axis=1) - donor_effect.mean(axis=1)
            boot = [donor_vals[rng.integers(0, n_donors_mode, n_donors_mode)].mean() for _ in range(args.bootstrap)]
            rows.append({
                "mode": mode, "component": c, "module": name, "n_genes": len(index),
                "module_mean_effect": value, "null_mean": float(null.mean()), "null_sd": float(null.std()),
                "null_z": float((value - null.mean()) / null.std()),
                "empirical_p_two_sided": float((np.sum(np.abs(null - null.mean()) >= abs(value - null.mean())) + 1) / (len(null) + 1)),
                "centered_effect_mean": float(donor_vals.mean()),
                "centered_ci_low": float(np.percentile(boot, 2.5)), "centered_ci_high": float(np.percentile(boot, 97.5)),
                "frac_donors_positive": float(np.mean(donor_vals > 0)),
                "observed_de_module_mean": float(observed_de[index].mean() - observed_de.mean()),
            })
    pd.DataFrame(rows).to_csv(args.outdir / "module_effects.tsv", sep="\t", index=False)

    if H is not None:
        rows = []
        for dc, hc in (("aligned", "aligned"), ("shared", "shared"), ("total", "total"), ("shared", "aligned")):
            d_sig, h_sig = C[dc].mean(axis=0), H[hc].mean(axis=0)
            boots = []
            for _ in range(args.bootstrap):
                di = rng.integers(0, len(C[dc]), len(C[dc]))
                hi = rng.integers(0, len(H[hc]), len(H[hc]))
                boots.append(corr(C[dc][di].mean(axis=0), H[hc][hi].mean(axis=0)))
            rows.append({
                "disease_component": dc, "healthy_component": hc, "pearson": corr(d_sig, h_sig),
                "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5)),
                "disease_grand_mean_rms": float(np.sqrt(np.mean(d_sig ** 2))),
                "healthy_grand_mean_rms": float(np.sqrt(np.mean(h_sig ** 2))),
                "n_disease_donors": len(C[dc]), "n_healthy_donors": len(H[hc]),
            })
        pd.DataFrame(rows).to_csv(args.outdir / "healthy_reference.tsv", sep="\t", index=False)

    # 6. Preranked GSEA on donor t-statistics (panel genes only).
    gene_sets = {}
    panel_set = set(symbols)
    for path in sorted(args.gmt_dir.glob("*.gmt")):
        if "GenAge" in path.name or "CellAge" in path.name:
            continue
        for line in path.read_text().splitlines():
            parts = line.split("\t")
            genes = sorted(set(parts[2:]) & panel_set)
            if 10 <= len(genes) <= 300:
                gene_sets[parts[0]] = genes
    print(f"[gsea] {len(gene_sets)} gene sets with 10-300 panel genes")
    for c in ("aligned", "shared"):
        rank = gene_rows[["symbol", f"{c}_t"]].dropna().groupby("symbol").mean()[f"{c}_t"].sort_values(ascending=False)
        result = gp.prerank(rnk=rank, gene_sets=gene_sets, permutation_num=args.permutations,
                            min_size=10, max_size=300, threads=args.threads, seed=args.seed,
                            outdir=None, no_plot=True, verbose=False)
        table = result.res2d.sort_values("FDR q-val")
        table.to_csv(args.outdir / f"gsea_{c}.tsv", sep="\t", index=False)
        print(f"[gsea] {c}: {int((table['FDR q-val'].astype(float) < 0.05).sum())} sets at FDR<0.05")
    print("[done]")


if __name__ == "__main__":
    main()
