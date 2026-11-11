#!/usr/bin/env python3
"""Per-donor within-patient residual energy by monocyte subtype (COVID-19 vs healthy).

Rebuilt from the saved per-cell decoded outputs
(artifacts/residual_programs/decoded/<cohort>/fold*/Myeloid[_healthy]/
percell_components.tsv.gz); nothing is re-decoded.  Definitions follow
analyze_covid_residual_programs.py:
  * within_sq = ||r_i - r_D||^2 per matched cell (latent);
  * subtype = harmonised label of the matched COVID/older-control (target) cell,
    attached after coarse-group OT;
  * a donor contributes a subtype in a fold only with >= 30 such cells; its value is pair-weighted
    (mean over the donor's cells of that subtype within each matched pair, then over pairs), the
    convention of prepare_program_tables.py.
Every stats table gets q_two_sided: BH-FDR over CD14 and CD16 within each cohort.
Per-donor values are averaged over the folds in which the donor is a target,
so each donor appears once per subtype.

Outputs (manuscript/figure_5/results/; Fig. 5f):
  covid_subtype_donor_within_energy.csv   one row per cohort x condition x donor x subtype
  covid_subtype_cell_share.csv            % of Myeloid target cells per subtype (donor mean over
                                          folds, then mean over donors; donor-bootstrap CI)
  covid_subtype_excess_donor_stats.csv    COVID - healthy difference of donor means with a
                                          pooled donor bootstrap CI, permutation p and
                                          Mann-Whitney p
  covid_subtype_donor_within_energy_same_subtype.csv   donor table for the same-subtype matches
  covid_subtype_excess_same_subtype_stats.csv
                                          as above, using only cells whose matched control (source)
                                          cell has the same subtype label, so composition-driven
                                          cross-subtype matches (e.g. COVID CD16 -> control CD14)
                                          do not contribute; separate RNG so the tables above are
                                          unchanged
  covid_subtype_cell_share_change.csv     COVID - healthy difference of the donor-mean subtype %
                                          (percentage points) with the same pooled donor bootstrap
                                          CI and permutation p as the excess (separate RNG)
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "artifacts" / "residual_programs"
OUT = REPO / "manuscript" / "figure_5" / "results"
H5AD = {
    "Ren": str(REPO / "data/covid/ren_covid_case_control_aida2000_mean_imputed.h5ad"),
    "COMBAT": str(REPO / "data/covid/combat_covid_case_control_aida2000_mean_imputed.h5ad"),
    "Stephenson": str(REPO / "data/covid/stephenson_covid_case_control_aida2000_mean_imputed.h5ad"),
}
LABEL_KEY = {"Ren": "celltype", "COMBAT": "Annotation_major_subset", "Stephenson": "initial_clustering"}
MIN_CELLS = 30
BOOTSTRAP = 10000
PERMUTATIONS = 10000
SEED = 20261012


def subtype(cohort: str, label: str) -> str | None:
    if cohort == "Ren":
        if label.startswith("Mono_c5"):
            return "CD16_Mono"
        if label.startswith("Mono_c") and not label.startswith("Mono_c4"):
            return "CD14_Mono"
        return None
    if cohort == "COMBAT":
        return {"cMono": "CD14_Mono", "ncMono": "CD16_Mono"}.get(label)
    return {"CD14": "CD14_Mono", "CD16": "CD16_Mono"}.get(label)


def donor_means(table: pd.DataFrame) -> pd.DataFrame:
    return (table.groupby(["target_donor", "subtype"], as_index=False)
            .agg(mean_within_sq=("mean_within_sq", "mean"), n_cells_mean=("n_cells", "mean"),
                 n_folds=("fold", "nunique")))


def per_fold_means(cells: pd.DataFrame, fold: str) -> pd.DataFrame:
    pairs = (cells.groupby(["target_donor", "subtype", "pair_id"])["within_sq"]
             .agg(n="size", m="mean").reset_index())
    agg = (pairs.groupby(["target_donor", "subtype"])
           .agg(n_cells=("n", "sum"), mean_within_sq=("m", "mean")).reset_index())
    return agg[agg["n_cells"] >= MIN_CELLS].assign(fold=fold)


def add_q(t: pd.DataFrame) -> pd.DataFrame:
    from prepare_program_tables import bh
    return t.assign(q_two_sided=t.groupby("cohort", group_keys=False)["perm_p_two_sided"].apply(bh))


def excess_stats(donors: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for (cohort, sub), group in donors.groupby(["cohort", "subtype"]):
        d = group.loc[group["condition"] == "COVID-19", "mean_within_sq"].to_numpy()
        h = group.loc[group["condition"] == "Healthy", "mean_within_sq"].to_numpy()
        diff = d.mean() - h.mean()
        boot = np.asarray([rng.choice(d, len(d)).mean() - rng.choice(h, len(h)).mean() for _ in range(BOOTSTRAP)])
        pooled = np.concatenate([d, h])
        perm = np.empty(PERMUTATIONS)
        for i in range(PERMUTATIONS):
            idx = rng.permutation(len(pooled))
            perm[i] = pooled[idx[:len(d)]].mean() - pooled[idx[len(d):]].mean()
        rows.append({
            "cohort": cohort, "subtype": sub, "n_covid_donors": len(d), "n_healthy_donors": len(h),
            "covid_mean": d.mean(), "healthy_mean": h.mean(), "excess": diff,
            "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)),
            "perm_p_two_sided": float((np.sum(np.abs(perm) >= abs(diff)) + 1) / (PERMUTATIONS + 1)),
            "mannwhitney_p_two_sided": float(stats.mannwhitneyu(d, h, alternative="two-sided").pvalue),
        })
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    donor_rows, same_rows, share_rows, share_donor_rows = [], [], [], []
    for cohort, path in H5AD.items():
        obj = ad.read_h5ad(path, backed="r")
        raw = obj.obs[LABEL_KEY[cohort]].astype(str).to_numpy()
        obj.file.close()
        mapped = np.asarray([subtype(cohort, x) for x in raw], dtype=object)
        for condition, suffix in (("COVID-19", ""), ("Healthy", "_healthy")):
            per_fold, same_fold, share_fold = [], [], []
            for fold_dir in sorted((RES / "decoded" / cohort).glob("fold*")):
                cells = pd.read_csv(fold_dir / f"Myeloid{suffix}" / "percell_components.tsv.gz", sep="\t",
                                    usecols=["pair_id", "target_donor", "source_row", "target_row", "within_sq"])
                cells["subtype"] = mapped[cells["target_row"].to_numpy()]
                source = mapped[cells["source_row"].to_numpy()]
                # share of all Myeloid target cells (other labels count in the denominator)
                counts = pd.crosstab(cells["target_donor"], cells["subtype"].fillna("other"))
                share = counts.div(counts.sum(axis=1), axis=0).reindex(columns=["CD14_Mono", "CD16_Mono"], fill_value=0)
                share_fold.append(share)
                same = cells[cells["subtype"].to_numpy() == source]
                cells = cells.dropna(subset=["subtype"])
                per_fold.append(per_fold_means(cells, fold_dir.name))
                same_fold.append(per_fold_means(same.dropna(subset=["subtype"]), fold_dir.name))
            donor_rows.append(donor_means(pd.concat(per_fold)).assign(cohort=cohort, condition=condition))
            same_rows.append(donor_means(pd.concat(same_fold)).assign(cohort=cohort, condition=condition))
            donor_share = pd.concat(share_fold).groupby(level=0).mean()
            share_donor_rows.append(
                donor_share.rename_axis("target_donor").reset_index()
                .melt(id_vars="target_donor", var_name="subtype", value_name="mean_within_sq")
                .assign(mean_within_sq=lambda d: 100 * d["mean_within_sq"], cohort=cohort, condition=condition))
            for sub in ("CD14_Mono", "CD16_Mono"):
                values = donor_share[sub].to_numpy()
                boot = [rng.choice(values, len(values)).mean() for _ in range(BOOTSTRAP)]
                share_rows.append({"cohort": cohort, "condition": condition, "subtype": sub,
                                   "n_donors": len(values), "percent": 100 * values.mean(),
                                   "ci_low": 100 * float(np.percentile(boot, 2.5)),
                                   "ci_high": 100 * float(np.percentile(boot, 97.5))})
    columns = ["cohort", "condition", "donor", "subtype", "mean_within_sq", "n_cells_mean", "n_folds"]
    donors = pd.concat(donor_rows).rename(columns={"target_donor": "donor"})[columns]
    donors.to_csv(OUT / "covid_subtype_donor_within_energy.csv", index=False)
    pd.DataFrame(share_rows).to_csv(OUT / "covid_subtype_cell_share.csv", index=False)

    result = add_q(excess_stats(donors, rng))
    result.to_csv(OUT / "covid_subtype_excess_donor_stats.csv", index=False)
    print(result.round(3).to_string(index=False))
    same = pd.concat(same_rows).rename(columns={"target_donor": "donor"})[columns]
    same.to_csv(OUT / "covid_subtype_donor_within_energy_same_subtype.csv", index=False)
    same_result = add_q(excess_stats(same, np.random.default_rng(SEED + 1)))
    same_result.to_csv(OUT / "covid_subtype_excess_same_subtype_stats.csv", index=False)
    print("same-subtype matches only")
    print(same_result.round(3).to_string(index=False))

    # excess_stats() works on a mean_within_sq column; here it holds each donor's subtype % instead
    share_change = add_q(excess_stats(pd.concat(share_donor_rows), np.random.default_rng(SEED + 3)))
    share_change = share_change.rename(columns={"covid_mean": "covid_percent", "healthy_mean": "healthy_percent",
                                                "excess": "change_pp"})
    share_change.to_csv(OUT / "covid_subtype_cell_share_change.csv", index=False)
    print("CD16 / CD14 % of Myeloid cells, COVID - healthy (percentage points)")
    print(share_change.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
