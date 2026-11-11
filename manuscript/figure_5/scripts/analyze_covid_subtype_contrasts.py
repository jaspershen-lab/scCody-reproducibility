#!/usr/bin/env python3
"""Direct CD16-vs-CD14 contrasts of the COVID-19 effects in Fig. 5f (same-subtype matches).

For each cohort and outcome, the disease effect of each subtype is the one plotted in Fig. 5f:
  energy    Delta E = mean COVID - mean healthy of donor within-patient residual energy per cell
  identity  Delta I = the same for the donor-mean within-step Azimuth CD16 identity score (centred)
  calpro_sd R = mean COVID / mean healthy of the within-donor SD of per-cell within-step calprotectin
            complex (GO:1990660) scores
  ident_sd  R = the same for the Azimuth CD16 identity score
Contrast: Delta_CD16 - Delta_CD14 (energy, identity) or log(R_CD16 / R_CD14) (dispersion; reported
also as the ratio of ratios).  Donor values follow prepare_program_tables.py and
prepare_covid_subtype_donor_table.py (pair-weighted, >= 30 cells of the subtype per fold, averaged over
folds), matching the plotted subtype effects.

Paired donor bootstrap (10,000): COVID and healthy donors are resampled separately by donor ID, and
both subtype effects are recomputed from the same resampled donors (each from the donors that have a
value for that subtype), keeping the CD14 / CD16 pairing and healthy-reference uncertainty.  95%
percentile CI; two-sided bootstrap p = 2 x min(share <= 0, share >= 0) with +1 correction.  A
disease-label permutation is not used: it tests the stronger null that neither subtype changes.
Multiple testing: BH-FDR over the four outcomes within each cohort.

Output: results/covid_subtype_contrasts.csv.
Per-subtype estimates, confidence intervals and p/q values remain in the main
Figure 5 result tables.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_program_tables import (DATASETS, MOD, OUT, SEED, bh, donor_gene_effects,  # noqa: E402
                                       rng_for, within_sd)

BOOTSTRAP = 10000
SUBTYPES = ("CD14_Mono", "CD16_Mono")
CALPRO = "Calprotectin complex (GO:1990660)"
AZIMUTH = "CD16 monocyte (Azimuth PBMC, Hao 2021)"
OUTCOMES = {"energy": "Δ residual energy / cell", "identity": "Δ CD16 identity",
            "calpro_sd": "Calprotectin SD ratio", "ident_sd": "CD16 identity SD ratio"}


def donor_values(spec: dict) -> dict:
    """{outcome: {condition: {subtype: Series donor -> value}}}"""
    modules = {k: np.asarray(v) for k, v in json.loads(spec["modules"].read_text()).items()}
    labels = pd.read_csv(spec["labels"], sep="\t")["label"].astype(str).to_numpy()
    s = list(("aligned", "shared", "between", "within", "total")).index("within")
    out = {k: {} for k in OUTCOMES}
    energy = pd.read_csv(OUT / "covid_subtype_donor_within_energy_same_subtype.csv").pipe(lambda t: t[t.cohort == spec["cohort"]])
    for cond, run in (("COVID-19", spec["disease"]), ("Healthy", spec["healthy"])):
        gene = donor_gene_effects(spec, run)
        sd = within_sd(spec, run, labels, [f"within__{CALPRO}", f"within__{AZIMUTH}"])
        e = energy[energy.condition == cond]
        for k in OUTCOMES:
            out[k][cond] = {}
        for sub in SUBTYPES:
            key = f"{sub}|same"
            ix = modules[AZIMUTH]
            out["identity"][cond][sub] = pd.Series({d: g[s][ix].mean() - g[s].mean() for d, g in gene[key].items()})
            out["calpro_sd"][cond][sub] = sd[key][f"within__{CALPRO}"].dropna()
            out["ident_sd"][cond][sub] = sd[key][f"within__{AZIMUTH}"].dropna()
            out["energy"][cond][sub] = e[e.subtype == sub].set_index("donor")["mean_within_sq"]
    return out


def effect(vals: dict, outcome: str, sub: str, cd: np.ndarray | None = None, hd: np.ndarray | None = None) -> float:
    c, h = vals["COVID-19"][sub], vals["Healthy"][sub]
    c = c if cd is None else c.reindex(cd).dropna()
    h = h if hd is None else h.reindex(hd).dropna()
    return c.mean() / h.mean() if outcome.endswith("_sd") else c.mean() - h.mean()


def contrast(vals: dict, outcome: str, cd=None, hd=None) -> float:
    a, b = effect(vals, outcome, "CD16_Mono", cd, hd), effect(vals, outcome, "CD14_Mono", cd, hd)
    return np.log(a / b) if outcome.endswith("_sd") else a - b


def main() -> None:
    rows = []
    for spec in [d for d in DATASETS if d["dataset"] == "COVID"]:
        cohort = spec["cohort"]
        vals = donor_values(spec)
        for k, label in OUTCOMES.items():
            v = vals[k]
            cdon = np.asarray(sorted(set().union(*[set(v["COVID-19"][s].index) for s in SUBTYPES])))
            hdon = np.asarray(sorted(set().union(*[set(v["Healthy"][s].index) for s in SUBTYPES])))
            rng = rng_for("COVID", cohort, "subtype_contrast", k)
            boot = np.asarray([contrast(v, k, rng.choice(cdon, len(cdon)), rng.choice(hdon, len(hdon)))
                               for _ in range(BOOTSTRAP)])
            boot = boot[np.isfinite(boot)]
            est = contrast(v, k)
            p = min(1.0, 2 * min((np.sum(boot <= 0) + 1), (np.sum(boot >= 0) + 1)) / (len(boot) + 1))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            log = k.endswith("_sd")
            rows.append({"cohort": cohort, "outcome": k, "label": label,
                         "contrast": "log(R_CD16 / R_CD14)" if log else "Delta_CD16 - Delta_CD14",
                         "estimate": est, "ci_low": lo, "ci_high": hi,
                         "ratio_of_ratios": np.exp(est) if log else np.nan,
                         "ratio_ci_low": np.exp(lo) if log else np.nan, "ratio_ci_high": np.exp(hi) if log else np.nan,
                         "effect_CD14": effect(v, k, "CD14_Mono"), "effect_CD16": effect(v, k, "CD16_Mono"),
                         "n_covid_CD14": len(v["COVID-19"]["CD14_Mono"]), "n_covid_CD16": len(v["COVID-19"]["CD16_Mono"]),
                         "n_healthy_CD14": len(v["Healthy"]["CD14_Mono"]), "n_healthy_CD16": len(v["Healthy"]["CD16_Mono"]),
                         "n_bootstrap": len(boot), "p_bootstrap_two_sided": p})
        print(f"[contrast] {cohort}", flush=True)
    res = pd.DataFrame(rows)
    res["q"] = res.groupby("cohort", group_keys=False)["p_bootstrap_two_sided"].apply(bh)
    res.to_csv(OUT / "covid_subtype_contrasts.csv", index=False)

    print(res[["cohort", "outcome", "estimate", "ci_low", "ci_high", "ratio_of_ratios", "p_bootstrap_two_sided", "q"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
