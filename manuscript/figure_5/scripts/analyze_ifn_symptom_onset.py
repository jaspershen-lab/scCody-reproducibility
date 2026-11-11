#!/usr/bin/env python3
"""Donor IFN response vs days since symptom onset and severity (COVID Myeloid, decoded programs).

Per COVID donor with a single sample (r_D pools a donor's samples), the pair-weighted donor mean of
the per-cell centred program score for each displacement step (component_program_scores.tsv.gz),
averaged over folds.  Clinical annotations from each cohort's h5ad:
  Ren         days after symptom onset; severity mild/moderate < severe/critical
  COMBAT      TimeSinceOnset; severity community-mild < hospital-mild < severe < critical
  Stephenson  no onset date; severity asymptomatic < mild < moderate < severe < critical (on the day)
Tests: Spearman rho with days and with severity rank; joint OLS on ranks (score ~ rank(days) + severity)
for days adjusted for severity and severity adjusted for days.
Multiple testing: BH-FDR over the three IFN programs within each cohort x step x test.
Output: results/covid_ifn_symptom_onset.csv (statistics), results/covid_ifn_symptom_onset_donors.csv
(one row per donor: days, severity label and rank, severe/critical flag, step scores for Fig. 5e)
"""
from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_covid_subtype_donor_table import H5AD, OUT, RES  # noqa: E402
from prepare_program_tables import bh  # noqa: E402

PROGS = {"Interferon α/β signaling": "IFN-α/β (Reactome)", "Interferon-α response": "IFN-α (Hallmark)",
         "Interferon-γ response": "IFN-γ (Hallmark)"}
STEPS = ("aligned", "shared", "between", "within", "total")
META = {"Ren": ("sampleID", "Sampling day (Days after symptom onset)", "CoVID-19 severity",
                ["mild/moderate", "severe/critical"]),
        "COMBAT": ("COMBAT_participant_timepoint_ID", "TimeSinceOnset", "disease_group",
                   ["COVID_HCW_MILD", "COVID_MILD", "COVID_SEV", "COVID_CRIT"]),
        "Stephenson": ("sample_id", None, "status_on_day_collection_summary",
                       ["Asymptomatic", "Mild", "Moderate", "Severe", "Critical"])}


def main() -> None:
    cols = [f"{s}__{p}" for s in STEPS for p in PROGS]
    rows, donors = [], []
    severe = {"severe/critical", "COVID_SEV", "COVID_CRIT", "Severe", "Critical"}
    for cohort, path in H5AD.items():
        scol, dcol, vcol, order = META[cohort]
        obs = ad.read_h5ad(path, backed="r").obs.reset_index(drop=True)
        per = []
        for fd in sorted((RES / "decoded_programs" / cohort).glob("fold*")):
            c = pd.read_csv(fd / "Myeloid" / "component_program_scores.tsv.gz", sep="\t",
                            usecols=["target_donor", "pair_id", "target_row"] + cols)
            c["sample"] = obs[scol].astype(str).to_numpy()[c.target_row]
            nsamp = c.groupby("target_donor")["sample"].nunique()
            c = c[c.target_donor.isin(nsamp.index[nsamp == 1])]
            per.append(c.groupby(["target_donor", "sample", "pair_id"])[cols].mean().groupby(level=[0, 1]).mean())
        t = pd.concat(per).groupby(level=[0, 1]).mean().reset_index()
        sm = obs.groupby(obs[scol].astype(str), observed=True).first()
        t["days"] = pd.to_numeric(sm.loc[t["sample"], dcol].to_numpy(), errors="coerce") if dcol else np.nan
        sev = sm.loc[t["sample"], vcol].astype(str).str.strip().to_numpy()
        t["sev"] = [order.index(x) if x in order else np.nan for x in sev]
        t["severity"] = sev
        t["severe_critical"] = [x in severe for x in sev]
        t = t.dropna(subset=["sev"])
        donors.append(t.assign(cohort=cohort))
        for step in STEPS:
            for p, name in PROGS.items():
                y = t[f"{step}__{p}"]
                base = {"cohort": cohort, "step": step, "program": name, "n_donors": len(t)}
                rho, pv = stats.spearmanr(t.sev, y)
                rows.append({**base, "test": "severity", "estimate": rho, "p": pv})
                if dcol:
                    rho, pv = stats.spearmanr(t.days, y, nan_policy="omit")
                    rows.append({**base, "test": "days", "estimate": rho, "p": pv})
                    g = pd.DataFrame({"y": stats.rankdata(y), "rd": stats.rankdata(t.days), "rs": t.sev})
                    fit = smf.ols("y ~ rd + rs", data=g).fit()
                    rows.append({**base, "test": "days_adj_severity", "estimate": fit.params["rd"], "p": fit.pvalues["rd"]})
                    rows.append({**base, "test": "severity_adj_days", "estimate": fit.params["rs"], "p": fit.pvalues["rs"]})
    out = pd.DataFrame(rows)
    out["q"] = out.groupby(["cohort", "step", "test"], group_keys=False)["p"].apply(bh)
    out.to_csv(OUT / "covid_ifn_symptom_onset.csv", index=False)
    pd.concat(donors).rename(columns={"target_donor": "donor", "sev": "severity_rank"}).to_csv(
        OUT / "covid_ifn_symptom_onset_donors.csv", index=False)
    print(out[out.step == "between"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
