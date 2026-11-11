#!/usr/bin/env python3
"""Program change and dispersion tables for pooled and subtype groups, COVID and AD.

Inputs (decode_component_program_scores.py --labels, CPU; RES/decoded_programs/...)
  group_gene_effects.npz          gene-level decoded effect per donor x cell group x step
  component_program_scores.tsv.gz per-cell centred program score per step
  configs/programs/curated/*_modules.json, RES/group_labels/<dataset>.tsv
COVID: Ren, COMBAT, Stephenson (3 folds; disease = Myeloid, healthy = Myeloid_healthy, the +10 y
older-control reference).  AD: PsychAD EN (one fold; healthy = EN_healthy).

Cell groups (labels from the target cell): "pooled" (all cells), and per label (CD14_Mono /
CD16_Mono for COVID, EN subclass for AD) with "all" matches or "same" (matched source cell has the
same label).  A donor contributes a group in a fold with >= 30 of its cells there; donor values are
pair-weighted (mean over cells within each pair, then over pairs) and averaged over folds.

Centred score (all steps, both tables): mean over module genes minus mean over all panel genes.

Program change (program_change.csv)
  difference      mean over disease donors - mean over healthy donors of the donor-mean centred
                  step score; 95% CI from 10,000 donor-bootstrap draws (groups resampled
                  independently); two-sided donor-label permutation p (10,000).
  size_matched_z  the same difference, standardised against 2,000 random gene sets of the module's
                  size drawn from the usable genes (measured genes for COVID; all panel genes for AD):
                  z = (difference - mean null) / SD null, null sets fixed per dataset/cohort and size.
Program dispersion (program_dispersion.csv), disease / healthy SD ratios of centred scores
  between_patient SD across donors (ddof 1) of the donor-mean step score.
  within_patient  mean over donors of the within-donor SD of per-cell step scores (per fold, >= 30
                  cells; averaged over folds).
  95% CI from 10,000 donor-bootstrap draws; permutation p one-sided (ratio > 1 and ratio < 1) and
  two-sided (|log|).
Multiple testing: Benjamini-Hochberg q-values within families of programs.  Change: one family per
dataset x cohort x cell group x match set x step.  Dispersion: one family per dataset x cohort x cell
group x match set x step x statistic (q for each p column).  Pooled families hold only the Fig. 5
programs (the subtype-only modules get no pooled q); subtype / subclass families hold all modules.

Also written: pooled views in the column layout of the Fig. 5 plots (covid_program_heatmap.csv,
covid_component_dispersion.csv, ad_en_program_heatmap.csv).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "artifacts" / "residual_programs"
MOD = REPO / "configs" / "programs"
OUT = REPO / "manuscript" / "figure_5" / "results"
BOOTSTRAP = 10000
PERMUTATIONS = 10000
NULL_DRAWS = 2000
MIN_CELLS = 30
SEED = 20261013
STEPS = ("aligned", "shared", "between", "within", "total")

DATASETS = [
    {"dataset": "COVID", "cohort": c, "root": RES / "decoded_programs" / c, "folds": ["fold0", "fold1", "fold2"],
     "disease": "Myeloid", "healthy": "Myeloid_healthy", "modules": MOD / "curated/COVID_modules.json",
     "panel": MOD / "COVID_panel.tsv", "labels": RES / f"group_labels/{c}.tsv", "groups": ["CD14_Mono", "CD16_Mono"]}
    for c in ("Ren", "COMBAT", "Stephenson")
] + [
    {"dataset": "AD", "cohort": "PsychAD", "root": RES / "decoded_programs" / "AD", "folds": ["."],
     "disease": "EN", "healthy": "EN_healthy", "modules": MOD / "curated/AD_modules.json",
     "panel": MOD / "AD_panel.tsv", "labels": RES / "group_labels/AD.tsv", "groups": None},
]


def rng_for(*labels: object) -> np.random.Generator:
    key = "|".join(map(str, labels)).encode()
    return np.random.default_rng([SEED, int.from_bytes(hashlib.sha256(key).digest()[:8], "little")])


def donor_gene_effects(spec: dict, run: str) -> dict:
    """{group key: {donor: steps x genes}} averaged over folds where the donor has >= MIN_CELLS."""
    acc: dict = {}
    for fold in spec["folds"]:
        with np.load(spec["root"] / fold / run / "group_gene_effects.npz") as g:   # read each array once
            groups, donors, n_cells, effects = g["groups"], g["donors"], g["n_cells"], g["effects"]
        for j, key in enumerate(groups):
            for k, donor in enumerate(donors):
                if n_cells[j, k] >= MIN_CELLS:
                    acc.setdefault(str(key), {}).setdefault(str(donor), []).append(effects[j, k].astype(np.float64))
    return {key: {d: np.mean(v, axis=0) for d, v in donors.items()} for key, donors in acc.items()}


def within_sd(spec: dict, run: str, labels: np.ndarray, cols: list[str]) -> dict:
    """{group key: DataFrame donors x cols of within-donor SD of per-cell scores}, averaged over folds."""
    acc: dict = {}
    for fold in spec["folds"]:
        cells = pd.read_csv(spec["root"] / fold / run / "component_program_scores.tsv.gz", sep="\t",
                            usecols=["target_donor", "source_row", "target_row"] + cols)
        tgt, src = labels[cells["target_row"].to_numpy()], labels[cells["source_row"].to_numpy()]
        masks = {"ALL|all": np.ones(len(cells), bool)}
        for label in np.unique(tgt):
            masks[f"{label}|all"] = tgt == label
            masks[f"{label}|same"] = (tgt == label) & (src == label)
        for key, m in masks.items():
            sub = cells[m]
            n = sub.groupby("target_donor").size()
            sd = sub[sub["target_donor"].isin(n.index[n >= MIN_CELLS])].groupby("target_donor")[cols].std()
            acc.setdefault(key, []).append(sd)
    return {key: pd.concat(frames).groupby(level=0).mean() for key, frames in acc.items()}


def boot_perm(d: np.ndarray, h: np.ndarray, stat, rng: np.random.Generator):
    """Vectorised over modules (columns): bootstrap draws and permutation draws of stat(d), stat(h)."""
    bd = stat(d[rng.integers(0, len(d), (BOOTSTRAP, len(d)))])
    bh = stat(h[rng.integers(0, len(h), (BOOTSTRAP, len(h)))])
    pooled = np.vstack([d, h])
    perm = np.argsort(rng.random((PERMUTATIONS, len(pooled))), axis=1)
    pd_, ph = stat(pooled[perm[:, :len(d)]]), stat(pooled[perm[:, len(d):]])
    return bd, bh, pd_, ph


mean_ = lambda x: x.mean(axis=-2)
sd_ = lambda x: x.std(axis=-2, ddof=1)


def change_rows(dis: dict, hea: dict, modules: dict, usable: np.ndarray, tag: tuple, null_sets: dict) -> list[dict]:
    donors_d, donors_h = sorted(dis), sorted(hea)
    Gd = np.stack([dis[k] for k in donors_d]); Gh = np.stack([hea[k] for k in donors_h])   # donors x steps x genes
    rows = []
    for s, step in enumerate(STEPS):
        gd, gh = Gd[:, s, :], Gh[:, s, :]
        sc = lambda G: np.stack([G[:, ix].mean(axis=1) - G.mean(axis=1) for ix in modules.values()], axis=1)
        d, h = sc(gd), sc(gh)
        diff = d.mean(0) - h.mean(0)
        bd, bh, pd_, ph = boot_perm(d, h, mean_, rng_for(*tag, step, "change"))
        boot, perm = bd - bh, pd_ - ph
        delta = gd.mean(0) - gh.mean(0)
        for m, (name, ix) in enumerate(modules.items()):
            null = delta[null_sets[len(ix)]].mean(axis=1) - delta.mean()
            rows.append({"step": step, "module": name, "n_genes": len(ix), "n_disease_donors": len(d),
                         "n_healthy_donors": len(h), "disease_mean": d[:, m].mean(), "healthy_mean": h[:, m].mean(),
                         "difference": diff[m], "ci_low": np.percentile(boot[:, m], 2.5),
                         "ci_high": np.percentile(boot[:, m], 97.5),
                         "perm_p_two_sided": (np.sum(np.abs(perm[:, m]) >= abs(diff[m])) + 1) / (PERMUTATIONS + 1),
                         "size_matched_z": (diff[m] - null.mean()) / null.std()})
    return rows


def ratio_rows(d: np.ndarray, h: np.ndarray, stat, names: list[str], rng: np.random.Generator) -> list[dict]:
    ratio = stat(d) / stat(h)
    bd, bh, pd_, ph = boot_perm(d, h, stat, rng)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot, perm = bd / bh, pd_ / ph
    rows = []
    for m, name in enumerate(names):
        b = boot[:, m][np.isfinite(boot[:, m])]
        rows.append({"module": name, "n_disease_donors": len(d), "n_healthy_donors": len(h),
                     "disease_value": stat(d)[m], "healthy_value": stat(h)[m], "ratio": ratio[m],
                     "ci_low": np.percentile(b, 2.5), "ci_high": np.percentile(b, 97.5),
                     "perm_p_greater": (np.sum(perm[:, m] >= ratio[m]) + 1) / (PERMUTATIONS + 1),
                     "perm_p_less": (np.sum(perm[:, m] <= ratio[m]) + 1) / (PERMUTATIONS + 1),
                     "perm_p_two_sided": (np.sum(np.abs(np.log(perm[:, m])) >= abs(np.log(ratio[m]))) + 1)
                     / (PERMUTATIONS + 1)})
    return rows


SUBTYPE_ONLY = {"CD16 monocyte (Azimuth PBMC, Hao 2021)", "Calprotectin complex (GO:1990660)"}


def bh(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg q-values (step-up, monotone, capped at 1)."""
    p = p.astype(float)
    order = np.argsort(p.to_numpy())
    ranked = p.to_numpy()[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    out = np.empty(len(p)); out[order] = q
    return pd.Series(out, index=p.index)


def add_fdr(t: pd.DataFrame, family: list[str], pcols: list[str]) -> pd.DataFrame:
    t = t.copy()
    tested = ~((t.cells == "pooled") & t.module.isin(SUBTYPE_ONLY))
    for pc in pcols:
        q = pc.replace("perm_p", "q")
        t[q] = np.nan
        t.loc[tested, q] = t[tested].groupby(family, group_keys=False)[pc].apply(bh)
    return t


def group_keys(spec: dict, available: set) -> list[tuple[str, str, str]]:
    labels = spec["groups"] or sorted({k.split("|")[0] for k in available} - {"ALL"})
    keys = [("pooled", "all", "ALL|all")]
    keys += [(lab, m, f"{lab}|{m}") for lab in labels for m in ("all", "same")]
    return [k for k in keys if k[2] in available]


def main() -> None:
    change, dispersion = [], []
    for spec in DATASETS:
        modules = {k: np.asarray(v) for k, v in json.loads(spec["modules"].read_text()).items()}
        panel = pd.read_csv(spec["panel"], sep="\t")
        usable = np.flatnonzero(panel["observed"].to_numpy(bool)) if "observed" in panel else np.arange(len(panel))
        labels = pd.read_csv(spec["labels"], sep="\t")["label"].astype(str).to_numpy()
        rng = rng_for(spec["dataset"], spec["cohort"], "null")
        null_sets = {n: np.stack([rng.choice(usable, n, replace=False) for _ in range(NULL_DRAWS)])
                     for n in sorted({len(v) for v in modules.values()})}
        dis, hea = donor_gene_effects(spec, spec["disease"]), donor_gene_effects(spec, spec["healthy"])
        cols = [f"{s}__{m}" for s in STEPS for m in modules]
        wd = within_sd(spec, spec["disease"], labels, cols)
        wh = within_sd(spec, spec["healthy"], labels, cols)
        base = {"dataset": spec["dataset"], "cohort": spec["cohort"]}
        for cells, matches, key in group_keys(spec, set(dis) & set(hea)):
            if len(dis[key]) < 3 or len(hea[key]) < 3:
                continue
            tag = (spec["dataset"], spec["cohort"], cells, matches)
            ids = {**base, "cells": cells, "matches": matches}
            change += [{**ids, **r} for r in change_rows(dis[key], hea[key], modules, usable, tag, null_sets)]
            # between-patient SD: donor-mean centred scores, from the same gene-level donor effects
            dd, dh = sorted(dis[key]), sorted(hea[key])
            for s, step in enumerate(STEPS):
                sc = lambda E, ds: np.stack([[E[x][s][ix].mean() - E[x][s].mean() for ix in modules.values()] for x in ds])
                rows = ratio_rows(sc(dis[key], dd), sc(hea[key], dh), sd_, list(modules), rng_for(*tag, step, "between_sd"))
                dispersion += [{**ids, "step": step, "statistic": "between_patient", **r} for r in rows]
                if key in wd and key in wh and len(wd[key]) >= 3 and len(wh[key]) >= 3:
                    c = [f"{step}__{m}" for m in modules]
                    rows = ratio_rows(wd[key][c].to_numpy(), wh[key][c].to_numpy(), mean_, list(modules),
                                      rng_for(*tag, step, "within_sd"))
                    dispersion += [{**ids, "step": step, "statistic": "within_patient", **r} for r in rows]
            print(f"[stats] {spec['cohort']} {cells} {matches}", flush=True)
    change, dispersion = pd.DataFrame(change), pd.DataFrame(dispersion)
    fam = ["dataset", "cohort", "cells", "matches", "step"]
    change = add_fdr(change, fam, ["perm_p_two_sided"])
    dispersion = add_fdr(dispersion, fam + ["statistic"], ["perm_p_greater", "perm_p_less", "perm_p_two_sided"])
    change.to_csv(OUT / "program_change.csv", index=False)
    dispersion.to_csv(OUT / "program_dispersion.csv", index=False)

    # pooled views in the Fig. 5 plot layout
    pooled = change[(change.cells == "pooled") & change.step.isin(STEPS[:4])].rename(
        columns={"step": "component", "size_matched_z": "null_z"})
    disp = dispersion[(dispersion.cells == "pooled") &
                      (((dispersion.step == "between") & (dispersion.statistic == "between_patient")) |
                       ((dispersion.step == "within") & (dispersion.statistic == "within_patient")))].rename(
        columns={"step": "component"})
    for dataset, prefix in (("COVID", "covid"), ("AD", "ad_en")):
        pooled[pooled.dataset == dataset].to_csv(OUT / f"{prefix}_program_heatmap.csv", index=False)
        if dataset == "COVID":
            disp[disp.dataset == dataset].to_csv(OUT / f"{prefix}_component_dispersion.csv", index=False)

    print("[done]")


if __name__ == "__main__":
    main()
