"""Fit or apply the original PsychAD donor-pseudobulk ElasticNet comparator.

This optional fitting command is separate from figure/table regeneration.
Use --predict-only with saved estimators to produce predictions without fitting.
"""
import argparse
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
from anndata.io import read_elem
from scipy import sparse
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

from sccody_analyses.preprocessing.h5ad_metadata import read_csr_rows


def pseudobulk(path, scope, block_rows=20000):
    with h5py.File(path) as f:
        obs = read_elem(f["obs"])
        genes = read_elem(f["var"]).index.astype(str).to_numpy()
        donors = obs.donor_id.astype(str).to_numpy()
        groups = obs["class"].astype(str).to_numpy() if scope == "cell_type" else np.full(len(obs), "Global")
        keys = pd.MultiIndex.from_arrays([donors, groups])
        codes, unique = pd.factorize(keys, sort=True)
        counts = np.zeros((len(unique), len(genes)), dtype=np.float64)
        n_cells = np.bincount(codes, minlength=len(unique))
        for start in range(0, len(obs), block_rows):
            stop = min(start + block_rows, len(obs))
            assignment = sparse.csr_matrix((np.ones(stop-start), (codes[start:stop], np.arange(stop-start))),
                                            shape=(len(unique), stop-start))
            counts += (assignment @ read_csr_rows(f["layers/counts"], start, stop)).toarray()
    ages = obs.groupby("donor_id", observed=True).age.first()
    meta = pd.DataFrame({"donor_id": unique.get_level_values(0), "population": unique.get_level_values(1), "n_cells": n_cells})
    meta["age"] = meta.donor_id.map(ages)
    library = counts.sum(axis=1, keepdims=True)
    return np.log1p(counts / np.clip(library, 1, None) * 10000.), meta, genes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path, default=Path("data/psychad/psychad_dlpfc__normal.h5ad"))
    parser.add_argument("--disease", type=Path, default=Path("data/psychad/psychad_dlpfc__dementia_Alzheimer_disease.h5ad"))
    parser.add_argument("--holdout", type=Path, default=Path("configs/donors/psychad_healthy_holdout.json"))
    parser.add_argument("--outdir", type=Path, default=Path("artifacts/figure4/ad_clock"))
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()
    holdouts = json.loads(args.holdout.read_text())["holdout_donors"]
    args.outdir.mkdir(parents=True, exist_ok=True)
    model_path = args.outdir / "estimators.joblib"
    estimators = joblib.load(model_path) if args.predict_only else {}
    records = []
    for scope in ["global", "cell_type"]:
        normal, normal_meta, genes = pseudobulk(args.normal, scope)
        disease, disease_meta, disease_genes = pseudobulk(args.disease, scope)
        disease = disease[:, pd.Series(np.arange(len(disease_genes)), index=disease_genes).loc[genes].to_numpy()]
        for group in sorted(normal_meta.population.unique()):
            mask = normal_meta.population.eq(group).to_numpy()
            m = normal_meta.loc[mask].reset_index(drop=True)
            x = normal[mask]
            validation = m.donor_id.astype(str).isin(holdouts).to_numpy()
            key = (scope, group)
            if not args.predict_only:
                scaler = StandardScaler()
                train = scaler.fit_transform(x[~validation])
                model = ElasticNetCV(l1_ratio=[.1, .5, .9], cv=5, n_jobs=1, max_iter=5000, random_state=0)
                model.fit(train, m.loc[~validation, "age"].to_numpy())
                estimators[key] = dict(scaler=scaler, model=model, genes=genes)
            saved = estimators[key]
            order = pd.Series(np.arange(len(genes)), index=genes).loc[saved["genes"]].to_numpy()
            target = disease_meta.population.eq(group).to_numpy()
            for kind, meta, values in [("calibration", m.loc[validation], x[validation]),
                                        ("disease", disease_meta.loc[target], disease[target])]:
                if meta.empty:
                    continue
                prediction = saved["model"].predict(saved["scaler"].transform(values[:, order]))
                records.append(meta[["donor_id", "population", "age"]].assign(pred_age=prediction,
                    projection_kind=kind, scope=scope, cohort="PsychAD_AD"))
    if not args.predict_only:
        joblib.dump(estimators, model_path)
    pd.concat(records, ignore_index=True).to_csv(args.outdir / "donor_predictions.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
