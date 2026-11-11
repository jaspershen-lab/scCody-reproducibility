#!/usr/bin/env python3
"""Figure 2h population-reconstruction category: global mean-expression correlation.

For each model and fit, the generated cells at every external age are compared
with the exact real target rows recorded in the whole-reference prediction
artifact (Pearson over the 2,000 gene means). Writes
manuscript/figure_2/results/mean_expression_correlation_global.tsv; then run
Figure2DEFGH.R.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import scanpy as sc

REPO = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True,
                        help="Whole-reference prediction manifest(s) with an 'artifacts' list of {model, fold, path}")
    parser.add_argument("--h5ad", type=Path, default=REPO / "data/aida_counts_hvg.h5ad")
    parser.add_argument("--out", type=Path, default=REPO / "manuscript/figure_2/results/mean_expression_correlation_global.tsv")
    args = parser.parse_args()
    paths = {}
    for manifest in args.manifest:
        for item in json.loads(manifest.read_text())["artifacts"]:
            paths[(item["model"], item["fold"])] = Path(item["path"])
    assert len(paths) == 15, len(paths)
    adata = sc.read_h5ad(args.h5ad, backed="r")
    records, means = [], {}
    for (model, fold), path in sorted(paths.items()):
        with np.load(path, allow_pickle=True) as a:
            genes = a["gene_names"].astype(str); pred = a["x_pred"]; ages = a["actual_age"]
            idx = (adata.obs_names.get_indexer(a["holdout_obs_names"].astype(str)) if "holdout_obs_names" in a
                   else a["holdout_eval_indices"].astype(int))
        assert np.array_equal(genes, adata.var_names.astype(str).to_numpy())
        assert (idx >= 0).all() and (idx < adata.n_obs).all()
        for age in sorted(np.unique(ages)):
            mask = np.isclose(ages, age); realidx = idx[mask]
            assert np.allclose(adata.obs.iloc[realidx].age.astype(float), age)
            key = tuple(realidx.tolist())
            if key not in means:
                # Preserve the saved target-row sampling/weighting, including repetitions.
                means[key] = np.asarray(adata.X[realidx].mean(axis=0)).ravel()
            r = float(np.corrcoef(means[key], pred[mask].mean(axis=0))[0, 1])
            records.append(dict(model=model, scope="Global", fold=fold, target_age=float(age), mean_expression_corr=r,
                                n_genes=len(genes), n_pred_cells=int(mask.sum()), n_real_rows=len(realidx),
                                source_npz=path.name))
    adata.file.close()
    frame = pd.DataFrame(records)
    assert len(frame) == 90
    frame.to_csv(args.out, sep="\t", index=False)
    print(frame.groupby("model").mean_expression_corr.mean())


if __name__ == "__main__":
    main()
