# Figure 2

The five methods are scCody-G, scCody-CT, scNODE, scDiffEq and ARTEMIS. Panels b and d–h use corrected data and current selected checkpoints. Panel b retains six external ages; all DE panels use ages 21/31/41/51/61, paired over three folds (15 values per model and side).

- b: mean-expression Pearson r, global and B/Myeloid/NK/T. Displayed axis 0.80–1.00; error bars and labels remain unclipped.
- d: DE Pearson r versus young/old reference.
- e: top 5% (100 genes; solid) and top 10% (200; hatched extension) overlap from the 2,000-gene panel. Stars test top 10%; both tests are saved.
- f: DE sign concordance at reference absolute-logFC thresholds 0/0.25/0.5.
- g: equal young/old DE Pearson r average by lineage.
- h: ranks for population reconstruction, DE effect recovery, top-DE recovery, DE direction recovery and cell-type robustness. All DE scores use the same reference and five-age set as panels d–g. Overall rank averages these five category ranks equally. The donor-aware category is removed; raw component scores and ranks are saved.

All DE panels use matching normalized-half versus independently fitted donor-pseudobulk-half controls. Target donors are disjoint and share the fixed development reference. Up to 500 unique partitions per age are available (10 at age 51 and 462 at age 61). Panel f uses 500 balanced resampling draws for the band and the exact equal-age control mean. Figure descriptions remain unchanged. Panels a/c retain the schematics.

To rebuild the tables after population evaluation and donor-disjoint controls,
run `python manuscript/figure_2/scripts/Figure2B.collect_mean_expression.py`.
It reads `artifacts/population_evaluation/`; no private checkout is used.
