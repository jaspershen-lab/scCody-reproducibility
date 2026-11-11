# Figure 3

Current selected checkpoints; 71 external donors for global localization. Each donor prediction averages three folds. Global MAE/Pearson intervals bootstrap donors 2,000 times. Paired MAE tests and Williams correlation tests are refreshed against the best dynamics comparator.

- c/d: all five dynamical models and corrected Lasso/ElasticNet/tAge clocks. tAge uses its saved donor aggregate before the fold ensemble.
- e: T, Myeloid, DC, B and NK comparator PCA-KNN readout, with the retained DC filter-first fallback. DC targets use at least five cells across all models; other lineage cutoffs remain unchanged. Rare-lineage filtering precedes the source-cell cap; every prescribed sex-matched source estimate is required. Coverage is recorded in the source tables. Bundled tables retain the saved results; regenerated reports use the shared five-cell DC cutoff.
- f–h: current scCody-CT annual programs for T, Myeloid, B, NK and DC at ages 31–68 and recomputed corrected observed expression. Automatic y-axis limits include all intervals within the displayed age range.

Panels a and b use bundled artwork in `figure_files/schematics/a.png` and `b.png`. Panel b is cropped from the supplied `Figure_3_updated.svg` and embedded in the same way as panel a; it is not regenerated from donor predictions. Panels c–h retain their generated plots and existing labels and positions.

To rebuild these tables after model reports, run
`python manuscript/evaluate_rare_cell_types.py`, then
`Rscript manuscript/figure_3/scripts/Figure3CD.assemble_localization.R`.
This uses the public report directories, DC comparator profiles and clock
refits described in the repository README. Plot redrawing itself uses only
the bundled tables.
