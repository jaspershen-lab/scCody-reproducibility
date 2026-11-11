# sccody_analyses

Workflows and comparator engines for the scCody manuscript, covering AIDA, the PsychAD MSSM DLPFC cohort and COVID-19 analyses. Install with `pip install -e .` from the repository root and run modules from there (`python -m sccody_analyses.<name>`). The top-level README gives the run order, and `manuscript/README.md` maps each figure panel to its script.

| Module | Role |
|---|---|
| `preprocessing/prepare_aida.py`, `preprocessing/split_aida_age_holdout.py` | Full-gene CP10K/log1p on the frozen development panel; exact-age external and optional internal splits |
| `preprocessing/prepare_psychad.py`, `preprocessing/split_psychad.py` | PsychAD MSSM DLPFC panel, exact-age filtering and AD case/control split |
| `preprocessing/prepare_*covid*_h5ad.py` | Ren, COMBAT and Stephenson reindexed onto the AIDA panel, with unassayed genes imputed from the healthy AIDA mean |
| `run_pipeline.py`, `selection.py` | Fit, internal selection, `attach` of the recorded AIDA-trained weights, external localization and fit ensembles |
| `benchmarks/` (`python -m sccody_analyses.benchmarks.run {scnode,scdiffeq,artemis} @args`) | scNODE, scDiffEq and ARTEMIS training and reports |
| `refit_supervised.py`, `clocks.py`, `supervised.py` | Cutoff-5 clocks refit with the selected penalties and the age-68 fitting cap |
| `extract_residual_fields.py` | Exact-OT replay of recorded disease and +10 y healthy pairs; per-cell latent fields |
| `decode_residual_components.py`, `decode_*_program_scores.py` | Shared/between/within decomposition and decoded program scores |
| `analyze_ad_*.py`, `analyze_covid_*.py` | AD programs and EN subclasses; COVID program associations and spread |
| `orthogonal_residual_variance.py`, `orthogonal_variance/` | Pair and donor residual moments, the D vs H10 summary, and the decomposition panel data |
| `evaluate_populations.py`, `donor_pseudobulk_de.py`, `metrics/` | Reconstruction, distribution and DE recovery metrics, edgeR and donor-disjoint controls |
| `annual_programs.py` | Annual pathway trajectories from the recorded source cells and frozen controls |

The selected epochs are G 30/24/28 and CT 29/26/30 for the AIDA-trained models, and 5 for the models trained on healthy PsychAD MSSM donors. Refit with `run_pipeline --phase train`, then report with `--phase report`. The shared scorer rolls pooled youngest-age training cells to each internal donor’s age; the final selected weights and selection record are saved without epoch histories. Refitting can select different epochs.
