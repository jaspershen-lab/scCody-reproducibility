# Figure 4

Panels a/c/d/f redraw from the bundled `results/` tables; b/e use bundled schematics. Redraw the complete manuscript with `Rscript manuscript/render_main_figures.R`. This uses the recorded results and performs no model fitting, selection or inference.

The commands below rebuild scientific inputs when expression data and existing model/clock outputs are available. Run them from the repository root. They require only scCody and scCody-reproducibility. Existing predictions and residual moments can be reused directly; the table builders do not invoke fitting or checkpoint selection.

## COVID and PsychAD inputs

`python -m sccody_analyses.preprocessing.prepare_covid --input-dir data/raw/covid` writes the three case/control files expected by the disease recipes and clocks. `preprocessing.split_psychad` writes `data/psychad/ad_case_control.h5ad` with condition labels. See the root README for data access and preprocessing.

## Panel a: donor age displacement

To regenerate predictions with the existing checkpoints:

```bash
for cohort in Ren COMBAT Stephenson; do
  for scope in global cell_type; do
    for fold in 0 1 2; do
      python -m sccody_analyses.figure4.project --cohort "$cohort" --scope "$scope" --fold "$fold"
    done
  done
done
python -m sccody_analyses.figure4.project --cohort AD --scope global
python -m sccody_analyses.figure4.project --cohort AD --scope cell_type
```

The COVID models use their recorded training donors, sex-matched source pools and 19–77 candidate grid. PsychAD uses its epoch-5 models, training references, healthy holdout targets and 19–89 grid. `configs/figure4/psychad_sources.json` supplies the recorded source ages per sex, including the four-source female Immune population. These commands only run inference and save donor predictions under `artifacts/figure4/projections/`.

Use saved predictions to build age-displacement inputs:

```bash
python -m sccody_analyses.figure4.tables age-shifts \
  --predictions artifacts/figure4/projections/*.csv
```

Predictions contain cohort, scope, population, donor ID, age, predicted age and condition. COVID predictions are averaged over fits before analysis. Effects subtract the mean healthy-reference acceleration within ±5 years of each case. PsychAD retains its original minimum of three nearby calibration donors; COVID uses available cohort controls. The original bootstrap settings are retained: PsychAD 2,000 draws/seed 20260903; COVID 10,000 draws/seed 20261027 per cohort and scope. Outputs are `age_displacement_summary.csv` and `covid_age_displacement_donor_effects.tsv`.

## Panel c: exact healthy-age gaps

The original signed-alignment and orthogonal-slope inputs use different recorded samples. Preserve both:

```bash
python -m sccody_analyses.figure4.healthy --seed 20260921 \
  --outdir artifacts/figure4/healthy/signed
python -m sccody_analyses.figure4.healthy --seed 20260919 \
  --outdir artifacts/figure4/healthy/orthogonal
python -m sccody_analyses.figure4.tables healthy-relationships \
  --healthy-pairs artifacts/figure4/healthy/signed/*_healthy_pairs.tsv \
  --orthogonal-pairs artifacts/figure4/healthy/orthogonal/*_healthy_pairs.tsv
```

The inference command uses same-sex healthy holdout donor pairs at exact gaps of 1–15 years, 256 OT cells and the existing PsychAD models. `--groups` permits a subset of populations. The table builder writes the signed-pair input and the descriptive orthogonal-RMS OLS slopes with 95% intervals. The original two samples remain bundled in their plotted form: signed pair moments and the orthogonal slope summary.

## Panels d/f: signed alignment and orthogonal components

Run the existing `orthogonal_residual_variance` extract/decompose workflow and the `orthogonal_variance.summarize`/`panel_f` commands from the root README, or reuse their completed outputs. Then:

```bash
python -m sccody_analyses.figure4.tables replay
```

This combines the scientific per-pair residual moments with the supplied donor-pair definitions. It writes only `signed_aligned_rms_pair_replay_all.tsv.gz`: donor/group/fold identifiers, ages and the moments used in panel d. Signed aligned RMS is `inner_product / sqrt(aging_energy)`. Per-pair values are averaged within donor before the equal-cohort COVID aggregation. AD H10 target donors remain restricted to the healthy holdout list. No metadata JSON or audit sidecars are required.

Copy the two orthogonal summary CSVs to `results/orthogonal_residual_variance/` as described in the root README. Panel f retains the within-cohort energy decomposition and equal cohort weighting.

## Panel a: clock comparison

Use the existing PsychAD ElasticNet donor predictions and saved COVID per-fold metacell predictions:

```bash
python -m sccody_analyses.figure4.tables clocks \
  --ad-clock-predictions artifacts/figure4/ad_clock/donor_predictions.tsv \
  --clock-root artifacts/clocks --covid-input-dir data/covid
```

The AD table uses the original ±5-year donor effects and mean ±1.96 standard-error intervals. COVID predictions are averaged across the three fits within donor and lineage; the saved all-lineage donor aggregate is not substituted for each lineage. The outputs are `ad_elasticnet_forest_plot_data.csv` and `covid_elasticnet_age_acceleration_donor_predictions.tsv`.

For a future reproduction without saved AD clock predictions, `python -m sccody_analyses.figure4.ad_clock` explicitly fits the original donor-pseudobulk ElasticNetCV comparator and retains its estimators. Subsequent `--predict-only` calls reuse them. This optional fitting command is separate from all table and plotting commands.

Every table command accepts `--outdir` to review rebuilt inputs in another directory before replacing the bundled results.
