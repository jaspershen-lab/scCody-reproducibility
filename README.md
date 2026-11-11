# scCody-reproducibility

Study code for scCody: healthy-aging models, AIDA donor-age localization benchmarks, population reconstruction and differential-expression recovery, immune-aging programs, and Alzheimer's disease and COVID-19 displacement analyses. The model implementation is provided by the sibling [scCody](../scCody) package.

The manuscript uses scCody-G and scCody-CT models trained on AIDA for blood aging and COVID-19 analyses, and models trained on healthy donors from the PsychAD MSSM DLPFC cohort for Alzheimer's disease analyses. The plotted model selections are listed in [manuscript/models/README.md](manuscript/models/README.md).

## Current manuscript run

The main figures use full-gene normalization, 2,000 development-only HVGs and internal holdout gene-space Sinkhorn selection. The final plotting tables are bundled in `manuscript/figure_*/results/`; [manuscript/README.md](manuscript/README.md) lists the plotting commands. scNODE is the main comparator shown in these figures.

## Redraw the current figures

All inputs for the retained main-figure data panels are bundled in `manuscript/figure_*/results/` and `configs/`. With R and Cairo graphics support installed, run:

```bash
Rscript manuscript/render_main_figures.R --install-deps
```

This installs any missing CRAN plotting packages and redraws the panels. On subsequent runs, omit `--install-deps`. The assembled-figure step uses Python 3 standard-library modules and bundled schematic crops; checkpoints, expression matrices, upstream artifacts and Slurm are not needed. Use `--panels-only` for the R-only panel route. The plotting styles use Arial; make that font available to R for matching typography. Rendered panels go to `manuscript/figure_*/figure_files/panels/`; a few scripts also refresh derived summaries in `results/`. The command redraws data panels and regenerates the five assembled SVG/PDF/PNG figures.

The complete plot-only command was checked with R 4.5.2 in an isolated copy containing `manuscript/` and the two plotting inputs `configs/analyses/aging_program_biological_expectations.tsv` and `configs/donors/psychad_healthy_holdout.json`.

The bundled results are the final plotting tables and small schematic assets. Redrawing plots does not require intermediate runs, source hashes or training histories.

## Repository layout

```
scCody-reproducibility/
├── manuscript/               figures, panel scripts and the tables they plot
│   ├── figure_1/ … figure_5/
│   │   ├── figure_files/     assembled figure; regenerated main panels/
│   │   ├── scripts/          one script per panel, named Figure<N><panels>
│   │   └── results/          tables behind each panel
│   ├── models/               pointer to the trained checkpoints
│   ├── significance.R        shared significance marks
│   └── README.md             panel-by-panel index
├── sccody_analyses/          installable workflows and comparator engines
│   ├── preprocessing/        AIDA, PsychAD MSSM and COVID inputs
│   ├── benchmarks/           scNODE, scDiffEq and ARTEMIS
│   └── metrics/, orthogonal_variance/, *.py
└── configs/                  donor splits, model configs, pair tables, gene programs
```

The panel index is in [manuscript/README.md](manuscript/README.md). Each figure's plots re-render from its bundled `results/` tables without the expression data. The workflows below run preprocessing, model fitting and scientific analyses on supplied study data.

| Module or directory | Result |
|---|---|
| `sccody_analyses.preprocessing`, `configs/aida*.yaml`, `sccody_analyses.run_pipeline` | AIDA preparation, three fits per variant (ages 19–77), internal selection and external donor-age localization (Fig. 3) |
| `sccody_analyses.benchmarks`, `configs/benchmarks/` | scNODE, scDiffEq and ARTEMIS on the same donor splits and source ages (Fig. 2, 3) |
| `sccody_analyses.refit_supervised`, `configs/supervised.json` | Cutoff-5 Lasso, ElasticNet and tAge clocks, plus the COVID clocks (Fig. 3, 4) |
| `sccody_analyses.evaluate_populations`, `sccody_analyses.metrics` | Reconstruction, distribution and DE recovery, real-count edgeR and donor-disjoint controls (Fig. 2) |
| `sccody_analyses.annual_programs` | Immune-aging pathway trajectories (Fig. 3f–h) |
| `sccody_analyses.extract_residual_fields`, `decode_*`, `analyze_*` | Aligned/orthogonal displacement, shared/between/within effects, decoded programs and AD EN subclasses (Fig. 5) |
| `sccody_analyses.orthogonal_residual_variance`, `orthogonal_variance` | Orthogonal energy, disease vs +10 y healthy aging, for every AD and COVID group (Fig. 4f) |
| `sccody_analyses.figure4` | Disease localization, age-window effects, healthy-gap relationships, clock comparisons and signed-alignment tables (Fig. 4) |

## Environment and inputs

Keep the two checkouts as siblings. Run every command below from `scCody-reproducibility`:

```bash
pip install -r requirements.txt   # installs ../scCody and sccody_analyses
```

Comparator-specific Python requirements are in `sccody_analyses/benchmarks/requirements-{scnode,scdiffeq,artemis}.txt`; install the matching file from this checkout. ARTEMIS source and its upstream license are bundled in `sccody_analyses/benchmarks/vendor/artemis`.

The preprocessing and analysis environment needs Python 3.10+, Scanpy, scikit-misc, scikit-learn, statsmodels and GSEApy. edgeR comparisons need R with Bioconductor `edgeR` and `statmod`. Plots use `ggplot2`, `dplyr`, `readr`, `tidyr`, `purrr`, `stringr`, `patchwork`, `ggrepel`, `svglite`, `ragg` and `jsonlite`. Model fitting, field extraction and comparator inference use CUDA in the recorded commands. Slurm is optional; no scheduler jobs are submitted automatically.

| Dataset | Access | Required local input |
|---|---|---|
| AIDA Phase 1 freeze v2 | [AIDA resource and access instructions](https://prabhakarlab.github.io/AIDA.html), [CELLxGENE collection](https://cellxgene.cziscience.com/collections/ced320a1-29f3-47c1-a735-513c7084d508) | Raw counts and donor/cell metadata |
| PsychAD MSSM DLPFC cohort (`MSSM_Cohort`) | [PsychAD resources](https://psych-ad.org/research/), [CELLxGENE collection](https://cellxgene.cziscience.com/collections/84ce6837-548d-4a1f-919f-0bc0d9a3952f) | Dataset `37a17b78-4864-4a42-b67b-31c00962795a`, H5AD version `0e853475-e298-4b09-881a-ed0b60d5a8c9`; exact-age normal and dementia/Alzheimer groups |
| Ren COVID-19 | [GEO GSE158055](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE158055) | PBMC counts, patient IDs, ages, sex, severity, sampling day and original cell annotations |
| COMBAT COVID-19 | [Study data-availability statement](https://doi.org/10.1016/j.cell.2022.01.012) | Released COMBAT CITE-seq object and clinical metadata |
| Stephenson COVID-19 | [Study data-availability statement](https://www.nature.com/articles/s41591-021-01329-2), accession E-MTAB-10026 | Released multimodal PBMC object and clinical metadata |

Here, PsychAD refers to the Mount Sinai (MSSM) DLPFC cohort released as `MSSM_Cohort` on CELLxGENE. The AD analyses use its exact-age healthy and dementia/Alzheimer subsets; they do not combine the MSSM, HBCC and RADC cohorts into the full PsychAD atlas. The collection's [public metadata](https://api.cellxgene.cziscience.com/curation/v1/collections/84ce6837-548d-4a1f-919f-0bc0d9a3952f) identifies the dataset and H5AD version above.

Public expression matrices do not always include every clinical or donor field. Request the required metadata through the originating study's access process, or through the AD Knowledge Portal/Synapse for PsychAD, and keep the donor identifiers used by the supplied splits. No datasets are bundled or downloaded. COMBAT and Stephenson ages are released decade-bin midpoints.

MSigDB sets require access under [MSigDB's terms](https://www.gsea-msigdb.org/gsea/msigdb/). `configs/programs/curated/module_sources.tsv` records each program's citation and selection rule.

The curated programs in `configs/programs/curated/` are shipped and do not need rebuilding. To rebuild them, `sccody_analyses.build_curated_program_modules` reads:

- `<modules-dir>/gmt/`: MSigDB `h.all.v2026.1`, `c2.cp.reactome.v2026.1`, `c5.go.bp.v2026.1` and `c5.go.cc.v2024.1` (`.Hs.symbols.gmt`)
- `<downloads>/msig/`: MSigDB `c2.cgp.v2026.1` and `c8.all.v2026.1` (`.Hs.symbols.gmt`)
- `<downloads>/mathys2023_decontx/`: the 15 excitatory-neuron `Exc_*_gpath.csv` tables from [mathyslab7/ROSMAP_snRNAseq_PFC](https://github.com/mathyslab7/ROSMAP_snRNAseq_PFC/tree/main/Results/DecontX_RUVr_Differential_gene_expression_analysis/muscat_DecontX/Results/gpath) (DecontX/RUVr muscat results for global AD pathology; spaces in file names replaced by underscores). The aggregate `Exc_IT_gpath.csv` is skipped, leaving 14 subtypes.

```bash
python -m sccody_analyses.build_curated_program_modules --modules-dir configs/programs --downloads <downloads>
```

The Mathys AD modules keep genes significant (per-subtype BH FDR < 0.05) in the same direction in ≥ 7 of 14 subtypes (`--mathys-min-subtypes`) and in no subtype in the opposite direction. This is stricter than the ≥ 3-subtype consensus signature in Mathys et al. (2023). Programs use only panel genes with `observed == True`. A program is kept if it has ≥ 5 panel genes (`--min-genes`); the calprotectin programs are exempt.

## Models, panel and splits

Download `scCody_weights.zip` from [Figshare](https://figshare.com/articles/software/scCody_Model_Weight/34951348?file=71249662). With the ZIP in
the parent directory of the two sibling checkouts, run from `scCody-reproducibility`:

```bash
unzip ../scCody_weights.zip -d ..
```

This places the six AIDA and two PsychAD checkpoints at the paths below.
The complete archive layout is in [manuscript/models/README.md](manuscript/models/README.md).

| Training dataset / model | Checkpoints | Selected epochs, fits 0/1/2 | Use |
|---|---|---|---|
| AIDA / scCody-G | `checkpoints/AIDA_scCody-G_fit{0,1,2}.pt` | 30 / 24 / 28 | AIDA localization; COVID global fields |
| AIDA / scCody-CT | `checkpoints/AIDA_scCody-CT_fit{0,1,2}.pt` | 29 / 26 / 30 | AIDA localization; COVID cell-type fields (Fig. 5) |
| PsychAD (MSSM) / scCody-G and scCody-CT | `checkpoints/PsychAD_scCody-{G,CT}_fit0.pt` | 5 | AD fields (Fig. 5) |

The AIDA-trained models listed here use the same ordered 2,000-gene panel (`configs/aida_genes.tsv`, identical to `configs/programs/COVID_panel.tsv`), a GMVAE with latent dimension 50, hidden layers 512/512 and 20 components, and a FlowMap with hidden layers 256/256. The current AIDA run uses ages 19–77 without an upper fitting-age cap, ages divided by 100, sex-stratified donor trajectories with 24 nodes and bidirectional pairs, and unit weights for gene OT, latent OT and endpoint consistency. Per-fold seeds are 42/43/44. The six `configs/aida*fit*.yaml` files specify the public training settings. Training retains its selected model and reusable representation checkpoint.

`configs/donors/` holds the exact internal and external donor sets. The 71 external donors are at ages 21/31/41/51/61/68. Internal holdouts are ages 22/30/40/50/60/67, 23/32/42/52/62/66 and 24/33/43/53/63/65 for fits 0/1/2. Source ages are sex-matched pools at 19/33/44/55/66 (fit 0), 19/33/44/55/67 (fit 1) and 19/32/44/55/67 (fit 2). Current localization searches ages 19–77 and never reads the target's chronological age.

Keep each dataset's gene panel, prepared inputs, predictions and module indices aligned. `configs/programs/*_modules.json` index `AD_panel.tsv` and `COVID_panel.tsv` by column.

## Reproduction order

### 1. Prepare the data

Prepare AIDA directly from its full-gene raw-count H5AD:

```bash
python -m sccody_analyses.preprocessing.prepare_aida \
  --input data/raw_aida.h5ad --outdir data --panel configs/aida_genes.tsv
```

This streams `raw.X`, applies full-library CP10K and log1p, and selects the bundled 2,000-gene panel fitted on the 554 development donors. Reusing this panel keeps inputs aligned with the supplied checkpoints. Use `--count-source X` or `--count-source layer:<name>` if raw counts are stored elsewhere. The outputs are `aida_counts_hvg.h5ad`, `aida_age_holdout_train.h5ad` (development) and `aida_age_holdout_test.h5ad` (external). Source cell order is preserved.

Splits use exact ages: 21/31/41/51/61/68 are external; all other ages are development. The per-fit internal ages listed in `configs/donors/internal_<k>.json` are withheld within development when fitting each model. No old prepared AIDA templates or development-donor TSV are required. To materialize a particular internal split, `split_aida_age_holdout` accepts `--internal-ages`, for example `22 30 40 50 60 67` for fit 0, and writes train, validation and external test files to the requested `--outdir`.

The PsychAD MSSM input uses its own full-library preprocessing. Preparation retains only exact `N-year-old stage` age labels; splitting selects the exact disease labels `normal` and `dementia || Alzheimer disease`:

```bash
python -m sccody_analyses.preprocessing.prepare_psychad --input data/raw_psychad.h5ad --outdir data/psychad
python -m sccody_analyses.preprocessing.split_psychad --input data/psychad/DLPFC_neurodegen_preprocessed.h5ad --outdir data/psychad
```

The PsychAD split command also writes `data/psychad/ad_case_control.h5ad`, with the healthy and disease condition labels used downstream. Prepare the three COVID cohorts directly from their released raw-count H5ADs:

```bash
python -m sccody_analyses.preprocessing.prepare_covid --input-dir data/raw/covid
```

The input directory contains `COVID19_ALL.h5ad`, `COMBAT-CITESeq-DATA.h5ad` and `Stephenson_E-MTAB-10026_haniffa21.processed.h5ad`. The command normalizes full gene-expression libraries before panel selection, excludes antibody features, and fills unassayed genes with the development AIDA means. It writes expression, counts and measured-gene flags to `data/covid/{ren,combat,stephenson}_covid_case_control_aida2000_mean_imputed.h5ad`, the exact paths used by the disease recipes, COVID clocks and Figure 5. Stephenson cases and controls are prepared together with the original per-condition sampling. No old prepared templates are required. Use `--cohort ren`, `combat` or `stephenson` for one cohort, and `--aida-reference` to supply another prepared development input.

### 2. scCody models and AIDA localization

`configs/aida.json` defines six tasks: global fits 0–2, then CT fits 0–2. To use the recorded AIDA-trained weights, attach the healthy training references and report:

```bash
python -m sccody_analyses.run_pipeline configs/aida.json --phase attach --task 0
python -m sccody_analyses.run_pipeline configs/aida.json --phase report --task 0
```

The current selected checkpoints minimize internal holdout gene-space Sinkhorn; scCody-G epochs are 30/24/28 and scCody-CT 29/26/30. Training selects its checkpoint internally from pooled youngest-age sources, with the shared donor-level scorer in `sccody.selection`. The supplied weights and recorded selected epochs are used directly; no training or checkpoint selection runs during attachment, reporting or plotting. For an optional future refit, `--phase train` followed by `--phase report` both use `fit/model.pt`. `fit/selection.json` records only the selected criterion, epoch and score. After all six reports:

```bash
python -m sccody_analyses.run_pipeline configs/aida.json --phase summarize
```

Outputs go under `artifacts/aida/<arm>/fit_<k>/`: `report/global/predictions.csv`, the `min5`/`min100` cell-type readouts, and `report/truth.csv`.

### 3. Comparators

The recorded argument files under `configs/benchmarks/` take one argument per line. Training uses the internal holdout donors in `configs/donors/internal_<k>.json`; reports use `configs/donors/external_<k>.json` and fixed source ages from `fit<k>_fixed_sources_bidirectional.json` (scNODE, ARTEMIS) or `fit<k>_fixed_source_age19_forward.json` (scDiffEq).

```bash
python -m sccody_analyses.benchmarks.run scnode @configs/benchmarks/scnode_fit0_train.args
python -m sccody_analyses.benchmarks.run scnode @configs/benchmarks/scnode_fit0_report.args
```

Repeat this for `scdiffeq` and `artemis` and for fits 1/2. Each report also exports
`artifacts/predictions/<method>_fit<k>.npz` for population evaluation. Run the
DC filter-first profile for each method and fit:

```bash
python -m sccody_analyses.benchmarks.run scnode @configs/benchmarks/scnode_fit0_dc_report.args
```

The Slurm wrapper includes this downstream report. All localization profiles
use the 19–77 age grid; scDiffEq retains its forward-only simulation from age 19.
The standard profiles include the train-only PCA/KNN lineage readout. scNODE uses encoder 50,50, no decoder hidden layer, drift 50,50 and latent weight 0.197. scDiffEq uses drift 2048, diffusion 128, ratio target 1.0 and learning rate 3·10⁻⁴; its report uses the fixed first age forward source. ARTEMIS uses an age/100 axis with 100 SDE steps, latent dimension 50 and VAE β 10⁻³. Current comparator training selects by internal `holdout_L_gene`. HPO (`--metric-profile hpo_objective`) minimizes the same donor-averaged gene-space Sinkhorn score on internal holdout donors. Reports write `artifacts/comparators_final/<method>/fold_<k>/report/*_calibration_projection.csv`.

**Current comparator inputs.** scNODE, scDiffEq and ARTEMIS checkpoints are selected by internal holdout_L_gene. The main scNODE uses Euler with step_size=None (CLI 0), selected epochs 5/6/11. scDiffEq selects epochs 500/500/475 and ARTEMIS phases 19/17/16. Benchmark argument files document the method settings; [sccody_analyses/benchmarks/README.md](sccody_analyses/benchmarks/README.md) describes how each comparator was trained, selected and evaluated. The plotting command reads the bundled final tables and performs no training.

### 4. Supervised clocks

```bash
for model in lasso_global elasticnet_global tage lasso_cell_type elasticnet_cell_type; do
  python -m sccody_analyses.refit_supervised --model $model \
    --train data/aida_age_holdout_train.h5ad --full data/aida_counts_hvg.h5ad \
    --settings configs/supervised.json \
    --fold-manifest-template 'configs/donors/internal_{fold}.json' \
    --external-manifest-template 'configs/donors/external_{fold}.json' \
    --outdir artifacts/clocks/$model
done
```

These refit the selected penalties: Lasso α 0.501; ElasticNet α 0.501 with l1 ratio 0.99; tAge α 0.020 with l1 ratio 1, on per-cell-type donor pseudobulks. AIDA localization clocks use pseudobulks with ≥ 5 cells. All model refits use the eligible development ages without an upper fitting-age cap, retaining the internal and external donor exclusions. The bundled clock predictions remain the saved results from the earlier fitting protocol. The separate `covid_clock_global` and `covid_clock_cell_type` configurations use their recorded 100-cell training cutoff and penalties α 0.0501/0.0316 with l1 ratio 1. Their optional refit adds `--covid-input-dir data/covid`. Existing saved clock predictions can be used directly by the Figure 4 builders.

Each clock fit retains its estimator and one `preprocessing.npz` containing the ordered genes, fitted normalization, scaling and healthy-reference values.

### 5. Figure 3 localization tables

```bash
python manuscript/evaluate_rare_cell_types.py
Rscript manuscript/figure_3/scripts/Figure3CD.assemble_localization.R
Rscript manuscript/figure_3/scripts/Figure3C.R
Rscript manuscript/figure_3/scripts/Figure3D.R
Rscript manuscript/figure_3/scripts/Figure3D.test_localization.R
```

Each model's three fit predictions are averaged per donor, and MAE and Pearson r are computed once across the 71 donors with 2,000-draw donor-bootstrap intervals. The panel scripts are listed in [manuscript/figure_3/README.md](manuscript/figure_3/README.md).

### 6. Disease displacement and Figure 5

**Fields and decoded programs.** `configs/disease_fields.json` lists 38 recipes: the AD EN field and the Ren, COMBAT and Stephenson DC and Myeloid fields, each as disease and +10 y healthy. Pair tables are in `configs/pairs/`. Write each recipe's fields to `artifacts/residual_programs/fields/<cohort>[/fold<k>]`:

```bash
python -m sccody_analyses.extract_residual_fields --mode disease \
  --pairs configs/pairs/AD/EN.tsv --model checkpoints/PsychAD_scCody-CT_fit0.pt \
  --target-h5ad data/psychad/ad_case_control.h5ad --scope cell_type --fold 0 --groups EN \
  --cell-type-key class --max-ot-cells 256 --seed 20260902 --outdir artifacts/residual_programs/fields/AD
```

Then decode, using `--fields-dir artifacts/residual_programs/fields/<cohort>[/fold<k>]` and the model of that fold:

| Step | Command | Output under `artifacts/residual_programs/` |
|---|---|---|
| Components | `sccody_analyses.decode_residual_components` | `decoded/<cohort>/fold<k>/<group>/` (AD: `decoded/AD/<group>/`) |
| Group labels | `python manuscript/figure_5/scripts/make_program_group_labels.py` | `group_labels/<dataset>.tsv` |
| Program scores | `sccody_analyses.decode_component_program_scores --labels group_labels/<dataset>.tsv`, plus `decode_within_program_scores` and `decode_total_program_scores` | `decoded_programs/<cohort>/fold<k>/<group>/` (AD: `decoded_programs/AD/<group>/`) |
| AD subclasses | `sccody_analyses.analyze_ad_subclass_residual_readout --panel configs/programs/AD_panel.tsv` | `analysis/AD_EN_subclass/` |

Use `--modules configs/programs/curated/{AD,COVID}_modules.json` and `--group <group>` or `<group>_healthy`.

**Orthogonal residual variance.** `configs/orthogonal_variance.json` holds 54 recipes covering every AD group and every COVID cohort × fold × scope, each as disease and +10 y healthy. AD healthy targets are restricted to `configs/donors/psychad_healthy_holdout.json`.

```bash
python -m sccody_analyses.orthogonal_residual_variance extract      # or --task <i>
python -m sccody_analyses.orthogonal_residual_variance decompose
python -m sccody_analyses.orthogonal_variance.summarize --ad-root artifacts/orthogonal_residual_variance/AD \
  --covid-root artifacts/orthogonal_residual_variance/COVID \
  --outdir artifacts/orthogonal_residual_variance/summary --bootstrap-repeats 10000 --seed 20260921
python -m sccody_analyses.orthogonal_variance.panel_f --ad-root artifacts/orthogonal_residual_variance/AD \
  --covid-root artifacts/orthogonal_residual_variance/COVID \
  --outdir artifacts/orthogonal_residual_variance/summary
```

Each component is reported as `100·(D_k − H10_k)/H10_total` with a 10,000-draw target-donor bootstrap. COVID values are decomposed within each cohort and then averaged with equal cohort weight.

Comparisons with missing decomposition files are skipped. The panel-f builder reads the decomposition files directly.

Copy `orthogonal_residual_variance_D_H10_all.csv` and `orthogonal_residual_variance_panel_f_data.csv` from `artifacts/orthogonal_residual_variance/summary/` to `manuscript/figure_4/results/orthogonal_residual_variance/` for Fig. 4d and 4f.

**Figure 4.** [manuscript/figure_4/README.md](manuscript/figure_4/README.md) gives the public commands for disease localization with the existing weights, age-window effects, the two recorded healthy-gap samples, saved clock predictions and signed-alignment tables. COVID field recipes use the plotted 384-cell OT samples and current donor pairs. Table building and plotting do not train or select models.

**Figure 5.** The table builders and plots are listed in [manuscript/figure_5/README.md](manuscript/figure_5/README.md). They read `artifacts/residual_programs/` and `artifacts/orthogonal_residual_variance/summary/` and write `manuscript/figure_5/results/`.

### 7. Reconstruction and annual programs

`configs/population_evaluation.json` uses the selected models from step 2 and `data/aida_counts_hvg.h5ad`:

```bash
python -m sccody_analyses.evaluate_populations configs/population_evaluation.json --phase prepare
python -m sccody_analyses.evaluate_populations configs/population_evaluation.json --phase reference
python -m sccody_analyses.evaluate_populations configs/population_evaluation.json --phase generate --task 0
python -m sccody_analyses.evaluate_populations configs/population_evaluation.json --phase metrics --task 0
python -m sccody_analyses.evaluate_populations configs/population_evaluation.json --phase controls
```

`prepare` builds the development reference indices and per-fit target templates
from the donor splits. Target sampling is shared with the comparators; the
reference uses development donors only. Run `generate` for all six tasks,
`metrics` for all 15 tasks, and `controls` once. The controls compare normalized
expression in one donor half against independently fitted edgeR in the other,
using up to 500 unique donor partitions per age. Then assemble Figure 2 inputs:

```bash
python manuscript/figure_2/scripts/Figure2B.collect_mean_expression.py
```

Annual programs reuse the recorded held-out source cells and frozen control genes in `configs/annual_programs/`:

```bash
python -m sccody_analyses.annual_programs export --scope cell_type --fold 0 \
  --model checkpoints/AIDA_scCody-CT_fit0.pt \
  --base configs/annual_programs --input data/aida_counts_hvg.h5ad --outdir artifacts/programs --device cuda
```

Export every fit before running `score`. Fig. 3f–h uses the cell-type scope; copy the score table to `manuscript/figure_3/results/annual_programs_cell_type/` and run `Figure3FGH.R`.

The retained plotting scripts render main Figures 1–5. Supplementary figures and tables are excluded from this release.

## Artifact availability

The eight scCody weights are distributed as `scCody_weights.zip` on [Figshare](https://figshare.com/articles/software/scCody_Model_Weight/34951348?file=71249662) and excluded from Git. Extracting the archive places all weights in `checkpoints/`. They contain no healthy-reference embeddings, which is why `run_pipeline --phase attach` exists. Expression data, fields, decoded outputs and prediction arrays are not bundled. The bundled inputs are the figure tables under `manuscript/figure_*/results/`, the donor splits, source-age manifests, pair tables (`configs/pairs/`, including gzipped orthogonal-variance tables), module definitions, annual-program source rows and frozen controls. Licensing and a scCody publication citation require author-provided metadata.

Full-gene input reconstruction is available without a private checkout:

```bash
python -m sccody_analyses.preprocessing.rebuild_fullgene_development_hvg_inputs --help
python -m sccody_analyses.preprocessing.rebuild_covid_development_hvg_inputs --help
```

AIDA reconstruction is the age-based `prepare_aida` workflow, and COVID
reconstruction uses `prepare_covid` directly on raw counts and cohort metadata.
Neither requires old prepared H5AD templates. The NPZ population-evaluation
inputs are generated separately by `evaluate_populations --phase prepare`.
Expression datasets are not bundled with the plotting results. For edgeR
analyses, install the R packages:

```r
install.packages(c("BiocManager", "statmod"))
BiocManager::install("edgeR")
```
