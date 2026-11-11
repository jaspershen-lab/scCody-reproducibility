# Figure 5

All six panels use the bundled `results/` tables: current corrected COVID results and unchanged PsychAD results. Program indices match the corrected ordered gene panel. Original program-change, dispersion, onset and subtype-contrast statistical definitions are retained. Individual vector panels and legends remain available for editing.

Run the commands below from the repository root.

## Redraw the panels

With the R plotting dependencies described in the [root README](../../README.md) installed:

```bash
Rscript manuscript/figure_5/scripts/plot_figure5.R
```

This reads only the bundled tables and writes SVG, PDF and PNG panels and separate legends to `manuscript/figure_5/figure_files/panels/`. To redraw and assemble all five main figures, use `Rscript manuscript/render_main_figures.R`.

| Panel | Analysis | Main input tables in `results/` |
|---|---|---|
| a | AD excitatory-neuron program enrichment | `ad_en_program_heatmap.csv` |
| b | AD excitatory-neuron subclass shared contribution | `ad_en_subclass_shared_contribution.csv`, `ad_en_subclass_latent_readout.csv` |
| c | COVID myeloid program enrichment | `covid_program_heatmap.csv` |
| d | Between- and within-patient program dispersion | `covid_component_dispersion.csv` |
| e | Interferon programs and symptom onset | `covid_ifn_symptom_onset.csv`, `covid_ifn_symptom_onset_donors.csv` |
| f | CD14/CD16 monocyte abundance, residual energy and programs | `covid_subtype_cell_share_change.csv`, `covid_subtype_excess_same_subtype_stats.csv`, `program_change.csv`, `program_dispersion.csv`, `covid_subtype_contrasts.csv` |

## Rebuild the tables from saved analysis outputs

The [root reproduction workflow](../../README.md#6-disease-displacement-and-figure-5) describes field extraction and decoding with the existing checkpoints. The table builders below use those completed outputs and prepared expression metadata. They do not fit models or select checkpoints, and they overwrite the corresponding tables in `manuscript/figure_5/results/`.

Required inputs are:

- Prepared COVID H5ADs under `data/covid/` and PsychAD `data/psychad/ad_case_control.h5ad`, retaining the original subtype, subclass and clinical annotations.
- Curated modules and ordered gene panels under `configs/programs/`.
- Per-cell decoded residuals at `artifacts/residual_programs/decoded/<cohort>/fold<k>/Myeloid[_healthy]/percell_components.tsv.gz` for Ren, COMBAT and Stephenson, fits 0–2.
- Program-score outputs `component_program_scores.tsv.gz` and `group_gene_effects.npz` at `artifacts/residual_programs/decoded_programs/<cohort>/fold<k>/Myeloid[_healthy]/`; the corresponding AD outputs use `decoded_programs/AD/EN[_healthy]/` without a fold directory.
- AD subclass outputs `subclass_latent_readout.tsv` and `subclass_shared_contribution.tsv` under `artifacts/residual_programs/analysis/AD_EN_subclass/`.

If group labels or program-score outputs have not been generated, create the labels before running `decode_component_program_scores --labels`:

```bash
python manuscript/figure_5/scripts/make_program_group_labels.py
python -m sccody_analyses.decode_component_program_scores \
  --model checkpoints/AIDA_scCody-CT_fit0.pt \
  --fields-dir artifacts/residual_programs/fields/Ren/fold0 \
  --group Myeloid --modules configs/programs/curated/COVID_modules.json \
  --labels artifacts/residual_programs/group_labels/Ren.tsv \
  --outdir artifacts/residual_programs/decoded_programs/Ren/fold0/Myeloid
```

Repeat the scoring command for all three COVID cohorts and fits, and for both `Myeloid` and `Myeloid_healthy`. For AD, use `checkpoints/PsychAD_scCody-CT_fit0.pt`, modules `configs/programs/curated/AD_modules.json`, fields under `fields/AD`, labels `group_labels/AD.tsv`, and groups `EN` and `EN_healthy`, writing to `decoded_programs/AD/<group>/`. The fields, labels and output directories in this paragraph are relative to `artifacts/residual_programs/`.

To generate the AD subclass inputs from saved fields:

```bash
python -m sccody_analyses.analyze_ad_subclass_residual_readout \
  --model checkpoints/PsychAD_scCody-CT_fit0.pt \
  --fields-dir artifacts/residual_programs/fields/AD --group EN \
  --panel configs/programs/AD_panel.tsv \
  --outdir artifacts/residual_programs/analysis/AD_EN_subclass
```

Once these inputs exist, run the table builders in this order:

```bash
python manuscript/figure_5/scripts/prepare_figure5_data.py
python manuscript/figure_5/scripts/prepare_covid_subtype_donor_table.py
python manuscript/figure_5/scripts/prepare_program_tables.py
python manuscript/figure_5/scripts/analyze_ifn_symptom_onset.py
python manuscript/figure_5/scripts/analyze_covid_subtype_contrasts.py
```

The first script copies the AD subclass summaries. The second builds monocyte abundance and residual-energy estimates, including the same-subtype donor table needed by the final contrast analysis. `prepare_program_tables.py` builds program changes, dispersion and pooled heatmaps for AD and COVID. The onset analysis uses the saved per-cell program scores and clinical metadata; Stephenson contributes severity analyses but has no symptom-onset dates. The final script computes direct CD16-minus-CD14 contrasts. Run the plotting command above after rebuilding the tables.
