#!/usr/bin/env bash
# Figure 1c: donor coverage by lineage and age. Run from the repository root.
set -euo pipefail
F=manuscript/figure_1
S=$F/scripts/Figure1C.age_coverage_heatmap.R
O=$F/figure_files/panels
mkdir -p "$O"
Rscript $S $F/results/age_coverage_aida.csv "AIDA" $O/Figure1C.aida.png 10 Annotation_Level1 donors 5
Rscript $S $F/results/age_coverage_covid_pooled_normal.csv "COVID controls" $O/Figure1C.covid_normal.png 10 Annotation_Level1 donors 10 8.4 0,99
Rscript $S $F/results/age_coverage_covid_pooled_patient.csv "COVID patients" $O/Figure1C.covid_patient.png 10 Annotation_Level1 donors 10 8.4 0,99
