#!/usr/bin/env Rscript
# Redraw the current main-figure data panels using bundled tables only.

args <- commandArgs(trailingOnly = TRUE)
if ("--help" %in% args) {
  cat("Usage: Rscript manuscript/render_main_figures.R [--install-deps] [--panels-only]\n",
      "Redraws main-figure data panels from bundled results and configs.\n",
      "--install-deps installs missing CRAN plotting packages before rendering.\n",
      sep = "")
  quit(save = "no", status = 0)
}
if (any(!args %in% c("--install-deps", "--panels-only"))) stop("Unknown argument; use --help.")
script <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script) != 1L) stop("Run this file with Rscript.")
repo <- normalizePath(file.path(dirname(sub("^--file=", "", script)), ".."), mustWork = TRUE)
setwd(repo)

packages <- c("cowplot", "dplyr", "ggplot2", "ggrepel", "gridExtra", "jsonlite",
              "patchwork", "purrr", "ragg", "readr", "rsvg", "scales", "stringr", "svglite",
              "tibble", "tidyr")
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) && "--install-deps" %in% args) {
  install.packages(missing, repos = "https://cloud.r-project.org")
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
}
if (length(missing)) {
  stop("Missing R packages: ", paste(missing, collapse = ", "),
       ". Rerun with --install-deps to install them.")
}
if (!capabilities("cairo")) stop("This R installation needs Cairo graphics support for PDF export.")

# Figure 1c has three coverage views within one manuscript panel.
figure1 <- "manuscript/figure_1"
dir.create(file.path(figure1, "figure_files/panels"), recursive = TRUE, showWarnings = FALSE)
jobs <- list(
  c("Figure 1c AIDA", file.path(figure1, "scripts/Figure1C.age_coverage_heatmap.R"),
    file.path(figure1, "results/age_coverage_aida.csv"), "AIDA",
    file.path(figure1, "figure_files/panels/Figure1C.aida.png"), "10", "Annotation_Level1", "donors", "5"),
  c("Figure 1c COVID controls", file.path(figure1, "scripts/Figure1C.age_coverage_heatmap.R"),
    file.path(figure1, "results/age_coverage_covid_pooled_normal.csv"), "COVID controls",
    file.path(figure1, "figure_files/panels/Figure1C.covid_normal.png"), "10", "Annotation_Level1", "donors", "10", "8.4", "0,99"),
  c("Figure 1c COVID patients", file.path(figure1, "scripts/Figure1C.age_coverage_heatmap.R"),
    file.path(figure1, "results/age_coverage_covid_pooled_patient.csv"), "COVID patients",
    file.path(figure1, "figure_files/panels/Figure1C.covid_patient.png"), "10", "Annotation_Level1", "donors", "10", "8.4", "0,99"),
  c("Figure 2b", "manuscript/figure_2/scripts/Figure2B.R"),
  c("Figure 2d-h", "manuscript/figure_2/scripts/Figure2DEFGH.R"),
  c("Figure 3c", "manuscript/figure_3/scripts/Figure3C.R"),
  c("Figure 3d", "manuscript/figure_3/scripts/Figure3D.R"),
  c("Figure 3e", "manuscript/figure_3/scripts/Figure3E.R"),
  c("Figure 3f-h", "manuscript/figure_3/scripts/Figure3FGH.R"),
  c("Figure 4a", "manuscript/figure_4/scripts/Figure4A.R"),
  c("Figure 4c", "manuscript/figure_4/scripts/Figure4C.R"),
  c("Figure 4d", "manuscript/figure_4/scripts/Figure4D.R"),
  c("Figure 4f", "manuscript/figure_4/scripts/Figure4F.R"),
  c("Figure 5a-f", "manuscript/figure_5/scripts/plot_figure5.R")
)
rscript <- file.path(R.home("bin"), "Rscript")
for (job in jobs) {
  cat("[render] ", job[[1]], "\n", sep = "")
  status <- system2(rscript, args = vapply(job[-1], shQuote, character(1)))
  if (status != 0L) stop(job[[1]], " failed with exit status ", status)
}
if (!"--panels-only" %in% args) {
  python <- Sys.getenv("SCCODY_PYTHON", "python3")
  code <- system2(python, "manuscript/assemble_main_figures.py")
  if (code != 0L) stop("Main-figure assembly failed: ", code)
  source("manuscript/export_assembled_figures.R")
}
cat("[done] Main figures and panels are in manuscript/figure_*/figure_files/.\n")
