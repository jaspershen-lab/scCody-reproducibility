#!/usr/bin/env Rscript
# Age coverage heatmap: coarse lineage (cell type) x age (optionally binned).
# metric=donors (default): fill = number of distinct donors clearing the
#   min-cell QC threshold, per (cell type, age bin). Donors are recomputed
#   from per-donor cell totals within each bin, not summed across per-year
#   donor counts, so a donor spanning two years within one bin is counted once.
# metric=cells: fill = total number of cells, per (cell type, age bin),
#   log-scaled color mapping (raw counts shown in the legend) since cell
#   counts span orders of magnitude across lineages.
# White = 0, dark blue = maximum, in both cases.
#
# Figure 1c: AIDA (5-year bins), pooled COVID normal and patient (10-year bins).
# Input: donor x age x Annotation_Level1 cell counts from Figure1C.export_age_coverage.py.
# Usage: Rscript Figure1C.age_coverage_heatmap.R <input_csv> <dataset_label> <output_png> [min_cells_per_donor_celltype] [source_field_name] [metric: donors|cells] [bin_years] [width_in] [age_min,age_max]

suppressPackageStartupMessages({
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
input_csv    <- args[1]
dataset_lbl  <- args[2]
output_png   <- args[3]
min_cells    <- if (length(args) >= 4) as.integer(args[4]) else 10L
source_field <- if (length(args) >= 5) args[5] else "Annotation_Level1"
metric       <- if (length(args) >= 6) args[6] else "donors"
bin_years    <- if (length(args) >= 7) as.integer(args[7]) else 1L
width_in     <- if (length(args) >= 8) as.numeric(args[8]) else 8.4
# Optional fixed axis, e.g. "0,99", so panels from different cohorts share bins.
age_range    <- if (length(args) >= 9) as.numeric(strsplit(args[9], ",")[[1]]) else NULL
stopifnot(metric %in% c("donors", "cells"))

obs <- read.csv(input_csv, stringsAsFactors = FALSE)
stopifnot(all(c("donor_id", "age", "Annotation_Level1", "n_cells") %in% colnames(obs)))

if (bin_years > 1) {
  bin_start <- (obs$age %/% bin_years) * bin_years
  obs$age_group <- paste0(bin_start, "-", bin_start + bin_years - 1)
  bin_order <- if (is.null(age_range)) sort(unique(bin_start)) else
    seq((age_range[1] %/% bin_years) * bin_years, (age_range[2] %/% bin_years) * bin_years, by = bin_years)
  bin_labels <- paste0(bin_order, "-", bin_order + bin_years - 1)
} else {
  obs$age_group <- as.character(obs$age)
  bin_labels <- as.character(sort(unique(obs$age)))
}

all_bins  <- bin_labels
observed_types <- sort(unique(obs$Annotation_Level1))
# If every observed type belongs to AIDA's canonical Annotation_Level1 scheme,
# show the full canonical set of rows (e.g. ILC at 0) even when a given
# dataset has no cells of that type, so AIDA-scheme plots are directly
# comparable row-for-row. Other schemes (e.g. the PsychAD brain "class" field) keep
# only the types actually observed.
AIDA_TYPE_ORDER <- c("Myeloid", "T", "NK", "B", "Platelet", "DC", "Plasma_Cell", "ILC", "CD34_HSPC")
all_types <- if (all(observed_types %in% AIDA_TYPE_ORDER)) AIDA_TYPE_ORDER else observed_types
grid <- expand.grid(cell_type = all_types, age_group = all_bins, stringsAsFactors = FALSE)

if (metric == "donors") {
  cell_counts <- aggregate(
    list(n_cells = obs$n_cells),
    by = list(donor_id = obs$donor_id, age_group = obs$age_group, cell_type = obs$Annotation_Level1),
    FUN = sum
  )
  cell_counts <- cell_counts[cell_counts$n_cells >= min_cells, ]
  coverage <- aggregate(
    list(value = cell_counts$donor_id),
    by = list(cell_type = cell_counts$cell_type, age_group = cell_counts$age_group),
    FUN = function(x) length(unique(x))
  )
  fill_name <- "Donors"
} else {
  coverage <- aggregate(
    list(value = obs$n_cells),
    by = list(cell_type = obs$Annotation_Level1, age_group = obs$age_group),
    FUN = sum
  )
  fill_name <- "Cells"
}

plot_df <- merge(grid, coverage, by = c("cell_type", "age_group"), all.x = TRUE)
plot_df$value[is.na(plot_df$value)] <- 0
plot_df$age_group <- factor(plot_df$age_group, levels = all_bins)

# Cell types are ordered to match AIDA's canonical Annotation_Level1 order
# whenever every observed type is part of that scheme, so all AIDA-scheme
# plots (AIDA, OneK1K, Ren, Stephenson, COMBAT, combined COVID) read the same
# way top-to-bottom. Other schemes (e.g. the PsychAD brain "class" field) fall back
# to ordering by total coverage (descending).
if (all(observed_types %in% AIDA_TYPE_ORDER)) {
  type_order <- AIDA_TYPE_ORDER
} else {
  type_totals <- aggregate(list(total = plot_df$value), by = list(cell_type = plot_df$cell_type), FUN = sum)
  type_order <- type_totals[order(-type_totals$total), "cell_type"]
}
plot_df$cell_type <- factor(plot_df$cell_type, levels = rev(type_order))

if (metric == "cells") {
  # Log-scale the color mapping (cell counts span orders of magnitude across
  # lineages) but label the legend with raw cell counts, not log values.
  plot_df$fill_val <- log1p(plot_df$value)
  raw_breaks <- c(0, 10, 100, 1000, 10000, 100000)
  raw_breaks <- raw_breaks[raw_breaks <= max(plot_df$value)]
  if (length(raw_breaks) < 2) raw_breaks <- c(0, max(plot_df$value))
  fill_breaks <- log1p(raw_breaks)
  fill_labels <- scales::comma(raw_breaks)
} else {
  plot_df$fill_val <- plot_df$value
  fill_breaks <- scales::pretty_breaks()(c(0, max(plot_df$value)))
  fill_labels <- fill_breaks
}

title_full <- dataset_lbl
title_wrapped <- paste(strwrap(title_full, width = max(14, round(width_in * 6))), collapse = "\n")

p <- ggplot(plot_df, aes(x = age_group, y = cell_type, fill = fill_val)) +
  geom_tile(color = "white", linewidth = 0.6) +
  scale_fill_gradient(
    name = fill_name, low = "#EFF1F8", high = "#3D60AA",
    breaks = fill_breaks, labels = fill_labels,
    limits = c(0, max(plot_df$fill_val))
  ) +
  labs(
    title = title_wrapped,
    x = if (bin_years > 1) "Age bin (years)" else "Age (years)",
    y = NULL
  ) +
  theme_minimal(base_size = 12, base_family = "Arial") +
  theme(
    panel.grid = element_blank(),
    plot.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 8)
  )

ggsave(output_png, plot = p, width = width_in, height = 5, dpi = 150)
output_svg <- sub("\\.png$", ".svg", output_png)
ggsave(output_svg, plot = p, width = width_in, height = 5)
ggsave(sub("\\.png$", ".pdf", output_png), plot = p, width = width_in, height = 5, device = cairo_pdf)
cat("Wrote", output_png, "and", output_svg, "\n")
