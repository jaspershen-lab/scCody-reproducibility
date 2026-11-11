#!/usr/bin/env Rscript

# Figure 3e: per-lineage localization (MAE, Pearson r) on the 71 external donors,
# lineage eligibility >= 5 cells, three-fit donor ensembles. Fig. 3e is the
# PCA-KNN version (DC dynamics cells use filter-first values).
# Stars: scCody-G/CT vs the best dynamics competitor per lineage and metric
# (MAE: paired t-test of absolute errors; r: Williams' test), unadjusted;
# stars only where scCody is better.
# Inputs (manuscript/figure_3/results/): aida_forest_cell_type_plot_<version>_mae_r.tsv
# and aida_cell_type_scCody_vs_best_dynamics_pvalues.tsv.
# Run from the repository root:  Rscript manuscript/figure_3/scripts/Figure3E.R

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(readr)
})

root <- normalizePath(".")
outdir <- file.path(root, "manuscript/figure_3/results")
panel_dir <- file.path(root, "manuscript/figure_3/figure_files/panels")
dir.create(panel_dir, recursive = TRUE, showWarnings = FALSE)
source(file.path(root, "manuscript/significance.R"))

cell_types <- c("T", "Myeloid", "DC", "B", "NK")
method_levels <- c(
  "scCody-G", "scCody-CT", "scNODE", "scDiffEq", "ARTEMIS",
  "Lasso", "ElasticNet", "tAge"
)

method_cols <- c(
  "scCody-G" = "#0072B2",
  "scCody-CT" = "#009E73",
  "scNODE" = "#CC79A7",
  "scDiffEq" = "#6A3D9A",
  "ARTEMIS" = "#D55E00",
  "Lasso" = "#E69F00",
  "ElasticNet" = "#56B4E9",
  "tAge" = "#F0E442"
)

read_plot_data <- function(version) {
  path <- file.path(
    outdir,
    paste0("aida_forest_cell_type_plot_", version, "_mae_r.tsv")
  )
  frame <- read_tsv(path, show_col_types = FALSE) %>%
    filter(
      cell_type %in% cell_types,
      model %in% method_levels,
      metric %in% c("MAE", "pearson_r")
    )

  if (version == "pca_knn") {
    filter_first_path <- file.path(
      outdir, "aida_forest_cell_type_plot_filter_first_mae_r.tsv"
    )
    filter_first_dc <- read_tsv(filter_first_path, show_col_types = FALSE) %>%
      filter(
        cell_type == "DC",
        model %in% c("scNODE", "scDiffEq", "ARTEMIS"),
        metric %in% c("MAE", "pearson_r")
      ) %>%
      mutate(source = paste0("filter_first fallback; ", source))
    frame <- bind_rows(
      filter(frame, !(cell_type == "DC" & model %in% c("scNODE", "scDiffEq", "ARTEMIS"))),
      filter_first_dc
    )
  }

  # Keep a tile for methods without DC evaluations and label those values NA.
  frame <- tidyr::expand_grid(
    model = method_levels,
    cell_type = cell_types,
    metric = c("MAE", "pearson_r")
  ) %>%
    left_join(frame, by = c("model", "cell_type", "metric"))

  if (nrow(frame) != length(cell_types) * length(method_levels) * 2) {
    stop("Unexpected number of rows in ", path, ": ", nrow(frame))
  }
  stars <- read_tsv(file.path(outdir, "aida_cell_type_scCody_vs_best_dynamics_pvalues.tsv"), show_col_types = FALSE) %>%
    filter(version == .env$version) %>%
    transmute(model = scody_model, cell_type,
              metric = recode(comparison, "Pearson r" = "pearson_r"),
              stars = direction_marks(p_value, if_else(metric == "MAE", difference_scody_minus_dynamic < 0,
                                                       difference_scody_minus_dynamic > 0)))
  frame %>%
    left_join(stars, by = c("model", "cell_type", "metric")) %>%
    mutate(stars = coalesce(stars, "")) %>%
    mutate(
      model = factor(model, levels = method_levels),
      cell_type = factor(cell_type, levels = cell_types),
      x = match(as.character(cell_type), cell_types),
      y = length(method_levels) - match(as.character(model), method_levels) + 1,
      annotation = case_when(
        metric == "MAE" & is.finite(value) ~ sprintf("%.1f", value),
        metric == "pearson_r" & is.finite(value) ~ sprintf("%.2f", value),
        TRUE ~ "NA"
      )
    )
}

heatmap_panel <- function(frame, metric_name) {
  panel <- frame %>% filter(metric == metric_name)
  method_rows <- tibble(
    model = factor(method_levels, levels = method_levels),
    y = rev(seq_along(method_levels))
  )

  if (metric_name == "MAE") {
    fill_scale <- scale_fill_gradient(
      low = "#2166AC", high = "#F7FBFF", na.value = "#E5E7EB",
      name = "MAE (years)"
    )
    title <- "MAE ↓"
  } else {
    fill_scale <- scale_fill_gradient2(
      low = "#B2182B", mid = "#F7F7F7", high = "#2166AC", midpoint = 0.4,
      na.value = "#E5E7EB", name = "Pearson r"
    )
    title <- "Pearson r ↑"
  }

  ggplot(panel, aes(x = x, y = y)) +
    geom_tile(aes(fill = value), width = 0.94, height = 0.88,
              colour = "white", linewidth = 0.8) +
    geom_text(aes(label = annotation), colour = "#111827", size = 3.3,
              fontface = "bold") +
    geom_text(aes(x = x + 0.43, y = y + 0.38, label = stars), colour = "#111827",
              size = 3.0, hjust = 1, vjust = 1) +
    geom_text(
      data = method_rows,
      aes(x = 0.40, y = y, label = model, colour = model),
      inherit.aes = FALSE,
      hjust = 1,
      size = 3.5,
      fontface = "bold"
    ) +
    annotate(
      "segment", x = 0.54, xend = length(cell_types) + 0.46, y = 3.5, yend = 3.5,
      colour = "#111111", linewidth = 0.9
    ) +
    fill_scale +
    scale_colour_manual(values = method_cols, guide = "none") +
    scale_x_continuous(
      breaks = seq_along(cell_types),
      labels = cell_types,
      limits = c(-1.15, length(cell_types) + 0.55),
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      breaks = seq_along(method_levels),
      labels = NULL,
      limits = c(0.5, length(method_levels) + 0.5),
      expand = c(0, 0)
    ) +
    labs(title = title, x = NULL, y = NULL) +
    coord_cartesian(clip = "off") +
    theme_minimal(base_size = 11.5) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.62, colour = "#111827"),
      axis.text.x = element_text(face = "bold", colour = "#111827"),
      axis.text.y = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank(),
      legend.position = "bottom",
      legend.title = element_text(face = "bold"),
      legend.key.height = grid::unit(0.42, "cm"),
      plot.margin = margin(8, 8, 8, 72)
    )
}

save_heatmap <- function(version) {
  data <- read_plot_data(version)
  plot <- heatmap_panel(data, "MAE") + heatmap_panel(data, "pearson_r") +
    plot_layout(ncol = 2, guides = "keep")
  if (version == "pca_knn") {
    plot <- plot + plot_annotation(
      caption = NULL
    )
  }
  stem <- file.path(panel_dir, "Figure3E.cell_type_heatmap")
  dir.create(dirname(stem), recursive = TRUE, showWarnings = FALSE)
  ggsave(paste0(stem, ".svg"), plot, width = 12.5, height = 7.2, units = "in", bg = "white")
  ggsave(paste0(stem, ".pdf"), plot, width = 12.5, height = 7.2, units = "in", device = cairo_pdf, bg = "white")
  ggsave(paste0(stem, ".png"), plot, width = 12.5, height = 7.2, units = "in", dpi = 360, bg = "white")
  cat("Saved", stem, "\n")
}

save_heatmap("pca_knn")
