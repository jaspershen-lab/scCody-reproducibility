#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(jsonlite)
  library(readr)
  library(tidyr)
  library(gridExtra)
})

# Figure 2d-h and the whole-reference DE panels they are chosen from:
#   2d global_predicted_vs_real_de_shared_reference_pearson_r
#   2e global_predicted_vs_real_de_shared_reference_top_100_200_overlap (top 5%/10% of 2,000 genes)
#   2f global_sign_agreement_series
#   2g celltype_breakdown_predicted_vs_real_de_shared_reference_pearson_r_young_old_average
#   2h aida_model_category_rankings
# Inputs (manuscript/figure_2/results/): whole-reference DE metrics for the five
# models and the real-data half-split control, the 500 donor-disjoint split
# replicates, and mean_expression_correlation_global.tsv (Figure2H.population_rank.py).
# Run from the repository root:  Rscript manuscript/figure_2/scripts/Figure2DEFGH.R

root <- normalizePath(".", mustWork = TRUE)
results_dir <- file.path(root, "manuscript/figure_2/results")
table_path <- file.path(results_dir, "whole_ref_model_and_half_split_metrics.tsv.gz")
outdir <- file.path(root, "manuscript/figure_2/figure_files/panels")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
# Render only the panels selected for the main figure.
main_panels <- c(
  global_predicted_vs_real_de_shared_reference_pearson_r = "Figure2D.global_de_pearson_r",
  global_predicted_vs_real_de_shared_reference_top_100_200_overlap = "Figure2E.global_top_de_overlap",
  global_sign_agreement_series = "Figure2F.global_sign_agreement",
  celltype_breakdown_predicted_vs_real_de_shared_reference_pearson_r_young_old_average = "Figure2G.cell_type_de_pearson_r",
  aida_model_category_rankings = "Figure2H.model_category_rankings"
)
panel_path <- function(stem) {
  if (!stem %in% names(main_panels)) stop("Unselected panel: ", stem)
  file.path(outdir, main_panels[[stem]])
}
ranking_only <- identical(Sys.getenv("SCCODY_RANKING_ONLY"), "1")

model_keys <- c("FlowMap-global", "FlowMap-cell-type", "scNODE", "scDiffEq", "ARTEMIS")
model_labels <- c("FlowMap-global" = "scCody-G", "FlowMap-cell-type" = "scCody-CT", "scNODE" = "scNODE", "scDiffEq" = "scDiffEq", "ARTEMIS" = "ARTEMIS")
model_order <- unname(model_labels[model_keys])
model_cols <- c("scCody-G" = "#0072B2", "scCody-CT" = "#009E73", "scNODE" = "#CC79A7", "scDiffEq" = "#6A3D9A", "ARTEMIS" = "#D55E00")
cell_types <- c("B", "Myeloid", "NK", "T")
age_groups <- c("young", "old")

metric_fields <- c(
  "de_vs_reference_logfc_pearson_all", "de_vs_reference_logfc_spearman_all",
  "de_vs_reference_top100_overlap_frac",
  "de_vs_reference_top200_overlap_frac",
  "de_direction_agreement_all", "de_direction_agreement_nonzero",
  "sign_agreement_true_abs_log2fc_gt_0p25", "sign_agreement_true_abs_log2fc_gt_0p5"
)

pretty_metric <- function(field) {
  out <- field
  out <- sub("^de_vs_reference_logfc_", "Predicted vs real DE (shared reference) ", out)
  out <- sub("^de_vs_reference_", "Predicted vs real DE (shared reference) ", out)
  out <- sub("^de_direction_agreement_", "DE direction agreement ", out)
  out <- sub("^sign_agreement_true_", "Sign agreement ", out)
  out <- sub("^normalized_pred_vs_edgeR_real_", "Predicted vs real DE (edgeR) ", out)
  out <- sub("^normalized_real_vs_edgeR_real_", "Real vs real DE (edgeR) ", out)
  out <- gsub("_", " ", out)
  out <- gsub("pearson all", "Pearson r", out, fixed = TRUE)
  out <- gsub("spearman all", "Spearman ρ", out, fixed = TRUE)
  out <- gsub("ols slope", "OLS slope", out, fixed = TRUE)
  out <- gsub("ols intercept", "OLS intercept", out, fixed = TRUE)
  out <- gsub("ols r2", "OLS R²", out, fixed = TRUE)
  out <- gsub("pearson top100", "top-100 Pearson r", out, fixed = TRUE)
  out <- gsub("spearman top100", "top-100 Spearman ρ", out, fixed = TRUE)
  out <- gsub("top100 overlap frac", "top-100 overlap", out, fixed = TRUE)
  out <- gsub("top200 overlap frac", "top-200 overlap", out, fixed = TRUE)
  out <- gsub("pearson top500", "top-500 Pearson r", out, fixed = TRUE)
  out <- gsub("spearman top500", "top-500 Spearman ρ", out, fixed = TRUE)
  out <- gsub("top500 overlap frac", "top-500 overlap", out, fixed = TRUE)
  out <- gsub("abs log2fc gt 0p25", "|log2FC| > 0.25", out, fixed = TRUE)
  out <- gsub("abs log2fc gt 0p5", "|log2FC| > 0.5", out, fixed = TRUE)
  out <- gsub("fdr lt 0p05", "FDR < 0.05", out, fixed = TRUE)
  out <- gsub("nonzero", "nonzero", out, fixed = TRUE)
  out
}

metric_specs <- tibble(
  field = metric_fields,
  title = vapply(metric_fields, pretty_metric, character(1)),
  target = ifelse(grepl("ols_slope", metric_fields), "target 1", ifelse(grepl("ols_intercept", metric_fields), "target 0", "higher"))
)

de_half_map <- c(
  "de_vs_reference_logfc_pearson_all" = "pearson_r", "de_vs_reference_logfc_spearman_all" = "spearman_r",
  "de_vs_reference_logfc_pearson_top100" = "pearson_r_top100", "de_vs_reference_logfc_spearman_top100" = "spearman_r_top100", "de_vs_reference_top100_overlap_frac" = "top100_overlap_frac",
  "de_vs_reference_top200_overlap_frac" = "top200_overlap_frac",
  "de_vs_reference_logfc_pearson_top500" = "pearson_r_top500", "de_vs_reference_logfc_spearman_top500" = "spearman_r_top500", "de_vs_reference_top500_overlap_frac" = "top500_overlap_frac",
  "de_direction_agreement_all" = "direction_agreement_all", "de_direction_agreement_nonzero" = "direction_agreement_nonzero", "sign_agreement_true_fdr_lt_0p05" = "sign_agreement_true_fdr_lt_0p05", "sign_agreement_true_abs_log2fc_gt_0p25" = "sign_agreement_true_abs_log2fc_gt_0p25", "sign_agreement_true_abs_log2fc_gt_0p5" = "sign_agreement_true_abs_log2fc_gt_0p5"
)

raw <- read_tsv(table_path, show_col_types = FALSE, progress = FALSE)
required <- c("analysis", "method", "reference", "scope", "cell_type", "target_age", "side", "metric", "value", "statistic")
missing <- setdiff(required, names(raw))
if (length(missing)) stop("Missing columns: ", paste(missing, collapse = ", "))

model_df <- raw %>%
  filter(analysis == "model_performance", method %in% model_keys, metric %in% metric_fields) %>%
  mutate(model_key = method, model = unname(model_labels[method]), side = tolower(as.character(side)), value = suppressWarnings(as.numeric(value))) %>%
  filter(side %in% age_groups)

half_df <- raw %>%
  filter(analysis == "real_half_split_stability", method == "real_half_split", reference == "sc_ref_whole_dataset", statistic == "mean") %>%
  mutate(side = tolower(as.character(side)), value = suppressWarnings(as.numeric(value)), metric_plot = names(de_half_map)[match(as.character(metric), unname(de_half_map))]) %>%
  filter(side %in% age_groups, !is.na(metric_plot))

# Stars: scCody-G/CT vs the best competitor, paired over fit x external age (unadjusted).
source(file.path(root, "manuscript/significance.R"))
test_records <- list()
record_tests <- function(tests, panel, field) {
  test_records[[length(test_records) + 1L]] <<- tests %>% mutate(panel = panel, metric = field, .before = 1)
  tests
}

required_celltype <- expand_grid(model_key = model_keys, scope = cell_types, side = age_groups)
has_complete_celltype <- function(field) {
  observed <- model_df %>% filter(metric == field, scope %in% cell_types, is.finite(value)) %>% distinct(model_key, scope, side)
  nrow(anti_join(required_celltype, observed, by = c("model_key", "scope", "side"))) == 0L
}
celltype_fields <- metric_specs$field[vapply(metric_specs$field, has_complete_celltype, logical(1))]

model_summary <- function(field, scope_kind) {
  subset <- model_df %>% filter(metric == field, is.finite(value))
  lookup <- tibble(model_key = model_keys, model = unname(model_labels[model_keys]))
  if (scope_kind == "global") {
    values <- subset %>% filter(scope == "all") %>% group_by(model_key, side) %>%
      summarise(mean = mean(value), sd = if (n() > 1L) sd(value) else 0, n = n(), .groups = "drop")
    return(expand_grid(model_key = model_keys, side = age_groups) %>% left_join(lookup, by = "model_key") %>% left_join(values, by = c("model_key", "side")))
  }
  values <- subset %>% filter(scope %in% cell_types) %>% group_by(model_key, scope, side) %>%
    summarise(cell_mean = mean(value), cell_n = n(), .groups = "drop")
  if (scope_kind == "celltype_macro") {
    macro <- values %>% group_by(model_key, side) %>%
      summarise(mean = mean(cell_mean), sd = if (n() > 1L) sd(cell_mean) else 0, n = n(), .groups = "drop")
    return(expand_grid(model_key = model_keys, side = age_groups) %>% left_join(lookup, by = "model_key") %>% left_join(macro, by = c("model_key", "side")) %>% mutate(cell_type = "macro"))
  }
  expand_grid(model_key = model_keys, scope = cell_types, side = age_groups) %>%
    left_join(lookup, by = "model_key") %>% left_join(values, by = c("model_key", "scope", "side")) %>%
    transmute(model_key, model, side, cell_type = scope, mean = cell_mean, sd = 0, n = cell_n)
}

half_summary <- function(field, scope_kind) {
  subset <- half_df %>% filter(metric_plot == field, is.finite(value))
  if (scope_kind == "global") return(subset %>% filter(scope == "all") %>% group_by(side) %>% summarise(half_mean = mean(value), half_n = n(), .groups = "drop"))
  by_cell <- subset %>% filter(scope %in% cell_types) %>% group_by(side, scope) %>% summarise(cell_mean = mean(value), .groups = "drop")
  if (scope_kind == "celltype_macro") return(by_cell %>% group_by(side) %>% summarise(half_mean = mean(cell_mean), half_n = n(), .groups = "drop"))
  by_cell %>% transmute(side, cell_type = scope, half_mean = cell_mean, half_n = 1L)
}

limits_for <- function(values, target) {
  values <- values[is.finite(values)]
  if (!length(values)) return(c(0, 1))
  lo <- min(values); hi <- max(values); span <- max(hi - lo, 0.15)
  if (target == "higher" && lo >= -0.02) {
    lo <- min(0, lo - 0.06 * span); hi <- hi + 0.16 * span
  } else if (target %in% c("target 0", "target 1")) {
    lo <- min(0, lo - 0.12 * span); hi <- hi + 0.16 * span
  } else {
    lo <- lo - 0.12 * span; hi <- hi + 0.16 * span
  }
  c(lo, hi)
}

clean_name <- function(value) gsub("[^a-z0-9]+", "_", tolower(value)) |> sub("^_+|_+$", "", x = _)

base_theme <- theme_classic(base_size = 12, base_family = "Arial") + theme(
  plot.title = element_text(face = "bold", size = 16, hjust = 0.5, margin = margin(b = 5)),
  strip.text = element_text(face = "bold", size = 13),
  axis.text.x = element_text(size = 10.5, color = "#111827"), axis.text.y = element_text(size = 10.5, color = "#111827"),
  axis.title.y = element_text(size = 12.5, margin = margin(r = 8)), legend.position = "top", legend.justification = "center",
  legend.text = element_text(size = 10.5), strip.background = element_blank(), panel.grid.major.y = element_line(color = "#e5e7eb", linewidth = 0.35),
  panel.grid.minor = element_blank(), axis.line = element_line(color = "#111827", linewidth = 0.35),
  axis.ticks = element_line(color = "#111827", linewidth = 0.35), panel.spacing = unit(1.2, "lines"), plot.margin = margin(12, 18, 12, 14)
)

add_half_layer <- function(plot, half_data) {
  if (!nrow(half_data)) return(plot)
  plot +
    geom_hline(data = half_data, aes(yintercept = half_mean, linetype = "Split-half"), inherit.aes = FALSE, color = "#000000", linewidth = 1.05) +
    geom_point(data = half_data, aes(x = model_order[length(model_order)], y = half_mean), inherit.aes = FALSE, shape = 18, size = 3.4, color = "#000000", show.legend = FALSE) +
    scale_linetype_manual(values = c("Split-half" = "dashed"), name = NULL)
}

save_plot <- function(plot, stem, width, height) {
  ggsave(paste0(panel_path(stem), ".png"), plot, width = width, height = height, units = "in", dpi = 360, device = ragg::agg_png, bg = "white")
  ggsave(paste0(panel_path(stem), ".pdf"), plot, width = width, height = height, units = "in", device = grDevices::cairo_pdf, bg = "white")
  ggsave(paste0(panel_path(stem), ".svg"), plot, width = width, height = height, units = "in", device = svglite::svglite, system_fonts = list(sans = "Arial"), bg = "white")
}

plot_global <- function(spec) {
  field <- spec$field; title <- spec$title; target <- spec$target
  values <- model_summary(field, "global"); plot_values <- values %>% mutate(model = factor(model, levels = model_order), side = factor(side, levels = age_groups))
  hs <- half_summary(field, "global") %>% mutate(side = factor(side, levels = age_groups))
  y_limits <- limits_for(c(values$mean, values$mean - values$sd, values$mean + values$sd, hs$half_mean), target)
  labels <- values %>% mutate(model = factor(model, levels = model_order), side = factor(side, levels = age_groups), label = sprintf("%.2f", mean), label_y = if_else(mean >= 0, mean + 0.045, mean - 0.045))
  stars <- record_tests(best_competitor_tests(model_df %>% filter(metric == field, scope == "all"), "side"), "global", field) %>%
    inner_join(labels %>% transmute(model = as.character(model), side = as.character(side), label_y, mean), by = c("model", "side")) %>%
    mutate(model = factor(model, levels = model_order), side = factor(side, levels = age_groups),
           star_y = label_y + if_else(mean >= 0, 1, -1) * 0.06 * diff(y_limits))
  p <- ggplot(plot_values, aes(model, mean, fill = model)) +
    geom_col(position = position_dodge(width = 0.78), width = 0.64, color = "#1f2937", linewidth = 0.25, alpha = 0.88) +
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), position = position_dodge(width = 0.78), width = 0.18, linewidth = 0.55, color = "#111827") +
    geom_text(data = labels, aes(y = label_y, label = label), position = position_dodge(width = 0.78), size = 3.0, color = "#111827", show.legend = FALSE) +
    geom_text(data = stars, aes(x = model, y = star_y, label = stars), inherit.aes = FALSE, size = 4.2, color = "#111827") +
    scale_fill_manual(values = model_cols, drop = FALSE) + facet_wrap(~side, nrow = 1, labeller = as_labeller(c(young = "Young", old = "Old"))) +
    coord_cartesian(ylim = y_limits) + scale_y_continuous(breaks = scales::pretty_breaks(n = 5), expand = expansion(mult = c(0, 0.02))) +
    labs(title = "DE correlation", x = NULL, y = "Pearson r", fill = NULL) + base_theme + guides(fill = guide_legend(nrow = 2, byrow = TRUE))
  p <- add_half_layer(p, hs)
  save_plot(p, paste0("global_", clean_name(title)), 11.5, 5.8)
  values %>% mutate(plot_scope = "global", cell_type = "all", metric = field) %>% left_join(hs %>% select(side, half_mean, half_n), by = "side") %>% select(plot_scope, cell_type, metric, side, model, mean, sd, n, half_mean, half_n)
}

plot_celltype_pearson_heatmap <- function(values, hs, age_average = FALSE, output_stem = "celltype_breakdown_predicted_vs_real_de_shared_reference_pearson_r", plot_title = "AIDA cell-type DE — predicted vs real DE (shared reference)", plot_subtitle = "Pearson r; values are averaged across available target ages", width = 15.0) {
  row_order <- c("Split-half", model_order)
  cell_order <- c("B", "Myeloid", "NK", "T", "Avg")
  side_levels <- if (age_average) "average" else age_groups
  if (age_average) {
    values <- values %>%
      group_by(model_key, model, cell_type) %>%
      summarise(mean = mean(mean), .groups = "drop") %>%
      mutate(side = "average")
    hs <- hs %>%
      group_by(cell_type) %>%
      summarise(half_mean = mean(half_mean), .groups = "drop") %>%
      mutate(side = "average")
  }
  x_positions <- if (age_average) {
    setNames(seq_along(cell_order), paste("average", cell_order))
  } else {
    c(
      setNames(seq_along(cell_order), paste("young", cell_order)),
      setNames(6 + seq_along(cell_order), paste("old", cell_order))
    )
  }

  model_cells <- values %>%
    transmute(
      row = as.character(model), side = tolower(as.character(side)),
      cell_type = as.character(cell_type), value = mean
    )
  model_avg <- model_cells %>%
    group_by(row, side) %>%
    summarise(cell_type = "Avg", value = mean(value), .groups = "drop")
  split_cells <- hs %>%
    transmute(
      row = "Split-half", side = tolower(as.character(side)),
      cell_type = as.character(cell_type), value = half_mean
    )
  split_avg <- split_cells %>%
    group_by(row, side) %>%
    summarise(cell_type = "Avg", value = mean(value), .groups = "drop")

  heat <- bind_rows(model_cells, model_avg, split_cells, split_avg) %>%
    mutate(
      row = factor(row, levels = row_order),
      side = factor(side, levels = side_levels),
      cell_type = factor(cell_type, levels = cell_order),
      x = unname(x_positions[paste(as.character(side), as.character(cell_type))]),
      y = length(row_order) - as.integer(row) + 1L,
      label = sprintf("%.3f", value)
    )

  row_labels <- tibble(
    x = -0.30,
    y = length(row_order) - seq_along(row_order) + 1L,
    label = row_order,
    label_color = c("#111827", unname(model_cols[model_order]))
  )
  reference_label <- tibble(x = -0.30, y = 6.30, label = "Reference")
  column_labels <- if (age_average) {
    tibble(x = seq_len(5), y = 6.62, label = cell_order)
  } else {
    tibble(x = c(seq_len(5), 6 + seq_len(5)), y = 6.62, label = rep(cell_order, 2))
  }
  group_labels <- if (age_average) {
    tibble(x = 3, y = 7.25, label = "YOUNG + OLD\nAVERAGE")
  } else {
    tibble(x = c(3, 9), y = 7.25, label = c("YOUNG", "OLD"))
  }

  # Stars next to the scCody Avg cells (age-averaged version only): cell-type and
  # Young/Old average per fit x external age, vs the best competitor.
  avg_stars <- if (age_average) {
    record_tests(best_competitor_tests(
      model_df %>% filter(metric == "de_vs_reference_logfc_pearson_all", scope %in% cell_types) %>%
        group_by(model, fold, target_age) %>% summarise(value = mean(value), .groups = "drop") %>%
        mutate(column = "Avg"), "column"), "celltype_average", "de_vs_reference_logfc_pearson_all") %>%
      mutate(x = 5.56, y = length(row_order) - match(model, row_order) + 1L)
  } else tibble(x = numeric(), y = numeric(), stars = character())

  p <- ggplot(heat) +
    geom_tile(
      aes(x = x, y = y, fill = value),
      width = 0.96, height = 0.88, color = "#ffffff", linewidth = 0.8
    ) +
    geom_text(
      data = filter(heat, cell_type != "Avg"),
      aes(x = x, y = y, label = label),
      family = "Arial", size = 3.6, color = "#111827"
    ) +
    geom_text(
      data = filter(heat, cell_type == "Avg"),
      aes(x = x, y = y, label = label),
      family = "Arial", fontface = "bold", size = 3.7, color = "#111827"
    ) +
    geom_text(
      data = avg_stars, aes(x = x, y = y, label = stars),
      inherit.aes = FALSE, hjust = 0, family = "Arial", size = 4.2, color = "#111827"
    ) +
    geom_text(
      data = row_labels,
      aes(x = x, y = y, label = label, color = label_color),
      inherit.aes = FALSE, hjust = 1, family = "Arial", fontface = "bold", size = 4.0
    ) +
    geom_text(
      data = reference_label,
      aes(x = x, y = y, label = label),
      inherit.aes = FALSE, hjust = 1, family = "Arial", fontface = "italic", size = 3.0, color = "#6b7280"
    ) +
    geom_text(
      data = column_labels,
      aes(x = x, y = y, label = label),
      inherit.aes = FALSE, family = "Arial", fontface = "bold", size = 3.8, color = "#111827"
    ) +
    geom_text(
      data = group_labels,
      aes(x = x, y = y, label = label),
      inherit.aes = FALSE, family = "Arial", fontface = "bold", size = 4.4, color = "#111827"
    ) +
    annotate("segment", x = 4.5, xend = 4.5, y = 0.55, yend = 6.42, color = "#4b5563", linewidth = 1.15) +
    annotate("segment", x = 0.5, xend = if (age_average) 5.5 else 11.5, y = 5.50, yend = 5.50, color = "#374151", linewidth = 1.45) +
    annotate("segment", x = 0.5, xend = if (age_average) 5.5 else 11.5, y = 6.48, yend = 6.48, color = "#6b7280", linewidth = 0.7) +
    scale_fill_gradient2(
      low = "#D6A0A0", mid = "#FAF9F6", high = "#8FA8BC", midpoint = 0,
      limits = c(-0.20, 0.55), oob = scales::squish,
      breaks = c(-0.20, 0, 0.20, 0.40), labels = c("−0.20", "0", "0.20", "0.40")
    ) +
    scale_color_identity() +
    coord_cartesian(xlim = if (age_average) c(-2.75, 5.70) else c(-2.75, 12.15), ylim = c(0.35, 7.65), clip = "off") +
    labs(
      title = plot_title,
      subtitle = plot_subtitle,
      x = NULL, y = NULL, fill = "Pearson r"
    ) +
    base_theme +
    theme(
      plot.title = element_text(size = 16, hjust = 0.5, margin = margin(b = 3)),
      plot.subtitle = element_text(size = 11, hjust = 0.5, margin = margin(b = 12)),
      axis.text = element_blank(), axis.text.x = element_blank(), axis.text.y = element_blank(),
      axis.title = element_blank(), axis.title.x = element_blank(), axis.title.y = element_blank(),
      axis.ticks = element_blank(), axis.ticks.x = element_blank(), axis.ticks.y = element_blank(),
      axis.line = element_blank(), panel.grid = element_blank(),
      panel.grid.major.x = element_blank(), panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
      legend.position = "right", legend.title = element_text(size = 11.5, face = "bold"),
      legend.text = element_text(size = 10.5), legend.key.height = unit(1.2, "lines"),
      plot.margin = margin(14, 20, 14, 20)
    )

  if (!age_average) {
    p <- p +
      annotate("segment", x = 5.5, xend = 5.5, y = 0.55, yend = 6.42, color = "#6b7280", linewidth = 0.7) +
      annotate("segment", x = 10.5, xend = 10.5, y = 0.55, yend = 6.42, color = "#4b5563", linewidth = 1.15)
  }
  # Rebuild the plot after adding age-specific separators so the age-averaged
  # version has only its single Avg separator. (The separator annotations are
  # otherwise identical and are intentionally kept out of the data tables.)
  save_plot(p, output_stem, width, 7.2)
}

plot_breakdown <- function(spec) {
  field <- spec$field
  stopifnot(field == "de_vs_reference_logfc_pearson_all")
  values <- model_summary(field, "celltype_breakdown") %>% mutate(model = factor(model, levels = model_order), side = factor(side, levels = age_groups), cell_type = factor(cell_type, levels = cell_types))
  hs <- half_summary(field, "celltype_breakdown") %>% mutate(side = factor(side, levels = age_groups), cell_type = factor(cell_type, levels = cell_types))
  plot_celltype_pearson_heatmap(
    values, hs, age_average = TRUE,
    output_stem = "celltype_breakdown_predicted_vs_real_de_shared_reference_pearson_r_young_old_average",
    plot_title = "Cell-type DE",
    plot_subtitle = NULL,
    width = 11.5
  )
}

make_hatch_segments <- function(xmin, xmax, ymin, ymax, spacing = 0.035) {
  if (!all(is.finite(c(xmin, xmax, ymin, ymax))) || ymax <= ymin) return(tibble())
  # Diagonal strokes are explicitly clipped to each extension rectangle.
  intercepts <- seq(ymin - xmax, ymax - xmin, by = spacing)
  segments <- lapply(intercepts, function(intercept) {
    x_start <- max(xmin, ymin - intercept)
    x_end <- min(xmax, ymax - intercept)
    if (x_end <= x_start) return(NULL)
    tibble(x = x_start, xend = x_end, y = x_start + intercept, yend = x_end + intercept)
  })
  bind_rows(segments)
}

plot_global_top100_top200_composite <- function() {
  top100 <- model_summary("de_vs_reference_top100_overlap_frac", "global") %>%
    transmute(model_key, model, side, top100 = mean, top100_sd = sd)
  top200 <- model_summary("de_vs_reference_top200_overlap_frac", "global") %>%
    transmute(model_key, side, top200 = mean, top200_sd = sd)
  bars <- top100 %>%
    left_join(top200, by = c("model_key", "side")) %>%
    mutate(
      model = factor(model, levels = model_order),
      side = factor(side, levels = age_groups),
      x = match(as.character(model), model_order),
      xmin = x - 0.36,
      xmax = x + 0.36
    )
  if (any(!is.finite(bars$top100)) || any(!is.finite(bars$top200))) stop("Composite top-100/top-200 plot has missing global values.")
  if (any(bars$top200 < bars$top100)) stop("Composite top-200 values must be at least the top-100 values.")

  # Stars above each bar test the top-200 (top 10%) overlap; top-100 tests are recorded too.
  overlap_tests <- function(field) record_tests(best_competitor_tests(model_df %>% filter(metric == field, scope == "all"), "side"), "global", field)
  overlap_tests("de_vs_reference_top100_overlap_frac")
  stars <- overlap_tests("de_vs_reference_top200_overlap_frac") %>%
    inner_join(bars %>% transmute(model = as.character(model), side = as.character(side), x, top200), by = c("model", "side")) %>%
    mutate(side = factor(side, levels = age_groups))

  hatch_segments <- bind_rows(lapply(seq_len(nrow(bars)), function(i) {
    make_hatch_segments(bars$xmin[i], bars$xmax[i], bars$top100[i], bars$top200[i]) %>%
      mutate(model = as.character(bars$model[i]), side = bars$side[i])
  }))

  controls <- bind_rows(
    half_summary("de_vs_reference_top100_overlap_frac", "global") %>% mutate(control = "Split-half (top 5%)"),
    half_summary("de_vs_reference_top200_overlap_frac", "global") %>% mutate(control = "Split-half (top 10%)")
  ) %>% mutate(side = factor(side, levels = age_groups), control = factor(control, levels = c("Split-half (top 5%)", "Split-half (top 10%)")))

  p <- ggplot(bars) +
    geom_rect(
      aes(xmin = xmin, xmax = xmax, ymin = 0, ymax = top100, fill = model),
      color = "#1f2937", linewidth = 0.25, alpha = 0.88
    ) +
    geom_rect(
      aes(xmin = xmin, xmax = xmax, ymin = top100, ymax = top200, fill = model),
      color = "#1f2937", linewidth = 0.25, alpha = 0.25
    )

  for (model_name in model_order) {
    model_hatch <- hatch_segments %>% filter(model == model_name)
    if (nrow(model_hatch)) {
      p <- p + geom_segment(
        data = model_hatch,
        aes(x = x, xend = xend, y = y, yend = yend),
        inherit.aes = FALSE, color = unname(model_cols[[model_name]]), linewidth = 0.55
      )
    }
  }

  p <- p +
    geom_errorbar(
      aes(x = x, ymin = pmax(0, top100 - top100_sd), ymax = top100 + top100_sd),
      width = 0.18, linewidth = 0.50, color = "#111827"
    ) +
    geom_errorbar(
      aes(x = x, ymin = pmax(0, top200 - top200_sd), ymax = top200 + top200_sd),
      width = 0.22, linewidth = 0.50, color = "#111827"
    ) +
    geom_text(
      aes(x = x, y = pmax(0.025, top100 - 0.035), label = sprintf("%.2f", top100)),
      size = 3.0, color = "#111827", show.legend = FALSE
    ) +
    geom_text(
      aes(x = x, y = top200 + 0.018, label = sprintf("%.2f", top200)),
      size = 3.0, color = "#111827", show.legend = FALSE
    ) +
    geom_text(
      data = stars, aes(x = x, y = top200 + 0.055, label = stars),
      inherit.aes = FALSE, size = 4.2, color = "#111827"
    ) +
    geom_hline(
      data = controls,
      aes(yintercept = half_mean, color = control),
      inherit.aes = FALSE, linetype = "dashed", linewidth = 1.05
    ) +
    geom_point(
      data = controls,
      aes(x = 5.35, y = half_mean, color = control),
      inherit.aes = FALSE, shape = 18, size = 3.3
    ) +
    scale_fill_manual(values = model_cols, drop = FALSE, name = NULL) +
    scale_color_manual(
      values = c("Split-half (top 5%)" = "#000000", "Split-half (top 10%)" = "#777777"),
      name = NULL
    ) +
    scale_x_continuous(breaks = seq_along(model_order), labels = model_order, limits = c(0.5, 5.65), expand = c(0, 0)) +
    scale_y_continuous(breaks = seq(0, 0.8, by = 0.2), expand = expansion(mult = c(0, 0.04))) +
    facet_wrap(~side, nrow = 1, labeller = as_labeller(c(young = "Young", old = "Old"))) +
    coord_cartesian(ylim = c(0, 0.75)) +
    labs(
      title = "Top 5% / 10% DE overlap",
      subtitle = NULL,
      x = NULL, y = "Overlap fraction", fill = NULL, color = NULL,
      caption = NULL
    ) +
    base_theme +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5, margin = margin(b = 3)),
      plot.subtitle = element_text(size = 11, hjust = 0.5, margin = margin(b = 5)),
      plot.caption = element_text(size = 9.5, color = "#4b5563", hjust = 0.5, margin = margin(t = 7)),
      axis.text.x = element_text(size = 10.5, color = "#111827", angle = 0),
      axis.title.y = element_text(size = 12.5, margin = margin(r = 8)),
      strip.text = element_text(face = "bold", size = 13),
      legend.position = "top",
      legend.box = "vertical",
      legend.text = element_text(size = 10.5),
      legend.key.width = unit(1.05, "lines"),
      plot.margin = margin(12, 18, 12, 14)
    ) +
    guides(
      fill = guide_legend(order = 1, nrow = 1),
      color = guide_legend(order = 2, nrow = 1, override.aes = list(linewidth = 1.05, shape = 18, size = 3.0))
    )

  save_plot(p, "global_predicted_vs_real_de_shared_reference_top_100_200_overlap", 11.5, 6.2)
}

plot_global_sign_agreement_series <- function() {
  # Show the three global sign-agreement thresholds in one reader-facing
  # series. Model values are averaged across Young and Old; the control is the
  # matching normalized-half versus edgeR-half control, averaged across ages
  # and Young/Old. Age 68 is excluded from these two edgeR-based analyses.
  sign_fields <- c(
    "All genes" = "de_direction_agreement_nonzero",
    "|log₂FC| > 0.25" = "sign_agreement_true_abs_log2fc_gt_0p25",
    "|log₂FC| > 0.5" = "sign_agreement_true_abs_log2fc_gt_0p5"
  )
  split_fields <- c(
    "All genes" = "direction_agreement_nonzero",
    "|log₂FC| > 0.25" = "sign_agreement_true_abs_log2fc_gt_0p25",
    "|log₂FC| > 0.5" = "sign_agreement_true_abs_log2fc_gt_0p5"
  )
  metric_levels <- names(sign_fields)

  model_lines <- bind_rows(lapply(metric_levels, function(metric_label) {
    field <- unname(sign_fields[[metric_label]])
    model_summary(field, "global") %>%
      group_by(model_key, model) %>%
      summarise(value = mean(mean, na.rm = TRUE), .groups = "drop") %>%
      mutate(metric = metric_label)
  })) %>%
    mutate(
      metric = factor(metric, levels = metric_levels),
      model = factor(model, levels = model_order)
    )

  # Read the split-level results rather than reusing only the per-condition
  # means. This gives an empirical interval over the 500 donor-split
  # balanced draws from the available unique partitions. Average across ages
  # and both age sides, matching the mean shown by the control line.
  half_metrics_path <- file.path(results_dir, "real_de_split_half_metrics_global.tsv.gz")
  if (!file.exists(half_metrics_path)) stop("Missing split-level half metrics: ", half_metrics_path)
  half_metrics <- read_tsv(
    half_metrics_path,
    col_select = c(reference, scope, side, split, all_of(unname(split_fields))),
    show_col_types = FALSE, progress = FALSE
  ) %>%
    filter(reference == "sc_ref_whole_dataset", scope == "all", side %in% age_groups) %>%
    pivot_longer(cols = all_of(unname(split_fields)), names_to = "field", values_to = "value") %>%
    group_by(field, split) %>%
    summarise(split_value = mean(value, na.rm = TRUE), .groups = "drop")

  controls <- half_metrics %>%
    group_by(field) %>%
    summarise(
      value = mean(split_value, na.rm = TRUE),
      ymin = as.numeric(quantile(split_value, probs = 0.05, na.rm = TRUE, names = FALSE)),
      ymax = as.numeric(quantile(split_value, probs = 0.95, na.rm = TRUE, names = FALSE)),
      n_splits = n_distinct(split),
      .groups = "drop"
    ) %>%
    mutate(
      metric = metric_levels[match(field, unname(split_fields))],
      control = "Split-half",
      metric = factor(metric, levels = metric_levels)
    ) %>%
    select(metric, control, value, ymin, ymax, n_splits)

  if (any(controls$n_splits != 500L)) {
    stop("Expected 500 split replicates for every global sign-agreement threshold; found ", paste(unique(controls$n_splits), collapse = ", "))
  }
  # Use the exact equal-age mean; balanced draws provide the interval only.
  controls$value <- vapply(as.character(controls$metric), function(label) {
    mean(half_df$value[half_df$scope == "all" & half_df$metric_plot == sign_fields[[label]]], na.rm = TRUE)
  }, numeric(1))
  write_tsv(controls, file.path(results_dir, "aida_whole_ref_sign_agreement_series_control.tsv"), na = "NA")

  if (any(!is.finite(model_lines$value)) || any(!is.finite(controls$value))) {
    stop("Global sign-agreement series contains missing values.")
  }

  # Tests on the Young/Old average of each fit x external age, per threshold.
  sign_stars <- bind_rows(lapply(metric_levels, function(metric_label) {
    field <- unname(sign_fields[[metric_label]])
    record_tests(best_competitor_tests(
      model_df %>% filter(metric == field, scope == "all") %>%
        group_by(model, fold, target_age) %>% summarise(value = mean(value), .groups = "drop") %>%
        mutate(threshold = metric_label), "threshold"), "global_sign_series", field)
  })) %>%
    inner_join(model_lines %>% transmute(model = as.character(model), threshold = as.character(metric), value),
               by = c("model", "threshold")) %>%
    mutate(metric = factor(threshold, levels = metric_levels),
           y = value + if_else(model == "scCody-G", 0.006, -0.006))

  p <- ggplot() +
    geom_ribbon(
      data = controls,
      aes(x = metric, ymin = ymin, ymax = ymax, group = 1),
      inherit.aes = FALSE, fill = "#d1d5db", alpha = 0.55, show.legend = FALSE
    ) +
    geom_line(
      data = model_lines,
      aes(x = metric, y = value, color = model, group = model),
      linewidth = 1.05
    ) +
    geom_point(
      data = model_lines,
      aes(x = metric, y = value, color = model),
      size = 3.1
    ) +
    geom_line(
      data = controls,
      aes(x = metric, y = value, linetype = control, group = 1),
      inherit.aes = FALSE, color = "#000000", linewidth = 1.15
    ) +
    geom_point(
      data = controls,
      aes(x = metric, y = value),
      inherit.aes = FALSE, color = "#000000", shape = 18, size = 3.8
    ) +
    geom_text(
      data = sign_stars, aes(x = metric, y = y, label = stars),
      inherit.aes = FALSE, hjust = 1, nudge_x = -0.08, size = 4.0, color = "#111827"
    ) +
    scale_color_manual(values = model_cols, limits = model_order, drop = FALSE, name = NULL) +
    scale_linetype_manual(values = c("Split-half" = "dashed"), name = NULL) +
    scale_x_discrete(drop = FALSE) +
    scale_y_continuous(
      breaks = scales::pretty_breaks(n = 5),
      labels = scales::label_number(accuracy = 0.01),
      expand = expansion(mult = c(0, 0.02))
    ) +
    coord_cartesian(ylim = range(c(model_lines$value, controls$ymin, controls$ymax), finite = TRUE) + c(-0.035, 0.045)) +
    labs(
      title = "DE sign concordance",
      x = NULL, y = "Sign agreement"
    ) +
    base_theme +
    theme(
      plot.title = element_text(size = 16, hjust = 0.5, margin = margin(b = 10)),
      axis.text.x = element_text(size = 11.5, color = "#111827"),
      axis.text.y = element_text(size = 10.5, color = "#111827"),
      axis.title.y = element_text(size = 12.5, margin = margin(r = 8)),
      legend.position = "top",
      legend.text = element_text(size = 10.5),
      legend.key.width = unit(1.15, "lines"),
      plot.margin = margin(12, 18, 14, 14)
    ) +
    guides(
      color = guide_legend(order = 1, nrow = 1),
      linetype = guide_legend(order = 2, override.aes = list(color = "#000000", linewidth = 1.15))
    )

  save_plot(p, "global_sign_agreement_series", 11.5, 5.8)
}

write_model_category_rank_table <- function() {
  # Compute the category ranks from the same whole-AIDA source used
  # for the model figures. Each component is averaged over Young/Old first.
  side_average <- function(field_name, celltype = FALSE) {
    subset <- model_df %>% filter(metric == field_name, is.finite(value))
    if (celltype) {
      subset %>%
        filter(scope %in% cell_types) %>%
        group_by(model, side, scope) %>%
        summarise(component = mean(value), .groups = "drop") %>%
        group_by(model, side) %>%
        summarise(side_mean = mean(component), .groups = "drop") %>%
        group_by(model) %>%
        summarise(value = mean(side_mean), .groups = "drop")
    } else {
      subset %>%
        filter(scope == "all") %>%
        group_by(model, side) %>%
        summarise(side_mean = mean(value), .groups = "drop") %>%
        group_by(model) %>%
        summarise(value = mean(side_mean), .groups = "drop")
    }
  }

  component <- function(field_name, celltype = FALSE) {
    side_average(field_name, celltype = celltype) %>%
      rename(!!field_name := value)
  }

  global_components <- Reduce(
    function(left, right) left_join(left, right, by = "model"),
    lapply(c(
      "de_vs_reference_logfc_pearson_all", "de_vs_reference_logfc_spearman_all",
      "de_vs_reference_top100_overlap_frac", "de_vs_reference_top200_overlap_frac",
      "de_direction_agreement_nonzero", "sign_agreement_true_abs_log2fc_gt_0p25", "sign_agreement_true_abs_log2fc_gt_0p5"
    ), component)
  ) %>%
    mutate(
      de_effect = rowMeans(across(c(de_vs_reference_logfc_pearson_all, de_vs_reference_logfc_spearman_all))),
      top_de = rowMeans(across(c(de_vs_reference_top100_overlap_frac, de_vs_reference_top200_overlap_frac))),
      direction = rowMeans(across(c(de_direction_agreement_nonzero, sign_agreement_true_abs_log2fc_gt_0p25, sign_agreement_true_abs_log2fc_gt_0p5)))
    )

  celltype_components <- Reduce(
    function(left, right) left_join(left, right, by = "model"),
    lapply(c(
      "de_vs_reference_logfc_pearson_all", "de_vs_reference_logfc_spearman_all",
      "de_vs_reference_top100_overlap_frac", "de_vs_reference_top200_overlap_frac",
      "de_direction_agreement_nonzero"
    ), function(field_name) component(field_name, celltype = TRUE))
  ) %>%
    mutate(
      celltype_robustness = rowMeans(across(c(
        de_vs_reference_logfc_pearson_all, de_vs_reference_logfc_spearman_all,
        de_vs_reference_top100_overlap_frac, de_vs_reference_top200_overlap_frac,
        de_direction_agreement_nonzero
      )))
    )

  expression_path <- file.path(results_dir, "mean_expression_correlation_global.tsv")
  if (!file.exists(expression_path)) stop("Missing mean-expression correlation table: ", expression_path)
  population <- read_tsv(expression_path, show_col_types = FALSE) %>%
    mutate(model = recode(as.character(model), "scCody-Global" = "scCody-G", "scCody-Cell type" = "scCody-CT", "FlowMap-global" = "scCody-G", "FlowMap-cell-type" = "scCody-CT", .default = as.character(model))) %>%
    filter(scope == "Global", is.finite(mean_expression_corr)) %>%
    group_by(model) %>%
    summarise(population = mean(mean_expression_corr), .groups = "drop")

  scores <- population %>%
    left_join(global_components %>% select(model, de_effect, top_de, direction), by = "model") %>%
    left_join(celltype_components %>% select(model, celltype_robustness), by = "model") %>%
    filter(model %in% model_order) %>%
    mutate(model = factor(model, levels = model_order))
  if (nrow(scores) != length(model_order) || any(!complete.cases(scores))) stop("Incomplete model category values; refusing to write a partial rank table.")

  scores <- scores %>%
    mutate(
      population_rank = min_rank(desc(population)),
      de_effect_rank = min_rank(desc(de_effect)),
      top_de_rank = min_rank(desc(top_de)),
      direction_rank = min_rank(desc(direction)),
      celltype_robustness_rank = min_rank(desc(celltype_robustness)),
      overall_mean_rank = rowMeans(across(c(population_rank, de_effect_rank, top_de_rank, direction_rank, celltype_robustness_rank)))
    ) %>%
    arrange(overall_mean_rank, model)
  write_tsv(scores, file.path(results_dir, "aida_model_category_scores.tsv"), na = "NA")

  scores_by_model <- scores %>% arrange(match(as.character(model), model_order))
  category_labels <- c(
    "Population reconstruction", "DE effect recovery",
    "Top-DE recovery", "DE direction recovery", "Cell-type robustness", "Overall mean rank"
  )
  rank_table <- list(Category = category_labels)
  for (model_name in model_order) {
    model_row <- scores_by_model %>% filter(as.character(model) == model_name)
    rank_table[[model_name]] <- c(
      model_row$population_rank, model_row$de_effect_rank,
      model_row$top_de_rank, model_row$direction_rank, model_row$celltype_robustness_rank,
      sprintf("%.2f", model_row$overall_mean_rank)
    )
  }
  rank_table <- as_tibble(rank_table)
  write_tsv(rank_table, file.path(results_dir, "aida_model_category_rankings.tsv"), na = "NA")

  table_theme <- ttheme_minimal(
    base_size = 12.5, base_colour = "#111827", base_family = "Arial",
    padding = unit(c(4, 5), "mm")
  )
  table_theme$colhead$fg_params <- modifyList(table_theme$colhead$fg_params, list(fontface = "bold", fontfamily = "Arial", col = "#111827"))
  table_theme$core$fg_params <- modifyList(table_theme$core$fg_params, list(fontfamily = "Arial", col = "#111827"))
  table_theme$colhead$bg_params <- modifyList(table_theme$colhead$bg_params, list(fill = "#e5e7eb", col = "#9ca3af", lwd = 0.7))
  table_theme$core$bg_params <- modifyList(table_theme$core$bg_params, list(fill = rep(c("#ffffff", "#f8fafc"), length.out = nrow(rank_table)), col = "#d1d5db", lwd = 0.5))
  table_grob <- tableGrob(rank_table, rows = NULL, theme = table_theme)
  # Color rank cells from green (best) through red (worst); the model-name
  # column remains neutral for readability.
  rank_fill <- function(value) {
    palette <- grDevices::colorRampPalette(c("#d9ead3", "#f4cccc"))(101)
    index <- round(pmin(1, pmax(0, (as.numeric(value) - 1) / 4)) * 100) + 1
    palette[index]
  }
  body_bg <- which(table_grob$layout$name == "core-bg")
  for (grob_index in body_bg) {
    row_index <- table_grob$layout$t[grob_index] - 1L
    col_index <- table_grob$layout$l[grob_index]
    if (col_index == 1L) {
      fill <- if (row_index %% 2L) "#ffffff" else "#f8fafc"
    } else {
      fill <- rank_fill(rank_table[[col_index]][[row_index]])
    }
    table_grob$grobs[[grob_index]]$gp$fill <- fill
    table_grob$grobs[[grob_index]]$gp$col <- "#ffffff"
  }
  # Rotate only model headers so the table body can stay compact while the
  # category names remain horizontal and easy to scan.
  header_fg <- which(table_grob$layout$name == "colhead-fg")
  for (grob_index in header_fg) {
    if (table_grob$grobs[[grob_index]]$label %in% model_order) {
      table_grob$grobs[[grob_index]]$rot <- 60
    }
  }
  table_grob$widths <- unit(c(2.85, rep(0.72, length(model_order))), "in")
  table_grob$heights <- unit(c(0.85, rep(0.43, nrow(rank_table))), "in")
  stem <- panel_path("aida_model_category_rankings")
  for (fmt in c("svg", "pdf", "png")) {
    path <- paste0(stem, ".", fmt)
    if (fmt == "svg") svglite::svglite(path, width = 10, height = 4.35, system_fonts = list(sans = "Arial"))
    if (fmt == "pdf") grDevices::cairo_pdf(path, width = 10, height = 4.35, family = "Arial")
    if (fmt == "png") ragg::agg_png(path, width = 10, height = 4.35, units = "in", res = 360)
    grid::grid.newpage(); grid::grid.draw(table_grob); invisible(dev.off())
  }
}

if (ranking_only) {
  write_model_category_rank_table()
  quit(save = "no", status = 0)
}

pearson_spec <- metric_specs %>% filter(field == "de_vs_reference_logfc_pearson_all")
plot_global(pearson_spec)
plot_breakdown(pearson_spec)
plot_global_top100_top200_composite()
plot_global_sign_agreement_series()
write_model_category_rank_table()
write_tsv(distinct(bind_rows(test_records)), file.path(results_dir, "figure2_best_competitor_tests.tsv"), na = "NA")
cat("[saved] Main Figure 2 panels -> ", outdir, "\n", sep = "")
