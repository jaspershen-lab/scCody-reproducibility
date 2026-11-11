#!/usr/bin/env Rscript

# Figure 3f-h: annual scCody-CT projections of immune-aging programs for held-out
# age-31 source donors (3f Cellular senescence, 3g TNF-NFkB, 3h BCR/TCR signaling).
# Scores come from sccody_analyses.annual_programs (export, then score) with the
# AIDA CT checkpoints. Run from the repository root:
#   Rscript manuscript/figure_3/scripts/Figure3FGH.R
# Plot annual scCody projections for held-out age-31 source donors. The
# selected age-31 donors remain held out, but most intermediate target ages
# overlap ages seen during model training.

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

args <- commandArgs(trailingOnly = TRUE)
score_path <- if (length(args) >= 1) args[[1]] else
  "manuscript/figure_3/results/annual_programs_cell_type/heldout_projection_program_scores.tsv.gz"
expectation_path <- if (length(args) >= 2) args[[2]] else
  "configs/analyses/aging_program_biological_expectations.tsv"
output_dir <- if (length(args) >= 3) args[[3]] else
  "manuscript/figure_3/figure_files"

model_scope <- if (length(args) >= 4) args[[4]] else "cell_type"
if (!model_scope %in% c("global", "cell_type")) stop("Model scope must be global or cell_type")
model_label <- if (model_scope == "global") "scCody-G" else "scCody-CT"
model_colour <- if (model_scope == "global") "#0072B2" else "#009E73"

for (path in c(score_path, expectation_path)) {
  if (!file.exists(path)) stop("Missing input: ", path)
}
# Render only the programs selected for Figure 3f-h.
main_programs <- c(Cellular_senescence = "Figure3F.cellular_senescence", TNF_NFkB = "Figure3G.tnf_nfkb",
                   BCR_TCR_signaling = "Figure3H.bcr_tcr_signaling")
program_stem <- function(program) {
  if (program %in% names(main_programs)) file.path(output_dir, "panels", main_programs[[program]])
  else stop("Unselected program: ", program)
}
dir.create(file.path(output_dir, "panels"), recursive = TRUE, showWarnings = FALSE)

scores <- read.delim(gzfile(score_path), stringsAsFactors = FALSE, check.names = FALSE)
expectations <- read.delim(expectation_path, stringsAsFactors = FALSE, check.names = FALSE)

selected_programs <- c(
  "TNF_NFkB", "Cellular_senescence",
  "BCR_signaling", "TCR_signaling"
)
plot_order <- c("T", "Myeloid", "B", "NK", "DC")
display_order <- c(
  T = "T", Myeloid = "Myeloid", B = "B", NK = "NK", DC = "DC"
)

expectation_row <- function(program) {
  row <- expectations[expectations$program == program, , drop = FALSE]
  if (nrow(row) != 1) stop("Expected exactly one expectation row for ", program)
  row
}

eligible_lineages <- function(program) {
  row <- expectation_row(program)
  allowed <- strsplit(row$applicable_lineages[[1]], ",", fixed = TRUE)[[1]]
  plot_order[plot_order %in% allowed]
}

mean_se <- function(values) {
  values <- values[is.finite(values)]
  if (!length(values)) return(c(mean = NA_real_, se = NA_real_))
  c(mean = mean(values), se = if (length(values) > 1) sd(values) / sqrt(length(values)) else NA_real_)
}

trajectory_panel <- function(program, cell_type, observed_ages,
                             show_x_axis_title = TRUE,
                             show_y_axis_title = TRUE) {
  model <- scores[
    scores$dataset == "flowmap_projected" &
      scores$program == program & scores$cell_type == cell_type,
    c("fold", "donor_id", "age", "score"), drop = FALSE
  ]
  real <- scores[
    scores$dataset == "aida_observed" &
      scores$program == program & scores$cell_type == cell_type &
      scores$age %in% observed_ages &
      scores$age >= min(model$age) & scores$age <= max(model$age),
    c("age", "score"), drop = FALSE
  ]
  if (!nrow(model)) stop("No projected scores for ", program, "/", cell_type)
  model <- model[order(model$fold, model$donor_id, model$age), , drop = FALSE]
  group_key <- interaction(model$fold, model$donor_id, drop = TRUE, lex.order = TRUE)
  model$change <- model$score - ave(model$score, group_key, FUN = function(x) x[[1]])

  donor <- aggregate(change ~ fold + donor_id + age, data = model, FUN = mean)
  fold_age <- aggregate(change ~ fold + age, data = donor, FUN = mean)
  center <- aggregate(change ~ age, data = fold_age, FUN = mean)
  across_folds <- aggregate(change ~ donor_id + age, data = donor, FUN = mean)
  spread <- do.call(rbind, lapply(split(across_folds$change, across_folds$age), mean_se))
  spread <- data.frame(age = as.numeric(names(split(across_folds$change, across_folds$age))), spread)
  spread <- spread[order(spread$age), , drop = FALSE]
  names(center)[names(center) == "change"] <- "mean_center"
  names(spread)[names(spread) == "mean"] <- "mean_spread"
  model_band <- merge(center, spread, by = "age")
  model_band$ymin <- model_band$mean_spread - 1.96 * model_band$se
  model_band$ymax <- model_band$mean_spread + 1.96 * model_band$se

  real_plot <- NULL
  if (nrow(real) && 31 %in% real$age) {
    real_summary <- do.call(rbind, lapply(split(real$score, real$age), mean_se))
    real_summary <- data.frame(age = as.numeric(names(split(real$score, real$age))), real_summary)
    real_summary <- real_summary[order(real_summary$age), , drop = FALSE]
    baseline <- real_summary$mean[real_summary$age == 31][[1]]
    real_summary$change <- real_summary$mean - baseline
    real_summary$ymin <- real_summary$change - 1.96 * real_summary$se
    real_summary$ymax <- real_summary$change + 1.96 * real_summary$se
    real_plot <- real_summary
  }

  model_band$cell_type <- cell_type
  if (!is.null(real_plot)) real_plot$cell_type <- cell_type
  p <- ggplot() +
    geom_ribbon(
      data = model_band,
      aes(x = age, ymin = ymin, ymax = ymax, fill = model_label),
      alpha = .18
    ) +
    labs(
      title = display_order[[cell_type]],
      x = if (show_x_axis_title) "Age" else NULL,
      y = if (show_y_axis_title) "\u0394 program score" else NULL,
      colour = "Relative to age 31"
    ) +
    scale_colour_manual(
      values = c(setNames(model_colour, model_label), "AIDA reference" = "#7F7F7F"),
      breaks = c(model_label, "AIDA reference"),
      drop = TRUE
    ) +
    scale_fill_manual(
      values = setNames(model_colour, model_label),
      guide = "none"
    ) +
    # Match the paper benchmark figures' classic theme, Arial family, and
    # print-size typography (plot_aida_mean_expression_correlation_20260826.R).
    theme_classic(base_size = 12, base_family = "Arial") +
    theme(
      plot.title = element_text(face = "bold", hjust = .5, size = 12,
                                margin = margin(b = 5)),
      axis.text.x = element_text(size = 10.5, colour = "#111827"),
      axis.text.y = element_text(size = 10.5, colour = "#111827"),
      axis.title.x = if (show_x_axis_title) {
        element_text(size = 12.5, margin = margin(t = 7))
      } else element_blank(),
      axis.title.y = if (show_y_axis_title) {
        element_text(size = 12.5, margin = margin(r = 8))
      } else element_blank(),
      axis.line = element_line(colour = "#111827", linewidth = .35),
      axis.ticks = element_line(colour = "#111827", linewidth = .35),
      axis.ticks.length = unit(2.5, "pt"),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.box = "horizontal",
      legend.justification = "center",
      legend.title = element_text(size = 10.5),
      legend.text = element_text(size = 10.5),
      legend.key.width = unit(1.2, "lines"),
      plot.margin = margin(4, 4, 8, 4)
    )
  if (!is.null(real_plot)) {
    p <- p +
      geom_errorbar(
        data = real_plot,
        aes(x = age, ymin = ymin, ymax = ymax, colour = "AIDA reference"),
        width = 0, linewidth = .35, alpha = .58
      ) +
      geom_point(
        data = real_plot,
        aes(x = age, y = change, colour = "AIDA reference"),
        shape = 15, size = 1.8, alpha = .78
      )
  }
  # Draw the model-colored projected values last so they stay visible where
  # they overlap the grey AIDA reference curve and its error bars.
  p <- p +
    geom_line(
      data = model_band,
      aes(x = age, y = mean_center, colour = model_label),
      linewidth = .85
    ) +
    geom_point(
      data = model_band,
      aes(x = age, y = mean_center, colour = model_label),
      size = 1.55
    )
  p <- p + coord_cartesian(xlim = range(model_band$age))
  p
}

save_single_program <- function(program, lineages, observed_ages) {
  ncol <- min(3, length(lineages))
  nrows <- ceiling(length(lineages) / ncol)
  panels <- lapply(seq_along(lineages), function(i) {
    row_start <- (nrows - 1) * ncol + 1
    trajectory_panel(
      program, lineages[[i]], observed_ages,
      show_x_axis_title = i >= row_start,
      show_y_axis_title = ((i - 1) %% ncol) == 0
    )
  })
  figure <- wrap_plots(panels, ncol = ncol, guides = "collect") &
    theme(legend.position = "bottom", legend.direction = "horizontal")
  width <- if (length(lineages) == 1) 10 else 14
  height <- if (length(lineages) == 1) 5 else 4.1 * nrows
  ggsave(paste0(program_stem(program), ".pdf"), figure,
         width = width, height = height, units = "in", device = cairo_pdf)
  ggsave(paste0(program_stem(program), ".png"), figure,
         width = width, height = height, units = "in", dpi = 360, bg = "white")
  ggsave(paste0(program_stem(program), ".svg"), figure,
         width = width, height = height, units = "in", device = svglite::svglite,
         system_fonts = list(sans = "Arial"), bg = "white")
}

observed_ages <- sort(unique(scores$age[scores$dataset == "aida_observed"]))
for (program in selected_programs[!selected_programs %in% c("BCR_signaling", "TCR_signaling")]) {
  lineages <- eligible_lineages(program)
  save_single_program(program, lineages, observed_ages)
}

# BCR and TCR are intentionally combined into one figure, each with only its
# eligible lineage panel.
bcr <- trajectory_panel(
  "BCR_signaling", "B", observed_ages,
  show_x_axis_title = TRUE, show_y_axis_title = TRUE
)
tcr <- trajectory_panel(
  "TCR_signaling", "T", observed_ages,
  show_x_axis_title = TRUE, show_y_axis_title = FALSE
)
combined <- (bcr | tcr) + plot_layout(guides = "collect") &
  theme(legend.position = "bottom", legend.direction = "horizontal")
# Match the actual SVG plot-area bounds of a Cellular senescence panel
# (290.97 x 217.69 pt), accounting for fixed axes, labels, margins, and legend.
# These canvas dimensions yield those plot-area bounds in the rendered SVG.
bcr_tcr_width <- 674.36 / 72
bcr_tcr_height <- 330.06 / 72
ggsave(paste0(program_stem("BCR_TCR_signaling"), ".pdf"), combined,
       width = bcr_tcr_width, height = bcr_tcr_height, units = "in", device = cairo_pdf)
ggsave(paste0(program_stem("BCR_TCR_signaling"), ".png"), combined,
       width = bcr_tcr_width, height = bcr_tcr_height, units = "in", dpi = 360, bg = "white")
ggsave(paste0(program_stem("BCR_TCR_signaling"), ".svg"), combined,
       width = bcr_tcr_width, height = bcr_tcr_height, units = "in", device = svglite::svglite,
       system_fonts = list(sans = "Arial"), bg = "white")


cat("Wrote eligible R trajectory plots to ", output_dir, "\n", sep = "")
