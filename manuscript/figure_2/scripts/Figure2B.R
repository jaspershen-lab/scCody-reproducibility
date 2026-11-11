#!/usr/bin/env Rscript
# Figure 2b: mean-expression correlation between generated and held-out populations,
# per fit and external age (points), mean +/- SD across fits and ages (bars).
# Input: manuscript/figure_2/results/mean_expression_correlation_long.csv
#   (Figure2B.collect_mean_expression.py). Run from the repository root.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(patchwork)
})

root <- normalizePath(".")
source(file.path(root, "manuscript/significance.R"))
outdir <- file.path(root, "manuscript/figure_2/figure_files/panels")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

model_levels <- c("scCody-G", "scCody-CT", "scNODE", "scDiffEq", "ARTEMIS")
model_cols <- c(
  "scCody-G" = "#0072B2",
  "scCody-CT" = "#009E73",
  "scNODE" = "#CC79A7",
  "scDiffEq" = "#6A3D9A",
  "ARTEMIS" = "#D55E00"
)

normalize_model <- function(x) {
  recode(
    as.character(x),
    "scCody-Global" = "scCody-G",
    "scCody-Cell type" = "scCody-CT",
    .default = as.character(x)
  )
}

rows <- read_csv(file.path(root, "manuscript/figure_2/results/mean_expression_correlation_long.csv"), show_col_types = FALSE) %>%
  mutate(
    model = factor(normalize_model(model), levels = model_levels),
    scope = factor(scope, levels = c("Global", "Cell type")),
    cell_type = factor(cell_type, levels = c("All", "B", "Myeloid", "NK", "T"))
  ) %>%
  filter(is.finite(mean_expression_corr))

summary <- rows %>%
  group_by(scope, cell_type, model) %>%
  summarise(
    mean = mean(mean_expression_corr),
    sd = sd(mean_expression_corr),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    sd = if_else(is.finite(sd), sd, 0),
    ymin = mean - sd,
    ymax = mean + sd,
    # Keep labels visibly clear of the error-bar caps, especially in the
    # near-one global panel where the bars otherwise leave little headroom.
    label_y = ymax + 0.008
  )

# Stars: each scCody arm vs the best competitor in that panel, paired over fit x age (unadjusted).
tests <- best_competitor_tests(
  rows %>% transmute(scope = as.character(scope), cell_type = as.character(cell_type),
                     model = as.character(model), fold, target_age, value = mean_expression_corr),
  c("scope", "cell_type"))
write_tsv(tests, file.path(root, "manuscript/figure_2/results/figure2b_best_competitor_tests.tsv"), na = "NA")
summary <- summary %>%
  left_join(tests %>% select(scope, cell_type, model, stars) %>%
              mutate(scope = factor(scope, levels = levels(summary$scope)),
                     cell_type = factor(cell_type, levels = levels(summary$cell_type)),
                     model = factor(model, levels = model_levels)),
            by = c("scope", "cell_type", "model")) %>%
  mutate(stars = coalesce(stars, ""))

write_csv(summary, file.path(root, "manuscript/figure_2/results/mean_expression_correlation_summary.csv"))

pd <- position_dodge(width = 0.86)
pj <- position_jitterdodge(jitter.width = 0.12, jitter.height = 0, dodge.width = 0.86, seed = 11)

panel_plot <- function(panel_scope, panel_title, show_y_axis = TRUE) {
  panel_summary <- summary %>% filter(scope == panel_scope)
  panel_rows <- rows %>% filter(scope == panel_scope)

  ggplot(panel_summary, aes(x = cell_type, y = mean, fill = model, color = model)) +
    # Bars start at the visible baseline so overflow can remain unclipped.
    geom_hline(yintercept = 0.80, linewidth = 0.4, color = "#6b7280") +
    geom_tile(aes(y = (mean + 0.80) / 2, height = mean - 0.80), position = pd, width = 0.68, color = "#1f2937", linewidth = 0.25, alpha = 0.88) +
    geom_errorbar(
      aes(ymin = ymin, ymax = ymax),
      position = pd, width = 0.18, linewidth = 0.55, color = "#111827"
    ) +
    geom_point(
      data = panel_rows,
      aes(x = cell_type, y = mean_expression_corr, color = model),
      position = pj, inherit.aes = FALSE, size = 1.2, alpha = 0.35
    ) +
    geom_text(
      aes(label = sprintf("%.3f", mean), y = label_y),
      position = pd, vjust = -0.25, size = 3.0, color = "#111827", show.legend = FALSE
    ) +
    geom_text(
      aes(label = stars, y = label_y + 0.012),
      position = pd, vjust = -0.25, size = 3.6, color = "#111827", show.legend = FALSE
    ) +
    scale_fill_manual(values = model_cols, drop = FALSE) +
    scale_color_manual(values = model_cols, drop = FALSE, guide = "none") +
    scale_x_discrete(drop = TRUE) +
    scale_y_continuous(
      breaks = seq(0.80, 1.00, by = 0.05), labels = scales::label_number(accuracy = 0.01),
      expand = expansion(mult = c(0, 0))
    ) +
    coord_cartesian(ylim = c(0.80, 1.00), clip = "off") +
    labs(title = panel_title, x = NULL, y = if (show_y_axis) "Pearson R" else NULL, fill = NULL) +
    theme_classic(base_size = 12, base_family = "Arial") +
    theme(
      plot.title = element_text(face = "bold", size = 12, hjust = 0.5, margin = margin(b = 24)),
      axis.text.x = element_text(size = 10.5, color = "#111827"),
      axis.text.y = if (show_y_axis) element_text(size = 10.5, color = "#111827") else element_blank(),
      axis.title.y = if (show_y_axis) element_text(size = 12.5, margin = margin(r = 8)) else element_blank(),
      axis.ticks.y = if (show_y_axis) element_line(color = "#111827", linewidth = 0.35) else element_blank(),
      legend.position = "top",
      legend.justification = "center",
      legend.text = element_text(size = 10.5),
      panel.grid.major.y = element_line(color = "#e5e7eb", linewidth = 0.35),
      panel.grid.minor = element_blank(),
      axis.line = element_line(color = "#111827", linewidth = 0.35),
      axis.ticks = element_line(color = "#111827", linewidth = 0.35),
      plot.margin = margin(4, 4, 8, 4)
    )
}

p_global <- panel_plot("Global", "Global", show_y_axis = TRUE)
p_cell_type <- panel_plot("Cell type", "Cell type", show_y_axis = FALSE)
p <- (p_global + p_cell_type + guide_area()) +
  plot_layout(widths = c(3, 7), heights = c(1, 8), design = "CC\nAB", guides = "collect") +
  plot_annotation(
    title = "Mean expression",
    theme = theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5, margin = margin(b = 5)),
      plot.margin = margin(10, 18, 8, 14)
    )
  ) &
  theme(legend.position = "top")

ggsave(file.path(outdir, "Figure2B.mean_expression_correlation.png"), p, width = 15.5, height = 6.6, units = "in", dpi = 360, device = ragg::agg_png, bg = "white")
ggsave(file.path(outdir, "Figure2B.mean_expression_correlation.pdf"), p, width = 15.5, height = 6.6, units = "in", device = grDevices::cairo_pdf, bg = "white")
ggsave(file.path(outdir, "Figure2B.mean_expression_correlation.svg"), p, width = 15.5, height = 6.6, units = "in", device = svglite::svglite, system_fonts = list(sans = "Arial"), bg = "white")

print(summary)
cat("[saved] ", outdir, "\n", sep = "")
