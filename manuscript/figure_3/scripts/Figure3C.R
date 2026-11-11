#!/usr/bin/env Rscript

# Plot the three-fold ensemble prediction for every AIDA age-localization
# model in separate panels.
# Run from the repository root:  Rscript manuscript/figure_3/scripts/Figure3C.R

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
})

root <- normalizePath(".", mustWork = TRUE)
result_dir <- file.path(root, "manuscript/figure_3/results")
canonical_path <- file.path(result_dir, "aida_selected_external_ensemble_per_donor.tsv")
panel_dir <- file.path(root, "manuscript/figure_3/figure_files/panels")
dir.create(panel_dir, recursive = TRUE, showWarnings = FALSE)
output_stem <- file.path(panel_dir, "Figure3C.age_real_vs_predicted")

model_levels <- c("scCody-G", "scCody-CT", "scNODE", "scDiffEq", "ARTEMIS", "Lasso", "ElasticNet", "tAge")
model_cols <- c(
  "scCody-G" = "#0072B2",
  "scCody-CT" = "#009E73",
  "scNODE" = "#CC79A7",
  "scDiffEq" = "#6A3D9A",
  "ARTEMIS" = "#D55E00",
  "Lasso" = "#E69F00",
  "ElasticNet" = "#56B4E9",
  "tAge" = "#F0E442"
)

canonical <- read_tsv(canonical_path, show_col_types = FALSE) %>%
  transmute(
    model = model,
    donor_id = as.character(donor_id),
    true_age = as.numeric(true_age),
    predicted_age = as.numeric(predicted_age)
  )

read_clock_ensemble <- function(path, model_name) {
  if (!file.exists(path)) stop("Missing ensemble prediction file: ", path)
  read_tsv(path, show_col_types = FALSE, progress = FALSE) %>%
    transmute(
      model = model_name,
      donor_id = as.character(donor_id),
      true_age = as.numeric(age),
      predicted_age = as.numeric(predicted_age)
    )
}

data <- bind_rows(
  canonical,
  read_clock_ensemble(
    file.path(root, "manuscript/figure_3/results/clocks/lasso_global/external_predictions_ensemble.tsv"),
    "Lasso"
  ),
  read_clock_ensemble(
    file.path(root, "manuscript/figure_3/results/clocks/elasticnet_global/external_predictions_ensemble.tsv"),
    "ElasticNet"
  ),
  read_clock_ensemble(
    file.path(root, "manuscript/figure_3/results/clocks/tage/external_predictions_ensemble.tsv"),
    "tAge"
  )
) %>%
  mutate(model = factor(model, levels = model_levels))

stopifnot(nrow(data) == length(model_levels) * 71)
stopifnot(all(data %>% count(model) %>% pull(n) == 71))
stopifnot(all(data %>% count(model, donor_id) %>% pull(n) == 1))

safe_cor <- function(x, y, method = "pearson") {
  keep <- is.finite(x) & is.finite(y)
  x <- x[keep]
  y <- y[keep]
  if (length(x) < 2L || sd(x) == 0 || sd(y) == 0) return(NA_real_)
  cor(x, y, method = method)
}

format_metric <- function(value, digits) {
  ifelse(is.finite(value), sprintf(paste0("%.", digits, "f"), value), "NA")
}

model_metric_labels <- data %>%
  group_by(model) %>%
  summarise(
    mae = mean(abs(predicted_age - true_age)),
    median_mae = median(abs(predicted_age - true_age)),
    pearson_r = safe_cor(true_age, predicted_age, "pearson"),
    spearman_r = safe_cor(true_age, predicted_age, "spearman"),
    .groups = "drop"
  ) %>%
  arrange(model) %>%
  mutate(
    panel_label = paste0(
      "MAE ", format_metric(mae, 2),
      "\nr = ", format_metric(pearson_r, 3)
    )
  )
limits <- range(c(data$true_age, data$predicted_age), na.rm = TRUE) + c(-3, 3)
panel_annotations <- model_metric_labels %>%
  mutate(
    model = factor(model, levels = model_levels),
    x = max(limits) - 0.8,
    y = min(limits) + 0.8
  )

p <- ggplot(data, aes(x = true_age, y = predicted_age, colour = model)) +
  geom_abline(slope = 1, intercept = 0, colour = "#555555",
              linetype = "dashed", linewidth = 0.45) +
  geom_point(size = 2.0, alpha = 0.60, stroke = 0.2) +
  geom_label(
    data = panel_annotations,
    aes(x = x, y = y, label = panel_label),
    inherit.aes = FALSE,
    hjust = 1,
    vjust = 0,
    lineheight = 1.05,
    size = 3.5,
    colour = "#111827",
    fill = "white",
    alpha = 0.80,
    linewidth = 0,
    label.padding = grid::unit(0.21, "lines")
  ) +
  scale_colour_manual(values = model_cols, guide = "none") +
  facet_wrap(~model, ncol = 4, axes = "all", axis.labels = "all") +
  coord_equal(xlim = limits, ylim = limits, expand = FALSE) +
  labs(
    title = "Age prediction",
    x = "Real age (years)",
    y = "Predicted age (years)"
  ) +
  theme_classic(base_size = 12, base_family = "Arial") +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5, margin = margin(b = 8)),
    axis.text = element_text(size = 9.5, color = "#111827"),
    axis.title = element_text(size = 13),
    strip.text = element_text(face = "bold", size = 11.5),
    panel.grid.major = element_line(colour = "#e5e7eb", linewidth = 0.35),
    panel.grid.minor = element_blank(),
    plot.margin = margin(14, 18, 14, 18)
  )

ggsave(paste0(output_stem, ".png"), p, width = 14.0, height = 8.8,
       units = "in", dpi = 360, device = ragg::agg_png, bg = "white")
ggsave(paste0(output_stem, ".svg"), p, width = 14.0, height = 8.8,
       units = "in", device = svglite::svglite,
       system_fonts = list(sans = "Arial"), bg = "white")
ggsave(paste0(output_stem, ".pdf"), p, width = 14.0, height = 8.8,
       units = "in", device = cairo_pdf, bg = "white")

cat(sprintf("Saved %s.[png|svg|pdf]\nrows=%d models=%d donors_per_model=%d\n",
            output_stem, nrow(data), nlevels(data$model), nrow(data) / nlevels(data$model)))
