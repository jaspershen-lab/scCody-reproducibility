#!/usr/bin/env Rscript

# AIDA global localization: the five dynamic models from assemble_localization.R
# plus the cutoff-5 Lasso, ElasticNet, and tAge refits (sccody_analyses.refit_supervised).
# Run from the repository root:  Rscript manuscript/figure_3/scripts/Figure3D.R

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(tidyr)
})

root <- normalizePath(".", mustWork = TRUE)
outdir <- file.path(root, "manuscript/figure_3/results")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

model_order <- c(
  "scCody-G", "scCody-CT", "scNODE", "scDiffEq", "ARTEMIS",
  "Lasso", "ElasticNet", "tAge"
)
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

safe_pearson <- function(truth, prediction) {
  keep <- is.finite(truth) & is.finite(prediction)
  truth <- truth[keep]
  prediction <- prediction[keep]
  if (length(truth) < 2L || sd(truth) == 0 || sd(prediction) == 0) {
    return(NA_real_)
  }
  cor(truth, prediction, method = "pearson")
}

bootstrap_metrics <- function(frame, model_name, source_name, seed, repeats = 2000L) {
  truth <- as.numeric(frame$true_age)
  prediction <- as.numeric(frame$predicted_age)
  keep <- is.finite(truth) & is.finite(prediction)
  truth <- truth[keep]
  prediction <- prediction[keep]
  if (length(truth) != 71L) {
    stop(model_name, " must contain 71 complete external donors; found ", length(truth), ".")
  }

  set.seed(seed)
  bootstrap_mae <- numeric(repeats)
  bootstrap_r <- numeric(repeats)
  for (iteration in seq_len(repeats)) {
    indices <- sample.int(length(truth), length(truth), replace = TRUE)
    bootstrap_mae[[iteration]] <- mean(abs(prediction[indices] - truth[indices]))
    bootstrap_r[[iteration]] <- safe_pearson(truth[indices], prediction[indices])
  }

  tibble(
    model = model_name,
    metric = c("MAE", "pearson_r"),
    value = c(
      mean(abs(prediction - truth)),
      safe_pearson(truth, prediction)
    ),
    ci_low = c(
      unname(quantile(bootstrap_mae, 0.025, na.rm = TRUE)),
      unname(quantile(bootstrap_r, 0.025, na.rm = TRUE))
    ),
    ci_high = c(
      unname(quantile(bootstrap_mae, 0.975, na.rm = TRUE)),
      unname(quantile(bootstrap_r, 0.975, na.rm = TRUE))
    ),
    n_unique_donors = length(truth),
    source = source_name
  )
}

read_clock_predictions <- function(path, model_name, source_name, seed) {
  if (!file.exists(path)) stop("Missing prediction file: ", path)
  frame <- read_tsv(path, show_col_types = FALSE, progress = FALSE) %>%
    transmute(
      donor_id = as.character(donor_id),
      true_age = as.numeric(age),
      predicted_age = as.numeric(predicted_age),
      n_folds = as.numeric(n_folds)
    )
  if (nrow(frame) != 71L || anyDuplicated(frame$donor_id) || any(frame$n_folds != 3)) {
    stop(model_name, " prediction file is not a complete 71-donor three-fold ensemble.")
  }
  bootstrap_metrics(frame, model_name, source_name, seed)
}

canonical_metrics_path <- file.path(
  outdir,
  "aida_selected_external_ensemble_performance_metrics.tsv"
)
canonical_metrics <- read_tsv(canonical_metrics_path, show_col_types = FALSE) %>%
  filter(metric %in% c("MAE", "pearson_r")) %>%
  mutate(
    value = as.numeric(value),
    ci_low = as.numeric(ci_low),
    ci_high = as.numeric(ci_high),
    source = "internally selected scCody, scNODE, scDiffEq and ARTEMIS; donor bootstrap"
  ) %>%
  select(model, metric, value, ci_low, ci_high, n_unique_donors, source)

new_metrics <- bind_rows(
  read_clock_predictions(
    file.path(root, "manuscript/figure_3/results/clocks/lasso_global/external_predictions_ensemble.tsv"),
    "Lasso",
    "cutoff-5 global Lasso",
    20260914L
  ),
  read_clock_predictions(
    file.path(root, "manuscript/figure_3/results/clocks/elasticnet_global/external_predictions_ensemble.tsv"),
    "ElasticNet",
    "cutoff-5 whole-population ElasticNet; l1_ratio < 1",
    20260915L
  ),
  read_clock_predictions(
    file.path(root, "manuscript/figure_3/results/clocks/tage/external_predictions_ensemble.tsv"),
    "tAge",
    "cutoff-5 tAge per-cell-type donor aggregate",
    20260916L
  )
)

metrics <- bind_rows(canonical_metrics, new_metrics) %>%
  mutate(
    model = factor(model, levels = model_order),
    metric = factor(metric, levels = c("MAE", "pearson_r"))
  ) %>%
  arrange(metric, model)

if (anyDuplicated(metrics %>% select(model, metric))) {
  stop("Duplicate model/metric rows found in the combined comparison.")
}

write_tsv(
  metrics %>% mutate(across(where(is.factor), as.character)),
  file.path(outdir, "aida_selected_external_ensemble_localization_with_lasso_elasticnet_tage_metrics.tsv"),
  na = ""
)

plot_data <- metrics %>%
  mutate(
    metric = factor(
      metric,
      levels = c("MAE", "pearson_r"),
      labels = c("MAE (years) ↓", "Pearson r ↑")
    ),
    value_label = case_when(
      is.na(value) ~ NA_character_,
      as.character(metric) == "Pearson r ↑" ~ sprintf("%.3f", value),
      TRUE ~ sprintf("%.2f", value)
    ),
    label_y = case_when(
      is.na(value) ~ NA_real_,
      as.character(metric) == "Pearson r ↑" & value >= 0 ~ pmax(value + 0.025, ci_high + 0.025),
      as.character(metric) == "Pearson r ↑" ~ pmin(value - 0.025, ci_low - 0.025),
      TRUE ~ pmax(value + 0.30, ci_high + 0.30)
    )
  )

# Stars: scCody-G/CT vs the best dynamics competitor per metric on the same 71
# donors (MAE: paired t-test of absolute errors; r: Williams' test), unadjusted.
source(file.path(root, "manuscript/significance.R"))
best_tests <- read_tsv(file.path(outdir, "aida_scCody_vs_best_dynamics_pvalues.tsv"), show_col_types = FALSE) %>%
  transmute(model = scody_model,
            metric = recode(comparison, "MAE" = "MAE (years) ↓", "Pearson r" = "Pearson r ↑"),
            stars = direction_marks(p_value, if_else(comparison == "MAE", difference_scody_minus_dynamic < 0,
                                                     difference_scody_minus_dynamic > 0)))
plot_data <- plot_data %>%
  mutate(model_chr = as.character(model), metric_chr = as.character(metric)) %>%
  left_join(best_tests, by = c("model_chr" = "model", "metric_chr" = "metric")) %>%
  mutate(stars = coalesce(stars, ""),
         star_y = label_y + if_else(metric_chr == "Pearson r ↑", if_else(value >= 0, 0.06, -0.06), 1.2)) %>%
  select(-model_chr, -metric_chr)

base_theme <- theme_classic(base_size = 12, base_family = "Arial") +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5, margin = margin(b = 5)),
    strip.text = element_text(face = "bold", size = 13),
    axis.text.x = element_text(size = 10.5, color = "#111827", angle = 20, hjust = 1),
    axis.text.y = element_text(size = 10.5, color = "#111827"),
    axis.title.y = element_text(size = 12.5, margin = margin(r = 8)),
    strip.background = element_blank(),
    panel.grid.major.y = element_line(color = "#e5e7eb", linewidth = 0.35),
    panel.grid.minor = element_blank(),
    axis.line = element_line(color = "#111827", linewidth = 0.35),
    axis.ticks = element_line(color = "#111827", linewidth = 0.35),
    panel.spacing = grid::unit(1.2, "lines"),
    plot.margin = margin(12, 18, 12, 14)
  )

p <- ggplot(plot_data, aes(model, value, fill = model)) +
  geom_col(
    width = 0.64,
    color = "#1f2937",
    linewidth = 0.25,
    alpha = 0.88
  ) +
  geom_vline(
    xintercept = 5.5,
    color = "black",
    linewidth = 0.9,
    inherit.aes = FALSE
  ) +
  geom_errorbar(
    aes(ymin = ci_low, ymax = ci_high),
    width = 0.14,
    linewidth = 0.5,
    color = "#111827"
  ) +
  geom_text(
    aes(y = star_y, label = stars),
    size = 4.2,
    color = "#111827",
    na.rm = TRUE,
    show.legend = FALSE
  ) +
  geom_text(
    aes(y = label_y, label = value_label),
    size = 3.4,
    color = "#111827",
    na.rm = TRUE,
    show.legend = FALSE
  ) +
  scale_fill_manual(values = model_cols, drop = FALSE, guide = "none") +
  scale_x_discrete(drop = FALSE) +
  facet_wrap(
    ~metric,
    scales = "free_y",
    nrow = 1,
    labeller = as_labeller(c("MAE (years) ↓" = "MAE (years) ↓", "Pearson r ↑" = "Pearson r ↑"))
  ) +
  scale_y_continuous(
    breaks = scales::pretty_breaks(n = 5),
    expand = expansion(mult = c(0.06, 0.16))
  ) +
  expand_limits(y = 0) +
  labs(
    title = "Age prediction",
    x = NULL,
    y = NULL
  ) +
  base_theme

panel_dir <- file.path(root, "manuscript/figure_3/figure_files/panels")
dir.create(panel_dir, recursive = TRUE, showWarnings = FALSE)
output_stem <- file.path(
  panel_dir,
  "Figure3D.localization_mae_r"
)
ggsave(
  paste0(output_stem, ".svg"),
  p,
  width = 14,
  height = 6.3,
  units = "in",
  device = svglite::svglite,
  system_fonts = list(sans = "Arial"),
  bg = "white"
)

ggsave(
  paste0(output_stem, ".png"),
  p,
  width = 14,
  height = 6.3,
  units = "in",
  dpi = 360,
  device = ragg::agg_png,
  bg = "white"
)

ggsave(
  paste0(output_stem, ".pdf"),
  p,
  width = 14,
  height = 6.3,
  units = "in",
  device = grDevices::cairo_pdf,
  bg = "white"
)
