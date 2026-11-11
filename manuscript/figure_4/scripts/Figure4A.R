#!/usr/bin/env Rscript
# Figure 4a: local 5-year age-match acceleration, scCody (filled) vs ElasticNet (open);
# AD (PsychAD, epoch 5) and donor-pooled COVID-19. Run from the repository root.

# Two-panel forest plot of the 5-year local age-match acceleration.
#
# Panel A is the direct PsychAD/AD estimate for each population. Panel B pools
# COVID by concatenating donor-level local-age-match effects across cohorts,
# so each point is a donor-average estimate. ElasticNet is overlaid as an
# open circle; scCody is the filled circle.
#
# Run from the repository root:  Rscript manuscript/figure_4/scripts/Figure4A.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(tidyr)
  library(patchwork)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- if (length(script_arg)) sub("^--file=", "", script_arg[[1]]) else ""
repo <- if (nzchar(script_file)) {
  normalizePath(file.path(dirname(script_file), "../../.."), mustWork = TRUE)
} else {
  normalizePath(getwd(), mustWork = TRUE)
}
outdir <- file.path(repo, "manuscript/figure_4/results")
panel_dir <- file.path(repo, "manuscript/figure_4/figure_files/panels")
dir.create(panel_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

summary_path <- file.path(outdir, "age_displacement_summary.csv")
if (!file.exists(summary_path)) stop("Missing input: ", summary_path)
donor_effect_path <- file.path(outdir, "covid_age_displacement_donor_effects.tsv")
if (!file.exists(donor_effect_path)) stop("Missing input: ", donor_effect_path)
elasticnet_donor_path <- file.path(outdir, "covid_elasticnet_age_acceleration_donor_predictions.tsv")
if (!file.exists(elasticnet_donor_path)) stop("Missing input: ", elasticnet_donor_path)
ad_elasticnet_path <- file.path(outdir, "ad_elasticnet_forest_plot_data.csv")
if (!file.exists(ad_elasticnet_path)) stop("Missing input: ", ad_elasticnet_path)

cohort_order <- c("PsychAD_AD", "COVID_Ren", "COVID_COMBAT", "COVID_Stephenson")
cohort_short <- c(
  PsychAD_AD = "AD",
  COVID_Ren = "Ren",
  COVID_COMBAT = "COMBAT",
  COVID_Stephenson = "Stephenson"
)

# Established order used by the PsychAD cell-type panels. COVID uses the
# established COVID order. Neither panel is sorted by effect size.
ad_cell_type_order <- c("Global", "OPC", "IN", "Mural", "Astro", "EN", "Oligo", "Endo", "Immune")
covid_cell_type_order <- c("Global", "B", "Myeloid", "NK", "Platelet", "T")

ad_color <- "#2166AC"
covid_color <- "#D95F02"

age <- readr::read_csv(summary_path, show_col_types = FALSE) %>%
  filter(scope %in% c("global", "cell_type")) %>%
  transmute(
    cohort = as.character(cohort),
    cohort_label = as.character(cohort_label),
    disease = if_else(cohort == "PsychAD_AD", "AD", "COVID"),
    cell_type = as.character(population),
    estimate_years = as.numeric(local_age_match_effect_years),
    ci_low_years = as.numeric(bootstrap_ci_low_years),
    ci_high_years = as.numeric(bootstrap_ci_high_years),
    n_effects = as.integer(n_effects),
    n_disease_donors = as.integer(n_disease_donors),
    n_reference_donors = as.integer(n_reference_donors),
    p_two_sided_t = as.numeric(p_two_sided_t),
    pFDR = as.numeric(pFDR),
    analysis_method = as.character(analysis_method),
    reference_label = as.character(reference_label)
  ) %>%
  filter(cohort %in% cohort_order)

if (!nrow(age)) stop("No rows remain after filtering.")

bootstrap_se <- function(ci_low, ci_high) {
  if (!is.finite(ci_low) || !is.finite(ci_high) || ci_high <= ci_low) return(NA_real_)
  (ci_high - ci_low) / (2 * qnorm(0.975))
}

# Bootstrap the mean of donor-level effects. The resampling unit is the
# disease donor effect, not cells and not all reusable case-control pairs.
bootstrap_donor_mean_ci <- function(values, seed, repeats = 10000L) {
  values <- values[is.finite(values)]
  if (length(values) < 2L) return(c(NA_real_, NA_real_))
  set.seed(seed)
  boot <- replicate(
    repeats,
    mean(sample(values, size = length(values), replace = TRUE))
  )
  as.numeric(stats::quantile(boot, probs = c(0.025, 0.975), names = FALSE))
}

# Construct one local age-match effect per disease donor. Healthy controls may
# be reused across disease donors, but they are averaged within each donor's
# age window before the donor-level bootstrap is applied.
local_donor_effects <- function(frame, age_window = 5.0) {
  frame <- frame %>%
    filter(condition %in% c("case", "control"), is.finite(age), is.finite(acceleration))
  cases <- frame %>% filter(condition == "case")
  controls <- frame %>% filter(condition == "control")
  if (!nrow(cases) || !nrow(controls)) return(tibble())

  rows <- lapply(seq_len(nrow(cases)), function(i) {
    case <- cases[i, ]
    near <- controls %>% filter(abs(age - case$age[[1]]) <= age_window)
    if (!nrow(near)) return(NULL)
    tibble(
      disease_donor = as.character(case$donor_id[[1]]),
      effect_years = as.numeric(case$acceleration[[1]] - mean(near$acceleration)),
      n_nearby_controls = nrow(near)
    )
  })
  bind_rows(rows)
}

summarize_pooled_donor_effects <- function(
    effects, label, method, panel, panel_color, seed
) {
  effects <- effects %>% filter(is.finite(effect_years))
  if (!nrow(effects)) return(tibble())
  ci <- bootstrap_donor_mean_ci(effects$effect_years, seed = seed)
  cohort_counts <- effects %>% count(cohort, name = "n")
  cohort_means <- effects %>%
    group_by(cohort) %>%
    summarise(mean = mean(effect_years), .groups = "drop")
  tibble(
    cohort = if (panel == "COVID-19") "COVID_pooled" else "PsychAD_AD",
    cohort_label = if (panel == "COVID-19") "COVID-19 pooled" else "PsychAD Alzheimer disease",
    disease = if (panel == "COVID-19") "COVID" else "AD",
    cell_type = label,
    method = method,
    panel = panel,
    panel_title = if (panel == "COVID-19") "B  COVID-19" else "A  AD",
    estimate_years = mean(effects$effect_years),
    ci_low_years = ci[[1]],
    ci_high_years = ci[[2]],
    n_effects = nrow(effects),
    n_disease_donors = nrow(effects),
    n_reference_donors = NA_integer_,
    p_two_sided_t = NA_real_,
    pFDR = NA_real_,
    analysis_method = "pooled donor-level local age-match mean",
    reference_label = if (panel == "COVID-19") "COVID control" else "PsychAD calibration",
    n_cohorts = n_distinct(effects$cohort),
    se_years = NA_real_,
    tau2_years2 = NA_real_,
    Q = NA_real_,
    I2_percent = NA_real_,
    cohorts_used = paste(sort(unique(effects$cohort)), collapse = ";"),
    cohort_estimates = paste(
      sprintf("%s:%0.6f", cohort_means$cohort, cohort_means$mean), collapse = ";"
    ),
    cohort_se = NA_character_,
    cohort_cis = NA_character_,
    n_effects_by_cohort = paste(
      sprintf("%s:%d", cohort_counts$cohort, cohort_counts$n), collapse = ";"
    ),
    panel_color = panel_color
  )
}

# Direct AD rows. Global is explicitly first and separated from the cell-type
# rows in the panel.
ad <- age %>%
  filter(cohort == "PsychAD_AD", cell_type %in% ad_cell_type_order) %>%
  mutate(
    panel = "AD",
    panel_title = "A  AD",
    n_cohorts = 1L,
    se_years = mapply(bootstrap_se, ci_low_years, ci_high_years),
    tau2_years2 = 0,
    Q = 0,
    I2_percent = 0,
    cohorts_used = cohort,
    cohort_estimates = sprintf("%s:%0.6f", cohort, estimate_years),
    cohort_se = sprintf("%s:%0.6f", cohort, se_years),
    cohort_cis = sprintf("%s:[%0.6f,%0.6f]", cohort, ci_low_years, ci_high_years),
    n_effects_by_cohort = sprintf("%s:%d", cohort, n_effects),
    panel_color = ad_color,
    method = "scCody"
  )

covid_cohorts <- c("COVID_Ren", "COVID_COMBAT", "COVID_Stephenson")

# Pool scCody donor effects directly across COVID cohorts. Platelet uses Ren and
# Stephenson only, because COMBAT has too few Platelet control donors.
scody_donors <- readr::read_tsv(donor_effect_path, show_col_types = FALSE) %>%
  transmute(
    cohort = as.character(cohort),
    cell_type = as.character(population),
    disease_donor = as.character(disease_donor),
    effect_years = as.numeric(local_age_match_effect)
  ) %>%
  filter(cohort %in% covid_cohorts, cell_type %in% covid_cell_type_order)

covid_scody <- bind_rows(lapply(seq_along(covid_cell_type_order), function(i) {
  label <- covid_cell_type_order[[i]]
  summarize_pooled_donor_effects(
    scody_donors %>% filter(cell_type == label),
    label = label,
    method = "scCody",
    panel = "COVID-19",
    panel_color = covid_color,
    seed = 20260918L + i
  )
}))

# Frozen ElasticNet donor predictions are converted to the same one-effect-per
# disease-donor age-window contrast before pooling. COMBAT Platelet is removed
# because its two healthy donors do not provide defensible control support.
elasticnet_donors <- readr::read_tsv(elasticnet_donor_path, show_col_types = FALSE) %>%
  filter(
    cohort %in% covid_cohorts,
    !(cohort == "COVID_COMBAT" & cell_type == "Platelet"),
    (cell_type == "ALL" & scope == "whole_population") |
      (cell_type != "ALL" & scope == "percelltype")
  )

elasticnet_effects <- bind_rows(lapply(seq_along(covid_cell_type_order), function(i) {
  label <- covid_cell_type_order[[i]]
  raw_label <- if (label == "Global") "ALL" else label
  bind_rows(lapply(covid_cohorts, function(cohort_name) {
    raw <- elasticnet_donors %>%
      filter(cohort == cohort_name, cell_type == raw_label)
    local_donor_effects(raw) %>% mutate(cohort = cohort_name)
  })) %>% mutate(cell_type = label)
}))

covid_elasticnet <- bind_rows(lapply(seq_along(covid_cell_type_order), function(i) {
  label <- covid_cell_type_order[[i]]
  summarize_pooled_donor_effects(
    elasticnet_effects %>% filter(cell_type == label),
    label = label,
    method = "ElasticNet",
    panel = "COVID-19",
    panel_color = covid_color,
    seed = 20300918L + i
  )
}))

# The saved PsychAD ElasticNet forest data already contains the donor-level
# estimate and 1.96-standard-error interval for each PsychAD population.
ad_elasticnet <- readr::read_csv(ad_elasticnet_path, show_col_types = FALSE) %>%
  filter(panel == "elasticnet_effect") %>%
  transmute(
    cohort = "PsychAD_AD",
    cohort_label = "PsychAD Alzheimer disease",
    disease = "AD",
    cell_type = if_else(population == "Global population", "Global", as.character(population)),
    method = "ElasticNet",
    panel = "AD",
    panel_title = "A  AD",
    estimate_years = as.numeric(mean),
    ci_low_years = as.numeric(lo),
    ci_high_years = as.numeric(hi),
    n_effects = as.integer(n),
    n_disease_donors = as.integer(n),
    n_reference_donors = NA_integer_,
    p_two_sided_t = NA_real_,
    pFDR = NA_real_,
    analysis_method = "PsychAD ElasticNet local age-match estimate",
    reference_label = "PsychAD calibration",
    n_cohorts = 1L,
    se_years = mapply(bootstrap_se, as.numeric(lo), as.numeric(hi)),
    tau2_years2 = NA_real_,
    Q = NA_real_,
    I2_percent = NA_real_,
    cohorts_used = "PsychAD_AD",
    cohort_estimates = sprintf("PsychAD_AD:%0.6f", as.numeric(mean)),
    cohort_se = NA_character_,
    cohort_cis = sprintf("PsychAD_AD:[%0.6f,%0.6f]", as.numeric(lo), as.numeric(hi)),
    n_effects_by_cohort = sprintf("PsychAD_AD:%d", as.integer(n)),
    panel_color = ad_color
  ) %>%
  filter(cell_type %in% ad_cell_type_order)

if (!nrow(ad)) stop("No AD scCody rows available for the plotted cell types.")
if (!nrow(covid_scody) || !nrow(covid_elasticnet)) stop("No COVID donor-level rows available.")

format_panel <- function(frame, order) {
  present <- order[order %in% unique(frame$cell_type)]
  frame %>%
    mutate(
      cell_type = factor(cell_type, levels = present),
      row_id = length(present) - match(as.character(cell_type), present) + 1L,
      global_separator = length(present) - 0.5
    ) %>%
    arrange(row_id)
}

ad_plot <- format_panel(bind_rows(ad, ad_elasticnet), ad_cell_type_order)
covid_plot <- format_panel(bind_rows(covid_scody, covid_elasticnet), covid_cell_type_order)
plot_data <- bind_rows(ad_plot, covid_plot) %>%
  mutate(
    panel = as.character(panel),
    cell_type = as.character(cell_type),
    panel_color = as.character(panel_color),
    method = factor(method, levels = c("scCody", "ElasticNet")),
    plot_y = row_id + if_else(method == "ElasticNet", 0.12, -0.12)
  )

# Export the exact direct and pooled rows behind the figure.
write_csv(plot_data, file.path(outdir, "age_acceleration_5y_forest_data.csv"), na = "")

x_limits <- c(
  floor(min(plot_data$ci_low_years, na.rm = TRUE) / 5) * 5,
  ceiling(max(plot_data$ci_high_years, na.rm = TRUE) / 5) * 5
)
x_breaks <- seq(x_limits[[1]], x_limits[[2]], by = 5)

make_panel <- function(frame, title) {
  n_rows <- max(frame$row_id)
  labels <- unique(frame$cell_type[order(frame$row_id)])
  ggplot(frame, aes(x = estimate_years, y = plot_y)) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.42, color = "grey55") +
    geom_hline(yintercept = n_rows - 0.5, linewidth = 0.35, color = "grey72") +
    geom_errorbar(
      aes(xmin = ci_low_years, xmax = ci_high_years, color = panel_color),
      orientation = "y", width = 0.16, linewidth = 0.52, alpha = 0.72,
      show.legend = FALSE
    ) +
    geom_point(aes(color = panel_color, shape = method), fill = "white", size = 2.9, stroke = 0.85) +
    scale_color_identity() +
    scale_shape_manual(
      values = c(scCody = 16, ElasticNet = 21),
      name = "Method",
      labels = c(scCody = "scCody", ElasticNet = "ElasticNet")
    ) +
    scale_y_continuous(
      breaks = seq_len(n_rows),
      labels = labels,
      limits = c(0.5, n_rows + 0.5),
      expand = c(0, 0)
    ) +
    scale_x_continuous(
      name = "Age shift (years)",
      limits = x_limits,
      breaks = x_breaks,
      expand = c(0, 0)
    ) +
    labs(title = title) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
      axis.title.y = element_blank(),
      axis.text.y = element_text(size = 10),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_blank(),
      panel.grid.major.x = element_line(color = "grey88", linewidth = 0.3),
      panel.border = element_rect(color = "grey70", fill = NA, linewidth = 0.4),
      plot.margin = margin(7, 8, 7, 7)
    )
}

p_ad <- make_panel(plot_data %>% filter(panel == "AD"), "A  AD")
p_covid <- make_panel(plot_data %>% filter(panel == "COVID-19"), "B  COVID-19")
p <- (p_ad + p_covid + plot_layout(widths = c(1, 1), guides = "collect")) &
  theme(legend.position = "bottom")

plot_stem <- file.path(panel_dir, "Figure4A.age_acceleration_5y_forest")
ggsave(paste0(plot_stem, ".png"), p, width = 14.0, height = 7.3, dpi = 320)
ggsave(paste0(plot_stem, ".pdf"), p, width = 14.0, height = 7.3, useDingbats = FALSE)
ggsave(paste0(plot_stem, ".svg"), p, width = 14.0, height = 7.3)

fmt <- function(x, digits = 3) {
  ifelse(is.na(x), "NA", formatC(x, format = "f", digits = digits))
}
md <- c(
  "# 5-year local age-match acceleration forest plot",
  "",
  "Generated by `plot_age_acceleration_5y_forest.R`.",
  "",
  "- Panel A: direct PsychAD/AD local age-match estimates from scCody and the saved PsychAD ElasticNet donor-level analysis.",
  "- Panel B: one COVID estimate per cell type, pooled by concatenating donor-level effects across available cohorts for each method.",
  "- Each disease donor contributes one effect: disease prediction minus the mean healthy-control prediction within ±5 years; controls may be reused across disease donors.",
  "- Confidence intervals are donor-level bootstrap 95% intervals; no cell-level replication is introduced.",
  "- scCody is shown as a filled circle and ElasticNet as an open circle; color identifies disease panel.",
  "- Global is shown first and separated from the cell-type rows; rows use the established fixed cell-type order and are not sorted by effect.",
  paste0("- Shared x-axis limits: [", x_limits[[1]], ", ", x_limits[[2]], "] years."),
  "- Plasma_Cell is not pooled because it is available in only COMBAT; Platelet is pooled from Ren and Stephenson.",
  "- All N values are donor-level effects; no cell-level replication is introduced.",
  "",
  "## AD estimates by method",
  "",
  "| Cell type | Method | Acceleration (years) | 95% CI | n effects |",
  "|---|---|---:|---|---:|"
)
ad_md <- ad_plot %>% arrange(row_id, method)
for (i in seq_len(nrow(ad_md))) {
  r <- ad_md[i, ]
  md <- c(md, paste0(
    "| ", r$cell_type, " | ", r$method, " | ", fmt(r$estimate_years), " | [",
    fmt(r$ci_low_years), ", ", fmt(r$ci_high_years), "] | ", r$n_effects, " |"
  ))
}
md <- c(
  md,
  "",
  "## COVID donor-pooled estimates by method",
  "",
  "| Cell type | Method | Acceleration (years) | 95% CI | n effects | Cohorts | Cohort means |",
  "|---|---|---:|---|---:|---:|---|"
)
covid_md <- covid_plot %>% arrange(row_id, method)
for (i in seq_len(nrow(covid_md))) {
  r <- covid_md[i, ]
  md <- c(md, paste0(
    "| ", r$cell_type, " | ", r$method, " | ", fmt(r$estimate_years), " | [",
    fmt(r$ci_low_years), ", ", fmt(r$ci_high_years), "] | ", r$n_effects, " | ",
    r$n_cohorts, " | `", r$cohort_estimates, "` |"
  ))
}
writeLines(md, file.path(outdir, "age_acceleration_5y_forest.md"))

cat("Wrote", nrow(ad_plot), "AD method rows and", nrow(covid_plot), "COVID method rows to", outdir, "\n")
