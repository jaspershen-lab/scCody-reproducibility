#!/usr/bin/env Rscript
# Figure 4d: disease vs +10 y healthy aging, log2 signed aligned RMS D/H10 against
# total orthogonal RMS change. Run from the repository root:
#   Rscript manuscript/figure_4/scripts/Figure4D.R
# Options: --metric aging_rms, --keep-oligo, --pair-replay-dir <dir>, --output-stem <name>.
#
# Each plotted point is one cell type within one disease. Signed aligned RMS is
# averaged per target donor from the per-pair replay table (PsychAD epoch-5 AD
# fields; AIDA COVID fields). COVID
# is pooled by first calculating donor means within each cohort, then giving
# each included cohort equal weight; Platelet is retained as a two-cohort point
# and flagged in the exported data as inconsistent across cohorts.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggrepel)
  library(jsonlite)
})

`%||%` <- function(x, y) if (is.null(x)) y else x

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

trailing_args <- commandArgs(trailingOnly = TRUE)
metric_arg <- "signed_aligned_rms"
metric_pos <- which(trailing_args == "--metric")
if (length(metric_pos)) {
  if (metric_pos[[1]] >= length(trailing_args)) stop("--metric requires a value")
  metric_arg <- trailing_args[[metric_pos[[1]] + 1L]]
}
if (!metric_arg %in% c("aging_rms", "signed_aligned_rms")) {
  stop("--metric must be aging_rms or signed_aligned_rms")
}
pair_replay_dir_arg <- outdir
pair_replay_pos <- which(trailing_args == "--pair-replay-dir")
if (length(pair_replay_pos)) {
  if (pair_replay_pos[[1]] >= length(trailing_args)) stop("--pair-replay-dir requires a value")
  pair_replay_dir_arg <- trailing_args[[pair_replay_pos[[1]] + 1L]]
}
output_stem_arg <- "Figure4D.aging_alignment_scatter"
output_stem_pos <- which(trailing_args == "--output-stem")
if (length(output_stem_pos)) {
  if (output_stem_pos[[1]] >= length(trailing_args)) stop("--output-stem requires a value")
  output_stem_arg <- trailing_args[[output_stem_pos[[1]] + 1L]]
}
drop_oligo <- !("--keep-oligo" %in% trailing_args)
squeeze_horizontal <- "--squeeze-horizontal" %in% trailing_args

plot_excluded_cell_types <- c("IN", "Mural")
if (drop_oligo) plot_excluded_cell_types <- c(plot_excluded_cell_types, "Oligo")

# H10 is the healthy-aging reference: plus_10y, pair_set=window.
bootstrap_repeats <- 5000L
random_seed <- 20260917L
ad_holdout_manifest <- file.path(repo, "configs/donors/psychad_healthy_holdout.json")
ad_holdout_donors <- as.character(fromJSON(ad_holdout_manifest)$holdout_donors)

mean_or_na <- function(x) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]
  if (!length(x)) NA_real_ else mean(x)
}

read_pair_replay_targets <- function(path) {
  input <- file.path(path, "signed_aligned_rms_pair_replay_all.tsv.gz")
  if (!file.exists(input)) stop("Missing signed-aligned RMS replay table: ", input)
  x <- read_tsv(input, show_col_types = FALSE) %>%
    filter(arm %in% c("disease", "healthy")) %>%
    mutate(
      dataset = as.character(dataset),
      cohort = if_else(
        dataset == "AD", "AD", paste0("COVID_", as.character(cohort))
      ),
      cell_type = as.character(group_normalized),
      target_donor = as.character(target_donor_replay),
      aging_energy_replay = as.numeric(aging_energy_replay),
      inner_product_replay = as.numeric(inner_product_replay),
      alpha_replay = as.numeric(alpha_replay),
      orthogonal_energy = as.numeric(orthogonal_energy),
      disease_energy_replay = orthogonal_energy + alpha_replay^2 * aging_energy_replay,
      signed_parallel_fraction_replay = if_else(
        is.finite(inner_product_replay) & aging_energy_replay > 1e-12 &
          is.finite(disease_energy_replay) & disease_energy_replay > 1e-12,
        sign(inner_product_replay) * inner_product_replay^2 /
          (aging_energy_replay * disease_energy_replay),
        NA_real_
      ),
      target_age = as.numeric(target_age)
    ) %>%
    group_by(dataset, cohort, arm, cell_type, target_donor) %>%
    summarise(
      target_age = mean_or_na(target_age),
      target_sex = NA_character_,
      signed_aligned_rms = mean_or_na(signed_aligned_rms),
      aging_rms = {
        z <- aging_energy_replay[is.finite(aging_energy_replay)]
        if (!length(z)) NA_real_ else sqrt(mean(z))
      },
      parallel_fraction = mean_or_na(signed_parallel_fraction_replay),
      orthogonal_rms = {
        z <- orthogonal_energy[is.finite(orthogonal_energy)]
        if (!length(z)) NA_real_ else sqrt(mean(z))
      },
      .groups = "drop"
    )
}

replay_targets <- read_pair_replay_targets(pair_replay_dir_arg)
all_disease <- replay_targets %>% filter(arm == "disease")
all_healthy <- replay_targets %>% filter(arm == "healthy")

# The PsychAD healthy holdout manifest applies to healthy H10 target donors.  The AD
# disease arm is retained in full because this manifest contains normal
# donors, not disease donors.
all_healthy <- all_healthy %>%
  filter(cohort != "AD" | target_donor %in% ad_holdout_donors)

# COVID populations in the cross-cohort design:
# four cell types in all three cohorts, and Platelet in two cohorts.
covid_required <- c("Global", "B", "Myeloid", "NK", "T")
covid_optional_two_cohort <- "Platelet"
covid_ids <- c("COVID_Ren", "COVID_COMBAT", "COVID_Stephenson")

safe_quantile <- function(x, probs = c(0.025, 0.975)) {
  x <- x[is.finite(x)]
  if (!length(x)) return(c(NA_real_, NA_real_))
  as.numeric(quantile(x, probs = probs, names = FALSE, type = 6))
}

balanced_summary <- function(frame, value_col, source_cohorts, repeats = bootstrap_repeats) {
  # Each cohort contributes one cohort mean. Bootstrap resampling is done
  # within cohort before taking the equal-weight mean across cohorts.
  values <- lapply(source_cohorts, function(cohort_id) {
    z <- as.numeric(frame[[value_col]][frame$cohort == cohort_id])
    z[is.finite(z)]
  })
  names(values) <- source_cohorts
  values <- values[lengths(values) > 0L]
  if (!length(values)) {
    return(list(mean = NA_real_, ci_low = NA_real_, ci_high = NA_real_, n = 0L,
                n_by_cohort = "", boot = numeric()))
  }
  cohort_means <- vapply(values, mean, numeric(1))
  point_mean <- mean(cohort_means)
  if (repeats <= 0L) {
    boot <- point_mean
  } else {
    boot <- replicate(
      repeats,
      mean(vapply(values, function(z) mean(sample(z, length(z), replace = TRUE)), numeric(1)))
    )
  }
  ci <- safe_quantile(boot)
  list(
    mean = point_mean,
    ci_low = ci[[1]],
    ci_high = ci[[2]],
    n = sum(lengths(values)),
    n_by_cohort = paste0(names(values), ":", lengths(values), collapse = ";"),
    boot = boot
  )
}

pooled_point <- function(disease, healthy, disease_name, cell_type, source_cohorts,
                         inconsistent = FALSE) {
  d <- disease %>% filter(cell_type == !!cell_type, cohort %in% source_cohorts)
  h <- healthy %>% filter(cell_type == !!cell_type, cohort %in% source_cohorts)
  if (!nrow(d) || !nrow(h)) return(NULL)

  d_par <- balanced_summary(d, "parallel_fraction", source_cohorts)
  h_par <- balanced_summary(h, "parallel_fraction", source_cohorts)
  d_aging <- balanced_summary(d, "aging_rms", source_cohorts)
  h_aging <- balanced_summary(h, "aging_rms", source_cohorts)
  d_signed_aligned <- balanced_summary(d, "signed_aligned_rms", source_cohorts)
  h_signed_aligned <- balanced_summary(h, "signed_aligned_rms", source_cohorts)
  d_orth <- balanced_summary(d, "orthogonal_rms", source_cohorts)
  h_orth <- balanced_summary(h, "orthogonal_rms", source_cohorts)

  d_metric <- if (metric_arg == "signed_aligned_rms") d_signed_aligned else d_aging
  h_metric <- if (metric_arg == "signed_aligned_rms") h_signed_aligned else h_aging

  y_value <- if (d_metric$mean > 0 && h_metric$mean > 0) {
    log2(d_metric$mean / h_metric$mean)
  } else {
    NA_real_
  }
  x_value <- if (d_orth$mean > 0 && h_orth$mean > 0) {
    100 * (d_orth$mean / h_orth$mean - 1)
  } else {
    NA_real_
  }

  y_boot <- numeric()
  y_boot_domain <- numeric()
  y_boot_total <- 0L
  y_boot_invalid <- 0L
  if (length(d_metric$boot) && length(h_metric$boot)) {
    y_boot_total <- min(length(d_metric$boot), length(h_metric$boot))
    d_boot <- d_metric$boot[seq_len(y_boot_total)]
    h_boot <- h_metric$boot[seq_len(y_boot_total)]
    ok <- d_boot > 0 & h_boot > 0 & is.finite(d_boot) & is.finite(h_boot)
    y_boot_invalid <- sum(!ok)
    y_boot <- log2(d_boot[ok] / h_boot[ok])
    # Preserve invalid log-domain draws as -Inf.  They should affect the
    # percentile rather than being silently removed.  A lower 2.5% bound is
    # finite whenever fewer than 2.5% of bootstrap draws are invalid.
    y_boot_domain <- rep(-Inf, y_boot_total)
    y_boot_domain[ok] <- log2(d_boot[ok] / h_boot[ok])
  }
  x_boot <- numeric()
  if (length(d_orth$boot) && length(h_orth$boot)) {
    ok <- d_orth$boot > 0 & h_orth$boot > 0 & is.finite(d_orth$boot) & is.finite(h_orth$boot)
    x_boot <- 100 * (d_orth$boot[ok] / h_orth$boot[ok] - 1)
  }
  # Invalid log-domain draws are retained as -Inf in the bootstrap quantile.
  # Only when the invalid mass reaches the lower 2.5% tail does the lower CI
  # become undefined and require the downward-arrow display.
  y_ci_domain <- if (is.finite(y_value) && length(y_boot_domain)) {
    as.numeric(quantile(y_boot_domain, probs = c(0.025, 0.975),
                        names = FALSE, type = 6))
  } else {
    c(NA_real_, NA_real_)
  }
  y_ci_defined <- is.finite(y_value) && y_boot_total > 0L &&
    y_boot_invalid < 0.025 * y_boot_total
  y_ci_valid <- if (is.finite(y_value) && length(y_boot)) safe_quantile(y_boot) else c(NA_real_, NA_real_)
  y_ci <- if (y_ci_defined) y_ci_domain else c(NA_real_, y_ci_domain[[2]])
  y_ci_status <- case_when(
    !is.finite(y_value) ~ "undefined: point outside log domain",
    y_boot_total == 0L ~ "undefined: no bootstrap replicates",
    y_boot_invalid >= 0.025 * y_boot_total ~ "undefined: >2.5% bootstrap draws outside non-positive domain",
    y_boot_invalid > 0L ~ "defined: invalid draws retained as -Inf in percentile bootstrap",
    TRUE ~ "defined: percentile donor bootstrap"
  )
  x_ci <- if (!is.na(x_value)) safe_quantile(x_boot) else c(NA_real_, NA_real_)

  tibble(
    disease = disease_name,
    cell_type = cell_type,
    par_D = 100 * d_par$mean,
    par_H = 100 * h_par$mean,
    aging_D = d_aging$mean,
    aging_H = h_aging$mean,
    aging_D_ci_low = d_aging$ci_low,
    aging_D_ci_high = d_aging$ci_high,
    aging_H_ci_low = h_aging$ci_low,
    aging_H_ci_high = h_aging$ci_high,
    signed_aligned_D = d_signed_aligned$mean,
    signed_aligned_H = h_signed_aligned$mean,
    metric_D_mean = d_metric$mean,
    metric_H_mean = h_metric$mean,
    signed_aligned_D_ci_low = d_signed_aligned$ci_low,
    signed_aligned_D_ci_high = d_signed_aligned$ci_high,
    signed_aligned_H_ci_low = h_signed_aligned$ci_low,
    signed_aligned_H_ci_high = h_signed_aligned$ci_high,
    orth_D = d_orth$mean,
    orth_H = h_orth$mean,
    par_D_ci_low = 100 * d_par$ci_low,
    par_D_ci_high = 100 * d_par$ci_high,
    par_H_ci_low = 100 * h_par$ci_low,
    par_H_ci_high = 100 * h_par$ci_high,
    orth_D_ci_low = d_orth$ci_low,
    orth_D_ci_high = d_orth$ci_high,
    orth_H_ci_low = h_orth$ci_low,
    orth_H_ci_high = h_orth$ci_high,
    y = y_value,
    y_ci_low = y_ci[[1]],
    y_ci_high = y_ci[[2]],
    y_ci_valid_low = y_ci_valid[[1]],
    y_ci_valid_high = y_ci_valid[[2]],
    x = x_value,
    x_ci_low = x_ci[[1]],
    x_ci_high = x_ci[[2]],
    n_D = d_metric$n,
    n_H = h_metric$n,
    n_D_orthogonal = d_orth$n,
    n_H_orthogonal = h_orth$n,
    n_D_by_cohort = d_par$n_by_cohort,
    n_H_by_cohort = h_par$n_by_cohort,
    n_boot_y_valid = length(y_boot),
    n_boot_y_total = y_boot_total,
    n_boot_y_invalid = y_boot_invalid,
    y_boot_invalid_fraction = if (y_boot_total) y_boot_invalid / y_boot_total else NA_real_,
    y_ci_defined = y_ci_defined,
    y_ci_status = y_ci_status,
    n_boot_x_valid = length(x_boot),
    cohort_count = length(source_cohorts),
    cohorts_used = paste(source_cohorts, collapse = ";"),
    inconsistent_across_cohorts = inconsistent,
    nonpositive_y = !(is.finite(d_metric$mean) && d_metric$mean > 0 &&
                       is.finite(h_metric$mean) && h_metric$mean > 0),
    nonpositive_parallel = !(is.finite(d_par$mean) && d_par$mean > 0 &&
                               is.finite(h_par$mean) && h_par$mean > 0),
    nonpositive_reason = case_when(
      d_metric$mean <= 0 ~ paste0(metric_arg, " D <= 0"),
      h_metric$mean <= 0 ~ paste0(metric_arg, " H <= 0"),
      TRUE ~ ""
    )
  )
}

# Make all donor bootstrap results reproducible.
set.seed(random_seed)

# AD is a single cohort; Global is shown alongside the cell-type rows.
ad_groups <- intersect(
  unique(all_disease$cell_type[all_disease$cohort == "AD"]),
  unique(all_healthy$cell_type[all_healthy$cohort == "AD"])
)
ad_points <- bind_rows(lapply(sort(ad_groups), function(cell_type) {
  pooled_point(all_disease, all_healthy, "AD", cell_type, "AD", inconsistent = FALSE)
}))

covid_counts <- all_disease %>%
  filter(cohort %in% covid_ids) %>%
  distinct(cohort, cell_type) %>%
  count(cell_type, name = "n_cohorts_disease")
covid_h_counts <- all_healthy %>%
  filter(cohort %in% covid_ids) %>%
  distinct(cohort, cell_type) %>%
  count(cell_type, name = "n_cohorts_healthy")
covid_groups <- bind_rows(
  tibble(cell_type = covid_required, needed = 3L),
  tibble(cell_type = covid_optional_two_cohort, needed = 2L)
) %>%
  left_join(covid_counts, by = "cell_type") %>%
  left_join(covid_h_counts, by = "cell_type") %>%
  mutate(
    n_cohorts_disease = replace_na(n_cohorts_disease, 0L),
    n_cohorts_healthy = replace_na(n_cohorts_healthy, 0L)
  ) %>%
  filter(n_cohorts_disease >= needed, n_cohorts_healthy >= needed) %>%
  pull(cell_type)

covid_points <- bind_rows(lapply(sort(covid_groups), function(cell_type) {
  source_cohorts <- covid_ids[
    covid_ids %in% unique(all_disease$cohort[all_disease$cell_type == cell_type]) &
      covid_ids %in% unique(all_healthy$cohort[all_healthy$cell_type == cell_type])
  ]
  pooled_point(
    all_disease, all_healthy, "COVID", cell_type, source_cohorts,
    inconsistent = cell_type == "Platelet"
  )
}))

points <- bind_rows(ad_points, covid_points) %>%
  filter(!cell_type %in% plot_excluded_cell_types) %>%
  mutate(
    plot_label = if_else(
      nonpositive_y,
      paste0(cell_type, " (D = ", sprintf("%.3f", metric_D_mean), ")"),
      cell_type
    ),
    shape_key = case_when(
      nonpositive_y ~ "nonpositive",
      TRUE ~ "regular"
    )
  )

# The x coordinate is the total orthogonal RMS from the orthogonal-variance
# decomposition. The RMS is computed after summing mean-shift, between-donor, and within-donor
# energies, then compared as D/H10.  It is deliberately read from the
# consolidated table so the plot and decomposition report use the same
# estimand.
orth_variance_path <- file.path(
  outdir, "orthogonal_residual_variance",
  "orthogonal_residual_variance_D_H10_all.csv"
)
if (!file.exists(orth_variance_path)) {
  stop("Missing orthogonal residual variance table: ", orth_variance_path)
}
orth_variance <- read_csv(orth_variance_path, show_col_types = FALSE) %>%
  filter(dataset %in% c("AD", "COVID_cohort_balanced")) %>%
  transmute(
    disease = if_else(dataset == "AD", "AD", "COVID"),
    cell_type = as.character(group),
    orth_energy_D = D_orthogonal_energy,
    orth_energy_H10 = H10_orthogonal_energy,
    orth_rms_D = D_orthogonal_rms,
    orth_rms_H10 = H10_orthogonal_rms,
    orth_rms_change = D_H10_percent_change_orthogonal_rms,
    orth_rms_ci_low = 100 * (ratio_orthogonal_rms_ci_low - 1),
    orth_rms_ci_high = 100 * (ratio_orthogonal_rms_ci_high - 1)
  )
if (anyDuplicated(orth_variance[c("disease", "cell_type")])) {
  stop("Duplicate disease/cell-type rows in orthogonal variance table")
}
points <- points %>%
  left_join(orth_variance, by = c("disease", "cell_type"))
missing_orth <- points %>% filter(!is.finite(orth_rms_change))
if (nrow(missing_orth)) {
  stop(
    "Missing total orthogonal variance rows for: ",
    paste(paste(missing_orth$disease, missing_orth$cell_type, sep = "/"), collapse = ", ")
  )
}
points <- points %>%
  mutate(
    x = orth_rms_change,
    x_ci_low = orth_rms_ci_low,
    x_ci_high = orth_rms_ci_high
  )

plot_stem <- if (!is.null(output_stem_arg)) {
  file.path(panel_dir, output_stem_arg)
} else if (metric_arg == "signed_aligned_rms") {
  suffix <- if (drop_oligo) "_no_oligo" else ""
  file.path(panel_dir, paste0("ad_covid_aging_alignment_scatter_signed_aligned_rms", suffix))
} else {
  file.path(panel_dir, "ad_covid_aging_alignment_scatter")
}

# Put invalid signed ratios at a visible lower plotting floor rather than
# silently dropping them.  The underlying y remains NA in the data table.
observed_y <- c(points$y, points$y_ci_low, points$y_ci_high)
observed_y <- observed_y[is.finite(observed_y)]
has_floor <- any(points$nonpositive_y)
y_axis_cut <- -3.2
# Keep the triangle center above the panel boundary so its full marker is
# visible.  The arrow still terminates just below the marker.
y_floor <- if (has_floor) y_axis_cut + 0.18 else NA_real_
points <- points %>%
  mutate(
    y_plot = if_else(nonpositive_y, y_floor, y),
    y_plot_ci_low = if_else(nonpositive_y, NA_real_, y_ci_low),
    y_plot_ci_high = if_else(nonpositive_y, NA_real_, y_ci_high)
  )

write.csv(points, file.path(outdir, paste0(basename(plot_stem), "_data.csv")), row.names = FALSE, na = "")


plot_data <- points
positive_data <- plot_data %>% filter(!nonpositive_y)
nonpositive_data <- plot_data %>% filter(nonpositive_y)
y_limits_data <- c(plot_data$y_plot, plot_data$y_plot_ci_low, plot_data$y_plot_ci_high)
y_limits_data <- y_limits_data[is.finite(y_limits_data)]
y_span <- diff(range(y_limits_data))
y_padding <- max(0.10, 0.03 * y_span)
y_lower <- if (has_floor) y_axis_cut else min(y_limits_data) - y_padding
y_upper <- max(y_limits_data) + y_padding
truncated_y_data <- plot_data %>%
  filter(!y_ci_defined, is.finite(y_ci_high), n_boot_y_invalid > 0L)

p <- ggplot(plot_data, aes(x = x, y = y_plot)) +
  geom_segment(
    aes(x = x_ci_low, xend = x_ci_high, y = y_plot, yend = y_plot, color = disease),
    na.rm = TRUE, linewidth = 0.65, alpha = 0.55, show.legend = FALSE
  ) +
  geom_segment(
    aes(x = x, xend = x, y = y_plot_ci_low, yend = y_plot_ci_high, color = disease),
    na.rm = TRUE, linewidth = 0.65, alpha = 0.55, show.legend = FALSE
  ) +
  geom_segment(
    data = truncated_y_data,
    aes(x = x, xend = x, y = y_ci_high, yend = y_floor - 0.06, color = disease),
    na.rm = TRUE, linewidth = 0.65, alpha = 0.55,
    arrow = grid::arrow(length = grid::unit(0.12, "cm"), type = "closed"),
    show.legend = FALSE
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.35, color = "grey55") +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.35, color = "grey55") +
  geom_point(
    data = positive_data, aes(color = disease), shape = 16,
    size = 4.6
  ) +
  geom_point(
    data = nonpositive_data, aes(color = disease), shape = 25, fill = "white",
    size = 5.0, stroke = 1.2, show.legend = FALSE
  ) +
  geom_text_repel(
    data = positive_data,
    aes(label = plot_label, color = disease),
    seed = random_seed, size = 3.8, box.padding = 0.40, point.padding = 0.22,
    min.segment.length = Inf, segment.color = NA, force = 1.1,
    max.overlaps = Inf, show.legend = FALSE
  ) +
  geom_text(
    data = nonpositive_data,
    aes(label = plot_label, color = disease),
    hjust = 0, vjust = 0.5, nudge_x = 0.45, size = 3.8,
    show.legend = FALSE
  ) +
  scale_color_manual(values = c(AD = "#2166AC", COVID = "#D95F02"), name = "Disease") +
  scale_x_continuous(name = "Orthogonal displacement (% change vs H10)", expand = expansion(mult = 0.08)) +
  scale_y_continuous(
    name = expression("Aging-aligned displacement (log"[2] * " D/H10)"),
    expand = expansion(mult = 0)
  ) +
  coord_cartesian(ylim = c(y_lower, y_upper)) +
  labs(
    title = if (metric_arg == "signed_aligned_rms") {
      "Aging alignment"
    } else {
      "Aging alignment"
    }
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 15),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "grey88", linewidth = 0.3),
    panel.border = element_rect(color = "grey70", fill = NA, linewidth = 0.4)
  )

scatter_width <- if (squeeze_horizontal) 8.5 else 11
ggsave(paste0(plot_stem, ".png"), p, width = scatter_width, height = 7.7, dpi = 320)
ggsave(paste0(plot_stem, ".pdf"), p, width = scatter_width, height = 7.7, useDingbats = FALSE)
ggsave(paste0(plot_stem, ".svg"), p, width = scatter_width, height = 7.7)

cat("Wrote", nrow(points), "cell-type points to", outdir, "\n")
print(points %>% select(disease, cell_type, x, y_plot, par_D, par_H, orth_D, orth_H, n_D, n_H))
