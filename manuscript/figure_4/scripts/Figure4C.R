#!/usr/bin/env Rscript
# Figure 4c: slope of the
# signed aligned and orthogonal RMS against exact age gap in healthy PsychAD holdout
# donor pairs, per cell group. Run from the repository root.

# Uses healthy PsychAD donor pairs in which both source and target donors are in
# the healthy holdout manifest (epoch-5 models).

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
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
data_dir <- outdir

summary_path <- file.path(data_dir, "ad_epoch5_exact_gap_relationship_holdout_mapped_summary.csv")
if (!file.exists(summary_path)) stop("Missing slope summary: ", summary_path)

group_order <- c("Global", "Astro", "EN", "Endo", "IN", "Immune", "Mural", "OPC", "Oligo")
group_colors <- c(
  Global = "#4D4D4D", Astro = "#1B9E77", EN = "#D95F02", Endo = "#7570B3",
  IN = "#E7298A", Immune = "#66A61E", Mural = "#E6AB02", OPC = "#A6761D",
  Oligo = "#1F78B4"
)

slopes_raw <- read_csv(summary_path, show_col_types = FALSE)
# Read holdout projection-RMS pairs for the signed aligned component,
# using the recorded exact-gap and group filters.
signed_pair_files <- file.path(outdir, "ad_holdout_real_healthy_aging_pairs.tsv.gz")
if (!file.exists(signed_pair_files)) stop("Missing signed-aligned-RMS pair table: ", signed_pair_files)
signed_pairs <- bind_rows(lapply(signed_pair_files, function(path) {
  read_tsv(path, show_col_types = FALSE)
})) %>%
  mutate(
    group = recode(group, `Global population` = "Global"),
    age_gap = as.numeric(age_gap),
    inner_product = as.numeric(inner_product),
    aging_energy = as.numeric(aging_energy),
    signed_aligned_rms = inner_product / sqrt(aging_energy)
  ) %>%
  filter(
    group %in% group_order,
    between(age_gap, 1, 15),
    is.finite(inner_product), is.finite(aging_energy), aging_energy > 1e-12,
    is.finite(signed_aligned_rms)
  )

signed_slopes_raw <- signed_pairs %>%
  group_by(group) %>%
  group_modify(~ {
    fit <- lm(signed_aligned_rms ~ age_gap, data = .x)
    ci <- confint(fit, "age_gap", level = 0.95)
    tibble(
      metric = "signed_aligned_rms",
      metric_label = "signed aligned RMS",
      n_pairs = nrow(.x),
      pearson_r = cor(.x$age_gap, .x$signed_aligned_rms),
      ols_slope_per_year = unname(coef(fit)[["age_gap"]]),
      ols_slope_ci95_low = ci[1],
      ols_slope_ci95_high = ci[2]
    )
  }) %>%
  ungroup()

signed_slope_input <- bind_rows(
  signed_slopes_raw,
  slopes_raw %>% filter(metric == "orthogonal_rms")
)

make_signed_slope_plot <- function(input, excluded = character()) {
  slope_group_order <- input %>%
    filter(metric == "signed_aligned_rms", !group %in% excluded) %>%
    arrange(desc(ols_slope_per_year)) %>%
    pull(group)
  slopes <- input %>%
    filter(metric %in% c("signed_aligned_rms", "orthogonal_rms"), !group %in% excluded) %>%
    mutate(
      group = factor(group, levels = rev(slope_group_order)),
      metric = factor(
        metric,
        levels = c("signed_aligned_rms", "orthogonal_rms"),
        labels = c("Signed aligned RMS", "Orthogonal RMS")
      ),
      y_base = as.numeric(group),
      y = y_base + if_else(metric == "Signed aligned RMS", 0.16, -0.16)
    )
  x_data_min <- min(slopes$ols_slope_ci95_low, na.rm = TRUE)
  x_data_max <- max(slopes$ols_slope_ci95_high, na.rm = TRUE)
  x_span <- x_data_max - x_data_min
  x_left <- x_data_min - 0.05 * x_span
  x_label <- x_data_max + 0.04 * x_span
  x_right <- x_label + 0.13 * x_span
  slopes <- slopes %>% mutate(x_label = x_label, r_label = sprintf("r=%.3f", pearson_r))

  ggplot(slopes) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.45, color = "grey45") +
    geom_segment(
      aes(x = ols_slope_ci95_low, xend = ols_slope_ci95_high,
          y = y, yend = y, color = metric),
      linewidth = 0.75, lineend = "round"
    ) +
    geom_point(aes(x = ols_slope_per_year, y = y, color = metric), size = 3.4) +
    geom_text(
      aes(x = x_label, y = y, label = r_label, color = metric),
      hjust = 0, size = 3.1, show.legend = FALSE
    ) +
    scale_y_continuous(
      breaks = seq_along(rev(slope_group_order)), labels = rev(slope_group_order),
      limits = c(0.35, length(slope_group_order) + 0.65), expand = c(0, 0)
    ) +
    scale_x_continuous(
      limits = c(x_left, x_right),
      breaks = pretty(c(x_left, x_data_max), n = 5),
      expand = c(0, 0)
    ) +
    scale_color_manual(
      values = c("Signed aligned RMS" = "#1D9E75", "Orthogonal RMS" = "#456294"),
      name = NULL
    ) +
    labs(
      title = "Aging slopes",
      x = "Slope per year",
      y = NULL
    ) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "bottom",
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "grey70", fill = NA, linewidth = 0.4),
      plot.title = element_text(face = "bold", size = 14)
    )
}

p_signed_slope <- make_signed_slope_plot(signed_slope_input)

signed_slope_stem <- file.path(panel_dir, "Figure4C.gap_slope_dotplot")
ggsave(paste0(signed_slope_stem, ".svg"), p_signed_slope, width = 8.7, height = 6.6)
ggsave(paste0(signed_slope_stem, ".png"), p_signed_slope, width = 8.7, height = 6.6, dpi = 320)
ggsave(paste0(signed_slope_stem, ".pdf"), p_signed_slope, width = 8.7, height = 6.6, useDingbats = FALSE)
write_csv(signed_slope_input, file.path(outdir, "Figure4C.gap_slope_dotplot_data.csv"))
