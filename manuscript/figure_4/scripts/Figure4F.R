#!/usr/bin/env Rscript
# Figure 4f: change in shared, between- and within-donor orthogonal energy vs +10 y
# healthy aging, 100*(D_k - H10_k)/H10_total, 95% donor-bootstrap CI. Data from
# sccody_analyses.orthogonal_variance.panel_f. Run from the repository root.

# Figure 4 panel f: orthogonal displacement decomposition.
# The Python companion creates the donor-bootstrap data and performs the
# component-sum assertions; this script renders the ggplot figure.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(grid)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- if (length(script_arg)) sub("^--file=", "", script_arg[[1]]) else ""
repo <- if (nzchar(script_file)) {
  normalizePath(file.path(dirname(script_file), "../../.."), mustWork = TRUE)
} else {
  normalizePath(getwd(), mustWork = TRUE)
}
outdir <- file.path(repo, "manuscript/figure_4/results/orthogonal_residual_variance")
panel_dir <- file.path(repo, "manuscript/figure_4/figure_files/panels")
dir.create(panel_dir, recursive = TRUE, showWarnings = FALSE)
data_path <- file.path(outdir, "orthogonal_residual_variance_panel_f_data.csv")
if (!file.exists(data_path)) stop("Missing input: ", data_path)

data <- readr::read_csv(data_path, show_col_types = FALSE) %>%
  mutate(
    dataset = factor(dataset, levels = c("AD", "COVID")),
    component_label = factor(
      component_label,
      levels = c("Shared shift", "Between donors", "Within donors")
    ),
    color = if_else(dataset == "AD", "AD", "COVID-19"),
    estimate = as.numeric(estimate_percent),
    low = as.numeric(ci_low_percent),
    high = as.numeric(ci_high_percent)
  )

# This is the order used by the current Figure 4 age-acceleration panels.
ad_order <- c("Global", "OPC", "Astro", "EN", "Oligo", "Endo", "Immune")
covid_order <- c("Global", "B", "Myeloid", "NK", "Platelet", "T")
rows <- tibble::tibble(
  dataset = c(rep("AD", length(ad_order)), rep("COVID", length(covid_order))),
  cell_type = c(ad_order, covid_order),
  y = c(seq(length(ad_order) + length(covid_order), length(covid_order) + 1),
        seq(length(covid_order), 1))
)
data <- data %>% left_join(rows, by = c("dataset", "cell_type"))
if (any(is.na(data$y))) stop("Unmapped cell type in panel data.")
if (any(!is.finite(c(data$estimate, data$low, data$high)))) {
  stop("Panel data contains non-finite estimates or confidence limits.")
}

ad_mid <- max(rows$y[rows$dataset == "AD"]) + 0.25
covid_mid <- max(rows$y[rows$dataset == "COVID"]) + 0.25
boundary <- length(covid_order) + 0.5
shared_range <- range(
  c(data$low[data$component == "shared"], data$high[data$component == "shared"], 0),
  finite = TRUE
)
left_x <- shared_range[[1]] + 0.02 * diff(shared_range)
block_labels <- tibble::tibble(
  x = left_x,
  y = c(ad_mid, covid_mid),
  label = c("AD", "COVID-19"),
  component_label = factor(
    rep("Shared shift", 2),
    levels = levels(data$component_label)
  )
)

p <- ggplot(data, aes(y = y)) +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.35, color = "#666666") +
  geom_segment(
    aes(x = low, xend = high, yend = y, color = color),
    linewidth = 0.55, alpha = 0.78, lineend = "round"
  ) +
  geom_point(aes(x = estimate, color = color), size = 2.0, stroke = 0.2) +
  geom_hline(yintercept = boundary, linewidth = 0.35, color = "#9E9E9E") +
  geom_text(
    data = block_labels,
    aes(x = x, y = y, label = label),
    inherit.aes = FALSE, hjust = 0, fontface = "bold", size = 3.5
  ) +
  facet_grid(. ~ component_label, scales = "free_x") +
  scale_color_manual(values = c("AD" = "#2166AC", "COVID-19" = "#D95F02"), guide = "none") +
  scale_x_continuous(
    expand = expansion(mult = c(0.08, 0.08)),
    name = "Change in orthogonal energy (% of healthy total)"
  ) +
  scale_y_continuous(
    breaks = rows$y, labels = rows$cell_type,
    limits = c(0.5, max(rows$y) + 0.5), expand = c(0, 0), name = NULL
  ) +
  labs(title = "Orthogonal components") +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(size = 12, face = "bold", hjust = 0),
    axis.title.x = element_text(size = 10, margin = margin(t = 8)),
    axis.text.x = element_text(size = 8, color = "#333333"),
    axis.text.y = element_text(size = 9, color = "#333333"),
    strip.background = element_blank(),
    strip.text = element_text(size = 9, face = "bold"),
    panel.grid.major.x = element_line(color = "#E6E6E6", linewidth = 0.3),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.border = element_blank(),
    axis.line.x = element_line(color = "#333333", linewidth = 0.35),
    axis.line.y = element_line(color = "#333333", linewidth = 0.35),
    plot.margin = margin(8, 10, 10, 48),
    panel.spacing.x = unit(0.8, "lines")
  ) +
  coord_cartesian(clip = "off")

stem <- file.path(panel_dir, "Figure4F.orthogonal_displacement_decomposition")
ggsave(paste0(stem, ".svg"), p, width = 12.0, height = 6.0, units = "in", device = "svg")
ggsave(paste0(stem, ".pdf"), p, width = 12.0, height = 6.0, units = "in", device = cairo_pdf)
ggsave(paste0(stem, ".png"), p, width = 12.0, height = 6.0, units = "in", dpi = 300)

readr::write_lines(c(
  "# Figure 4 panel f: orthogonal displacement decomposition",
  "",
  "The plotted value is `100 * (D_k - H10_k) / H10_total`, where k is shared shift, between donors, within donors, or total. AD uses the PsychAD holdout-only healthy reference. COVID is decomposed within Ren, COMBAT, and Stephenson and then averaged with equal cohort weight; Platelet uses Ren and Stephenson because it is absent from COMBAT.",
  "",
  "Intervals are 95% donor-bootstrap percentile intervals. Disease and healthy values are drawn independently within each cohort but used in the same bootstrap replicate, so numerator and denominator are paired within replicate.",
  "",
  "The plotted AD order follows the current Figure 4 cell-type panels and excludes IN/Mural; the COVID populations are Global, B, Myeloid, NK, Platelet and T."
), file.path(outdir, "orthogonal_residual_variance_panel_f.md"))

cat("Wrote:", paste0(stem, ".svg"), "\n")
