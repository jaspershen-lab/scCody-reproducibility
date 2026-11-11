#!/usr/bin/env Rscript
# Figure 5: disease-associated displacement in AD excitatory neurons and COVID-19 myeloid cells.
#   a  AD EN program heatmap (all components)       b  AD EN subclass shared contribution
#   c  COVID program heatmap (all components)       d  COVID program dispersion (between / within)
#   e  COVID IFN score vs symptom onset             f  COVID CD14 / CD16 monocyte remodeling
#   Only the six main panels are rendered (names in figure5_style.R).
# Run prepare_figure5_data.py, prepare_covid_subtype_donor_table.py, prepare_program_tables.py and
# analyze_ifn_symptom_onset.py first (see README).
#   Rscript manuscript/figure_5/scripts/plot_figure5.R

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
source(file.path(dirname(sub("^--file=", "", script_arg[[1]])), "figure5_style.R"))

# COVID-19 program heatmaps and dispersion (Figure 5c-d).
suppressPackageStartupMessages(library(patchwork))
add_groups <- function(d) {
  d %>% filter(module %in% names(program_labels)) %>% mutate(program = factor(program_labels[module], levels = rev(program_row_order)),
               group = factor(program_group[as.character(program)], levels = names(program_group_colors)))
}
heat <- read_csv(file.path(data_dir, "covid_program_heatmap.csv"), show_col_types = FALSE) %>%
  add_groups() %>%
  mutate(cohort = factor(cohort, levels = cohort_order),
         component_label = factor(component_key[component], levels = names(component_colors)))
z_lim <- ceiling(max(abs(heat$null_z)))

group_gap <- unit(1.4, "mm")
heat_theme <- function() {
  theme_fig() +
    theme(axis.line = element_blank(), axis.ticks = element_blank(),
          axis.text.x = element_text(angle = 45, hjust = 1),
          axis.text.y = element_blank(),
          strip.text.y = element_blank(), panel.spacing.y = group_gap, panel.spacing.x = unit(3, "mm"))
}
group_bar <- function(d, colours = program_group_colors) {
  # Thin bar placed directly against the heatmap; program labels sit to its left.
  rows <- distinct(d, program, group)
  ggplot(rows, aes(x = 1, y = program, fill = group)) +
    geom_tile(width = 1, colour = NA) +
    facet_grid(group ~ ., scales = "free_y", space = "free_y") +
    scale_fill_manual(values = colours, name = NULL, drop = FALSE) +
    scale_x_continuous(expand = c(0, 0)) +
    labs(x = NULL, y = NULL) +
    theme_fig() +
    theme(axis.line = element_blank(), axis.ticks = element_blank(), axis.text.x = element_blank(),
          strip.text.y = element_blank(), panel.spacing.y = group_gap,
          plot.margin = margin(4, 0.5, 4, 4))
}
extract_legend <- function(plot, stem) {
  leg <- cowplot::get_plot_component(plot, "guide-box-right", return_all = TRUE)
  if (is.list(leg) && !inherits(leg, "grob")) leg <- leg[[1]]
  lw <- grid::convertWidth(sum(leg$widths), "in", valueOnly = TRUE) + 0.3  # room for long labels
  lh <- grid::convertHeight(sum(leg$heights), "in", valueOnly = TRUE) + 0.1
  save_device(cowplot::ggdraw(leg), stem, max(lw, 0.6), max(lh, 0.4))
}
save_grouped <- function(heat_plot, data, stem, width, height, bar_mm = 1.2, colours = program_group_colors) {
  bar <- group_bar(data, colours)
  extract_legend(heat_plot, paste0(stem, "__legend"))                # colour scale
  extract_legend(bar, paste0(stem, "__group_legend"))                # program groups
  combined <- (bar + theme(legend.position = "none")) +
    (heat_plot + theme(legend.position = "none", plot.margin = margin(4, 6, 4, 0.5))) +
    plot_layout(widths = unit(c(bar_mm, 1), c("mm", "null")))       # thin fixed-width group bar
  save_device(combined, stem, width, height)
  message("saved ", stem)
}
z_fill <- scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B", midpoint = 0,
                               limits = c(-z_lim, z_lim), name = "COVID-19 \u2212 healthy (z)",
                               guide = guide_colourbar(display = "rectangles"))
# z is the size-matched gene-set standardisation of the disease - healthy difference; it does not
# carry donor-level uncertainty, so tiles are muted and unstarred unless the BH-FDR q < 0.05 (two-sided
# permutation p, adjusted within each cohort x step family of the 10 plotted programs).  The aligned column is relative to the healthy +10 y movement, which is
# itself almost entirely aging-aligned.
sig_label <- function(z, p) paste0(sprintf("%.1f", z), if_else(p < 0.05, "*", ""))
sig_alpha <- scale_alpha_manual(values = c(`TRUE` = 1, `FALSE` = 0.35), guide = "none")
heat_tiles <- function(d) {
  ggplot(d, aes(cohort, program, fill = null_z)) +
    geom_tile(aes(alpha = q_two_sided < 0.05), colour = "white", linewidth = 0.4) +
    geom_text(aes(label = sig_label(null_z, q_two_sided)), size = 1.9, family = "sans") +
    z_fill + sig_alpha + labs(x = NULL, y = NULL) + heat_theme()
}

# Fig. 5c: all four movement components across cohorts.
p <- heat_tiles(heat) + facet_grid(group ~ component_label, scales = "free_y", space = "free_y")
save_grouped(p, heat, "e_COVID_program_heatmap_all_components", 7.0, 2.8)

# Fig. 5d. Dispersion counterpart of the between/within mean-shift columns:
#     COVID-19 / healthy ratio of the spread of the decoded program shift
#     (between: SD across donors; within: mean within-donor SD).
disp <- read_csv(file.path(data_dir, "covid_component_dispersion.csv"), show_col_types = FALSE) %>%
  add_groups() %>%
  mutate(cohort = factor(cohort, levels = cohort_order),
         component_label = factor(paste0(component_key[component], "\n(spread ratio)"),
                                  levels = paste0(component_key[c("between", "within")], "\n(spread ratio)")),
         log2_ratio = log2(ratio),
         label = paste0(sprintf("%.2f", ratio), if_else(q_greater < 0.05, "*", "")))  # BH-FDR, cohort x level
lr_lim <- 3
p <- ggplot(disp, aes(cohort, program, fill = pmax(pmin(log2_ratio, lr_lim), -lr_lim))) +
  geom_tile(colour = "white", linewidth = 0.4) +
  geom_text(aes(label = label), size = 1.9, family = "sans") +
  facet_grid(group ~ component_label, scales = "free_y", space = "free_y") +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B", midpoint = 0,
                       limits = c(-lr_lim, lr_lim), breaks = -3:3, labels = function(x) 2^x,
                       name = "Spread ratio",
                       guide = guide_colourbar(display = "rectangles")) +
  labs(x = NULL, y = NULL) + heat_theme()
save_grouped(p, disp, "f_COVID_program_dispersion_between_within", 4.5, 2.9)

# Fig. 5a (PsychAD EN; prepare_program_tables.py): AD - healthy, size-matched z.  The colour
# scale is set from AD values only (no cross-dataset comparison of z magnitudes).
ad_heat <- read_csv(file.path(data_dir, "ad_en_program_heatmap.csv"), show_col_types = FALSE) %>%
  filter(module %in% names(ad_program_labels)) %>%
  mutate(program = factor(ad_program_labels[module], levels = rev(ad_program_row_order)),
         group = factor(ad_program_group[as.character(program)], levels = names(ad_program_group_colors)),
         component_label = factor(component_key[component], levels = names(component_colors)))
ad_lim <- ceiling(max(abs(ad_heat$null_z)))
p <- ggplot(ad_heat, aes(component_label, program, fill = null_z)) +
  geom_tile(aes(alpha = q_two_sided < 0.05), colour = "white", linewidth = 0.4) +
  geom_text(aes(label = sig_label(null_z, q_two_sided)), size = 1.9, family = "sans") +
  sig_alpha +
  facet_grid(group ~ ., scales = "free_y", space = "free_y") +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B", midpoint = 0,
                       limits = c(-ad_lim, ad_lim), name = "AD \u2212 healthy (z)",
                       guide = guide_colourbar(display = "rectangles")) +
  labs(x = NULL, y = NULL) + heat_theme()
save_grouped(p, ad_heat, "c_AD_EN_program_heatmap_all_components", 3.6, 2.6, colours = ad_program_group_colors)

# Fig. 5f. COVID-19 CD14 / CD16 monocytes, COVID-19 vs healthy in every column; rows are the
#    same in all columns (cohort facets, CD14 / CD16 rows).  Within-patient columns (b-e) use
#    same-subtype matches; donor values pair-weighted with >= 30 cells per fold.
#      a  change in CD14 and CD16 % of Myeloid cells (pp)                covid_subtype_cell_share_change.csv
#      b  excess within-patient residual energy per cell                 covid_subtype_excess_same_subtype_stats.csv
#      c  within-patient change, Azimuth CD16 identity program           program_change.csv
#      d  within-patient dispersion ratio, calprotectin complex (GO)     program_dispersion.csv
#      e  within-patient dispersion ratio, CD16 identity (Azimuth)       program_dispersion.csv
#    95% donor-bootstrap CIs; filled points throughout, each labelled with its estimate.  * between the CD14 and CD16 rows (b-e) = the
#    direct CD16 - CD14 contrast of the COVID effect (log ratio of SD ratios in d, e) passes BH-FDR
#    q < 0.05 over the four outcomes within the cohort (paired donor bootstrap;
#    analyze_covid_subtype_contrasts.py -> covid_subtype_contrasts.csv).  Per-subtype estimates, CIs
#    and adjusted p: covid_subtype_contrasts.csv.
#    prepare_covid_subtype_donor_table.py, prepare_program_tables.py
# -----------------------------------------------------------------------------
recode_subtype <- function(x) recode(x, CD14_Mono = "CD14 monocytes", CD16_Mono = "CD16 monocytes")
sub_levels <- rev(names(subtype_colors))
tidy_sub <- function(d) {
  d %>% mutate(subtype = factor(recode_subtype(subtype), levels = sub_levels),
               cohort = factor(cohort, levels = cohort_order))
}
frac <- read_csv(file.path(data_dir, "covid_subtype_cell_share_change.csv"), show_col_types = FALSE) %>%
  tidy_sub() %>%
  transmute(cohort, subtype, est = change_pp, lo = ci_low, hi = ci_high, q = q_two_sided)
energy <- read_csv(file.path(data_dir, "covid_subtype_excess_same_subtype_stats.csv"), show_col_types = FALSE) %>%
  tidy_sub() %>% transmute(cohort, subtype, est = excess, lo = ci_low, hi = ci_high, q = q_two_sided)
v3_sub <- function(file, ...) {
  read_csv(file.path(data_dir, file), show_col_types = FALSE) %>%
    filter(dataset == "COVID", matches == "same", step == "within", cells %in% c("CD14_Mono", "CD16_Mono"), ...) %>%
    rename(subtype = cells) %>% tidy_sub()
}
azimuth <- "CD16 monocyte (Azimuth PBMC, Hao 2021)"
ident <- v3_sub("program_change.csv", module == azimuth) %>%
  transmute(cohort, subtype, est = difference, lo = ci_low, hi = ci_high, q = q_two_sided)
wsd <- v3_sub("program_dispersion.csv", statistic == "within_patient")
calpro_sd <- wsd %>% filter(module == "Calprotectin complex (GO:1990660)") %>%
  transmute(cohort, subtype, est = ratio, lo = ci_low, hi = ci_high, q = q_greater)
ident_sd <- wsd %>% filter(module == azimuth) %>%
  transmute(cohort, subtype, est = ratio, lo = ci_low, hi = ci_high, q = q_greater)

contrasts6 <- read_csv(file.path(data_dir, "covid_subtype_contrasts.csv"), show_col_types = FALSE) %>%
  mutate(cohort = factor(cohort, levels = cohort_order))
sd_range <- range(c(calpro_sd$lo, calpro_sd$hi, ident_sd$lo, ident_sd$hi))
sd_range[2] <- sd_range[2] * 1.12                       # room for the contrast asterisk
sub_panel <- function(d, xlab, ref = 0, log = FALSE, first = FALSE, last = FALSE, breaks = waiver(),
                      outcome = NULL, digits = 2) {
  p <- ggplot(d, aes(est, subtype)) +
    geom_vline(xintercept = ref, linetype = "dashed", linewidth = 0.3, colour = "grey45") +
    geom_errorbar(aes(xmin = lo, xmax = hi, colour = subtype), width = 0, orientation = "y", linewidth = 0.5) +
    geom_point(aes(colour = subtype), size = 1.6) +
    geom_label(aes(label = sub("^-(0\\.?0*)$", "\\1", formatC(est, format = "f", digits = digits)), colour = subtype),
               nudge_y = 0.32, size = 1.7, family = "sans", fill = "white", linewidth = 0,
               label.padding = unit(0.25, "mm"), show.legend = FALSE) +
    facet_grid(cohort ~ .) +
    scale_colour_manual(values = subtype_colors, name = NULL, breaks = names(subtype_colors), drop = FALSE) +
    scale_y_discrete(drop = FALSE, labels = function(x) sub(" monocytes", "", x)) +
    labs(x = xlab, y = NULL) + theme_fig() +
    theme(strip.text.y = element_blank(), panel.spacing.y = unit(2, "mm"), legend.position = "none",
          plot.margin = margin(4, 3, 4, 3))
  if (!is.null(outcome)) {
    # asterisk midway between the CD14 and CD16 rows, just right of the larger upper CI
    star <- d %>% group_by(cohort) %>% summarise(hi = max(hi), .groups = "drop") %>%
      inner_join(filter(contrasts6, outcome == !!outcome, q < 0.05), by = "cohort")
    span <- if (log) NA else diff(range(c(d$lo, d$hi)))
    star <- mutate(star, x = if (log) hi * 1.05 else hi + 0.05 * span)
    p <- p + geom_text(data = star, aes(x = x, y = 1.5), label = "*", size = 3, family = "sans",
                       vjust = 0.75, inherit.aes = FALSE)
  }
  p <- p + if (log) scale_x_log10(breaks = c(0.75, 1, 1.5), limits = sd_range) else
    scale_x_continuous(breaks = breaks, expand = expansion(mult = c(0.05, if (is.null(outcome)) 0.05 else 0.12)))
  if (!first) p <- p + theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
  if (last) p <- p + theme(strip.text.y = element_text(angle = 0, hjust = 0, face = "bold"))
  p
}
panels6 <- list(
  sub_panel(frac, "\u0394 cell fraction (pp)", first = TRUE, breaks = c(-10, 0, 10), digits = 1),
  sub_panel(energy, "\u0394 residual energy / cell", outcome = "energy", breaks = c(0, 40, 80), digits = 0),
  sub_panel(ident, "\u0394 CD16 identity", outcome = "identity"),
  sub_panel(calpro_sd, "Calprotectin SD ratio", ref = 1, log = TRUE, outcome = "calpro_sd"),
  sub_panel(ident_sd, "CD16 identity SD ratio", ref = 1, log = TRUE, last = TRUE, outcome = "ident_sd")
)
# legend: subtype colours and the asterisk definition
leg_plot <- ggplot(tibble(subtype = factor(names(subtype_colors), levels = names(subtype_colors)), x = 1, y = 1),
                   aes(x, y, colour = subtype)) +
  geom_point(size = 1.6) + scale_colour_manual(values = subtype_colors, name = NULL) + theme_fig()
leg6 <- cowplot::get_plot_component(leg_plot, "guide-box-right", return_all = TRUE)
if (is.list(leg6) && !inherits(leg6, "grob")) leg6 <- leg6[[1]]
leg6 <- cowplot::plot_grid(leg6, cowplot::ggdraw() +
                             cowplot::draw_label("* q < 0.05",
                                                 size = 7, hjust = 0, x = 0.08, fontfamily = "sans"),
                           ncol = 1, rel_heights = c(1, 0.8))
save_device(leg6, "h_COVID_monocyte_subtype_remodeling__legend", 1.6, 0.9)
save_device(wrap_plots(panels6, nrow = 1, widths = c(1, 1, 1, 1, 1)),
            "h_COVID_monocyte_subtype_remodeling", 7.2, 2.4)
message("saved h_COVID_monocyte_subtype_remodeling")

# -----------------------------------------------------------------------------
# Fig. 5b: subclass shared contribution (plot_figure5_ad_subclass.R)
# -----------------------------------------------------------------------------
source(file.path(script_dir, "plot_figure5_ad_subclass.R"))

# -----------------------------------------------------------------------------
# Fig. 5e: IFN score vs symptom onset (plot_figure5_subtype_onset.R)
# -----------------------------------------------------------------------------
source(file.path(script_dir, "plot_figure5_subtype_onset.R"))
