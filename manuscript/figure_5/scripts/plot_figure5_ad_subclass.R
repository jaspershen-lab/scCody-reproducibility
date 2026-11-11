# Fig. 5b: AD excitatory-neuron subclasses
# (analyze_ad_subclass_residual_readout.py).  Sourced from plot_figure5.R.
# Cells are grouped by the subclass of the control (source) cell; labels are
# attached after coarse-group OT and do not constrain matching.

readout <- read_csv(file.path(data_dir, "ad_en_subclass_latent_readout.csv"), show_col_types = FALSE) %>%
  filter(disease_n_donors >= 5, healthy_n_donors >= 5)  # drops EN_L5_ET (0 AD cells, 1 healthy donor)
subclass_order <- sort(unique(readout$label))
pretty_subclass <- function(x) gsub("_", " ", gsub("(\\d)_(\\d)", "\\1/\\2", sub("^EN_", "", x)))
readout <- readout %>% mutate(subclass = factor(pretty_subclass(label), levels = rev(pretty_subclass(subclass_order))))
mode_colors <- c("AD" = "black", "Healthy +10 y" = "grey60")

long_metric <- function(metric) {
  bind_rows(
    readout %>% transmute(subclass, mode = "AD", value = .data[[paste0("disease_", metric)]],
                          low = .data[[paste0("disease_", metric, "_ci_low")]],
                          high = .data[[paste0("disease_", metric, "_ci_high")]]),
    readout %>% transmute(subclass, mode = "Healthy +10 y", value = .data[[paste0("healthy_", metric)]],
                          low = .data[[paste0("healthy_", metric, "_ci_low")]],
                          high = .data[[paste0("healthy_", metric, "_ci_high")]])
  ) %>% filter(is.finite(value)) %>% mutate(mode = factor(mode, levels = names(mode_colors)))
}

dot_ci <- function(d, xlab, fill_colour, vline = 0) {
  ggplot(d, aes(value, subclass, shape = mode)) +
    { if (!is.null(vline)) geom_vline(xintercept = vline, linetype = "dashed", linewidth = 0.3, colour = "grey45") } +
    geom_errorbar(aes(xmin = low, xmax = high, colour = mode), width = 0, orientation = "y", linewidth = 0.45,
                  position = position_dodge(width = 0.55)) +
    geom_point(aes(colour = mode, fill = mode), size = 1.5, position = position_dodge(width = 0.55)) +
    scale_shape_manual(values = c("AD" = 21, "Healthy +10 y" = 21), name = NULL) +
    scale_colour_manual(values = c("AD" = fill_colour, "Healthy +10 y" = "grey55"), name = NULL) +
    scale_fill_manual(values = c("AD" = fill_colour, "Healthy +10 y" = "white"), name = NULL) +
    labs(x = xlab, y = NULL) + theme_fig()
}


# Fig. 5b. Signed displacement along the pooled AD shared direction (AD vs healthy +10 y,
# healthy projected on the same direction; donor-bootstrap 95% CI) with an adjacent
# column of each subclass's contribution to the population shared shift
# (AD cell share x mean projection, as % of the summed contribution; donor-bootstrap
# 95% CI from analyze_ad_subclass_residual_readout.py, 2,000 draws).
suppressPackageStartupMessages(library(patchwork))
contrib <- read_csv(file.path(data_dir, "ad_en_subclass_shared_contribution.csv"), show_col_types = FALSE) %>%
  filter(label %in% readout$label) %>%
  mutate(subclass = factor(pretty_subclass(label), levels = levels(readout$subclass)),
         text = sprintf("%.1f%%", percent))
dots <- dot_ci(long_metric("shared_proj"), "Signed displacement along pooled AD\nshared direction (latent units)",
               component_colors[["Shared shift"]])
bars <- ggplot(contrib, aes(percent, subclass)) +
  geom_col(fill = component_colors[["Shared shift"]], alpha = 0.35, width = 0.6) +
  geom_errorbar(aes(xmin = ci_low, xmax = ci_high), width = 0, orientation = "y", linewidth = 0.4,
                colour = component_colors[["Shared shift"]]) +
  geom_text(aes(x = pmax(ci_high, percent) + 2, label = text), hjust = 0, size = 1.9, family = "sans") +
  scale_x_continuous(limits = c(0, 100), breaks = c(0, 50, 100), expand = c(0, 0)) +
  labs(x = "Contribution\nto shift (%)", y = NULL) +
  theme_fig() +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(), axis.line.y = element_blank(),
        plot.margin = margin(4, 6, 4, 2))
leg <- cowplot::get_plot_component(dots + theme(legend.position = "bottom", legend.direction = "horizontal"),
                                   "guide-box-bottom", return_all = TRUE)
if (is.list(leg) && !inherits(leg, "grob")) leg <- leg[[1]]
save_device(cowplot::ggdraw(leg), "d_AD_EN_subclass_shared_contribution__legend",
            grid::convertWidth(sum(leg$widths), "in", valueOnly = TRUE) + 0.3,
            grid::convertHeight(sum(leg$heights), "in", valueOnly = TRUE) + 0.1)
combined <- (dots + theme(legend.position = "none", plot.margin = margin(4, 2, 4, 4))) + bars +
  plot_layout(widths = c(1, 0.45))
save_device(combined, "d_AD_EN_subclass_shared_contribution", 3.9, 3.0)
message("saved d_AD_EN_subclass_shared_contribution")
