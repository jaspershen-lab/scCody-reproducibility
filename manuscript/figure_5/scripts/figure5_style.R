# Shared style for the Figure 5 plots: palettes, theme and export helper.
# Every plot is saved individually (SVG, PDF, PNG) with its legend in a separate
# file, so panels can be assembled in Illustrator.  Text stays as text:
#   SVG  svglite (text elements, font-family Arial)
#   PDF  cairo_pdf, embedded Arial-metric font subset; text remains text (needed for
#        Greek letters and the minus sign, which the base pdf() device cannot encode)

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(tidyr)
  library(svglite)
  library(cowplot)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  normalizePath(dirname(sub("^--file=", "", script_arg[[1]])), mustWork = TRUE)
} else {
  normalizePath(file.path(getwd(), "manuscript", "figure_5", "scripts"), mustWork = TRUE)
}
# Tables are read from manuscript/figure_5/results; plots go to figure_files/panels.
fig_dir <- file.path(dirname(script_dir), "figure_files", "panels")
data_dir <- file.path(dirname(script_dir), "results")
for (d in c("svg", "pdf", "png")) dir.create(file.path(fig_dir, d), recursive = TRUE, showWarnings = FALSE)

# Decomposition components (consistent across AD and COVID plots).
component_colors <- c(
  "Aging-aligned"   = "#0072B2",
  "Shared shift"    = "#D55E00",
  "Between-patient" = "#009E73",
  "Within-patient"  = "#CC79A7"
)
component_key <- c(aligned = "Aging-aligned", shared = "Shared shift",
                   between = "Between-patient", within = "Within-patient")

# COVID gene programs (consistent across COVID plots).  Keys are the curated
# module names (configs/programs/curated/module_sources.tsv); values are display labels.
program_labels <- c(
  "Interferon \u03b1/\u03b2 signaling" = "IFN-\u03b1/\u03b2 signaling (Reactome)",
  "Interferon-\u03b1 response" = "IFN-\u03b1 response (Hallmark)",
  "Interferon-\u03b3 response" = "IFN-\u03b3 response (Hallmark)",
  "MHC class II protein complex" = "MHC class II complex (GO)",
  "MS1 monocyte score (Reyes 2020)" = "MS1 monocyte score (Reyes 2020)",
  "Calprotectin S100A8/A9 + S100A12" = "S100A8/A9/A12",
  "TNF\u03b1 signaling via NF-\u03baB" = "TNF\u03b1/NF-\u03baB (Hallmark)",
  "Inflammatory response" = "Inflammatory response (Hallmark)",
  "IL-6/JAK/STAT3 signaling" = "IL-6/JAK/STAT3 (Hallmark)",
  "Non-classical monocyte (Travaglini 2020)" = "Non-classical monocyte (Travaglini 2020)"
)
program_colors <- setNames(
  c("#08519C", "#3182BD", "#9ECAE1", "#009E73", "#E69F00", "#8C510A", "#CC79A7", "#B2182B", "#8C6D31", "#666666"),
  unname(program_labels)
)
program_order <- names(program_colors)

# Row groups and fixed row order for the COVID program heatmaps (Fig. 5c, 5d and the supplement).
# One qualitative palette for the program-group bars of all COVID and AD heatmaps, so no two
# groups across the two datasets share or nearly share a colour.
group_palette <- c("Interferon signaling" = "#4E79A7", "Inflammatory signaling" = "#E15759",
                   "Myeloid remodeling" = "#59A14F", "AD pathology-associated signatures" = "#B07AA1",
                   "Neuronal signaling and function" = "#F28E2B", "Stress and inflammatory responses" = "#9C755F")
program_group_colors <- group_palette[c("Interferon signaling", "Inflammatory signaling", "Myeloid remodeling")]
program_groups <- list(
  "Interferon signaling" = c("IFN-\u03b1/\u03b2 signaling (Reactome)", "IFN-\u03b1 response (Hallmark)",
                             "IFN-\u03b3 response (Hallmark)"),
  "Inflammatory signaling" = c("TNF\u03b1/NF-\u03baB (Hallmark)", "Inflammatory response (Hallmark)",
                               "IL-6/JAK/STAT3 (Hallmark)"),
  "Myeloid remodeling" = c("MHC class II complex (GO)", "S100A8/A9/A12", "MS1 monocyte score (Reyes 2020)",
                           "Non-classical monocyte (Travaglini 2020)")
)
program_row_order <- unlist(program_groups, use.names = FALSE)
program_group <- setNames(rep(names(program_groups), lengths(program_groups)), program_row_order)
stopifnot(setequal(program_row_order, unname(program_labels)))

# AD EN gene programs (curated modules).
ad_program_labels <- c(
  "AD-pathology EN genes, up (Mathys 2023)" = "AD-pathology EN genes up (Mathys 2023)",
  "AD-pathology EN genes, down (Mathys 2023)" = "AD-pathology EN genes down (Mathys 2023)",
  "Synaptic signaling" = "Synaptic signaling (GO)",
  "Calcium ion transmembrane transport" = "Ca\u00b2\u207a transmembrane transport (GO)",
  "Response to oxidative stress" = "Response to oxidative stress (GO)",
  "Response to heat" = "Response to heat (GO)",
  "TNF\u03b1 signaling via NF-\u03baB" = "TNF\u03b1/NF-\u03baB (Hallmark)"
)

dataset_shapes <- c("AD" = 16, "COVID-19" = 17)
cohort_order <- c("Ren", "COMBAT", "Stephenson")
subtype_colors <- c("CD14 monocytes" = "#4D4D4D", "CD16 monocytes" = "#B8860B")
gene_class_colors <- c("Layer / neuronal identity" = "#D55E00", "Synaptic" = "#0072B2", "Other" = "#BDBDBD")

theme_fig <- function(base_size = 7) {
  theme_classic(base_size = base_size, base_family = "sans") +
    theme(
      axis.line = element_line(linewidth = 0.3, colour = "black"),
      axis.ticks = element_line(linewidth = 0.3, colour = "black"),
      axis.text = element_text(colour = "black", size = base_size),
      axis.title = element_text(colour = "black", size = base_size + 0.5),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = base_size + 0.5),
      legend.title = element_text(size = base_size),
      legend.text = element_text(size = base_size),
      legend.key.size = unit(3, "mm"),
      plot.title = element_blank(),
      plot.margin = margin(4, 6, 4, 4)
    )
}

# File names of the Figure 5 main panels (plot stems on the left).
main_panel_names <- c(
  c_AD_EN_program_heatmap_all_components = "Figure5A.AD_EN_program_heatmap",
  d_AD_EN_subclass_shared_contribution = "Figure5B.AD_EN_subclass_shared_contribution",
  e_COVID_program_heatmap_all_components = "Figure5C.COVID_program_heatmap",
  f_COVID_program_dispersion_between_within = "Figure5D.COVID_program_dispersion",
  g_COVID_IFN_symptom_onset = "Figure5E.COVID_IFN_symptom_onset",
  h_COVID_monocyte_subtype_remodeling = "Figure5F.COVID_monocyte_subtype_remodeling"
)

save_device <- function(plot, stem, width, height) {
  out_dir <- fig_dir
  base <- sub("__(group_)?legend$", "", stem)
  if (!base %in% names(main_panel_names)) stop("Unselected panel: ", stem)
  if (base %in% names(main_panel_names)) stem <- paste0(main_panel_names[[base]], substring(stem, nchar(base) + 1))
  for (d in c("svg", "pdf", "png")) dir.create(file.path(out_dir, d), recursive = TRUE, showWarnings = FALSE)
  svglite::svglite(file.path(out_dir, "svg", paste0(stem, ".svg")), width = width, height = height,
                   system_fonts = list(sans = "Arial"), fix_text_size = FALSE)
  print(plot); invisible(dev.off())
  grDevices::cairo_pdf(file.path(out_dir, "pdf", paste0(stem, ".pdf")), width = width, height = height,
                       family = "Arial")
  print(plot); invisible(dev.off())
  ggsave(file.path(out_dir, "png", paste0(stem, ".png")), plot, width = width, height = height,
         dpi = 600, bg = "white")
}

# Save the plot without its legend, and the legend (if any) as its own file.
save_fig <- function(plot, stem, width, height, legend = TRUE) {
  if (legend) {
    leg <- tryCatch(cowplot::get_plot_component(plot, "guide-box-right", return_all = TRUE),
                    error = function(e) NULL)
    if (is.list(leg) && !inherits(leg, "grob")) leg <- leg[[1]]
    if (!is.null(leg) && !inherits(leg, "zeroGrob")) {
      lw <- grid::convertWidth(sum(leg$widths), "in", valueOnly = TRUE) + 0.1
      lh <- grid::convertHeight(sum(leg$heights), "in", valueOnly = TRUE) + 0.1
      save_device(cowplot::ggdraw(leg), paste0(stem, "__legend"), max(lw, 0.6), max(lh, 0.4))
    }
    plot <- plot + theme(legend.position = "none")
  }
  save_device(plot, stem, width, height)
  message("saved ", stem)
}

# Row groups and fixed row order for the AD EN program heatmap (Fig. 5a).
ad_program_group_colors <- group_palette[c("AD pathology-associated signatures", "Neuronal signaling and function",
                                           "Stress and inflammatory responses")]
ad_program_groups <- list(
  "AD pathology-associated signatures" = c("AD-pathology EN genes up (Mathys 2023)",
                                           "AD-pathology EN genes down (Mathys 2023)"),
  "Neuronal signaling and function" = c("Synaptic signaling (GO)", "Ca\u00b2\u207a transmembrane transport (GO)"),
  "Stress and inflammatory responses" = c("Response to oxidative stress (GO)", "Response to heat (GO)",
                                          "TNF\u03b1/NF-\u03baB (Hallmark)")
)
ad_program_row_order <- unlist(ad_program_groups, use.names = FALSE)
ad_program_group <- setNames(rep(names(ad_program_groups), lengths(ad_program_groups)), ad_program_row_order)
stopifnot(setequal(ad_program_row_order, unname(ad_program_labels)))
