#!/usr/bin/env Rscript
# Preserve vector plot content in the assembled PDFs; PNGs are review exports.
repo <- normalizePath(".", mustWork = TRUE)
for (n in 1:5) {
  stem <- file.path(repo, "manuscript", paste0("figure_", n), "figure_files", paste0("figure", n, "_sccody"))
  svg <- paste0(stem, ".svg")
  if (!file.exists(svg)) stop("Missing assembly: ", svg)
  rsvg::rsvg_png(svg, paste0(stem, ".png"), width = 4800)
  rsvg::rsvg_pdf(svg, paste0(stem, ".pdf"), width = 1120)
  cat("[assembled] Figure ", n, "\n", sep = "")
}
