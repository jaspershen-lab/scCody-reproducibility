#!/usr/bin/env Rscript
# Assemble plotting tables from the public model and comparator reports.
python <- Sys.getenv("SCCODY_PYTHON", "python3")
status <- system2(python, c("-m", "sccody_analyses.collect_figure3"))
if (status != 0L) stop("Figure 3 table assembly failed: ", status)
