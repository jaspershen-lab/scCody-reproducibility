# Main figures

The final plotting tables are bundled in `figure_*/results/`, with the small
schematic assets in `figure_*/figure_files/schematics/`. View the assembled
figures in `figure_*/figure_files/figureN_sccody.{svg,pdf,png}`.

To redraw every panel and assemble the five figures:

```bash
Rscript manuscript/render_main_figures.R --install-deps
```

R with Cairo support and Python 3 are required. The command installs missing
CRAN plotting packages when `--install-deps` is supplied. Use `--panels-only`
to render only the data panels. Raw expression data, model fitting and private
repositories are not needed for plotting.

For preprocessing, fitting and scientific analyses, see the repository README.
