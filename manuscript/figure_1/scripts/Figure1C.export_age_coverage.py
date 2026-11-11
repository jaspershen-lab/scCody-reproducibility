"""Export AIDA donor/age/lineage counts from the corrected linked matrix.

Use the public preprocessing workflows to prepare inputs for Figure 1
inputs, including all three current analyzed COVID case/control cohorts.
"""
import argparse
from pathlib import Path

import anndata as ad
import pandas as pd

REPO = Path(__file__).resolve().parents[3]


def counts(path):
    obs = ad.read_h5ad(path, backed="r").obs
    frame = pd.DataFrame({
        "donor_id": obs["donor_id"].astype(str).to_numpy(),
        "age": pd.to_numeric(obs["age"]).round().astype(int).to_numpy(),
        "Annotation_Level1": obs["Annotation_Level1"].astype(str).to_numpy(),
    })
    return frame.groupby(list(frame.columns), observed=True).size().rename("n_cells").reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aida", type=Path, default=REPO / "data/aida_counts_hvg.h5ad")
    parser.add_argument("--outdir", type=Path, default=REPO / "manuscript/figure_1/results")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    aida = counts(args.aida)
    aida.to_csv(args.outdir / "age_coverage_aida.csv", index=False)
    print("AIDA", aida.donor_id.nunique(), "donors")


if __name__ == "__main__":
    main()
