"""Split PsychAD MSSM DLPFC into healthy and dementia/Alzheimer groups."""
import argparse
from pathlib import Path

import anndata as ad

from .h5ad_metadata import atomic_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    adata = ad.read_h5ad(args.input)
    cohorts = {"normal": "normal", "dementia_Alzheimer_disease": "dementia || Alzheimer disease"}
    parts = []
    for name, label in cohorts.items():
        part = adata[adata.obs["disease"] == label].copy()
        part.obs["condition"] = "control" if name == "normal" else "case"
        parts.append(part)
        path = args.outdir / f"psychad_dlpfc__{name}.h5ad"
        atomic_write(part, path)
        print(f"[write] {path}: {part.n_obs:,} cells")
    atomic_write(ad.concat(parts, merge="same", uns_merge="same"), args.outdir / "ad_case_control.h5ad")


if __name__ == "__main__":
    main()
