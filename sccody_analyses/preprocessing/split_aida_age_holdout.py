"""Split AIDA by the fixed external ages and optional internal validation ages."""
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .h5ad_metadata import atomic_write

EXTERNAL_AGES = [21, 31, 41, 51, 61, 68]


def age_masks(ages, internal_ages=()):
    ages = np.asarray(ages, dtype=float)
    finite = np.isfinite(ages)
    external = finite & np.isin(ages, EXTERNAL_AGES)
    internal = finite & ~external & np.isin(ages, internal_ages)
    return {"train": finite & ~external & ~internal, "validation": internal, "test": external}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/aida_counts_hvg.h5ad"))
    parser.add_argument("--outdir", type=Path, default=Path("data"))
    parser.add_argument("--age-key", default="age")
    parser.add_argument("--internal-ages", type=float, nargs="*", default=[])
    args = parser.parse_args()
    adata = ad.read_h5ad(args.input)
    masks = age_masks(pd.to_numeric(adata.obs[args.age_key], errors="coerce"), args.internal_ages)
    for split in ["train", "test"] + (["validation"] if args.internal_ages else []):
        path = args.outdir / f"aida_age_holdout_{split}.h5ad"
        part = adata[masks[split]].copy()
        atomic_write(part, path)
        print(f"[write] {path}: {part.n_obs:,} cells")


if __name__ == "__main__":
    main()
