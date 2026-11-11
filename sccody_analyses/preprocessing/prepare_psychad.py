"""Prepare the PsychAD MSSM DLPFC panel with exact ages and full-library CP10K."""
import argparse
import gc
import re
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp

from .h5ad_metadata import atomic_write, make_h5ad_frames_writable

_EXACT_AGE_RE = re.compile(r'^\d+-year-old stage$')
_AGE_EXTRACT_RE = re.compile(r'(?P<age>\d+(?:\.\d+)?)\s*[- ]?\s*year\s*[- ]?\s*old', re.IGNORECASE)


def add_age_from_development_stage(adata, source_col="development_stage", new_col="age"):
    s = adata.obs[source_col].astype(str)
    adata.obs[new_col] = [
        float(m.group("age")) if (m := _AGE_EXTRACT_RE.search(v)) else np.nan
        for v in s
    ]
    return adata


def filter_exact_age_label(adata, dev_col="development_stage"):
    """Keep only cells whose development_stage is exactly 'N-year-old stage'."""
    mask = adata.obs[dev_col].astype(str).str.match(_EXACT_AGE_RE)
    adata = adata[mask].copy()
    return adata


def ensure_raw_counts(adata):
    """
    CellxGene published h5ads store normalised data in X and raw counts in raw.X.
    """
    if adata.raw is not None:
        raw = adata.raw.to_adata()
        raw.obs = adata.obs.copy()
        return raw
    return adata


def preprocess_qc_hvg_log(
    adata,
    *,
    hvg_n=2000,
    target_sum=1e4,
    max_pct_mt=20,
    min_cells_per_gene=10,
    keep_counts_layer=True,
):
    add_age_from_development_stage(adata)

    total_counts_full = np.asarray(adata.X.sum(axis=1)).flatten()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    mt_counts = np.asarray(adata.X[:, adata.var["mt"].to_numpy()].sum(axis=1)).ravel()
    mt_mask = mt_counts / total_counts_full * 100 <= max_pct_mt
    gene_ok = np.asarray((adata.X != 0).sum(axis=0)).flatten() >= min_cells_per_gene
    hvg_subsample_max = 1_000_000
    hvg_strat_col = "class" if "class" in adata.obs else None
    mt_pass_obs = adata.obs.loc[mt_mask]
    if mt_pass_obs.shape[0] > hvg_subsample_max and hvg_strat_col is not None:
        frac = hvg_subsample_max / mt_pass_obs.shape[0]
        sub_idx = (
            mt_pass_obs.groupby(hvg_strat_col, observed=True)
            .apply(lambda g: g.sample(n=max(1, round(len(g) * frac)), random_state=0).index)
        )
        sub_idx = np.concatenate([np.asarray(v) for v in sub_idx.to_numpy()])
        hvg_input = adata[sub_idx, gene_ok].copy()
    else:
        sub_idx = mt_pass_obs.index.to_numpy()
        hvg_input = adata[sub_idx, gene_ok].copy()

    sc.pp.highly_variable_genes(
        hvg_input,
        n_top_genes=min(hvg_n, hvg_input.n_vars),
        flavor="seurat_v3",
        batch_key=None,
        inplace=True,
    )
    hvg_mask_within_gene_ok = hvg_input.var["highly_variable"].to_numpy().copy()
    del hvg_input
    gc.collect()

    hvg_mask = np.zeros(adata.n_vars, dtype=bool)
    hvg_mask[np.flatnonzero(gene_ok)[hvg_mask_within_gene_ok]] = True

    adata_gene_sliced = adata[:, hvg_mask].copy()
    del adata
    gc.collect()

    adata_hvg = adata_gene_sliced[mt_mask].copy()
    total_counts_final = total_counts_full[mt_mask]
    del adata_gene_sliced
    gc.collect()

    if keep_counts_layer:
        adata_hvg.layers["counts"] = adata_hvg.X.copy()

    size_factor = target_sum / np.clip(total_counts_final, 1, None)
    adata_hvg.X = sp.diags(size_factor).dot(adata_hvg.X).tocsr()
    sc.pp.log1p(adata_hvg)

    adata_hvg.uns["preprocessing"] = {"target_sum": target_sum, "input_space": "log1p",
                                        "normalization": "full-library CP10K then log1p",
                                        "hvg_flavor": "seurat_v3"}
    return adata_hvg


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    adata = ensure_raw_counts(sc.read_h5ad(args.input))
    make_h5ad_frames_writable(adata)
    adata = filter_exact_age_label(adata)
    result = preprocess_qc_hvg_log(adata)
    output = args.outdir / "DLPFC_neurodegen_preprocessed.h5ad"
    atomic_write(result, output)
    print(f"[write] {output}: {result.n_obs:,} cells, {result.n_vars:,} genes")


if __name__ == "__main__":
    main()
