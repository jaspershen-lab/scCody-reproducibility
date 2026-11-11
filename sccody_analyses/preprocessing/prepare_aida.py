"""Prepare AIDA on the frozen development-only panel with full-gene CP10K/log1p.

Read raw counts in cell blocks and retain source row order. External ages define
whole-donor holdouts; the development input retains the per-fit internal ages.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

from .h5ad_metadata import read_csr_rows
from .rebuild_fullgene_development_hvg_inputs import Writer, normalize_before_subset
from .split_aida_age_holdout import age_masks

AIDA_L1_MAP = {"B": "B cell", "CD34_HSPC": "Progenitor cell", "DC": "DC", "ILC": "ILC",
               "Myeloid": "Myeloid", "NK": "NK cell", "Plasma_Cell": "Plasma cell",
               "Platelet": "Platelet", "T": "T cell"}
DEFAULT_PANEL = Path(__file__).resolve().parents[2] / "configs/aida_genes.tsv"


def prepare_obs(obs, age_key="age", donor_key="donor_id", cell_type_key="Annotation_Level1"):
    obs = obs.copy()
    ages = (obs[age_key] if age_key in obs else
            obs["development_stage"].astype(str).str.extract(r"(\d+(?:\.\d+)?)", expand=False))
    obs[age_key] = pd.to_numeric(ages, errors="coerce").astype(float)
    obs["cohort"] = "aida"
    obs["source_dataset"] = "aida"
    types = obs[cell_type_key].astype(str)
    obs["unified_cell_type"] = types.map(AIDA_L1_MAP).fillna(types)
    obs["sample_key"] = obs[donor_key].astype(str) + "_age" + obs[age_key].round().astype("Int64").astype(str)
    if "batch_key" not in obs:
        obs["batch_key"] = obs["library_id"].astype(str) if "library_id" in obs else "aida"
    return obs


def preprocess_aida(input_h5ad, outdir, *, panel=DEFAULT_PANEL, count_source="raw.X",
                    block_rows=5000, age_key="age", donor_key="donor_id",
                    cell_type_key="Annotation_Level1", max_pct_mt=20.0):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    matrix_key, var_key = {"raw.X": ("raw/X", "raw/var"), "X": ("X", "var")}.get(
        count_source, ("layers/" + count_source.removeprefix("layer:"), "var"))
    with h5py.File(input_h5ad) as src:
        obs = prepare_obs(read_elem(src["obs"]), age_key, donor_key, cell_type_key)
        var = read_elem(src[var_key])
        genes = pd.read_csv(panel, sep="\t")["gene_id"]
        positions = var.index.get_indexer(genes)
        # .loc keeps the frozen panel order and reports missing genes naturally.
        selected_var = var.loc[genes].copy()
        selected_var["highly_variable"] = True
        raw = src[matrix_key]
        keep = np.isfinite(obs[age_key].to_numpy())
        mt = var.index.astype(str).str.startswith("MT-")
        if mt.any():
            for start in range(0, len(obs), block_rows):
                stop = min(start + block_rows, len(obs))
                counts = read_csr_rows(raw, start, stop)
                totals = np.asarray(counts.sum(axis=1)).ravel()
                keep[start:stop] &= np.asarray(counts[:, mt].sum(axis=1)).ravel() / totals * 100 <= max_pct_mt
        splits = age_masks(obs[age_key].to_numpy())
        masks = [keep, keep & splits["train"], keep & splits["test"]]
        names = ["aida_counts_hvg.h5ad", "aida_age_holdout_train.h5ad", "aida_age_holdout_test.h5ad"]
        settings = {"count_source": count_source, "target_sum": 10000.0,
                    "normalization_order": "full genes -> CP10K -> log1p -> development panel"}
        writers = [Writer(outdir / name, input_h5ad, selected_var, settings, obs=obs.loc[mask].copy())
                   for name, mask in zip(names, masks)]
        for start in range(0, len(obs), block_rows):
            stop = min(start + block_rows, len(obs))
            selected = keep[start:stop]
            if not selected.any():
                continue
            expression, counts, totals = normalize_before_subset(read_csr_rows(raw, start, stop)[selected], positions)
            for writer, mask in zip(writers, masks):
                local = mask[start:stop][selected]
                if local.any():
                    writer.append(expression[local], counts[local], totals[local])
            print(f"[prepare] {stop:,}/{len(obs):,} cells", flush=True)
        for writer in writers:
            writer.finish()
            writer.temporary.replace(writer.path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--raw", dest="input", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--count-source", default="raw.X", help="raw.X, X, or layer:<name>")
    parser.add_argument("--block-rows", type=int, default=5000)
    args = parser.parse_args()
    preprocess_aida(args.input, args.outdir, panel=args.panel, count_source=args.count_source,
                    block_rows=args.block_rows)


if __name__ == "__main__":
    main()
