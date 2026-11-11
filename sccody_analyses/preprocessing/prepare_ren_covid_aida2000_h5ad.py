#!/usr/bin/env python3
"""Prepare Ren COVID cells from full-gene raw counts on the frozen AIDA panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from .h5ad_metadata import atomic_write, reference_mean, add_reference_means, mark_imputed
from .rebuild_fullgene_development_hvg_inputs import block, normalize_before_subset


DEFAULT_SOURCE = "data/covid/COVID19_ALL.h5ad"
DEFAULT_AIDA = (
    "data/aida_age_holdout_train.h5ad"
)
DEFAULT_OUTPUT = (
    "data/covid/"
    "ren_covid_case_control_aida2000_mean_imputed.h5ad"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-h5ad", default=DEFAULT_SOURCE)
    parser.add_argument("--aida-reference", default=DEFAULT_AIDA)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--min-age", type=float, default=19)
    parser.add_argument("--max-age", type=float, default=66)
    parser.add_argument("--block-rows", type=int, default=20_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").astype(str)


def _select_cells(
    obs: pd.DataFrame, *, min_age: float, max_age: float
) -> tuple[np.ndarray, dict[str, object]]:
    age = pd.to_numeric(_text(obs["Age"]), errors="coerce")
    sex = _text(obs["Sex"]).str.strip().str.lower()
    sample_type = _text(obs["Sample type"]).str.strip().str.lower()
    sample_time = _text(obs["Sample time"]).str.strip()
    severity = _text(obs["CoVID-19 severity"]).str.strip().str.lower()
    donor = _text(obs["PatientID"])
    major_type = _text(obs["majorType"]).str.strip()

    type_map = {
        "B": "B",
        "CD4": "T",
        "CD8": "T",
        "Mono": "Myeloid",
        "Macro": "Myeloid",
        "NK": "NK",
        "DC": "DC",
        "Mega": "Platelet",
        "Plasma": "Plasma_Cell",
    }
    mapped = major_type.map(type_map)
    donor_age_nunique = (
        pd.DataFrame({"donor": donor, "age": age})
        .loc[age.notna()]
        .groupby("donor")["age"]
        .nunique()
    )
    inconsistent_donors = set(donor_age_nunique[donor_age_nunique > 1].index.astype(str))

    sample_mask = sample_type.isin({"fresh pbmc", "frozen pbmc"})
    age_mask = age.between(min_age, max_age, inclusive="both")
    sex_mask = sex.isin({"f", "m"})
    severity_mask = severity.isin({"control", "mild/moderate", "severe/critical"})
    supported_mask = mapped.notna()
    consistent_mask = ~donor.isin(inconsistent_donors)
    selected = (
        sample_mask & age_mask & sex_mask & severity_mask & supported_mask & consistent_mask
    ).to_numpy()
    rows = np.flatnonzero(selected).astype(np.int64)

    condition = np.where(severity.to_numpy() == "control", "control", "case")
    return rows, {
        "age": age,
        "sex": sex,
        "sample_type": sample_type,
        "sample_time": sample_time,
        "severity": severity,
        "donor": donor,
        "major_type": major_type,
        "mapped": mapped,
        "condition": condition,
    }


def build(args: argparse.Namespace) -> None:
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite to replace it: {args.output}")

    print(f"[source] {args.source_h5ad}", flush=True)
    source = ad.read_h5ad(args.source_h5ad, backed="r")
    source_obs = source.obs
    rows, fields = _select_cells(
        source_obs, min_age=args.min_age, max_age=args.max_age
    )
    print(f"[select] {len(rows):,} cells", flush=True)

    print(f"[reference] {args.aida_reference}", flush=True)
    reference_means, reference_var, aida_categories = reference_mean(args.aida_reference, args.block_rows, sum_dtype=np.float64)
    reference_genes = reference_var.index.astype(str).tolist()
    reference_symbols = reference_var.feature_name.astype(str).tolist()

    source_genes = pd.Index(source.raw.var_names.astype(str))
    source_lookup = {gene: index for index, gene in enumerate(source_genes)}
    source_positions = np.asarray(
        [source_lookup.get(symbol, -1) for symbol in reference_symbols], dtype=np.int64
    )
    present_dest = np.flatnonzero(source_positions >= 0).astype(np.int64)
    missing_dest = np.flatnonzero(source_positions < 0).astype(np.int64)
    present_source = source_positions[present_dest]
    print(
        f"[axis] AIDA genes={len(reference_genes)}, observed={len(present_dest)}, "
        f"mean-imputed={len(missing_dest)}",
        flush=True,
    )

    selection = sparse.csr_matrix(
        (
            np.ones(len(present_dest), dtype=np.float32),
            (np.arange(len(present_dest)), present_dest),
        ),
        shape=(len(present_dest), len(reference_genes)),
    )
    selected_set = np.zeros(source.n_obs, dtype=bool)
    selected_set[rows] = True
    x_blocks, count_blocks, library_sizes = [], [], []
    raw_handle = h5py.File(args.source_h5ad)
    raw = raw_handle["raw/X"]
    for start in range(0, source.n_obs, args.block_rows):
        stop = min(start + args.block_rows, source.n_obs)
        local_rows = np.flatnonzero(selected_set[start:stop])
        if len(local_rows) == 0:
            continue
        full = block(raw, start, stop)[local_rows]
        expression, counts, totals = normalize_before_subset(full, present_source)
        aligned = add_reference_means((expression @ selection).tocsr(), missing_dest, reference_means)
        x_blocks.append(aligned)
        count_blocks.append((counts @ selection).tocsr())
        library_sizes.append(totals)
    raw_handle.close()
    matrix = sparse.vstack(x_blocks, format="csr", dtype=np.float32)

    selected_obs = source_obs.iloc[rows].copy()
    selected_obs["sample_id"] = _text(selected_obs["sampleID"]).to_numpy()
    selected_obs["donor_id"] = fields["donor"].iloc[rows].to_numpy()
    selected_obs["biological_donor_id"] = fields["donor"].iloc[rows].to_numpy()
    selected_obs["age"] = fields["age"].iloc[rows].to_numpy(dtype=np.float32)
    selected_obs["sex"] = fields["sex"].iloc[rows].map(
        {"f": "female", "m": "male"}
    ).to_numpy()
    selected_obs["condition"] = fields["condition"][rows]
    selected_obs["disease_group"] = fields["severity"].iloc[rows].to_numpy()
    selected_obs["sample_time"] = fields["sample_time"].iloc[rows].to_numpy()
    selected_obs["Annotation_Level1"] = pd.Categorical(
        fields["mapped"].iloc[rows].to_numpy(), categories=aida_categories
    )
    selected_obs["full_gene_library_size"] = np.concatenate(library_sizes)
    selected_obs.index = pd.Index(source.obs_names[rows].astype(str), dtype=object)
    source_obs_column_map = {
        str(column): str(column).replace("/", "_")
        for column in selected_obs.columns
        if "/" in str(column)
    }
    if source_obs_column_map:
        selected_obs = selected_obs.rename(columns=source_obs_column_map)

    reference_var.index = pd.Index(reference_genes, dtype=object)
    result = ad.AnnData(X=matrix, obs=selected_obs, var=reference_var)
    result.layers["counts"] = sparse.vstack(count_blocks, format="csr", dtype=np.float32)
    mark_imputed(result, missing_dest, reference_means)
    result.uns["normalization"] = {"target_sum": 10000.0, "input_space": "log1p"}
    print(f"[write] {result.shape} -> {args.output}", flush=True)
    atomic_write(result, args.output)
    source.file.close()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
