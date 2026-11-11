#!/usr/bin/env python3
"""Prepare Stephenson COVID case/control cells for the frozen AIDA models.

The Stephenson release contains raw CITE-seq counts in ``layers/raw`` and a
processed ``X`` that is not the AIDA model input space.  This script:

* keeps ``Status == Covid`` (case) or ``Status == Healthy`` (control),
* converts ``Age_interval`` to its interval midpoint,
* maps the release's ``initial_clustering`` labels to AIDA's nine level-1
  cell-type labels,
* caps cells per donor/cell type for balanced donor-level evaluation, and
* normalizes the gene-expression counts over the full gene set before
  reindexing to AIDA's ordered 2,000-gene panel.

Antibody-capture features are excluded from the library-size denominator.
Missing AIDA genes are filled with the corresponding per-gene mean from the
AIDA training ``X`` and recorded in ``uns``.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .h5ad_metadata import (atomic_write, constant_columns, dataframe_index, decode,
                             read_vector, read_obs_vector, read_csr_rows, reference_mean, mark_imputed)


DEFAULT_INPUT = (
    "data/covid/"
    "Stephenson_E-MTAB-10026_haniffa21.processed.h5ad"
)
DEFAULT_AIDA = "data/aida_age_holdout_train.h5ad"

CELL_TYPE_MAP = {
    "B_cell": "B",
    "CD4": "T",
    "CD8": "T",
    "CD14": "Myeloid",
    "CD16": "Myeloid",
    "DCs": "DC",
    "HSC": "CD34_HSPC",
    "Lymph_prolif": "T",
    "MAIT": "T",
    "Mono_prolif": "Myeloid",
    "NK_16hi": "NK",
    "NK_56hi": "NK",
    "Plasmablast": "Plasma_Cell",
    "Platelets": "Platelet",
    "Treg": "T",
    "gdT": "T",
    "pDC": "DC",
}

AGE_INTERVAL_RE = re.compile(
    r"^\s*\(\s*(?P<lower>[0-9]+(?:\.[0-9]+)?)\s*,\s*"
    r"(?P<upper>[0-9]+(?:\.[0-9]+)?)\s*\]\s*$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", default=DEFAULT_INPUT)
    parser.add_argument("--aida-h5ad", default=DEFAULT_AIDA)
    parser.add_argument("--status", choices=("Covid", "Healthy"), help="Omit to prepare both conditions")
    parser.add_argument("--out", default="data/covid/stephenson_covid_case_control_aida2000_mean_imputed.h5ad")
    parser.add_argument("--max-age", type=float, default=68.0)
    parser.add_argument("--min-age", type=float, default=0.0)
    parser.add_argument("--max-cells-per-donor-cell-type", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--target-sum", type=float, default=10000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def interval_midpoints(values: np.ndarray) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=float)
    for index, value in enumerate(values):
        match = AGE_INTERVAL_RE.match(str(value))
        if match:
            output[index] = (
                float(match.group("lower")) + float(match.group("upper"))
            ) / 2.0
            continue
        try:
            output[index] = float(value)
        except (TypeError, ValueError):
            pass
    return output


def build(args) -> None:
    output = Path(args.out).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")

    aida_means, aida_var, _ = reference_mean(args.aida_h5ad, args.chunk_size, sum_dtype=np.float64)
    aida_genes = aida_var.index.astype(str).to_numpy()
    aida_symbols = aida_var.feature_name.astype(str).to_numpy()

    with h5py.File(args.input_h5ad, "r") as handle:
        obs = handle["obs"]
        matrix = handle["layers"]["raw"]
        matrix_shape = tuple(int(value) for value in matrix.attrs["shape"])
        source_genes = decode(handle["var"][dataframe_index(handle["var"])][:]).astype(str)
        feature_type = read_vector(
            handle["var"]["feature_types"], handle["var"].get("__categories")
        ).astype(str)
        gene_positions = np.flatnonzero(feature_type == "Gene Expression").astype(np.int32)
        source_lookup = {gene: int(position) for position, gene in enumerate(source_genes)}
        shared_aida_pos = np.asarray(
            [position for position, symbol in enumerate(aida_symbols) if symbol in source_lookup],
            dtype=np.int32,
        )
        shared_source_pos = np.asarray(
            [source_lookup[aida_symbols[position]] for position in shared_aida_pos],
            dtype=np.int32,
        )
        missing_aida_pos = np.asarray(
            [position for position, symbol in enumerate(aida_symbols)
             if symbol not in source_lookup],
            dtype=np.int32,
        )

        status = read_obs_vector(obs, "Status").astype(str)
        age_interval = read_obs_vector(obs, "Age_interval").astype(str)
        age = interval_midpoints(age_interval)
        sex = np.char.lower(read_obs_vector(obs, "Sex").astype(str))
        donor = read_obs_vector(obs, "patient_id").astype(str)
        initial_type = read_obs_vector(obs, "initial_clustering").astype(str)
        mapped_type = pd.Series(initial_type, copy=False).map(CELL_TYPE_MAP)
        valid = np.isin(status, [args.status] if args.status else ["Covid", "Healthy"])
        valid &= np.isfinite(age) & (age >= args.min_age) & (age <= args.max_age)
        valid &= np.isin(sex, ["female", "male"])
        valid &= mapped_type.notna().to_numpy()

        selected_parts = []
        # Preserve each condition's recorded sampling when preparing both at once.
        for condition in ([args.status] if args.status else ["Covid", "Healthy"]):
            rng = np.random.default_rng(args.seed)
            grouped = defaultdict(list)
            for position in np.flatnonzero(valid & (status == condition)):
                grouped[(donor[position], str(mapped_type.iloc[position]))].append(position)
            for key in sorted(grouped):
                positions = np.asarray(grouped[key], dtype=np.int64)
                if len(positions) > args.max_cells_per_donor_cell_type:
                    positions = rng.choice(positions, args.max_cells_per_donor_cell_type, replace=False)
                selected_parts.append(np.sort(positions))
        selected = np.sort(np.concatenate(selected_parts)).astype(np.int64)

        index_node = obs[dataframe_index(obs)]
        source_cell_ids = read_vector(index_node)
        cell_id = source_cell_ids[selected].astype(str)

        expression_blocks: list[sp.csr_matrix] = []
        count_blocks: list[sp.csr_matrix] = []
        for start in range(0, matrix_shape[0], args.chunk_size):
            stop = min(start + args.chunk_size, matrix_shape[0])
            lo, hi = np.searchsorted(selected, [start, stop])
            if lo == hi:
                continue
            block = read_csr_rows(matrix, start, stop)[selected[lo:hi] - start]
            gene_counts = block[:, gene_positions].tocsr()
            totals = np.asarray(gene_counts.sum(axis=1)).ravel().astype(np.float32)
            shared = gene_counts[:, shared_source_pos].tocoo()
            aligned = sp.csr_matrix(
                (
                    shared.data,
                    (shared.row, shared_aida_pos[shared.col]),
                ),
                shape=(len(totals), len(aida_genes)),
                dtype=np.float32,
            )
            normalized = aligned.multiply((float(args.target_sum) / totals)[:, None]).tocsr()
            np.log1p(normalized.data, out=normalized.data)
            if len(missing_aida_pos):
                normalized = (
                    normalized
                    + constant_columns(
                        len(totals), len(aida_genes), missing_aida_pos,
                        aida_means[missing_aida_pos],
                    )
                ).tocsr()
            expression_blocks.append(normalized.astype(np.float32))
            count_blocks.append(aligned)
            print(f"[matrix] {stop:,}/{matrix_shape[0]:,} source rows processed", flush=True)

        expression = sp.vstack(expression_blocks, format="csr", dtype=np.float32)
        counts = sp.vstack(count_blocks, format="csr", dtype=np.float32)

        extra_fields = {
            "sample_id": "sample_id",
            "outcome": "Outcome",
            "status_on_day_collection": "Status_on_day_collection",
            "status_on_day_collection_summary": "Status_on_day_collection_summary",
            "swab_result": "Swab_result",
            "worst_clinical_status": "Worst_Clinical_Status",
            "resample": "Resample",
            "site": "Site",
        }
        metadata = pd.DataFrame(index=pd.Index(cell_id, name="cell_id"))
        metadata["sample_id"] = read_obs_vector(obs, "sample_id")[selected].astype(str)
        metadata["patient_id"] = donor[selected]
        metadata["biological_donor_id"] = donor[selected]
        metadata["donor_id"] = donor[selected]
        metadata["age"] = age[selected].astype(float)
        metadata["age_interval"] = age_interval[selected]
        metadata["age_inference"] = "Age_interval midpoint"
        metadata["sex"] = sex[selected]
        metadata["Status"] = status[selected]
        metadata["condition"] = np.where(status[selected] == "Covid", "case", "control")
        metadata["disease_group"] = status[selected]
        metadata["initial_clustering"] = initial_type[selected]
        metadata["Annotation_Level1"] = mapped_type.iloc[selected].to_numpy()
        for output_name, source_name in extra_fields.items():
            if source_name in obs and output_name not in metadata:
                metadata[output_name] = read_obs_vector(obs, source_name)[selected].astype(str)

    var = pd.DataFrame(
        {"feature_name": aida_symbols},
        index=pd.Index(aida_genes, name="gene_id"),
    )
    result = ad.AnnData(X=expression, obs=metadata, var=var)
    result.layers["counts"] = counts
    mark_imputed(result, missing_aida_pos, aida_means)
    result.uns["normalization"] = {"target_sum": args.target_sum, "input_space": "log1p"}
    atomic_write(result, output)
    print(
        f"[done] {output}: {result.n_obs:,} cells, {result.n_vars:,} genes, "
        f"{result.obs['donor_id'].nunique()} donors, {len(shared_aida_pos)} shared genes, "
        f"{len(missing_aida_pos)} mean-imputed genes",
        flush=True,
    )


def main():
    build(parse_args())


if __name__ == "__main__":
    main()
