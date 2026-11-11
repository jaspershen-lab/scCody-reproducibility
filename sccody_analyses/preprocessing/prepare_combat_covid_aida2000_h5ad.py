#!/usr/bin/env python3
"""Prepare the COMBAT CITE-seq cohort for projection with the frozen AIDA models.

The COMBAT ``Age`` field is decade-coded (for example, 3 means the 30s and
1.8 denotes the 18--29 decade).  This script converts those bins to their
midpoints, uses the raw gene-expression layer, and aligns the result to the
ordered AIDA 2,000-gene axis.  AIDA genes absent from COMBAT are filled with
their unweighted AIDA-training mean in model-input space.

The primary severity analysis keeps the standard COMBAT groups:

* ``HV`` -> control
* ``COVID_MILD`` and ``COVID_HCW_MILD`` -> mild case
* ``COVID_SEV`` and ``COVID_CRIT`` -> severe case

Sepsis, influenza, and COVID_LDN are retained in the source file but are
excluded from this primary case/control projection.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .h5ad_metadata import (atomic_write, constant_columns, dataframe_index, decode,
                             read_vector, read_obs_vector, read_csr_rows, reference_mean, mark_imputed)


DEFAULT_INPUT = "data/covid/COMBAT-CITESeq-DATA.h5ad"
DEFAULT_AIDA = "data/aida_age_holdout_train.h5ad"

SOURCE_TO_SEVERITY = {
    "HV": "control",
    "COVID_MILD": "mild/moderate",
    "COVID_HCW_MILD": "mild/moderate",
    "COVID_SEV": "severe/critical",
    "COVID_CRIT": "severe/critical",
}

# COMBAT stores the first age bin as 1.8 and later bins as decade lower
# bounds.  Midpoints are used because the public metadata does not provide
# exact donor ages.
AGE_MIDPOINTS = {
    1.8: 23.5,  # 18--29
    3.0: 34.5,  # 30--39
    4.0: 44.5,  # 40--49
    5.0: 54.5,  # 50--59
    6.0: 64.5,  # 60--69
    7.0: 74.5,  # 70--79
    8.0: 84.5,  # 80--89
    9.0: 94.5,  # 90--99
}

# Use the major-subset labels so DC remains distinct from monocytes, as in
# the AIDA level-1 annotation.
CELL_TYPE_MAP = {
    "CD4": "T",
    "CD8": "T",
    "GDT": "T",
    "MAIT": "T",
    "iNKT": "T",
    "DP": "T",
    "DN": "T",
    "cMono": "Myeloid",
    "ncMono": "Myeloid",
    "DC": "DC",
    "NK": "NK",
    "B": "B",
    "PB": "Plasma_Cell",
    "PLT": "Platelet",
    "HSC": "CD34_HSPC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", default=DEFAULT_INPUT)
    parser.add_argument("--aida-h5ad", default=DEFAULT_AIDA)
    parser.add_argument("--out", default="data/covid/combat_covid_case_control_aida2000_mean_imputed.h5ad")
    parser.add_argument("--min-age", type=float, default=19.0)
    parser.add_argument("--max-age", type=float, default=77.0)
    parser.add_argument("--max-cells-per-donor-cell-type", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--target-sum", type=float, default=10000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def age_midpoints(values: np.ndarray) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    output = np.full(len(numeric), np.nan, dtype=float)
    for encoded, midpoint in AGE_MIDPOINTS.items():
        output[np.isclose(numeric, encoded, equal_nan=False)] = midpoint
    return output


def build(args) -> None:
    output = Path(args.out).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")

    aida_means, aida_var, aida_categories = reference_mean(args.aida_h5ad, args.chunk_size, sum_dtype=np.float64)
    aida_genes = aida_var.index.astype(str).to_numpy()
    aida_symbols = aida_var.feature_name.astype(str).to_numpy()
    rng = np.random.default_rng(args.seed)

    with h5py.File(args.input_h5ad, "r") as handle:
        obs = handle["obs"]
        raw_matrix = handle["layers"]["raw"]
        matrix_shape = tuple(int(value) for value in raw_matrix.attrs["shape"])

        var = handle["var"]
        source_features = decode(var[dataframe_index(var)][:]).astype(str)
        feature_types = read_vector(
            var["feature_types"], var.get("__categories")
        ).astype(str)
        gene_positions = np.flatnonzero(
            feature_types == "Gene Expression"
        ).astype(np.int32)
        source_genes = source_features[gene_positions]

        source_lookup = {gene: int(pos) for pos, gene in enumerate(source_genes)}
        shared_aida_pos = np.asarray(
            [
                pos
                for pos, symbol in enumerate(aida_symbols)
                if symbol in source_lookup
            ],
            dtype=np.int32,
        )
        shared_source_gene_pos = np.asarray(
            [
                source_lookup[aida_symbols[pos]]
                for pos in shared_aida_pos
            ],
            dtype=np.int32,
        )
        missing_aida_pos = np.asarray(
            [
                pos
                for pos, symbol in enumerate(aida_symbols)
                if symbol not in source_lookup
            ],
            dtype=np.int32,
        )

        combat_id = read_obs_vector(obs, "COMBAT_ID").astype(str)
        age_code = pd.to_numeric(
            pd.Series(read_obs_vector(obs, "Age")), errors="coerce"
        ).to_numpy(dtype=float)
        age = age_midpoints(age_code)
        sex_code = pd.to_numeric(
            pd.Series(read_obs_vector(obs, "Sex")), errors="coerce"
        ).to_numpy(dtype=float)
        # COMBAT's public CITE-seq metadata uses 0=female and 1=male.
        sex = np.where(sex_code == 0, "female", np.where(sex_code == 1, "male", ""))
        source = read_obs_vector(obs, "Source").astype(str)
        major_subset = read_obs_vector(obs, "Annotation_major_subset").astype(str)
        mapped_type = pd.Series(major_subset).map(CELL_TYPE_MAP)
        severity = np.asarray([SOURCE_TO_SEVERITY.get(value, "") for value in source])

        donor_age_table = pd.DataFrame({"donor": combat_id, "age": age})
        donor_age_nunique = donor_age_table.groupby("donor")["age"].nunique()
        inconsistent_donors = set(
            donor_age_nunique[donor_age_nunique > 1].index.astype(str)
        )
        valid = np.isfinite(age) & (age >= args.min_age) & (age <= args.max_age)
        valid &= np.isin(sex, ["female", "male"])
        valid &= np.isin(source, list(SOURCE_TO_SEVERITY))
        valid &= mapped_type.notna().to_numpy()
        valid &= ~np.isin(combat_id, list(inconsistent_donors))

        selected_by_group: dict[tuple[str, str], list[int]] = defaultdict(list)
        for position in np.flatnonzero(valid):
            selected_by_group[(combat_id[position], str(mapped_type.iloc[position]))].append(
                int(position)
            )
        selected_parts: list[np.ndarray] = []
        for key in sorted(selected_by_group):
            positions = np.asarray(selected_by_group[key], dtype=np.int64)
            if len(positions) > args.max_cells_per_donor_cell_type:
                positions = rng.choice(
                    positions, size=args.max_cells_per_donor_cell_type, replace=False
                )
            selected_parts.append(np.sort(positions))
        selected = np.sort(np.concatenate(selected_parts)).astype(np.int64)

        obs_index = obs[dataframe_index(obs)]
        source_cell_ids = read_vector(obs_index).astype(str)
        cell_ids = source_cell_ids[selected]

        expression_blocks: list[sp.csr_matrix] = []
        count_blocks: list[sp.csr_matrix] = []
        for start in range(0, matrix_shape[0], args.chunk_size):
            stop = min(start + args.chunk_size, matrix_shape[0])
            lo, hi = np.searchsorted(selected, [start, stop])
            if lo == hi:
                continue
            local = selected[lo:hi] - start
            block = read_csr_rows(raw_matrix, start, stop)[local]
            gene_counts = block[:, gene_positions].tocsr()
            totals = np.asarray(gene_counts.sum(axis=1)).reshape(-1).astype(np.float32)
            shared = gene_counts[:, shared_source_gene_pos].tocoo()
            aligned = sp.csr_matrix(
                (shared.data, (shared.row, shared_aida_pos[shared.col])),
                shape=(len(local), len(aida_genes)),
                dtype=np.float32,
            )
            normalized = aligned.multiply(
                (float(args.target_sum) / totals)[:, None]
            ).tocsr()
            np.log1p(normalized.data, out=normalized.data)
            if len(missing_aida_pos):
                normalized = (
                    normalized
                    + constant_columns(
                        len(local),
                        len(aida_genes),
                        missing_aida_pos,
                        aida_means[missing_aida_pos],
                    )
                ).tocsr()
            expression_blocks.append(normalized.astype(np.float32))
            count_blocks.append(aligned)
            print(f"[matrix] {stop:,}/{matrix_shape[0]:,} source rows processed", flush=True)

        expression = sp.vstack(expression_blocks, format="csr", dtype=np.float32)
        counts = sp.vstack(count_blocks, format="csr", dtype=np.float32)

        extra_columns = [
            "scRNASeq_sample_ID",
            "COMBAT_participant_timepoint_ID",
            "Outcome",
            "Symptomatic",
            "TimeSinceOnset",
            "Institute",
            "GEX_region",
        ]
        metadata = pd.DataFrame(index=pd.Index(cell_ids, name="cell_id"))
        metadata["donor_id"] = combat_id[selected]
        metadata["biological_donor_id"] = combat_id[selected]
        metadata["COMBAT_ID"] = combat_id[selected]
        metadata["age_code"] = age_code[selected]
        metadata["age"] = age[selected].astype(np.float32)
        metadata["age_inference"] = "COMBAT decade-bin midpoint"
        metadata["sex"] = sex[selected]
        metadata["Source"] = source[selected]
        metadata["condition"] = np.where(
            severity[selected] == "control", "control", "case"
        )
        metadata["disease_group"] = source[selected]
        metadata["severity_group"] = severity[selected]
        # Keep the labels expected by the existing severity-stratified report.
        metadata["CoVID-19 severity"] = severity[selected]
        metadata["Annotation_major_subset"] = major_subset[selected]
        metadata["Annotation_Level1"] = pd.Categorical(
            mapped_type.iloc[selected].to_numpy(), categories=aida_categories
        )
        for column in extra_columns:
            if column in obs:
                metadata[column] = read_obs_vector(obs, column)[selected].astype(str)

    var_frame = pd.DataFrame(
        {"feature_name": aida_symbols}, index=pd.Index(aida_genes, name="gene_id")
    )
    result = ad.AnnData(X=expression, obs=metadata, var=var_frame)
    result.layers["counts"] = counts
    mark_imputed(result, missing_aida_pos, aida_means)
    result.uns["normalization"] = {"target_sum": args.target_sum, "input_space": "log1p"}
    atomic_write(result, output)
    print(f"[write] {output}: {result.n_obs:,} cells, {result.n_vars:,} genes", flush=True)


def main():
    build(parse_args())


if __name__ == "__main__":
    main()
