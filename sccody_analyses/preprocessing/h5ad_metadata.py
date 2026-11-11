"""Shared H5AD I/O and assay alignment for study preprocessing."""
import os
from pathlib import Path

import anndata as ad
import h5py
from anndata.io import read_elem
import numpy as np
import pandas as pd
from scipy import sparse


def make_frames_writable(*frames):
    for frame in frames:
        frame.index = pd.Index(np.asarray(frame.index, dtype=object), dtype=object)
        for column in frame.columns:
            series = frame[column]
            if isinstance(series.dtype, pd.CategoricalDtype):
                categories = series.cat.categories
                if isinstance(categories.dtype, pd.StringDtype) or "ArrowStringArray" in type(
                    categories.array
                ).__name__:
                    frame[column] = pd.Categorical.from_codes(
                        series.cat.codes.to_numpy(),
                        categories=pd.Index(
                            np.asarray(categories, dtype=object), dtype=object
                        ),
                        ordered=series.cat.ordered,
                    )
            elif isinstance(series.dtype, pd.StringDtype) or "ArrowStringArray" in type(
                series.array
            ).__name__:
                frame[column] = pd.Series(
                    np.asarray(series, dtype=object), index=frame.index, dtype=object
                )


def make_h5ad_frames_writable(adata):
    make_frames_writable(adata.obs, adata.var)
    if adata.raw is not None:
        make_frames_writable(adata.raw.var)


def atomic_write(adata, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        with pd.option_context("future.infer_string", False):
            make_h5ad_frames_writable(adata)
            adata.write_h5ad(temporary, compression="gzip")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def decode(values):
    values = np.asarray(values)
    if values.dtype.kind in {"S", "O", "U"}:
        return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in values], dtype=object)
    return values


def dataframe_index(group):
    key = group.attrs.get("_index", "_index")
    return key.decode() if isinstance(key, bytes) else str(key)


def read_vector(node, categories=None):
    if isinstance(node, h5py.Group):
        codes, values = node["codes"][:], decode(read_elem(node["categories"]))
    elif categories is not None and node.name.rsplit("/", 1)[-1] in categories:
        codes, values = node[:], decode(read_elem(categories[node.name.rsplit("/", 1)[-1]]))
    else:
        return decode(node[:])
    result = np.full(len(codes), None, dtype=object)
    valid = codes >= 0
    result[valid] = values[codes[valid]]
    return result


def read_obs_vector(obs, field):
    return read_vector(obs[field], obs.get("__categories"))


def read_csr_rows(matrix, start, stop):
    pointers = np.asarray(matrix["indptr"][start:stop + 1], dtype=np.int64)
    lo, hi = int(pointers[0]), int(pointers[-1])
    return sparse.csr_matrix((matrix["data"][lo:hi], matrix["indices"][lo:hi], pointers - lo),
                             shape=(stop - start, int(matrix.attrs["shape"][1])))


def reference_mean(path, block_rows, *, sum_dtype=None):
    reference = ad.read_h5ad(path, backed="r")
    try:
        sums = np.zeros(reference.n_vars, dtype=np.float64)
        for start in range(0, reference.n_obs, block_rows):
            values = reference.X[start:start + block_rows]
            if sum_dtype is not None:
                values = values.astype(sum_dtype)
            sums += (np.asarray(values.sum(axis=0)).ravel() if sparse.issparse(values)
                     else np.asarray(values, dtype=np.float64).sum(axis=0))
        series = reference.obs["Annotation_Level1"]
        categories = (series.cat.categories.astype(str).tolist() if isinstance(series.dtype, pd.CategoricalDtype)
                      else pd.unique(series.astype(str)).tolist())
        return (sums / reference.n_obs).astype(np.float32), reference.var.copy(), categories
    finally:
        reference.file.close()


def constant_columns(n_rows, n_columns, columns, values):
    columns, values = np.asarray(columns, dtype=np.int32), np.asarray(values, dtype=np.float32)
    keep = values != 0
    columns, values = columns[keep], values[keep]
    return sparse.csr_matrix((np.tile(values, n_rows), np.tile(columns, n_rows),
                              np.arange(n_rows + 1, dtype=np.int64) * len(columns)),
                             shape=(n_rows, n_columns))


def add_reference_means(observed, missing, means):
    return (observed.astype(np.float32, copy=False)
            + constant_columns(*observed.shape, missing, means[missing])).tocsr()


def mark_imputed(adata, missing, means):
    observed = np.ones(adata.n_vars, dtype=bool)
    observed[missing] = False
    adata.var["assay_observed"] = observed
    adata.var["counts_measured"] = observed
    adata.uns["reference_mean_imputation"] = {
        "missing_features": np.asarray(adata.var_names[missing], dtype=str),
        "missing_feature_reference_means": means[missing],
    }
