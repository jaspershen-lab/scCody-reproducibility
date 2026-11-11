"""Streaming H5AD writer shared by AIDA and COVID raw-count reconstruction.

The command-line entry point uses the canonical prepare_aida workflow.
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem, write_elem

from .h5ad_metadata import dataframe_index, make_frames_writable, read_csr_rows as block


def index(group):
    return pd.Index(group[dataframe_index(group)].asstr()[:])


def normalize_before_subset(counts, positions):
    totals = np.asarray(counts.sum(axis=1), dtype=np.float64).ravel()
    expression = counts.astype(np.float32, copy=True)
    expression.data *= np.repeat((10000.0 / totals).astype(np.float32), np.diff(expression.indptr))
    np.log1p(expression.data, out=expression.data)
    return expression[:, positions].tocsr(), counts[:, positions].tocsr(), totals


class Writer:
    def __init__(self, path, template, var, settings, *, obs=None):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.temporary = self.path.with_name("." + self.path.name + ".building")
        self.f = h5py.File(self.temporary, "x")
        with h5py.File(template) as src:
            self.f.attrs.update(src.attrs)
            if obs is None:
                obs = read_elem(src["obs"])
        obs, var = obs.copy(), var.copy()
        make_frames_writable(obs, var)
        self.n_obs = len(obs)
        write_elem(self.f, "obs", obs)
        write_elem(self.f, "var", var)
        write_elem(self.f, "uns", {"aida_preprocess": settings})
        for name in ["obsm", "obsp", "varm", "varp", "layers"]:
            write_elem(self.f, name, {})
        key = "full_gene_library_size"
        if key in self.f["obs"]:
            del self.f["obs"][key]
        columns = list(self.f["obs"].attrs["column-order"])
        if key not in columns:
            columns.append(key)
        self.f["obs"].attrs["column-order"] = np.asarray(columns, dtype=h5py.string_dtype())
        self.totals = self.f["obs"].create_dataset(key, shape=(self.n_obs,), dtype="float64")
        self.totals.attrs.update({"encoding-type": "array", "encoding-version": "0.2.0"})
        self.row, self.nnz, self.groups = 0, [0, 0], []
        for name in ["X", "layers/counts"]:
            group = self.f.create_group(name)
            group.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0",
                                "shape": np.asarray([self.n_obs, len(var)], dtype=np.int64)})
            for key, dtype in [("data", "float32"), ("indices", "int32")]:
                group.create_dataset(key, shape=(0,), maxshape=(None,), dtype=dtype,
                                     chunks=(262144,), compression="gzip", compression_opts=1)
            group.create_dataset("indptr", shape=(self.n_obs + 1,), dtype="int64")
            self.groups.append(group)

    def append(self, expression, counts, totals):
        for i, (group, matrix) in enumerate(zip(self.groups, [expression, counts])):
            for name, value in [("data", matrix.data), ("indices", matrix.indices)]:
                dataset = group[name]
                dataset.resize((self.nnz[i] + len(value),))
                dataset[self.nnz[i]:] = value
            group["indptr"][self.row:self.row + matrix.shape[0] + 1] = matrix.indptr + self.nnz[i]
            self.nnz[i] += matrix.nnz
        self.totals[self.row:self.row + len(totals)] = totals
        self.row += counts.shape[0]

    def finish(self):
        self.f.close()


def main():
    from .prepare_aida import main as prepare
    prepare()


if __name__ == "__main__":
    main()
