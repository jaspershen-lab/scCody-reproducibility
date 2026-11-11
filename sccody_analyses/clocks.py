"""Donor pseudobulk preprocessing for the supervised age clocks (Lasso, ElasticNet, tAge)."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.stats import pearsonr, spearmanr
TRAIN_MANIFEST_TEMPLATE = None
EXTERNAL_MANIFEST_TEMPLATE = None
YOUNG_REF_AGE_MAX = 30.0
MIN_REF_METACELLS = 5
GENE_DETECT_FRAC = 0.20

def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)

def donor_age_map(obs: pd.DataFrame, label: str) -> pd.Series:
    donor_age = (
        obs[["donor_id", "age"]]
        .assign(
            donor_id=lambda frame: frame["donor_id"].astype(str),
            age=lambda frame: pd.to_numeric(frame["age"], errors="coerce"),
        )
        .dropna(subset=["age"])
        .drop_duplicates()
    )
    return donor_age.drop_duplicates("donor_id").set_index("donor_id")["age"]

def resolve_fold_donors(
    fold: int,
    train_donor_age: pd.Series,
    external_donor_age: pd.Series,
) -> tuple[list[str], list[str]]:
    internal = read_json(Path(str(TRAIN_MANIFEST_TEMPLATE).format(fold=fold)))
    external = read_json(Path(str(EXTERNAL_MANIFEST_TEMPLATE).format(fold=fold)))
    excluded = internal["holdout_ages"] + external["external_report_ages"]
    train = train_donor_age.loc[~train_donor_age.isin(excluded)]
    test = external_donor_age.loc[external_donor_age.isin(external["external_report_ages"])]
    return sorted(train.index.astype(str)), sorted(test.index.astype(str))


def load_obs_counts(
    path: Path, *, allow_x_fallback: bool = False
) -> tuple[pd.DataFrame, sp.csr_matrix, np.ndarray]:
    adata = ad.read_h5ad(path)
    obs = adata.obs[
        ["donor_id", "age", "Annotation_Level1"]
    ].copy()
    obs["donor_id"] = obs["donor_id"].astype(str)
    obs["age"] = pd.to_numeric(obs["age"], errors="coerce").astype(np.float64)
    obs["Annotation_Level1"] = obs["Annotation_Level1"].astype(str)
    if "counts" in adata.layers:
        counts = adata.layers["counts"]
    elif allow_x_fallback:
        # Some external processed cohorts expose only normalized/log-scale X.
        # Keep this opt-in because the frozen AIDA clock was trained with
        # count-layer input and this fallback is exploratory only.
        counts = adata.X
    else:
        raise KeyError(
            f"{path} has no layers['counts']; pass allow_x_fallback=True only "
            "when deliberately applying the count-based clock to X"
        )
    counts = counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(counts)
    genes = adata.var_names.astype(str).to_numpy()
    return obs, counts, genes

def align_gene_matrices(
    train_counts: sp.csr_matrix,
    train_genes: np.ndarray,
    external_counts: sp.csr_matrix,
    external_genes: np.ndarray,
) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    if np.array_equal(train_genes, external_genes):
        return train_counts, external_counts, train_genes

    external_index = pd.Index(external_genes)
    shared = [gene for gene in train_genes if gene in external_index]
    if not shared:
        raise ValueError("training and external H5ADs have no genes in common")
    train_index = pd.Index(train_genes)
    train_positions = train_index.get_indexer(shared)
    external_positions = external_index.get_indexer(shared)
    return (
        train_counts[:, train_positions].tocsr(),
        external_counts[:, external_positions].tocsr(),
        np.asarray(shared),
    )

def build_metacells(
    obs: pd.DataFrame,
    counts: sp.csr_matrix,
    donor_ids: list[str],
    mode: str,
    min_cells: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    selected = obs["donor_id"].isin(set(donor_ids)).to_numpy()
    positions = np.flatnonzero(selected)
    subset = obs.iloc[positions].copy()
    subset["_position"] = positions
    if mode == "percelltype":
        subset["_stratum"] = subset["Annotation_Level1"]
        subset["_group"] = (
            subset["donor_id"] + "||" + subset["Annotation_Level1"]
        )
    else:
        subset["_stratum"] = "ALL"
        subset["_group"] = subset["donor_id"]

    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for group_key, group in subset.groupby("_group", observed=True, sort=False):
        group_positions = group["_position"].to_numpy()
        n_cells = int(len(group_positions))
        donor_id = str(group["donor_id"].iloc[0])
        stratum = str(group["_stratum"].iloc[0])
        age = float(group["age"].median())
        row = {
            "metacell_id": str(group_key),
            "donor_id": donor_id,
            "cell_type": stratum,
            "age": age,
            "n_cells": n_cells,
        }
        if n_cells < min_cells:
            continue
        vector = np.asarray(counts[group_positions].sum(axis=0)).ravel()
        vector = vector.astype(np.float64)
        rows.append({**row, "total_counts": float(vector.sum())})
        vectors.append(vector)

    metadata = pd.DataFrame(rows).reset_index(drop=True)
    matrix = (
        np.vstack(vectors)
        if vectors
        else np.empty((0, counts.shape[1]), dtype=np.float64)
    )
    return metadata, matrix

def filter_genes(
    train_counts: np.ndarray,
    gene_names: np.ndarray,
    min_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    detection_fraction = (train_counts > 0).mean(axis=0)
    keep = detection_fraction >= min_fraction
    if not keep.any():
        raise ValueError("no genes passed the detection-frequency filter")
    return gene_names[keep], keep

def fit_rle(train_counts: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        log_counts = np.where(train_counts > 0, np.log(train_counts), np.nan)
        gene_log_mean = np.nanmean(log_counts, axis=0)
    gene_geomean = np.exp(gene_log_mean)
    gene_geomean[~np.isfinite(gene_geomean)] = 0.0
    return gene_geomean

def apply_rle(
    counts: np.ndarray,
    gene_geomean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid_genes = gene_geomean > 0
    size_factors = np.ones(counts.shape[0], dtype=np.float64)
    for row_index, row in enumerate(counts):
        usable = valid_genes & (row > 0)
        if usable.any():
            size_factors[row_index] = np.median(
                row[usable] / gene_geomean[usable]
            )
    size_factors[~np.isfinite(size_factors) | (size_factors <= 0)] = 1.0
    return counts / size_factors[:, None], size_factors

def fit_scaler(
    train_log: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_log.mean(axis=0)
    standard_deviation = train_log.std(axis=0, ddof=0)
    zero_variance = standard_deviation <= 1e-12
    safe_standard_deviation = np.where(
        zero_variance, 1.0, standard_deviation
    )
    return mean, safe_standard_deviation, zero_variance

def apply_scaler(
    log_matrix: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    zero_variance: np.ndarray,
) -> np.ndarray:
    scaled = (log_matrix - mean) / standard_deviation
    scaled[:, zero_variance] = 0.0
    return scaled

def fit_reference(
    train_metadata: pd.DataFrame,
    train_scaled: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    expression_medians: dict[str, np.ndarray] = {}
    age_medians: dict[str, float] = {}
    young = train_metadata["age"].to_numpy() <= YOUNG_REF_AGE_MAX

    for stratum in sorted(train_metadata["cell_type"].unique()):
        in_stratum = (train_metadata["cell_type"] == stratum).to_numpy()
        reference_mask = in_stratum & young
        if int(reference_mask.sum()) < MIN_REF_METACELLS:
            reference_mask = young.copy()
        n_reference = int(reference_mask.sum())
        if n_reference:
            expression_medians[stratum] = np.median(
                train_scaled[reference_mask], axis=0
            )
            age_medians[stratum] = float(
                np.median(train_metadata.loc[reference_mask, "age"])
            )
        else:
            expression_medians[stratum] = np.zeros(train_scaled.shape[1])
            age_medians[stratum] = float(train_metadata["age"].median())
    return expression_medians, age_medians

def relative_transform(
    metadata: pd.DataFrame,
    scaled: np.ndarray,
    expression_medians: dict[str, np.ndarray],
    age_medians: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    relative_expression = np.empty_like(scaled)
    relative_age = np.empty(len(metadata), dtype=np.float64)
    ages = metadata["age"].to_numpy()
    for row, stratum in enumerate(metadata["cell_type"].to_numpy()):
        relative_expression[row] = scaled[row] - expression_medians[stratum]
        relative_age[row] = ages[row] - age_medians[stratum]
    return relative_expression, relative_age

def evaluate(true_age: Any, predicted_age: Any) -> dict[str, Any]:
    truth = np.asarray(true_age, dtype=np.float64)
    prediction = np.asarray(predicted_age, dtype=np.float64)
    n_observations = len(truth)
    if n_observations < 2 or np.std(prediction) == 0:
        pearson = np.nan
        spearman = np.nan
        slope = np.nan
        intercept = np.nan
    else:
        pearson = float(pearsonr(truth, prediction)[0])
        spearman = float(spearmanr(truth, prediction)[0])
        slope, intercept = np.polyfit(prediction, truth, 1)
    residual_sum = float(np.sum((truth - prediction) ** 2))
    total_sum = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "n": n_observations,
        "mae": float(np.mean(np.abs(truth - prediction))),
        "rmse": float(np.sqrt(np.mean((truth - prediction) ** 2))),
        "r2": float(1.0 - residual_sum / total_sum) if total_sum > 0 else np.nan,
        "pearson_r": pearson,
        "spearman_r": spearman,
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
    }

def aggregate_donor_predictions(
    metacell_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for donor_id, group in metacell_predictions.groupby(
        "donor_id", observed=True, sort=False
    ):
        rows.append(
            {
                "donor_id": donor_id,
                "age": float(group["age"].iloc[0]),
                "pred_age": float(
                    np.average(group["pred_age"], weights=group["n_cells"])
                ),
                "n_metacells": int(len(group)),
                "total_cells": int(group["n_cells"].sum()),
            }
        )
    return pd.DataFrame(rows)
