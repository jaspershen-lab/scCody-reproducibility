"""Fixed-penalty Lasso/ElasticNet age clocks, fit globally or per lineage."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
import sccody_analyses.clocks as clock

def prepare_split(
    *,
    fold: int,
    mode: str,
    min_cells: int,
    min_eval_cells: int | None = None,
    eval_gene_observed: np.ndarray | None = None,
    train_obs: pd.DataFrame,
    train_counts: Any,
    eval_obs: pd.DataFrame,
    eval_counts: Any,
    gene_names: np.ndarray,
    train_donors: list[str],
    eval_donors: list[str],
    eval_label: str,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    train_metadata, train_raw = clock.build_metacells(
        train_obs,
        train_counts,
        train_donors,
        mode,
        min_cells,
    )
    eval_metadata, eval_raw = clock.build_metacells(
        eval_obs,
        eval_counts,
        eval_donors,
        mode,
        min_cells if min_eval_cells is None else min_eval_cells,
    )
    if train_metadata.empty or eval_metadata.empty:
        raise RuntimeError(
            f"fold {fold} {eval_label}: no metacells survived "
            f"min_cells={min_cells}"
        )

    train_strata = set(train_metadata["cell_type"].unique())
    has_reference = eval_metadata["cell_type"].isin(train_strata).to_numpy()
    if not has_reference.all():
        eval_metadata = eval_metadata.loc[has_reference].reset_index(drop=True)
        eval_raw = eval_raw[has_reference]
    if eval_metadata.empty:
        raise RuntimeError(
            f"fold {fold} {eval_label}: no evaluation metacells have "
            "training references"
        )

    retained_genes, gene_mask = clock.filter_genes(
        train_raw, gene_names, clock.GENE_DETECT_FRAC
    )
    train_gene_counts = train_raw[:, gene_mask]
    eval_gene_counts = eval_raw[:, gene_mask].astype(float)
    if eval_gene_observed is not None:
        eval_gene_counts[:, ~eval_gene_observed[gene_mask]] = np.nan
    gene_geomean = clock.fit_rle(train_gene_counts)
    train_rle, _ = clock.apply_rle(
        train_gene_counts, gene_geomean
    )
    eval_rle, _ = clock.apply_rle(
        eval_gene_counts, gene_geomean
    )
    train_log = np.log2(train_rle + 1.0)
    eval_log = np.log2(eval_rle + 1.0)
    mean, standard_deviation, zero_variance = clock.fit_scaler(train_log)
    train_scaled = clock.apply_scaler(
        train_log, mean, standard_deviation, zero_variance
    )
    eval_scaled = clock.apply_scaler(
        eval_log, mean, standard_deviation, zero_variance
    )
    reference_expression, reference_ages = clock.fit_reference(train_metadata, train_scaled)
    train_relative, train_relative_age = clock.relative_transform(
        train_metadata,
        train_scaled,
        reference_expression,
        reference_ages,
    )
    eval_relative, eval_relative_age = clock.relative_transform(
        eval_metadata,
        eval_scaled,
        reference_expression,
        reference_ages,
    )
    train_gene_median = np.nanmedian(train_relative, axis=0)
    train_relative = np.where(
        np.isnan(train_relative), train_gene_median, train_relative
    )
    eval_relative = np.where(
        np.isnan(eval_relative), train_gene_median, eval_relative
    )

    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        strata = sorted(reference_ages)
        np.savez_compressed(artifact_dir / "preprocessing.npz", genes=retained_genes,
            rle_gene_geomean=gene_geomean, mean=mean, sd=standard_deviation,
            zero_variance=zero_variance, impute_median=train_gene_median,
            cell_types=np.asarray(strata), reference_ages=np.asarray([reference_ages[s] for s in strata]),
            reference_expression=np.stack([reference_expression[s] for s in strata]))

    return {
        "fold": fold,
        "train_metadata": train_metadata,
        "eval_metadata": eval_metadata,
        "train_x": train_relative,
        "train_y": train_relative_age,
        "eval_x": eval_relative,
        "eval_y": eval_relative_age,
        "reference_ages": reference_ages,
        "retained_genes": retained_genes,
    }

def predictions_from_model(
    model: ElasticNet,
    prepared: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    eval_metadata = prepared["eval_metadata"].copy()
    predicted_relative = model.predict(prepared["eval_x"])
    reference_age = np.asarray(
        [
            prepared["reference_ages"][stratum]
            for stratum in eval_metadata["cell_type"].to_numpy()
        ]
    )
    eval_metadata["fold"] = prepared["fold"]
    eval_metadata["pred_relative_age"] = predicted_relative
    eval_metadata["true_relative_age"] = prepared["eval_y"]
    eval_metadata["pred_age"] = predicted_relative + reference_age
    eval_metadata["error"] = eval_metadata["pred_age"] - eval_metadata["age"]
    donor_predictions = clock.aggregate_donor_predictions(eval_metadata)
    donor_predictions["fold"] = prepared["fold"]
    donor_predictions["error"] = (
        donor_predictions["pred_age"] - donor_predictions["age"]
    )
    metrics = clock.evaluate(
        donor_predictions["age"], donor_predictions["pred_age"]
    )
    return eval_metadata, donor_predictions, metrics

def fit_selected_model(
    prepared: dict[str, Any],
    selected: dict[str, float],
    max_iter: int,
    seed: int,
) -> ElasticNet:
    model = ElasticNet(
        alpha=selected["alpha"],
        l1_ratio=selected["l1_ratio"],
        max_iter=max_iter,
        random_state=seed + int(prepared["fold"]),
        selection="cyclic",
    )
    model.fit(prepared["train_x"], prepared["train_y"])
    if model.n_iter_ >= max_iter:
        raise RuntimeError(
            f"fold {prepared['fold']}: selected Elastic Net did not converge "
            f"within {max_iter} iterations"
        )
    return model

def cell_type_view(
    prepared: dict[str, Any],
    cell_type: str,
    split: str,
) -> dict[str, Any] | None:
    metadata = prepared[f"{split}_metadata"]
    mask = metadata["cell_type"].astype(str).to_numpy() == cell_type
    if not mask.any():
        return None
    return {
        "metadata": metadata.loc[mask].reset_index(drop=True),
        "x": prepared[f"{split}_x"][mask],
        "y": prepared[f"{split}_y"][mask],
        "reference_age": float(prepared["reference_ages"][cell_type]),
    }

def fit_model(
    prepared: dict[str, Any],
    cell_type: str,
    selected: dict[str, float],
    max_iter: int,
    seed: int,
) -> tuple[ElasticNet, dict[str, Any]]:
    train = cell_type_view(prepared, cell_type, "train")
    if train is None:
        raise RuntimeError(f"fold {prepared['fold']} {cell_type}: no training pseudobulk")
    model = ElasticNet(
        alpha=selected["alpha"],
        l1_ratio=selected["l1_ratio"],
        max_iter=max_iter,
        random_state=seed + int(prepared["fold"]),
        selection="cyclic",
    )
    model.fit(train["x"], train["y"])
    if model.n_iter_ >= max_iter:
        raise RuntimeError(
            f"fold {prepared['fold']} {cell_type}: selected model did not converge"
        )
    return model, train

def predict_external_type(
    prepared: dict[str, Any],
    cell_type: str,
    model: ElasticNet,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    test = cell_type_view(prepared, cell_type, "eval")
    if test is None:
        return pd.DataFrame(), {
            "cell_type": cell_type,
            "fold": int(prepared["fold"]),
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "pearson_r": np.nan,
            "spearman_r": np.nan,
            "calibration_slope": np.nan,
            "calibration_intercept": np.nan,
        }
    predicted_age = model.predict(test["x"]) + test["reference_age"]
    frame = test["metadata"].copy()
    frame["fold"] = int(prepared["fold"])
    frame["pred_relative_age"] = model.predict(test["x"])
    frame["pred_age"] = predicted_age
    frame["error"] = predicted_age - frame["age"]
    metrics = clock.evaluate(frame["age"], frame["pred_age"])
    metrics.update({"cell_type": cell_type, "fold": int(prepared["fold"])})
    return frame, metrics
