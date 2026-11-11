"""Post-rollout PCA-KNN cell-type age-localization readout."""
from __future__ import annotations
from typing import Any, Iterable
import numpy as np
import torch
from sccody_analyses.benchmarks.engine.optim.loss_func import SinkhornLoss
from sccody_analyses.benchmarks.engine.evaluation.metrics import fit_pca_knn_cell_type, predict_pca_knn_cell_type
from sccody_analyses.benchmarks.engine.projection.metrics import summarize_projection_age_scores

def _sample_indices(indices: np.ndarray, maximum: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if maximum > 0 and len(indices) > maximum:
        return np.linspace(0, len(indices) - 1, num=maximum, dtype=np.int64)
    return indices

def fit_train_only_pca_knn_classifier(adata_train: Any, args: Any) -> tuple[dict[str, Any], list[str]]:
    """Fit the existing PCA-KNN readout on the post-split training cells only."""
    from sccody_analyses.benchmarks.engine.methods.dynamics import get_expr
    codes = adata_train.obs['_ct_code'].to_numpy(dtype=np.int64)
    valid = np.flatnonzero(codes >= 0)
    maximum = int(getattr(args, 'max_eval_ref_cells', 0))
    if maximum > 0 and len(valid) > maximum:
        rng = np.random.default_rng(int(args.seed) + 2400017)
        groups = [valid[codes[valid] == code] for code in sorted(np.unique(codes[valid]))]
        per_group = max(1, maximum // max(1, len(groups)))
        picked = [group if len(group) <= per_group else rng.choice(group, per_group, replace=False) for group in groups]
        valid = np.concatenate(picked) if picked else valid[:0]
        if len(valid) > maximum:
            valid = rng.choice(valid, maximum, replace=False)
    expression = get_expr(adata_train, valid, args)
    classifier = fit_pca_knn_cell_type(expression, codes[valid])
    categories = [str(value) for value in adata_train.uns.get('_ct_categories', [])]
    return (classifier, categories)

def select_target_indices_for_pca_knn(target_cell_types: np.ndarray | None, *, n: int, rng: np.random.Generator, categories: Iterable[str], requested_cell_types: Iterable[str] | None) -> np.ndarray | None:
    """Keep the global source pool but retain requested target lineages."""
    if target_cell_types is None or requested_cell_types is None:
        return None
    categories = [str(value) for value in categories]
    requested = {str(value) for value in requested_cell_types}
    requested_codes = {code for code, name in enumerate(categories) if name in requested}
    candidates = np.flatnonzero(np.isin(target_cell_types, list(requested_codes)))
    if len(candidates) < 2:
        return candidates
    count = min(int(n), len(candidates))
    return rng.choice(candidates, size=count, replace=False)

def _sinkhorn_latent(target: np.ndarray, predicted: np.ndarray, *, blur: float, scaling: float) -> float:
    with torch.no_grad():
        value = SinkhornLoss([torch.as_tensor(target, dtype=torch.float32)], torch.as_tensor(predicted, dtype=torch.float32).unsqueeze(1), blur=float(blur), scaling=float(scaling), batch_size=None)
    return float(value.detach().cpu())

def compute_pca_knn_cell_type_age_localization(*, predicted_expression_by_age: np.ndarray, predicted_latent_by_age: np.ndarray, target_expression: np.ndarray, target_latent: np.ndarray, target_cell_types: np.ndarray, candidate_ages: Iterable[float], target_age: float, classifier: dict[str, Any], cell_type_categories: Iterable[str], requested_cell_types: Iterable[str] | None, min_cells: int, max_sinkhorn_cells: int, blur: float, scaling: float, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Annotate global generated cells at each age, then score each lineage."""
    pred_expr = np.asarray(predicted_expression_by_age, dtype=np.float32)
    pred_latent = np.asarray(predicted_latent_by_age, dtype=np.float32)
    target_expr = np.asarray(target_expression, dtype=np.float32)
    target_latent = np.asarray(target_latent, dtype=np.float32)
    target_codes = np.asarray(target_cell_types, dtype=np.int64)
    ages = [float(age) for age in candidate_ages]
    if pred_expr.ndim != 3 or pred_latent.ndim != 3:
        raise ValueError('PCA-KNN age readout expects (cells, ages, features) arrays')
    if pred_expr.shape[:2] != pred_latent.shape[:2] or pred_expr.shape[1] != len(ages):
        raise ValueError('PCA-KNN predictions and age grid are not aligned')
    if len(target_expr) != len(target_latent) or len(target_expr) != len(target_codes):
        raise ValueError('PCA-KNN target arrays are not aligned')
    categories = [str(value) for value in cell_type_categories]
    requested = None if requested_cell_types is None else {str(value) for value in requested_cell_types}
    code_by_name = {name: code for code, name in enumerate(categories)}
    if requested is None:
        codes = sorted((int(code) for code in np.unique(target_codes)))
    else:
        codes = sorted((code_by_name[name] for name in requested if name in code_by_name))
    predicted_codes_by_age = [np.asarray(predict_pca_knn_cell_type(pred_expr[:, age_index, :], classifier)['predicted'], dtype=np.int64) for age_index in range(len(ages))]
    rows: list[dict[str, Any]] = []
    for code in codes:
        name = categories[code] if 0 <= code < len(categories) else str(code)
        minimum = 5 if name == 'DC' else int(min_cells)
        target_idx = np.flatnonzero(target_codes == int(code))
        if len(target_idx) < minimum:
            continue
        target_pick = _sample_indices(target_idx, int(max_sinkhorn_cells))
        target_pool = target_latent[target_pick]
        scores: list[float] = []
        valid_ages: list[float] = []
        n_pred_by_age: list[int] = []
        n_sinkhorn_by_age: list[int] = []
        for age_index, age in enumerate(ages):
            pred_idx = np.flatnonzero(predicted_codes_by_age[age_index] == int(code))
            if len(pred_idx) < minimum:
                continue
            pred_pick = _sample_indices(pred_idx, int(max_sinkhorn_cells))
            n = min(len(target_pool), len(pred_pick))
            if n < 2:
                continue
            scores.append(_sinkhorn_latent(target_pool[:n], pred_latent[pred_pick[:n], age_index, :], blur=blur, scaling=scaling))
            valid_ages.append(age)
            n_pred_by_age.append(int(len(pred_idx)))
            n_sinkhorn_by_age.append(int(n))
        if not scores:
            continue
        curve = summarize_projection_age_scores(valid_ages, float(target_age), sinkhorn_scores=scores, rng=rng)
        best_index = int(curve['best_sinkhorn_index'])
        rows.append({'cell_type': str(name), 'cell_type_code': int(code), 'ct_assignment': 'pca_knn', 't_hat_sinkhorn': float(curve['t_hat_sinkhorn']), 'delta_t_sinkhorn_raw': float(curve['t_hat_sinkhorn'] - float(target_age)), 'age_prediction_abs_error_sinkhorn': abs(float(curve['t_hat_sinkhorn']) - float(target_age)), 'sinkhorn_at_t_hat': float(curve['sinkhorn_at_t_hat']), 'sinkhorn_at_chrono': float(curve['sinkhorn_at_chrono']), 'sinkhorn_improvement': float(curve['sinkhorn_improvement']), 'n_target_cells': int(len(target_idx)), 'n_pred_cells_at_t_hat': int(n_pred_by_age[best_index]), 'n_sinkhorn_cells': int(n_sinkhorn_by_age[best_index]), 'n_age_grid_points_with_cell_type': int(len(valid_ages)), 'age_minimum_selection': 'random_within_minimum', 'n_minimum_age_candidates': int(curve['n_minimum_age_candidates'])})
    return rows
