from __future__ import annotations
from collections.abc import Callable, Sequence
import numpy as np
import torch
from sccody_analyses.benchmarks.engine.optim.loss_func import SinkhornLoss

def sinkhorn_distance(true_data: np.ndarray, pred_data: np.ndarray, *, max_cells: int=1000, blur: float=0.05, scaling: float=0.5) -> float:
    """Compute a Sinkhorn distance between decoded expression matrices."""
    true = np.asarray(true_data, dtype=np.float32)
    pred = np.asarray(pred_data, dtype=np.float32)
    if true.ndim != 2 or pred.ndim != 2:
        raise ValueError('sinkhorn_distance expects two (cells, genes) matrices')
    if true.shape[1] != pred.shape[1]:
        raise ValueError('sinkhorn_distance expression matrices have different gene counts')
    true = true[np.isfinite(true).all(axis=1)]
    pred = pred[np.isfinite(pred).all(axis=1)]
    n = min(len(true), len(pred))
    if n < 2:
        return float('nan')
    true, pred = (true[:n], pred[:n])
    if int(max_cells) > 0 and n > int(max_cells):
        selected = np.linspace(0, n - 1, num=int(max_cells), dtype=np.int64)
        true, pred = (true[selected], pred[selected])
    with torch.no_grad():
        value = SinkhornLoss([torch.from_numpy(true)], torch.from_numpy(pred).unsqueeze(1), blur=float(blur), scaling=float(scaling), batch_size=None)
    return float(value.detach().cpu())

def summarize_projection_age_scores(ages: Sequence[float], target_age: float, *, sinkhorn_scores: Sequence[float], rng: np.random.Generator, minimum_rtol: float=1e-06, minimum_atol: float=1e-08) -> dict[str, object]:
    """Select best and chronological ages from precomputed curve scores.

    Candidate ages whose score is numerically tied with the global minimum
    are sampled uniformly.  This avoids adding an implicit chronological
    preference when the age curve has a flat minimum.
    """
    ages = [float(age) for age in ages]
    if not ages:
        raise ValueError('Age scores must be non-empty')
    chronological = int(np.argmin(np.abs(np.asarray(ages) - float(target_age))))
    scores_array = np.asarray(sinkhorn_scores, dtype=np.float64)
    if scores_array.shape != (len(ages),):
        raise ValueError('Age scores must align one-to-one with ages')
    minimum = float(np.min(scores_array))
    minimum_indices = np.flatnonzero(np.isclose(scores_array, minimum, rtol=float(minimum_rtol), atol=float(minimum_atol)))
    best_sinkhorn = int(rng.choice(minimum_indices))
    sinkhorn_best = float(scores_array[best_sinkhorn])
    sinkhorn_chrono = float(scores_array[chronological])
    return {'best_sinkhorn_index': best_sinkhorn, 'chronological_index': chronological, 't_hat_sinkhorn': ages[best_sinkhorn], 'sinkhorn_at_t_hat': sinkhorn_best, 'sinkhorn_at_chrono': sinkhorn_chrono, 'sinkhorn_improvement': sinkhorn_chrono - sinkhorn_best, 'n_minimum_age_candidates': int(len(minimum_indices))}

def score_projection_age_grid(predictions: Sequence[object], ages: Sequence[float], target_age: float, *, sinkhorn_score: Callable[[object], float], rng: np.random.Generator) -> dict[str, object]:
    """Score predictions, select best ages, and retain chronological comparisons."""
    if not predictions or len(predictions) != len(ages):
        raise ValueError('Predictions and ages must be non-empty and aligned')
    sinkhorn_scores = [float(sinkhorn_score(prediction)) for prediction in predictions]
    return summarize_projection_age_scores(ages, target_age, sinkhorn_scores=sinkhorn_scores, rng=rng)
