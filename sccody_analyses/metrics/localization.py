"""Numerical prediction accuracy, calibration and population bootstrap."""
import numpy as np
from scipy.stats import rankdata


def _measure(actual, predicted):
    difference = predicted - actual
    centered = actual - actual.mean(axis=1, keepdims=True)
    predicted_centered = predicted - predicted.mean(axis=1, keepdims=True)
    total = np.square(centered).sum(axis=1)
    predicted_total = np.square(predicted_centered).sum(axis=1)
    covariance = (centered * predicted_centered).sum(axis=1)
    def divide(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)
    actual_rank, predicted_rank = rankdata(actual, axis=1), rankdata(predicted, axis=1)
    actual_rank -= actual_rank.mean(axis=1, keepdims=True)
    predicted_rank -= predicted_rank.mean(axis=1, keepdims=True)
    slope = divide(covariance, total)
    return {
        "mae": np.abs(difference).mean(axis=1),
        "rmse": np.sqrt(np.square(difference).mean(axis=1)),
        "bias": difference.mean(axis=1),
        "pearson": divide(covariance, np.sqrt(total * predicted_total)),
        "spearman": divide((actual_rank * predicted_rank).sum(axis=1), np.sqrt(np.square(actual_rank).sum(axis=1) * np.square(predicted_rank).sum(axis=1))),
        "prediction_r2": 1 - divide(np.square(difference).sum(axis=1), total),
        "calibration_slope": slope,
        "calibration_intercept": predicted.mean(axis=1) - slope * actual.mean(axis=1),
        "spread_ratio": np.sqrt(divide(predicted_total, total)),
    }


def prediction_metrics(actual, predicted, *, bootstrap_repeats=0, seed=42):
    """Evaluate aligned predictions, optionally resampling independent units.

    Supply one row per independent unit before enabling bootstrap (for example,
    one biological donor after averaging source/fit replicas). Calibration
    regresses predicted values on actual values with an intercept. Prediction
    R² uses actual-value variance and can be negative; it is not regression R².
    """
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or predicted.shape != actual.shape or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Prediction metrics require aligned finite one-dimensional arrays")
    if bootstrap_repeats < 0:
        raise ValueError("bootstrap_repeats must be nonnegative")
    if not len(actual):
        return {"n_populations": 0, **{name: None for name in ["mae", "rmse", "bias", "pearson", "spearman", "prediction_r2", "calibration_slope", "calibration_intercept", "spread_ratio"]}}
    point = _measure(actual[None, :], predicted[None, :])
    result = {"n_populations": len(actual), **{k: float(v[0]) if np.isfinite(v[0]) else None for k, v in point.items()}}
    if bootstrap_repeats:
        indices = np.random.default_rng(seed).integers(0, len(actual), (bootstrap_repeats, len(actual)))
        repeats = _measure(actual[indices], predicted[indices])
        for name, values in repeats.items():
            finite = values[np.isfinite(values)]
            low, high = np.quantile(finite, [.025, .975]) if len(finite) else (None, None)
            result[name + "_ci_low"] = float(low) if low is not None else None
            result[name + "_ci_high"] = float(high) if high is not None else None
            result[name + "_finite_bootstraps"] = len(finite)
        result["bootstrap_repeats"] = bootstrap_repeats
    return result
