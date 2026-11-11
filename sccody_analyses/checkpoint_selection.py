"""Checkpoint selection shared by scCody and the comparators.

Every model is scored on its internal-holdout donors only; external donors are
never read. The rule for choosing among scored epochs lives here so that all
models use the same one.

  holdout_L_gene    mean gene-space Sinkhorn loss between predicted and observed
                    cells at each holdout donor's chronological age; ties go to
                    the earlier epoch.
  fixed_last_epoch  the final epoch.
"""
from __future__ import annotations

import copy
import math

import numpy as np

METRICS = ("holdout_L_gene", "fixed_last_epoch")


def localization_metrics(true_age, predicted_age) -> dict:
    """MAE, Pearson r, OLS slope and SD ratio of one prediction per donor."""
    y = np.asarray(true_age, dtype=float)
    p = np.asarray(predicted_age, dtype=float)
    if y.shape != p.shape or not len(y):
        raise ValueError("Need one finite prediction per donor")
    true_sd, predicted_sd = float(np.std(y)), float(np.std(p))
    pearson = float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 and true_sd and predicted_sd else None
    slope = float(np.cov(y, p, ddof=0)[0, 1] / np.var(y)) if true_sd else None
    ratio = predicted_sd / true_sd if true_sd else None
    return {"n_samples": int(len(y)), "mae": float(np.mean(np.abs(y - p))), "pearson": pearson,
            "slope": slope, "spread_ratio": ratio}


def selection_key(record: dict, metric: str):
    """Sort key of one scored epoch (lower is better), or None if it is not eligible."""
    epoch = int(record["epoch"])
    if metric == "fixed_last_epoch":
        return (-epoch,)
    if metric == "holdout_L_gene":
        score = float(record.get("selection_score", math.inf))
        return (score, epoch) if math.isfinite(score) else None
    raise ValueError(f"Unknown selection metric {metric!r}; expected one of {METRICS}")


def choose_checkpoint(records, metric: str) -> dict:
    """Best eligible record under ``metric``."""
    eligible = [(selection_key(r, metric), r) for r in records]
    eligible = [(key, r) for key, r in eligible if key is not None]
    if not eligible:
        raise RuntimeError(f"No checkpoint has a finite {metric} score")
    return min(eligible, key=lambda item: item[0])[1]


def _clone_to_cpu(value):
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return copy.deepcopy(value)


class CheckpointSelector:
    """Epoch schedule and best-state retention, with a compact final selection."""

    def __init__(self, metric: str, *, total_epochs: int, every_epochs: int = 1):
        if metric not in METRICS:
            raise ValueError(f"Unknown selection metric {metric!r}; expected one of {METRICS}")
        self.metric = str(metric)
        self.total_epochs = int(total_epochs)
        self.every_epochs = max(1, int(every_epochs))
        self.best: dict[str, object] | None = None
        self.best_key = None
        self.best_state = None

    def should_score(self, epoch: int) -> bool:
        epoch = int(epoch)
        if self.metric == "fixed_last_epoch":
            return epoch == self.total_epochs
        return epoch % self.every_epochs == 0 or epoch == self.total_epochs

    def observe(self, epoch: int, record: dict[str, object], state, *, clone=_clone_to_cpu) -> bool:
        item = {"epoch": int(epoch), "selection_metric": self.metric, **record}
        key = selection_key(item, self.metric)
        if key is None:
            return False
        if self.best is None or key < self.best_key:
            self.best, self.best_key = item, key
            self.best_state = clone(state)
            return True
        return False

    def observe_fixed_last(self, epoch: int, state) -> bool:
        if not self.should_score(epoch):
            return False
        return self.observe(epoch, {"selection_score": 0.0}, state)

    def require_best_state(self):
        if self.best_state is None or self.best is None:
            raise RuntimeError(f"Training finished without an eligible {self.metric} checkpoint")
        return self.best_state

    def payload(self) -> dict[str, object]:
        if self.best is None:
            raise RuntimeError("Checkpoint selection has no best epoch")
        return {"epoch": int(self.best["epoch"]), "selection_metric": self.metric,
                "selection_score": float(self.best["selection_score"]), "source_policy": "pooled_youngest_forward"}
