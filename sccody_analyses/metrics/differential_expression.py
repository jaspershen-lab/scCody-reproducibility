from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.stats import rankdata

def de_logfc_diagnostics(
    x_pred: np.ndarray,
    x_true_target: np.ndarray,
    x_true_other: np.ndarray,
    *,
    top_k: Tuple[int, ...] = (100, 200, 500),
    eps: float = 1e-4,
) -> dict:
    """
    DE log fold-change correlation between generated and real cells.

    Both generated and real target-age cells are compared against the SAME
    reference population. Their logFC difference cancels the reference, but
    correlation between the two vectors can still reflect shared reference
    variation. Interpret it alongside direction, top-k and magnitude metrics.

        logFC_real[g] = log2( mean(x_true_target[:, g]) / mean(x_true_other[:, g]) )
        logFC_pred[g] = log2( mean(x_pred[:, g])        / mean(x_true_other[:, g]) )

    Pearson r and Spearman rho are computed:
      - over ALL genes
      - over the top-k most differentially expressed genes by |logFC_real|

    Parameters
    ----------
    x_pred        : (N_pred, G)  generated cells at target age
    x_true_target : (N_tgt,  G)  real cells at target holdout age
    x_true_other  : (N_oth,  G)  reference cells (training, other ages)
    top_k         : which top-DE subsets to evaluate (by |logFC_real|)
    eps           : pseudocount added to means before log to avoid log(0)

    Returns
    -------
    dict with keys:
      logfc_pearson_all    : float | None  Pearson r over all usable genes
      logfc_spearman_all   : float | None  Spearman rho over all usable genes
      logfc_pearson_top{k} : float | None  Pearson r restricted to top-k DE genes
      logfc_spearman_top{k}: float | None  Spearman rho restricted to top-k DE genes
      top{k}_overlap_frac  : float | None  overlap between real/pred top-k genes
      logfc_real           : ndarray        real target-vs-reference logFC
      logfc_pred           : ndarray        generated-vs-reference logFC
      finite_mask          : ndarray[bool]  finite genes usable for correlations
      n_genes              : int    total genes evaluated
      n_genes_used         : int    finite genes used for correlation
      n_pred_mean_clipped  : int    predicted gene means clipped at zero
      mean_ref             : float  mean(x_true_other) — sanity check
    """
    from scipy.stats import pearsonr, spearmanr

    def _safe_corr(a, b, corr_fn):
        mask = np.isfinite(a) & np.isfinite(b)
        if int(mask.sum()) < 3:
            return None
        aa = a[mask]
        bb = b[mask]
        if np.allclose(aa, aa[0]) or np.allclose(bb, bb[0]):
            return None
        val, _ = corr_fn(aa, bb)
        return float(val) if np.isfinite(val) else None

    def _safe_ols(real, predicted):
        """Regress predicted logFC on real logFC with an intercept."""
        mask = np.isfinite(real) & np.isfinite(predicted)
        if int(mask.sum()) < 3:
            return None
        x = np.asarray(real[mask], dtype=np.float64)
        y = np.asarray(predicted[mask], dtype=np.float64)
        x_centered = x - x.mean()
        denominator = float(np.dot(x_centered, x_centered))
        if denominator <= 0.0:
            return None
        slope = float(np.dot(x_centered, y - y.mean()) / denominator)
        intercept = float(y.mean() - slope * x.mean())
        residual = y - (intercept + slope * x)
        total = float(np.dot(y - y.mean(), y - y.mean()))
        r2 = float(1.0 - np.dot(residual, residual) / total) if total > 0.0 else None
        return {"slope": slope, "intercept": intercept, "r2": r2}

    x_pred         = np.asarray(x_pred,         dtype=np.float64)
    x_true_target  = np.asarray(x_true_target,  dtype=np.float64)
    x_true_other   = np.asarray(x_true_other,   dtype=np.float64)

    # Some unconstrained decoders can emit slightly negative reconstructed
    # expression. Expression logFC is only defined for non-negative means, so
    # clamp before applying the pseudocount.
    mean_ref    = np.clip(x_true_other.mean(axis=0),  0.0, None) + eps  # (G,)
    mean_target = np.clip(x_true_target.mean(axis=0), 0.0, None) + eps  # (G,)
    mean_pred   = np.clip(x_pred.mean(axis=0),        0.0, None) + eps  # (G,)

    logfc_real = np.log2(mean_target) - np.log2(mean_ref)  # (G,)
    logfc_pred = np.log2(mean_pred)   - np.log2(mean_ref)  # (G,)

    finite_mask = np.isfinite(logfc_real) & np.isfinite(logfc_pred)
    result: dict = {
        "n_genes": int(x_pred.shape[1]),
        "n_genes_used": int(finite_mask.sum()),
        "n_pred_mean_clipped": int((x_pred.mean(axis=0) < 0.0).sum()),
        "logfc_real": logfc_real.astype(np.float32),
        "logfc_pred": logfc_pred.astype(np.float32),
        "finite_mask": finite_mask.astype(bool),
    }

    # All genes
    result["logfc_pearson_all"] = _safe_corr(logfc_real, logfc_pred, pearsonr)
    result["logfc_spearman_all"] = _safe_corr(logfc_real, logfc_pred, spearmanr)
    ols_all = _safe_ols(logfc_real, logfc_pred)
    result["logfc_ols_slope_all"] = ols_all["slope"] if ols_all else None
    result["logfc_ols_intercept_all"] = ols_all["intercept"] if ols_all else None
    result["logfc_ols_r2_all"] = ols_all["r2"] if ols_all else None

    # Top-k subsets. Correlations use the real top-k set; overlaps compare
    # real top-k vs predicted top-k, both ranked by absolute logFC.
    real_rank_idx = np.argsort(np.where(finite_mask, np.abs(logfc_real), -np.inf))[::-1]
    pred_rank_idx = np.argsort(np.where(finite_mask, np.abs(logfc_pred), -np.inf))[::-1]
    for k in top_k:
        k_eff = min(k, int(finite_mask.sum()))
        idx_k = real_rank_idx[:k_eff]
        result[f"logfc_pearson_top{k}"] = _safe_corr(logfc_real[idx_k], logfc_pred[idx_k], pearsonr)
        result[f"logfc_spearman_top{k}"] = _safe_corr(logfc_real[idx_k], logfc_pred[idx_k], spearmanr)
        ols_top = _safe_ols(logfc_real[idx_k], logfc_pred[idx_k])
        result[f"logfc_ols_slope_top{k}"] = ols_top["slope"] if ols_top else None
        result[f"logfc_ols_intercept_top{k}"] = ols_top["intercept"] if ols_top else None
        result[f"logfc_ols_r2_top{k}"] = ols_top["r2"] if ols_top else None
        real_mask = np.zeros(x_pred.shape[1], dtype=bool)
        pred_mask = np.zeros(x_pred.shape[1], dtype=bool)
        if k_eff > 0:
            real_mask[real_rank_idx[:k_eff]] = True
            pred_mask[pred_rank_idx[:k_eff]] = True
            result[f"top{k}_overlap_frac"] = float(np.logical_and(real_mask, pred_mask).sum() / k_eff)
        else:
            result[f"top{k}_overlap_frac"] = None
        result[f"real_top{k}_mask"] = real_mask
        result[f"pred_top{k}_mask"] = pred_mask

    result["mean_ref"] = float(mean_ref.mean())
    return result


def de_logfc_correlation(
    x_pred: np.ndarray,
    x_true_target: np.ndarray,
    x_true_other: np.ndarray,
    *,
    top_k: Tuple[int, ...] = (100, 200, 500),
    eps: float = 1e-4,
) -> dict:
    """Scalar DE logFC metrics, excluding arrays used for optional plots."""
    result = de_logfc_diagnostics(
        x_pred,
        x_true_target,
        x_true_other,
        top_k=top_k,
        eps=eps,
    )
    return {
        k: v for k, v in result.items()
        if not isinstance(v, np.ndarray)
    }


