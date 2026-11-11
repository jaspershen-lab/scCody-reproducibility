from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

EDGER_R = r"""
suppressPackageStartupMessages({
  library(edgeR)
})
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 3) {
  stop("usage: Rscript edger_pb.R counts.tsv metadata.tsv out.tsv", call.=FALSE)
}
counts_path <- args[[1]]
meta_path <- args[[2]]
out_path <- args[[3]]
counts <- read.delim(counts_path, check.names=FALSE, stringsAsFactors=FALSE)
genes <- counts[[1]]
mat <- as.matrix(counts[, -1, drop=FALSE])
rownames(mat) <- genes
storage.mode(mat) <- "integer"
meta <- read.delim(meta_path, stringsAsFactors=FALSE)
if (!all(colnames(mat) == meta$sample_id)) {
  stop("count columns must exactly match metadata sample_id order", call.=FALSE)
}
meta$condition <- factor(meta$condition, levels=c("reference", "target"))
y <- DGEList(counts=mat, samples=meta, group=meta$condition)
keep <- filterByExpr(y, group=meta$condition)
y <- y[keep, , keep.lib.sizes=FALSE]
y <- calcNormFactors(y, method="TMM")
design <- model.matrix(~ condition, data=meta)
y <- estimateDisp(y, design)
fit <- glmQLFit(y, design, robust=TRUE)
qlf <- glmQLFTest(fit, coef="conditiontarget")
tab <- topTags(qlf, n=Inf, sort.by="none")$table
tab$gene <- rownames(tab)
tab <- tab[, c("gene", "logFC", "F", "PValue", "FDR")]
colnames(tab) <- c("gene", "edgeR_logFC", "edgeR_F", "edgeR_PValue", "edgeR_FDR")
write.table(tab, file=out_path, sep="\t", quote=FALSE, row.names=FALSE)
"""


def _as_counts(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or not np.isfinite(arr).all() or (arr < 0).any() or not np.allclose(arr, np.rint(arr), atol=1e-5, rtol=0):
        raise ValueError("edgeR requires finite, nonnegative integer counts")
    return np.rint(arr).astype(np.int64)


def _pseudobulk(
    x: np.ndarray,
    sample_ids: Iterable[Any],
    gene_names: Iterable[str],
    *,
    prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = _as_counts(x)
    sample_ids = np.asarray([str(s) for s in sample_ids], dtype=object)
    if x.shape[0] != len(sample_ids):
        raise ValueError(f"{prefix}: rows and sample_ids differ ({x.shape[0]} vs {len(sample_ids)})")
    genes = list(map(str, gene_names))
    cols = []
    meta_rows = []
    for sid in pd.unique(sample_ids):
        mask = sample_ids == sid
        col = f"{prefix}_{sid}"
        cols.append((col, x[mask].sum(axis=0)))
        meta_rows.append({"sample_id": col, "source_id": sid, "n_cells": int(mask.sum())})
    counts = pd.DataFrame({col: vals for col, vals in cols}, index=genes)
    counts.insert(0, "gene", genes)
    return counts, pd.DataFrame(meta_rows)


def _write_edger_inputs(
    target_counts: pd.DataFrame,
    target_meta: pd.DataFrame,
    ref_counts: pd.DataFrame,
    ref_meta: pd.DataFrame,
    out_prefix: Path,
) -> Tuple[Path, Path]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    counts = target_counts.merge(ref_counts, on="gene", how="inner")
    meta = pd.concat(
        [
            target_meta.assign(condition="target"),
            ref_meta.assign(condition="reference"),
        ],
        ignore_index=True,
    )
    counts_path = out_prefix.with_suffix(".counts.tsv")
    meta_path = out_prefix.with_suffix(".metadata.tsv")
    counts.to_csv(counts_path, sep="\t", index=False)
    meta[["sample_id", "condition", "source_id", "n_cells"]].to_csv(meta_path, sep="\t", index=False)
    return counts_path, meta_path


def _run_edger(counts_path: Path, meta_path: Path, out_path: Path, *, rscript: str) -> Optional[str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as handle:
        handle.write(EDGER_R)
        script_path = handle.name
    try:
        proc = subprocess.run(
            [rscript, script_path, str(counts_path), str(meta_path), str(out_path)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass
    if proc.returncode != 0:
        return (proc.stderr or proc.stdout or f"edgeR failed with exit code {proc.returncode}").strip()
    return None


def _safe_corr(a: np.ndarray, b: np.ndarray, *, method: str) -> Optional[float]:
    from scipy.stats import pearsonr, spearmanr

    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return None
    aa, bb = a[mask], b[mask]
    if np.allclose(aa, aa[0]) or np.allclose(bb, bb[0]):
        return None
    fn = pearsonr if method == "pearson" else spearmanr
    val, _ = fn(aa, bb)
    return float(val) if np.isfinite(val) else None


def _compare_edger(real_path: Path, pred_path: Path, *, top_k: Tuple[int, ...]) -> Dict[str, Any]:
    real = pd.read_csv(real_path, sep="\t")
    pred = pd.read_csv(pred_path, sep="\t")
    merged = real.merge(pred, on="gene", suffixes=("_real", "_pred"))
    logfc_real = merged["edgeR_logFC_real"].to_numpy(dtype=np.float64)
    logfc_pred = merged["edgeR_logFC_pred"].to_numpy(dtype=np.float64)
    finite = np.isfinite(logfc_real) & np.isfinite(logfc_pred)
    out: Dict[str, Any] = {
        "edgeR_n_genes": int(len(merged)),
        "edgeR_n_genes_used": int(finite.sum()),
        "edgeR_logFC_pearson_all": _safe_corr(logfc_real, logfc_pred, method="pearson"),
        "edgeR_logFC_spearman_all": _safe_corr(logfc_real, logfc_pred, method="spearman"),
    }
    real_rank = np.argsort(np.where(finite, np.abs(logfc_real), -np.inf))[::-1]
    pred_rank = np.argsort(np.where(finite, np.abs(logfc_pred), -np.inf))[::-1]
    for k in top_k:
        k_eff = min(int(k), int(finite.sum()))
        if k_eff <= 0:
            out[f"edgeR_logFC_pearson_top{k}"] = None
            out[f"edgeR_logFC_spearman_top{k}"] = None
            out[f"edgeR_top{k}_overlap_frac"] = None
            continue
        idx = real_rank[:k_eff]
        out[f"edgeR_logFC_pearson_top{k}"] = _safe_corr(logfc_real[idx], logfc_pred[idx], method="pearson")
        out[f"edgeR_logFC_spearman_top{k}"] = _safe_corr(logfc_real[idx], logfc_pred[idx], method="spearman")
        out[f"edgeR_top{k}_overlap_frac"] = float(len(set(real_rank[:k_eff]) & set(pred_rank[:k_eff])) / k_eff)
    return out






@dataclass(frozen=True)
class EdgeRContrast:
    x_true_counts: np.ndarray
    true_sample_ids: Iterable[Any]
    x_ref_counts: np.ndarray
    ref_sample_ids: Iterable[Any]
    gene_names: Iterable[str]
    outdir: Path | str
    run_name: str
    target_age: int | float
    contrast: str
    x_pred_counts: np.ndarray | None = None
    pred_sample_ids: Iterable[Any] | None = None
    rscript: str = "Rscript"
    top_k: Tuple[int, ...] = (100, 500)
    min_replicates_per_group: int = 2
    strict: bool = True


def run_edger_contrast(request: EdgeRContrast) -> Dict[str, Any]:
    """Run one real-only or generated-vs-real edgeR contrast."""
    x_pred_counts, pred_sample_ids = request.x_pred_counts, request.pred_sample_ids
    x_true_counts, true_sample_ids = request.x_true_counts, request.true_sample_ids
    x_ref_counts, ref_sample_ids = request.x_ref_counts, request.ref_sample_ids
    gene_names, outdir = request.gene_names, request.outdir
    run_name, target_age, contrast = request.run_name, request.target_age, request.contrast
    rscript, top_k = request.rscript, request.top_k
    min_replicates_per_group, strict = request.min_replicates_per_group, request.strict
    outdir = Path(outdir)
    prefix = outdir / f"{run_name}_ho{int(float(target_age))}_{contrast}"
    try:
        with tempfile.TemporaryDirectory(prefix="sccody-edger-") as temporary:
            input_prefix = Path(temporary) / prefix.name
            true_counts, true_meta = _pseudobulk(x_true_counts, true_sample_ids, gene_names, prefix="real")
            ref_counts, ref_meta = _pseudobulk(x_ref_counts, ref_sample_ids, gene_names, prefix="ref")
            predicted = x_pred_counts is not None and pred_sample_ids is not None
            pred_counts, pred_meta = (
                _pseudobulk(x_pred_counts, pred_sample_ids, gene_names, prefix="pred")
                if predicted else (None, None)
            )
            replicate_counts = [len(true_meta), len(ref_meta)] + ([len(pred_meta)] if predicted else [])
            if min(replicate_counts) < min_replicates_per_group:
                metrics = {
                    "edgeR_enabled": True,
                    "edgeR_error": "too_few_pseudobulk_replicates",
                    "edgeR_real_target_replicates": int(len(true_meta)),
                    "edgeR_ref_replicates": int(len(ref_meta)),
                    "edgeR_min_replicates_per_group": int(min_replicates_per_group),
                }
                if predicted:
                    metrics["edgeR_pred_target_replicates"] = int(len(pred_meta))
                if not strict:
                    return metrics
                raise ValueError(
                    "too_few_pseudobulk_replicates: "
                    f"real={len(true_meta)}, pred={len(pred_meta) if predicted else 'n/a'}, ref={len(ref_meta)}, "
                    f"min={min_replicates_per_group}"
                )
            real_counts_path, real_meta_path = _write_edger_inputs(true_counts, true_meta, ref_counts, ref_meta, input_prefix.with_name(input_prefix.name + "_real"))
            real_out = prefix.with_name(prefix.name + "_real_edgeR.tsv")
            err = _run_edger(real_counts_path, real_meta_path, real_out, rscript=rscript)
            if err is not None:
                if not strict:
                    return {"edgeR_enabled": True, "edgeR_error": err, "edgeR_stage": "real"}
                raise RuntimeError(f"edgeR real-stage failed: {err}")
            metrics = {
                "edgeR_enabled": True,
                "edgeR_method": "edgeR_glmQLFit_TMM_pseudobulk",
                "edgeR_real_path": str(real_out),
                "edgeR_real_target_replicates": int(len(true_meta)),
                "edgeR_ref_replicates": int(len(ref_meta)),
            }
            if predicted:
                pred_counts_path, pred_meta_path = _write_edger_inputs(
                    pred_counts, pred_meta, ref_counts, ref_meta, input_prefix.with_name(input_prefix.name + "_pred")
                )
                pred_out = prefix.with_name(prefix.name + "_pred_edgeR.tsv")
                err = _run_edger(pred_counts_path, pred_meta_path, pred_out, rscript=rscript)
                if err is not None:
                    if not strict:
                        return {"edgeR_enabled": True, "edgeR_error": err, "edgeR_stage": "pred"}
                    raise RuntimeError(f"edgeR pred-stage failed: {err}")
                metrics.update(_compare_edger(real_out, pred_out, top_k=top_k))
                metrics.update({"edgeR_pred_path": str(pred_out), "edgeR_pred_target_replicates": len(pred_meta)})
            return metrics
    except Exception as exc:
        if not strict:
            return {"edgeR_enabled": True, "edgeR_error": repr(exc)}
        raise RuntimeError(
            f"edgeR diagnostics failed for run={run_name!r} "
            f"target_age={target_age!r} contrast={contrast!r}"
        ) from exc


def safe_label(value: Any) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return label.strip("_") or "unknown"


def _corr(a: np.ndarray, b: np.ndarray, *, method: str = "pearson") -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return None
    aa = np.asarray(a[mask], dtype=np.float64)
    bb = np.asarray(b[mask], dtype=np.float64)
    if method == "spearman":
        aa = pd.Series(aa).rank(method="average").to_numpy(dtype=np.float64)
        bb = pd.Series(bb).rank(method="average").to_numpy(dtype=np.float64)
    if np.allclose(aa, aa[0]) or np.allclose(bb, bb[0]):
        return None
    val = float(np.corrcoef(aa, bb)[0, 1])
    return val if np.isfinite(val) else None


def _top_metrics(a: np.ndarray, b: np.ndarray, *, rank: np.ndarray, k: int) -> dict[str, Any]:
    finite = np.isfinite(a) & np.isfinite(b) & np.isfinite(rank)
    idx = np.where(finite)[0]
    k_eff = min(int(k), int(len(idx)))
    if k_eff <= 0:
        return {
            f"pearson_top{k}": None,
            f"spearman_top{k}": None,
            f"top{k}_overlap_frac": None,
        }
    rank_idx = idx[np.argsort(np.abs(rank[idx]))[::-1][:k_eff]]
    a_idx = idx[np.argsort(np.abs(a[idx]))[::-1][:k_eff]]
    b_idx = idx[np.argsort(np.abs(b[idx]))[::-1][:k_eff]]
    return {
        f"pearson_top{k}": _corr(a[rank_idx], b[rank_idx], method="pearson"),
        f"spearman_top{k}": _corr(a[rank_idx], b[rank_idx], method="spearman"),
        f"top{k}_overlap_frac": float(len(set(a_idx) & set(b_idx)) / k_eff),
    }


def _compare_vectors(a: np.ndarray, b: np.ndarray, *, rank: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(a) & np.isfinite(b)
    aa = np.asarray(a[finite], dtype=np.float64)
    bb = np.asarray(b[finite], dtype=np.float64)
    # ``a_vs_b`` metrics treat b as the explanatory/real vector and a as the
    # predicted vector, so the calibration regression is a = intercept + slope*b.
    if len(bb) >= 3 and not np.allclose(bb, bb[0]):
        centered = bb - bb.mean()
        slope = float(np.dot(centered, aa - aa.mean()) / np.dot(centered, centered))
        intercept = float(aa.mean() - slope * bb.mean())
        residual = aa - (intercept + slope * bb)
        total = float(np.dot(aa - aa.mean(), aa - aa.mean()))
        r2 = float(1.0 - np.dot(residual, residual) / total) if total > 0.0 else None
    else:
        slope = intercept = r2 = None
    out: dict[str, Any] = {
        "n_genes_used": int(finite.sum()),
        "pearson_all": _corr(a, b, method="pearson"),
        "spearman_all": _corr(a, b, method="spearman"),
        "ols_slope_all": slope,
        "ols_intercept_all": intercept,
        "ols_r2_all": r2,
    }
    for k in (100, 200, 500):
        out.update(_top_metrics(a, b, rank=rank, k=k))
    return out


def _edge_logfc_for_genes(edge_path: str | Path, gene_names: Iterable[str]) -> np.ndarray | None:
    if not edge_path:
        return None
    path = Path(edge_path)
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t")
    if "gene" not in df.columns or "edgeR_logFC" not in df.columns:
        return None
    mapping = dict(zip(df["gene"].astype(str), df["edgeR_logFC"].astype(float)))
    return np.asarray([mapping.get(str(g), np.nan) for g in gene_names], dtype=np.float64)


def compare_diagnostics_to_edger(
    diagnostics: dict[str, Any],
    *,
    edge_path: str | Path,
    gene_names: Iterable[str],
) -> dict[str, Any]:
    edge = _edge_logfc_for_genes(edge_path, gene_names)
    if edge is None:
        return {"edgeR_real_available": False}
    pred = np.asarray(diagnostics["logfc_pred"], dtype=np.float64)
    real = np.asarray(diagnostics["logfc_real"], dtype=np.float64)
    out: dict[str, Any] = {"edgeR_real_available": True}
    out.update(
        {
            f"normalized_pred_vs_edgeR_real_{k}": v
            for k, v in _compare_vectors(pred, edge, rank=edge).items()
        }
    )
    out.update(
        {
            f"normalized_real_vs_edgeR_real_{k}": v
            for k, v in _compare_vectors(real, edge, rank=edge).items()
        }
    )
    return out


def compute_per_cell_type_edger(
    *, rows, de_logfc_diagnostics_fn, x_pred, x_true, pred_ct_codes,
    true_ct_codes, x_ref, ref_ages, ref_ct_codes, x_true_counts, true_sample_ids,
    x_ref_counts, ref_sample_ids, actual_age, ct_categories, gene_names, outdir,
    run_name, target_age, x_pred_counts=None, pred_sample_ids=None, rscript="Rscript",
    min_cells=50, 
) -> list[dict[str, Any]]:
    """Append per-cell-type edgeR/logFC diagnostics to existing per-CT rows."""
    by_key = {(int(r["cell_type_code"]), int(r["target_age"])): r for r in rows}
    gene_names = list(gene_names)
    pred_ct_codes = np.asarray(pred_ct_codes, dtype=np.int64)
    true_ct_codes = np.asarray(true_ct_codes, dtype=np.int64)
    ref_ct_codes = np.asarray(ref_ct_codes, dtype=np.int64)
    ref_ages = np.asarray(ref_ages, dtype=np.float64)
    true_sample_ids_arr = np.asarray(list(true_sample_ids), dtype=object) if true_sample_ids is not None else None
    pred_sample_ids_arr = np.asarray(list(pred_sample_ids), dtype=object) if pred_sample_ids is not None else None
    ref_sample_ids_arr = np.asarray(list(ref_sample_ids), dtype=object) if ref_sample_ids is not None else None
    outdir = Path(outdir)

    for ct_code in sorted(set(pred_ct_codes.tolist()) | set(true_ct_codes.tolist())):
        key = (int(ct_code), int(target_age))
        row = by_key.get(key)
        if row is None:
            continue
        ct_name = str(ct_categories[int(ct_code)]) if 0 <= int(ct_code) < len(ct_categories) else str(ct_code)
        ct_label = safe_label(ct_name)
        pred_mask = pred_ct_codes == ct_code
        true_mask = true_ct_codes == ct_code
        ref_ct_mask = ref_ct_codes == ct_code
        if int(pred_mask.sum()) < min_cells or int(true_mask.sum()) < min_cells:
            continue

        for contrast, ref_age_mask in (("young", ref_ages < actual_age), ("old", ref_ages > actual_age)):
            ref_mask = ref_ct_mask & ref_age_mask
            if int(ref_mask.sum()) < min_cells:
                continue
            diag = de_logfc_diagnostics_fn(x_pred[pred_mask], x_true[true_mask], x_ref[ref_mask])
            if diag is None:
                continue

            ct_run_name = f"{run_name}_ct_{ct_label}"
            edge_metrics = None
            if (
                x_true_counts is not None
                and x_ref_counts is not None
                and true_sample_ids_arr is not None
                and ref_sample_ids_arr is not None
            ):
                if x_pred_counts is not None and pred_sample_ids_arr is not None:
                    edge_metrics = run_edger_contrast(EdgeRContrast(
                        x_pred_counts=x_pred_counts[pred_mask],
                        pred_sample_ids=pred_sample_ids_arr[pred_mask],
                        x_true_counts=x_true_counts[true_mask],
                        true_sample_ids=true_sample_ids_arr[true_mask],
                        x_ref_counts=x_ref_counts[ref_mask],
                        ref_sample_ids=ref_sample_ids_arr[ref_mask],
                        gene_names=gene_names,
                        outdir=outdir,
                        run_name=ct_run_name,
                        target_age=actual_age,
                        contrast=contrast,
                        rscript=rscript,
                        strict=False,
                    ))
                else:
                    edge_metrics = run_edger_contrast(EdgeRContrast(
                        x_true_counts=x_true_counts[true_mask],
                        true_sample_ids=true_sample_ids_arr[true_mask],
                        x_ref_counts=x_ref_counts[ref_mask],
                        ref_sample_ids=ref_sample_ids_arr[ref_mask],
                        gene_names=gene_names,
                        outdir=outdir,
                        run_name=ct_run_name,
                        target_age=actual_age,
                        contrast=contrast,
                        rscript=rscript,
                        strict=False,
                    ))

            row[f"edgeR_{contrast}"] = edge_metrics
            if edge_metrics is None:
                continue
            for metric_key, metric_val in edge_metrics.items():
                if isinstance(metric_val, (int, float, str, bool)) or metric_val is None:
                    row[f"edgeR_{contrast}_{metric_key}"] = metric_val
            edge_path = edge_metrics.get("edgeR_real_path")
            if edge_path:
                joined = compare_diagnostics_to_edger(diag, edge_path=edge_path, gene_names=gene_names)
                for metric_key, metric_val in joined.items():
                    row[f"{contrast}_{metric_key}"] = metric_val
    return rows
