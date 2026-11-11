from __future__ import annotations
import warnings
from pathlib import Path
from typing import Any, Iterable
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sccody_analyses.metrics.differential_expression import de_logfc_diagnostics
from sccody_analyses.metrics.edger import EdgeRContrast, compare_diagnostics_to_edger, compute_per_cell_type_edger, run_edger_contrast
from sccody_analyses.benchmarks.engine.evaluation.config import EvaluationConfig
from sccody_analyses.benchmarks.engine.evaluation.populations import EvaluationReference, ProjectionResult
from sccody_analyses.benchmarks.engine.projection.metrics import sinkhorn_distance

def scalar_de_metrics(de: dict[str, Any] | None) -> dict[str, Any] | None:
    if de is None:
        return None
    return {k: v for k, v in de.items() if not isinstance(v, np.ndarray)}

def flatten_edger_scalars(edge_metrics: dict[str, Any] | None, *, contrast: str) -> dict[str, Any]:
    if not edge_metrics:
        return {}
    return {f'edgeR_{contrast}_{metric_key}': metric_val for metric_key, metric_val in edge_metrics.items() if isinstance(metric_val, (int, float, str, bool)) or metric_val is None}

from sccody_analyses.metrics.differential_expression import de_logfc_correlation
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

def per_gene_wasserstein(true_data, pred_data):
    true, pred = (np.asarray(true_data), np.asarray(pred_data))
    assert true.shape[1] == pred.shape[1], 'true_data and pred_data must have same #genes (columns)'
    distances = np.empty(true.shape[1], dtype=float)
    for index in range(true.shape[1]):
        x, y = (true[:, index], pred[:, index])
        distances[index] = wasserstein_distance(x[np.isfinite(x)], y[np.isfinite(y)])
    return (distances, float(np.mean(distances)), float(np.median(distances)))

def mean_expression_corr(true_data, pred_data, eps=1e-12):
    true, pred = (np.asarray(true_data, dtype=np.float64), np.asarray(pred_data, dtype=np.float64))
    a, b = (np.nanmean(true, axis=0), np.nanmean(pred, axis=0))
    a, b = (a - a.mean(), b - b.mean())
    sa, sb = (a.std(), b.std())
    if not np.isfinite(sa) or not np.isfinite(sb) or sa < eps or (sb < eps):
        return 0.0
    value = float(a @ b / (len(a) * (sa * sb + eps)))
    return float(np.clip(value, -1.0, 1.0)) if np.isfinite(value) else 0.0

def _pairwise_rbf_sum(first, second, gamma, block_size=256, exclude_diag=False, dtype=np.float32):
    first, second = (np.asarray(first, dtype=dtype), np.asarray(second, dtype=dtype))
    first_norm, second_norm = (np.sum(first * first, axis=1), np.sum(second * second, axis=1))
    total = 0.0
    for first_start in range(0, len(first), block_size):
        first_block = first[first_start:first_start + block_size]
        for second_start in range(0, len(second), block_size):
            second_block = second[second_start:second_start + block_size]
            distances = first_norm[first_start:first_start + len(first_block), None] + second_norm[second_start:second_start + len(second_block)][None, :] - 2.0 * first_block @ second_block.T
            kernel = np.exp(-gamma * distances, dtype=np.float64)
            if exclude_diag and first is second:
                for index in np.intersect1d(np.arange(first_start, first_start + len(first_block)), np.arange(second_start, second_start + len(second_block))):
                    kernel[index - first_start, index - second_start] = 0.0
            total += float(kernel.sum())
    return total

def mmd_rbf_unbiased_blocked(true_data, pred_data, *, gamma, block_size=256, dtype=np.float32):
    true, pred = (np.asarray(true_data, dtype=dtype), np.asarray(pred_data, dtype=dtype))
    n, m = (len(true), len(pred))
    xx = _pairwise_rbf_sum(true, true, gamma, block_size=block_size, exclude_diag=True, dtype=dtype)
    yy = _pairwise_rbf_sum(pred, pred, gamma, block_size=block_size, exclude_diag=True, dtype=dtype)
    xy = _pairwise_rbf_sum(true, pred, gamma, block_size=block_size, dtype=dtype)
    value = xx / (n * (n - 1) + 1e-12) if n > 1 else 0.0
    value += yy / (m * (m - 1) + 1e-12) if m > 1 else 0.0
    value -= 2.0 * xy / (n * m + 1e-12)
    return max(float(value), 0.0)

def get_fixed_gamma(target_true_data):
    subset = target_true_data[:500]
    distances = cdist(subset, subset, metric='sqeuclidean')
    return 1.0 / (np.median(distances[distances > 0]) + 1e-06)

def global_distribution_metrics(x_true: np.ndarray, x_pred: np.ndarray, *, mmd_block_size: int=256, sinkhorn_max_cells: int=1000, sinkhorn_blur: float=0.05, sinkhorn_scaling: float=0.5) -> dict[str, Any]:
    gamma = get_fixed_gamma(x_true[:min(500, len(x_true))])
    _, wasserstein_mean, wasserstein_median = per_gene_wasserstein(x_true, x_pred)
    mmd = float(mmd_rbf_unbiased_blocked(x_true, x_pred, gamma=gamma, block_size=mmd_block_size))
    gene_sinkhorn = sinkhorn_distance(x_true, x_pred, max_cells=sinkhorn_max_cells, blur=sinkhorn_blur, scaling=sinkhorn_scaling)
    n_sinkhorn = min(len(x_true), len(x_pred), sinkhorn_max_cells if sinkhorn_max_cells > 0 else min(len(x_true), len(x_pred)))
    return {'per_gene_wasserstein_mean': float(wasserstein_mean), 'per_gene_wasserstein_median': float(wasserstein_median), 'gene_residual_wasserstein_mean': float(wasserstein_mean), 'gene_residual_wasserstein_median': float(wasserstein_median), 'mmd': mmd, 'gene_mmd2_at_chronological_age': mmd, 'gene_space_sinkhorn_at_chrono': float(gene_sinkhorn), 'gene_space_sinkhorn_space': 'decoded_expression', 'gene_space_sinkhorn_n_cells': int(n_sinkhorn), 'mean_expression_corr': float(mean_expression_corr(x_true, x_pred))}

def fit_pca_knn_cell_type(x_ref, ct_codes, *, n_pcs=50, k=15) -> dict[str, Any]:
    x_ref = np.asarray(x_ref, dtype=np.float32)
    scaler = StandardScaler()
    pca = PCA(n_components=min(int(n_pcs), x_ref.shape[1], max(1, x_ref.shape[0] - 1)), random_state=0)
    latent = pca.fit_transform(scaler.fit_transform(x_ref))
    return {'scaler': scaler, 'pca': pca, 'neighbors': NearestNeighbors(n_neighbors=min(int(k), len(latent))).fit(latent), 'cell_types': np.asarray(ct_codes, dtype=np.int64)}

def predict_pca_knn_cell_type(expression, classifier) -> dict[str, Any]:
    latent = classifier['pca'].transform(classifier['scaler'].transform(expression))
    distances, indices = classifier['neighbors'].kneighbors(latent)
    labels = classifier['cell_types']
    predicted, confidence = (np.empty(len(latent), dtype=np.int64), np.empty(len(latent), dtype=np.float32))
    for row, neighbors in enumerate(indices):
        values, counts = np.unique(labels[neighbors], return_counts=True)
        tied = values[counts == counts.max()]
        selected = tied[0] if len(tied) == 1 else tied[np.argmin([distances[row][labels[neighbors] == value].sum() for value in tied])]
        predicted[row], confidence[row] = (selected, counts.max() / len(neighbors))
    return {'predicted': predicted, 'mean_confidence': float(confidence.mean()) if len(confidence) else None}

def cell_type_counts(codes, categories) -> dict[str, int]:
    return {str(categories[int(code)]) if 0 <= int(code) < len(categories) else str(code): int(count) for code, count in zip(*np.unique(np.asarray(codes, dtype=np.int64), return_counts=True))}

def compute_within_cell_type_metrics(*, predicted_expression, true_expression, predicted_cell_types, true_cell_types, reference_expression, reference_ages, reference_cell_types, actual_age, cell_type_categories, min_cells=50, max_sinkhorn_cells=1000, sinkhorn_blur=0.05, sinkhorn_scaling=0.5, method=None, run_name=None, target_age=None, ct_assignment='provided') -> list:
    x_pred, x_true, pred_codes, true_codes = (np.asarray(predicted_expression, dtype=np.float32), np.asarray(true_expression, dtype=np.float32), np.asarray(predicted_cell_types, dtype=np.int64), np.asarray(true_cell_types, dtype=np.int64))
    x_ref, ref_ages, ref_codes = (np.asarray(reference_expression, dtype=np.float32), np.asarray(reference_ages, dtype=np.float64), np.asarray(reference_cell_types, dtype=np.int64))
    rows = []
    for code in sorted(set(pred_codes.tolist()) | set(true_codes.tolist())):
        pred_mask, true_mask = (pred_codes == code, true_codes == code)
        if int(pred_mask.sum()) < min_cells or int(true_mask.sum()) < min_cells:
            continue
        predicted, truth = (x_pred[pred_mask], x_true[true_mask])
        ref_mask = ref_codes == code
        young, old = (ref_mask & (ref_ages < actual_age), ref_mask & (ref_ages > actual_age))
        de_young = de_logfc_correlation(predicted, truth, x_ref[young]) if int(young.sum()) >= min_cells else None
        de_old = de_logfc_correlation(predicted, truth, x_ref[old]) if int(old.sum()) >= min_cells else None
        _, wasserstein_mean, wasserstein_median = per_gene_wasserstein(truth, predicted)
        gene_sinkhorn = sinkhorn_distance(truth, predicted, max_cells=max_sinkhorn_cells, blur=sinkhorn_blur, scaling=sinkhorn_scaling)
        n_sinkhorn = min(len(truth), len(predicted), max_sinkhorn_cells if max_sinkhorn_cells > 0 else min(len(truth), len(predicted)))
        row = {'method': method, 'run_name': run_name, 'target_age': int(target_age) if target_age is not None else None, 'actual_holdout_age': float(actual_age), 'cell_type': str(cell_type_categories[int(code)]) if 0 <= int(code) < len(cell_type_categories) else str(code), 'cell_type_code': int(code), 'ct_assignment': str(ct_assignment), 'n_pred_cells': int(pred_mask.sum()), 'n_true_cells': int(true_mask.sum()), 'n_ref_cells': int(ref_mask.sum()), 'per_gene_wasserstein_mean': float(wasserstein_mean), 'per_gene_wasserstein_median': float(wasserstein_median), 'gene_space_sinkhorn_at_chrono': float(gene_sinkhorn), 'gene_space_sinkhorn_space': 'decoded_expression', 'gene_space_sinkhorn_n_cells': int(n_sinkhorn), 'mean_expression_corr': float(mean_expression_corr(truth, predicted)), 'mmd': float(mmd_rbf_unbiased_blocked(truth, predicted, gamma=get_fixed_gamma(truth))), 'de_young_n_ref': int(young.sum()), 'de_old_n_ref': int(old.sum())}
        for label, result in (('young', de_young), ('old', de_old)):
            for key in ('logfc_pearson_all', 'logfc_spearman_all', 'logfc_ols_slope_all', 'logfc_ols_intercept_all', 'logfc_ols_r2_all', 'logfc_pearson_top100', 'logfc_spearman_top100', 'logfc_pearson_top500', 'logfc_spearman_top500', 'top100_overlap_frac', 'top500_overlap_frac'):
                row[f'de_vs_{label}_{key}'] = result.get(key) if result else None
        rows.append(row)
    return rows

def edger_metrics_for_contrast(de_diag, contrast, batch, reference, ref_mask, config, run_name) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    actual_age = batch.actual_age
    x_ref_counts = reference.counts[ref_mask] if reference.counts is not None else None
    ref_sample_ids = np.asarray(reference.donors)[ref_mask] if reference.donors is not None else None
    gene_names = reference.gene_names
    if config.de_method != 'edger_pseudobulk' or de_diag is None:
        return (None, {})
    if batch.true_counts is None or x_ref_counts is None or batch.true_donors is None or (ref_sample_ids is None):
        raise ValueError(f"edgeR pseudobulk requested but true/ref raw counts or sample IDs are missing for contrast '{contrast}' in run '{run_name}'")
    edge_metrics = run_edger_contrast(EdgeRContrast(x_pred_counts=batch.predicted_counts, pred_sample_ids=batch.predicted_donors, x_true_counts=batch.true_counts, true_sample_ids=batch.true_donors, x_ref_counts=x_ref_counts, ref_sample_ids=ref_sample_ids, gene_names=gene_names, outdir=config.edger_dir, run_name=run_name, target_age=actual_age, contrast=contrast, rscript=config.rscript, strict=False))
    joined_metrics: dict[str, Any] = {}
    if edge_metrics and edge_metrics.get('edgeR_error'):
        err = str(edge_metrics.get('edgeR_error'))
        if err == 'too_few_pseudobulk_replicates':
            warnings.warn(f"Skipping edgeR diagnostic because pseudobulk replicate support is too low: run={run_name!r}, target_age={actual_age!r}, contrast={contrast!r}, real={edge_metrics.get('edgeR_real_target_replicates')}, pred={edge_metrics.get('edgeR_pred_target_replicates')}, ref={edge_metrics.get('edgeR_ref_replicates')}, min={edge_metrics.get('edgeR_min_replicates_per_group')}", RuntimeWarning)
            return (edge_metrics, joined_metrics)
        raise RuntimeError(f"edgeR failed for contrast '{contrast}' in run '{run_name}': {err}")
    if edge_metrics and edge_metrics.get('edgeR_real_path'):
        joined_metrics = compare_diagnostics_to_edger(de_diag, edge_path=edge_metrics['edgeR_real_path'], gene_names=gene_names)
        edge_metrics.update(joined_metrics)
    return (edge_metrics, {f'{contrast}_{metric_key}': metric_val for metric_key, metric_val in joined_metrics.items() if isinstance(metric_val, (int, float, str, bool)) or metric_val is None})

def _add_de_scalars(output: dict[str, Any], young: dict[str, Any] | None, old: dict[str, Any] | None, young_count: int, old_count: int) -> None:
    output['de_young_n_ref'] = young_count
    output['de_old_n_ref'] = old_count
    keys = ('logfc_pearson_all', 'logfc_spearman_all', 'logfc_ols_slope_all', 'logfc_ols_intercept_all', 'logfc_ols_r2_all', 'logfc_pearson_top100', 'logfc_spearman_top100', 'logfc_ols_slope_top100', 'logfc_ols_intercept_top100', 'logfc_ols_r2_top100', 'logfc_pearson_top500', 'logfc_spearman_top500', 'logfc_ols_slope_top500', 'logfc_ols_intercept_top500', 'logfc_ols_r2_top500', 'top100_overlap_frac', 'top500_overlap_frac')
    for prefix, values in (('de_vs_young', scalar_de_metrics(young)), ('de_vs_old', scalar_de_metrics(old))):
        output.update({f'{prefix}_{key}': values.get(key) if values else None for key in keys})

def _evaluate_cell_types(batch: ProjectionResult, reference: EvaluationReference, config: EvaluationConfig, output: dict[str, Any], *, method: str, run_name: str, target_age: int) -> list[dict[str, Any]]:
    predicted = batch.predicted_cell_types
    if predicted is not None:
        assignment = 'provided'
        output['ct_assignment'] = assignment
    elif batch.true_cell_types is not None and reference.cell_type_codes is not None:
        classifier = reference.cell_type_classifier or fit_pca_knn_cell_type(reference.expression, reference.cell_type_codes)
        prediction = predict_pca_knn_cell_type(batch.predicted_expression, classifier)
        predicted = prediction['predicted']
        assignment = 'pca_knn'
        output['ct_assignment'] = assignment
        output['ct_knn_mean_confidence'] = prediction['mean_confidence']
        output['generated_cell_type_counts_knn'] = cell_type_counts(predicted, reference.cell_type_categories)
    else:
        output['ct_assignment'] = ''
        return []
    if batch.true_cell_types is None or reference.cell_type_codes is None:
        return []
    rows = compute_within_cell_type_metrics(predicted_expression=batch.predicted_expression, true_expression=batch.true_expression, predicted_cell_types=np.asarray(predicted, dtype=np.int64), true_cell_types=np.asarray(batch.true_cell_types, dtype=np.int64), reference_expression=reference.expression, reference_ages=reference.ages, reference_cell_types=np.asarray(reference.cell_type_codes, dtype=np.int64), actual_age=batch.actual_age, cell_type_categories=list(reference.cell_type_categories), min_cells=config.min_cells, max_sinkhorn_cells=config.max_sinkhorn_cells, sinkhorn_blur=config.projection_sinkhorn_blur, sinkhorn_scaling=config.projection_sinkhorn_scaling, method=method, run_name=run_name, target_age=target_age, ct_assignment=assignment)
    if config.de_method == 'edger_pseudobulk':
        compute_per_cell_type_edger(rows=rows, de_logfc_diagnostics_fn=de_logfc_diagnostics, x_pred=batch.predicted_expression, x_true=batch.true_expression, pred_ct_codes=np.asarray(predicted, dtype=np.int64), true_ct_codes=np.asarray(batch.true_cell_types, dtype=np.int64), x_ref=reference.expression, ref_ages=reference.ages, ref_ct_codes=np.asarray(reference.cell_type_codes, dtype=np.int64), x_true_counts=batch.true_counts, true_sample_ids=batch.true_donors, x_ref_counts=reference.counts, ref_sample_ids=reference.donors, x_pred_counts=batch.predicted_counts, pred_sample_ids=batch.predicted_donors, actual_age=batch.actual_age, ct_categories=list(reference.cell_type_categories), gene_names=reference.gene_names, outdir=config.edger_dir, run_name=run_name, target_age=target_age, rscript=config.rscript, min_cells=config.min_cells)
    return rows

def evaluate_projection_batch(batch: ProjectionResult, *, reference: EvaluationReference, config: EvaluationConfig, method: str, run_name: str) -> dict[str, Any]:
    """Evaluate one normal-reference rollout prediction with shared metrics.

    ``full`` is the experiment/reporting contract. ``hpo_objective`` computes
    only held-out gene-expression Sinkhorn divergence.
    ``de_calibration`` emits direct logFC correlations and OLS calibration
    statistics. Both reduced profiles skip edgeR, per-cell-type and auxiliary
    DE diagnostics.
    """
    x_pred = np.asarray(batch.predicted_expression, dtype=np.float32)
    x_true = np.asarray(batch.true_expression, dtype=np.float32)
    if config.metric_profile == 'hpo_objective':
        return {'selection_score': sinkhorn_distance(x_true, x_pred, max_cells=config.max_sinkhorn_cells,
                    blur=config.projection_sinkhorn_blur, scaling=config.projection_sinkhorn_scaling),
                'metric_profile': 'hpo_objective', 'per_cell_type': []}
    x_ref = np.asarray(reference.expression, dtype=np.float32)
    ref_ages = np.asarray(reference.ages, dtype=np.float64)
    actual_age = float(batch.actual_age)
    target_age_int = int(round(float(batch.target_age)))
    gene_names = list(reference.gene_names)
    if config.metric_profile not in {'full', 'hpo_objective', 'de_calibration'}:
        raise ValueError(f'Unknown metric_profile={config.metric_profile!r}')
    young_mask = ref_ages < actual_age
    old_mask = ref_ages > actual_age
    de_young_diag = de_logfc_diagnostics(x_pred, x_true, x_ref[young_mask]) if int(young_mask.sum()) >= int(config.min_de_ref) else None
    de_old_diag = de_logfc_diagnostics(x_pred, x_true, x_ref[old_mask]) if int(old_mask.sum()) >= int(config.min_de_ref) else None
    if config.metric_profile == 'de_calibration':
        out: dict[str, Any] = {}
        _add_de_scalars(out, de_young_diag, de_old_diag, int(young_mask.sum()), int(old_mask.sum()))
        out['metric_profile'] = 'de_calibration'
        out['per_cell_type'] = []
        return out
    out = global_distribution_metrics(x_true, x_pred, mmd_block_size=config.mmd_block_size, sinkhorn_max_cells=config.max_sinkhorn_cells, sinkhorn_blur=config.projection_sinkhorn_blur, sinkhorn_scaling=config.projection_sinkhorn_scaling)
    _add_de_scalars(out, de_young_diag, de_old_diag, int(young_mask.sum()), int(old_mask.sum()))
    for contrast, de_diag, ref_mask in (('young', de_young_diag, young_mask), ('old', de_old_diag, old_mask)):
        edge_metrics, joined = edger_metrics_for_contrast(de_diag, contrast, batch, reference, ref_mask, config, run_name)
        out[f'edgeR_{contrast}'] = edge_metrics
        out.update(flatten_edger_scalars(edge_metrics, contrast=contrast))
        out.update(joined)
    per_cell_type = _evaluate_cell_types(batch, reference, config, out, method=method, run_name=run_name, target_age=target_age_int)
    out['metric_profile'] = 'full'
    out['per_cell_type'] = per_cell_type
    return out
