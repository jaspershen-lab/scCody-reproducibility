"""Evaluation-reference sampling for trained dynamics models."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class EvaluationReference:
    expression: np.ndarray
    counts: np.ndarray | None
    ages: np.ndarray
    cell_type_codes: np.ndarray | None
    donors: np.ndarray | None
    gene_names: tuple[str, ...]
    cell_type_categories: tuple[str, ...] = ()
    cell_type_classifier: object | None = None

@dataclass(frozen=True)
class ProjectionResult:
    """Canonical projection output for evaluation and tabular APIs."""
    target_age: float | None = None
    actual_age: float | None = None
    predicted_expression: np.ndarray | None = None
    true_expression: np.ndarray | None = None
    predicted_counts: np.ndarray | None = None
    true_counts: np.ndarray | None = None
    predicted_cell_types: np.ndarray | None = None
    true_cell_types: np.ndarray | None = None
    predicted_donors: np.ndarray | None = None
    true_donors: np.ndarray | None = None
    values: dict[str, object] | None = None
from pathlib import Path
from typing import Any, Callable
import pandas as pd
from sccody_analyses.benchmarks.engine.evaluation.config import EvaluationConfig
from sccody_analyses.benchmarks.engine.projection.source_pools import SourcePoolResolver

@dataclass(frozen=True)
class _PredictionContext:
    normal_holdout: Any
    entries: Any
    donors_by_age: Any
    normal_train: Any
    settings: Any
    rng: np.random.Generator
    run_name: str
    method: str
    rollout: Callable[..., Any]
    prediction_records: list[dict[str, Any]] | None = None
    source_resolver: SourcePoolResolver | None = None
    target_age: float = 0.0
    eval_idx: np.ndarray | None = None
    true_counts: np.ndarray | None = None
    true_donors: np.ndarray | None = None
    true_cell_types: np.ndarray | None = None
    cell_type_categories: list[str] | None = None

@dataclass(frozen=True)
class _TargetPrediction:
    predicted: np.ndarray
    true: np.ndarray
    true_counts: np.ndarray | None
    true_donors: np.ndarray
    true_cell_types: np.ndarray
    predicted_cell_types: np.ndarray | None
    true_obs_names: np.ndarray
    predicted_donors: np.ndarray | None = None
    predicted_source_ages: np.ndarray | None = None

def _matched_target_prediction(context: _PredictionContext) -> _TargetPrediction | None:
    from sccody_analyses.benchmarks.engine.methods.dynamics import get_count_x_or_none, get_expr, matched_entries_for_group, str_array
    from sccody_analyses.benchmarks.engine.projection.rollout import groups_for_target_age, target_group_indices
    args = context.settings
    if args.projection_unit != 'donor':
        raise ValueError('Matched-source evaluation requires --projection-unit donor')
    predictions, predicted_donors, predicted_source_ages, truths, counts, donors, cell_types, obs_names = ([], [], [], [], [], [], [], [])
    assigned = np.zeros(len(context.eval_idx), dtype=bool)
    for group in groups_for_target_age(args.matched_source_manifest, context.target_age):
        group_idx = target_group_indices(context.normal_holdout.obs, context.eval_idx, group)
        if not len(group_idx):
            continue
        local_mask = np.isin(context.eval_idx, group_idx)
        if np.any(assigned & local_mask):
            raise RuntimeError(f"Overlapping matched-source group {group['id']}")
        assigned |= local_mask
        entries, donors_by_age, group_args = matched_entries_for_group(context.normal_train, group, args)
        if not entries:
            print(f"[matched-source unsupported] {group['id']} target_cells={len(group_idx)}", flush=True)
            continue
        rollout_result = context.rollout(context.target_age, entries, donors_by_age, group_args, len(group_idx), context.rng)
        if rollout_result is None:
            continue
        if len(rollout_result) != 3:
            raise RuntimeError('Matched rollout did not return source-age labels')
        predicted, source_donors, source_ages = rollout_result
        if len(predicted) < 2:
            continue
        predictions.append(predicted)
        predicted_donors.append(np.asarray(source_donors, dtype=str))
        predicted_source_ages.append(np.asarray(source_ages, dtype=np.float32))
        truths.append(get_expr(context.normal_holdout, group_idx, args))
        group_counts = get_count_x_or_none(context.normal_holdout, group_idx, args)
        if group_counts is not None:
            counts.append(group_counts)
        donors.append(context.normal_holdout.obs[args.donor_key].astype(str).to_numpy()[group_idx])
        cell_types.append(context.normal_holdout.obs['_ct_code'].to_numpy(dtype=np.int64)[group_idx])
        obs_names.append(str_array(context.normal_holdout.obs_names[group_idx]))
    if not assigned.all():
        raise RuntimeError(f'Matched-source manifest did not assign {int((~assigned).sum())} cells at target age {context.target_age:g}')
    if not predictions:
        return None
    return _TargetPrediction(np.concatenate(predictions), np.concatenate(truths), np.concatenate(counts) if counts else None, np.concatenate(donors), np.concatenate(cell_types), None, np.concatenate(obs_names), np.concatenate(predicted_donors), np.concatenate(predicted_source_ages))

def _cell_type_target_prediction(context: _PredictionContext) -> _TargetPrediction | None:
    from sccody_analyses.benchmarks.engine.methods.dynamics import get_expr, str_array
    args = context.settings
    predictions, predicted_donors, predicted_source_ages, truths, predicted_types, true_types, counts, donors, obs_names = ([], [], [], [], [], [], [], [], [])
    for code in sorted(set(context.true_cell_types.tolist())):
        group_idx = context.eval_idx[context.true_cell_types == code]
        if len(group_idx) < args.min_eval_cells_per_ct:
            continue
        name = context.cell_type_categories[int(code)] if 0 <= int(code) < len(context.cell_type_categories) else str(code)
        pool = context.source_resolver.for_cell_type(name)
        rollout_result = context.rollout(context.target_age, pool.entries, pool.donors_by_age, args, len(group_idx), context.rng)
        if rollout_result is None:
            continue
        if len(rollout_result) != 3:
            raise RuntimeError('Cell-type rollout did not return source-age labels')
        predicted, source_donors, source_ages = rollout_result
        if len(predicted) < 2:
            continue
        true = get_expr(context.normal_holdout, group_idx, args)
        predictions.append(predicted)
        predicted_donors.append(np.asarray(source_donors, dtype=str))
        predicted_source_ages.append(np.asarray(source_ages, dtype=np.float32))
        truths.append(true)
        predicted_types.append(np.full(len(predicted), int(code), dtype=np.int64))
        true_types.append(np.full(len(true), int(code), dtype=np.int64))
        obs_names.append(str_array(context.normal_holdout.obs_names[group_idx]))
        if context.true_counts is not None:
            positions = np.flatnonzero(context.true_cell_types == code)
            counts.append(context.true_counts[positions])
            donors.append(context.true_donors[positions])
    if not predictions:
        return None
    return _TargetPrediction(np.concatenate(predictions), np.concatenate(truths), np.concatenate(counts) if counts else None, np.concatenate(donors) if donors else context.true_donors, np.concatenate(true_types), np.concatenate(predicted_types), np.concatenate(obs_names), np.concatenate(predicted_donors), np.concatenate(predicted_source_ages))

def _global_target_prediction(context: _PredictionContext) -> _TargetPrediction | None:
    from sccody_analyses.benchmarks.engine.methods.dynamics import get_expr, str_array
    args = context.settings
    rollout_result = context.rollout(context.target_age, context.entries, context.donors_by_age, args, len(context.eval_idx), context.rng)
    if rollout_result is None:
        return None
    if len(rollout_result) != 3:
        raise RuntimeError('Global rollout did not return source-age labels')
    predicted, source_donors, source_ages = rollout_result
    if len(predicted) < 2:
        return None
    return _TargetPrediction(predicted, get_expr(context.normal_holdout, context.eval_idx, args), context.true_counts, context.true_donors, context.true_cell_types, None, str_array(context.normal_holdout.obs_names[context.eval_idx]), np.asarray(source_donors, dtype=str), np.asarray(source_ages, dtype=np.float32))

def _append_prediction_record(records: list[dict[str, Any]] | None, prediction: _TargetPrediction, target_age: float, categories: list[str]) -> None:
    from sccody_analyses.benchmarks.engine.methods.dynamics import str_array
    if records is None:
        return
    names = np.full(len(prediction.predicted), 'global', dtype='<U32') if prediction.predicted_cell_types is None else str_array((categories[int(code)] if 0 <= int(code) < len(categories) else code for code in prediction.predicted_cell_types))
    source_donors = prediction.predicted_donors
    if source_donors is None or len(source_donors) != len(prediction.predicted):
        raise RuntimeError('Dynamics prediction is missing aligned source-donor metadata')
    source_ages = prediction.predicted_source_ages
    if source_ages is None or len(source_ages) != len(prediction.predicted):
        raise RuntimeError('Dynamics prediction is missing aligned source-age metadata')
    records.append({'x_pred': np.asarray(prediction.predicted, dtype=np.float32), 'target_age': np.full(len(prediction.predicted), target_age, dtype=np.float32), 'actual_age': np.full(len(prediction.predicted), target_age, dtype=np.float32), 'pred_cell_type': names, 'pred_source_donor': np.asarray(source_donors, dtype=str), 'pred_source_age': np.asarray(source_ages, dtype=np.float32), 'holdout_obs_names': prediction.true_obs_names})

def evaluate_calibration_rollout(*, normal_holdout, entries, donors_by_age, normal_train, settings, rng, run_name, method, rollout, prediction_records=None, de_reference_indices=None, de_reference_adata=None):
    """Evaluate internal-validation rollouts with the requested metric profile."""
    from sccody_analyses.benchmarks.engine.evaluation.metrics import evaluate_projection_batch
    from sccody_analyses.benchmarks.engine.methods.dynamics import get_count_x_or_none, get_expr, target_age_allowed
    from sccody_analyses.benchmarks.engine.utilities.parsing import parse_float_targets
    args = settings
    holdout_ages = normal_holdout.obs[args.age_key].astype(float).to_numpy()
    holdout_ct_codes = normal_holdout.obs['_ct_code'].to_numpy(dtype=np.int64)
    holdout_donors = normal_holdout.obs[args.donor_key].astype(str).to_numpy()
    if de_reference_indices is None or de_reference_adata is None:
        raise ValueError('Calibration DE evaluation requires the dataset-configured sc_ref')
    reference_adata = de_reference_adata
    train_ages = reference_adata.obs[args.age_key].astype(float).to_numpy()
    ref_idx = np.asarray(de_reference_indices, dtype=np.int64)
    x_ref, x_ref_counts = (get_expr(reference_adata, ref_idx, args), get_count_x_or_none(reference_adata, ref_idx, args))
    ref_ages = train_ages[ref_idx]
    ref_ct_codes = reference_adata.obs['_ct_code'].to_numpy(dtype=np.int64)[ref_idx]
    ref_donors = reference_adata.obs[args.donor_key].astype(str).to_numpy()[ref_idx]
    categories, gene_names = (list(map(str, reference_adata.uns.get('_ct_categories', []))), list(map(str, reference_adata.var_names)))
    classifier = None
    if args.projection_unit == 'donor':
        from sccody_analyses.benchmarks.engine.evaluation.metrics import fit_pca_knn_cell_type
        classifier = fit_pca_knn_cell_type(x_ref, ref_ct_codes)
    edger_dir = Path(args.edger_dir) if args.edger_dir else Path(args.outdir or '.') / f'{run_name}_edger'
    source_resolver = SourcePoolResolver(normal_train=normal_train, entries=entries, donors_by_age=donors_by_age, projection_unit=args.projection_unit, cell_type_key=args.cell_type_key, donor_key=args.donor_key, min_cells=args.min_normal_cells)
    targets = parse_float_targets(args.calibration_targets)
    rows = []
    for target_age in [age for age in sorted(np.unique(holdout_ages)) if target_age_allowed(age, targets)]:
        holdout_idx = np.flatnonzero(holdout_ages == target_age)
        if len(holdout_idx) < args.min_calibration_cells:
            continue
        n_eval = min(args.max_eval_cells, len(holdout_idx)) if args.max_eval_cells > 0 else len(holdout_idx)
        from sccody_analyses.population_inputs import target_rows
        eval_idx = target_rows(normal_holdout.obs, args.age_key, [target_age], args.max_eval_cells, args.seed,
                               tuple(key for key in args.reference_stratify_by.split(',') if key))
        context = _PredictionContext(normal_holdout=normal_holdout, entries=entries, donors_by_age=donors_by_age, normal_train=normal_train, settings=args, rng=rng, run_name=run_name, method=method, rollout=rollout, prediction_records=prediction_records, source_resolver=source_resolver, target_age=float(target_age), eval_idx=eval_idx, true_counts=get_count_x_or_none(normal_holdout, eval_idx, args), true_donors=holdout_donors[eval_idx], true_cell_types=holdout_ct_codes[eval_idx], cell_type_categories=categories)
        prediction = _matched_target_prediction(context) if args.matched_source_manifest is not None else _cell_type_target_prediction(context) if args.projection_unit == 'donor_cell_type' else _global_target_prediction(context)
        if prediction is None:
            continue
        _append_prediction_record(prediction_records, prediction, float(target_age), categories)
        metrics = evaluate_projection_batch(ProjectionResult(target_age=float(target_age), actual_age=float(target_age), predicted_expression=prediction.predicted, true_expression=prediction.true, true_counts=prediction.true_counts, true_cell_types=prediction.true_cell_types, true_donors=prediction.true_donors, predicted_cell_types=prediction.predicted_cell_types, predicted_donors=prediction.predicted_donors), reference=EvaluationReference(expression=x_ref, counts=x_ref_counts, ages=ref_ages, cell_type_codes=ref_ct_codes, donors=ref_donors, gene_names=tuple(gene_names), cell_type_categories=tuple(categories), cell_type_classifier=classifier), config=EvaluationConfig(metric_profile=args.metric_profile, de_method=args.de_method, min_cells=args.min_eval_cells_per_ct, rscript=args.rscript, edger_dir=edger_dir, mmd_block_size=args.mmd_block_size, min_de_ref=50), method=method, run_name=run_name)
        rows.append({'method': method, 'run_name': run_name, 'target_age': int(round(float(target_age))),
                     'actual_holdout_age': float(target_age), 'n_holdout_cells': int(len(holdout_idx)),
                     'n_eval_cells_matched': int(len(prediction.predicted)),
                     'per_cell_type': metrics.pop('per_cell_type', []), **metrics})
    if not rows:
        raise RuntimeError('No calibration metric rows were produced')
    return rows

def calibration_summary(calibration, args):
    rows = []
    if calibration.empty or 'delta_t_sinkhorn_raw' not in calibration:
        return pd.DataFrame(rows)
    age_bins = pd.cut(calibration['chronological_age'], bins=np.arange(0, 200, args.projection_age_step * 5))
    grouped = calibration.groupby([calibration['cell_type'].astype(str), age_bins], observed=True) if 'cell_type' in calibration else calibration.groupby(age_bins, observed=True)
    for key, group in grouped:
        cell_type, age = key if 'cell_type' in calibration else (None, key)
        row = {'age_group_mid': float(age.mid), 'age_group_lo': float(age.left), 'age_group_hi': float(age.right), 'n_calibration_donors': int(len(group))}
        if cell_type is not None:
            row['cell_type'] = str(cell_type)
        has_metric = False
        for metric in ('sinkhorn',):
            column = f'delta_t_{metric}_raw'
            values = group[column].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            has_metric = True
            median_delta, median_abs, mean_abs = (float(np.median(values)), float(np.median(np.abs(values))), float(np.mean(np.abs(values))))
            row.update({f'median_delta_t_bias_{metric}': median_delta, f'median_abs_delta_t_{metric}': median_abs, f'mean_abs_delta_t_{metric}': mean_abs})
            row.update({'median_delta_t_bias': median_delta, 'median_abs_delta_t': median_abs, 'mean_abs_delta_t': mean_abs})
        if has_metric:
            rows.append(row)
    return pd.DataFrame(rows)

def project_calibration_rollout(normal_holdout, entries, donors_by_age, normal_train, age_grid, args, rng, *, project, projection_records=None):
    """Project internal-validation groups and summarize age bias."""
    from sccody_analyses.benchmarks.engine.projection.rollout import group_for_target_values
    from sccody_analyses.benchmarks.engine.methods.dynamics import get_expr, iter_projection_targets, matched_entries_for_group, target_age_allowed
    from sccody_analyses.benchmarks.engine.utilities.parsing import parse_float_targets
    ages = normal_holdout.obs[args.age_key].astype(float).to_numpy()
    donors = normal_holdout.obs[args.donor_key].astype(str).to_numpy()
    targets = parse_float_targets(args.calibration_targets)
    snapshot_targets = parse_float_targets(getattr(args, 'projection_snapshot_target_ages', ''))
    snapshot_counts = {}
    resolver = SourcePoolResolver(normal_train=normal_train, entries=entries, donors_by_age=donors_by_age, projection_unit=args.projection_unit, cell_type_key=args.cell_type_key, donor_key=args.donor_key, min_cells=args.min_normal_cells)
    rows = []
    for donor, target_age, target_idx, cell_type in iter_projection_targets(normal_holdout, ages, donors, args, args.min_calibration_cells):
        if not target_age_allowed(target_age, targets):
            continue
        if projection_records is not None and getattr(args, 'projection_snapshot_only', False) and (int(getattr(args, 'projection_snapshot_max_rows', 0)) > 0) and (len(projection_records) >= int(args.projection_snapshot_max_rows)):
            break
        matched_group = None
        projection_args = args
        if args.matched_source_manifest is not None:
            values = {key: str(normal_holdout.obs.iloc[int(target_idx[0])][key]) for key in args.matched_source_manifest['match_by']}
            matched_group = group_for_target_values(args.matched_source_manifest, target_age=target_age, values=values)
            if matched_group is None or not matched_group.get('source_ages'):
                continue
            target_entries, target_donors, projection_args = matched_entries_for_group(normal_train, matched_group, args, cell_type=cell_type if args.projection_unit == 'donor_cell_type' else None)
            setattr(projection_args, '_artemis_field_key', ('matched', tuple(sorted(((str(key), str(value)) for key, value in matched_group['match'].items()))), str(getattr(projection_args, 'source_direction', 'earlier'))))
        else:
            pool = resolver.for_cell_type(cell_type)
            target_entries, target_donors = (pool.entries, pool.donors_by_age)
            setattr(projection_args, '_artemis_field_key', ('all', str(cell_type) if cell_type is not None else 'donor'))
        if len(target_entries) < (1 if matched_group is not None else 2):
            continue
        selected_records = None
        if projection_records is not None:
            tolerance = max(float(getattr(args, 'projection_snapshot_age_tolerance', 0)), 0)
            within_age = not snapshot_targets or any((abs(float(target_age) - target) <= tolerance for target in snapshot_targets))
            within_rows = int(getattr(args, 'projection_snapshot_max_rows', 0)) <= 0 or len(projection_records) < int(args.projection_snapshot_max_rows)
            age_key = float(min(snapshot_targets, key=lambda age: abs(age - float(target_age)))) if snapshot_targets else float(target_age)
            age_cap = int(getattr(args, 'projection_snapshot_rows_per_age', 0))
            if within_age and within_rows and (age_cap <= 0 or snapshot_counts.get(age_key, 0) < age_cap):
                selected_records = projection_records
                snapshot_counts[age_key] = snapshot_counts.get(age_key, 0) + 1
        result = project(get_expr(normal_holdout, target_idx, args), target_age, target_entries, target_donors, projection_args, rng, selected_records, {'label': str(donor), 'group': str(cell_type) if cell_type is not None else 'donor', 'projection_kind': 'normal_holdout_calibration'}, target_cell_types=normal_holdout.obs['_ct_code'].to_numpy(dtype=np.int64)[target_idx])
        if result is None:
            continue
        update = {'donor_id': str(donor), 'chronological_age': target_age, 'projection_kind': 'normal_holdout_calibration', 'projection_unit': args.projection_unit, 'n_holdout_cells': int(len(target_idx)), 'matched_source_group': matched_group['id'] if matched_group is not None else None}
        for metric in ('sinkhorn',):
            value = result.get(f't_hat_{metric}', np.nan)
            if np.isfinite(value):
                update[f'age_prediction_abs_error_{metric}'] = abs(float(value) - target_age)
        if cell_type is not None:
            update['cell_type'] = str(cell_type)
        result.update(update)
        if args.cell_type_readout == 'pca_knn':
            for cell_type_result in result.pop('per_cell_type', []):
                rows.append({**result, **cell_type_result, 'projection_unit': 'donor_pca_knn_cell_type', 'projection_readout': 'pca_knn', 'donor_id': str(donor), 'chronological_age': target_age, 'n_holdout_cells': int(len(target_idx))})
        else:
            rows.append(result)
    calibration = pd.DataFrame(rows)
    return (calibration, calibration_summary(calibration, args))

def project_disease_rollout(*, disease, entries, donors_by_age, normal_train, settings, rng, disease_label, method, project):
    """Project disease groups without healthy-cohort bias correction."""
    from sccody_analyses.benchmarks.engine.methods.dynamics import get_expr, iter_projection_targets
    args = settings
    ages = disease.obs[args.age_key].astype(float).to_numpy()
    donors = disease.obs[args.donor_key].astype(str).to_numpy()
    resolver = SourcePoolResolver(normal_train=normal_train, entries=entries, donors_by_age=donors_by_age, projection_unit=args.projection_unit, cell_type_key=args.cell_type_key, donor_key=args.donor_key, min_cells=args.min_normal_cells)
    rows = []
    for donor, target_age, target_idx, cell_type in iter_projection_targets(disease, ages, donors, args, args.min_disease_cells):
        pool = resolver.for_cell_type(cell_type)
        if len(pool.entries) < 2:
            continue
        result = project(get_expr(disease, target_idx, args), target_age, pool.entries, pool.donors_by_age, args, rng, None, None)
        if result is None:
            continue
        update = {'method': method, 'projection_kind': f'disease_donor_{method.lower()}', 'projection_unit': args.projection_unit, 'target_label': disease_label, 'disease': disease_label, 'donor_id': str(donor), 'chronological_age': target_age, 'target_age': target_age, 'n_disease_cells': int(len(target_idx)), 'eval_source_policy': args.eval_source_policy, 'rollout_source_policy': args.rollout_source_policy}
        if cell_type is not None:
            update['cell_type'] = str(cell_type)
            update['projection_kind'] = f'disease_donor_cell_type_{method.lower()}'
        result.update(update)
        rows.append(result)
    return pd.DataFrame(rows)
