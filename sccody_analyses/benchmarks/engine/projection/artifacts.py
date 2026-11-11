from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np

def string_array(values: Sequence[object]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype='U')

def save_holdout_predictions(records: Sequence[Mapping[str, Any]], *, path: str | Path, train_obs_names: Sequence[object], gene_names: Sequence[object]) -> Path:
    """Write the shared scNODE/scDiffEq holdout-prediction NPZ contract."""
    if not records:
        raise RuntimeError('Holdout prediction export was requested, but no predictions were collected')
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    predicted = np.concatenate([record['x_pred'] for record in records], axis=0)
    predicted_donors = np.concatenate([record['pred_source_donor'] for record in records], axis=0)
    if any(('pred_source_age' not in record for record in records)):
        raise RuntimeError('Holdout prediction export is missing per-cell source-age labels')
    predicted_source_ages = np.concatenate([record['pred_source_age'] for record in records], axis=0).astype(np.float32, copy=False)
    if len(predicted_source_ages) != len(predicted):
        raise RuntimeError('Holdout prediction source-age metadata is not aligned with generated cells')
    np.savez_compressed(output, x_pred=predicted.astype(np.float32), target_age=np.concatenate([record['target_age'] for record in records], axis=0).astype(np.float32), actual_age=np.concatenate([record['actual_age'] for record in records], axis=0).astype(np.float32), pred_cell_type=np.concatenate([record['pred_cell_type'] for record in records], axis=0), pred_source_donor=predicted_donors, pred_source_age=predicted_source_ages, holdout_obs_names=np.concatenate([record['holdout_obs_names'] for record in records], axis=0), train_obs_names=string_array(train_obs_names), gene_names=string_array(gene_names))
    return output

def build_calibration_snapshot_arrays(records: Sequence[Mapping[str, Any]], *, include_expression: bool, extra_string_fields: Sequence[str]=()) -> dict[str, np.ndarray]:
    if not records:
        raise RuntimeError('Projection snapshot export was requested, but no snapshots were collected')

    def concatenate(name: str) -> np.ndarray:
        return np.concatenate([record[name] for record in records], axis=0)

    def repeat_float(name: str) -> np.ndarray:
        return np.concatenate([np.full(record['target_latent'].shape[0], float(record[name]), dtype=np.float32) for record in records])

    def repeat_string(name: str) -> np.ndarray:
        return string_array([record[name] for record in records for _ in range(record['target_latent'].shape[0])])
    arrays = {'target_latent': concatenate('target_latent').astype(np.float32), 'pred_chrono_latent': concatenate('pred_chrono_latent').astype(np.float32), 'pred_sinkhorn_latent': concatenate('pred_sinkhorn_latent').astype(np.float32), 'target_age': repeat_float('target_age'), 'source_age': repeat_float('source_age'), 't_hat_sinkhorn': repeat_float('t_hat_sinkhorn'), 'label': repeat_string('label'), 'group': repeat_string('group'), 'projection_kind': repeat_string('projection_kind')}
    for field in extra_string_fields:
        arrays[field] = repeat_string(field)
    if include_expression:
        arrays.update({'target_expr': concatenate('target_expr').astype(np.float32), 'pred_chrono_expr': concatenate('pred_chrono_expr').astype(np.float32), 'pred_sinkhorn_expr': concatenate('pred_sinkhorn_expr').astype(np.float32)})
    optional_arrays = ('pred_all_particle_chrono_latent', 'pred_all_particle_sinkhorn_latent', 'pred_all_particle_chrono_expr', 'pred_all_particle_sinkhorn_expr')
    if all((field in records[0] for field in optional_arrays)):
        arrays.update({field: concatenate(field).astype(np.float32) for field in optional_arrays})
    return arrays

def save_projection_snapshots(records: Sequence[Mapping[str, Any]], *, path: str | Path, include_expression: bool=True, extra_string_fields: Sequence[str]=(), extra_arrays: Mapping[str, np.ndarray] | None=None) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = build_calibration_snapshot_arrays(records, include_expression=include_expression, extra_string_fields=extra_string_fields)
    arrays.update(dict(extra_arrays or {}))
    np.savez_compressed(output, **arrays)
    return output
