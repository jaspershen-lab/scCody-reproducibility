from __future__ import annotations
from collections.abc import Callable, Sequence
from typing import Any, Iterable
import numpy as np
import pandas as pd
SOURCE_DIRECTIONS = {'earlier', 'bidirectional'}

def source_age_allowed(source_age: float, target_age: float, direction: str) -> bool:
    """Return whether an age is eligible under the requested source direction."""
    if direction not in SOURCE_DIRECTIONS:
        raise ValueError(f'Unknown source direction {direction!r}; expected one of {sorted(SOURCE_DIRECTIONS)}')
    if direction == 'earlier':
        return float(source_age) < float(target_age)
    return True

def capped_eval_count(max_eval_cells: int, *population_sizes: int) -> int:
    if not population_sizes:
        return 0
    sizes = [max(0, int(size)) for size in population_sizes]
    cap = int(max_eval_cells)
    return min(sizes) if cap <= 0 else min(cap, *sizes)

def allocate_weighted_counts(total: int, weights: Sequence[float]) -> np.ndarray:
    total = max(0, int(total))
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) == 0:
        return np.zeros(0, dtype=np.int64)
    weights = np.clip(weights, 0.0, None)
    weights = weights / weights.sum() if weights.sum() else np.full(len(weights), 1.0 / len(weights))
    raw = total * weights
    counts = np.floor(raw).astype(np.int64)
    remainder = total - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw - np.floor(raw)), kind='stable')
        counts[order[:remainder]] += 1
    return counts

def aggregate_source_projection(entries: dict[float, np.ndarray], specs: Sequence[dict[str, Any]], target_age: float, project_one: Callable[[dict[str, Any], np.ndarray], dict[str, Any] | None]) -> dict[str, Any] | None:
    """Aggregate per-source age-grid projections using their protocol weights."""
    results = []
    for spec in specs:
        indices = entries[spec['source_age']][spec['source_locs']]
        result = project_one(spec, indices)
        if result is not None:
            results.append((result, float(spec['weight'])))
    sinkhorn_results = [(result, weight) for result, weight in results if np.isfinite(result.get('t_hat_sinkhorn', np.nan))]
    sinkhorn_weight = sum((weight for _result, weight in sinkhorn_results))
    if sinkhorn_weight <= 0 or len(sinkhorn_results) != len(specs):
        return None
    t_hat_sinkhorn = sum((result['t_hat_sinkhorn'] * weight for result, weight in sinkhorn_results)) / sinkhorn_weight
    best_sinkhorn = min(sinkhorn_results, key=lambda item: abs(item[0]['t_hat_sinkhorn'] - t_hat_sinkhorn))[0]
    sinkhorn_fields = ('sinkhorn_at_t_hat', 'sinkhorn_at_chrono', 'sinkhorn_improvement', 'n_sinkhorn_cells', 'sinkhorn_space', 'gene_space_sinkhorn_at_chrono', 'gene_space_sinkhorn_at_latent_t_hat', 'gene_space_sinkhorn_space', 'gene_space_sinkhorn_n_cells')
    result = {'delta_t_sinkhorn_raw': t_hat_sinkhorn - float(target_age), 't_hat_sinkhorn': t_hat_sinkhorn, **{field: best_sinkhorn.get(field) for field in sinkhorn_fields}, **{field: best_sinkhorn[field] for field in ('n_eval_cells', 'source_age') if field in best_sinkhorn}, 'n_source_specs': len(sinkhorn_results), 'age_estimate_primary': 'sinkhorn'}
    for field in ('model_native_latent_sinkhorn', 'decoded_gene_space_sinkhorn', 'forward_backward_direction', 'generated_cell_count', 'alive_generated_cell_count', 'survival_ratio', 'source_sde_step', 'target_sde_step', 'jax_seed', 'jax_key'):
        if field in best_sinkhorn:
            result[field] = best_sinkhorn[field]
    return result

def aggregate_pooled_source_age_projection(entries: dict[float, np.ndarray], donors_by_age: dict[float, dict[str, np.ndarray]], specs: Sequence[dict[str, Any]], target_age: float, project_one: Callable[[dict[str, Any], np.ndarray], dict[str, Any] | None], *, source_direction: str='earlier', records: list[dict[str, Any]] | None=None, record_metadata: dict[str, Any] | None=None, include_per_cell_type: bool=False) -> dict[str, Any] | None:
    """Project each source-age population once, matching scCody's per-source diagnostic unit.

    All eligible training cells at a selected source age are pooled before the
    age-grid sweep. The returned donor estimate is the unweighted mean of the
    valid source-age estimates, matching scCody's per-source estimate summary.
    """
    source_ages = sorted({float(spec['source_age']) for spec in specs if float(spec['source_age']) in entries and len(entries[float(spec['source_age'])])})
    if not source_ages or len(source_ages) != len({float(spec['source_age']) for spec in specs}):
        return None
    pooled_specs = [{'source_age': source_age, 'source_locs': np.arange(len(entries[source_age]), dtype=np.int64), 'source_donor': 'pooled_all_eligible_donors', 'weight': 1.0, 'replicate': 0} for source_age in source_ages]
    source_results: list[tuple[dict[str, Any], float]] = []

    def project_and_record(spec: dict[str, Any], indices: np.ndarray):
        result = project_one(spec, indices)
        if result is None:
            return None
        if np.isfinite(result.get('t_hat_sinkhorn', np.nan)):
            source_results.append((result, 1.0))
        if records is not None:
            source_age = float(spec['source_age'])
            t_hat_sinkhorn = float(result.get('t_hat_sinkhorn', np.nan))
            donor_map = donors_by_age.get(source_age, {})
            records.append({**dict(record_metadata or {}), **result, 'true_age': float(target_age), 'source_age': source_age, 'source_age_distance': float(target_age) - source_age, 't_hat': t_hat_sinkhorn, 'age_prediction_abs_error_sinkhorn': abs(t_hat_sinkhorn - float(target_age)) if np.isfinite(t_hat_sinkhorn) else np.nan, 'abs_error': abs(t_hat_sinkhorn - float(target_age)) if np.isfinite(t_hat_sinkhorn) else np.nan, 'n_source_cells_available': int(len(indices)), 'n_source_donors': int(len(donor_map)), 'source_population': 'pooled_all_eligible_donors_at_source_age'})
        return result
    aggregate = aggregate_source_projection(entries, pooled_specs, target_age, project_and_record)
    if aggregate is None or not include_per_cell_type:
        return aggregate
    per_cell_type: dict[str, list[tuple[dict[str, Any], float]]] = {}
    for result, weight in source_results:
        for row in result.get('per_cell_type', []):
            cell_type = str(row.get('cell_type', ''))
            if cell_type and np.isfinite(row.get('t_hat_sinkhorn', np.nan)):
                per_cell_type.setdefault(cell_type, []).append((row, weight))
    aggregate['per_cell_type'] = []
    for cell_type, values in sorted(per_cell_type.items()):
        total_weight = sum((weight for _row, weight in values))
        if total_weight <= 0 or len(values) != len(source_ages):
            continue
        best_row = min((row for row, _weight in values), key=lambda row: float(row['t_hat_sinkhorn']))
        t_hat = sum((float(row['t_hat_sinkhorn']) * weight for row, weight in values)) / total_weight
        aggregate['per_cell_type'].append({**best_row, 'cell_type': cell_type, 't_hat_sinkhorn': t_hat, 'delta_t_sinkhorn_raw': t_hat - float(target_age), 'age_prediction_abs_error_sinkhorn': abs(t_hat - float(target_age)), 'n_source_specs': int(len(values))})
    return aggregate

def rollout_source_ensemble(entries: dict[float, np.ndarray], specs: Sequence[dict[str, Any]], target_age: float, n_cells: int, rng: np.random.Generator, project_one: Callable[[float, np.ndarray, int], np.ndarray | tuple[np.ndarray, np.ndarray]], *, source_direction: str='earlier', source_donors_by_age: dict[float, dict[str, np.ndarray]] | None=None, return_source_donors: bool=False, return_source_ages: bool=False) -> np.ndarray | tuple[np.ndarray, np.ndarray] | None:
    """Allocate cells across valid chronological sources and concatenate projections."""
    valid = []
    for spec in specs:
        source_age = float(spec['source_age'])
        indices = entries[source_age][spec['source_locs']]
        if source_age_allowed(source_age, target_age, source_direction) and len(indices):
            valid.append((spec, source_age, indices))
    if not valid or len(valid) != len(specs):
        return None
    counts = allocate_weighted_counts(n_cells, [max(0.0, float(spec.get('weight', 0.0))) for spec, _age, _idx in valid])
    if np.any(counts <= 0):
        return None
    chunks = []
    donor_chunks = []
    source_age_chunks = []
    for (_spec, source_age, indices), count in zip(valid, counts):
        if count <= 0:
            continue
        selected = rng.choice(indices, size=count, replace=len(indices) < count)
        projected = project_one(source_age, selected, int(count))
        output_source_positions = None
        if isinstance(projected, tuple):
            if len(projected) != 2:
                raise ValueError('project_one tuple results must be (prediction, source_positions)')
            projected, output_source_positions = projected
        if len(projected) == 0 or not np.isfinite(projected).all():
            return None
        chunks.append(projected)
        if return_source_ages:
            source_age_chunks.append(np.full(len(projected), float(source_age), dtype=np.float32))
        if return_source_donors:
            if source_donors_by_age is None:
                raise ValueError('source_donors_by_age is required when returning source donors')
            donor_map = source_donors_by_age.get(float(source_age), {})
            global_to_donor = {}
            source_indices = np.asarray(entries[float(source_age)], dtype=np.int64)
            for donor, donor_locations in donor_map.items():
                for location in np.asarray(donor_locations, dtype=np.int64):
                    global_to_donor[int(source_indices[int(location)])] = str(donor)
            donor_indices = selected if output_source_positions is None else selected[np.asarray(output_source_positions, dtype=np.int64)]
            donor_chunks.append(np.asarray([global_to_donor.get(int(index), 'unknown_source_donor') for index in donor_indices], dtype=str))
    if not chunks:
        return None
    predicted = np.concatenate(chunks, axis=0)
    if return_source_donors:
        donors = np.concatenate(donor_chunks, axis=0)
        if return_source_ages:
            return (predicted, donors, np.concatenate(source_age_chunks, axis=0))
        return (predicted, donors)
    if return_source_ages:
        return (predicted, np.concatenate(source_age_chunks, axis=0))
    return predicted

def groups_for_target_age(payload: dict[str, Any], target_age: float) -> list[dict[str, Any]]:
    return [group for group in payload['groups'] if abs(float(group['target_age']) - float(target_age)) < 1e-06]

def observation_match_mask(obs: pd.DataFrame, match: dict[str, Any]) -> np.ndarray:
    mask = np.ones(len(obs), dtype=bool)
    for key, value in match.items():
        if key not in obs:
            raise KeyError(f'Matched rollout key {key!r} is absent from adata.obs')
        mask &= obs[key].astype(str).to_numpy() == str(value)
    return mask

def group_for_target_values(payload: dict[str, Any], *, target_age: float, values: dict[str, Any]) -> dict[str, Any] | None:
    matches = [group for group in groups_for_target_age(payload, target_age) if all((str(group['match'][key]) == str(values[key]) for key in payload['match_by']))]
    if len(matches) > 1:
        raise RuntimeError(f'Multiple matched-source groups apply to age={target_age:g}, values={values}')
    return matches[0] if matches else None

def target_group_indices(obs: pd.DataFrame, indices: Iterable[int], group: dict[str, Any]) -> np.ndarray:
    indices = np.asarray(list(indices), dtype=np.int64)
    return indices[observation_match_mask(obs.iloc[indices], group['match'])]

def matched_source_indices(obs: pd.DataFrame, train_indices: Iterable[int], *, age_key: str, group: dict[str, Any]) -> np.ndarray:
    train_indices = np.asarray(list(train_indices), dtype=np.int64)
    subset = obs.iloc[train_indices]
    ages = pd.to_numeric(subset[age_key], errors='coerce').to_numpy(dtype=float)
    mask = observation_match_mask(subset, group['match'])
    mask &= np.isin(ages, np.asarray(group.get('source_ages', []), dtype=float))
    return train_indices[mask]
