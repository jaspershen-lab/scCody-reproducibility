from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
from sccody_analyses.benchmarks.engine.dynamics.identity import BenchmarkMethodIdentity

def build_checkpoint_envelope(identity: BenchmarkMethodIdentity, *, args: Any, normal_train: Any, train_ages: Iterable[float], age_min: float, age_max: float, selection: Mapping[str, Any] | None) -> dict[str, Any]:
    ages = [float(age) for age in train_ages]
    return {'method': identity.method_id, 'format_version': 1, 'n_genes': int(normal_train.n_vars), 'var_names': [str(value) for value in normal_train.var_names], 'layer': str(args.layer), 'age_key': str(args.age_key), 'cell_type_key': str(args.cell_type_key), 'donor_key': str(args.donor_key), 'train_ages': ages, 'age_min': float(age_min), 'age_max': float(age_max), 'time_normalization': str(args.time_normalization), 'time_origin': float(0.0 if args.time_normalization == 'age_100' else age_min), 'time_scale': float(100.0 if args.time_normalization == 'age_100' else age_max - age_min), 'seed': int(args.seed), 'checkpoint_selection': dict(selection or {})}

def validate_checkpoint_envelope(payload: Mapping[str, Any], *, path: str | Path, identity: BenchmarkMethodIdentity, n_genes: int, var_names: Sequence[object], required_keys: Iterable[str]=()) -> None:
    if payload.get('method') != identity.method_id:
        raise ValueError(f'Checkpoint {path} is not a {identity.display_name} checkpoint')
    if int(payload.get('n_genes', -1)) != int(n_genes):
        raise ValueError(f"Checkpoint n_genes={payload.get('n_genes')} does not match current data n_genes={n_genes}")
    checkpoint_vars = payload.get('var_names')
    if checkpoint_vars is not None and list(map(str, checkpoint_vars)) != list(map(str, var_names)):
        raise ValueError('Checkpoint feature order does not match current AnnData var_names')
    for key in required_keys:
        if key not in payload:
            raise ValueError(f'Checkpoint {path} is missing {key}')

def validate_checkpoint_time_axis(payload: Mapping[str, Any], *, requested_normalization: str) -> None:
    checkpoint_normalization = str(payload.get('time_normalization', 'fit_minmax'))
    if checkpoint_normalization != requested_normalization:
        raise ValueError(f'Checkpoint time normalization does not match this run: checkpoint={checkpoint_normalization}, requested={requested_normalization}')

def apply_cross_dataset_age_domain(source_plan: Any, checkpoint: Mapping[str, Any], *, age_key: str, projection_age_step: float, projection_max_age: float) -> None:
    """Restrict foreign-cohort sources to fitted support and allow inference extrapolation."""
    checkpoint_min = float(checkpoint['age_min'])
    checkpoint_max = float(checkpoint['age_max'])
    checkpoint_time_normalization = str(checkpoint.get('time_normalization', 'fit_minmax'))
    requested_time_normalization = str(getattr(source_plan, 'time_normalization', 'fit_minmax'))
    if checkpoint_time_normalization != requested_time_normalization:
        raise ValueError(f'Evaluation time normalization does not match checkpoint: checkpoint={checkpoint_time_normalization}, requested={requested_time_normalization}')
    if checkpoint_max < checkpoint_min:
        raise ValueError(f'Empty checkpoint age domain [{checkpoint_min:g}, {checkpoint_max:g}]')
    entries = {float(age): indices for age, indices in source_plan.entries.items() if checkpoint_min <= float(age) <= checkpoint_max}
    if len(entries) < 2:
        raise RuntimeError(f'Need at least two foreign-cohort source ages inside checkpoint fitted domain [{checkpoint_min:g}, {checkpoint_max:g}]')
    source_plan.entries = entries
    source_plan.donors_by_age = {age: source_plan.donors_by_age[age] for age in entries}
    source_plan.sorted_ages = sorted(entries)
    source_plan.age_min = checkpoint_min
    source_plan.age_max = checkpoint_max
    source_plan.time_normalization = checkpoint_time_normalization
    source_plan.time_origin = float(checkpoint.get('time_origin', checkpoint_min))
    source_plan.time_scale = float(checkpoint.get('time_scale', checkpoint_max - checkpoint_min))
    source_plan.age_grid = np.arange(checkpoint_min, float(projection_max_age) + 1e-08, float(projection_age_step))
