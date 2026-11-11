"""The one dataset-configured DE reference shared by every benchmark."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from sccody_analyses.benchmarks.engine.config import DatasetSpec
def stratified_reference_sample(indices, ages, cell_types, max_cells, rng, *, extra_strata=None):
    """Sample one deterministic stratified pool, retaining global row indices."""
    indices = np.asarray(indices, dtype=np.int64)
    if max_cells is None or len(indices) <= int(max_cells):
        return indices
    arrays = [np.asarray(ages), np.asarray(cell_types)] + [np.asarray(x) for x in extra_strata or []]
    if any((len(x) != len(indices) for x in arrays)):
        raise ValueError('DE-reference strata are not row-aligned')
    groups: dict[tuple[str, ...], list[int]] = {}
    for row in range(len(indices)):
        key = tuple((str(x[row]) for x in arrays))
        groups.setdefault(key, []).append(row)
    per_group = max(1, int(max_cells) // len(groups))
    chosen = np.concatenate([rng.choice(rows, size=min(len(rows), per_group), replace=False) for rows in groups.values()])
    if len(chosen) > int(max_cells):
        chosen = rng.choice(chosen, size=int(max_cells), replace=False)
    elif len(chosen) < int(max_cells):
        remaining = np.setdiff1d(np.arange(len(indices)), chosen, assume_unique=False)
        if len(remaining):
            chosen = np.concatenate([chosen, rng.choice(remaining, size=min(int(max_cells) - len(chosen), len(remaining)), replace=False)])
    return indices[chosen]

def _column(obs: Any, name: str, n_rows: int) -> np.ndarray:
    if name not in obs:
        raise KeyError(f'DE-reference stratum {name!r} is absent from AnnData.obs')
    values = np.asarray(obs[name].astype(str).to_numpy(), dtype=str)
    if len(values) != n_rows:
        raise ValueError(f'DE-reference stratum {name!r} is not row-aligned')
    return values

def build_dataset_de_reference(spec: DatasetSpec, *, ages: np.ndarray, obs: Any) -> np.ndarray:
    """Return the dataset-level canonical single-pool DE reference.

    The config owns exclusions, age domain, strata, cap, and seed; no fold or
    model-specific input can alter this index set.
    """
    config = dict(spec.de_reference)
    variant = str(config.get('variant', 'sc_ref'))
    if variant in {'sc_ref_2', 'sc_ref_3'}:
        raise ValueError(f'{variant} is side-specific; use build_dataset_de_reference_by_side with the target ages instead of silently building one shared pool')
    ages = np.asarray(ages, dtype=float)
    candidates = np.arange(len(ages), dtype=np.int64)
    domain = list(config.get('age_domain', [None, None]))
    lo, hi = (domain + [None, None])[:2]
    mask = np.isfinite(ages[candidates])
    if lo is not None:
        mask &= ages[candidates] >= float(lo)
    if hi is not None:
        mask &= ages[candidates] <= float(hi)
    excluded = np.asarray(config.get('exclude_ages', []), dtype=float)
    if len(excluded):
        mask &= ~np.isin(ages[candidates], excluded)
    candidates = candidates[mask]
    cell_type_key = str(config.get('cell_type_key', spec.cell_type_key))
    cell_types = _column(obs, cell_type_key, len(ages))
    strata = [str(x) for x in config.get('strata', ['age', cell_type_key])]
    extras = [_column(obs, key, len(ages))[candidates] for key in strata if key != 'age']
    limit = int(config.get('max_cells', 50000))
    rng = np.random.default_rng(int(config.get('seed', 42)))
    indices = np.asarray(stratified_reference_sample(candidates, ages[candidates], cell_types[candidates], limit if limit > 0 else None, rng, extra_strata=extras), dtype=np.int64)
    return indices

def load_dataset_config(path: str | Path) -> DatasetSpec:
    from sccody_analyses.benchmarks.engine.config import load_dataset_spec
    return load_dataset_spec(path)
