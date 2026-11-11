"""Explicit source selection shared by localization and endpoint generation.

Target age is deliberately absent from source selection. Matched-source
manifests are accepted only when their source selection is target independent.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import numpy as np

class IncompleteSourceSupport(ValueError):
    """A required source population is unavailable; no ensemble is valid."""

def prescribed_ages(selections: Sequence[Sequence[float]], *, first_only=False):
    """Validate that a manifest uses one fixed, ordered source-age selection."""
    rows = [tuple((float(age) for age in row)) for row in selections]
    if not rows or not rows[0] or any((row != rows[0] for row in rows)):
        raise ValueError('An explicit, target-independent fixed-source manifest is required')
    ages = rows[0][:1] if first_only else rows[0]
    if len(set(ages)) != len(ages) or not np.isfinite(ages).all():
        raise ValueError('Source ages must be distinct and finite')
    return ages

def require_support(ages, entries: Mapping, *, minimum=1):
    missing = [age for age in ages if len(entries.get(float(age), ())) < minimum]
    if missing:
        raise IncompleteSourceSupport(f'Missing required source support: {missing}')

def select_source_ages(observed_ages, *, selections=(), policy='first_start', count=5, rng=None):
    """Resolve an explicit manifest or a configured target-independent policy.

    The experiment owns the source count. Automatic policies operate on the
    reference alone and never substitute ages for an incomplete manifest.
    """
    if selections:
        return prescribed_ages(selections)
    observed = sorted(set((float(age) for age in observed_ages)))
    if len(observed) < count:
        raise IncompleteSourceSupport(f'Requested {count} source ages; available {len(observed)}')
    if policy == 'first_start':
        return tuple(observed[:count])
    if policy in {'random_train_pool', 'random'}:
        return tuple(sorted(rng.choice(observed, size=count, replace=False).tolist()))
    raise ValueError(f'Source policy {policy!r} requires an explicit manifest')

def minimum_target_cells(unit: str):
    if unit not in {'donor', 'donor_cell_type'}:
        raise ValueError('Localization must produce donor or donor × cell-type outputs')
    return 100 if unit == 'donor' else 5
