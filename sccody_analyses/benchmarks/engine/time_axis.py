"""Shared chronological-age to model-time transforms."""
from __future__ import annotations
from typing import Any
import numpy as np
TIME_NORMALIZATIONS = ('fit_minmax', 'age_100')

def resolve_time_axis(normalization: str, *, age_min: float, age_max: float) -> tuple[float, float]:
    """Return the origin and scale used to map chronological age to model time."""
    if normalization == 'age_100':
        return (0.0, 100.0)
    if normalization == 'fit_minmax':
        scale = float(age_max) - float(age_min)
        if scale <= 0:
            raise ValueError('Cannot min-max normalize a collapsed fitted age domain')
        return (float(age_min), scale)
    raise ValueError(f'Unknown time normalization {normalization!r}; expected one of {TIME_NORMALIZATIONS}')

def age_to_time(age: Any, *, origin: float, scale: float) -> Any:
    """Map chronological years to model time without clipping extrapolation."""
    if float(scale) <= 0:
        raise ValueError('Model-time scale must be positive')
    values = (np.asarray(age, dtype=np.float64) - float(origin)) / float(scale)
    return float(values) if values.ndim == 0 else values
