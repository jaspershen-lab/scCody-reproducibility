from __future__ import annotations
from collections.abc import Sequence
from typing import Any

def parse_float_targets(value: Any) -> list[float]:
    """Parse ordered numeric targets from CSV text or a sequence."""
    if value is None:
        return []
    if isinstance(value, str):
        items = [item.strip() for item in value.split(',') if item.strip()]
    elif isinstance(value, Sequence) and (not isinstance(value, (bytes, bytearray))):
        items = list(value)
    else:
        items = [value]
    try:
        return [float(item) for item in items]
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Float targets must be numeric: {value!r}') from exc
