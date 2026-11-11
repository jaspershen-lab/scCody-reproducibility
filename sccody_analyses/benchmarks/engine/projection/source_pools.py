from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping
import numpy as np

@dataclass(frozen=True)
class SourcePool:
    entries: Mapping[float, np.ndarray]
    donors_by_age: Mapping[float, Mapping[str, np.ndarray]]
    cell_type: str | None

    @property
    def n_entries(self) -> int:
        return len(self.entries)

class SourcePoolResolver:
    """Resolve and cache pooled or cell-type-specific trajectory source pools."""

    def __init__(self, *, normal_train: Any, entries: Mapping[float, np.ndarray], donors_by_age: Mapping[float, Mapping[str, np.ndarray]], projection_unit: str, cell_type_key: str, donor_key: str, min_cells: int) -> None:
        self._normal_train = normal_train
        self._entries = entries
        self._donors_by_age = donors_by_age
        self._projection_unit = str(projection_unit)
        self._cell_type_key = str(cell_type_key)
        self._donor_key = str(donor_key)
        self._min_cells = int(min_cells)
        self._cache: dict[str | None, SourcePool] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def for_cell_type(self, cell_type: str | None) -> SourcePool:
        key = str(cell_type) if self._projection_unit == 'donor_cell_type' else None
        if key not in self._cache:
            if key is None:
                entries = self._entries
                donors = self._donors_by_age
            else:
                entries = self._filter_entries(key)
                donors = self._build_donors(entries)
            self._cache[key] = SourcePool(entries=MappingProxyType(dict(entries)), donors_by_age=MappingProxyType({age: MappingProxyType(dict(by_donor)) for age, by_donor in donors.items()}), cell_type=key)
        return self._cache[key]

    def _filter_entries(self, cell_type: str) -> dict[float, np.ndarray]:
        cell_types = self._normal_train.obs[self._cell_type_key].astype(str).to_numpy()
        filtered: dict[float, np.ndarray] = {}
        for age, indices in self._entries.items():
            indices = np.asarray(indices, dtype=np.int64)
            keep = indices[cell_types[indices] == cell_type]
            if len(keep) >= self._min_cells:
                filtered[float(age)] = keep
        return filtered

    def _build_donors(self, entries: Mapping[float, np.ndarray]) -> dict[float, dict[str, np.ndarray]]:
        donor_values = self._normal_train.obs[self._donor_key].astype(str).to_numpy()
        result: dict[float, dict[str, np.ndarray]] = {}
        for age, indices in entries.items():
            by_donor: dict[str, list[int]] = {}
            for location, cell_index in enumerate(indices):
                by_donor.setdefault(donor_values[int(cell_index)], []).append(location)
            result[float(age)] = {donor: np.asarray(locations, dtype=np.int64) for donor, locations in by_donor.items()}
        return result
