from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
import yaml
DATASET_ALIASES = {'aida': 'normal_pbmc', 'pbmc': 'normal_pbmc', 'normal_pbmc': 'normal_pbmc'}

def canonical_dataset_id(dataset_id: str) -> str:
    """Return the canonical dataset id, preserving unknown ids."""
    key = str(dataset_id).strip().lower()
    return DATASET_ALIASES.get(key, key)

def load_mapping(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() == '.json':
        loaded = json.loads(text)
    else:
        loaded = yaml.safe_load(text)
        if loaded is None:
            loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError(f'Expected a mapping at the top level of {path}')
    return loaded

@dataclass(frozen=True)
class DiseaseSpec:
    id: str
    h5ad: str
    label: str | None = None
    disease_value: str | None = None
    description: str | None = None

    @classmethod
    def from_mapping(cls, disease_id: str, data: dict[str, Any]) -> 'DiseaseSpec':
        return cls(id=str(data.get('id', disease_id)), h5ad=str(data['h5ad']), label=data.get('label') or data.get('cohort_label') or disease_id, disease_value=data.get('disease_value') or data.get('value') or disease_id, description=data.get('description'))

@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    h5ad: str
    age_key: str = 'age'
    donor_key: str = 'donor_id'
    cell_type_key: str = 'cell_type'
    cohort_key: str | None = None
    normal_label: str = 'normal'
    feature_key: str | None = None
    split: dict[str, Any] = field(default_factory=dict)
    projection: dict[str, Any] = field(default_factory=dict)
    de_reference: dict[str, Any] = field(default_factory=dict)
    methods: dict[str, Any] = field(default_factory=dict)
    results_dir: str = 'results'
    checkpoint_dir: str = 'checkpoints'
    aliases: tuple[str, ...] = field(default_factory=tuple)
    diseases: dict[str, DiseaseSpec] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> 'DatasetSpec':
        raw_dataset_id = str(data.get('dataset_id', data.get('id', 'normal_pbmc')))
        dataset_id = canonical_dataset_id(raw_dataset_id)
        raw_diseases = data.get('diseases', {})
        diseases: dict[str, DiseaseSpec] = {}
        if isinstance(raw_diseases, list):
            for item in raw_diseases:
                disease_id = str(item['id'])
                diseases[disease_id] = DiseaseSpec.from_mapping(disease_id, item)
        elif isinstance(raw_diseases, dict):
            for disease_id, item in raw_diseases.items():
                diseases[str(disease_id)] = DiseaseSpec.from_mapping(str(disease_id), item)
        else:
            raise TypeError('diseases must be a mapping or list')
        aliases = tuple((str(x) for x in data.get('aliases', ())))
        if raw_dataset_id != dataset_id and raw_dataset_id not in aliases:
            aliases = aliases + (raw_dataset_id,)
        return cls(dataset_id=dataset_id, h5ad=str(data['h5ad']), age_key=str(data.get('age_key', 'age')), donor_key=str(data.get('donor_key', 'donor_id')), cell_type_key=str(data.get('cell_type_key', 'cell_type')), cohort_key=data.get('cohort_key'), normal_label=str(data.get('normal_label', 'normal')).lower(), feature_key=data.get('feature_key'), split=dict(data.get('split', {}) or {}), projection=dict(data.get('projection', {}) or {}), de_reference=dict(data.get('de_reference', {}) or {}), methods=dict(data.get('methods', {}) or {}), results_dir=str(data.get('results_dir', 'results')), checkpoint_dir=str(data.get('checkpoint_dir', 'checkpoints')), aliases=aliases, diseases=diseases)

    def disease(self, disease_id: str) -> DiseaseSpec:
        if disease_id not in self.diseases:
            known = ', '.join(sorted(self.diseases)) or '<none>'
            raise KeyError(f'Unknown disease id {disease_id!r}; known: {known}')
        return self.diseases[disease_id]

def load_dataset_spec(path: str | Path) -> DatasetSpec:
    return DatasetSpec.from_mapping(load_mapping(path))
