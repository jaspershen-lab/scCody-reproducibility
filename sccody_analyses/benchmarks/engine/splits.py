from __future__ import annotations
from dataclasses import dataclass
import json
import os
from typing import Any
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class NormalHoldoutSplit:
    holdout_mask: np.ndarray
    train_mask: np.ndarray
    internal_validation_mask: np.ndarray
    external_report_mask: np.ndarray
    holdout_design: str | None
    holdout_ages: list[float]
    internal_validation_ages: list[float]
    external_report_ages: list[float]
    holdout_donors: list[str]
    manifest_train_donors: list[str]
    manifest_path: str | None

    @property
    def n_holdout_cells(self) -> int:
        return int(self.holdout_mask.sum())

    @property
    def n_train_cells(self) -> int:
        return int(self.train_mask.sum())

    @property
    def n_internal_validation_cells(self) -> int:
        return int(self.internal_validation_mask.sum())

    @property
    def n_external_report_cells(self) -> int:
        return int(self.external_report_mask.sum())

def canonical_holdout_design(manifest: dict[str, Any] | None) -> str:
    manifest = manifest or {}
    design = str(manifest.get('holdout_design', '') or '').strip().lower()
    if design in {'age_target', 'target_age', 'exact_age_holdout'}:
        return 'age_target_cv'
    if design:
        return design
    holdout = manifest.get('holdout', {})
    if isinstance(holdout, dict):
        scheme = str(holdout.get('scheme', '') or '').strip().lower()
        if scheme in {'exact_age_holdout', 'age_target_cv', 'age_target'}:
            return 'age_target_cv'
        if scheme:
            return 'donor_holdout'
    if manifest.get('holdout_ages'):
        return 'age_target_cv'
    return 'donor_holdout'

def _manifest_holdout_ages(manifest: dict[str, Any]) -> list[float]:
    values = manifest.get('holdout_ages', [])
    if not values and isinstance(manifest.get('holdout'), dict):
        values = manifest['holdout'].get('ages', [])
    return sorted({float(x) for x in values})

def _manifest_holdout_donors(manifest: dict[str, Any]) -> list[str]:
    values = manifest.get('holdout_donors', [])
    if not values and isinstance(manifest.get('holdout'), dict):
        values = manifest['holdout'].get('donor_ids', [])
    return sorted({str(x) for x in values if str(x)})

def _manifest_train_donors(manifest: dict[str, Any]) -> list[str]:
    return sorted({str(x) for x in manifest.get('train_donors', []) if str(x)})

def read_normal_holdout_manifest(path: str | os.PathLike[str] | None) -> dict[str, Any] | None:
    """Read and normalize a normal-reference holdout manifest.

    Supported designs:
      - donor_holdout: flat ``holdout_donors`` or nested ``holdout.donor_ids``
      - age_target_cv: flat ``holdout_ages`` or nested exact-age holdout ages

    The command-line option is ``--normal-holdout-donors-file`` even when the
    manifest describes an age-target split.
    """
    if not path:
        return None
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Normal holdout manifest not found: {path}')
    ext = os.path.splitext(path)[1].lower()
    if ext == '.json':
        with open(path) as f:
            manifest = json.load(f)
    elif ext in {'.csv', '.tsv'}:
        sep = '\t' if ext == '.tsv' else ','
        df = pd.read_csv(path, sep=sep)
        if 'split' in df.columns and 'donor_id' in df.columns:
            holdout = df.loc[df['split'].astype(str).str.lower().eq('holdout'), 'donor_id']
            train = df.loc[df['split'].astype(str).str.lower().eq('train'), 'donor_id']
        elif 'holdout_donor' in df.columns:
            holdout = df['holdout_donor']
            train = pd.Series(dtype=object)
        elif 'holdout_donors' in df.columns:
            holdout = df['holdout_donors']
            train = pd.Series(dtype=object)
        else:
            raise ValueError('CSV/TSV holdout manifest must contain either split+donor_id, holdout_donor, or holdout_donors columns')
        manifest = {'holdout_design': 'donor_holdout', 'holdout_donors': [str(x) for x in holdout.dropna().astype(str).tolist()], 'train_donors': [str(x) for x in train.dropna().astype(str).tolist()]}
    else:
        raise ValueError(f'Unsupported holdout manifest extension: {ext}')
    manifest = dict(manifest)
    design = canonical_holdout_design(manifest)
    holdout_donors = _manifest_holdout_donors(manifest)
    train_donors = _manifest_train_donors(manifest)
    holdout_ages = _manifest_holdout_ages(manifest)
    if design == 'age_target_cv':
        if not holdout_ages:
            raise ValueError(f'Age-target holdout manifest has no holdout_ages: {path}')
    elif design == 'donor_holdout':
        if not holdout_donors:
            raise ValueError(f'Donor holdout manifest has no holdout_donors: {path}')
        overlap = sorted(set(holdout_donors) & set(train_donors))
        if overlap:
            raise ValueError(f'Holdout manifest train/holdout donor overlap: {overlap[:10]}')
    else:
        raise ValueError(f'Unsupported holdout_design in manifest: {design!r}')
    manifest['holdout_design'] = design
    manifest['holdout_ages'] = holdout_ages
    manifest['holdout_donors'] = holdout_donors
    manifest['train_donors'] = train_donors
    manifest['path'] = os.path.abspath(path)
    return manifest

def age_target_mask(ages: Any, holdout_ages: list[float] | set[float]) -> np.ndarray:
    rounded_ages = np.rint(np.asarray(ages, dtype=float)).astype(int)
    rounded_holdout = sorted({int(round(float(a))) for a in holdout_ages})
    return np.isin(rounded_ages, rounded_holdout)

def normal_holdout_split_from_arrays(ages: Any, donors: Any, manifest: dict[str, Any] | None, normal_mask: Any | None=None) -> NormalHoldoutSplit:
    ages = np.asarray(ages, dtype=float)
    donors = np.asarray(donors).astype(str)
    if ages.shape[0] != donors.shape[0]:
        raise ValueError('ages and donors must have the same length')
    if normal_mask is None:
        base_mask = np.ones(ages.shape[0], dtype=bool)
    else:
        base_mask = np.asarray(normal_mask, dtype=bool)
        if base_mask.shape[0] != ages.shape[0]:
            raise ValueError('normal_mask must have the same length as ages')
    if manifest is None:
        holdout_mask = np.zeros(ages.shape[0], dtype=bool)
        train_mask = base_mask.copy()
        return NormalHoldoutSplit(holdout_mask=holdout_mask, train_mask=train_mask, internal_validation_mask=holdout_mask.copy(), external_report_mask=holdout_mask.copy(), holdout_design=None, holdout_ages=[], internal_validation_ages=[], external_report_ages=[], holdout_donors=[], manifest_train_donors=[], manifest_path=None)
    design = canonical_holdout_design(manifest)
    holdout_ages = _manifest_holdout_ages(manifest)
    external_report_ages = sorted({float(age) for age in manifest.get('external_report_ages', [])})
    internal_validation_ages = sorted({float(age) for age in manifest.get('internal_validation_ages', [])})
    if design == 'age_target_cv' and (not internal_validation_ages) and (not external_report_ages):
        internal_validation_ages = list(holdout_ages)
    manifest_train_donors = _manifest_train_donors(manifest)
    if design == 'age_target_cv':
        holdout_mask = base_mask & age_target_mask(ages, holdout_ages)
        holdout_donors = sorted(pd.unique(donors[holdout_mask]).astype(str).tolist())
    elif design == 'donor_holdout':
        requested = set(_manifest_holdout_donors(manifest))
        available = set(pd.unique(donors[base_mask]).astype(str).tolist())
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError(f'Holdout donors not found among normal-reference cells: {unknown[:10]}')
        holdout_mask = base_mask & np.isin(donors, sorted(requested))
        holdout_donors = sorted(requested)
    else:
        raise ValueError(f'Unsupported holdout_design in manifest: {design!r}')
    train_mask = base_mask & ~holdout_mask
    internal_validation_mask = base_mask & age_target_mask(ages, internal_validation_ages)
    external_report_mask = base_mask & age_target_mask(ages, external_report_ages)
    return NormalHoldoutSplit(holdout_mask=holdout_mask, train_mask=train_mask, internal_validation_mask=internal_validation_mask, external_report_mask=external_report_mask, holdout_design=design, holdout_ages=holdout_ages, internal_validation_ages=internal_validation_ages, external_report_ages=external_report_ages, holdout_donors=holdout_donors, manifest_train_donors=manifest_train_donors, manifest_path=manifest.get('path'))

def holdout_ages(manifest: dict[str, Any]) -> set[float]:
    return {float(x) for x in manifest.get('holdout', {}).get('ages', [])}
