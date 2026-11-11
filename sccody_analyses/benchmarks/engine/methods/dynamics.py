"""Dynamics experiment pipeline."""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sccody_analyses.benchmarks.engine.optim.loss_func import SinkhornLoss
from sccody_analyses.benchmarks.engine.projection.rollout import matched_source_indices
from sccody_analyses.benchmarks.engine.splits import normal_holdout_split_from_arrays, read_normal_holdout_manifest

from sccody.selection import evaluate_holdout_l_gene, pooled_youngest_forward


def safe_name(value):
    value = re.sub('[^A-Za-z0-9_.+-]+', '_', str(value).strip())
    return re.sub('_+', '_', value).strip('_.') or 'run'

def parse_filter(value):
    return {item.strip() for item in str(value).split(',') if item.strip()}

def target_age_allowed(age, targets):
    return not targets or any((np.isclose(float(age), float(target)) for target in targets))

def str_array(values):
    return np.asarray([str(value) for value in values], dtype='U')

def add_age(adata, args):
    if args.age_key in adata.obs:
        adata.obs[args.age_key] = adata.obs[args.age_key].astype(float)
        return adata
    if args.development_stage_key not in adata.obs:
        raise KeyError(f'Need {args.age_key!r} or {args.development_stage_key!r}')
    pattern = re.compile('(\\d+(?:\\.\\d+)?)\\s*[- ]?\\s*year\\s*[- ]?\\s*old', re.I)
    adata.obs[args.age_key] = [float(match.group(1)) if (match := pattern.search(str(value))) else np.nan for value in adata.obs[args.development_stage_key]]
    return adata

def prepare_pair(normal, disease, args, *, preserve_backed=False):
    same_input = bool(preserve_backed and normal is disease)

    def subset(adata, mask):
        mask = np.asarray(mask, dtype=bool)
        if preserve_backed and mask.all():
            return adata
        view = adata[mask]
        if preserve_backed:
            return view
        return view.to_memory() if getattr(adata, 'isbacked', False) else view.copy()
    normal = add_age(normal, args)
    if same_input:
        disease = normal
    else:
        disease = add_age(disease, args)
        if args.disease_key not in disease.obs:
            disease.obs[args.disease_key] = Path(args.disease_h5ad).stem.split('__')[-1]
    if normal.n_vars != disease.n_vars or not np.array_equal(normal.var_names, disease.var_names):
        raise ValueError('normal and disease h5ad must have identical feature space')
    for key in (args.cell_type_key, args.donor_key):
        if key not in normal.obs or key not in disease.obs:
            raise KeyError(f'Missing obs key {key!r}')
    if same_input and args.disease_key not in disease.obs:
        disease.obs[args.disease_key] = Path(args.disease_h5ad).stem.split('__')[-1]
    normal_keep = normal.obs[args.age_key].notna().to_numpy().copy()
    disease_keep = disease.obs[args.age_key].notna().to_numpy().copy()
    if args.cell_types:
        keep = parse_filter(args.cell_types)
        normal_keep &= normal.obs[args.cell_type_key].astype(str).isin(keep).to_numpy()
        disease_keep &= disease.obs[args.cell_type_key].astype(str).isin(keep).to_numpy()
    categories = list(map(str, pd.Categorical(pd.concat([normal.obs.loc[normal_keep, args.cell_type_key].astype(str), disease.obs.loc[disease_keep, args.cell_type_key].astype(str)])).categories))
    if preserve_backed:
        normal.obs['_ct_code'] = pd.Categorical(normal.obs[args.cell_type_key].astype(str), categories=categories).codes.astype(np.int32)
        normal.uns['_ct_categories'] = categories
        if not same_input:
            disease.obs['_ct_code'] = pd.Categorical(disease.obs[args.cell_type_key].astype(str), categories=categories).codes.astype(np.int32)
            disease.uns['_ct_categories'] = categories
        normal = subset(normal, normal_keep)
        disease = normal if same_input else subset(disease, disease_keep)
    else:
        normal = subset(normal, normal_keep)
        disease = normal if same_input else subset(disease, disease_keep)
        normal.obs['_ct_code'] = pd.Categorical(normal.obs[args.cell_type_key].astype(str), categories=categories).codes.astype(np.int32)
        disease.obs['_ct_code'] = pd.Categorical(disease.obs[args.cell_type_key].astype(str), categories=categories).codes.astype(np.int32)
        normal.uns['_ct_categories'] = categories
        disease.uns['_ct_categories'] = categories
    return (normal, disease)

def get_x(adata, idx, layer=''):
    values = adata.X[idx] if not layer else adata.layers[layer][idx]
    if sp.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)

def get_count_x_or_none(adata, idx, args):
    if args.de_method != 'edger_pseudobulk':
        return None
    if args.count_layer:
        if args.count_layer not in adata.layers:
            return None
        return get_x(adata, idx, args.count_layer)
    return get_x(adata, idx)

def log_normalize_dense(values, target_sum=10000.0):
    values = np.asarray(values, dtype=np.float32)
    totals = values.sum(axis=1, keepdims=True)
    return np.log1p(values * (target_sum / np.maximum(totals, 1e-08))).astype(np.float32)

def get_expr(adata, idx, args):
    dense_cache = getattr(args, '_dense_expression_cache', None)
    if dense_cache is not None and id(adata) in dense_cache:
        values = dense_cache[id(adata)][np.asarray(idx, dtype=np.int64)]
    else:
        values = get_x(adata, idx, args.layer)
    if args.layer and args.count_layer and (str(args.layer) == str(args.count_layer)):
        return log_normalize_dense(values)
    return values

def build_dense_expression_cache(adata, args, *, chunk_size=8192):
    """Densify one AnnData expression matrix once for repeated evaluation reads.

    ARTEMIS evaluation repeatedly samples the same backed train/holdout rows
    across donor, source-age, and candidate-age loops.  Keeping this cache on
    the CPU avoids a sparse slice plus ``toarray()`` allocation for every
    rollout while leaving GPU memory for the model and Sinkhorn calculations.
    """
    source = adata.X if not args.layer else adata.layers[args.layer]
    n_rows, n_cols = (int(adata.n_obs), int(adata.n_vars))
    chunk_size = max(1, int(chunk_size))
    dense = np.empty((n_rows, n_cols), dtype=np.float32)
    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        values = source[start:stop]
        if sp.issparse(values):
            values = values.toarray()
        dense[start:stop] = np.asarray(values, dtype=np.float32)
    print(f"[dense eval cache] rows={n_rows:,} genes={n_cols:,} size_gb={dense.nbytes / 1024 ** 3:.2f} layer={args.layer or 'X'}", flush=True)
    return dense

def split_normal_holdout_donors(normal, disease, args, rng, *, preserve_backed=False):
    """Construct the protocol's training and internal-validation populations."""
    del rng
    donors = normal.obs[args.donor_key].astype(str)
    unique_donors = np.array(sorted(donors.unique()))
    if args.normal_holdout_donors_file:
        manifest = read_normal_holdout_manifest(args.normal_holdout_donors_file)
        split = normal_holdout_split_from_arrays(normal.obs[args.age_key].astype(float).to_numpy(), donors.to_numpy(), manifest)
        holdout_mask = split.holdout_mask
        if not split.n_holdout_cells:
            empty = normal[:0] if preserve_backed else normal[:0].copy()
            return (normal if preserve_backed else normal.copy(), empty)
        if split.n_holdout_cells < args.min_calibration_cells:
            empty = normal[:0] if preserve_backed else normal[:0].copy()
            return (normal if preserve_backed else normal.copy(), empty)
        train = normal[~holdout_mask]
        holdout = normal[holdout_mask]
        if not preserve_backed:
            train, holdout = (train.copy(), holdout.copy())
        return (train, holdout)
    if len(unique_donors) < 3 or args.normal_holdout_fraction <= 0:
        empty = normal[:0] if preserve_backed else normal[:0].copy()
        return (normal if preserve_backed else normal.copy(), empty)
    disease_ages = disease.obs[args.age_key].astype(float).to_numpy()
    donor_age = normal.obs.assign(_d=donors).groupby('_d', observed=False)[args.age_key].median()
    scores = []
    for donor, age in donor_age.items():
        distance = float(np.min(np.abs(disease_ages - float(age)))) if len(disease_ages) else 0.0
        scores.append((distance, str(donor), float(age)))
    scores.sort(key=lambda item: (item[0], item[1]))
    requested = max(1, int(round(args.normal_holdout_fraction * len(unique_donors))))
    strict_cap = int(np.ceil(args.normal_holdout_max_donor_fraction * len(unique_donors)) - 1)
    n_holdout = min(requested, strict_cap, len(unique_donors) - 1)
    if n_holdout < 1:
        empty = normal[:0] if preserve_backed else normal[:0].copy()
        return (normal if preserve_backed else normal.copy(), empty)
    holdout_records = scores[:n_holdout]
    holdout_donors = {donor for _, donor, _ in holdout_records}
    holdout_mask = donors.isin(holdout_donors).to_numpy()
    if int(holdout_mask.sum()) < args.min_calibration_cells:
        empty = normal[:0] if preserve_backed else normal[:0].copy()
        return (normal if preserve_backed else normal.copy(), empty)
    train = normal[~holdout_mask]
    holdout = normal[holdout_mask]
    if not preserve_backed:
        train, holdout = (train.copy(), holdout.copy())
    return (train, holdout)

def build_donors_by_age(adata, entries, donor_key):
    donors = adata.obs[donor_key].astype(str).to_numpy()
    result = {}
    for age, indices in entries.items():
        by_donor = {}
        for location, cell_index in enumerate(indices):
            by_donor.setdefault(donors[cell_index], []).append(location)
        result[age] = {donor: np.asarray(locations, dtype=np.int64) for donor, locations in by_donor.items()}
    return result

def matched_entries_for_group(adata_train, group, args, *, cell_type=None):
    source_indices = matched_source_indices(adata_train.obs, np.arange(adata_train.n_obs, dtype=np.int64), age_key=args.age_key, group=group)
    if cell_type is not None:
        source_cell_types = adata_train.obs[args.cell_type_key].astype(str).to_numpy()
        source_indices = source_indices[source_cell_types[source_indices] == str(cell_type)]
    ages = adata_train.obs[args.age_key].astype(float).to_numpy()
    min_source_cells = int(getattr(args, 'min_normal_cells', 0)) if cell_type is not None else 0
    entries = {}
    for age in group.get('source_ages', []):
        age_indices = source_indices[ages[source_indices] == float(age)]
        if len(age_indices) >= max(1, min_source_cells):
            entries[float(age)] = age_indices
    donors = build_donors_by_age(adata_train, entries, args.donor_key)
    group_args = argparse.Namespace(**vars(args))
    group_args.source_direction = str(group.get('source_direction', 'earlier'))
    supported_source_ages = [float(age) for age in group.get('source_ages', []) if float(age) in entries]
    group_args.random_source_ages_by_target = {float(group['target_age']): supported_source_ages}
    return (entries, donors, group_args)

def iter_projection_targets(adata, ages, donors, args, min_cells):
    from sccody_analyses.benchmarks.engine.projection.source_selection import minimum_target_cells
    min_cells = max(min_cells, minimum_target_cells(args.projection_unit))
    if args.projection_unit == 'donor_cell_type':
        cell_types = adata.obs[args.cell_type_key].astype(str).to_numpy()
        for donor in sorted(pd.unique(donors)):
            donor_mask = donors == donor
            donor_age = float(np.median(ages[donor_mask]))
            for cell_type in sorted(pd.unique(cell_types[donor_mask])):
                mask = donor_mask & (cell_types == str(cell_type))
                indices = np.where(mask)[0]
                if len(indices) >= (5 if cell_type == 'DC' else min_cells):
                    yield (str(donor), donor_age, indices, str(cell_type))
        return
    for donor in sorted(pd.unique(donors)):
        mask = donors == donor
        indices = np.where(mask)[0]
        if len(indices) >= min_cells:
            yield (str(donor), float(np.median(ages[mask])), indices, None)

def rollout_source_specs(entries, donors_by_age, target_age, args, rng):
    """Use prescribed ages for both localization and requested endpoints."""
    from sccody_analyses.benchmarks.engine.projection.source_selection import select_source_ages, require_support
    selections = list(getattr(args, 'random_source_ages_by_target', {}).values())
    source_ages = select_source_ages(entries, selections=selections, policy=args.eval_source_policy, count=args.rollout_source_ages_per_target, rng=np.random.default_rng(getattr(args, 'seed', 42)))
    require_support(source_ages, entries)
    return [{'source_age': age, 'source_locs': np.arange(len(entries[age]), dtype=np.int64), 'source_donor': 'pooled_all_eligible_donors', 'weight': 1.0 / len(source_ages), 'replicate': 0} for age in source_ages]
