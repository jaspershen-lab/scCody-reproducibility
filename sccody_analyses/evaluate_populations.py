"""Held-out population reconstruction and DE from native models or baseline arrays."""
import argparse
import json
import logging
import os
from pathlib import Path
import time
import anndata as ad
import h5py
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from anndata.io import read_elem, sparse_dataset
import torch
from sccody import ScCody
from sccody.training import flow, decode
from sccody_analyses.metrics.differential_expression import de_logfc_diagnostics
from sccody_analyses.metrics.edger import EdgeRContrast, run_edger_contrast, _compare_vectors
from sccody_analyses.metrics.population import distribution_metrics, shared_reference_metrics, direction_agreement

def metadata(manifest):
    with h5py.File(manifest['data'], 'r') as f:
        obs = read_elem(f['obs'])
        genes = np.asarray(read_elem(f['var']).index, dtype=str)
    d = manifest['columns']
    return (obs[[d['sample'], d['time'], d['type'], *d['strata']]].copy(), genes)

def rows(handle, indices, layer='X'):
    matrix = sparse_dataset(handle['X'] if layer == 'X' else handle['layers'][layer])
    values = matrix[np.asarray(indices, dtype=np.int64)]
    return values.toarray().astype(np.float32)

def predict_types(expression, classifier, categories):
    """Majority vote over the k nearest reference cells; ties go to the smaller summed distance."""
    scaler, pca, neighbors = classifier
    embedding = pca.transform(scaler.transform(expression))
    distances, indices = neighbors.kneighbors(embedding)
    labels = neighbors.classes_[neighbors._y[indices]]
    predicted = np.empty(len(embedding), dtype=int)
    for row, (distance, votes) in enumerate(zip(distances, labels)):
        values, counts = np.unique(votes, return_counts=True)
        tied = values[counts == counts.max()]
        selected = tied[0] if len(tied) == 1 else tied[np.argmin([distance[votes == code].sum() for code in tied])]
        predicted[row] = selected
    return np.asarray(categories)[predicted]

def prepare_inputs(manifest):
    """Build the development reference and per-fit external target templates."""
    from sccody import ScCodyConfig
    from .population_inputs import target_rows
    obs, _ = metadata(manifest)
    d = manifest['columns']
    external = obs[d['sample']].astype(str).isin(manifest['external_samples']).to_numpy()
    path = Path(manifest['reference_indices'])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, indices=np.flatnonzero(~external))
    study = json.loads(Path(manifest['study_config']).read_text())
    for task in manifest['tasks']:
        run = next(t for t in study['tasks'] if t['fit'] == task['fit'] and
                   ScCodyConfig.load(t['config']).model.scope == task['scope'])
        config = ScCodyConfig.load(run['config'])
        training = ~external & ~obs[d['sample']].astype(str).isin(config.split.validation_samples).to_numpy()
        eligible = np.flatnonzero(external)
        chosen = eligible[target_rows(obs.iloc[eligible], d['time'], manifest['target_times'], 2000,
                                      config.training.seed, d['strata'])]
        path = Path(task['target_template'])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, holdout_eval_indices=chosen,
                            actual_age=obs.iloc[chosen][d['time']].to_numpy(dtype=float),
                            train_indices=np.flatnonzero(training))


def build_reference(manifest, root):
    out = root / 'reference'
    out.mkdir(parents=True, exist_ok=True)
    obs, genes = metadata(manifest)
    d = manifest['columns']
    reference = np.load(manifest['reference_indices'], allow_pickle=False)['indices']
    age = obs[d['time']].astype(float).to_numpy()
    ids = obs[d['sample']].astype(str).to_numpy(dtype=str)
    types = obs[d['type']].astype(str).to_numpy(dtype=str)
    assert not set(ids[reference]) & set(manifest['external_samples'])
    pairs = pd.MultiIndex.from_arrays([ids, types])
    codes, groups = pd.factorize(pairs, sort=True)
    first = np.full(len(groups), len(obs), dtype=int)
    np.minimum.at(first, codes, np.arange(len(obs)))
    n_cells = np.bincount(codes, minlength=len(groups))
    expression = np.zeros((len(groups), len(genes)), dtype=np.float64)
    counts = np.zeros_like(expression)
    with h5py.File(manifest['data'], 'r') as f:
        x = sparse_dataset(f['X'])
        c = sparse_dataset(f['layers'][manifest['count_layer']])
        for start in range(0, len(obs), 8192):
            stop = min(start + 8192, len(obs))
            present, local = np.unique(codes[start:stop], return_inverse=True)
            membership = sparse.csr_matrix((np.ones(stop - start), (local, np.arange(stop - start))), shape=(len(present), stop - start))
            expression[present] += (membership @ x[start:stop]).toarray()
            counts[present] += (membership @ c[start:stop]).toarray()
        rng = np.random.default_rng(42)
        selected = np.sort(rng.choice(reference, min(50000, len(reference)), replace=False))
        training = rows(f, selected)
    np.savez_compressed(out / 'pseudobulks.npz', expression_sum=expression, count_sum=counts, n_cells=n_cells, sample=ids[first], time=age[first], cell_type=types[first], genes=genes, is_reference=np.isin(ids[first], ids[reference]))
    scaler = StandardScaler()
    pca = PCA(n_components=min(50, len(genes), len(selected) - 1), random_state=0)
    embedded = pca.fit_transform(scaler.fit_transform(training))
    codes = np.asarray([manifest['cell_types'].index(t) for t in types[selected]])
    classifier = KNeighborsClassifier(n_neighbors=15, weights='uniform', n_jobs=4).fit(embedded, codes)
    joblib.dump((scaler, pca, classifier), out / 'classifier.joblib')
    logging.info('Reference and classifier complete')

def sample_sources(obs, training_indices, target_indices, source_times, columns, eligible_types, seed):
    """Equal source-time quotas within each target-matched stratum."""
    rng = np.random.default_rng(seed)
    source_rows = []
    grouping = obs.iloc[target_indices].copy()
    keys = columns['strata']
    groups = grouping.groupby(keys, observed=True, sort=True) if keys else [(None, grouping)]
    for _, target in groups:
        pool = obs.iloc[training_indices]
        mask = np.ones(len(pool), dtype=bool)
        for key in keys:
            mask &= pool[key].astype(str).to_numpy() == str(target[key].iloc[0])
        if eligible_types:
            mask &= pool[columns['type']].astype(str).isin(eligible_types).to_numpy()
        pool_indices = training_indices[mask]
        times = obs.iloc[pool_indices][columns['time']].astype(float).to_numpy()
        quotient, remainder = divmod(len(target), len(source_times))
        for rank, source_time in enumerate(source_times):
            candidates = pool_indices[times == source_time]
            if not len(candidates):
                raise ValueError(f'Source time {source_time} lacks matched support')
            size = quotient + (rank < remainder)
            source_rows.extend(rng.choice(candidates, size, replace=len(candidates) < size).tolist())
    return np.asarray(source_rows, dtype=np.int64)

def generate(manifest, root, task_id):
    task = manifest['tasks'][task_id]
    out = root / 'generation' / task['name']
    out.mkdir(parents=True, exist_ok=True)
    obs, genes = metadata(manifest)
    columns = manifest['columns']
    pipeline = ScCody.load(task['checkpoint'], device=manifest.get('device', 'cuda'))
    with np.load(task['target_template'], allow_pickle=True) as template:
        target_indices = template['holdout_eval_indices'].astype(int)
        ages = template['actual_age'].astype(float)
        training_indices = template['train_indices'].astype(int)
    assert np.array_equal(genes, np.asarray(pipeline.genes))
    assert set(obs.iloc[target_indices][columns['sample']].astype(str)) <= set(manifest['external_samples'])
    assert not set(obs.iloc[training_indices][columns['sample']].astype(str)) & set(manifest['external_samples'])
    all_pred, all_target, all_source, all_time, all_types = ([], [], [], [], [])
    times = sorted(np.unique(ages))
    with h5py.File(manifest['data'], 'r') as f:
        for time_value in times:
            target = target_indices[ages == time_value]
            source = sample_sources(obs, training_indices, target, task['source_times'], columns, task.get('eligible_types'), 42 + task['fit'] + int(time_value) * 100)
            source_data = ad.AnnData(rows(f, source), obs=obs.iloc[source].copy(), var=pd.DataFrame(index=genes))
            original_ids = source_data.obs_names.astype(str).to_numpy()
            source_data.obs_names = pd.Index([f'source_{i}' for i in range(len(source))])
            predicted = pipeline.generate(source_data, time_value).X
            assert np.isfinite(predicted).all()
            if pipeline.config.model.scope == 'cell_type':
                pred_types = source_data.obs[columns['type']].astype(str).to_numpy()
            else:
                pred_types = predict_types(predicted, joblib.load(root / 'reference/classifier.joblib'), manifest['cell_types'])
            all_pred.append(predicted)
            all_target.append(target)
            all_source.append(source)
            all_types.append(pred_types)
            all_time.extend([time_value] * len(predicted))
            logging.info('Generated %s time=%s cells=%s', task['name'], time_value, len(predicted))
    np.savez_compressed(out / 'predictions.npz', x_pred=np.concatenate(all_pred), actual_age=np.asarray(all_time), holdout_eval_indices=np.concatenate(all_target), source_indices=np.concatenate(all_source), pred_cell_type=np.concatenate(all_types), gene_names=genes)

def scope_bulk(cache, scope, side=None, time_value=None, reference=True):
    mask = cache['is_reference'] if reference else ~cache['is_reference']
    if scope != 'all':
        mask = mask & (cache['cell_type'] == scope)
    if side:
        mask = mask & (cache['time'] < time_value if side == 'young' else cache['time'] > time_value)
    elif time_value is not None:
        mask = mask & (cache['time'] == time_value)
    ids = cache['sample'][mask]
    donors, code = np.unique(ids, return_inverse=True)
    expr = np.zeros((len(donors), len(cache['genes'])), dtype=float)
    counts = np.zeros_like(expr)
    size = np.zeros(len(donors), dtype=int)
    np.add.at(expr, code, cache['expression_sum'][mask])
    np.add.at(counts, code, cache['count_sum'][mask])
    np.add.at(size, code, cache['n_cells'][mask])
    return (expr, counts, size, donors)

def evaluate(manifest, root, task_id):
    task = manifest['evaluation_tasks'][task_id]
    output = root / 'metrics' / task['name']
    output.mkdir(parents=True, exist_ok=True)
    with np.load(root / 'reference/pseudobulks.npz', allow_pickle=False) as values:
        cache = {k: values[k] for k in values.files}
    obs, genes = metadata(manifest)
    d = manifest['columns']
    artifact = task['artifact']
    with np.load(artifact, allow_pickle=True) as a:
        prediction = a['x_pred']
        ages = a['actual_age']
        if 'holdout_eval_indices' in a.files:
            target_rows = a['holdout_eval_indices'].astype(int)
        elif 'holdout_obs_names' in a.files:
            target_rows = obs.index.astype(str).get_indexer(a['holdout_obs_names'].astype(str))
        else:
            raise ValueError('Prediction artifact requires explicit target row indices or cell IDs')
        predicted_types = a['pred_cell_type'].astype(str) if 'pred_cell_type' in a.files else np.full(len(prediction), 'unassigned')
        assert np.array_equal(a['gene_names'].astype(str), genes)
    assert len(prediction) == len(ages) == len(target_rows) and (target_rows >= 0).all() and (target_rows < len(obs)).all()
    assert np.allclose(obs.iloc[target_rows][d['time']].astype(float).to_numpy(), ages)
    assert set(obs.iloc[target_rows][d['sample']].astype(str)) <= set(manifest['external_samples'])
    if not np.isin(predicted_types, obs[d['type']].astype(str).unique()).all():
        predicted_types = predict_types(prediction, joblib.load(root / 'reference/classifier.joblib'), manifest['cell_types'])
    records, de_records = ([], [])
    with h5py.File(manifest['data'], 'r') as f:
        for time_value in sorted(np.unique(ages)):
            target = target_rows[ages == time_value]
            predicted = prediction[ages == time_value]
            pred_types = predicted_types[ages == time_value]
            real = rows(f, target)
            counts = rows(f, target, manifest['count_layer'])
            true_types = obs.iloc[target][d['type']].astype(str).to_numpy()
            donors = obs.iloc[target][d['sample']].astype(str).to_numpy()
            for scope in ['all', *manifest['cell_types']]:
                p_mask = pred_types == scope if scope != 'all' else np.ones(len(predicted), dtype=bool)
                t_mask = true_types == scope if scope != 'all' else np.ones(len(real), dtype=bool)
                eligible = min(p_mask.sum(), t_mask.sum()) >= manifest['min_cells']
                base = {'model': task['model'], 'fit': task['fit'], 'task': task['name'], 'scope': scope, 'target_age': float(time_value), 'n_pred_cells': int(p_mask.sum()), 'n_true_cells': int(t_mask.sum())}
                if not eligible:
                    records.append({**base, 'status': 'ineligible_cells'})
                    continue
                metrics, distances = distribution_metrics(real[t_mask], predicted[p_mask], device=manifest.get('device', 'cuda'), max_sinkhorn_cells=2000)
                records.append({**base, 'status': 'ok', **metrics})
                for side in ['young', 'old']:
                    expr, reference_counts, reference_size, reference_donors = scope_bulk(cache, scope, side, time_value)
                    edge_directory = output / 'edgeR'
                    name = f'{scope}_{side}'
                    edge_path = edge_directory / f'{name}_ho{int(time_value)}_{side}_real_edgeR.tsv'
                    if len(reference_donors) < 2 or len(np.unique(donors[t_mask])) < 2:
                        edge = None
                        edge_status = 'ineligible_donor_replicates'
                    else:
                        if not edge_path.exists():
                            result = run_edger_contrast(EdgeRContrast(counts[t_mask], donors[t_mask], reference_counts, reference_donors, genes, edge_directory, name, time_value, side, strict=True))
                            edge_path = Path(result['edgeR_real_path'])
                        edge = pd.read_csv(edge_path, sep='\t')
                        edge_status = 'ok'
                    reference_mean = expr.sum(0) / reference_size.sum()
                    de, diag = shared_reference_metrics(predicted[p_mask], real[t_mask], reference_mean, edge_table=edge, genes=genes)
                    de_records.append({**base, 'side': side, 'status': 'ok', 'edgeR_status': edge_status, 'n_reference_cells': int(reference_size.sum()), 'n_reference_donors': len(reference_donors), 'n_true_donors': len(np.unique(donors[t_mask])), **de})
                    gene_table = pd.DataFrame({'gene': genes, 'observed_expression_log2fc': diag['logfc_real'], 'generated_expression_log2fc': diag['logfc_pred'], 'generated_vs_observed_log2fc': diag['logfc_pred'] - diag['logfc_real']})
                    if edge is not None:
                        gene_table = gene_table.merge(edge, on='gene', how='left', validate='one_to_one')
                    gene_table.to_csv(output / f'de_genes_{scope}_{int(time_value)}_{side}.csv.gz', index=False)
                logging.info('Evaluated %s time=%s scope=%s', task['name'], time_value, scope)
                pd.DataFrame(records).to_csv(output / 'distribution.csv', index=False)
                pd.DataFrame(de_records).to_csv(output / 'differential_expression.csv', index=False)
    pd.DataFrame(records).to_csv(output / 'distribution.csv', index=False)
    pd.DataFrame(de_records).to_csv(output / 'differential_expression.csv', index=False)

def half_split_controls(manifest, root):
    with np.load(root / 'reference/pseudobulks.npz', allow_pickle=False) as values:
        cache = {k: values[k] for k in values.files}
    out = root / 'controls'
    out.mkdir(exist_ok=True)
    from itertools import combinations
    from math import comb
    from tempfile import TemporaryDirectory
    from .metrics.population import edge_reference_metrics
    records = []
    rng = np.random.default_rng(42)
    for scope in ['all', *manifest['macro_cell_types']]:
        for target_time in [age for age in manifest['target_times'] if age != 68]:
            expression, counts, sizes, donors = scope_bulk(cache, scope, time_value=target_time, reference=False)
            n, half = len(donors), len(donors) // 2
            if half < 2 or sizes.sum() < manifest['min_cells']:
                continue
            total = comb(n, half) // (2 if n % 2 == 0 else 1)
            if total <= 500:
                partitions = [tuple(a) for a in combinations(range(n), half) if n % 2 or 0 in a]
            else:
                partitions = set()
                while len(partitions) < 500:
                    a = np.sort(rng.choice(n, half, replace=False))
                    if n % 2 == 0 and 0 not in a:
                        a = np.setdiff1d(np.arange(n), a)
                    partitions.add(tuple(a))
                partitions = sorted(partitions)
            for side in ['young', 'old']:
                ref_expr, ref_count, ref_size, ref_donors = scope_bulk(cache, scope, side, target_time)
                ref_mean = ref_expr.sum(0) / ref_size.sum()
                for repeat, selected in enumerate(partitions):
                    a = np.asarray(selected)
                    b = np.setdiff1d(np.arange(n), a)
                    first = expression[a].sum(0) / sizes[a].sum()
                    logfc = np.log2(np.maximum(first, 0) + 1e-4) - np.log2(np.maximum(ref_mean, 0) + 1e-4)
                    with TemporaryDirectory(prefix='sccody-edger-') as temporary:
                        result = run_edger_contrast(EdgeRContrast(counts[b], donors[b], ref_count, ref_donors,
                            cache['genes'], temporary, scope, target_time, side, strict=True))
                        edge = pd.read_csv(result['edgeR_real_path'], sep='\t').set_index('gene').reindex(cache['genes'])
                    metrics = edge_reference_metrics(logfc, edge.edgeR_logFC.to_numpy())
                    records.append(dict(scope=scope, target_age=target_time, side=side, repeat=repeat, **metrics))
            logging.info('Half-split controls %s time=%s', scope, target_time)
    pd.DataFrame(records).to_csv(out / 'half_split_repeats.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--phase", required=True, choices=["prepare", "reference", "generate", "metrics", "controls"])
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = Path(manifest["output"])
    root.mkdir(parents=True, exist_ok=True)
    if args.phase == "prepare": prepare_inputs(manifest)
    elif args.phase == "reference": build_reference(manifest, root)
    elif args.phase == "generate": generate(manifest, root, args.task)
    elif args.phase == "metrics": evaluate(manifest, root, args.task)
    else: half_split_controls(manifest, root)

if __name__ == "__main__":
    main()
