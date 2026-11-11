"""scDiffEq normal-trajectory disease-age baseline for split normal/disease h5ads.

Matches disease_age_accel_mrvi_flowmap.py experimental controls:
- Normal holdout: donor manifests hold out donors; age-target CV manifests hold
  out all cells at target ages for calibration/model-selection.
- Age-grid projection selects t_hat by latent Sinkhorn distance.
- Per-donor disease delta_t from raw Sinkhorn age localization.
- Calibration pass flags using the same threshold logic as mrvi_flowmap.
"""
import argparse
import types
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.preprocessing import StandardScaler
from sccody_analyses.benchmarks.engine.optim.loss_func import SinkhornLoss
from sccody_analyses.benchmarks.engine.optim.running import CheckpointSelector, add_checkpoint_selection_args
from sccody_analyses.benchmarks.engine.projection.rollout import aggregate_pooled_source_age_projection, capped_eval_count, rollout_source_ensemble
from sccody_analyses.benchmarks.engine.projection.metrics import score_projection_age_grid, sinkhorn_distance
from sccody_analyses.benchmarks.engine.methods.dynamics import evaluate_holdout_l_gene, get_expr, rollout_source_specs, pooled_youngest_forward
from sccody_analyses.benchmarks.engine.artifacts import atomic_write_json
from sccody_analyses.benchmarks.engine.utilities.parsing import parse_float_targets
from sccody_analyses.benchmarks.engine.time_axis import age_to_time
from sccody_analyses.benchmarks.engine.dynamics.identity import BenchmarkMethodIdentity
from sccody_analyses.benchmarks.engine.dynamics.cli import build_common_argument_parser
from sccody_analyses.benchmarks.engine.dynamics.checkpoints import apply_cross_dataset_age_domain, build_checkpoint_envelope, validate_checkpoint_envelope, validate_checkpoint_time_axis
from sccody_analyses.benchmarks.engine.dynamics.runner import run_dynamics_evaluation
from sccody_analyses.benchmarks.engine.dynamics.pca_knn_localization import compute_pca_knn_cell_type_age_localization, select_target_indices_for_pca_knn
IDENTITY = BenchmarkMethodIdentity('scdiffeq', 'scDiffEq', 'scdiffeq_latent')
INCREMENTAL_PCA_GENE_THRESHOLD = 5000

def parse_args(argv=None):
    parser = build_common_argument_parser('Train scDiffEq on normal cells and roll out to disease ages.', 'scDiffEq')
    add_checkpoint_selection_args(parser, default_every=25)
    parser.add_argument('--latent-dim', type=int, default=50)
    parser.add_argument('--scdiffeq-epochs', type=int, default=500)
    parser.add_argument('--scdiffeq-drift-h', type=int, default=512)
    parser.add_argument('--scdiffeq-diff-h', type=int, default=32)
    parser.add_argument('--scdiffeq-ratio-target', type=float, default=2.5)
    parser.add_argument('--scdiffeq-lr', type=float, default=0.0001)
    return parser.parse_args(argv)

def scdiffeq_model_config(args, latent_dim):
    return {'latent_dim': int(latent_dim), 'scdiffeq_drift_h': int(args.scdiffeq_drift_h), 'scdiffeq_diff_h': int(args.scdiffeq_diff_h), 'scdiffeq_ratio_target': float(args.scdiffeq_ratio_target), 'dt': 0.01 if args.time_normalization == 'age_100' else 1.0}

def save_scdiffeq_checkpoint(path, DiffEq, pca, scaler, args, normal_train, latent_dim, sorted_ages, age_min, age_max, selection=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_checkpoint_envelope(IDENTITY, args=args, normal_train=normal_train, train_ages=sorted_ages, age_min=age_min, age_max=age_max, selection=selection)
    payload.update({'diffeq_state_dict': DiffEq.state_dict(), 'pca': pca, 'scaler': scaler, 'model_config': scdiffeq_model_config(args, latent_dim), 'latent_dim_used': int(latent_dim)})
    torch.save(payload, path)
    return path

def load_scdiffeq_checkpoint(path, n_genes, var_names, device):
    path = Path(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    validate_checkpoint_envelope(ckpt, path=path, identity=IDENTITY, n_genes=n_genes, var_names=var_names, required_keys=('diffeq_state_dict', 'pca', 'scaler', 'model_config'))
    return ckpt

def _iter_index_batches(indices, max_batch_size):
    """Yield bounded contiguous batches while keeping every batch usable by IPCA."""
    n_cells = len(indices)
    n_batches = max(1, int(np.ceil(n_cells / max_batch_size)))
    batch_size = int(np.ceil(n_cells / n_batches))
    for start in range(0, n_cells, batch_size):
        yield indices[start:start + batch_size]

def _fit_scdiffeq_pca_streaming(adata, train_idx, args, latent_dim):
    """Fit and transform PCA in bounded expression batches."""
    batch_size = min(4096, len(train_idx))
    batches = list(_iter_index_batches(train_idx, batch_size))
    scaler = StandardScaler()
    for batch_idx in batches:
        x_batch = get_expr(adata, batch_idx, args)
        scaler.partial_fit(x_batch)
        del x_batch
    pca = IncrementalPCA(n_components=latent_dim, batch_size=batch_size)
    for batch_idx in batches:
        x_batch = get_expr(adata, batch_idx, args)
        x_batch = scaler.transform(x_batch, copy=False)
        pca.partial_fit(x_batch)
        del x_batch
    z_train = np.empty((len(train_idx), latent_dim), dtype=np.float32)
    offset = 0
    for batch_idx in batches:
        x_batch = get_expr(adata, batch_idx, args)
        x_batch = scaler.transform(x_batch, copy=False)
        next_offset = offset + len(batch_idx)
        z_train[offset:next_offset] = pca.transform(x_batch).astype(np.float32)
        offset = next_offset
        del x_batch
    explained = float(np.sum(getattr(pca, 'explained_variance_ratio_', np.array([np.nan]))))
    print(f'[PCA] streaming dim={latent_dim} batch_size={batch_size} explained={explained:.3f}', flush=True)
    return (scaler, pca, z_train)

def _fit_scdiffeq_pca(adata, train_idx, args, latent_dim):
    """Fit PCA with a bounded path for full-protein-sized feature spaces."""
    n_genes = int(adata.n_vars)
    if n_genes > INCREMENTAL_PCA_GENE_THRESHOLD:
        return _fit_scdiffeq_pca_streaming(adata, train_idx, args, latent_dim)
    x_train = get_expr(adata, train_idx, args)
    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train)
    del x_train
    pca = PCA(n_components=latent_dim, random_state=args.seed)
    z_train = pca.fit_transform(x_train_sc).astype(np.float32)
    del x_train_sc
    print(f'[PCA] in-memory dim={latent_dim} features={n_genes} explained={pca.explained_variance_ratio_.sum():.3f}', flush=True)
    return (scaler, pca, z_train)

class _CachedGroupedPcaDataset(Dataset):
    """Grouped scDiffEq batches backed by one GPU-resident PCA tensor.

    torch-adata normally materializes a new CPU tensor every time it creates
    the grouped train/validation dataset.  The public scDiffEq batch contract
    is just ``[time_groups, X_by_group, W_by_group]``, so the same contract can
    be served by indexing the one cached PCA tensor directly.
    """

    def __init__(self, x_pca, group_indices, groups, weights):
        self.x_pca = x_pca
        self.group_indices = torch.as_tensor(group_indices, dtype=torch.long, device=x_pca.device)
        self.groups = torch.as_tensor(groups, dtype=torch.float32, device=x_pca.device)
        self.weights = weights

    def __len__(self):
        return int(self.group_indices.shape[1])

    def __getitem__(self, idx):
        row_idx = self.group_indices[:, int(idx)]
        return [self.groups, self.x_pca.index_select(0, row_idx), self.weights.index_select(0, row_idx)]

def _grouped_pca_indices(data_module, dataset_key):
    """Match torch-adata's grouped/padded row selection without materializing X."""
    access_key = data_module.data_keys[dataset_key]
    adata = data_module.adata
    obs = adata.obs
    selected = np.flatnonzero(obs[access_key].to_numpy(dtype=bool))
    if selected.size == 0:
        raise RuntimeError(f'scDiffEq {dataset_key} split is empty')
    groupby = data_module.hparams.get('groupby')
    if not groupby:
        raise RuntimeError('GPU-cached scDiffEq data requires a time group')
    grouped = obs.iloc[selected].groupby(groupby, sort=True)
    group_rows = [selected[np.asarray(rows, dtype=np.int64)] for rows in grouped.indices.values()]
    if not group_rows:
        raise RuntimeError(f'scDiffEq {dataset_key} split has no time groups')
    max_group_size = max((len(rows) for rows in group_rows))
    padded_rows = []
    for rows in group_rows:
        n_pad = max_group_size - len(rows)
        if n_pad:
            rows = np.concatenate([rows, np.random.choice(rows, size=n_pad, replace=True)])
        padded_rows.append(rows)
    return (np.stack(padded_rows, axis=0), np.asarray(list(grouped.groups.keys())))

def _cache_scdiffeq_pca_on_gpu(model, device):
    """Install GPU-backed scDiffEq loaders and drop the duplicate CPU PCA copy."""
    if device.type != 'cuda':
        return False
    data_module = model.LitDataModule
    adata = data_module.adata
    pca_array = adata.obsm.get('X_pca')
    if pca_array is None:
        raise RuntimeError("scDiffEq AnnData is missing obsm['X_pca']")
    x_pca = torch.as_tensor(pca_array, dtype=torch.float32, device=device)
    weights = torch.as_tensor(adata.obs[data_module._weight_key].to_numpy(), dtype=torch.float32, device=device).reshape(-1, 1)
    del adata.obsm['X_pca']
    data_module._cached_x_pca = x_pca
    data_module._cached_weights = weights

    def cached_loader(self, dataset_key):
        if dataset_key in {'train', 'val'}:
            self.configure_train_val_split()
            if self.hparams['shuffle_time_labels']:
                self.shuffle_time_labels()
        group_indices, groups = _grouped_pca_indices(self, dataset_key)
        dataset = _CachedGroupedPcaDataset(self._cached_x_pca, group_indices, groups, self._cached_weights)
        shuffle = dataset_key == 'train' and self.hparams['shuffle']
        batch_size = self.hparams['test_batch_size' if dataset_key == 'test' else 'batch_size']
        return DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=shuffle)
    data_module._return_loader = types.MethodType(cached_loader, data_module)
    print(f'[scDiffEq] cached PCA input on {device} shape={tuple(x_pca.shape)} ({x_pca.numel() * x_pca.element_size() / 2 ** 30:.2f} GiB)', flush=True)
    return True

def simulate(DiffEq, z0, t_eval, device):
    t_eval_t = torch.tensor(t_eval, dtype=torch.float32, device=device)
    chunks = []
    with torch.enable_grad():
        for start in range(0, len(z0), 256):
            z = torch.from_numpy(z0[start:start + 256]).float().to(device)
            z_hat = DiffEq.forward(z, t_eval_t)
            chunks.append(z_hat.detach().cpu().numpy().transpose(1, 0, 2))
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    return np.concatenate(chunks, axis=0)

def sweep_and_project(x_target, source_idx_global, source_age, target_age, age_grid, adata_train, pca, scaler, DiffEq, args, rng, device, target_cell_types=None):
    """Sweep age_grid, compute latent-space Sinkhorn, return best t_hat + metrics.

    x_target: (N, G) log-normalised gene matrix of target group (disease or holdout normal)
    source_idx_global: global indices into adata_train for the source pool
    Returns dict of metrics or None if insufficient cells.
    """
    n = capped_eval_count(args.max_eval_cells, len(source_idx_global), len(x_target))
    if n < 2:
        return None
    target_cell_types_array = None if target_cell_types is None else np.asarray(target_cell_types, dtype=np.int64)
    requested_cell_types = [item.strip() for item in str(getattr(args, 'pca_knn_cell_types', '')).split(',') if item.strip()] or None
    selected_target = select_target_indices_for_pca_knn(target_cell_types_array, n=n, rng=rng, categories=getattr(args, '_pca_knn_categories', []), requested_cell_types=requested_cell_types)
    tidx = selected_target if selected_target is not None else rng.choice(len(x_target), size=n, replace=False)
    n = len(tidx)
    if n < 2:
        return None
    sidx = rng.choice(len(source_idx_global), size=n, replace=len(source_idx_global) < n)
    x_src = get_expr(adata_train, source_idx_global[sidx], args)
    z_src = pca.transform(scaler.transform(x_src)).astype(np.float32)
    z_tgt = pca.transform(scaler.transform(x_target[tidx])).astype(np.float32)
    forward_grid = [float(a) for a in age_grid if float(a) > float(source_age)]
    if not forward_grid:
        return None
    t_eval = age_to_time([float(source_age)] + forward_grid, origin=args._time_origin, scale=args._time_scale)
    z_pred_all = simulate(DiffEq, z_src, t_eval, device)
    ns = min(args.max_sinkhorn_cells, len(z_tgt), len(z_src))
    target_sinkhorn = torch.from_numpy(z_tgt[:ns]).float().to(device) if ns >= 2 else None

    def sinkhorn_score(index):
        prediction = torch.from_numpy(z_pred_all[:ns, index + 1, :]).float().to(device)
        sinkhorn = SinkhornLoss([target_sinkhorn], prediction.unsqueeze(1), blur=args.projection_sinkhorn_blur, scaling=args.projection_sinkhorn_scaling, batch_size=None)
        return float(sinkhorn.detach().cpu())
    curve = score_projection_age_grid(list(range(len(forward_grid))), forward_grid, target_age, sinkhorn_score=sinkhorn_score, rng=rng)
    t_hat_sinkhorn = curve['t_hat_sinkhorn']
    chrono_idx = int(curve['chronological_index']) + 1
    latent_t_hat_idx = int(curve['best_sinkhorn_index']) + 1
    pred_chrono_expr = np.clip(scaler.inverse_transform(pca.inverse_transform(z_pred_all[:ns, chrono_idx, :])), 0, None).astype(np.float32)
    pred_latent_t_hat_expr = np.clip(scaler.inverse_transform(pca.inverse_transform(z_pred_all[:ns, latent_t_hat_idx, :])), 0, None).astype(np.float32)
    result = {'t_hat_sinkhorn': t_hat_sinkhorn, 'sinkhorn_at_t_hat': curve['sinkhorn_at_t_hat'], 'sinkhorn_at_chrono': curve['sinkhorn_at_chrono'], 'sinkhorn_improvement': curve['sinkhorn_improvement'], 'n_sinkhorn_cells': int(ns) if target_sinkhorn is not None else 0, 'sinkhorn_space': 'pca_latent', 'gene_space_sinkhorn_at_chrono': sinkhorn_distance(x_target[tidx][:ns], pred_chrono_expr, max_cells=ns, blur=args.projection_sinkhorn_blur, scaling=args.projection_sinkhorn_scaling), 'gene_space_sinkhorn_at_latent_t_hat': sinkhorn_distance(x_target[tidx][:ns], pred_latent_t_hat_expr, max_cells=ns, blur=args.projection_sinkhorn_blur, scaling=args.projection_sinkhorn_scaling), 'gene_space_sinkhorn_space': 'decoded_expression', 'gene_space_sinkhorn_n_cells': int(ns), 'n_eval_cells': int(n), 'source_age': float(source_age)}
    if args.cell_type_readout == 'pca_knn':
        if target_cell_types is None:
            raise ValueError('PCA-KNN localization requires target cell-type labels')
        predicted_expression_by_age = np.stack([np.clip(scaler.inverse_transform(pca.inverse_transform(z_pred_all[:, index + 1, :])), 0, None).astype(np.float32) for index in range(len(forward_grid))], axis=1)
        result['per_cell_type'] = compute_pca_knn_cell_type_age_localization(predicted_expression_by_age=predicted_expression_by_age, predicted_latent_by_age=z_pred_all[:, 1:, :], target_expression=x_target[tidx], target_latent=z_tgt, target_cell_types=target_cell_types_array[tidx], candidate_ages=forward_grid, target_age=target_age, classifier=args._pca_knn_classifier, cell_type_categories=args._pca_knn_categories, requested_cell_types=requested_cell_types, min_cells=args.min_calibration_cells, max_sinkhorn_cells=args.max_sinkhorn_cells, blur=args.projection_sinkhorn_blur, scaling=args.projection_sinkhorn_scaling, rng=rng)
    return result

def project_scdiffeq_source(DiffEq, pca, scaler, adata_train, args, device, source_age, indices, n_cells, target_age):
    del n_cells
    x_source = get_expr(adata_train, indices, args)
    z_source = pca.transform(scaler.transform(x_source)).astype(np.float32)
    model_times = age_to_time([source_age, float(target_age)], origin=args._time_origin, scale=args._time_scale)
    z_pred = simulate(DiffEq, z_source, model_times, device)[:, -1, :]
    x_pred = scaler.inverse_transform(pca.inverse_transform(z_pred)).astype(np.float32)
    return np.clip(x_pred, 0, None)

def run_benchmark(args, source_plan):
    args._time_origin = float(source_plan.time_origin)
    args._time_scale = float(source_plan.time_scale)
    if args.unconstrained_age_search:
        raise ValueError('scDiffEq uses an Ito SDE whose torchsde integrator requires strictly increasing evaluation times; reverse-time age search is not defined without a separately learned reverse/score model.')
    run_name, outdir, proj_result, rng = (source_plan.run_name, source_plan.outdir, source_plan.projection_path, source_plan.rng)
    normal_path, disease_path = (source_plan.normal_path, source_plan.disease_path)
    normal, disease = (source_plan.normal, source_plan.disease)
    normal_train, normal_holdout = (source_plan.normal_train, source_plan.normal_holdout)
    normal_train_ages = source_plan.normal_train_ages
    entries, donors_by_age_train = (source_plan.entries, source_plan.donors_by_age)
    sorted_ages = source_plan.sorted_ages
    age_min, age_max, age_grid = (source_plan.age_min, source_plan.age_max, source_plan.age_grid)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[scDiffEq] device={device}', flush=True)
    checkpoint = None
    checkpoint_path = Path(args.checkpoint).expanduser() if args.checkpoint else None
    if checkpoint_path:
        print(f'[scDiffEq] loading checkpoint {checkpoint_path}', flush=True)
        checkpoint = load_scdiffeq_checkpoint(checkpoint_path, n_genes=normal_train.n_vars, var_names=normal_train.var_names, device=torch.device('cpu'))
        validate_checkpoint_time_axis(checkpoint, requested_normalization=args.time_normalization)
        if args.cross_dataset_eval:
            apply_cross_dataset_age_domain(source_plan, checkpoint, age_key=args.age_key, projection_age_step=args.projection_age_step, projection_max_age=max(parse_float_targets(args.calibration_targets) or source_plan.disease.obs[args.age_key].astype(float).tolist()))
            entries = source_plan.entries
            donors_by_age_train = source_plan.donors_by_age
            sorted_ages = source_plan.sorted_ages
            age_min, age_max, age_grid = (source_plan.age_min, source_plan.age_max, source_plan.age_grid)
            args._time_origin = float(source_plan.time_origin)
            args._time_scale = float(source_plan.time_scale)
            print(f'[scDiffEq] cross-dataset evaluation: using checkpoint feature order; checkpoint age domain=[{age_min:.0f},{age_max:.0f}]', flush=True)
    train_idx = np.concatenate([entries[a] for a in sorted_ages]).astype(np.int64)
    if checkpoint is not None:
        scaler = checkpoint['scaler']
        pca = checkpoint['pca']
        latent_dim = int(checkpoint['latent_dim_used'])
        z_train = np.zeros((len(train_idx), latent_dim), dtype=np.float32)
        explained = float(np.sum(getattr(pca, 'explained_variance_ratio_', np.array([np.nan]))))
        print(f'[PCA] loaded dim={latent_dim} explained={explained:.3f}', flush=True)
    else:
        latent_dim = min(args.latent_dim, len(train_idx) - 1, normal_train.n_vars)
        if latent_dim < 2:
            raise RuntimeError('Too few cells/features for PCA')
        scaler, pca, z_train = _fit_scdiffeq_pca(normal_train, train_idx, args, latent_dim)
    import anndata as ad
    ad_sc = ad.AnnData(obs=normal_train.obs.iloc[train_idx][[]].copy())
    ad_sc.obsm['X_pca'] = z_train
    ad_sc.obs['t'] = age_to_time(normal_train_ages[train_idx], origin=args._time_origin, scale=args._time_scale).astype(np.float32)
    ad_sc.obs['train'] = True
    ad_sc.obs['test'] = False
    ad_sc.obs['W'] = 1.0
    import scdiffeq as sdq
    model_config = checkpoint.get('model_config', scdiffeq_model_config(args, latent_dim)) if checkpoint is not None else scdiffeq_model_config(args, latent_dim)
    if int(model_config.get('latent_dim', latent_dim)) != int(latent_dim):
        raise ValueError(f"Checkpoint latent_dim={model_config.get('latent_dim')} does not match PCA dim={latent_dim}")
    drift_h = int(model_config.get('scdiffeq_drift_h', args.scdiffeq_drift_h))
    diff_h = int(model_config.get('scdiffeq_diff_h', args.scdiffeq_diff_h))
    ratio_target = float(model_config.get('scdiffeq_ratio_target', args.scdiffeq_ratio_target))
    model = sdq.scDiffEq(adata=ad_sc, use_key='X_pca', time_key='t', dt=float(model_config.get('dt', 1.0)), t_min=age_to_time(age_min, origin=args._time_origin, scale=args._time_scale), t_max=age_to_time(age_max, origin=args._time_origin, scale=args._time_scale), train_key='train', test_key='test', weight_key='W', latent_dim=latent_dim, mu_hidden=[drift_h, drift_h], sigma_hidden=[diff_h, diff_h], velocity_ratio_params={'target': ratio_target, 'enforce': 100, 'method': 'square'}, name=run_name, working_dir=str(outdir))
    # scDiffEq 1.1.0 otherwise adds a CSV logger even when logger=None.
    from lightning.pytorch.loggers.logger import DummyLogger
    model._LOGGER = [DummyLogger()]
    del ad_sc, z_train
    if checkpoint is not None:
        model.DiffEq.load_state_dict(checkpoint['diffeq_state_dict'])
        print(f'[scDiffEq] loaded checkpoint drift_h={drift_h} diff_h={diff_h} ratio_target={ratio_target}', flush=True)
    else:
        _cache_scdiffeq_pca_on_gpu(model, device)
        print(f'[scDiffEq] training epochs={args.scdiffeq_epochs} drift_h={drift_h} diff_h={diff_h} ratio_target={ratio_target} lr={args.scdiffeq_lr}', flush=True)
        selector = CheckpointSelector(args.epoch_selection_metric, total_epochs=args.scdiffeq_epochs, every_epochs=args.normal_holdout_selection_every_epochs)
        selection_callback = None
        if args.epoch_selection_metric == 'holdout_L_gene':
            import lightning
            class HoldoutSelection(lightning.pytorch.callbacks.Callback):

                def on_train_epoch_end(self, trainer, pl_module, *unused):
                    epoch = int(trainer.current_epoch) + 1
                    if not selector.should_score(epoch):
                        return
                    was_training = pl_module.training
                    pl_module.eval()

                    def predict_group(target_age, target_indices, score_rng):
                        return pooled_youngest_forward(entries, target_age, len(target_indices), score_rng,
                            lambda source_age, indices, count, age: project_scdiffeq_source(pl_module, pca, scaler, normal_train, args, device, source_age, indices, count, age))

                    record = evaluate_holdout_l_gene(normal_holdout, age_key=args.age_key, donor_key=args.donor_key, target_ages=parse_float_targets(args.calibration_targets), min_cells=args.normal_holdout_selection_min_cells, max_groups=args.normal_holdout_selection_max_groups, max_cells=args.normal_holdout_selection_max_cells, seed=args.seed, device=device, get_expression=lambda adata, indices: get_expr(adata, indices, args), predict_group=predict_group, blur=args.projection_sinkhorn_blur, scaling=args.projection_sinkhorn_scaling)
                    if was_training:
                        pl_module.train()
                    selector.observe(epoch, record, pl_module.state_dict())
            selection_callback = HoldoutSelection()
        callbacks = [selection_callback] if selection_callback is not None else []
        model.fit(train_epochs=args.scdiffeq_epochs, train_lr=args.scdiffeq_lr, accelerator='cuda' if torch.cuda.is_available() else 'cpu', train_callbacks=callbacks, save_last_ckpt=False, keep_ckpts=0)
        if selection_callback is None:
            selector.observe_fixed_last(args.scdiffeq_epochs, model.DiffEq.state_dict())
        model.DiffEq.load_state_dict(selector.require_best_state())
        selection_payload = selector.payload()
        selection_path = outdir / f'{run_name}_scdiffeq_checkpoint_selection.json'
        atomic_write_json(selection_path, selection_payload)
        print(f'[selected] scDiffEq epoch={selection_payload["epoch"]}', flush=True)
    DiffEq = model.DiffEq.to(device)
    DiffEq.eval()
    if checkpoint is None and args.checkpoint_dir:
        ckpt_out = Path(args.checkpoint_dir).expanduser() / f'{run_name}_scdiffeq.pt'
        save_scdiffeq_checkpoint(ckpt_out, DiffEq=DiffEq, pca=pca, scaler=scaler, args=args, normal_train=normal_train, latent_dim=latent_dim, sorted_ages=sorted_ages, age_min=age_min, age_max=age_max, selection=selection_payload)
        print(f'[saved] {ckpt_out}', flush=True)
    if checkpoint is None and args.train_only:
        print('[scDiffEq] train-only complete; skipping report metrics', flush=True)
        return
    source_age_projection_records = []

    def project_adapter(x_target, target_age, source_entries, source_donors, projection_args, projection_rng, snapshot_records, snapshot_meta, target_cell_types=None):
        del snapshot_records
        specs = rollout_source_specs(source_entries, source_donors, target_age, projection_args, projection_rng)
        return aggregate_pooled_source_age_projection(source_entries, source_donors, specs, target_age, lambda spec, indices: sweep_and_project(x_target, indices, spec['source_age'], target_age, age_grid, normal_train, pca, scaler, DiffEq, projection_args, projection_rng, device, target_cell_types=target_cell_types), records=source_age_projection_records, record_metadata={'donor_id': snapshot_meta.get('label') if snapshot_meta else None, 'projection_kind': snapshot_meta.get('projection_kind') if snapshot_meta else None, 'projection_unit': args.projection_unit, 'n_target_cells': int(len(x_target))}, include_per_cell_type=args.cell_type_readout == 'pca_knn')

    def rollout_adapter(target_age, source_entries, source_donors, rollout_args, n_cells, rollout_rng):
        specs = rollout_source_specs(source_entries, source_donors, target_age, rollout_args, rollout_rng)
        return rollout_source_ensemble(source_entries, specs, target_age, n_cells, rollout_rng, lambda source_age, indices, count: project_scdiffeq_source(DiffEq, pca, scaler, normal_train, rollout_args, device, source_age, indices, count, target_age), source_donors_by_age=source_donors, return_source_donors=True, return_source_ages=True)
    run_dynamics_evaluation(source_plan=source_plan, args=args, identity=IDENTITY, project=project_adapter, rollout=rollout_adapter, source_age_projection_records=source_age_projection_records)
