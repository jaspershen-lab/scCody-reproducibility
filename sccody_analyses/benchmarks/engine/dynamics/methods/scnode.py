"""scNODE normal-trajectory disease-age baseline for split normal/disease h5ads.

Same experimental controls as disease_age_accel_scdiffeq.py:
- Normal holdout: donor manifests hold out donors; age-target CV manifests hold
  out all cells at target ages for calibration/model-selection.
- Age-grid projection selects t_hat by latent Sinkhorn distance.
- Per-donor disease delta_t from raw Sinkhorn age localization.
- Calibration pass flags using the same threshold logic as mrvi_flowmap.
"""
import argparse
import gc
from pathlib import Path
import numpy as np
import torch
from sccody_analyses.benchmarks.engine.optim.running import CheckpointSelector, add_checkpoint_selection_args, constructscNODEModel, scNODEPredict
from sccody_analyses.benchmarks.engine.optim.loss_func import MSELoss, SinkhornLoss
from sccody.gmvae import sample_gaussian
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
IDENTITY = BenchmarkMethodIdentity('scnode', 'scNODE', 'scnode_latent')

def parse_args(argv=None):
    parser = build_common_argument_parser('Train scNODE on normal cells and roll out to disease ages.', 'scNODE')
    parser.add_argument('--cache-input-on-gpu', action='store_true')
    add_checkpoint_selection_args(parser, default_every=1)
    parser.add_argument('--scnode-epochs', type=int, default=50)
    parser.add_argument('--scnode-iters', type=int, default=200)
    parser.add_argument('--scnode-pretrain-iters', type=int, default=200)
    parser.add_argument('--scnode-pretrain-batch-size', type=int, default=0)
    parser.add_argument('--scnode-latent-dim', type=int, default=50)
    parser.add_argument('--scnode-batch-size', type=int, default=200)
    parser.add_argument('--scnode-lr', type=float, default=0.001)
    parser.add_argument('--scnode-pretrain-lr', type=float, default=0.001)
    parser.add_argument('--scnode-latent-coeff', type=float, default=1.0)
    parser.add_argument('--scnode-enc-latent-list', default='128')
    parser.add_argument('--scnode-dec-latent-list', default='128')
    parser.add_argument('--scnode-drift-latent-size', default='256')
    parser.add_argument('--scnode-ode-step-size', type=float, default=1.0)
    parser.add_argument('--save-projection-snapshots', default='')
    parser.add_argument('--projection-snapshot-target-ages', default='')
    parser.add_argument('--projection-snapshot-age-tolerance', type=float, default=0.51)
    parser.add_argument('--projection-snapshot-max-rows', type=int, default=80)
    parser.add_argument('--projection-snapshot-rows-per-age', type=int, default=0)
    parser.add_argument('--projection-snapshot-max-cells-per-row', type=int, default=500)
    parser.add_argument('--projection-snapshot-only', action='store_true')
    return parser.parse_args(argv)

def parse_int_list(x):
    text = str(x or '').strip()
    if text.lower() in {'', 'none', 'null'}:
        return tuple()
    return tuple((int(v) for v in text.split(',') if v.strip()))

def scnode_model_config(args):
    step_size = float(args.scnode_ode_step_size)
    return {'latent_dim': int(args.scnode_latent_dim), 'enc_latent_list': list(parse_int_list(args.scnode_enc_latent_list)), 'dec_latent_list': list(parse_int_list(args.scnode_dec_latent_list)), 'drift_latent_size': list(parse_int_list(args.scnode_drift_latent_size)), 'ode_step_size': step_size if step_size > 0 else None}

def build_scnode_model(n_genes, config, device):
    return constructscNODEModel(n_genes, latent_dim=int(config['latent_dim']), enc_latent_list=tuple(config['enc_latent_list']), dec_latent_list=tuple(config['dec_latent_list']), drift_latent_size=tuple(config['drift_latent_size']), latent_enc_act='none', latent_dec_act='relu', drift_act='relu', ode_method='euler', ode_step_size=config.get('ode_step_size')).to(device)

def save_scnode_checkpoint(path, model, args, normal_train, sorted_ages, n_genes, age_min, age_max, selection=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_checkpoint_envelope(IDENTITY, args=args, normal_train=normal_train, train_ages=sorted_ages, age_min=age_min, age_max=age_max, selection=selection)
    payload.update({'model_state_dict': model.state_dict(), 'model_config': scnode_model_config(args), 'n_genes': int(n_genes)})
    torch.save(payload, path)
    return path

def load_scnode_checkpoint(path, n_genes, var_names, device):
    path = Path(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    validate_checkpoint_envelope(ckpt, path=path, identity=IDENTITY, n_genes=n_genes, var_names=var_names, required_keys=('model_state_dict', 'model_config'))
    model_config = ckpt['model_config']
    model = build_scnode_model(n_genes, model_config, device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return (model, ckpt)

def scnode_train_streaming(train_data, train_tps, model, latent_coeff, epochs, iters, batch_size, lr, pretrain_iters=50, pretrain_lr=0.001, rng=None, device=None, pretrain_batch_size=0, epoch_callback=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if rng is None:
        rng = np.random.default_rng(42)
    train_tps_dev = train_tps.to(device)
    pretrain_batch_size = int(pretrain_batch_size) if int(pretrain_batch_size) > 0 else int(batch_size)
    latent_encoder = model.vae.encode
    obs_decoder = model.vae.decode
    bin_weights = np.array([x.shape[0] for x in train_data], dtype=np.float64)
    bin_weights = bin_weights / bin_weights.sum()

    def _sample(x, n):
        idx = rng.choice(x.shape[0], size=n, replace=x.shape[0] < n)
        if x.device.type == 'cuda':
            idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
            return x.index_select(0, idx_t)
        return x[idx, :].to(device)
    if pretrain_iters > 0:
        params = list(model.vae.parameters())
        opt = torch.optim.Adam(params=params, lr=pretrain_lr, betas=(0.95, 0.99))
        model.vae.train()
        bin_counts = np.maximum(1, np.round(bin_weights * pretrain_batch_size)).astype(int)
        for it in range(pretrain_iters):
            xb = torch.cat([_sample(x, int(n)) for x, n in zip(train_data, bin_counts)], dim=0)
            opt.zero_grad(set_to_none=True)
            mu, logvar = latent_encoder(xb)
            std = torch.exp(0.5 * logvar) + 1e-08
            z = sample_gaussian(mu, std)
            recon = obs_decoder(z)
            loss = MSELoss(xb, recon)
            loss.backward()
            opt.step()
            if it % 25 == 0:
                print(f'  [pretrain {it:04d}/{pretrain_iters}] loss={loss.item():.4f} n={xb.shape[0]}', flush=True)
    model.train()
    opt = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.95, 0.99))
    for ep in range(epochs):
        for it in range(iters):
            source_batch = _sample(train_data[0], batch_size)
            ot_targets = [_sample(x, batch_size) for x in train_data]
            encoded_targets = [_sample(x, batch_size) for x in train_data]
            opt.zero_grad(set_to_none=True)
            recon_obs, _, _, latent_seq, _ = model([source_batch], train_tps_dev, batch_size=None)
            encoder_latent_seq = [model.vaeReconstruct([x])[0][0] for x in encoded_targets]
            ot_loss = SinkhornLoss(ot_targets, recon_obs, blur=0.05, scaling=0.5, batch_size=None)
            dyn_reg = SinkhornLoss(encoder_latent_seq, latent_seq, blur=0.05, scaling=0.5, batch_size=None)
            loss = ot_loss + latent_coeff * dyn_reg
            loss.backward()
            opt.step()
            if it % 50 == 0:
                print(f'[ep {ep + 1:02d}/{epochs} it {it:03d}/{iters}] loss={loss.item():.4f} ot={ot_loss.item():.4f} dyn={dyn_reg.item():.4f}', flush=True)
            del source_batch, ot_targets, encoded_targets, recon_obs, latent_seq, encoder_latent_seq
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if epoch_callback is not None:
            epoch_callback(ep + 1, model)
    model.eval()
    return model

def sweep_and_project_scnode(x_target, source_idx_global, source_age, target_age, age_grid, adata_train, model, args, rng, device, projection_records=None, projection_meta=None, target_cell_types=None):
    """Sweep the chronological candidate-age grid from one source population.

    Sinkhorn is scored in the learned scNODE latent space to avoid high-dimensional
    expression OT, as for the scCody and scDiffEq latent comparisons.

    x_target: (N, G) log-normalised gene matrix of target group (already normalised by caller).
    source_idx_global: global indices into adata_train for the source pool.
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
    x_tgt = x_target[tidx]
    candidate_ages = [float(age) for age in age_grid if args.unconstrained_age_search or float(age) > float(source_age)]
    if not candidate_ages:
        return None
    x_src_t = torch.from_numpy(x_src).to(device)
    with torch.no_grad():
        target_latent_mu, _ = model.vae.encode(torch.from_numpy(x_tgt).to(device))
    expression_by_age = {}
    latent_by_age = {}

    def simulate_segment(segment_ages):
        model_times = torch.tensor(age_to_time(segment_ages, origin=args._time_origin, scale=args._time_scale), dtype=torch.float32, device=device)
        with torch.no_grad():
            _, latent_segment, expression_segment, *_ = model.predict(x_src_t, model_times, n_cells=n)
        for position, age in enumerate(segment_ages[1:], start=1):
            expression_by_age[float(age)] = expression_segment[:, position, :].detach().cpu()
            latent_by_age[float(age)] = latent_segment[:, position, :].detach()
        return (expression_segment[:, 0, :].detach().cpu(), latent_segment[:, 0, :].detach())
    source_expression = source_latent = None
    lower = sorted((age for age in candidate_ages if age < float(source_age)), reverse=True)
    upper = sorted((age for age in candidate_ages if age > float(source_age)))
    if lower:
        source_expression, source_latent = simulate_segment([float(source_age), *lower])
    if upper:
        current_expression, current_latent = simulate_segment([float(source_age), *upper])
        if source_expression is None:
            source_expression, source_latent = (current_expression, current_latent)
    if any((np.isclose(age, float(source_age)) for age in candidate_ages)):
        if source_expression is None:
            source_expression, source_latent = simulate_segment([float(source_age)])
        expression_by_age[float(source_age)] = source_expression
        latent_by_age[float(source_age)] = source_latent
    all_pred = torch.stack([expression_by_age[age] for age in candidate_ages], dim=1).numpy()
    latent_pred_all = torch.stack([latent_by_age[age] for age in candidate_ages], dim=1)
    ns = min(args.max_sinkhorn_cells, len(x_tgt), len(x_src))
    target_sinkhorn = target_latent_mu[:ns] if ns >= 2 else None

    def sinkhorn_score(index):
        sinkhorn = SinkhornLoss([target_sinkhorn], latent_pred_all[:ns, index, :].unsqueeze(1), blur=args.projection_sinkhorn_blur, scaling=args.projection_sinkhorn_scaling, batch_size=None)
        return float(sinkhorn.detach().cpu())
    curve = score_projection_age_grid(list(range(len(candidate_ages))), candidate_ages, target_age, sinkhorn_score=sinkhorn_score, rng=rng)
    chrono_idx = curve['chronological_index']
    t_hat_sinkhorn = curve['t_hat_sinkhorn']
    gene_space_sinkhorn_at_chrono = sinkhorn_distance(x_tgt[:ns], all_pred[:ns, chrono_idx, :], max_cells=ns, blur=args.projection_sinkhorn_blur, scaling=args.projection_sinkhorn_scaling)
    gene_space_sinkhorn_at_latent_t_hat = sinkhorn_distance(x_tgt[:ns], all_pred[:ns, curve['best_sinkhorn_index'], :], max_cells=ns, blur=args.projection_sinkhorn_blur, scaling=args.projection_sinkhorn_scaling)
    if projection_records is not None:
        gi_sinkhorn = curve['best_sinkhorn_index']
        snap_n = int(len(x_tgt))
        max_snap = int(args.projection_snapshot_max_cells_per_row)
        if max_snap > 0 and snap_n > max_snap:
            snap_idx = rng.choice(snap_n, size=max_snap, replace=False)
        else:
            snap_idx = np.arange(snap_n, dtype=np.int64)
        meta = dict(projection_meta or {})
        projection_records.append({'target_expr': x_tgt[snap_idx].astype(np.float32, copy=True), 'pred_chrono_expr': all_pred[snap_idx, chrono_idx, :].astype(np.float32, copy=True), 'pred_sinkhorn_expr': all_pred[snap_idx, gi_sinkhorn, :].astype(np.float32, copy=True), 'target_latent': target_latent_mu.detach().cpu().numpy()[snap_idx].astype(np.float32, copy=True), 'pred_chrono_latent': latent_pred_all.detach().cpu().numpy()[snap_idx, chrono_idx, :].astype(np.float32, copy=True), 'pred_sinkhorn_latent': latent_pred_all.detach().cpu().numpy()[snap_idx, gi_sinkhorn, :].astype(np.float32, copy=True), 'target_age': float(target_age), 'source_age': float(source_age), 't_hat_sinkhorn': float(t_hat_sinkhorn), 'label': str(meta.get('label', '')), 'group': str(meta.get('group', '')), 'projection_kind': str(meta.get('projection_kind', 'normal_holdout_calibration'))})
    result = {'t_hat_sinkhorn': t_hat_sinkhorn, 'sinkhorn_at_t_hat': curve['sinkhorn_at_t_hat'], 'sinkhorn_at_chrono': curve['sinkhorn_at_chrono'], 'sinkhorn_improvement': curve['sinkhorn_improvement'], 'n_sinkhorn_cells': int(ns) if target_sinkhorn is not None else 0, 'sinkhorn_space': 'scnode_latent', 'gene_space_sinkhorn_at_chrono': gene_space_sinkhorn_at_chrono, 'gene_space_sinkhorn_at_latent_t_hat': gene_space_sinkhorn_at_latent_t_hat, 'gene_space_sinkhorn_space': 'decoded_expression', 'gene_space_sinkhorn_n_cells': int(ns), 'n_eval_cells': int(n), 'source_age': float(source_age)}
    if args.cell_type_readout == 'pca_knn':
        if target_cell_types is None:
            raise ValueError('PCA-KNN localization requires target cell-type labels')
        result['per_cell_type'] = compute_pca_knn_cell_type_age_localization(predicted_expression_by_age=all_pred, predicted_latent_by_age=latent_pred_all.detach().cpu().numpy(), target_expression=x_tgt, target_latent=target_latent_mu.detach().cpu().numpy(), target_cell_types=target_cell_types_array[tidx], candidate_ages=candidate_ages, target_age=target_age, classifier=args._pca_knn_classifier, cell_type_categories=args._pca_knn_categories, requested_cell_types=requested_cell_types, min_cells=args.min_calibration_cells, max_sinkhorn_cells=args.max_sinkhorn_cells, blur=args.projection_sinkhorn_blur, scaling=args.projection_sinkhorn_scaling, rng=rng)
    return result

def project_scnode_source(model, adata_train, args, device, source_age, indices, n_cells, target_age):
    x_source = get_expr(adata_train, indices, args)
    times = torch.tensor(age_to_time([source_age, float(target_age)], origin=args._time_origin, scale=args._time_scale), dtype=torch.float32, device=device)
    with torch.no_grad():
        prediction, _ = scNODEPredict(model, torch.from_numpy(x_source).to(device), times, n_cells=n_cells)
    return prediction[:, -1, :].astype(np.float32)

def run_benchmark(args, source_plan):
    args._time_origin = float(source_plan.time_origin)
    args._time_scale = float(source_plan.time_scale)
    run_name, outdir, proj_result, rng = (source_plan.run_name, source_plan.outdir, source_plan.projection_path, source_plan.rng)
    normal_path, disease_path = (source_plan.normal_path, source_plan.disease_path)
    normal, disease = (source_plan.normal, source_plan.disease)
    normal_train, normal_holdout = (source_plan.normal_train, source_plan.normal_holdout)
    normal_train_ages = source_plan.normal_train_ages
    entries, donors_by_age_train = (source_plan.entries, source_plan.donors_by_age)
    sorted_ages = source_plan.sorted_ages
    age_min, age_max, age_grid = (source_plan.age_min, source_plan.age_max, source_plan.age_grid)
    n_genes = normal_train.n_vars
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[scNODE] device={device}', flush=True)
    checkpoint_path = Path(args.checkpoint).expanduser() if args.checkpoint else None
    if checkpoint_path:
        print(f'[scNODE] loading checkpoint {checkpoint_path}', flush=True)
        model, ckpt = load_scnode_checkpoint(checkpoint_path, n_genes=n_genes, var_names=normal_train.var_names, device=device)
        validate_checkpoint_time_axis(ckpt, requested_normalization=args.time_normalization)
        if args.cross_dataset_eval:
            apply_cross_dataset_age_domain(source_plan, ckpt, age_key=args.age_key, projection_age_step=args.projection_age_step, projection_max_age=max(parse_float_targets(args.calibration_targets) or source_plan.disease.obs[args.age_key].astype(float).tolist()))
            entries = source_plan.entries
            donors_by_age_train = source_plan.donors_by_age
            sorted_ages = source_plan.sorted_ages
            age_min, age_max, age_grid = (source_plan.age_min, source_plan.age_max, source_plan.age_grid)
            args._time_origin = float(source_plan.time_origin)
            args._time_scale = float(source_plan.time_scale)
            print(f'[scNODE] cross-dataset evaluation: using checkpoint feature order; checkpoint age domain=[{age_min:.0f},{age_max:.0f}]', flush=True)
        ckpt_config = ckpt.get('model_config', {})
        print(f"[scNODE] loaded checkpoint latent_dim={ckpt_config.get('latent_dim')} n_bins={len(sorted_ages)} age=[{age_min:.0f},{age_max:.0f}]", flush=True)
    else:
        model_config = scnode_model_config(args)
        print(f"[scNODE] building model n_genes={n_genes} latent_dim={model_config['latent_dim']} ode_step_size={model_config['ode_step_size']} n_bins={len(sorted_ages)} age=[{age_min:.0f},{age_max:.0f}]", flush=True)
        model = build_scnode_model(n_genes, model_config, device)
        train_data = []
        if args.cache_input_on_gpu and device.type == 'cuda':
            for age in sorted_ages:
                idx = entries[age]
                x_bin = get_expr(normal_train, idx, args)
                train_data.append(torch.from_numpy(x_bin).to(device))
                del x_bin
            gib = sum((x.numel() * x.element_size() for x in train_data)) / 1024 ** 3
            print(f'[scNODE] cached training tensors on GPU: {gib:.2f} GiB', flush=True)
        else:
            for age in sorted_ages:
                idx = entries[age]
                x_bin = get_expr(normal_train, idx, args)
                train_data.append(torch.from_numpy(x_bin))
                del x_bin
        train_tps = torch.tensor(age_to_time(sorted_ages, origin=args._time_origin, scale=args._time_scale), dtype=torch.float32)
        print(f'[scNODE] training epochs={args.scnode_epochs} iters={args.scnode_iters} pretrain_iters={args.scnode_pretrain_iters} pretrain_batch_size={args.scnode_pretrain_batch_size or args.scnode_batch_size} lr={args.scnode_lr} pretrain_lr={args.scnode_pretrain_lr} latent_coeff={args.scnode_latent_coeff}', flush=True)
        selector = CheckpointSelector(args.epoch_selection_metric, total_epochs=args.scnode_epochs, every_epochs=args.normal_holdout_selection_every_epochs)
        def select_epoch(epoch, candidate_model):
            if args.epoch_selection_metric == 'fixed_last_epoch':
                selector.observe_fixed_last(epoch, candidate_model.state_dict())
                return
            if not selector.should_score(epoch):
                return
            was_training = candidate_model.training
            candidate_model.eval()

            def predict_group(target_age, target_indices, score_rng):
                return pooled_youngest_forward(entries, target_age, len(target_indices), score_rng,
                    lambda source_age, indices, count, age: project_scnode_source(candidate_model, normal_train, args, device, source_age, indices, count, age))

            record = evaluate_holdout_l_gene(normal_holdout, age_key=args.age_key, donor_key=args.donor_key, target_ages=parse_float_targets(args.calibration_targets), min_cells=args.normal_holdout_selection_min_cells, max_groups=args.normal_holdout_selection_max_groups, max_cells=args.normal_holdout_selection_max_cells, seed=args.seed, device=device, get_expression=lambda adata, indices: get_expr(adata, indices, args), predict_group=predict_group, blur=args.projection_sinkhorn_blur, scaling=args.projection_sinkhorn_scaling)
            if was_training:
                candidate_model.train()
            selector.observe(epoch, record, candidate_model.state_dict())
        model = scnode_train_streaming(train_data, train_tps, model, latent_coeff=args.scnode_latent_coeff, epochs=args.scnode_epochs, iters=args.scnode_iters, batch_size=args.scnode_batch_size, lr=args.scnode_lr, pretrain_iters=args.scnode_pretrain_iters, pretrain_lr=args.scnode_pretrain_lr, pretrain_batch_size=args.scnode_pretrain_batch_size, rng=rng, device=device, epoch_callback=select_epoch)
        model.load_state_dict(selector.require_best_state())
        model.to(device).eval()
        selection_payload = selector.payload()
        selection_path = outdir / f'{run_name}_scnode_checkpoint_selection.json'
        atomic_write_json(selection_path, selection_payload)
        print(f'[selected] scNODE epoch={selection_payload["epoch"]}', flush=True)
        print('[scNODE] training done', flush=True)
        if args.checkpoint_dir:
            ckpt_out = Path(args.checkpoint_dir).expanduser() / f'{run_name}_scnode.pt'
            save_scnode_checkpoint(ckpt_out, model=model, args=args, normal_train=normal_train, sorted_ages=sorted_ages, n_genes=n_genes, age_min=age_min, age_max=age_max, selection=selection_payload)
            print(f'[saved] {ckpt_out}', flush=True)
        if args.train_only:
            print('[scNODE] train-only complete; skipping report metrics', flush=True)
            return
    source_age_projection_records = []

    def project_adapter(x_target, target_age, source_entries, source_donors, projection_args, projection_rng, snapshot_records, snapshot_meta, target_cell_types=None):
        specs = rollout_source_specs(source_entries, source_donors, target_age, projection_args, projection_rng)
        return aggregate_pooled_source_age_projection(source_entries, source_donors, specs, target_age, lambda spec, indices: sweep_and_project_scnode(x_target, indices, spec['source_age'], target_age, age_grid, normal_train, model, projection_args, projection_rng, device, projection_records=snapshot_records, projection_meta=snapshot_meta, target_cell_types=target_cell_types), source_direction=getattr(projection_args, 'source_direction', 'earlier'), records=source_age_projection_records, record_metadata={'donor_id': snapshot_meta.get('label') if snapshot_meta else None, 'projection_kind': snapshot_meta.get('projection_kind') if snapshot_meta else None, 'projection_unit': args.projection_unit, 'n_target_cells': int(len(x_target))}, include_per_cell_type=args.cell_type_readout == 'pca_knn')

    def rollout_adapter(target_age, source_entries, source_donors, rollout_args, n_cells, rollout_rng):
        specs = rollout_source_specs(source_entries, source_donors, target_age, rollout_args, rollout_rng)
        return rollout_source_ensemble(source_entries, specs, target_age, n_cells, rollout_rng, lambda source_age, indices, count: project_scnode_source(model, normal_train, rollout_args, device, source_age, indices, count, target_age), source_direction=getattr(rollout_args, 'source_direction', 'earlier'), source_donors_by_age=source_donors, return_source_donors=True, return_source_ages=True)
    run_dynamics_evaluation(source_plan=source_plan, args=args, identity=IDENTITY, project=project_adapter, rollout=rollout_adapter, source_age_projection_records=source_age_projection_records)
