"""ARTEMIS forward/backward SDE dynamics benchmark integration."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import numpy as np
import torch
from sccody_analyses.benchmarks.engine.optim.loss_func import SinkhornLoss
from sccody_analyses.benchmarks.engine.artifacts import atomic_write_json
from sccody_analyses.benchmarks.engine.dynamics.checkpoints import apply_cross_dataset_age_domain, build_checkpoint_envelope, validate_checkpoint_envelope, validate_checkpoint_time_axis
from sccody_analyses.benchmarks.engine.dynamics.cli import build_common_argument_parser
from sccody_analyses.benchmarks.engine.dynamics.identity import BenchmarkMethodIdentity
from sccody_analyses.benchmarks.engine.dynamics.runner import run_dynamics_evaluation
from sccody_analyses.benchmarks.engine.dynamics.pca_knn_localization import compute_pca_knn_cell_type_age_localization, select_target_indices_for_pca_knn
from sccody_analyses.benchmarks.engine.methods.dynamics import build_dense_expression_cache, evaluate_holdout_l_gene, get_expr, matched_entries_for_group, rollout_source_specs, pooled_youngest_forward
from sccody_analyses.benchmarks.engine.projection.metrics import score_projection_age_grid, sinkhorn_distance
from sccody_analyses.benchmarks.engine.projection.rollout import aggregate_pooled_source_age_projection, capped_eval_count, rollout_source_ensemble
from sccody_analyses.benchmarks.engine.utilities.parsing import parse_float_targets
from sccody_analyses.benchmarks.engine.optim.running import add_checkpoint_selection_args
from sccody_analyses.benchmarks.engine.adapters.artemis import ArtemisBridge, ArtemisVAE, checkpoint_payload, dataset_stats, decode_streaming, fit_streaming_vae, load_artemis_modules, make_artemis_dataset, make_artemis_setup, make_latent_frame, train_bridge_in_phases, _to_jax_tree
IDENTITY = BenchmarkMethodIdentity('artemis', 'ARTEMIS', 'artemis_latent')

def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    add_checkpoint_selection_args(parser, default_every=1)

def parse_int_list(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(int(item) for item in value)
    text = str(value or '').strip()
    if text.lower() in {'', 'none', 'null'}:
        return tuple()
    return tuple(int(item.strip()) for item in text.split(',') if item.strip())

def parse_args(argv=None):
    parser = build_common_argument_parser('Train ARTEMIS forward/backward SDEs on normal cells and roll out to target ages.', 'ARTEMIS')
    _add_selection_args(parser)
    parser.add_argument('--artemis-src', default='', help='ARTEMIS checkout or its src directory.')
    parser.add_argument('--artemis-latent-dim', type=int, default=64)
    parser.add_argument('--artemis-vae-hidden-dim', type=int, default=512)
    parser.add_argument('--artemis-vae-epochs', type=int, default=100)
    parser.add_argument('--artemis-vae-batch-size', type=int, default=512)
    parser.add_argument('--artemis-vae-lr', type=float, default=0.001)
    parser.add_argument('--artemis-vae-beta', type=float, default=0.001)
    parser.add_argument('--artemis-cache-input-on-gpu', action='store_true', help='Cache the selected dense expression matrix on GPU before VAE training.')
    parser.add_argument('--artemis-cache-eval-input-on-cpu', action='store_true', help='Densify evaluation train/holdout matrices once on CPU before repeated donor/source-age rollouts.')
    parser.add_argument('--artemis-cache-chunk-size', type=int, default=8192, help='Rows loaded per chunk while building the GPU input cache.')
    parser.add_argument('--artemis-steps', type=int, default=100)
    parser.add_argument('--artemis-base-drift', type=float, default=0.0, help='ARTEMIS dataset base-drift f_val passed to the reference Input_Dataset.')
    parser.add_argument('--artemis-bridge-phases', type=int, default=20)
    parser.add_argument('--artemis-epochs-per-phase', type=int, default=1)
    parser.add_argument('--artemis-batch-size', type=int, default=512)
    parser.add_argument('--artemis-lr', type=float, default=0.001)
    parser.add_argument('--artemis-ferryman-lr', type=float, default=0.001)
    parser.add_argument('--artemis-objective', choices=['mean_matching', 'divergence', 'combined'], default='divergence')
    parser.add_argument('--artemis-hidden-dim', default='64')
    parser.add_argument('--artemis-decoder-dim', default='64,1')
    parser.add_argument('--artemis-ferryman-dim', default='64')
    parser.add_argument('--artemis-vae-hidden-layers', default='512,512')
    parser.add_argument('--artemis-time-dim', type=int, default=16)
    parser.add_argument('--artemis-paths-reuse', type=int, default=5)
    parser.add_argument('--artemis-num-sde', type=int, default=10, help='Number of upstream forward/backward SDE passes per training epoch.')
    parser.add_argument('--artemis-epochs', type=int, default=5, help='Upstream Training_Setup epoch default; bridge schedule may override phase granularity.')
    parser.add_argument('--artemis-td-coeff', type=float, default=0.001)
    parser.add_argument('--artemis-reality-coefficient', type=float, default=0.1)
    parser.add_argument('--artemis-splitting-births-frac', type=float, default=0.2)
    parser.add_argument('--artemis-death-importance-rate', type=float, default=100.0)
    parser.add_argument('--artemis-std-threshold', type=float, default=2.0)
    parser.add_argument('--artemis-killer-cutoff', type=float, default=0.2)
    parser.add_argument('--artemis-mb-prior', type=float, default=5.0)
    parser.add_argument('--artemis-ipf-mask-dead', action='store_true')
    parser.add_argument('--artemis-log-every', type=int, default=1)
    return parser.parse_args(argv)

def _normalise_args(args: Any) -> None:
    args.artemis_hidden_dim = parse_int_list(args.artemis_hidden_dim)
    args.artemis_decoder_dim = parse_int_list(args.artemis_decoder_dim)
    args.artemis_ferryman_dim = parse_int_list(args.artemis_ferryman_dim)
    args.artemis_vae_hidden_dim_list = parse_int_list(args.artemis_vae_hidden_layers)
    if args.artemis_decoder_dim[-1] != 1:
        args.artemis_decoder_dim = tuple(args.artemis_decoder_dim) + (1,)

def artemis_config(args: Any) -> dict[str, Any]:
    return {'latent_dim': int(args.artemis_latent_dim), 'vae_hidden_dim': int(args.artemis_vae_hidden_dim), 'vae_hidden_layers': list(args.artemis_vae_hidden_dim_list), 'steps': int(args.artemis_steps), 'base_drift': float(args.artemis_base_drift), 'num_sde': int(args.artemis_num_sde), 'epochs': int(args.artemis_epochs), 'time_normalization': str(args.time_normalization), 'bridge_phases': int(args.artemis_bridge_phases), 'epochs_per_phase': int(args.artemis_epochs_per_phase), 'batch_size': int(args.artemis_batch_size), 'vae_batch_size': int(args.artemis_vae_batch_size), 'cache_input_on_gpu': bool(args.artemis_cache_input_on_gpu), 'cache_eval_input_on_cpu': bool(args.artemis_cache_eval_input_on_cpu), 'cache_chunk_size': int(args.artemis_cache_chunk_size), 'objective': str(args.artemis_objective), 'hidden_dim': list(args.artemis_hidden_dim), 'decoder_dim': list(args.artemis_decoder_dim), 'ferryman_dim': list(args.artemis_ferryman_dim), 'time_dim': int(args.artemis_time_dim), 'paths_reuse': int(args.artemis_paths_reuse), 'td_coeff': float(args.artemis_td_coeff), 'reality_coefficient': float(args.artemis_reality_coefficient), 'splitting_births_frac': float(args.artemis_splitting_births_frac), 'death_importance_rate': float(args.artemis_death_importance_rate), 'std_threshold': float(args.artemis_std_threshold), 'killer_cutoff': float(args.artemis_killer_cutoff), 'mb_prior': float(args.artemis_mb_prior)}

def _apply_checkpoint_config(args: Any, config: dict[str, Any], payload: dict[str, Any]) -> None:
    args.artemis_latent_dim = int(config['latent_dim'])
    args.artemis_vae_hidden_dim = int(config['vae_hidden_dim'])
    args.artemis_vae_hidden_dim_list = parse_int_list(config.get('vae_hidden_layers', '128,128'))
    args.artemis_steps = int(payload.get('artemis_steps', config['steps']))
    args.artemis_base_drift = float(config.get('base_drift', args.artemis_base_drift))
    args.artemis_num_sde = int(config.get('num_sde', args.artemis_num_sde))
    args.artemis_epochs = int(config.get('epochs', args.artemis_epochs))
    args.artemis_bridge_phases = int(config.get('bridge_phases', 1))
    args.artemis_epochs_per_phase = int(config.get('epochs_per_phase', 1))
    args.artemis_batch_size = int(config.get('batch_size', args.artemis_batch_size))
    args.artemis_vae_batch_size = int(config.get('vae_batch_size', args.artemis_vae_batch_size))
    args.artemis_objective = str(config.get('objective', args.artemis_objective))
    args.artemis_hidden_dim = tuple(config.get('hidden_dim', args.artemis_hidden_dim))
    args.artemis_decoder_dim = tuple(config.get('decoder_dim', args.artemis_decoder_dim))
    args.artemis_ferryman_dim = tuple(config.get('ferryman_dim', args.artemis_ferryman_dim))
    args.artemis_time_dim = int(config.get('time_dim', args.artemis_time_dim))
    args.artemis_paths_reuse = int(config.get('paths_reuse', args.artemis_paths_reuse))
    args.artemis_td_coeff = float(config.get('td_coeff', args.artemis_td_coeff))
    args.artemis_reality_coefficient = float(config.get('reality_coefficient', args.artemis_reality_coefficient))
    args.artemis_splitting_births_frac = float(config.get('splitting_births_frac', args.artemis_splitting_births_frac))
    args.artemis_death_importance_rate = float(config.get('death_importance_rate', args.artemis_death_importance_rate))
    args.artemis_std_threshold = float(config.get('std_threshold', args.artemis_std_threshold))
    args.artemis_killer_cutoff = float(config.get('killer_cutoff', args.artemis_killer_cutoff))
    args.artemis_mb_prior = float(config.get('mb_prior', args.artemis_mb_prior))

def _checkpoint_dataset(stats: dict[str, Any], args: Any, modules: dict[str, Any], latent_dim: int) -> Any:
    import pandas as pd
    times_orig = np.asarray(stats['times_orig'], dtype=np.float32)
    times = [int(value) for value in stats['times']]
    mean_by_time = stats['mean']
    rows = []
    for step, time in zip(times, times_orig):
        fallback = mean_by_time.get(str(float(time)))
        value = np.asarray(mean_by_time.get(str(step), fallback), dtype=np.float32).reshape(-1)
        if len(value) != latent_dim:
            raise ValueError(f'Checkpoint dataset statistic for age {time:g} has wrong latent dimension')
        rows.append(value)
    frame = pd.DataFrame(np.asarray(rows, dtype=np.float32))
    frame['time'] = times_orig
    dataset = make_artemis_dataset(frame, modules=modules, steps_num=int(args.artemis_steps), time_normalization=args.time_normalization, splitting_births_frac=args.artemis_splitting_births_frac, death_importance_rate=args.artemis_death_importance_rate, std_threshold=args.artemis_std_threshold, cutoff=args.artemis_killer_cutoff, mb_prior=args.artemis_mb_prior, base_drift=args.artemis_base_drift)
    dataset.times = times
    dataset.mass = [int(value) for value in stats['mass']]
    dataset.mean = {int(key): np.asarray(value, dtype=np.float32) for key, value in stats['mean'].items()}
    dataset.std = {int(key): np.asarray(value, dtype=np.float32) for key, value in stats['std'].items()}
    return dataset

def save_artemis_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path

def load_artemis_checkpoint(path: str | Path, n_genes: int, var_names: Any, args: Any) -> tuple[ArtemisBridge, dict[str, Any]]:
    path = Path(path)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    validate_checkpoint_envelope(payload, path=path, identity=IDENTITY, n_genes=n_genes, var_names=var_names, required_keys=('artemis_config', 'vae_state_dict', 'forward_params', 'backward_params', 'ferryman_params', 'dataset_stats', 'artemis_steps'))
    validate_checkpoint_time_axis(payload, requested_normalization=args.time_normalization)
    _apply_checkpoint_config(args, dict(payload['artemis_config']), payload)
    modules = load_artemis_modules(args.artemis_src)
    import jax
    latent_dim = int(payload['artemis_latent_dim'])
    dataset = _checkpoint_dataset(payload['dataset_stats'], args, modules, latent_dim)
    setup = make_artemis_setup(dataset, modules=modules, args=args, latent_dim=latent_dim, key=jax.random.PRNGKey(int(payload.get('artemis_seed', args.seed))))
    vae = ArtemisVAE(n_genes, latent_dim, int(payload['artemis_config']['vae_hidden_dim']))
    vae.load_state_dict({key: torch.as_tensor(value) for key, value in payload['vae_state_dict'].items()})
    vae.eval()
    params = {modules['utils'].FORWARD: _to_jax_tree(payload['forward_params'], jax.numpy), modules['utils'].BACKWARD: _to_jax_tree(payload['backward_params'], jax.numpy), modules['utils'].FERRYMAN: _to_jax_tree(payload['ferryman_params'], jax.numpy)}
    bridge = ArtemisBridge(modules=modules, setup=setup, params=params, vae=vae, device=torch.device('cpu'), age_min=float(payload['artemis_age_min']), age_max=float(payload['artemis_age_max']), steps_num=int(payload['artemis_steps']), latent_dim=latent_dim, seed=int(payload.get('artemis_seed', args.seed)))
    return (bridge, payload)

def _jax_key(seed: int, *parts: int) -> Any:
    import jax
    key = jax.random.PRNGKey(int(seed))
    for part in parts:
        key = jax.random.fold_in(key, int(part))
    return key

def _resize_rows(values: np.ndarray, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return (values, np.asarray([], dtype=np.int64))
    positions = rng.choice(len(values), size=int(count), replace=len(values) < int(count))
    return (values[positions], np.asarray(positions, dtype=np.int64))

def project_artemis_source(bridge: ArtemisBridge, adata_train: Any, args: Any, source_age: float, indices: np.ndarray, n_cells: int, target_age: float, rng: np.random.Generator, *, key_parts: tuple[int, ...]=(), return_source_indices: bool=False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    x_source = get_expr(adata_train, indices, args)
    with torch.no_grad():
        z_source = bridge.vae.encode(torch.as_tensor(x_source, dtype=torch.float32, device=bridge.device)).cpu().numpy().astype(np.float32)
    result = bridge.rollout(z_source, float(source_age), float(target_age), key=_jax_key(bridge.seed, *key_parts))
    use_alive = result['latent_alive']
    source_positions = result['source_particle_ids'][result['alive']]
    if len(use_alive) < 2:
        use_alive = result['latent_all']
        source_positions = np.arange(len(use_alive), dtype=np.int32)
    selected, positions = _resize_rows(use_alive, n_cells, rng)
    source_positions = np.asarray(source_positions, dtype=np.int64)[positions]
    predicted = decode_streaming(bridge.vae, selected, bridge.device)
    if return_source_indices:
        return (predicted, source_positions)
    return predicted

def _candidate_ages(bridge: ArtemisBridge, source_age: float, age_grid: np.ndarray, args: Any) -> list[float]:
    direction = str(getattr(args, 'source_direction', 'earlier'))
    candidates = []
    for age in age_grid:
        age = float(age)
        if age < bridge.age_min - 1e-06 or age > bridge.age_max + 1e-06:
            continue
        if np.isclose(age, float(source_age)):
            candidates.append(age)
        elif direction == 'bidirectional' or args.unconstrained_age_search:
            candidates.append(age)
        elif age > float(source_age):
            candidates.append(age)
    return candidates

def sweep_and_project_artemis(x_target: np.ndarray, source_idx_global: np.ndarray, source_age: float, target_age: float, age_grid: np.ndarray, adata_train: Any, bridge: ArtemisBridge, args: Any, rng: np.random.Generator, *, device: torch.device, projection_records: list[dict[str, Any]] | None=None, projection_meta: dict[str, Any] | None=None, field_cache: dict[Any, dict[str, Any]] | None=None, field_key: Any=None, target_cell_types: np.ndarray | None=None) -> dict[str, Any] | None:
    if float(target_age) < float(bridge.age_min) - 1e-06 or float(target_age) > float(bridge.age_max) + 1e-06:
        return None
    n = capped_eval_count(args.max_eval_cells, len(source_idx_global), len(x_target))
    if n < 2:
        return None
    target_cell_types_array = None if target_cell_types is None else np.asarray(target_cell_types, dtype=np.int64)
    requested_cell_types = [item.strip() for item in str(getattr(args, 'pca_knn_cell_types', '')).split(',') if item.strip()] or None
    selected_target = select_target_indices_for_pca_knn(target_cell_types_array, n=n, rng=rng, categories=getattr(args, '_pca_knn_categories', []), requested_cell_types=requested_cell_types)
    target_idx = selected_target if selected_target is not None else rng.choice(len(x_target), size=n, replace=False)
    n = len(target_idx)
    if n < 2:
        return None
    with torch.no_grad():
        z_target = bridge.vae.encode(torch.as_tensor(x_target[target_idx], dtype=torch.float32, device=device)).cpu().numpy().astype(np.float32)
    candidate_ages = _candidate_ages(bridge, source_age, age_grid, args)
    if not candidate_ages:
        return None
    if field_cache is None:
        field_cache = {}
    cache_key = (field_key, float(source_age), tuple(candidate_ages))
    field = field_cache.get(cache_key)
    if field is None:
        field_n = int(args.max_eval_cells) if int(args.max_eval_cells) > 0 else len(source_idx_global)
        field_n = min(field_n, len(source_idx_global))
        if field_n < 2:
            return None
        source_idx = rng.choice(len(source_idx_global), size=field_n, replace=len(source_idx_global) < field_n)
        source_global = np.asarray(source_idx_global, dtype=np.int64)[source_idx]
        x_source = get_expr(adata_train, source_global, args)
        with torch.no_grad():
            z_source = bridge.vae.encode(torch.as_tensor(x_source, dtype=torch.float32, device=device)).cpu().numpy().astype(np.float32)
        print(f'[ARTEMIS field] source_age={float(source_age):g} field_cells={field_n} candidate_ages={len(candidate_ages)}', flush=True)
        field = {'field_n': int(field_n), 'results': bridge.rollout_latent_many(z_source, float(source_age), candidate_ages)}
        field_cache[cache_key] = field
    latent_by_age, details_by_age = ({}, {})
    for candidate_age in candidate_ages:
        result = field['results'][float(candidate_age)]
        alive_latent = result['latent_alive']
        if len(alive_latent) < 2:
            alive_latent = result['latent_all']
        positions = rng.choice(len(alive_latent), size=n, replace=len(alive_latent) < n)
        latent_by_age[candidate_age] = alive_latent[positions].astype(np.float32, copy=False)
        details_by_age[candidate_age] = result
    latent_pred = np.stack([latent_by_age[age] for age in candidate_ages], axis=1)
    ns = min(int(args.max_sinkhorn_cells), len(z_target), n)
    if ns < 2:
        return None
    target_tensor = torch.as_tensor(z_target[:ns], dtype=torch.float32, device=device)

    def sinkhorn_score(index: int) -> float:
        prediction = torch.as_tensor(latent_pred[:ns, index, :], dtype=torch.float32, device=device)
        value = SinkhornLoss([target_tensor], prediction.unsqueeze(1), blur=args.projection_sinkhorn_blur, scaling=args.projection_sinkhorn_scaling, batch_size=None)
        return float(value.detach().cpu())
    curve = score_projection_age_grid(list(range(len(candidate_ages))), candidate_ages, target_age, sinkhorn_score=sinkhorn_score, rng=rng)
    chrono_idx = int(curve['chronological_index'])
    t_hat_idx = int(curve['best_sinkhorn_index'])
    chrono_expression = decode_streaming(bridge.vae, latent_by_age[candidate_ages[chrono_idx]], bridge.device)
    t_hat_expression = decode_streaming(bridge.vae, latent_by_age[candidate_ages[t_hat_idx]], bridge.device)
    gene_chrono = sinkhorn_distance(x_target[target_idx][:ns], chrono_expression[:ns], max_cells=ns, blur=args.projection_sinkhorn_blur, scaling=args.projection_sinkhorn_scaling)
    gene_t_hat = sinkhorn_distance(x_target[target_idx][:ns], t_hat_expression[:ns], max_cells=ns, blur=args.projection_sinkhorn_blur, scaling=args.projection_sinkhorn_scaling)
    chrono_detail = details_by_age[candidate_ages[chrono_idx]]
    t_hat_detail = details_by_age[candidate_ages[t_hat_idx]]
    if projection_records is not None:
        snap_n = min(n, int(args.projection_snapshot_max_cells_per_row))
        snap_idx = target_idx[:snap_n] if snap_n else np.asarray([], dtype=np.int64)
        snap_rows = np.arange(snap_n, dtype=np.int64)
        if snap_n:
            chrono_all_expression = decode_streaming(bridge.vae, chrono_detail['latent_all'][:snap_n], bridge.device)
            t_hat_all_expression = decode_streaming(bridge.vae, t_hat_detail['latent_all'][:snap_n], bridge.device)
        else:
            empty_expression = np.empty((0, int(bridge.vae.n_genes)), dtype=np.float32)
            chrono_all_expression = empty_expression
            t_hat_all_expression = empty_expression
        meta = dict(projection_meta or {})
        projection_records.append({'target_expr': x_target[snap_idx].astype(np.float32, copy=True), 'pred_chrono_expr': chrono_expression[snap_rows].astype(np.float32, copy=True), 'pred_sinkhorn_expr': t_hat_expression[snap_rows].astype(np.float32, copy=True), 'target_latent': z_target[snap_rows].astype(np.float32, copy=True), 'pred_chrono_latent': latent_pred[snap_rows, chrono_idx, :].astype(np.float32, copy=True), 'pred_sinkhorn_latent': latent_pred[snap_rows, t_hat_idx, :].astype(np.float32, copy=True), 'pred_all_particle_chrono_expr': chrono_all_expression.astype(np.float32, copy=True), 'pred_all_particle_sinkhorn_expr': t_hat_all_expression.astype(np.float32, copy=True), 'pred_all_particle_chrono_latent': chrono_detail['latent_all'][snap_rows].astype(np.float32, copy=True), 'pred_all_particle_sinkhorn_latent': t_hat_detail['latent_all'][snap_rows].astype(np.float32, copy=True), 'target_age': float(target_age), 'source_age': float(source_age), 't_hat_sinkhorn': float(curve['t_hat_sinkhorn']), 'label': str(meta.get('label', '')), 'group': str(meta.get('group', '')), 'projection_kind': str(meta.get('projection_kind', 'normal_holdout_calibration'))})
    result = {'age_estimate_primary': 'sinkhorn', 't_hat_sinkhorn': curve['t_hat_sinkhorn'], 'sinkhorn_at_t_hat': curve['sinkhorn_at_t_hat'], 'sinkhorn_at_chrono': curve['sinkhorn_at_chrono'], 'sinkhorn_improvement': curve['sinkhorn_improvement'], 'n_sinkhorn_cells': int(ns), 'sinkhorn_space': 'artemis_latent', 'model_native_latent_sinkhorn': float(curve['sinkhorn_at_t_hat']), 'gene_space_sinkhorn_at_chrono': gene_chrono, 'gene_space_sinkhorn_at_latent_t_hat': gene_t_hat, 'decoded_gene_space_sinkhorn': float(gene_t_hat), 'gene_space_sinkhorn_space': 'decoded_expression_alive_primary', 'gene_space_sinkhorn_n_cells': int(ns), 'n_eval_cells': int(n), 'source_age': float(source_age), 'forward_backward_direction': str(chrono_detail['direction']), 'generated_cell_count': int(chrono_detail['generated_cell_count']), 'alive_generated_cell_count': int(chrono_detail['alive_generated_cell_count']), 'survival_ratio': float(chrono_detail['survival_ratio']), 'source_sde_step': int(chrono_detail['source_step']), 'target_sde_step': int(chrono_detail['target_step']), 'jax_seed': int(bridge.seed), 'all_particle_generated_cell_count': int(t_hat_detail['generated_cell_count']), 'all_particle_gene_space_sinkhorn_at_chrono': float(sinkhorn_distance(x_target[target_idx][:ns], t_hat_expression[:ns], max_cells=ns, blur=args.projection_sinkhorn_blur, scaling=args.projection_sinkhorn_scaling))}
    if args.cell_type_readout == 'pca_knn':
        if target_cell_types is None:
            raise ValueError('PCA-KNN localization requires target cell-type labels')
        predicted_expression_by_age = np.stack([decode_streaming(bridge.vae, latent_by_age[age], bridge.device) for age in candidate_ages], axis=1)
        result['per_cell_type'] = compute_pca_knn_cell_type_age_localization(predicted_expression_by_age=predicted_expression_by_age, predicted_latent_by_age=latent_pred, target_expression=x_target[target_idx], target_latent=z_target, target_cell_types=target_cell_types_array[target_idx], candidate_ages=candidate_ages, target_age=target_age, classifier=args._pca_knn_classifier, cell_type_categories=args._pca_knn_categories, requested_cell_types=requested_cell_types, min_cells=args.min_calibration_cells, max_sinkhorn_cells=args.max_sinkhorn_cells, blur=args.projection_sinkhorn_blur, scaling=args.projection_sinkhorn_scaling, rng=rng)
    return result

def _selection_callback(*, args: Any, bridge: ArtemisBridge, source_plan: Any, normal_train: Any, entries: dict[float, np.ndarray], donors_by_age: dict[float, dict[str, np.ndarray]], device: torch.device):
    def callback(phase: int, current_bridge: ArtemisBridge) -> dict[str, Any]:
        if args.epoch_selection_metric != 'holdout_L_gene':
            return {}
        if source_plan.normal_holdout.n_obs < args.normal_holdout_selection_min_cells:
            return {}

        def predict_group(target_age, target_indices, score_rng):
            return pooled_youngest_forward(entries, target_age, len(target_indices), score_rng,
                lambda source_age, indices, count, age: project_artemis_source(current_bridge, normal_train, args,
                    source_age, indices, count, age, score_rng, key_parts=(int(round(age * 100)),)))
        requested_targets = parse_float_targets(args.calibration_targets)
        target_ages = [age for age in requested_targets if current_bridge.age_min - 1e-06 <= float(age) <= current_bridge.age_max + 1e-06]
        skipped_targets = [age for age in requested_targets if age not in target_ages]
        if skipped_targets:
            print(f'[ARTEMIS selection] skipping out-of-domain target ages={skipped_targets} domain=[{current_bridge.age_min:g},{current_bridge.age_max:g}]', flush=True)
        if not target_ages:
            return {}
        return evaluate_holdout_l_gene(source_plan.normal_holdout, age_key=args.age_key, donor_key=args.donor_key, target_ages=target_ages, min_cells=args.normal_holdout_selection_min_cells, max_groups=args.normal_holdout_selection_max_groups, max_cells=args.normal_holdout_selection_max_cells, seed=args.seed, device=device, get_expression=lambda adata, indices: get_expr(adata, indices, args), predict_group=predict_group, blur=args.projection_sinkhorn_blur, scaling=args.projection_sinkhorn_scaling)
    return callback

def run_benchmark(args, source_plan):
    _normalise_args(args)
    args._time_origin = float(source_plan.time_origin)
    args._time_scale = float(source_plan.time_scale)
    run_name, outdir, rng = (source_plan.run_name, source_plan.outdir, source_plan.rng)
    normal_train = source_plan.normal_train
    normal_holdout = source_plan.normal_holdout
    entries, donors_by_age = (source_plan.entries, source_plan.donors_by_age)
    sorted_ages = source_plan.sorted_ages
    age_grid = source_plan.age_grid
    n_genes = int(normal_train.n_vars)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[ARTEMIS] device={device}', flush=True)
    checkpoint_path = Path(args.checkpoint).expanduser() if args.checkpoint else None
    selection_payload: dict[str, Any] = {}
    if checkpoint_path:
        print(f'[ARTEMIS] loading checkpoint {checkpoint_path}', flush=True)
        bridge, checkpoint = load_artemis_checkpoint(checkpoint_path, n_genes=n_genes, var_names=normal_train.var_names, args=args)
        bridge.device = device
        bridge.vae.to(device).eval()
        if args.cross_dataset_eval:
            apply_cross_dataset_age_domain(source_plan, checkpoint, age_key=args.age_key, projection_age_step=args.projection_age_step, projection_max_age=max(parse_float_targets(args.calibration_targets) or source_plan.disease.obs[args.age_key].astype(float).tolist()))
            entries, donors_by_age = (source_plan.entries, source_plan.donors_by_age)
            sorted_ages, age_grid = (source_plan.sorted_ages, source_plan.age_grid)
            args._time_origin, args._time_scale = (source_plan.time_origin, source_plan.time_scale)
        print(f'[ARTEMIS] loaded latent_dim={bridge.latent_dim} steps={bridge.steps_num} age=[{bridge.age_min:g},{bridge.age_max:g}]', flush=True)
    else:
        modules = load_artemis_modules(args.artemis_src)
        import jax
        train_idx = np.concatenate([entries[age] for age in sorted_ages]).astype(np.int64)
        print(f'[ARTEMIS] streaming VAE cells={len(train_idx):,} genes={n_genes} latent_dim={args.artemis_latent_dim}', flush=True)
        vae, latent = fit_streaming_vae(normal_train, train_idx, args, n_genes=n_genes, latent_dim=args.artemis_latent_dim, hidden_dim=args.artemis_vae_hidden_dim, epochs=args.artemis_vae_epochs, batch_size=args.artemis_vae_batch_size, learning_rate=args.artemis_vae_lr, beta=args.artemis_vae_beta, device=device, rng=rng)
        latent_ages = normal_train.obs[args.age_key].astype(float).to_numpy()[train_idx]
        latent_frame = make_latent_frame(latent, latent_ages)
        dataset = make_artemis_dataset(latent_frame, modules=modules, steps_num=args.artemis_steps, time_normalization=args.time_normalization, splitting_births_frac=args.artemis_splitting_births_frac, death_importance_rate=args.artemis_death_importance_rate, std_threshold=args.artemis_std_threshold, cutoff=args.artemis_killer_cutoff, mb_prior=args.artemis_mb_prior, base_drift=args.artemis_base_drift)
        setup = make_artemis_setup(dataset, modules=modules, args=args, latent_dim=args.artemis_latent_dim, key=jax.random.PRNGKey(args.seed))
        trainer = modules['training'].Trainer(dataset, setup, jax.random.PRNGKey(args.seed), lr=args.artemis_lr, ferryman_lr=args.artemis_ferryman_lr, vae_lr=args.artemis_vae_lr)
        bridge = ArtemisBridge(modules=modules, setup=setup, params=setup.state[1], vae=vae, device=device, age_min=0.0 if args.time_normalization == 'age_100' else float(source_plan.age_min), age_max=100.0 if args.time_normalization == 'age_100' else float(source_plan.age_max), steps_num=args.artemis_steps, latent_dim=args.artemis_latent_dim, seed=args.seed)
        callback = _selection_callback(args=args, bridge=bridge, source_plan=source_plan, normal_train=normal_train, entries=entries, donors_by_age=donors_by_age, device=device)
        selection_payload = train_bridge_in_phases(bridge, trainer=trainer, args=args, phase_callback=callback)
        selection_path = outdir / f'{run_name}_artemis_checkpoint_selection.json'
        atomic_write_json(selection_path, selection_payload)
        print(f'[selected] ARTEMIS phase={selection_payload["phase"]}', flush=True)
        if args.checkpoint_dir:
            envelope = build_checkpoint_envelope(IDENTITY, args=args, normal_train=normal_train, train_ages=sorted_ages, age_min=source_plan.age_min, age_max=source_plan.age_max, selection=selection_payload)
            checkpoint = checkpoint_payload(envelope=envelope, bridge=bridge, vae=vae, config=artemis_config(args), selection=selection_payload, stats=dataset_stats(dataset))
            checkpoint_out = Path(args.checkpoint_dir).expanduser() / f'{run_name}_artemis.pt'
            save_artemis_checkpoint(checkpoint_out, checkpoint)
            print(f'[saved] {checkpoint_out}', flush=True)
        if args.train_only:
            print('[ARTEMIS] train-only complete; skipping report metrics', flush=True)
            return
    if args.train_only:
        print('[ARTEMIS] train-only complete; skipping report metrics', flush=True)
        return
    if args.artemis_cache_eval_input_on_cpu:
        eval_cache = {}
        cache_targets = [source_plan.normal_train, source_plan.normal_holdout]
        if not args.calibration_only and source_plan.disease is not None:
            cache_targets.append(source_plan.disease)
        for adata in cache_targets:
            if id(adata) not in eval_cache:
                eval_cache[id(adata)] = build_dense_expression_cache(adata, args, chunk_size=args.artemis_cache_chunk_size)
        args._dense_expression_cache = eval_cache
        print(f'[dense eval cache] ready matrices={len(eval_cache)}; repeated evaluation reads will use dense CPU arrays', flush=True)
    source_age_projection_records: list[dict[str, Any]] = []
    field_cache: dict[Any, dict[str, Any]] = {}

    def project_adapter(x_target, target_age, source_entries, source_donors, projection_args, projection_rng, snapshot_records, snapshot_meta, target_cell_types=None):
        if float(target_age) < float(bridge.age_min) - 1e-06 or float(target_age) > float(bridge.age_max) + 1e-06:
            return None
        specs = rollout_source_specs(source_entries, source_donors, target_age, projection_args, projection_rng)
        field_key = getattr(projection_args, '_artemis_field_key', ('all', str(snapshot_meta.get('group', 'donor')) if snapshot_meta else 'donor'))
        return aggregate_pooled_source_age_projection(source_entries, source_donors, specs, target_age, lambda spec, indices: sweep_and_project_artemis(x_target, indices, spec['source_age'], target_age, age_grid, normal_train, bridge, projection_args, projection_rng, device=device, projection_records=snapshot_records, projection_meta=snapshot_meta, field_cache=field_cache, field_key=field_key, target_cell_types=target_cell_types), source_direction=getattr(projection_args, 'source_direction', 'earlier'), records=source_age_projection_records, record_metadata={'donor_id': snapshot_meta.get('label') if snapshot_meta else None, 'projection_kind': snapshot_meta.get('projection_kind') if snapshot_meta else None, 'projection_unit': args.projection_unit, 'n_target_cells': int(len(x_target))}, include_per_cell_type=args.cell_type_readout == 'pca_knn')

    def rollout_adapter(target_age, source_entries, source_donors, rollout_args, n_cells, rollout_rng):
        if float(target_age) < float(bridge.age_min) - 1e-06 or float(target_age) > float(bridge.age_max) + 1e-06:
            return None
        specs = rollout_source_specs(source_entries, source_donors, target_age, rollout_args, rollout_rng)
        return rollout_source_ensemble(source_entries, specs, target_age, n_cells, rollout_rng, lambda source_age, indices, count: project_artemis_source(bridge, normal_train, rollout_args, source_age, indices, count, target_age, rollout_rng, key_parts=(int(round(target_age * 100)), int(round(source_age * 100))), return_source_indices=True), source_direction=getattr(rollout_args, 'source_direction', 'earlier'), source_donors_by_age=source_donors, return_source_donors=True, return_source_ages=True)
    run_dynamics_evaluation(source_plan=source_plan, args=args, identity=IDENTITY, project=project_adapter, rollout=rollout_adapter, source_age_projection_records=source_age_projection_records)
