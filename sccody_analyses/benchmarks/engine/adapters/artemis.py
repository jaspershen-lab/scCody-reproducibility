"""Streaming and inference helpers for the upstream ARTEMIS bridge.

ARTEMIS' reference scripts are intentionally kept in their own repository.
This adapter supplies the benchmark-facing glue: a bounded PyTorch VAE pass,
latent-array bridge fitting through the upstream IPF trainer, and deterministic
checkpoint/inference helpers.  The large expression matrix is read in
minibatches by default; production runs can explicitly cache the selected
training matrix on the GPU after one sparse-to-dense load.
"""
from __future__ import annotations
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping
import numpy as np
import torch
from torch import nn
from sccody_analyses.benchmarks.engine.methods.dynamics import get_expr
from sccody_analyses.checkpoint_selection import selection_key
DEFAULT_ARTEMIS_ROOT = Path(__file__).resolve().parents[2] / 'vendor' / 'artemis'

class ArtemisVAE(nn.Module):
    """Small deterministic decoder-backed VAE with a streaming training API.

    The bridge only needs a stable latent coordinate and a gene-space decoder.
    Keeping the encoder/decoder in PyTorch lets the expression input remain a
    sparse/backed AnnData object while ARTEMIS operates on compact JAX arrays.
    ``logvar`` is retained in the module for a conventional VAE checkpoint and
    is used during training; inference encodes the posterior mean.
    """

    def __init__(self, n_genes: int, latent_dim: int, hidden_dim: int=512):
        super().__init__()
        n_genes = int(n_genes)
        latent_dim = int(latent_dim)
        hidden_dim = int(hidden_dim)
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(nn.Linear(n_genes, hidden_dim), nn.ReLU())
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_genes))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.mu(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden).clamp(-12.0, 12.0)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return (self.decode(z), mu, logvar)

def resolve_artemis_src(value: str | os.PathLike[str] | None=None) -> Path:
    """Resolve an ARTEMIS ``src`` directory without importing it yet."""
    candidates = []
    if value:
        supplied = Path(value).expanduser()
        candidates.extend([supplied, supplied / 'src'])
    env_value = os.environ.get('ARTEMIS_SRC')
    if env_value:
        supplied = Path(env_value).expanduser()
        candidates.extend([supplied, supplied / 'src'])
    candidates.extend([DEFAULT_ARTEMIS_ROOT / 'src', DEFAULT_ARTEMIS_ROOT])
    for candidate in candidates:
        if (candidate / 'sde.py').is_file():
            return candidate.resolve()
    rendered = ', '.join((str(item) for item in candidates))
    raise FileNotFoundError('Could not locate ARTEMIS src (expected sde.py); checked: ' + rendered)

def load_artemis_modules(src: str | os.PathLike[str] | None=None) -> dict[str, Any]:
    """Import the upstream top-level ARTEMIS modules from a selected checkout."""
    root = str(resolve_artemis_src(src))
    if root not in sys.path:
        sys.path.insert(0, root)
    names = ('utils', 'models', 'datasets', 'sde', 'training_setup', 'training')
    modules = {name: importlib.import_module(name) for name in names}
    modules['src'] = Path(root)
    return modules

def _copy_tree(value: Any) -> Any:
    """Convert JAX/device leaves to host NumPy leaves before serialization."""
    if isinstance(value, dict):
        return {key: _copy_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple((_copy_tree(item) for item in value))
    if isinstance(value, list):
        return [_copy_tree(item) for item in value]
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    if hasattr(value, '__array__'):
        try:
            return np.array(value, copy=True)
        except (TypeError, ValueError):
            pass
    return value

def _jax_key(seed: int, *parts: int) -> Any:
    """Create a deterministic JAX key without coupling the adapter to benchmarks."""
    jax = importlib.import_module('jax')
    key = jax.random.PRNGKey(int(seed))
    for part in parts:
        key = jax.random.fold_in(key, int(part))
    return key

def _to_jax_tree(value: Any, jnp: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_jax_tree(item, jnp) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple((_to_jax_tree(item, jnp) for item in value))
    if isinstance(value, list):
        return [_to_jax_tree(item, jnp) for item in value]
    if isinstance(value, np.ndarray):
        return jnp.asarray(value)
    return value


def _iter_batches(adata: Any, indices: np.ndarray, args: Any, batch_size: int, *, rng: np.random.Generator | None=None, shuffle: bool=False) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    indices = np.asarray(indices, dtype=np.int64)
    order = np.arange(len(indices), dtype=np.int64)
    if shuffle and rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), int(batch_size)):
        locations = order[start:start + int(batch_size)]
        selected = indices[locations]
        yield (selected, get_expr(adata, selected, args))

def _iter_cached_batches(cache: torch.Tensor, batch_size: int, *, rng: np.random.Generator | None=None, shuffle: bool=False) -> Iterable[tuple[np.ndarray, torch.Tensor]]:
    """Iterate rows from a GPU-resident expression cache."""
    n_rows = int(cache.shape[0])
    order = np.arange(n_rows, dtype=np.int64)
    if shuffle and rng is not None:
        rng.shuffle(order)
    for start in range(0, n_rows, int(batch_size)):
        locations = order[start:start + int(batch_size)]
        positions = torch.as_tensor(locations, dtype=torch.long, device=cache.device)
        yield (locations, cache.index_select(0, positions))

def cache_expression_on_gpu(adata: Any, train_indices: np.ndarray, args: Any, *, n_genes: int, device: torch.device, chunk_size: int) -> torch.Tensor:
    """Load selected expression rows once, densify once, and retain on GPU.

    H5AD inputs are commonly backed CSR matrices.  Reading sorted row chunks
    here avoids repeating sparse random reads and ``toarray()`` conversion for
    every VAE epoch.  The cache follows ``train_indices`` order so latent rows
    remain aligned with downstream age/donor metadata.
    """
    if device.type != 'cuda':
        raise RuntimeError('--artemis-cache-input-on-gpu requires a CUDA device')
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if len(train_indices) == 0:
        raise ValueError('Cannot cache an empty expression matrix')
    chunk_size = max(1, int(chunk_size))
    required_bytes = int(len(train_indices)) * int(n_genes) * np.dtype(np.float32).itemsize
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    print(f'[ARTEMIS cache] allocating cells={len(train_indices):,} genes={int(n_genes):,} size_gb={required_bytes / 1024 ** 3:.2f} free_gb={free_bytes / 1024 ** 3:.2f} device={torch.cuda.get_device_name(device)}', flush=True)
    if required_bytes > int(free_bytes * 0.9):
        raise RuntimeError(f'ARTEMIS GPU input cache needs {required_bytes / 1024 ** 3:.2f} GiB but only {free_bytes / 1024 ** 3:.2f} GiB is free')
    cache = torch.empty((len(train_indices), int(n_genes)), dtype=torch.float32, device=device)
    sorted_positions = np.argsort(train_indices, kind='stable')
    for start in range(0, len(sorted_positions), chunk_size):
        positions = sorted_positions[start:start + chunk_size]
        selected = train_indices[positions]
        values = get_expr(adata, selected, args)
        values_tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
        cache.index_copy_(0, torch.as_tensor(positions, dtype=torch.long, device=device), values_tensor)
    print('[ARTEMIS cache] ready; VAE epochs will reuse GPU-resident input', flush=True)
    return cache

def fit_streaming_vae(adata: Any, train_indices: np.ndarray, args: Any, *, n_genes: int, latent_dim: int, hidden_dim: int, epochs: int, batch_size: int, learning_rate: float, beta: float, device: torch.device, rng: np.random.Generator) -> tuple[ArtemisVAE, np.ndarray]:
    """Fit the expression VAE, optionally reusing a GPU-resident input cache."""
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if len(train_indices) == 0:
        raise ValueError('Cannot train ARTEMIS VAE on an empty population')
    gpu_cache = None
    if bool(getattr(args, 'artemis_cache_input_on_gpu', False)):
        gpu_cache = cache_expression_on_gpu(adata, train_indices, args, n_genes=n_genes, device=device, chunk_size=int(getattr(args, 'artemis_cache_chunk_size', 8192)))
    model = ArtemisVAE(n_genes, latent_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    for epoch in range(int(epochs)):
        model.train()
        losses = []
        if gpu_cache is None:
            batches = _iter_batches(adata, train_indices, args, batch_size, rng=rng, shuffle=True)
        else:
            batches = _iter_cached_batches(gpu_cache, batch_size, rng=rng, shuffle=True)
        for _selected, values in batches:
            batch = values if isinstance(values, torch.Tensor) else torch.as_tensor(values, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, mu, logvar = model(batch)
            reconstruction_loss = torch.mean((reconstruction - batch) ** 2)
            kl = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
            loss = reconstruction_loss + float(beta) * kl
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 0 or (epoch + 1) % max(1, int(args.artemis_log_every)) == 0:
            print(f'[ARTEMIS VAE] epoch={epoch + 1}/{int(epochs)} loss={np.mean(losses):.6f} cells={len(train_indices)}', flush=True)
    return (model.eval(), encode_streaming(adata, train_indices, args, model, batch_size, device, gpu_cache=gpu_cache))

def encode_streaming(adata: Any, indices: np.ndarray, args: Any, model: ArtemisVAE, batch_size: int, device: torch.device, *, gpu_cache: torch.Tensor | None=None) -> np.ndarray:
    """Encode every selected cell in bounded chunks, retaining row alignment."""
    indices = np.asarray(indices, dtype=np.int64)
    latent = np.empty((len(indices), model.latent_dim), dtype=np.float32)
    model.eval()
    offset = 0
    with torch.no_grad():
        if gpu_cache is None:
            batches = _iter_batches(adata, indices, args, batch_size)
        else:
            batches = _iter_cached_batches(gpu_cache, batch_size)
        for _selected, values in batches:
            batch = values if isinstance(values, torch.Tensor) else torch.as_tensor(values, dtype=torch.float32, device=device)
            encoded = model.encode(batch).detach().cpu().numpy().astype(np.float32)
            latent[offset:offset + len(encoded)] = encoded
            offset += len(encoded)
    return latent

def decode_streaming(model: ArtemisVAE, latent: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        values = model.decode(torch.as_tensor(np.array(latent, dtype=np.float32, copy=True), device=device))
    return values.detach().cpu().numpy().astype(np.float32)

def make_latent_frame(latent: np.ndarray, ages: np.ndarray) -> Any:
    import pandas as pd
    frame = pd.DataFrame(np.asarray(latent, dtype=np.float32))
    frame['time'] = np.asarray(ages, dtype=np.float32)
    return frame

def make_artemis_dataset(latent_frame: Any, *, modules: Mapping[str, Any], steps_num: int, time_normalization: str='fit_minmax', splitting_births_frac: float, death_importance_rate: float, std_threshold: float, cutoff: float, mb_prior: float, base_drift: float=0.0) -> Any:
    dataset_cls = modules['datasets'].Input_Dataset
    dataset = dataset_cls(latent_frame, meta=None, splitting_births_frac=float(splitting_births_frac), steps_num=int(steps_num), death_importance_rate=float(death_importance_rate), std_threshold=float(std_threshold), cutoff=float(cutoff), mb_prior=float(mb_prior), f_val=float(base_drift))
    if str(time_normalization) == 'age_100':
        jnp = importlib.import_module('jax.numpy')
        times_orig = [float(value) for value in dataset.times_orig]
        fixed_times = [int(np.rint(age / 100.0 * int(steps_num))) for age in times_orig]
        if any((step < 0 or step > int(steps_num) for step in fixed_times)):
            raise ValueError('ARTEMIS age_100 requires observed ages in [0, 100]')
        grouped_ages: dict[int, list[float]] = {}
        for age, step in zip(times_orig, fixed_times):
            grouped_ages.setdefault(step, []).append(age)
        if len(grouped_ages) != len(times_orig):
            collision_count = len(times_orig) - len(grouped_ages)
            print(f'[ARTEMIS] age_100 grouped {collision_count} sub-resolution age bins into {len(grouped_ages)} SDE nodes (steps={int(steps_num)})', flush=True)
        time_x = {}
        grouped_times_orig = []
        age_to_group = {}
        for step in sorted(grouped_ages):
            ages = grouped_ages[step]
            rows = dataset.x.loc[dataset.x['time'].isin(ages)].iloc[:, :-1].to_numpy(dtype=float)
            time_x[step] = jnp.asarray(rows)
            representative = float(np.mean(ages))
            grouped_times_orig.append(representative)
            for age in ages:
                age_to_group[age] = representative
        dataset.x = dataset.x.copy()
        dataset.x['time'] = dataset.x['time'].map(age_to_group).astype(float)
        dataset.times_orig = grouped_times_orig
        dataset.time_x = time_x
        dataset.times = sorted(time_x)
        dataset.mass = [int(time_x[step].shape[0]) for step in dataset.times]
        dataset.mean = {step: jnp.mean(time_x[step], axis=0, keepdims=True) for step in dataset.times}
        dataset.std = {step: jnp.std(time_x[step], axis=0, keepdims=True) for step in dataset.times}
        print(f'[ARTEMIS] fixed age_100 grid observed ages=[{times_orig[0]:g},{times_orig[-1]:g}] steps=[{dataset.times[0]},{dataset.times[-1]}]', flush=True)
    elif str(time_normalization) != 'fit_minmax':
        raise ValueError(f'Unknown ARTEMIS time normalization {time_normalization!r}')
    return dataset

def make_artemis_setup(dataset: Any, *, modules: Mapping[str, Any], args: Any, latent_dim: int, key: Any) -> Any:
    setup_cls = modules['training_setup'].Training_Setup
    setup = setup_cls(dataset, steps_num=int(args.artemis_steps), epochs=int(args.artemis_epochs), vae_epochs=int(args.artemis_vae_epochs), key=key, params=None, objective=str(args.artemis_objective), batch_size=int(args.artemis_batch_size), hidden_dim=list(args.artemis_hidden_dim), dec_hidden_size=list(args.artemis_decoder_dim), ferryman_hidden_dim=list(args.artemis_ferryman_dim), ferryman_activate_final=True, ipf_mask_dead=bool(args.artemis_ipf_mask_dead), reality_coefficient=float(args.artemis_reality_coefficient), paths_reuse=int(args.artemis_paths_reuse), num_sde=int(args.artemis_num_sde), resnet=False, t_dim=int(args.artemis_time_dim), vae_input_dim=int(latent_dim), vae_enc_hidden_dim=list(args.artemis_vae_hidden_dim_list), vae_dec_hidden_dim=list(args.artemis_vae_hidden_dim_list), vae_t_dim=8, calc_latent_loss=True, calc_recon_loss=True, vae_latent_dim=int(latent_dim), vae_batch_size=int(args.artemis_vae_batch_size))
    setup.train_start_step = int(dataset.times[0])
    setup.train_end_step = int(dataset.times[-1])
    return setup

@dataclass
class ArtemisBridge:
    """Inference wrapper around ARTEMIS forward/backward score networks."""
    modules: Mapping[str, Any]
    setup: Any
    params: Mapping[str, Any]
    vae: ArtemisVAE
    device: torch.device
    age_min: float
    age_max: float
    steps_num: int
    latent_dim: int
    seed: int

    def __post_init__(self) -> None:
        self.params = dict(self.params)

    def update_params(self, params: Mapping[str, Any]) -> None:
        self.params = params

    def age_to_step(self, age: float) -> int:
        age = float(age)
        if age < self.age_min - 1e-06 or age > self.age_max + 1e-06:
            raise ValueError(f'Age {age:g} is outside ARTEMIS inference domain [{self.age_min:g}, {self.age_max:g}]')
        span = max(self.age_max - self.age_min, 1e-12)
        return int(np.clip(np.rint((age - self.age_min) / span * self.steps_num), 0, self.steps_num))

    def _score_and_ferryman(self) -> tuple[dict[str, Any], Callable[..., Any]]:
        jax = importlib.import_module('jax')
        from functools import partial
        utils = self.modules['utils']
        setup = self.setup
        score = {utils.FORWARD: partial(setup.score(setup.model[utils.FORWARD]), self.params[utils.FORWARD]), utils.BACKWARD: partial(setup.score(setup.model[utils.BACKWARD]), self.params[utils.BACKWARD])}
        ferryman = partial(setup.ferryman.apply, params=self.params[utils.FERRYMAN], direction=utils.FORWARD)
        del jax
        return (score, ferryman)

    def rollout(self, x_source_latent: np.ndarray, source_age: float, target_age: float, *, key: Any | None=None, return_all: bool=True) -> dict[str, Any]:
        jax = importlib.import_module('jax')
        utils = self.modules['utils']
        source_step = self.age_to_step(source_age)
        target_step = self.age_to_step(target_age)
        if np.isclose(float(source_age), float(target_age)):
            latent = np.asarray(x_source_latent, dtype=np.float32).copy()
            alive = np.ones(len(latent), dtype=bool)
            ids = np.arange(len(latent), dtype=np.int32)
            direction = 'identity'
        else:
            direction = 'forward' if target_step > source_step else 'backward'
            direction_key = utils.FORWARD if direction == 'forward' else utils.BACKWARD
            score, ferryman = self._score_and_ferryman()
            if key is None:
                key = jax.random.PRNGKey(int(self.seed))
            result = self.setup.sde.sample_trajectory(key, direction_key, jax.numpy.asarray(x_source_latent, dtype=jax.numpy.float32), score, ferryman, t_0=source_step, t_end=target_step, corrector='', test=True, max_size=len(x_source_latent), return_particle_ids=True)
            trajectory, _unused, statuses, _births, particle_ids = result
            latent = np.asarray(trajectory[target_step, :len(x_source_latent)])
            alive = np.asarray(statuses[target_step, :len(x_source_latent)], dtype=bool)
            ids = np.asarray(particle_ids[target_step, :len(x_source_latent)], dtype=np.int32)
        decoded = decode_streaming(self.vae, latent, self.device)
        alive_count = int(alive.sum())
        return {'latent_all': latent.astype(np.float32, copy=False), 'expression_all': decoded, 'alive': alive, 'latent_alive': latent[alive].astype(np.float32, copy=False), 'expression_alive': decoded[alive].astype(np.float32, copy=False), 'source_particle_ids': ids, 'direction': direction, 'source_step': int(source_step), 'target_step': int(target_step), 'generated_cell_count': int(len(latent)), 'alive_generated_cell_count': alive_count, 'survival_ratio': float(alive_count / max(1, len(latent))), 'all_particles': bool(return_all)}

    def rollout_latent_many(self, x_source_latent: np.ndarray, source_age: float, target_ages: list[float] | np.ndarray) -> dict[float, dict[str, Any]]:
        """Sample reusable latent trajectories for all requested ages.

        A single source batch has one forward path for ages above the source
        and one backward path for ages below it.  The returned states stay in
        latent space so inverse-age localization can compare many donors
        against the same source field without decoding or re-integrating it.
        """
        jax = importlib.import_module('jax')
        utils = self.modules['utils']
        source_age = float(source_age)
        requested = sorted({float(age) for age in target_ages})
        if not requested:
            return {}
        source_step = self.age_to_step(source_age)
        results: dict[float, dict[str, Any]] = {}
        latent_source = np.asarray(x_source_latent, dtype=np.float32)
        if any((np.isclose(age, source_age) for age in requested)):
            results[source_age] = {'latent_all': latent_source.copy(), 'alive': np.ones(len(latent_source), dtype=bool), 'latent_alive': latent_source.copy(), 'source_particle_ids': np.arange(len(latent_source), dtype=np.int32), 'direction': 'identity', 'source_step': int(source_step), 'target_step': int(source_step), 'generated_cell_count': int(len(latent_source)), 'alive_generated_cell_count': int(len(latent_source)), 'survival_ratio': 1.0, 'all_particles': True, 'jax_seed': int(self.seed), 'jax_key': None}
        direction_groups = {'forward': [age for age in requested if age > source_age], 'backward': [age for age in requested if age < source_age]}
        for direction, ages in direction_groups.items():
            if not ages:
                continue
            direction_key = utils.FORWARD if direction == 'forward' else utils.BACKWARD
            target_step = max((self.age_to_step(age) for age in ages)) if direction == 'forward' else min((self.age_to_step(age) for age in ages))
            key = _jax_key(self.seed, int(round(source_age * 100)), 1 if direction == 'forward' else 2, int(target_step))
            score, ferryman = self._score_and_ferryman()
            sampled = self.setup.sde.sample_trajectory(key, direction_key, jax.numpy.asarray(latent_source, dtype=jax.numpy.float32), score, ferryman, t_0=source_step, t_end=target_step, corrector='', test=True, max_size=len(latent_source), return_particle_ids=True)
            trajectory, _unused, statuses, _births, particle_ids = sampled
            for age in ages:
                step = self.age_to_step(age)
                latent = np.asarray(trajectory[step, :len(latent_source)])
                alive = np.asarray(statuses[step, :len(latent_source)], dtype=bool)
                results[age] = {'latent_all': latent.astype(np.float32, copy=False), 'alive': alive, 'latent_alive': latent[alive].astype(np.float32, copy=False), 'source_particle_ids': np.asarray(particle_ids[step, :len(latent_source)], dtype=np.int32), 'direction': direction, 'source_step': int(source_step), 'target_step': int(step), 'generated_cell_count': int(len(latent)), 'alive_generated_cell_count': int(alive.sum()), 'survival_ratio': float(alive.sum() / max(1, len(latent))), 'all_particles': True}
        return results

def train_bridge_in_phases(bridge: ArtemisBridge, *, trainer: Any, args: Any, phase_callback: Callable[[int, ArtemisBridge], Mapping[str, Any] | None] | None=None, checkpoint_callback: Callable[[int, ArtemisBridge, Mapping[str, Any], bool], None] | None=None) -> dict[str, Any]:
    """Run one fixed-size IPF phase at a time and expose selection checkpoints."""
    key, params = trainer.training_setup.state
    opt_state = trainer.opt_state
    metric = str(getattr(args, 'epoch_selection_metric', 'holdout_L_gene'))
    best_params = None
    best_key = None
    best_phase = None
    best_score = None
    phases = max(1, int(args.artemis_bridge_phases))
    for phase in range(1, phases + 1):
        key, params, opt_state = trainer.training_phase(key, params, opt_state, td_coeff=float(args.artemis_td_coeff), epchs_num=int(args.artemis_epochs_per_phase))
        trainer.training_setup.state = (key, params)
        trainer.opt_state = opt_state
        bridge.update_params(params)
        record = dict(phase_callback(phase, bridge) or {}) if phase_callback else {}
        score = float(record.get('selection_score', np.inf))
        record.update({'phase': phase, 'selection_score': score})
        candidate_key = selection_key({**record, 'epoch': phase}, metric)
        is_best = candidate_key is not None and (best_key is None or candidate_key < best_key)
        if is_best:
            best_key = candidate_key
            best_phase = phase
            best_score = score
            best_params = _copy_tree(params)
        if checkpoint_callback is not None:
            checkpoint_callback(phase, bridge, record, is_best)
    if best_params is None:
        raise RuntimeError(f'Training finished without an eligible {metric} ARTEMIS checkpoint')
    bridge.update_params(_to_jax_tree(best_params, importlib.import_module('jax').numpy))
    return {'metric': metric, 'phase': int(best_phase), 'selection_score': best_score if metric == 'holdout_L_gene' else 0.0, 'source_policy': 'pooled_youngest_forward'}

def dataset_stats(dataset: Any) -> dict[str, Any]:
    return {'times': [int(value) for value in dataset.times], 'mass': [int(value) for value in dataset.mass], 'mean': {str(key): _copy_tree(value) for key, value in dataset.mean.items()}, 'std': {str(key): _copy_tree(value) for key, value in dataset.std.items()}, 'times_orig': [float(value) for value in dataset.times_orig]}

def checkpoint_payload(*, envelope: Mapping[str, Any], bridge: ArtemisBridge, vae: ArtemisVAE, config: Mapping[str, Any], selection: Mapping[str, Any], stats: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(envelope)
    payload.update({'format_version': max(2, int(payload.get('format_version', 1))), 'artemis_config': dict(config), 'artemis_steps': int(bridge.steps_num), 'artemis_age_min': float(bridge.age_min), 'artemis_age_max': float(bridge.age_max), 'artemis_latent_dim': int(bridge.latent_dim), 'artemis_seed': int(bridge.seed), 'vae_state_dict': _copy_tree(vae.state_dict()), 'forward_params': _copy_tree(bridge.params[bridge.modules['utils'].FORWARD]), 'backward_params': _copy_tree(bridge.params[bridge.modules['utils'].BACKWARD]), 'ferryman_params': _copy_tree(bridge.params[bridge.modules['utils'].FERRYMAN]), 'dataset_stats': dict(stats), 'checkpoint_selection': dict(selection)})
    return payload
