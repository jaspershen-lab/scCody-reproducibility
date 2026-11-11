from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any
import numpy as np
from sccody_analyses.benchmarks.engine.methods.dynamics import build_donors_by_age, prepare_pair, safe_name, split_normal_holdout_donors
from sccody_analyses.benchmarks.engine.projection.manifests import RandomSourceAgeManifest, load_matched_source_manifest
from sccody_analyses.benchmarks.engine.utilities.process import repository_root
from sccody_analyses.benchmarks.engine.time_axis import resolve_time_axis
from sccody_analyses.benchmarks.engine.de_reference import build_dataset_de_reference, load_dataset_config
from sccody_analyses.benchmarks.engine.dynamics.identity import BenchmarkMethodIdentity

@dataclass
class DynamicsSourcePlan:
    run_name: str
    outdir: Path
    projection_path: Path
    skipped: bool
    rng: np.random.Generator
    normal_path: Path | None = None
    disease_path: Path | None = None
    normal: Any | None = None
    disease: Any | None = None
    normal_train: Any | None = None
    normal_holdout: Any | None = None
    normal_train_ages: np.ndarray | None = None
    entries: dict[float, np.ndarray] | None = None
    donors_by_age: dict[float, dict[str, np.ndarray]] | None = None
    sorted_ages: list[float] | None = None
    age_min: float | None = None
    age_max: float | None = None
    age_grid: np.ndarray | None = None
    time_normalization: str = 'fit_minmax'
    time_origin: float | None = None
    time_scale: float | None = None
    de_reference_indices: np.ndarray | None = None

def validate_common_arguments(args: Any) -> None:
    stage = getattr(args, 'evaluation_stage', 'full')
    if stage != 'full':
        if not args.checkpoint or args.train_only:
            raise ValueError('Independent evaluation stages require --checkpoint')
        if stage != 'localization' and (not args.calibration_only):
            raise ValueError('Endpoint generation/DE stages require --calibration-only; disease comparisons use their own stage entrypoint')
        args.metric_profile = 'full'
        args.calibration_objective = 'projection' if stage == 'localization' else 'logfc'
        args.also_calibration_projection = False
        if stage != 'expression_de':
            args.de_method = 'none'
    if args.cell_type_readout == 'pca_knn' and args.projection_unit != 'donor':
        raise ValueError('--cell-type-readout pca_knn requires --projection-unit donor; use --projection-unit donor_cell_type for the filter-first analysis')
    if args.train_only and (not args.checkpoint_dir):
        raise ValueError('--train-only requires --checkpoint-dir')
    if args.train_only and args.checkpoint:
        raise ValueError('--train-only cannot be combined with --checkpoint')
    if args.cross_dataset_eval and (not args.checkpoint):
        raise ValueError('--cross-dataset-eval requires --checkpoint')
    if args.cross_dataset_eval and args.train_only:
        raise ValueError('--cross-dataset-eval cannot be combined with --train-only')
    if args.full_forward_cohort_eval:
        if not args.cross_dataset_eval:
            raise ValueError('--full-forward-cohort-eval requires --cross-dataset-eval')
        if args.normal_holdout_donors_file:
            raise ValueError('--full-forward-cohort-eval cannot use --normal-holdout-donors-file')
        if not args.matched_source_ages_file:
            raise ValueError('--full-forward-cohort-eval requires --matched-source-ages-file')
    if args.epoch_selection_metric == 'holdout_L_gene' and (not args.normal_holdout_donors_file) and (not args.checkpoint):
        raise ValueError(f'--epoch-selection-metric {args.epoch_selection_metric} requires a holdout manifest')
    if args.metric_profile != 'hpo_objective' and stage != 'generation' and args.calibration_objective == 'logfc' and (args.de_method != 'edger_pseudobulk'):
        raise ValueError('Full logFC experiments require --de-method edger_pseudobulk; use --metric-profile hpo_objective --de-method none only for HPO.')
    if args.metric_profile != 'hpo_objective' and args.calibration_objective == 'logfc' and (not args.dataset_config):
        raise ValueError('LogFC evaluation requires a dataset config owning canonical sc_ref')
    if stage == 'full' and args.metric_profile == 'full' and (args.calibration_objective == 'logfc'):
        args.also_calibration_projection = True

def prepare_source_plan(args: Any, identity: BenchmarkMethodIdentity, *, validate_input_layer: bool) -> DynamicsSourcePlan:
    """Own common validation, loading, splitting, and source-plan construction."""
    import scanpy as sc
    import torch
    validate_common_arguments(args)
    random_manifest = RandomSourceAgeManifest.load(args.random_source_ages_file)
    args.random_source_ages_by_target = random_manifest.as_dict() if random_manifest is not None else {}
    args.matched_source_manifest = load_matched_source_manifest(args.matched_source_ages_file)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if identity.method_id == 'scnode':
        torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(args.seed)
    normal_path = Path(args.normal_h5ad).resolve()
    disease_path = Path(args.disease_h5ad).resolve()
    run_name = safe_name(args.run_name or f'{normal_path.stem}__{disease_path.stem}__{identity.method_id}')
    default_root = repository_root(__file__) / f'disease_age_res_{identity.method_id}'
    outdir = Path(args.outdir) if args.outdir else default_root / run_name
    outdir.mkdir(parents=True, exist_ok=True)
    projection_path = outdir / f'{run_name}_{identity.method_id}_disease_projection.csv'
    if args.skip_existing and projection_path.exists() and (projection_path.stat().st_size > 0) and (not args.overwrite):
        print(f'[skip] existing {projection_path}', flush=True)
        return DynamicsSourcePlan(run_name=run_name, outdir=outdir, projection_path=projection_path, skipped=True, rng=rng, normal_path=normal_path, disease_path=disease_path)
    read_backed_artemis = identity.method_id == 'artemis'
    preserve_backed_artemis = read_backed_artemis and (not bool(args.cell_types))
    read_kwargs = {'backed': 'r'} if read_backed_artemis else {}
    normal = sc.read_h5ad(normal_path, **read_kwargs)
    train_only = bool(args.train_only)
    if read_backed_artemis and (train_only or disease_path == normal_path):
        disease = normal
    else:
        disease = sc.read_h5ad(disease_path, **read_kwargs)
    normal, disease = prepare_pair(normal, disease, args, preserve_backed=preserve_backed_artemis)
    if read_backed_artemis and train_only:
        disease = None
    if validate_input_layer and args.layer:
        if args.layer not in normal.layers:
            raise KeyError(f'Normal AnnData missing layer {args.layer!r}')
        if disease is not None and args.layer not in disease.layers:
            raise KeyError(f'Disease AnnData missing layer {args.layer!r}')
    if disease is None:
        print(f'[load] normal={normal.shape} disease=SKIPPED (train-only)', flush=True)
    else:
        print(f'[load] normal={normal.shape} disease={disease.shape}', flush=True)
    if args.full_forward_cohort_eval:
        normal_train = normal
        normal_holdout = normal
    else:
        normal_train, normal_holdout = split_normal_holdout_donors(normal, disease if disease is not None else normal, args, rng, preserve_backed=preserve_backed_artemis)
    de_reference_indices = None
    if args.dataset_config and args.metric_profile != 'hpo_objective':
        spec = load_dataset_config(args.dataset_config)
        if Path(spec.h5ad).expanduser().resolve() != normal_path:
            raise ValueError(f'Canonical DE reference dataset config must point to the normal H5AD: {spec.h5ad!r} != {normal_path!s}')
        de_reference_indices = build_dataset_de_reference(spec, ages=normal.obs[args.age_key].astype(float).to_numpy(), obs=normal.obs)
    print(f"[holdout] train_cells={normal_train.n_obs} holdout_cells={normal_holdout.n_obs}", flush=True)
    train_ages = normal_train.obs[args.age_key].astype(float).to_numpy()
    entries = {}
    for age in sorted(np.unique(train_ages)):
        indices = np.where(train_ages == age)[0]
        if len(indices) >= args.min_normal_cells:
            entries[float(age)] = indices
    if len(entries) < 2:
        raise RuntimeError(f'Need at least two normal train age bins for {identity.display_name}')
    donors = build_donors_by_age(normal_train, entries, args.donor_key)
    sorted_ages = sorted(entries)
    age_min = float(min(sorted_ages))
    age_max = float(max(sorted_ages))
    time_origin, time_scale = resolve_time_axis(args.time_normalization, age_min=age_min, age_max=age_max)
    print(f'[time] {args.time_normalization} origin={time_origin:g} scale={time_scale:g} years', flush=True)
    grid_min = getattr(args, 'projection_min_age', None)
    grid_max = getattr(args, 'projection_max_age', None)
    grid_min = age_min if grid_min is None else float(grid_min)
    grid_max = age_max if grid_max is None else float(grid_max)
    age_grid = np.arange(grid_min, grid_max + 1e-08, args.projection_age_step)
    return DynamicsSourcePlan(run_name=run_name, outdir=outdir, projection_path=projection_path, skipped=False, rng=rng, normal_path=normal_path, disease_path=disease_path, normal=normal, disease=disease, normal_train=normal_train, normal_holdout=normal_holdout, normal_train_ages=train_ages, entries=entries, donors_by_age=donors, sorted_ages=sorted_ages, age_min=age_min, age_max=age_max, age_grid=age_grid, time_normalization=args.time_normalization, time_origin=time_origin, time_scale=time_scale, de_reference_indices=de_reference_indices)
