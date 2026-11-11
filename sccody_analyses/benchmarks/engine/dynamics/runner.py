from __future__ import annotations
import gc
from copy import copy
from typing import Any, Callable
import numpy as np
from sccody_analyses.benchmarks.engine.dynamics.source_plan import DynamicsSourcePlan

def run_dynamics_evaluation(*, source_plan: DynamicsSourcePlan, args: Any, identity: Any, project: Callable[..., Any], rollout: Callable[..., Any], source_age_projection_records: list[dict[str, Any]] | None=None) -> None:
    """Run shared calibration, disease projection, and artifact serialization."""
    import pandas as pd
    from sccody_analyses.benchmarks.engine.evaluation.populations import evaluate_calibration_rollout, project_calibration_rollout, project_disease_rollout
    from sccody_analyses.benchmarks.engine.artifacts import atomic_write_json
    from sccody_analyses.benchmarks.engine.projection.artifacts import save_holdout_predictions, save_projection_snapshots
    if args.metric_profile == 'hpo_objective':
        import torch
        from sccody_analyses.benchmarks.engine.methods.dynamics import evaluate_holdout_l_gene, get_expr, pooled_youngest_forward
        from sccody_analyses.benchmarks.engine.utilities.parsing import parse_float_targets
        score_args = copy(args)
        score_args.eval_source_policy = 'first_start'
        score_args.rollout_source_policy = 'pooled_by_source_age'
        score_args.rollout_source_ages_per_target = 1
        score_args.random_source_ages_by_target = {}

        def predict_group(target_age, target_indices, rng):
            def forward(source_age, indices, count, age):
                result = rollout(age, {source_age: indices}, source_plan.donors_by_age, score_args, count, rng)
                return result[0] if result is not None else None
            return pooled_youngest_forward(source_plan.entries, target_age, len(target_indices), rng, forward)

        score = evaluate_holdout_l_gene(source_plan.normal_holdout, age_key=args.age_key, donor_key=args.donor_key,
            target_ages=parse_float_targets(args.calibration_targets), min_cells=args.normal_holdout_selection_min_cells,
            max_groups=args.normal_holdout_selection_max_groups, max_cells=args.normal_holdout_selection_max_cells,
            seed=args.seed, device='cuda' if torch.cuda.is_available() else 'cpu',
            get_expression=lambda adata, indices: get_expr(adata, indices, args),
            predict_group=predict_group, blur=args.projection_sinkhorn_blur, scaling=args.projection_sinkhorn_scaling)
        atomic_write_json(source_plan.outdir / f'{source_plan.run_name}_{identity.method_id}_hpo_objective.json',
                          {'metric_profile': 'hpo_objective', **score})
        return
    normal = source_plan.normal
    disease = source_plan.disease
    normal_train = source_plan.normal_train
    normal_holdout = source_plan.normal_holdout
    entries = source_plan.entries
    donors_by_age = source_plan.donors_by_age
    train_ages = source_plan.normal_train_ages
    age_grid = source_plan.age_grid
    run_name = source_plan.run_name
    outdir = source_plan.outdir
    method_id = identity.method_id
    method_name = identity.display_name
    if args.cell_type_readout == 'pca_knn':
        from sccody_analyses.benchmarks.engine.dynamics.pca_knn_localization import fit_train_only_pca_knn_classifier
        classifier, categories = fit_train_only_pca_knn_classifier(normal_train, args)
        args._pca_knn_classifier = classifier
        args._pca_knn_categories = categories
    eval_rng = np.random.default_rng(args.seed)
    prediction_records = [] if args.save_holdout_predictions else None
    snapshot_path = getattr(args, 'save_projection_snapshots', '')
    projection_records = [] if snapshot_path else None
    empty = (pd.DataFrame(), pd.DataFrame())

    def run_calibration_projection():
        frames = project_calibration_rollout(normal_holdout=normal_holdout, entries=entries, donors_by_age=donors_by_age, normal_train=normal_train, age_grid=age_grid, args=args, rng=eval_rng, project=project, projection_records=projection_records)
        calibration, summary = frames
        calibration_path = outdir / f'{run_name}_{method_id}_calibration_projection.csv'
        summary_path = outdir / f'{run_name}_{method_id}_calibration_summary.csv'
        calibration.to_csv(calibration_path, index=False)
        summary.to_csv(summary_path, index=False)
        print(f'[saved] {calibration_path}', flush=True)
        print(f'[saved] {summary_path}', flush=True)
        if source_age_projection_records is not None:
            source_age_frame = pd.DataFrame(source_age_projection_records)
            if source_age_frame.empty:
                raise RuntimeError('Requested pooled source-age projections, but no source-age rows were produced')
            source_age_frame.insert(0, 'method', method_name)
            sort_columns = [column for column in ('true_age', 'donor_id', 'cell_type', 'source_age') if column in source_age_frame]
            if sort_columns:
                source_age_frame = source_age_frame.sort_values(sort_columns, kind='stable').reset_index(drop=True)
            source_age_path = outdir / f'{run_name}_{method_id}_calibration_projection_per_source_age.csv'
            source_age_frame.to_csv(source_age_path, index=False)
            print(f'[saved] {source_age_path}', flush=True)
        return frames
    if normal_holdout.n_obs >= args.min_calibration_cells:
        if args.calibration_objective == 'logfc':
            print('[calibrate] running calibration logFC evaluation on held-out normal donors ...', flush=True)
            logfc_rows = evaluate_calibration_rollout(normal_holdout=normal_holdout, entries=entries, donors_by_age=donors_by_age, normal_train=normal_train, settings=args, rng=eval_rng, run_name=run_name, method=method_name, rollout=rollout, prediction_records=prediction_records, de_reference_indices=source_plan.de_reference_indices, de_reference_adata=source_plan.normal)
            logfc_path = outdir / f'{run_name}_{method_id}_calibration_logfc_summary.json'
            atomic_write_json(logfc_path, logfc_rows)
            print(f'[saved] {logfc_path}', flush=True)
            if args.also_calibration_projection:
                print('[calibrate] additionally running calibration age projection on held-out normal donors ...', flush=True)
                cal_df, cal_summary = run_calibration_projection()
            else:
                cal_df, cal_summary = empty
        else:
            print('[calibrate] running calibration projection on held-out normal donors ...', flush=True)
            cal_df, cal_summary = run_calibration_projection()
    else:
        print(f'[calibrate] skipping — holdout has {normal_holdout.n_obs} cells (min_calibration_cells={args.min_calibration_cells})', flush=True)
        cal_df, cal_summary = empty
    if args.save_holdout_predictions:
        saved = save_holdout_predictions(prediction_records, path=args.save_holdout_predictions, train_obs_names=normal_train.obs_names, gene_names=normal.var_names)
        print(f'[saved] holdout predictions {saved}', flush=True)
    if snapshot_path:
        saved = save_projection_snapshots(projection_records, path=snapshot_path)
        print(f'[saved] projection snapshots {saved}', flush=True)
    if args.calibration_only:
        print('[calibration-only] skipping disease projection / population rollout', flush=True)
        gc.collect()
        return
    disease_label = str(disease.obs[args.disease_key].astype(str).value_counts().index[0])
    print(f'[project] running per-donor disease projection ({disease_label}) ...', flush=True)
    projected = project_disease_rollout(disease=disease, entries=entries, donors_by_age=donors_by_age, normal_train=normal_train, settings=args, rng=eval_rng, disease_label=disease_label, method=method_name, project=project)
    projected.to_csv(source_plan.projection_path, index=False)
    print(f'[saved] {source_plan.projection_path}', flush=True)
    gc.collect()
