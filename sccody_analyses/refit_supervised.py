"""Refit the selected cutoff-5 age-clock penalties on the AIDA development data.

Uses the preprocessing in clocks.py and the fit functions in supervised.py; no hyperparameter search.
All preprocessing remains fit to each fold's training donors. Save fitted
estimators and donor predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

import sccody_analyses.clocks as clock
import sccody_analyses.supervised as shared
from sccody_analyses.supervised import prepare_split as prepare_target_split


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fold-manifest-template', required=True)
    p.add_argument('--external-manifest-template', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--train', type=Path, required=True)
    p.add_argument('--full', type=Path, required=True)
    p.add_argument('--settings', type=Path, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--covid-input-dir', type=Path)
    args = p.parse_args()
    clock.TRAIN_MANIFEST_TEMPLATE = args.fold_manifest_template
    clock.EXTERNAL_MANIFEST_TEMPLATE = args.external_manifest_template
    config = json.loads(args.settings.read_text())['supervised_models'][args.model]
    selected = config['selected_parameters']
    protocol = config['protocol']
    min_cells = int(protocol['min_cells']); max_iter = int(protocol['max_iter'])
    by_type = args.model.endswith('cell_type') and not args.model.startswith('covid_clock')
    mode = 'percelltype' if by_type or args.model in ['tage', 'covid_clock_cell_type'] else 'whole_population'
    args.outdir.mkdir(parents=True, exist_ok=True)
    clock.TRAIN_H5AD = args.train; clock.EXTERNAL_H5AD = args.full
    train_obs, train_counts, genes = clock.load_obs_counts(args.train)
    test_obs, test_counts, test_genes = clock.load_obs_counts(args.full)
    train_counts, test_counts, genes = clock.align_gene_matrices(train_counts, genes, test_counts, test_genes)
    train_age = clock.donor_age_map(train_obs, 'development')
    test_age = clock.donor_age_map(test_obs, 'external')
    predictions = []
    covid_predictions = {cohort: [] for cohort in ['ren', 'combat', 'stephenson']}
    for fold in range(3):
        train_donors, test_donors = clock.resolve_fold_donors(fold, train_age, test_age)
        folder = args.outdir / f'fold_{fold}'; folder.mkdir(exist_ok=True)
        prepared = shared.prepare_split(fold=fold, mode=mode, min_cells=min_cells,
            train_obs=train_obs, train_counts=train_counts, eval_obs=test_obs,
            eval_counts=test_counts, gene_names=genes, train_donors=train_donors,
            eval_donors=test_donors, eval_label='external_test', artifact_dir=folder)
        if by_type:
            frames = []
            models = {}
            for ct, penalty in selected.items():
                if ct not in prepared['reference_ages']:
                    continue
                model, train = shared.fit_model(prepared, ct, penalty, max_iter, 42)
                frame, metrics = shared.predict_external_type(prepared, ct, model)
                models[ct] = model
                if not frame.empty:
                    frame['model'] = args.model; frames.append(frame)
            prediction = pd.concat(frames, ignore_index=True)
            joblib.dump(models, folder / 'fitted_models.joblib')
        else:
            model = shared.fit_selected_model(prepared, selected, max_iter, 42)
            metacell, donor, metrics = shared.predictions_from_model(model, prepared)
            prediction = donor
            prediction['model'] = args.model
            joblib.dump(model, folder / 'fitted_model.joblib')
            metacell.to_csv(folder / 'external_metacell_predictions.tsv', sep='\t', index=False)
            donor.to_csv(folder / 'external_donor_predictions.tsv', sep='\t', index=False)
        prediction.to_csv(folder / 'external_predictions.tsv', sep='\t', index=False)
        predictions.append(prediction)
        print(f'[refit] {args.model} fold={fold} external_rows={len(prediction)}', flush=True)
        # The COVID clock uses the same source-only fitted
        # transformations, frozen penalties, and fold-specific training set.
        if args.model.startswith('covid_clock') and args.covid_input_dir:
            for cohort in ['ren', 'combat', 'stephenson']:
                path = args.covid_input_dir / f'{cohort}_covid_case_control_aida2000_mean_imputed.h5ad'
                target_obs, target_counts, target_genes = clock.load_obs_counts(path)
                np.testing.assert_array_equal(genes, target_genes)
                target_donors = sorted(target_obs.donor_id.astype(str).unique())
                import anndata as ad
                target_data = ad.read_h5ad(path, backed='r')
                measured = target_data.var.assay_observed.to_numpy(dtype=bool)
                target_data.file.close()
                target = prepare_target_split(fold=fold, mode=mode, min_cells=min_cells,
                    min_eval_cells=20, eval_gene_observed=measured,
                    train_obs=train_obs, train_counts=train_counts, eval_obs=target_obs,
                    eval_counts=target_counts, gene_names=genes, train_donors=train_donors,
                    eval_donors=target_donors, eval_label=cohort)
                np.testing.assert_array_equal(target['retained_genes'], prepared['retained_genes'])
                m, d, _ = shared.predictions_from_model(model, target)
                covid_predictions[cohort].append(d)
                m.to_csv(folder / f'{cohort}_metacell_predictions.tsv', sep='\t', index=False)
                d.to_csv(folder / f'{cohort}_donor_predictions.tsv', sep='\t', index=False)
                del target_counts, target
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions.to_csv(args.outdir / 'external_predictions_all_folds.tsv', sep='\t', index=False)
    group_keys = ['donor_id'] + (['cell_type'] if by_type else [])
    ensemble = all_predictions.groupby(group_keys, observed=True).agg(
        age=('age', 'first'), predicted_age=('pred_age', 'mean'), n_folds=('fold', 'nunique')).reset_index()
    ensemble.to_csv(args.outdir / 'external_predictions_ensemble.tsv', sep='\t', index=False)
    for cohort, frames in covid_predictions.items():
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        cohort_ensemble = combined.groupby('donor_id', observed=True).agg(
            age=('age', 'first'), pred_age=('pred_age', 'mean'), n_folds=('fold', 'nunique')).reset_index()
        target_dir = args.outdir.parent.parent / 'covid' / cohort / 'elasticnet' / mode
        target_dir.mkdir(parents=True, exist_ok=True)
        cohort_ensemble.to_csv(target_dir / 'donor_predictions_ensemble.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()
