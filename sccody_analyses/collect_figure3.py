"""Assemble Figure 3 inputs from public scCody, comparator and clock reports."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from scipy import stats

MODELS = {'flowmap_global': 'scCody-G', 'flowmap_cell_type': 'scCody-CT',
          'scnode': 'scNODE', 'scdiffeq': 'scDiffEq', 'artemis': 'ARTEMIS'}
DYNAMICS = ['scNODE', 'scDiffEq', 'ARTEMIS']
LINEAGES = ['T', 'Myeloid', 'B', 'NK', 'DC']


def bootstrap(frame, rng):
    y, p = frame.true_age.to_numpy(float), frame.predicted_age.to_numpy(float)
    indices = rng.integers(len(y), size=(2000, len(y)))
    yb, pb = y[indices], p[indices]
    errors = np.abs(pb - yb).mean(axis=1)
    yc, pc = yb - yb.mean(axis=1, keepdims=True), pb - pb.mean(axis=1, keepdims=True)
    denominator = np.sqrt((yc ** 2).sum(axis=1) * (pc ** 2).sum(axis=1))
    correlations = np.divide((yc * pc).sum(axis=1), denominator,
                             out=np.full(2000, np.nan), where=denominator > 0)
    return [dict(metric=metric, value=value, ci_low=np.nanquantile(draws, .025),
                 ci_high=np.nanquantile(draws, .975), n_unique_donors=len(y))
            for metric, value, draws in [('MAE', np.abs(p-y).mean(), errors),
                                         ('pearson_r', np.corrcoef(y, p)[0, 1], correlations)]]


def williams(y, a, b):
    r, s, t = np.corrcoef([y, a, b])[np.triu_indices(3, 1)]
    determinant = 1-r*r-s*s-t*t+2*r*s*t
    denominator = 2*(len(y)-1)/(len(y)-3)*determinant+((r+s)/2)**2*(1-t)**3
    statistic = (r-s)*np.sqrt((len(y)-1)*(1+t))/np.sqrt(denominator) if denominator > 0 else np.nan
    return statistic, 2*stats.t.sf(abs(statistic), len(y)-3)


def comparisons(frame, **labels):
    records = []
    for arm in ['scCody-G', 'scCody-CT']:
        candidates = []
        for method in DYNAMICS:
            pair = frame.loc[frame.model.eq(arm)].merge(frame.loc[frame.model.eq(method)],
                on=['donor_id', 'true_age'], suffixes=('_a', '_b'))
            if len(pair) < 4:
                continue
            y, a, b = pair.true_age.to_numpy(), pair.predicted_age_a.to_numpy(), pair.predicted_age_b.to_numpy()
            test, p = stats.ttest_rel(abs(a-y), abs(b-y))
            w, wp = williams(y, a, b)
            for metric, av, bv, statistic, pv, name in [
                ('MAE', abs(a-y).mean(), abs(b-y).mean(), test, p, 'paired two-sided t-test'),
                ('Pearson r', np.corrcoef(y,a)[0,1], np.corrcoef(y,b)[0,1], w, wp, 'Williams two-sided test')]:
                candidates.append(dict(**labels, scody_model=arm, dynamic_model=method, comparison=metric,
                    n=len(pair), scody_value=av, dynamic_value=bv, difference_scody_minus_dynamic=av-bv,
                    statistic=statistic, p_value=pv, test=name))
        for metric in ['MAE', 'Pearson r']:
            eligible = [r for r in candidates if r['comparison'] == metric and np.isfinite(r['dynamic_value'])]
            if eligible:
                records.append(min(eligible, key=lambda r: (r['dynamic_value'] if metric == 'MAE' else -r['dynamic_value'], r['dynamic_model'])))
    return records


def assemble(study='configs/aida.json', outdir='manuscript/figure_3/results'):
    from sccody import ScCodyConfig
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    study = json.loads(Path(study).read_text())
    global_rows, lineage_rows = [], []
    for task in study['tasks']:
        report = Path(task['output']) / 'report'
        config = ScCodyConfig.load(task['config'])
        truth = pd.read_csv(report / 'truth.csv').rename(columns={config.data.time_key: 'true_age'})
        for scope, target in [('global', global_rows), ('cell_type_min5', lineage_rows)]:
            d = pd.read_csv(report / scope / 'predictions.csv')
            d = d.loc[d.status.eq('ok')].rename(columns={'predicted_time': 'predicted_age', config.data.cell_type_key: 'cell_type'})
            d = d.merge(truth[['donor_id', 'true_age']], on='donor_id')
            d['model'], d['fold'] = MODELS[task['arm']], task['fit']
            if scope == 'global':
                target.append(d)
            else:
                # Standard 100-cell lineages; DC uses the filter-first five-cell report.
                d = d.loc[d.cell_type.isin(LINEAGES) & (d.n_cells.ge(100) | d.cell_type.eq('DC'))]
                rare = pd.read_csv(Path('artifacts/rare_cell_types') / task['arm'] / f"fold_{task['fit']}" / 'predictions.csv')
                rare = rare.loc[rare.status.eq('ok') & rare[config.data.cell_type_key].eq('DC')]
                rare = rare.rename(columns={'predicted_time':'predicted_age', config.data.cell_type_key:'cell_type'})
                rare = rare.merge(truth[['donor_id','true_age']], on='donor_id').assign(model=MODELS[task['arm']],fold=task['fit'])
                d = pd.concat([d.loc[~d.cell_type.eq('DC')], rare], ignore_index=True)
                target.extend([d.assign(version=version) for version in ['pca_knn', 'filter_first']])
    for method in ['scnode', 'scdiffeq', 'artemis']:
        for fold in range(3):
            base = Path(f'artifacts/comparators_final/{method}/fold_{fold}')
            d = pd.read_csv(base / f'report/{method}_fit{fold}_{method}_calibration_projection.csv')
            d = d.rename(columns={'chronological_age':'true_age','t_hat_sinkhorn':'predicted_age'}).assign(model=MODELS[method],fold=fold)
            global_rows.append(d.loc[d.projection_unit.eq('donor')])
            lineage_rows.append(d.loc[d.projection_unit.eq('donor_pca_knn_cell_type')].assign(version='pca_knn'))
            dc = pd.read_csv(base / f'report_dc/{method}_fit{fold}_{method}_calibration_projection.csv')
            dc = dc.rename(columns={'chronological_age':'true_age','t_hat_sinkhorn':'predicted_age'}).assign(model=MODELS[method],fold=fold,version='filter_first')
            lineage_rows.append(dc)
    folds = pd.concat(global_rows,ignore_index=True)
    folds = folds.loc[np.isfinite(folds.predicted_age)]
    ensemble = folds.groupby(['model','donor_id','true_age'],as_index=False).agg(predicted_age=('predicted_age','mean'),n_fold_predictions=('fold','nunique'))
    ensemble = ensemble.loc[ensemble.n_fold_predictions.eq(3)]
    ensemble.to_csv(out/'aida_selected_external_ensemble_per_donor.tsv',sep='\t',index=False)
    rng = np.random.default_rng(20260826)
    metrics = [dict(model=model,**row) for model,g in ensemble.groupby('model',sort=False) for row in bootstrap(g,rng)]
    pd.DataFrame(metrics).to_csv(out/'aida_selected_external_ensemble_performance_metrics.tsv',sep='\t',index=False)
    pd.DataFrame(comparisons(ensemble)).to_csv(out/'aida_scCody_vs_best_dynamics_pvalues.tsv',sep='\t',index=False)
    for name in ['lasso_global','elasticnet_global','tage']:
        destination = out/f'clocks/{name}'
        destination.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path('artifacts/clocks')/name/'external_predictions_ensemble.tsv',destination/'external_predictions_ensemble.tsv')
    d = pd.concat(lineage_rows,ignore_index=True)
    d = d.loc[np.isfinite(d.predicted_age) & d.cell_type.isin(LINEAGES)]
    d = d.groupby(['version','model','cell_type','donor_id','true_age'],as_index=False).agg(predicted_age=('predicted_age','mean'),n_folds=('fold','nunique'))
    d = d.loc[d.n_folds.eq(3)]
    for name,label in [('lasso_cell_type','Lasso'),('elasticnet_cell_type','ElasticNet'),('tage','tAge')]:
        if name == 'tage':
            clock = pd.concat([pd.read_csv(Path('artifacts/clocks/tage')/f'fold_{fold}/external_metacell_predictions.tsv',sep='\t') for fold in range(3)])
            clock = clock.groupby(['donor_id','cell_type','age'],as_index=False).agg(predicted_age=('pred_age','mean'),n_folds=('fold','nunique'))
        else:
            clock = pd.read_csv(Path('artifacts/clocks')/name/'external_predictions_ensemble.tsv',sep='\t')
        clock = clock.rename(columns={'age':'true_age'}).assign(model=label)
        clock = clock.loc[clock.n_folds.eq(3) & np.isfinite(clock.predicted_age)]
        d = pd.concat([d,*[clock.assign(version=v) for v in ['pca_knn','filter_first']]],ignore_index=True)
    rng = np.random.default_rng(20261028)
    metrics = [dict(version=v,model=m,cell_type=ct,source='three-fold ensemble',**row)
               for (v,m,ct),g in d.groupby(['version','model','cell_type']) if len(g)>3 for row in bootstrap(g,rng)]
    metrics = pd.DataFrame(metrics)
    tests = []
    for version in ['pca_knn','filter_first']:
        metrics.loc[metrics.version.eq(version)].drop(columns='version').to_csv(out/f'aida_forest_cell_type_plot_{version}_mae_r.tsv',sep='\t',index=False)
        for ct in LINEAGES:
            part = d.loc[d.version.eq(version) & d.cell_type.eq(ct)]
            if version == 'pca_knn' and ct == 'DC':
                part = pd.concat([part.loc[~part.model.isin(DYNAMICS)],d.loc[d.version.eq('filter_first') & d.cell_type.eq(ct) & d.model.isin(DYNAMICS)]])
            tests.extend(comparisons(part,version=version,cell_type=ct))
    pd.DataFrame(tests).to_csv(out/'aida_cell_type_scCody_vs_best_dynamics_pvalues.tsv',sep='\t',index=False)


if __name__ == '__main__':
    assemble()
