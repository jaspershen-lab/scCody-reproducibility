"""Collect Figure 2 plotting tables after population evaluation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .metrics.population import edge_reference_metrics

MODELS = {'scCody-global':'scCody-G','scCody-cell_type':'scCody-CT',
          'scnode':'scNODE','scdiffeq':'scDiffEq','artemis':'ARTEMIS'}
CONTROL_FIELDS = {'de_vs_reference_logfc_pearson_all':'pearson_r',
    'de_vs_reference_logfc_spearman_all':'spearman_r',
    'de_vs_reference_top100_overlap_frac':'top100_overlap_frac',
    'de_vs_reference_top200_overlap_frac':'top200_overlap_frac',
    'de_direction_agreement_all':'direction_agreement_all',
    'de_direction_agreement_nonzero':'direction_agreement_nonzero'}


def collect(settings='configs/population_evaluation.json', outdir='manuscript/figure_2/results'):
    config = json.loads(Path(settings).read_text())
    root, out = Path(config['output']), Path(outdir)
    out.mkdir(parents=True,exist_ok=True)
    means, records = [], []
    for task in config['evaluation_tasks']:
        model = MODELS[task['model']]
        folder = root/'metrics'/task['name']
        d = pd.read_csv(folder/'distribution.csv')
        d = d.loc[d.status.eq('ok') & d.scope.isin(['all','B','Myeloid','NK','T'])].copy()
        means.append(pd.DataFrame({'scope':np.where(d.scope.eq('all'),'Global','Cell type'),
            'cell_type':d.scope.replace({'all':'All'}),'model':model,'fold':task['fit'],
            'target_age':d.target_age,'mean_expression_corr':d.mean_expression_corr}))
        contrasts = pd.read_csv(folder/'differential_expression.csv')
        for row in contrasts.loc[contrasts.scope.isin(['all','B','Myeloid','NK','T']) & contrasts.target_age.isin([21,31,41,51,61])].itertuples():
            genes = pd.read_csv(folder/f'de_genes_{row.scope}_{int(row.target_age)}_{row.side}.csv.gz')
            values = edge_reference_metrics(genes.generated_expression_log2fc,genes.edgeR_logFC)
            common = dict(analysis='model_performance',method={'scCody-G':'FlowMap-global','scCody-CT':'FlowMap-cell-type'}.get(model,model),
                reference='sc_ref_whole_dataset',scope=row.scope,cell_type=row.scope,target_age=row.target_age,
                side=row.side,fold=task['fit'],statistic='value')
            records.extend(dict(**common,metric=k,value=v) for k,v in values.items())
    means = pd.concat(means,ignore_index=True)
    means.to_csv(out/'mean_expression_correlation_long.csv',index=False)
    means.loc[means.scope.eq('Global')].to_csv(out/'mean_expression_correlation_global.tsv',sep='\t',index=False)
    controls = pd.read_csv(root/'controls/half_split_repeats.csv')
    fields = [c for c in controls if c.startswith(('de_','sign_agreement_'))]
    for (scope,age,side),group in controls.groupby(['scope','target_age','side']):
        common = dict(analysis='real_half_split_stability',method='real_half_split',reference='sc_ref_whole_dataset',
            scope=scope,cell_type=scope,target_age=age,side=side,fold=-1,statistic='mean')
        records.extend(dict(**common,metric=CONTROL_FIELDS.get(k,k),value=group[k].mean()) for k in fields)
    pd.DataFrame(records).to_csv(out/'whole_ref_model_and_half_split_metrics.tsv.gz',sep='\t',index=False)
    # Display 500 balanced draws from the available unique donor partitions.
    draws = []
    rng = np.random.default_rng(42)
    for (age,side),group in controls.loc[controls.scope.eq('all')].groupby(['target_age','side']):
        indices = np.tile(rng.permutation(len(group)),int(np.ceil(500/len(group))))[:500]
        d = group.iloc[indices].rename(columns=CONTROL_FIELDS).copy()
        d['split'],d['reference'] = np.arange(500),'sc_ref_whole_dataset'
        draws.append(d)
    pd.concat(draws,ignore_index=True).to_csv(out/'real_de_split_half_metrics_global.tsv.gz',sep='\t',index=False)


if __name__ == '__main__':
    collect()
