"""Annual projected pathway scores for held-out age-31 source donors.

``export`` transports the recorded source cells (configs/annual_programs/) to every
integer age with one AIDA checkpoint; ``score`` computes program scores
against the frozen expression-matched control genes for both the projected and
the observed AIDA populations.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import anndata as ad
from sccody import ScCody
BASE = None

def sources(fold):
    paths=[BASE/f'all_age_flowmap_projection/fold_{fold}.npz', BASE/f'all_age_flowmap_projection_platelet_min30/fold_{fold}.npz']
    exports=[]
    for path in paths:
        with np.load(path) as a: exports.append({k:a[k] for k in a.files})
    assert np.array_equal(exports[0]['gene_ids'],exports[1]['gene_ids'])
    merged={k:np.concatenate([a[k] for a in exports]) for k in ['donor_id','cell_type','sex','institute','n_source_cells','source_indices']}
    merged.update(gene_ids=exports[0]['gene_ids'],gene_symbols=exports[0]['gene_symbols'],ages=exports[0]['ages'])
    return merged

def generate(args):
    model = ScCody.load(args.model, device=args.device)
    if model.config.model.scope != args.scope:
        raise ValueError("--scope differs from checkpoint")
    if model.config.data.input_space != "log1p" or model.config.data.layer is not None:
        raise ValueError("Annual scoring expects the prepared AIDA log1p X input")
    src = sources(args.fold)
    groups = np.repeat(np.arange(len(src["donor_id"])), src["n_source_cells"])
    data = ad.read_h5ad(args.input, backed="r")
    try:
        obs = data.obs.iloc[src["source_indices"]].copy()
        matrix = data[src["source_indices"]].X
        genes = data.var_names.copy()
        symbols = data.var["feature_name"].to_numpy(dtype=str)
    finally:
        data.file.close()
    if list(genes) != model.genes:
        raise ValueError("Annual source gene order must match the checkpoint")
    train = set(model.split_table.loc[model.split_table.split == "train", model.config.data.sample_key].astype(str))
    assert not train.intersection(obs.donor_id.astype(str))
    population = ad.AnnData(matrix, obs=obs, var=pd.DataFrame(index=genes))
    latent = model.encode(population)
    codes = np.array([model.cell_types.index(str(c)) for c in obs.Annotation_Level1]) if model.config.model.scope == "cell_type" else None
    counts = src["n_source_cells"][:, None]
    expression = []
    for age in src["ages"]:
        z = model.transport_latent(latent, 31., float(age), cell_type_codes=codes)
        sums = np.zeros((len(counts), len(genes)), dtype=np.float64)
        # Aggregate the same heldout source cells at every candidate age.
        for start in range(0, len(z), args.batch_size):
            decoded = model.decode_latent(z[start:start + args.batch_size])
            np.add.at(sums, groups[start:start + args.batch_size], decoded)
        expression.append((sums / counts).astype(np.float32))
    observed = np.zeros((len(counts), len(genes)), dtype=np.float64)
    for start in range(0, len(groups), args.batch_size):
        values = matrix[start:start + args.batch_size]
        values = values.toarray() if hasattr(values, "toarray") else values
        np.add.at(observed, groups[start:start + args.batch_size], values)
    src.update(gene_ids=np.asarray(genes, dtype=str), gene_symbols=symbols)
    out = args.outdir / "annual_programs" / ("flowmap_" + model.config.model.scope)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"fold_{args.fold}.npz", expression=np.stack(expression),
                        observed_source_expression=(observed / counts).astype(np.float32), **src)

def score(args):
    from scipy import sparse
    template=BASE/'all_age_analysis_platelet_min30'
    coverage=pd.read_csv(template/'program_gene_coverage.tsv',sep='\t')
    controls=pd.read_csv(template/'fixed_scanpy_control_genes.tsv',sep='\t')
    recorded=pd.read_csv(template/'heldout_projection_program_scores.tsv.gz',sep='\t')
    # Take only the observed donor/lineage groups from the recorded table; scores are recomputed.
    observed=recorded.loc[recorded.dataset.eq('aida_observed') & ~recorded.cell_type.eq('equal_lineage_global')].copy()
    data=ad.read_h5ad(args.input,backed='r')
    symbols=data.var.feature_name.astype(str).to_numpy()
    lookup={str(s):i for i,s in enumerate(symbols)}
    keys=pd.MultiIndex.from_frame(observed[['donor_id','cell_type']].drop_duplicates())
    obs_keys=pd.MultiIndex.from_arrays([data.obs.donor_id.astype(str),data.obs.Annotation_Level1.astype(str)])
    codes=keys.get_indexer(obs_keys)
    sums=np.zeros((len(keys),data.n_vars),dtype=np.float64)
    counts=np.zeros(len(keys),dtype=np.int64)
    for start in range(0,data.n_obs,8192):
        stop=min(start+8192,data.n_obs);group=codes[start:stop];use=group>=0
        matrix=data.X[start:stop][use]
        assignment=sparse.csr_matrix((np.ones(use.sum()),(group[use],np.arange(use.sum()))),shape=(len(keys),use.sum()))
        sums+=(assignment@matrix).toarray()
        counts+=np.bincount(group[use],minlength=len(keys))
    data.file.close()
    means=sums/counts[:,None]
    frozen={}
    for item in controls.itertuples(index=False):
        original_genes=coverage.loc[coverage.program==item.program,'panel_genes'].iloc[0].split(',')
        original_control=item.control_genes.split(',')
        genes=[g for g in original_genes if g in lookup];control=[g for g in original_control if g in lookup]
        if not genes or not control:
            raise RuntimeError(f'No development-panel support for {item.cell_type}/{item.program}')
        frozen[(str(item.cell_type),str(item.program))]=([lookup[g] for g in genes],[lookup[g] for g in control])
    for (ct,program),indices in observed.groupby(['cell_type','program'],observed=True).groups.items():
        group=keys.get_indexer(pd.MultiIndex.from_frame(observed.loc[indices,['donor_id','cell_type']]))
        observed.loc[indices,'n_cells']=counts[group]
        gi,ci=frozen[(str(ct),str(program))]
        observed.loc[indices,'score']=means[group][:,gi].mean(axis=1)-means[group][:,ci].mean(axis=1)
    primary=observed.loc[~observed.cell_type.eq('Platelet')].copy()
    scales=primary.loc[primary.age.between(30,60)].groupby(['cell_type','program'],observed=True).score.std(ddof=0).replace(0,1)
    def with_global(frame):
        work=frame.loc[~frame.cell_type.eq('Platelet')].copy()
        work['score_z']=[row.score/scales.loc[(row.cell_type,row.program)] for row in work.itertuples(index=False)]
        grouped=work.groupby(['dataset','fold','donor_id','age','sex','institute','program'],dropna=False,observed=True,as_index=False).agg(score=('score_z','mean'),n_lineages=('cell_type','nunique'))
        grouped=grouped.loc[grouped.n_lineages>=4].copy();grouped['cell_type']='equal_lineage_global'
        return pd.concat([frame,grouped],ignore_index=True)
    observed=with_global(observed)
    rows=[]
    for fold in range(3):
        path=args.outdir/'annual_programs'/f'flowmap_{args.scope}'/f'fold_{fold}.npz'
        with np.load(path) as archive:
            a={k:archive[k] for k in archive.files}
        np.testing.assert_array_equal(a['gene_symbols'],symbols)
        for group,ct in enumerate(a['cell_type']):
            for item in controls.loc[controls.cell_type.eq(ct)].itertuples(index=False):
                gi,ci=frozen[(str(ct),str(item.program))]
                expr=a['expression'][:,group,:]
                values=expr[:,gi].mean(axis=1)-expr[:,ci].mean(axis=1)
                for age,value in zip(a['ages'],values):
                    rows.append(dict(dataset='flowmap_projected',fold=fold,donor_id=str(a['donor_id'][group]),cell_type=str(ct),age=float(age),sex=str(a['sex'][group]),institute=str(a['institute'][group]),n_cells=int(a['n_source_cells'][group]),program=item.program,score=float(value)))
    dest=args.outdir/'annual_programs'/f'flowmap_{args.scope}'/'analysis';dest.mkdir(parents=True,exist_ok=True)
    pd.concat([observed,with_global(pd.DataFrame(rows))],ignore_index=True).to_csv(dest/'heldout_projection_program_scores.tsv.gz',sep='\t',index=False)

def main():
    global BASE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["export", "score"])
    p.add_argument("--base", type=Path, default=Path(__file__).resolve().parents[1]/"configs/annual_programs", help="Recorded source exports and frozen program controls")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--scope", choices=["global", "cell_type"], required=True)
    p.add_argument("--fold", type=int, choices=[0,1,2], default=0)
    p.add_argument("--model", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=1024)
    args = p.parse_args()
    BASE = args.base
    if args.command == "export":
        if args.model is None: p.error("export requires --model")
        generate(args)
    else: score(args)

if __name__ == "__main__":
    main()
