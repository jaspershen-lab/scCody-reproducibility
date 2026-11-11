"""Evaluate rare lineages with public canonical checkpoints and healthy references."""
from pathlib import Path
import argparse
import json
import pandas as pd
from sccody import ScCody, ScCodyConfig
from sccody_analyses.run_pipeline import load_input
from sccody_analyses.paths import ROOT, study_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=ROOT/"configs/aida.json")
    parser.add_argument("--outdir",type=Path,default=ROOT/"artifacts/rare_cell_types")
    parser.add_argument("--task",type=int,default=None)
    parser.add_argument("--device",default="auto")
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    tasks=manifest["tasks"] if args.task is None else [manifest["tasks"][args.task]]
    predictions=[]
    for task in tasks:
        config=ScCodyConfig.load(study_path(task["config"]))
        d=config.data
        columns=list(dict.fromkeys([d.sample_key,d.time_key,d.cell_type_key,*d.stratify_keys]))
        model=ScCody.load(study_path(task["canonical_checkpoint"]),device=args.device)
        table=model.split_table
        training=load_input(study_path(manifest["train_input"]),columns,sample_key=d.sample_key,
                            sample_ids=table.loc[table.split=="train",d.sample_key].astype(str).tolist())
        model.set_reference(training)
        target=load_input(study_path(manifest["report_input"]),columns,sample_key=d.sample_key,
                          sample_ids=manifest["external_samples"])
        target=target[target.obs[d.cell_type_key].astype(str).isin(["DC","Platelet"])].copy()
        result=model.localize(target,task["candidate_times"],source_times=task["source_times"],
                              max_cells=2000,min_cells=5,population_scope="cell_type",balanced_cells=True)
        destination=args.outdir/task["arm"]/f"fold_{task['fit']}"
        result.save(destination)
        predictions.append(result.predictions.assign(model=task["arm"],fold=task["fit"]))
    pd.concat(predictions,ignore_index=True).to_csv(args.outdir/"all_predictions.tsv",sep="\t",index=False)


if __name__=="__main__":
    main()
