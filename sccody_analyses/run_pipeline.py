"""Run manifest-defined fits and dependent evaluation stages on a scheduler."""
import argparse
import json
import logging
import os
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch
from anndata.io import read_elem

from sccody import ScCody, ScCodyConfig
from sccody_analyses.selection import localization_score


def load_input(path, columns, *, sample_key=None, sample_ids=None):
    """Read only the model matrix and required metadata, omitting large layers."""
    if sample_ids is not None:
        backed = ad.read_h5ad(path, backed="r")
        try:
            rows = np.flatnonzero(backed.obs[sample_key].astype(str).isin(sample_ids).to_numpy())
            matrix = backed[rows].X
            obs, genes = backed.obs.iloc[rows][columns].copy(), backed.var_names.copy()
        finally:
            backed.file.close()
    else:
        with h5py.File(path, "r") as handle:
            matrix = read_elem(handle["X"])
            obs = read_elem(handle["obs"])[columns].copy()
            genes = read_elem(handle["var"]).index
    return ad.AnnData(matrix, obs=obs, var=pd.DataFrame(index=genes))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--phase", choices=["train", "attach", "report", "summarize"], required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    manifest = json.loads(Path(args.manifest).read_text())
    if args.phase == "summarize":
        summarize(manifest)
        return
    task = manifest["tasks"][args.task]
    config = ScCodyConfig.load(task["config"])
    config.training.device = args.device
    d = config.data
    columns = list(dict.fromkeys([d.sample_key, d.time_key, *d.stratify_keys,
                                 *([d.cell_type_key] if d.cell_type_key else [])]))
    output = Path(task["output"])
    if args.phase == "train":
        data = load_input(manifest["train_input"], columns)
        ScCody(config).fit(data, output / "fit")
    elif args.phase == "attach":
        # Supplied canonical weights carry no healthy references; attach the
        # recorded training donors so report can run without refitting.
        pipeline = ScCody.load(task["canonical_checkpoint"], device=args.device)
        table = pipeline.split_table
        training = load_input(manifest["train_input"], columns, sample_key=d.sample_key,
                              sample_ids=table.loc[table["split"] == "train", d.sample_key].astype(str).tolist())
        pipeline.set_reference(training)
        (output / "fit").mkdir(parents=True, exist_ok=False)
        pipeline.save(output / "fit/model.pt")
        write_json(output / "fit/selection.json", pipeline.selection)
    else:
        pipeline = ScCody.load(output / "fit/model.pt", device=args.device)
        target = load_input(manifest["report_input"], columns, sample_key=d.sample_key,
                            sample_ids=manifest["external_samples"])
        report = output / "report"
        report.mkdir(parents=True, exist_ok=False)
        labels = target.obs[[d.sample_key, d.time_key, *d.stratify_keys]].drop_duplicates(d.sample_key)
        labels.to_csv(report / "truth.csv", index=False)
        global_result = pipeline.localize(target, task["candidate_times"], source_times=task["source_times"],
                                          max_cells=2000, min_cells=100, population_scope="global", balanced_cells=True)
        global_result.save(report / "global")
        write_json(report / "global_metrics.json", localization_score(global_result.predictions, labels, d.sample_key, d.time_key))
        if d.cell_type_key:
            # Report both the standard and small-population eligibility rules.
            lineage_result = pipeline.localize(target, task["candidate_times"], source_times=task["source_times"],
                                               max_cells=2000, min_cells=5, population_scope="cell_type",
                                               balanced_cells=True, pooled_source_cap=True)
            lineage_result.save(report / "cell_type_min5")
            standard = lineage_result.predictions.copy()
            standard.loc[standard.n_cells < 100, "status"] = "too_few_cells"
            standard.loc[standard.n_cells < 100, "predicted_time"] = np.nan
            standard.to_csv(report / "cell_type_min100.csv", index=False)


def summarize(manifest):
    output = Path(manifest["output"]) / "summary"
    output.mkdir(parents=True, exist_ok=False)
    for arm in sorted({t["arm"] for t in manifest["tasks"]}):
        tasks = [t for t in manifest["tasks"] if t["arm"] == arm]
        sample_key = ScCodyConfig.load(tasks[0]["config"]).data.sample_key
        time_key = ScCodyConfig.load(tasks[0]["config"]).data.time_key
        type_key = ScCodyConfig.load(tasks[0]["config"]).data.cell_type_key
        for scope, filename, keys in [("global", "global/predictions.csv", [sample_key]),
                                      ("cell_type_min5", "cell_type_min5/predictions.csv", [sample_key, type_key]),
                                      ("cell_type_min100", "cell_type_min100.csv", [sample_key, type_key])]:
            frames = []
            for task in tasks:
                predictions = pd.read_csv(Path(task["output"]) / "report" / filename)
                predictions = predictions.loc[predictions.status == "ok"].copy()
                predictions["fit"] = task["fit"]
                frames.append(predictions)
            all_predictions = pd.concat(frames, ignore_index=True)
            all_predictions.to_csv(output / f"{arm}_{scope}_fit_predictions.csv", index=False)
            ensemble = all_predictions.groupby(keys, observed=True).agg(
                predicted_time=("predicted_time", "mean"), n_fits=("fit", "nunique")).reset_index()
            ensemble = ensemble.loc[ensemble.n_fits == len(tasks)].copy()
            ensemble["status"] = "ok"
            truth = pd.read_csv(Path(tasks[0]["output"]) / "report/truth.csv")
            ensemble = ensemble.merge(truth, on=sample_key, validate="many_to_one")
            ensemble.to_csv(output / f"{arm}_{scope}_ensemble.csv", index=False)
            if scope == "global":
                score = localization_score(ensemble, truth, sample_key, time_key)
                write_json(output / f"{arm}_global_metrics.json", score)


if __name__ == "__main__":
    main()
