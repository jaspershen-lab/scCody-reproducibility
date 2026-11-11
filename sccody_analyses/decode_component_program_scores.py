#!/usr/bin/env python3
"""Per-cell decoded program scores for every displacement step (CPU).

Recomputes, from ``extract_residual_fields.py`` output, the decoded chain of
``decode_residual_components.py`` for each matched cell (source z -> target z'):

    x0 = dec(z)            x1 = dec(z + alpha a)        x2 = dec(z + alpha a + r_bar)
    x3 = dec(z + alpha a + r_D)                         x4 = dec(z')
    aligned = x1 - x0   shared = x2 - x1   between = x3 - x2   within = x4 - x3   (sum = x4 - x0)

r_D is the donor mean residual and r_bar the mean of r_D over target donors (equal pair / donor
weights), exactly as in decode_residual_components.py.  Per cell and step, the score is the mean
over each module's genes minus the mean over all panel genes (centring of the program heatmaps);
donor means of these scores reproduce the centred donor-mean effects.

Output: <outdir>/component_program_scores.tsv.gz
  pair_id, target_donor, source_row, target_row, <step>__<module> ...

With --labels (one label per row of the target h5ad, e.g. CD14_Mono / CD16_Mono / other, or EN
subclass), also <outdir>/group_gene_effects.npz: gene-level decoded effects per donor x cell group x
step, for size-matched gene-set standardisation.  A group is a target-cell label ("ALL" = every
cell) crossed with a match set ("all" = every cell of that label; "same" = only cells whose matched
source cell has the same label).  Donor values use the pooled weighting: mean over the donor's
cells within each pair, then mean over pairs (pairs without cells of the group are skipped).
  groups, donors, steps, genes-axis effects (groups x donors x steps x genes, float32, NaN = absent),
  n_pairs, n_cells (groups x donors)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sccody import ScCody
from sccody_analyses.decode_residual_components import field_components


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Canonical scCody checkpoint")
    parser.add_argument("--fields-dir", type=Path, required=True)
    parser.add_argument("--group", required=True, help="file prefix, e.g. Myeloid or Myeloid_healthy")
    parser.add_argument("--modules", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="TSV with a 'label' column, one row per target-h5ad row")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    fields = np.load(args.fields_dir / f"{args.group}_percell_fields.npz")
    pairs = pd.read_csv(args.fields_dir / f"{args.group}_pairs.tsv", sep="\t")
    modules = {k: np.asarray(v) for k, v in json.loads(args.modules.read_text()).items()}
    model = ScCody.load(args.model, device="cpu")

    pair_id = fields["pair_id"]
    z = fields["z_source"].astype(np.float64)
    z_target = fields["z_target"].astype(np.float64)
    donor_of_pair = pairs.set_index("pair_id")["target_donor"].astype(str)
    terms = field_components(fields, pairs)
    aligned, residual, shared = terms["aligned"], terms["residual"], terms["shared"]
    cell_donor = terms["donor_ids"]
    r_d = shared + terms["between"]

    steps = ("aligned", "shared", "between", "within", "total")
    scores = {(st, name): np.empty(len(z), dtype=np.float32) for st in steps for name in modules}
    dec = lambda x: model.decode_latent(x.astype(np.float32)).astype(np.float64)

    grouped = args.labels is not None
    if grouped:
        row_label = pd.read_csv(args.labels, sep="\t")["label"].astype(str).to_numpy()
        tgt_label, src_label = row_label[fields["target_row"]], row_label[fields["source_row"]]
        pair_acc: dict = {}    # (pair, label, match set) -> [sum (steps x genes), n cells]
        donor_acc: dict = {}   # (donor, label, match set) -> [sum of pair means, n pairs, n cells]
        last_cell = pd.Series(np.arange(len(z))).groupby(np.sort(pair_id)).max()
    order = np.argsort(pair_id, kind="stable")   # pairs are contiguous, so each pair can be closed out
    for i in range(0, len(z), args.batch_size):
        idx = order[i:i + args.batch_size]
        base = z[idx] + aligned[idx]
        x = [dec(z[idx]), dec(base), dec(base + shared), dec(base + r_d[idx]), dec(z_target[idx])]
        effects = {"aligned": x[1] - x[0], "shared": x[2] - x[1], "between": x[3] - x[2],
                   "within": x[4] - x[3], "total": x[4] - x[0]}
        for st, step in effects.items():
            centre = step.mean(axis=1)
            for name, index in modules.items():
                scores[(st, name)][idx] = step[:, index].mean(axis=1) - centre
        if not grouped:
            continue
        stack = np.stack([effects[st] for st in steps], axis=1)   # cells x steps x genes
        bp, bt, bs = pair_id[idx], tgt_label[idx], src_label[idx]
        for p in np.unique(bp):
            rows = bp == p
            for label in ["ALL", *np.unique(bt[rows])]:
                lab = rows if label == "ALL" else rows & (bt == label)
                for match, sel in (("all", lab), ("same", lab & (bt == bs))):
                    n = int(sel.sum())
                    if n:
                        acc = pair_acc.setdefault((p, label, match), [0.0, 0])
                        acc[0] = acc[0] + stack[sel].sum(axis=0); acc[1] += n
            if last_cell.loc[p] < i + args.batch_size:   # pair complete: add its group means to the donor
                for key in [k for k in pair_acc if k[0] == p]:
                    total, n = pair_acc.pop(key)
                    acc = donor_acc.setdefault((donor_of_pair.loc[p], key[1], key[2]), [0.0, 0, 0])
                    acc[0] = acc[0] + total / n; acc[1] += 1; acc[2] += n
    if grouped:
        assert not pair_acc, "unclosed pairs"
        donors = sorted({k[0] for k in donor_acc})
        groups = sorted({f"{k[1]}|{k[2]}" for k in donor_acc})
        out = np.full((len(groups), len(donors), len(steps), x[0].shape[1]), np.nan, dtype=np.float32)
        n_pairs = np.zeros((len(groups), len(donors)), dtype=np.int32); n_cells = np.zeros_like(n_pairs)
        gi, di = {g: j for j, g in enumerate(groups)}, {d: j for j, d in enumerate(donors)}
        for (d, label, match), (total, npair, ncell) in donor_acc.items():
            j, k = gi[f"{label}|{match}"], di[d]
            out[j, k] = total / npair; n_pairs[j, k] = npair; n_cells[j, k] = ncell
        np.savez_compressed(args.outdir / "group_gene_effects.npz", groups=np.asarray(groups), donors=np.asarray(donors),
                            steps=np.asarray(steps), effects=out, n_pairs=n_pairs, n_cells=n_cells)
    frame = pd.DataFrame({"pair_id": pair_id, "target_donor": cell_donor,
                          "source_row": fields["source_row"], "target_row": fields["target_row"]})
    for (st, name), values in scores.items():
        frame[f"{st}__{name}"] = values
    frame.to_csv(args.outdir / "component_program_scores.tsv.gz", sep="\t", index=False)
    print(f"[done] {args.group}: {len(z)} cells, {len(modules)} modules", flush=True)


if __name__ == "__main__":
    main()
