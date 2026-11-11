#!/usr/bin/env python3
"""Per-cell decoded total program change for an arbitrary module set (CPU).

Recomputes, from ``extract_residual_fields.py`` output, the total step of
``decode_residual_components.py`` (matched source cell -> target cell)

    total_i = dec(z'_i) - dec(z_i)

and stores, per cell, its mean over each module's genes minus its mean over all panel genes
(the centring of the program heatmaps; linear, so donor means of
these per-cell scores equal the centred score of the donor-mean effect).  The decoder is small,
so this runs on CPU.

Output: <outdir>/total_program_scores.tsv.gz
  pair_id, target_donor, source_row, target_row, total__<module> ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sccody import ScCody, decompose_displacements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Canonical scCody checkpoint")
    parser.add_argument("--fields-dir", type=Path, required=True)
    parser.add_argument("--group", required=True, help="file prefix, e.g. Myeloid or Myeloid_healthy")
    parser.add_argument("--modules", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=65536)
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
    cell_donor = donor_of_pair.loc[pair_id].to_numpy()

    scores = {name: np.empty(len(z), dtype=np.float32) for name in modules}
    for i in range(0, len(z), args.batch_size):
        x0 = model.decode_latent(z[i:i + args.batch_size].astype(np.float32))
        x4 = model.decode_latent(z_target[i:i + args.batch_size].astype(np.float32))
        step = (x4 - x0).astype(np.float64)
        centre = step.mean(axis=1)
        for name, index in modules.items():
            scores[name][i:i + args.batch_size] = step[:, index].mean(axis=1) - centre
    frame = pd.DataFrame({"pair_id": pair_id, "target_donor": cell_donor,
                          "source_row": fields["source_row"], "target_row": fields["target_row"]})
    for name, values in scores.items():
        frame[f"total__{name}"] = values
    frame.to_csv(args.outdir / "total_program_scores.tsv.gz", sep="\t", index=False)
    print(f"[done] {args.group}: {len(z)} cells, {len(modules)} modules", flush=True)


if __name__ == "__main__":
    main()
