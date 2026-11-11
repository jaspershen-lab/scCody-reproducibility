#!/usr/bin/env python3
"""Decode aligned / shared / between-donor / within-donor residual components.

Input is the output of ``extract_residual_fields.py`` for one group.
For source cell z (control) matched to disease cell z_t, with pair alpha and
aging displacement a, residual r = (z_t - z) - alpha a, donor mean residual
r_D (equal pair weight within target donor) and shared shift r_bar (equal
target-donor weight, as in the orthogonal variance decomposition):

    x0 = dec(z)
    x1 = dec(z + alpha a)                      aligned      = x1 - x0
    x2 = dec(z + alpha a + r_bar)              shared       = x2 - x1
    x3 = dec(z + alpha a + r_D)                between      = x3 - x2
    x4 = dec(z_t) = dec(z + alpha a + r)       within       = x4 - x3

The chain telescopes to x4 - x0 exactly, so the decoder nonlinearity is
quantified by path dependence instead: the reverse-order effects

    shared_alt  = dec(z + r_bar) - x0
    aligned_alt = x2 - dec(z + r_bar)

are saved alongside and compared with the chained effects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sccody import ScCody, decompose_displacements, decode_components

COMPONENTS = ["base", "aligned", "shared", "between", "within", "total", "shared_alt", "aligned_alt"]


def field_components(fields, pairs):
    """Apply canonical geometry to recorded pair fields and coefficients."""
    pair_ids = fields["pair_id"]
    lookup = pairs.set_index("pair_id")
    return decompose_displacements(
        fields["z_target"].astype(np.float64) - fields["z_source"].astype(np.float64),
        fields["aging"].astype(np.float64), pair_ids=pair_ids,
        donor_ids=lookup.loc[pair_ids, "target_donor"].astype(str).to_numpy(),
        pair_alphas={str(k): float(v) for k, v in lookup["alpha"].items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Native scCody checkpoint")
    parser.add_argument("--fields", type=Path, required=True, help="<group>_percell_fields.npz")
    parser.add_argument("--pairs", type=Path, required=True, help="<group>_pairs.tsv")
    parser.add_argument("--modules", type=Path, required=True,
                        help="JSON {module: [panel column indices]} for per-cell module effects")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    fields = np.load(args.fields)
    pairs = pd.read_csv(args.pairs, sep="\t")
    modules = {name: np.asarray(index, dtype=np.int64) for name, index in json.loads(args.modules.read_text()).items()}
    model = ScCody.load(args.model, device=args.device)
    no_softplus = model.config.model.no_softplus_decode

    pair_id = fields["pair_id"]
    z_source = fields["z_source"].astype(np.float64)
    z_target = fields["z_target"].astype(np.float64)
    aging = fields["aging"].astype(np.float64)
    terms = field_components(fields, pairs)
    aligned, residual, shared = terms["aligned"], terms["residual"], terms["shared"]
    donors = list(terms["donors"])
    donor_mean = dict(zip(donors, terms["donor_means"]))

    n_genes = None
    donor_sums: dict[str, np.ndarray] = {}
    donor_pairs: dict[str, int] = {d: 0 for d in donors}
    cell_rows: list[pd.DataFrame] = []
    for p, pair in pairs.iterrows():
        mask = pair_id == pair["pair_id"]
        donor = str(pair["target_donor"])
        z, a, r = z_source[mask], aligned[mask], residual[mask]
        r_d = donor_mean[donor]
        effects = decode_components(model, z, {
            "aligned": a, "shared": shared,
            "between": r_d - shared, "within": r - r_d})
        x0 = effects["base"]
        if n_genes is None:
            n_genes = x0.shape[1]
            donor_sums = {d: np.zeros((len(COMPONENTS), n_genes)) for d in donors}
        donor_sums[donor] += np.stack([effects[c].mean(axis=0) for c in COMPONENTS])
        donor_pairs[donor] += 1

        within_latent = r - r_d
        frame = pd.DataFrame({
            "pair_id": int(pair["pair_id"]), "target_donor": donor,
            "reference_donor": str(pair["reference_donor"]),
            "source_row": fields["source_row"][mask], "target_row": fields["target_row"][mask],
            "aligned_norm": np.linalg.norm(a, axis=1),
            "residual_norm": np.linalg.norm(r, axis=1),
            "within_norm": np.linalg.norm(within_latent, axis=1),
            "within_sq": np.sum(within_latent ** 2, axis=1),
            "disease_norm": np.linalg.norm(z_target[mask] - z, axis=1),
        })
        for name, index in modules.items():
            frame[f"dec_within__{name}"] = effects["within"][:, index].mean(axis=1)
            frame[f"dec_total__{name}"] = effects["total"][:, index].mean(axis=1)
        cell_rows.append(frame)
        if (p + 1) % 100 == 0 or p + 1 == len(pairs):
            print(f"[decode] {p + 1}/{len(pairs)} pairs", flush=True)

    donor_effects = np.stack([donor_sums[d] / donor_pairs[d] for d in donors])
    np.savez_compressed(
        args.outdir / "donor_component_effects.npz",
        effects=donor_effects.astype(np.float32), components=np.asarray(COMPONENTS),
        donors=np.asarray(donors), n_pairs=np.asarray([donor_pairs[d] for d in donors]),
        shared_latent=shared, donor_mean_latent=np.stack([donor_mean[d] for d in donors]),
    )
    pd.concat(cell_rows, ignore_index=True).to_csv(args.outdir / "percell_components.tsv.gz", sep="\t", index=False)


if __name__ == "__main__":
    main()
