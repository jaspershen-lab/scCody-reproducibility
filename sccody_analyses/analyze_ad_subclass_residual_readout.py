#!/usr/bin/env python3
"""Per-subclass readout of the aging-aligned and shared residual components (PsychAD AD, excitatory neurons).

Uses ``extract_residual_fields.py`` output for EN disease pairs and the
healthy (+10y, holdout-target) reference.  Subclass labels are attached to the
matched cells after OT; they never constrain matching.  Cells are grouped by
the subclass of the control (source) cell, where the displacement starts.

Per cell (latent):
  aligned_proj   alpha_pair * ||a_i||                (signed aligned displacement)
  shared_proj    r_i . u,  u = r_bar / ||r_bar||     (u from the disease fit; the
                                                      healthy run is projected on the
                                                      same u as its reference)
  within_sq      ||r_i - r_D||^2

Per subclass: donor-weighted means with donor-bootstrap CIs, disease and
healthy, plus the rate at which source cells are matched to a different
subclass.  The decoded shared step dec(z + alpha a + r_bar) - dec(z + alpha a)
is averaged per subclass (equal donor weight) and compared with the overall
shared signature.

Outputs
  subclass_latent_readout.tsv
  subclass_shared_signature_similarity.tsv
  subclass_shared_top_genes.tsv
  subclass_shared_signatures.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from sccody import ScCody
from sccody_analyses.decode_residual_components import field_components


def load_fields(fields_dir: Path, group: str) -> tuple[dict, pd.DataFrame]:
    data = dict(np.load(fields_dir / f"{group}_percell_fields.npz"))
    pairs = pd.read_csv(fields_dir / f"{group}_pairs.tsv", sep="\t")
    return data, pairs


def residual_terms(data: dict, pairs: pd.DataFrame) -> dict:
    terms = field_components(data, pairs)
    alpha = pairs.set_index("pair_id").loc[data["pair_id"], "alpha"].to_numpy(float)
    return {**terms, "alpha": alpha, "cell_donor": terms["donor_ids"],
            "within_sq": np.sum(terms["within"] ** 2, axis=1),
            "aligned_proj": alpha * np.linalg.norm(data["aging"].astype(np.float64), axis=1)}


def donor_bootstrap(frame: pd.DataFrame, column: str, rng: np.random.Generator, n: int) -> tuple[float, float, float]:
    per_donor = frame.groupby("donor")[column].mean().to_numpy()
    boot = [per_donor[rng.integers(0, len(per_donor), len(per_donor))].mean() for _ in range(n)]
    return float(per_donor.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Canonical scCody checkpoint")
    parser.add_argument("--fields-dir", type=Path, required=True)
    parser.add_argument("--target-h5ad", type=Path, default=Path("data/psychad/ad_case_control.h5ad"))
    parser.add_argument("--group", default="EN")
    parser.add_argument("--label-key", default="subclass")
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    dis_data, dis_pairs = load_fields(args.fields_dir, args.group)
    hea_data, hea_pairs = load_fields(args.fields_dir, f"{args.group}_healthy")
    obj = ad.read_h5ad(args.target_h5ad, backed="r")
    labels = obj.obs[args.label_key].astype(str).to_numpy()
    obj.file.close()

    dis = residual_terms(dis_data, dis_pairs)
    hea = residual_terms(hea_data, hea_pairs)
    u = dis["shared"] / np.linalg.norm(dis["shared"])

    def cell_frame(data: dict, terms: dict) -> pd.DataFrame:
        src = labels[data["source_row"]]
        tgt = labels[data["target_row"]]
        return pd.DataFrame({
            "donor": terms["cell_donor"], "label": src, "cross_label": src != tgt,
            "aligned_proj": terms["aligned_proj"], "shared_proj": terms["residual"] @ u,
            "within_sq": terms["within_sq"],
        })

    frames = {"disease": cell_frame(dis_data, dis), "healthy": cell_frame(hea_data, hea)}

    rows = []
    for label in sorted(set(frames["disease"]["label"]) | set(frames["healthy"]["label"])):
        row = {"label": label}
        for mode, frame in frames.items():
            sub = frame[frame["label"] == label]
            counts = sub.groupby("donor").size()
            sub = sub[sub["donor"].isin(counts.index[counts >= args.min_cells])]
            row[f"{mode}_n_donors"] = int(sub["donor"].nunique())
            row[f"{mode}_cell_share"] = float(len(sub) / max(len(frame), 1))
            if sub.empty:
                continue
            for column in ("aligned_proj", "shared_proj", "within_sq", "cross_label"):
                mean, low, high = donor_bootstrap(sub.assign(cross_label=sub["cross_label"].astype(float)), column, rng, args.bootstrap)
                row[f"{mode}_{column}"], row[f"{mode}_{column}_ci_low"], row[f"{mode}_{column}_ci_high"] = mean, low, high
        rows.append(row)
    readout = pd.DataFrame(rows)
    for column in ("aligned_proj", "shared_proj", "within_sq"):
        readout[f"excess_{column}"] = readout.get(f"disease_{column}") - readout.get(f"healthy_{column}")
    readout.to_csv(args.outdir / "subclass_latent_readout.tsv", sep="\t", index=False)
    print(readout[["label", "disease_n_donors", "disease_aligned_proj", "disease_shared_proj",
                   "healthy_shared_proj", "disease_cross_label"]].round(3).to_string(index=False), flush=True)

    # Subclass contribution to the population shared shift, with a donor bootstrap.
    # contribution_l = AD cell share_l x donor-mean shared projection_l (donors with >=
    # min_cells cells of l), as % of the sum over subclasses kept for plotting
    # (>= 5 AD and >= 5 healthy donors).  Resampling AD donors with replacement updates
    # both the cell shares and the mean projections.
    kept = readout.loc[(readout["disease_n_donors"] >= 5) & (readout["healthy_n_donors"] >= 5), "label"].tolist()
    dis_frame = frames["disease"]
    donors = np.asarray(sorted(dis_frame["donor"].unique()))
    total = dis_frame.groupby("donor").size().reindex(donors).to_numpy(float)
    stats = dis_frame.groupby(["donor", "label"])["shared_proj"].agg(["size", "mean"])
    n = stats["size"].unstack().reindex(index=donors, columns=kept).fillna(0).to_numpy(float)
    m = stats["mean"].unstack().reindex(index=donors, columns=kept).fillna(0).to_numpy(float)
    ok = n >= args.min_cells

    def contribution_percent(weight: np.ndarray) -> np.ndarray:
        share = (weight[:, None] * n * ok).sum(axis=0) / (weight * total).sum()
        proj = (weight[:, None] * m * ok).sum(axis=0) / np.maximum((weight[:, None] * ok).sum(axis=0), 1e-12)
        contrib = share * proj
        return 100 * contrib / contrib.sum()

    point = contribution_percent(np.ones(len(donors)))
    boot = np.stack([contribution_percent(np.bincount(rng.integers(0, len(donors), len(donors)),
                                                      minlength=len(donors)).astype(float))
                     for _ in range(args.bootstrap)])
    contribution = pd.DataFrame({
        "label": kept, "percent": point,
        "ci_low": np.percentile(boot, 2.5, axis=0), "ci_high": np.percentile(boot, 97.5, axis=0),
        "n_ad_donors": ok.sum(axis=0), "bootstrap": args.bootstrap,
    })
    contribution.to_csv(args.outdir / "subclass_shared_contribution.tsv", sep="\t", index=False)
    print(contribution.round(2).to_string(index=False), flush=True)

    # Decoded shared step per source subclass (disease fit).
    model = ScCody.load(args.model, device=args.device)
    start = dis_data["z_source"].astype(np.float64) + dis["aligned"]
    n_cells = len(start)
    effect = np.empty((n_cells, len(model.genes)), dtype=np.float32)
    for i in range(0, n_cells, args.batch_size):
        z1 = start[i:i + args.batch_size]
        x1 = model.decode_latent(z1.astype(np.float32))
        x2 = model.decode_latent((z1 + dis["shared"]).astype(np.float32))
        effect[i:i + args.batch_size] = x2 - x1
    frame = frames["disease"]
    symbols = pd.read_csv(args.panel, sep="\t")["symbol"].astype(str).to_numpy()

    def donor_signature(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        donors = frame["donor"].to_numpy()[mask]
        values = effect[mask]
        per_donor = []
        for d in np.unique(donors):
            m = donors == d
            if m.sum() >= args.min_cells:
                per_donor.append(values[m].mean(axis=0))
        return np.mean(per_donor, axis=0) if per_donor else None, np.asarray(per_donor)

    overall, _ = donor_signature(np.ones(n_cells, dtype=bool))
    sims, tops, sigs = [], [], {"overall": overall}
    for label in sorted(frame["label"].unique()):
        sig, per_donor = donor_signature(frame["label"].to_numpy() == label)
        if sig is None or len(per_donor) < 5:
            continue
        sigs[label] = sig
        boot = [np.corrcoef(per_donor[rng.integers(0, len(per_donor), len(per_donor))].mean(axis=0), overall)[0, 1]
                for _ in range(args.bootstrap)]
        sims.append({"label": label, "n_donors": len(per_donor),
                     "pearson_vs_overall": float(np.corrcoef(sig, overall)[0, 1]),
                     "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)),
                     "signature_rms": float(np.sqrt(np.mean(sig ** 2)))})
        order = np.argsort(sig)
        tops.append({"label": label, "top_up": ",".join(symbols[order[::-1][:20]]), "top_down": ",".join(symbols[order[:20]])})
    labels_kept = [k for k in sigs if k != "overall"]
    if labels_kept:
        matrix = np.corrcoef(np.stack([sigs[k] for k in labels_kept]))
        pd.DataFrame(matrix, index=labels_kept, columns=labels_kept).to_csv(args.outdir / "subclass_shared_signature_corr_matrix.tsv", sep="\t")
    pd.DataFrame(sims).to_csv(args.outdir / "subclass_shared_signature_similarity.tsv", sep="\t", index=False)
    pd.DataFrame(tops).to_csv(args.outdir / "subclass_shared_top_genes.tsv", sep="\t", index=False)
    np.savez_compressed(args.outdir / "subclass_shared_signatures.npz",
                        labels=np.asarray(list(sigs)), signatures=np.stack(list(sigs.values())).astype(np.float32),
                        symbols=symbols)
    print(pd.DataFrame(sims).round(3).to_string(index=False))
    print("[done]")


if __name__ == "__main__":
    main()
