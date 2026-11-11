#!/usr/bin/env python3
"""Compute normalized donor-pseudobulk DE recovery from saved predictions.

The generated side is aggregated by the source donor labels saved in each
prediction artifact.  The real side is aggregated by donor labels read from
the source H5AD, so this never treats generated cells as independent
replicates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sccody_analyses.metrics.edger import _compare_vectors


def rows_from_h5ad(adata, indices: np.ndarray) -> np.ndarray:
    values = adata.X[np.asarray(indices, dtype=np.int64)]
    if sp.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=np.float64)


def reference_indices_from_artifact(
    values: np.lib.npyio.NpzFile,
    obs_names: np.ndarray,
) -> np.ndarray:
    """Resolve the fixed normal-training cells recorded in a prediction NPZ."""
    if "train_indices" in values:
        indices = np.asarray(values["train_indices"], dtype=np.int64)
    elif "train_obs_names" in values:
        lookup = {name: index for index, name in enumerate(obs_names)}
        names = np.asarray(values["train_obs_names"], dtype=str)
        missing = sorted(set(names) - set(lookup))
        if missing:
            raise ValueError(
                "Prediction artifact contains unknown train obs names: "
                f"{missing[:3]}"
            )
        indices = np.asarray([lookup[name] for name in names], dtype=np.int64)
    else:
        raise ValueError(
            "Prediction artifact has neither train_indices nor train_obs_names; "
            "cannot reconstruct its fixed reference donor pool"
        )
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Prediction artifact contains duplicate fixed-reference indices")
    if np.any(indices < 0) or np.any(indices >= len(obs_names)):
        raise ValueError("Prediction artifact contains out-of-range train reference indices")
    return indices


def build_fixed_reference_summary(
    adata,
    indices: np.ndarray,
    ages_all: np.ndarray,
    donors_all: np.ndarray,
    cell_types_all: np.ndarray,
    sexes_all: np.ndarray,
    *,
    chunk_size: int,
) -> dict:
    """Aggregate every fixed-reference cell into donor-level normalized means.

    The source H5AD is opened backed, so this reads the full fixed reference
    donor pool in bounded chunks instead of materializing the full matrix.
    Donor ages are required to be invariant, as they are for this dataset.
    """
    donor_sums: dict[str, np.ndarray] = {}
    donor_counts: dict[str, int] = {}
    donor_ct_sums: dict[tuple[str, str], np.ndarray] = {}
    donor_ct_counts: dict[tuple[str, str], int] = {}
    donor_ages: dict[str, set[float]] = {}
    donor_sexes: dict[str, set[str]] = {}

    for start in range(0, len(indices), int(chunk_size)):
        chunk_indices = indices[start : start + int(chunk_size)]
        values = rows_from_h5ad(adata, chunk_indices)
        chunk_ages = np.asarray(ages_all[chunk_indices], dtype=float)
        chunk_donors = np.asarray(donors_all[chunk_indices], dtype=str)
        chunk_cell_types = np.asarray(cell_types_all[chunk_indices], dtype=str)
        chunk_sexes = np.asarray(sexes_all[chunk_indices], dtype=str)
        for donor in pd.unique(chunk_donors):
            mask = chunk_donors == donor
            donor_sums[donor] = donor_sums.get(donor, 0.0) + values[mask].sum(axis=0)
            donor_counts[donor] = donor_counts.get(donor, 0) + int(mask.sum())
            donor_ages.setdefault(donor, set()).update(chunk_ages[mask].tolist())
            donor_sexes.setdefault(donor, set()).update(chunk_sexes[mask].tolist())
        group_labels = np.asarray(
            [f"{donor}\x1f{cell_type}" for donor, cell_type in zip(chunk_donors, chunk_cell_types)],
            dtype=str,
        )
        for group in pd.unique(group_labels):
            mask = group_labels == group
            donor, cell_type = group.split("\x1f", 1)
            key = (donor, cell_type)
            donor_ct_sums[key] = donor_ct_sums.get(key, 0.0) + values[mask].sum(axis=0)
            donor_ct_counts[key] = donor_ct_counts.get(key, 0) + int(mask.sum())

    invalid_ages = {
        donor: sorted(ages)
        for donor, ages in donor_ages.items()
        if len(ages) != 1
    }
    if invalid_ages:
        raise ValueError(
            "Fixed reference donor has multiple ages; donor-level age exclusion "
            f"would be ambiguous: {dict(list(invalid_ages.items())[:3])}"
        )
    return {
        "donor_sums": donor_sums,
        "donor_counts": donor_counts,
        "donor_ct_sums": donor_ct_sums,
        "donor_ct_counts": donor_ct_counts,
        "donor_ages": {donor: next(iter(ages)) for donor, ages in donor_ages.items()},
        "donor_sexes": {donor: ",".join(sorted(sexes)) for donor, sexes in donor_sexes.items()},
        "n_reference_cells": int(len(indices)),
        "n_reference_donors": int(len(donor_sums)),
    }


def fixed_reference_group(
    summary: dict,
    *,
    target_age: float,
    direction: str,
    scope: str,
) -> tuple[np.ndarray, np.ndarray, int, list[float]]:
    """Return donor means for all fixed refs on one side of target age."""
    donor_ages = summary["donor_ages"]
    selected = {
        donor
        for donor, age in donor_ages.items()
        if not np.isclose(float(age), float(target_age))
        and ((direction == "young" and float(age) < float(target_age))
             or (direction == "old" and float(age) > float(target_age)))
    }
    sums = (
        {
            donor: summary["donor_sums"][donor]
            for donor in selected
        }
        if scope == "all"
        else {
            donor: value
            for (donor, cell_type), value in summary["donor_ct_sums"].items()
            if cell_type == scope and donor in selected
        }
    )
    counts = (
        {
            donor: summary["donor_counts"][donor]
            for donor in selected
        }
        if scope == "all"
        else {
            donor: value
            for (donor, cell_type), value in summary["donor_ct_counts"].items()
            if cell_type == scope and donor in selected
        }
    )
    selected_donors = sorted(set(sums).intersection(counts))
    if not selected_donors:
        return np.empty((0, 0), dtype=np.float64), np.asarray([], dtype=str), 0, []
    means = np.stack([sums[donor] / float(counts[donor]) for donor in selected_donors], axis=0)
    n_cells = int(sum(counts[donor] for donor in selected_donors))
    ages = sorted({float(donor_ages[donor]) for donor in selected_donors})
    return means, np.asarray(selected_donors, dtype=str), n_cells, ages


def donor_pseudobulk_mean(values: np.ndarray, donors: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    donors = np.asarray(donors, dtype=str)
    unique = pd.unique(donors)
    if len(unique) == 0:
        raise ValueError("Cannot aggregate an empty donor group")
    donor_means = np.stack([values[donors == donor].mean(axis=0) for donor in unique], axis=0)
    return np.clip(donor_means.mean(axis=0), 0.0, None) + eps


def normalized_logfc(target: np.ndarray, target_donors: np.ndarray, reference: np.ndarray, reference_donors: np.ndarray) -> np.ndarray:
    return np.log2(donor_pseudobulk_mean(target, target_donors)) - np.log2(
        donor_pseudobulk_mean(reference, reference_donors)
    )


def prediction_indices(values: np.lib.npyio.NpzFile, obs_names: np.ndarray) -> np.ndarray:
    if "holdout_obs_names" in values:
        lookup = {name: index for index, name in enumerate(obs_names)}
        names = np.asarray(values["holdout_obs_names"], dtype=str)
        missing = sorted(set(names) - set(lookup))
        if missing:
            raise ValueError(f"Prediction artifact contains unknown holdout obs names: {missing[:3]}")
        return np.asarray([lookup[name] for name in names], dtype=np.int64)
    if "holdout_eval_indices" in values:
        indices = np.asarray(values["holdout_eval_indices"], dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= len(obs_names)):
            raise ValueError("Prediction artifact contains out-of-range holdout_eval_indices")
        return indices
    raise ValueError("Prediction artifact has neither holdout_obs_names nor holdout_eval_indices")


def corr_rows(predicted: np.ndarray, real: np.ndarray) -> dict:
    return _compare_vectors(predicted, real, rank=real)


def direction_rows(predicted: np.ndarray, real: np.ndarray) -> dict:
    """Agreement of the predicted and real logFC signs, gene by gene."""
    predicted = np.asarray(predicted, dtype=float)
    real = np.asarray(real, dtype=float)
    finite = np.isfinite(predicted) & np.isfinite(real)
    if not finite.any():
        return {
            "direction_agreement_all": None,
            "direction_agreement_nonzero": None,
            "n_direction_genes": 0,
            "n_direction_nonzero_genes": 0,
        }
    predicted_sign = np.sign(predicted[finite])
    real_sign = np.sign(real[finite])
    nonzero = (predicted_sign != 0) & (real_sign != 0)
    return {
        "direction_agreement_all": float(np.mean(predicted_sign == real_sign)),
        "direction_agreement_nonzero": float(np.mean(predicted_sign[nonzero] == real_sign[nonzero])) if nonzero.any() else None,
        "n_direction_genes": int(finite.sum()),
        "n_direction_nonzero_genes": int(nonzero.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cell-type-key", default="Annotation_Level1")
    parser.add_argument("--donor-key", default="donor_id")
    parser.add_argument("--min-cells", type=int, default=50)
    parser.add_argument(
        "--reference-mode",
        choices=("within_heldout", "fixed_train_all_excluding_target_age"),
        default="within_heldout",
    )
    parser.add_argument("--reference-chunk-size", type=int, default=8192)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(args.h5ad, backed="r")
    obs_names = np.asarray(adata.obs_names, dtype=str)
    gene_names = np.asarray(adata.var_names, dtype=str)
    real_ages_all = adata.obs["age"].to_numpy(dtype=float)
    real_donors_all = adata.obs[args.donor_key].astype(str).to_numpy()
    real_cell_types_all = adata.obs[args.cell_type_key].astype(str).to_numpy()
    real_sexes_all = (
        adata.obs["sex"].astype(str).to_numpy()
        if "sex" in adata.obs
        else np.full(len(adata), "", dtype=str)
    )
    categories = set(real_cell_types_all)
    metric_rows: list[dict] = []
    reference_cache: dict[str, dict] = {}

    for artifact_arg in args.artifact:
        artifact = Path(artifact_arg).resolve()
        # Prediction artifacts may store var_names as an object array; they are
        # local NPZs written by this pipeline.
        with np.load(artifact, allow_pickle=True) as values:
            x_pred = np.asarray(values["x_pred"], dtype=np.float64)
            target_age = np.asarray(values["target_age"], dtype=float)
            pred_donors = np.asarray(values["pred_source_donor"], dtype=str)
            pred_ct = np.asarray(values["pred_cell_type"], dtype=str) if "pred_cell_type" in values else np.full(len(x_pred), "global", dtype=str)
            indices = prediction_indices(values, obs_names)
            fixed_reference_indices = reference_indices_from_artifact(values, obs_names)
            artifact_genes = np.asarray(values["gene_names"], dtype=str)
        if x_pred.ndim != 2 or x_pred.shape[1] != len(gene_names):
            raise ValueError(f"{artifact}: x_pred shape does not match the H5AD gene space")
        if len({len(target_age), len(pred_donors), len(pred_ct), len(indices)}) != 1:
            raise ValueError(f"{artifact}: prediction metadata arrays are not aligned")
        if not np.array_equal(artifact_genes, gene_names):
            raise ValueError(f"{artifact}: gene_names are not exactly aligned to the H5AD")
        source_donor_ids = set(real_donors_all)
        unknown_source_donors = sorted(set(pred_donors) - source_donor_ids)
        if unknown_source_donors:
            raise ValueError(f"{artifact}: generated source donors not present in source H5AD: {unknown_source_donors[:5]}")

        true_expr = rows_from_h5ad(adata, indices)
        true_donors = real_donors_all[indices]
        true_ct = real_cell_types_all[indices]
        artifact_id = str(artifact)
        reference_key = fixed_reference_indices.tobytes()
        if args.reference_mode == "fixed_train_all_excluding_target_age":
            if reference_key not in reference_cache:
                reference_cache[reference_key] = build_fixed_reference_summary(
                    adata,
                    fixed_reference_indices,
                    real_ages_all,
                    real_donors_all,
                    real_cell_types_all,
                    real_sexes_all,
                    chunk_size=args.reference_chunk_size,
                )
            fixed_reference = reference_cache[reference_key]
        else:
            fixed_reference = None
        age_values = sorted(float(age) for age in pd.unique(target_age))
        for age in age_values:
            target_mask = np.isclose(target_age, age)
            younger_mask = target_age < age - 1e-6
            older_mask = target_age > age + 1e-6
            for direction, reference_mask in (("young", younger_mask), ("old", older_mask)):
                if args.reference_mode == "within_heldout" and not np.any(reference_mask):
                    continue
                scope_labels = ["all"]
                common_ct = sorted(categories.intersection(set(pred_ct[target_mask])))
                if common_ct and not (len(common_ct) == 1 and common_ct[0] in {"global", "unknown"}):
                    scope_labels.extend(common_ct)
                for scope in scope_labels:
                    if scope == "all":
                        target_scope = target_mask
                        reference_scope = reference_mask
                        true_target_scope = target_mask
                        true_reference_scope = reference_mask
                    else:
                        target_scope = target_mask & (pred_ct == scope)
                        reference_scope = reference_mask & (pred_ct == scope)
                        true_target_scope = target_mask & (true_ct == scope)
                        true_reference_scope = reference_mask & (true_ct == scope)
                    if args.reference_mode == "fixed_train_all_excluding_target_age":
                        fixed_expr, fixed_donors, fixed_cells, fixed_ages = fixed_reference_group(
                            fixed_reference,
                            target_age=age,
                            direction=direction,
                            scope=scope,
                        )
                        reference_scope_count = fixed_cells
                    else:
                        fixed_expr = None
                        fixed_donors = None
                        fixed_cells = int(reference_scope.sum())
                        fixed_ages = sorted(float(value) for value in pd.unique(target_age[reference_scope])) if np.any(reference_scope) else []
                        reference_scope_count = int(reference_scope.sum())
                    if (
                        target_scope.sum() < args.min_cells
                        or reference_scope_count < args.min_cells
                        or true_target_scope.sum() < args.min_cells
                        or (true_reference_scope.sum() if args.reference_mode == "within_heldout" else reference_scope_count) < args.min_cells
                    ):
                        continue
                    if args.reference_mode == "fixed_train_all_excluding_target_age":
                        generated_logfc = normalized_logfc(
                            x_pred[target_scope], pred_donors[target_scope],
                            fixed_expr, fixed_donors,
                        )
                        real_logfc = normalized_logfc(
                            true_expr[true_target_scope], true_donors[true_target_scope],
                            fixed_expr, fixed_donors,
                        )
                        n_reference_donors = int(len(fixed_donors))
                        reference_donor_labels = ",".join(fixed_donors.tolist())
                    else:
                        generated_logfc = normalized_logfc(x_pred[target_scope], pred_donors[target_scope], x_pred[reference_scope], pred_donors[reference_scope])
                        real_logfc = normalized_logfc(true_expr[true_target_scope], true_donors[true_target_scope], true_expr[true_reference_scope], true_donors[true_reference_scope])
                        n_reference_donors = int(len(pd.unique(pred_donors[reference_scope])))
                        reference_donor_labels = ",".join(sorted(pd.unique(pred_donors[reference_scope])))
                    row = {
                        "artifact": artifact_id,
                        "scope": scope,
                        "target_age": int(round(age)),
                        "contrast": direction,
                        "method": (
                            "normalized_donor_pseudobulk_logfc_fixed_train_all_excluding_target_age"
                            if args.reference_mode == "fixed_train_all_excluding_target_age"
                            else "normalized_donor_pseudobulk_logfc"
                        ),
                        "reference_mode": args.reference_mode,
                        "n_generated_target_cells": int(target_scope.sum()),
                        "n_generated_reference_cells": int(fixed_cells),
                        "n_real_target_cells": int(true_target_scope.sum()),
                        "n_real_reference_cells": int(fixed_cells),
                        "n_generated_target_donors": int(len(pd.unique(pred_donors[target_scope]))),
                        "n_generated_reference_donors": n_reference_donors,
                        "n_real_target_donors": int(len(pd.unique(true_donors[true_target_scope]))),
                        "n_real_reference_donors": n_reference_donors,
                        "generated_target_source_donors": ",".join(sorted(pd.unique(pred_donors[target_scope]))),
                        "generated_reference_source_donors": reference_donor_labels,
                        "fixed_reference_ages": ",".join(str(int(round(value))) for value in fixed_ages),
                        "fixed_reference_excludes_target_age": True,
                    }
                    row.update(corr_rows(generated_logfc, real_logfc))
                    row.update(direction_rows(generated_logfc, real_logfc))
                    metric_rows.append(row)

    metrics_path = output_dir / "donor_pseudobulk_normalized_de_metrics.tsv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, sep="\t", index=False)
    print(f"[saved] {metrics_path} ({len(metric_rows)} rows)")


if __name__ == "__main__":
    main()
