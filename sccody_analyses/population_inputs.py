"""Shared target-cell sampling for held-out population reconstruction."""
import numpy as np


def target_rows(obs, age_key, target_ages, max_cells, seed, strata=("sex",)):
    ages = obs[age_key].to_numpy(dtype=float)
    selected = []
    for age in sorted(target_ages):
        rows = np.flatnonzero(np.isclose(ages, age))
        rng = np.random.default_rng(int(seed) + int(round(age * 100)))
        rows = rng.choice(rows, min(max_cells, len(rows)) if max_cells > 0 else len(rows), replace=False)
        # Keep prediction and observed rows in the same matched-stratum order.
        if strata and len(rows):
            groups = obs.iloc[rows].groupby(list(strata), observed=True, sort=True).indices
            rows = np.concatenate([rows[positions] for positions in groups.values()])
        selected.append(rows)
    return np.concatenate(selected).astype(np.int64)
