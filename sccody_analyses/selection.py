"""Donor-level localization reporting metrics."""
import pandas as pd

from sccody_analyses.checkpoint_selection import localization_metrics

def localization_score(predictions, truth, sample_key, time_key):
    # Collapse population readouts before scoring biological samples.
    valid = predictions.loc[predictions["status"] == "ok", [sample_key, "predicted_time"]].copy()
    if not len(valid):
        raise ValueError("No supported validation sample predictions")
    valid[sample_key] = valid[sample_key].astype(str)
    predicted = valid.groupby(sample_key, observed=True)["predicted_time"].mean().reset_index()
    labels = truth[[sample_key, time_key]].copy()
    labels[sample_key] = labels[sample_key].astype(str)
    if (labels.groupby(sample_key, observed=True)[time_key].nunique() != 1).any():
        raise ValueError("Validation labels must be constant within a sample")
    joined = predicted.merge(labels.drop_duplicates(sample_key), on=sample_key, validate="one_to_one")
    y = pd.to_numeric(joined[time_key]).to_numpy(dtype=float)
    p = joined["predicted_time"].to_numpy(dtype=float)
    return {**localization_metrics(y, p), "samples": sorted(joined[sample_key].tolist())}

