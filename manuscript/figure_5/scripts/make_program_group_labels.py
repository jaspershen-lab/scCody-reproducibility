#!/usr/bin/env python3
"""Per-row cell-group labels for decode_component_program_scores.py --labels.

COVID cohorts: CD14_Mono / CD16_Mono (harmonised as in prepare_covid_subtype_donor_table.py), other
myeloid labels -> "other".  AD (PsychAD): EN subclass.  One row per row of each target h5ad.
Output: RES/group_labels/<dataset>.tsv  (column: label)
"""
from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_covid_subtype_donor_table import H5AD, LABEL_KEY, RES, subtype  # noqa: E402

OUT = RES / "group_labels"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for cohort, path in H5AD.items():
        raw = ad.read_h5ad(path, backed="r").obs[LABEL_KEY[cohort]].astype(str).to_numpy()
        labels = np.asarray([subtype(cohort, x) or "other" for x in raw])
        pd.DataFrame({"label": labels}).to_csv(OUT / f"{cohort}.tsv", sep="\t", index=False)
        print(cohort, pd.Series(labels).value_counts().to_dict())
    ad_h5ad = Path(__file__).resolve().parents[3] / "data/psychad/ad_case_control.h5ad"
    labels = ad.read_h5ad(ad_h5ad, backed="r").obs["subclass"].astype(str).to_numpy()
    pd.DataFrame({"label": labels}).to_csv(OUT / "AD.tsv", sep="\t", index=False)
    print("AD", len(labels), "rows")


if __name__ == "__main__":
    main()
