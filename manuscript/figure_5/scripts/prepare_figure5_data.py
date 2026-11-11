#!/usr/bin/env python3
"""Build the two main Figure 5b subclass tables from saved analysis outputs."""
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "artifacts" / "residual_programs"
OUT = REPO / "manuscript" / "figure_5" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def ad_subclass() -> bool:
    root = RES / "analysis/AD_EN_subclass"
    if not (root / "subclass_latent_readout.tsv").exists():
        return False
    for name in ("subclass_latent_readout", "subclass_shared_contribution"):
        pd.read_csv(root / f"{name}.tsv", sep="\t").to_csv(OUT / f"ad_en_{name}.csv", index=False)
    return True


if __name__ == "__main__":
    print("AD subclass results:", "included" if ad_subclass() else "missing: run analyze_ad_subclass_residual_readout.py")
    print("Wrote tables to", OUT)
