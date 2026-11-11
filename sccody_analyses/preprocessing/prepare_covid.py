"""Prepare all three COVID cohorts at the paths used by the analysis configs."""
import argparse
from pathlib import Path
from types import SimpleNamespace

from . import prepare_ren_covid_aida2000_h5ad as ren
from . import prepare_combat_covid_aida2000_h5ad as combat
from . import prepare_stephenson_covid_case_control_h5ad as stephenson

COHORTS = {
    "ren": "COVID19_ALL.h5ad",
    "combat": "COMBAT-CITESeq-DATA.h5ad",
    "stephenson": "Stephenson_E-MTAB-10026_haniffa21.processed.h5ad",
}


def output_path(outdir, cohort):
    return Path(outdir) / f"{cohort}_covid_case_control_aida2000_mean_imputed.h5ad"


def prepare(cohort, input_dir, outdir, aida_reference, *, block_rows=10000, overwrite=False):
    source = Path(input_dir) / COHORTS[cohort]
    output = output_path(outdir, cohort)
    common = dict(overwrite=overwrite, seed=42, target_sum=10000.0,
                  aida_h5ad=str(aida_reference), input_h5ad=str(source), out=str(output),
                  chunk_size=block_rows, max_cells_per_donor_cell_type=2000)
    if cohort == "ren":
        ren.build(SimpleNamespace(source_h5ad=str(source), aida_reference=str(aida_reference), output=output,
                                  min_age=19., max_age=66., block_rows=block_rows, overwrite=overwrite))
    elif cohort == "combat":
        combat.build(SimpleNamespace(**common, min_age=19., max_age=77.))
    else:
        stephenson.build(SimpleNamespace(**common, status=None, min_age=0., max_age=68.))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing the three source H5AD files")
    parser.add_argument("--outdir", type=Path, default=Path("data/covid"))
    parser.add_argument("--aida-reference", "--aida-train", dest="aida_reference", type=Path,
                        default=Path("data/aida_age_holdout_train.h5ad"))
    parser.add_argument("--cohort", choices=list(COHORTS), help="Omit to prepare all three cohorts")
    parser.add_argument("--block-rows", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for cohort in [args.cohort] if args.cohort else COHORTS:
        prepare(cohort, args.input_dir, args.outdir, args.aida_reference,
                block_rows=args.block_rows, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
