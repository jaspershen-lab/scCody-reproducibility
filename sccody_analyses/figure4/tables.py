"""Build Figure 4 inputs from saved predictions, healthy pairs and residual moments."""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

from sccody_analyses.orthogonal_residual_variance import arm_dir, recipe_groups
from sccody_analyses.preprocessing.prepare_covid import output_path
from sccody_analyses.orthogonal_variance.summarize import COHORT_GROUPS, COVID_GROUPS

AD_GROUPS = ["Global", "Astro", "EN", "Endo", "IN", "Immune", "Mural", "OPC", "Oligo"]
COHORTS = {"Ren": "ren", "COMBAT": "combat", "Stephenson": "stephenson"}
HEALTHY_COLUMNS = ["scope", "group", "source_donor", "target_donor", "source_age", "target_age",
                   "age_gap", "inner_product", "aging_energy", "perpendicular_energy"]


def read_table(path):
    return pd.read_csv(path, sep="\t" if ".tsv" in str(path) else ",")


def normalize_group(value):
    return "Global" if str(value) in ["Global population", "ALL", "Global"] else str(value)


def donor_predictions(frame):
    """Use donor predictions directly, averaging the recorded COVID fits."""
    frame = frame.rename(columns={"cell_type": "population", "chronological_age": "age"}).copy()
    frame["population"] = frame.population.map(normalize_group)
    frame["scope"] = frame.scope.replace({"lineage": "cell_type", "whole_population": "global", "percelltype": "cell_type"})
    frame["cohort"] = frame.cohort.replace({"D11_AD": "PsychAD_AD", "AD": "PsychAD_AD"})
    if "condition" not in frame:
        frame["condition"] = np.where(frame.projection_kind.eq("calibration"), "control", "case")
    frame["delta"] = frame.pred_age - frame.age
    return frame.groupby(["cohort", "scope", "population", "donor_id", "condition"], as_index=False, observed=True).agg(
        age=("age", "first"), delta=("delta", "mean"))


def local_effects(frame, *, min_controls, covid=False):
    cases = frame.loc[frame.condition.eq("case") & frame.age.notna() & frame.delta.notna()]
    controls = frame.loc[frame.condition.eq("control") & frame.age.notna() & frame.delta.notna()]
    if covid:
        cases = cases.sort_values(["age", "donor_id"])
    rows = []
    for case in cases.itertuples():
        near = controls.loc[controls.age.between(case.age - 5, case.age + 5)]
        if len(near) < min_controls:
            continue
        rows.append(dict(population=case.population, disease_donor=case.donor_id, disease_age=case.age,
                         disease_delta=case.delta, n_nearby_reference_donors=len(near),
                         nearby_reference_mean_delta=near.delta.mean(),
                         local_age_match_effect=case.delta - near.delta.mean()))
    return pd.DataFrame(rows), len(cases), len(controls)


def bh(values):
    result = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(np.asarray(values)[valid])]
    adjusted = np.asarray(values)[order] * len(order) / np.arange(1, len(order) + 1)
    result[order] = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    return result


def age_displacement(frame, *, ad_repeats=2000, covid_repeats=10000):
    donors = donor_predictions(frame)
    rows, effects = [], []
    ad_rng = np.random.default_rng(20260903)
    for cohort in ["PsychAD_AD", *["COVID_" + c for c in COHORTS]]:
        is_ad = cohort == "PsychAD_AD"
        for scope in ["global", "cell_type"]:
            rng = ad_rng if is_ad else np.random.default_rng(20261027)
            source = donors.loc[donors.cohort.eq(cohort) & donors.scope.eq(scope)]
            if not is_ad:
                source = source.loc[source.population.isin(COHORT_GROUPS[cohort.removeprefix("COVID_")])]
            for population, group in source.groupby("population", observed=True):
                effect, n_case, n_control = local_effects(group, min_controls=3 if is_ad else 1, covid=not is_ad)
                if effect.empty:
                    continue
                values = effect.local_age_match_effect.to_numpy()
                lo = hi = p = np.nan
                if len(values) >= 3:
                    repeats = ad_repeats if is_ad else covid_repeats
                    draws = [rng.choice(values, len(values), replace=True).mean() for _ in range(repeats)]
                    lo, hi = np.percentile(draws, [2.5, 97.5])
                    p = stats.ttest_1samp(values, 0).pvalue
                rows.append(dict(cohort=cohort, scope=scope,
                    cohort_label="PsychAD Alzheimer disease" if is_ad else "COVID " + cohort.removeprefix("COVID_"),
                    n_effects=len(values), n_disease_donors=n_case, n_reference_donors=n_control,
                    local_age_match_effect_years=values.mean(), bootstrap_ci_low_years=lo,
                    bootstrap_ci_high_years=hi, p_two_sided_t=p, population=population,
                    analysis_method="local_age_match_calibration" if is_ad else "case_minus_age_window_control",
                    reference_label="PsychAD calibration" if is_ad else "COVID control"))
                effects.append(effect.assign(cohort=cohort, scope=scope))
    summary = pd.DataFrame(rows)
    summary["pFDR"] = summary.groupby(["cohort", "scope"], sort=False).p_two_sided_t.transform(lambda s: bh(s.to_numpy()))
    return summary, pd.concat(effects, ignore_index=True)


def healthy_relationships(pairs, holdout_donors):
    pairs = pairs.copy()
    pairs["group"] = pairs.group.map(normalize_group)
    pairs = pairs.loc[pairs.source_donor.astype(str).isin(holdout_donors)
                      & pairs.target_donor.astype(str).isin(holdout_donors)
                      & pairs.age_gap.between(1, 15)]
    pairs["orthogonal_rms"] = np.sqrt(pairs.perpendicular_energy.clip(lower=0))
    unique = pairs.groupby(["scope", "group", "source_donor", "target_donor"], as_index=False, observed=True).agg(
        source_age=("source_age", "first"), target_age=("target_age", "first"),
        age_gap=("age_gap", "first"), orthogonal_rms=("orthogonal_rms", "mean"))
    rows = []
    for group in AD_GROUPS:
        unit = unique.loc[unique.group.eq(group)].dropna(subset=["age_gap", "orthogonal_rms"])
        if unit.empty:
            continue
        x, y = unit.age_gap.to_numpy(), unit.orthogonal_rms.to_numpy()
        fit = stats.linregress(x, y)
        margin = stats.t.ppf(.975, len(x) - 2) * fit.stderr
        rows.append(dict(metric="orthogonal_rms", metric_label="orthogonal RMS", group=group,
            n_pairs=len(x), n_source_donors=unit.source_donor.nunique(), n_target_donors=unit.target_donor.nunique(),
            gap_min_years=x.min(), gap_max_years=x.max(), mean=y.mean(), median=np.median(y),
            pearson_r=fit.rvalue, pearson_p_pair_level=stats.pearsonr(x, y).pvalue,
            spearman_rho=stats.spearmanr(x, y).statistic, spearman_p_pair_level=stats.spearmanr(x, y).pvalue,
            ols_slope_per_year=fit.slope, ols_slope_ci95_low=fit.slope-margin,
            ols_slope_ci95_high=fit.slope+margin, ols_intercept=fit.intercept))
    return pd.DataFrame(rows)


def replay_table(recipes, root):
    frames = []
    for recipe in recipes:
        for group in recipe_groups(recipe):
            frame = read_table(arm_dir(root, recipe, group) / "pair_residual_decomposition.tsv")
            source = read_table(recipe["pairs"])
            source = source.loc[source.group.map(normalize_group).eq(group)]
            if recipe["mode"] == "healthy":
                source = source.loc[source.shift_label.eq("plus_10y")]
                if recipe["target_donor_manifest"]:
                    donors = json.loads(Path(recipe["target_donor_manifest"]).read_text())["holdout_donors"]
                    source = source.loc[source.target_donor.astype(str).isin(donors)]
            else:
                source = source.loc[source.target_kind.eq("disease")]
            reference_key = "source_donor" if recipe["mode"] == "healthy" else "reference_donor"
            source.index = pd.MultiIndex.from_arrays([source.target_donor.astype(str), source[reference_key].astype(str)])
            pair_keys = pd.MultiIndex.from_arrays([frame.target_donor.astype(str), frame.reference_donor.astype(str)])
            source = source.loc[pair_keys]
            inner = frame.inner_product.to_numpy()
            aging = frame.aging_energy.to_numpy()
            frames.append(pd.DataFrame(dict(dataset="AD" if recipe["cohort"] == "AD" else "COVID",
                cohort="PsychAD" if recipe["cohort"] == "AD" else recipe["cohort"], arm=recipe["mode"],
                group_normalized=group, fold=recipe["fold"], target_donor_replay=frame.target_donor.to_numpy(),
                reference_donor_replay=frame.reference_donor.to_numpy(), target_age=source.target_age.to_numpy(),
                inner_product_replay=inner, aging_energy_replay=aging, alpha_replay=frame.alpha.to_numpy(),
                orthogonal_energy=frame.orthogonal_energy.to_numpy(),
                signed_aligned_rms=np.divide(inner, np.sqrt(aging), out=np.full(len(aging), np.nan), where=aging > 1e-12))))
    return pd.concat(frames, ignore_index=True)


def ad_clock_forest(predictions):
    frame = predictions.rename(columns={"cell_type": "population", "chronological_age": "age"}).copy()
    frame["population"] = frame.population.map(normalize_group)
    frame["condition"] = np.where(frame.projection_kind.eq("calibration"), "control", "case")
    frame["delta"] = frame.pred_age - frame.age
    rows = []
    for population, group in frame.groupby("population", observed=True):
        effects, _, _ = local_effects(group, min_controls=3)
        values = effects.local_age_match_effect.to_numpy()
        margin = 1.96 * stats.sem(values)
        rows.append(dict(panel="elasticnet_effect", population=population, mean=values.mean(),
                         lo=values.mean()-margin, hi=values.mean()+margin, n=len(values)))
    return pd.DataFrame(rows)


def covid_clock_predictions(clock_root, input_dir):
    rows = []
    for cohort, key in COHORTS.items():
        obj = ad.read_h5ad(output_path(input_dir, key), backed="r")
        columns = [c for c in ["donor_id", "age", "sex", "condition", "disease_group", "CoVID-19 severity"] if c in obj.obs]
        meta = obj.obs[columns].copy().groupby("donor_id", observed=True).first().reset_index()
        obj.file.close()
        for name, scope in [("covid_clock_global", "whole_population"), ("covid_clock_cell_type", "percelltype")]:
            frames = [read_table(Path(clock_root) / name / f"fold_{fold}/{key}_metacell_predictions.tsv") for fold in range(3)]
            combined = pd.concat(frames, ignore_index=True)
            ensemble = combined.groupby(["donor_id", "cell_type"], as_index=False, observed=True).agg(
                pred_age=("pred_age", "mean"), pred_age_sd_across_folds=("pred_age", "std"), n_folds=("fold", "nunique"))
            ensemble = ensemble.loc[ensemble.cell_type.isin(["ALL", *COVID_GROUPS])]
            ensemble = ensemble.merge(meta, on="donor_id", how="left")
            rows.append(ensemble.assign(cohort="COVID_"+cohort, scope=scope, acceleration=ensemble.pred_age-ensemble.age))
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["age-shifts", "healthy-relationships", "replay", "clocks"])
    parser.add_argument("--outdir", type=Path, default=Path("manuscript/figure_4/results"))
    parser.add_argument("--predictions", nargs="+", type=Path)
    parser.add_argument("--healthy-pairs", nargs="+", type=Path)
    parser.add_argument("--orthogonal-pairs", nargs="+", type=Path,
                        help="Original orthogonal-slope sample, when different from the signed-alignment sample")
    parser.add_argument("--holdout", type=Path, default=Path("configs/donors/psychad_healthy_holdout.json"))
    parser.add_argument("--recipes", type=Path, default=Path("configs/orthogonal_variance.json"))
    parser.add_argument("--orthogonal-root", type=Path, default=Path("artifacts/orthogonal_residual_variance"))
    parser.add_argument("--ad-clock-predictions", type=Path)
    parser.add_argument("--clock-root", type=Path, default=Path("artifacts/clocks"))
    parser.add_argument("--covid-input-dir", type=Path, default=Path("data/covid"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.phase == "age-shifts":
        summary, effects = age_displacement(pd.concat([read_table(p) for p in args.predictions], ignore_index=True))
        summary.to_csv(args.outdir / "age_displacement_summary.csv", index=False)
        effects.loc[effects.cohort.ne("PsychAD_AD")].to_csv(args.outdir / "covid_age_displacement_donor_effects.tsv", sep="\t", index=False)
    elif args.phase == "healthy-relationships":
        pairs = pd.concat([read_table(p) for p in args.healthy_pairs], ignore_index=True)
        donors = json.loads(args.holdout.read_text())["holdout_donors"]
        pairs = pairs.loc[pairs.source_donor.astype(str).isin(donors) & pairs.target_donor.astype(str).isin(donors),
                          HEALTHY_COLUMNS]
        pairs.to_csv(args.outdir / "ad_holdout_real_healthy_aging_pairs.tsv.gz", sep="\t", index=False)
        orthogonal = (pd.concat([read_table(p) for p in args.orthogonal_pairs], ignore_index=True)
                      if args.orthogonal_pairs else pairs)
        healthy_relationships(orthogonal, donors).to_csv(args.outdir / "ad_epoch5_exact_gap_relationship_holdout_mapped_summary.csv", index=False)
    elif args.phase == "replay":
        replay_table(json.loads(args.recipes.read_text()), args.orthogonal_root).to_csv(
            args.outdir / "signed_aligned_rms_pair_replay_all.tsv.gz", sep="\t", index=False)
    else:
        ad_clock_forest(read_table(args.ad_clock_predictions)).to_csv(args.outdir / "ad_elasticnet_forest_plot_data.csv", index=False)
        covid_clock_predictions(args.clock_root, args.covid_input_dir).to_csv(
            args.outdir / "covid_elasticnet_age_acceleration_donor_predictions.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
