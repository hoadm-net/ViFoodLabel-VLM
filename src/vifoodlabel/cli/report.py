"""Combine whatever tiers have been run into summary tables + the
manuscript's figures (Fig1-Fig5 in results/figures/, plus a supplementary
cost-effectiveness figure) -- see figstyle.py for the journal's submission
constraints and plotting.py for the chart builders. Captions (which belong
in the manuscript text, not baked into the image) are written to
results/figures/captions.md.

Tier 1 (results/scored/tier1_*.csv) is required for the fullest report;
Tier 2/3/4 and cost outputs are used if present, skipped with a message
otherwise. Safe to re-run at any time as more scored data appears.
"""

from __future__ import annotations

import argparse

import pandas as pd

from vifoodlabel.config import FIGURES_DIR, RAW_RESULTS_DIR, SCORED_RESULTS_DIR
from vifoodlabel.cost import LEDGER_PATH
from vifoodlabel.io_utils import load_json
from vifoodlabel.figstyle import apply_style, save_figure, save_online_resource
from vifoodlabel.metrics import model_field_summary, model_summary
from vifoodlabel.plotting import (
    plot_cost_effectiveness,
    plot_fig1_heatmap,
    plot_fig2_nutrition,
    plot_fig3_prompt_sensitivity,
    plot_fig4_degradation,
    plot_fig5_error_taxonomy,
)
from vifoodlabel.prompts import CANONICAL_CONDITION
from vifoodlabel.stats import (
    aggregation_robustness,
    bootstrap_ci,
    bootstrap_ci_table,
    mcnemar_pairwise,
    paired_bootstrap_diff,
    paired_tests_pairwise,
)

# Categories under 1% of every model's errors are pooled into "other" for
# Fig 5 -- kept separate they'd each be a legend entry pointing at an
# invisible sliver of a bar. Full 10-category detail stays in error_taxonomy.csv.
_OTHER_CATEGORIES = ["pairing_error", "diacritics", "wrong_unit", "malformed_json"]

CAPTIONS = {
    1: (
        "Fig. 1 Per-field F1 for the seven evaluated models under the canonical "
        "Vietnamese zero-shot condition on all 600 images. Scalar fields use the "
        "lenient match criterion, list fields use Hungarian set matching, and the "
        "nutrition column reports name F1. The rightmost column reports the macro "
        "field F1 averaged over all nine fields. Cell shading encodes F1 on a "
        "fixed 0 to 1 scale"
    ),
    2: (
        "Fig. 2 Decomposition of nutrition table extraction into three "
        "independent questions. a Name F1 measures whether the model recovers "
        "the correct set of nutrient names, matched through a bilingual "
        "Vietnamese and English alias table before fuzzy fallback. b Pairing "
        "accuracy measures whether values are attached to the correct nutrient "
        "row, plotted on a truncated vertical scale from 0.95 to 1.00 because all "
        "models fall within that range. c Value accuracy measures whether the "
        "number and unit are read correctly for matched name pairs"
    ),
    3: (
        "Fig. 3 Prompt sensitivity on the pinned 120 image subset, expressed as a "
        "change in macro field F1 relative to the Vietnamese zero-shot baseline. "
        "a Effect of adding one synthetic exemplar while holding the instruction "
        "language fixed. b Effect of switching the instruction language from "
        "Vietnamese to English while holding the shot count fixed. Both panels "
        "share the same horizontal scale. The vertical line in b marks zero change"
    ),
    4: (
        "Fig. 4 Degradation of macro field F1 under three synthetic corruptions "
        "applied to the pinned 120 image subset, where severity 0 is the clean "
        "Tier 1 baseline. a Gaussian blur at sigma 1.5, 3.5 and 6.0. b Radial "
        "glare highlight at three intensity and radius settings with randomised "
        "centre position. c Rotation by 5, 15 and 30 degrees, cropped to the "
        "largest inscribed rectangle so that no blank corners remain. The circled "
        "point in a marks a condition where JSON validity fell to 0.34, so the "
        "reported score reflects parse failures as well as extraction quality"
    ),
    5: (
        "Fig. 5 Distribution of automatically classified error categories, shown "
        "as the share of each model's own incorrect field instances. Categories "
        "accounting for under one percent of every model's errors are grouped as "
        "other. Hatching distinguishes categories independently of colour"
    ),
    "OnlineResource1": (
        "Online Resource 1 Cumulative API cost across all four experimental "
        "tiers plotted against Tier 1 macro field F1. Vintern-3B-beta sits at "
        "zero cost because it is self-hosted; the GPU cost of running it is not "
        "included"
    ),
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass


def _load(name: str) -> pd.DataFrame | None:
    path = SCORED_RESULTS_DIR / name
    if not path.exists():
        print(f"  (skipping, not found: {path})")
        return None
    return pd.read_csv(path)


def _save(fig, number: int, online_resource: bool = False, vector_dpi: float | None = None) -> None:
    saver = save_online_resource if online_resource else save_figure
    paths = saver(fig, number, vector_dpi=vector_dpi)
    print(f"Saved {', '.join(paths)}")


def _write_captions() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "captions.md"
    lines = [CAPTIONS[k] for k in (1, 2, 3, 4, 5, "OnlineResource1")]
    path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote captions to {path}")


def report_tier1() -> pd.DataFrame | None:
    print("\n=== Tier 1: main benchmark ===")
    field_df = _load("tier1_field_scores.csv")
    image_df = _load("tier1_image_scores.csv")
    if field_df is None or image_df is None:
        print("Run 'main.py benchmark' first.")
        return None

    print("\nPer-model summary:")
    summary = model_summary(image_df)
    print(summary.to_string(index=False))
    summary.to_csv(SCORED_RESULTS_DIR / "tier1_model_summary.csv", index=False)

    print("\nBootstrap 95% CI (macro field-level F1):")
    ci_f1 = bootstrap_ci_table(image_df, value_col="macro_field_f1")
    print(ci_f1.to_string(index=False))
    ci_f1.to_csv(SCORED_RESULTS_DIR / "tier1_bootstrap_ci_f1.csv", index=False)

    print("\nBootstrap 95% CI (nutrition pairing accuracy):")
    ci_pairing = bootstrap_ci_table(image_df, value_col="pairing_accuracy")
    print(ci_pairing.to_string(index=False))
    ci_pairing.to_csv(SCORED_RESULTS_DIR / "tier1_bootstrap_ci_pairing.csv", index=False)

    print("\nPer-field summary (model x field):")
    field_summary = model_field_summary(field_df)
    print(field_summary.to_string(index=False))
    field_summary.to_csv(SCORED_RESULTS_DIR / "tier1_field_summary.csv", index=False)

    print("\nMcNemar pairwise tests (per field, exact-match binary outcome, Holm-corrected):")
    mcnemar_rows = []
    for f in field_df["field"].unique():
        sub = field_df[field_df["field"] == f].copy()
        sub["correct"] = (sub["f1"] >= 0.999).astype(int)
        result = mcnemar_pairwise(sub, value_col="correct")
        if result.empty:
            continue
        result.insert(0, "field", f)
        mcnemar_rows.append(result)
    if mcnemar_rows:
        mcnemar_df = pd.concat(mcnemar_rows, ignore_index=True)
        print(mcnemar_df.to_string(index=False))
        mcnemar_df.to_csv(SCORED_RESULTS_DIR / "tier1_mcnemar.csv", index=False)
        _mcnemar_counts(mcnemar_df)
    else:
        print("  (not enough models/images yet for a pairwise comparison)")

    _tier1_statistics_robustness(image_df)
    _tier1_emission(field_df)

    # vector_dpi=150: the heatmap's imshow cells are a tiny array (7x9);
    # left at the default 600dpi, the embedded raster balloons the PDF/EPS
    # to tens of MB for zero visual gain (see figstyle._export).
    macro_f1 = image_df.groupby("model_key")["macro_field_f1"].mean()
    _save(plot_fig1_heatmap(field_summary, macro_f1), 1, vector_dpi=150)

    # model_field_summary() only aggregates precision/recall/f1; pairing_accuracy
    # and value_accuracy live only on the raw nutrition rows of field_df.
    nutrition_rows = field_df[field_df["field"] == "nutrition"]
    if not nutrition_rows.empty:
        nutrition_summary = nutrition_rows.groupby("model_key").agg(
            f1=("f1", "mean"), pairing_accuracy=("pairing_accuracy", "mean"), value_accuracy=("value_accuracy", "mean"),
        ).reset_index()
        # Name precision/recall alongside -- pairing/value accuracy are
        # conditional on name matching, so they can't be read without them.
        nutrition_detail = nutrition_rows.groupby("model_key").agg(
            name_precision=("precision", "mean"), name_recall=("recall", "mean"), name_f1=("f1", "mean"),
            pairing_accuracy=("pairing_accuracy", "mean"), value_accuracy=("value_accuracy", "mean"),
            n_images=("image_id", "nunique"),
        ).reset_index()
        nutrition_detail["recall_minus_precision"] = nutrition_detail["name_recall"] - nutrition_detail["name_precision"]
        print("\nNutrition decomposition:")
        print(nutrition_detail.round(4).to_string(index=False))
        nutrition_detail.to_csv(SCORED_RESULTS_DIR / "tier1_nutrition_summary.csv", index=False)
        _save(plot_fig2_nutrition(nutrition_summary), 2)

    return image_df


def _mcnemar_counts(mcnemar_df: pd.DataFrame) -> None:
    """Appendix Table 5 equivalent: per model pair, how many of the nine
    fields separate at alpha=0.05 -- under Holm (as reported) and under
    plain Bonferroni (for the Holm-vs-Bonferroni question). Plus per field:
    how many pairs separate, and the smallest discordant count behind any test."""
    sig_holm = [c for c in mcnemar_df.columns if c.startswith("significant_at_")][0]
    sig_bonf = [c for c in mcnemar_df.columns if c.startswith("significant_bonferroni_at_")][0]
    df = mcnemar_df.copy()
    df["n_discordant"] = df["a_only_correct"] + df["b_only_correct"]
    pairs = df.groupby(["group_a", "group_b"]).agg(
        fields_sig_holm=(sig_holm, "sum"), fields_sig_bonferroni=(sig_bonf, "sum"),
        min_discordant=("n_discordant", "min"),
    ).reset_index()
    by_field = df.groupby("field").agg(
        pairs_sig_holm=(sig_holm, "sum"), pairs_sig_bonferroni=(sig_bonf, "sum"),
        n_pairs=("group_a", "size"), min_discordant=("n_discordant", "min"),
    ).reset_index()
    print("\nMcNemar: fields separated per model pair (Holm vs Bonferroni):")
    print(pairs.to_string(index=False))
    print("\nMcNemar: pairs separated per field:")
    print(by_field.to_string(index=False))
    print(f"  total significant tests: Holm {int(df[sig_holm].sum())}, Bonferroni {int(df[sig_bonf].sum())} of {len(df)}")
    pairs.to_csv(SCORED_RESULTS_DIR / "tier1_mcnemar_pair_counts.csv", index=False)
    by_field.to_csv(SCORED_RESULTS_DIR / "tier1_mcnemar_field_counts.csv", index=False)


def _tier1_statistics_robustness(image_df: pd.DataFrame) -> None:
    """Answers to the statistics questions in review: does the ranking
    depend on the arithmetic mean (vs median / trimmed mean), and do
    parametric (paired t) and non-parametric (Wilcoxon) paired tests on the
    per-image macro F1 agree?"""
    tier1 = image_df[image_df["condition"] == CANONICAL_CONDITION]
    robust = aggregation_robustness(tier1, "macro_field_f1")
    print("\nAggregation robustness (macro field F1):")
    print(robust.round(4).to_string(index=False))
    robust.to_csv(SCORED_RESULTS_DIR / "tier1_aggregation_robustness.csv", index=False)

    paired = paired_tests_pairwise(tier1, "macro_field_f1")
    if not paired.empty:
        print("\nPaired tests on per-image macro field F1 (Holm within each test):")
        print(paired.round(5).to_string(index=False))
        paired.to_csv(SCORED_RESULTS_DIR / "tier1_paired_tests_macro_f1.csv", index=False)


def _tier1_emission(field_df: pd.DataFrame) -> None:
    """How much each model *emits* relative to the reference: per list/table
    field, predicted vs reference item counts (on images where the
    reference is non-empty), share of such images left empty, and per-model
    output length / stop reasons from the raw response cache. This is the
    direct evidence for the "emission vs recognition" reading of Vintern."""
    if "n_pred" not in field_df.columns or field_df["n_pred"].isna().any():
        print("  (skipping emission analysis: scored CSV predates n_pred/n_gt -- re-run 'benchmark' to re-score from cache)")
        return
    df = field_df[(field_df["condition"] == CANONICAL_CONDITION) & (field_df["n_gt"] > 0)]
    emission = df.groupby(["model_key", "field"]).agg(
        n_images=("image_id", "nunique"),
        mean_n_gt=("n_gt", "mean"), mean_n_pred=("n_pred", "mean"),
        sum_n_gt=("n_gt", "sum"), sum_n_pred=("n_pred", "sum"),
        empty_rate=("n_pred", lambda v: float((v == 0).mean())),
        mean_recall=("recall", "mean"), mean_precision=("precision", "mean"),
    ).reset_index()
    emission["emitted_ratio"] = emission["sum_n_pred"] / emission["sum_n_gt"]
    print("\nEmission: predicted vs reference item counts (images with non-empty reference):")
    print(emission.round(3).to_string(index=False))
    emission.to_csv(SCORED_RESULTS_DIR / "tier1_emission_by_field.csv", index=False)

    rows = []
    tier_dir = RAW_RESULTS_DIR / "benchmark"
    for model_dir in sorted(tier_dir.glob("*")) if tier_dir.exists() else []:
        recs = [load_json(p) for p in model_dir.glob(f"*__{CANONICAL_CONDITION}.json")]
        if not recs:
            continue
        out = pd.Series([r.get("output_tokens") or 0 for r in recs], dtype=float)
        fr = pd.Series([r.get("finish_reason") for r in recs])
        rows.append({
            "model_key": recs[0].get("model_key"), "n": len(recs),
            "mean_output_tokens": out.mean(), "median_output_tokens": out.median(),
            "share_truncated": float((fr == "length").mean()), "share_loop_cutoff": float((fr == "loop_cutoff").mean()),
        })
    if rows:
        tokens = pd.DataFrame(rows)
        print("\nOutput length and stop reasons (Tier 1 raw responses):")
        print(tokens.round(3).to_string(index=False))
        tokens.to_csv(SCORED_RESULTS_DIR / "tier1_output_length.csv", index=False)


def report_tier2() -> None:
    print("\n=== Tier 2: prompt sensitivity ===")
    image_df = _load("tier2_image_scores.csv")
    if image_df is None:
        print("Run 'main.py prompt-sensitivity' first.")
        return
    summary = model_summary(image_df)
    print(summary.sort_values(["model_key", "condition"]).to_string(index=False))
    summary.to_csv(SCORED_RESULTS_DIR / "tier2_condition_summary.csv", index=False)

    # Excludes en_one on purpose -- not part of the active ablation (see
    # prompts.py); any leftover cached rows from before that decision would
    # otherwise skew the pivot below.
    active = summary[summary["condition"].isin(["vi_zero", "vi_one", "en_zero"])]
    pivot = active.pivot(index="model_key", columns="condition", values="mean_macro_field_f1")
    if {"vi_zero", "vi_one", "en_zero"}.issubset(pivot.columns):
        pivot["shot_effect"] = pivot["vi_one"] - pivot["vi_zero"]
        pivot["language_effect"] = pivot["en_zero"] - pivot["vi_zero"]
        _save(plot_fig3_prompt_sensitivity(pivot), 3)


def report_tier3(tier1_image_df: pd.DataFrame | None) -> None:
    print("\n=== Tier 3: perturbation robustness ===")
    image_df = _load("tier3_image_scores.csv")
    if image_df is None:
        print("Run 'main.py perturbation' first.")
        return

    image_df = image_df.copy()
    image_df[["kind", "severity"]] = image_df["condition"].str.extract(r"(\w+)_s(\d)")
    image_df["severity"] = image_df["severity"].astype(int)

    if tier1_image_df is not None:
        clean = tier1_image_df[tier1_image_df["condition"] == CANONICAL_CONDITION].copy()
        subset_ids = set(image_df["image_id"].unique())
        clean = clean[clean["image_id"].isin(subset_ids)]
        for kind in image_df["kind"].unique():
            clean_k = clean.copy()
            clean_k["kind"] = kind
            clean_k["severity"] = 0
            image_df = pd.concat([image_df, clean_k], ignore_index=True)
    else:
        print("  (tier1 not available — degradation curve will not include the clean/severity-0 baseline)")

    curve = (
        image_df.groupby(["model_key", "kind", "severity"])
        .agg(
            mean_macro_field_f1=("macro_field_f1", "mean"),
            json_valid_rate=("json_valid", "mean"),
            n_images=("image_id", "nunique"),
        )
        .reset_index()
        .sort_values(["kind", "model_key", "severity"])
    )
    print(curve.to_string(index=False))
    curve.to_csv(SCORED_RESULTS_DIR / "tier3_degradation_curve.csv", index=False)
    _tier3_detail(image_df)

    _save(plot_fig4_degradation(curve), 4)


def _tier3_detail(image_df: pd.DataFrame) -> None:
    """Per model x corruption x severity: mean with bootstrap CI, and the
    paired change vs. the clean (severity 0) score on the same images with
    its own bootstrap CI and relative change -- the measurement detail
    behind the degradation curves (e.g. rotation's gradual decline). Plus
    the per-field breakdown of each corruption at each severity."""
    if 0 not in set(image_df["severity"]):
        print("  (skipping Tier 3 detail: no clean severity-0 baseline available)")
        return
    rows = []
    for (model_key, kind), g in image_df.groupby(["model_key", "kind"]):
        clean = g[g["severity"] == 0].set_index("image_id")["macro_field_f1"]
        for severity, gs in g.groupby("severity"):
            vals = gs.set_index("image_id")["macro_field_f1"]
            ci = bootstrap_ci(vals.to_numpy())
            common = vals.index.intersection(clean.index)
            diff = paired_bootstrap_diff(vals.loc[common].to_numpy(), clean.loc[common].to_numpy())
            rows.append({
                "model_key": model_key, "kind": kind, "severity": int(severity),
                "mean": ci["mean"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                "delta_vs_clean": diff["mean_diff"], "delta_ci_low": diff["diff_ci_low"], "delta_ci_high": diff["diff_ci_high"],
                "relative_change": diff["mean_diff"] / clean.loc[common].mean() if len(common) else float("nan"),
                "json_valid_rate": float(gs["json_valid"].mean()), "n_images": len(vals),
            })
    detail = pd.DataFrame(rows).sort_values(["kind", "model_key", "severity"])
    print("\nTier 3 detail (paired change vs clean, bootstrap 95% CI):")
    print(detail.round(4).to_string(index=False))
    detail.to_csv(SCORED_RESULTS_DIR / "tier3_detail.csv", index=False)

    field_df = _load("tier3_field_scores.csv")
    tier1_field = _load("tier1_field_scores.csv")
    if field_df is None or tier1_field is None:
        return
    field_df = field_df.copy()
    field_df[["kind", "severity"]] = field_df["condition"].str.extract(r"(\w+)_s(\d)")
    field_df["severity"] = field_df["severity"].astype(int)
    clean = tier1_field[(tier1_field["condition"] == CANONICAL_CONDITION) & tier1_field["image_id"].isin(set(field_df["image_id"]))]
    parts = [field_df]
    for kind in field_df["kind"].unique():
        parts.append(clean.assign(kind=kind, severity=0))
    by_field = (
        pd.concat(parts, ignore_index=True)
        .groupby(["kind", "model_key", "field", "severity"])["f1"].mean()
        .unstack("severity").reset_index()
    )
    by_field.columns = [f"s{c}" if isinstance(c, (int, float)) else c for c in by_field.columns]
    print("\nTier 3 per-field F1 by severity (s0 = clean):")
    print(by_field.round(3).to_string(index=False))
    by_field.to_csv(SCORED_RESULTS_DIR / "tier3_by_field.csv", index=False)


def report_ocr_pipeline(tier1_image_df: pd.DataFrame | None) -> None:
    """OCR -> LLM baseline vs. image input. Two comparisons, each on the
    images both arms scored: the same model fed OCR text instead of the
    image (e.g. MiMo-V2.5), and Vintern's language backbone fed OCR text
    vs. the ~3B VLMs fed the image."""
    print("\n=== OCR -> LLM baseline ===")
    image_df = _load("ocr_pipeline_image_scores.csv")
    field_df = _load("ocr_pipeline_field_scores.csv")
    if image_df is None or field_df is None:
        print("Run scripts/run_ocr.py, then 'main.py ocr-pipeline' first.")
        return
    summary = model_summary(image_df)
    ci = bootstrap_ci_table(image_df, value_col="macro_field_f1")
    summary = summary.merge(ci[["model_key", "condition", "ci_low", "ci_high"]], on=["model_key", "condition"])
    print(summary.round(4).to_string(index=False))
    summary.to_csv(SCORED_RESULTS_DIR / "ocr_pipeline_summary.csv", index=False)

    fields = model_field_summary(field_df)
    print(fields.round(3).to_string(index=False))
    fields.to_csv(SCORED_RESULTS_DIR / "ocr_pipeline_field_summary.csv", index=False)

    if tier1_image_df is None:
        return
    tier1 = tier1_image_df[tier1_image_df["condition"] == CANONICAL_CONDITION]
    rows = []
    for (ocr_model, condition), g in image_df.groupby(["model_key", "condition"]):
        ocr = g.set_index("image_id")["macro_field_f1"]
        for vlm, gv in tier1.groupby("model_key"):
            img = gv.set_index("image_id")["macro_field_f1"]
            common = ocr.index.intersection(img.index)
            if not len(common):
                continue
            d = paired_bootstrap_diff(ocr.loc[common].to_numpy(), img.loc[common].to_numpy())
            rows.append({
                "ocr_model": ocr_model, "ocr_condition": condition, "image_model": vlm,
                "same_model": ocr_model == vlm, "n_images": len(common),
                "ocr_macro_f1": ocr.loc[common].mean(), "image_macro_f1": img.loc[common].mean(),
                "delta_ocr_minus_image": d["mean_diff"], "delta_ci_low": d["diff_ci_low"], "delta_ci_high": d["diff_ci_high"],
            })
    if rows:
        comp = pd.DataFrame(rows).sort_values(["ocr_model", "image_macro_f1"], ascending=[True, False])
        print("\nOCR text vs image input (paired on common images):")
        print(comp.round(4).to_string(index=False))
        comp.to_csv(SCORED_RESULTS_DIR / "ocr_pipeline_vs_image.csv", index=False)


def report_tier4() -> None:
    print("\n=== Tier 4: error taxonomy ===")
    df = _load("error_taxonomy.csv")
    if df is None:
        print("Run 'main.py error-sample' first.")
        return
    df = df.copy()
    df["error_category"] = df["error_category"].where(~df["error_category"].isin(_OTHER_CATEGORIES), "other")
    shares = pd.crosstab(df["model_key"], df["error_category"], normalize="index")
    _save(plot_fig5_error_taxonomy(shares), 5)


def report_cost(tier1_image_df: pd.DataFrame | None) -> None:
    print("\n=== Cost vs. accuracy (supplementary) ===")
    if tier1_image_df is None or not LEDGER_PATH.exists():
        print(f"  (skipping, need Tier 1 scores and {LEDGER_PATH})")
        return
    ledger = pd.read_csv(LEDGER_PATH)
    cost_by_model = ledger.groupby("model_key")["cost_usd"].sum()
    macro_f1 = tier1_image_df.groupby("model_key")["macro_field_f1"].mean()
    print(cost_by_model.reindex(macro_f1.index).round(2).to_string())
    _save(plot_cost_effectiveness(cost_by_model, macro_f1), 1, online_resource=True)


def run(args: argparse.Namespace) -> None:
    apply_style()
    tier1_image_df = report_tier1()
    report_tier2()
    report_tier3(tier1_image_df)
    report_tier4()
    report_ocr_pipeline(tier1_image_df)
    report_cost(tier1_image_df)
    _write_captions()
