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

from vifoodlabel.config import FIGURES_DIR, SCORED_RESULTS_DIR
from vifoodlabel.cost import LEDGER_PATH
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
from vifoodlabel.stats import bootstrap_ci_table, mcnemar_pairwise

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
    else:
        print("  (not enough models/images yet for a pairwise comparison)")

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
        _save(plot_fig2_nutrition(nutrition_summary), 2)

    return image_df


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

    _save(plot_fig4_degradation(curve), 4)


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
    report_cost(tier1_image_df)
    _write_captions()
