"""Combine whatever tiers have been run into summary tables + the paper's
figures (results/figures/*.png) -- see plotting.py for the shared style.

Tier 1 (results/scored/tier1_*.csv) is required for the fullest report;
Tier 2/3/4 and cost outputs are used if present, skipped with a message
otherwise. Safe to re-run at any time as more scored data appears.
"""

from __future__ import annotations

import argparse

import pandas as pd

from vifoodlabel.config import FIGURES_DIR, SCORED_RESULTS_DIR
from vifoodlabel.cost import LEDGER_PATH
from vifoodlabel.metrics import model_field_summary, model_summary
from vifoodlabel.plotting import (
    plot_cost_effectiveness,
    plot_degradation_curve,
    plot_error_taxonomy,
    plot_field_heatmap,
    plot_leaderboard,
    plot_nutrition_breakdown,
    plot_prompt_sensitivity,
    setup_style,
)
from vifoodlabel.prompts import CANONICAL_CONDITION
from vifoodlabel.stats import bootstrap_ci_table, mcnemar_pairwise


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass


def _load(name: str) -> pd.DataFrame | None:
    path = SCORED_RESULTS_DIR / name
    if not path.exists():
        print(f"  (skipping, not found: {path})")
        return None
    return pd.read_csv(path)


def _save_figure(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / name
    fig.savefig(out_path)
    print(f"Saved {out_path}")


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

    _save_figure(plot_leaderboard(image_df, ci_f1), "tier1_leaderboard.png")
    _save_figure(plot_field_heatmap(field_summary), "tier1_field_heatmap.png")

    # model_field_summary() only aggregates precision/recall/f1; pairing_accuracy
    # and value_accuracy live only on the raw nutrition rows of field_df.
    nutrition_rows = field_df[field_df["field"] == "nutrition"]
    if not nutrition_rows.empty:
        nutrition_summary = nutrition_rows.groupby("model_key").agg(
            f1=("f1", "mean"), pairing_accuracy=("pairing_accuracy", "mean"), value_accuracy=("value_accuracy", "mean"),
        ).reset_index()
        _save_figure(plot_nutrition_breakdown(nutrition_summary), "tier1_nutrition_breakdown.png")

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
        _save_figure(plot_prompt_sensitivity(pivot), "tier2_prompt_sensitivity.png")


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
        .agg(mean_macro_field_f1=("macro_field_f1", "mean"), n_images=("image_id", "nunique"))
        .reset_index()
        .sort_values(["kind", "model_key", "severity"])
    )
    print(curve.to_string(index=False))
    curve.to_csv(SCORED_RESULTS_DIR / "tier3_degradation_curve.csv", index=False)

    _save_figure(plot_degradation_curve(curve), "tier3_degradation_curve.png")


def report_tier4() -> None:
    print("\n=== Tier 4: error taxonomy ===")
    df = _load("error_taxonomy.csv")
    if df is None:
        print("Run 'main.py error-sample' first.")
        return
    shares = pd.crosstab(df["model_key"], df["error_category"], normalize="index")
    _save_figure(plot_error_taxonomy(shares), "tier4_error_taxonomy.png")


def report_cost(tier1_image_df: pd.DataFrame | None) -> None:
    print("\n=== Cost vs. accuracy ===")
    if tier1_image_df is None or not LEDGER_PATH.exists():
        print(f"  (skipping, need Tier 1 scores and {LEDGER_PATH})")
        return
    ledger = pd.read_csv(LEDGER_PATH)
    cost_by_model = ledger.groupby("model_key")["cost_usd"].sum()
    macro_f1 = tier1_image_df.groupby("model_key")["macro_field_f1"].mean()
    print(cost_by_model.reindex(macro_f1.index).round(2).to_string())
    _save_figure(plot_cost_effectiveness(cost_by_model, macro_f1), "cost_effectiveness.png")


def run(args: argparse.Namespace) -> None:
    setup_style()
    tier1_image_df = report_tier1()
    report_tier2()
    report_tier3(tier1_image_df)
    report_tier4()
    report_cost(tier1_image_df)
