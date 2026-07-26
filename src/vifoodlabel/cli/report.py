"""Combine whatever tiers have been run into summary tables + the degradation curve figure.

Tier 1 (results/scored/tier1_*.csv) is required for the fullest report;
Tier 2/3 outputs are used if present, skipped with a message otherwise. Safe
to re-run at any time as more scored data appears.
"""

from __future__ import annotations

import argparse

import pandas as pd

from vifoodlabel.config import FIGURES_DIR, SCORED_RESULTS_DIR
from vifoodlabel.metrics import model_field_summary, model_summary
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

    try:
        import matplotlib.pyplot as plt

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        kinds = sorted(curve["kind"].unique())
        fig, axes = plt.subplots(1, len(kinds), figsize=(5 * len(kinds), 4), sharey=True)
        axes = [axes] if len(kinds) == 1 else axes
        for ax, kind in zip(axes, kinds):
            sub = curve[curve["kind"] == kind]
            for model_key, g in sub.groupby("model_key"):
                g = g.sort_values("severity")
                ax.plot(g["severity"], g["mean_macro_field_f1"], marker="o", label=model_key)
            ax.set_title(kind)
            ax.set_xlabel("severity (0 = clean)")
        axes[0].set_ylabel("mean macro field-level F1")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        out_path = FIGURES_DIR / "degradation_curve.png"
        fig.savefig(out_path, dpi=150)
        print(f"\nSaved degradation curve figure to {out_path}")
    except ImportError:
        print("matplotlib not available — skipped figure generation.")


def run(args: argparse.Namespace) -> None:
    tier1_image_df = report_tier1()
    report_tier2()
    report_tier3(tier1_image_df)
