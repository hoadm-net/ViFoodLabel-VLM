"""Tier 4b -- Cohen's kappa + agreement report from two completed coding sheets.

Run `error-sample` first, have two people independently fill in the
`error_category` column of error_sample_coder_a.csv and _coder_b.csv, then
run this.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from vifoodlabel.config import SCORED_RESULTS_DIR

KEY_COLUMNS = ["image_id", "model_key", "condition", "field"]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--coder-a", default=str(SCORED_RESULTS_DIR / "error_sample_coder_a.csv"))
    parser.add_argument("--coder-b", default=str(SCORED_RESULTS_DIR / "error_sample_coder_b.csv"))


def run(args: argparse.Namespace) -> None:
    path_a, path_b = Path(args.coder_a), Path(args.coder_b)
    if not path_a.exists() or not path_b.exists():
        print(f"Missing coding sheet(s): {path_a} / {path_b}. Run 'main.py error-sample' first.")
        return

    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)

    merged = df_a.merge(df_b, on=KEY_COLUMNS, suffixes=("_a", "_b"))
    if len(merged) != len(df_a) or len(merged) != len(df_b):
        print(f"Warning: row counts don't match after merging on {KEY_COLUMNS} "
              f"(a={len(df_a)}, b={len(df_b)}, merged={len(merged)}). "
              "Did a coder add/remove/reorder rows? Continuing with the intersection.")

    merged["error_category_a"] = merged["error_category_a"].fillna("").str.strip()
    merged["error_category_b"] = merged["error_category_b"].fillna("").str.strip()
    coded = merged[(merged["error_category_a"] != "") & (merged["error_category_b"] != "")]
    n_uncoded = len(merged) - len(coded)
    if n_uncoded:
        print(f"{n_uncoded}/{len(merged)} rows are not yet coded by both coders — excluded from kappa.")
    if coded.empty:
        print("No fully-coded rows yet.")
        return

    kappa = cohen_kappa_score(coded["error_category_a"], coded["error_category_b"])
    agree_rate = (coded["error_category_a"] == coded["error_category_b"]).mean()
    print(f"n coded (both): {len(coded)}")
    print(f"Raw agreement rate: {agree_rate:.3f}")
    print(f"Cohen's kappa: {kappa:.3f}")

    coded = coded.copy()
    coded["agree"] = coded["error_category_a"] == coded["error_category_b"]
    coded["final_category"] = coded["error_category_a"].where(coded["agree"], "NEEDS_ADJUDICATION")

    agreement_path = SCORED_RESULTS_DIR / "error_taxonomy_agreement.csv"
    coded.to_csv(agreement_path, index=False)
    print(f"\nWrote per-row agreement detail to {agreement_path}")

    agreed_only = coded[coded["agree"]]
    distribution = (
        agreed_only["final_category"].value_counts(normalize=True).rename("share")
        .to_frame().join(agreed_only["final_category"].value_counts().rename("count"))
        .reset_index(names="error_category")
        .sort_values("count", ascending=False)
    )
    distribution_path = SCORED_RESULTS_DIR / "error_taxonomy_distribution.csv"
    distribution.to_csv(distribution_path, index=False)
    print(f"Wrote preliminary category distribution (agreed rows only) to {distribution_path}")
    print(f"\n{len(coded) - len(agreed_only)} rows need manual adjudication (see NEEDS_ADJUDICATION rows in {agreement_path.name}).")
