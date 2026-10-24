"""Bootstrap confidence intervals and McNemar's pairwise significance testing."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

DEFAULT_N_RESAMPLES = 10_000
DEFAULT_CI = 0.95
DEFAULT_SEED = 42


def bootstrap_ci(
    values: np.ndarray | list[float],
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci: float = DEFAULT_CI,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Percentile-method bootstrap CI over image-level resampling."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resample_means = arr[idx].mean(axis=1)
    alpha = 1 - ci
    lo, hi = np.quantile(resample_means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(arr.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": n}


def bootstrap_ci_table(
    image_df: pd.DataFrame,
    value_col: str,
    group_cols: tuple[str, ...] = ("model_key", "condition"),
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Bootstrap CI of `value_col`, one row per group in `group_cols`."""
    rows = []
    for keys, g in image_df.groupby(list(group_cols)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        result = bootstrap_ci(g[value_col].to_numpy(), n_resamples=n_resamples, seed=seed)
        row = dict(zip(group_cols, keys))
        row.update(result)
        row["metric"] = value_col
        rows.append(row)
    return pd.DataFrame(rows)


def mcnemar_pairwise(
    binary_df: pd.DataFrame,
    id_col: str = "image_id",
    group_col: str = "model_key",
    value_col: str = "correct",
    holm_alpha: float = 0.05,
) -> pd.DataFrame:
    """Pairwise McNemar's test across every pair of groups (models), Holm-Bonferroni corrected.

    `binary_df` must have one row per (id_col, group_col) with a 0/1 `value_col`
    (e.g. "did the model get this exact field right for this image").
    """
    pivot = binary_df.pivot(index=id_col, columns=group_col, values=value_col)
    groups = list(pivot.columns)
    results = []
    for a, b in itertools.combinations(groups, 2):
        sub = pivot[[a, b]].dropna()
        a_only = int(((sub[a] == 1) & (sub[b] == 0)).sum())
        b_only = int(((sub[a] == 0) & (sub[b] == 1)).sum())
        both_correct = int(((sub[a] == 1) & (sub[b] == 1)).sum())
        both_wrong = int(((sub[a] == 0) & (sub[b] == 0)).sum())
        table = [[both_correct, a_only], [b_only, both_wrong]]
        res = mcnemar(table, exact=(a_only + b_only < 25), correction=True)
        results.append({
            "group_a": a, "group_b": b, "n": len(sub),
            "a_only_correct": a_only, "b_only_correct": b_only,
            "statistic": float(res.statistic), "p_value": float(res.pvalue),
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df
    _, p_holm, _, _ = multipletests(df["p_value"], alpha=holm_alpha, method="holm")
    df["p_holm"] = p_holm
    df[f"significant_at_{holm_alpha}"] = df["p_holm"] < holm_alpha
    # Plain Bonferroni alongside, for the Holm-vs-Bonferroni comparison
    # (Holm is uniformly at least as powerful at the same FWER guarantee;
    # this column shows how much that matters on the actual data).
    _, p_bonf, _, _ = multipletests(df["p_value"], alpha=holm_alpha, method="bonferroni")
    df["p_bonferroni"] = p_bonf
    df[f"significant_bonferroni_at_{holm_alpha}"] = df["p_bonferroni"] < holm_alpha
    return df.sort_values("p_value").reset_index(drop=True)


def paired_tests_pairwise(
    image_df: pd.DataFrame,
    value_col: str,
    id_col: str = "image_id",
    group_col: str = "model_key",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Pairwise paired tests on a continuous per-image score (e.g. macro
    field F1), every pair of groups, Holm-corrected within each test family.

    Two tests side by side: the paired t-test (parametric, on the mean
    difference) and the Wilcoxon signed-rank test (non-parametric). The
    per-image macro score is bounded in [0, 1], often piles up at the ends,
    and its paired differences are far from normal -- reporting both shows
    whether the conclusion depends on that assumption.
    """
    pivot = image_df.pivot(index=id_col, columns=group_col, values=value_col)
    rows = []
    for a, b in itertools.combinations(list(pivot.columns), 2):
        sub = pivot[[a, b]].dropna()
        diff = (sub[a] - sub[b]).to_numpy(dtype=float)
        t = scipy_stats.ttest_rel(sub[a], sub[b])
        nonzero = diff[diff != 0]
        w_p = float(scipy_stats.wilcoxon(nonzero).pvalue) if len(nonzero) > 0 else 1.0
        rows.append({
            "group_a": a, "group_b": b, "n": len(sub),
            "mean_diff": float(diff.mean()), "median_diff": float(np.median(diff)),
            "t_p_value": float(t.pvalue), "wilcoxon_p_value": w_p,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for test in ("t", "wilcoxon"):
        _, p_holm, _, _ = multipletests(df[f"{test}_p_value"], alpha=alpha, method="holm")
        df[f"{test}_p_holm"] = p_holm
        df[f"{test}_significant_at_{alpha}"] = p_holm < alpha
    return df


def aggregation_robustness(image_df: pd.DataFrame, value_col: str, group_col: str = "model_key", trim: float = 0.1) -> pd.DataFrame:
    """Per-group mean, median and `trim`-trimmed mean of a per-image score,
    with the group's rank under each -- does the model ranking depend on
    choosing the arithmetic mean as the aggregate?"""
    g = image_df.groupby(group_col)[value_col]
    df = pd.DataFrame({
        "mean": g.mean(),
        "median": g.median(),
        f"trimmed_mean_{int(trim * 100)}": g.apply(lambda v: float(scipy_stats.trim_mean(v, trim))),
        "std": g.std(),
        "n": g.size(),
    })
    for col in ["mean", "median", f"trimmed_mean_{int(trim * 100)}"]:
        df[f"rank_{col}"] = df[col].rank(ascending=False, method="min").astype(int)
    return df.sort_values("mean", ascending=False).reset_index()


def paired_bootstrap_diff(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci: float = DEFAULT_CI,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Bootstrap CI of mean(a - b) for paired per-image scores -- the
    per-condition change relative to a baseline on the same images."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    r = bootstrap_ci(diff, n_resamples=n_resamples, ci=ci, seed=seed)
    return {"mean_diff": r["mean"], "diff_ci_low": r["ci_low"], "diff_ci_high": r["ci_high"], "n": r["n"]}
