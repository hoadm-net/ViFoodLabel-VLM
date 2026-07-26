"""Bootstrap confidence intervals and McNemar's pairwise significance testing."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
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
    return df.sort_values("p_value").reset_index(drop=True)
