"""Shared matplotlib style + chart builders for the paper's figures.

One consistent look across every figure: the same model -> color mapping
regardless of which chart or model ordering is in play, the same
fonts/DPI/margins, models within a chart always ordered by Tier 1 macro F1
(descending) so the "leaderboard order" reads the same way everywhere.
Colors are the Okabe-Ito palette (colorblind-safe, standard for scientific
figures).

Each `plot_*` function takes an already-aggregated DataFrame/Series (built
by report.py from the scored CSVs) and returns a Figure; report.py owns
deciding what to compute and where to save it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vifoodlabel.schema import LIST_FIELDS, NUTRITION_FIELD, SCALAR_FIELDS

MODEL_COLORS: dict[str, str] = {
    "gpt-5.4": "#E69F00",
    "claude-sonnet-5": "#56B4E9",
    "grok-4.20": "#009E73",
    "qwen3-vl-235b": "#F0E442",
    "glm-4.6v": "#0072B2",
    "mimo-v2.5": "#D55E00",
    "vintern-3b": "#CC79A7",
}

ERROR_CATEGORY_COLORS: dict[str, str] = {
    "wrong_value": "#0072B2",
    "hallucination": "#D55E00",
    "missing_field": "#E69F00",
    "missing_additive": "#F0E442",
    "generation_loop": "#CC79A7",
    "output_truncated": "#56B4E9",
    "pairing_error": "#009E73",
    "diacritics": "#999999",
    "wrong_unit": "#000000",
    "malformed_json": "#8C510A",
}

_DEFAULT_COLOR = "#333333"


def setup_style() -> None:
    """Call once before drawing any figure."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.3,
        "legend.fontsize": 8,
        "legend.frameon": False,
    })


def model_order(image_df: pd.DataFrame) -> list[str]:
    """Canonical model ordering for every chart: Tier 1 macro F1, descending."""
    means = image_df.groupby("model_key")["macro_field_f1"].mean().sort_values(ascending=False)
    return list(means.index)


def _color(model_key: str) -> str:
    return MODEL_COLORS.get(model_key, _DEFAULT_COLOR)


def plot_leaderboard(image_df: pd.DataFrame, ci_df: pd.DataFrame):
    """Tier 1 headline: macro field F1 per model, with bootstrap 95% CI."""
    import matplotlib.pyplot as plt

    order = model_order(image_df)
    ci = ci_df.set_index("model_key").loc[order]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(order))
    means = ci["mean"].to_numpy()
    errs = np.vstack([means - ci["ci_low"].to_numpy(), ci["ci_high"].to_numpy() - means])
    ax.bar(x, means, yerr=errs, capsize=3, color=[_color(m) for m in order], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("Macro field-level F1 (lenient)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def plot_field_heatmap(field_summary: pd.DataFrame):
    """Model x field F1 heatmap (Tier 1)."""
    import matplotlib.pyplot as plt

    fields = SCALAR_FIELDS + LIST_FIELDS + [NUTRITION_FIELD]
    order = field_summary.groupby("model_key")["f1"].mean().sort_values(ascending=False).index.tolist()
    pivot = field_summary.pivot(index="model_key", columns="field", values="f1").reindex(index=order, columns=fields)

    fig, ax = plt.subplots(figsize=(7, 0.55 * len(order) + 1.5))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(fields)))
    ax.set_xticklabels(fields, rotation=30, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    for i in range(len(order)):
        for j in range(len(fields)):
            val = pivot.to_numpy()[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color="white" if val < 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="F1", shrink=0.8)
    fig.tight_layout()
    return fig


def plot_nutrition_breakdown(nutrition_df: pd.DataFrame):
    """3-panel small multiples: name F1, pairing accuracy, value accuracy."""
    import matplotlib.pyplot as plt

    order = nutrition_df.sort_values("f1", ascending=False)["model_key"].tolist()
    df = nutrition_df.set_index("model_key").loc[order]

    metrics = [("f1", "Name F1"), ("pairing_accuracy", "Pairing accuracy"), ("value_accuracy", "Value accuracy")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(order))
    for ax, (col, label) in zip(axes, metrics):
        ax.bar(x, df[col].to_numpy(), color=[_color(m) for m in order], edgecolor="black", linewidth=0.5)
        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=45, ha="right")
        ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def plot_prompt_sensitivity(effects: pd.DataFrame):
    """Shot effect and language effect vs. the vi_zero baseline, per model.

    `effects`: index=model_key, columns include vi_zero, shot_effect,
    language_effect (see report.py for how these are derived).
    """
    import matplotlib.pyplot as plt

    order = effects.sort_values("vi_zero", ascending=False).index.tolist()
    df = effects.loc[order]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    y = np.arange(len(order))
    panels = [("shot_effect", "Δ macro field F1 (vi_one − vi_zero)"),
              ("language_effect", "Δ macro field F1 (en_zero − vi_zero)")]
    for ax, (col, xlabel) in zip(axes, panels):
        ax.barh(y, df[col].to_numpy(), color=[_color(m) for m in order], edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(xlabel)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order)
    fig.tight_layout()
    return fig


def plot_degradation_curve(curve: pd.DataFrame):
    """Tier 3: macro F1 vs. severity, one subplot per corruption kind."""
    import matplotlib.pyplot as plt

    kinds = sorted(curve["kind"].unique())
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.5 * len(kinds), 4), sharey=True)
    axes = [axes] if len(kinds) == 1 else list(axes)
    for ax, kind in zip(axes, kinds):
        sub = curve[curve["kind"] == kind]
        for model_key, g in sub.groupby("model_key"):
            g = g.sort_values("severity")
            ax.plot(g["severity"], g["mean_macro_field_f1"], marker="o",
                     color=_color(model_key), label=model_key)
        ax.set_xlabel(f"{kind} severity (0 = clean)")
    axes[0].set_ylabel("mean macro field-level F1")
    axes[0].set_ylim(0, 1)
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def plot_error_taxonomy(shares: pd.DataFrame):
    """100% stacked bar: share of each model's errors per category."""
    import matplotlib.pyplot as plt

    categories = [c for c in ERROR_CATEGORY_COLORS if c in shares.columns]
    df = shares[categories].sort_index()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottom = np.zeros(len(df))
    x = np.arange(len(df))
    for cat in categories:
        vals = df[cat].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=ERROR_CATEGORY_COLORS[cat], label=cat, edgecolor="white", linewidth=0.3)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=30, ha="right")
    ax.set_ylabel("Share of model's errors")
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def plot_cost_effectiveness(cost_by_model: pd.Series, macro_f1: pd.Series):
    """Cost (x) vs. macro field F1 (y), one point per model."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for model_key in macro_f1.index:
        cost = float(cost_by_model.get(model_key, 0.0))
        f1 = float(macro_f1[model_key])
        ax.scatter(cost, f1, color=_color(model_key), s=80, edgecolor="black", linewidth=0.6, zorder=3)
        ax.annotate(model_key, (cost, f1), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("Total cost so far (USD, all tiers)")
    ax.set_ylabel("Macro field-level F1 (Tier 1)")
    fig.tight_layout()
    return fig
