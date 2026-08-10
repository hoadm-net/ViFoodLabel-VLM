"""Chart builders for the manuscript's figures. Style/identity constants
live in figstyle.py and are imported here, never redefined -- see that
module's docstring for the submission constraints this jointly satisfies.

Five main-text figures (Fig1-Fig5, one per tier plus the nutrition
decomposition that carries the RQ) and one supplementary figure
(cost-effectiveness). Each `plot_fig*` function takes an already-aggregated
DataFrame/Series (built by report.py from the scored CSVs) and returns a
Figure; report.py owns computing the inputs and saving the output.
No in-figure titles -- captions belong in the manuscript text (report.py
writes them to results/figures/captions.md). Panel letters (a/b/c) are not
titles and are added directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vifoodlabel.figstyle import (
    CATEGORY_COLOR,
    CATEGORY_HATCH,
    MODEL_COLOR,
    MODEL_LINESTYLE,
    MODEL_MARKER,
    MODEL_ORDER,
    WIDTH_DOUBLE,
    WIDTH_SINGLE,
    model_color,
)
from vifoodlabel.schema import LIST_FIELDS, NUTRITION_FIELD, SCALAR_FIELDS

JSON_VALIDITY_FLAG_THRESHOLD = 0.9  # below this, a point gets circled on Fig4


def _ordered(models: pd.Index | list[str]) -> list[str]:
    present = set(models)
    return [m for m in MODEL_ORDER if m in present]


def _panel_label(ax, letter: str) -> None:
    ax.text(-0.12, 1.02, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def _text_color(rgba) -> str:
    r, g, b = rgba[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if lum > 0.55 else "white"


def plot_fig1_heatmap(field_summary: pd.DataFrame, macro_f1: pd.Series):
    """Fig 1: per-field F1 heatmap, one extra "macro" column."""
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    fields = SCALAR_FIELDS + LIST_FIELDS + [NUTRITION_FIELD]
    order = _ordered(field_summary["model_key"].unique())
    pivot = field_summary.pivot(index="model_key", columns="field", values="f1").reindex(index=order, columns=fields)
    macro = macro_f1.reindex(order)
    cmap = colormaps["viridis"]

    fig, (ax_fields, ax_macro) = plt.subplots(
        1, 2, figsize=(WIDTH_DOUBLE, 0.35 * len(order) + 1.3),
        gridspec_kw={"width_ratios": [len(fields), 1], "wspace": 0.1},
    )
    im = ax_fields.imshow(pivot.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax_fields.set_xticks(range(len(fields)))
    ax_fields.set_xticklabels(fields, rotation=30, ha="right")
    ax_fields.set_yticks(range(len(order)))
    ax_fields.set_yticklabels(order)
    for i in range(len(order)):
        for j in range(len(fields)):
            val = pivot.to_numpy()[i, j]
            if np.isnan(val):
                continue
            ax_fields.text(j, i, f"{val:.2f}", ha="center", va="center", color=_text_color(cmap(val)), fontsize=8)

    macro_vals = macro.to_numpy(dtype=float)
    ax_macro.imshow(macro_vals.reshape(-1, 1), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax_macro.set_xticks([0])
    ax_macro.set_xticklabels(["macro"], rotation=30, ha="right")
    ax_macro.set_yticks([])
    for i, val in enumerate(macro_vals):
        ax_macro.text(0, i, f"{val:.2f}", ha="center", va="center", color=_text_color(cmap(val)), fontsize=8)

    fig.colorbar(im, ax=[ax_fields, ax_macro], label="F1", shrink=0.8)
    return fig


def plot_fig2_nutrition(nutrition_df: pd.DataFrame):
    """Fig 2: nutrition decomposition, 3 panels (a name F1, b pairing
    accuracy on a truncated scale, c value accuracy). Model order is
    data-driven (name F1 descending), identical across all three panels."""
    import matplotlib.pyplot as plt

    order = nutrition_df.sort_values("f1", ascending=False)["model_key"].tolist()
    df = nutrition_df.set_index("model_key").loc[order]
    x = np.arange(len(order))
    colors = [MODEL_COLOR.get(m, "#333333") for m in order]

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(WIDTH_DOUBLE, 2.4))

    ax_a.bar(x, df["f1"].to_numpy(), color=colors, edgecolor="black", linewidth=0.5)
    ax_a.set_ylabel("Name F1")
    ax_a.set_ylim(0, 1)
    _panel_label(ax_a, "a")

    for i, m in enumerate(order):
        ax_b.plot(i, df.loc[m, "pairing_accuracy"], marker=MODEL_MARKER.get(m, "o"),
                   color=MODEL_COLOR.get(m, "#333333"), markeredgecolor="black", markeredgewidth=0.4, markersize=5)
    ax_b.set_ylabel("Pairing accuracy")
    ax_b.set_ylim(0.95, 1.00)
    _panel_label(ax_b, "b")

    ax_c.bar(x, df["value_accuracy"].to_numpy(), color=colors, edgecolor="black", linewidth=0.5)
    ax_c.set_ylabel("Value accuracy")
    ax_c.set_ylim(0, 1)
    _panel_label(ax_c, "c")

    for ax in (ax_a, ax_b, ax_c):
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=45, ha="right")
    fig.tight_layout()
    return fig


def plot_fig3_prompt_sensitivity(effects: pd.DataFrame):
    """Fig 3: shot effect (a) and instruction-language effect (b) vs. the
    vi_zero baseline, per model. Both panels share one horizontal scale so
    bar lengths are visually comparable across panels."""
    import matplotlib.pyplot as plt

    order = _ordered(effects.index)
    df = effects.loc[order]
    y = np.arange(len(order))
    colors = [MODEL_COLOR.get(m, "#333333") for m in order]
    xlim = (-0.09, 0.09)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(WIDTH_DOUBLE, 2.6), sharey=True)

    ax_a.barh(y, df["shot_effect"].to_numpy(), color=colors, edgecolor="black", linewidth=0.5)
    ax_a.axvline(0, color="black", linewidth=0.8)
    ax_a.set_xlabel("Δ macro field F1 (vi_one − vi_zero)")
    ax_a.set_xlim(*xlim)
    _panel_label(ax_a, "a")

    ax_b.barh(y, df["language_effect"].to_numpy(), color=colors, edgecolor="black", linewidth=0.5)
    ax_b.axvline(0, color="black", linewidth=0.8)
    ax_b.set_xlabel("Δ macro field F1 (en_zero − vi_zero)")
    ax_b.set_xlim(*xlim)
    _panel_label(ax_b, "b")

    ax_a.set_yticks(y)
    ax_a.set_yticklabels(order)
    fig.tight_layout()
    return fig


def plot_fig4_degradation(curve: pd.DataFrame):
    """Fig 4: macro F1 vs. severity, one panel per corruption kind (a blur,
    b glare, c rotation). Points whose json_validity_rate falls below
    JSON_VALIDITY_FLAG_THRESHOLD are circled -- a large drop there reflects
    parse failures, not necessarily weaker extraction, and shouldn't be
    read as a clean accuracy point without that caveat."""
    import matplotlib.pyplot as plt

    kinds = sorted(curve["kind"].unique())  # blur, glare, rotation
    letters = "abcdefgh"
    fig, axes = plt.subplots(1, len(kinds), figsize=(WIDTH_DOUBLE, 2.6), sharey=True)
    axes = [axes] if len(kinds) == 1 else list(axes)

    for ax, kind, letter in zip(axes, kinds, letters):
        sub = curve[curve["kind"] == kind]
        for model_key in _ordered(sub["model_key"].unique()):
            g = sub[sub["model_key"] == model_key].sort_values("severity")
            ax.plot(
                g["severity"], g["mean_macro_field_f1"],
                marker=MODEL_MARKER.get(model_key, "o"), linestyle=MODEL_LINESTYLE.get(model_key, "-"),
                color=model_color(model_key), label=model_key,
            )
            if "json_valid_rate" in g:
                flagged = g[g["json_valid_rate"] < JSON_VALIDITY_FLAG_THRESHOLD]
                if not flagged.empty:
                    ax.scatter(
                        flagged["severity"], flagged["mean_macro_field_f1"],
                        s=140, facecolors="none", edgecolors="black", linewidths=1.0, zorder=5,
                    )
        ax.set_xlabel(f"{kind} severity")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_ylim(0, 0.9)
        _panel_label(ax, letter)

    axes[0].set_ylabel("macro field F1")
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def plot_fig5_error_taxonomy(shares: pd.DataFrame):
    """Fig 5: 100%-stacked bar, share of each model's errors per category.
    `shares` columns already consolidated to the 6 named categories plus
    "other" (see report.py) -- full 10-category detail goes in the SI."""
    import matplotlib.pyplot as plt

    order = _ordered(shares.index)
    categories = [c for c in CATEGORY_HATCH if c in shares.columns]
    df = shares.loc[order, categories]
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE, 3.2))
    bottom = np.zeros(len(df))
    for cat in categories:
        vals = df[cat].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=CATEGORY_COLOR.get(cat, "#BFBFBF"),
               edgecolor="black", linewidth=0.4, hatch=CATEGORY_HATCH[cat], label=cat)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("Share of model's errors")
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def plot_cost_effectiveness(cost_by_model: pd.Series, macro_f1: pd.Series):
    """Supplementary: cumulative cost (x) vs. Tier 1 macro F1 (y)."""
    import matplotlib.pyplot as plt

    order = _ordered(macro_f1.index)
    fig, ax = plt.subplots(figsize=(WIDTH_SINGLE, WIDTH_SINGLE))
    for model_key in order:
        cost = float(cost_by_model.get(model_key, 0.0))
        f1 = float(macro_f1[model_key])
        ax.scatter(cost, f1, marker=MODEL_MARKER.get(model_key, "o"), color=model_color(model_key),
                   s=60, edgecolor="black", linewidth=0.6, zorder=3)
        ax.annotate(model_key, (cost, f1), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("Cumulative cost across all four tiers (USD)")
    ax.set_ylabel("Macro field-level F1 (Tier 1)")
    fig.tight_layout()
    return fig
