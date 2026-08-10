"""Shared style, model identity, and export conventions for every figure in
the manuscript -- imported by plotting.py, never duplicated per-figure (two
figures independently setting font sizes is how they end up mismatched).

Springer *Discover Artificial Intelligence* submission constraints this
encodes: sans-serif 8-12pt with minimal within-figure variance, embedded
TrueType fonts, no in-figure titles (captions live in the manuscript text),
panels labelled a/b/c, colour never the only channel encoding information
(pattern/marker/linestyle required alongside it for accessibility and
grayscale printing), Fig{N}.eps naming, 600 DPI for combination art if
rasterized (vector export sidesteps the DPI question entirely).
"""

from __future__ import annotations

MM = 1 / 25.4  # millimeters to inches

WIDTH_SINGLE = 84 * MM   # one-panel figures
WIDTH_DOUBLE = 174 * MM  # multi-panel figures

# Fixed per-model identity, frozen rather than left to matplotlib's default
# cycle -- the same model must be the same color/marker/linestyle in every
# figure regardless of that figure's own model ordering.
MODEL_ORDER = [
    "claude-sonnet-5", "gpt-5.4", "mimo-v2.5", "qwen3-vl-235b",
    "grok-4.20", "glm-4.6v", "vintern-3b",
]

# Okabe-Ito palette (colorblind-safe).
MODEL_COLOR: dict[str, str] = {
    "claude-sonnet-5": "#56B4E9",
    "gpt-5.4": "#E69F00",
    "mimo-v2.5": "#D55E00",
    "qwen3-vl-235b": "#F0E442",
    "grok-4.20": "#009E73",
    "glm-4.6v": "#0072B2",
    "vintern-3b": "#CC79A7",
}

# Second information channel, independent of color -- distinguishes all 7
# models on a grayscale printout.
MODEL_MARKER: dict[str, str] = {
    "claude-sonnet-5": "o", "gpt-5.4": "s", "mimo-v2.5": "^",
    "qwen3-vl-235b": "D", "grok-4.20": "v", "glm-4.6v": "P",
    "vintern-3b": "X",
}
MODEL_LINESTYLE: dict[str, object] = {
    "claude-sonnet-5": "-", "gpt-5.4": "--", "mimo-v2.5": "-.",
    "qwen3-vl-235b": ":", "grok-4.20": (0, (3, 1, 1, 1)),
    "glm-4.6v": (0, (5, 2)), "vintern-3b": (0, (1, 1)),
}

# Error-taxonomy categories: hatch is the color-independent channel for the
# Tier 4 stacked bar. "" (no hatch) is reserved for the largest category,
# which is unambiguous from its size alone. Colors are fill only -- hatch
# is what actually has to survive grayscale printing.
CATEGORY_HATCH: dict[str, str] = {
    "wrong_value": "",
    "hallucination": "///",
    "missing_field": "...",
    "missing_additive": "\\\\\\",
    "generation_loop": "xxx",
    "output_truncated": "|||",
    "other": "+++",
}
CATEGORY_COLOR: dict[str, str] = {
    "wrong_value": "#0072B2",
    "hallucination": "#E69F00",
    "missing_field": "#009E73",
    "missing_additive": "#F0E442",
    "generation_loop": "#CC79A7",
    "output_truncated": "#56B4E9",
    "other": "#999999",
}

_DEFAULT_COLOR = "#333333"


def model_color(model_key: str) -> str:
    return MODEL_COLOR.get(model_key, _DEFAULT_COLOR)


def apply_style() -> None:
    """Call once before drawing any figure."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

        # Every size in-figure sits within [8, 10]pt -- a 2pt max spread,
        # inside the journal's "minimal variance" requirement.
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,

        # Embed TrueType fonts in PDF/EPS output -- without this the fonts
        # aren't embedded and the file gets bounced by the publisher.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": "#D9D9D9",

        "figure.dpi": 150,   # screen preview only
        "savefig.dpi": 600,  # combination-art threshold if rasterized
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def check_fonts() -> str:
    """Returns the resolved sans-serif font file. If the path contains
    DejaVuSans, Arial/Helvetica isn't actually installed and every figure
    is silently falling back to a font the journal doesn't recommend."""
    from matplotlib.font_manager import FontProperties, findfont

    # A bare string is parsed as a fontconfig pattern by FontProperties and
    # chokes on the hyphen in "sans-serif" -- pass it as a single-item list.
    return findfont(FontProperties(family=["sans-serif"]))


def _export(fig, stem: str, vector_dpi: float | None = None) -> list[str]:
    """`vector_dpi` overrides the PDF/EPS render resolution -- irrelevant
    for genuinely vector content (text stays embedded glyphs regardless,
    per pdf.fonttype/ps.fonttype=42), but matters for any rasterized
    artist (imshow, hatched fills): those get resampled to figure-size x
    dpi pixels even inside a "vector" file, so a small array (e.g. a 7x9
    heatmap) at the default 600 dpi embeds a needlessly huge bitmap. Leave
    unset for figures with no such artist; TIFF always renders at the full
    600 dpi regardless, since it's meant to BE a raster at that resolution.
    """
    paths = [f"{stem}.pdf", f"{stem}.eps", f"{stem}.tiff"]
    if vector_dpi is not None:
        fig.savefig(paths[0], dpi=vector_dpi)
        fig.savefig(paths[1], dpi=vector_dpi)
    else:
        fig.savefig(paths[0])
        fig.savefig(paths[1])
    fig.savefig(paths[2], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    return paths


def save_figure(fig, number: int, outdir=None, vector_dpi: float | None = None) -> list[str]:
    """Exports Fig{number}.{pdf,eps,tiff} per the journal's main-text
    naming convention. PDF is the working/inspection copy, EPS is the
    preferred vector format, TIFF-600dpi is the raster fallback if EPS
    mangles a hatch or font. Returns the written paths.

    EPS has no alpha channel -- any `alpha=` in a plotting function forces
    that element to rasterize, silently breaking the file's vector-ness.
    Use pre-blended colors instead.
    """
    from vifoodlabel.config import FIGURES_DIR

    outdir = outdir or FIGURES_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    return _export(fig, f"{outdir}/Fig{number}", vector_dpi=vector_dpi)


def save_online_resource(fig, number: int, outdir=None, vector_dpi: float | None = None) -> list[str]:
    """Same export, Supplementary-Information naming: numbered separately
    from the main-text Fig{N} sequence, cited by file name in the text
    (e.g. "Online Resource 1")."""
    from vifoodlabel.config import FIGURES_DIR

    outdir = outdir or FIGURES_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    return _export(fig, f"{outdir}/OnlineResource{number}", vector_dpi=vector_dpi)
