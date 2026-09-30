"""House plotting style for every project figure (EDA, model evaluation, report).

Colour is used only where it carries meaning: the class pair below marks "no disease" versus
"disease". Sites are identified by position and labels, never by colour. The pair was checked
with a colour-vision-deficiency validator on the light surface: worst simulated separation
Delta E 16.5 (protanopia; target >= 8), normal-vision Delta E 25.2 (floor 15), both >= 3:1
contrast against the surface.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
INK = "#0b0b0b"  # titles and values
INK_SECONDARY = "#52514e"  # axis labels
INK_MUTED = "#898781"  # ticks, annotations
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

CLASS_COLORS = {0: "#3b6ea8", 1: "#c8553d"}  # steel blue = no disease, terracotta = disease
CLASS_LABELS = {0: "No disease", 1: "Disease"}
NEUTRAL = "#5f6b73"  # single-series marks with no class meaning
DIVERGING_MID = "#f0efec"

# Sequential (magnitude, e.g. % missing): one violet hue, light to dark.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "heartrisk_sequential", ["#f4f1f8", "#c7b8e0", "#8467b8", "#4a2f80", "#2a1a4d"]
)
# Diverging (signed, e.g. correlation): class poles around a neutral grey midpoint,
# so "moves with disease" reads warm and "moves against disease" reads cool.
DIVERGING = LinearSegmentedColormap.from_list(
    "heartrisk_diverging", [CLASS_COLORS[0], DIVERGING_MID, CLASS_COLORS[1]]
)

FIG_DPI = 150
BASE_FONT_PT = 11
FIGURES_DIR = Path("reports/figures")


def apply_style() -> None:
    """Set the project-wide matplotlib defaults (fonts >= 11 pt, recessive hairline chrome)."""
    plt.rcParams.update(
        {
            "font.size": BASE_FONT_PT,
            "axes.titlesize": BASE_FONT_PT + 2,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": BASE_FONT_PT,
            "xtick.labelsize": BASE_FONT_PT,
            "ytick.labelsize": BASE_FONT_PT,
            "legend.fontsize": BASE_FONT_PT,
            "legend.title_fontsize": BASE_FONT_PT,
            "figure.titlesize": BASE_FONT_PT + 3,
            "figure.titleweight": "bold",
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "lines.linewidth": 2,
            "patch.linewidth": 0,
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
        }
    )


def save_fig(fig: Figure, name: str, out_dir: Path = FIGURES_DIR) -> Path:
    """Save `fig` as `<out_dir>/<name>.png` at 150 dpi; returns the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    # No "Software" metadata, so re-rendering the same figure gives the same bytes.
    fig.savefig(path, dpi=FIG_DPI, facecolor=SURFACE, metadata={"Software": None})
    return path
