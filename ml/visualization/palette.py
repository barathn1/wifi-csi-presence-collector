"""Shared chart palette (from the dataviz skill's validated reference palette, light mode).
One place to swap colors; every visualization script imports from here rather than picking its own.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_FIG_DIR = REPO_ROOT / "visualizations" / "static"
INTERACTIVE_DIR = REPO_ROOT / "visualizations" / "interactive"

# Fixed categorical order -- never cycled/reassigned per chart. Chosen so the same class always gets
# the same color across every figure in ml/visualization/.
CATEGORICAL = {
    "authorized": "#2a78d6",     # slot 1, blue
    "unauthorized": "#eb6834",   # slot 2, orange
    "none": "#1baf7a",           # slot 3, aqua
    "anjali": "#2a78d6",         # reuse slot 1/2 for the 2-identity plots (never both used alongside
    "barath": "#eb6834",         # the 3-class ones in the same figure, so no collision)
}

# Sequential (single hue, light->dark) for unsigned magnitude heatmaps.
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

# Diverging (two hues + neutral gray midpoint) for signed differences (e.g. Cohen's d).
DIVERGING = {"neg": "#e34948", "mid": "#f0efec", "pos": "#2a78d6"}  # red <-> blue

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
