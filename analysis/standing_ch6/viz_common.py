"""Shared plotting constants: categorical colors per identity, sequential ramp for heatmaps.

Palette values come from the project's dataviz skill reference (references/palette.md) -- categorical
hues assigned in fixed order, one hue light->dark for magnitude, never a rainbow colormap.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")

# Categorical: fixed hue per identity group, never reassigned/cycled across plots.
IDENTITY_COLOR = {
    "anjali": "#2a78d6",     # slot 1 blue
    "barath": "#eb6834",     # slot 2 orange
    "stranger": "#1baf7a",   # slot 3 aqua (pooled strangers)
    "empty_room": "#898781",  # muted ink -- reads as "background/other", not a person
}
# Per-stranger tints (context only, not the primary identity target) -- aqua family, light->dark.
STRANGER_TINTS = ["#1baf7a", "#0d8a5c", "#5fcf9e", "#0a6e49", "#8fdcbb"]

DAY_COLOR = {  # ordinal ramp, blue steps 300/450/600 (light->dark = earlier->later day)
    "2026-09-21": "#6da7ec",
    "2026-09-22": "#2a78d6",
    "2026-09-24": "#184f95",
}

# Sequential single-hue ramp (blue, light->dark) for amplitude heatmaps -- no rainbow/jet/viridis.
_BLUE_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
BLUE_SEQUENTIAL = LinearSegmentedColormap.from_list("blue_sequential", _BLUE_STEPS)

GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
MUTED_TEXT = "#898781"
PRIMARY_TEXT = "#0b0b0b"
SECONDARY_TEXT = "#52514e"


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=SECONDARY_TEXT)
    ax.xaxis.label.set_color(PRIMARY_TEXT)
    ax.yaxis.label.set_color(PRIMARY_TEXT)
    ax.title.set_color(PRIMARY_TEXT)
    ax.grid(True, color=GRIDLINE, linewidth=0.6)
    ax.set_axisbelow(True)


def identity_color(identity: str) -> str:
    if identity.startswith("stranger:"):
        return IDENTITY_COLOR["stranger"]
    return IDENTITY_COLOR.get(identity, "#000000")


FIGURES_DIR = None  # set by caller scripts via analysis.standing_ch6 paths
