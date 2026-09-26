"""Generates ml/reports/wifi_csi_pipeline.pptx -- summarizes the current branch's CSI
preprocessing pipeline and the transformer (BranchEncoder / SignatureModel) architecture.

Source material: ml/reports/home_model_and_evm_architecture.md, home_model_explainer.md,
ml/models/common.py, ml/models/signature_evm.py, ml/data_pipeline/*.

Run: .venv-ml/Scripts/python.exe ml/reports/build_pipeline_deck.py
"""
from __future__ import annotations

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.ns import qn

# ---- palette (validated categorical set, light-surface steps) --------------------------------
INK = RGBColor(0x0B, 0x0B, 0x0B)
INK_SECONDARY = RGBColor(0x52, 0x51, 0x4E)
MUTED = RGBColor(0x89, 0x87, 0x81)
SURFACE = RGBColor(0xFC, 0xFC, 0xFB)
PAGE = RGBColor(0xF9, 0xF9, 0xF7)
BORDER = RGBColor(0xC3, 0xC2, 0xB7)

BLUE = RGBColor(0x2A, 0x78, 0xD6)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
AQUA = RGBColor(0x1B, 0xAF, 0x7A)
YELLOW = RGBColor(0xED, 0xA1, 0x00)
VIOLET = RGBColor(0x4A, 0x3A, 0xA7)
RED = RGBColor(0xE3, 0x49, 0x48)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
BLANK = prs.slide_layouts[6]


def add_slide():
    slide = prs.slides.add_slide(BLANK)
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = PAGE
    return slide


def add_rect(slide, x, y, w, h, fill=None, line=None, line_w=Pt(1)):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    try:
        shp.adjustments[0] = 0.06
    except Exception:
        pass
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = line_w
    shp.shadow.inherit = False
    return shp


def set_text(shape, text, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.MIDDLE, font="Segoe UI", wrap=True):
    tf = shape.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = Pt(6)
    tf.margin_right = Pt(6)
    tf.margin_top = Pt(4)
    tf.margin_bottom = Pt(4)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.bold = bold
        run.font.name = font
    return shape


def add_textbox(slide, x, y, w, h, text, size=14, color=INK, bold=False,
                 align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(x, y, w, h)
    return set_text(box, text, size=size, color=color, bold=bold, align=align, anchor=anchor)


def add_bullets(slide, x, y, w, h, items, size=15, color=INK, bullet_color=None,
                 line_spacing=1.15):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if isinstance(item, tuple):
            text, level = item
        else:
            text, level = item, 0
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = level
        p.line_spacing = line_spacing
        p.space_after = Pt(6)
        marker = "•  " if level == 0 else "–  "
        run = p.add_run()
        run.text = marker + text
        run.font.size = Pt(size - level * 1)
        run.font.color.rgb = color if level == 0 else INK_SECONDARY
        run.font.name = "Segoe UI"
    return box


def add_kicker_title(slide, kicker, title, accent=BLUE):
    add_rect(slide, Inches(0.55), Inches(0.35), Inches(0.09), Inches(0.85), fill=accent)
    add_textbox(slide, Inches(0.8), Inches(0.32), Inches(9), Inches(0.35), kicker.upper(),
                size=12, color=accent, bold=True)
    add_textbox(slide, Inches(0.8), Inches(0.6), Inches(11.7), Inches(0.65), title,
                size=26, color=INK, bold=True)


def add_footer(slide, n):
    add_textbox(slide, Inches(0.55), Inches(7.12), Inches(6), Inches(0.3),
                "WiFi-CSI Presence Collector — Pipeline & Model Architecture",
                size=9, color=MUTED)
    add_textbox(slide, Inches(12.3), Inches(7.12), Inches(0.5), Inches(0.3), str(n),
                size=9, color=MUTED, align=PP_ALIGN.RIGHT)


def add_arrow(slide, x1, y1, x2, y2, color=MUTED, width=Pt(1.5)):
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    conn.line.color.rgb = color
    conn.line.width = width
    line = conn.line._get_or_add_ln()
    arrow = line.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    line.append(arrow)
    return conn


def flow_step(slide, x, y, w, h, label, sub=None, fill=SURFACE, line=BORDER, label_color=INK):
    box = add_rect(slide, x, y, w, h, fill=fill, line=line, line_w=Pt(1.25))
    if sub:
        tf = box.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        for i, line in enumerate(label.split("\n")):
            p0 = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p0.alignment = PP_ALIGN.CENTER
            r0 = p0.add_run()
            r0.text = line
            r0.font.size = Pt(13)
            r0.font.bold = True
            r0.font.color.rgb = label_color
            r0.font.name = "Segoe UI"
        for line in sub.split("\n"):
            p1 = tf.add_paragraph()
            p1.alignment = PP_ALIGN.CENTER
            r1 = p1.add_run()
            r1.text = line
            r1.font.size = Pt(10)
            r1.font.color.rgb = INK_SECONDARY
            r1.font.name = "Segoe UI"
    else:
        set_text(box, label, size=13, color=label_color, bold=True, align=PP_ALIGN.CENTER)
    return box


# =================================================================================================
# Slide 1 -- Title
# =================================================================================================
s = add_slide()
add_rect(s, 0, 0, SLIDE_W, Inches(7.5), fill=RGBColor(0x11, 0x1E, 0x2E))
add_rect(s, Inches(0.9), Inches(2.55), Inches(0.12), Inches(1.7), fill=ORANGE)
add_textbox(s, Inches(1.2), Inches(2.5), Inches(11), Inches(0.4), "WIFI-CSI PRESENCE COLLECTOR",
            size=14, color=RGBColor(0x9E, 0xC5, 0xF4), bold=True)
add_textbox(s, Inches(1.15), Inches(2.9), Inches(11.5), Inches(1.4),
            "CSI Preprocessing Pipeline &\nTransformer Architecture", size=36, color=WHITE, bold=True)
add_textbox(s, Inches(1.2), Inches(4.35), Inches(10.5), Inches(0.9),
            "How raw WiFi Channel State Information becomes clip / window features, and how the\n"
            "signature transformer (BranchEncoder) consumes and manipulates that data.",
            size=15, color=RGBColor(0xC3, 0xC2, 0xB7))
add_textbox(s, Inches(1.2), Inches(6.6), Inches(8), Inches(0.4),
            "Current branch — Home Model (Random Forest) + Signature/EVM (Transformer)",
            size=11, color=MUTED)

# =================================================================================================
# Slide 2 -- Overview: two pipelines on this branch
# =================================================================================================
s = add_slide()
add_kicker_title(s, "Overview", "Two model families live on this branch", accent=BLUE)
add_textbox(s, Inches(0.8), Inches(1.25), Inches(11.7), Inches(0.5),
            "Same raw CSI source, two unrelated pipelines and feature representations.",
            size=13, color=INK_SECONDARY)

col_w = Inches(5.7)
# Home model card
x1 = Inches(0.8)
card1 = add_rect(s, x1, Inches(1.95), col_w, Inches(4.9), fill=SURFACE, line=BORDER)
add_rect(s, x1, Inches(1.95), col_w, Inches(0.55), fill=BLUE)
set_text(card1, "", size=1)
add_textbox(s, x1 + Inches(0.25), Inches(2.05), col_w - Inches(0.5), Inches(0.4),
            "HOME MODEL  (v2, deployed)", size=15, color=WHITE, bold=True)
add_bullets(s, x1 + Inches(0.3), Inches(2.7), col_w - Inches(0.6), Inches(3.9), [
    "Task: closed 2-class identity — anjali vs barath, always picks one",
    "Model: Random Forest, 300 trees, class_weight=\"balanced\"",
    "Features: 4 statistical moments (mean/std/skew/kurtosis) per subcarrier",
    "Windowing: 3 s clips, 1 s stride, 50% coverage gate",
    "Feature vector: 4 × 128 subcarriers = 512 numbers per clip",
    "No neural network anywhere in this path",
    "Same-day accuracy ~70–95%; cross-day ~55–70%",
], size=13)

# Signature/EVM card
x2 = Inches(6.85)
card2 = add_rect(s, x2, Inches(1.95), col_w, Inches(4.9), fill=SURFACE, line=BORDER)
add_rect(s, x2, Inches(1.95), col_w, Inches(0.55), fill=ORANGE)
add_textbox(s, x2 + Inches(0.25), Inches(2.05), col_w - Inches(0.5), Inches(0.4),
            "SIGNATURE + EVM  (experimental, open-set)", size=15, color=WHITE, bold=True)
add_bullets(s, x2 + Inches(0.3), Inches(2.7), col_w - Inches(0.6), Inches(3.9), [
    "Task: open-set accept/reject — can say “unknown”, plus identify",
    "Model: Transformer encoder (BranchEncoder) → 32-d signature",
    "+ Extreme Value Machine (Weibull tails) for the accept/reject decision",
    "Features: raw amplitude sequence — no hand-engineered moments",
    "Windowing: 200-packet windows, 100-packet stride (50% overlap)",
    "Trained with in-batch-negative contrastive loss (WhoFi-style)",
    "Mean AUROC ≈ 0.45 across held-out strangers — experimental, not deployed",
], size=13)

add_footer(s, 2)

# =================================================================================================
# Slide 3 -- End-to-end flowchart
# =================================================================================================
s = add_slide()
add_kicker_title(s, "Flowchart", "End-to-end: raw packets → preprocessing → both pipelines", accent=BLUE)

# --- shared preprocessing, top strip ---
top_box = flow_step(s, Inches(4.9), Inches(1.2), Inches(3.5), Inches(0.65),
                     "Raw CSI packets", "~100–150 pkt/s, 128 complex subcarriers",
                     fill=RGBColor(0x11, 0x1E, 0x2E), line=RGBColor(0x11, 0x1E, 0x2E), label_color=WHITE)
add_arrow(s, Inches(6.65), Inches(1.85), Inches(6.65), Inches(2.05))

shared_steps = [("Clean", "drop bad packets"), ("Dominant length", "keep mode csi_len"),
                ("Amplitude", "sqrt(real²+imag²)")]
sw, sgap = Inches(2.05), Inches(0.2)
stotal = sw * 3 + sgap * 2
sx0 = int((SLIDE_W - stotal) / 2)
for i, (label, sub) in enumerate(shared_steps):
    x = sx0 + i * (sw + sgap)
    flow_step(s, x, Inches(2.1), sw, Inches(0.65), label, sub, fill=AQUA, line=AQUA, label_color=WHITE)
    if i < 2:
        add_arrow(s, x + sw, Inches(2.425), x + sw + sgap, Inches(2.425))

# --- split point ---
mid_x = SLIDE_W / 2
add_arrow(s, mid_x, Inches(2.75), mid_x, Inches(2.95))
left_col_cx = Inches(3.65)
right_col_cx = Inches(9.7)
add_arrow(s, mid_x, Inches(2.95), left_col_cx, Inches(3.15))
add_arrow(s, mid_x, Inches(2.95), right_col_cx, Inches(3.15))

# --- column headers ---
lx, lw = Inches(0.8), Inches(5.7)
rx, rw = Inches(6.85), Inches(5.65)
add_rect(s, lx, Inches(3.15), lw, Inches(0.4), fill=BLUE)
add_textbox(s, lx, Inches(3.15), lw, Inches(0.4), "HOME MODEL  —  Random Forest",
            size=12.5, color=WHITE, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
add_rect(s, rx, Inches(3.15), rw, Inches(0.4), fill=ORANGE)
add_textbox(s, rx, Inches(3.15), rw, Inches(0.4), "SIGNATURE + EVM  —  Transformer",
            size=12.5, color=WHITE, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

# --- left branch: Home Model ---
left_steps = [
    "Clip windowing (3 s / 1 s stride) + 50% coverage gate",
    "Per-subcarrier moments: mean, std, skew, kurtosis",
    "Feature vector: 4 × 128 subcarriers = 512-d",
    "Random Forest — 300 trees, bagging + feature subsampling",
    "Majority vote across clips  →  anjali / barath",
]
# --- right branch: Signature + EVM ---
right_steps = [
    "Time-normalize: resample to common packet rate",
    "Windowing: 200-packet windows / 100-packet stride",
    "Transformer encoder (BranchEncoder): attention → 32-d",
    "Mean-pool + L2-normalize  →  unit-norm signature",
    "EVM Weibull scoring  →  accept/reject + identity / “unknown”",
]

box_h = Inches(0.5)
vgap = Inches(0.13)
y0 = Inches(3.65)
left_colors = [SURFACE, SURFACE, SURFACE, BLUE, BLUE]
right_colors = [SURFACE, SURFACE, SURFACE, ORANGE, ORANGE]
for i, (label, color) in enumerate(zip(left_steps, left_colors)):
    y = y0 + i * (box_h + vgap)
    is_accent = color != SURFACE
    box = flow_step(s, lx, y, lw, box_h, label,
                     fill=color, line=(color if is_accent else BLUE),
                     label_color=(WHITE if is_accent else INK))
    for para in box.text_frame.paragraphs:
        for run in para.runs:
            run.font.size = Pt(12)
    if i < len(left_steps) - 1:
        add_arrow(s, lx + lw / 2, y + box_h, lx + lw / 2, y + box_h + vgap)

for i, (label, color) in enumerate(zip(right_steps, right_colors)):
    y = y0 + i * (box_h + vgap)
    is_accent = color != SURFACE
    box = flow_step(s, rx, y, rw, box_h, label,
                     fill=color, line=(color if is_accent else ORANGE),
                     label_color=(WHITE if is_accent else INK))
    for para in box.text_frame.paragraphs:
        for run in para.runs:
            run.font.size = Pt(12)
    if i < len(right_steps) - 1:
        add_arrow(s, rx + rw / 2, y + box_h, rx + rw / 2, y + box_h + vgap)

add_footer(s, 3)

# =================================================================================================
# Slide 4 -- What is CSI
# =================================================================================================
s = add_slide()
add_kicker_title(s, "Raw data", "What a CSI reading actually is", accent=BLUE)

add_textbox(s, Inches(0.8), Inches(1.35), Inches(5.6), Inches(4.9), "", size=1)
add_bullets(s, Inches(0.8), Inches(1.4), Inches(5.7), Inches(4.6), [
    "Every WiFi packet the ESP32 hears yields one CSI reading",
    "The channel is split into ~128 narrow frequency slices (subcarriers)",
    "Each subcarrier's CSI value is a complex number: real + imaginary part",
    "Packets arrive continuously — roughly 100–150 packets/second",
    ("Amplitude is used, not phase:", 0),
    ("amplitude = sqrt(real² + imaginary²)", 1),
    ("Phase is noisy on cheap radios (ESP32 clock/sampling offsets) —", 1),
    ("amplitude is the standard, more robust channel for this kind of sensing", 1),
], size=15)

# right-side visual: packet -> vector of subcarriers
diag_x = Inches(7.0)
add_rect(s, diag_x, Inches(1.5), Inches(5.3), Inches(4.3), fill=SURFACE, line=BORDER)
add_textbox(s, diag_x + Inches(0.3), Inches(1.65), Inches(4.7), Inches(0.35),
            "One packet → one CSI vector", size=13, color=INK_SECONDARY, bold=True)
pkt = flow_step(s, diag_x + Inches(0.4), Inches(2.15), Inches(1.7), Inches(0.9),
                 "WiFi packet", fill=BLUE, line=BLUE, label_color=WHITE)
add_arrow(s, diag_x + Inches(2.15), Inches(2.6), diag_x + Inches(2.75), Inches(2.6))
vec = flow_step(s, diag_x + Inches(2.8), Inches(2.05), Inches(2.1), Inches(1.1),
                "128 complex\nsubcarriers", "one CSI reading", fill=AQUA, line=AQUA, label_color=WHITE)
add_arrow(s, diag_x + Inches(1.25), Inches(3.05), diag_x + Inches(1.25), Inches(3.4))
amp = flow_step(s, diag_x + Inches(0.4), Inches(3.45), Inches(4.5), Inches(0.85),
                "amplitude = |real + i·imag|", "sqrt(real² + imag²), per subcarrier",
                fill=ORANGE, line=ORANGE, label_color=WHITE)
add_textbox(s, diag_x + Inches(0.3), Inches(4.55), Inches(4.7), Inches(1.1),
            "A full session = a stream of these amplitude vectors over time —\n"
            "variable length, noisy, not directly usable by a fixed-input model.",
            size=12, color=INK_SECONDARY)

add_footer(s, 4)

# =================================================================================================
# Slide 5 -- Common preprocessing pipeline
# =================================================================================================
s = add_slide()
add_kicker_title(s, "Preprocessing", "Raw session → clean amplitude timeline", accent=BLUE)
add_textbox(s, Inches(0.8), Inches(1.2), Inches(11.7), Inches(0.4),
            "Shared by both pipelines before they diverge into clip-moments (Home Model) vs raw windows (Signature model).",
            size=12.5, color=INK_SECONDARY)

steps = [
    ("Raw packets", "per-session CSI stream", BLUE),
    ("Clean", "drop first_word_invalid +\nduplicate-timestamp packets", AQUA),
    ("Dominant length", "keep the mode csi_len;\ndrop odd-shaped frames", AQUA),
    ("Amplitude", "sqrt(real²+imag²)\nper subcarrier, per packet", ORANGE),
]
n = len(steps)
step_w = Inches(2.55)
gap = Inches(0.35)
total_w = step_w * n + gap * (n - 1)
start_x = int((SLIDE_W - total_w) / 2)
y = Inches(2.15)
for i, (label, sub, color) in enumerate(steps):
    x = start_x + i * (step_w + gap)
    flow_step(s, x, y, step_w, Inches(1.35), label, sub, fill=color, line=color, label_color=WHITE)
    if i < n - 1:
        add_arrow(s, x + step_w, y + Inches(0.675), x + step_w + gap, y + Inches(0.675))

add_rect(s, Inches(0.8), Inches(4.0), Inches(11.7), Inches(2.55), fill=SURFACE, line=BORDER)
add_textbox(s, Inches(1.1), Inches(4.15), Inches(11), Inches(0.35),
            "Then the two pipelines diverge:", size=14, color=INK, bold=True)
add_bullets(s, Inches(1.1), Inches(4.6), Inches(5.3), Inches(1.8), [
    "Home Model: cut into 3 s / 1 s-stride clips → reduce each clip to 4 moments per subcarrier",
], size=13, color=BLUE)
add_bullets(s, Inches(6.5), Inches(4.6), Inches(5.3), Inches(1.8), [
    "Signature model: resample to a common packet rate → cut into 200-packet windows, fed raw to the transformer",
], size=13, color=ORANGE)

add_footer(s, 5)

# =================================================================================================
# Slide 6 -- Home model feature pipeline (moments)
# =================================================================================================
s = add_slide()
add_kicker_title(s, "Home model pipeline", "Clip windowing → statistical moments", accent=BLUE)

steps = [
    ("Clip windowing", "3.0 s clips, 1.0 s stride\n(67% overlap)"),
    ("Coverage gate", "drop clip if\n< 50% expected packets"),
    ("Per-subcarrier moments", "mean, std, skew,\nexcess kurtosis"),
    ("Feature vector", "4 × 128 subcarriers\n= 512 numbers"),
]
step_w = Inches(2.75)
gap = Inches(0.3)
total_w = step_w * len(steps) + gap * (len(steps) - 1)
start_x = int((SLIDE_W - total_w) / 2)
y = Inches(1.55)
for i, (label, sub) in enumerate(steps):
    x = start_x + i * (step_w + gap)
    flow_step(s, x, y, step_w, Inches(1.3), label, sub, fill=SURFACE, line=BLUE)
    if i < len(steps) - 1:
        add_arrow(s, x + step_w, y + Inches(0.65), x + step_w + gap, y + Inches(0.65))

add_rect(s, Inches(0.8), Inches(3.25), Inches(11.7), Inches(3.35), fill=SURFACE, line=BORDER)
add_textbox(s, Inches(1.1), Inches(3.4), Inches(11), Inches(0.35),
            "The four moment formulas (per subcarrier, over all N amplitude samples in a clip):",
            size=13.5, color=INK, bold=True)
add_bullets(s, Inches(1.1), Inches(3.85), Inches(11), Inches(2.6), [
    "Mean:  μ = (1/N) Σ xᵢ",
    "Std dev (population):  σ = sqrt( (1/N) Σ (xᵢ − μ)² )",
    "Skewness (Fisher–Pearson):  skew = (1/N) Σ (xᵢ − μ)³ / (σ³ + ε)  — asymmetry of the fade pattern",
    "Excess kurtosis (Fisher):  kurt = (1/N) Σ (xᵢ − μ)⁴ / (σ⁴ + ε) − 3  — spikiness from motion",
    "ε = 1e-6 guards divide-by-zero on constant subcarriers; resulting NaNs → 0",
], size=14)

add_footer(s, 6)

# =================================================================================================
# Slide 7 -- Home model: Random Forest + majority vote
# =================================================================================================
s = add_slide()
add_kicker_title(s, "Home model", "Random Forest classifier + session vote", accent=BLUE)

add_bullets(s, Inches(0.8), Inches(1.4), Inches(5.7), Inches(4.6), [
    "300 decision trees, class_weight=\"balanced\"",
    ("Bagging: each tree trains on an independent bootstrap resample of clips", 1),
    ("Feature subsampling: each split only considers sqrt(512) random features", 1),
    ("Split rule: Gini impurity reduction, Gini = 1 − Σ pₖ²", 1),
    ("Prediction: majority vote across all 300 trees, per clip", 1),
    "No empty-room calibration needed",
    ("RF splits are threshold tests — invariant to affine (z-score) rescaling,", 1),
    ("so raw clip moments are used directly", 1),
    "Session-level output = majority vote across all overlapping clips",
    ("smooths out noisy, correlated per-clip guesses, like averaging many weak estimates", 1),
], size=14)

diag_x = Inches(7.0)
add_rect(s, diag_x, Inches(1.4), Inches(5.4), Inches(4.9), fill=SURFACE, line=BORDER)
flow_step(s, diag_x + Inches(0.4), Inches(1.75), Inches(4.6), Inches(0.75),
          "512-d clip feature vector", fill=BLUE, line=BLUE, label_color=WHITE)
add_arrow(s, diag_x + Inches(2.7), Inches(2.5), diag_x + Inches(2.7), Inches(2.8))
flow_step(s, diag_x + Inches(0.4), Inches(2.85), Inches(4.6), Inches(0.85),
          "300 decision trees", "bagging + random feature subsets",
          fill=AQUA, line=AQUA, label_color=WHITE)
add_arrow(s, diag_x + Inches(2.7), Inches(3.7), diag_x + Inches(2.7), Inches(4.0))
flow_step(s, diag_x + Inches(0.4), Inches(4.05), Inches(4.6), Inches(0.7),
          "per-clip prediction + confidence", fill=SURFACE, line=BORDER)
add_arrow(s, diag_x + Inches(2.7), Inches(4.75), diag_x + Inches(2.7), Inches(5.05))
flow_step(s, diag_x + Inches(0.4), Inches(5.1), Inches(4.6), Inches(0.75),
          "session-level majority vote", fill=ORANGE, line=ORANGE, label_color=WHITE)

add_footer(s, 7)

# =================================================================================================
# Slide 8 -- Signature model preprocessing
# =================================================================================================
s = add_slide()
add_kicker_title(s, "Signature model pipeline", "Preprocessing before the transformer", accent=ORANGE)

steps = [
    ("Session pool", "Day3+Day4,\nchannel-6-only"),
    ("Time normalization", "resample to a common\npacket rate (bin-average)"),
    ("Windowing", "200-packet windows,\n100-packet stride"),
    ("No calibration", "raw amplitude,\nno empty-room z-score"),
]
step_w = Inches(2.75)
gap = Inches(0.3)
total_w = step_w * len(steps) + gap * (len(steps) - 1)
start_x = int((SLIDE_W - total_w) / 2)
y = Inches(1.55)
for i, (label, sub) in enumerate(steps):
    x = start_x + i * (step_w + gap)
    flow_step(s, x, y, step_w, Inches(1.3), label, sub, fill=SURFACE, line=ORANGE)
    if i < len(steps) - 1:
        add_arrow(s, x + step_w, y + Inches(0.65), x + step_w + gap, y + Inches(0.65))

add_bullets(s, Inches(0.8), Inches(3.35), Inches(11.7), Inches(3.2), [
    "Why time-normalize: different sessions/days have slightly different native packet rates —"
    " without this a model could learn to distinguish sessions by packet density instead of CSI content",
    "Why windows, not moments: a transformer already has attention to summarize a sequence itself,"
    " so it is given the raw amplitude sequence rather than a hand-reduced feature vector",
    "Why no calibration: the reference project tested empty-room z-scoring for this contrastive setup and"
    " found it neutral-to-harmful — it let the model implicitly answer “is this authorized data” instead"
    " of learning identity signal directly",
    "Amplitude only, no phase — same reasoning as the home model (matches WhoFi's published design choice)",
    "Trained on all 8 recorded identities (anjali, barath + 6 strangers) so the open-set boundary has"
    " impostor examples to reject against",
    "Input shape to the encoder: (batch, T=200, n_subcarriers=128)",
], size=14.5)

add_footer(s, 8)

# =================================================================================================
# Slide 9 -- Transformer architecture (BranchEncoder) -- the core ask
# =================================================================================================
s = add_slide()
add_kicker_title(s, "Transformer architecture", "BranchEncoder — how it consumes CSI data", accent=ORANGE)

diag_x = Inches(0.8)
diag_w = Inches(5.6)
y = Inches(1.25)
box_h = Inches(0.6)
box_h_tall = Inches(0.82)
gap = Inches(0.12)

layers = [
    ("Input: amplitude (B, T=200, S=128)", BLUE),
    ("Linear projection:  128 → d_model=32", AQUA),
    ("+ Sinusoidal positional encoding", AQUA),
    ("Transformer encoder layer × 1\n(self-attn → Add&Norm → FFN → Add&Norm)", ORANGE),
    ("Mean-pool over time: (B,T,32) → (B,32)", VIOLET),
    ("Linear 32→32, then L2-normalize", VIOLET),
    ("Output: unit-norm signature (B, 32)", BLUE),
]
cy = y
for i, (label, color) in enumerate(layers):
    h = box_h_tall if "\n" in label else box_h
    box = flow_step(s, diag_x, cy, diag_w, h, label, fill=color, line=color, label_color=WHITE)
    if "\n" not in label:
        for para in box.text_frame.paragraphs:
            for run in para.runs:
                run.font.size = Pt(12)
    cy = cy + h
    if i < len(layers) - 1:
        add_arrow(s, diag_x + diag_w / 2, cy, diag_x + diag_w / 2, cy + gap)
        cy = cy + gap

# right column: what "attention" does to CSI, in words
rx = Inches(6.75)
add_rect(s, rx, Inches(1.3), Inches(5.75), Inches(5.55), fill=SURFACE, line=BORDER)
add_textbox(s, rx + Inches(0.3), Inches(1.45), Inches(5.15), Inches(0.4),
            "What each stage does to the CSI window", size=14.5, color=INK, bold=True)
add_bullets(s, rx + Inches(0.3), Inches(1.9), Inches(5.15), Inches(4.9), [
    "Input projection — each packet's 128-subcarrier amplitude vector becomes one 32-dim “token”;"
    " 200 packets = 200 tokens in the sequence",
    "Positional encoding — attention has no notion of order by itself, so a sinusoidal pattern"
    " is added per position so the model can tell packet 1 from packet 150",
    "Self-attention — every packet-token looks at every other packet-token in the window and"
    " reweights itself by how relevant they are to each other (see next slide for the math)",
    "Feed-forward network — a small per-token MLP (32→64→32) applied identically to every"
    " token, adding non-linear capacity after attention mixes information across time",
    "Mean-pool — collapses the 200 refined tokens into a single 32-d vector per window,"
    " a simple parameter-free way to summarize a variable-length window",
    "L2-normalize — projects the embedding onto the unit hypersphere so distance and cosine"
    " similarity between two people's signatures become directly comparable",
], size=12)

add_footer(s, 9)

# =================================================================================================
# Slide 10 -- Self-attention math detail
# =================================================================================================
s = add_slide()
add_kicker_title(s, "Transformer internals", "Scaled dot-product multi-head self-attention", accent=ORANGE)

add_rect(s, Inches(0.8), Inches(1.3), Inches(11.7), Inches(1.9), fill=SURFACE, line=BORDER)
add_textbox(s, Inches(1.1), Inches(1.45), Inches(11), Inches(0.35),
            "Per head:", size=14, color=INK, bold=True)
add_textbox(s, Inches(1.1), Inches(1.85), Inches(11), Inches(0.55),
            "Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V", size=20, color=BLUE, bold=True)
add_textbox(s, Inches(1.1), Inches(2.5), Inches(11), Inches(0.6),
            "Q, K, V are learned linear projections of the 32-dim token sequence;"
            " d_k = d_model / n_heads = 32 / 4 = 8 here.", size=13, color=INK_SECONDARY)

add_bullets(s, Inches(0.8), Inches(3.4), Inches(5.7), Inches(3.5), [
    "4 heads run this in parallel, then concatenate — each head can latch onto"
    " different subcarrier-amplitude relationships across the window",
    "Post-LN block structure (this repo's default):",
    ("x = LayerNorm( x + Dropout( SelfAttention(x) ) )", 1),
    ("x = LayerNorm( x + Dropout( FFN(x) ) )", 1),
    "d_model=32, 1 layer, 4 heads is deliberately small — both the WhoFi paper and"
    " the ESP32 person-ID paper found going deeper hurt on datasets this small"
    " (thousands of windows, not millions)",
], size=14)

add_bullets(s, Inches(6.85), Inches(3.4), Inches(5.65), Inches(3.5), [
    "Sinusoidal positional encoding, added elementwise to the projected input:",
    ("PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )", 1),
    ("PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )", 1),
    "L2 normalization at the output:",
    ("s = z / ‖z‖₂  →  ‖s‖ = 1", 1),
    ("makes Euclidean distance and cosine similarity monotonically related:", 1),
    ("‖a−b‖² = 2 − 2·cos(a,b)  for unit vectors a, b", 1),
], size=14)

add_footer(s, 10)

# =================================================================================================
# Slide 11 -- Training objective (contrastive)
# =================================================================================================
s = add_slide()
add_kicker_title(s, "Training objective", "In-batch-negative contrastive loss (DPR / WhoFi-style)", accent=ORANGE)

add_bullets(s, Inches(0.8), Inches(1.35), Inches(5.7), Inches(5.2), [
    "PK-sampling: every identity with ≥2 windows contributes one (query, gallery) pair per step",
    "Query and gallery windows are drawn from different (date, motion) contexts when available",
    ("Why: a diagnostic found the encoder otherwise learning to separate recording-session", 1),
    ("context instead of identity — forcing cross-context pairs makes the loss only", 1),
    ("satisfiable by identity signal that survives a context change", 1),
    "Loss (cosine-similarity matrix over the batch, cross-entropy on the diagonal):",
], size=13)

add_rect(s, Inches(0.8), Inches(4.65), Inches(11.7), Inches(1.3), fill=SURFACE, line=BORDER)
add_textbox(s, Inches(1.1), Inches(4.72), Inches(11), Inches(0.4),
            "sim = Sq · Sgᵀ  ∈  ℝ^(B×B)          (dot product of L2-normalized query/gallery batches)",
            size=15, color=BLUE, bold=True)
add_textbox(s, Inches(1.1), Inches(5.15), Inches(11), Inches(0.4),
            "L = −(1/B) Σᵢ log( exp(simᵢᵢ) / Σⱼ exp(simᵢⱼ) )",
            size=15, color=BLUE, bold=True)

add_textbox(s, Inches(0.8), Inches(6.1), Inches(11.7), Inches(0.8),
            "Every other identity in the batch is a free “in-batch negative” — no separate negative-mining"
            " step needed. Adam optimizer, lr=1e-3, 300 epochs × 40 steps/epoch, seed=0.",
            size=13, color=INK_SECONDARY)

add_footer(s, 11)

# =================================================================================================
# Slide 12 -- EVM accept/reject layer
# =================================================================================================
s = add_slide()
add_kicker_title(s, "Open-set decision", "Extreme Value Machine (EVM) turns embeddings into accept/reject", accent=ORANGE)

diag_x = Inches(0.8)
flow_step(s, diag_x, Inches(1.35), Inches(3.4), Inches(0.85), "Signature embedding (32-d)",
          "unit hypersphere", fill=VIOLET, line=VIOLET, label_color=WHITE)
add_arrow(s, diag_x + Inches(3.4), Inches(1.775), diag_x + Inches(3.85), Inches(1.775))
flow_step(s, diag_x + Inches(3.9), Inches(1.35), Inches(3.6), Inches(0.85),
          "Weibull tail per identity", "fit on nearest-τ distances", fill=ORANGE, line=ORANGE, label_color=WHITE)
add_arrow(s, diag_x + Inches(7.5), Inches(1.775), diag_x + Inches(7.95), Inches(1.775))
flow_step(s, diag_x + Inches(8.0), Inches(1.35), Inches(3.5), Inches(0.85),
          "ψ-score → accept/reject", "threshold = 0.18", fill=BLUE, line=BLUE, label_color=WHITE)

add_bullets(s, Inches(0.8), Inches(2.55), Inches(11.7), Inches(4.3), [
    "Why not a plain classifier: a closed-set softmax structurally cannot say “none of these”"
    " — measured 68.5% false-accept rate against a genuinely novel stranger",
    "EVM idea (Extreme Value Theory): the distance from a point to another class's points behaves"
    " like an extreme-value statistic — modeled here as a Weibull tail per support point",
    "Score: ψ(x, point) = exp( −(‖x − point‖ / λ)^k )  — the Weibull survival function,"
    " used as an inclusion/similarity score (1 at zero distance, decaying with distance)",
    "This run uses centroid_only=True: each identity collapses to one centroid before fitting"
    " (per-point EVM was noisier with only ~289 points per identity)",
    "Final accept score = top-5-averaged ψ across an identity's support points, max over enrolled identities",
    "Honest result: mean AUROC ≈ 0.45 across held-out strangers — experimental, not deployed;"
    " the home model (Random Forest) remains the shipped path",
], size=14)

add_footer(s, 12)

# =================================================================================================
# Slide 13 -- Comparison summary
# =================================================================================================
s = add_slide()
add_kicker_title(s, "Summary", "Home Model vs Signature + EVM, side by side", accent=BLUE)

rows = [
    ("", "Home Model (v2)", "Signature + EVM"),
    ("Task", "Closed 2-class ID (anjali/barath)", "Open-set accept/reject + ID"),
    ("Model", "Random Forest (300 trees)", "Transformer encoder (32-d) + Weibull/EVM"),
    ("Input to model", "512-d per-clip moment vector", "Raw (T=200, 128-sub) amplitude sequence"),
    ("Windowing", "3 s clips / 1 s stride, 50% coverage gate", "200-packet windows / 100-packet stride"),
    ("Calibration", "None (RF invariant to affine rescale)", "None (found harmful for this setup)"),
    ("Can say “unknown”?", "No", "Yes"),
    ("Status", "Deployed", "Experimental"),
]
table_x, table_y = Inches(0.8), Inches(1.35)
table_w, table_h = Inches(11.7), Inches(5.5)
n_rows, n_cols = len(rows), 3
gtable = s.shapes.add_table(n_rows, n_cols, table_x, table_y, table_w, table_h).table
gtable.columns[0].width = Inches(2.7)
gtable.columns[1].width = Inches(4.5)
gtable.columns[2].width = Inches(4.5)
for r, row in enumerate(rows):
    for c, val in enumerate(row):
        cell = gtable.cell(r, c)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = Pt(8)
        cell.margin_right = Pt(8)
        cell.margin_top = Pt(4)
        cell.margin_bottom = Pt(4)
        para = cell.text_frame.paragraphs[0]
        run = para.add_run()
        run.text = val
        run.font.size = Pt(13 if r > 0 else 14)
        run.font.name = "Segoe UI"
        if r == 0:
            run.font.bold = True
            run.font.color.rgb = WHITE
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0x11, 0x1E, 0x2E)
        else:
            run.font.bold = (c == 0)
            run.font.color.rgb = INK if c == 0 else INK_SECONDARY
            cell.fill.solid()
            cell.fill.fore_color.rgb = SURFACE if r % 2 else PAGE

add_footer(s, 13)

# =================================================================================================
# Slide 14 -- Honest limitations / next steps
# =================================================================================================
s = add_slide()
add_kicker_title(s, "Limitations & next steps", "What this pipeline does not do yet", accent=RED)

add_bullets(s, Inches(0.8), Inches(1.4), Inches(5.7), Inches(5), [
    "Home model only knows 2 people — a stranger still gets forced into anjali or barath",
    "Day3→Day4 was never isolated/cross-validated before v2 shipped — it pools both days",
    "Sensitive to WiFi channel — trained only on channel 6 / 20 MHz",
    "Signature+EVM false-accept rate is still too high for anything beyond experimentation",
    "No cross-day validation beyond Day3+Day4 for the signature model (no Day5 yet)",
], size=14.5)

add_bullets(s, Inches(6.85), Inches(1.4), Inches(5.65), Inches(5), [
    "Documented next levers for EVM (not bugs, just not built yet):",
    ("Cohort / T-norm score normalization — biggest false-accept-rate reducer in the reference project", 1),
    ("Per-identity EER thresholds instead of one global 0.18", 1),
    ("Multi-seed ensembling", 1),
    ("Walking-motion gating", 1),
    ("Temporal smoothing over consecutive windows", 1),
], size=14.5)

add_footer(s, 14)

prs.save("ml/reports/wifi_csi_pipeline.pptx")
print("Saved ml/reports/wifi_csi_pipeline.pptx")
