"""Build one browsable HTML gallery over every static PNG in visualizations/static/, so the whole
analysis can be stepped through in sequence (Prev/Next buttons + arrow keys, thumbnail strip to jump
around) instead of opening 30+ files one at a time in a file browser.
"""
from __future__ import annotations

import json
from pathlib import Path

from ml.visualization.palette import STATIC_FIG_DIR

# Ordered categories -- this defines "the sequence" the gallery steps through, roughly the same order
# as ml/reports/day1_findings.md walks through the analysis.
CATEGORIES = [
    ("signal_vs_noise_", "1. Signal vs. noise (empirical noise floor)",
     "Real per-subcarrier effect size (colored bars) against the range of effect sizes pure chance "
     "produces from 5000 session-label shuffles (gray band). Bars shown solid survive FDR correction "
     "across all 186 subcarriers; everything faded is statistically indistinguishable from noise. "
     "Start here -- this is the direct answer to \"what's real.\""),
    ("distributions_", "2. Raw distributions for the top subcarriers",
     "Violin plots of the actual per-window values for the subcarriers ranked most significant in the "
     "noise-floor test above -- see the separation directly instead of reading a summary statistic."),
    ("effect_size_", "3. Effect size heatmaps (Cohen's d, unadjusted)",
     "Cohen's d per subcarrier x class-pair -- a quicker look than the noise-floor test, but NOT "
     "corrected for multiple comparisons or session structure. Cross-check against panel 1 before "
     "trusting any single cell here."),
    ("fingerprint_", "4. Per-class amplitude/phase fingerprints",
     "Mean +/- std band per subcarrier for each class -- a compact 'signature' rather than raw traces."),
    ("feature_importance_", "5. RandomForest feature importance",
     "Which subcarriers the classical baseline actually leans on per task, summed over "
     "mean/std/skew/kurtosis -- a second, model-agnostic cross-check against the effect-size view."),
    ("embedding_", "6. UMAP embeddings",
     "2D projection of window-level statistics, colored by class/identity -- cluster separation (or "
     "lack of it) at a glance."),
    ("attention_", "7. Cross-attention transformer attention weights",
     "Which moments within the ~1s window the trained cross-attention model leans on (temporal, not "
     "spectral -- see panels above for which subcarriers matter)."),
]


def categorize(filename: str) -> int:
    for i, (prefix, _, _) in enumerate(CATEGORIES):
        if filename.startswith(prefix):
            return i
    return len(CATEGORIES)


def humanize(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0]
    for prefix, _, _ in CATEGORIES:
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    return stem.replace("_", " ")


def build_gallery(fig_dir: Path, out_path: Path) -> None:
    files = sorted(fig_dir.glob("*.png"))
    if not files:
        print(f"no PNGs found in {fig_dir} -- run the static visualization scripts first")
        return

    slides = []
    for f in files:
        cat_idx = categorize(f.name)
        cat_title, cat_desc = CATEGORIES[cat_idx][1:] if cat_idx < len(CATEGORIES) else ("Other", "")
        slides.append({"file": f.name, "category": cat_title, "description": cat_desc, "title": humanize(f.name)})
    slides.sort(key=lambda s: (s["category"], s["file"]))

    thumbs_by_cat: dict[str, list[dict]] = {}
    for i, s in enumerate(slides):
        s["index"] = i
        thumbs_by_cat.setdefault(s["category"], []).append(s)

    thumb_sections = "\n".join(
        f'<div class="cat-header">{cat}</div><div class="thumb-row">' +
        "".join(f'<img class="thumb" data-idx="{s["index"]}" src="{s["file"]}" title="{s["title"]}">'
                for s in items) +
        "</div>"
        for cat, items in thumbs_by_cat.items()
    )

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>CSI analysis gallery</title>
<style>
:root {{ --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9; --blue:#2a78d6; }}
body {{ font-family: system-ui, -apple-system, sans-serif; background:var(--surface); color:var(--ink); margin:0; padding:24px; }}
h2 {{ margin-top:0; }}
.viewer {{ display:flex; gap:24px; align-items:flex-start; margin-bottom:28px; }}
.main {{ flex:1; min-width:0; }}
.main img {{ width:100%; border:1px solid var(--line); border-radius:6px; background:#fff; }}
.controls {{ display:flex; align-items:center; gap:12px; margin-top:12px; }}
button {{ padding:8px 16px; border:1px solid var(--line); border-radius:4px; background:#fff; color:var(--ink);
          cursor:pointer; font-size:14px; }}
button:hover {{ background:#f0efec; }}
.counter {{ color:var(--muted); font-size:13px; }}
.category {{ color:var(--blue); font-weight:600; font-size:14px; margin-top:4px; }}
.title {{ font-size:20px; font-weight:600; margin:4px 0; }}
.description {{ color:var(--ink2); font-size:14px; line-height:1.5; max-width:900px; }}
.thumbs {{ width:220px; max-height:80vh; overflow-y:auto; border-left:1px solid var(--line); padding-left:16px; }}
.cat-header {{ color:var(--ink2); font-size:12px; font-weight:600; text-transform:uppercase; margin:14px 0 6px; }}
.cat-header:first-child {{ margin-top:0; }}
.thumb-row {{ display:flex; flex-direction:column; gap:6px; }}
.thumb {{ width:100%; border:2px solid transparent; border-radius:4px; cursor:pointer; opacity:0.75; }}
.thumb:hover {{ opacity:1; }}
.thumb.active {{ border-color:var(--blue); opacity:1; }}
kbd {{ background:#f0efec; border:1px solid var(--line); border-radius:3px; padding:1px 6px; font-size:12px; }}
</style></head><body>
<h2>CSI analysis gallery</h2>
<p style="color:var(--ink2)">Step through every static figure in sequence with the buttons, arrow keys
(<kbd>&larr;</kbd>/<kbd>&rarr;</kbd>), or jump directly via the thumbnails on the right.</p>
<div class="viewer">
  <div class="main">
    <div class="category" id="category"></div>
    <div class="title" id="title"></div>
    <img id="mainImg" src="">
    <div class="controls">
      <button onclick="go(-1)">&larr; Prev</button>
      <button onclick="go(1)">Next &rarr;</button>
      <span class="counter" id="counter"></span>
    </div>
    <p class="description" id="description"></p>
  </div>
  <div class="thumbs">
    {thumb_sections}
  </div>
</div>
<script>
const slides = {json.dumps(slides)};
let current = 0;

function render() {{
  const s = slides[current];
  document.getElementById('mainImg').src = s.file;
  document.getElementById('category').textContent = s.category;
  document.getElementById('title').textContent = s.title;
  document.getElementById('description').textContent = s.description;
  document.getElementById('counter').textContent = `${{current + 1}} / ${{slides.length}}`;
  document.querySelectorAll('.thumb').forEach(t => t.classList.toggle('active', Number(t.dataset.idx) === current));
  const activeThumb = document.querySelector('.thumb.active');
  if (activeThumb) activeThumb.scrollIntoView({{block: 'nearest'}});
}}
function go(delta) {{
  current = (current + delta + slides.length) % slides.length;
  render();
}}
document.querySelectorAll('.thumb').forEach(t => t.addEventListener('click', () => {{
  current = Number(t.dataset.idx);
  render();
}}));
document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft') go(-1);
  if (e.key === 'ArrowRight') go(1);
}});
render();
</script>
</body></html>"""
    out_path.write_text(html)
    print(f"wrote {out_path} ({len(slides)} images)")


if __name__ == "__main__":
    build_gallery(STATIC_FIG_DIR, STATIC_FIG_DIR / "index.html")
