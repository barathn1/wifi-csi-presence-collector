"""Builds a standalone HTML visualization of the preprocessed (cleaned + 3s/1s-stride
windowed + coverage-filtered) empty-room CSI data, from _empty_room_viz.json.
Scratch tool -- safe to delete alongside its siblings.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>__TITLE__</title>
<style>
  .viz-root {
    color-scheme: light;
    --surface-1:      #fcfcfb;
    --page:           #f9f9f7;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --text-muted:     #898781;
    --grid:           #e1e0d9;
    --axis:           #c3c2b7;
    --series-1:       #2a78d6;
    --status-critical:#d03b3b;
    --border:         rgba(11,11,11,0.10);
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) .viz-root {
      color-scheme: dark;
      --surface-1:      #1a1a19;
      --page:           #0d0d0d;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted:     #898781;
      --grid:           #2c2c2a;
      --axis:           #383835;
      --series-1:       #3987e5;
      --status-critical:#e66767;
      --border:         rgba(255,255,255,0.10);
    }
  }
  :root[data-theme="dark"] .viz-root {
    color-scheme: dark;
    --surface-1:      #1a1a19;
    --page:           #0d0d0d;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #898781;
    --grid:           #2c2c2a;
    --axis:           #383835;
    --series-1:       #3987e5;
    --status-critical:#e66767;
    --border:         rgba(255,255,255,0.10);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px; background: var(--page);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--text-primary);
  }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .subtitle { color: var(--text-secondary); font-size: 13px; margin: 0 0 20px; max-width: 900px; }
  .theme-toggle {
    position: fixed; top: 20px; right: 24px; font-size: 12px;
    padding: 6px 10px; border-radius: 6px; border: 1px solid var(--border);
    background: var(--surface-1); color: var(--text-primary); cursor: pointer;
  }
  .panel {
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px 18px 20px; margin-bottom: 22px;
  }
  .panel h2 { font-size: 15px; margin: 0 0 2px; }
  .meta { color: var(--text-secondary); font-size: 12px; margin: 0 0 4px; }
  .stat { display: inline-flex; align-items: baseline; gap: 4px; margin-right: 18px; }
  .stat b { font-size: 15px; }
  .stats-row { margin: 8px 0 12px; }
  .chart-wrap { position: relative; margin-top: 6px; }
  canvas { display: block; width: 100%; }
  .strip-label { font-size: 10px; color: var(--text-muted); margin: 10px 0 2px; }
  .hover-line {
    position: absolute; top: 0; bottom: 0; width: 1px;
    background: var(--text-muted); pointer-events: none; display: none;
  }
  .tooltip {
    position: absolute; pointer-events: none; display: none;
    background: var(--text-primary); color: var(--surface-1);
    font-size: 11px; padding: 6px 8px; border-radius: 6px; line-height: 1.5;
    white-space: nowrap; z-index: 5; transform: translate(-50%, -110%);
  }
  .legend { display: flex; align-items: center; gap: 8px; margin-top: 10px; font-size: 11px; color: var(--text-secondary); }
  .ramp { width: 160px; height: 10px; border-radius: 4px;
    background: linear-gradient(to right, #cde2fb, #6da7ec, #2a78d6, #184f95, #0d366b); }
</style>
</head>
<body>
<div class="viz-root">
  <button class="theme-toggle" id="themeToggle">Dark mode</button>
  <h1>__TITLE__</h1>
  <p class="subtitle">__SUBTITLE__</p>
  <div id="panels"></div>
</div>
<script>
const DATA = __PAYLOAD__;

function hexToRgb(h) { h = h.replace('#',''); return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)]; }
const RAMP = ["#cde2fb","#b7d3f6","#9ec5f4","#86b6ef","#6da7ec","#5598e7","#3987e5","#2a78d6","#256abf","#1c5cab","#184f95","#104281","#0d366b"].map(hexToRgb);
function ampColor(v, vmin, vmax) {
  if (v < 0) return [225, 224, 217];
  let t = Math.max(0, Math.min(1, (v - vmin) / (vmax - vmin)));
  const pos = t * (RAMP.length - 1);
  const i0 = Math.floor(pos), i1 = Math.min(RAMP.length - 1, i0 + 1);
  const f = pos - i0;
  return [0,1,2].map(i => Math.round(RAMP[i0][i] + (RAMP[i1][i]-RAMP[i0][i])*f));
}

function buildPanel(day, vmin, vmax) {
  const panel = document.createElement('div');
  panel.className = 'panel';
  const pct = (100 * day.n_clips_kept / day.n_clips_total).toFixed(1);
  panel.innerHTML = `
    <h2>${day.day}</h2>
    <p class="meta">${day.n_sessions} sessions &middot; ${day.n_subcarriers} subcarriers (dominant CSI mode) &middot; ${day.clip_len_s}s clips / ${day.stride_s}s stride</p>
    <div class="stats-row">
      <span class="stat"><b>${day.total_raw_packets.toLocaleString()}</b><span>raw packets</span></span>
      <span class="stat"><b>${day.total_clean_packets.toLocaleString()}</b><span>after clean (drop incomplete/corrupted/dup)</span></span>
      <span class="stat"><b>${day.n_clips_kept.toLocaleString()} / ${day.n_clips_total.toLocaleString()}</b><span>clips kept (${pct}%, &ge;50% coverage)</span></span>
    </div>
    <div class="chart-wrap" id="wrap-${day.day}">
      <canvas class="heatmap" width="1100" height="220"></canvas>
      <div class="strip-label">mean amplitude per clip (dropped clips shown as gaps)</div>
      <canvas class="lines" width="1100" height="80"></canvas>
      <div class="strip-label">clip kept / dropped</div>
      <canvas class="strip" width="1100" height="18"></canvas>
      <div class="hover-line"></div>
      <div class="tooltip"></div>
    </div>
    <div class="legend">
      <span>amplitude (raw int8 units)</span>
      <div class="ramp"></div>
      <span>${vmin.toFixed(0)} -> ${vmax.toFixed(0)}</span>
    </div>
  `;
  document.getElementById('panels').appendChild(panel);

  const wrap = panel.querySelector('.chart-wrap');
  const heatCanvas = panel.querySelector('canvas.heatmap');
  const lineCanvas = panel.querySelector('canvas.lines');
  const stripCanvas = panel.querySelector('canvas.strip');
  const hoverLine = panel.querySelector('.hover-line');
  const tooltip = panel.querySelector('.tooltip');

  const W = heatCanvas.width, H = heatCanvas.height;
  const hctx = heatCanvas.getContext('2d');
  const nBuckets = day.n_buckets, nSub = day.n_subcarriers;
  const cellW = W / nBuckets, cellH = H / nSub;

  for (let b = 0; b < nBuckets; b++) {
    const col = day.heat[b];
    for (let s = 0; s < nSub; s++) {
      const [r,g,bl] = ampColor(col[s], vmin, vmax);
      hctx.fillStyle = `rgb(${r},${g},${bl})`;
      hctx.fillRect(b*cellW, H - (s+1)*cellH, Math.ceil(cellW)+0.5, Math.ceil(cellH)+0.5);
    }
  }
  hctx.strokeStyle = 'rgba(11,11,11,0.35)';
  hctx.setLineDash([3,3]);
  day.session_boundaries_bucket.forEach(bb => {
    const x = bb * cellW;
    hctx.beginPath(); hctx.moveTo(x, 0); hctx.lineTo(x, H); hctx.stroke();
  });

  const nClips = day.n_clips_total;
  const lctx = lineCanvas.getContext('2d');
  const LW = lineCanvas.width, LH = lineCanvas.height;
  const means = day.mean_amp;
  const validMeans = means.filter(v => v >= 0);
  const lmin = 0, lmax = Math.max(...validMeans) * 1.1;
  const xAt = i => (i / nClips) * LW;
  const yAt = v => LH - 6 - ((v - lmin) / (lmax - lmin)) * (LH - 12);

  lctx.strokeStyle = 'rgba(137,135,129,0.3)';
  lctx.lineWidth = 1;
  [0, 0.5, 1].forEach(f => { const y = 6 + f * (LH - 12); lctx.beginPath(); lctx.moveTo(0,y); lctx.lineTo(LW,y); lctx.stroke(); });

  lctx.beginPath();
  let started = false;
  for (let i = 0; i < nClips; i++) {
    if (means[i] < 0) { started = false; continue; }
    const x = xAt(i), y = yAt(means[i]);
    if (!started) { lctx.moveTo(x, y); started = true; } else { lctx.lineTo(x, y); }
  }
  lctx.strokeStyle = '#2a78d6';
  lctx.lineWidth = 1.5;
  lctx.stroke();

  const sctx = stripCanvas.getContext('2d');
  const SW = stripCanvas.width, SH = stripCanvas.height;
  const kept = day.kept_flags;
  const cw = SW / nClips;
  for (let i = 0; i < nClips; i++) {
    sctx.fillStyle = kept[i] ? '#2a78d6' : '#d03b3b';
    sctx.fillRect(i*cw, 0, Math.ceil(cw)+0.5, SH);
  }

  wrap.addEventListener('mousemove', (ev) => {
    const rect = heatCanvas.getBoundingClientRect();
    const xFrac = (ev.clientX - rect.left) / rect.width;
    const clipIdx = Math.max(0, Math.min(nClips - 1, Math.floor(xFrac * nClips)));
    const bucketIdx = Math.max(0, Math.min(nBuckets - 1, Math.floor(xFrac * nBuckets)));
    hoverLine.style.display = 'block';
    hoverLine.style.left = (xFrac * rect.width) + 'px';
    hoverLine.style.height = (rect.height + 80 + 18 + 24) + 'px';

    const mean = means[clipIdx];
    const isKept = kept[clipIdx];
    tooltip.innerHTML = `clip ${clipIdx} (t&approx;${clipIdx * day.stride_s}s)<br>` +
      `${isKept ? 'kept' : 'dropped (&lt;50% coverage)'}<br>` +
      `mean amp: ${mean < 0 ? 'n/a' : mean.toFixed(1)}`;
    tooltip.style.display = 'block';
    tooltip.style.left = (xFrac * rect.width) + 'px';
    tooltip.style.top = '0px';
  });
  wrap.addEventListener('mouseleave', () => { hoverLine.style.display = 'none'; tooltip.style.display = 'none'; });
}

DATA.days.forEach(day => buildPanel(day, DATA.vmin, DATA.vmax));

document.getElementById('themeToggle').addEventListener('click', () => {
  const root = document.documentElement;
  root.setAttribute('data-theme', root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
});
</script>
</body>
</html>
"""

DEFAULT_SUBTITLE = (
    "Pipeline applied per session: amplitude = &radic;(real&sup2;+imag&sup2;) per subcarrier &rarr; "
    "drop incomplete/corrupted (<code>first_word_invalid</code>)/duplicate-timestamp packets &rarr; "
    "3s clips, 1s stride &rarr; drop clips with &lt;50% of the session's expected packet count. "
    "Heatmap = mean amplitude per subcarrier, grouped into display buckets of consecutive clips (shared color scale, 0-99th pct). "
    "Bottom strip: kept (blue) vs dropped (red) clip, at full 1s-stride resolution. Dashed lines mark session boundaries."
)


def build(json_name: str, out_name: str, title: str, subtitle: str = DEFAULT_SUBTITLE) -> Path:
    data = json.loads((HERE / json_name).read_text())
    all_vals = np.concatenate([np.array(d["heat"])[np.array(d["heat"]) >= 0] for d in data])
    vmin, vmax = 0.0, float(np.percentile(all_vals, 99))
    payload = json.dumps({"days": data, "vmin": vmin, "vmax": vmax})

    html = (HTML_TEMPLATE
            .replace("__PAYLOAD__", payload)
            .replace("__TITLE__", title)
            .replace("__SUBTITLE__", subtitle))
    out_path = HERE / out_name
    out_path.write_text(html, encoding="utf-8")
    print("wrote", out_path)
    return out_path


if __name__ == "__main__":
    build("_empty_room_viz.json", "empty_room_preprocessed.html", "Preprocessed empty-room CSI -- Day 1 vs Day 2")
