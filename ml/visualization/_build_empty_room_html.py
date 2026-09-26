"""Builds a standalone HTML comparison of raw empty-room CSI amplitude (Day 1 vs Day 2)
from _empty_room_scratch.json. Scratch tool -- safe to delete alongside its sibling.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
data = json.loads((HERE / "_empty_room_scratch.json").read_text())

all_vals = []
for d in data:
    arr = np.array(d["heat"])
    all_vals.append(arr[arr >= 0])
all_vals = np.concatenate(all_vals)
VMIN = 0.0
VMAX = float(np.percentile(all_vals, 99))

payload = json.dumps({"days": data, "vmin": VMIN, "vmax": VMAX})

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Empty-room raw CSI amplitude -- Day 1 vs Day 2</title>
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
    --series-2:       #eb6834;
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
      --series-2:       #d95926;
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
    --series-2:       #d95926;
    --border:         rgba(255,255,255,0.10);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px; background: var(--page);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--text-primary);
  }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .subtitle { color: var(--text-secondary); font-size: 13px; margin: 0 0 20px; }
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
  .meta { color: var(--text-secondary); font-size: 12px; margin: 0 0 12px; }
  .chart-wrap { position: relative; }
  canvas { display: block; width: 100%; }
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
  .axis-label { font-size: 11px; fill: var(--text-muted); }
  svg text { fill: var(--text-muted); font-size: 11px; }
  svg line.grid { stroke: var(--grid); stroke-width: 1; }
  svg line.boundary { stroke: var(--text-muted); stroke-width: 1; stroke-dasharray: 3 3; }
  .line-series { fill: none; stroke: var(--series-1); stroke-width: 2; }
  .line-band { fill: var(--series-1); opacity: 0.12; }
</style>
</head>
<body>
<div class="viz-root">
  <button class="theme-toggle" id="themeToggle">Dark mode</button>
  <h1>Empty-room raw CSI amplitude -- Day 1 vs Day 2</h1>
  <p class="subtitle">Every "none" (unoccupied room) session per day, concatenated in capture order.
    Heatmap = mean |CSI| per subcarrier per __BIN_S__s bin (shared color scale, 0-99th pct). Dashed lines mark session boundaries. Hover for exact values.</p>
  <div id="panels"></div>
</div>
<script>
const DATA = __PAYLOAD__;

function lerpColor(a, b, t) {
  return [0,1,2].map(i => Math.round(a[i] + (b[i]-a[i])*t));
}
function hexToRgb(h) {
  h = h.replace('#','');
  return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)];
}
const RAMP_HEX = ["#cde2fb","#b7d3f6","#9ec5f4","#86b6ef","#6da7ec","#5598e7","#3987e5","#2a78d6","#256abf","#1c5cab","#184f95","#104281","#0d366b"];
const RAMP = RAMP_HEX.map(hexToRgb);
function ampColor(v, vmin, vmax) {
  if (v < 0) return [225, 224, 217]; // missing bin -> gridline gray
  let t = Math.max(0, Math.min(1, (v - vmin) / (vmax - vmin)));
  const pos = t * (RAMP.length - 1);
  const i0 = Math.floor(pos), i1 = Math.min(RAMP.length - 1, i0 + 1);
  const rgb = lerpColor(RAMP[i0], RAMP[i1], pos - i0);
  return rgb;
}

function buildPanel(day, vmin, vmax) {
  const panel = document.createElement('div');
  panel.className = 'panel';
  const totalS = day.n_bins * day.bin_width_s;
  panel.innerHTML = `
    <h2>${day.day}</h2>
    <p class="meta">${day.n_sessions} sessions &middot; ${day.total_packets.toLocaleString()} packets &middot; ${day.n_subcarriers} subcarriers (dominant CSI mode) &middot; ~${Math.round(totalS)}s total capture time</p>
    <div class="chart-wrap" id="wrap-${day.day}">
      <canvas class="heatmap" width="1100" height="240"></canvas>
      <canvas class="lines" width="1100" height="90"></canvas>
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
  const hoverLine = panel.querySelector('.hover-line');
  const tooltip = panel.querySelector('.tooltip');

  const W = heatCanvas.width, H = heatCanvas.height;
  const hctx = heatCanvas.getContext('2d');
  const nBins = day.n_bins, nSub = day.n_subcarriers;
  const cellW = W / nBins;
  const cellH = H / nSub;

  for (let b = 0; b < nBins; b++) {
    const col = day.heat[b];
    for (let s = 0; s < nSub; s++) {
      const [r,g,bl] = ampColor(col[s], vmin, vmax);
      hctx.fillStyle = `rgb(${r},${g},${bl})`;
      hctx.fillRect(b*cellW, H - (s+1)*cellH, Math.ceil(cellW)+0.5, Math.ceil(cellH)+0.5);
    }
  }
  // session boundaries
  hctx.strokeStyle = 'rgba(11,11,11,0.35)';
  hctx.setLineDash([3,3]);
  day.session_boundaries_s.forEach(t => {
    const x = (t / day.bin_width_s / nBins) * W * (nBins / nBins); // t already in bins-space via /bin_width_s
    const xb = (t / day.bin_width_s) * cellW;
    hctx.beginPath(); hctx.moveTo(xb, 0); hctx.lineTo(xb, H); hctx.stroke();
  });

  // line chart: mean amplitude +/- std
  const lctx = lineCanvas.getContext('2d');
  const LW = lineCanvas.width, LH = lineCanvas.height;
  const means = day.line_mean, stds = day.line_std;
  const validMeans = means.filter(v => v >= 0);
  const lmin = 0, lmax = Math.max(...validMeans.map((v,i)=>v)) * 1.15;
  function xAt(b) { return (b / nBins) * LW; }
  function yAt(v) { return LH - 8 - ((v - lmin) / (lmax - lmin)) * (LH - 16); }

  lctx.strokeStyle = getComputedStyle(document.querySelector('.viz-root')).getPropertyValue('--grid');
  lctx.lineWidth = 1;
  [0, 0.5, 1].forEach(f => {
    const y = 8 + f * (LH - 16);
    lctx.beginPath(); lctx.moveTo(0, y); lctx.lineTo(LW, y); lctx.stroke();
  });

  // std band
  lctx.beginPath();
  let started = false;
  for (let b = 0; b < nBins; b++) {
    if (means[b] < 0) { started = false; continue; }
    const x = xAt(b), yTop = yAt(means[b] + stds[b]);
    if (!started) { lctx.moveTo(x, yTop); started = true; } else { lctx.lineTo(x, yTop); }
  }
  for (let b = nBins - 1; b >= 0; b--) {
    if (means[b] < 0) continue;
    lctx.lineTo(xAt(b), yAt(Math.max(lmin, means[b] - stds[b])));
  }
  lctx.closePath();
  lctx.fillStyle = 'rgba(42,120,214,0.15)';
  lctx.fill();

  lctx.beginPath();
  started = false;
  for (let b = 0; b < nBins; b++) {
    if (means[b] < 0) { started = false; continue; }
    const x = xAt(b), y = yAt(means[b]);
    if (!started) { lctx.moveTo(x, y); started = true; } else { lctx.lineTo(x, y); }
  }
  lctx.strokeStyle = '#2a78d6';
  lctx.lineWidth = 2;
  lctx.stroke();

  // hover interaction (shared across both canvases, mapped by x only)
  wrap.addEventListener('mousemove', (ev) => {
    const rect = heatCanvas.getBoundingClientRect();
    const xFrac = (ev.clientX - rect.left) / rect.width;
    const yFracHeat = (ev.clientY - rect.top) / rect.height;
    const bin = Math.max(0, Math.min(nBins - 1, Math.floor(xFrac * nBins)));
    const timeS = (bin * day.bin_width_s).toFixed(0);
    hoverLine.style.display = 'block';
    hoverLine.style.left = (xFrac * rect.width) + 'px';
    hoverLine.style.height = (rect.height + 90) + 'px';

    let detail = '';
    if (ev.clientY - rect.top <= rect.height) {
      const sub = Math.max(0, Math.min(nSub - 1, Math.floor((1 - yFracHeat) * nSub)));
      const v = day.heat[bin][sub];
      detail = `subcarrier ${sub}: ${v < 0 ? 'n/a' : v.toFixed(1)}<br>`;
    }
    const mean = day.line_mean[bin];
    tooltip.innerHTML = `t=${timeS}s<br>${detail}mean amp: ${mean < 0 ? 'n/a' : mean.toFixed(1)}`;
    tooltip.style.display = 'block';
    tooltip.style.left = (xFrac * rect.width) + 'px';
    tooltip.style.top = '0px';
  });
  wrap.addEventListener('mouseleave', () => {
    hoverLine.style.display = 'none';
    tooltip.style.display = 'none';
  });
}

DATA.days.forEach(day => buildPanel(day, DATA.vmin, DATA.vmax));

document.getElementById('themeToggle').addEventListener('click', () => {
  const root = document.documentElement;
  const cur = root.getAttribute('data-theme');
  if (cur === 'dark') { root.setAttribute('data-theme', 'light'); }
  else { root.setAttribute('data-theme', 'dark'); }
});
</script>
</body>
</html>
"""

HTML = HTML.replace("__PAYLOAD__", payload).replace("__BIN_S__", str(data[0]["bin_width_s"]))

out_path = HERE / "empty_room_comparison.html"
out_path.write_text(HTML, encoding="utf-8")
print("wrote", out_path)
