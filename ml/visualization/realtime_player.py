"""Writes ONE static, data-free HTML file: visualizations/interactive/player.html. It has no embedded
session list and no per-session data files -- on load it connects to `player_server.py` over WebSocket,
asks for the session list, and when you click a session, asks the server to build that session's data
right then (a few hundred ms even for the longest session -- see player_server.py's docstring for why
nothing is pre-built ahead of time). Requires player_server.py running:

    python3 -m ml.visualization.player_server

Rendering design (see visualizations/README.md for the literature this is based on):
- Raw absolute amplitude with a single-hue colormap and a fixed global scale looked flat/uninformative.
  Fix: viridis colormap, per-session adaptive percentile contrast, and a live-tunable client-side
  preprocessing chain (Hampel filter -> Butterworth low-pass -> mean-subtract/z-score) -- all computed
  in the browser from the raw values the server sends, so every parameter is instantly tweakable
  without another round trip.
- Phase is unusable raw (wrapped, corrupted by CFO/SFO) -- "phase sanitize" unwraps and removes the
  linear trend across subcarriers per time step before it reaches the colormap.
- "Real-time" playback reveals a pre-rendered (client-side, from the current preprocessing settings)
  offscreen canvas at a rate tied to the session's actual duration, scaled by a speed multiplier.
- One WebSocket connection handles both the recorded-session request/response traffic AND the "Live
  (ESP32)" tab's streaming -- player_server.py is the single process behind both.
"""
from __future__ import annotations

from ml.visualization.palette import INTERACTIVE_DIR

# matplotlib's viridis, sampled to 32 stops -- perceptually uniform, unlike jet or a single hue.
# Baked in here (not computed at runtime) so this module has no matplotlib dependency.
VIRIDIS_LUT = [
    [68, 1, 84], [71, 13, 96], [72, 24, 106], [72, 35, 116], [71, 46, 124], [69, 56, 130],
    [66, 65, 134], [62, 74, 137], [58, 84, 140], [54, 93, 141], [50, 101, 142], [46, 109, 142],
    [43, 117, 142], [40, 125, 142], [37, 132, 142], [34, 140, 141], [31, 148, 140], [30, 156, 137],
    [32, 163, 134], [37, 171, 130], [46, 179, 124], [58, 186, 118], [72, 193, 110], [88, 199, 101],
    [108, 205, 90], [127, 211, 78], [147, 215, 65], [168, 219, 52], [192, 223, 37], [213, 226, 26],
    [234, 229, 26], [253, 231, 37],
]


def build_player_html(out_path) -> None:
    import json
    viridis_json = json.dumps(VIRIDIS_LUT)

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>CSI real-time player</title>
<style>
:root {{ --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9; --blue:#2a78d6;
         --orange:#eb6834; --aqua:#1baf7a; }}
* {{ box-sizing: border-box; }}
body {{ font-family: system-ui, -apple-system, sans-serif; background:var(--surface); color:var(--ink);
        margin:0; padding:20px; }}
h2 {{ margin: 0 0 4px; }}
.sub {{ color: var(--ink2); margin: 0 0 16px; font-size: 14px; }}
code {{ background:#f0efec; padding:1px 5px; border-radius:3px; font-size:12px; }}
.layout {{ display:flex; gap:20px; align-items:flex-start; }}
.list {{ width: 320px; flex-shrink:0; border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
.list-controls {{ padding:10px; border-bottom:1px solid var(--line); display:flex; gap:8px; }}
.list-controls select, .list-controls input {{ flex:1; padding:6px 8px; border:1px solid var(--line);
        border-radius:4px; font-size:13px; }}
.rows {{ max-height: 640px; overflow-y:auto; }}
.row {{ padding:10px 12px; border-bottom:1px solid var(--line); cursor:pointer; font-size:13px; }}
.row:hover {{ background:#f0efec; }}
.row.selected {{ background:#e8f0fb; border-left:3px solid var(--blue); }}
.row .label {{ font-weight:600; }}
.row .label.authorized {{ color:var(--blue); }}
.row .label.unauthorized {{ color:var(--orange); }}
.row .label.none {{ color:var(--aqua); }}
.row .meta {{ color:var(--muted); font-size:12px; margin-top:2px; }}
.player {{ flex:1; min-width:0; }}
.player-header {{ margin-bottom:10px; }}
.player-title {{ font-size:18px; font-weight:600; }}
.player-title .label.authorized {{ color:var(--blue); }}
.player-title .label.unauthorized {{ color:var(--orange); }}
.player-title .label.none {{ color:var(--aqua); }}
.canvas-wrap {{ border:1px solid var(--line); border-radius:8px; background:#fff; padding:10px; position:relative; }}
canvas {{ width:100%; height:520px; display:block; background:#1a1a1a; border-radius:4px; image-rendering:pixelated; }}
.loading-overlay {{ position:absolute; inset:10px; display:flex; align-items:center; justify-content:center;
        background:rgba(26,26,26,0.85); color:#fff; font-size:14px; border-radius:4px; }}
.controls {{ display:flex; align-items:center; gap:10px; margin-top:12px; flex-wrap:wrap; }}
button {{ padding:8px 16px; border:1px solid var(--line); border-radius:4px; background:#fff; color:var(--ink);
          cursor:pointer; font-size:14px; }}
button:hover {{ background:#f0efec; }}
button:disabled {{ opacity: 0.4; cursor: default; }}
select {{ padding:8px 10px; border:1px solid var(--line); border-radius:4px; font-size:14px; }}
.progress-wrap {{ flex:1; min-width:200px; }}
.progress {{ width:100%; height:8px; background:#e1e0d9; border-radius:4px; cursor:pointer; position:relative; }}
.progress-fill {{ height:100%; background:var(--blue); border-radius:4px; width:0%; }}
.time-label {{ color:var(--ink2); font-size:13px; white-space:nowrap; }}
.placeholder {{ color:var(--muted); padding:80px 20px; text-align:center; }}
.axis-note {{ color:var(--muted); font-size:12px; margin-top:6px; }}
.channel-toggle {{ display:flex; border:1px solid var(--line); border-radius:4px; overflow:hidden; }}
.channelBtn {{ border:none; border-radius:0; background:#fff; padding:8px 14px; }}
.channelBtn + .channelBtn {{ border-left:1px solid var(--line); }}
.channelBtn.active {{ background:var(--blue); color:#fff; }}
.top-tabs {{ display:flex; gap:8px; align-items:center; margin-bottom:16px; }}
.top-tabs button {{ padding:10px 20px; font-weight:600; }}
.top-tabs button.active {{ background:var(--blue); color:#fff; border-color:var(--blue); }}
.server-status {{ display:flex; align-items:center; gap:8px; margin-left:auto; font-size:13px; color:var(--ink2); }}
.dot {{ width:10px; height:10px; border-radius:50%; background:var(--muted); flex-shrink:0; }}
.dot.connected {{ background:#0ca30c; }}
.dot.error {{ background:#d03b3b; }}
.live-status {{ display:flex; align-items:center; gap:10px; margin-bottom:14px; font-size:14px; }}
.live-status input {{ padding:6px 10px; border:1px solid var(--line); border-radius:4px; font-size:13px; width:200px; }}
.live-canvas-wrap {{ border:1px solid var(--line); border-radius:8px; background:#fff; padding:10px; max-width:1100px; }}
.live-canvas-wrap canvas {{ background:#111; height:260px; }}
.live-stats {{ display:flex; gap:24px; margin-top:10px; color:var(--ink2); font-size:13px; }}
.live-stats b {{ color:var(--ink); }}
.preproc {{ border:1px solid var(--line); border-radius:8px; padding:12px 16px; margin-top:14px; }}
.preproc h4 {{ margin:0 0 10px; font-size:13px; text-transform:uppercase; letter-spacing:0.03em; color:var(--ink2); }}
.preproc-row {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:8px; }}
.preproc-row label {{ font-size:13px; display:flex; align-items:center; gap:6px; min-width:150px; }}
.preproc-row input[type=range] {{ width:140px; }}
.preproc-row input[type=number] {{ width:60px; padding:4px 6px; border:1px solid var(--line); border-radius:4px; }}
.preproc-val {{ color:var(--ink2); font-size:12px; min-width:70px; }}
.preproc select {{ padding:6px 8px; font-size:13px; }}
</style></head><body>
<h2>CSI real-time player</h2>
<div class="top-tabs">
  <button id="tabRecorded" class="active">Recorded sessions</button>
  <button id="tabLive">&#9679; Live (ESP32)</button>
  <div class="server-status">
    <span class="dot" id="serverDot"></span>
    <span id="serverStatusText">connecting to player_server...</span>
    <input id="serverUrl" value="ws://localhost:8765" style="width:170px;padding:4px 8px;border:1px solid var(--line);border-radius:4px;font-size:12px;">
    <button id="serverReconnectBtn" style="padding:4px 10px;font-size:12px;">Reconnect</button>
  </div>
</div>

<div id="recordedView">
<p class="sub">Pick a session on the left. Its data is built by <code>player_server.py</code> on demand
right when you click it (not pre-built) -- give it a moment before scrubbing to the end. Preprocessing
below applies live once loaded. Open this same file in a second tab and pick a different session to
compare side by side (both talk to the same server).</p>
<div class="layout">
  <div class="list">
    <div class="list-controls">
      <select id="labelFilter"><option value="">all labels</option></select>
      <select id="personFilter"><option value="">all people</option></select>
      <select id="motionFilter"><option value="">all motion</option></select>
    </div>
    <div class="rows" id="rows"><div style="padding:16px; color:var(--muted); font-size:13px;">waiting for player_server...</div></div>
  </div>
  <div class="player">
    <div id="empty" class="canvas-wrap placeholder">Select a session on the left to begin.</div>
    <div id="playerBody" style="display:none;">
      <div class="player-header">
        <div class="player-title" id="title"></div>
        <div class="sub" id="subtitle" style="margin:2px 0 0;"></div>
      </div>
      <div class="canvas-wrap">
        <canvas id="canvas"></canvas>
        <div class="loading-overlay" id="loadingOverlay" style="display:none;">building session data on the server...</div>
        <div class="axis-note">subcarrier 0 (top) &rarr; 185 (bottom) &nbsp;|&nbsp;
          <span id="channelNote">amplitude</span>, viridis colormap, per-session adaptive contrast</div>
      </div>
      <div class="controls">
        <div class="channel-toggle" id="channelToggle">
          <button class="channelBtn active" data-channel="amplitude">amplitude</button>
          <button class="channelBtn" data-channel="phase">phase</button>
        </div>
        <button id="playBtn">&#9654; Play</button>
        <button id="resetBtn">&#8634; Reset</button>
        <select id="speed">
          <option value="1">1x (real time)</option>
          <option value="2">2x</option>
          <option value="5" selected>5x</option>
          <option value="10">10x</option>
          <option value="20">20x</option>
          <option value="50">50x</option>
        </select>
        <div class="progress-wrap">
          <div class="progress" id="progress"><div class="progress-fill" id="progressFill"></div></div>
        </div>
        <div class="time-label" id="timeLabel">0.0s / 0.0s</div>
      </div>

      <div class="preproc">
        <h4>Preprocessing (applied in this order, live)</h4>
        <div class="preproc-row">
          <label><input type="checkbox" id="phaseSanitizeOn" checked> Phase sanitize (unwrap + linear detrend across subcarriers)</label>
          <span class="preproc-val" id="phaseSanitizeNote">phase channel only</span>
        </div>
        <div class="preproc-row">
          <label><input type="checkbox" id="hampelOn" checked> Hampel filter</label>
          window <input type="number" id="hampelWindow" min="3" max="51" value="10">
          n&sigma; <input type="range" id="hampelSigma" min="1" max="6" step="0.5" value="3">
          <span class="preproc-val" id="hampelSigmaVal">3.0</span>
        </div>
        <div class="preproc-row">
          <label><input type="checkbox" id="butterOn" checked> Butterworth low-pass</label>
          cutoff <input type="range" id="butterCutoff" min="0.05" max="0.95" step="0.05" value="0.3">
          <span class="preproc-val" id="butterCutoffVal">0.30 &times; Nyquist</span>
        </div>
        <div class="preproc-row">
          <label>Normalize</label>
          <select id="normalizeMode">
            <option value="none">none (raw)</option>
            <option value="mean" selected>mean-subtract (detrend)</option>
            <option value="zscore">z-score</option>
          </select>
        </div>
        <div class="preproc-row">
          <label>Contrast percentile</label>
          low <input type="number" id="contrastLo" min="0" max="49" value="2">%
          high <input type="number" id="contrastHi" min="51" max="100" value="98">%
        </div>
      </div>
    </div>
  </div>
</div>
</div>

<div id="liveView" style="display:none;">
  <p class="sub">Uses the same connection to <code>player_server.py</code> to scroll a live ESP32 CSI
  stream in real time -- if an ESP32 is actually connected and the server was started against real
  hardware (not <code>--simulate</code>). Preview only, nothing is saved here; use
  <code>python3 -m collector.cli_collect ...</code> separately to record a session (the two can't run
  at the same time -- they'd compete for the same serial/TCP port).</p>
  <div class="live-canvas-wrap">
    <canvas id="liveCanvas" width="900" height="200"></canvas>
    <div class="axis-note">subcarrier 0 (top) &rarr; ~185 (bottom) &nbsp;|&nbsp; scrolling right-to-left, most recent packet at the right edge &nbsp;|&nbsp; viridis colormap, rolling adaptive contrast</div>
  </div>
  <div class="live-stats">
    <div>rate: <b id="liveRate">0</b> Hz</div>
    <div>rssi: <b id="liveRssi">--</b> dBm</div>
    <div>seq: <b id="liveSeq">--</b></div>
    <div>packets received: <b id="liveCount">0</b></div>
  </div>
</div>
<script>
const VIRIDIS = {viridis_json};

function viridisColor(t) {{
  t = Math.max(0, Math.min(1, t));
  const idx = t * (VIRIDIS.length - 1);
  const i0 = Math.floor(idx), i1 = Math.min(VIRIDIS.length - 1, i0 + 1);
  const f = idx - i0;
  const c0 = VIRIDIS[i0], c1 = VIRIDIS[i1];
  return [
    Math.round(c0[0] + (c1[0] - c0[0]) * f),
    Math.round(c0[1] + (c1[1] - c0[1]) * f),
    Math.round(c0[2] + (c1[2] - c0[2]) * f),
  ];
}}

// --- Preprocessing primitives ---
function unwrap1D(arr) {{
  const out = arr.slice();
  for (let i = 1; i < out.length; i++) {{
    let diff = out[i] - out[i - 1];
    while (diff > Math.PI) {{ out[i] -= 2 * Math.PI; diff = out[i] - out[i - 1]; }}
    while (diff < -Math.PI) {{ out[i] += 2 * Math.PI; diff = out[i] - out[i - 1]; }}
  }}
  return out;
}}

function hampelFilter1D(arr, windowSize, nSigma) {{
  // Pre-allocated typed-array scratch buffers, reused every point, instead of the naive version's
  // 4 new array allocations (slice/slice/sort/map/sort) PER POINT -- that was the actual bottleneck
  // (~5s for a 7500x186 grid), not the O(n*w log w) comparison count itself.
  const n = arr.length;
  const out = Float64Array.from(arr);
  const half = Math.floor(windowSize / 2);
  const scratch = new Float64Array(windowSize);
  const devScratch = new Float64Array(windowSize);
  for (let i = 0; i < n; i++) {{
    const lo = Math.max(0, i - half), hi = Math.min(n, i + half + 1);
    const len = hi - lo;
    for (let k = 0; k < len; k++) scratch[k] = arr[lo + k];
    const windowView = scratch.subarray(0, len);
    windowView.sort();  // TypedArray.sort() defaults to numeric ascending, unlike Array.sort()
    const median = windowView[len >> 1];
    for (let k = 0; k < len; k++) devScratch[k] = Math.abs(arr[lo + k] - median);
    const devView = devScratch.subarray(0, len);
    devView.sort();
    const mad = devView[len >> 1];
    const threshold = nSigma * 1.4826 * mad;
    if (Math.abs(arr[i] - median) > threshold) out[i] = median;
  }}
  return out;
}}

function butterworthLowpass1D(arr, cutoffFraction) {{
  const w0 = Math.PI * Math.max(0.01, Math.min(0.99, cutoffFraction));
  const cosw0 = Math.cos(w0), sinw0 = Math.sin(w0);
  const Q = Math.SQRT1_2;
  const alpha = sinw0 / (2 * Q);
  let b0 = (1 - cosw0) / 2, b1 = 1 - cosw0, b2 = (1 - cosw0) / 2;
  const a0 = 1 + alpha, a1 = -2 * cosw0, a2 = 1 - alpha;
  b0 /= a0; b1 /= a0; b2 /= a0;
  const na1 = a1 / a0, na2 = a2 / a0;
  const out = new Array(arr.length);
  let x1 = arr[0], x2 = arr[0], y1 = arr[0], y2 = arr[0];
  for (let i = 0; i < arr.length; i++) {{
    const x0 = arr[i];
    const y0 = b0 * x0 + b1 * x1 + b2 * x2 - na1 * y1 - na2 * y2;
    out[i] = y0;
    x2 = x1; x1 = x0; y2 = y1; y1 = y0;
  }}
  return out;
}}

function applyPreprocessing(rawValues, width, height, channel, opts) {{
  const rows = [];
  for (let r = 0; r < height; r++) rows.push(Float64Array.from(rawValues.slice(r * width, (r + 1) * width)));

  if (channel === 'phase' && opts.phaseSanitize) {{
    for (let c = 0; c < width; c++) {{
      let col = [];
      for (let r = 0; r < height; r++) col.push(rows[r][c]);
      col = unwrap1D(col);
      const n = height;
      let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
      for (let r = 0; r < n; r++) {{ sumX += r; sumY += col[r]; sumXY += r * col[r]; sumXX += r * r; }}
      const denom = (n * sumXX - sumX * sumX) || 1e-9;
      const a = (n * sumXY - sumX * sumY) / denom;
      const b = (sumY - a * sumX) / n;
      for (let r = 0; r < n; r++) rows[r][c] = col[r] - (a * r + b);
    }}
  }}

  if (opts.hampelOn) {{
    for (let r = 0; r < height; r++) rows[r] = hampelFilter1D(rows[r], opts.hampelWindow, opts.hampelSigma);
  }}
  if (opts.butterOn) {{
    for (let r = 0; r < height; r++) rows[r] = butterworthLowpass1D(rows[r], opts.butterCutoff);
  }}
  if (opts.normalize === 'mean' || opts.normalize === 'zscore') {{
    for (let r = 0; r < height; r++) {{
      let sum = 0; for (let c = 0; c < width; c++) sum += rows[r][c];
      const mean = sum / width;
      if (opts.normalize === 'mean') {{
        for (let c = 0; c < width; c++) rows[r][c] -= mean;
      }} else {{
        let sq = 0; for (let c = 0; c < width; c++) sq += (rows[r][c] - mean) ** 2;
        const std = Math.sqrt(sq / width) || 1e-6;
        for (let c = 0; c < width; c++) rows[r][c] = (rows[r][c] - mean) / std;
      }}
    }}
  }}

  const flat = new Float64Array(width * height);
  for (let r = 0; r < height; r++) for (let c = 0; c < width; c++) flat[r * width + c] = rows[r][c];
  return flat;
}}

function percentileRange(flat, loPct, hiPct) {{
  const sorted = Float64Array.from(flat).sort();
  const lo = sorted[Math.max(0, Math.floor(sorted.length * loPct / 100))];
  const hi = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * hiPct / 100))];
  return [lo, Math.max(hi, lo + 1e-6)];
}}

function renderToOffscreen(flat, width, height, loPct, hiPct) {{
  const [lo, hi] = percentileRange(flat, loPct, hiPct);
  const range = hi - lo;
  const off = document.createElement('canvas');
  off.width = width; off.height = height;
  const octx = off.getContext('2d');
  const imgData = octx.createImageData(width, height);
  for (let i = 0; i < flat.length; i++) {{
    const t = (flat[i] - lo) / range;
    const [red, green, blue] = viridisColor(t);
    imgData.data[i * 4] = red; imgData.data[i * 4 + 1] = green; imgData.data[i * 4 + 2] = blue; imgData.data[i * 4 + 3] = 255;
  }}
  octx.putImageData(imgData, 0, 0);
  return off;
}}

// --- Shared WebSocket connection to player_server.py (recorded-session requests + live streaming) ---
const serverDot = document.getElementById('serverDot');
const serverStatusText = document.getElementById('serverStatusText');
const serverUrlInput = document.getElementById('serverUrl');
const serverReconnectBtn = document.getElementById('serverReconnectBtn');

let ws = null;
let sessions = [];
const sessionDataCache = {{}}; // "session_dir|channel" -> {{width, height, duration_s, values}}
let pendingLoadKey = null;

function setServerStatus(cls, text) {{
  serverDot.className = 'dot ' + cls;
  serverStatusText.textContent = text;
}}

function connectServer() {{
  if (ws) ws.close();
  setServerStatus('', 'connecting to player_server...');
  ws = new WebSocket(serverUrlInput.value);
  ws.onopen = () => {{
    setServerStatus('connected', 'connected to player_server');
    ws.send(JSON.stringify({{type: 'list_sessions'}}));
  }};
  ws.onclose = () => setServerStatus('', 'disconnected -- is player_server.py running?');
  ws.onerror = () => setServerStatus('error', 'connection error -- start: python3 -m ml.visualization.player_server');
  ws.onmessage = (event) => {{
    const msg = JSON.parse(event.data);
    if (msg.type === 'session_list') {{
      sessions = msg.sessions;
      populateFilterOptions();
      renderRows();
    }} else if (msg.type === 'session_data') {{
      const key = msg.session_dir + '|' + msg.channel;
      sessionDataCache[key] = {{width: msg.width, height: msg.height, duration_s: msg.duration_s, values: msg.values}};
      if (key === pendingLoadKey) onSessionDataReady(key);
    }} else if (msg.type === 'session_data_error') {{
      if (loadingOverlay) {{ loadingOverlay.style.display = 'flex'; loadingOverlay.textContent = 'error: ' + msg.message; }}
    }} else if (msg.type === 'error') {{
      setLiveStatus('error', msg.message);
    }} else if (msg.type === 'status') {{
      setLiveStatus('connected', msg.message);
    }} else if (msg.type === 'sample') {{
      handleLiveSample(msg);
    }}
  }};
}}
serverReconnectBtn.addEventListener('click', connectServer);

// --- Recorded sessions ---
const rowsEl = document.getElementById('rows');
const labelFilter = document.getElementById('labelFilter');
const personFilter = document.getElementById('personFilter');
const motionFilter = document.getElementById('motionFilter');
const channelToggle = document.getElementById('channelToggle');
const channelNote = document.getElementById('channelNote');
const emptyEl = document.getElementById('empty');
const playerBody = document.getElementById('playerBody');
const loadingOverlay = document.getElementById('loadingOverlay');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const playBtn = document.getElementById('playBtn');
const resetBtn = document.getElementById('resetBtn');
const speedSelect = document.getElementById('speed');
const progress = document.getElementById('progress');
const progressFill = document.getElementById('progressFill');
const timeLabel = document.getElementById('timeLabel');
const titleEl = document.getElementById('title');
const subtitleEl = document.getElementById('subtitle');

const phaseSanitizeOn = document.getElementById('phaseSanitizeOn');
const hampelOn = document.getElementById('hampelOn');
const hampelWindow = document.getElementById('hampelWindow');
const hampelSigma = document.getElementById('hampelSigma');
const hampelSigmaVal = document.getElementById('hampelSigmaVal');
const butterOn = document.getElementById('butterOn');
const butterCutoff = document.getElementById('butterCutoff');
const butterCutoffVal = document.getElementById('butterCutoffVal');
const normalizeMode = document.getElementById('normalizeMode');
const contrastLo = document.getElementById('contrastLo');
const contrastHi = document.getElementById('contrastHi');

function populateFilterOptions() {{
  const labels = [...new Set(sessions.map(s => s.label))].sort();
  const persons = [...new Set(sessions.map(s => s.person_id).filter(Boolean))].sort();
  const motions = [...new Set(sessions.map(s => s.motion).filter(Boolean))].sort();
  labelFilter.innerHTML = '<option value="">all labels</option>' + labels.map(l => `<option value="${{l}}">${{l}}</option>`).join('');
  personFilter.innerHTML = '<option value="">all people</option>' + persons.map(p => `<option value="${{p}}">${{p}}</option>`).join('');
  motionFilter.innerHTML = '<option value="">all motion</option>' + motions.map(m => `<option value="${{m}}">${{m}}</option>`).join('');
}}

let selected = null;
let currentChannel = 'amplitude';
let rawData = null;
let offscreenCanvas = null;
let playing = false;
let elapsed = 0;
let lastFrameTs = null;
let rafId = null;
let recomputeTimer = null;

function renderRows() {{
  const lf = labelFilter.value, pf = personFilter.value, mf = motionFilter.value;
  rowsEl.innerHTML = '';
  if (sessions.length === 0) {{
    rowsEl.innerHTML = '<div style="padding:16px; color:var(--muted); font-size:13px;">no sessions (check player_server.py is running)</div>';
    return;
  }}
  for (const s of sessions) {{
    if (lf && s.label !== lf) continue;
    if (pf && s.person_id !== pf) continue;
    if (mf && s.motion !== mf) continue;
    const div = document.createElement('div');
    div.className = 'row' + (selected && selected.session_dir === s.session_dir ? ' selected' : '');
    div.innerHTML = `<div class="label ${{s.label}}">${{s.label}}${{s.person_id ? ' &middot; ' + s.person_id : ''}}</div>
      <div class="meta">${{s.date}} ${{s.time}} &middot; ${{s.motion || 'n/a'}} &middot; ${{s.duration_s}}s</div>`;
    div.addEventListener('click', () => selectSession(s));
    rowsEl.appendChild(div);
  }}
}}
labelFilter.addEventListener('change', renderRows);
personFilter.addEventListener('change', renderRows);
motionFilter.addEventListener('change', renderRows);

function collectOptions() {{
  return {{
    phaseSanitize: phaseSanitizeOn.checked,
    hampelOn: hampelOn.checked,
    hampelWindow: parseInt(hampelWindow.value, 10),
    hampelSigma: parseFloat(hampelSigma.value),
    butterOn: butterOn.checked,
    butterCutoff: parseFloat(butterCutoff.value),
    normalize: normalizeMode.value,
    loPct: parseFloat(contrastLo.value),
    hiPct: parseFloat(contrastHi.value),
  }};
}}

function recomputeAndRender() {{
  if (!rawData) return;
  const opts = collectOptions();
  const processed = applyPreprocessing(rawData.values, rawData.width, rawData.height, currentChannel, opts);
  offscreenCanvas = renderToOffscreen(processed, rawData.width, rawData.height, opts.loPct, opts.hiPct);
  canvas.width = rawData.width;
  canvas.height = rawData.height;
  draw();
}}

function scheduleRecompute() {{
  clearTimeout(recomputeTimer);
  recomputeTimer = setTimeout(recomputeAndRender, 120);
}}

[phaseSanitizeOn, hampelOn, butterOn].forEach(el => el.addEventListener('change', scheduleRecompute));
[hampelWindow, hampelSigma, butterCutoff, contrastLo, contrastHi].forEach(el => el.addEventListener('input', () => {{
  hampelSigmaVal.textContent = parseFloat(hampelSigma.value).toFixed(1);
  butterCutoffVal.textContent = parseFloat(butterCutoff.value).toFixed(2) + ' \\u00d7 Nyquist';
  scheduleRecompute();
}}));
normalizeMode.addEventListener('change', scheduleRecompute);

channelToggle.querySelectorAll('.channelBtn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    currentChannel = btn.dataset.channel;
    channelToggle.querySelectorAll('.channelBtn').forEach(b => b.classList.toggle('active', b === btn));
    channelNote.textContent = currentChannel;
    if (selected) requestSessionData(selected);
  }});
}});

function onSessionDataReady(key) {{
  rawData = sessionDataCache[key];
  loadingOverlay.style.display = 'none';
  recomputeAndRender();
}}

function requestSessionData(s) {{
  const key = s.session_dir + '|' + currentChannel;
  pendingLoadKey = key;
  if (sessionDataCache[key]) {{
    onSessionDataReady(key);
    return;
  }}
  loadingOverlay.style.display = 'flex';
  loadingOverlay.textContent = 'building session data on the server...';
  rawData = null;
  if (ws && ws.readyState === WebSocket.OPEN) {{
    ws.send(JSON.stringify({{type: 'load_session', session_dir: s.session_dir, channel: currentChannel}}));
  }} else {{
    loadingOverlay.textContent = 'not connected to player_server -- click Reconnect above';
  }}
}}

function selectSession(s) {{
  pause();
  selected = s;
  elapsed = 0;
  emptyEl.style.display = 'none';
  playerBody.style.display = 'block';
  const who = s.person_id ? `${{s.label}} (${{s.person_id}}, ${{s.motion || 'n/a'}})` : s.label;
  titleEl.innerHTML = `<span class="label ${{s.label}}">${{who}}</span>`;
  subtitleEl.textContent = `${{s.session_dir}}  --  ${{s.date}} ${{s.time}}`;
  requestSessionData(s);
  updateTimeLabel();
  renderRows();
}}

function draw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#1a1a1a';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!selected || !offscreenCanvas) return;
  const frac = Math.min(1, elapsed / selected.duration_s);
  const revealPx = Math.max(1, Math.round(frac * canvas.width));
  ctx.drawImage(offscreenCanvas, 0, 0, revealPx, offscreenCanvas.height, 0, 0, revealPx, canvas.height);
  if (frac < 1) {{
    ctx.strokeStyle = '#e34948';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(revealPx, 0);
    ctx.lineTo(revealPx, canvas.height);
    ctx.stroke();
  }}
  progressFill.style.width = (frac * 100) + '%';
}}

function updateTimeLabel() {{
  const dur = selected ? selected.duration_s : 0;
  timeLabel.textContent = `${{elapsed.toFixed(1)}}s / ${{dur.toFixed(1)}}s`;
}}

function step(ts) {{
  if (lastFrameTs === null) lastFrameTs = ts;
  const dtSeconds = (ts - lastFrameTs) / 1000;
  lastFrameTs = ts;
  const speed = parseFloat(speedSelect.value);
  elapsed = Math.min(selected.duration_s, elapsed + dtSeconds * speed);
  draw();
  updateTimeLabel();
  if (elapsed >= selected.duration_s) {{
    pause();
    return;
  }}
  rafId = requestAnimationFrame(step);
}}

function play() {{
  if (!selected || playing) return;
  playing = true;
  lastFrameTs = null;
  playBtn.innerHTML = '&#10074;&#10074; Pause';
  rafId = requestAnimationFrame(step);
}}
function pause() {{
  playing = false;
  playBtn.innerHTML = '&#9654; Play';
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;
  lastFrameTs = null;
}}
playBtn.addEventListener('click', () => playing ? pause() : play());
resetBtn.addEventListener('click', () => {{ pause(); elapsed = 0; draw(); updateTimeLabel(); }});

progress.addEventListener('click', (e) => {{
  if (!selected) return;
  const rect = progress.getBoundingClientRect();
  const frac = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
  elapsed = frac * selected.duration_s;
  draw();
  updateTimeLabel();
}});

// --- Live (ESP32) tab ---
const tabRecorded = document.getElementById('tabRecorded');
const tabLive = document.getElementById('tabLive');
const recordedView = document.getElementById('recordedView');
const liveView = document.getElementById('liveView');
const liveCanvas = document.getElementById('liveCanvas');
const liveCtx = liveCanvas.getContext('2d');
const liveRateEl = document.getElementById('liveRate');
const liveRssiEl = document.getElementById('liveRssi');
const liveSeqEl = document.getElementById('liveSeq');
const liveCountEl = document.getElementById('liveCount');

let liveCount = 0, liveRatePackets = 0, liveRateWindowStart = performance.now(), liveNSub = null;
let liveRecentMin = Infinity, liveRecentMax = -Infinity, liveRangeResetCounter = 0;

function setLiveStatus(cls, text) {{
  if (tabLive.classList.contains('active')) setServerStatus(cls, text);
}}

function livePushColumn(amplitude) {{
  if (liveNSub === null) {{ liveNSub = amplitude.length; liveCanvas.height = liveNSub; }}
  liveRangeResetCounter++;
  if (liveRangeResetCounter > 400) {{ liveRecentMin = Infinity; liveRecentMax = -Infinity; liveRangeResetCounter = 0; }}
  for (const v of amplitude) {{ if (v < liveRecentMin) liveRecentMin = v; if (v > liveRecentMax) liveRecentMax = v; }}
  const range = Math.max(liveRecentMax - liveRecentMin, 1e-6);

  const img = liveCtx.getImageData(1, 0, liveCanvas.width - 1, liveCanvas.height);
  liveCtx.putImageData(img, 0, 0);
  for (let y = 0; y < liveNSub; y++) {{
    const t = (amplitude[y] - liveRecentMin) / range;
    const [r, g, b] = viridisColor(t);
    liveCtx.fillStyle = `rgb(${{r}},${{g}},${{b}})`;
    liveCtx.fillRect(liveCanvas.width - 1, y, 1, 1);
  }}
}}

function handleLiveSample(msg) {{
  // Skip the actual canvas work (getImageData/putImageData is not cheap) while the Live tab isn't
  // visible -- this was competing for main-thread time with recorded-session preprocessing whenever
  // a live source (even --simulate) was running in the background, adding real, measured delay.
  if (!tabLive.classList.contains('active')) return;
  if (liveCount === 0) setLiveStatus('connected', 'streaming live data');
  livePushColumn(msg.amplitude);
  liveSeqEl.textContent = msg.seq;
  liveRssiEl.textContent = msg.rssi;
  liveCount++; liveRatePackets++;
  liveCountEl.textContent = liveCount;
  const now = performance.now();
  if (now - liveRateWindowStart >= 1000) {{
    liveRateEl.textContent = liveRatePackets;
    liveRatePackets = 0;
    liveRateWindowStart = now;
  }}
}}

tabRecorded.addEventListener('click', () => {{
  tabRecorded.classList.add('active');
  tabLive.classList.remove('active');
  recordedView.style.display = 'block';
  liveView.style.display = 'none';
}});
tabLive.addEventListener('click', () => {{
  tabLive.classList.add('active');
  tabRecorded.classList.remove('active');
  recordedView.style.display = 'none';
  liveView.style.display = 'block';
}});

connectServer();
</script>
</body></html>"""
    out_path.write_text(html)
    print(f"wrote {out_path}")


def main() -> None:
    build_player_html(INTERACTIVE_DIR / "player.html")
    print("Start the server separately: python3 -m ml.visualization.player_server")


if __name__ == "__main__":
    main()
