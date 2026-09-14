"""Interactive, playable CSI heatmap for one session -- a real answer to "let me scrub through the data
and see the label/time as I go," not a static PNG.

Each output HTML has two linked panels:
  - top: the WHOLE session as a heatmap (subcarrier x time-in-seconds), downsampled to a fixed number of
    time bins so it stays fast to render regardless of session length. Fully interactive: zoom, pan,
    hover shows exact (time, subcarrier, amplitude).
  - bottom: a "playback" heatmap of one real ~1-second (200-packet), full-resolution window at a time,
    driven by a slider + play/pause button (Plotly's native animation frames). A vertical line on the
    top panel tracks the current playback position, and the title shows the session's label/person/
    motion/date plus the current elapsed time -- so the label and timestamp are always on screen, not
    just implied by which file you opened.

Frame count is capped (~120) regardless of session length so the self-contained HTML stays a
reasonable size (a naive "one frame per second of the whole session" would produce 500+ frames / tens
of MB for the longest sessions) -- frames are evenly spaced across the session's full duration instead
of covering every single packet, which still gives a faithful sense of how the signal evolves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.visualization.palette import INTERACTIVE_DIR, SEQUENTIAL_BLUE
from ml.data_pipeline.windowing import cache_session, load_manifest

WINDOW_PACKETS = 200
WINDOW_DISPLAY_COLS = 100  # downsample each ~1s window for display -- see note below on file size
MAX_FRAMES = 60
OVERVIEW_BINS = 400

PLOTLY_BLUE_SCALE = [[i / (len(SEQUENTIAL_BLUE) - 1), c] for i, c in enumerate(SEQUENTIAL_BLUE)]


def _load_session_arrays(session_dir_rel: str) -> dict:
    cache_path = cache_session(session_dir_rel, "native")
    with np.load(cache_path) as d:
        return {"amplitude": d["amplitude"], "phase": d["phase"], "device_time_us": d["device_time_us"]}


def _overview_binned(values: np.ndarray, t_seconds: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """values: (n_packets, n_sub) -> (n_sub, n_bins) time-averaged, plus (n_bins,) bin center times."""
    n_packets = values.shape[0]
    n_bins = min(n_bins, n_packets)
    edges = np.linspace(0, n_packets, n_bins + 1).astype(int)
    binned = np.empty((values.shape[1], n_bins))
    bin_t = np.empty(n_bins)
    for i in range(n_bins):
        s, e = edges[i], max(edges[i] + 1, edges[i + 1])
        binned[:, i] = values[s:e].mean(axis=0)
        bin_t[i] = t_seconds[s:e].mean()
    return np.round(binned, 2), np.round(bin_t, 2)


def _frame_starts(n_packets: int, window_packets: int, max_frames: int) -> list[int]:
    if n_packets <= window_packets:
        return [0]
    last_start = n_packets - window_packets
    n_frames = min(max_frames, last_start + 1)
    return sorted(set(np.linspace(0, last_start, n_frames).astype(int).tolist()))


def build_session_player(session_dir_rel: str, manifest_row: pd.Series, channel: str = "amplitude") -> "go.Figure":
    arrays = _load_session_arrays(session_dir_rel)
    values = arrays[channel]
    device_time_us = arrays["device_time_us"]
    t_seconds = (device_time_us - device_time_us[0]) / 1e6
    n_packets, n_sub = values.shape

    overview, overview_t = _overview_binned(values, t_seconds, OVERVIEW_BINS)
    vmax = np.percentile(values, 99)

    label = manifest_row["label"]
    person = manifest_row["person_id"] if isinstance(manifest_row["person_id"], str) else ""
    motion = manifest_row["motion"] if isinstance(manifest_row["motion"], str) else ""
    who = f"{label}" + (f" ({person}, {motion})" if person else "")

    starts = _frame_starts(n_packets, WINDOW_PACKETS, MAX_FRAMES)

    def window_slice(start: int) -> np.ndarray:
        # Plotly encodes heatmap z-arrays as binary float32 (base64), not verbose JSON text, so the
        # only real lever on file size is DATA VOLUME, not decimal precision -- mean-pool each ~1s
        # window down to WINDOW_DISPLAY_COLS columns (a 26MB/session file at full 200-column
        # resolution x 120 frames was too slow to open comfortably; this brings it under ~6MB).
        end = min(start + WINDOW_PACKETS, n_packets)
        window = values[start:end]
        edges = np.linspace(0, len(window), WINDOW_DISPLAY_COLS + 1).astype(int)
        pooled = np.stack([window[edges[i]:max(edges[i] + 1, edges[i + 1])].mean(axis=0)
                            for i in range(WINDOW_DISPLAY_COLS)], axis=1)
        return pooled  # (n_sub, WINDOW_DISPLAY_COLS)

    fig = go.Figure()

    # Trace 0: overview heatmap (subcarrier x whole-session time)
    fig.add_trace(go.Heatmap(
        z=overview, x=overview_t, y=np.arange(n_sub), zmin=0, zmax=vmax, colorscale=PLOTLY_BLUE_SCALE,
        colorbar=dict(title=channel, x=1.02), hovertemplate="t=%{x:.2f}s<br>subcarrier=%{y}<br>value=%{z:.2f}<extra></extra>",
    ))
    # Trace 1: current-window playback heatmap (subcarrier x packet-within-window)
    fig.add_trace(go.Heatmap(
        z=window_slice(starts[0]), x=np.arange(WINDOW_DISPLAY_COLS), y=np.arange(n_sub), zmin=0, zmax=vmax,
        colorscale=PLOTLY_BLUE_SCALE, showscale=False, xaxis="x2", yaxis="y2",
        hovertemplate="packet in window (downsampled)=%{x}<br>subcarrier=%{y}<br>value=%{z:.2f}<extra></extra>",
    ))

    total_duration = t_seconds[-1]

    def marker_shape(start: int):
        t0 = t_seconds[start]
        return dict(type="line", xref="x", yref="paper", x0=t0, x1=t0, y0=0.60, y1=0.95,
                    line=dict(color="#e34948", width=2))

    def frame_title(start: int) -> str:
        t0 = t_seconds[start]
        return (f"{who} -- {session_dir_rel}<br>"
                f"<span style='font-size:13px'>playback t={t0:.2f}s / {total_duration:.1f}s "
                f"(window starts at packet {start}/{n_packets})</span>")

    frames = []
    for start in starts:
        frames.append(go.Frame(
            name=str(start),
            data=[go.Heatmap(z=window_slice(start))],
            traces=[1],
            layout=go.Layout(shapes=[marker_shape(start)], title=dict(text=frame_title(start))),
        ))
    fig.frames = frames

    fig.update_layout(
        shapes=[marker_shape(starts[0])],
        title=dict(text=frame_title(starts[0])),
        xaxis=dict(domain=[0.0, 1.0], anchor="y", title="session time (s)"),
        yaxis=dict(domain=[0.60, 0.95], title="subcarrier"),
        xaxis2=dict(domain=[0.0, 1.0], anchor="y2", title="packet within ~1s playback window"),
        yaxis2=dict(domain=[0.0, 0.40], title="subcarrier"),
        paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb", font=dict(color="#0b0b0b"),
        margin=dict(t=160, b=40),
        updatemenus=[dict(
            type="buttons", direction="left", x=0.0, y=1.28, xanchor="left",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame": {"duration": 300, "redraw": True}, "fromcurrent": True,
                                   "transition": {"duration": 0}}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.0, y=1.14, len=1.0,
            currentvalue=dict(prefix="window start packet: "),
            steps=[dict(label=str(s), method="animate",
                        args=[[str(s)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}])
                   for s in starts],
        )],
        annotations=[dict(text="full session (drag to zoom, hover for values)", x=0, y=0.96, xref="paper",
                           yref="paper", xanchor="left", yanchor="bottom", showarrow=False,
                           font=dict(size=11, color="#898781")),
                     dict(text="current playback window (red line above = position)", x=0, y=0.41, xref="paper",
                           yref="paper", xanchor="left", yanchor="bottom", showarrow=False,
                           font=dict(size=11, color="#898781"))],
    )
    return fig


def build_index(entries: list[dict], out_path) -> None:
    labels = sorted({e["label"] for e in entries})
    persons = sorted({e["person_id"] for e in entries if e["person_id"]})

    rows = "\n".join(
        f'<tr class="row" data-label="{e["label"]}" data-person="{e["person_id"]}" data-motion="{e["motion"]}">'
        f'<td><a href="{e["file"]}">{e["session_dir"]}</a></td><td>{e["label"]}</td><td>{e["person_id"]}</td>'
        f'<td>{e["motion"]}</td><td>{e["date"]}</td><td>{e["duration_s"]}</td></tr>'
        for e in entries
    )
    label_options = "\n".join(f'<option value="{l}">{l}</option>' for l in labels)
    person_options = "\n".join(f'<option value="{p}">{p}</option>' for p in persons)

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>CSI session players</title>
<style>
body {{ font-family: system-ui, -apple-system, sans-serif; background:#fcfcfb; color:#0b0b0b; padding:24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #e1e0d9; }}
th {{ color: #52514e; font-weight: 600; }}
a {{ color: #2a78d6; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
select, input {{ padding: 6px 10px; margin-right: 12px; border: 1px solid #e1e0d9; border-radius: 4px;
                  background: #fff; color: #0b0b0b; }}
.controls {{ margin-bottom: 16px; }}
.count {{ color: #898781; font-size: 13px; margin-bottom: 8px; }}
tr.hidden {{ display: none; }}
</style></head><body>
<h2>Interactive CSI session players</h2>
<p style="color:#52514e">Click a session to open its interactive heatmap: drag to zoom/pan the full session
(top), Play or drag the slider to scrub the ~1s playback window (bottom). Filter the list below first.</p>
<div class="controls">
  <select id="labelFilter"><option value="">all labels</option>{label_options}</select>
  <select id="personFilter"><option value="">all people</option>{person_options}</select>
  <input id="searchFilter" type="text" placeholder="search session_dir...">
</div>
<div class="count" id="count"></div>
<table><thead><tr><th>session</th><th>label</th><th>person</th><th>motion</th><th>date</th><th>duration (s)</th></tr></thead>
<tbody id="rows">
{rows}
</tbody></table>
<script>
const labelFilter = document.getElementById('labelFilter');
const personFilter = document.getElementById('personFilter');
const searchFilter = document.getElementById('searchFilter');
const countEl = document.getElementById('count');
const allRows = Array.from(document.querySelectorAll('#rows tr.row'));

function applyFilters() {{
  const label = labelFilter.value;
  const person = personFilter.value;
  const search = searchFilter.value.toLowerCase();
  let visible = 0;
  for (const row of allRows) {{
    const matches = (!label || row.dataset.label === label)
      && (!person || row.dataset.person === person)
      && (!search || row.textContent.toLowerCase().includes(search));
    row.classList.toggle('hidden', !matches);
    if (matches) visible++;
  }}
  countEl.textContent = `${{visible}} / ${{allRows.length}} sessions shown`;
}}
labelFilter.addEventListener('change', applyFilters);
personFilter.addEventListener('change', applyFilters);
searchFilter.addEventListener('input', applyFilters);
applyFilters();
</script>
</body></html>"""
    out_path.write_text(html)
    print(f"wrote {out_path}")


CURATED_PICKS = [
    ("authorized", "anjali"), ("authorized", "barath"),
    ("unauthorized", "divya"), ("unauthorized", "manas"), ("unauthorized", "kishore"),
]


def _curated_session_dirs(manifest: pd.DataFrame) -> list[str]:
    picks = []
    for label, person in CURATED_PICKS:
        rows = manifest[(manifest["label"] == label) & (manifest["person_id"] == person)]
        picks.append(rows.iloc[0]["session_dir"])
    picks.append(manifest[manifest["label"] == "none"].iloc[0]["session_dir"])
    return picks


def main(session_dirs: list[str] | None = None, channel: str = "amplitude", curated_only: bool = False) -> None:
    manifest = load_manifest()
    if session_dirs is None:
        session_dirs = _curated_session_dirs(manifest) if curated_only else manifest["session_dir"].tolist()

    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for session_dir in session_dirs:
        row = manifest[manifest["session_dir"] == session_dir].iloc[0]
        fig = build_session_player(session_dir, row, channel=channel)
        out_name = f"session_{session_dir.replace('/', '__')}.html"
        out_path = INTERACTIVE_DIR / out_name
        fig.write_html(out_path, include_plotlyjs="cdn")
        print(f"wrote {out_path}")
        entries.append({
            "file": out_name, "label": row["label"],
            "person_id": row["person_id"] if isinstance(row["person_id"], str) else "",
            "motion": row["motion"] if isinstance(row["motion"], str) else "",
            "date": session_dir.split("/")[1], "session_dir": session_dir,
            "duration_s": round(float(row["duration_s"]), 1),
        })

    entries.sort(key=lambda e: (e["label"], e["person_id"], e["session_dir"]))
    build_index(entries, INTERACTIVE_DIR / "index.html")
    print(f"\n{len(entries)} sessions -> open {INTERACTIVE_DIR / 'index.html'}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", action="append", help="specific session_dir(s) to render; default is ALL sessions")
    p.add_argument("--curated", action="store_true", help="render only a small representative subset (fast)")
    p.add_argument("--channel", default="amplitude", choices=["amplitude", "phase"])
    args = p.parse_args()
    main(session_dirs=args.session, channel=args.channel, curated_only=args.curated)
