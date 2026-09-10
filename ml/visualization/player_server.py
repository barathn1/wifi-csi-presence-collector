"""The one server behind player.html -- handles BOTH of its jobs over a single WebSocket:

1. On-demand recorded-session data: player.html ships with NO embedded session data at all (a single
   ~30KB static file, not one-file-per-session). When you click a session, the browser asks this server
   for it right then, the server decodes+downsamples it (a few hundred ms even for the longest session,
   since `cache_session()` caches the raw decode after the first request), and sends it back. Nothing
   is pre-built ahead of time -- exactly what was asked for after the first version pre-generated 346MB
   of per-session files whether you ever looked at them or not.
2. Live ESP32 streaming (unchanged from the old live_server.py): bridges a live-connected board to the
   browser for real-time visualization. Never writes session files -- use
   `python3 -m collector.cli_collect` for that. The ESP32 transport can only be held by one process at a
   time, so stop this before a real recording session (or vice versa).

    python3 -m ml.visualization.player_server                # real hardware, via config/config.yaml
    python3 -m ml.visualization.player_server --simulate      # synthetic live data, no hardware needed

Then open visualizations/interactive/player.html -- it connects here automatically for the session list
and on-demand data, and its "Live (ESP32)" tab uses the same connection for live streaming.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import queue
import threading
import time

import numpy as np

from ml.data_pipeline.windowing import DOMINANT_CSI_LEN, load_manifest
from ml.visualization.session_data import build_session_grid

try:
    import websockets
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install websockets (added to ml/requirements.txt)") from e


# --- On-demand recorded-session data ---

def build_session_list() -> list[dict]:
    import pandas as pd

    manifest = load_manifest()
    entries = []
    for _, row in manifest.iterrows():
        session_dir = row["session_dir"]
        start_ts = pd.to_datetime(row["start_ts"], unit="s", utc=True).tz_convert("Asia/Kolkata")
        entries.append({
            "session_dir": session_dir, "label": row["label"],
            "person_id": row["person_id"] if isinstance(row["person_id"], str) else "",
            "motion": row["motion"] if isinstance(row["motion"], str) else "",
            "date": session_dir.split("/")[1], "time": start_ts.strftime("%H:%M:%S"),
            "duration_s": round(float(row["duration_s"]), 1),
        })
    entries.sort(key=lambda e: (e["label"], e["person_id"], e["session_dir"]))
    return entries


def _build_session_payload(session_dir: str, channel: str, duration_s: float) -> str:
    """Runs entirely in a worker thread (see handle_request) -- grid build, rounding, AND json.dumps,
    so none of it blocks the asyncio event loop."""
    grid = build_session_grid(session_dir, channel, duration_s)
    height, width = grid.shape
    flat = np.round(grid.astype(np.float64), 1).ravel().tolist()
    return json.dumps({
        "type": "session_data", "session_dir": session_dir, "channel": channel,
        "width": width, "height": height, "duration_s": duration_s, "values": flat,
    })


async def handle_request(websocket, request: dict) -> None:
    req_type = request.get("type")

    if req_type == "list_sessions":
        await websocket.send(json.dumps({"type": "session_list", "sessions": build_session_list()}))
        return

    if req_type == "load_session":
        session_dir, channel = request.get("session_dir"), request.get("channel")
        try:
            manifest = load_manifest()
            row = manifest[manifest["session_dir"] == session_dir].iloc[0]
            duration_s = float(row["duration_s"])
            loop = asyncio.get_running_loop()
            # grid-build + rounding + json.dumps all in the executor thread, not just the grid build --
            # json.dumps on ~1.4M floats alone took ~500ms measured, which would otherwise run
            # synchronously on the asyncio event loop and block every other connection (including the
            # live-sample broadcast) for that whole time.
            payload = await loop.run_in_executor(None, _build_session_payload, session_dir, channel, duration_s)
            await websocket.send(payload)
        except Exception as e:  # noqa: BLE001 -- report to the browser, don't crash the server
            await websocket.send(json.dumps({
                "type": "session_data_error", "session_dir": session_dir, "message": str(e),
            }))
        return


# --- Live ESP32 streaming (unchanged behavior from the original live_server.py) ---

def decode_csi_data(csi_data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Same (imag,real) int8 pair decode as ml/data_pipeline/decode_csi.py, applied to one live packet."""
    raw = np.frombuffer(csi_data, dtype=np.int8).astype(np.float32)
    imag, real = raw[0::2], raw[1::2]
    return np.hypot(real, imag), np.arctan2(imag, real)


class SimulatedSample:
    """Stand-in for collector.models.CsiSample when --simulate is used -- same fields the real receiver
    produces, filled with a slowly-drifting synthetic signal so the browser/WebSocket plumbing can be
    exercised and demoed without real hardware attached."""

    def __init__(self, seq: int, n_sub: int = DOMINANT_CSI_LEN // 2):
        t = seq / 200.0
        base = 15 + 8 * np.sin(np.linspace(0, 3, n_sub) + t * 0.5)
        noise = np.random.default_rng(seq).normal(0, 1.5, n_sub)
        amp = np.clip(base + noise, 0, None)
        phase = np.random.default_rng(seq + 1).uniform(-np.pi, np.pi, n_sub)
        real = (amp * np.cos(phase)).astype(np.int8)
        imag = (amp * np.sin(phase)).astype(np.int8)
        interleaved = np.empty(n_sub * 2, dtype=np.int8)
        interleaved[0::2] = imag
        interleaved[1::2] = real
        self.seq = seq
        self.device_time_us = int(t * 1e6)
        self.rssi = -55 + int(np.random.default_rng(seq + 2).integers(-3, 3))
        self.csi_len = DOMINANT_CSI_LEN
        self.csi_data = interleaved.tobytes()


def simulate_samples(stop_event: threading.Event, rate_hz: float = 200.0):
    seq = 0
    period = 1.0 / rate_hz
    next_tick = time.monotonic()
    while not stop_event.is_set():
        yield SimulatedSample(seq)
        seq += 1
        next_tick += period
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)


def receiver_thread_main(sample_queue: "queue.Queue", stop_event: threading.Event, simulate: bool) -> None:
    def emit(msg_type: str, message: str) -> None:
        print(f"[player_server] {msg_type}: {message}", flush=True)
        sample_queue.put({"type": msg_type, "message": message})

    source = None
    try:
        if simulate:
            emit("status", "simulated data source (no hardware) -- for demo/dev only")
            source = simulate_samples(stop_event)
        else:
            from collector import preflight
            from collector.config import load_config
            from collector.receiver import get_receiver

            cfg = load_config()
            ok, reason = preflight.check_board_connected(cfg)
            if not ok:
                emit("error", f"board not connected: {reason}")
                return
            emit("status", f"connected: {reason}")
            source = get_receiver(cfg, stop_event=stop_event)
    except Exception as e:  # noqa: BLE001 -- e.g. missing dependency, bad config -- report, don't crash silently
        emit("error", f"could not start receiver: {e}")
        return

    try:
        for sample in source:
            if sample.csi_len != DOMINANT_CSI_LEN:
                continue  # only the dominant frame type -- consistent with the rest of the pipeline
            amplitude, _phase = decode_csi_data(sample.csi_data)
            sample_queue.put({
                "type": "sample", "seq": sample.seq, "rssi": sample.rssi,
                "amplitude": np.round(amplitude, 1).tolist(),
            })
    except Exception as e:  # noqa: BLE001 -- surface any transport error to the browser instead of dying silently
        sample_queue.put({"type": "error", "message": f"receiver stopped: {e}"})
    finally:
        if hasattr(source, "close"):
            source.close()


async def broadcast_loop(sample_queue: "queue.Queue", clients: set, last_status: dict) -> None:
    loop = asyncio.get_running_loop()
    while True:
        msg = await loop.run_in_executor(None, sample_queue.get)
        if msg["type"] in ("status", "error"):
            last_status["message"] = msg  # remembered so late-joining clients aren't stuck at "waiting..."
        if clients:
            websockets.broadcast(clients, json.dumps(msg))


async def serve(host: str, port: int, simulate: bool) -> None:
    sample_queue: "queue.Queue" = queue.Queue(maxsize=2000)
    stop_event = threading.Event()
    clients: set = set()
    last_status: dict = {}

    t = threading.Thread(target=receiver_thread_main, args=(sample_queue, stop_event, simulate), daemon=True)
    t.start()

    async def handler(websocket):
        clients.add(websocket)
        print(f"client connected ({len(clients)} total)", flush=True)
        if "message" in last_status:
            await websocket.send(json.dumps(last_status["message"]))
        try:
            async for raw_message in websocket:
                try:
                    request = json.loads(raw_message)
                except json.JSONDecodeError:
                    continue
                await handle_request(websocket, request)
        finally:
            clients.discard(websocket)
            print(f"client disconnected ({len(clients)} total)", flush=True)

    broadcaster = asyncio.create_task(broadcast_loop(sample_queue, clients, last_status))
    try:
        async with websockets.serve(handler, host, port, max_size=None):
            print(f"player server listening on ws://{host}:{port}", flush=True)
            print("open visualizations/interactive/player.html", flush=True)
            await asyncio.Future()
    finally:
        stop_event.set()
        broadcaster.cancel()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--simulate", action="store_true", help="synthetic live data, no ESP32 hardware needed")
    args = p.parse_args()

    if args.simulate:
        print("SIMULATE MODE: live tab streams synthetic data, no ESP32 hardware required")
    if args.port != 8765 or args.host != "localhost":
        print(f"note: player.html defaults to ws://localhost:8765 -- "
              f"change the URL field there to ws://{args.host}:{args.port}")
    asyncio.run(serve(args.host, args.port, args.simulate))


if __name__ == "__main__":
    main()
