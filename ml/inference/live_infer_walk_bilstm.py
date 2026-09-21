"""Run the walking BiLSTM (`ml/training/train_walk_bilstm_final.py`'s checkpoint) LIVE against the
ESP32's real-time CSI stream, to watch it distinguish anjali from barath while one of them walks.

Reuses this repo's existing live-connection machinery (`collector.receiver`/`preflight`/`stimulus`,
`decode_csi.decode_one_sample`) exactly as `ml/inference/live_infer.py` does, but everything downstream
of the raw decode is this project's own walking-only pipeline (`walk_bilstm_pipeline.py`'s phase
sanitization / Hampel / Butterworth / time-based windowing+interpolation / top-30-variance subcarrier
selection / per-window normalization), reimplemented for a live, continuously-updating buffer instead
of a whole pre-recorded session array.

This model is CLOSED-SET and WALKING-ONLY by construction (see `walk_bilstm_pipeline.py`'s own
docstring and the project's earlier open-set investigation, which found no combination of
loss/architecture separated authorized from unauthorized walkers on this hardware): it can tell you
WHICH of anjali/barath it thinks is walking, with no "neither" option, and the model has only ever
seen WALKING data -- standing or otherwise-idle packets feed noise into the model, not a meaningful
"standing" pattern.

Live-vs-offline differences, called out here rather than silently glossed over:
- The offline pipeline runs Hampel/Butterworth ONCE over an entire session's continuous packet stream
  before windowing, so every point (including near a window's edges) has neighbors on both sides. Live,
  we can only ever see the past. This script keeps a `BUFFER_SECONDS` (default 8s) rolling buffer and
  re-filters that whole buffer every step, so most of each emitted 4s window's points still get
  two-sided context from the buffer -- only the newest ~0-2s at the buffer's leading edge has less
  right-context than the offline version would give it. Treat this as a live smoke test, not a
  byte-identical replay of the offline pipeline.
- The link/PHY-combo validation `select_dominant_packets` does per RECORDED SESSION (pick the single
  most common (channel/cwb/stbc/mcs/csi_len) combo after the fact) has no equivalent "after the fact"
  in a live stream -- this script locks in the dominant combo from the first `COMBO_LOCK_PACKETS`
  matching (channel 6, dst_mac==board_mac) packets instead, then gates every later packet against it.

    python -m ml.inference.live_infer_walk_bilstm
    python -m ml.inference.live_infer_walk_bilstm --aggregate-windows 5
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from collections import Counter, deque

import numpy as np
import torch
from scipy.interpolate import interp1d

from collector import preflight, stimulus
from collector.config import load_config
from collector.receiver import get_receiver
from ml.data_pipeline.decode_csi import REPO_ROOT, decode_one_sample
from ml.data_pipeline.paper_2507_12854_preprocessing import butterworth_lowpass, hampel_filter
from ml.data_pipeline.walk_bilstm_pipeline import apply_topk, per_window_zscore, sanitize_phase
from ml.models.bilstm_triplet import BiLSTMTriplet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = REPO_ROOT / "ml/checkpoints/walk_bilstm_final.pt"
N_SUB_EXPECTED = 128
EXPECTED_CSI_LEN_BYTES = 2 * N_SUB_EXPECTED
COMBO_LOCK_PACKETS = 100  # how many (channel-6, dst==board_mac) packets to see before locking the dominant PHY combo
BUFFER_SECONDS = 8.0      # rolling context buffer -- must be > window_sec so windows get two-sided filtering context
MIN_PACKETS_FOR_FIRST_WINDOW = 20


class _GracefulExit(Exception):
    pass


def _handle_sigterm(signum, frame):
    raise _GracefulExit()


def load_walk_checkpoint(path=CHECKPOINT_PATH):
    if not path.exists():
        logger.error("missing checkpoint %s -- run `python -m ml.training.train_walk_bilstm_final` first", path)
        sys.exit(1)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = BiLSTMTriplet(n_features=ck["n_features"], num_persons=ck["num_persons"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck


def combo_key(sample) -> tuple:
    return (sample.cwb, sample.channel_secondary, sample.stbc, sample.mcs, sample.csi_len)


def build_window(amp_buf: np.ndarray, phase_buf: np.ndarray, t_us_buf: np.ndarray, window_start_us: int,
                  window_sec: float, final_len: int, hampel_window: int, hampel_n_sigmas: float,
                  butter_cutoff_hz: float, butter_order: int, idx: np.ndarray) -> np.ndarray | None:
    """Filter the whole buffered context, then slice+interpolate just [window_start_us,
    window_start_us + window_sec) to `final_len` points -- mirrors `walk_bilstm_pipeline.make_windows`
    for one window, using the live rolling buffer as context instead of a whole recorded session."""
    phase_sanitized = sanitize_phase(phase_buf)
    features = np.concatenate([amp_buf, phase_sanitized], axis=1)
    features = hampel_filter(features, window=hampel_window, n_sigmas=hampel_n_sigmas)
    native_rate_hz = (len(t_us_buf) - 1) / ((t_us_buf[-1] - t_us_buf[0]) / 1e6)
    features = butterworth_lowpass(features, fs=native_rate_hz, cutoff=butter_cutoff_hz, order=butter_order)

    t = (t_us_buf.astype(np.float64) - window_start_us) / 1e6
    end_t = window_sec
    sel = np.flatnonzero((t >= 0) & (t < end_t))
    if len(sel) < MIN_PACKETS_FOR_FIRST_WINDOW:
        return None
    seg_t, seg = t[sel], features[sel]
    new_t = np.linspace(0, end_t, final_len, endpoint=False)
    interp = interp1d(seg_t, seg, axis=0, kind="linear", fill_value="extrapolate", assume_sorted=True)
    window = interp(new_t).astype(np.float32)[None, ...]  # (1, final_len, 256)

    window = apply_topk(window, idx)      # (1, final_len, 2k)
    window = per_window_zscore(window)    # (1, final_len, 2k)
    return window[0]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--expect-channel", type=int, default=6)
    p.add_argument("--aggregate-windows", type=int, default=5, help="rolling majority-vote window count "
                    "(~10s at 50% overlap, 4s windows)")
    p.add_argument("--no-stimulus", action="store_true")
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    cfg = load_config(args.config) if args.config else load_config()

    ok, reason = preflight.check_board_connected(cfg)
    if not ok:
        logger.error("preflight failed: %s", reason)
        return 1
    logger.info("preflight ok: %s (pid=%d, Ctrl+C or `kill %d` to stop)", reason, os.getpid(), os.getpid())

    model, ck = load_walk_checkpoint()
    idx = ck["subcarrier_idx"]
    pp = ck["preprocessing"]
    classes = ck["classes"]
    logger.info("loaded checkpoint: classes=%s, %d subcarriers selected, window=%.1fs overlap=%.0f%%",
                classes, len(idx), pp["window_sec"], 100 * pp["overlap"])
    logger.info("NOTE: this model only ever saw WALKING data -- have the person walk continuously, "
                 "not stand still, for meaningful readings.")

    board_mac_str = cfg.device.mac.lower()
    step_sec = pp["window_sec"] * (1 - pp["overlap"])

    stop_event = threading.Event()
    stimulus_thread = None
    if cfg.stimulus.enabled and not args.no_stimulus:
        stimulus_thread = threading.Thread(target=stimulus.run_stimulus, args=(cfg, stop_event), daemon=True)
        stimulus_thread.start()
        logger.info("stimulus traffic generator started (needed for a steady CSI packet rate)")

    receiver = get_receiver(cfg, stop_event=stop_event, on_stat_line=lambda line: logger.debug(line))

    combo_counts: Counter = Counter()
    locked_combo: tuple | None = None
    n_matched = 0
    buf_amp: deque = deque()
    buf_phase: deque = deque()
    buf_t: deque = deque()
    next_window_start_us: int | None = None
    window_idx = 0
    aggregator: deque = deque(maxlen=args.aggregate_windows)

    try:
        for sample in receiver:
            if sample.dst_mac_str.lower() != board_mac_str or sample.channel_primary != args.expect_channel:
                continue

            if locked_combo is None:
                combo_counts[combo_key(sample)] += 1
                n_matched += 1
                if n_matched < COMBO_LOCK_PACKETS:
                    continue
                locked_combo = combo_counts.most_common(1)[0][0]
                logger.info("locked dominant PHY combo (cwb,ch2,stbc,mcs,csi_len)=%s from first %d packets",
                            locked_combo, n_matched)
                if locked_combo[-1] != EXPECTED_CSI_LEN_BYTES:
                    logger.error("locked csi_len=%d bytes (%d subcarriers), expected %d bytes (%d subcarriers) "
                                 "-- this checkpoint was trained on %d-subcarrier packets; aborting",
                                 locked_combo[-1], locked_combo[-1] // 2, EXPECTED_CSI_LEN_BYTES, N_SUB_EXPECTED)
                    return 1
                continue

            if combo_key(sample) != locked_combo:
                continue

            amplitude, phase = decode_one_sample(sample.csi_data)
            buf_amp.append(amplitude)
            buf_phase.append(phase)
            buf_t.append(sample.device_time_us)
            if next_window_start_us is None:
                next_window_start_us = sample.device_time_us

            cutoff_us = sample.device_time_us - int(BUFFER_SECONDS * 1e6)
            while buf_t and buf_t[0] < cutoff_us:
                buf_t.popleft(); buf_amp.popleft(); buf_phase.popleft()

            if sample.device_time_us - next_window_start_us < pp["window_sec"] * 1e6:
                continue

            amp_buf = np.stack(buf_amp)
            phase_buf = np.stack(buf_phase)
            t_buf = np.array(buf_t, dtype=np.int64)
            window = build_window(amp_buf, phase_buf, t_buf, next_window_start_us, pp["window_sec"],
                                   pp["final_len"], pp["hampel_window"], pp["hampel_n_sigmas"],
                                   pp["butter_cutoff_hz"], pp["butter_order"], idx)
            next_window_start_us += int(step_sec * 1e6)
            if window is None:
                continue

            window_idx += 1
            x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                _, logits = model(x)
                probs = torch.softmax(logits, dim=-1)[0].numpy()
            pred_idx = int(probs.argmax())
            aggregator.append(probs)
            agg_probs = np.mean(aggregator, axis=0)
            agg_pred_idx = int(agg_probs.argmax())

            print(f"[{time.strftime('%H:%M:%S')}] window#{window_idx:4d}  "
                  f"instant: {classes[pred_idx]:8s} ({probs[pred_idx]:.2f})   "
                  f"aggregate({len(aggregator)}/{args.aggregate_windows}): "
                  f"{classes[agg_pred_idx]:8s} ({agg_probs[agg_pred_idx]:.2f})   "
                  f"probs={dict(zip(classes, np.round(probs, 2)))}")

    except (KeyboardInterrupt, _GracefulExit):
        logger.info("stopping (interrupted)")
    finally:
        stop_event.set()
        receiver.close()
        if stimulus_thread is not None:
            stimulus_thread.join(timeout=2.0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
