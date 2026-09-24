"""Run the two ml_2 "creative" checkpoints -- SubcarrierGNN (`gnn_subcarrier`) and RadarInspiredCNN
(`radar_cnn`) -- LIVE against the ESP32's real-time CSI stream: loads both checkpoints
ml_2.training.train_live_checkpoints produces, does a live empty-room calibration, then continuously
prints BOTH models' live authorized/not-authorized reading (instantaneous + rolling aggregate) side by
side so you can see which model says what, in real time.

    python3 -m ml_2.inference.live_infer
    python3 -m ml_2.inference.live_infer --calib-seconds 30 --aggregate-windows 40

See ml_2/LIVE_TEST_RUNBOOK.md for the full step-by-step protocol.

Both checkpoints were trained ONLY on channel-6 sessions (see ml_2/training/common_data.py), so this
hard-gates on channel 6 / 20MHz by default, same reasoning as DAY3_CH6_RUNBOOK.md's `--expect-channel`.

Pass `--collect` to ALSO save this session as a new labeled training session under `data/`, in the
exact same `{metadata.json, samples.npz}` format `collector.cli_collect` writes -- you'll be prompted
interactively for label/person_id/motion/notes right after preflight (instead of passing them as CLI
args like cli_collect.py requires). Recording only starts once the live empty-room calibration
finishes (not during it), so the necessarily-empty calibration window never gets saved under whatever
label you gave -- if you want a `none`/empty-room session specifically, answer `none` at the prompt and
just don't let anyone in the room for the whole run.

Mutual exclusion: the ESP32 transport can only be held by one process at a time -- stop
collector.cli_collect / ml.visualization.player_server / ml.inference.live_infer before running this.

The printed number is each model's live positive-class probability (confidence), aggregated over a
rolling window -- NOT a verified accuracy (there's no ground truth during a live run). See
ml_2/evaluation/results/DAY_PERSON_MODEL_ACCURACY.md for each model's actual measured accuracy.
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

from collector import preflight, stimulus, wire
from collector.build_manifest import build_manifest
from collector.config import Config, load_config
from collector.receiver import get_receiver
from collector.session_writer import SessionWriter
from ml.data_pipeline.decode_csi import channel_width_summary_live, decode_one_sample
from ml.inference.live_window import RollingWindower
from ml_2.data.calibration import apply_baseline
from ml_2.data.decode import REPO_ROOT
from ml_2.data.windowing import N_LEGACY_SUBCARRIERS
from ml_2.inference.checkpoint import LoadedCheckpoint, load_checkpoint
from ml_2.inference.live_calibration import compute_live_baseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = REPO_ROOT / "ml_2/checkpoints"
MODEL_STEMS = {"gnn_subcarrier": "gnn_subcarrier_calibA", "radar_cnn": "radar_cnn_calibA"}
DISPLAY_NAMES = {"gnn_subcarrier": "GNN", "radar_cnn": "RADAR_CNN"}
FULL_N_SUBCARRIERS = 128  # channel-6 sessions merge LLTF+HT-LTF into a 128-value buffer (256 bytes) --
# see ml_2/data/windowing.py's N_LEGACY_SUBCARRIERS comment; sliced down to the first 64 below, exactly
# as training-time caching does, so live windows match the shape the checkpoints were trained on.
EXPECTED_CSI_LEN = 2 * FULL_N_SUBCARRIERS
BANDWIDTH_CHECK_PACKETS = 50  # checked BEFORE the calibration wait, so a bad config fails in seconds
CALIB_COUNTDOWN_INTERVAL_S = 1.0  # tick the "Ns left" countdown about this often during calibration


class _GracefulExit(Exception):
    """Same pattern as collector.cli_collect -- lets `kill <pid>` unwind through the same
    try/finally as Ctrl+C instead of terminating immediately."""


def _handle_sigterm(signum, frame):
    raise _GracefulExit()


def load_all_checkpoints() -> dict[str, LoadedCheckpoint]:
    checkpoints = {}
    for model_name, stem in MODEL_STEMS.items():
        path = CHECKPOINT_DIR / f"{stem}.pt"
        if not path.exists():
            logger.error("missing checkpoint %s -- run `python3 -m ml_2.training.train_live_checkpoints` first", path)
            sys.exit(1)
        checkpoints[model_name] = load_checkpoint(path)
    n_subcarriers = {ck.arch_kwargs["n_subcarriers"] for ck in checkpoints.values()}
    assert len(n_subcarriers) == 1, f"checkpoints disagree on n_subcarriers: {n_subcarriers}"
    windows = {(ck.meta["window_packets"], ck.meta["stride_packets"]) for ck in checkpoints.values()}
    assert len(windows) == 1, f"checkpoints disagree on window/stride packets: {windows}"
    return checkpoints


def bandwidth_gate(combo_counts: dict, expect_mhz: int, force: bool, expect_channel: int | None) -> None:
    info = channel_width_summary_live(combo_counts)
    actual_mhz = 40 if info["cwb"] == 1 else 20
    logger.info("live bandwidth check: %s (%.1f%% of packets)", info["description"], 100 * info["fraction_of_packets"])
    problems = []
    if actual_mhz != expect_mhz:
        problems.append(f"bandwidth MISMATCH: expected {expect_mhz}MHz, got {actual_mhz}MHz")
    if expect_channel is not None and info["channel_primary"] != expect_channel:
        problems.append(f"channel MISMATCH: expected channel {expect_channel}, got channel {info['channel_primary']}")
    if problems:
        msg = ("; ".join(problems) + f" ({info['description']}). Both checkpoints here were trained ONLY on "
               "channel-6 sessions (see ml_2/training/common_data.py) -- a different band means data the "
               "models never saw.")
        if force:
            logger.warning("%s (continuing anyway: --force)", msg)
        else:
            logger.error("%s (pass --force to continue anyway, not recommended)", msg)
            sys.exit(1)


def prompt_collection_labels(cfg: Config) -> tuple[str, str, str, str]:
    """Interactive equivalent of collector.cli_collect's --label/--person-id/--motion/--notes flags --
    same validation rules (person_id and motion required unless label is 'none')."""
    labels = cfg.dataset.labels
    while True:
        label = input(f"[collect] label ({'/'.join(labels)}): ").strip()
        if label in labels:
            break
        print(f"  '{label}' is not one of {labels}, try again")

    person_id, motion = "", ""
    if label != "none":
        while not person_id:
            person_id = input("[collect] person_id (required): ").strip()
        while motion not in ("standing", "walking"):
            motion = input("[collect] motion (standing/walking): ").strip()

    notes = input("[collect] notes (optional): ").strip()
    return label, person_id, motion, notes


def format_reading(name: str, proba_inst: float, proba_agg: float | None, n_agg: int, agg_target: int,
                    display_labels: list[str], threshold: float = 0.5) -> str:
    inst_label = display_labels[1] if proba_inst >= threshold else display_labels[0]
    if proba_agg is None:
        return f"{name}: {inst_label} ({proba_inst:.2f}, {n_agg}/{agg_target})"
    agg_label = display_labels[1] if proba_agg >= threshold else display_labels[0]
    return f"{name}: {agg_label} ({proba_agg:.2f}, {n_agg}/{agg_target} avg)"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--expect-mhz", type=int, default=20, choices=[20, 40])
    p.add_argument("--expect-channel", type=int, default=6, choices=range(1, 15), metavar="1-14",
                    help="hard-gate on the exact 2.4GHz channel number -- default 6, matching the "
                         "channel-6-only dataset both checkpoints were trained on")
    p.add_argument("--force", action="store_true", help="continue even if the live bandwidth/channel check fails")
    p.add_argument("--calib-seconds", type=float, default=60.0)
    p.add_argument("--aggregate-windows", type=int, default=60, help="rolling decision window, in windows")
    p.add_argument("--no-stimulus", action="store_true")
    p.add_argument("--config", default=None)
    p.add_argument("--collect", action="store_true",
                    help="also save this session to data/ as a new labeled training session "
                         "(prompts for label/person_id/motion/notes interactively)")
    args = p.parse_args(argv)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    cfg = load_config(args.config) if args.config else load_config()

    ok, reason = preflight.check_board_connected(cfg)
    if not ok:
        logger.error("preflight failed: %s", reason)
        return 1
    logger.info("preflight ok: %s (pid=%d, Ctrl+C or `kill %d` to stop)", reason, os.getpid(), os.getpid())

    collect_label = collect_person_id = collect_motion = collect_notes = None
    if args.collect:
        collect_label, collect_person_id, collect_motion, collect_notes = prompt_collection_labels(cfg)
        logger.info("collect mode: will save as label=%s person_id=%s motion=%s -- recording starts "
                    "once calibration finishes (not during it)", collect_label, collect_person_id or "n/a",
                    collect_motion or "n/a")

    checkpoints = load_all_checkpoints()
    window_packets = next(iter(checkpoints.values())).meta["window_packets"]
    stride_packets = next(iter(checkpoints.values())).meta["stride_packets"]
    logger.info("loaded checkpoints: %s (window_packets=%d, stride_packets=%d)",
                list(checkpoints), window_packets, stride_packets)

    stop_event = threading.Event()
    stimulus_thread = None
    if cfg.stimulus.enabled and not args.no_stimulus:
        stimulus_thread = threading.Thread(target=stimulus.run_stimulus, args=(cfg, stop_event), daemon=True)
        stimulus_thread.start()
        logger.info("stimulus traffic generator started (needed for a steady CSI packet rate)")

    receiver = get_receiver(cfg, stop_event=stop_event, on_stat_line=lambda line: logger.debug(line))

    combo_counts: Counter = Counter()
    n_accepted = 0
    bandwidth_checked = False
    calibrating = False
    calib_start_time: float | None = None
    last_countdown_tick: float = 0.0
    calib_amp: list[np.ndarray] = []
    calib_phase: list[np.ndarray] = []
    baseline = None
    windower = RollingWindower(window_packets=window_packets, stride_packets=stride_packets)
    aggregators = {name: deque(maxlen=args.aggregate_windows) for name in checkpoints}
    window_idx = 0
    writer: SessionWriter | None = None
    writer_last_seq: int | None = None
    writer_dropped = 0

    try:
        for sample in receiver:
            if writer is not None:
                writer.add(sample)
                if writer_last_seq is not None:
                    gap = sample.seq - writer_last_seq - 1
                    if gap > 0:
                        writer_dropped += gap
                writer_last_seq = sample.seq

            if sample.csi_len != EXPECTED_CSI_LEN:
                continue
            n_accepted += 1
            combo_counts[(sample.cwb, sample.channel_primary, sample.channel_secondary)] += 1

            if not bandwidth_checked:
                if n_accepted < BANDWIDTH_CHECK_PACKETS:
                    continue
                bandwidth_gate(combo_counts, args.expect_mhz, args.force, args.expect_channel)
                bandwidth_checked = True
                calibrating = True
                calib_start_time = time.monotonic()
                last_countdown_tick = calib_start_time
                logger.info("=== CALIBRATION: stand OUTSIDE the room / away from the sensor now (%.0fs) ===",
                            args.calib_seconds)
                continue

            amplitude_full, phase_full = decode_one_sample(sample.csi_data)
            amplitude, phase = amplitude_full[:N_LEGACY_SUBCARRIERS], phase_full[:N_LEGACY_SUBCARRIERS]

            if not calibrating and n_accepted % 3000 == 0:
                bandwidth_gate(combo_counts, args.expect_mhz, force=True, expect_channel=args.expect_channel)

            if calibrating:
                calib_amp.append(amplitude)
                calib_phase.append(phase)
                elapsed = time.monotonic() - calib_start_time
                enough_time = elapsed >= args.calib_seconds
                enough_packets = len(calib_amp) >= window_packets
                if enough_time and enough_packets:
                    print()  # end the countdown line before logging on a fresh line
                    baseline = compute_live_baseline(np.stack(calib_amp), np.stack(calib_phase))
                    logger.info("calibration complete: %d packets, amp_mean range [%.2f, %.2f], "
                                "amp_std range [%.2f, %.2f]", len(calib_amp),
                                baseline.amp_mean.min(), baseline.amp_mean.max(),
                                baseline.amp_std.min(), baseline.amp_std.max())
                    calibrating = False
                    if args.collect:
                        writer = SessionWriter(cfg, collect_label, collect_person_id, collect_notes,
                                                motion=collect_motion)
                        logger.info("collect mode: recording started (label=%s)", collect_label)
                    logger.info("=== streaming -- have a person walk in/out as needed, Ctrl+C to stop ===")
                elif enough_time and not enough_packets:
                    if time.monotonic() - last_countdown_tick >= CALIB_COUNTDOWN_INTERVAL_S:
                        print(f"\rcalibrating... waiting for enough packets ({len(calib_amp)}/{window_packets}) "
                              "-- past the nominal duration, packet rate is unusually low   ", end="", flush=True)
                        last_countdown_tick = time.monotonic()
                elif time.monotonic() - last_countdown_tick >= CALIB_COUNTDOWN_INTERVAL_S:
                    remaining = max(0.0, args.calib_seconds - elapsed)
                    print(f"\r=== CALIBRATING: {remaining:0.0f} seconds left "
                          f"({len(calib_amp)} packets so far) ===   ", end="", flush=True)
                    last_countdown_tick = time.monotonic()
                continue

            if not windower.push(amplitude, phase):
                continue
            window_idx += 1
            amp_w, phase_w = windower.get_window()
            amp_z, phase_z = apply_baseline(amp_w, phase_w, baseline)
            amp_t = torch.from_numpy(amp_z.astype(np.float32)).unsqueeze(0)
            phase_t = torch.from_numpy(phase_z.astype(np.float32)).unsqueeze(0)

            parts = []
            for model_name, ck in checkpoints.items():
                with torch.no_grad():
                    logits = ck.model(amp_t, phase_t)
                    proba = torch.softmax(logits, dim=-1)[0, 1].item()
                aggregators[model_name].append(proba)
                proba_agg = float(np.mean(aggregators[model_name]))
                parts.append(format_reading(
                    DISPLAY_NAMES[model_name], proba, proba_agg,
                    len(aggregators[model_name]), args.aggregate_windows, ck.display_labels,
                ))

            print(f"[{time.strftime('%H:%M:%S')}] window#{window_idx:5d}  " + "  |  ".join(parts))

    except (KeyboardInterrupt, _GracefulExit):
        logger.info("stopping (interrupted)")
    finally:
        stop_event.set()
        receiver.close()
        if stimulus_thread is not None:
            stimulus_thread.join(timeout=2.0)

    if args.collect:
        if writer is None or not writer.samples:
            logger.warning("collect mode: no samples were recorded (stopped before calibration finished?) "
                            "-- nothing saved")
        else:
            # writing samples.npz can't be safely interrupted partway through -- same reasoning as
            # collector.cli_collect.py -- ignore a second Ctrl+C/kill here so it can't corrupt the save.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            logger.info("collect mode: saving %d samples to disk -- please wait", len(writer.samples))
            out_dir = writer.finish(
                transport_used=cfg.transport.mode,
                board_mac=preflight.get_board_mac(cfg) or "unknown",
                firmware_version=f"wire-v{wire.VERSION}",
                dropped_samples=writer_dropped,
            )
            n = len(writer.samples)
            loss_pct = 100.0 * writer_dropped / (writer_dropped + n) if (writer_dropped + n) else 0.0
            logger.info("collect mode: wrote %d samples to %s (%d dropped, %.1f%% loss)",
                        n, out_dir, writer_dropped, loss_pct)
            build_manifest(cfg)  # keep data/manifest.csv current, same as cli_collect.py

    return 0


if __name__ == "__main__":
    sys.exit(main())
