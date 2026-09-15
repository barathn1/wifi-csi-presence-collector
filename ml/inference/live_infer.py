"""Run the checkpointed models LIVE against the ESP32 board's real-time CSI stream -- a smoke test
before a real labeled Day 3 dataset exists. Loads the 3 checkpoints ml.training.train_final_model
produces (presence, auth, motion), does a live empty-room calibration, then continuously prints all
three live decisions (instantaneous + a ~30s rolling aggregate, which is the one to trust -- see
ml/reports/day2_next_steps.md's validated 5.5%-false-accept result).

    python3 -m ml.inference.live_infer
    python3 -m ml.inference.live_infer --calib-seconds 20 --aggregate-windows 40
    python3 -m ml.inference.live_infer --expect-mhz 40 --force   # e.g. testing against Day-1-era config

See LIVE_INFERENCE_RUNBOOK.md for the full step-by-step testing protocol (how long to stand still to
calibrate, how long to stand vs. walk, what output to expect at each step).

Mutual exclusion: the ESP32 transport can only be held by one process at a time -- stop
collector.cli_collect / ml.visualization.player_server before running this.
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

from collector import preflight, stimulus
from collector.config import load_config
from collector.receiver import get_receiver
from ml.data_pipeline.calibration import apply_variant_a
from ml.data_pipeline.decode_csi import REPO_ROOT, channel_width_summary_live, decode_one_sample
from ml.inference.checkpoint import LoadedCheckpoint, load_checkpoint
from ml.inference.live_calibration import compute_live_baseline
from ml.inference.live_window import RollingWindower

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = REPO_ROOT / "ml/checkpoints"
CHECKPOINT_FILE_STEMS = {
    "task0_presence": "whofi_task0_presence_calibA",
    "taskD_auth_vs_nonauth": "whofi_taskD_calibA",
    "taskE_motion_standing_vs_walking": "whofi_taskE_motion_calibA",
}
BANDWIDTH_CHECK_PACKETS = 50  # checked BEFORE the calibration wait, so a bad config fails in seconds
CALIB_STATUS_INTERVAL_S = 5.0  # print a "time remaining" line at least this often during calibration


class _GracefulExit(Exception):
    """Same pattern as collector.cli_collect -- lets `kill <pid>` unwind through the same
    try/finally as Ctrl+C instead of terminating immediately."""


def _handle_sigterm(signum, frame):
    raise _GracefulExit()


def load_all_checkpoints(suffix: str = "") -> dict[str, LoadedCheckpoint]:
    checkpoints = {}
    for task_name, stem in CHECKPOINT_FILE_STEMS.items():
        path = CHECKPOINT_DIR / f"{stem}{suffix}.pt"
        if not path.exists():
            logger.error("missing checkpoint %s -- run `python3 -m ml.training.train_final_model` first", path)
            sys.exit(1)
        ck = load_checkpoint(path)
        if ck.meta["preprocessing"] != "calibA" or ck.meta["mode"] != "resampled":
            logger.error("checkpoint %s has unexpected preprocessing/mode %r/%r -- this script assumes "
                         "calibA/resampled", path, ck.meta["preprocessing"], ck.meta["mode"])
            sys.exit(1)
        checkpoints[task_name] = ck
    n_subcarriers = {ck.arch_kwargs["n_subcarriers"] for ck in checkpoints.values()}
    assert len(n_subcarriers) == 1, f"checkpoints disagree on n_subcarriers: {n_subcarriers}"
    return checkpoints


def bandwidth_gate(combo_counts: dict, expect_mhz: int, force: bool, expect_channel: int | None = None) -> None:
    info = channel_width_summary_live(combo_counts)
    actual_mhz = 40 if info["cwb"] == 1 else 20
    logger.info("live bandwidth check: %s (%.1f%% of packets)", info["description"], 100 * info["fraction_of_packets"])
    problems = []
    if actual_mhz != expect_mhz:
        problems.append(f"bandwidth MISMATCH: expected {expect_mhz}MHz, got {actual_mhz}MHz")
    # --expect-mhz alone can't catch this: two channels can share a bandwidth (e.g. ch6 and ch11 are
    # both 20MHz) while still being different, non-overlapping RF bands the checkpoint was never
    # trained on -- see [[project-day2-cross-channel-root-cause]].
    if expect_channel is not None and info["channel_primary"] != expect_channel:
        problems.append(f"channel MISMATCH: expected channel {expect_channel}, got channel {info['channel_primary']}")
    if problems:
        msg = ("; ".join(problems) + f" ({info['description']}). The loaded checkpoint(s) were trained on a "
               f"specific band -- see ml/reports/day2_next_steps.md's cross-channel findings for why this matters.")
        if force:
            logger.warning("%s (continuing anyway: --force)", msg)
        else:
            logger.error("%s (pass --force to continue anyway, not recommended)", msg)
            sys.exit(1)


def format_reading(name: str, proba_inst: float, proba_agg: float | None, n_agg: int, agg_target: int,
                    display_labels: list[str], gated: bool) -> str:
    if gated:
        return f"{name}: n/a (no one present)"
    inst_label = display_labels[1] if proba_inst >= 0.5 else display_labels[0]
    if proba_agg is None:
        return f"{name}: {inst_label} ({proba_inst:.2f}, {n_agg}/{agg_target})"
    agg_label = display_labels[1] if proba_agg >= 0.5 else display_labels[0]
    return f"{name}: {agg_label} ({proba_agg:.2f}, {n_agg}/{agg_target} avg)"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--expect-mhz", type=int, default=20, choices=[20, 40])
    p.add_argument("--expect-channel", type=int, default=None, choices=range(1, 15), metavar="1-14",
                    help="also hard-gate on the exact 2.4GHz channel number, not just bandwidth -- "
                         "e.g. --expect-channel 6 for the day3ch6 checkpoints, since two different "
                         "channels can share the same bandwidth")
    p.add_argument("--checkpoint-suffix", default="",
                    help="load '<stem><suffix>.pt' instead of the default Day1+2-pooled checkpoints, "
                         "e.g. --checkpoint-suffix _day3ch6 for ml.training.train_day3_ch6_model's output")
    p.add_argument("--force", action="store_true", help="continue even if the live bandwidth check fails")
    p.add_argument("--calib-seconds", type=float, default=15.0)
    p.add_argument("--aggregate-windows", type=int, default=60, help="~30s at the validated recipe's rate")
    p.add_argument("--stand-seconds", type=float, default=60.0,
                   help="guided phase: how long to prompt STAND STILL for after calibration. 0 to skip.")
    p.add_argument("--walk-seconds", type=float, default=60.0,
                   help="guided phase: how long to prompt WALK AROUND for after the stand phase. 0 to skip.")
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

    checkpoints = load_all_checkpoints(args.checkpoint_suffix)
    n_subcarriers = next(iter(checkpoints.values())).arch_kwargs["n_subcarriers"]
    expected_csi_len = 2 * n_subcarriers
    window_packets = next(iter(checkpoints.values())).meta["window_packets"]
    stride_packets = next(iter(checkpoints.values())).meta["stride_packets"]
    logger.info("loaded %d checkpoints, expecting csi_len=%d bytes (%d subcarriers)",
                len(checkpoints), expected_csi_len, n_subcarriers)

    stop_event = threading.Event()
    stimulus_thread = None
    if cfg.stimulus.enabled and not args.no_stimulus:
        stimulus_thread = threading.Thread(target=stimulus.run_stimulus, args=(cfg, stop_event), daemon=True)
        stimulus_thread.start()
        logger.info("stimulus traffic generator started (needed for a steady CSI packet rate)")

    receiver = get_receiver(cfg, stop_event=stop_event, on_stat_line=lambda line: logger.debug(line))

    # guided test phases -- prompts + a live countdown so you know exactly when to tell the person to
    # stand/walk, instead of watching a separate stopwatch. Auth itself doesn't need this split (it
    # works the same whether the person is standing or walking) -- this exists to also exercise Motion,
    # which needs to see both to show a transition. 0 for either skips that phase.
    guided_phases = []
    if args.stand_seconds > 0:
        guided_phases.append(("STAND STILL", args.stand_seconds))
    if args.walk_seconds > 0:
        guided_phases.append(("WALK AROUND", args.walk_seconds))
    phase_idx = 0
    phase_start_time: float | None = None

    # --- phase state ---
    combo_counts: Counter = Counter()
    n_accepted = 0
    bandwidth_checked = False
    calibrating = False
    calib_start_time: float | None = None
    last_calib_status: float = 0.0
    calib_amp: list[np.ndarray] = []
    calib_phase: list[np.ndarray] = []
    baseline = None
    windower = RollingWindower(window_packets=window_packets, stride_packets=stride_packets)
    aggregators = {task: deque(maxlen=args.aggregate_windows) for task in checkpoints}
    window_idx = 0

    try:
        for sample in receiver:
            if sample.csi_len != expected_csi_len:
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
                last_calib_status = calib_start_time
                logger.info("=== CALIBRATION: stand OUTSIDE the room / away from the sensor now (%.0fs) ===",
                            args.calib_seconds)
                continue

            amplitude, phase = decode_one_sample(sample.csi_data)

            if calibrating:
                calib_amp.append(amplitude)
                calib_phase.append(phase)
                elapsed = time.monotonic() - calib_start_time
                enough_time = elapsed >= args.calib_seconds
                enough_packets = len(calib_amp) >= window_packets
                if enough_time and enough_packets:
                    baseline = compute_live_baseline(np.stack(calib_amp), np.stack(calib_phase))
                    logger.info("calibration complete: %d packets, amp_mean range [%.2f, %.2f], "
                                "amp_std range [%.2f, %.2f]", len(calib_amp),
                                baseline.amp_mean.min(), baseline.amp_mean.max(),
                                baseline.amp_std.min(), baseline.amp_std.max())
                    calibrating = False
                    if guided_phases:
                        phase_start_time = time.monotonic()
                        label, duration = guided_phases[0]
                        logger.info("=== NOW: have the authorized person %s (%.0fs) ===", label, duration)
                    else:
                        logger.info("=== streaming (no guided phases configured) -- Ctrl+C to stop ===")
                elif enough_time and not enough_packets:
                    pass  # safety floor: keep waiting for packets even past the nominal duration
                elif time.monotonic() - last_calib_status >= CALIB_STATUS_INTERVAL_S:
                    remaining = max(0.0, args.calib_seconds - elapsed)
                    logger.info("calibrating... %.0fs remaining (%d packets so far)", remaining, len(calib_amp))
                    last_calib_status = time.monotonic()
                continue

            # periodic bandwidth re-check during streaming -- warns (doesn't abort) on drift, since
            # aborting a live demo mid-flight over a possibly-transient blip is worse than a warning
            if n_accepted % 3000 == 0:
                bandwidth_gate(combo_counts, args.expect_mhz, force=True, expect_channel=args.expect_channel)

            if not windower.push(amplitude, phase):
                continue
            window_idx += 1
            amp_w, phase_w = windower.get_window()
            amp_z, phase_z = apply_variant_a(amp_w, phase_w, baseline)
            amp_t = torch.from_numpy(amp_z.astype(np.float32)).unsqueeze(0)
            phase_t = torch.from_numpy(phase_z.astype(np.float32)).unsqueeze(0)

            proba_inst, proba_agg = {}, {}
            for task_name, ck in checkpoints.items():
                with torch.no_grad():
                    logits = ck.model(amp_t, phase_t)
                    proba = torch.softmax(logits, dim=-1)[0, 1].item()
                proba_inst[task_name] = proba
                aggregators[task_name].append(proba)
                proba_agg[task_name] = float(np.mean(aggregators[task_name]))

            presence_ck = checkpoints["task0_presence"]
            present = proba_agg["task0_presence"] >= 0.5
            parts = [format_reading(
                "Presence", proba_inst["task0_presence"], proba_agg["task0_presence"],
                len(aggregators["task0_presence"]), args.aggregate_windows, presence_ck.display_labels, gated=False,
            )]
            for task_name, label in [("taskD_auth_vs_nonauth", "Auth"), ("taskE_motion_standing_vs_walking", "Motion")]:
                ck = checkpoints[task_name]
                parts.append(format_reading(
                    label, proba_inst[task_name], proba_agg[task_name],
                    len(aggregators[task_name]), args.aggregate_windows, ck.display_labels, gated=not present,
                ))

            # guided-phase countdown, prepended to the status line so you can tell the person when to
            # switch without watching a separate stopwatch. Advances/announces the next phase (or
            # "free monitoring") the moment the current one's duration elapses.
            phase_tag = ""
            if guided_phases and phase_idx < len(guided_phases):
                label, duration = guided_phases[phase_idx]
                elapsed_phase = time.monotonic() - phase_start_time
                if elapsed_phase >= duration:
                    phase_idx += 1
                    phase_start_time = time.monotonic()
                    if phase_idx < len(guided_phases):
                        next_label, next_duration = guided_phases[phase_idx]
                        logger.info("=== NOW: have the authorized person %s (%.0fs) ===", next_label, next_duration)
                        phase_tag = f"[{next_label}, just started] "
                    else:
                        logger.info("=== guided phases done -- free monitoring now, Ctrl+C to stop ===")
                else:
                    phase_tag = f"[{label}, {duration - elapsed_phase:.0f}s left] "

            print(f"[{time.strftime('%H:%M:%S')}] {phase_tag}window#{window_idx:5d}  " + "  |  ".join(parts))

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
