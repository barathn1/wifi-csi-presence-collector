"""CLI: collect one labeled CSI session.

    python -m collector.cli_collect --label authorized --person-id alice --motion standing --duration 120
    python -m collector.cli_collect --label none --duration 60          # empty room
    python -m collector.cli_collect --label unauthorized --person-id bob --motion walking   # Ctrl+C to stop

Runs a preflight check first and refuses to start with a clear error if
the board isn't actually associated to the hotspot, rather than silently
writing an empty session.

Stopping early: Ctrl+C (SIGINT) or `kill <pid>` (SIGTERM) both stop
collection and write out whatever was captured so far -- neither loses
the session. `kill -9` (SIGKILL) cannot be handled by any process, by
design of the OS, so it will lose the in-memory session; there's no way
around that short of streaming every sample to disk incrementally, which
this v1 deliberately doesn't do (see README).
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import sys
import threading
import time

from collector import preflight, stimulus, wire
from collector.build_manifest import build_manifest
from collector.config import load_config
from collector.receiver import get_multi_receiver
from collector.session_writer import SessionWriter
from collector.transport_tcp import resolve_mac_for_ip

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class _GracefulExit(Exception):
    """Raised from the SIGTERM handler so `kill <pid>` unwinds through the
    same try/finally as Ctrl+C (KeyboardInterrupt), instead of the
    default SIGTERM behavior of terminating the process immediately and
    losing everything collected so far."""


def _handle_sigterm(signum, frame):
    raise _GracefulExit()


def _resolve_board_mac(cfg, key: str) -> str:
    """`key` is a TCP peer IP or (for serial) the fixed configured port --
    neither is the board's real MAC, so resolve the best display value for
    metadata.json/board_tag."""
    if cfg.transport.mode == "tcp":
        return resolve_mac_for_ip(cfg.network.laptop_iface, key) or key
    return preflight.get_board_mac(cfg) or key


def _sanitize_tag(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "", s)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Collect one labeled CSI session.")
    p.add_argument("--label", required=True, help="authorized | unauthorized | none")
    p.add_argument("--person-id", default="", help="required unless --label none")
    p.add_argument("--motion", default="", choices=["", "standing", "walking"],
                    help="standing | walking -- required unless --label none")
    p.add_argument("--notes", default="")
    p.add_argument("--duration", type=float, default=None, help="seconds; omit to stop with Ctrl+C")
    p.add_argument("--no-stimulus", action="store_true", help="disable the UDP traffic generator")
    p.add_argument("--config", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    signal.signal(signal.SIGTERM, _handle_sigterm)

    args = parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()

    if args.label != "none" and not args.person_id:
        logger.error("--person-id is required unless --label none")
        return 2
    if args.label != "none" and not args.motion:
        logger.error("--motion (standing|walking) is required unless --label none")
        return 2

    ok, reason = preflight.check_board_connected(cfg)
    if not ok:
        logger.error("preflight failed: %s", reason)
        return 1
    logger.info("preflight ok: %s", reason)

    stop_event = threading.Event()
    stimulus_thread = None
    if cfg.stimulus.enabled and not args.no_stimulus:
        stimulus_thread = threading.Thread(
            target=stimulus.run_stimulus, args=(cfg, stop_event), daemon=True
        )
        stimulus_thread.start()

    receiver = get_multi_receiver(cfg, stop_event=stop_event, on_stat_line=lambda line: logger.debug(line))

    logger.info(
        "collecting label=%s person_id=%s motion=%s duration=%s transport=%s "
        "(Ctrl+C or `kill %d` to stop early and still save what was collected)",
        args.label, args.person_id, args.motion or "n/a", args.duration, cfg.transport.mode, os.getpid(),
    )

    # One SessionWriter per connected board, created lazily on its first
    # sample -- this is what turns N simultaneously-connected boards into
    # N separate {metadata.json, samples.npz} pairs from one invocation.
    # `seq` is a monotonic counter the firmware assigns at CSI-capture
    # time (see wire_format.c); a gap between consecutive received seq
    # values (tracked independently per board) means samples were dropped
    # somewhere between that board's capture queue and here.
    writers: dict[str, SessionWriter] = {}
    last_seq: dict[str, int] = {}
    dropped: dict[str, int] = {}
    STATUS_INTERVAL_S = 5.0

    start = time.monotonic()
    last_status = start
    try:
        for key, sample in receiver:
            if key not in writers:
                writers[key] = SessionWriter(cfg, args.label, args.person_id, args.notes, motion=args.motion)
                dropped[key] = 0
                logger.info("board %s: first sample received (%d board(s) active)", key, len(writers))
            writer = writers[key]
            writer.add(sample)

            if key in last_seq:
                gap = sample.seq - last_seq[key] - 1
                if gap > 0:
                    dropped[key] += gap
                elif gap < 0:
                    logger.warning(
                        "board %s: sequence went backward (seq=%d after %d) -- board likely reset mid-session",
                        key, sample.seq, last_seq[key],
                    )
            last_seq[key] = sample.seq

            now = time.monotonic()
            if now - last_status >= STATUS_INTERVAL_S:
                elapsed = now - start
                per_board = " | ".join(
                    f"{k}: {len(w.samples)} ({len(w.samples) / elapsed if elapsed > 0 else 0.0:.1f} Hz, "
                    f"{dropped[k]} dropped)"
                    for k, w in writers.items()
                )
                if args.duration is not None:
                    remaining = max(0.0, args.duration - elapsed)
                    logger.info("%.0fs elapsed, %.0fs remaining | %s", elapsed, remaining, per_board)
                else:
                    logger.info("%.0fs elapsed | %s", elapsed, per_board)
                last_status = now

            if args.duration is not None and (time.monotonic() - start) >= args.duration:
                break
    except (KeyboardInterrupt, _GracefulExit):
        total = sum(len(w.samples) for w in writers.values())
        logger.info(
            "stopping early (interrupted) -- saving %d samples collected so far across %d board(s)",
            total, len(writers),
        )
    finally:
        stop_event.set()
        receiver.close()
        if stimulus_thread is not None:
            stimulus_thread.join(timeout=2.0)

    if not writers:
        logger.warning("no boards ever sent data -- nothing written")
        build_manifest(cfg)
        return 0

    # Writing samples.npz is slow for large sessions and can't be safely
    # interrupted partway through (it's a streaming zip write) -- ignore
    # further Ctrl+C/`kill` here so a second interrupt during save (easy
    # to trigger right after the first one that stopped collection)
    # can't corrupt/abort a save that was otherwise going to succeed.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    logger.info("saving %d board(s)' data to disk -- please wait, this can take a moment", len(writers))

    multi_board = len(writers) > 1
    elapsed = time.monotonic() - start
    for key, writer in writers.items():
        board_mac = _resolve_board_mac(cfg, key)
        out_dir = writer.finish(
            transport_used=cfg.transport.mode,
            board_mac=board_mac,
            # Firmware doesn't yet report its own build ID over the wire;
            # the CSI wire-format version is the best available proxy today.
            firmware_version=f"wire-v{wire.VERSION}",
            dropped_samples=dropped[key],
            board_tag=_sanitize_tag(board_mac) if multi_board else "",
        )
        n = len(writer.samples)
        d = dropped[key]
        loss_pct = 100.0 * d / (d + n) if (d + n) else 0.0
        avg_hz = n / elapsed if elapsed > 0 else 0.0
        logger.info(
            "board %s: wrote %d samples to %s -- avg %.1f Hz over %.0fs (%d dropped, %.1f%% loss)",
            key, n, out_dir, avg_hz, elapsed, d, loss_pct,
        )

    build_manifest(cfg)  # keep data/manifest.csv current after every session
    return 0


if __name__ == "__main__":
    sys.exit(main())
