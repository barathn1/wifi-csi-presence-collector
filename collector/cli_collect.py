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
import signal
import sys
import threading
import time

from collector import preflight, stimulus, wire
from collector.build_manifest import build_manifest
from collector.config import load_config
from collector.receiver import get_receiver
from collector.session_writer import SessionWriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class _GracefulExit(Exception):
    """Raised from the SIGTERM handler so `kill <pid>` unwinds through the
    same try/finally as Ctrl+C (KeyboardInterrupt), instead of the
    default SIGTERM behavior of terminating the process immediately and
    losing everything collected so far."""


def _handle_sigterm(signum, frame):
    raise _GracefulExit()


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
    board_mac = preflight.get_board_mac(cfg) or "unknown"

    stop_event = threading.Event()
    stimulus_thread = None
    if cfg.stimulus.enabled and not args.no_stimulus:
        stimulus_thread = threading.Thread(
            target=stimulus.run_stimulus, args=(cfg, stop_event), daemon=True
        )
        stimulus_thread.start()

    writer = SessionWriter(cfg, args.label, args.person_id, args.notes, motion=args.motion)
    receiver = get_receiver(cfg, stop_event=stop_event, on_stat_line=lambda line: logger.debug(line))

    logger.info(
        "collecting label=%s person_id=%s motion=%s duration=%s transport=%s "
        "(Ctrl+C or `kill %d` to stop early and still save what was collected)",
        args.label, args.person_id, args.motion or "n/a", args.duration, cfg.transport.mode, os.getpid(),
    )

    # `seq` is a monotonic counter the firmware assigns at CSI-capture
    # time (see wire_format.c); a gap between consecutive received seq
    # values means samples were dropped somewhere between the board's
    # capture queue and here (queue-full, tx failure, or real transport
    # loss) -- this is real loss detection, not a guess.
    last_seq = None
    dropped = 0
    STATUS_INTERVAL_S = 5.0

    start = time.monotonic()
    last_status = start
    try:
        for sample in receiver:
            writer.add(sample)

            if last_seq is not None:
                gap = sample.seq - last_seq - 1
                if gap > 0:
                    dropped += gap
                elif gap < 0:
                    logger.warning(
                        "sequence went backward (seq=%d after %d) -- board likely reset mid-session",
                        sample.seq, last_seq,
                    )
            last_seq = sample.seq

            now = time.monotonic()
            if now - last_status >= STATUS_INTERVAL_S:
                elapsed = now - start
                n = len(writer.samples)
                rate = n / elapsed if elapsed > 0 else 0.0
                if args.duration is not None:
                    remaining = max(0.0, args.duration - elapsed)
                    logger.info(
                        "%d samples (%.1f Hz) | %.0fs elapsed, %.0fs remaining | %d dropped (%.1f%% loss)",
                        n, rate, elapsed, remaining, dropped, 100.0 * dropped / (dropped + n) if (dropped + n) else 0.0,
                    )
                else:
                    logger.info(
                        "%d samples (%.1f Hz) | %.0fs elapsed | %d dropped (%.1f%% loss)",
                        n, rate, elapsed, dropped, 100.0 * dropped / (dropped + n) if (dropped + n) else 0.0,
                    )
                last_status = now

            if args.duration is not None and (time.monotonic() - start) >= args.duration:
                break
    except (KeyboardInterrupt, _GracefulExit):
        logger.info("stopping early (interrupted) -- saving %d samples collected so far", len(writer.samples))
    finally:
        stop_event.set()
        receiver.close()
        if stimulus_thread is not None:
            stimulus_thread.join(timeout=2.0)

    out_dir = writer.finish(
        transport_used=cfg.transport.mode,
        board_mac=board_mac,
        # Firmware doesn't yet report its own build ID over the wire;
        # the CSI wire-format version is the best available proxy today.
        firmware_version=f"wire-v{wire.VERSION}",
        dropped_samples=dropped,
    )
    n = len(writer.samples)
    loss_pct = 100.0 * dropped / (dropped + n) if (dropped + n) else 0.0
    elapsed = time.monotonic() - start
    avg_hz = n / elapsed if elapsed > 0 else 0.0
    logger.info(
        "wrote %d samples to %s -- avg %.1f Hz over %.0fs (%d dropped, %.1f%% loss)",
        n, out_dir, avg_hz, elapsed, dropped, loss_pct,
    )

    build_manifest(cfg)  # keep data/manifest.csv current after every session
    return 0


if __name__ == "__main__":
    sys.exit(main())
