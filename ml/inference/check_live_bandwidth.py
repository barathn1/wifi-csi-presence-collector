"""Check the CURRENT LIVE channel/bandwidth the board is actually seeing right now -- straight from the
board's real-time stream, not from a previously-recorded session file (see scripts/check_bandwidth.sh
for that). Connects, samples a handful of live packets, reports the dominant (cwb, channel_primary,
channel_secondary) combo, and exits -- it's a quick snapshot, not a continuous monitor.

    python3 -m ml.inference.check_live_bandwidth                  # just report what it sees
    python3 -m ml.inference.check_live_bandwidth --expect-mhz 20  # also exit non-zero on a mismatch,
                                                                   # usable as a preflight gate

Same mutual-exclusion rule as every other live tool in this project: stop collector.cli_collect /
ml.visualization.player_server / ml.inference.live_infer first, only one process can hold the
transport at a time.
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from collections import Counter

from collector import preflight, stimulus
from collector.config import load_config
from collector.receiver import get_receiver
from ml.data_pipeline.decode_csi import channel_width_summary_live

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--packets", type=int, default=100, help="how many live packets to sample before reporting")
    p.add_argument("--timeout", type=float, default=15.0, help="give up waiting for enough packets after this many seconds")
    p.add_argument("--expect-mhz", type=int, default=None, choices=[20, 40],
                   help="if given, exit 1 when the live bandwidth doesn't match")
    p.add_argument("--no-stimulus", action="store_true")
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config) if args.config else load_config()

    ok, reason = preflight.check_board_connected(cfg)
    if not ok:
        logger.error("preflight failed: %s", reason)
        return 1
    logger.info("preflight ok: %s", reason)

    stop_event = threading.Event()
    stimulus_thread = None
    if cfg.stimulus.enabled and not args.no_stimulus:
        stimulus_thread = threading.Thread(target=stimulus.run_stimulus, args=(cfg, stop_event), daemon=True)
        stimulus_thread.start()

    receiver = get_receiver(cfg, stop_event=stop_event)
    combo_counts: Counter = Counter()
    start = time.monotonic()
    try:
        for sample in receiver:
            combo_counts[(sample.cwb, sample.channel_primary, sample.channel_secondary)] += 1
            if sum(combo_counts.values()) >= args.packets:
                break
            if time.monotonic() - start >= args.timeout:
                logger.warning("only got %d/%d packets within %.0fs -- reporting anyway",
                                sum(combo_counts.values()), args.packets, args.timeout)
                break
    finally:
        stop_event.set()
        receiver.close()
        if stimulus_thread is not None:
            stimulus_thread.join(timeout=2.0)

    if not combo_counts:
        logger.error("no CSI packets received at all within %.0fs -- check the board/stimulus, not just bandwidth", args.timeout)
        return 2

    info = channel_width_summary_live(combo_counts)
    n_total = sum(combo_counts.values())
    print(f"LIVE: {info['description']} ({100 * info['fraction_of_packets']:.1f}% of {n_total} sampled packets)")

    if args.expect_mhz is not None:
        actual_mhz = 40 if info["cwb"] == 1 else 20
        if actual_mhz != args.expect_mhz:
            print(f"MISMATCH: expected {args.expect_mhz}MHz")
            return 1
        print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
