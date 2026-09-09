"""Compares TCP vs USB-serial CSI throughput/loss on real hardware and
picks a default transport. This is what answers "what's the ESP's
capacity" with real numbers instead of a guess.

    python -m collector.capacity_test --sweep
    python -m collector.capacity_test --sweep --apply
    python -m collector.capacity_test --find-max-rate --transport tcp --apply

For each (transport, rate) combination: pushes that config to the board's
NVS (via scripts/push_config.sh), waits for it to reassociate, drives a
controlled UDP stimulus at the target rate, and measures drop%/effective
Hz/jitter from real received frames cross-checked against the board's own
"#STAT" on-device counters (the ground truth "packets generated" count).

Note: the "#STAT" debug line is always emitted over USB serial regardless
of which transport carries CSI data (see firmware design). When TCP is
under test, CSI flows over the network but a *separate* thread must tail
serial just for those stat lines -- the TCP receiver never touches the
serial port at all.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import serial

from collector import preflight, stimulus
from collector.config import Config, load_config
from collector.config_editor import set_key_inplace
from collector.receiver import get_receiver
from collector.wire import StreamFramer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Below this many received samples, drop%/jitter are statistically
# meaningless (a handful of samples can trivially show 0% drop) and must
# never be allowed to "win" over a well-populated result.
MIN_VALID_SAMPLES = 30


@dataclass
class CapacityResult:
    transport: str
    rate_hz: float
    packets_sent: int
    packets_received: int
    drop_pct: float
    effective_rate_hz: float
    jitter_mean_ms: float
    jitter_stdev_ms: float
    jitter_p95_ms: float
    passed: bool
    note: str = ""


class StatTracker:
    """Parses '#STAT,cb=..,enq=..,dqf=..,dtx=..,upt=..' lines and keeps the
    latest cumulative counters."""

    def __init__(self):
        self.lock = threading.Lock()
        self.latest: dict[str, int] = {}

    def on_line(self, line: str) -> None:
        if not line.startswith("#STAT"):
            return
        fields = {}
        for part in line.split(",")[1:]:
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            try:
                fields[k] = int(v)
            except ValueError:
                pass
        with self.lock:
            self.latest = fields


def push_transport_config(cfg: Config, transport: str, rate_hz: float) -> Config:
    set_key_inplace(cfg.path, "transport.mode", transport)
    set_key_inplace(cfg.path, "capture.target_sample_rate_hz", int(rate_hz))
    subprocess.run([str(REPO_ROOT / "scripts" / "push_config.sh"), "receiver"], check=True)
    return load_config(cfg.path)


def _tail_serial_stats(cfg: Config, stats: StatTracker, stop_event: threading.Event) -> None:
    """Only used when transport == tcp: tail the USB serial link purely
    for '#STAT' lines, independent of the TCP data path."""
    ser = serial.Serial(cfg.transport.serial_port, cfg.transport.serial_baud, timeout=1.0)
    framer = StreamFramer(on_stat_line=stats.on_line)
    try:
        while not stop_event.is_set():
            data = ser.read(4096)
            if data:
                framer.feed(data)
    finally:
        ser.close()


def run_one(cfg: Config, transport: str, rate_hz: float) -> CapacityResult:
    cfg = push_transport_config(cfg, transport, rate_hz)

    ok, reason = preflight.check_board_connected(cfg)
    if not ok:
        logger.error("skipping transport=%s rate=%s: %s", transport, rate_hz, reason)
        return CapacityResult(transport, rate_hz, 0, 0, 100.0, 0.0, 0.0, 0.0, 0.0, False, note=reason)

    stats = StatTracker()
    stop_event = threading.Event()

    stim_thread = threading.Thread(
        target=stimulus.run_stimulus, args=(cfg, stop_event, rate_hz), daemon=True
    )
    stim_thread.start()

    stats_thread = None
    if transport == "tcp":
        stats_thread = threading.Thread(
            target=_tail_serial_stats, args=(cfg, stats, stop_event), daemon=True
        )
        stats_thread.start()
        receiver = get_receiver(cfg, stop_event=stop_event)
    else:
        receiver = get_receiver(cfg, stop_event=stop_event, on_stat_line=stats.on_line)

    seqs: list[int] = []
    arrivals_ns: list[int] = []

    def consume():
        for sample in receiver:
            seqs.append(sample.seq)
            arrivals_ns.append(sample.host_recv_time_ns)

    consume_thread = threading.Thread(target=consume, daemon=True)
    consume_thread.start()

    logger.info(
        "running capacity test: transport=%s rate=%s Hz for %ss",
        transport, rate_hz, cfg.capacity_test.duration_s,
    )
    time.sleep(cfg.capacity_test.duration_s)

    stop_event.set()
    # Do NOT call receiver.close() here: consume_thread (a different
    # thread) is driving this generator, and closing a generator from a
    # thread other than the one executing it raises "generator already
    # executing". Setting stop_event is enough -- the generator's own
    # loop notices within ~1s (socket timeout) and returns on its own,
    # which consume_thread's `for` loop observes as normal completion.
    stim_thread.join(timeout=2.0)
    if stats_thread is not None:
        stats_thread.join(timeout=2.0)
    consume_thread.join(timeout=2.0)

    packets_received = len(set(seqs))
    packets_sent = stats.latest.get("cb")
    note = ""
    if packets_sent is None:
        note = "no #STAT line observed -- drop%% is unreliable (using packets_received as denominator)"
        logger.warning(note)
        packets_sent = packets_received

    # Clamp to 0: a negative value only ever means the on-device "cb" stat
    # snapshot was stale (its last print predates our exact stop instant
    # by up to one ~2s report interval) relative to packets_received, not
    # that more packets arrived than were ever generated.
    drop_pct = max(0.0, 100.0 * (1 - packets_received / packets_sent)) if packets_sent else 100.0

    if len(arrivals_ns) >= 2:
        span_s = (arrivals_ns[-1] - arrivals_ns[0]) / 1e9
        effective_rate_hz = (len(arrivals_ns) - 1) / span_s if span_s > 0 else 0.0
        deltas_ms = [(b - a) / 1e6 for a, b in zip(arrivals_ns, arrivals_ns[1:])]
        nominal_ms = 1000.0 / rate_hz
        jitter_ms = sorted(abs(d - nominal_ms) for d in deltas_ms)
        jitter_mean = statistics.mean(jitter_ms)
        jitter_stdev = statistics.stdev(jitter_ms) if len(jitter_ms) > 1 else 0.0
        jitter_p95 = jitter_ms[int(0.95 * (len(jitter_ms) - 1))]
    else:
        effective_rate_hz = 0.0
        jitter_mean = jitter_stdev = jitter_p95 = 0.0

    passed = (
        packets_received >= MIN_VALID_SAMPLES
        and drop_pct <= cfg.capacity_test.max_acceptable_drop_pct
        and jitter_p95 <= cfg.capacity_test.max_acceptable_jitter_ms
    )
    if packets_received < MIN_VALID_SAMPLES:
        note = (note + "; " if note else "") + (
            f"only {packets_received} samples received -- below the "
            f"{MIN_VALID_SAMPLES}-sample validity floor, result is not trustworthy"
        )

    return CapacityResult(
        transport=transport,
        rate_hz=rate_hz,
        packets_sent=packets_sent,
        packets_received=packets_received,
        drop_pct=drop_pct,
        effective_rate_hz=effective_rate_hz,
        jitter_mean_ms=jitter_mean,
        jitter_stdev_ms=jitter_stdev,
        jitter_p95_ms=jitter_p95,
        passed=passed,
        note=note,
    )


def run_sweep(cfg: Config, transports: list[str], rates: list[float]) -> list[CapacityResult]:
    results = []
    for transport in transports:
        for rate in rates:
            result = run_one(cfg, transport, rate)
            logger.info("result: %s", result)
            results.append(result)
    return results


def pick_winner(results: list[CapacityResult], desired_rate: float) -> tuple[str, float]:
    at_rate = [r for r in results if r.rate_hz == desired_rate]
    passing = {r.transport: r for r in at_rate if r.passed}
    for preferred in ("tcp", "serial"):
        if preferred in passing:
            return preferred, desired_rate

    # Fallback: pick the lowest-drop result, but ONLY among results with
    # enough samples to be statistically meaningful -- a handful of
    # samples can trivially show 0% drop and must never outrank a
    # well-populated, slightly-lossy result.
    candidates = [r for r in results if r.packets_received >= MIN_VALID_SAMPLES]
    if not candidates:
        logger.warning(
            "no result anywhere reached the %d-sample validity floor -- "
            "something is likely wrong with the setup, not just the rate",
            MIN_VALID_SAMPLES,
        )
        candidates = results
    if not candidates:
        raise RuntimeError("no capacity test results to pick from")

    best = min(candidates, key=lambda r: r.drop_pct)
    logger.warning(
        "no transport passed at %s Hz; falling back to lowest-drop option: "
        "%s at %s Hz (%.1f%% drop, %d samples) -- consider lowering target_sample_rate_hz",
        desired_rate, best.transport, best.rate_hz, best.drop_pct, best.packets_received,
    )
    return best.transport, best.rate_hz


def find_max_rate(cfg: Config, transport: str, rates: list[float]) -> float | None:
    """Scans ascending rates, returns the highest one that still passes,
    stopping at the first failure."""
    max_passing = None
    for rate in sorted(rates):
        result = run_one(cfg, transport, rate)
        logger.info("result: %s", result)
        if result.passed:
            max_passing = rate
        else:
            break
    return max_passing


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compare TCP vs serial CSI throughput/loss on real hardware.")
    p.add_argument("--sweep", action="store_true", help="test every (transport, rate) combination")
    p.add_argument("--find-max-rate", action="store_true", help="find the max sustainable rate for --transport")
    p.add_argument("--transport", choices=["tcp", "serial"], help="required with --find-max-rate")
    p.add_argument("--apply", action="store_true", help="write the winning transport/rate back to config.yaml")
    p.add_argument("--config", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    original_mode = cfg.transport.mode
    original_rate = cfg.capture.target_sample_rate_hz
    # Every push_transport_config() call during the sweep mutates
    # config.yaml as a side effect (push_config.sh reads it from disk).
    # Snapshot the raw bytes now and restore them before making the
    # final decision, so config.yaml never reflects an arbitrary
    # intermediate test combo -- only the pre-sweep state or the
    # explicitly applied winner.
    original_bytes = cfg.path.read_bytes()

    if args.find_max_rate:
        if not args.transport:
            logger.error("--find-max-rate requires --transport")
            return 2
        max_rate = find_max_rate(cfg, args.transport, cfg.capacity_test.candidate_rates_hz)
        logger.info("max sustainable rate for %s: %s Hz", args.transport, max_rate)
        cfg.path.write_bytes(original_bytes)
        if args.apply and max_rate is not None:
            push_transport_config(cfg, args.transport, max_rate)
        else:
            push_transport_config(cfg, original_mode, original_rate)
        return 0

    if args.sweep:
        results = run_sweep(cfg, ["tcp", "serial"], cfg.capacity_test.candidate_rates_hz)
        cfg.path.write_bytes(original_bytes)
        report = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "results": [asdict(r) for r in results],
        }
        report_dir = REPO_ROOT / "reports" / "capacity"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_capacity_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        logger.info("wrote report to %s", report_path)

        winner_transport, winner_rate = pick_winner(results, original_rate)
        logger.info("decision: transport=%s rate=%s Hz", winner_transport, winner_rate)

        if args.apply:
            push_transport_config(cfg, winner_transport, winner_rate)
        else:
            push_transport_config(cfg, original_mode, original_rate)
        return 0

    logger.error("specify --sweep or --find-max-rate")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
