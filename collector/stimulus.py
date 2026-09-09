"""UDP traffic generator.

Promiscuous-mode CSI capture needs ambient WiFi traffic to overhear (a
plain associated STA only sees CSI from frames addressed to it, which is
too sparse). This sends UDP datagrams to the hotspot's gateway (the phone)
at a controlled rate so both the capacity test and real collection
sessions get a roughly steady, reproducible CSI rate. No listener is
required on the phone -- the ESP32 overhears the traffic on-air.
"""
from __future__ import annotations

import logging
import re
import socket
import subprocess
import threading
import time

from collector.config import Config

logger = logging.getLogger(__name__)

# Arbitrary, fixed port -- nothing needs to listen on it.
STIMULUS_UDP_PORT = 45871


def resolve_target_ip(cfg: Config) -> str:
    if cfg.stimulus.target_ip != "auto":
        return cfg.stimulus.target_ip
    out = subprocess.run(
        ["ip", "route", "show", "default"],
        capture_output=True, text=True, check=True,
    ).stdout
    m = re.search(r"default via (\d+\.\d+\.\d+\.\d+)", out)
    if not m:
        raise RuntimeError(f"could not find a default gateway in: {out!r}")
    return m.group(1)


def run_stimulus(cfg: Config, stop_event: threading.Event, rate_hz: float | None = None) -> None:
    """Blocks, sending datagrams until stop_event is set. Run on a
    dedicated thread."""
    target_ip = resolve_target_ip(cfg)
    rate_hz = rate_hz if rate_hz is not None else cfg.stimulus.rate_hz
    period = 1.0 / rate_hz
    payload = bytes(cfg.stimulus.packet_size_bytes)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    logger.info("Stimulus: sending to %s:%s at %s Hz", target_ip, STIMULUS_UDP_PORT, rate_hz)
    next_send = time.monotonic()
    try:
        while not stop_event.is_set():
            sock.sendto(payload, (target_ip, STIMULUS_UDP_PORT))
            next_send += period
            sleep_for = next_send - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_send = time.monotonic()
    finally:
        sock.close()
