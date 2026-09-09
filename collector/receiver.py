"""Single entry point for CSI ingestion, dispatching by transport.mode.

cli_collect.py and capacity_test.py both go through this so they exercise
identical parsing/framing code regardless of transport.
"""
from __future__ import annotations

import threading
from typing import Callable, Iterator, Optional

from collector import transport_serial, transport_tcp
from collector.config import Config
from collector.models import CsiSample


def get_receiver(
    cfg: Config,
    stop_event: Optional[threading.Event] = None,
    on_stat_line: Optional[Callable[[str], None]] = None,
) -> Iterator[CsiSample]:
    if cfg.transport.mode == "tcp":
        return transport_tcp.iter_samples(cfg, stop_event=stop_event, on_stat_line=on_stat_line)
    if cfg.transport.mode == "serial":
        return transport_serial.iter_samples(cfg, stop_event=stop_event, on_stat_line=on_stat_line)
    raise ValueError(f"unknown transport.mode: {cfg.transport.mode!r}")
