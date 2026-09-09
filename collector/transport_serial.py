"""Serial receiver: reads the ESP32's USB-CDC stream directly.

Binary CSI frames and ASCII "#STAT,..." debug lines are interleaved on the
same stream; StreamFramer demuxes both.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Iterator, Optional

import serial

from collector.config import Config
from collector.models import CsiSample
from collector.wire import StreamFramer

logger = logging.getLogger(__name__)


def iter_samples(
    cfg: Config,
    stop_event: Optional[threading.Event] = None,
    on_stat_line: Optional[Callable[[str], None]] = None,
) -> Iterator[CsiSample]:
    port = cfg.transport.serial_port
    baud = cfg.transport.serial_baud

    ser = serial.Serial(port, baud, timeout=1.0)
    logger.info("Serial receiver open on %s @ %s baud", port, baud)
    framer = StreamFramer(on_stat_line=on_stat_line)
    try:
        while stop_event is None or not stop_event.is_set():
            data = ser.read(4096)
            if not data:
                continue
            yield from framer.feed(data)
    finally:
        ser.close()
