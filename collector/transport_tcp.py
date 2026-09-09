"""TCP receiver: the laptop is the SERVER, the ESP32 dials in as a client.

This is deliberate -- it means the main data path never needs to discover
the ESP32's DHCP-assigned IP, only the laptop's own (which
scripts/get_laptop_ip.sh keeps current in config.yaml).
"""
from __future__ import annotations

import logging
import socket
import threading
from typing import Callable, Iterator, Optional

from collector.config import Config
from collector.models import CsiSample
from collector.wire import StreamFramer

logger = logging.getLogger(__name__)


def iter_samples(
    cfg: Config,
    stop_event: Optional[threading.Event] = None,
    on_stat_line: Optional[Callable[[str], None]] = None,
) -> Iterator[CsiSample]:
    host = cfg.network.laptop_ip
    port = cfg.network.tcp_port

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    server.settimeout(1.0)
    logger.info("TCP receiver listening on %s:%s", host, port)

    try:
        while stop_event is None or not stop_event.is_set():
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue

            logger.info("ESP32 connected from %s", addr)
            framer = StreamFramer(on_stat_line=on_stat_line)
            conn.settimeout(1.0)
            with conn:
                while stop_event is None or not stop_event.is_set():
                    try:
                        data = conn.recv(4096)
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    if not data:
                        break
                    yield from framer.feed(data)
            logger.warning("ESP32 disconnected -- waiting for reconnect")
    finally:
        server.close()
