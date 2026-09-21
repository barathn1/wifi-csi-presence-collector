"""TCP receiver: the laptop is the SERVER, the ESP32 dials in as a client.

This is deliberate -- it means the main data path never needs to discover
the ESP32's DHCP-assigned IP, only the laptop's own (which
scripts/get_laptop_ip.sh keeps current in config.yaml).
"""
from __future__ import annotations

import logging
import queue
import re
import socket
import subprocess
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


def resolve_mac_for_ip(iface: str, ip: str) -> Optional[str]:
    """Best-effort ARP/neighbor-table lookup, so multi-board sessions can
    tag output with the board's real MAC instead of just its DHCP IP.
    Purely cosmetic (see the `key` docstring on iter_multi_samples for why
    the peer IP, not this, is what identifies a board) -- returns None
    if the entry isn't in the table yet rather than blocking on it."""
    try:
        out = subprocess.run(
            ["ip", "neigh", "show", "dev", iface, ip],
            capture_output=True, text=True, timeout=5.0,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"lladdr\s+([0-9A-Fa-f:]{17})", out)
    return m.group(1).lower() if m else None


def _handle_connection(
    conn: socket.socket,
    peer_ip: str,
    out_q: "queue.Queue[tuple[str, CsiSample]]",
    stop_event: Optional[threading.Event],
    on_stat_line: Optional[Callable[[str], None]],
) -> None:
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
            for sample in framer.feed(data):
                out_q.put((peer_ip, sample))
    logger.warning("ESP32 at %s disconnected", peer_ip)


def iter_multi_samples(
    cfg: Config,
    stop_event: Optional[threading.Event] = None,
    on_stat_line: Optional[Callable[[str], None]] = None,
) -> Iterator[tuple[str, CsiSample]]:
    """Like iter_samples, but accepts as many concurrent ESP32 connections
    as show up instead of exactly one -- yields (key, sample) pairs so the
    caller can fan CSI out to one dataset file per board.

    `key` is the board's peer IP (stable for the life of the connection,
    unlike its MAC which isn't on the wire anywhere -- see wire.py). One
    thread per accepted connection feeds a shared queue that this
    generator drains, so N boards streaming at once don't block each
    other.
    """
    host = cfg.network.laptop_ip
    port = cfg.network.tcp_port

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(8)
    server.settimeout(0.2)  # short, so queued samples aren't held back waiting on a new connection
    logger.info("TCP receiver listening on %s:%s (multi-board)", host, port)

    out_q: "queue.Queue[tuple[str, CsiSample]]" = queue.Queue()
    threads: list[threading.Thread] = []
    try:
        while stop_event is None or not stop_event.is_set():
            try:
                conn, addr = server.accept()
            except socket.timeout:
                pass
            else:
                logger.info("ESP32 connected from %s (%d board(s) now connected)", addr, len(threads) + 1)
                t = threading.Thread(
                    target=_handle_connection,
                    args=(conn, addr[0], out_q, stop_event, on_stat_line),
                    daemon=True,
                )
                t.start()
                threads.append(t)

            while True:
                try:
                    yield out_q.get_nowait()
                except queue.Empty:
                    break
    finally:
        server.close()
        for t in threads:
            t.join(timeout=2.0)
