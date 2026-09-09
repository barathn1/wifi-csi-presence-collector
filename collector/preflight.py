"""Checks the ESP32 has actually associated to the hotspot before a
collection/capacity-test session starts, so a disconnected board produces
a clear error instead of a silently-empty dataset.

Two independent, non-invasive signals:
  1. The laptop's own WiFi interface is associated to the configured SSID.
  2. The board's MAC shows up in the laptop's ARP/neighbor table on that
     interface.

The board's MAC is normally read from config.yaml's device.mac (cached by
scripts/detect_board.sh --set-mac) rather than queried live via esptool,
because every esptool interaction resets the board (it must enter the ROM
bootloader to talk to the chip) -- fine to do once during setup, too
disruptive to do before every session/capacity-test iteration.
"""
from __future__ import annotations

import re
import subprocess
import time

from collector.config import Config

ESPTOOL_CANDIDATES = ["esptool", "esptool.py"]

# After a board reset (e.g. from an NVS reflash), give it this long to
# reboot, rejoin the hotspot, and show up in the ARP table before giving up.
RECONNECT_TIMEOUT_S = 20.0
RECONNECT_POLL_INTERVAL_S = 1.0


def _run(cmd: list[str], timeout: float = 10.0) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def query_board_mac_via_usb(cfg: Config) -> str | None:
    """Reads the MAC directly over USB via esptool. This resets the board
    (esptool must enter the ROM bootloader) -- only call this during
    one-time setup (scripts/detect_board.sh --set-mac), not in a hot loop."""
    for exe in ESPTOOL_CANDIDATES:
        out = _run([exe, "--port", cfg.transport.serial_port, "chip-id"], timeout=15.0)
        m = re.search(r"MAC:\s*([0-9A-Fa-f:]{17})", out)
        if m:
            return m.group(1).lower()
    return None


def get_board_mac(cfg: Config) -> str | None:
    """Returns the cached MAC from config.yaml if set, without touching
    the board. Falls back to a live (reset-inducing) USB query if no MAC
    is cached yet."""
    if cfg.device.mac:
        return cfg.device.mac.lower()
    return query_board_mac_via_usb(cfg)


def check_laptop_associated(cfg: Config) -> tuple[bool, str]:
    iface = cfg.network.laptop_iface
    out = _run(["iw", "dev", iface, "link"])
    if "Not connected" in out or not out.strip():
        return False, f"laptop interface {iface} is not associated to any WiFi network"
    m = re.search(r"SSID:\s*(.+)", out)
    ssid = m.group(1).strip() if m else None
    if ssid != cfg.network.hotspot_ssid:
        return False, (
            f"laptop interface {iface} is on SSID {ssid!r}, "
            f"expected {cfg.network.hotspot_ssid!r}"
        )
    return True, f"laptop is associated to {ssid!r}"


def check_board_in_neighbor_table(cfg: Config, board_mac: str) -> tuple[bool, str]:
    iface = cfg.network.laptop_iface
    out = _run(["ip", "neigh", "show", "dev", iface])
    macs = {m.lower() for m in re.findall(r"([0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5})", out)}
    if board_mac in macs:
        return True, f"board MAC {board_mac} present in ARP table on {iface}"
    return False, (
        f"board MAC {board_mac} not seen in ARP table on {iface} yet "
        f"(neighbors seen: {sorted(macs) or 'none'})"
    )


def check_board_connected(cfg: Config) -> tuple[bool, str]:
    """Returns (ok, reason). Retries the association/ARP checks for up to
    RECONNECT_TIMEOUT_S -- the board may have just rebooted (e.g. after an
    NVS config push) and needs a few seconds to rejoin the hotspot."""
    board_mac = get_board_mac(cfg)
    if board_mac is None:
        return False, (
            f"could not read the board's MAC over USB on "
            f"{cfg.transport.serial_port} -- check the board is powered "
            f"and connected, and that no other process (e.g. idf.py monitor) "
            f"is holding the port open"
        )

    deadline = time.monotonic() + RECONNECT_TIMEOUT_S
    last_reason = ""
    while True:
        ssid_ok, ssid_reason = check_laptop_associated(cfg)
        if ssid_ok:
            neigh_ok, neigh_reason = check_board_in_neighbor_table(cfg, board_mac)
            if neigh_ok:
                return True, f"board {board_mac} connected: {ssid_reason}; {neigh_reason}"
            last_reason = neigh_reason
        else:
            last_reason = ssid_reason

        if time.monotonic() >= deadline:
            return False, f"{last_reason} (gave up after {RECONNECT_TIMEOUT_S:.0f}s)"
        time.sleep(RECONNECT_POLL_INTERVAL_S)
