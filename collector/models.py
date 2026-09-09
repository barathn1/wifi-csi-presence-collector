"""Shared data structures for CSI samples and session metadata."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CsiSample:
    seq: int
    device_time_us: int
    host_recv_time_ns: int
    first_word_invalid: bool
    rssi: int
    channel_primary: int
    channel_secondary: int
    sig_mode: int
    mcs: int
    cwb: int
    stbc: int
    noise_floor: int
    src_mac: bytes
    dst_mac: bytes
    wifi_rx_seq: int
    csi_len: int
    csi_data: bytes

    @property
    def src_mac_str(self) -> str:
        return ":".join(f"{b:02x}" for b in self.src_mac)

    @property
    def dst_mac_str(self) -> str:
        return ":".join(f"{b:02x}" for b in self.dst_mac)


@dataclass
class SessionMetadata:
    label: str
    person_id: str
    notes: str
    start_ts: float
    end_ts: float
    duration_s: float
    transport_used: str
    board_mac: str
    firmware_version: str
    config_snapshot: dict
    sample_count: int
    ap_source: str = ""  # "esp" or an external-AP label (e.g. "android", "google-ap")
    motion: str = ""     # "standing" | "walking" -- protocol used during this session
    dropped_samples: int = 0
    avg_rate_hz: float = 0.0
    format_version: int = 1
