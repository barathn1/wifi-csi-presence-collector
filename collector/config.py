"""Loads config/config.yaml into typed dataclasses.

This is the ONLY place config.yaml gets parsed -- every other module takes
a Config object, it never re-reads the YAML file itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


@dataclass
class DeviceConfig:
    board_id_hint: str
    mac: str


@dataclass
class NetworkConfig:
    ap_source: str  # "esp" (an ESP32 hosts it) or any other label describing the
                     # external AP hardware in use (e.g. "android", "google-ap")
    hotspot_ssid: str
    hotspot_password: str
    laptop_iface: str
    laptop_ip: str
    tcp_port: int


@dataclass
class HotspotConfig:
    serial_port: str
    channel: int


@dataclass
class TransportConfig:
    mode: str  # "tcp" | "serial"
    serial_port: str
    serial_baud: int


@dataclass
class CsiConfig:
    lltf_en: bool
    htltf_en: bool
    stbc_htltf2_en: bool
    ltf_merge_en: bool
    channel_filter_en: bool
    manu_scale: bool
    shift: int
    dump_ack_en: bool
    promiscuous: bool
    promiscuous_filter: str
    bssid_filter_enabled: bool


@dataclass
class CaptureConfig:
    target_sample_rate_hz: int
    max_csi_payload_bytes: int


@dataclass
class StimulusConfig:
    enabled: bool
    target_ip: str  # "auto" resolves to the laptop's default-route gateway
    rate_hz: int
    packet_size_bytes: int


@dataclass
class CapacityTestConfig:
    duration_s: int
    candidate_rates_hz: list[int]
    max_acceptable_drop_pct: float
    max_acceptable_jitter_ms: float


@dataclass
class DatasetConfig:
    output_dir: str
    labels: list[str]


@dataclass
class FirmwareConfig:
    idf_path: str
    project_dir: str


@dataclass
class Config:
    device: DeviceConfig
    network: NetworkConfig
    hotspot: HotspotConfig
    transport: TransportConfig
    csi: CsiConfig
    capture: CaptureConfig
    stimulus: StimulusConfig
    capacity_test: CapacityTestConfig
    dataset: DatasetConfig
    firmware: FirmwareConfig
    path: Path
    raw: dict[str, Any] = field(repr=False)

    def dataset_dir(self) -> Path:
        p = Path(self.dataset.output_dir)
        return p if p.is_absolute() else (self.path.parent.parent / p).resolve()

    def firmware_dir(self) -> Path:
        p = Path(self.firmware.project_dir)
        return p if p.is_absolute() else (self.path.parent.parent / p).resolve()


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    return Config(
        device=DeviceConfig(**raw["device"]),
        network=NetworkConfig(**raw["network"]),
        hotspot=HotspotConfig(**raw["hotspot"]),
        transport=TransportConfig(**raw["transport"]),
        csi=CsiConfig(**raw["csi"]),
        capture=CaptureConfig(**raw["capture"]),
        stimulus=StimulusConfig(**raw["stimulus"]),
        capacity_test=CapacityTestConfig(**raw["capacity_test"]),
        dataset=DatasetConfig(**raw["dataset"]),
        firmware=FirmwareConfig(**raw["firmware"]),
        path=path,
        raw=raw,
    )
