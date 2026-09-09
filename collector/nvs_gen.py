"""Generates the NVS CSV for the firmware's "cfgstore" partition from
config.yaml, for whichever board role ("hotspot" or "receiver") is
being configured.

Only fields the firmware actually reads at runtime are included here --
capacity_test/dataset settings are laptop-only orchestration config and
are never pushed to the device. NVS key names are capped at 15 usable
characters (NVS_KEY_NAME_MAX_SIZE=16 including the null terminator) --
every key below respects that.
"""
from __future__ import annotations

import csv
from pathlib import Path

from collector.config import Config

NAMESPACE = "cfgstore"
ROLES = ("hotspot", "transmitter", "receiver")


def build_csv_rows(cfg: Config, role: str) -> list[tuple[str, str, str, str]]:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")

    rows: list[tuple[str, str, str, str]] = [(NAMESPACE, "namespace", "", "")]

    def add(key: str, encoding: str, value) -> None:
        assert len(key) <= 15, f"NVS key too long (>15 chars): {key!r}"
        rows.append((key, "data", encoding, str(value)))

    def add_bool(key: str, value: bool) -> None:
        add(key, "u8", 1 if value else 0)

    # Shared by all roles: which network to join (STA roles) or host (AP role).
    add("role", "string", role)
    add("ssid", "string", cfg.network.hotspot_ssid)
    add("pass", "string", cfg.network.hotspot_password)

    if role == "hotspot":
        add("ap_channel", "u8", cfg.hotspot.channel)
        return rows

    if role == "transmitter":
        # Optional 3rd-board role, not part of the default 2-board (hotspot +
        # receiver) topology -- needs a `transmitter:` section (serial_port,
        # tx_rate_hz, packet_size_bytes) and a matching TransmitterConfig
        # dataclass re-added to collector/config.py before this works.
        add("tx_rate_hz", "u16", cfg.transmitter.tx_rate_hz)
        add("tx_pkt_sz", "u16", cfg.transmitter.packet_size_bytes)
        return rows

    # role == "receiver": everything the existing single-board setup needed.
    add("laptop_ip", "string", cfg.network.laptop_ip)
    add("tcp_port", "u16", cfg.network.tcp_port)
    add("mode", "string", cfg.transport.mode)
    add("max_csi_len", "u16", cfg.capture.max_csi_payload_bytes)

    add_bool("lltf_en", cfg.csi.lltf_en)
    add_bool("htltf_en", cfg.csi.htltf_en)
    add_bool("stbc_en", cfg.csi.stbc_htltf2_en)
    add_bool("ltfmrg_en", cfg.csi.ltf_merge_en)
    add_bool("chfilt_en", cfg.csi.channel_filter_en)
    add_bool("manu_scale", cfg.csi.manu_scale)
    add("shift", "u8", cfg.csi.shift)
    add_bool("dumpack_en", cfg.csi.dump_ack_en)

    add_bool("promisc_en", cfg.csi.promiscuous)
    add("promisc_flt", "string", cfg.csi.promiscuous_filter)
    add_bool("bssidflt_en", cfg.csi.bssid_filter_enabled)

    return rows


def write_csv(cfg: Config, role: str, path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "type", "encoding", "value"])
        writer.writerows(build_csv_rows(cfg, role))


if __name__ == "__main__":
    import sys

    from collector.config import load_config

    role_arg = sys.argv[1] if len(sys.argv) > 1 else "receiver"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/cfgstore.csv"
    write_csv(load_config(), role_arg, out_path)
