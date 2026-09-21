"""Accumulates CsiSamples for one labeled session and flushes them to
data/<label>/<date>/<session_id>/{metadata.json, samples.npz}.

CSI payloads vary in length per packet (bandwidth/HT/STBC-dependent), so
they're stored as a flat concatenated buffer plus a per-packet offset
array (ragged/CSR-style) rather than a padded fixed-width table.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from collector.config import Config
from collector.models import CsiSample, SessionMetadata


def session_dir_for(cfg: Config, label: str, person_id: str, start_ts: float) -> Path:
    """Session output directory for a given label/person_id/start time --
    shared by SessionWriter's own default (single-board) path and by
    cli_collect.py's multi-board path, which computes one shared dir up
    front for every board in a run instead of letting each board derive
    its own from its (slightly different) first-sample time."""
    session_id = time.strftime("%Y%m%d_%H%M%S", time.localtime(start_ts))
    if person_id:
        session_id += f"_{person_id}"
    date_str = time.strftime("%Y-%m-%d", time.localtime(start_ts))
    return cfg.dataset_dir() / label / date_str / session_id


class SessionWriter:
    def __init__(self, cfg: Config, label: str, person_id: str, notes: str, motion: str = ""):
        if label not in cfg.dataset.labels:
            raise ValueError(f"label {label!r} not in {cfg.dataset.labels}")
        self.cfg = cfg
        self.label = label
        self.person_id = person_id
        self.notes = notes
        self.motion = motion
        self.samples: list[CsiSample] = []
        self.start_ts = time.time()

    def add(self, sample: CsiSample) -> None:
        self.samples.append(sample)

    def finish(
        self,
        transport_used: str,
        board_mac: str,
        firmware_version: str,
        dropped_samples: int = 0,
        board_tag: str = "",
        session_dir: Optional[Path] = None,
    ) -> Path:
        """`session_dir`, when given, is a shared folder that multiple
        boards' files get written into together (cli_collect.py's
        multi-board path passes one shared dir for every board in a run,
        computed up front) -- `board_tag` then becomes a filename prefix
        (`<tag>_samples.npz`/`<tag>_metadata.json`) so files from the same
        run are obviously grouped and never collide, instead of each
        board getting its own similarly-but-not-identically-timestamped
        subfolder. Leave both blank for the single-board case: unchanged
        `samples.npz`/`metadata.json` in their own session dir."""
        end_ts = time.time()
        duration_s = end_ts - self.start_ts
        avg_rate_hz = len(self.samples) / duration_s if duration_s > 0 else 0.0

        if session_dir is not None:
            out_dir = session_dir
            prefix = f"{board_tag}_" if board_tag else ""
        else:
            out_dir = session_dir_for(self.cfg, self.label, self.person_id, self.start_ts)
            prefix = ""
        out_dir.mkdir(parents=True, exist_ok=True)

        self._write_samples(out_dir / f"{prefix}samples.npz")

        meta = SessionMetadata(
            label=self.label,
            person_id=self.person_id,
            notes=self.notes,
            start_ts=self.start_ts,
            end_ts=end_ts,
            duration_s=duration_s,
            transport_used=transport_used,
            board_mac=board_mac,
            firmware_version=firmware_version,
            config_snapshot=self.cfg.raw,
            sample_count=len(self.samples),
            ap_source=self.cfg.network.ap_source,
            motion=self.motion,
            dropped_samples=dropped_samples,
            avg_rate_hz=avg_rate_hz,
        )
        (out_dir / f"{prefix}metadata.json").write_text(json.dumps(asdict(meta), indent=2))
        return out_dir

    def _write_samples(self, path: Path) -> None:
        n = len(self.samples)
        seq = np.empty(n, dtype=np.uint32)
        device_time_us = np.empty(n, dtype=np.uint64)
        host_recv_time_ns = np.empty(n, dtype=np.int64)
        first_word_invalid = np.empty(n, dtype=bool)
        rssi = np.empty(n, dtype=np.int8)
        channel_primary = np.empty(n, dtype=np.uint8)
        channel_secondary = np.empty(n, dtype=np.uint8)
        sig_mode = np.empty(n, dtype=np.uint8)
        mcs = np.empty(n, dtype=np.uint8)
        cwb = np.empty(n, dtype=np.uint8)
        stbc = np.empty(n, dtype=np.uint8)
        noise_floor = np.empty(n, dtype=np.int8)
        wifi_rx_seq = np.empty(n, dtype=np.uint16)
        csi_len = np.empty(n, dtype=np.uint16)
        csi_offset = np.empty(n, dtype=np.uint32)
        src_mac = np.empty((n, 6), dtype=np.uint8)
        dst_mac = np.empty((n, 6), dtype=np.uint8)

        csi_chunks = []
        offset = 0
        for i, s in enumerate(self.samples):
            seq[i] = s.seq
            device_time_us[i] = s.device_time_us
            host_recv_time_ns[i] = s.host_recv_time_ns
            first_word_invalid[i] = s.first_word_invalid
            rssi[i] = s.rssi
            channel_primary[i] = s.channel_primary
            channel_secondary[i] = s.channel_secondary
            sig_mode[i] = s.sig_mode
            mcs[i] = s.mcs
            cwb[i] = s.cwb
            stbc[i] = s.stbc
            noise_floor[i] = s.noise_floor
            wifi_rx_seq[i] = s.wifi_rx_seq
            csi_len[i] = s.csi_len
            csi_offset[i] = offset
            src_mac[i] = np.frombuffer(s.src_mac, dtype=np.uint8)
            dst_mac[i] = np.frombuffer(s.dst_mac, dtype=np.uint8)
            csi_chunks.append(np.frombuffer(s.csi_data, dtype=np.int8))
            offset += s.csi_len

        csi_flat = np.concatenate(csi_chunks) if csi_chunks else np.empty(0, dtype=np.int8)

        np.savez_compressed(
            path,
            seq=seq,
            device_time_us=device_time_us,
            host_recv_time_ns=host_recv_time_ns,
            first_word_invalid=first_word_invalid,
            rssi=rssi,
            channel_primary=channel_primary,
            channel_secondary=channel_secondary,
            sig_mode=sig_mode,
            mcs=mcs,
            cwb=cwb,
            stbc=stbc,
            noise_floor=noise_floor,
            wifi_rx_seq=wifi_rx_seq,
            csi_len=csi_len,
            csi_offset=csi_offset,
            src_mac=src_mac,
            dst_mac=dst_mac,
            csi_flat=csi_flat,
        )
