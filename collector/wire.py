"""Binary CSI wire format shared by the TCP and serial transports.

Frame layout (little-endian, self-delimiting -- NOT newline-delimited,
since raw CSI bytes can legitimately contain 0x0A):

  magic(u16=0xC51A) | version(u8) | flags(u8, bit0=first_word_invalid) |
  seq(u32) | device_time_us(u64) | rssi(i8) | channel_primary(u8) |
  channel_secondary(u8) | sig_mode(u8) | mcs(u8) | cwb(u8) | stbc(u8) |
  noise_floor(u8) | src_mac(6B) | dst_mac(6B) | wifi_rx_seq(u16) |
  csi_len(u16) | csi_data[csi_len]

On serial, ASCII "#STAT,...\n" debug lines are interleaved with binary
frames. This module demuxes both from a single byte stream.
"""
from __future__ import annotations

import struct
import time
from typing import Callable, Optional

from collector.models import CsiSample

MAGIC = 0xC51A
VERSION = 1

# '<' = little-endian, no padding. rssi and noise_floor are signed (i8).
HEADER_FMT = "<HBBIQbBBBBBBb6s6sHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 40, HEADER_SIZE

_MAGIC_BYTES = struct.pack("<H", MAGIC)

FLAG_FIRST_WORD_INVALID = 0x01


def pack_frame(
    *,
    flags: int,
    seq: int,
    device_time_us: int,
    rssi: int,
    channel_primary: int,
    channel_secondary: int,
    sig_mode: int,
    mcs: int,
    cwb: int,
    stbc: int,
    noise_floor: int,
    src_mac: bytes,
    dst_mac: bytes,
    wifi_rx_seq: int,
    csi_data: bytes,
) -> bytes:
    """Build one wire frame. Used by test tooling; firmware builds the
    equivalent bytes in C from the same field layout."""
    header = struct.pack(
        HEADER_FMT,
        MAGIC,
        VERSION,
        flags,
        seq,
        device_time_us,
        rssi,
        channel_primary,
        channel_secondary,
        sig_mode,
        mcs,
        cwb,
        stbc,
        noise_floor,
        src_mac,
        dst_mac,
        wifi_rx_seq,
        len(csi_data),
    )
    return header + csi_data


class StreamFramer:
    """Feed raw bytes in; get back parsed CsiSample objects.

    Handles: partial reads (buffers until a full frame is available),
    resync after corruption (rescans for the next magic sequence),
    and ASCII "#STAT,..." lines interleaved on the same stream.
    """

    def __init__(self, on_stat_line: Optional[Callable[[str], None]] = None):
        self._buf = bytearray()
        self._on_stat_line = on_stat_line

    def feed(self, data: bytes) -> list[CsiSample]:
        self._buf.extend(data)
        samples: list[CsiSample] = []
        while True:
            sample = self._try_extract_one()
            if sample is None:
                break
            samples.append(sample)
        return samples

    def _try_extract_one(self) -> Optional[CsiSample]:
        buf = self._buf
        if not buf:
            return None

        # ASCII stat line: "#STAT,...\n"
        if buf[0:1] == b"#":
            nl = buf.find(b"\n")
            if nl == -1:
                return None  # wait for the rest of the line
            line = bytes(buf[:nl]).decode("ascii", errors="replace")
            del buf[: nl + 1]
            if self._on_stat_line is not None:
                self._on_stat_line(line)
            return self._try_extract_one()

        # Binary frame: must start with magic.
        if bytes(buf[:2]) != _MAGIC_BYTES:
            idx = buf.find(_MAGIC_BYTES, 1)
            hash_idx = buf.find(b"#", 1)
            if idx == -1 and hash_idx == -1:
                # Nothing recognizable ahead; keep only a tail long enough
                # to still catch a magic/hash split across feed() calls.
                if len(buf) > 4096:
                    del buf[:-1]
                return None
            candidates = [i for i in (idx, hash_idx) if i != -1]
            del buf[: min(candidates)]
            return self._try_extract_one()

        if len(buf) < HEADER_SIZE:
            return None  # wait for the rest of the header

        fields = struct.unpack_from(HEADER_FMT, buf, 0)
        (
            magic,
            version,
            flags,
            seq,
            device_time_us,
            rssi,
            channel_primary,
            channel_secondary,
            sig_mode,
            mcs,
            cwb,
            stbc,
            noise_floor,
            src_mac,
            dst_mac,
            wifi_rx_seq,
            csi_len,
        ) = fields

        if version != VERSION:
            # False-positive magic match on random data; skip past it and
            # keep scanning rather than getting stuck on the same bytes.
            del buf[:2]
            return self._try_extract_one()

        total_len = HEADER_SIZE + csi_len
        if len(buf) < total_len:
            return None  # wait for the rest of the payload

        csi_data = bytes(buf[HEADER_SIZE:total_len])
        del buf[:total_len]

        return CsiSample(
            seq=seq,
            device_time_us=device_time_us,
            host_recv_time_ns=time.time_ns(),
            first_word_invalid=bool(flags & FLAG_FIRST_WORD_INVALID),
            rssi=rssi,
            channel_primary=channel_primary,
            channel_secondary=channel_secondary,
            sig_mode=sig_mode,
            mcs=mcs,
            cwb=cwb,
            stbc=stbc,
            noise_floor=noise_floor,
            src_mac=bytes(src_mac),
            dst_mac=bytes(dst_mac),
            wifi_rx_seq=wifi_rx_seq,
            csi_len=csi_len,
            csi_data=csi_data,
        )
