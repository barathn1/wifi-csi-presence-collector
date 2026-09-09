"""View everything inside a collected session (metadata.json + samples.npz).

    python3 -m collector.inspect_npz data/authorized/2026-09-08/20260908_073213
    python3 -m collector.inspect_npz data/authorized/2026-09-08/20260908_073213/samples.npz
    python3 -m collector.inspect_npz <session_dir> --sample 5   # full detail on one sample, incl. decoded CSI
    python3 -m collector.inspect_npz <session_dir> --csv        # dump every field, incl. raw CSI, to a CSV

With no flags, prints: the session's metadata, a per-field summary (dtype,
shape, min/max) for every array in samples.npz, and a breakdown of how many
samples had each csi_len value (i.e. how many subcarrier readings each
packet type carried).

`--csv` always writes to <session_dir>/csi_export.csv -- next to the
samples.npz/metadata.json it was generated from, not wherever the shell
happens to be, so exports are always found in the same predictable place.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def resolve_npz_path(path: Path) -> Path:
    if path.is_dir():
        return path / "samples.npz"
    return path


def print_metadata(session_dir: Path) -> None:
    meta_path = session_dir / "metadata.json"
    if not meta_path.exists():
        print(f"(no metadata.json found at {meta_path})")
        return
    meta = json.loads(meta_path.read_text())
    print("=== metadata.json ===")
    for key, value in meta.items():
        if key == "config_snapshot":
            continue  # huge; available in the file if you need it
        print(f"  {key}: {value}")
    print("  (config_snapshot omitted here -- open metadata.json directly to see it)")
    print()


def print_summary(data: np.lib.npyio.NpzFile) -> None:
    n = len(data["seq"])
    print(f"=== samples.npz: {n} samples ===")
    for key in data.files:
        arr = data[key]
        if arr.ndim == 1 and arr.dtype.kind in "iuf" and len(arr) > 0:
            print(f"  {key:18s} dtype={str(arr.dtype):8s} shape={arr.shape!s:12s} "
                  f"min={arr.min()} max={arr.max()} mean={arr.mean():.2f}")
        else:
            print(f"  {key:18s} dtype={str(arr.dtype):8s} shape={arr.shape}")
    print()

    if n == 0:
        return

    device_span_s = (int(data["device_time_us"][-1]) - int(data["device_time_us"][0])) / 1e6
    if device_span_s > 0:
        print(f"  observed rate: {n / device_span_s:.1f} Hz over {device_span_s:.1f}s "
              f"(device_time_us span)")

    print("\n  csi_len breakdown (2 bytes per subcarrier reading -- imaginary,real "
          "int8 pairs; this varies per packet by design, see README):")
    lens, counts = np.unique(data["csi_len"], return_counts=True)
    for length, count in zip(lens, counts):
        print(f"    csi_len={length:4d} bytes -> {length // 2:3d} subcarrier readings "
              f"-- {count} samples ({100 * count / n:.1f}%)")
    print()


def print_one_sample(data: np.lib.npyio.NpzFile, index: int) -> None:
    n = len(data["seq"])
    if not (0 <= index < n):
        print(f"index {index} out of range (0..{n - 1})")
        return

    print(f"=== sample #{index} (full detail) ===")
    scalar_fields = [k for k in data.files if k not in ("csi_flat", "csi_offset", "src_mac", "dst_mac")]
    for key in scalar_fields:
        print(f"  {key}: {data[key][index]}")

    mac_fmt = lambda m: ":".join(f"{b:02x}" for b in m)
    print(f"  src_mac: {mac_fmt(data['src_mac'][index])}")
    print(f"  dst_mac: {mac_fmt(data['dst_mac'][index])}")

    off = int(data["csi_offset"][index])
    length = int(data["csi_len"][index])
    csi = data["csi_flat"][off:off + length]
    print(f"  csi_data ({length} bytes = {length // 2} subcarriers, as (imag, real) int8 pairs):")
    pairs = [(int(a), int(b)) for a, b in zip(csi[0::2], csi[1::2])]
    print(f"    {pairs}")


def dump_csv(data: np.lib.npyio.NpzFile, out_path: Path) -> None:
    scalar_fields = [k for k in data.files if k not in ("csi_flat", "csi_offset", "src_mac", "dst_mac")]
    n = len(data["seq"])
    mac_fmt = lambda m: ":".join(f"{b:02x}" for b in m)

    csi_flat = data["csi_flat"]
    csi_offset = data["csi_offset"]
    csi_len = data["csi_len"]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(scalar_fields + ["src_mac", "dst_mac", "csi_iq_pairs"])
        for i in range(n):
            row = [data[k][i] for k in scalar_fields]
            row += [mac_fmt(data["src_mac"][i]), mac_fmt(data["dst_mac"][i])]

            off, length = int(csi_offset[i]), int(csi_len[i])
            csi = csi_flat[off:off + length]
            pairs = ";".join(f"({a},{b})" for a, b in zip(csi[0::2], csi[1::2]))
            row.append(pairs)

            writer.writerow(row)
    print(f"wrote {n} rows to {out_path} (csi_iq_pairs holds every subcarrier as "
          f"semicolon-separated (imag,real) pairs, e.g. \"(9,10);(10,10);...\")")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="session directory or direct path to samples.npz")
    p.add_argument("--sample", type=int, help="print full detail (incl. decoded CSI) for one sample index")
    p.add_argument("--csv", action="store_true",
                    help="dump every field (incl. raw CSI) to <session_dir>/csi_export.csv")
    args = p.parse_args()

    input_path = Path(args.path)
    npz_path = resolve_npz_path(input_path)
    session_dir = npz_path.parent

    print_metadata(session_dir)
    with np.load(npz_path) as data:
        print_summary(data)
        if args.sample is not None:
            print_one_sample(data, args.sample)
        if args.csv:
            dump_csv(data, session_dir / "csi_export.csv")


if __name__ == "__main__":
    main()
