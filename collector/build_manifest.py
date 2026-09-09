"""Rebuilds data/manifest.csv from every session's metadata.json.

Derived index -- safe to regenerate any time, never hand-edited.

    python -m collector.build_manifest
"""
from __future__ import annotations

import csv
import json

from collector.config import Config, load_config

FIELDS = [
    "session_dir", "label", "person_id", "motion", "start_ts", "duration_s",
    "transport_used", "board_mac", "ap_source", "sample_count",
]


def build_manifest(cfg: Config) -> int:
    dataset_dir = cfg.dataset_dir()
    rows = []
    for meta_path in sorted(dataset_dir.glob("*/*/*/metadata.json")):
        meta = json.loads(meta_path.read_text())
        # .get(): sessions collected before a field existed (e.g. ap_source)
        # shouldn't break manifest rebuilds -- they just show blank there.
        rows.append({field: meta.get(field, "") if field != "session_dir" else
                     str(meta_path.parent.relative_to(dataset_dir)) for field in FIELDS})

    manifest_path = dataset_dir / "manifest.csv"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} sessions to {manifest_path}")
    return len(rows)


def main() -> None:
    build_manifest(load_config())


if __name__ == "__main__":
    main()
