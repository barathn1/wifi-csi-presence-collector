"""Targeted single-line edits to config.yaml.

Rewrites exactly one "key: value" line in place, leaving every other line
in the file untouched -- deliberately not a full YAML load/dump round trip,
which would disturb comments and formatting in the hand-edited file.

Used by scripts/get_laptop_ip.sh --set and capacity_test.py --apply.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _format_yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def set_key_inplace(path: str | Path, dotted_key: str, value: Any) -> None:
    """dotted_key is "section.key", e.g. "network.laptop_ip" -- exactly one
    level of nesting, matching config.yaml's structure."""
    section, key = dotted_key.split(".", 1)
    if "." in key:
        raise ValueError(f"only one level of nesting is supported: {dotted_key}")

    path = Path(path)
    lines = path.read_text().splitlines(keepends=True)

    section_re = re.compile(rf"^{re.escape(section)}:\s*(#.*)?$")
    key_re = re.compile(rf"^(\s+){re.escape(key)}:(\s.*)?$")

    in_section = False
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith((" ", "\t")):
            in_section = bool(section_re.match(line))
            continue
        if in_section:
            m = key_re.match(line)
            if m:
                indent = m.group(1)
                lines[i] = f"{indent}{key}: {_format_yaml_scalar(value)}\n"
                path.write_text("".join(lines))
                return

    raise KeyError(f'key "{dotted_key}" not found in {path}')
