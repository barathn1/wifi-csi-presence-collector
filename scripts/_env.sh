#!/usr/bin/env bash
# Shared by every script in this directory. Sources the ESP-IDF environment
# (idf.py/esptool are not on PATH by default) using the path from
# config.yaml -- the one place to change if the IDF install ever moves.
set -euo pipefail

_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$_ENV_SCRIPT_DIR")"

if ! command -v idf.py >/dev/null 2>&1; then
  IDF_PATH_FROM_CONFIG=$(python3 -c "
import sys; sys.path.insert(0, '$REPO_ROOT')
from collector.config import load_config
print(load_config().firmware.idf_path)
")
  # shellcheck disable=SC1090,SC1091
  source "$IDF_PATH_FROM_CONFIG/export.sh" >/dev/null
fi
