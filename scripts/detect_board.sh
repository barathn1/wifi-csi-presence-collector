#!/usr/bin/env bash
# Enumerates /dev/ttyACM*/ttyUSB*, probes each with esptool, and prints the
# port of the ESP32-S3 found. Never touches flash.
#
#   scripts/detect_board.sh                 # print the detected port
#   scripts/detect_board.sh --set-mac       # also cache the board's MAC into config.yaml
#   scripts/detect_board.sh --port /dev/ttyACM0   # skip scanning, probe this port only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SET_MAC=false
ONLY_PORT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --set-mac) SET_MAC=true; shift ;;
    --port) ONLY_PORT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -n "$ONLY_PORT" ]]; then
  CANDIDATE_PORTS=("$ONLY_PORT")
else
  CANDIDATE_PORTS=()
  for p in /dev/ttyACM* /dev/ttyUSB*; do
    [[ -e "$p" ]] && CANDIDATE_PORTS+=("$p")
  done
fi

if [[ ${#CANDIDATE_PORTS[@]} -eq 0 ]]; then
  echo "ERROR: no board found (checked /dev/ttyACM* and /dev/ttyUSB*). Check cable/power." >&2
  exit 1
fi

FOUND=()
for port in "${CANDIDATE_PORTS[@]}"; do
  echo "Probing $port ..." >&2
  if out=$(esptool --port "$port" chip-id 2>&1); then
    chip=$(echo "$out" | grep -oP 'Chip type:\s*\K[^(]*' | head -1 | sed 's/[[:space:]]*$//')
    mac=$(echo "$out" | grep -oP 'MAC:\s*\K[0-9A-Fa-f:]{17}' | head -1)
    echo "  -> $chip, MAC $mac" >&2
    FOUND+=("$port|$chip|$mac")
  else
    echo "  -> no response" >&2
  fi
done

if [[ ${#FOUND[@]} -eq 0 ]]; then
  echo "ERROR: found serial port(s) but none responded to esptool. Check the board is powered and not held by another process (e.g. idf.py monitor)." >&2
  exit 1
fi

if [[ ${#FOUND[@]} -gt 1 && -z "$ONLY_PORT" ]]; then
  REPO_ROOT="$(dirname "$SCRIPT_DIR")"
  HOTSPOT_PORT=$(python3 -c "
import sys; sys.path.insert(0, '$REPO_ROOT')
from collector.config import load_config
print(load_config().hotspot.serial_port)
" 2>/dev/null || echo "?")
  RX_PORT=$(python3 -c "
import sys; sys.path.insert(0, '$REPO_ROOT')
from collector.config import load_config
print(load_config().transport.serial_port)
" 2>/dev/null || echo "?")
  echo "Multiple boards found -- re-run with --port <path>, or use --port hotspot/transmitter/receiver-aware scripts (flash.sh, monitor.sh, push_config.sh):" >&2
  for entry in "${FOUND[@]}"; do
    IFS='|' read -r p chip mac <<< "$entry"
    role=""
    [[ "$p" == "$HOTSPOT_PORT" ]] && role=" (configured as hotspot)"
    [[ "$p" == "$RX_PORT" ]] && role=" (configured as receiver)"
    echo "  $p -- $chip, MAC $mac$role" >&2
  done
  exit 1
fi

IFS='|' read -r port chip mac <<< "${FOUND[0]}"
echo "Found $chip on $port, MAC $mac" >&2
echo "$port"

if $SET_MAC; then
  REPO_ROOT="$(dirname "$SCRIPT_DIR")"
  python3 -c "
import sys; sys.path.insert(0, '$REPO_ROOT')
from collector.config import DEFAULT_CONFIG_PATH
from collector.config_editor import set_key_inplace
set_key_inplace(DEFAULT_CONFIG_PATH, 'device.mac', '$mac')
print('cached MAC $mac into config.yaml (device.mac)', file=sys.stderr)
"
fi
