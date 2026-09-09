#!/usr/bin/env bash
# Flashes the firmware -- the SAME binary for every role; only the pushed
# NVS config (scripts/push_config.sh) decides hotspot/transmitter/receiver
# behavior at boot.
#
#   scripts/flash.sh hotspot       # flashes config.yaml's hotspot.serial_port
#   scripts/flash.sh receiver      # flashes config.yaml's transport.serial_port
#   scripts/flash.sh transmitter   # optional 3rd-board role
#   scripts/flash.sh /dev/ttyACM2  # or flash an explicit port directly
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

ARG="${1:-}"
if [[ "$ARG" == /dev/* ]]; then
  CONFIGURED_PORT="$ARG"
elif [[ "$ARG" == "hotspot" || "$ARG" == "transmitter" || "$ARG" == "receiver" ]]; then
  CONFIGURED_PORT=$(python3 -c "
import sys; sys.path.insert(0, '$REPO_ROOT')
from collector.config import load_config
cfg = load_config()
ports = {'hotspot': cfg.hotspot.serial_port, 'transmitter': getattr(cfg, 'transmitter', None) and cfg.transmitter.serial_port, 'receiver': cfg.transport.serial_port}
print(ports['$ARG'])
")
else
  echo "usage: $0 <hotspot|transmitter|receiver|/dev/ttyACMx>" >&2
  exit 2
fi

DETECTED_PORT=$("$SCRIPT_DIR/detect_board.sh" --port "$CONFIGURED_PORT")

idf.py -C "$REPO_ROOT/firmware" -p "$DETECTED_PORT" flash
