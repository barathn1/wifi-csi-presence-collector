#!/usr/bin/env bash
# Live serial console for one board.
#   scripts/monitor.sh hotspot
#   scripts/monitor.sh receiver
#   scripts/monitor.sh transmitter
#   scripts/monitor.sh /dev/ttyACM2
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

ARG="${1:-}"
if [[ "$ARG" == /dev/* ]]; then
  PORT="$ARG"
elif [[ "$ARG" == "hotspot" || "$ARG" == "transmitter" || "$ARG" == "receiver" ]]; then
  PORT=$(python3 -c "
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

idf.py -C "$REPO_ROOT/firmware" -p "$PORT" monitor
