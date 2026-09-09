#!/usr/bin/env bash
# Regenerates just the "cfgstore" NVS partition from config.yaml for ONE
# board role, and flashes only that (not the app partition) -- ~1-2s, so
# changing an IP/SSID/rate never requires a full rebuild+reflash.
#
# For role=receiver, also re-caches device.mac in config.yaml from the
# board it actually finds on the configured port -- no separate "cache
# the MAC" step needed, ever.
#
#   scripts/push_config.sh hotspot       # pushes to config.yaml's hotspot.serial_port
#   scripts/push_config.sh receiver      # pushes to config.yaml's transport.serial_port
#   scripts/push_config.sh transmitter   # optional 3rd-board role; needs a transmitter: config section
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

ROLE="${1:-}"
if [[ "$ROLE" != "hotspot" && "$ROLE" != "transmitter" && "$ROLE" != "receiver" ]]; then
  echo "usage: $0 <hotspot|transmitter|receiver>" >&2
  exit 2
fi

CFGSTORE_SIZE=0x6000  # must match firmware/partitions.csv

CONFIGURED_PORT=$(python3 -c "
import sys; sys.path.insert(0, '$REPO_ROOT')
from collector.config import load_config
cfg = load_config()
ports = {'hotspot': cfg.hotspot.serial_port, 'transmitter': getattr(cfg, 'transmitter', None) and cfg.transmitter.serial_port, 'receiver': cfg.transport.serial_port}
print(ports['$ROLE'])
")

# For the receiver, --set-mac re-caches device.mac (used by the ARP-based
# preflight check) on every push -- free, since detect_board.sh already
# resets/probes the board here anyway. Self-heals if the board ever gets
# swapped; nothing to remember to run separately.
DETECT_ARGS=(--port "$CONFIGURED_PORT")
[[ "$ROLE" == "receiver" ]] && DETECT_ARGS+=(--set-mac)
PORT=$("$SCRIPT_DIR/detect_board.sh" "${DETECT_ARGS[@]}")

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

python3 -c "
import sys; sys.path.insert(0, '$REPO_ROOT')
from collector.config import load_config
from collector.nvs_gen import write_csv
write_csv(load_config(), '$ROLE', '$TMP_DIR/cfgstore.csv')
"

python3 "$IDF_PATH/components/nvs_flash/nvs_partition_generator/nvs_partition_gen.py" generate \
  "$TMP_DIR/cfgstore.csv" "$TMP_DIR/cfgstore.bin" "$CFGSTORE_SIZE"

python3 "$IDF_PATH/components/partition_table/parttool.py" --port "$PORT" \
  write_partition --partition-name cfgstore --input "$TMP_DIR/cfgstore.bin"

echo "pushed config.yaml (role=$ROLE) -> cfgstore NVS partition on $PORT"
