#!/usr/bin/env bash
# Run this for routine day-to-day use, once the receiver board (and, if
# network.ap_source is "esp", the hotspot board too) has been flashed and
# configured at least once (see SETUP_*.md "First time only").
#
# Reads network.ap_source from config.yaml to decide whether there's an
# ESP hotspot board to push config to at all -- if ap_source is anything
# other than "esp" (e.g. "android", "google-ap"), that step is skipped and
# you're expected to have that external AP already on/broadcasting yourself.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

AP_SOURCE=$(python3 -c "
import sys; sys.path.insert(0, '$REPO_ROOT')
from collector.config import load_config
print(load_config().network.ap_source)
")

if [[ "$AP_SOURCE" == "esp" ]]; then
  echo "== ap_source=esp: pushing config to hotspot board (resets it briefly) =="
  "$SCRIPT_DIR/push_config.sh" hotspot

  echo "== waiting for the hotspot to come back up =="
  sleep 3
else
  echo "== ap_source=$AP_SOURCE: skipping ESP hotspot push -- make sure that AP is already on =="
fi

echo "== refreshing laptop IP (make sure your laptop's WiFi is connected to the hotspot first) =="
"$SCRIPT_DIR/get_laptop_ip.sh" --set

echo "== pushing config to receiver =="
"$SCRIPT_DIR/push_config.sh" receiver

echo "== confirming receiver is connected =="
"$SCRIPT_DIR/connect_check.sh"
