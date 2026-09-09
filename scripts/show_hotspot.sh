#!/usr/bin/env bash
# Prints the SSID/password the hotspot board (config.yaml's hotspot.*)
# will broadcast, so you can manually connect your laptop's WiFi to the
# same network the receiver board joins -- without having to open
# config.yaml yourself.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

python3 -c "
import sys; sys.path.insert(0, '$REPO_ROOT')
from collector.config import load_config
cfg = load_config()
print(f'SSID:     {cfg.network.hotspot_ssid}')
print(f'Password: {cfg.network.hotspot_password}')
"
