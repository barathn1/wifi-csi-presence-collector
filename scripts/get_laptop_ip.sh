#!/usr/bin/env bash
# Prints (and optionally writes back into config.yaml) the laptop's
# current IPv4 address on the hotspot-facing interface. Hotspot IPs are
# DHCP-assigned and churn between sessions -- this is the one command to
# refresh it.
#
#   scripts/get_laptop_ip.sh          # just print it
#   scripts/get_laptop_ip.sh --set    # print it and write into config.yaml
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

IFACE=$(python3 -c "
import sys; sys.path.insert(0, '$REPO_ROOT')
from collector.config import load_config
print(load_config().network.laptop_iface)
")

IP=$(ip -4 -o addr show "$IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)

if [[ -z "$IP" ]]; then
  echo "ERROR: interface $IFACE has no IPv4 address -- is the laptop connected to the hotspot?" >&2
  exit 1
fi

echo "$IP"

if [[ "${1:-}" == "--set" ]]; then
  python3 -c "
import sys; sys.path.insert(0, '$REPO_ROOT')
from collector.config import DEFAULT_CONFIG_PATH
from collector.config_editor import set_key_inplace
set_key_inplace(DEFAULT_CONFIG_PATH, 'network.laptop_ip', '$IP')
print('updated config.yaml: network.laptop_ip = $IP', file=sys.stderr)
"
fi
