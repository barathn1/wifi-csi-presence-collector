#!/usr/bin/env bash
# Thin wrapper around collector.preflight -- fails loudly with a specific
# reason instead of letting a disconnected board silently produce an
# empty dataset. Run this before a collection session or capacity test.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

python3 -c "
from collector.config import load_config
from collector.preflight import check_board_connected

ok, reason = check_board_connected(load_config())
print(('OK: ' if ok else 'NOT CONNECTED: ') + reason)
raise SystemExit(0 if ok else 1)
"
