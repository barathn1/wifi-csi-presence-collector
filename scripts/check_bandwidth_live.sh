#!/usr/bin/env bash
# Checks the CURRENT LIVE channel/bandwidth the board is actually seeing right now -- straight from
# the real-time stream, not from a previously-recorded session (see scripts/check_bandwidth.sh for
# that one). Connects, samples a handful of live packets, reports, exits.
#
#   scripts/check_bandwidth_live.sh                    # just report what it sees
#   scripts/check_bandwidth_live.sh --expect-mhz 20    # also exit non-zero on a mismatch (preflight gate)
#
# Mutual exclusion: stop collector.cli_collect / ml.visualization.player_server / ml.inference.live_infer
# first -- only one process can hold the transport at a time.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

python3 -m ml.inference.check_live_bandwidth "$@"
