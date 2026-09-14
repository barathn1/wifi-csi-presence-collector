#!/usr/bin/env bash
# Verifies the actual WiFi channel/bandwidth (20MHz vs 40MHz) a collected session used, straight from
# the per-packet rx_ctrl fields already recorded in samples.npz -- no router admin access needed. Exists
# because the router silently auto-switched bandwidth between Day1 (40MHz) and Day2 (20MHz) once
# already, which broke cross-day model comparisons badly (see scripts/check_bandwidth.md and
# ml/reports/day2_next_steps.md). Run this right after every collection session, not just when
# something already looks wrong.
#
#   scripts/check_bandwidth.sh                    # check the most recently collected session
#   scripts/check_bandwidth.sh <session_dir>       # check one specific session
#   scripts/check_bandwidth.sh --all               # audit every session under data/, flag mismatches
#   scripts/check_bandwidth.sh --expect-mhz 40 ... # override the expected default (20MHz)
#
# Exit code is non-zero if the checked session(s) don't match --expect-mhz -- safe to use as a
# preflight gate, e.g.:  scripts/check_bandwidth.sh || echo "reject this session, recollect it"
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

EXPECT_MHZ=20
MODE="latest"
TARGET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) MODE="all"; shift ;;
    --expect-mhz) EXPECT_MHZ="$2"; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) TARGET="$1"; MODE="one"; shift ;;
  esac
done

python3 -c "
import sys
from pathlib import Path
from ml.data_pipeline.decode_csi import load_session, list_sessions, channel_width_summary

expect_mhz = $EXPECT_MHZ
mode = '$MODE'
target = '$TARGET'


def check_one(session_dir) -> bool:
    session = load_session(Path(session_dir))
    info = channel_width_summary(session.npz)
    actual_mhz = 40 if info['cwb'] == 1 else 20
    ok = actual_mhz == expect_mhz
    status = 'OK      ' if ok else 'MISMATCH'
    print(f'[{status}] {session_dir}: {info[\"description\"]} '
          f'({100 * info[\"fraction_of_packets\"]:.1f}% of packets)')
    return ok


if mode == 'all':
    sessions = list_sessions()
    if not sessions:
        print('no sessions found under data/', file=sys.stderr)
        sys.exit(2)
    results = [check_one(s) for s in sessions]
    n_bad = results.count(False)
    print()
    print(f'{len(sessions)} sessions checked, expected {expect_mhz}MHz -- {n_bad} mismatch(es).')
    sys.exit(0 if n_bad == 0 else 1)
elif mode == 'one':
    sys.exit(0 if check_one(target) else 1)
else:
    sessions = list_sessions()
    if not sessions:
        print('no sessions found under data/', file=sys.stderr)
        sys.exit(2)
    latest = max(sessions, key=lambda p: (p / 'samples.npz').stat().st_mtime)
    sys.exit(0 if check_one(latest) else 1)
"
