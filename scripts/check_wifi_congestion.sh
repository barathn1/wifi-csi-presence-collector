#!/usr/bin/env bash
# Surveys nearby 2.4GHz WiFi APs from this machine's WiFi card and scores channels 1/6/11 by
# congestion, to help pick the least-congested channel to PIN the router to for CSI collection
# (see [[project-day2-cross-channel-root-cause]] in memory / ml/reports/day2_next_steps.md --
# the actual fix for cross-day drift is a fixed channel+bandwidth, not any specific channel number).
#
# Scores each candidate by summing every detected AP's signal, weighted by how much its channel's
# 22MHz-wide occupancy overlaps the candidate (full weight at 0 channel separation, zero at >=5,
# since 2.4GHz channels are 5MHz apart and a 20MHz channel occupies ~4-5 channel-steps of width).
# This catches interference from e.g. a strong neighbor on channel 3 bleeding into both 1 and 6,
# which a naive "count APs on exactly channel N" tally would miss.
#
#   scripts/check_wifi_congestion.sh              # uses the first WiFi interface nmcli finds
#   scripts/check_wifi_congestion.sh wlp0s20f3    # or name one explicitly
set -euo pipefail

IFACE="${1:-}"
if [[ -z "$IFACE" ]]; then
    IFACE="$(nmcli -t -f DEVICE,TYPE dev status | awk -F: '$2=="wifi"{print $1; exit}')"
fi
if [[ -z "$IFACE" ]]; then
    echo "No WiFi interface found. Pass one explicitly, e.g.: $0 wlan0" >&2
    exit 1
fi

echo "Scanning on $IFACE ..."
SCAN="$(nmcli -t -f CHAN,SIGNAL dev wifi list ifname "$IFACE" --rescan yes)"

echo
echo "Raw AP count per 2.4GHz channel (channel:count):"
echo "$SCAN" | awk -F: '$1>=1 && $1<=13 {print $1}' | sort -n | uniq -c | awk '{printf "  ch%-3s %s\n", $2, $1}'

echo
echo "Overlap-weighted congestion score for candidate channels 1/6/11 (lower is better):"
echo "$SCAN" | awk -F: '
    $1 >= 1 && $1 <= 13 {
        ch = $1; sig = $2
        for (i = 1; i <= n; i++) {
            cand = cands[i]
            diff = ch - cand; if (diff < 0) diff = -diff
            weight = (5 - diff) / 5
            if (weight > 0) score[cand] += weight * sig
        }
    }
    BEGIN { split("1 6 11", c, " "); n = 3; for (i=1;i<=n;i++) cands[i]=c[i] }
    END {
        best = ""; bestscore = -1
        for (i = 1; i <= n; i++) {
            cand = cands[i]
            printf "  ch%-3s score=%.0f\n", cand, score[cand]
            if (bestscore < 0 || score[cand] < bestscore) { bestscore = score[cand]; best = cand }
        }
        print ""
        print "Least congested candidate: ch" best
    }
'
