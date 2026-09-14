# `check_bandwidth.sh` — verify collection bandwidth before trusting new data

## Why this exists

The router auto-switched WiFi channel and bandwidth between Day 1 and Day 2 of this project's
collection, with no warning: Day 1 was captured on a 40MHz-bonded channel (channel 1, ~2402-2442MHz,
186 usable CSI subcarriers), Day 2 on a plain 20MHz channel (channel 11, ~2452-2472MHz, 128 usable
subcarriers). Those two frequency ranges don't overlap at all — CSI amplitude/phase encodes
frequency-selective multipath fading, so a model trained on one day's frequencies doesn't reliably
transfer to the other's. This broke cross-day model comparisons badly (see
`ml/reports/day2_next_steps.md` and the `project-day2-cross-channel-root-cause` memory note for the
full root-cause writeup) and cost real time to diagnose after the fact.

This script exists so that problem never happens silently again. It reads the channel/bandwidth
directly from fields the firmware already records per-packet in `samples.npz`
(`sig_mode`/`mcs`/`cwb`/`channel_primary`/`channel_secondary`) — no router admin access needed, and it
works on data already on disk, not just live capture.

**Literature-backed recommendation** (see the bandwidth-choice research done this project): default to
**20MHz on a fixed channel**, not 40MHz. 40MHz cannot actually be permanently pinned in the 2.4GHz band
— the 802.11 standard requires APs to fall back to 20MHz whenever a neighboring network is in range, so
"pin 40MHz" is not an achievable state long-term. Packet rate and observation-window length matter far
more than bandwidth for both presence detection and gait, per the papers checked. The script's default
`--expect-mhz` is therefore `20`.

## Usage

```bash
# Check the most recently collected session (run this right after every collection):
scripts/check_bandwidth.sh

# Check one specific session:
scripts/check_bandwidth.sh data/authorized/2026-09-10/20260910_153408_anjali

# Audit every session already on disk (useful after collecting a whole new day):
scripts/check_bandwidth.sh --all

# Override the expected bandwidth (e.g. if you deliberately want to check for 40MHz):
scripts/check_bandwidth.sh --expect-mhz 40
```

Output looks like:

```
[OK      ] data/unauthorized/2026-09-10/20260910_170103_siva: 20MHz (HT20), channel 11, 2452-2472 MHz (100.0% of packets)
[MISMATCH] data/authorized/2026-09-09/20260909_152250_anjali: 40MHz (HT40), channel 1+ABOVE, 2402-2442 MHz (88.9% of packets)
```

The percentage is the fraction of packets in that session that actually matched the reported dominant
channel/bandwidth combination — a session mixing formats (rare, but possible if the AP itself
re-negotiated mid-session) will show a lower percentage here, worth a second look even if the dominant
mode matches what you expect.

## Exit codes

- `0` — every checked session matched `--expect-mhz`.
- `1` — at least one checked session did not match (a real mismatch was found).
- `2` — no sessions found under `data/` at all (nothing to check).

Because it's a normal shell exit code, it's safe to use as a preflight gate in a collection workflow,
e.g.:

```bash
scripts/check_bandwidth.sh || { echo "bandwidth mismatch — recollect before trusting this session"; exit 1; }
```

**Recommended habit**: run `scripts/check_bandwidth.sh` (no arguments) immediately after every
collection session, before moving on to the next one. Catching a bandwidth drift on session 1 of a new
day costs nothing; catching it after collecting 30 sessions costs the entire day.

## How it works

Thin wrapper around `ml.data_pipeline.decode_csi.channel_width_summary()`, which:
1. Finds the (`cwb`, `channel_primary`, `channel_secondary`) combination that the most packets in the
   session share.
2. Translates it into a human-readable description using the 2.4GHz channel-to-frequency formula
   (`2412 + 5*(channel-1)` MHz) and the HT20/HT40 bandwidth (`cwb`: 0=20MHz, 1=40MHz) plus secondary
   channel direction (`channel_secondary`, an ESP-IDF enum: 0=none, 1=above, 2=below the primary).

See `ml/data_pipeline/decode_csi.py::channel_width_summary` for the implementation, or run
`python3 -m ml.data_pipeline.decode_csi <session_dir> --channel` directly for the same check without
going through this wrapper.
