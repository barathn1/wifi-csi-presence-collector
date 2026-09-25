# CSI Presence Dataset

WiFi CSI (Channel State Information) recordings used for presence detection
and person identification. Small ESP32 receiver boards measure how WiFi
packets get distorted in the room; a person's motion changes those
distortions, and that's the signal captured here.

## Folder layout

```
data/
  authorized/     <- known/allowed person present
  unauthorized/   <- person present who should not have access
  none/           <- empty room, no person
  manifest.csv    <- one row per recording, across every session/board
```

Inside each label folder: date, then one folder per recording session.

```
authorized/2026-09-24/20260924_165513_barath/
```

- `2026-09-24` — recording date
- `20260924_165513_barath` — start time + person's name (omitted for `none`
  sessions)

## Session contents

Each session was recorded by one or more receiver boards at once. One file
pair per board:

```
ac276ea55bc8_metadata.json
ac276ea55bc8_samples.npz
```

- Prefix `ac276ea55bc8` — the board's MAC address, colons stripped. No prefix
  (plain `metadata.json` / `samples.npz`) means the session only had one
  board.
- `metadata.json` — session details (person, motion, duration, sample
  count, etc.), plain JSON.
- `samples.npz` — the CSI measurements themselves, NumPy-loadable.

## `metadata.json` fields

| Field | Meaning |
|---|---|
| `label` | `authorized` / `unauthorized` / `none` |
| `person_id` | person present during the recording |
| `motion` | `standing` or `walking` |
| `start_ts` / `end_ts` | recording start/end timestamp |
| `duration_s` | recording length, seconds |
| `sample_count` | number of CSI measurements captured |
| `avg_rate_hz` | average capture rate |
| `dropped_samples` | measurements lost/corrupted and excluded |
| `board_mac` | receiver board that captured this file |
| `ap_source` | WiFi network/router the signal was measured against |

Per-packet radio parameters (WiFi channel, bandwidth, RSSI, etc.) are stored
inside `samples.npz` itself (`channel_primary`, `cwb`, `rssi`, ...), not in
`metadata.json`.

## Last 3 collection days (2026-09-21, 2026-09-22, 2026-09-24)

These are the most relevant sessions to work from: all three use 3 ESP32
receiver boards capturing simultaneously (`a4cb8fd452b0`, `ac276ea2f278`,
`ac276ea55bc8`), all on **WiFi channel 6, 20 MHz bandwidth**
(`channel_primary=6`, `cwb=0`). Earlier dates (2026-09-09 through 09-17) used
a single board and different channel/bandwidth settings, so they don't mix
cleanly with this later data.

## `manifest.csv`

One row per recording (every board, every session): session folder, label,
person, motion, timing, sample count. Fastest way to scan the whole dataset
without opening individual files.
