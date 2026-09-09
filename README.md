# WiFi CSI Presence Collector

A data collection pipeline for WiFi Channel State Information (CSI), built around ESP32-S3 boards,
for an authorized/unauthorized person presence-recognition task: capture CSI while a specific person
is present (`authorized`, or `unauthorized` for anyone else), or while the room is empty (`none`),
with real, verified packet-loss measurement over TCP.

**Scope**: this repo covers data collection only -- firmware, host-side collector, capacity testing,
and setup docs. Model training/testing on the collected data is explicit future work; see
`RESEARCH_NOTES.md` for the literature review informing that future step.

## Everything configurable lives in one file

`config/config.yaml` is the single source of truth for network settings, board roles, CSI capture
parameters, and dataset output location. Every script and the firmware (via NVS push) reads from it.

## Setup

Pick the guide matching how you want to host the WiFi network (switchable via `network.ap_source`):

- **[SETUP_ANDROID_HOTSPOT.md](SETUP_ANDROID_HOTSPOT.md)** -- an Android phone hosts the network.
- **[SETUP_ESP_HOTSPOT.md](SETUP_ESP_HOTSPOT.md)** -- a second ESP32-S3 hosts the network itself.
- **[SETUP_GOOGLE_AP.md](SETUP_GOOGLE_AP.md)** -- an existing home/office AP (Google WiFi or similar)
  hosts the network.

Each is self-contained: exact config values to change, first-time setup, and routine day-to-day use.

## Testing and collection

Once setup is done, **[TESTING_RUNBOOK.md](TESTING_RUNBOOK.md)** has the exact command sequence:
capacity testing (real data-loss measurement), collecting a labeled session, and inspecting what you
collected.

## Repo layout

```
config/config.yaml   single source of truth for all configurable values
firmware/             ESP-IDF project (target esp32s3) -- ONE binary, role picked via NVS
scripts/              board detection, build/flash/monitor, NVS config push, IP refresh, preflight
collector/            Python: config loading, wire-format parsing, transports, capacity test,
                       CLI collector, inspect_npz.py viewer
data/                 collected sessions (gitignored)
reports/capacity/     capacity test reports (gitignored)
```

## Hardware

ESP32-S3 boards (native USB-CDC/JTAG serial), using `esp_wifi_set_csi_rx_cb`/promiscuous-mode capture
for CSI density. See `RESEARCH_NOTES.md` for why single-antenna ESP32-class hardware is viable for
this specific task (verifying a small set of known people) but not for simultaneous multi-person or
gait-based identification.
