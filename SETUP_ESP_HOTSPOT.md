# Setup: a second ESP32 as hotspot

One of two switchable setups -- see `SETUP_ANDROID_HOTSPOT.md` for the other
(an Android phone hosts the network instead of an ESP32). Switching between
them is just changing `network.ap_source` in `config/config.yaml` and
following the matching guide; nothing else in this repo needs to change.

**For testing commands after setup is done, see `TESTING_RUNBOOK.md`.**

## Setup

### Every time (routine use, after first-time setup below is done once)

1. Connect your laptop's WiFi to the ESP-hosted network
   (`scripts/show_hotspot.sh` for the credentials).
2. Run:
   ```
   scripts/reconnect.sh
   ```
   That's it. It sees `ap_source: "esp"`, re-pushes the hotspot board's
   config (harmless if unchanged), waits for it to come back up, refreshes
   your laptop's IP, re-detects and pushes the receiver's config (which also
   re-caches its MAC), and confirms it joined -- **you never need to look up
   or type in a MAC address or IP yourself**, every run, even across board
   swaps.

**Always run `reconnect.sh` right before collecting, every session -- don't
skip it just because it's the same laptop and the same two boards as last
time.** Both ESPs' MACs are genuinely constant (burned into the chip), but
the laptop's IP is not: it's assigned by the hotspot board's DHCP server,
and DHCP doesn't guarantee handing back the same address on every reconnect.
That's exactly the failure mode `reconnect.sh` closes off by re-checking the
IP every time instead of trusting a cached value.

### First time only (or after changing firmware code)

1. Set `network.ap_source: "esp"` in `config/config.yaml`, and
   `network.hotspot_ssid` / `hotspot_password` to whatever you want the ESP
   to broadcast.
2. Plug in **both** ESP32-S3 boards over USB (`/dev/ttyACM0` = hotspot,
   `/dev/ttyACM1` = receiver -- swap the `serial_port` values in config.yaml
   if yours enumerate the other way around).
3. Build once, flash both with the same binary:
   ```
   scripts/build.sh
   scripts/flash.sh receiver
   scripts/flash.sh hotspot
   ```
4. Push the hotspot's config **first** -- this is what makes it start
   broadcasting at all, and doesn't need any network up yet:
   ```
   scripts/push_config.sh hotspot
   ```
5. Connect your laptop's WiFi to the ESP-hosted network:
   ```
   scripts/show_hotspot.sh
   ```
   prints the SSID/password to connect to manually.
6. Push the receiver's config and confirm:
   ```
   scripts/push_config.sh receiver
   scripts/connect_check.sh
   ```

## Collect a labeled session

Once `reconnect.sh` (or `connect_check.sh`) reports OK, here's every
`cli_collect.py` parameter spelled out:
```
python3 -m collector.cli_collect \
  --label authorized \
  --person-id alice \
  --motion standing \
  --notes "living room, facing the ESP hotspot" \
  --duration 120 \
  --config config/config.yaml
```

| Flag | Meaning |
|---|---|
| `--label` | `authorized \| unauthorized \| none` -- **required**. |
| `--person-id` | Who this session is of -- **required unless `--label none`**. |
| `--motion` | `standing \| walking` -- **required unless `--label none`**. |
| `--notes` | Free-text, optional -- saved into `metadata.json` (default: empty). |
| `--duration` | Seconds to record. Omit it to run until you press Ctrl+C instead. |
| `--no-stimulus` | Flag (no value) that disables the laptop's UDP traffic generator. Don't pass this in this setup -- there's no dedicated transmitter board here, so the laptop's stimulus is the only thing giving the receiver CSI-triggering traffic. |
| `--config` | Path to an alternate config.yaml. Omit to use the default `config/config.yaml`. |

Repeat per person/label/motion as needed:
```
python3 -m collector.cli_collect --label authorized --person-id alice --motion standing --duration 120
python3 -m collector.cli_collect --label authorized --person-id alice --motion walking --duration 120
python3 -m collector.cli_collect --label unauthorized --person-id bob --motion standing --duration 120
python3 -m collector.cli_collect --label unauthorized --person-id bob --motion walking --duration 120
python3 -m collector.cli_collect --label none --duration 60        # empty room -- no --person-id/--motion needed
```
Runs a preflight check first (fails loudly if the receiver isn't actually on
the ESP-hosted network), then streams real CSI, printing a live status line
every 5s with time remaining and packet loss so far, and writes to
`data/<label>/<date>/<session_id>/{metadata.json, samples.npz}`. See
`TESTING_RUNBOOK.md` for the full sequence (capacity test first, then this,
then inspecting what you collected).

## Topology

- **Two ESP32-S3 boards**, both wired to the laptop over USB. No phone needed:
  - **`/dev/ttyACM0` -- HOTSPOT.** Runs in WiFi AP mode, hosting
    `network.hotspot_ssid`/`hotspot_password` on `hotspot.channel`. Does no
    CSI work at all -- it only hosts the network the other devices join.
  - **`/dev/ttyACM1` -- RECEIVER.** Joins that ESP-hosted network as a
    station, captures CSI in promiscuous mode, and streams it to the laptop
    over TCP.
- The laptop also joins the ESP-hosted network, and is:
  - the **TCP server** the receiver connects to (so the main data path never
    needs to know the receiver's DHCP-assigned IP -- only the laptop's own IP,
    refreshed via `scripts/get_laptop_ip.sh --set`), and
  - the **traffic generator** (`collector/stimulus.py`, wired into
    `cli_collect.py` by default) -- there's no dedicated transmitter board in
    this setup, so the laptop's UDP stimulus is what gives the receiver
    something to extract CSI from.
- **Both boards run the exact same compiled firmware binary** -- role
  (`hotspot`/`receiver`) is a config value pushed to each board's NVS
  separately, not a build-time choice. `scripts/build.sh`/`flash.sh` only
  build once; only `push_config.sh <role>` differs per board.

**Note on your laptop's internet**: joining the ESP-hosted network means your
laptop leaves whatever network it was on. The ESP hotspot has no internet
uplink of its own, so you lose internet access for the duration of testing --
inherent to the topology, not something the scripts can work around.

## Everything configurable lives in one file

**`config/config.yaml`** -- AP source switch, hotspot SSID/password/channel,
both boards' serial ports, the laptop's IP, CSI capture parameters, dataset
output dir. Never edit IPs/SSIDs/rates/ports anywhere else.

```yaml
network:
  ap_source: "esp"             # switch to "android" to use SETUP_ANDROID_HOTSPOT.md instead
  hotspot_ssid: "..."           # SSID the ESP-hosted AP will broadcast
  hotspot_password: "..."       # >= 8 chars (WPA2-PSK minimum)
  laptop_ip: "..."              # refreshed by scripts/get_laptop_ip.sh --set

hotspot:
  serial_port: "/dev/ttyACM0"   # the board hosting the WiFi network
  channel: 1

transport:
  serial_port: "/dev/ttyACM1"   # the RECEIVER board
  mode: "tcp"
```

### Config values you need to change for this setup

| Field | What to set it to |
|---|---|
| `network.ap_source` | `"esp"` -- must be this for the steps below to apply. |
| `network.hotspot_ssid` | Whatever SSID you want the ESP hotspot board to broadcast -- pick anything. |
| `network.hotspot_password` | Whatever password you want it to require. >= 8 characters (WPA2-PSK minimum). |
| `network.laptop_iface` | Your laptop's WiFi interface name (e.g. `wlp0s20f3`). Only needs changing if your machine's interface name differs -- check with `ip link`. |
| `network.laptop_ip` | Leave as-is; `scripts/get_laptop_ip.sh --set` (called by `reconnect.sh`) fills this in automatically once you're on the ESP-hosted network. |
| `hotspot.serial_port` | The hotspot board's port. Default `/dev/ttyACM0` -- change if `scripts/detect_board.sh` shows it enumerating elsewhere on your machine. |
| `hotspot.channel` | WiFi channel the ESP-hosted AP broadcasts on. Default `1` works fine; only change if you have interference on that channel. |
| `transport.serial_port` | The receiver board's port. Default `/dev/ttyACM1` -- change if it enumerates elsewhere. |
| `device.mac` | Leave blank/as-is; `scripts/push_config.sh receiver` (called by `reconnect.sh`) re-detects and caches it automatically every run -- never edit this by hand. |

Everything else in `config/config.yaml` (CSI capture parameters, capacity-test
rates, dataset output dir) has working defaults and doesn't need to change for
this setup specifically.

## Quick reference

| Script | Role-aware? | When to run it |
|---|---|---|
| `scripts/show_hotspot.sh` | -- | Print the SSID/password the hotspot board will broadcast, so you can manually connect your laptop's WiFi to it. |
| `scripts/detect_board.sh [--port P]` | lists both, tags configured roles | Find which `/dev/ttyACM*` is which. With 2 boards connected and no `--port`, lists both with role hints instead of erroring. (`--set-mac` is only for manual debugging -- `push_config.sh receiver` already does this automatically.) |
| `scripts/build.sh` | no (one binary, every role) | Only after editing **firmware C code**. |
| `scripts/flash.sh <hotspot\|receiver\|/dev/ttyACMx>` | **yes** | Flash one board. Same binary either way -- only `push_config.sh` differentiates behavior. |
| `scripts/get_laptop_ip.sh --set` | -- | Laptop's IP on the network changed. No board interaction. |
| `scripts/push_config.sh <hotspot\|receiver>` | **yes, required** | Push config.yaml to one board's NVS (~1-2s, no reflash). Run once per board after any config change. For `hotspot`, must run before the network even exists for the laptop to join. |
| `scripts/reconnect.sh` | pushes both | **The one command for routine use.** Sees `ap_source: "esp"`, re-pushes the hotspot config too, refreshes laptop IP, pushes receiver config, confirms it joined. |
| `scripts/connect_check.sh` | receiver only | Quick "is the receiver actually on the network right now?" check. |
| `scripts/monitor.sh <hotspot\|receiver\|/dev/ttyACMx>` | **yes** | Live serial console for one board, for debugging. |


## Data loss -- shown live, not just at the end

`cli_collect.py` prints a status line every 5 seconds with time remaining and a
running packet-loss count, e.g.:
```
1234 samples (65.2 Hz) | 45s elapsed, 75s remaining | 0 dropped (0.0% loss)
```
Loss is detected from real gaps in the `seq` field the firmware assigns to every
CSI frame at capture time -- a gap means samples were actually lost (queue
overflow, tx failure, or transport loss), not an estimate. The final count is
also saved in `metadata.json` as `dropped_samples`, and the average rate as
`avg_rate_hz`.

**Caveat found while testing this setup without reconnecting the laptop**:
forcing the receiver to `serial` mode (to test over USB alone, with no network
involved) reproduced a pre-existing, already-known issue -- the
USB-Serial/JTAG console link corrupts under sustained load, which can make the
`seq`-gap loss count wildly overstated (a corrupted `seq` value can look like a
huge gap). This is exactly why **TCP is the default and recommended
transport**; it showed ~0% real loss across thousands of samples in every
prior test. `transport.mode` should stay `"tcp"`.

## What a collected session looks like

```
data/<label>/<date>/<session_id>/
  metadata.json   # label, person_id, notes, timing, transport, board MAC, ap_source, motion, dropped_samples, avg_rate_hz, config snapshot
  samples.npz      # CSI samples: fixed-width scalar fields + ragged CSI payloads
                    # (csi_flat + csi_offset + csi_len, since CSI length varies per packet)
```
Every session collected under this setup will have `ap_source: "esp"` in its
metadata and in the `data/manifest.csv` row, so later analysis can filter or
compare against `"android"`-sourced sessions from `SETUP_ANDROID_HOTSPOT.md`.

`data/manifest.csv` is rebuilt automatically after every `cli_collect` run --
never hand-edit it; to force a rebuild, run `python3 -m collector.build_manifest`.

**Look inside a collected session**:
```
python3 -m collector.inspect_npz data/authorized/<date>/<session_id>              # summary + csi_len breakdown
python3 -m collector.inspect_npz data/authorized/<date>/<session_id> --sample 100  # one full sample, decoded CSI
python3 -m collector.inspect_npz data/authorized/<date>/<session_id> --csv         # every field incl. raw CSI -> <session_dir>/csi_export.csv
```

## Repo layout

```
config/config.yaml   single source of truth for all configurable values
firmware/            ESP-IDF project (target esp32s3) -- ONE binary, role picked via NVS
scripts/             board detection, build/flash/monitor (role-aware), NVS config push, IP refresh, hotspot info, preflight
collector/           Python: config loading, wire-format parsing, transports, capacity test, CLI collector, inspect_npz.py viewer
data/                collected sessions (gitignored)
reports/capacity/    capacity test reports
```

## Notes / known constraints

- CSI density depends on ambient WiFi traffic (promiscuous mode is required for
  useful density -- a plain associated STA only sees CSI from frames addressed
  to it). With no dedicated transmitter board in this setup, the laptop's
  `collector/stimulus.py` UDP generator is the primary traffic source -- it
  runs automatically during `cli_collect` unless `--no-stimulus` is passed;
  don't pass that flag here, there's nothing else to generate traffic.
- The software BSSID filter (`csi.bssid_filter_enabled`) rejects CSI triggered
  by frames from unrelated nearby WiFi networks once promiscuous mode is on --
  works the same regardless of who hosts the network.
- An earlier hardware test (3-board: dedicated transmitter + receiver)
  confirmed the core CSI-capture-from-another-device mechanism works: with a
  transmitter sending steady 100Hz traffic, ~97% of the receiver's captured
  CSI samples shared one `src_mac` (the transmitter) rather than being
  scattered ambient noise. The same mechanism applies with the laptop as the
  traffic source instead.
- An optional 3rd-board `transmitter` role still exists in the firmware and
  `collector/nvs_gen.py` if you ever want to add a third ESP32 instead of
  relying on the laptop's stimulus -- it needs a `transmitter:` config section
  and a small `TransmitterConfig` dataclass re-added to `collector/config.py`.
- Model training/testing is explicit future work, not part of this repo -- see
  `RESEARCH_NOTES.md` for the literature review on that.
