# Setup: ACT Fibernet home router as hotspot

One of several switchable setups -- see `SETUP_ANDROID_HOTSPOT.md` (phone),
`SETUP_ESP_HOTSPOT.md` (a second ESP32 hosts the network), and
`SETUP_GOOGLE_AP.md` (Google WiFi/Nest) for the others. Switching between them
is just changing `network.ap_source` in `config/config.yaml` and following the
matching guide; nothing else in this repo needs to change.

**For testing commands after setup is done, see `TESTING_RUNBOOK.md`.**

## Setup

### Every time (routine use, after first-time setup below is done once)

1. Connect your laptop's WiFi to `ACTFIBERNET` (it should already be
   broadcasting -- home routers don't need to be manually turned on).
2. Run:
   ```
   scripts/reconnect.sh
   ```
   That's it. It sees `ap_source: "home-router"`, skips the ESP-hotspot step,
   refreshes your laptop's IP, re-detects and pushes the receiver's config
   (which also re-caches its MAC), and confirms it joined -- **you never need
   to look up or type in a MAC address or IP yourself**, every run, even
   across board swaps.

**Always run `reconnect.sh` right before collecting, every session -- don't
skip it just because it's the same laptop and the same router as last time.**
The ESP's MAC is genuinely constant (it's burned into the chip), but the
laptop's IP is not: it's assigned by the router's DHCP server, and DHCP
doesn't guarantee handing back the same address on every reconnect (see the
same failure mode documented in `SETUP_GOOGLE_AP.md` -- it drifted once and
caused every collection attempt to silently fail with 0 samples even though
the ESP had joined WiFi fine). That's exactly the failure mode `reconnect.sh`
closes off by re-checking the IP every time instead of trusting a cached
value.

### First time only (or after changing firmware code)

1. Plug in the RECEIVER board over USB (`/dev/ttyACM0`).
2. Build and flash it:
   ```
   scripts/build.sh
   scripts/flash.sh receiver
   ```
3. Connect your laptop's WiFi to `ACTFIBERNET` manually.
4. Refresh the laptop's IP for this network, push the receiver's config, and confirm:
   ```
   scripts/get_laptop_ip.sh --set
   scripts/push_config.sh receiver
   scripts/connect_check.sh
   ```
   Skipping `get_laptop_ip.sh --set` leaves whatever `network.laptop_ip` was last
   in `config.yaml` (e.g. a stale value from a previous `ap_source`) pushed to
   the board -- it'll join WiFi fine but never reach your laptop, and
   `connect_check.sh` will time out with the board's MAC missing from the ARP
   table (it only ever shows the router's MAC).

## Collect a labeled session

Once `reconnect.sh` (or `connect_check.sh`) reports OK, here's every
`cli_collect.py` parameter spelled out:
```
python3 -m collector.cli_collect \
  --label authorized \
  --person-id alice \
  --motion standing \
  --notes "living room, facing the router" \
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
the router's network), then streams real CSI, printing a live status line
every 5s with time remaining and packet loss so far, and writes to
`data/<label>/<date>/<session_id>/{metadata.json, samples.npz}`. See
`TESTING_RUNBOOK.md` for the full sequence (capacity test first, then this,
then inspecting what you collected).

## Topology

- An **ACT Fibernet home router** hosts the network. Like most ISP-provided
  routers, there's usually nothing to manually "turn on" -- it's already
  broadcasting. Just confirm it's reachable on **2.4GHz** -- some ISP routers
  band-steer a single SSID across 2.4GHz and 5GHz automatically, or expose
  separate `ACTFIBERNET`/`ACTFIBERNET_5G`-style SSIDs; the ESP32-S3 radio is
  2.4GHz-only, so if it fails to associate, check the router's band settings
  (usually in its admin page) first.
- **One ESP32-S3 board (`/dev/ttyACM0`) -- RECEIVER.** Joins the router's
  network as a station, captures CSI in promiscuous mode, and streams it to
  the laptop over TCP. This is the only board needed for this setup.
- The laptop also joins the router's network, and is:
  - the **TCP server** the receiver connects to (so the main data path never
    needs to know the receiver's DHCP-assigned IP -- only the laptop's own IP,
    refreshed via `scripts/get_laptop_ip.sh --set`), and
  - the **traffic generator** (`collector/stimulus.py`, wired into
    `cli_collect.py` by default) -- there's no dedicated transmitter board in
    this setup, so the laptop's UDP stimulus is what gives the receiver
    something to extract CSI from.

**Note on your laptop's internet**: joining the router's network means your
laptop leaves whatever network it was on. Since this is your home/office
router's own internet connection, you keep internet through it.

## Everything configurable lives in one file

**`config/config.yaml`** -- AP source label, hotspot SSID/password, the
receiver's serial port, the laptop's IP, CSI capture parameters, dataset
output dir. Never edit IPs/SSIDs/rates/ports anywhere else.

```yaml
network:
  ap_source: "home-router"      # switch to "esp"/"android"/"google-ap" to use the other SETUP_*.md guides instead
  hotspot_ssid: "ACTFIBERNET"   # the router's SSID
  hotspot_password: ""          # the router's password (empty if it's an open network)
  laptop_ip: "..."              # refreshed by scripts/get_laptop_ip.sh --set

transport:
  serial_port: "/dev/ttyACM0"   # the RECEIVER board
  mode: "tcp"
```

### Config values you need to change for this setup

| Field | What to set it to |
|---|---|
| `network.ap_source` | `"home-router"` -- already set. Purely descriptive (any value other than `"esp"` just means "some external AP already running") -- it gets recorded in every session's `metadata.json`/`data/manifest.csv` so later analysis can tell this router's sessions apart from `"android"`-, `"esp"`-, or `"google-ap"`-sourced sessions. |
| `network.hotspot_ssid` | `"ACTFIBERNET"` -- already set, matches the router. |
| `network.hotspot_password` | `""` -- already set, matches the router (currently an open network with no password). |
| `network.laptop_iface` | Your laptop's WiFi interface name (e.g. `wlp0s20f3`). Only needs changing if your machine's interface name differs -- check with `ip link`. |
| `network.laptop_ip` | Leave as-is; `scripts/get_laptop_ip.sh --set` (called by `reconnect.sh`) fills this in automatically once you're on the router's network -- it will be a different subnet than any previous `ap_source` you used. |
| `transport.serial_port` | `"/dev/ttyACM0"` -- already set for this scenario's single receiver board. Change if `scripts/detect_board.sh` shows it enumerating elsewhere on your machine. |
| `device.mac` | Leave blank/as-is; `scripts/push_config.sh receiver` (called by `reconnect.sh`) re-detects and caches it automatically every run -- never edit this by hand. |

Everything else in `config/config.yaml` (CSI capture parameters, capacity-test
rates, dataset output dir) has working defaults and doesn't need to change for
this setup specifically.

## Quick reference

| Script | When to run it |
|---|---|
| `scripts/detect_board.sh` | Find which `/dev/ttyACM*` the receiver is on. (`--set-mac` is only for manual debugging -- `push_config.sh receiver` already does this automatically.) |
| `scripts/build.sh` | Only after editing **firmware C code**. |
| `scripts/flash.sh receiver` | Flash the receiver board. |
| `scripts/get_laptop_ip.sh --set` | Laptop's IP on the network changed. No board interaction. |
| `scripts/push_config.sh receiver` | Push config.yaml to the receiver's NVS (~1-2s, no reflash). Run after any config change. |
| `scripts/reconnect.sh` | **The one command for routine use.** Sees `ap_source: "home-router"` (not `"esp"`), skips any ESP-hotspot step, refreshes laptop IP, pushes receiver config, confirms it joined. |
| `scripts/connect_check.sh` | Quick "is the receiver actually on the network right now?" check. |
| `scripts/monitor.sh receiver` | Live serial console for the receiver, for debugging. |

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

**TCP is the default and recommended transport** -- it showed ~0% real loss
across thousands of samples in every test on this hardware. `transport.mode`
should stay `"tcp"`; `serial` has a known USB-console reliability issue under
sustained load, don't switch to it for real collection.

## What a collected session looks like

```
data/<label>/<date>/<session_id>/
  metadata.json   # label, person_id, notes, timing, transport, board MAC, ap_source, motion, dropped_samples, avg_rate_hz, config snapshot
  samples.npz      # CSI samples: fixed-width scalar fields + ragged CSI payloads
                    # (csi_flat + csi_offset + csi_len, since CSI length varies per packet)
```
Every session collected under this setup will have `ap_source: "home-router"`
in its metadata and in the `data/manifest.csv` row, so later analysis can
filter or compare against `"android"`-, `"esp"`-, or `"google-ap"`-sourced
sessions from the other `SETUP_*.md` guides.

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
scripts/             board detection, build/flash/monitor, NVS config push, IP refresh, preflight
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
- A home router may have other devices already connected to it (phones,
  laptops, IoT devices already on your home network) -- the BSSID filter means
  only CSI triggered by frames to/from this specific router's BSSID counts,
  but ambient traffic from other real devices on the same network will also
  register as CSI-triggering traffic, same as any other `ap_source`.
- Model training/testing is explicit future work, not part of this repo -- see
  `RESEARCH_NOTES.md` for the literature review on that.
