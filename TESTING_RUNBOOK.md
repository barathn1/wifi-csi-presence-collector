# Testing Runbook

Exact commands, in order, to run once setup is done. **For first-time setup,
see one of:**
- **`SETUP_ANDROID_HOTSPOT.md`** -- `network.ap_source: "android"`
- **`SETUP_ESP_HOTSPOT.md`** -- `network.ap_source: "esp"`
- **`SETUP_GOOGLE_AP.md`** -- `network.ap_source: "google-ap"`

Everything below is identical regardless of which one you used.

Run everything below from `/home/bn3273/Desktop/wifi-final`.

---

## 0. Before you start

Connect your laptop to the network (see the "Every time" steps at the top of
whichever `SETUP_*.md` matches your `network.ap_source`), then run:
```
scripts/reconnect.sh
```
every time, even if it's the same laptop/AP/board as last time -- it
re-detects and pushes fresh config (MAC + laptop IP) rather than trusting
anything cached, which is what makes it safe to skip manually checking
either. It ends by running `connect_check.sh` for you and should print
`OK: board ... connected: ...`. If it doesn't, go back to whichever
`SETUP_*.md` guide matches your `network.ap_source` -- nothing below will
produce real data until this passes.

---

## 1. (Optional) Watch the hotspot board live -- only if `ap_source: "esp"`

In a separate terminal:
```
scripts/monitor.sh hotspot
```
You should see `hosting SSID '...' on channel ...`, then `station XX:XX:XX:XX:XX:XX
joined, aid=1` once the receiver (and/or your laptop) connects. Ctrl+] to exit
without resetting the board.

---

## 2. Test the receiver's transport capacity (data-loss check)

```
python3 -m collector.capacity_test --sweep --apply
```
Sweeps TCP and serial at several rates, measures real drop%/effective-Hz/jitter
using the receiver's own on-device counters as ground truth, writes a report to
`reports/capacity/`, and (`--apply`) writes the winning transport+rate back into
`config.yaml`. Expect TCP to win with ~0% loss, matching every prior test on
this hardware -- serial has a known USB-console reliability issue (see the
"Data loss" section in `SETUP_ANDROID_HOTSPOT.md` / `SETUP_ESP_HOTSPOT.md`).

To just find the max sustainable rate for one transport instead:
```
python3 -m collector.capacity_test --find-max-rate --transport tcp --apply
```

---

## 3. Collect a real labeled session

```
python3 -m collector.cli_collect --label authorized --person-id alice --motion standing --duration 120
```
Runs a preflight check first (fails loudly if the receiver isn't actually on
the network), then prints a live status line every 5s with time remaining and
a running packet-loss count:
```
1234 samples (65.2 Hz) | 45s elapsed, 75s remaining | 0 dropped (0.0% loss)
```
Loss here is real (detected from gaps in the firmware-assigned `seq` field per
sample), not an estimate. Ctrl+C or `kill <pid>` stop early and still save
whatever was captured -- only `kill -9` loses the in-progress session.

Repeat for other labels/people as needed:
```
python3 -m collector.cli_collect --label authorized --person-id bob --motion walking --duration 120
python3 -m collector.cli_collect --label unauthorized --person-id charlie --motion standing --duration 120
python3 -m collector.cli_collect --label none --duration 60
```

---

## 4. Inspect what you collected

```
python3 -m collector.inspect_npz data/authorized/<date>/<session_id>              # summary + csi_len breakdown
python3 -m collector.inspect_npz data/authorized/<date>/<session_id> --sample 100  # one full sample, decoded CSI
python3 -m collector.inspect_npz data/authorized/<date>/<session_id> --csv         # every field incl. raw CSI -> <session_dir>/csi_export.csv
```
`data/manifest.csv` is rebuilt automatically after every `cli_collect` run --
no manual step needed. Check its `ap_source` column matches what you expect
for the session you just collected.

---

## Troubleshooting quick reference

| Symptom | Likely cause / fix |
|---|---|
| `scripts/detect_board.sh` says no boards found | Check the needed `/dev/ttyACM*` device exists; replug USB. |
| `connect_check.sh` fails, `ap_source: "esp"` | Laptop not actually on the network yet, or SSID/password mismatch between `config.yaml` and what got pushed to the hotspot board -- re-run `scripts/push_config.sh hotspot` then reconnect your laptop's WiFi. |
| `connect_check.sh` fails, `ap_source: "android"` | Phone's hotspot isn't actually on, is on 5GHz (ESP32-S3 is 2.4GHz-only), or SSID/password on the phone doesn't match `config.yaml` -- fix whichever's mismatched. |
| `connect_check.sh` fails, `ap_source: "google-ap"` (or any other external AP) | AP is on 5GHz-only or band-steering away from 2.4GHz, or SSID/password doesn't match `config.yaml` -- fix whichever's mismatched. |
| `cli_collect.py` preflight fails | Same as above. `device.mac`/`network.laptop_ip` are both re-detected and re-pushed automatically by `scripts/reconnect.sh` -- if you skipped straight to `cli_collect.py` without running it first, run it now. If `reconnect.sh` itself fails, check the AP is actually up and the laptop is on it (`iw dev <iface> link`). |
| High packet loss in `cli_collect`/`capacity_test` on `transport.mode: tcp` | Shouldn't happen -- every prior test showed ~0% on TCP. If it does, re-run `capacity_test.py --sweep` and check the report in `reports/capacity/`. |
| High packet loss on `transport.mode: serial` | Expected/known issue (USB-Serial/JTAG console corrupts under load) -- switch back to `tcp`, don't chase this. |
| `data/manifest.csv`'s `ap_source` column is blank for a session | That session was collected before this field existed -- harmless, just no data to show for it. |
