"""Actually run this against a REAL, currently-walking person -- not a replay of a recorded file.

Requires the ESP32 receiver board configured in config/config.yaml (`ac:27:6e:a5:5b:c8` by default --
the exact board every model here was trained on; a different board's data would be out-of-distribution,
see DEPLOYMENT.md) to be powered on and reachable over whatever transport that config specifies
(TCP by default). Uses the existing collector/ package (collector.receiver, collector.config) to speak
that transport/wire protocol -- re-deriving a binary framing format that already has tested, working
code for this exact hardware would be reinventing plumbing, not doing new analysis.

    python run_live.py [reveal_after_s] [model_name]
    python run_live.py 35 cnn_attention

NOT YET TESTED AGAINST REAL HARDWARE -- no ESP32 was connected to this machine while this was written.
The packet-source and windowing logic follow the exact same wire format/decode convention verified
against real recorded sessions elsewhere in this package (replay_demo.py), but the live connection
path itself (collector.receiver.get_receiver) has not been exercised end-to-end here. Run this with
real hardware connected before trusting it, and expect to debug the connection step specifically.
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

# collector/ lives in the wifi-csi-presence-collector repo -- find it whether this script is running
# from its dev copy (a sibling of that repo) or its pushed copy (inside csi_auth/ within that repo).
_here = Path(__file__).resolve().parent
for _candidate in (_here.parent, _here.parent / "wifi-csi-presence-collector"):
    if (_candidate / "collector").is_dir():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError("can't find the collector/ package (wifi-csi-presence-collector repo) -- "
                        "run this from inside that repo, or alongside it as a sibling directory")

from collector.config import load_config  # noqa: E402
from collector.receiver import get_receiver  # noqa: E402

from cleaning import hampel_filter_amplitude
from decode import decode_one_packet_bytes
from live_inference import LiveIdentitySession
from subcarrier_mask import keep_mask
from windowing import STRIDE_PACKETS, WINDOW_PACKETS

KEEP = keep_mask()
TARGET_CSI_LEN = 256  # 128 subcarriers -- the one format every model here was trained on; any other
                       # frame type this session negotiates gets silently dropped, same as the offline
                       # "keep only the dominant bucket" rule in decode.py, just decided up front here
                       # since there's no whole-session array to find a dominant bucket from live.


def run(reveal_after_s: float = 35.0, model_name: str = "cnn_attention") -> None:
    cfg = load_config()
    if cfg.device.mac.lower() != "ac:27:6e:a5:5b:c8":
        print(f"WARNING: config/config.yaml's device.mac is {cfg.device.mac!r}, not the "
              f"ac:27:6e:a5:5b:c8 every model here was trained on -- predictions will likely be "
              f"garbage (see DEPLOYMENT.md's board-mismatch bug).")
    print(f"connecting via transport.mode={cfg.transport.mode!r} "
          f"({'tcp://' + cfg.network.laptop_ip + ':' + str(cfg.network.tcp_port) if cfg.transport.mode == 'tcp' else cfg.transport.serial_port}) ...")

    session = LiveIdentitySession(reveal_after_s=reveal_after_s, identity_model_name=model_name)
    print(f"model={model_name!r}, reveal_after_s={reveal_after_s} -- waiting for presence...\n")

    amp_buf: deque = deque(maxlen=WINDOW_PACKETS)
    phase_buf: deque = deque(maxlen=WINDOW_PACKETS)
    rssi_buf: deque = deque(maxlen=WINDOW_PACKETS)
    packets_since_last_window = 0
    dropped_wrong_format = 0

    last_device_time_us: int | None = None
    cumulative_elapsed_s = 0.0
    last_display = None

    for sample in get_receiver(cfg):
        if sample.csi_len != TARGET_CSI_LEN:
            dropped_wrong_format += 1
            continue

        if last_device_time_us is not None:
            delta = max(0, sample.device_time_us - last_device_time_us)  # clock-reset guard, see
            cumulative_elapsed_s += delta / 1e6                          # decode.monotonic_elapsed_seconds
        last_device_time_us = sample.device_time_us

        amplitude, phase = decode_one_packet_bytes(sample.csi_data)
        amp_buf.append(amplitude)
        phase_buf.append(phase)
        rssi_buf.append(float(sample.rssi))
        packets_since_last_window += 1

        if len(amp_buf) < WINDOW_PACKETS or packets_since_last_window < STRIDE_PACKETS:
            continue
        packets_since_last_window = 0

        amp = np.stack(amp_buf)
        phase = np.stack(phase_buf)
        rssi = np.array(rssi_buf)
        amp, _ = hampel_filter_amplitude(amp)
        amp, phase = amp[:, KEEP], phase[:, KEEP]

        status = session.on_window(amp, phase, rssi, cumulative_elapsed_s)
        if status["display"] != last_display:
            print(f"  t={cumulative_elapsed_s:7.1f}s  -> {status}")
        last_display = status["display"]


if __name__ == "__main__":
    reveal_s = float(sys.argv[1]) if len(sys.argv) > 1 else 35.0
    model_name = sys.argv[2] if len(sys.argv) > 2 else "cnn_attention"
    run(reveal_after_s=reveal_s, model_name=model_name)
