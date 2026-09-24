"""Does fusing all 3 ESP32 receivers help auth-vs-nonauth detection, compared to the single-receiver
setup used everywhere else in this project? Only 2026-09-21/22 ever had 3 receivers recording in
parallel (a4:cb:8f:d4:52:b0, ac:27:6e:a2:f2:78, ac:27:6e:a5:5b:c8) -- 2026-09-09/10/15/16/17 only ever
had 1 -- so this comparison is deliberately scoped to ONLY those 2 dates for both models: the single-
receiver control and the 3-receiver fusion model train/eval on the EXACT SAME 21+22 sessions, so any
difference measured is attributable to receiver count, not to which days were used.

Sessions missing any of the 3 receivers (2 legacy single-receiver `none` sessions recorded before the
3rd board joined, 1 `unauthorized` session where 2 of the 3 receivers dropped connection -- recorded as
an IP address in manifest.csv's board_mac field instead of a MAC, see manifest.csv row inspection) are
excluded from BOTH models, not just the fusion one, so the comparison stays apples-to-apples.

Receiver alignment caveat: the 3 receivers have independent, unsynchronized onboard clocks
(device_time_us is time-since-boot, not wall-clock) -- there is no exact cross-receiver timestamp to
join on. Windows are aligned by POSITION within each session instead (the Nth window from each
receiver, after every receiver's own stream is independently resampled onto the SAME target_rate_hz --
see time_resample.py), relying on all 3 receivers starting to capture at close to the same real moment
(the same collector script triggers all 3 simultaneously per session). This is an approximation, not an
exact sync -- worth knowing if the fusion result looks worse than expected.

    python3 -m ml.training.compare_receiver_fusion
    python3 -m ml.training.compare_receiver_fusion --epochs 6 --seed 1
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ml.data_pipeline.calibration import apply_variant_a, compute_day_baseline_from_sessions
from ml.data_pipeline.decode_csi import REPO_ROOT, channel_width_summary, load_session
from ml.data_pipeline.splits import assert_no_group_leakage, leave_one_day_out, leave_one_unauthorized_person_out
from ml.data_pipeline.time_resample import compute_target_rate_hz, session_native_rate_hz
from ml.data_pipeline.torch_dataset import CsiWindowDataset
from ml.data_pipeline.windowing import build_window_index, cache_session, load_manifest
from ml.evaluation.metrics import compute_auroc, compute_eer
from ml.models.receiver_fusion import ReceiverFusionTransformer
from ml.models.transformer_whofi import WhoFiTransformer
from ml.training.train import train_classifier
from ml.training.train_day3_ch6_model import evaluate_taskD, evaluate_taskD_day_disjoint, WINNING_ARCH_KWARGS

TRAIN_DATES = ["2026-09-21", "2026-09-22"]
TARGET_CHANNEL = 6
RECEIVER_MACS = ["a4:cb:8f:d4:52:b0", "ac:27:6e:a2:f2:78", "ac:27:6e:a5:5b:c8"]
SINGLE_RECEIVER_MAC = "ac:27:6e:a5:5b:c8"  # the receiver present on every date elsewhere in this project
CACHE_MODE = "resampled_timenorm"
LOG_PATH = REPO_ROOT / "ml/evaluation/results/receiver_fusion_log.csv"
LOG_FIELDNAMES = ["timestamp", "model", "stage", "held_out", "accuracy", "eer", "auroc",
                   "false_accept_unauthorized", "false_accept_none", "n_train", "n_test", "notes"]


def log_rows(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in LOG_FIELDNAMES})


def _mac_no_colons(mac: str) -> str:
    return mac.replace(":", "").lower()


def build_3receiver_manifest() -> pd.DataFrame:
    """2026-09-21/22 sessions where all 3 receivers actually wrote a file, confirmed channel-6 on the
    primary receiver (matches train_day3_ch6_model.py's own channel check).

    Cross-receiver alignment caveat (see also the module docstring): a device_time_us clock-reset
    (time_resample.fix_clock_reset) turned out to affect roughly half of these sessions on AT LEAST one
    of the 3 receivers -- almost always a small reset within the first ~0.1-2% of packets (consistent
    with a receiver re-initializing its timer right as each capture starts, not a random mid-session
    crash), not the rare, large mid-stream reset time_resample.py's docstring was originally written
    for. Requiring every receiver to be reset-free on every session would keep only 6 of 38 sessions --
    not usable. Instead, each receiver's own reset is repaired independently (fix_clock_reset, already
    applied inside cache_session's resampled_timenorm path) and position-based alignment is used
    as-is. Net effect: cross-receiver window alignment here is APPROXIMATE, off by up to roughly the
    reset's own offset (typically a fraction of a second to a couple of seconds) on affected sessions --
    a real, disclosed limitation of this specific comparison, not a hidden one."""
    manifest = load_manifest()
    date_mask = manifest["session_dir"].apply(lambda s: any(f"/{d}/" in s for d in TRAIN_DATES))
    pooled = manifest[date_mask & (manifest["board_mac"] == SINGLE_RECEIVER_MAC)].copy()

    def has_all_receivers(session_dir: str) -> bool:
        d = REPO_ROOT / "data" / session_dir
        return all((d / f"{_mac_no_colons(mac)}_samples.npz").exists() for mac in RECEIVER_MACS)

    pooled = pooled[pooled["session_dir"].apply(has_all_receivers)]

    def channel_ok(session_dir: str) -> bool:
        session = load_session(REPO_ROOT / "data" / session_dir, board_mac=SINGLE_RECEIVER_MAC)
        return channel_width_summary(session.npz)["channel_primary"] == TARGET_CHANNEL

    pooled = pooled[pooled["session_dir"].apply(channel_ok)]
    return pooled.reset_index(drop=True)


def compute_shared_target_rate_hz(manifest: pd.DataFrame) -> float:
    """Derived from ALL 3 receivers' native rates across every session in this 2-day pool (not just the
    single receiver's rates) -- so the single-receiver control model and the fusion model use the exact
    same window duration, keeping the comparison fair."""
    rates = []
    for session_dir in manifest["session_dir"]:
        for mac in RECEIVER_MACS:
            cache_path = cache_session(session_dir, mode="resampled", board_mac=mac)
            with np.load(cache_path) as d:
                rates.append(session_native_rate_hz(d["device_time_us"]))
    target = compute_target_rate_hz(rates)
    print(f"native packet rates across all 3 receivers: {min(rates):.1f}-{max(rates):.1f} Hz -> "
          f"shared time-normalization target: {target:.2f} Hz (200 packets/window -> "
          f"{200 / target:.2f}s/window)")
    return target


class FusedWindowDataset(Dataset):
    """One row per (session, window position); each row carries 3 cache paths (one per receiver,
    already time-normalized onto the shared rate) sliced at the SAME [start:end) -- see module
    docstring for the position-based alignment assumption."""

    def __init__(self, rows: pd.DataFrame, calibration_by_mac: dict[str, callable]):
        self.rows = rows.reset_index(drop=True)
        self.calibration_by_mac = calibration_by_mac
        self._session_cache: dict[str, dict[str, np.ndarray]] = {}
        self.classes = ["nonauth", "authorized"]

    def _arrays(self, cache_path: str) -> dict[str, np.ndarray]:
        cached = self._session_cache.get(cache_path)
        if cached is None:
            with np.load(cache_path) as d:
                cached = {"amplitude": d["amplitude"], "phase": d["phase"]}
            self._session_cache[cache_path] = cached
        return cached

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows.iloc[i]
        sel = slice(int(row["start"]), int(row["end"]))
        amps = []
        for mac in RECEIVER_MACS:
            arrays = self._arrays(row[f"cache_path_{_mac_no_colons(mac)}"])
            amp, phase = arrays["amplitude"][sel], arrays["phase"][sel]
            calib = self.calibration_by_mac.get(mac)
            if calib is not None:
                amp, phase = calib(amp, phase, row)
            amps.append(torch.from_numpy(np.ascontiguousarray(amp, dtype=np.float32)))
        label_idx = 1 if row["label"] == "authorized" else 0
        return amps, label_idx

    def subset_by_index_rows(self, row_positions: np.ndarray) -> "FusedWindowDataset":
        sub = object.__new__(FusedWindowDataset)
        sub.rows = self.rows.iloc[row_positions].reset_index(drop=True)
        sub.calibration_by_mac = self.calibration_by_mac
        sub._session_cache = self._session_cache
        sub.classes = self.classes
        return sub


def fused_collate(batch):
    amps_lists, labels = zip(*batch)
    n_receivers = len(amps_lists[0])
    stacked = [torch.stack([sample[r] for sample in amps_lists]) for r in range(n_receivers)]
    return stacked, torch.tensor(labels, dtype=torch.long)


def build_fused_window_index(manifest: pd.DataFrame, target_rate_hz: float, window_packets: int = 200,
                              stride_packets: int = 100) -> pd.DataFrame:
    per_mac_index = {
        mac: build_window_index(manifest=manifest, mode=CACHE_MODE, window_packets=window_packets,
                                 stride_packets=stride_packets, target_rate_hz=target_rate_hz, board_mac=mac)
        for mac in RECEIVER_MACS
    }
    rows = []
    for session_dir in manifest["session_dir"]:
        per_mac_rows = {mac: per_mac_index[mac][per_mac_index[mac]["session_dir"] == session_dir]
                         .reset_index(drop=True) for mac in RECEIVER_MACS}
        n = min(len(r) for r in per_mac_rows.values())
        if n == 0:
            continue
        base = per_mac_rows[SINGLE_RECEIVER_MAC]
        for pos in range(n):
            row = {
                "session_dir": session_dir, "label": base.loc[pos, "label"],
                "person_id": base.loc[pos, "person_id"], "motion": base.loc[pos, "motion"],
                "date": base.loc[pos, "date"], "start": int(base.loc[pos, "start"]),
                "end": int(base.loc[pos, "end"]), "window_start_time_us": int(base.loc[pos, "window_start_time_us"]),
            }
            for mac in RECEIVER_MACS:
                row[f"cache_path_{_mac_no_colons(mac)}"] = per_mac_rows[mac].loc[pos, "cache_path"]
            rows.append(row)
    return pd.DataFrame(rows)


@torch.no_grad()
def fused_binary_eval(model: torch.nn.Module, test_ds: FusedWindowDataset) -> dict:
    model.eval()
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=fused_collate)
    all_logits, all_y = [], []
    for amps, y in loader:
        all_logits.append(model(amps).numpy())
        all_y.append(y.numpy())
    logits, y_true = np.concatenate(all_logits), np.concatenate(all_y)
    proba = torch.softmax(torch.from_numpy(logits), dim=-1)[:, 1].numpy()
    pred = logits.argmax(axis=-1)
    eer, _ = compute_eer(y_true, proba)
    auroc = compute_auroc(y_true, proba)
    out = {"accuracy": float((pred == y_true).mean()), "eer": eer, "auroc": auroc}
    original_labels = test_ds.rows["label"].values
    for neg in ("unauthorized", "none"):
        m = original_labels == neg
        if m.sum() > 0:
            out[f"false_accept_{neg}"] = float((pred[m] == 1).mean())
    return out


def train_fused_classifier(model: torch.nn.Module, train_ds: FusedWindowDataset, epochs: int, seed: int,
                            batch_size: int = 64, lr: float = 1e-3) -> None:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=fused_collate)
    model.train()
    for _ in range(epochs):
        for amps, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(amps), y)
            loss.backward()
            opt.step()


def evaluate_fused_taskD(window_index: pd.DataFrame, calibration_by_mac, epochs: int, seed: int) -> None:
    full_ds = FusedWindowDataset(window_index, calibration_by_mac)
    print(f"\n=== 3receiver_fusion / taskD: leave-one-unauthorized-person-out ({len(full_ds)} fused windows) ===")
    fold_metrics = []
    for held_out, train_idx, test_idx in leave_one_unauthorized_person_out(window_index, include_none=True):
        assert_no_group_leakage(window_index, train_idx, test_idx, "session_dir")
        train_ds, test_ds = full_ds.subset_by_index_rows(train_idx), full_ds.subset_by_index_rows(test_idx)
        model = ReceiverFusionTransformer(n_subcarriers=128, n_classes=2, n_receivers=3, **WINNING_ARCH_KWARGS)
        train_fused_classifier(model, train_ds, epochs, seed)
        metrics = fused_binary_eval(model, test_ds)
        metrics.update({"held_out": held_out, "n_train": len(train_ds), "n_test": len(test_ds)})
        fold_metrics.append(metrics)
        print(f"  held out '{held_out}': acc={metrics['accuracy']:.3f} auroc={metrics['auroc']:.3f} "
              f"false_accept_unauth={metrics.get('false_accept_unauthorized', float('nan')):.3f} "
              f"false_accept_none={metrics.get('false_accept_none', float('nan')):.3f}")
        log_rows([{**metrics, "timestamp": datetime.now(timezone.utc).isoformat(), "model": "3receiver_fusion",
                   "stage": "eval_loo", "notes": "leave-one-unauthorized-person-out, 21+22 only"}])
    mean_auroc = np.nanmean([m["auroc"] for m in fold_metrics])
    mean_fa_u = np.nanmean([m.get("false_accept_unauthorized", float("nan")) for m in fold_metrics])
    mean_fa_n = np.nanmean([m.get("false_accept_none", float("nan")) for m in fold_metrics])
    print(f"  MEAN across {len(fold_metrics)} held-out strangers: auroc={mean_auroc:.3f} "
          f"false_accept_unauth={mean_fa_u:.3f} false_accept_none={mean_fa_n:.3f}")


def evaluate_fused_taskD_day_disjoint(window_index: pd.DataFrame, calibration_by_mac, epochs: int, seed: int) -> None:
    full_ds = FusedWindowDataset(window_index, calibration_by_mac)
    print(f"\n=== 3receiver_fusion / taskD: leave-one-DAY-out ({len(full_ds)} fused windows) ===")
    fold_metrics = []
    for held_out_date, train_idx, test_idx in leave_one_day_out(window_index):
        train_ds, test_ds = full_ds.subset_by_index_rows(train_idx), full_ds.subset_by_index_rows(test_idx)
        if (test_ds.rows["label"] == "authorized").sum() == 0:
            print(f"  skipping {held_out_date}: no authorized windows this day")
            continue
        model = ReceiverFusionTransformer(n_subcarriers=128, n_classes=2, n_receivers=3, **WINNING_ARCH_KWARGS)
        train_fused_classifier(model, train_ds, epochs, seed)
        metrics = fused_binary_eval(model, test_ds)
        metrics.update({"held_out": held_out_date, "n_train": len(train_ds), "n_test": len(test_ds)})
        fold_metrics.append(metrics)
        print(f"  held out day '{held_out_date}': acc={metrics['accuracy']:.3f} auroc={metrics['auroc']:.3f} "
              f"false_accept_unauth={metrics.get('false_accept_unauthorized', float('nan')):.3f} "
              f"false_accept_none={metrics.get('false_accept_none', float('nan')):.3f}")
        log_rows([{**metrics, "timestamp": datetime.now(timezone.utc).isoformat(), "model": "3receiver_fusion",
                   "stage": "eval_day_loo", "notes": "leave-one-day-out, 21+22 only"}])
    mean_auroc = np.nanmean([m["auroc"] for m in fold_metrics])
    mean_fa_u = np.nanmean([m.get("false_accept_unauthorized", float("nan")) for m in fold_metrics])
    mean_fa_n = np.nanmean([m.get("false_accept_none", float("nan")) for m in fold_metrics])
    print(f"  MEAN across {len(fold_metrics)} held-out days: auroc={mean_auroc:.3f} "
          f"false_accept_unauth={mean_fa_u:.3f} false_accept_none={mean_fa_n:.3f}")


def main(epochs: int, seed: int) -> None:
    manifest = build_3receiver_manifest()
    print(f"2026-09-21/22, all-3-receivers-present sessions: {len(manifest)} "
          f"({manifest['label'].value_counts().to_dict()})")
    target_rate_hz = compute_shared_target_rate_hz(manifest)

    none_sessions = manifest.loc[manifest["label"] == "none", "session_dir"].tolist()
    baseline_by_mac = {
        mac: compute_day_baseline_from_sessions(none_sessions, label=f"21-22_{_mac_no_colons(mac)}",
                                                  mode=CACHE_MODE, target_rate_hz=target_rate_hz, board_mac=mac)
        for mac in RECEIVER_MACS
    }
    calibration_by_mac = {mac: (lambda amp, phase, row, b=baseline_by_mac[mac]: apply_variant_a(amp, phase, b))
                           for mac in RECEIVER_MACS}

    # ---- single-receiver control (same 21+22 sessions, same shared target_rate_hz) ----
    single_index = build_window_index(manifest=manifest, mode=CACHE_MODE, window_packets=200, stride_packets=100,
                                       target_rate_hz=target_rate_hz, board_mac=SINGLE_RECEIVER_MAC)
    print(f"single-receiver window index: {len(single_index)} windows")

    def single_calibration(amp, phase, row):
        return apply_variant_a(amp, phase, baseline_by_mac[SINGLE_RECEIVER_MAC])

    print("\n########## SINGLE RECEIVER (control) ##########")
    evaluate_taskD(single_index, single_calibration, epochs, seed)
    evaluate_taskD_day_disjoint(single_index, single_calibration, epochs, seed)

    # ---- 3-receiver fusion ----
    fused_index = build_fused_window_index(manifest, target_rate_hz)
    print(f"\nfused (3-receiver) window index: {len(fused_index)} windows")

    print("\n########## 3-RECEIVER FUSION ##########")
    evaluate_fused_taskD(fused_index, calibration_by_mac, epochs, seed)
    evaluate_fused_taskD_day_disjoint(fused_index, calibration_by_mac, epochs, seed)

    print(f"\nDone. Single-receiver rows -> day3_ch6_model_log.csv, fusion rows -> {LOG_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(epochs=args.epochs, seed=args.seed)
