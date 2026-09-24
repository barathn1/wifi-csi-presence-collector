"""Explicit 3-receiver fusion (ml_2/models/receiver_fusion.py), scoped to sessions where all 3 ESP32s
actually recorded (2026-09-21/22 in this dataset). This is the alternative comparison to "each receiver
is its own independent sample" (used by every other model here): does explicitly fusing the 3 views of
the SAME event beat treating them as 3 separate training examples?

Cross-receiver alignment caveat: the 3 receivers have independent, unsynchronized onboard clocks, so
windows are aligned by POSITION within each session (the Nth window from each receiver) rather than an
exact shared timestamp -- an approximation, not exact sync.

    python3 -m ml_2.training.train_receiver_fusion_gpu --epochs 6
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ml_2.data.decode import DATA_DIR, REPO_ROOT
from ml_2.data.manifest import build_channel6_manifest
from ml_2.data.calibration import make_calibration_fn, per_receiver_day_baselines
from ml_2.data.splits import assert_no_group_leakage, leave_one_day_out, leave_one_unauthorized_person_out
from ml_2.data.metrics import compute_auroc, compute_eer, save_session_breakdown_csv
from ml_2.data.windowing import cache_all
from ml_2.models.receiver_fusion import ReceiverFusionTransformer
from ml_2.training.common_data import N_SUBCARRIERS
from ml_2.training.gpu_utils import DEVICE, FOLD_DEVICES, run_folds_parallel

LOG_PATH = REPO_ROOT / "ml_2/evaluation/results/receiver_fusion_gpu_log.csv"
SESSION_CSV_PATH = REPO_ROOT / "ml_2/evaluation/results/session_breakdown.csv"
LOG_FIELDNAMES = ["timestamp", "split_type", "held_out", "accuracy", "eer", "auroc",
                   "session_accuracy", "session_auroc", "n_sessions",
                   "false_accept_unauthorized", "false_accept_none", "n_train", "n_test", "notes"]
ARCH_KWARGS = dict(d_model=32, n_heads=4, d_ff=64, num_layers=1, norm_first=False, dropout=0.2)
WINDOW_PACKETS, STRIDE_PACKETS = 200, 100


def log_rows(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in LOG_FIELDNAMES})


def build_3receiver_manifest():
    manifest = build_channel6_manifest()
    counts = manifest.groupby("session_dir")["receiver_mac"].nunique()
    triple_sessions = counts[counts == 3].index
    manifest = manifest[manifest["session_dir"].isin(triple_sessions)].reset_index(drop=True)
    receiver_macs = sorted(manifest["receiver_mac"].unique())
    print(f"3-receiver sessions: {manifest['session_dir'].nunique()}, receivers={receiver_macs}, "
          f"dates={sorted(manifest['date'].unique())}")
    return cache_all(manifest), receiver_macs


class FusedWindowDataset(Dataset):
    def __init__(self, rows, receiver_macs: list[str], calibration):
        self.rows = rows.reset_index(drop=True)
        self.receiver_macs = receiver_macs
        self.calibration = calibration
        self.classes = [0, 1]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows.iloc[i]
        amps = []
        for mac in self.receiver_macs:
            with np.load(row[f"cache_path_{mac.replace(':', '')}"], mmap_mode="r") as d:
                sel = slice(int(row["start"]), int(row["end"]))
                amp, phase = np.array(d["amplitude"][sel]), np.array(d["phase"][sel])
            fake_row = {"date": row["date"], "receiver_mac": mac}
            amp, _ = self.calibration(amp, phase, fake_row)
            amps.append(torch.from_numpy(np.ascontiguousarray(amp, dtype=np.float32)))
        label = 1 if row["label"] == "authorized" else 0
        return amps, label

    def subset_by_index_rows(self, row_positions: np.ndarray) -> "FusedWindowDataset":
        sub = object.__new__(FusedWindowDataset)
        sub.rows = self.rows.iloc[row_positions].reset_index(drop=True)
        sub.receiver_macs = self.receiver_macs
        sub.calibration = self.calibration
        sub.classes = self.classes
        return sub


def fused_collate(batch):
    amps_lists, labels = zip(*batch)
    n_receivers = len(amps_lists[0])
    stacked = [torch.stack([sample[r] for sample in amps_lists]) for r in range(n_receivers)]
    return stacked, torch.tensor(labels, dtype=torch.long)


def build_fused_window_index(manifest, receiver_macs: list[str]):
    per_mac_paths = {mac: manifest[manifest["receiver_mac"] == mac].set_index("session_dir")["cache_path"]
                      for mac in receiver_macs}
    rows = []
    for session_dir, group in manifest.groupby("session_dir"):
        if not all(session_dir in per_mac_paths[mac].index for mac in receiver_macs):
            continue
        n_list = []
        for mac in receiver_macs:
            with np.load(per_mac_paths[mac][session_dir], mmap_mode="r") as d:
                n_list.append(d["amplitude"].shape[0])
        n = min(n_list)
        if n < WINDOW_PACKETS:
            continue
        base_row = group.iloc[0]
        for start in range(0, n - WINDOW_PACKETS + 1, STRIDE_PACKETS):
            row = {"session_dir": session_dir, "label": base_row["label"], "person_id": base_row["person_id"],
                   "date": base_row["date"], "start": start, "end": start + WINDOW_PACKETS}
            for mac in receiver_macs:
                row[f"cache_path_{mac.replace(':', '')}"] = per_mac_paths[mac][session_dir]
            rows.append(row)
    import pandas as pd
    return pd.DataFrame(rows)


def train_fused(model, train_ds, epochs: int, seed: int, batch_size: int = 64, lr: float = 1e-3,
                 device: torch.device = DEVICE) -> None:
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, multiprocessing_context="fork", collate_fn=fused_collate)
    model.train()
    for _ in range(epochs):
        for amps, y in loader:
            amps, y = [a.to(device) for a in amps], y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(amps), y)
            loss.backward()
            opt.step()


@torch.no_grad()
def fused_eval(model, test_ds, device: torch.device = DEVICE) -> dict:
    model.eval()
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, multiprocessing_context="fork", collate_fn=fused_collate)
    all_logits, all_y = [], []
    for amps, y in loader:
        logits = model([a.to(device) for a in amps])
        all_logits.append(logits.detach().cpu().numpy())
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

    from ml_2.data.metrics import format_session_breakdown, session_level_metrics
    sm = session_level_metrics(y_true, proba, pred, test_ds.rows["session_dir"].values, test_ds.rows["date"].values)
    out["session_accuracy"], out["session_auroc"], out["n_sessions"] = \
        sm["session_accuracy"], sm["session_auroc"], sm["n_sessions"]
    out["_session_breakdown_str"] = format_session_breakdown(sm)
    out["_per_session_table"] = sm["per_session_table"]
    return out


def main(epochs: int, seed: int) -> None:
    print(f"training on device: {DEVICE}")
    manifest, receiver_macs = build_3receiver_manifest()
    baselines = per_receiver_day_baselines(manifest)
    calibration = make_calibration_fn(baselines)
    fused_index = build_fused_window_index(manifest, receiver_macs)
    print(f"fused window index: {len(fused_index)} windows")
    full_ds = FusedWindowDataset(fused_index, receiver_macs, calibration)

    def _run_one(spec, device):
        held_out, train_idx, test_idx = spec
        train_ds, test_ds = full_ds.subset_by_index_rows(train_idx), full_ds.subset_by_index_rows(test_idx)
        model = ReceiverFusionTransformer(n_subcarriers=N_SUBCARRIERS, n_classes=2, n_receivers=len(receiver_macs),
                                           **ARCH_KWARGS)
        train_fused(model, train_ds, epochs, seed, device=device)
        m = fused_eval(model, test_ds, device=device)
        m.update({"held_out": held_out, "n_train": len(train_ds), "n_test": len(test_ds)})
        return m

    print(f"\n=== 3-receiver fusion (GPU) open-set: leave-one-unauthorized-person-out "
          f"(parallel across {len(FOLD_DEVICES)} device(s)) ===")
    fold_specs = []
    for held_out, train_idx, test_idx in leave_one_unauthorized_person_out(fused_index, include_none=True):
        assert_no_group_leakage(fused_index, train_idx, test_idx)
        fold_specs.append((held_out, train_idx, test_idx))
    fold_metrics = run_folds_parallel(fold_specs, _run_one)
    for m in fold_metrics:
        print(f"  held out '{m['held_out']}': WINDOW auroc={m['auroc']:.3f} eer={m['eer']:.3f}")
        print(m.pop("_session_breakdown_str", ""))
        per_session_table = m.pop("_per_session_table", None)
        if per_session_table is not None:
            save_session_breakdown_csv(per_session_table, SESSION_CSV_PATH, "receiver_fusion_3esp",
                                        "open_set_loo_stranger", m["held_out"])
        log_rows([{**m, "timestamp": datetime.now(timezone.utc).isoformat(), "split_type": "open_set_loo_stranger",
                    "notes": "leave-one-unauthorized-person-out, 3-receiver sessions only"}])
    print(f"  MEAN window_auroc={np.nanmean([m['auroc'] for m in fold_metrics]):.3f}  "
          f"MEAN session_auroc={np.nanmean([m['session_auroc'] for m in fold_metrics]):.3f}")

    print(f"\n=== 3-receiver fusion (GPU) cross-day: leave-one-day-out "
          f"(parallel across {len(FOLD_DEVICES)} device(s)) ===")
    fold_specs = []
    for held_out_date, train_idx, test_idx in leave_one_day_out(fused_index):
        if (fused_index.iloc[test_idx]["label"] == "authorized").sum() == 0:
            print(f"  skipping {held_out_date}: no authorized windows this day")
            continue
        fold_specs.append((held_out_date, train_idx, test_idx))
    fold_metrics = run_folds_parallel(fold_specs, _run_one)
    for m in fold_metrics:
        print(f"  held out day '{m['held_out']}': WINDOW auroc={m['auroc']:.3f} eer={m['eer']:.3f}")
        print(m.pop("_session_breakdown_str", ""))
        per_session_table = m.pop("_per_session_table", None)
        if per_session_table is not None:
            save_session_breakdown_csv(per_session_table, SESSION_CSV_PATH, "receiver_fusion_3esp",
                                        "cross_day_loo", m["held_out"])
        log_rows([{**m, "timestamp": datetime.now(timezone.utc).isoformat(), "split_type": "cross_day_loo",
                    "notes": "leave-one-day-out, 3-receiver sessions only"}])
    print(f"  MEAN window_auroc={np.nanmean([m['auroc'] for m in fold_metrics]):.3f}  "
          f"MEAN session_auroc={np.nanmean([m['session_auroc'] for m in fold_metrics]):.3f}")
    print(f"\nDone. Results -> {LOG_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(epochs=args.epochs, seed=args.seed)
