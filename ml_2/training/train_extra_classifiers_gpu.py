"""Trains the two "creative" classifier-style models -- SubcarrierGNN and RadarInspiredCNN -- on GPU,
same harness as train_transformer_gpu.py (both share the forward(amplitude, phase) -> logits
interface, so they reuse the exact same training/eval code, just a different MODEL_FACTORIES entry).

    python3 -m ml_2.training.train_extra_classifiers_gpu --epochs 6
"""
from __future__ import annotations

import argparse

from ml_2.models.gnn_subcarrier import SubcarrierGNN
from ml_2.models.radar_cnn import RadarInspiredCNN
from ml_2.training import train_transformer_gpu as base
from ml_2.training.common_data import N_SUBCARRIERS

EXTRA_FACTORIES = {
    "gnn_subcarrier": lambda n_cls: SubcarrierGNN(n_subcarriers=N_SUBCARRIERS, n_classes=n_cls),
    "radar_cnn": lambda n_cls: RadarInspiredCNN(n_subcarriers=N_SUBCARRIERS, n_classes=n_cls),
}


def main(epochs: int, seed: int) -> None:
    print(f"training on device: {base.DEVICE}")
    dataset = base.load_channel6_dataset()
    full_ds = base.CsiWindowDataset(dataset.window_index, "auth_vs_nonauth", calibration=dataset.calibration)

    base.MODEL_FACTORIES.update(EXTRA_FACTORIES)
    for model_name in EXTRA_FACTORIES:
        base.evaluate_open_set(model_name, full_ds, dataset.window_index, epochs, seed)
        base.evaluate_cross_day(model_name, full_ds, dataset.window_index, epochs, seed)
    print(f"\nDone. Results -> {base.LOG_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(epochs=args.epochs, seed=args.seed)
