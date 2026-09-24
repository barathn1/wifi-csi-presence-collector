"""Train the deployable GNN (SubcarrierGNN) and radar-CNN (RadarInspiredCNN) checkpoints for live
testing -- trains on ALL of the channel-6 dataset (no held-out fold), same "this IS the shipped
artifact" philosophy as ml/training/train_final_model.py / ml/training/train_day3_ch6_model.py's final
stage. The honest generalization estimate for these two models already exists (leave-one-stranger-out /
leave-one-day-out, see ml_2/evaluation/results/DAY_PERSON_MODEL_ACCURACY.md) -- this script is purely
about producing a checkpoint file that ml_2/inference/live_infer.py can load, it does not re-derive or
replace that evaluation.

    python3 -m ml_2.training.train_live_checkpoints
    python3 -m ml_2.training.train_live_checkpoints --epochs 8
"""
from __future__ import annotations

import argparse

from ml_2.data.decode import REPO_ROOT
from ml_2.data.torch_dataset import CsiWindowDataset
from ml_2.inference.checkpoint import save_checkpoint
from ml_2.models.gnn_subcarrier import SubcarrierGNN
from ml_2.models.radar_cnn import RadarInspiredCNN
from ml_2.training.common_data import N_SUBCARRIERS, load_channel6_dataset
from ml_2.training.gpu_utils import DEVICE, train_classifier_gpu

CHECKPOINT_DIR = REPO_ROOT / "ml_2/checkpoints"
TASK_NAME = "auth_vs_nonauth"
WINDOW_PACKETS = 200
STRIDE_PACKETS = 100

MODEL_SPECS = {
    "gnn_subcarrier": ("SubcarrierGNN", lambda n_cls: SubcarrierGNN(n_subcarriers=N_SUBCARRIERS, n_classes=n_cls)),
    "radar_cnn": ("RadarInspiredCNN", lambda n_cls: RadarInspiredCNN(n_subcarriers=N_SUBCARRIERS, n_classes=n_cls)),
}
# display_labels[i] names classes[i] -- task_auth_vs_nonauth's y is 0/1 int, so sorted(set(y)) == [0, 1]
# always, with 1 == authorized (see ml_2/data/tasks.py::task_auth_vs_nonauth).
CLASSES = [0, 1]
DISPLAY_LABELS = ["NOT AUTHORIZED", "AUTHORIZED"]


def main(epochs: int, seed: int) -> None:
    print(f"training on device: {DEVICE}")
    dataset = load_channel6_dataset(window_packets=WINDOW_PACKETS, stride_packets=STRIDE_PACKETS)
    full_ds = CsiWindowDataset(dataset.window_index, TASK_NAME, calibration=dataset.calibration, classes=CLASSES)
    trivial_baseline = max((full_ds.y == c).mean() for c in CLASSES)

    for name, (model_class, factory) in MODEL_SPECS.items():
        print(f"\n=== {name}: {len(full_ds)} windows, classes={CLASSES}, "
              f"trivial-always-predict-majority baseline={trivial_baseline:.4f} ===")
        model = factory(len(CLASSES))
        # no held-out split: this checkpoint is meant to be trained on every available window before
        # shipping, same rationale as train_final_model.py -- the live test itself is the real,
        # never-seen-in-training evaluation. The printed accuracy below is an IN-SAMPLE SANITY CHECK
        # ONLY (confirms training worked; compare against the trivial baseline above), not a
        # generalization estimate -- see ml_2/evaluation/results/DAY_PERSON_MODEL_ACCURACY.md for the
        # real held-out numbers for these two models.
        result = train_classifier_gpu(model, full_ds, full_ds, epochs=epochs, seed=seed)
        print(f"  in-sample sanity accuracy (NOT a generalization estimate): {result['accuracy']:.4f} "
              f"(trivial baseline: {trivial_baseline:.4f})")

        path = save_checkpoint(
            model, CHECKPOINT_DIR / f"{name}_calibA.pt", model_class=model_class,
            arch_kwargs={"n_subcarriers": N_SUBCARRIERS, "n_classes": len(CLASSES)},
            classes=CLASSES, display_labels=DISPLAY_LABELS, task_name=TASK_NAME,
            window_packets=WINDOW_PACKETS, stride_packets=STRIDE_PACKETS, seed=seed, epochs=epochs,
            train_loss_curve=result["train_loss_curve"],
            notes="trained on 100% of the channel-6 pooled dataset, no held-out split -- see "
                  "ml_2/evaluation/results/DAY_PERSON_MODEL_ACCURACY.md for the real generalization numbers",
        )
        print(f"  saved -> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(epochs=args.epochs, seed=args.seed)
