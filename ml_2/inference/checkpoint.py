"""Model checkpoint save/load for the ml_2 live-inference models -- same bundle shape as
ml/inference/checkpoint.py (plain torch.save'd dict: state_dict + constructor kwargs + label mapping +
provenance), kept as a separate registry/module rather than extending the ml/ one so ml_2 stays
self-contained (no ml.* import) per the rest of ml_2/data and ml_2/models.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ml_2.models.gnn_subcarrier import SubcarrierGNN
from ml_2.models.radar_cnn import RadarInspiredCNN

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "SubcarrierGNN": SubcarrierGNN,
    "RadarInspiredCNN": RadarInspiredCNN,
}


@dataclass
class LoadedCheckpoint:
    model: nn.Module          # already .eval()'d -- this loader is inference-only
    classes: list
    display_labels: list[str]  # display_labels[i] is the human-readable name for classes[i]
    arch_kwargs: dict
    meta: dict[str, Any]


def save_checkpoint(
    model: nn.Module, path: Path, *, model_class: str, arch_kwargs: dict, classes: list,
    display_labels: list[str], task_name: str, window_packets: int, stride_packets: int,
    seed: int, epochs: int, train_loss_curve: list[float], notes: str = "",
) -> Path:
    assert model_class in MODEL_REGISTRY, f"unknown model_class {model_class!r}, add it to MODEL_REGISTRY"
    assert len(classes) == len(display_labels), (classes, display_labels)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_class": model_class,
        "arch_kwargs": arch_kwargs,
        "state_dict": model.state_dict(),
        "classes": classes,
        "display_labels": display_labels,
        "meta": {
            "task_name": task_name, "preprocessing": "calibA",
            "window_packets": window_packets, "stride_packets": stride_packets,
            "seed": seed, "epochs": epochs, "train_loss_curve": train_loss_curve,
            "notes": notes, "torch_version": torch.__version__,
        },
    }
    torch.save(bundle, path)
    return path


def load_checkpoint(path: Path, map_location: str = "cpu") -> LoadedCheckpoint:
    # weights_only=False: this bundle is locally produced and trusted (plain dict of primitives + one
    # state_dict), same documented choice as ml/inference/checkpoint.py.
    bundle = torch.load(Path(path), map_location=map_location, weights_only=False)
    cls = MODEL_REGISTRY[bundle["model_class"]]
    model = cls(**bundle["arch_kwargs"])
    model.load_state_dict(bundle["state_dict"])
    model.to(map_location)
    model.eval()
    return LoadedCheckpoint(
        model=model, classes=bundle["classes"], display_labels=bundle["display_labels"],
        arch_kwargs=bundle["arch_kwargs"], meta=bundle["meta"],
    )
