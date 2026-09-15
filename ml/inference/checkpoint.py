"""Model checkpoint save/load -- didn't exist anywhere in this repo before (every training script
trained in-memory and discarded the model on exit). A checkpoint is a single `torch.save`'d plain dict:
the trained `state_dict`, the exact constructor kwargs used (never re-derived from the model class's
CURRENT defaults -- a checkpoint must stay correct even if e.g. WhoFiTransformer's own defaults change
later), the task-specific label mapping (`classes`, e.g. `[0, 1]` for taskD/task0_presence but
`["standing", "walking"]` for taskE after its string labels sort alphabetically -- this differs by task
and must be read from the checkpoint, never assumed), a human-readable `display_labels` pair for live
printing, and provenance metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ml.models.transformer_whofi import WhoFiTransformer

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "WhoFiTransformer": WhoFiTransformer,
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
    display_labels: list[str], task_name: str, preprocessing: str, mode: str,
    window_packets: int, stride_packets: int, train_dates: list[str], seed: int, epochs: int,
    train_loss_curve: list[float], notes: str = "",
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
            "task_name": task_name, "preprocessing": preprocessing, "mode": mode,
            "window_packets": window_packets, "stride_packets": stride_packets,
            "train_dates": train_dates, "seed": seed, "epochs": epochs,
            "train_loss_curve": train_loss_curve, "notes": notes,
            "torch_version": torch.__version__,
        },
    }
    torch.save(bundle, path)
    return path


def load_checkpoint(path: Path, map_location: str = "cpu") -> LoadedCheckpoint:
    # weights_only=False: this bundle is locally produced and trusted (plain dict of primitives + one
    # state_dict, not arbitrary untrusted input) -- explicit and documented, not a workaround for a
    # problem that exists today (verified this round-trips fine under either setting on torch 2.14).
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
