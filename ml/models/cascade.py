"""OWN idea, systems-level rather than architectural: a presence-gate -> identity cascade. None of the
cited papers test this -- they all evaluate a single end-to-end classifier. Task 0 (presence) looks like
the easiest, highest-confidence sub-problem (RandomForest already gets ~88% same-day, and OpenCSI
reports 0.99 F1 for the analogous binary-occupancy problem elsewhere), so gating the harder
identity/verification model behind a cheap, reliable presence check should reduce how often the
harder model is even asked to make a call on an empty room -- where it has no business making an
identity guess in the first place.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PresenceIdentityCascade:
    def __init__(self, presence_model: nn.Module, identity_model: nn.Module, identity_classes: list[str],
                 presence_threshold: float = 0.5):
        self.presence_model = presence_model.eval()
        self.identity_model = identity_model.eval()
        self.identity_classes = identity_classes
        self.presence_threshold = presence_threshold

    @torch.no_grad()
    def predict(self, amplitude: torch.Tensor, phase: torch.Tensor) -> list[str]:
        """presence_model: binary logits (index 1 = occupied). identity_model: logits over
        identity_classes, e.g. ["authorized_anjali", "authorized_barath", "unauthorized"]."""
        presence_logits = self.presence_model(amplitude, phase)
        occupied_prob = torch.softmax(presence_logits, dim=-1)[:, 1]
        occupied = occupied_prob > self.presence_threshold

        predictions = ["none"] * amplitude.shape[0]
        if occupied.any():
            idx = occupied.nonzero(as_tuple=True)[0]
            identity_logits = self.identity_model(amplitude[idx], phase[idx])
            identity_pred = identity_logits.argmax(dim=-1)
            for pos, i in enumerate(idx.tolist()):
                predictions[i] = self.identity_classes[identity_pred[pos].item()]
        return predictions
