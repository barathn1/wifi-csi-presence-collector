"""Live inference engine: presence-gate -> silent accumulation -> reveal -> continuous update.

See DEPLOYMENT.md for the design rationale (why gate on presence first, why reveal at ~30-45s rather
than immediately or at 60s, why CNN+Attention is the model driving the identity call). This module is
transport-agnostic -- it doesn't know or care whether windows come from a live ESP32 stream or a
replayed recording (see replay_demo.py for the latter); it just needs one already-decoded,
already-cleaned, already-masked window at a time, plus that window's elapsed time since the stream
started (from the hardware's own device_time_us, not wall-clock/window-count -- see FINDINGS.md on why
"1 window" is not "1 second").
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import torch

from features import window_features
from models import CnnAttention, CnnLstm

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

REVEAL_AFTER_S = 35.0  # middle of the recommended 30-45s window -- see DEPLOYMENT.md
PRESENCE_THRESHOLD = 0.5

# Which identity model to run -- see DEPLOYMENT.md for why cnn_attention is the recommended default
# (strongest or tied-strongest at nearly every checkpoint in the timing table); svm/cnn_bilstm are
# here so you can compare them live, not because either beats it.
IDENTITY_MODEL_CLASSES = {"cnn_attention": CnnAttention, "cnn_bilstm": CnnLstm}  # "svm" handled separately

# Debounce: require a rolling MAJORITY of the last N presence votes to agree before confirming OR
# dropping presence, rather than trusting any single window. Found necessary by testing, not
# theoretical: the presence gate's headline 99.5% is a POOLED training-set number (see
# train_presence.py), not a per-window guarantee -- replaying a real empty-room recording
# window-by-window (as a live feed actually would) hit a false-positive on window 0 without this.
PRESENCE_VOTE_WINDOW = 5
PRESENCE_VOTE_MIN_POSITIVE = 4


def load_presence_model():
    return joblib.load(CHECKPOINT_DIR / "presence_final.joblib")


def load_identity_model(model_name: str = "cnn_attention"):
    """Returns (kind, model, extra) where kind is "svm" or "torch" -- LiveIdentitySession dispatches
    on it since the SVM pipeline and the two torch models take different inputs (handcrafted stats
    vector vs. raw amplitude window)."""
    if model_name == "svm":
        pipeline = joblib.load(CHECKPOINT_DIR / "svm_final.joblib")
        return "svm", pipeline, {"label_meaning": {0: "anjali", 1: "barath"}}
    if model_name not in IDENTITY_MODEL_CLASSES:
        raise ValueError(f"model_name must be one of {['svm', *IDENTITY_MODEL_CLASSES]}, got {model_name!r}")
    ckpt = torch.load(CHECKPOINT_DIR / f"{model_name}_final.pt", weights_only=False)
    model = IDENTITY_MODEL_CLASSES[model_name](n_subcarriers=ckpt["n_subcarriers"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return "torch", model, {"sub_mean": ckpt["sub_mean"], "sub_std": ckpt["sub_std"],
                             "label_meaning": ckpt["label_meaning"]}


@dataclass
class _State:
    presence_confirmed: bool = False
    presence_start_s: float | None = None
    revealed: bool = False
    window_probas: list = field(default_factory=list)  # [(elapsed_s, P(barath))]
    last_label: str | None = None
    last_confidence: float | None = None
    last_elapsed_s: float = 0.0


class LiveIdentitySession:
    """One instance per continuous presence-tracking session. Feed windows via `on_window`; call
    `status()` any time for what should currently be on screen."""

    def __init__(self, reveal_after_s: float = REVEAL_AFTER_S, identity_model_name: str = "cnn_attention"):
        self.reveal_after_s = reveal_after_s
        self.identity_model_name = identity_model_name
        self.presence_model = load_presence_model()
        self.identity_kind, self.identity_model, extra = load_identity_model(identity_model_name)
        self.label_meaning = extra["label_meaning"]
        if self.identity_kind == "torch":
            self.sub_mean, self.sub_std = extra["sub_mean"], extra["sub_std"]
        self.state = _State()
        self.presence_votes: deque = deque(maxlen=PRESENCE_VOTE_WINDOW)

    def _presence_proba(self, stats_feat: np.ndarray) -> float:
        return float(self.presence_model.predict_proba(stats_feat[None, :])[0, 1])

    def _identity_proba(self, stats_feat: np.ndarray, raw_amp_window: np.ndarray) -> float:
        if self.identity_kind == "svm":
            return float(self.identity_model.predict_proba(stats_feat[None, :])[0, 1])
        x = ((raw_amp_window - self.sub_mean[0]) / self.sub_std[0]).astype(np.float32)
        with torch.no_grad():
            logit = self.identity_model(torch.from_numpy(x[None, ...]))
        return float(torch.sigmoid(logit).item())

    def on_window(self, amplitude: np.ndarray, phase: np.ndarray, rssi: np.ndarray,
                  elapsed_s: float) -> dict:
        """amplitude/phase: (window_packets, 109) -- ALREADY Hampel-cleaned and null-subcarrier-masked
        (see dataset.py/train_final.py for the exact prep). rssi: (window_packets,). `elapsed_s`:
        seconds since this window's stream started, from device_time_us."""
        stats_feat = window_features(amplitude, phase, rssi)
        presence_p = self._presence_proba(stats_feat)
        self.presence_votes.append(presence_p > PRESENCE_THRESHOLD)

        votes_say_present = (len(self.presence_votes) == PRESENCE_VOTE_WINDOW
                              and sum(self.presence_votes) >= PRESENCE_VOTE_MIN_POSITIVE)
        if not votes_say_present:
            if self.state.presence_confirmed:
                # sustained absence after being present -- they left; reset for a fresh next visit
                self.state = _State()
                self.presence_votes.clear()
            return self.status()

        if not self.state.presence_confirmed:
            self.state.presence_confirmed = True
            self.state.presence_start_s = elapsed_s

        identity_p = self._identity_proba(stats_feat, amplitude)
        self.state.window_probas.append((elapsed_s, identity_p))
        self.state.last_elapsed_s = elapsed_s

        time_since_presence = elapsed_s - self.state.presence_start_s
        if time_since_presence >= self.reveal_after_s:
            running_mean = float(np.mean([p for _, p in self.state.window_probas]))
            predicted_class = int(running_mean > 0.5)
            self.state.revealed = True
            self.state.last_confidence = running_mean if predicted_class == 1 else 1.0 - running_mean
            self.state.last_label = self.label_meaning[predicted_class]

        return self.status()

    def status(self) -> dict:
        if not self.state.presence_confirmed:
            return {"display": "idle", "label": None, "confidence": None}
        elapsed_since_presence = self.state.last_elapsed_s - self.state.presence_start_s
        if not self.state.revealed:
            return {"display": "accumulating (silent)", "label": None, "confidence": None,
                    "seconds_since_presence": round(elapsed_since_presence, 1),
                    "seconds_until_reveal": round(max(0.0, self.reveal_after_s - elapsed_since_presence), 1)}
        return {"display": "revealed", "label": self.state.last_label,
                "confidence": round(self.state.last_confidence, 3),
                "seconds_since_presence": round(elapsed_since_presence, 1)}
