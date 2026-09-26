"""The model zoo: two classical baselines (paper-1-style SVM, plus RandomForest) and a CNN+LSTM hybrid
deep model on raw amplitude sequences (paper-2-style), all independently implemented here.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC


def make_svm():
    return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced",
               max_iter=3000)


def make_rf():
    return RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced",
                                   random_state=0, n_jobs=-1)


def make_hgb():
    """Gradient-boosted trees -- usually a stronger tabular baseline than RandomForest, and a cheap
    thing to try given RF was already competitive with the deep branch."""
    return HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.08,
                                           class_weight="balanced", random_state=0)


class CnnLstm(nn.Module):
    """(batch, time, subcarriers) amplitude -> P(auth). Conv1d over the time axis extracts local
    temporal shape per subcarrier-channel; a BIDIRECTIONAL LSTM then models how that shape evolves
    across the window (both directions -- a stride-100 window has no meaningful "forward-only" causal
    constraint, so there's no reason to throw away the backward-pass information); a small MLP head
    makes the final call. Attention-weighted pooling over the LSTM's per-timestep outputs replaces
    "just take the last hidden state" -- lets the model weight whichever part of the window was most
    informative instead of relying on whatever the last few timesteps happened to look like."""

    def __init__(self, n_subcarriers: int = 128, hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_subcarriers, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(input_size=hidden, hidden_size=hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden * 2, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1),
        )

    def forward(self, x):  # x: (batch, time, subcarriers)
        x = x.transpose(1, 2)          # (batch, subcarriers, time) for Conv1d
        x = self.conv(x)               # (batch, hidden, time/4)
        x = x.transpose(1, 2)          # (batch, time/4, hidden)
        seq, _ = self.lstm(x)          # (batch, time/4, hidden*2)
        weights = torch.softmax(self.attn(seq).squeeze(-1), dim=1)   # (batch, time/4)
        pooled = (seq * weights.unsqueeze(-1)).sum(dim=1)            # (batch, hidden*2)
        return self.head(pooled).squeeze(-1)  # logits


class CnnAttention(nn.Module):
    """(batch, time, subcarriers) amplitude -> P(auth). Same conv frontend as CnnLstm, but NO
    recurrence: a single multi-head self-attention block lets every timestep of the conv output attend
    to every other timestep directly (no need to pass information through a recurrent state), then the
    same attention-weighted pooling collapses the sequence before the MLP head. A learned positional
    embedding is added before self-attention since attention itself is permutation-invariant and would
    otherwise lose the sequence's time ordering entirely."""

    def __init__(self, n_subcarriers: int = 128, hidden: int = 64, n_heads: int = 4, seq_len: int = 50):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_subcarriers, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden))
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.pool_attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1),
        )

    def embed(self, x):  # x: (batch, time, subcarriers) -> (batch, hidden) pre-classifier embedding
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)                       # (batch, T, hidden)
        x = x + self.pos_embed[:, :x.shape[1], :]
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm(x + attn_out)
        weights = torch.softmax(self.pool_attn(x).squeeze(-1), dim=1)
        return (x * weights.unsqueeze(-1)).sum(dim=1)

    def forward(self, x):  # x: (batch, time, subcarriers) -> (batch,) logits
        return self.head(self.embed(x)).squeeze(-1)


def train_torch_model(model_cls, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray | None = None,
                       y_val: np.ndarray | None = None,
                       epochs: int = 8, batch_size: int = 64, seed: int = 0):
    torch.manual_seed(seed)
    model = model_cls(n_subcarriers=X_train.shape[2])
    n_pos, n_neg = (y_train == 1).sum(), (y_train == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    Xt = torch.from_numpy(X_train)
    yt = torch.from_numpy(y_train.astype(np.float32))
    n = len(Xt)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            logits = model(Xt[idx])
            loss = criterion(logits, yt[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        msg = f"    epoch {epoch + 1}/{epochs}: train_loss={total_loss / n:.4f}"
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_logits = model(torch.from_numpy(X_val))
                val_pred = (torch.sigmoid(val_logits) > 0.5).numpy().astype(int)
            msg += f" inner_val_acc={(val_pred == y_val).mean():.3f}"
        print(msg)
    return model


def torch_model_predict_proba(model, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        return torch.sigmoid(logits).numpy()


def torch_model_embed(model: CnnAttention, X: np.ndarray) -> np.ndarray:
    """The 64-dim pre-classifier embedding -- CnnAttention only (CnnLstm doesn't expose one; add an
    analogous .embed() there if the open-set approach ends up needing it too)."""
    model.eval()
    with torch.no_grad():
        return model.embed(torch.from_numpy(X)).numpy()


# backward-compatible aliases (permutation_test.py / identity_permutation.py call these by name)
def train_cnn_lstm(*args, **kwargs):
    return train_torch_model(CnnLstm, *args, **kwargs)


cnn_lstm_predict_proba = torch_model_predict_proba
