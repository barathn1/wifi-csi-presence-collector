"""Task C, done properly: WhoFi-style embedding + contrastive loss + cosine-similarity verification,
not the closed-set RandomForest proxy in run_stage1.py. Enroll authorized identities as the mean
embedding of their training windows; score every test window by cosine similarity to the nearest
enrolled centroid; tune accept/reject via EER. Evaluated with leave-one-unauthorized-person-out (same
split as the Stage 1 proxy) so a genuinely novel intruder is being rejected, not one seen in training.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.splits import assert_no_group_leakage, leave_one_unauthorized_person_out
from ml.data_pipeline.windowing import load_manifest, load_window
from ml.evaluation.metrics import compute_auroc, compute_eer
from ml.models.transformer_whofi import WhoFiTransformer
from ml.training.losses import in_batch_contrastive_loss
from ml.training.train import log_rows

DEVICE = torch.device("cpu")


class _Rows(torch.utils.data.Dataset):
    """Thin wrapper so leave_one_unauthorized_person_out's plain row-position arrays work with a
    torch DataLoader, independent of CsiWindowDataset's task-mask assumptions."""

    def __init__(self, index: pd.DataFrame, label_to_id: dict):
        self.index = index.reset_index(drop=True)
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        row = self.index.iloc[i]
        amp, phase, _ = load_window(row)
        identity_key = row["person_id"] if row["person_id"] else "none"
        return (torch.from_numpy(np.ascontiguousarray(amp, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(phase, dtype=np.float32)),
                self.label_to_id[identity_key])


def train_embedding_model(model, train_ds, epochs=4, batch_size=64, lr=1e-3):
    model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    for epoch in range(epochs):
        total_loss, n = 0.0, 0
        for amp, phase, label in loader:
            amp, label = amp.to(DEVICE), label.to(DEVICE)
            opt.zero_grad()
            emb = model.embed(amp)
            loss = in_batch_contrastive_loss(emb, label)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n += 1
        print(f"    epoch {epoch}: contrastive loss = {total_loss / max(n, 1):.4f}")
    return model


@torch.no_grad()
def enroll_and_score(model, enroll_ds, test_index: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Enroll = mean embedding per authorized identity from enroll_ds. Score each test window by max
    cosine similarity to any enrolled centroid. Returns (y_true, scores) for EER/AUROC."""
    model.eval()
    enroll_loader = DataLoader(enroll_ds, batch_size=64, shuffle=False, num_workers=0)
    embeddings_by_id: dict[int, list[torch.Tensor]] = {}
    for amp, _, label in enroll_loader:
        emb = model.embed(amp.to(DEVICE))
        for e, l in zip(emb, label.tolist()):
            embeddings_by_id.setdefault(l, []).append(e)
    centroids = torch.stack([torch.stack(v).mean(dim=0) for v in embeddings_by_id.values()])
    centroids = torch.nn.functional.normalize(centroids, p=2, dim=-1)

    test_ds = _Rows(test_index, {pid: 0 for pid in set(test_index["person_id"]) | {"none"}})
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    y_true = (test_index["label"] == "authorized").astype(int).values
    scores = []
    for amp, _, _ in test_loader:
        emb = model.embed(amp.to(DEVICE))
        sim = (emb @ centroids.T).max(dim=-1).values
        scores.append(sim.numpy())
    return y_true, np.concatenate(scores)


def main(epochs: int = 4) -> None:
    index_path = REPO_ROOT / "ml/data_pipeline/cache/window_index_w200_s100.csv"
    window_index = pd.read_csv(index_path)
    n_subcarriers = 186

    rows_out = []
    for held_out, train_idx, test_idx in leave_one_unauthorized_person_out(window_index):
        print(f"=== held out unauthorized person: {held_out} ===")
        train_index = window_index.iloc[train_idx].reset_index(drop=True)
        test_index = window_index.iloc[test_idx].reset_index(drop=True)
        assert_no_group_leakage(window_index, train_idx, test_idx, "session_dir")

        identities = sorted(set(train_index["person_id"]) | {"none"})
        label_to_id = {ident: i for i, ident in enumerate(identities)}
        n_classes = len(identities)

        model = WhoFiTransformer(n_subcarriers, n_classes)
        train_ds = _Rows(train_index, label_to_id)
        train_embedding_model(model, train_ds, epochs=epochs)

        enroll_index = train_index[train_index["label"] == "authorized"].reset_index(drop=True)
        enroll_ds = _Rows(enroll_index, label_to_id)
        y_true, scores = enroll_and_score(model, enroll_ds, test_index)

        eer, thresh = compute_eer(y_true, scores)
        auroc = compute_auroc(y_true, scores)
        print(f"  EER={eer:.4f}  AUROC={auroc:.4f}  n_test={len(y_true)} (authorized={y_true.sum()}, "
              f"unauthorized={len(y_true) - y_true.sum()})")
        rows_out.append({
            "timestamp": datetime.now(timezone.utc).isoformat(), "stage": 2, "task": "taskC_openset_embedding",
            "preprocessing": "raw", "model": "whofi_transformer_embedding",
            "split_type": "leave_one_unauth_person_out", "fold": held_out,
            "eer": eer, "auroc": auroc, "n_train": len(train_idx), "n_test": len(test_idx),
        })

    log_rows(rows_out)
    mean_eer = np.mean([r["eer"] for r in rows_out])
    mean_auroc = np.mean([r["auroc"] for r in rows_out])
    print(f"\nmean EER={mean_eer:.4f}  mean AUROC={mean_auroc:.4f} across {len(rows_out)} held-out people")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=4)
    args = p.parse_args()
    main(epochs=args.epochs)
