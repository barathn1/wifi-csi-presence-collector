"""FewSense-style few-shot cross-day fine-tuning -- see ml/reports/day2_next_steps.md item 10 (the
"more pragmatic" stretch-goal option, vs. full adversarial domain adaptation like CrossRF). Pretrain on
all of Day 1, hold out a handful of whole Day 2 SESSIONS (one per label, so few-shot data never
straddles the eval set) as fine-tuning data, fine-tune briefly on just those, then evaluate zero-shot
vs few-shot on the REMAINING Day 2 sessions.

Reports a k-shot curve (k = number of fine-tuning SESSIONS' windows, not hand-picked individual
windows) rather than a single number, per the research recommendation to report zero-shot AND k-shot
together as the credible result on a 2-day dataset.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.splits import assert_no_group_leakage
from ml.data_pipeline.torch_dataset import CsiWindowDataset
from ml.data_pipeline.windowing import load_manifest
from ml.evaluation.metrics import compute_auroc, compute_eer
from ml.models.transformer_whofi import WhoFiTransformer
from ml.training.train import DEVICE, train_classifier

INDEX_PATH = REPO_ROOT / "ml/data_pipeline/cache/window_index_w200_s100.csv"
TASK = "taskD_auth_vs_nonauth"


@torch.no_grad()
def evaluate(model, ds) -> dict:
    from torch.utils.data import DataLoader
    model.eval()
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    scores, preds, ys = [], [], []
    for amp, phase, y in loader:
        logits = model(amp.to(DEVICE), phase.to(DEVICE))
        proba = torch.softmax(logits, dim=-1)[:, 1]
        scores.append(proba.numpy()); preds.append(logits.argmax(-1).numpy()); ys.append(np.asarray(y))
    scores, preds, ys = np.concatenate(scores), np.concatenate(preds), np.concatenate(ys)
    eer, _ = compute_eer(ys, scores)
    auroc = compute_auroc(ys, scores)
    return {"accuracy": float((preds == ys).mean()), "eer": eer, "auroc": auroc}


def fine_tune(model, ds, epochs: int = 3, lr: float = 1e-4, seed: int = 0):
    torch.manual_seed(seed)
    model = copy.deepcopy(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=min(32, len(ds)), shuffle=True)
    model.train()
    for _ in range(epochs):
        for amp, phase, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(amp.to(DEVICE), phase.to(DEVICE)), y.to(DEVICE))
            loss.backward()
            opt.step()
    return model


def main(epochs_pretrain: int = 5, epochs_finetune: int = 3, seed: int = 0) -> None:
    manifest = load_manifest()
    window_index = pd.read_csv(INDEX_PATH)
    dates = sorted(window_index["date"].unique())
    train_date, test_date = dates[0], dates[1]

    full_ds = CsiWindowDataset(window_index, TASK)
    train_idx = np.flatnonzero((full_ds.index["date"] == train_date).values)
    day2_idx = np.flatnonzero((full_ds.index["date"] == test_date).values)
    assert_no_group_leakage(full_ds.index, train_idx, day2_idx, "session_dir")

    # one Day-2 session per label held out as the few-shot fine-tuning pool, rest is the eval set.
    day2_index = full_ds.index.iloc[day2_idx]
    fewshot_sessions = day2_index.groupby("label")["session_dir"].first().tolist()
    fewshot_mask = day2_index["session_dir"].isin(fewshot_sessions).values
    fewshot_idx = day2_idx[fewshot_mask]
    eval_idx = day2_idx[~fewshot_mask]
    print(f"pretrain (Day1): {len(train_idx)} windows | few-shot pool (Day2, {len(fewshot_sessions)} sessions): "
          f"{len(fewshot_idx)} windows | eval (remaining Day2): {len(eval_idx)} windows")

    train_ds = full_ds.subset_by_index_rows(train_idx)
    fewshot_ds = full_ds.subset_by_index_rows(fewshot_idx)
    eval_ds = full_ds.subset_by_index_rows(eval_idx)

    print("pretraining on Day 1...")
    base_model = WhoFiTransformer(128, len(full_ds.classes))
    train_classifier(base_model, train_ds, eval_ds, epochs=epochs_pretrain, seed=seed)

    zero_shot = evaluate(base_model, eval_ds)
    print(f"zero-shot (no Day2 data at all): {zero_shot}")

    for k_sessions in (1, len(fewshot_sessions)):
        # k=1: only the first few-shot session's windows; k=all: the full held-out pool.
        sessions_k = fewshot_sessions[:k_sessions] if k_sessions < len(fewshot_sessions) else fewshot_sessions
        k_mask = np.isin(full_ds.index.iloc[fewshot_idx]["session_dir"].values, sessions_k)
        k_idx = fewshot_idx[k_mask]
        k_ds = full_ds.subset_by_index_rows(k_idx)
        finetuned = fine_tune(base_model, k_ds, epochs=epochs_finetune, seed=seed)
        result = evaluate(finetuned, eval_ds)
        print(f"few-shot, {len(sessions_k)}/{len(fewshot_sessions)} fine-tune sessions "
              f"({len(k_idx)} windows): {result}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs-pretrain", type=int, default=5)
    p.add_argument("--epochs-finetune", type=int, default=3)
    args = p.parse_args()
    main(epochs_pretrain=args.epochs_pretrain, epochs_finetune=args.epochs_finetune)
