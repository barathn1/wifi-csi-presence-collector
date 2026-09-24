"""Prototypical-network open-set head (CAUTION-style, IEEE IoT J. 2022): per-identity centroids in
embedding space + a distance-ratio intruder threshold, calibrated WITHOUT any stranger data (using
held-out known-identity windows as the pseudo-unknown set) -- instead of a softmax classifier.

This fills the gap ml/data_pipeline/tasks.py's own taskF_identity_or_nonauth docstring names explicitly:
"this is still a closed-set softmax classifier ... not expected to fully solve open-set generalization
... see the embedding/centroid approach for that." This IS that embedding/centroid approach.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_centroids(embeddings: torch.Tensor, labels: torch.Tensor, classes: list[int]) -> torch.Tensor:
    return torch.stack([embeddings[labels == c].mean(dim=0) for c in classes])


def nearest_centroid_distance_ratio(embeddings: torch.Tensor, centroids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Euclidean distance from each embedding to every centroid -> (nearest_class_idx, ratio =
    d_nearest / d_second_nearest). Ratio near 0 = confidently one class; ratio near 1 = about as close
    to a second class as the nearest one, i.e. exactly the ambiguous case an intruder threshold should
    catch (see CAUTION eq. 4)."""
    dists = torch.cdist(embeddings, centroids)  # (N, n_classes)
    sorted_d, sorted_idx = dists.sort(dim=-1)
    if sorted_d.shape[1] > 1:
        d1, d2 = sorted_d[:, 0], sorted_d[:, 1]
    else:
        d1, d2 = sorted_d[:, 0], sorted_d[:, 0]
    ratio = d1 / (d2 + 1e-6)
    return sorted_idx[:, 0], ratio


def prototypical_loss(embeddings: torch.Tensor, labels: torch.Tensor, centroids: torch.Tensor,
                       classes: list[int]) -> torch.Tensor:
    """Cross-entropy over negative squared distances to centroids (Snell et al. 2017 / CAUTION eq. 2-3)."""
    dists = torch.cdist(embeddings, centroids) ** 2
    class_to_col = {c: i for i, c in enumerate(classes)}
    target = torch.tensor([class_to_col[int(l)] for l in labels], device=embeddings.device)
    return F.cross_entropy(-dists, target)


def episodic_split(labels: torch.Tensor, classes: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a batch into support (first half of each class's indices) and query (the rest) --
    CAUTION's own train/test split of the support set (section III-B). Falls back to reusing every
    index as both support and query for a class with only 1 example in this batch (degenerate small-
    batch case; harmless since the centroid is then just that one embedding either way)."""
    support_idx, query_idx = [], []
    for c in classes:
        idx = torch.nonzero(labels == c, as_tuple=True)[0]
        if len(idx) == 0:
            continue
        if len(idx) == 1:
            support_idx.append(idx)
            query_idx.append(idx)
            continue
        half = len(idx) // 2
        perm = idx[torch.randperm(len(idx))]
        support_idx.append(perm[:half])
        query_idx.append(perm[half:])
    return torch.cat(support_idx), torch.cat(query_idx)
