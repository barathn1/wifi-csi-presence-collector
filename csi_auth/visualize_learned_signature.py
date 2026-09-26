"""What did CNN+Attention actually learn to tell Anjali and Barath apart? Three views:

1. Embedding scatter (PCA of the 64-dim pre-classifier vector, every window) -- does the model's
   LEARNED space visually separate them, unlike the raw-feature views in FINDINGS.md that didn't?
2. Centroid signature heatmap -- each person's mean embedding vector, z-scored, as a heatmap, plus
   their difference -- literally "what the model thinks Anjali/Barath typically look like."
3. Attention-weight profile -- for one representative window per person, which moments in the ~2s
   window the self-attention pooling step weighted most heavily (see PIPELINE_EXPLAINER.md's
   architecture section: this is the "which part of the gait cycle mattered" mechanism).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from dataset import build_dataset
from models import CnnAttention, torch_model_embed
from train_final import CHECKPOINT_DIR

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

ANJALI_COLOR, BARATH_COLOR = "#2a78d6", "#eb6834"
BLUE_SEQUENTIAL = ["#f0efec", "#cde2fb", "#6da7ec", "#2a78d6", "#104281"]
DIVERGING = plt.get_cmap("RdBu_r")


def load_model():
    ckpt = torch.load(CHECKPOINT_DIR / "cnn_attention_final.pt", weights_only=False)
    model = CnnAttention(n_subcarriers=ckpt["n_subcarriers"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["sub_mean"], ckpt["sub_std"]


def attention_weights_for_window(model: CnnAttention, x: np.ndarray) -> np.ndarray:
    """Replicates CnnAttention.embed()'s forward pass just far enough to pull out the pooling
    weights themselves, without changing the model class again."""
    with torch.no_grad():
        t = torch.from_numpy(x[None, ...]).transpose(1, 2)
        t = model.conv(t).transpose(1, 2)
        t = t + model.pos_embed[:, :t.shape[1], :]
        attn_out, _ = model.self_attn(t, t, t)
        t = model.norm(t + attn_out)
        weights = torch.softmax(model.pool_attn(t).squeeze(-1), dim=1)
    return weights.squeeze(0).numpy()


def main():
    window_table, _, X_seq = build_dataset()
    mask = window_table["person_id"].isin(["anjali", "barath"]).values
    wt = window_table.loc[mask].reset_index(drop=True)
    Xq = X_seq[mask]

    model, sub_mean, sub_std = load_model()
    seq_norm = ((Xq - sub_mean) / sub_std).astype(np.float32)
    embed = torch_model_embed(model, seq_norm)
    is_anjali = (wt["person_id"] == "anjali").values
    is_barath = (wt["person_id"] == "barath").values

    # --- 1. embedding scatter ---
    pca = PCA(n_components=2, random_state=0)
    proj = pca.fit_transform(embed)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(proj[is_anjali, 0], proj[is_anjali, 1], s=10, alpha=0.4, color=ANJALI_COLOR, label="anjali")
    ax.scatter(proj[is_barath, 0], proj[is_barath, 1], s=10, alpha=0.4, color=BARATH_COLOR, label="barath")
    ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.0f}% var)")
    ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.0f}% var)")
    ax.set_title("CNN+Attention's learned embedding space\n(PCA of the 64-dim pre-classifier vector, every window)")
    ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "learned_embedding_scatter.png", dpi=150)
    plt.close(fig)

    # --- 2. centroid signature heatmap ---
    z = (embed - embed.mean(axis=0)) / (embed.std(axis=0) + 1e-6)
    anjali_sig = z[is_anjali].mean(axis=0)
    barath_sig = z[is_barath].mean(axis=0)
    diff = anjali_sig - barath_sig
    order = np.argsort(-np.abs(diff))  # sort dims by how much they differ, most-distinguishing first

    fig, axes = plt.subplots(3, 1, figsize=(12, 5), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1, 1]})
    vmax = max(np.abs(anjali_sig).max(), np.abs(barath_sig).max())
    for ax, sig, label in [(axes[0], anjali_sig[order], "anjali"), (axes[1], barath_sig[order], "barath")]:
        ax.imshow(sig[None, :], aspect="auto", cmap=DIVERGING, vmin=-vmax, vmax=vmax)
        ax.set_yticks([0]); ax.set_yticklabels([label])
        ax.set_xticks([])
    im = axes[2].imshow(diff[order][None, :], aspect="auto", cmap=DIVERGING,
                         vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    axes[2].set_yticks([0]); axes[2].set_yticklabels(["anjali - barath"])
    axes[2].set_xlabel("embedding dimension, sorted by how much it distinguishes the two (left = most)")
    fig.suptitle("Learned signature: mean embedding per person (z-scored), and their difference",
                 y=1.02)
    fig.colorbar(im, ax=axes, shrink=0.8, label="z-scored activation")
    fig.savefig(FIG_DIR / "learned_signature_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 3. attention-weight profile, one representative window per person ---
    anjali_idx = np.flatnonzero(is_anjali)[0]
    barath_idx = np.flatnonzero(is_barath)[0]
    w_anjali = attention_weights_for_window(model, seq_norm[anjali_idx])
    w_barath = attention_weights_for_window(model, seq_norm[barath_idx])

    fig, ax = plt.subplots(figsize=(9, 4))
    t_axis = np.arange(len(w_anjali))
    ax.plot(t_axis, w_anjali, color=ANJALI_COLOR, linewidth=1.8, marker="o", markersize=3, label="anjali (example window)")
    ax.plot(t_axis, w_barath, color=BARATH_COLOR, linewidth=1.8, marker="o", markersize=3, label="barath (example window)")
    ax.set_xlabel("timestep within the ~2s window (after conv downsampling, 50 steps)")
    ax.set_ylabel("attention-pooling weight")
    ax.set_title("Which moments in the window the model weighted most heavily")
    ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "learned_attention_profile.png", dpi=150)
    plt.close(fig)

    print(f"wrote figures to {FIG_DIR}")
    print(f"top 5 most-distinguishing embedding dims: {order[:5].tolist()}, "
          f"|anjali-barath| = {np.abs(diff[order[:5]]).round(2).tolist()}")


if __name__ == "__main__":
    main()
