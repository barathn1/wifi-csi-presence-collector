"""Part 5 -- same-person vs different-person distance distributions, split by within-session /
cross-session-same-day / cross-day, with a separability measure (ROC-AUC and KS statistic treating
"is this pair the same person?" as the label and pairwise distance as the score).
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from analysis.walking_ch6.load_data import build
from analysis.walking_ch6.viz_common import IDENTITY_COLOR, PRIMARY_TEXT, style_axes
from analysis.walking_ch6.baselines import restrict_to_two_person

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path(__file__).resolve().parent / "cache"

PER_SESSION_SAMPLE = 40
RNG = np.random.default_rng(0)


def subsample(wi: pd.DataFrame, X: np.ndarray):
    keep = []
    for _, g in wi.groupby("session_dir"):
        idx = g.index.to_numpy()
        if len(idx) > PER_SESSION_SAMPLE:
            idx = RNG.choice(idx, size=PER_SESSION_SAMPLE, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    return wi.loc[keep].reset_index(drop=True), X[keep]


def pair_categories(wi: pd.DataFrame) -> pd.DataFrame:
    n = len(wi)
    identity = wi["identity"].values
    session = wi["session_dir"].values
    date = wi["date"].values

    idx_i, idx_j = np.array(list(combinations(range(n), 2))).T
    same_person = identity[idx_i] == identity[idx_j]
    same_session = session[idx_i] == session[idx_j]
    same_day = date[idx_i] == date[idx_j]

    category = np.where(
        same_person & same_session, "same_person_same_session",
        np.where(same_person & same_day, "same_person_cross_session_same_day",
                 np.where(same_person, "same_person_cross_day",
                          np.where(same_day, "diff_person_same_day", "diff_person_cross_day"))))
    return pd.DataFrame({"i": idx_i, "j": idx_j, "same_person": same_person, "category": category})


def main():
    window_index, X = build()
    wi, X2, _ = restrict_to_two_person(window_index, X)
    wi, X2 = subsample(wi, X2)
    print(f"distance analysis on {len(wi)} windows ({wi['session_dir'].nunique()} sessions)")

    Xs = StandardScaler().fit_transform(X2)
    D = squareform(pdist(Xs, metric="euclidean"))

    pairs = pair_categories(wi)
    pairs["distance"] = D[pairs["i"].to_numpy(), pairs["j"].to_numpy()]
    pairs.to_csv(CACHE_DIR / "pairwise_distances.csv", index=False)

    same = pairs.loc[pairs["same_person"], "distance"]
    diff = pairs.loc[~pairs["same_person"], "distance"]
    auc = roc_auc_score((~pairs["same_person"]).astype(int), pairs["distance"])  # higher dist -> more likely diff-person
    ks_stat, ks_p = ks_2samp(same, diff)
    print(f"\noverall same-person distance: mean={same.mean():.3f} n={len(same)}")
    print(f"overall diff-person distance: mean={diff.mean():.3f} n={len(diff)}")
    print(f"separability: ROC-AUC(distance predicts different-person) = {auc:.3f} "
          f"(0.5 = no separability, 1.0 = perfectly separable)")
    print(f"KS statistic = {ks_stat:.3f}, p = {ks_p:.2e}")

    # breakdown by category granularity
    print("\nper-category distance summary:")
    print(pairs.groupby("category")["distance"].agg(["mean", "std", "count"]).round(3))

    breakdown_rows = []
    for cat_same, cat_diff, level in [
        ("same_person_same_session", "diff_person_same_day", "within-session vs diff-person-same-day"),
        ("same_person_cross_session_same_day", "diff_person_same_day", "cross-session-same-day"),
        ("same_person_cross_day", "diff_person_cross_day", "cross-day"),
    ]:
        s = pairs.loc[pairs["category"] == cat_same, "distance"]
        d = pairs.loc[pairs["category"] == cat_diff, "distance"]
        if len(s) == 0 or len(d) == 0:
            continue
        labels = np.concatenate([np.zeros(len(s)), np.ones(len(d))])
        scores = np.concatenate([s, d])
        a = roc_auc_score(labels, scores)
        breakdown_rows.append({"level": level, "same_mean": s.mean(), "diff_mean": d.mean(),
                                "auc": a, "n_same": len(s), "n_diff": len(d)})
    breakdown = pd.DataFrame(breakdown_rows)
    breakdown.to_csv(CACHE_DIR / "distance_separability_by_level.csv", index=False)
    print("\nseparability by level (same-person distance vs different-person distance, same time-scope):")
    print(breakdown.round(3))

    # plot: same-person vs different-person, overall
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(same, bins=50, density=True, alpha=0.6, color=IDENTITY_COLOR["anjali"], label="same person")
    ax.hist(diff, bins=50, density=True, alpha=0.6, color=IDENTITY_COLOR["barath"], label="different person")
    ax.set_xlabel("pairwise Euclidean distance (standardized features)")
    ax.set_ylabel("density")
    ax.set_title(f"Same-person vs different-person distances (AUC={auc:.3f})", color=PRIMARY_TEXT)
    ax.legend(frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "distance_same_vs_diff_person.png", dpi=150)
    plt.close(fig)

    # plot: broken out by within-session / cross-session-same-day / cross-day
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    pairs_plot = [
        ("same_person_same_session", "diff_person_same_day", "within-session"),
        ("same_person_cross_session_same_day", "diff_person_same_day", "cross-session, same day"),
        ("same_person_cross_day", "diff_person_cross_day", "cross-day"),
    ]
    for ax, (cat_same, cat_diff, title) in zip(axes, pairs_plot):
        s = pairs.loc[pairs["category"] == cat_same, "distance"]
        d = pairs.loc[pairs["category"] == cat_diff, "distance"]
        if len(s) and len(d):
            bins = np.linspace(min(s.min(), d.min()), max(s.max(), d.max()), 40)
            ax.hist(s, bins=bins, density=True, alpha=0.6, color=IDENTITY_COLOR["anjali"], label="same person")
            ax.hist(d, bins=bins, density=True, alpha=0.6, color=IDENTITY_COLOR["barath"], label="different person")
        ax.set_title(title, color=PRIMARY_TEXT)
        ax.set_xlabel("distance")
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("Same- vs different-person distance, by time scope", color=PRIMARY_TEXT)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "distance_by_scope.png", dpi=150)
    plt.close(fig)
    print(f"\nfigures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
