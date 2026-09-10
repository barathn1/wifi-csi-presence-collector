"""The direct answer to "what's noise and what's not": a session-level permutation test builds an
empirical noise floor -- shuffle which sessions are labeled authorized/unauthorized thousands of times
(keeping group sizes fixed), recompute Cohen's d each time, and see what effect sizes pure chance
produces. Any subcarrier whose REAL effect size sticks out above that noise band is a genuine signal;
everything inside the band is statistically indistinguishable from a random relabeling of the data.

Why permute SESSIONS, not windows: windows overlap 50% and many windows come from the same session, so
they are not independent samples -- a naive per-window permutation test would treat 60,000 correlated
windows as 60,000 independent trials and report near-certain "significance" everywhere, which would be
fake precision. Shuffling at the session level (8 authorized sessions, 18 unauthorized sessions) respects
the real unit of independence and gives an honest, appropriately conservative test -- same principle as
every session-disjoint split used elsewhere in this codebase.

Produces two companion figures per class-pair/channel:
  - signal_vs_noise_*.png: per-subcarrier Cohen's d bar chart with the noise-floor band overlaid,
    significant subcarriers (FDR<0.05, Benjamini-Hochberg) shown solid, everything else faded.
  - distributions_*.png: actual per-window value distributions (violin plots) for the top significant
    subcarriers -- so you can see the separation directly, not just read a summary statistic.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.features import feature_slices
from ml.visualization.palette import STATIC_FIG_DIR, CATEGORICAL, DIVERGING, GRIDLINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY



def _session_stats(window_index_subset: pd.DataFrame, values: np.ndarray) -> dict[str, tuple]:
    """Per-session (sum, sumsq, n) per subcarrier -- lets permutations recombine group stats by simple
    arithmetic instead of re-concatenating and re-scanning every window on every one of the 1000 draws."""
    stats = {}
    for session_dir, idx in window_index_subset.groupby("session_dir").groups.items():
        idx = np.asarray(idx)
        v = values[idx]
        stats[session_dir] = (v.sum(axis=0), (v ** 2).sum(axis=0), len(idx))
    return stats


def _group_mean_var(session_stats: dict, sessions: set) -> tuple[np.ndarray, np.ndarray, int]:
    total_sum = total_sumsq = 0.0
    total_n = 0
    for s in sessions:
        s_sum, s_sumsq, s_n = session_stats[s]
        total_sum = total_sum + s_sum
        total_sumsq = total_sumsq + s_sumsq
        total_n += s_n
    mean = total_sum / total_n
    var = np.maximum(total_sumsq / total_n - mean ** 2, 0.0)
    return mean, var, total_n


def _cohens_d(mean_a, var_a, mean_b, var_b) -> np.ndarray:
    pooled_std = np.sqrt((var_a + var_b) / 2)
    return (mean_a - mean_b) / np.maximum(pooled_std, 1e-6)


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    passed = ranked <= thresholds
    significant = np.zeros(n, dtype=bool)
    if passed.any():
        max_idx = int(np.max(np.flatnonzero(passed)))
        significant[order[: max_idx + 1]] = True
    return significant


def session_permutation_test(
    window_index: pd.DataFrame, X: np.ndarray, channel: str, group_col: str, group_a: str, group_b: str,
    n_perm: int = 1000, seed: int = 0, alpha: float = 0.05,
) -> dict:
    slices = feature_slices()
    mean_cols = X[:, slices[f"{channel}_mean"]]

    mask = window_index[group_col].isin([group_a, group_b]).values
    sub_index = window_index[mask].reset_index(drop=True)
    sub_values = mean_cols[mask]

    session_stats = _session_stats(sub_index, sub_values)
    session_true_group = sub_index.groupby("session_dir")[group_col].first()
    sessions = session_true_group.index.values
    true_a_sessions = set(sessions[session_true_group.values == group_a])
    n_a_sessions = len(true_a_sessions)

    def d_for(a_sessions: set) -> np.ndarray:
        b_sessions = set(sessions) - a_sessions
        mean_a, var_a, _ = _group_mean_var(session_stats, a_sessions)
        mean_b, var_b, _ = _group_mean_var(session_stats, b_sessions)
        return _cohens_d(mean_a, var_a, mean_b, var_b)

    real_d = d_for(true_a_sessions)

    rng = np.random.default_rng(seed)
    n_sub = mean_cols.shape[1]
    null_ds = np.empty((n_perm, n_sub))
    for i in range(n_perm):
        perm = rng.permutation(sessions)
        null_ds[i] = d_for(set(perm[:n_a_sessions]))

    p_values = (np.sum(np.abs(null_ds) >= np.abs(real_d), axis=0) + 1) / (n_perm + 1)
    significant = benjamini_hochberg(p_values, alpha=alpha)

    return {
        "real_d": real_d, "null_lo": np.percentile(null_ds, 2.5, axis=0),
        "null_hi": np.percentile(null_ds, 97.5, axis=0), "p_values": p_values,
        "significant": significant, "n_sessions_a": n_a_sessions, "n_sessions_b": len(sessions) - n_a_sessions,
    }


def plot_signal_vs_noise(result: dict, channel: str, group_a: str, group_b: str, out_name: str) -> None:
    real_d, null_lo, null_hi, significant = result["real_d"], result["null_lo"], result["null_hi"], result["significant"]
    n_sub = len(real_d)
    x = np.arange(n_sub)
    n_sig = int(significant.sum())

    fig, ax = plt.subplots(figsize=(13, 4.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.fill_between(x, null_lo, null_hi, color=INK_MUTED, alpha=0.25, linewidth=0,
                     label="noise floor (95% of shuffled-label outcomes)")

    for xi, d, sig in zip(x, real_d, significant):
        color = DIVERGING["pos"] if d >= 0 else DIVERGING["neg"]
        ax.bar(xi, d, width=1.0, color=color, alpha=1.0 if sig else 0.22, linewidth=0)

    ax.axhline(0, color=INK_MUTED, linewidth=0.8)
    ax.set_xlabel("subcarrier index", color=INK_SECONDARY)
    ax.set_ylabel(f"Cohen's d ({group_a} - {group_b})", color=INK_SECONDARY)
    ax.set_title(
        f"{channel}: {group_a} vs {group_b} -- {n_sig}/{n_sub} subcarriers are real signal "
        f"(FDR<0.05, {result['n_sessions_a']} vs {result['n_sessions_b']} sessions permuted)",
        color=INK_PRIMARY, fontsize=12, loc="left",
    )
    ax.tick_params(colors=INK_MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRIDLINE)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="upper right")

    STATIC_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(STATIC_FIG_DIR / out_name, dpi=150)
    plt.close(fig)
    print(f"wrote {STATIC_FIG_DIR / out_name}  ({n_sig}/{n_sub} significant)")


def plot_top_distributions(
    window_index: pd.DataFrame, X: np.ndarray, channel: str, group_col: str, group_a: str, group_b: str,
    result: dict, top_k: int, out_name: str,
) -> None:
    slices = feature_slices()
    mean_cols = X[:, slices[f"{channel}_mean"]]
    real_d, significant, p_values = result["real_d"], result["significant"], result["p_values"]

    sig_idx = np.flatnonzero(significant)
    ranking_pool = sig_idx if len(sig_idx) > 0 else np.arange(len(real_d))
    top = ranking_pool[np.argsort(-np.abs(real_d[ranking_pool]))][:top_k]
    label = "top significant" if len(sig_idx) > 0 else "top by |d| (NONE were significant -- shown for reference only)"

    mask = window_index[group_col].isin([group_a, group_b]).values
    sub_index = window_index[mask].reset_index(drop=True)
    sub_values = mean_cols[mask]
    groups = sub_index[group_col].values

    fig, axes = plt.subplots(1, len(top), figsize=(3.4 * len(top), 4.2), facecolor="#fcfcfb")
    axes = np.atleast_1d(axes)
    color_a, color_b = CATEGORICAL.get(group_a, "#2a78d6"), CATEGORICAL.get(group_b, "#eb6834")

    for ax, sc in zip(axes, top):
        ax.set_facecolor("#fcfcfb")
        data_a = sub_values[groups == group_a, sc]
        data_b = sub_values[groups == group_b, sc]
        parts = ax.violinplot([data_a, data_b], showmedians=True, showextrema=False)
        for body, color in zip(parts["bodies"], (color_a, color_b)):
            body.set_facecolor(color)
            body.set_alpha(0.6)
        parts["cmedians"].set_color(INK_PRIMARY)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([group_a, group_b], color=INK_SECONDARY)
        ax.set_title(f"subcarrier {sc}\nd={real_d[sc]:.2f}, p={p_values[sc]:.4f}", fontsize=10, color=INK_PRIMARY)
        ax.tick_params(colors=INK_MUTED)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRIDLINE)

    fig.suptitle(f"{channel}: {group_a} vs {group_b} -- {label}", color=INK_PRIMARY, fontsize=12)
    STATIC_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(STATIC_FIG_DIR / out_name, dpi=150)
    plt.close(fig)
    print(f"wrote {STATIC_FIG_DIR / out_name}")


def main(n_perm: int = 1000) -> None:
    cache_dir = REPO_ROOT / "ml/data_pipeline/cache"
    window_index = pd.read_csv(cache_dir / "window_index_w200_s100.csv")
    X = np.load(cache_dir / "features_raw_window_index_w200_s100.npy")

    pairs = [
        ("label", "authorized", "unauthorized"),
        ("label", "authorized", "none"),
        ("label", "unauthorized", "none"),
    ]
    for group_col, group_a, group_b in pairs:
        for channel in ("amp", "phase"):
            print(f"\n=== {channel}: {group_a} vs {group_b} ({n_perm} session-level permutations) ===")
            result = session_permutation_test(window_index, X, channel, group_col, group_a, group_b, n_perm=n_perm)
            tag = f"{channel}_{group_a}_vs_{group_b}"
            plot_signal_vs_noise(result, channel, group_a, group_b, f"signal_vs_noise_{tag}.png")
            plot_top_distributions(window_index, X, channel, group_col, group_a, group_b, result, top_k=4,
                                    out_name=f"distributions_{tag}.png")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-perm", type=int, default=1000)
    args = p.parse_args()
    main(n_perm=args.n_perm)
