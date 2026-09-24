"""EER/AUROC for open-set/verification-style scoring, shared by every model in ml_2."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """y_true: 1=genuine/authorized, 0=impostor. scores: higher = more likely genuine."""
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2), float(thresholds[idx])


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def session_level_metrics(y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray,
                           session_dirs: np.ndarray, dates: np.ndarray | None = None) -> dict:
    """A pooled/window-level AUROC over this dataset is measured on windows that overlap 50% within
    a session -- highly correlated, near-duplicate samples, not independent draws. A fold's window-
    level number can look strong (or weak) purely because a handful of sessions happen to contribute
    thousands of windows each, not because the model actually separates the classes on genuinely
    different sessions/days. This collapses to ONE decision per session (mean probability + majority
    vote across that session's windows) and recomputes accuracy/AUROC at that level -- the number to
    trust over the pooled one. Also breaks down by date for the same reason at a coarser grain.
    """
    df = pd.DataFrame({"session_dir": session_dirs, "y_true": y_true, "proba": proba, "pred": pred})
    if dates is not None:
        df["date"] = dates

    per_session = df.groupby("session_dir").agg(
        y_true=("y_true", "first"), mean_proba=("proba", "mean"),
        n_windows=("pred", "size"), window_accuracy=("pred", lambda s: float((s == df.loc[s.index, "y_true"]).mean())),
    )
    per_session["majority_pred"] = (df.groupby("session_dir")["pred"].mean() >= 0.5).astype(int)
    per_session = per_session.reset_index()

    session_acc = float((per_session["majority_pred"] == per_session["y_true"]).mean())
    session_auroc = compute_auroc(per_session["y_true"].values, per_session["mean_proba"].values)

    out = {"session_accuracy": session_acc, "session_auroc": session_auroc, "n_sessions": len(per_session),
           "per_session_table": per_session}

    if dates is not None:
        per_day = df.groupby("date").agg(
            n_windows=("pred", "size"), n_sessions=("session_dir", "nunique"),
            window_accuracy=("pred", lambda s: float((s == df.loc[s.index, "y_true"]).mean())),
        ).reset_index()
        out["per_day_table"] = per_day
    return out


def format_session_breakdown(session_metrics: dict, max_rows: int = 12) -> str:
    """Human-readable multi-line summary: session-level acc/AUROC first (the trustworthy number),
    then the per-session table (worst sessions first, since those are where a pooled number would
    hide a real weakness), truncated to `max_rows` sessions to keep logs readable."""
    lines = [
        f"    SESSION-LEVEL (not window-level): acc={session_metrics['session_accuracy']:.3f} "
        f"auroc={session_metrics['session_auroc']:.3f} (n_sessions={session_metrics['n_sessions']})",
    ]
    table = session_metrics["per_session_table"].sort_values("window_accuracy")
    for _, row in table.head(max_rows).iterrows():
        flag = " <-- window_accuracy far from majority vote" \
            if abs(row["window_accuracy"] - float(row["majority_pred"] == row["y_true"])) > 0.3 else ""
        lines.append(f"      {row['session_dir']:45s} y_true={int(row['y_true'])} "
                     f"majority_pred={int(row['majority_pred'])} window_acc={row['window_accuracy']:.3f} "
                     f"mean_proba={row['mean_proba']:.3f} n_windows={int(row['n_windows'])}{flag}")
    if len(table) > max_rows:
        lines.append(f"      ... ({len(table) - max_rows} more sessions, see full CSV)")
    return "\n".join(lines)


def save_session_breakdown_csv(per_session_table: pd.DataFrame, csv_path: Path, model: str,
                                split_type: str, held_out) -> None:
    """Appends the FULL per-session table (every session, not the printed top-12) to one shared CSV
    -- what format_session_breakdown's own text was already claiming existed ("see full CSV") before
    this function was written. One row per (fold, session): model/split_type/held_out context columns
    + session_dir/y_true/majority_pred/window_accuracy/mean_proba/n_windows."""
    table = per_session_table.copy()
    table.insert(0, "model", model)
    table.insert(1, "split_type", split_type)
    table.insert(2, "held_out", str(held_out))
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
