"""One-off extraction: parses the ALREADY-COMPLETED stages' .log files for the per-session breakdown
text that was printed (via format_session_breakdown) but never saved as a clean CSV -- see
ml_2/data/metrics.py::save_session_breakdown_csv, added only after these stages had already run.

This is PARTIAL data: format_session_breakdown truncates to the worst 12 sessions per fold (sorted by
window_accuracy ascending), not the full per-session table. Every stage from extra_classifiers onward
gets the REAL, complete session_breakdown.csv automatically; this script only backfills what's
recoverable for the stages that ran before that fix existed.

    python3 -m ml_2.evaluation.extract_partial_session_breakdown
"""
from __future__ import annotations

import re

import pandas as pd

from ml_2.data.decode import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "ml_2/evaluation/results"
LOG_DIR = RESULTS_DIR / "logs"
OUT_PATH = RESULTS_DIR / "session_breakdown_PARTIAL_backfill.csv"

# (log filename, function to derive (model, split_type) from a "===...===" header line)
HEADER_RE = re.compile(r"^=== (.+?) ===\s*$")
HELD_OUT_RE = re.compile(r"held out (?:day )?'([^']+)':")
SESSION_LINE_RE = re.compile(
    r"^\s+(\S+)\s+y_true=(\d)\s+majority_pred=(\d)\s+window_acc=([\d.]+)\s+mean_proba=(-?[\d.]+)\s+n_windows=(\d+)"
)


def _model_split_from_header(header: str) -> tuple[str, str] | None:
    """Header text varies per script; this maps every header format actually produced so far to
    (model, split_type). Returns None for headers with no session-level breakdown (e.g. closed-set,
    which was never given session-level reporting)."""
    h = header.lower()
    if "closed-set" in h or "session-disjoint" in h:
        return None  # closed-set folds never got session_level_metrics wired in

    if "open-set" in h:
        split_type = "open_set_loo_stranger"
    elif "cross-day" in h:
        split_type = "cross_day_loo"
    else:
        return None

    m = re.match(r"svm\[(\w+)\]", h)
    if m:
        return m.group(1), split_type
    if h.startswith("gbm"):
        return "gbm_xgboost", split_type
    if h.startswith("cusum"):
        return "cusum_detector", split_type
    if h.startswith("prototypical"):
        return "prototypical_embedding", split_type
    if h.startswith("arcface"):
        return "arcface_embedding", split_type
    m = re.match(r"(\w+) \(gpu\)", h)
    if m:
        return m.group(1), split_type
    return None


def extract_log(log_path) -> list[dict]:
    rows = []
    model, split_type, held_out = None, None, None
    with open(log_path) as f:
        for line in f:
            header_match = HEADER_RE.match(line)
            if header_match:
                parsed = _model_split_from_header(header_match.group(1))
                model, split_type = parsed if parsed else (None, None)
                held_out = None
                continue
            if model is None:
                continue
            held_out_match = HELD_OUT_RE.search(line)
            if held_out_match:
                held_out = held_out_match.group(1)
                continue
            session_match = SESSION_LINE_RE.match(line)
            if session_match and held_out is not None:
                session_dir, y_true, majority_pred, window_acc, mean_proba, n_windows = session_match.groups()
                rows.append({
                    "model": model, "split_type": split_type, "held_out": held_out,
                    "session_dir": session_dir, "y_true": int(y_true), "majority_pred": int(majority_pred),
                    "window_accuracy": float(window_acc), "mean_proba": float(mean_proba),
                    "n_windows": int(n_windows), "source_log": log_path.name,
                })
    return rows


def main() -> None:
    all_rows = []
    for log_path in sorted(LOG_DIR.glob("*.log")):
        rows = extract_log(log_path)
        print(f"  {log_path.name}: {len(rows)} per-session rows recovered (worst-12-per-fold only)")
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n{len(df)} total rows -> {OUT_PATH}")
    print(f"models covered: {sorted(df['model'].unique()) if len(df) else '(none)'}")
    print("NOTE: this is PARTIAL -- only the worst 12 sessions per fold were ever printed to these "
          "logs. Stages run from now on (extra_classifiers onward) get the full per-session table "
          "automatically in session_breakdown.csv instead.")


if __name__ == "__main__":
    main()
