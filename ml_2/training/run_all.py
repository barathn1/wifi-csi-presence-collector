"""Fires every ml_2 training script ONE AT A TIME, in sequence -- each model trains AND is evaluated
to completion before the next one starts automatically. Runs each stage as its own subprocess (so a
crash in one model doesn't take down the rest of the run) with live output going to
ml_2/evaluation/results/logs/<model>.log, then runs the consolidated backtest at the end.

    python3 -m ml_2.training.run_all
"""
from __future__ import annotations

import subprocess
import sys
import time

from ml_2.data.decode import REPO_ROOT

LOG_DIR = REPO_ROOT / "ml_2/evaluation/results/logs"

STAGES: list[tuple[str, list[str]]] = [
    ("ml_2.training.train_svm_gait", []),
    ("ml_2.training.train_gbm", []),
    ("ml_2.training.train_cusum", []),
    ("ml_2.training.train_prototypical", ["--epochs", "8"]),
    ("ml_2.training.train_arcface", ["--epochs", "8"]),
    ("ml_2.training.train_transformer_gpu", ["--epochs", "6"]),
    ("ml_2.training.train_extra_classifiers_gpu", ["--epochs", "6"]),
    ("ml_2.training.train_receiver_fusion_gpu", ["--epochs", "6"]),
    ("ml_2.training.train_contrastive", ["--pretrain-epochs", "10", "--probe-epochs", "10"]),
    ("ml_2.training.train_maml", ["--meta-epochs", "10"]),
    ("ml_2.training.train_generative_hard_negative", []),
]


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for module, args in STAGES:
        log_path = LOG_DIR / f"{module.rsplit('.', 1)[-1]}.log"
        print(f"\n{'#' * 100}\n# STARTING: {module} {' '.join(args)}\n# live output -> {log_path}\n{'#' * 100}")
        t0 = time.time()
        with open(log_path, "w") as f:
            result = subprocess.run([sys.executable, "-u", "-m", module, *args], cwd=REPO_ROOT,
                                     stdout=f, stderr=subprocess.STDOUT)
        elapsed = time.time() - t0
        status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        print(f"# {module}: {status} in {elapsed:.1f}s")
        if result.returncode != 0:
            failures.append(module)
            print(f"# last 20 lines of {log_path}:")
            print("".join(log_path.read_text().splitlines(keepends=True)[-20:]))
        print(f"# auto-advancing to next stage...")

    print(f"\n{'#' * 100}\n# BACKTEST (consolidated comparison)\n{'#' * 100}")
    from ml_2.evaluation import backtest_all
    try:
        backtest_all.main()
    except Exception as exc:
        print(f"!!! BACKTEST FAILED: {exc} !!!")
        failures.append("backtest_all")

    print(f"\n{'=' * 100}")
    print(f"DONE WITH FAILURES: {failures}" if failures else "DONE. Every stage completed.")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
