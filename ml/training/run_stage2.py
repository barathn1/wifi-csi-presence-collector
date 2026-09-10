"""Stage 2: the transformer/attention model zoo (plus CNN/LSTM for completeness) x tasks 0/A/B, each a
single session-disjoint 80/20 split (see train.py's docstring for why not full CV here -- CPU-only
budget). This is the priority sweep -- dual-branch late-fusion (paper replica) vs the two "own idea"
cross-attention architectures vs WhoFi's single-stream embedding model vs CNN/LSTM floors.

Task C (open-set verification) needs a different eval path (embedding + EER via leave-one-unauthorized-
out, not plain accuracy) -- that's run separately, see run_stage2_openset.py.
"""
from __future__ import annotations

from datetime import datetime, timezone

from ml.models.cnn1d import CNN1DDualBranch
from ml.models.lstm import BiLSTMDualBranch
from ml.models.transformer_calib_context import CalibrationContextTransformer
from ml.models.transformer_crossattn import CrossAttentionTransformer
from ml.models.transformer_dualbranch import DualBranchTransformer
from ml.models.transformer_whofi import WhoFiTransformer
from ml.training.train import log_rows, run_model_on_task

MODEL_FACTORIES = {
    "cnn1d": lambda n_sub, n_cls: CNN1DDualBranch(n_sub, n_cls),
    "bilstm": lambda n_sub, n_cls: BiLSTMDualBranch(n_sub, n_cls),
    "whofi_transformer": lambda n_sub, n_cls: WhoFiTransformer(n_sub, n_cls),
    "dualbranch_transformer": lambda n_sub, n_cls: DualBranchTransformer(n_sub, n_cls),
    "crossattn_transformer": lambda n_sub, n_cls: CrossAttentionTransformer(n_sub, n_cls),
    "calib_context_transformer": lambda n_sub, n_cls: CalibrationContextTransformer(n_sub, n_cls),
}

TASKS = ["task0_presence", "taskA_threeway", "taskB_identity"]


def _needs_calibration_context(model_name: str) -> bool:
    return model_name == "calib_context_transformer"


def main(epochs: int = 6) -> None:
    from ml.data_pipeline.calibration import compute_day_baseline
    from ml.data_pipeline.windowing import load_manifest

    manifest = load_manifest()
    date = manifest["session_dir"].iloc[0].split("/")[1]
    baseline = compute_day_baseline(manifest, date)

    all_rows = []
    for task_name in TASKS:
        for model_name, factory in MODEL_FACTORIES.items():
            print(f"=== {model_name} | {task_name} ===", flush=True)
            try:
                if _needs_calibration_context(model_name):
                    def wrapped_factory(n_sub, n_cls, factory=factory):
                        model = factory(n_sub, n_cls)
                        model.set_calibration_context(baseline.amp_mean, baseline.amp_std,
                                                       baseline.phase_mean, baseline.phase_std)
                        return model
                    result = run_model_on_task(wrapped_factory, model_name, task_name, epochs=epochs)
                else:
                    result = run_model_on_task(factory, model_name, task_name, epochs=epochs)
            except Exception as e:  # noqa: BLE001 -- log and keep sweeping the rest of the zoo
                print(f"  FAILED: {e}")
                continue

            print(f"  accuracy={result['accuracy']:.4f} (n_train={result['n_train']}, n_test={result['n_test']})")
            all_rows.append({
                "timestamp": datetime.now(timezone.utc).isoformat(), "stage": 2, "task": task_name,
                "preprocessing": "calibA_context" if _needs_calibration_context(model_name) else "raw",
                "model": model_name, "split_type": "session_disjoint_single_split", "fold": 0,
                "accuracy": result["accuracy"], "n_train": result["n_train"], "n_test": result["n_test"],
            })

    log_rows(all_rows)
    print(f"\nlogged {len(all_rows)} rows")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=6)
    args = p.parse_args()
    main(epochs=args.epochs)
