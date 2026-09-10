"""PyTorch Dataset over the window index -- reads one window's slice on demand via windowing.load_window,
same memory-safe design as the classical-baseline feature pipeline (never materializes the full
overlapping-window dataset in RAM).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ml.data_pipeline.tasks import TASKS
from ml.data_pipeline.windowing import load_window


class CsiWindowDataset(Dataset):
    def __init__(self, window_index: pd.DataFrame, task_name: str, calibration=None, classes: list | None = None):
        mask, y = TASKS[task_name](window_index)
        self.index = window_index[mask].reset_index(drop=True)
        self.y = np.asarray(y)
        self.calibration = calibration  # optional callable (amp, phase, row) -> (amp, phase)
        self.classes = classes if classes is not None else sorted(set(self.y.tolist()))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        row = self.index.iloc[i]
        amp, phase, rssi = load_window(row)
        if self.calibration is not None:
            amp, phase = self.calibration(amp, phase, row)
        amp_t = torch.from_numpy(np.ascontiguousarray(amp, dtype=np.float32))
        phase_t = torch.from_numpy(np.ascontiguousarray(phase, dtype=np.float32))
        label_idx = self.class_to_idx[self.y[i]]
        return amp_t, phase_t, label_idx

    def subset_by_index_rows(self, row_positions: np.ndarray) -> "CsiWindowDataset":
        """Build a dataset restricted to given row positions in self.index -- for CV folds."""
        sub = object.__new__(CsiWindowDataset)
        sub.index = self.index.iloc[row_positions].reset_index(drop=True)
        sub.y = self.y[row_positions]
        sub.calibration = self.calibration
        sub.classes = self.classes
        sub.class_to_idx = self.class_to_idx
        return sub
