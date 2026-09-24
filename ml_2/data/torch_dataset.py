"""PyTorch Dataset over the window index, self-contained (no ml.data_pipeline import)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ml_2.data.tasks import TASKS
from ml_2.data.windowing import load_window


class CsiWindowDataset(Dataset):
    def __init__(self, window_index: pd.DataFrame, task_name: str, calibration=None, classes: list | None = None):
        mask, y = TASKS[task_name](window_index)
        self.index = window_index[mask].reset_index(drop=True)
        self.y = np.asarray(y)
        self.calibration = calibration
        self.classes = classes if classes is not None else sorted(set(self.y.tolist()))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        row = self.index.iloc[i]
        amp, phase, _ = load_window(row)
        if self.calibration is not None:
            amp, phase = self.calibration(amp, phase, row)
        amp_t = torch.from_numpy(np.ascontiguousarray(amp, dtype=np.float32))
        phase_t = torch.from_numpy(np.ascontiguousarray(phase, dtype=np.float32))
        return amp_t, phase_t, self.class_to_idx[self.y[i]]

    def subset_by_index_rows(self, row_positions: np.ndarray) -> "CsiWindowDataset":
        sub = object.__new__(CsiWindowDataset)
        sub.index = self.index.iloc[row_positions].reset_index(drop=True)
        sub.y = self.y[row_positions]
        sub.calibration = self.calibration
        sub.classes = self.classes
        sub.class_to_idx = self.class_to_idx
        return sub
