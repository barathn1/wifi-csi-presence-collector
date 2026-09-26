"""Which of the 128 decoded subcarrier bins are real signal vs guard-band/DC/pilot nulls this specific
ESP32-S3 config (lltf_en+htltf_en+ltf_merge_en -> 128 merged bins) reports as ~zero amplitude
REGARDLESS of occupancy or person -- verified empirically across 6 sessions spanning every collection
date, both receiver boards, authorized/unauthorized/empty-room alike: mean amplitude < 1.0 at exactly
the same 19 indices every time. Feeding these into features/the CNN as if they were signal only adds
dead-weight dimensions.
"""
from __future__ import annotations

import numpy as np

NULL_SUBCARRIERS = [0] + list(range(27, 38)) + list(range(93, 100))  # 19 of 128, verified empirically
N_SUBCARRIERS_RAW = 128


def keep_mask() -> np.ndarray:
    mask = np.ones(N_SUBCARRIERS_RAW, dtype=bool)
    mask[NULL_SUBCARRIERS] = False
    return mask
