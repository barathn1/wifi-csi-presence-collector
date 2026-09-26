# Final deployable checkpoints

```json
{
  "trained_on": "all 6 channel-6 walking days, Anjali vs Barath, pooled (no held-out day)",
  "n_windows": 6600,
  "n_sessions": 44,
  "n_days": 6,
  "dates": [
    "2026-09-15",
    "2026-09-16",
    "2026-09-17",
    "2026-09-21",
    "2026-09-22",
    "2026-09-24"
  ],
  "label_meaning": {
    "0": "anjali",
    "1": "barath"
  },
  "warning": "training-set/pooled-set accuracy above is NOT a cross-day generalization estimate -- see csi_auth/README.md and identity_permutation.py output for the honest leave-one-day-out numbers (82-90% session-level, confirmed above a permutation chance floor) that justify shipping these weights at all."
}
```

Load with:

```python
import torch, joblib
svm = joblib.load('svm_final.joblib')  # scaler+SVM pipeline, call .predict_proba(X_stats)
ckpt = torch.load('cnn_attention_final.pt')  # or cnn_bilstm_final.pt
# reconstruct with models.CnnAttention(n_subcarriers=ckpt['n_subcarriers']), load_state_dict(ckpt['state_dict']), normalize inputs with ckpt['sub_mean']/['sub_std']
```

## `presence_final.joblib` (`train_presence.py`)

Occupied-vs-empty-room gate used by `live_inference.py` to decide WHEN to start the identity clock --
trained on all 6 days pooled (any walking session, either person or a stranger, vs. `label == "none"`
empty-room sessions), same "pooled-set number, not a cross-day estimate" caveat (0.995 pooled;
see `analysis/*_ch6/environment_control.py` for the honest leave-one-day-out presence numbers,
77-96%, that justify shipping this). Load with `joblib.load('presence_final.joblib')`, call
`.predict_proba(X_stats)[:, 1]` -- same 874-dim handcrafted feature vector as the identity SVM.
**Do not trust a single window's presence vote live** -- see `DEPLOYMENT.md`'s debounce discussion.
