# ml_2 training run -- current status and where everything lives

## Current status: PAUSED (not stopped, not lost)

The training pipeline (`ml_2/training/run_all.py`, PID 141562, and its current child stage) was
paused with `SIGSTOP` on request. This freezes every process (the main `run_all` orchestrator, the
currently-running stage's main process, and all its DataLoader worker processes) exactly where they
were -- no progress lost, no state discarded, unlike killing it. GPU/CPU usage drops to zero while
paused since the OS scheduler simply never runs a stopped process.

**To resume exactly where it left off:**
```bash
pkill -CONT -f "ml_2.training.run_all"
pkill -CONT -f "ml_2.training.train_receiver_fusion_gpu"   # whichever stage was running -- check
                                                              # RUN_STATUS.md's "which stage" section
                                                              # below, or `ps aux | grep ml_2.training`
```
(Careful if running this from an interactive shell: `pkill -f` matches against the FULL command
line of every process, including the shell command doing the matching itself if that command line
also contains the search string -- this bit me once already while pausing. Safer to check PIDs with
`ps aux | grep -E "receiver_fusion|run_all"` first, then `kill -CONT <pid> <pid> ...` explicitly.)

**Which stage was paused**: `train_receiver_fusion_gpu` (3-receiver fusion), mid-way through its
first open-set fold batch (9 folds training concurrently across both GPUs, none had finished yet).
Everything before it (SVM/GMM, GBM, CUSUM, Prototypical, ArcFace, the transformer zoo, GNN+radar-CNN)
already finished and their results are saved and unaffected by the pause. Everything after it
(contrastive, MAML, VAE hard-negative) hasn't started and will pick up automatically once
`train_receiver_fusion_gpu` finishes, after resuming.

---

## Where to find everything

### Live/raw training logs (one per model stage, plain stdout+stderr)
```
ml_2/evaluation/results/logs/
├── train_svm_gait.log                    # OC-SVM, binary SVC, GMM+SVM (3 variants in one log)
├── train_gbm.log                         # XGBoost
├── train_cusum.log                       # CUSUM adaptive detector
├── train_prototypical.log                # Prototypical embedding
├── train_arcface.log                     # ArcFace embedding
├── train_transformer_gpu.log             # WhoFi, dual-branch, cross-attention (3 in one log)
├── train_extra_classifiers_gpu.log       # GNN (subcarrier graph), radar-inspired CNN
├── train_receiver_fusion_gpu.log         # 3-receiver fusion (currently paused mid-run)
├── train_contrastive.log                 # not started yet
├── train_maml.log                        # not started yet
└── train_generative_hard_negative.log    # not started yet
```
The OVERALL orchestrator log (which stage started/finished/failed, with timing) is separate, at
`/tmp/ml2_run_all.log` -- not under the repo, since it's just a process-launch log, not a result.

### Machine-readable per-fold results (one CSV per model family, append-only)
```
ml_2/evaluation/results/
├── svm_gait_log.csv
├── gbm_log.csv
├── cusum_log.csv
├── prototypical_log.csv
├── arcface_log.csv
├── transformer_gpu_log.csv               # covers whofi/dualbranch/crossattn AND gnn/radar (same file)
├── receiver_fusion_gpu_log.csv
├── contrastive_log.csv                   # will appear once that stage runs
├── maml_log.csv                          # will appear once that stage runs
└── generative_hard_negative_log.csv      # will appear once that stage runs
```
Each row = one fold (one held-out stranger, or one held-out day) for one model, with both the
pooled WINDOW-level metrics (`accuracy`, `eer`, `auroc`) and the SESSION-level metrics
(`session_accuracy`, `session_auroc`, `n_sessions`) -- see `MODEL_TECHNICAL_REFERENCE.md` section 0
for why both are reported and which one to trust.

### Full per-session detail (every session, every fold, every model that's run since the fix)
```
ml_2/evaluation/results/session_breakdown.csv                    # gnn_subcarrier, radar_cnn so far;
                                                                     grows automatically as later
                                                                     stages (receiver fusion onward)
                                                                     finish -- no action needed
ml_2/evaluation/results/session_breakdown_PARTIAL_backfill.csv   # everything BEFORE that fix existed
                                                                     (SVM/GMM, GBM, CUSUM, Prototypical,
                                                                     ArcFace, transformer zoo) --
                                                                     worst-12-sessions-per-fold ONLY,
                                                                     see the file's own header caveat
```

### Human-readable write-ups (the ones I've actually written prose analysis into)
```
ml_2/evaluation/results/SESSION_BREAKDOWN_SUMMARY.md      # session-level findings: the receiver
                                                             anomaly, the cross-day authorized-session
                                                             flip -- gnn_subcarrier/radar_cnn so far
ml_2/evaluation/results/DAY_PERSON_MODEL_ACCURACY.md      # full day x person x model accuracy table,
                                                             all 12 models trained so far
ml_2/MODEL_TECHNICAL_REFERENCE.md                          # architecture/kernel/regularization/
                                                             parameter-count detail for every model
ml_2/PLAN.md                                               # why this battery of models, what's
                                                             deliberately out of scope
ml_2/OPEN_SET_MODEL_STRATEGY.md (repo root)                 # the original research-informed strategy
                                                             doc this whole ml_2 effort was built from
```

### The consolidated leaderboard (regenerate any time, works on whatever's finished so far)
```bash
python3 -m ml_2.evaluation.backtest_all
# writes ml_2/evaluation/results/backtest_summary.csv and prints a ranked comparison to stdout
```
This hasn't been re-run since GNN/radar-CNN finished -- the copy on disk (if any) may be stale
relative to the leaderboard numbers reported in chat. Safe to re-run any time, including while the
main pipeline is paused.
