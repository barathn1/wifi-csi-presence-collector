# Per-day, per-person accuracy -- every model

Session-level accuracy (fraction of that person's sessions on that day that the model's majority
vote got right), broken down by date and by person, for every model trained so far.

## READ THIS FIRST -- two very different kinds of numbers are mixed in this table

Models fall into two groups with fundamentally different data quality, and they look identical in
the table unless you check which group a column belongs to:

- **FULL coverage (`gnn_subcarrier`, `radar_cnn`)**: every session that model was ever tested on is
  counted. These numbers are the real, complete accuracy for that day/person.
- **PARTIAL coverage (every other model: `oc_svm`, `binary_svc`, `gmm_svm_wii`, `gbm_xgboost`,
  `cusum_detector`, `prototypical_embedding`, `arcface_embedding`, `whofi`, `dualbranch`,
  `crossattn`)**: these finished training BEFORE full per-session logging existed. All that survived
  is a printed list of the **worst 12 sessions per fold** (sorted worst-first). That means for these
  columns:
  - **A number shown is biased LOW.** It's computed only from the sessions that were bad enough to
    make the "worst 12" list -- sessions that scored well were never printed, so they can't pull the
    average up. The true accuracy for that day/person is almost certainly higher than what's shown.
  - **A blank (NaN) does NOT mean "no data" in the usual sense -- it usually means the model did FINE
    on every session for that day/person** (good enough that none of them ranked in the fold's worst
    12), so nothing was ever printed for them to recover. A small number of blanks are genuinely
    because the model was never evaluated on that person/day at all (e.g. if that person's identity
    was the one held out FROM training rather than held out as the stranger being tested).

**Bottom line: do not compare a PARTIAL column's low number against a FULL column's low number as if
they mean the same thing.** `gnn_subcarrier`/`radar_cnn` numbers are ground truth. Every other
column's numbers are a worst-case-only snapshot, useful for spotting real problem sessions (a 0.00
in a partial column IS a real, bad session) but not for computing "this model's overall accuracy
for this person" (the blanks are hiding better results).

---

## Accuracy table (1.00 = every session correct, 0.00 = every session wrong, blank = see caveat above)

| Date | Person | oc_svm | binary_svc | gmm_svm_wii | gbm_xgboost | cusum | prototypical | arcface | whofi | dualbranch | crossattn | **gnn (FULL)** | **radar_cnn (FULL)** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-15 | abdul | -- | 0.00 | -- | 0.00 | -- | 0.50 | 0.00 | 0.00 | 0.00 | 0.50 | **0.00** | **0.00** |
| 2026-09-15 | anjali | -- | 0.50 | -- | 0.00 | -- | 0.29 | 0.00 | 0.00 | 0.20 | -- | **1.00** | **1.00** |
| 2026-09-15 | barath | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.67 | 0.50 | 1.00 | **1.00** | **1.00** |
| 2026-09-15 | divya | -- | 0.00 | -- | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | **0.00** |
| 2026-09-15 | harshitha | -- | 0.00 | -- | 0.00 | -- | 0.50 | 0.00 | 0.33 | 0.00 | 0.00 | **0.00** | **0.00** |
| 2026-09-15 | none | 0.00 | 1.00 | 0.00 | -- | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | -- | **1.00** | **0.76** |
| 2026-09-15 | siva | -- | 0.00 | -- | 0.00 | -- | 0.00 | 0.00 | 0.25 | 0.25 | 0.50 | **0.00** | **0.00** |
| 2026-09-15 | sumanth | -- | 0.33 | -- | 0.33 | -- | 0.00 | 0.00 | 0.33 | 0.00 | 0.50 | **0.00** | **0.00** |
| 2026-09-16 | anjali | 0.00 | 1.00 | 0.00 | 0.93 | 0.00 | 0.32 | 0.00 | 0.88 | 0.95 | 0.82 | **1.00** | **1.00** |
| 2026-09-16 | barath | 0.00 | 0.93 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.96 | **1.00** | **1.00** |
| 2026-09-16 | divya | -- | 0.25 | -- | 0.00 | -- | 0.00 | -- | 0.00 | 0.00 | 0.00 | **0.00** | **0.00** |
| 2026-09-16 | kishore | -- | 0.25 | -- | 0.00 | -- | 0.00 | -- | 0.00 | 0.00 | 1.00 | **0.00** | **0.00** |
| 2026-09-16 | manas | -- | 0.00 | -- | 0.00 | -- | 0.00 | -- | 0.25 | 0.00 | 0.00 | **0.00** | **0.00** |
| 2026-09-16 | none | 0.00 | -- | 0.00 | -- | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 | -- | **1.00** | **1.00** |
| 2026-09-16 | sumanth | -- | 0.33 | -- | 0.33 | -- | 0.00 | -- | 0.67 | 0.25 | 0.00 | **0.00** | **0.00** |
| 2026-09-17 | anjali | 0.00 | 0.23 | 0.25 | 0.25 | 0.33 | 0.67 | 0.00 | 0.29 | 0.42 | 0.33 | **1.00** | **0.87** |
| 2026-09-17 | barath | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.33 | 0.00 | 0.64 | 1.00 | 0.88 | **1.00** | **1.00** |
| 2026-09-17 | divya | 0.00 | 0.20 | 0.00 | 0.25 | 0.00 | 0.00 | -- | 0.60 | 0.33 | 0.67 | **0.00** | **0.00** |
| 2026-09-17 | harshitha | -- | 0.25 | -- | 0.50 | -- | 0.00 | -- | 0.00 | 0.67 | 1.00 | **0.00** | **0.25** |
| 2026-09-17 | manas | -- | 0.00 | -- | 0.00 | 1.00 | 0.00 | -- | 0.00 | 0.00 | 0.00 | **0.00** | **0.00** |
| 2026-09-17 | none | 0.00 | -- | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | -- | -- | -- | **1.00** | **1.00** |
| 2026-09-21 | anjali | 0.00 | 0.80 | 0.67 | 0.73 | 0.67 | 0.33 | 0.29 | 0.20 | 0.38 | 1.00 | **0.67** | **0.07** |
| 2026-09-21 | barath | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.25 | -- | **0.00** | **0.17** |
| 2026-09-21 | none | 0.00 | 0.50 | 0.00 | 0.67 | 0.33 | 0.00 | 0.00 | 0.71 | 0.83 | 0.86 | **0.23** | **1.00** |
| 2026-09-21 | promoda | -- | 0.00 | -- | 0.00 | 0.00 | -- | 0.00 | 0.25 | 0.00 | 0.00 | **0.50** | **0.50** |
| 2026-09-21 | sumanth | 0.50 | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.50** | **0.75** |
| 2026-09-22 | anjali | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.21 | 0.00 | 1.00 | 1.00 | 1.00 | **1.00** | **0.33** |
| 2026-09-22 | barath | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.44 | 0.00 | 1.00 | 1.00 | -- | **1.00** | **1.00** |
| 2026-09-22 | divya | -- | 0.50 | -- | 0.67 | -- | 0.25 | 0.00 | 0.33 | 0.33 | 1.00 | **0.00** | **0.25** |
| 2026-09-22 | harshitha | 0.00 | 0.75 | 0.00 | 0.50 | -- | 0.00 | -- | 0.33 | 0.00 | -- | **0.25** | **0.50** |
| 2026-09-22 | none | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.50 | -- | 0.33 | 0.67 | -- | **1.00** | **1.00** |
| 2026-09-22 | vishwa | -- | 0.00 | -- | 0.50 | -- | 0.00 | 0.00 | 0.33 | 0.50 | 1.00 | **0.00** | **0.25** |

`anjali`/`barath` = the 2 authorized identities. `abdul`/`divya`/`harshitha`/`kishore`/`manas`/
`promoda`/`siva`/`sumanth`/`vishwa` = the 9 unauthorized stranger identities. `none` = empty room.
`--` = no data survived for this cell (see the coverage explanation above for what that means per
column). Bold columns are the two FULL-coverage models.

---

## What actually jumps out, reading the FULL-coverage columns only (gnn / radar_cnn)

- **Both models are consistently perfect (1.00) on `barath` every single day** -- 09-15 through
  09-22, no exceptions. Barath's authorized gait signature is the single most reliably recognized
  thing in the whole dataset.
- **`anjali` is much less stable**: 1.00 on 09-15/16/22 for GNN, but GNN drops to 0.67 on 09-21 and
  radar-CNN specifically collapses to **0.07 on 09-21** and **0.33 on 09-22** -- anjali's sessions on
  the 3-receiver days (09-21/22) are noticeably harder for radar-CNN than her sessions on the
  single-receiver days. Worth checking whether this is receiver-specific (radar-CNN only uses
  amplitude, no phase -- a receiver-specific amplitude calibration issue on 09-21/22 would show up
  exactly like this).
- **`divya` is the single hardest stranger for both FULL models**: accuracy 0.00 on every day she
  appears (09-15, 09-16, 09-17) except a partial recovery on 09-22 (0.00 GNN / 0.25 radar-CNN). Both
  models are consistently, confidently WRONG about her -- this isn't noise, it's a systematic miss on
  one specific person across multiple days.
- **`kishore`, `manas`, `harshitha` are also weak spots for GNN** (0.00 on every day they appear
  except one 1.00 for kishore on 09-16) -- these look like the strangers whose gait the model
  confuses most often with an authorized person.
- **`none` (empty room) is reliable for both models on 09-15/16/17 (1.00) but degrades on 09-21/22**
  (GNN: 0.23/1.00, radar-CNN: 0.76/1.00 -- inconsistent) -- consistent with the multi-receiver days
  introducing empty-room noise the single-receiver days didn't have (see the `ac276ea2f278`
  receiver-specific anomaly already flagged in `SESSION_BREAKDOWN_SUMMARY.md`).

## Caveat on the partial-coverage columns worth stating plainly

Looking only at the partial columns, `binary_svc` and `crossattn` visually look "the least bad" of
the 10 partial models simply because they scored better on whichever sessions DID make the worst-12
list -- that is a real, comparable signal (their worst sessions are less bad than e.g. `oc_svm`'s
worst sessions, which is 0.00 almost everywhere it has data at all). But do not read a blank cell for
`binary_svc` as "unknown" and a blank for `oc_svm` as "also unknown" and treat them the same --
`oc_svm` has real 0.00s scattered through nearly every date/person it appears for, which is
consistent with its already-reported overall AUROC being worse than chance.

---

*Generated from `session_breakdown.csv` (full, gnn/radar_cnn) and
`session_breakdown_PARTIAL_backfill.csv` (partial, everything else) as of the point receiver fusion
was still training. Will be extended with real FULL-coverage columns for receiver fusion,
contrastive, MAML, and the VAE hard-negative model as they finish.*
