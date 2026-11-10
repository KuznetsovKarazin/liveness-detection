# Early-stopping patience 4 and 12, the 36 models in compact form (all columns, sources and checks in `patience_table.md`)

| model | in the ten | restored epoch p4 → p12 | first recovery p4 → p12 | recovery holds at last epoch p4 → p12 | dev acc at ckpt p4 → p12 | restored epoch changed |
|---|---|---|---|---|---|---|
| B fold 1 seed 42 | yes | 1 → 10 | — → 8 | no → yes | 0.5000 → 0.9950 | yes |
| B fold 1 seed 43 | — | 11 → 11 | 5 → 5 | yes → yes | 0.9975 → 0.9975 | — |
| B fold 1 seed 44 | — | 6 → 11 | 6 → 6 | yes → yes | 0.9950 → 0.9975 | yes |
| B fold 2 seed 42 | — | 9 → 9 | 5 → 5 | no → no | 0.9475 → 0.9475 | — |
| B fold 2 seed 43 | — | 7 → 7 | 7 → 7 | yes → yes | 1.0000 → 1.0000 | — |
| B fold 2 seed 44 | — | 10 → 10 | 6 → 6 | yes → yes | 0.9975 → 0.9975 | — |
| B fold 3 seed 42 | — | 12 → 12 | 6 → 6 | yes → yes | 1.0000 → 1.0000 | — |
| B fold 3 seed 43 | — | 10 → 10 | 6 → 6 | yes → yes | 0.9925 → 0.9925 | — |
| B fold 3 seed 44 | — | 3 → 11 | — → 10 | no → yes | 0.2375 → 1.0000 | yes |
| B fold 4 seed 42 | — | 11 → 11 | 6 → 6 | yes → yes | 1.0000 → 1.0000 | — |
| B fold 4 seed 43 | — | 6 → 12 | 6 → 6 | yes → yes | 1.0000 → 1.0000 | yes |
| B fold 4 seed 44 | yes | 2 → 11 | — → 7 | no → yes | 0.5000 → 0.9925 | yes |
| A* fold 1 seed 42 | — | 10 → 10 | 6 → 6 | yes → yes | 0.9950 → 0.9950 | — |
| A* fold 1 seed 43 | — | 9 → 9 | 7 → 7 | yes → yes | 0.9975 → 0.9975 | — |
| A* fold 1 seed 44 | — | 11 → 11 | 7 → 7 | yes → yes | 0.9875 → 0.9875 | — |
| A* fold 2 seed 42 | — | 11 → 11 | 5 → 5 | yes → yes | 1.0000 → 1.0000 | — |
| A* fold 2 seed 43 | yes | 1 → 10 | — → 7 | no → yes | 0.5000 → 1.0000 | yes |
| A* fold 2 seed 44 | yes | 1 → 12 | — → 10 | no → yes | 0.5000 → 0.9975 | yes |
| A* fold 3 seed 42 | yes | 1 → 9 | — → 9 | no → yes | 0.5000 → 0.9425 | yes |
| A* fold 3 seed 43 | — | 12 → 12 | 6 → 6 | yes → yes | 0.9975 → 0.9975 | — |
| A* fold 3 seed 44 | yes | 1 → 11 | — → 11 | no → no | 0.5000 → 0.8900 | yes |
| A* fold 4 seed 42 | — | 10 → 10 | 10 → 10 | no → no | 0.7975 → 0.7975 | — |
| A* fold 4 seed 43 | — | 9 → 9 | 7 → 7 | no → no | 0.9950 → 0.9950 | — |
| A* fold 4 seed 44 | yes | 1 → 12 | — → 10 | no → yes | 0.5000 → 0.8650 | yes |
| B' fold 1 seed 42 | — | 2 → 12 | 2 → 2 | no → yes | 0.7725 → 0.9975 | yes |
| B' fold 1 seed 43 | — | 4 → 4 | 4 → 4 | no → no | 0.9400 → 0.9400 | — |
| B' fold 1 seed 44 | yes | 1 → 9 | — → 7 | no → no | 0.5000 → 0.9950 | yes |
| B' fold 2 seed 42 | — | 11 → 11 | 11 → 11 | no → no | 0.8775 → 0.8775 | — |
| B' fold 2 seed 43 | — | 10 → 10 | 3 → 3 | yes → yes | 1.0000 → 1.0000 | — |
| B' fold 2 seed 44 | yes | 2 → 12 | — → 12 | no → yes | 0.5000 → 0.9875 | yes |
| B' fold 3 seed 42 | — | 1 → 11 | 1 → 1 | no → yes | 0.8325 → 0.9900 | yes |
| B' fold 3 seed 43 | — | 9 → 9 | 7 → 7 | yes → yes | 1.0000 → 1.0000 | — |
| B' fold 3 seed 44 | — | 9 → 9 | 7 → 7 | yes → yes | 0.9975 → 0.9975 | — |
| B' fold 4 seed 42 | — | 2 → 2 | 1 → 1 | no → no | 0.9175 → 0.9175 | — |
| B' fold 4 seed 43 | — | 3 → 3 | 3 → 3 | no → yes | 1.0000 → 1.0000 | — |
| B' fold 4 seed 44 | yes | 1 → 11 | — → 10 | no → yes | 0.5000 → 0.9850 | yes |

Legend. p4 = run `20261106-SESSABLRUN-d999458` (early-stopping patience 4), p12 = run `20261109-SESSABLRUN-6a50265` (patience 12); in the ten = one of the ten degenerate trainings of the p4 run (— = not among the ten); first recovery = first epoch meeting the recovery criterion (— when none); recovery holds at last epoch = that criterion met at the last epoch run; dev acc at ckpt = dev accuracy at the restored epoch; restored epoch changed = the two restored epochs differ (— when equal). Train and dev only; the test per model is in `patience_table.md`.

- Models: 36 (3 arms x 4 folds x 3 seeds); complete: 36 with patience 4, 36 with patience 12.
- In the ten original degenerate trainings: 10.
- Degenerate with patience 4: 10 of 36 (10 of the ten).
- Degenerate with patience 12: 0 of 36 (0 of the ten).
- Shared histories identical: 36 of 36 (shared epochs = epochs run with patience 4 in 36 of 36).
- Epochs run with patience 4: 5-12, 17 of 36 models at 12.
- Epochs run with patience 12: 12, 36 of 36 models at 12.
- With a recovery (an epoch meeting the recovery criterion) with patience 4: 25 of 36 (0 of the ten).
- With a recovery (an epoch meeting the recovery criterion) with patience 12: 36 of 36 (10 of the ten).
- Recovery holds at the last epoch with patience 4: 16 of 36 (0 of the ten).
- Recovery holds at the last epoch with patience 12: 28 of 36 (8 of the ten).
- First recovery of the ten with patience 12: median (min-max) 9.5 (7-12).
- First recovery of the ten at epoch 10 or later with patience 12: 5 of the ten (cut-off of CONCLUSION.md).
- Restored epoch changed: 15 of 36 (10 of the ten, 5 of the other 26).
- Stopped before epoch 12 with patience 4: 19 models, 9 outside the ten, of which 5 changed restored epoch.
- Dev accuracy at the checkpoint, the ten, median (min-max) with patience 4: 0.5000 (0.5000-0.5000).
- Dev accuracy at the checkpoint, the ten, median (min-max) with patience 12: 0.9900 (0.8650-1.0000).
- Dev accuracy at the checkpoint, the other 26, median (min-max) with patience 4: 0.9963 (0.2375-1.0000).
- Dev accuracy at the checkpoint, the other 26, median (min-max) with patience 12: 0.9975 (0.7975-1.0000).
- Dev accuracy at the checkpoint below 0.9 with patience 12, the ten: 2 (cut-off of CONCLUSION.md).
- Dev accuracy at the checkpoint below 0.9 with patience 12, the other 26: 2 (cut-off of CONCLUSION.md).
