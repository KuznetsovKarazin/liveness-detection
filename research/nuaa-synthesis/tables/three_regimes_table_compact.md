# The three regimes, the 36 models in compact form (all columns, sources, checks and the test in `three_regimes_table.md`)

| model | in the ten | restored epoch p4 → p12 → p12c | first recovery p4 → p12 → p12c | holds at last epoch p4 → p12 → p12c | dev acc at ckpt p4 → p12 → p12c | restored epoch changed p4→p12 / p12→p12c |
|---|---|---|---|---|---|---|
| B fold 1 seed 42 | yes | 1 → 10 → 9 | — → 8 → 4 | no → yes → yes | 0.5000 → 0.9950 → 1.0000 | yes / yes |
| B fold 1 seed 43 | — | 11 → 11 → 9 | 5 → 5 → 5 | yes → yes → yes | 0.9975 → 0.9975 → 1.0000 | — / yes |
| B fold 1 seed 44 | — | 6 → 11 → 11 | 6 → 6 → 6 | yes → yes → yes | 0.9950 → 0.9975 → 1.0000 | yes / — |
| B fold 2 seed 42 | — | 9 → 9 → 10 | 5 → 5 → 8 | no → no → yes | 0.9475 → 0.9475 → 0.9950 | — / yes |
| B fold 2 seed 43 | — | 7 → 7 → 7 | 7 → 7 → 6 | yes → yes → yes | 1.0000 → 1.0000 → 1.0000 | — / — |
| B fold 2 seed 44 | — | 10 → 10 → 9 | 6 → 6 → 5 | yes → yes → no | 0.9975 → 0.9975 → 0.9975 | — / yes |
| B fold 3 seed 42 | — | 12 → 12 → 12 | 6 → 6 → 7 | yes → yes → yes | 1.0000 → 1.0000 → 1.0000 | — / — |
| B fold 3 seed 43 | — | 10 → 10 → 9 | 6 → 6 → 6 | yes → yes → yes | 0.9925 → 0.9925 → 0.9250 | — / yes |
| B fold 3 seed 44 | — | 3 → 11 → 10 | — → 10 → 7 | no → yes → yes | 0.2375 → 1.0000 → 0.9150 | yes / yes |
| B fold 4 seed 42 | — | 11 → 11 → 11 | 6 → 6 → 6 | yes → yes → yes | 1.0000 → 1.0000 → 1.0000 | — / — |
| B fold 4 seed 43 | — | 6 → 12 → 6 | 6 → 6 → 6 | yes → yes → no | 1.0000 → 1.0000 → 1.0000 | yes / yes |
| B fold 4 seed 44 | yes | 2 → 11 → 11 | — → 7 → 10 | no → yes → yes | 0.5000 → 0.9925 → 1.0000 | yes / — |
| A* fold 1 seed 42 | — | 10 → 10 → 10 | 6 → 6 → 6 | yes → yes → yes | 0.9950 → 0.9950 → 0.9950 | — / — |
| A* fold 1 seed 43 | — | 9 → 9 → 9 | 7 → 7 → 7 | yes → yes → no | 0.9975 → 0.9975 → 0.9975 | — / — |
| A* fold 1 seed 44 | — | 11 → 11 → 9 | 7 → 7 → 7 | yes → yes → yes | 0.9875 → 0.9875 → 0.9975 | — / yes |
| A* fold 2 seed 42 | — | 11 → 11 → 7 | 5 → 5 → 5 | yes → yes → yes | 1.0000 → 1.0000 → 1.0000 | — / yes |
| A* fold 2 seed 43 | yes | 1 → 10 → 9 | — → 7 → 6 | no → yes → yes | 0.5000 → 1.0000 → 1.0000 | yes / yes |
| A* fold 2 seed 44 | yes | 1 → 12 → 10 | — → 10 → 5 | no → yes → yes | 0.5000 → 0.9975 → 1.0000 | yes / yes |
| A* fold 3 seed 42 | yes | 1 → 9 → 10 | — → 9 → 6 | no → yes → no | 0.5000 → 0.9425 → 0.9900 | yes / yes |
| A* fold 3 seed 43 | — | 12 → 12 → 12 | 6 → 6 → 6 | yes → yes → yes | 0.9975 → 0.9975 → 0.9725 | — / — |
| A* fold 3 seed 44 | yes | 1 → 11 → 8 | — → 11 → 7 | no → no → yes | 0.5000 → 0.8900 → 0.8750 | yes / yes |
| A* fold 4 seed 42 | — | 10 → 10 → 11 | 10 → 10 → 9 | no → no → yes | 0.7975 → 0.7975 → 0.8975 | — / yes |
| A* fold 4 seed 43 | — | 9 → 9 → 9 | 7 → 7 → 7 | no → no → no | 0.9950 → 0.9950 → 0.9950 | — / — |
| A* fold 4 seed 44 | yes | 1 → 12 → 8 | — → 10 → 6 | no → yes → yes | 0.5000 → 0.8650 → 0.9975 | yes / yes |
| B' fold 1 seed 42 | — | 2 → 12 → 10 | 2 → 2 → 2 | no → yes → yes | 0.7725 → 0.9975 → 0.9975 | yes / yes |
| B' fold 1 seed 43 | — | 4 → 4 → 12 | 4 → 4 → 4 | no → no → yes | 0.9400 → 0.9400 → 0.9675 | — / yes |
| B' fold 1 seed 44 | yes | 1 → 9 → 6 | — → 7 → 6 | no → no → yes | 0.5000 → 0.9950 → 0.9975 | yes / yes |
| B' fold 2 seed 42 | — | 11 → 11 → 10 | 11 → 11 → — | no → no → no | 0.8775 → 0.8775 → 0.8550 | — / yes |
| B' fold 2 seed 43 | — | 10 → 10 → 12 | 3 → 3 → 3 | yes → yes → yes | 1.0000 → 1.0000 → 0.9950 | — / yes |
| B' fold 2 seed 44 | yes | 2 → 12 → 12 | — → 12 → 7 | no → yes → yes | 0.5000 → 0.9875 → 0.9950 | yes / — |
| B' fold 3 seed 42 | — | 1 → 11 → 4 | 1 → 1 → 1 | no → yes → yes | 0.8325 → 0.9900 → 0.9900 | yes / yes |
| B' fold 3 seed 43 | — | 9 → 9 → 9 | 7 → 7 → 7 | yes → yes → no | 1.0000 → 1.0000 → 1.0000 | — / — |
| B' fold 3 seed 44 | — | 9 → 9 → 9 | 7 → 7 → 8 | yes → yes → no | 0.9975 → 0.9975 → 0.9850 | — / — |
| B' fold 4 seed 42 | — | 2 → 2 → 11 | 1 → 1 → 1 | no → no → no | 0.9175 → 0.9175 → 0.9700 | — / yes |
| B' fold 4 seed 43 | — | 3 → 3 → 3 | 3 → 3 → 3 | no → yes → yes | 1.0000 → 1.0000 → 1.0000 | — / — |
| B' fold 4 seed 44 | yes | 1 → 11 → 10 | — → 10 → 10 | no → yes → no | 0.5000 → 0.9850 → 0.8900 | yes / yes |

Legend. p4 = run `20261106-SESSABLRUN-d999458` (early-stopping patience 4, ReduceLROnPlateau), p12 = run `20261109-SESSABLRUN-6a50265` (patience 12, ReduceLROnPlateau), p12c = run `20261111-SESSABLRUN-8568ffa` (patience 12, constant learning rate 0.001); the same 36 models, data and seeds in the three runs; in the ten = one of the ten degenerate trainings of the p4 run (— = not among the ten); first recovery = first epoch meeting the recovery criterion (— when none); holds at last epoch = that criterion met at the last epoch run; dev acc at ckpt = dev accuracy at the restored epoch; restored epoch changed = the two restored epochs differ (— when equal). Train and dev only.

**Recovery** (train and dev only)

- Models: 36 (3 arms x 4 folds x 3 seeds); complete: p4 36, p12 36, p12c 36.
- The ten (degenerate with p4) with a recovery: p4 0 of 10, p12 10 of 10, p12c 10 of 10.
- Degenerate among the 36: p4 10, p12 0, p12c 0.
- With a recovery among the 36: p4 25, p12 36, p12c 35.
- First recovery of the ten, median (min-max): p4 —, p12 9.5 (7-12), p12c 6 (4-10).
- First recovery of the ten, p12c against p12: earlier in 8, at the same epoch in 1, later in 1.
- First recovery of the other 26, median (min-max) [with one]: p4 6 (1-11) [25], p12 6 (1-11) [26], p12c 6 (1-9) [25].
- Recovery lost (outside the ten): p4 to p12 none; p12 to p12c B' fold 2 seed 42.

**Stability** (train and dev only)

- Recovery holds at the last epoch: p4 16 of 36, p12 28 of 36, p12c 26 of 36.
- Recovery holds at the last epoch, the ten: p4 0, p12 8, p12c 8; the other 26: p4 16, p12 20, p12c 18.
- Holds at the last epoch with p12 but not with p12c: 7; with p12c but not with p12: 5.
- Models with an epoch at 0.5 after their first recovery (a relapse): p4 5, p12 5, p12c 6.
- B fold 3 seed 44 (reported apart): first recovery — → 10 → 7; holds at last no → yes → yes; at 0.5 after it — → 0 → 0.
- B' fold 2 seed 42 (recovery lost): first recovery 11 → 11 → —; holds at last no → no → no; at 0.5 after it 0 → 0 → —.
- B' fold 4 seed 42 (reported apart): first recovery 1 → 1 → 1; holds at last no → no → no; at 0.5 after it 2 → 3 → 3.

**Selected checkpoint** (train and dev only)

- Restored epoch changed: p4 to p12 15 of 36 (10 of the ten), p12 to p12c 23 of 36 (8 of the ten).
- Same restored epoch with p12 and p12c: 13; of these, same weights (inside the shared window): 5.
- Different checkpoints with p12 and p12c: 31 of 36 (another restored epoch 23, same epoch after the window 8).
- Best val_loss at the checkpoint, p12c against p12: higher in 12, lower in 19, equal in 5.
- Dev accuracy at the checkpoint, p12c against p12: higher in 14, lower in 8, equal in 14.
- Dev accuracy and AUROC at the checkpoint, next four lines: median (min-max) per regime.
- Dev acc at ckpt, the ten: p4 0.5000 (0.5000-0.5000), p12 0.9900 (0.8650-1.0000), p12c 0.9975 (0.8750-1.0000).
- Dev acc at ckpt, the other 26: p4 0.9963 (0.2375-1.0000), p12 0.9975 (0.7975-1.0000), p12c 0.9963 (0.8550-1.0000).
- Dev AUROC at ckpt, the ten: p4 0.5000 (0.2114-0.9975), p12 0.9955 (0.9895-1.0000), p12c 1.0000 (0.9831-1.0000).
- Dev AUROC at ckpt, the other 26: p4 1.0000 (0.0950-1.0000), p12 1.0000 (0.9830-1.0000), p12c 1.0000 (0.8678-1.0000).
- Dev acc at ckpt below 0.9: p4 15 of 36, p12 4 of 36, p12c 4 of 36 (the ten: p4 10, p12 2, p12c 2).
- B fold 3 seed 44: restored epoch 3 → 11 → 10; dev acc at ckpt 0.2375 → 1.0000 → 0.9150.
- B' fold 2 seed 42: restored epoch 11 → 11 → 10; dev acc at ckpt 0.8775 → 0.8775 → 0.8550.
- B' fold 4 seed 42: restored epoch 2 → 2 → 11; dev acc at ckpt 0.9175 → 0.9175 → 0.9700.
