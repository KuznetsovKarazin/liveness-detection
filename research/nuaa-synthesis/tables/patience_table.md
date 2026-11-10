# Annex: early-stopping patience 4 and 12, the 36 LivenessNet models of the session ablation run (runs `20261106-SESSABLRUN-d999458` and `20261109-SESSABLRUN-6a50265`, comparison `20261109-PATVER-4f2b925`)

Train and dev from the public comparison of the two runs (`nuaa-patience-verification/comparison/comparison.csv`, one row per model, and `nuaa-patience-verification/comparison/curves.csv`, one row per epoch of the patience-12 run), checked by `nuaa_synthesis.py`: every value of the comparison except the dev AUROC at the checkpoint (copied; checked against the JSON rows and the group medians) rebuilt from the epoch curves with the criteria of the comparison aggregate (restored epoch = first epoch of minimum dev val_loss; epochs run = the early-stopping replay with the patience of the run, at most 12 epochs; degenerate: restored_epoch in {1, 2} and val_acc_at_best = 0.5 (within 1e-6): the rule written in the delivery of the session ablation run, re-applied here; recovery: first epoch with val_accuracy >= 0.75 and val_loss < 1.0 (dev only)), equal to the JSON rows of the comparison and to the training cards of the two public aggregates; the first epochs of the patience-12 run are those of the patience-4 run (shared histories, identical: the largest absolute differences of the comparison are required to be 0), so the patience-4 columns are read from the same curves. "Recovery holds at last epoch" = the recovery criterion met at the last epoch run (epoch 12 with patience 12). Checkpoint = restored epoch; dev = the common dev of the fold (sessions 01-02). "Restored epoch changed" compares the restored epochs of the two runs only: with the same restored epoch the weights and threshold file hashes still differ in every model; identity is shown by identical test values and dev threshold (checked). Comparison verdict: positive.

## Train and dev, per model (patience 4, then patience 12)

| arm | fold | seed | in the ten | epochs run (p4) | restored epoch (p4) | degenerate (p4) | first recovery (p4) | recovery holds at last epoch (p4) | best val_loss (p4) | dev acc at ckpt (p4) | dev AUROC at ckpt (p4) | epochs at 0.5 (p4) | last lr (p4) | epochs run (p12) | restored epoch (p12) | degenerate (p12) | first recovery (p12) | recovery holds at last epoch (p12) | best val_loss (p12) | dev acc at ckpt (p12) | dev AUROC at ckpt (p12) | epochs at 0.5 (p12) | last lr (p12) | shared epochs | shared identical | restored epoch changed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B | 1 | 42 | yes | 5 | 1 | yes | — | no | 7.0611 | 0.5000 | 0.9975 | 5 | 0.0005 | 12 | 10 | no | 8 | yes | 0.0378 | 0.9950 | 0.9960 | 5 | 0.00025 | 5 | yes | yes |
| B | 1 | 43 | no | 12 | 11 | no | 5 | yes | 0.0056 | 0.9975 | 1.0000 | 3 | 0.0005 | 12 | 11 | no | 5 | yes | 0.0056 | 0.9975 | 1.0000 | 3 | 0.0005 | 12 | yes | no |
| B | 1 | 44 | no | 10 | 6 | no | 6 | yes | 0.0298 | 0.9950 | 0.9985 | 3 | 0.0005 | 12 | 11 | no | 6 | yes | 0.0224 | 0.9975 | 1.0000 | 3 | 0.00025 | 10 | yes | yes |
| B | 2 | 42 | no | 12 | 9 | no | 5 | no | 0.2308 | 0.9475 | 0.9888 | 3 | 0.000125 | 12 | 9 | no | 5 | no | 0.2308 | 0.9475 | 0.9888 | 3 | 0.000125 | 12 | yes | no |
| B | 2 | 43 | no | 11 | 7 | no | 7 | yes | 0.0011 | 1.0000 | 1.0000 | 4 | 0.00025 | 12 | 7 | no | 7 | yes | 0.0011 | 1.0000 | 1.0000 | 4 | 0.000125 | 11 | yes | no |
| B | 2 | 44 | no | 12 | 10 | no | 6 | yes | 0.0030 | 0.9975 | 1.0000 | 4 | 0.00025 | 12 | 10 | no | 6 | yes | 0.0030 | 0.9975 | 1.0000 | 4 | 0.00025 | 12 | yes | no |
| B | 3 | 42 | no | 12 | 12 | no | 6 | yes | 0.0168 | 1.0000 | 1.0000 | 4 | 0.00025 | 12 | 12 | no | 6 | yes | 0.0168 | 1.0000 | 1.0000 | 4 | 0.00025 | 12 | yes | no |
| B | 3 | 43 | no | 12 | 10 | no | 6 | yes | 0.0531 | 0.9925 | 1.0000 | 4 | 0.0005 | 12 | 10 | no | 6 | yes | 0.0531 | 0.9925 | 1.0000 | 4 | 0.0005 | 12 | yes | no |
| B | 3 | 44 | no | 7 | 3 | no | — | no | 3.2731 | 0.2375 | 0.0950 | 3 | 0.0005 | 12 | 11 | no | 10 | yes | 0.0119 | 1.0000 | 1.0000 | 3 | 0.000125 | 7 | yes | yes |
| B | 4 | 42 | no | 12 | 11 | no | 6 | yes | 0.0005 | 1.0000 | 1.0000 | 3 | 0.00025 | 12 | 11 | no | 6 | yes | 0.0005 | 1.0000 | 1.0000 | 3 | 0.00025 | 12 | yes | no |
| B | 4 | 43 | no | 10 | 6 | no | 6 | yes | 0.0013 | 1.0000 | 1.0000 | 3 | 0.0005 | 12 | 12 | no | 6 | yes | 0.0007 | 1.0000 | 1.0000 | 3 | 0.00025 | 10 | yes | yes |
| B | 4 | 44 | yes | 6 | 2 | yes | — | no | 2.3605 | 0.5000 | 0.9652 | 2 | 0.0005 | 12 | 11 | no | 7 | yes | 0.0262 | 0.9925 | 1.0000 | 2 | 0.000125 | 6 | yes | yes |
| A* | 1 | 42 | no | 12 | 10 | no | 6 | yes | 0.0104 | 0.9950 | 1.0000 | 4 | 0.001 | 12 | 10 | no | 6 | yes | 0.0104 | 0.9950 | 1.0000 | 4 | 0.001 | 12 | yes | no |
| A* | 1 | 43 | no | 12 | 9 | no | 7 | yes | 0.0165 | 0.9975 | 0.9995 | 3 | 0.0005 | 12 | 9 | no | 7 | yes | 0.0165 | 0.9975 | 0.9995 | 3 | 0.0005 | 12 | yes | no |
| A* | 1 | 44 | no | 12 | 11 | no | 7 | yes | 0.0345 | 0.9875 | 0.9991 | 2 | 0.0005 | 12 | 11 | no | 7 | yes | 0.0345 | 0.9875 | 0.9991 | 2 | 0.0005 | 12 | yes | no |
| A* | 2 | 42 | no | 12 | 11 | no | 5 | yes | 0.0001 | 1.0000 | 1.0000 | 3 | 0.00025 | 12 | 11 | no | 5 | yes | 0.0001 | 1.0000 | 1.0000 | 3 | 0.00025 | 12 | yes | no |
| A* | 2 | 43 | yes | 5 | 1 | yes | — | no | 60.9868 | 0.5000 | 0.5000 | 4 | 0.0005 | 12 | 10 | no | 7 | yes | 0.0004 | 1.0000 | 1.0000 | 4 | 0.00025 | 5 | yes | yes |
| A* | 2 | 44 | yes | 5 | 1 | yes | — | no | 18.0934 | 0.5000 | 0.2114 | 3 | 0.0005 | 12 | 12 | no | 10 | yes | 0.2171 | 0.9975 | 0.9950 | 3 | 0.00025 | 5 | yes | yes |
| A* | 3 | 42 | yes | 5 | 1 | yes | — | no | 3.1296 | 0.5000 | 0.4690 | 3 | 0.0005 | 12 | 9 | no | 9 | yes | 0.3282 | 0.9425 | 0.9900 | 3 | 0.000125 | 5 | yes | yes |
| A* | 3 | 43 | no | 12 | 12 | no | 6 | yes | 0.0088 | 0.9975 | 1.0000 | 5 | 0.0005 | 12 | 12 | no | 6 | yes | 0.0088 | 0.9975 | 1.0000 | 5 | 0.0005 | 12 | yes | no |
| A* | 3 | 44 | yes | 5 | 1 | yes | — | no | 33.4090 | 0.5000 | 0.5000 | 4 | 0.0005 | 12 | 11 | no | 11 | no | 0.5239 | 0.8900 | 0.9895 | 4 | 0.000125 | 5 | yes | yes |
| A* | 4 | 42 | no | 12 | 10 | no | 10 | no | 0.8106 | 0.7975 | 0.9990 | 3 | 0.0005 | 12 | 10 | no | 10 | no | 0.8106 | 0.7975 | 0.9990 | 3 | 0.0005 | 12 | yes | no |
| A* | 4 | 43 | no | 12 | 9 | no | 7 | no | 0.0272 | 0.9950 | 0.9993 | 4 | 0.0005 | 12 | 9 | no | 7 | no | 0.0272 | 0.9950 | 0.9993 | 4 | 0.0005 | 12 | yes | no |
| A* | 4 | 44 | yes | 5 | 1 | yes | — | no | 22.8274 | 0.5000 | 0.5000 | 5 | 0.0005 | 12 | 12 | no | 10 | yes | 0.3154 | 0.8650 | 0.9999 | 5 | 0.000125 | 5 | yes | yes |
| B' | 1 | 42 | no | 6 | 2 | no | 2 | no | 0.6101 | 0.7725 | 0.9937 | 5 | 0.0005 | 12 | 12 | no | 2 | yes | 0.0033 | 0.9975 | 1.0000 | 5 | 0.000125 | 6 | yes | yes |
| B' | 1 | 43 | no | 8 | 4 | no | 4 | no | 0.8647 | 0.9400 | 0.9899 | 1 | 0.0005 | 12 | 4 | no | 4 | no | 0.8647 | 0.9400 | 0.9899 | 1 | 0.000125 | 8 | yes | no |
| B' | 1 | 44 | yes | 5 | 1 | yes | — | no | 12.1449 | 0.5000 | 0.4975 | 5 | 0.0005 | 12 | 9 | no | 7 | no | 0.1895 | 0.9950 | 0.9950 | 5 | 0.000125 | 5 | yes | yes |
| B' | 2 | 42 | no | 12 | 11 | no | 11 | no | 0.8750 | 0.8775 | 0.9830 | 7 | 0.0005 | 12 | 11 | no | 11 | no | 0.8750 | 0.8775 | 0.9830 | 7 | 0.0005 | 12 | yes | no |
| B' | 2 | 43 | no | 12 | 10 | no | 3 | yes | 0.0042 | 1.0000 | 1.0000 | 3 | 0.0005 | 12 | 10 | no | 3 | yes | 0.0042 | 1.0000 | 1.0000 | 3 | 0.0005 | 12 | yes | no |
| B' | 2 | 44 | yes | 6 | 2 | yes | — | no | 11.4259 | 0.5000 | 0.5000 | 6 | 0.0005 | 12 | 12 | no | 12 | yes | 0.0736 | 0.9875 | 0.9950 | 7 | 0.00025 | 6 | yes | yes |
| B' | 3 | 42 | no | 5 | 1 | no | 1 | no | 0.3211 | 0.8325 | 1.0000 | 3 | 0.0005 | 12 | 11 | no | 1 | yes | 0.0516 | 0.9900 | 0.9975 | 3 | 0.000125 | 5 | yes | yes |
| B' | 3 | 43 | no | 12 | 9 | no | 7 | yes | 0.0011 | 1.0000 | 1.0000 | 5 | 0.0005 | 12 | 9 | no | 7 | yes | 0.0011 | 1.0000 | 1.0000 | 5 | 0.0005 | 12 | yes | no |
| B' | 3 | 44 | no | 12 | 9 | no | 7 | yes | 0.0341 | 0.9975 | 1.0000 | 6 | 0.00025 | 12 | 9 | no | 7 | yes | 0.0341 | 0.9975 | 1.0000 | 6 | 0.00025 | 12 | yes | no |
| B' | 4 | 42 | no | 6 | 2 | no | 1 | no | 0.1375 | 0.9175 | 0.9966 | 2 | 0.0005 | 12 | 2 | no | 1 | no | 0.1375 | 0.9175 | 0.9966 | 3 | 0.0000625 | 6 | yes | no |
| B' | 4 | 43 | no | 7 | 3 | no | 3 | no | 0.0142 | 1.0000 | 1.0000 | 3 | 0.0005 | 12 | 3 | no | 3 | yes | 0.0142 | 1.0000 | 1.0000 | 3 | 0.0000625 | 7 | yes | no |
| B' | 4 | 44 | yes | 5 | 1 | yes | — | no | 17.2907 | 0.5000 | 0.5000 | 4 | 0.0005 | 12 | 11 | no | 10 | yes | 0.0851 | 0.9850 | 0.9988 | 4 | 0.00025 | 5 | yes | yes |

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

## Test, per model (not used to choose or judge the recipe; patience 4, then patience 12)

From the public `aggregate.json` of the two runs (`nuaa-session-ablation-run/tables/aggregate.json`, `nuaa-patience-verification/tables/aggregate.json`): every count and rate checked against the fold schema and summed into the pooled rows of `session_ablation_table.md`. Strata 03 and all of the held-out subjects of the fold; 0.5 = the fixed threshold, dev = the EER threshold of the common dev of the model.

| arm | fold | seed | in the ten | restored epoch changed | 03 ACER 0.5 (p4) | 03 ACER dev (p4) | 03 AUROC (p4) | all ACER 0.5 (p4) | all ACER dev (p4) | all AUROC (p4) | 03 ACER 0.5 (p12) | 03 ACER dev (p12) | 03 AUROC (p12) | all ACER 0.5 (p12) | all ACER dev (p12) | all AUROC (p12) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B | 1 | 42 | yes | yes | 0.5000 | 0.2937 | 0.6134 | 0.5000 | 0.2650 | 0.7389 | 0.2224 | 0.4045 | 0.8867 | 0.1618 | 0.3699 | 0.9439 |
| B | 1 | 43 | no | no | 0.4715 | 0.4680 | 0.7812 | 0.3234 | 0.3187 | 0.8948 | 0.4715 | 0.4680 | 0.7812 | 0.3234 | 0.3187 | 0.8948 |
| B | 1 | 44 | no | yes | 0.4970 | 0.5020 | 0.3809 | 0.3565 | 0.3643 | 0.6400 | 0.2396 | 0.0954 | 0.9751 | 0.2167 | 0.0696 | 0.9840 |
| B | 2 | 42 | no | no | 0.3643 | 0.3708 | 0.7238 | 0.2757 | 0.2861 | 0.8368 | 0.3643 | 0.3708 | 0.7238 | 0.2757 | 0.2861 | 0.8368 |
| B | 2 | 43 | no | no | 0.4650 | 0.4496 | 0.4744 | 0.3479 | 0.3357 | 0.6701 | 0.4650 | 0.4496 | 0.4744 | 0.3479 | 0.3357 | 0.6701 |
| B | 2 | 44 | no | no | 0.2219 | 0.2015 | 0.9779 | 0.1659 | 0.1508 | 0.9865 | 0.2219 | 0.2015 | 0.9779 | 0.1659 | 0.1508 | 0.9865 |
| B | 3 | 42 | no | no | 0.3154 | 0.3055 | 0.6672 | 0.2091 | 0.2010 | 0.8298 | 0.3154 | 0.3055 | 0.6672 | 0.2091 | 0.2010 | 0.8298 |
| B | 3 | 43 | no | no | 0.3726 | 0.2869 | 0.5781 | 0.2555 | 0.1857 | 0.7776 | 0.3726 | 0.2869 | 0.5781 | 0.2555 | 0.1857 | 0.7776 |
| B | 3 | 44 | no | yes | 0.6554 | 0.5711 | 0.3959 | 0.7265 | 0.6312 | 0.2781 | 0.3469 | 0.3226 | 0.5634 | 0.2365 | 0.2131 | 0.7523 |
| B | 4 | 42 | no | no | 0.5710 | 0.4980 | 0.5357 | 0.3936 | 0.3274 | 0.7502 | 0.5710 | 0.4980 | 0.5357 | 0.3936 | 0.3274 | 0.7502 |
| B | 4 | 43 | no | yes | 0.6316 | 0.4686 | 0.2230 | 0.4276 | 0.2874 | 0.5968 | 0.6493 | 0.5486 | 0.4227 | 0.4635 | 0.3508 | 0.7043 |
| B | 4 | 44 | yes | yes | 0.5000 | 0.7290 | 0.1344 | 0.5000 | 0.5078 | 0.4672 | 0.5950 | 0.6196 | 0.4882 | 0.3916 | 0.4239 | 0.7329 |
| A* | 1 | 42 | no | no | 0.1243 | 0.1359 | 0.9647 | 0.0839 | 0.0918 | 0.9828 | 0.1243 | 0.1359 | 0.9647 | 0.0839 | 0.0918 | 0.9828 |
| A* | 1 | 43 | no | no | 0.0281 | 0.0242 | 0.9909 | 0.0513 | 0.0435 | 0.9921 | 0.0281 | 0.0242 | 0.9909 | 0.0513 | 0.0435 | 0.9921 |
| A* | 1 | 44 | no | no | 0.1972 | 0.2164 | 0.8401 | 0.1428 | 0.1609 | 0.8963 | 0.1972 | 0.2164 | 0.8401 | 0.1428 | 0.1609 | 0.8963 |
| A* | 2 | 42 | no | no | 0.1216 | 0.1601 | 0.9502 | 0.0937 | 0.1203 | 0.9701 | 0.1216 | 0.1601 | 0.9502 | 0.0937 | 0.1203 | 0.9701 |
| A* | 2 | 43 | yes | yes | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0814 | 0.1068 | 0.9647 | 0.0613 | 0.0802 | 0.9776 |
| A* | 2 | 44 | yes | yes | 0.5000 | 0.5593 | 0.2646 | 0.5000 | 0.5546 | 0.2782 | 0.0693 | 0.1082 | 0.9804 | 0.0539 | 0.0859 | 0.9872 |
| A* | 3 | 42 | yes | yes | 0.5000 | 0.4254 | 0.5029 | 0.5000 | 0.4723 | 0.5255 | 0.0903 | 0.1102 | 0.9775 | 0.0657 | 0.0898 | 0.9879 |
| A* | 3 | 43 | no | no | 0.2333 | 0.2351 | 0.9113 | 0.1836 | 0.1850 | 0.9534 | 0.2333 | 0.2351 | 0.9113 | 0.1836 | 0.1850 | 0.9534 |
| A* | 3 | 44 | yes | yes | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0613 | 0.2495 | 0.9935 | 0.0526 | 0.2222 | 0.9902 |
| A* | 4 | 42 | no | no | 0.1270 | 0.2407 | 0.9714 | 0.1763 | 0.1911 | 0.9271 | 0.1270 | 0.2407 | 0.9714 | 0.1763 | 0.1911 | 0.9271 |
| A* | 4 | 43 | no | no | 0.2251 | 0.2238 | 0.8788 | 0.1777 | 0.1767 | 0.9227 | 0.2251 | 0.2238 | 0.8788 | 0.1777 | 0.1767 | 0.9227 |
| A* | 4 | 44 | yes | yes | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.1233 | 0.1961 | 0.9797 | 0.1023 | 0.1577 | 0.9599 |
| B' | 1 | 42 | no | yes | 0.7216 | 0.6112 | 0.1698 | 0.4933 | 0.4445 | 0.5029 | 0.2942 | 0.2831 | 0.8475 | 0.1975 | 0.1879 | 0.9240 |
| B' | 1 | 43 | no | no | 0.5057 | 0.5239 | 0.4761 | 0.4131 | 0.4173 | 0.5857 | 0.5057 | 0.5239 | 0.4761 | 0.4131 | 0.4173 | 0.5857 |
| B' | 1 | 44 | yes | yes | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5170 | 0.5170 | 0.6939 | 0.3510 | 0.3513 | 0.8430 |
| B' | 2 | 42 | no | no | 0.0994 | 0.3794 | 0.9685 | 0.1241 | 0.3122 | 0.9281 | 0.0994 | 0.3794 | 0.9685 | 0.1241 | 0.3122 | 0.9281 |
| B' | 2 | 43 | no | no | 0.4986 | 0.4534 | 0.3946 | 0.3759 | 0.3394 | 0.6391 | 0.4986 | 0.4534 | 0.3946 | 0.3759 | 0.3394 | 0.6391 |
| B' | 2 | 44 | yes | yes | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.4105 | 0.4342 | 0.6486 | 0.3097 | 0.3288 | 0.7606 |
| B' | 3 | 42 | no | yes | 0.4593 | 0.4639 | 0.5306 | 0.3314 | 0.3253 | 0.7411 | 0.2641 | 0.2660 | 0.7624 | 0.1677 | 0.1693 | 0.8763 |
| B' | 3 | 43 | no | no | 0.3984 | 0.3508 | 0.4509 | 0.2715 | 0.2704 | 0.6528 | 0.3984 | 0.3508 | 0.4509 | 0.2715 | 0.2704 | 0.6528 |
| B' | 3 | 44 | no | no | 0.3228 | 0.3185 | 0.5907 | 0.2031 | 0.2006 | 0.7675 | 0.3228 | 0.3185 | 0.5907 | 0.2031 | 0.2006 | 0.7675 |
| B' | 4 | 42 | no | no | 0.2207 | 0.2144 | 0.7858 | 0.1606 | 0.1698 | 0.8717 | 0.2207 | 0.2144 | 0.7858 | 0.1606 | 0.1698 | 0.8717 |
| B' | 4 | 43 | no | no | 0.5756 | 0.5536 | 0.1470 | 0.3959 | 0.3761 | 0.4826 | 0.5756 | 0.5536 | 0.1470 | 0.3959 | 0.3761 | 0.4826 |
| B' | 4 | 44 | yes | yes | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5703 | 0.6534 | 0.2515 | 0.3679 | 0.4996 | 0.6188 |

- Same test values and dev threshold in both runs (checked): all 21 models whose restored epoch did not change.
- The differences below come from the 15 models whose restored epoch changed ("changed" below).
- Stratum 03, ACER at 0.5, patience 12 vs 4: lower in 11 of 36 (11 of the 15 changed, 7 of the ten).
- Stratum 03, ACER at 0.5, patience 12 vs 4: equal in 21 of 36 (0 of the 15 changed, 0 of the ten).
- Stratum 03, ACER at 0.5, patience 12 vs 4: higher in 4 of 36 (4 of the 15 changed, 3 of the ten).
- Stratum 03, ACER at the dev threshold, patience 12 vs 4: lower in 11 of 36 (11 of the 15 changed, 7 of the ten).
- Stratum 03, ACER at the dev threshold, patience 12 vs 4: equal in 21 of 36 (0 of the 15 changed, 0 of the ten).
- Stratum 03, ACER at the dev threshold, patience 12 vs 4: higher in 4 of 36 (4 of the 15 changed, 3 of the ten).
- Stratum 03, AUROC, patience 12 vs 4: lower in 1 of 36 (1 of the 15 changed, 1 of the ten).
- Stratum 03, AUROC, patience 12 vs 4: equal in 21 of 36 (0 of the 15 changed, 0 of the ten).
- Stratum 03, AUROC, patience 12 vs 4: higher in 14 of 36 (14 of the 15 changed, 9 of the ten).
- Stratum all, ACER at 0.5, patience 12 vs 4: lower in 14 of 36 (14 of the 15 changed, 10 of the ten).
- Stratum all, ACER at 0.5, patience 12 vs 4: equal in 21 of 36 (0 of the 15 changed, 0 of the ten).
- Stratum all, ACER at 0.5, patience 12 vs 4: higher in 1 of 36 (1 of the 15 changed, 0 of the ten).
- Stratum all, ACER at the dev threshold, patience 12 vs 4: lower in 13 of 36 (13 of the 15 changed, 9 of the ten).
- Stratum all, ACER at the dev threshold, patience 12 vs 4: equal in 21 of 36 (0 of the 15 changed, 0 of the ten).
- Stratum all, ACER at the dev threshold, patience 12 vs 4: higher in 2 of 36 (2 of the 15 changed, 1 of the ten).
- Stratum all, AUROC, patience 12 vs 4: lower in 0 of 36 (0 of the 15 changed, 0 of the ten).
- Stratum all, AUROC, patience 12 vs 4: equal in 21 of 36 (0 of the 15 changed, 0 of the ten).
- Stratum all, AUROC, patience 12 vs 4: higher in 15 of 36 (15 of the 15 changed, 10 of the ten).
