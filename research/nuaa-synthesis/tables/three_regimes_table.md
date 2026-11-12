# Annex: the three regimes on the same 36 LivenessNet models of the session ablation design (patience 4; patience 12 with ReduceLROnPlateau; patience 12 with a constant learning rate)

p4 = run `20261106-SESSABLRUN-d999458` (early-stopping patience 4, ReduceLROnPlateau), p12 = run `20261109-SESSABLRUN-6a50265` (patience 12, ReduceLROnPlateau), p12c = run `20261111-SESSABLRUN-8568ffa` (patience 12, constant learning rate 0.001); the same 36 models, data and seeds in the three runs. Train and dev from the public comparisons `20261109-PATVER-4f2b925` (p4 against p12: `nuaa-patience-verification/comparison/comparison.csv`, `nuaa-patience-verification/comparison/curves.csv`) and `20261111-LRCTRL-ff53279` (p12 against p12c: `nuaa-constant-lr-control/comparison/comparison.csv`, `nuaa-constant-lr-control/comparison/curves.csv`), checked by `nuaa_synthesis.py`: every value rebuilt from the epoch curves with the criteria of the comparison aggregates (restored epoch = first epoch of minimum dev val_loss; epochs run = the early-stopping replay; degenerate: restored_epoch in {1, 2} and val_acc_at_best = 0.5 (within 1e-6): the rule written in the delivery of the session ablation run, re-applied here; recovery: first epoch with val_accuracy >= 0.75 and val_loss < 1.0 (dev only)), equal to the JSON rows and to the training cards of the three public aggregates; the p12 columns of the second comparison are required to equal those of the first (the same run), and the curves of p12 and p12c to be identical over the epochs at the recipe rate of p12 (shared window: from epoch 1 to the epoch at whose end p12 first reduced the rate; all 12 when never reduced), and the rate of p12c to be 0.0010000000474974513 at every epoch. "First recovery" = first epoch meeting the recovery criterion (— when none); "recovery holds at last epoch" = the recovery criterion met at the last epoch run; "epochs at 0.5 after first recovery" = dev epochs at 0.5 after the first recovery (a relapse; — without a recovery); "first reduced epoch (p12)" = first epoch of p12 at a reduced rate (— when never reduced). Checkpoint = restored epoch; dev = the common dev of the fold (sessions 01-02). "Same checkpoint (window) p12/p12c" = the same restored epoch inside the shared window, where the two trainings are identical: in the public aggregates `weights_sha256` and `dev_threshold_sha256` still differ between p12 and p12c in every model (different files), and the identity of these checkpoints is shown by the restored epoch inside the shared window, identical test values and an identical `dev_threshold`, all checked by `nuaa_synthesis.py`; a same restored epoch after the window is another training. Comparisons of p12c against p12 in the counts (higher, lower, equal) use the exact values, not the four decimals printed: "equal" best val_loss means the same value. Verdicts of the comparisons: positive, p4 against p12; mixed (no overall verdict), p12 against p12c.

## Train and dev, per model (p4, then p12, then p12c)

| arm | fold | seed | in the ten | epochs run (p4) | restored epoch (p4) | degenerate (p4) | first recovery (p4) | recovery holds at last epoch (p4) | epochs at 0.5 (p4) | epochs at 0.5 after first recovery (p4) | best val_loss (p4) | dev acc at ckpt (p4) | dev AUROC at ckpt (p4) | last lr (p4) | epochs run (p12) | restored epoch (p12) | degenerate (p12) | first recovery (p12) | recovery holds at last epoch (p12) | epochs at 0.5 (p12) | epochs at 0.5 after first recovery (p12) | best val_loss (p12) | dev acc at ckpt (p12) | dev AUROC at ckpt (p12) | last lr (p12) | epochs run (p12c) | restored epoch (p12c) | degenerate (p12c) | first recovery (p12c) | recovery holds at last epoch (p12c) | epochs at 0.5 (p12c) | epochs at 0.5 after first recovery (p12c) | best val_loss (p12c) | dev acc at ckpt (p12c) | dev AUROC at ckpt (p12c) | last lr (p12c) | first reduced epoch (p12) | shared epochs p12/p12c | restored epoch changed p4→p12 | restored epoch changed p12→p12c | same checkpoint (window) p12/p12c |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B | 1 | 42 | yes | 5 | 1 | yes | — | no | 5 | — | 7.0611 | 0.5000 | 0.9975 | 0.0005 | 12 | 10 | no | 8 | yes | 5 | 0 | 0.0378 | 0.9950 | 0.9960 | 0.00025 | 12 | 9 | no | 4 | yes | 3 | 0 | 0.0003 | 1.0000 | 1.0000 | 0.001 | 4 | 3 | yes | yes | no |
| B | 1 | 43 | no | 12 | 11 | no | 5 | yes | 3 | 0 | 0.0056 | 0.9975 | 1.0000 | 0.0005 | 12 | 11 | no | 5 | yes | 3 | 0 | 0.0056 | 0.9975 | 1.0000 | 0.0005 | 12 | 9 | no | 5 | yes | 3 | 0 | 0.0022 | 1.0000 | 1.0000 | 0.001 | 9 | 8 | no | yes | no |
| B | 1 | 44 | no | 10 | 6 | no | 6 | yes | 3 | 0 | 0.0298 | 0.9950 | 0.9985 | 0.0005 | 12 | 11 | no | 6 | yes | 3 | 0 | 0.0224 | 0.9975 | 1.0000 | 0.00025 | 12 | 11 | no | 6 | yes | 3 | 0 | 0.0007 | 1.0000 | 1.0000 | 0.001 | 9 | 8 | yes | no | no |
| B | 2 | 42 | no | 12 | 9 | no | 5 | no | 3 | 0 | 0.2308 | 0.9475 | 0.9888 | 0.000125 | 12 | 9 | no | 5 | no | 3 | 0 | 0.2308 | 0.9475 | 0.9888 | 0.000125 | 12 | 10 | no | 8 | yes | 4 | 0 | 0.0125 | 0.9950 | 1.0000 | 0.001 | 5 | 4 | no | yes | no |
| B | 2 | 43 | no | 11 | 7 | no | 7 | yes | 4 | 0 | 0.0011 | 1.0000 | 1.0000 | 0.00025 | 12 | 7 | no | 7 | yes | 4 | 0 | 0.0011 | 1.0000 | 1.0000 | 0.000125 | 12 | 7 | no | 6 | yes | 4 | 0 | 0.0009 | 1.0000 | 1.0000 | 0.001 | 4 | 3 | no | no | no |
| B | 2 | 44 | no | 12 | 10 | no | 6 | yes | 4 | 0 | 0.0030 | 0.9975 | 1.0000 | 0.00025 | 12 | 10 | no | 6 | yes | 4 | 0 | 0.0030 | 0.9975 | 1.0000 | 0.00025 | 12 | 9 | no | 5 | no | 5 | 1 | 0.0022 | 0.9975 | 1.0000 | 0.001 | 4 | 3 | no | yes | no |
| B | 3 | 42 | no | 12 | 12 | no | 6 | yes | 4 | 0 | 0.0168 | 1.0000 | 1.0000 | 0.00025 | 12 | 12 | no | 6 | yes | 4 | 0 | 0.0168 | 1.0000 | 1.0000 | 0.00025 | 12 | 12 | no | 7 | yes | 5 | 0 | 0.0039 | 1.0000 | 1.0000 | 0.001 | 6 | 5 | no | no | no |
| B | 3 | 43 | no | 12 | 10 | no | 6 | yes | 4 | 0 | 0.0531 | 0.9925 | 1.0000 | 0.0005 | 12 | 10 | no | 6 | yes | 4 | 0 | 0.0531 | 0.9925 | 1.0000 | 0.0005 | 12 | 9 | no | 6 | yes | 4 | 0 | 0.1083 | 0.9250 | 0.9999 | 0.001 | 9 | 8 | no | yes | no |
| B | 3 | 44 | no | 7 | 3 | no | — | no | 3 | — | 3.2731 | 0.2375 | 0.0950 | 0.0005 | 12 | 11 | no | 10 | yes | 3 | 0 | 0.0119 | 1.0000 | 1.0000 | 0.000125 | 12 | 10 | no | 7 | yes | 3 | 0 | 0.1457 | 0.9150 | 1.0000 | 0.001 | 6 | 5 | yes | yes | no |
| B | 4 | 42 | no | 12 | 11 | no | 6 | yes | 3 | 0 | 0.0005 | 1.0000 | 1.0000 | 0.00025 | 12 | 11 | no | 6 | yes | 3 | 0 | 0.0005 | 1.0000 | 1.0000 | 0.00025 | 12 | 11 | no | 6 | yes | 3 | 0 | 0.0001 | 1.0000 | 1.0000 | 0.001 | 6 | 5 | no | no | no |
| B | 4 | 43 | no | 10 | 6 | no | 6 | yes | 3 | 0 | 0.0013 | 1.0000 | 1.0000 | 0.0005 | 12 | 12 | no | 6 | yes | 3 | 0 | 0.0007 | 1.0000 | 1.0000 | 0.00025 | 12 | 6 | no | 6 | no | 3 | 0 | 0.0013 | 1.0000 | 1.0000 | 0.001 | 9 | 8 | yes | yes | no |
| B | 4 | 44 | yes | 6 | 2 | yes | — | no | 2 | — | 2.3605 | 0.5000 | 0.9652 | 0.0005 | 12 | 11 | no | 7 | yes | 2 | 0 | 0.0262 | 0.9925 | 1.0000 | 0.000125 | 12 | 11 | no | 10 | yes | 2 | 0 | 0.0037 | 1.0000 | 1.0000 | 0.001 | 5 | 4 | yes | no | no |
| A* | 1 | 42 | no | 12 | 10 | no | 6 | yes | 4 | 0 | 0.0104 | 0.9950 | 1.0000 | 0.001 | 12 | 10 | no | 6 | yes | 4 | 0 | 0.0104 | 0.9950 | 1.0000 | 0.001 | 12 | 10 | no | 6 | yes | 4 | 0 | 0.0104 | 0.9950 | 1.0000 | 0.001 | — | 12 | no | no | yes |
| A* | 1 | 43 | no | 12 | 9 | no | 7 | yes | 3 | 0 | 0.0165 | 0.9975 | 0.9995 | 0.0005 | 12 | 9 | no | 7 | yes | 3 | 0 | 0.0165 | 0.9975 | 0.9995 | 0.0005 | 12 | 9 | no | 7 | no | 3 | 0 | 0.0165 | 0.9975 | 0.9995 | 0.001 | 12 | 11 | no | no | yes |
| A* | 1 | 44 | no | 12 | 11 | no | 7 | yes | 2 | 0 | 0.0345 | 0.9875 | 0.9991 | 0.0005 | 12 | 11 | no | 7 | yes | 2 | 0 | 0.0345 | 0.9875 | 0.9991 | 0.0005 | 12 | 9 | no | 7 | yes | 2 | 0 | 0.0269 | 0.9975 | 0.9979 | 0.001 | 4 | 3 | no | yes | no |
| A* | 2 | 42 | no | 12 | 11 | no | 5 | yes | 3 | 0 | 0.0001 | 1.0000 | 1.0000 | 0.00025 | 12 | 11 | no | 5 | yes | 3 | 0 | 0.0001 | 1.0000 | 1.0000 | 0.00025 | 12 | 7 | no | 5 | yes | 3 | 0 | 0.0001 | 1.0000 | 1.0000 | 0.001 | 10 | 9 | no | yes | no |
| A* | 2 | 43 | yes | 5 | 1 | yes | — | no | 4 | — | 60.9868 | 0.5000 | 0.5000 | 0.0005 | 12 | 10 | no | 7 | yes | 4 | 0 | 0.0004 | 1.0000 | 1.0000 | 0.00025 | 12 | 9 | no | 6 | yes | 5 | 0 | 0.0051 | 1.0000 | 1.0000 | 0.001 | 4 | 3 | yes | yes | no |
| A* | 2 | 44 | yes | 5 | 1 | yes | — | no | 3 | — | 18.0934 | 0.5000 | 0.2114 | 0.0005 | 12 | 12 | no | 10 | yes | 3 | 0 | 0.2171 | 0.9975 | 0.9950 | 0.00025 | 12 | 10 | no | 5 | yes | 3 | 0 | 0.0001 | 1.0000 | 1.0000 | 0.001 | 4 | 3 | yes | yes | no |
| A* | 3 | 42 | yes | 5 | 1 | yes | — | no | 3 | — | 3.1296 | 0.5000 | 0.4690 | 0.0005 | 12 | 9 | no | 9 | yes | 3 | 0 | 0.3282 | 0.9425 | 0.9900 | 0.000125 | 12 | 10 | no | 6 | no | 2 | 0 | 0.0256 | 0.9900 | 1.0000 | 0.001 | 4 | 3 | yes | yes | no |
| A* | 3 | 43 | no | 12 | 12 | no | 6 | yes | 5 | 0 | 0.0088 | 0.9975 | 1.0000 | 0.0005 | 12 | 12 | no | 6 | yes | 5 | 0 | 0.0088 | 0.9975 | 1.0000 | 0.0005 | 12 | 12 | no | 6 | yes | 5 | 0 | 0.0976 | 0.9725 | 0.9978 | 0.001 | 9 | 8 | no | no | no |
| A* | 3 | 44 | yes | 5 | 1 | yes | — | no | 4 | — | 33.4090 | 0.5000 | 0.5000 | 0.0005 | 12 | 11 | no | 11 | no | 4 | 0 | 0.5239 | 0.8900 | 0.9895 | 0.000125 | 12 | 8 | no | 7 | yes | 4 | 0 | 0.2450 | 0.8750 | 1.0000 | 0.001 | 4 | 3 | yes | yes | no |
| A* | 4 | 42 | no | 12 | 10 | no | 10 | no | 3 | 0 | 0.8106 | 0.7975 | 0.9990 | 0.0005 | 12 | 10 | no | 10 | no | 3 | 0 | 0.8106 | 0.7975 | 0.9990 | 0.0005 | 12 | 11 | no | 9 | yes | 3 | 0 | 0.2092 | 0.8975 | 0.9715 | 0.001 | 5 | 4 | no | yes | no |
| A* | 4 | 43 | no | 12 | 9 | no | 7 | no | 4 | 0 | 0.0272 | 0.9950 | 0.9993 | 0.0005 | 12 | 9 | no | 7 | no | 4 | 0 | 0.0272 | 0.9950 | 0.9993 | 0.0005 | 12 | 9 | no | 7 | no | 4 | 0 | 0.0272 | 0.9950 | 0.9993 | 0.001 | 12 | 11 | no | no | yes |
| A* | 4 | 44 | yes | 5 | 1 | yes | — | no | 5 | — | 22.8274 | 0.5000 | 0.5000 | 0.0005 | 12 | 12 | no | 10 | yes | 5 | 0 | 0.3154 | 0.8650 | 0.9999 | 0.000125 | 12 | 8 | no | 6 | yes | 4 | 0 | 0.0123 | 0.9975 | 1.0000 | 0.001 | 4 | 3 | yes | yes | no |
| B' | 1 | 42 | no | 6 | 2 | no | 2 | no | 5 | 4 | 0.6101 | 0.7725 | 0.9937 | 0.0005 | 12 | 12 | no | 2 | yes | 5 | 4 | 0.0033 | 0.9975 | 1.0000 | 0.000125 | 12 | 10 | no | 2 | yes | 3 | 2 | 0.0144 | 0.9975 | 0.9997 | 0.001 | 5 | 4 | yes | yes | no |
| B' | 1 | 43 | no | 8 | 4 | no | 4 | no | 1 | 0 | 0.8647 | 0.9400 | 0.9899 | 0.0005 | 12 | 4 | no | 4 | no | 1 | 0 | 0.8647 | 0.9400 | 0.9899 | 0.000125 | 12 | 12 | no | 4 | yes | 1 | 0 | 0.0642 | 0.9675 | 1.0000 | 0.001 | 7 | 6 | no | yes | no |
| B' | 1 | 44 | yes | 5 | 1 | yes | — | no | 5 | — | 12.1449 | 0.5000 | 0.4975 | 0.0005 | 12 | 9 | no | 7 | no | 5 | 0 | 0.1895 | 0.9950 | 0.9950 | 0.000125 | 12 | 6 | no | 6 | yes | 5 | 0 | 0.3141 | 0.9975 | 0.9975 | 0.001 | 4 | 3 | yes | yes | no |
| B' | 2 | 42 | no | 12 | 11 | no | 11 | no | 7 | 0 | 0.8750 | 0.8775 | 0.9830 | 0.0005 | 12 | 11 | no | 11 | no | 7 | 0 | 0.8750 | 0.8775 | 0.9830 | 0.0005 | 12 | 10 | no | — | no | 7 | — | 1.6847 | 0.8550 | 0.8678 | 0.001 | 9 | 8 | no | yes | no |
| B' | 2 | 43 | no | 12 | 10 | no | 3 | yes | 3 | 1 | 0.0042 | 1.0000 | 1.0000 | 0.0005 | 12 | 10 | no | 3 | yes | 3 | 1 | 0.0042 | 1.0000 | 1.0000 | 0.0005 | 12 | 12 | no | 3 | yes | 3 | 1 | 0.0217 | 0.9950 | 1.0000 | 0.001 | 9 | 8 | no | yes | no |
| B' | 2 | 44 | yes | 6 | 2 | yes | — | no | 6 | — | 11.4259 | 0.5000 | 0.5000 | 0.0005 | 12 | 12 | no | 12 | yes | 7 | 0 | 0.0736 | 0.9875 | 0.9950 | 0.00025 | 12 | 12 | no | 7 | yes | 5 | 0 | 0.0160 | 0.9950 | 1.0000 | 0.001 | 5 | 4 | yes | no | no |
| B' | 3 | 42 | no | 5 | 1 | no | 1 | no | 3 | 3 | 0.3211 | 0.8325 | 1.0000 | 0.0005 | 12 | 11 | no | 1 | yes | 3 | 3 | 0.0516 | 0.9900 | 0.9975 | 0.000125 | 12 | 4 | no | 1 | yes | 5 | 5 | 0.0260 | 0.9900 | 1.0000 | 0.001 | 4 | 3 | yes | yes | no |
| B' | 3 | 43 | no | 12 | 9 | no | 7 | yes | 5 | 0 | 0.0011 | 1.0000 | 1.0000 | 0.0005 | 12 | 9 | no | 7 | yes | 5 | 0 | 0.0011 | 1.0000 | 1.0000 | 0.0005 | 12 | 9 | no | 7 | no | 5 | 0 | 0.0011 | 1.0000 | 1.0000 | 0.001 | 12 | 11 | no | no | yes |
| B' | 3 | 44 | no | 12 | 9 | no | 7 | yes | 6 | 0 | 0.0341 | 0.9975 | 1.0000 | 0.00025 | 12 | 9 | no | 7 | yes | 6 | 0 | 0.0341 | 0.9975 | 1.0000 | 0.00025 | 12 | 9 | no | 8 | no | 7 | 0 | 0.0814 | 0.9850 | 1.0000 | 0.001 | 6 | 5 | no | no | no |
| B' | 4 | 42 | no | 6 | 2 | no | 1 | no | 2 | 2 | 0.1375 | 0.9175 | 0.9966 | 0.0005 | 12 | 2 | no | 1 | no | 3 | 3 | 0.1375 | 0.9175 | 0.9966 | 0.0000625 | 12 | 11 | no | 1 | no | 3 | 3 | 0.0968 | 0.9700 | 0.9940 | 0.001 | 5 | 4 | no | yes | no |
| B' | 4 | 43 | no | 7 | 3 | no | 3 | no | 3 | 2 | 0.0142 | 1.0000 | 1.0000 | 0.0005 | 12 | 3 | no | 3 | yes | 3 | 2 | 0.0142 | 1.0000 | 1.0000 | 0.0000625 | 12 | 3 | no | 3 | yes | 4 | 3 | 0.0142 | 1.0000 | 1.0000 | 0.001 | 6 | 5 | no | no | yes |
| B' | 4 | 44 | yes | 5 | 1 | yes | — | no | 4 | — | 17.2907 | 0.5000 | 0.5000 | 0.0005 | 12 | 11 | no | 10 | yes | 4 | 0 | 0.0851 | 0.9850 | 0.9988 | 0.00025 | 12 | 10 | no | 10 | no | 4 | 0 | 0.2785 | 0.8900 | 0.9831 | 0.001 | 4 | 3 | yes | yes | no |

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

## Test, per model (not used to choose or judge the recipe; p4, then p12, then p12c)

From the public `aggregate.json` of the three runs (`nuaa-session-ablation-run/tables/aggregate.json`, `nuaa-patience-verification/tables/aggregate.json`, `nuaa-constant-lr-control/tables/aggregate.json`): every count and rate checked against the fold schema and summed into the pooled rows of `session_ablation_table.md` and `session_ablation_table_constant_lr.md`. Strata 03 and all of the held-out subjects of the fold; 0.5 = the fixed threshold, dev = the EER threshold of the common dev of the model.

| arm | fold | seed | in the ten | restored epoch changed p4→p12 | restored epoch changed p12→p12c | same checkpoint (window) p12/p12c | 03 ACER 0.5 (p4) | 03 ACER dev (p4) | 03 AUROC (p4) | all ACER 0.5 (p4) | all ACER dev (p4) | all AUROC (p4) | 03 ACER 0.5 (p12) | 03 ACER dev (p12) | 03 AUROC (p12) | all ACER 0.5 (p12) | all ACER dev (p12) | all AUROC (p12) | 03 ACER 0.5 (p12c) | 03 ACER dev (p12c) | 03 AUROC (p12c) | all ACER 0.5 (p12c) | all ACER dev (p12c) | all AUROC (p12c) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B | 1 | 42 | yes | yes | yes | no | 0.5000 | 0.2937 | 0.6134 | 0.5000 | 0.2650 | 0.7389 | 0.2224 | 0.4045 | 0.8867 | 0.1618 | 0.3699 | 0.9439 | 0.1742 | 0.1914 | 0.8089 | 0.1153 | 0.1413 | 0.9106 |
| B | 1 | 43 | no | no | yes | no | 0.4715 | 0.4680 | 0.7812 | 0.3234 | 0.3187 | 0.8948 | 0.4715 | 0.4680 | 0.7812 | 0.3234 | 0.3187 | 0.8948 | 0.5350 | 0.5126 | 0.2599 | 0.3562 | 0.3465 | 0.6319 |
| B | 1 | 44 | no | yes | no | no | 0.4970 | 0.5020 | 0.3809 | 0.3565 | 0.3643 | 0.6400 | 0.2396 | 0.0954 | 0.9751 | 0.2167 | 0.0696 | 0.9840 | 0.4302 | 0.4719 | 0.8101 | 0.2921 | 0.3334 | 0.9069 |
| B | 2 | 42 | no | no | yes | no | 0.3643 | 0.3708 | 0.7238 | 0.2757 | 0.2861 | 0.8368 | 0.3643 | 0.3708 | 0.7238 | 0.2757 | 0.2861 | 0.8368 | 0.1571 | 0.1416 | 0.9423 | 0.1225 | 0.1076 | 0.9660 |
| B | 2 | 43 | no | no | no | no | 0.4650 | 0.4496 | 0.4744 | 0.3479 | 0.3357 | 0.6701 | 0.4650 | 0.4496 | 0.4744 | 0.3479 | 0.3357 | 0.6701 | 0.5007 | 0.3992 | 0.3768 | 0.3789 | 0.2986 | 0.5875 |
| B | 2 | 44 | no | no | yes | no | 0.2219 | 0.2015 | 0.9779 | 0.1659 | 0.1508 | 0.9865 | 0.2219 | 0.2015 | 0.9779 | 0.1659 | 0.1508 | 0.9865 | 0.2535 | 0.2461 | 0.8187 | 0.1943 | 0.1888 | 0.8815 |
| B | 3 | 42 | no | no | no | no | 0.3154 | 0.3055 | 0.6672 | 0.2091 | 0.2010 | 0.8298 | 0.3154 | 0.3055 | 0.6672 | 0.2091 | 0.2010 | 0.8298 | 0.2616 | 0.2671 | 0.8561 | 0.1914 | 0.1801 | 0.9265 |
| B | 3 | 43 | no | no | yes | no | 0.3726 | 0.2869 | 0.5781 | 0.2555 | 0.1857 | 0.7776 | 0.3726 | 0.2869 | 0.5781 | 0.2555 | 0.1857 | 0.7776 | 0.4109 | 0.3280 | 0.5977 | 0.2879 | 0.2191 | 0.7909 |
| B | 3 | 44 | no | yes | yes | no | 0.6554 | 0.5711 | 0.3959 | 0.7265 | 0.6312 | 0.2781 | 0.3469 | 0.3226 | 0.5634 | 0.2365 | 0.2131 | 0.7523 | 0.4053 | 0.2888 | 0.6817 | 0.2906 | 0.1890 | 0.8325 |
| B | 4 | 42 | no | no | no | no | 0.5710 | 0.4980 | 0.5357 | 0.3936 | 0.3274 | 0.7502 | 0.5710 | 0.4980 | 0.5357 | 0.3936 | 0.3274 | 0.7502 | 0.6577 | 0.5400 | 0.1991 | 0.4624 | 0.3571 | 0.5651 |
| B | 4 | 43 | no | yes | yes | no | 0.6316 | 0.4686 | 0.2230 | 0.4276 | 0.2874 | 0.5968 | 0.6493 | 0.5486 | 0.4227 | 0.4635 | 0.3508 | 0.7043 | 0.6316 | 0.4686 | 0.2230 | 0.4276 | 0.2874 | 0.5968 |
| B | 4 | 44 | yes | yes | no | no | 0.5000 | 0.7290 | 0.1344 | 0.5000 | 0.5078 | 0.4672 | 0.5950 | 0.6196 | 0.4882 | 0.3916 | 0.4239 | 0.7329 | 0.6905 | 0.6219 | 0.2757 | 0.4903 | 0.4341 | 0.6261 |
| A* | 1 | 42 | no | no | no | yes | 0.1243 | 0.1359 | 0.9647 | 0.0839 | 0.0918 | 0.9828 | 0.1243 | 0.1359 | 0.9647 | 0.0839 | 0.0918 | 0.9828 | 0.1243 | 0.1359 | 0.9647 | 0.0839 | 0.0918 | 0.9828 |
| A* | 1 | 43 | no | no | no | yes | 0.0281 | 0.0242 | 0.9909 | 0.0513 | 0.0435 | 0.9921 | 0.0281 | 0.0242 | 0.9909 | 0.0513 | 0.0435 | 0.9921 | 0.0281 | 0.0242 | 0.9909 | 0.0513 | 0.0435 | 0.9921 |
| A* | 1 | 44 | no | no | yes | no | 0.1972 | 0.2164 | 0.8401 | 0.1428 | 0.1609 | 0.8963 | 0.1972 | 0.2164 | 0.8401 | 0.1428 | 0.1609 | 0.8963 | 0.3724 | 0.2892 | 0.5214 | 0.3124 | 0.2393 | 0.7368 |
| A* | 2 | 42 | no | no | yes | no | 0.1216 | 0.1601 | 0.9502 | 0.0937 | 0.1203 | 0.9701 | 0.1216 | 0.1601 | 0.9502 | 0.0937 | 0.1203 | 0.9701 | 0.1061 | 0.3288 | 0.9700 | 0.0799 | 0.2453 | 0.9823 |
| A* | 2 | 43 | yes | yes | yes | no | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0814 | 0.1068 | 0.9647 | 0.0613 | 0.0802 | 0.9776 | 0.1468 | 0.0769 | 0.9790 | 0.1152 | 0.0595 | 0.9866 |
| A* | 2 | 44 | yes | yes | yes | no | 0.5000 | 0.5593 | 0.2646 | 0.5000 | 0.5546 | 0.2782 | 0.0693 | 0.1082 | 0.9804 | 0.0539 | 0.0859 | 0.9872 | 0.0554 | 0.0617 | 0.9957 | 0.0445 | 0.0470 | 0.9963 |
| A* | 3 | 42 | yes | yes | yes | no | 0.5000 | 0.4254 | 0.5029 | 0.5000 | 0.4723 | 0.5255 | 0.0903 | 0.1102 | 0.9775 | 0.0657 | 0.0898 | 0.9879 | 0.3021 | 0.1332 | 0.9729 | 0.2493 | 0.1058 | 0.9818 |
| A* | 3 | 43 | no | no | no | no | 0.2333 | 0.2351 | 0.9113 | 0.1836 | 0.1850 | 0.9534 | 0.2333 | 0.2351 | 0.9113 | 0.1836 | 0.1850 | 0.9534 | 0.2406 | 0.3432 | 0.8745 | 0.1893 | 0.2917 | 0.9362 |
| A* | 3 | 44 | yes | yes | yes | no | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0613 | 0.2495 | 0.9935 | 0.0526 | 0.2222 | 0.9902 | 0.0665 | 0.0913 | 0.9868 | 0.0462 | 0.0741 | 0.9933 |
| A* | 4 | 42 | no | no | yes | no | 0.1270 | 0.2407 | 0.9714 | 0.1763 | 0.1911 | 0.9271 | 0.1270 | 0.2407 | 0.9714 | 0.1763 | 0.1911 | 0.9271 | 0.2223 | 0.2273 | 0.9866 | 0.1859 | 0.1906 | 0.9581 |
| A* | 4 | 43 | no | no | no | yes | 0.2251 | 0.2238 | 0.8788 | 0.1777 | 0.1767 | 0.9227 | 0.2251 | 0.2238 | 0.8788 | 0.1777 | 0.1767 | 0.9227 | 0.2251 | 0.2238 | 0.8788 | 0.1777 | 0.1767 | 0.9227 |
| A* | 4 | 44 | yes | yes | yes | no | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.1233 | 0.1961 | 0.9797 | 0.1023 | 0.1577 | 0.9599 | 0.1420 | 0.1450 | 0.9802 | 0.1146 | 0.1170 | 0.9714 |
| B' | 1 | 42 | no | yes | yes | no | 0.7216 | 0.6112 | 0.1698 | 0.4933 | 0.4445 | 0.5029 | 0.2942 | 0.2831 | 0.8475 | 0.1975 | 0.1879 | 0.9240 | 0.5008 | 0.4507 | 0.4740 | 0.3628 | 0.2963 | 0.6923 |
| B' | 1 | 43 | no | no | yes | no | 0.5057 | 0.5239 | 0.4761 | 0.4131 | 0.4173 | 0.5857 | 0.5057 | 0.5239 | 0.4761 | 0.4131 | 0.4173 | 0.5857 | 0.4649 | 0.5477 | 0.4227 | 0.3132 | 0.3820 | 0.6517 |
| B' | 1 | 44 | yes | yes | yes | no | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5170 | 0.5170 | 0.6939 | 0.3510 | 0.3513 | 0.8430 | 0.5969 | 0.5450 | 0.3621 | 0.4353 | 0.4279 | 0.5552 |
| B' | 2 | 42 | no | no | yes | no | 0.0994 | 0.3794 | 0.9685 | 0.1241 | 0.3122 | 0.9281 | 0.0994 | 0.3794 | 0.9685 | 0.1241 | 0.3122 | 0.9281 | 0.2403 | 0.5244 | 0.6699 | 0.2629 | 0.4765 | 0.6590 |
| B' | 2 | 43 | no | no | yes | no | 0.4986 | 0.4534 | 0.3946 | 0.3759 | 0.3394 | 0.6391 | 0.4986 | 0.4534 | 0.3946 | 0.3759 | 0.3394 | 0.6391 | 0.5706 | 0.4258 | 0.4985 | 0.4391 | 0.3181 | 0.6793 |
| B' | 2 | 44 | yes | yes | no | no | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.4105 | 0.4342 | 0.6486 | 0.3097 | 0.3288 | 0.7606 | 0.4822 | 0.3712 | 0.6014 | 0.3672 | 0.2780 | 0.7655 |
| B' | 3 | 42 | no | yes | yes | no | 0.4593 | 0.4639 | 0.5306 | 0.3314 | 0.3253 | 0.7411 | 0.2641 | 0.2660 | 0.7624 | 0.1677 | 0.1693 | 0.8763 | 0.3951 | 0.4193 | 0.5157 | 0.2690 | 0.2891 | 0.7327 |
| B' | 3 | 43 | no | no | no | yes | 0.3984 | 0.3508 | 0.4509 | 0.2715 | 0.2704 | 0.6528 | 0.3984 | 0.3508 | 0.4509 | 0.2715 | 0.2704 | 0.6528 | 0.3984 | 0.3508 | 0.4509 | 0.2715 | 0.2704 | 0.6528 |
| B' | 3 | 44 | no | no | no | no | 0.3228 | 0.3185 | 0.5907 | 0.2031 | 0.2006 | 0.7675 | 0.3228 | 0.3185 | 0.5907 | 0.2031 | 0.2006 | 0.7675 | 0.2712 | 0.2710 | 0.7566 | 0.1706 | 0.1736 | 0.8682 |
| B' | 4 | 42 | no | no | yes | no | 0.2207 | 0.2144 | 0.7858 | 0.1606 | 0.1698 | 0.8717 | 0.2207 | 0.2144 | 0.7858 | 0.1606 | 0.1698 | 0.8717 | 0.4843 | 0.3952 | 0.6259 | 0.3412 | 0.3072 | 0.7667 |
| B' | 4 | 43 | no | no | no | yes | 0.5756 | 0.5536 | 0.1470 | 0.3959 | 0.3761 | 0.4826 | 0.5756 | 0.5536 | 0.1470 | 0.3959 | 0.3761 | 0.4826 | 0.5756 | 0.5536 | 0.1470 | 0.3959 | 0.3761 | 0.4826 |
| B' | 4 | 44 | yes | yes | yes | no | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5703 | 0.6534 | 0.2515 | 0.3679 | 0.4996 | 0.6188 | 0.5881 | 0.3931 | 0.5270 | 0.3860 | 0.3305 | 0.7471 |

- Same test values and dev threshold with p12 and p12c (checked): the 5 models with the same weights.
- Stratum 03, ACER at 0.5, p12c vs p12: lower 8, equal 5, higher 23 of 36.
- Stratum 03, ACER at the dev threshold, p12c vs p12: lower 15, equal 5, higher 16 of 36.
- Stratum 03, AUROC, p12c vs p12: lower 19, equal 5, higher 12 of 36.
- Stratum all, ACER at 0.5, p12c vs p12: lower 9, equal 5, higher 22 of 36.
- Stratum all, ACER at the dev threshold, p12c vs p12: lower 16, equal 5, higher 15 of 36.
- Stratum all, AUROC, p12c vs p12: lower 16, equal 5, higher 15 of 36.
- Verdicts, 84 per run, consistent: p4 1, p12 6, p12c 10.
- Verdicts that differ: p4 vs p12 7 of 84, p12 vs p12c 12 of 84 (list in `session_ablation_table_constant_lr.md`).
- Primary verdict (A* - B, stratum 03, auroc): p4 inconclusive, p12 inconclusive, p12c inconclusive.
- Primary verdict (A* - B, stratum 03, acer_dev): p4 inconclusive, p12 inconclusive, p12c inconclusive.
