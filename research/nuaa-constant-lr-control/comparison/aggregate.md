# Control run constant_lr against the reference run patience12 (20261111-LRCTRL-ff53279)

The patience12 run 20261109-SESSABLRUN-6a50265 (early_stopping_patience 12, PASS, 36 models); constant_lr run 20261111-SESSABLRUN-8568ffa (early_stopping_patience 12, PASS, 36 complete models). The one change: training.scheduler from ReduceLROnPlateau(monitor=val_loss, factor=0.5, patience=2, min_lr=1e-6) to constant (no ReduceLROnPlateau; the initial learning rate for every epoch; the learning rate in effect is recorded in history.lr at the end of every epoch). Reading rule fixed before the run (request of 2026-11-10); criteria of the previous package: restored_epoch in {1, 2} and val_acc_at_best = 0.5 (within 1e-6): the rule written in the delivery of the session ablation run, re-applied here; recovery: first epoch with val_accuracy >= 0.75 and val_loss < 1.0 (dev only). No training, no inference, no test score read. Figures: figure_ten.png, figure_no_recovery.png, figure_val_accuracy_all.png written.

## (a) Shared histories

validity: for every model the history of the control run must be identical, in loss, accuracy, val_loss, val_accuracy and lr (difference 0), to the reference run over the epochs recorded at the recipe rate in that run (Keras records the rate of an epoch before reducing it: epochs 1 to 3 when the first lower rate is recorded at epoch 4; all 12 for a model never reduced), and the rate of the control run must be the recipe rate at every epoch; otherwise the largest difference (or the models with another rate) is reported and the comparison is declared not valid: no verdict.

Window of the shared histories (parent_epochs_before_first_lr_reduction): for each model the first epochs of the patience12 run with history.lr equal to 0.0010000000474974513 (the recipe rate as recorded, compared exactly), up to the epoch at whose end the first reduction happened; all epochs for a model never reduced. Shared epochs per model: 3 to 12; first reduced epoch between 4 and 12; never reduced: 1 of 36.
Learning rate of the constant_lr run equal to 0.0010000000474974513 at every epoch: 36 of 36 models; values recorded: 0.0010000000474974513.
Models compared: 36; identical shared histories: 36 of 36; valid: yes.
Largest absolute difference per field: loss 0.0, accuracy 0.0, val_loss 0.0, val_accuracy 0.0, lr 0.0.

## (b) Primary outcome

primary outcome on train/dev: among the ten original degenerate trainings of the patience-4 run, the number that recover within 12 epochs with the constant rate, and among the 36 the number degenerate (same criteria). If all ten recover and none of the 36 is degenerate, the recovery of the ten is reproduced without the reduction: consistent with the extra epochs being sufficient in these trainings. If at least one of the ten does not recover, the outcome is consistent with a contribution of the reduction in these trainings, and the models are listed one by one. A mixed outcome (for example all ten recover but another model becomes degenerate or loses its recovery) is reported as such, model by model, with no overall verdict.

**Verdict: mixed (no overall verdict).** Mixed outcome, no overall verdict: the ten original degenerate trainings all recover within 12 epochs with the constant learning rate (0.001), but degenerate among the 36: none; recovery of the patience12 run not reproduced outside the ten: B' fold 2 seed 42. Reported model by model.

The ten original degenerate trainings: 10 of 10 complete recover within 12 epochs (first recovery: A\* fold 2 seed 43 6, A\* fold 2 seed 44 5, A\* fold 3 seed 42 6, A\* fold 3 seed 44 7, A\* fold 4 seed 44 6, B fold 1 seed 42 4, B fold 4 seed 44 10, B' fold 1 seed 44 6, B' fold 2 seed 44 7, B' fold 4 seed 44 10).
Without recovery among the ten: none.
Degenerate among the complete models of the constant_lr run: none.
Without recovery outside the ten (listed separately): B' fold 2 seed 42.
Without recovery in the patience12 run: none.
Recovery of the patience12 run not reproduced outside the ten: B' fold 2 seed 42.

| the ten, model by model | first recovery (patience12) | first recovery (constant_lr) | recovered at epoch 12 (constant_lr) | restored (constant_lr) | degenerate (constant_lr) |
|---|---|---|---|---|---|
| A\* fold 2 seed 43 | 7 | 6 | yes | 9 | no |
| A\* fold 2 seed 44 | 10 | 5 | yes | 10 | no |
| A\* fold 3 seed 42 | 9 | 6 | no | 10 | no |
| A\* fold 3 seed 44 | 11 | 7 | yes | 8 | no |
| A\* fold 4 seed 44 | 10 | 6 | yes | 8 | no |
| B fold 1 seed 42 | 8 | 4 | yes | 9 | no |
| B fold 4 seed 44 | 7 | 10 | yes | 11 | no |
| B' fold 1 seed 44 | 7 | 6 | yes | 6 | no |
| B' fold 2 seed 44 | 12 | 7 | yes | 12 | no |
| B' fold 4 seed 44 | 10 | 10 | no | 10 | no |

## (c) Secondary outcomes

secondary outcomes per model, reference against control run: first-recovery epoch, whether the recovery criterion still holds at the last epoch, restored epoch and whether it changes, best val_loss, dev accuracy and dev AUROC at the checkpoint (from the dev scores, checked row by row), epochs at 0.5, the first reduced epoch of the reference run and the rate of the control run; B fold 3 seed 44 and B' fold 4 seed 42 reported apart.

Models with a recovery: patience12 36 of 36, constant_lr run 35 of 36 complete.
Recovery criterion still met at epoch 12: patience12 28, constant_lr 26; restored epoch changed in 23 models.

| model | ten | epochs run | restored | degenerate | first recovery | best val_loss | dev acc | dev AUROC | epochs at 0.5 | lr last | shared identical | shared epochs | first reduced epoch | recovered at last epoch | lr constant |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B fold 1 seed 42 (patience12) | yes | 12 | 10 | no | 8 | 0.0378 | 0.9950 | 0.9960 | 5 | 0.00025 | - | 3 | 4 | yes | - |
| B fold 1 seed 42 (constant_lr) | yes | 12 | 9 | no | 4 | 0.0003 | 1.0000 | 1.0000 | 3 | 0.001 | yes | - | - | yes | yes |
| B fold 1 seed 43 (patience12) | - | 12 | 11 | no | 5 | 0.0056 | 0.9975 | 1.0000 | 3 | 0.0005 | - | 8 | 9 | yes | - |
| B fold 1 seed 43 (constant_lr) | - | 12 | 9 | no | 5 | 0.0022 | 1.0000 | 1.0000 | 3 | 0.001 | yes | - | - | yes | yes |
| B fold 1 seed 44 (patience12) | - | 12 | 11 | no | 6 | 0.0224 | 0.9975 | 1.0000 | 3 | 0.00025 | - | 8 | 9 | yes | - |
| B fold 1 seed 44 (constant_lr) | - | 12 | 11 | no | 6 | 0.0007 | 1.0000 | 1.0000 | 3 | 0.001 | yes | - | - | yes | yes |
| B fold 2 seed 42 (patience12) | - | 12 | 9 | no | 5 | 0.2308 | 0.9475 | 0.9888 | 3 | 0.000125 | - | 4 | 5 | no | - |
| B fold 2 seed 42 (constant_lr) | - | 12 | 10 | no | 8 | 0.0125 | 0.9950 | 1.0000 | 4 | 0.001 | yes | - | - | yes | yes |
| B fold 2 seed 43 (patience12) | - | 12 | 7 | no | 7 | 0.0011 | 1.0000 | 1.0000 | 4 | 0.000125 | - | 3 | 4 | yes | - |
| B fold 2 seed 43 (constant_lr) | - | 12 | 7 | no | 6 | 0.0009 | 1.0000 | 1.0000 | 4 | 0.001 | yes | - | - | yes | yes |
| B fold 2 seed 44 (patience12) | - | 12 | 10 | no | 6 | 0.0030 | 0.9975 | 1.0000 | 4 | 0.00025 | - | 3 | 4 | yes | - |
| B fold 2 seed 44 (constant_lr) | - | 12 | 9 | no | 5 | 0.0022 | 0.9975 | 1.0000 | 5 | 0.001 | yes | - | - | no | yes |
| B fold 3 seed 42 (patience12) | - | 12 | 12 | no | 6 | 0.0168 | 1.0000 | 1.0000 | 4 | 0.00025 | - | 5 | 6 | yes | - |
| B fold 3 seed 42 (constant_lr) | - | 12 | 12 | no | 7 | 0.0039 | 1.0000 | 1.0000 | 5 | 0.001 | yes | - | - | yes | yes |
| B fold 3 seed 43 (patience12) | - | 12 | 10 | no | 6 | 0.0531 | 0.9925 | 1.0000 | 4 | 0.0005 | - | 8 | 9 | yes | - |
| B fold 3 seed 43 (constant_lr) | - | 12 | 9 | no | 6 | 0.1083 | 0.9250 | 0.9999 | 4 | 0.001 | yes | - | - | yes | yes |
| B fold 3 seed 44 (patience12) | - | 12 | 11 | no | 10 | 0.0119 | 1.0000 | 1.0000 | 3 | 0.000125 | - | 5 | 6 | yes | - |
| B fold 3 seed 44 (constant_lr) | - | 12 | 10 | no | 7 | 0.1457 | 0.9150 | 1.0000 | 3 | 0.001 | yes | - | - | yes | yes |
| B fold 4 seed 42 (patience12) | - | 12 | 11 | no | 6 | 0.0005 | 1.0000 | 1.0000 | 3 | 0.00025 | - | 5 | 6 | yes | - |
| B fold 4 seed 42 (constant_lr) | - | 12 | 11 | no | 6 | 0.0001 | 1.0000 | 1.0000 | 3 | 0.001 | yes | - | - | yes | yes |
| B fold 4 seed 43 (patience12) | - | 12 | 12 | no | 6 | 0.0007 | 1.0000 | 1.0000 | 3 | 0.00025 | - | 8 | 9 | yes | - |
| B fold 4 seed 43 (constant_lr) | - | 12 | 6 | no | 6 | 0.0013 | 1.0000 | 1.0000 | 3 | 0.001 | yes | - | - | no | yes |
| B fold 4 seed 44 (patience12) | yes | 12 | 11 | no | 7 | 0.0262 | 0.9925 | 1.0000 | 2 | 0.000125 | - | 4 | 5 | yes | - |
| B fold 4 seed 44 (constant_lr) | yes | 12 | 11 | no | 10 | 0.0037 | 1.0000 | 1.0000 | 2 | 0.001 | yes | - | - | yes | yes |
| A\* fold 1 seed 42 (patience12) | - | 12 | 10 | no | 6 | 0.0104 | 0.9950 | 1.0000 | 4 | 0.001 | - | 12 | - | yes | - |
| A\* fold 1 seed 42 (constant_lr) | - | 12 | 10 | no | 6 | 0.0104 | 0.9950 | 1.0000 | 4 | 0.001 | yes | - | - | yes | yes |
| A\* fold 1 seed 43 (patience12) | - | 12 | 9 | no | 7 | 0.0165 | 0.9975 | 0.9995 | 3 | 0.0005 | - | 11 | 12 | yes | - |
| A\* fold 1 seed 43 (constant_lr) | - | 12 | 9 | no | 7 | 0.0165 | 0.9975 | 0.9995 | 3 | 0.001 | yes | - | - | no | yes |
| A\* fold 1 seed 44 (patience12) | - | 12 | 11 | no | 7 | 0.0345 | 0.9875 | 0.9991 | 2 | 0.0005 | - | 3 | 4 | yes | - |
| A\* fold 1 seed 44 (constant_lr) | - | 12 | 9 | no | 7 | 0.0269 | 0.9975 | 0.9979 | 2 | 0.001 | yes | - | - | yes | yes |
| A\* fold 2 seed 42 (patience12) | - | 12 | 11 | no | 5 | 0.0001 | 1.0000 | 1.0000 | 3 | 0.00025 | - | 9 | 10 | yes | - |
| A\* fold 2 seed 42 (constant_lr) | - | 12 | 7 | no | 5 | 0.0001 | 1.0000 | 1.0000 | 3 | 0.001 | yes | - | - | yes | yes |
| A\* fold 2 seed 43 (patience12) | yes | 12 | 10 | no | 7 | 0.0004 | 1.0000 | 1.0000 | 4 | 0.00025 | - | 3 | 4 | yes | - |
| A\* fold 2 seed 43 (constant_lr) | yes | 12 | 9 | no | 6 | 0.0051 | 1.0000 | 1.0000 | 5 | 0.001 | yes | - | - | yes | yes |
| A\* fold 2 seed 44 (patience12) | yes | 12 | 12 | no | 10 | 0.2171 | 0.9975 | 0.9950 | 3 | 0.00025 | - | 3 | 4 | yes | - |
| A\* fold 2 seed 44 (constant_lr) | yes | 12 | 10 | no | 5 | 0.0001 | 1.0000 | 1.0000 | 3 | 0.001 | yes | - | - | yes | yes |
| A\* fold 3 seed 42 (patience12) | yes | 12 | 9 | no | 9 | 0.3282 | 0.9425 | 0.9900 | 3 | 0.000125 | - | 3 | 4 | yes | - |
| A\* fold 3 seed 42 (constant_lr) | yes | 12 | 10 | no | 6 | 0.0256 | 0.9900 | 1.0000 | 2 | 0.001 | yes | - | - | no | yes |
| A\* fold 3 seed 43 (patience12) | - | 12 | 12 | no | 6 | 0.0088 | 0.9975 | 1.0000 | 5 | 0.0005 | - | 8 | 9 | yes | - |
| A\* fold 3 seed 43 (constant_lr) | - | 12 | 12 | no | 6 | 0.0976 | 0.9725 | 0.9978 | 5 | 0.001 | yes | - | - | yes | yes |
| A\* fold 3 seed 44 (patience12) | yes | 12 | 11 | no | 11 | 0.5239 | 0.8900 | 0.9895 | 4 | 0.000125 | - | 3 | 4 | no | - |
| A\* fold 3 seed 44 (constant_lr) | yes | 12 | 8 | no | 7 | 0.2450 | 0.8750 | 1.0000 | 4 | 0.001 | yes | - | - | yes | yes |
| A\* fold 4 seed 42 (patience12) | - | 12 | 10 | no | 10 | 0.8106 | 0.7975 | 0.9990 | 3 | 0.0005 | - | 4 | 5 | no | - |
| A\* fold 4 seed 42 (constant_lr) | - | 12 | 11 | no | 9 | 0.2092 | 0.8975 | 0.9715 | 3 | 0.001 | yes | - | - | yes | yes |
| A\* fold 4 seed 43 (patience12) | - | 12 | 9 | no | 7 | 0.0272 | 0.9950 | 0.9993 | 4 | 0.0005 | - | 11 | 12 | no | - |
| A\* fold 4 seed 43 (constant_lr) | - | 12 | 9 | no | 7 | 0.0272 | 0.9950 | 0.9993 | 4 | 0.001 | yes | - | - | no | yes |
| A\* fold 4 seed 44 (patience12) | yes | 12 | 12 | no | 10 | 0.3154 | 0.8650 | 0.9999 | 5 | 0.000125 | - | 3 | 4 | yes | - |
| A\* fold 4 seed 44 (constant_lr) | yes | 12 | 8 | no | 6 | 0.0123 | 0.9975 | 1.0000 | 4 | 0.001 | yes | - | - | yes | yes |
| B' fold 1 seed 42 (patience12) | - | 12 | 12 | no | 2 | 0.0033 | 0.9975 | 1.0000 | 5 | 0.000125 | - | 4 | 5 | yes | - |
| B' fold 1 seed 42 (constant_lr) | - | 12 | 10 | no | 2 | 0.0144 | 0.9975 | 0.9997 | 3 | 0.001 | yes | - | - | yes | yes |
| B' fold 1 seed 43 (patience12) | - | 12 | 4 | no | 4 | 0.8647 | 0.9400 | 0.9899 | 1 | 0.000125 | - | 6 | 7 | no | - |
| B' fold 1 seed 43 (constant_lr) | - | 12 | 12 | no | 4 | 0.0642 | 0.9675 | 1.0000 | 1 | 0.001 | yes | - | - | yes | yes |
| B' fold 1 seed 44 (patience12) | yes | 12 | 9 | no | 7 | 0.1895 | 0.9950 | 0.9950 | 5 | 0.000125 | - | 3 | 4 | no | - |
| B' fold 1 seed 44 (constant_lr) | yes | 12 | 6 | no | 6 | 0.3141 | 0.9975 | 0.9975 | 5 | 0.001 | yes | - | - | yes | yes |
| B' fold 2 seed 42 (patience12) | - | 12 | 11 | no | 11 | 0.8750 | 0.8775 | 0.9830 | 7 | 0.0005 | - | 8 | 9 | no | - |
| B' fold 2 seed 42 (constant_lr) | - | 12 | 10 | no | - | 1.6847 | 0.8550 | 0.8678 | 7 | 0.001 | yes | - | - | no | yes |
| B' fold 2 seed 43 (patience12) | - | 12 | 10 | no | 3 | 0.0042 | 1.0000 | 1.0000 | 3 | 0.0005 | - | 8 | 9 | yes | - |
| B' fold 2 seed 43 (constant_lr) | - | 12 | 12 | no | 3 | 0.0217 | 0.9950 | 1.0000 | 3 | 0.001 | yes | - | - | yes | yes |
| B' fold 2 seed 44 (patience12) | yes | 12 | 12 | no | 12 | 0.0736 | 0.9875 | 0.9950 | 7 | 0.00025 | - | 4 | 5 | yes | - |
| B' fold 2 seed 44 (constant_lr) | yes | 12 | 12 | no | 7 | 0.0160 | 0.9950 | 1.0000 | 5 | 0.001 | yes | - | - | yes | yes |
| B' fold 3 seed 42 (patience12) | - | 12 | 11 | no | 1 | 0.0516 | 0.9900 | 0.9975 | 3 | 0.000125 | - | 3 | 4 | yes | - |
| B' fold 3 seed 42 (constant_lr) | - | 12 | 4 | no | 1 | 0.0260 | 0.9900 | 1.0000 | 5 | 0.001 | yes | - | - | yes | yes |
| B' fold 3 seed 43 (patience12) | - | 12 | 9 | no | 7 | 0.0011 | 1.0000 | 1.0000 | 5 | 0.0005 | - | 11 | 12 | yes | - |
| B' fold 3 seed 43 (constant_lr) | - | 12 | 9 | no | 7 | 0.0011 | 1.0000 | 1.0000 | 5 | 0.001 | yes | - | - | no | yes |
| B' fold 3 seed 44 (patience12) | - | 12 | 9 | no | 7 | 0.0341 | 0.9975 | 1.0000 | 6 | 0.00025 | - | 5 | 6 | yes | - |
| B' fold 3 seed 44 (constant_lr) | - | 12 | 9 | no | 8 | 0.0814 | 0.9850 | 1.0000 | 7 | 0.001 | yes | - | - | no | yes |
| B' fold 4 seed 42 (patience12) | - | 12 | 2 | no | 1 | 0.1375 | 0.9175 | 0.9966 | 3 | 6.25e-05 | - | 4 | 5 | no | - |
| B' fold 4 seed 42 (constant_lr) | - | 12 | 11 | no | 1 | 0.0968 | 0.9700 | 0.9940 | 3 | 0.001 | yes | - | - | no | yes |
| B' fold 4 seed 43 (patience12) | - | 12 | 3 | no | 3 | 0.0142 | 1.0000 | 1.0000 | 3 | 6.25e-05 | - | 5 | 6 | yes | - |
| B' fold 4 seed 43 (constant_lr) | - | 12 | 3 | no | 3 | 0.0142 | 1.0000 | 1.0000 | 4 | 0.001 | yes | - | - | yes | yes |
| B' fold 4 seed 44 (patience12) | yes | 12 | 11 | no | 10 | 0.0851 | 0.9850 | 0.9988 | 4 | 0.00025 | - | 3 | 4 | yes | - |
| B' fold 4 seed 44 (constant_lr) | yes | 12 | 10 | no | 10 | 0.2785 | 0.8900 | 0.9831 | 4 | 0.001 | yes | - | - | no | yes |

Medians [min-max] per group:

| group | run | first_recovery | restored_epoch | best_val_loss | dev_acc | dev_auroc | epochs_at_half | lr_last |
|---|---|---|---|---|---|---|---|---|
| the ten | patience12 | 9.5000 [7.0000-12.0000] (n 10) | 11.0000 [9.0000-12.0000] (n 10) | 0.1373 [0.0004-0.5239] (n 10) | 0.9900 [0.8650-1.0000] (n 10) | 0.9955 [0.9895-1.0000] (n 10) | 4.0000 [2.0000-7.0000] (n 10) | 0.0002 [0.0001-0.0003] (n 10) |
| the ten | constant_lr | 6.0000 [4.0000-10.0000] (n 10) | 9.5000 [6.0000-12.0000] (n 10) | 0.0142 [0.0001-0.3141] (n 10) | 0.9975 [0.8750-1.0000] (n 10) | 1.0000 [0.9831-1.0000] (n 10) | 4.0000 [2.0000-5.0000] (n 10) | 0.0010 [0.0010-0.0010] (n 10) |
| the other 26 | patience12 | 6.0000 [1.0000-11.0000] (n 26) | 10.0000 [2.0000-12.0000] (n 26) | 0.0153 [0.0001-0.8750] (n 26) | 0.9975 [0.7975-1.0000] (n 26) | 1.0000 [0.9830-1.0000] (n 26) | 3.0000 [1.0000-7.0000] (n 26) | 0.0003 [0.0001-0.0010] (n 26) |
| the other 26 | constant_lr | 6.0000 [1.0000-9.0000] (n 25) | 9.5000 [3.0000-12.0000] (n 26) | 0.0154 [0.0001-1.6847] (n 26) | 0.9963 [0.8550-1.0000] (n 26) | 1.0000 [0.8678-1.0000] (n 26) | 3.5000 [1.0000-7.0000] (n 26) | 0.0010 [0.0010-0.0010] (n 26) |
| all 36 | patience12 | 7.0000 [1.0000-12.0000] (n 36) | 10.5000 [2.0000-12.0000] (n 36) | 0.0267 [0.0001-0.8750] (n 36) | 0.9963 [0.7975-1.0000] (n 36) | 1.0000 [0.9830-1.0000] (n 36) | 3.5000 [1.0000-7.0000] (n 36) | 0.0003 [0.0001-0.0010] (n 36) |
| all 36 | constant_lr | 6.0000 [1.0000-10.0000] (n 35) | 9.5000 [3.0000-12.0000] (n 36) | 0.0152 [0.0001-1.6847] (n 36) | 0.9975 [0.8550-1.0000] (n 36) | 1.0000 [0.8678-1.0000] (n 36) | 4.0000 [1.0000-7.0000] (n 36) | 0.0010 [0.0010-0.0010] (n 36) |

Reported apart: B fold 3 seed 44, B' fold 4 seed 42.
- B fold 3 seed 44: patience12 first recovery 10, restored 11, epochs at 0.5 3, first reduced epoch 6; constant_lr first recovery 7, recovered at epoch 12 yes, restored 10, epochs at 0.5 3, dev acc 0.9150, dev AUROC 1.0000.
- B' fold 4 seed 42: patience12 first recovery 1, restored 2, epochs at 0.5 3, first reduced epoch 5; constant_lr first recovery 1, recovered at epoch 12 no, restored 11, epochs at 0.5 3, dev acc 0.9700, dev AUROC 0.9940.

## (d) Test, not used to choose or judge the recipe

the test: only from the public aggregate.json of the two runs (never from the restricted test scores), the verdicts and the median differences over the seeds of A\* - B, B' - B and A\* - B' side by side, in a separate section, never used to choose or judge.

Source: public/aggregate.json of the two runs (never the restricted test scores). Verdicts that differ from the patience12 run: 12 of 84.

| primary (A\* - B, stratum 03) | patience12 | constant_lr run |
|---|---|---|
| auroc | inconclusive | inconclusive |
| acer_dev | inconclusive | inconclusive |

| comparison | stratum | metric | patience12 verdict | patience12 median difference | constant_lr verdict | constant_lr median difference |
|---|---|---|---|---|---|---|
| A\* - B | 01 | apcer_05 | inconclusive | -0.0468 | inconclusive | -0.1088 |
| A\* - B | 01 | bpcer_05 | inconclusive | -0.0495 | inconclusive | 0.0799 |
| A\* - B | 01 | acer_05 | inconclusive | -0.0481 | inconclusive | -0.0122 |
| A\* - B | 01 | apcer_dev | inconclusive | 0.0012 | inconclusive | -0.0503 |
| A\* - B | 01 | bpcer_dev | inconclusive | -0.0090 | inconclusive | 0.0427 |
| A\* - B | 01 | acer_dev | inconclusive | -0.0197 | inconclusive | -0.0336 |
| A\* - B | 01 | auroc | inconclusive | 0.0038 | inconclusive | 0.0070 |
| A\* - B | 02 | apcer_05 | inconclusive | 0.0011 | inconclusive | 0.0034 |
| A\* - B | 02 | bpcer_05 | inconclusive | 0.0012 | inconclusive | 0.0082 |
| A\* - B | 02 | acer_05 | inconclusive | 0.0011 | inconclusive | 0.0299 |
| A\* - B | 02 | apcer_dev | inconclusive | 0.0022 | inconclusive | 0.0403 |
| A\* - B | 02 | bpcer_dev | inconclusive | -0.0035 | inconclusive | 0.0012 |
| A\* - B | 02 | acer_dev | inconclusive | -0.0012 | inconclusive | 0.0207 |
| A\* - B | 02 | auroc | inconclusive | -0.0065 | inconclusive | -0.0012 |
| A\* - B | 03 | apcer_05 | inconclusive | -0.0194 | inconclusive | 0.0050 |
| A\* - B | 03 | bpcer_05 | consistent (A\* lower BPCER (0.5) than B) | -0.4170 | inconclusive | -0.3195 |
| A\* - B | 03 | acer_05 | consistent (A\* lower ACER (0.5) than B) | -0.2381 | inconclusive | -0.2647 |
| A\* - B | 03 | apcer_dev | inconclusive | 0.1210 | inconclusive | 0.1113 |
| A\* - B | 03 | bpcer_dev | inconclusive | -0.4432 | inconclusive | -0.3968 |
| A\* - B | 03 | acer_dev | inconclusive | -0.2007 | inconclusive | -0.2293 |
| A\* - B | 03 | auroc | inconclusive | 0.2186 | inconclusive | 0.1945 |
| A\* - B | all | apcer_05 | inconclusive | -0.0172 | inconclusive | -0.0012 |
| A\* - B | all | bpcer_05 | consistent (A\* lower BPCER (0.5) than B) | -0.2993 | inconclusive | -0.1777 |
| A\* - B | all | acer_05 | consistent (A\* lower ACER (0.5) than B) | -0.1575 | inconclusive | -0.1756 |
| A\* - B | all | apcer_dev | inconclusive | 0.0973 | inconclusive | 0.0844 |
| A\* - B | all | bpcer_dev | inconclusive | -0.3571 | inconclusive | -0.2472 |
| A\* - B | all | acer_dev | inconclusive | -0.1622 | inconclusive | -0.1383 |
| A\* - B | all | auroc | inconclusive | 0.1105 | inconclusive | 0.0933 |
| B' - B | 01 | apcer_05 | inconclusive | -0.0164 | inconclusive | 0.0047 |
| B' - B | 01 | bpcer_05 | inconclusive | 0.0562 | inconclusive | -0.0090 |
| B' - B | 01 | acer_05 | inconclusive | 0.0261 | inconclusive | 0.0253 |
| B' - B | 01 | apcer_dev | inconclusive | 0.0491 | inconclusive | 0.0678 |
| B' - B | 01 | bpcer_dev | inconclusive | 0.0439 | inconclusive | 0.0315 |
| B' - B | 01 | acer_dev | inconclusive | 0.0465 | inconclusive | 0.0637 |
| B' - B | 01 | auroc | inconclusive | -0.0310 | inconclusive | -0.0028 |
| B' - B | 02 | apcer_05 | inconclusive | 0.0011 | inconclusive | 0.0246 |
| B' - B | 02 | bpcer_05 | inconclusive | 0.0070 | inconclusive | 0.0141 |
| B' - B | 02 | acer_05 | inconclusive | 0.0046 | inconclusive | 0.0546 |
| B' - B | 02 | apcer_dev | inconclusive | 0.0022 | inconclusive | 0.0112 |
| B' - B | 02 | bpcer_dev | inconclusive | 0.0023 | inconclusive | 0.1311 |
| B' - B | 02 | acer_dev | inconclusive | 0.0006 | inconclusive | 0.0673 |
| B' - B | 02 | auroc | inconclusive | -0.0012 | inconclusive | -0.0779 |
| B' - B | 03 | apcer_05 | inconclusive | -0.0424 | inconclusive | -0.0330 |
| B' - B | 03 | bpcer_05 | inconclusive | 0.0291 | inconclusive | 0.1591 |
| B' - B | 03 | acer_05 | inconclusive | -0.0066 | inconclusive | 0.0368 |
| B' - B | 03 | apcer_dev | inconclusive | 0.0616 | inconclusive | 0.0594 |
| B' - B | 03 | bpcer_dev | inconclusive | 0.0187 | inconclusive | 0.0309 |
| B' - B | 03 | acer_dev | inconclusive | 0.0312 | inconclusive | 0.0416 |
| B' - B | 03 | auroc | inconclusive | -0.1084 | inconclusive | -0.1149 |
| B' - B | all | apcer_05 | inconclusive | -0.0296 | inconclusive | -0.0135 |
| B' - B | all | bpcer_05 | inconclusive | 0.0576 | inconclusive | 0.1226 |
| B' - B | all | acer_05 | inconclusive | 0.0140 | inconclusive | 0.0225 |
| B' - B | all | apcer_dev | inconclusive | 0.0527 | inconclusive | 0.0569 |
| B' - B | all | bpcer_dev | inconclusive | 0.0678 | inconclusive | 0.0456 |
| B' - B | all | acer_dev | inconclusive | 0.0563 | inconclusive | 0.0500 |
| B' - B | all | auroc | inconclusive | -0.1122 | inconclusive | -0.1104 |
| A\* - B' | 01 | apcer_05 | inconclusive | 0.0012 | inconclusive | -0.1076 |
| A\* - B' | 01 | bpcer_05 | inconclusive | -0.0461 | inconclusive | 0.0889 |
| A\* - B' | 01 | acer_05 | inconclusive | -0.0549 | inconclusive | -0.0094 |
| A\* - B' | 01 | apcer_dev | inconclusive | -0.0795 | inconclusive | -0.1181 |
| A\* - B' | 01 | bpcer_dev | inconclusive | -0.0484 | inconclusive | 0.0157 |
| A\* - B' | 01 | acer_dev | inconclusive | -0.0552 | inconclusive | -0.0336 |
| A\* - B' | 01 | auroc | inconclusive | 0.0348 | inconclusive | -0.0007 |
| A\* - B' | 02 | apcer_05 | inconclusive | 0.0000 | inconclusive | -0.0213 |
| A\* - B' | 02 | bpcer_05 | inconclusive | 0.0527 | inconclusive | -0.0059 |
| A\* - B' | 02 | acer_05 | inconclusive | 0.0297 | inconclusive | -0.0158 |
| A\* - B' | 02 | apcer_dev | inconclusive | -0.0011 | inconclusive | -0.0101 |
| A\* - B' | 02 | bpcer_dev | inconclusive | 0.0012 | inconclusive | -0.1300 |
| A\* - B' | 02 | acer_dev | inconclusive | -0.0011 | inconclusive | -0.0415 |
| A\* - B' | 02 | auroc | inconclusive | -0.0053 | inconclusive | 0.0745 |
| A\* - B' | 03 | apcer_05 | inconclusive | 0.0090 | inconclusive | -0.0370 |
| A\* - B' | 03 | bpcer_05 | inconclusive | -0.6701 | consistent (A\* lower BPCER (0.5) than B') | -0.4786 |
| A\* - B' | 03 | acer_05 | inconclusive | -0.3306 | consistent (A\* lower ACER (0.5) than B') | -0.3015 |
| A\* - B' | 03 | apcer_dev | inconclusive | 0.0594 | inconclusive | -0.0604 |
| A\* - B' | 03 | bpcer_dev | inconclusive | -0.6511 | consistent (A\* lower BPCER (dev threshold) than B') | -0.4298 |
| A\* - B' | 03 | acer_dev | inconclusive | -0.2959 | consistent (A\* lower ACER (dev threshold) than B') | -0.2618 |
| A\* - B' | 03 | auroc | consistent (A\* higher AUROC than B') | 0.3689 | consistent (A\* higher AUROC than B') | 0.3094 |
| A\* - B' | all | apcer_05 | inconclusive | 0.0087 | inconclusive | -0.0431 |
| A\* - B' | all | bpcer_05 | inconclusive | -0.4405 | consistent (A\* lower BPCER (0.5) than B') | -0.3250 |
| A\* - B' | all | acer_05 | inconclusive | -0.2159 | consistent (A\* lower ACER (0.5) than B') | -0.1981 |
| A\* - B' | all | apcer_dev | inconclusive | 0.0446 | inconclusive | -0.0643 |
| A\* - B' | all | bpcer_dev | inconclusive | -0.4284 | consistent (A\* lower BPCER (dev threshold) than B') | -0.3138 |
| A\* - B' | all | acer_dev | inconclusive | -0.1919 | consistent (A\* lower ACER (dev threshold) than B') | -0.1891 |
| A\* - B' | all | auroc | consistent (A\* higher AUROC than B') | 0.2046 | consistent (A\* higher AUROC than B') | 0.2162 |

## (e) Times

if the control run is INCOMPLETE (cap or voluntary stop) or FAIL, no verdict is given and what is complete is reported; the time is reported against the estimate.

| run | arm | models | epochs | train_seconds | s/epoch (weighted) |
|---|---|---|---|---|---|
| patience12 | B | 12 | 144 | 6135.8000 | 42.6097 |
| patience12 | A\* | 12 | 144 | 5762.6000 | 40.0181 |
| patience12 | B' | 12 | 144 | 4967.8000 | 34.4986 |
| constant_lr | B | 12 | 144 | 5258.2000 | 36.5153 |
| constant_lr | A\* | 12 | 144 | 5335.0000 | 37.0486 |
| constant_lr | B' | 12 | 144 | 3735.1000 | 25.9382 |

The constant_lr run: training 14328.3 s, wall 15217.0 s against the estimate of 17941 s (difference -2723.9 s), cap 21600 s (within: yes); final writes after the cap: no; stopped: no. The patience12 run wall 17940.9 s.

## Declarations

- no_training: yes
- no_inference: yes
- no_test_scores_read: yes
- test_from_public_aggregate_only: yes
- test_not_used_to_judge_the_recipe: yes
- reading_rule_fixed_before_the_run: yes
- criteria_defined_in_the_previous_package: yes
- no_identifiers: the public files hold no subject identifier, no image key, no per-frame row and no per-subject count; subject_id is read from the dev scores only to scan the public files
- dev_columns_read: dev_scores.csv: label and score_attack for the measures; subject_id only to scan the public files, never written; arm, fold, train_seed, role, checkpoint_sha256, sample_id, file_sha256, prediction_05, prediction_dev, threshold_dev and threshold_sha256 only for row-by-row consistency checks (with the card, the frozen threshold, the other models of the fold and the same model of the other run). threshold.json: arm, fold, train_seed, checkpoint_sha256, value, dev_scores_sha256, frozen_at for consistency only. Training cards: the history and the recipe fields, train_seconds for the times, weights_file, weights_sha256, dev_keys_sha256, saved and test_scored_after_threshold for consistency only. status.json: identity, outcome, models, anchors and wall_seconds. Public aggregate.json: run_id, status, config_sha256, strata, primary_verdicts and, for the three comparisons, verdicts and pooled_summary_over_seeds medians only
- limits: the criteria are those of the previous package, defined after the patience-4 run; a constant rate tests the absence of the reduction, not the level of the rate (0.001 is never varied); with a different rate path the checkpoint chosen by the minimum dev val_loss can change, so dev and test values at the checkpoint differ for reasons beyond the recovery; the dev is small and chooses the checkpoint among 12 epochs; the reduction is removed in all 36 models; one dataset, one attack type, three seeds; the test section is descriptive and never part of the verdict
