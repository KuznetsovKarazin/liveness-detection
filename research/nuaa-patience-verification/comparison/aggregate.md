# Verification run with early_stopping_patience 12 against the parent run (20261109-PATVER-4f2b925)

Parent run 20261106-SESSABLRUN-d999458 (patience 4, PASS, 36 models); verification run 20261109-SESSABLRUN-6a50265 (patience 12, PASS, 36 complete models). Reading rule agreed on 7 November 2026 and fixed before the run; criteria of the previous package: restored_epoch in {1, 2} and val_acc_at_best = 0.5 (within 1e-6): the rule written in the delivery of the session ablation run, re-applied here; recovery: first epoch with val_accuracy >= 0.75 and val_loss < 1.0 (dev only). No training, no inference, no test score read. Figures: figure_ten.png, figure_no_recovery.png, figure_val_accuracy_all.png written.

## (a) Shared histories

shared histories: for every model, the first epochs_run epochs of the parent run must be identical in the verification run (loss, accuracy, val_loss, val_accuracy, lr; difference 0); otherwise the largest difference is reported (model, epoch, field) and the comparison is declared not valid: no verdict.

Models compared: 36; identical shared histories: 36 of 36; valid: yes.
Largest absolute difference per field: loss 0.0, accuracy 0.0, val_loss 0.0, val_accuracy 0.0, lr 0.0.

## (b) Primary outcome

primary outcome: positive if the ten original degenerate trainings all recover within 12 epochs and none of the 36 models is degenerate; otherwise negative ('this change is not enough'). Every model without recovery is listed separately, inside or outside the ten (for example B fold 3 seed 44); if a model outside the ten does not recover, the conclusion adds that the change is not enough to make every model stable; in either case (one of the ten or a model outside the ten without recovery) this does not rule out a role of early stopping (the lr halved by ReduceLROnPlateau is the declared confounder); a positive outcome does not by itself show that every model is stable and does not separate the stop from that lr.

**Verdict: positive.** Positive: the ten original degenerate trainings recover within 12 epochs and none of the 36 models is degenerate. The recovery does not separate the stop from the lr halved by ReduceLROnPlateau (declared confounder), and a positive outcome does not by itself show that every model is stable.

The ten original degenerate trainings: 10 of 10 complete recover within 12 epochs (first recovery: A\* fold 2 seed 43 7, A\* fold 2 seed 44 10, A\* fold 3 seed 42 9, A\* fold 3 seed 44 11, A\* fold 4 seed 44 10, B fold 1 seed 42 8, B fold 4 seed 44 7, B' fold 1 seed 44 7, B' fold 2 seed 44 12, B' fold 4 seed 44 10).
Without recovery among the ten: none.
Degenerate among the complete models of the verification run: none.
Without recovery outside the ten (listed separately): none.
Without recovery in the parent run: B fold 1 seed 42, B fold 3 seed 44, B fold 4 seed 44, A\* fold 2 seed 43, A\* fold 2 seed 44, A\* fold 3 seed 42, A\* fold 3 seed 44, A\* fold 4 seed 44, B' fold 1 seed 44, B' fold 2 seed 44, B' fold 4 seed 44.

## (c) Secondary outcomes

secondary outcomes per model, parent against verification run: epoch of first recovery, restored epoch, best val_loss, dev accuracy and dev AUROC at the checkpoint (from the dev scores, checked row by row), epochs at 0.5, lr per epoch.

Models with a recovery: parent 25 of 36, verification run 36 of 36 complete.

| model | ten | epochs run | restored | degenerate | first recovery | best val_loss | dev acc | dev AUROC | epochs at 0.5 | lr last | shared identical |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B fold 1 seed 42 (parent) | yes | 5 | 1 | yes | - | 7.0611 | 0.5000 | 0.9975 | 5 | 0.0005 | - |
| B fold 1 seed 42 (new) | yes | 12 | 10 | no | 8 | 0.0378 | 0.9950 | 0.9960 | 5 | 0.00025 | yes |
| B fold 1 seed 43 (parent) | - | 12 | 11 | no | 5 | 0.0056 | 0.9975 | 1.0000 | 3 | 0.0005 | - |
| B fold 1 seed 43 (new) | - | 12 | 11 | no | 5 | 0.0056 | 0.9975 | 1.0000 | 3 | 0.0005 | yes |
| B fold 1 seed 44 (parent) | - | 10 | 6 | no | 6 | 0.0298 | 0.9950 | 0.9985 | 3 | 0.0005 | - |
| B fold 1 seed 44 (new) | - | 12 | 11 | no | 6 | 0.0224 | 0.9975 | 1.0000 | 3 | 0.00025 | yes |
| B fold 2 seed 42 (parent) | - | 12 | 9 | no | 5 | 0.2308 | 0.9475 | 0.9888 | 3 | 0.000125 | - |
| B fold 2 seed 42 (new) | - | 12 | 9 | no | 5 | 0.2308 | 0.9475 | 0.9888 | 3 | 0.000125 | yes |
| B fold 2 seed 43 (parent) | - | 11 | 7 | no | 7 | 0.0011 | 1.0000 | 1.0000 | 4 | 0.00025 | - |
| B fold 2 seed 43 (new) | - | 12 | 7 | no | 7 | 0.0011 | 1.0000 | 1.0000 | 4 | 0.000125 | yes |
| B fold 2 seed 44 (parent) | - | 12 | 10 | no | 6 | 0.0030 | 0.9975 | 1.0000 | 4 | 0.00025 | - |
| B fold 2 seed 44 (new) | - | 12 | 10 | no | 6 | 0.0030 | 0.9975 | 1.0000 | 4 | 0.00025 | yes |
| B fold 3 seed 42 (parent) | - | 12 | 12 | no | 6 | 0.0168 | 1.0000 | 1.0000 | 4 | 0.00025 | - |
| B fold 3 seed 42 (new) | - | 12 | 12 | no | 6 | 0.0168 | 1.0000 | 1.0000 | 4 | 0.00025 | yes |
| B fold 3 seed 43 (parent) | - | 12 | 10 | no | 6 | 0.0531 | 0.9925 | 1.0000 | 4 | 0.0005 | - |
| B fold 3 seed 43 (new) | - | 12 | 10 | no | 6 | 0.0531 | 0.9925 | 1.0000 | 4 | 0.0005 | yes |
| B fold 3 seed 44 (parent) | - | 7 | 3 | no | - | 3.2731 | 0.2375 | 0.0950 | 3 | 0.0005 | - |
| B fold 3 seed 44 (new) | - | 12 | 11 | no | 10 | 0.0119 | 1.0000 | 1.0000 | 3 | 0.000125 | yes |
| B fold 4 seed 42 (parent) | - | 12 | 11 | no | 6 | 0.0005 | 1.0000 | 1.0000 | 3 | 0.00025 | - |
| B fold 4 seed 42 (new) | - | 12 | 11 | no | 6 | 0.0005 | 1.0000 | 1.0000 | 3 | 0.00025 | yes |
| B fold 4 seed 43 (parent) | - | 10 | 6 | no | 6 | 0.0013 | 1.0000 | 1.0000 | 3 | 0.0005 | - |
| B fold 4 seed 43 (new) | - | 12 | 12 | no | 6 | 0.0007 | 1.0000 | 1.0000 | 3 | 0.00025 | yes |
| B fold 4 seed 44 (parent) | yes | 6 | 2 | yes | - | 2.3605 | 0.5000 | 0.9652 | 2 | 0.0005 | - |
| B fold 4 seed 44 (new) | yes | 12 | 11 | no | 7 | 0.0262 | 0.9925 | 1.0000 | 2 | 0.000125 | yes |
| A\* fold 1 seed 42 (parent) | - | 12 | 10 | no | 6 | 0.0104 | 0.9950 | 1.0000 | 4 | 0.001 | - |
| A\* fold 1 seed 42 (new) | - | 12 | 10 | no | 6 | 0.0104 | 0.9950 | 1.0000 | 4 | 0.001 | yes |
| A\* fold 1 seed 43 (parent) | - | 12 | 9 | no | 7 | 0.0165 | 0.9975 | 0.9995 | 3 | 0.0005 | - |
| A\* fold 1 seed 43 (new) | - | 12 | 9 | no | 7 | 0.0165 | 0.9975 | 0.9995 | 3 | 0.0005 | yes |
| A\* fold 1 seed 44 (parent) | - | 12 | 11 | no | 7 | 0.0345 | 0.9875 | 0.9991 | 2 | 0.0005 | - |
| A\* fold 1 seed 44 (new) | - | 12 | 11 | no | 7 | 0.0345 | 0.9875 | 0.9991 | 2 | 0.0005 | yes |
| A\* fold 2 seed 42 (parent) | - | 12 | 11 | no | 5 | 0.0001 | 1.0000 | 1.0000 | 3 | 0.00025 | - |
| A\* fold 2 seed 42 (new) | - | 12 | 11 | no | 5 | 0.0001 | 1.0000 | 1.0000 | 3 | 0.00025 | yes |
| A\* fold 2 seed 43 (parent) | yes | 5 | 1 | yes | - | 60.9868 | 0.5000 | 0.5000 | 4 | 0.0005 | - |
| A\* fold 2 seed 43 (new) | yes | 12 | 10 | no | 7 | 0.0004 | 1.0000 | 1.0000 | 4 | 0.00025 | yes |
| A\* fold 2 seed 44 (parent) | yes | 5 | 1 | yes | - | 18.0934 | 0.5000 | 0.2114 | 3 | 0.0005 | - |
| A\* fold 2 seed 44 (new) | yes | 12 | 12 | no | 10 | 0.2171 | 0.9975 | 0.9950 | 3 | 0.00025 | yes |
| A\* fold 3 seed 42 (parent) | yes | 5 | 1 | yes | - | 3.1296 | 0.5000 | 0.4690 | 3 | 0.0005 | - |
| A\* fold 3 seed 42 (new) | yes | 12 | 9 | no | 9 | 0.3282 | 0.9425 | 0.9900 | 3 | 0.000125 | yes |
| A\* fold 3 seed 43 (parent) | - | 12 | 12 | no | 6 | 0.0088 | 0.9975 | 1.0000 | 5 | 0.0005 | - |
| A\* fold 3 seed 43 (new) | - | 12 | 12 | no | 6 | 0.0088 | 0.9975 | 1.0000 | 5 | 0.0005 | yes |
| A\* fold 3 seed 44 (parent) | yes | 5 | 1 | yes | - | 33.4090 | 0.5000 | 0.5000 | 4 | 0.0005 | - |
| A\* fold 3 seed 44 (new) | yes | 12 | 11 | no | 11 | 0.5239 | 0.8900 | 0.9895 | 4 | 0.000125 | yes |
| A\* fold 4 seed 42 (parent) | - | 12 | 10 | no | 10 | 0.8106 | 0.7975 | 0.9990 | 3 | 0.0005 | - |
| A\* fold 4 seed 42 (new) | - | 12 | 10 | no | 10 | 0.8106 | 0.7975 | 0.9990 | 3 | 0.0005 | yes |
| A\* fold 4 seed 43 (parent) | - | 12 | 9 | no | 7 | 0.0272 | 0.9950 | 0.9993 | 4 | 0.0005 | - |
| A\* fold 4 seed 43 (new) | - | 12 | 9 | no | 7 | 0.0272 | 0.9950 | 0.9993 | 4 | 0.0005 | yes |
| A\* fold 4 seed 44 (parent) | yes | 5 | 1 | yes | - | 22.8274 | 0.5000 | 0.5000 | 5 | 0.0005 | - |
| A\* fold 4 seed 44 (new) | yes | 12 | 12 | no | 10 | 0.3154 | 0.8650 | 0.9999 | 5 | 0.000125 | yes |
| B' fold 1 seed 42 (parent) | - | 6 | 2 | no | 2 | 0.6101 | 0.7725 | 0.9937 | 5 | 0.0005 | - |
| B' fold 1 seed 42 (new) | - | 12 | 12 | no | 2 | 0.0033 | 0.9975 | 1.0000 | 5 | 0.000125 | yes |
| B' fold 1 seed 43 (parent) | - | 8 | 4 | no | 4 | 0.8647 | 0.9400 | 0.9899 | 1 | 0.0005 | - |
| B' fold 1 seed 43 (new) | - | 12 | 4 | no | 4 | 0.8647 | 0.9400 | 0.9899 | 1 | 0.000125 | yes |
| B' fold 1 seed 44 (parent) | yes | 5 | 1 | yes | - | 12.1449 | 0.5000 | 0.4975 | 5 | 0.0005 | - |
| B' fold 1 seed 44 (new) | yes | 12 | 9 | no | 7 | 0.1895 | 0.9950 | 0.9950 | 5 | 0.000125 | yes |
| B' fold 2 seed 42 (parent) | - | 12 | 11 | no | 11 | 0.8750 | 0.8775 | 0.9830 | 7 | 0.0005 | - |
| B' fold 2 seed 42 (new) | - | 12 | 11 | no | 11 | 0.8750 | 0.8775 | 0.9830 | 7 | 0.0005 | yes |
| B' fold 2 seed 43 (parent) | - | 12 | 10 | no | 3 | 0.0042 | 1.0000 | 1.0000 | 3 | 0.0005 | - |
| B' fold 2 seed 43 (new) | - | 12 | 10 | no | 3 | 0.0042 | 1.0000 | 1.0000 | 3 | 0.0005 | yes |
| B' fold 2 seed 44 (parent) | yes | 6 | 2 | yes | - | 11.4259 | 0.5000 | 0.5000 | 6 | 0.0005 | - |
| B' fold 2 seed 44 (new) | yes | 12 | 12 | no | 12 | 0.0736 | 0.9875 | 0.9950 | 7 | 0.00025 | yes |
| B' fold 3 seed 42 (parent) | - | 5 | 1 | no | 1 | 0.3211 | 0.8325 | 1.0000 | 3 | 0.0005 | - |
| B' fold 3 seed 42 (new) | - | 12 | 11 | no | 1 | 0.0516 | 0.9900 | 0.9975 | 3 | 0.000125 | yes |
| B' fold 3 seed 43 (parent) | - | 12 | 9 | no | 7 | 0.0011 | 1.0000 | 1.0000 | 5 | 0.0005 | - |
| B' fold 3 seed 43 (new) | - | 12 | 9 | no | 7 | 0.0011 | 1.0000 | 1.0000 | 5 | 0.0005 | yes |
| B' fold 3 seed 44 (parent) | - | 12 | 9 | no | 7 | 0.0341 | 0.9975 | 1.0000 | 6 | 0.00025 | - |
| B' fold 3 seed 44 (new) | - | 12 | 9 | no | 7 | 0.0341 | 0.9975 | 1.0000 | 6 | 0.00025 | yes |
| B' fold 4 seed 42 (parent) | - | 6 | 2 | no | 1 | 0.1375 | 0.9175 | 0.9966 | 2 | 0.0005 | - |
| B' fold 4 seed 42 (new) | - | 12 | 2 | no | 1 | 0.1375 | 0.9175 | 0.9966 | 3 | 6.25e-05 | yes |
| B' fold 4 seed 43 (parent) | - | 7 | 3 | no | 3 | 0.0142 | 1.0000 | 1.0000 | 3 | 0.0005 | - |
| B' fold 4 seed 43 (new) | - | 12 | 3 | no | 3 | 0.0142 | 1.0000 | 1.0000 | 3 | 6.25e-05 | yes |
| B' fold 4 seed 44 (parent) | yes | 5 | 1 | yes | - | 17.2907 | 0.5000 | 0.5000 | 4 | 0.0005 | - |
| B' fold 4 seed 44 (new) | yes | 12 | 11 | no | 10 | 0.0851 | 0.9850 | 0.9988 | 4 | 0.00025 | yes |

Medians [min-max] per group:

| group | run | first_recovery | restored_epoch | best_val_loss | dev_acc | dev_auroc | epochs_at_half | lr_last |
|---|---|---|---|---|---|---|---|---|
| the ten | parent | - [---] (n 0) | 1.0000 [1.0000-2.0000] (n 10) | 14.7178 [2.3605-60.9868] (n 10) | 0.5000 [0.5000-0.5000] (n 10) | 0.5000 [0.2114-0.9975] (n 10) | 4.0000 [2.0000-6.0000] (n 10) | 0.0005 [0.0005-0.0005] (n 10) |
| the ten | new | 9.5000 [7.0000-12.0000] (n 10) | 11.0000 [9.0000-12.0000] (n 10) | 0.1373 [0.0004-0.5239] (n 10) | 0.9900 [0.8650-1.0000] (n 10) | 0.9955 [0.9895-1.0000] (n 10) | 4.0000 [2.0000-7.0000] (n 10) | 0.0002 [0.0001-0.0003] (n 10) |
| the other 26 | parent | 6.0000 [1.0000-11.0000] (n 25) | 9.0000 [1.0000-12.0000] (n 26) | 0.0220 [0.0001-3.2731] (n 26) | 0.9963 [0.2375-1.0000] (n 26) | 1.0000 [0.0950-1.0000] (n 26) | 3.0000 [1.0000-7.0000] (n 26) | 0.0005 [0.0001-0.0010] (n 26) |
| the other 26 | new | 6.0000 [1.0000-11.0000] (n 26) | 10.0000 [2.0000-12.0000] (n 26) | 0.0153 [0.0001-0.8750] (n 26) | 0.9975 [0.7975-1.0000] (n 26) | 1.0000 [0.9830-1.0000] (n 26) | 3.0000 [1.0000-7.0000] (n 26) | 0.0003 [0.0001-0.0010] (n 26) |
| all 36 | parent | 6.0000 [1.0000-11.0000] (n 25) | 6.5000 [1.0000-12.0000] (n 36) | 0.0953 [0.0001-60.9868] (n 36) | 0.9675 [0.2375-1.0000] (n 36) | 0.9991 [0.0950-1.0000] (n 36) | 3.5000 [1.0000-7.0000] (n 36) | 0.0005 [0.0001-0.0010] (n 36) |
| all 36 | new | 7.0000 [1.0000-12.0000] (n 36) | 10.5000 [2.0000-12.0000] (n 36) | 0.0267 [0.0001-0.8750] (n 36) | 0.9963 [0.7975-1.0000] (n 36) | 1.0000 [0.9830-1.0000] (n 36) | 3.5000 [1.0000-7.0000] (n 36) | 0.0003 [0.0001-0.0010] (n 36) |

## (d) Test, not used to choose or judge the recipe

the test: only from the public aggregate.json of the two runs (never from the restricted test scores), the verdicts and the median differences over the seeds of A\* - B, B' - B and A\* - B' side by side, in a separate section, not used to choose or judge the recipe.

Source: public/aggregate.json of the two runs (never the restricted test scores). Verdicts that differ from the parent run: 7 of 84.

| primary (A\* - B, stratum 03) | parent | verification run |
|---|---|---|
| auroc | inconclusive | inconclusive |
| acer_dev | inconclusive | inconclusive |

| comparison | stratum | metric | parent verdict | parent median difference | new verdict | new median difference |
|---|---|---|---|---|---|---|
| A\* - B | 01 | apcer_05 | inconclusive | 0.0070 | inconclusive | -0.0468 |
| A\* - B | 01 | bpcer_05 | inconclusive | -0.1046 | inconclusive | -0.0495 |
| A\* - B | 01 | acer_05 | inconclusive | -0.0488 | inconclusive | -0.0481 |
| A\* - B | 01 | apcer_dev | inconclusive | 0.2199 | inconclusive | 0.0012 |
| A\* - B | 01 | bpcer_dev | inconclusive | 0.0439 | inconclusive | -0.0090 |
| A\* - B | 01 | acer_dev | inconclusive | 0.0944 | inconclusive | -0.0197 |
| A\* - B | 01 | auroc | inconclusive | -0.2295 | inconclusive | 0.0038 |
| A\* - B | 02 | apcer_05 | inconclusive | 0.0011 | inconclusive | 0.0011 |
| A\* - B | 02 | bpcer_05 | inconclusive | 0.0890 | inconclusive | 0.0012 |
| A\* - B | 02 | acer_05 | inconclusive | 0.0528 | inconclusive | 0.0011 |
| A\* - B | 02 | apcer_dev | inconclusive | 0.2172 | inconclusive | 0.0022 |
| A\* - B | 02 | bpcer_dev | inconclusive | -0.0047 | inconclusive | -0.0035 |
| A\* - B | 02 | acer_dev | inconclusive | 0.1063 | inconclusive | -0.0012 |
| A\* - B | 02 | auroc | inconclusive | -0.1199 | inconclusive | -0.0065 |
| A\* - B | 03 | apcer_05 | inconclusive | 0.0859 | inconclusive | -0.0194 |
| A\* - B | 03 | bpcer_05 | inconclusive | -0.4471 | consistent (A\* lower BPCER (0.5) than B) | -0.4170 |
| A\* - B | 03 | acer_05 | inconclusive | -0.1956 | consistent (A\* lower ACER (0.5) than B) | -0.2381 |
| A\* - B | 03 | apcer_dev | inconclusive | 0.3352 | inconclusive | 0.1210 |
| A\* - B | 03 | bpcer_dev | inconclusive | -0.3367 | inconclusive | -0.4432 |
| A\* - B | 03 | acer_dev | inconclusive | -0.0854 | inconclusive | -0.2007 |
| A\* - B | 03 | auroc | inconclusive | 0.2064 | inconclusive | 0.2186 |
| A\* - B | all | apcer_05 | inconclusive | 0.0666 | inconclusive | -0.0172 |
| A\* - B | all | bpcer_05 | inconclusive | -0.2168 | consistent (A\* lower BPCER (0.5) than B) | -0.2993 |
| A\* - B | all | acer_05 | inconclusive | -0.0981 | consistent (A\* lower ACER (0.5) than B) | -0.1575 |
| A\* - B | all | apcer_dev | inconclusive | 0.3427 | inconclusive | 0.0973 |
| A\* - B | all | bpcer_dev | inconclusive | -0.2846 | inconclusive | -0.3571 |
| A\* - B | all | acer_dev | inconclusive | -0.0279 | inconclusive | -0.1622 |
| A\* - B | all | auroc | inconclusive | 0.0202 | inconclusive | 0.1105 |
| B' - B | 01 | apcer_05 | inconclusive | -0.0339 | inconclusive | -0.0164 |
| B' - B | 01 | bpcer_05 | inconclusive | 0.0562 | inconclusive | 0.0562 |
| B' - B | 01 | acer_05 | inconclusive | -0.0395 | inconclusive | 0.0261 |
| B' - B | 01 | apcer_dev | inconclusive | 0.0959 | inconclusive | 0.0491 |
| B' - B | 01 | bpcer_dev | inconclusive | -0.0956 | inconclusive | 0.0439 |
| B' - B | 01 | acer_dev | inconclusive | 0.0660 | inconclusive | 0.0465 |
| B' - B | 01 | auroc | inconclusive | -0.0814 | inconclusive | -0.0310 |
| B' - B | 02 | apcer_05 | inconclusive | 0.0011 | inconclusive | 0.0011 |
| B' - B | 02 | bpcer_05 | inconclusive | 0.1710 | inconclusive | 0.0070 |
| B' - B | 02 | acer_05 | inconclusive | -0.0179 | inconclusive | 0.0046 |
| B' - B | 02 | apcer_dev | inconclusive | 0.0101 | inconclusive | 0.0022 |
| B' - B | 02 | bpcer_dev | inconclusive | 0.0152 | inconclusive | 0.0023 |
| B' - B | 02 | acer_dev | inconclusive | 0.1342 | inconclusive | 0.0006 |
| B' - B | 02 | auroc | inconclusive | -0.1230 | inconclusive | -0.0012 |
| B' - B | 03 | apcer_05 | inconclusive | 0.0064 | inconclusive | -0.0424 |
| B' - B | 03 | bpcer_05 | inconclusive | 0.0042 | inconclusive | 0.0291 |
| B' - B | 03 | acer_05 | inconclusive | -0.0088 | inconclusive | -0.0066 |
| B' - B | 03 | apcer_dev | consistent (B' higher APCER (dev threshold) than B) | 0.2352 | inconclusive | 0.0616 |
| B' - B | 03 | bpcer_dev | inconclusive | -0.0547 | inconclusive | 0.0187 |
| B' - B | 03 | acer_dev | inconclusive | 0.0521 | inconclusive | 0.0312 |
| B' - B | 03 | auroc | inconclusive | -0.0699 | inconclusive | -0.1084 |
| B' - B | all | apcer_05 | inconclusive | 0.0115 | inconclusive | -0.0296 |
| B' - B | all | bpcer_05 | inconclusive | 0.0411 | inconclusive | 0.0576 |
| B' - B | all | acer_05 | inconclusive | -0.0129 | inconclusive | 0.0140 |
| B' - B | all | apcer_dev | inconclusive | 0.1823 | inconclusive | 0.0527 |
| B' - B | all | bpcer_dev | inconclusive | -0.0511 | inconclusive | 0.0678 |
| B' - B | all | acer_dev | inconclusive | 0.0656 | inconclusive | 0.0563 |
| B' - B | all | auroc | inconclusive | -0.1223 | inconclusive | -0.1122 |
| A\* - B' | 01 | apcer_05 | inconclusive | 0.2047 | inconclusive | 0.0012 |
| A\* - B' | 01 | bpcer_05 | inconclusive | -0.2261 | inconclusive | -0.0461 |
| A\* - B' | 01 | acer_05 | inconclusive | -0.0093 | inconclusive | -0.0549 |
| A\* - B' | 01 | apcer_dev | inconclusive | 0.1158 | inconclusive | -0.0795 |
| A\* - B' | 01 | bpcer_dev | inconclusive | -0.0169 | inconclusive | -0.0484 |
| A\* - B' | 01 | acer_dev | inconclusive | 0.0536 | inconclusive | -0.0552 |
| A\* - B' | 01 | auroc | inconclusive | -0.0297 | inconclusive | 0.0348 |
| A\* - B' | 02 | apcer_05 | inconclusive | 0.1892 | inconclusive | 0.0000 |
| A\* - B' | 02 | bpcer_05 | inconclusive | -0.0340 | inconclusive | 0.0527 |
| A\* - B' | 02 | acer_05 | inconclusive | 0.0618 | inconclusive | 0.0297 |
| A\* - B' | 02 | apcer_dev | inconclusive | 0.0918 | inconclusive | -0.0011 |
| A\* - B' | 02 | bpcer_dev | inconclusive | -0.0948 | inconclusive | 0.0012 |
| A\* - B' | 02 | acer_dev | inconclusive | -0.0015 | inconclusive | -0.0011 |
| A\* - B' | 02 | auroc | inconclusive | -0.1035 | inconclusive | -0.0053 |
| A\* - B' | 03 | apcer_05 | inconclusive | 0.0505 | inconclusive | 0.0090 |
| A\* - B' | 03 | bpcer_05 | inconclusive | -0.3953 | inconclusive | -0.6701 |
| A\* - B' | 03 | acer_05 | inconclusive | -0.1192 | inconclusive | -0.3306 |
| A\* - B' | 03 | apcer_dev | inconclusive | 0.0854 | inconclusive | 0.0594 |
| A\* - B' | 03 | bpcer_dev | inconclusive | -0.1618 | inconclusive | -0.6511 |
| A\* - B' | 03 | acer_dev | inconclusive | -0.1757 | inconclusive | -0.2959 |
| A\* - B' | 03 | auroc | inconclusive | 0.2325 | consistent (A\* higher AUROC than B') | 0.3689 |
| A\* - B' | all | apcer_05 | inconclusive | 0.0846 | inconclusive | 0.0087 |
| A\* - B' | all | bpcer_05 | inconclusive | -0.2580 | inconclusive | -0.4405 |
| A\* - B' | all | acer_05 | inconclusive | -0.0610 | inconclusive | -0.2159 |
| A\* - B' | all | apcer_dev | inconclusive | 0.0896 | inconclusive | 0.0446 |
| A\* - B' | all | bpcer_dev | inconclusive | -0.0639 | inconclusive | -0.4284 |
| A\* - B' | all | acer_dev | inconclusive | -0.0935 | inconclusive | -0.1919 |
| A\* - B' | all | auroc | inconclusive | 0.1324 | consistent (A\* higher AUROC than B') | 0.2046 |

## (e) Times

if the verification run is INCOMPLETE (cap or voluntary stop) or FAIL, no verdict is given and what is complete is reported.

| run | arm | models | epochs | train_seconds | s/epoch (weighted) |
|---|---|---|---|---|---|
| parent | B | 12 | 121 | 3926.9000 | 32.4537 |
| parent | A\* | 12 | 109 | 3472.0000 | 31.8532 |
| parent | B' | 12 | 96 | 2038.8000 | 21.2375 |
| new | B | 12 | 144 | 6135.8000 | 42.6097 |
| new | A\* | 12 | 144 | 5762.6000 | 40.0181 |
| new | B' | 12 | 144 | 4967.8000 | 34.4986 |

Verification run: training 16866.2 s, wall 17940.9 s against the estimate of 13090 s (difference +4850.9 s), cap 21600 s (within: yes); final writes after the cap: no; stopped: no. Parent run wall 10209.2 s.

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
- limits: the criteria are those of the previous package, defined after the parent run; with patience 12 the checkpoint is chosen on the dev among 12 epochs (more optimistic selection on the dev, already declared); a recovery does not separate the stop from the lr halved by ReduceLROnPlateau; the test section is descriptive and never part of the verdict
