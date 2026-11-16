# Facts for the saturation control note (generated; every number from the public sources in `sources.json`)

Runs: patience 4 `20261106-SESSABLRUN-d999458`, patience 12 `20261109-SESSABLRUN-6a50265`, patience 12 at a constant learning rate `20261111-SESSABLRUN-8568ffa`; degenerate study `20261107-DEGEN-a3b4dc0`.

| fact | value |
|---|---|
| models of the patience-4 run | 36 |
| degenerate (restored epoch 1 or 2 with dev accuracy 0.5 (within 1e-06)) | 10 of 36 (B 2, A* 5, B' 3) |
| restored epoch of degenerate | epoch 1: 8, epoch 2: 2 |
| dev accuracy 0.5 at the restored epoch, degenerate | 10 of 10 |
| best val_loss of degenerate | median 14.7178 [2.3605-60.9868] |
| train accuracy at epoch 1, degenerate | median 0.9495 [0.9143-0.9807] |
| train accuracy at epoch 1, non-degenerate | median 0.9691 [0.9021-0.9886] |
| train accuracy at the restored epoch, degenerate | median 0.9548 [0.9143-0.9943] |
| dev AUROC at the checkpoint, degenerate | 0.2114, 0.4690, 0.4975, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.9652, 0.9975 (above 0.9: 2; exactly 0.5: 5; below 0.5: 3) |
| dev accuracy 0.5 at epoch 1 | degenerate 10 of 10, non-degenerate 24 of 26 |
| non-degenerate at the recovery criterion at their checkpoint | 25 of 26 |
| non-degenerate with a dev accuracy of 0.5 at epoch 3 or later | 24 of 26 |
| BN momentum (Keras default, no momentum argument in LivenessNet) | 0.99 |
| BN residual after epochs 1, 2, 3, B (88 steps per epoch) | 0.4129, 0.1705, 0.0704 |
| BN residual after epochs 1, 2, 3, A* (88 steps per epoch) | 0.4129, 0.1705, 0.0704 |
| BN residual after epochs 1, 2, 3, B' (59 steps per epoch) | 0.5527, 0.3055, 0.1688 |
| BN residual at the restored epoch, degenerate | median 0.4129 [0.1705-0.5527] |
| BatchNormalization layers (position) | bn1_1 (4), bn1_2 (7), bn2_1 (12), bn2_2 (15), bn_dense (20) |
| first BN layer after each dropout | bn2_1 after dropout1, bn_dense after dropout2 |
| BN layers downstream of a dropout | bn2_1, bn2_2, bn_dense |
| Dropout layers (position) | dropout1 (9), dropout2 (17), dropout_dense (21) |
| fitting frames per model | B 1,400, A* 1,400, B' 934 |
| dev frames per fold; batch; scoring batch; device | 400; 16; 16; CPU |
| patience 12: degenerate with an epoch at 0.5; epochs at 0.5 per model; at epoch 1; last epoch at 0.5 (arm, BN residual) | 10 of 10; 2-7 (median 4); 10 of 10; epoch 7 (B', 0.0158) |
| patience 12, constant lr: degenerate with an epoch at 0.5; epochs at 0.5 per model; at epoch 1; last epoch at 0.5 (arm, BN residual) | 10 of 10; 2-5 (median 4); 10 of 10; epoch 6 (B', 0.0285) |
| dev subjects per fold | 2-3 |
| non-degenerate outside the recovery criterion at its checkpoint | B fold 3 seed 44: restored epoch 3, dev accuracy 0.2375, val_loss 3.2731, dev AUROC 0.0950 |
| cost inputs (patience-4 run record) | threshold step (load + 2 dev passes of 400 frames, CPU) median 4.1 s, max 7.9 s; slowest per-epoch mean of the arm (train_seconds / epochs_run, maximum over its models) B 36.9 s, A* 36.5 s, B' 23.0 s |
| passes per model | 5 recalibration and 2 evaluation passes over the fitting set, 2 over the dev |
| cost per model, B (10,600 frame passes) | central 58 s, upper 266 s |
| cost per model, A* (10,600 frame passes) | central 58 s, upper 263 s |
| cost per model, B' (7,338 frame passes) | central 42 s, upper 169 s |
| cost, 36 models (derived, not measured) | central 32 minutes, upper 140 minutes |

Central: frame passes x (median threshold step / 800 frame passes) + median threshold step. Upper: the 7 fitting passes each priced as the slowest per-epoch mean of the arm + the slowest threshold step (which holds the two dev passes). Neither prices the reading of the fitting frames.
