# Degenerate trainings of the session ablation run: train/dev curves (20261107-DEGEN-a3b4dc0)

Observations from the training cards already recorded; no training, no inference, no test score or test metric read. The possible explanations are not computed here. Criteria defined after the run: restored_epoch in {1, 2} and val_acc_at_best = 0.5 (within 1e-6): the rule written in the delivery of the session ablation run, re-applied here; recovery: first epoch with val_accuracy >= 0.75 and val_loss < 1.0 (dev only).

Inputs: 20261106-SESSABLRUN-d999458 (36 cards), 20261103-FOLDSRUN-e174812 (12 cards), 20261106-SESSABLRUN-3b4021f (6 cards, determinism only). Early stopping replay: 54/54 cards reproduced. Figures: figure_examples.png, figure_val_accuracy_all.png, figure_val_loss_all.png written.

## T1. Degenerate trainings (criterion re-applied)

| arm | fold | seed | epochs_run | restored_epoch | best_val_loss | val_acc_at_best |
|---|---|---|---|---|---|---|
| arm_a_star | 2 | 43 | 5 | 1 | 60.9868 | 0.5000 |
| arm_a_star | 2 | 44 | 5 | 1 | 18.0934 | 0.5000 |
| arm_a_star | 3 | 42 | 5 | 1 | 3.1296 | 0.5000 |
| arm_a_star | 3 | 44 | 5 | 1 | 33.4090 | 0.5000 |
| arm_a_star | 4 | 44 | 5 | 1 | 22.8274 | 0.5000 |
| arm_b | 1 | 42 | 5 | 1 | 7.0611 | 0.5000 |
| arm_b | 4 | 44 | 6 | 2 | 2.3605 | 0.5000 |
| arm_b_prime | 1 | 44 | 5 | 1 | 12.1449 | 0.5000 |
| arm_b_prime | 2 | 44 | 6 | 2 | 11.4259 | 0.5000 |
| arm_b_prime | 4 | 44 | 5 | 1 | 17.2907 | 0.5000 |

Count: 10 of 36. Per arm: arm_b 2, arm_a_star 5, arm_b_prime 3; per seed: seed 42 2, seed 43 1, seed 44 7; per fold: fold 1 2, fold 2 3, fold 3 2, fold 4 3.

Boundary cases (non-degenerate with restored_epoch <= 3): arm_b f3 s44 restored 3, val_acc 0.2375, val_loss 3.273; arm_b_prime f1 s42 restored 2, val_acc 0.7725, val_loss 0.61; arm_b_prime f3 s42 restored 1, val_acc 0.8325, val_loss 0.321; arm_b_prime f4 s42 restored 2, val_acc 0.9175, val_loss 0.138; arm_b_prime f4 s43 restored 3, val_acc 1.0, val_loss 0.014.

## T2. Fold x seed per arm (D = degenerate, number = restored epoch)

| arm | seed | fold1 | fold2 | fold3 | fold4 |
|---|---|---|---|---|---|
| arm_b | 42 | D1 | 9 | 12 | 11 |
| arm_b | 43 | 11 | 7 | 10 | 6 |
| arm_b | 44 | 6 | 10 | 3 | D2 |
| arm_a_star | 42 | 10 | 11 | D1 | 10 |
| arm_a_star | 43 | 9 | D1 | 12 | 9 |
| arm_a_star | 44 | 11 | D1 | D1 | D1 |
| arm_b_prime | 42 | 2 | 11 | 1 | 2 |
| arm_b_prime | 43 | 4 | 10 | 9 | 3 |
| arm_b_prime | 44 | D1 | D2 | 9 | D1 |

## T3. Dev saturation (val_accuracy = 0.5) and the training side in the same epochs

| group | models | epoch 1 at 0.5 | epoch 2 at 0.5 | epochs at 0.5 | epochs | val_loss min | val_loss median | val_loss max | train acc min | train acc median | train loss max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| degenerate | 10 | 10 | 9 | 41 | 52 | 2.3605 | 25.1748 | 197.8555 | 0.9143 | 0.9879 | 0.2393 |
| non-degenerate | 26 | 24 | 22 | 93 | 274 | 2.2389 | 17.9936 | 105.2555 | 0.9021 | 0.9879 | 0.2460 |
| folds run | 12 | 12 | 12 | 30 | 144 | 3.0750 | 49.2110 | 264.1550 | 0.9243 | 0.9745 | 0.2064 |

Models (48) with at least one epoch at 0.5: 48; epochs at 0.5 with train accuracy > 0.9: 164 of 164.

Epochs with val_accuracy < 0.45: arm_a_star f1 s44 epoch 3 (0.2775); arm_a_star f2 s44 epoch 5 (0.28); arm_a_star f3 s42 epoch 5 (0.2625); arm_a_star f4 s43 epoch 4 (0.28); arm_b f3 s44 epoch 3 (0.2375); arm_b_prime f3 s43 epoch 5 (0.24).

## T4. First recovery on the dev, per group

Criterion: first epoch with val_accuracy >= 0.75 and val_loss < 1.0 (dev only).

| group | models | with recovery | min | median | max | median in steps | recovery after epoch 5 |
|---|---|---|---|---|---|---|---|
| arm_b | 12 | 9 | 5 | 6.0000 | 7 | 528.0000 | 7 |
| arm_a_star | 12 | 7 | 5 | 7.0000 | 10 | 616.0000 | 6 |
| arm_b_prime | 12 | 9 | 1 | 3.0000 | 11 | 177.0000 | 3 |
| seed 42 | 12 | 10 | 1 | 5.5000 | 11 | 484.0000 | 5 |
| seed 43 | 12 | 11 | 3 | 6.0000 | 7 | 528.0000 | 7 |
| seed 44 | 12 | 4 | 6 | 6.5000 | 7 | 528.0000 | 4 |
| fold 1 | 9 | 7 | 2 | 6.0000 | 7 | 528.0000 | 4 |
| fold 2 | 9 | 6 | 3 | 5.5000 | 11 | 484.0000 | 3 |
| fold 3 | 9 | 6 | 1 | 6.0000 | 7 | 470.5000 | 5 |
| fold 4 | 9 | 6 | 1 | 6.0000 | 10 | 528.0000 | 4 |
| non-degenerate | 26 | 25 | 1 | 6.0000 | 11 | 528.0000 | 16 |
| degenerate | 10 | 0 | - | - | - | - | 0 |
| folds run | 12 | 12 | 3 | 5.5000 | 7 | 726.0000 | 6 |
| arm_b and arm_a_star non-degenerate (88 steps) | 17 | 16 | 5 | 6.0000 | 10 | 528.0000 | 13 |
| arm_b_prime non-degenerate (59 steps) | 9 | 9 | 1 | 3.0000 | 11 | 177.0000 | 3 |

First recovery per model, folds run: [3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7].
First recovery per model, arm_b and arm_a_star non-degenerate: [5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 10].
First recovery per model, arm_b_prime non-degenerate: [1, 1, 2, 3, 3, 4, 7, 7, 11].

## T5. Sensitivity of the recovery criterion

| criterion | degenerate | non-degenerate | folds run |
|---|---|---|---|
| acc>=0.70, loss<0.7 | 0/10 med - | 22/26 med 6.0000 | 12/12 med 6.0000 |
| acc>=0.70, loss<1.0 | 0/10 med - | 25/26 med 6.0000 | 12/12 med 5.5000 |
| acc>=0.70, loss<1.5 | 0/10 med - | 25/26 med 6.0000 | 12/12 med 5.0000 |
| acc>=0.75, loss<0.7 | 0/10 med - | 22/26 med 6.0000 | 12/12 med 6.0000 |
| acc>=0.75, loss<1.0 | 0/10 med - | 25/26 med 6.0000 | 12/12 med 5.5000 |
| acc>=0.75, loss<1.5 | 0/10 med - | 25/26 med 6.0000 | 12/12 med 5.0000 |
| acc>=0.80, loss<0.7 | 0/10 med - | 21/26 med 6.0000 | 12/12 med 6.0000 |
| acc>=0.80, loss<1.0 | 0/10 med - | 23/26 med 6.0000 | 12/12 med 5.5000 |
| acc>=0.80, loss<1.5 | 0/10 med - | 23/26 med 6.0000 | 12/12 med 5.0000 |

## T6. Mechanics of the stop (degenerate trainings)

| arm | fold | seed | restored | epochs_run | val_loss per epoch | val_acc per epoch | val_loss last vs previous | val_acc last vs previous |
|---|---|---|---|---|---|---|---|---|
| arm_a_star | 2 | 43 | 1 | 5 | 61.0 107.3 129.9 141.1 82.6 | 0.500 0.500 0.500 0.500 0.505 | down | up |
| arm_a_star | 2 | 44 | 1 | 5 | 18.1 54.3 53.4 31.7 67.1 | 0.500 0.500 0.500 0.598 0.280 | up | down |
| arm_a_star | 3 | 42 | 1 | 5 | 3.1 4.1 48.6 13.5 70.9 | 0.500 0.498 0.500 0.500 0.262 | up | down |
| arm_a_star | 3 | 44 | 1 | 5 | 33.4 44.9 197.9 133.1 66.0 | 0.500 0.500 0.500 0.500 0.493 | down | down |
| arm_a_star | 4 | 44 | 1 | 5 | 22.8 61.3 40.9 61.1 37.1 | 0.500 0.500 0.500 0.500 0.500 | down | = |
| arm_b | 1 | 42 | 1 | 5 | 7.1 11.1 25.3 70.4 58.0 | 0.500 0.500 0.500 0.500 0.500 | down | = |
| arm_b | 4 | 44 | 2 | 6 | 16.5 2.4 3.5 14.5 4.2 4.6 | 0.500 0.500 0.480 0.520 0.515 0.815 | up | up |
| arm_b_prime | 1 | 44 | 1 | 5 | 12.1 22.7 19.3 22.3 25.9 | 0.500 0.500 0.500 0.500 0.500 | up | = |
| arm_b_prime | 2 | 44 | 2 | 6 | 25.5 11.4 15.2 18.5 13.8 24.8 | 0.500 0.500 0.500 0.500 0.500 0.500 | up | = |
| arm_b_prime | 4 | 44 | 1 | 5 | 17.3 24.6 19.4 25.2 30.1 | 0.500 0.500 0.500 0.500 0.502 | up | up |

non-degenerate: first epoch with val_loss < val_loss of epoch 1 -> [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 5, 5, 5, None]; with val_accuracy still 0.5 in that epoch: 16; models with val_accuracy 0.5 at epoch 1: 24.

folds run: first epoch with val_loss < val_loss of epoch 1 -> [2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4]; with val_accuracy still 0.5 in that epoch: 6; models with val_accuracy 0.5 at epoch 1: 12.

degenerate: first epoch with val_loss < val_loss of epoch 1 -> [2, 2, None, None, None, None, None, None, None, None]; with val_accuracy still 0.5 in that epoch: 2; models with val_accuracy 0.5 at epoch 1: 10.

Non-degenerate trainings (by the criterion; ablation run and folds run) with the first recovery after epoch 5: 22 of 37 with recovery; never recovered: arm_b f3 s44.

## T7. Alternative early stopping rules replayed on the recorded histories (exact or not determinable)

| rule | degenerate | non-degenerate | folds run |
|---|---|---|---|
| patience 4 (run) | 10/10 determinable, 10 degenerate, restored median 1.0000 | 26/26 determinable, 0 degenerate, restored median 9.0000 | 12/12 determinable, 0 degenerate, restored median 10.0000 |
| patience 2 | 10/10 determinable, 10 degenerate, restored median 1.0000 | 26/26 determinable, 5 degenerate, restored median 3.5000 | 12/12 determinable, 1 degenerate, restored median 6.5000 |
| patience 3 | 10/10 determinable, 10 degenerate, restored median 1.0000 | 26/26 determinable, 3 degenerate, restored median 6.5000 | 12/12 determinable, 0 degenerate, restored median 10.0000 |
| patience 6 | 0/10 determinable, 0 degenerate, restored median - | 17/26 determinable, 0 degenerate, restored median 10.0000 | 12/12 determinable, 0 degenerate, restored median 10.0000 |
| patience 8 | 0/10 determinable, 0 degenerate, restored median - | 17/26 determinable, 0 degenerate, restored median 10.0000 | 12/12 determinable, 0 degenerate, restored median 10.0000 |
| start_from_epoch 3 | 0/10 determinable, 0 degenerate, restored median - | 21/26 determinable, 0 degenerate, restored median 10.0000 | 12/12 determinable, 0 degenerate, restored median 10.0000 |
| start_from_epoch 5 | 0/10 determinable, 0 degenerate, restored median - | 20/26 determinable, 0 degenerate, restored median 10.0000 | 12/12 determinable, 0 degenerate, restored median 10.0000 |
| monitor val_accuracy | 6/10 determinable, 6 degenerate, restored median 1.0000 | 24/26 determinable, 4 degenerate, restored median 6.5000 | 12/12 determinable, 0 degenerate, restored median 10.5000 |

Non-degenerate: restored epoch of the run -> of the rule ('?' = not determinable):
- patience 2: arm_a_star f1 s44: 11 -> 1 (val_acc 0.9875 -> 0.5000); arm_a_star f2 s42: 11 -> 7 (val_acc 1.0000 -> 1.0000); arm_a_star f3 s43: 12 -> 6 (val_acc 0.9975 -> 0.8275); arm_a_star f4 s42: 10 -> 2 (val_acc 0.7975 -> 0.5000); arm_b f1 s43: 11 -> 6 (val_acc 0.9975 -> 0.7200); arm_b f2 s42: 9 -> 2 (val_acc 0.9475 -> 0.5000); arm_b f2 s43: 7 -> 1 (val_acc 1.0000 -> 0.5000); arm_b f2 s44: 10 -> 1 (val_acc 0.9975 -> 0.5000); arm_b f3 s42: 12 -> 3 (val_acc 1.0000 -> 0.5000); arm_b f3 s43: 10 -> 6 (val_acc 0.9925 -> 0.9050); arm_b f4 s42: 11 -> 3 (val_acc 1.0000 -> 0.7025); arm_b_prime f2 s42: 11 -> 6 (val_acc 0.8775 -> 0.6225); arm_b_prime f2 s43: 10 -> 6 (val_acc 1.0000 -> 0.9950); arm_b_prime f3 s44: 9 -> 3 (val_acc 0.9975 -> 0.5000)
- patience 3: arm_a_star f1 s44: 11 -> 1 (val_acc 0.9875 -> 0.5000); arm_a_star f2 s42: 11 -> 7 (val_acc 1.0000 -> 1.0000); arm_b f2 s42: 9 -> 5 (val_acc 0.9475 -> 0.9250); arm_b f2 s43: 7 -> 1 (val_acc 1.0000 -> 0.5000); arm_b f2 s44: 10 -> 1 (val_acc 0.9975 -> 0.5000); arm_b_prime f2 s43: 10 -> 6 (val_acc 1.0000 -> 0.9950)
- patience 6: arm_b f1 s44: 6 -> ?; arm_b f2 s43: 7 -> ?; arm_b f3 s44: 3 -> ?; arm_b f4 s43: 6 -> ?; arm_b_prime f1 s42: 2 -> ?; arm_b_prime f1 s43: 4 -> ?; arm_b_prime f3 s42: 1 -> ?; arm_b_prime f4 s42: 2 -> ?; arm_b_prime f4 s43: 3 -> ?
- patience 8: arm_b f1 s44: 6 -> ?; arm_b f2 s43: 7 -> ?; arm_b f3 s44: 3 -> ?; arm_b f4 s43: 6 -> ?; arm_b_prime f1 s42: 2 -> ?; arm_b_prime f1 s43: 4 -> ?; arm_b_prime f3 s42: 1 -> ?; arm_b_prime f4 s42: 2 -> ?; arm_b_prime f4 s43: 3 -> ?
- start_from_epoch 3: arm_b f3 s44: 3 -> ?; arm_b_prime f1 s42: 2 -> ?; arm_b_prime f3 s42: 1 -> ?; arm_b_prime f4 s42: 2 -> ?; arm_b_prime f4 s43: 3 -> ?
- start_from_epoch 5: arm_b f3 s44: 3 -> ?; arm_b_prime f1 s42: 2 -> ?; arm_b_prime f1 s43: 4 -> ?; arm_b_prime f3 s42: 1 -> ?; arm_b_prime f4 s42: 2 -> ?; arm_b_prime f4 s43: 3 -> ?
- monitor val_accuracy: arm_a_star f1 s42: 10 -> 9 (val_acc 0.9950 -> 0.9950); arm_a_star f1 s44: 11 -> 10 (val_acc 0.9875 -> 0.9925); arm_a_star f2 s42: 11 -> 6 (val_acc 1.0000 -> 1.0000); arm_a_star f3 s43: 12 -> 1 (val_acc 0.9975 -> 0.5000); arm_b f1 s44: 6 -> ?; arm_b f2 s42: 9 -> 10 (val_acc 0.9475 -> 0.9575); arm_b f3 s44: 3 -> ?; arm_b f4 s42: 11 -> 6 (val_acc 1.0000 -> 1.0000); arm_b_prime f2 s42: 11 -> 1 (val_acc 0.8775 -> 0.5000); arm_b_prime f3 s43: 9 -> 1 (val_acc 1.0000 -> 0.5000); arm_b_prime f3 s44: 9 -> 1 (val_acc 0.9975 -> 0.5000); arm_b_prime f4 s42: 2 -> 1 (val_acc 0.9175 -> 0.9325)

Non-degenerate trainings (by the criterion; ablation run and folds run) that would become degenerate under the rule:
- patience 2: arm_a_star f1 s44; arm_a_star f4 s42; arm_b f2 s42; arm_b f2 s43; arm_b f2 s44; folds f4 s43 (foldsrun)
- patience 3: arm_a_star f1 s44; arm_b f2 s43; arm_b f2 s44
- patience 6: none
- patience 8: none
- start_from_epoch 3: none
- start_from_epoch 5: none
- monitor val_accuracy: arm_a_star f3 s43; arm_b_prime f2 s42; arm_b_prime f3 s43; arm_b_prime f3 s44

## T8. Learning rate (history lr = the lr used in the epoch)

| group | models with a reduction | first epoch at reduced lr | reductions decided with the dev at 0.5 | lr at the last epoch |
|---|---|---|---|---|
| degenerate | 10 | [4, 4, 4, 4, 4, 4, 4, 4, 5, 5] | 9 | 5.00e-04: 10 |
| non-degenerate | 25 | [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 9, 9, 9, 9, 9, 9, 9, 10, 12, 12, 12] | 7 | 5.00e-04: 18, 2.50e-04: 6, 1.25e-04: 1, 1.00e-03: 1 |
| folds run | 11 | [4, 6, 7, 9, 9, 9, 10, 11, 11, 12, 12] | 1 | 5.00e-04: 7, 2.50e-04: 4, 1.00e-03: 1 |

## T9. Training side per epoch (median [min-max])

| epoch | degenerate | non-degenerate | folds run |
|---|---|---|---|
| 1 | acc 0.9495 [0.9143-0.9807]; loss 0.1361 [0.0584-0.2393] | acc 0.9691 [0.9021-0.9886]; loss 0.0904 [0.0312-0.2460] | acc 0.9374 [0.9243-0.9562]; loss 0.1636 [0.1213-0.2064] |
| 2 | acc 0.9889 [0.9650-0.9943]; loss 0.0344 [0.0204-0.1015] | acc 0.9893 [0.9643-0.9989]; loss 0.0343 [0.0103-0.1072] | acc 0.9760 [0.9662-0.9800]; loss 0.0716 [0.0638-0.0970] |
| 3 | acc 0.9875 [0.9707-0.9946]; loss 0.0419 [0.0155-0.0818] | acc 0.9889 [0.9714-0.9986]; loss 0.0364 [0.0099-0.0864] | acc 0.9798 [0.9757-0.9838]; loss 0.0608 [0.0447-0.0790] |
| 4 | acc 0.9902 [0.9850-0.9989]; loss 0.0376 [0.0081-0.0460] | acc 0.9918 [0.9779-0.9986]; loss 0.0254 [0.0078-0.0594] | acc 0.9807 [0.9729-0.9895]; loss 0.0573 [0.0462-0.0844] |
| 5 | acc 0.9943 [0.9829-0.9989]; loss 0.0186 [0.0068-0.0481] | acc 0.9929 [0.9775-1.0000]; loss 0.0240 [0.0034-0.0638] | acc 0.9845 [0.9776-0.9914]; loss 0.0466 [0.0324-0.0689] |

Pairs with the same arm and fold (degenerate against another seed), train accuracy of epochs 1-3:
- arm_a_star f2: degenerate s43 [0.9143, 0.9671, 0.9707] against s42 [0.9321, 0.9836, 0.9871]
- arm_a_star f2: degenerate s44 [0.92, 0.9807, 0.9829] against s42 [0.9321, 0.9836, 0.9871]
- arm_a_star f3: degenerate s42 [0.935, 0.965, 0.975] against s43 [0.9021, 0.9643, 0.9714]
- arm_a_star f3: degenerate s44 [0.9229, 0.975, 0.975] against s43 [0.9021, 0.9643, 0.9714]
- arm_a_star f4: degenerate s44 [0.945, 0.9929, 0.9857] against s42 [0.9636, 0.9857, 0.985]
- arm_a_star f4: degenerate s44 [0.945, 0.9929, 0.9857] against s43 [0.9479, 0.9829, 0.98]
- arm_b f1: degenerate s42 [0.9807, 0.9886, 0.9943] against s43 [0.9714, 0.9943, 0.9886]
- arm_b f1: degenerate s42 [0.9807, 0.9886, 0.9943] against s44 [0.9571, 0.9936, 0.9871]
- arm_b f4: degenerate s44 [0.9729, 0.9943, 0.9943] against s42 [0.9886, 0.9943, 0.9943]
- arm_b f4: degenerate s44 [0.9729, 0.9943, 0.9943] against s43 [0.9714, 0.9921, 0.9921]
- arm_b_prime f1: degenerate s44 [0.9647, 0.9893, 0.9893] against s42 [0.9732, 0.9946, 0.9957]
- arm_b_prime f1: degenerate s44 [0.9647, 0.9893, 0.9893] against s43 [0.9572, 0.9914, 0.9829]
- arm_b_prime f2: degenerate s44 [0.954, 0.9893, 0.9925] against s42 [0.9668, 0.9925, 0.9946]
- arm_b_prime f2: degenerate s44 [0.954, 0.9893, 0.9925] against s43 [0.9722, 0.9872, 0.9904]
- arm_b_prime f4: degenerate s44 [0.9732, 0.9893, 0.9946] against s42 [0.9807, 0.9946, 0.9936]
- arm_b_prime f4: degenerate s44 [0.9732, 0.9893, 0.9946] against s43 [0.9839, 0.9979, 0.985]

## T10. Checkpoint on the dev (dev_scores.csv: threshold 0.5 and dev AUROC)

| arm | fold | seed | deg | restored | acc 0.5 | share predicted attack | mean score bona fide | mean score attack | dev AUROC | distinct scores |
|---|---|---|---|---|---|---|---|---|---|---|
| arm_a_star | 2 | 43 | D | 1 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 1 |
| arm_a_star | 2 | 44 | D | 1 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.2114 | 400 |
| arm_a_star | 3 | 42 | D | 1 | 0.5000 | 0.0000 | 0.0030 | 0.0036 | 0.4690 | 400 |
| arm_a_star | 3 | 44 | D | 1 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 1 |
| arm_a_star | 4 | 44 | D | 1 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 1 |
| arm_b | 1 | 42 | D | 1 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 0.9975 | 33 |
| arm_b | 4 | 44 | D | 2 | 0.5000 | 1.0000 | 0.9853 | 0.9991 | 0.9652 | 397 |
| arm_b_prime | 1 | 44 | D | 1 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 0.4975 | 2 |
| arm_b_prime | 2 | 44 | D | 2 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 1 |
| arm_b_prime | 4 | 44 | D | 1 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 1 |
| arm_b | 3 | 44 | - | 3 | 0.2375 | 0.2975 | 0.6127 | 0.0974 | 0.0950 | 399 |
| arm_b_prime | 1 | 42 | - | 2 | 0.7725 | 0.2725 | 0.0010 | 0.5179 | 0.9937 | 400 |
| arm_b_prime | 3 | 42 | - | 1 | 0.8325 | 0.6675 | 0.4544 | 0.9834 | 1.0000 | 400 |
| arm_b_prime | 4 | 42 | - | 2 | 0.9175 | 0.5775 | 0.1697 | 0.9886 | 0.9966 | 399 |
| arm_b_prime | 4 | 43 | - | 3 | 1.0000 | 0.5000 | 0.0112 | 0.9854 | 1.0000 | 388 |

Dev AUROC of the non-degenerate: median 1.0000 [0.0950-1.0000]; folds run: median 0.9976 [0.9504-1.0000].

Distinct subjects in the dev per fold (count only): fold 1 2, fold 2 2, fold 3 2, fold 4 3; dev frames: fold 1 400, fold 2 400, fold 3 400, fold 4 400; folds run: fold 1 4, fold 2 4, fold 3 4, fold 4 4.

Agreement of the accuracy at 0.5 of the checkpoint (dev_scores) with val_acc_at_best of the card, largest difference: 2.1457672083613488e-08 (folds run 2.3047129316111636e-08).

## T11. Determinism: failed attempt against the run

| model | same epochs | same restored epoch | largest history difference | identical history | same weights SHA-256 | train_s failed attempt | train_s run |
|---|---|---|---|---|---|---|---|
| arm_a_star_fold1_seed42 | True | True | 0.0000 | True | False | 324.8000 | 363.1000 |
| arm_a_star_fold1_seed43 | True | True | 0.0000 | True | False | 369.4000 | 370.8000 |
| arm_a_star_fold1_seed44 | True | True | 0.0000 | True | False | 361.7000 | 386.2000 |
| arm_b_fold1_seed42 | True | True | 0.0000 | True | False | 125.3000 | 130.8000 |
| arm_b_fold1_seed43 | True | True | 0.0000 | True | False | 342.4000 | 372.7000 |
| arm_b_fold1_seed44 | True | True | 0.0000 | True | False | 319.7000 | 311.6000 |

Identical histories: 6 of 6 (a difference is reported, not an error).

## T12. Times

| arm | models | steps/epoch | epochs | train_seconds | s/epoch median | min | max | s/epoch weighted |
|---|---|---|---|---|---|---|---|---|
| arm_b | 12 | 88 | 121 | 3926.9000 | 31.9618 | 26.1600 | 36.8667 | 32.4537 |
| arm_a_star | 12 | 88 | 109 | 3472.0000 | 31.9583 | 29.8917 | 36.4800 | 31.8532 |
| arm_b_prime | 12 | 59 | 96 | 2038.8000 | 20.7875 | 20.1600 | 22.9917 | 21.2375 |
| folds | 12 | 132 | 144 | 9111.9000 | 62.4083 | 58.5083 | 69.8500 | 63.2771 |

Run: train_seconds 9437.7 s, wall 10209.2 s (status.json; the run.log line gives 10209 s, whole seconds, used for the rest below), rest (checks, loading, scoring, aggregate) 771.3 s = 21.4 s per model.
From the saved weights to the threshold (dev scores): median 4 s [3-8]; from the threshold to the test scores: median 16 s [13-21] (resolution 1 s).
- Scenario 36 models x 12 epochs (36 models, 10 degenerate): training 12318 s (3.42 h), with the rest per model of the run 13090 s (3.64 h).
- Scenario arm_a_star and arm_b, 24 models x 12 epochs (24 models, 7 degenerate): training 9260 s (2.57 h), with the rest per model of the run 9774 s (2.72 h).
- Scenario the 10 degenerate models x 12 epochs (10 models, 10 degenerate): training 3455 s (0.96 h).
- Scenario models stopped before epoch 12, x 12 epochs (19 models, 10 degenerate): training 6287 s (1.75 h), with the rest per model of the run 6694 s (1.86 h).
Original run: 9438 s of training over 326 epochs, wall 10209 s (2.84 h).

## T13. Residual weight of the initial value in the moving statistics of BatchNormalization (momentum 0.99)

| group | after epoch 1 | epoch 2 | epoch 3 | epoch 4 | epoch 5 |
|---|---|---|---|---|---|
| arm_b_prime (59 steps) | 0.5527 | 0.3055 | 0.1688 | 0.0933 | 0.0516 |
| arm_b and arm_a_star (88 steps) | 0.4129 | 0.1705 | 0.0704 | 0.0291 | 0.0120 |
| folds run (132 steps) | 0.2654 | 0.0704 | 0.0187 | 0.0050 | 0.0013 |

(0.99^steps: share of the moving average still due to the initial value (mean 0, variance 1) if the batch statistic were constant; with changing weights the moving average also chases a moving target.)

## Declarations

- no_training: True
- no_inference: True
- no_test_scores_read: True
- descriptive: True
- criteria_defined_after_the_run: True
- dev_columns_read: dev_scores.csv: label and score_attack for the measures; subject_id only to count the distinct subjects of each dev and to scan the public files, never written; arm, fold, train_seed, role, checkpoint_sha256, sample_id, file_sha256, prediction_05, prediction_dev, threshold_dev and threshold_sha256 only for row-by-row consistency checks (with the card, the frozen threshold and the other models of the fold). threshold.json (both runs): frozen_at for the times; arm, fold, train_seed, checkpoint_sha256, value and dev_scores_sha256 for consistency only. Training cards: the history and the recipe fields, train_seconds, saved and test_scored_after_threshold for the times, weights_file, weights_sha256 and dev_keys_sha256 for consistency only. run.log of the ablation run: the wall time line only.
- observations_not_explanations: the outputs report what the curves show; the possible explanations are hypotheses discussed elsewhere and are not tested by this study
- limits: the degenerate criterion was defined after the run and misses at least one failure of the same family; the recovery criterion is ours (sensitivity reported); the alternative early stopping rules are exact only where the recorded history covers the rule; determinism is compared on six models only; train and dev accuracy are measured in different modes (training mode with augmentation and dropout, inference mode on the dev); the folds run is a reference with other data, dev and steps per epoch, not a paired control
