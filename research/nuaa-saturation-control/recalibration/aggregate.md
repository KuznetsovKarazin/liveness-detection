# BatchNormalization recalibration of the 36 patience-4 checkpoints (20261114-BNRECAL-a10a082)

Status: **PASS**. Parent run `20261106-SESSABLRUN-d999458`; configuration `d97e2e7497fb`; revised note `55ddd8390150`. No training. The fold schema (fold<k>_test.csv, the test role of our folds) is never opened and no <model>_test_scores.csv is opened; the M2 manifests official_test and cnn_test (NUAA's official image lists, not the test role of our folds) are read only to index the cache; as in the parent run the frame files (cache and extracted frames) are hashed and loaded whole, but only fitting and dev frames are assembled; the public aggregate of the parent run, which holds test metrics, is read only as bytes for its SHA-256 and never parsed; no public file of the folds run is opened; no test frame is assembled or evaluated and no test score or metric is read, computed or written.

## Primary reading (fixed before the run)

primary: the number of the ten degenerate checkpoints that meet the recovery criterion on the dev after the re-estimation (dev accuracy >= 0.75 and val_loss < 1.0); applied mechanically by the script

**all ten (10 of 10) meet the recovery criterion on the dev after the re-estimation: on these checkpoints, re-estimating the statistics on the fitting set, every trained parameter unchanged, is sufficient to meet the criterion (the moving statistics, dropout and the augmentation of the fitting set, hypotheses 1-3, are not separated); the later saturated epochs, at residual <= 0.0704, are not examined. A recovery shows the effect of this recalibration on the examined checkpoints, not a single cause of the saturation**

Meeting the criterion after: arm_b_fold1_seed42, arm_a_star_fold2_seed43, arm_a_star_fold2_seed44, arm_a_star_fold3_seed42, arm_a_star_fold3_seed44, arm_b_fold4_seed44, arm_a_star_fold4_seed44, arm_b_prime_fold1_seed44, arm_b_prime_fold2_seed44, arm_b_prime_fold4_seed44; not meeting it: none.

## Controls and the model reported apart

secondary, not deciding: the controls (the 25 of the 26 meeting the criterion at their checkpoint); each one that no longer meets the criterion on the dev after the re-estimation is listed as a worsening

25 of the 25 controls keep the criterion, 0 lose it.

reported apart, not deciding: arm B fold 3 seed 44 (restored epoch 3, dev accuracy 0.2375, not degenerate, outside the criterion at its checkpoint), its dev values before and after

- arm_b_fold3_seed44: dev accuracy 0.2375 -> 0.9950, val_loss 3.2731 -> 0.0417, AUROC 0.0950 -> 0.9986

## Fitting before x dev after (2x2)

fitting before (original statistics, inference mode) x dev after: each cell counts checkpoints, no cell names a cause

| fitting before (original statistics, inference mode) | dev after: meets the criterion | dev after: does not meet it |
|---|---|---|
| meets the criterion | both: 0 of the ten, 22 of all | fitting only: 0 of the ten, 0 of all |
| does not meet it | dev after only: 10 of the ten, 14 of all | neither: 0 of the ten, 0 of all |


## Worsening (every case listed)

every worsening of every model is reported: criterion lost on the dev, fewer correct dev frames, higher dev val_loss, lower dev AUROC, fewer correct fitting frames, higher fitting loss; any change in that direction counts, however small, and aggregate.md gives its size (before -> after, difference) so that numerical noise can be recognised

| flag | models |
|---|---|
| gained_criterion | 11 |
| lost_criterion | 0 |
| dev_accuracy_lower | 11 |
| val_loss_higher | 15 |
| dev_auroc_lower | 7 |
| fitting_accuracy_lower | 3 |
| fitting_loss_higher | 4 |


- arm_a_star_fold1_seed42 (control): val_loss_higher 0.0104333 -> 0.0134321 (difference 0.00299882)
- arm_b_fold1_seed43 (control): dev_accuracy_lower 399 -> 397 (difference -2); val_loss_higher 0.00558041 -> 0.182965 (difference 0.177385); dev_auroc_lower 0.999975 -> 0.995 (difference -0.004975)
- arm_a_star_fold1_seed43 (control): dev_accuracy_lower 399 -> 398 (difference -1)
- arm_b_fold1_seed44 (control): val_loss_higher 0.029843 -> 0.344459 (difference 0.314616); dev_auroc_lower 0.998475 -> 0.994687 (difference -0.0037875); fitting_accuracy_lower 1 -> 0.999286 (difference -0.000714302); fitting_loss_higher 0.00126252 -> 0.00144349 (difference 0.000180971)
- arm_a_star_fold1_seed44 (control): val_loss_higher 0.0345145 -> 0.532653 (difference 0.498139); dev_auroc_lower 0.99915 -> 0.98945 (difference -0.0097); fitting_accuracy_lower 1 -> 0.998571 (difference -0.00142854); fitting_loss_higher 0.000694501 -> 0.00280055 (difference 0.00210605)
- arm_a_star_fold2_seed42 (control): val_loss_higher 0.000107111 -> 0.000317227 (difference 0.000210116); fitting_loss_higher 3.77205e-05 -> 0.000341481 (difference 0.000303761)
- arm_b_fold3_seed43 (control): dev_accuracy_lower 397 -> 387 (difference -10); val_loss_higher 0.0530546 -> 0.0639548 (difference 0.0109002)
- arm_a_star_fold3_seed43 (control): dev_accuracy_lower 399 -> 367 (difference -32); val_loss_higher 0.00883641 -> 0.166968 (difference 0.158132)
- arm_b_fold4_seed42 (control): dev_accuracy_lower 400 -> 399 (difference -1); val_loss_higher 0.00054232 -> 0.162219 (difference 0.161677); dev_auroc_lower 1 -> 0.995 (difference -0.005)
- arm_a_star_fold4_seed42 (control): dev_accuracy_lower 319 -> 315 (difference -4); dev_auroc_lower 0.999 -> 0.991625 (difference -0.007375); fitting_accuracy_lower 0.998571 -> 0.997143 (difference -0.0014286)
- arm_b_fold4_seed43 (control): dev_accuracy_lower 400 -> 399 (difference -1); val_loss_higher 0.00128923 -> 0.00466482 (difference 0.00337558)
- arm_a_star_fold4_seed43 (control): dev_accuracy_lower 398 -> 367 (difference -31); val_loss_higher 0.0271906 -> 0.151603 (difference 0.124413); dev_auroc_lower 0.9993 -> 0.99915 (difference -0.00015)
- arm_b_prime_fold2_seed43 (control): dev_accuracy_lower 400 -> 399 (difference -1); val_loss_higher 0.00423285 -> 0.00931295 (difference 0.00508009)
- arm_b_prime_fold3_seed42 (control): val_loss_higher 0.321121 -> 0.344856 (difference 0.0237347)
- arm_b_prime_fold3_seed43 (control): dev_accuracy_lower 400 -> 399 (difference -1); val_loss_higher 0.00113761 -> 0.0279358 (difference 0.0267982)
- arm_b_prime_fold3_seed44 (control): dev_accuracy_lower 399 -> 367 (difference -32); val_loss_higher 0.0340957 -> 0.12177 (difference 0.0876745)
- arm_b_prime_fold4_seed42 (control): val_loss_higher 0.137541 -> 0.361438 (difference 0.223897); dev_auroc_lower 0.99655 -> 0.948475 (difference -0.048075); fitting_loss_higher 0.0259913 -> 0.0603952 (difference 0.0344039)

## Limits

each cell and each case states what is observed with these statistics on these checkpoints and names no cause; a recovery shows the effect of this recalibration on the examined checkpoints, not a single cause of the saturation; a missing recovery does not, in general, exclude a role of BatchNormalization or dropout; only the restored checkpoint of each training is examined, and it was selected on the same dev; small dev (2-3 subjects per fold); one estimator; the criteria were defined after the patience-4 run

the report gives all 36 models, every worsening included, the fitting and dev values before and after, and the actual time against the estimate

## All models

| model | group | epoch | dev acc | dev val_loss | dev AUROC | fitting acc | fitting loss | dev crit | cell |
|---|---|---|---|---|---|---|---|---|---|
| arm_b_fold1_seed42 | ten | 1 | 0.5000 -> 0.9950 | 7.0611 -> 0.0244 | 0.9975 -> 0.9998 | 0.5000 -> 0.9993 | 7.3138 -> 0.0015 | 0 -> 1 | fitting_not_met_dev_met |
| arm_a_star_fold1_seed42 | control | 10 | 0.9950 -> 0.9975 | 0.0104 -> 0.0134 | 1.0000 -> 1.0000 | 0.9986 -> 1.0000 | 0.0232 -> 0.0001 | 1 -> 1 | fitting_met_dev_met |
| arm_b_fold1_seed43 | control | 11 | 0.9975 -> 0.9925 | 0.0056 -> 0.1830 | 1.0000 -> 0.9950 | 1.0000 -> 1.0000 | 0.0001 -> 0.0000 | 1 -> 1 | fitting_met_dev_met |
| arm_a_star_fold1_seed43 | control | 9 | 0.9975 -> 0.9950 | 0.0165 -> 0.0128 | 0.9995 -> 1.0000 | 0.9943 -> 1.0000 | 0.0421 -> 0.0005 | 1 -> 1 | fitting_met_dev_met |
| arm_b_fold1_seed44 | control | 6 | 0.9950 -> 0.9950 | 0.0298 -> 0.3445 | 0.9985 -> 0.9947 | 1.0000 -> 0.9993 | 0.0013 -> 0.0014 | 1 -> 1 | fitting_met_dev_met |
| arm_a_star_fold1_seed44 | control | 11 | 0.9875 -> 0.9875 | 0.0345 -> 0.5327 | 0.9991 -> 0.9895 | 1.0000 -> 0.9986 | 0.0007 -> 0.0028 | 1 -> 1 | fitting_met_dev_met |
| arm_b_fold2_seed42 | control | 9 | 0.9475 -> 1.0000 | 0.2308 -> 0.0087 | 0.9888 -> 1.0000 | 0.9857 -> 0.9971 | 0.2986 -> 0.0076 | 1 -> 1 | fitting_met_dev_met |
| arm_a_star_fold2_seed42 | control | 11 | 1.0000 -> 1.0000 | 0.0001 -> 0.0003 | 1.0000 -> 1.0000 | 1.0000 -> 1.0000 | 0.0000 -> 0.0003 | 1 -> 1 | fitting_met_dev_met |
| arm_b_fold2_seed43 | control | 7 | 1.0000 -> 1.0000 | 0.0011 -> 0.0004 | 1.0000 -> 1.0000 | 0.9950 -> 0.9993 | 0.2597 -> 0.0043 | 1 -> 1 | fitting_met_dev_met |
| arm_a_star_fold2_seed43 | ten | 1 | 0.5000 -> 1.0000 | 60.9868 -> 0.0031 | 0.5000 -> 1.0000 | 0.5000 -> 0.9886 | 60.2927 -> 0.0371 | 0 -> 1 | fitting_not_met_dev_met |
| arm_b_fold2_seed44 | control | 10 | 0.9975 -> 1.0000 | 0.0030 -> 0.0029 | 1.0000 -> 1.0000 | 0.9893 -> 0.9986 | 0.0313 -> 0.0031 | 1 -> 1 | fitting_met_dev_met |
| arm_a_star_fold2_seed44 | ten | 1 | 0.5000 -> 0.9800 | 18.0934 -> 0.3553 | 0.2114 -> 0.9814 | 0.5000 -> 0.9957 | 15.9745 -> 0.0492 | 0 -> 1 | fitting_not_met_dev_met |
| arm_b_fold3_seed42 | control | 12 | 1.0000 -> 1.0000 | 0.0168 -> 0.0075 | 1.0000 -> 1.0000 | 0.9993 -> 1.0000 | 0.0051 -> 0.0005 | 1 -> 1 | fitting_met_dev_met |
| arm_a_star_fold3_seed42 | ten | 1 | 0.5000 -> 1.0000 | 3.1296 -> 0.0137 | 0.4690 -> 1.0000 | 0.5000 -> 0.9986 | 2.8493 -> 0.0147 | 0 -> 1 | fitting_not_met_dev_met |
| arm_b_fold3_seed43 | control | 10 | 0.9925 -> 0.9675 | 0.0531 -> 0.0640 | 1.0000 -> 1.0000 | 1.0000 -> 1.0000 | 0.0002 -> 0.0001 | 1 -> 1 | fitting_met_dev_met |
| arm_a_star_fold3_seed43 | control | 12 | 0.9975 -> 0.9175 | 0.0088 -> 0.1670 | 1.0000 -> 1.0000 | 1.0000 -> 1.0000 | 0.0014 -> 0.0002 | 1 -> 1 | fitting_met_dev_met |
| arm_b_fold3_seed44 | apart | 3 | 0.2375 -> 0.9950 | 3.2731 -> 0.0417 | 0.0950 -> 0.9986 | 0.2343 -> 0.9979 | 2.4545 -> 0.0130 | 0 -> 1 | fitting_not_met_dev_met |
| arm_a_star_fold3_seed44 | ten | 1 | 0.5000 -> 0.8275 | 33.4090 -> 0.2594 | 0.5000 -> 1.0000 | 0.5000 -> 0.9971 | 31.6305 -> 0.0172 | 0 -> 1 | fitting_not_met_dev_met |
| arm_b_fold4_seed42 | control | 11 | 1.0000 -> 0.9975 | 0.0005 -> 0.1622 | 1.0000 -> 0.9950 | 1.0000 -> 1.0000 | 0.0000 -> 0.0000 | 1 -> 1 | fitting_met_dev_met |
| arm_a_star_fold4_seed42 | control | 10 | 0.7975 -> 0.7875 | 0.8106 -> 0.7175 | 0.9990 -> 0.9916 | 0.9986 -> 0.9971 | 0.0580 -> 0.0111 | 1 -> 1 | fitting_met_dev_met |
| arm_b_fold4_seed43 | control | 6 | 1.0000 -> 0.9975 | 0.0013 -> 0.0047 | 1.0000 -> 1.0000 | 0.9993 -> 1.0000 | 0.0071 -> 0.0001 | 1 -> 1 | fitting_met_dev_met |
| arm_a_star_fold4_seed43 | control | 9 | 0.9950 -> 0.9175 | 0.0272 -> 0.1516 | 0.9993 -> 0.9991 | 0.9993 -> 0.9993 | 0.0036 -> 0.0024 | 1 -> 1 | fitting_met_dev_met |
| arm_b_fold4_seed44 | ten | 2 | 0.5000 -> 1.0000 | 2.3605 -> 0.0013 | 0.9652 -> 1.0000 | 0.5000 -> 0.9971 | 2.2382 -> 0.0143 | 0 -> 1 | fitting_not_met_dev_met |
| arm_a_star_fold4_seed44 | ten | 1 | 0.5000 -> 0.8100 | 22.8274 -> 0.6411 | 0.5000 -> 0.9909 | 0.5000 -> 0.9986 | 22.9596 -> 0.0063 | 0 -> 1 | fitting_not_met_dev_met |
| arm_b_prime_fold1_seed42 | control | 2 | 0.7725 -> 0.9950 | 0.6101 -> 0.3244 | 0.9937 -> 0.9948 | 0.7976 -> 0.9979 | 0.5912 -> 0.0121 | 1 -> 1 | fitting_met_dev_met |
| arm_b_prime_fold1_seed43 | control | 4 | 0.9400 -> 0.9975 | 0.8647 -> 0.0096 | 0.9899 -> 1.0000 | 0.8126 -> 0.9989 | 9.4140 -> 0.0020 | 1 -> 1 | fitting_not_met_dev_met |
| arm_b_prime_fold1_seed44 | ten | 1 | 0.5000 -> 0.9950 | 12.1449 -> 0.0146 | 0.4975 -> 0.9998 | 0.5000 -> 0.9989 | 13.0656 -> 0.0037 | 0 -> 1 | fitting_not_met_dev_met |
| arm_b_prime_fold2_seed42 | control | 11 | 0.8775 -> 0.9875 | 0.8750 -> 0.0215 | 0.9830 -> 1.0000 | 0.7784 -> 0.9989 | 1.0785 -> 0.0015 | 1 -> 1 | fitting_not_met_dev_met |
| arm_b_prime_fold2_seed43 | control | 10 | 1.0000 -> 0.9975 | 0.0042 -> 0.0093 | 1.0000 -> 1.0000 | 1.0000 -> 1.0000 | 0.0052 -> 0.0001 | 1 -> 1 | fitting_met_dev_met |
| arm_b_prime_fold2_seed44 | ten | 2 | 0.5000 -> 0.9975 | 11.4259 -> 0.0065 | 0.5000 -> 1.0000 | 0.5000 -> 1.0000 | 11.6160 -> 0.0003 | 0 -> 1 | fitting_not_met_dev_met |
| arm_b_prime_fold3_seed42 | control | 1 | 0.8325 -> 0.8800 | 0.3211 -> 0.3449 | 1.0000 -> 1.0000 | 0.7216 -> 1.0000 | 0.4431 -> 0.0027 | 1 -> 1 | fitting_not_met_dev_met |
| arm_b_prime_fold3_seed43 | control | 9 | 1.0000 -> 0.9975 | 0.0011 -> 0.0279 | 1.0000 -> 1.0000 | 0.9925 -> 1.0000 | 0.1127 -> 0.0001 | 1 -> 1 | fitting_met_dev_met |
| arm_b_prime_fold3_seed44 | control | 9 | 0.9975 -> 0.9175 | 0.0341 -> 0.1218 | 1.0000 -> 1.0000 | 0.9968 -> 1.0000 | 0.0385 -> 0.0001 | 1 -> 1 | fitting_met_dev_met |
| arm_b_prime_fold4_seed42 | control | 2 | 0.9175 -> 0.9500 | 0.1375 -> 0.3614 | 0.9966 -> 0.9485 | 0.9968 -> 0.9968 | 0.0260 -> 0.0604 | 1 -> 1 | fitting_met_dev_met |
| arm_b_prime_fold4_seed43 | control | 3 | 1.0000 -> 1.0000 | 0.0142 -> 0.0037 | 1.0000 -> 1.0000 | 0.9722 -> 0.9989 | 0.0761 -> 0.0024 | 1 -> 1 | fitting_met_dev_met |
| arm_b_prime_fold4_seed44 | ten | 1 | 0.5000 -> 1.0000 | 17.2907 -> 0.0281 | 0.5000 -> 1.0000 | 0.5000 -> 0.9979 | 16.4776 -> 0.0075 | 0 -> 1 | fitting_not_met_dev_met |


## Time

Wall time 1749 s (29.1 min) against the note's estimate of about 32 minutes (upper estimate about 140); cap 240 minutes.

| phase | seconds |
|---|---|
| phase 0 inputs | 2.8 |
| phase 1 baseline | 342.0 |
| phase 2 recalibration and evaluation | 1400.9 |
| final checks and writes | 0.6 |

