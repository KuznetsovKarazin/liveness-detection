# Annex: BatchNormalization statistics, dev (`20261114-BNRECAL-a10a082`) and test (`20261117-BNTEST-6245b10`)

The 36 checkpoints of the patience-4 run, re-estimated on the fitting set. Copied from the public outputs of the two
runs and checked by `nuaa_synthesis.py`: identity and anchors (clean tree, status `PASS`, run id ending with the commit,
`status.json` -> `SHA256SUMS` -> every file read, 36 models complete within the cap, parent run = run
`20261106-SESSABLRUN-d999458` read here, by run id and by the SHA-256 of its public aggregate and `SHA256SUMS`); dev:
group of every checkpoint from the comparison of the patience runs, restored epoch and frame counts from the parent run,
dev accuracy before within 1e-06 and val_loss before within 0.001 of the values at the checkpoint (the baseline
tolerances of the run), dev AUROC before equal, criterion, 2x2 cell and every worsening indicator recomputed from the
values before and after, the readings and counts of the aggregate recomputed, the largest baseline differences within
the tolerances of the run; test: denominators equal to the parent run, every value before equal to the published
per-model value of the parent run (APCER, BPCER, ACER at 0.5 and AUROC, test = stratum all, session 03 = stratum 03),
APCER and BPCER consistent with whole counts over the denominators (rate x denominator an integer within 1e-6) and ACER
their mean, differences = after - before, indicators, groups and the dev worsening equal to the dev control, in every
paired summary the models, folds, subjects, means, lower / equal / higher counts and the reading from the interval
recomputed from the rows (the subjects with each class and the valid resamples copied), equal to the aggregate, primary
reading, baseline and re-estimation checks 36 of 36 with the largest baseline differences within the tolerances of the
run. The bootstrap intervals are copied (they need the per-subject results, which are restricted). Declared by the test
run: post hoc diagnostic extension on a test already analysed: the before metrics of these 36 checkpoints on this test
are already published (parent run 20261106-SESSABLRUN-d999458), and the re-estimation was chosen and its effect first
measured on the dev of the same folds (PR #27); this is not an independent validation. attack if score_attack >
threshold (tie = bona fide); label 0 bona fide, 1 attack; score = probability of attack.

## Sources

| key | file (in research/) | SHA-256 |
|---|---|---|
| bnrecal_aggregate | `nuaa-saturation-control/recalibration/aggregate.json` | 09318a4cd31f8be87fa5320a7d05321f461c3a81fed0314c0cf11a14995af186 |
| bnrecal_results | `nuaa-saturation-control/recalibration/results.csv` | e3ec4c329fcad806dfd41b6d57a49ef54fd0b28ddb11df777d7d0743a23704d6 |
| bnrecal_baseline | `nuaa-saturation-control/recalibration/baseline.csv` | 589beed4cf571e210e1b36e865d174ba6a8301dde2931af7c840e64959998155 |
| bnrecal_record | `nuaa-saturation-control/recalibration/run_record.json` | 7ca7a7fa03a78559d0b3d2b257ff3e89f8888bfd6b70426c46a07475b0f3b553 |
| bnrecal_sums | `nuaa-saturation-control/recalibration/SHA256SUMS` | 0b4a7089ddefc2b2b7f3cc0853677dd5b87411b250caaa9f704ba1bfa59b0452 |
| bnrecal_status | `nuaa-saturation-control/run/status.json` | 30721eaf1e9939f9c553a846b97d92ea57425f5c09d56b73998b2360fab23df5 |
| bntest_aggregate | `nuaa-bn-test/comparison/aggregate.json` | cf99038f715b483b2977e012b7675a94c536ad7857859d9f2501eb2cc1bd85c0 |
| bntest_results | `nuaa-bn-test/comparison/results.csv` | 9515512382c421dba1b3c9d21ff3bf497a36dfe4dde3798db1977de6fb4672a6 |
| bntest_summary | `nuaa-bn-test/comparison/summary.csv` | 6d34f924ce6cc1946dba8c6a62e74c85bae9a6402460ada4fdd2e63c1e0fb75d |
| bntest_baseline | `nuaa-bn-test/comparison/baseline.csv` | 9030a30ae65b5ea8f4dd7d4ac4939c57b484ea54c5d93639074bc496ce38519b |
| bntest_recalibration | `nuaa-bn-test/comparison/recalibration_check.csv` | ad58654bf3e6b7e9316593a73713ef396e6cc744fb4e5b2a5c0bc01e9060c1dc |
| bntest_record | `nuaa-bn-test/comparison/run_record.json` | 78efa7af489df4bd6e1deb6dc683aa3c4baf1dc13fc83e596fb15909996acc22 |
| bntest_sums | `nuaa-bn-test/comparison/SHA256SUMS` | ba8ec7b696889459944c4bdbc3d0e5a05409d3278a898a619c8171d0e174afc2 |
| bntest_status | `nuaa-bn-test/run/status.json` | 9c0691af72a6815de391165a1ee09e908e3cc105c0961154e86fc2c30b8521eb |

## Dev, per checkpoint, before → after (run `20261114-BNRECAL-a10a082`)

| model | group | restored epoch | dev accuracy | dev correct | dev val_loss | dev AUROC | fitting accuracy | fitting loss | dev criterion | fitting criterion | cell (fitting before x dev after) | worsened |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B fold 1 seed 42 | ten | 1 | 0.5000 → 0.9950 | 200 → 398 | 7.0611 → 0.0244 | 0.9975 → 0.9998 | 0.5000 → 0.9993 | 7.3138 → 0.0015 | no → yes | no → yes | fitting_not_met_dev_met | — |
| B fold 1 seed 43 | control | 11 | 0.9975 → 0.9925 | 399 → 397 | 0.0056 → 0.1830 | 1.0000 → 0.9950 | 1.0000 → 1.0000 | 0.0001 → 0.0000 | yes → yes | yes → yes | fitting_met_dev_met | dev acc, dev val_loss, dev AUROC |
| B fold 1 seed 44 | control | 6 | 0.9950 → 0.9950 | 398 → 398 | 0.0298 → 0.3445 | 0.9985 → 0.9947 | 1.0000 → 0.9993 | 0.0013 → 0.0014 | yes → yes | yes → yes | fitting_met_dev_met | dev val_loss, dev AUROC, fitting acc, fitting loss |
| B fold 2 seed 42 | control | 9 | 0.9475 → 1.0000 | 379 → 400 | 0.2308 → 0.0087 | 0.9888 → 1.0000 | 0.9857 → 0.9971 | 0.2986 → 0.0076 | yes → yes | yes → yes | fitting_met_dev_met | — |
| B fold 2 seed 43 | control | 7 | 1.0000 → 1.0000 | 400 → 400 | 0.0011 → 0.0004 | 1.0000 → 1.0000 | 0.9950 → 0.9993 | 0.2597 → 0.0043 | yes → yes | yes → yes | fitting_met_dev_met | — |
| B fold 2 seed 44 | control | 10 | 0.9975 → 1.0000 | 399 → 400 | 0.0030 → 0.0029 | 1.0000 → 1.0000 | 0.9893 → 0.9986 | 0.0313 → 0.0031 | yes → yes | yes → yes | fitting_met_dev_met | — |
| B fold 3 seed 42 | control | 12 | 1.0000 → 1.0000 | 400 → 400 | 0.0168 → 0.0075 | 1.0000 → 1.0000 | 0.9993 → 1.0000 | 0.0051 → 0.0005 | yes → yes | yes → yes | fitting_met_dev_met | — |
| B fold 3 seed 43 | control | 10 | 0.9925 → 0.9675 | 397 → 387 | 0.0531 → 0.0640 | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 0.0002 → 0.0001 | yes → yes | yes → yes | fitting_met_dev_met | dev acc, dev val_loss |
| B fold 3 seed 44 | apart | 3 | 0.2375 → 0.9950 | 95 → 398 | 3.2731 → 0.0417 | 0.0950 → 0.9986 | 0.2343 → 0.9979 | 2.4545 → 0.0130 | no → yes | no → yes | fitting_not_met_dev_met | — |
| B fold 4 seed 42 | control | 11 | 1.0000 → 0.9975 | 400 → 399 | 0.0005 → 0.1622 | 1.0000 → 0.9950 | 1.0000 → 1.0000 | 0.0000 → 0.0000 | yes → yes | yes → yes | fitting_met_dev_met | dev acc, dev val_loss, dev AUROC |
| B fold 4 seed 43 | control | 6 | 1.0000 → 0.9975 | 400 → 399 | 0.0013 → 0.0047 | 1.0000 → 1.0000 | 0.9993 → 1.0000 | 0.0071 → 0.0001 | yes → yes | yes → yes | fitting_met_dev_met | dev acc, dev val_loss |
| B fold 4 seed 44 | ten | 2 | 0.5000 → 1.0000 | 200 → 400 | 2.3605 → 0.0013 | 0.9652 → 1.0000 | 0.5000 → 0.9971 | 2.2382 → 0.0143 | no → yes | no → yes | fitting_not_met_dev_met | — |
| A* fold 1 seed 42 | control | 10 | 0.9950 → 0.9975 | 398 → 399 | 0.0104 → 0.0134 | 1.0000 → 1.0000 | 0.9986 → 1.0000 | 0.0232 → 0.0001 | yes → yes | yes → yes | fitting_met_dev_met | dev val_loss |
| A* fold 1 seed 43 | control | 9 | 0.9975 → 0.9950 | 399 → 398 | 0.0165 → 0.0128 | 0.9995 → 1.0000 | 0.9943 → 1.0000 | 0.0421 → 0.0005 | yes → yes | yes → yes | fitting_met_dev_met | dev acc |
| A* fold 1 seed 44 | control | 11 | 0.9875 → 0.9875 | 395 → 395 | 0.0345 → 0.5327 | 0.9991 → 0.9895 | 1.0000 → 0.9986 | 0.0007 → 0.0028 | yes → yes | yes → yes | fitting_met_dev_met | dev val_loss, dev AUROC, fitting acc, fitting loss |
| A* fold 2 seed 42 | control | 11 | 1.0000 → 1.0000 | 400 → 400 | 0.0001 → 0.0003 | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 0.0000 → 0.0003 | yes → yes | yes → yes | fitting_met_dev_met | dev val_loss, fitting loss |
| A* fold 2 seed 43 | ten | 1 | 0.5000 → 1.0000 | 200 → 400 | 60.9868 → 0.0031 | 0.5000 → 1.0000 | 0.5000 → 0.9886 | 60.2927 → 0.0371 | no → yes | no → yes | fitting_not_met_dev_met | — |
| A* fold 2 seed 44 | ten | 1 | 0.5000 → 0.9800 | 200 → 392 | 18.0934 → 0.3553 | 0.2114 → 0.9814 | 0.5000 → 0.9957 | 15.9745 → 0.0492 | no → yes | no → yes | fitting_not_met_dev_met | — |
| A* fold 3 seed 42 | ten | 1 | 0.5000 → 1.0000 | 200 → 400 | 3.1296 → 0.0137 | 0.4690 → 1.0000 | 0.5000 → 0.9986 | 2.8493 → 0.0147 | no → yes | no → yes | fitting_not_met_dev_met | — |
| A* fold 3 seed 43 | control | 12 | 0.9975 → 0.9175 | 399 → 367 | 0.0088 → 0.1670 | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 0.0014 → 0.0002 | yes → yes | yes → yes | fitting_met_dev_met | dev acc, dev val_loss |
| A* fold 3 seed 44 | ten | 1 | 0.5000 → 0.8275 | 200 → 331 | 33.4090 → 0.2594 | 0.5000 → 1.0000 | 0.5000 → 0.9971 | 31.6305 → 0.0172 | no → yes | no → yes | fitting_not_met_dev_met | — |
| A* fold 4 seed 42 | control | 10 | 0.7975 → 0.7875 | 319 → 315 | 0.8106 → 0.7175 | 0.9990 → 0.9916 | 0.9986 → 0.9971 | 0.0580 → 0.0111 | yes → yes | yes → yes | fitting_met_dev_met | dev acc, dev AUROC, fitting acc |
| A* fold 4 seed 43 | control | 9 | 0.9950 → 0.9175 | 398 → 367 | 0.0272 → 0.1516 | 0.9993 → 0.9991 | 0.9993 → 0.9993 | 0.0036 → 0.0024 | yes → yes | yes → yes | fitting_met_dev_met | dev acc, dev val_loss, dev AUROC |
| A* fold 4 seed 44 | ten | 1 | 0.5000 → 0.8100 | 200 → 324 | 22.8274 → 0.6411 | 0.5000 → 0.9909 | 0.5000 → 0.9986 | 22.9596 → 0.0063 | no → yes | no → yes | fitting_not_met_dev_met | — |
| B' fold 1 seed 42 | control | 2 | 0.7725 → 0.9950 | 309 → 398 | 0.6101 → 0.3244 | 0.9937 → 0.9948 | 0.7976 → 0.9979 | 0.5912 → 0.0121 | yes → yes | yes → yes | fitting_met_dev_met | — |
| B' fold 1 seed 43 | control | 4 | 0.9400 → 0.9975 | 376 → 399 | 0.8647 → 0.0096 | 0.9899 → 1.0000 | 0.8126 → 0.9989 | 9.4140 → 0.0020 | yes → yes | no → yes | fitting_not_met_dev_met | — |
| B' fold 1 seed 44 | ten | 1 | 0.5000 → 0.9950 | 200 → 398 | 12.1449 → 0.0146 | 0.4975 → 0.9998 | 0.5000 → 0.9989 | 13.0656 → 0.0037 | no → yes | no → yes | fitting_not_met_dev_met | — |
| B' fold 2 seed 42 | control | 11 | 0.8775 → 0.9875 | 351 → 395 | 0.8750 → 0.0215 | 0.9830 → 1.0000 | 0.7784 → 0.9989 | 1.0785 → 0.0015 | yes → yes | no → yes | fitting_not_met_dev_met | — |
| B' fold 2 seed 43 | control | 10 | 1.0000 → 0.9975 | 400 → 399 | 0.0042 → 0.0093 | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 0.0052 → 0.0001 | yes → yes | yes → yes | fitting_met_dev_met | dev acc, dev val_loss |
| B' fold 2 seed 44 | ten | 2 | 0.5000 → 0.9975 | 200 → 399 | 11.4259 → 0.0065 | 0.5000 → 1.0000 | 0.5000 → 1.0000 | 11.6160 → 0.0003 | no → yes | no → yes | fitting_not_met_dev_met | — |
| B' fold 3 seed 42 | control | 1 | 0.8325 → 0.8800 | 333 → 352 | 0.3211 → 0.3449 | 1.0000 → 1.0000 | 0.7216 → 1.0000 | 0.4431 → 0.0027 | yes → yes | no → yes | fitting_not_met_dev_met | dev val_loss |
| B' fold 3 seed 43 | control | 9 | 1.0000 → 0.9975 | 400 → 399 | 0.0011 → 0.0279 | 1.0000 → 1.0000 | 0.9925 → 1.0000 | 0.1127 → 0.0001 | yes → yes | yes → yes | fitting_met_dev_met | dev acc, dev val_loss |
| B' fold 3 seed 44 | control | 9 | 0.9975 → 0.9175 | 399 → 367 | 0.0341 → 0.1218 | 1.0000 → 1.0000 | 0.9968 → 1.0000 | 0.0385 → 0.0001 | yes → yes | yes → yes | fitting_met_dev_met | dev acc, dev val_loss |
| B' fold 4 seed 42 | control | 2 | 0.9175 → 0.9500 | 367 → 380 | 0.1375 → 0.3614 | 0.9966 → 0.9485 | 0.9968 → 0.9968 | 0.0260 → 0.0604 | yes → yes | yes → yes | fitting_met_dev_met | dev val_loss, dev AUROC, fitting loss |
| B' fold 4 seed 43 | control | 3 | 1.0000 → 1.0000 | 400 → 400 | 0.0142 → 0.0037 | 1.0000 → 1.0000 | 0.9722 → 0.9989 | 0.0761 → 0.0024 | yes → yes | yes → yes | fitting_met_dev_met | — |
| B' fold 4 seed 44 | ten | 1 | 0.5000 → 1.0000 | 200 → 400 | 17.2907 → 0.0281 | 0.5000 → 1.0000 | 0.5000 → 0.9979 | 16.4776 → 0.0075 | no → yes | no → yes | fitting_not_met_dev_met | — |

Legend. Dev = run `20261114-BNRECAL-a10a082` (PR #27), test = run `20261117-BNTEST-6245b10` (PR #28), both on the 36
checkpoints of run `20261106-SESSABLRUN-d999458` (early-stopping patience 4); before = original statistics, after =
moving mean and variance of the five BatchNormalization layers re-estimated on the fitting set of the model's arm and
fold, every trained parameter unchanged; group: ten = the ten degenerate trainings, control = the 25 control checkpoints
(the other models meeting the recovery criterion at their checkpoint), apart = the 1 that do not; dev criterion =
recovery criterion (dev accuracy >= 0.75 and val_loss < 1.0); worsened on the dev = indicators set in PR #27 (any change
in the worsening direction, however small; — none); test = every frame of the held-out subjects of the fold; ACER at the
fixed threshold 0.5; AUROC threshold-free. The test reading is a post hoc diagnostic extension on a test already
analysed, not an independent validation.

- Models: 36 (the ten degenerate trainings 10, control checkpoints 25, reported apart 1: B fold 3 seed 44); dev 400
  frames per model; fitting 1,400 frames (B, A*) or 934 (B').
- Baseline with the original statistics, before any re-estimation: 36 of 36 within every tolerance (largest
  attack-probability difference 0, changed decisions at 0.5 0, largest dev val_loss difference 2.876e-06, largest dev
  accuracy difference 0, largest softmax gap 3.329e-06).
- The ten meeting the recovery criterion on the dev: before 0 of 10, after 10 of 10 (primary reading of the run: all).
- The ten after the re-estimation: dev accuracy 0.8100-1.0000, dev val_loss 0.0013-0.6411, dev AUROC 0.9814-1.0000;
  fitting accuracy before 0.5000-0.5000, after 0.9886-1.0000.
- Control checkpoints meeting the criterion on the dev: before 25 of 25, after 25 of 25 (lost: 0).
- Control checkpoints with at least one worsened indicator: 17 of 25 (dev accuracy lower 11, dev val_loss higher 15, dev
  AUROC lower 7, fitting accuracy lower 3, fitting loss higher 4); with none: 8. The ten and the model apart with a
  worsened indicator: 0.
- Control checkpoints, correct dev frames after against before: higher 8, equal 6, lower 11; largest loss 32 of 400 (A*
  fold 3 seed 43, B' fold 3 seed 44); dev accuracy after 0.7875-1.0000.
- Model reported apart, B fold 3 seed 44: dev accuracy 0.2375 → 0.9950, dev val_loss 3.2731 → 0.0417, criterion no →
  yes.
- Fitting before x dev after (checkpoints counted, no cause named): both 22, dev after only 14 (the ten 10, apart 1,
  control checkpoints 3), fitting only 0, neither 0.

## Test, per checkpoint and scope, before → after (difference) (run `20261117-BNTEST-6245b10`)

| model | group | worsened on the dev | scope | n bona fide / attack | APCER 0.5 | BPCER 0.5 | ACER 0.5 | AUROC | indicators set |
|---|---|---|---|---|---|---|---|---|---|
| B fold 1 seed 42 | ten | no | test | 1423 / 1819 | 0.0000 → 0.0110 (0.0110) | 1.0000 → 0.3865 (-0.6135) | 0.5000 → 0.1988 (-0.3012) | 0.7389 → 0.9217 (0.1828) | APCER higher |
| B fold 1 seed 42 | ten | no | session 03 | 925 / 1232 | 0.0000 → 0.0162 (0.0162) | 1.0000 → 0.5676 (-0.4324) | 0.5000 → 0.2919 (-0.2081) | 0.6134 → 0.8269 (0.2135) | APCER higher |
| B fold 1 seed 43 | control | yes | test | 1423 / 1819 | 0.0038 → 0.0027 (-0.0011) | 0.6430 → 0.7048 (0.0618) | 0.3234 → 0.3538 (0.0304) | 0.8948 → 0.7598 (-0.1351) | ACER higher, BPCER higher, AUROC lower |
| B fold 1 seed 43 | control | yes | session 03 | 925 / 1232 | 0.0057 → 0.0041 (-0.0016) | 0.9373 → 0.9978 (0.0605) | 0.4715 → 0.5009 (0.0295) | 0.7812 → 0.5268 (-0.2544) | ACER higher, BPCER higher, AUROC lower |
| B fold 1 seed 44 | control | yes | test | 1423 / 1819 | 0.0033 → 0.0902 (0.0869) | 0.7098 → 0.5369 (-0.1729) | 0.3565 → 0.3135 (-0.0430) | 0.6400 → 0.7759 (0.1359) | APCER higher |
| B fold 1 seed 44 | control | yes | session 03 | 925 / 1232 | 0.0049 → 0.1331 (0.1282) | 0.9892 → 0.7935 (-0.1957) | 0.4970 → 0.4633 (-0.0337) | 0.3809 → 0.5595 (0.1786) | APCER higher |
| B fold 2 seed 42 | control | no | test | 1370 / 1744 | 0.1623 → 0.4008 (0.2385) | 0.3891 → 0.0197 (-0.3693) | 0.2757 → 0.2103 (-0.0654) | 0.8368 → 0.9406 (0.1038) | APCER higher |
| B fold 2 seed 42 | control | no | session 03 | 1019 / 1364 | 0.2075 → 0.5117 (0.3043) | 0.5211 → 0.0255 (-0.4956) | 0.3643 → 0.2686 (-0.0957) | 0.7238 → 0.8991 (0.1753) | APCER higher |
| B fold 2 seed 43 | control | no | test | 1370 / 1744 | 0.0688 → 0.2018 (0.1330) | 0.6270 → 0.5394 (-0.0876) | 0.3479 → 0.3706 (0.0227) | 0.6701 → 0.6939 (0.0238) | ACER higher, APCER higher |
| B fold 2 seed 43 | control | no | session 03 | 1019 / 1364 | 0.0880 → 0.2581 (0.1701) | 0.8420 → 0.7252 (-0.1168) | 0.4650 → 0.4916 (0.0267) | 0.4744 → 0.5270 (0.0526) | ACER higher, APCER higher |
| B fold 2 seed 44 | control | no | test | 1370 / 1744 | 0.0063 → 0.1227 (0.1164) | 0.3255 → 0.0350 (-0.2905) | 0.1659 → 0.0789 (-0.0871) | 0.9865 → 0.9706 (-0.0159) | APCER higher, AUROC lower |
| B fold 2 seed 44 | control | no | session 03 | 1019 / 1364 | 0.0081 → 0.1569 (0.1488) | 0.4357 → 0.0461 (-0.3896) | 0.2219 → 0.1015 (-0.1204) | 0.9779 → 0.9521 (-0.0258) | APCER higher, AUROC lower |
| B fold 3 seed 42 | control | no | test | 1387 / 2052 | 0.1038 → 0.1696 (0.0658) | 0.3143 → 0.3216 (0.0072) | 0.2091 → 0.2456 (0.0365) | 0.8298 → 0.6852 (-0.1445) | ACER higher, APCER higher, BPCER higher, AUROC lower |
| B fold 3 seed 42 | control | no | session 03 | 865 / 1665 | 0.1279 → 0.2078 (0.0799) | 0.5029 → 0.5156 (0.0127) | 0.3154 → 0.3617 (0.0463) | 0.6672 → 0.4946 (-0.1726) | ACER higher, APCER higher, BPCER higher, AUROC lower |
| B fold 3 seed 43 | control | yes | test | 1387 / 2052 | 0.2003 → 0.2061 (0.0058) | 0.3107 → 0.3107 (0.0000) | 0.2555 → 0.2584 (0.0029) | 0.7776 → 0.7653 (-0.0124) | ACER higher, APCER higher, AUROC lower |
| B fold 3 seed 43 | control | yes | session 03 | 865 / 1665 | 0.2468 → 0.2541 (0.0072) | 0.4983 → 0.4983 (0.0000) | 0.3726 → 0.3762 (0.0036) | 0.5781 → 0.5658 (-0.0123) | ACER higher, APCER higher, AUROC lower |
| B fold 3 seed 44 | apart | no | test | 1387 / 2052 | 0.6418 → 0.0351 (-0.6067) | 0.8111 → 0.3432 (-0.4679) | 0.7265 → 0.1891 (-0.5373) | 0.2781 → 0.8301 (0.5520) | — |
| B fold 3 seed 44 | apart | no | session 03 | 865 / 1665 | 0.5652 → 0.0432 (-0.5219) | 0.7457 → 0.5503 (-0.1954) | 0.6554 → 0.2968 (-0.3586) | 0.3959 → 0.6774 (0.2815) | — |
| B fold 4 seed 42 | control | yes | test | 925 / 1894 | 0.3590 → 0.3374 (-0.0216) | 0.4281 → 0.5081 (0.0800) | 0.3936 → 0.4227 (0.0292) | 0.7502 → 0.6143 (-0.1359) | ACER higher, BPCER higher, AUROC lower |
| B fold 4 seed 42 | control | yes | session 03 | 553 / 1500 | 0.4260 → 0.3780 (-0.0480) | 0.7161 → 0.8499 (0.1338) | 0.5710 → 0.6140 (0.0429) | 0.5357 → 0.3829 (-0.1527) | ACER higher, BPCER higher, AUROC lower |
| B fold 4 seed 43 | control | yes | test | 925 / 1894 | 0.3395 → 0.4240 (0.0845) | 0.5157 → 0.5114 (-0.0043) | 0.4276 → 0.4677 (0.0401) | 0.5968 → 0.6553 (0.0585) | ACER higher, APCER higher |
| B fold 4 seed 43 | control | yes | session 03 | 553 / 1500 | 0.4007 → 0.5067 (0.1060) | 0.8626 → 0.8553 (-0.0072) | 0.6316 → 0.6810 (0.0494) | 0.2230 → 0.3315 (0.1085) | ACER higher, APCER higher |
| B fold 4 seed 44 | ten | no | test | 925 / 1894 | 0.0000 → 0.5370 (0.5370) | 1.0000 → 0.5103 (-0.4897) | 0.5000 → 0.5236 (0.0236) | 0.4672 → 0.6272 (0.1600) | ACER higher, APCER higher |
| B fold 4 seed 44 | ten | no | session 03 | 553 / 1500 | 0.0000 → 0.6313 (0.6313) | 1.0000 → 0.8535 (-0.1465) | 0.5000 → 0.7424 (0.2424) | 0.1344 → 0.2809 (0.1465) | ACER higher, APCER higher |
| A* fold 1 seed 42 | control | yes | test | 1423 / 1819 | 0.0132 → 0.0610 (0.0478) | 0.1546 → 0.2024 (0.0478) | 0.0839 → 0.1317 (0.0478) | 0.9828 → 0.9555 (-0.0273) | ACER higher, APCER higher, BPCER higher, AUROC lower |
| A* fold 1 seed 42 | control | yes | session 03 | 925 / 1232 | 0.0195 → 0.0893 (0.0698) | 0.2292 → 0.2249 (-0.0043) | 0.1243 → 0.1571 (0.0327) | 0.9647 → 0.9337 (-0.0310) | ACER higher, APCER higher, AUROC lower |
| A* fold 1 seed 43 | control | yes | test | 1423 / 1819 | 0.0000 → 0.0000 (0.0000) | 0.1026 → 0.3120 (0.2094) | 0.0513 → 0.1560 (0.1047) | 0.9921 → 0.9934 (0.0012) | ACER higher, BPCER higher |
| A* fold 1 seed 43 | control | yes | session 03 | 925 / 1232 | 0.0000 → 0.0000 (0.0000) | 0.0562 → 0.3589 (0.3027) | 0.0281 → 0.1795 (0.1514) | 0.9909 → 0.9879 (-0.0030) | ACER higher, BPCER higher, AUROC lower |
| A* fold 1 seed 44 | control | yes | test | 1423 / 1819 | 0.0495 → 0.0011 (-0.0484) | 0.2361 → 0.3450 (0.1089) | 0.1428 → 0.1731 (0.0303) | 0.8963 → 0.9165 (0.0201) | ACER higher, BPCER higher |
| A* fold 1 seed 44 | control | yes | session 03 | 925 / 1232 | 0.0722 → 0.0016 (-0.0706) | 0.3222 → 0.3957 (0.0735) | 0.1972 → 0.1986 (0.0014) | 0.8401 → 0.8750 (0.0348) | ACER higher, BPCER higher |
| A* fold 2 seed 42 | control | yes | test | 1370 / 1744 | 0.1181 → 0.0986 (-0.0195) | 0.0693 → 0.0942 (0.0248) | 0.0937 → 0.0964 (0.0027) | 0.9701 → 0.9607 (-0.0094) | ACER higher, BPCER higher, AUROC lower |
| A* fold 2 seed 42 | control | yes | session 03 | 1019 / 1364 | 0.1510 → 0.1261 (-0.0249) | 0.0922 → 0.1256 (0.0334) | 0.1216 → 0.1259 (0.0042) | 0.9502 → 0.9364 (-0.0138) | ACER higher, BPCER higher, AUROC lower |
| A* fold 2 seed 43 | ten | no | test | 1370 / 1744 | 0.0000 → 0.1181 (0.1181) | 1.0000 → 0.0051 (-0.9949) | 0.5000 → 0.0616 (-0.4384) | 0.5000 → 0.9972 (0.4972) | APCER higher |
| A* fold 2 seed 43 | ten | no | session 03 | 1019 / 1364 | 0.0000 → 0.1334 (0.1334) | 1.0000 → 0.0059 (-0.9941) | 0.5000 → 0.0697 (-0.4303) | 0.5000 → 0.9961 (0.4961) | APCER higher |
| A* fold 2 seed 44 | ten | no | test | 1370 / 1744 | 1.0000 → 0.1055 (-0.8945) | 0.0000 → 0.0029 (0.0029) | 0.5000 → 0.0542 (-0.4458) | 0.2782 → 0.9981 (0.7199) | BPCER higher |
| A* fold 2 seed 44 | ten | no | session 03 | 1019 / 1364 | 1.0000 → 0.1334 (-0.8666) | 0.0000 → 0.0020 (0.0020) | 0.5000 → 0.0677 (-0.4323) | 0.2646 → 0.9994 (0.7348) | BPCER higher |
| A* fold 3 seed 42 | ten | no | test | 1387 / 2052 | 1.0000 → 0.1857 (-0.8143) | 0.0000 → 0.0360 (0.0360) | 0.5000 → 0.1109 (-0.3891) | 0.5255 → 0.9727 (0.4472) | BPCER higher |
| A* fold 3 seed 42 | ten | no | session 03 | 865 / 1665 | 1.0000 → 0.2288 (-0.7712) | 0.0000 → 0.0578 (0.0578) | 0.5000 → 0.1433 (-0.3567) | 0.5029 → 0.9471 (0.4441) | BPCER higher |
| A* fold 3 seed 43 | control | yes | test | 1387 / 2052 | 0.3289 → 0.1209 (-0.2081) | 0.0382 → 0.0231 (-0.0151) | 0.1836 → 0.0720 (-0.1116) | 0.9534 → 0.9727 (0.0194) | — |
| A* fold 3 seed 43 | control | yes | session 03 | 865 / 1665 | 0.4054 → 0.1489 (-0.2565) | 0.0613 → 0.0370 (-0.0243) | 0.2333 → 0.0930 (-0.1404) | 0.9113 → 0.9546 (0.0433) | — |
| A* fold 3 seed 44 | ten | no | test | 1387 / 2052 | 0.0000 → 0.1140 (0.1140) | 1.0000 → 0.2322 (-0.7678) | 0.5000 → 0.1731 (-0.3269) | 0.5000 → 0.9023 (0.4023) | APCER higher |
| A* fold 3 seed 44 | ten | no | session 03 | 865 / 1665 | 0.0000 → 0.1381 (0.1381) | 1.0000 → 0.3723 (-0.6277) | 0.5000 → 0.2552 (-0.2448) | 0.5000 → 0.8202 (0.3202) | APCER higher |
| A* fold 4 seed 42 | control | yes | test | 925 / 1894 | 0.2012 → 0.1605 (-0.0407) | 0.1514 → 0.1546 (0.0032) | 0.1763 → 0.1576 (-0.0187) | 0.9271 → 0.9416 (0.0145) | BPCER higher |
| A* fold 4 seed 42 | control | yes | session 03 | 553 / 1500 | 0.2540 → 0.2027 (-0.0513) | 0.0000 → 0.0000 (0.0000) | 0.1270 → 0.1013 (-0.0257) | 0.9714 → 0.9872 (0.0157) | — |
| A* fold 4 seed 43 | control | yes | test | 925 / 1894 | 0.3490 → 0.2091 (-0.1399) | 0.0065 → 0.2205 (0.2141) | 0.1777 → 0.2148 (0.0371) | 0.9227 → 0.7956 (-0.1271) | ACER higher, BPCER higher, AUROC lower |
| A* fold 4 seed 43 | control | yes | session 03 | 553 / 1500 | 0.4393 → 0.2633 (-0.1760) | 0.0108 → 0.3671 (0.3562) | 0.2251 → 0.3152 (0.0901) | 0.8788 → 0.6723 (-0.2065) | ACER higher, BPCER higher, AUROC lower |
| A* fold 4 seed 44 | ten | no | test | 925 / 1894 | 0.0000 → 0.0211 (0.0211) | 1.0000 → 0.1892 (-0.8108) | 0.5000 → 0.1052 (-0.3948) | 0.5000 → 0.9799 (0.4799) | APCER higher |
| A* fold 4 seed 44 | ten | no | session 03 | 553 / 1500 | 0.0000 → 0.0267 (0.0267) | 1.0000 → 0.0181 (-0.9819) | 0.5000 → 0.0224 (-0.4776) | 0.5000 → 0.9973 (0.4973) | APCER higher |
| B' fold 1 seed 42 | control | no | test | 1423 / 1819 | 0.6668 → 0.0220 (-0.6449) | 0.3197 → 0.4034 (0.0836) | 0.4933 → 0.2127 (-0.2806) | 0.5029 → 0.9156 (0.4127) | BPCER higher |
| B' fold 1 seed 42 | control | no | session 03 | 925 / 1232 | 0.9513 → 0.0325 (-0.9188) | 0.4919 → 0.5211 (0.0292) | 0.7216 → 0.2768 (-0.4448) | 0.1698 → 0.8392 (0.6694) | BPCER higher |
| B' fold 1 seed 43 | control | no | test | 1423 / 1819 | 0.0082 → 0.0220 (0.0137) | 0.8180 → 0.6894 (-0.1286) | 0.4131 → 0.3557 (-0.0574) | 0.5857 → 0.6022 (0.0165) | APCER higher |
| B' fold 1 seed 43 | control | no | session 03 | 925 / 1232 | 0.0114 → 0.0308 (0.0195) | 1.0000 → 1.0000 (0.0000) | 0.5057 → 0.5154 (0.0097) | 0.4761 → 0.2593 (-0.2168) | ACER higher, APCER higher, AUROC lower |
| B' fold 1 seed 44 | ten | no | test | 1423 / 1819 | 0.0000 → 0.1039 (0.1039) | 1.0000 → 0.5502 (-0.4498) | 0.5000 → 0.3271 (-0.1729) | 0.5000 → 0.8025 (0.3025) | APCER higher |
| B' fold 1 seed 44 | ten | no | session 03 | 925 / 1232 | 0.0000 → 0.1534 (0.1534) | 1.0000 → 0.7968 (-0.2032) | 0.5000 → 0.4751 (-0.0249) | 0.5000 → 0.5930 (0.0930) | APCER higher |
| B' fold 2 seed 42 | control | no | test | 1370 / 1744 | 0.1519 → 0.3612 (0.2093) | 0.0964 → 0.0015 (-0.0949) | 0.1241 → 0.1813 (0.0572) | 0.9281 → 0.9271 (-0.0010) | ACER higher, APCER higher, AUROC lower |
| B' fold 2 seed 42 | control | no | session 03 | 1019 / 1364 | 0.1928 → 0.4619 (0.2691) | 0.0059 → 0.0010 (-0.0049) | 0.0994 → 0.2314 (0.1321) | 0.9685 → 0.8756 (-0.0929) | ACER higher, APCER higher, AUROC lower |
| B' fold 2 seed 43 | control | yes | test | 1370 / 1744 | 0.2058 → 0.3584 (0.1525) | 0.5460 → 0.5102 (-0.0358) | 0.3759 → 0.4343 (0.0584) | 0.6391 → 0.6623 (0.0232) | ACER higher, APCER higher |
| B' fold 2 seed 43 | control | yes | session 03 | 1019 / 1364 | 0.2632 → 0.4582 (0.1950) | 0.7341 → 0.6860 (-0.0481) | 0.4986 → 0.5721 (0.0735) | 0.3946 → 0.4625 (0.0680) | ACER higher, APCER higher |
| B' fold 2 seed 44 | ten | no | test | 1370 / 1744 | 0.0000 → 0.2735 (0.2735) | 1.0000 → 0.0036 (-0.9964) | 0.5000 → 0.1386 (-0.3614) | 0.5000 → 0.9845 (0.4845) | APCER higher |
| B' fold 2 seed 44 | ten | no | session 03 | 1019 / 1364 | 0.0000 → 0.3490 (0.3490) | 1.0000 → 0.0049 (-0.9951) | 0.5000 → 0.1769 (-0.3231) | 0.5000 → 0.9747 (0.4747) | APCER higher |
| B' fold 3 seed 42 | control | yes | test | 1387 / 2052 | 0.0643 → 0.6316 (0.5673) | 0.5984 → 0.2422 (-0.3562) | 0.3314 → 0.4369 (0.1055) | 0.7411 → 0.7787 (0.0376) | ACER higher, APCER higher |
| B' fold 3 seed 42 | control | yes | session 03 | 865 / 1665 | 0.0793 → 0.7706 (0.6913) | 0.8393 → 0.3884 (-0.4509) | 0.4593 → 0.5795 (0.1202) | 0.5306 → 0.5749 (0.0444) | ACER higher, APCER higher |
| B' fold 3 seed 43 | control | yes | test | 1387 / 2052 | 0.1803 → 0.2295 (0.0492) | 0.3627 → 0.3425 (-0.0202) | 0.2715 → 0.2860 (0.0145) | 0.6528 → 0.7142 (0.0614) | ACER higher, APCER higher |
| B' fold 3 seed 43 | control | yes | session 03 | 865 / 1665 | 0.2222 → 0.2829 (0.0607) | 0.5746 → 0.5491 (-0.0254) | 0.3984 → 0.4160 (0.0176) | 0.4509 → 0.5011 (0.0502) | ACER higher, APCER higher |
| B' fold 3 seed 44 | control | yes | test | 1387 / 2052 | 0.0097 → 0.2958 (0.2861) | 0.3965 → 0.1146 (-0.2819) | 0.2031 → 0.2052 (0.0021) | 0.7675 → 0.9255 (0.1579) | ACER higher, APCER higher |
| B' fold 3 seed 44 | control | yes | session 03 | 865 / 1665 | 0.0120 → 0.3646 (0.3526) | 0.6335 → 0.1838 (-0.4497) | 0.3228 → 0.2742 (-0.0486) | 0.5907 → 0.8551 (0.2644) | APCER higher |
| B' fold 4 seed 42 | control | yes | test | 925 / 1894 | 0.2207 → 0.2249 (0.0042) | 0.1005 → 0.5135 (0.4130) | 0.1606 → 0.3692 (0.2086) | 0.8717 → 0.7445 (-0.1273) | ACER higher, APCER higher, BPCER higher, AUROC lower |
| B' fold 4 seed 42 | control | yes | session 03 | 553 / 1500 | 0.2787 → 0.2813 (0.0027) | 0.1627 → 0.8590 (0.6962) | 0.2207 → 0.5701 (0.3494) | 0.7858 → 0.5129 (-0.2728) | ACER higher, APCER higher, BPCER higher, AUROC lower |
| B' fold 4 seed 43 | control | no | test | 925 / 1894 | 0.2761 → 0.3627 (0.0866) | 0.5157 → 0.5038 (-0.0119) | 0.3959 → 0.4333 (0.0373) | 0.4826 → 0.7240 (0.2413) | ACER higher, APCER higher |
| B' fold 4 seed 43 | control | no | session 03 | 553 / 1500 | 0.2887 → 0.4573 (0.1687) | 0.8626 → 0.8427 (-0.0199) | 0.5756 → 0.6500 (0.0744) | 0.1470 → 0.4671 (0.3202) | ACER higher, APCER higher |
| B' fold 4 seed 44 | ten | no | test | 925 / 1894 | 0.0000 → 0.1246 (0.1246) | 1.0000 → 0.5157 (-0.4843) | 0.5000 → 0.3201 (-0.1799) | 0.5000 → 0.6134 (0.1134) | APCER higher |
| B' fold 4 seed 44 | ten | no | session 03 | 553 / 1500 | 0.0000 → 0.1567 (0.1567) | 1.0000 → 0.8626 (-0.1374) | 0.5000 → 0.5096 (0.0096) | 0.5000 → 0.2328 (-0.2672) | ACER higher, APCER higher, AUROC lower |

## Test, paired summaries

| summary | scope | metric | label | models | subjects (bona fide / attack) | mean before | mean after | mean difference [interval] reading | valid resamples | lower / equal / higher |
|---|---|---|---|---|---|---|---|---|---|---|
| group ten | test | ACER 0.5 | primary | 10 | 16 (15 / 15) | 0.5000 | 0.2013 | -0.2987 [-0.4033, -0.2147] below | 926 of 1000 | 9 / 0 / 1 |
| group ten | test | APCER 0.5 | secondary | 10 | 16 (15 / 15) | 0.2000 | 0.1594 | -0.0406 [-0.0948, 0.0279] contains | 950 of 1000 | 2 / 0 / 8 |
| group ten | test | BPCER 0.5 | secondary | 10 | 16 (15 / 15) | 0.8000 | 0.2432 | -0.5568 [-0.7246, -0.4507] below | 940 of 1000 | 8 / 0 / 2 |
| group ten | test | AUROC | secondary | 10 | 16 (15 / 15) | 0.5010 | 0.8799 | 0.3790 [0.2635, 0.4344] above | 926 of 1000 | 0 / 0 / 10 |
| group ten | session 03 | ACER 0.5 | secondary | 10 | 16 (9 / 15) | 0.5000 | 0.2754 | -0.2246 [-0.3759, -0.1425] below | 650 of 1000 | 8 / 0 / 2 |
| group ten | session 03 | APCER 0.5 | secondary | 10 | 16 (9 / 15) | 0.2000 | 0.1967 | -0.0033 [-0.0734, 0.0651] contains | 950 of 1000 | 2 / 0 / 8 |
| group ten | session 03 | BPCER 0.5 | secondary | 10 | 16 (9 / 15) | 0.8000 | 0.3541 | -0.4459 [-0.6941, -0.3185] below | 661 of 1000 | 8 / 0 / 2 |
| group ten | session 03 | AUROC | secondary | 10 | 16 (9 / 15) | 0.4515 | 0.7668 | 0.3153 [0.2027, 0.4447] above | 650 of 1000 | 1 / 0 / 9 |
| group control | test | ACER 0.5 | secondary | 25 | 16 (15 / 15) | 0.2573 | 0.2655 | 0.0082 [-0.0220, 0.0316] contains | 926 of 1000 | 7 / 0 / 18 |
| group control | test | APCER 0.5 | secondary | 25 | 16 (15 / 15) | 0.1636 | 0.2046 | 0.0409 [0.0026, 0.0714] above | 950 of 1000 | 8 / 1 / 16 |
| group control | test | BPCER 0.5 | secondary | 25 | 16 (15 / 15) | 0.3510 | 0.3264 | -0.0246 [-0.0600, 0.0130] contains | 940 of 1000 | 13 / 1 / 11 |
| group control | test | AUROC | secondary | 25 | 16 (15 / 15) | 0.7920 | 0.8156 | 0.0237 [0.0039, 0.0396] above | 926 of 1000 | 10 / 0 / 15 |
| group control | session 03 | ACER 0.5 | secondary | 25 | 16 (9 / 15) | 0.3508 | 0.3646 | 0.0138 [-0.0295, 0.0424] contains | 650 of 1000 | 7 / 0 / 18 |
| group control | session 03 | APCER 0.5 | secondary | 25 | 16 (9 / 15) | 0.2063 | 0.2553 | 0.0490 [-0.0004, 0.0973] contains | 950 of 1000 | 8 / 1 / 16 |
| group control | session 03 | BPCER 0.5 | secondary | 25 | 16 (9 / 15) | 0.4953 | 0.4739 | -0.0214 [-0.0794, 0.0195] contains | 661 of 1000 | 13 / 3 / 9 |
| group control | session 03 | AUROC | secondary | 25 | 16 (9 / 15) | 0.6545 | 0.6774 | 0.0228 [0.0018, 0.0448] above | 650 of 1000 | 12 / 0 / 13 |
| group apart | test | ACER 0.5 | secondary | 1 | 4 (3 / 4) | 0.7265 | 0.1891 | -0.5373 no interval, not_available | 0 of 1000 | 1 / 0 / 0 |
| group apart | test | APCER 0.5 | secondary | 1 | 4 (3 / 4) | 0.6418 | 0.0351 | -0.6067 no interval, not_available | 0 of 1000 | 1 / 0 / 0 |
| group apart | test | BPCER 0.5 | secondary | 1 | 4 (3 / 4) | 0.8111 | 0.3432 | -0.4679 no interval, not_available | 0 of 1000 | 1 / 0 / 0 |
| group apart | test | AUROC | secondary | 1 | 4 (3 / 4) | 0.2781 | 0.8301 | 0.5520 no interval, not_available | 0 of 1000 | 0 / 0 / 1 |
| group apart | session 03 | ACER 0.5 | secondary | 1 | 4 (2 / 4) | 0.6554 | 0.2968 | -0.3586 no interval, not_available | 0 of 1000 | 1 / 0 / 0 |
| group apart | session 03 | APCER 0.5 | secondary | 1 | 4 (2 / 4) | 0.5652 | 0.0432 | -0.5219 no interval, not_available | 0 of 1000 | 1 / 0 / 0 |
| group apart | session 03 | BPCER 0.5 | secondary | 1 | 4 (2 / 4) | 0.7457 | 0.5503 | -0.1954 no interval, not_available | 0 of 1000 | 1 / 0 / 0 |
| group apart | session 03 | AUROC | secondary | 1 | 4 (2 / 4) | 0.3959 | 0.6774 | 0.2815 no interval, not_available | 0 of 1000 | 0 / 0 / 1 |
| group all | test | ACER 0.5 | secondary | 36 | 16 (15 / 15) | 0.3378 | 0.2456 | -0.0922 [-0.1399, -0.0533] below | 926 of 1000 | 17 / 0 / 19 |
| group all | test | APCER 0.5 | secondary | 36 | 16 (15 / 15) | 0.1870 | 0.1873 | 0.0003 [-0.0334, 0.0383] contains | 950 of 1000 | 11 / 1 / 24 |
| group all | test | BPCER 0.5 | secondary | 36 | 16 (15 / 15) | 0.4885 | 0.3038 | -0.1848 [-0.2502, -0.1359] below | 940 of 1000 | 22 / 1 / 13 |
| group all | test | AUROC | secondary | 36 | 16 (15 / 15) | 0.6969 | 0.8339 | 0.1370 [0.0863, 0.1598] above | 926 of 1000 | 10 / 0 / 26 |
| group all | session 03 | ACER 0.5 | secondary | 36 | 16 (9 / 15) | 0.4007 | 0.3379 | -0.0627 [-0.1322, -0.0232] below | 650 of 1000 | 16 / 0 / 20 |
| group all | session 03 | APCER 0.5 | secondary | 36 | 16 (9 / 15) | 0.2145 | 0.2331 | 0.0186 [-0.0219, 0.0661] contains | 950 of 1000 | 11 / 1 / 24 |
| group all | session 03 | BPCER 0.5 | secondary | 36 | 16 (9 / 15) | 0.5869 | 0.4428 | -0.1441 [-0.2525, -0.0928] below | 661 of 1000 | 22 / 3 / 11 |
| group all | session 03 | AUROC | secondary | 36 | 16 (9 / 15) | 0.5910 | 0.7022 | 0.1113 [0.0645, 0.1586] above | 650 of 1000 | 13 / 0 / 23 |
| arm B | test | ACER 0.5 | secondary | 12 | 16 (15 / 15) | 0.3735 | 0.3028 | -0.0707 [-0.1333, -0.0041] below | 926 of 1000 | 5 / 0 / 7 |
| arm B | test | APCER 0.5 | secondary | 12 | 16 (15 / 15) | 0.1574 | 0.2115 | 0.0541 [0.0087, 0.1012] above | 950 of 1000 | 3 / 0 / 9 |
| arm B | test | BPCER 0.5 | secondary | 12 | 16 (15 / 15) | 0.5895 | 0.3940 | -0.1956 [-0.2880, -0.1083] below | 940 of 1000 | 8 / 1 / 3 |
| arm B | test | AUROC | secondary | 12 | 16 (15 / 15) | 0.7056 | 0.7700 | 0.0644 [-0.0282, 0.1224] contains | 926 of 1000 | 5 / 0 / 7 |
| arm B | session 03 | ACER 0.5 | secondary | 12 | 16 (9 / 15) | 0.4638 | 0.4325 | -0.0313 [-0.1071, 0.0292] contains | 650 of 1000 | 5 / 0 / 7 |
| arm B | session 03 | APCER 0.5 | secondary | 12 | 16 (9 / 15) | 0.1734 | 0.2584 | 0.0850 [0.0305, 0.1442] above | 950 of 1000 | 3 / 0 / 9 |
| arm B | session 03 | BPCER 0.5 | secondary | 12 | 16 (9 / 15) | 0.7542 | 0.6066 | -0.1477 [-0.2818, -0.0457] below | 661 of 1000 | 8 / 1 / 3 |
| arm B | session 03 | AUROC | secondary | 12 | 16 (9 / 15) | 0.5405 | 0.5854 | 0.0449 [-0.0333, 0.1262] contains | 650 of 1000 | 5 / 0 / 7 |
| arm A* | test | ACER 0.5 | secondary | 12 | 16 (15 / 15) | 0.2841 | 0.1255 | -0.1586 [-0.1937, -0.1160] below | 926 of 1000 | 7 / 0 / 5 |
| arm A* | test | APCER 0.5 | secondary | 12 | 16 (15 / 15) | 0.2550 | 0.0996 | -0.1554 [-0.1919, -0.1112] below | 950 of 1000 | 7 / 1 / 4 |
| arm A* | test | BPCER 0.5 | secondary | 12 | 16 (15 / 15) | 0.3132 | 0.1514 | -0.1618 [-0.2083, -0.1069] below | 940 of 1000 | 4 / 0 / 8 |
| arm A* | test | AUROC | secondary | 12 | 16 (15 / 15) | 0.7457 | 0.9489 | 0.2032 [0.1418, 0.2394] above | 926 of 1000 | 3 / 0 / 9 |
| arm A* | session 03 | ACER 0.5 | secondary | 12 | 16 (9 / 15) | 0.2964 | 0.1441 | -0.1523 [-0.1981, -0.1179] below | 650 of 1000 | 7 / 0 / 5 |
| arm A* | session 03 | APCER 0.5 | secondary | 12 | 16 (9 / 15) | 0.2785 | 0.1244 | -0.1541 [-0.1903, -0.1117] below | 950 of 1000 | 7 / 1 / 4 |
| arm A* | session 03 | BPCER 0.5 | secondary | 12 | 16 (9 / 15) | 0.3143 | 0.1638 | -0.1506 [-0.2170, -0.1101] below | 661 of 1000 | 5 / 1 / 6 |
| arm A* | session 03 | AUROC | secondary | 12 | 16 (9 / 15) | 0.7312 | 0.9256 | 0.1943 [0.1364, 0.2574] above | 650 of 1000 | 4 / 0 / 8 |
| arm B' | test | ACER 0.5 | secondary | 12 | 16 (15 / 15) | 0.3558 | 0.3084 | -0.0474 [-0.1223, 0.0072] contains | 926 of 1000 | 5 / 0 / 7 |
| arm B' | test | APCER 0.5 | secondary | 12 | 16 (15 / 15) | 0.1487 | 0.2508 | 0.1022 [0.0414, 0.1643] above | 950 of 1000 | 1 / 0 / 11 |
| arm B' | test | BPCER 0.5 | secondary | 12 | 16 (15 / 15) | 0.5628 | 0.3659 | -0.1969 [-0.3181, -0.1204] below | 940 of 1000 | 10 / 0 / 2 |
| arm B' | test | AUROC | secondary | 12 | 16 (15 / 15) | 0.6393 | 0.7829 | 0.1436 [0.0811, 0.1895] above | 926 of 1000 | 2 / 0 / 10 |
| arm B' | session 03 | ACER 0.5 | secondary | 12 | 16 (9 / 15) | 0.4418 | 0.4373 | -0.0046 [-0.1064, 0.0427] contains | 650 of 1000 | 4 / 0 / 8 |
| arm B' | session 03 | APCER 0.5 | secondary | 12 | 16 (9 / 15) | 0.1916 | 0.3166 | 0.1250 [0.0473, 0.2090] above | 950 of 1000 | 1 / 0 / 11 |
| arm B' | session 03 | BPCER 0.5 | secondary | 12 | 16 (9 / 15) | 0.6920 | 0.5579 | -0.1341 [-0.2746, -0.0995] below | 661 of 1000 | 9 / 1 / 2 |
| arm B' | session 03 | AUROC | secondary | 12 | 16 (9 / 15) | 0.5011 | 0.5957 | 0.0945 [0.0063, 0.2002] above | 650 of 1000 | 4 / 0 / 8 |

Legend. Paired summaries of the test control, copied from `summary.csv` (means, model counts, folds, subjects and
readings recomputed from its rows; subjects with each class, valid resamples and intervals copied): per metric, mean
over the models of the difference after - before, with the subject-bootstrap interval [2.5, 97.5] (1,000 resamples of
the held-out subjects pooled over the folds, seed 0, the same resamples before and after; no interval with fewer than 5
subjects) and the reading fixed before the run (below / contains / above zero, not_available without an interval), then
the models lower / equal / higher after (for AUROC, higher = AUROC higher); for APCER, BPCER and ACER a negative value
is a lower error rate. The summaries share models and subjects and are not independent.

- Baseline on the test with the original statistics: 36 of 36 within every tolerance (largest attack-probability
  difference 0, changed decisions at 0.5 0); the values before equal the published per-model values of the parent run in
  72 of 72 model and scope rows (checked here); re-estimated statistics identical to those of PR #27 in 36 of 36.
- Primary reading (fixed before the run): the ten, scope test, ACER at 0.5, mean difference after - before -0.2987
  [-0.4033, -0.2147] below; lower 9, equal 0, higher 1 (higher: B fold 4 seed 44 0.5000 → 0.5236).
- The ten, session 03, ACER at 0.5: mean difference -0.2246 [-0.3759, -0.1425] below; lower 8, equal 0, higher 2
  (higher: B fold 4 seed 44 0.5000 → 0.7424, B' fold 4 seed 44 0.5000 → 0.5096).
- Control checkpoints, models higher after: ACER at 0.5 18 of 25 on the test and 18 in session 03; APCER at 0.5 16 and
  16; BPCER at 0.5 11 and 9; AUROC lower 10 and 12.
- Control checkpoints with a worsened indicator on the dev in PR #27 (17): ACER at 0.5 higher on the test in 14 of them,
  in session 03 in 13.
- Indicators set, models (test / session 03): ACER higher 19 / 20, APCER higher 24 / 24, BPCER higher 13 / 11, AUROC
  lower 10 / 13.
- Subjects behind the intervals: 16 held-out subjects, 15 with bona fide and 15 with attack frames on the test, 9 and 15
  in session 03; valid resamples for ACER 926 of 1,000 on the test and 650 in session 03; the model apart (1 fold, 4
  subjects) has no interval.

## Time

- Dev control: wall 1,748.6 s (29.1 minutes); estimate 32 minutes (upper 140); cap 240 minutes.
- Test control: wall 3,027.4 s (50.5 minutes), process CPU time over all threads 18,438.4 s; estimate 42 minutes (upper
  112); cap 240 minutes of elapsed time.
- Parent run (patience 4): 36 trainings, 326 epochs, 9,437.7 s of training (wall 10,209.2 s); seconds per epoch B
  32.4537, A* 31.8532, B' 21.2375 (fitting + dev frames 1,800, 1,800, 1,334).
