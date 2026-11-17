# BatchNormalization statistics, dev (`20261114-BNRECAL-a10a082`) and test (`20261117-BNTEST-6245b10`), compact form

The 36 checkpoints of the patience-4 run; all columns, sources and checks in `bn_recalibration_table.md`.

| model | group | restored epoch | dev accuracy before → after | dev val_loss before → after | dev criterion before → after | worsened on the dev | test ACER 0.5 before → after | session 03 ACER 0.5 before → after | test AUROC before → after |
|---|---|---|---|---|---|---|---|---|---|
| B fold 1 seed 42 | ten | 1 | 0.5000 → 0.9950 | 7.0611 → 0.0244 | no → yes | — | 0.5000 → 0.1988 | 0.5000 → 0.2919 | 0.7389 → 0.9217 |
| B fold 1 seed 43 | control | 11 | 0.9975 → 0.9925 | 0.0056 → 0.1830 | yes → yes | dev acc, dev val_loss, dev AUROC | 0.3234 → 0.3538 | 0.4715 → 0.5009 | 0.8948 → 0.7598 |
| B fold 1 seed 44 | control | 6 | 0.9950 → 0.9950 | 0.0298 → 0.3445 | yes → yes | dev val_loss, dev AUROC, fitting acc, fitting loss | 0.3565 → 0.3135 | 0.4970 → 0.4633 | 0.6400 → 0.7759 |
| B fold 2 seed 42 | control | 9 | 0.9475 → 1.0000 | 0.2308 → 0.0087 | yes → yes | — | 0.2757 → 0.2103 | 0.3643 → 0.2686 | 0.8368 → 0.9406 |
| B fold 2 seed 43 | control | 7 | 1.0000 → 1.0000 | 0.0011 → 0.0004 | yes → yes | — | 0.3479 → 0.3706 | 0.4650 → 0.4916 | 0.6701 → 0.6939 |
| B fold 2 seed 44 | control | 10 | 0.9975 → 1.0000 | 0.0030 → 0.0029 | yes → yes | — | 0.1659 → 0.0789 | 0.2219 → 0.1015 | 0.9865 → 0.9706 |
| B fold 3 seed 42 | control | 12 | 1.0000 → 1.0000 | 0.0168 → 0.0075 | yes → yes | — | 0.2091 → 0.2456 | 0.3154 → 0.3617 | 0.8298 → 0.6852 |
| B fold 3 seed 43 | control | 10 | 0.9925 → 0.9675 | 0.0531 → 0.0640 | yes → yes | dev acc, dev val_loss | 0.2555 → 0.2584 | 0.3726 → 0.3762 | 0.7776 → 0.7653 |
| B fold 3 seed 44 | apart | 3 | 0.2375 → 0.9950 | 3.2731 → 0.0417 | no → yes | — | 0.7265 → 0.1891 | 0.6554 → 0.2968 | 0.2781 → 0.8301 |
| B fold 4 seed 42 | control | 11 | 1.0000 → 0.9975 | 0.0005 → 0.1622 | yes → yes | dev acc, dev val_loss, dev AUROC | 0.3936 → 0.4227 | 0.5710 → 0.6140 | 0.7502 → 0.6143 |
| B fold 4 seed 43 | control | 6 | 1.0000 → 0.9975 | 0.0013 → 0.0047 | yes → yes | dev acc, dev val_loss | 0.4276 → 0.4677 | 0.6316 → 0.6810 | 0.5968 → 0.6553 |
| B fold 4 seed 44 | ten | 2 | 0.5000 → 1.0000 | 2.3605 → 0.0013 | no → yes | — | 0.5000 → 0.5236 | 0.5000 → 0.7424 | 0.4672 → 0.6272 |
| A* fold 1 seed 42 | control | 10 | 0.9950 → 0.9975 | 0.0104 → 0.0134 | yes → yes | dev val_loss | 0.0839 → 0.1317 | 0.1243 → 0.1571 | 0.9828 → 0.9555 |
| A* fold 1 seed 43 | control | 9 | 0.9975 → 0.9950 | 0.0165 → 0.0128 | yes → yes | dev acc | 0.0513 → 0.1560 | 0.0281 → 0.1795 | 0.9921 → 0.9934 |
| A* fold 1 seed 44 | control | 11 | 0.9875 → 0.9875 | 0.0345 → 0.5327 | yes → yes | dev val_loss, dev AUROC, fitting acc, fitting loss | 0.1428 → 0.1731 | 0.1972 → 0.1986 | 0.8963 → 0.9165 |
| A* fold 2 seed 42 | control | 11 | 1.0000 → 1.0000 | 0.0001 → 0.0003 | yes → yes | dev val_loss, fitting loss | 0.0937 → 0.0964 | 0.1216 → 0.1259 | 0.9701 → 0.9607 |
| A* fold 2 seed 43 | ten | 1 | 0.5000 → 1.0000 | 60.9868 → 0.0031 | no → yes | — | 0.5000 → 0.0616 | 0.5000 → 0.0697 | 0.5000 → 0.9972 |
| A* fold 2 seed 44 | ten | 1 | 0.5000 → 0.9800 | 18.0934 → 0.3553 | no → yes | — | 0.5000 → 0.0542 | 0.5000 → 0.0677 | 0.2782 → 0.9981 |
| A* fold 3 seed 42 | ten | 1 | 0.5000 → 1.0000 | 3.1296 → 0.0137 | no → yes | — | 0.5000 → 0.1109 | 0.5000 → 0.1433 | 0.5255 → 0.9727 |
| A* fold 3 seed 43 | control | 12 | 0.9975 → 0.9175 | 0.0088 → 0.1670 | yes → yes | dev acc, dev val_loss | 0.1836 → 0.0720 | 0.2333 → 0.0930 | 0.9534 → 0.9727 |
| A* fold 3 seed 44 | ten | 1 | 0.5000 → 0.8275 | 33.4090 → 0.2594 | no → yes | — | 0.5000 → 0.1731 | 0.5000 → 0.2552 | 0.5000 → 0.9023 |
| A* fold 4 seed 42 | control | 10 | 0.7975 → 0.7875 | 0.8106 → 0.7175 | yes → yes | dev acc, dev AUROC, fitting acc | 0.1763 → 0.1576 | 0.1270 → 0.1013 | 0.9271 → 0.9416 |
| A* fold 4 seed 43 | control | 9 | 0.9950 → 0.9175 | 0.0272 → 0.1516 | yes → yes | dev acc, dev val_loss, dev AUROC | 0.1777 → 0.2148 | 0.2251 → 0.3152 | 0.9227 → 0.7956 |
| A* fold 4 seed 44 | ten | 1 | 0.5000 → 0.8100 | 22.8274 → 0.6411 | no → yes | — | 0.5000 → 0.1052 | 0.5000 → 0.0224 | 0.5000 → 0.9799 |
| B' fold 1 seed 42 | control | 2 | 0.7725 → 0.9950 | 0.6101 → 0.3244 | yes → yes | — | 0.4933 → 0.2127 | 0.7216 → 0.2768 | 0.5029 → 0.9156 |
| B' fold 1 seed 43 | control | 4 | 0.9400 → 0.9975 | 0.8647 → 0.0096 | yes → yes | — | 0.4131 → 0.3557 | 0.5057 → 0.5154 | 0.5857 → 0.6022 |
| B' fold 1 seed 44 | ten | 1 | 0.5000 → 0.9950 | 12.1449 → 0.0146 | no → yes | — | 0.5000 → 0.3271 | 0.5000 → 0.4751 | 0.5000 → 0.8025 |
| B' fold 2 seed 42 | control | 11 | 0.8775 → 0.9875 | 0.8750 → 0.0215 | yes → yes | — | 0.1241 → 0.1813 | 0.0994 → 0.2314 | 0.9281 → 0.9271 |
| B' fold 2 seed 43 | control | 10 | 1.0000 → 0.9975 | 0.0042 → 0.0093 | yes → yes | dev acc, dev val_loss | 0.3759 → 0.4343 | 0.4986 → 0.5721 | 0.6391 → 0.6623 |
| B' fold 2 seed 44 | ten | 2 | 0.5000 → 0.9975 | 11.4259 → 0.0065 | no → yes | — | 0.5000 → 0.1386 | 0.5000 → 0.1769 | 0.5000 → 0.9845 |
| B' fold 3 seed 42 | control | 1 | 0.8325 → 0.8800 | 0.3211 → 0.3449 | yes → yes | dev val_loss | 0.3314 → 0.4369 | 0.4593 → 0.5795 | 0.7411 → 0.7787 |
| B' fold 3 seed 43 | control | 9 | 1.0000 → 0.9975 | 0.0011 → 0.0279 | yes → yes | dev acc, dev val_loss | 0.2715 → 0.2860 | 0.3984 → 0.4160 | 0.6528 → 0.7142 |
| B' fold 3 seed 44 | control | 9 | 0.9975 → 0.9175 | 0.0341 → 0.1218 | yes → yes | dev acc, dev val_loss | 0.2031 → 0.2052 | 0.3228 → 0.2742 | 0.7675 → 0.9255 |
| B' fold 4 seed 42 | control | 2 | 0.9175 → 0.9500 | 0.1375 → 0.3614 | yes → yes | dev val_loss, dev AUROC, fitting loss | 0.1606 → 0.3692 | 0.2207 → 0.5701 | 0.8717 → 0.7445 |
| B' fold 4 seed 43 | control | 3 | 1.0000 → 1.0000 | 0.0142 → 0.0037 | yes → yes | — | 0.3959 → 0.4333 | 0.5756 → 0.6500 | 0.4826 → 0.7240 |
| B' fold 4 seed 44 | ten | 1 | 0.5000 → 1.0000 | 17.2907 → 0.0281 | no → yes | — | 0.5000 → 0.3201 | 0.5000 → 0.5096 | 0.5000 → 0.6134 |

Legend. Dev = run `20261114-BNRECAL-a10a082` (PR #27), test = run `20261117-BNTEST-6245b10` (PR #28), both on the 36
checkpoints of run `20261106-SESSABLRUN-d999458` (early-stopping patience 4); before = original statistics, after =
moving mean and variance of the five BatchNormalization layers re-estimated on the fitting set of the model's arm and
fold, every trained parameter unchanged; group: ten = the ten degenerate trainings, control = the 25 control checkpoints
(the other models meeting the recovery criterion at their checkpoint), apart = the 1 that do not; dev criterion =
recovery criterion (dev accuracy >= 0.75 and val_loss < 1.0); worsened on the dev = indicators set in PR #27 (any change
in the worsening direction, however small; — none); test = every frame of the held-out subjects of the fold; ACER at the
fixed threshold 0.5; AUROC threshold-free. The test reading is a post hoc diagnostic extension on a test already
analysed, not an independent validation.

**Dev (PR #27)**

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

| summary | scope | models | subjects (bona fide / attack) | ACER 0.5 | APCER 0.5 | BPCER 0.5 | AUROC | valid resamples (ACER / APCER / BPCER / AUROC) |
|---|---|---|---|---|---|---|---|---|
| group ten | test | 10 | 16 (15 / 15) | -0.2987 [-0.4033, -0.2147] below; 9/0/1 | -0.0406 [-0.0948, 0.0279] contains; 2/0/8 | -0.5568 [-0.7246, -0.4507] below; 8/0/2 | 0.3790 [0.2635, 0.4344] above; 0/0/10 | 926 / 950 / 940 / 926 |
| group ten | session 03 | 10 | 16 (9 / 15) | -0.2246 [-0.3759, -0.1425] below; 8/0/2 | -0.0033 [-0.0734, 0.0651] contains; 2/0/8 | -0.4459 [-0.6941, -0.3185] below; 8/0/2 | 0.3153 [0.2027, 0.4447] above; 1/0/9 | 650 / 950 / 661 / 650 |
| group control | test | 25 | 16 (15 / 15) | 0.0082 [-0.0220, 0.0316] contains; 7/0/18 | 0.0409 [0.0026, 0.0714] above; 8/1/16 | -0.0246 [-0.0600, 0.0130] contains; 13/1/11 | 0.0237 [0.0039, 0.0396] above; 10/0/15 | 926 / 950 / 940 / 926 |
| group control | session 03 | 25 | 16 (9 / 15) | 0.0138 [-0.0295, 0.0424] contains; 7/0/18 | 0.0490 [-0.0004, 0.0973] contains; 8/1/16 | -0.0214 [-0.0794, 0.0195] contains; 13/3/9 | 0.0228 [0.0018, 0.0448] above; 12/0/13 | 650 / 950 / 661 / 650 |
| group apart | test | 1 | 4 (3 / 4) | -0.5373 no interval, not_available; 1/0/0 | -0.6067 no interval, not_available; 1/0/0 | -0.4679 no interval, not_available; 1/0/0 | 0.5520 no interval, not_available; 0/0/1 | 0 / 0 / 0 / 0 |
| group apart | session 03 | 1 | 4 (2 / 4) | -0.3586 no interval, not_available; 1/0/0 | -0.5219 no interval, not_available; 1/0/0 | -0.1954 no interval, not_available; 1/0/0 | 0.2815 no interval, not_available; 0/0/1 | 0 / 0 / 0 / 0 |
| group all | test | 36 | 16 (15 / 15) | -0.0922 [-0.1399, -0.0533] below; 17/0/19 | 0.0003 [-0.0334, 0.0383] contains; 11/1/24 | -0.1848 [-0.2502, -0.1359] below; 22/1/13 | 0.1370 [0.0863, 0.1598] above; 10/0/26 | 926 / 950 / 940 / 926 |
| group all | session 03 | 36 | 16 (9 / 15) | -0.0627 [-0.1322, -0.0232] below; 16/0/20 | 0.0186 [-0.0219, 0.0661] contains; 11/1/24 | -0.1441 [-0.2525, -0.0928] below; 22/3/11 | 0.1113 [0.0645, 0.1586] above; 13/0/23 | 650 / 950 / 661 / 650 |
| arm B | test | 12 | 16 (15 / 15) | -0.0707 [-0.1333, -0.0041] below; 5/0/7 | 0.0541 [0.0087, 0.1012] above; 3/0/9 | -0.1956 [-0.2880, -0.1083] below; 8/1/3 | 0.0644 [-0.0282, 0.1224] contains; 5/0/7 | 926 / 950 / 940 / 926 |
| arm B | session 03 | 12 | 16 (9 / 15) | -0.0313 [-0.1071, 0.0292] contains; 5/0/7 | 0.0850 [0.0305, 0.1442] above; 3/0/9 | -0.1477 [-0.2818, -0.0457] below; 8/1/3 | 0.0449 [-0.0333, 0.1262] contains; 5/0/7 | 650 / 950 / 661 / 650 |
| arm A* | test | 12 | 16 (15 / 15) | -0.1586 [-0.1937, -0.1160] below; 7/0/5 | -0.1554 [-0.1919, -0.1112] below; 7/1/4 | -0.1618 [-0.2083, -0.1069] below; 4/0/8 | 0.2032 [0.1418, 0.2394] above; 3/0/9 | 926 / 950 / 940 / 926 |
| arm A* | session 03 | 12 | 16 (9 / 15) | -0.1523 [-0.1981, -0.1179] below; 7/0/5 | -0.1541 [-0.1903, -0.1117] below; 7/1/4 | -0.1506 [-0.2170, -0.1101] below; 5/1/6 | 0.1943 [0.1364, 0.2574] above; 4/0/8 | 650 / 950 / 661 / 650 |
| arm B' | test | 12 | 16 (15 / 15) | -0.0474 [-0.1223, 0.0072] contains; 5/0/7 | 0.1022 [0.0414, 0.1643] above; 1/0/11 | -0.1969 [-0.3181, -0.1204] below; 10/0/2 | 0.1436 [0.0811, 0.1895] above; 2/0/10 | 926 / 950 / 940 / 926 |
| arm B' | session 03 | 12 | 16 (9 / 15) | -0.0046 [-0.1064, 0.0427] contains; 4/0/8 | 0.1250 [0.0473, 0.2090] above; 1/0/11 | -0.1341 [-0.2746, -0.0995] below; 9/1/2 | 0.0945 [0.0063, 0.2002] above; 4/0/8 | 650 / 950 / 661 / 650 |

Legend. Paired summaries of the test control, copied from `summary.csv` (means, model counts, folds, subjects and
readings recomputed from its rows; subjects with each class, valid resamples and intervals copied): per metric, mean
over the models of the difference after - before, with the subject-bootstrap interval [2.5, 97.5] (1,000 resamples of
the held-out subjects pooled over the folds, seed 0, the same resamples before and after; no interval with fewer than 5
subjects) and the reading fixed before the run (below / contains / above zero, not_available without an interval), then
the models lower / equal / higher after (for AUROC, higher = AUROC higher); for APCER, BPCER and ACER a negative value
is a lower error rate. The summaries share models and subjects and are not independent.

**Test (PR #28)**

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

**Time**

- Dev control: wall 1,748.6 s (29.1 minutes); estimate 32 minutes (upper 140); cap 240 minutes.
- Test control: wall 3,027.4 s (50.5 minutes), process CPU time over all threads 18,438.4 s; estimate 42 minutes (upper
  112); cap 240 minutes of elapsed time.
- Parent run (patience 4): 36 trainings, 326 epochs, 9,437.7 s of training (wall 10,209.2 s); seconds per epoch B
  32.4537, A* 31.8532, B' 21.2375 (fitting + dev frames 1,800, 1,800, 1,334).
