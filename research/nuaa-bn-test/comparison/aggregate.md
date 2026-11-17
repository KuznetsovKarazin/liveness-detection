# Before/after on the NUAA test: BatchNormalization statistics re-estimated as in PR #27 (20261117-BNTEST-6245b10)

Status: **PASS**. Parent run `20261106-SESSABLRUN-d999458`; BNRECAL run `20261114-BNRECAL-a10a082`; configuration `1d72956997fd`; note `574c06c1735b`; estimator `scripts/nuaa_bn_recalibration.py` `8c228ef400f5`. No training.

## Declarations (fixed before the run)

- post hoc diagnostic extension on a test already analysed: the before metrics of these 36 checkpoints on this test are already published (parent run 20261106-SESSABLRUN-d999458), and the re-estimation was chosen and its effect first measured on the dev of the same folds (PR #27); this is not an independent validation
- the decision threshold is fixed at 0.5 (attack if score_attack > 0.5, tie = bona fide); no threshold is selected or re-selected, on the dev or on the test; AUROC is threshold-free
- the procedure, the tolerances, the reading rule and the outputs are fixed in this configuration and in the note before the run and are not changed after any test result is seen; every outcome is reported
- no statement on the cause of the dev saturation: every reading is limited to the effect of this re-estimation on these checkpoints on this test, at the fixed threshold 0.5 for APCER, BPCER and ACER (AUROC is threshold-free)
- all 36 models are included, among them the 17 with a dev or fitting indicator worsened in PR #27, marked in results.csv (column dev_worsened) and, in aggregate.md, in the table of all models and in the list of worsenings; every model and scope whose ACER, APCER or BPCER is higher, or whose AUROC is lower, after the re-estimation is listed with its size, however small

What is read of the test: this run reads the test: in phase 0 it opens and validates the fold schema (fold<k>_fitting.csv, fold<k>_dev.csv and fold<k>_test.csv, as the parent run did) and hashes every <model>_test_scores.csv of the parent run against its restricted SHA256SUMS without parsing it; in phase 1 it assembles the test frames of each fold (every frame of the held-out subjects, from the cache and the extracted frames, in key order, as the parent run did), scores them with each checkpoint and its original statistics with the evaluator of nuaa_bn_recalibration (the parent's own scorer is used only on the dev) and parses the parent's <model>_test_scores.csv to compare the per-frame attack probabilities; the evaluator also computes a test accuracy and AUROC in memory, which are discarded: no test metric is written or logged in phase 1; only if phases 1 and 2 pass for all 36 models, phase 3a reassembles the same test frames and scores them with the re-estimated statistics for all 36, the 36 .keras files, the parent's restricted SHA256SUMS and the two SHA256SUMS of the BNRECAL run are re-hashed, and only then phase 3b computes the test metrics (APCER, BPCER and ACER at the fixed threshold 0.5, AUROC) before and after and, as a reported check, from the parent's stored scores; the M2 manifests official_test and cnn_test are read to index the cache; the public aggregate of the parent run, which holds test metrics already published, is read only as bytes for its SHA-256 and never parsed; no public file of the folds run is opened.

## Primary reading (fixed before the run, applied by the script)

primary: the ten degenerate checkpoints, scope test (all test frames of the fold), ACER at the fixed threshold 0.5: the mean over the ten of the paired difference after - before with its subject-bootstrap interval, read with three pre-written statements (interval entirely below zero; containing zero, a bound exactly at zero counting as containing zero; entirely above zero), or a fourth, not_available, when the metric is not defined for a model of the summary or there is no interval (fewer than 5 subjects): no reading; plus the count of the ten whose ACER is lower, equal or higher after; applied mechanically by the script

**group ten (10 models), scope test, ACER at the fixed threshold 0.5: mean paired difference (after - before) -0.2987, subject-bootstrap interval [-0.4033, -0.2147] entirely below zero: on this test, the effect of this re-estimation on these checkpoints is a lower mean value; a post hoc diagnostic reading, not an independent validation, and no statement on the cause of the saturation**

Of the 10 degenerate checkpoints, ACER at the fixed threshold 0.5 on the whole test is lower after the re-estimation for 9, equal for 0 and higher for 1; every model is listed in the tables Higher after: arm_b_fold4_seed44.

## Paired summaries (mean over the models of after - before; subject bootstrap, held-out subjects pooled over the folds)

secondary, descriptive, no verdict beyond the same statements: session 03, the controls, the model reported apart (one fold, four held-out subjects: no interval, so not_available), all 36, the three arms, and APCER, BPCER and AUROC; for AUROC a value above zero is a higher AUROC; for APCER, BPCER and ACER a value below zero is a lower error rate

| summary | scope | metric | label | models | subjects | mean before | mean after | mean difference [interval] | reading | lower / equal / higher / NA |
|---|---|---|---|---|---|---|---|---|---|---|
| group ten | test | APCER at the fixed threshold 0.5 | secondary | 10 | 16 | 0.2000 | 0.1594 | -0.0406 [-0.0948, 0.0279] | contains | 2 / 0 / 8 / 0 |
| group ten | test | BPCER at the fixed threshold 0.5 | secondary | 10 | 16 | 0.8000 | 0.2432 | -0.5568 [-0.7246, -0.4507] | below | 8 / 0 / 2 / 0 |
| group ten | test | ACER at the fixed threshold 0.5 | primary | 10 | 16 | 0.5000 | 0.2013 | -0.2987 [-0.4033, -0.2147] | below | 9 / 0 / 1 / 0 |
| group ten | test | AUROC (threshold-free) | secondary | 10 | 16 | 0.5010 | 0.8799 | 0.3790 [0.2635, 0.4344] | above | 0 / 0 / 10 / 0 |
| group ten | session_03 | APCER at the fixed threshold 0.5 | secondary | 10 | 16 | 0.2000 | 0.1967 | -0.0033 [-0.0734, 0.0651] | contains | 2 / 0 / 8 / 0 |
| group ten | session_03 | BPCER at the fixed threshold 0.5 | secondary | 10 | 16 | 0.8000 | 0.3541 | -0.4459 [-0.6941, -0.3185] | below | 8 / 0 / 2 / 0 |
| group ten | session_03 | ACER at the fixed threshold 0.5 | secondary | 10 | 16 | 0.5000 | 0.2754 | -0.2246 [-0.3759, -0.1425] | below | 8 / 0 / 2 / 0 |
| group ten | session_03 | AUROC (threshold-free) | secondary | 10 | 16 | 0.4515 | 0.7668 | 0.3153 [0.2027, 0.4447] | above | 1 / 0 / 9 / 0 |
| group control | test | APCER at the fixed threshold 0.5 | secondary | 25 | 16 | 0.1636 | 0.2046 | 0.0409 [0.0026, 0.0714] | above | 8 / 1 / 16 / 0 |
| group control | test | BPCER at the fixed threshold 0.5 | secondary | 25 | 16 | 0.3510 | 0.3264 | -0.0246 [-0.0600, 0.0130] | contains | 13 / 1 / 11 / 0 |
| group control | test | ACER at the fixed threshold 0.5 | secondary | 25 | 16 | 0.2573 | 0.2655 | 0.0082 [-0.0220, 0.0316] | contains | 7 / 0 / 18 / 0 |
| group control | test | AUROC (threshold-free) | secondary | 25 | 16 | 0.7920 | 0.8156 | 0.0237 [0.0039, 0.0396] | above | 10 / 0 / 15 / 0 |
| group control | session_03 | APCER at the fixed threshold 0.5 | secondary | 25 | 16 | 0.2063 | 0.2553 | 0.0490 [-0.0004, 0.0973] | contains | 8 / 1 / 16 / 0 |
| group control | session_03 | BPCER at the fixed threshold 0.5 | secondary | 25 | 16 | 0.4953 | 0.4739 | -0.0214 [-0.0794, 0.0195] | contains | 13 / 3 / 9 / 0 |
| group control | session_03 | ACER at the fixed threshold 0.5 | secondary | 25 | 16 | 0.3508 | 0.3646 | 0.0138 [-0.0295, 0.0424] | contains | 7 / 0 / 18 / 0 |
| group control | session_03 | AUROC (threshold-free) | secondary | 25 | 16 | 0.6545 | 0.6774 | 0.0228 [0.0018, 0.0448] | above | 12 / 0 / 13 / 0 |
| group apart | test | APCER at the fixed threshold 0.5 | secondary | 1 | 4 | 0.6418 | 0.0351 | -0.6067 no interval | not_available | 1 / 0 / 0 / 0 |
| group apart | test | BPCER at the fixed threshold 0.5 | secondary | 1 | 4 | 0.8111 | 0.3432 | -0.4679 no interval | not_available | 1 / 0 / 0 / 0 |
| group apart | test | ACER at the fixed threshold 0.5 | secondary | 1 | 4 | 0.7265 | 0.1891 | -0.5373 no interval | not_available | 1 / 0 / 0 / 0 |
| group apart | test | AUROC (threshold-free) | secondary | 1 | 4 | 0.2781 | 0.8301 | 0.5520 no interval | not_available | 0 / 0 / 1 / 0 |
| group apart | session_03 | APCER at the fixed threshold 0.5 | secondary | 1 | 4 | 0.5652 | 0.0432 | -0.5219 no interval | not_available | 1 / 0 / 0 / 0 |
| group apart | session_03 | BPCER at the fixed threshold 0.5 | secondary | 1 | 4 | 0.7457 | 0.5503 | -0.1954 no interval | not_available | 1 / 0 / 0 / 0 |
| group apart | session_03 | ACER at the fixed threshold 0.5 | secondary | 1 | 4 | 0.6554 | 0.2968 | -0.3586 no interval | not_available | 1 / 0 / 0 / 0 |
| group apart | session_03 | AUROC (threshold-free) | secondary | 1 | 4 | 0.3959 | 0.6774 | 0.2815 no interval | not_available | 0 / 0 / 1 / 0 |
| group all | test | APCER at the fixed threshold 0.5 | secondary | 36 | 16 | 0.1870 | 0.1873 | 0.0003 [-0.0334, 0.0383] | contains | 11 / 1 / 24 / 0 |
| group all | test | BPCER at the fixed threshold 0.5 | secondary | 36 | 16 | 0.4885 | 0.3038 | -0.1848 [-0.2502, -0.1359] | below | 22 / 1 / 13 / 0 |
| group all | test | ACER at the fixed threshold 0.5 | secondary | 36 | 16 | 0.3378 | 0.2456 | -0.0922 [-0.1399, -0.0533] | below | 17 / 0 / 19 / 0 |
| group all | test | AUROC (threshold-free) | secondary | 36 | 16 | 0.6969 | 0.8339 | 0.1370 [0.0863, 0.1598] | above | 10 / 0 / 26 / 0 |
| group all | session_03 | APCER at the fixed threshold 0.5 | secondary | 36 | 16 | 0.2145 | 0.2331 | 0.0186 [-0.0219, 0.0661] | contains | 11 / 1 / 24 / 0 |
| group all | session_03 | BPCER at the fixed threshold 0.5 | secondary | 36 | 16 | 0.5869 | 0.4428 | -0.1441 [-0.2525, -0.0928] | below | 22 / 3 / 11 / 0 |
| group all | session_03 | ACER at the fixed threshold 0.5 | secondary | 36 | 16 | 0.4007 | 0.3379 | -0.0627 [-0.1322, -0.0232] | below | 16 / 0 / 20 / 0 |
| group all | session_03 | AUROC (threshold-free) | secondary | 36 | 16 | 0.5910 | 0.7022 | 0.1113 [0.0645, 0.1586] | above | 13 / 0 / 23 / 0 |
| arm arm_b | test | APCER at the fixed threshold 0.5 | secondary | 12 | 16 | 0.1574 | 0.2115 | 0.0541 [0.0087, 0.1012] | above | 3 / 0 / 9 / 0 |
| arm arm_b | test | BPCER at the fixed threshold 0.5 | secondary | 12 | 16 | 0.5895 | 0.3940 | -0.1956 [-0.2880, -0.1083] | below | 8 / 1 / 3 / 0 |
| arm arm_b | test | ACER at the fixed threshold 0.5 | secondary | 12 | 16 | 0.3735 | 0.3028 | -0.0707 [-0.1333, -0.0041] | below | 5 / 0 / 7 / 0 |
| arm arm_b | test | AUROC (threshold-free) | secondary | 12 | 16 | 0.7056 | 0.7700 | 0.0644 [-0.0282, 0.1224] | contains | 5 / 0 / 7 / 0 |
| arm arm_b | session_03 | APCER at the fixed threshold 0.5 | secondary | 12 | 16 | 0.1734 | 0.2584 | 0.0850 [0.0305, 0.1442] | above | 3 / 0 / 9 / 0 |
| arm arm_b | session_03 | BPCER at the fixed threshold 0.5 | secondary | 12 | 16 | 0.7542 | 0.6066 | -0.1477 [-0.2818, -0.0457] | below | 8 / 1 / 3 / 0 |
| arm arm_b | session_03 | ACER at the fixed threshold 0.5 | secondary | 12 | 16 | 0.4638 | 0.4325 | -0.0313 [-0.1071, 0.0292] | contains | 5 / 0 / 7 / 0 |
| arm arm_b | session_03 | AUROC (threshold-free) | secondary | 12 | 16 | 0.5405 | 0.5854 | 0.0449 [-0.0333, 0.1262] | contains | 5 / 0 / 7 / 0 |
| arm arm_a_star | test | APCER at the fixed threshold 0.5 | secondary | 12 | 16 | 0.2550 | 0.0996 | -0.1554 [-0.1919, -0.1112] | below | 7 / 1 / 4 / 0 |
| arm arm_a_star | test | BPCER at the fixed threshold 0.5 | secondary | 12 | 16 | 0.3132 | 0.1514 | -0.1618 [-0.2083, -0.1069] | below | 4 / 0 / 8 / 0 |
| arm arm_a_star | test | ACER at the fixed threshold 0.5 | secondary | 12 | 16 | 0.2841 | 0.1255 | -0.1586 [-0.1937, -0.1160] | below | 7 / 0 / 5 / 0 |
| arm arm_a_star | test | AUROC (threshold-free) | secondary | 12 | 16 | 0.7457 | 0.9489 | 0.2032 [0.1418, 0.2394] | above | 3 / 0 / 9 / 0 |
| arm arm_a_star | session_03 | APCER at the fixed threshold 0.5 | secondary | 12 | 16 | 0.2785 | 0.1244 | -0.1541 [-0.1903, -0.1117] | below | 7 / 1 / 4 / 0 |
| arm arm_a_star | session_03 | BPCER at the fixed threshold 0.5 | secondary | 12 | 16 | 0.3143 | 0.1638 | -0.1506 [-0.2170, -0.1101] | below | 5 / 1 / 6 / 0 |
| arm arm_a_star | session_03 | ACER at the fixed threshold 0.5 | secondary | 12 | 16 | 0.2964 | 0.1441 | -0.1523 [-0.1981, -0.1179] | below | 7 / 0 / 5 / 0 |
| arm arm_a_star | session_03 | AUROC (threshold-free) | secondary | 12 | 16 | 0.7312 | 0.9256 | 0.1943 [0.1364, 0.2574] | above | 4 / 0 / 8 / 0 |
| arm arm_b_prime | test | APCER at the fixed threshold 0.5 | secondary | 12 | 16 | 0.1487 | 0.2508 | 0.1022 [0.0414, 0.1643] | above | 1 / 0 / 11 / 0 |
| arm arm_b_prime | test | BPCER at the fixed threshold 0.5 | secondary | 12 | 16 | 0.5628 | 0.3659 | -0.1969 [-0.3181, -0.1204] | below | 10 / 0 / 2 / 0 |
| arm arm_b_prime | test | ACER at the fixed threshold 0.5 | secondary | 12 | 16 | 0.3558 | 0.3084 | -0.0474 [-0.1223, 0.0072] | contains | 5 / 0 / 7 / 0 |
| arm arm_b_prime | test | AUROC (threshold-free) | secondary | 12 | 16 | 0.6393 | 0.7829 | 0.1436 [0.0811, 0.1895] | above | 2 / 0 / 10 / 0 |
| arm arm_b_prime | session_03 | APCER at the fixed threshold 0.5 | secondary | 12 | 16 | 0.1916 | 0.3166 | 0.1250 [0.0473, 0.2090] | above | 1 / 0 / 11 / 0 |
| arm arm_b_prime | session_03 | BPCER at the fixed threshold 0.5 | secondary | 12 | 16 | 0.6920 | 0.5579 | -0.1341 [-0.2746, -0.0995] | below | 9 / 1 / 2 / 0 |
| arm arm_b_prime | session_03 | ACER at the fixed threshold 0.5 | secondary | 12 | 16 | 0.4418 | 0.4373 | -0.0046 [-0.1064, 0.0427] | contains | 4 / 0 / 8 / 0 |
| arm arm_b_prime | session_03 | AUROC (threshold-free) | secondary | 12 | 16 | 0.5011 | 0.5957 | 0.0945 [0.0063, 0.2002] | above | 4 / 0 / 8 / 0 |


## Worsening (every case listed)

per model and scope: acer_higher, apcer_higher, bpcer_higher (value after > value before) and auroc_lower (after < before); any change in that direction counts, however small; every case is listed in aggregate.md with before, after and difference

| flag (scope) | models |
|---|---|
| acer_higher (test) | 19 |
| apcer_higher (test) | 24 |
| bpcer_higher (test) | 13 |
| auroc_lower (test) | 10 |
| acer_higher (session_03) | 20 |
| apcer_higher (session_03) | 24 |
| bpcer_higher (session_03) | 11 |
| auroc_lower (session_03) | 13 |


- arm_b_fold1_seed42 (ten), test: apcer_higher 0.0000 -> 0.0110 (difference 0.0109951)
- arm_b_fold1_seed42 (ten), session_03: apcer_higher 0.0000 -> 0.0162 (difference 0.0162338)
- arm_a_star_fold1_seed42 (control, dev indicator worsened in PR #27), test: apcer_higher 0.0132 -> 0.0610 (difference 0.0478285)
- arm_a_star_fold1_seed42 (control, dev indicator worsened in PR #27), test: bpcer_higher 0.1546 -> 0.2024 (difference 0.0477864)
- arm_a_star_fold1_seed42 (control, dev indicator worsened in PR #27), test: acer_higher 0.0839 -> 0.1317 (difference 0.0478074)
- arm_a_star_fold1_seed42 (control, dev indicator worsened in PR #27), test: auroc_lower 0.9828 -> 0.9555 (difference -0.0272993)
- arm_a_star_fold1_seed42 (control, dev indicator worsened in PR #27), session_03: apcer_higher 0.0195 -> 0.0893 (difference 0.0698052)
- arm_a_star_fold1_seed42 (control, dev indicator worsened in PR #27), session_03: acer_higher 0.1243 -> 0.1571 (difference 0.0327404)
- arm_a_star_fold1_seed42 (control, dev indicator worsened in PR #27), session_03: auroc_lower 0.9647 -> 0.9337 (difference -0.0309916)
- arm_b_fold1_seed43 (control, dev indicator worsened in PR #27), test: bpcer_higher 0.6430 -> 0.7048 (difference 0.0618412)
- arm_b_fold1_seed43 (control, dev indicator worsened in PR #27), test: acer_higher 0.3234 -> 0.3538 (difference 0.0303708)
- arm_b_fold1_seed43 (control, dev indicator worsened in PR #27), test: auroc_lower 0.8948 -> 0.7598 (difference -0.135054)
- arm_b_fold1_seed43 (control, dev indicator worsened in PR #27), session_03: bpcer_higher 0.9373 -> 0.9978 (difference 0.0605405)
- arm_b_fold1_seed43 (control, dev indicator worsened in PR #27), session_03: acer_higher 0.4715 -> 0.5009 (difference 0.0294586)
- arm_b_fold1_seed43 (control, dev indicator worsened in PR #27), session_03: auroc_lower 0.7812 -> 0.5268 (difference -0.254364)
- arm_a_star_fold1_seed43 (control, dev indicator worsened in PR #27), test: bpcer_higher 0.1026 -> 0.3120 (difference 0.209417)
- arm_a_star_fold1_seed43 (control, dev indicator worsened in PR #27), test: acer_higher 0.0513 -> 0.1560 (difference 0.104708)
- arm_a_star_fold1_seed43 (control, dev indicator worsened in PR #27), session_03: bpcer_higher 0.0562 -> 0.3589 (difference 0.302703)
- arm_a_star_fold1_seed43 (control, dev indicator worsened in PR #27), session_03: acer_higher 0.0281 -> 0.1795 (difference 0.151351)
- arm_a_star_fold1_seed43 (control, dev indicator worsened in PR #27), session_03: auroc_lower 0.9909 -> 0.9879 (difference -0.00302387)
- arm_b_fold1_seed44 (control, dev indicator worsened in PR #27), test: apcer_higher 0.0033 -> 0.0902 (difference 0.0868609)
- arm_b_fold1_seed44 (control, dev indicator worsened in PR #27), session_03: apcer_higher 0.0049 -> 0.1331 (difference 0.128247)
- arm_a_star_fold1_seed44 (control, dev indicator worsened in PR #27), test: bpcer_higher 0.2361 -> 0.3450 (difference 0.108925)
- arm_a_star_fold1_seed44 (control, dev indicator worsened in PR #27), test: acer_higher 0.1428 -> 0.1731 (difference 0.0302733)
- arm_a_star_fold1_seed44 (control, dev indicator worsened in PR #27), session_03: bpcer_higher 0.3222 -> 0.3957 (difference 0.0735135)
- arm_a_star_fold1_seed44 (control, dev indicator worsened in PR #27), session_03: acer_higher 0.1972 -> 0.1986 (difference 0.00144832)
- arm_b_fold2_seed42 (control), test: apcer_higher 0.1623 -> 0.4008 (difference 0.238532)
- arm_b_fold2_seed42 (control), session_03: apcer_higher 0.2075 -> 0.5117 (difference 0.304252)
- arm_a_star_fold2_seed42 (control, dev indicator worsened in PR #27), test: bpcer_higher 0.0693 -> 0.0942 (difference 0.0248175)
- arm_a_star_fold2_seed42 (control, dev indicator worsened in PR #27), test: acer_higher 0.0937 -> 0.0964 (difference 0.00266105)
- arm_a_star_fold2_seed42 (control, dev indicator worsened in PR #27), test: auroc_lower 0.9701 -> 0.9607 (difference -0.00938902)
- arm_a_star_fold2_seed42 (control, dev indicator worsened in PR #27), session_03: bpcer_higher 0.0922 -> 0.1256 (difference 0.033366)
- arm_a_star_fold2_seed42 (control, dev indicator worsened in PR #27), session_03: acer_higher 0.1216 -> 0.1259 (difference 0.00421968)
- arm_a_star_fold2_seed42 (control, dev indicator worsened in PR #27), session_03: auroc_lower 0.9502 -> 0.9364 (difference -0.0138073)
- arm_b_fold2_seed43 (control), test: apcer_higher 0.0688 -> 0.2018 (difference 0.133028)
- arm_b_fold2_seed43 (control), test: acer_higher 0.3479 -> 0.3706 (difference 0.0227181)
- arm_b_fold2_seed43 (control), session_03: apcer_higher 0.0880 -> 0.2581 (difference 0.170088)
- arm_b_fold2_seed43 (control), session_03: acer_higher 0.4650 -> 0.4916 (difference 0.0266534)
- arm_a_star_fold2_seed43 (ten), test: apcer_higher 0.0000 -> 0.1181 (difference 0.118119)
- arm_a_star_fold2_seed43 (ten), session_03: apcer_higher 0.0000 -> 0.1334 (difference 0.133431)
- arm_b_fold2_seed44 (control), test: apcer_higher 0.0063 -> 0.1227 (difference 0.116399)
- arm_b_fold2_seed44 (control), test: auroc_lower 0.9865 -> 0.9706 (difference -0.015891)
- arm_b_fold2_seed44 (control), session_03: apcer_higher 0.0081 -> 0.1569 (difference 0.148827)
- arm_b_fold2_seed44 (control), session_03: auroc_lower 0.9779 -> 0.9521 (difference -0.0258077)
- arm_a_star_fold2_seed44 (ten), test: bpcer_higher 0.0000 -> 0.0029 (difference 0.00291971)
- arm_a_star_fold2_seed44 (ten), session_03: bpcer_higher 0.0000 -> 0.0020 (difference 0.00196271)
- arm_b_fold3_seed42 (control), test: apcer_higher 0.1038 -> 0.1696 (difference 0.0657895)
- arm_b_fold3_seed42 (control), test: bpcer_higher 0.3143 -> 0.3216 (difference 0.00720981)
- arm_b_fold3_seed42 (control), test: acer_higher 0.2091 -> 0.2456 (difference 0.0364996)
- arm_b_fold3_seed42 (control), test: auroc_lower 0.8298 -> 0.6852 (difference -0.144534)
- arm_b_fold3_seed42 (control), session_03: apcer_higher 0.1279 -> 0.2078 (difference 0.0798799)
- arm_b_fold3_seed42 (control), session_03: bpcer_higher 0.5029 -> 0.5156 (difference 0.0127168)
- arm_b_fold3_seed42 (control), session_03: acer_higher 0.3154 -> 0.3617 (difference 0.0462983)
- arm_b_fold3_seed42 (control), session_03: auroc_lower 0.6672 -> 0.4946 (difference -0.172601)
- arm_a_star_fold3_seed42 (ten), test: bpcer_higher 0.0000 -> 0.0360 (difference 0.036049)
- arm_a_star_fold3_seed42 (ten), session_03: bpcer_higher 0.0000 -> 0.0578 (difference 0.0578035)
- arm_b_fold3_seed43 (control, dev indicator worsened in PR #27), test: apcer_higher 0.2003 -> 0.2061 (difference 0.00584795)
- arm_b_fold3_seed43 (control, dev indicator worsened in PR #27), test: acer_higher 0.2555 -> 0.2584 (difference 0.00292398)
- arm_b_fold3_seed43 (control, dev indicator worsened in PR #27), test: auroc_lower 0.7776 -> 0.7653 (difference -0.0123582)
- arm_b_fold3_seed43 (control, dev indicator worsened in PR #27), session_03: apcer_higher 0.2468 -> 0.2541 (difference 0.00720721)
- arm_b_fold3_seed43 (control, dev indicator worsened in PR #27), session_03: acer_higher 0.3726 -> 0.3762 (difference 0.0036036)
- arm_b_fold3_seed43 (control, dev indicator worsened in PR #27), session_03: auroc_lower 0.5781 -> 0.5658 (difference -0.0123106)
- arm_a_star_fold3_seed44 (ten), test: apcer_higher 0.0000 -> 0.1140 (difference 0.114035)
- arm_a_star_fold3_seed44 (ten), session_03: apcer_higher 0.0000 -> 0.1381 (difference 0.138138)
- arm_b_fold4_seed42 (control, dev indicator worsened in PR #27), test: bpcer_higher 0.4281 -> 0.5081 (difference 0.08)
- arm_b_fold4_seed42 (control, dev indicator worsened in PR #27), test: acer_higher 0.3936 -> 0.4227 (difference 0.0291763)
- arm_b_fold4_seed42 (control, dev indicator worsened in PR #27), test: auroc_lower 0.7502 -> 0.6143 (difference -0.135896)
- arm_b_fold4_seed42 (control, dev indicator worsened in PR #27), session_03: bpcer_higher 0.7161 -> 0.8499 (difference 0.133816)
- arm_b_fold4_seed42 (control, dev indicator worsened in PR #27), session_03: acer_higher 0.5710 -> 0.6140 (difference 0.0429078)
- arm_b_fold4_seed42 (control, dev indicator worsened in PR #27), session_03: auroc_lower 0.5357 -> 0.3829 (difference -0.15274)
- arm_a_star_fold4_seed42 (control, dev indicator worsened in PR #27), test: bpcer_higher 0.1514 -> 0.1546 (difference 0.00324324)
- arm_b_fold4_seed43 (control, dev indicator worsened in PR #27), test: apcer_higher 0.3395 -> 0.4240 (difference 0.0844773)
- arm_b_fold4_seed43 (control, dev indicator worsened in PR #27), test: acer_higher 0.4276 -> 0.4677 (difference 0.0400765)
- arm_b_fold4_seed43 (control, dev indicator worsened in PR #27), session_03: apcer_higher 0.4007 -> 0.5067 (difference 0.106)
- arm_b_fold4_seed43 (control, dev indicator worsened in PR #27), session_03: acer_higher 0.6316 -> 0.6810 (difference 0.0493834)
- arm_a_star_fold4_seed43 (control, dev indicator worsened in PR #27), test: bpcer_higher 0.0065 -> 0.2205 (difference 0.214054)
- arm_a_star_fold4_seed43 (control, dev indicator worsened in PR #27), test: acer_higher 0.1777 -> 0.2148 (difference 0.0370693)
- arm_a_star_fold4_seed43 (control, dev indicator worsened in PR #27), test: auroc_lower 0.9227 -> 0.7956 (difference -0.127078)
- arm_a_star_fold4_seed43 (control, dev indicator worsened in PR #27), session_03: bpcer_higher 0.0108 -> 0.3671 (difference 0.356239)
- arm_a_star_fold4_seed43 (control, dev indicator worsened in PR #27), session_03: acer_higher 0.2251 -> 0.3152 (difference 0.0901193)
- arm_a_star_fold4_seed43 (control, dev indicator worsened in PR #27), session_03: auroc_lower 0.8788 -> 0.6723 (difference -0.206542)
- arm_b_fold4_seed44 (ten), test: apcer_higher 0.0000 -> 0.5370 (difference 0.536959)
- arm_b_fold4_seed44 (ten), test: acer_higher 0.5000 -> 0.5236 (difference 0.0236145)
- arm_b_fold4_seed44 (ten), session_03: apcer_higher 0.0000 -> 0.6313 (difference 0.631333)
- arm_b_fold4_seed44 (ten), session_03: acer_higher 0.5000 -> 0.7424 (difference 0.24243)
- arm_a_star_fold4_seed44 (ten), test: apcer_higher 0.0000 -> 0.0211 (difference 0.0211193)
- arm_a_star_fold4_seed44 (ten), session_03: apcer_higher 0.0000 -> 0.0267 (difference 0.0266667)
- arm_b_prime_fold1_seed42 (control), test: bpcer_higher 0.3197 -> 0.4034 (difference 0.0836261)
- arm_b_prime_fold1_seed42 (control), session_03: bpcer_higher 0.4919 -> 0.5211 (difference 0.0291892)
- arm_b_prime_fold1_seed43 (control), test: apcer_higher 0.0082 -> 0.0220 (difference 0.0137438)
- arm_b_prime_fold1_seed43 (control), session_03: apcer_higher 0.0114 -> 0.0308 (difference 0.0194805)
- arm_b_prime_fold1_seed43 (control), session_03: acer_higher 0.5057 -> 0.5154 (difference 0.00974026)
- arm_b_prime_fold1_seed43 (control), session_03: auroc_lower 0.4761 -> 0.2593 (difference -0.216796)
- arm_b_prime_fold1_seed44 (ten), test: apcer_higher 0.0000 -> 0.1039 (difference 0.103903)
- arm_b_prime_fold1_seed44 (ten), session_03: apcer_higher 0.0000 -> 0.1534 (difference 0.153409)
- arm_b_prime_fold2_seed42 (control), test: apcer_higher 0.1519 -> 0.3612 (difference 0.209289)
- arm_b_prime_fold2_seed42 (control), test: acer_higher 0.1241 -> 0.1813 (difference 0.0571992)
- arm_b_prime_fold2_seed42 (control), test: auroc_lower 0.9281 -> 0.9271 (difference -0.001007)
- arm_b_prime_fold2_seed42 (control), session_03: apcer_higher 0.1928 -> 0.4619 (difference 0.269062)
- arm_b_prime_fold2_seed42 (control), session_03: acer_higher 0.0994 -> 0.2314 (difference 0.132077)
- arm_b_prime_fold2_seed42 (control), session_03: auroc_lower 0.9685 -> 0.8756 (difference -0.0929092)
- arm_b_prime_fold2_seed43 (control, dev indicator worsened in PR #27), test: apcer_higher 0.2058 -> 0.3584 (difference 0.152523)
- arm_b_prime_fold2_seed43 (control, dev indicator worsened in PR #27), test: acer_higher 0.3759 -> 0.4343 (difference 0.0583783)
- arm_b_prime_fold2_seed43 (control, dev indicator worsened in PR #27), session_03: apcer_higher 0.2632 -> 0.4582 (difference 0.195015)
- arm_b_prime_fold2_seed43 (control, dev indicator worsened in PR #27), session_03: acer_higher 0.4986 -> 0.5721 (difference 0.0734642)
- arm_b_prime_fold2_seed44 (ten), test: apcer_higher 0.0000 -> 0.2735 (difference 0.273509)
- arm_b_prime_fold2_seed44 (ten), session_03: apcer_higher 0.0000 -> 0.3490 (difference 0.348974)
- arm_b_prime_fold3_seed42 (control, dev indicator worsened in PR #27), test: apcer_higher 0.0643 -> 0.6316 (difference 0.567251)
- arm_b_prime_fold3_seed42 (control, dev indicator worsened in PR #27), test: acer_higher 0.3314 -> 0.4369 (difference 0.105544)
- arm_b_prime_fold3_seed42 (control, dev indicator worsened in PR #27), session_03: apcer_higher 0.0793 -> 0.7706 (difference 0.691291)
- arm_b_prime_fold3_seed42 (control, dev indicator worsened in PR #27), session_03: acer_higher 0.4593 -> 0.5795 (difference 0.120212)
- arm_b_prime_fold3_seed43 (control, dev indicator worsened in PR #27), test: apcer_higher 0.1803 -> 0.2295 (difference 0.0492203)
- arm_b_prime_fold3_seed43 (control, dev indicator worsened in PR #27), test: acer_higher 0.2715 -> 0.2860 (difference 0.0145164)
- arm_b_prime_fold3_seed43 (control, dev indicator worsened in PR #27), session_03: apcer_higher 0.2222 -> 0.2829 (difference 0.0606607)
- arm_b_prime_fold3_seed43 (control, dev indicator worsened in PR #27), session_03: acer_higher 0.3984 -> 0.4160 (difference 0.0176136)
- arm_b_prime_fold3_seed44 (control, dev indicator worsened in PR #27), test: apcer_higher 0.0097 -> 0.2958 (difference 0.286062)
- arm_b_prime_fold3_seed44 (control, dev indicator worsened in PR #27), test: acer_higher 0.2031 -> 0.2052 (difference 0.00207949)
- arm_b_prime_fold3_seed44 (control, dev indicator worsened in PR #27), session_03: apcer_higher 0.0120 -> 0.3646 (difference 0.352553)
- arm_b_prime_fold4_seed42 (control, dev indicator worsened in PR #27), test: apcer_higher 0.2207 -> 0.2249 (difference 0.00422386)
- arm_b_prime_fold4_seed42 (control, dev indicator worsened in PR #27), test: bpcer_higher 0.1005 -> 0.5135 (difference 0.412973)
- arm_b_prime_fold4_seed42 (control, dev indicator worsened in PR #27), test: acer_higher 0.1606 -> 0.3692 (difference 0.208598)
- arm_b_prime_fold4_seed42 (control, dev indicator worsened in PR #27), test: auroc_lower 0.8717 -> 0.7445 (difference -0.127285)
- arm_b_prime_fold4_seed42 (control, dev indicator worsened in PR #27), session_03: apcer_higher 0.2787 -> 0.2813 (difference 0.00266667)
- arm_b_prime_fold4_seed42 (control, dev indicator worsened in PR #27), session_03: bpcer_higher 0.1627 -> 0.8590 (difference 0.696203)
- arm_b_prime_fold4_seed42 (control, dev indicator worsened in PR #27), session_03: acer_higher 0.2207 -> 0.5701 (difference 0.349435)
- arm_b_prime_fold4_seed42 (control, dev indicator worsened in PR #27), session_03: auroc_lower 0.7858 -> 0.5129 (difference -0.272815)
- arm_b_prime_fold4_seed43 (control), test: apcer_higher 0.2761 -> 0.3627 (difference 0.0865892)
- arm_b_prime_fold4_seed43 (control), test: acer_higher 0.3959 -> 0.4333 (difference 0.0373487)
- arm_b_prime_fold4_seed43 (control), session_03: apcer_higher 0.2887 -> 0.4573 (difference 0.168667)
- arm_b_prime_fold4_seed43 (control), session_03: acer_higher 0.5756 -> 0.6500 (difference 0.0743876)
- arm_b_prime_fold4_seed44 (ten), test: apcer_higher 0.0000 -> 0.1246 (difference 0.124604)
- arm_b_prime_fold4_seed44 (ten), session_03: apcer_higher 0.0000 -> 0.1567 (difference 0.156667)
- arm_b_prime_fold4_seed44 (ten), session_03: acer_higher 0.5000 -> 0.5096 (difference 0.00961724)
- arm_b_prime_fold4_seed44 (ten), session_03: auroc_lower 0.5000 -> 0.2328 (difference -0.267193)

Metrics not defined (a class absent in the scope): none.

## Limits

post hoc on a test already analysed, with the before metrics published and the re-estimation chosen on the dev of the same folds; one estimator, one threshold (0.5), the restored checkpoint of each training only; the subject intervals rest on 16 held-out subjects (four per fold; 15 with bona fide and 15 with attack frames on the whole test, and in session 03 only nine with bona fide frames: 2, 3, 2 and 2 per fold); with so few clusters a percentile interval may be too narrow; resamples dropped for an undefined metric are reported, not corrected; the summaries are not independent of each other (arms and groups share models and subjects); the effect on these checkpoints on this test, not a statement on other checkpoints, other thresholds, other data or the cause of the saturation

## All models, before -> after

| model | group | dev worsened (PR #27) | scope | n bona fide / attack | APCER 0.5 | BPCER 0.5 | ACER 0.5 | AUROC |
|---|---|---|---|---|---|---|---|---|
| arm_b_fold1_seed42 | ten | 0 | test | 1423 / 1819 | 0.0000 -> 0.0110 | 1.0000 -> 0.3865 | 0.5000 -> 0.1988 | 0.7389 -> 0.9217 |
| arm_b_fold1_seed42 | ten | 0 | session_03 | 925 / 1232 | 0.0000 -> 0.0162 | 1.0000 -> 0.5676 | 0.5000 -> 0.2919 | 0.6134 -> 0.8269 |
| arm_a_star_fold1_seed42 | control | 1 | test | 1423 / 1819 | 0.0132 -> 0.0610 | 0.1546 -> 0.2024 | 0.0839 -> 0.1317 | 0.9828 -> 0.9555 |
| arm_a_star_fold1_seed42 | control | 1 | session_03 | 925 / 1232 | 0.0195 -> 0.0893 | 0.2292 -> 0.2249 | 0.1243 -> 0.1571 | 0.9647 -> 0.9337 |
| arm_b_fold1_seed43 | control | 1 | test | 1423 / 1819 | 0.0038 -> 0.0027 | 0.6430 -> 0.7048 | 0.3234 -> 0.3538 | 0.8948 -> 0.7598 |
| arm_b_fold1_seed43 | control | 1 | session_03 | 925 / 1232 | 0.0057 -> 0.0041 | 0.9373 -> 0.9978 | 0.4715 -> 0.5009 | 0.7812 -> 0.5268 |
| arm_a_star_fold1_seed43 | control | 1 | test | 1423 / 1819 | 0.0000 -> 0.0000 | 0.1026 -> 0.3120 | 0.0513 -> 0.1560 | 0.9921 -> 0.9934 |
| arm_a_star_fold1_seed43 | control | 1 | session_03 | 925 / 1232 | 0.0000 -> 0.0000 | 0.0562 -> 0.3589 | 0.0281 -> 0.1795 | 0.9909 -> 0.9879 |
| arm_b_fold1_seed44 | control | 1 | test | 1423 / 1819 | 0.0033 -> 0.0902 | 0.7098 -> 0.5369 | 0.3565 -> 0.3135 | 0.6400 -> 0.7759 |
| arm_b_fold1_seed44 | control | 1 | session_03 | 925 / 1232 | 0.0049 -> 0.1331 | 0.9892 -> 0.7935 | 0.4970 -> 0.4633 | 0.3809 -> 0.5595 |
| arm_a_star_fold1_seed44 | control | 1 | test | 1423 / 1819 | 0.0495 -> 0.0011 | 0.2361 -> 0.3450 | 0.1428 -> 0.1731 | 0.8963 -> 0.9165 |
| arm_a_star_fold1_seed44 | control | 1 | session_03 | 925 / 1232 | 0.0722 -> 0.0016 | 0.3222 -> 0.3957 | 0.1972 -> 0.1986 | 0.8401 -> 0.8750 |
| arm_b_fold2_seed42 | control | 0 | test | 1370 / 1744 | 0.1623 -> 0.4008 | 0.3891 -> 0.0197 | 0.2757 -> 0.2103 | 0.8368 -> 0.9406 |
| arm_b_fold2_seed42 | control | 0 | session_03 | 1019 / 1364 | 0.2075 -> 0.5117 | 0.5211 -> 0.0255 | 0.3643 -> 0.2686 | 0.7238 -> 0.8991 |
| arm_a_star_fold2_seed42 | control | 1 | test | 1370 / 1744 | 0.1181 -> 0.0986 | 0.0693 -> 0.0942 | 0.0937 -> 0.0964 | 0.9701 -> 0.9607 |
| arm_a_star_fold2_seed42 | control | 1 | session_03 | 1019 / 1364 | 0.1510 -> 0.1261 | 0.0922 -> 0.1256 | 0.1216 -> 0.1259 | 0.9502 -> 0.9364 |
| arm_b_fold2_seed43 | control | 0 | test | 1370 / 1744 | 0.0688 -> 0.2018 | 0.6270 -> 0.5394 | 0.3479 -> 0.3706 | 0.6701 -> 0.6939 |
| arm_b_fold2_seed43 | control | 0 | session_03 | 1019 / 1364 | 0.0880 -> 0.2581 | 0.8420 -> 0.7252 | 0.4650 -> 0.4916 | 0.4744 -> 0.5270 |
| arm_a_star_fold2_seed43 | ten | 0 | test | 1370 / 1744 | 0.0000 -> 0.1181 | 1.0000 -> 0.0051 | 0.5000 -> 0.0616 | 0.5000 -> 0.9972 |
| arm_a_star_fold2_seed43 | ten | 0 | session_03 | 1019 / 1364 | 0.0000 -> 0.1334 | 1.0000 -> 0.0059 | 0.5000 -> 0.0697 | 0.5000 -> 0.9961 |
| arm_b_fold2_seed44 | control | 0 | test | 1370 / 1744 | 0.0063 -> 0.1227 | 0.3255 -> 0.0350 | 0.1659 -> 0.0789 | 0.9865 -> 0.9706 |
| arm_b_fold2_seed44 | control | 0 | session_03 | 1019 / 1364 | 0.0081 -> 0.1569 | 0.4357 -> 0.0461 | 0.2219 -> 0.1015 | 0.9779 -> 0.9521 |
| arm_a_star_fold2_seed44 | ten | 0 | test | 1370 / 1744 | 1.0000 -> 0.1055 | 0.0000 -> 0.0029 | 0.5000 -> 0.0542 | 0.2782 -> 0.9981 |
| arm_a_star_fold2_seed44 | ten | 0 | session_03 | 1019 / 1364 | 1.0000 -> 0.1334 | 0.0000 -> 0.0020 | 0.5000 -> 0.0677 | 0.2646 -> 0.9994 |
| arm_b_fold3_seed42 | control | 0 | test | 1387 / 2052 | 0.1038 -> 0.1696 | 0.3143 -> 0.3216 | 0.2091 -> 0.2456 | 0.8298 -> 0.6852 |
| arm_b_fold3_seed42 | control | 0 | session_03 | 865 / 1665 | 0.1279 -> 0.2078 | 0.5029 -> 0.5156 | 0.3154 -> 0.3617 | 0.6672 -> 0.4946 |
| arm_a_star_fold3_seed42 | ten | 0 | test | 1387 / 2052 | 1.0000 -> 0.1857 | 0.0000 -> 0.0360 | 0.5000 -> 0.1109 | 0.5255 -> 0.9727 |
| arm_a_star_fold3_seed42 | ten | 0 | session_03 | 865 / 1665 | 1.0000 -> 0.2288 | 0.0000 -> 0.0578 | 0.5000 -> 0.1433 | 0.5029 -> 0.9471 |
| arm_b_fold3_seed43 | control | 1 | test | 1387 / 2052 | 0.2003 -> 0.2061 | 0.3107 -> 0.3107 | 0.2555 -> 0.2584 | 0.7776 -> 0.7653 |
| arm_b_fold3_seed43 | control | 1 | session_03 | 865 / 1665 | 0.2468 -> 0.2541 | 0.4983 -> 0.4983 | 0.3726 -> 0.3762 | 0.5781 -> 0.5658 |
| arm_a_star_fold3_seed43 | control | 1 | test | 1387 / 2052 | 0.3289 -> 0.1209 | 0.0382 -> 0.0231 | 0.1836 -> 0.0720 | 0.9534 -> 0.9727 |
| arm_a_star_fold3_seed43 | control | 1 | session_03 | 865 / 1665 | 0.4054 -> 0.1489 | 0.0613 -> 0.0370 | 0.2333 -> 0.0930 | 0.9113 -> 0.9546 |
| arm_b_fold3_seed44 | apart | 0 | test | 1387 / 2052 | 0.6418 -> 0.0351 | 0.8111 -> 0.3432 | 0.7265 -> 0.1891 | 0.2781 -> 0.8301 |
| arm_b_fold3_seed44 | apart | 0 | session_03 | 865 / 1665 | 0.5652 -> 0.0432 | 0.7457 -> 0.5503 | 0.6554 -> 0.2968 | 0.3959 -> 0.6774 |
| arm_a_star_fold3_seed44 | ten | 0 | test | 1387 / 2052 | 0.0000 -> 0.1140 | 1.0000 -> 0.2322 | 0.5000 -> 0.1731 | 0.5000 -> 0.9023 |
| arm_a_star_fold3_seed44 | ten | 0 | session_03 | 865 / 1665 | 0.0000 -> 0.1381 | 1.0000 -> 0.3723 | 0.5000 -> 0.2552 | 0.5000 -> 0.8202 |
| arm_b_fold4_seed42 | control | 1 | test | 925 / 1894 | 0.3590 -> 0.3374 | 0.4281 -> 0.5081 | 0.3936 -> 0.4227 | 0.7502 -> 0.6143 |
| arm_b_fold4_seed42 | control | 1 | session_03 | 553 / 1500 | 0.4260 -> 0.3780 | 0.7161 -> 0.8499 | 0.5710 -> 0.6140 | 0.5357 -> 0.3829 |
| arm_a_star_fold4_seed42 | control | 1 | test | 925 / 1894 | 0.2012 -> 0.1605 | 0.1514 -> 0.1546 | 0.1763 -> 0.1576 | 0.9271 -> 0.9416 |
| arm_a_star_fold4_seed42 | control | 1 | session_03 | 553 / 1500 | 0.2540 -> 0.2027 | 0.0000 -> 0.0000 | 0.1270 -> 0.1013 | 0.9714 -> 0.9872 |
| arm_b_fold4_seed43 | control | 1 | test | 925 / 1894 | 0.3395 -> 0.4240 | 0.5157 -> 0.5114 | 0.4276 -> 0.4677 | 0.5968 -> 0.6553 |
| arm_b_fold4_seed43 | control | 1 | session_03 | 553 / 1500 | 0.4007 -> 0.5067 | 0.8626 -> 0.8553 | 0.6316 -> 0.6810 | 0.2230 -> 0.3315 |
| arm_a_star_fold4_seed43 | control | 1 | test | 925 / 1894 | 0.3490 -> 0.2091 | 0.0065 -> 0.2205 | 0.1777 -> 0.2148 | 0.9227 -> 0.7956 |
| arm_a_star_fold4_seed43 | control | 1 | session_03 | 553 / 1500 | 0.4393 -> 0.2633 | 0.0108 -> 0.3671 | 0.2251 -> 0.3152 | 0.8788 -> 0.6723 |
| arm_b_fold4_seed44 | ten | 0 | test | 925 / 1894 | 0.0000 -> 0.5370 | 1.0000 -> 0.5103 | 0.5000 -> 0.5236 | 0.4672 -> 0.6272 |
| arm_b_fold4_seed44 | ten | 0 | session_03 | 553 / 1500 | 0.0000 -> 0.6313 | 1.0000 -> 0.8535 | 0.5000 -> 0.7424 | 0.1344 -> 0.2809 |
| arm_a_star_fold4_seed44 | ten | 0 | test | 925 / 1894 | 0.0000 -> 0.0211 | 1.0000 -> 0.1892 | 0.5000 -> 0.1052 | 0.5000 -> 0.9799 |
| arm_a_star_fold4_seed44 | ten | 0 | session_03 | 553 / 1500 | 0.0000 -> 0.0267 | 1.0000 -> 0.0181 | 0.5000 -> 0.0224 | 0.5000 -> 0.9973 |
| arm_b_prime_fold1_seed42 | control | 0 | test | 1423 / 1819 | 0.6668 -> 0.0220 | 0.3197 -> 0.4034 | 0.4933 -> 0.2127 | 0.5029 -> 0.9156 |
| arm_b_prime_fold1_seed42 | control | 0 | session_03 | 925 / 1232 | 0.9513 -> 0.0325 | 0.4919 -> 0.5211 | 0.7216 -> 0.2768 | 0.1698 -> 0.8392 |
| arm_b_prime_fold1_seed43 | control | 0 | test | 1423 / 1819 | 0.0082 -> 0.0220 | 0.8180 -> 0.6894 | 0.4131 -> 0.3557 | 0.5857 -> 0.6022 |
| arm_b_prime_fold1_seed43 | control | 0 | session_03 | 925 / 1232 | 0.0114 -> 0.0308 | 1.0000 -> 1.0000 | 0.5057 -> 0.5154 | 0.4761 -> 0.2593 |
| arm_b_prime_fold1_seed44 | ten | 0 | test | 1423 / 1819 | 0.0000 -> 0.1039 | 1.0000 -> 0.5502 | 0.5000 -> 0.3271 | 0.5000 -> 0.8025 |
| arm_b_prime_fold1_seed44 | ten | 0 | session_03 | 925 / 1232 | 0.0000 -> 0.1534 | 1.0000 -> 0.7968 | 0.5000 -> 0.4751 | 0.5000 -> 0.5930 |
| arm_b_prime_fold2_seed42 | control | 0 | test | 1370 / 1744 | 0.1519 -> 0.3612 | 0.0964 -> 0.0015 | 0.1241 -> 0.1813 | 0.9281 -> 0.9271 |
| arm_b_prime_fold2_seed42 | control | 0 | session_03 | 1019 / 1364 | 0.1928 -> 0.4619 | 0.0059 -> 0.0010 | 0.0994 -> 0.2314 | 0.9685 -> 0.8756 |
| arm_b_prime_fold2_seed43 | control | 1 | test | 1370 / 1744 | 0.2058 -> 0.3584 | 0.5460 -> 0.5102 | 0.3759 -> 0.4343 | 0.6391 -> 0.6623 |
| arm_b_prime_fold2_seed43 | control | 1 | session_03 | 1019 / 1364 | 0.2632 -> 0.4582 | 0.7341 -> 0.6860 | 0.4986 -> 0.5721 | 0.3946 -> 0.4625 |
| arm_b_prime_fold2_seed44 | ten | 0 | test | 1370 / 1744 | 0.0000 -> 0.2735 | 1.0000 -> 0.0036 | 0.5000 -> 0.1386 | 0.5000 -> 0.9845 |
| arm_b_prime_fold2_seed44 | ten | 0 | session_03 | 1019 / 1364 | 0.0000 -> 0.3490 | 1.0000 -> 0.0049 | 0.5000 -> 0.1769 | 0.5000 -> 0.9747 |
| arm_b_prime_fold3_seed42 | control | 1 | test | 1387 / 2052 | 0.0643 -> 0.6316 | 0.5984 -> 0.2422 | 0.3314 -> 0.4369 | 0.7411 -> 0.7787 |
| arm_b_prime_fold3_seed42 | control | 1 | session_03 | 865 / 1665 | 0.0793 -> 0.7706 | 0.8393 -> 0.3884 | 0.4593 -> 0.5795 | 0.5306 -> 0.5749 |
| arm_b_prime_fold3_seed43 | control | 1 | test | 1387 / 2052 | 0.1803 -> 0.2295 | 0.3627 -> 0.3425 | 0.2715 -> 0.2860 | 0.6528 -> 0.7142 |
| arm_b_prime_fold3_seed43 | control | 1 | session_03 | 865 / 1665 | 0.2222 -> 0.2829 | 0.5746 -> 0.5491 | 0.3984 -> 0.4160 | 0.4509 -> 0.5011 |
| arm_b_prime_fold3_seed44 | control | 1 | test | 1387 / 2052 | 0.0097 -> 0.2958 | 0.3965 -> 0.1146 | 0.2031 -> 0.2052 | 0.7675 -> 0.9255 |
| arm_b_prime_fold3_seed44 | control | 1 | session_03 | 865 / 1665 | 0.0120 -> 0.3646 | 0.6335 -> 0.1838 | 0.3228 -> 0.2742 | 0.5907 -> 0.8551 |
| arm_b_prime_fold4_seed42 | control | 1 | test | 925 / 1894 | 0.2207 -> 0.2249 | 0.1005 -> 0.5135 | 0.1606 -> 0.3692 | 0.8717 -> 0.7445 |
| arm_b_prime_fold4_seed42 | control | 1 | session_03 | 553 / 1500 | 0.2787 -> 0.2813 | 0.1627 -> 0.8590 | 0.2207 -> 0.5701 | 0.7858 -> 0.5129 |
| arm_b_prime_fold4_seed43 | control | 0 | test | 925 / 1894 | 0.2761 -> 0.3627 | 0.5157 -> 0.5038 | 0.3959 -> 0.4333 | 0.4826 -> 0.7240 |
| arm_b_prime_fold4_seed43 | control | 0 | session_03 | 553 / 1500 | 0.2887 -> 0.4573 | 0.8626 -> 0.8427 | 0.5756 -> 0.6500 | 0.1470 -> 0.4671 |
| arm_b_prime_fold4_seed44 | ten | 0 | test | 925 / 1894 | 0.0000 -> 0.1246 | 1.0000 -> 0.5157 | 0.5000 -> 0.3201 | 0.5000 -> 0.6134 |
| arm_b_prime_fold4_seed44 | ten | 0 | session_03 | 553 / 1500 | 0.0000 -> 0.1567 | 1.0000 -> 0.8626 | 0.5000 -> 0.5096 | 0.5000 -> 0.2328 |


## Time

Wall time 3027 s (50.5 min), process CPU time 18438 s, against the note's estimate of about 42 minutes (upper estimate about 112); wall-time cap 240 minutes.

| phase | seconds |
|---|---|
| phase 0 inputs | 3.7 |
| phase 1 baseline | 744.3 |
| phase 2 recalibration | 1607.1 |
| phase 3a scoring with the re-estimated statistics | 664.6 |
| input re-verification | 0.8 |
| phase 3b metrics | 2.2 |
| summaries, final checks and writes | 2.1 |

