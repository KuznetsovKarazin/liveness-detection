# Annex: folds run versus N1 on the same 3,000 frames (descriptive, run `20261104-FOLDSCOMPARE-3b3bcca`)

Copied from the public aggregate of the comparison (`nuaa-folds-n1-compare/tables/aggregate.json`, checked by `nuaa_synthesis.py`: N1 values equal to the N1 rows of the unified table, folds counts inside the pooled session 03 of the folds run, differences = folds minus N1, paired decisions and subject counts consistent with the counts, medians and ranges equal to the per-seed values, control values reproduced). The 3,000 frames are those of `nuaa_cnn_test` (1,500 bona fide, 1,500 attacks, 16 subjects, session 03), on which N1 was tested; they are a subset of the pooled session 03 of the folds run, so the folds values here differ from the pooled rows of `folds_run_table.md`. Descriptive and post hoc: existing scores and thresholds already frozen; no training, no inference, no new threshold. N1 has only the fixed threshold 0.5: the dev values compare the folds run at its frozen dev thresholds with N1 at 0.5. attack if score_attack > threshold (tie = bona fide); label 0 bona fide, 1 attack; score = probability of attack. The AUROC of the folds run pools the scores of four models (one per fold). Training roles differ: N1 was trained on 9 of the 16 subjects (sessions 01-02); in the folds run no frame is scored by a model that saw its subject. N1 AUROC is below 0.5 on these frames (0.3485-0.4582, as published by N1); the AUROC difference includes that. Bootstrap over the 16 subjects, paired: 1000 resamples, seed 0, percentiles 2.5-97.5; recipe: generator numpy.random.default_rng(0) (PCG64); subjects in ascending identifier order; at every resample integers(0, n, n) with n = the number of subjects (16); the generator restarts from 0 for every training seed (the same 1,000 resamples for 42, 43 and 44); every metric computed on the pooled frames of the resampled subjects, not as a mean of per-subject values; the same subject indices for both experiments; percentiles 2.5 and 97.5 with np.percentile, linear interpolation; a resample without a class does not count for the metrics that need it; no interval with fewer than 5 subjects. Indicative; no causal reading.

## Per seed

| seed | experiment | rej. bf / acc. att. | BPCER | APCER | ACER [bootstrap] | AUROC [bootstrap] |
|---|---|---|---|---|---|---|
| 42 | folds, 0.5 | 157 / 170 | 0.1047 | 0.1133 | 0.1090 [0.0244, 0.2148] | 0.9385 [0.8650, 0.9961] |
| 42 | folds, dev | 153 / 133 | 0.1020 | 0.0887 | 0.0953 [0.0215, 0.1932] | as above |
| 42 | N1, 0.5 | 1155 / 160 | 0.7700 | 0.1067 | 0.4383 [0.2985, 0.5450] | 0.3485 [0.1197, 0.6331] |
| 43 | folds, 0.5 | 237 / 213 | 0.1580 | 0.1420 | 0.1500 [0.0453, 0.2943] | 0.9527 [0.8602, 0.9968] |
| 43 | folds, dev | 203 / 256 | 0.1353 | 0.1707 | 0.1530 [0.0520, 0.2898] | as above |
| 43 | N1, 0.5 | 1152 / 391 | 0.7680 | 0.2607 | 0.5143 [0.3480, 0.6572] | 0.3754 [0.1491, 0.6476] |
| 44 | folds, 0.5 | 235 / 65 | 0.1567 | 0.0433 | 0.1000 [0.0182, 0.2309] | 0.9670 [0.9075, 0.9990] |
| 44 | folds, dev | 212 / 139 | 0.1413 | 0.0927 | 0.1170 [0.0308, 0.2438] | as above |
| 44 | N1, 0.5 | 1131 / 72 | 0.7540 | 0.0480 | 0.4010 [0.2510, 0.5119] | 0.4582 [0.2350, 0.7307] |

## Differences, folds minus N1 (paired bootstrap over subjects)

| seed | ACER folds 0.5 - N1 0.5 | ACER folds dev - N1 0.5 | AUROC folds - N1 |
|---|---|---|---|
| 42 | -0.3293 [-0.4796, -0.1527] | -0.3430 [-0.4847, -0.1684] | 0.5900 [0.2821, 0.8453] |
| 43 | -0.3643 [-0.5270, -0.1841] | -0.3613 [-0.5206, -0.1856] | 0.5772 [0.3090, 0.8056] |
| 44 | -0.3010 [-0.4600, -0.0945] | -0.2840 [-0.4439, -0.0815] | 0.5088 [0.2094, 0.7525] |

## Median and range over the three seeds

| measure | median | min | max |
|---|---|---|---|
| folds ACER 0.5 | 0.1090 | 0.1000 | 0.1500 |
| folds ACER dev | 0.1170 | 0.0953 | 0.1530 |
| folds AUROC | 0.9527 | 0.9385 | 0.9670 |
| N1 ACER 0.5 | 0.4383 | 0.4010 | 0.5143 |
| N1 AUROC | 0.3754 | 0.3485 | 0.4582 |
| difference ACER 0.5 | -0.3293 | -0.3643 | -0.3010 |
| difference ACER dev vs N1 0.5 | -0.3430 | -0.3613 | -0.2840 |
| difference AUROC | 0.5772 | 0.5088 | 0.5900 |

## Paired decisions per frame

Per class: frames correct in both experiments, only in N1, only in the folds run, in neither; N1 always at 0.5. Changed decisions = frames of either class on which the two decisions differ, one count per seed and threshold.

| seed | folds at | bona fide: both ok / N1 only ok / folds only ok / both wrong | attack: same cells | changed decisions (both classes) |
|---|---|---|---|---|
| 42 | 0.5 | 277 / 68 / 1066 / 89 | 1183 / 157 / 147 / 13 | 1438 |
| 42 | dev | 279 / 66 / 1068 / 87 | 1215 / 125 / 152 / 8 | 1411 |
| 43 | 0.5 | 317 / 31 / 946 / 206 | 948 / 161 / 339 / 52 | 1477 |
| 43 | dev | 326 / 22 / 971 / 181 | 918 / 191 / 326 / 65 | 1510 |
| 44 | 0.5 | 275 / 94 / 990 / 141 | 1371 / 57 / 64 / 8 | 1205 |
| 44 | dev | 275 / 94 / 1013 / 118 | 1305 / 123 / 56 / 16 | 1286 |

## Subjects (counts only)

Subjects where the folds run is lower, equal or higher than N1 at 0.5; error rate = wrong frames / frames, ACER only for the 8 subjects with both classes.

| seed | folds at | measure | folds lower | equal | folds higher | not defined |
|---|---|---|---|---|---|---|
| 42 | 0.5 | error rate | 12 | 0 | 4 | 0 |
| 42 | 0.5 | acer | 5 | 0 | 3 | 8 |
| 42 | dev | error rate | 12 | 0 | 4 | 0 |
| 42 | dev | acer | 5 | 0 | 3 | 8 |
| 43 | 0.5 | error rate | 14 | 0 | 2 | 0 |
| 43 | 0.5 | acer | 7 | 0 | 1 | 8 |
| 43 | dev | error rate | 13 | 0 | 3 | 0 |
| 43 | dev | acer | 7 | 0 | 1 | 8 |
| 44 | 0.5 | error rate | 10 | 2 | 4 | 0 |
| 44 | 0.5 | acer | 4 | 1 | 3 | 8 |
| 44 | dev | error rate | 10 | 2 | 4 | 0 |
| 44 | dev | acer | 4 | 1 | 3 | 8 |

## Control values of the coordinator

| measure | seed | control (as given) | computed |
|---|---|---|---|
| acer_05 | 42 | 10.90 % | 0.10900 |
| acer_05 | 43 | 15.00 % | 0.15000 |
| acer_05 | 44 | 10.00 % | 0.10000 |
| acer_dev | 42 | 9.53 % | 0.09533 |
| acer_dev | 43 | 15.30 % | 0.15300 |
| acer_dev | 44 | 11.70 % | 0.11700 |
| auroc | 42 | 0.93851 | 0.93851 |
| auroc | 43 | 0.95268 | 0.95268 |
| auroc | 44 | 0.96701 | 0.96701 |

Comparison: agreement at the digits given by the coordinator: round(100 * ACER, 2) equal to the given percentage, round(AUROC, 5) equal to the given AUROC.
