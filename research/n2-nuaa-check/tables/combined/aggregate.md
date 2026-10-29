# N2 part 2: the historical combined AttackNetV2.2 on the 3,000 NUAA test frames of N1, two image-preparation variants (study `20261029-N2-combined-247dfd7`)

Checkpoint `AttackNetV2_2_combined_all_final.h5` (SHA-256 `3dd0c85341e3…`), loaded as saved, no retraining. Same 3,000 frames as N1 (manifest `nuaa_cnn_test.csv`), every frame scored (no quality filter). Threshold 0.5 fixed before the test, attack if score_attack > 0.5 (tie = bona fide); no calibration, no score inversion. The N1 rows are the three LivenessNet trainings of study `20261024-N1-LivenessNet-fb93496`, read from its aggregate. The comparison with N1 changes architecture, training recipe and training data at once; the two variants are a sensitivity analysis to the image preparation, not a reproduction of the historical preparation.

## Single table

| model / variant | set | n bona fide | n attack | rejected bona fide | accepted attacks | APCER | BPCER | ACER | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| combined AttackNetV2.2, whole image | test (3,000 frames of nuaa_cnn_test) | 1500 | 1500 | 1492 | 222 | 0.1480 | 0.9947 | 0.5713 | 0.2169 |
| combined AttackNetV2.2, whole image | C1 subset (300 frames of nuaa_c1_v2, inside the test) | 150 | 150 | 149 | 23 | 0.1533 | 0.9933 | 0.5733 | 0.1975 |
| combined AttackNetV2.2, face crop | test (3,000 frames of nuaa_cnn_test) | 1500 | 1500 | 1224 | 353 | 0.2353 | 0.8160 | 0.5257 | 0.3706 |
| combined AttackNetV2.2, face crop | C1 subset (300 frames of nuaa_c1_v2, inside the test) | 150 | 150 | 125 | 36 | 0.2400 | 0.8333 | 0.5367 | 0.3499 |
| N1 LivenessNet, seed 42 | test (3,000 frames of nuaa_cnn_test) | 1500 | 1500 | 1155 | 160 | 0.1067 | 0.7700 | 0.4383 | 0.3485 |
| N1 LivenessNet, seed 42 | C1 subset (300 frames of nuaa_c1_v2, inside the test) | 150 | 150 | 117 | 13 | 0.0867 | 0.7800 | 0.4333 | 0.3374 |
| N1 LivenessNet, seed 43 | test (3,000 frames of nuaa_cnn_test) | 1500 | 1500 | 1152 | 391 | 0.2607 | 0.7680 | 0.5143 | 0.3754 |
| N1 LivenessNet, seed 43 | C1 subset (300 frames of nuaa_c1_v2, inside the test) | 150 | 150 | 117 | 38 | 0.2533 | 0.7800 | 0.5167 | 0.3714 |
| N1 LivenessNet, seed 44 | test (3,000 frames of nuaa_cnn_test) | 1500 | 1500 | 1131 | 72 | 0.0480 | 0.7540 | 0.4010 | 0.4582 |
| N1 LivenessNet, seed 44 | C1 subset (300 frames of nuaa_c1_v2, inside the test) | 150 | 150 | 115 | 6 | 0.0400 | 0.7667 | 0.4033 | 0.4641 |

## Image preparation per variant

| variant | detector counts on the 3,000 frames | frames without a face (whole image used) |
|---|---|---|
| whole image | n/a 3000 | NA |
| face crop | Haar 3, RetinaFace 2996, none 1 | 1 |

## Score distributions per class, test 3,000 (counts per bin of width 0.1)

| variant | class | [0.0, 0.1) | [0.1, 0.2) | [0.2, 0.3) | [0.3, 0.4) | [0.4, 0.5) | [0.5, 0.6) | [0.6, 0.7) | [0.7, 0.8) | [0.8, 0.9) | [0.9, 1.0] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| whole image | bona fide | 1 | 2 | 3 | 1 | 1 | 1 | 3 | 3 | 8 | 1477 |
| whole image | attack | 106 | 45 | 20 | 19 | 32 | 23 | 29 | 46 | 60 | 1120 |
| face crop | bona fide | 214 | 17 | 20 | 12 | 13 | 16 | 12 | 14 | 33 | 1149 |
| face crop | attack | 203 | 41 | 42 | 31 | 36 | 32 | 43 | 50 | 74 | 948 |

NA = not computable (class absent). Values are rounded to four decimals; aggregate.json holds the exact values.
