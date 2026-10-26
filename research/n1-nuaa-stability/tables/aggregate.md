# N1 part 2: LivenessNet on NUAA, three training seeds (study `20261024-N1-LivenessNet-fb93496`)

Split seed 42 for every run (2,507 fitting / 493 validation frames, M2 manifests); training seeds 42, 43, 44. Threshold 0.5 on the attack probability, attack if score_attack > 0.5 (tie = bona fide). Scores on CPU. No seed selection: every seed of the configuration is reported. Median and range (minimum-maximum) over the three seeds.

## test (3,000 frames of nuaa_cnn_test)

| metric | seed 42 | seed 43 | seed 44 | median | range |
|---|---|---|---|---|---|
| APCER | 0.1067 | 0.2607 | 0.0480 | 0.1067 | 0.0480-0.2607 |
| BPCER | 0.7700 | 0.7680 | 0.7540 | 0.7680 | 0.7540-0.7700 |
| ACER | 0.4383 | 0.5143 | 0.4010 | 0.4383 | 0.4010-0.5143 |
| AUROC | 0.3485 | 0.3754 | 0.4582 | 0.3754 | 0.3485-0.4582 |
| accepted attacks / attacks | 160/1500 | 391/1500 | 72/1500 | | |
| rejected bona fide / bona fide | 1155/1500 | 1152/1500 | 1131/1500 | | |

## C1 subset (300 frames of nuaa_c1_v2, inside the test)

| metric | seed 42 | seed 43 | seed 44 | median | range |
|---|---|---|---|---|---|
| APCER | 0.0867 | 0.2533 | 0.0400 | 0.0867 | 0.0400-0.2533 |
| BPCER | 0.7800 | 0.7800 | 0.7667 | 0.7800 | 0.7667-0.7800 |
| ACER | 0.4333 | 0.5167 | 0.4033 | 0.4333 | 0.4033-0.5167 |
| AUROC | 0.3374 | 0.3714 | 0.4641 | 0.3714 | 0.3374-0.4641 |
| accepted attacks / attacks | 13/150 | 38/150 | 6/150 | | |
| rejected bona fide / bona fide | 117/150 | 117/150 | 115/150 | | |

## Training

| | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| epochs run (max 12) | 9 | 12 | 12 |
| restored epoch (minimum val_loss) | 5 | 12 | 8 |
| minimum val_loss | 0.000898 | 0.012337 | 0.000542 |
| val_loss recomputed on CPU | 0.000899 | 0.012337 | 0.000542 |
| training wall time (s) | 496.0 | 671.0 | 671.4 |
| weights SHA-256 (.h5) | `c63a858107b6…` | `34bac22cafd3…` | `41c128484ff7…` |

NA = not computable (class absent). Values are rounded to four decimals; aggregate.json holds the exact values.
