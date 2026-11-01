# N2 part 1: the three N1 LivenessNet checkpoints on the NUAA frames of sessions 01-02 outside fitting and validation (study `20261029-N2-sessions-2ac3e29`)

Target: 491 frames of the official training list (243 bona fide, 248 attacks, 9 subjects: 7 fitting, 2 validation; 81 also in the M1 dev set), preprocessed with the frozen detector that reproduces the training cache (6000 of 6000 cache frames bit for bit). Reference: the session 03 scores of the N1 study, read as stored (parity on 24 keys: max |difference| 0.00e+00, tolerance 1e-06). Threshold 0.5 on the attack probability, attack if score_attack > 0.5 (tie = bona fide); scores on CPU; no seed selection, no threshold change, no score inversion. Median and range (minimum-maximum) over the three seeds. NA = not computable (class absent).

## Single table: every set, every seed

| set | seed | n bona fide | n attack | rejected bona fide | accepted attacks | APCER | BPCER | ACER | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| sessions 01-02: the 491 target frames | 42 | 243 | 248 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| sessions 01-02: the 491 target frames | 43 | 243 | 248 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| sessions 01-02: the 491 target frames | 44 | 243 | 248 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| sessions 01-02: the 491 target frames | median (range) | | | | | 0.0000 (0.0000-0.0000) | 0.0000 (0.0000-0.0000) | 0.0000 (0.0000-0.0000) | 1.0000 (1.0000-1.0000) |
| sessions 01-02: target frames of the 7 fitting subjects | 42 | 218 | 192 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| sessions 01-02: target frames of the 7 fitting subjects | 43 | 218 | 192 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| sessions 01-02: target frames of the 7 fitting subjects | 44 | 218 | 192 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| sessions 01-02: target frames of the 7 fitting subjects | median (range) | | | | | 0.0000 (0.0000-0.0000) | 0.0000 (0.0000-0.0000) | 0.0000 (0.0000-0.0000) | 1.0000 (1.0000-1.0000) |
| sessions 01-02: target frames of the 2 validation subjects | 42 | 25 | 56 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| sessions 01-02: target frames of the 2 validation subjects | 43 | 25 | 56 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| sessions 01-02: target frames of the 2 validation subjects | 44 | 25 | 56 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| sessions 01-02: target frames of the 2 validation subjects | median (range) | | | | | 0.0000 (0.0000-0.0000) | 0.0000 (0.0000-0.0000) | 0.0000 (0.0000-0.0000) | 1.0000 (1.0000-1.0000) |
| sessions 01-02: target frames outside the M1 dev set | 42 | 218 | 192 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| sessions 01-02: target frames outside the M1 dev set | 43 | 218 | 192 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| sessions 01-02: target frames outside the M1 dev set | 44 | 218 | 192 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| sessions 01-02: target frames outside the M1 dev set | median (range) | | | | | 0.0000 (0.0000-0.0000) | 0.0000 (0.0000-0.0000) | 0.0000 (0.0000-0.0000) | 1.0000 (1.0000-1.0000) |
| session 03: the 3,000 nuaa_cnn_test frames (N1 scores, read as stored) | 42 | 1500 | 1500 | 1155 | 160 | 0.1067 | 0.7700 | 0.4383 | 0.3485 |
| session 03: the 3,000 nuaa_cnn_test frames (N1 scores, read as stored) | 43 | 1500 | 1500 | 1152 | 391 | 0.2607 | 0.7680 | 0.5143 | 0.3754 |
| session 03: the 3,000 nuaa_cnn_test frames (N1 scores, read as stored) | 44 | 1500 | 1500 | 1131 | 72 | 0.0480 | 0.7540 | 0.4010 | 0.4582 |
| session 03: the 3,000 nuaa_cnn_test frames (N1 scores, read as stored) | median (range) | | | | | 0.1067 (0.0480-0.2607) | 0.7680 (0.7540-0.7700) | 0.4383 (0.4010-0.5143) | 0.3754 (0.3485-0.4582) |
| session 03: test frames of the 9 target subjects (N1 scores) | 42 | 614 | 924 | 422 | 154 | 0.1667 | 0.6873 | 0.4270 | 0.3230 |
| session 03: test frames of the 9 target subjects (N1 scores) | 43 | 614 | 924 | 422 | 359 | 0.3885 | 0.6873 | 0.5379 | 0.3171 |
| session 03: test frames of the 9 target subjects (N1 scores) | 44 | 614 | 924 | 422 | 63 | 0.0682 | 0.6873 | 0.3777 | 0.4569 |
| session 03: test frames of the 9 target subjects (N1 scores) | median (range) | | | | | 0.1667 (0.0682-0.3885) | 0.6873 (0.6873-0.6873) | 0.4270 (0.3777-0.5379) | 0.3230 (0.3171-0.4569) |
| session 03: test frames of the 7 fitting subjects (N1 scores) | 42 | 614 | 711 | 422 | 116 | 0.1632 | 0.6873 | 0.4252 | 0.3256 |
| session 03: test frames of the 7 fitting subjects (N1 scores) | 43 | 614 | 711 | 422 | 270 | 0.3797 | 0.6873 | 0.5335 | 0.3191 |
| session 03: test frames of the 7 fitting subjects (N1 scores) | 44 | 614 | 711 | 422 | 42 | 0.0591 | 0.6873 | 0.3732 | 0.4526 |
| session 03: test frames of the 7 fitting subjects (N1 scores) | median (range) | | | | | 0.1632 (0.0591-0.3797) | 0.6873 (0.6873-0.6873) | 0.4252 (0.3732-0.5335) | 0.3256 (0.3191-0.4526) |
| session 03: test frames of the 2 validation subjects (N1 scores) | 42 | 0 | 213 | 0 | 38 | 0.1784 | NA | NA | NA |
| session 03: test frames of the 2 validation subjects (N1 scores) | 43 | 0 | 213 | 0 | 89 | 0.4178 | NA | NA | NA |
| session 03: test frames of the 2 validation subjects (N1 scores) | 44 | 0 | 213 | 0 | 21 | 0.0986 | NA | NA | NA |
| session 03: test frames of the 2 validation subjects (N1 scores) | median (range) | | | | | 0.1784 (0.0986-0.4178) | NA (NA) | NA (NA) | NA (NA) |

## Subject x class strata (sessions 01-02 against session 03, same subjects)

Per seed, group and class: strata computable in both sets and how many have a lower, equal or higher error rate in sessions 01-02 than in session 03 (per-subject values are in the restricted `strata_all_seeds.csv` and `figure_strata.png`).

| seed | group / class | strata | lower in 01-02 | equal | higher in 01-02 | not comparable |
|---|---|---|---|---|---|---|
| 42 | fitting/bona fide | 7 | 2 | 1 | 0 | 4 |
| 42 | fitting/attack | 7 | 7 | 0 | 0 | 0 |
| 42 | validation/bona fide | 2 | 0 | 0 | 0 | 2 |
| 42 | validation/attack | 2 | 2 | 0 | 0 | 0 |
| 43 | fitting/bona fide | 7 | 2 | 1 | 0 | 4 |
| 43 | fitting/attack | 7 | 7 | 0 | 0 | 0 |
| 43 | validation/bona fide | 2 | 0 | 0 | 0 | 2 |
| 43 | validation/attack | 2 | 2 | 0 | 0 | 0 |
| 44 | fitting/bona fide | 7 | 2 | 1 | 0 | 4 |
| 44 | fitting/attack | 7 | 6 | 1 | 0 | 0 |
| 44 | validation/bona fide | 2 | 0 | 0 | 0 | 2 |
| 44 | validation/attack | 2 | 1 | 1 | 0 | 0 |

## Score distributions per class (counts per bin of width 0.1)

### sessions 01-02: the 491 target frames

| seed | class | [0.0, 0.1) | [0.1, 0.2) | [0.2, 0.3) | [0.3, 0.4) | [0.4, 0.5) | [0.5, 0.6) | [0.6, 0.7) | [0.7, 0.8) | [0.8, 0.9) | [0.9, 1.0] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | bona fide | 242 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 42 | attack | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 248 |
| 43 | bona fide | 243 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 43 | attack | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 245 |
| 44 | bona fide | 242 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 44 | attack | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 248 |

### session 03: the 3,000 nuaa_cnn_test frames (N1 scores, read as stored)

| seed | class | [0.0, 0.1) | [0.1, 0.2) | [0.2, 0.3) | [0.3, 0.4) | [0.4, 0.5) | [0.5, 0.6) | [0.6, 0.7) | [0.7, 0.8) | [0.8, 0.9) | [0.9, 1.0] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | bona fide | 329 | 9 | 5 | 1 | 1 | 1 | 2 | 1 | 3 | 1148 |
| 42 | attack | 102 | 27 | 14 | 9 | 8 | 17 | 19 | 21 | 38 | 1245 |
| 43 | bona fide | 334 | 5 | 3 | 0 | 6 | 7 | 5 | 9 | 17 | 1114 |
| 43 | attack | 184 | 60 | 52 | 52 | 43 | 34 | 23 | 37 | 73 | 942 |
| 44 | bona fide | 353 | 10 | 4 | 1 | 1 | 2 | 1 | 3 | 4 | 1121 |
| 44 | attack | 39 | 10 | 9 | 9 | 5 | 15 | 17 | 39 | 54 | 1303 |

Detectors on the target frames: Haar 13, RetinaFace 478. Values are rounded to four decimals; aggregate.json holds the exact values.
