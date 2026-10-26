# N1 part 1: sensitivity of the M1 result to the test subjects (aggregate, no subject identifiers)

Run `20261024-N1-subjects-a5fcbbc` (commit `a5fcbbc`, dirty tree: no), 2026-10-24T20:55:03+0200. 300 NUAA test images, 16 subjects; existing C1 and M1 scores only, verified by SHA-256; no training, no inference, no new threshold.

Decision rule: attack if score_attack > threshold (tie = bona fide). Thresholds: 0.5 and the four frozen M1 thresholds (unchanged). ΔACER = ACER_M1 − ACER_0.5. Leave-one-subject-out: one subject excluded from the computation only, 16 values per CNN. Sign change: sign of the leave-one-out ΔACER (+, 0, −) different from the full one. A class absent gives NA, never 0.

Full-set values reproduced exactly from the accepted M1 table (56 cells of the main block and 56 of the block without the dev subjects).

| analyzer | M1 threshold | n bona fide | n attack | FA@0.5 | FR@0.5 | ACER@0.5 | FA@M1 | FR@M1 | ACER@M1 | ΔACER full | LOSO ΔACER min | LOSO ΔACER max | sign changes | of which opposite sign | zero | NA | n bona fide after exclusion | n attack after exclusion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| livenessnet__nuaa | 0.181055 | 150 | 150 | 24 | 109 | 0.443333 | 5 | 118 | 0.410000 | -0.033333 | -0.051750 | -0.009568 | 0/16 | 0 | 0 | 0 | 128–150 | 138–150 |
| attacknet_v1__nuaa | 0.329237 | 150 | 150 | 42 | 40 | 0.273333 | 37 | 44 | 0.270000 | -0.003333 | -0.010165 | 0.002542 | 2/16 | 2 | 0 | 0 | 128–150 | 138–150 |
| attacknet_v2_1__nuaa | 0.870329 | 150 | 150 | 3 | 115 | 0.393333 | 11 | 91 | 0.340000 | -0.053333 | -0.067033 | -0.019845 | 0/16 | 0 | 0 | 0 | 128–150 | 138–150 |
| attacknet_v2_2__nuaa | 0.696933 | 150 | 150 | 22 | 118 | 0.466667 | 38 | 112 | 0.500000 | 0.033333 | 0.019910 | 0.048762 | 0/16 | 0 | 0 | 0 | 128–150 | 138–150 |

Class composition of the subjects (same for every CNN): 8 with both classes, 7 with attacks only, 1 with bona fide only.

Limits: The test images include persons already present in training (NUAA is not subject-disjoint between its official train and test splits): the result does not show generalisation. 16 subjects with unbalanced class composition: the leave-one-out values are not independent and are not a confidence interval.
This is a sensitivity analysis of existing results, not cross-validation and not a new independent validation.
