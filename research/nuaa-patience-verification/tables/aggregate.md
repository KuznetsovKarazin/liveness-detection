# NUAA session-03 ablation: LivenessNet, arms B / A* / B', 4 subject folds x 3 seeds (aggregate, no identifiers)

Run `20261109-SESSABLRUN-6a50265` (commit `6a50265`, dirty tree: no), 2026-11-09T19:06:18+0100. Outcome: **PASS**, 36 of 36 trainings complete. Configuration `ca039df5d311` (patience12_verification: training.early_stopping_patience 4 -> 12, every other field equal to the parent configuration `4c2d29d13ded`); cells run 20261104-SESSABL-5d89888, fold schema 20261103-FOLDS-67b7f3d, frames of 20261103-FOLDSRUN-e174812. Every number below is reported whatever its value (pre-commitment). Diagnostic, not causal; no statistical test.

Arms (same fitting subjects S12 per fold): B = sessions 01-02 (350/350/350/350 per cell, 1400 frames); A* = sessions 01-03 (234/233/233/234/233/233, 1400); B' = sessions 01-02 at the per-cell counts of A* (234/233/233/234, 934). Common dev: sessions 01-02 of the dev-fold subjects (400 frames), for early stopping, the learning-rate schedule, the restored epoch and the EER threshold. Test: every frame of the held-out subjects (the fold schema, unchanged).

Decision: attack if score_attack > threshold (tie = bona fide). `05` = the fixed threshold 0.5; `dev` = the EER threshold of the common dev (M1 criterion), frozen before any test scoring. AUROC: threshold-free (Mann-Whitney, ties 0.5). NA where a class is absent. Differences are minuend - subtrahend, paired on the same test frames and seeds (nominal pairing: batch order and augmentation differ with the data).

## Reading rule (fixed in the configuration before the run)

For each comparison, stratum and metric, the pooled difference (minuend - subtrahend) is 'consistent' only if it has the same non-zero sign for all three seeds AND the paired bootstrap interval of every seed excludes 0 on the side of that sign; otherwise 'inconclusive' (also when a seed has no pooled value, no difference or no interval); a consistent verdict states the direction (for example 'consistent: A* higher AUROC than B'); primary: AUROC and ACER at the dev threshold in stratum 03 of a_star_minus_b; the same rule, labelled secondary, for the other strata, the other metrics and the two secondary comparisons; no statistical test; applied mechanically by the script; every verdict reported.

## Primary verdicts (A* - B, stratum 03)

| metric | seed 42: difference [bootstrap] | seed 43: difference [bootstrap] | seed 44: difference [bootstrap] | verdict | reason |
|---|---|---|---|---|---|
| AUROC | 0.2186 [0.0643, 0.4011] | 0.3971 [0.2309, 0.5873] | 0.1689 [-0.0214, 0.3888] | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| ACER (dev threshold) | -0.2007 [-0.3589, 0.0024] | -0.2772 [-0.3922, -0.1407] | -0.0811 [-0.2545, 0.0999] | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |


## Comparison A* - B (primary): pooled over the 4 folds, per seed

| seed | stratum | subjects | APCER (0.5): difference [bootstrap] | BPCER (0.5): difference [bootstrap] | ACER (0.5): difference [bootstrap] | APCER (dev threshold): difference [bootstrap] | BPCER (dev threshold): difference [bootstrap] | ACER (dev threshold): difference [bootstrap] | AUROC: difference [bootstrap] |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 01 | 9 | -0.0468 [-0.1439, 0.0024] | -0.0495 [-0.1589, 0.0000] | -0.0481 [-0.1275, 0.0002] | 0.0035 [0.0012, 0.0069] | -0.1159 [-0.3867, 0.0026] | -0.0562 [-0.1912, 0.0028] | 0.0038 [-0.0034, 0.0284] |
| 42 | 02 | 9 | 0.0000 [0.0000, 0.0000] | 0.1663 [0.0021, 0.4468] | 0.0831 [0.0010, 0.2234] | 0.0022 [0.0000, 0.0056] | -0.2693 [-0.5413, -0.0299] | -0.1335 [-0.2703, -0.0132] | -0.0065 [-0.0224, 0.0009] |
| 42 | 03 | 16 | -0.0687 [-0.1988, 0.0461] | -0.4075 [-0.7199, -0.1017] | -0.2381 [-0.4128, -0.0827] | 0.0418 [-0.1820, 0.2894] | -0.4432 [-0.7716, -0.0018] | -0.2007 [-0.3589, 0.0024] | 0.2186 [0.0643, 0.4011] |
| 42 | all | 16 | -0.0581 [-0.1675, 0.0378] | -0.2492 [-0.4683, -0.0061] | -0.1536 [-0.2562, -0.0469] | 0.0328 [-0.1360, 0.2354] | -0.3571 [-0.6457, -0.0392] | -0.1622 [-0.2728, -0.0228] | 0.1105 [0.0272, 0.2113] |
| 43 | 01 | 9 | -0.0795 [-0.2397, 0.0000] | 0.0506 [0.0000, 0.1689] | -0.0145 [-0.1172, 0.0669] | 0.0012 [0.0000, 0.0035] | 0.0450 [0.0000, 0.1501] | 0.0231 [0.0000, 0.0751] | 0.0030 [-0.0042, 0.0212] |
| 43 | 02 | 9 | 0.0011 [0.0000, 0.0034] | 0.0012 [0.0000, 0.0035] | 0.0011 [0.0000, 0.0027] | 0.0011 [0.0000, 0.0034] | -0.0035 [-0.0082, 0.0000] | -0.0012 [-0.0040, 0.0011] | 0.0011 [-0.0000, 0.0037] |
| 43 | 03 | 16 | -0.0108 [-0.1749, 0.1687] | -0.6811 [-0.8673, -0.4171] | -0.3460 [-0.4688, -0.2047] | 0.1462 [-0.0041, 0.3339] | -0.7005 [-0.8955, -0.4193] | -0.2772 [-0.3922, -0.1407] | 0.3971 [0.2309, 0.5873] |
| 43 | all | 16 | -0.0172 [-0.1465, 0.1375] | -0.4396 [-0.6254, -0.2317] | -0.2284 [-0.3231, -0.1151] | 0.1124 [-0.0026, 0.2774] | -0.4541 [-0.6489, -0.2334] | -0.1708 [-0.2526, -0.0683] | 0.2143 [0.1072, 0.3233] |
| 44 | 01 | 9 | -0.0082 [-0.0421, 0.0127] | -0.1159 [-0.3835, 0.0000] | -0.0620 [-0.2020, 0.0041] | -0.0304 [-0.1604, 0.0539] | -0.0090 [-0.0271, 0.0000] | -0.0197 [-0.0831, 0.0233] | 0.0050 [-0.0176, 0.0653] |
| 44 | 02 | 9 | 0.0011 [0.0000, 0.0034] | -0.0269 [-0.1078, 0.0304] | -0.0129 [-0.0533, 0.0162] | 0.0672 [0.0000, 0.1556] | 0.0597 [-0.0024, 0.1694] | 0.0635 [0.0134, 0.1213] | -0.0113 [-0.0219, -0.0005] |
| 44 | 03 | 16 | -0.0194 [-0.1087, 0.0736] | -0.4170 [-0.6879, -0.1383] | -0.2182 [-0.3593, -0.0779] | 0.1210 [-0.0429, 0.3243] | -0.2832 [-0.6541, 0.1139] | -0.0811 [-0.2545, 0.0999] | 0.1689 [-0.0214, 0.3888] |
| 44 | all | 16 | -0.0157 [-0.0841, 0.0560] | -0.2993 [-0.5039, -0.1189] | -0.1575 [-0.2608, -0.0635] | 0.0973 [-0.0320, 0.2752] | -0.1781 [-0.4431, 0.0763] | -0.0404 [-0.1622, 0.0782] | 0.0925 [-0.0118, 0.2426] |


Verdicts (A* - B; primary only for AUROC and ACER dev in stratum 03 of A* - B)

| stratum | metric | label | verdict | reason |
|---|---|---|---|---|
| 01 | APCER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 01 | BPCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 01 | ACER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 01 | APCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 01 | BPCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 01 | ACER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 01 | AUROC | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 02 | APCER (0.5) | secondary | inconclusive | a seed has a zero difference |
| 02 | BPCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 02 | ACER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 02 | APCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 02 | BPCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 02 | ACER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 02 | AUROC | secondary | inconclusive | the sign differs across the seeds |
| 03 | APCER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | BPCER (0.5) | secondary | consistent: A* lower BPCER (0.5) than B | same sign for every seed and every interval excludes 0 |
| 03 | ACER (0.5) | secondary | consistent: A* lower ACER (0.5) than B | same sign for every seed and every interval excludes 0 |
| 03 | APCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | BPCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | ACER (dev threshold) | primary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | AUROC | primary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | APCER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | BPCER (0.5) | secondary | consistent: A* lower BPCER (0.5) than B | same sign for every seed and every interval excludes 0 |
| all | ACER (0.5) | secondary | consistent: A* lower ACER (0.5) than B | same sign for every seed and every interval excludes 0 |
| all | APCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | BPCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | ACER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | AUROC | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |


Median and range over the seeds of the pooled differences

| stratum | APCER (0.5) | BPCER (0.5) | ACER (0.5) | APCER (dev threshold) | BPCER (dev threshold) | ACER (dev threshold) | AUROC |
|---|---|---|---|---|---|---|---|
| 01 | -0.0468 (-0.0795 to -0.0082) | -0.0495 (-0.1159 to 0.0506) | -0.0481 (-0.0620 to -0.0145) | 0.0012 (-0.0304 to 0.0035) | -0.0090 (-0.1159 to 0.0450) | -0.0197 (-0.0562 to 0.0231) | 0.0038 (0.0030 to 0.0050) |
| 02 | 0.0011 (0.0000 to 0.0011) | 0.0012 (-0.0269 to 0.1663) | 0.0011 (-0.0129 to 0.0831) | 0.0022 (0.0011 to 0.0672) | -0.0035 (-0.2693 to 0.0597) | -0.0012 (-0.1335 to 0.0635) | -0.0065 (-0.0113 to 0.0011) |
| 03 | -0.0194 (-0.0687 to -0.0108) | -0.4170 (-0.6811 to -0.4075) | -0.2381 (-0.3460 to -0.2182) | 0.1210 (0.0418 to 0.1462) | -0.4432 (-0.7005 to -0.2832) | -0.2007 (-0.2772 to -0.0811) | 0.2186 (0.1689 to 0.3971) |
| all | -0.0172 (-0.0581 to -0.0157) | -0.2993 (-0.4396 to -0.2492) | -0.1575 (-0.2284 to -0.1536) | 0.0973 (0.0328 to 0.1124) | -0.3571 (-0.4541 to -0.1781) | -0.1622 (-0.1708 to -0.0404) | 0.1105 (0.0925 to 0.2143) |


Per fold and seed (A* - B)

| fold | seed | stratum | APCER (0.5) | BPCER (0.5) | ACER (0.5) | APCER (dev threshold) | BPCER (dev threshold) | ACER (dev threshold) | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 42 | 01 | 0.0000 | -0.1654 | -0.0827 | 0.0035 | -0.4055 | -0.2010 | 0.0000 |
| 1 | 42 | 02 | 0.0000 | 0.0082 | 0.0041 | 0.0000 | -0.8197 | -0.4098 | -0.0000 |
| 1 | 42 | 03 | -0.0081 | -0.1881 | -0.0981 | 0.1883 | -0.7254 | -0.2685 | 0.0780 |
| 1 | 42 | all | -0.0055 | -0.1504 | -0.0779 | 0.1281 | -0.6845 | -0.2782 | 0.0389 |
| 1 | 43 | 01 | 0.0000 | 0.1772 | 0.0886 | 0.0000 | 0.1575 | 0.0787 | -0.0021 |
| 1 | 43 | 02 | 0.0000 | 0.0041 | 0.0020 | 0.0000 | -0.0082 | -0.0041 | 0.0000 |
| 1 | 43 | 03 | -0.0057 | -0.8811 | -0.4434 | -0.0032 | -0.8843 | -0.4438 | 0.2097 |
| 1 | 43 | all | -0.0038 | -0.5404 | -0.2721 | -0.0022 | -0.5481 | -0.2752 | 0.0973 |
| 1 | 44 | 01 | 0.0035 | -0.4016 | -0.1990 | 0.0000 | -0.0276 | -0.0138 | -0.0035 |
| 1 | 44 | 02 | 0.0000 | -0.1352 | -0.0676 | 0.0000 | 0.1926 | 0.0963 | -0.0245 |
| 1 | 44 | 03 | 0.0633 | -0.1481 | -0.0424 | -0.1071 | 0.3492 | 0.1210 | -0.1349 |
| 1 | 44 | all | 0.0434 | -0.1911 | -0.0739 | -0.0726 | 0.2551 | 0.0913 | -0.0877 |
| 2 | 42 | 01 | 0.0000 | -0.0043 | -0.0021 | 0.0000 | 0.0043 | 0.0021 | 0.0000 |
| 2 | 42 | 02 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0085 | 0.0043 | 0.0061 |
| 2 | 42 | 03 | -0.0565 | -0.4289 | -0.2427 | -0.5117 | 0.0903 | -0.2107 | 0.2264 |
| 2 | 42 | all | -0.0442 | -0.3197 | -0.1819 | -0.4002 | 0.0686 | -0.1658 | 0.1333 |
| 2 | 43 | 01 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | 43 | 02 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0085 | -0.0043 | 0.0073 |
| 2 | 43 | 03 | -0.0704 | -0.6968 | -0.3836 | -0.0132 | -0.6722 | -0.3427 | 0.4903 |
| 2 | 43 | all | -0.0550 | -0.5182 | -0.2866 | -0.0103 | -0.5007 | -0.2555 | 0.3075 |
| 2 | 44 | 01 | 0.0107 | -0.0043 | 0.0032 | 0.0267 | -0.0043 | 0.0112 | -0.0043 |
| 2 | 44 | 02 | 0.0052 | 0.0000 | 0.0026 | 0.0104 | -0.0085 | 0.0009 | -0.0001 |
| 2 | 44 | 03 | 0.0491 | -0.3543 | -0.1526 | 0.1686 | -0.3553 | -0.0933 | 0.0025 |
| 2 | 44 | all | 0.0401 | -0.2642 | -0.1120 | 0.1359 | -0.2657 | -0.0649 | 0.0007 |
| 3 | 42 | 01 | 0.0053 | -0.0036 | 0.0009 | 0.0053 | -0.0036 | 0.0009 | -0.0053 |
| 3 | 42 | 02 | 0.0000 | 0.0000 | 0.0000 | 0.0050 | 0.0000 | 0.0025 | 0.0000 |
| 3 | 42 | 03 | -0.0306 | -0.4197 | -0.2251 | 0.1111 | -0.5017 | -0.1953 | 0.3103 |
| 3 | 42 | all | -0.0244 | -0.2624 | -0.1434 | 0.0911 | -0.3136 | -0.1112 | 0.1582 |
| 3 | 43 | 01 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 43 | 02 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 43 | 03 | 0.1586 | -0.4370 | -0.1392 | 0.3369 | -0.4405 | -0.0518 | 0.3333 |
| 3 | 43 | all | 0.1287 | -0.2725 | -0.0719 | 0.2734 | -0.2747 | -0.0007 | 0.1757 |
| 3 | 44 | 01 | 0.0160 | 0.0000 | 0.0080 | 0.0909 | 0.0000 | 0.0455 | -0.0283 |
| 3 | 44 | 02 | 0.0000 | 0.0000 | 0.0000 | 0.2900 | 0.0000 | 0.1450 | -0.0250 |
| 3 | 44 | 03 | -0.0685 | -0.5029 | -0.2857 | 0.3880 | -0.5341 | -0.0731 | 0.4301 |
| 3 | 44 | all | -0.0541 | -0.3136 | -0.1839 | 0.3514 | -0.3331 | 0.0091 | 0.2379 |
| 4 | 42 | 01 | -0.2113 | 0.0000 | -0.1057 | 0.0052 | 0.0000 | 0.0026 | 0.0000 |
| 4 | 42 | 02 | 0.0000 | 0.5668 | 0.2834 | 0.0050 | -0.1255 | -0.0603 | 0.0000 |
| 4 | 42 | 03 | -0.1720 | -0.7161 | -0.4440 | 0.3480 | -0.8626 | -0.2573 | 0.4358 |
| 4 | 42 | all | -0.1579 | -0.2768 | -0.2173 | 0.2767 | -0.5492 | -0.1363 | 0.1769 |
| 4 | 43 | 01 | -0.3505 | 0.0000 | -0.1753 | 0.0052 | 0.0000 | 0.0026 | 0.0000 |
| 4 | 43 | 02 | 0.0050 | 0.0000 | 0.0025 | 0.0050 | 0.0000 | 0.0025 | 0.0000 |
| 4 | 43 | 03 | -0.1487 | -0.6998 | -0.4242 | 0.2020 | -0.8517 | -0.3249 | 0.4562 |
| 4 | 43 | all | -0.1531 | -0.4184 | -0.2857 | 0.1610 | -0.5092 | -0.1741 | 0.2184 |
| 4 | 44 | 01 | -0.0670 | 0.0000 | -0.0335 | -0.2474 | 0.0000 | -0.1237 | 0.0000 |
| 4 | 44 | 02 | 0.0000 | 0.0405 | 0.0202 | 0.0000 | 0.0202 | 0.0101 | -0.0119 |
| 4 | 44 | 03 | -0.0953 | -0.8481 | -0.4717 | -0.0313 | -0.8156 | -0.4234 | 0.4915 |
| 4 | 44 | all | -0.0824 | -0.4962 | -0.2893 | -0.0502 | -0.4822 | -0.2662 | 0.2270 |


## Comparison B' - B (secondary): pooled over the 4 folds, per seed

| seed | stratum | subjects | APCER (0.5): difference [bootstrap] | BPCER (0.5): difference [bootstrap] | ACER (0.5): difference [bootstrap] | APCER (dev threshold): difference [bootstrap] | BPCER (dev threshold): difference [bootstrap] | ACER (dev threshold): difference [bootstrap] | AUROC: difference [bootstrap] |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 01 | 9 | -0.0480 [-0.1445, 0.0000] | 0.1001 [-0.1077, 0.4089] | 0.0261 [-0.0956, 0.1776] | 0.0047 [0.0011, 0.0094] | -0.0675 [-0.3523, 0.1155] | -0.0314 [-0.1749, 0.0620] | -0.0310 [-0.1161, 0.0209] |
| 42 | 02 | 9 | 0.0022 [0.0000, 0.0068] | 0.0070 [0.0010, 0.0147] | 0.0046 [0.0011, 0.0081] | 0.0056 [0.0000, 0.0148] | -0.2705 [-0.5376, -0.0339] | -0.1324 [-0.2683, -0.0119] | -0.0012 [-0.0037, 0.0000] |
| 42 | 03 | 16 | -0.0665 [-0.1381, 0.0161] | -0.2139 [-0.5262, 0.0550] | -0.1402 [-0.3150, 0.0149] | 0.1193 [-0.0146, 0.2678] | -0.2870 [-0.5607, -0.0204] | -0.0839 [-0.2355, 0.0709] | 0.0706 [-0.0551, 0.2354] |
| 42 | all | 16 | -0.0562 [-0.1202, 0.0097] | -0.1222 [-0.3228, 0.0613] | -0.0892 [-0.2032, 0.0188] | 0.0927 [-0.0097, 0.2155] | -0.2460 [-0.4374, -0.0799] | -0.0767 [-0.1806, 0.0311] | 0.0211 [-0.0493, 0.1050] |
| 43 | 01 | 9 | 0.0246 [-0.0034, 0.0770] | 0.0562 [0.0000, 0.1822] | 0.0404 [0.0000, 0.1112] | 0.0959 [0.0000, 0.2867] | 0.0607 [-0.0018, 0.2008] | 0.0783 [0.0000, 0.1971] | -0.0778 [-0.2231, 0.0000] |
| 43 | 02 | 9 | 0.0011 [0.0000, 0.0034] | 0.1710 [0.0009, 0.4536] | 0.0860 [0.0011, 0.2273] | 0.0022 [0.0000, 0.0067] | 0.2681 [0.0020, 0.5878] | 0.1352 [0.0034, 0.2950] | -0.1230 [-0.3297, 0.0018] |
| 43 | 03 | 16 | -0.0424 [-0.1929, 0.0915] | 0.0291 [-0.0804, 0.1132] | -0.0066 [-0.0810, 0.0690] | 0.0437 [-0.0405, 0.1216] | 0.0187 [-0.0594, 0.0872] | 0.0312 [-0.0273, 0.0987] | -0.1084 [-0.2903, 0.0684] |
| 43 | all | 16 | -0.0296 [-0.1523, 0.0698] | 0.0576 [-0.0335, 0.1279] | 0.0140 [-0.0517, 0.0729] | 0.0447 [-0.0310, 0.1123] | 0.0678 [-0.0171, 0.1399] | 0.0563 [-0.0071, 0.1078] | -0.1423 [-0.2685, 0.0012] |
| 44 | 01 | 9 | -0.0164 [-0.0490, 0.0070] | -0.0697 [-0.2348, 0.0046] | -0.0431 [-0.1292, 0.0017] | 0.0491 [-0.0095, 0.1552] | 0.0439 [0.0000, 0.1080] | 0.0465 [0.0007, 0.1111] | -0.0029 [-0.0291, 0.0242] |
| 44 | 02 | 9 | -0.0056 [-0.0168, 0.0000] | -0.0796 [-0.2378, 0.0027] | -0.0426 [-0.1200, 0.0000] | -0.0011 [-0.0067, 0.0034] | 0.0023 [0.0000, 0.0055] | 0.0006 [-0.0011, 0.0024] | 0.0010 [0.0000, 0.0029] |
| 44 | 03 | 16 | -0.0285 [-0.1254, 0.0520] | 0.2531 [0.0623, 0.4296] | 0.1123 [0.0058, 0.2233] | 0.0616 [-0.0679, 0.2109] | 0.3679 [-0.0208, 0.6975] | 0.2148 [0.0420, 0.3807] | -0.2000 [-0.3374, -0.0653] |
| 44 | all | 16 | -0.0244 [-0.1062, 0.0373] | 0.1412 [0.0071, 0.2770] | 0.0584 [-0.0154, 0.1413] | 0.0527 [-0.0515, 0.1686] | 0.2503 [-0.0117, 0.4776] | 0.1515 [0.0360, 0.2572] | -0.1122 [-0.1922, -0.0353] |


Verdicts (B' - B; primary only for AUROC and ACER dev in stratum 03 of A* - B)

| stratum | metric | label | verdict | reason |
|---|---|---|---|---|
| 01 | APCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 01 | BPCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 01 | ACER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 01 | APCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 01 | BPCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 01 | ACER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 01 | AUROC | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 02 | APCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 02 | BPCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 02 | ACER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 02 | APCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 02 | BPCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 02 | ACER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 02 | AUROC | secondary | inconclusive | the sign differs across the seeds |
| 03 | APCER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | BPCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 03 | ACER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 03 | APCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | BPCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 03 | ACER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 03 | AUROC | secondary | inconclusive | the sign differs across the seeds |
| all | APCER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | BPCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| all | ACER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| all | APCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | BPCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| all | ACER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| all | AUROC | secondary | inconclusive | the sign differs across the seeds |


Median and range over the seeds of the pooled differences

| stratum | APCER (0.5) | BPCER (0.5) | ACER (0.5) | APCER (dev threshold) | BPCER (dev threshold) | ACER (dev threshold) | AUROC |
|---|---|---|---|---|---|---|---|
| 01 | -0.0164 (-0.0480 to 0.0246) | 0.0562 (-0.0697 to 0.1001) | 0.0261 (-0.0431 to 0.0404) | 0.0491 (0.0047 to 0.0959) | 0.0439 (-0.0675 to 0.0607) | 0.0465 (-0.0314 to 0.0783) | -0.0310 (-0.0778 to -0.0029) |
| 02 | 0.0011 (-0.0056 to 0.0022) | 0.0070 (-0.0796 to 0.1710) | 0.0046 (-0.0426 to 0.0860) | 0.0022 (-0.0011 to 0.0056) | 0.0023 (-0.2705 to 0.2681) | 0.0006 (-0.1324 to 0.1352) | -0.0012 (-0.1230 to 0.0010) |
| 03 | -0.0424 (-0.0665 to -0.0285) | 0.0291 (-0.2139 to 0.2531) | -0.0066 (-0.1402 to 0.1123) | 0.0616 (0.0437 to 0.1193) | 0.0187 (-0.2870 to 0.3679) | 0.0312 (-0.0839 to 0.2148) | -0.1084 (-0.2000 to 0.0706) |
| all | -0.0296 (-0.0562 to -0.0244) | 0.0576 (-0.1222 to 0.1412) | 0.0140 (-0.0892 to 0.0584) | 0.0527 (0.0447 to 0.0927) | 0.0678 (-0.2460 to 0.2503) | 0.0563 (-0.0767 to 0.1515) | -0.1122 (-0.1423 to 0.0211) |


Per fold and seed (B' - B)

| fold | seed | stratum | APCER (0.5) | BPCER (0.5) | ACER (0.5) | APCER (dev threshold) | BPCER (dev threshold) | ACER (dev threshold) | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 42 | 01 | 0.0000 | -0.1378 | -0.0689 | 0.0000 | -0.3780 | -0.1890 | 0.0000 |
| 1 | 42 | 02 | 0.0000 | 0.0123 | 0.0061 | 0.0000 | -0.8156 | -0.4078 | 0.0000 |
| 1 | 42 | 03 | 0.0203 | 0.1232 | 0.0718 | 0.0341 | -0.2768 | -0.1213 | -0.0392 |
| 1 | 42 | all | 0.0137 | 0.0576 | 0.0357 | 0.0231 | -0.3872 | -0.1821 | -0.0199 |
| 1 | 43 | 01 | 0.0000 | 0.1890 | 0.0945 | 0.0035 | 0.2087 | 0.1061 | -0.1739 |
| 1 | 43 | 02 | 0.0033 | 0.5861 | 0.2947 | 0.0067 | 0.4918 | 0.2492 | -0.2524 |
| 1 | 43 | 03 | 0.0057 | 0.0627 | 0.0342 | 0.0406 | 0.0714 | 0.0560 | -0.3051 |
| 1 | 43 | all | 0.0044 | 0.1750 | 0.0897 | 0.0291 | 0.1680 | 0.0985 | -0.3092 |
| 1 | 44 | 01 | 0.0000 | -0.2480 | -0.1240 | 0.0000 | 0.1024 | 0.0512 | -0.0230 |
| 1 | 44 | 02 | 0.0000 | -0.2828 | -0.1414 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1 | 44 | 03 | 0.0252 | 0.5297 | 0.2774 | -0.1144 | 0.9578 | 0.4217 | -0.2811 |
| 1 | 44 | all | 0.0170 | 0.2516 | 0.1343 | -0.0775 | 0.6409 | 0.2817 | -0.1410 |
| 2 | 42 | 01 | 0.0000 | 0.5299 | 0.2650 | 0.0107 | 0.1581 | 0.0844 | -0.0038 |
| 2 | 42 | 02 | 0.0104 | 0.0000 | 0.0052 | 0.0259 | -0.0085 | 0.0087 | 0.0009 |
| 2 | 42 | 03 | -0.0147 | -0.5152 | -0.2649 | 0.2419 | -0.2247 | 0.0086 | 0.2447 |
| 2 | 42 | all | -0.0103 | -0.2927 | -0.1515 | 0.1932 | -0.1409 | 0.0262 | 0.0913 |
| 2 | 43 | 01 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0043 | -0.0021 | 0.0000 |
| 2 | 43 | 02 | 0.0000 | -0.0085 | -0.0043 | 0.0000 | -0.0085 | -0.0043 | 0.0073 |
| 2 | 43 | 03 | 0.1752 | -0.1079 | 0.0336 | 0.0784 | -0.0707 | 0.0039 | -0.0798 |
| 2 | 43 | all | 0.1370 | -0.0810 | 0.0280 | 0.0614 | -0.0540 | 0.0037 | -0.0310 |
| 2 | 44 | 01 | 0.0160 | 0.0000 | 0.0080 | 0.0053 | 0.0513 | 0.0283 | -0.0083 |
| 2 | 44 | 02 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0085 | 0.0043 | 0.0000 |
| 2 | 44 | 03 | 0.1378 | 0.2395 | 0.1886 | -0.0095 | 0.4750 | 0.2327 | -0.3293 |
| 2 | 44 | all | 0.1095 | 0.1781 | 0.1438 | -0.0069 | 0.3628 | 0.1779 | -0.2259 |
| 3 | 42 | 01 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0036 | -0.0018 | 0.0000 |
| 3 | 42 | 02 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 42 | 03 | -0.1003 | -0.0023 | -0.0513 | -0.0721 | -0.0069 | -0.0395 | 0.0952 |
| 3 | 42 | all | -0.0814 | -0.0014 | -0.0414 | -0.0585 | -0.0050 | -0.0318 | 0.0465 |
| 3 | 43 | 01 | 0.0000 | 0.0072 | 0.0036 | 0.0000 | 0.0072 | 0.0036 | -0.0036 |
| 3 | 43 | 02 | 0.0000 | 0.0163 | 0.0081 | 0.0000 | 0.4472 | 0.2236 | 0.0000 |
| 3 | 43 | 03 | -0.0246 | 0.0763 | 0.0258 | 0.0480 | 0.0798 | 0.0639 | -0.1272 |
| 3 | 43 | all | -0.0200 | 0.0519 | 0.0160 | 0.0390 | 0.1305 | 0.0847 | -0.1248 |
| 3 | 44 | 01 | -0.0214 | 0.0036 | -0.0089 | -0.0214 | 0.0036 | -0.0089 | 0.0091 |
| 3 | 44 | 02 | -0.0250 | 0.0041 | -0.0105 | -0.0100 | 0.0041 | -0.0030 | 0.0000 |
| 3 | 44 | 03 | -0.1790 | 0.1306 | -0.0242 | -0.0973 | 0.0890 | -0.0041 | 0.0273 |
| 3 | 44 | all | -0.1496 | 0.0829 | -0.0333 | -0.0819 | 0.0570 | -0.0125 | 0.0152 |
| 4 | 42 | 01 | -0.2113 | 0.0000 | -0.1057 | 0.0103 | 0.0000 | 0.0052 | 0.0000 |
| 4 | 42 | 02 | 0.0000 | 0.0121 | 0.0061 | 0.0000 | -0.1255 | -0.0628 | 0.0000 |
| 4 | 42 | 03 | -0.1473 | -0.5533 | -0.3503 | 0.2900 | -0.8571 | -0.2836 | 0.2501 |
| 4 | 42 | all | -0.1383 | -0.3276 | -0.2329 | 0.2307 | -0.5459 | -0.1576 | 0.1215 |
| 4 | 43 | 01 | 0.1082 | 0.0000 | 0.0541 | 0.4175 | 0.0000 | 0.2088 | 0.0000 |
| 4 | 43 | 02 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4 | 43 | 03 | -0.2993 | 0.1519 | -0.0737 | 0.0100 | 0.0000 | 0.0050 | -0.2757 |
| 4 | 43 | all | -0.2260 | 0.0908 | -0.0676 | 0.0507 | 0.0000 | 0.0253 | -0.2217 |
| 4 | 44 | 01 | -0.0670 | 0.0000 | -0.0335 | 0.2320 | 0.0000 | 0.1160 | 0.0000 |
| 4 | 44 | 02 | 0.0000 | 0.0000 | 0.0000 | 0.0050 | 0.0000 | 0.0025 | 0.0000 |
| 4 | 44 | 03 | -0.0567 | 0.0072 | -0.0247 | 0.4473 | -0.3797 | 0.0338 | -0.2367 |
| 4 | 44 | all | -0.0517 | 0.0043 | -0.0237 | 0.3786 | -0.2270 | 0.0758 | -0.1142 |


## Comparison A* - B' (secondary): pooled over the 4 folds, per seed

| seed | stratum | subjects | APCER (0.5): difference [bootstrap] | BPCER (0.5): difference [bootstrap] | ACER (0.5): difference [bootstrap] | APCER (dev threshold): difference [bootstrap] | BPCER (dev threshold): difference [bootstrap] | ACER (dev threshold): difference [bootstrap] | AUROC: difference [bootstrap] |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 01 | 9 | 0.0012 [0.0000, 0.0035] | -0.1496 [-0.4353, -0.0023] | -0.0742 [-0.2177, -0.0008] | -0.0012 [-0.0082, 0.0048] | -0.0484 [-0.1371, 0.0000] | -0.0248 [-0.0706, 0.0006] | 0.0348 [-0.0026, 0.1209] |
| 42 | 02 | 9 | -0.0022 [-0.0068, 0.0000] | 0.1593 [-0.0023, 0.4346] | 0.0785 [-0.0030, 0.2160] | -0.0034 [-0.0137, 0.0044] | 0.0012 [-0.0033, 0.0070] | -0.0011 [-0.0042, 0.0018] | -0.0053 [-0.0216, 0.0035] |
| 42 | 03 | 16 | -0.0023 [-0.1428, 0.1474] | -0.1936 [-0.4593, 0.0195] | -0.0979 [-0.2418, 0.0153] | -0.0774 [-0.3415, 0.1962] | -0.1562 [-0.5471, 0.1925] | -0.1168 [-0.3018, 0.0414] | 0.1479 [0.0112, 0.3214] |
| 42 | all | 16 | -0.0019 [-0.1060, 0.1156] | -0.1269 [-0.3092, 0.0374] | -0.0644 [-0.1487, 0.0102] | -0.0599 [-0.2593, 0.1572] | -0.1111 [-0.3557, 0.1228] | -0.0855 [-0.1973, 0.0290] | 0.0894 [0.0161, 0.1904] |
| 43 | 01 | 9 | -0.1041 [-0.3162, 0.0034] | -0.0056 [-0.0133, 0.0000] | -0.0549 [-0.1618, 0.0001] | -0.0947 [-0.2857, 0.0024] | -0.0157 [-0.0506, 0.0018] | -0.0552 [-0.1539, 0.0006] | 0.0808 [0.0000, 0.2327] |
| 43 | 02 | 9 | 0.0000 [-0.0033, 0.0034] | -0.1698 [-0.4524, -0.0009] | -0.0849 [-0.2267, -0.0005] | -0.0011 [-0.0067, 0.0034] | -0.2717 [-0.5884, -0.0098] | -0.1364 [-0.2948, -0.0056] | 0.1241 [0.0003, 0.3306] |
| 43 | 03 | 16 | 0.0316 [-0.2168, 0.2985] | -0.7103 [-0.9114, -0.4444] | -0.3393 [-0.4873, -0.1937] | 0.1024 [-0.1145, 0.3563] | -0.7192 [-0.9147, -0.4594] | -0.3084 [-0.4414, -0.1717] | 0.5054 [0.3160, 0.6699] |
| 43 | all | 16 | 0.0124 [-0.1814, 0.2494] | -0.4972 [-0.6960, -0.2708] | -0.2424 [-0.3445, -0.1328] | 0.0677 [-0.1044, 0.2976] | -0.5218 [-0.7194, -0.3015] | -0.2271 [-0.3220, -0.1195] | 0.3566 [0.1821, 0.4990] |
| 44 | 01 | 9 | 0.0082 [-0.0046, 0.0251] | -0.0461 [-0.1476, 0.0000] | -0.0190 [-0.0731, 0.0100] | -0.0795 [-0.3110, 0.0556] | -0.0529 [-0.1341, -0.0016] | -0.0662 [-0.1919, 0.0162] | 0.0079 [-0.0239, 0.0465] |
| 44 | 02 | 9 | 0.0067 [0.0000, 0.0178] | 0.0527 [-0.0012, 0.1380] | 0.0297 [0.0027, 0.0705] | 0.0683 [-0.0011, 0.1557] | 0.0574 [-0.0065, 0.1694] | 0.0628 [0.0124, 0.1223] | -0.0123 [-0.0227, -0.0013] |
| 44 | 03 | 16 | 0.0090 [-0.1031, 0.1167] | -0.6701 [-0.8678, -0.4123] | -0.3306 [-0.4375, -0.1995] | 0.0594 [-0.1817, 0.3242] | -0.6511 [-0.8561, -0.4103] | -0.2959 [-0.4416, -0.1447] | 0.3689 [0.1673, 0.5579] |
| 44 | all | 16 | 0.0087 [-0.0745, 0.0986] | -0.4405 [-0.6443, -0.2212] | -0.2159 [-0.3088, -0.1084] | 0.0446 [-0.1520, 0.2715] | -0.4284 [-0.6239, -0.2120] | -0.1919 [-0.3050, -0.0682] | 0.2046 [0.0755, 0.3557] |


Verdicts (A* - B'; primary only for AUROC and ACER dev in stratum 03 of A* - B)

| stratum | metric | label | verdict | reason |
|---|---|---|---|---|
| 01 | APCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 01 | BPCER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 01 | ACER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 01 | APCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 01 | BPCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 01 | ACER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 01 | AUROC | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 02 | APCER (0.5) | secondary | inconclusive | a seed has a zero difference |
| 02 | BPCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 02 | ACER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 02 | APCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 02 | BPCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 02 | ACER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 02 | AUROC | secondary | inconclusive | the sign differs across the seeds |
| 03 | APCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 03 | BPCER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | ACER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | APCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 03 | BPCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | ACER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | AUROC | secondary | consistent: A* higher AUROC than B' | same sign for every seed and every interval excludes 0 |
| all | APCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| all | BPCER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | ACER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | APCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| all | BPCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | ACER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | AUROC | secondary | consistent: A* higher AUROC than B' | same sign for every seed and every interval excludes 0 |


Median and range over the seeds of the pooled differences

| stratum | APCER (0.5) | BPCER (0.5) | ACER (0.5) | APCER (dev threshold) | BPCER (dev threshold) | ACER (dev threshold) | AUROC |
|---|---|---|---|---|---|---|---|
| 01 | 0.0012 (-0.1041 to 0.0082) | -0.0461 (-0.1496 to -0.0056) | -0.0549 (-0.0742 to -0.0190) | -0.0795 (-0.0947 to -0.0012) | -0.0484 (-0.0529 to -0.0157) | -0.0552 (-0.0662 to -0.0248) | 0.0348 (0.0079 to 0.0808) |
| 02 | 0.0000 (-0.0022 to 0.0067) | 0.0527 (-0.1698 to 0.1593) | 0.0297 (-0.0849 to 0.0785) | -0.0011 (-0.0034 to 0.0683) | 0.0012 (-0.2717 to 0.0574) | -0.0011 (-0.1364 to 0.0628) | -0.0053 (-0.0123 to 0.1241) |
| 03 | 0.0090 (-0.0023 to 0.0316) | -0.6701 (-0.7103 to -0.1936) | -0.3306 (-0.3393 to -0.0979) | 0.0594 (-0.0774 to 0.1024) | -0.6511 (-0.7192 to -0.1562) | -0.2959 (-0.3084 to -0.1168) | 0.3689 (0.1479 to 0.5054) |
| all | 0.0087 (-0.0019 to 0.0124) | -0.4405 (-0.4972 to -0.1269) | -0.2159 (-0.2424 to -0.0644) | 0.0446 (-0.0599 to 0.0677) | -0.4284 (-0.5218 to -0.1111) | -0.1919 (-0.2271 to -0.0855) | 0.2046 (0.0894 to 0.3566) |


Per fold and seed (A* - B')

| fold | seed | stratum | APCER (0.5) | BPCER (0.5) | ACER (0.5) | APCER (dev threshold) | BPCER (dev threshold) | ACER (dev threshold) | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 42 | 01 | 0.0000 | -0.0276 | -0.0138 | 0.0035 | -0.0276 | -0.0120 | 0.0000 |
| 1 | 42 | 02 | 0.0000 | -0.0041 | -0.0020 | 0.0000 | -0.0041 | -0.0020 | -0.0000 |
| 1 | 42 | 03 | -0.0284 | -0.3114 | -0.1699 | 0.1542 | -0.4486 | -0.1472 | 0.1172 |
| 1 | 42 | all | -0.0192 | -0.2080 | -0.1136 | 0.1050 | -0.2973 | -0.0961 | 0.0588 |
| 1 | 43 | 01 | 0.0000 | -0.0118 | -0.0059 | -0.0035 | -0.0512 | -0.0273 | 0.1718 |
| 1 | 43 | 02 | -0.0033 | -0.5820 | -0.2927 | -0.0067 | -0.5000 | -0.2533 | 0.2524 |
| 1 | 43 | 03 | -0.0114 | -0.9438 | -0.4776 | -0.0438 | -0.9557 | -0.4998 | 0.5148 |
| 1 | 43 | all | -0.0082 | -0.7154 | -0.3618 | -0.0313 | -0.7161 | -0.3737 | 0.4064 |
| 1 | 44 | 01 | 0.0035 | -0.1535 | -0.0750 | 0.0000 | -0.1299 | -0.0650 | 0.0195 |
| 1 | 44 | 02 | 0.0000 | 0.1475 | 0.0738 | 0.0000 | 0.1926 | 0.0963 | -0.0245 |
| 1 | 44 | 03 | 0.0381 | -0.6778 | -0.3198 | 0.0073 | -0.6086 | -0.3007 | 0.1462 |
| 1 | 44 | all | 0.0264 | -0.4427 | -0.2082 | 0.0049 | -0.3858 | -0.1904 | 0.0534 |
| 2 | 42 | 01 | 0.0000 | -0.5342 | -0.2671 | -0.0107 | -0.1538 | -0.0823 | 0.0038 |
| 2 | 42 | 02 | -0.0104 | 0.0000 | -0.0052 | -0.0259 | 0.0171 | -0.0044 | 0.0052 |
| 2 | 42 | 03 | -0.0418 | 0.0864 | 0.0223 | -0.7537 | 0.3150 | -0.2193 | -0.0183 |
| 2 | 42 | all | -0.0338 | -0.0270 | -0.0304 | -0.5935 | 0.2095 | -0.1920 | 0.0420 |
| 2 | 43 | 01 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0043 | 0.0021 | 0.0000 |
| 2 | 43 | 02 | 0.0000 | 0.0085 | 0.0043 | 0.0000 | 0.0000 | 0.0000 | -0.0000 |
| 2 | 43 | 03 | -0.2456 | -0.5888 | -0.4172 | -0.0916 | -0.6016 | -0.3466 | 0.5701 |
| 2 | 43 | all | -0.1921 | -0.4372 | -0.3147 | -0.0717 | -0.4467 | -0.2592 | 0.3385 |
| 2 | 44 | 01 | -0.0053 | -0.0043 | -0.0048 | 0.0214 | -0.0556 | -0.0171 | 0.0040 |
| 2 | 44 | 02 | 0.0052 | 0.0000 | 0.0026 | 0.0104 | -0.0171 | -0.0034 | -0.0001 |
| 2 | 44 | 03 | -0.0887 | -0.5937 | -0.3412 | 0.1782 | -0.8302 | -0.3260 | 0.3318 |
| 2 | 44 | all | -0.0694 | -0.4423 | -0.2559 | 0.1428 | -0.6285 | -0.2428 | 0.2266 |
| 3 | 42 | 01 | 0.0053 | -0.0036 | 0.0009 | 0.0053 | 0.0000 | 0.0027 | -0.0053 |
| 3 | 42 | 02 | 0.0000 | 0.0000 | 0.0000 | 0.0050 | 0.0000 | 0.0025 | 0.0000 |
| 3 | 42 | 03 | 0.0697 | -0.4173 | -0.1738 | 0.1832 | -0.4948 | -0.1558 | 0.2151 |
| 3 | 42 | all | 0.0570 | -0.2610 | -0.1020 | 0.1496 | -0.3086 | -0.0795 | 0.1117 |
| 3 | 43 | 01 | 0.0000 | -0.0072 | -0.0036 | 0.0000 | -0.0072 | -0.0036 | 0.0036 |
| 3 | 43 | 02 | 0.0000 | -0.0163 | -0.0081 | 0.0000 | -0.4472 | -0.2236 | 0.0000 |
| 3 | 43 | 03 | 0.1832 | -0.5133 | -0.1651 | 0.2889 | -0.5202 | -0.1157 | 0.4604 |
| 3 | 43 | all | 0.1486 | -0.3244 | -0.0879 | 0.2344 | -0.4052 | -0.0854 | 0.3006 |
| 3 | 44 | 01 | 0.0374 | -0.0036 | 0.0169 | 0.1123 | -0.0036 | 0.0543 | -0.0374 |
| 3 | 44 | 02 | 0.0250 | -0.0041 | 0.0105 | 0.3000 | -0.0041 | 0.1480 | -0.0250 |
| 3 | 44 | 03 | 0.1105 | -0.6335 | -0.2615 | 0.4853 | -0.6231 | -0.0689 | 0.4028 |
| 3 | 44 | all | 0.0955 | -0.3965 | -0.1505 | 0.4332 | -0.3901 | 0.0216 | 0.2227 |
| 4 | 42 | 01 | 0.0000 | 0.0000 | 0.0000 | -0.0052 | 0.0000 | -0.0026 | 0.0000 |
| 4 | 42 | 02 | 0.0000 | 0.5547 | 0.2773 | 0.0050 | 0.0000 | 0.0025 | 0.0000 |
| 4 | 42 | 03 | -0.0247 | -0.1627 | -0.0937 | 0.0580 | -0.0054 | 0.0263 | 0.1857 |
| 4 | 42 | all | -0.0195 | 0.0508 | 0.0156 | 0.0459 | -0.0032 | 0.0213 | 0.0554 |
| 4 | 43 | 01 | -0.4588 | 0.0000 | -0.2294 | -0.4124 | 0.0000 | -0.2062 | 0.0000 |
| 4 | 43 | 02 | 0.0050 | 0.0000 | 0.0025 | 0.0050 | 0.0000 | 0.0025 | 0.0000 |
| 4 | 43 | 03 | 0.1507 | -0.8517 | -0.3505 | 0.1920 | -0.8517 | -0.3299 | 0.7319 |
| 4 | 43 | all | 0.0729 | -0.5092 | -0.2182 | 0.1103 | -0.5092 | -0.1994 | 0.4400 |
| 4 | 44 | 01 | 0.0000 | 0.0000 | 0.0000 | -0.4794 | 0.0000 | -0.2397 | 0.0000 |
| 4 | 44 | 02 | 0.0000 | 0.0405 | 0.0202 | -0.0050 | 0.0202 | 0.0076 | -0.0119 |
| 4 | 44 | 03 | -0.0387 | -0.8553 | -0.4470 | -0.4787 | -0.4358 | -0.4572 | 0.7282 |
| 4 | 44 | all | -0.0306 | -0.5005 | -0.2656 | -0.4287 | -0.2551 | -0.3419 | 0.3411 |


## Per arm: pooled over the folds, per seed (bootstrap over subjects)

| arm | seed | stratum | n bona fide | n attack | subjects | BPCER 05 | APCER 05 | ACER 05 [bootstrap] | BPCER dev | APCER dev | ACER dev [bootstrap] | AUROC [bootstrap] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B | 42 | 01 | 889 | 855 | 9 | 0.0551 | 0.0480 | 0.0515 [0.0008, 0.1351] | 0.1170 | 0.0000 | 0.0585 [0.0000, 0.1933] | 0.9950 [0.9716, 1.0000] |
| B | 42 | 02 | 854 | 893 | 9 | 0.0023 | 0.0000 | 0.0012 [0.0000, 0.0027] | 0.2728 | 0.0000 | 0.1364 [0.0169, 0.2713] | 0.9998 [0.9987, 1.0000] |
| B | 42 | 03 | 3362 | 5761 | 16 | 0.5199 | 0.2029 | 0.3614 [0.1801, 0.5290] | 0.5577 | 0.1916 | 0.3747 [0.2204, 0.5101] | 0.7314 [0.5483, 0.9004] |
| B | 42 | all | 5105 | 7509 | 16 | 0.3524 | 0.1611 | 0.2568 [0.1265, 0.3780] | 0.4333 | 0.1470 | 0.2902 [0.1686, 0.4148] | 0.8517 [0.7290, 0.9467] |
| B | 43 | 01 | 889 | 855 | 9 | 0.0517 | 0.0807 | 0.0662 [0.0000, 0.1649] | 0.0450 | 0.0000 | 0.0225 [0.0000, 0.0733] | 0.9895 [0.9460, 1.0000] |
| B | 43 | 02 | 854 | 893 | 9 | 0.0035 | 0.0000 | 0.0018 [0.0000, 0.0041] | 0.0047 | 0.0000 | 0.0023 [0.0000, 0.0053] | 0.9989 [0.9963, 1.0000] |
| B | 43 | 03 | 3362 | 5761 | 16 | 0.7582 | 0.2465 | 0.5023 [0.3398, 0.6277] | 0.7948 | 0.0868 | 0.4408 [0.2772, 0.5476] | 0.5185 [0.3296, 0.7155] |
| B | 43 | all | 5105 | 7509 | 16 | 0.5089 | 0.1983 | 0.3536 [0.2162, 0.4836] | 0.5320 | 0.0666 | 0.2993 [0.1649, 0.4158] | 0.7314 [0.5871, 0.8645] |
| B | 44 | 01 | 889 | 855 | 9 | 0.1170 | 0.0199 | 0.0684 [0.0024, 0.2069] | 0.0180 | 0.0608 | 0.0394 [0.0024, 0.0984] | 0.9846 [0.9307, 1.0000] |
| B | 44 | 02 | 854 | 893 | 9 | 0.0831 | 0.0056 | 0.0444 [0.0014, 0.1223] | 0.0023 | 0.0022 | 0.0023 [0.0000, 0.0045] | 0.9990 [0.9971, 1.0000] |
| B | 44 | 03 | 3362 | 5761 | 16 | 0.5315 | 0.1462 | 0.3388 [0.1953, 0.4846] | 0.4030 | 0.1755 | 0.2893 [0.1297, 0.4502] | 0.7646 [0.5618, 0.9343] |
| B | 44 | all | 5105 | 7509 | 16 | 0.3843 | 0.1151 | 0.2497 [0.1380, 0.3821] | 0.2690 | 0.1418 | 0.2054 [0.0908, 0.3437] | 0.8602 [0.7103, 0.9618] |
| A* | 42 | 01 | 889 | 855 | 9 | 0.0056 | 0.0012 | 0.0034 [0.0000, 0.0097] | 0.0011 | 0.0035 | 0.0023 [0.0006, 0.0039] | 0.9988 [0.9965, 1.0000] |
| A* | 42 | 02 | 854 | 893 | 9 | 0.1686 | 0.0000 | 0.0843 [0.0026, 0.2245] | 0.0035 | 0.0022 | 0.0029 [0.0006, 0.0053] | 0.9933 [0.9774, 1.0000] |
| A* | 42 | 03 | 3362 | 5761 | 16 | 0.1124 | 0.1342 | 0.1233 [0.0575, 0.1911] | 0.1145 | 0.2335 | 0.1740 [0.0791, 0.2746] | 0.9500 [0.8937, 0.9894] |
| A* | 42 | all | 5105 | 7509 | 16 | 0.1032 | 0.1031 | 0.1032 [0.0515, 0.1647] | 0.0762 | 0.1798 | 0.1280 [0.0547, 0.2181] | 0.9622 [0.9176, 0.9892] |
| A* | 43 | 01 | 889 | 855 | 9 | 0.1024 | 0.0012 | 0.0518 [0.0000, 0.1707] | 0.0900 | 0.0012 | 0.0456 [0.0005, 0.1485] | 0.9925 [0.9645, 1.0000] |
| A* | 43 | 02 | 854 | 893 | 9 | 0.0047 | 0.0011 | 0.0029 [0.0005, 0.0064] | 0.0012 | 0.0011 | 0.0011 [0.0000, 0.0027] | 1.0000 [0.9999, 1.0000] |
| A* | 43 | 03 | 3362 | 5761 | 16 | 0.0770 | 0.2357 | 0.1564 [0.0688, 0.2623] | 0.0943 | 0.2329 | 0.1636 [0.0705, 0.2729] | 0.9155 [0.8333, 0.9745] |
| A* | 43 | all | 5105 | 7509 | 16 | 0.0693 | 0.1811 | 0.1252 [0.0557, 0.2251] | 0.0780 | 0.1790 | 0.1285 [0.0558, 0.2289] | 0.9457 [0.8770, 0.9859] |
| A* | 44 | 01 | 889 | 855 | 9 | 0.0011 | 0.0117 | 0.0064 [0.0012, 0.0140] | 0.0090 | 0.0304 | 0.0197 [0.0041, 0.0362] | 0.9896 [0.9740, 0.9992] |
| A* | 44 | 02 | 854 | 893 | 9 | 0.0562 | 0.0067 | 0.0315 [0.0045, 0.0728] | 0.0621 | 0.0694 | 0.0657 [0.0154, 0.1245] | 0.9877 [0.9773, 0.9987] |
| A* | 44 | 03 | 3362 | 5761 | 16 | 0.1145 | 0.1267 | 0.1206 [0.0535, 0.2033] | 0.1199 | 0.2965 | 0.2082 [0.1116, 0.3139] | 0.9335 [0.8355, 0.9895] |
| A* | 44 | all | 5105 | 7509 | 16 | 0.0850 | 0.0993 | 0.0922 [0.0444, 0.1502] | 0.0909 | 0.2392 | 0.1650 [0.0927, 0.2566] | 0.9526 [0.8914, 0.9889] |
| B' | 42 | 01 | 889 | 855 | 9 | 0.1552 | 0.0000 | 0.0776 [0.0011, 0.2177] | 0.0495 | 0.0047 | 0.0271 [0.0012, 0.0734] | 0.9640 [0.8787, 1.0000] |
| B' | 42 | 02 | 854 | 893 | 9 | 0.0094 | 0.0022 | 0.0058 [0.0017, 0.0101] | 0.0023 | 0.0056 | 0.0040 [0.0006, 0.0084] | 0.9985 [0.9955, 1.0000] |
| B' | 42 | 03 | 3362 | 5761 | 16 | 0.3061 | 0.1364 | 0.2213 [0.0810, 0.3934] | 0.2707 | 0.3109 | 0.2908 [0.1450, 0.4551] | 0.8020 [0.6041, 0.9632] |
| B' | 42 | all | 5105 | 7509 | 16 | 0.2302 | 0.1049 | 0.1676 [0.0724, 0.2823] | 0.1873 | 0.2397 | 0.2135 [0.1087, 0.3291] | 0.8728 [0.7530, 0.9637] |
| B' | 43 | 01 | 889 | 855 | 9 | 0.1080 | 0.1053 | 0.1066 [0.0009, 0.2834] | 0.1057 | 0.0959 | 0.1008 [0.0009, 0.2657] | 0.9117 [0.7342, 1.0000] |
| B' | 43 | 02 | 854 | 893 | 9 | 0.1745 | 0.0011 | 0.0878 [0.0021, 0.2294] | 0.2728 | 0.0022 | 0.1375 [0.0066, 0.2953] | 0.8759 [0.6694, 0.9996] |
| B' | 43 | 03 | 3362 | 5761 | 16 | 0.7873 | 0.2041 | 0.4957 [0.3698, 0.6005] | 0.8135 | 0.1305 | 0.4720 [0.3537, 0.5580] | 0.4101 [0.2368, 0.6175] |
| B' | 43 | all | 5105 | 7509 | 16 | 0.5665 | 0.1687 | 0.3676 [0.2485, 0.4706] | 0.5998 | 0.1113 | 0.3556 [0.2349, 0.4570] | 0.5891 [0.4036, 0.7904] |
| B' | 44 | 01 | 889 | 855 | 9 | 0.0472 | 0.0035 | 0.0254 [0.0008, 0.0771] | 0.0619 | 0.1099 | 0.0859 [0.0048, 0.2100] | 0.9817 [0.9476, 0.9998] |
| B' | 44 | 02 | 854 | 893 | 9 | 0.0035 | 0.0000 | 0.0018 [0.0000, 0.0033] | 0.0047 | 0.0011 | 0.0029 [0.0010, 0.0053] | 1.0000 [1.0000, 1.0000] |
| B' | 44 | 03 | 3362 | 5761 | 16 | 0.7847 | 0.1177 | 0.4512 [0.3135, 0.5588] | 0.7710 | 0.2371 | 0.5040 [0.3658, 0.6256] | 0.5646 [0.3819, 0.7772] |
| B' | 44 | all | 5105 | 7509 | 16 | 0.5256 | 0.0907 | 0.3081 [0.1918, 0.4183] | 0.5193 | 0.1946 | 0.3569 [0.2250, 0.4738] | 0.7480 [0.6079, 0.8911] |


## Per arm, fold and seed (held-out subjects, by session stratum)

| arm | fold | seed | stratum | n bona fide | n attack | subjects | BPCER 05 | APCER 05 | ACER 05 | BPCER dev | APCER dev | ACER dev | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B | 1 | 42 | 01 | 254 | 287 | 3 | 0.1850 | 0.0000 | 0.0925 | 0.4055 | 0.0000 | 0.2028 | 1.0000 |
| B | 1 | 42 | 02 | 244 | 300 | 3 | 0.0041 | 0.0000 | 0.0020 | 0.8238 | 0.0000 | 0.4119 | 1.0000 |
| B | 1 | 42 | 03 | 925 | 1232 | 4 | 0.4173 | 0.0276 | 0.2224 | 0.7935 | 0.0154 | 0.4045 | 0.8867 |
| B | 1 | 42 | all | 1423 | 1819 | 4 | 0.3050 | 0.0187 | 0.1618 | 0.7294 | 0.0104 | 0.3699 | 0.9439 |
| B | 1 | 43 | 01 | 254 | 287 | 3 | 0.1811 | 0.0000 | 0.0906 | 0.1535 | 0.0000 | 0.0768 | 0.9921 |
| B | 1 | 43 | 02 | 244 | 300 | 3 | 0.0082 | 0.0000 | 0.0041 | 0.0082 | 0.0000 | 0.0041 | 1.0000 |
| B | 1 | 43 | 03 | 925 | 1232 | 4 | 0.9373 | 0.0057 | 0.4715 | 0.9286 | 0.0073 | 0.4680 | 0.7812 |
| B | 1 | 43 | all | 1423 | 1819 | 4 | 0.6430 | 0.0038 | 0.3234 | 0.6325 | 0.0049 | 0.3187 | 0.8948 |
| B | 1 | 44 | 01 | 254 | 287 | 3 | 0.4055 | 0.0000 | 0.2028 | 0.0591 | 0.0000 | 0.0295 | 0.9999 |
| B | 1 | 44 | 02 | 244 | 300 | 3 | 0.2869 | 0.0000 | 0.1434 | 0.0041 | 0.0000 | 0.0020 | 1.0000 |
| B | 1 | 44 | 03 | 925 | 1232 | 4 | 0.4703 | 0.0089 | 0.2396 | 0.0422 | 0.1485 | 0.0954 | 0.9751 |
| B | 1 | 44 | all | 1423 | 1819 | 4 | 0.4273 | 0.0060 | 0.2167 | 0.0387 | 0.1006 | 0.0696 | 0.9840 |
| B | 2 | 42 | 01 | 234 | 187 | 2 | 0.0043 | 0.0000 | 0.0021 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B | 2 | 42 | 02 | 117 | 193 | 2 | 0.0085 | 0.0000 | 0.0043 | 0.0085 | 0.0000 | 0.0043 | 0.9938 |
| B | 2 | 42 | 03 | 1019 | 1364 | 4 | 0.5211 | 0.2075 | 0.3643 | 0.2247 | 0.5169 | 0.3708 | 0.7238 |
| B | 2 | 42 | all | 1370 | 1744 | 4 | 0.3891 | 0.1623 | 0.2757 | 0.1679 | 0.4042 | 0.2861 | 0.8368 |
| B | 2 | 43 | 01 | 234 | 187 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0043 | 0.0000 | 0.0021 | 1.0000 |
| B | 2 | 43 | 02 | 117 | 193 | 2 | 0.0085 | 0.0000 | 0.0043 | 0.0171 | 0.0000 | 0.0085 | 0.9927 |
| B | 2 | 43 | 03 | 1019 | 1364 | 4 | 0.8420 | 0.0880 | 0.4650 | 0.8852 | 0.0139 | 0.4496 | 0.4744 |
| B | 2 | 43 | all | 1370 | 1744 | 4 | 0.6270 | 0.0688 | 0.3479 | 0.6606 | 0.0109 | 0.3357 | 0.6701 |
| B | 2 | 44 | 01 | 234 | 187 | 2 | 0.0043 | 0.0000 | 0.0021 | 0.0043 | 0.0000 | 0.0021 | 1.0000 |
| B | 2 | 44 | 02 | 117 | 193 | 2 | 0.0085 | 0.0000 | 0.0043 | 0.0085 | 0.0000 | 0.0043 | 1.0000 |
| B | 2 | 44 | 03 | 1019 | 1364 | 4 | 0.4357 | 0.0081 | 0.2219 | 0.3935 | 0.0095 | 0.2015 | 0.9779 |
| B | 2 | 44 | all | 1370 | 1744 | 4 | 0.3255 | 0.0063 | 0.1659 | 0.2942 | 0.0075 | 0.1508 | 0.9865 |
| B | 3 | 42 | 01 | 276 | 187 | 2 | 0.0036 | 0.0000 | 0.0018 | 0.0036 | 0.0000 | 0.0018 | 1.0000 |
| B | 3 | 42 | 02 | 246 | 200 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B | 3 | 42 | 03 | 865 | 1665 | 4 | 0.5029 | 0.1279 | 0.3154 | 0.5029 | 0.1081 | 0.3055 | 0.6672 |
| B | 3 | 42 | all | 1387 | 2052 | 4 | 0.3143 | 0.1038 | 0.2091 | 0.3143 | 0.0877 | 0.2010 | 0.8298 |
| B | 3 | 43 | 01 | 276 | 187 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B | 3 | 43 | 02 | 246 | 200 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B | 3 | 43 | 03 | 865 | 1665 | 4 | 0.4983 | 0.2468 | 0.3726 | 0.5017 | 0.0721 | 0.2869 | 0.5781 |
| B | 3 | 43 | all | 1387 | 2052 | 4 | 0.3107 | 0.2003 | 0.2555 | 0.3129 | 0.0585 | 0.1857 | 0.7776 |
| B | 3 | 44 | 01 | 276 | 187 | 2 | 0.0000 | 0.0214 | 0.0107 | 0.0000 | 0.0214 | 0.0107 | 0.9909 |
| B | 3 | 44 | 02 | 246 | 200 | 2 | 0.0000 | 0.0250 | 0.0125 | 0.0000 | 0.0100 | 0.0050 | 1.0000 |
| B | 3 | 44 | 03 | 865 | 1665 | 4 | 0.5029 | 0.1910 | 0.3469 | 0.5341 | 0.1111 | 0.3226 | 0.5634 |
| B | 3 | 44 | all | 1387 | 2052 | 4 | 0.3136 | 0.1594 | 0.2365 | 0.3331 | 0.0931 | 0.2131 | 0.7523 |
| B | 4 | 42 | 01 | 125 | 194 | 2 | 0.0000 | 0.2113 | 0.1057 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B | 4 | 42 | 02 | 247 | 200 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.1255 | 0.0000 | 0.0628 | 1.0000 |
| B | 4 | 42 | 03 | 553 | 1500 | 4 | 0.7161 | 0.4260 | 0.5710 | 0.8626 | 0.1333 | 0.4980 | 0.5357 |
| B | 4 | 42 | all | 925 | 1894 | 4 | 0.4281 | 0.3590 | 0.3936 | 0.5492 | 0.1056 | 0.3274 | 0.7502 |
| B | 4 | 43 | 01 | 125 | 194 | 2 | 0.0000 | 0.3557 | 0.1778 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B | 4 | 43 | 02 | 247 | 200 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B | 4 | 43 | 03 | 553 | 1500 | 4 | 0.7107 | 0.5880 | 0.6493 | 0.8626 | 0.2347 | 0.5486 | 0.4227 |
| B | 4 | 43 | all | 925 | 1894 | 4 | 0.4249 | 0.5021 | 0.4635 | 0.5157 | 0.1859 | 0.3508 | 0.7043 |
| B | 4 | 44 | 01 | 125 | 194 | 2 | 0.0000 | 0.0670 | 0.0335 | 0.0000 | 0.2474 | 0.1237 | 1.0000 |
| B | 4 | 44 | 02 | 247 | 200 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B | 4 | 44 | 03 | 553 | 1500 | 4 | 0.8553 | 0.3347 | 0.5950 | 0.8192 | 0.4200 | 0.6196 | 0.4882 |
| B | 4 | 44 | all | 925 | 1894 | 4 | 0.5114 | 0.2719 | 0.3916 | 0.4897 | 0.3580 | 0.4239 | 0.7329 |
| A* | 1 | 42 | 01 | 254 | 287 | 3 | 0.0197 | 0.0000 | 0.0098 | 0.0000 | 0.0035 | 0.0017 | 1.0000 |
| A* | 1 | 42 | 02 | 244 | 300 | 3 | 0.0123 | 0.0000 | 0.0061 | 0.0041 | 0.0000 | 0.0020 | 1.0000 |
| A* | 1 | 42 | 03 | 925 | 1232 | 4 | 0.2292 | 0.0195 | 0.1243 | 0.0681 | 0.2037 | 0.1359 | 0.9647 |
| A* | 1 | 42 | all | 1423 | 1819 | 4 | 0.1546 | 0.0132 | 0.0839 | 0.0450 | 0.1385 | 0.0918 | 0.9828 |
| A* | 1 | 43 | 01 | 254 | 287 | 3 | 0.3583 | 0.0000 | 0.1791 | 0.3110 | 0.0000 | 0.1555 | 0.9901 |
| A* | 1 | 43 | 02 | 244 | 300 | 3 | 0.0123 | 0.0000 | 0.0061 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| A* | 1 | 43 | 03 | 925 | 1232 | 4 | 0.0562 | 0.0000 | 0.0281 | 0.0443 | 0.0041 | 0.0242 | 0.9909 |
| A* | 1 | 43 | all | 1423 | 1819 | 4 | 0.1026 | 0.0000 | 0.0513 | 0.0843 | 0.0027 | 0.0435 | 0.9921 |
| A* | 1 | 44 | 01 | 254 | 287 | 3 | 0.0039 | 0.0035 | 0.0037 | 0.0315 | 0.0000 | 0.0157 | 0.9964 |
| A* | 1 | 44 | 02 | 244 | 300 | 3 | 0.1516 | 0.0000 | 0.0758 | 0.1967 | 0.0000 | 0.0984 | 0.9755 |
| A* | 1 | 44 | 03 | 925 | 1232 | 4 | 0.3222 | 0.0722 | 0.1972 | 0.3914 | 0.0414 | 0.2164 | 0.8401 |
| A* | 1 | 44 | all | 1423 | 1819 | 4 | 0.2361 | 0.0495 | 0.1428 | 0.2937 | 0.0280 | 0.1609 | 0.8963 |
| A* | 2 | 42 | 01 | 234 | 187 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0043 | 0.0000 | 0.0021 | 1.0000 |
| A* | 2 | 42 | 02 | 117 | 193 | 2 | 0.0085 | 0.0000 | 0.0043 | 0.0171 | 0.0000 | 0.0085 | 1.0000 |
| A* | 2 | 42 | 03 | 1019 | 1364 | 4 | 0.0922 | 0.1510 | 0.1216 | 0.3150 | 0.0051 | 0.1601 | 0.9502 |
| A* | 2 | 42 | all | 1370 | 1744 | 4 | 0.0693 | 0.1181 | 0.0937 | 0.2365 | 0.0040 | 0.1203 | 0.9701 |
| A* | 2 | 43 | 01 | 234 | 187 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0043 | 0.0000 | 0.0021 | 1.0000 |
| A* | 2 | 43 | 02 | 117 | 193 | 2 | 0.0085 | 0.0000 | 0.0043 | 0.0085 | 0.0000 | 0.0043 | 1.0000 |
| A* | 2 | 43 | 03 | 1019 | 1364 | 4 | 0.1452 | 0.0176 | 0.0814 | 0.2130 | 0.0007 | 0.1068 | 0.9647 |
| A* | 2 | 43 | all | 1370 | 1744 | 4 | 0.1088 | 0.0138 | 0.0613 | 0.1599 | 0.0006 | 0.0802 | 0.9776 |
| A* | 2 | 44 | 01 | 234 | 187 | 2 | 0.0000 | 0.0107 | 0.0053 | 0.0000 | 0.0267 | 0.0134 | 0.9957 |
| A* | 2 | 44 | 02 | 117 | 193 | 2 | 0.0085 | 0.0052 | 0.0069 | 0.0000 | 0.0104 | 0.0052 | 0.9999 |
| A* | 2 | 44 | 03 | 1019 | 1364 | 4 | 0.0815 | 0.0572 | 0.0693 | 0.0383 | 0.1782 | 0.1082 | 0.9804 |
| A* | 2 | 44 | all | 1370 | 1744 | 4 | 0.0613 | 0.0464 | 0.0539 | 0.0285 | 0.1433 | 0.0859 | 0.9872 |
| A* | 3 | 42 | 01 | 276 | 187 | 2 | 0.0000 | 0.0053 | 0.0027 | 0.0000 | 0.0053 | 0.0027 | 0.9947 |
| A* | 3 | 42 | 02 | 246 | 200 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0050 | 0.0025 | 1.0000 |
| A* | 3 | 42 | 03 | 865 | 1665 | 4 | 0.0832 | 0.0973 | 0.0903 | 0.0012 | 0.2192 | 0.1102 | 0.9775 |
| A* | 3 | 42 | all | 1387 | 2052 | 4 | 0.0519 | 0.0794 | 0.0657 | 0.0007 | 0.1788 | 0.0898 | 0.9879 |
| A* | 3 | 43 | 01 | 276 | 187 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| A* | 3 | 43 | 02 | 246 | 200 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| A* | 3 | 43 | 03 | 865 | 1665 | 4 | 0.0613 | 0.4054 | 0.2333 | 0.0613 | 0.4090 | 0.2351 | 0.9113 |
| A* | 3 | 43 | all | 1387 | 2052 | 4 | 0.0382 | 0.3289 | 0.1836 | 0.0382 | 0.3319 | 0.1850 | 0.9534 |
| A* | 3 | 44 | 01 | 276 | 187 | 2 | 0.0000 | 0.0374 | 0.0187 | 0.0000 | 0.1123 | 0.0561 | 0.9626 |
| A* | 3 | 44 | 02 | 246 | 200 | 2 | 0.0000 | 0.0250 | 0.0125 | 0.0000 | 0.3000 | 0.1500 | 0.9750 |
| A* | 3 | 44 | 03 | 865 | 1665 | 4 | 0.0000 | 0.1225 | 0.0613 | 0.0000 | 0.4991 | 0.2495 | 0.9935 |
| A* | 3 | 44 | all | 1387 | 2052 | 4 | 0.0000 | 0.1053 | 0.0526 | 0.0000 | 0.4444 | 0.2222 | 0.9902 |
| A* | 4 | 42 | 01 | 125 | 194 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0052 | 0.0026 | 1.0000 |
| A* | 4 | 42 | 02 | 247 | 200 | 2 | 0.5668 | 0.0000 | 0.2834 | 0.0000 | 0.0050 | 0.0025 | 1.0000 |
| A* | 4 | 42 | 03 | 553 | 1500 | 4 | 0.0000 | 0.2540 | 0.1270 | 0.0000 | 0.4813 | 0.2407 | 0.9714 |
| A* | 4 | 42 | all | 925 | 1894 | 4 | 0.1514 | 0.2012 | 0.1763 | 0.0000 | 0.3823 | 0.1911 | 0.9271 |
| A* | 4 | 43 | 01 | 125 | 194 | 2 | 0.0000 | 0.0052 | 0.0026 | 0.0000 | 0.0052 | 0.0026 | 1.0000 |
| A* | 4 | 43 | 02 | 247 | 200 | 2 | 0.0000 | 0.0050 | 0.0025 | 0.0000 | 0.0050 | 0.0025 | 1.0000 |
| A* | 4 | 43 | 03 | 553 | 1500 | 4 | 0.0108 | 0.4393 | 0.2251 | 0.0108 | 0.4367 | 0.2238 | 0.8788 |
| A* | 4 | 43 | all | 925 | 1894 | 4 | 0.0065 | 0.3490 | 0.1777 | 0.0065 | 0.3469 | 0.1767 | 0.9227 |
| A* | 4 | 44 | 01 | 125 | 194 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| A* | 4 | 44 | 02 | 247 | 200 | 2 | 0.0405 | 0.0000 | 0.0202 | 0.0202 | 0.0000 | 0.0101 | 0.9881 |
| A* | 4 | 44 | 03 | 553 | 1500 | 4 | 0.0072 | 0.2393 | 0.1233 | 0.0036 | 0.3887 | 0.1961 | 0.9797 |
| A* | 4 | 44 | all | 925 | 1894 | 4 | 0.0151 | 0.1895 | 0.1023 | 0.0076 | 0.3078 | 0.1577 | 0.9599 |
| B' | 1 | 42 | 01 | 254 | 287 | 3 | 0.0472 | 0.0000 | 0.0236 | 0.0276 | 0.0000 | 0.0138 | 1.0000 |
| B' | 1 | 42 | 02 | 244 | 300 | 3 | 0.0164 | 0.0000 | 0.0082 | 0.0082 | 0.0000 | 0.0041 | 1.0000 |
| B' | 1 | 42 | 03 | 925 | 1232 | 4 | 0.5405 | 0.0479 | 0.2942 | 0.5168 | 0.0495 | 0.2831 | 0.8475 |
| B' | 1 | 42 | all | 1423 | 1819 | 4 | 0.3626 | 0.0324 | 0.1975 | 0.3422 | 0.0335 | 0.1879 | 0.9240 |
| B' | 1 | 43 | 01 | 254 | 287 | 3 | 0.3701 | 0.0000 | 0.1850 | 0.3622 | 0.0035 | 0.1828 | 0.8183 |
| B' | 1 | 43 | 02 | 244 | 300 | 3 | 0.5943 | 0.0033 | 0.2988 | 0.5000 | 0.0067 | 0.2533 | 0.7476 |
| B' | 1 | 43 | 03 | 925 | 1232 | 4 | 1.0000 | 0.0114 | 0.5057 | 1.0000 | 0.0479 | 0.5239 | 0.4761 |
| B' | 1 | 43 | all | 1423 | 1819 | 4 | 0.8180 | 0.0082 | 0.4131 | 0.8004 | 0.0341 | 0.4173 | 0.5857 |
| B' | 1 | 44 | 01 | 254 | 287 | 3 | 0.1575 | 0.0000 | 0.0787 | 0.1614 | 0.0000 | 0.0807 | 0.9768 |
| B' | 1 | 44 | 02 | 244 | 300 | 3 | 0.0041 | 0.0000 | 0.0020 | 0.0041 | 0.0000 | 0.0020 | 1.0000 |
| B' | 1 | 44 | 03 | 925 | 1232 | 4 | 1.0000 | 0.0341 | 0.5170 | 1.0000 | 0.0341 | 0.5170 | 0.6939 |
| B' | 1 | 44 | all | 1423 | 1819 | 4 | 0.6788 | 0.0231 | 0.3510 | 0.6796 | 0.0231 | 0.3513 | 0.8430 |
| B' | 2 | 42 | 01 | 234 | 187 | 2 | 0.5342 | 0.0000 | 0.2671 | 0.1581 | 0.0107 | 0.0844 | 0.9962 |
| B' | 2 | 42 | 02 | 117 | 193 | 2 | 0.0085 | 0.0104 | 0.0095 | 0.0000 | 0.0259 | 0.0130 | 0.9947 |
| B' | 2 | 42 | 03 | 1019 | 1364 | 4 | 0.0059 | 0.1928 | 0.0994 | 0.0000 | 0.7588 | 0.3794 | 0.9685 |
| B' | 2 | 42 | all | 1370 | 1744 | 4 | 0.0964 | 0.1519 | 0.1241 | 0.0270 | 0.5975 | 0.3122 | 0.9281 |
| B' | 2 | 43 | 01 | 234 | 187 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B' | 2 | 43 | 02 | 117 | 193 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0085 | 0.0000 | 0.0043 | 1.0000 |
| B' | 2 | 43 | 03 | 1019 | 1364 | 4 | 0.7341 | 0.2632 | 0.4986 | 0.8145 | 0.0924 | 0.4534 | 0.3946 |
| B' | 2 | 43 | all | 1370 | 1744 | 4 | 0.5460 | 0.2058 | 0.3759 | 0.6066 | 0.0722 | 0.3394 | 0.6391 |
| B' | 2 | 44 | 01 | 234 | 187 | 2 | 0.0043 | 0.0160 | 0.0102 | 0.0556 | 0.0053 | 0.0305 | 0.9917 |
| B' | 2 | 44 | 02 | 117 | 193 | 2 | 0.0085 | 0.0000 | 0.0043 | 0.0171 | 0.0000 | 0.0085 | 1.0000 |
| B' | 2 | 44 | 03 | 1019 | 1364 | 4 | 0.6752 | 0.1459 | 0.4105 | 0.8685 | 0.0000 | 0.4342 | 0.6486 |
| B' | 2 | 44 | all | 1370 | 1744 | 4 | 0.5036 | 0.1158 | 0.3097 | 0.6569 | 0.0006 | 0.3288 | 0.7606 |
| B' | 3 | 42 | 01 | 276 | 187 | 2 | 0.0036 | 0.0000 | 0.0018 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B' | 3 | 42 | 02 | 246 | 200 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B' | 3 | 42 | 03 | 865 | 1665 | 4 | 0.5006 | 0.0276 | 0.2641 | 0.4960 | 0.0360 | 0.2660 | 0.7624 |
| B' | 3 | 42 | all | 1387 | 2052 | 4 | 0.3129 | 0.0224 | 0.1677 | 0.3093 | 0.0292 | 0.1693 | 0.8763 |
| B' | 3 | 43 | 01 | 276 | 187 | 2 | 0.0072 | 0.0000 | 0.0036 | 0.0072 | 0.0000 | 0.0036 | 0.9964 |
| B' | 3 | 43 | 02 | 246 | 200 | 2 | 0.0163 | 0.0000 | 0.0081 | 0.4472 | 0.0000 | 0.2236 | 1.0000 |
| B' | 3 | 43 | 03 | 865 | 1665 | 4 | 0.5746 | 0.2222 | 0.3984 | 0.5815 | 0.1201 | 0.3508 | 0.4509 |
| B' | 3 | 43 | all | 1387 | 2052 | 4 | 0.3627 | 0.1803 | 0.2715 | 0.4434 | 0.0975 | 0.2704 | 0.6528 |
| B' | 3 | 44 | 01 | 276 | 187 | 2 | 0.0036 | 0.0000 | 0.0018 | 0.0036 | 0.0000 | 0.0018 | 1.0000 |
| B' | 3 | 44 | 02 | 246 | 200 | 2 | 0.0041 | 0.0000 | 0.0020 | 0.0041 | 0.0000 | 0.0020 | 1.0000 |
| B' | 3 | 44 | 03 | 865 | 1665 | 4 | 0.6335 | 0.0120 | 0.3228 | 0.6231 | 0.0138 | 0.3185 | 0.5907 |
| B' | 3 | 44 | all | 1387 | 2052 | 4 | 0.3965 | 0.0097 | 0.2031 | 0.3901 | 0.0112 | 0.2006 | 0.7675 |
| B' | 4 | 42 | 01 | 125 | 194 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0103 | 0.0052 | 1.0000 |
| B' | 4 | 42 | 02 | 247 | 200 | 2 | 0.0121 | 0.0000 | 0.0061 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B' | 4 | 42 | 03 | 553 | 1500 | 4 | 0.1627 | 0.2787 | 0.2207 | 0.0054 | 0.4233 | 0.2144 | 0.7858 |
| B' | 4 | 42 | all | 925 | 1894 | 4 | 0.1005 | 0.2207 | 0.1606 | 0.0032 | 0.3363 | 0.1698 | 0.8717 |
| B' | 4 | 43 | 01 | 125 | 194 | 2 | 0.0000 | 0.4639 | 0.2320 | 0.0000 | 0.4175 | 0.2088 | 1.0000 |
| B' | 4 | 43 | 02 | 247 | 200 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B' | 4 | 43 | 03 | 553 | 1500 | 4 | 0.8626 | 0.2887 | 0.5756 | 0.8626 | 0.2447 | 0.5536 | 0.1470 |
| B' | 4 | 43 | all | 925 | 1894 | 4 | 0.5157 | 0.2761 | 0.3959 | 0.5157 | 0.2365 | 0.3761 | 0.4826 |
| B' | 4 | 44 | 01 | 125 | 194 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4794 | 0.2397 | 1.0000 |
| B' | 4 | 44 | 02 | 247 | 200 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0050 | 0.0025 | 1.0000 |
| B' | 4 | 44 | 03 | 553 | 1500 | 4 | 0.8626 | 0.2780 | 0.5703 | 0.4394 | 0.8673 | 0.6534 | 0.2515 |
| B' | 4 | 44 | all | 925 | 1894 | 4 | 0.5157 | 0.2202 | 0.3679 | 0.2627 | 0.7365 | 0.4996 | 0.6188 |


## Reference A (reference, different fitting frames and dev: indicative)

Run 20261103-FOLDSRUN-e174812 (aggregate.json `8ba8cd475966`): 350 frames per cell over six cells, all eight fitting subjects, dev with session 03; pooled values per seed copied unchanged; not compared with a bootstrap.

| seed | stratum | ACER 05 [bootstrap] | ACER dev [bootstrap] | AUROC [bootstrap] |
|---|---|---|---|---|
| 42 | 01 | 0.0613 [0.0006, 0.1997] | 0.0613 [0.0006, 0.1997] | 0.9901 [0.9582, 1.0000] |
| 42 | 02 | 0.0053 [0.0000, 0.0134] | 0.0141 [0.0010, 0.0338] | 0.9998 [0.9993, 1.0000] |
| 42 | 03 | 0.1094 [0.0251, 0.2128] | 0.0937 [0.0203, 0.1900] | 0.9377 [0.8618, 0.9960] |
| 42 | all | 0.0900 [0.0325, 0.1718] | 0.0796 [0.0284, 0.1503] | 0.9520 [0.8975, 0.9906] |
| 43 | 01 | 0.0064 [0.0006, 0.0162] | 0.0076 [0.0006, 0.0197] | 0.9997 [0.9990, 1.0000] |
| 43 | 02 | 0.0017 [0.0000, 0.0042] | 0.0034 [0.0000, 0.0086] | 0.9997 [0.9990, 1.0000] |
| 43 | 03 | 0.1515 [0.0458, 0.3012] | 0.1532 [0.0514, 0.2931] | 0.9527 [0.8579, 0.9972] |
| 43 | all | 0.1088 [0.0343, 0.2090] | 0.1118 [0.0399, 0.2072] | 0.9740 [0.9218, 0.9982] |
| 44 | 01 | 0.0641 [0.0009, 0.2054] | 0.0535 [0.0000, 0.1755] | 0.9887 [0.9285, 1.0000] |
| 44 | 02 | 0.0187 [0.0000, 0.0524] | 0.0018 [0.0000, 0.0042] | 0.9996 [0.9991, 1.0000] |
| 44 | 03 | 0.0975 [0.0170, 0.2290] | 0.1178 [0.0291, 0.2458] | 0.9664 [0.9051, 0.9989] |
| 44 | all | 0.0808 [0.0257, 0.1668] | 0.0925 [0.0328, 0.1862] | 0.9755 [0.9351, 0.9970] |


## Training cards and dev thresholds

| model | n fitting | n dev | epochs | restored epoch | restored by | min val_loss | seconds | dev threshold | dev BPCER at threshold | dev APCER at threshold |
|---|---|---|---|---|---|---|---|---|---|---|
| arm_a_star_fold1_seed42 | 1400 | 400 | 12 | 10 | end of epochs (set_weights) | 0.0104 | 534 | 0.895917 | 0.0000 | 0.0000 |
| arm_a_star_fold1_seed43 | 1400 | 400 | 12 | 9 | end of epochs (set_weights) | 0.0165 | 451 | 0.749943 | 0.0050 | 0.0050 |
| arm_a_star_fold1_seed44 | 1400 | 400 | 12 | 11 | end of epochs (set_weights) | 0.0345 | 400 | 0.264618 | 0.0100 | 0.0100 |
| arm_a_star_fold2_seed42 | 1400 | 400 | 12 | 11 | end of epochs (set_weights) | 0.0001 | 441 | 0.000383 | 0.0000 | 0.0000 |
| arm_a_star_fold2_seed43 | 1400 | 400 | 12 | 10 | end of epochs (set_weights) | 0.0004 | 567 | 0.038164 | 0.0000 | 0.0000 |
| arm_a_star_fold2_seed44 | 1400 | 400 | 12 | 12 | end of epochs (set_weights) | 0.2171 | 398 | 0.891529 | 0.0050 | 0.0050 |
| arm_a_star_fold3_seed42 | 1400 | 400 | 12 | 9 | end of epochs (set_weights) | 0.3282 | 468 | 0.908766 | 0.0100 | 0.0100 |
| arm_a_star_fold3_seed43 | 1400 | 400 | 12 | 12 | end of epochs (set_weights) | 0.0088 | 546 | 0.517414 | 0.0000 | 0.0000 |
| arm_a_star_fold3_seed44 | 1400 | 400 | 12 | 11 | end of epochs (set_weights) | 0.5239 | 440 | 0.998619 | 0.0150 | 0.0150 |
| arm_a_star_fold4_seed42 | 1400 | 400 | 12 | 10 | end of epochs (set_weights) | 0.8106 | 557 | 0.999374 | 0.0200 | 0.0200 |
| arm_a_star_fold4_seed43 | 1400 | 400 | 12 | 9 | end of epochs (set_weights) | 0.0272 | 436 | 0.477774 | 0.0050 | 0.0050 |
| arm_a_star_fold4_seed44 | 1400 | 400 | 12 | 12 | end of epochs (set_weights) | 0.3154 | 524 | 0.983724 | 0.0050 | 0.0050 |
| arm_b_fold1_seed42 | 1400 | 400 | 12 | 10 | end of epochs (set_weights) | 0.0378 | 568 | 0.000768 | 0.0050 | 0.0050 |
| arm_b_fold1_seed43 | 1400 | 400 | 12 | 11 | end of epochs (set_weights) | 0.0056 | 554 | 0.542399 | 0.0050 | 0.0050 |
| arm_b_fold1_seed44 | 1400 | 400 | 12 | 11 | end of epochs (set_weights) | 0.0224 | 403 | 0.998644 | 0.0050 | 0.0050 |
| arm_b_fold2_seed42 | 1400 | 400 | 12 | 9 | end of epochs (set_weights) | 0.2308 | 405 | 0.920426 | 0.0400 | 0.0400 |
| arm_b_fold2_seed43 | 1400 | 400 | 12 | 7 | end of epochs (set_weights) | 0.0011 | 563 | 0.046983 | 0.0000 | 0.0000 |
| arm_b_fold2_seed44 | 1400 | 400 | 12 | 10 | end of epochs (set_weights) | 0.0030 | 443 | 0.578237 | 0.0000 | 0.0000 |
| arm_b_fold3_seed42 | 1400 | 400 | 12 | 12 | end of epochs (set_weights) | 0.0168 | 396 | 0.381979 | 0.0000 | 0.0000 |
| arm_b_fold3_seed43 | 1400 | 400 | 12 | 10 | end of epochs (set_weights) | 0.0531 | 683 | 0.031980 | 0.0000 | 0.0000 |
| arm_b_fold3_seed44 | 1400 | 400 | 12 | 11 | end of epochs (set_weights) | 0.0119 | 493 | 0.203914 | 0.0000 | 0.0000 |
| arm_b_fold4_seed42 | 1400 | 400 | 12 | 11 | end of epochs (set_weights) | 0.0005 | 648 | 0.016334 | 0.0000 | 0.0000 |
| arm_b_fold4_seed43 | 1400 | 400 | 12 | 12 | end of epochs (set_weights) | 0.0007 | 503 | 0.012818 | 0.0000 | 0.0000 |
| arm_b_fold4_seed44 | 1400 | 400 | 12 | 11 | end of epochs (set_weights) | 0.0262 | 477 | 0.759310 | 0.0000 | 0.0000 |
| arm_b_prime_fold1_seed42 | 934 | 400 | 12 | 12 | end of epochs (set_weights) | 0.0033 | 394 | 0.676444 | 0.0000 | 0.0000 |
| arm_b_prime_fold1_seed43 | 934 | 400 | 12 | 4 | end of epochs (set_weights) | 0.8647 | 346 | 1.000000 | 0.0150 | 0.0150 |
| arm_b_prime_fold1_seed44 | 934 | 400 | 12 | 9 | end of epochs (set_weights) | 0.1895 | 380 | 0.464886 | 0.0050 | 0.0050 |
| arm_b_prime_fold2_seed42 | 934 | 400 | 12 | 11 | end of epochs (set_weights) | 0.8750 | 390 | 0.999752 | 0.0850 | 0.0850 |
| arm_b_prime_fold2_seed43 | 934 | 400 | 12 | 10 | end of epochs (set_weights) | 0.0042 | 596 | 0.143735 | 0.0000 | 0.0000 |
| arm_b_prime_fold2_seed44 | 934 | 400 | 12 | 12 | end of epochs (set_weights) | 0.0736 | 583 | 0.003290 | 0.0050 | 0.0050 |
| arm_b_prime_fold3_seed42 | 934 | 400 | 12 | 11 | end of epochs (set_weights) | 0.0516 | 429 | 0.640867 | 0.0050 | 0.0050 |
| arm_b_prime_fold3_seed43 | 934 | 400 | 12 | 9 | end of epochs (set_weights) | 0.0011 | 460 | 0.013271 | 0.0000 | 0.0000 |
| arm_b_prime_fold3_seed44 | 934 | 400 | 12 | 9 | end of epochs (set_weights) | 0.0341 | 367 | 0.561295 | 0.0000 | 0.0000 |
| arm_b_prime_fold4_seed42 | 934 | 400 | 12 | 2 | end of epochs (set_weights) | 0.1375 | 329 | 0.763160 | 0.0050 | 0.0050 |
| arm_b_prime_fold4_seed43 | 934 | 400 | 12 | 3 | end of epochs (set_weights) | 0.0142 | 367 | 0.322349 | 0.0000 | 0.0000 |
| arm_b_prime_fold4_seed44 | 934 | 400 | 12 | 11 | end of epochs (set_weights) | 0.0851 | 326 | 0.998282 | 0.0150 | 0.0150 |


## Steps, budget, environment and inputs

| step | seconds (wall clock) | seconds (monotonic) |
|---|---|---|
| check-inputs | 4 | 4 |
| train arm_b_fold1_seed42 | 570 | 570 |
| threshold arm_b_fold1_seed42 | 5 | 5 |
| score arm_b_fold1_seed42 | 21 | 21 |
| train arm_a_star_fold1_seed42 | 536 | 536 |
| threshold arm_a_star_fold1_seed42 | 6 | 6 |
| score arm_a_star_fold1_seed42 | 24 | 24 |
| train arm_b_fold1_seed43 | 556 | 556 |
| threshold arm_b_fold1_seed43 | 5 | 5 |
| score arm_b_fold1_seed43 | 19 | 19 |
| train arm_a_star_fold1_seed43 | 452 | 452 |
| threshold arm_a_star_fold1_seed43 | 4 | 4 |
| score arm_a_star_fold1_seed43 | 17 | 17 |
| train arm_b_fold1_seed44 | 404 | 404 |
| threshold arm_b_fold1_seed44 | 4 | 4 |
| score arm_b_fold1_seed44 | 17 | 17 |
| train arm_a_star_fold1_seed44 | 401 | 401 |
| threshold arm_a_star_fold1_seed44 | 4 | 4 |
| score arm_a_star_fold1_seed44 | 17 | 17 |
| train arm_b_fold2_seed42 | 406 | 406 |
| threshold arm_b_fold2_seed42 | 4 | 4 |
| score arm_b_fold2_seed42 | 18 | 18 |
| train arm_a_star_fold2_seed42 | 442 | 442 |
| threshold arm_a_star_fold2_seed42 | 5 | 5 |
| score arm_a_star_fold2_seed42 | 18 | 18 |
| train arm_b_fold2_seed43 | 564 | 564 |
| threshold arm_b_fold2_seed43 | 6 | 6 |
| score arm_b_fold2_seed43 | 24 | 24 |
| train arm_a_star_fold2_seed43 | 569 | 569 |
| threshold arm_a_star_fold2_seed43 | 6 | 6 |
| score arm_a_star_fold2_seed43 | 20 | 20 |
| train arm_b_fold2_seed44 | 444 | 444 |
| threshold arm_b_fold2_seed44 | 4 | 4 |
| score arm_b_fold2_seed44 | 16 | 16 |
| train arm_a_star_fold2_seed44 | 399 | 399 |
| threshold arm_a_star_fold2_seed44 | 4 | 4 |
| score arm_a_star_fold2_seed44 | 16 | 16 |
| train arm_b_fold3_seed42 | 397 | 397 |
| threshold arm_b_fold3_seed42 | 4 | 4 |
| score arm_b_fold3_seed42 | 20 | 20 |
| train arm_a_star_fold3_seed42 | 469 | 469 |
| threshold arm_a_star_fold3_seed42 | 9 | 9 |
| score arm_a_star_fold3_seed42 | 26 | 26 |
| train arm_b_fold3_seed43 | 686 | 686 |
| threshold arm_b_fold3_seed43 | 9 | 9 |
| score arm_b_fold3_seed43 | 32 | 32 |
| train arm_a_star_fold3_seed43 | 548 | 548 |
| threshold arm_a_star_fold3_seed43 | 4 | 4 |
| score arm_a_star_fold3_seed43 | 19 | 19 |
| train arm_b_fold3_seed44 | 494 | 494 |
| threshold arm_b_fold3_seed44 | 5 | 5 |
| score arm_b_fold3_seed44 | 19 | 19 |
| train arm_a_star_fold3_seed44 | 441 | 441 |
| threshold arm_a_star_fold3_seed44 | 6 | 6 |
| score arm_a_star_fold3_seed44 | 27 | 27 |
| train arm_b_fold4_seed42 | 650 | 650 |
| threshold arm_b_fold4_seed42 | 9 | 9 |
| score arm_b_fold4_seed42 | 27 | 27 |
| train arm_a_star_fold4_seed42 | 559 | 559 |
| threshold arm_a_star_fold4_seed42 | 6 | 6 |
| score arm_a_star_fold4_seed42 | 18 | 18 |
| train arm_b_fold4_seed43 | 504 | 504 |
| threshold arm_b_fold4_seed43 | 5 | 5 |
| score arm_b_fold4_seed43 | 17 | 17 |
| train arm_a_star_fold4_seed43 | 438 | 438 |
| threshold arm_a_star_fold4_seed43 | 4 | 4 |
| score arm_a_star_fold4_seed43 | 15 | 15 |
| train arm_b_fold4_seed44 | 478 | 478 |
| threshold arm_b_fold4_seed44 | 5 | 5 |
| score arm_b_fold4_seed44 | 17 | 17 |
| train arm_a_star_fold4_seed44 | 525 | 525 |
| threshold arm_a_star_fold4_seed44 | 7 | 7 |
| score arm_a_star_fold4_seed44 | 21 | 21 |
| train arm_b_prime_fold1_seed42 | 396 | 396 |
| threshold arm_b_prime_fold1_seed42 | 5 | 5 |
| score arm_b_prime_fold1_seed42 | 21 | 21 |
| train arm_b_prime_fold1_seed43 | 347 | 347 |
| threshold arm_b_prime_fold1_seed43 | 5 | 5 |
| score arm_b_prime_fold1_seed43 | 20 | 20 |
| train arm_b_prime_fold1_seed44 | 381 | 381 |
| threshold arm_b_prime_fold1_seed44 | 5 | 5 |
| score arm_b_prime_fold1_seed44 | 20 | 20 |
| train arm_b_prime_fold2_seed42 | 392 | 392 |
| threshold arm_b_prime_fold2_seed42 | 7 | 7 |
| score arm_b_prime_fold2_seed42 | 36 | 36 |
| train arm_b_prime_fold2_seed43 | 599 | 599 |
| threshold arm_b_prime_fold2_seed43 | 10 | 10 |
| score arm_b_prime_fold2_seed43 | 39 | 39 |
| train arm_b_prime_fold2_seed44 | 586 | 586 |
| threshold arm_b_prime_fold2_seed44 | 10 | 10 |
| score arm_b_prime_fold2_seed44 | 31 | 31 |
| train arm_b_prime_fold3_seed42 | 432 | 432 |
| threshold arm_b_prime_fold3_seed42 | 6 | 6 |
| score arm_b_prime_fold3_seed42 | 29 | 29 |
| train arm_b_prime_fold3_seed43 | 462 | 462 |
| threshold arm_b_prime_fold3_seed43 | 6 | 6 |
| score arm_b_prime_fold3_seed43 | 25 | 25 |
| train arm_b_prime_fold3_seed44 | 369 | 369 |
| threshold arm_b_prime_fold3_seed44 | 5 | 5 |
| score arm_b_prime_fold3_seed44 | 20 | 20 |
| train arm_b_prime_fold4_seed42 | 330 | 330 |
| threshold arm_b_prime_fold4_seed42 | 5 | 5 |
| score arm_b_prime_fold4_seed42 | 22 | 22 |
| train arm_b_prime_fold4_seed43 | 369 | 369 |
| threshold arm_b_prime_fold4_seed43 | 5 | 5 |
| score arm_b_prime_fold4_seed43 | 17 | 17 |
| train arm_b_prime_fold4_seed44 | 328 | 328 |
| threshold arm_b_prime_fold4_seed44 | 5 | 5 |
| score arm_b_prime_fold4_seed44 | 17 | 17 |
| aggregate | 2 | 2 |

Wall time 17941 s of the cap 21600 s (checked before every training, at the end of every epoch, before every threshold and every test scoring; the aggregation and the final writes run after the last model, also after the cap; cap already exceeded before them: no). Environment: {"keras": "2.15.0", "numpy": "1.26.4", "opencv": "4.11.0", "platform": "darwin", "python": "3.11.16", "tensorflow": "2.15.0", "tensorflow-metal": "1.1.0", "tensorflow_installed": "2.15.0"}.

Inputs (SHA-256): cells run restricted SHA256SUMS `0904ee72546160eadb493a94d8489a8fb21687006cd8fe1970b26007c567723c`, fold schema restricted SHA256SUMS `f3b30ae75a7167646544ef524225749bbded99e55d2249b9415eef2481e0d0f1`, cache `dc286dc371e61d638da1842c062a38c5368a9ec4da07ccbc36b7fff6f3177920`, extracted frames `c3fd9903589b7b074c46880870ba87924763f244846c744b657f9aaa3ebeee77`, architecture `a83c02b0788bd6f98262abf00f5fe970abed501f231edba348820482aea64d30`.

## Limits

Bootstrap: generator numpy.random.default_rng(0) (PCG64), restarted for every seed, comparison (or arm) and stratum; the held-out subjects of the stratum (pooled over the four folds) in ascending identifier order; at every resample integers(0, n, n) with n = the number of subjects; every metric computed on the pooled frames of the resampled subjects (through per-subject counts and pairwise Mann-Whitney counts, identical to recomputing it on the frames); the same subject indices for both arms of a comparison and for every metric; percentiles 2.5 and 97.5 with np.percentile, linear interpolation; a resample without a class does not count for the metrics that need it; no interval with fewer than 5 subjects. the session-03 bona fide cell of S12 comes from 1-2 subjects per fold; two dev cells rest on one subject; frames of one subject and session are correlated (n counts frames); three seeds give a range; the four folds are not independent (the dev of a fold is the test of the next one); the dev without session 03 calibrates early stopping and the EER threshold on the session 01-02 distribution for every arm; the pooled strata hold 9 (sessions 01, 02) or 16 (03, all) subjects; one dataset, one attack type (print)
