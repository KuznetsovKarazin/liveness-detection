# NUAA session-03 ablation: LivenessNet, arms B / A* / B', 4 subject folds x 3 seeds (aggregate, no identifiers)

Run `20261111-SESSABLRUN-8568ffa` (commit `8568ffa`, dirty tree: no), 2026-11-11T04:21:10+0100. Outcome: **PASS**, 36 of 36 trainings complete. Configuration `60b38e094da8` (constant_lr_control: training.scheduler ReduceLROnPlateau -> constant, every other field equal to the parent configuration `ca039df5d311`); cells run 20261104-SESSABL-5d89888, fold schema 20261103-FOLDS-67b7f3d, frames of 20261103-FOLDSRUN-e174812. Every number below is reported whatever its value (pre-commitment). Diagnostic, not causal; no statistical test.

Arms (same fitting subjects S12 per fold): B = sessions 01-02 (350/350/350/350 per cell, 1400 frames); A* = sessions 01-03 (234/233/233/234/233/233, 1400); B' = sessions 01-02 at the per-cell counts of A* (234/233/233/234, 934). Common dev: sessions 01-02 of the dev-fold subjects (400 frames), for early stopping, the restored epoch and the EER threshold (constant learning rate, no schedule). Test: every frame of the held-out subjects (the fold schema, unchanged).

Decision: attack if score_attack > threshold (tie = bona fide). `05` = the fixed threshold 0.5; `dev` = the EER threshold of the common dev (M1 criterion), frozen before any test scoring. AUROC: threshold-free (Mann-Whitney, ties 0.5). NA where a class is absent. Differences are minuend - subtrahend, paired on the same test frames and seeds (nominal pairing: batch order and augmentation differ with the data).

## Reading rule (fixed in the configuration before the run)

For each comparison, stratum and metric, the pooled difference (minuend - subtrahend) is 'consistent' only if it has the same non-zero sign for all three seeds AND the paired bootstrap interval of every seed excludes 0 on the side of that sign; otherwise 'inconclusive' (also when a seed has no pooled value, no difference or no interval); a consistent verdict states the direction (for example 'consistent: A* higher AUROC than B'); primary: AUROC and ACER at the dev threshold in stratum 03 of a_star_minus_b; the same rule, labelled secondary, for the other strata, the other metrics and the two secondary comparisons; no statistical test; applied mechanically by the script; every verdict reported.

## Primary verdicts (A* - B, stratum 03)

| metric | seed 42: difference [bootstrap] | seed 43: difference [bootstrap] | seed 44: difference [bootstrap] | verdict | reason |
|---|---|---|---|---|---|
| AUROC | 0.1286 [-0.0576, 0.3794] | 0.5694 [0.3222, 0.7760] | 0.1945 [-0.0458, 0.4088] | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| ACER (dev threshold) | -0.0343 [-0.1970, 0.1349] | -0.2425 [-0.3920, -0.0661] | -0.2293 [-0.3733, -0.0629] | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |


## Comparison A* - B (primary): pooled over the 4 folds, per seed

| seed | stratum | subjects | APCER (0.5): difference [bootstrap] | BPCER (0.5): difference [bootstrap] | ACER (0.5): difference [bootstrap] | APCER (dev threshold): difference [bootstrap] | BPCER (dev threshold): difference [bootstrap] | ACER (dev threshold): difference [bootstrap] | AUROC: difference [bootstrap] |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 01 | 9 | -0.1099 [-0.3314, 0.0000] | -0.0011 [-0.0038, 0.0000] | -0.0555 [-0.1661, 0.0000] | -0.0503 [-0.1570, 0.0047] | -0.0551 [-0.1814, 0.0000] | -0.0527 [-0.1410, 0.0007] | 0.0102 [-0.0003, 0.0586] |
| 42 | 02 | 9 | 0.0515 [0.0000, 0.1206] | 0.0082 [0.0035, 0.0129] | 0.0299 [0.0041, 0.0639] | 0.0403 [0.0000, 0.1209] | 0.0012 [-0.0027, 0.0051] | 0.0207 [-0.0008, 0.0616] | -0.0034 [-0.0075, -0.0000] |
| 42 | 03 | 16 | 0.0246 [-0.2031, 0.2473] | -0.1791 [-0.4976, 0.0875] | -0.0772 [-0.2176, 0.0615] | 0.1113 [-0.0810, 0.3120] | -0.1800 [-0.5677, 0.2369] | -0.0343 [-0.1970, 0.1349] | 0.1286 [-0.0576, 0.3794] |
| 42 | all | 16 | 0.0125 [-0.1626, 0.2137] | -0.1167 [-0.3388, 0.0594] | -0.0521 [-0.1453, 0.0507] | 0.0844 [-0.0677, 0.2594] | -0.1279 [-0.4073, 0.1437] | -0.0217 [-0.1258, 0.0921] | 0.0629 [-0.0421, 0.2032] |
| 43 | 01 | 9 | -0.0480 [-0.1474, 0.0034] | 0.0799 [0.0000, 0.2632] | 0.0160 [-0.0694, 0.1136] | 0.0129 [0.0000, 0.0360] | 0.0472 [0.0000, 0.1576] | 0.0301 [0.0006, 0.0812] | -0.0035 [-0.0125, 0.0000] |
| 43 | 02 | 9 | 0.0034 [-0.0023, 0.0111] | 0.0047 [0.0000, 0.0117] | 0.0040 [0.0005, 0.0084] | 0.0504 [0.0000, 0.1489] | 0.0012 [0.0000, 0.0037] | 0.0258 [0.0000, 0.0746] | -0.0001 [-0.0004, 0.0000] |
| 43 | 03 | 16 | 0.0050 [-0.1842, 0.2062] | -0.7064 [-0.9114, -0.4244] | -0.3507 [-0.4827, -0.2035] | 0.2475 [0.0631, 0.4818] | -0.7326 [-0.9469, -0.4361] | -0.2425 [-0.3920, -0.0661] | 0.5694 [0.3222, 0.7760] |
| 43 | all | 16 | -0.0012 [-0.1463, 0.1722] | -0.4505 [-0.6597, -0.2204] | -0.2259 [-0.3237, -0.1155] | 0.1974 [0.0476, 0.4118] | -0.4740 [-0.6842, -0.2358] | -0.1383 [-0.2496, -0.0000] | 0.3102 [0.1516, 0.4441] |
| 44 | 01 | 9 | -0.1088 [-0.3302, 0.0035] | 0.0844 [-0.0075, 0.2900] | -0.0122 [-0.1603, 0.1191] | -0.1099 [-0.3314, 0.0000] | 0.0427 [-0.0075, 0.1525] | -0.0336 [-0.1643, 0.0575] | 0.0070 [-0.0159, 0.0521] |
| 44 | 02 | 9 | 0.0000 [-0.0033, 0.0034] | 0.1077 [-0.0023, 0.3249] | 0.0539 [-0.0018, 0.1624] | 0.0011 [0.0000, 0.0034] | 0.0398 [-0.0162, 0.1339] | 0.0205 [-0.0076, 0.0676] | -0.0012 [-0.0036, 0.0001] |
| 44 | 03 | 16 | -0.2100 [-0.3486, -0.0612] | -0.3195 [-0.5671, -0.0847] | -0.2647 [-0.4036, -0.1175] | -0.0618 [-0.2076, 0.0791] | -0.3968 [-0.6788, -0.0885] | -0.2293 [-0.3733, -0.0629] | 0.1945 [-0.0458, 0.4088] |
| 44 | all | 16 | -0.1735 [-0.2928, -0.0507] | -0.1777 [-0.3973, 0.0208] | -0.1756 [-0.3034, -0.0502] | -0.0598 [-0.1829, 0.0633] | -0.2472 [-0.5097, -0.0251] | -0.1535 [-0.2796, -0.0318] | 0.0933 [-0.0503, 0.2529] |


Verdicts (A* - B; primary only for AUROC and ACER dev in stratum 03 of A* - B)

| stratum | metric | label | verdict | reason |
|---|---|---|---|---|
| 01 | APCER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 01 | BPCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 01 | ACER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 01 | APCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 01 | BPCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 01 | ACER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 01 | AUROC | secondary | inconclusive | the sign differs across the seeds |
| 02 | APCER (0.5) | secondary | inconclusive | a seed has a zero difference |
| 02 | BPCER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 02 | ACER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 02 | APCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 02 | BPCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 02 | ACER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 02 | AUROC | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | APCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 03 | BPCER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | ACER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | APCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 03 | BPCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | ACER (dev threshold) | primary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | AUROC | primary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | APCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| all | BPCER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | ACER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | APCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| all | BPCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | ACER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | AUROC | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |


Median and range over the seeds of the pooled differences

| stratum | APCER (0.5) | BPCER (0.5) | ACER (0.5) | APCER (dev threshold) | BPCER (dev threshold) | ACER (dev threshold) | AUROC |
|---|---|---|---|---|---|---|---|
| 01 | -0.1088 (-0.1099 to -0.0480) | 0.0799 (-0.0011 to 0.0844) | -0.0122 (-0.0555 to 0.0160) | -0.0503 (-0.1099 to 0.0129) | 0.0427 (-0.0551 to 0.0472) | -0.0336 (-0.0527 to 0.0301) | 0.0070 (-0.0035 to 0.0102) |
| 02 | 0.0034 (0.0000 to 0.0515) | 0.0082 (0.0047 to 0.1077) | 0.0299 (0.0040 to 0.0539) | 0.0403 (0.0011 to 0.0504) | 0.0012 (0.0012 to 0.0398) | 0.0207 (0.0205 to 0.0258) | -0.0012 (-0.0034 to -0.0001) |
| 03 | 0.0050 (-0.2100 to 0.0246) | -0.3195 (-0.7064 to -0.1791) | -0.2647 (-0.3507 to -0.0772) | 0.1113 (-0.0618 to 0.2475) | -0.3968 (-0.7326 to -0.1800) | -0.2293 (-0.2425 to -0.0343) | 0.1945 (0.1286 to 0.5694) |
| all | -0.0012 (-0.1735 to 0.0125) | -0.1777 (-0.4505 to -0.1167) | -0.1756 (-0.2259 to -0.0521) | 0.0844 (-0.0598 to 0.1974) | -0.2472 (-0.4740 to -0.1279) | -0.1383 (-0.1535 to -0.0217) | 0.0933 (0.0629 to 0.3102) |


Per fold and seed (A* - B)

| fold | seed | stratum | APCER (0.5) | BPCER (0.5) | ACER (0.5) | APCER (dev threshold) | BPCER (dev threshold) | ACER (dev threshold) | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 42 | 01 | 0.0000 | 0.0000 | 0.0000 | 0.0035 | -0.1890 | -0.0927 | 0.0000 |
| 1 | 42 | 02 | 0.0000 | 0.0123 | 0.0061 | 0.0000 | 0.0041 | 0.0020 | -0.0000 |
| 1 | 42 | 03 | -0.0024 | -0.0973 | -0.0499 | 0.2005 | -0.3114 | -0.0554 | 0.1558 |
| 1 | 42 | all | -0.0016 | -0.0611 | -0.0314 | 0.1363 | -0.2354 | -0.0495 | 0.0722 |
| 1 | 43 | 01 | 0.0000 | 0.2756 | 0.1378 | 0.0000 | 0.1654 | 0.0827 | 0.0044 |
| 1 | 43 | 02 | 0.0000 | 0.0123 | 0.0061 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1 | 43 | 03 | -0.0722 | -0.9416 | -0.5069 | -0.0211 | -0.9557 | -0.4884 | 0.7310 |
| 1 | 43 | all | -0.0489 | -0.5608 | -0.3049 | -0.0143 | -0.5917 | -0.3030 | 0.3602 |
| 1 | 44 | 01 | 0.0000 | 0.3071 | 0.1535 | 0.0000 | 0.1614 | 0.0807 | 0.0000 |
| 1 | 44 | 02 | 0.0000 | 0.3607 | 0.1803 | 0.0000 | 0.1230 | 0.0615 | 0.0000 |
| 1 | 44 | 03 | -0.0398 | -0.0757 | -0.0577 | -0.0195 | -0.3459 | -0.1827 | -0.2888 |
| 1 | 44 | all | -0.0269 | 0.0675 | 0.0203 | -0.0132 | -0.1750 | -0.0941 | -0.1701 |
| 2 | 42 | 01 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | 42 | 02 | 0.0000 | 0.0085 | 0.0043 | 0.0000 | 0.0085 | 0.0043 | 0.0000 |
| 2 | 42 | 03 | -0.2610 | 0.1590 | -0.0510 | -0.0968 | 0.4711 | 0.1871 | 0.0277 |
| 2 | 42 | all | -0.2041 | 0.1190 | -0.0426 | -0.0757 | 0.3511 | 0.1377 | 0.0163 |
| 2 | 43 | 01 | 0.0000 | 0.0043 | 0.0021 | 0.0000 | 0.0000 | 0.0000 | -0.0000 |
| 2 | 43 | 02 | -0.0052 | 0.0085 | 0.0017 | 0.0000 | 0.0085 | 0.0043 | -0.0007 |
| 2 | 43 | 03 | -0.0455 | -0.6624 | -0.3539 | 0.0169 | -0.6614 | -0.3223 | 0.6022 |
| 2 | 43 | all | -0.0361 | -0.4912 | -0.2637 | 0.0132 | -0.4912 | -0.2390 | 0.3991 |
| 2 | 44 | 01 | 0.0053 | 0.0000 | 0.0027 | 0.0000 | 0.0000 | 0.0000 | -0.0000 |
| 2 | 44 | 02 | 0.0052 | 0.0000 | 0.0026 | 0.0052 | 0.0000 | 0.0026 | -0.0053 |
| 2 | 44 | 03 | -0.1540 | -0.2424 | -0.1982 | -0.2588 | -0.1099 | -0.1844 | 0.1770 |
| 2 | 44 | all | -0.1193 | -0.1803 | -0.1498 | -0.2018 | -0.0818 | -0.1418 | 0.1148 |
| 3 | 42 | 01 | 0.0000 | -0.0036 | -0.0018 | 0.0000 | -0.0036 | -0.0018 | 0.0000 |
| 3 | 42 | 02 | 0.0800 | 0.0000 | 0.0400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 42 | 03 | 0.3099 | -0.2289 | 0.0405 | 0.1021 | -0.3699 | -0.1339 | 0.1169 |
| 3 | 42 | all | 0.2593 | -0.1435 | 0.0579 | 0.0828 | -0.2314 | -0.0743 | 0.0553 |
| 3 | 43 | 01 | 0.0000 | 0.0000 | 0.0000 | 0.0535 | 0.0000 | 0.0267 | 0.0000 |
| 3 | 43 | 02 | 0.0150 | 0.0000 | 0.0075 | 0.2200 | 0.0000 | 0.1100 | 0.0000 |
| 3 | 43 | 03 | 0.0733 | -0.4139 | -0.1703 | 0.5321 | -0.5017 | 0.0152 | 0.2768 |
| 3 | 43 | all | 0.0609 | -0.2581 | -0.0986 | 0.4581 | -0.3129 | 0.0726 | 0.1452 |
| 3 | 44 | 01 | 0.0000 | -0.0109 | -0.0054 | 0.0000 | -0.0109 | -0.0054 | 0.0037 |
| 3 | 44 | 02 | -0.0050 | 0.0000 | -0.0025 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 44 | 03 | -0.3387 | -0.3387 | -0.3387 | 0.0997 | -0.4948 | -0.1975 | 0.3051 |
| 3 | 44 | all | -0.2753 | -0.2134 | -0.2444 | 0.0809 | -0.3107 | -0.1149 | 0.1608 |
| 4 | 42 | 01 | -0.4845 | 0.0000 | -0.2423 | -0.2268 | 0.0000 | -0.1134 | 0.0000 |
| 4 | 42 | 02 | 0.1500 | 0.0121 | 0.0811 | 0.1800 | -0.0040 | 0.0880 | -0.0136 |
| 4 | 42 | 03 | -0.0100 | -0.8608 | -0.4354 | 0.2373 | -0.8626 | -0.3126 | 0.7875 |
| 4 | 42 | all | -0.0417 | -0.5114 | -0.2765 | 0.1837 | -0.5168 | -0.1665 | 0.3929 |
| 4 | 43 | 01 | -0.2113 | 0.0000 | -0.1057 | 0.0052 | 0.0000 | 0.0026 | 0.0000 |
| 4 | 43 | 02 | 0.0050 | 0.0000 | 0.0025 | 0.0050 | 0.0000 | 0.0025 | 0.0000 |
| 4 | 43 | 03 | 0.0387 | -0.8517 | -0.4065 | 0.3620 | -0.8517 | -0.2449 | 0.6558 |
| 4 | 43 | all | 0.0095 | -0.5092 | -0.2498 | 0.2878 | -0.5092 | -0.1107 | 0.3259 |
| 4 | 44 | 01 | -0.4845 | 0.0000 | -0.2423 | -0.4845 | 0.0000 | -0.2423 | 0.0000 |
| 4 | 44 | 02 | 0.0000 | 0.0162 | 0.0081 | 0.0000 | 0.0162 | 0.0081 | 0.0000 |
| 4 | 44 | 03 | -0.2580 | -0.8391 | -0.5485 | -0.0967 | -0.8571 | -0.4769 | 0.7045 |
| 4 | 44 | all | -0.2540 | -0.4973 | -0.3756 | -0.1262 | -0.5081 | -0.3171 | 0.3453 |


## Comparison B' - B (secondary): pooled over the 4 folds, per seed

| seed | stratum | subjects | APCER (0.5): difference [bootstrap] | BPCER (0.5): difference [bootstrap] | ACER (0.5): difference [bootstrap] | APCER (dev threshold): difference [bootstrap] | BPCER (dev threshold): difference [bootstrap] | ACER (dev threshold): difference [bootstrap] | AUROC: difference [bootstrap] |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 01 | 9 | 0.0047 [-0.0473, 0.0505] | 0.2497 [0.0000, 0.5016] | 0.1272 [-0.0002, 0.2607] | 0.0678 [0.0102, 0.1547] | 0.1204 [-0.1038, 0.3648] | 0.0941 [-0.0274, 0.2342] | -0.0878 [-0.1903, -0.0018] |
| 42 | 02 | 9 | 0.0952 [-0.0022, 0.2352] | 0.0141 [0.0021, 0.0327] | 0.0546 [0.0046, 0.1280] | 0.0784 [0.0000, 0.1971] | 0.0023 [-0.0058, 0.0115] | 0.0404 [-0.0017, 0.1018] | -0.0779 [-0.1963, 0.0000] |
| 42 | 03 | 16 | -0.0330 [-0.1301, 0.0793] | 0.2466 [-0.0306, 0.5083] | 0.1068 [-0.0414, 0.2509] | 0.1717 [0.0699, 0.3080] | 0.2499 [-0.1296, 0.5111] | 0.2108 [0.0541, 0.3282] | -0.1739 [-0.4328, 0.1288] |
| 42 | all | 16 | -0.0135 [-0.0969, 0.0930] | 0.2082 [0.0279, 0.4008] | 0.0974 [-0.0046, 0.2092] | 0.1488 [0.0614, 0.2723] | 0.1859 [-0.0644, 0.3746] | 0.1673 [0.0588, 0.2722] | -0.1533 [-0.3239, 0.0255] |
| 43 | 01 | 9 | 0.0596 [0.0000, 0.1728] | -0.0090 [-0.0354, 0.0047] | 0.0253 [-0.0085, 0.0839] | 0.0959 [0.0000, 0.2867] | 0.0315 [-0.0018, 0.1037] | 0.0637 [0.0000, 0.1656] | -0.0028 [-0.0051, 0.0000] |
| 43 | 02 | 9 | 0.0246 [-0.0012, 0.0678] | 0.0047 [0.0000, 0.0139] | 0.0147 [0.0003, 0.0369] | 0.0034 [0.0000, 0.0101] | 0.1311 [0.0000, 0.3462] | 0.0673 [0.0017, 0.1743] | -0.0035 [-0.0101, 0.0000] |
| 43 | 03 | 16 | 0.0420 [-0.0873, 0.1657] | -0.1071 [-0.3359, 0.0525] | -0.0325 [-0.1504, 0.0782] | 0.0594 [-0.0380, 0.1527] | 0.0238 [-0.0461, 0.1103] | 0.0416 [-0.0203, 0.1010] | 0.0808 [-0.0980, 0.2297] |
| 43 | all | 16 | 0.0419 [-0.0658, 0.1293] | -0.0713 [-0.2342, 0.0321] | -0.0147 [-0.1077, 0.0594] | 0.0569 [-0.0294, 0.1375] | 0.0431 [-0.0110, 0.1083] | 0.0500 [-0.0043, 0.1004] | 0.0009 [-0.1125, 0.1119] |
| 44 | 01 | 9 | -0.1053 [-0.3173, 0.0000] | -0.0315 [-0.0967, 0.0036] | -0.0684 [-0.1796, 0.0000] | 0.0140 [-0.0409, 0.0827] | -0.0191 [-0.0823, 0.0254] | -0.0025 [-0.0449, 0.0388] | 0.0126 [-0.0031, 0.0678] |
| 44 | 02 | 9 | 0.0000 [-0.0033, 0.0033] | 0.1393 [-0.0040, 0.4274] | 0.0697 [-0.0025, 0.2137] | 0.0112 [0.0000, 0.0300] | 0.1710 [0.0000, 0.4566] | 0.0911 [0.0055, 0.2321] | -0.1154 [-0.3226, -0.0000] |
| 44 | 03 | 16 | -0.0856 [-0.2513, 0.0650] | 0.1591 [0.0137, 0.3194] | 0.0368 [-0.0741, 0.1611] | 0.0340 [-0.0998, 0.1736] | 0.0309 [-0.3191, 0.3151] | 0.0325 [-0.1447, 0.1845] | -0.1149 [-0.2806, 0.0730] |
| 44 | all | 16 | -0.0776 [-0.2138, 0.0410] | 0.1226 [0.0213, 0.2203] | 0.0225 [-0.0651, 0.1079] | 0.0290 [-0.0788, 0.1409] | 0.0456 [-0.2041, 0.2176] | 0.0373 [-0.0948, 0.1314] | -0.1104 [-0.2311, 0.0249] |


Verdicts (B' - B; primary only for AUROC and ACER dev in stratum 03 of A* - B)

| stratum | metric | label | verdict | reason |
|---|---|---|---|---|
| 01 | APCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 01 | BPCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 01 | ACER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 01 | APCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 01 | BPCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 01 | ACER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 01 | AUROC | secondary | inconclusive | the sign differs across the seeds |
| 02 | APCER (0.5) | secondary | inconclusive | a seed has a zero difference |
| 02 | BPCER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 02 | ACER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 02 | APCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 02 | BPCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 02 | ACER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 02 | AUROC | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | APCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 03 | BPCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 03 | ACER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 03 | APCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | BPCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | ACER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | AUROC | secondary | inconclusive | the sign differs across the seeds |
| all | APCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| all | BPCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| all | ACER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| all | APCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | BPCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | ACER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | AUROC | secondary | inconclusive | the sign differs across the seeds |


Median and range over the seeds of the pooled differences

| stratum | APCER (0.5) | BPCER (0.5) | ACER (0.5) | APCER (dev threshold) | BPCER (dev threshold) | ACER (dev threshold) | AUROC |
|---|---|---|---|---|---|---|---|
| 01 | 0.0047 (-0.1053 to 0.0596) | -0.0090 (-0.0315 to 0.2497) | 0.0253 (-0.0684 to 0.1272) | 0.0678 (0.0140 to 0.0959) | 0.0315 (-0.0191 to 0.1204) | 0.0637 (-0.0025 to 0.0941) | -0.0028 (-0.0878 to 0.0126) |
| 02 | 0.0246 (0.0000 to 0.0952) | 0.0141 (0.0047 to 0.1393) | 0.0546 (0.0147 to 0.0697) | 0.0112 (0.0034 to 0.0784) | 0.1311 (0.0023 to 0.1710) | 0.0673 (0.0404 to 0.0911) | -0.0779 (-0.1154 to -0.0035) |
| 03 | -0.0330 (-0.0856 to 0.0420) | 0.1591 (-0.1071 to 0.2466) | 0.0368 (-0.0325 to 0.1068) | 0.0594 (0.0340 to 0.1717) | 0.0309 (0.0238 to 0.2499) | 0.0416 (0.0325 to 0.2108) | -0.1149 (-0.1739 to 0.0808) |
| all | -0.0135 (-0.0776 to 0.0419) | 0.1226 (-0.0713 to 0.2082) | 0.0225 (-0.0147 to 0.0974) | 0.0569 (0.0290 to 0.1488) | 0.0456 (0.0431 to 0.1859) | 0.0500 (0.0373 to 0.1673) | -0.1104 (-0.1533 to 0.0009) |


Per fold and seed (B' - B)

| fold | seed | stratum | APCER (0.5) | BPCER (0.5) | ACER (0.5) | APCER (dev threshold) | BPCER (dev threshold) | ACER (dev threshold) | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 42 | 01 | 0.0000 | 0.3898 | 0.1949 | 0.0000 | -0.1575 | -0.0787 | -0.0076 |
| 1 | 42 | 02 | 0.0000 | 0.0082 | 0.0041 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1 | 42 | 03 | -0.0203 | 0.6735 | 0.3266 | 0.0333 | 0.4854 | 0.2593 | -0.3349 |
| 1 | 42 | all | -0.0137 | 0.5088 | 0.2475 | 0.0225 | 0.2874 | 0.1550 | -0.2182 |
| 1 | 43 | 01 | 0.0105 | -0.0394 | -0.0145 | 0.0035 | 0.1063 | 0.0549 | -0.0054 |
| 1 | 43 | 02 | 0.0133 | 0.0000 | 0.0067 | 0.0100 | 0.0000 | 0.0050 | -0.0099 |
| 1 | 43 | 03 | 0.3052 | -0.4454 | -0.0701 | 0.1534 | -0.0832 | 0.0351 | 0.1628 |
| 1 | 43 | all | 0.2106 | -0.2966 | -0.0430 | 0.1061 | -0.0351 | 0.0355 | 0.0198 |
| 1 | 44 | 01 | 0.0000 | -0.0984 | -0.0492 | 0.0000 | -0.0551 | -0.0276 | -0.0020 |
| 1 | 44 | 02 | 0.0000 | 0.4836 | 0.2418 | 0.0000 | 0.5943 | 0.2971 | -0.2397 |
| 1 | 44 | 03 | 0.1583 | 0.1751 | 0.1667 | 0.0706 | 0.0757 | 0.0731 | -0.4480 |
| 1 | 44 | all | 0.1072 | 0.1792 | 0.1432 | 0.0478 | 0.1413 | 0.0945 | -0.3517 |
| 2 | 42 | 01 | 0.1123 | 0.5256 | 0.3190 | 0.1016 | 0.6282 | 0.3649 | -0.1842 |
| 2 | 42 | 02 | 0.4456 | 0.0000 | 0.2228 | 0.3627 | 0.0256 | 0.1942 | -0.3120 |
| 2 | 42 | 03 | 0.1547 | 0.0118 | 0.0832 | 0.2053 | 0.5604 | 0.3828 | -0.2724 |
| 2 | 42 | all | 0.1823 | 0.0985 | 0.1404 | 0.2116 | 0.5263 | 0.3689 | -0.3070 |
| 2 | 43 | 01 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0043 | -0.0021 | 0.0000 |
| 2 | 43 | 02 | 0.0933 | 0.0000 | 0.0466 | 0.0000 | 0.0171 | 0.0085 | 0.0000 |
| 2 | 43 | 03 | 0.1642 | -0.0245 | 0.0698 | -0.0330 | 0.0864 | 0.0267 | 0.1217 |
| 2 | 43 | all | 0.1388 | -0.0182 | 0.0603 | -0.0258 | 0.0650 | 0.0196 | 0.0917 |
| 2 | 44 | 01 | 0.0000 | -0.0043 | -0.0021 | 0.0000 | -0.0043 | -0.0021 | 0.0000 |
| 2 | 44 | 02 | 0.0000 | -0.0085 | -0.0043 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | 44 | 03 | 0.1884 | 0.2689 | 0.2287 | -0.1796 | 0.4298 | 0.1251 | -0.2173 |
| 2 | 44 | all | 0.1474 | 0.1985 | 0.1730 | -0.1405 | 0.3190 | 0.0892 | -0.1160 |
| 3 | 42 | 01 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 42 | 02 | -0.0050 | 0.0081 | 0.0016 | 0.0000 | 0.0041 | 0.0020 | -0.0000 |
| 3 | 42 | 03 | -0.0649 | 0.3318 | 0.1335 | 0.1471 | 0.1572 | 0.1522 | -0.3404 |
| 3 | 42 | all | -0.0531 | 0.2084 | 0.0776 | 0.1194 | 0.0988 | 0.1091 | -0.1938 |
| 3 | 43 | 01 | 0.0000 | 0.0072 | 0.0036 | 0.0000 | 0.0072 | 0.0036 | -0.0036 |
| 3 | 43 | 02 | 0.0000 | 0.0163 | 0.0081 | 0.0000 | 0.4472 | 0.2236 | 0.0000 |
| 3 | 43 | 03 | -0.1141 | 0.0890 | -0.0125 | -0.0342 | 0.0798 | 0.0228 | -0.1468 |
| 3 | 43 | all | -0.0926 | 0.0598 | -0.0164 | -0.0278 | 0.1305 | 0.0514 | -0.1382 |
| 3 | 44 | 01 | 0.0000 | -0.0072 | -0.0036 | 0.0000 | -0.0072 | -0.0036 | 0.0036 |
| 3 | 44 | 02 | -0.0050 | 0.0041 | -0.0005 | 0.0000 | 0.0041 | 0.0020 | 0.0000 |
| 3 | 44 | 03 | -0.3814 | 0.1133 | -0.1340 | -0.0414 | 0.0058 | -0.0178 | 0.0749 |
| 3 | 44 | all | -0.3099 | 0.0699 | -0.1200 | -0.0336 | 0.0029 | -0.0154 | 0.0357 |
| 4 | 42 | 01 | -0.0876 | 0.0000 | -0.0438 | 0.2010 | 0.0000 | 0.1005 | 0.0000 |
| 4 | 42 | 02 | 0.0000 | 0.0324 | 0.0162 | 0.0000 | -0.0081 | -0.0040 | 0.0000 |
| 4 | 42 | 03 | -0.1787 | -0.1682 | -0.1734 | 0.2820 | -0.5714 | -0.1447 | 0.4268 |
| 4 | 42 | all | -0.1505 | -0.0919 | -0.1212 | 0.2439 | -0.3438 | -0.0499 | 0.2016 |
| 4 | 43 | 01 | 0.2474 | 0.0000 | 0.1237 | 0.4175 | 0.0000 | 0.2088 | 0.0000 |
| 4 | 43 | 02 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4 | 43 | 03 | -0.1120 | 0.0000 | -0.0560 | 0.1700 | 0.0000 | 0.0850 | -0.0760 |
| 4 | 43 | all | -0.0634 | 0.0000 | -0.0317 | 0.1774 | 0.0000 | 0.0887 | -0.1142 |
| 4 | 44 | 01 | -0.4639 | 0.0000 | -0.2320 | 0.0619 | 0.0000 | 0.0309 | 0.0000 |
| 4 | 44 | 02 | 0.0050 | 0.0040 | 0.0045 | 0.0500 | 0.0000 | 0.0250 | -0.0001 |
| 4 | 44 | 03 | -0.2067 | 0.0018 | -0.1024 | 0.2820 | -0.7396 | -0.2288 | 0.2512 |
| 4 | 44 | all | -0.2107 | 0.0022 | -0.1043 | 0.2350 | -0.4422 | -0.1036 | 0.1210 |


## Comparison A* - B' (secondary): pooled over the 4 folds, per seed

| seed | stratum | subjects | APCER (0.5): difference [bootstrap] | BPCER (0.5): difference [bootstrap] | ACER (0.5): difference [bootstrap] | APCER (dev threshold): difference [bootstrap] | BPCER (dev threshold): difference [bootstrap] | ACER (dev threshold): difference [bootstrap] | AUROC: difference [bootstrap] |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 01 | 9 | -0.1146 [-0.2929, -0.0110] | -0.2508 [-0.5026, -0.0023] | -0.1827 [-0.3471, -0.0455] | -0.1181 [-0.3104, -0.0066] | -0.1755 [-0.3801, -0.0023] | -0.1468 [-0.2951, -0.0336] | 0.0981 [0.0021, 0.2099] |
| 42 | 02 | 9 | -0.0437 [-0.2180, 0.1044] | -0.0059 [-0.0214, 0.0061] | -0.0248 [-0.1126, 0.0494] | -0.0381 [-0.1786, 0.1019] | -0.0012 [-0.0083, 0.0047] | -0.0196 [-0.0935, 0.0515] | 0.0745 [-0.0062, 0.1956] |
| 42 | 03 | 16 | 0.0576 [-0.1570, 0.2673] | -0.4256 [-0.7262, -0.0680] | -0.1840 [-0.3506, -0.0047] | -0.0604 [-0.2374, 0.1221] | -0.4298 [-0.6878, -0.1650] | -0.2451 [-0.3914, -0.1025] | 0.3025 [0.0742, 0.5543] |
| 42 | all | 16 | 0.0260 [-0.1499, 0.2165] | -0.3250 [-0.5619, -0.0852] | -0.1495 [-0.2859, -0.0135] | -0.0643 [-0.2139, 0.0981] | -0.3138 [-0.5015, -0.1309] | -0.1891 [-0.3075, -0.0809] | 0.2162 [0.0676, 0.3796] |
| 43 | 01 | 9 | -0.1076 [-0.3196, 0.0023] | 0.0889 [-0.0042, 0.2970] | -0.0094 [-0.1513, 0.1169] | -0.0830 [-0.2826, 0.0262] | 0.0157 [-0.0042, 0.0552] | -0.0336 [-0.1318, 0.0250] | -0.0007 [-0.0081, 0.0031] |
| 43 | 02 | 9 | -0.0213 [-0.0661, 0.0078] | 0.0000 [-0.0122, 0.0107] | -0.0106 [-0.0333, 0.0038] | 0.0470 [-0.0089, 0.1478] | -0.1300 [-0.3462, 0.0000] | -0.0415 [-0.1541, 0.0393] | 0.0034 [-0.0003, 0.0101] |
| 43 | 03 | 16 | -0.0370 [-0.2902, 0.2513] | -0.5993 [-0.8407, -0.3403] | -0.3182 [-0.4832, -0.1496] | 0.1882 [-0.0467, 0.4663] | -0.7564 [-0.9162, -0.5271] | -0.2841 [-0.4313, -0.1109] | 0.4887 [0.2866, 0.6638] |
| 43 | all | 16 | -0.0431 [-0.2296, 0.2073] | -0.3792 [-0.5573, -0.1919] | -0.2112 [-0.3139, -0.1068] | 0.1405 [-0.0435, 0.4064] | -0.5171 [-0.6977, -0.3151] | -0.1883 [-0.2883, -0.0493] | 0.3092 [0.1661, 0.4263] |
| 44 | 01 | 9 | -0.0035 [-0.0138, 0.0035] | 0.1159 [-0.0042, 0.3855] | 0.0562 [-0.0064, 0.1904] | -0.1240 [-0.3184, 0.0000] | 0.0619 [-0.0275, 0.2316] | -0.0311 [-0.1476, 0.0818] | -0.0056 [-0.0267, 0.0030] |
| 44 | 02 | 9 | 0.0000 [-0.0033, 0.0034] | -0.0316 [-0.1080, 0.0126] | -0.0158 [-0.0535, 0.0057] | -0.0101 [-0.0282, 0.0023] | -0.1311 [-0.3349, 0.0095] | -0.0706 [-0.1722, -0.0024] | 0.1142 [-0.0023, 0.3223] |
| 44 | 03 | 16 | -0.1245 [-0.2689, 0.0345] | -0.4786 [-0.7520, -0.1820] | -0.3015 [-0.4532, -0.1452] | -0.0958 [-0.2867, 0.0822] | -0.4277 [-0.7246, -0.0904] | -0.2618 [-0.4261, -0.1010] | 0.3094 [0.1092, 0.5136] |
| 44 | all | 16 | -0.0959 [-0.2004, 0.0279] | -0.3003 [-0.5382, -0.0710] | -0.1981 [-0.3089, -0.0861] | -0.0888 [-0.2471, 0.0649] | -0.2929 [-0.5135, -0.0709] | -0.1908 [-0.3013, -0.0788] | 0.2037 [0.0842, 0.3456] |


Verdicts (A* - B'; primary only for AUROC and ACER dev in stratum 03 of A* - B)

| stratum | metric | label | verdict | reason |
|---|---|---|---|---|
| 01 | APCER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 01 | BPCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 01 | ACER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 01 | APCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 01 | BPCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 01 | ACER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 01 | AUROC | secondary | inconclusive | the sign differs across the seeds |
| 02 | APCER (0.5) | secondary | inconclusive | a seed has a zero difference |
| 02 | BPCER (0.5) | secondary | inconclusive | a seed has a zero difference |
| 02 | ACER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 02 | APCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 02 | BPCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 02 | ACER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 02 | AUROC | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | APCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 03 | BPCER (0.5) | secondary | consistent: A* lower BPCER (0.5) than B' | same sign for every seed and every interval excludes 0 |
| 03 | ACER (0.5) | secondary | consistent: A* lower ACER (0.5) than B' | same sign for every seed and every interval excludes 0 |
| 03 | APCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 03 | BPCER (dev threshold) | secondary | consistent: A* lower BPCER (dev threshold) than B' | same sign for every seed and every interval excludes 0 |
| 03 | ACER (dev threshold) | secondary | consistent: A* lower ACER (dev threshold) than B' | same sign for every seed and every interval excludes 0 |
| 03 | AUROC | secondary | consistent: A* higher AUROC than B' | same sign for every seed and every interval excludes 0 |
| all | APCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| all | BPCER (0.5) | secondary | consistent: A* lower BPCER (0.5) than B' | same sign for every seed and every interval excludes 0 |
| all | ACER (0.5) | secondary | consistent: A* lower ACER (0.5) than B' | same sign for every seed and every interval excludes 0 |
| all | APCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| all | BPCER (dev threshold) | secondary | consistent: A* lower BPCER (dev threshold) than B' | same sign for every seed and every interval excludes 0 |
| all | ACER (dev threshold) | secondary | consistent: A* lower ACER (dev threshold) than B' | same sign for every seed and every interval excludes 0 |
| all | AUROC | secondary | consistent: A* higher AUROC than B' | same sign for every seed and every interval excludes 0 |


Median and range over the seeds of the pooled differences

| stratum | APCER (0.5) | BPCER (0.5) | ACER (0.5) | APCER (dev threshold) | BPCER (dev threshold) | ACER (dev threshold) | AUROC |
|---|---|---|---|---|---|---|---|
| 01 | -0.1076 (-0.1146 to -0.0035) | 0.0889 (-0.2508 to 0.1159) | -0.0094 (-0.1827 to 0.0562) | -0.1181 (-0.1240 to -0.0830) | 0.0157 (-0.1755 to 0.0619) | -0.0336 (-0.1468 to -0.0311) | -0.0007 (-0.0056 to 0.0981) |
| 02 | -0.0213 (-0.0437 to 0.0000) | -0.0059 (-0.0316 to 0.0000) | -0.0158 (-0.0248 to -0.0106) | -0.0101 (-0.0381 to 0.0470) | -0.1300 (-0.1311 to -0.0012) | -0.0415 (-0.0706 to -0.0196) | 0.0745 (0.0034 to 0.1142) |
| 03 | -0.0370 (-0.1245 to 0.0576) | -0.4786 (-0.5993 to -0.4256) | -0.3015 (-0.3182 to -0.1840) | -0.0604 (-0.0958 to 0.1882) | -0.4298 (-0.7564 to -0.4277) | -0.2618 (-0.2841 to -0.2451) | 0.3094 (0.3025 to 0.4887) |
| all | -0.0431 (-0.0959 to 0.0260) | -0.3250 (-0.3792 to -0.3003) | -0.1981 (-0.2112 to -0.1495) | -0.0643 (-0.0888 to 0.1405) | -0.3138 (-0.5171 to -0.2929) | -0.1891 (-0.1908 to -0.1883) | 0.2162 (0.2037 to 0.3092) |


Per fold and seed (A* - B')

| fold | seed | stratum | APCER (0.5) | BPCER (0.5) | ACER (0.5) | APCER (dev threshold) | BPCER (dev threshold) | ACER (dev threshold) | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 42 | 01 | 0.0000 | -0.3898 | -0.1949 | 0.0035 | -0.0315 | -0.0140 | 0.0076 |
| 1 | 42 | 02 | 0.0000 | 0.0041 | 0.0020 | 0.0000 | 0.0041 | 0.0020 | -0.0000 |
| 1 | 42 | 03 | 0.0179 | -0.7708 | -0.3765 | 0.1672 | -0.7968 | -0.3148 | 0.4907 |
| 1 | 42 | all | 0.0121 | -0.5699 | -0.2789 | 0.1138 | -0.5228 | -0.2045 | 0.2905 |
| 1 | 43 | 01 | -0.0105 | 0.3150 | 0.1523 | -0.0035 | 0.0591 | 0.0278 | 0.0097 |
| 1 | 43 | 02 | -0.0133 | 0.0123 | -0.0005 | -0.0100 | 0.0000 | -0.0050 | 0.0099 |
| 1 | 43 | 03 | -0.3774 | -0.4962 | -0.4368 | -0.1745 | -0.8724 | -0.5235 | 0.5682 |
| 1 | 43 | all | -0.2595 | -0.2642 | -0.2619 | -0.1204 | -0.5566 | -0.3385 | 0.3404 |
| 1 | 44 | 01 | 0.0000 | 0.4055 | 0.2028 | 0.0000 | 0.2165 | 0.1083 | 0.0020 |
| 1 | 44 | 02 | 0.0000 | -0.1230 | -0.0615 | 0.0000 | -0.4713 | -0.2357 | 0.2398 |
| 1 | 44 | 03 | -0.1981 | -0.2508 | -0.2244 | -0.0901 | -0.4216 | -0.2559 | 0.1593 |
| 1 | 44 | all | -0.1341 | -0.1117 | -0.1229 | -0.0610 | -0.3162 | -0.1886 | 0.1816 |
| 2 | 42 | 01 | -0.1123 | -0.5256 | -0.3190 | -0.1016 | -0.6282 | -0.3649 | 0.1842 |
| 2 | 42 | 02 | -0.4456 | 0.0085 | -0.2185 | -0.3627 | -0.0171 | -0.1899 | 0.3120 |
| 2 | 42 | 03 | -0.4157 | 0.1472 | -0.1342 | -0.3021 | -0.0893 | -0.1957 | 0.3001 |
| 2 | 42 | all | -0.3865 | 0.0204 | -0.1830 | -0.2873 | -0.1752 | -0.2312 | 0.3233 |
| 2 | 43 | 01 | 0.0000 | 0.0043 | 0.0021 | 0.0000 | 0.0043 | 0.0021 | -0.0000 |
| 2 | 43 | 02 | -0.0984 | 0.0085 | -0.0449 | 0.0000 | -0.0085 | -0.0043 | -0.0007 |
| 2 | 43 | 03 | -0.2097 | -0.6379 | -0.4238 | 0.0499 | -0.7478 | -0.3490 | 0.4805 |
| 2 | 43 | all | -0.1749 | -0.4730 | -0.3239 | 0.0390 | -0.5562 | -0.2586 | 0.3073 |
| 2 | 44 | 01 | 0.0053 | 0.0043 | 0.0048 | 0.0000 | 0.0043 | 0.0021 | -0.0000 |
| 2 | 44 | 02 | 0.0052 | 0.0085 | 0.0069 | 0.0052 | 0.0000 | 0.0026 | -0.0053 |
| 2 | 44 | 03 | -0.3424 | -0.5113 | -0.4268 | -0.0792 | -0.5397 | -0.3095 | 0.3943 |
| 2 | 44 | all | -0.2666 | -0.3788 | -0.3227 | -0.0614 | -0.4007 | -0.2310 | 0.2308 |
| 3 | 42 | 01 | 0.0000 | -0.0036 | -0.0018 | 0.0000 | -0.0036 | -0.0018 | 0.0000 |
| 3 | 42 | 02 | 0.0850 | -0.0081 | 0.0384 | 0.0000 | -0.0041 | -0.0020 | 0.0000 |
| 3 | 42 | 03 | 0.3748 | -0.5607 | -0.0930 | -0.0450 | -0.5272 | -0.2861 | 0.4573 |
| 3 | 42 | all | 0.3124 | -0.3518 | -0.0197 | -0.0365 | -0.3302 | -0.1834 | 0.2491 |
| 3 | 43 | 01 | 0.0000 | -0.0072 | -0.0036 | 0.0535 | -0.0072 | 0.0231 | 0.0036 |
| 3 | 43 | 02 | 0.0150 | -0.0163 | -0.0006 | 0.2200 | -0.4472 | -0.1136 | 0.0000 |
| 3 | 43 | 03 | 0.1874 | -0.5029 | -0.1578 | 0.5664 | -0.5815 | -0.0076 | 0.4236 |
| 3 | 43 | all | 0.1535 | -0.3180 | -0.0822 | 0.4859 | -0.4434 | 0.0212 | 0.2834 |
| 3 | 44 | 01 | 0.0000 | -0.0036 | -0.0018 | 0.0000 | -0.0036 | -0.0018 | 0.0000 |
| 3 | 44 | 02 | 0.0000 | -0.0041 | -0.0020 | 0.0000 | -0.0041 | -0.0020 | 0.0000 |
| 3 | 44 | 03 | 0.0426 | -0.4520 | -0.2047 | 0.1411 | -0.5006 | -0.1797 | 0.2302 |
| 3 | 44 | all | 0.0346 | -0.2833 | -0.1244 | 0.1145 | -0.3136 | -0.0996 | 0.1251 |
| 4 | 42 | 01 | -0.3969 | 0.0000 | -0.1985 | -0.4278 | 0.0000 | -0.2139 | 0.0000 |
| 4 | 42 | 02 | 0.1500 | -0.0202 | 0.0649 | 0.1800 | 0.0040 | 0.0920 | -0.0136 |
| 4 | 42 | 03 | 0.1687 | -0.6926 | -0.2620 | -0.0447 | -0.2911 | -0.1679 | 0.3606 |
| 4 | 42 | all | 0.1088 | -0.4195 | -0.1553 | -0.0602 | -0.1730 | -0.1166 | 0.1913 |
| 4 | 43 | 01 | -0.4588 | 0.0000 | -0.2294 | -0.4124 | 0.0000 | -0.2062 | 0.0000 |
| 4 | 43 | 02 | 0.0050 | 0.0000 | 0.0025 | 0.0050 | 0.0000 | 0.0025 | 0.0000 |
| 4 | 43 | 03 | 0.1507 | -0.8517 | -0.3505 | 0.1920 | -0.8517 | -0.3299 | 0.7319 |
| 4 | 43 | all | 0.0729 | -0.5092 | -0.2182 | 0.1103 | -0.5092 | -0.1994 | 0.4400 |
| 4 | 44 | 01 | -0.0206 | 0.0000 | -0.0103 | -0.5464 | 0.0000 | -0.2732 | 0.0000 |
| 4 | 44 | 02 | -0.0050 | 0.0121 | 0.0036 | -0.0500 | 0.0162 | -0.0169 | 0.0001 |
| 4 | 44 | 03 | -0.0513 | -0.8409 | -0.4461 | -0.3787 | -0.1175 | -0.2481 | 0.4533 |
| 4 | 44 | all | -0.0433 | -0.4995 | -0.2714 | -0.3611 | -0.0659 | -0.2135 | 0.2244 |


## Per arm: pooled over the folds, per seed (bootstrap over subjects)

| arm | seed | stratum | n bona fide | n attack | subjects | BPCER 05 | APCER 05 | ACER 05 [bootstrap] | BPCER dev | APCER dev | ACER dev [bootstrap] | AUROC [bootstrap] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B | 42 | 01 | 889 | 855 | 9 | 0.0079 | 0.1111 | 0.0595 [0.0016, 0.1716] | 0.0562 | 0.0538 | 0.0550 [0.0008, 0.1450] | 0.9896 [0.9414, 1.0000] |
| B | 42 | 02 | 854 | 893 | 9 | 0.0000 | 0.0011 | 0.0006 [0.0000, 0.0017] | 0.0023 | 0.0000 | 0.0012 [0.0000, 0.0035] | 1.0000 [1.0000, 1.0000] |
| B | 42 | 03 | 3362 | 5761 | 16 | 0.3022 | 0.2732 | 0.2877 [0.1540, 0.4319] | 0.4042 | 0.1206 | 0.2624 [0.1210, 0.3957] | 0.7326 [0.5030, 0.9357] |
| B | 42 | all | 5105 | 7509 | 16 | 0.2004 | 0.2224 | 0.2114 [0.1165, 0.3292] | 0.2764 | 0.0987 | 0.1875 [0.0893, 0.2930] | 0.8473 [0.6925, 0.9667] |
| B | 43 | 01 | 889 | 855 | 9 | 0.0236 | 0.0491 | 0.0364 [0.0000, 0.0925] | 0.0427 | 0.0000 | 0.0214 [0.0000, 0.0695] | 0.9936 [0.9727, 1.0000] |
| B | 43 | 02 | 854 | 893 | 9 | 0.0000 | 0.0011 | 0.0006 [0.0000, 0.0017] | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] |
| B | 43 | 03 | 3362 | 5761 | 16 | 0.7478 | 0.2928 | 0.5203 [0.3503, 0.6486] | 0.7674 | 0.0856 | 0.4265 [0.2756, 0.5337] | 0.3608 [0.1528, 0.6247] |
| B | 43 | all | 5105 | 7509 | 16 | 0.4966 | 0.2304 | 0.3635 [0.2262, 0.4880] | 0.5128 | 0.0657 | 0.2892 [0.1655, 0.4025] | 0.6424 [0.4761, 0.8233] |
| B | 44 | 01 | 889 | 855 | 9 | 0.0394 | 0.1099 | 0.0747 [0.0017, 0.1921] | 0.0810 | 0.1099 | 0.0955 [0.0017, 0.2313] | 0.9852 [0.9275, 1.0000] |
| B | 44 | 02 | 854 | 893 | 9 | 0.0047 | 0.0011 | 0.0029 [0.0000, 0.0062] | 0.0094 | 0.0000 | 0.0047 [0.0000, 0.0128] | 1.0000 [0.9998, 1.0000] |
| B | 44 | 03 | 3362 | 5761 | 16 | 0.5476 | 0.3232 | 0.4354 [0.2657, 0.5800] | 0.5922 | 0.1909 | 0.3916 [0.2098, 0.5454] | 0.6470 [0.4556, 0.8351] |
| B | 44 | all | 5105 | 7509 | 16 | 0.3683 | 0.2606 | 0.3144 [0.1794, 0.4571] | 0.4057 | 0.1590 | 0.2823 [0.1394, 0.4272] | 0.7998 [0.6478, 0.9155] |
| A* | 42 | 01 | 889 | 855 | 9 | 0.0067 | 0.0012 | 0.0040 [0.0005, 0.0102] | 0.0011 | 0.0035 | 0.0023 [0.0006, 0.0041] | 0.9999 [0.9994, 1.0000] |
| A* | 42 | 02 | 854 | 893 | 9 | 0.0082 | 0.0526 | 0.0304 [0.0041, 0.0652] | 0.0035 | 0.0403 | 0.0219 [0.0010, 0.0623] | 0.9966 [0.9925, 1.0000] |
| A* | 42 | 03 | 3362 | 5761 | 16 | 0.1231 | 0.2979 | 0.2105 [0.1292, 0.2998] | 0.2243 | 0.2319 | 0.2281 [0.1217, 0.3337] | 0.8612 [0.7820, 0.9501] |
| A* | 42 | all | 5105 | 7509 | 16 | 0.0836 | 0.2349 | 0.1593 [0.0900, 0.2512] | 0.1485 | 0.1831 | 0.1658 [0.0847, 0.2547] | 0.9102 [0.8429, 0.9680] |
| A* | 43 | 01 | 889 | 855 | 9 | 0.1035 | 0.0012 | 0.0523 [0.0005, 0.1711] | 0.0900 | 0.0129 | 0.0514 [0.0012, 0.1511] | 0.9901 [0.9616, 1.0000] |
| A* | 43 | 02 | 854 | 893 | 9 | 0.0047 | 0.0045 | 0.0046 [0.0010, 0.0086] | 0.0012 | 0.0504 | 0.0258 [0.0000, 0.0746] | 0.9999 [0.9996, 1.0000] |
| A* | 43 | 03 | 3362 | 5761 | 16 | 0.0413 | 0.2979 | 0.1696 [0.0827, 0.2722] | 0.0348 | 0.3331 | 0.1840 [0.0931, 0.2948] | 0.9303 [0.8375, 0.9852] |
| A* | 43 | all | 5105 | 7509 | 16 | 0.0460 | 0.2292 | 0.1376 [0.0652, 0.2324] | 0.0388 | 0.2630 | 0.1509 [0.0748, 0.2572] | 0.9526 [0.8791, 0.9905] |
| A* | 44 | 01 | 889 | 855 | 9 | 0.1237 | 0.0012 | 0.0625 [0.0000, 0.2049] | 0.1237 | 0.0000 | 0.0619 [0.0000, 0.2049] | 0.9922 [0.9694, 1.0000] |
| A* | 44 | 02 | 854 | 893 | 9 | 0.1124 | 0.0011 | 0.0568 [0.0016, 0.1638] | 0.0492 | 0.0011 | 0.0252 [0.0012, 0.0690] | 0.9987 [0.9964, 1.0000] |
| A* | 44 | 03 | 3362 | 5761 | 16 | 0.2281 | 0.1132 | 0.1707 [0.0583, 0.2996] | 0.1954 | 0.1291 | 0.1623 [0.0667, 0.2856] | 0.8415 [0.5852, 0.9947] |
| A* | 44 | all | 5105 | 7509 | 16 | 0.1906 | 0.0871 | 0.1388 [0.0420, 0.2586] | 0.1585 | 0.0992 | 0.1288 [0.0462, 0.2372] | 0.8931 [0.7289, 0.9949] |
| B' | 42 | 01 | 889 | 855 | 9 | 0.2576 | 0.1158 | 0.1867 [0.0467, 0.3548] | 0.1766 | 0.1216 | 0.1491 [0.0351, 0.2981] | 0.9018 [0.7900, 0.9979] |
| B' | 42 | 02 | 854 | 893 | 9 | 0.0141 | 0.0963 | 0.0552 [0.0053, 0.1280] | 0.0047 | 0.0784 | 0.0415 [0.0000, 0.1019] | 0.9221 [0.8037, 1.0000] |
| B' | 42 | 03 | 3362 | 5761 | 16 | 0.5488 | 0.2402 | 0.3945 [0.2116, 0.5477] | 0.6541 | 0.2923 | 0.4732 [0.3264, 0.6035] | 0.5587 [0.3136, 0.8052] |
| B' | 42 | all | 5105 | 7509 | 16 | 0.4086 | 0.2089 | 0.3088 [0.1620, 0.4551] | 0.4623 | 0.2474 | 0.3549 [0.2273, 0.4862] | 0.6940 [0.5139, 0.8596] |
| B' | 43 | 01 | 889 | 855 | 9 | 0.0146 | 0.1088 | 0.0617 [0.0009, 0.1703] | 0.0742 | 0.0959 | 0.0851 [0.0009, 0.2104] | 0.9908 [0.9696, 1.0000] |
| B' | 43 | 02 | 854 | 893 | 9 | 0.0047 | 0.0258 | 0.0152 [0.0011, 0.0372] | 0.1311 | 0.0034 | 0.0673 [0.0017, 0.1743] | 0.9965 [0.9899, 1.0000] |
| B' | 43 | 03 | 3362 | 5761 | 16 | 0.6407 | 0.3348 | 0.4878 [0.3378, 0.6346] | 0.7912 | 0.1449 | 0.4681 [0.3512, 0.5618] | 0.4416 [0.2345, 0.6818] |
| B' | 43 | all | 5105 | 7509 | 16 | 0.4253 | 0.2723 | 0.3488 [0.2367, 0.4547] | 0.5559 | 0.1225 | 0.3392 [0.2313, 0.4338] | 0.6433 [0.4763, 0.8223] |
| B' | 44 | 01 | 889 | 855 | 9 | 0.0079 | 0.0047 | 0.0063 [0.0009, 0.0147] | 0.0619 | 0.1240 | 0.0929 [0.0141, 0.2110] | 0.9978 [0.9953, 1.0000] |
| B' | 44 | 02 | 854 | 893 | 9 | 0.1440 | 0.0011 | 0.0726 [0.0019, 0.2147] | 0.1803 | 0.0112 | 0.0958 [0.0063, 0.2363] | 0.8845 [0.6774, 1.0000] |
| B' | 44 | 03 | 3362 | 5761 | 16 | 0.7067 | 0.2376 | 0.4722 [0.2973, 0.6112] | 0.6231 | 0.2250 | 0.4241 [0.2579, 0.5793] | 0.5321 [0.3275, 0.7691] |
| B' | 44 | all | 5105 | 7509 | 16 | 0.4909 | 0.1830 | 0.3369 [0.2021, 0.4565] | 0.4513 | 0.1880 | 0.3197 [0.1816, 0.4338] | 0.6894 [0.5262, 0.8772] |


## Per arm, fold and seed (held-out subjects, by session stratum)

| arm | fold | seed | stratum | n bona fide | n attack | subjects | BPCER 05 | APCER 05 | ACER 05 | BPCER dev | APCER dev | ACER dev | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B | 1 | 42 | 01 | 254 | 287 | 3 | 0.0197 | 0.0000 | 0.0098 | 0.1890 | 0.0000 | 0.0945 | 1.0000 |
| B | 1 | 42 | 02 | 244 | 300 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B | 1 | 42 | 03 | 925 | 1232 | 4 | 0.3265 | 0.0219 | 0.1742 | 0.3795 | 0.0032 | 0.1914 | 0.8089 |
| B | 1 | 42 | all | 1423 | 1819 | 4 | 0.2157 | 0.0148 | 0.1153 | 0.2804 | 0.0022 | 0.1413 | 0.9106 |
| B | 1 | 43 | 01 | 254 | 287 | 3 | 0.0827 | 0.0000 | 0.0413 | 0.1457 | 0.0000 | 0.0728 | 0.9857 |
| B | 1 | 43 | 02 | 244 | 300 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B | 1 | 43 | 03 | 925 | 1232 | 4 | 0.9978 | 0.0722 | 0.5350 | 1.0000 | 0.0252 | 0.5126 | 0.2599 |
| B | 1 | 43 | all | 1423 | 1819 | 4 | 0.6634 | 0.0489 | 0.3562 | 0.6760 | 0.0170 | 0.3465 | 0.6319 |
| B | 1 | 44 | 01 | 254 | 287 | 3 | 0.1220 | 0.0000 | 0.0610 | 0.2677 | 0.0000 | 0.1339 | 1.0000 |
| B | 1 | 44 | 02 | 244 | 300 | 3 | 0.0123 | 0.0000 | 0.0061 | 0.0287 | 0.0000 | 0.0143 | 1.0000 |
| B | 1 | 44 | 03 | 925 | 1232 | 4 | 0.8205 | 0.0398 | 0.4302 | 0.9243 | 0.0195 | 0.4719 | 0.8101 |
| B | 1 | 44 | all | 1423 | 1819 | 4 | 0.5573 | 0.0269 | 0.2921 | 0.6535 | 0.0132 | 0.3334 | 0.9069 |
| B | 2 | 42 | 01 | 234 | 187 | 2 | 0.0043 | 0.0000 | 0.0021 | 0.0043 | 0.0000 | 0.0021 | 1.0000 |
| B | 2 | 42 | 02 | 117 | 193 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B | 2 | 42 | 03 | 1019 | 1364 | 4 | 0.0393 | 0.2749 | 0.1571 | 0.1865 | 0.0968 | 0.1416 | 0.9423 |
| B | 2 | 42 | all | 1370 | 1744 | 4 | 0.0299 | 0.2150 | 0.1225 | 0.1394 | 0.0757 | 0.1076 | 0.9660 |
| B | 2 | 43 | 01 | 234 | 187 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0043 | 0.0000 | 0.0021 | 1.0000 |
| B | 2 | 43 | 02 | 117 | 193 | 2 | 0.0000 | 0.0052 | 0.0026 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B | 2 | 43 | 03 | 1019 | 1364 | 4 | 0.6811 | 0.3204 | 0.5007 | 0.7301 | 0.0682 | 0.3992 | 0.3768 |
| B | 2 | 43 | all | 1370 | 1744 | 4 | 0.5066 | 0.2511 | 0.3789 | 0.5438 | 0.0533 | 0.2986 | 0.5875 |
| B | 2 | 44 | 01 | 234 | 187 | 2 | 0.0043 | 0.0000 | 0.0021 | 0.0043 | 0.0000 | 0.0021 | 1.0000 |
| B | 2 | 44 | 02 | 117 | 193 | 2 | 0.0085 | 0.0000 | 0.0043 | 0.0085 | 0.0000 | 0.0043 | 1.0000 |
| B | 2 | 44 | 03 | 1019 | 1364 | 4 | 0.2483 | 0.2588 | 0.2535 | 0.2296 | 0.2625 | 0.2461 | 0.8187 |
| B | 2 | 44 | all | 1370 | 1744 | 4 | 0.1861 | 0.2024 | 0.1943 | 0.1723 | 0.2053 | 0.1888 | 0.8815 |
| B | 3 | 42 | 01 | 276 | 187 | 2 | 0.0036 | 0.0000 | 0.0018 | 0.0036 | 0.0000 | 0.0018 | 1.0000 |
| B | 3 | 42 | 02 | 246 | 200 | 2 | 0.0000 | 0.0050 | 0.0025 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B | 3 | 42 | 03 | 865 | 1665 | 4 | 0.2289 | 0.2943 | 0.2616 | 0.3942 | 0.1399 | 0.2671 | 0.8561 |
| B | 3 | 42 | all | 1387 | 2052 | 4 | 0.1435 | 0.2393 | 0.1914 | 0.2466 | 0.1135 | 0.1801 | 0.9265 |
| B | 3 | 43 | 01 | 276 | 187 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B | 3 | 43 | 02 | 246 | 200 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B | 3 | 43 | 03 | 865 | 1665 | 4 | 0.4855 | 0.3363 | 0.4109 | 0.5017 | 0.1544 | 0.3280 | 0.5977 |
| B | 3 | 43 | all | 1387 | 2052 | 4 | 0.3028 | 0.2729 | 0.2879 | 0.3129 | 0.1252 | 0.2191 | 0.7909 |
| B | 3 | 44 | 01 | 276 | 187 | 2 | 0.0109 | 0.0000 | 0.0054 | 0.0109 | 0.0000 | 0.0054 | 0.9963 |
| B | 3 | 44 | 02 | 246 | 200 | 2 | 0.0000 | 0.0050 | 0.0025 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B | 3 | 44 | 03 | 865 | 1665 | 4 | 0.4220 | 0.3886 | 0.4053 | 0.4948 | 0.0829 | 0.2888 | 0.6817 |
| B | 3 | 44 | all | 1387 | 2052 | 4 | 0.2653 | 0.3158 | 0.2906 | 0.3107 | 0.0673 | 0.1890 | 0.8325 |
| B | 4 | 42 | 01 | 125 | 194 | 2 | 0.0000 | 0.4897 | 0.2448 | 0.0000 | 0.2371 | 0.1186 | 1.0000 |
| B | 4 | 42 | 02 | 247 | 200 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0081 | 0.0000 | 0.0040 | 1.0000 |
| B | 4 | 42 | 03 | 553 | 1500 | 4 | 0.8608 | 0.4547 | 0.6577 | 0.8626 | 0.2173 | 0.5400 | 0.1991 |
| B | 4 | 42 | all | 925 | 1894 | 4 | 0.5146 | 0.4102 | 0.4624 | 0.5178 | 0.1964 | 0.3571 | 0.5651 |
| B | 4 | 43 | 01 | 125 | 194 | 2 | 0.0000 | 0.2165 | 0.1082 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B | 4 | 43 | 02 | 247 | 200 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B | 4 | 43 | 03 | 553 | 1500 | 4 | 0.8626 | 0.4007 | 0.6316 | 0.8626 | 0.0747 | 0.4686 | 0.2230 |
| B | 4 | 43 | all | 925 | 1894 | 4 | 0.5157 | 0.3395 | 0.4276 | 0.5157 | 0.0591 | 0.2874 | 0.5968 |
| B | 4 | 44 | 01 | 125 | 194 | 2 | 0.0000 | 0.4845 | 0.2423 | 0.0000 | 0.4845 | 0.2423 | 1.0000 |
| B | 4 | 44 | 02 | 247 | 200 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B | 4 | 44 | 03 | 553 | 1500 | 4 | 0.8391 | 0.5420 | 0.6905 | 0.8571 | 0.3867 | 0.6219 | 0.2757 |
| B | 4 | 44 | all | 925 | 1894 | 4 | 0.5016 | 0.4789 | 0.4903 | 0.5124 | 0.3559 | 0.4341 | 0.6261 |
| A* | 1 | 42 | 01 | 254 | 287 | 3 | 0.0197 | 0.0000 | 0.0098 | 0.0000 | 0.0035 | 0.0017 | 1.0000 |
| A* | 1 | 42 | 02 | 244 | 300 | 3 | 0.0123 | 0.0000 | 0.0061 | 0.0041 | 0.0000 | 0.0020 | 1.0000 |
| A* | 1 | 42 | 03 | 925 | 1232 | 4 | 0.2292 | 0.0195 | 0.1243 | 0.0681 | 0.2037 | 0.1359 | 0.9647 |
| A* | 1 | 42 | all | 1423 | 1819 | 4 | 0.1546 | 0.0132 | 0.0839 | 0.0450 | 0.1385 | 0.0918 | 0.9828 |
| A* | 1 | 43 | 01 | 254 | 287 | 3 | 0.3583 | 0.0000 | 0.1791 | 0.3110 | 0.0000 | 0.1555 | 0.9901 |
| A* | 1 | 43 | 02 | 244 | 300 | 3 | 0.0123 | 0.0000 | 0.0061 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| A* | 1 | 43 | 03 | 925 | 1232 | 4 | 0.0562 | 0.0000 | 0.0281 | 0.0443 | 0.0041 | 0.0242 | 0.9909 |
| A* | 1 | 43 | all | 1423 | 1819 | 4 | 0.1026 | 0.0000 | 0.0513 | 0.0843 | 0.0027 | 0.0435 | 0.9921 |
| A* | 1 | 44 | 01 | 254 | 287 | 3 | 0.4291 | 0.0000 | 0.2146 | 0.4291 | 0.0000 | 0.2146 | 1.0000 |
| A* | 1 | 44 | 02 | 244 | 300 | 3 | 0.3730 | 0.0000 | 0.1865 | 0.1516 | 0.0000 | 0.0758 | 1.0000 |
| A* | 1 | 44 | 03 | 925 | 1232 | 4 | 0.7449 | 0.0000 | 0.3724 | 0.5784 | 0.0000 | 0.2892 | 0.5214 |
| A* | 1 | 44 | all | 1423 | 1819 | 4 | 0.6247 | 0.0000 | 0.3124 | 0.4786 | 0.0000 | 0.2393 | 0.7368 |
| A* | 2 | 42 | 01 | 234 | 187 | 2 | 0.0043 | 0.0000 | 0.0021 | 0.0043 | 0.0000 | 0.0021 | 1.0000 |
| A* | 2 | 42 | 02 | 117 | 193 | 2 | 0.0085 | 0.0000 | 0.0043 | 0.0085 | 0.0000 | 0.0043 | 1.0000 |
| A* | 2 | 42 | 03 | 1019 | 1364 | 4 | 0.1982 | 0.0139 | 0.1061 | 0.6575 | 0.0000 | 0.3288 | 0.9700 |
| A* | 2 | 42 | all | 1370 | 1744 | 4 | 0.1489 | 0.0109 | 0.0799 | 0.4905 | 0.0000 | 0.2453 | 0.9823 |
| A* | 2 | 43 | 01 | 234 | 187 | 2 | 0.0043 | 0.0000 | 0.0021 | 0.0043 | 0.0000 | 0.0021 | 1.0000 |
| A* | 2 | 43 | 02 | 117 | 193 | 2 | 0.0085 | 0.0000 | 0.0043 | 0.0085 | 0.0000 | 0.0043 | 0.9993 |
| A* | 2 | 43 | 03 | 1019 | 1364 | 4 | 0.0186 | 0.2749 | 0.1468 | 0.0687 | 0.0850 | 0.0769 | 0.9790 |
| A* | 2 | 43 | all | 1370 | 1744 | 4 | 0.0153 | 0.2150 | 0.1152 | 0.0526 | 0.0665 | 0.0595 | 0.9866 |
| A* | 2 | 44 | 01 | 234 | 187 | 2 | 0.0043 | 0.0053 | 0.0048 | 0.0043 | 0.0000 | 0.0021 | 1.0000 |
| A* | 2 | 44 | 02 | 117 | 193 | 2 | 0.0085 | 0.0052 | 0.0069 | 0.0085 | 0.0052 | 0.0069 | 0.9947 |
| A* | 2 | 44 | 03 | 1019 | 1364 | 4 | 0.0059 | 0.1048 | 0.0554 | 0.1197 | 0.0037 | 0.0617 | 0.9957 |
| A* | 2 | 44 | all | 1370 | 1744 | 4 | 0.0058 | 0.0831 | 0.0445 | 0.0905 | 0.0034 | 0.0470 | 0.9963 |
| A* | 3 | 42 | 01 | 276 | 187 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| A* | 3 | 42 | 02 | 246 | 200 | 2 | 0.0000 | 0.0850 | 0.0425 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| A* | 3 | 42 | 03 | 865 | 1665 | 4 | 0.0000 | 0.6042 | 0.3021 | 0.0243 | 0.2420 | 0.1332 | 0.9729 |
| A* | 3 | 42 | all | 1387 | 2052 | 4 | 0.0000 | 0.4985 | 0.2493 | 0.0151 | 0.1964 | 0.1058 | 0.9818 |
| A* | 3 | 43 | 01 | 276 | 187 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0535 | 0.0267 | 1.0000 |
| A* | 3 | 43 | 02 | 246 | 200 | 2 | 0.0000 | 0.0150 | 0.0075 | 0.0000 | 0.2200 | 0.1100 | 1.0000 |
| A* | 3 | 43 | 03 | 865 | 1665 | 4 | 0.0717 | 0.4096 | 0.2406 | 0.0000 | 0.6865 | 0.3432 | 0.8745 |
| A* | 3 | 43 | all | 1387 | 2052 | 4 | 0.0447 | 0.3338 | 0.1893 | 0.0000 | 0.5833 | 0.2917 | 0.9362 |
| A* | 3 | 44 | 01 | 276 | 187 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| A* | 3 | 44 | 02 | 246 | 200 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| A* | 3 | 44 | 03 | 865 | 1665 | 4 | 0.0832 | 0.0498 | 0.0665 | 0.0000 | 0.1826 | 0.0913 | 0.9868 |
| A* | 3 | 44 | all | 1387 | 2052 | 4 | 0.0519 | 0.0404 | 0.0462 | 0.0000 | 0.1481 | 0.0741 | 0.9933 |
| A* | 4 | 42 | 01 | 125 | 194 | 2 | 0.0000 | 0.0052 | 0.0026 | 0.0000 | 0.0103 | 0.0052 | 1.0000 |
| A* | 4 | 42 | 02 | 247 | 200 | 2 | 0.0121 | 0.1500 | 0.0811 | 0.0040 | 0.1800 | 0.0920 | 0.9864 |
| A* | 4 | 42 | 03 | 553 | 1500 | 4 | 0.0000 | 0.4447 | 0.2223 | 0.0000 | 0.4547 | 0.2273 | 0.9866 |
| A* | 4 | 42 | all | 925 | 1894 | 4 | 0.0032 | 0.3685 | 0.1859 | 0.0011 | 0.3801 | 0.1906 | 0.9581 |
| A* | 4 | 43 | 01 | 125 | 194 | 2 | 0.0000 | 0.0052 | 0.0026 | 0.0000 | 0.0052 | 0.0026 | 1.0000 |
| A* | 4 | 43 | 02 | 247 | 200 | 2 | 0.0000 | 0.0050 | 0.0025 | 0.0000 | 0.0050 | 0.0025 | 1.0000 |
| A* | 4 | 43 | 03 | 553 | 1500 | 4 | 0.0108 | 0.4393 | 0.2251 | 0.0108 | 0.4367 | 0.2238 | 0.8788 |
| A* | 4 | 43 | all | 925 | 1894 | 4 | 0.0065 | 0.3490 | 0.1777 | 0.0065 | 0.3469 | 0.1767 | 0.9227 |
| A* | 4 | 44 | 01 | 125 | 194 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| A* | 4 | 44 | 02 | 247 | 200 | 2 | 0.0162 | 0.0000 | 0.0081 | 0.0162 | 0.0000 | 0.0081 | 1.0000 |
| A* | 4 | 44 | 03 | 553 | 1500 | 4 | 0.0000 | 0.2840 | 0.1420 | 0.0000 | 0.2900 | 0.1450 | 0.9802 |
| A* | 4 | 44 | all | 925 | 1894 | 4 | 0.0043 | 0.2249 | 0.1146 | 0.0043 | 0.2297 | 0.1170 | 0.9714 |
| B' | 1 | 42 | 01 | 254 | 287 | 3 | 0.4094 | 0.0000 | 0.2047 | 0.0315 | 0.0000 | 0.0157 | 0.9924 |
| B' | 1 | 42 | 02 | 244 | 300 | 3 | 0.0082 | 0.0000 | 0.0041 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B' | 1 | 42 | 03 | 925 | 1232 | 4 | 1.0000 | 0.0016 | 0.5008 | 0.8649 | 0.0365 | 0.4507 | 0.4740 |
| B' | 1 | 42 | all | 1423 | 1819 | 4 | 0.7245 | 0.0011 | 0.3628 | 0.5678 | 0.0247 | 0.2963 | 0.6923 |
| B' | 1 | 43 | 01 | 254 | 287 | 3 | 0.0433 | 0.0105 | 0.0269 | 0.2520 | 0.0035 | 0.1277 | 0.9803 |
| B' | 1 | 43 | 02 | 244 | 300 | 3 | 0.0000 | 0.0133 | 0.0067 | 0.0000 | 0.0100 | 0.0050 | 0.9901 |
| B' | 1 | 43 | 03 | 925 | 1232 | 4 | 0.5524 | 0.3774 | 0.4649 | 0.9168 | 0.1786 | 0.5477 | 0.4227 |
| B' | 1 | 43 | all | 1423 | 1819 | 4 | 0.3668 | 0.2595 | 0.3132 | 0.6409 | 0.1231 | 0.3820 | 0.6517 |
| B' | 1 | 44 | 01 | 254 | 287 | 3 | 0.0236 | 0.0000 | 0.0118 | 0.2126 | 0.0000 | 0.1063 | 0.9980 |
| B' | 1 | 44 | 02 | 244 | 300 | 3 | 0.4959 | 0.0000 | 0.2480 | 0.6230 | 0.0000 | 0.3115 | 0.7602 |
| B' | 1 | 44 | 03 | 925 | 1232 | 4 | 0.9957 | 0.1981 | 0.5969 | 1.0000 | 0.0901 | 0.5450 | 0.3621 |
| B' | 1 | 44 | all | 1423 | 1819 | 4 | 0.7365 | 0.1341 | 0.4353 | 0.7948 | 0.0610 | 0.4279 | 0.5552 |
| B' | 2 | 42 | 01 | 234 | 187 | 2 | 0.5299 | 0.1123 | 0.3211 | 0.6325 | 0.1016 | 0.3670 | 0.8158 |
| B' | 2 | 42 | 02 | 117 | 193 | 2 | 0.0000 | 0.4456 | 0.2228 | 0.0256 | 0.3627 | 0.1942 | 0.6880 |
| B' | 2 | 42 | 03 | 1019 | 1364 | 4 | 0.0510 | 0.4296 | 0.2403 | 0.7468 | 0.3021 | 0.5244 | 0.6699 |
| B' | 2 | 42 | all | 1370 | 1744 | 4 | 0.1285 | 0.3974 | 0.2629 | 0.6657 | 0.2873 | 0.4765 | 0.6590 |
| B' | 2 | 43 | 01 | 234 | 187 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B' | 2 | 43 | 02 | 117 | 193 | 2 | 0.0000 | 0.0984 | 0.0492 | 0.0171 | 0.0000 | 0.0085 | 1.0000 |
| B' | 2 | 43 | 03 | 1019 | 1364 | 4 | 0.6565 | 0.4846 | 0.5706 | 0.8165 | 0.0352 | 0.4258 | 0.4985 |
| B' | 2 | 43 | all | 1370 | 1744 | 4 | 0.4883 | 0.3899 | 0.4391 | 0.6088 | 0.0275 | 0.3181 | 0.6793 |
| B' | 2 | 44 | 01 | 234 | 187 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B' | 2 | 44 | 02 | 117 | 193 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0085 | 0.0000 | 0.0043 | 1.0000 |
| B' | 2 | 44 | 03 | 1019 | 1364 | 4 | 0.5172 | 0.4472 | 0.4822 | 0.6595 | 0.0828 | 0.3712 | 0.6014 |
| B' | 2 | 44 | all | 1370 | 1744 | 4 | 0.3847 | 0.3498 | 0.3672 | 0.4912 | 0.0648 | 0.2780 | 0.7655 |
| B' | 3 | 42 | 01 | 276 | 187 | 2 | 0.0036 | 0.0000 | 0.0018 | 0.0036 | 0.0000 | 0.0018 | 1.0000 |
| B' | 3 | 42 | 02 | 246 | 200 | 2 | 0.0081 | 0.0000 | 0.0041 | 0.0041 | 0.0000 | 0.0020 | 1.0000 |
| B' | 3 | 42 | 03 | 865 | 1665 | 4 | 0.5607 | 0.2294 | 0.3951 | 0.5514 | 0.2871 | 0.4193 | 0.5157 |
| B' | 3 | 42 | all | 1387 | 2052 | 4 | 0.3518 | 0.1862 | 0.2690 | 0.3453 | 0.2329 | 0.2891 | 0.7327 |
| B' | 3 | 43 | 01 | 276 | 187 | 2 | 0.0072 | 0.0000 | 0.0036 | 0.0072 | 0.0000 | 0.0036 | 0.9964 |
| B' | 3 | 43 | 02 | 246 | 200 | 2 | 0.0163 | 0.0000 | 0.0081 | 0.4472 | 0.0000 | 0.2236 | 1.0000 |
| B' | 3 | 43 | 03 | 865 | 1665 | 4 | 0.5746 | 0.2222 | 0.3984 | 0.5815 | 0.1201 | 0.3508 | 0.4509 |
| B' | 3 | 43 | all | 1387 | 2052 | 4 | 0.3627 | 0.1803 | 0.2715 | 0.4434 | 0.0975 | 0.2704 | 0.6528 |
| B' | 3 | 44 | 01 | 276 | 187 | 2 | 0.0036 | 0.0000 | 0.0018 | 0.0036 | 0.0000 | 0.0018 | 1.0000 |
| B' | 3 | 44 | 02 | 246 | 200 | 2 | 0.0041 | 0.0000 | 0.0020 | 0.0041 | 0.0000 | 0.0020 | 1.0000 |
| B' | 3 | 44 | 03 | 865 | 1665 | 4 | 0.5353 | 0.0072 | 0.2712 | 0.5006 | 0.0414 | 0.2710 | 0.7566 |
| B' | 3 | 44 | all | 1387 | 2052 | 4 | 0.3353 | 0.0058 | 0.1706 | 0.3136 | 0.0336 | 0.1736 | 0.8682 |
| B' | 4 | 42 | 01 | 125 | 194 | 2 | 0.0000 | 0.4021 | 0.2010 | 0.0000 | 0.4381 | 0.2191 | 1.0000 |
| B' | 4 | 42 | 02 | 247 | 200 | 2 | 0.0324 | 0.0000 | 0.0162 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B' | 4 | 42 | 03 | 553 | 1500 | 4 | 0.6926 | 0.2760 | 0.4843 | 0.2911 | 0.4993 | 0.3952 | 0.6259 |
| B' | 4 | 42 | all | 925 | 1894 | 4 | 0.4227 | 0.2598 | 0.3412 | 0.1741 | 0.4403 | 0.3072 | 0.7667 |
| B' | 4 | 43 | 01 | 125 | 194 | 2 | 0.0000 | 0.4639 | 0.2320 | 0.0000 | 0.4175 | 0.2088 | 1.0000 |
| B' | 4 | 43 | 02 | 247 | 200 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B' | 4 | 43 | 03 | 553 | 1500 | 4 | 0.8626 | 0.2887 | 0.5756 | 0.8626 | 0.2447 | 0.5536 | 0.1470 |
| B' | 4 | 43 | all | 925 | 1894 | 4 | 0.5157 | 0.2761 | 0.3959 | 0.5157 | 0.2365 | 0.3761 | 0.4826 |
| B' | 4 | 44 | 01 | 125 | 194 | 2 | 0.0000 | 0.0206 | 0.0103 | 0.0000 | 0.5464 | 0.2732 | 1.0000 |
| B' | 4 | 44 | 02 | 247 | 200 | 2 | 0.0040 | 0.0050 | 0.0045 | 0.0000 | 0.0500 | 0.0250 | 0.9999 |
| B' | 4 | 44 | 03 | 553 | 1500 | 4 | 0.8409 | 0.3353 | 0.5881 | 0.1175 | 0.6687 | 0.3931 | 0.5270 |
| B' | 4 | 44 | all | 925 | 1894 | 4 | 0.5038 | 0.2682 | 0.3860 | 0.0703 | 0.5908 | 0.3305 | 0.7471 |


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
| arm_a_star_fold1_seed42 | 1400 | 400 | 12 | 10 | end of epochs (set_weights) | 0.0104 | 470 | 0.895917 | 0.0000 | 0.0000 |
| arm_a_star_fold1_seed43 | 1400 | 400 | 12 | 9 | end of epochs (set_weights) | 0.0165 | 459 | 0.749943 | 0.0050 | 0.0050 |
| arm_a_star_fold1_seed44 | 1400 | 400 | 12 | 9 | end of epochs (set_weights) | 0.0269 | 434 | 0.915815 | 0.0050 | 0.0050 |
| arm_a_star_fold2_seed42 | 1400 | 400 | 12 | 7 | end of epochs (set_weights) | 0.0001 | 433 | 0.033502 | 0.0000 | 0.0000 |
| arm_a_star_fold2_seed43 | 1400 | 400 | 12 | 9 | end of epochs (set_weights) | 0.0051 | 441 | 0.035079 | 0.0000 | 0.0000 |
| arm_a_star_fold2_seed44 | 1400 | 400 | 12 | 10 | end of epochs (set_weights) | 0.0001 | 439 | 0.000268 | 0.0000 | 0.0000 |
| arm_a_star_fold3_seed42 | 1400 | 400 | 12 | 10 | end of epochs (set_weights) | 0.0256 | 442 | 0.002318 | 0.0000 | 0.0000 |
| arm_a_star_fold3_seed43 | 1400 | 400 | 12 | 12 | end of epochs (set_weights) | 0.0976 | 445 | 0.963851 | 0.0250 | 0.0250 |
| arm_a_star_fold3_seed44 | 1400 | 400 | 12 | 8 | end of epochs (set_weights) | 0.2450 | 445 | 0.976119 | 0.0000 | 0.0000 |
| arm_a_star_fold4_seed42 | 1400 | 400 | 12 | 11 | end of epochs (set_weights) | 0.2092 | 439 | 0.572169 | 0.1000 | 0.1000 |
| arm_a_star_fold4_seed43 | 1400 | 400 | 12 | 9 | end of epochs (set_weights) | 0.0272 | 439 | 0.477774 | 0.0050 | 0.0050 |
| arm_a_star_fold4_seed44 | 1400 | 400 | 12 | 8 | end of epochs (set_weights) | 0.0123 | 449 | 0.527478 | 0.0000 | 0.0000 |
| arm_b_fold1_seed42 | 1400 | 400 | 12 | 9 | end of epochs (set_weights) | 0.0003 | 383 | 0.076832 | 0.0000 | 0.0000 |
| arm_b_fold1_seed43 | 1400 | 400 | 12 | 9 | end of epochs (set_weights) | 0.0022 | 469 | 0.260548 | 0.0000 | 0.0000 |
| arm_b_fold1_seed44 | 1400 | 400 | 12 | 11 | end of epochs (set_weights) | 0.0007 | 461 | 0.168149 | 0.0000 | 0.0000 |
| arm_b_fold2_seed42 | 1400 | 400 | 12 | 10 | end of epochs (set_weights) | 0.0125 | 438 | 0.117119 | 0.0000 | 0.0000 |
| arm_b_fold2_seed43 | 1400 | 400 | 12 | 7 | end of epochs (set_weights) | 0.0009 | 441 | 0.017887 | 0.0000 | 0.0000 |
| arm_b_fold2_seed44 | 1400 | 400 | 12 | 9 | end of epochs (set_weights) | 0.0022 | 434 | 0.553873 | 0.0000 | 0.0000 |
| arm_b_fold3_seed42 | 1400 | 400 | 12 | 12 | end of epochs (set_weights) | 0.0039 | 450 | 0.129490 | 0.0000 | 0.0000 |
| arm_b_fold3_seed43 | 1400 | 400 | 12 | 9 | end of epochs (set_weights) | 0.1083 | 432 | 0.093905 | 0.0050 | 0.0050 |
| arm_b_fold3_seed44 | 1400 | 400 | 12 | 10 | end of epochs (set_weights) | 0.1457 | 442 | 0.005974 | 0.0000 | 0.0000 |
| arm_b_fold4_seed42 | 1400 | 400 | 12 | 11 | end of epochs (set_weights) | 0.0001 | 434 | 0.006737 | 0.0000 | 0.0000 |
| arm_b_fold4_seed43 | 1400 | 400 | 12 | 6 | end of epochs (set_weights) | 0.0013 | 434 | 0.002956 | 0.0000 | 0.0000 |
| arm_b_fold4_seed44 | 1400 | 400 | 12 | 11 | end of epochs (set_weights) | 0.0037 | 441 | 0.112040 | 0.0000 | 0.0000 |
| arm_b_prime_fold1_seed42 | 934 | 400 | 12 | 10 | end of epochs (set_weights) | 0.0144 | 305 | 0.988504 | 0.0050 | 0.0050 |
| arm_b_prime_fold1_seed43 | 934 | 400 | 12 | 12 | end of epochs (set_weights) | 0.0642 | 306 | 0.005476 | 0.0000 | 0.0000 |
| arm_b_prime_fold1_seed44 | 934 | 400 | 12 | 6 | end of epochs (set_weights) | 0.3141 | 313 | 0.000000 | 0.0050 | 0.0000 |
| arm_b_prime_fold2_seed42 | 934 | 400 | 12 | 10 | end of epochs (set_weights) | 1.6847 | 319 | 0.016567 | 0.1550 | 0.1550 |
| arm_b_prime_fold2_seed43 | 934 | 400 | 12 | 12 | end of epochs (set_weights) | 0.0217 | 308 | 0.000089 | 0.0000 | 0.0000 |
| arm_b_prime_fold2_seed44 | 934 | 400 | 12 | 12 | end of epochs (set_weights) | 0.0160 | 307 | 0.013936 | 0.0000 | 0.0000 |
| arm_b_prime_fold3_seed42 | 934 | 400 | 12 | 4 | end of epochs (set_weights) | 0.0260 | 311 | 0.719105 | 0.0000 | 0.0000 |
| arm_b_prime_fold3_seed43 | 934 | 400 | 12 | 9 | end of epochs (set_weights) | 0.0011 | 308 | 0.013271 | 0.0000 | 0.0000 |
| arm_b_prime_fold3_seed44 | 934 | 400 | 12 | 9 | end of epochs (set_weights) | 0.0814 | 319 | 0.911178 | 0.0000 | 0.0000 |
| arm_b_prime_fold4_seed42 | 934 | 400 | 12 | 11 | end of epochs (set_weights) | 0.0968 | 309 | 0.861657 | 0.0100 | 0.0100 |
| arm_b_prime_fold4_seed43 | 934 | 400 | 12 | 3 | end of epochs (set_weights) | 0.0142 | 308 | 0.322349 | 0.0000 | 0.0000 |
| arm_b_prime_fold4_seed44 | 934 | 400 | 12 | 10 | end of epochs (set_weights) | 0.2785 | 322 | 0.944654 | 0.0750 | 0.0750 |


## Steps, budget, environment and inputs

| step | seconds (wall clock) | seconds (monotonic) |
|---|---|---|
| check-inputs | 4 | 4 |
| train arm_b_fold1_seed42 | 384 | 384 |
| threshold arm_b_fold1_seed42 | 4 | 4 |
| score arm_b_fold1_seed42 | 18 | 18 |
| train arm_a_star_fold1_seed42 | 472 | 472 |
| threshold arm_a_star_fold1_seed42 | 5 | 5 |
| score arm_a_star_fold1_seed42 | 18 | 18 |
| train arm_b_fold1_seed43 | 470 | 470 |
| threshold arm_b_fold1_seed43 | 5 | 5 |
| score arm_b_fold1_seed43 | 18 | 18 |
| train arm_a_star_fold1_seed43 | 461 | 461 |
| threshold arm_a_star_fold1_seed43 | 5 | 5 |
| score arm_a_star_fold1_seed43 | 18 | 18 |
| train arm_b_fold1_seed44 | 462 | 462 |
| threshold arm_b_fold1_seed44 | 4 | 4 |
| score arm_b_fold1_seed44 | 18 | 18 |
| train arm_a_star_fold1_seed44 | 435 | 435 |
| threshold arm_a_star_fold1_seed44 | 4 | 4 |
| score arm_a_star_fold1_seed44 | 18 | 18 |
| train arm_b_fold2_seed42 | 438 | 438 |
| threshold arm_b_fold2_seed42 | 5 | 5 |
| score arm_b_fold2_seed42 | 18 | 18 |
| train arm_a_star_fold2_seed42 | 434 | 434 |
| threshold arm_a_star_fold2_seed42 | 5 | 5 |
| score arm_a_star_fold2_seed42 | 17 | 17 |
| train arm_b_fold2_seed43 | 442 | 442 |
| threshold arm_b_fold2_seed43 | 5 | 5 |
| score arm_b_fold2_seed43 | 18 | 18 |
| train arm_a_star_fold2_seed43 | 443 | 443 |
| threshold arm_a_star_fold2_seed43 | 5 | 5 |
| score arm_a_star_fold2_seed43 | 17 | 17 |
| train arm_b_fold2_seed44 | 435 | 435 |
| threshold arm_b_fold2_seed44 | 5 | 5 |
| score arm_b_fold2_seed44 | 17 | 17 |
| train arm_a_star_fold2_seed44 | 440 | 440 |
| threshold arm_a_star_fold2_seed44 | 5 | 5 |
| score arm_a_star_fold2_seed44 | 18 | 18 |
| train arm_b_fold3_seed42 | 451 | 451 |
| threshold arm_b_fold3_seed42 | 5 | 5 |
| score arm_b_fold3_seed42 | 21 | 21 |
| train arm_a_star_fold3_seed42 | 444 | 444 |
| threshold arm_a_star_fold3_seed42 | 4 | 4 |
| score arm_a_star_fold3_seed42 | 20 | 20 |
| train arm_b_fold3_seed43 | 434 | 434 |
| threshold arm_b_fold3_seed43 | 4 | 4 |
| score arm_b_fold3_seed43 | 19 | 19 |
| train arm_a_star_fold3_seed43 | 446 | 446 |
| threshold arm_a_star_fold3_seed43 | 5 | 5 |
| score arm_a_star_fold3_seed43 | 20 | 20 |
| train arm_b_fold3_seed44 | 444 | 444 |
| threshold arm_b_fold3_seed44 | 5 | 5 |
| score arm_b_fold3_seed44 | 19 | 19 |
| train arm_a_star_fold3_seed44 | 446 | 446 |
| threshold arm_a_star_fold3_seed44 | 5 | 5 |
| score arm_a_star_fold3_seed44 | 19 | 19 |
| train arm_b_fold4_seed42 | 436 | 436 |
| threshold arm_b_fold4_seed42 | 4 | 4 |
| score arm_b_fold4_seed42 | 17 | 17 |
| train arm_a_star_fold4_seed42 | 440 | 440 |
| threshold arm_a_star_fold4_seed42 | 5 | 5 |
| score arm_a_star_fold4_seed42 | 16 | 16 |
| train arm_b_fold4_seed43 | 435 | 435 |
| threshold arm_b_fold4_seed43 | 5 | 5 |
| score arm_b_fold4_seed43 | 16 | 16 |
| train arm_a_star_fold4_seed43 | 440 | 440 |
| threshold arm_a_star_fold4_seed43 | 5 | 5 |
| score arm_a_star_fold4_seed43 | 16 | 16 |
| train arm_b_fold4_seed44 | 442 | 442 |
| threshold arm_b_fold4_seed44 | 5 | 5 |
| score arm_b_fold4_seed44 | 16 | 16 |
| train arm_a_star_fold4_seed44 | 450 | 450 |
| threshold arm_a_star_fold4_seed44 | 5 | 5 |
| score arm_a_star_fold4_seed44 | 16 | 16 |
| train arm_b_prime_fold1_seed42 | 306 | 306 |
| threshold arm_b_prime_fold1_seed42 | 5 | 5 |
| score arm_b_prime_fold1_seed42 | 20 | 20 |
| train arm_b_prime_fold1_seed43 | 308 | 308 |
| threshold arm_b_prime_fold1_seed43 | 4 | 4 |
| score arm_b_prime_fold1_seed43 | 19 | 19 |
| train arm_b_prime_fold1_seed44 | 314 | 314 |
| threshold arm_b_prime_fold1_seed44 | 6 | 6 |
| score arm_b_prime_fold1_seed44 | 20 | 20 |
| train arm_b_prime_fold2_seed42 | 321 | 321 |
| threshold arm_b_prime_fold2_seed42 | 5 | 5 |
| score arm_b_prime_fold2_seed42 | 19 | 19 |
| train arm_b_prime_fold2_seed43 | 309 | 309 |
| threshold arm_b_prime_fold2_seed43 | 5 | 5 |
| score arm_b_prime_fold2_seed43 | 18 | 18 |
| train arm_b_prime_fold2_seed44 | 308 | 308 |
| threshold arm_b_prime_fold2_seed44 | 5 | 5 |
| score arm_b_prime_fold2_seed44 | 18 | 18 |
| train arm_b_prime_fold3_seed42 | 312 | 312 |
| threshold arm_b_prime_fold3_seed42 | 5 | 5 |
| score arm_b_prime_fold3_seed42 | 21 | 21 |
| train arm_b_prime_fold3_seed43 | 309 | 309 |
| threshold arm_b_prime_fold3_seed43 | 5 | 5 |
| score arm_b_prime_fold3_seed43 | 20 | 20 |
| train arm_b_prime_fold3_seed44 | 320 | 320 |
| threshold arm_b_prime_fold3_seed44 | 5 | 5 |
| score arm_b_prime_fold3_seed44 | 19 | 19 |
| train arm_b_prime_fold4_seed42 | 310 | 310 |
| threshold arm_b_prime_fold4_seed42 | 5 | 5 |
| score arm_b_prime_fold4_seed42 | 17 | 17 |
| train arm_b_prime_fold4_seed43 | 309 | 309 |
| threshold arm_b_prime_fold4_seed43 | 5 | 5 |
| score arm_b_prime_fold4_seed43 | 17 | 17 |
| train arm_b_prime_fold4_seed44 | 324 | 324 |
| threshold arm_b_prime_fold4_seed44 | 5 | 5 |
| score arm_b_prime_fold4_seed44 | 16 | 16 |
| aggregate | 2 | 2 |

Wall time 15217 s of the cap 21600 s (checked before every training, at the end of every epoch, before every threshold and every test scoring; the aggregation and the final writes run after the last model, also after the cap; cap already exceeded before them: no). Environment: {"keras": "2.15.0", "numpy": "1.26.4", "opencv": "4.11.0", "platform": "darwin", "python": "3.11.16", "tensorflow": "2.15.0", "tensorflow-metal": "1.1.0", "tensorflow_installed": "2.15.0"}.

Inputs (SHA-256): cells run restricted SHA256SUMS `0904ee72546160eadb493a94d8489a8fb21687006cd8fe1970b26007c567723c`, fold schema restricted SHA256SUMS `f3b30ae75a7167646544ef524225749bbded99e55d2249b9415eef2481e0d0f1`, cache `dc286dc371e61d638da1842c062a38c5368a9ec4da07ccbc36b7fff6f3177920`, extracted frames `c3fd9903589b7b074c46880870ba87924763f244846c744b657f9aaa3ebeee77`, architecture `a83c02b0788bd6f98262abf00f5fe970abed501f231edba348820482aea64d30`.

## Limits

Bootstrap: generator numpy.random.default_rng(0) (PCG64), restarted for every seed, comparison (or arm) and stratum; the held-out subjects of the stratum (pooled over the four folds) in ascending identifier order; at every resample integers(0, n, n) with n = the number of subjects; every metric computed on the pooled frames of the resampled subjects (through per-subject counts and pairwise Mann-Whitney counts, identical to recomputing it on the frames); the same subject indices for both arms of a comparison and for every metric; percentiles 2.5 and 97.5 with np.percentile, linear interpolation; a resample without a class does not count for the metrics that need it; no interval with fewer than 5 subjects. the session-03 bona fide cell of S12 comes from 1-2 subjects per fold; two dev cells rest on one subject; frames of one subject and session are correlated (n counts frames); three seeds give a range; the four folds are not independent (the dev of a fold is the test of the next one); the dev without session 03 calibrates early stopping and the EER threshold on the session 01-02 distribution for every arm; the pooled strata hold 9 (sessions 01, 02) or 16 (03, all) subjects; one dataset, one attack type (print)
