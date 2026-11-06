# NUAA session-03 ablation: LivenessNet, arms B / A* / B', 4 subject folds x 3 seeds (aggregate, no identifiers)

Run `20261106-SESSABLRUN-d999458` (commit `d999458`, dirty tree: no), 2026-11-06T11:31:29+0100. Outcome: **PASS**, 36 of 36 trainings complete. Configuration `4c2d29d13ded`; cells run 20261104-SESSABL-5d89888, fold schema 20261103-FOLDS-67b7f3d, frames of 20261103-FOLDSRUN-e174812. Every number below is reported whatever its value (pre-commitment). Diagnostic, not causal; no statistical test.

Arms (same fitting subjects S12 per fold): B = sessions 01-02 (350/350/350/350 per cell, 1400 frames); A* = sessions 01-03 (234/233/233/234/233/233, 1400); B' = sessions 01-02 at the per-cell counts of A* (234/233/233/234, 934). Common dev: sessions 01-02 of the dev-fold subjects (400 frames), for early stopping, the learning-rate schedule, the restored epoch and the EER threshold. Test: every frame of the held-out subjects (the fold schema, unchanged).

Decision: attack if score_attack > threshold (tie = bona fide). `05` = the fixed threshold 0.5; `dev` = the EER threshold of the common dev (M1 criterion), frozen before any test scoring. AUROC: threshold-free (Mann-Whitney, ties 0.5). NA where a class is absent. Differences are minuend - subtrahend, paired on the same test frames and seeds (nominal pairing: batch order and augmentation differ with the data).

## Reading rule (fixed in the configuration before the run)

For each comparison, stratum and metric, the pooled difference (minuend - subtrahend) is 'consistent' only if it has the same non-zero sign for all three seeds AND the paired bootstrap interval of every seed excludes 0 on the side of that sign; otherwise 'inconclusive' (also when a seed has no pooled value, no difference or no interval); a consistent verdict states the direction (for example 'consistent: A* higher AUROC than B'); primary: AUROC and ACER at the dev threshold in stratum 03 of a_star_minus_b; the same rule, labelled secondary, for the other strata, the other metrics and the two secondary comparisons; no statistical test; applied mechanically by the script; every verdict reported.

## Primary verdicts (A* - B, stratum 03)

| metric | seed 42: difference [bootstrap] | seed 43: difference [bootstrap] | seed 44: difference [bootstrap] | verdict | reason |
|---|---|---|---|---|---|
| AUROC | 0.2658 [-0.0060, 0.4862] | 0.2064 [-0.0729, 0.4510] | -0.0103 [-0.2949, 0.2683] | inconclusive | the sign differs across the seeds |
| ACER (dev threshold) | -0.0854 [-0.3192, 0.1665] | -0.1703 [-0.3013, -0.0303] | -0.0008 [-0.1787, 0.2265] | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |


## Comparison A* - B (primary): pooled over the 4 folds, per seed

| seed | stratum | subjects | APCER (0.5): difference [bootstrap] | BPCER (0.5): difference [bootstrap] | ACER (0.5): difference [bootstrap] | APCER (dev threshold): difference [bootstrap] | BPCER (dev threshold): difference [bootstrap] | ACER (dev threshold): difference [bootstrap] | AUROC: difference [bootstrap] |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 01 | 9 | 0.1708 [-0.0965, 0.4521] | -0.2823 [-0.6667, -0.0420] | -0.0558 [-0.3366, 0.1880] | 0.1111 [0.0012, 0.3267] | 0.0776 [-0.2860, 0.3890] | 0.0944 [-0.1423, 0.3477] | 0.1219 [-0.1489, 0.2785] |
| 42 | 02 | 9 | 0.2240 [0.0000, 0.4576] | -0.1183 [-0.6007, 0.3345] | 0.0528 [-0.2408, 0.3352] | 0.0974 [0.0011, 0.2256] | 0.0843 [-0.2663, 0.4848] | 0.0909 [-0.1280, 0.3403] | 0.0427 [-0.2625, 0.2490] |
| 42 | 03 | 16 | 0.1981 [-0.0577, 0.4478] | -0.5892 [-0.8518, -0.2589] | -0.1956 [-0.3863, 0.0092] | 0.0457 [-0.1772, 0.3032] | -0.2165 [-0.6818, 0.3224] | -0.0854 [-0.3192, 0.1665] | 0.2658 [-0.0060, 0.4862] |
| 42 | all | 16 | 0.1980 [-0.0471, 0.4530] | -0.4570 [-0.7014, -0.1742] | -0.1295 [-0.2934, 0.0742] | 0.0593 [-0.1126, 0.2709] | -0.1150 [-0.4955, 0.2926] | -0.0279 [-0.2144, 0.1909] | 0.1833 [-0.0379, 0.3554] |
| 43 | 01 | 9 | -0.0480 [-0.1474, 0.0034] | 0.3138 [0.0000, 0.6212] | 0.1329 [-0.0239, 0.2942] | 0.2199 [0.0000, 0.5407] | 0.0439 [-0.0032, 0.1487] | 0.1319 [0.0017, 0.2717] | -0.2295 [-0.4459, 0.0010] |
| 43 | 02 | 9 | 0.0011 [0.0000, 0.0034] | 0.1370 [0.0000, 0.4317] | 0.0691 [0.0005, 0.2164] | 0.2172 [0.0000, 0.5436] | -0.0047 [-0.0107, 0.0000] | 0.1063 [-0.0018, 0.2666] | -0.1199 [-0.3419, 0.0000] |
| 43 | 03 | 16 | 0.0338 [-0.1490, 0.2317] | -0.4471 [-0.7671, -0.0775] | -0.2066 [-0.3568, -0.0288] | 0.4244 [0.2258, 0.6567] | -0.7650 [-0.9459, -0.4821] | -0.1703 [-0.3013, -0.0303] | 0.2064 [-0.0729, 0.4510] |
| 43 | all | 16 | 0.0206 [-0.1204, 0.1927] | -0.2168 [-0.5231, 0.0841] | -0.0981 [-0.2240, 0.0423] | 0.3765 [0.1851, 0.6240] | -0.4970 [-0.6999, -0.2630] | -0.0602 [-0.1612, 0.0586] | 0.0202 [-0.1948, 0.2297] |
| 44 | 01 | 9 | 0.0070 [-0.4209, 0.4379] | -0.1046 [-0.3459, 0.0028] | -0.0488 [-0.2690, 0.1730] | 0.3906 [0.1138, 0.6741] | -0.2846 [-0.7260, 0.2737] | 0.0530 [-0.2132, 0.3319] | -0.3071 [-0.5011, -0.0292] |
| 44 | 02 | 9 | -0.0011 [-0.4311, 0.4335] | 0.0890 [-0.0011, 0.2106] | 0.0439 [-0.1577, 0.2384] | 0.3449 [0.0948, 0.6189] | -0.0796 [-0.5504, 0.3781] | 0.1326 [-0.1321, 0.3728] | -0.3912 [-0.6089, -0.1349] |
| 44 | 03 | 16 | 0.0859 [-0.1800, 0.3979] | -0.2501 [-0.5135, 0.0579] | -0.0821 [-0.2470, 0.0797] | 0.3352 [0.1812, 0.5061] | -0.3367 [-0.7698, 0.1716] | -0.0008 [-0.1787, 0.2265] | -0.0103 [-0.2949, 0.2683] |
| 44 | all | 16 | 0.0666 [-0.2192, 0.3872] | -0.1681 [-0.3755, 0.0559] | -0.0507 [-0.1921, 0.0935] | 0.3427 [0.1909, 0.5040] | -0.2846 [-0.6546, 0.1582] | 0.0290 [-0.1282, 0.2205] | -0.1203 [-0.3305, 0.0987] |


Verdicts (A* - B; primary only for AUROC and ACER dev in stratum 03 of A* - B)

| stratum | metric | label | verdict | reason |
|---|---|---|---|---|
| 01 | APCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 01 | BPCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 01 | ACER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 01 | APCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 01 | BPCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 01 | ACER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 01 | AUROC | secondary | inconclusive | the sign differs across the seeds |
| 02 | APCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 02 | BPCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 02 | ACER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 02 | APCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 02 | BPCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 02 | ACER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 02 | AUROC | secondary | inconclusive | the sign differs across the seeds |
| 03 | APCER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | BPCER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | ACER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | APCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | BPCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | ACER (dev threshold) | primary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | AUROC | primary | inconclusive | the sign differs across the seeds |
| all | APCER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | BPCER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | ACER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | APCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | BPCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | ACER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| all | AUROC | secondary | inconclusive | the sign differs across the seeds |


Median and range over the seeds of the pooled differences

| stratum | APCER (0.5) | BPCER (0.5) | ACER (0.5) | APCER (dev threshold) | BPCER (dev threshold) | ACER (dev threshold) | AUROC |
|---|---|---|---|---|---|---|---|
| 01 | 0.0070 (-0.0480 to 0.1708) | -0.1046 (-0.2823 to 0.3138) | -0.0488 (-0.0558 to 0.1329) | 0.2199 (0.1111 to 0.3906) | 0.0439 (-0.2846 to 0.0776) | 0.0944 (0.0530 to 0.1319) | -0.2295 (-0.3071 to 0.1219) |
| 02 | 0.0011 (-0.0011 to 0.2240) | 0.0890 (-0.1183 to 0.1370) | 0.0528 (0.0439 to 0.0691) | 0.2172 (0.0974 to 0.3449) | -0.0047 (-0.0796 to 0.0843) | 0.1063 (0.0909 to 0.1326) | -0.1199 (-0.3912 to 0.0427) |
| 03 | 0.0859 (0.0338 to 0.1981) | -0.4471 (-0.5892 to -0.2501) | -0.1956 (-0.2066 to -0.0821) | 0.3352 (0.0457 to 0.4244) | -0.3367 (-0.7650 to -0.2165) | -0.0854 (-0.1703 to -0.0008) | 0.2064 (-0.0103 to 0.2658) |
| all | 0.0666 (0.0206 to 0.1980) | -0.2168 (-0.4570 to -0.1681) | -0.0981 (-0.1295 to -0.0507) | 0.3427 (0.0593 to 0.3765) | -0.2846 (-0.4970 to -0.1150) | -0.0279 (-0.0602 to 0.0290) | 0.0202 (-0.1203 to 0.1833) |


Per fold and seed (A* - B)

| fold | seed | stratum | APCER (0.5) | BPCER (0.5) | ACER (0.5) | APCER (dev threshold) | BPCER (dev threshold) | ACER (dev threshold) | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 42 | 01 | 0.0000 | -0.9803 | -0.4902 | 0.0035 | -0.4764 | -0.2364 | 0.0020 |
| 1 | 42 | 02 | 0.0000 | -0.9877 | -0.4939 | 0.0000 | -0.3484 | -0.1742 | 0.0143 |
| 1 | 42 | 03 | 0.0195 | -0.7708 | -0.3757 | 0.1071 | -0.4227 | -0.1578 | 0.3512 |
| 1 | 42 | all | 0.0132 | -0.8454 | -0.4161 | 0.0731 | -0.4195 | -0.1732 | 0.2439 |
| 1 | 43 | 01 | 0.0000 | 0.1772 | 0.0886 | 0.0000 | 0.1575 | 0.0787 | -0.0021 |
| 1 | 43 | 02 | 0.0000 | 0.0041 | 0.0020 | 0.0000 | -0.0082 | -0.0041 | 0.0000 |
| 1 | 43 | 03 | -0.0057 | -0.8811 | -0.4434 | -0.0032 | -0.8843 | -0.4438 | 0.2097 |
| 1 | 43 | all | -0.0038 | -0.5404 | -0.2721 | -0.0022 | -0.5481 | -0.2752 | 0.0973 |
| 1 | 44 | 01 | 0.0035 | -0.3661 | -0.1813 | 0.0000 | -0.3819 | -0.1909 | -0.0036 |
| 1 | 44 | 02 | 0.0000 | 0.1475 | 0.0738 | 0.0000 | 0.1844 | 0.0922 | -0.0245 |
| 1 | 44 | 03 | 0.0674 | -0.6670 | -0.2998 | 0.0373 | -0.6086 | -0.2857 | 0.4592 |
| 1 | 44 | all | 0.0462 | -0.4736 | -0.2137 | 0.0253 | -0.4322 | -0.2034 | 0.2564 |
| 2 | 42 | 01 | 0.0000 | -0.0043 | -0.0021 | 0.0000 | 0.0043 | 0.0021 | 0.0000 |
| 2 | 42 | 02 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0085 | 0.0043 | 0.0061 |
| 2 | 42 | 03 | -0.0565 | -0.4289 | -0.2427 | -0.5117 | 0.0903 | -0.2107 | 0.2264 |
| 2 | 42 | all | -0.0442 | -0.3197 | -0.1819 | -0.4002 | 0.0686 | -0.1658 | 0.1333 |
| 2 | 43 | 01 | 0.0000 | 1.0000 | 0.5000 | 1.0000 | -0.0043 | 0.4979 | -0.5000 |
| 2 | 43 | 02 | 0.0000 | 0.9915 | 0.4957 | 1.0000 | -0.0171 | 0.4915 | -0.4927 |
| 2 | 43 | 03 | -0.0880 | 0.1580 | 0.0350 | 0.9861 | -0.8852 | 0.0504 | 0.0256 |
| 2 | 43 | all | -0.0688 | 0.3730 | 0.1521 | 0.9891 | -0.6606 | 0.1643 | -0.1701 |
| 2 | 44 | 01 | 1.0000 | -0.0043 | 0.4979 | 0.5294 | 0.5128 | 0.5211 | -0.7589 |
| 2 | 44 | 02 | 1.0000 | -0.0085 | 0.4957 | 0.3886 | 0.9829 | 0.6858 | -0.8834 |
| 2 | 44 | 03 | 0.9919 | -0.4357 | 0.2781 | 0.1100 | 0.6055 | 0.3577 | -0.7133 |
| 2 | 44 | all | 0.9937 | -0.3255 | 0.3341 | 0.1858 | 0.6219 | 0.4038 | -0.7083 |
| 3 | 42 | 01 | 1.0000 | -0.0036 | 0.4982 | 0.4973 | 0.6848 | 0.5911 | -0.4965 |
| 3 | 42 | 02 | 1.0000 | 0.0000 | 0.5000 | 0.4300 | 0.7602 | 0.5951 | -0.5727 |
| 3 | 42 | 03 | 0.8721 | -0.5029 | 0.1846 | 0.1844 | 0.0555 | 0.1199 | -0.1643 |
| 3 | 42 | all | 0.8962 | -0.3143 | 0.2909 | 0.2368 | 0.3057 | 0.2713 | -0.3042 |
| 3 | 43 | 01 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 43 | 02 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 43 | 03 | 0.1586 | -0.4370 | -0.1392 | 0.3369 | -0.4405 | -0.0518 | 0.3333 |
| 3 | 43 | all | 0.1287 | -0.2725 | -0.0719 | 0.2734 | -0.2747 | -0.0007 | 0.1757 |
| 3 | 44 | 01 | -0.9733 | 0.0036 | -0.4848 | 0.4599 | -1.0000 | -0.2701 | 0.4996 |
| 3 | 44 | 02 | -0.9700 | 0.1667 | -0.4017 | 0.2100 | -0.9268 | -0.3584 | 0.4216 |
| 3 | 44 | 03 | -0.5652 | 0.2543 | -0.1554 | 0.6775 | -0.8197 | -0.0711 | 0.1041 |
| 3 | 44 | all | -0.6418 | 0.1889 | -0.2265 | 0.6121 | -0.8745 | -0.1312 | 0.2219 |
| 4 | 42 | 01 | -0.2113 | 0.0000 | -0.1057 | 0.0052 | 0.0000 | 0.0026 | 0.0000 |
| 4 | 42 | 02 | 0.0000 | 0.5668 | 0.2834 | 0.0050 | -0.1255 | -0.0603 | 0.0000 |
| 4 | 42 | 03 | -0.1720 | -0.7161 | -0.4440 | 0.3480 | -0.8626 | -0.2573 | 0.4358 |
| 4 | 42 | all | -0.1579 | -0.2768 | -0.2173 | 0.2767 | -0.5492 | -0.1363 | 0.1769 |
| 4 | 43 | 01 | -0.2113 | 0.0000 | -0.1057 | 0.0052 | 0.0000 | 0.0026 | 0.0000 |
| 4 | 43 | 02 | 0.0050 | 0.0000 | 0.0025 | 0.0050 | 0.0000 | 0.0025 | 0.0000 |
| 4 | 43 | 03 | 0.0387 | -0.8517 | -0.4065 | 0.3620 | -0.8517 | -0.2449 | 0.6558 |
| 4 | 43 | all | 0.0095 | -0.5092 | -0.2498 | 0.2878 | -0.5092 | -0.1107 | 0.3259 |
| 4 | 44 | 01 | 0.0000 | 0.0000 | 0.0000 | 0.7680 | 0.0000 | 0.3840 | -0.5000 |
| 4 | 44 | 02 | 0.0000 | 0.0000 | 0.0000 | 0.9550 | 0.0000 | 0.4775 | -0.4994 |
| 4 | 44 | 03 | 0.0000 | 0.0000 | 0.0000 | 0.4047 | -0.8626 | -0.2290 | 0.3656 |
| 4 | 44 | all | 0.0000 | 0.0000 | 0.0000 | 0.5000 | -0.5157 | -0.0078 | 0.0328 |


## Comparison B' - B (secondary): pooled over the 4 folds, per seed

| seed | stratum | subjects | APCER (0.5): difference [bootstrap] | BPCER (0.5): difference [bootstrap] | ACER (0.5): difference [bootstrap] | APCER (dev threshold): difference [bootstrap] | BPCER (dev threshold): difference [bootstrap] | ACER (dev threshold): difference [bootstrap] | AUROC: difference [bootstrap] |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 01 | 9 | -0.0339 [-0.1367, 0.0294] | -0.0562 [-0.6480, 0.3630] | -0.0451 [-0.3394, 0.1635] | 0.0058 [0.0012, 0.0106] | -0.0956 [-0.3156, 0.0762] | -0.0449 [-0.1559, 0.0428] | 0.0890 [-0.1207, 0.2742] |
| 42 | 02 | 9 | 0.0347 [0.0022, 0.0869] | -0.2553 [-0.6285, 0.0660] | -0.1103 [-0.2852, 0.0373] | 0.0101 [0.0000, 0.0205] | 0.0094 [-0.0933, 0.1119] | 0.0097 [-0.0433, 0.0589] | 0.1779 [-0.0093, 0.2777] |
| 42 | 03 | 16 | 0.1475 [-0.0565, 0.3818] | -0.3004 [-0.6638, 0.1267] | -0.0764 [-0.3001, 0.1751] | 0.2352 [0.0843, 0.3868] | -0.0547 [-0.4777, 0.3541] | 0.0902 [-0.1620, 0.3196] | 0.0333 [-0.2040, 0.3258] |
| 42 | all | 16 | 0.1135 [-0.0506, 0.2952] | -0.2503 [-0.5235, 0.0664] | -0.0684 [-0.2213, 0.0932] | 0.1823 [0.0667, 0.3146] | -0.0511 [-0.3233, 0.2128] | 0.0656 [-0.0981, 0.2181] | 0.0509 [-0.0825, 0.2189] |
| 43 | 01 | 9 | 0.0561 [0.0000, 0.1692] | 0.0562 [0.0000, 0.1822] | 0.0562 [0.0009, 0.1479] | 0.0959 [0.0000, 0.2867] | 0.0607 [-0.0018, 0.2008] | 0.0783 [0.0000, 0.1971] | -0.0814 [-0.2357, 0.0000] |
| 43 | 02 | 9 | 0.0011 [0.0000, 0.0034] | 0.1710 [0.0009, 0.4536] | 0.0860 [0.0011, 0.2273] | 0.0022 [0.0000, 0.0067] | 0.2681 [0.0020, 0.5878] | 0.1352 [0.0034, 0.2950] | -0.1230 [-0.3297, 0.0018] |
| 43 | 03 | 16 | 0.0064 [-0.0919, 0.1067] | 0.0042 [-0.0928, 0.0774] | 0.0053 [-0.0638, 0.0752] | 0.0854 [0.0136, 0.1653] | 0.0187 [-0.0594, 0.0872] | 0.0521 [-0.0080, 0.1167] | -0.0699 [-0.2489, 0.0859] |
| 43 | all | 16 | 0.0115 [-0.0720, 0.0829] | 0.0411 [-0.0429, 0.1135] | 0.0263 [-0.0339, 0.0803] | 0.0767 [0.0115, 0.1501] | 0.0678 [-0.0171, 0.1399] | 0.0722 [0.0113, 0.1227] | -0.1223 [-0.2504, 0.0154] |
| 44 | 01 | 9 | -0.2129 [-0.4408, 0.0000] | 0.1339 [-0.4634, 0.7684] | -0.0395 [-0.4560, 0.3660] | 0.2749 [-0.1220, 0.6584] | -0.1429 [-0.5985, 0.3922] | 0.0660 [-0.2734, 0.4285] | -0.2773 [-0.5000, 0.1629] |
| 44 | 02 | 9 | -0.2172 [-0.4504, 0.0000] | 0.1815 [-0.4046, 0.7075] | -0.0179 [-0.4329, 0.3366] | 0.2531 [-0.1848, 0.6733] | 0.0152 [-0.5349, 0.5576] | 0.1342 [-0.3108, 0.5105] | -0.2876 [-0.4999, 0.1393] |
| 44 | 03 | 16 | -0.1628 [-0.3439, -0.0049] | 0.1452 [-0.0260, 0.3659] | -0.0088 [-0.1499, 0.1397] | 0.2498 [0.0140, 0.5060] | -0.3117 [-0.5985, -0.0640] | -0.0310 [-0.1942, 0.1331] | -0.0815 [-0.3260, 0.1581] |
| 44 | all | 16 | -0.1750 [-0.3765, -0.0038] | 0.1493 [-0.1134, 0.4360] | -0.0129 [-0.2247, 0.1988] | 0.2530 [-0.0036, 0.5364] | -0.2276 [-0.4691, 0.0573] | 0.0127 [-0.1965, 0.2100] | -0.1484 [-0.3578, 0.0735] |


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
| 02 | APCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 02 | BPCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 02 | ACER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 02 | APCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 02 | BPCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 02 | ACER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 02 | AUROC | secondary | inconclusive | the sign differs across the seeds |
| 03 | APCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 03 | BPCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 03 | ACER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 03 | APCER (dev threshold) | secondary | consistent: B' higher APCER (dev threshold) than B | same sign for every seed and every interval excludes 0 |
| 03 | BPCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 03 | ACER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 03 | AUROC | secondary | inconclusive | the sign differs across the seeds |
| all | APCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| all | BPCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| all | ACER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| all | APCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | BPCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| all | ACER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | AUROC | secondary | inconclusive | the sign differs across the seeds |


Median and range over the seeds of the pooled differences

| stratum | APCER (0.5) | BPCER (0.5) | ACER (0.5) | APCER (dev threshold) | BPCER (dev threshold) | ACER (dev threshold) | AUROC |
|---|---|---|---|---|---|---|---|
| 01 | -0.0339 (-0.2129 to 0.0561) | 0.0562 (-0.0562 to 0.1339) | -0.0395 (-0.0451 to 0.0562) | 0.0959 (0.0058 to 0.2749) | -0.0956 (-0.1429 to 0.0607) | 0.0660 (-0.0449 to 0.0783) | -0.0814 (-0.2773 to 0.0890) |
| 02 | 0.0011 (-0.2172 to 0.0347) | 0.1710 (-0.2553 to 0.1815) | -0.0179 (-0.1103 to 0.0860) | 0.0101 (0.0022 to 0.2531) | 0.0152 (0.0094 to 0.2681) | 0.1342 (0.0097 to 0.1352) | -0.1230 (-0.2876 to 0.1779) |
| 03 | 0.0064 (-0.1628 to 0.1475) | 0.0042 (-0.3004 to 0.1452) | -0.0088 (-0.0764 to 0.0053) | 0.2352 (0.0854 to 0.2498) | -0.0547 (-0.3117 to 0.0187) | 0.0521 (-0.0310 to 0.0902) | -0.0699 (-0.0815 to 0.0333) |
| all | 0.0115 (-0.1750 to 0.1135) | 0.0411 (-0.2503 to 0.1493) | -0.0129 (-0.0684 to 0.0263) | 0.1823 (0.0767 to 0.2530) | -0.0511 (-0.2276 to 0.0678) | 0.0656 (0.0127 to 0.0722) | -0.1223 (-0.1484 to 0.0509) |


Per fold and seed (B' - B)

| fold | seed | stratum | APCER (0.5) | BPCER (0.5) | ACER (0.5) | APCER (dev threshold) | BPCER (dev threshold) | ACER (dev threshold) | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 42 | 01 | 0.0418 | -1.0000 | -0.4791 | 0.0035 | -0.4764 | -0.2364 | 0.0020 |
| 1 | 42 | 02 | 0.0967 | -1.0000 | -0.4517 | 0.0000 | 0.1598 | 0.0799 | 0.0075 |
| 1 | 42 | 03 | 0.9513 | -0.5081 | 0.2216 | 0.1258 | 0.5092 | 0.3175 | -0.4436 |
| 1 | 42 | all | 0.6668 | -0.6803 | -0.0067 | 0.0858 | 0.2734 | 0.1796 | -0.2360 |
| 1 | 43 | 01 | 0.0000 | 0.1890 | 0.0945 | 0.0035 | 0.2087 | 0.1061 | -0.1739 |
| 1 | 43 | 02 | 0.0033 | 0.5861 | 0.2947 | 0.0067 | 0.4918 | 0.2492 | -0.2524 |
| 1 | 43 | 03 | 0.0057 | 0.0627 | 0.0342 | 0.0406 | 0.0714 | 0.0560 | -0.3051 |
| 1 | 43 | all | 0.0044 | 0.1750 | 0.0897 | 0.0291 | 0.1680 | 0.0985 | -0.3092 |
| 1 | 44 | 01 | 0.0000 | 0.6299 | 0.3150 | 0.0000 | 0.5866 | 0.2933 | -0.5000 |
| 1 | 44 | 02 | 0.0000 | 0.9959 | 0.4980 | 0.0000 | 0.9877 | 0.4939 | -0.5000 |
| 1 | 44 | 03 | -0.0049 | 0.0108 | 0.0030 | -0.0041 | 0.0000 | -0.0020 | 0.1191 |
| 1 | 44 | all | -0.0033 | 0.2902 | 0.1435 | -0.0027 | 0.2741 | 0.1357 | -0.1400 |
| 2 | 42 | 01 | 0.0000 | 0.5299 | 0.2650 | 0.0107 | 0.1581 | 0.0844 | -0.0038 |
| 2 | 42 | 02 | 0.0104 | 0.0000 | 0.0052 | 0.0259 | -0.0085 | 0.0087 | 0.0009 |
| 2 | 42 | 03 | -0.0147 | -0.5152 | -0.2649 | 0.2419 | -0.2247 | 0.0086 | 0.2447 |
| 2 | 42 | all | -0.0103 | -0.2927 | -0.1515 | 0.1932 | -0.1409 | 0.0262 | 0.0913 |
| 2 | 43 | 01 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0043 | -0.0021 | 0.0000 |
| 2 | 43 | 02 | 0.0000 | -0.0085 | -0.0043 | 0.0000 | -0.0085 | -0.0043 | 0.0073 |
| 2 | 43 | 03 | 0.1752 | -0.1079 | 0.0336 | 0.0784 | -0.0707 | 0.0039 | -0.0798 |
| 2 | 43 | all | 0.1370 | -0.0810 | 0.0280 | 0.0614 | -0.0540 | 0.0037 | -0.0310 |
| 2 | 44 | 01 | 0.0000 | 0.9957 | 0.4979 | 1.0000 | -0.0043 | 0.4979 | -0.5000 |
| 2 | 44 | 02 | 0.0000 | 0.9915 | 0.4957 | 1.0000 | -0.0085 | 0.4957 | -0.5000 |
| 2 | 44 | 03 | -0.0081 | 0.5643 | 0.2781 | 0.9905 | -0.3935 | 0.2985 | -0.4779 |
| 2 | 44 | all | -0.0063 | 0.6745 | 0.3341 | 0.9925 | -0.2942 | 0.3492 | -0.4865 |
| 3 | 42 | 01 | 0.0000 | 0.2899 | 0.1449 | 0.0000 | -0.0036 | -0.0018 | 0.0000 |
| 3 | 42 | 02 | 0.0000 | 0.0935 | 0.0467 | 0.0200 | 0.0041 | 0.0120 | -0.0002 |
| 3 | 42 | 03 | -0.0486 | 0.3364 | 0.1439 | 0.2613 | 0.0555 | 0.1584 | -0.1367 |
| 3 | 42 | all | -0.0395 | 0.2841 | 0.1223 | 0.2139 | 0.0346 | 0.1243 | -0.0886 |
| 3 | 43 | 01 | 0.0000 | 0.0072 | 0.0036 | 0.0000 | 0.0072 | 0.0036 | -0.0036 |
| 3 | 43 | 02 | 0.0000 | 0.0163 | 0.0081 | 0.0000 | 0.4472 | 0.2236 | 0.0000 |
| 3 | 43 | 03 | -0.0246 | 0.0763 | 0.0258 | 0.0480 | 0.0798 | 0.0639 | -0.1272 |
| 3 | 43 | all | -0.0200 | 0.0519 | 0.0160 | 0.0390 | 0.1305 | 0.0847 | -0.1248 |
| 3 | 44 | 01 | -0.9733 | -0.9928 | -0.9830 | -0.5401 | -0.9964 | -0.7682 | 0.9996 |
| 3 | 44 | 02 | -0.9700 | -0.8293 | -0.8996 | -0.7900 | -0.9228 | -0.8564 | 0.9216 |
| 3 | 44 | 03 | -0.5532 | -0.1121 | -0.3326 | -0.3087 | -0.1965 | -0.2526 | 0.1948 |
| 3 | 44 | all | -0.6321 | -0.4146 | -0.5233 | -0.3767 | -0.4845 | -0.4306 | 0.4895 |
| 4 | 42 | 01 | -0.2113 | 0.0000 | -0.1057 | 0.0103 | 0.0000 | 0.0052 | 0.0000 |
| 4 | 42 | 02 | 0.0000 | 0.0121 | 0.0061 | 0.0000 | -0.1255 | -0.0628 | 0.0000 |
| 4 | 42 | 03 | -0.1473 | -0.5533 | -0.3503 | 0.2900 | -0.8571 | -0.2836 | 0.2501 |
| 4 | 42 | all | -0.1383 | -0.3276 | -0.2329 | 0.2307 | -0.5459 | -0.1576 | 0.1215 |
| 4 | 43 | 01 | 0.2474 | 0.0000 | 0.1237 | 0.4175 | 0.0000 | 0.2088 | 0.0000 |
| 4 | 43 | 02 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4 | 43 | 03 | -0.1120 | 0.0000 | -0.0560 | 0.1700 | 0.0000 | 0.0850 | -0.0760 |
| 4 | 43 | all | -0.0634 | 0.0000 | -0.0317 | 0.1774 | 0.0000 | 0.0887 | -0.1142 |
| 4 | 44 | 01 | 0.0000 | 0.0000 | 0.0000 | 0.7680 | 0.0000 | 0.3840 | -0.5000 |
| 4 | 44 | 02 | 0.0000 | 0.0000 | 0.0000 | 0.9550 | 0.0000 | 0.4775 | -0.4994 |
| 4 | 44 | 03 | 0.0000 | 0.0000 | 0.0000 | 0.4047 | -0.8626 | -0.2290 | 0.3656 |
| 4 | 44 | all | 0.0000 | 0.0000 | 0.0000 | 0.5000 | -0.5157 | -0.0078 | 0.0328 |


## Comparison A* - B' (secondary): pooled over the 4 folds, per seed

| seed | stratum | subjects | APCER (0.5): difference [bootstrap] | BPCER (0.5): difference [bootstrap] | ACER (0.5): difference [bootstrap] | APCER (dev threshold): difference [bootstrap] | BPCER (dev threshold): difference [bootstrap] | ACER (dev threshold): difference [bootstrap] | AUROC: difference [bootstrap] |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 01 | 9 | 0.2047 [-0.0243, 0.4572] | -0.2261 [-0.4836, 0.0064] | -0.0107 [-0.1832, 0.1523] | 0.1053 [-0.0083, 0.3222] | 0.1732 [-0.0902, 0.4255] | 0.1392 [-0.0471, 0.3546] | 0.0329 [-0.2194, 0.2300] |
| 42 | 02 | 9 | 0.1892 [-0.0569, 0.4495] | 0.1370 [-0.0595, 0.4194] | 0.1631 [0.0124, 0.3236] | 0.0873 [-0.0079, 0.2167] | 0.0749 [-0.3458, 0.4914] | 0.0811 [-0.1632, 0.3383] | -0.1352 [-0.3168, 0.0058] |
| 42 | 03 | 16 | 0.0505 [-0.3091, 0.4036] | -0.2888 [-0.5913, -0.0213] | -0.1192 [-0.3247, 0.0755] | -0.1896 [-0.4297, 0.0611] | -0.1618 [-0.6416, 0.3649] | -0.1757 [-0.4273, 0.0839] | 0.2325 [-0.0353, 0.4841] |
| 42 | all | 16 | 0.0846 [-0.2206, 0.4091] | -0.2067 [-0.4081, -0.0061] | -0.0610 [-0.2120, 0.0930] | -0.1231 [-0.3030, 0.0669] | -0.0639 [-0.4697, 0.3408] | -0.0935 [-0.2976, 0.1275] | 0.1324 [-0.0458, 0.3027] |
| 43 | 01 | 9 | -0.1041 [-0.3162, 0.0034] | 0.2576 [-0.0103, 0.5862] | 0.0767 [-0.1087, 0.2561] | 0.1240 [-0.1898, 0.4531] | -0.0169 [-0.0506, 0.0000] | 0.0536 [-0.1134, 0.2233] | -0.1481 [-0.4252, 0.1600] |
| 43 | 02 | 9 | 0.0000 [-0.0033, 0.0034] | -0.0340 [-0.3987, 0.3606] | -0.0170 [-0.1999, 0.1803] | 0.2150 [-0.0033, 0.5427] | -0.2728 [-0.5885, -0.0104] | -0.0289 [-0.2445, 0.1982] | 0.0031 [-0.2895, 0.3052] |
| 43 | 03 | 16 | 0.0274 [-0.2217, 0.2971] | -0.4512 [-0.7958, -0.0437] | -0.2119 [-0.3701, -0.0243] | 0.3390 [0.1027, 0.6107] | -0.7838 [-0.9390, -0.5447] | -0.2224 [-0.3636, -0.0788] | 0.2764 [-0.1004, 0.5931] |
| 43 | all | 16 | 0.0092 [-0.1831, 0.2458] | -0.2580 [-0.5872, 0.0783] | -0.1244 [-0.2550, 0.0244] | 0.2998 [0.0793, 0.5858] | -0.5647 [-0.7472, -0.3371] | -0.1325 [-0.2537, 0.0003] | 0.1425 [-0.1814, 0.4353] |
| 44 | 01 | 9 | 0.2199 [0.0000, 0.5395] | -0.2385 [-0.8516, 0.4122] | -0.0093 [-0.3333, 0.2753] | 0.1158 [-0.2085, 0.4435] | -0.1417 [-0.6312, 0.2901] | -0.0130 [-0.2761, 0.2181] | -0.0297 [-0.2669, 0.2690] |
| 44 | 02 | 9 | 0.2161 [0.0000, 0.5414] | -0.0925 [-0.6265, 0.5063] | 0.0618 [-0.1944, 0.3653] | 0.0918 [-0.2605, 0.4444] | -0.0948 [-0.4837, 0.3200] | -0.0015 [-0.2326, 0.2256] | -0.1035 [-0.3948, 0.1313] |
| 44 | 03 | 16 | 0.2487 [0.0688, 0.4817] | -0.3953 [-0.7915, 0.0649] | -0.0733 [-0.2371, 0.1273] | 0.0854 [-0.2646, 0.4099] | -0.0250 [-0.5155, 0.5547] | 0.0302 [-0.1768, 0.2680] | 0.0712 [-0.2017, 0.3076] |
| 44 | all | 16 | 0.2416 [0.0653, 0.4781] | -0.3173 [-0.7191, 0.1408] | -0.0379 [-0.2086, 0.1665] | 0.0896 [-0.2364, 0.4016] | -0.0570 [-0.4540, 0.3795] | 0.0163 [-0.1633, 0.1964] | 0.0281 [-0.1873, 0.2156] |


Verdicts (A* - B'; primary only for AUROC and ACER dev in stratum 03 of A* - B)

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
| 02 | BPCER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 02 | ACER (0.5) | secondary | inconclusive | the sign differs across the seeds |
| 02 | APCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 02 | BPCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 02 | ACER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 02 | AUROC | secondary | inconclusive | the sign differs across the seeds |
| 03 | APCER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | BPCER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | ACER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | APCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 03 | BPCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| 03 | ACER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| 03 | AUROC | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | APCER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | BPCER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | ACER (0.5) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | APCER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| all | BPCER (dev threshold) | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |
| all | ACER (dev threshold) | secondary | inconclusive | the sign differs across the seeds |
| all | AUROC | secondary | inconclusive | the bootstrap interval of a seed does not exclude 0 on the side of the difference |


Median and range over the seeds of the pooled differences

| stratum | APCER (0.5) | BPCER (0.5) | ACER (0.5) | APCER (dev threshold) | BPCER (dev threshold) | ACER (dev threshold) | AUROC |
|---|---|---|---|---|---|---|---|
| 01 | 0.2047 (-0.1041 to 0.2199) | -0.2261 (-0.2385 to 0.2576) | -0.0093 (-0.0107 to 0.0767) | 0.1158 (0.1053 to 0.1240) | -0.0169 (-0.1417 to 0.1732) | 0.0536 (-0.0130 to 0.1392) | -0.0297 (-0.1481 to 0.0329) |
| 02 | 0.1892 (0.0000 to 0.2161) | -0.0340 (-0.0925 to 0.1370) | 0.0618 (-0.0170 to 0.1631) | 0.0918 (0.0873 to 0.2150) | -0.0948 (-0.2728 to 0.0749) | -0.0015 (-0.0289 to 0.0811) | -0.1035 (-0.1352 to 0.0031) |
| 03 | 0.0505 (0.0274 to 0.2487) | -0.3953 (-0.4512 to -0.2888) | -0.1192 (-0.2119 to -0.0733) | 0.0854 (-0.1896 to 0.3390) | -0.1618 (-0.7838 to -0.0250) | -0.1757 (-0.2224 to 0.0302) | 0.2325 (0.0712 to 0.2764) |
| all | 0.0846 (0.0092 to 0.2416) | -0.2580 (-0.3173 to -0.2067) | -0.0610 (-0.1244 to -0.0379) | 0.0896 (-0.1231 to 0.2998) | -0.0639 (-0.5647 to -0.0570) | -0.0935 (-0.1325 to 0.0163) | 0.1324 (0.0281 to 0.1425) |


Per fold and seed (A* - B')

| fold | seed | stratum | APCER (0.5) | BPCER (0.5) | ACER (0.5) | APCER (dev threshold) | BPCER (dev threshold) | ACER (dev threshold) | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 42 | 01 | -0.0418 | 0.0197 | -0.0111 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1 | 42 | 02 | -0.0967 | 0.0123 | -0.0422 | 0.0000 | -0.5082 | -0.2541 | 0.0068 |
| 1 | 42 | 03 | -0.9318 | -0.2627 | -0.5973 | -0.0187 | -0.9319 | -0.4753 | 0.7948 |
| 1 | 42 | all | -0.6537 | -0.1651 | -0.4094 | -0.0126 | -0.6929 | -0.3528 | 0.4799 |
| 1 | 43 | 01 | 0.0000 | -0.0118 | -0.0059 | -0.0035 | -0.0512 | -0.0273 | 0.1718 |
| 1 | 43 | 02 | -0.0033 | -0.5820 | -0.2927 | -0.0067 | -0.5000 | -0.2533 | 0.2524 |
| 1 | 43 | 03 | -0.0114 | -0.9438 | -0.4776 | -0.0438 | -0.9557 | -0.4998 | 0.5148 |
| 1 | 43 | all | -0.0082 | -0.7154 | -0.3618 | -0.0313 | -0.7161 | -0.3737 | 0.4064 |
| 1 | 44 | 01 | 0.0035 | -0.9961 | -0.4963 | 0.0000 | -0.9685 | -0.4843 | 0.4964 |
| 1 | 44 | 02 | 0.0000 | -0.8484 | -0.4242 | 0.0000 | -0.8033 | -0.4016 | 0.4755 |
| 1 | 44 | 03 | 0.0722 | -0.6778 | -0.3028 | 0.0414 | -0.6086 | -0.2836 | 0.3401 |
| 1 | 44 | all | 0.0495 | -0.7639 | -0.3572 | 0.0280 | -0.7063 | -0.3391 | 0.3963 |
| 2 | 42 | 01 | 0.0000 | -0.5342 | -0.2671 | -0.0107 | -0.1538 | -0.0823 | 0.0038 |
| 2 | 42 | 02 | -0.0104 | 0.0000 | -0.0052 | -0.0259 | 0.0171 | -0.0044 | 0.0052 |
| 2 | 42 | 03 | -0.0418 | 0.0864 | 0.0223 | -0.7537 | 0.3150 | -0.2193 | -0.0183 |
| 2 | 42 | all | -0.0338 | -0.0270 | -0.0304 | -0.5935 | 0.2095 | -0.1920 | 0.0420 |
| 2 | 43 | 01 | 0.0000 | 1.0000 | 0.5000 | 1.0000 | 0.0000 | 0.5000 | -0.5000 |
| 2 | 43 | 02 | 0.0000 | 1.0000 | 0.5000 | 1.0000 | -0.0085 | 0.4957 | -0.5000 |
| 2 | 43 | 03 | -0.2632 | 0.2659 | 0.0014 | 0.9076 | -0.8145 | 0.0466 | 0.1054 |
| 2 | 43 | all | -0.2058 | 0.4540 | 0.1241 | 0.9278 | -0.6066 | 0.1606 | -0.1391 |
| 2 | 44 | 01 | 1.0000 | -1.0000 | 0.0000 | -0.4706 | 0.5171 | 0.0233 | -0.2589 |
| 2 | 44 | 02 | 1.0000 | -1.0000 | 0.0000 | -0.6114 | 0.9915 | 0.1900 | -0.3834 |
| 2 | 44 | 03 | 1.0000 | -1.0000 | 0.0000 | -0.8805 | 0.9990 | 0.0593 | -0.2354 |
| 2 | 44 | all | 1.0000 | -1.0000 | 0.0000 | -0.8068 | 0.9161 | 0.0546 | -0.2218 |
| 3 | 42 | 01 | 1.0000 | -0.2935 | 0.3533 | 0.4973 | 0.6884 | 0.5929 | -0.4965 |
| 3 | 42 | 02 | 1.0000 | -0.0935 | 0.4533 | 0.4100 | 0.7561 | 0.5830 | -0.5726 |
| 3 | 42 | 03 | 0.9207 | -0.8393 | 0.0407 | -0.0769 | 0.0000 | -0.0384 | -0.0276 |
| 3 | 42 | all | 0.9357 | -0.5984 | 0.1686 | 0.0229 | 0.2711 | 0.1470 | -0.2156 |
| 3 | 43 | 01 | 0.0000 | -0.0072 | -0.0036 | 0.0000 | -0.0072 | -0.0036 | 0.0036 |
| 3 | 43 | 02 | 0.0000 | -0.0163 | -0.0081 | 0.0000 | -0.4472 | -0.2236 | 0.0000 |
| 3 | 43 | 03 | 0.1832 | -0.5133 | -0.1651 | 0.2889 | -0.5202 | -0.1157 | 0.4604 |
| 3 | 43 | all | 0.1486 | -0.3244 | -0.0879 | 0.2344 | -0.4052 | -0.0854 | 0.3006 |
| 3 | 44 | 01 | 0.0000 | 0.9964 | 0.4982 | 1.0000 | -0.0036 | 0.4982 | -0.5000 |
| 3 | 44 | 02 | 0.0000 | 0.9959 | 0.4980 | 1.0000 | -0.0041 | 0.4980 | -0.5000 |
| 3 | 44 | 03 | -0.0120 | 0.3665 | 0.1772 | 0.9862 | -0.6231 | 0.1815 | -0.0907 |
| 3 | 44 | all | -0.0097 | 0.6035 | 0.2969 | 0.9888 | -0.3901 | 0.2994 | -0.2675 |
| 4 | 42 | 01 | 0.0000 | 0.0000 | 0.0000 | -0.0052 | 0.0000 | -0.0026 | 0.0000 |
| 4 | 42 | 02 | 0.0000 | 0.5547 | 0.2773 | 0.0050 | 0.0000 | 0.0025 | 0.0000 |
| 4 | 42 | 03 | -0.0247 | -0.1627 | -0.0937 | 0.0580 | -0.0054 | 0.0263 | 0.1857 |
| 4 | 42 | all | -0.0195 | 0.0508 | 0.0156 | 0.0459 | -0.0032 | 0.0213 | 0.0554 |
| 4 | 43 | 01 | -0.4588 | 0.0000 | -0.2294 | -0.4124 | 0.0000 | -0.2062 | 0.0000 |
| 4 | 43 | 02 | 0.0050 | 0.0000 | 0.0025 | 0.0050 | 0.0000 | 0.0025 | 0.0000 |
| 4 | 43 | 03 | 0.1507 | -0.8517 | -0.3505 | 0.1920 | -0.8517 | -0.3299 | 0.7319 |
| 4 | 43 | all | 0.0729 | -0.5092 | -0.2182 | 0.1103 | -0.5092 | -0.1994 | 0.4400 |
| 4 | 44 | 01 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4 | 44 | 02 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4 | 44 | 03 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4 | 44 | all | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |


## Per arm: pooled over the folds, per seed (bootstrap over subjects)

| arm | seed | stratum | n bona fide | n attack | subjects | BPCER 05 | APCER 05 | ACER 05 [bootstrap] | BPCER dev | APCER dev | ACER dev [bootstrap] | AUROC [bootstrap] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B | 42 | 01 | 889 | 855 | 9 | 0.2880 | 0.0480 | 0.1680 [0.0386, 0.3794] | 0.1372 | 0.0000 | 0.0686 [0.0073, 0.1688] | 0.8113 [0.7007, 0.9642] |
| B | 42 | 02 | 854 | 893 | 9 | 0.2869 | 0.0000 | 0.1434 [0.0000, 0.3156] | 0.1382 | 0.0000 | 0.0691 [0.0023, 0.1706] | 0.8110 [0.7156, 1.0000] |
| B | 42 | 03 | 3362 | 5761 | 16 | 0.6802 | 0.1970 | 0.4386 [0.2521, 0.5801] | 0.4744 | 0.2090 | 0.3417 [0.1669, 0.4932] | 0.5786 [0.3726, 0.8222] |
| B | 42 | all | 5105 | 7509 | 16 | 0.5461 | 0.1566 | 0.3514 [0.2023, 0.4713] | 0.3595 | 0.1603 | 0.2599 [0.1359, 0.3813] | 0.6739 [0.5166, 0.8598] |
| B | 43 | 01 | 889 | 855 | 9 | 0.0517 | 0.0491 | 0.0504 [0.0000, 0.1356] | 0.0450 | 0.0000 | 0.0225 [0.0000, 0.0733] | 0.9932 [0.9661, 1.0000] |
| B | 43 | 02 | 854 | 893 | 9 | 0.0035 | 0.0000 | 0.0018 [0.0000, 0.0041] | 0.0047 | 0.0000 | 0.0023 [0.0000, 0.0053] | 0.9989 [0.9963, 1.0000] |
| B | 43 | 03 | 3362 | 5761 | 16 | 0.7832 | 0.1977 | 0.4904 [0.3328, 0.6065] | 0.7948 | 0.0451 | 0.4199 [0.2683, 0.5161] | 0.4800 [0.2815, 0.6885] |
| B | 43 | all | 5105 | 7509 | 16 | 0.5254 | 0.1573 | 0.3413 [0.2101, 0.4605] | 0.5320 | 0.0346 | 0.2833 [0.1550, 0.3915] | 0.7114 [0.5528, 0.8547] |
| B | 44 | 01 | 889 | 855 | 9 | 0.5568 | 0.2129 | 0.3848 [0.1158, 0.6467] | 0.4297 | 0.1708 | 0.3002 [0.0543, 0.5227] | 0.8616 [0.5668, 1.0000] |
| B | 44 | 02 | 854 | 893 | 9 | 0.5316 | 0.2172 | 0.3744 [0.1321, 0.6220] | 0.2717 | 0.1870 | 0.2293 [0.0074, 0.5032] | 0.8691 [0.6099, 0.9999] |
| B | 44 | 03 | 3362 | 5761 | 16 | 0.7606 | 0.1663 | 0.4634 [0.3472, 0.5609] | 0.7472 | 0.2513 | 0.4993 [0.3623, 0.6222] | 0.5717 [0.3233, 0.8155] |
| B | 44 | all | 5105 | 7509 | 16 | 0.6868 | 0.1777 | 0.4322 [0.2947, 0.5570] | 0.6123 | 0.2345 | 0.4234 [0.2782, 0.5550] | 0.6696 [0.4842, 0.8564] |
| A* | 42 | 01 | 889 | 855 | 9 | 0.0056 | 0.2187 | 0.1122 [0.0000, 0.2322] | 0.2148 | 0.1111 | 0.1630 [0.0018, 0.3680] | 0.9332 [0.7518, 1.0000] |
| A* | 42 | 02 | 854 | 893 | 9 | 0.1686 | 0.2240 | 0.1963 [0.0596, 0.3517] | 0.2225 | 0.0974 | 0.1600 [0.0021, 0.3653] | 0.8537 [0.6775, 0.9998] |
| A* | 42 | 03 | 3362 | 5761 | 16 | 0.0910 | 0.3951 | 0.2430 [0.1406, 0.3443] | 0.2579 | 0.2546 | 0.2563 [0.1303, 0.3840] | 0.8444 [0.7128, 0.9509] |
| A* | 42 | all | 5105 | 7509 | 16 | 0.0891 | 0.3546 | 0.2219 [0.1244, 0.3239] | 0.2445 | 0.2196 | 0.2320 [0.1106, 0.3655] | 0.8572 [0.7404, 0.9531] |
| A* | 43 | 01 | 889 | 855 | 9 | 0.3656 | 0.0012 | 0.1834 [0.0017, 0.3507] | 0.0889 | 0.2199 | 0.1544 [0.0017, 0.2911] | 0.7636 [0.5441, 1.0000] |
| A* | 43 | 02 | 854 | 893 | 9 | 0.1405 | 0.0011 | 0.0708 [0.0006, 0.2183] | 0.0000 | 0.2172 | 0.1086 [0.0000, 0.2718] | 0.8789 [0.6546, 1.0000] |
| A* | 43 | 03 | 3362 | 5761 | 16 | 0.3361 | 0.2316 | 0.2838 [0.1514, 0.4338] | 0.0297 | 0.4695 | 0.2496 [0.1497, 0.3675] | 0.6865 [0.4291, 0.9059] |
| A* | 43 | all | 5105 | 7509 | 16 | 0.3085 | 0.1779 | 0.2432 [0.1203, 0.3827] | 0.0351 | 0.4111 | 0.2231 [0.1330, 0.3461] | 0.7316 [0.5325, 0.9205] |
| A* | 44 | 01 | 889 | 855 | 9 | 0.4522 | 0.2199 | 0.3360 [0.1385, 0.4788] | 0.1451 | 0.5614 | 0.3533 [0.1872, 0.5061] | 0.5545 [0.4319, 0.7748] |
| A* | 44 | 02 | 854 | 893 | 9 | 0.6206 | 0.2161 | 0.4184 [0.2532, 0.5957] | 0.1920 | 0.5319 | 0.3620 [0.2233, 0.4874] | 0.4779 [0.3149, 0.6564] |
| A* | 44 | 03 | 3362 | 5761 | 16 | 0.5104 | 0.2522 | 0.3813 [0.2534, 0.5164] | 0.4105 | 0.5865 | 0.4985 [0.3815, 0.6314] | 0.5614 [0.4033, 0.7139] |
| A* | 44 | all | 5105 | 7509 | 16 | 0.5187 | 0.2442 | 0.3815 [0.2785, 0.4990] | 0.3277 | 0.5772 | 0.4524 [0.3443, 0.5615] | 0.5494 [0.4330, 0.6648] |
| B' | 42 | 01 | 889 | 855 | 9 | 0.2317 | 0.0140 | 0.1229 [0.0142, 0.2458] | 0.0416 | 0.0058 | 0.0237 [0.0012, 0.0717] | 0.9002 [0.7489, 0.9990] |
| B' | 42 | 02 | 854 | 893 | 9 | 0.0316 | 0.0347 | 0.0332 [0.0061, 0.0653] | 0.1475 | 0.0101 | 0.0788 [0.0058, 0.2150] | 0.9889 [0.9764, 0.9981] |
| B' | 42 | 03 | 3362 | 5761 | 16 | 0.3798 | 0.3446 | 0.3622 [0.1826, 0.5706] | 0.4197 | 0.4442 | 0.4319 [0.2321, 0.6172] | 0.6118 [0.3394, 0.8658] |
| B' | 42 | all | 5105 | 7509 | 16 | 0.2958 | 0.2701 | 0.2829 [0.1646, 0.4153] | 0.3083 | 0.3427 | 0.3255 [0.1773, 0.4806] | 0.7248 [0.5518, 0.8813] |
| B' | 43 | 01 | 889 | 855 | 9 | 0.1080 | 0.1053 | 0.1066 [0.0009, 0.2834] | 0.1057 | 0.0959 | 0.1008 [0.0009, 0.2657] | 0.9117 [0.7342, 1.0000] |
| B' | 43 | 02 | 854 | 893 | 9 | 0.1745 | 0.0011 | 0.0878 [0.0021, 0.2294] | 0.2728 | 0.0022 | 0.1375 [0.0066, 0.2953] | 0.8759 [0.6694, 0.9996] |
| B' | 43 | 03 | 3362 | 5761 | 16 | 0.7873 | 0.2041 | 0.4957 [0.3698, 0.6005] | 0.8135 | 0.1305 | 0.4720 [0.3537, 0.5580] | 0.4101 [0.2368, 0.6175] |
| B' | 43 | all | 5105 | 7509 | 16 | 0.5665 | 0.1687 | 0.3676 [0.2485, 0.4706] | 0.5998 | 0.1113 | 0.3556 [0.2349, 0.4570] | 0.5891 [0.4036, 0.7904] |
| B' | 44 | 01 | 889 | 855 | 9 | 0.6907 | 0.0000 | 0.3453 [0.1716, 0.5000] | 0.2868 | 0.4456 | 0.3662 [0.2171, 0.5259] | 0.5843 [0.4894, 0.7601] |
| B' | 44 | 02 | 854 | 893 | 9 | 0.7131 | 0.0000 | 0.3566 [0.1673, 0.5000] | 0.2869 | 0.4401 | 0.3635 [0.1781, 0.5237] | 0.5814 [0.5000, 0.7722] |
| B' | 44 | 03 | 3362 | 5761 | 16 | 0.9057 | 0.0035 | 0.4546 [0.3533, 0.5036] | 0.4355 | 0.5011 | 0.4683 [0.3096, 0.6223] | 0.4902 [0.3657, 0.6342] |
| B' | 44 | all | 5105 | 7509 | 16 | 0.8360 | 0.0027 | 0.4194 [0.3073, 0.5019] | 0.3847 | 0.4875 | 0.4361 [0.2957, 0.5700] | 0.5213 [0.4207, 0.6421] |


## Per arm, fold and seed (held-out subjects, by session stratum)

| arm | fold | seed | stratum | n bona fide | n attack | subjects | BPCER 05 | APCER 05 | ACER 05 | BPCER dev | APCER dev | ACER dev | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B | 1 | 42 | 01 | 254 | 287 | 3 | 1.0000 | 0.0000 | 0.5000 | 0.4764 | 0.0000 | 0.2382 | 0.9980 |
| B | 1 | 42 | 02 | 244 | 300 | 3 | 1.0000 | 0.0000 | 0.5000 | 0.3525 | 0.0000 | 0.1762 | 0.9857 |
| B | 1 | 42 | 03 | 925 | 1232 | 4 | 1.0000 | 0.0000 | 0.5000 | 0.4908 | 0.0966 | 0.2937 | 0.6134 |
| B | 1 | 42 | all | 1423 | 1819 | 4 | 1.0000 | 0.0000 | 0.5000 | 0.4645 | 0.0654 | 0.2650 | 0.7389 |
| B | 1 | 43 | 01 | 254 | 287 | 3 | 0.1811 | 0.0000 | 0.0906 | 0.1535 | 0.0000 | 0.0768 | 0.9921 |
| B | 1 | 43 | 02 | 244 | 300 | 3 | 0.0082 | 0.0000 | 0.0041 | 0.0082 | 0.0000 | 0.0041 | 1.0000 |
| B | 1 | 43 | 03 | 925 | 1232 | 4 | 0.9373 | 0.0057 | 0.4715 | 0.9286 | 0.0073 | 0.4680 | 0.7812 |
| B | 1 | 43 | all | 1423 | 1819 | 4 | 0.6430 | 0.0038 | 0.3234 | 0.6325 | 0.0049 | 0.3187 | 0.8948 |
| B | 1 | 44 | 01 | 254 | 287 | 3 | 0.3701 | 0.0000 | 0.1850 | 0.4134 | 0.0000 | 0.2067 | 1.0000 |
| B | 1 | 44 | 02 | 244 | 300 | 3 | 0.0041 | 0.0000 | 0.0020 | 0.0123 | 0.0000 | 0.0061 | 1.0000 |
| B | 1 | 44 | 03 | 925 | 1232 | 4 | 0.9892 | 0.0049 | 0.4970 | 1.0000 | 0.0041 | 0.5020 | 0.3809 |
| B | 1 | 44 | all | 1423 | 1819 | 4 | 0.7098 | 0.0033 | 0.3565 | 0.7259 | 0.0027 | 0.3643 | 0.6400 |
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
| B | 3 | 44 | 01 | 276 | 187 | 2 | 0.9964 | 0.9733 | 0.9848 | 1.0000 | 0.5401 | 0.7701 | 0.0004 |
| B | 3 | 44 | 02 | 246 | 200 | 2 | 0.8333 | 0.9700 | 0.9017 | 0.9268 | 0.7900 | 0.8584 | 0.0784 |
| B | 3 | 44 | 03 | 865 | 1665 | 4 | 0.7457 | 0.5652 | 0.6554 | 0.8197 | 0.3225 | 0.5711 | 0.3959 |
| B | 3 | 44 | all | 1387 | 2052 | 4 | 0.8111 | 0.6418 | 0.7265 | 0.8745 | 0.3879 | 0.6312 | 0.2781 |
| B | 4 | 42 | 01 | 125 | 194 | 2 | 0.0000 | 0.2113 | 0.1057 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B | 4 | 42 | 02 | 247 | 200 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.1255 | 0.0000 | 0.0628 | 1.0000 |
| B | 4 | 42 | 03 | 553 | 1500 | 4 | 0.7161 | 0.4260 | 0.5710 | 0.8626 | 0.1333 | 0.4980 | 0.5357 |
| B | 4 | 42 | all | 925 | 1894 | 4 | 0.4281 | 0.3590 | 0.3936 | 0.5492 | 0.1056 | 0.3274 | 0.7502 |
| B | 4 | 43 | 01 | 125 | 194 | 2 | 0.0000 | 0.2165 | 0.1082 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B | 4 | 43 | 02 | 247 | 200 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B | 4 | 43 | 03 | 553 | 1500 | 4 | 0.8626 | 0.4007 | 0.6316 | 0.8626 | 0.0747 | 0.4686 | 0.2230 |
| B | 4 | 43 | all | 925 | 1894 | 4 | 0.5157 | 0.3395 | 0.4276 | 0.5157 | 0.0591 | 0.2874 | 0.5968 |
| B | 4 | 44 | 01 | 125 | 194 | 2 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.2320 | 0.1160 | 1.0000 |
| B | 4 | 44 | 02 | 247 | 200 | 2 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0450 | 0.0225 | 0.9994 |
| B | 4 | 44 | 03 | 553 | 1500 | 4 | 1.0000 | 0.0000 | 0.5000 | 0.8626 | 0.5953 | 0.7290 | 0.1344 |
| B | 4 | 44 | all | 925 | 1894 | 4 | 1.0000 | 0.0000 | 0.5000 | 0.5157 | 0.5000 | 0.5078 | 0.4672 |
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
| A* | 2 | 43 | 01 | 234 | 187 | 2 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 | 0.5000 |
| A* | 2 | 43 | 02 | 117 | 193 | 2 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 | 0.5000 |
| A* | 2 | 43 | 03 | 1019 | 1364 | 4 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 | 0.5000 |
| A* | 2 | 43 | all | 1370 | 1744 | 4 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 | 0.5000 |
| A* | 2 | 44 | 01 | 234 | 187 | 2 | 0.0000 | 1.0000 | 0.5000 | 0.5171 | 0.5294 | 0.5233 | 0.2411 |
| A* | 2 | 44 | 02 | 117 | 193 | 2 | 0.0000 | 1.0000 | 0.5000 | 0.9915 | 0.3886 | 0.6900 | 0.1166 |
| A* | 2 | 44 | 03 | 1019 | 1364 | 4 | 0.0000 | 1.0000 | 0.5000 | 0.9990 | 0.1195 | 0.5593 | 0.2646 |
| A* | 2 | 44 | all | 1370 | 1744 | 4 | 0.0000 | 1.0000 | 0.5000 | 0.9161 | 0.1932 | 0.5546 | 0.2782 |
| A* | 3 | 42 | 01 | 276 | 187 | 2 | 0.0000 | 1.0000 | 0.5000 | 0.6884 | 0.4973 | 0.5929 | 0.5035 |
| A* | 3 | 42 | 02 | 246 | 200 | 2 | 0.0000 | 1.0000 | 0.5000 | 0.7602 | 0.4300 | 0.5951 | 0.4273 |
| A* | 3 | 42 | 03 | 865 | 1665 | 4 | 0.0000 | 1.0000 | 0.5000 | 0.5584 | 0.2925 | 0.4254 | 0.5029 |
| A* | 3 | 42 | all | 1387 | 2052 | 4 | 0.0000 | 1.0000 | 0.5000 | 0.6200 | 0.3246 | 0.4723 | 0.5255 |
| A* | 3 | 43 | 01 | 276 | 187 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| A* | 3 | 43 | 02 | 246 | 200 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| A* | 3 | 43 | 03 | 865 | 1665 | 4 | 0.0613 | 0.4054 | 0.2333 | 0.0613 | 0.4090 | 0.2351 | 0.9113 |
| A* | 3 | 43 | all | 1387 | 2052 | 4 | 0.0382 | 0.3289 | 0.1836 | 0.0382 | 0.3319 | 0.1850 | 0.9534 |
| A* | 3 | 44 | 01 | 276 | 187 | 2 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 | 0.5000 |
| A* | 3 | 44 | 02 | 246 | 200 | 2 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 | 0.5000 |
| A* | 3 | 44 | 03 | 865 | 1665 | 4 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 | 0.5000 |
| A* | 3 | 44 | all | 1387 | 2052 | 4 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 | 0.5000 |
| A* | 4 | 42 | 01 | 125 | 194 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0052 | 0.0026 | 1.0000 |
| A* | 4 | 42 | 02 | 247 | 200 | 2 | 0.5668 | 0.0000 | 0.2834 | 0.0000 | 0.0050 | 0.0025 | 1.0000 |
| A* | 4 | 42 | 03 | 553 | 1500 | 4 | 0.0000 | 0.2540 | 0.1270 | 0.0000 | 0.4813 | 0.2407 | 0.9714 |
| A* | 4 | 42 | all | 925 | 1894 | 4 | 0.1514 | 0.2012 | 0.1763 | 0.0000 | 0.3823 | 0.1911 | 0.9271 |
| A* | 4 | 43 | 01 | 125 | 194 | 2 | 0.0000 | 0.0052 | 0.0026 | 0.0000 | 0.0052 | 0.0026 | 1.0000 |
| A* | 4 | 43 | 02 | 247 | 200 | 2 | 0.0000 | 0.0050 | 0.0025 | 0.0000 | 0.0050 | 0.0025 | 1.0000 |
| A* | 4 | 43 | 03 | 553 | 1500 | 4 | 0.0108 | 0.4393 | 0.2251 | 0.0108 | 0.4367 | 0.2238 | 0.8788 |
| A* | 4 | 43 | all | 925 | 1894 | 4 | 0.0065 | 0.3490 | 0.1777 | 0.0065 | 0.3469 | 0.1767 | 0.9227 |
| A* | 4 | 44 | 01 | 125 | 194 | 2 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 | 0.5000 |
| A* | 4 | 44 | 02 | 247 | 200 | 2 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 | 0.5000 |
| A* | 4 | 44 | 03 | 553 | 1500 | 4 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 | 0.5000 |
| A* | 4 | 44 | all | 925 | 1894 | 4 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 | 0.5000 |
| B' | 1 | 42 | 01 | 254 | 287 | 3 | 0.0000 | 0.0418 | 0.0209 | 0.0000 | 0.0035 | 0.0017 | 1.0000 |
| B' | 1 | 42 | 02 | 244 | 300 | 3 | 0.0000 | 0.0967 | 0.0483 | 0.5123 | 0.0000 | 0.2561 | 0.9932 |
| B' | 1 | 42 | 03 | 925 | 1232 | 4 | 0.4919 | 0.9513 | 0.7216 | 1.0000 | 0.2224 | 0.6112 | 0.1698 |
| B' | 1 | 42 | all | 1423 | 1819 | 4 | 0.3197 | 0.6668 | 0.4933 | 0.7379 | 0.1512 | 0.4445 | 0.5029 |
| B' | 1 | 43 | 01 | 254 | 287 | 3 | 0.3701 | 0.0000 | 0.1850 | 0.3622 | 0.0035 | 0.1828 | 0.8183 |
| B' | 1 | 43 | 02 | 244 | 300 | 3 | 0.5943 | 0.0033 | 0.2988 | 0.5000 | 0.0067 | 0.2533 | 0.7476 |
| B' | 1 | 43 | 03 | 925 | 1232 | 4 | 1.0000 | 0.0114 | 0.5057 | 1.0000 | 0.0479 | 0.5239 | 0.4761 |
| B' | 1 | 43 | all | 1423 | 1819 | 4 | 0.8180 | 0.0082 | 0.4131 | 0.8004 | 0.0341 | 0.4173 | 0.5857 |
| B' | 1 | 44 | 01 | 254 | 287 | 3 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.0000 | 0.5000 | 0.5000 |
| B' | 1 | 44 | 02 | 244 | 300 | 3 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.0000 | 0.5000 | 0.5000 |
| B' | 1 | 44 | 03 | 925 | 1232 | 4 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.0000 | 0.5000 | 0.5000 |
| B' | 1 | 44 | all | 1423 | 1819 | 4 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.0000 | 0.5000 | 0.5000 |
| B' | 2 | 42 | 01 | 234 | 187 | 2 | 0.5342 | 0.0000 | 0.2671 | 0.1581 | 0.0107 | 0.0844 | 0.9962 |
| B' | 2 | 42 | 02 | 117 | 193 | 2 | 0.0085 | 0.0104 | 0.0095 | 0.0000 | 0.0259 | 0.0130 | 0.9947 |
| B' | 2 | 42 | 03 | 1019 | 1364 | 4 | 0.0059 | 0.1928 | 0.0994 | 0.0000 | 0.7588 | 0.3794 | 0.9685 |
| B' | 2 | 42 | all | 1370 | 1744 | 4 | 0.0964 | 0.1519 | 0.1241 | 0.0270 | 0.5975 | 0.3122 | 0.9281 |
| B' | 2 | 43 | 01 | 234 | 187 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B' | 2 | 43 | 02 | 117 | 193 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0085 | 0.0000 | 0.0043 | 1.0000 |
| B' | 2 | 43 | 03 | 1019 | 1364 | 4 | 0.7341 | 0.2632 | 0.4986 | 0.8145 | 0.0924 | 0.4534 | 0.3946 |
| B' | 2 | 43 | all | 1370 | 1744 | 4 | 0.5460 | 0.2058 | 0.3759 | 0.6066 | 0.0722 | 0.3394 | 0.6391 |
| B' | 2 | 44 | 01 | 234 | 187 | 2 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 | 0.5000 |
| B' | 2 | 44 | 02 | 117 | 193 | 2 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 | 0.5000 |
| B' | 2 | 44 | 03 | 1019 | 1364 | 4 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 | 0.5000 |
| B' | 2 | 44 | all | 1370 | 1744 | 4 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 | 0.5000 |
| B' | 3 | 42 | 01 | 276 | 187 | 2 | 0.2935 | 0.0000 | 0.1467 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B' | 3 | 42 | 02 | 246 | 200 | 2 | 0.0935 | 0.0000 | 0.0467 | 0.0041 | 0.0200 | 0.0120 | 0.9998 |
| B' | 3 | 42 | 03 | 865 | 1665 | 4 | 0.8393 | 0.0793 | 0.4593 | 0.5584 | 0.3694 | 0.4639 | 0.5306 |
| B' | 3 | 42 | all | 1387 | 2052 | 4 | 0.5984 | 0.0643 | 0.3314 | 0.3490 | 0.3017 | 0.3253 | 0.7411 |
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
| B' | 4 | 44 | 01 | 125 | 194 | 2 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 | 0.5000 |
| B' | 4 | 44 | 02 | 247 | 200 | 2 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 | 0.5000 |
| B' | 4 | 44 | 03 | 553 | 1500 | 4 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 | 0.5000 |
| B' | 4 | 44 | all | 925 | 1894 | 4 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 | 0.5000 |


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
| arm_a_star_fold1_seed42 | 1400 | 400 | 12 | 10 | end of epochs (set_weights) | 0.0104 | 363 | 0.895917 | 0.0000 | 0.0000 |
| arm_a_star_fold1_seed43 | 1400 | 400 | 12 | 9 | end of epochs (set_weights) | 0.0165 | 371 | 0.749943 | 0.0050 | 0.0050 |
| arm_a_star_fold1_seed44 | 1400 | 400 | 12 | 11 | end of epochs (set_weights) | 0.0345 | 386 | 0.264618 | 0.0100 | 0.0100 |
| arm_a_star_fold2_seed42 | 1400 | 400 | 12 | 11 | end of epochs (set_weights) | 0.0001 | 405 | 0.000383 | 0.0000 | 0.0000 |
| arm_a_star_fold2_seed43 | 1400 | 400 | 5 | 1 | early stopping | 60.9868 | 168 | 1.000000 | 0.0000 | 1.0000 |
| arm_a_star_fold2_seed44 | 1400 | 400 | 5 | 1 | early stopping | 18.0934 | 167 | 0.000000 | 0.6450 | 0.6450 |
| arm_a_star_fold3_seed42 | 1400 | 400 | 5 | 1 | early stopping | 3.1296 | 182 | 0.001888 | 0.4800 | 0.4800 |
| arm_a_star_fold3_seed43 | 1400 | 400 | 12 | 12 | end of epochs (set_weights) | 0.0088 | 378 | 0.517414 | 0.0000 | 0.0000 |
| arm_a_star_fold3_seed44 | 1400 | 400 | 5 | 1 | early stopping | 33.4090 | 162 | 1.000000 | 0.0000 | 1.0000 |
| arm_a_star_fold4_seed42 | 1400 | 400 | 12 | 10 | end of epochs (set_weights) | 0.8106 | 359 | 0.999374 | 0.0200 | 0.0200 |
| arm_a_star_fold4_seed43 | 1400 | 400 | 12 | 9 | end of epochs (set_weights) | 0.0272 | 381 | 0.477774 | 0.0050 | 0.0050 |
| arm_a_star_fold4_seed44 | 1400 | 400 | 5 | 1 | early stopping | 22.8274 | 150 | 1.000000 | 0.0000 | 1.0000 |
| arm_b_fold1_seed42 | 1400 | 400 | 5 | 1 | early stopping | 7.0611 | 131 | 1.000000 | 0.0050 | 0.0000 |
| arm_b_fold1_seed43 | 1400 | 400 | 12 | 11 | end of epochs (set_weights) | 0.0056 | 373 | 0.542399 | 0.0050 | 0.0050 |
| arm_b_fold1_seed44 | 1400 | 400 | 10 | 6 | early stopping | 0.0298 | 312 | 0.230382 | 0.0050 | 0.0050 |
| arm_b_fold2_seed42 | 1400 | 400 | 12 | 9 | end of epochs (set_weights) | 0.2308 | 394 | 0.920426 | 0.0400 | 0.0400 |
| arm_b_fold2_seed43 | 1400 | 400 | 11 | 7 | early stopping | 0.0011 | 360 | 0.046983 | 0.0000 | 0.0000 |
| arm_b_fold2_seed44 | 1400 | 400 | 12 | 10 | end of epochs (set_weights) | 0.0030 | 403 | 0.578237 | 0.0000 | 0.0000 |
| arm_b_fold3_seed42 | 1400 | 400 | 12 | 12 | end of epochs (set_weights) | 0.0168 | 442 | 0.381979 | 0.0000 | 0.0000 |
| arm_b_fold3_seed43 | 1400 | 400 | 12 | 10 | end of epochs (set_weights) | 0.0531 | 423 | 0.031980 | 0.0000 | 0.0000 |
| arm_b_fold3_seed44 | 1400 | 400 | 7 | 3 | early stopping | 3.2731 | 256 | 0.120550 | 0.7800 | 0.7800 |
| arm_b_fold4_seed42 | 1400 | 400 | 12 | 11 | end of epochs (set_weights) | 0.0005 | 360 | 0.016334 | 0.0000 | 0.0000 |
| arm_b_fold4_seed43 | 1400 | 400 | 10 | 6 | early stopping | 0.0013 | 298 | 0.002956 | 0.0000 | 0.0000 |
| arm_b_fold4_seed44 | 1400 | 400 | 6 | 2 | early stopping | 2.3605 | 176 | 0.997861 | 0.0900 | 0.0900 |
| arm_b_prime_fold1_seed42 | 934 | 400 | 6 | 2 | early stopping | 0.6101 | 121 | 0.000583 | 0.0050 | 0.0050 |
| arm_b_prime_fold1_seed43 | 934 | 400 | 8 | 4 | early stopping | 0.8647 | 161 | 1.000000 | 0.0150 | 0.0150 |
| arm_b_prime_fold1_seed44 | 934 | 400 | 5 | 1 | early stopping | 12.1449 | 101 | 0.999998 | 1.0000 | 0.0050 |
| arm_b_prime_fold2_seed42 | 934 | 400 | 12 | 11 | end of epochs (set_weights) | 0.8750 | 259 | 0.999752 | 0.0850 | 0.0850 |
| arm_b_prime_fold2_seed43 | 934 | 400 | 12 | 10 | end of epochs (set_weights) | 0.0042 | 276 | 0.143735 | 0.0000 | 0.0000 |
| arm_b_prime_fold2_seed44 | 934 | 400 | 6 | 2 | early stopping | 11.4259 | 137 | 1.000000 | 0.0000 | 1.0000 |
| arm_b_prime_fold3_seed42 | 934 | 400 | 5 | 1 | early stopping | 0.3211 | 103 | 0.745805 | 0.0000 | 0.0000 |
| arm_b_prime_fold3_seed43 | 934 | 400 | 12 | 9 | end of epochs (set_weights) | 0.0011 | 248 | 0.013271 | 0.0000 | 0.0000 |
| arm_b_prime_fold3_seed44 | 934 | 400 | 12 | 9 | end of epochs (set_weights) | 0.0341 | 249 | 0.561295 | 0.0000 | 0.0000 |
| arm_b_prime_fold4_seed42 | 934 | 400 | 6 | 2 | early stopping | 0.1375 | 125 | 0.763160 | 0.0050 | 0.0050 |
| arm_b_prime_fold4_seed43 | 934 | 400 | 7 | 3 | early stopping | 0.0142 | 147 | 0.322349 | 0.0000 | 0.0000 |
| arm_b_prime_fold4_seed44 | 934 | 400 | 5 | 1 | early stopping | 17.2907 | 113 | 1.000000 | 0.0000 | 1.0000 |


## Steps, budget, environment and inputs

| step | seconds (wall clock) | seconds (monotonic) |
|---|---|---|
| check-inputs | 2 | 2 |
| train arm_b_fold1_seed42 | 132 | 132 |
| threshold arm_b_fold1_seed42 | 4 | 4 |
| score arm_b_fold1_seed42 | 14 | 14 |
| train arm_a_star_fold1_seed42 | 364 | 364 |
| threshold arm_a_star_fold1_seed42 | 4 | 4 |
| score arm_a_star_fold1_seed42 | 16 | 16 |
| train arm_b_fold1_seed43 | 373 | 373 |
| threshold arm_b_fold1_seed43 | 4 | 4 |
| score arm_b_fold1_seed43 | 16 | 16 |
| train arm_a_star_fold1_seed43 | 371 | 371 |
| threshold arm_a_star_fold1_seed43 | 4 | 4 |
| score arm_a_star_fold1_seed43 | 16 | 16 |
| train arm_b_fold1_seed44 | 313 | 313 |
| threshold arm_b_fold1_seed44 | 4 | 4 |
| score arm_b_fold1_seed44 | 16 | 16 |
| train arm_a_star_fold1_seed44 | 387 | 387 |
| threshold arm_a_star_fold1_seed44 | 4 | 4 |
| score arm_a_star_fold1_seed44 | 16 | 16 |
| train arm_b_fold2_seed42 | 394 | 394 |
| threshold arm_b_fold2_seed42 | 4 | 4 |
| score arm_b_fold2_seed42 | 16 | 16 |
| train arm_a_star_fold2_seed42 | 405 | 405 |
| threshold arm_a_star_fold2_seed42 | 4 | 4 |
| score arm_a_star_fold2_seed42 | 16 | 16 |
| train arm_b_fold2_seed43 | 361 | 361 |
| threshold arm_b_fold2_seed43 | 4 | 4 |
| score arm_b_fold2_seed43 | 16 | 16 |
| train arm_a_star_fold2_seed43 | 169 | 169 |
| threshold arm_a_star_fold2_seed43 | 4 | 4 |
| score arm_a_star_fold2_seed43 | 15 | 15 |
| train arm_b_fold2_seed44 | 404 | 404 |
| threshold arm_b_fold2_seed44 | 4 | 4 |
| score arm_b_fold2_seed44 | 16 | 16 |
| train arm_a_star_fold2_seed44 | 168 | 168 |
| threshold arm_a_star_fold2_seed44 | 5 | 5 |
| score arm_a_star_fold2_seed44 | 17 | 17 |
| train arm_b_fold3_seed42 | 444 | 444 |
| threshold arm_b_fold3_seed42 | 4 | 4 |
| score arm_b_fold3_seed42 | 21 | 21 |
| train arm_a_star_fold3_seed42 | 183 | 183 |
| threshold arm_a_star_fold3_seed42 | 8 | 8 |
| score arm_a_star_fold3_seed42 | 19 | 19 |
| train arm_b_fold3_seed43 | 425 | 425 |
| threshold arm_b_fold3_seed43 | 4 | 4 |
| score arm_b_fold3_seed43 | 16 | 16 |
| train arm_a_star_fold3_seed43 | 379 | 379 |
| threshold arm_a_star_fold3_seed43 | 4 | 4 |
| score arm_a_star_fold3_seed43 | 19 | 19 |
| train arm_b_fold3_seed44 | 257 | 257 |
| threshold arm_b_fold3_seed44 | 4 | 4 |
| score arm_b_fold3_seed44 | 18 | 18 |
| train arm_a_star_fold3_seed44 | 163 | 163 |
| threshold arm_a_star_fold3_seed44 | 4 | 4 |
| score arm_a_star_fold3_seed44 | 17 | 17 |
| train arm_b_fold4_seed42 | 360 | 360 |
| threshold arm_b_fold4_seed42 | 4 | 4 |
| score arm_b_fold4_seed42 | 14 | 14 |
| train arm_a_star_fold4_seed42 | 360 | 360 |
| threshold arm_a_star_fold4_seed42 | 4 | 4 |
| score arm_a_star_fold4_seed42 | 13 | 13 |
| train arm_b_fold4_seed43 | 298 | 298 |
| threshold arm_b_fold4_seed43 | 4 | 4 |
| score arm_b_fold4_seed43 | 14 | 14 |
| train arm_a_star_fold4_seed43 | 382 | 382 |
| threshold arm_a_star_fold4_seed43 | 4 | 4 |
| score arm_a_star_fold4_seed43 | 13 | 13 |
| train arm_b_fold4_seed44 | 177 | 177 |
| threshold arm_b_fold4_seed44 | 4 | 4 |
| score arm_b_fold4_seed44 | 13 | 13 |
| train arm_a_star_fold4_seed44 | 151 | 151 |
| threshold arm_a_star_fold4_seed44 | 4 | 4 |
| score arm_a_star_fold4_seed44 | 14 | 14 |
| train arm_b_prime_fold1_seed42 | 122 | 122 |
| threshold arm_b_prime_fold1_seed42 | 4 | 4 |
| score arm_b_prime_fold1_seed42 | 16 | 16 |
| train arm_b_prime_fold1_seed43 | 162 | 162 |
| threshold arm_b_prime_fold1_seed43 | 4 | 4 |
| score arm_b_prime_fold1_seed43 | 15 | 15 |
| train arm_b_prime_fold1_seed44 | 101 | 101 |
| threshold arm_b_prime_fold1_seed44 | 4 | 4 |
| score arm_b_prime_fold1_seed44 | 15 | 15 |
| train arm_b_prime_fold2_seed42 | 260 | 260 |
| threshold arm_b_prime_fold2_seed42 | 4 | 4 |
| score arm_b_prime_fold2_seed42 | 16 | 16 |
| train arm_b_prime_fold2_seed43 | 277 | 277 |
| threshold arm_b_prime_fold2_seed43 | 4 | 4 |
| score arm_b_prime_fold2_seed43 | 17 | 17 |
| train arm_b_prime_fold2_seed44 | 138 | 138 |
| threshold arm_b_prime_fold2_seed44 | 4 | 4 |
| score arm_b_prime_fold2_seed44 | 15 | 15 |
| train arm_b_prime_fold3_seed42 | 104 | 104 |
| threshold arm_b_prime_fold3_seed42 | 4 | 4 |
| score arm_b_prime_fold3_seed42 | 18 | 18 |
| train arm_b_prime_fold3_seed43 | 249 | 249 |
| threshold arm_b_prime_fold3_seed43 | 4 | 4 |
| score arm_b_prime_fold3_seed43 | 16 | 16 |
| train arm_b_prime_fold3_seed44 | 250 | 250 |
| threshold arm_b_prime_fold3_seed44 | 4 | 4 |
| score arm_b_prime_fold3_seed44 | 16 | 16 |
| train arm_b_prime_fold4_seed42 | 126 | 126 |
| threshold arm_b_prime_fold4_seed42 | 4 | 4 |
| score arm_b_prime_fold4_seed42 | 14 | 14 |
| train arm_b_prime_fold4_seed43 | 148 | 148 |
| threshold arm_b_prime_fold4_seed43 | 4 | 4 |
| score arm_b_prime_fold4_seed43 | 14 | 14 |
| train arm_b_prime_fold4_seed44 | 114 | 114 |
| threshold arm_b_prime_fold4_seed44 | 5 | 5 |
| score arm_b_prime_fold4_seed44 | 15 | 15 |
| aggregate | 2 | 2 |

Wall time 10209 s of the cap 21600 s (checked before every training, at the end of every epoch, before every threshold and every test scoring; the aggregation and the final writes run after the last model, also after the cap; cap already exceeded before them: no). Environment: {"keras": "2.15.0", "numpy": "1.26.4", "opencv": "4.11.0", "platform": "darwin", "python": "3.11.16", "tensorflow": "2.15.0", "tensorflow-metal": "1.1.0", "tensorflow_installed": "2.15.0"}.

Inputs (SHA-256): cells run restricted SHA256SUMS `0904ee72546160eadb493a94d8489a8fb21687006cd8fe1970b26007c567723c`, fold schema restricted SHA256SUMS `f3b30ae75a7167646544ef524225749bbded99e55d2249b9415eef2481e0d0f1`, cache `dc286dc371e61d638da1842c062a38c5368a9ec4da07ccbc36b7fff6f3177920`, extracted frames `c3fd9903589b7b074c46880870ba87924763f244846c744b657f9aaa3ebeee77`, architecture `a83c02b0788bd6f98262abf00f5fe970abed501f231edba348820482aea64d30`.

## Limits

Bootstrap: generator numpy.random.default_rng(0) (PCG64), restarted for every seed, comparison (or arm) and stratum; the held-out subjects of the stratum (pooled over the four folds) in ascending identifier order; at every resample integers(0, n, n) with n = the number of subjects; every metric computed on the pooled frames of the resampled subjects (through per-subject counts and pairwise Mann-Whitney counts, identical to recomputing it on the frames); the same subject indices for both arms of a comparison and for every metric; percentiles 2.5 and 97.5 with np.percentile, linear interpolation; a resample without a class does not count for the metrics that need it; no interval with fewer than 5 subjects. the session-03 bona fide cell of S12 comes from 1-2 subjects per fold; two dev cells rest on one subject; frames of one subject and session are correlated (n counts frames); three seeds give a range; the four folds are not independent (the dev of a fold is the test of the next one); the dev without session 03 calibrates early stopping and the EER threshold on the session 01-02 distribution for every arm; the pooled strata hold 9 (sessions 01, 02) or 16 (03, all) subjects; one dataset, one attack type (print)
