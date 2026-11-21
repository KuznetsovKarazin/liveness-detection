# Appendix. Per seed: LivenessNet and ResNet-18 on the NUAA test, patience 4, threshold 0.5

Reported apart; not part of the reading (`PROTOCOL.md` §6). Each seed has 4 units, one per fold. Its subject
intervals are those of `primary_analysis.per_seed` of `results.json`, over the same 16 test subjects
(`n_subjects` of each block, checked), with valid resamples 926 for ACER, 950 for APCER, 940 for BPCER
(checked in every block); they are not recomputable from the public files. Values in percentage points, as
in Table 1; full precision in `per_seed.csv`.

**(a)** Each network: mean over the 4 units of the seed.

| seed | network | BN state | APCER (%) | BPCER (%) | ACER (%) |
|---|---|---|---:|---:|---:|
| 42 | LivenessNet | ordinary | 15.63 | 53.29 | 34.46 |
| 42 | ResNet-18 | ordinary | 32.38 | 0.98 | 16.68 |
| 42 | LivenessNet | re-estimated | 22.97 | 30.90 | 26.93 |
| 42 | ResNet-18 | re-estimated | 29.01 | 9.62 | 19.32 |
| 43 | LivenessNet | ordinary | 15.31 | 52.41 | 33.86 |
| 43 | ResNet-18 | ordinary | 42.14 | 6.19 | 24.16 |
| 43 | LivenessNet | re-estimated | 20.87 | 51.66 | 36.26 |
| 43 | ResNet-18 | re-estimated | 41.92 | 12.51 | 27.22 |
| 44 | LivenessNet | ordinary | 16.29 | 71.16 | 43.72 |
| 44 | ResNet-18 | ordinary | 26.74 | 19.32 | 23.03 |
| 44 | LivenessNet | re-estimated | 19.62 | 35.63 | 27.63 |
| 44 | ResNet-18 | re-estimated | 39.22 | 12.92 | 26.07 |

**(b)** Gaps (ResNet-18 − LivenessNet) and contrast (re-estimated − ordinary), mean over the 4 units of the
seed [subject interval]. Labels quoted from `results.json`: the rules of `PROTOCOL.md` §6 applied to each
seed, which do not enter the reading.

| seed | quantity | APCER (points) | BPCER (points) | ACER (points) | ACER units | label (rule of `PROTOCOL.md` §6 applied to the seed; not part of the reading) |
|---|---|---|---|---|---|---|
| 42 | gap, ordinary BN | +16.75 [+5.04, +36.12] | −52.31 [−74.18, −26.31] | −17.78 [−28.21, −2.27] | 4 / 0 / 0 | "ResNet-18 lower" (interval entirely below zero) |
| 42 | gap, re-estimated BN | +6.04 [−10.12, +29.60] | −21.27 [−39.57, −4.04] | −7.62 [−15.57, +4.79] | 3 / 1 / 0 | "unresolved" (interval contains zero) |
| 42 | contrast, re-estimated minus ordinary | −10.71 [−20.96, +1.39] | +31.03 [+14.09, +45.20] | +10.16 [−2.26, +22.63] | gap changes sign in 1 of 4 | "unresolved" (interval contains zero) |
| 43 | gap, ordinary BN | +26.83 [+17.85, +41.04] | −46.23 [−69.92, −17.58] | −9.70 [−19.61, +6.15] | 4 / 0 / 0 | "unresolved" (interval contains zero) |
| 43 | gap, re-estimated BN | +21.05 [+10.92, +37.31] | −39.15 [−57.28, −14.11] | −9.05 [−18.50, +4.53] | 4 / 0 / 0 | "unresolved" (interval contains zero) |
| 43 | contrast, re-estimated minus ordinary | −5.77 [−14.14, +4.53] | +7.08 [−11.11, +22.85] | +0.65 [−8.19, +7.94] | gap changes sign in 0 of 4 | "unresolved" (interval contains zero) |
| 44 | gap, ordinary BN | +10.46 [−0.40, +28.98] | −51.84 [−73.24, −26.57] | −20.69 [−34.72, −2.59] | 3 / 1 / 0 | "ResNet-18 lower" (interval entirely below zero) |
| 44 | gap, re-estimated BN | +19.59 [+11.12, +33.49] | −22.71 [−34.93, −6.23] | −1.56 [−7.96, +8.96] | 2 / 2 / 0 | "unresolved" (interval contains zero) |
| 44 | contrast, re-estimated minus ordinary | +9.14 [−4.81, +21.70] | +29.13 [−0.00, +58.34] | +19.13 [−1.08, +37.71] | gap changes sign in 1 of 4 | "unresolved" (interval contains zero) |

A value written −0.00 or +0.00 is non-zero and rounds to zero; its sign is kept (lower bound of the BPCER contrast of
seed 44: −0.0037 points).
