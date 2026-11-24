# Appendix A. Per seed, per unit and per fold: patience 4, threshold 0.5

Reported apart, outside the reading (`PROTOCOL.md` §6 (d)). Values in percentage points, as in Table 1; full
precision in `aggregate.csv`. Each seed has 4 units, one per fold.

## Seed 42

**(a)** APCER, BPCER and ACER of each network, mean over the 4 units of the seed.

| network | BN state | APCER (%) | BPCER (%) | ACER (%) | units | attack frames | bona fide frames | subjects |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LivenessNet | ordinary | 15.63 | 53.29 | 34.46 | 4 | 7,509 | 5,105 | 16 |
| LivenessNet | re-estimated | 22.97 | 30.90 | 26.93 | 4 | 7,509 | 5,105 | 16 |
| AttackNetV1 | ordinary | 28.19 | 34.28 | 31.24 | 4 | 7,509 | 5,105 | 16 |
| AttackNetV1 | re-estimated | 49.80 | 20.80 | 35.30 | 4 | 7,509 | 5,105 | 16 |
| AttackNetV2_1 | ordinary | 29.75 | 39.82 | 34.79 | 4 | 7,509 | 5,105 | 16 |
| AttackNetV2_1 | re-estimated | 51.70 | 38.72 | 45.21 | 4 | 7,509 | 5,105 | 16 |
| AttackNetV2_2 | ordinary | 27.10 | 36.41 | 31.76 | 4 | 7,509 | 5,105 | 16 |
| AttackNetV2_2 | re-estimated | 34.17 | 31.59 | 32.88 | 4 | 7,509 | 5,105 | 16 |
| ResNet-18 | ordinary | 32.38 | 0.98 | 16.68 | 4 | 7,509 | 5,105 | 16 |
| ResNet-18 | re-estimated | 29.01 | 9.62 | 19.32 | 4 | 7,509 | 5,105 | 16 |

**(b)** ACER variations and contrasts, mean over the 4 units of the seed [subject interval]. The
intervals are those of `primary_analysis.per_seed` of `results.json`, over the same 16 subjects; no
reading is given per seed.

| variation of | APCER (points) | BPCER (points) | ACER (points) | ACER units above / below / at zero |
|---|---|---|---|---|
| LivenessNet | +7.34 [−0.80, +12.82] | −22.39 [−35.78, −8.99] | −7.52 [−16.15, +0.69] | 2 / 2 / 0 |
| AttackNetV1 | +21.61 [+15.72, +28.25] | −13.48 [−29.26, −0.55] | +4.07 [−4.88, +12.54] | 3 / 1 / 0 |
| AttackNetV2_1 | +21.95 [+14.17, +27.49] | −1.10 [−12.20, +9.75] | +10.43 [+4.09, +15.18] | 3 / 1 / 0 |
| AttackNetV2_2 | +7.08 [+1.46, +12.57] | −4.82 [−18.25, +7.87] | +1.13 [−5.81, +7.79] | 2 / 2 / 0 |
| ResNet-18 | −3.37 [−10.64, +3.25] | +8.64 [−0.04, +15.44] | +2.64 [−4.16, +7.46] | 3 / 1 / 0 |

| contrast of | APCER (points) | BPCER (points) | ACER (points) | ACER units above / below / at zero |
|---|---|---|---|---|
| AttackNetV1 | +14.27 [+6.85, +24.80] | +8.91 [−11.81, +28.92] | +11.59 [+1.27, +21.67] | 3 / 1 / 0 |
| AttackNetV2_1 | +14.61 [+4.83, +25.40] | +21.29 [+6.33, +38.02] | +17.95 [+8.93, +28.02] | 4 / 0 / 0 |
| AttackNetV2_2 | −0.26 [−8.08, +11.23] | +17.57 [+3.35, +36.67] | +8.65 [+0.25, +18.05] | 3 / 1 / 0 |

## Seed 43

**(a)** APCER, BPCER and ACER of each network, mean over the 4 units of the seed.

| network | BN state | APCER (%) | BPCER (%) | ACER (%) | units | attack frames | bona fide frames | subjects |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LivenessNet | ordinary | 15.31 | 52.41 | 33.86 | 4 | 7,509 | 5,105 | 16 |
| LivenessNet | re-estimated | 20.87 | 51.66 | 36.26 | 4 | 7,509 | 5,105 | 16 |
| AttackNetV1 | ordinary | 27.03 | 32.53 | 29.78 | 4 | 7,509 | 5,105 | 16 |
| AttackNetV1 | re-estimated | 46.51 | 28.63 | 37.57 | 4 | 7,509 | 5,105 | 16 |
| AttackNetV2_1 | ordinary | 22.53 | 54.26 | 38.39 | 4 | 7,509 | 5,105 | 16 |
| AttackNetV2_1 | re-estimated | 55.06 | 36.61 | 45.83 | 4 | 7,509 | 5,105 | 16 |
| AttackNetV2_2 | ordinary | 19.51 | 42.10 | 30.80 | 4 | 7,509 | 5,105 | 16 |
| AttackNetV2_2 | re-estimated | 41.17 | 39.54 | 40.36 | 4 | 7,509 | 5,105 | 16 |
| ResNet-18 | ordinary | 42.14 | 6.19 | 24.16 | 4 | 7,509 | 5,105 | 16 |
| ResNet-18 | re-estimated | 41.92 | 12.51 | 27.22 | 4 | 7,509 | 5,105 | 16 |

**(b)** ACER variations and contrasts, mean over the 4 units of the seed [subject interval]. The
intervals are those of `primary_analysis.per_seed` of `results.json`, over the same 16 subjects; no
reading is given per seed.

| variation of | APCER (points) | BPCER (points) | ACER (points) | ACER units above / below / at zero |
|---|---|---|---|---|
| LivenessNet | +5.56 [+1.61, +10.31] | −0.75 [−5.61, +2.30] | +2.40 [−1.49, +5.43] | 4 / 0 / 0 |
| AttackNetV1 | +19.49 [+14.93, +25.11] | −3.90 [−14.12, +2.72] | +7.79 [+2.84, +11.41] | 2 / 2 / 0 |
| AttackNetV2_1 | +32.52 [+22.92, +44.41] | −17.65 [−29.31, −6.43] | +7.44 [+1.57, +15.88] | 2 / 2 / 0 |
| AttackNetV2_2 | +21.66 [+18.07, +26.63] | −2.55 [−11.38, +11.34] | +9.55 [+4.43, +17.06] | 2 / 2 / 0 |
| ResNet-18 | −0.22 [−6.36, +7.50] | +6.33 [−11.06, +20.89] | +3.06 [−5.67, +10.25] | 3 / 1 / 0 |

| contrast of | APCER (points) | BPCER (points) | ACER (points) | ACER units above / below / at zero |
|---|---|---|---|---|
| AttackNetV1 | +13.93 [+7.93, +21.78] | −3.15 [−11.89, +2.57] | +5.39 [−0.35, +9.81] | 2 / 2 / 0 |
| AttackNetV2_1 | +26.97 [+16.71, +40.29] | −16.90 [−30.80, −1.76] | +5.04 [−2.54, +17.62] | 2 / 2 / 0 |
| AttackNetV2_2 | +16.10 [+10.93, +21.62] | −1.80 [−9.41, +10.61] | +7.15 [+2.54, +13.56] | 2 / 2 / 0 |

## Seed 44

**(a)** APCER, BPCER and ACER of each network, mean over the 4 units of the seed.

| network | BN state | APCER (%) | BPCER (%) | ACER (%) | units | attack frames | bona fide frames | subjects |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LivenessNet | ordinary | 16.29 | 71.16 | 43.72 | 4 | 7,509 | 5,105 | 16 |
| LivenessNet | re-estimated | 19.62 | 35.63 | 27.63 | 4 | 7,509 | 5,105 | 16 |
| AttackNetV1 | ordinary | 37.34 | 42.26 | 39.80 | 4 | 7,509 | 5,105 | 16 |
| AttackNetV1 | re-estimated | 56.75 | 47.52 | 52.14 | 4 | 7,509 | 5,105 | 16 |
| AttackNetV2_1 | ordinary | 32.90 | 37.51 | 35.20 | 4 | 7,509 | 5,105 | 16 |
| AttackNetV2_1 | re-estimated | 50.31 | 25.62 | 37.96 | 4 | 7,509 | 5,105 | 16 |
| AttackNetV2_2 | ordinary | 22.72 | 43.94 | 33.33 | 4 | 7,509 | 5,105 | 16 |
| AttackNetV2_2 | re-estimated | 52.04 | 20.45 | 36.25 | 4 | 7,509 | 5,105 | 16 |
| ResNet-18 | ordinary | 26.74 | 19.32 | 23.03 | 4 | 7,509 | 5,105 | 16 |
| ResNet-18 | re-estimated | 39.22 | 12.92 | 26.07 | 4 | 7,509 | 5,105 | 16 |

**(b)** ACER variations and contrasts, mean over the 4 units of the seed [subject interval]. The
intervals are those of `primary_analysis.per_seed` of `results.json`, over the same 16 subjects; no
reading is given per seed.

| variation of | APCER (points) | BPCER (points) | ACER (points) | ACER units above / below / at zero |
|---|---|---|---|---|
| LivenessNet | +3.34 [−7.96, +19.40] | −35.53 [−57.52, −16.19] | −16.09 [−31.57, +0.14] | 1 / 3 / 0 |
| AttackNetV1 | +19.41 [+11.27, +26.81] | +5.27 [−1.87, +8.87] | +12.34 [+7.28, +16.74] | 3 / 1 / 0 |
| AttackNetV2_1 | +17.41 [+11.26, +27.35] | −11.89 [−28.43, +4.57] | +2.76 [−5.91, +12.12] | 3 / 1 / 0 |
| AttackNetV2_2 | +29.32 [+21.81, +37.98] | −23.48 [−39.61, −10.06] | +2.92 [−5.34, +11.33] | 3 / 1 / 0 |
| ResNet-18 | +12.47 [+8.22, +18.32] | −6.39 [−16.96, +2.25] | +3.04 [−1.15, +7.72] | 4 / 0 / 0 |

| contrast of | APCER (points) | BPCER (points) | ACER (points) | ACER units above / below / at zero |
|---|---|---|---|---|
| AttackNetV1 | +16.07 [+0.63, +27.04] | +40.79 [+18.58, +63.59] | +28.43 [+10.82, +44.37] | 3 / 1 / 0 |
| AttackNetV2_1 | +14.07 [−3.07, +25.94] | +23.64 [−7.99, +56.61] | +18.86 [−2.96, +38.94] | 3 / 1 / 0 |
| AttackNetV2_2 | +25.99 [+10.98, +38.40] | +12.04 [−21.31, +42.66] | +19.01 [−4.02, +39.06] | 3 / 1 / 0 |

## Per unit

Each network × fold × seed: APCER, BPCER and ACER with ordinary (ord.) and re-estimated (re.) BN, the
ACER variation and, for each AttackNet, its ACER contrast with LivenessNet in the same unit. No interval.

| fold | seed | network | APCER ord. | BPCER ord. | ACER ord. | APCER re. | BPCER re. | ACER re. | ACER variation | ACER contrast |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42 | LivenessNet | 0.00 | 100.00 | 50.00 | 1.10 | 38.65 | 19.88 | −30.12 |  |
| 1 | 42 | AttackNetV1 | 11.00 | 64.93 | 37.96 | 20.73 | 62.47 | 41.60 | +3.64 | +33.76 |
| 1 | 42 | AttackNetV2_1 | 22.16 | 59.10 | 40.63 | 27.21 | 48.07 | 37.64 | −2.99 | +27.14 |
| 1 | 42 | AttackNetV2_2 | 5.11 | 73.79 | 39.45 | 15.01 | 49.68 | 32.35 | −7.10 | +23.02 |
| 1 | 42 | ResNet-18 | 28.04 | 3.44 | 15.74 | 3.41 | 6.96 | 5.18 | −10.56 |  |
| 1 | 43 | LivenessNet | 0.38 | 64.30 | 32.34 | 0.27 | 70.48 | 35.38 | +3.04 |  |
| 1 | 43 | AttackNetV1 | 19.46 | 29.52 | 24.49 | 23.36 | 22.21 | 22.79 | −1.70 | −4.74 |
| 1 | 43 | AttackNetV2_1 | 1.15 | 90.93 | 46.04 | 37.66 | 32.54 | 35.10 | −10.95 | −13.98 |
| 1 | 43 | AttackNetV2_2 | 2.86 | 38.30 | 20.58 | 4.78 | 60.65 | 32.71 | +12.14 | +9.10 |
| 1 | 43 | ResNet-18 | 11.43 | 24.60 | 18.02 | 26.83 | 0.00 | 13.41 | −4.60 |  |
| 1 | 44 | LivenessNet | 0.33 | 70.98 | 35.65 | 9.02 | 53.69 | 31.35 | −4.30 |  |
| 1 | 44 | AttackNetV1 | 36.78 | 59.52 | 48.15 | 25.51 | 58.12 | 41.81 | −6.34 | −2.04 |
| 1 | 44 | AttackNetV2_1 | 26.39 | 34.65 | 30.52 | 20.62 | 47.86 | 34.24 | +3.72 | +8.02 |
| 1 | 44 | AttackNetV2_2 | 9.46 | 35.00 | 22.23 | 4.73 | 41.25 | 22.99 | +0.76 | +5.06 |
| 1 | 44 | ResNet-18 | 2.97 | 0.14 | 1.55 | 0.44 | 6.75 | 3.59 | +2.04 |  |
| 2 | 42 | LivenessNet | 16.23 | 38.91 | 27.57 | 40.08 | 1.97 | 21.03 | −6.54 |  |
| 2 | 42 | AttackNetV1 | 42.20 | 2.77 | 22.49 | 68.75 | 1.46 | 35.10 | +12.62 | +19.16 |
| 2 | 42 | AttackNetV2_1 | 31.19 | 16.86 | 24.03 | 55.39 | 19.49 | 37.44 | +13.41 | +19.95 |
| 2 | 42 | AttackNetV2_2 | 26.72 | 11.82 | 19.27 | 51.61 | 20.29 | 35.95 | +16.68 | +23.22 |
| 2 | 42 | ResNet-18 | 35.15 | 0.22 | 17.68 | 45.07 | 0.07 | 22.57 | +4.89 |  |
| 2 | 43 | LivenessNet | 6.88 | 62.70 | 34.79 | 20.18 | 53.94 | 37.06 | +2.27 |  |
| 2 | 43 | AttackNetV1 | 20.13 | 37.96 | 29.04 | 60.49 | 40.58 | 50.54 | +21.50 | +19.23 |
| 2 | 43 | AttackNetV2_1 | 37.21 | 36.86 | 37.04 | 69.21 | 43.87 | 56.54 | +19.50 | +17.23 |
| 2 | 43 | AttackNetV2_2 | 17.78 | 53.28 | 35.53 | 49.43 | 14.96 | 32.20 | −3.33 | −5.61 |
| 2 | 43 | ResNet-18 | 48.80 | 0.15 | 24.47 | 68.58 | 0.00 | 34.29 | +9.82 |  |
| 2 | 44 | LivenessNet | 0.63 | 32.55 | 16.59 | 12.27 | 3.50 | 7.89 | −8.71 |  |
| 2 | 44 | AttackNetV1 | 68.58 | 20.73 | 44.65 | 75.92 | 48.25 | 62.08 | +17.43 | +26.13 |
| 2 | 44 | AttackNetV2_1 | 16.80 | 39.27 | 28.04 | 55.16 | 8.83 | 32.00 | +3.96 | +12.67 |
| 2 | 44 | AttackNetV2_2 | 5.05 | 58.25 | 31.65 | 75.46 | 1.24 | 38.35 | +6.70 | +15.41 |
| 2 | 44 | ResNet-18 | 45.53 | 0.07 | 22.80 | 50.92 | 0.07 | 25.50 | +2.69 |  |
| 3 | 42 | LivenessNet | 10.38 | 31.43 | 20.91 | 16.96 | 32.16 | 24.56 | +3.65 |  |
| 3 | 42 | AttackNetV1 | 22.71 | 26.82 | 24.77 | 64.38 | 0.58 | 32.48 | +7.71 | +4.06 |
| 3 | 42 | AttackNetV2_1 | 46.59 | 31.65 | 39.12 | 69.35 | 34.46 | 51.90 | +12.79 | +9.14 |
| 3 | 42 | AttackNetV2_2 | 9.50 | 22.42 | 15.96 | 38.99 | 6.78 | 22.88 | +6.92 | +3.27 |
| 3 | 42 | ResNet-18 | 18.23 | 0.14 | 9.19 | 34.94 | 0.00 | 17.47 | +8.29 |  |
| 3 | 43 | LivenessNet | 20.03 | 31.07 | 25.55 | 20.61 | 31.07 | 25.84 | +0.29 |  |
| 3 | 43 | AttackNetV1 | 29.87 | 11.18 | 20.52 | 69.93 | 0.14 | 35.04 | +14.51 | +14.22 |
| 3 | 43 | AttackNetV2_1 | 23.25 | 28.05 | 25.65 | 80.41 | 30.57 | 55.49 | +29.84 | +29.55 |
| 3 | 43 | AttackNetV2_2 | 14.23 | 23.94 | 19.08 | 73.10 | 31.43 | 52.27 | +33.18 | +32.89 |
| 3 | 43 | ResNet-18 | 33.82 | 0.00 | 16.91 | 36.79 | 0.00 | 18.40 | +1.49 |  |
| 3 | 44 | LivenessNet | 64.18 | 81.11 | 72.65 | 3.51 | 34.32 | 18.91 | −53.73 |  |
| 3 | 44 | AttackNetV1 | 1.61 | 38.07 | 19.84 | 76.36 | 32.16 | 54.26 | +34.42 | +88.15 |
| 3 | 44 | AttackNetV2_1 | 42.15 | 24.44 | 33.30 | 82.89 | 0.07 | 41.48 | +8.19 | +61.92 |
| 3 | 44 | AttackNetV2_2 | 8.63 | 34.17 | 21.40 | 79.78 | 3.32 | 41.55 | +20.15 | +73.88 |
| 3 | 44 | ResNet-18 | 11.06 | 25.59 | 18.33 | 27.00 | 10.38 | 18.69 | +0.36 |  |
| 4 | 42 | LivenessNet | 35.90 | 42.81 | 39.36 | 33.74 | 50.81 | 42.27 | +2.92 |  |
| 4 | 42 | AttackNetV1 | 36.85 | 42.59 | 39.72 | 45.35 | 18.70 | 32.03 | −7.70 | −10.61 |
| 4 | 42 | AttackNetV2_1 | 19.06 | 51.68 | 35.37 | 54.86 | 52.86 | 53.86 | +18.49 | +15.58 |
| 4 | 42 | AttackNetV2_2 | 67.05 | 37.62 | 52.34 | 31.10 | 49.62 | 40.36 | −11.98 | −14.90 |
| 4 | 42 | ResNet-18 | 48.10 | 0.11 | 24.10 | 32.63 | 31.46 | 32.04 | +7.94 |  |
| 4 | 43 | LivenessNet | 33.95 | 51.57 | 42.76 | 42.40 | 51.14 | 46.77 | +4.01 |  |
| 4 | 43 | AttackNetV1 | 38.65 | 51.46 | 45.05 | 32.26 | 51.57 | 41.91 | −3.14 | −7.15 |
| 4 | 43 | AttackNetV2_1 | 28.51 | 61.19 | 44.85 | 32.95 | 39.46 | 36.20 | −8.65 | −12.65 |
| 4 | 43 | AttackNetV2_2 | 43.19 | 52.86 | 48.03 | 37.38 | 51.14 | 44.26 | −3.77 | −7.78 |
| 4 | 43 | ResNet-18 | 74.50 | 0.00 | 37.25 | 35.48 | 50.05 | 42.77 | +5.52 |  |
| 4 | 44 | LivenessNet | 0.00 | 100.00 | 50.00 | 53.70 | 51.03 | 52.36 | +2.36 |  |
| 4 | 44 | AttackNetV1 | 42.40 | 50.70 | 46.55 | 49.21 | 51.57 | 50.39 | +3.84 | +1.48 |
| 4 | 44 | AttackNetV2_1 | 46.25 | 51.68 | 48.96 | 42.56 | 45.73 | 44.14 | −4.82 | −7.18 |
| 4 | 44 | AttackNetV2_2 | 67.74 | 48.32 | 58.03 | 48.20 | 36.00 | 42.10 | −15.93 | −18.29 |
| 4 | 44 | ResNet-18 | 47.41 | 51.46 | 49.44 | 78.51 | 34.49 | 56.50 | +7.06 |  |

## Per fold

A choice of the author, outside the reading: for each fold, the mean over its 3 units (one per seed) of the
ACER variations and contrasts, and how many of the 3 unit values are above, below and at zero. No interval:
`results.json` holds none per fold.

| fold | quantity | ACER (points) | units above / below / at zero |
|---:|---|---:|---|
| 1 | variation of LivenessNet | −10.46 | 1 / 2 / 0 |
| 1 | variation of AttackNetV1 | −1.47 | 1 / 2 / 0 |
| 1 | variation of AttackNetV2_1 | −3.41 | 1 / 2 / 0 |
| 1 | variation of AttackNetV2_2 | +1.93 | 2 / 1 / 0 |
| 1 | variation of ResNet-18 | −4.37 | 1 / 2 / 0 |
| 1 | contrast of AttackNetV1 | +8.99 | 1 / 2 / 0 |
| 1 | contrast of AttackNetV2_1 | +7.06 | 2 / 1 / 0 |
| 1 | contrast of AttackNetV2_2 | +12.39 | 3 / 0 / 0 |
| 2 | variation of LivenessNet | −4.32 | 1 / 2 / 0 |
| 2 | variation of AttackNetV1 | +17.18 | 3 / 0 / 0 |
| 2 | variation of AttackNetV2_1 | +12.29 | 3 / 0 / 0 |
| 2 | variation of AttackNetV2_2 | +6.68 | 2 / 1 / 0 |
| 2 | variation of ResNet-18 | +5.80 | 3 / 0 / 0 |
| 2 | contrast of AttackNetV1 | +21.51 | 3 / 0 / 0 |
| 2 | contrast of AttackNetV2_1 | +16.62 | 3 / 0 / 0 |
| 2 | contrast of AttackNetV2_2 | +11.01 | 2 / 1 / 0 |
| 3 | variation of LivenessNet | −16.60 | 2 / 1 / 0 |
| 3 | variation of AttackNetV1 | +18.88 | 3 / 0 / 0 |
| 3 | variation of AttackNetV2_1 | +16.94 | 3 / 0 / 0 |
| 3 | variation of AttackNetV2_2 | +20.08 | 3 / 0 / 0 |
| 3 | variation of ResNet-18 | +3.38 | 3 / 0 / 0 |
| 3 | contrast of AttackNetV1 | +35.48 | 3 / 0 / 0 |
| 3 | contrast of AttackNetV2_1 | +33.53 | 3 / 0 / 0 |
| 3 | contrast of AttackNetV2_2 | +36.68 | 3 / 0 / 0 |
| 4 | variation of LivenessNet | +3.10 | 3 / 0 / 0 |
| 4 | variation of AttackNetV1 | −2.33 | 1 / 2 / 0 |
| 4 | variation of AttackNetV2_1 | +1.67 | 1 / 2 / 0 |
| 4 | variation of AttackNetV2_2 | −10.56 | 0 / 3 / 0 |
| 4 | variation of ResNet-18 | +6.84 | 3 / 0 / 0 |
| 4 | contrast of AttackNetV1 | −5.43 | 1 / 2 / 0 |
| 4 | contrast of AttackNetV2_1 | −1.42 | 1 / 2 / 0 |
| 4 | contrast of AttackNetV2_2 | −13.65 | 0 / 3 / 0 |
