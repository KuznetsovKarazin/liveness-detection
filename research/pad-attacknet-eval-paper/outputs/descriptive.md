# Appendix C. Descriptive analyses: patience 12, the dev EER threshold, the comparison with ResNet-18, leaving
one unit out, rankings

Descriptive analyses (`PROTOCOL.md` §6 (g), (m), (o)): reported with their intervals and no reading; not
results of the primary analysis. Values in percentage points; intervals: individual 95% subject intervals,
copied from `results.json` (valid resamples 926 for ACER, 950 for APCER and 940 for BPCER); means and unit counts
recomputed from `units.csv`. The EER threshold is estimated on the dev of the fold (2–3 subjects) in the same
BN state.

## Patience 12, threshold 0.5 (descriptive)

APCER, BPCER and ACER of each network, mean over the 12 units:

| network | BN state | APCER (%) | BPCER (%) | ACER (%) | units | attack frames | bona fide frames | subjects |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LivenessNet | ordinary | 15.52 | 41.83 | 28.68 | 12 | 22,527 | 15,315 | 16 |
| LivenessNet | re-estimated | 18.12 | 36.94 | 27.53 | 12 | 22,527 | 15,315 | 16 |
| AttackNetV1 | ordinary | 29.17 | 31.94 | 30.55 | 12 | 22,527 | 15,315 | 16 |
| AttackNetV1 | re-estimated | 49.80 | 28.67 | 39.24 | 12 | 22,527 | 15,315 | 16 |
| AttackNetV2_1 | ordinary | 27.56 | 44.00 | 35.78 | 12 | 22,527 | 15,315 | 16 |
| AttackNetV2_1 | re-estimated | 50.99 | 31.44 | 41.21 | 12 | 22,527 | 15,315 | 16 |
| AttackNetV2_2 | ordinary | 28.67 | 28.65 | 28.66 | 12 | 22,527 | 15,315 | 16 |
| AttackNetV2_2 | re-estimated | 40.66 | 32.03 | 36.35 | 12 | 22,527 | 15,315 | 16 |
| ResNet-18 | ordinary | 29.81 | 6.93 | 18.37 | 12 | 22,527 | 15,315 | 16 |
| ResNet-18 | re-estimated | 34.92 | 12.93 | 23.92 | 12 | 22,527 | 15,315 | 16 |

ACER variations (re-estimated − ordinary BN) and contrasts (AttackNet − LivenessNet), mean [subject interval]:

| variation of | APCER (points) | BPCER (points) | ACER (points) | ACER units above / below / at zero |
|---|---|---|---|---|
| LivenessNet | +2.59 [−1.74, +6.18] | −4.90 [−7.76, +1.57] | −1.15 [−3.67, +3.12] | 7 / 5 / 0 |
| AttackNetV1 | +20.63 [+17.24, +24.92] | −3.26 [−10.28, +3.25] | +8.69 [+5.05, +12.59] | 8 / 4 / 0 |
| AttackNetV2_1 | +23.44 [+18.58, +30.00] | −12.57 [−17.67, −6.58] | +5.44 [+2.43, +9.26] | 8 / 4 / 0 |
| AttackNetV2_2 | +11.99 [+9.06, +15.14] | +3.39 [−3.00, +11.42] | +7.69 [+4.03, +11.61] | 8 / 4 / 0 |
| ResNet-18 | +5.11 [+0.97, +8.26] | +6.00 [−2.36, +11.28] | +5.55 [+0.98, +8.02] | 11 / 1 / 0 |

| contrast of | APCER (points) | BPCER (points) | ACER (points) | ACER units above / below / at zero |
|---|---|---|---|---|
| AttackNetV1 | +18.04 [+12.95, +25.06] | +1.64 [−9.70, +10.24] | +9.84 [+3.01, +14.51] | 9 / 3 / 0 |
| AttackNetV2_1 | +20.84 [+13.50, +31.25] | −7.67 [−13.13, −2.89] | +6.59 [+2.95, +11.24] | 8 / 4 / 0 |
| AttackNetV2_2 | +9.40 [+5.34, +14.74] | +8.29 [+1.87, +15.17] | +8.84 [+4.95, +13.14] | 9 / 3 / 0 |

## Patience 4, dev EER threshold (descriptive)

APCER, BPCER and ACER of each network, mean over the 12 units:

| network | BN state | APCER (%) | BPCER (%) | ACER (%) | units | attack frames | bona fide frames | subjects |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LivenessNet | ordinary | 14.12 | 50.23 | 32.18 | 12 | 22,527 | 15,315 | 16 |
| LivenessNet | re-estimated | 9.02 | 47.54 | 28.28 | 12 | 22,527 | 15,315 | 16 |
| AttackNetV1 | ordinary | 27.00 | 39.05 | 33.02 | 12 | 22,527 | 15,315 | 16 |
| AttackNetV1 | re-estimated | 36.82 | 33.92 | 35.37 | 12 | 22,527 | 15,315 | 16 |
| AttackNetV2_1 | ordinary | 33.70 | 39.04 | 36.37 | 12 | 22,527 | 15,315 | 16 |
| AttackNetV2_1 | re-estimated | 41.44 | 38.59 | 40.02 | 12 | 22,527 | 15,315 | 16 |
| AttackNetV2_2 | ordinary | 23.42 | 39.52 | 31.47 | 12 | 22,527 | 15,315 | 16 |
| AttackNetV2_2 | re-estimated | 30.99 | 36.39 | 33.69 | 12 | 22,527 | 15,315 | 16 |
| ResNet-18 | ordinary | 24.26 | 11.26 | 17.76 | 12 | 22,527 | 15,315 | 16 |
| ResNet-18 | re-estimated | 24.21 | 13.36 | 18.79 | 12 | 22,527 | 15,315 | 16 |

ACER variations (re-estimated − ordinary BN) and contrasts (AttackNet − LivenessNet), mean [subject interval]:

| variation of | APCER (points) | BPCER (points) | ACER (points) | ACER units above / below / at zero |
|---|---|---|---|---|
| LivenessNet | −5.10 [−9.28, −0.09] | −2.69 [−8.79, +3.37] | −3.90 [−7.86, +0.32] | 6 / 6 / 0 |
| AttackNetV1 | +9.83 [+5.64, +14.50] | −5.12 [−12.25, +0.82] | +2.35 [−1.13, +5.77] | 7 / 5 / 0 |
| AttackNetV2_1 | +7.74 [+4.02, +13.29] | −0.45 [−5.48, +5.95] | +3.64 [+0.62, +7.43] | 7 / 5 / 0 |
| AttackNetV2_2 | +7.57 [+4.82, +10.53] | −3.12 [−9.75, +6.29] | +2.22 [−1.43, +7.81] | 7 / 5 / 0 |
| ResNet-18 | −0.05 [−4.69, +6.65] | +2.10 [−5.46, +8.33] | +1.03 [−4.09, +6.61] | 7 / 5 / 0 |

| contrast of | APCER (points) | BPCER (points) | ACER (points) | ACER units above / below / at zero |
|---|---|---|---|---|
| AttackNetV1 | +14.93 [+10.50, +19.72] | −2.43 [−13.28, +6.27] | +6.25 [+0.03, +11.54] | 7 / 5 / 0 |
| AttackNetV2_1 | +12.84 [+5.85, +19.52] | +2.25 [−7.53, +12.31] | +7.54 [+1.14, +13.49] | 8 / 4 / 0 |
| AttackNetV2_2 | +12.67 [+7.67, +17.72] | −0.43 [−11.16, +12.09] | +6.12 [+0.54, +13.32] | 6 / 6 / 0 |

## Patience 12, dev EER threshold (descriptive)

APCER, BPCER and ACER of each network, mean over the 12 units:

| network | BN state | APCER (%) | BPCER (%) | ACER (%) | units | attack frames | bona fide frames | subjects |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LivenessNet | ordinary | 11.89 | 41.98 | 26.94 | 12 | 22,527 | 15,315 | 16 |
| LivenessNet | re-estimated | 7.01 | 53.24 | 30.12 | 12 | 22,527 | 15,315 | 16 |
| AttackNetV1 | ordinary | 25.89 | 34.71 | 30.30 | 12 | 22,527 | 15,315 | 16 |
| AttackNetV1 | re-estimated | 35.24 | 30.83 | 33.03 | 12 | 22,527 | 15,315 | 16 |
| AttackNetV2_1 | ordinary | 32.85 | 39.22 | 36.04 | 12 | 22,527 | 15,315 | 16 |
| AttackNetV2_1 | re-estimated | 40.92 | 36.86 | 38.89 | 12 | 22,527 | 15,315 | 16 |
| AttackNetV2_2 | ordinary | 26.44 | 31.67 | 29.06 | 12 | 22,527 | 15,315 | 16 |
| AttackNetV2_2 | re-estimated | 27.58 | 37.10 | 32.34 | 12 | 22,527 | 15,315 | 16 |
| ResNet-18 | ordinary | 23.44 | 10.98 | 17.21 | 12 | 22,527 | 15,315 | 16 |
| ResNet-18 | re-estimated | 26.24 | 15.71 | 20.98 | 12 | 22,527 | 15,315 | 16 |

ACER variations (re-estimated − ordinary BN) and contrasts (AttackNet − LivenessNet), mean [subject interval]:

| variation of | APCER (points) | BPCER (points) | ACER (points) | ACER units above / below / at zero |
|---|---|---|---|---|
| LivenessNet | −4.89 [−7.22, −2.19] | +11.26 [+3.90, +15.15] | +3.19 [−0.33, +4.95] | 7 / 5 / 0 |
| AttackNetV1 | +9.35 [+5.32, +13.82] | −3.89 [−10.03, +1.65] | +2.73 [−0.38, +5.91] | 7 / 5 / 0 |
| AttackNetV2_1 | +8.07 [+4.33, +13.50] | −2.36 [−6.00, +0.61] | +2.85 [+0.49, +5.22] | 7 / 5 / 0 |
| AttackNetV2_2 | +1.14 [−0.45, +2.74] | +5.42 [−0.79, +14.33] | +3.28 [+0.02, +7.87] | 7 / 5 / 0 |
| ResNet-18 | +2.80 [+0.73, +6.05] | +4.73 [−0.83, +8.23] | +3.76 [+0.42, +6.56] | 9 / 3 / 0 |

| contrast of | APCER (points) | BPCER (points) | ACER (points) | ACER units above / below / at zero |
|---|---|---|---|---|
| AttackNetV1 | +14.23 [+9.66, +19.17] | −15.14 [−21.14, −5.04] | −0.46 [−2.80, +3.96] | 7 / 5 / 0 |
| AttackNetV2_1 | +12.96 [+7.53, +20.55] | −13.62 [−18.56, −6.14] | −0.33 [−3.37, +4.37] | 5 / 7 / 0 |
| AttackNetV2_2 | +6.03 [+3.04, +8.94] | −5.83 [−13.70, +6.15] | +0.10 [−4.08, +6.39] | 6 / 6 / 0 |

## Comparison with ResNet-18 (descriptive, `PROTOCOL.md` §6 (m))

For each of LivenessNet and the three AttackNets: the gap metric(network) − metric(ResNet-18) in each BN
state and the difference of the variations V(network) − V(ResNet-18), mean [subject interval]. For
LivenessNet these are the C2 gaps and contrast with the opposite sign (checked against the reproduction
of C2 recorded in `results.json`). No reading.

**patience 4, threshold 0.5**

| network | quantity | APCER (points) | BPCER (points) | ACER (points) | ACER units above / below / at zero |
|---|---|---|---|---|---|
| LivenessNet | gap, ordinary BN | −18.01 [−33.72, −8.71] | +50.13 [+33.30, +61.44] | +16.06 [+7.41, +20.79] | 11 / 1 / 0 |
| LivenessNet | gap, re-estimated BN | −15.56 [−32.11, −5.81] | +27.71 [+8.56, +43.33] | +6.07 [−4.76, +12.53] | 9 / 3 / 0 |
| LivenessNet | variation difference | +2.45 [−4.15, +9.12] | −22.42 [−30.88, −12.02] | −9.98 [−16.32, −2.56] | 1 / 11 / 0 |
| AttackNetV1 | gap, ordinary BN | −2.90 [−17.57, +6.11] | +27.53 [+10.47, +38.26] | +12.31 [+2.12, +17.14] | 11 / 1 / 0 |
| AttackNetV1 | gap, re-estimated BN | +14.30 [−0.40, +23.15] | +20.63 [+6.77, +26.44] | +17.47 [+8.67, +21.72] | 9 / 3 / 0 |
| AttackNetV1 | variation difference | +17.21 [+11.80, +23.22] | −6.90 [−16.15, +2.77] | +5.15 [−1.11, +11.73] | 7 / 5 / 0 |
| AttackNetV2_1 | gap, ordinary BN | −5.36 [−22.51, +5.16] | +35.04 [+17.42, +46.23] | +14.84 [+2.78, +20.49] | 11 / 1 / 0 |
| AttackNetV2_1 | gap, re-estimated BN | +15.64 [−0.34, +25.01] | +21.96 [+8.14, +31.68] | +18.80 [+10.09, +23.46] | 10 / 2 / 0 |
| AttackNetV2_1 | variation difference | +21.00 [+14.97, +28.00] | −13.07 [−22.59, −1.20] | +3.96 [−0.03, +10.39] | 9 / 3 / 0 |
| AttackNetV2_2 | gap, ordinary BN | −10.64 [−25.30, −2.46] | +31.99 [+15.71, +43.19] | +10.67 [+1.57, +14.56] | 12 / 0 / 0 |
| AttackNetV2_2 | gap, re-estimated BN | +5.75 [−8.82, +14.15] | +18.84 [+5.09, +25.84] | +12.30 [+3.75, +15.55] | 10 / 2 / 0 |
| AttackNetV2_2 | variation difference | +16.39 [+11.97, +20.05] | −13.14 [−20.04, −3.04] | +1.62 [−2.72, +6.99] | 6 / 6 / 0 |

**patience 12, threshold 0.5**

| network | quantity | APCER (points) | BPCER (points) | ACER (points) | ACER units above / below / at zero |
|---|---|---|---|---|---|
| LivenessNet | gap, ordinary BN | −14.28 [−29.80, −6.40] | +34.90 [+15.37, +49.56] | +10.31 [−1.76, +17.01] | 10 / 2 / 0 |
| LivenessNet | gap, re-estimated BN | −16.80 [−32.38, −7.90] | +24.01 [+8.55, +39.69] | +3.60 [−6.13, +10.16] | 8 / 4 / 0 |
| LivenessNet | variation difference | −2.52 [−7.24, +2.18] | −10.89 [−15.53, −1.47] | −6.71 [−10.15, −1.27] | 1 / 11 / 0 |
| AttackNetV1 | gap, ordinary BN | −0.64 [−16.72, +8.77] | +25.01 [+8.51, +32.86] | +12.18 [+2.35, +16.14] | 11 / 1 / 0 |
| AttackNetV1 | gap, re-estimated BN | +14.88 [+0.53, +23.06] | +15.75 [+2.55, +24.89] | +15.31 [+6.42, +21.19] | 9 / 3 / 0 |
| AttackNetV1 | variation difference | +15.52 [+10.68, +23.03] | −9.26 [−17.66, +2.94] | +3.13 [−1.53, +10.18] | 7 / 5 / 0 |
| AttackNetV2_1 | gap, ordinary BN | −2.25 [−21.48, +8.74] | +37.07 [+19.69, +47.35] | +17.41 [+5.09, +22.43] | 11 / 1 / 0 |
| AttackNetV2_1 | gap, re-estimated BN | +16.07 [−1.15, +25.02] | +18.51 [+6.08, +27.57] | +17.29 [+8.38, +21.96] | 10 / 2 / 0 |
| AttackNetV2_1 | variation difference | +18.32 [+12.68, +25.74] | −18.56 [−25.53, −7.92] | −0.12 [−2.63, +4.72] | 7 / 5 / 0 |
| AttackNetV2_2 | gap, ordinary BN | −1.14 [−19.06, +9.03] | +21.72 [+7.12, +30.54] | +10.29 [−1.22, +14.68] | 11 / 1 / 0 |
| AttackNetV2_2 | gap, re-estimated BN | +5.74 [−11.62, +15.19] | +19.11 [+6.60, +28.80] | +12.43 [+3.05, +17.47] | 9 / 3 / 0 |
| AttackNetV2_2 | variation difference | +6.88 [+3.52, +11.34] | −2.61 [−11.19, +9.66] | +2.14 [−2.37, +8.36] | 5 / 7 / 0 |

**patience 4, dev EER threshold**

| network | quantity | APCER (points) | BPCER (points) | ACER (points) | ACER units above / below / at zero |
|---|---|---|---|---|---|
| LivenessNet | gap, ordinary BN | −10.14 [−23.55, −1.99] | +38.97 [+22.29, +48.98] | +14.42 [+5.60, +18.65] | 11 / 1 / 0 |
| LivenessNet | gap, re-estimated BN | −15.19 [−29.66, −7.45] | +34.18 [+14.99, +47.48] | +9.49 [−1.11, +15.96] | 9 / 3 / 0 |
| LivenessNet | variation difference | −5.05 [−12.87, +1.14] | −4.80 [−13.48, +5.70] | −4.92 [−12.16, +2.20] | 7 / 5 / 0 |
| AttackNetV1 | gap, ordinary BN | +2.73 [−10.48, +10.30] | +27.79 [+11.49, +38.83] | +15.26 [+5.92, +20.47] | 12 / 0 / 0 |
| AttackNetV1 | gap, re-estimated BN | +12.61 [−3.86, +22.97] | +20.56 [+7.54, +25.88] | +16.59 [+6.79, +22.10] | 10 / 2 / 0 |
| AttackNetV1 | variation difference | +9.88 [+2.18, +15.52] | −7.23 [−17.16, +2.96] | +1.33 [−5.69, +7.58] | 6 / 6 / 0 |
| AttackNetV2_1 | gap, ordinary BN | +9.44 [−7.35, +19.36] | +27.78 [+11.75, +38.05] | +18.61 [+6.81, +24.15] | 12 / 0 / 0 |
| AttackNetV2_1 | gap, re-estimated BN | +17.23 [−3.75, +30.23] | +25.23 [+11.20, +33.31] | +21.23 [+9.93, +26.59] | 11 / 1 / 0 |
| AttackNetV2_1 | variation difference | +7.79 [+0.13, +12.59] | −2.55 [−11.56, +8.19] | +2.62 [−1.89, +7.92] | 8 / 4 / 0 |
| AttackNetV2_2 | gap, ordinary BN | −0.84 [−15.43, +6.29] | +28.26 [+10.15, +36.97] | +13.71 [+2.65, +18.35] | 12 / 0 / 0 |
| AttackNetV2_2 | gap, re-estimated BN | +6.78 [−11.94, +16.97] | +23.03 [+8.68, +32.96] | +14.90 [+5.30, +20.13] | 11 / 1 / 0 |
| AttackNetV2_2 | variation difference | +7.62 [−0.18, +13.18] | −5.23 [−16.28, +8.87] | +1.20 [−6.10, +10.42] | 8 / 4 / 0 |

**patience 12, dev EER threshold**

| network | quantity | APCER (points) | BPCER (points) | ACER (points) | ACER units above / below / at zero |
|---|---|---|---|---|---|
| LivenessNet | gap, ordinary BN | −11.55 [−25.86, −5.31] | +31.00 [+15.08, +43.75] | +9.73 [+0.20, +14.98] | 10 / 2 / 0 |
| LivenessNet | gap, re-estimated BN | −19.23 [−32.53, −12.20] | +37.53 [+17.73, +50.96] | +9.15 [−1.25, +15.62] | 10 / 2 / 0 |
| LivenessNet | variation difference | −7.68 [−12.41, −4.07] | +6.53 [−0.22, +14.43] | −0.58 [−5.09, +3.85] | 5 / 7 / 0 |
| AttackNetV1 | gap, ordinary BN | +2.45 [−13.01, +11.05] | +23.73 [+7.78, +32.03] | +13.09 [+3.87, +17.82] | 11 / 1 / 0 |
| AttackNetV1 | gap, re-estimated BN | +9.00 [−6.92, +18.73] | +15.12 [+2.63, +23.57] | +12.06 [+2.07, +19.21] | 9 / 3 / 0 |
| AttackNetV1 | variation difference | +6.55 [+0.83, +11.92] | −8.62 [−16.42, +1.24] | −1.03 [−5.81, +4.64] | 5 / 7 / 0 |
| AttackNetV2_1 | gap, ordinary BN | +9.41 [−9.33, +20.50] | +28.24 [+12.39, +37.51] | +18.83 [+7.36, +23.97] | 12 / 0 / 0 |
| AttackNetV2_1 | gap, re-estimated BN | +14.68 [−6.15, +26.83] | +21.15 [+8.49, +28.43] | +17.92 [+6.35, +23.63] | 11 / 1 / 0 |
| AttackNetV2_1 | variation difference | +5.27 [−0.33, +9.04] | −7.09 [−12.01, −0.05] | −0.91 [−3.16, +2.41] | 6 / 6 / 0 |
| AttackNetV2_2 | gap, ordinary BN | +3.00 [−13.50, +11.60] | +20.69 [+7.58, +26.47] | +11.85 [+0.53, +16.00] | 12 / 0 / 0 |
| AttackNetV2_2 | gap, re-estimated BN | +1.34 [−17.50, +11.42] | +21.39 [+8.66, +31.92] | +11.36 [+1.24, +17.77] | 9 / 3 / 0 |
| AttackNetV2_2 | variation difference | −1.66 [−5.42, +1.09] | +0.70 [−6.61, +11.92] | −0.48 [−4.78, +6.07] | 6 / 6 / 0 |

## Leaving one unit out

Leaving one unit out (a choice of the author, descriptive, not in `PROTOCOL.md`): means over the other 11
units; no interval, since subject intervals need the per-subject counts, which are not in the public files.
Patience 4, threshold 0.5, ACER, percentage points.

| unit left out | variation of LivenessNet | variation of AttackNetV1 | variation of AttackNetV2_1 | variation of AttackNetV2_2 | variation of ResNet-18 | contrast of AttackNetV1 | contrast of AttackNetV2_1 | contrast of AttackNetV2_2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none (all 12 units) | −7.07 | +8.07 | +6.87 | +4.53 | +2.91 | +15.14 | +13.95 | +11.61 |
| fold 1, seed 42 | −4.98 | +8.47 | +7.77 | +5.59 | +4.14 | +13.44 | +12.75 | +10.57 |
| fold 1, seed 43 | −7.99 | +8.95 | +8.50 | +3.84 | +3.59 | +16.94 | +16.49 | +11.83 |
| fold 1, seed 44 | −7.32 | +9.38 | +7.16 | +4.88 | +2.99 | +16.70 | +14.49 | +12.20 |
| fold 2, seed 42 | −7.12 | +7.65 | +6.28 | +3.43 | +2.73 | +14.77 | +13.40 | +10.55 |
| fold 2, seed 43 | −7.92 | +6.84 | +5.73 | +5.25 | +2.28 | +14.77 | +13.65 | +13.17 |
| fold 2, seed 44 | −6.92 | +7.21 | +7.14 | +4.34 | +2.93 | +14.14 | +14.06 | +11.26 |
| fold 3, seed 42 | −8.05 | +8.10 | +6.34 | +4.32 | +2.42 | +16.14 | +14.38 | +12.36 |
| fold 3, seed 43 | −7.74 | +7.48 | +4.79 | +1.93 | +3.04 | +15.22 | +12.53 | +9.67 |
| fold 3, seed 44 | −2.83 | +5.67 | +6.76 | +3.12 | +3.14 | +8.50 | +9.59 | +5.95 |
| fold 4, seed 42 | −7.98 | +9.50 | +5.82 | +6.04 | +2.45 | +17.48 | +13.80 | +14.02 |
| fold 4, seed 43 | −8.08 | +9.08 | +8.29 | +5.29 | +2.67 | +17.16 | +16.37 | +13.37 |
| fold 4, seed 44 | −7.93 | +8.45 | +7.94 | +6.39 | +2.53 | +16.38 | +15.87 | +14.32 |

## Rankings (descriptive, `PROTOCOL.md` §6 (g))

Not a result: no interval. Per condition the networks present in every unit, ordered by mean ACER (exact
fractions; ties share the lowest rank), and, per network, how many of the 12 units give it each rank
(same rule on the unit ACER). Recomputed from `units.csv` and checked against `results.json`.

| condition | BN state | order by mean ACER (%) | units at rank 1 / 2 / 3 / 4 / 5 |
|---|---|---|---|
| patience 4, threshold 0.5 | ordinary | 1. ResNet-18 21.29, 2. AttackNetV2_2 31.96, 3. AttackNetV1 33.60, 4. AttackNetV2_1 36.13, 5. LivenessNet 37.35. | LivenessNet 1 / 1 / 3 / 4 / 3; AttackNetV1 1 / 3 / 3 / 3 / 2; AttackNetV2_1 0 / 2 / 3 / 3 / 4; AttackNetV2_2 0 / 5 / 2 / 2 / 3; ResNet-18 10 / 1 / 1 / 0 / 0 |
| patience 4, threshold 0.5 | re-estimated | 1. ResNet-18 24.20, 2. LivenessNet 30.28, 3. AttackNetV2_2 36.50, 4. AttackNetV1 41.67, 5. AttackNetV2_1 43.00. | LivenessNet 2 / 3 / 3 / 2 / 2; AttackNetV1 1 / 2 / 3 / 2 / 4; AttackNetV2_1 1 / 1 / 2 / 3 / 5; AttackNetV2_2 2 / 2 / 3 / 5 / 0; ResNet-18 6 / 4 / 1 / 0 / 1 |
| patience 4, dev EER threshold | ordinary | 1. ResNet-18 17.76, 2. AttackNetV2_2 31.47, 3. LivenessNet 32.18, 4. AttackNetV1 33.02, 5. AttackNetV2_1 36.37. | LivenessNet 1 / 4 / 3 / 2 / 2; AttackNetV1 0 / 3 / 3 / 4 / 2; AttackNetV2_1 0 / 0 / 4 / 3 / 5; AttackNetV2_2 0 / 4 / 2 / 3 / 3; ResNet-18 11 / 1 / 0 / 0 / 0 |
| patience 4, dev EER threshold | re-estimated | 1. ResNet-18 18.79, 2. LivenessNet 28.28, 3. AttackNetV2_2 33.69, 4. AttackNetV1 35.37, 5. AttackNetV2_1 40.02. | LivenessNet 3 / 4 / 3 / 2 / 0; AttackNetV1 1 / 2 / 3 / 2 / 4; AttackNetV2_1 0 / 1 / 2 / 3 / 6; AttackNetV2_2 0 / 3 / 3 / 4 / 2; ResNet-18 8 / 2 / 1 / 1 / 0 |
| patience 12, threshold 0.5 | ordinary | 1. ResNet-18 18.37, 2. AttackNetV2_2 28.66, 3. LivenessNet 28.68, 4. AttackNetV1 30.55, 5. AttackNetV2_1 35.78. | LivenessNet 1 / 3 / 2 / 4 / 2; AttackNetV1 1 / 2 / 5 / 2 / 2; AttackNetV2_1 0 / 2 / 1 / 4 / 5; AttackNetV2_2 0 / 4 / 4 / 2 / 2; ResNet-18 10 / 1 / 0 / 0 / 1 |
| patience 12, threshold 0.5 | re-estimated | 1. ResNet-18 23.92, 2. LivenessNet 27.53, 3. AttackNetV2_2 36.35, 4. AttackNetV1 39.24, 5. AttackNetV2_1 41.21. | LivenessNet 3 / 5 / 2 / 1 / 1; AttackNetV1 2 / 1 / 3 / 1 / 5; AttackNetV2_1 0 / 1 / 3 / 4 / 4; AttackNetV2_2 1 / 1 / 4 / 6 / 0; ResNet-18 6 / 4 / 0 / 0 / 2 |
| patience 12, dev EER threshold | ordinary | 1. ResNet-18 17.21, 2. LivenessNet 26.94, 3. AttackNetV2_2 29.06, 4. AttackNetV1 30.30, 5. AttackNetV2_1 36.04. | LivenessNet 2 / 4 / 3 / 1 / 2; AttackNetV1 1 / 3 / 3 / 2 / 3; AttackNetV2_1 0 / 0 / 1 / 6 / 5; AttackNetV2_2 0 / 2 / 5 / 3 / 2; ResNet-18 9 / 3 / 0 / 0 / 0 |
| patience 12, dev EER threshold | re-estimated | 1. ResNet-18 20.98, 2. LivenessNet 30.12, 3. AttackNetV2_2 32.34, 4. AttackNetV1 33.03, 5. AttackNetV2_1 38.89. | LivenessNet 1 / 3 / 5 / 3 / 0; AttackNetV1 2 / 2 / 3 / 1 / 4; AttackNetV2_1 0 / 1 / 2 / 4 / 5; AttackNetV2_2 1 / 4 / 2 / 3 / 2; ResNet-18 8 / 2 / 0 / 1 / 1 |
