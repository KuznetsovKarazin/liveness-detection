Test APCER and BPCER (%) of sessions 1–2 and of session 3 at the development thresholds of pull request #36 (development APCER target 5% or 10%), and the paired difference session 3 − sessions 1–2. Each cell is the mean of twelve fold–seed units with equal weight; the thresholds are transferred from the development scores of each checkpoint and BN state, as in Table A3, and are the same for the two groups. Units up/zero/down: fold–seed units with a positive, zero or negative difference, decided on integer counts. Every value is rounded separately to two decimals from its exact rational value (half to even), so the difference shown can differ by 0.01 from the difference of the two values shown; full values are in session_summary.csv. Rows in bold: highlighted outcome (patience 4, target 5%; author's choice).

| Patience | Detector | Target (%) | BN state | APCER (%) sessions 1–2 | APCER (%) session 3 | APCER (%) difference | APCER units up/zero/down | BPCER (%) sessions 1–2 | BPCER (%) session 3 | BPCER (%) difference | BPCER units up/zero/down |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **4** | **LivenessNet** | **5** | **stored** | **6.13** | **42.15** | **+36.02** | **11/0/1** | **11.68** | **45.74** | **+34.05** | **11/0/1** |
| **4** | **LivenessNet** | **5** | **re-estimated** | **9.13** | **63.02** | **+53.88** | **12/0/0** | **0.03** | **23.86** | **+23.83** | **12/0/0** |
| 4 | LivenessNet | 10 | stored | 8.62 | 49.15 | +40.53 | 11/0/1 | 11.64 | 41.61 | +29.97 | 11/0/1 |
| 4 | LivenessNet | 10 | re-estimated | 10.98 | 67.29 | +56.31 | 12/0/0 | 0.03 | 21.84 | +21.81 | 11/1/0 |
| **4** | **ResNet-18** | **5** | **stored** | **10.25** | **79.72** | **+69.46** | **12/0/0** | **0.07** | **2.95** | **+2.87** | **2/8/2** |
| **4** | **ResNet-18** | **5** | **re-estimated** | **8.40** | **77.35** | **+68.95** | **12/0/0** | **0.05** | **0.42** | **+0.37** | **2/8/2** |
| 4 | ResNet-18 | 10 | stored | 12.49 | 84.48 | +71.99 | 12/0/0 | 0.02 | 2.91 | +2.88 | 2/9/1 |
| 4 | ResNet-18 | 10 | re-estimated | 9.99 | 81.87 | +71.88 | 12/0/0 | 0.02 | 0.00 | -0.02 | 0/11/1 |
| 12 | LivenessNet | 5 | stored | 10.60 | 60.44 | +49.83 | 12/0/0 | 0.12 | 19.67 | +19.55 | 9/3/0 |
| 12 | LivenessNet | 5 | re-estimated | 9.64 | 60.02 | +50.38 | 12/0/0 | 0.03 | 22.96 | +22.93 | 12/0/0 |
| 12 | LivenessNet | 10 | stored | 12.33 | 65.72 | +53.39 | 12/0/0 | 0.10 | 17.71 | +17.61 | 9/3/0 |
| 12 | LivenessNet | 10 | re-estimated | 11.41 | 64.22 | +52.82 | 12/0/0 | 0.03 | 21.66 | +21.63 | 12/0/0 |
| 12 | ResNet-18 | 5 | stored | 10.28 | 77.75 | +67.48 | 12/0/0 | 0.07 | 2.95 | +2.87 | 2/8/2 |
| 12 | ResNet-18 | 5 | re-estimated | 8.99 | 77.97 | +68.98 | 12/0/0 | 0.05 | 2.14 | +2.09 | 3/7/2 |
| 12 | ResNet-18 | 10 | stored | 12.85 | 83.91 | +71.06 | 12/0/0 | 0.02 | 2.91 | +2.88 | 2/9/1 |
| 12 | ResNet-18 | 10 | re-estimated | 11.39 | 82.55 | +71.16 | 12/0/0 | 0.02 | 0.14 | +0.11 | 1/10/1 |

Test frames per fold, sessions 1–2 and session 3: fold 1: photograph 587 and 1,232, bona fide 498 and 925; fold 2: photograph 380 and 1,364, bona fide 351 and 1,019; fold 3: photograph 387 and 1,665, bona fide 522 and 865; fold 4: photograph 394 and 1,500, bona fide 372 and 553. The two groups contain different sets of manifest identifiers (PROTOCOL.md, limits).
