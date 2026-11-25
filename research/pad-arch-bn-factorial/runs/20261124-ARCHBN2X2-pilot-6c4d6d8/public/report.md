# 2x2 BN re-estimation on fitting/dev (20261124-ARCHBN2X2-pilot-6c4d6d8)

Status: **PASS**. Scope pilot: 2 units, 2 complete. No test frame is assembled or evaluated, no test score or metric is read; every number is a fitting or development number (the development frames were also the early-stopping set of the checkpoints: in-sample).

Started 2026-11-24T22:30:12+0100, finished 2026-11-25T00:27:47+0100; wall time 7055 s of a cap of 200 minutes; process CPU time (all threads) 45024 s; peak resident memory 5.43 GB.

## Controls

| run | network | fold | seed | control | pass |
|---|---|---|---|---|---|
| nuaa_patience_4 | LivenessNet | 1 | 42 | structure | 1 |
| nuaa_patience_4 | LivenessNet | 1 | 42 | baseline | 1 |
| nuaa_patience_4 | LivenessNet | 1 | 42 | E00 | 1 |
| nuaa_patience_4 | LivenessNet | 1 | 42 | E00_chain | 1 |
| nuaa_patience_4 | LivenessNet | 1 | 42 | duplicates_E01 | 1 |
| nuaa_patience_4 | LivenessNet | 1 | 42 | duplicates_E11 | 1 |
| nuaa_patience_4 | LivenessNet | 1 | 42 | determinism | 1 |
| nuaa_patience_4 | LivenessNet | 1 | 42 | weights | 1 |
| nuaa_patience_4 | LivenessNet | 1 | 42 | masks_E01_all | 1 |
| nuaa_patience_4 | LivenessNet | 1 | 42 | masks_E11_all | 1 |
| nuaa_patience_4 | LivenessNet | 1 | 42 | masks_E01_101-104 | 1 |
| nuaa_patience_4 | LivenessNet | 1 | 42 | masks_E01_105-108 | 1 |
| nuaa_patience_4 | LivenessNet | 1 | 42 | masks_E11_101-104 | 1 |
| nuaa_patience_4 | LivenessNet | 1 | 42 | masks_E11_105-108 | 1 |
| nuaa_patience_4 | ResNet-18 | 1 | 42 | structure | 1 |
| nuaa_patience_4 | ResNet-18 | 1 | 42 | baseline | 1 |
| nuaa_patience_4 | ResNet-18 | 1 | 42 | E00 | 1 |
| nuaa_patience_4 | ResNet-18 | 1 | 42 | duplicates_E01 | 1 |
| nuaa_patience_4 | ResNet-18 | 1 | 42 | duplicates_E11 | 1 |
| nuaa_patience_4 | ResNet-18 | 1 | 42 | weights | 1 |

## Time per unit

| unit | wall s | CPU s |
|---|---|---|
| p4_LivenessNet_fold1_seed42 | 1822.9 | 7965.3 |
| p4_ResNet-18_fold1_seed42 | 5195.4 | 36987.6 |

Cell metrics: public (cells.csv, effects.csv, montecarlo.csv).
