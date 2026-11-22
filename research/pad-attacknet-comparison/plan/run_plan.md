# Planned runs: AttackNetV1, AttackNetV2_1 and AttackNetV2_2 on NUAA

Generated from the configuration 6f5297d08d49 and the trial 20261122-ANETTRIAL-a951ed0 (trial.json 8f6724517ae8).

72 new trainings (3 AttackNets x 4 folds x 3 seeds x 2 patiences, 36 per patience), arm B, at most 12 epochs, in six runs of 12 (`run_matrix.csv`, `run_order.csv`); 48 reused C2 cells (LivenessNet and ResNet-18, `reference_cells.csv`), not trained and not re-evaluated.

Hours are hours of wall time on the machine of the trial; CPU hours use the CPU / wall ratio measured in the trial. Training of a network at a patience = 12 x (epochs x seconds per epoch + seconds outside the epochs + seconds of phase 2), epochs from min(patience + 1, 12) to 12; seconds per epoch: mean of the measured epochs, the first included (graph tracing: a cautious estimate), validation included; phase 2: saving, dev scorer, reloading, ordinary dev, re-estimation, consistency pass, re-estimated dev, as measured in the trial. Evaluation (done once, by the C5 evaluation, not by the runs): (test scores + dev reproductions) x inference seconds per 1,000 frames; test scores = 3 seeds x test frames of the 4 folds x 2 BN states; dev reproductions = 12 x dev frames x 2 BN states. Outside the estimate: loading the inputs (once per process), assembling the test, loading each checkpoint and installing its statistics in the evaluation. Wall time is elapsed real time (the "elapsed time" of C2).

| network | patience | units | new_trainings | epochs_min | epochs_max | fitting_frames_per_unit | dev_frames_per_unit | seconds_per_epoch | seconds_outside_epochs | seconds_phase_2 | training_wall_hours_min | training_wall_hours_max | test_scores | dev_reproductions | evaluation_wall_hours | cpu_over_wall | training_cpu_hours_min | training_cpu_hours_max | evaluation_cpu_hours |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AttackNetV1 | 4 | 12 | 12 | 5 | 12 | 1400 | 400 | 26.78 | 0.98 | 45.07 | 0.60 | 1.22 | 75684 | 9600 | 0.10 | 7.65 | 4.59 | 9.37 | 0.78 |
| AttackNetV1 | 12 | 12 | 12 | 12 | 12 | 1400 | 400 | 26.78 | 0.98 | 45.07 | 1.22 | 1.22 | 75684 | 9600 | 0.10 | 7.65 | 9.37 | 9.37 | 0.78 |
| AttackNetV2_1 | 4 | 12 | 12 | 5 | 12 | 1400 | 400 | 27.11 | 1.18 | 45.45 | 0.61 | 1.24 | 75684 | 9600 | 0.10 | 7.65 | 4.64 | 9.48 | 0.78 |
| AttackNetV2_1 | 12 | 12 | 12 | 12 | 12 | 1400 | 400 | 27.11 | 1.18 | 45.45 | 1.24 | 1.24 | 75684 | 9600 | 0.10 | 7.65 | 9.48 | 9.48 | 0.78 |
| AttackNetV2_2 | 4 | 12 | 12 | 5 | 12 | 1400 | 400 | 22.36 | 1.27 | 38.41 | 0.50 | 1.03 | 75684 | 9600 | 0.09 | 7.63 | 3.85 | 7.83 | 0.69 |
| AttackNetV2_2 | 12 | 12 | 12 | 12 | 12 | 1400 | 400 | 22.36 | 1.27 | 38.41 | 1.03 | 1.03 | 75684 | 9600 | 0.09 | 7.63 | 7.83 | 7.83 | 0.69 |

## Totals

- Training, 72 trainings with phase 2: 5.20-6.98 hours of wall time; 39.77-53.37 CPU hours.
- Evaluation (once): 454104 test scores and 57600 dev reproductions, 0.59 hours of wall time; 4.50 CPU hours.

## Caps of the six runs (`run_caps.csv`)

Rule: a choice of the author: cap of a run = (largest estimate + (anomaly factor - 1) x the largest cost of one training) x multiplier, rounded up to a multiple of the rounding step; the anomaly factor covers one training at about three times its cost, as the first model of a C2 run (49.0 s per epoch against 14.4-16.3, 3.0-3.4 times); unit: hours of wall time (elapsed real time, not CPU time); anomaly factor 3, multiplier 1.5, rounded up to a multiple of 0.5 hours.

| run | network | patience | trainings | wall_hours_min | wall_hours_max | cap_wall_hours | cpu_hours_min | cpu_hours_max |
|---|---|---|---|---|---|---|---|---|
| anet_v1_p4 | AttackNetV1 | 4 | 12 | 0.60 | 1.22 | 2.50 | 4.59 | 9.37 |
| anet_v2_1_p4 | AttackNetV2_1 | 4 | 12 | 0.61 | 1.24 | 2.50 | 4.64 | 9.48 |
| anet_v2_2_p4 | AttackNetV2_2 | 4 | 12 | 0.50 | 1.03 | 2.00 | 3.85 | 7.83 |
| anet_v1_p12 | AttackNetV1 | 12 | 12 | 1.22 | 1.22 | 2.50 | 9.37 | 9.37 |
| anet_v2_1_p12 | AttackNetV2_1 | 12 | 12 | 1.24 | 1.24 | 2.50 | 9.48 | 9.48 |
| anet_v2_2_p12 | AttackNetV2_2 | 12 | 12 | 1.03 | 1.03 | 2.00 | 7.83 | 7.83 |

Sum of the six caps: 14.00 hours of wall time.

Order: a choice of the author: six runs of 12 trainings, one process each, patience 4 then 12, in each AttackNetV1, AttackNetV2_1, AttackNetV2_2; in each run folds 1-4 and seeds 42, 43, 44; the trainer clears the session and resets the seeds before every model, a failure costs 12 trainings and not 36.

Machine rule: every training and evaluation of the plan runs on the same machine with the same threads and versions as the official C5 trial, as in C2.

Reused cells: the 48 C2 checkpoints of LivenessNet and ResNet-18 (runs 20261120-ARCHRUN-p4-7faa49f and 20261120-ARCHRUN-p12-ea7fdca) are not trained and not re-evaluated: their per-frame test scores already computed by the C2 evaluation are read, each file checked against the restricted SHA256SUMS whose SHA-256 is in the public status.json of that evaluation.

Evaluation: the NUAA test is read once, by one C5 evaluation command published before it is run; the test frames of each fold are assembled once and scored by the evaluable new models of that fold in both BN states.

Machine of the trial 20261122-ANETTRIAL-a951ed0: Apple M5 Pro, 15 logical CPUs, 24.0 GB RAM, macOS 27.2.1.
