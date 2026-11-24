# NUAA run anet_v1_p4 (20261122-ANETRUN-v1-p4-b8e0b3c): AttackNetV1, patience 4

Status: **PASS**. Attempt 1. 12 positions of `run_matrix.csv`; phase 1 (training, rule (n), checkpoint, dev scored with the scorer of the accepted runs) complete for 12, phase 2 (both BN states on the dev, BN re-estimated on the fitting set) for 12; not evaluable under rule (n): 0. No test frame is assembled or evaluated, no test score or metric is read: the frame files, hashed and loaded whole, also hold frames of the held-out subjects, of which only the fitting and dev frames of arm B are assembled; every number below is a dev or training number.

Started 2026-11-22T21:01:18+0100, finished 2026-11-22T22:13:23+0100; wall time 4325 s of a cap of 2.50 hours of wall time (elapsed real time, not CPU time); process CPU time (all threads) 35043 s.

Machine: Apple M5 Pro, cores as named by the system: 5 Super, 10 Performance; 15 logical CPUs, 24.0 GB RAM, macOS 27.2.1; equal to the official trial 20261122-ANETTRIAL-a951ed0 in processor, core_levels, ram_gb, os, logical_cpus, architecture and in the versions python, numpy, tensorflow_installed, keras, platform (machine model and versions, not the physical unit).

TensorFlow threads set by this runner: requested intra-op 10 and inter-op 10, effective (read back from tf.config.threading) intra-op 10 and inter-op 10; environment variables: TF_NUM_INTRAOP_THREADS absent, TF_NUM_INTEROP_THREADS absent; logical CPUs 15.

BN estimator: checked by 20261122-ANETEST-c882362 (36 models of the BNRECAL run reproduced).

## Models

| position | fold | seed | status | epochs | restored epoch | non-finite history | ordinary: EER threshold | ordinary: one-class at 0.5 | re-estimated: EER threshold | re-estimated: one-class at 0.5 | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 42 | complete | 11 | 7 | none | 0.580804 | no | 0.723585 | no | 303.9 |
| 2 | 1 | 43 | complete | 12 | 11 | none | 0.020701 | no | 0.960184 | no | 332.7 |
| 3 | 1 | 44 | complete | 12 | 10 | none | 0.666175 | no | 0.896421 | no | 335.9 |
| 4 | 2 | 42 | complete | 12 | 9 | none | 0.828177 | no | 0.044381 | no | 334.1 |
| 5 | 2 | 43 | complete | 11 | 7 | none | 0.769083 | no | 0.055477 | no | 302.4 |
| 6 | 2 | 44 | complete | 10 | 6 | none | 0.116050 | no | 0.091112 | no | 275.4 |
| 7 | 3 | 42 | complete | 11 | 7 | none | 0.376175 | no | 0.022495 | no | 306.4 |
| 8 | 3 | 43 | complete | 12 | 11 | none | 0.037172 | no | 0.008309 | no | 327.5 |
| 9 | 3 | 44 | complete | 10 | 6 | none | 0.921647 | no | 0.007954 | no | 270.9 |
| 10 | 4 | 42 | complete | 12 | 11 | none | 0.063471 | no | 0.132951 | no | 327.8 |
| 11 | 4 | 43 | complete | 12 | 12 | none | 0.362851 | no | 0.024059 | no | 327.8 |
| 12 | 4 | 44 | complete | 7 | 3 | none | 0.283929 | no | 0.534116 | no | 191.9 |

Rule (n) of the pre-registration (PROTOCOL.md §6): a non-finite value in the history does not stop the run; without a finite val_loss, with non-finite restored weights, non-finite scores or logits, or non-finite re-estimated statistics the model is not evaluable and the run goes on.

## One-class runs at threshold 0.5 on the dev (definition (a))

| network | patience | BN state | count |
|---|---|---|---|
| AttackNetV1 | 4 | ordinary | 0 of 12 evaluable (all bona fide 0, all attack 0, not evaluable 0) |
| AttackNetV1 | 4 | re-estimated | 0 of 12 evaluable (all bona fide 0, all attack 0, not evaluable 0) |

Dev frame identifiers (public outputs hold only counts and SHA-256): 12 models compared over 4 folds, both BN states; every model of a fold used the same frames in the same order.

Command: `python -I -B scripts/pad_anet_run.py run --config delivery/anetrun/pad_anet_run.json --out-root results/anetrun --run anet_v1_p4 --estimator-check results/anetrun/20261122-ANETEST-c882362 --attempt 1 --i-am-authorized`
