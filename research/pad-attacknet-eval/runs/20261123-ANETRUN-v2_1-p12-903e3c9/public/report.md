# NUAA run anet_v2_1_p12 (20261123-ANETRUN-v2_1-p12-903e3c9): AttackNetV2_1, patience 12

Status: **PASS**. Attempt 1. 12 positions of `run_matrix.csv`; phase 1 (training, rule (n), checkpoint, dev scored with the scorer of the accepted runs) complete for 12, phase 2 (both BN states on the dev, BN re-estimated on the fitting set) for 12; not evaluable under rule (n): 0. No test frame is assembled or evaluated, no test score or metric is read: the frame files, hashed and loaded whole, also hold frames of the held-out subjects, of which only the fitting and dev frames of arm B are assembled; every number below is a dev or training number.

Started 2026-11-23T01:54:17+0100, finished 2026-11-23T03:14:08+0100; wall time 4791 s of a cap of 2.50 hours of wall time (elapsed real time, not CPU time); process CPU time (all threads) 38718 s.

Machine: Apple M5 Pro, cores as named by the system: 5 Super, 10 Performance; 15 logical CPUs, 24.0 GB RAM, macOS 27.2.1; equal to the official trial 20261122-ANETTRIAL-a951ed0 in processor, core_levels, ram_gb, os, logical_cpus, architecture and in the versions python, numpy, tensorflow_installed, keras, platform (machine model and versions, not the physical unit).

TensorFlow threads set by this runner: requested intra-op 10 and inter-op 10, effective (read back from tf.config.threading) intra-op 10 and inter-op 10; environment variables: TF_NUM_INTRAOP_THREADS absent, TF_NUM_INTEROP_THREADS absent; logical CPUs 15.

BN estimator: checked by 20261122-ANETEST-c882362 (36 models of the BNRECAL run reproduced).

## Models

| position | fold | seed | status | epochs | restored epoch | non-finite history | ordinary: EER threshold | ordinary: one-class at 0.5 | re-estimated: EER threshold | re-estimated: one-class at 0.5 | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 42 | complete | 12 | 5 | none | 0.165987 | no | 0.698656 | no | 327.6 |
| 2 | 1 | 43 | complete | 12 | 4 | none | 0.999571 | no | 0.991717 | no | 330.0 |
| 3 | 1 | 44 | complete | 12 | 9 | none | 0.744874 | no | 0.944474 | no | 332.0 |
| 4 | 2 | 42 | complete | 12 | 11 | none | 0.614446 | no | 0.124407 | no | 334.0 |
| 5 | 2 | 43 | complete | 12 | 10 | none | 0.383446 | no | 0.034515 | no | 333.9 |
| 6 | 2 | 44 | complete | 12 | 9 | none | 0.917765 | no | 0.030055 | no | 335.3 |
| 7 | 3 | 42 | complete | 12 | 2 | none | 0.702218 | no | 0.091196 | no | 339.7 |
| 8 | 3 | 43 | complete | 12 | 5 | none | 0.045766 | no | 0.002946 | no | 337.2 |
| 9 | 3 | 44 | complete | 12 | 6 | none | 0.239744 | no | 0.004179 | no | 337.0 |
| 10 | 4 | 42 | complete | 12 | 3 | none | 0.860965 | no | 0.997444 | no | 336.2 |
| 11 | 4 | 43 | complete | 12 | 7 | none | 0.398819 | no | 0.144483 | no | 337.3 |
| 12 | 4 | 44 | complete | 12 | 11 | none | 0.876810 | no | 0.990037 | no | 382.5 |

Rule (n) of the pre-registration (PROTOCOL.md §6): a non-finite value in the history does not stop the run; without a finite val_loss, with non-finite restored weights, non-finite scores or logits, or non-finite re-estimated statistics the model is not evaluable and the run goes on.

## One-class runs at threshold 0.5 on the dev (definition (a))

| network | patience | BN state | count |
|---|---|---|---|
| AttackNetV2_1 | 12 | ordinary | 0 of 12 evaluable (all bona fide 0, all attack 0, not evaluable 0) |
| AttackNetV2_1 | 12 | re-estimated | 0 of 12 evaluable (all bona fide 0, all attack 0, not evaluable 0) |

Dev frame identifiers (public outputs hold only counts and SHA-256): 12 models compared over 4 folds, both BN states; every model of a fold used the same frames in the same order.

Command: `python -I -B scripts/pad_anet_run.py run --config delivery/anetrun/pad_anet_run.json --out-root results/anetrun --run anet_v2_1_p12 --estimator-check results/anetrun/20261122-ANETEST-c882362 --attempt 1 --i-am-authorized`
