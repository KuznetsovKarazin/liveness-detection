# NUAA run anet_v1_p12 (20261123-ANETRUN-v1-p12-1ef418b): AttackNetV1, patience 12

Status: **PASS**. Attempt 1. 12 positions of `run_matrix.csv`; phase 1 (training, rule (n), checkpoint, dev scored with the scorer of the accepted runs) complete for 12, phase 2 (both BN states on the dev, BN re-estimated on the fitting set) for 12; not evaluable under rule (n): 0. No test frame is assembled or evaluated, no test score or metric is read: the frame files, hashed and loaded whole, also hold frames of the held-out subjects, of which only the fitting and dev frames of arm B are assembled; every number below is a dev or training number.

Started 2026-11-23T00:31:20+0100, finished 2026-11-23T01:49:01+0100; wall time 4661 s of a cap of 2.50 hours of wall time (elapsed real time, not CPU time); process CPU time (all threads) 37845 s.

Machine: Apple M5 Pro, cores as named by the system: 5 Super, 10 Performance; 15 logical CPUs, 24.0 GB RAM, macOS 27.2.1; equal to the official trial 20261122-ANETTRIAL-a951ed0 in processor, core_levels, ram_gb, os, logical_cpus, architecture and in the versions python, numpy, tensorflow_installed, keras, platform (machine model and versions, not the physical unit).

TensorFlow threads set by this runner: requested intra-op 10 and inter-op 10, effective (read back from tf.config.threading) intra-op 10 and inter-op 10; environment variables: TF_NUM_INTRAOP_THREADS absent, TF_NUM_INTEROP_THREADS absent; logical CPUs 15.

BN estimator: checked by 20261122-ANETEST-c882362 (36 models of the BNRECAL run reproduced).

## Models

| position | fold | seed | status | epochs | restored epoch | non-finite history | ordinary: EER threshold | ordinary: one-class at 0.5 | re-estimated: EER threshold | re-estimated: one-class at 0.5 | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 42 | complete | 12 | 7 | none | 0.580804 | no | 0.723585 | no | 327.3 |
| 2 | 1 | 43 | complete | 12 | 11 | none | 0.020701 | no | 0.960184 | no | 328.3 |
| 3 | 1 | 44 | complete | 12 | 10 | none | 0.666175 | no | 0.896421 | no | 338.4 |
| 4 | 2 | 42 | complete | 12 | 9 | none | 0.828177 | no | 0.044381 | no | 331.3 |
| 5 | 2 | 43 | complete | 12 | 7 | none | 0.769083 | no | 0.055477 | no | 329.8 |
| 6 | 2 | 44 | complete | 12 | 6 | none | 0.116050 | no | 0.091112 | no | 330.1 |
| 7 | 3 | 42 | complete | 12 | 12 | none | 0.083535 | no | 0.003320 | no | 338.2 |
| 8 | 3 | 43 | complete | 12 | 11 | none | 0.037172 | no | 0.008309 | no | 337.3 |
| 9 | 3 | 44 | complete | 12 | 6 | none | 0.921647 | no | 0.007954 | no | 330.7 |
| 10 | 4 | 42 | complete | 12 | 11 | none | 0.063471 | no | 0.132951 | no | 332.7 |
| 11 | 4 | 43 | complete | 12 | 12 | none | 0.362851 | no | 0.024059 | no | 331.7 |
| 12 | 4 | 44 | complete | 12 | 12 | none | 0.721819 | no | 0.986754 | no | 333.1 |

Rule (n) of the pre-registration (PROTOCOL.md §6): a non-finite value in the history does not stop the run; without a finite val_loss, with non-finite restored weights, non-finite scores or logits, or non-finite re-estimated statistics the model is not evaluable and the run goes on.

## One-class runs at threshold 0.5 on the dev (definition (a))

| network | patience | BN state | count |
|---|---|---|---|
| AttackNetV1 | 12 | ordinary | 0 of 12 evaluable (all bona fide 0, all attack 0, not evaluable 0) |
| AttackNetV1 | 12 | re-estimated | 0 of 12 evaluable (all bona fide 0, all attack 0, not evaluable 0) |

Dev frame identifiers (public outputs hold only counts and SHA-256): 12 models compared over 4 folds, both BN states; every model of a fold used the same frames in the same order.

Command: `python -I -B scripts/pad_anet_run.py run --config delivery/anetrun/pad_anet_run.json --out-root results/anetrun --run anet_v1_p12 --estimator-check results/anetrun/20261122-ANETEST-c882362 --attempt 1 --i-am-authorized`
