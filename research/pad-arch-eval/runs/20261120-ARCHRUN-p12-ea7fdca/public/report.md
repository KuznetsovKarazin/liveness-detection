# NUAA run nuaa_patience_12 (20261120-ARCHRUN-p12-ea7fdca)

Status: **PASS**. Attempt 1. Patience 12; 24 positions of `run_order.csv`; phase 1 (training, checkpoint, dev scored with the scorer of the accepted runs) complete for 24, phase 2 (both BN states on the dev with the evaluator of PR #27, BN re-estimated on the fitting set) for 24. No test frame is assembled or evaluated, no test score or metric is read: the frame files, hashed and loaded whole, also hold frames of the held-out subjects, of which only the fitting and dev frames of arm B are assembled; every number below is a dev or training number.

Started 2026-11-20T22:03:54+0100, finished 2026-11-21T00:53:14+0100; elapsed time 10160 s of a cap of 5.00 hours of wall time (elapsed real time, not CPU time); process CPU time (all threads) 88270 s.

Machine: Apple M5 Pro, cores as named by the system: 5 Super, 10 Performance; 15 logical CPUs, 24.0 GB RAM, macOS 27.2.1; equal to the official trial 20261119-ARCHTRIAL-b8f7cca in processor, core_levels, ram_gb, os, logical_cpus, architecture and in the versions python, numpy, tensorflow_installed, keras, platform (machine model and versions, not the physical unit).

TensorFlow threads set by this runner: requested intra-op 10 and inter-op 10, effective (read back from tf.config.threading) intra-op 10 and inter-op 10; environment variables: TF_NUM_INTRAOP_THREADS absent, TF_NUM_INTEROP_THREADS absent; logical CPUs 15.

BN estimator: checked by 20261120-ARCHEST-6c4f9e8 (36 models of the accepted run reproduced).

## Models

| position | network | fold | seed | status | epochs | restored epoch | history check | first differing epoch | ordinary: EER threshold | ordinary: one-class at 0.5 | re-estimated: EER threshold | re-estimated: one-class at 0.5 | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | LivenessNet | 1 | 42 | complete | 12 | 10 | identical | none | 0.000768 | no | 0.000799 | no | 174.3 |
| 2 | ResNet-18 | 1 | 42 | complete | 12 | 3 | not applicable | NA | 0.485716 | no | 0.788562 | no | 473.1 |
| 3 | LivenessNet | 1 | 43 | complete | 12 | 11 | identical | none | 0.542399 | no | 0.516693 | no | 189.2 |
| 4 | ResNet-18 | 1 | 43 | complete | 12 | 12 | not applicable | NA | 0.293295 | no | 0.170445 | no | 471.9 |
| 5 | LivenessNet | 1 | 44 | complete | 12 | 11 | identical | none | 0.998644 | no | 0.012823 | no | 176.7 |
| 6 | ResNet-18 | 1 | 44 | complete | 12 | 9 | not applicable | NA | 0.326342 | no | 0.924811 | no | 449.1 |
| 7 | LivenessNet | 2 | 42 | complete | 12 | 9 | identical | none | 0.920426 | no | 0.009805 | no | 178.2 |
| 8 | ResNet-18 | 2 | 42 | complete | 12 | 10 | not applicable | NA | 0.263727 | no | 0.058028 | no | 458.9 |
| 9 | LivenessNet | 2 | 43 | complete | 12 | 7 | identical | none | 0.046983 | no | 0.001650 | no | 172.3 |
| 10 | ResNet-18 | 2 | 43 | complete | 12 | 9 | not applicable | NA | 0.837763 | no | 0.092863 | no | 453.4 |
| 11 | LivenessNet | 2 | 44 | complete | 12 | 10 | identical | none | 0.578237 | no | 0.002602 | no | 173.1 |
| 12 | ResNet-18 | 2 | 44 | complete | 12 | 11 | not applicable | NA | 0.817033 | no | 0.383225 | no | 465.6 |
| 13 | LivenessNet | 3 | 42 | complete | 12 | 12 | identical | none | 0.381979 | no | 0.194226 | no | 187.2 |
| 14 | ResNet-18 | 3 | 42 | complete | 12 | 11 | not applicable | NA | 0.359764 | no | 0.031843 | no | 497.6 |
| 15 | LivenessNet | 3 | 43 | complete | 12 | 10 | identical | none | 0.031980 | no | 0.017807 | no | 190.1 |
| 16 | ResNet-18 | 3 | 43 | complete | 12 | 9 | not applicable | NA | 0.048411 | no | 0.005191 | no | 497.4 |
| 17 | LivenessNet | 3 | 44 | complete | 12 | 11 | identical | none | 0.203914 | no | 0.227202 | no | 187.4 |
| 18 | ResNet-18 | 3 | 44 | complete | 12 | 11 | not applicable | NA | 0.176968 | no | 0.068312 | no | 494.2 |
| 19 | LivenessNet | 4 | 42 | complete | 12 | 11 | identical | none | 0.016334 | no | 0.001697 | no | 187.3 |
| 20 | ResNet-18 | 4 | 42 | complete | 12 | 10 | not applicable | NA | 0.132695 | no | 0.481099 | no | 494.2 |
| 21 | LivenessNet | 4 | 43 | complete | 12 | 12 | identical | none | 0.012818 | no | 0.084566 | no | 188.0 |
| 22 | ResNet-18 | 4 | 43 | complete | 12 | 12 | not applicable | NA | 0.059742 | no | 0.627932 | no | 494.1 |
| 23 | LivenessNet | 4 | 44 | complete | 12 | 11 | identical | none | 0.759310 | no | 0.295633 | no | 189.2 |
| 24 | ResNet-18 | 4 | 44 | complete | 12 | 2 | not applicable | NA | 0.011534 | no | 0.047052 | no | 483.0 |

History check (LivenessNet, against the card of the same arm B fold and seed in the accepted run at the same patience; five fields, difference 0; an epoch present in one history only counts as differing): a different history is reported and the model stays in the comparison (PROTOCOL.md §7).

## One-class runs at threshold 0.5 on the dev (definition (a))

| network | patience | BN state | count |
|---|---|---|---|
| LivenessNet | 12 | ordinary | 0 of 12 evaluable (all bona fide 0, all attack 0, not evaluable 0) |
| LivenessNet | 12 | re-estimated | 0 of 12 evaluable (all bona fide 0, all attack 0, not evaluable 0) |
| ResNet-18 | 12 | ordinary | 0 of 12 evaluable (all bona fide 0, all attack 0, not evaluable 0) |
| ResNet-18 | 12 | re-estimated | 0 of 12 evaluable (all bona fide 0, all attack 0, not evaluable 0) |

Dev frame identifiers (public outputs hold only counts and SHA-256): comparisons made between the two networks of a unit: 12 units with ordinary BN and 12 with re-estimated BN; between the two BN states: 12 units; every comparison made found the same frames in the same order.

Command: `python -I -B scripts/pad_arch_run.py run --config delivery/archrun/pad_arch_run.json --out-root results/archrun --run nuaa_patience_12 --estimator-check results/archrun/20261120-ARCHEST-6c4f9e8 --attempt 1 --i-am-authorized`
