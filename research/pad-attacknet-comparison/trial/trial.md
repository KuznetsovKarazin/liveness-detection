# Technical trial 20261122-ANETTRIAL-a951ed0: AttackNetV1, AttackNetV2_1, AttackNetV2_2

Status: **PASS**. NUAA, arm B, fold 1, seed 42: fitting 1400 frames, dev 400 frames; no test frame assembled. 3 epochs, patience 4, batch 16, device CPU (GPU hidden). Wall-time cap 60 minutes of wall time (elapsed real time, not CPU time).

Started 2026-11-22T01:12:46+0100, finished 2026-11-22T01:20:10+0100; wall time 444 s, process CPU time (all threads) 3362 s; peak RSS 5621 MB. Attempt 1; earlier attempts: none.

Machine: Apple M5 Pro, cores as named by the system: 5 Super, 10 Performance; 15 logical CPUs, 24.0 GB RAM, macOS 27.2.1; TensorFlow 2.15.0, Keras 2.15.0, Python 3.11.16, NumPy 1.26.4.

TensorFlow threads set by this runner: requested intra-op 10 and inter-op 10, effective (read back from tf.config.threading) intra-op 10 and inter-op 10; environment variables: TF_NUM_INTRAOP_THREADS absent, TF_NUM_INTEROP_THREADS absent; logical CPUs 15.

## LivenessNet control (not one of the three trials)

Outcome: **reproduced**. Rule: LivenessNet is trained first, with the same trainer, as a control of the machine and threads (not one of the three trials): the five quantities of epochs 1-3 must be identical, difference 0, to the accepted card arm_b_fold1_seed42 and to the C2 trial 20261119-ARCHTRIAL-b8f7cca, and the machine and versions equal, field by field, to those of the C2 trial (the rule of pad_arch_run.machine_rule, recorded); otherwise the outcome is CONTROL_FAILED

- against the accepted card (arm_b_fold1_seed42 of 20261106-SESSABLRUN-d999458): reproduced
- against the C2 trial 20261119-ARCHTRIAL-b8f7cca: reproduced
- machine and versions against the C2 trial: equal

## Epochs

| network | epoch | loss | accuracy | val_loss | val_accuracy | lr | seconds | validation seconds | CPU seconds |
|---|---|---|---|---|---|---|---|---|---|
| LivenessNet | 1 | 0.058354 | 0.980714 | 7.061067 | 0.500000 | 0.001 | 14.1 | 0.9 | 113.4 |
| LivenessNet | 2 | 0.029926 | 0.988571 | 11.087085 | 0.500000 | 0.001 | 13.6 | 0.8 | 113.7 |
| LivenessNet | 3 | 0.015531 | 0.994286 | 25.260088 | 0.500000 | 0.001 | 13.7 | 0.8 | 114.6 |
| AttackNetV1 | 1 | 0.246902 | 0.907143 | 0.907735 | 0.650000 | 0.001 | 27.4 | 1.7 | 228.6 |
| AttackNetV1 | 2 | 0.186740 | 0.935000 | 4.744017 | 0.500000 | 0.001 | 26.6 | 1.5 | 226.1 |
| AttackNetV1 | 3 | 0.182728 | 0.940714 | 4.108798 | 0.500000 | 0.001 | 26.4 | 1.6 | 224.9 |
| AttackNetV2_1 | 1 | 0.227617 | 0.917143 | 0.881117 | 0.500000 | 0.001 | 27.5 | 1.7 | 231.5 |
| AttackNetV2_1 | 2 | 0.169946 | 0.942143 | 4.433551 | 0.500000 | 0.001 | 27.1 | 1.6 | 230.6 |
| AttackNetV2_1 | 3 | 0.181086 | 0.932143 | 2.829569 | 0.620000 | 0.001 | 26.8 | 1.6 | 227.9 |
| AttackNetV2_2 | 1 | 0.207670 | 0.920714 | 2.011398 | 0.500000 | 0.001 | 22.8 | 1.5 | 187.5 |
| AttackNetV2_2 | 2 | 0.280183 | 0.900000 | 0.605395 | 0.745000 | 0.001 | 22.2 | 1.4 | 186.3 |
| AttackNetV2_2 | 3 | 0.182729 | 0.931429 | 0.056570 | 0.982500 | 0.001 | 22.1 | 1.4 | 184.3 |

The histories are technical outputs of the trial, not results.

## Outcome, BN check and timings per AttackNet

| network | evaluable | trainable parameters | BN layers | BN check | s per epoch | s outside the epochs | s of phase 2 | inference s per 1,000 frames | CPU / wall |
|---|---|---|---|---|---|---|---|---|---|
| AttackNetV1 | yes | 33588194 | 5 | passed | 26.8 | 1.0 | 45.1 | 4.3 | 7.65 |
| AttackNetV2_1 | yes | 33588194 | 5 | passed | 27.1 | 1.2 | 45.5 | 4.3 | 7.65 |
| AttackNetV2_2 | yes | 16806274 | 5 | passed | 22.4 | 1.3 | 38.4 | 3.8 | 7.63 |

Save, reload and copies (seconds; largest absolute differences of the dev attack scores; agreement: at most 1e-5 and no discordant decision at 0.5 and at the EER threshold, the rule of PR #27):

| network | inference fitting s | inference dev s | save s | scorer s | reload s | ordinary_dev s | reestimate s | consistency s | re_estimated_dev s | installed_copy s | reloaded weights equal | reloaded vs memory | scorer vs evaluator | installed copy vs re-estimated copy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AttackNetV1 | 6.0 | 1.7 | 0.1 | 1.9 | 0.2 | 1.7 | 25.9 | 13.5 | 1.8 | 1.7 | True | 0 | 0 | 0 |
| AttackNetV2_1 | 6.1 | 1.7 | 0.1 | 1.9 | 0.2 | 1.7 | 26.0 | 13.6 | 1.8 | 1.7 | True | 0 | 0 | 0 |
| AttackNetV2_2 | 5.4 | 1.5 | 0.1 | 1.7 | 0.1 | 1.5 | 22.5 | 11.0 | 1.6 | 1.5 | True | 0 | 0 | 0 |

BN check per layer (values per channel counted against frames x H x W of the registry; changes of the installed float32 statistics against the original ones):

| network | layer | channels | values per channel | expected | mean changed | variance changed | largest change of the mean | largest change of the variance |
|---|---|---|---|---|---|---|---|---|
| AttackNetV1 | bn1_1 | 16 | 91750400 | 91750400 | True | True | 0.137 | 0.414 |
| AttackNetV1 | bn1_final | 32 | 91750400 | 91750400 | True | True | 0.444 | 0.42 |
| AttackNetV1 | bn2_1 | 32 | 22937600 | 22937600 | True | True | 0.704 | 2.84 |
| AttackNetV1 | bn2_final | 64 | 22937600 | 22937600 | True | True | 1.55 | 12 |
| AttackNetV1 | bn_dense | 128 | 1400 | 1400 | True | True | 1.08 | 0.895 |
| AttackNetV2_1 | bn1_1 | 16 | 91750400 | 91750400 | True | True | 0.185 | 0.413 |
| AttackNetV2_1 | bn1_final | 32 | 91750400 | 91750400 | True | True | 0.454 | 0.435 |
| AttackNetV2_1 | bn2_1 | 32 | 22937600 | 22937600 | True | True | 0.516 | 1.31 |
| AttackNetV2_1 | bn2_final | 64 | 22937600 | 22937600 | True | True | 1.32 | 5.61 |
| AttackNetV2_1 | bn_dense | 128 | 1400 | 1400 | True | True | 1.15 | 0.695 |
| AttackNetV2_2 | bn1_1 | 16 | 91750400 | 91750400 | True | True | 0.0815 | 0.0723 |
| AttackNetV2_2 | bn1_final | 16 | 91750400 | 91750400 | True | True | 0.4 | 1.12 |
| AttackNetV2_2 | bn2_1 | 32 | 22937600 | 22937600 | True | True | 0.524 | 2.46 |
| AttackNetV2_2 | bn2_final | 32 | 22937600 | 22937600 | True | True | 1.57 | 14 |
| AttackNetV2_2 | bn_dense | 128 | 1400 | 1400 | True | True | 0.972 | 0.646 |

## Hours of the plan from this trial

Training of the 72 trainings with phase 2: 5.20-6.98 hours of wall time, 39.77-53.37 CPU hours; evaluation (once): 0.59 hours of wall time, 4.50 CPU hours; caps of the six runs: anet_v1_p4 2.5 h, anet_v2_1_p4 2.5 h, anet_v2_2_p4 2.0 h, anet_v1_p12 2.5 h, anet_v2_1_p12 2.5 h, anet_v2_2_p12 2.0 h. Rule: `plan/run_plan.md`.

Checkpoints: outside the repository (unofficial output, not published); SHA-256: AttackNetV1 4bc4347e1de9f72eeff2e343ee7f5cf6c92d1d4e028b165988f00497342488a1, AttackNetV2_1 ddcdaeb9c9b89526f89d907fe44baae38340e5517ec470f31b9e0a95cbbae388, AttackNetV2_2 b8db8da5d4c81a8d7b084256a27bfdbe5db46fe1e1d82cf38ef2bd6fdef466bf.

Command: `python -I -B scripts/pad_anet_comparison.py trial --attempt 1 --i-am-authorized --checkpoint-root <a folder outside the repository>`
