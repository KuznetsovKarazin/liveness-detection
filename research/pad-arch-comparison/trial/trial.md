# Technical trial 20261119-ARCHTRIAL-b8f7cca

Status: **PASS**. NUAA, arm B, fold 1, seed 42: fitting 1400 frames, dev 400 frames; no test frame assembled. 3 fixed epochs per network, batch 16, device CPU (GPU hidden); no checkpoint saved. Wall-time cap 60 minutes of wall time (elapsed real time, not CPU time).

Started 2026-11-19T21:04:47+0100, finished 2026-11-19T21:07:47+0100; wall time 180 s, process CPU time (all threads) 1534 s; peak RSS 5299 MB.

Machine: Apple M5 Pro, cores as named by the system: 5 Super, 10 Performance; 15 logical CPUs, 24.0 GB RAM, macOS 27.2.1; TensorFlow 2.15.0, Keras 2.15.0, Python 3.11.16, NumPy 1.26.4.

TensorFlow threads set by this runner: requested intra-op 10 and inter-op 10, effective (read back from tf.config.threading) intra-op 10 and inter-op 10; environment variables: TF_NUM_INTRAOP_THREADS absent, TF_NUM_INTEROP_THREADS absent; logical CPUs 15.

## Epochs

| network | epoch | loss | accuracy | val_loss | val_accuracy | lr | seconds | validation seconds | CPU seconds |
|---|---|---|---|---|---|---|---|---|---|
| LivenessNet | 1 | 0.058354 | 0.9807 | 7.061067 | 0.5000 | 0.001 | 13.8 | 0.9 | 112.6 |
| LivenessNet | 2 | 0.029926 | 0.9886 | 11.087085 | 0.5000 | 0.001 | 13.7 | 0.8 | 114.7 |
| LivenessNet | 3 | 0.015531 | 0.9943 | 25.260088 | 0.5000 | 0.001 | 13.9 | 0.8 | 115.9 |
| ResNet-18 | 1 | 0.226699 | 0.9479 | 1.007320 | 0.8225 | 0.001 | 37.6 | 3.0 | 342.9 |
| ResNet-18 | 2 | 0.061645 | 0.9750 | 0.339604 | 0.8450 | 0.001 | 36.8 | 2.9 | 341.1 |
| ResNet-18 | 3 | 0.043853 | 0.9836 | 0.001874 | 1.0000 | 0.001 | 37.1 | 2.9 | 343.3 |

The histories are technical outputs of the trial, not results.

## Inference and rates

| network | trainable parameters | BN layers | steps per epoch | inference fitting s | inference dev s | train s per 1,000 fitting frames per epoch | validation s per 1,000 frames | inference s per 1,000 frames | s per training outside the epochs | CPU / wall |
|---|---|---|---|---|---|---|---|---|---|---|
| LivenessNet | 8405778 | 5 | 88 | 3.4 | 1.0 | 9.2 | 2.2 | 2.4 | 1.3 | 8.11 |
| ResNet-18 | 11177538 | 20 | 88 | 10.8 | 3.1 | 24.5 | 7.3 | 7.7 | 0.8 | 9.09 |

Rates: mean over the measured epochs, the first included (it includes the graph tracing: a cautious estimate); outside the epochs: the whole trainer minus the sum of its epochs (model building, data preparation before the fit, time between and after the epochs).

Data: the assembled fitting and dev keys are equal to fitting_keys_sha256 and dev_keys_sha256 of the reference card: fitting_keys_equal_to_reference_card True, dev_keys_equal_to_reference_card True.

## LivenessNet history against the accepted card

Outcome: **reproduced** (first 3 epochs of arm_b_fold1_seed42 in 20261106-SESSABLRUN-d999458; fields loss, accuracy, val_loss, val_accuracy, lr). Rule: the shared-history rule of the project (nuaa_patience_verify.shared_history, criteria.shared_history_difference = 0 of nuaa_patience_verify.json): the five values of each of the first 3 epochs identical, difference 0, no tolerance, computed with one source, pad_arch_comparison.history_differences (a non-finite value, NaN or infinite, in the compared epochs stops the trial as FAIL with no outputs); per field the first epoch at which the two values differ (first_differing_epoch), the largest absolute difference (max_abs_difference) and the first epoch at which it is reached (epoch_of_max_abs_difference) are reported; in PR #29 the field first_differing_epoch held the epoch of the largest difference (our defect, corrected with a test on 19 November 2026)

| field | first differing epoch | largest absolute difference | epoch of the largest difference |
|---|---|---|---|
| loss | none | 0 | none |
| accuracy | none | 0 | none |
| val_loss | none | 0 | none |
| val_accuracy | none | 0 | none |
| lr | none | 0 | none |

First differing epoch: the first epoch at which the two values differ; the largest difference is reported with the first epoch at which it is reached.

Consequence (fixed before the trial): the outcome is recorded and does not change the status of the trial; it is the first check of the copy of the trainer; in the plan the same check is repeated on every epoch of every one of the 24 LivenessNet trainings (against the accepted run at the same patience); if the history is not reproduced in the trial, the difference is reported and the pre-registration is not closed before it is explained

Command: `python -I -B scripts/pad_arch_comparison.py trial --i-am-authorized`
