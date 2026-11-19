# Technical trial 20261118-ARCHTRIAL-0858ae7

Status: **PASS**. NUAA, arm B, fold 1, seed 42: fitting 1400 frames, dev 400 frames; no test frame assembled. 3 fixed epochs per network, batch 16, device CPU (GPU hidden); no checkpoint saved. Wall-time cap 60 minutes of wall time (elapsed real time, not CPU time).

Started 2026-11-18T22:54:48+0100, finished 2026-11-18T22:59:14+0100; wall time 266 s, process CPU time (all threads) 1618 s; peak RSS 5776 MB.

Machine: Apple M5 Pro, 5 performance and 10 efficiency cores, 24.0 GB RAM, macOS 27.2.1; TensorFlow 2.15.0, Keras 2.15.0, Python 3.11.16, NumPy 1.26.4.

## Epochs

| network | epoch | loss | accuracy | val_loss | val_accuracy | lr | seconds | validation seconds | CPU seconds |
|---|---|---|---|---|---|---|---|---|---|
| LivenessNet | 1 | 0.058099 | 0.9800 | 11.523386 | 0.5000 | 0.001 | 23.7 | 1.6 | 115.8 |
| LivenessNet | 2 | 0.024543 | 0.9921 | 17.259323 | 0.5000 | 0.001 | 24.8 | 1.4 | 114.0 |
| LivenessNet | 3 | 0.017976 | 0.9936 | 22.904610 | 0.5000 | 0.001 | 24.6 | 1.4 | 114.8 |
| ResNet-18 | 1 | 0.254850 | 0.9357 | 27.826838 | 0.4525 | 0.001 | 64.5 | 4.7 | 339.2 |
| ResNet-18 | 2 | 0.053078 | 0.9800 | 0.168487 | 0.9650 | 0.001 | 52.5 | 3.0 | 370.6 |
| ResNet-18 | 3 | 0.035589 | 0.9886 | 33.603695 | 0.5025 | 0.001 | 46.7 | 3.0 | 384.4 |

The histories are technical outputs of the trial, not results.

## Inference and rates

| network | trainable parameters | BN layers | steps per epoch | inference fitting s | inference dev s | train s per 1,000 fitting frames per epoch | validation s per 1,000 frames | inference s per 1,000 frames | s per training outside the epochs | CPU / wall |
|---|---|---|---|---|---|---|---|---|---|---|
| LivenessNet | 8405778 | 5 | 88 | 5.3 | 1.6 | 16.4 | 3.6 | 3.8 | 1.8 | 4.69 |
| ResNet-18 | 11177538 | 20 | 88 | 9.8 | 2.7 | 36.4 | 8.9 | 7.0 | 1.3 | 6.92 |

Rates: mean over the measured epochs, the first included (it includes the graph tracing: a cautious estimate); outside the epochs: the whole trainer minus the sum of its epochs (model building, data preparation before the fit, time between and after the epochs).

Data: the assembled fitting and dev keys are equal to fitting_keys_sha256 and dev_keys_sha256 of the reference card: fitting_keys_equal_to_reference_card True, dev_keys_equal_to_reference_card True.

## LivenessNet history against the accepted card

Outcome: **not reproduced** (first 3 epochs of arm_b_fold1_seed42 in 20261106-SESSABLRUN-d999458; fields loss, accuracy, val_loss, val_accuracy, lr). Rule: the shared-history rule of the project (nuaa_patience_verify.shared_history, criteria.shared_history_difference = 0 of nuaa_patience_verify.json): the five values of each of the first 3 epochs identical, difference 0, no tolerance; per field the largest absolute difference and the first differing epoch are reported

Largest absolute difference per field: loss 0.00538, accuracy 0.00357, val_loss 6.17, val_accuracy 0, lr 0.

Consequence (fixed before the trial): the outcome is recorded and does not change the status of the trial; it is the first check of the copy of the trainer; in the plan the same check is repeated on every epoch of every one of the 24 LivenessNet trainings (against the accepted run at the same patience); if the history is not reproduced in the trial, the difference is reported and the pre-registration is not closed before it is explained

Command: `python -I -B scripts/pad_arch_comparison.py trial --i-am-authorized`
