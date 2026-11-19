# Technical trial 20261119-ARCHTRIAL-b8e78b2

Status: **PASS**. NUAA, arm B, fold 1, seed 42: fitting 1400 frames, dev 400 frames; no test frame assembled. 3 fixed epochs per network, batch 16, device CPU (GPU hidden); no checkpoint saved. Wall-time cap 60 minutes of wall time (elapsed real time, not CPU time).

Started 2026-11-19T06:11:44+0100, finished 2026-11-19T06:14:44+0100; wall time 180 s, process CPU time (all threads) 1526 s; peak RSS 5893 MB.

Machine: Apple M5 Pro, 5 performance and 10 efficiency cores, 24.0 GB RAM, macOS 27.2.1; TensorFlow 2.15.0, Keras 2.15.0, Python 3.11.16, NumPy 1.26.4.

## Epochs

| network | epoch | loss | accuracy | val_loss | val_accuracy | lr | seconds | validation seconds | CPU seconds |
|---|---|---|---|---|---|---|---|---|---|
| LivenessNet | 1 | 0.058354 | 0.9807 | 7.061067 | 0.5000 | 0.001 | 13.6 | 0.9 | 110.9 |
| LivenessNet | 2 | 0.029926 | 0.9886 | 11.087085 | 0.5000 | 0.001 | 13.3 | 0.8 | 111.6 |
| LivenessNet | 3 | 0.015531 | 0.9943 | 25.260088 | 0.5000 | 0.001 | 13.6 | 0.9 | 114.0 |
| ResNet-18 | 1 | 0.226699 | 0.9479 | 1.007320 | 0.8225 | 0.001 | 37.1 | 3.0 | 336.9 |
| ResNet-18 | 2 | 0.061645 | 0.9750 | 0.339604 | 0.8450 | 0.001 | 36.9 | 2.9 | 340.8 |
| ResNet-18 | 3 | 0.043853 | 0.9836 | 0.001874 | 1.0000 | 0.001 | 37.7 | 3.0 | 346.7 |

The histories are technical outputs of the trial, not results.

## Inference and rates

| network | trainable parameters | BN layers | steps per epoch | inference fitting s | inference dev s | train s per 1,000 fitting frames per epoch | validation s per 1,000 frames | inference s per 1,000 frames | s per training outside the epochs | CPU / wall |
|---|---|---|---|---|---|---|---|---|---|---|
| LivenessNet | 8405778 | 5 | 88 | 3.4 | 1.0 | 9.0 | 2.2 | 2.4 | 1.2 | 8.12 |
| ResNet-18 | 11177538 | 20 | 88 | 11.1 | 3.2 | 24.5 | 7.3 | 7.9 | 1.2 | 9.02 |

Rates: mean over the measured epochs, the first included (it includes the graph tracing: a cautious estimate); outside the epochs: the whole trainer minus the sum of its epochs (model building, data preparation before the fit, time between and after the epochs).

Data: the assembled fitting and dev keys are equal to fitting_keys_sha256 and dev_keys_sha256 of the reference card: fitting_keys_equal_to_reference_card True, dev_keys_equal_to_reference_card True.

## LivenessNet history against the accepted card

Outcome: **reproduced** (first 3 epochs of arm_b_fold1_seed42 in 20261106-SESSABLRUN-d999458; fields loss, accuracy, val_loss, val_accuracy, lr). Rule: the shared-history rule of the project (nuaa_patience_verify.shared_history, criteria.shared_history_difference = 0 of nuaa_patience_verify.json): the five values of each of the first 3 epochs identical, difference 0, no tolerance; per field the largest absolute difference and the first differing epoch are reported

Largest absolute difference per field: loss 0, accuracy 0, val_loss 0, val_accuracy 0, lr 0.

Consequence (fixed before the trial): the outcome is recorded and does not change the status of the trial; it is the first check of the copy of the trainer; in the plan the same check is repeated on every epoch of every one of the 24 LivenessNet trainings (against the accepted run at the same patience); if the history is not reproduced in the trial, the difference is reported and the pre-registration is not closed before it is explained

Command: `python -I -B scripts/pad_arch_comparison.py trial --i-am-authorized`
