# Early dev saturation: one control on the BatchNormalization statistics (proposal, 12 November 2026; revised 14 November 2026)

Purpose: one control on the early dev saturation. Proposed on 12 November 2026; approved by the coordinator on 13
November 2026 on the 36 existing checkpoints, without new training and without reading the test, with three points to
specify before the run: the estimator (section 2); the baseline tolerances with the stop rule (section 4); conclusions
limited in both directions, 2×2 table included (section 4). This revision was frozen before any recalibration was run; the runner records its
SHA-256.

## 1 Hypothesis

Train accuracy is recorded in training mode (batch statistics, dropout) on augmented frames; the dev uses moving
statistics (momentum 0.99; 88 steps per epoch for B and A\*, 59 for B'). Hypotheses 1 and 2 of
`research/nuaa-degenerate-trainings/NOTE.md` (not tested): the moving statistics are not settled, or `bn2_1` and
`bn_dense` (first after `dropout1`, `dropout2`) estimate the variance with dropout and use it without; under them the
trained parameters would separate the dev and the evaluation statistics would not. Alternatives (hypotheses 4 and 7:
learning rate 1e-3 too high early, initialisation): the trained parameters themselves would not separate the dev.

For: dev accuracy 0.5 at epoch 1 in 10 of 10 and 24 of 26, train accuracy 0.9495 [0.9143-0.9807] (the ten), 0.9691 (the
26); residual (weight of the initial value) 0.4129 after epoch 1 (B, A\*; B' 0.5527). Against: residual 0.0704 after
epoch 3 (B, A\*; B' 0.1688), yet 24 of the 26 hit 0.5 at epoch 3 or later; in the 12-epoch runs the ten spend 2-7 epochs
at 0.5 (2-5, constant lr), the last at epoch 7 (B', residual 0.0158); five of the ten have dev AUROC 0.5000.

## 2 The one modification

For each checkpoint of run `20261106-SESSABLRUN-d999458` (the ten, saturated at epoch 1 or 2; the 26 as controls),
re-estimate mean and variance of the five BatchNormalization layers on the fitting set of its arm and fold (1,400 frames
for B and A\*, 934 for B'); evaluate the dev as the trainer does (400 frames, batch 16), before and after.

Estimator (fixed before the run), for each checkpoint, on an in-memory copy:

- Statistics: per channel, the global mean and variance of the input of each BatchNormalization layer over the whole
  fitting set of the arm and fold.
- Order: layer by layer in network order (`bn1_1`, `bn1_2`, `bn2_1`, `bn2_2`, `bn_dense`); the input of a layer is
  computed with the earlier layers already re-estimated (one fitting pass per layer).
- Mode: inference (`training=False`): dropout off, BatchNormalization on its moving statistics; augmentation off
  (frames / 255, as in scoring; the trainer's augmentation is a separate pipeline, not a layer of the saved model).
- Counts: N = frames × H × W per channel for the convolutional layers (256 × 256 for `bn1_1` and `bn1_2`, 128 × 128 for
  `bn2_1` and `bn2_2`: input 256 × 256, `same` padding, one 2 × 2 pooling, `src/architectures.py`), N = frames for
  `bn_dense`; batches of 16, the last one shorter (1,400 = 87 × 16 + 8; 934 = 58 × 16 + 6) and weighted by its own
  count.
- Variance: population variance (divisor N), accumulated in float64 by the exact union of the batches (count, mean and
  sum of squared deviations of each batch), equal to Σx²/N − mean² and to Σ_b n_b (var_b + (mean_b − mean)²)/N. It is
  not the mean of the batch variances; the mean is not the unweighted mean of the batch means. (During training Keras
  updated the moving variance of the four convolutional layers with the n/(n−1) batch variance: an N/(N−1) version
  would differ from the one used here by the relative factor 1/(N−1), less than 1e-7 for those layers and at most
  1/933 for `bn_dense`.)
- What changes: only `moving_mean` and `moving_variance`, set to the float32 of the float64 result. After the five
  layers every tensor of the copy (trained parameters and moving statistics) is compared byte for byte with the
  original: exactly these 10 tensors differ; kernels, biases, gamma and beta are identical, otherwise the run stops
  (FAIL).
- Consistency check: one more pass over the fitting set with the final copy recomputes the moments of the five inputs;
  they must equal the installed statistics within 1e-4 relative plus 1e-6 absolute (float32 cast and forward pass
  against float64 accumulation), otherwise FAIL and no reading. The same pass gives fitting accuracy and loss after.

## 3 What stays fixed

Trained parameters (kernels, biases, gamma, beta; compared byte for byte, section 2), restored epoch, data, folds,
scoring code, TF/Keras version, CPU; recalibration on an in-memory copy, `.keras` files intact (SHA-256 checked before
each load and at the end). No test frame is assembled or evaluated and no test score or metric is read: as in the
parent run, the frame files (cache and extracted frames) are hashed and loaded whole, but only fitting and dev frames
are assembled, and the fold schema (the test role) is not opened; the public aggregate of the parent run, which
contains test metrics, is only hashed as bytes, not parsed; no public file of the folds run is opened.

## 4 Reading rule (fixed before the run)

Baseline, before any recalibration: each file matches its `weights_sha256` (checked just before loading), and all 36
checkpoints are evaluated on the dev with their original statistics as the parent run did (CPU, batch 16, frames / 255,
inference mode). Tolerances:

- per-frame attack probability against the stored dev scores, with the parent's scorer and with the control's own
  evaluator: largest absolute difference ≤ 1e-5 and no decision at 0.5 changed (the tolerance of the coordinator's R2
  replication package, used in the C1 crop check, `research/c1-crop-check/README.md`: there scores stored on the GPU
  one image at a time were reproduced within 8.9e-8 in those conditions and within 2.9e-6 on the CPU in batches of 16;
  here stored scores and baseline share file, CPU and batch 16; not measured before the run, since no inference
  precedes it);
- val_loss from the logits (as Keras) against the recorded best val_loss: ≤ 1e-3 absolute (the absolute part of the
  parent runner's check; its alternative 1 % branch is not used, since 1 % of the largest recorded value, 60.9868, is
  0.61); the difference from the parent's own recomputation is reported;
- dev accuracy (argmax) against the recorded one: ≤ 1e-6, the same number of correct frames out of 400;
- softmax of the float64 logits against the float32 probabilities: ≤ 1e-4 (the parent runner's own check); the same
  bound holds after the recalibration, on the dev and on the fitting, where a failure stops the run (FAIL).

Measured on the stored files, without inference: the accuracy at 0.5 of the stored dev scores equals the recorded one
for 36 of 36 models; the val_loss recomputed by the parent runner differs from the recorded best val_loss by at most
2.9e-6 absolute and 2.5e-5 relative (two different models). The fitting is evaluated too (accuracy, unweighted loss),
without a reference. Stop rule: if any baseline check fails for at least one model, the run stops before any
recalibration (status BASELINE_FAILED) and the discrepancy is documented without interpretation (per model and check:
value, reference, difference, tolerance); no re-estimated statistic is computed and nothing is read from the
recalibration. An error that prevents the evaluation itself (unexpected architecture, non-finite output, SHA-256
mismatch) is a FAIL with its reason recorded and no table.

Positive: the recalibrated checkpoint meets the recovery criterion (dev accuracy >= 0.75, val_loss < 1.0). Primary:
positives among the ten. All ten: on these checkpoints, re-estimating the statistics on the fitting set, every trained
parameter unchanged, is sufficient to meet the criterion (the moving statistics, dropout and the augmentation of the
fitting set, hypotheses 1-3, are not separated); the later saturated epochs, at residual ≤ 0.0704, are not examined.
None: at the epoch-1/2 checkpoints this re-estimation is not sufficient to meet the criterion (other estimators, other
checkpoints and the behaviour during training are not examined). Some: the effect differs between checkpoints; no
overall statement, each case reported. In every case a recovery shows the effect of this recalibration on the examined
checkpoints, not a single cause of the saturation, and a missing recovery does not, in general, exclude a role of
BatchNormalization or dropout.

Secondary, not deciding: the controls (the 25 of the 26 meeting it at their checkpoint), where each one that no longer
meets the criterion on the dev after the re-estimation is listed as a worsening; B fold 3 seed 44 (restored epoch 3,
dev accuracy 0.2375, not degenerate), reported apart; dev AUROC; fitting accuracy and loss in inference mode, against
the same criterion. Fitting before × dev after: each cell counts checkpoints, no cell names a cause.

| fitting before (original statistics, inference mode) | dev after: meets the criterion | dev after: does not meet it |
|---|---|---|
| meets the criterion | both | fitting only |
| does not meet it | dev after only | neither |

Expected, not observed: no outcome is favoured. The five tied at AUROC 0.5000 may stay tied; the two that already order
the dev (AUROC 0.9975, 0.9652) still have dev accuracy 0.5 at their checkpoint. A mixed outcome, with no overall
statement, is possible.

## 5 Expected cost

On CPU, per model: five recalibration and two evaluation passes over the fitting set, two over the dev (10,600 frame
passes for B and A\*, 7,338 for B'). Derived, not measured: about 32 minutes for 36 models, upper estimate about 140
minutes; frame reading not priced. As implemented: the same seven fitting passes (the pass after also gives the
consistency check) and one more dev pass (the parent's scorer in the baseline); cap 240 minutes.

## 6 Report

All 36 checkpoints, every worsening included (criterion lost; fewer correct dev or fitting frames; higher dev val_loss
or fitting loss; lower dev AUROC; any change in that direction counts, with its size); dev and fitting before and
after; the per-layer change of the statistics; the baseline table; the actual time against the estimate.

## Limits

Only the restored checkpoint of each training is saved: other epochs cannot be examined. The checkpoints were selected
on the same dev (minimum dev val_loss). Small dev (2-3 subjects). The residual is the initial value's weight, not the
lag behind changing weights. Criteria defined after the patience-4 run. One estimator. `.keras` files and frames are
private.

## Not proposed

Chosen over a training control on the momentum or lr 1e-4 (each changes every trajectory): this one changes no trained
parameter and no trajectory. Possibilities only, to be agreed after the report; none follows from the outcome alone:
the momentum, the learning rate, the initialisation.

## Pre-commitment

Tolerances, stop rule, groups and the text of each case are fixed in the runner's configuration
(`livenesslab/delivery/bnrecal/nuaa_bn_recalibration.json`), which records the SHA-256 of this note; no training, no
test read.

Numbers: `tables/facts.md`; criteria: `research/nuaa-degenerate-trainings/NOTE.md`.
