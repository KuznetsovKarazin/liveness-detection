# Early dev saturation: one control on the BatchNormalization statistics (proposal, 12 November 2026)

Purpose: one control on the early dev saturation. Proposal only: nothing has been run.

## 1 Hypothesis

Train accuracy is recorded in training mode (batch statistics, dropout) on augmented frames; the dev uses moving
statistics (momentum 0.99; 88 steps per epoch for B and A\*, 59 for B'). Hypotheses 1, 2 (degenerate note): statistics
not settled, or `bn2_1` and `bn_dense` (first after `dropout1`, `dropout2`) estimate the variance with dropout and use
it without; the weights discriminate, the evaluation statistics do not. Alternatives (hypotheses 4 and 7): the weights
do not separate the dev.

For: dev accuracy 0.5 at epoch 1 in 10 of 10 and 24 of 26, train accuracy 0.9495 [0.9143-0.9807] (the ten), 0.9691 (the
26); residual (weight of the initial value) 0.4129 after epoch 1 (B, A\*; B' 0.5527). Against: residual 0.0704 after
epoch 3 (B, A\*; B' 0.1688), yet 24 of the 26 hit 0.5 at epoch 3 or later; in the 12-epoch runs the ten spend 2-7 epochs
at 0.5 (2-5, constant lr), the last at epoch 7 (B', residual 0.0158); five of the ten have dev AUROC 0.5000.

## 2 The one modification

For each checkpoint of run `20261106-SESSABLRUN-d999458` (the ten, saturated at epoch 1 or 2; the 26 as controls),
re-estimate mean and variance of the five BatchNormalization layers on the fitting set of its arm and fold (1,400 frames
for B and A\*, 934 for B'), in inference mode, unaugmented, layer by layer (divisor N, as Keras computes the batch
variance); evaluate the dev as the trainer does (400 frames, batch 16), before and after.

## 3 What stays fixed

Weights, restored epoch, data, folds, scoring code, TF/Keras version, CPU; recalibration on a copy, `.keras` files
intact.

## 4 Reading rule (fixed before the run)

Validity first: each file matches its `weights_sha256` and reproduces, before recalibration, the recorded dev accuracy
and val_loss (largest difference reported); otherwise no verdict.

Positive: the recalibrated checkpoint meets the recovery criterion (dev accuracy >= 0.75, val_loss < 1.0). Primary:
positives among the ten. All ten: at that checkpoint the saturation is attributable to the evaluation statistics
(hypotheses 1, 2 and the augmentation part of 3, not separated), not to the weights; this would not explain saturated
epochs at residual ≤ 0.0704. None: statistics re-estimated on the fitting set are not sufficient at the epoch-1/2
checkpoint. Some: mixed, no overall verdict.

Secondary, not deciding: losses of the criterion among the 25 of the 26 meeting it (a loss means the recalibration
degrades the dev, weakening a negative); B fold 3 seed 44 (restored epoch 3, dev accuracy 0.2375, not degenerate) apart;
dev AUROC; fitting accuracy and loss in inference mode, above or below the same criterion. Fitting before × dev after:
below/recovered, a train/inference mismatch (statistics, dropout or augmentation); below/not, fails even on its own
frames; above/recovered, the moving statistics penalise the dev only; above/not, against 1 and 2 at that checkpoint: the
weights separate fitting frames, not dev subjects (subject shift; 7 remains possible).

Expected, not observed: no outcome is favoured. The five tied at AUROC 0.5000 may stay tied; the two that already order
the dev (AUROC 0.9975, 0.9652) need only an output shift. A mixed outcome, with no overall verdict, is possible.

## 5 Expected cost

On CPU, per model: five recalibration and two evaluation passes over the fitting set, two over the dev (10,600 frame
passes for B and A\*, 7,338 for B'). Derived, not measured: about 32 minutes for 36 models, upper estimate about 140
minutes; frame reading not priced.

## Limits

Only the restored checkpoint is saved (recovered in the 12-epoch runs). Small dev (2-3 subjects). The residual is the
initial value's weight, not the lag behind changing weights. Criteria defined after the patience-4 run. `.keras` files
and frames are private.

## Not proposed

Chosen over a training control on the momentum or lr 1e-4 (each changes every trajectory): this one changes no weight
and no trajectory. Possibilities only, to be agreed: if positive, the momentum; if negative, learning rate or
initialisation.

## Pre-commitment

No inference before the coordinator agrees; no test read.

Numbers: `tables/facts.md`; criteria: `research/nuaa-degenerate-trainings/NOTE.md`.
