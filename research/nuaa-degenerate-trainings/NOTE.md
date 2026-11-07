# Degenerate trainings of the session ablation run: what the train/dev curves show, one change to the recipe, verification protocol

Numbers from run `20261107-DEGEN-a3b4dc0` (`tables/aggregate.md`, T1-T13; `curves.csv` for single epochs); no training.

## Question and material

The coordinator asked (C2, 6 November 2026) to study the 10 degenerate trainings of run `20261106-SESSABLRUN-d999458`
with the recorded train/dev curves and propose one motivated recipe change, chosen on train/dev only, with protocol and
time estimate. Material: the 36 training cards of that run; the 12 of the folds run `20261103-FOLDSRUN-e174812` (same
recipe, other fitting set and dev, 132 steps per epoch against 88 for B and A\* and 59 for B'); the 6 of the failed
attempt `20261106-SESSABLRUN-3b4021f` (determinism only). A card holds the restored epoch and, per epoch, `loss`,
`accuracy`, `val_loss`, `val_accuracy`, `lr`; the dev scores of the checkpoint are read too. `loss`/`accuracy` are in
training mode (augmentation, dropout, batch statistics), `val_*` on the dev in inference mode (moving statistics, no
dropout). Criteria defined after the run; "non-degenerate" = by the criterion, not necessarily recovered.

## What we observe

**The ten (T1, T6).** Degenerate = restored epoch 1 or 2 with dev accuracy 0.5 at that epoch: 10 of 36; B 2, A\* 5,
B' 3; seed 42: 2, seed 43: 1, seed 44: 7. All stopped at epoch 5 (restored 1) or 6 (restored 2).

| arm | fold | seed | epochs run | restored | best val_loss | dev val_loss per epoch |
|---|---|---|---|---|---|---|
| A\* | 2 | 43 | 5 | 1 | 60.9868 | 61.0 107.3 129.9 141.1 82.6 |
| A\* | 2 | 44 | 5 | 1 | 18.0934 | 18.1 54.3 53.4 31.7 67.1 |
| A\* | 3 | 42 | 5 | 1 | 3.1296 | 3.1 4.1 48.6 13.5 70.9 |
| A\* | 3 | 44 | 5 | 1 | 33.4090 | 33.4 44.9 197.9 133.1 66.0 |
| A\* | 4 | 44 | 5 | 1 | 22.8274 | 22.8 61.3 40.9 61.1 37.1 |
| B | 1 | 42 | 5 | 1 | 7.0611 | 7.1 11.1 25.3 70.4 58.0 |
| B | 4 | 44 | 6 | 2 | 2.3605 | 16.5 2.4 3.5 14.5 4.2 4.6 |
| B' | 1 | 44 | 5 | 1 | 12.1449 | 12.1 22.7 19.3 22.3 25.9 |
| B' | 2 | 44 | 6 | 2 | 11.4259 | 25.5 11.4 15.2 18.5 13.8 24.8 |
| B' | 4 | 44 | 5 | 1 | 17.2907 | 17.3 24.6 19.4 25.2 30.1 |

**Saturation is not specific to the ten (T3).** Dev accuracy is exactly 0.5 at epoch 1 in 10/10 degenerate, 24/26
non-degenerate and 12/12 folds-run models (epoch 2: 9/10, 22/26, 12/12). In all 164 saturated epochs the train accuracy
is above 0.9, while the dev val_loss ranges from 2.2389 to 264.1550 (medians 25.1748, 17.9936, 49.2110), far above ln 2:
confident errors on half of the balanced dev. **Checkpoint (T10):** each of the ten puts every dev frame in one class
(attack in 8, bona fide in 2); six give 1-2 distinct scores (dev AUROC 0.5000 or 0.4975); B fold 1 seed 42 and fold 4
seed 44 still order the dev (0.9975, 0.9652), A\* fold 2 seed 44 and fold 3 seed 42 do not (0.2114, 0.4690).

**Recovery (T4, T5)** = first epoch with dev accuracy >= 0.75 and val_loss < 1.0. Degenerate 0/10 within their 5-6
epochs; non-degenerate 25/26, median epoch 6 (1-11); folds run 12/12, median 5.5 (3-7). Of the 37 non-degenerate models
of both runs that recover, 22 do so after epoch 5 (13 of the 16 of B and A\*); these are models that by construction
went beyond epoch 5. Over the nine variants of the criterion (0.70/0.75/0.80 x 0.7/1.0/1.5) the degenerate stay 0/10;
the non-degenerate range from 21 to 25 of 26.

**Mechanics of the stop (T6, T7).** The exact replay of the Keras 2.15 `EarlyStopping` reproduces stop and restored
epoch of 54/54 cards. The degenerate criterion selects by construction the models restored at epoch 1-2, so in the ten
the val_loss never falls below that of epoch 1 except at a restored epoch 2 (2 of 10): this part is definitional. What
the replay shows is that, with the recorded histories, the stop at epoch 5-6 follows from patience 4 counted from a
saturated "best" epoch, and that in many non-degenerate models the counter was reset with the dev saturated: the
val_loss falls below that of epoch 1 in 25 of 26 (12 of 12 in the folds run), and in 16 of the 26 (6 of the 12) this
first fall happens while the dev is still at 0.5. Same fold and seed: B fold 1 seed 42, saturated with val_loss 7.1,
11.1, 25.3, 70.4, 58.0, stops at epoch 5 and restores epoch 1; A\* fold 1 seed 42, saturated in epochs 1-3 with 18.6,
6.2, 5.6 (`curves.csv`), continues, recovers at epoch 6, restores epoch 10 (dev accuracy 0.9950). Replayed rules (T7):
patience 2 would make 5 of the 26 and 1 of the 12 degenerate, patience 3 3 of the 26 (a subset of the 5); patience 6 or
8 and `start_from_epoch` 3 or 5 change nothing where determinable (17, 17, 21, 20 of 26; 12/12) and are not determinable
for the ten; monitoring `val_accuracy` keeps the 6 determinable degenerate models degenerate and adds 4 of the 26.

**Learning rate (T8).** ReduceLROnPlateau halved the lr in 10/10 degenerate models (first reduced epoch 4 in 8, 5 in 2),
9 times on a decision taken with the dev at 0.5; also in 25/26 non-degenerate and 11/12 folds-run models. **Training
side (T9):** indistinguishable; train accuracy at epoch 1 median 0.9495 [0.9143-0.9807] degenerate against 0.9691
[0.9021-0.9886] non-degenerate, overlapping ranges from epoch 2. The failure is visible only on the dev. **Seed 44 and
fold 4 (T2):** seven of the ten are seed 44; fold 4 seed 44 is degenerate in all three arms, fold 2 seed 44 in A\* and
B'; the folds run has no degenerate model. **Last epochs (T6):** only B fold 4 seed 44 rises clearly (0.515 to 0.815,
val_loss 4.6); A\* fold 2 seed 44 reached 0.598 at epoch 4, then 0.280. **Determinism (T11):** the six histories of the
failed attempt equal those of the run (largest difference 0.0000; all fold 1), while `weights_sha256` differs in 6/6; we
hypothesise that the `.keras` file also holds metadata (date, names); no model was loaded to check it. **Boundary
(T1):** B fold 3 seed 44 restores epoch 3 with dev accuracy 0.2375 and val_loss 3.273 and never recovers, a failure of
the same family that the criterion misses; four B' models restore epoch 1-3 with dev accuracy 0.7725-1.0000 and then
fall back to 0.5.

## Possible explanations (hypotheses, not tested)

1. *Moving statistics of the five BatchNormalization layers (momentum 0.99) not settled.* For: train near 1, dev one
   class with confident logits; after one epoch of 88 steps 0.4129 of the moving average still comes from the initial
   value (B' 0.5527, folds run 0.2654; T13). Against: B' recovers first with the fewest steps (median 177 steps; 528 for
   the non-degenerate B and A\*, 726 for the folds run); after epoch 3 the residual is 0.0704, yet many stay saturated.
2. *Dropout before BatchNormalization* (`bn2_1`, `bn_dense`): variance estimated with dropout, used without it. For: the
   gap is between the two modes only. Against: that shift would be stable, while 37 of the 38 non-degenerate models
   recover. Not separable from 1 with the curves.
3. *Augmentation on the fitting set only.* Against: the dev jumps within one epoch with the train unchanged (A\* fold 1
   seed 42: 0.9950 at epoch 10, 0.5350 at epoch 11); the folds run has the same augmentation and always recovers.
4. *Learning rate 1e-3 too high early.* For: val_loss jumps by orders of magnitude between epochs; B' relapses.
   Against: smooth train curves. Not testable without changing every trajectory.
5. *Early stopping counted from a saturated "best" epoch.* Partly definitional, since the criterion selects the models
   restored at epoch 1-2. What the replay adds (T6, T7): with the recorded histories the stop at epoch 5-6 follows from
   patience 4 counted from a saturated best epoch, and in 16 of the 26 non-degenerate models the counter was first reset
   with the dev saturated; in the saturated phase val_loss measures how confident the error is, not discrimination. It
   explains why the ten stop, not why the dev saturates.
6. *Small dev* (2-3 subjects, 400 frames per fold; T10): a noisy val_loss makes the stop more random, not the output
   constant. 7. *Initialisation (seed 44):* fits the concentration, but seed 44 never degenerates in the folds run.
Only 5 can be examined with the curves, in part by definition; 1-4, 6 and 7 cannot be told apart with them.

## The one change proposed

`early_stopping_patience` from 4 to 12 (= `max_epochs`): no early stop, every model runs 12 epochs. The checkpoint rule
is unchanged (epoch of minimum dev val_loss, applied with `set_weights` as already for the 17 models that reached epoch
12); the configuration of ReduceLROnPlateau, lr, batch, augmentation, class weights, network, data, dev, EER threshold,
test, seeds, order and runner checks is unchanged. Why: it acts on the stop mechanism that the replay reproduces
(hypothesis 5); 22 of 37 recoveries come after epoch 5 (from models that by construction went beyond epoch 5); shorter
patience gives more degenerate models. Consequence of this change alone, not a second change: with ReduceLROnPlateau as
configured (patience 2, factor 0.5, min 1e-6), a model without improvement keeps halving its lr every two epochs, down
to about 3e-5 (1e-3 / 32) at epoch 12, against 5e-4 at the last epoch of the ten today (T8). A recovery would therefore
not separate hypothesis 5 (the stop) from 4 (the lr): the lr history is among the secondary outcomes. Not chosen:
`start_from_epoch` (changes the checkpoint rule and the runner check too); BatchNormalization momentum (modifies the
reference architecture and every trajectory; not identified as the cause); lr 1e-4 (every trajectory changes,
convergence within 12 epochs ambiguous); monitoring `val_accuracy` (discarded by the replay); `max_epochs` 20 (the ten
stop at 5-6). If it works, the ten recover within 12 epochs. If not, some model stays saturated until epoch 12: the stop
is not the cause, and the next step would concern the cause (BatchNormalization or lr), to be agreed. Declared risk: the
change treats the stop, not the saturation; that the ten recover within 12 epochs is an expectation, not an observation
(their histories end at epoch 5-6).

## Verification protocol (proposal, to be agreed)

Same runner and configuration as run `20261106-SESSABLRUN-d999458` with only `early_stopping_patience` = 12 (new
configuration anchored by SHA-256; the difference declared and checked by a test that every other field is equal); same
36 models (3 arms x 4 folds x seeds 42/43/44), same order, same cap of 360 minutes. Reading rule on train/dev, fixed
before the run: (a) determinism: for every model the epochs shared with the previous run, its first `epochs_run` there
(5-6 for the ten, 5-11 for the 9 non-degenerate models stopped before epoch 12, 12 for the 17 complete ones), must have
identical histories (all five fields, difference 0, as observed on 6/6 in T11); otherwise the largest difference is
reported and the comparison is declared not valid. With (a) valid no new degenerate model can appear, by construction:
the first epochs are identical and a later minimum cannot fall at epoch 1-2. (b) Primary: over the 36 models, the number
of degenerate models and the number without recovery within 12 epochs (same criteria), expected 0 and 0; positive if all
ten recover within 12 epochs and none of the 36 is degenerate; negative if at least one of the ten is still degenerate
or without recovery; any other model without recovery (today B fold 3 seed 44) is reported with the secondary outcomes
and does not decide the verdict. (c) Secondary: epoch of first recovery, restored epoch, best val_loss, dev accuracy and
dev AUROC at the checkpoint, epochs at 0.5, lr per epoch. (d) The test is computed and sealed by the runner as before,
with the same reading rule for A\* - B, B' - B and A\* - B', reported in a separate section after the train/dev outcome;
the change is neither chosen nor judged on it. The change also lets the 9 non-degenerate models stopped before epoch 12
continue, so their checkpoints, and the primary test comparison A\* - B, may change. (e) If the run is stopped by the
cap or is incomplete, the verdict is not given; what is complete is reported, as in the previous run.

## Time estimate (T12)

Seconds per epoch (weighted): B 32.4537, A\* 31.8532, B' 21.2375; per model from -18 % to +15 % of the arm median. All
36 x 12 epochs: 12,318 s of training, 13,090 s (3.64 h) with checks, scoring and aggregate (21.4 s per model as in the
run), within the cap of 21,600 s. Only A\* and B (24): 9,774 s (2.72 h). Only the 19 stopped before epoch 12: 6,694 s
(1.86 h), relying on determinism verified on six models. Proposed: all 36, which checks determinism on every model.

## Figures

`figure_examples.png` (train/dev accuracy and loss per epoch, restored and last epoch marked): (1) B, (2) A\* fold 1
seed 42: the mechanics of the stop; (3) A\* fold 2 seed 44, degenerate, inverted ordering; (4) B, same fold and seed,
recovered at epoch 6; (5) folds run fold 4 seed 44, recovered at epoch 3 (fold and seed of three degenerate models; same
held-out subjects, but a dev of 4 subjects and 600 frames instead of 3 and 400); (6) B' fold 4 seed 43, dev 1.0 at epoch
3, then 0.5 again. `figure_val_*_all.png`: dev curves of all 48, the ten shaded.

## Limits

Criteria ours and post hoc (the degenerate one misses B fold 3 seed 44); replayed rules exact only where the history
covers them (nothing known about the ten beyond epoch 5-6); determinism on six models; train and dev in different modes;
the dev (2-3 subjects) also selects checkpoint and threshold, and with patience 12 it chooses the checkpoint among 12
epochs instead of 5-12, a more optimistic selection on the dev (reuse of the dev, already declared); folds run a
reference, not a paired control, and with patience 12 the recipe is no longer identical to the folds-run one (patience
4).

## Pre-commitment

No training before the coordinator agrees on the change and the protocol. Criteria used here: defined after the run,
declared as such; the reading rule above: fixed before the verification run. No test data were read for this note.
