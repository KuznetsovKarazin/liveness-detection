# Constant learning-rate control: duration or learning-rate reduction (protocol fixed before the run)

**Outcome** [added 11 Nov, after the run]. The control run `20261111-SESSABLRUN-8568ffa` ended PASS, 36 of 36 models,
no discrepancy, within the cap (wall 15,217.0 s against the estimate of 17,940.9 s). The comparison
`20261111-LRCTRL-ff53279`: (a) shared histories identical in 36 of 36 models and rate 0.001 at every epoch, the
comparison is valid; (b) the ten recover, none of the 36 is degenerate, one recovery of the reference run is lost
outside the ten (B' fold 2 seed 42): a mixed outcome, reported model by model, as this rule prescribes, with no
overall verdict. The "no overall verdict" is the wording of `reading_rule.b` of the configuration of the comparison, fixed before the run
together with this note. The reading is in `CONCLUSION.md`. Below, only the sentence on what this note reports was
adapted to this paragraph; nothing else was changed after the run.

Protocol of the control proposed in `delivery/synthesis/SYNTHESIS.md`, section 5, and agreed by the coordinator on
10 November 2026 (C2 card; the comment is quoted in `amendment.authorisation` of the configuration). Fixed before the
run. No training, inference or test score was run or read for the protocol below, which reports no outcome; the
outcome is only in the paragraph above and in `CONCLUSION.md`.

Configuration: `delivery/constlr/nuaa_session_ablation_run_constant_lr.json`, pinned in the runner
(`CONSTANT_LR_CONFIG_SHA256`); parent: the patience-12 configuration
`delivery/patienceverify/nuaa_session_ablation_run_patience12.json` (`ca039df5…`), whose run is the reference
`20261109-SESSABLRUN-6a50265`.

## Question

In the patience-12 setting the ten original degenerate trainings recovered within 12 epochs, and none of the 36 models
was degenerate. In that run ReduceLROnPlateau halved the learning rate in 35 of the 36 models (first lower rate recorded
between epoch 4 and epoch 12; A\* fold 1 seed 42 was never reduced), so the recovery does not separate the duration of
training from the learning-rate reduction. Question: do the ten recover because training lasts 12 epochs, or does the
reduction contribute? The control changes one thing with respect to the reference run and nothing else; no extra seed is
added.

## What stays fixed (equal to the reference run)

- The same 36 models (arms B, A\* and B' x 4 folds x seeds 42, 43, 44) in the same order.
- The same fitting frames, common dev, test frames, folds and seeds; the same cells run and folds run as inputs.
- LivenessNet as in the frozen `architectures.py`; augmentation on the fitting frames only; batch 16; Adam (legacy),
  learning rate 0.001, clipnorm 1.0; balanced class weights.
- `early_stopping_patience` 12, at most 12 epochs; checkpoint = the epoch of minimum dev `val_loss`.
- EER threshold on the dev only, frozen before any test frame is scored; 0.5 also reported.
- CPU only; overall cap of 360 minutes.

All 36 models are trained, not only the ten, because the primary reading requires that none of the 36 be degenerate and
the validity check applies to every model.

## The one change

`training.scheduler` goes from `ReduceLROnPlateau(monitor=val_loss, factor=0.5, patience=2, min_lr=1e-6)` to the
constant learning rate: no ReduceLROnPlateau, so the rate stays at `initial_learning_rate` 0.001 for every epoch of
every model. Section 5 of the synthesis wrote that the control "sets that field to `none`"; the value written here is
the string `constant (no ReduceLROnPlateau; the initial learning rate for every epoch; the learning rate in effect is
recorded in history.lr at the end of every epoch)`, which says what replaces the schedule; the meaning is the one of
section 5, and the trainer accepts only this string or the recipe string. Every other field of the configuration is byte-for-byte the parent's (same key order and serialisation);
only `training.scheduler` and the `amendment` block differ. The descriptive texts of the parent (task, budget, previous
attempt and so on) are kept unchanged, as the amendment mechanism requires; where they describe ReduceLROnPlateau or the
patience-12 verification, the `amendment` block prevails.

## Implementation (checked before the run)

- Trainer (`scripts/nuaa_folds_run.py`). The field `training.scheduler`, until now only descriptive, is read by the
  trainer. Two values only: the recipe string (`SCHEDULER_REDUCE`) and the constant one (`SCHEDULER_CONSTANT`); any
  other value is refused before TensorFlow is imported. The callbacks are built by one function: with the recipe
  string they are, as before, EarlyStopping (patience of the configuration, `restore_best_weights`), ReduceLROnPlateau
  (factor 0.5, patience 2, min_lr 1e-6) and the cap check, in this order; with the constant string the second one is
  replaced by a callback that does not change the rate and, at the end of every epoch, writes into the epoch logs the
  rate in effect in the optimizer, with the same expression and at the same point as ReduceLROnPlateau. Keras 2.15
  ReduceLROnPlateau writes `lr` at the start of its end-of-epoch step, before any reduction, so in both modes `history.lr`
  holds the rate with which each epoch was trained. The frozen recipe of the folds run (`FROZEN_TRAINING`) is unchanged;
  the folds run and the two earlier configurations keep ReduceLROnPlateau.
- Training card. It records `scheduler`; it is refused if the scheduler differs from the configuration, if `history.lr`
  does not hold one value per epoch, or, with the constant scheduler, if a recorded rate is not 0.001 (compared in
  float32, as the optimizer holds it).
- Runner (`scripts/nuaa_session_ablation_run.py`). The amendment check, until now tied to the patience, reads a table of
  known amended configurations; each entry gives the configuration, its pinned SHA-256, its name, the pinned SHA-256 and
  path of its parent, the changed field and its "from" and "to" values. Entries: `patience12_verification` (parent: the
  delivered configuration; `training.early_stopping_patience` 4 → 12) and `constant_lr_control` (parent: the
  patience-12 configuration; `training.scheduler` ReduceLROnPlateau → constant). Before any data is read, the runner
  refuses the configuration unless: its SHA-256 is the pinned one; the declared parent is the pinned parent; the parent
  read from the repository has that SHA-256; every field other than the changed one and `amendment` equals the parent
  (strict types); the parent holds the "from" value and the configuration the "to" value; `training` is the frozen
  recipe with only the cumulated changes (patience 12 and the constant rate); the `amendment` block has exactly the
  agreed keys, with five agreed conditions. `status.json`, `run_record.json`, `aggregate.json` and `aggregate.md` record
  `configuration_name` = `constant_lr_control` and the amendment (parent, field, from, to).
- Shared code. The runner requires the trainer used by the folds run; it now also accepts one pinned revision of
  `nuaa_folds_run.py` (`TRAINER_REVISION_SHA256`), the one with the operative scheduler, and records which of the two
  was used.
- Tests. The suites of the trainer and of the runner check the two scheduler values and the refusals, the callbacks in
  both modes with their parameters and order, the training card, the unchanged frozen recipe, the new configuration
  (pinned SHA-256, parent, only the allowed difference, refusals) and the recording of `configuration_name` in the
  outputs; each new check has a declared mutation, killed by the suites.

## Reading rule (fixed before the run)

Same criteria as the earlier notes, defined after the parent run and declared as such: degenerate = restored epoch 1 or
2 with dev accuracy 0.5 (within 1e-6) at that epoch; recovery = first epoch with dev accuracy ≥ 0.75 and dev `val_loss`
< 1.0.

(a) Validity. For every model, the history must be identical, in all five fields (`loss`, `accuracy`, `val_loss`,
`val_accuracy`, `lr`), to the reference run over the epochs recorded there at 0.001, that is before the first
reduction. Keras records the rate of an epoch before reducing it, so the epoch at whose end the first reduction happens
was trained at 0.001 and belongs to the window: epochs 1 to 3 when the first lower rate is recorded at epoch 4; all 12
epochs for A\* fold 1 seed 42, never reduced. Otherwise the largest difference is reported and the comparison is not
valid. The rule relies on the determinism already observed (36 of 36 shared histories identical between the patience-4
and patience-12 runs, largest difference 0.0). The comparison is made after the run by a separate script, not by the
runner, and before any other reading.

(b) Primary. Among the ten original degenerate trainings, the number that recover within 12 epochs with the constant
rate; among the 36, the number degenerate. If all ten recover and none of the 36 is degenerate, the recovery of the ten
is reproduced without the reduction: consistent with the extra epochs being sufficient in these 36 trainings. If at
least one of the ten does not recover, the outcome is consistent with a contribution of the reduction in these
trainings, and the models are listed one by one; one failure among the ten is enough for this second reading. A mixed
outcome (for example all ten recover but another model becomes degenerate or loses its recovery) is reported as such,
model by model.

(c) Secondary, for every model: first-recovery epoch against the reference run; whether the recovery holds at epoch 12;
restored epoch; dev accuracy and dev AUROC at the checkpoint; epochs at 0.5. B fold 3 seed 44 (recovered at epoch 10
with patience 12) and B' fold 4 seed 42 (the relapse: epochs 5-6 already at 0.5 with patience 4, shared history)
reported apart.

(d) Test. Computed and sealed by the runner as before, with the same comparisons and reading rule, reported in a
separate section after the train/dev outcome; never used to choose or to judge.

(e) Time, against the estimate. A run stopped by the cap, or incomplete, gets no verdict.

## Budget and stop conditions

36 trainings x 12 epochs, the same as the reference run, whose wall time was 17,940.9 s (training 16,866.2 s) on a
machine not dedicated to the run. Estimate 17,940.9 s; cap 21,600 s (360 minutes); margin 3,659.1 s, about 20 % of the
estimate, thinner than the +37 % by which the reference run exceeded its own estimate. The run is launched alone, as the
only heavy process.

As the coordinator asked: every outcome is kept. If a check fails (including a discrepancy in the shared histories) or
the cap is exceeded, we stop and report the point reached, with no further attempt and no conclusion on an incomplete
run. Once the checks above pass, the run starts without a further go-ahead.

## Declared limits

- A constant rate tests the absence of the reduction, not the level of the rate: 0.001 is never varied.
- With a different rate path the checkpoint chosen by the minimum dev `val_loss` can change, so dev and test values at
  the checkpoint can differ for reasons other than the recovery.
- The dev is small (2-3 subjects, 400 frames per fold) and chooses the checkpoint among 12 epochs.
- The reduction is removed in all 36 models, including the 16 whose first recovery already came at 0.001 with patience
  12.
- The criteria were defined after the parent run.
- One dataset, one attack type, three seeds per fold.

## What it does not decide

- The quality on the test (the test is sealed and not used to choose).
- The cause of the early dev saturation (BatchNormalization statistics, dropout, augmentation, level of the learning
  rate, small dev, initialisation stay untested hypotheses).
- Whether a lower constant rate would also work.
- Whether the reduction alone, without the extra epochs, would be enough.
- The session-03 question of the ablation.

## Commands

```
.venv/bin/python -I scripts/nuaa_session_ablation_run.py check-inputs --config delivery/constlr/nuaa_session_ablation_run_constant_lr.json
.venv/bin/python -I scripts/nuaa_session_ablation_run.py run --i-am-authorized --config delivery/constlr/nuaa_session_ablation_run_constant_lr.json
```
