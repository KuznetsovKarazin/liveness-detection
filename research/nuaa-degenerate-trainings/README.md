# NUAA: the degenerate trainings of the session ablation run, from the recorded train/dev curves (no training)

Study requested by the coordinator on the C2 card on 6 November 2026: look into the 10 degenerate trainings of the
session ablation run `20261106-SESSABLRUN-d999458` using the train/dev curves already available, compare them with
trainings that are not degenerate by the criterion
(not all of which recover) and show a few representative examples, keeping what we observe apart from the possible
explanations. This folder holds the configuration written before the run (`nuaa_degenerate_trainings.json`, at the
default path of the script), this README and `NOTE.md` (the short note for the coordinator: observations, possible
explanations, the one proposed change of the recipe, its verification protocol and the time estimate); the code is
`scripts/nuaa_degenerate_trainings.py` with its tests (`scripts/check_nuaa_degenerate_trainings.py`) and mutation
check (`scripts/check_nuaa_degenerate_trainings_mutations.py`).

## What it does not do

No training, no inference, no model loaded, no image read, no test score or test metric read. The study reads the
training cards (per-epoch history), the dev scores of the checkpoint and the frozen-at time of the dev threshold of
runs already delivered, and reports observations. The possible explanations (BatchNormalization moving statistics,
dropout before BatchNormalization, augmentation, learning rate, the early stopping counted from a saturated epoch, a
small dev, the initialisation) are hypotheses discussed in the note; none of them is tested here. The criteria
(degenerate, saturated epoch, recovery) were defined after the run, from the cards. The script itself proposes no
recipe change; the change, its protocol and the time estimate are in `NOTE.md`, as a proposal to be agreed before any
training.

## Delivery status

- Run `20261107-DEGEN-a3b4dc0` from the clean commit `a3b4dc0` of the development repository (2026-11-07), outcome
  PASS: `public/` copied here as `tables/` (same bytes); `status.json` and `run.log` copied as `run/status.json` and
  `run/run.log.txt`; the public `SHA256SUMS` of the run has SHA-256
  `c591ef088257ec5be42c639f43829293f76e3e099d826213b093b579422e2776`.
- The configuration (`nuaa_degenerate_trainings.json`, SHA-256
  `8edda999f3f47f84e90ef32d4c1166db51d3fc9650747f58028cfdda7d6195e4`, fixed in the script as `CONFIG_SHA256`) was
  written and every input verified with `check-inputs` before the run.
- A first run (`20261107-DEGEN-22ba8ff`, commit `22ba8ff`, same numbers, same bytes of `curves.csv`, `models.csv` and
  the figures) preceded an independent review of 7 November 2026, after which the script gained row-by-row checks of
  the dev scores and thresholds, one check per condition on the cards, a public-scan reason without the identifier
  value and one number format in `aggregate.md`; the configuration changed only in `declarations.dev_columns_read`.
  The run above is the rerun from the corrected commit; the first run is kept in the development repository only.
- No restricted folder: the study writes no restricted output, so there is nothing to upload to Drive for this run.
  The cards and dev scores it reads stay in the restricted folders of the runs already delivered.

## Inputs (read-only, fixed by SHA-256 in the configuration)

| input | read | verification |
|---|---|---|
| ablation run `20261106-SESSABLRUN-d999458` (36 models) | `restricted/<arm>_fold<k>_seed<s>_card.json` (history per epoch); `_dev_scores.csv` (`label`, `score_attack` for the measures; `subject_id` only to count the subjects of each dev and to scan the public files, never written; `arm`, `fold`, `train_seed`, `role`, `checkpoint_sha256`, `sample_id`, `file_sha256`, `prediction_05`, `prediction_dev`, `threshold_dev`, `threshold_sha256` only for row-by-row checks); `_threshold.json` (`frozen_at` for the times; `arm`, `fold`, `train_seed`, `checkpoint_sha256`, `value`, `dev_scores_sha256` for consistency); the card fields `saved` and `test_scored_after_threshold` for the times; `run.log` (the wall-time line only) | `status.json` (PASS) fixed by SHA-256; public and restricted `SHA256SUMS` and `run.log` equal to the anchors of `status.json` and of the configuration; every file read equal to its line of the restricted `SHA256SUMS`; exactly the 36 expected cards listed |
| folds run `20261103-FOLDSRUN-e174812` (12 models) | `restricted/fold<k>_seed<s>_card.json`, `_dev_scores.csv`, `_threshold.json` (same columns and fields; the threshold times are checked, not used) | same anchors |
| failed attempt `20261106-SESSABLRUN-3b4021f` (FAIL folder, six complete models) | the six `_card.json`, for the determinism comparison only | `status.json` (FAIL) fixed by SHA-256; restricted `SHA256SUMS` anchored; each card fixed one by one in the configuration and equal to its `SHA256SUMS` line; the six names equal to `kept_complete_models` and to models of the ablation run |

Checks on every card: keys of a closed set (per run); recipe equal to the configuration (`max_epochs` 12, patience 4,
batch 16, lr 0.001); a history with exactly `loss`, `accuracy`, `val_loss`, `val_accuracy`, `lr`, each of length
`epochs_run`, finite values in range; `restored_epoch` = the first epoch of minimum `val_loss`; `best_val_loss` and
`val_acc_at_best` read from the history at that epoch; the exact replay of the Keras 2.15 `EarlyStopping`
(`min_delta` 0, strict improvement, `restore_best_weights`, as in `keras/src/callbacks.py`) reproduces `epochs_run`,
`restored_epoch` and `restored_by` (a stop at the last epoch counts as early stopping, as Keras does); `weights_file`
and `weights_sha256` listed in the `SHA256SUMS` (the weights are not read). Checks on every threshold file:
arm, fold, seed and checkpoint of the card; `value` a finite number; `frozen_at` between the saved weights and the
test scores. Checks on every dev scores file, row by row: role `dev`, arm, fold, seed and checkpoint of the card;
label in {0, 1}; score in [0, 1]; `threshold_dev` and `threshold_sha256` equal to the frozen threshold; `prediction_05`
and `prediction_dev` following "attack if score > threshold, a tie is bona fide" (0.5 and the frozen value); then
`sample_id` and `file_sha256` unique; the list of `sample_id` equal to `dev_keys_sha256` of the card and the scores
equal to `dev_scores_sha256` of the threshold (the SHA-256 of the values joined by a newline, the formulas of the
runners); rows = `n_dev` of the configuration (400 or 600) and of the card; balanced; accuracy at 0.5 equal to
`val_acc_at_best` within 1e-6; the same rows (key, file, label, subject, in the same order) for all the models of a
fold. Any failed check stops the
command with an explicit reason; nothing is defaulted.

## Criteria (fixed in the configuration)

- Degenerate: `restored_epoch` in {1, 2} and `val_acc_at_best` = 0.5 (within 1e-6), the rule of the ablation run delivery.
- Saturated epoch: `val_accuracy` = 0.5 within 1e-6 (same value and tolerance, one definition).
- Recovery: first epoch with `val_accuracy` >= 0.75 and `val_loss` < 1.0 (dev only); sensitivity grid 0.70/0.75/0.80
  × 0.7/1.0/1.5.
- Alternative early stopping rules replayed on the recorded histories: patience 2, 3, 6, 8; `start_from_epoch` 3, 5;
  monitor `val_accuracy`. The replay is exact when the rule stops within the recorded epochs or the history reaches 12
  epochs (the weights do not depend on the early stopping until it stops); otherwise "not determinable".
- Determinism: the six histories of the failed attempt against the same models of the run. A difference is reported
  (`identical_history` false, largest difference), not raised as an error: it is an observation of the study.

## Outputs

`results/degenerate/<run-id>/` (`YYYYMMDD-DEGEN-<commit>`, `-dev` if the tree is modified): `status.json` (RUNNING,
then PASS or FAIL; it states that there is no restricted output), `run.log`, and `public/`:

- `curves.csv`: one row per model and epoch (533 rows: 36 + 12 + 6 models), with `run`, `arm`, `fold`, `seed`,
  `n_fitting`, `steps_per_epoch`, `epoch`, `loss`, `accuracy`, `val_loss`, `val_accuracy`, `lr`, `is_best`,
  `degenerate`, `run_id`, `stopped_epoch` (= `epochs_run`, the last epoch run);
- `models.csv`: one row per model (54), with the restored epoch, the first recovery, the epochs at 0.5, the first epoch
  with `val_loss` below epoch 1, the last values, the seconds per epoch and `run_id`;
- `aggregate.json` and `aggregate.md`: tables T1–T13 (T1 criterion and the 10 degenerate models with the boundary
  cases; T2 fold × seed per arm; T3 saturation and the training side in the same epochs; T4/T5 recovery and its
  sensitivity; T6/T7 mechanics of the stop and the alternative rules; T8 learning rate; T9 training side and pairs;
  T10 the checkpoint on the dev, including constant outputs; T11 determinism; T12 seconds per epoch and time
  scenarios; T13 residual of the BatchNormalization moving averages, 0.99^steps);
- `figure_examples.png` (the six examples of the configuration: train and dev accuracy and loss per epoch, loss on a
  log scale, restored and last epoch marked), `figure_val_accuracy_all.png` and `figure_val_loss_all.png` (small
  multiples of the 36 + 12 models, degenerate ones highlighted, restored epoch marked); written only when matplotlib
  is available, otherwise skipped with a declared note in `aggregate` and `status`;
- `SHA256SUMS`.

The public files hold no date and no run time (only `status.json` and `run.log` do): two runs with the same run id
give byte-identical public files. Before writing, every public text and every figure label is scanned: image keys,
the subject identifiers read from the dev scores (as whole words, so that digits inside a number are not a match),
absolute paths, credentials, non-local addresses, unfinished-work markers and, in the development repository, the
local operator, host and tool patterns; any hit stops the run with no public output. The run folder is validated
for identity before any change (never a protected folder, never inside an input, no foreign entry); previous outputs
are removed right after the admission, and on any error only `status.json = FAIL` and `run.log` remain. A failure
reason names the category and the file (for example `public file aggregate.json refused: subject id`), never the
value found. A run folder that holds an entry this command does not write (for example a file added by hand to
`public/` after a PASS) is not ours: the command refuses it before any change, prints the reason and touches
nothing, so the previous `status.json` (PASS included) and `public/` stay as they were. This is the behaviour of
`m2_writers.output_problem` and of the run-folder check shared with the accepted runners (step zero, folds
comparison); archive or clean that folder by hand, or run into another one.

## Commands

From the development repository, with the run folders present:

```bash
.venv/bin/python -I scripts/nuaa_degenerate_trainings.py check-inputs
.venv/bin/python -I scripts/nuaa_degenerate_trainings.py run --i-am-authorized
```

## Verification without data

Tests (synthetic world in temporary folders; no data needed) and mutation check:

```bash
.venv/bin/python -I scripts/check_nuaa_degenerate_trainings.py            # add --real to also run the read-only test on the real inputs
.venv/bin/python -I scripts/check_nuaa_degenerate_trainings_mutations.py  # --jobs 1 by default, at most 3
```

Expected lines. Without `--real`: `skipped __main__.TestRealInputs.test_real_inputs_read_only: missing input: the
read-only test on the real inputs runs only with --real`, `tests run: 90, failures: 0, errors: 0, skipped: 1` and
`RESULT: PASS`. With `--real` and the run folders present: `tests run: 90, failures: 0, errors: 0, skipped: 0`.
Mutation check: `mutations killed: 99/99`, `control mutations survived: 2/2 (1 control, 1 declared redundant)` and
`RESULT: PASS`. The mutants never read the real inputs (no `--real`, input root on an empty folder).

Dependencies: Python 3.11 or later (verified on 3.11 and 3.14), standard library only; matplotlib optional (figures); `scripts/m2_writers.py`,
`scripts/path_guard.py`, `scripts/package_rules.py`. No NumPy, no TensorFlow, no Keras (the early stopping rule is
replayed in plain Python and the tests check it against hand-computed cases).

## Limits

The degenerate criterion was defined after the run and from the cards only; it misses at least one failure of the
same family (`arm_b` fold 3 seed 44, restored epoch 3 with dev accuracy 0.2375) and does not count the `arm_b_prime`
models that fell back to 0.5 after a good start. The recovery criterion is ours; the sensitivity grid is reported.
The alternative rules are exact only where the recorded history covers them: for the 10 degenerate models nothing is
known beyond epoch 5–6. Determinism is compared on six models only (all fold 1), same machine and code. Train and dev
accuracy are measured in different modes (training mode with augmentation, dropout and batch statistics; inference
mode on the dev): comparing them directly is indicative only. The folds run is a reference with other data, another
dev (600 frames, 4 subjects) and more steps per epoch, not a paired control. All dev metrics are also those that
select the checkpoint and the threshold (reuse of the dev, already declared). Observations, not causes.
