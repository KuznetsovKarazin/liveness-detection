# Verification run of the agreed change: `early_stopping_patience` 4 -> 12, same 36 LivenessNet models

Verification run of the one recipe change proposed in `delivery/degenerate/NOTE.md` (PR #22, merged) and agreed by
the coordinator on the C2 card on 7 November 2026: `early_stopping_patience` from 4 to 12, at most 12 epochs, the same
36 models (arms B, A\* and B' x 4 folds x seeds 42/43/44) and every other setting unchanged; the checkpoint of minimum
dev val_loss and the overall cap of 360 minutes kept. The agreed protocol, with the two precisions asked by the
coordinator, is in `NOTE.md` (the amended note: section "Verification protocol (agreed on 7 November 2026)" and the
conclusion of "The one change proposed", passages marked [amended 9 Nov]).

Status: **verification run `20261109-SESSABLRUN-6a50265` complete (PASS, 36 of 36, 9 November 2026); comparison of the
two runs `20261109-PATVER-4f2b925` complete: shared histories identical 36/36, verdict positive.** The reading of the
outcomes, with its cautions and limits, is in `CONCLUSION.md`. This folder holds the configuration, the amended note,
the conclusion, the configuration of the comparison and the logs of the checks (sections "Run" and "Comparison of the
two runs"); the package layout of the code repository is at the end.

## Configuration and declared difference

`nuaa_session_ablation_run_patience12.json`, SHA-256 `ca039df5d3114f9cb983c7256330aae671de54ae34d949ad4e9b9c80beb6aa43`
(pinned in the runner as `PATIENCE12_CONFIG_SHA256` before the run).

Parent: `delivery/sessionablationrun/nuaa_session_ablation_run.json`, SHA-256
`4c2d29d13dede8a48c3a8e74e1ca48379c5c7e7721eb0a1f55fececbc684e23b` (the configuration of run
`20261106-SESSABLRUN-d999458`, `CONFIG_SHA256` in the runner). The new configuration is the parent with exactly two
differences:

- `training.early_stopping_patience`: 4 -> 12;
- one more top-level key, `amendment` (last, after `outputs`): `parent_config_sha256`, `changed_field`, `from`, `to`,
  `authorisation` (the coordinator's comment of 7 November 2026, quoted in Italian, as `authorisation` quotes the one of
  5 November), `agreed_precisions` (the two precisions), `note`, `time_estimate`, `shared_history_check`.

Every other field is identical, `task`, `authorisation` and `budget.estimate` included; same key order and indentation
(`diff` of the two files: one changed line and the added `amendment` block).

## Checks

The runner `scripts/nuaa_session_ablation_run.py` is the same as for the parent run, with these additions:

- `load_config` accepts the `amendment` key as the only extra key; with it, `training` must be the recipe of the folds
  run with only `early_stopping_patience` = 12 (without it, the unchanged recipe as before), and `amendment` must have
  exactly its nine keys, `changed_field` = `training.early_stopping_patience`, `from` 4 and `to` 12 (integers), a
  well-formed parent SHA-256, non-empty texts and two agreed precisions.
- `check-inputs` and `run`, before reading any data, verify that the SHA-256 of an amended configuration is
  `PATIENCE12_CONFIG_SHA256`, that `amendment.parent_config_sha256` is `CONFIG_SHA256`, that the parent configuration
  read from the repository has that SHA-256, and that the two are equal in every field except
  `training.early_stopping_patience` (4 in the parent, 12 here) and `amendment` (`config_difference`, a pure function:
  strict types, missing and extra keys, nested values and list lengths, first differing field named with a dotted
  path); otherwise FAIL with that field in the message, nothing written.
- `status.json`, `run_record.json` and `aggregate.json` record `configuration_name` (`session_ablation_run` for the
  parent, `patience12_verification` here, null for an unknown configuration) and `delivered_configuration` (true for
  both known configurations, false otherwise); with the amended configuration they also record `amendment` (parent
  SHA-256, field, 4 -> 12), and `aggregate.md` states it next to the configuration SHA-256.
- The trainer reads the patience from the configuration: `train_one` passes `cfg["training"]` to
  `nuaa_folds_run.default_trainer`, which sets
  `EarlyStopping(monitor="val_loss", patience=training["early_stopping_patience"], restore_best_weights=True)`; the
  card records the patience and the runner refuses a card whose patience differs from the configuration.
  With patience 12 = `max_epochs` early stopping cannot trigger: every model runs 12 epochs and the
  weights of the epoch of minimum dev val_loss are restored with `set_weights`, as already for the models that reached
  epoch 12 in the parent run.
- Unchanged: run id `YYYYMMDD-SESSABLRUN-<commit>[-dev]`, folder `results/sessionablationrun/`, input checks, order,
  threshold, scoring, aggregate, reading rule, cap and outcomes. The parent configuration passes as before (same
  `check-inputs` output; its `status.json` and public files only gain `configuration_name`).

Tests (`scripts/check_nuaa_session_ablation_run.py`): 99 (81 + 18 new: the pinned SHA-256 of the amended configuration
and its parent, the real files differing only in the allowed field and key with the same key order, `config_difference`
on every kind of difference, refusals by `check-inputs` and `run` for an unpinned SHA-256, a different declared parent,
a parent file with another SHA-256 or missing, a difference in another field, a parent patience other than 4 and a
patience 12.0 instead of 12, the amendment check made before any input is read, the closed `amendment` fields (extra or
missing keys, null or non-object, upper-case parent SHA-256, precisions not a list of exactly two), `configuration_name`
and `delivered_configuration` in the outputs for the two configurations and an unknown one, the authorisation message,
the patience received by the trainer and set in `EarlyStopping`). Mutation check
(`scripts/check_nuaa_session_ablation_run_mutations.py`): 210 declared mutations (158, two of them with the pattern
updated to the changed lines, + 52 on the new code), a control and three declared redundant; one mutant at a time
(`--jobs 1`), never on the real data. The copy used by the mutation check also holds this configuration, so its two file
tests run in every mutant.

Logs in `test_logs/` (each opens with the SHA-256 of the scripts and of the configurations):

| Log | Content |
|---|---|
| `check_nuaa_session_ablation_run.txt` | the runner suite on macOS with the real inputs and both configurations: `tests run: 99, failures: 0, errors: 0, skipped: 0`, `RESULT: PASS` |
| `check_inputs_patience12.txt` | `check-inputs` on the real inputs with the amended configuration and with the parent: both `RESULT: PASS (check-inputs)` with the same input line (12,614 originals, 6,000 cache frames, 6,614 extracted frames, 16 cell manifests); the amended one adds the `AMENDMENT` line |
| `check_nuaa_session_ablation_run_mutations.txt` | the runner mutation check, `--jobs 1`, every copy without data: `mutations killed: 210/210`, `control mutations survived: 4/4 (1 control, 3 declared redundant)`, `RESULT: PASS` |
| `check_nuaa_patience_verify.txt` | the comparison suite, synthetic: `tests run: 135, failures: 0, errors: 0, skipped: 1` (the read-only test on the real inputs, without `--real`), `RESULT: PASS` |
| `check_nuaa_patience_verify_real.txt` | the comparison suite with `--real`, the real inputs read only: `tests run: 135, failures: 0, errors: 0, skipped: 0`, `RESULT: PASS` |
| `check_nuaa_patience_verify_mutations.txt` | the comparison mutation check, `--jobs 2` (two mutants at a time), `PATVER_INPUT_ROOT` on an empty folder: `mutations killed: 263/263`, `control mutations survived: 13/13` (1 control, 12 declared redundant), `RESULT: PASS` |
| `check_inputs_patience_verify.txt` | `check-inputs` of the comparison on the real inputs: parent and verification run, cards, dev scores and thresholds, Keras `EarlyStopping` replay 36/36 in each run, `RESULT: PASS (check-inputs)` |
| `linux_package_no_data.txt` | the two suites, the two `check-inputs` and the two mutation checks (`--jobs 2`) from the package layout in a Linux container (`python:3.12`, NumPy only, no OpenCV, no matplotlib, no TensorFlow), without data: runner suite `tests run: 99, failures: 0, errors: 0, skipped: 1` (the check on the real inputs), `RESULT: PASS`; runner `check-inputs` with the amended configuration `RESULT: FAIL (check-inputs)` at the first missing manifest, after the amendment check; comparison suite `tests run: 135, failures: 0, errors: 0, skipped: 2` (the configuration of the previous package, the `--real` test), `RESULT: PASS`; comparison `check-inputs` `RESULT: FAIL (check-inputs)` at the missing parent run; runner mutations `210/210` killed, `4/4` control survived, `RESULT: PASS`; comparison mutations `263/263` killed, `13/13` control survived, `RESULT: PASS` |

## Run

Command, from the development repository root with the data present, after the configuration and the runner were
committed (clean tree):

```bash
.venv/bin/python -I scripts/nuaa_session_ablation_run.py check-inputs --config delivery/patienceverify/nuaa_session_ablation_run_patience12.json
nohup caffeinate -dims .venv/bin/python -I scripts/nuaa_session_ablation_run.py run --i-am-authorized --config delivery/patienceverify/nuaa_session_ablation_run_patience12.json > logs/patience12-console.txt 2>&1 &
```

- Run `20261109-SESSABLRUN-6a50265` (clean commit `6a50265` of the development repository, `dirty: false`; delivered
  configuration `ca039df5d311`, `configuration_name` `patience12_verification`, amendment 4 -> 12 with every other
  field equal to the parent `4c2d29d13ded`), started 2026-11-09 14:07:17+0100, finished 19:06:18+0100: `PASS`, 36 of
  36 trainings complete, none missing, not stopped.
- Times: wall 17,940.9 s (about 299 of the 360 minutes; estimate 13,090 s, +4,850.9 s, +37 %), of which training
  16,866.2 s (every model 12 epochs; seconds per epoch, weighted: B 42.6097, A\* 40.0181, B' 34.4986, against 32.4537,
  31.8532 and 21.2375 in the parent run); aggregate 2.3 s. The Mac was not dedicated to the run (other processes active,
  not measured), so the seconds per epoch are not a measure of the recipe. Cap not reached; no write after the cap
  (`final_writes_after_cap: false`).
- Peak resident memory 5,121 MB (1,371 MB after the check-inputs made by `run` at the start, as in `run.log`).
- Primary verdicts printed by the runner (A\* - B, stratum 03), both `inconclusive`: they belong to the test, which is
  reported in the separate section (d) of the comparison and is not used to judge the change.
- Where the outcomes are: in the run folder, `status.json` (SHA-256 `af6a1d1e1f7b...`), `run.log`, `public/`
  (`aggregate.json`, `aggregate.md`, `run_record.json` and `SHA256SUMS`, SHA-256 `018d5079917856b5...` recorded in
  `status.json`) and `restricted/` (36 x card, dev scores, test scores, threshold and weights, `per_subject.csv`,
  `public_SHA256SUMS.sha256`: 182 files listed in its `SHA256SUMS`, SHA-256 `6787bc2bfb280fc0...`). In the package of
  the code repository, `public/` is copied as `tables/`, `status.json` as `run/status.json` and `run.log` as
  `run/run.log.txt` (same bytes: `*.log` is ignored by git there). The whole run folder (public, restricted,
  `status.json`, `run.log`) goes to the shared N1 folder on Drive, as for the parent run; the restricted files only
  there.

## Comparison of the two runs

A separate script, not the runner, compares the training cards of the verification run with those of
`20261106-SESSABLRUN-d999458`, by the reading rule agreed on 7 November 2026 (`NOTE.md`, points (a)-(e)).

- Script: `scripts/nuaa_patience_verify.py` (`check-inputs`; `run --i-am-authorized`, which reads the restricted
  training cards, dev scores and frozen thresholds of the two runs; `--config`, `--input-root`, `--out-root`, default
  `results/patienceverify`, `--date`). Self-contained: the functions taken from the curves runner
  (`scripts/nuaa_degenerate_trainings.py`) are copied, not imported, and listed in its docstring. Standard library and
  the shared modules `scripts/m2_writers.py`, `scripts/path_guard.py`, `scripts/package_rules.py`; matplotlib optional
  for the figures (otherwise a declared skip); no NumPy, no TensorFlow.
- Configuration: `nuaa_patience_verify.json` (schema `nuaa-patience-verify-config/1`), pinned in the script as
  `CONFIG_SHA256`. It anchors the parent run by SHA-256 (`status.json`, public and restricted `SHA256SUMS`, the same
  values as `delivery/degenerate/nuaa_degenerate_trainings.json`), names the verification run (run id, folder,
  `config_sha256` `ca039df5...`, commit `6a50265`, `configuration_name` `patience12_verification`, amendment 4 -> 12),
  copies the criteria of the previous package with the same values (degenerate, saturation, recovery, checkpoint
  agreement), lists the ten original degenerate models and `models_without_recovery_parent` (B fold 3 seed 44), and
  states the reading rule (a)-(e), the time estimate (13,090 s, cap 21,600 s), the declarations and the outputs. The
  three anchors of the verification run were written in the configuration before the training run and filled after it
  from the files of the run folder: `status_sha256` `af6a1d1e1f7b3c90...`, `public_sums_sha256` `018d5079917856b5...`,
  `restricted_sums_sha256` `6787bc2bfb280fc0...`; the configuration with the anchors has SHA-256
  `24c3f75bab49ee9506245c4b922a92f26388c2f21a7ba9d97919c6bbb0f53f48` (`CONFIG_SHA256` in the script). With null anchors
  `run` refuses and `check-inputs` checks only the configuration and the parent run and says so; `public_sums_sha256`
  is `"none"` only for a FAIL run (no public folder); a FAIL run without complete models has nothing to compare and is
  refused.
- Checks (any failure: FAIL, nothing public): the anchors; the parent `PASS`; the verification run `PASS`, `INCOMPLETE`
  or `FAIL`, of the run id, commit, `config_sha256`, `configuration_name` and amendment of the configuration, from a
  clean tree, with complete and missing models that split the 36 and exactly the cards of the complete models in its
  restricted `SHA256SUMS`; every card as in the curves runner (closed keys, recipe with the patience of its run,
  history, restored epoch = argmin, exact replay of the Keras `EarlyStopping`, weights listed); thresholds and dev
  scores row by row; the same dev for every model of a fold and for the same model in the two runs; the same fitting
  set as the parent; the ten and the models without recovery of the configuration equal to what the criteria give on
  the parent cards; the public `aggregate.json` of each run listed in its public `SHA256SUMS`, of the same run,
  configuration and status, with the closed structure of the test section; a run folder that already holds
  `status.json` is refused. The `*_test_scores.csv` files are never opened.
- Outputs (`results/patienceverify/YYYYMMDD-PATVER-<commit>[-dev]/`: `status.json`, `run.log`, `public/`; no
  restricted folder): `comparison.csv` (one row per model: arm, fold, seed, one of the ten, status in the verification
  run; parent and new epochs run, restored epoch, degenerate, recovered, first recovery, best val_loss, dev accuracy and
  AUROC at the checkpoint, epochs at 0.5, last lr; shared epochs, identical, largest absolute difference per field),
  `curves.csv` (verification run, per epoch), `aggregate.json` and `aggregate.md` (sections (a)-(e) in this order and
  the declarations), `figure_ten.png`, `figure_no_recovery.png`, `figure_val_accuracy_all.png` (parent dashed,
  verification run solid, val_loss on a log scale), `SHA256SUMS`. Scan of the public files and figure labels as in the
  curves runner; no date in the public files (two runs with the same id give the same bytes; dates only in
  `status.json` and `run.log`).

How to read the verdict (`aggregate.md`, section (b); `status.json` `verdict`), with this precedence:

1. `no verdict`: the verification run is INCOMPLETE or FAIL (e); the complete models are reported, the test section
   is not.
2. `not valid`: a shared history differs (a), even by 1e-9; the largest difference is given as model, epoch, field.
3. `positive`: the ten recover within 12 epochs and none of the 36 is degenerate; the recovery does not separate the
   stop from the lr halved by ReduceLROnPlateau, and a positive outcome does not by itself show that every model is
   stable.
4. `negative`: otherwise, "this change is not enough"; it does not rule out a role of early stopping.

With `positive` or `negative`, a model outside the ten without recovery (for example B fold 3 seed 44) is listed and
the conclusion adds that the change is not enough to make every model stable; with `positive` it also adds that this
does not rule out a role of early stopping ("in either case" of the protocol). Section (d) puts the verdicts and median
differences over the seeds of A\* - B, B' - B and A\* - B' of the two public aggregates side by side, labelled "not
used to choose or judge the recipe"; it is never part of the verdict.

Tests: `scripts/check_nuaa_patience_verify.py`, 135 tests (synthetic world, 36 small cards in both runs, histories
built by hand: all recover -> positive; one of the ten stays at 0.5 -> negative; one of the ten without recovery but
not degenerate -> negative; the recovery flag needs val_loss < 1.0; one of the ten recovers but keeps its checkpoint at
epoch 1 -> negative; B fold 3 seed 44 without recovery -> positive with the extended conclusion; a shared
history differing by 1e-9 -> not valid; INCOMPLETE and FAIL -> no verdict; null anchors -> run refused, check-inputs
structural; test read only from the public aggregates; no identifier in the public files; determinism; the refusals of
anchors, status, cards, thresholds and dev rows; after the audit of the checks: closed sets and values of every
configuration section and of the `amendment` block, card fields, recipe and history, booleans refused in place of integers such as fold and seed,
malformed JSON, CSV and `SHA256SUMS`, missing and irregular files, symbolic links for the output root and the run log,
an input inside the run folder, the cap reached exactly, an unreadable git state). Skips, only for a declared missing
input: in the development repository one without `--real` (the read-only test on the real inputs) and none with
`--real`; in the package layout also the check that the criteria equal those of the configuration of the previous
package (`delivery/degenerate/nuaa_degenerate_trainings.json`, not part of this package), and with `--real` the test on
the real inputs, which are not in the package. Mutation check: `scripts/check_nuaa_patience_verify_mutations.py`
(263 declared mutations, a control and 12 declared redundant, each with its reason in the script; `--list` checks
that every pattern is found exactly once without running anything; `--jobs 1`, `--only <name>` for a selection, never
on the real data: `PATVER_INPUT_ROOT` on an empty folder).

Steps executed after the training run (the Mac free, one heavy process at a time):

```bash
R=results/sessionablationrun/20261109-SESSABLRUN-6a50265
cat $R/status.json                                  # PASS, 36 of 36
shasum -a 256 $R/status.json $R/public/SHA256SUMS $R/restricted/SHA256SUMS
# 1. status_sha256, public_sums_sha256, restricted_sums_sha256 written in nuaa_patience_verify.json
# 2. its new SHA-256 (24c3f75b...) pinned as CONFIG_SHA256 in scripts/nuaa_patience_verify.py
shasum -a 256 delivery/patienceverify/nuaa_patience_verify.json
# 3. tests (synthetic, then read-only on the real inputs)
.venv/bin/python -I scripts/check_nuaa_patience_verify.py
.venv/bin/python -I scripts/check_nuaa_patience_verify.py --real
# 4. commit, then check-inputs and the run from a clean tree
.venv/bin/python -I scripts/nuaa_patience_verify.py check-inputs
.venv/bin/python -I scripts/nuaa_patience_verify.py run --i-am-authorized
# 5. the complete mutation check, one process, after the run (the mutants never read the real inputs)
.venv/bin/python -I scripts/check_nuaa_patience_verify_mutations.py --jobs 1
```

### Run of the comparison and outcomes

Run `20261109-PATVER-4f2b925` (clean commit `4f2b925`, configuration `24c3f75bab49`): `PASS`, verdict **positive**.
Public folder: `aggregate.json`, `aggregate.md`, `comparison.csv`, `curves.csv`, `figure_ten.png`,
`figure_no_recovery.png`, `figure_val_accuracy_all.png` and `SHA256SUMS` (SHA-256
`ba62e10f42e8c63b793146f2512fa573b3d1f63acad4b27a323b93664b5a4857`); no restricted folder. Every number below is in
`aggregate.md` (sections (a)-(e)) and `comparison.csv`; the reading is in `CONCLUSION.md`.

- (a) Shared histories: identical for 36 of 36 models (the first `epochs_run` epochs of the parent run, five fields);
  largest absolute difference 0.0 in `loss`, `accuracy`, `val_loss`, `val_accuracy` and `lr`. The comparison is valid.
- (b) Primary outcome: **positive**. The ten original degenerate trainings all recover within 12 epochs (first
  recovery at epoch 7 to 12, median 9.5); none of the 36 models is degenerate; no model without recovery, inside or
  outside the ten. B fold 3 seed 44, the only non-degenerate model without recovery in the parent run, is reported
  separately: it recovers at epoch 10 and restores epoch 11 (dev accuracy 1.0000). The recovery does not separate the
  stop from the lr halved by ReduceLROnPlateau (declared confounder), and a positive outcome does not by itself show
  that every model is stable.
- (c) Secondary outcomes per model, parent against verification run, and medians per group (the ten, the other 26, all
  36): models with a recovery 25 of 36 in the parent run, 36 of 36 here; for the ten, restored epoch median 11 (9-12),
  best val_loss 0.1373 (0.0004-0.5239), dev accuracy at the checkpoint 0.9900 (0.8650-1.0000), dev AUROC 0.9955
  (0.9895-1.0000), last lr 0.00025 or 0.000125 (1e-3 halved two or three times; 6.25e-05 in two models outside the ten).
  Five of the ten recover at epoch 10 or later and two keep a dev accuracy below 0.9 at the checkpoint; 15 of the 36
  checkpoints change (the ten and five of the nine non-degenerate models stopped early in the parent run); B' fold 4
  seed 42 still restores epoch 2 and falls to 0.33-0.34 at epochs 8-9 (observation, not a criterion; `CONCLUSION.md`).
- (d) Test, reported separately and not used to choose or judge the recipe (from the public `aggregate.json` of the
  two runs only): the two primary verdicts (A\* - B, stratum 03) stay `inconclusive`; 7 of the 84 verdicts differ from
  the parent run (6 consistent here against 1 there).
- (e) Times: wall 17,940.9 s against the estimate of 13,090 s (+4,850.9 s, +37 %), within the cap of 21,600 s; no write
  after the cap; not stopped.

## Files

| File | Content |
|---|---|
| `README.md` | this document |
| `NOTE.md` | the note of `delivery/degenerate/` amended after the agreement: protocol agreed on 7 November 2026; a short "Outcome" paragraph at the top points to `CONCLUSION.md` |
| `CONCLUSION.md` | the reading of the outcomes (a)-(e), cautions, limits, what is not concluded |
| `nuaa_session_ablation_run_patience12.json` | the amended configuration of the training run (`ca039df5d311`) |
| `nuaa_patience_verify.json` | the configuration of the comparison of the two runs, with the anchors of both runs (`24c3f75bab49`) |
| `test_logs/` | the eight logs listed under "Checks" |

Outputs, outside this folder: the training run `results/sessionablationrun/20261109-SESSABLRUN-6a50265/` (section
"Run") and the comparison run `results/patienceverify/20261109-PATVER-4f2b925/` (`status.json`, `run.log`, `public/`).

## Package layout (code repository)

One new folder, `research/nuaa-patience-verification/`, next to the packages already delivered; nothing accepted is
modified.

| Path | Content |
|---|---|
| `README.md`, `NOTE.md`, `CONCLUSION.md` | this document, the amended note, the conclusion |
| `SHA256SUMS` | SHA-256 of every file of the package |
| `livenesslab/delivery/patienceverify/nuaa_session_ablation_run_patience12.json` | the amended configuration of the training run |
| `livenesslab/delivery/patienceverify/nuaa_patience_verify.json` | the configuration of the comparison |
| `livenesslab/delivery/sessionablationrun/nuaa_session_ablation_run.json` | the parent configuration (`4c2d29d13ded`, same bytes as in `research/nuaa-session-ablation-run/`), read by the amendment check before any data; with it the two file tests of the amended configuration run in the package layout too |
| `livenesslab/scripts/nuaa_session_ablation_run.py`, `check_nuaa_session_ablation_run.py`, `check_nuaa_session_ablation_run_mutations.py` | the runner with the amendment check, its suite (99 tests) and its mutation check (210 declared mutations) |
| `livenesslab/scripts/nuaa_folds_run.py`, `check_nuaa_folds_run.py` | the trainer and scorer imported by the runner, and their suite, unchanged |
| `livenesslab/scripts/nuaa_patience_verify.py`, `check_nuaa_patience_verify.py`, `check_nuaa_patience_verify_mutations.py` | the comparison script, its suite (135 tests) and its mutation check (263 declared mutations) |
| `livenesslab/scripts/m2_writers.py`, `path_guard.py`, `package_rules.py` | the shared modules, unchanged |
| `run/status.json`, `run/run.log.txt` | final status and log of the training run (`run.log` renamed, same bytes) |
| `tables/` | `public/` of the training run: `aggregate.json`, `aggregate.md`, `run_record.json`, `SHA256SUMS` (same bytes) |
| `comparison/` | `public/` of the comparison run: `aggregate.json`, `aggregate.md`, `comparison.csv`, `curves.csv`, the three figures, `SHA256SUMS` (same bytes) |
| `comparison_run/status.json`, `comparison_run/run.log.txt` | final status and log of the comparison run (`run.log` renamed, same bytes) |
| `test_logs/` | the eight logs listed under "Checks" |

Restricted: the `restricted/` folder of the training run goes to the shared N1 folder on Drive only, as for the parent
run; the comparison run writes no restricted output.
