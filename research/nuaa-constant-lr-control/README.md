# Constant learning-rate control: the same 36 LivenessNet models with patience 12 and no ReduceLROnPlateau

Control proposed in section 5 of `delivery/synthesis/SYNTHESIS.md` (PR #24, merged) and confirmed by the coordinator
on the C2 card on 10 November 2026: the same 36 models (arms B, A\* and B' x 4 folds x seeds 42/43/44) and data as the
patience-12 run `20261109-SESSABLRUN-6a50265`, `early_stopping_patience` 12, at most 12 epochs, the learning rate
constant at 0.001 (ReduceLROnPlateau removed), the cap of 360 minutes, the test separate; only the scheduler changes,
the original path stays unchanged, the effective learning rate is recorded at every epoch, every outcome is kept, the
histories are checked against the reference run before its first learning-rate reduction, and on a discrepancy or over
budget the run stops and the point reached is reported. The protocol fixed before the run is `NOTE.md`.

Status: **control run `20261111-SESSABLRUN-8568ffa` complete (PASS, 36 of 36, 11 November 2026, within the cap);
comparison with the reference run complete (run `20261111-LRCTRL-ff53279`): shared histories identical 36/36, rate
constant 36/36, primary outcome mixed (the ten recover, none degenerate, one recovery of the reference run lost: B'
fold 2 seed 42), reported model by model with no overall verdict, as the rule fixed before the run prescribes.** The reading, with its cautions and limits, is in `CONCLUSION.md`. This
folder holds the configuration of the training run, the note, the conclusion, the configuration of the comparison and
the logs of the checks (sections "Run" and "Comparison of the two runs"); the package layout of the code repository is
at the end.

## Configuration and declared difference

`nuaa_session_ablation_run_constant_lr.json`, SHA-256
`60b38e094da8c2243f3f9386f69d216bf8b853a3511b73538feec39e09b0d58f` (pinned in the runner as
`CONSTANT_LR_CONFIG_SHA256` before the run).

Parent: `delivery/patienceverify/nuaa_session_ablation_run_patience12.json`, SHA-256
`ca039df5d3114f9cb983c7256330aae671de54ae34d949ad4e9b9c80beb6aa43` (the configuration of the reference run,
`PATIENCE12_CONFIG_SHA256` in the runner), itself the configuration of run `20261106-SESSABLRUN-d999458`
(`4c2d29d13ded`) with patience 12. The new configuration is the parent with exactly two differences:

- `training.scheduler`: from `ReduceLROnPlateau(monitor=val_loss, factor=0.5, patience=2, min_lr=1e-6)` to
  `constant (no ReduceLROnPlateau; the initial learning rate for every epoch; the learning rate in effect is recorded in
  history.lr at the end of every epoch)`;
- the `amendment` block, rewritten for this control (nine keys, as in the parent, with `agreed_conditions` in place of
  `agreed_precisions`): `parent_config_sha256`
  (`ca039df5...`), `changed_field` (`training.scheduler`), `from`, `to`, `authorisation` (the coordinator's comment of
  10 November 2026, quoted in Italian), `agreed_conditions` (five: only the scheduler changes; the original path stays
  unchanged; the rate recorded at every epoch; every outcome kept and the histories checked, stop on a discrepancy or
  over budget; the test separate), `note`, `time_estimate` (17,940.9 s, cap 21,600 s), `shared_history_check` (the
  epochs of the reference run before its first reduction).

Every other field is identical, `task`, `authorisation`, `budget` and `early_stopping_patience` 12 included; same key
order and indentation (`diff` of the two files: one changed line and the `amendment` block). The descriptive texts of
the parent that mention ReduceLROnPlateau are kept unchanged, as the amendment mechanism requires; the `amendment`
block prevails (`NOTE.md`, "The one change").

## Checks

The trainer `scripts/nuaa_folds_run.py`, imported by the runner, now reads `training.scheduler`:

- Two values only, the recipe string (`SCHEDULER_REDUCE`) and the constant one (`SCHEDULER_CONSTANT`); any other value
  is refused before TensorFlow is imported (`check_scheduler`).
- `training_callbacks` builds the callbacks: with the recipe string `[EarlyStopping, ReduceLROnPlateau, Cap]` as
  before (patience of the configuration, `restore_best_weights`; factor 0.5, patience 2, min_lr 1e-6); with the
  constant string `[EarlyStopping, LearningRateLog, Cap]`, where `LearningRateLog` changes nothing and, at the end of
  every epoch, writes into the epoch logs the rate held by the optimizer with the same expression as Keras 2.15
  ReduceLROnPlateau (`backend.get_value(optimizer.lr)`), so that `history.lr` holds one value per epoch in both modes:
  the rate with which the epoch was trained (ReduceLROnPlateau records it before any reduction).
- The training card records `scheduler`; `card_problems` refuses a card whose scheduler differs from the
  configuration, whose `history.lr` does not hold one value per epoch or, with the constant scheduler, whose recorded
  rate is not 0.001 (compared in float32, as the optimizer holds it).
- `FROZEN_TRAINING` (the recipe of the folds run, with ReduceLROnPlateau) is unchanged; the folds run and the two
  earlier configurations keep the original path.

The runner `scripts/nuaa_session_ablation_run.py` is the one of the reference run with these additions:

- The amendment check reads a table of known amended configurations (`AMENDMENTS`): `patience12_verification` (parent
  the delivered configuration `4c2d29d13ded`, `training.early_stopping_patience` 4 -> 12, two agreed precisions) and
  `constant_lr_control` (parent the patience-12 configuration `ca039df5d311`, `training.scheduler` ReduceLROnPlateau
  -> constant, five agreed conditions). Before any data is read, `check-inputs` and `run` refuse a configuration unless
  its SHA-256 is the pinned one, the declared parent is the pinned parent, the parent read from the repository has that
  SHA-256, every field other than the changed one and `amendment` equals the parent (`config_difference`, strict
  types), the parent holds the "from" value and the configuration the "to" value, `training` is the frozen recipe with
  only the cumulated changes (patience 12 and the constant rate) and the `amendment` block has exactly its nine keys
  with the agreed conditions; otherwise FAIL with the field in the message, nothing written.
- `status.json`, `run_record.json`, `aggregate.json` and `aggregate.md` record `configuration_name`
  (`constant_lr_control`) and the amendment (parent SHA-256, field, from, to); `aggregate.md` says "constant learning
  rate, no schedule" where it describes the recipe.
- Shared code: the runner requires the trainer used by the folds run; it also accepts one pinned revision of
  `nuaa_folds_run.py` (`TRAINER_REVISION_SHA256`, the one with the operative scheduler) and records which of the two
  was used; the other shared files stay compared as before.
- Unchanged: run id `YYYYMMDD-SESSABLRUN-<commit>[-dev]`, folder `results/sessionablationrun/`, input checks, order,
  threshold, scoring, aggregate, reading rule, cap and outcomes. The delivered and the patience-12 configurations pass
  as before.

Tests. Trainer suite `scripts/check_nuaa_folds_run.py`: 79 (71 + 8 new: the two scheduler values and the refusal of
any other, the callbacks in both modes with their parameters and order, the rate written by `LearningRateLog` with the
Keras expression, the card with `scheduler` and the refusals of a missing or wrong rate, `FROZEN_TRAINING` unchanged);
mutation check `scripts/check_nuaa_folds_run_mutations.py`: 133 declared mutations (112 + 21 new), a control and three
declared redundant, the mutants never on the real inputs (`FOLDSRUN_INPUT_ROOT` on the copy). Runner suite
`scripts/check_nuaa_session_ablation_run.py`: 119 (99 + 20 new: the pinned SHA-256 of the new configuration and of its
parent, the real files differing only in the allowed field and block, the amendment table and its closed fields, the
refusals of an unpinned SHA-256, a wrong parent, a difference in another field, a wrong "from" or "to" value, a
training block beyond the cumulated changes, the five conditions, `configuration_name` in the outputs, the pinned
trainer revision); mutation check `scripts/check_nuaa_session_ablation_run_mutations.py`: 241 declared mutations (210 +
31 new), a control and three declared redundant (`SESSABLRUN_INPUT_ROOT` on the copy). The comparison suite and
mutation check are under "Comparison of the two runs".

Logs in `test_logs/` (each opens with the environment and the SHA-256 of the scripts and configurations it ran on):

| Log | Content |
|---|---|
| `check_nuaa_folds_run.txt` | the trainer suite on macOS with the real inputs present (read only): `tests run: 79, failures: 0, errors: 0, skipped: 0`, `RESULT: PASS` |
| `check_nuaa_folds_run_mutations.txt` | the trainer mutation check, `--jobs 2`, every copy without data: `mutations killed: 133/133`, `control mutations survived: 4/4 (1 control, 3 declared redundant)`, `RESULT: PASS` |
| `check_nuaa_session_ablation_run.txt` | the runner suite on macOS with the real inputs and the three configurations: `tests run: 119, failures: 0, errors: 0, skipped: 0`, `RESULT: PASS` |
| `check_nuaa_session_ablation_run_mutations.txt` | the runner mutation check, `--jobs 2`, every copy without data: `mutations killed: 241/241`, `control mutations survived: 4/4 (1 control, 3 declared redundant)`, `RESULT: PASS` |
| `check_inputs_constant_lr.txt` | `check-inputs` of the runner on the real inputs with the new configuration: the `AMENDMENT` line, the input line (16 cell manifests, 6,000 cache frames, 6,614 extracted frames), `RESULT: PASS (check-inputs)` |
| `check_nuaa_patience_verify.txt` | the comparison suite, synthetic, twice: with the development interpreter (`tests run: 178, failures: 0, errors: 0, skipped: 4`, the four read-only tests on the real inputs without `--real`) and with the system `python3` (standard library only), `RESULT: PASS` both |
| `check_nuaa_patience_verify_real.txt` | the comparison suite with `--real`, the real inputs of the two runs read only: `tests run: 178, failures: 0, errors: 0, skipped: 0`, `RESULT: PASS` |
| `check_nuaa_patience_verify_mutations.txt` | the comparison mutation check, `--jobs 2`, `PATVER_INPUT_ROOT` on an empty folder: `mutations killed: 329/329`, `control mutations survived: 14/14` (1 control, 13 declared redundant), `RESULT: PASS` |
| `check_inputs_constant_lr_verify.txt` | `check-inputs` of the comparison on the real inputs: reference and control run, cards, dev scores and thresholds, Keras `EarlyStopping` replay 36/36 in each run, the shared-history windows (3-12 epochs, never reduced 1/36), `RESULT: PASS (check-inputs)` |
| `linux_package_no_data.txt` | the three suites, the two `check-inputs` and the three mutation checks (`--jobs 2`) from the package layout in a Linux container (`python:3.12`, NumPy only, no OpenCV, no matplotlib, no TensorFlow), without data: trainer suite `tests run: 79 ... skipped: 1`, runner suite `tests run: 119 ... skipped: 1` (the checks on the real inputs), comparison suite `tests run: 178 ... skipped: 8` (the four tests on the configuration of the previous package, `delivery/patienceverify/nuaa_patience_verify.json`, not part of this package, and the four `--real` tests), all `RESULT: PASS`; runner `check-inputs` `RESULT: FAIL (check-inputs)` at the first missing input, after the amendment check; comparison `check-inputs` `RESULT: FAIL (check-inputs)` at the missing reference run; mutations `133/133`, `241/241` and `329/329` killed, `RESULT: PASS` |

## Run

Commands, from the development repository root with the data present, after the trainer, the runner, the
configuration and the logs were committed (clean tree):

```bash
.venv/bin/python -I scripts/nuaa_session_ablation_run.py check-inputs --config delivery/constlr/nuaa_session_ablation_run_constant_lr.json
caffeinate -dims .venv/bin/python -I scripts/nuaa_session_ablation_run.py run --i-am-authorized --config delivery/constlr/nuaa_session_ablation_run_constant_lr.json
```

- Run `20261111-SESSABLRUN-8568ffa` (clean commit `8568ffa` of the development repository, `dirty: false`; delivered
  configuration `60b38e094da8`, `configuration_name` `constant_lr_control`, amendment `training.scheduler`
  ReduceLROnPlateau -> constant with every other field equal to the parent `ca039df5d311`), started 2026-11-11
  00:07:33+0100, finished 04:21:10+0100: `PASS`, 36 of 36 trainings complete, none missing, not stopped.
- While the run was in progress each new training card was checked, as soon as it was written, against the reference
  run (identical history over the shared window, rate 0.001 at every epoch) with a read-only script (first card at
  00:15, 36th at 04:21): no discrepancy at any of the 36 cards, so there was no cause to stop. The same check is made
  by the comparison, section (a).
- Times: wall 15,217.0 s (about 254 of the 360 minutes; estimate 17,940.9 s, -2,723.9 s, -15 %), of which training
  14,328.3 s (every model 12 epochs; seconds per epoch, weighted: B 36.5153, A\* 37.0486, B' 25.9382, against 42.6097,
  40.0181 and 34.4986 in the reference run); aggregate 2.2 s. The run was the only heavy process, at night, on a Mac
  not dedicated to it, so the seconds per epoch are not a measure of the recipe. Cap not reached; no write after the cap
  (`final_writes_after_cap: false`).
- Peak resident memory 5,078 MB (1,378 MB after the check-inputs made by `run` at the start, as in `run.log`).
- Primary verdicts printed by the runner (A\* - B, stratum 03), both `inconclusive`: they belong to the test, which is
  reported in the separate section (d) of the comparison and is not used to judge the control.
- Where the outcomes are: in the run folder, `status.json` (SHA-256 `f16a2720013888...`), `run.log`, `public/`
  (`aggregate.json`, `aggregate.md`, `run_record.json` and `SHA256SUMS`, SHA-256 `cb353b754318e209...` recorded in
  `status.json`) and `restricted/` (36 x card, dev scores, test scores, threshold and weights, `per_subject.csv`,
  `public_SHA256SUMS.sha256`: 182 files listed in its `SHA256SUMS`, SHA-256 `ea821cbad2d8ccf8...`). In the package of
  the code repository, `public/` is copied as `tables/`, `status.json` as `run/status.json` and `run.log` as
  `run/run.log.txt` (same bytes). The whole run folder (public, restricted, `status.json`, `run.log`) goes to the shared
  N1 folder on Drive, as for the reference run; the restricted files only there.

## Comparison of the two runs

A separate script, not the runner, compares the training cards of the control run with those of the reference run
`20261109-SESSABLRUN-6a50265`, by the reading rule fixed before the run (`NOTE.md`, points (a)-(e)).

- Script: `scripts/nuaa_patience_verify.py`, the comparison script of the previous package with a second configuration
  schema (`nuaa-patience-verify-config/2`); schema 1 is unchanged and gives the same outputs as before (regression on
  run `20261109-PATVER-4f2b925`: tables and figures identical byte for byte). `check-inputs`; `run --i-am-authorized`,
  which reads the restricted training cards, dev scores and frozen thresholds of the two runs; `--config`,
  `--input-root`, `--out-root` (default `results/lrcontrol` with schema 2), `--date`. Standard library and the shared
  modules `scripts/m2_writers.py`, `scripts/path_guard.py`, `scripts/package_rules.py`; matplotlib optional for the
  figures (otherwise a declared skip); no NumPy, no TensorFlow.
- Configuration: `nuaa_constant_lr_verify.json` (schema 2), pinned in the script as `CONFIG_SHA256_V2`. It anchors the
  reference run by SHA-256 (`status.json` `af6a1d1e1f7b...`, public `SHA256SUMS` `018d5079917856...`, restricted
  `SHA256SUMS` `6787bc2bfb280f...`, the same values as `delivery/patienceverify/nuaa_patience_verify.json`), names the
  control run (run id, folder, `config_sha256` `60b38e09...`, commit `8568ffa`, `configuration_name`
  `constant_lr_control`, amendment `training.scheduler` from ReduceLROnPlateau to constant), copies the criteria of
  the previous packages with the same values (degenerate, saturation, recovery, checkpoint agreement, difference 0),
  adds the window rule (`shared_history_window` = the epochs of the reference run before its first reduction) and the
  recorded recipe rate (`recorded_recipe_lr` 0.0010000000474974513, compared exactly), lists the ten original
  degenerate models and the two models reported apart (B fold 3 seed 44, B' fold 4 seed 42), and states the reading
  rule (a)-(e), the time estimate (17,940.9 s, cap 21,600 s), the declarations and the outputs. The anchors of the
  control run were null before the training run and were filled after it from the files of the run folder:
  `status_sha256` `f16a2720013888f5...`, `public_sums_sha256` `cb353b754318e209...`, `restricted_sums_sha256`
  `ea821cbad2d8ccf8...`; the configuration with the anchors has SHA-256
  `e4896a2640f4e26c5b2c1ff220965a9aad6c0f78461571ca53dc9240a0714c4f`. With null anchors `run` refuses and
  `check-inputs` checks only the configuration and the reference run and says so; `public_sums_sha256` is `"none"` only
  for a FAIL run; a FAIL run without complete models is refused.
- Checks (any failure: FAIL, nothing public): the anchors; the reference run `PASS`, with patience 12 and the recipe
  scheduler; the control run `PASS`, `INCOMPLETE` or `FAIL`, of the run id, commit, `config_sha256`,
  `configuration_name` and amendment of the configuration, from a clean tree, with complete and missing models that
  split the 36 and exactly the cards of the complete models in its restricted `SHA256SUMS`; every card as in the
  previous package (closed keys, recipe, history, restored epoch = argmin, exact replay of the Keras `EarlyStopping`,
  weights listed), the cards of the control run with exactly the extra key `scheduler` equal to `amendment.to` and
  those of the reference run without it; the reference histories with the first epoch at the recipe rate and a rate
  that never rises; thresholds and dev scores row by row; the same dev for every model of a fold and for the same model
  in the two runs; the same fitting set; the ten equal to what the criteria give on the cards of the patience-4 run as
  recorded; the public `aggregate.json` of each run listed in its public `SHA256SUMS`, of the same run, configuration
  and status, with the closed structure of the test section; a run folder that already holds `status.json` is refused.
  The `*_test_scores.csv` files are never opened.
- Outputs (`results/lrcontrol/YYYYMMDD-LRCTRL-<commit>[-dev]/`: `status.json`, `run.log`, `public/`; no restricted
  folder): `comparison.csv` (one row per model: arm, fold, seed, one of the ten, status in the control run; reference
  and control epochs run, restored epoch, degenerate, recovered, first recovery, best val_loss, dev accuracy and AUROC
  at the checkpoint, epochs at 0.5, last lr; shared epochs, identical, largest absolute difference per field; recovered
  at the last epoch, restored epoch changed, first reduced epoch of the reference run, rate constant and values of the
  control run, reported apart), `curves.csv` (control run, per epoch), `aggregate.json` and `aggregate.md` (sections
  (a)-(e) in this order and the declarations), `figure_ten.png`, `figure_no_recovery.png`,
  `figure_val_accuracy_all.png` (reference dashed, control solid, val_loss on a log scale), `SHA256SUMS`. Scan of the
  public files and figure labels as before; no date in the public files.

How to read the verdict (`aggregate.md`, section (b); `status.json` `verdict`), with this precedence:

1. `no verdict`: the control run is INCOMPLETE or FAIL (e); the complete models are reported, the test section is
   not.
2. `not valid`: a shared history differs (a), even by 1e-9, or the rate of the control run is not 0.001 at some
   epoch; the largest difference, or the models with another rate, is given.
3. `consistent with sufficient epochs`: the ten recover within 12 epochs, none of the 36 is degenerate and no
   recovery of the reference run is lost outside the ten.
4. `consistent with a contribution of the reduction`: at least one of the ten does not recover; the models are listed
   one by one.
5. `mixed (no overall verdict)`: every other outcome (all ten recover, but a model is degenerate or a recovery of the
   reference run is lost outside the ten); reported model by model, with no overall verdict.

Section (d) puts the verdicts and median differences over the seeds of A\* - B, B' - B and A\* - B' of the two public
aggregates side by side, labelled "not used to choose or judge the recipe"; it is never part of the verdict.

Tests: `scripts/check_nuaa_patience_verify.py`, 178 tests (135 of the previous package, unchanged, + 43 for schema 2:
the pinned SHA-256 of the new configuration; the window per model, its edge cases (first reduced epoch at 2, never
reduced, a first epoch at another rate or a rising rate refused, the exact float32 comparison); (a) with a difference
inside and outside the window and with a rate other than 0.001 in the control run; (b) sufficient, contribution and
mixed, including all ten recovered with one degenerate among the 36 and with a recovery of the reference run lost
outside the ten; (c) recovered at the last epoch, restored epoch changed, the models reported apart; the cards of the
control run with the `scheduler` key and the refusals of a missing, extra or wrong key in either run; the labels of the
two runs in texts, tables and figures; the out root by schema; schema 1 unchanged). Skips, only for a declared missing
input: in the development repository four without `--real` (the read-only tests on the real inputs) and none with
`--real`; in the package layout also the check against the configuration of the previous package (not part of this
package). Mutation check: `scripts/check_nuaa_patience_verify_mutations.py` (329 declared mutations, 263 of the previous
package + 66 for schema 2, a control and 13 declared redundant, each with its reason in the script; `--list` checks that
every pattern is found exactly once without running anything; `--jobs`, `--only <name>` for a selection, never on the
real data: `PATVER_INPUT_ROOT` on an empty folder).

Steps executed after the training run (the Mac free, one heavy process at a time):

```bash
R=results/sessionablationrun/20261111-SESSABLRUN-8568ffa
cat $R/status.json                                  # PASS, 36 of 36
shasum -a 256 $R/status.json $R/public/SHA256SUMS $R/restricted/SHA256SUMS
# 1. run_id, folder, commit, status_sha256, public_sums_sha256, restricted_sums_sha256 written in nuaa_constant_lr_verify.json
# 2. its new SHA-256 (e4896a26...) pinned as CONFIG_SHA256_V2 in scripts/nuaa_patience_verify.py
shasum -a 256 delivery/constlr/nuaa_constant_lr_verify.json
# 3. tests (synthetic, then read-only on the real inputs)
.venv/bin/python -I scripts/check_nuaa_patience_verify.py
.venv/bin/python -I scripts/check_nuaa_patience_verify.py --real
# 4. commit (the run folder too), then check-inputs and the run from a clean tree
.venv/bin/python -I scripts/nuaa_patience_verify.py check-inputs --config delivery/constlr/nuaa_constant_lr_verify.json
.venv/bin/python -I scripts/nuaa_patience_verify.py run --i-am-authorized --config delivery/constlr/nuaa_constant_lr_verify.json
# 5. the complete mutation check after the run (the mutants never read the real inputs)
.venv/bin/python -I scripts/check_nuaa_patience_verify_mutations.py --jobs 2
```

### Run of the comparison and outcomes

Run `20261111-LRCTRL-ff53279` (clean commit `ff53279`, configuration `e4896a2640f4`): `PASS`, verdict **mixed (no
overall verdict)**. Public folder: `aggregate.json`, `aggregate.md`, `comparison.csv`, `curves.csv`,
`figure_ten.png`, `figure_no_recovery.png`, `figure_val_accuracy_all.png` and `SHA256SUMS` (SHA-256
`3b15fdeb2f4a86920cef791d792ea96d2d48238d5aae6be56a8b1c7194f309f3`); no restricted folder. Every number below is in
`aggregate.md` (sections (a)-(e)) and `comparison.csv`; the reading is in `CONCLUSION.md`.

- (a) Shared histories: identical for 36 of 36 models over the epochs of the reference run before its first reduction
  (3 to 12 epochs; A\* fold 1 seed 42 never reduced), largest absolute difference 0.0 in `loss`, `accuracy`,
  `val_loss`, `val_accuracy` and `lr`; rate 0.001 at every epoch of the control run (36 of 36). The comparison is valid.
- (b) Primary outcome: **mixed, no overall verdict**. The ten original degenerate trainings all recover within 12
  epochs with the constant rate (first recovery at epoch 4 to 10, median 6; 7 to 12, median 9.5, in the reference run,
  where every recovery of the ten came after the first reduction); none of the 36 models is degenerate; one recovery
  of the reference run is lost outside the ten: B' fold 2 seed 42 (recovered at epoch 11 at 0.0005 in the reference
  run; here dev accuracy 0.77-0.8625 at epochs 9-12 with val_loss never below 1.6; restored epoch 10, dev accuracy
  0.8550, dev AUROC 0.8678). By the rule, the outcome is reported model by model with no overall verdict.
- (c) Secondary outcomes per model, reference against control run, and medians per group (the ten, the other 26, all
  36): models with a recovery 36 of 36 there, 35 of 36 here; the recovery criterion holds at epoch 12 in 28 there and
  26 here; the restored epoch changes in 23 of 36 (8 of the ten); for the ten, restored epoch median 9.5 (6-12), best
  val_loss 0.0142 (0.0001-0.3141), dev accuracy at the checkpoint 0.9975 (0.8750-1.0000), dev AUROC 1.0000
  (0.9831-1.0000); the last rate is 0.001 in every model. B fold 3 seed 44 recovers at epoch 7 (10 there) and restores
  epoch 10 (dev accuracy 0.9150); B' fold 4 seed 42 relapses to 0.5 at epochs 6-8 (5-7 in the reference run, then 0.33-0.34 at
  epochs 8-9) and restores epoch 11 (dev accuracy 0.9700) instead of epoch 2 (`CONCLUSION.md`).
- (d) Test, reported separately and not used to choose or judge the recipe (from the public `aggregate.json` of the
  two runs only): the two primary verdicts (A\* - B, stratum 03) stay `inconclusive`; 12 of the 84 verdicts differ
  from the reference run (4 consistent there and inconclusive here, 8 inconclusive there and consistent here).
- (e) Times: wall 15,217.0 s against the estimate of 17,940.9 s (-2,723.9 s, -15 %), within the cap of 21,600 s; no
  write after the cap; not stopped.

## Files

| File | Content |
|---|---|
| `README.md` | this document |
| `NOTE.md` | the protocol fixed before the run (question, what stays fixed, the one change, implementation, reading rule, budget, limits); a short "Outcome" paragraph at the top points to `CONCLUSION.md` |
| `CONCLUSION.md` | the reading of the outcomes (a)-(e), cautions, limits, what is not concluded |
| `nuaa_session_ablation_run_constant_lr.json` | the amended configuration of the training run (`60b38e094da8`) |
| `nuaa_constant_lr_verify.json` | the configuration of the comparison of the two runs, with the anchors of both runs (`e4896a2640f4`); its `reading_rule.b` holds the "no overall verdict" of the mixed case |
| `test_logs/` | the ten logs listed under "Checks" |

Outputs, outside this folder: the training run `results/sessionablationrun/20261111-SESSABLRUN-8568ffa/` (section
"Run") and the comparison run `results/lrcontrol/20261111-LRCTRL-ff53279/` (`status.json`, `run.log`, `public/`).

## Package layout (code repository)

One new folder, `research/nuaa-constant-lr-control/`, next to the packages already delivered; nothing accepted is
modified.

| Path | Content |
|---|---|
| `README.md`, `NOTE.md`, `CONCLUSION.md` | this document, the note, the conclusion |
| `SHA256SUMS` | SHA-256 of every file of the package |
| `livenesslab/delivery/constlr/nuaa_session_ablation_run_constant_lr.json` | the amended configuration of the training run |
| `livenesslab/delivery/constlr/nuaa_constant_lr_verify.json` | the configuration of the comparison |
| `livenesslab/delivery/patienceverify/nuaa_session_ablation_run_patience12.json` | the parent configuration (`ca039df5d311`, same bytes as in `research/nuaa-patience-verification/`), read by the amendment check before any data |
| `livenesslab/delivery/sessionablationrun/nuaa_session_ablation_run.json` | the delivered configuration (`4c2d29d13ded`, same bytes as in `research/nuaa-session-ablation-run/`), the parent of the parent, read by the same check |
| `livenesslab/delivery/foldsrun/nuaa_folds_run.json` | the configuration of the folds run (same bytes as in `research/nuaa-subject-folds/`), read by the trainer suite |
| `livenesslab/scripts/nuaa_folds_run.py`, `check_nuaa_folds_run.py`, `check_nuaa_folds_run_mutations.py`, `train_cnn.py` | the trainer with the operative scheduler, its suite (79 tests), its mutation check (133 declared mutations) and the module it imports, unchanged |
| `livenesslab/scripts/nuaa_session_ablation_run.py`, `check_nuaa_session_ablation_run.py`, `check_nuaa_session_ablation_run_mutations.py` | the runner with the amendment table, its suite (119 tests) and its mutation check (241 declared mutations) |
| `livenesslab/scripts/nuaa_patience_verify.py`, `check_nuaa_patience_verify.py`, `check_nuaa_patience_verify_mutations.py` | the comparison script with schema 2, its suite (178 tests) and its mutation check (329 declared mutations) |
| `livenesslab/scripts/m2_writers.py`, `path_guard.py`, `package_rules.py` | the shared modules, unchanged |
| `run/status.json`, `run/run.log.txt` | final status and log of the training run (`run.log` renamed, same bytes) |
| `tables/` | `public/` of the training run: `aggregate.json`, `aggregate.md`, `run_record.json`, `SHA256SUMS` (same bytes) |
| `comparison/` | `public/` of the comparison run: `aggregate.json`, `aggregate.md`, `comparison.csv`, `curves.csv`, the three figures, `SHA256SUMS` (same bytes) |
| `comparison_run/status.json`, `comparison_run/run.log.txt` | final status and log of the comparison run (`run.log` renamed, same bytes) |
| `test_logs/` | the ten logs listed under "Checks" |

Restricted: the `restricted/` folder of the training run goes to the shared N1 folder on Drive only, as for the
reference run; the comparison run writes no restricted output.
