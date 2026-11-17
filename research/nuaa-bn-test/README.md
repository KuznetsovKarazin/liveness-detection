# NUAA: BatchNormalization re-estimation of PR #27 on the test, before/after (pre-run: note, configuration, code)

Assigned by the coordinator on the C2 card on 16 November 2026, after the merge of PR #27 (`c234064`): before/after
comparison, on the NUAA test of the same 36 patience-4 checkpoints of run `20261106-SESSABLRUN-d999458` and without new
training, of the BatchNormalization re-estimation of PR #27, with note, configuration and code published before the
run. In the coordinator's terms, a **post hoc diagnostic extension on a test already analysed, not an independent
validation**: the "before" test metrics are already published (parent run), and the re-estimation was chosen and first
measured on the dev of the same folds.

**Status at this commit.** Pre-run: note, configuration, runner, tests and their logs. No checkpoint has been loaded
as a model, no test frame assembled or scored, no `<model>_test_scores.csv` of the parent run parsed. The only command
run on the private inputs is `check-inputs` (PASS, `test_logs/check_inputs_bntest.txt`): configuration and pinned code
against PR #27; the parent run's cards, dev scores, thresholds, `.keras` files and `<model>_test_scores.csv` against
its restricted SHA256SUMS (test score files hashed only); the fold schema with its test manifests (12,614 test rows);
the frame files (cache and extracted frames) by SHA-256, the extracted frames loaded whole and each frame checked
against its recorded SHA-256, no frame assembled; the BNRECAL run of PR #27; the groups (10 degenerate, 25 controls,
1 apart; 17 with a worsened dev indicator in PR #27). No model loaded, TensorFlow not imported, nothing written. The
run starts after this commit is pushed, from the published code, without a further approval, as the coordinator
wrote; if every check passes, the report (before/after table, figure of the differences, conclusion) and the public
outputs follow in a later commit of this pull request, otherwise the stop and its table (`baseline.csv` or
`recalibration_check.csv`).

## What the runner reads of the test

Summarised from the field `test` of the configuration (the runner requires its exact text):

- phase 0: the fold schema is opened and validated (`fold<k>_fitting.csv`, `fold<k>_dev.csv`, `fold<k>_test.csv`, as
  in the parent run); every `<model>_test_scores.csv` of the parent run is hashed, not parsed;
- phase 1: the test frames of each fold (every frame of the held-out subjects: 3,242, 3,114, 3,439 and 2,819) are
  assembled and scored with the original statistics by the evaluator of PR #27; the parent's
  `<model>_test_scores.csv` are parsed to compare the per-frame attack probabilities; the test accuracy, val_loss
  and AUROC that the evaluator computes in memory (here and again in phase 3a) are discarded: no test metric is
  written or logged before phase 3b;
- only if phases 1 and 2 pass for all 36 models: phase 3a reassembles the same frames and scores them with the
  re-estimated statistics, then the 36 checkpoints and the three anchoring SHA256SUMS are re-hashed; phase 3b
  computes APCER, BPCER and ACER at the fixed threshold 0.5 and AUROC, before and after and, as a reported check,
  from the parent's stored scores;
- the M2 manifests `official_test` and `cnn_test` are read to index the cache; the parent's public aggregate, which
  holds published test metrics, is read only as bytes for its SHA-256; no public file of the folds run is opened;
- precision on the frozen texts: where the note (section 3), the configuration (`budget.on_cap_exceeded`) and the
  docstring of `score_phase` say that no test metric is computed before phase 3b or in an `INCOMPLETE` run, this
  refers to the comparison metrics; the evaluator's in-memory accuracy, val_loss and AUROC of phases 1 and 3a are
  computed and discarded, never written or logged.

## Tests (synthetic data only)

From this folder, with TensorFlow 2.15 on CPU, NumPy and matplotlib:

```bash
python -I -B livenesslab/scripts/check_nuaa_bn_test.py
```

```bash
python -I -B livenesslab/scripts/check_nuaa_bn_test_mutations.py
```

The suite: 64 tests on a synthetic world in a temporary folder (tiny Keras models, 8×8 frames, a BNRECAL run produced
by `nuaa_bn_recalibration.py`), 268 s in the log. The mutation check: 119 mutants of `nuaa_bn_test.py`, one at a time,
119 killed; 1 control and 6 declared redundant mutants survived, as required; started at 22:17 CET on 16 November
and finished at 00:26 on 17 November in the log. Neither uses a real checkpoint, frame or score: the tolerances on the
real test, the time and the memory are checked only by the run.

`livenesslab/` mirrors the working repository, where the runner is executed on the private inputs (`check-inputs`,
then `run --i-am-authorized`); the run refuses a working tree that is not clean and a null `note_sha256`.

## Time

About 42 minutes, derived from the PR #27 run on the same machine (phase 1 588 s, phase 2 1,401 s, phase 3a 447 s,
about 90 s unmeasured); upper estimate about 112. Cap: the four-hour limit as elapsed time of a CPU-only run, 240
minutes of wall time (checked before each model and, in phase 2, before each layer; then `INCOMPLETE`, no test
metric published); process CPU time over all threads is recorded, not capped or estimated, and can exceed the elapsed
time.

## Contents

| path | what it is |
|---|---|
| `NOTE.md` | the pre-run note |
| `livenesslab/scripts/nuaa_bn_test.py` | the runner: `check-inputs`, `run`, `figure` |
| `livenesslab/delivery/bntest/nuaa_bn_test.json` | its configuration, a closed set: declarations, inputs with their SHA-256, pinned code, tolerances with their sources, stop rules, comparison, bootstrap, reading rule, outputs, budget, `note_sha256`, `test` |
| `livenesslab/scripts/check_nuaa_bn_test.py` | the synthetic suite (64 tests) |
| `livenesslab/scripts/check_nuaa_bn_test_mutations.py` | the mutation check (119 mutants, 1 control, 6 declared redundant) |
| `livenesslab/scripts/nuaa_bn_recalibration.py` | the estimator and evaluator of PR #27, imported unchanged; SHA-256 `8c228ef400f5ef4337612477b23113fb6b8cac5364a5ff026e2c0ac8096734f5`, pinned in the runner and in the configuration |
| `livenesslab/scripts/check_nuaa_bn_recalibration.py`, `nuaa_session_ablation_run.py`, `check_nuaa_session_ablation_run.py`, `nuaa_folds_run.py`, `check_nuaa_folds_run.py`, `m2_writers.py`, `path_guard.py`, `package_rules.py`, `train_cnn.py` | code of the earlier runs that the runner and its suite import, and `train_cnn.py`, imported by `nuaa_folds_run.py` only for frame extraction (not used here); unchanged copies |
| `livenesslab/delivery/bnrecal/nuaa_bn_recalibration.json`, `livenesslab/delivery/sessionablationrun/nuaa_session_ablation_run.json`, `livenesslab/delivery/foldsrun/nuaa_folds_run.json` | configurations of PR #27, of the parent run and of the folds run; unchanged copies |
| `test_logs/check_nuaa_bn_test.txt`, `test_logs/check_nuaa_bn_test_mutations.txt` | output of the suite and of the mutation check on macOS (Python 3.11, TensorFlow 2.15.0, NumPy 1.26.4), with the SHA-256 of the files tested |
| `test_logs/check_inputs_bntest.txt` | output of `check-inputs` on the private inputs, before the run |
| `SHA256SUMS` | SHA-256 of every other file of this folder |

Unchanged copies: byte for byte the files of `research/nuaa-saturation-control/livenesslab` (PR #27).
