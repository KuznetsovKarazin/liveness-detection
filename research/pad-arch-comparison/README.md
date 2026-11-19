# LivenessNet and ResNet-18 under early stopping, BatchNormalization and threshold (final pre-registration of the NUAA part: protocol, run table, definitions, technical trial)

Asked by the coordinator on the C2 card on 17 November 2026, after merging PR #28 (`616c3a8`): for the article, to
verify whether early stopping, BatchNorm and the threshold change which PAD network comes out better. In a new pull
request towards `research/pad-2026` starting from `616c3a8`: a short protocol and a table of the planned runs for
LivenessNet and ResNet-18, both from scratch, with the same subject splits, seeds and preprocessing, with patience
4/12, ordinary BN or BN re-estimated on the fitting set only, and threshold 0.5 or the EER threshold estimated on the
source dev; how to count the runs that, at threshold 0.5, assign the whole dev to one class when both are present; for
the comparison of the networks, the ACER difference (ResNet-18 minus LivenessNet), how to recognise a change of sign,
also reporting ties, subject intervals, APCER and BPCER; a short technical trial per network on NUAA, arm B, fold 1,
seed 42, fitting and dev only, a few epochs, reporting the command or configuration, the device, the batch, the
seconds per epoch and the estimated hours of the full plan; for CelebA-Spoof and Replay-Attack, the official source,
the identity split and the missing accesses. The tests and the external runs will follow the pre-registration and the
confirmation of the accesses.

On 19 November 2026, on the C2 card (about PR #29), the coordinator accepted the Apple M5 Pro and asked: for the whole NUAA plan, 10
intra-op and 10 inter-op TensorFlow threads; to complete PR #29 as proposed, with the threads set and recorded by the
runner, `first_differing_epoch` corrected with a test, the official three-epoch trial on fitting and dev repeated, a
check that the runner reproduces the trial run with the 10 threads set through environment variables, and the times
updated in the protocol; a check of the frame identifiers associated with the scores, both between the two networks
and between the conditions, with a test for each case confirming that the comparison stops if the identifiers differ
for the same subject and class; for the primary analysis, patience 4 and threshold 0.5, the ACER gap (ResNet-18 minus
LivenessNet) with ordinary BN and with BN re-estimated on the fitting data only, the first gap subtracted from the
second, and the per-seed results reported separately. The 48 trainings start after the coordinator's verification, the
merge and the coordinator's go-ahead.

**Status of this delivery.** The final pre-registration of the NUAA part: the protocol (`PROTOCOL.md`), the table of
the planned runs with the order and caps of the runs, generated from the configuration and the official trial
(`plan/`), the operational definitions as code with tests (`livenesslab/scripts/pad_arch_definitions.py`, suite and
mutation check), the sheet of the two external datasets (`DATASETS.md`, unchanged; they will be pre-registered
separately after access is confirmed), the official technical trial (`trial/`), its comparison with the trial run with
environment variables (`trial_comparison/`) and the earlier trials (`trials_previous/`). No test frame assembled or
evaluated, no test score or metric read; no external dataset used; no checkpoint saved. The only trainings are the
3-epoch trainings of the technical trials. Everything of this delivery is in this folder: the trials write no
restricted output, and nothing is uploaded to the shared drive. Access to the external datasets: `DATASETS.md` gives
the state as of 17 November 2026; according to the coordinator's cards as of 19 November 2026, the reminder to the
university offices was sent on 17 November and no reply is recorded.

## What changed since commit `e004b20` (PR #29 as opened on 18 November)

- Requested by the coordinator on 19 November:
  - threads: 10 intra-op and 10 inter-op, `training.threads` of the configuration, set, read back and recorded by the
    runner (`pad_arch_comparison.set_threads`; field `threads` of `trial/status.json`, `trial/trial.json` and
    `trial/trial.md`; `PROTOCOL.md` §7);
  - `first_differing_epoch` corrected with a test (`pad_arch_comparison.history_differences`; test
    `TestQFirstDifferingEpoch`); the records now also give `epoch_of_max_abs_difference`;
  - the official trial repeated: `trial/` (`20261119-ARCHTRIAL-b8f7cca`, Apple M5 Pro);
  - the check against the trial run with the threads set through environment variables: `trial_comparison/`
    (reproduced, difference 0 for both networks), by the new read-only command `compare-trials`;
  - the times updated: `PROTOCOL.md` §4, §7, §8 and `plan/`, regenerated from the new trial;
  - the frame identifiers (`sample_id`) checked between the two networks and between the conditions
    (`pad_arch_definitions.prepare_units` and `same_frames`; tests in `TestBFrameIds` and
    `TestCFrameIdsBetweenConditions`, one for each case with the same subjects and classes and one identifier
    different); `PROTOCOL.md` §6;
  - the primary analysis: `pad_arch_definitions.primary_analysis`, per seed apart; `PROTOCOL.md` §6 (tests in
    `TestNPrimaryAnalysis`).
- Promised in the draft: the per-seed reading as code (`pad_arch_definitions.per_seed_reading`); the training order
  of the plan (`plan/run_order.csv`); a cap for each run in elapsed hours with the estimate in CPU hours apart
  (`plan/run_caps.csv`).
- Our defects, corrected: `first_differing_epoch` (above); the names of the core groups in the machine record
  (`performance_cores` / `efficiency_cores`, wrong on the Apple M5 Pro, now `core_levels` with the names given by the
  system). The records written before the corrections are not modified: errata in `trials_previous/README.md`.
- The earlier trials, with their errata: `trials_previous/` (each with the reason why it is there).
- Choices of the author added, stated as such (this README, "Choices of the author", items 7–15; `PROTOCOL.md` §9).
- A tool to run the suite as without TensorFlow: `livenesslab/tools/run_without_tensorflow.py`.

## The technical trial

Run `20261119-ARCHTRIAL-b8f7cca`, status **PASS** (`trial/status.json`, `trial/trial.json`, `trial/trial.md`,
`trial/run.log.txt`; `trial/conditions.txt`, written by the author around the trial, not by the runner). It ran from commit `b8f7cca` of our working repository with a clean tree (`dirty: false`); the
configuration was committed there before the trial (our own record). The code and configuration of this folder are
byte for byte those recorded by the run: `config_sha256` and `code_sha256` of `trial/trial.json`, and the headers of
the logs.

- Command: `python -I -B scripts/pad_arch_comparison.py trial --i-am-authorized`, from the root of the working
  repository (here: `livenesslab/`), with no `TF_NUM_` variable in the environment. Configuration
  `livenesslab/delivery/archcmp/pad_arch_comparison.json`, SHA-256 `5a83325e13d29d1a67dcda4f129838d19ec39b8f119f5de647693b68c5128b0a`.
- Unit: NUAA, arm B, fold 1, seed 42; fitting 1,400 frames (700 bona fide, 700 attack, 4 subjects), dev 400 frames
  (200 and 200, 2 subjects). 3 fixed epochs per network, the recipe otherwise unchanged; then one inference-only pass
  over the fitting set and one over the dev.
- Device: CPU only (GPU hidden), Apple M5 Pro (cores as named by the system: 5 Super, 10 Performance; 15 logical CPUs), 24 GB RAM, macOS 27.2.1; Python 3.11.16, TensorFlow 2.15.0, Keras 2.15.0, NumPy 1.26.4.
- TensorFlow threads: intra-op 10 and inter-op 10 requested and set by the runner, effective (read back from `tf.config.threading`) 10 and 10; `TF_NUM_INTRAOP_THREADS` and `TF_NUM_INTEROP_THREADS` absent (`threads` of `trial/status.json` and `trial/trial.json`).
- Batch: 16 for training (88 steps per epoch) and 16 for the inference passes.

| | LivenessNet | ResNet-18 |
|---|---|---|
| seconds per epoch, elapsed (epochs 1, 2, 3) | 13.8, 13.7, 13.9 | 37.6, 36.8, 37.1 |
| of which validation on the dev | 0.9, 0.8, 0.8 | 3.0, 2.9, 2.9 |
| process CPU seconds per epoch (all threads) | 112.6, 114.7, 115.9 | 342.9, 341.1, 343.3 |
| inference pass, fitting / dev (seconds) | 3.4 / 1.0 | 10.8 / 3.1 |
| CPU / elapsed ratio | 8.11 | 9.09 |

- Whole trial: started 2026-11-19T21:04:47+0100, finished 2026-11-19T21:07:47+0100; 180.0 s of elapsed time, against a cap of 60 minutes of elapsed time; process CPU time over all threads 1,534.1 s (both from `trial/status.json`); peak memory (RSS) 5,299.1 MB.

- Check fixed before the trial: the LivenessNet history over the 3 epochs (`loss`, `accuracy`, `val_loss`,
  `val_accuracy`, `lr`) against epochs 1–3 of the accepted card of arm B, fold 1, seed 42 (run
  `20261106-SESSABLRUN-d999458`): **reproduced**, largest absolute difference 0 in each field, no first differing
  epoch; the assembled fitting and dev keys are equal to those of the card.
- Check requested by the coordinator (`trial_comparison/`): against `20261119-ARCHTRIAL-b8e78b2` (code of PR #29),
  **reproduced**: difference 0 in the five fields over the 3 epochs, for LivenessNet and for ResNet-18, computed from
  the two records. Limit: the record of that trial (format `pad-arch-trial/1`) gives the threads as 0 (not set through
  `tf.config.threading`) and does not record environment variables; that `TF_NUM_INTRAOP_THREADS` and
  `TF_NUM_INTEROP_THREADS` were set to 10 on its command line rests on a conditions file written by the author around
  the trial (`trial_comparison/reference_conditions.txt`, copied from our private log; the comparison reports its
  SHA-256), not on the record; the code checked only that the file holds the commit of that record and one line
  setting both variables to integers equal to the required 10. With the threads not set by the runner on the same machine (15
  logical CPUs, `20261118-ARCHTRIAL-0858ae7`) the LivenessNet history is not reproduced (`trials_previous/`).
- The ResNet-18 history (`trial/trial.md`) is a technical output of the trial, not a result.
- Estimated hours of the full plan (`plan/run_plan.md`): NUAA, 48 trainings with their evaluation, **5.03–6.22 hours of elapsed time** on this machine (training 2.90–4.09, evaluation at most 2.13), that is **44.70–55.20 hours of process CPU time** over all threads; in two runs of 24 trainings, patience 4 (1.92–3.11 elapsed hours) and patience 12 (3.11), each with a cap of 5.00 hours of elapsed time (`plan/run_caps.csv`). CelebA-Spoof and Replay-Attack, 12 trainings each at the proposed budget: 0.73–1.02 hours of elapsed time each, for the training alone; their evaluation is not estimated until the test size is fixed.

## What the trial reads of the data

Summarised from the field `test` of the configuration (the runner requires its exact text; the same text is
`test_read` of `trial/trial.json`): no test frame assembled or evaluated, no test score or metric read.

- `check-inputs` and `trial` read the five M2 manifests (`official_train`, `official_test`, `cnn_train`, `cnn_es`,
  `cnn_test`), to index the cache and to check the rows of the extracted frames against the official lists;
  `official_test` and `cnn_test` are NUAA's official lists, not the test role of the folds.
- The frame files are read whole: the cache is hashed whole, with its label and subject members read (only `trial`
  loads its image arrays, whole); the extracted frames file is loaded whole, with the SHA-256 of every frame. These
  files also hold frames of the held-out subjects, but only the fitting and dev frames of arm B, fold 1 are
  assembled, by `trial` only; an explicit check refuses any other role.
- Of the earlier runs: of the cells run, `status.json`, the public and restricted SHA256SUMS, its configuration and
  the cell manifests of the arms and of the common dev; of the folds run, `status.json`, the restricted SHA256SUMS and
  `extracted_frames.csv` (no public file opened); of the accepted run at patience 4, `status.json`, the restricted
  SHA256SUMS and one training card (arm B, fold 1, seed 42); of the accepted run at patience 12, `status.json` and the
  restricted SHA256SUMS; the configurations of the two accepted runs and the LivenessNet architecture file.
- The fold schema (`fold<k>_test.csv`, the test role of the folds) is never loaded; no `<model>_test_scores.csv`,
  public aggregate or run record of any run is opened.
- `plan` reads only the configuration, the configurations of the two accepted runs, the pinned code and the
  `status.json` and `trial.json` of the trial; `compare-trials` reads only the `status.json` and `trial.json` of the
  two trials it compares (and, if given, the conditions file).

## Choices of the author

Not requested by the coordinator, stated as such in `PROTOCOL.md` (§9); `authorisation` of the configuration labels
those that the code implements, (i)–(x):

1. Every training of the plan is new: 48 NUAA trainings, 24 per network. The accepted LivenessNet checkpoints of runs
   `20261106-SESSABLRUN-d999458` (patience 4) and `20261109-SESSABLRUN-6a50265` (patience 12) are not reused; they are
   only the reference of the history reproduction check, repeated in the plan on every epoch of the 24 LivenessNet
   trainings.
2. ResNet-18 with projection shortcuts where the dimension changes (option B of He et al., §3.3, as in the torchvision
   reference implementation); the 18-layer results of the paper (Table 2) use option A.
3. The sign of a condition is read on the interval of the three seeds together, unlike the reading rule of PR #21
   (the same non-zero sign for each seed, each per-seed interval excluding zero), which is reported next to it as a
   secondary per-seed reading.
4. A leave-one-subject-out sensitivity of the sign, in the form of the coordinator's N1 protocol of 24 October 2026
   (PR #10); it enters neither the sign rule nor the change-of-sign rule.
5. One NUAA arm, arm B (the arm named for the trial): folds 1–4 × seeds 42, 43, 44, 12 units.
6. For the external datasets, a proposed budget: fitting and dev subsampled to those of arm B (1,400 and 400 frames,
   balanced by class), official identity split × seeds 42, 43, 44; final numbers in their pre-registration.
7. The reading rule of the primary contrast, from its interval over the three seeds together, and its names
   (`PROTOCOL.md` §6).
8. The other six conditions and every comparison between conditions are secondary, descriptive analyses.
9. Units of the same split must hold the same frames in the same order, a frame identifier must belong to one split
   only, the units of one comparison must belong to one dataset, and a unit seed must be one of the seeds of the plan.
10. The order of the 48 trainings and their split into two runs, one per patience (`plan/run_order.csv`).
11. The rule of the caps (`plan/run_caps.csv`).
12. The earlier trials kept in this folder (`trials_previous/`).
13. The command `compare-trials` and its rule (difference 0, no tolerance).
14. In the primary analysis, the subject-bootstrap intervals of the two gaps and of the contrast, the same values for
    APCER and BPCER, and the per-seed values.
15. The threads applied to every TensorFlow command of the NUAA plan, trainings and evaluations, and contradicting
    environment variables refused before any work.

The number of threads was our proposal; the coordinator fixed it.

## Limits of the trial and of the estimate

- The load average at the start of the trial was 2.23 according to `uptime` (`trial/conditions.txt`, written by the
  author, not by the runner); it can move the times.
- The estimate rests on 3 epochs of a single unit (fold 1, seed 42), the first epoch included (it carries the graph
  tracing); each training is costed between the fewest epochs possible and 12. Outside the estimate: saving the
  checkpoints, assembling the test, loading the inputs.
- CPU / elapsed ratio 8.11 (LivenessNet) and 9.09 (ResNet-18): the CPU hours of the plan (44.70–55.20) are many more than its elapsed hours (5.03–6.22). Every time limit of the protocol states its unit.
- The configuration holds the estimate written before the trial (`trial.estimate`: about 4 minutes, at most 10; a
  peak of about 6 GB). The trial took 3.0 minutes and the peak RSS was 5,299.1 MB. The text is left as it was because the configuration is pinned by SHA-256 in
  the trial record; the costs of the plan use only the measured rates.
- The reproduction with 10 intra-op and 10 inter-op TensorFlow threads is verified on 3 epochs of one unit; the
  results depend on the number of threads and hold for this setting.
- The plan states that every training and evaluation runs on the machine of the official trial with the same threads
  (`plan.machine_rule`): a declared rule, which the runner of the 48 trainings will have to apply; today no code
  compares the machine.
- The logs are from macOS (Python 3.11.16, TensorFlow 2.15.0, NumPy 1.26.4); none of these commands has been run on
  Linux by us.

## What was fixed before the trial and what after

Code and configuration were in commit `b8f7cca` of our working repository (21:04:34 +0100). The trial started at 21:04:47 (our own records, not verifiable from outside). After the trial: `compare-trials` against the earlier trials, the plan regenerated from the trial, and the protocol amended (numbers filled from the generated files). The trial assembled no test frame and read no test score or metric (`test_read` of `trial/trial.json`): the only information on the models seen before the protocol was amended is the fitting and dev history of the trials (`trial/trial.md`, `trials_previous/`).

## Verification without images or weights

From `livenesslab/` of this folder, with Python (3.11 here) and NumPy; TensorFlow 2.15 (Keras 2.15) on the CPU for
the tests that build or train a Keras model.

```bash
python -I -B scripts/check_pad_arch_comparison.py
```

Expected (about 35 s here, `test_logs/check_pad_arch_comparison_from_this_folder.txt`):

```
tests run: 154, failures: 0, errors: 0, skipped: 0
RESULT: PASS
```

Synthetic data only (8×8 frames, tiny worlds in a temporary folder; ResNet-18 built at 256×256 only to count its
parameters): no real frame, checkpoint or score is read. The test of the delivered configuration reads
`livenesslab/delivery/` of this folder. Without TensorFlow, the 14 tests that build or train a Keras model or run TensorFlow in a subprocess are skipped with the declared reason ("TensorFlow is not installed: the tests that build or train a Keras model are skipped"): `tests run: 154, failures: 0, errors: 0, skipped: 14` and `RESULT: PASS` (`test_logs/check_pad_arch_comparison_without_tensorflow_from_this_folder.txt` and, in the working repository, `check_pad_arch_comparison_without_tensorflow.txt`, both with TensorFlow hidden by `tools/run_without_tensorflow.py`: `python -I -B tools/run_without_tensorflow.py scripts/check_pad_arch_comparison.py`). On a machine without TensorFlow the command above gives the same lines; if the configurations of `livenesslab/delivery/` are absent, the test of the delivered configuration is also skipped, with its own declared reason: `tests run: 154, failures: 0, errors: 0, skipped: 15` and `RESULT: PASS` (run once on a copy of this folder, not logged here).

```bash
python -I -B scripts/check_pad_arch_comparison_mutations.py
```

Expected (about 72 minutes here; the same per-mutant lines as the log of the working repository,
`test_logs/check_pad_arch_comparison_mutations_from_this_folder.txt`):

```
mutations killed: 267/267
control mutations survived: 4/4 (1 control, 3 declared redundant)
RESULT: PASS
```

267 mutants of `pad_arch_definitions.py`, `pad_arch_models.py` and `pad_arch_comparison.py`, one at a time, in a temporary copy; the unmutated copy gives `tests run: 154, failures: 0, errors: 0, skipped: 1` (the delivered configuration is not copied: its test is skipped with the declared reason). The mutation check needs TensorFlow; without it, the Keras tests of the unmutated copy are skipped for a reason that is not a declared missing input, and it stops with `RESULT: FAIL (the unmutated copy skips tests for a reason that is not a declared missing input)` (read from the code, not run).

```bash
python -I -B scripts/pad_arch_comparison.py plan --trial-run ../trial --out-dir <new folder outside the repository>
```

Expected, in a few seconds: `plan: 12 rows, 4 with every number; files run_plan.csv, run_plan.md, run_order.csv, run_caps.csv in <folder name>` and `RESULT: PASS (plan)`, with the four files byte for byte equal to those of `plan/` (`test_logs/plan_from_this_folder.txt`: once as is, once with TensorFlow and OpenCV hidden; only NumPy is needed). The command checks the pinned code against the configuration and the trial against its `status.json`, the configuration, the code and the threads before writing.

```bash
python -I -B scripts/pad_arch_comparison.py compare-trials --trial-run ../trial \
  --reference-run ../trials_previous/20261119-ARCHTRIAL-b8e78b2 \
  --reference-conditions ../trial_comparison/reference_conditions.txt --compare-out-dir <new folder outside the repository>
```

Expected: `compare-trials: 20261119-ARCHTRIAL-b8f7cca against 20261119-ARCHTRIAL-b8e78b2: reproduced; files trial_comparison.json, trial_comparison.md in <folder name>` and `RESULT: PASS (compare-trials)`, with `trial_comparison.json` and `trial_comparison.md` byte for byte equal to those of `trial_comparison/` (`test_logs/compare_trials_from_this_folder.txt`, which also regenerates, without `--reference-conditions`, every `trials_previous/<run>/comparison_with_official/`). It reads only the `status.json` and `trial.json` of the two trials and the conditions file; it needs neither TensorFlow nor the configuration.

From this folder:

```bash
shasum -a 256 -c SHA256SUMS
```

Expected: 73 lines `OK`.

`check-inputs` and `trial` need the restricted inputs (M2 manifests, cache, extracted frames, the accepted runs) and
cannot be run from this folder; their outputs are `test_logs/check_inputs_archcmp.txt` and `trial/`. The four logs of the working repository (`check_inputs_archcmp.txt`, `check_pad_arch_comparison.txt`, `check_pad_arch_comparison_without_tensorflow.txt`, `check_pad_arch_comparison_mutations.txt`) were produced with the files later committed as `b8f7cca` (three of them before the commit: "uncommitted working files" in their headers); the SHA-256 in their headers are those of the files of this folder.

## Contents

74 files: 73 listed in `SHA256SUMS`, plus `SHA256SUMS`.

| path | what it is |
|---|---|
| `README.md` | this file |
| `PROTOCOL.md` | the protocol, final pre-registration of the NUAA part: question, what stays fixed, factors, planned runs, cells already measured, definitions with the primary analysis, technical trial, threads and order, cost and caps, choices of the author, limits |
| `DATASETS.md` | CelebA-Spoof and Replay-Attack: official source, identity split, conditions of use, local state, missing accesses (unchanged since 18 November) |
| `plan/run_plan.md`, `plan/run_plan.csv` | the table of the planned runs, generated by `plan` from the configuration and the official trial |
| `plan/run_order.csv`, `plan/run_caps.csv` | the order of the 48 NUAA trainings in two runs, and the cap and estimates of each run |
| `trial/status.json`, `trial/trial.json`, `trial/trial.md`, `trial/run.log.txt` | status, record, report and log of the official trial (`run.log` renamed) |
| `trial/conditions.txt` | load, command and commit around the official trial, written by the author (not by the runner) |
| `trial_comparison/trial_comparison.json`, `trial_comparison.md` | `compare-trials`: the official trial against the trial run with environment variables |
| `trial_comparison/reference_conditions.txt` | command and conditions of that trial, copied from our private log (its SHA-256 is in the comparison) |
| `trials_previous/README.md` | the earlier trials: table, comparisons with the official trial, errata |
| `trials_previous/<run>/status.json`, `trial.json`, `trial.md`, `run.log.txt` | the public files of each earlier trial, unmodified |
| `trials_previous/<run>/comparison_with_official/trial_comparison.json`, `trial_comparison.md` | `compare-trials`, the official trial against that trial (for each earlier trial except `b8e78b2`, whose comparison is `trial_comparison/`) |
| `livenesslab/scripts/pad_arch_comparison.py` | the runner: `check-inputs`, `trial`, `plan`, `compare-trials` |
| `livenesslab/scripts/pad_arch_definitions.py` | the operational definitions: one-class run at 0.5, frame identifiers, comparison, sign and change of sign, per-seed reading, leave-one-subject-out, primary analysis |
| `livenesslab/scripts/pad_arch_models.py` | ResNet-18 and the parameter register of both networks (unchanged) |
| `livenesslab/scripts/check_pad_arch_comparison.py` | the synthetic suite (154 tests) |
| `livenesslab/scripts/check_pad_arch_comparison_mutations.py` | the mutation check (267 mutants, 1 control, 3 declared redundant) |
| `livenesslab/scripts/nuaa_folds_run.py`, `nuaa_session_ablation_run.py`, `nuaa_bn_recalibration.py`, `nuaa_patience_verify.py`, `m2_writers.py`, `path_guard.py`, `package_rules.py`, `check_nuaa_folds_run.py`, `check_nuaa_session_ablation_run.py`, `train_cnn.py` | code of the earlier runs that the runner and its suite import, and `train_cnn.py`, imported by `nuaa_folds_run.py` only for frame extraction (not used here); unchanged copies, byte for byte those already published in `research/nuaa-bn-test/livenesslab/scripts/` and, for `nuaa_patience_verify.py`, in `research/nuaa-constant-lr-control/livenesslab/scripts/` |
| `livenesslab/delivery/archcmp/pad_arch_comparison.json` | the configuration, a closed set: task, declarations, accepted runs, pinned code, architectures, training with the threads, trial, check, definitions with the frame identifiers and the primary analysis, plan with the runs and caps, outputs, `test` |
| `livenesslab/delivery/sessionablationrun/nuaa_session_ablation_run.json`, `livenesslab/delivery/patienceverify/nuaa_session_ablation_run_patience12.json` | the configurations of the accepted runs at patience 4 and 12, pinned by SHA-256 in the configuration; unchanged copies (as in `research/nuaa-bn-test/` and `research/nuaa-constant-lr-control/`) |
| `livenesslab/src/livedetection/src/architectures.py` | `src/architectures.py` of this repository, byte for byte (SHA-256 `a83c02b0788bd6f98262abf00f5fe970abed501f231edba348820482aea64d30`, pinned in the configuration), at the path where the working repository holds it |
| `livenesslab/tools/run_without_tensorflow.py` | runs a script with TensorFlow (and, with `--hide`, other modules) not importable, as on a machine without it; used for the logs "without TensorFlow" |
| `test_logs/check_pad_arch_comparison.txt`, `check_pad_arch_comparison_without_tensorflow.txt`, `check_pad_arch_comparison_mutations.txt` | suite, suite without TensorFlow and mutation check in the working repository |
| `test_logs/check_inputs_archcmp.txt` | `check-inputs` on the restricted inputs |
| `test_logs/check_pad_arch_comparison_from_this_folder.txt`, `check_pad_arch_comparison_without_tensorflow_from_this_folder.txt`, `check_pad_arch_comparison_mutations_from_this_folder.txt`, `plan_from_this_folder.txt`, `compare_trials_from_this_folder.txt` | the suite (with and without TensorFlow), the mutation check, `plan` and `compare-trials` run from `livenesslab/` of this folder |
| `SHA256SUMS` | SHA-256 of every other file of this folder |

`plan/run_plan.md` (generated text) cites `delivery/patienceverify/CONCLUSION.md`, a path of our working repository;
in the coordinator's repository the same document is `research/nuaa-patience-verification/CONCLUSION.md`.

## What follows

The coordinator verifies and merges. The command of the two NUAA runs is not part of this pre-registration: it will be
published, with its configuration, in a new pull request before the runs, and it will apply and record the machine and
thread rule. The 48 NUAA trainings start only after the coordinator's go-ahead; the NUAA test according to this
pre-registration, with the code of the evaluation published before the test is read; the external datasets after
access is confirmed.
