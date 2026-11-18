# LivenessNet and ResNet-18 under early stopping, BatchNormalization and threshold (protocol draft, run table, definitions, technical trial)

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

**Status of this delivery.** A draft of the pre-registration, not the final one: the protocol (`PROTOCOL.md`), the
table of the planned runs generated from the configuration (`plan/`), the operational definitions as code with tests
(`livenesslab/scripts/pad_arch_definitions.py`, suite and mutation check), the sheet of the two external datasets
(`DATASETS.md`) and the technical trial, run (`trial/`). No test frame assembled or evaluated, no test score or metric
read; no external dataset used; no checkpoint saved. The only trainings are the two 3-epoch trainings of the trial.

## The technical trial

Run `20261118-ARCHTRIAL-3f059b7`, status **PASS** (`trial/status.json`, `trial/trial.json`, `trial/trial.md`,
`trial/run.log.txt`). It ran from commit `3f059b7` of our working repository with a clean tree (`dirty: false`); the
configuration was committed there before the trial (our own record). The code and configuration of this folder are
byte for byte those recorded by the run: `config_sha256` and `code_sha256` of `trial/trial.json`, and the headers of
the logs.

- Command: `python -I -B scripts/pad_arch_comparison.py trial --i-am-authorized`, from the root of the working
  repository (here: `livenesslab/`). Configuration `livenesslab/delivery/archcmp/pad_arch_comparison.json`, SHA-256
  `9264ec0201c148519b3cd07a5e8493ff49b25aadfbd2680636d0a4b0d9bcaab8`.
- Unit: NUAA, arm B, fold 1, seed 42; fitting 1,400 frames (700 bona fide, 700 attack, 4 subjects), dev 400 frames
  (200 and 200, 2 subjects). 3 fixed epochs per network, the recipe otherwise unchanged; then one inference-only pass
  over the fitting set and one over the dev.
- Device: CPU only (GPU hidden), Apple M4 (4 performance and 6 efficiency cores, 10 logical CPUs), 16 GB RAM,
  macOS 27.2.1; Python 3.11.16, TensorFlow 2.15.0, Keras 2.15.0, NumPy 1.26.4; TensorFlow thread settings not set.
- Batch: 16 for training (88 steps per epoch) and 16 for the inference passes.

| | LivenessNet | ResNet-18 |
|---|---|---|
| seconds per epoch, elapsed (epochs 1, 2, 3) | 26.8, 26.4, 28.4 | 69.0, 70.0, 69.2 |
| of which validation on the dev | 2.0, 1.5, 1.6 | 5.4, 5.2, 5.1 |
| process CPU seconds per epoch (all threads) | 189.6, 195.9, 199.7 | 560.3, 565.0, 571.4 |
| inference pass, fitting / dev (seconds) | 7.6 / 1.8 | 19.1 / 5.4 |
| CPU / elapsed ratio | 6.90 | 8.11 |

- Whole trial: started 2026-11-18T10:59:46+0100, finished 2026-11-18T11:05:23+0100; 336.7 s of elapsed time, against
  a cap of 60 minutes of elapsed time; process CPU time over all threads 2,555.9 s; peak memory (RSS) 2,249.6 MB.
  `status.json`, written after `trial.json`, records 336.8 s and 2,556.0 s.
- Check fixed before the trial: the LivenessNet history over the 3 epochs (`loss`, `accuracy`, `val_loss`,
  `val_accuracy`, `lr`) against epochs 1–3 of the accepted card of arm B, fold 1, seed 42 (run
  `20261106-SESSABLRUN-d999458`): **reproduced**, largest absolute difference 0 in each field; the assembled fitting
  and dev keys are equal to those of the card.
- The ResNet-18 history (`trial/trial.md`) is a technical output of the trial, not a result.
- Estimated hours of the full plan (`plan/run_plan.md`): NUAA, 48 trainings with their evaluation, **9.36–11.62 hours
  of elapsed time** on this machine (training 5.51–7.76, evaluation at most 3.86), that is **73.34–90.84 hours of
  process CPU time** over all threads. CelebA-Spoof and Replay-Attack, 12 trainings each at the proposed budget:
  1.38–1.94 hours of elapsed time each, for the training alone; their evaluation is not estimated until the test size
  is fixed.

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
  `status.json` and `trial.json` of the trial.

## Choices of the author

Not requested by the coordinator, stated as such in `PROTOCOL.md` (§9); the configuration, fixed before the trial,
labels two of them (`authorisation`, (i) and (ii)):

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
   balanced by class), official identity split × seeds 42, 43, 44; final numbers in the pre-registration.

## Limits of the trial and of the estimate

- The machine was in ordinary use during the trial: the load average at the start was about 4 according to `uptime`,
  a value read from the shell and not recorded by the runner.
- The estimate rests on 3 epochs of a single unit (fold 1, seed 42), the first epoch included (it carries the graph
  tracing); each training is costed between the fewest epochs possible and 12. Outside the estimate: saving the
  checkpoints, assembling the test, loading the inputs.
- CPU / elapsed ratio 6.90 (LivenessNet) and 8.11 (ResNet-18): the CPU hours of the plan (73.34–90.84) are many more
  than its elapsed hours (9.36–11.62). Every time limit of the protocol states its unit.
- The configuration holds the estimate written before the trial (`trial.estimate`: about 15 minutes, at most 40;
  ResNet-18 about 160 s per epoch from a FLOPs ratio of about 6; a peak of about 6–7 GB). The trial took 5.6
  minutes, ResNet-18 about 69 s per epoch (about 2.6 times LivenessNet) and the peak RSS was 2,249.6 MB. The text is
  left as it was because the configuration is pinned by SHA-256 in the trial record; the costs of the plan use only
  the measured rates.
- The logs are from macOS (Python 3.11.16, TensorFlow 2.15.0, NumPy 1.26.4); none of these commands has been run on
  Linux by us.

## What was fixed before the trial and what after

Code, configuration and the draft of the protocol with its definitions were in commit `3f059b7` of our working
repository (10:59:39 +0100); the trial started at 10:59:46 (our own records, not verifiable from outside). After the
trial, the four placeholders of the protocol (ResNet-18 parameter count, table of the planned runs, trial table,
cost summary) were filled with the generated values; after an independent check, the text was then shortened and
corrected. Additions after the trial that are not mere cuts: the consequence in the plan of a LivenessNet history
that differs from the accepted card (§7); the secondary per-seed reading (§6); the sentence on the caps of the runs
of the plan in elapsed hours and the CPU-hour estimates (§8); the section of the choices of the author (§9); the
sentence on the training order of the plan (§7); the example of scale in §8; the note that totals are computed before
rounding (§4); the wording of the external datasets in §5; in the limits, "not guaranteed bit-deterministic" with the
outcome of the trial in place of "not bit-deterministic by declaration" (§10). The trial assembled no test frame and
read no test score or metric (`test_read` of `trial/trial.json`): the only information on the models seen before
these additions is the fitting and dev history published in `trial/trial.md`.

## Verification without images or weights

From `livenesslab/` of this folder, with Python (3.11 here) and NumPy; TensorFlow 2.15 (Keras 2.15) on the CPU for
the tests that build or train a Keras model.

```bash
python -I -B scripts/check_pad_arch_comparison.py
```

Expected (about 30 s; 31 s here, `test_logs/check_pad_arch_comparison_from_this_folder.txt`):

```
tests run: 83, failures: 0, errors: 0, skipped: 0
RESULT: PASS
```

Synthetic data only (8×8 frames, tiny worlds in a temporary folder; ResNet-18 built at 256×256 only to count its
parameters): no real frame, checkpoint or score is read, so no test is skipped for missing data. The test of the
delivered configuration reads `livenesslab/delivery/` of this folder. Without TensorFlow, the 13 tests that build or
train a Keras model are skipped with the declared reason, `tests run: 83, failures: 0, errors: 0, skipped: 13` and
`RESULT: PASS` (`test_logs/check_pad_arch_comparison_without_tensorflow.txt`, run in the working repository with
TensorFlow hidden by an import blocker); the same 13 are skipped, with their own declared reason, if
`livenesslab/src/livedetection/src/architectures.py` is absent.

```bash
python -I -B scripts/check_pad_arch_comparison_mutations.py
```

Expected (about 15 minutes here, from 11:10 to 11:25 CET; the same per-mutant lines as the log of the working
repository, `test_logs/check_pad_arch_comparison_mutations_from_this_folder.txt`):

```
mutations killed: 134/134
control mutations survived: 4/4 (1 control, 3 declared redundant)
RESULT: PASS
```

134 mutants of `pad_arch_definitions.py`, `pad_arch_models.py` and `pad_arch_comparison.py`, one at a time, in a
temporary copy; the unmutated copy gives `tests run: 83, failures: 0, errors: 0, skipped: 1` (the delivered
configuration is not copied: its test is skipped with the declared reason). The mutation check needs TensorFlow;
without it, the 13 Keras tests of the unmutated copy are skipped for a reason that is not a declared missing input,
and it stops with `RESULT: FAIL (the unmutated copy skips tests for a reason that is not a declared missing input)`
(read from the code, not run).

```bash
python -I -B scripts/pad_arch_comparison.py plan --trial-run ../trial --out-dir <new folder outside the repository>
```

Expected, in about a second: `plan: 12 rows, 4 with every number; files run_plan.csv, run_plan.md in <folder name>`
and `RESULT: PASS (plan)`, with the two files byte for byte equal to `plan/run_plan.csv` and `plan/run_plan.md`
(`test_logs/plan_from_this_folder.txt`: once as is, once with TensorFlow and OpenCV hidden; only NumPy is needed). The
command checks the pinned code against the configuration and the trial against its `status.json` and the configuration
before writing.

From this folder:

```bash
shasum -a 256 -c SHA256SUMS
```

Expected: 35 lines `OK`.

`check-inputs` and `trial` need the restricted inputs (M2 manifests, cache, extracted frames, the accepted runs) and
cannot be run from this folder; their outputs are `test_logs/check_inputs_archcmp.txt` and `trial/`. The four logs
from the working repository were produced before commit `3f059b7` ("uncommitted working files" in their headers); the
SHA-256 in their headers are those of the files of this folder.

## Contents

36 files: 35 listed in `SHA256SUMS`, plus `SHA256SUMS`.

| path | what it is |
|---|---|
| `README.md` | this file |
| `PROTOCOL.md` | the protocol, draft of the pre-registration: question, what stays fixed, factors, planned runs, cells already measured, definitions, technical trial, cost, limits |
| `DATASETS.md` | CelebA-Spoof and Replay-Attack: official source, identity split, conditions of use, local state, missing accesses |
| `plan/run_plan.md`, `plan/run_plan.csv` | the table of the planned runs, generated by `plan` from the configuration and the trial |
| `trial/status.json`, `trial/trial.json`, `trial/trial.md`, `trial/run.log.txt` | status, record, report and log of the trial (`run.log` renamed) |
| `livenesslab/scripts/pad_arch_comparison.py` | the runner: `check-inputs`, `trial`, `plan` |
| `livenesslab/scripts/pad_arch_definitions.py` | the operational definitions: one-class run at 0.5, comparison, sign and change of sign, leave-one-subject-out |
| `livenesslab/scripts/pad_arch_models.py` | ResNet-18 and the parameter register of both networks |
| `livenesslab/scripts/check_pad_arch_comparison.py` | the synthetic suite (83 tests) |
| `livenesslab/scripts/check_pad_arch_comparison_mutations.py` | the mutation check (134 mutants, 1 control, 3 declared redundant) |
| `livenesslab/scripts/nuaa_folds_run.py`, `nuaa_session_ablation_run.py`, `nuaa_bn_recalibration.py`, `nuaa_patience_verify.py`, `m2_writers.py`, `path_guard.py`, `package_rules.py`, `check_nuaa_folds_run.py`, `check_nuaa_session_ablation_run.py`, `train_cnn.py` | code of the earlier runs that the runner and its suite import, and `train_cnn.py`, imported by `nuaa_folds_run.py` only for frame extraction (not used here); unchanged copies, byte for byte those already published in `research/nuaa-bn-test/livenesslab/scripts/` and, for `nuaa_patience_verify.py`, in `research/nuaa-constant-lr-control/livenesslab/scripts/` |
| `livenesslab/delivery/archcmp/pad_arch_comparison.json` | the configuration, a closed set: task, declarations, accepted runs, pinned code, architectures, training, trial, check, definitions, plan, outputs, `test` |
| `livenesslab/delivery/sessionablationrun/nuaa_session_ablation_run.json`, `livenesslab/delivery/patienceverify/nuaa_session_ablation_run_patience12.json` | the configurations of the accepted runs at patience 4 and 12, pinned by SHA-256 in the configuration; unchanged copies (as in `research/nuaa-bn-test/` and `research/nuaa-constant-lr-control/`) |
| `livenesslab/src/livedetection/src/architectures.py` | `src/architectures.py` of this repository, byte for byte (SHA-256 `a83c02b0788bd6f98262abf00f5fe970abed501f231edba348820482aea64d30`, pinned in the configuration), at the path where the working repository holds it |
| `test_logs/check_pad_arch_comparison.txt`, `check_pad_arch_comparison_without_tensorflow.txt`, `check_pad_arch_comparison_mutations.txt` | suite, suite without TensorFlow and mutation check in the working repository, before the trial |
| `test_logs/check_inputs_archcmp.txt` | `check-inputs` on the restricted inputs, before the trial |
| `test_logs/check_pad_arch_comparison_from_this_folder.txt`, `check_pad_arch_comparison_mutations_from_this_folder.txt`, `plan_from_this_folder.txt` | the suite, the mutation check and `plan` run from `livenesslab/` of this folder |
| `SHA256SUMS` | SHA-256 of every other file of this folder |

## What follows

The final pre-registration, after the coordinator's observations on this protocol; then the trainings of both
networks on NUAA; the NUAA test according to the pre-registration; the external datasets after the accesses are
confirmed.
