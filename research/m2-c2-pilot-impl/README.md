# C2 pilot, minimal implementation: `c2_pilot.py` and the `train_cnn.py` adaptations (synthetic tests only)

Implementation of the C2 pilot protocol (CASIA-FASD → NUAA; LivenessNet and AttackNet V2.2; seed 42), as asked by the
coordinator on 20 October 2026, in a pull request separate from the protocol (`research/m2-c2-pilot-protocol/`, PR #7,
merged in `research/pad-2026` at `5b52c02`). Configuration used: `research/m2-c2-pilot-protocol/c2_pilot_casia_to_nuaa.json`,
unchanged (SHA-256 `4065d853…f84f7a`).

**Nothing has been run on real data: no download, no training, no inference, and not even `check` on the restricted
manifests and caches.** Every result below comes from synthetic manifests and caches generated in temporary folders. The
real run waits for the coordinator's go-ahead after the check of the manifests (D-4 and the 21 restricted CSV are open).
`select_threshold.py`, C1, M1, R2 and the earlier packages are unchanged.

## Delivery status

Nothing uploaded; no run folder exists (neither restricted nor public). This folder holds code, instructions and the
logs of the tests run from this package layout (in `test_logs/`: a folder named `logs/` is ignored by the `.gitignore`
of the group repository).

## What is delivered

- `livenesslab/scripts/c2_pilot.py`: subcommands `check`, `train`, `score --set dev | source_test,target`, `select`,
  `apply` and `all` (the steps in order), each reading the configuration and a run folder.
- `livenesslab/scripts/train_cnn.py`: the three adaptations of the protocol, all optional; without them the behaviour is
  the one of the copy in `research/c1-livenesslab/` (see "Changes to `train_cnn.py`").
- `livenesslab/scripts/check_c2_pilot.py` (165 tests on synthetic data) and `livenesslab/scripts/check_c2_mutations.py`
  (262 declared mutations of `c2_pilot.py` and of the `train_cnn.py` adaptations).
- Unchanged copies, byte-identical to the published ones in `research/pad-2026` `5b52c02`: `select_threshold.py`,
  `path_guard.py`, `rgb_manifests.py`, `nuaa_manifests.py`, `package_rules.py`, `src/tesi_app/paths.py`, `__init__.py`
  (as in `research/m2-rgb-manifests/`); `src/tesi_app/evaluation.py`, `core.py`, `face.py` and `requirements.txt` (as in
  `research/m1-metrics/`).
- `test_logs/`: the outputs of the test commands below, run from this layout (section "Test logs").

## Steps

| step | reads | writes (run folder, restricted) | refuses |
|---|---|---|---|
| `check` | configuration, 4 manifests, 2 caches (no detector, no weights) | `check.json` | SHA-256 or count (rows, classes, subjects, videos) different from the configuration; invalid or non-canonical rows; fitting, early-stopping and source-test rows sharing a key, file SHA-256, subject key or video key; rows on the wrong side of the mirror split; a CASIA-FASD file or derived frame equal to a NUAA one; cache arrays not linked to the manifests; a pooled or existing checkpoint, another criterion or set in the configuration |
| `train` | configuration, `check.json`, 4 manifests, both caches (only the fitting and early-stopping frames go to the trainer) | `checkpoints/` (weights, Keras file, card), `train.json` | missing `--i-am-authorized`; `--epochs` other than the budget (12); a split that does not reproduce `cnn_train` / `cnn_es` by SHA-256; weights or cards failing the weights check; wall time above the cap |
| `score --set dev` | configuration, `check.json`, 4 manifests, both caches (only the dev frames are scored), `train.json` and `checkpoints/` (weights check) | `raw_scores_dev.csv`, `score_dev.json` | missing `--i-am-authorized`; weights check; scores non-finite or outside [0, 1] |
| `select` | configuration, `check.json`, 4 manifests (no cache), `train.json` and `checkpoints/` (weights check), and as scores only `raw_scores_dev.csv` with `score_dev.json` | `threshold.json` | any row whose key **or** file SHA-256 belongs to the source test or to NUAA; missing, duplicated or inconsistent rows; another checkpoint or configuration; a single-class dev |
| `score --set source_test,target` | configuration, `check.json`, 4 manifests, both caches (only the test frames are scored), `train.json` and `checkpoints/`, `threshold.json`, and `raw_scores_dev.csv` with `score_dev.json` (to recompute the threshold) | `raw_scores_source_test.csv`, `raw_scores_target.csv`, `score_tests.json` | missing `--i-am-authorized`; missing `threshold.json`, or one different from the selection recomputed on the raw dev scores; `threshold.json` changed during scoring |
| `apply` | configuration, `check.json`, 4 manifests (no cache), `train.json` and `checkpoints/`, `threshold.json`, the three raw score files with `score_dev.json` and `score_tests.json` | `scores_E1..E4.csv`, `scores_dev.csv`, `c2_table.json`, `c2_table.md`, `subject_dispersion.json`, `apply.json` | `threshold.json` changed since its SHA-256 was recorded before test scoring, or different from the recomputed selection; raw scores incomplete, duplicated, NaN, outside the manifest, with fields different from the manifest, or changed after scoring |

Every step also writes `status.json` (see "Behaviour on failure and on refusals").

Weights check (end of `train`, then before every `score`, `select` and `apply`): file SHA-256 of the `.h5` and `.keras`
equal to the card and to `train.json`; card fields (architecture, dataset, protocol, seed 42, `n_train`, `n_val`, split
by subject, learning rate, batch, `test_evaluation: disabled`, no `test` or `n_test`, split check passed); epochs within
the budget and restored epoch = epoch of minimum `val_loss`; the two models with different weights; weights whose
SHA-256 is listed in the configuration (official or pooled) refused.

Threshold: `threshold_counts`, `candidate_thresholds` and `select_threshold` imported from `select_threshold.py`
unchanged, criterion `eer`; refusal of test rows by key and SHA-256 (the NUAA session rule is not used: on CASIA-FASD
names it would pass silently). `threshold.json` holds values, aggregate dev counts, subject counts per class, dev AUC,
the non-informative flag of M1 and SHA-256 values (configuration, raw dev scores, checkpoints, selector); no identifier
(scanned). It is read back and recomputed after writing.

Score records: fields of `research/templates/score-schema.json` plus `model_id` and `file_sha256`; `score_attack` written
with `repr` (exact round trip); `threshold` and `prediction` are `null` in the raw files (declared phase rule) and are
written only by `apply`, with `prediction` = 1 iff `score_attack > threshold`. Missing values are the text `null`.
Table: per evaluation, at the dev threshold and at 0.5 (labelled "reference only"): `n`, bona fide, attacks, accepted
attacks, rejected bona fide, ties, APCER (pooled and per attack type with `metrics_by_attack_type`: CASIA-FASD warped /
cut / video from the video token, NUAA print), BPCER, ACER, subjects per class; AUC, EER and BPCER@APCER10 descriptive;
a missing class is `null`, never 0. `subject_dispersion.json` (restricted): per evaluation, the BPCER of each subject with
bona fide samples and the APCER of each subject with attacks at the frozen threshold, as sorted values with minimum,
median and maximum, without identifiers.

## Run folder, configuration and repository state

The run folder is named `YYYYMMDD-C2-seed42-<shortcommit>` and must be new at `check` (missing, empty, or holding only a
regular file `status.json` left by a failed `check`; any other entry, such as a folder, a link, `check.json` or a
`.DS_Store` added by the macOS Finder, makes it not new), or end with `-trial`. Every folder is checked with the shared `path_guard.py` (file identity, both directions;
its own rules, including the filesystem root, the home folder and their ancestors, are tested in
`check_export_m2_rgb.py`, and a run folder can never be one of those because of its name). The repository and its
ancestors and `data/`, `models/`, `results/` with every baseline in it, `export/`, `docs/`, `scripts/`, `src/`,
`delivery/`, `config/`, `.git` are refused (one `check_c2_pilot.py` test per folder); inside the repository only
`results/c2/` is accepted; the weights, data and delivered results folders are refused also when moved with
`LIVENESSLAB_*_DIR`. A non-trial run needs the approved configuration (its SHA-256 is a constant of the code), a clean
working tree and the commit of the run id. Every later step refuses a configuration, code (SHA-256 of `c2_pilot.py`,
`train_cnn.py`, `select_threshold.py`, `path_guard.py`, `rgb_manifests.py`, `nuaa_manifests.py`, `package_rules.py`,
`src/tesi_app/evaluation.py`, `src/tesi_app/paths.py`, and of the reference architectures
`$LIVENESSLAB_LIVEDETECTION/src/architectures.py` used by `train` and `score`, recorded as `absent` when the file is not
there) or, for a non-trial run, a commit different from those recorded in `check.json`.
Training and scoring need `--i-am-authorized`.

## Behaviour on failure and on refusals

On any failure the command prints `ERROR: <reason>` and `RESULT: FAIL (<command>)` and exits with 1, also for unexpected
errors (for example `OSError`, `KeyError` or a TensorFlow error: `ERROR: unexpected error <type>: ...`), and with 130 on
`KeyboardInterrupt`; on success `RESULT: PASS (<command>)` (if `status.json` cannot be written, the command fails). Files of a step are written together at the end (temporary files and renames; on any exception
no file of the group and no temporary file remain). `status.json` in the run folder records the result of the last
command (`result`, `command`, `step`, and for a failure `reason` and `outputs`):

- refusals before the run folder is accepted (argument errors, name, protected place, missing folder, non-trial folder
  that is not new at `check`): nothing is written or removed, not even `status.json`;
- as soon as the folder is accepted, before anything is removed, `status.json` becomes `"result": "RUNNING"`;
- refusals of the run identity (`check.json` missing, not passed or of another folder; configuration, code or commit different from
  `check.json`): the existing outputs stay as they are (they belong to the recorded run) and `status.json` becomes
  `"result": "FAIL"` with `"outputs": "untouched: run identity refused, the existing outputs are not current"`;
- every other refusal or error (authorisation, `--epochs`, inputs, weights, scores, threshold, unexpected errors,
  interruption): the
  step has already removed its own outputs and those of the later steps (also `checkpoints/`, through `path_guard.py`),
  and `status.json` becomes `"result": "FAIL"` with `"outputs": "removed: outputs of this step and of the later steps"`.

So no earlier `PASS` output can be read as current next to a failure: either it was removed, or `status.json` says
FAIL (or RUNNING, if the process was killed).

## Changes to `train_cnn.py`

Compared with the copy in `research/c1-livenesslab/livenesslab/scripts/train_cnn.py` (SHA-256 `4d655345…`), which stays
unchanged, only these top-level nodes differ: the module docstring, the new functions `out_dir_problem` and
`check_split`, the function `train` and the `__main__` block; every other node (preprocessing, sampling, subject names,
cache) is identical. In the development repository the exports accept exactly two versions of the file: the C1 one
(SHA-256 `4d655345…`) and this one (its exact SHA-256 is a constant, checked against the file by a test, plus the AST
comparison with the verified C1 copy).

1. `--out-dir` / `out_dir`: weights and card in that folder, without `manifest.json`. The folder is checked with
   `path_guard.py`: the weights and data folders (also when moved) and every protected folder of the repository are
   refused; inside the repository only `results/c2/` is accepted. The card adds `restored_epoch`, `restored_by`,
   `val_loss_history` and `keras_sha256`.
2. `split_check` (from code only): the subject split must reproduce the fitting and early-stopping manifests by file
   SHA-256 (same length and same multiset on each side), otherwise it stops; it requires the split by subject and the official protocol; the card adds `split_check`.
3. `--no-test-eval` / `test_eval=False`: no final evaluation on `X_test`; it requires `--out-dir` (without it the command
   stops before any work, so `models/weights/` and its `manifest.json` are never written without test metrics); the card
   has no `test` and `n_test` and adds `test_evaluation: disabled`.

Without these options the files, card fields and their order, and `manifest.json` are the ones of the C1 copy
(behavioural test with a fake TensorFlow module, plus a test of the command-line options and defaults).

## Differences from the approved protocol

- (a) `c2_pilot.py train` always disables the test evaluation (no option); `--no-test-eval` exists only in
  `train_cnn.py` and requires `--out-dir`.
- (b) New command-line options: `--run-dir`, `--source-manifests`, `--target-manifests`, `--source-cache`,
  `--target-cache` (defaults: the local M2 paths `results/m2_rgb/casia_fasd/manifests`, `results/m2/manifests`,
  `data/processed/`) and `--i-am-authorized`; the configuration is unchanged and holds no path.
- (c) Score records carry two fields more than `score-schema.json`: `model_id` and `file_sha256` (needed for the refusal
  by SHA-256).
- (d) The wall-time cap is checked after training: a longer training fails without results, it is not interrupted.
- (e) `-trial` folders accept an unapproved configuration and a modified working tree (recorded in `check.json`); inside
  the repository the only accepted place is `results/c2/`, while the protocol names `02_Experiments/C2/<run-id>/`
  (the copy to the shared folder is done later, by the coordinator).
- (f) A dev with a single class is an explicit error (M1 rule), no threshold.
- (g) Behaviour on refusals and `status.json`, as in "Behaviour on failure and on refusals".

## Commands (real run: not executed, needs the coordinator's go-ahead)

From inside `livenesslab/`, with the restricted copies of the M2 manifests and the two caches:

```bash
CFG=../../m2-c2-pilot-protocol/c2_pilot_casia_to_nuaa.json; RUN=<restricted folder>/YYYYMMDD-C2-seed42-<shortcommit>
IN="--source-manifests $CASIA_MANIFESTS --target-manifests $NUAA_MANIFESTS --source-cache $CACHE/casia_fasd_1500.npz --target-cache $CACHE/nuaa_1500.npz"
python scripts/c2_pilot.py check  --config $CFG --run-dir $RUN $IN                                  # no detector, no weights, no TensorFlow
python scripts/c2_pilot.py train  --config $CFG --run-dir $RUN $IN --epochs 12 --i-am-authorized
python scripts/c2_pilot.py score  --config $CFG --run-dir $RUN $IN --set dev --i-am-authorized      # CPU
python scripts/c2_pilot.py select --config $CFG --run-dir $RUN $IN
python scripts/c2_pilot.py score  --config $CFG --run-dir $RUN $IN --set source_test,target --i-am-authorized
python scripts/c2_pilot.py apply  --config $CFG --run-dir $RUN $IN
```

Dependencies: numpy for every step; scikit-learn for `train` and for `select` / `apply` (AUC through
`src/tesi_app/evaluation.py`, which also imports OpenCV); TensorFlow 2.15, OpenCV and the reference architectures
(`LIVENESSLAB_LIVEDETECTION`) for `train` and `score` only.

## Verification without data, weights or TensorFlow

From inside `livenesslab/`, with numpy and scikit-learn. TensorFlow is never imported: training goes through the real
`train_cnn.train` with a fake `tensorflow` module, scoring through a deterministic fake scorer, and the real scorer is
tested with a fake module too.

```bash
python scripts/check_c2_pilot.py
python scripts/check_c2_mutations.py
```

`check_c2_pilot.py`: a test skips only for a missing input, with the reason printed as `skipped <test>: missing input: ...`
(the approved configuration not found in `../../m2-c2-pilot-protocol/`, or no git work tree for the test that compares
`git status` and the files under `data/processed`, `models/weights`, `results` and `export` before and after the suite).
In a git work tree with `../../m2-c2-pilot-protocol/` next to this folder (as in `research/` of the group repository) the
last two lines are:

```
tests run: 165, failures: 0, errors: 0, skipped: 0
RESULT: PASS
```

Outside a git work tree the last three lines are:

```
skipped test_real_repository_untouched: missing input: git not available or the code is not in a git work tree: the repository-state sentinel is not compared
tests run: 165, failures: 0, errors: 0, skipped: 1
RESULT: PASS
```

`check_c2_mutations.py` copies the code to a temporary folder, applies 262 declared mutations (each pattern must occur
exactly once), runs the suite on each copy and counts a mutation only if the suite ran to its summary with at least one
failure or error (a skip is never a detection). The bare copy has neither the configuration nor git, so its first line is
`unmutated copy: scripts/check_c2_pilot.py exit 0 (tests run: 165, failures: 0, errors: 0, skipped: 2)`; the last two lines
are (about 12 minutes on a 10-core Mac with the default of 4 mutations in parallel, 30–60 minutes on a 2-core
machine; `--jobs N` sets how many run in parallel, `--jobs 1` one at a time; a copy whose suite does not finish within
30 minutes is reported as an invalid mutation, `timeout after 1800 s`, never as a detection):

```
mutations killed: 262/262
RESULT: PASS
```

Packages made absent: `--block` makes the listed packages raise `ModuleNotFoundError` when imported (the subprocess test
of `check` always blocks TensorFlow, Keras, h5py, OpenCV, scikit-learn and the detector, and records any attempt to import
them or to open a weights file). `python scripts/check_c2_pilot.py --block tensorflow,keras,h5py` prints, in a git work
tree next to the protocol folder, as its first line and its last three lines:

```
NOTE: imports blocked for this run: tensorflow, keras, h5py
blocked imports attempted: 0; blocked packages loaded: none
tests run: 165, failures: 0, errors: 0, skipped: 0
RESULT: PASS
```

With OpenCV blocked too (`--block tensorflow,keras,h5py,cv2`) the first two lines are
`NOTE: imports blocked for this run: tensorflow, keras, h5py, cv2` and `NOTE: OpenCV (cv2) not available: tesi_app.evaluation
imported with an empty cv2 stand-in (its image functions are not used)`, and the last three:

```
blocked imports attempted: 1; blocked packages loaded: none
tests run: 165, failures: 0, errors: 0, skipped: 0
RESULT: PASS
```

(the one attempt is the suite's own probe of `cv2`). `python scripts/check_c2_mutations.py --block tensorflow,keras,h5py,cv2`
ends with the same two lines `mutations killed: 262/262` and `RESULT: PASS`. Verified on macOS (Python 3.11.16, NumPy 1.26.4,
scikit-learn 1.3.2), on the default volume and on a case-sensitive APFS volume. Also run on Linux, in the official
`python:3.12` container (Python 3.12.15, aarch64), with NumPy 2.3.5 and scikit-learn 1.8.0 and without TensorFlow and OpenCV
actually installed, from this folder laid out as in `research/`: `tests run: 165, failures: 0, errors: 0, skipped: 0`
(after the OpenCV `NOTE` line), `RESULT: PASS`; `mutations killed: 262/262`, `RESULT: PASS`. Not run on x86_64.

## Test logs

Outputs of the commands above, run from this package layout (git work tree, `../../m2-c2-pilot-protocol/` next to it):

- `test_logs/check_c2_pilot.txt`: `python scripts/check_c2_pilot.py`;
- `test_logs/check_c2_pilot_block.txt`: `python scripts/check_c2_pilot.py --block tensorflow,keras,h5py,cv2`;
- `test_logs/check_c2_mutations.txt`: `python scripts/check_c2_mutations.py`.

## Not done, limits

- No step was run on the real manifests, caches or checkpoints; the real scorer (TensorFlow on CPU) and the real
  training were exercised only with fake modules. The reproduction of `cnn_es` by `GroupShuffleSplit` on the real data is
  the one already checked in M2 (`rgb_manifests.py reconstruct`); `train` repeats it by SHA-256 before fitting.
- Cache frames are linked to the manifest rows by sampling order and the label and group arrays (no file name in the
  caches); exact hashes only, no near-duplicate search.
- Subjects of `train_img` and `test_img` are numbered locally to each split: `check` verifies that fitting and early
  stopping come from `train_img` and the source test from `test_img`, and their disjointness by key, file SHA-256,
  subject key and video key; that the persons differ is documented by the protocol, not verified.
- The detector fallback rate stays "not available" (no detection pass).

## Files and integrity

This `README.md`, `SHA256SUMS`, `test_logs/` (3 files) and `livenesslab/` (`requirements.txt`; `scripts/`: `c2_pilot.py`,
`check_c2_pilot.py`, `check_c2_mutations.py`, `train_cnn.py`, `select_threshold.py`, `path_guard.py`, `rgb_manifests.py`,
`nuaa_manifests.py`, `package_rules.py`; `src/tesi_app/`: `__init__.py`, `paths.py`, `evaluation.py`, `core.py`,
`face.py`): 19 files listed in `SHA256SUMS`, plus `SHA256SUMS`. Running the tests creates `__pycache__` folders
(Python byte code of the imported modules), which the command below ignores. No file of this folder is ignored by the
`.gitignore` of the group repository (checked with `git check-ignore` and on a clone of the full tree; the logs are in
`test_logs/` because `logs/` is ignored there). From inside this folder:

```bash
shasum -a 256 -c SHA256SUMS && test -z "$(find . -type l)" && diff <(awk '{print $2}' SHA256SUMS | sed 's|^\*||' | sort) <(find . -type f ! -type l ! -name SHA256SUMS ! -path '*/__pycache__/*' | sed 's|^\./||' | sort) && echo COMPLETE
```
