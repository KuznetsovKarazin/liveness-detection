# NUAA subject folds: LivenessNet training run, 4 folds x 3 seeds on the CPU

Run authorised by the coordinator on the C2 card on 3 November 2026, after the approval of the fold schema (PR #16,
run `20261103-FOLDS-67b7f3d`): LivenessNet on the CPU, four subject folds, seeds 42/43/44, overall compute cap of
6 hours, the delivered configuration unchanged; input and cache checks first; preprocessing, split and recipe fixed;
threshold determined on the dev only and frozen before the test; every outcome kept and delivered (code,
configuration, scores, tables, logs, SHA-256). The question (schema, PR #16): when the capture session is represented
in training, does the N1 recipe generalise to subjects it has never seen? N2 already measured the opposite cell (the 9
training subjects in session 03); this run measures new subjects with every session represented in training.

Diagnostic design, not a benchmark and not a complete causal isolation (it mixes the three sessions in training and
does not preserve the official NUAA protocol); exploratory, because the NUAA session-03 results of C1, M1, N1 and N2
are already known. Pre-commitment: every rule below was fixed before any training; every number is reported whatever
its value.

## What it does and what it does not do

- Trains 12 LivenessNet models (4 folds x 3 seeds) on the CPU with the N1 recipe unchanged, scores the held-out
  subjects of each fold with its model, and aggregates. Nothing else: no other architecture, no other session
  direction, no selection of fold, seed or threshold on a result.
- Extracts, once, the 6,614 official NUAA frames that are not in the training cache, with the frozen cache extractor
  (`face.py` of commit 0ac7143) and the unchanged chain of `train_cnn.preprocess`, after reproducing the 6,000 cache
  frames bit for bit (the N2 cache check, repeated).
- Writes the dev threshold of every model before any test frame is scored (the test frames are assembled only
  afterwards), and records the SHA-256 of the threshold file, read back, in every score row.
- Verifies the approved schema itself: `delivery/folds/nuaa_subject_folds.json` must have the SHA-256 fixed in the
  code and in the configuration (`d1d43c14...`), and its seeds, cap, threshold criterion, training recipe, bootstrap
  numbers and sampling counts must equal those of this run.
- Does not read or write the accepted runs (C1, M1, M2, N1, N2, crop check, schema, step zero); the original images,
  the cache and the weights stay local; the restricted outputs (per-frame scores, the per-subject table, the list of
  extracted frames with their detector, weights, thresholds, cards) go to the shared folder only.

## Delivery status

- Run `20261103-FOLDSRUN-e174812` (clean commit `e174812` of the development repository): `public/` copied here as
  `tables/` (its `SHA256SUMS` included: it is the anchor recorded in `status.json` and in the
  restricted folder); the run folder (`public/`, `restricted/` with 12 cards, 12 thresholds, 12 dev
  score files, 12 test score files, 12 checkpoints, `per_subject.csv`, `extracted_frames.csv`,
  `public_SHA256SUMS.sha256`, `SHA256SUMS`; `status.json`; `run.log`) uploaded on 2026-11-04 to the
  shared N1 folder (provisional destination, as for N2) as `20261103-FOLDSRUN-e174812/`.
- The test suite was amended after the run (commit 45d8176: the SIGTERM test fails cleanly when the
  handler is missing, instead of killing the suite); the script, the configuration and the shared
  modules are those of the run (`code_sha256` in `tables/run_record.json`).
- The configuration (`livenesslab/delivery/foldsrun/nuaa_folds_run.json`, SHA-256 in `SHA256SUMS`) was written and
  verified with `check-inputs` before the run (log in `test_logs/check_nuaa_folds_run.txt`).

## Inputs (every one pinned by SHA-256 in the configuration and verified before anything is written)

| Input | What is checked |
|---|---|
| The five M2 manifests (`results/m2/manifests/`, package 20261006-M2-seed42-36c677a) | SHA-256, rows and counts per class and subject, declared sessions, no duplicated key or file SHA-256, official lists disjoint, cache manifests inside the official lists with the same row |
| The twelve fold manifests of the schema run `20261103-FOLDS-67b7f3d` (`restricted/fold<k>_<role>.csv`) | SHA-256 of the restricted `SHA256SUMS` (anchor) and of every file against it; 2,100 fitting, 600 dev and 3,242/3,114/3,439/2,819 test rows; fold and role of every row; every key in the official lists with the same label, subject, session, SHA-256 and size; `in_cache` equal to the membership in cnn_train/cnn_es/cnn_test; roles disjoint by subject and by key; dev of fold k = test subjects of fold k mod 4 + 1; fitting = the subjects of the other two folds; the test folds partition the 12,614 images |
| The training cache `data/processed/nuaa_1500.npz` | SHA-256 `dc286dc3...`, members and shapes read from the `.npy` headers without pickle, labels and subjects equal to the manifest rows in sampling order (spread per class) |
| The frozen extractor `delivery/n2/face_0ac7143.py`, the RetinaFace resources, the Haar cascade, the reference `create_datasets.py` and `architectures.py` | SHA-256 |
| The 12,614 original images (`data/NUAA/images`, layout `images`) | every file located, regular, with the size and SHA-256 of its manifest row (a few seconds) |
| The environment | OpenCV 4.11.0 (the version under which the cache reproduces) and TensorFlow 2.15.0 installed (read from the package metadata without importing it); TensorFlow not imported during the checks; the variables `LIVENESSLAB_SILENT_FACE` and `LIVENESSLAB_LIVEDETECTION` unset (they would redirect the detector resources or the reference code); GPU hidden and the visible devices equal to `["CPU"]` before any training or scoring |
| The extracted-frames file | must not exist yet (`data/processed/nuaa_folds_frames.npz` is never overwritten: move a previous one away) |

## Steps of `run` (one process, in this order; every step timed)

1. `check-inputs`: everything above; nothing written.
2. `cache-check`: each of the 6,000 cache frames reproduced from its original with the frozen extractor and compared
   bit for bit (the N2 check: worst absolute difference and a `DIFF` line per differing frame); detector used
   (RetinaFace, Haar, none) counted and required equal to the N2 cache check (train: Haar 51, RetinaFace 2,949; test:
   Haar 3, RetinaFace 2,996, none 1, anchored in the configuration); any difference, duplicated frame or frame shared
   between X_train and X_test is FAIL.
3. `extract`: the 6,614 frames outside the cache, extracted once with the same chain and saved to
   `data/processed/nuaa_folds_frames.npz` (keys sorted, uint8 256x256x3; never overwritten); SHA-256 of the file and of
   every frame, detector per frame (restricted `extracted_frames.csv`). Identical frames (exact pixel hash) across
   different subjects, or identical to a cache frame of another subject, are FAIL; identical frames of the same subject
   (consecutive frames of one capture can coincide) are counted and reported.
4. `train`, for every fold k and seed: fitting (2,100 frames) and dev (600) assembled from the fold manifests in key
   order, from the cache or the extracted file; the N1 recipe, line by line: Adam (legacy, learning rate 0.001,
   clipnorm 1.0), categorical crossentropy on one-hot labels, class weights balanced on the fitting set, augmentation on
   the fitting set only, tf.data shuffle(4096, seed), batch 16, at most 12 epochs, EarlyStopping on the dev val_loss
   (patience 4, restore_best_weights), ReduceLROnPlateau (0.5, patience 2, min 1e-6), set_weights of the best epoch
   when the epochs end first; seed for random, numpy and TensorFlow; CPU only; `clear_session` before every model and
   the arrays released after `fit` (memory only, the recipe is unchanged). Checkpoint `.keras` saved through a
   temporary name and renamed (restricted) with SHA-256; card with the history per epoch, the restored epoch (the
   minimum of val_loss) and the seconds. CPU non-determinism is declared as in N1: the seeds fix the initialisation,
   the augmentation and the batch order, but parallel reductions on the CPU may still give minimal differences
   between two runs with the same seed.
5. `threshold`: dev frames scored with the saved checkpoint (CPU, batch 16, BatchNorm in inference); val_loss
   recomputed and compared with the recorded minimum (tolerance 1e-3 absolute or 1 % relative, as in N1); EER
   threshold of the M1 protocol (candidates = distinct dev scores plus one value below the minimum; attack if score >
   value; minimum of |fp x n_attack - fn x n_bona_fide|, ties to the highest value) written to
   `fold<k>_seed<s>_threshold.json`, read back, SHA-256 recorded; only then
6. `score`: every frame of the held-out subjects scored; decision at 0.5 (`05`) and at the frozen dev threshold
   (`dev`); the threshold SHA-256 and the checkpoint SHA-256 in every row.
7. `aggregate`: see below; public files scanned for identifiers and forbidden text; checksums.

Wall-time cap: 360 minutes, checked after the cache check and the extraction, before every training and every
scoring, at the end of every epoch, before the aggregation and before the final writes;
the elapsed time is the larger of the monotonic clock and the wall clock (on macOS the monotonic clock does not
advance during sleep; both are recorded per step). When exceeded the run stops, the outputs are removed and
`status.json` is FAIL (protocol: no result, reported as a failed run). `SIGTERM` and `SIGHUP` stop the run the same
way; a `status.json` left at RUNNING without a live process means a failed run.

## Metrics and aggregation

Per fold, seed and session stratum of the held-out subjects (01, 02, 03 and all; NA where a class is absent):
n, bona fide and attacks, subjects, rejected bona fide and accepted attacks, APCER, BPCER and ACER at 0.5 and at the
dev threshold, AUROC (Mann-Whitney with ties 0.5, score_attack as the separator). Per fold: median and range over the
three seeds. Pooled over the four folds per seed and stratum (every subject held out exactly once, so the pooled
session-03 test covers the 16 subjects; four models, four dev thresholds): the same counts and rates; median and range
over the seeds. Uncertainty: bootstrap over subjects, 1,000 resamples with replacement of the held-out subjects of the
stratum (pooled, per seed), seed 0, percentiles 2.5-97.5 of ACER (dev and 0.5) and AUROC; no interval for a stratum
with fewer than 5 subjects; no statistical test. Per subject and stratum: restricted `per_subject.csv`.

## Budget and memory

12 trainings; in N1 a training took 8-11 minutes on the CPU for 2,507 fitting frames (wall 496-671 s), here 2,100
frames, so about 2.5 hours at most (measured in run 20261103-FOLDSRUN-e174812, slightly above the estimate: 702-838 s
per fit from the cards, 9,154 s for the twelve training steps of the run record at 706-843 s each, 9,703 s in all);
cache check (6,000 frames), extraction (6,614) and 12 scoring passes on top; cap 6 hours. The estimate holds on an idle
machine: under load a reviewer measured 1.5 s per step against 0.4 in N1, so the run is launched with the machine
otherwise idle. Timings of every step (wall clock and monotonic) in `run_record.json` and in the aggregate. Memory: the
cache (1.2 GB) and the extracted frames (1.3 GB) stay in memory as uint8; the fitting and dev arrays are converted to
float32 in place, released after `fit`, the test frames are assembled only after the threshold is frozen, and the Keras
session is cleared before every model (estimated peak 8-10 GB on a 16 GB machine).

## Limits

- Frames of one subject and session are correlated (consecutive frames of one capture): n counts frames, not independent
  units; the bootstrap over subjects is the only uncertainty, and the per-fold strata hold 4 subjects (session 03 and
  all) or 2-3 (sessions 01 and 02), the pooled strata 9 (sessions 01, 02) or 16 (03, all; session-03 bona fide for 9 of
  them).
- The dev of a held-out fold is the test fold of the next one: the four fold results are not independent (different
  models, no leakage); three seeds give a range, not a confidence interval.
- The dev frames drive early stopping, the learning-rate schedule, the restored epoch and the threshold; the dev does
  not identify a unique threshold when its classes separate completely and gives no guarantee of transfer.
- Session 03 is under-sampled in fitting (about 18-25 % of its bona fide and 11-13 % of its attacks); sessions 01-02
  exist for 9 subjects only; a session x class cell of the sessions 01-02 strata can rest on one subject.
- One dataset, one attack type (print). Diagnostic, exploratory; associations, not causes.

## Commands (from the development repository root, the data present)

```bash
.venv/bin/python -I scripts/nuaa_folds_run.py check-inputs
nohup caffeinate -dims .venv/bin/python -I scripts/nuaa_folds_run.py run --i-am-authorized > logs/foldsrun-console.txt 2>&1 &
```

The first command verifies only and writes nothing.

The run is started with `nohup` and `caffeinate -dims` (no idle, display or disk sleep), with the console redirected
to a file under `logs/` (ignored by git: a file in the repository root would make the tree dirty and the run id
`-dev`); the C++ messages of TensorFlow do not enter `run.log`; never through `| tee`, which would stop the run with
the terminal. `caffeinate -s` (no system sleep) holds only with the power adapter connected; with the lid closed the
machine may still sleep, and the sleep time counts towards the cap (wall clock). Under `nohup` SIGHUP stays ignored,
so closing the terminal does not stop the run; SIGTERM stops it with `status.json` = FAIL.

Run folder `<out-root>/YYYYMMDD-FOLDSRUN-<commit>[-dev]` (`-dev` when the git tree is not clean): `status.json`
(RUNNING, then PASS or FAIL; it records whether the configuration is the delivered one, by SHA-256), `run.log` (one log
per attempt, truncated at the start, its SHA-256 in the final `status.json`), `public/` (`aggregate.json`,
`aggregate.md`, `run_record.json`, `SHA256SUMS`), `restricted/` (per model `fold<k>_seed<s>_card.json`,
`_threshold.json`, `_dev_scores.csv`, `_test_scores.csv`, `_LivenessNet.keras`; `per_subject.csv`,
`extracted_frames.csv`, `public_SHA256SUMS.sha256`, `SHA256SUMS`). Inside the repository the run folder is accepted only
under `results/foldsrun/` and the extracted frames only under `data/processed/`; the run folder is validated before any
change, leftovers of an interrupted attempt are removed first, every write is atomic (checkpoints included). A folder
that already holds a `status.json` is refused: every attempt keeps its outcome, also a negative one; to try again,
archive the failed folder (for example rename it `<run-id>.failed-1`) and move the parked frames file away. On any error
the outputs are removed (a file that cannot be removed is listed as left behind in `status.json`) and what remains is
`status.json` = FAIL (with the reason), `run.log`, the empty `public/` and `restricted/` folders (a checkpoint being
saved is written through a temporary name that is removed on error, so no residue stays in `restricted/`) and, when the
extraction had completed, the extracted frames parked as `data/processed/nuaa_folds_frames.<run-id>.failed.npz` (named
in `status.json`; if the renaming itself fails, `status.json` says so), so that the next attempt is not blocked and the
file stays available for inspection. After a completed run the extracted frames file is in place and `check-inputs` (and
the real-input test of the suite) refuses it on purpose: rename it aside (its SHA-256 is in the run record) before
running `check-inputs` or the suite again with the real inputs.

## Files of the package

| File | Content |
|---|---|
| `README.md` | this document |
| `livenesslab/delivery/foldsrun/nuaa_folds_run.json` | the configuration: schema run, manifests, cache, extractor, architecture, recipe, threshold, metrics, bootstrap, budget, declarations |
| `livenesslab/scripts/nuaa_folds_run.py` | the script (check-inputs, run) |
| `livenesslab/scripts/check_nuaa_folds_run.py` | 71 tests on a synthetic world (identifiers and structure unlike NUAA, fake extractor, fake TensorFlow that records the recipe, fake trainer and scorer) |
| `livenesslab/scripts/check_nuaa_folds_run_mutations.py` | 112 declared mutations of the script, each detected by the suite on a copy, plus a control and three declared redundant |
| `livenesslab/scripts/m2_writers.py`, `path_guard.py`, `package_rules.py`, `train_cnn.py` | the shared modules, unchanged (atomic writes, folder guards, forbidden-text rules, the preprocessing chain) |
| `tables/aggregate.json`, `tables/aggregate.md`, `tables/run_record.json`, `tables/SHA256SUMS` | the public outputs of run `20261103-FOLDSRUN-e174812` |
| `test_logs/` | `check_nuaa_folds_run.txt` (macOS: suite and the real `check-inputs`), `check_nuaa_folds_run_mutations.txt`, `linux_package_no_data.txt` (python:3.12 container, package layout, no data, no TensorFlow: suite with one declared skip (`real inputs`) and mutation check) |
| `SHA256SUMS` | SHA-256 of every file of the package |

Dependencies: Python 3.11 or 3.12, NumPy, OpenCV 4.11.0 (extraction), TensorFlow 2.15 (training and scoring, CPU only;
the Metal GPU defect of N1 is the reason), the reference code of the coordinator (`architectures.py`,
`create_datasets.py`) and the frozen extractor `delivery/n2/face_0ac7143.py`. The tests need NumPy only.

Expected test lines. `tests run: 71, failures: 0, errors: 0, skipped: 0` with the real inputs and the delivered
configuration present; without the data `skipped: 1` (declared: `real inputs`; `delivered configuration` only when the
configuration file is absent too); `RESULT: PASS`. Mutation check: `mutations killed: 112/112`, `control mutations
survived: 4/4 (1 control, 3 declared redundant)`, `RESULT: PASS`.
