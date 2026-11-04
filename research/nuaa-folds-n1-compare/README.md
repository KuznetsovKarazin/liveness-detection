# NUAA: folds run versus N1 on the same 3,000 frames (descriptive comparison)

Requested by the coordinator on the C2 card on 4 November 2026, after the folds run: the descriptive comparison with
N1 on the same 3,000 IDs, using the existing scores and the thresholds already frozen. The two experiments compared
are the LivenessNet run with four subject folds and three training seeds (`20261103-FOLDSRUN-e174812`) and N1
(`20261024-N1-LivenessNet-fb93496`), on the 3,000 frames of the manifest `nuaa_cnn_test` (session 03, 1,500 bona fide
and 1,500 attacks of 16 subjects), seeds 42, 43 and 44.

Descriptive and post hoc: computed after both runs, from their per-frame scores and their frozen thresholds. No
training, no inference, no image read, no new threshold. Indicative; no causal reading.

## What it does and what it does not do

- Reads the existing per-frame scores of the two runs, verifies them against the manifest and against the checksums
  the runs recorded, and compares the two experiments frame by frame on the same 3,000 frames, per seed.
- In the folds run every frame of `nuaa_cnn_test` is in the test of exactly one fold per seed; its score, its decision
  at 0.5 and its decision at the frozen dev threshold are those of the model of that fold (the dev threshold is the one
  of the fold that produced the score). N1 has the fixed threshold 0.5 only.
- Does not train, score, read images, select or tune any threshold; does not change or write the two runs.
- The restricted outputs (per-frame pairs, per-subject table) go to the shared folder only; the public outputs carry
  no key and no subject identifier.

## Delivery status

- Run `20261104-FOLDSCOMPARE-3b3bcca` (clean commit `3b3bcca` of the development repository):
  `public/` copied here as `tables/` (its `SHA256SUMS` included: it is the anchor recorded in `status.json` and in
  the restricted folder).
- The restricted outputs (`paired_frames.csv`, `per_subject.csv`, `SHA256SUMS`,
  `public_SHA256SUMS.sha256`), with `status.json` and `run.log`, uploaded on 2026-11-04 to the
  shared N1 folder as the run folder `20261104-FOLDSCOMPARE-3b3bcca/` (provisional destination, as for N2 and the folds
  run).
- The configuration (`livenesslab/delivery/foldscompare/nuaa_folds_n1_compare.json`, SHA-256 fixed in the script as
  `CONFIG_SHA256`) was written and verified with `check-inputs` before the run (log in
  `test_logs/check_nuaa_folds_n1_compare.txt`).

## Inputs (every one pinned by SHA-256 and verified before any output is written)

check-inputs writes nothing; run writes status.json and run.log first, then verifies.

- The schema configuration `delivery/folds/nuaa_subject_folds.json`: SHA-256 fixed in the script and in the
  configuration (`d1d43c14...`); its `manifests.cnn_test`, `manifests.cnn_train` and `manifests.cnn_es` carry the same
  file names and SHA-256 as this configuration.
- The manifest `results/m2/manifests/nuaa_cnn_test.csv`: SHA-256 `ce9669dc...` (the full value equals
  `manifests.cnn_test` of the schema configuration); rows read with `nuaa_manifests.load_manifest`; 3,000 rows, 1,500
  / 1,500, 16 subjects, session 03 only; no duplicated key or file SHA-256.
- The N1 training manifests `nuaa_cnn_train.csv`, `nuaa_cnn_early_stopping.csv`: SHA-256; sessions 01-02 only; no key
  shared with cnn_test; 9 of the 16 cnn_test subjects are N1 training subjects (declared in the configuration,
  counted).
- The folds run `results/foldsrun/20261103-FOLDSRUN-e174812`: `status.json` by SHA-256, PASS, same run id, its anchors
  equal to the configuration; public `SHA256SUMS` and restricted `SHA256SUMS` equal to those anchors;
  `public/aggregate.json` as listed; the 12 `fold<k>_seed<s>_test_scores.csv` and the 12
  `fold<k>_seed<s>_threshold.json` as listed; every threshold of its fold, seed, set `dev fold only` and checkpoint
  (the listed `.keras` SHA-256, the weights are not read).
- The rows of the 12 score files: exact columns; fold, seed and role `test`; label 0/1; score a number in [0, 1];
  `threshold_dev` equal (text and value) to the frozen threshold of its fold; `threshold_sha256` equal to the SHA-256
  of that threshold file; `checkpoint_sha256` equal to the listed checkpoint; `prediction_05` and `prediction_dev`
  equal to the decision rule (attack if score > threshold, tie = bona fide); each subject in the test of one fold
  only; the held-out subjects of each fold the same for the three seeds.
- Per seed, the four files together: every cnn_test key exactly once (3,000 rows, none missing, none twice); label,
  subject, session and file SHA-256 equal to the manifest.
- N1 `results/n1/20261024-N1-LivenessNet-fb93496`: `SHA256SUMS` by SHA-256; `status.json` (PASS), `aggregate.json`
  (passed, same run id, threshold 0.5), `seed<s>/metrics.json` and `seed<s>/scores_test.csv` as listed; the scores
  file equal to `scores_sha256` of the aggregate; the weights SHA-256 of `metrics.json` equal to the aggregate.
- The rows of the 3 N1 score files: exact columns; run id and seed; checkpoint and configuration SHA-256 equal to the
  ones N1 recorded; label 0/1; score in [0, 1]; `prediction` equal to the rule at 0.5; the 3,000 cnn_test keys, each
  once; label, subject, session and file SHA-256 equal to the manifest.
- Anchors: the folds-run metrics over all its held-out frames per seed, recomputed from the 12 files, equal
  `pooled_per_seed.all` of its public `aggregate.json` (counts exact, rates and AUROC within 1e-12); the N1 test
  metrics recomputed from its scores equal `metrics.json` and `aggregate.json`.
- Control values of the coordinator, folds run on the 3,000 frames, seeds 42 / 43 / 44: ACER at 0.5 10.90 / 15.00 /
  10.00 %, ACER at the dev thresholds 9.53 / 15.30 / 11.70 %, AUROC 0.93851 / 0.95268 / 0.96701; agreement at the
  digits given by the coordinator: round(100 * ACER, 2) equal to the given percentage, round(AUROC, 5) equal to the
  given AUROC.

## Metrics

Per seed, on the 3,000 frames: bona fide and attacks; rejected bona fide and accepted attacks; BPCER, APCER and ACER
at 0.5 for both experiments, and at the frozen dev thresholds for the folds run; AUROC of both (Mann-Whitney, ties
0.5, score_attack as the separator). The AUROC of the folds run pools the scores of four models (one per fold) on one
ranking: it is not the AUROC of a single model. Differences, folds minus N1: ACER at 0.5 minus N1 at 0.5, ACER at the
dev thresholds minus N1 at 0.5, AUROC minus AUROC. Median and range over the three seeds. Bootstrap over the 16
subjects, 1,000 resamples with replacement, repeatable from this description: generator `numpy.random.default_rng(0)`
(PCG64); subjects in ascending identifier order; at every resample `integers(0, 16, 16)`; the generator restarts from
0 for every training seed (the same 1,000 resamples for 42, 43 and 44); every metric computed on the pooled frames of
the resampled subjects, not as a mean of per-subject values; the same subject indices for the two experiments, so the
interval of a difference is paired by subject; percentiles 2.5 and 97.5 with `np.percentile`, linear interpolation; a
resample without a class does not count for the metrics that need it. Paired decisions per frame: per seed, class and
threshold, the 2 x 2 table of frames correct in both, only in N1, only in the folds run, in neither, and per seed and
threshold the number of frames whose decision differs. Per subject (restricted `per_subject.csv`): errors, error rate,
ACER (both classes only) and AUROC of the two experiments; public: only the counts of subjects with the folds run
lower, equal or higher than N1.

## Declarations (in the configuration and in `aggregate.json`)

- Descriptive, post hoc, no training, no inference, no image read, no new threshold.
- Training roles differ: N1 was trained on 9 subjects (sessions 01-02) and 9 of the 16 subjects of the 3,000 frames
  are those training subjects, seen in other sessions; in the folds run none of the 16 subjects was in the training
  of the model that scored it (fitting on the 8 subjects of two other folds, dev on the 4 of the next fold).
- Indicative: frames correlated within subject and capture, three seeds give a range, the bootstrap over 16 subjects
  is coarse, subject-level ACER exists only for the 8 subjects with both classes.
- N1 AUROC is below 0.5 on these frames (0.3485-0.4582, as published by N1); the AUROC difference includes that.
- No causal reading: the two experiments differ in training subjects, sessions, sample sizes, dev sets and thresholds
  at once. One dataset, one attack type (print); no statistical test.

## Commands (from the development repository root, the data present)

```bash
.venv/bin/python -I scripts/nuaa_folds_n1_compare.py check-inputs
.venv/bin/python -I scripts/nuaa_folds_n1_compare.py run --i-am-authorized
.venv/bin/python -I scripts/check_nuaa_folds_n1_compare.py --verbose
.venv/bin/python -I scripts/check_nuaa_folds_n1_compare_mutations.py
```

The first command verifies only and writes nothing; it prints the control values as recomputed. `run` repeats every
check and writes `<out-root>/YYYYMMDD-FOLDSCOMPARE-<commit>[-dev]` (`-dev` when the git tree is not clean; inside the
repository only under `results/foldscompare/`): `status.json` (RUNNING, then PASS or FAIL with the reason; it records
whether the configuration is the delivered one and the SHA-256 of both `SHA256SUMS` and of `run.log`), `run.log`,
`public/` (`aggregate.json`, `aggregate.md`, `SHA256SUMS`), `restricted/` (`paired_frames.csv`, `per_subject.csv`,
`public_SHA256SUMS.sha256`, `SHA256SUMS`). The run folder is validated before any change (not a link, not inside or
containing an input, no foreign entry); a folder that already holds a `status.json` is refused; leftovers are removed
first; every write is atomic; on any error the outputs are removed and `status.json` is FAIL. The public files are
scanned for keys, image SHA-256 values, subject identifiers and forbidden text before they are written.

## Files of the package

- `README.md`: this document.
- `livenesslab/delivery/foldscompare/nuaa_folds_n1_compare.json`: the configuration (inputs and anchors, metrics,
  bootstrap, control values, declarations).
- `livenesslab/scripts/nuaa_folds_n1_compare.py`: the script (check-inputs, run).
- `livenesslab/scripts/check_nuaa_folds_n1_compare.py`: 151 tests on a synthetic world (16 subjects 1101-1116,
  structure unlike NUAA, scores built for answers computed by hand) and one test on the real inputs.
- `livenesslab/scripts/check_nuaa_folds_n1_compare_mutations.py`: 72 declared mutations of the script, each detected
  by the suite on a copy, plus a control.
- `livenesslab/scripts/m2_writers.py`, `path_guard.py`, `package_rules.py`, `nuaa_manifests.py`,
  `select_threshold.py`, `livenesslab/src/tesi_app/__init__.py`, `paths.py`: the shared modules, unchanged (atomic
  writes, folder guards, forbidden-text rules, `load_manifest` and its imports).
- `tables/aggregate.json`, `tables/aggregate.md`, `tables/SHA256SUMS`: the public outputs of run
  `20261104-FOLDSCOMPARE-3b3bcca`.
- `test_logs/`: `check_nuaa_folds_n1_compare.txt` (macOS: suite and the real `check-inputs`),
  `check_nuaa_folds_n1_compare_mutations.txt`, `linux_package_no_data.txt` (python:3.12 container, package layout
  mounted read-only, no data: suite with the declared skip `real inputs`, `check-inputs` refusing cleanly because the
  inputs are absent, mutation check).
- `SHA256SUMS`: SHA-256 of every file of the package.

Dependencies: Python 3.11 or 3.12 and NumPy.

Expected test lines. `tests run: 151, failures: 0, errors: 0, skipped: 0` with the real inputs and the delivered
configuration present; without the data `skipped: 1` (declared: `real inputs`); `RESULT: PASS`. The real-input test
prints the recomputed values (`real inputs: acer_05 0.10900/0.15000/0.10000; acer_dev 0.09533/0.15300/0.11700; auroc
0.93851/0.95268/0.96701; n1_acer 0.43833/0.51433/0.40100; n1_auroc 0.34849/0.37544/0.45822`). Mutation check:
`mutations killed: 72/72`, `control mutations survived: 1/1`, `RESULT: PASS`.
