# Run report

## Identity
- Task and run ID: N1 part 1 (subject sensitivity of the accepted M1 result) — `20261024-N1-subjects-a5fcbbc`. The
  group format is `YYYYMMDD-task-seed-shortcommit`; this id omits the seed because the run has no random component (no
  training, no inference, no sampling): it reads existing scores only.
- Date / operator: 2026-10-24 20:55 (`status.json`) / Valerio Cassano.
- Git commit / environment lock: development repository commit `a5fcbbcbee24e48c0d4b50f63d5c30ab06884a1d`, clean
  working tree (`dirty: false` in `status.json` and in the public aggregate); `livenesslab/requirements.txt`. The
  script uses the Python standard library plus `m2_writers.py` and `path_guard.py` (SHA-256 of the three code files
  recorded in the public aggregate: `n1_subject_sensitivity.py` `9df02cf7…`, `m2_writers.py` `22760abc…`,
  `path_guard.py` `e5fa9c83…`, equal to the files of this folder).
- Command: `python scripts/n1_subject_sensitivity.py check-inputs`, then
  `python scripts/n1_subject_sensitivity.py run` (from the development repository root (= `livenesslab/` in the
  group repository), with the restricted C1 and M1 run folders at the paths of the configuration).
- Configuration path and SHA-256: `livenesslab/delivery/n1/subject_sensitivity/n1_subject_sensitivity.json` —
  `c77b72413e1f06c87fe4f0eba62198e8329e16ed47c6c5573966ca0e86ae9531`.
- Hardware / software: development Mac (Apple M4), macOS, Python 3.11 of the project environment; GPU not used, no
  TensorFlow (exact fractions with `fractions.Fraction`). Library versions are not recorded by this run (standard
  library only).
- Checkpoint origin and SHA-256: not applicable (no model is loaded; the scores are those of the accepted C1 and M1
  runs, whose checkpoints are recorded there).
- Pretraining / external data: not applicable.

## Data and protocol
- Dataset version / authorization register entry: NUAA as in C1/M1/M2 (register entry `dataset_register_nuaa.md` of
  the accepted M2 package, run `20261006-M2-seed42-36c677a`); only score files are read, no image.
- Manifest and split SHA-256: inputs verified against the configuration and against the `SHA256SUMS` of their run
  folders (whose SHA-256 is fixed in the configuration): C1 `scores.csv` `0a52d875…`, C1 `nuaa_manifest.csv`
  `86095da1…` (run `20260928-C1-seed42-a80f6f9`, `SHA256SUMS` `c32099d2…`); M1 `scores_test_dev_threshold.csv`
  `4a279428…`, `threshold.json` `cd1df877…`, `nuaa_test_dev_threshold_table.md` `60728fbe…`, `nuaa_dev_manifest.csv`
  `d4637ea6…` (run `20261003-M1-seed42-e67ecc7`, `SHA256SUMS` `86f05639…`).
- Independent unit / train-dev-test counts: unit = image; 300 NUAA test images (150 bona fide, 150 attacks) from 16
  subjects; 8 subjects with both classes, 7 with attacks only, 1 with bona fide only. No training or dev set is used
  (the dev manifest only identifies the two dev subjects, to reproduce the M1 block without them).
- Subject, session, video overlap and duplicate checks: same samples, subjects, labels and score text in C1, M1 and
  the manifest, row by row; no duplicated sample; dataset `nuaa`, split `test`. NUAA is not subject-disjoint between
  its official train and test splits (declared in M2).
- Label mapping / score direction: label 0 bona fide, 1 attack; score = probability of attack (C1 and M1 records).
- Crop, resize, orientation, channel order, normalization: not applicable (existing scores; preprocessing as in C1).
- Model selection, threshold selection and calibration: none in this run; thresholds 0.5 and the four frozen M1
  thresholds (`livenessnet__nuaa` 0.181054949760437, `attacknet_v1__nuaa` 0.32923662662506104, `attacknet_v2_1__nuaa`
  0.8703288733959198, `attacknet_v2_2__nuaa` 0.6969330608844757), unchanged; no recalibration on the test.
- Attack decision at threshold (including equality): attack if `score_attack > threshold`; a tie is bona fide; the
  stored predictions are recomputed and must match.
- Source-only development rule for external targets: not applicable (no new threshold; the M1 thresholds were chosen
  on the source dev set in M1).

## Evidence
- Restricted score file location and SHA-256: `02_Experiments/N1/20261024-N1-subjects-a5fcbbc/` (the earlier run from
  a modified tree goes to `02_Experiments/N1/20261024-N1-subjects-2c44c64-dev/`): `restricted/` `n1_per_image.csv`
  `02a50772…`, `n1_per_subject.csv` `3e8cda83…`, `n1_loso.csv` `bf67f71f…`, `n1_subject_sensitivity_restricted.json`
  `e6d8dce7…`, `.md` `f433740d…`, `public_SHA256SUMS.sha256` `381a7d0b…`, `SHA256SUMS`
  `2e9ff3c4cdc3dc0070b114cf1634dceb6785cb8758dbfabd39a23056c7148daf`; `public/` `SHA256SUMS`
  `9752e00eeafb948c190c72f67dddc445877ae88b7e37e72b9fc88314ec8295f8` (the two public aggregates are in
  `tables/subject_sensitivity/` of this folder).
- APCER / BPCER / ACER with denominators (150 attacks, 150 bona fide; FA = accepted attacks, FR = rejected bona fide):

  | analyzer | FA/FR at 0.5 | ACER at 0.5 | FA/FR at M1 | ACER at M1 | ΔACER full | leave-one-out ΔACER min … max | sign changes |
  |---|---|---|---|---|---|---|---|
  | livenessnet__nuaa | 24/109 | 0.4433 | 5/118 | 0.4100 | −0.033333 | −0.051750 … −0.009568 | 0/16 |
  | attacknet_v1__nuaa | 42/40 | 0.2733 | 37/44 | 0.2700 | −0.003333 | −0.010165 … +0.002542 | 2/16 |
  | attacknet_v2_1__nuaa | 3/115 | 0.3933 | 11/91 | 0.3400 | −0.053333 | −0.067033 … −0.019845 | 0/16 |
  | attacknet_v2_2__nuaa | 22/118 | 0.4667 | 38/112 | 0.5000 | +0.033333 | +0.019910 … +0.048762 | 0/16 |

  Denominators after excluding one subject: bona fide 128–150, attacks 138–150 (no class empties, no NA in the
  leave-one-out); per subject, APCER or BPCER is NA for the 8 subjects with one class only. The full-set values
  reproduce the accepted M1 table exactly (56 cells of the main block and 56 of the block without the dev subjects).
- Attack types / acquisition conditions: NUAA print attacks only; test images from capture session 03.
- Seed(s) / uncertainty / exclusions: no seed (no random component). The 16 leave-one-out values are a sensitivity
  description, not a confidence interval; no image excluded.
- Runtime and latency protocol, if relevant: not relevant.
- Reproduction command / logs: commands above; tests `python scripts/check_n1_subject_sensitivity.py` (47 tests; 2
  skipped in the package layout, where the restricted run folders are absent) and
  `python scripts/check_n1_subject_sensitivity_mutations.py` (40/40, control mutation surviving); logs in
  `test_logs/`. An earlier run from a modified tree (`20261024-N1-subjects-2c44c64-dev`) gave the same values.

## Interpretation
- Research question answered: whether the sign of the M1 change of ACER (frozen dev threshold against 0.5) depends on
  single test subjects.
- Observed result and comparison: for livenessnet, attacknet_v2_1 and attacknet_v2_2 the sign of ΔACER does not change
  when any single subject is excluded, although its size varies; for attacknet_v1 the full ΔACER (−1/300) is smaller
  in magnitude than the largest change from excluding a single subject, and the sign changes in 2 of 16 exclusions
  (two subjects with attacks only). Descriptive, no statistical test.
- Limitations / failed checks: the test images include persons already present in training (NUAA is not
  subject-disjoint between its official splits), so the result does not show generalisation; 16 subjects with
  unbalanced class composition; leave-one-out values are not independent and not a confidence interval; sensitivity
  analysis of existing results, not cross-validation and not a new independent validation; image-level counts with
  correlated frames; the 300 C1 images are verified against the official source (M2), the verification is not extended
  beyond them. No failed check.
- Next decision: none from this run alone; reviewed together with N1 part 2.
- Reviewer and review outcome: pending.
