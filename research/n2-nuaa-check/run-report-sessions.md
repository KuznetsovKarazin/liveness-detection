# Run report

## Identity
- Task and run ID: N2 part 1 (the three N1 LivenessNet checkpoints on the NUAA official-train frames of sessions
  01-02 outside fitting and validation) — `20261029-N2-sessions-2ac3e29`. The group format is `YYYYMMDD-task-seed-shortcommit`; this
  study id omits the seed because no seed is chosen here: the three N1 training seeds are scored, one subfolder each
  (`seed42/`, `seed43/`, `seed44/`).
- Date / operator: 2026-10-29 (`status.json`, `check.json`, `aggregate.json`) / Valerio Cassano.
- Git commit / environment lock: development repository commit `2ac3e290c9081454e9d058a294ab84156de96235` (`check.json`: `commit`, `dirty`
  false); `livenesslab/requirements.txt`. SHA-256 of the eleven code files recorded in `check.json` and compared at
  every step; equal to the files of this folder (`n2_sessions.py`, `n1_train.py`, `train_cnn.py`, `nuaa_manifests.py`,
  `select_threshold.py`, `path_guard.py`, `m2_writers.py`, `package_rules.py`, `evaluation.py`, `paths.py`, `face.py`),
  plus the frozen module `face_0ac7143.py` (`c72f7fa8…`), the reference `create_datasets.py` (`306ea1c8…`) and
  `architectures.py` (`a83c02b0…`).
- Command: from the development repository root (= `livenesslab/` in the group repository), bash/zsh array form:
  `C=(--config delivery/n2/n2_sessions.json --manifests results/m2/manifests --cache data/processed/nuaa_1500.npz --images data/NUAA/images --n1-study results/n1/20261024-N1-LivenessNet-fb93496 --frozen-face delivery/n2/face_0ac7143.py --study-dir results/n2/20261029-N2-sessions-2ac3e29)`;
  `python scripts/n2_sessions.py check "${C[@]}"`; `cache-check`; `parity`; for s in 42 43 44:
  `score "${C[@]}" --train-seed $s --i-am-authorized`; `aggregate "${C[@]}"`. Every step `RESULT: PASS`.
- Configuration path and SHA-256: `n2_sessions.json` (`delivery/n2/` in the development repository, the root of this
  folder in the group repository) — `41e6bfe7cff404fc69117610499fa7158575414d214fd708a618f38aa824e43f`, written before
  the runs and bound in `n2_sessions.py` (a non-trial run accepts no other configuration).
- Hardware / software: development Mac (Apple M4), macOS; recorded in the records: Python 3.11.16, TensorFlow 2.15.0,
  tensorflow-metal 1.1.0 (installed, GPU hidden for scoring), Keras 2.15.0, NumPy 1.26.4, scikit-learn 1.3.2, OpenCV
  4.11.0 (the version pinned in `requirements.txt`, under which the cache of 12 September 2026 reproduces). Scoring on the CPU only.
- Checkpoint origin and SHA-256: the three N1 checkpoints of study `20261024-N1-LivenessNet-fb93496` (`.h5`: seed 42
  `c63a858107b6f60375a27222a915e9b4b35f267fe9f34cf16117d673ca06546e`, seed 43
  `34bac22cafd3f4470895ee1ea3e22d86e8cd67139e036491fc94955925d9f0d6`, seed 44
  `41c128484ff76b4d754c75085f2c17f95c22897a59f34d8d27e7f4b940d23432`), verified against the N1 `train.json`, cards and
  the files before every scoring; no new training; the weights stay with the operator within the authorised scope.
- Pretraining / external data: none.

## Data and protocol
- Dataset version / authorization register entry: NUAA, register entry `dataset_register_nuaa.md` of the accepted M2
  package (run `20261006-M2-seed42-36c677a`); originals under `data/NUAA/images/` (copies of the official files,
  verified by SHA-256 against the manifests); cache `nuaa_1500.npz` `dc286dc3…177920` used only to verify the frozen
  preprocessing (bit-for-bit reproduction) and for the parity rows.
- Manifest and split SHA-256: M2 manifests `nuaa_official_train.csv` `2a7e8906…`, `nuaa_official_test.csv`
  `2fad23ff…`, `nuaa_cnn_train.csv` `6d46ec7d…`, `nuaa_cnn_early_stopping.csv` `17282be8…`, `nuaa_cnn_test.csv`
  `ce9669dc…`, `nuaa_c1_v2.csv` `1c2cf97c…`, `nuaa_m1_dev.csv` `1e4e49c4…` (full values in the configuration and in
  `check.json`); target manifest `target_manifest.csv` `5b481818…` (restricted; SHA-256 in the configuration).
- Independent unit / train-dev-test counts: unit = image frame. Target: 491 frames of the official training list,
  sessions 01 (244) and 02 (247), 243 bona fide and 248 attacks, 9 subjects (7 fitting: 410 frames; 2 validation: 81
  frames); the 81 frames of the validation subjects are also the 81 target frames of the M1 dev set. Reference: the 3,000 session 03 test frames of N1 (1,500 + 1,500, 16
  subjects), of which 1,538 belong to the 9 target subjects (1,325 to the 7 fitting subjects: 614 bona fide, 711
  attacks; 213 to the 2 validation subjects: attacks only).
- Subject, session, video overlap and duplicate checks (`check`, `cache-check`): target disjoint from `cnn_train`,
  `cnn_es`, `cnn_test` and `c1` by key and by file SHA-256; no duplicated file SHA-256; every original file found with
  size and SHA-256 equal to the manifest row; sampling order of the official lists equal to `cnn_train` + `cnn_es` and to
  `cnn_test`; after preprocessing no duplicated derived frame within the 491, between the 491 and the 6,000 cache
  frames, or inside the cache (exact pixel hash). The 9 target subjects are the 7 fitting and the 2 validation subjects
  (NUAA official train and test differ by session, not by subject).
- Label mapping / score direction: label 0 bona fide, 1 attack; score = softmax output 1 of the N1 LivenessNet
  (probability of attack), the N1 scoring path unchanged (`n1_train.tf_scorer`).
- Crop, resize, orientation, channel order, normalization: `train_cnn.preprocess` with the frozen detector module
  `face_0ac7143.py` (RetinaFace via OpenCV DNN, Haar cascade when RetinaFace finds nothing, whole image when neither
  does): face crop with a 15 % margin, square, 256×256 `INTER_AREA`, BGR to RGB, the reference enhancement, uint8,
  divided by 255 at scoring. Detectors on the 491: RetinaFace 478, Haar 13, no frame without a face. Verification:
  every one of the 6,000 cache frames reproduced bit for bit from its original with this chain (`cache_check.json`,
  `cache_check.log`: train Haar 51, RetinaFace 2949; test Haar 3, RetinaFace 2996, none 1); parity on 24 keys chosen by identifier (rule and SHA-256 of the list in the configuration; list restricted): frames equal to the
  cache rows, scores within 1e-6 of the stored N1 scores for every seed (largest difference 0).
- Model selection, threshold selection and calibration: none. The three N1 checkpoints as they are; threshold 0.5
  fixed a priori; no seed selection, no score inversion, no M1 threshold reused.
- Attack decision at threshold (including equality): attack if `score_attack > 0.5`; a tie is bona fide (0 ties).
- Source-only development rule for external targets: not applicable (intra-dataset NUAA; nothing chosen on the target).

## Evidence
- Restricted score file location and SHA-256: `02_Experiments/N2/20261029-N2-sessions-2ac3e29/` (to be uploaded when the folder is
  shared; see README "Delivery status"): per seed `scores_sessions.csv` (491 rows) `1505aa92…`, `24811e3d…`,
  `4831b639…`; `target_manifest.csv` `5b481818…`; `frames_target.csv` `60d58326…`; `parity.csv` `b76e70d2…`;
  `strata_all_seeds.csv` `82a2e0a0…`; `check.json` `772dc482…`; folder `SHA256SUMS` `40ee121e…` (37
  files).
- APCER / BPCER / ACER with denominators (threshold 0.5):

  | set | seed | accepted attacks / attacks | rejected bona fide / bona fide | APCER | BPCER | ACER | AUROC |
  |---|---|---|---|---|---|---|---|
  | sessions 01-02: the 491 target frames | 42 | 0/248 | 0/243 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
  | sessions 01-02: the 491 target frames | 43 | 0/248 | 0/243 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
  | sessions 01-02: the 491 target frames | 44 | 0/248 | 0/243 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
  | session 03: test frames of the 9 target subjects (N1 scores) | 42 | 154/924 | 422/614 | 0.1667 | 0.6873 | 0.4270 | 0.3230 |
  | session 03: test frames of the 9 target subjects (N1 scores) | 43 | 359/924 | 422/614 | 0.3885 | 0.6873 | 0.5379 | 0.3171 |
  | session 03: test frames of the 9 target subjects (N1 scores) | 44 | 63/924 | 422/614 | 0.0682 | 0.6873 | 0.3777 | 0.4569 |
  | session 03: the 3,000 nuaa_cnn_test frames (N1 scores, read as stored) | 42 | 160/1500 | 1155/1500 | 0.1067 | 0.7700 | 0.4383 | 0.3485 |
  | session 03: the 3,000 nuaa_cnn_test frames (N1 scores, read as stored) | 43 | 391/1500 | 1152/1500 | 0.2607 | 0.7680 | 0.5143 | 0.3754 |
  | session 03: the 3,000 nuaa_cnn_test frames (N1 scores, read as stored) | 44 | 72/1500 | 1131/1500 | 0.0480 | 0.7540 | 0.4010 | 0.4582 |

  Median (range) on the 491: APCER 0.0000 (0.0000-0.0000), BPCER 0.0000 (0.0000-0.0000), ACER 0.0000 (0.0000-0.0000), AUROC 1.0000 (1.0000-1.0000). Session 03, same 9 subjects: APCER 0.1667 (0.0682-0.3885), BPCER 0.6873 (0.6873-0.6873), ACER 0.4270 (0.3777-0.5379), AUROC 0.3230 (0.3171-0.4569).
- Attack types / acquisition conditions: NUAA print attacks only; target from capture sessions 01-02 (the training
  sessions), reference from session 03.
- Seed(s) / uncertainty / exclusions: the three N1 training seeds (42, 43, 44); three values give a range, not a
  confidence interval. No frame excluded (every original readable; no face-detection failure on the 491).
- Runtime and latency protocol, if relevant: not relevant. The scripts record no durations; from the timestamps of the
  records of this study: cache-check about 1.5 minutes, parity under 1 minute, scoring about 20 s per seed.
- Reproduction command / logs: commands above; logs `check.log`, `cache_check.log`, `parity.log`, `seed<N>/score.log`,
  `aggregate.log` in the restricted folder (`cache_check.log` also in `test_logs/`, as `cache_check_log.txt`). Tests: `check_n2_sessions.py`
  93 tests, `check_n2_mutations.py` 85/85 with the control mutation surviving; same on Linux (`test_logs/linux.txt`).

## Interpretation
- Research question answered: whether the three N1 checkpoints work on frames of the training sessions that were
  used neither for fitting nor for early stopping, compared with the session 03 test, with models, preprocessing and
  threshold unchanged.
- Observed result and comparison: every seed classifies the 491 frames without error (AUROC 1.0; fitting and validation
  subjects alike), while on session 03 the same checkpoints give BPCER 0.687 (614 bona fide frames of the 3 target
  subjects that have bona fide frames there) and AUROC 0.317–0.457 on the 1,538 frames of the same 9 persons;
  every comparable stratum has a lower or equal error rate in sessions 01-02. Consistent with the session-shift reading
  of N1; not a causal test and not a generalisation estimate.
- Limitations / failed checks: exploratory check after N1, not an independent validation (frames from the training
  sessions and persons; 81 already in the M1 dev set); session 03 holds bona fide frames for only 3 of the 9 subjects
  (6 bona fide strata NA); the frozen module is dated four days after the cache (equivalence empirical: 6,000 frames
  bit for bit on this machine and OpenCV version); one cache test frame holds the whole image (no face detected on 12
  September), kept as stored; image-level counts with correlated frames; three seeds give a range. No failed check.
- Next decision: to be discussed with the reviewer (see the note): whether to examine the session shift directly (for example a split by session with a held-out training-session
  subset, or capture-condition descriptors per session) before any further training on NUAA; no new experiment is
  proposed here.
- Reviewer and review outcome: pending.
