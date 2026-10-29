# N2: NUAA diagnostic check (part 1: the N1 checkpoints on the sessions 01-02 frames outside fitting and validation; part 2: the historical combined AttackNetV2.2 on the N1 test, two image-preparation variants)

Card N2 (the task described by the coordinator on the operator's Trello card; discussion on the C2 card, 28-29 October 2026). No new
training. Part 1: the three N1 LivenessNet checkpoints (training seeds 42, 43, 44, CPU) applied to the 491 frames of
the NUAA official training list (sessions 01-02) that were used neither for fitting (`cnn_train`) nor for early
stopping (`cnn_es`), with the preprocessing that built the training cache, compared with the session 03 test of N1.
Part 2: `AttackNetV2_2_combined_all_final.h5`, trained by the coordinator's group on `combined_all` (no NUAA), applied
as saved to the same 3,000 NUAA test frames of N1 with two image-preparation variants fixed before the test (whole
image; face crop), as a sensitivity analysis to the image preparation. This folder holds the code, the two
configurations saved before the runs, the frozen detector module, the run reports, the publishable tables and
figures, the note and the test logs. Per-image scores, the 491-frame manifest, per-subject tables and logs go to
the restricted area (upload pending the shared folder; see "Delivery status").

## Delivery status

- Part 1: study `20261029-N2-sessions-2ac3e29` on 2026-10-29, from the clean commit `2ac3e29` of the development repository (recorded
  tree state clean, SHA-256 of every code file recorded at `check` and equal to the files of this folder). The
  tables and figures below come from it. Trial executions from a modified tree (`-trial` studies in the session's scratch
  area, not part of the delivery) gave the same scores for every frame.
- Part 2: study `20261029-N2-combined-247dfd7` on 2026-10-29, from the clean commit `247dfd7`, which adds to `2ac3e29` only the
  results of part 1 (the eleven code files have the same SHA-256 in both `check.json`). Trial executions from a
  modified tree (not part of the delivery) gave the same scores for every frame.
- Restricted folders: `20261029-N2-sessions-2ac3e29/` and `20261029-N2-combined-247dfd7/` (per-image scores, the 491 manifest with keys and
  SHA-256, frames, parity rows, per-subject and per-stratum tables, logs, records; no images, no cache, no weights),
  ready for upload to `02_Experiments/N2/` as soon as the folder is shared; status in the Trello card.
- Images, cache and weights stay local (the N1 weights within the already authorised scope; the historical combined
  checkpoint as shared by the coordinator).

## What is delivered

Run reports (group template `templates/run-report.md`, one per study): `run-report-sessions.md` (part 1) and
`run-report-combined.md` (part 2). The two study ids omit the seed of the `YYYYMMDD-task-seed-shortcommit` format: no
training seed is chosen here (part 1 holds the three N1 seeds in `seed42/`, `seed43/`, `seed44/`; part 2 holds the two
variants in `whole_image/`, `face_crop/`). `N2_note.md`: the two-page note (results, reading, recommendation, limits).

Part 1:
- `n2_sessions.json`: the configuration, written before the runs; its SHA-256
  (`41e6bfe7cff404fc69117610499fa7158575414d214fd708a618f38aa824e43f`) is bound in `n2_sessions.py`: a non-trial run
  accepts no other configuration.
- `face_0ac7143.py`: the frozen face-detection module, byte-identical to `src/tesi_app/face.py` of the development
  repository at commit `0ac7143` (16 September 2026; SHA-256
  `c72f7fa8a9f3b099babce2255530466e5a34731fb5d021f97fd412a98cb58261`), the earliest committed version of the module,
  which reproduces the training cache `nuaa_1500.npz` (built on 12 September) bit for bit; from 24 September (our change
  `44971f3`) the RetinaFace box of our application is one pixel narrower and shorter. It is loaded as
  `tesi_app.face_0ac7143`; the current `face.py` is never used by part 1.
- `livenesslab/scripts/n2_sessions.py`: subcommands `check`, `cache-check`, `parity`, `score`, `aggregate`.
- `livenesslab/scripts/check_n2_sessions.py` (tests on synthetic data, no TensorFlow, no detector) and
  `check_n2_mutations.py` (declared mutations).
- `tables/sessions/`: `aggregate.json`, `aggregate.md` (the single table, the strata summary, the distributions),
  `figure_distributions.pdf` and `figure_groups.pdf` (both drawn from `aggregate.json` only: no identifiers, no
  per-subject values), `cache_check.json`, `parity.json` (counts only). `test_logs/cache_check_log.txt`: the log of the
  bit-for-bit reproduction of the 6,000 cache frames. The study writes the figures as PNG and the log as
  `cache_check.log`; the group repository ignores `*.png` and `*.log`, so this folder holds the same images converted to
  PDF (bitmap, pixels unchanged) and the log under a `.txt` name; the originals, with their SHA-256, are in the
  restricted study folder. The per-subject figure (`figure_strata.png`, one point per subject × class, without
  identifiers) is restricted, like the per-subject tables.

Part 2:
- `n2_combined.json`: the configuration, written before the runs; SHA-256
  (`0527cc91c88700b94550a178bb78f890e8cc1d987bf1ef932b30afb61567c154`) bound in `n2_combined.py`.
- `livenesslab/scripts/n2_combined.py`: subcommands `check`, `score`, `aggregate`.
- `livenesslab/scripts/check_n2_combined.py` (tests) and `check_n2_combined_mutations.py`.
- `tables/combined/`: `aggregate.json`, `aggregate.md`, `figure_distributions.pdf` (PNG in the restricted folder).

Unchanged copies, byte-identical to the accepted packages of `research/pad-2026`: `n1_train.py` (N1: the scoring path
of LivenessNet, the records, the metrics and the forbidden-content scan are imported from it), `train_cnn.py`
(`preprocess`, the chain of the cache; package `c2-livenessnet-device`), `nuaa_manifests.py`, `select_threshold.py`,
`path_guard.py`, `m2_writers.py`, `package_rules.py`, `src/tesi_app/{__init__,paths,evaluation,core,face}.py`,
`requirements.txt`. The SHA-256 of `n1_train.py` and `train_cnn.py` are fixed by a test.

## Part 1: the N1 checkpoints on the sessions 01-02 frames outside fitting and validation

### Target and reference

Target: the rows of `nuaa_official_train.csv` whose key is neither in `nuaa_cnn_train.csv` nor in
`nuaa_cnn_early_stopping.csv`, sorted by key (`target_manifest.csv`, restricted, SHA-256
`5b481818c050c2d35aff44030d49ac4d420d0b9a2a7e2549822ddb8b771af11d` written in the configuration): 491 frames, 243 bona
fide and 248 attacks, session 01: 244, session 02: 247, 9 subjects, of which 7 are the fitting subjects of `cnn_train`
(410 frames) and 2 the validation subjects of `cnn_es` (81 frames); 81 frames are also in the M1 dev set
(`nuaa_m1_dev.csv`), none is in the test or in the C1 subset. Checked at `check`: disjoint from `cnn_train`, `cnn_es`,
`cnn_test` and `c1` by key and by file SHA-256; no duplicated file SHA-256; every original file located under the images
folder with size and SHA-256 equal to the manifest row.

Reference: the 3,000 session 03 scores of the N1 study `20261024-N1-LivenessNet-fb93496` (`scores_test.csv` of every
seed, SHA-256 in the configuration), read as stored, never rescored; each row is checked against its `cnn_test`
manifest row (subject, session, label, file SHA-256, checkpoint SHA-256, decision rule).

### Preprocessing (frozen) and its verification

`train_cnn.preprocess`, unchanged: `cv2.imread`, the frozen detector, `Face.crop(margin=0.15, square=True)` or the
whole image when no face is found, `cv2.resize` 256×256 `INTER_AREA`, BGR to RGB, `advanced_image_enhancement` of the
reference `create_datasets.py`, uint8; scored as frames / 255 through the N1 scorer (`n1_train.tf_scorer`: reference
LivenessNet architecture, weights verified by SHA-256, CPU, BatchNorm in inference, batch 16, softmax column 1).

- `cache-check`: every one of the 6,000 frames of the cache (`X_train` and `X_test`, in sampling order) reproduced from
  its original file and compared bit for bit: 6000 of 6000 frames equal, 0 different (`cache_check.json`; detectors: train Haar 51, RetinaFace 2949, test
  Haar 3, RetinaFace 2996, none 1). The 491 target frames: 478 RetinaFace, 13 Haar; no duplicated derived frame within the 491, between the
  491 and the cache, or inside the cache (exact pixel hash).
- `parity`: 24 keys chosen by identifier before any computation (rule written in the configuration: per test subject,
  the first bona fide and the first attack key of `nuaa_cnn_test` in key order; subjects without a class give one key),
  with the count and the SHA-256 of the derived list fixed in the configuration; the list itself (`parity_keys.csv`) is
  restricted, like every identifier: every frame equal to its cache row bit for bit, and every score recomputed on the
  CPU within the declared tolerance 1e-6 of the stored N1 score (largest difference 0 over 24 keys × 3 seeds).
- SHA-256 fixed in the configuration and verified at every step: the frozen module (also its file name), the
  detector resources (`deploy.prototxt` and `Widerface-RetinaFace.caffemodel` of Silent-Face-Anti-Spoofing;
  `haarcascade_frontalface_default.xml` is read from the installed OpenCV, so a different OpenCV build stops the check
  by design), the reference `create_datasets.py` and `architectures.py`, the N1 records, weights and scores. OpenCV
  version recorded (4.11.0, the version pinned in `requirements.txt`, under which the cache reproduces).
- Originals: located by key in the prepared layout (`<class>_<split>/<name>`) or in the raw mirror layout
  (`ClientRaw|ImposterRaw/<subject>/<name without the leading subject field>`); a symbolic link is refused; size and
  SHA-256 must equal the manifest row (the provenance is the hash, not the path).

### Results (study `20261029-N2-sessions-2ac3e29`, `tables/sessions/aggregate.md`)

Threshold 0.5 on the attack probability, attack if `score_attack > 0.5` (tie = bona fide); every N1 seed reported, no
seed selection, no threshold change, no score inversion. Median and range over the three seeds. NA = class absent.

| set | seed | n bona fide | n attack | rejected bona fide | accepted attacks | APCER | BPCER | ACER | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| sessions 01-02: the 491 target frames | 42 | 243 | 248 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| sessions 01-02: the 491 target frames | 43 | 243 | 248 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| sessions 01-02: the 491 target frames | 44 | 243 | 248 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| sessions 01-02: the 491 target frames | median (range) | | | | | 0.0000 (0.0000-0.0000) | 0.0000 (0.0000-0.0000) | 0.0000 (0.0000-0.0000) | 1.0000 (1.0000-1.0000) |
| sessions 01-02: target frames of the 7 fitting subjects | 42 | 218 | 192 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| sessions 01-02: target frames of the 7 fitting subjects | 43 | 218 | 192 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| sessions 01-02: target frames of the 7 fitting subjects | 44 | 218 | 192 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| sessions 01-02: target frames of the 7 fitting subjects | median (range) | | | | | 0.0000 (0.0000-0.0000) | 0.0000 (0.0000-0.0000) | 0.0000 (0.0000-0.0000) | 1.0000 (1.0000-1.0000) |
| sessions 01-02: target frames of the 2 validation subjects | 42 | 25 | 56 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| sessions 01-02: target frames of the 2 validation subjects | 43 | 25 | 56 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| sessions 01-02: target frames of the 2 validation subjects | 44 | 25 | 56 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| sessions 01-02: target frames of the 2 validation subjects | median (range) | | | | | 0.0000 (0.0000-0.0000) | 0.0000 (0.0000-0.0000) | 0.0000 (0.0000-0.0000) | 1.0000 (1.0000-1.0000) |
| sessions 01-02: target frames outside the M1 dev set | 42 | 218 | 192 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| sessions 01-02: target frames outside the M1 dev set | 43 | 218 | 192 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| sessions 01-02: target frames outside the M1 dev set | 44 | 218 | 192 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| sessions 01-02: target frames outside the M1 dev set | median (range) | | | | | 0.0000 (0.0000-0.0000) | 0.0000 (0.0000-0.0000) | 0.0000 (0.0000-0.0000) | 1.0000 (1.0000-1.0000) |
| session 03: the 3,000 nuaa_cnn_test frames (N1 scores, read as stored) | 42 | 1500 | 1500 | 1155 | 160 | 0.1067 | 0.7700 | 0.4383 | 0.3485 |
| session 03: the 3,000 nuaa_cnn_test frames (N1 scores, read as stored) | 43 | 1500 | 1500 | 1152 | 391 | 0.2607 | 0.7680 | 0.5143 | 0.3754 |
| session 03: the 3,000 nuaa_cnn_test frames (N1 scores, read as stored) | 44 | 1500 | 1500 | 1131 | 72 | 0.0480 | 0.7540 | 0.4010 | 0.4582 |
| session 03: the 3,000 nuaa_cnn_test frames (N1 scores, read as stored) | median (range) | | | | | 0.1067 (0.0480-0.2607) | 0.7680 (0.7540-0.7700) | 0.4383 (0.4010-0.5143) | 0.3754 (0.3485-0.4582) |
| session 03: test frames of the 9 target subjects (N1 scores) | 42 | 614 | 924 | 422 | 154 | 0.1667 | 0.6873 | 0.4270 | 0.3230 |
| session 03: test frames of the 9 target subjects (N1 scores) | 43 | 614 | 924 | 422 | 359 | 0.3885 | 0.6873 | 0.5379 | 0.3171 |
| session 03: test frames of the 9 target subjects (N1 scores) | 44 | 614 | 924 | 422 | 63 | 0.0682 | 0.6873 | 0.3777 | 0.4569 |
| session 03: test frames of the 9 target subjects (N1 scores) | median (range) | | | | | 0.1667 (0.0682-0.3885) | 0.6873 (0.6873-0.6873) | 0.4270 (0.3777-0.5379) | 0.3230 (0.3171-0.4569) |
| session 03: test frames of the 7 fitting subjects (N1 scores) | 42 | 614 | 711 | 422 | 116 | 0.1632 | 0.6873 | 0.4252 | 0.3256 |
| session 03: test frames of the 7 fitting subjects (N1 scores) | 43 | 614 | 711 | 422 | 270 | 0.3797 | 0.6873 | 0.5335 | 0.3191 |
| session 03: test frames of the 7 fitting subjects (N1 scores) | 44 | 614 | 711 | 422 | 42 | 0.0591 | 0.6873 | 0.3732 | 0.4526 |
| session 03: test frames of the 7 fitting subjects (N1 scores) | median (range) | | | | | 0.1632 (0.0591-0.3797) | 0.6873 (0.6873-0.6873) | 0.4252 (0.3732-0.5335) | 0.3256 (0.3191-0.4526) |
| session 03: test frames of the 2 validation subjects (N1 scores) | 42 | 0 | 213 | 0 | 38 | 0.1784 | NA | NA | NA |
| session 03: test frames of the 2 validation subjects (N1 scores) | 43 | 0 | 213 | 0 | 89 | 0.4178 | NA | NA | NA |
| session 03: test frames of the 2 validation subjects (N1 scores) | 44 | 0 | 213 | 0 | 21 | 0.0986 | NA | NA | NA |
| session 03: test frames of the 2 validation subjects (N1 scores) | median (range) | | | | | 0.1784 (0.0986-0.4178) | NA (NA) | NA (NA) | NA (NA) |

The 81 target frames of the M1 dev set are exactly the 81 frames of the 2 validation subjects (the M1 dev set holds those
two subjects), so the rows "outside the M1 dev set" and "of the 7 fitting subjects" are the same 410 frames.

Strata (subject × class, sessions 01-02 against session 03 on the same subjects; per-subject values restricted,
`strata_all_seeds.csv` and `figure_strata.png`):
| seed | group / class | strata | lower in 01-02 | equal | higher in 01-02 | not comparable |
|---|---|---|---|---|---|---|
| 42 | fitting/bona fide | 7 | 2 | 1 | 0 | 4 |
| 42 | fitting/attack | 7 | 7 | 0 | 0 | 0 |
| 42 | validation/bona fide | 2 | 0 | 0 | 0 | 2 |
| 42 | validation/attack | 2 | 2 | 0 | 0 | 0 |
| 43 | fitting/bona fide | 7 | 2 | 1 | 0 | 4 |
| 43 | fitting/attack | 7 | 7 | 0 | 0 | 0 |
| 43 | validation/bona fide | 2 | 0 | 0 | 0 | 2 |
| 43 | validation/attack | 2 | 2 | 0 | 0 | 0 |
| 44 | fitting/bona fide | 7 | 2 | 1 | 0 | 4 |
| 44 | fitting/attack | 7 | 6 | 1 | 0 | 0 |
| 44 | validation/bona fide | 2 | 0 | 0 | 0 | 2 |
| 44 | validation/attack | 2 | 1 | 1 | 0 | 0 |

Score distributions per class: `tables/sessions/aggregate.md` and `figure_distributions.pdf`. Error rate per group of
subjects and class, sessions 01-02 against session 03: `figure_groups.pdf`.

### Reading (descriptive)

- On the 491 frames of sessions 01-02 every seed separates the two classes completely: no rejected bona fide, no
  accepted attack, AUROC 1.0; the lowest attack score is above the highest bona fide score for every seed (margins in the
  restricted scores). The 81 frames of the 2 validation subjects (never used for gradient updates; they are also the
  81 frames already in the M1 dev set) behave like the 410 frames of the 7 fitting subjects.
- On session 03 the same checkpoints reject about two bona fide frames in three (BPCER 0.687 for every seed, on the 614
  session 03 bona fide frames of the 3 target subjects that have bona fide frames there) and rank bona fide frames above
  attacks (AUROC 0.317-0.457 on the 1,538 frames of the 9 target subjects). Every stratum of the target has error rate 0;
  where session 03 has a value, the target error rate is lower or equal (never higher).
- Reading: with models, preprocessing and threshold unchanged, the error appears when the capture session changes, not
  when the frames change within the training sessions; this is consistent with the session-shift explanation proposed in
  N1; it does not test the weak-validation explanation (with a near-zero validation loss, a perfect separation of frames
  from the training sessions is expected under both). It is a diagnostic reading, not a causal test: the target frames
  share sessions, persons and acquisition conditions with the training frames, so the perfect separation measures
  recognition of the training sessions, not generalisation.

### Limits of part 1

- Exploratory check after N1, not an independent validation: the 491 frames come from the same capture sessions
  (01-02) and the same 9 persons as the fitting and validation frames, so they are correlated with the training data;
  81 of them were already scored in M1 (dev set). A difference between sessions suggests what to examine next, it does
  not prove a cause.
- The frozen module `0ac7143` is dated 16 September, four days after the cache: its equivalence is empirical (the
  6,000 frames reproduced bit for bit with OpenCV 4.11.0 on this machine), not a record of the build.
- Session 03 has bona fide frames for only 3 of the 9 target subjects: 6 bona fide strata are NA in the comparison.
- One cache frame of the test (session 03) had no detected face and holds the whole image, as built on 12 September;
  it is kept as it is (the N1 scores are read as stored).
- Three seeds give a range, not a confidence interval; image-level counts with correlated frames.

## Part 2: the historical combined AttackNetV2.2 on the N1 test, two image-preparation variants

### Model, chain and rule (fixed before the test)

- Checkpoint `AttackNetV2_2_combined_all_final.h5`, 201,809,888 bytes, SHA-256
  `3dd0c85341e323a5d1c0342b1196d3825af41ceb286798b62779a391bdcf6039`, shared by the coordinator (Drive, 7 July 2026),
  loaded as saved with `tf.keras.models.load_model(compile=False)` on the CPU (Keras 2.15 HDF5, 16,806,722
  parameters, input 256×256×3, `Dense(2, softmax)`); training data per the historical code: 3DMAD, CSMAD,
  Replay-Attack, MSSpoof, Our Dataset (no NUAA). Classes 0 bona fide, 1 attack; score = softmax column 1.
- Chain of the historical `create_datasets.process_single_image` and `dataset_loader._normalize_image`: BGR to RGB,
  `cv2.resize` 256×256 `INTER_LANCZOS4`, `advanced_image_enhancement`, uint8, then (x / 255 − 0.5) × 2. No quality
  filter: every one of the 3,000 frames is scored (declared deviation from the historical 0.65 filter).
- Variant `whole_image`: the whole original frame. Variant `face_crop`: the face found by the current application
  detector (`src/tesi_app/face.py`, SHA-256 `ca0d41cb2a83dfe6369c970f222862c67f5f06ea2e9d8d0332ed670ae31af536`;
  RetinaFace via OpenCV DNN, Haar when RetinaFace finds nothing), `Face.crop(margin=0.15, square=True)`, the whole frame
  when no face is found (recorded per frame), then the same chain.
- Threshold 0.5 fixed before the test (the historical argmax rule, the N1 rule); not calibrated on the historical
  source validation (not available here) nor on this test; no score inversion. Same 3,000 frames as N1 (`nuaa_cnn_test`,
  originals verified by SHA-256), the C1 300 subset as a secondary set.

### Results (study `20261029-N2-combined-247dfd7`, `tables/combined/aggregate.md`)

| model / variant | set | n bona fide | n attack | rejected bona fide | accepted attacks | APCER | BPCER | ACER | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| combined AttackNetV2.2, whole image | test (3,000 frames of nuaa_cnn_test) | 1500 | 1500 | 1492 | 222 | 0.1480 | 0.9947 | 0.5713 | 0.2169 |
| combined AttackNetV2.2, whole image | C1 subset (300 frames of nuaa_c1_v2, inside the test) | 150 | 150 | 149 | 23 | 0.1533 | 0.9933 | 0.5733 | 0.1975 |
| combined AttackNetV2.2, face crop | test (3,000 frames of nuaa_cnn_test) | 1500 | 1500 | 1224 | 353 | 0.2353 | 0.8160 | 0.5257 | 0.3706 |
| combined AttackNetV2.2, face crop | C1 subset (300 frames of nuaa_c1_v2, inside the test) | 150 | 150 | 125 | 36 | 0.2400 | 0.8333 | 0.5367 | 0.3499 |
| N1 LivenessNet, seed 42 | test (3,000 frames of nuaa_cnn_test) | 1500 | 1500 | 1155 | 160 | 0.1067 | 0.7700 | 0.4383 | 0.3485 |
| N1 LivenessNet, seed 42 | C1 subset (300 frames of nuaa_c1_v2, inside the test) | 150 | 150 | 117 | 13 | 0.0867 | 0.7800 | 0.4333 | 0.3374 |
| N1 LivenessNet, seed 43 | test (3,000 frames of nuaa_cnn_test) | 1500 | 1500 | 1152 | 391 | 0.2607 | 0.7680 | 0.5143 | 0.3754 |
| N1 LivenessNet, seed 43 | C1 subset (300 frames of nuaa_c1_v2, inside the test) | 150 | 150 | 117 | 38 | 0.2533 | 0.7800 | 0.5167 | 0.3714 |
| N1 LivenessNet, seed 44 | test (3,000 frames of nuaa_cnn_test) | 1500 | 1500 | 1131 | 72 | 0.0480 | 0.7540 | 0.4010 | 0.4582 |
| N1 LivenessNet, seed 44 | C1 subset (300 frames of nuaa_c1_v2, inside the test) | 150 | 150 | 115 | 6 | 0.0400 | 0.7667 | 0.4033 | 0.4641 |

Image preparation: whole image, no detector; face crop: Haar 3, RetinaFace 2996, none 1 (1 frame without a face, whole frame used).

### Reading and recommendation

- Whole image: the checkpoint calls almost every NUAA test frame an attack (BPCER 0.995, 1,492 of 1,500 bona fide
  rejected; APCER 0.148) and ranks bona fide frames above attacks (AUROC 0.217). Face crop: the collapse is smaller
  (BPCER 0.816, APCER 0.235, ACER 0.526) and the ranking improves (AUROC 0.371) but stays below 0.5. The C1 300 subset is
  within about 0.02 of the full test for both variants (largest difference 0.021, the AUROC of the face crop).
- Against N1 (ACER 0.401-0.514, AUROC 0.348-0.458 on the same frames): neither variant improves on N1 on ACER or AUROC;
  the face-crop ACER is above the N1 range, its AUROC inside it; the whole-image variant is worse than every N1 seed on
  BPCER, ACER and AUROC, while its APCER (0.148) falls inside the N1 range (0.048-0.261).
- The image preparation matters: between the two variants BPCER changes by 0.18 and AUROC by 0.15, with the same
  checkpoint, frames and threshold. Score distributions (`figure_distributions.pdf`): with the whole image 98 % of the
  bona fide and 75 % of the attacks score above 0.9; with the face crop 14 % of the bona fide move below 0.1.
- Recommendation (to be verified on separate data, not a result of this test): if one variant is carried forward, the
  face crop, because it is the less degenerate of the two and the closer to the article's description (detected,
  aligned and cropped faces); at threshold 0.5 neither variant is usable on NUAA, and a threshold chosen on a source
  validation, as in the article's protocol, would be needed before any operating-point comparison. The result says
  nothing about the checkpoint on its own training datasets.

### Limits of part 2

- A sensitivity analysis to the image preparation, not a reproduction of the historical preparation: the
  documentation does not say whether the files given to `create_datasets.py` were already face crops; neither variant
  is a certain replica.
- The comparison with N1 changes architecture, training recipe and training data at once; it does not isolate the
  effect of the architecture.
- The threshold is not the historical one (chosen on the source validation in the article's protocol); 0.5 is the
  argmax rule of the historical evaluation code.
- Frames from a dataset absent from the model's training (NUAA), correlated at image level; no uncertainty beyond the
  two variants.

## Restricted area (not in this repository)

`02_Experiments/N2/20261029-N2-sessions-2ac3e29/`: `status.json`, `check.json`, `check.log`, `target_manifest.csv` (491 rows: key,
label, subject, session, file SHA-256 and size), `parity_keys.csv` (24 rows), `cache_check.json`, `cache_check.log`, `frames_target.csv` (per frame:
detector, file and frame SHA-256), `parity.json`, `parity.csv` (24 keys × 3 seeds: stored and recomputed scores),
`seed<N>/` (`status.json`, `score.log`, `score.json`, `scores_sessions.csv` with 491 rows: key, subject, session, label,
group, M1-dev flag, score with exact round trip, decision at 0.5, detector, file, frame, checkpoint and configuration
SHA-256; `per_subject.csv`; `metrics.json`), `aggregate.log`, `aggregate.json`, `aggregate.md`,
`per_subject_all_seeds.csv`, `strata_all_seeds.csv` (subject × class × set × seed: n, errors, error rate or NA), the
three figures (the two public ones and the per-subject `figure_strata.png`). `02_Experiments/N2/20261029-N2-combined-247dfd7/`: `status.json`, `check.json`, `check.log`, `<variant>/` (`status.json`,
`score.log`, `score.json`, `scores_test.csv` with 3,000 rows, `frames.csv`, `per_subject.csv`, `metrics.json`),
`aggregate.log`, `aggregate.json`, `aggregate.md`, `per_subject_all_variants.csv`, the figure. `SHA256SUMS` of each
folder next to it.


Per-image records and `research/templates/score-schema.json`: own columns, as in N1 (mapping: `sample_id` = manifest
key, `dataset_id` = `nuaa`, `split` = `train` for part 1 (official list; sessions 01-02) and `test` for part 2,
`attack_type` = `print` for label 1, `modality` = `RGB`, `threshold` = 0.5 fixed a priori; `prediction` = 1 iff
`score_attack > 0.5`).

## Commands

From `livenesslab/` of this folder (in the development repository, from its root). Part 1, with the restricted inputs
(M2 manifests folder, the cache, the original images, the N1 study folder with its checkpoints and scores) and the
reference code (`LIVENESSLAB_LIVEDETECTION` pointing to the folder with `scripts/create_datasets.py` and
`src/architectures.py`; `LIVENESSLAB_SILENT_FACE` to the Silent-Face-Anti-Spoofing folder with the RetinaFace model):

```
C=(--config ../n2_sessions.json --manifests <M2 manifests folder> --cache <path>/nuaa_1500.npz --images <NUAA images root> --n1-study <path>/20261024-N1-LivenessNet-fb93496 --frozen-face ../face_0ac7143.py --study-dir <outside the repository>/YYYYMMDD-N2-sessions-<shortcommit>)
python scripts/n2_sessions.py check "${C[@]}"
python scripts/n2_sessions.py cache-check "${C[@]}"
python scripts/n2_sessions.py parity "${C[@]}"
for s in 42 43 44; do python scripts/n2_sessions.py score "${C[@]}" --train-seed $s --i-am-authorized; done
python scripts/n2_sessions.py aggregate "${C[@]}"
```

Part 2:

```
C=(--config ../n2_combined.json --manifests <M2 manifests folder> --images <NUAA images root> --model <path>/AttackNetV2_2_combined_all_final.h5 --n1-study <path>/20261024-N1-LivenessNet-fb93496 --study-dir <outside the repository>/YYYYMMDD-N2-combined-<shortcommit>)
python scripts/n2_combined.py check "${C[@]}"
for v in whole_image face_crop; do python scripts/n2_combined.py score "${C[@]}" --variant $v --i-am-authorized; done
python scripts/n2_combined.py aggregate "${C[@]}"
```

Every command ends with `RESULT: PASS (<command>)` or `RESULT: FAIL (<command>)` (exit code 1; a wrong command line is
refused by the argument parser with exit code 2 before any step starts). The study folder must be new at `check`;
inside the development repository only `results/n2/` is accepted, inside the group repository no output folder is
accepted. A non-trial study requires the delivered configuration (SHA-256 bound in the code) and the current commit in
its name; the working-tree state and the SHA-256 of every code file are recorded at `check` and compared at every later
step; the files written by `check` (the target manifest, the parity list) and by every later step are verified by
SHA-256 before the next step, and `parity.json` is bound to the `cache_check.json` it was computed from. `check` imports
neither TensorFlow (asserted at run time) nor the detector (the step has no code path to it); `cache-check` uses the
detector only; `parity` and
`score` use TensorFlow on the CPU. The outputs of `cache-check`, `parity` and `score` are never recomputed in place:
after a `FAIL` the folder keeps its log and `status.json` (`FAIL`, with the step and the reason) and the step is run
again in a new study folder (or, for `score`, after removing the failed `seed<N>` folder by hand); `status.json` at the
root of the study records the last step run at the root (`check`, `cache-check`, `parity`, `aggregate`), each
`seed<N>/status.json` its own `score`. `aggregate` removes its previous outputs before recomputing.

## Tests

From `research/n2-nuaa-check/livenesslab/` (`python` is Python 3.11 or later with NumPy and scikit-learn; data,
weights, results folders, TensorFlow and OpenCV are not needed: without OpenCV the suites register an empty `cv2`
module and print a `NOTE` line):

```
python scripts/check_n2_sessions.py
python scripts/check_n2_mutations.py
python scripts/check_n2_combined.py
python scripts/check_n2_combined_mutations.py
```

Expected last lines (also in `test_logs/`, with the Linux run in `test_logs/linux.txt`):

```
tests run: 93, failures: 0, errors: 0, skipped: 0
RESULT: PASS
```

```
unmutated copy (after): PASS (tests run: 93, failures: 0, errors: 0)
mutations killed: 85/85
control mutations survived: 1/1
RESULT: PASS
```

```
tests run: 33, failures: 0, errors: 0, skipped: 0
RESULT: PASS
```

```
unmutated copy (after): PASS (tests run: 33, failures: 0, errors: 0)
mutations killed: 43/43
control mutations survived: 1/1
RESULT: PASS
```

The mutation harnesses are those of N1: every mutation in a fresh copy of the code, the unmutated suite in a fresh copy
before and after, a control mutation (a comment only) that must survive, a mutation counted as killed only when the
suite runs and reports failures or errors with the same number of tests. The copies include the delivered
configurations and the frozen module (`delivery/n2/` in the development repository, `../` here), so that the two
`test_delivered_configuration` tests run: they check the configurations against the schema and against the SHA-256
bound in the code, and the frozen module against its SHA-256. The suites inject the preprocessing, the scorer, the
figures and the repository state: no image, no model, no detector, no TensorFlow. The only possible skips are the two
`test_delivered_configuration` when the configurations are not next to the code.

## Files and integrity

`SHA256SUMS` lists the 41 files of `research/n2-nuaa-check/` other than itself (42 files in all); check
with `shasum -a 256 -c SHA256SUMS` from this folder. No file of the folder is ignored by the `.gitignore` of the group
repository; the tables are in `tables/` because `results/` is ignored there.
