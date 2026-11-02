# Run report

## Identity
- Task and run ID: C1 crop check (the training-cache face extractor against the C1 evaluation extractor on AttackNet
  V2.1 official NUAA and the 300 C1 images; task of the coordinator of 2026-11-01) — `20261102-C1CROP-f6747cc`. The group format is
  `YYYYMMDD-task-seed-shortcommit`; this study id omits the seed because the run has no random component (an existing
  checkpoint, deterministic preprocessing, CPU inference).
- Date / operator: `2026-11-02` (`status.json`, `check.json`, `aggregate.json`) / Valerio Cassano.
- Git commit / environment lock: development repository commit `f6747cc9e56657976403a63f6f85a16720650149` (`check.json`: `commit`, `dirty` false; a
  non-trial run is refused unless the work tree is clean); `livenesslab/requirements.txt`. SHA-256 of the twelve code
  files, of the two frozen extractor copies and of the reference `create_datasets.py` (`306ea1c8…`) and
  `architectures.py` (`a83c02b0…`) recorded in `check.json` and compared at every step; equal to the files of this
  folder.
- Command: from the development repository root (= `livenesslab/` in the group repository):
  `C=(--config delivery/c1crop/c1crop.json --manifests results/m2/manifests --images data/NUAA/images --cache data/processed/nuaa_1500.npz --weights models/weights/AttackNetV2_1__nuaa.h5 --c1-scores results/c1/20260928-C1-seed42-a80f6f9/scores.csv --face-cache delivery/c1crop/face_0ac7143.py --face-c1 delivery/c1crop/face_a80f6f9.py --study-dir results/c1crop/20261102-C1CROP-f6747cc)`;
  `python scripts/c1crop_check.py check "${C[@]}"`; `frames`; `parity`; `score "${C[@]}" --i-am-authorized`;
  `aggregate`. Every step `RESULT: PASS`.
- Configuration path and SHA-256: `c1crop.json` — `caa1156ac7608089790c12b337040cfc4f2443611be7735599ebe32b75d8e4ae`, written before the run and bound in
  `c1crop_check.py`.
- Hardware / software: development Mac (Apple M4), macOS; Python 3.11.16, TensorFlow 2.15.0, Keras 2.15.0,
  tensorflow-metal 1.1.0 installed, NumPy 1.26.4, scikit-learn 1.3.2, OpenCV 4.11.0. Scoring of the study on the CPU
  only (GPU hidden), batch 16, BatchNorm in inference mode; the informative reproduction of `parity` on the default
  device of the process (here the Metal GPU), one image at a time.
- Checkpoint origin and SHA-256: `AttackNetV2_1__nuaa.h5`, 134,417,864 bytes, SHA-256
  `67d9138d6c43e33164d4510b4b34868c2fdc0a637f27664816c388c4e059f8dc`, the C1 checkpoint (trained on 2026-09-12 on the
  training cache; its SHA-256 is the fingerprint recorded in the 300 C1 rows of `attacknet_v2_1__nuaa`); loaded as the
  application does (reference `architectures.py`, `create_model('AttackNetV2_1').get_model().load_weights`, Grappler
  remapping off); SHA-256 and size verified at every step. No retraining, no fine-tuning.
- Pretraining / external data: none; the checkpoint's training data are the NUAA training cache of C1 (not used here).

## Data and protocol
- Dataset version / authorization register entry: NUAA, register entry of the accepted M2 package (run
  `20261006-M2-seed42-36c677a`); originals under `data/NUAA/images/` verified by SHA-256 against the manifests.
- Manifest and split SHA-256: `nuaa_official_test.csv` `2fad23ff…`, `nuaa_cnn_test.csv` `ce9669dc…` (the 3,000 N1 test
  frames), `nuaa_c1_v2.csv` `1c2cf97c…` (the 300 C1 images, inside the test; `c1` rows equal to their `cnn_test` rows,
  `cnn_test` rows equal to their `official_test` rows; sampling order of the official test list equal to `cnn_test`,
  the order of `X_test` in the cache). Cache `nuaa_1500.npz` `dc286dc3…` (members and shapes in the configuration).
  Stored C1 scores: `20260928-C1-seed42-a80f6f9/scores.csv` `0a52d875…`, the 300 rows of `attacknet_v2_1__nuaa`, each
  bound to the C1 run, the C1 configuration (`f0dfdb90…`), the checkpoint and its manifest row (key, subject, label;
  the `session_id` column of the C1 schema is the third field of the file name and is not compared).
- Independent unit / train-dev-test counts: unit = image frame; 300 test images (150 bona fide, 150 attacks, 16
  subjects, session 03). No training or development set here.
- Subject, session, video overlap and duplicate checks: no duplicated key or SHA-256 in the manifests; every original
  found with size and SHA-256 equal to the manifest; no duplicated preprocessed frame within an extractor (exact pixel
  hash); the checkpoint was trained on the official training sessions 01-02 (the C1 images are session 03 test frames,
  as in C1).
- Label mapping / score direction: label 0 bona fide, 1 attack; score = softmax column 1 (probability of attack), as in
  C1 and N1.
- Crop, resize, orientation, channel order, normalization: `train_cnn.py` `preprocess`, unchanged: `cv2.imread` (BGR),
  detector (`face_0ac7143.py` for `cache_0ac7143`, `face_a80f6f9.py` for `c1_a80f6f9`: RetinaFace via OpenCV DNN with
  the Haar cascade as fallback, confidence 0.6, boxes under 16 px rejected), `Face.crop(margin=0.15, square=True)`, the
  whole frame when no face is found, `cv2.resize` 256×256 `INTER_AREA`, BGR to RGB, `advanced_image_enhancement` of the
  reference `create_datasets.py`, uint8, float32 / 255 at scoring. The only difference between the two extractors is the
  RetinaFace box (`int(min(W, r) - x + 1)` against `int(min(W, r)) - x`, and the same for the height): one pixel
  narrower and one pixel shorter with the C1 extractor. Detectors on the 300: RetinaFace 300 with both extractors; box
  delta dx 0, dy 0, dw −1, dh −1 for all 300.
- Model selection, threshold selection and calibration: none; threshold 0.5 fixed in C1 (`nuaa_config.json`), unchanged;
  both extractors reported; no score inversion.
- Attack decision at threshold (including equality): attack if `score_attack > 0.5`; a tie is bona fide (0 ties).
- Source-only development rule for external targets: the threshold is fixed a priori; nothing is chosen on these images.

## Evidence
- Restricted score file location and SHA-256: `20261102-C1CROP-f6747cc/` in the restricted area (shared N1 folder, as for N2, status on
  the C2 card): `scores_paired.csv` `980fdcc9…`, `parity.csv` `49c7b3c0…`, `frames.csv` `2e05bd9b…`,
  `per_subject.csv`, `check.json` `c21a1666…`, folder `SHA256SUMS` `e058073c…` (`18` files).
- APCER / BPCER / ACER with denominators: `cache_0ac7143`: APCER 0.0267 (4/150), BPCER 0.8067 (121/150), ACER 0.4167,
  AUROC 0.7092; `c1_a80f6f9`: APCER 0.0200 (3/150), BPCER 0.7667 (115/150), ACER 0.3933, AUROC 0.7103, equal to the
  delivered C1 row to four decimals (same 115 and 3, same 300 decisions; AUROC 0.71029 against 0.71027). Paired (cache minus C1): maximum absolute difference 0.3214,
  median 8.6e-6, mean 0.0142; 97 images with the same score, 18 above 0.1; decisions changed 9/300 (7 bona fide
  accepted → rejected, 1 bona fide rejected → accepted, 1 attack rejected → accepted).
- Attack types / acquisition conditions: NUAA printed photographs re-captured by webcam (attack type `print`), session
  03 images of the official test list, as in C1.
- Seed(s) / uncertainty / exclusions: no random component; no exclusion (every one of the 300 images scored with both
  extractors); no uncertainty estimate (a paired comparison on 300 images of one session).
- Runtime and latency protocol, if relevant: each step under one minute on the CPU (not a latency measurement).
- Reproduction command / logs: the commands above; `check.log`, `frames.log`, `parity.log`, `score.log`,
  `aggregate.log` and `status.json` in the restricted folder; `test_logs/frames_log.txt` and `parity_log.txt` here.
  Reconstructions: cache chain 300/300 frames equal bit for bit (`frames.json`); C1 chain 300/300 scores within 1e-5 of
  the stored C1 scores (maximum 2.891e-6, 292 within 1e-6, 209 within 1e-7), 300/300 same decisions; in the C1 conditions
  (GPU, one image at a time, informative) maximum 8.941e-8, 300/300 within 1e-7; on the CPU one image at a time
  (informative) maximum 6.998e-5, 297/300 within 1e-5, 300/300 same decisions (`parity.json`); a one-off measurement of
  2026-11-02 on the GPU with batches of 16 gave 1.1e-5 (not recorded by the study). The tolerance 1e-5 is the one of the
  coordinator's R2 replication package (1e-5 absolute, no discordant decision, same metric row to four decimals), met in
  the scoring conditions of the study; a first trial with 1e-6 was refused (8 of 300 CPU batch-16 scores between 1e-6
  and 2.9e-6, all decisions equal) and the tolerance was set before the official run, with the reason recorded in the
  configuration. A CPU replication one image at a time can differ from the stored C1 scores by up to 7e-5 on 3 images
  with every decision unchanged.

## Interpretation
- Research question answered: how much the one-pixel difference of the RetinaFace box between the training cache
  (extractor of 12 September) and the C1 evaluation (extractor of 24 September) changes the scores and the decisions
  of AttackNet V2.1 on the 300 C1 images, everything else unchanged.
- Observed result and comparison: with the training-consistent extractor the checkpoint is slightly worse at the fixed
  threshold (ACER 0.4167 against 0.3933: 6 more bona fide rejected and 1 more attack accepted out of 150 each), AUROC
  unchanged within 0.002; six images in ten keep the same score within 1e-4, 18 move by more than 0.1, 9 cross the
  threshold. The one-pixel box does not explain the C1 result (BPCER above 0.76 with either extractor). The delivered
  C1 row is reproduced by the C1 extractor to four decimals.
- Limitations / failed checks: one checkpoint, one subset of 300 images of one session, no uncertainty estimate; no
  Haar or no-face case among the 300; the C1 scores came from the GPU one image at a time while this study scores on the
  CPU in batches (bounded by the parity check); no failed check in the official run; the first trial with tolerance
  1e-6 is reported above.
- Next decision: for the coordinator; the other three C1 CNNs and the M1 dev scores (same detector of 24 September)
  can be recomputed with the same script in a separate run if useful. C1, M1, M2, N1, N2, the C2 runner and the R2 reference are unchanged.
- Reviewer and review outcome: pending.
