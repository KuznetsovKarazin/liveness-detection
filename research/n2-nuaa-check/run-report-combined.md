# Run report

## Identity
- Task and run ID: N2 part 2 (the historical combined AttackNetV2.2 on the 3,000 NUAA test frames of N1, two
  image-preparation variants) — `20261029-N2-combined-247dfd7`. The group format is `YYYYMMDD-task-seed-shortcommit`; this study id
  omits the seed because the run has no random component (an existing checkpoint, deterministic preprocessing, CPU
  inference); the two variants are in `whole_image/` and `face_crop/`.
- Date / operator: 2026-10-29 (`status.json`, `check.json`, `aggregate.json`) / Valerio Cassano.
- Git commit / environment lock: development repository commit `247dfd79d565985a4f3133b553fee36a27a4aa26` (`check.json`: `commit`, `dirty`
  false; this commit adds to `2ac3e29`, the commit of part 1, only the results folder of part 1: the eleven code files have
  the same SHA-256 in both studies); `livenesslab/requirements.txt`. SHA-256 of the eleven code files recorded in `check.json` and compared at
  every step; equal to the files of this folder, plus the reference `create_datasets.py` (`306ea1c8…`) and
  `dataset_loader.py` (`169365b5…`).
- Command: from the development repository root (= `livenesslab/` in the group repository):
  `C=(--config delivery/n2/n2_combined.json --manifests results/m2/manifests --images data/NUAA/images --model models/historical/n2/AttackNetV2_2_combined_all_final.h5 --n1-study results/n1/20261024-N1-LivenessNet-fb93496 --study-dir results/n2/20261029-N2-combined-247dfd7)`;
  `python scripts/n2_combined.py check "${C[@]}"`; for v in whole_image face_crop:
  `score "${C[@]}" --variant $v --i-am-authorized`; `aggregate "${C[@]}"`. Every step `RESULT: PASS`.
- Configuration path and SHA-256: `n2_combined.json` — `0527cc91c88700b94550a178bb78f890e8cc1d987bf1ef932b30afb61567c154`,
  written before the runs and bound in `n2_combined.py`.
- Hardware / software: development Mac (Apple M4), macOS; Python 3.11.16, TensorFlow 2.15.0, Keras 2.15.0 (the
  checkpoint's version), tensorflow-metal 1.1.0 installed with the GPU hidden, NumPy 1.26.4, scikit-learn 1.3.2, OpenCV
  4.11.0. Inference on the CPU only, batch 16, BatchNorm in inference mode.
- Checkpoint origin and SHA-256: `AttackNetV2_2_combined_all_final.h5`, 201,809,888 bytes, SHA-256
  `3dd0c85341e323a5d1c0342b1196d3825af41ceb286798b62779a391bdcf6039`, shared by the coordinator (Google Drive, file of
  7 July 2026), downloaded on 29 October 2026 and verified by SHA-256 and size at every step; loaded as saved
  (`compile=False`, no custom objects); Keras 2.15.0, 16,806,722 parameters, input 256×256×3, `Dense(128, tanh)`,
  `Dense(2, softmax)`, BatchNorm layers inside, no preprocessing layer. No retraining, no fine-tuning.
- Pretraining / external data: the checkpoint's training data per the historical code (`combined_all`: 3DMAD, CSMAD,
  Replay-Attack, MSSpoof, Our Dataset), never available here; no NUAA in its training.

## Data and protocol
- Dataset version / authorization register entry: NUAA, register entry of the accepted M2 package (run
  `20261006-M2-seed42-36c677a`); originals under `data/NUAA/images/` verified by SHA-256 against the manifests.
- Manifest and split SHA-256: `nuaa_official_test.csv` `2fad23ff…`, `nuaa_cnn_test.csv` `ce9669dc…` (the 3,000 N1 test
  frames), `nuaa_c1_v2.csv` `1c2cf97c…` (the C1 300 subset, inside the test).
- Independent unit / train-dev-test counts: unit = image frame; 3,000 test frames (1,500 bona fide, 1,500 attacks, 16
  subjects, session 03); secondary C1 subset of 300 (150 + 150). No training or development set here.
- Subject, session, video overlap and duplicate checks: `c1` rows equal to their `cnn_test` rows, `cnn_test` rows equal
  to their `official_test` rows; every original found with size and SHA-256 equal to the manifest; after preprocessing
  no duplicated derived frame within each variant (exact pixel hash). The checkpoint never saw NUAA.
- Label mapping / score direction: label 0 bona fide, 1 attack, as in the historical code (`np.eye(2)` one-hot, class 1
  = attack); score = softmax column 1 (probability of attack), as in the historical evaluation (`proba[:, 1]`, argmax).
- Crop, resize, orientation, channel order, normalization: historical chain of `create_datasets.process_single_image`
  and `dataset_loader._normalize_image`: BGR to RGB, `cv2.resize` 256×256 `INTER_LANCZOS4`, `advanced_image_enhancement`
  (bilateral filter, CLAHE on L, unsharp mask, gamma 1.2, contrast/brightness), uint8, then (x / 255 − 0.5) × 2 at
  inference; no quality filter (declared deviation: every frame scored). Variant `whole_image`: the whole frame.
  Variant `face_crop`: the current application detector (`src/tesi_app/face.py` `ca0d41cb…`; RetinaFace via OpenCV
  DNN, Haar fallback), `Face.crop(margin=0.15, square=True)`, the whole frame when no face is found, then the same
  chain; detectors on the 3,000: Haar 3, RetinaFace 2996, none 1 (1 frame without a face, whole frame used).
- Model selection, threshold selection and calibration: none; threshold 0.5 fixed before the test (the historical
  argmax rule and the N1 rule), not calibrated on the historical source validation (not available) nor on this test;
  both variants reported.
- Attack decision at threshold (including equality): attack if `score_attack > 0.5`; a tie is bona fide (0 ties).
- Source-only development rule for external targets: the threshold is fixed a priori; nothing is chosen on the target.

## Evidence
- Restricted score file location and SHA-256: `02_Experiments/N2/20261029-N2-combined-247dfd7/`: `whole_image/scores_test.csv`
  `2fdbb0ca…`, `face_crop/scores_test.csv` `cc22b93e…`, `frames.csv` and `per_subject.csv` per variant, `check.json`
  `0399be93…`, folder `SHA256SUMS` `bfa3e257…` (22 files).
- APCER / BPCER / ACER with denominators (threshold 0.5), next to the N1 rows (study `20261024-N1-LivenessNet-fb93496`,
  `aggregate.json` `d74e7643…`):

  | model / variant | set | accepted attacks / attacks | rejected bona fide / bona fide | APCER | BPCER | ACER | AUROC |
  |---|---|---|---|---|---|---|---|
  | combined, whole image | test (3,000 frames of nuaa_cnn_test) | 222/1500 | 1492/1500 | 0.1480 | 0.9947 | 0.5713 | 0.2169 |
  | combined, whole image | C1 subset (300 frames of nuaa_c1_v2, inside the test) | 23/150 | 149/150 | 0.1533 | 0.9933 | 0.5733 | 0.1975 |
  | combined, face crop | test (3,000 frames of nuaa_cnn_test) | 353/1500 | 1224/1500 | 0.2353 | 0.8160 | 0.5257 | 0.3706 |
  | combined, face crop | C1 subset (300 frames of nuaa_c1_v2, inside the test) | 36/150 | 125/150 | 0.2400 | 0.8333 | 0.5367 | 0.3499 |
  | N1 LivenessNet seed 42 | test (3,000 frames of nuaa_cnn_test) | 160/1500 | 1155/1500 | 0.1067 | 0.7700 | 0.4383 | 0.3485 |
  | N1 LivenessNet seed 43 | test (3,000 frames of nuaa_cnn_test) | 391/1500 | 1152/1500 | 0.2607 | 0.7680 | 0.5143 | 0.3754 |
  | N1 LivenessNet seed 44 | test (3,000 frames of nuaa_cnn_test) | 72/1500 | 1131/1500 | 0.0480 | 0.7540 | 0.4010 | 0.4582 |

- Attack types / acquisition conditions: NUAA print attacks, session 03.
- Seed(s) / uncertainty / exclusions: no random component; no frame excluded (1 frame without a detected
  face scored on the whole frame in the `face_crop` variant, recorded); no uncertainty beyond the two variants.
- Runtime and latency protocol, if relevant: not relevant. The scripts record no durations; from the timestamps of the
  records of this study: about 1 minute (whole image) and 1.5 minutes (face crop) per variant on the CPU.
- Reproduction command / logs: commands above; logs `check.log`, `<variant>/score.log`, `aggregate.log` in the
  restricted folder. Tests: `check_n2_combined.py` 33 tests, `check_n2_combined_mutations.py` 43/43 with the
  control mutation surviving; same on Linux (`test_logs/linux.txt`).

## Interpretation
- Research question answered: how much the image preparation (whole frame against face crop) changes the behaviour of
  the historical combined AttackNetV2.2 on the NUAA test of N1, and whether that checkpoint improves on the N1 result,
  with the threshold fixed before the test.
- Observed result and comparison: whole image: APCER 0.148, BPCER 0.995, ACER 0.571, AUROC 0.217; face crop: APCER 0.235, BPCER 0.816, ACER 0.526,
  AUROC 0.371 (C1 300 within about 0.02, largest difference 0.021). Neither variant improves on the N1 rows on ACER or
  AUROC (ACER 0.401–0.514, AUROC 0.348–0.458); the
  preparation alone moves BPCER by 0.18 and AUROC by 0.15. Recommendation in the note: the face-crop variant, if one is
  verified on separate data; neither is usable at 0.5 on NUAA.
- Limitations / failed checks: sensitivity analysis to the image preparation, not a reproduction of the historical
  preparation (the documentation does not say whether the training files were already face crops); the comparison with
  N1 changes architecture, training recipe and training data at once; the threshold is not the historical
  source-validation one; NUAA is absent from the checkpoint's training; image-level counts with correlated frames; no
  seed or variance beyond the two variants. No failed check.
- Next decision: the coordinator's choice of the candidate variant to verify on separate data (the note recommends the face crop)
  and whether the preparation of the historical training files can be documented, which would replace the two variants
  with one.
- Reviewer and review outcome: pending.
