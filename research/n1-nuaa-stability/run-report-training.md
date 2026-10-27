# Run report

## Identity
- Task and run ID: N1 part 2 (small LivenessNet training pilot on NUAA) — `20261024-N1-LivenessNet-fb93496`. The group
  format is `YYYYMMDD-task-seed-shortcommit`; this study id names the architecture instead of a seed because it holds
  three trainings, one per training seed, in the subfolders `seed42/`, `seed43/`, `seed44/` (split seed 42 for all).
- Date / operator: 2026-10-24, check 20:56, trainings and scores 20:56–21:28, aggregate 21:35 (records) / Valerio
  Cassano.
- Git commit / environment lock: development repository commit `fb9349685dfc15fae99f97706ea86f4eabd67c6b`, clean
  working tree (`check.json`: `dirty: false`, no modified path); `livenesslab/requirements.txt`. SHA-256 of the nine
  code files recorded in `check.json` and compared at every step; equal to the files of this folder (`n1_train.py`,
  `n1_trainer.py`, `nuaa_manifests.py`, `select_threshold.py`, `path_guard.py`, `m2_writers.py`, `package_rules.py`,
  `evaluation.py`, `paths.py`); reference `architectures.py` `a83c02b0…` (not part of this folder).
- Command: from the development repository root (= `livenesslab/` in the group repository), bash/zsh array form:
  `C=(--config delivery/n1/n1_nuaa_stability.json --manifests results/m2/manifests --cache data/processed/nuaa_1500.npz --study-dir results/n1/20261024-N1-LivenessNet-fb93496)`;
  `python scripts/n1_train.py check "${C[@]}"`; for s in 42 43 44:
  `python scripts/n1_train.py train "${C[@]}" --train-seed $s --i-am-authorized` and
  `python scripts/n1_train.py score "${C[@]}" --train-seed $s --i-am-authorized`;
  `python scripts/n1_train.py aggregate "${C[@]}"`. Every step `RESULT: PASS`.
- Configuration path and SHA-256: `n1_nuaa_stability.json` (`delivery/n1/` in the development repository, the root of
  this folder in the group repository) — `ffaf934b9a24ecc7ae9643ebadef318f566e12d8d6ea7935281754c0475ccfaa`, written
  before the runs; values frozen in `n1_train.py`.
- Hardware / software: development Mac (Apple M4), macOS; recorded in each `train.json`: Python 3.11.16, TensorFlow
  2.15.0, tensorflow-metal 1.1.0 (installed, GPU hidden), Keras 2.15.0, NumPy 1.26.4, scikit-learn 1.3.2, visible
  devices `['CPU']`. Training and scoring on the CPU only (see "Limitations").
- Checkpoint origin and SHA-256: new trainings of this run from the architecture initialisation (no pretraining); best
  `val_loss` weights, `.h5`: seed 42 `c63a858107b6f60375a27222a915e9b4b35f267fe9f34cf16117d673ca06546e`, seed 43
  `34bac22cafd3f4470895ee1ea3e22d86e8cd67139e036491fc94955925d9f0d6`, seed 44
  `41c128484ff76b4d754c75085f2c17f95c22897a59f34d8d27e7f4b940d23432`; `.keras` `b9115fc6…`, `fce40356…`, `54241106…`;
  cards `5c828deed64350d1b0f0d1a73c320042eab6e95944d03e417777855a595ddbe9`,
  `845103da264c8e06f90a56720df1a27ae8fb6476f16d1b4a29166f759179f586`,
  `1c4096a34cc0d2f3da6aa9ae4f14e2b8bb17b7115b83f6cae96495d379959ca7`. Cards and SHA-256 are in the
  restricted folder; the weights are kept by the operator, within the authorised scope, available to the coordinator
  on request, never uploaded publicly; the four previous LivenessNet NUAA checkpoints of C1 are refused by hash.
- Pretraining / external data: none.

## Data and protocol
- Dataset version / authorization register entry: NUAA, register entry `dataset_register_nuaa.md` of the accepted M2
  package (run `20261006-M2-seed42-36c677a`); local cache `nuaa_1500.npz`
  `dc286dc371e61d638da1842c062a38c5368a9ec4da07ccbc36b7fff6f3177920` (members `X_train`/`X_test` uint8 3000×256×256×3,
  `y_*` int64, `s_*` <U4, read from the `.npy` headers, no pickle).
- Manifest and split SHA-256: M2 manifests `nuaa_official_train.csv` `2a7e8906…`, `nuaa_official_test.csv`
  `2fad23ff…`, `nuaa_cnn_train.csv` `6d46ec7d…`, `nuaa_cnn_early_stopping.csv` `17282be8…`, `nuaa_cnn_test.csv`
  `ce9669dc…`, `nuaa_c1_v2.csv` `1c2cf97c…` (full values in the configuration and in `check.json`).
- Independent unit / train-dev-test counts: unit = image frame; 2,507 fitting frames (7 subjects), 493 validation
  frames (2 subjects, early stopping), 3,000 test frames (1,500 bona fide, 1,500 attacks, 16 subjects); secondary C1
  subset of 300 test frames (150 + 150) inside the test.
- Subject, session, video overlap and duplicate checks: `check` (repeated in `train` and `score`): sampling order
  `spread(…, 1500)` per class on the official lists equal to `cnn_train ∪ cnn_es` and to `cnn_test`; the cache label
  and subject arrays equal, element by element, to the manifest rows in that order; GroupShuffleSplit with split seed
  42 equal to `cnn_train` / `cnn_es` by key and by file SHA-256; no subject shared by fitting and validation; C1
  inside the test; 0 duplicated derived frames, 0 shared between fitting, validation and test. Fitting and validation
  come from capture sessions 01–02, the test from session 03; the 16 test subjects include the 7 fitting and the 2
  validation subjects (NUAA official splits differ by session, not by subject).
- Label mapping / score direction: label 0 bona fide, 1 attack; one-hot index 0 bona fide, 1 attack; score = softmax
  output 1, probability of attack.
- Crop, resize, orientation, channel order, normalization: frames of the cache as built for the C1 checkpoints (face
  crop with a 15 % square margin, 256×256 `INTER_AREA`, the reference enhancement, RGB order), divided by 255;
  training augmentation only on the fitting set (flip, rotation 0.03, zoom 0.12, translation 0.06, brightness 0.15,
  contrast 0.2, Gaussian noise 0.02).
- Model selection, threshold selection and calibration: LivenessNet only; at most 12 epochs, batch 16, legacy Adam
  (initial learning rate 0.001, clipnorm 1.0), categorical cross-entropy, balanced class weights, ReduceLROnPlateau
  (0.5, patience 2, min 1e-6), early stopping on `val_loss` with patience 4, weights of the epoch with minimum
  `val_loss`; no seed selection; fixed threshold 0.5 chosen a priori, no calibration, no M1 threshold reused.
- Attack decision at threshold (including equality): attack if `score_attack > 0.5`; a tie is bona fide (0 ties in
  every set).
- Source-only development rule for external targets: not applicable (intra-dataset NUAA; the test is not used for any
  choice).

## Evidence
- Restricted score file location and SHA-256: `02_Experiments/N1/20261024-N1-LivenessNet-fb93496/` (with the three
  checkpoint cards `seed<N>/checkpoints/LivenessNet__nuaa.json`, without `.h5`/`.keras`; the first execution is in
  `02_Experiments/N1/20261024-N1-LivenessNet-2c44c64/`), with its `SHA256SUMS`
  `668e3fec4b68758741160c00850a648db8669f64881f104048b631fbea07ace0` (34 files); per-image scores `scores_test.csv`:
  seed 42 `60e6c401…`, seed 43 `a1a71b6e…`, seed 44 `943d1b46…`; per subject `per_subject.csv` `22848d2d…`,
  `bbd82453…`, `d602336f…`; `per_subject_all_seeds.csv` `b2af6cb7…`; `check.json` `fd93353e…`.
- APCER / BPCER / ACER with denominators (threshold 0.5):

  | set | seed | accepted attacks / attacks | rejected bona fide / bona fide | APCER | BPCER | ACER | AUROC |
  |---|---|---|---|---|---|---|---|
  | test 3,000 | 42 | 160/1500 | 1155/1500 | 0.1067 | 0.7700 | 0.4383 | 0.3485 |
  | test 3,000 | 43 | 391/1500 | 1152/1500 | 0.2607 | 0.7680 | 0.5143 | 0.3754 |
  | test 3,000 | 44 | 72/1500 | 1131/1500 | 0.0480 | 0.7540 | 0.4010 | 0.4582 |
  | C1 300 | 42 | 13/150 | 117/150 | 0.0867 | 0.7800 | 0.4333 | 0.3374 |
  | C1 300 | 43 | 38/150 | 117/150 | 0.2533 | 0.7800 | 0.5167 | 0.3714 |
  | C1 300 | 44 | 6/150 | 115/150 | 0.0400 | 0.7667 | 0.4033 | 0.4641 |

  Median (range) on the test: APCER 0.1067 (0.0480–0.2607), BPCER 0.7680 (0.7540–0.7700), ACER 0.4383
  (0.4010–0.5143), AUROC 0.3754 (0.3485–0.4582). No class absent.
- Attack types / acquisition conditions: NUAA print attacks only; test from capture session 03.
- Seed(s) / uncertainty / exclusions: training seeds 42, 43, 44 (random, NumPy, TensorFlow, `tf.data` shuffle); split
  seed 42. Training: 9 / 12 / 12 epochs, restored epoch 5 / 12 / 8 (early stopping / end of epochs / early stopping),
  minimum `val_loss` 0.000898 / 0.012337 / 0.000542, recomputed on the CPU with a largest difference of 5.96e-08; wall
  time 496.0 / 671.0 / 671.4 s. Three seeds give a range, not a confidence interval. No frame excluded.
- Runtime and latency protocol, if relevant: not relevant (training wall times recorded only).
- Reproduction command / logs: commands above; logs `check.log`, `seed<N>/train.log`, `seed<N>/score.log`,
  `aggregate.log` in the restricted folder. A first execution from a modified tree (`20261024-N1-LivenessNet-2c44c64`,
  kept unchanged) gave the same `.h5` SHA-256 for the three seeds, the same `scores_test.csv` apart from `run_id`, and
  the same aggregate apart from the wall times, on this machine with the same software (the `.keras` archives differ
  in their stored metadata). Tests: `check_n1_train.py` 91, `check_n1_mutations.py` 72/72 with the control mutation
  surviving (`test_logs/`).

## Interpretation
- Research question answered: how much the result of the current LivenessNet training recipe on NUAA varies with the
  training seed, with split, preprocessing and protocol fixed.
- Observed result and comparison: near-zero validation loss in every run, but poor test results: BPCER 0.754–0.770,
  AUROC below 0.5 for every seed; the variation between seeds is mostly in APCER (0.048–0.261). The poor test results
  are not limited to new people: on the 1,325 test frames of the 7 fitting subjects BPCER is 0.687 for every seed and
  AUROC 0.319–0.453. The C1 subset is within 0.02 of the full test (a subset, not independent evidence). The earlier C1
  LivenessNet checkpoint scored through the same CPU path gives AUROC 0.3910 on the 3,000 frames and 0.3813 on the C1
  300 (`n1_metal_checks.py c1-rescore`), descriptive only (different checkpoint rule and training device).
- Limitations / failed checks: test subjects already seen in training; possible explanations of the poor test results,
  to be verified and not tested causes: the session shift between training (01–02) and test (03), and the weak
  validation (2 subjects and 493 frames, `val_loss` very unstable, `val_accuracy` 0.6856, accuracy matching an
  all-attack baseline, in 5 of 33 epochs), which may make the selected epoch noisy; the C1
  subset is secondary; image-level counts with correlated frames. Device: with TensorFlow 2.15 and tensorflow-metal
  the ReLU after Dense (MatMul+BiasAdd+ReLU) is dropped on the GPU in graph mode; eager GPU and CPU are correct; only
  `disable_meta_optimizer` fixes it; the AttackNets are unaffected; hence CPU only. Established with the scripts of
  this folder: the defect (`n1_metal_relu_repro.py`); with the C1 LivenessNet weights, GPU `predict` equals the CPU
  network with `dense1` linear (`n1_metal_checks.py c1-linear`); `fit` on the GPU trains a different network
  (`fit-steps`); the C1 scores agree closely with a CPU re-scoring and, from reading the code, were computed in eager
  mode. Not established: the device behind the C1 card metrics (AUC 0.6023, computed with `predict`) and the training
  device of the C1 LivenessNet checkpoint (not recorded; GPU likely); the effect on the real-data metrics of C1
  LivenessNet; the cause inside tensorflow-metal. Bit-for-bit reproducibility is shown only on this machine and
  software. C1 subset: the 300 C1 images are verified against the official source (M2); this verification is not
  extended to the whole cache, whose frames are linked to the manifests by sampling order and by the label and subject
  arrays. Per-image records: own columns instead of `templates/score-schema.json`, mapping in the README ("Restricted
  area"); a schema-conformant export can be added as a follow-up. No failed check.
- Next decision: to be discussed with the reviewer: validation design and session shift on NUAA; whether earlier
  GPU-trained LivenessNet checkpoints need re-examination; a separate task for the accepted `train_cnn.py` and the C2
  runner (`c2_pilot.py` trains LivenessNet through `train_cnn.train` without hiding the GPU), which are exposed to the
  same defect for LivenessNet on Apple-silicon GPUs (from reading the code and the minimal reproduction; not run on C2
  data); N1 does not change them, as the card requires.
- Reviewer and review outcome: pending.
