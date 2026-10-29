# N2 — NUAA diagnostic check: note

Study `20261029-N2-sessions-2ac3e29` (part 1, commit `2ac3e29`) and `20261029-N2-combined-247dfd7` (part 2, commit `247dfd7`, which
adds only the results of part 1; same code), 2026-10-29. Details, SHA-256 and
commands in `README.md`, `run-report-sessions.md` and `run-report-combined.md`.

## 1. Question

After N1 (three LivenessNet trainings on NUAA with near-perfect validation and poor session 03 test: ACER 0.401–0.514,
AUROC 0.348–0.458), the coordinator asked whether the same checkpoints work on frames of the training sessions (01–02)
that were used neither for fitting nor for early stopping, against the session 03 test; and, separately, how the
historical combined AttackNetV2.2 (trained, per the historical code, on 3DMAD, CSMAD, Replay-Attack, MSSpoof and Our
Dataset, not on NUAA)
behaves on the same 3,000 NUAA test frames under two image preparations.

## 2. What was fixed before the runs

- Part 1 target: official training list minus `cnn_train` minus `cnn_es` = 491 frames (243 bona fide, 248 attacks;
  9 subjects: 7 fitting, 2 validation; the 81 frames of the validation subjects are also the 81 in the M1 dev set; none
  in the test or in C1). Manifest with keys and
  SHA-256 in the restricted area; its SHA-256 in the configuration.
- Preprocessing of the training cache, frozen: `face.py` at commit `0ac7143` (the earliest committed version of our
  detector module). Verified by reproducing bit for bit every one of the 6,000 cache frames from the originals
  (`cache_check.json`, `test_logs/cache_check_log.txt`); the current detector of our application (box one pixel narrower and
  shorter since our change of 24 September) is never used in part 1.
- Parity with N1 on 24 keys chosen by identifier (rule in the configuration: first bona fide and first attack key per test subject; list restricted, its SHA-256 in the configuration): frames equal
  to the cache rows, recomputed scores within 1e-6 of the stored ones (largest difference 0).
- Rule: threshold 0.5 on the attack probability, attack if score > 0.5, tie = bona fide; every N1 seed reported, no seed
  selection, no score inversion; a class absent gives NA.
- Part 2: checkpoint loaded as saved (SHA-256 `3dd0c853…`), historical chain (BGR→RGB, Lanczos 256, enhancement,
  (x/255 − 0.5)·2), no quality filter, two variants (whole image; face crop with the current detector, margin 0.15,
  square, whole frame when no face), threshold 0.5, same 3,000 frames as N1.

## 3. Results

### Part 1 — sessions 01–02 (491 frames) against session 03

| set | seed | APCER | BPCER | ACER | AUROC |
|---|---|---|---|---|---|
| sessions 01-02: the 491 target frames | 42 | 0.0000 (0/248) | 0.0000 (0/243) | 0.0000 | 1.0000 |
| sessions 01-02: the 491 target frames | 43 | 0.0000 (0/248) | 0.0000 (0/243) | 0.0000 | 1.0000 |
| sessions 01-02: the 491 target frames | 44 | 0.0000 (0/248) | 0.0000 (0/243) | 0.0000 | 1.0000 |
| session 03: test frames of the 9 target subjects (N1 scores) | 42 | 0.1667 (154/924) | 0.6873 (422/614) | 0.4270 | 0.3230 |
| session 03: test frames of the 9 target subjects (N1 scores) | 43 | 0.3885 (359/924) | 0.6873 (422/614) | 0.5379 | 0.3171 |
| session 03: test frames of the 9 target subjects (N1 scores) | 44 | 0.0682 (63/924) | 0.6873 (422/614) | 0.3777 | 0.4569 |
| session 03: the 3,000 nuaa_cnn_test frames (N1 scores, read as stored) | 42 | 0.1067 (160/1500) | 0.7700 (1155/1500) | 0.4383 | 0.3485 |
| session 03: the 3,000 nuaa_cnn_test frames (N1 scores, read as stored) | 43 | 0.2607 (391/1500) | 0.7680 (1152/1500) | 0.5143 | 0.3754 |
| session 03: the 3,000 nuaa_cnn_test frames (N1 scores, read as stored) | 44 | 0.0480 (72/1500) | 0.7540 (1131/1500) | 0.4010 | 0.4582 |

Every seed separates the 491 frames completely (no error, AUROC 1.0; the lowest attack score above the highest bona
fide score), for the fitting and the validation subjects alike (the 81 frames of the validation subjects are also the 81
already in the M1 dev set). On session 03 the same checkpoints reject two bona fide frames in three (BPCER 0.687 for every
seed, on the 614 session 03 bona fide frames of the 3 target subjects that have bona fide frames there) and rank bona fide
above attacks (AUROC 0.317–0.457 on the 1,538 frames of the 9 target subjects); every comparable stratum has a lower or
equal error rate in sessions 01–02. With models, preprocessing and threshold unchanged, the error appears when the
capture session changes: this is consistent with the session-shift explanation of N1; it does not test the
weak-validation explanation (with a near-zero validation loss, a perfect separation of training-session frames is expected
under both). It is a diagnostic reading, not a causal test (the target frames share sessions, persons and acquisition
conditions with the training frames).

### Part 2 — the combined checkpoint on the 3,000 N1 test frames

| model / variant | APCER | BPCER | ACER | AUROC |
|---|---|---|---|---|
| combined AttackNetV2.2, whole image | 0.1480 (222/1500) | 0.9947 (1492/1500) | 0.5713 | 0.2169 |
| combined AttackNetV2.2, face crop | 0.2353 (353/1500) | 0.8160 (1224/1500) | 0.5257 | 0.3706 |
| N1 LivenessNet, seed 42 | 0.1067 (160/1500) | 0.7700 (1155/1500) | 0.4383 | 0.3485 |
| N1 LivenessNet, seed 43 | 0.2607 (391/1500) | 0.7680 (1152/1500) | 0.5143 | 0.3754 |
| N1 LivenessNet, seed 44 | 0.0480 (72/1500) | 0.7540 (1131/1500) | 0.4010 | 0.4582 |

(test 3,000 frames; the C1 300 subset is within about 0.02 for every row, largest difference 0.021: `tables/combined/aggregate.md`)

With the whole image the checkpoint calls almost every frame an attack (BPCER 0.995) and ranks bona fide above attacks
(AUROC 0.217); with the face crop the collapse is smaller (BPCER 0.816, ACER 0.526, AUROC 0.371) but the ranking stays
below chance. Neither variant improves on N1 on ACER or AUROC on the same frames (ACER 0.401–0.514, AUROC 0.348–0.458); the
whole-image APCER (0.148) and the face-crop APCER (0.235) fall inside the N1 range (0.048–0.261). The image
preparation alone moves BPCER by 0.18 and AUROC by 0.15.

## 4. Recommendation (to be verified on separate data, not a result of this test)

If one variant is carried forward to a verification on separate data, the face crop: it is the less degenerate of the
two and the closer to the article's description of the training images (detected, aligned and cropped faces). At
threshold 0.5 neither variant is usable on NUAA; before any operating-point comparison a threshold chosen on a source
validation, as in the article's protocol, would be needed, and the preparation of the historical training files should
be documented, because the two variants bracket it without reproducing it. Nothing here concerns the checkpoint on its
own training datasets.

## 5. Limits

- Part 1 is an exploratory check, not an independent validation: the 491 frames share sessions and persons with the
  fitting and validation frames (correlated with the training data); 81 were already scored in M1. A difference between
  sessions points to what to examine next; it does not prove a cause.
- Session 03 holds bona fide frames for only 3 of the 9 subjects: 6 bona fide strata are not comparable.
- The frozen module is dated four days after the cache: its equivalence is empirical (6,000 frames reproduced bit for
  bit on this machine with OpenCV 4.11.0).
- Part 2 is a sensitivity analysis to the image preparation: the historical preparation of the training files is not
  documented, neither variant is a certain replica; the comparison with N1 changes architecture, recipe and training
  data at once; the threshold is not the historical one (source-validation threshold not available).
- Three seeds give a range, not a confidence interval; image-level counts with correlated frames.
