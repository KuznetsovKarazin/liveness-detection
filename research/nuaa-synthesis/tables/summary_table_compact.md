# Unified table, compact form (codes in the legend below; full text, row ids and sources in `summary_table.md`)

| block | model | training | test set | extractor | prep. | threshold | rejected bf / accepted att. | APCER | BPCER | ACER | AUROC | run | transfer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | LivenessNet (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | 0.5 (fixed) | 109 / 24 | 0.1600 | 0.7267 | 0.4433 | 0.3782 | 20260928-C1-seed42-a80f6f9 | session |
| C1 | AttackNet V1 (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | 0.5 (fixed) | 40 / 42 | 0.2800 | 0.2667 | 0.2733 | 0.7547 | 20260928-C1-seed42-a80f6f9 | session |
| C1 | AttackNet V2.1 (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | 0.5 (fixed) | 115 / 3 | 0.0200 | 0.7667 | 0.3933 | 0.7103 | 20260928-C1-seed42-a80f6f9 | session |
| C1 | AttackNet V2.2 (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | 0.5 (fixed) | 118 / 22 | 0.1467 | 0.7867 | 0.4667 | 0.4985 | 20260928-C1-seed42-a80f6f9 | session |
| M1 | LivenessNet (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | 0.181055 (dev EER) | 118 / 5 | 0.0333 | 0.7867 | 0.4100 | 0.3782 | 20261003-M1-seed42-e67ecc7 | session |
| M1 | AttackNet V1 (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | 0.329237 (dev EER) | 44 / 37 | 0.2467 | 0.2933 | 0.2700 | 0.7547 | 20261003-M1-seed42-e67ecc7 | session |
| M1 | AttackNet V2.1 (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | 0.870329 (dev EER) | 91 / 11 | 0.0733 | 0.6067 | 0.3400 | 0.7103 | 20261003-M1-seed42-e67ecc7 | session |
| M1 | AttackNet V2.2 (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | 0.696933 (dev EER) | 112 / 38 | 0.2533 | 0.7467 | 0.5000 | 0.4985 | 20261003-M1-seed42-e67ecc7 | session |
| N1 | LivenessNet, new CPU training, seed 42 | T-NUAA | S-3000 | E-CACHE | P-REF | 0.5 (fixed) | 1155 / 160 | 0.1067 | 0.7700 | 0.4383 | 0.3485 | 20261024-N1-LivenessNet-fb93496 | session |
| N1 | LivenessNet, new CPU training, seed 43 | T-NUAA | S-3000 | E-CACHE | P-REF | 0.5 (fixed) | 1152 / 391 | 0.2607 | 0.7680 | 0.5143 | 0.3754 | 20261024-N1-LivenessNet-fb93496 | session |
| N1 | LivenessNet, new CPU training, seed 44 | T-NUAA | S-3000 | E-CACHE | P-REF | 0.5 (fixed) | 1131 / 72 | 0.0480 | 0.7540 | 0.4010 | 0.4582 | 20261024-N1-LivenessNet-fb93496 | session |
| N2-S | LivenessNet, N1 ckpt seed 42 | T-NUAA | S-491 | E-CACHE | P-REF | 0.5 (fixed) | 0 / 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 20261029-N2-sessions-2ac3e29 | same |
| N2-S | LivenessNet, N1 ckpt seed 42 | T-NUAA | S-03/9 | E-CACHE | P-REF | 0.5 (fixed) | 422 / 154 | 0.1667 | 0.6873 | 0.4270 | 0.3230 | 20261029-N2-sessions-2ac3e29 | session/9 |
| N2-S | LivenessNet, N1 ckpt seed 43 | T-NUAA | S-491 | E-CACHE | P-REF | 0.5 (fixed) | 0 / 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 20261029-N2-sessions-2ac3e29 | same |
| N2-S | LivenessNet, N1 ckpt seed 43 | T-NUAA | S-03/9 | E-CACHE | P-REF | 0.5 (fixed) | 422 / 359 | 0.3885 | 0.6873 | 0.5379 | 0.3171 | 20261029-N2-sessions-2ac3e29 | session/9 |
| N2-S | LivenessNet, N1 ckpt seed 44 | T-NUAA | S-491 | E-CACHE | P-REF | 0.5 (fixed) | 0 / 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 20261029-N2-sessions-2ac3e29 | same |
| N2-S | LivenessNet, N1 ckpt seed 44 | T-NUAA | S-03/9 | E-CACHE | P-REF | 0.5 (fixed) | 422 / 63 | 0.0682 | 0.6873 | 0.3777 | 0.4569 | 20261029-N2-sessions-2ac3e29 | session/9 |
| N2-C | AttackNet V2.2 combined (coordinator's ckpt) | T-COMB | S-3000 | E-WHOLE | P-HIST | 0.5 (fixed) | 1492 / 222 | 0.1480 | 0.9947 | 0.5713 | 0.2169 | 20261029-N2-combined-247dfd7 | dataset |
| N2-C | AttackNet V2.2 combined (coordinator's ckpt) | T-COMB | S-3000 | E-CROP | P-HIST | 0.5 (fixed) | 1224 / 353 | 0.2353 | 0.8160 | 0.5257 | 0.3706 | 20261029-N2-combined-247dfd7 | dataset |
| CROP | AttackNet V2.1 (C1 ckpt 12/09), CPU batch 16 | T-NUAA | S-300 | E-CACHE | P-REF | 0.5 (fixed) | 121 / 4 | 0.0267 | 0.8067 | 0.4167 | 0.7092 | 20261102-C1CROP-f6747cc | session |
| CROP | AttackNet V2.1 (C1 ckpt 12/09), CPU batch 16 | T-NUAA | S-300 | E-C1 | P-REF | 0.5 (fixed) | 115 / 3 | 0.0200 | 0.7667 | 0.3933 | 0.7103 | 20261102-C1CROP-f6747cc | session |

Legend. Blocks: C1, M1, N1 (part 2), N2-S (N2 part 1, sessions), N2-C (N2 part 2, combined model), CROP (C1 crop check); ckpt = checkpoint; "rejected bf / accepted att." = rejected bona fide / accepted attacks, the numerators of BPCER and APCER. Unit = image (frame) in every row; attack if score_attack > threshold (tie = bona fide); label 0 bona fide, 1 attack; score = probability of attack; APCER = accepted attacks / attacks, BPCER = rejected bona fide / bona fide, ACER = their mean, AUROC threshold-free (`auc` in the C1 table, `test AUC` in the M1 table).

- `T-NUAA` (training): NUAA official train, sessions 01-02: 2,507 fitting frames (7 subjects); early stopping on 493 frames of 2 other subjects, sessions 01-02 (early stopping)
- `T-COMB` (training): 3DMAD, CSMAD, Replay-Attack, MSSpoof, Our Dataset (historical code; no NUAA); historical early stopping
- `S-300` (test): C1 300 (150 + 150), session 03, 16 subjects
- `S-3000` (test): nuaa_cnn_test 3,000 (1,500 + 1,500), session 03, 16 subjects
- `S-491` (test): 491 frames of the official training list outside fitting and early stopping (243 + 248), sessions 01-02, 9 subjects
- `S-03/9` (test): session 03 test frames of the same 9 training subjects (614 + 924), inside nuaa_cnn_test
- `E-C1` (extractor): c1_a80f6f9: the C1 evaluation extractor (RetinaFace box 1 px narrower and shorter than the cache)
- `E-CACHE` (extractor): cache_0ac7143: the training-cache extractor (cache frames, or the frozen copy on new frames)
- `E-WHOLE` (extractor): whole image (no face detection)
- `E-CROP` (extractor): face crop (current detector, margin 0.15, square; whole frame without a face)
- `P-REF` (preprocessing): detector crop (margin 0.15, square; whole frame without a face), 256x256 INTER_AREA, BGR to RGB, reference enhancement, /255 in [0, 1]
- `P-HIST` (preprocessing): historical chain of the combined model: BGR to RGB, 256x256 Lanczos, reference enhancement, (x/255 - 0.5) * 2 in [-1, 1]
- `same` (transfer): same sessions as training (01-02), fitting and early-stopping subjects, frames never used
- `session` (transfer): session transfer (01-02 to 03; the training subjects and the test-only subjects)
- `session/9` (transfer): session transfer (01-02 to 03; the training subjects only)
- `dataset` (transfer): dataset transfer (zero-shot: model never trained on NUAA)

Runs: C1 `20260928-C1-seed42-a80f6f9`, M1 `20261003-M1-seed42-e67ecc7`, N1 `20261024-N1-LivenessNet-fb93496`, N1_subjects `20261024-N1-subjects-a5fcbbc`, N2_sessions `20261029-N2-sessions-2ac3e29`, N2_combined `20261029-N2-combined-247dfd7`, C1_crop_check `20261102-C1CROP-f6747cc`.
