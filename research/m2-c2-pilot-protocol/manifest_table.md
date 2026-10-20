# C2 pilot (CASIA-FASD → NUAA): manifests and splits

Aggregate counts only. The per-image manifests are restricted (M2 run folders, `02_Experiments/M2/`); they are
named here by file and SHA-256. No per-subject count and no subject or video key is given.

## Status labels

- **[VER-LOCAL]** verified locally on 19 October 2026: SHA-256 recomputed with `shasum -a 256` on the restricted
  copy; per-class, subject and video counts recounted from the CSV columns `label`, `subject_key`, `video_key`,
  `subject`.
- **[M2-PASS]** declared in the versioned M2 rules and checked by the M2 overlap rules in the listed run (CASIA-FASD
  code and aggregates merged in `research/pad-2026` with PR #5, merge `eb218a8`; the merge does not close the
  independent check below nor approve the splits).
- **[ACCEPTED]** part of the M2 NUAA tranche accepted by the coordinator on 8 October 2026 (PR #4).
- **[PENDING]** independent check of the restricted CSV not yet done (the 21 restricted CSV of the second M2 tranche
  go through the channel handled by the coordinator).
- **[DECLARED]** recorded by a training run or a card, not reproducible from the files.
- **[C1]** C1 table accepted by the coordinator (`research/c1-livenesslab/tables/nuaa_table.md`).
- **[DOC-EXT]** documented by an external source, not verifiable on our copy.

## Manifests

CASIA-FASD manifests: restricted run `20261019-M2-rgb-seed42-cbcda11`, status [VER-LOCAL] [M2-PASS] [PENDING].
NUAA manifests: restricted run `20261006-M2-seed42-36c677a`, status [VER-LOCAL] [ACCEPTED].

| id | role in C2 | M2 manifest | images (bona fide + attack) | subjects (bf / attack) | videos (bf + attack) |
|---|---|---|---|---|---|
| S1 | source train (fitting) | CASIA `cnn_train` | 1,339 (333 + 1,006) | 16 / 16 | 192 (48 + 144) |
| S2 | source dev (early stopping, threshold) | CASIA `cnn_es` | 316 (71 + 245) | 4 / 4 | 48 (12 + 36) |
| S3 | source test | CASIA `cnn_test` | 2,091 (591 + 1,500) | 30 / 30 | 360 (90 + 270) |
| S4 | (reference) source test, all frames | CASIA `mirror_test` | 2,408 (591 + 1,817) | 30 / 30 | 360 (90 + 270) |
| T1 | target test (D-3, agreed) | NUAA `cnn_test` | 3,000 (1,500 + 1,500) | 9 / 15 | — |
| T2 | (alternative) target, all images | NUAA `official_test` | 9,123 (3,362 + 5,761) | 9 / 15 | — |
| T3 | (not proposed) C1 evaluation set | NUAA `c1` | 300 (150 + 150) | 9 / 15 | — |

| id | file | SHA-256 |
|---|---|---|
| S1 | `casia_cnn_train.csv` | `a1d5ec4c1fea5c2fa854fe780fff53d9e61fa6e2a29f74ca95bee1f58c8a195d` |
| S2 | `casia_cnn_early_stopping.csv` | `3d6b601dd8a3d38f7dcea37f6b1cdbcde555d88b6b3164cbbc269f6cfb9c8d39` |
| S3 | `casia_cnn_test.csv` | `20b6915145fb2dfa490b80e5ff8128ed05a86eef3298ce18bd4501b291ec304a` |
| S4 | `casia_mirror_test.csv` | `c84357167228b6850fd195e02625b2f859b1c4698ac3f1585b2dd87bb6c62fc8` |
| T1 | `nuaa_cnn_test.csv` | `ce9669dccae37041e778ff7bcba5f4cef08fa79f50d65be84806b084c5d95748` |
| T2 | `nuaa_official_test.csv` | `2fad23ff295cb2c1d3d0be238deef02faad252aa717933c30588bd5a9c4106c7` |
| T3 | `nuaa_c1_v2.csv` | `1c2cf97cb7115e7ae98640103670c628a3910a37c25b89f76369cc7ecad87426` |

Origin: S1 and S2 are all frames of `train_img` of the mirror (16 and 4 subjects); S3 is `spread(test_img, 1500)` per
class (317 attack frames not sampled); S4 is all of `test_img`. T1 is `spread(official test, 1500)` per class of the
NUAA official test split (session 03), T2 the whole split (mirror copy of the official lists), T3
`spread(official test, 150)`, the frozen C1 baseline.

## Derived frames

Preprocessed `uint8` arrays, linked to the files by sampling order and the label and group arrays, not by file name.

- `casia_fasd_1500.npz`: `X_train` 1,655 (S1 + S2), `X_test` 2,091 (S3); [VER-LOCAL] [M2-PASS]; SHA-256
  `7d0a2b22146375bc4e629a430f0f5a161655d56ca26bb62a5e160cbff58bf5a6`.
- `nuaa_1500.npz`: `X_test` 3,000 (T1); [VER-LOCAL] [ACCEPTED]; SHA-256
  `dc286dc371e61d638da1842c062a38c5368a9ec4da07ccbc36b7fff6f3177920`.

## Relations

- S1 and S2 are disjoint by key, hash, subject and video (rule C08) and together are all 1,655 frames of `train_img`
  (C07); S1/S2 and S3 are disjoint by key and hash (C09). Whole subjects, hence whole videos, are on one side.
  [M2-PASS]
- `train_img` and `test_img` hold different persons by protocol (20 and 30 subjects) [DOC-EXT]; not verifiable on the
  copy, whose subject numbers are local to each split.
- No CASIA-FASD image file (4,063 SHA-256) equals a NUAA image file (12,614), and no frame of the CASIA cache (3,746)
  equals a frame of the NUAA test cache (3,000) [VER-LOCAL] (read-only comparison; the run repeats it in `check`).
- T3 ⊂ T1 ⊂ T2 (NUAA rules R09, R10, R13) [ACCEPTED].
