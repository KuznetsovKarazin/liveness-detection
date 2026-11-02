# C1 crop check: the cache extractor against the C1 extractor, AttackNet V2.1 official NUAA, the 300 C1 images (study `20261102-C1CROP-f6747cc`)

Checkpoint `AttackNetV2_1__nuaa.h5` (SHA-256 `67d9138d6c43…`), the one of C1, loaded as is. Same 300 images (manifest `nuaa_c1_v2.csv`), same chain (`train_cnn.py` preprocess: crop with margin 0.15, square, whole frame without a face; 256×256 INTER_AREA; reference enhancement; /255), both extractors scored on the CPU in this study (the stored C1 scores came from the application on the GPU, one image at a time: see the parity line below), threshold 0.5 fixed in C1, attack if score_attack > 0.5 (tie = bona fide); only the face extractor varies: `cache_0ac7143` built the training cache, `c1_a80f6f9` was used in the C1 evaluation (RetinaFace box 1 px narrower and 1 px shorter). No training, no calibration, no score inversion.

## Metrics per extractor (same 300 images)

| extractor | n bona fide | n attack | rejected bona fide | accepted attacks | APCER | BPCER | ACER | AUROC |
|---|---|---|---|---|---|---|---|---|
| cache_0ac7143 | 150 | 150 | 121 | 4 | 0.0267 | 0.8067 | 0.4167 | 0.7092 |
| c1_a80f6f9 | 150 | 150 | 115 | 3 | 0.0200 | 0.7667 | 0.3933 | 0.7103 |
| C1 delivered (stored scores, 20260928-C1-seed42-a80f6f9) | 150 | 150 | 115 | 3 | 0.0200 | 0.7667 | 0.3933 | 0.7103 |

Parity of the C1 chain (CPU, batch 16, the scoring of this study): 300/300 recomputed scores within 1e-05 of the stored C1 scores (max abs difference 2.891e-06; within 1e-6: 292), 300/300 same decisions. In the C1 conditions (GPU, one image at a time, informative): max abs difference 8.941e-08, within 1e-6 300/300, same decisions 300/300. On the CPU one image at a time (informative): max abs difference 6.998e-05, within 1e-5 297/300, same decisions 300/300. Cache chain: 300/300 frames of `cache_0ac7143` equal bit for bit to the cache rows.

## Paired differences (cache extractor minus C1 extractor, per image)

| statistic | value |
|---|---|
| images | 300 |
| frames identical between the extractors | 0 |
| max abs score difference | 0.321353 |
| median abs score difference | 0.000009 |
| mean abs score difference | 0.014237 |
| signed difference: min / mean / max | -0.316398 / 0.004335 / 0.321353 |
| decisions changed at 0.5 | 9 |
| attack (C1 extractor) to bona fide (cache extractor): all / bona fide / attack images | 2 / 1 / 1 |
| bona fide (C1 extractor) to attack (cache extractor): all / bona fide / attack images | 7 / 7 / 0 |

| abs score difference | images |
|---|---|
| zero | 97 |
| (0, 0.0001] | 84 |
| (0.0001, 0.001] | 36 |
| (0.001, 0.01] | 37 |
| (0.01, 0.1] | 28 |
| (0.1, 1] | 18 |

## Detector per image

| detector pair (cache / C1) | images |
|---|---|
| RetinaFace/RetinaFace | 300 |

| box delta (C1 minus cache: dx, dy, dw, dh) | images |
|---|---|
| dx 0, dy 0, dw -1, dh -1 | 300 |

Pixel differences between the two preprocessed frames: max abs 250, mean abs over the different frames 6.417 (levels on 0-255).

NA = not computable (class absent). Values are rounded to four decimals (metrics), six (score differences), three significant digits (parity maximum) or three decimals (pixel mean); aggregate.json holds the exact values.
