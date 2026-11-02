# C1 crop check — note

Study `20261102-C1CROP-f6747cc` (commit `f6747cc`, `2026-11-02`). Details, SHA-256 and commands in `README.md` and `run-report.md`.

## 1. Question

C1 (28 September 2026) scored the NUAA CNNs with the application detector of 24 September; the CNNs had been trained
on 12 September on a cache built with the earlier detector. For every face found by RetinaFace the box of the C1
evaluation is one pixel narrower and one pixel shorter than the box of the training cache (our change `44971f3`:
`int(min(W, r) - x + 1)` became `int(min(W, r)) - x`, and the same for the height; the margin 0.15 and the square crop
are applied to that box). The coordinator asked (2026-11-01) to quantify the effect on one checkpoint, AttackNet V2.1
official NUAA, and the 300 C1 images: how much do the scores and the decisions change, everything else unchanged?

## 2. What was fixed before the run

- Checkpoint `AttackNetV2_1__nuaa.h5` (SHA-256 `67d9138d…`, the C1 checkpoint), loaded as the application does;
  the 300 C1 images (`nuaa_c1_v2.csv`, 150 + 150, 16 subjects, session 03), every file verified by SHA-256.
- Two frozen extractors: `face_0ac7143.py` (the training cache; `c72f7fa8…`) and `face_a80f6f9.py` (the C1 evaluation,
  identical to the current application module; `ca0d41cb…`). Same chain for both (`train_cnn.py` `preprocess`: crop
  with margin 0.15, square, whole frame without a face, 256×256 INTER_AREA, BGR→RGB, reference enhancement, /255),
  same CPU scoring (batch 16), same threshold 0.5 (attack if score > 0.5, tie = bona fide). No training, no
  calibration, no score inversion.
- Two reconstructions before the comparison, each able to stop the study: the 300 frames of the cache extractor
  equal bit for bit to their rows of the training cache; the 300 scores of the C1 extractor within 1e-5 of the stored
  C1 scores with the same decisions (the tolerance of the coordinator's R2 package; the C1 scores came from the GPU
  one image at a time, this study scores on the CPU in batches of 16: a first trial with 1e-6 was refused by 8 images
  between 1e-6 and 2.9e-6, all decisions equal, while in the C1 conditions the reproduction stays within 9e-8; the
  parity step records both informative reproductions, in the C1 conditions and on the CPU one image at a time).

## 3. Results

Both reconstructions passed: cache chain 300/300; C1 chain 300/300 within 1e-5 (maximum 2.9e-6; 292 within 1e-6),
300/300 same decisions; informative: in the C1 conditions maximum 8.9e-8; on the CPU one image at a time maximum
7.0e-5 (297/300 within 1e-5, every decision unchanged). RetinaFace found the face in all 300 images with both
extractors (no Haar, no whole-frame fallback); the box differs by dw −1, dh −1 for all 300; no preprocessed frame is
equal between the extractors.

| extractor | rejected bona fide / 150 | accepted attacks / 150 | APCER | BPCER | ACER | AUROC |
|---|---|---|---|---|---|---|
| `cache_0ac7143` (training cache) | 121 | 4 | 0.0267 | 0.8067 | 0.4167 | 0.7092 |
| `c1_a80f6f9` (C1 evaluation) | 115 | 3 | 0.0200 | 0.7667 | 0.3933 | 0.7103 |
| C1 delivered (stored scores) | 115 | 3 | 0.0200 | 0.7667 | 0.3933 | 0.7103 |

Paired differences (cache minus C1): maximum absolute 0.3214, median 8.6e-6, mean 0.0142 (signed mean +0.0043). The
score is exactly the same for 97 images, within 1e-4 for 181, within 1e-2 for 254; 18 images move by more than 0.1.
Decisions changed: 9 of 300, from the C1 extractor to the cache extractor: 7 bona fide from accepted to rejected,
1 bona fide from rejected to accepted, 1 attack from rejected to accepted; the nine C1 scores lie between 0.36 and 0.62.

## 4. Reading

- The training-consistent extractor makes the checkpoint slightly worse on the 300 C1 images at the fixed threshold
  (ACER +0.0233, from 6 more bona fide rejected and 1 more attack accepted), with the AUROC unchanged within 0.002.
  The one-pixel box of the C1 evaluation is therefore not the cause of the C1 result: the BPCER stays above 0.76
  either way, and the C1 row is reproduced by the C1 extractor to four decimals (AUROC 0.71029 against 0.71027).
- The effect is concentrated on a minority of images: six in ten keep the same score within 1e-4 (181 of 300), 18 move
  by more than 0.1, 9 cross the threshold; on average the cache extractor gives slightly higher attack scores (+0.004).
- Repeated score values across different images occur (1.0 for nine images with either extractor; a few other values,
  almost all above 0.9999, shared by 2 to 5 images): observed, not investigated here.

## 5. Limits

One checkpoint and one subset of 300 images of one session (adjacent frames are not independent samples); no
uncertainty estimate; no Haar or no-face case among the 300 (the comparison concerns the one-pixel RetinaFace box
only); C1 on the GPU one image at a time against the CPU in batches here, bounded by the parity check (2.9e-6, no
decision changed; a CPU replication one image at a time can differ by up to 7e-5 on 3 images, decisions unchanged). The other three C1 CNNs and the M1 dev scores (same detector of 24 September) are not recomputed
here; they can be, with the same script, in a separate run if the coordinator finds it useful. C1, M1, M2, N1, N2, the C2 runner and the R2 reference are unchanged; this
is not the iOS comparison.
