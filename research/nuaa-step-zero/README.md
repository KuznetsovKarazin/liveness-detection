# NUAA step zero: image statistics of the authorised NUAA materials against the delivered scores (no training, no inference)

The step zero proposed in section 5 of the NUAA synthesis (`research/nuaa-synthesis/SYNTHESIS.md`) and authorised by
the coordinator on the C2 card on 3 November 2026, on the NUAA materials already authorised. Per-image statistics of
the original NUAA images (raw view) and of the frames of the training cache (crop view), set against the per-image
scores already delivered with C1, N1 and N2. Every number is a statistic of an image or a descriptive comparison
between those statistics and accepted scores: **associations, not causes**. This folder holds the configuration
written before the run (`nuaa_step_zero.json`, at the default path of the script, `delivery/stepzero/` under the
package root `livenesslab/`), the test logs and this note; the code is `livenesslab/scripts/nuaa_step_zero.py` with
its tests and mutation check. Per-image rows and per-subject tables go to the restricted area.

## What it does not do

No training, no inference of any model, no face detection, no new threshold, no selection or change of any accepted
result. The statistics are global image properties (luminance, contrast, saturation, sharpness, size, file size) and,
for the 300 C1 images only, the relative area of the RetinaFace boxes already recorded by the crop check; they do not
identify what a network uses. The only face-level view is the crop view; the raw view is the whole frame (background
included; NUAA attacks are re-photographed prints with a border). The crop view describes the network input after
the reference enhancement chain (bilateral filter, CLAHE, unsharp mask, gamma and a final linear contrast/brightness
step), not the face region of the original.
Face boxes exist only for the 300 C1 images (two extractors); no boxes for the 3,000 without a new detection pass,
which is outside this step. The measures are descriptive, without statistical tests, on frames that are correlated
within a subject and session; the only uncertainty is a bootstrap over subjects of the class AUROC.

## Delivery status

- Run `20261103-STEPZERO-78ca9e4` from the clean commit `78ca9e4` of the development repository (2026-11-03): `public/`
  copied here as `tables/` (`aggregate.json`, `aggregate.md`, `figure_distributions.png`; the public `SHA256SUMS` of the
  run has SHA-256 `dc92b8c18708ed113e29b6624ac1769281973553441a271e970ed9e0f4a861fb`, the anchor kept in the restricted
  folder). The configuration (`nuaa_step_zero.json`, SHA-256 in `SHA256SUMS`) was written and every input verified with
  `check-inputs` before the run (log in `test_logs/check_nuaa_step_zero.txt`).
- Restricted folder (`per_image_raw.csv`, `per_image_crop.csv`, `per_subject.csv`, `per_subject_auroc.csv`,
  `public_SHA256SUMS.sha256`, `SHA256SUMS`; no images, no cache, no weights, no scores copied): uploaded on 2026-11-03
  to the shared N1 folder (provisional destination, as for N2) as `20261103-STEPZERO-78ca9e4/` with `public/`,
  `restricted/` and `status.json`.
- Images, cache and scores stay local.

## Inputs (read-only, fixed by SHA-256 in the configuration)

| input | content | verification |
|---|---|---|
| M2 manifests (`results/m2/manifests/`, accepted package `20261006-M2-seed42-36c677a`) | `nuaa_official_train.csv` 3,491 (sessions 01–02, 9 subjects), `nuaa_official_test.csv` 9,123 (session 03, 16 subjects), `nuaa_cnn_train.csv` 2,507, `nuaa_cnn_early_stopping.csv` 493, `nuaa_cnn_test.csv` 3,000, `nuaa_c1_v2.csv` 300 | SHA-256 of each file; rows, class counts and subjects against the configuration; subject, session and split read from the file name and compared with the columns, label compared with the key folder; official lists disjoint by file SHA-256 (the declared sessions of the two lists must not overlap, `official_split` must equal the list, so the key check between the lists is unconditionally redundant and declared so in the mutation check); subsets inside their list with the same SHA-256 and bytes; `cnn_train + cnn_es` = spread of the training list and `cnn_test` = spread of the test list (the M2 rule), `c1` inside `cnn_test`; `target_491` = official train minus `cnn_train` minus `cnn_es` |
| original images (`data/NUAA/images/<class>_<split>/<name>`) | the 12,614 files of the two official lists | every file located by key and accepted only with size and SHA-256 equal to the manifest row (two distinct messages); symbolic links refused |
| training cache `nuaa_1500.npz` (built 2026-09-12, extractor `face.py` 0ac7143, reproduced bit for bit in N2) | `X_train` 3,000 = `cnn_train + cnn_es` and `X_test` 3,000 = `cnn_test`, uint8 256×256×3 RGB, in the spread order; `y_*`, `s_*` | SHA-256 of the file; member types, shapes and C order from the `.npy` headers (Fortran order and Python objects refused, no pickle); frame count against the expected count and the sampling orders; labels and subjects of every row equal to the manifest row of its key; members decompressed one at a time (compressed npz: no memory mapping) |
| scores N1 (`20261024-N1-LivenessNet-fb93496/seed{42,43,44}/scores_test.csv`, 3,000 each) | LivenessNet retrained on the CPU, session 03 | SHA-256 of the file = configuration = line of the folder's `SHA256SUMS`, whose SHA-256 is fixed too; rows filtered by `train_seed`; every key in `cnn_test` and as many rows as keys; subject and label = manifest; finite score; stored prediction = score > 0.5 (a tie is bona fide) |
| scores N2 part 1 (`20261029-N2-sessions-2ac3e29/seed{42,43,44}/scores_sessions.csv`, 491 each) | the N1 checkpoints on the 491 target frames, sessions 01–02 | same checks; keys of `target_491` |
| scores N2 part 2 (`20261029-N2-combined-247dfd7/{whole_image,face_crop}/scores_test.csv`, 3,000 each) | the historical combined AttackNet V2.2, two preparations | same checks; rows filtered by `variant`; keys of `cnn_test` |
| scores C1 (`20260928-C1-seed42-a80f6f9/scores.csv`, 300 per analyzer) | the four NUAA CNN checkpoints of 2026-09-12 | same checks; rows filtered by `analyzer_id`; keys of `c1`; the session is read from the file name, never from the `session_id` column of this file (the field error corrected by the M2 manifest v2) |
| face boxes (`20261102-C1CROP-f6747cc/frames.csv`, 300 × 2 extractors) | RetinaFace boxes of the C1 images with the cache extractor and the C1 extractor | same checks; one row per image and extractor; positive width and height |

## Statistics and measures

Per image, raw view (`cv2.imread`, BGR) and crop view (cache frame, RGB): `lum_mean` and `lum_std` (mean and population
standard deviation of the OpenCV grey image, 0.299 R + 0.587 G + 0.114 B); `sat_mean` (mean of the S channel of the
OpenCV HSV conversion, 0–255); `lap_var` (population variance of the 3×3 Laplacian of the grey image, `cv2.Laplacian`,
`CV_64F`, `ksize` 1). Raw view only: `width`, `height`, `bytes` (file size from the manifest).

Measures, all descriptive:
- per set (official train by session 01/02, official test = session 03, `cnn_train`, `cnn_es`, `cnn_test`, `c1`,
  `target_491`; crop view for the sets inside the cache) × session × class: n, median, first and third quartile;
- per set × session: AUROC of each statistic as a class separator (label 1 = attack, higher statistic → attack;
  Mann–Whitney U with ties counted 0.5): a value below 0.5 means that the bona fide frames have the higher statistic;
  with a bootstrap over subjects (1,000 resamples with replacement of the subjects of the session, seed 0, the same
  resamples for every statistic, percentiles 2.5 and 97.5; a resample without one of the classes does not count and
  the number of valid resamples is reported);
- per set × session: the **within-subject** class AUROC (computed inside each subject × session with both classes),
  summarised as median, minimum and maximum over those subjects with their number; the per-subject values are in the
  restricted `per_subject_auroc.csv`; in the `all` row of a set with two sessions the sessions of a subject are pooled,
  for the within-subject AUROC and for the bootstrap alike;
- per score file × session × class (bona fide, attack, all): Spearman rank correlation (average ranks for ties) between
  each statistic and `score_attack`, with n; NA with fewer than 3 pairs or a constant variable; the crop view where
  every frame of the set is in the cache (not for the 491);
- bona fide frames of session 03, N1 (each seed) and C1 (each CNN): AUROC of each statistic between the frames rejected
  (prediction 1 at 0.5) and accepted (rejected = positive);
- annex: relative area of the RetinaFace box (`bbox_w · bbox_h / (width · height)`) per extractor, quartiles per class
  and AUROC as a class separator.

Reading caveat: the aggregate class AUROC and the within-class Spearman confound subject and class (in session 03 the
bona fide frames come from 9 subjects, the attacks from 15, 8 subjects have both classes; the run computes these counts
from the manifest and writes them in `counts.subjects_session_03` and in the limits); the within-subject AUROC is the
reading that holds the subject fixed. The measure texts and the view descriptions of the aggregate are read from the
configuration (`measures`, `views`), whose fields are a closed set validated by the script, and the bootstrap recipe of
the configuration (`bootstrap`) must equal the one of the script. AUROC and Spearman are implemented in NumPy; the tests
compare them with scikit-learn and scipy when those packages are present (hand-computed cases otherwise).

## Commands

From the development repository, with the data (the only place where `check-inputs` and `run` can be executed):

```bash
python scripts/nuaa_step_zero.py check-inputs
python scripts/nuaa_step_zero.py run --i-am-authorized
```

`check-inputs` verifies the configuration and every input (manifests, originals, cache, scores, boxes) and writes
nothing. `run` writes `results/stepzero/<run-id>/` (`YYYYMMDD-STEPZERO-<commit>`, `-dev` if the tree is modified):
`status.json` (RUNNING, then PASS or FAIL), `public/` (`aggregate.json`, `aggregate.md`, `figure_distributions.png`
when matplotlib is available, `SHA256SUMS`) and `restricted/` (`per_image_raw.csv`, `per_image_crop.csv`,
`per_subject.csv`, `per_subject_auroc.csv`, `public_SHA256SUMS.sha256`, `SHA256SUMS`). The public texts are scanned
before writing (image keys, image SHA-256, subject identifiers, absolute paths, credentials, non-local addresses,
unfinished-work markers and, in the development repository, local operator, host and tool patterns) and the run fails
on any hit. The run folder is validated for identity before any change (never a protected folder, never inside an
input package, no foreign entry); previous outputs are removed right after the admission, and on any error only
`status.json = FAIL` remains. Inputs are never written.

Tests and mutation check, from the package root `livenesslab/` (synthetic world in temporary folders; no data needed;
the one read-only check of the real inputs is skipped when they are absent) or from the development repository root:

```bash
python scripts/check_nuaa_step_zero.py
python scripts/check_nuaa_step_zero_mutations.py
```

Expected lines. With the real inputs present: `tests run: 57, failures: 0, errors: 0, skipped: 0` and `RESULT: PASS`.
Without data (package layout): `skipped test_real_check_inputs_read_only: missing input: the NUAA images, the M2
manifests, the training cache or the accepted run folders are not present`, `tests run: 57, failures: 0, errors: 0,
skipped: 1` and `RESULT: PASS`. Mutation check, in both layouts: `mutations killed: 60/60`, `control mutations survived:
2/2 (1 control, 1 declared redundant)` and `RESULT: PASS` (the mutation check copies the configuration from
`delivery/stepzero/` next to the scripts, or next to the package folder). Logs in `test_logs/`:
`check_nuaa_step_zero.txt` and `check_nuaa_step_zero_mutations.txt` (macOS, development repository, real inputs present;
written to a temporary folder and copied here afterwards), `linux_with_data.txt` (`python:3.12` container, development
layout with the real inputs mounted read-only: suite and `check-inputs`), `linux_package_no_data.txt` (same container,
package layout without data: suite with the declared skip and mutation check). The container has NumPy and OpenCV
installed, without matplotlib, scikit-learn and scipy.

Dependencies: Python 3.11 or 3.12, NumPy, OpenCV (`cv2`); matplotlib optional (the figure is skipped with a declared
note when it is missing); `scripts/m2_writers.py`, `scripts/path_guard.py`, `scripts/package_rules.py`. No TensorFlow.

## Limits

Frames of one subject and session are correlated: n counts frames, not independent units. The measures are descriptive,
without statistical tests; the bootstrap over subjects of the class AUROC is the only uncertainty, and three seeds and
four checkpoints give ranges, not distributions. The aggregate class AUROC and the within-class Spearman confound
subject and class; the within-subject AUROC holds the subject fixed but rests on the subjects with both classes in the
session (8 in session 03: bona fide from 9 subjects, attacks from 15, intersection 8; the run writes the number as
`within_subject_auroc.n_subjects`). The statistics are global image properties; they do not identify what a network
uses. The only face-level view is the crop view, which describes the network input after the reference enhancement
chain, not the face region of the original (bilateral filter, CLAHE, unsharp mask, gamma and a final linear
contrast/brightness step); the raw view is the whole frame, background and print border included. The crop view covers
the 6,000 cache frames only; the 491 target frames have the raw view only. Face boxes exist only for the 300 C1 images
(two extractors); no boxes for the 3,000 without a new detection pass, which is outside this step. Associations, not
causes.
