# C1 crop check: the training-cache face extractor against the C1 evaluation extractor on AttackNet V2.1 and the 300 C1 images

Task described by the coordinator on the operator's Trello card on 2026-11-01 (discussion and date on the C2 card). A
paired comparison of two face extractors on the same checkpoint (AttackNet V2.1, official NUAA protocol, the C1
checkpoint) and the same 300 C1 images: `cache_0ac7143` is `src/tesi_app/face.py` at commit `0ac7143` (16 September 2026, the first commit of the
repository), the extractor that built the training cache `nuaa_1500.npz` of 2026-09-12 (reproduced bit for bit by N2); `c1_a80f6f9` is `face.py` at commit `a80f6f9`, the
extractor of the C1 evaluation of 2026-09-28 (the application module since our change `44971f3` of 2026-09-24, unchanged
since). The only numerical difference between the two files is the RetinaFace box, one pixel narrower and one pixel
shorter with the C1 extractor. Weights, images, chain, normalisation, CPU scoring and threshold are the same for both;
no training, no calibration, no score inversion. This folder holds the code, the configuration saved before the run,
the two frozen extractor copies, the run report, the note, the publishable tables and figure and the test logs.
Per-image scores, boxes, frames and logs go to the restricted area (see "Delivery status").

## Delivery status

- Study `20261102-C1CROP-f6747cc` on `2026-11-02`, from the clean commit `f6747cc` of the development repository (a non-trial run is refused
  unless the work tree is clean; `check.json` records the commit and the SHA-256 of every code file and of the two
  frozen copies, equal to the files of this folder). The tables and figure below come from it. A trial execution from a
  modified tree (`-trial` study in the session's scratch area, not part of the delivery) gave the same scores for every
  image.
- Restricted folder `20261102-C1CROP-f6747cc/` (per-image rows: `frames.csv`, `parity.csv`, `scores_paired.csv`, `per_subject.csv`; the
  step records; the logs; the figure as PNG; no images, no cache, no weights), ready for upload, as for N2, into the
  shared N1 folder of the restricted area indicated by the coordinator (status updated on the C2 card).
- Images, cache and weights stay local (within the already authorised scope).

## What is delivered

- `c1crop.json`: the configuration, written before the run; its SHA-256
  (`caa1156ac7608089790c12b337040cfc4f2443611be7735599ebe32b75d8e4ae`) is bound in `c1crop_check.py`: a non-trial run accepts no other configuration. It fixes the
  checkpoint (file, size, SHA-256), the two extractors (file, commit, SHA-256), the detector resources and the reference
  code (SHA-256), the chain, the cache (file, SHA-256, members), the stored C1 scores (file, SHA-256, analyzer, columns),
  the parity rule and tolerance, the manifests (file, SHA-256, counts) and the classification of every output as public
  or restricted.
- `face_0ac7143.py`: the frozen copy of `src/tesi_app/face.py` at commit `0ac7143` (16 September 2026; SHA-256
  `c72f7fa8a9f3b099babce2255530466e5a34731fb5d021f97fd412a98cb58261`), the same copy delivered with N2, which
  reproduces the training cache bit for bit. `face_a80f6f9.py`: the frozen copy of `face.py` at commit `a80f6f9`
  (28 September 2026, the C1 commit; SHA-256 `ca0d41cb2a83dfe6369c970f222862c67f5f06ea2e9d8d0332ed670ae31af536`),
  byte-identical to the current application module (checked at every step). Both are loaded as `tesi_app.face_<name>`
  modules; the application's own `face.py` is never used for the frames.
- `livenesslab/scripts/c1crop_check.py`: subcommands `check`, `frames`, `parity`, `score`, `aggregate`.
- `livenesslab/scripts/check_c1crop.py` (tests on synthetic data, no TensorFlow, no detector, no images) and
  `check_c1crop_mutations.py` (declared mutations).
- `tables/`: `aggregate.json`, `aggregate.md` (metrics per extractor next to the delivered C1 row, paired
  differences, changed decisions, detector and box counts), `figure_paired.pdf` (one point per image, no identifiers;
  the study writes it as PNG, which the group repository ignores: the PDF holds the same bitmap), `frames.json` and
  `parity.json` (counts and SHA-256 only, including the two informative reproductions). `test_logs/`: `frames_log.txt`, `parity_log.txt` (the logs of the two chain
  reconstructions; the study writes them as `.log`), `check_c1crop.txt`, `check_c1crop_mutations.txt`, `linux.txt`.
- `run-report.md` (group template) and `C1CROP_note.md` (results, reading, limits).

Unchanged copies, byte-identical to the accepted packages of `research/pad-2026`: `n1_train.py` (records, atomic
writes, metrics, the forbidden-content scan), `n2_sessions.py` (manifest loading, original lookup by SHA-256, detector
resources, step records), `train_cnn.py` (`preprocess`, the chain of the cache), `nuaa_manifests.py`,
`select_threshold.py`, `path_guard.py`, `m2_writers.py`, `package_rules.py`,
`src/tesi_app/{__init__,paths,evaluation,core,face}.py`, `requirements.txt`.

## Question

C1 (28 September) scored the four NUAA CNNs with the application detector of 24 September, while the CNNs had been
trained on 12 September on a cache built with the earlier detector: for every face found by RetinaFace the box used in
the C1 evaluation is one pixel narrower and one pixel shorter than the box used in training (the margin 0.15 and the
square crop are applied to that box). We reported this on 2026-10-29 with the N2 start; the coordinator asked on
2026-11-01 to quantify it on one checkpoint (AttackNet V2.1 official NUAA) and the 300 C1 images: how much does the
extractor change the scores and the decisions, everything else unchanged? This is not the iOS comparison, which stays
separate; C1, M1, M2, N1, N2, the C2 runner and the R2 reference are unchanged.

## Fixed before the run

- Checkpoint `AttackNetV2_1__nuaa.h5` (134,417,864 bytes, SHA-256 `67d9138d6c43e33164d4510b4b34868c2fdc0a637f27664816c388c4e059f8dc`),
  the C1 checkpoint (its SHA-256 is the fingerprint of the 300 C1 rows), loaded as the application does: the reference
  `architectures.py`, `create_model('AttackNetV2_1').get_model().load_weights(...)`, Grappler remapping off; SHA-256
  verified before loading. CPU only (GPU hidden), batch 16, BatchNorm in inference mode.
- Images: the 300 of the C1 manifest (`nuaa_c1_v2.csv`, 150 bona fide, 150 attacks, 16 subjects, session 03, inside
  the 3,000 N1 test frames), every file located by key and verified by size and SHA-256 at every step.
- Chain (`train_cnn.py` `preprocess`, unchanged): `cv2.imread` (BGR), detector, `Face.crop(margin=0.15, square=True)`
  (the whole frame when no face is found), `cv2.resize` 256×256 `INTER_AREA`, BGR to RGB, `advanced_image_enhancement`
  of the reference code, uint8, then float32 / 255 at scoring. Extractor for extractor, it is the chain of the
  application analyzer used in C1 (same steps, same reader: verified on synthetic images) and the chain that built the
  cache.
- Rule: threshold 0.5 on the attack probability, fixed in C1; attack if `score_attack > 0.5`, a tie is bona fide;
  APCER, BPCER, ACER with denominators and AUROC per extractor; per image the two scores, their absolute and signed
  difference (cache minus C1), the two decisions and the change from the C1 extractor to the cache extractor.
- Two reconstructions precede the comparison and stop the study if they fail:
  - `frames`: the 300 frames preprocessed with `cache_0ac7143` must equal bit for bit their rows of `X_test` in the cache
    (the cache chain); the detector, the box and the frame hash of every image are recorded for both extractors.
  - `parity`: the 300 scores recomputed with `c1_a80f6f9` on the CPU (batch 16, the scoring of this study) must be
    within 1e-5 of the stored C1 scores with the same decision at 0.5 (the C1 chain). The tolerance is the absolute
    tolerance of the coordinator's R2 replication package for the same checkpoint and the same 300 images (1e-5, no
    discordant decision). The stored C1 scores were computed by the application on the Metal GPU one image at a time:
    a first trial with 1e-6 was refused (8 of 300 CPU scores between 1e-6 and 2.9e-6, all decisions equal), while the
    reproduction in the C1 conditions (GPU, one image at a time) stayed within 9e-8; the residual is device and batch
    arithmetic, not the chain. `parity` therefore also records two informative reproductions, in the C1 conditions (the
    default device of the process, one image at a time) and on the CPU one image at a time, with the counts within
    1e-5, 1e-6 and 1e-7; the binding check is the one in the scoring conditions of the study (CPU, batch 16).

## Results (study `20261102-C1CROP-f6747cc`, `tables/aggregate.md`)

Reconstructions: cache chain 300/300 frames equal bit for bit; C1 chain 300/300 scores within 1e-5 (maximum absolute
difference 2.891e-6, 292 within 1e-6, 209 within 1e-7), 300/300 same decisions: the R2 criterion (1e-5, no discordant
decision, same metric row to four decimals) is met in the scoring conditions of the study. Informative reproductions
of the stored C1 scores: in the C1 conditions (GPU, one image at a time) maximum 8.941e-8, 300/300 within 1e-7; on the
CPU one image at a time maximum 6.998e-5, 297/300 within 1e-5, 300/300 same decisions (a one-off measurement of
2026-11-02 on the GPU with batches of 16, not recorded by the study, gave 1.1e-5). A CPU replication of the stored C1
scores one image at a time with this chain can therefore differ by up to 7e-5 on 3 of the 300 images while keeping
every decision: worth knowing for an R2-like replication.

Detector: RetinaFace for all 300 images with both extractors (no Haar fallback, no image without a face); box delta
(C1 minus cache) dx 0, dy 0, dw −1, dh −1 for all 300; no preprocessed frame equal between the extractors (maximum
pixel difference 250 levels, mean 6.4 over the 300 frames).

| extractor | n bona fide | n attack | rejected bona fide | accepted attacks | APCER | BPCER | ACER | AUROC |
|---|---|---|---|---|---|---|---|---|
| `cache_0ac7143` (training cache) | 150 | 150 | 121 | 4 | 0.0267 | 0.8067 | 0.4167 | 0.7092 |
| `c1_a80f6f9` (C1 evaluation) | 150 | 150 | 115 | 3 | 0.0200 | 0.7667 | 0.3933 | 0.7103 |
| C1 delivered (stored scores, `20260928-C1-seed42-a80f6f9`) | 150 | 150 | 115 | 3 | 0.0200 | 0.7667 | 0.3933 | 0.7103 |

Paired differences (cache minus C1, 300 images): maximum absolute 0.3214, median 8.6e-6, mean 0.0142; signed minimum
−0.3164, mean +0.0043, maximum +0.3214. Absolute difference exactly 0 for 97 images, in (0, 1e-4] for 84, in (1e-4, 1e-3]
for 36, in (1e-3, 1e-2] for 37, in (1e-2, 0.1] for 28, above 0.1 for 18. Decisions changed at 0.5: 9 of 300, from the C1
extractor to the cache extractor: 7 bona fide from accepted to rejected, 1 bona fide from rejected to accepted, 1 attack
from rejected to accepted (the nine C1 scores lie between 0.36 and 0.62).

## Reading (descriptive)

- With the extractor consistent with the training cache the checkpoint is slightly worse on the 300 C1 images at the
  fixed threshold: ACER 0.4167 against 0.3933 (+0.0233, that is 6 more bona fide rejected and 1 more attack accepted
  out of 150 each); AUROC unchanged within 0.002 (0.7092 against 0.7103). The one-pixel box difference does not explain
  the C1 result: the BPCER stays above 0.76 with either extractor.
- The effect is concentrated: six images in ten keep the same score to within 1e-4 (181 of 300) and 97 keep exactly the
  same score, while 18 images move by more than 0.1 and 9 cross the threshold (ACER +0.0233). The sign of the mean difference is positive
  (the cache extractor gives slightly higher attack scores on average: +0.004).
- Repeated score values occur (1.0 for 9 images with either extractor; a few other values, almost all above 0.9999,
  shared by 2 to 5 images): observed, not investigated here.
- The delivered C1 row is reproduced by the C1 extractor to four decimals (same 115 and 3, same 300 decisions; AUROC
  0.71029 against 0.71027): the stored C1 scores are consistent with the chain described in C1.

## Limits

- One checkpoint, one subset of 300 images (150 + 150, 16 subjects, session 03), adjacent frames of one session are
  not independent samples; no uncertainty estimate (a paired comparison on 300 images, no resampling).
- The comparison is between two extractors that differ by one pixel of the RetinaFace box on these images (no Haar
  or no-face case in the 300); it says nothing about other detectors, margins or images.
- The C1 scores were computed on the GPU one image at a time; this study scores both extractors on the CPU in batches
  of 16. The parity check bounds that difference (2.9e-6 at most on the 300, no decision changed); the informative
  reproductions show it is device and batch arithmetic (8.9e-8 in the C1 conditions, 7.0e-5 on the CPU one image at a
  time).
- No training, no calibration, no selection: the threshold is the one fixed in C1; nothing is chosen on these images.
  The other three C1 CNNs and the M1 dev scores (574 frames, same detector of 24 September) are not recomputed here;
  C1, M1, M2, N1, N2, the C2 runner and the R2 reference are unchanged.

## Restricted area (not in this repository)

`20261102-C1CROP-f6747cc/`: `status.json`, `check.json`, `check.log`, `frames.csv` (600 rows: per extractor and image, detector,
confidence, box, file and frame SHA-256, equal to the cache row or not), `frames.json`, `frames.log`, `parity.csv` (300
rows: stored and recomputed score, difference, decisions, score in the C1 conditions, frame SHA-256), `parity.json`,
`parity.log`, `scores_paired.csv` (300 rows: key, subject, session, label, file SHA-256, detector and frame SHA-256 per
extractor, the two scores with exact round trip, differences, decisions, change, checkpoint and configuration SHA-256),
`per_subject.csv` (per extractor and subject), `metrics.json`, `score.json`, `score.log`, `aggregate.json`,
`aggregate.md`, `aggregate.log`, `figure_paired.png`; `SHA256SUMS` next to the folder. The step records and the
aggregates hold no identifiers (scanned); the logs can name an image in an error message and stay restricted.

Per-image records and `research/templates/score-schema.json`: own columns (mapping: `sample_id` = manifest key,
`dataset_id` = `nuaa`, `split` = `test`, `attack_type` = `print` for label 1, `modality` = `RGB`, `threshold` = 0.5
fixed a priori; `prediction_*` = 1 iff `score_* > 0.5`).

## Commands

From `livenesslab/` of this folder (in the development repository, from its root), with the restricted inputs (the M2
manifests folder, the original images, the cache, the checkpoint, the C1 `scores.csv`) and the reference code
(`LIVENESSLAB_LIVEDETECTION` pointing to the folder with `scripts/create_datasets.py` and `src/architectures.py`;
`LIVENESSLAB_SILENT_FACE` to the Silent-Face-Anti-Spoofing folder with the RetinaFace model):

```
C=(--config ../c1crop.json --manifests <M2 manifests folder> --images <NUAA images root> --cache <path>/nuaa_1500.npz --weights <path>/AttackNetV2_1__nuaa.h5 --c1-scores <path>/20260928-C1-seed42-a80f6f9/scores.csv --face-cache ../face_0ac7143.py --face-c1 ../face_a80f6f9.py --study-dir <outside the repository>/YYYYMMDD-C1CROP-<shortcommit>)
python scripts/c1crop_check.py check "${C[@]}"
python scripts/c1crop_check.py frames "${C[@]}"
python scripts/c1crop_check.py parity "${C[@]}"
python scripts/c1crop_check.py score "${C[@]}" --i-am-authorized
python scripts/c1crop_check.py aggregate "${C[@]}"
```

Every command ends with `RESULT: PASS (<command>)` or `RESULT: FAIL (<command>)` (exit code 1; a wrong command line is
refused by the argument parser with exit code 2 before any step starts). The study folder must be new at `check`;
inside the development repository only `results/c1crop/` is accepted, inside the group repository no output folder is
accepted. A non-trial study requires the delivered configuration (SHA-256 bound in the code), the current commit in its
name and a clean work tree; the SHA-256 of every code file and of the two frozen copies are recorded at `check` and
compared at every later step; every record is bound to the previous one by SHA-256 (`check.json` → `frames.json` →
`parity.json` → `score.json`), and the files a record lists are verified before the next step. `check` imports neither
TensorFlow (`tf_state`: the step fails if TensorFlow is loaded, and `check.json` records `not imported`) nor the detectors; `frames` uses the two detectors only; `parity` and `score` use
TensorFlow (the scoring of the study on the CPU in batches of 16; the two informative reproductions of `parity` on the
default device and on the CPU, one image at a time). The
outputs of `frames`, `parity` and `score` are never recomputed in place (after a `FAIL` the folder keeps its log and
`status.json` with the step and the reason, and the step is run again in a new study folder); `aggregate` removes its
previous outputs before recomputing. A refusal before the folder is accepted (wrong inputs, missing record, existing
outputs) leaves the `status.json` of the previous step untouched.

## Tests

From `research/c1-crop-check/livenesslab/` (`python` is Python 3.11 or later with NumPy and scikit-learn; data,
weights, results folders, TensorFlow and OpenCV are not needed: without OpenCV the suite registers an empty `cv2` module
and prints a `NOTE` line):

```
python scripts/check_c1crop.py
python scripts/check_c1crop_mutations.py
```

Expected last lines (also in `test_logs/`, with the Linux run in `test_logs/linux.txt`):

```
tests run: 52, failures: 0, errors: 0, skipped: 0
RESULT: PASS
```

```
unmutated copy (after): PASS (tests run: 52, failures: 0, errors: 0)
mutations killed: 74/74
control mutations survived: 1/1
RESULT: PASS
```

The mutation harness is the one of N1 and N2: every mutation in a fresh copy of the code, the unmutated suite in a fresh
copy before and after, a control mutation (a comment only) that must survive, a mutation counted as killed only when
the suite runs and reports failures or errors with the same number of tests. The copies include the delivered
configuration and the two frozen copies (`delivery/c1crop/` in the development repository, `../` here), so that
`test_delivered_configuration` runs: it checks the configuration against the schema and the SHA-256 bound in the code,
the two frozen copies against their SHA-256 and the C1 copy against the current `face.py`. The suite injects the
preprocessing, the two scorers, the figure and the repository state: no image, no model, no detector, no TensorFlow. The
only possible skip is `test_delivered_configuration` when the configuration is not next to the code. Two checks of the
code are declared redundant and not mutated: `same_inputs` after `frames` (every later record is bound to `check.json`
by SHA-256) and the exact CSV round trip of the scores (`repr` is exact).

## Files and integrity

`SHA256SUMS` lists every file of this folder but itself (`shasum -a 256 -c SHA256SUMS` from this folder). The SHA-256 of
the delivered configuration is bound in `c1crop_check.py`; the SHA-256 of the code files and of the two frozen copies
recorded in `check.json` of the study are equal to the files of this folder.
