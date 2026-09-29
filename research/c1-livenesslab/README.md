# C1 — LivenessLab: traceable starting point

Task **C1** of the PAD research workflow (`research/README.md`): share the current version of LivenessLab, its
environment and run instructions, and the material needed to reproduce one table that was already obtained.
No new training was run for this task.

This folder is a filtered export of the operator's development repository (private; the run report records its
current commit, `73d00d3`, and the table header the commit at which the scores were produced, `a80f6f9`): application code, scripts, environment files,
checkpoint cards and aggregate results, without data, weights, per-subject records or working notes. One file is
redacted in the export: `src/tesi_app/usage_report.py` (usage report of the public instance) has its list of the
operator's IP addresses emptied. Code comments, user-interface texts and log messages are in Italian by the
developer's convention; every document of this folder, the table header and the per-sample schema are in English.

Contents of this folder:

| path | what it is |
|---|---|
| `run-report.md` | run report compiled from `research/templates/run-report.md` |
| `inventory.md` | CNNs, datasets, checkpoints and pre-trained models with origin and SHA-256 |
| `config/nuaa_config.json` | evaluation configuration: conventions, threshold, analyzers with fingerprints, checkpoints with hashes, commands |
| `manifests/README.md`, `manifests/nuaa_manifest.sha256` | summary of the evaluation manifest: counts, selection rule and SHA-256 of the per-image manifest (the per-image file is a per-subject record and stays in the restricted folder) |
| `tables/nuaa_table.md` | the reference table (aggregate metrics per analyzer) with its reproducibility header |
| `checkpoints/*.json`, `checkpoints/PROVENANCE.md` | one card per weight file (hash, protocol, training counts, test metrics); the weight files themselves are not here |
| `livenesslab/` | the application and the scripts (`src/tesi_app`, `scripts`, `run_app.py`, `requirements*.txt`) |

Per-sample scores (`scores.csv` in the schema of `research/templates/score-schema.json`), the per-image manifest,
the weight files and the images stay out of this public repository, as required: they are delivered in the project
workspace under `02_Experiments/C1/<run-id>/` and `03_Models/C1/`, with the SHA-256 listed in `run-report.md`.

## The reference table

`tables/nuaa_table.md`: NUAA Imposter, 300 images from the official test set (150 bona fide, 150 printed-photo
attacks, taken at a constant step over the official file lists; the bona fide cover 9 subjects and the attacks 15),
evaluated with all 35 analyzers of LivenessLab at the fixed threshold 0.5 on the attack score. Label 0 = bona fide,
1 = attack; the score is the probability of attack; a tie at the threshold counts as bona fide. Metrics follow
ISO/IEC 30107-3: APCER, BPCER, ACER and accuracy at the threshold, EER on the full ROC without interpolation,
BPCER at APCER = 10 %, AUC. The main rows are the four CNNs trained on the official NUAA training split
(`livenessnet__nuaa`, `attacknet_v1__nuaa`, `attacknet_v2_1__nuaa`, `attacknet_v2_2__nuaa`); the same table also
reports the 16 checkpoints trained on CASIA-FASD, CelebA-Spoof and SynthASpoof (cross-dataset rows), the pre-trained
models and the classic methods. The four "pooled" NUAA checkpoints are listed but not evaluated: their random 80/20
split shares subjects and sessions with these test images (24 CNN checkpoints in total: 4 + 4 + 16).

The table was computed with the evaluation module of LivenessLab, which differs from `src/evaluation_utils.py` of this
repository in three documented points: BPCER@APCER10 counts rejected bona fide (not accepted ones), non-computable
values are `null` instead of 0, and the EER is searched on the full ROC (`drop_intermediate=False`).

## Two commands, kept distinct

All commands run inside `livenesslab/` with the environment below.

**1. Recompute the table from the saved scores** (no images, no weights, a few seconds):

```bash
python scripts/eval_dataset.py --dataset nuaa --from-cache --lang en \
    --export results/c1/nuaa_scores.csv --table results/c1/nuaa_table.md
```

Real dependencies of this mode: the score cache `results/eval/nuaa.json` (restricted artefact; SHA-256 in the run
report), placed under `LIVENESSLAB_RESULTS_DIR/eval/`, and the checkpoint JSON cards of `checkpoints/` copied into
`LIVENESSLAB_WEIGHTS_DIR` (they enumerate the CNN variants and classifiers and carry their SHA-256). Nothing else:
no weight files, no images, no Silent-Face resources, no Hugging Face models. In this mode the model fingerprints
are read from the cache and reported in the CSV, not recomputed from local files (the script says so on start);
verification of fingerprints and of checkpoint hashes belongs to the inference mode. The script recomputes every
metric from the per-image scores and rewrites the table; a reviewer can diff the result against
`tables/nuaa_table.md` in this folder. Verified on 2026-09-28 from this export, with `src/architectures.py` of this
repository as `LIVENESSLAB_LIVEDETECTION`, the checkpoint cards only (no weight files), an empty Silent-Face folder
and an empty, offline Hugging Face cache: the output is identical (`recompute_from_cache.log`,
`hf_offline_recompute.log` and `operator_recompute_from_cache.log` in the restricted run folder).

**2. Run a new inference** (images and weights required; 535 s measured on an Apple M4 with the Metal backend for the delivered run, see `new_inference.log`; longer on CPU only):

```bash
python scripts/eval_dataset.py --dataset nuaa --force --lang en \
    --export results/c1/nuaa_scores.csv --table results/c1/nuaa_table.md
```

It needs `data/eval/nuaa/` (the 300 images of the restricted manifest, verifiable by hash), `models/weights/`
(the `.h5` checkpoints listed in `inventory.md` with their JSON cards), the two external code bases and the Hugging
Face cache of the pre-trained models (see below). Before running, the script compares the SHA-256 of every checkpoint
file with the value in its card and stops on any mismatch or missing file. Without `--force` the script reuses
cached scores whose model fingerprint is unchanged and computes only what is missing.

Expected result: the table matches `tables/nuaa_table.md` in every printed digit. Full new inferences run on
2026-09-28 (300 images, all analyzers, Apple M4 with the Metal GPU backend for the CNNs) reproduced all 9,000
per-sample scores with a maximum absolute difference of 1.3e-7 with respect to the delivered scores
(`new_inference.log`, `new_inference_vs_delivered.txt`, `operator_new_inference.log` in the restricted run folder);
no CNN score lies within 1e-5 of the threshold, so differences of that order do not change the table. A CPU-only
replication may show differences of the same order; any larger difference must be reported in the run report.

## Environment

- Python 3.11; `pip install -r requirements.txt` (macOS and Linux: `tensorflow-metal` is marked for macOS on Apple Silicon only) or `requirements-windows.txt` (Windows, CPU).
- Reference CNN code: this repository's `src/architectures.py` is loaded by file; point
  `LIVENESSLAB_LIVEDETECTION` at the repository root (the folder that contains `src/`).
- MiniFASNet and RetinaFace: clone `https://github.com/minivision-ai/Silent-Face-Anti-Spoofing` (Apache-2.0) and
  point `LIVENESSLAB_SILENT_FACE` at it (the commit used is recorded in `config/nuaa_config.json`).
- Pre-trained models: `python scripts/download_models.py` downloads CLIP ViT-B/32, Depth Anything V2 small and
  DINOv2 small at pinned revisions into the Hugging Face cache, and the MediaPipe face landmarker
  (`face_landmarker.task`) into `<weights folder>/../third_party/` (i.e. `models/third_party/` next to
  `models/weights/`; with `LIVENESSLAB_WEIGHTS_DIR=/x/03_Models/C1` it goes to `/x/third_party/`). Face Mesh is a
  descriptive analyzer without score: if the file is missing, its row reports the error and the table is unaffected.
- Data and weights go where the environment variables say (never inside the repository):

| variable | content | default |
|---|---|---|
| `LIVENESSLAB_DATA_DIR` | `eval/nuaa/{real,attack}/*.jpg` | `livenesslab/data` |
| `LIVENESSLAB_WEIGHTS_DIR` | `*.h5` (CNN, the file the app loads) / `*.joblib` (classifiers) with their `*.json` cards (cards alone suffice to recompute from the cache) | `livenesslab/models/weights` |
| `LIVENESSLAB_RESULTS_DIR` | `eval/<dataset>.json` score caches, `c1/` outputs | `livenesslab/results` |
| `LIVENESSLAB_LIVEDETECTION` | folder containing `src/architectures.py` | `livenesslab/src/livedetection` |
| `LIVENESSLAB_SILENT_FACE` | Silent-Face-Anti-Spoofing clone | `livenesslab/src/third_party/Silent-Face-Anti-Spoofing` |

**Consistency check** (repeatable), after command 1:

```bash
python scripts/check_c1_consistency.py --dataset nuaa --c1 results/c1 \
    --run-dir <restricted run folder> --self-test
```

It compares the score cache, the exported CSV (`results/c1/nuaa_scores.csv`), `scores.csv` of the run folder and
the table (`results/c1/nuaa_table.md`) by sets: the analyzers with a score must be exactly the registered ones that
are neither excluded nor descriptive, the samples exactly those of the manifest (`nuaa_manifest.csv`, which must
be non-empty and without duplicates), with no missing, extra or duplicated row in the CSV files and the table.
Row policy for the two CSV files: every row must be an expected (analyzer, sample) pair, present exactly once, with a
finite score in [0, 1]; sets and duplicates are checked on every row before any score is read, so a row with an empty
or `nan` score still counts as an extra or duplicated row and its score is rejected as well; rows of excluded or
descriptive analyzers must not appear (they are rejected as extra). Scores compared by exact float equality,
predictions at the threshold, `n`, the two denominators and the error / no-face counts on every table row, metrics
recomputed and compared at four decimals, the rows without score checked as a set. For this delivery: 30 numeric
rows × 300 samples = 9,000 unique pairs and 5 rows without score. `--self-test` repeats the check on nine faulty
copies (a table row removed, a table row duplicated, a denominator altered, a sample removed from `scores.csv`, a
row duplicated in `scores.csv`, an extra analyzer in the CSV, an extra analyzer in the CSV with an empty score, a
duplicated pair in the CSV with an empty score, a duplicated pair in the CSV with a `nan` score) and requires each
to be rejected; it is skipped
when the main check already reports problems. It writes `consistency_check.txt` in the run folder (also when an
input file is missing, provided the run folder exists) and exits with status 1 on any problem; an unexpected error
during the check is reported as a problem with the stage it occurred in, not as a traceback. `--c1` is the folder written by command 1 (relative to the current directory); `--run-dir` is the
restricted run folder, where the checker also finds `nuaa_manifest.csv` and `nuaa_config.json` when they are not in
`--c1` (command 1 does not write them; `scripts/make_c1_manifest.py` regenerates them from the images).

Optional: `python run_app.py` starts the web application (http://127.0.0.1:8000), whose "Valutazione dataset" tab
runs the same evaluation interactively and exports Excel/PDF; `python scripts/validate_caches.py` checks that every
cached score matches the current model fingerprints. The sample images of the public instance
(https://tesi.valeriocassano.com) are not part of this folder: the application starts without them (upload or
webcam). Verified on 2026-09-26 from a copy of this folder with the same environment variables: 35 analyzers registered,
a full analysis of an image completed through the WebSocket.

## Differences from the material sent by e-mail on 2026-09-26

The table is the same one attached to the e-mail (PDF/Excel "Risultati completi", NUAA sheet), regenerated from the
same score cache at the commit recorded in its header; numbers are unchanged. New in this delivery: the per-image
manifest with hashes (restricted folder; summary and hash here), the configuration file, the inventory with SHA-256
for every checkpoint, the per-sample score file in the group schema, and the environment variables for data paths
(previously fixed inside the repository). Changes after the first review (2026-09-27): per-image manifest removed
from the public folder; checkpoint entries name the `.h5` file actually loaded (its hash was already the one
reported) and list the `.keras` copies separately; hash check before inference; cache mode made independent of
model files; per-sample scores exported at round-trip precision; `tensorflow-metal` restricted to macOS. Changes
after the second review (2026-09-28): consistency check rewritten by sets with a self-test (see above); run report,
inventory and this README aligned on the seed of the 2026-09-12 checkpoints, on the count of CNN checkpoints
(24 = 4 + 4 + 16) and on the review status; numbers unchanged. Changes after the third review (2026-09-29): in the
consistency check, sets and duplicates of the exported CSV and of `scores.csv` are verified on every row before any
score filter, empty or non-finite scores are rejected, and three faulty copies with empty or `nan` scores were added
to the self-test; numbers unchanged.

## Scope and limits of the NUAA table

- The table is a descriptive evaluation of existing checkpoints and models on a 300-image subset of the official
  NUAA test set (150 bona fide, 150 print attacks; the full test set has 3,362 and 5,761), at one fixed threshold
  (0.5 on the attack probability) chosen a priori, without any development set. EER, BPCER@APCER10 and AUC are
  threshold-free descriptive statistics of the same scores. No confidence intervals are given.
- The evaluation unit is the image; images of the same subject and session are correlated (still frames of webcam
  sessions), so the 300 samples are not 300 independent observations, and the counts (150 / 150) are image counts.
- NUAA's official protocol shares subjects 1–9 between the training set (subjects 1–9) and the test set (subjects
  1–16), in different sessions; the four reference checkpoints follow that protocol, so they are not subject-disjoint
  from the test images by design (in this subset, 3 of the 9 bona fide subjects and 9 of the 15 attack subjects also
  appear in the training set).
  The pooled checkpoints, whose random 80/20 split shares subjects and sessions, are excluded from the table.
- Checkpoints trained on other datasets and the pre-trained models are cross-dataset or zero-shot rows: same threshold,
  no adaptation to NUAA. Their numbers on this subset say nothing about generalization in general (single dataset,
  single seed).
- The checkpoints dated 2026-09-12 were trained with a version of `scripts/train_cnn.py` that predates the
  repository (same pipeline, seed 42 only for splitting and shuffling): see `checkpoints/PROVENANCE.md`.
- The table is not a replication of any published result and does not certify the numbers of the reference
  publications: different data, subset and threshold.
