# C1 — LivenessLab: traceable starting point

Task **C1** of the PAD research workflow (`research/README.md`): share the current version of LivenessLab, its
environment and run instructions, and the material needed to reproduce one table that was already obtained.
No new training was run for this task.

Contents of this folder:

| path | what it is |
|---|---|
| `run-report.md` | run report compiled from `research/templates/run-report.md` |
| `inventory.md` | CNNs, datasets, checkpoints and pre-trained models with origin and SHA-256 |
| `config/nuaa_config.json` | evaluation configuration: conventions, threshold, analyzers with fingerprints, checkpoints with hashes, commands |
| `manifests/nuaa_manifest.csv` | the 300 evaluation images (150 bona fide, 150 print attacks): file, label, subject, session, SHA-256 |
| `tables/nuaa_table.md` | the reference table (aggregate metrics per analyzer) with its reproducibility header |
| `checkpoints/*.json`, `checkpoints/PROVENANCE.md` | one card per weight file (hash, protocol, training counts, test metrics); the weight files themselves are not here |
| `livenesslab/` | the application and the scripts (`src/tesi_app`, `scripts`, `run_app.py`, `requirements*.txt`) |

Per-sample scores (`scores.csv` in the schema of `research/templates/score-schema.json`), the weight files and the
images stay out of this public repository, as required: they are delivered in the project workspace under
`02_Experiments/C1/<run-id>/` and `03_Models/`, with the SHA-256 listed in `run-report.md`.

## The reference table

`tables/nuaa_table.md`: NUAA Imposter, 300 images from the official test set (150 bona fide, 150 printed-photo
attacks, taken at a constant step over the official file lists; the bona fide cover 9 subjects and the attacks 15),
evaluated with all 35 analyzers of LivenessLab at the fixed threshold 0.5 on the attack score. Label 0 = bona fide,
1 = attack; the score is the probability of attack; a tie at the threshold counts as bona fide. Metrics follow
ISO/IEC 30107-3: APCER, BPCER, ACER and accuracy at the threshold, EER on the full ROC without interpolation,
BPCER at APCER = 10 %, AUC. The main rows are the four CNNs trained on the official NUAA training split
(`livenessnet__nuaa`, `attacknet_v1__nuaa`, `attacknet_v2_1__nuaa`, `attacknet_v2_2__nuaa`); the same table also
reports the checkpoints trained on CASIA-FASD, CelebA-Spoof and SynthASpoof (cross-dataset rows), the pre-trained
models and the classic methods. The four "pooled" NUAA checkpoints are listed but not evaluated: their random 80/20
split shares subjects and sessions with these test images.

The table was computed with the evaluation module of LivenessLab, which differs from `src/evaluation_utils.py` of this
repository in three documented points: BPCER@APCER10 counts rejected bona fide (not accepted ones), non-computable
values are `null` instead of 0, and the EER is searched on the full ROC (`drop_intermediate=False`).

## Two commands, kept distinct

All commands run inside `livenesslab/` with the environment below.

**1. Recompute the table from the saved scores** (no images, no weights, a few seconds):

```bash
python scripts/eval_dataset.py --dataset nuaa --from-cache \
    --export results/c1/nuaa_scores.csv --table results/c1/nuaa_table.md
```

It needs the score cache `results/eval/nuaa.json` (delivered with the restricted artefacts; SHA-256 in the run
report), placed under `LIVENESSLAB_RESULTS_DIR/eval/`, and the checkpoint cards of `checkpoints/` copied into
`LIVENESSLAB_WEIGHTS_DIR` (the JSON cards alone identify every checkpoint by its SHA-256; the weight files are not
required). No image is needed. It recomputes every metric from the per-image scores and rewrites the table; a
reviewer can diff the result against `tables/nuaa_table.md` in this folder. Verified on 2026-09-26 from this
export, with `src/architectures.py` of this repository as `LIVENESSLAB_LIVEDETECTION`: the output is identical.

**2. Run a new inference** (images and weights required; a few minutes on CPU):

```bash
python scripts/eval_dataset.py --dataset nuaa --force \
    --export results/c1/nuaa_scores.csv --table results/c1/nuaa_table.md
```

It needs `data/eval/nuaa/` (the 300 images listed in `manifests/nuaa_manifest.csv`, verifiable by hash),
`models/weights/` (the checkpoints listed in `inventory.md`), the two external code bases and the Hugging Face cache
of the pre-trained models (see below). Without `--force` the script reuses cached scores whose model fingerprint is
unchanged and computes only what is missing.

Expected result: the table matches `tables/nuaa_table.md` in every printed digit. A full new inference run by the
operator on 2026-09-26 (300 images, all analyzers, Apple M4) reproduced all 9,000 per-sample scores with a maximum
absolute difference of 1.3e-7 with respect to the delivered scores (floating-point noise of the CNN backend); any
larger difference must be reported in the run report.

## Environment

- Python 3.11; `pip install -r requirements.txt` (macOS/Linux) or `requirements-windows.txt` (Windows, CPU).
- Reference CNN code: this repository's `src/architectures.py` is loaded by file; point
  `LIVENESSLAB_LIVEDETECTION` at the repository root (the folder that contains `src/`).
- MiniFASNet and RetinaFace: clone `https://github.com/minivision-ai/Silent-Face-Anti-Spoofing` (Apache-2.0) and
  point `LIVENESSLAB_SILENT_FACE` at it (the commit used is recorded in `config/nuaa_config.json`).
- Pre-trained models: `python scripts/download_models.py` downloads CLIP ViT-B/32, Depth Anything V2 small and
  DINOv2 small at pinned revisions into the Hugging Face cache.
- Data and weights go where the environment variables say (never inside the repository):

| variable | content | default |
|---|---|---|
| `LIVENESSLAB_DATA_DIR` | `eval/nuaa/{real,attack}/*.jpg` | `livenesslab/data` |
| `LIVENESSLAB_WEIGHTS_DIR` | `*.keras` / `*.joblib` with their `*.json` cards (cards alone suffice to recompute from the cache) | `livenesslab/models/weights` |
| `LIVENESSLAB_RESULTS_DIR` | `eval/<dataset>.json` score caches, `c1/` outputs | `livenesslab/results` |
| `LIVENESSLAB_LIVEDETECTION` | folder containing `src/architectures.py` | `livenesslab/src/livedetection` |
| `LIVENESSLAB_SILENT_FACE` | Silent-Face-Anti-Spoofing clone | `livenesslab/src/third_party/Silent-Face-Anti-Spoofing` |

Optional: `python run_app.py` starts the web application (http://127.0.0.1:8000), whose "Valutazione dataset" tab
runs the same evaluation interactively and exports Excel/PDF; `python scripts/validate_caches.py` checks that every
cached score matches the current model fingerprints. The sample images of the public instance
(https://tesi.valeriocassano.com) are not part of this folder: the application starts without them (upload or
webcam). Verified on 2026-09-26 from this folder with the same environment variables: 35 analyzers registered, a
full analysis of an image completed through the WebSocket.

## Differences from the material sent by e-mail on 2026-09-26

The table is the same one attached to the e-mail (PDF/Excel "Risultati completi", NUAA sheet), regenerated from the
same score cache at the commit recorded in its header; numbers are unchanged. New in this delivery: the manifest with
per-image hashes, the configuration file, the inventory with SHA-256 for every checkpoint, the per-sample score file
in the group schema, and the environment variables for data paths (previously fixed inside the repository).

## Known limits recorded for review

- The threshold is fixed a priori at 0.5, not selected on a development set; EER and BPCER@APCER10 are descriptive.
- The checkpoints dated 2026-09-12 were trained with a version of `scripts/train_cnn.py` that predates the
  repository (same pipeline, seed 42 only for splitting and shuffling): see `checkpoints/PROVENANCE.md`.
- NUAA's official split is not subject-disjoint (subjects 1–9 appear in both train and test, in different sessions).
- Frames of the same session are not independent samples; counts are per image.
