# M2, first tranche: NUAA manifests and split overlap check

Task **M2** of the PAD research workflow: versioned manifests of the RGB datasets, linked to their source, classes,
transformations and per-split counts, and a repeatable check of overlaps and duplicates between train, dev and test.
This first tranche covers **NUAA** only (CASIA-FASD and CelebA-Spoof follow). No training and no inference: the
scripts list and hash files and read the training cache (subject, label and frame arrays; the frames are only hashed).

This folder is a filtered export of the operator's development repository (private; development commit `36c677a`,
run `20261006-M2-seed42-36c677a`): the scripts needed to rerun the checks, the documents, the declared rules and the aggregate results,
without data, weights, per-image manifests or per-subject records. The per-image manifests, the private summary, the
restricted overlap and short reports and the logs are in the project workspace under `02_Experiments/M2/20261006-M2-seed42-36c677a/`,
with their SHA-256 listed in the public aggregate and in `run-report.md`. Code comments, command-line help and progress
messages are in Italian by the developer's convention; reports, error messages and every document of this folder are
in English.

Per-subject counts (by class and by subject and session) and the identifiers of the two validation subjects are in
the restricted summary `nuaa_manifests_summary.json` (`by_subject`, `by_subject_session`), because `research/README.md`
keeps per-subject records out of the public repository. The public aggregate gives counts per class, session, glasses
and pos, the number of subjects per class, and counts of rows of shared subjects without identifiers.

## Glossary

- **validation** = `cnn_es` (493 images, 2 subjects): never used for gradient updates; their `val_loss` controlled
  early stopping (patience 4, best weights restored), hence which epoch's weights were saved, and the learning-rate
  reduction (ReduceLROnPlateau, factor 0.5, patience 2). `cnn_es` ⊂ M1 dev (574 images: all official training images
  of the same two subjects).
- The prepared folders `{bonafide,attack}_validation` contain the **official test split** (their name comes from the
  preparation script); they were never used as validation data.
- **split lists (mirror copy)**: the four `*_raw.txt` lists of the mirror archive (see Source).

## Source and field names

- Field order: `ID_glasses_pos_session_picNo`, as stated on the M2 task card (3 October 2026), where the coordinator
  took it from the README of the original release; that README is not in the mirror and was not available to us. The
  reference publication of the database is Tan et al., ECCV 2010. On the data we verified the structure (five numeric
  fields), that ID equals the subject folder, and that the session field separates the official splits (01–02 train,
  03 test); the meaning of the glasses and pos codes is not verified and they are carried as opaque codes.
- Split lists: the mirror's lists carry the path prefix `/kaggle/input/nuaaaa/raw/`, added by the mirror author: they
  are the mirror's version of the official lists. Only file names are compared. We did not verify their equivalence
  with the lists of the original release; the coordinator rebuilt the 300 C1 images from the official source with
  identical hashes (M1 card, 2 October 2026). `build` accepts only the mirror's list format.
- Subjects: the 4-digit IDs of the official folders and file names. No identity is inferred from image content or from
  groups of images. Treating the same ID in ClientRaw and ImposterRaw as the same person follows the naming of the
  release and is not verified on the images.

Details in `naming_schema.md`.

## What is delivered

| file | what it is | where |
|---|---|---|
| `naming_schema.md` | NUAA file naming schema (field order from the M2 card, structure checked on the data), parsing, checks against the split lists, C1 session correction and versioning | public |
| `transformations.md` | transformations from the image files to the derived frames of the training cache and to the CNN inputs (preprocessing, augmentation, use of the validation images), from the code | public |
| `nuaa_overlap_rules.json` | the declared manifests (file, expected rows, expected SHA-256) and the overlap and derivation rules (must-hold and report-only) | public |
| `nuaa_manifests_aggregate.json` | source and revision, SHA-256 of the archive and of the four split lists, per-manifest SHA-256 and counts per class, session, glasses and pos and number of subjects, checkpoint linkage, reconstruction checks, derived-frame duplicate check, exclusions, C1 correction counts, recountable block (`recount`) | public |
| `overlap_report_public.md`, `overlap_report_public.json` | pairwise overlap matrix (key, content hash, subject, subject and session), internal duplicates, rule verdicts; counts only | public |
| `overlap_report.md`, `overlap_report.json` | the same, with example keys for any failed must-hold check | restricted |
| `overlap_self_test.md`, `overlap_self_test.json` | self-test of the rules on faulty copies of the real manifests | restricted (summary in `report.md` and `run-report.md`) |
| `report.md` | short report: counts, overlaps, exclusions, test images from unseen subjects, supported comparisons, information that cannot be reconstructed, declarations | public version without subject identifiers |
| `run-report.md` | run report compiled from `research/templates/run-report.md` (also in the restricted folder) | public |
| `manifests/*.csv` | per-image manifests (below) | restricted |
| `nuaa_manifests_summary.json`, private report | as the aggregate, plus per-subject counts and the identifiers of the two validation subjects | restricted |
| `dataset_register_nuaa.md` | draft entry of the dataset register (`01_Dataset_Register`) | restricted |
| `livenesslab/` | `scripts/nuaa_manifests.py` (builder and checker), `scripts/check_nuaa_manifests.py` (offline tests), `scripts/package_rules.py` (forbidden-content rules), `scripts/select_threshold.py` (reconstruction of the CNN split, reused), `src/tesi_app/paths.py`, `delivery/m2/` (rules and report template read by the scripts), `requirements*.txt` | public (code) |
| `SHA256SUMS` | SHA-256 of every other file of this folder, `livenesslab/` included | public |

Fields that change at every run and are not part of the evidence: `created`, `commit`, `dirty` (summary and
aggregate), `checked`, `commit`, `dirty` and `manifests_dir` (overlap reports and self-test).

## Manifests

Columns: `key` (`real/<name>` or `attack/<name>`: class folder plus prepared file name; the split folder is
deliberately not part of the key, and the class is, because the same name exists in both classes), `label`
(0 bona fide, 1 attack), `subject`, `glasses`, `pos`, `session`, `pic` (fields of the file name, see
`naming_schema.md`), `official_split` (train/test from the split lists), `sha256`, `bytes`. The archive member of a
row is `raw/<ClientRaw|ImposterRaw>/<subject>/<key without the class folder and the subject prefix>`; `build` checks
for all 12,614 rows that the prepared file is that member (same inode, or same SHA-256). There is no attack-type
column: NUAA has a single attack type (printed photographs), implied by label 1.

| name | file | content | images |
|---|---|---|---|
| `official_train` | `nuaa_official_train.csv` | all images of the train split lists | 3,491 |
| `official_test` | `nuaa_official_test.csv` | all images of the test split lists | 9,123 |
| `cnn_train` | `nuaa_cnn_train.csv` | images that fitted the weights of the four NUAA official CNNs | 2,507 |
| `cnn_es` | `nuaa_cnn_early_stopping.csv` | validation images of the four CNNs (see Glossary) | 493 |
| `cnn_test` | `nuaa_cnn_test.csv` | test images of the checkpoint cards: `spread(..., 1500)` per class of the official test split | 3,000 |
| `m1_dev` | `nuaa_m1_dev.csv` | M1 source-dev set: all official training images of the two validation subjects | 574 |
| `c1` | `nuaa_c1_v2.csv` | the 300 C1 test images, manifest version 2 (session corrected) | 300 |

The training split of the CNNs is rebuilt with the code of the training script (`spread(sorted file list, 1500)` per
class, `GroupShuffleSplit(test_size=0.2, random_state=42)` on the subject prefix) and verified against the checkpoint
cards (n_train 2,507, n_val 493, n_test 3,000) and against the subject and label arrays of the training cache
(`nuaa_1500.npz`, train and test). Independently of the scikit-learn version, among the 36 pairs of the 9 training
subjects only one gives 493 validation images, the count recorded in the cards. Each of the four CNNs (LivenessNet,
AttackNetV1, AttackNetV2_1, AttackNetV2_2, `__nuaa`) is linked by the SHA-256 of its `.h5` weights file, read from the
card and recomputed on the file. No new split is defined in M2; the CNN split is by subject, after the per-image
preprocessing and before the augmentation; no frame is extracted from video. The NUAA pooled checkpoints are not covered (they are excluded from the C1 NUAA table because their random split shares subjects and sessions with the test images). The CNNs are trained from scratch (C1 inventory); the derived frames use the pre-trained
RetinaFace detector.

Derived frames: the training cache holds the preprocessed frames (`X_train`, `X_test`, 3,000 each). `build` hashes
each frame as stored and requires no duplicate inside either array, none between them, and none between the fitting
and the validation part of the training array. The frames carry no file names: they are linked to the files through
the sampling order and the subject and label arrays, not image by image.

## Commands

Dependencies: Python 3.11; numpy 1.26.4 for every subcommand and for the tests (imported through
`select_threshold.py`; without it the scripts stop with a clear error); scikit-learn 1.3.2 for `build` and
`reconstruct`. `build` also needs the images, the checkpoint cards and weights, the training cache and the C1 and M1
results. From inside `livenesslab/` (or the root of the development repository), with the NUAA data prepared by
`scripts/prepare_datasets.py` of the C1 package (the archive and its split lists under `data/raw/`, the prepared
folders under `data/NUAA/images/`), the checkpoint cards and weights in `models/weights/`, the training cache in
`data/processed/`, and the C1 and M1 results (`results/c1/`, `results/m1/`) that `build` cross-checks:

```bash
python scripts/nuaa_manifests.py build                       # manifests, private summary, public aggregate (exit 1 if a check fails)
python scripts/nuaa_manifests.py check-overlap --self-test   # rules of nuaa_overlap_rules.json and self-test (exit 1 on any failure)
python scripts/nuaa_manifests.py report                      # short report, private and public versions
python scripts/check_nuaa_manifests.py                       # offline tests (real-data tests skipped without data)
```

Data, results and weights folders can be moved with `LIVENESSLAB_DATA_DIR`, `LIVENESSLAB_RESULTS_DIR` and
`LIVENESSLAB_WEIGHTS_DIR`; every command takes `--out` to write somewhere else.

## Verification without images or weights

With `M2_RUN` pointing at the restricted run folder and `C1_RUN` at the restricted C1 run folder (it holds the C1
manifest version 1, `nuaa_manifest.csv`), from inside `livenesslab/`. Without the images, content hashes are taken as
recorded: a changed hash of an image outside every derived manifest is not detected; only `build` checks the hashes
against the files.

1. completeness and anchor of the two folders (section "Integrity");
2. overlap rules and self-test on the restricted manifests (numpy only); the verdicts must equal those of
   `overlap_report_public.json` and the self-test must reject 7 of 7 faulty copies:

   ```bash
   python scripts/nuaa_manifests.py check-overlap --self-test --manifests-dir "$M2_RUN/manifests" --out /tmp/m2-overlap
   ```
3. counts of the public aggregate recomputed from the manifests alone (per manifest, exclusions, test images from
   subjects absent from the train, reuse of the validation images); exit 1 if they differ:

   ```bash
   python scripts/nuaa_manifests.py recount --manifests-dir "$M2_RUN/manifests" --aggregate ../nuaa_manifests_aggregate.json
   ```
4. CNN split and M1 dev rebuilt from `nuaa_official_train.csv` alone (scikit-learn, no image read) and compared with
   `cnn_train`, `cnn_es` and `m1_dev`:

   ```bash
   python scripts/nuaa_manifests.py reconstruct --manifests-dir "$M2_RUN/manifests"
   ```
5. C1 manifest version 1 against version 2: 300/300 equal by key, SHA-256, bytes, subject and label, and the old
   `session` column equal to `glasses`:

   ```bash
   python scripts/nuaa_manifests.py compare-c1 --v1 "$C1_RUN/nuaa_manifest.csv" --v2 "$M2_RUN/manifests/nuaa_c1_v2.csv"
   ```
6. all of the above as tests, on the restricted folder (`--real-dir`; the test that compares `package_rules.py` with
   the development-only `export_m1.py` is skipped in this package):

   ```bash
   python scripts/check_nuaa_manifests.py --require-real --real-dir "$M2_RUN" --real-aggregate ../nuaa_manifests_aggregate.json --c1-v1 "$C1_RUN/nuaa_manifest.csv"
   ```

## Integrity

Completeness of a folder (this one, or the restricted run folder): every listed file matches its SHA-256 and no file
is missing from, or added to, the list. From inside the folder:

```bash
shasum -a 256 -c SHA256SUMS && test -z "$(find . -type l)" && diff <(awk '{print $2}' SHA256SUMS | sed 's|^\*||' | sort) <(find . -type f ! -type l ! -name SHA256SUMS | sed 's|^\./||' | sort) && echo COMPLETE
```

The last line printed must be `COMPLETE`. Run it on the folders as delivered, before the commands above (they may write
caches such as `__pycache__/`). The restricted run folder anchors this one: its file `public_SHA256SUMS.sha256` records
the SHA-256, and its `README.md` the SHA-256 and the file count, of this folder's `SHA256SUMS`. From inside this
folder:

```bash
shasum -a 256 -c "$M2_RUN/public_SHA256SUMS.sha256" && echo MATCH || echo MISMATCH
```

The last line printed must be `MATCH`. `run-report.md` records the number of files of the restricted folder and the
SHA-256 of its main files, not the SHA-256 of its `SHA256SUMS` (that file lists the anchor, so the record would be
circular).

## Overlap and derivation rules

Every manifest is declared in `nuaa_overlap_rules.json` with its expected number of rows and SHA-256: a missing,
truncated or different manifest stops `check-overlap` with code 1 before any rule. Every row is validated against its
key: label from the `real/`/`attack/` prefix, subject, glasses, pos, session and picture number from the file name,
official split from the session, compared as exact text (a row with a lost leading zero, a space or an empty field is
refused, so no rule, not even a report-only one, runs on inconsistent data). Every pair of manifests is compared by
key, content hash, subject and (subject, session), and each manifest is checked for internal duplicates by key and by
hash. A rule on an empty manifest fails instead of passing vacuously. Duplicates are exact (file SHA-256, or frame
array bytes); near-duplicates (consecutive frames, the same printed photograph) are not searched. This tranche does
not establish absence of leakage.

| rule | relation | severity |
|---|---|---|
| R01 | no internal duplicates by key or hash in any manifest | must |
| R02 | CNN train and validation (`cnn_es`) disjoint by key, hash and subject | must |
| R03 | CNN train and M1 dev disjoint by key, hash and subject | must |
| R04 | training side (official train, CNN train, validation, dev) and test side (official test, CNN test, C1) disjoint by key, hash and (subject, session); the (subject, session) part holds by construction of the NUAA protocol (sessions 01–02 vs 03) and is kept for the other datasets | must |
| R05 | the same pairs by subject | report only (known limit) |
| R06 | CNN train, validation and dev contained in the official train split | must |
| R07, R08 | validation contained in the dev; dev ∩ validation = 493 images (declared reuse) | must |
| R09, R10 | C1 ⊆ CNN test ⊆ official test | must |
| R11 | CNN train and validation disjoint, and together equal to `spread(official train, 1500)` per class | must |
| R12 | M1 dev = all official training images of the validation subjects | must |
| R13 | CNN test = `spread(official test, 1500)` per class | must |
| R14 | C1 = `spread(official test, 150)` per class | must |

`check-overlap` exits with code 1 when a must-hold rule fails; it lists up to three example keys for the failed checks
in the restricted report only; the public report has counts and verdicts only and is scanned for forbidden content
(subject identifiers, paths, addresses, credentials). `check-overlap --self-test` then repeats the check on seven
faulty copies of the real manifests, with their rows and SHA-256 re-declared, and requires each to fail exactly the
expected rules: a dev row removed (R12), a train row removed (R11), a validation row added to the train (R02, R03,
R11), a train row added to C1 (R04, R09, R10, R14), a train image's SHA-256 under a test name (R04, R09), a duplicated
test row (R01, R13), a C1 row removed from the official test (R09, R13, R14). The offline tests cover the same rules on
synthetic manifests with known overlaps, plus inconsistent rows, truncated or altered manifests, byte-order marks,
duplicated columns, malformed or vacuous rules, the public report and duplicated derived frames.

## Supported comparisons (NUAA)

- Supported: a descriptive comparison of the four official CNNs trained on the same split, on the official protocol (disjoint by session, not by subject), single seed, without an uncertainty estimate; per-class error rates also reflect the class-dependent share of unseen subjects (below).
- Not supported: generalisation to subjects never seen in training; the M1 calibration is exploratory (validation
  images reused, two subjects); C1 is a subset of the test images. The NUAA pooled checkpoints are not covered (they are excluded from the C1 NUAA table because their random split shares subjects and sessions with the test images).
- In the test sets the share of images from subjects absent from the official train differs by class (official test:
  1,987 of 3,362 bona fide and 2,216 of 5,761 attacks; values for every test manifest in `report.md`): the class is
  confounded with the novelty of the subject.

## Limits

- NUAA is not subject-disjoint between the official train and test splits: the training subjects reappear in the test
  split in a different capture session. Separation from the test set holds by image, content hash and (subject,
  session), not by subject; the shared subjects are reported (rule R05).
- The M1 dev set is not independent of the training run: 493 of its 574 images are the validation images of the CNNs
  (see Glossary).
- 22 C1 test images (attacks, session 03) come from the two validation subjects of the CNN training.
- The training run did not save its file list: it is rebuilt from the sorted directory listing and verified against
  the card counts and the training cache arrays.
- The training cache (the derived artefact of the images) holds preprocessed arrays and subject prefixes, not paths
  or hashes: it is linked by group and label arrays in sampling order, not image by image; its frames are checked for
  duplicates by hash.
- The README of the original release is not in the mirror and was not available to us (see "Source and field names");
  the split lists are the mirror's copy.
- The cards record the SHA-256 of the `.h5` only. For two of the four (`LivenessNet__nuaa`, `AttackNetV2_2__nuaa`) the
  `.h5` was exported from the `.keras` and found identical tensor by tensor (C1 `checkpoints/PROVENANCE.md`); the
  `.keras` files are not used.
- NUAA has no video identifiers: frames are grouped by subject and session only, and consecutive frames of a session
  are not independent samples.
