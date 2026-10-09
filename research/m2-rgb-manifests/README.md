# M2, second tranche: CASIA-FASD and CelebA-Spoof manifests and split overlap check

Task **M2** of the PAD research workflow, second tranche: versioned per-image manifests of **CASIA-FASD** and
**CelebA-Spoof**, built on the local copies already available (no new transfer), linked to their source, classes,
transformations and checkpoints, and a repeatable check of overlaps and duplicates between the splits. Same structure
as the accepted NUAA tranche (`research/m2-nuaa-manifests/`), whose code is imported, not copied. No training and no
inference: the scripts list and hash files and read the training caches (label, group and frame arrays; the frames are
only hashed).

This folder is a filtered export of the operator's development repository (private; development commit `80b7d14`,
run `20261009-M2-rgb-seed42-80b7d14`): the scripts needed to rerun the checks, the documents, the declared rules and the aggregate results,
without data, weights, per-image manifests, subject or video keys, or row indices. The per-image manifests, the private
summaries, the restricted overlap and short reports, the self-tests, the dataset register drafts and the logs are in the
restricted run folder `20261009-M2-rgb-seed42-80b7d14` (see "Delivery status"), with their SHA-256 listed in the public aggregates and in
`run-report.md`. Code comments, command-line help and progress messages are in Italian by the developer's convention;
reports, error messages and every document of this folder are in English.

Licence documents: none in our local copies (the mirror cards state no licence and no source). Authorisations are
handled by the coordinator (task D1): for CelebA-Spoof he received on 8 October 2026 the confirmation of the requested
use conditions; for CASIA-FASD the agreement arrived on 9 October 2026 and is to be signed through the University.
This package does not itself rely on or certify any authorisation; the mirror copies' provenance is documented
separately.

## Delivery status

The restricted folder of this run (`20261009-M2-rgb-seed42-80b7d14`) is kept locally and not uploaded. The provenance note `M2_provenienza_RGB_20261009.pdf` was placed on 9 October 2026 in the shared M2 folder `20261006-M2-seed42-36c677a` as a supplementary document outside the NUAA run (not listed in its `SHA256SUMS`), as indicated in the coordinator's message of 9 October 2026 (email and M2 card); the restricted RGB run folder stays local and is uploaded to `02_Experiments/M2/20261009-M2-rgb-seed42-80b7d14/` only when he asks for it. No earlier upload of a CASIA-FASD or CelebA-Spoof restricted folder is declared.

## Glossary

- **official** (checkpoint protocol): CASIA-FASD fitting and early stopping inside `train_img`, test on `test_img`;
  CelebA-Spoof fitting, early stopping and test **all inside shard 0 of the mirror's `test` split** (correspondence
  with the official test split not verified), divided by us per image: for CelebA-Spoof "official" is only the protocol
  name recorded in the cards, not the original CelebA-Spoof protocol.
- **pooled**: training and test arrays of the cache concatenated and split at random (per frame or per image).
- **early stopping** (`cnn_es`, `pooled_es`): images never used for gradient updates; their `val_loss` controlled early
  stopping and the learning-rate reduction.
- **prepared `_validation` folders**: they hold the test side (CASIA-FASD `test_img`; CelebA-Spoof the 900 test images);
  they were never used as validation data.
- **eval**: the 300 images of the application evaluation set (`data/eval/<dataset>`), frozen: only compared.

## Source and field names

### CASIA-FASD

For CASIA-FASD, the protocol (20 training and 30 test subjects, disjoint; 12 videos per subject) is cited from the
reference publication (ICB 2012) [DOC-EXT]; what is verified on our copy is listed separately and does not include the
identity of the subjects.

- Original source: CASIA Face Anti-Spoofing Database (CASIA-FASD), Institute of Automation, Chinese Academy of
  Sciences; reference publication: Z. Zhang, J. Yan, S. Liu, Z. Lei, D. Yi, S. Z. Li, "A face antispoofing database
  with diverse attacks", ICB 2012.
- Documented [DOC-EXT]: cited from the reference publication (ICB 2012) as the protocol source, not verified on the
  copy: 50 subjects, 20 in the training set and 30 in the test set, disjoint by protocol; 12 videos per subject
  (genuine, warped photo, cut photo and video replay, each at low, normal and high quality); the mapping of the video
  tokens to attack type and quality is not in the mirror and is not applied: the manifests carry the token as an
  opaque code.
- Verified on the copy: the downloaded archive holds only extracted JPEG frames, without videos, README or split lists
  (the mirror also holds two file-name/label lists, `train/` and `test/`, which were not downloaded): `train_img`
  1,655 and `test_img` 2,408 colour frames named `<subject>_<video>.avi_<frame>_<real|fake>.jpg`; subject numbers 1–20
  in `train_img` and 1–30 in `test_img` (the numbering is local to the split: the same numbers occur in both); 12
  video tokens per subject, none missing; tokens 1, 2 and HR_1 always `real`, the others always `fake`; in the archive
  the frames are dated 15 November 2022 and the folders 4 June 2025 (the same day as the mirror upload), so the frames
  predate the mirror.
- Mirror: verified on 9 October 2026 (Hugging Face API): revision `4ea8ec45dee3765e17bb3efedfdcd7af39a3ebfe` is still
  the mirror's HEAD and the LFS object of `casiafasd.tar.gz` equals our SHA-256 (69,265,514 bytes); the mirror card
  states no licence and no source. Correspondence with the official release is not verified.
- Not verified: that the subjects of `train_img` and of `test_img` are different persons; the attack type and quality
  of each token; who extracted and cropped the frames, and from which copy; the origin of the `depth/` maps (not
  used); the correspondence of the copy with the official release.
- Subjects and videos: a subject is keyed `<mirror split>:<number>` and a video `<mirror split>:<number>:<token>`; a
  number alone is never compared across splits as an identity. No identity is inferred from image content.

### CelebA-Spoof

For CelebA-Spoof, "official" is only the protocol name recorded in the checkpoint cards: the split is a per-image
split made by us inside shard 0 of the mirror's `test` split, not the original CelebA-Spoof protocol, and identity
overlap between fitting, early stopping and test can be neither excluded nor measured.

- Original source: CelebA-Spoof; reference publication: Y. Zhang, Z. Yin, Y. Li, G. Yin, J. Yan, J. Shao, Z. Liu,
  "CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset with Rich Annotations", ECCV 2020; official repository
  `ZhangYuanhan-AI/CelebA-Spoof`.
- Documented [DOC-EXT]: the original release organises the images by identity, with official training and test splits
  and annotations of attack type, illumination, environment, sensor and face attributes; its README (official
  repository) states that the live images come from CelebA.
- Verified on the copy: shard 0 of 10 of the mirror's `test` split: one parquet file with the columns `cropped_image`
  (`bytes`, `path`), `labels` (0 live, 1 spoof) and `labelNames`; 6,717 rows (2,030 live, 4,687 spoof), 39 of them
  without image bytes; `path` empty in every row; every image is a PNG; the prepared files are byte-identical to the
  PNG bytes of their rows (with a `.jpg` extension).
- Mirror: verified on 9 October 2026 (Hugging Face API): revision `f158d9f71d786089d4f8b3b6c16b7bcae5989a3f` is still
  the mirror's HEAD and the LFS object of `data/test-00000-of-00010.parquet` equals our SHA-256 (495,744,415 bytes);
  the mirror card states no licence and no source, and names the split `test` (67,170 examples). Correspondence of the
  mirror's `test` split with the official test split is not verified.
- Not verified: the identity, attack type and acquisition conditions of any row; the order of the rows; how the faces
  were cropped; how the shard relates to the official test split; the procedure that converted the parquet into the
  prepared folders (a rule that reproduces them exactly is checked; the procedure itself is not versioned).
- Identities: none: the copy has no identity, subject or video field, so no split can be checked for identity overlap;
  no identity is inferred from row order, content or similarity.

Details, keys, closed value sets and checks in `naming_schema_rgb.md`; preprocessing in `transformations_rgb.md`.

## What is delivered

| file | what it is | where |
|---|---|---|
| `naming_schema_rgb.md` | names, keys and fields of both datasets (documented [DOC-EXT], verified on the copy, not verified), closed value sets, conversion rule of CelebA-Spoof | public |
| `transformations_rgb.md` | from the image files to the derived frames and the CNN inputs; splits of the checkpoints | public |
| `casia_fasd_overlap_rules.json`, `celeba_spoof_overlap_rules.json` | declared manifests (file, expected rows, expected SHA-256) and rules (must-hold, report-only, not checkable) | public |
| `<dataset>_manifests_aggregate.json` | source and revision, SHA-256 of the source file, per-manifest SHA-256 and counts, checkpoint linkage, reconstruction, derived-frame duplicate check, recountable block | public |
| `<dataset>_overlap_report_public.md`, `.json` | pairwise overlap matrix, rule verdicts; counts only | public |
| `<dataset>_report.md` | short report: source, counts, checkpoints, rules, self-test, supported comparisons, limits | public version without identifiers |
| `run-report.md` | run report (also in the restricted folder) | public |
| `livenesslab/` | `scripts/rgb_manifests.py` (builder and checker), `scripts/check_rgb_manifests.py` (offline tests), the NUAA modules it imports (`nuaa_manifests.py`, `select_threshold.py`, `package_rules.py`), `src/tesi_app/paths.py`, `delivery/m2/rgb/` (rules and report template read by the scripts), `scripts/m2_inventory/` (the six scripts that regenerate the numbers of the published inventory), `requirements.txt` | public (code) |
| `SHA256SUMS` | SHA-256 of every other file of this folder, `livenesslab/` included | public |
| `<dataset>/manifests/*.csv` | per-image manifests (below) | restricted |
| `<dataset>/<dataset>_manifests_summary.json`, private reports | as the aggregate, plus subject keys of the early-stopping subjects (CASIA-FASD), row indices of the excluded and duplicated rows (CelebA-Spoof) | restricted |
| `<dataset>/overlap_report.*`, `overlap_self_test.*` | restricted overlap report (example keys of failed checks) and self-test | restricted |
| `dataset_register_casia_fasd.md`, `dataset_register_celeba_spoof.md` | draft entries of the dataset register (`01_Dataset_Register`) | restricted |

Fields that change at every run and are not part of the evidence: `created`, `commit`, `dirty`, `checked`,
`manifests_dir`, `library_versions`.

## Manifests

### CASIA-FASD

Columns: `key` (`<mirror split>/<file name>`), `label`, `mirror_split`, `subject`, `subject_key`, `video`,
`video_key`, `frame`, `sha256`, `bytes`.

| name | content | images | subjects | videos |
|---|---|---|---|---|
| `mirror_train` | all colour frames of `train_img` | 1,655 (404 + 1,251) | 20 | 240 |
| `mirror_test` | all colour frames of `test_img` | 2,408 (591 + 1,817) | 30 | 360 |
| `prepared_training`, `prepared_validation` | the prepared folders (same rows as the mirror, hard links) | 1,655, 2,408 | | |
| `eval` | application evaluation set: `spread(test_img, 150)` per class | 300 (150 + 150) | 30 | 239 |
| `cnn_train` | fitted the four official CNNs | 1,339 (333 + 1,006) | 16 | 192 |
| `cnn_es` | early stopping of the four official CNNs | 316 (71 + 245) | 4 | 48 |
| `cnn_test` | test of the official cards: `spread(test_img, 1500)` per class | 2,091 (591 + 1,500) | 30 | 360 |
| `pooled_train`, `pooled_es`, `pooled_test` | the two pooled CNNs | 2,696, 300, 750 | 50, 50, 50 | 598, 240, 445 |

### CelebA-Spoof

For CelebA-Spoof, "official" is only the protocol name recorded in the checkpoint cards: the split is a per-image
split made by us inside shard 0 of the mirror's `test` split, not the original CelebA-Spoof protocol, and identity
overlap between fitting, early stopping and test can be neither excluded nor measured.

Columns: `key` (`celeba_<row>`), `label`, `label_name`, `row`, `official_split` (always `test`), `sha256`, `bytes`.

| name | content | images (bona fide + attack) |
|---|---|---|
| `parquet` | all rows of shard 0 with image bytes (39 rows without image excluded) | 6,678 (2,021 + 4,657) |
| `prepared_training` | prepared `_training` folders | 3,000 (1,500 + 1,500) |
| `prepared_validation` | prepared `_validation` folders (includes `eval`) | 900 (450 + 450) |
| `eval` | application evaluation set | 300 (150 + 150) |
| `cnn_train`, `cnn_es` | fitting and early stopping of the four "official" CNNs | 2,400 (1,205 + 1,195), 600 (295 + 305) |
| `cnn_test` | test of the "official" cards (= `prepared_validation`) | 900 (450 + 450) |
| `pooled_train`, `pooled_es`, `pooled_test` | the two pooled CNNs | 2,808, 312, 780 |

Each checkpoint (CASIA-FASD and CelebA-Spoof: `LivenessNet`, `AttackNetV1`, `AttackNetV2_1`, `AttackNetV2_2` official;
`LivenessNet`, `AttackNetV2_2` pooled) is linked by the SHA-256 of its `.h5`, read from the card, equal to
`models/weights/manifest.json` and recomputed on the file; its card counts n_train / n_val / n_test and test counts per
class equal the rows of the manifests. The splits are rebuilt with the logic of commit `0ac7143` (first versioned
version of `scripts/train_cnn.py`; the training run used an earlier, unversioned version) and verified against the
cards and against the label and group arrays of the training caches. The training run did not record its file list.

## Commands

Dependencies: Python 3.11; numpy 1.26.4 for every subcommand and for the tests; scikit-learn 1.3.2 for `build`,
`reconstruct` and the tests; pyarrow for the CelebA-Spoof `build` only. `build` also needs the copies under `data/`
(mirror archive and extracted frames, parquet, prepared folders, evaluation folders, training caches) and the
checkpoint cards and weights in `models/weights/`. From inside `livenesslab/` (or the root of the development
repository):

```bash
python scripts/rgb_manifests.py build --dataset casia_fasd                       # manifests, private summary, public aggregate (exit 1 if a check fails)
python scripts/rgb_manifests.py check-overlap --dataset casia_fasd --self-test   # declared rules and self-test (exit 1 on any failure)
python scripts/rgb_manifests.py report --dataset casia_fasd                      # short report, private and public versions
python scripts/check_rgb_manifests.py                                            # offline tests (real-data tests skipped without data)
```

`python scripts/m2_inventory/run_all.py` regenerates the 71 numbers of the published inventory (CASIA-FASD and
CelebA-Spoof, 8 October 2026), printing MATCH or DIFF for each; it needs the local dataset copies in `data/` and the
checkpoint cards and weights in `models/weights/` next to `scripts/` (it does not read the `LIVENESSLAB_*` variables;
without the data it stops with a message and exit code 2). Its progress messages are in Italian. The `val_split`
history check reads two commits of the development repository and gives a DIFF outside it; the hard-link and file-time
checks give a DIFF on a copy that does not preserve them. `--private` adds subject and row identifiers: restricted use only.

The same with `--dataset celeba_spoof`. Outputs go to `results/m2_rgb/<dataset>/` (`--out-root` or `--out` to write
elsewhere); data, results and weights folders can be moved with `LIVENESSLAB_DATA_DIR`, `LIVENESSLAB_RESULTS_DIR` and
`LIVENESSLAB_WEIGHTS_DIR`.

## Verification without images or weights

With `M2_RUN` pointing at the restricted run folder, from inside `livenesslab/`. Without the images, content hashes,
byte sizes and labels are taken as recorded: a hash or a `bytes` value changed consistently in every manifest where the
key appears passes undetected; a CelebA-Spoof label changed consistently in every manifest is usually caught by S03 or
S14, but not guaranteed; only `build` checks them against the files. Split membership: the declared counts (C18, S14)
catch a subject or a row moved between fitting and early stopping without scikit-learn, but a swap of rows of the same
class between two manifests (for example fitting and early stopping of CelebA-Spoof) keeps every count: the membership
of the fitting, early-stopping and pooled manifests is pinned only by their declared SHA-256 and by `reconstruct`.

1. completeness and anchor of the two folders (section "Integrity");
2. rules and self-test on the restricted manifests (numpy only); the verdicts must equal those of the public overlap
   reports and the self-test must reject 10 of 10 faulty copies per dataset:

   ```bash
   python scripts/rgb_manifests.py check-overlap --dataset casia_fasd --self-test --manifests-dir "$M2_RUN/casia_fasd/manifests" --out /tmp/m2-casia
   python scripts/rgb_manifests.py check-overlap --dataset celeba_spoof --self-test --manifests-dir "$M2_RUN/celeba_spoof/manifests" --out /tmp/m2-celeba
   ```
3. counts of the public aggregate recomputed from the manifests alone (exit 1 if they differ):

   ```bash
   python scripts/rgb_manifests.py recount --dataset casia_fasd --manifests-dir "$M2_RUN/casia_fasd/manifests" --aggregate ../casia_fasd_manifests_aggregate.json
   ```
4. official and pooled splits rebuilt from the manifests alone (scikit-learn, no image read) and compared with the
   declared ones; with a different scikit-learn version, a difference is reported as a possible drift of the library
   (the command reports the possibility; it does not establish it):

   ```bash
   python scripts/rgb_manifests.py reconstruct --dataset casia_fasd --manifests-dir "$M2_RUN/casia_fasd/manifests"
   ```
5. all of the above as tests (`--real-dir` is the folder that contains `casia_fasd/` and `celeba_spoof/`; the public
   aggregates are read from this folder, one level above `livenesslab/`); the test
   `test_library_drift_canary` fails with an explicit message if scikit-learn splits a fixed input differently from
   scikit-learn 1.3.2. A test skips, with its reason printed by `--verbose`, only when what it needs is absent: the tests on
   the real manifests without `--require-real` and the restricted folder, and the two document checks (comparison of the
   source declarations, and check of the CelebA-Spoof, CASIA-FASD and licence notes) when `README.md` and
   `naming_schema_rgb.md` are not next to `livenesslab/`. Run from this folder with the command
   below, no test should skip:

   ```bash
   python scripts/check_rgb_manifests.py --require-real --real-dir "$M2_RUN"
   ```

## Integrity

From inside a folder (this one, or the restricted run folder):

```bash
shasum -a 256 -c SHA256SUMS && test -z "$(find . -type l)" && diff <(awk '{print $2}' SHA256SUMS | sed 's|^\*||' | sort) <(find . -type f ! -type l ! -name SHA256SUMS | sed 's|^\./||' | sort) && echo COMPLETE
```

The last line printed must be `COMPLETE`. The restricted run folder anchors this one: its file
`public_SHA256SUMS.sha256` records the SHA-256 of this folder's `SHA256SUMS`. From inside this folder:

```bash
shasum -a 256 -c "$M2_RUN/public_SHA256SUMS.sha256" && echo MATCH || echo MISMATCH
```

## Overlap rules

The rule files are locked by the offline tests (`test_versioned_rules_are_locked`): expected rows, the table of rule
ids, kinds, severities and dimensions, and a SHA-256 of the normative content. Rule descriptions and the reasons of the
declarations are excluded from that digest, so a change of their wording only is not detected by the tests.

Every manifest is declared with its expected number of rows and SHA-256: a missing, truncated or different manifest
stops `check-overlap` before any rule; a manifest must also be byte-identical to the canonical form written by `build`
(LF line endings, no byte-order mark, rows sorted by key). Every row is then validated against its key and against closed value sets (no
fallback: a value outside the set, an empty field, a space, a leading zero or an extra column is refused, and no rule
runs). A rule on an empty manifest fails instead of passing vacuously. In disjoint rules every manifest of one side is
compared with every manifest of the other; in subset rules a manifest with the union of the other side; in
intersection rules the two unions, with the declared number of shared values. Duplicates are exact (file SHA-256, or
frame array bytes); near-duplicates (consecutive frames of a video, the same face in another capture) are not
searched. This tranche does not establish absence of leakage.

### CASIA-FASD (`casia_fasd_overlap_rules.json`)

| rule | relation | severity |
|---|---|---|
| C01 | no internal duplicates by key or hash | must |
| C02 | `train_img` and `test_img` disjoint by key and hash | must |
| C03 | identity disjointness of `train_img` and `test_img` (protocol, [DOC-EXT]) | not checkable (declared) |
| C04 | subject numbers alone shared by the two splits (local numbering, not identity) | report only |
| C05, C06 | prepared folders = mirror splits | must |
| C07 | official fitting + early stopping = `spread(train_img, 1500)` per class, disjoint | must |
| C08 | official fitting and early stopping disjoint by key, hash, subject and video | must |
| C09 | training side and test side disjoint by key and hash (by subject key and video key: by construction, the keys include the mirror split) | must |
| C10, C11 | test of the official checkpoints = `spread(test_img, 1500)`, eval = `spread(test_img, 150)` per class | must |
| C12 | eval ⊆ test of the official checkpoints | must |
| C13, C14 | pooled fitting + early stopping + test = the cache frames, disjoint by key and hash | must |
| C15 | declared leak: 235 of the 300 eval images in the pooled fitting or early stopping | must (declared count) |
| C16 | declared leak: 50 of 50 subjects and 444 of 600 videos on both pooled sides | must (declared count) |
| C17 | key consistency: same label, split, subject, video, frame, SHA-256 and bytes wherever a key appears | must |
| C18 | declared counts of every manifest (rows, bona fide, attacks, subjects, videos), without scikit-learn | must (declared counts) |

Self-test (10 faulty copies, written in canonical form with rows and SHA-256 re-declared, so that neither the format
check nor the checksum stops them): early-stopping row added to the fitting images (C07, C08, C18); two SHA-256 swapped
in the test of the official checkpoints (C17 only); mirror split `dev` (row validation); label inconsistent with the key (row validation);
duplicated pooled test row (C01, C13, C18); a test image's SHA-256 under a training key (C09, C17); pooled test row
removed (C13, C18); evaluation image replaced (C11); prepared test row removed (C06, C18); a whole fitting subject
moved to the early stopping (C18 only).

### CelebA-Spoof (`celeba_spoof_overlap_rules.json`)

For CelebA-Spoof, "official" is only the protocol name recorded in the checkpoint cards: the split is a per-image
split made by us inside shard 0 of the mirror's `test` split, not the original CelebA-Spoof protocol, and identity
overlap between fitting, early stopping and test can be neither excluded nor measured.

| rule | relation | severity |
|---|---|---|
| S01 | no internal duplicates by key | must |
| S02 | declared exact duplicates by hash: one pair in `parquet`, `prepared_training`, `cnn_train`, `pooled_train`; none elsewhere | must (declared count) |
| S03 | prepared folders and eval reproduced by the conversion rule | must |
| S04 | eval ⊆ prepared `_validation` | must |
| S05 | "official" fitting + early stopping = prepared `_training`, disjoint | must |
| S06, S07 | fitting vs early stopping; training side vs test side: disjoint by key and hash | must |
| S08 | identity disjointness of the "official" split | not checkable (declared) |
| S09 | every manifest ⊆ the parquet rows of shard 0 of the mirror's `test` split | must |
| S10 | pooled fitting + early stopping + test = the cache images, disjoint | must |
| S11 | "official" test = prepared `_validation` | must |
| S12 | declared leak: 244 of the 300 eval images in the pooled fitting or early stopping | must (declared count) |
| S13 | key consistency: same label, label name, row, split, SHA-256 and bytes wherever a key appears | must |
| S14 | declared counts of every manifest (rows, bona fide, attacks), without scikit-learn | must (declared counts) |

Self-test (10 faulty copies): early-stopping row added to the fitting images (S05, S06, S14); two SHA-256 swapped in
the test (S13 only); `label_name` `unknown` (row validation); `official_split` `train` (row validation); duplicated
pooled test row (S01, S02, S10, S14); a test image's SHA-256 under a training key (S07, S13); evaluation image replaced
(S03); label and label name flipped together in one row (S13, S14); prepared training row removed (S03, S05, S14); a
fitting row moved to the early stopping (S14 only).

## Supported comparisons

### CASIA-FASD

- Supported: a descriptive comparison of the four official CNNs trained on the same split (fitting and early stopping
  subject-disjoint: verified on the reconstructed split, recorded as 'by subject' in the cards; test on `test_img`,
  subject-disjoint by protocol, documented), single seed, without an uncertainty estimate.
- Not supported: any claim on the pooled checkpoints evaluated on the evaluation set or on `test_img` (their training
  includes evaluation images, subjects and videos of the test side); a claim of subject disjointness between
  `train_img` and `test_img` as verified.

### CelebA-Spoof

For CelebA-Spoof, "official" is only the protocol name recorded in the checkpoint cards: the split is a per-image
split made by us inside shard 0 of the mirror's `test` split, not the original CelebA-Spoof protocol, and identity
overlap between fitting, early stopping and test can be neither excluded nor measured.

- Supported: a descriptive, within-shard comparison of the four "official" CNNs (card protocol name) on the same
  per-image split made by us, declared as not guaranteed to be identity-disjoint; CelebA-Spoof as a target dataset for
  models trained elsewhere, declared as shard 0 of the mirror's `test` split, without identities.
- Not supported: conclusions on unseen subjects; comparisons with results on the official CelebA-Spoof protocol; any
  claim on the pooled checkpoints evaluated on the evaluation set.

## Limits

### CASIA-FASD

- The downloaded archive has only extracted frames: no original video, README or split list (the mirror's two
  file-name/label lists were not downloaded). That `train_img` and `test_img` hold different persons is documented by
  the protocol [DOC-EXT] and cannot be verified on the copy, because subject numbers are local to the split.
- Frames of the same video are not independent samples; every split of this tranche keeps the frames of a video
  together except the pooled one.
- The pooled checkpoints use a per-frame random split: all 50 subjects and 444 of the 600 videos are on both sides,
  and 235 of the 300 evaluation images were used to fit or validate them.
- The training run did not save its file list: the splits are rebuilt with the logic of commit `0ac7143` and verified
  against the card counts and the label and subject arrays of the training cache; the card counts alone do not
  determine the four held-out subjects (75 combinations of 4 of the 20 training subjects give 316 images), so the
  official split relies on `GroupShuffleSplit` with seed 42. The training cache is linked to the files by sampling
  order, not image by image.
- 317 test attack frames are not in the test of the official checkpoints (spread(..., 1500) per class); the `depth/`
  maps are not used.

### CelebA-Spoof

- The "official" checkpoints (protocol name recorded in the cards) are not trained on the original CelebA-Spoof
  protocol: fitting, early stopping and test images all come from shard 0 of the mirror's `test` split (correspondence
  with the official test split not verified), divided by us per image.
- No identity is available: fitting, early-stopping and test images are not guaranteed to be identity-disjoint
  (overlap can neither be excluded nor measured; declared as not checkable); test results say nothing about unseen
  subjects.
- The early-stopping split is rebuilt with the first versioned logic (`GroupShuffleSplit` on file names, one image per
  group); a stratified split gives the same totals 2,400 / 600 (with 300 / 300 per class in the early stopping instead
  of 295 / 305). Which of the two the training run used is not verified. The `val_split` field of the four "official"
  cards was 'by subject' in commit `0ac7143` and was rewritten as 'random per image' in commit `4d11e67`; with the
  logic of `0ac7143` the first value corresponds to `GroupShuffleSplit` on groups of one image.
- The procedure that converted the parquet into the prepared folders is not versioned; a rule that reproduces their
  row sets exactly is checked. 39 rows without image bytes are excluded; 71 live and 2,707 spoof rows are not used.
- One pair of rows has byte-identical images (exact duplicate, both bona fide); near-duplicates are not searched.
- The pooled checkpoints use a random split of the same images: 244 of the 300 evaluation images were used to fit or
  validate them.

### Both

- Licence documents and authorisations: see the paragraph at the top of this README; for CelebA-Spoof a
  personal-data assessment is also needed (faces of real, identifiable persons; the README of the official CelebA-Spoof
  repository states that the live images come from CelebA).
- Both copies are unofficial mirrors. Verified on 9 October 2026 (Hugging Face API): each revision is still the
  mirror's HEAD and its LFS object equals our SHA-256; the mirror cards state no licence and no source. Correspondence
  with the official releases is not verified.
- The `.keras` files next to the `.h5` are not used by the application and are not hashed here.
- No new split is made in this tranche; a new CASIA-FASD split would assign whole subjects (all 12 videos) before any
  frame selection or augmentation; CelebA-Spoof cannot be split by identity.
