# CASIA-FASD and CelebA-Spoof: naming schema and manifest fields

How the file names of the local copies of CASIA-FASD and CelebA-Spoof are parsed into the fields of the M2
manifests, where each piece of the schema comes from, and what was checked on the copy. Labels used throughout:
**[DOC-EXT]** documented by an external source (cited, not archived in the repository and not in the mirror);
**verified** checked on our copy by `scripts/rgb_manifests.py build`; **not verified** not checkable with the
material available. The examples use fictional values (CASIA frame indices that are not multiples of 25, CelebA rows
outside the shard), so that no real image is named.

## CASIA-FASD

### Source

| item | value | status |
|---|---|---|
| Copy used | Hugging Face mirror `akahana/anti-spoofing-casiafasd` (unofficial) | verified (cache metadata) |
| Mirror revision | `4ea8ec45dee3765e17bb3efedfdcd7af39a3ebfe` | verified |
| Archive | `casiafasd.tar.gz`, 69,265,514 bytes, SHA-256 `76139d96360767adfecc1a8c01c062af14da664f665ba13ba5ce78111d0d9ec9`, equal to the etag recorded by the Hugging Face download cache (12 September 2026) | verified |
| Content of the archive | `train_img/train_img/{color,depth}` 1,655 + 1,655 files, `test_img/test_img/{color,depth}` 2,408 + 2,408; no video, README or split list | verified (listed with `tarfile`, not extracted) |
| Download URL | `https://huggingface.co/datasets/akahana/anti-spoofing-casiafasd/resolve/4ea8ec45dee3765e17bb3efedfdcd7af39a3ebfe/casiafasd.tar.gz` | verified (revision and file name of the cache) |
| Download | 12 September 2026, 11:39:21 Europe/Rome, with `huggingface_hub` 1.31.0 (`hf_xet` 1.6.0), `local_dir` `data/raw`; the exact call was not recorded | verified (cache metadata and download log) |
| Mirror today | on 9 October 2026 the revision is still the mirror's HEAD (last modified 4 June 2025) and the LFS object of `casiafasd.tar.gz` at that revision equals our SHA-256 and size; the mirror card states no licence and no source | verified (Hugging Face API, 9 October 2026) |
| File dates in the archive | the 8 folders are dated 4 June 2025 02:31 UTC (the same day as the mirror upload), all 8,126 JPEG files 15 November 2022 01:24 UTC: the frames predate the mirror; who extracted them, and from which copy, is not verified | verified (archive listing) |
| Other files of the mirror | two file-name/label lists (`train/`, `test/`), not downloaded: "no split list" refers to the downloaded archive | verified (Hugging Face API) |

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

### Names and keys

| where | form | example (fictional) |
|---|---|---|
| mirror file (colour frame) | `{train_img,test_img}/…/color/<subject>_<video>.avi_<frame>_<real|fake>.jpg` | `test_img/test_img/color/9_HR_4.avi_620_fake.jpg` |
| prepared file | `{bonafide,attack}_{training,validation}/<same name>` (hard link to the mirror file) | `attack_validation/9_HR_4.avi_620_fake.jpg` |
| evaluation file | `data/eval/casia_fasd/{real,attack}/<same name>` (hard link) | `attack/9_HR_4.avi_620_fake.jpg` |
| manifest key | `<mirror split>/<name>`, mirror split `train` (`train_img`) or `test` (`test_img`) | `test/9_HR_4.avi_620_fake.jpg` |

The mirror split is part of the key because the same names occur in `train_img` and `test_img` (the subject numbering
is local to the split). The class is in the name (`_real`, `_fake`), so it is not part of the key. Prepared
`_training` folders hold `train_img`, prepared `_validation` folders hold `test_img`: they were never used as
validation data.

### Fields

| column | meaning | parsing and closed set |
|---|---|---|
| `key` | mirror split and file name | `^(train|test)/([1-9]\d?)_((?:HR_)?[1-8])\.avi_(0|[1-9]\d*)_(real|fake)\.jpg$` |
| `label` | 0 bona fide (`_real`), 1 attack (`_fake`) | from the class suffix |
| `mirror_split` | `train` or `test` | first part of the key |
| `subject` | subject number, local to the split | 1–20 in `train`, 1–30 in `test`, no leading zero |
| `subject_key` | `<mirror split>:<subject>` | the key used by every subject comparison |
| `video` | video token, opaque code | one of `1`–`8`, `HR_1`–`HR_4`; tokens `1`, `2`, `HR_1` must carry `_real`, the others `_fake` (verified on the copy) |
| `video_key` | `<mirror split>:<subject>:<video>` | the key used by every video comparison |
| `frame` | frame index within the video | integer without leading zeros |
| `sha256`, `bytes` | content hash and size of the file | computed on the file; bytes a positive integer |

Every field must equal the value derived from the key, as exact text; a value outside a closed set (for example a
mirror split `dev`, subject 31 or token `HR_5`), an empty field, a space or a leading zero stops `check-overlap` before
any rule. There is no attack-type or quality column: the mapping of the tokens is in the reference publication
[DOC-EXT] and is not applied.

### Checks of `build`

1. the archive SHA-256 equals the etag of the download cache; the archive is listed (not extracted);
2. every colour frame name of the mirror matches the pattern above; subjects are exactly 1–20 (`train_img`) and 1–30
   (`test_img`); every subject has all 12 video tokens;
3. each prepared folder holds exactly the colour frames of its mirror split and class, and every prepared file is the
   mirror file (same inode) or has its SHA-256; the evaluation files are hashed and checked against their class;
4. the label and group arrays of the training cache (`casia_fasd_1500.npz`) equal the sampling rebuilt from the
   prepared folders, train and test;
5. the checkpoint cards, `models/weights/manifest.json` and the `.h5` files agree on the SHA-256, and the card counts
   equal the manifests.

Observed values (verified; observations, not documented meanings): frame indices are multiples of 25; 3 to 15 frames
per video in `train_img` and 1 to 19 in `test_img`; the `depth/` maps have the same names as the colour frames (not
used by any script).

## CelebA-Spoof

### Source

| item | value | status |
|---|---|---|
| Copy used | Hugging Face mirror `nguyenkhoa/celeba-spoof-for-face-antispoofing-test` (unofficial), shard 0 of 10 of the mirror's `test` split (correspondence with the official test split not verified) | verified (cache metadata and file name) |
| Mirror revision | `f158d9f71d786089d4f8b3b6c16b7bcae5989a3f` | verified |
| File | `test-00000-of-00010.parquet`, 495,744,415 bytes, SHA-256 `b5ac612899ebd7a825f86be4dbe3ffb84534b05bbc60172ba73e154b6e83cf09`, equal to the etag recorded by the Hugging Face download cache (12 September 2026) | verified |
| Conversion script | not in the repository: the parquet was converted once outside the preparation script | verified (absence) |
| Download URL | `https://huggingface.co/datasets/nguyenkhoa/celeba-spoof-for-face-antispoofing-test/resolve/f158d9f71d786089d4f8b3b6c16b7bcae5989a3f/data/test-00000-of-00010.parquet` | verified (revision and file name of the cache) |
| Download | 12 September 2026, 12:03:25 Europe/Rome, with `huggingface_hub` 1.31.0 (`hf_xet` 1.6.0), `local_dir` `data/raw/celeba_spoof`; the exact call was not recorded | verified (cache metadata and download log) |
| Mirror today | on 9 October 2026 the revision is still the mirror's HEAD (last modified 28 December 2024) and the LFS object of the shard at that revision equals our SHA-256 and size; the mirror card states no licence and no source, and names the split `test` (67,170 examples) | verified (Hugging Face API, 9 October 2026) |

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

### Names and keys

| where | form | example (fictional) |
|---|---|---|
| parquet row | row index `i` of the shard (0–6,716), image in `cropped_image.bytes` (PNG) | row 9,999 (outside the shard) |
| prepared file | `{bonafide,attack}_{training,validation}/celeba_<i, 5 digits>.jpg` (PNG content, `.jpg` extension) | `bonafide_training/celeba_09999.jpg` |
| evaluation file | `data/eval/celeba_spoof/{real,attack}/celeba_<i, 5 digits>.jpg` (separate copy, byte-identical) | `real/celeba_09999.jpg` |
| manifest key | `celeba_<i, 5 digits>` | `celeba_09999` |

### Fields

| column | meaning | parsing and closed set |
|---|---|---|
| `key` | row of shard 0 | `^celeba_(\d{5})$`, row below 6,717 |
| `label` | 0 bona fide (`live`), 1 attack (`spoof`) | from the `labels` column; not derivable from the key |
| `label_name` | `live` or `spoof` | must match `label` |
| `row` | row index | the integer of the key, without leading zeros |
| `official_split` | `test` | every row of the copy belongs to the mirror's `test` split (correspondence with the official test split not verified); any other value is refused |
| `sha256`, `bytes` | hash and size of the PNG bytes of the row | equal to those of the prepared and evaluation files (checked by `build`) |

The label cannot be checked against the key: a label changed in one manifest is caught by the key-consistency rule
(S13), which compares it with the parquet manifest. A label changed consistently in every manifest is usually caught by
the conversion rule (S03) or by the declared counts (S14), but this is not guaranteed (for example a row that no
prepared folder uses); only `build`, which reads the parquet, checks it against the source. The same holds for the
`bytes` field: a value changed consistently in every manifest passes every rule undetected without the files.

### Conversion rule (reconstructed)

The procedure that converted the parquet is not versioned. `build` and rule S03 check that this rule reproduces the
row sets of every prepared folder and of the evaluation set exactly. Per class, on the rows with image bytes in row
order:

1. evaluation set = `spread(rows, 150)` (150 rows at a constant step);
2. `_training` = the first 1,500 remaining rows;
3. `_validation` = the 150 evaluation rows plus the next 300 rows (450 per class);
4. the remaining 71 live and 2,707 spoof rows are not used.

### Checks of `build`

1. the parquet SHA-256 equals the etag of the download cache; columns, 6,717 rows, labels consistent with `labelNames`,
   39 rows without image bytes (9 live, 30 spoof; excluded from every manifest, row indices in the restricted
   summary), every image a PNG;
2. every prepared and evaluation file is byte-identical to the PNG bytes of its row and in the folder of its label;
3. the label and group arrays of the training cache (`celeba_spoof_1500.npz`) equal the sampling rebuilt from the
   prepared folders (groups = file names, as in commit `0ac7143`);
4. the checkpoint cards, `models/weights/manifest.json` and the `.h5` files agree on the SHA-256, and the card counts
   equal the manifests.

Observed values (verified): one pair of rows with byte-identical images (both live, both prepared training images);
image sizes vary (faces already cropped by the mirror author).
