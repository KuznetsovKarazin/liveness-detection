# NUAA file naming schema and manifest fields

This note documents how the NUAA file names are parsed into the fields of the M2 manifests, where the schema comes
from, what was checked on the data, and why the C1 manifest gets a new version.

## Source

- Copy used: public Hugging Face mirror `akahana/anti-spoofing-nuaaaa`, revision
  `6dd1ec237710732f92e73ca7653e695fd47c7a86`, archive `nuaaaa.tar.gz` with SHA-256
  `c0780c6f1e581b720eb3c48faaeef7713748b174233d3fcafa8da716b5c81a3e`.
- The archive contains the images (`raw/ClientRaw/<ID>/`, `raw/ImposterRaw/<ID>/`) and four split lists
  (`raw/client_train_raw.txt`, `raw/imposter_train_raw.txt`, `raw/client_test_raw.txt`,
  `raw/imposter_test_raw.txt`), but **no README**.
- Original source: NUAA Photograph Imposter Database, Nanjing University of Aeronautics and Astronautics; reference
  publication: X. Tan, Y. Li, J. Liu, L. Jiang, "Face Liveness Detection from a Single Image with Sparse Low Rank
  Bilinear Discriminative Model", ECCV 2010.
- Documented: the field order `ID_glasses_pos_session_picNo`, from the README of the original release as reported by
  the coordinator on the M2 task card (3 October 2026); that README is not in the mirror and was not available to us.
- Verified in the mirror: the structure of the names (five numeric fields), that ID equals the subject folder, and
  that the session field separates the splits (01–02 train, 03 test).
- Not verified: the meaning of the glasses and pos codes, carried as opaque two-digit codes.
- Split lists: the mirror's lists carry the path prefix `/kaggle/input/nuaaaa/raw/`, added by the mirror author: they
  are the mirror's version of the official lists. Only file names are compared. We did not verify their equivalence
  with the lists of the original release; the coordinator rebuilt the 300 C1 images from the official source with
  identical hashes (M1 card, 2 October 2026). `build` accepts only the mirror's list format.
- Subjects: the 4-digit IDs of the official folders and file names. No identity is inferred from image content or from
  groups of images. Treating the same ID in ClientRaw and ImposterRaw as the same person follows the naming of the
  release and is not verified on the images.

## Official name and name on disk

The examples use a fictional picture number (618, outside the observed range 0–499), so that no real image is named.

| where | form | example (fictional) |
|---|---|---|
| split list entry (mirror copy) | `/kaggle/input/nuaaaa/raw/{ClientRaw,ImposterRaw}/<ID>/<ID>_<glasses>_<pos>_<session>_<picNo>.jpg` | `.../ClientRaw/0004/0004_01_06_03_618.jpg` |
| archive member (original file name) | `raw/{ClientRaw,ImposterRaw}/<ID>/<ID>_<glasses>_<pos>_<session>_<picNo>.jpg` | `raw/ClientRaw/0004/0004_01_06_03_618.jpg` |
| prepared file name (on disk) | `<subject>_<ID>_<glasses>_<pos>_<session>_<picNo>.jpg` | `0004_0004_01_06_03_618.jpg` |
| manifest key | `real/<prepared name>` (ClientRaw) or `attack/<prepared name>` (ImposterRaw) | `real/0004_0004_01_06_03_618.jpg` |

`scripts/prepare_datasets.py` builds the prepared name by replacing the `/` between the subject folder and the
original name with `_`, so the subject folder appears as a prefix. Images of the train lists go to
`{bonafide,attack}_training`, images of the test lists to `{bonafide,attack}_validation` (hard links, no copy; the
`_validation` folders hold the official **test** split and were never used as validation data). The archive member
of every row is therefore `raw/<ClientRaw if label 0, ImposterRaw if label 1>/<subject>/<key without the class
folder and without the subject prefix>`; `build` checks, for all 12,614 rows, that the prepared file and that archive
member are the same file (same inode) or have the same SHA-256 (all 12,614 are hard links in this copy). The key is
the class folder plus the prepared name: the split folder is deliberately not part of the key (the split is the
`official_split` column). The class is part of the key because the same prepared name can exist in both classes: 165
names of the official training split exist both as a bona fide and as an attack image (with different content).

## Fields of the manifests

| column | meaning | parsing |
|---|---|---|
| `key` | class folder and prepared file name | `real/` or `attack/` + name |
| `label` | 0 bona fide (ClientRaw), 1 attack (ImposterRaw) | from the split list / folder |
| `subject` | subject ID | 1st field of the prepared name (folder); must equal the 2nd field (ID of the original name) |
| `glasses` | glasses field (two-digit code, opaque) | 3rd field of the prepared name |
| `pos` | pos field (two-digit code, opaque) | 4th field |
| `session` | capture session | 5th field |
| `pic` | picture number within the session | 6th field (integer) |
| `official_split` | `train` or `test` | from the split list that contains the image (equal to the session rule: 01–02 train, 03 test) |
| `sha256`, `bytes` | content hash and size of the file | computed on the prepared file |

There is no attack-type column: NUAA has a single attack type (printed photographs re-captured by webcam), implied
by label 1. There are no video identifiers: images are grouped only by subject and session.

Regular expressions used by `scripts/nuaa_manifests.py`:

- prepared name: `^(\d{4})_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d+)\.jpg$`, refused if the first two fields differ;
- list entry: `^/kaggle/input/nuaaaa/raw/(ClientRaw|ImposterRaw)/(\d{4})/((\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d+)\.jpg)$`,
  refused if the folder differs from the ID field or the class folder differs from the list (client/imposter).

## Checks against the split lists

`nuaa_manifests.py build` checks, and stops with exit code 1 otherwise, that:

1. every line of the four split lists matches the list-entry pattern, with the subject folder equal to the ID field
   (12,614 entries, no duplicates);
2. each prepared folder contains exactly the images of its list, name by name (1,743 + 1,748 + 3,362 + 5,761);
3. every prepared image is the archive member derived from its key (same inode, or same SHA-256);
4. the archive SHA-256 equals the etag recorded by the Hugging Face download cache.

Observed values (split lists, bona fide / attack; observations, not documented meanings):

| field | train | test |
|---|---|---|
| session | 01, 02 (both classes) | 03 only (both classes) |
| subjects | 0001–0009 (both classes) | bona fide: 9 subjects (3 shared with train); attack: 15 subjects (all 9 training subjects plus 6 more) |
| glasses | 00, 01 | 00, 01 |
| pos | bona fide 00; attack 00–04 | bona fide 05, 06, 07; attack 01–04, 08 |
| picNo | 0–499 | 0–499 |

The pos codes differ by class in the test split (bona fide 05–07, attacks 01–04 and 08): an observation, recorded
because the code is class-specific there; its meaning is not verified. The session field separates the official
splits (01–02 in train, 03 in test). The subject does not: NUAA is not subject-disjoint between train and test.

## C1 manifest: session correction and versioning

The C1 manifest (`nuaa_manifest.csv`, 300 rows, SHA-256
`86095da1531b1f24b1233c9668ac1ab37714978e444b313456732c028ccfb40c`) was written by `scripts/make_c1_manifest.py`
with the pattern `^(\d{4})_\d{4}_(\d{2})_`, which takes the **3rd** field of the prepared name. That field is the
glasses code, not the session: for all 300 rows the `session` column equals the glasses field (values 00 and 01),
while the capture session (5th field) is 03 for all 300 images. Keys, labels, subjects, SHA-256 and bytes are correct.

- **Version 1** (`nuaa_manifest.csv`) stays frozen: it is the manifest delivered with C1 and used by the R2
  replication; its SHA-256 is registered in the code (`C1_MANIFEST_SHA256`) and checked by `build`.
- **Version 2** (`nuaa_c1_v2.csv`) is written by `build` with the columns of this note: the `file` column of version
  1 becomes `key` (same values), every field is re-parsed from the name, `session` = 03 for all rows, `glasses` holds
  the value of the old `session` column. It is a per-image, per-subject record and stays in the restricted M2 folder;
  its SHA-256 is listed in the public aggregate (`nuaa_manifests_aggregate.json`) together with the correction
  counts. `nuaa_manifests.py compare-c1` checks the two versions row by row without images.
