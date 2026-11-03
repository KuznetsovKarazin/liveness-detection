# NUAA subject folds: schema of the subject-held-out, session-matched check (LivenessNet, 4 folds, 3 seeds)

Schema asked by the coordinator on the C2 card on 3 November 2026, to be fixed **before** any training: four subject
folds, three training seeds, counts per class and session, fitting/dev/test separation and budget. This folder holds
the configuration (the rules, written before the derivation), the script that derives the folds and the per-fold
manifests from the accepted M2 manifests, its tests, the public schema (counts only) and the test logs. **Nothing has
been trained, scored or read from an image**: the schema is a deterministic derivation from file names and manifests,
with no random number, that the coordinator can repeat from the restricted M2 manifests already received.

The question (synthesis, section 5): when the capture session is represented in training, does the N1 recipe
generalise to subjects it has never seen? N2 already measured the opposite cell (the 9 training subjects in session 03:
BPCER 0.6873 for every seed, AUROC 0.3171–0.4569). Diagnostic design, not a benchmark and not a complete causal
isolation (it mixes the three sessions in training and does not preserve the official NUAA protocol); exploratory,
because the NUAA session-03 results are already known: every choice is a rule fixed here, none is chosen on a result.

## Delivery status

- Run `20261103-FOLDS-67b7f3d` from the clean commit `67b7f3d` of the development repository: `public/` copied here as
  `tables/`; `restricted/` (fold members, per-subject counts, the twelve per-fold manifests and the anchor of the public
  folder: 15 files plus `SHA256SUMS`) to be uploaded to the shared N1 folder (provisional destination, as for N2); this
  line is updated at the upload.
- Status: proposed; to be fixed with the coordinator before any training. The training run is a separate package.

## The schema

**Unit.** The subject: the frames of one subject in one session are correlated (consecutive frames of one capture, M2).
NUAA has 16 subjects: 9 with frames in sessions 01–02 (the official training subjects) and 16 in session 03; in
session 03, 9 subjects have bona fide frames (3 of them training subjects), 15 have attacks.

**Strata and folds (no random number).** Four strata from the official lists: A = frames in sessions 01–02 and bona fide
frames in session 03 (3 subjects); B = frames in sessions 01–02, no bona fide frame in session 03 (6); C = no frame in
sessions 01–02, bona fide frames in session 03 (6); D = no frame in sessions 01–02, no bona fide frame in session 03
(1). Balanced dealing, one subject at a time in stratum order A, B, C, D and ascending identifier: to the fold with the
fewest subjects; ties go to the fold with the fewest subjects sharing a property with the subject (frames in sessions
01–02 for A and B, bona fide frames in session 03 for A and C, the sum of the two counts for A), then to the lowest fold
index. Result: four folds of 4 subjects; subjects with bona fide frames in session 03 per fold 2, 3, 2, 2; with attack
frames in session 03 per fold 3, 4, 4, 4; with frames in sessions 01–02 per fold 3, 2, 2, 2. The script fails unless
every fold holds at least two subjects of each of the three kinds (no test or dev stratum rests on one subject in the
session 03 classes; no fallback). The members are restricted (`fold_members.csv`, SHA-256 in the public schema).

**Roles for the held-out fold k.** Test = the 4 subjects of fold k, every frame, reported per session stratum (NA where
a class is absent). Dev = the 4 subjects of fold k mod 4 + 1 (early stopping, learning-rate schedule, restored epoch and
EER threshold). Fitting = the 8 subjects of the other two folds. The three roles are disjoint by subject: no test
subject is ever seen in fitting or dev. Over the four folds every subject is held out exactly once, so the pooled
session-03 test covers the 16 subjects and every one of the 12,614 official images is scored once as a test frame.

**Counts.** Fitting: 350 frames per cell (class × session, 6 cells: 2,100 frames, close to the 2,507 of N1) of the
fitting subjects; dev: 100 frames per cell (600 frames, close to the 493 of N1) of the dev subjects; both with the
deterministic `spread` rule of M2 and `train_cnn.py` (one every len/n of the file names of the cell sorted ascending); a
cell with fewer frames than required fails. Equal counts per session give the three sessions the same weight in fitting
(the session is the factor under study); in session 03 this uses a minority of the available frames (about 18–25 % of
the bona fide and 11–13 % of the attacks), in sessions 01–02 most of them (about 69–97 %). The counts are the largest
multiples of 50 below the smallest availability over the four held-out folds (fitting: 361 bona fide frames of session
02 for held-out fold 3, so 350; dev: 117 bona fide frames of session 02 for held-out fold 1, so 100). Test: all frames.
The per-fold counts and the availability of every cell are in `tables/`.

**Frames.** 6,000 frames are already in the training cache `nuaa_1500.npz` (reproduced bit for bit from the originals
in N2); the other 6,614 are extracted once, before the trainings, with the frozen cache extractor (`face.py` of commit
`0ac7143`, delivered with N2) and the unchanged `train_cnn.preprocess` chain, after the N2 cache check is repeated.

**Training (N1 recipe, unchanged).** LivenessNet (chosen by the coordinator on 3 November to stay comparable with N1),
CPU, at most 12 epochs, batch 16, Adam 1e-3 with clipnorm 1.0, balanced class weights, the N1 augmentation on the
fitting frames only, early stopping on the dev `val_loss` (patience 4, best weights restored), `ReduceLROnPlateau`;
training seeds 42, 43, 44 (initialisation, augmentation, batch order); no split seed. 12 trainings, one checkpoint
each (weights restricted; cards and SHA-256 delivered).

**Threshold.** EER on the dev frames with the M1 criterion, frozen before any test scoring and applied unchanged; 0.5
also reported, labelled. Declared: the dev drives early stopping, the restored epoch and the threshold (as in N1 and
the C2 protocol); when the dev classes separate completely the dev does not identify a unique threshold (M1) and gives
no guarantee of transfer.

**Metrics.** Per fold, seed and session stratum: denominators, error counts, APCER/BPCER/ACER at 0.5 and at the dev
threshold, AUROC; per subject (restricted); median and range over seeds per fold; pooled counts over the four folds per
session; a bootstrap over subjects (1,000 resamples of the 16 held-out subjects, seed 0, percentile 2.5–97.5 of the
pooled ACER and AUROC per session) as the uncertainty; no statistical test. Declared: the pooled estimate is a
cross-validation aggregate scored by four models with four dev thresholds; the session strata hold at most 9 subjects,
so the interval is coarse, and no interval is given for a stratum with fewer than 5 subjects. Pre-commitment: reported
whatever the outcome; no change of recipe, criterion, folds or counts after any score is read.

**Budget.** 12 trainings of about the N1 size (LivenessNet: 496–671 s each on the CPU in N1 for 2,507 fitting frames;
2,100 here), so at most about 2.5 h of training; one extraction pass of the 6,614 frames outside the cache; 12 scoring
passes; wall-time cap 6 h, else stop, no result, reported as a failed run.

**Reading caveat.** Adding session 03 to training also adds variety: a low error on the held-out subjects would be
consistent with an N1/C1 failure due to "session not represented, or training data too narrow" rather than to the
session alone; a high error would be consistent with a failure of the recipe; both readings are diagnostic, not causal.
Limits: frames stay correlated; per-subject counts are unbalanced; the session 01–02 test and dev strata rest on 2–3
subjects per fold and a single session × class cell there can rest on one subject (two training subjects lack bona fide
frames in one of the two sessions); the fixed counts under-sample session 03; the dev of a held-out fold is the test
fold of the next one, so the four fold results are not independent (different models, no leakage); three seeds give a
range, not a confidence interval; one dataset, one attack type.

## Files

| path | what it is |
|---|---|
| `livenesslab/delivery/folds/nuaa_subject_folds.json` | the configuration (at the default path of the script): rules, counts per cell, recipe, threshold, metrics, budget, declarations; SHA-256 of the five M2 manifests |
| `livenesslab/scripts/nuaa_subject_folds.py` | `check-inputs` (manifests verified, schema derived, nothing written) and `run` (public schema and restricted manifests) |
| `livenesslab/scripts/check_nuaa_subject_folds.py` | 43 tests on synthetic manifests with identifiers and structure unlike NUAA (strata, balanced dealing, roles, sampling, availability, counts, SHA256SUMS completeness, error paths, public without identifiers, no stale PASS); one test on the real manifests, skipped without them |
| `livenesslab/scripts/check_nuaa_subject_folds_mutations.py` | 46 declared mutations of the script (plus a control and one declared redundant), each run against the test suite on a copy: 46/46 detected |
| `tables/fold_schema_public.json`, `tables/fold_schema_public.md` | the public schema of run `20261103-FOLDS-67b7f3d`: counts per fold, role, session and class, subjects per cell, frames in cache and to extract, SHA-256 of the restricted lists, budget |
| `test_logs/` | outputs of the test commands on macOS and in a Linux container |
| `SHA256SUMS` | SHA-256 of every file of this folder |

Unchanged copies of `m2_writers.py`, `path_guard.py` and `package_rules.py` are those of the accepted packages.

## Commands (from `livenesslab/` of this folder, or from the development repository root)

```bash
python scripts/nuaa_subject_folds.py check-inputs --manifests <folder of the M2 NUAA manifests>
python scripts/nuaa_subject_folds.py run --manifests <folder of the M2 NUAA manifests> --out-root <folder>
python scripts/check_nuaa_subject_folds.py
python scripts/check_nuaa_subject_folds_mutations.py
```

Dependencies: Python 3.11 or 3.12, standard library only. Without the restricted manifests the suite runs on synthetic
manifests and skips the one test on the real ones (declared). Limit of the input checks: the configuration is the
declaration of the inputs (the five M2 manifests by SHA-256 and counts); a configuration that names different
manifests describes a different schema, and the check does not see it as an error.

## What this package does not do

No training, no inference, no image read, no face detection. It fixes the schema only; the training run, once the
schema is agreed, is a separate package with its own configuration, cache check, extraction records, run report and
restricted scores.
