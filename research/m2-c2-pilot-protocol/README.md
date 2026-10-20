# C2 pilot protocol draft: CASIA-FASD → NUAA (LivenessNet, AttackNet V2.2, one seed)

Draft of the protocol and configuration of the first RGB cross-dataset pilot, prepared under the M2 card as asked by
the coordinator on 19 October 2026, for review **before** any run. Scope: CASIA-FASD (source) → NUAA (target);
LivenessNet and AttackNet V2.2; seed 42; four evaluations (each model on the CASIA-FASD test and on the NUAA target).
Reverse direction, CelebA-Spoof and more seeds are deferred.

**Nothing has been run for C2: no download, no training, no inference.** The facts come from existing files (M2
manifests and aggregates, caches, checkpoint cards, code), with SHA-256 and counts recomputed read-only on 19 October
2026. No new split; C1, M1, R2 and the M2 runs are unchanged; the C2 run stays in Backlog until the M2 and D1
approvals. Status labels: see `manifest_table.md`. PR #5 (CASIA-FASD code and aggregates) is merged in
`research/pad-2026` at `eb218a8`; the merge does not close the independent check of the 21 restricted CSV [PENDING]
nor approve the splits.

## 1. Manifests

Manifests, counts, SHA-256, origin and status: `manifest_table.md`. Independent unit: the subject (CASIA-FASD: 12
videos each, bona fide tokens 1, 2, HR_1 from the real/fake suffix of the names [VER-LOCAL], frames of a video not
independent; NUAA: still images). Subjects with bona fide / attack samples: dev 4/4, source test 30/30, target 9/15
[VER-LOCAL]: the NUAA BPCER rests on 9 subjects, the dev on 4.

**Source dev (no new split).** `cnn_es`: the 4 subjects held out of `train_img` by `GroupShuffleSplit(n_splits=1,
test_size=0.2, random_state=42)` on the local subject number, rebuilt in M2 with the logic of commit `0ac7143` (the
training of 12 September 2026 saved no file list; the rebuilt split matches the card count n_val = 316); whole
subjects and videos on one side, split before augmentation (rules C07–C09); 75 combinations of 4 of 20 subjects give
316 frames, so the subjects are pinned by the manifest SHA-256 [M2-PASS]. The run must reproduce both manifests by
SHA-256 or stop.

**Which NUAA test, and why.** `cnn_test`, 3,000 images (1,500 + 1,500), `spread(official test, 1500)` per class: (1)
manifest, rule R13 and derived frames (`nuaa_1500.npz`, 3,000 distinct frames) accepted in M2 [ACCEPTED]; the frames
were made by the same preprocessing function as the CASIA-FASD cache [DECLARED: `transformations.md`], so no target
image is preprocessed again; (2) same 16 subjects as the full test (9,123): more images, not more independent units; (3)
balanced classes; (4) no CASIA-FASD file or derived frame equals a NUAA one [VER-LOCAL]; (5) C1 ⊂ this set (R10)
[ACCEPTED] and is only read. The C1 set (300) is not proposed: frozen baseline, too small. Decision D-3: `cnn_test`
3,000, agreed by the reviewer on 19 October 2026.

## 2. Choices fixed before the run

**Preprocessing (source and target alike, `scripts/train_cnn.py`).** RetinaFace crop (Haar cascade, then whole image
as fallbacks), margin 0.15, square; 256 x 256 `INTER_AREA`; BGR → RGB; per-image enhancement of the reference code;
pixels / 255. No dataset statistic; batch normalisation at scoring with `training=False` (source statistics only);
augmentation on source fitting frames only. The detector used per image is not recorded in the caches, so the
historical fallback rate (Haar or whole image) is **not available** for this pilot; measuring it would need a new
detection pass on the images, a separate measurement to be authorised.
The target is never used to fit, stop, calibrate, select or normalise.

**Checkpoints: new training (D-1, agreed by the reviewer on 19 October 2026).** Reasons, source-side provenance and
reproducibility only (`models/weights/PROVENANCE.md`, card notes) [DECLARED]: the existing official CASIA-FASD weights
come from a script older than the repository; seed 42 covered only split and shuffling (initialisation not seeded); the
best epoch is not recorded; best weights were restored only when early stopping fired (LivenessNet stopped after 5
epochs with patience 4, so its restored epoch would be epoch 1, a deduction). Their dev accuracy equals one class share
(71/316 and 245/316), consistent with, not proof of, a constant decision at 0.5; dev scores were never computed.

**Numbers known before this protocol, not used for any choice:** - card metrics on the CASIA-FASD test [DECLARED]:
LivenessNet APCER 1.000, BPCER 0.000, EER 0.4504, AUC 0.550; AttackNet V2.2 APCER 0.000, BPCER 1.000, EER 0.6184, AUC
0.378; - C1 NUAA table, 300 images [C1]: LivenessNet APCER 1.0000, BPCER 0.0000, EER 0.2800, AUC 0.7468; AttackNet
V2.2 APCER 0.0000, BPCER 1.0000, EER 0.3267, AUC 0.6283; and the rows of the pooled CASIA-FASD checkpoints; - local
evaluation of 25 September 2026 on the 300 CASIA-FASD evaluation images, a subset of the source test, not part of the
C1 package: LivenessNet APCER 1.0000, BPCER 0.0000, EER 0.4667, AUC 0.5336; AttackNet V2.2 APCER 0.0000, BPCER 1.0000,
EER 0.6333, AUC 0.3839; - evaluations of the same weights on CelebA-Spoof, SynthASpoof and the sample set (September
2026, outside the C1 package).

**Pre-commitment.** The new models are reported whatever the outcome (flagged `non-informative` if degenerate); no
retraining, hyper-parameter or criterion change after seeing dev, test or target scores. Hyper-parameters are those of
`scripts/train_cnn.py` used for the 12 September 2026 trainings (optimizer, loss and early stopping from the reference
configuration), fixed before C1, not chosen on NUAA or C2 data: at most 12 epochs (`--epochs 12`, the limit of the
existing cards; the script default is 10), batch 16, Adam 1e-3 with `clipnorm` 1.0, class weights, early stopping on
dev `val_loss` (patience 4, best weights restored also at the epoch limit), `ReduceLROnPlateau` (0.5, patience 2).
Checkpoint selection: only that restored epoch. Weights and cards go to the restricted run folder, never to
`models/weights/`. Excluded: the pooled CASIA-FASD checkpoints (all 50 subjects on both sides; 235 of the 300
CASIA-FASD evaluation images in their fitting or early stopping, rules C15, C16) [M2-PASS].

**Threshold (source dev only).** Criterion `eer` of the M1 protocol (`delivery/m1/metrics.md`, amendment of 2 October
2026): candidates = distinct dev scores plus `nextafter(min, −∞)` if ≥ 0; attack iff `score > t` (tie = bona fide);
minimum of `|fp · n_attack − fn · n_bona_fide|` on integers, ties to the highest value; frame unit. Frozen in
`threshold.json`, re-read and recomputed, applied unchanged to source test and target; 0.5 also reported, labelled.
`apcer10` not used (declared: same primary criterion as M1). The dev drives early stopping, learning-rate reduction,
the restored epoch **and** the threshold (declared; optimistic for the source; a separate calibration subset would
leave 2 subjects per role).

**Metrics.** APCER = accepted attacks / attacks, pooled over all attacks (group convention); BPCER = rejected bona
fide / bona fide; ACER = their mean (group convention). ISO/IEC 30107-3 APCER is per PAI species, with the worst case:
on NUAA (print only) the two coincide; on CASIA-FASD they differ, so the per-species APCER is also reported with
`metrics_by_attack_type`, using the token mapping of the Idiap `bob.db.casia_fasd` interface (v2.0.9, `get_type`: 3,
4, HR_2 warped; 5, 6, HR_3 cut; 7, 8, HR_4 video), consistent with the three attack types of Zhang et al., ICB 2012
[DOC-EXT], and assumed to hold for the mirror's file names; applied here for the first time and labelled (declared
choice). Every cell carries `n`, `n_bona_fide`, `n_attack`, accepted attacks, rejected bona fide, ties,
`n_subjects_bona_fide`, `n_subjects_attack`; missing class = `null` with counts. AUC, EER and BPCER@APCER10
descriptive. Unit: frame / image. No frame-level binomial interval (frames are not independent); subject-level
dispersion only in the restricted output. Source test: `cnn_test` (2,091, as the cards and the cache; declared).
Single seed, single split: no variance estimate; a diagnostic pilot, not a generalisation estimate.

## 3. Configuration, commands, inputs and outputs

`c2_pilot_casia_to_nuaa.json`: manifests by SHA-256, preprocessing, budget, selection, threshold, metrics, outputs,
environment; run id `YYYYMMDD-C2-seed42-<shortcommit>`, folder `02_Experiments/C2/<run-id>/` (created by the
coordinator); run report from `research/templates/run-report.md` at run time. Reproducibility: Metal training is not
deterministic, so the evidence is the saved weights and scores; scoring on CPU.

| id | model | set | inputs | output (restricted) |
|---|---|---|---|---|
| E1 | LivenessNet | CASIA `cnn_test` 2,091 | CASIA cache `X_test` + manifest | `scores_E1.csv` |
| E2 | AttackNet V2.2 | CASIA `cnn_test` 2,091 | same | `scores_E2.csv` |
| E3 | LivenessNet | NUAA `cnn_test` 3,000 | NUAA cache `X_test` + manifest | `scores_E3.csv` |
| E4 | AttackNet V2.2 | NUAA `cnn_test` 3,000 | same | `scores_E4.csv` |

**Two phases of the score records** (fields of `research/templates/score-schema.json`, scores written with
round-trip precision, `repr`):
- *Before `select`*: `score --set dev` writes `raw_scores_dev.csv` (316 rows per model). No source-dev threshold
  exists yet, so `threshold` and `prediction` are `null` (declared phase rule: the schema's `threshold` is the number
  frozen from source dev, which does not exist before `select` and is written into records only by `apply`).
  `select` reads only these records.
- *After `select`*: `threshold.json` holds the frozen value per model. `score --set source_test,target` writes
  `raw_scores_source_test.csv` and `raw_scores_target.csv`, also with `threshold` and `prediction` `null`. Only `apply`
  writes the decision records `scores_E1.csv` … `scores_E4.csv` (and the dev at the frozen threshold, for reference),
  with `threshold` = the frozen value and `prediction` = 1 iff `score_attack > threshold` (tie = bona fide). Results
  at 0.5 appear only in the aggregate table, labelled, never in the `threshold` field.

```bash
python scripts/c2_pilot.py check  --config c2_pilot_casia_to_nuaa.json     # manifests, caches, splits; no detector
python scripts/c2_pilot.py train  --config ... --epochs 12 --no-test-eval  # authorisation needed
python scripts/c2_pilot.py score  --config ... --set dev                     # CPU; authorisation needed
python scripts/c2_pilot.py select --config ...                               # dev scores only; writes threshold.json
python scripts/c2_pilot.py score  --config ... --set source_test,target      # refused without threshold.json
python scripts/c2_pilot.py apply  --config ...                               # E1–E4 at the dev threshold and at 0.5
```

`score` records the SHA-256 of `threshold.json` before scoring the test sets; `apply` refuses it if it has changed.

**Only adaptations.** (1) `scripts/train_cnn.py`: an output-folder option (today it writes `models/weights/` and
`manifest.json`, overwriting the existing cards), a split check against `cnn_train` / `cnn_es` by SHA-256, and an
option to skip the final test evaluation (today it scores `X_test` into the card), so no source-test result is seen
before the threshold is frozen. (2) A small `scripts/c2_pilot.py` that scores the cached frames and calls the
selector.

**M1 selector.** `scripts/select_threshold.py` stays unchanged; `threshold_counts`, `candidate_thresholds` and
`select_threshold` are imported. Its NUAA-specific constants (2,507 / 493, 9,123, C1 cache, `ELIGIBLE`), key form
(`real/` / `attack/`) and `subject_of` (split-local on CASIA-FASD) are replaced in `c2_pilot.py`. Its test refusal by
session is not reusable: `session_of` returns `""` on CASIA-FASD names, so the name rule of `check_not_test` would
**pass silently**; it is replaced by refusal by key and SHA-256 of every test and NUAA row. Edge cases (ties, minimum
0.0, constant scores, single-class dev, dev AUC < 0.5): the M1 hand-derived cases, as tests.

**Failure behaviour.** `check` (before training: manifests, caches and splits only, no detector and no weights) stops
with an explicit FAIL and no public output on any SHA-256, count or split mismatch, test or cross-dataset overlap, or a
pooled checkpoint in the configuration. After `train` and before any `score`, the new weights are checked (file SHA-256
against their card, card fields, restored epoch; existing or pooled weights refused); `score` and `apply` refuse
incomplete scores or an edited `threshold.json`; outputs are atomic, deletions go through `path_guard.py`.

**Cost (estimate).** From the cards [DECLARED], about 19 and 26 s per epoch on the training machine (Apple Silicon,
Metal): at most about 4 + 5 minutes of training at 12 epochs, no preprocessing, CPU scoring of minutes; under 30
minutes.

**Perimeter.** Public: code, configuration and its SHA-256, the E1–E4 table with denominators and subject counts,
`threshold.json` (values, aggregate dev counts, SHA-256; no identifier; as published for M1), run report,
`SHA256SUMS`. Restricted (`02_Experiments/C2/<run-id>/`): per-frame scores, dev scores, weights and cards, logs,
subject-level dispersion, copy of the run report.

## Decisions

- **D-1** New training of both models, seed 42: agreed by the reviewer on 19 October 2026.
- **D-3** NUAA target `cnn_test` 3,000: agreed by the reviewer on 19 October 2026.
- **D-4** (open, followed by the coordinator in D1) NUAA copy: the mirror copy accepted in M2 (proposed; the official
  release needs a new download, manifests and preprocessing, outside the current perimeter); and the CASIA-FASD
  agreement (9 October 2026, to be signed through the University) in place before the run.

## Limits

- One direction, one seed, two models; source dev of 4 subjects; NUAA target of 16 subjects (9 with bona fide).
- Both copies are unofficial mirrors (CASIA-FASD `4ea8ec45…`, frames only; NUAA `6dd1ec23…`). CASIA-FASD subject
  disjointness between train and test is documented by the protocol [DOC-EXT], not verified; attack type per token
  [DOC-EXT]. Authorisations are handled by the coordinator (D1); this package certifies none.
- The CASIA-FASD manifests await the independent check of the restricted CSV [PENDING].
- Exact hashes only (no near-duplicate search); cache frames linked to files by sampling order and label/group arrays.

Files: this `README.md`, `c2_pilot_casia_to_nuaa.json`, `manifest_table.md`, `SHA256SUMS`. No per-subject count here.