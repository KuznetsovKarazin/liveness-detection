# Session comparison of the 48 LivenessNet and ResNet-18 checkpoints on NUAA at the registered development thresholds: results

Card C2. Run `20261123-ARCHSESS-356b041`. Aggregates only: no per-frame score, no frame identifier and no subject
identifier is in this folder. The protocol, the configuration and the scripts are those registered in
`research/pad-arch-session-compare/` (commit `356b0418b685fb2b7f8d4db81affa6ffe3f19be9`, pull request #38), published
and linked on the card before the run. The order of this note is that of `PROTOCOL.md` §9 (point 3), with the
denominators, asked for this delivery, after the reproduction checks: the reproduction checks first, then the
denominators, the highlighted outcome, the table, the figure and the limits; small differences and differences of
opposite sign are reported as such.

## Run

- One execution, on 23 November 2026 from 20:05:43 to 20:05:54 (+0100), exit code 0, from the registered folder at
  commit `356b041` (committed 2026-11-23T08:46:59+01:00), with the trees clean at launch
  (`runs/20261123-ARCHSESS-356b041/conditions.txt`). The SHA-256 of the registered `SHA256SUMS` and of
  `session_config.json` are written in `conditions.txt`; both scripts printed the registered SHA-256 of the
  configuration as the first line of their output (`run.log.txt`).
- Order: `check_synthetic.py` first, passed ("Synthetic end-to-end check passed: 192 operating points, 16 combinations,
  table and figure, 41 stop cases."); then `session_compare.py` once, with a new output folder, exit code 0; then
  `session_summary.py`, exit code 0.
- Inputs: the restricted score files kept from the evaluation `20261121-ARCHEVAL-332fbdf` (48 test files) and of the
  training runs `20261120-ARCHRUN-p4-7faa49f` and `20261120-ARCHRUN-p12-ea7fdca` (96 development files), with their
  SHA-256 manifests validated immediately before the run; `score_metrics.csv` (SM) and `paired_summary.csv` (PS) of
  pull request #36, with the SHA-256 registered in the configuration (`conditions.txt`).
- Python 3.11.16, NumPy 1.26.4, Matplotlib 3.11.2 (`conditions.txt`).
- No training, no model loaded, no image read: the two scripts read score files and CSV files only.
- `private commit` in `conditions.txt` is the commit of the author's working repository at launch, which is not
  public. The last line of `run.log.txt` ("fork still clean") is written by the launch wrapper after the run: the
  fork tree, which holds the registered folder, was still clean.

## Reproduction checks

No check failed. Each check stops the script with exit code 4 and nothing is written (`PROTOCOL.md` §7);
`session_compare.py` ended with exit code 0 and the line "192 operating points written to session_counts.csv; checks
C1-C12 passed." (`run.log.txt`). C2, C3, C4, C5, C7, the file part of C10 and C11 need the restricted score files or
the run folder: for them the only evidence in this folder is that line and the exit code. On the real inputs,
therefore:

- **Public inputs** (before any restricted file was opened). C1: SM is the published file of pull request #36
  (SHA-256, 96 rows, the 96 expected keys). The first part of C12: the SHA-256 of PS is the registered one. The first
  part of C6: `expected_counts.csv` is the registered one, with its 16 cells.
- **Rows, pairing, checkpoints, sessions.** C2: detector, fold, seed, BN state and role of every row equal to the file
  name, no duplicated `sample_id`, the stored decision at 0.5 consistent with the score, scores finite in [0, 1].
  C3: within each fold the ordered tuple (`sample_id`, `subject_id`, `label`, `session_id`) of the test is identical
  across detectors, seeds, patience settings and BN states, row by row; development and test share no frame and no
  identifier; each development partition has 400 frames. C4: one checkpoint SHA-256 per model, the same in the two
  states and in the development and test files. C5: `session_id` is `01`, `02` or `03`, equal to the session field of
  the file name; the class prefix of `sample_id` consistent with the label; `subject_id` equal to both identifier
  fields.
- **Denominators.** C6: the test frames and the manifest identifiers per fold × class × session group equal
  `expected_counts.csv`, registered before the run (16 cells, exact); every denominator is above 0.
- **Thresholds.** C7: the 192 thresholds re-selected from the development scores with the selection of the script of
  pull request #35 copied unchanged are equal to those of SM, with their development APCER and BPCER. C8: the same
  threshold, equal to the string of SM, is written for the two groups in all 192 rows of `session_counts.csv`.
- **Pooling = pull request #36.** C9: for each of the 192 operating points the pooled denominators are the fold totals
  and the pooled photograph and bona fide errors give back exactly the test APCER and BPCER of SM: 384 equalities,
  zero tolerance on counts. C12: for the 16 combinations the twelve-unit mean of the pooled rates, computed as in the
  script of pull request #35, equals `stored_mean` or `reestimated_mean` of PS: 32 values.
- **Completeness and output.** C10: the 144 input files exist; 16 combinations × 12 fold–seed units, each once, with
  both groups. C11: the output folder was new.

`session_summary.py` repeated C1, C6 (frame counts), C8, C9, C10 and C12 on the public files only and ended with exit
code 0 ("checks C1, C6, C8, C9, C10 and C12 passed"). C8, C9 and C12 need only public files: `session_counts.csv`
(columns `threshold_*`, `*_errors`, `*_n_*`, `pooled_*`, `published_test_*`) and SM and PS of pull request #36.
C1, C6 (frame counts), C8, C9, C10 (rows) and C12 can be repeated by anyone from the files of this folder and those of
pull request #36.

**Outcome of the check against pull request #36: pooling the counts of each fold and seed gives back exactly the 384
test rates of SM (C9) and the 32 means of PS (C12).**

## Denominators

Test frames per fold, class and session group, with the number of manifest identifiers in brackets
(`expected_counts.csv` of the registration; identifiers are counted, never listed). In `session_counts.csv` the four
frame counts of a fold are the same in all 48 rows of that fold (2 detectors × 2 patience settings × 3 seeds × 2 BN
states × 2 targets), as check C6 requires.

| Fold | Photograph, sessions 1–2 | Photograph, session 3 | Bona fide, sessions 1–2 | Bona fide, session 3 | Fold total |
|---|---|---|---|---|---|
| 1 | 587 (3) | 1,232 (3) | 498 (3) | 925 (2) | 3,242 |
| 2 | 380 (2) | 1,364 (4) | 351 (2) | 1,019 (3) | 3,114 |
| 3 | 387 (2) | 1,665 (4) | 522 (2) | 865 (2) | 3,439 |
| 4 | 394 (2) | 1,500 (4) | 372 (2) | 553 (2) | 2,819 |
| All | 1,748 | 5,761 | 1,743 | 3,362 | 12,614 |

Sessions 1–2 hold 3,491 test frames and session 3 9,123; photograph 7,509 and bona fide 5,105; 12,614 in all. Both
groups contain both classes in every fold.

## Highlighted outcome

Patience 4 and development APCER target 5%, for both detectors and both BN states, APCER and BPCER separately: the
author's choice fixed in `PROTOCOL.md` §5 (point 4) before the run; the request names none. Percentage points; each
value is the mean of the twelve fold–seed units with equal weight; difference = session 3 − sessions 1–2 (positive:
more errors on session 3); minimum and maximum of the twelve unit differences; units up/zero/down decided on integer
counts. Every value is rounded once to two decimals from its exact rational value (half to even), as in the table.

| Detector | BN state | APCER (%) sessions 1–2 | APCER (%) session 3 | APCER (%) difference | APCER difference min to max | APCER units up/zero/down | BPCER (%) sessions 1–2 | BPCER (%) session 3 | BPCER (%) difference | BPCER difference min to max | BPCER units up/zero/down |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LivenessNet | stored | 6.13 | 42.15 | +36.02 | -1.03 to +71.21 | 11/0/1 | 11.68 | 45.74 | +34.05 | -5.84 to +86.17 | 11/0/1 |
| LivenessNet | re-estimated | 9.13 | 63.02 | +53.88 | +24.08 to +72.98 | 12/0/0 | 0.03 | 23.86 | +23.83 | +0.29 to +50.29 | 12/0/0 |
| ResNet-18 | stored | 10.25 | 79.72 | +69.46 | +53.42 to +92.35 | 12/0/0 | 0.07 | 2.95 | +2.87 | -0.57 to +34.54 | 2/8/2 |
| ResNet-18 | re-estimated | 8.40 | 77.35 | +68.95 | +40.72 to +90.27 | 12/0/0 | 0.05 | 0.42 | +0.37 | -0.28 to +4.88 | 2/8/2 |

- APCER: the mean difference is positive for both detectors and both states, +36.02 and +53.88 points for
  LivenessNet (stored, re-estimated) and +69.46 and +68.95 for ResNet-18; 47 of the 48 unit differences are positive
  and 1 is negative.
- BPCER: +34.05 and +23.83 points for LivenessNet; +2.87 and +0.37 for ResNet-18, whose twelve units are 2 up, 8 zero
  and 2 down in each state.
- The negative unit of LivenessNet in both metrics is fold 3, seed 44 with the stored statistics (the unit whose
  development BPCER is 1 at both targets in pull request #36): -1.03 points of APCER and -5.84 of BPCER at the 5%
  target. At the 5% threshold 506 of the 522 bona fide frames of sessions 1–2 and 788 of the 865 of session 3 are
  rejected (780 of the 865 at the 10% threshold).

## Table

`runs/20261123-ARCHSESS-356b041/public/session_table.md`, as written by `session_summary.py`, reproduced here
unchanged (full values in `session_summary.csv`):

Test APCER and BPCER (%) of sessions 1–2 and of session 3 at the development thresholds of pull request #36 (development APCER target 5% or 10%), and the paired difference session 3 − sessions 1–2. Each cell is the mean of twelve fold–seed units with equal weight; the thresholds are transferred from the development scores of each checkpoint and BN state, as in Table A3, and are the same for the two groups. Units up/zero/down: fold–seed units with a positive, zero or negative difference, decided on integer counts. Every value is rounded separately to two decimals from its exact rational value (half to even), so the difference shown can differ by 0.01 from the difference of the two values shown; full values are in session_summary.csv. Rows in bold: highlighted outcome (patience 4, target 5%; author's choice).

| Patience | Detector | Target (%) | BN state | APCER (%) sessions 1–2 | APCER (%) session 3 | APCER (%) difference | APCER units up/zero/down | BPCER (%) sessions 1–2 | BPCER (%) session 3 | BPCER (%) difference | BPCER units up/zero/down |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **4** | **LivenessNet** | **5** | **stored** | **6.13** | **42.15** | **+36.02** | **11/0/1** | **11.68** | **45.74** | **+34.05** | **11/0/1** |
| **4** | **LivenessNet** | **5** | **re-estimated** | **9.13** | **63.02** | **+53.88** | **12/0/0** | **0.03** | **23.86** | **+23.83** | **12/0/0** |
| 4 | LivenessNet | 10 | stored | 8.62 | 49.15 | +40.53 | 11/0/1 | 11.64 | 41.61 | +29.97 | 11/0/1 |
| 4 | LivenessNet | 10 | re-estimated | 10.98 | 67.29 | +56.31 | 12/0/0 | 0.03 | 21.84 | +21.81 | 11/1/0 |
| **4** | **ResNet-18** | **5** | **stored** | **10.25** | **79.72** | **+69.46** | **12/0/0** | **0.07** | **2.95** | **+2.87** | **2/8/2** |
| **4** | **ResNet-18** | **5** | **re-estimated** | **8.40** | **77.35** | **+68.95** | **12/0/0** | **0.05** | **0.42** | **+0.37** | **2/8/2** |
| 4 | ResNet-18 | 10 | stored | 12.49 | 84.48 | +71.99 | 12/0/0 | 0.02 | 2.91 | +2.88 | 2/9/1 |
| 4 | ResNet-18 | 10 | re-estimated | 9.99 | 81.87 | +71.88 | 12/0/0 | 0.02 | 0.00 | -0.02 | 0/11/1 |
| 12 | LivenessNet | 5 | stored | 10.60 | 60.44 | +49.83 | 12/0/0 | 0.12 | 19.67 | +19.55 | 9/3/0 |
| 12 | LivenessNet | 5 | re-estimated | 9.64 | 60.02 | +50.38 | 12/0/0 | 0.03 | 22.96 | +22.93 | 12/0/0 |
| 12 | LivenessNet | 10 | stored | 12.33 | 65.72 | +53.39 | 12/0/0 | 0.10 | 17.71 | +17.61 | 9/3/0 |
| 12 | LivenessNet | 10 | re-estimated | 11.41 | 64.22 | +52.82 | 12/0/0 | 0.03 | 21.66 | +21.63 | 12/0/0 |
| 12 | ResNet-18 | 5 | stored | 10.28 | 77.75 | +67.48 | 12/0/0 | 0.07 | 2.95 | +2.87 | 2/8/2 |
| 12 | ResNet-18 | 5 | re-estimated | 8.99 | 77.97 | +68.98 | 12/0/0 | 0.05 | 2.14 | +2.09 | 3/7/2 |
| 12 | ResNet-18 | 10 | stored | 12.85 | 83.91 | +71.06 | 12/0/0 | 0.02 | 2.91 | +2.88 | 2/9/1 |
| 12 | ResNet-18 | 10 | re-estimated | 11.39 | 82.55 | +71.16 | 12/0/0 | 0.02 | 0.14 | +0.11 | 1/10/1 |

Test frames per fold, sessions 1–2 and session 3: fold 1: photograph 587 and 1,232, bona fide 498 and 925; fold 2: photograph 380 and 1,364, bona fide 351 and 1,019; fold 3: photograph 387 and 1,665, bona fide 522 and 865; fold 4: photograph 394 and 1,500, bona fide 372 and 553. The two groups contain different sets of manifest identifiers (PROTOCOL.md, limits).

## Figure

`runs/20261123-ARCHSESS-356b041/public/session_differences.pdf`, one page: the APCER difference (top row) and the BPCER
difference (bottom row), session 3 − sessions 1–2, in percentage points, for LivenessNet patience 4, ResNet-18
patience 4, LivenessNet patience 12 and ResNet-18 patience 12 (columns, the order of the table); in each panel the
positions 5% stored, 5% re-estimated, 10% stored and 10% re-estimated, the twelve fold–seed units as dots, the
equal-weight mean as a dash and the zero line. The dots are computed from `session_counts.csv`, the dashes are the
means of `session_summary.csv`.

## Signs, minimum and maximum over the 16 combinations

Minimum and maximum of the twelve unit differences of each combination (percentage points; `session_summary.csv`,
columns `*_min_difference` and `*_max_difference`); the units up/zero/down are in the table.

| Patience | Detector | Target (%) | BN state | APCER difference, minimum | APCER difference, maximum | BPCER difference, minimum | BPCER difference, maximum |
|---|---|---|---|---|---|---|---|
| 4 | LivenessNet | 5 | stored | -1.03 | +71.21 | -5.84 | +86.17 |
| 4 | LivenessNet | 5 | re-estimated | +24.08 | +72.98 | +0.29 | +50.29 |
| 4 | LivenessNet | 10 | stored | -3.30 | +69.49 | -6.76 | +86.26 |
| 4 | LivenessNet | 10 | re-estimated | +27.51 | +79.86 | 0.00 | +50.29 |
| 4 | ResNet-18 | 5 | stored | +53.42 | +92.35 | -0.57 | +34.54 |
| 4 | ResNet-18 | 5 | re-estimated | +40.72 | +90.27 | -0.28 | +4.88 |
| 4 | ResNet-18 | 10 | stored | +54.74 | +91.87 | -0.28 | +34.54 |
| 4 | ResNet-18 | 10 | re-estimated | +42.16 | +92.27 | -0.28 | 0.00 |
| 12 | LivenessNet | 5 | stored | +27.15 | +73.10 | 0.00 | +61.54 |
| 12 | LivenessNet | 5 | re-estimated | +24.08 | +72.98 | +0.29 | +50.29 |
| 12 | LivenessNet | 10 | stored | +30.99 | +72.89 | 0.00 | +55.85 |
| 12 | LivenessNet | 10 | re-estimated | +27.51 | +79.86 | +0.10 | +50.29 |
| 12 | ResNet-18 | 5 | stored | +53.42 | +90.68 | -0.57 | +34.54 |
| 12 | ResNet-18 | 5 | re-estimated | +40.72 | +92.23 | -0.28 | +20.61 |
| 12 | ResNet-18 | 10 | stored | +54.74 | +92.06 | -0.28 | +34.54 |
| 12 | ResNet-18 | 10 | re-estimated | +42.16 | +93.41 | -0.28 | +1.63 |

- APCER: the 16 means are positive, from +36.02 to +71.99 points; of the 192 unit differences 190 are positive, none
  is zero and 2 are negative; they range from -3.30 to +93.41.
- BPCER: the means range from -0.02 to +34.05 points; of the 192 unit differences 101 are positive, 77 zero and 14
  negative; they range from -6.76 to +86.26.

## Small differences and differences of opposite sign

- **BPCER of ResNet-18 (small differences).** In all 8 combinations the mean difference is between -0.02 and +2.88
  points; 70 of the 96 unit differences are zero (7 to 11 per combination), 14 positive and 12 negative. The mean
  BPCER is between 0.02 and 0.07 points on sessions 1–2 and between 0.00 and 2.95 on session 3.
- **Where the positive BPCER means of ResNet-18 come from.** With the stored statistics (both patience settings, both
  targets) one unit, fold 4, seed 44, goes from 0 of the 372 bona fide frames of sessions 1–2 rejected to 191 of the
  553 of session 3 (+34.54 points) and alone contributes +2.88 points to the mean; the other eleven units together
  contribute less than 0.01 points in absolute value. With the re-estimated statistics the positive means also come
  from fold 4: at target 5%, seed 44 (27 of 553, +4.88) and seed 42 (1 of 553, +0.18) at both patience settings, and
  at patience 12 also seed 43 (114 of 553, +20.61); at target 10%, patience 12, seed 43 (9 of 553, +1.63). At target
  10%, patience 4, re-estimated, no unit of fold 4 rejects a bona fide frame and the mean is the negative one below.
  With the re-estimated statistics the units outside fold 4 contribute between -0.05 and -0.02 points.
- **One negative mean.** ResNet-18, patience 4, target 10%, re-estimated statistics, BPCER: -0.02 points
  (-0.00023741690408357076 as a proportion in `session_summary.csv`, that is -1/4212). It comes from one unit, fold 2,
  seed 42, with 1 of the 351 bona fide frames of sessions 1–2 rejected and none of the 1,019 of session 3; the other
  eleven units have no bona fide error in either group (11 zero, 1 down).
- **Negative unit differences** (all 16; percentage points):

  | Patience | Detector | Target (%) | BN state | Metric | Fold | Seed | Difference |
  |---|---|---|---|---|---|---|---|
  | 4 | LivenessNet | 5 | stored | APCER | 3 | 44 | -1.03 |
  | 4 | LivenessNet | 5 | stored | BPCER | 3 | 44 | -5.84 |
  | 4 | LivenessNet | 10 | stored | APCER | 3 | 44 | -3.30 |
  | 4 | LivenessNet | 10 | stored | BPCER | 3 | 44 | -6.76 |
  | 4 | ResNet-18 | 5 | stored | BPCER | 2 | 42 | -0.57 |
  | 4 | ResNet-18 | 5 | stored | BPCER | 2 | 44 | -0.28 |
  | 4 | ResNet-18 | 5 | re-estimated | BPCER | 2 | 42 | -0.28 |
  | 4 | ResNet-18 | 5 | re-estimated | BPCER | 2 | 44 | -0.28 |
  | 4 | ResNet-18 | 10 | stored | BPCER | 2 | 42 | -0.28 |
  | 4 | ResNet-18 | 10 | re-estimated | BPCER | 2 | 42 | -0.28 |
  | 12 | ResNet-18 | 5 | stored | BPCER | 2 | 42 | -0.57 |
  | 12 | ResNet-18 | 5 | stored | BPCER | 2 | 44 | -0.28 |
  | 12 | ResNet-18 | 5 | re-estimated | BPCER | 2 | 42 | -0.28 |
  | 12 | ResNet-18 | 5 | re-estimated | BPCER | 2 | 44 | -0.28 |
  | 12 | ResNet-18 | 10 | stored | BPCER | 2 | 42 | -0.28 |
  | 12 | ResNet-18 | 10 | re-estimated | BPCER | 2 | 42 | -0.28 |

  LivenessNet: only fold 3, seed 44 with patience 4 and the stored statistics, in both metrics and at both targets.
  ResNet-18: BPCER only, fold 2, seeds 42 and 44, by 1 or 2 of the 351 bona fide frames of sessions 1–2, with none of
  the 1,019 of session 3 rejected.
- **Zero differences of LivenessNet.** BPCER has 7 zero unit differences of 96 (1 at patience 4, target 10%,
  re-estimated; 3 in each of the two stored combinations at patience 12).

## Ties at the threshold (diagnostic)

A score equal to the threshold is bona fide (`PROTOCOL.md` §4): a tied photograph frame is an error, a tied bona fide
frame is not. Ties occur in 16 of the 192 operating points (10 LivenessNet, 6 ResNet-18): 14 photograph and 274 bona
fide frames of sessions 1–2, 179 photograph and 112 bona fide frames of session 3 in all. The counts per operating
point are in the `*_ties` columns of `session_counts.csv`; they are counts, not errors. The largest case is
LivenessNet, patience 4, stored statistics, fold 1, seed 42 (threshold 0.9999997615814209, both targets): 137 of the
498 bona fide frames of sessions 1–2 and 10 of the 925 of session 3 are tied and count as accepted; 63 of the 1,232
photograph frames of session 3 are tied and count as errors.

## Limits (`PROTOCOL.md` §10, unchanged)

- The session contrast is not adjusted for identity: within a fold and class, sessions 1–2 and session 3 contain
  different sets of manifest identifiers, so the difference reflects both session and identifier composition. For
  the photograph class the identifiers of sessions 1–2 are contained in those of session 3 in every fold (session 3
  adds 0, 2, 2 and 2); for the bona fide class one identifier is in both groups in folds 1–3 and none in fold 4; the
  two one-class identifiers are in session 3 only. The bona fide part of sessions 1–2 rests on 2–3 identifiers per
  fold.
- The comparison describes the errors at operating points transferred to the two session groups; it does not
  isolate the effect of the session that the draft v0.3 (§5.2) names as a plausible, unmeasured contributor to the
  high test errors.
- Manifest identifiers are the 16 of pull request #36; here they are only counted.
- The outcomes on the whole test were known, and per-session errors of other checkpoints on the same splits had been
  reported (see the heading).

In the last limit, "the heading" is the heading of `PROTOCOL.md`.

Also from the protocol: the development thresholds are chosen on sessions 1–2 frames of other manifest identifiers; on
the test, sessions 1–2 shares the acquisition sessions of the fitting frames and session 3 does not (§3). No interval,
bootstrap or test is computed; minimum, maximum and signs of the twelve differences take their place (§5, point 7).
The equal-weight mean of the units is not the rate of the pooled test, which weights frames (§5, point 3).

## What was not done

- No training, no model loaded, no image read; no input other than those listed under "Run", the registered
  configuration and `expected_counts.csv`.
- One launch only: the registered scripts were not modified and were not run a second time on the score files.
- No combination chosen after the numbers; no unit excluded (LivenessNet, patience 4, stored statistics, fold 3,
  seed 44 included).
- Not computed (`PROTOCOL.md` §5, point 8): ACER, AUROC or DET by group; sessions 01 and 02 separately; a restriction
  to the manifest identifiers present in both groups; the difference between sessions of the BN change or of the gap
  between detectors; one difference averaged over detectors, patience settings and states; the difference of
  frame-pooled rates.
- The BN re-estimation with augmentation and dropout is a later step and is not part of this run.

## Privacy scan

On 23 November 2026, after the run and before publication, the separate scan of `PROTOCOL.md` §6 was run on every
file of this folder except `SHA256SUMS` and `privacy_scan.txt` (date and time in the first line of
`privacy_scan.txt`). The scan script is kept outside the registered folder: it is not part of pull request #38 and
does not appear in `run.log.txt`, because it reads the `sample_id` and `subject_id` columns of the 48 restricted test
files (no score column). It compares every CSV field, and every token with the file name it contains, with the 16
`subject_id`, the 12,614 `sample_id` and their file names; every maximal digit run not preceded by a decimal point is
compared with the 16 `subject_id` (a `sample_id` or a file name contains letters and underscores, so a digit run
cannot equal one). It also searches for absolute paths, in PDF files on the raw bytes and on the decompressed
streams. Matches: 0. Its output, counts only, is `privacy_scan.txt`.

## Files

`runs/20261123-ARCHSESS-356b041/` holds `conditions.txt`, `run.log.txt` (the `run.log` of the run),
`public_SHA256SUMS.txt` (SHA-256 of the four outputs, written at the end of the run) and `public/` with the four
outputs (`session_counts.csv` 192 rows, `session_summary.csv` 16 rows, `session_table.md`, `session_differences.pdf`),
all copied unchanged from the run folder and compared byte for byte. `SHA256SUMS` lists the nine other files of
this folder.
