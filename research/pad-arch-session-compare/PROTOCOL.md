# Session comparison of the 48 checkpoints at the registered development thresholds

Card C2. Version 1.0, written on 22–23 November 2026 on the research branch at `19605ee` (after pull requests #35 and
#36). **Registration before any computation:** no APCER or BPCER has been computed by session group on the score files
of the 48 checkpoints, by this script or by any other program. The outcomes on the whole test (pull request #36, Table
A3 of the draft v0.3) were known when this protocol was written; so were the per-session test errors of other
LivenessNet checkpoints on the same splits in earlier deliveries
(`research/nuaa-patience-verification/tables/aggregate.md`, `research/nuaa-bn-test/README.md`,
`research/n2-nuaa-check/run-report-sessions.md`); the per-session class counts of the official NUAA lists were published
in pull request #35 (`research/pad-arch-score-followup/NOTES.md`: the train lists are sessions 01–02, the test lists
session 03), so the group sessions 1–2 coincides with the official train lists. The comparison is therefore not blind to
the phenomenon.

References: **SA** = `research/pad-arch-score-followup/score_analysis.py` at commit
`fe87b3d0012f0de7c59528caa6bfe4bd1e3e27f0` (pull request #35); **SM** and **PS** = `score_metrics.csv` and
`paired_summary.csv` in `research/pad-arch-score-results/runs/20261122-ARCHSCORE-fe87b3d/public/` (pull request #36),
SHA-256 `1397786e065efff2214757ae34c90f8984e1d926780593987756705d48331bdd` and
`f59e52da1303ed3241b448a97dc53491116d7b3873c2b41cd0c4c67e4c7590c3`.

## 1. Request and reading

Requested by the coordinator: for the 48 checkpoints and the two BN states, reuse the development thresholds already
registered (development APCER target 5%, then 10%); apply them identically to the test frames of sessions 1–2 and to
those of session 3; for each group report the counts per class and APCER/BPCER; verify that pooling the counts of each
fold and seed gives back the values of pull request #36; summarise with the paired difference between sessions, one
table and one figure, with equal weight for the twelve fold–seed pairs; a small difference is reported as well.

- Units: 2 detectors (LivenessNet, ResNet-18) × 2 patience settings (4, 12) × 4 folds × 3 seeds (42, 43, 44) = 48
  checkpoints; × 2 BN states (stored, re-estimated) = 96 checkpoint-states, the 96 rows of SM; × 2 targets = 192
  operating points. A combination is detector × patience × BN state × target (16); each has 12 fold–seed units.
- "Then 10%" is read as the second target, reported in full with the same structure as a descriptive result. It is
  not a fallback to use if the 5% target does not work, nor an alternative to choose afterwards.
- ACER is not requested and is not reported.

## 2. Inputs

- Restricted test scores of the evaluation `20261121-ARCHEVAL-332fbdf`: 48 files
  `p{4,12}_<detector>_fold<k>_seed<s>_test_scores.csv`, each with the two BN states. Restricted development scores of
  the training runs `20261120-ARCHRUN-p4-7faa49f` and `20261120-ARCHRUN-p12-ea7fdca`: 96 files
  `<detector>_fold<k>_seed<s>_dev_{ordinary,re-estimated}.csv` (names as in SA). Columns read: `network`, `fold`,
  `train_seed`, `bn_state`, `role`, `sample_id`, `subject_id`, `session_id`, `label`, `score_attack`,
  `prediction_05`, `checkpoint_sha256`, and `patience` in the test files. The development files serve checks C2–C4,
  C7 and C10; no error is computed on them.
- SM and PS (public).
- `session_config.json` (registered configuration: SHA-256 of SM, of PS and of `expected_counts.csv`, the two
  groups, the two targets, the highlighted outcome) and `expected_counts.csv` (the 16 denominators of section 3 with
  the number of manifest identifiers per cell), both in this folder. The SHA-256 of SM is not a constant of the code
  nor a free argument. `session_compare.py` and `session_summary.py` read by default the configuration next to
  themselves; `--config` exists for `check_synthetic.py` only. The launch wrapper of the registered run does not pass
  `--config`, validates `SHA256SUMS` of this folder (configuration included) and writes the SHA-256 of the
  configuration in `conditions.txt`; each script prints the SHA-256 of the configuration it read as the first line of
  its log, and the wrapper checks that it is the registered one.

## 3. Groups

- The session of a test frame is the column `session_id`, checked against the session field of the original file
  name, which `sample_id` carries in the form `<class>/<ID>_<ID>_<glasses>_<pos>_<session>_<pic>.jpg`.
  **sessions 1–2** = `session_id` `01` or `02`; **session 3** = `session_id` `03` (exact strings).
- For bona fide and photograph frames alike the session is the session field of the original file name, in the same
  position. What this field represents for a printed photograph is a matter of the NUAA documentation and is not
  verified here.
- In the original NUAA lists the train lists hold sessions 01 and 02 only and the test lists session 03 only; in the
  folds every frame is in the test of one fold. Fitting and development of all 48 models use frames of sessions 01
  and 02 only (in each of the 96 development files: 100 frames per session and class). The development thresholds are
  therefore chosen on sessions 1–2 frames of other manifest identifiers; on the test, sessions 1–2 shares the
  acquisition sessions of the fitting frames and session 3 does not.

Test denominators (frames; manifest identifiers in brackets), the same in all 24 conditions of a fold
(`expected_counts.csv`):

| Fold | Photograph, sessions 1–2 | Photograph, session 3 | Bona fide, sessions 1–2 | Bona fide, session 3 |
|---|---|---|---|---|
| 1 | 587 (3) | 1,232 (3) | 498 (3) | 925 (2) |
| 2 | 380 (2) | 1,364 (4) | 351 (2) | 1,019 (3) |
| 3 | 387 (2) | 1,665 (4) | 522 (2) | 865 (2) |
| 4 | 394 (2) | 1,500 (4) | 372 (2) | 553 (2) |
| All | 1,748 | 5,761 | 1,743 | 3,362 |

Both groups contain both classes in every fold. The two manifest identifiers with one class only (472 bona fide
frames in fold 1, 468 photograph frames in fold 3) are entirely in session 3.

## 4. Thresholds and decision rule

- For each checkpoint-state and target the threshold is `<target>_threshold` of SM, read as a float. It is not
  selected again. SA chose it on the development scores of the same checkpoint and state as the highest candidate
  threshold with development APCER not above the target (candidates: the distinct development scores and the value
  just below the minimum).
- Decision rule of SA: attack if `score_attack` > threshold. A photograph frame is an error if its score is ≤ the
  threshold (APCER), a bona fide frame if its score is > the threshold (BPCER); a score equal to the threshold is
  bona fide. The same threshold, passed to one counting function, is applied to the two groups.
- Ties (author's choice, diagnostic): the frames of each class whose score equals the threshold are counted and
  reported, so that the effect of the direction of the inequality is visible. They are counts, not errors.

## 5. Estimates

1. Per operating point and group: frames per class, photograph errors, bona fide errors, ties per class, APCER =
   photograph errors / photograph frames, BPCER = bona fide errors / bona fide frames.
2. Paired difference, for each combination and fold–seed unit: d = APCER(session 3) − APCER(sessions 1–2), and the
   same for BPCER. Positive means more errors on session 3. Paired, because both groups are scored by the same
   checkpoint, state and threshold.
3. Summary per combination: the mean of the twelve differences with weight 1/12 each, in exact rational arithmetic
   with one final rounding to a float; the twelve-unit means of APCER and BPCER of each group; minimum and maximum of
   the twelve differences; the number of units with a difference > 0, = 0 and < 0, decided on integer counts
   (photograph errors of session 3 × photograph frames of sessions 1–2 against photograph errors of sessions 1–2 ×
   photograph frames of session 3, and the same for bona fide). The equal-weight mean of the units is not the rate
   of the pooled test, which weights frames; pooling serves checks C9 and C12 only.
4. **Highlighted outcome (author's choice; the request names none):** patience 4 (the primary analysis of C2) and
   target 5% (the first one named), for both detectors and both BN states, APCER and BPCER separately: 8 means, the
   4 rows marked in the table. Everything else (patience 12, target 10%) is reported in full with the same structure,
   as descriptive. No combination is chosen after the numbers are seen.
5. No exclusion: all 16 combinations and all 12 units, including LivenessNet, patience 4, stored statistics, fold 3,
   seed 44. The variant without that unit in pull request #36 was a post hoc diagnostic and is not repeated.
6. Undefined quantities (fixed before the numbers; with the verified denominators of section 3 it cannot occur,
   because check C6 stops the script on a denominator 0): a rate with denominator 0 is written empty with its count
   0, never as 0 or 1. If any of the twelve differences of a combination is undefined, the mean of that combination
   is not reported: the cell is empty and gives the number of defined units, because a mean of fewer than twelve
   units does not give equal weight to the twelve pairs. Discarded alternatives: a mean over the defined units only,
   substituting 0 or a bound, pooling counts across folds, imputing from another seed.
7. No interval, bootstrap or test: the request asks for the paired difference; with 2–3 bona fide manifest
   identifiers per fold in sessions 1–2 and different compositions of the two groups (section 10) an interval by
   identifier would not be defensible. In their place: minimum, maximum and signs of the twelve differences.
8. Not computed (alternative readings of the estimand, all derivable from the public `session_counts.csv` if
   requested): the difference between sessions of the BN change (re-estimated − stored) or of the gap between
   detectors; one difference per metric averaged over detectors, patience settings and states; the difference of
   frame-pooled rates. Not done: ACER, AUROC or DET by group, sessions 01 and 02 separately, a restriction to the
   manifest identifiers present in both groups.

## 6. Outputs (aggregate only)

| File | Rows | Content |
|---|---|---|
| `session_counts.csv` | 192 | key (patience, fold, seed, network, bn_state, target); the threshold written for each group (`repr` of the float used, equal to the string of SM); per group frames, errors and ties per class, APCER, BPCER; pooled APCER and BPCER; the test APCER and BPCER of SM. |
| `session_summary.csv` | 16 | per combination: highlighted, units, defined units, the means of each group, mean, minimum and maximum difference, units up/zero/down, for APCER and BPCER; proportions. |
| `session_table.md` | 16 and header | the table of section 5, in the order of Table A3 of the draft (patience, detector, target, then stored and re-estimated as two consecutive rows); percentage points with two decimals (conversion × 100), each value rounded once from its exact rational value with the rule half to even; highlighted rows in bold; every value is rounded separately, so a difference shown can differ by 0.01 from the difference of the two values shown. |
| `session_differences.pdf` | — | 2 rows (APCER and BPCER difference, percentage points) × 4 columns in this order: patience 4 LivenessNet, patience 4 ResNet-18, patience 12 LivenessNet, patience 12 ResNet-18 (the order of the rows of Table A3); in each panel the 5% stored, 5% re-estimated, 10% stored and 10% re-estimated positions, the twelve unit points (fixed horizontal offset per fold, no random jitter), the mean as a dash, the zero line; colours of the two states as in `det_curves.pdf`; no date metadata. |
| `run.log`, `conditions.txt`, `public_SHA256SUMS.txt` | — | as in pull request #36; `conditions.txt` with the commits, the SHA-256 of this folder and of the configuration, versions and times. |

`session_summary.py` reads public files only (`session_counts.csv`, SM, PS, the configuration and
`expected_counts.csv`), repeats checks C1, C6, C8, C9, C10 and C12 and regenerates the table and the figure. It
repeats C6 on the frame counts only, because `session_counts.csv` carries no identifier counts. Each script builds its
outputs completely before writing them, so that a failure (of the figure, for instance) leaves no partial output. No
per-frame score, frame identifier or subject identifier leaves the restricted files. Before the results are
published, a separate privacy scan makes an exact comparison of every CSV field, and of every maximal digit run not
preceded by a decimal point, with the set of 16 `subject_id` and 12,614 `sample_id` (and the file names they
contain); only the number of matches is printed, and it must be 0. The same scan searches the outputs and the log for
absolute paths.

## 7. Checks

Each check stops the script with exit code 4 and one line naming the check and a key of fixed values (fold,
detector, seed, state, target), never an identifier; nothing is written if any check fails. An unexpected error
stops it with exit code 5 and its type only; exit code 2 is left to argument errors. Order: first the public inputs
and the output folder (C11, C1, the PS part of C12, the `expected_counts.csv` part of C6, the existence of the input
files in C10), before any restricted score file is opened; then, file by file, C2, C4, C5, C3 and C7; then C10 on the
rows, C6, C9, C8, C12 and C11 again before writing.

| # | Check | How |
|---|---|---|
| C1 | SM is the published file | SHA-256 of SM = `metrics_sha256` of the configuration; 96 rows, unique keys, the 96 expected. |
| C2 | Row metadata | as SA `prepare`: detector, fold, seed, state and role equal to the file name; no duplicated `sample_id`; `prediction_05` = (score > 0.5); scores finite in [0, 1]; labels 0 or 1; in a test file, only the two BN states and the patience of the file name. |
| C3 | Pairing | within a fold, the ordered tuple (`sample_id`, `subject_id`, `label`, `session_id`) of the test, and (`sample_id`, `subject_id`, `label`) of the development partition, identical across detectors, seeds, patience settings and states (row by row); development and test share no `sample_id` and no `subject_id`, and the development partition has 400 frames (as SA line 211). |
| C4 | Checkpoint SHA-256 | one per model, the same in the two states and in the development and test files (as SA). |
| C5 | Session | `session_id` among the configured values, exact strings; equal to the session field of `sample_id`; class prefix of `sample_id` consistent with `label`; `subject_id` equal to both identifier fields. |
| C6 | Denominators | `expected_counts.csv` readable, with the SHA-256 of the configuration and the 16 cells (before the score files); frames and manifest identifiers per fold × class × group equal to it (exact); every denominator > 0. |
| C7 | Thresholds | re-selection from the development scores with `profile` and the selection of SA copied unchanged: threshold, development APCER and development BPCER equal to SM (development BPCER in the form `1.0 − k/200` of SA), 192 thresholds. |
| C8 | Application | one counting function serves the two groups; the threshold written for each group is `repr` of the float passed to it; before writing, and again in `session_summary.py`, the two thresholds written are equal to each other and to the string of SM (192 rows). |
| C9 | Pooling = pull request #36, per checkpoint-state | for the 192 operating points: the pooled denominators are the fold totals; the pooled photograph errors e equal round(APCER of SM × n), with \|APCER × n − e\| < 1e-6 and e / n == APCER of SM (float equality); the same for BPCER; 384 equalities, zero tolerance on counts. |
| C10 | Completeness | the 48 test and 96 development files exist (before they are opened); 16 combinations × 12 fold–seed units, each once, with both groups. |
| C11 | Output folder | new or empty; writing there only. |
| C12 | Pooling = pull request #36, means | SHA-256 of PS equals `ps_sha256` in the configuration (before the score files); then, for the 16 combinations, the 12-unit mean of the pooled rates, computed as in SA (`np.array` in (fold, seed) order, `.mean()`), equals `stored_mean`/`reestimated_mean` of PS (32 values). On the real inputs this check is implied by C9 together with the published consistency of PS with SM (32/32 verified before registration); it is kept because the request names the values of pull request #36. |

Confidentiality of messages: every exception is caught; the log receives the check name and the key only, or the
type of an unexpected error, never a traceback; library warnings are silenced, because they print source paths. The
full traceback is written only to `results/archsess/<run>/restricted/traceback.txt`, which is never copied to the
delivery; `run.log` receives the check name and key only.

`check_synthetic.py` runs both scripts on invented scores (ties at the thresholds in both classes and groups, an
operating point and a summary computed by hand, development scores where `1.0 − k/200` differs from `(200 − k)/200`,
identifiers in one group only and in both). It compares every count, every cell of the table and every value of the
figure with an independent computation, checks the rule for undefined quantities in the summary, the table, the CSV
and the figure, and runs stop cases for every check and sub-check, each built so that this check fails and no other.
`check_mutations.py` applies 56 mutations one at a time to copies of the scripts: removing each of C1–C12; `<` for
`<=` in APCER; `>=` for `>` in BPCER; threshold 0.5; threshold of one group from the other BN state; from the other
target; the two groups swapped; `03` read as sessions 1–2; `01` read as session 3; frame-weighted mean; development
BPCER as `(200 − k)/200`; difference sessions 1–2 − session 3; a zero difference counted as up; and 32 more added
after the independent review of the code (cells, rounding, signs and labels of the table, values of the figure, each
sub-check of C1–C7 and C12, swapped classes and metrics). Each must make the synthetic check fail, and each removal of
a check must be caught first by a stop case of that check.

## 8. Failed run

1. If a check C1–C12 fails on the real inputs after the script has opened a restricted score file, no output is read
   or published (the script writes none); the restricted log is kept and the card reports which check stopped the
   run, without interpretation.
2. The registered script is not modified: a correction goes into a new dated commit with an erratum describing the
   cause and the change, linked on the card before a second launch; the second launch uses a new folder and is
   declared as the second one, the first one being kept.
3. A stop before any restricted score file is read does not count as a run: a stop of the launch wrapper (tree not
   clean, invalid manifests, existing run folder, configuration SHA-256 other than the registered one, a failed
   synthetic check) or a stop of the script on its first checks (the configuration; C11; C1; the PS part of C12; the
   `expected_counts.csv` part of C6; missing input files in C10). The condition is corrected and the launch repeated,
   which is declared in `run.log`.

## 9. Order

1. This folder is committed with a date, opened as a pull request and linked on the C2 card before any computation.
2. Then, once, from the committed folder and a clean tree: `check_synthetic.py`; `session_compare.py` once on the
   score files of the 48 checkpoints, with a new output folder; `session_summary.py`; SHA-256 of the outputs; privacy scan.
3. Results with the reproduction checks first, then the highlighted outcome, the table, the figure and the limits;
   a small difference is reported as such.

## 10. Limits

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

## 11. Author's choices

Fixed by the request: the 48 checkpoints and two BN states, the registered thresholds at 5% and 10% (those of SM, not
re-selected; C7 is a check only), the two groups sessions 1–2 and session 3 with the same threshold, counts per class
and APCER/BPCER per group, the pooling check against pull request #36, the paired difference with equal weight for the
twelve fold–seed pairs, one table and one figure, reporting a small difference. Taken from the script of pull request
#35 (SA): the decision rule, attack if score > threshold, so that a score equal to the threshold is bona fide.
Author's choices, each of which can move a number:

1. The session from `session_id`, checked against the file name (sessions 1–2 = `01` and `02`, session 3 = `03`).
2. Direction of the difference: session 3 − sessions 1–2.
3. Mean of the twelve units in exact arithmetic; no exclusion.
4. Highlighted outcome: patience 4, target 5%, both detectors and states, APCER and BPCER separately; "then 10%" as
   the second target, reported.
5. Undefined quantities: empty with counts; no mean if a unit is undefined.
6. No interval; minimum, maximum and signs as descriptives.
7. Presentation: percentage points, two decimals rounded half to even from the exact value, order of Table A3 for
   the table and the figure; signs decided on integers.
8. No ACER.

Ties at the threshold are counted and reported (a diagnostic; no rate changes).
