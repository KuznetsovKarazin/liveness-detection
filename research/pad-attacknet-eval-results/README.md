# AttackNetV1, AttackNetV2_1 and AttackNetV2_2 against LivenessNet, with ResNet-18 as external reference, on the NUAA test: the files of the evaluation (20261124-ANETEVAL-7c22b1a, PASS)

Card C5, outcome of the evaluation. The coordinator's mail of 23 November 2026: "Potrà quindi eseguire check-inputs e,
se i controlli passano, evaluate una sola volta e consegnare la tabella e la figura di confronto fra le cinque reti"
(check-inputs may then be run and, if the checks pass, evaluate once, and the table and the figure comparing the five
networks delivered). The evaluation code and configuration were published in pull request #39 (commit `0669e86`,
folder `research/pad-attacknet-eval/`) and linked on the C5 card before `check-inputs` and `evaluate`; the comment
with that link announced, as next step, `evaluate` once, "l'esito qui, comunque vada, e poi la tabella e la figura di
confronto fra le cinque reti" (the outcome on the card, however it went, and then the table and the figure comparing
the five networks). This folder is the outcome: the files of the evaluation as written by the command. The table,
the figure, the per-seed appendix, the aggregated CSV and the script for the article are not in this folder
([What is not in this folder](#what-is-not-in-this-folder)).

| asked on the C5 card | where it is in this folder |
|---|---|
| primary analysis: for each network the ACER variation (re-estimated minus ordinary BN, percentage points) at patience 4 and threshold 0.5 | [Primary analysis](#primary-analysis); `public/report.md`; `public/results.json`: `primary_analysis.variations` |
| the three contrasts with LivenessNet, on the same 12 fold-seed pairs and the same subjects, with paired subject bootstrap intervals stated as individual 95% intervals | [Primary analysis](#primary-analysis); `public/report.md`; `public/results.json`: `primary_analysis.contrasts`, `primary_analysis.intervals` |
| patience 12, the EER threshold and the rankings as descriptive analyses | [Descriptive analyses](#descriptive-analyses); `public/results.json`: `descriptive`, `primary_analysis.rankings`; `public/conditions.csv` |
| unfavourable results kept | [Primary analysis](#primary-analysis), [What cannot be concluded](#what-cannot-be-concluded) |
| one-class cases, not evaluable models, denominators | [One-class models and not evaluable models on the test](#one-class-models-and-not-evaluable-models-on-the-test); [The five networks](#the-five-networks-at-patience-4-and-threshold-05); `public/results.json`: `test_one_class`, `not_evaluable`, `primary_analysis.denominators`; `public/table.csv` |

**Status of this delivery.** This folder holds only the records of one evaluation, copied byte for byte from the
author's working repository: `runs/20261124-ANETEVAL-7c22b1a/` with `status.json`, `run.log` and `public/` (`report.md`,
`results.json`, `table.csv`, `conditions.csv`, `units.csv`, `reproduction.csv`, `SHA256SUMS`), and the two conditions
files written by `delivery/aneteval/launch.zsh` of pull request #39 at the launches of `check-inputs` and `evaluate`,
copied unchanged (`conditions.txt` is the file the script wrote as `evaluate-attempt1-conditions.txt`;
`eval-check-inputs-20261124-084002-conditions.txt` keeps its name), besides this `README.md` and the output of the
privacy scan, `privacy_scan.txt` ([Integrity of the files](#integrity-of-the-files)). Nothing of `restricted/` (the per-frame test scores
of the 72 new models, with the frame identifiers); `status.json` records the SHA-256 of its `SHA256SUMS`
(`restricted_sha256sums_sha256`, `95afed119e76…`). No code is in this folder: the code that produced these files
is that of pull request #39 ([Verifying the files against pull request #39](#verifying-the-files-against-pull-request-39)),
which is not merged at the time of this pull request; this folder cites it by commit. No file of the evaluation was edited after the command
ended (`status.json` records the SHA-256 of `run.log` and of `public/SHA256SUMS`; `status.json` itself is covered only by
the `SHA256SUMS` of this folder). Every number below is copied from `public/report.md`, with its rounding, or read from
the file and field named; percentage points are those of the evaluator: "percentage points = proportion x 100, rounded to two decimals half away from zero from the exact value, at presentation only; the computations use exact fractions" (`public/results.json`,
`primary_analysis.presentation`).

## The evaluation

| | value | file and field |
|---|---|---|
| ID, status | `20261124-ANETEVAL-7c22b1a`, **PASS** | `status.json`: `run_id`, `status`, `passed` |
| attempt | 1, no earlier attempt, no second reading | `status.json`: `attempt` (`attempt` 1, `previous` and `earlier_readings` empty, `second_reading` false) |
| official, reading of the test started | true, true | `status.json`: `official`, `test_read_started` |
| started, finished (local time, +01:00) | 2026-11-24 11:12:29, 11:50:08 | `status.json`: `started`, `finished` |
| wall time | 2,259.2 s (37.7 minutes), cap 2.00 hours of wall time (elapsed real time, not CPU time) | `status.json`: `wall_seconds`, `cap_hours`, `cap_unit` |
| process CPU time (all threads) | 19,427.0 s (5.40 hours) | `status.json`: `process_cpu_seconds` |
| models | 72 new models evaluated in both BN states, 0 not evaluable, no non-finite test score; with the 48 C2 reference cells of LivenessNet and ResNet-18 | `public/results.json`: `models`, `not_evaluable`; `status.json`: `not_evaluable`, `non_finite_test` |
| dev reproduced before the test | 144 of 144 model-state pairs (72 models × 2 BN states, 400 dev frames each) within 1e-05, largest absolute difference 0.0, 0 discordant decisions | `public/results.json`: `dev_reproduction`; `status.json`: `dev_reproduction`; `public/reproduction.csv` |
| test frames per fold | 3,242, 3,114, 3,439, 2,819 (12,614) | `public/results.json`: `test_frames_per_fold` |
| threads | 10 intra-op and 10 inter-op, set by the command and read back; `TF_NUM_` variables absent; 15 logical CPUs; equal to the official trial | `status.json`: `threads`, `threads_rule` |
| machine and versions | Apple M5 Pro, 24 GB, macOS 27.2.1, arm64; Python 3.11.16, NumPy 1.26.4, TensorFlow 2.15.0, Keras 2.15.0; equal to the official trial `20261122-ANETTRIAL-a951ed0` | `public/results.json`: `machine`, `environment`; `status.json`: `machine_rule` (`equal` true) |
| commit of the author's working repository, work tree clean | `7c22b1a`, `dirty` false | `status.json`: `commit`, `dirty` |
| configuration | `e5ea7902d665…` | `status.json`: `config_sha256` |

The commit is a commit of the author's working repository, which also holds unpublished work in progress; the link
between this record and the published code is the SHA-256 of the configuration (`config_sha256` of `status.json`) and
of the code (`code_sha256` of `public/results.json`), checked below against the files of pull request #39.
`command_line` (`python -I -B scripts/pad_anet_eval.py evaluate --config delivery/aneteval/pad_anet_eval.json --out-root results/aneteval --runs-root results/anetrun --attempt 1 --i-am-authorized`) is built by the command from the parsed arguments: it writes `--config`,
`--out-root` and `--runs-root` also at their defaults and starts with `python`; the launch line of the script is in
`conditions.txt` (`caffeinate -dims .venv/bin/python -I -B scripts/pad_anet_eval.py evaluate --attempt 1 --i-am-authorized`).

## Order of the times

- Pull request #39 opened on 2026-11-24 at 08:36:10 CET (`createdAt` of the pull request), head `0669e86`.
- Comment with the link to pull request #39 on the C5 card: 2026-11-24 at 08:38:51 CET.
- `check-inputs`, launched through `delivery/aneteval/launch.zsh`: from 08:40:02 to 08:40:26, `RESULT: PASS
  (check-inputs)`, configuration `e5ea7902d665`, work tree clean at commit `95c835b` of the author's working
  repository (`eval-check-inputs-20261124-084002-conditions.txt`).
- Launch of `evaluate` through the same script at 11:12:15: `check-machine` and `check-inputs` again, both PASS,
  then `evaluate --attempt 1`, `started` 11:12:29, `finished` 11:50:08, exit code 0; conditions after the command
  at 11:50:09 (`conditions.txt`; `status.json`).
- Declared in pull request #39 (`README.md`, "Verification without images or weights", and
  `test_logs/check_inputs_20261122_development.txt`): one development run of `check-inputs` on 22 November 2026 at 14:57
  CET, with an earlier version of the code and of the configuration, before the 72 trainings; it assembled no frame.

The two times of pull request #39 and of the comment are those of GitHub and Trello; they are not in a file of this
folder. The two runs of `check-inputs` on 24 November print the same line (same configuration, same
17 pinned code files); between their two commits of the author's working repository only documentation files of that repository
changed (the author's record; not verifiable from this folder).

## What was read, and when

From `public/results.json`, `test_read`, as written by the command: "the NUAA test of the new models is read once, by
evaluate, after every check that does not need it". `check-inputs` assembles no frame and does not import TensorFlow
(its line in both conditions files: "no frame assembled; TensorFlow not imported; nothing written"); it reads the 48 C2
test score files of LivenessNet and ResNet-18, already published as aggregates in the C2 evaluation
`20261121-ARCHEVAL-332fbdf`, to reproduce C2. `evaluate` reads the test frames of the new models only after the C2
reproduction and the dev reproduction of the 72 models: in `run.log` the line "C2 reproduced exactly" comes
first, then the 72 lines "dev reproduced", then the 72 lines "test scored". No test frame of the new
models was read before this run: it is attempt 1 with no earlier attempt (`status.json`, `attempt`), and
`check-inputs` assembles no frame.

## Checks passed before the test of the new models

As recorded by the command (`status.json`: `checks_before_test`, `c2_reproduction`, `dev_reproduction`, `run_records`;
`run.log`):

- configuration unchanged; 18 code files unchanged: the 17 files pinned by the
  configuration and `scripts/pad_anet_eval.py` itself, which the configuration cannot pin (its SHA-256,
  `3892b1b2439c…`, is that of the file of pull request #39, checked below); the 6 runs unchanged
  (each the last attempt, PASS); the records of 7 commands in pull request #39 equal to the runs
  root (46 files compared);
- machine, versions and threads equal to the official trial `20261122-ANETTRIAL-a951ed0`;
- C2 reproduced exactly from the 48 reused score files: the primary analysis, the 8 conditions and
  `units.csv` of C2, and the 48 EER thresholds; the C2 contrast recomputed is 9.98 [2.56, 16.32]
  percentage points with 926 valid resamples; the descriptive comparison of LivenessNet with ResNet-18
  on the C2 units is the exact opposite of the C2 gaps and contrast (`opposite_before_test`, and again after the
  analysis, `public/results.json`: `c2_opposite`);
- the dev of each of the 72 new models reproduced in both BN states (above).

## Primary analysis

Patience 4, threshold 0.5. Variation of a network = ACER with re-estimated BN minus ACER with ordinary BN, per unit (fold
and seed), as exact fractions, mean over the 12 units; contrast of an AttackNet = its variation minus that of
LivenessNet, per unit, on the same 12 fold-seed pairs and the same test frames (`PROTOCOL.md` §6 (a), amended
copy in pull request #37). Subject intervals: percentiles 2.5–97.5 of 1,000 resamples of the 16 test subjects
pooled over the 4 folds, paired (the same resampled subjects for every network and both BN states). "individual 95% intervals: no correction for multiplicity, no simultaneous coverage over the contrasts, variations or comparisons"
(`public/results.json`, `primary_analysis.intervals`). Values in percentage points, from `public/report.md`; units: number
of units with the per-unit value above / below / equal to zero (`public/results.json`, `counts`).

| contrast with LivenessNet (ACER) | mean [individual 95% interval] | valid resamples | units above / below / at zero | reading |
|---|---|---|---|---|
| AttackNetV1 | 15.14 [4.96, 24.05] | 926 | 8 / 4 / 0 | ACER variation of AttackNetV1 minus that of LivenessNet higher than zero |
| AttackNetV2_1 | 13.95 [7.56, 21.63] | 926 | 9 / 3 / 0 | ACER variation of AttackNetV2_1 minus that of LivenessNet higher than zero |
| AttackNetV2_2 | 11.61 [4.46, 18.95] | 926 | 8 / 4 / 0 | ACER variation of AttackNetV2_2 minus that of LivenessNet higher than zero |

| ACER variation (re-estimated minus ordinary BN) | mean [individual 95% interval] | units above / below / at zero |
|---|---|---|
| LivenessNet | -7.07 [-13.33, -0.41] | 7 / 5 / 0 |
| AttackNetV1 | 8.07 [3.47, 11.66] | 8 / 4 / 0 |
| AttackNetV2_1 | 6.87 [3.35, 11.85] | 8 / 4 / 0 |
| AttackNetV2_2 | 4.53 [1.00, 8.82] | 7 / 5 / 0 |
| ResNet-18 | 2.91 [-1.73, 6.63] | 10 / 2 / 0 |

The guardian of the primary analysis passed for every network: effective threshold 0.5 and 12 of 12 pairs evaluable in
both BN states (`public/results.json`, `primary_analysis.guardian`). Valid resamples: 926 of 1,000 for every
ACER interval of this section, 950 for APCER and 940 for BPCER (`valid_resamples` in
`public/results.json`, where the APCER and BPCER of every variation and contrast are also given).

**For each AttackNet the contrast interval lies entirely above zero** (one reading per AttackNet; the protocol fixes
no joint statement on the three, `PROTOCOL.md` §6 (c)). The conclusions written before the test, with the alternative
resolved, as in `public/results.json` (`primary_analysis.contrasts.<network>.conclusion`):

- "On these checkpoints and test frames, the ACER variation with this BN re-estimation (re-estimated minus ordinary BN) is algebraically higher for AttackNetV1 than for LivenessNet; it does not identify a cause."
- "On these checkpoints and test frames, the ACER variation with this BN re-estimation (re-estimated minus ordinary BN) is algebraically higher for AttackNetV2_1 than for LivenessNet; it does not identify a cause."
- "On these checkpoints and test frames, the ACER variation with this BN re-estimation (re-estimated minus ordinary BN) is algebraically higher for AttackNetV2_2 than for LivenessNet; it does not identify a cause."

The five variations are reported with their intervals and no reading (`PROTOCOL.md` §6 (a)).

**Facts from `public/units.csv` and `public/results.json` that bear on these numbers** (a choice of the author,
descriptive, not in `PROTOCOL.md`; means only where no field is named, since subject intervals need the per-subject
counts). The per-unit ACER variation is above zero in 8, 8 and
7 of the 12 units for AttackNetV1, AttackNetV2_1 and AttackNetV2_2, and below zero in
5 of the 12 units for LivenessNet (median of its 12 values 1.28). For each
AttackNet the mean APCER variation is positive (20.17, 23.96,
19.35) and the mean BPCER variation negative (-4.04,
-10.21, -10.29) (`primary_analysis.variations`). The mean variation of
LivenessNet depends on two units: fold 3 seed 44 (ACER 72.65 with ordinary BN and 18.91 with
re-estimated BN, variation -53.73) and fold 1 seed 42 (one of the two one-class models below, variation
-30.12). Without fold 3 seed 44 the mean variation of LivenessNet is -2.83 and the three
contrasts are 8.50, 9.59 and 5.95 (means
only; a computation of the author from the public files, not provided by the protocol). The units against the sign of
the mean contrast are fold 1 seed 43, fold 1 seed 44, fold 4 seed 42 and fold 4 seed 43 for AttackNetV1; fold 1 seed 43, fold 4 seed 43 and fold 4 seed 44 for AttackNetV2_1;
fold 2 seed 43, fold 4 seed 42, fold 4 seed 43 and fold 4 seed 44 for AttackNetV2_2 (the three units of fold 4 among them). The variation of each
AttackNet minus that of ResNet-18 (`primary_analysis.descriptive_vs_resnet18`, descriptive) is
5.15 [-1.11, 11.73], 3.96 [-0.03, 10.39] and 1.62 [-2.72, 6.99] (individual 95% intervals): each
interval contains zero.

## The five networks at patience 4 and threshold 0.5

From `public/table.csv` (exact fractions, converted to percentage points with the rule above): the mean over the
12 units of the per-unit APCER, BPCER and ACER, the number of units, the denominators summed over the units (attack frames
for APCER, bona fide frames for BPCER) and the subjects. This is the evaluator's table, not the table for the article.

| network | BN | APCER | BPCER | ACER | units | attack frames | bona fide frames | subjects |
|---|---|---|---|---|---|---|---|---|
| LivenessNet | ordinary | 15.74 | 58.95 | 37.35 | 12 | 22,527 | 15,315 | 16 |
| AttackNetV1 | ordinary | 30.85 | 36.35 | 33.60 | 12 | 22,527 | 15,315 | 16 |
| AttackNetV2_1 | ordinary | 28.39 | 43.86 | 36.13 | 12 | 22,527 | 15,315 | 16 |
| AttackNetV2_2 | ordinary | 23.11 | 40.82 | 31.96 | 12 | 22,527 | 15,315 | 16 |
| ResNet-18 | ordinary | 33.75 | 8.83 | 21.29 | 12 | 22,527 | 15,315 | 16 |
| LivenessNet | re-estimated | 21.15 | 39.40 | 30.28 | 12 | 22,527 | 15,315 | 16 |
| AttackNetV1 | re-estimated | 51.02 | 32.32 | 41.67 | 12 | 22,527 | 15,315 | 16 |
| AttackNetV2_1 | re-estimated | 52.35 | 33.65 | 43.00 | 12 | 22,527 | 15,315 | 16 |
| AttackNetV2_2 | re-estimated | 42.46 | 30.53 | 36.50 | 12 | 22,527 | 15,315 | 16 |
| ResNet-18 | re-estimated | 36.72 | 11.69 | 24.20 | 12 | 22,527 | 15,315 | 16 |

Variations and contrasts are computed from the exact fractions: a difference of rounded values can differ from them by
0.01 (AttackNetV2_2 variation: 36.50 - 31.96 = 4.54 against 4.53; AttackNetV2_1 contrast from this table: (43.00 - 36.13) - (30.28 - 37.35) = 13.94 against 13.95; AttackNetV2_2 contrast from the rounded variations above: 4.53 + 7.07 = 11.60 against 11.61).

Denominators per fold (`public/results.json`, `primary_analysis.denominators`, the same in both BN states; each fold
counts once per seed, 3 seeds):

| fold | test frames | attack | bona fide | subjects |
|---|---|---|---|---|
| 1 | 3,242 | 1,819 | 1,423 | 4 |
| 2 | 3,114 | 1,744 | 1,370 | 4 |
| 3 | 3,439 | 2,052 | 1,387 | 4 |
| 4 | 2,819 | 1,894 | 925 | 4 |

Summed over the 12 units: 37,842 test frames, 22,527 attack and 15,315 bona fide; each test frame
counts once per seed (12,614 distinct test frames: 7,509 attack and 5,105 bona fide).

## One-class models and not evaluable models on the test

- Not evaluable: 0 (`public/results.json`, `not_evaluable`; `status.json`, `not_evaluable`); no model with non-finite
  test scores (`status.json`, `non_finite_test`).
- Models that decide one class for every test frame (`PROTOCOL.md` §6 (f), a descriptive count per network × patience ×
  BN state × threshold, `public/results.json`, `test_one_class`): 39 of the 40
  cells have none. The only cell with such models is LivenessNet, patience 4, ordinary BN, threshold 0.5:
  2 of 12, both deciding every frame attack (APCER 0, BPCER 1, ACER 1/2 in `public/units.csv`: units
  fold 1 seed 42 and fold 4 seed 44). They are C2 reference cells, reused and not re-scored. No AttackNet and no ResNet-18 model
  decides one class for every test frame, at either threshold, in either BN state, at either patience.
- Every model enters the comparisons, these two included; both are inside the ordinary-BN mean of LivenessNet, and so
  inside its variation and every contrast; nothing was excluded after the test was seen.

## Descriptive analyses

Patience 12 at threshold 0.5, and the dev EER threshold at both patiences: computed as the primary analysis, with
individual intervals and no reading (`PROTOCOL.md` §6 (o); `public/results.json`, `descriptive.pairs`, note
"descriptive: no reading"; `public/conditions.csv`); 926 valid resamples for every ACER interval. Percentage
points, mean [individual 95% interval].

| condition | LivenessNet | AttackNetV1 | AttackNetV2_1 | AttackNetV2_2 | ResNet-18 |
|---|---|---|---|---|---|
| patience 12, threshold 0.5 | -1.15 [-3.67, 3.12] | 8.69 [5.05, 12.59] | 5.44 [2.43, 9.26] | 7.69 [4.03, 11.61] | 5.55 [0.98, 8.02] |
| patience 4, dev EER threshold | -3.90 [-7.86, 0.32] | 2.35 [-1.13, 5.77] | 3.64 [0.62, 7.43] | 2.22 [-1.43, 7.81] | 1.03 [-4.09, 6.61] |
| patience 12, dev EER threshold | 3.19 [-0.33, 4.95] | 2.73 [-0.38, 5.91] | 2.85 [0.49, 5.22] | 3.28 [0.02, 7.87] | 3.76 [0.42, 6.56] |

| condition (contrast with LivenessNet) | AttackNetV1 | AttackNetV2_1 | AttackNetV2_2 |
|---|---|---|---|
| patience 12, threshold 0.5 | 9.84 [3.01, 14.51] | 6.59 [2.95, 11.24] | 8.84 [4.95, 13.14] |
| patience 4, dev EER threshold | 6.25 [0.03, 11.54] | 7.54 [1.14, 13.49] | 6.12 [0.54, 13.32] |
| patience 12, dev EER threshold | -0.46 [-2.80, 3.96] | -0.33 [-3.37, 4.37] | 0.10 [-4.08, 6.39] |

At patience 12 with the dev EER threshold the variation of LivenessNet is positive (3.19) and the three
contrasts are close to zero (-0.46, -0.33, 0.10), each interval containing zero.

The rankings of the five networks by mean ACER in each of the 8 conditions, with the per-unit rank counts, are in
`public/results.json` (`primary_analysis.rankings`, `descriptive.rankings`), descriptive and with no interval
(`PROTOCOL.md` §6 (g)); the descriptive comparison of each network with ResNet-18 (`PROTOCOL.md` §6 (m)) is in
`primary_analysis.descriptive_vs_resnet18` and in the same field of each descriptive pair.

**Per seed** (`PROTOCOL.md` §6 (d): reported apart, outside the reading; `public/results.json`,
`primary_analysis.per_seed`, 4 units per seed, the same 16 subjects, 926 valid resamples for every
ACER interval, individual intervals): the 9 per-seed contrast means are all above zero (from 5.04 to
28.43 percentage points); 4 of their 9 intervals contain zero
(seed 43 AttackNetV1, seed 43 AttackNetV2_1, seed 44 AttackNetV2_1, seed 44 AttackNetV2_2). The per-seed table goes in the appendix of the following delivery.
`primary_analysis.per_seed` also holds, for each seed, the fields `reading` and `conclusion` written by the
evaluator; by `PROTOCOL.md` §6 (d) they are outside the reading and are reported here as values only.

## What cannot be concluded

- A cause: each conclusion says that the result "does not identify a cause"; it is quoted, not extended.
- A joint statement on the three AttackNets: the protocol fixes one conclusion per AttackNet (`PROTOCOL.md` §6 (c)).
- Anything from the five variations taken alone, from the descriptive conditions or from the per-seed values: they have
  no reading (`PROTOCOL.md` §6 (a), (d), (o)).
- An independent validation: this is a post hoc extension of C2 written after the C2 results, and LivenessNet and
  ResNet-18 had already been read on this test (`PROTOCOL.md` §5, §10); in each contrast only the AttackNet side is new.
- Anything beyond these networks, recipe and data (`PROTOCOL.md` §10).

## Limits

`PROTOCOL.md` §10 (amended copy of pull request #37, unchanged from pull request #34), quoted: "Those of C2 (C2 PROTOCOL §10): one recipe; one NUAA arm; dev of 2–3 subjects per fold, on which the EER threshold rests; 16 test subjects pooled over the folds (the 16 subject identifiers of the official NUAA lists, each held out once; the subject bootstrap resamples these identifiers; whether they correspond to 16 or to 15 participants is being clarified with the curators of NUAA, and two identifiers of one participant would be resampled as independent); 3 seeds, paired only nominally (the same seed does not give the networks the same initial weights). In addition: a post hoc extension written after the C2 results; three contrasts and five variations with individual intervals, without correction for multiplicity; LivenessNet and ResNet-18 already read on this test; the recipe not adapted to the AttackNets; CPU trainings not guaranteed bit-deterministic beyond what the control of §7 shows. Any result applies only to these networks, recipe and data."

In addition, for this evaluation:

- **Valid resamples: 926 of 1,000** for every ACER interval (950 for APCER, 940 for
  BPCER). As fixed in `PROTOCOL.md` §6 (a), a resample is dropped for a metric if a unit's value is undefined; at every
  resample the subjects are the same for every network and both BN states, and in this record every ACER interval,
  primary, descriptive and per seed, has 926 valid resamples.
- The subject intervals cannot be recomputed from public files: they need per-subject counts, which are not public.
  Neither can the dev reproduction and the C2 reproduction, which need the checkpoints, the frames and the restricted
  C2 test scores; they are recorded in `status.json`, `run.log` and `public/reproduction.csv`. The means, the numbers
  of units above and below zero and the one-class counts can be recomputed from `public/units.csv` (exact fractions
  per unit, written reduced, for example `0/1`: the denominators are not the frame counts; the frame counts are in
  `n_frames` and in `primary_analysis.denominators`).
- Process CPU time 5.40 hours against the estimate of 4.50 CPU hours for the evaluation in `PROTOCOL.md` §8;
  wall time 0.63 hours against the estimate of 0.59 hours; the cap is on wall time (2.00 hours).
- Choices of the author that enter these numbers: the closed list of `PROTOCOL.md` §9 (also `choices_of_the_author` of
  the configuration), among them the sign of the contrasts (AttackNet minus LivenessNet), the reading of each contrast
  and its conclusions, the descriptive count of the one-class models on the test, percentage points with two decimals.

## What is not in this folder

- The material for the article asked on the card: the table of the five networks with APCER, BPCER, ACER and
  denominators, the figure of the ACER variations with their intervals, the per-seed results and the one-class counts in
  an appendix, the aggregated CSV and the script that regenerates table and figure from the public files. They follow in
  a separate pull request.
- The per-frame test scores of the new models (`restricted/`): kept in the restricted repository already used for C2.
- The records of the six runs and of check-estimator: published in pull request #39 (`runs/`).

What was not done: no training; no second reading of the test; no model excluded; no file of the evaluation edited.

## Verifying the files against pull request #39

From the root of the repository, on the branch of this pull request, after fetching the head of pull request #39
(`upstream`: the remote of `KuznetsovKarazin/liveness-detection`; the fetch gives commit
`0669e86354caddd44219f3766f6f33b77fe3cf49`). The paths in `code_sha256` are relative to `livenesslab/` of
`research/pad-attacknet-eval/`.

```bash
git fetch upstream pull/39/head
D=research/pad-attacknet-eval-results/runs/20261124-ANETEVAL-7c22b1a
L=research/pad-attacknet-eval/livenesslab
python3 -I -c 'import json,sys; s=json.load(open(sys.argv[1])); r=json.load(open(sys.argv[2])); c=dict(r["code_sha256"]); c["delivery/aneteval/pad_anet_eval.json"]=s["config_sha256"]; [print(v+"  "+k) for k,v in sorted(c.items())]' $D/status.json $D/public/results.json |
while read -r h f; do [ "$(git show 0669e86:$L/$f | shasum -a 256 | cut -c1-64)" = "$h" ] && echo "OK $f" || echo "DIFFERENT $f"; done
```

Expected: 19 lines `OK` (the 18 code files and the configuration), none `DIFFERENT`.

The other inputs recorded in `inputs_sha256` of `public/results.json`, at the paths where pull request #39 holds them:

```bash
python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1]))["inputs_sha256"]; p={"run_configuration": "delivery/anetrun/pad_anet_run.json", "comparison": "delivery/anetcmp/pad_anet_comparison.json", "protocol": "delivery/pad-attacknet-run/pr34_amended/PROTOCOL.md", "c2_units": "delivery/pad-arch-eval-results/runs/20261121-ARCHEVAL-332fbdf/public/units.csv", "c2_results": "delivery/pad-arch-eval-results/runs/20261121-ARCHEVAL-332fbdf/public/results.json"}; [print(d[k]+"  "+v) for k,v in sorted(p.items())]' $D/public/results.json |
while read -r h f; do [ "$(git show 0669e86:$L/$f | shasum -a 256 | cut -c1-64)" = "$h" ] && echo "OK $f" || echo "DIFFERENT $f"; done
```

Expected: 5 lines `OK` (the remaining key of `inputs_sha256`, `configuration`, equals `config_sha256` and is checked by the
first block).

The six runs read by the evaluation are those published in `runs/` of pull request #39 (`runs` of `status.json`: the
SHA-256 of their `status.json`):

```bash
python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1])); [print(r["status_sha256"]+"  "+r["run_id"]+"/status.json") for r in d["runs"].values()]' $D/status.json |
while read -r h f; do [ "$(git show 0669e86:research/pad-attacknet-eval/runs/$f | shasum -a 256 | cut -c1-64)" = "$h" ] && echo "OK $f" || echo "DIFFERENT $f"; done
```

Expected: 6 lines `OK`.

## Integrity of the files

From this folder:

```bash
shasum -a 256 -c SHA256SUMS
```

Expected: 13 lines `OK`. From `runs/20261124-ANETEVAL-7c22b1a/public/`, `shasum -a 256 -c SHA256SUMS` gives 6 lines
`OK` (the file written by the command). From `runs/20261124-ANETEVAL-7c22b1a/`, the SHA-256 of `run.log` and of `public/SHA256SUMS` are
those recorded in `status.json` (`run_log_sha256`, `e38fbb76f75b…`; `public_sha256sums_sha256`, `1f6815b08c04…`):

```bash
shasum -a 256 run.log public/SHA256SUMS
python3 -I -c 'import json; d=json.load(open("status.json")); print(d["run_log_sha256"]); print(d["public_sha256sums_sha256"])'
```

`privacy_scan.txt` is the output of the author's scan of the 12 files of this folder other than `SHA256SUMS` and
itself, before publication: every CSV field and every token compared with the 16 subject identifiers, the
12,614 test frame identifiers and their file names, every maximal run of digits compared with the subject
identifiers (sets read from the restricted C2 test score files, which hold the same test frames), and absolute paths. It
ends with "identifier and path matches: 0" and prints counts only, never an identifier.

## Contents

14 files: 13 listed in `SHA256SUMS`, plus `SHA256SUMS`. One file matches the `.gitignore` of the repository (`*.log`):
`runs/20261124-ANETEVAL-7c22b1a/run.log`; it is committed explicitly (`git ls-files research/pad-attacknet-eval-results | wc -l`
gives 14).

| path | what it is |
|---|---|
| `README.md` | this file |
| `privacy_scan.txt` | the output of the privacy scan (above) |
| `runs/20261124-ANETEVAL-7c22b1a/status.json` | the final status of the evaluation |
| `runs/20261124-ANETEVAL-7c22b1a/run.log` | its log (no number of the analysis) |
| `runs/20261124-ANETEVAL-7c22b1a/conditions.txt` | the conditions of the launch of `evaluate` (written by the launch script, not by the evaluation) |
| `runs/20261124-ANETEVAL-7c22b1a/eval-check-inputs-20261124-084002-conditions.txt` | the conditions of the launch of `check-inputs` before `evaluate` |
| `runs/20261124-ANETEVAL-7c22b1a/public/report.md` | the report: contrasts and variations of the primary analysis |
| `runs/20261124-ANETEVAL-7c22b1a/public/results.json` | every result, with full precision: primary analysis, descriptive conditions, rankings, per seed, C2 reproduction, one-class counts on the test, denominators |
| `runs/20261124-ANETEVAL-7c22b1a/public/table.csv` | the five networks at patience 4 and threshold 0.5 in both BN states, exact fractions, with denominators |
| `runs/20261124-ANETEVAL-7c22b1a/public/conditions.csv` | variations and contrasts of the four pairs of conditions, with intervals |
| `runs/20261124-ANETEVAL-7c22b1a/public/units.csv` | per condition, unit and network: threshold, APCER, BPCER and ACER as exact fractions, the count of test frames and the SHA-256 of their sorted identifiers |
| `runs/20261124-ANETEVAL-7c22b1a/public/reproduction.csv` | the dev reproduction per model and BN state |
| `runs/20261124-ANETEVAL-7c22b1a/public/SHA256SUMS` | SHA-256 of the 6 public files above |
| `SHA256SUMS` | SHA-256 of every other file of this folder |

## What follows

The outcome on the C5 card, with the link to this pull request; then, in a separate pull request, the table, the figure,
the per-seed appendix, the aggregated CSV and the script for the article. No new reading of the NUAA test and no
training are in progress.

## Declarations

The NUAA test of the new models was read once, by this evaluation (attempt 1); no training; no external dataset used;
no restricted file in this folder; no file of the evaluation edited after the command ended.
