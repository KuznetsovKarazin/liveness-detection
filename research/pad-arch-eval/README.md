# LivenessNet and ResNet-18 on NUAA: the evaluation code on the test, published before the test is read, and the outcomes of the two runs of 24 trainings

Asked by the coordinator on the C2 card on 19 November 2026, in the evening, after merging PR #29 (merge `362c653` into
`research/pad-2026`): the command and the configuration of the two runs first, in a new pull request (PR #30), the link
on the card, then the runs; "Pubblichi il codice di valutazione prima di leggere il test" (the evaluation code is to be
published before the test is read); for the primary analysis, a check is to be added that stops it if the effective
threshold differs from 0.5 for one of the networks or if one of the 12 fold and seed combinations is missing in one of
the two BN states; at the end of the runs, the two ACER gaps, their contrast with the subject interval and the per-seed
results are to be reported on C2. On 20 November 2026 at 21:45, on the C2 card, after verifying PR #30, the coordinator
asked to report there, once the outcomes are available, how many of the 36 checks passed and, for each of the two runs,
its ID, its status and the number of trainings completed out of 24, and repeated that the evaluation code is to be
published before the test is read.

| request | where it is in this folder |
|---|---|
| the evaluation code published before the test is read | `livenesslab/scripts/pad_arch_eval.py` and its configuration `livenesslab/delivery/archeval/pad_arch_eval.json`; status below |
| the check that stops the primary analysis (effective threshold, 12 fold × seed combinations per BN state) | [The check requested by the coordinator](#the-check-requested-by-the-coordinator) |
| how many of the 36 checks passed; for each run its ID, status and trainings completed out of 24 | [Outcomes of the three official commands](#outcomes-of-the-three-official-commands) and `runs/` |
| the two ACER gaps, their contrast with the subject interval, the per-seed results | computed by `evaluate` (not yet run); [Outputs](#outputs) |

**Status of this delivery.** This folder holds the evaluation command (`livenesslab/scripts/pad_arch_eval.py`), its
configuration, its synthetic test suite and mutation check, their logs, and the records of the three official commands
of PR #30 (`runs/`: `status.json`, `run.log` and `public/` of each; nothing of `restricted/`). **`evaluate` has not been
run: no test frame has been assembled or scored, and no test score or metric exists.** The frame files (the cache and
the extracted frames file), which also hold frames of the held-out subjects, have been read whole by the commands of
PR #30 and by `check-inputs` (by `check-inputs`: the cache hashed whole, with its label and subject members read, and
the extracted frames file loaded whole); of their frames only fitting and dev frames have been assembled, by the
commands of PR #30 (`check-inputs` assembles none; [What the commands read](#what-the-commands-read)). `evaluate` is
launched only after this pull request is published and its link is on the C2 card. Of this command, only what the
read-only `check-inputs` does has been run on the real inputs: the delivered code on the two run folders
(`test_logs/check_inputs_archeval.txt`) and, before it, during development and review, `check-inputs` (also without the
run folders) and the checks of the run folders (`load_run`). Nothing was run on Linux by the author.

## Outcomes of the three official commands

The three commands of PR #30 (configuration `livenesslab/delivery/archrun/pad_arch_run.json`, SHA-256
`0df7a2ddb16d436a7552169c5ff332e44759237020e8c466077986b92d0d734d`), all official and at attempt 1. Each number below is
read from the file and field named in its row; the files are in `runs/<run id>/`.

| | value | file and field |
|---|---|---|
| checks of the estimator passed (36 accepted checkpoints of run `20261114-BNRECAL-a10a082`) | **36 of 36** | `runs/20261120-ARCHEST-6c4f9e8/status.json`: `models_checked` 36, `models_reproduced` 36; `public/estimator_check.json`: `models`, 36 entries with `recalibration_pass` true (also `statistics_pass` and `dev_pass` true) |
| check-estimator: ID, status | `20261120-ARCHEST-6c4f9e8`, **PASS** | `status.json`: `run_id`, `status`, `passed` |
| run at patience 4: ID, status | `20261120-ARCHRUN-p4-7faa49f`, **PASS** | `runs/20261120-ARCHRUN-p4-7faa49f/status.json`: `run_id`, `status` |
| run at patience 4: trainings completed | **24 of 24** | `status.json`: `complete_phase_1` (24 names), `missing_phase_1` (empty); `public/summary.json`: `positions` 24, `complete_phase_1` 24 |
| run at patience 12: ID, status | `20261120-ARCHRUN-p12-ea7fdca`, **PASS** | `runs/20261120-ARCHRUN-p12-ea7fdca/status.json`: `run_id`, `status` |
| run at patience 12: trainings completed | **24 of 24** | `status.json`: `complete_phase_1` (24 names), `missing_phase_1` (empty); `public/summary.json`: `positions` 24, `complete_phase_1` 24 |

Then, from the same files:

| | check-estimator | run at patience 4 | run at patience 12 | file and field |
|---|---|---|---|---|
| started, finished (local time, +01:00) | 2026-11-20 19:01:05, 19:26:50 | 2026-11-20 19:27:15, 21:58:15 | 2026-11-20 22:03:54, 2026-11-21 00:53:14 | `status.json`: `started`, `finished` |
| wall time | 1,544.5 s (cap 240 minutes) | 9,059.6 s, 2.52 h of the 5.00-hour cap | 10,160.2 s, 2.82 h of the 5.00-hour cap | `status.json`: `wall_seconds`, `cap_minutes` or `cap_hours` |
| process CPU time | 7,502.4 s | 76,185.7 s | 88,269.9 s | `status.json`: `process_cpu_seconds` |
| BN states evaluated on the dev (phase 2) | – | 24 of 24 | 24 of 24 | `status.json`: `complete_phase_2`; `evaluator_mismatches` empty |
| LivenessNet histories identical to the accepted run at the same patience | – | 12 of 12 | 12 of 12 | `public/models.csv`: `history_check` (`identical` for the 12 LivenessNet rows, `not applicable` for ResNet-18); `public/summary.json`: `models[].history_check.outcome` (`history_check` is null for ResNet-18) |
| machine and threads | equal to the official trial; 10 and 10 set and read back; variables absent | the same | the same | `status.json`: `machine_rule`, `threads`, `threads_rule` |
| commit of the author's working repository, work tree clean | `6c4f9e8`, `dirty` false | `7faa49f`, `dirty` false | `ea7fdca`, `dirty` false | `status.json`: `commit`, `dirty` |

The commits are commits of the author's working repository, which also holds unpublished work in progress; the link
between a record and the published code is the SHA-256 of the configuration (`config_sha256`) and of the code
(`code_sha256` of the two runs; `code_sha256` of `public/estimator_check.json`), to be compared with the files of
`livenesslab/`. Integrity of the records: in each run folder `public/SHA256SUMS` lists the public files
(`shasum -a 256 -c SHA256SUMS` in `public/`), and `status.json` records the SHA-256 of `run.log` (`run_log_sha256`) and of
`public/SHA256SUMS` (`public_sha256sums_sha256`); for check-estimator `status.json` records the SHA-256 of `run.log` and
of the two public files (`estimator_check_json_sha256`, `estimator_check_csv_sha256`). The restricted files of the runs
(checkpoints, cards, per-frame dev scores, BN statistics) are not in this folder; `status.json` records the SHA-256 of
their `SHA256SUMS` (`restricted_sha256sums_sha256`).

**Conditions in which the times were measured** (they can change the times; for the other numbers: the threads were
fixed at 10 and 10, the 24 LivenessNet histories are identical to the accepted ones and the 36 estimator checks
reproduce run `20261114-BNRECAL-a10a082`; for ResNet-18 there is no earlier history to compare with):

- the machine was set to low power mode during check-estimator (inferred, not observed during check-estimator: the
  machine was on mains power, for which low power mode was set, and the setting was changed only at 19:36:37) and during
  the first 9 minutes or so of the run at patience 4 (observed at about 19:33, removed at 19:36:37);
- the run at patience 4 ran at reduced priority (nice 5); check-estimator had been launched in the same way, and its
  priority was not observed; the run at patience 12 ran at normal priority (nice 0, observed);
- a test process without TensorFlow, started after check-estimator had begun (at a time not recorded), ran in parallel
  until about the first 5 minutes of the run at patience 4 (it was stopped at 19:32).

## The evaluation command

`pad_arch_eval.py` has two subcommands:

- `check-inputs`: read-only, TensorFlow not imported, nothing written; with `--run-p4` and `--run-p12` it also validates
  the two run folders and applies check (a) to the evaluation plan (below); it loads no fold schema and assembles or
  scores no test frame (it hashes the cache whole, reading its label and subject members, and loads the extracted frames
  file whole; both also hold frames of the held-out subjects).
- `evaluate --run-p4 <folder> --run-p12 <folder> --i-am-authorized`: the evaluation on the NUAA test of the 48 models, in
  one process.

The official commands, from the root of the author's working repository, with no `TF_NUM_INTRAOP_THREADS` or
`TF_NUM_INTEROP_THREADS` in the environment (the lines of the docstring of the command and of the field `operations` of
the configuration):

```bash
.venv/bin/python -I -B scripts/pad_arch_eval.py check-inputs --run-p4 results/archrun/20261120-ARCHRUN-p4-7faa49f --run-p12 results/archrun/20261120-ARCHRUN-p12-ea7fdca
nohup caffeinate -dims .venv/bin/python -I -B scripts/pad_arch_eval.py evaluate --run-p4 results/archrun/20261120-ARCHRUN-p4-7faa49f --run-p12 results/archrun/20261120-ARCHRUN-p12-ea7fdca --i-am-authorized > <a log file outside the repository> 2>&1 &
```

`evaluate` refuses a work tree that is not clean (the two run folders are committed before it, except the checkpoints,
which the repository ignores; their SHA-256 are in the restricted `SHA256SUMS` of each run). `nohup` because a SIGHUP
(closing the terminal) would stop the command as FAIL (SIGTERM does too); the output outside the repository because
`nohup` without a redirection writes `nohup.out` in the repository root and the command would refuse; `caffeinate`
because the cap counts elapsed real time. SIGUSR1 to the pid in `status.json` stops it as INCOMPLETE at the next check.
One heavy process at a time on the machine. The evaluation folder is `results/archeval/<YYYYMMDD>-ARCHEVAL-<commit>`,
with the suffix `-r<k>` for an attempt k > 1.

## The check requested by the coordinator

"Per l'analisi primaria aggiunga un controllo che la interrompa se la soglia effettiva è diversa da 0,5 per una delle
reti o se manca una delle 12 combinazioni di fold e seed in uno dei due stati BN" (for the primary analysis, a check that
stops it if the effective threshold differs from 0.5 for one of the networks or if one of the 12 fold and seed
combinations is missing in one of the two BN states). Where it is placed is a choice of the author
(`primary_check.where`): it is applied in two places, each with its own message and tests.

- **(a) Before the test** (`check_plan`): before any test list is used or any test frame is assembled, and also before
  the dev reproduction, on the evaluation plan built from the validated run folders (one entry per model × BN state ×
  threshold rule, with the threshold that will be applied, from `pad_arch_definitions.condition_threshold`): for
  patience 4 and each BN state, exactly the 12 fold × seed combinations for each network (none missing, duplicated or
  outside the plan), and the threshold of the fixed-threshold condition exactly 0.5 for each network. Message: "primary
  analysis stopped before the test: …".
- **(b) Immediately before the primary analysis** (`check_units`), on the units passed to
  `pad_arch_definitions.primary_analysis` (the same objects): the same two conditions on the effective values. The
  effective threshold of a network in a unit is the field `threshold` that `pad_arch_definitions.prepare_units` passes to
  `subject_counts` to derive the decisions (attack if score > threshold) that enter APCER, BPCER and ACER; its value comes
  from the entry of the evaluation plan, never from a constant read again (`primary_check.effective_threshold`); it must be
  a number, not a boolean, equal to 0.5. Message: "primary analysis stopped: …".
- A failure of (a) stops the command before the dev reproduction and before any test list or frame. A failure of (b)
  in one of the two primary conditions stops it before the primary analysis is computed; in the six secondary
  conditions (b) is applied after the primary analysis is computed, and a failure stops the command and discards it
  (`choices_of_the_author.secondary_incomplete`). In every case: FAIL, no result file, public or restricted.
- After (b) the same units go through `pad_arch_definitions.prepare_units` (its own checks of the frame identifiers) and
  the prepared units are compared with the 12 (NUAA, fold, seed) units: a consistency assertion that cannot fire after
  the completeness check (its mutant is declared redundant), not part of the requested check (`primary_check.where`).
- Also choices of the author (`condition_checks`): the completeness check is applied in (a) and (b) to all 8 conditions,
  the two primary ones first, and the threshold check to the four fixed-threshold conditions; the EER conditions have no
  check of the threshold value, only of completeness; an incomplete secondary condition at (b) stops the command and
  discards a primary analysis already computed (`choices_of_the_author.secondary_incomplete`).

Tests (`livenesslab/scripts/check_pad_arch_eval.py`, synthetic), one per case; all of them run without TensorFlow:

| case | (a) on the plan | (a) in the flow of `evaluate` | (b) on the units | (b) in the flow of `evaluate` |
|---|---|---|---|---|
| complete: passes | `TestCPlanCheck.test_complete_plan_passes` | `TestEFlow.test_pass_outputs` (the same flow: (a) is on its path) | `TestDUnitsCheck.test_complete_units_pass` | `TestEFlow.test_pass_outputs`, `TestEFlow.test_conditions_comparisons_and_sensitivity` (the summary of (b) in `results.json`) |
| effective threshold not 0.5 for LivenessNet only | `TestCPlanCheck.test_threshold_not_0_5_for_livenessnet_only` | `TestFRequestedChecksInTheFlow.test_a_threshold_livenessnet_only` | `TestDUnitsCheck.test_threshold_livenessnet_only` | `TestFRequestedChecksInTheFlow.test_b_threshold_livenessnet_only` |
| effective threshold not 0.5 for ResNet-18 only | `TestCPlanCheck.test_threshold_not_0_5_for_resnet18_only` | `TestFRequestedChecksInTheFlow.test_a_threshold_resnet18_only` | `TestDUnitsCheck.test_threshold_resnet18_only` | `TestFRequestedChecksInTheFlow.test_b_threshold_resnet18_only` |
| a fold × seed combination missing in the ordinary BN state only | `TestCPlanCheck.test_one_network_missing_in_the_ordinary_state_only` (one network; the plan has one entry per network) | `TestFRequestedChecksInTheFlow.test_a_one_network_missing_in_the_ordinary_state_only` | `TestDUnitsCheck.test_missing_unit_ordinary` (the unit, both networks), `TestDUnitsCheck.test_network_missing_in_a_unit` (one network of a unit) | `TestFRequestedChecksInTheFlow.test_b_missing_in_the_ordinary_state_only` |
| a fold × seed combination missing in the re-estimated BN state only | `TestCPlanCheck.test_one_network_missing_in_the_re_estimated_state_only` (one network) | `TestFRequestedChecksInTheFlow.test_a_one_network_missing_in_the_re_estimated_state_only` | `TestDUnitsCheck.test_missing_unit_re_estimated` (the unit, both networks) | `TestFRequestedChecksInTheFlow.test_b_missing_in_the_re_estimated_state_only` (the unit), `TestFRequestedChecksInTheFlow.test_b_network_missing_in_a_unit` (one network of a unit) |
| a duplicated combination | `TestCPlanCheck.test_duplicated_combination` | `TestFRequestedChecksInTheFlow.test_a_duplicated_combination` | `TestDUnitsCheck.test_duplicated_unit` | `TestFRequestedChecksInTheFlow.test_b_duplicated_combination` |

Further tests of the same checks: `TestCPlanCheck.test_combination_outside_the_plan`, `test_network_outside_the_plan`,
`test_threshold_one_entry_and_odd_values`, `test_secondary_conditions`; `TestDUnitsCheck.test_unit_of_another_dataset`,
`test_threshold_read_from_the_unit_used_by_prepare_units`, `test_eer_condition_completeness_only`;
`TestMPointBMore.test_odd_threshold_values`, `test_thirteen_units`, `test_fold_and_seed_swapped`,
`test_network_outside_the_plan`, `test_point_a_fold_and_seed_swapped`, `test_plan_threshold_from_the_record`;
`TestFRequestedChecksInTheFlow.test_secondary_condition_incomplete`;
`TestEFlow.test_primary_analysis_on_the_checked_units` (the primary analysis is computed on the same objects that (b)
checked).

## What `evaluate` does, in this order

One process; every step after the previous one.

0. **Before the evaluation folder is created**: the configuration (a closed set); the pinned files (`PROTOCOL.md` and
   `plan/run_plan.csv` of the pre-registration, each equal to its line of the package `SHA256SUMS`; the configuration of
   the runs, validated by `pad_arch_run.load_config`, and what `pad_arch_run.load_pins` reads with it; the code);
   `run_plan.csv` equal to the file regenerated from the pre-registration and the trial; the cap derived from the plan;
   the thread environment variables equal to the official trial (`pad_arch_run.thread_environment_rule`); the machine
   rule (`pad_arch_run.machine_rule`); the output root; the two run folders (`load_run`, field `run_checks` of the
   configuration: each a PASS of its run with the pinned configuration and code, from a clean work tree, machine and
   thread rule met, official and inside `results/archrun` for an official evaluation, every file equal to its
   `SHA256SUMS`, 24 complete models with their six files); a clean work tree; the attempt rule.
1. `status.json` RUNNING and `run.log`; the inputs (`pad_arch_comparison.load_inputs`: no fold schema); the two dev
   files of each of the 48 models against the manifests (identifiers, subjects, labels, sessions, checkpoint); the
   thresholds (`pad_arch_definitions.condition_threshold`) and the evaluation plan; **check (a)**; the one-class counts
   on the dev, compared cell by cell with `public/one_class.csv` of each run; the earlier "degenerate" count.
2. TensorFlow imported and the threads set to 10 and 10 and read back (`pad_arch_comparison.set_threads`), the thread
   record compared with the official trial and recorded (`pad_arch_run.threads_rule`), the GPU hidden. **The dev
   reproduced for the 48 models in both BN states before any test list or frame**: each checkpoint reloaded with its
   SHA-256 checked right before (`nuaa_bn_recalibration.load_model`) and checked (`pad_arch_run.check_loaded`), the dev
   scored in the ordinary state; an in-memory copy with the BN statistics saved by the run re-installed
   (`nuaa_bn_test.install`; only the moving tensors differ, equal to the saved values), the dev scored in the
   re-estimated state; each state compared with the dev scores saved by the run (`pad_arch_run.evaluator_agreement`: at
   most 1e-5 per frame, no discordant decision at 0.5 nor at the EER threshold). One model-state pair outside: MISMATCH;
   no test list is loaded and no test frame is assembled.
3. Only after the 48 reproductions: a re-verification of the configurations, the code, `PROTOCOL.md`, `run_plan.csv` and
   every file of the two run folders (a FAIL that needs no test comes before the test); the field `test_read_started`
   set to true (written to `status.json` when the command ends, PASS, INCOMPLETE or FAIL; a MISMATCH ends before this
   step, with the field false; the RUNNING `status.json` written at the start says false); then the fold schema; the
   number of test frames per fold equal to the pre-registration (3,242, 3,114, 3,439, 2,819); the test of a fold
   disjoint from its fitting and dev; **the test frames of each fold assembled once** and scored by the 12 models of the
   fold in the two BN states (probabilities and logits only: no test metric is computed while scoring).
4. The same re-verification again and the scan of the `run.log` written so far; the units of the 8 conditions with the
   identifier of every frame; **check (b)**; `pad_arch_definitions.primary_analysis`; the other conditions (with the
   same completeness check); the comparisons between conditions; the subject sensitivity; the scan of the public texts;
   the outputs. **No number of the analysis goes into `run.log`**: after the outputs it says only that they are written;
   "analysis computed" and the `RESULT` line are printed to the standard output only, after the final `status.json`.

The cap is checked before the dev reproduction of every model, before the re-verification that precedes the test lists,
before the assembly of the test of every fold, before the test scoring of every model and before the analysis
(`cap.checks`). The threads are read back after the dev reproduction and after the test scoring.

## Outputs

`results/archeval/<run id>/`: `status.json`, `run.log`;

- `public/`: `results.json`, `report.md`, `conditions.csv`, `units.csv`, `reproduction.csv`, `SHA256SUMS`; no frame or
  subject identifier (counts and SHA-256 of the sorted identifiers), scanned before writing. `report.md` and
  `results.json` lead with what the coordinator asked to see (`report`): the two ACER gaps (ordinary and re-estimated
  BN, patience 4, threshold 0.5) with their subject intervals, their contrast with the subject interval and its reading,
  the conclusion of `PROTOCOL.md` §6 quoted, then the per-seed results apart; then the checks requested by the
  coordinator, APCER and BPCER of the primary analysis, the eight conditions, the comparisons between conditions, the
  subject sensitivity in aggregate form, the one-class counts on the dev, the earlier "degenerate" count apart, the dev
  reproduction, the time and the machine.
- `restricted/`: per model the per-frame test scores with the frame identifiers in both BN states
  (`p4|p12_<network>_fold<k>_seed<s>_test_scores.csv`), `loso_values.csv` (per condition and excluded subject), and
  `SHA256SUMS`.

## Outcomes of `evaluate`

| outcome (exit code) | when | what is left |
|---|---|---|
| PASS (0) | every check passed | all outputs |
| MISMATCH (4) | the dev of at least one model-state pair is not reproduced | `status.json` (the discrepancy per model) and `run.log`; no test list is loaded and no test frame is assembled |
| INCOMPLETE (2) | the cap, or SIGUSR1 | no result; `status.json` records the step, the elapsed time and whether the reading of the test had started |
| FAIL (1) | any other failed check, check (a) or (b) included, or an error | `status.json` with the reason and `run.log`; any result file written is removed, the restricted test scores included (all or nothing); `status.json` says what was written and removed; `run.log` never holds a number of the analysis (`late_failures`) |

No automatic resume (`after_stop`): an evaluation that stops keeps its folder and its outcome; once `test_read_started`
is true any new attempt is a second reading of the NUAA test, and a new attempt is started only after the outcome has
been reported to the coordinator on the C2 card; it repeats every step in a new folder; no model or unit is dropped
after the test is seen. A `status.json` left RUNNING with no live process (a process killed without its clean-up; its
note: "RUNNING without a live process means a failed evaluation") still says `test_read_started` false, whatever the
step reached: it is treated as a reading of the test already started, and any new attempt is reported as a second
reading.

## What the commands read

From the configuration field `test` (the command requires its exact text, `TEST_READ` of the script): "the NUAA test is
read once, by evaluate, after every check that does not need it". For the cache the field says "loaded whole";
`TEST_READ` of `pad_arch_comparison.py` is more precise and is followed here.

- `check-inputs` reads the configurations and the pinned files, what `pad_arch_comparison.load_inputs` reads (the five
  M2 manifests, to index the cache; the cache, hashed whole with its label and subject members read, and the extracted
  frames file, loaded whole, which both also hold frames of the held-out subjects; files of the cells run, of the folds
  run without its public files and of the two accepted runs: their `status.json`, restricted `SHA256SUMS` and one
  training card) and, with `--run-p4` and `--run-p12`, every file of the two run folders; it loads no fold schema and
  assembles no test frame; TensorFlow is not imported and nothing is written.
- `evaluate` reads the same files and the image arrays of the cache, assembles the dev frames of each fold and scores
  them with the 48 checkpoints; it
  loads the fold schema and assembles the test frames only after the 48 dev reproductions and the re-verification; it
  opens no `aggregate.json`, `run_record.json` or `<model>_test_scores.csv` of any earlier run.

## Choices of the author

Not fixed by the pre-registration; each stated as a choice of the author in the configuration (`choices_of_the_author`);
in brackets the field of the configuration that gives the detail, or the key of `choices_of_the_author` where the
detail is there:

1. **Where the requested check is applied and how the effective threshold is read** (`primary_check.where`,
   `primary_check.effective_threshold`): above.
2. **The same checks on the other conditions** (`condition_checks`): completeness on all 8, the threshold on the four
   fixed-threshold ones.
3. **The re-estimated state at test** (`bn_states.re_estimated`): the statistics saved by the run re-installed on a copy
   of the checkpoint, not re-estimated a second time, so that the test is scored in the state whose dev scores, EER
   thresholds and one-class counts the run recorded before the test; alternative not chosen: a second re-estimation
   compared with the saved statistics within the tolerances of PR #28.
4. **The dev reproduced before the test** (`dev_reproduction`): all 48 models in both BN states, with the tolerances of
   the runner (configuration of PR #27), outcome MISMATCH.
5. **The source of the thresholds** (`thresholds`): the dev scores saved by the run, the EER threshold checked against
   their `threshold_dev_eer` column.
6. **A dev without one class** (`thresholds`) stops the evaluation before the test.
7. **Test scoring** (`test_scoring`): the fold schema loaded after the dev reproductions, the test of a fold assembled
   once, scored without labels, the counts and the disjointness checked.
8. **The pairs of conditions compared** (`comparisons.rule`): `PROTOCOL.md` §6 does not say which pairs; the 12 that
   differ in one factor (BN state, threshold, patience), contrast gap(first) − gap(second), the first with the second
   level of the factor; secondary and descriptive.
9. **The conclusions** (`conclusions.rule`): quoted from the pinned `PROTOCOL.md`, with the alternative "lower [higher]"
   resolved by the reading of the primary contrast.
10. **The cap** (`cap.derivation`): 3.5 hours of wall time; the pre-registration fixes no cap for a separate evaluation
    command; the rule of the caps of the plan (maximum estimate × 1.5, rounded up to a multiple of 0.5 hours) applied to
    the evaluation estimate of the plan (2.1284 hours), recomputed by the command from the pinned files.
11. **No automatic resume** (`after_stop`).
12. **Attempts** (`attempts`): the rule of `pad_arch_run.py` with the run id of this command; the official output root
    `results/archeval`; another root only outside the repository with `--unofficial-out-root`, recorded as not official.
13. **Official runs only** (`choices_of_the_author.official_runs`; detail in `attempts`): an official evaluation accepts
    only official PASS runs inside `results/archrun`; an unofficial evaluation (`--unofficial-out-root`, tests of the
    command) does not read an official run nor a folder of the official run root `results/archrun` (identity of the
    files), and does not use the repository as its input root; `check-inputs`, which loads no fold schema and assembles
    no test frame, accepts both. Two limits are declared (below).
14. **Subject sensitivity in public** (`choices_of_the_author.public_subject_sensitivity`): only the aggregate form of
    N1 (PR #10); the value of each exclusion, with its subject, in `restricted/loso_values.csv`.
15. **All or nothing after the test is read** (`choices_of_the_author.fail_after_test_read`; detail in `late_failures`).
16. **An incomplete secondary condition** (`choices_of_the_author.secondary_incomplete`) discards a primary analysis
    already computed.
17. **The re-verification before the test** (`choices_of_the_author.re_verification_before_test`) and again before the
    results.
18. **The earlier "degenerate" count** (`degenerate_apart`): with the rule of the group "ten" of PR #27
    (`nuaa_bn_recalibration.group_of`), on the training cards of the 48 models, per network × patience; applied to
    ResNet-18 as written; the rule was derived from LivenessNet histories.
19. **Names of the outputs** (`outputs`): files and columns, exact fractions written as p/q, the run id
    `YYYYMMDD-ARCHEVAL-<commit>[-r<k>]`, the folder `results/archeval`.

Declared in the pre-registration and not repeated here: the reading of the primary contrast
(`pad_arch_definitions.contrast_reading`, `PROTOCOL.md` §9).

## Limits

- Nothing was run on Linux by the author.
- `evaluate` has not been run; the suite uses synthetic data only (8×8 frames; two run folders produced by
  `pad_arch_run.py` itself with the fake TensorFlow, trainer and scorer of its suite; a real evaluation on two real runs
  of the runner on the CPU with a shortened trainer, with TensorFlow).
- Declared in the configuration (`attempts`), not detected by the command: a copy of an official run
  folder and of the restricted data outside the repository, with `official` edited in the `status.json` of the copy (no
  `SHA256SUMS` covers `status.json`); an input root outside the repository that holds the real inputs (copied, or
  reached through symbolic links), with unofficial runs made on them.
- Declared in the configuration (`late_failures`): after a failure, the line `ERROR: <reason>` written to `run.log` and
  the FAIL `status.json` do not pass through the scan of the publishable outputs (in `status.json` only the personal
  texts are replaced); the error messages of this command and of the functions it calls hold no frame identifier by
  construction (counts, file names, model names and SHA-256 only), with one known exception: the refusal of a scan
  names the subject identifier it found; an unexpected exception carries the text of the exception.
- `test_read_started` reaches `status.json` only when the command ends or fails with its clean-up (the field
  `after_stop` of the configuration, which says that `status.json` records it, is to be read with this limit): a process
  killed without it (SIGKILL, loss of power) after the reading of the test has started leaves `status.json` RUNNING with
  `test_read_started` false. Such a folder is treated as a reading of the test already started (above).
- The cap is derived from the estimate of the plan, which counts a BN re-estimation that this command does not repeat
  and leaves out loading the inputs and the checkpoints and assembling the test (`cap.derivation`).
- The times of check-estimator and of the two runs were measured in the conditions listed above.

## Checks

macOS on the Apple M5 Pro, Python 3.11.16, TensorFlow 2.15.0, NumPy 1.26.4; nothing was run on Linux by the author.

From `livenesslab/` of this folder, with Python and NumPy; TensorFlow 2.15 (Keras 2.15) on the CPU for the tests that
build or score a Keras model:

```bash
python -I -B scripts/check_pad_arch_eval.py
```

Expected (`test_logs/check_pad_arch_eval_from_this_folder.txt`; in the working repository
`test_logs/check_pad_arch_eval.txt`):

```
tests run: 138, failures: 0, errors: 0, skipped: 0
RESULT: PASS
```

**Without TensorFlow** (as on a machine without TensorFlow; here TensorFlow hidden with
`python -I -B tools/run_without_tensorflow.py scripts/check_pad_arch_eval.py`), the 5 tests that build or
score a Keras model are skipped with the declared reason ("TensorFlow is not installed: the tests that build or score a
Keras model are skipped"): `tests run: 138, failures: 0, errors: 0, skipped: 5` and `RESULT: PASS`
(`test_logs/check_pad_arch_eval_without_tensorflow_from_this_folder.txt` and
`test_logs/check_pad_arch_eval_without_tensorflow.txt`, which list the skipped tests by name: the classes `TestZA…` and
`TestZB…`, the scoring of real Keras models and a real evaluation). The only skips the suite admits are the three
declared missing inputs (TensorFlow, the reference architecture file, the delivered configuration or the files it pins).

```bash
python -I -B scripts/check_pad_arch_eval_mutations.py
```

Expected (`test_logs/check_pad_arch_eval_mutations_from_this_folder.txt` and
`test_logs/check_pad_arch_eval_mutations.txt`):

```
mutations killed without TensorFlow: 140/145
mutations killed only with TensorFlow: 5/145
control mutations survived: 8/8 (1 control, 7 declared redundant)
RESULT: PASS
```

The mutants are applied one at a time in a temporary copy; the delivered configuration is not copied there, so in the
copy `test_delivered_configuration` is skipped with the declared reason (6 skipped without TensorFlow, 1 with, in the
first lines of the logs). Without TensorFlow (`--without-tensorflow`, or on a machine without it) the second stage does
not run: the 5 mutants listed as `needs TensorFlow` in the logs are then "not verifiable without TensorFlow", the tool
never counts them as killed and ends with `mutations killed without TensorFlow: 140/145`,
`mutations not verifiable without TensorFlow (never counted as killed): 5` and
`RESULT: INCOMPLETE (some mutations need TensorFlow to be verified)`, exit code 2 (read from the code and from the first
stage of the logs; the mutation check was not run without TensorFlow).

`check-inputs` on the two real run folders (`test_logs/check_inputs_archeval.txt`): it needs the restricted inputs and
the restricted files of the runs and cannot be run from this folder.

The four logs of the working repository were produced with the files of commit `31184c7` of the author's working
repository (where this folder is `delivery/pad-arch-eval/`, the path in the header of one of them), the three from this
folder with the files of this folder; the SHA-256 in the headers of all seven logs are those of the files of this
folder.

The configuration is regenerated from `livenesslab/` with `python -I -B delivery/archeval/make_config.py`, which
rewrites `delivery/archeval/pad_arch_eval.json` from the files of `livenesslab/` and validates it with
`pad_arch_eval.load_config`. Expected: `written 8f00f5f11aa83c874048e4f1846c453b8cc7c64af18a1220372bd0e7bc603bb1`, the
same bytes (it overwrites the file in place: run it on a copy, or check afterwards with `shasum -a 256 -c SHA256SUMS`).
`test_delivered_configuration` loads the delivered configuration with the pinned files of `livenesslab/`.

From this folder:

```bash
shasum -a 256 -c SHA256SUMS
```

Expected: 65 lines `OK`.

The files of `livenesslab/` already published in PR #30 (commit `7c6229a`, open) are byte for byte those files: for
each, its line in this `SHA256SUMS` is the line of `research/pad-arch-run/SHA256SUMS` of that commit. From the root of
the repository, after fetching the head of PR #30 (for example `git fetch <remote> pull/30/head`, which gives commit
`7c6229a`):

```bash
comm -23 <(git show 7c6229a:research/pad-arch-run/SHA256SUMS | grep '  livenesslab/' | sort) <(sort research/pad-arch-eval/SHA256SUMS) | wc -l
```

Expected: `0` (each of the 22 `livenesslab/` lines of PR #30 is in this `SHA256SUMS`).

The 12 files of `livenesslab/` already merged in `research/pad-2026` (at `362c653`) are byte for byte those files. From
the root of the repository, on the branch of this pull request:

```bash
L=research/pad-arch-eval/livenesslab
for f in archcmp/pad_arch_comparison.json sessionablationrun/nuaa_session_ablation_run.json patienceverify/nuaa_session_ablation_run_patience12.json; do cmp research/pad-arch-comparison/livenesslab/delivery/$f $L/delivery/$f; done
for f in bnrecal/nuaa_bn_recalibration.json bntest/nuaa_bn_test.json; do cmp research/nuaa-bn-test/livenesslab/delivery/$f $L/delivery/$f; done
for f in PROTOCOL.md SHA256SUMS plan/run_plan.csv plan/run_order.csv plan/run_caps.csv; do cmp research/pad-arch-comparison/$f $L/delivery/pad-arch-comparison/$f; done
for f in status.json trial.json; do cmp research/pad-arch-comparison/trial/$f $L/results/archcmp/20261119-ARCHTRIAL-b8f7cca/$f; done
```

Expected: no output (12 files compared).

## Contents

66 files: 65 listed in `SHA256SUMS`, plus `SHA256SUMS`. In `livenesslab/`: 39 files (5 new, 12 merged, 22 of PR #30).
Five files match the `.gitignore` of the repository (`*.log`, `results/`): the three `run.log` of `runs/` and the two
files of `livenesslab/results/`; they are committed explicitly (`git ls-files research/pad-arch-eval | wc -l` gives 66).

| path | what it is |
|---|---|
| `README.md` | this file |
| `runs/20261120-ARCHEST-6c4f9e8/` | `status.json`, `run.log`, `public/estimator_check.json`, `public/estimator_check.csv` of check-estimator |
| `runs/20261120-ARCHRUN-p4-7faa49f/`, `runs/20261120-ARCHRUN-p12-ea7fdca/` | `status.json`, `run.log`, `public/` (`models.csv`, `one_class.csv`, `summary.json`, `report.md`, `SHA256SUMS`) of the two runs |
| `test_logs/` | suite (with and without TensorFlow), mutation check and `check-inputs` in the working repository; suite and mutation check from `livenesslab/` of this folder |
| `SHA256SUMS` | SHA-256 of every other file of this folder |

New in this pull request, in `livenesslab/` (5 files):

| path | what it is |
|---|---|
| `scripts/pad_arch_eval.py` | the evaluation command: `check-inputs`, `evaluate` |
| `scripts/check_pad_arch_eval.py` | the synthetic suite (138 tests) |
| `scripts/check_pad_arch_eval_mutations.py` | the mutation check (145 mutants, 1 control, 7 declared redundant) |
| `delivery/archeval/pad_arch_eval.json` | the configuration, a closed set |
| `delivery/archeval/make_config.py` | the generator of the configuration |

Merged in `research/pad-2026` at `362c653` and copied byte for byte, in `livenesslab/` (12 files):

| path | what it is |
|---|---|
| `delivery/archcmp/pad_arch_comparison.json`, `delivery/sessionablationrun/nuaa_session_ablation_run.json`, `delivery/patienceverify/nuaa_session_ablation_run_patience12.json` | configurations pinned by the run configuration, byte for byte those of `research/pad-arch-comparison/livenesslab/delivery/` |
| `delivery/bnrecal/nuaa_bn_recalibration.json`, `delivery/bntest/nuaa_bn_test.json` | the configurations of PRs #27 and #28, byte for byte those of `research/nuaa-bn-test/livenesslab/delivery/` |
| `delivery/pad-arch-comparison/PROTOCOL.md`, `SHA256SUMS`, `plan/run_plan.csv`, `plan/run_order.csv`, `plan/run_caps.csv` | the pre-registration files pinned by the two configurations, byte for byte those of `research/pad-arch-comparison/` |
| `results/archcmp/20261119-ARCHTRIAL-b8f7cca/status.json`, `trial.json` | the record of the official trial at the path pinned by the run configuration, byte for byte `research/pad-arch-comparison/trial/status.json` and `trial.json` |

Published in PR #30 (not merged) and copied byte for byte, in `livenesslab/`: `scripts/pad_arch_run.py`,
`check_pad_arch_run.py`, `check_pad_arch_run_mutations.py`, `pad_arch_comparison.py`, `pad_arch_definitions.py`,
`pad_arch_models.py`, `nuaa_bn_recalibration.py`, `nuaa_bn_test.py`, `nuaa_folds_run.py`,
`nuaa_session_ablation_run.py`, `nuaa_patience_verify.py`, `m2_writers.py`, `path_guard.py`, `package_rules.py`,
`check_pad_arch_comparison.py`, `check_nuaa_folds_run.py`, `check_nuaa_session_ablation_run.py`,
`check_nuaa_bn_recalibration.py`, `check_nuaa_bn_test.py`; `delivery/archrun/pad_arch_run.json`;
`src/livedetection/src/architectures.py`; `tools/run_without_tensorflow.py` (22 files).

## What follows

After the link to this pull request is on the C2 card: `check-inputs`, then `evaluate`, run once, on the same machine
with the same threads (the `status.json` of the evaluation records its start, field `started`, to be compared with the
time of this pull request and of the link); if it does not end PASS, its outcome is reported on C2 before any new
attempt (`after_stop`). Then the report on C2: the two ACER gaps, their contrast with the subject interval and the
per-seed results. The files of the evaluation (`status.json`, `run.log`, `public/`) in a following pull request.
