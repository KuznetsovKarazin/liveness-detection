# LivenessNet and ResNet-18 on NUAA: the files of the evaluation on the test (20261121-ARCHEVAL-332fbdf, PASS)

Asked by the coordinator on the C2 card on 19 November 2026, in the evening, after merging PR #29 (merge `362c653` into
`research/pad-2026`): "Pubblichi il codice di valutazione prima di leggere il test" (the evaluation code is to be
published before the test is read), with a check that stops the primary analysis if the effective threshold differs
from 0.5 for one of the networks or if one of the 12 fold and seed combinations is missing in one of the two BN states;
"A fine run riporti in C2 i due divari di ACER, il loro contrasto con l’intervallo per soggetto e i risultati per seed"
(at the end, the two ACER gaps, their contrast with the subject interval and the per-seed results are to be reported on
C2). The evaluation code was published in PR #31 (commit `060edbf`), whose text announced "The files of the evaluation
(`status.json`, `run.log`, `public/`) in a following pull request": this folder.

| request | where it is in this folder |
|---|---|
| the two ACER gaps (ordinary and re-estimated BN) with their subject intervals | [Primary analysis](#primary-analysis); `public/report.md`, "Primary analysis"; `public/results.json`: `primary_analysis.gap_ordinary`, `primary_analysis.gap_re_estimated` |
| their contrast with the subject interval | [Primary analysis](#primary-analysis); `public/results.json`: `primary_analysis.contrast` |
| the per-seed results | [Per seed](#per-seed); `public/report.md`, "Per seed"; `public/results.json`: `primary_analysis.per_seed` |
| the check that stops the primary analysis | [The check requested by the coordinator](#the-check-requested-by-the-coordinator); `public/results.json`: `checks_before_primary_analysis` |

**Status of this delivery.** This folder holds only the records of one evaluation, copied byte for byte from the
author's working repository: `runs/20261121-ARCHEVAL-332fbdf/` with `status.json`, `run.log` and `public/`
(`report.md`, `results.json`, `conditions.csv`, `units.csv`, `reproduction.csv`, `SHA256SUMS`). Nothing of
`restricted/` (the per-frame test scores with the frame identifiers and `loso_values.csv` with the excluded subject);
`status.json` records the SHA-256 of its `SHA256SUMS` (`restricted_sha256sums_sha256`). No code is in this folder: the
code that produced these files is that of PR #31 ([Verifying the files against PR #31](#verifying-the-files-against-pr-31)).
This folder is not self-sufficient by design: its records are verified against the code, the configuration and the run
folders of PR #31 (open, not merged), through `git show 060edbf:` after the fetch given in
[Verifying the files against PR #31](#verifying-the-files-against-pr-31).
No file of the evaluation was edited after the command ended (`status.json` records the SHA-256 of `run.log` and of
`public/SHA256SUMS`; `status.json` itself is covered only by the `SHA256SUMS` of this folder). Every number below is copied from `public/report.md`,
with its rounding, or read from the file and field named.

## The evaluation

| | value | file and field |
|---|---|---|
| ID, status | `20261121-ARCHEVAL-332fbdf`, **PASS** | `status.json`: `run_id`, `status`, `passed` |
| attempt | 1, no earlier attempt | `status.json`: `attempt` (`attempt` 1, `previous` empty) |
| official, reading of the test started | true, true | `status.json`: `official`, `test_read_started` |
| started, finished (local time, +01:00) | 2026-11-21 09:11:55, 09:42:42 | `status.json`: `started`, `finished` |
| wall time | 1,847.6 s, cap 3.5 hours of wall time (plan estimate 2.13 hours) | `status.json`: `wall_seconds`, `cap_hours`, `cap_unit`; `public/results.json`: `cap_derivation.evaluation_hours_estimate` |
| process CPU time (all threads) | 15,479.8 s | `status.json`: `process_cpu_seconds` |
| models | 48, 48 evaluated | `public/report.md`, line 3 |
| dev reproduced before the test | 96 of 96 model-state pairs within 1e-05, no discordant decision | `public/report.md`; `status.json`: `dev_reproduction`; `public/reproduction.csv` |
| test frames per fold | 3242, 3114, 3439, 2819 | `public/results.json`: `test_frames_per_fold` |
| threads | 10 intra-op and 10 inter-op, set by the command and read back; `TF_NUM_` variables absent; equal to the official trial | `status.json`: `threads`, `threads_rule` |
| machine | equal to the official trial `20261119-ARCHTRIAL-b8f7cca` | `status.json`: `machine_rule` (`equal` true) |
| commit of the author's working repository, work tree clean | `332fbdf`, `dirty` false | `status.json`: `commit`, `dirty` |

The commit is a commit of the author's working repository, which also holds unpublished work in progress; the link
between this record and the published code is the SHA-256 of the configuration (`config_sha256`) and of the code
(`code_sha256`), checked below against the files of PR #31. `command_line` is built by the command from the parsed
arguments (`attempts` of the configuration): it writes `--config`, `--out-root` and `--attempt` also at their defaults,
omits the options not given and starts with `python`; the launch (`nohup caffeinate -dims .venv/bin/python …`) is
described in `operations` of the configuration.

## Order of the times

- PR #31 opened on 2026-11-21 at 08:50:46 CET (07:50:46 UTC, `createdAt` of the pull request), head `060edbf`.
- Comment with the link to PR #31 on the C2 card: 2026-11-21 at 09:10:29 CET.
- `check-inputs` (read-only, no test frame) ended PASS at 09:11:47 CET (the author's record; its output is not a file of
  this folder and is not verifiable from outside).
- `evaluate`: `started` 2026-11-21T09:11:55+0100 (`status.json`; the same time opens `run.log`), `finished`
  2026-11-21T09:42:42+0100.

## Primary analysis

Requested by the coordinator: patience 4, threshold 0.5. ACER, APCER and BPCER are proportions between 0 and 1 (not
percentages). Gap = ACER(ResNet-18) - ACER(LivenessNet), mean over the 12 fold-seed units; contrast = gap(re-estimated
BN) - gap(ordinary BN). Subject interval: percentiles 2.5-97.5 of 1,000 resamples (seed 0) of the 16 test subjects
pooled over the 4 folds, with the valid resamples in parentheses; the resampling indices are those of the recipe of
the project (`nuaa_session_ablation_run.boot_picks` and `BOOT_RECIPE`: `numpy.random.default_rng(0)` restarted at
every call, subjects in ascending order, `integers(0, n, n)` per resample), the same for both networks and both BN
states. From `public/report.md`:

| | mean | subject interval (valid resamples) | units: ResNet-18 lower / LivenessNet lower / ties |
|---|---|---|---|
| ACER gap, ordinary BN | -0.1606 | [-0.2079, -0.0741] (926) | 11 / 1 / 0 |
| ACER gap, re-estimated BN | -0.0607 | [-0.1253, 0.0476] (926) | 9 / 3 / 0 |
| contrast | 0.0998 | [0.0256, 0.1632] (926) | units whose gap changes sign: 2 of 12 |

Reading of the contrast (a choice of the author, `pad_arch_definitions.contrast_reading`, `PROTOCOL.md` §6): **ACER
difference (ResNet-18 - LivenessNet) higher with re-estimated BN** (interval entirely above zero). "Higher" is
algebraic: the mean gap is negative in both states and less negative with re-estimated BN (-0.0607 against -0.1606).

Conclusion, written before the test and quoted from `PROTOCOL.md` §6 with the alternative resolved (`public/report.md`;
`public/results.json`: `primary_analysis.conclusion`): "on these checkpoints and test frames the signed ACER difference
(ResNet-18 − LivenessNet) is algebraically higher with this re-estimation; it does not identify a cause and says nothing
on which network has the lower ACER in each BN state (that is read from the two gaps, reported with their intervals)".

The two gaps read with the sign rule of `PROTOCOL.md` §6 (`public/results.json`: `primary_analysis.gap_ordinary.sign`,
`primary_analysis.gap_re_estimated.sign`): ordinary BN "ResNet-18 lower" (interval entirely below zero); re-estimated BN
"unresolved" (interval contains zero). By the same definition there is no change of sign between the two conditions
(a change of sign needs two resolved conditions of opposite sign; `public/report.md`, "Between conditions", first row:
sign change "no").

APCER and BPCER of the primary analysis (`public/report.md`): APCER gap 0.1801 [0.0871, 0.3372] with ordinary BN, 0.1556
[0.0581, 0.3211] with re-estimated BN, contrast -0.0245 [-0.0912, 0.0415]; BPCER gap -0.5013 [-0.6144, -0.3330] with
ordinary BN, -0.2771 [-0.4333, -0.0856] with re-estimated BN, contrast 0.2242 [0.1202, 0.3088]. Valid resamples: 950
for every APCER interval, 940 for every BPCER interval (`public/results.json`: `primary_analysis.<gap_ordinary,
gap_re_estimated, contrast>.interval.<apcer, bpcer>.valid_resamples`; `public/report.md` does not write them).

### Per seed

Reported apart; not part of the reading. Each seed has 4 units (one per fold), and the bootstrap of each seed uses the
same 16 test subjects (926 valid resamples for every ACER interval below; `public/results.json`:
`primary_analysis.per_seed`). From `public/report.md`:

| seed | ACER gap, ordinary [interval] | ACER gap, re-estimated [interval] | contrast [interval] | contrast reading |
|---|---|---|---|---|
| 42 | -0.1778 [-0.2821, -0.0227] | -0.0762 [-0.1557, 0.0479] | 0.1016 [-0.0226, 0.2263] | unresolved |
| 43 | -0.0970 [-0.1961, 0.0615] | -0.0905 [-0.1850, 0.0453] | 0.0065 [-0.0819, 0.0794] | unresolved |
| 44 | -0.2069 [-0.3472, -0.0259] | -0.0156 [-0.0796, 0.0896] | 0.1913 [-0.0108, 0.3771] | unresolved |

## The check requested by the coordinator

Applied in two places (a choice of the author, `primary_check.where` of the configuration) and passed in both before
the primary analysis (`public/report.md`, line 29; `public/results.json`:
`checks_before_primary_analysis`; `run.log`, line 3): (a) on the evaluation plan, before any test list or frame was
used: the 12 fold-seed combinations for each network in both BN states and the threshold to be applied 0.5 for both
networks (192 plan entries); (b) on the units passed to `pad_arch_definitions.primary_analysis`: 12 units in each of the
two primary conditions, effective threshold 0.5 for LivenessNet and for ResNet-18. The same completeness check passed on
the other 6 conditions (a choice of the author, `condition_checks`).

## What cannot be concluded

- Which network has the lower ACER with re-estimated BN: the interval of that gap contains zero (sign "unresolved").
- Anything per seed: the per-seed values do not enter the reading (`PROTOCOL.md` §6), and the contrast interval of each
  of the three seeds contains zero ("unresolved").
- A cause: the conclusion of `PROTOCOL.md` §6 says that the result "does not identify a cause"; it is quoted, not
  extended.
- An independent validation: on NUAA the comparison is not an independent validation (`PROTOCOL.md` §5), and the
  LivenessNet side of the primary analysis repeats cells already read on this test (below).
- Anything beyond these networks, recipe and data (`PROTOCOL.md` §10); the six secondary conditions and the comparisons
  between conditions in `public/report.md` are descriptive (`PROTOCOL.md` §6), eight conditions without correction for
  multiplicity (`PROTOCOL.md` §10).

## Limits and choices of the author

- **Valid resamples: 926 of 1,000** for every ACER interval (950 for APCER, 940 for BPCER; `public/results.json`,
  `valid_resamples`). As fixed in `PROTOCOL.md` §6, a resample is dropped if a unit's metric is undefined: APCER is
  undefined in a unit when the resampled subjects of its fold have no attack frame, BPCER when they have no bona fide
  frame, and ACER when either is undefined (`pad_arch_definitions.rates`, `subject_bootstrap`; validity is counted
  metric by metric). For ACER this happens in 74 of the 1,000 resamples: those in which, in some fold, no subject is
  drawn or the subjects drawn have no frame of one of the two classes (read from the code; the per-subject counts are
  not public). It depends only on the
  subjects drawn, not on the network, the BN state or the seed: the per-seed intervals also have 926 valid resamples.
- **Nothing excluded after the test was seen**: every model enters, one-class models included (`public/report.md`, line
  3; `after_stop` of the configuration). On the dev at threshold 0.5, 2 of the 12 LivenessNet models at patience 4 with
  ordinary BN decide every frame attack (`public/report.md`, "One-class runs"; the models of fold 1 seed 42 and fold 4
  seed 44, column `ordinary_one_class_05` of `research/pad-arch-eval/runs/20261120-ARCHRUN-p4-7faa49f/public/models.csv`
  in PR #31); no other network, patience and BN state
  has one. On the test at threshold 0.5 with ordinary BN the same two models decide every frame attack (APCER 0, BPCER
  1, ACER 1/2 in `public/units.csv`), and they enter the primary analysis like the others.
- **Only the ResNet-18 side of the primary analysis is new** (`PROTOCOL.md` §5): both LivenessNet cells of the primary
  analysis had already been analysed on this test with the accepted checkpoints (PR #21, PR #28); the 12 new LivenessNet
  trainings at patience 4 repeat the accepted ones with the same recipe, data and seeds and have the same history
  (`history_check` in PR #31), which does not prove identical weights (`PROTOCOL.md` §5).
- **One reading of the test by this evaluation**: attempt 1, no earlier attempt; any new attempt would be a second
  reading of the NUAA test (`after_stop`).
- 16 test subjects pooled over the folds; 3 seeds, paired only nominally (the same seed does not give the two networks
  the same initial weights) (`PROTOCOL.md` §10).
- Choices of the author that enter these numbers (`choices_of_the_author` of the configuration, `PROTOCOL.md` §6 and
  §9): the subject intervals, APCER and BPCER and the per-seed values in the primary analysis; the reading of the
  contrast; the re-estimated state at test with the BN statistics saved by the run re-installed, not re-estimated a
  second time (`bn_states.re_estimated`).
- The means and the unit counts can be recomputed from `public/units.csv` (exact fractions per unit, written reduced,
  for example `0/1`: the denominators are not the frame counts; the test frames per fold are in `n_frames` and in
  `test_frames_per_fold` of `public/results.json`); the subject intervals need the per-subject counts, which are not
  public.
- No restricted file is in this folder.

## Verifying the files against PR #31

From the root of the repository, on the branch of this pull request, after fetching the head of PR #31 (`upstream`:
the remote of `KuznetsovKarazin/liveness-detection`; the fetch gives commit
`060edbf52f0edf2d0bb15cf171aa33710ab0b96b`). The paths in `code_sha256` are relative to `livenesslab/` of
`research/pad-arch-eval/`.

```bash
git fetch upstream pull/31/head
S=research/pad-arch-eval-results/runs/20261121-ARCHEVAL-332fbdf/status.json
L=research/pad-arch-eval/livenesslab
python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1])); c=dict(d["code_sha256"]); c["delivery/archeval/pad_arch_eval.json"]=d["config_sha256"]; [print(v+"  "+k) for k,v in sorted(c.items())]' $S |
while read -r h f; do [ "$(git show 060edbf:$L/$f | shasum -a 256 | cut -c1-64)" = "$h" ] && echo "OK $f" || echo "DIFFERENT $f"; done
```

Expected: 14 lines `OK` (the 13 code files and the configuration), none `DIFFERENT`.

The other inputs recorded in `inputs_sha256` (the pre-registration files, the official trial, the configurations of the
runs and of PRs #27 and #28), at the paths where PR #31 holds them:

```bash
python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1]))["inputs_sha256"]; p={"run_configuration": "delivery/archrun/pad_arch_run.json", "preregistration": "delivery/archcmp/pad_arch_comparison.json", "package_sums": "delivery/pad-arch-comparison/SHA256SUMS", "protocol": "delivery/pad-arch-comparison/PROTOCOL.md", "run_plan": "delivery/pad-arch-comparison/plan/run_plan.csv", "run_order": "delivery/pad-arch-comparison/plan/run_order.csv", "run_caps": "delivery/pad-arch-comparison/plan/run_caps.csv", "trial_status": "results/archcmp/20261119-ARCHTRIAL-b8f7cca/status.json", "trial_json": "results/archcmp/20261119-ARCHTRIAL-b8f7cca/trial.json", "bn_recalibration_config": "delivery/bnrecal/nuaa_bn_recalibration.json", "bn_test_config": "delivery/bntest/nuaa_bn_test.json"}; [print(d[k]+"  "+v) for k,v in sorted(p.items())]' $S |
while read -r h f; do [ "$(git show 060edbf:$L/$f | shasum -a 256 | cut -c1-64)" = "$h" ] && echo "OK $f" || echo "DIFFERENT $f"; done
```

Expected: 11 lines `OK` (the twelfth key of `inputs_sha256`, `configuration`, equals `config_sha256` and is checked by
the first block).

The two run folders read by the evaluation are those published in `runs/` of PR #31 (`runs` of `status.json`: the
SHA-256 of their `status.json` and public `SHA256SUMS`):

```bash
python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1])); [print(r["status_sha256"]+"  "+r["run_id"]+"/status.json\n"+r["public_sha256sums_sha256"]+"  "+r["run_id"]+"/public/SHA256SUMS") for r in d["runs"].values()]' $S |
while read -r h f; do [ "$(git show 060edbf:research/pad-arch-eval/runs/$f | shasum -a 256 | cut -c1-64)" = "$h" ] && echo "OK $f" || echo "DIFFERENT $f"; done
```

Expected: 4 lines `OK`.

## Integrity of the files

From this folder:

```bash
shasum -a 256 -c SHA256SUMS
```

Expected: 9 lines `OK`. From `runs/20261121-ARCHEVAL-332fbdf/public/`, `shasum -a 256 -c SHA256SUMS` gives 5 lines `OK`
(the file written by the command). From `runs/20261121-ARCHEVAL-332fbdf/`, the SHA-256 of `run.log` and of
`public/SHA256SUMS` are those recorded in `status.json` (`run_log_sha256`, `public_sha256sums_sha256`):

```bash
shasum -a 256 run.log public/SHA256SUMS
python3 -I -c 'import json; d=json.load(open("status.json")); print(d["run_log_sha256"]); print(d["public_sha256sums_sha256"])'
```

## Contents

10 files: 9 listed in `SHA256SUMS`, plus `SHA256SUMS`. One file matches the `.gitignore` of the repository (`*.log`):
`runs/20261121-ARCHEVAL-332fbdf/run.log`; it is committed explicitly (`git ls-files research/pad-arch-eval-results | wc
-l` gives 10).

| path | what it is |
|---|---|
| `README.md` | this file |
| `runs/20261121-ARCHEVAL-332fbdf/status.json` | the final status of the evaluation |
| `runs/20261121-ARCHEVAL-332fbdf/run.log` | its log (no number of the analysis) |
| `runs/20261121-ARCHEVAL-332fbdf/public/report.md` | the report: primary analysis first, then the per-seed results, the check requested by the coordinator and the secondary analyses |
| `runs/20261121-ARCHEVAL-332fbdf/public/results.json` | every result of the report, with full precision |
| `runs/20261121-ARCHEVAL-332fbdf/public/conditions.csv` | per condition and scope (all seeds, each seed): mean gaps, intervals, unit counts, sign |
| `runs/20261121-ARCHEVAL-332fbdf/public/units.csv` | per condition and unit: APCER, BPCER and ACER of each network as exact fractions, the gaps, the count of test frames and the SHA-256 of their sorted identifiers |
| `runs/20261121-ARCHEVAL-332fbdf/public/reproduction.csv` | the dev reproduction per model and BN state |
| `runs/20261121-ARCHEVAL-332fbdf/public/SHA256SUMS` | SHA-256 of the five public files above |
| `SHA256SUMS` | SHA-256 of every other file of this folder |

## What follows

The report on the C2 card, with the link to this pull request: the two ACER gaps, their contrast with the subject
interval and the per-seed results. No new reading of the NUAA test and no training are in progress.

## Declarations

The NUAA test was read once by this evaluation (attempt 1); no training; no external dataset used; no restricted file
in this folder; no file of the evaluation edited after the command ended.
