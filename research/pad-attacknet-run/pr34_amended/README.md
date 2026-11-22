# AttackNetV1, AttackNetV2_1 and AttackNetV2_2 against LivenessNet and ResNet-18 on NUAA: protocol, run matrix, definitions, technical trials (C5, first step)

**Amended copy.** This file is `README.md` of the PR A package (pull request #34, folder
`research/pad-attacknet-comparison/`), which stays unchanged; the only differences from it are this heading and the
first sentence of "What follows". The SHA-256 of the original is in the `SHA256SUMS` of that package.

Asked by the coordinator on card C5 on 21 November 2026 ("Confrontare AttackNet V1, V2.1 e V2.2 con LivenessNet e
ResNet-18 su NUAA"). First step of the card: a pull request towards `research/pad-2026` with the C5
protocol and the run matrix; a short trial for each AttackNet on fitting and dev only (arm B, fold 1, seed 42); the
check that every BN layer is re-estimated, with the device, the time per epoch and the planned hours; configuration
and evaluation code fixed before any new test result is examined. The card also states: "Il test NUAA e i risultati di
C2 sono già noti" (the NUAA test and the C2 results are already known).

| request of the card | where it is in this folder |
|---|---|
| the C5 protocol | `PROTOCOL.md` (11 sections) |
| the run matrix (72 trainings, 36 per patience; 48 reused C2 cells) | `plan/run_matrix.csv`, `plan/reference_cells.csv`, `plan/run_order.csv`; checked by `check-plan` |
| the implementations of the project, from scratch | `livenesslab/scripts/pad_anet_models.py` (registry, `create_model` only), `PROTOCOL.md` §2 |
| a short trial per AttackNet on fitting/dev only (arm B, fold 1, seed 42) | `trial/`: run `20261122-ANETTRIAL-a951ed0`, PASS at attempt 1 (LivenessNet control reproduced, the three AttackNets evaluable); command, procedure and record in `PROTOCOL.md` §7 |
| the check that every BN layer is re-estimated | `pad_anet_comparison.bn_check` (five checks), `PROTOCOL.md` §7; passed on the five BN layers of each AttackNet (`trial/trial.md`) |
| device, time per epoch, planned hours | `trial/trial.md`, `plan/run_plan.md`, `plan/run_caps.csv`: CPU only (GPU hidden), Apple M5 Pro; 26.8, 27.1 and 22.4 seconds per epoch (AttackNetV1, AttackNetV2_1, AttackNetV2_2); training 5.20–6.98 hours of wall time, evaluation 0.59 hours, caps 2.5, 2.5 and 2.0 hours per run at each patience (`PROTOCOL.md` §7, §8) |
| configuration and evaluation code fixed before the new test results | this configuration and the definitions as code now; the evaluation code in a later pull request, published before the test is read (`PROTOCOL.md` §11) |
| primary analysis (patience 4, threshold 0.5, ACER variation per network, three contrasts with LivenessNet, individual 95% intervals) | `livenesslab/scripts/pad_anet_definitions.py`, `primary_analysis`; `PROTOCOL.md` §1, §6 |
| C5 checklist items concerned | "Protocollo e matrice dei 72 addestramenti pubblicati prima dei run completi"; "Tre prove su fitting/dev documentate con verifica BN e stima dei tempi" |

**Status of this delivery.** Protocol, run matrix, reused cells and definitions as code, with their suite and mutation
check; the command of the technical trial and its configuration; the official trial `20261122-ANETTRIAL-a951ed0` (22
November 2026, attempt 1, PASS), with its record in `trial/`, the hours and caps of `plan/` regenerated from it (`plan
--trial-run`) and `PROTOCOL.md` §7 and §8 filled from it. The trial trained the three AttackNets and the LivenessNet
control for 3 epochs on fitting and dev of arm B, fold 1, seed 42 only; no test frame assembled or evaluated, no new
test score or metric read; none of the 72 trainings has been run. Of the C2 results only the public files of the C2
evaluation are read (§ What is read).

## What is read

`TEST_READ` of `livenesslab/scripts/pad_anet_comparison.py` (also the field `test` of the configuration) says exactly
what each command reads. In short: `check-inputs` and `trial` read what the C2 runner reads for its trial (the M2
manifests, the cache hashed whole, the extracted frames file loaded whole, the cells and folds runs, the two accepted
LivenessNet runs and the card `arm_b_fold1_seed42`; the fold schema is never loaded); `trial` assembles only the
fitting and dev frames of arm B, fold 1. Every command reads the public `status.json` and `results.json` of the C2
evaluation `20261121-ARCHEVAL-332fbdf`; `results.json` holds the aggregated C2 test results, already published: it is
parsed and only its list of the 48 checkpoints (run, network, fold, seed, checkpoint SHA-256) is used. No restricted
file of the C2 runs or of the C2 evaluation and no `<model>_test_scores.csv` is opened.

## Verification without images or weights

From `livenesslab/` (the layout of this folder; every pinned configuration is copied under `livenesslab/delivery/`
byte for byte, see Contents):

| command (from `livenesslab/`) | expected last lines | log |
|---|---|---|
| `python -I -B scripts/check_pad_anet_comparison.py` (with TensorFlow 2.15; TMPDIR outside the folder) | `tests run: 104, failures: 0, errors: 0, skipped: 0` and `RESULT: PASS` | `test_logs/check_pad_anet_comparison_from_this_folder.txt` |
| `python -I -B tools/run_without_tensorflow.py scripts/check_pad_anet_comparison.py` (TensorFlow hidden) | `tests run: 104, failures: 0, errors: 0, skipped: 23` and `RESULT: PASS`, the 23 skips with the declared reason "TensorFlow is not installed" | `test_logs/check_pad_anet_comparison_without_tensorflow_from_this_folder.txt` |
| `python -I -B scripts/pad_anet_comparison.py check-plan --plan-dir ../plan --trial-run ../trial` (read only) | `check-plan: 6 files equal byte for byte to the regenerated ones; run matrix 72 rows ({4: 36, 12: 36} per patience, 24 per network, 6 runs of 12), 48 reused cells; trial 20261122-ANETTRIAL-a951ed0` and `RESULT: PASS (check-plan)` | `test_logs/check_plan_from_this_folder.txt` |
| `python -I -B scripts/pad_anet_comparison.py check-machine` (read only, no data, TensorFlow not imported) | on the machine of the trials: `check-machine: machine and versions equal to the C2 trial 20261119-ARCHTRIAL-b8f7cca (11 fields); TensorFlow not imported; nothing written` and `RESULT: PASS (check-machine)`; on another machine the differing fields and `RESULT: FAIL (check-machine)` | `test_logs/check_machine_from_this_folder.txt` |
| `python -I -B scripts/pad_anet_comparison.py plan --trial-run ../trial --out-dir <a new folder outside the repository>`, then `cmp` of the six files with `../plan` | `RESULT: PASS (plan)`; the six files identical | `test_logs/plan_from_this_folder.txt` |
| `python -I -B scripts/check_pad_anet_comparison_mutations.py --list` | `113 mutations (92 offline, 21 with TensorFlow), 1 control mutation and 7 declared redundant, every pattern found exactly once` | `test_logs/check_pad_anet_comparison_mutations_list.txt` |
| `python -I -B scripts/check_pad_anet_comparison_mutations.py --only <numbers>` in batches covering every mutation | `mutations killed: …` per batch; over the thirteen batches 113 of 113 killed and 8 of 8 control and declared redundant mutations survived (`--only c`) | `test_logs/check_pad_anet_comparison_mutations.txt` (every batch, with its command) |

Dependencies: NumPy for every command; TensorFlow 2.15 only for the tests that build or train a model and for `trial`.
In the author's repository the same suite gave the same lines with TensorFlow
(`test_logs/check_pad_anet_comparison.txt`) and with TensorFlow hidden
(`test_logs/check_pad_anet_comparison_without_tensorflow.txt`), and also with the system Python 3.14.7 and NumPy 2.5.5,
without TensorFlow, as `python3 -E -B scripts/check_pad_anet_comparison.py`
(`test_logs/check_pad_anet_comparison_system_python3.txt`); `check-machine` gave the same lines
(`test_logs/check_machine.txt`). The test of the delivered configuration is skipped only if a file it pins is missing,
never if a file differs.

Guard of the suite: it fails if anything changes, while it runs, in the folders that the tested code can write or that
the tests try to make it write (`results/anetcmp`, `results/anetcmp_ck`, `delivery/anetcmp`,
`delivery/x` (a folder the tests try to write), `delivery/pad-attacknet-comparison`, relative to the folder it runs
from); a change elsewhere under `results/`, `export/` or `delivery/` (another work in the same checkout) is reported as
a warning and does not fail it.

`check-inputs` and `trial` need the NUAA frames and the accepted runs: they are not runnable from this folder. `plan`
and `check-plan` with `--trial-run ../trial` read, of `trial/`, only `status.json` and `trial.json`, and accept them
only if `trial.json` has the SHA-256 recorded in `status.json` and was produced by the configuration and code of this
folder.

## Conditions before the launch of the trial

Pre-registered (`PROTOCOL.md` §7); in the author's repository, right before the trial: `python -I -B
scripts/pad_anet_comparison.py check-machine` (read only, no data, TensorFlow not imported, nothing written) must print
`check-machine: machine and versions equal to the C2 trial 20261119-ARCHTRIAL-b8f7cca (11 fields); TensorFlow not
imported; nothing written` and `RESULT: PASS (check-machine)` (on another machine it prints the differing fields and
`RESULT: FAIL (check-machine)`); `check-inputs` PASS; `sw_vers` 27.2.1 and power mode 0; power adapter and lid open;
`setopt nobgnice`; no `TF_NUM_` variable; no other heavy process; `git status --porcelain` empty. Then `python -I -B
scripts/pad_anet_comparison.py trial --attempt 1 --i-am-authorized --checkpoint-root <a folder outside the repository>`
(`PROTOCOL.md` §7). Which of these conditions the published files attest for the official trial, and which rest on the
author's reading with no published record (power mode, power adapter, open lid, `setopt nobgnice`, `check-inputs` PASS
and the `check-machine` right before the launch; `test_logs/check_machine.txt` is a PASS on the same code at 00:56; the
trial repeats the checks of the inputs: `trial/run.log.txt`, "inputs PASS"), is stated at the end of `PROTOCOL.md` §7.

## Contents

- `PROTOCOL.md`; this `README.md`; `SHA256SUMS` (every other file of this folder).
- `plan/`: `run_matrix.csv` (72 rows), `reference_cells.csv` (48 rows), `run_order.csv` (6 runs), `run_plan.csv`,
  `run_plan.md`, `run_caps.csv` (hours and caps from the official trial), generated by `plan --trial-run` from the
  configuration, the public files of the C2 evaluation and `trial/`, byte for byte those of `delivery/anetcmp/plan/` of
  the author's repository.
- `livenesslab/scripts/`: the new modules `pad_anet_models.py`, `pad_anet_definitions.py`, `pad_anet_comparison.py`,
  the suite `check_pad_anet_comparison.py` and the mutation check `check_pad_anet_comparison_mutations.py`; the C2
  modules they import, unchanged (`pad_arch_comparison.py`, `pad_arch_definitions.py`, `pad_arch_models.py`,
  `pad_arch_run.py`, `nuaa_bn_recalibration.py`, `nuaa_bn_test.py`, `nuaa_folds_run.py`, `nuaa_session_ablation_run.py`,
  `nuaa_patience_verify.py`, `m2_writers.py`, `path_guard.py`, `package_rules.py`); the C2 suites that build the
  synthetic world (`check_pad_arch_comparison.py`, `check_nuaa_session_ablation_run.py`, `check_nuaa_folds_run.py`).
- `livenesslab/src/livedetection/src/architectures.py`: the reference code (SHA-256 `a83c02b0…`).
- `livenesslab/tools/run_without_tensorflow.py`: the tool of the C2 package, unchanged.
- `livenesslab/delivery/anetcmp/`: `pad_anet_comparison.json` (the configuration) and `make_config.py` (regenerates it;
  the suite checks that the two agree).
- `livenesslab/delivery/`: the files the configuration pins, copied byte for byte from published packages: the C2
  pre-registration `archcmp/pad_arch_comparison.json` and the two accepted configurations it pins
  (`sessionablationrun/`, `patienceverify/`); the configuration of PR #27 (`bnrecal/`); the record of the C2 trial
  (`pad-arch-comparison/trial/status.json`, `trial.json`, PR #29); the public `status.json` and `public/results.json`
  of the C2 evaluation (`pad-arch-eval-results/runs/20261121-ARCHEVAL-332fbdf/`, PR #32).
- `test_logs/`: the logs of the table above, among them `check_machine.txt` and `check_machine_from_this_folder.txt`.
- `trial/`: `status.json`, `trial.json`, `trial.md` and `run.log.txt` (`run.log` renamed) of the official trial
  `20261122-ANETTRIAL-a951ed0`, unmodified; `conditions.txt`, written by the author around the trial (uptime, CPU usage,
  command, number of `TF_NUM_` variables set, commit and number of lines of `git status --short`, exit code), not by
  the runner, copied unchanged under a heading line added for this folder; it does not hold the power mode
  (`PROTOCOL.md` §7). The three trial checkpoints are not published (their SHA-256 are in `trial/trial.json`).

## Choices of the author

The closed list of `PROTOCOL.md` §9, also in the field `choices_of_the_author` of the configuration.

## What follows

The official trial is done (PASS, `PROTOCOL.md` §7); its outcome was reported on the C5 card after the pull request of
this package was opened, a deviation from the pre-registered order recorded in the amended `PROTOCOL.md` §7. Then, as
`PROTOCOL.md` §11: a pull request with the command, configuration and suite of the six runs of the 72 trainings,
published and linked on the C5 card before the runs start, unless the coordinator states otherwise on the card; the
runs; the evaluation code and configuration published, with the link on the card, before the NUAA test is read.
