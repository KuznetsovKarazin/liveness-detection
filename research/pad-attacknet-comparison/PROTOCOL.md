# AttackNetV1, AttackNetV2_1 and AttackNetV2_2 against LivenessNet, with ResNet-18 as external reference, under BatchNormalization re-estimation on NUAA (protocol)

**Status: pre-registration of the C5 comparison, first step of the C5 card (21 November 2026).** Written before any of
the 72 new trainings and before any new test score. The NUAA test and the C2 results are already known (§5): this is not
an independent validation, and the design is a post hoc extension of C2 to three more networks, written after the C2
contrast was seen. The parts that depend on the official technical trial (the record in §7, hours and caps in §8) were
filled from its record (`trial/`) after the trial of 22 November 2026; nothing in them is estimated by hand. "Wall time"
is elapsed real time (the "elapsed time" of C2).

## 1 Question

From the C5 card: "Quanto cambia il confronto fra le nostre quattro reti PAD quando si ristimano le statistiche di
BatchNorm (BN)?" (how much does the comparison between our four PAD networks change when the BatchNorm statistics are
re-estimated?). The four networks are LivenessNet and the three AttackNets; ResNet-18 is the external reference.

Primary analysis, fixed by the coordinator on the C5 card: with patience 4 and threshold 0.5, for each network the ACER
variation (re-estimated BN minus ordinary BN, in percentage points); for each AttackNet, this variation compared with
that of LivenessNet, on the same 12 fold-seed pairs and the same subjects; the three contrasts reported with paired
subject bootstrap intervals, stating that the 95% intervals are individual. Patience 12, the EER threshold and the
rankings are descriptive analyses.

## 2 What stays fixed

The three AttackNets are trained from scratch (random initial weights, no pre-trained weights) with the project
implementations, `src/livedetection/src/architectures.py` (SHA-256 `a83c02b0…`, the file pinned by C2), with the same
subject splits, frames, preprocessing, recipe and settings as C2.

| | AttackNetV1 | AttackNetV2_1 | AttackNetV2_2 |
|---|---|---|---|
| blocks | two phases: three 3×3 convolutions with LeakyReLU 0.2; the raw output of the first convolution concatenated with the LeakyReLU output of the third; 2×2 max-pool | as V1 | as V1, with a sum instead of the concatenation |
| BN (5 layers, Keras defaults: momentum 0.99, epsilon 1e-3) | `bn1_1` (16 channels, 256×256), `bn1_final` (32, 256×256), `bn2_1` (32, 128×128), `bn2_final` (64, 128×128), `bn_dense` (128) | as V1 | 16, 16, 32, 32, 128 channels, same sides |
| head | `Dense(128, tanh)` `dense1` → BN → dropout 0.5 → `Dense(2, softmax)` `output` | `Dense(128)` `dense1_linear` + `Activation("tanh")` → BN → dropout 0.5 → `output` | as V1 |
| dropout of the convolutional phases | 0.25 | 0.5 | 0.25 |
| trainable parameters / moving statistics | 33,588,194 / 544 | 33,588,194 / 544 | 16,806,274 / 448 |

The counts come from the registry (`scripts/pad_anet_models.py`), which derives them from a layer-by-layer
specification written from the three `build_model` functions, and were checked against the built Keras graphs at
256×256×3 and 8×8×3 (topology, attributes and counts identical; suite `check_pad_anet_comparison.py`, `TestBModels`).

**Constructors.** Three facts, reported together because the choice moves the numbers:

- The training code of the reference repository builds the models with `create_model(model_name, input_shape,
  dropout_rate=model_config.dropout_rate, l2_reg=config.l2_regularization)` (`scripts/train_all_models.py`, lines 85
  and 348): dropout, learning rate and batch from `config/model_configs.py` (lines 42–82: AttackNetV1 dropout 0.5,
  AttackNetV2_1 0.4, AttackNetV2_2 0.2; learning rate 3e-7, 4e-7, 4e-8; batch 8), while `config` is a `TrainingConfig`
  built without `l2_regularization`, so `l2_reg` is its default 1e-2 (`src/training_utils.py`, line 77); the value 1e-5
  of `model_configs.py` is not passed.
- C2 did not use those values for LivenessNet (`LivenessNetConfig`: dropout 0.2, learning rate 1e-7, batch 8;
  LivenessNet has no regularizer in any case): it built LivenessNet with `create_model("LivenessNet", input_shape=…)`,
  that is dropout 0.5 and `l2_reg=0.0`, with learning rate 1e-3 and batch 16. In C2 the model was built with
  `create_model` and its defaults, and trained with the recipe of the accepted runs (`pad_arch_comparison.json`,
  `architectures.LivenessNet.builder`).
- The same rule applied to the AttackNets gives dropout 0.5 and no L2: with `l2_reg=0.0` AttackNetV1 and AttackNetV2_1
  have no regularizer, and AttackNetV2_2 has none in any case. `BaseArchitecture` has `l2_reg=0.01` as default and
  `get_all_architectures` inherits it: the models are built only through `create_model`, and the registry checks on the
  built model that no layer has a regularizer and that the model has no additional loss.

The learning rate 1e-3 and batch 16 are those of the C2 recipe required by the card ("Stessi … ricetta e impostazioni di
C2"); only the dropout and `l2_reg` of the constructor are a choice of the author.

With these constructors AttackNetV1 and AttackNetV2_1 have the same parameters and BN layers and differ in three
points: the dropout of the convolutional phases (0.25 against 0.5); the head (`Dense(128, tanh)` against
`Dense(128)` plus `Activation("tanh")`, one more layer with the same function); the name of the model.

**Recipe** (that of C2, `delivery/archcmp/pad_arch_comparison.json`, `training.recipe`): CPU only, GPU hidden and the
remapping optimizer off; 10 intra-op and 10 inter-op TensorFlow threads, set and read back by the runner; legacy Adam
1e-3, clipnorm 1.0; batch 16; at most 12 epochs; early stopping on the dev `val_loss` with the best weights restored;
ReduceLROnPlateau (factor 0.5, patience 2, minimum 1e-6); balanced class weights; augmentation on the fitting frames
only; input frames / 255; NUAA arm B, folds 1–4, seeds 42, 43, 44; the same machine.

**Asymmetries**: the recipe was chosen on the LivenessNet curves and is not adapted to the AttackNets; their dense layer
has 33.6 million parameters on 262,144 inputs (AttackNetV1 and AttackNetV2_1).

**Known defects, with their source**: on the Metal GPU the remapping optimizer breaks with the AttackNets
(`scripts/train_cnn.py`, line 287, keeps `remapping: False` for every network, the comment naming the AttackNets); until
the C2 technical fix of 28 October 2026 (`research/c2-livenessnet-device/`), `train_cnn.py` did not refuse a NaN or
infinite graph/eager difference for the AttackNets (a defect of the check, shown with injected values); on Metal the
ReLU after the dense layer is skipped, which concerns LivenessNet. Here CPU only.

## 3 Factors

Identical to C2 (`delivery/archcmp/pad_arch_comparison.json`, `plan`; C2 PROTOCOL §3): patience 4 and 12; ordinary BN
and BN re-estimated on the fitting set only with the estimator of PR #27, extended to networks with branches by
`pad_arch_run.reestimate` (the order checked on the graph); threshold 0.5 and the EER threshold estimated on the dev
of the fold in the same BN state. Attack if score > threshold. 8 conditions per checkpoint. As in C2, the estimator
first reproduces the statistics of run `20261114-BNRECAL-a10a082` within the tolerance of PRs #27 and #28 before the
runs, otherwise the work stops.

## 4 Planned runs

72 new trainings (3 AttackNets × 4 folds × 3 seeds × 2 patiences; 36 per patience, 24 per network, 12 per network and
patience) in six runs of 12, one process each: `anet_v1_p4`, `anet_v2_1_p4`, `anet_v2_2_p4`, then the same at
patience 12; in each run folds 1–4 and seeds 42, 43, 44 (`plan/run_matrix.csv`, `plan/run_order.csv`). Per unit
1,400 fitting and 400 dev frames; test frames per fold 3,242, 3,114, 3,439 and 2,819, from the accepted configuration
(as in C2), not read.

48 reused cells: the C2 checkpoints of LivenessNet and ResNet-18 (runs `20261120-ARCHRUN-p4-7faa49f` and
`20261120-ARCHRUN-p12-ea7fdca`), not trained and not re-evaluated: their per-frame test scores already computed by
the C2 evaluation `20261121-ARCHEVAL-332fbdf` are read, each file checked against the restricted `SHA256SUMS` whose
SHA-256 is in the public `status.json` of that evaluation (`plan/reference_cells.csv`: network, fold, seed, patience,
run, checkpoint SHA-256, name of the restricted score file and the anchor). The 48 checkpoint SHA-256 are public only in
the `results.json` of the C2 evaluation (`models[].weights_sha256`), not in the public files of the two C2 runs; the
link with the checkpoints of the runs is checked in the evaluation, from the restricted files. Their dev EER thresholds
and dev one-class results are read from the public files of the C2 runs and evaluation (`results.json`,
`models[].dev_eer_threshold`; `public/models.csv` of the runs, `ordinary_one_class_05` and
`re_estimated_one_class_05`, and their counts in `public/one_class.csv`), not recomputed.

Six runs instead of two (a choice of the author): the trainer clears the session and resets the seeds before every
model; in C2 the LivenessNet histories were identical to the accepted ones in 24 of 24 with ResNet-18 trained in
between; for the AttackNets there is no reference history to preserve; a failure costs 12 trainings and not 36.

## 5 Already measured and not

- LivenessNet and ResNet-18 on NUAA, 8 conditions: C2 evaluation `20261121-ARCHEVAL-332fbdf` (PR #32), read once. At
  patience 4 and threshold 0.5 the C2 contrast (ACER gap ResNet-18 − LivenessNet, re-estimated minus ordinary BN) is
  +9.98 percentage points (`public/results.json`, `primary_analysis.contrast.mean.acer` = 0.0998). All 48 reused cells
  were read on this test by the C2 evaluation (PR #32). Before C2, the accepted LivenessNet checkpoints (same recipe,
  data and seeds) had been analysed on this test in some cells: ordinary BN at both thresholds (PR #21, #23) and
  patience 4, re-estimated BN, threshold 0.5 (PR #28). In each contrast only the AttackNet side is new.
- AttackNets on NUAA frames, other checkpoints and another recipe, not used here: in C1 (`research/c1-livenesslab/`,
  `PROVENANCE.md`; the NUAA synthesis, `SYNTHESIS.md`, «Models and training») AttackNetV1, AttackNetV2_1 and
  AttackNetV2_2 checkpoints trained on 12 September 2026 on `cnn_train` (early stopping on `cnn_es`, patience 4,
  training device not recorded) were scored on the 300 NUAA images of C1, also at the dev EER threshold (M1) and, for
  AttackNetV2_1, with the extractor of the training cache (PR #14); the C1 checkpoints trained on CASIA-FASD,
  CelebA-Spoof and SynthASpoof, AttackNets among them, were applied to the same 300 images (`SYNTHESIS.md`, §3.3); in
  N2 (`research/n2-nuaa-check/`) the combined AttackNetV2.2 checkpoint shared by the coordinator (trained on other
  datasets, no NUAA in the datasets named by its code) was scored on the 3,000 frames of the N1 test; the CASIA-FASD
  → NUAA pilot (`research/m2-c2-pilot-protocol/`, LivenessNet and AttackNet V2.2) is a protocol and an implementation
  tested on synthetic data only: no run on real data. Whether the C1 images and the N2
  frames overlap with the test frames of the folds is not established here.
- The 72 new AttackNet checkpoints: never trained, never evaluated. The three trial checkpoints (§7) are not used for
  any result.

## 6 Definitions

As code with tests in `scripts/pad_anet_definitions.py`; decision rule, thresholds, bootstrap and data domain are those
of C2, imported unchanged (`pad_arch_definitions`, `nuaa_session_ablation_run`).

(a) **Variations and contrasts, exact fractions.** For a unit u (fold, seed) and a network n, v_n(u) =
ACER_n,re(u) − ACER_n,ord(u) as an exact fraction of integer counts; for an AttackNet k, c_k(u) = (A_k,re − A_k,ord) −
(L_re − L_ord), never a difference of floats. V_n and C_k are the exact means over the 12 units, converted to floats
only for output. Subject bootstrap: the 16 test subjects pooled over the folds, 1,000 resamples, `default_rng(0)`,
percentiles 2.5–97.5 (NumPy `percentile`, linear interpolation), at least 5 subjects; at every resample each unit uses
the multiplicities of the subjects of its fold, the same for every network and both BN states; the statistic is the mean
of the floats of the per-unit fractions (as `pad_arch_definitions.compare_conditions`); a resample is dropped for a
metric if a unit's value is undefined; valid resamples reported. The same for APCER and BPCER. A test checks that C_k,
its interval and its valid resamples are exactly those of the frozen C2 function `pad_arch_definitions.primary_analysis`
given the AttackNet in place of ResNet-18. The sign of the contrast (AttackNet minus LivenessNet) is a choice of the
author; the five variations are reported with their intervals and no reading.

(b) **Intervals are individual**: no correction for multiplicity, no simultaneous coverage over the contrasts,
variations or comparisons.

(c) **Reading of each contrast**, from the interval of its ACER over the three seeds together:
"ACER variation of k minus that of LivenessNet lower than zero" if entirely below zero, "… higher than zero" if
entirely above, "unresolved" if it contains zero (a bound at zero contains it) or cannot be computed. Conclusions,
written before the test, one per AttackNet, no joint statement on the three:
- lower or higher: "On these checkpoints and test frames, the ACER variation with this BN re-estimation (re-estimated
  minus ordinary BN) is algebraically lower [higher] for k than for LivenessNet; it does not identify a cause."
- unresolved: "This comparison does not show that the ACER variation with this BN re-estimation differs between k and
  LivenessNet; it does not exclude it."
- not computed: "Not computed for k (j of 12 pairs evaluable in both BN states): <reason>."

(d) **Per seed**: the same values on the 4 units of each seed, reported apart, outside the reading.

(e) **Frame identifiers**: per unit, the networks present are scored on the same frame identifiers in the same order,
also between the two BN states and between new and reused cells (`prepare`, `pair_states`); units of a fold hold the
same frames; an identifier belongs to one fold only. Outputs hold counts and SHA-256 only.

(f) **One class**: per model and BN state, on the dev of its fold at threshold 0.5, the C2 rule (`one_class_run`);
counted per network × patience × BN state, not evaluable runs apart. On the test, as a descriptive count, the models
that decide every frame one class, at threshold 0.5 and at the EER threshold of its fold, per BN state.

(g) **Rankings** (descriptive): per condition the five networks (those present in every unit; the others listed as not
ranked) ordered by mean ACER (exact fractions; ties with the same lowest rank); per unit the counts of the ranks. No
interval.

(h) **Denominators**: units (12), subjects (16), test frames summed over the units with the attack frames (APCER
denominator) and bona fide frames (BPCER denominator) per fold and in total; the valid resamples are reported with each
interval.

(i) **Percentage points** only at presentation: proportion × 100, two decimals, half away from zero from the exact
value (`pp`: exact arithmetic on a fraction, the shortest decimal representation of a float; the rule of the C2 table
package); the computations and `results.json` keep exact fractions and full-precision proportions.

(j) **Reuse**: §4.

(k)/(n) **Non-finite training, not evaluable model, guardian per network**, fixed before the runs. Unlike C2, where a
non-finite value in a new history ended the run as FAIL (`pad-arch-run` README), here (a choice of the author, so that
one network does not stop the trainings of the others):
- a non-finite value in the history does not stop the run; the quantity and the first non-finite epoch are recorded;
  if a finite `val_loss` exists, the restored epoch is that of the smallest finite `val_loss`, and the checkpoint is
  saved and evaluated provided every restored tensor is finite;
- otherwise (no finite `val_loss`, or non-finite restored weights) the model is **not evaluable**: no checkpoint, no
  phase 2, no test; also not evaluable: non-finite scores or logits on the fitting or the dev in either BN state, or
  probabilities non-finite, outside [0, 1] or of the wrong shape in the timed inference pass (one message of the frozen
  C2 function, all counted as not evaluable; the shape is already guaranteed by the registry check), or non-finite
  re-estimated statistics, checked before the re-estimation and before its outputs are written;
- no new attempt with the same seed for a non-finite case (with these threads the LivenessNet histories of C2 were
  reproduced exactly, 24 of 24); other FAIL outcomes as in C2, reported before any new attempt;
- guardian, before the primary analysis: the effective threshold of every network 0.5 in both BN states, otherwise
  stop for every network; LivenessNet and ResNet-18 with the 12 fold-seed combinations in both BN states, otherwise
  stop; an AttackNet with fewer than 12 pairs evaluable in both states: its variation, contrast and comparison with
  ResNet-18 are "not computed (j of 12 pairs evaluable in both BN states)" with the reason, the other networks go on;
  no contrast on fewer than 12 pairs; the count of the not evaluable models goes in the appendix next to the one-class
  counts;
- a run stopped at its cap or by a FAIL is resumed only for the missing trainings, with the same seeds, code and
  configuration, before the evaluation; no training, re-training or replacement of a checkpoint after the NUAA test of
  the new models is read;
- a model with non-finite test scores in either BN state is reported with the count of such frames; the evaluation
  stops before the primary analysis and the case is reported on the card: no model is excluded after its test scores
  are computed.

(m) **Descriptive comparison with ResNet-18**, at patience 4 and threshold 0.5, for each of LivenessNet and the three
AttackNets: per BN state the difference metric(n) − metric(ResNet-18), with APCER and BPCER, and V_n − V_ResNet-18,
with individual subject intervals; no reading. For LivenessNet the comparison is computed in the orientation of C2
(ResNet-18 minus LivenessNet) and reported negated, low and high swapped: means and intervals are the exact opposite of
the C2 gaps and contrast (a test checks equality).

(o) **Descriptive conditions**: the variations, contrasts and comparisons of (a) and (m) are also computed, with their
intervals and no reading, for patience 12 at threshold 0.5 and for the EER threshold at both patiences
(`descriptive_analysis`; at the EER threshold the guardian does not require 0.5); the rankings (g) for each of the 8
conditions. All three pairs of conditions are computed; a missing condition stops the evaluation.

Every model enters the comparisons, one-class models included; unfavourable results are kept; nothing is excluded
after the test is seen.

## 7 Technical trials

Command: `python -I -B scripts/pad_anet_comparison.py trial --attempt 1 --i-am-authorized --checkpoint-root <a folder
outside the repository>`, from a clean committed tree. One process, NUAA arm B, fold 1, seed 42, fitting (1,400 frames)
and dev (400) only, keys equal to the accepted card `arm_b_fold1_seed42`; 3 epochs, patience 4; no test frame; the three
checkpoints saved only outside the repository.

Conditions before the launch: `check-machine` (read only, no data, TensorFlow not imported) prints `check-machine:
machine and versions equal to the C2 trial 20261119-ARCHTRIAL-b8f7cca (11 fields); TensorFlow not imported; nothing
written` and `RESULT: PASS (check-machine)`; `check-inputs` PASS; `sw_vers` 27.2.1 and power mode 0; power adapter and
lid open; `setopt nobgnice`; no `TF_NUM_` variable; no other heavy process; `git status --porcelain` empty.

1. **LivenessNet control**, not one of the three trials: trained first with the same trainer; the five quantities of
   epochs 1–3 must be identical, difference 0, to the accepted card and to the C2 trial `20261119-ARCHTRIAL-b8f7cca`,
   and the machine and versions equal, field by field, to those recorded by the C2 trial (the rule of
   `pad_arch_run.machine_rule`, recorded without stopping the trial); otherwise the outcome is `CONTROL_FAILED`.
2. **For each AttackNet**, in the order of phase 2 of the runs: timed inference on fitting and dev; the `.keras` saved
   (`pad_arch_run.save_checkpoint`) and its SHA-256; reloaded with the SHA-256 checked just before, checked against the
   registry, weights equal byte for byte to those in memory, dev scores of the reloaded model equal to those in memory
   within 1e-5 and without discordant decisions; the scorer of the accepted runs against the evaluator; on the reloaded
   model, the ordinary dev, an in-memory copy equal byte for byte, the re-estimation, the original unchanged, the
   consistency pass and the re-estimated dev, the checks of PR #27; a second copy with the installed statistics (the
   path of the test), equal byte for byte to the re-estimated copy and with the same dev scores.
3. **"Every BN layer re-estimated"**, five checks, one failed = `FAIL`: (1) the BatchNormalization layers read from the
   graph are the five of the registry, in the order of the re-estimation, with an admitted order on the graph and the
   moving statistics equal to all the non-trainable variables (544 or 448); (2) values per channel = frames × H × W of
   the registry (91,750,400 for `bn1_1` and `bn1_final`, 22,937,600 for `bn2_1` and `bn2_final`, 1,400 for
   `bn_dense`) with the channels of the registry; (3) every `moving_mean` and `moving_variance` changed and equal to the
   installed float32 values, every other tensor identical, the original unchanged; (4) consistency within
   `consistency_rel` and `consistency_abs` and softmax of the logits within the tolerance of PR #27; (5) finite
   statistics and dev scores different between the two BN states.

Recorded: device and versions, threads requested and effective, batch, seconds and CPU seconds per epoch, inference
seconds per 1,000 frames, seconds of saving, scorer, reloading, re-estimation and checks, peak memory, parameters and
BN layers, the outcome of each check, the attempt number and the earlier attempts with their status (`--attempt N`, N =
1 + the attempts found; from the second the run id ends with `-r<N>`; the date of the run id is today's). The line
`RESULT` of `run.log` is written only after the last check of the outputs. A LivenessNet control not reproduced or an
AttackNet not evaluable is written to `run.log` (`FINDING: …`) and to `status.json` (`findings`) as soon as it is
known, and stays in the final `status.json` also if the outcome is then `FAIL` or `INCOMPLETE`. `trial/conditions.txt`
(uptime, CPU usage, power mode, command, number of `TF_NUM_` variables set, commit and number of lines of `git status
--short`), written by the author around the trial, not by the runner. Wall-time cap of the trial: 60 minutes (estimate
20, upper estimate 40), a choice of the author.

Outcomes and exit codes: PASS 0, FAIL 1 (`trial.json` and `trial.md` removed; `status.json` and `run.log` kept),
INCOMPLETE 2 (cap or SIGUSR1), CONTROL_FAILED 3, NOT_EVALUABLE 5; precedence INCOMPLETE > NOT_EVALUABLE >
CONTROL_FAILED; after a failed control the trial goes on and the outcome is given at the end.

What follows each outcome (pre-registered, a choice of the author; the same words as `trial.after_outcome` of the
configuration): whatever the outcome (PASS, FAIL, CONTROL_FAILED, NOT_EVALUABLE, INCOMPLETE), it is reported on the C5
card before anything else; no new attempt changes the recipe, the constructor or the seed; an attempt in which the
LivenessNet control was not reproduced or an AttackNet was not evaluable, as recorded in its status.json, trial.json or
run.log, counts as CONTROL_FAILED or NOT_EVALUABLE whatever its final status, FAIL and INCOMPLETE included; after
CONTROL_FAILED or NOT_EVALUABLE no new attempt and no run before the coordinator's reply; an INCOMPLETE is reported with
its cause (cap or SIGUSR1) and step; otherwise a FAIL or INCOMPLETE is corrected in the code or the conditions,
reported, and the trial repeated as a new attempt from a clean committed tree, the earlier record kept.

**Official trial** (`trial/`: `status.json`, `trial.json`, `trial.md`, `run.log.txt` = `run.log` renamed,
`conditions.txt`):

| | AttackNetV1 | AttackNetV2_1 | AttackNetV2_2 |
|---|---|---|---|
| outcome | evaluable, restored epoch 1 | evaluable, restored epoch 1 | evaluable, restored epoch 3 |
| trainable parameters, BN layers | 33,588,194; 5 | 33,588,194; 5 | 16,806,274; 5 |
| BN check (five checks, five layers) | passed | passed | passed |
| seconds per epoch (epochs 1, 2, 3) | 27.4, 26.6, 26.4 | 27.5, 27.1, 26.8 | 22.8, 22.2, 22.1 |
| process CPU seconds per epoch | 228.6, 226.1, 224.9 | 231.5, 230.6, 227.9 | 187.5, 186.3, 184.3 |
| seconds per epoch used by the plan | 26.8 | 27.1 | 22.4 |
| inference seconds per 1,000 frames; CPU / wall | 4.3; 7.65 | 4.3; 7.65 | 3.8; 7.63 |

Run `20261122-ANETTRIAL-a951ed0`, attempt 1 (no earlier attempt), **PASS**, from commit `a951ed0` with a clean tree,
configuration SHA-256 `6f5297d08d49…`, every test hook unset: started 2026-11-22T01:12:46+0100, finished 01:20:10; wall
time 443.8 s of the 60-minute cap; process CPU time 3,362.4 s; peak RSS 5,621.2 MB. Device CPU only (GPU hidden), Apple
M5 Pro (cores as named by the system: 5 Super, 10 Performance; 15 logical CPUs), 24.0 GB RAM, macOS 27.2.1; TensorFlow
2.15.0, Keras 2.15.0, Python 3.11.16, NumPy 1.26.4; intra-op 10 and inter-op 10 threads requested and effective, set by
the runner, no `TF_NUM_` variable; batch 16, 3 epochs, NUAA arm B, fold 1, seed 42, 1,400 fitting and 400 dev frames, no
test frame assembled. LivenessNet control: **reproduced** (against the accepted card, against the C2 trial
`20261119-ARCHTRIAL-b8f7cca`, machine and versions equal). No finding. For each AttackNet the reloaded weights equal
byte for byte to those in memory, and the largest absolute differences of the dev attack scores 0 between the reloaded
model and the one in memory, between the scorer and the evaluator, and between the copy with the installed statistics
and the re-estimated copy; values per channel equal to the expected ones and every moving mean and variance changed on
the five layers of each (`trial/trial.md`). Load average 2.50 at the start (`trial/conditions.txt`).
`trial/conditions.txt` holds uptime, CPU usage, command, number of `TF_NUM_` variables set (0), commit and number of
lines of `git status --short` (0) and the exit code (0), not the power mode listed above and in `trial.conditions_file`
of the configuration: the script that wrote it did not record it. The author read it with `pmset` at 01:12:42, before
the start at 01:12:46 (power mode 0, on the power adapter); no published file records that reading. Of the other
conditions before the launch, the published files attest the operating system (`trial/trial.json`, `machine.os`), no
`TF_NUM_` variable and the clean tree (`trial/conditions.txt`, `trial/status.json`), the load and CPU usage at the start
(`trial/conditions.txt`: load average 2.50, 89.80% idle) and `check-machine` PASS at 00:56 on the same code
(`test_logs/check_machine.txt`); the run right before the launch (01:12) was read by the author; the LivenessNet control
repeated the comparison of machine and versions in the trial. `check-inputs` PASS at 01:12, the open lid and
`setopt nobgnice` (set by the launch script; the process ran with nice 0) were read by the author at the launch and have
no published record; the trial repeats the checks of the inputs (`trial/run.log.txt`, "inputs PASS"). The three trial
checkpoints stay outside the repository, with their SHA-256 in `trial/trial.json`.

## 8 Cost and caps

Rule (`plan/run_plan.md`, a choice of the author): per network × patience, training = 12 × (epochs × seconds per epoch
+ seconds outside the epochs + seconds of phase 2), epochs from min(patience + 1, 12) to 12; evaluation, once, =
(454,104 test scores + 57,600 dev reproductions) × inference seconds per 1,000 frames / 1,000 (test scores: 3 seeds ×
12,614 test frames × 2 BN states per network and patience; dev reproductions: 72 × 400 × 2); CPU hours = hours × the
CPU / wall ratio of the trial. Cap of a run = (largest estimate + 2 × the largest cost of one training) × 1.5, rounded
up to a multiple of 0.5 hours of wall time: the margin covers one training at about three times its cost (3.0–3.4
times in the case cited), as the first model of the C2 run at patience 4 (49.0 seconds per epoch for LivenessNet,
against 14.4–16.3 in the other 23 LivenessNet models; ResNet-18 37.4–41.5, and 61.8 in one model at patience 4;
`public/models.csv` of the two C2 runs, `train_seconds` / `epochs_run`). Outside the estimate: loading the inputs,
assembling the test, loading each checkpoint and installing its statistics in the evaluation.

Hours and caps from the official trial (`plan/` regenerated with `plan --trial-run`, `trial/trial.md`): training of
the 72 trainings with phase 2, 5.20–6.98 hours of wall time and 39.77–53.37 CPU hours; evaluation (once), 0.59 hours of
wall time and 4.50 CPU hours; caps of the six runs: `anet_v1_p4` 2.5 h, `anet_v2_1_p4` 2.5 h, `anet_v2_2_p4` 2.0 h,
`anet_v1_p12` 2.5 h, `anet_v2_1_p12` 2.5 h, `anet_v2_2_p12` 2.0 h; sum of the six caps 14.00 hours of wall time
(`plan/run_plan.md`).

## 9 Choices of the author (not requested by the coordinator)

A closed list, also in `choices_of_the_author` of the configuration: the constructors (§2); LivenessNet as a control in
the trial (§7); six runs of 12 and their order (§4); rule (n), a departure from the C2 rule, and the guardian per
network (§6); the reading of each contrast and its conclusions (§6 c); the rankings and the denominators (§6 g, h); the
descriptive count of the one-class models on the test (§6 f); percentage points with two decimals (§6 i); the
descriptive comparison with ResNet-18 in the form of §6 (m); the descriptive conditions (§6 o); what follows each trial
outcome and the trial outcomes `CONTROL_FAILED` and `NOT_EVALUABLE` with their outputs written (§7); the attempt rule
(§7); the cap rule with the anomaly factor (§8); the public `reference_cells.csv` with the name of each reused score
file and the anchor, not the per-file SHA-256 (§4); the tolerance of the copy with the installed statistics (1e-5, no
discordant decision, that of PR #27; §7).

## 10 Limits

Those of C2 (C2 PROTOCOL §10): one recipe; one NUAA arm; dev of 2–3 subjects per fold, on which the EER threshold rests;
16 test subjects pooled over the folds (the 16 subject identifiers of the official NUAA lists, each held out once; the
subject bootstrap resamples these identifiers; whether they correspond to 16 or to 15 participants is being clarified
with the curators of NUAA, and two identifiers of one participant would be resampled as independent); 3 seeds, paired
only nominally (the same seed does not give the networks the same initial weights). In addition: a post hoc extension
written after the C2 results; three contrasts and five variations with individual intervals, without correction for
multiplicity; LivenessNet and ResNet-18 already read on this test; the recipe not adapted to the AttackNets; CPU
trainings not guaranteed bit-deterministic beyond what the control of §7 shows. Any result applies only to these
networks, recipe and data.

## 11 What follows

Delivery for the article, as asked on the card: a table of the five networks with APCER, BPCER, ACER and
denominators at patience 4 and threshold 0.5 in both BN states; a figure of the five ACER variations and the three
contrasts with their intervals; in the appendix the per-seed results, the one-class counts and the not evaluable
models; the aggregated CSV of the 8 conditions and the script that regenerates table and figure from the public files;
unfavourable results kept. The per-frame test scores of the new models stay in the restricted repository already used
for C2; public outputs hold counts and SHA-256 only.

As in C2, the dev of every new model is reproduced in both BN states before any test frame is assembled; a discrepancy
stops the evaluation.

The 72 trainings start after the pull request with the command and configuration of the runs is published and its
link is on the C5 card, unless the coordinator states otherwise on the card; the evaluation code and configuration are
published, with the link on the card, before the NUAA test is read.
