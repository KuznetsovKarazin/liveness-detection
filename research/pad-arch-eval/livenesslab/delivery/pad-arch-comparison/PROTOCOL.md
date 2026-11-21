# LivenessNet and ResNet-18 under early stopping, BatchNormalization and threshold (protocol)

**Status: final pre-registration of the NUAA part, 19 November 2026** (coordinator's comments on the C2 card of 17 and
19 November 2026). It amends the draft of 18 November (PR #29, commit `e004b20`); the changes are listed in
`README.md`. The external datasets will be pre-registered separately after access is confirmed. No test frame
assembled or evaluated, no test score or metric read; the only trainings are the technical trials (§7).

## 1 Question

Do early stopping, BatchNormalization (BN) and the threshold change which of two PAD networks, LivenessNet and
ResNet-18, has the lower error?

## 2 What stays fixed

Both networks are trained from scratch (no pre-trained weights), with the same input, preprocessing, recipe, subject
splits and training seeds.

| | LivenessNet | ResNet-18 |
|---|---|---|
| definition | `create_model("LivenessNet")`, `src/livedetection/src/architectures.py`, unchanged | "18-layer" column of Table 1 of He et al. (CVPR 2016), 2-way softmax |
| BN | after each ReLU: 5 layers | after each convolution, before the activation (§3.4): 20 layers |
| BN parameters and other | Keras defaults (momentum 0.99, epsilon 1e-3); dropout | the same BN defaults; no dropout; projection shortcuts where the dimension changes: option B of He et al. (§3.3), as in the torchvision reference implementation, while their 18-layer results (Table 2) use option A; Keras `he_normal` on the convolutions, Keras default on the 2-way layer |
| trainable parameters | 8,405,778 | 11,177,538 |
| input, recipe | 256×256×3, frame / 255; CPU, 10 intra-op and 10 inter-op TensorFlow threads (§7); Adam 1e-3, clipnorm 1.0; batch 16; at most 12 epochs; early stopping on the dev `val_loss`, best weights restored; ReduceLROnPlateau; balanced class weights; augmentation on fitting frames only | the same |
| splits, seeds | subject folds of the accepted runs; seeds 42, 43, 44 | the same |

Asymmetries: the recipe is that of the accepted LivenessNet runs (patience 12 was chosen from LivenessNet's dev
curves, PR #22), not adapted to ResNet-18 (He et al.: 224×224, SGD, weight decay, batch 256); dropout; BN position;
parameters.

## 3 Factors

| factor | level 1 | level 2 | needs a training |
|---|---|---|---|
| early stopping | patience 4 | patience 12 | yes |
| BN statistics | ordinary | re-estimated on the fitting set only | no |
| threshold | 0.5 | EER threshold estimated on the source dev | no |

With at most 12 epochs, patience 12 never stops a training. The re-estimation is the estimator of PR #27 (only
`moving_mean` and `moving_variance` change, on a copy), extended to ResNet-18; it must first reproduce the statistics
of run `20261114-BNRECAL-a10a082` within the tolerance of PRs #27 and #28, otherwise the work stops. The EER threshold
is the project rule (`eer_threshold`, `scripts/nuaa_folds_run.py`) on the dev of the fold, in the same BN state.
Attack if score > threshold. 8 conditions.

## 4 Planned runs

| dataset (access) | network | patience | new trainings | epochs | fitting / dev per unit | training h | evaluation h (at most) | elapsed h | CPU h |
|---|---|---|---|---|---|---|---|---|---|
| NUAA | LivenessNet | 4 | 12 | 5–12 | 1,400 / 400 | 0.23–0.56 | 0.12 | 0.36–0.68 | 2.91–5.52 |
| NUAA | LivenessNet | 12 | 12 | 12 | 1,400 / 400 | 0.56 | 0.12 | 0.68 | 5.52 |
| NUAA | ResNet-18 | 4 | 12 | 5–12 | 1,400 / 400 | 0.62–1.49 | 0.94 | 1.56–2.43 | 14.20–22.08 |
| NUAA | ResNet-18 | 12 | 12 | 12 | 1,400 / 400 | 1.49 | 0.94 | 2.43 | 22.08 |
| **NUAA, total** | both | 4, 12 | 48 |  |  | 2.90–4.09 | 2.13 | 5.03–6.22 | 44.70–55.20 |
| **CelebA-Spoof, total** (access missing) | both | 4, 12 | 12 (3 units) | 5–12 | 1,400 / 400 | 0.73–1.02 | test size to be fixed |  |  |
| **Replay-Attack, total** (access missing) | both | 4, 12 | 12 (3 units) | 5–12 | 1,400 / 400 | 0.73–1.02 | test size to be fixed |  |  |

Hours: elapsed time on the machine of the official trial (Apple M5 Pro, §7); CPU hours: process CPU time over all
threads. Evaluation per training: two BN states × two thresholds. Test frames per NUAA fold: 3,242, 3,114, 3,439,
2,819. Totals before rounding; full table: `plan/run_plan.md`.

**NUAA**: arm B, folds 1–4 × seeds 42, 43, 44 = 12 units, 48 new trainings in two runs of 24 (§7, §8). The accepted
LivenessNet checkpoints (`20261106-SESSABLRUN-d999458`, patience 4; `20261109-SESSABLRUN-6a50265`, patience 12) are
not reused: they are the reference of the check of §7. **External** (`DATASETS.md`): official identity-disjoint split
× 3 seeds, fitting and dev at the arm B budget (proposal, `plan/run_plan.md`); their own pre-registration.

## 5 Already measured and not

- LivenessNet, NUAA arm B, patience 4 and 12, ordinary BN, both thresholds: test analysed on the accepted checkpoints
  (PR #21, PR #23); patience 4, re-estimated BN, threshold 0.5: PR #28; other cells: not analysed; same test frames.
- ResNet-18: trained only in the trials of §7; never evaluated on a test.
- CelebA-Spoof, Replay-Attack: used before only without an identity-disjoint split (Replay-Attack in the coordinator's
  reference code; CelebA-Spoof through one shard of an unofficial mirror).

On NUAA the comparison is therefore not an independent validation; the accepted model of arm B, fold 1, seed 42,
patience 4 is one of the ten "degenerate" trainings. For the primary analysis (§6): for LivenessNet both of its cells
have already been analysed on this test with the accepted checkpoints (PR #21, PR #28). The new LivenessNet trainings
repeat the accepted ones with the same recipe, data and seeds, and the history check (§7) will say whether they follow
the same history; an identical history does not prove identical weights. In any case the LivenessNet side of the
primary analysis repeats cells already read on this test; only the ResNet-18 side is new.

## 6 Definitions

**Primary analysis.** Requested by the coordinator on 19 November 2026: patience 4 and threshold 0.5; the gap Δ =
ACER(ResNet-18) − ACER(LivenessNet) with ordinary BN and with BN re-estimated on the fitting set only; the primary
contrast = Δ(re-estimated) − Δ(ordinary), the first gap subtracted from the second; the per-seed results reported
apart. Choices of the author (`pad_arch_definitions.primary_analysis`, which picks the two conditions itself): on the
12 units together, the mean and the subject-bootstrap interval of the two gaps and of the contrast (same resampling
indices for both conditions and networks); the same values for APCER and BPCER; for each seed, the same values on its
units alone; the reading below.

*Reading of the contrast* (a choice of the author, `pad_arch_definitions.contrast_reading`), from the interval of the
ACER contrast over the three seeds together: "ACER difference (ResNet-18 - LivenessNet) lower with re-estimated BN" if
entirely below zero, "… higher with re-estimated BN" if entirely above, "unresolved" if it contains zero (a bound at
zero contains it) or cannot be computed (`CONTRAST_LOWER`, `CONTRAST_HIGHER`, `CONTRAST_UNRESOLVED`; the field
`reason` tells the two cases apart). Per-seed values do not enter the reading. Conclusions, written before the test:
if the interval excludes zero, "on these checkpoints and test frames the signed ACER difference (ResNet-18 −
LivenessNet) is algebraically lower [higher] with this re-estimation; it does not identify a cause and says nothing on
which network has the lower ACER in each BN state (that is read from the two gaps, reported with their intervals)";
otherwise "this comparison does not show that the re-estimation changes the difference; it does not exclude it".

*Secondary analyses* (a choice of the author): the other six conditions and every comparison below are descriptive.

**Frame identifiers** (requested by the coordinator; `pad_arch_definitions.prepare_units`, `same_frames`). Every score
carries its frame identifier (`sample_id`, the key of the score files of the accepted runs). The comparison stops if
the two networks of a unit, or a unit in two conditions, do not have the same identifiers in the same order, also with
the same subjects and classes (one test for each case). A choice of the author: units of the same split hold the same
frames, and an identifier belongs to one split only. Outputs hold no identifier, only counts and a SHA-256.

**One-class run at threshold 0.5.** Per model and BN state, on the dev of its fold: "not evaluable" if the dev lacks a
class (outside the denominator); otherwise "one-class" if and only if 0 or all frames are decided attack. Reported: k
of n evaluable runs per network × patience × BN state, with the class (the earlier "degenerate" count apart).

**Comparison.** Unit = (dataset, fold or split, seed). Per unit and condition, Δ = ACER(ResNet-18) − ACER(LivenessNet)
on the same test frames, and the same for APCER and BPCER; tie = equal ACER as exact fractions. Per condition: mean Δ;
counts of Δ < 0, Δ > 0 and ties; the subject bootstrap of the project (test subjects pooled over the folds, 1,000
resamples, seed 0, percentiles 2.5–97.5, at least 5 subjects, same indices for both networks, a resample dropped if a
unit's metric is undefined, valid resamples reported); the same per seed.

**Sign.** "ResNet-18 lower" if the interval lies entirely below zero, "LivenessNet lower" if entirely above,
"unresolved" otherwise; read on the three seeds together. A **change of sign** only between two resolved conditions of
opposite sign; reported apart: inversion of the point estimate, the paired contrast Δ_c − Δ_c′ with its interval, the
units whose Δ changes sign. Per-seed reading (`pad_arch_definitions.per_seed_reading`): the rule of PR #21 on the
three seeds of the plan (a missing seed gives "inconclusive"); it enters neither rule.

**Subject sensitivity.** Per condition, the mean Δ excluding one test subject at a time (minimum, maximum, exclusions
changing the sign, denominators per class; absent class: NA), in the form of the N1 analysis (PR #10).

Every model enters the comparisons, one-class models included; nothing is excluded after the test is seen.

## 7 Technical trial, threads and order

One per network: NUAA arm B, fold 1, seed 42, fitting (1,400 frames) and dev (400) only, 3 epochs, no checkpoint, no
test frame. Check fixed before the trial: the LivenessNet history (five fields) identical, difference 0, to epochs 1–3
of the accepted card. In the plan a differing history is reported per model with its first differing epoch; the model
stays in the comparison. The ResNet-18 history is a technical output, not a result.

**Threads.** Requested by the coordinator for the whole NUAA plan (our proposal): 10 intra-op and 10 inter-op
TensorFlow threads, set and recorded by the runner. How (a choice of the author): applied to trainings and evaluations
(our reading of "the whole NUAA plan"); set through `tf.config.threading` before any TensorFlow operation and read
back; contradicting `TF_NUM_INTRAOP_THREADS` or `TF_NUM_INTEROP_THREADS` refused; requested and effective values, the
variables found and the logical CPUs recorded. Machine rule (`plan.machine_rule`, our proposal of 19 November): the
whole NUAA plan on the machine of the official trial with the same threads and versions; a declared rule that the
runner of the 48 trainings will have to apply: today no code compares the machine.

**Official trial** (`trial/`):

| | LivenessNet | ResNet-18 |
|---|---|---|
| command, configuration | `python -I -B scripts/pad_arch_comparison.py trial --i-am-authorized`; `pad_arch_comparison.json`, SHA-256 `5a83325e13d2…` | the same run |
| device | CPU only (GPU hidden), Apple M5 Pro (cores as named by the system: 5 Super, 10 Performance; 15 logical CPUs), 24 GB RAM, macOS 27.2.1; TensorFlow 2.15.0; intra-op 10 and inter-op 10 TensorFlow threads, set by the runner | the same |
| batch | 16 | 16 |
| seconds per epoch (epochs 1, 2, 3) | 13.8, 13.7, 13.9 | 37.6, 36.8, 37.1 |
| process CPU seconds per epoch | 112.6, 114.7, 115.9 | 342.9, 341.1, 343.3 |

Run `20261119-ARCHTRIAL-b8f7cca`, PASS, from commit `b8f7cca` with a clean tree: started 21:04:47, finished 21:07:47
(+0100); elapsed 180.0 s of a 60-minute elapsed-time cap; process CPU time 1,534.1 s; peak memory 5,299.1 MB. Load
average 2.23 at the start (`trial/conditions.txt`, written by the author, not by the runner). Check: **reproduced**,
largest difference 0; fitting and dev keys equal to the card's. Inference times: `trial/trial.md`.

**Earlier trials** (`trials_previous/README.md`, with errata): on the Apple M5 Pro with the threads not set by the
runner the LivenessNet history is **not reproduced** (`0858ae7`, different from epoch 1); with the two variables at 10
it is reproduced (`b8e78b2`).

**Check requested by the coordinator** (`trial_comparison/`): against `20261119-ARCHTRIAL-b8e78b2` (code of PR #29,
`TF_NUM_INTRAOP_THREADS=10 TF_NUM_INTEROP_THREADS=10` on the command line): **reproduced**, difference 0 in the five
fields over the 3 epochs of both networks, computed from the two records. The variables are declared by a conditions
file written by the author around that trial (`trial_comparison/reference_conditions.txt`, SHA-256 `591e60bfd564…`),
not by its record (format /1 records no environment variables); the code checked only that the file names the commit
of that record and sets both variables to the required 10.

**Correction.** In PR #29 `first_differing_epoch` held the epoch of the largest difference, not the first differing
epoch (our defect); corrected with a test. The earlier records are not modified.

**Order** (`plan/run_order.csv`, a choice of the author): two runs, one process each, patience 4 then 12; in each,
folds 1–4, seeds 42, 43, 44, LivenessNet then ResNet-18, so every LivenessNet training has the position in its process
that the arm B model of the same fold and seed had in the accepted run (preceded by a ResNet-18 model instead of an
arm A* model). In the accepted runs (Apple M4, default threads) a model's history did not change when earlier models
trained for more epochs (shared epochs identical in 36 of 36); a ResNet-18 model before it is not measured: the
history check of every LivenessNet training is the direct measure.

## 8 Cost and caps

NUAA, 48 trainings with evaluation: **5.03–6.22 hours of elapsed time** on the Apple M5 Pro (training 2.90–4.09,
evaluation at most 2.13), that is 44.70–55.20 process CPU hours over all threads (CPU / elapsed ratio measured in the
official trial: 8.11 for LivenessNet, 9.09 for ResNet-18; `trial/trial.md`). The rates are those of the official
trial, with a load average of 2.23 at its start (§7). Each external dataset: 0.73–1.02 elapsed hours for the training
alone; evaluation not estimated until the test size is fixed (measured rate: 2.4 and 7.7 s per 1,000 frames). Rule:
`plan/run_plan.md`.

The final status of every run records `started`, `finished`, the elapsed time and the process CPU time. Caps
(`plan/run_caps.csv`):

| run | trainings | estimate, elapsed hours | cap, elapsed hours | estimate, CPU hours |
|---|---|---|---|---|
| `nuaa_patience_4` (patience 4) | 24 | 1.92–3.11 | 5.00 | 17.10–27.60 |
| `nuaa_patience_12` (patience 12) | 24 | 3.11 | 5.00 | 27.60 |

Rule (a choice of the author): cap = the maximum estimate × 1.5, rounded up to a multiple of 0.5 hours; estimates:
sums of unrounded values; unit: hours of elapsed time, not CPU time; the CPU-hour estimate is not a cap. The margin
covers what the estimate leaves out (loading the inputs, saving the checkpoints, assembling the test) and a busier
machine; a run at its cap stops as INCOMPLETE.

## 9 Choices of the author (not requested by the coordinator)

All 48 NUAA trainings new (§4); ResNet-18 option B (§2); sign read on the three seeds together and the
leave-one-subject-out sensitivity (§6); one NUAA arm (§4); the budget of the external datasets (§4); the reading of
the primary contrast, the six secondary conditions and the frame rule between units (§6); the subject-bootstrap
intervals, APCER and BPCER and the per-seed values in the primary analysis (§6); the threads applied to every
TensorFlow command of the NUAA plan, evaluations included (our reading of "the whole NUAA plan"), and contradicting
`TF_NUM_` variables refused (§7); the order, the two runs and the caps (§7, §8); the earlier trials kept
(`trials_previous/`); `compare-trials` and its rule (difference 0). The number of threads was our proposal, fixed by
the coordinator (§7).

## 10 Limits

One recipe; one NUAA arm; dev of 2–3 subjects per fold, on which the EER threshold rests; 16 test subjects pooled over
the folds; 3 seeds, paired only nominally (the same seed does not give the networks the same initial weights); eight
conditions without correction for multiplicity. The machine changed between the draft (Apple M4) and this
pre-registration (Apple M5 Pro). CPU trainings are not guaranteed bit-deterministic: with these threads the Apple M5
Pro reproduced the accepted LivenessNet history on 3 epochs of one unit. In an unofficial check outside this package
(18 November 2026, threads imposed with the two environment variables at 10; reported to the coordinator in the
comment of 19 November on the C2 card), the score files and BN statistics files of the 36 accepted checkpoints in the
baseline check with the BN re-estimation of PR #27 (fitting and dev only) were byte for byte those of the accepted run
`20261114-BNRECAL-a10a082`. The results hold for this thread setting; same machine and threads is a declared rule, not
yet checked by code. In the primary analysis only the ResNet-18 side is new (§5). Any result applies only to these
networks, recipe and data.

## 11 What follows

The coordinator verifies and merges. The command of the two NUAA runs is not part of this pre-registration: it will be
published, with its configuration, in a new pull request before the runs, and it will apply and record the machine and
thread rule (§7). The 48 NUAA trainings start only after the coordinator's go-ahead; the NUAA test according to this
pre-registration, with the code of the evaluation published before the test is read; the external datasets after
access is confirmed.