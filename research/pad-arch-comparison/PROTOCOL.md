# LivenessNet and ResNet-18 under early stopping, BatchNormalization and threshold (protocol)

**Status: draft of the pre-registration, 18 November 2026** (coordinator's assignment, C2 card, 17 November 2026). No
test frame is assembled or evaluated and no external dataset is used; the only trainings are the two technical trials
of §7. The tests and the external runs will follow the pre-registration and the confirmation of access.

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
| input, recipe | 256×256×3, frame / 255; CPU; Adam 1e-3, clipnorm 1.0; batch 16; at most 12 epochs; early stopping on the dev `val_loss`, best weights restored; ReduceLROnPlateau; balanced class weights; augmentation on fitting frames only | the same |
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

Patience changes only `early_stopping_patience`; with at most 12 epochs, patience 12 never stops a training. The
re-estimation is the estimator of PR #27 (only `moving_mean` and `moving_variance` change, on a copy), extended to
ResNet-18. Before use it must reproduce, on the accepted patience-4 LivenessNet checkpoints, the statistics of run
`20261114-BNRECAL-a10a082` within the tolerance of PRs #27 and #28 (1e-4 relative plus 1e-6 absolute); otherwise the
work stops and the discrepancy is documented. The EER threshold is the project rule (`eer_threshold`,
`scripts/nuaa_folds_run.py`) on the dev of the fold, with the model in the same BN state. Attack if score >
threshold. 8 conditions per network.

## 4 Planned runs

| dataset | network | patience | new trainings | epochs | fitting / dev per unit | training h | evaluation h (at most) | elapsed h | CPU h |
|---|---|---|---|---|---|---|---|---|---|
| NUAA | LivenessNet | 4 | 12 | 5–12 | 1,400 / 400 | 0.46–1.10 | 0.27 | 0.73–1.37 | 5.06–9.44 |
| NUAA | LivenessNet | 12 | 12 | 12 | 1,400 / 400 | 1.10 | 0.27 | 1.37 | 9.44 |
| NUAA | ResNet-18 | 4 | 12 | 5–12 | 1,400 / 400 | 1.16–2.78 | 1.66 | 2.82–4.44 | 22.86–35.98 |
| NUAA | ResNet-18 | 12 | 12 | 12 | 1,400 / 400 | 2.78 | 1.66 | 4.44 | 35.98 |
| **NUAA, total** | both | 4, 12 | 48 | | | 5.51–7.76 | 3.86 | 9.36–11.62 | 73.34–90.84 |
| **CelebA-Spoof, total** | both | 4, 12 | 12 (3 units) | 5–12 | 1,400 / 400 | 1.38–1.94 | test size to be fixed | | |
| **Replay-Attack, total** | both | 4, 12 | 12 (3 units) | 5–12 | 1,400 / 400 | 1.38–1.94 | test size to be fixed | | |

Hours: elapsed time on the machine of the trial; CPU hours: process CPU time over all threads. Evaluation per
training: two BN states × two thresholds. Test frames per NUAA fold: 3,242, 3,114, 3,439, 2,819. Totals are computed
before rounding. Full table: `plan/run_plan.md`.

**NUAA**: arm B of the session ablation, folds 1–4 × seeds 42, 43, 44 = 12 units, 48 trainings, all new, one trainer,
one machine. The accepted LivenessNet checkpoints (`20261106-SESSABLRUN-d999458`, patience 4;
`20261109-SESSABLRUN-6a50265`, patience 12) are not reused: they are the reference of the check of §7.

**External**, after access is confirmed: official identity-disjoint split × seeds 42, 43, 44 (3 units). Proposed:
fitting and dev subsampled to 1,400 and 400 frames, balanced by class. Frame sampling, face crop and test size are
fixed in the pre-registration after the official archives are seen, not their test images (`DATASETS.md`).

## 5 Already measured and not

- LivenessNet, NUAA arm B, patience 4 and 12, ordinary BN, both thresholds: test analysed on the accepted checkpoints
  (PR #21, PR #23).
- LivenessNet, NUAA, patience 4, re-estimated BN, threshold 0.5: test analysed on the accepted checkpoints (PR #28).
- LivenessNet, NUAA, other cells: not analysed; same test frames.
- ResNet-18: trained only in the trial of §7; never evaluated on a test.
- CelebA-Spoof, Replay-Attack: used before only without an identity-disjoint split (Replay-Attack in the coordinator's
  reference code, `scripts/create_datasets.py`, with a label-stratified 80/20 split of the images; CelebA-Spoof here
  through one shard of an unofficial mirror); never with the official identity split.

On NUAA the comparison is therefore not an independent validation. The accepted model of arm B, fold 1, seed 42,
patience 4 is one of the ten "degenerate" trainings.

## 6 Definitions

**One-class run at threshold 0.5.** Per model and BN state, on the dev of its fold: "not evaluable" if the dev lacks a
class (outside the denominator); otherwise "one-class" if and only if 0 or all frames are decided attack. Reported: k
of n evaluable runs per network × patience × BN state, with the class. The earlier "degenerate" criterion is a
different count; both are reported where defined.

**Comparison.** Unit = (dataset, fold or split, seed). Per unit and condition, Δ = ACER(ResNet-18) − ACER(LivenessNet)
on the same test frames; the same for APCER and BPCER; tie = equal ACER as exact fractions. Per condition and
dataset: mean Δ; counts of Δ < 0, Δ > 0 and ties; the subject bootstrap of the project (`definitions.bootstrap`; test
subjects pooled over the folds, 1,000 resamples, seed 0, percentiles 2.5–97.5, at least 5 subjects, same indices for
both networks; a resample is dropped if any unit's metric is undefined; valid resamples reported). The same values
per seed.

**Sign.** "ResNet-18 lower" if the interval lies entirely below zero, "LivenessNet lower" if entirely above,
"unresolved" otherwise (a bound at zero counts as containing it); read on the three seeds together. A **change of
sign** is recognised only between two resolved conditions of opposite sign. Reported apart, under other names: an
inversion of the point estimate; the paired contrast Δ_c − Δ_c′ with its interval; the units whose Δ changes sign.
Secondary reading, under its own name: per-seed reading (the rule of PR #21): the same non-zero sign for each seed,
each per-seed interval excluding zero; reported next to the sign from the per-seed values the code already computes;
its function is added as code with the final pre-registration.

**Subject sensitivity.** Per condition, the mean Δ excluding one test subject at a time: minimum, maximum, exclusions
changing the sign of the point estimate (zero a sign of its own), denominators per class (absent class: NA). In the
form of the N1 analysis (PR #10); it enters neither rule.

Every model enters the comparisons, one-class models included; nothing is excluded after the test is seen.

## 7 Technical trial

One per network: NUAA arm B, fold 1, seed 42, fitting (1,400 frames) and dev (400) only, 3 epochs, recipe otherwise
unchanged, no checkpoint saved, no test frame assembled. Check fixed before the trial (configuration committed before
it, SHA-256 in the record): the LivenessNet history (`loss`, `accuracy`, `val_loss`, `val_accuracy`, `lr`) identical,
difference 0, to epochs 1–3 of the accepted card, the rule of the patience verification; if not met, the difference is
reported and the pre-registration is not closed before it is explained. In the plan, a LivenessNet history that
differs from the accepted card is reported per model with its first differing epoch; the run continues and the model
stays in the comparison (every model enters). The training order of the plan is fixed in the final
pre-registration. The ResNet-18 history is a technical output, not a result.

| | LivenessNet | ResNet-18 |
|---|---|---|
| command, configuration | `python -I -B scripts/pad_arch_comparison.py trial --i-am-authorized`; `pad_arch_comparison.json`, SHA-256 `9264ec0201c1…` | the same run |
| device | CPU only (GPU hidden), Apple M4, 16 GB RAM, macOS 27.2.1; TensorFlow 2.15.0 | the same |
| batch | 16 | 16 |
| seconds per epoch (epochs 1, 2, 3) | 26.8, 26.4, 28.4 | 69.0, 70.0, 69.2 |

Run `20261118-ARCHTRIAL-3f059b7`, PASS: started 10:59:46, finished 11:05:23 (+0100); elapsed 336.7 s (`status.json`,
written later: 336.8) of a 60-minute elapsed-time cap; process CPU time 2,555.9 s; peak memory 2,249.6 MB. Check:
**reproduced**, largest difference 0; fitting and dev keys equal to the card's. Other times: `trial/trial.md`.

## 8 Cost

NUAA, 48 trainings with evaluation: **9.36–11.62 hours of elapsed time** on this machine (training 5.51–7.76,
evaluation at most 3.86), that is 73.34–90.84 process CPU hours over all threads (CPU / elapsed 6.90 and 8.11). Each
external dataset: 1.38–1.94 elapsed hours for the training alone; evaluation not estimated until the test size is
fixed (measured rate: 5.2 and 13.6 s per 1,000 frames). As an example of scale, not a value of the plan: 10,000 test
frames scored in the two BN states add about 105 s (LivenessNet) and 272 s (ResNet-18) per training. Rule:
`plan/run_plan.md`.

Every time limit carries its unit: elapsed time on this machine and, separately, process CPU hours over all threads.
The final status of every run records `started`, `finished`, the elapsed time and the process CPU time. Each run of
the plan will have a cap stated in elapsed hours and, separately, an estimate in CPU hours, fixed in the final
pre-registration.

## 9 Choices of the author (not requested by the coordinator)

1. All 48 NUAA trainings new; accepted checkpoints only the reference of the check (§4, §7).
2. ResNet-18 with projection shortcuts, option B (§2).
3. Sign read on the three seeds together (§6).
4. Leave-one-subject-out sensitivity in the form of the N1 protocol (§6).
5. One NUAA arm, arm B (§4).
6. Proposed budget for the external datasets (§4).

## 10 Limits

One recipe; one NUAA arm; dev of 2–3 subjects per fold, on which the EER threshold rests; 16 test subjects pooled
over the folds; 3 seeds, paired only nominally (the same seed does not give the networks the same initial weights);
eight conditions without correction for multiplicity, so a change of sign is a descriptive reading; CPU trainings not
guaranteed bit-deterministic (the trial reproduced the 3 epochs exactly, on this machine); all trainings on the same
machine. Any result applies only to these networks, recipe and data.

## 11 What follows

The final pre-registration, after the coordinator's observations; the trainings of both networks on NUAA; the NUAA
test according to the pre-registration; the external datasets after access is confirmed.
