# Secondary score analysis of the 48 LivenessNet and ResNet-18 checkpoints on NUAA: results

Card C2. Run `20261122-ARCHSCORE-fe87b3d`. Aggregates only: no per-frame score, no frame identifier and no subject
identifier is in this folder. The protocol and the script are those registered in `research/pad-arch-score-followup/`
(version 0.1, commit `fe87b3d0012f0de7c59528caa6bfe4bd1e3e27f0`, pull request #35), published and linked on the card
before the run.

## Run

- One execution, on 22 November 2026 from 09:58:30 to 09:58:33 (+0100), from the registered folder at commit `fe87b3d`
  with a clean tree; `check_synthetic.py` was run first and passed; then `score_analysis.py` was run once, with the
  command of the registered `README.md`, `units.csv` of the registered folder and a new output folder. Exit code 0.
- Inputs: the `restricted/` folders of `20261121-ARCHEVAL-332fbdf`, `20261120-ARCHRUN-p4-7faa49f` and
  `20261120-ARCHRUN-p12-ea7fdca`, with their SHA-256 manifests validated immediately before the run.
- Python 3.11.16, NumPy 1.26.4, SciPy 1.16.1, Matplotlib 3.11.2 (`runs/20261122-ARCHSCORE-fe87b3d/conditions.txt`).
- No training, no model loaded, no image read.

## Reproduction

No failed reproduction. The script stops at the first check that fails and it ended normally, so for all 48 checkpoints
and both BN states: the checkpoint SHA-256 is the same in the two states and in the development and test files; the
original development EER threshold is reproduced exactly from the development scores; the public test APCER and BPCER
of `units.csv` are reproduced at the fixed threshold 0.5 and at the development EER threshold; the stored decisions at
0.5 agree with the scores; frames, identifiers and labels are paired identically within each fold; the development
partition has 400 frames and shares no frame and no identifier with the test; the identifiers are 16, none in two
folds. The primary change of the architecture gap is reproduced:
+9.983325 percentage points.

## Outputs, as written by the script

`runs/20261122-ARCHSCORE-fe87b3d/public/`: `score_metrics.csv` (96 rows), `paired_summary.csv` (24 rows),
`det_grid.csv` (1,152 rows), `det_curves.pdf`, `sensitivity.json`. `derived/` holds the two summaries fixed in
`NOTES.md` of the registration before the run, computed by `derive.py` from these outputs only: `det_mean.csv` (mean
of the twelve curves, 96 rows) and `operating_points.csv` (achieved development APCER and all-attack points).

## Change of the architecture gap and of each model (patience 4, threshold 0.5)

ACER after re-estimation minus ACER with the stored statistics, in percentage points; for the gap, ResNet-18 minus
LivenessNet. 10,000 attempted draws of the 16 manifest identifiers (PCG64, seed 20261122): the same pooled draws for
every pooled row; the within-fold row uses the second set of 10,000 draws, resampled within fold. These are
sensitivity estimates; with 16 clusters no interval has a coverage guarantee. The original interval
(1,000 draws, 926 valid) stays the one of the evaluation: +2.56 to +16.32.

| Quantity | Estimate | Percentile 95% | BCa 95% | Valid draws |
|---|---|---|---|---|
| Gap, identifiers pooled | +9.98 | +2.58 to +16.29 | +1.77 to +15.80 | 9,076 of 10,000 |
| Gap, identifiers resampled within fold | +9.98 | +3.23 to +16.01 | — | 9,929 of 10,000 |
| Gap, pooled, without fold 3 seed 44 (post hoc) | +5.97 | -0.40 to +11.32 | — | 9,076 of 10,000 |
| LivenessNet, pooled | -7.07 | -13.30 to -0.82 | -12.39 to +0.35 | 9,076 of 10,000 |
| ResNet-18, pooled | +2.91 | -1.73 to +6.74 | -1.01 to +7.73 | 9,076 of 10,000 |

Deleting one identifier at a time (16 of 16 deletions valid), the change of the gap ranges from +8.10 to +12.57.
Without the unit fold 3, seed 44 the percentile interval of the change of the gap contains zero. The BCa interval of
the LivenessNet change contains zero and its percentile interval does not. Both intervals of the ResNet-18 change
contain zero.

## AUROC and descriptive EER on the test (mean of the twelve units; proportions)

| Patience | Network | AUROC stored | AUROC re-estimated | Change | Units up / down | EER stored | EER re-estimated | Change |
|---|---|---|---|---|---|---|---|---|
| 4 | LivenessNet | 0.7056 | 0.7700 | +0.0644 | 7 / 5 | 0.3671 | 0.3160 | -0.0511 |
| 4 | ResNet-18 | 0.9113 | 0.8961 | -0.0152 | 7 / 5 | 0.1629 | 0.1742 | +0.0113 |
| 12 | LivenessNet | 0.8219 | 0.7839 | -0.0381 | 4 / 8 | 0.2835 | 0.2915 | +0.0080 |
| 12 | ResNet-18 | 0.9056 | 0.8868 | -0.0189 | 8 / 4 | 0.1657 | 0.1825 | +0.0168 |

The mean AUROC changes together with the gap: with patience 4 it rises for LivenessNet and falls for ResNet-18
(descriptive, no interval). Two checkpoint-states have a test AUROC below 0.5, both LivenessNet with patience 4 and
the stored statistics: fold 3, seed 44 (0.2781) and fold 4, seed 44 (0.4672). For the unit fold 3, seed 44 with
patience 4 the AUROC goes from 0.2781 to 0.8301 for LivenessNet and from 0.9358 to 0.9259 for ResNet-18. The test EER
is descriptive and sets no threshold.

## Thresholds chosen on the development scores (targets 5% and 10% APCER; mean of the twelve units; proportions)

| Patience | Network | Target | Test APCER stored | Test APCER re-estimated | Test BPCER stored | Test BPCER re-estimated |
|---|---|---|---|---|---|---|
| 4 | LivenessNet | 5% | 0.3405 | 0.5035 | 0.3358 | 0.1532 |
| 4 | LivenessNet | 10% | 0.3999 | 0.5403 | 0.3083 | 0.1398 |
| 4 | ResNet-18 | 5% | 0.6311 | 0.6114 | 0.0178 | 0.0026 |
| 4 | ResNet-18 | 10% | 0.6733 | 0.6499 | 0.0174 | 0.0001 |
| 12 | LivenessNet | 5% | 0.4881 | 0.4826 | 0.1312 | 0.1468 |
| 12 | LivenessNet | 10% | 0.5325 | 0.5192 | 0.1171 | 0.1380 |
| 12 | ResNet-18 | 5% | 0.6185 | 0.6181 | 0.0178 | 0.0129 |
| 12 | ResNet-18 | 10% | 0.6710 | 0.6589 | 0.0174 | 0.0009 |

No all-attack operating point: no checkpoint-state has development APCER 0 with development BPCER 1. Of the 96
checkpoint-states, the achieved development APCER at the 5% target is 0 in 1, 0.025 in 2 and 0.05 in 93; at the 10%
target it is 0 in 1, 0.025 in 2, 0.085 in 1 and 0.1 in 92 (`derived/operating_points.csv`): where it is below the
target, it is because tied scores stay together. In one checkpoint-state (LivenessNet, patience 4, stored statistics,
fold 3, seed 44, the one with test AUROC 0.2781) the chosen threshold reaches the target at both 5% and 10% with
development BPCER 1: every bona fide development frame is rejected, which is not an all-attack point by the registered
definition (`score_metrics.csv`). On the test the mean APCER at these thresholds is between 0.3405 and 0.6733 in the
sixteen means above: a development target does not carry over to the test.

## Mean of the twelve DET curves (patience 4; minimum attainable test BPCER at each APCER ceiling)

| Network | BN state | APCER ≤ 0.01 | APCER ≤ 0.05 | APCER ≤ 0.1 | APCER ≤ 0.2 | APCER ≤ 0.4 |
|---|---|---|---|---|---|---|
| LivenessNet | stored | 0.6037 | 0.5112 | 0.4694 | 0.4333 | 0.3418 |
| LivenessNet | re-estimated | 0.5780 | 0.4799 | 0.3967 | 0.3481 | 0.2790 |
| ResNet-18 | stored | 0.5055 | 0.3550 | 0.2563 | 0.1360 | 0.0455 |
| ResNet-18 | re-estimated | 0.4790 | 0.3141 | 0.2293 | 0.1465 | 0.1057 |

All twelve ceilings and patience 12 are in `derived/det_mean.csv`; the curves per checkpoint are in `det_grid.csv` and
in `det_curves.pdf`.

## Limits

- The grouping is by manifest identifier: 16 identifiers, whose correspondence with the 15 participants of the dataset
  paper is being clarified by the coordinator with the curators of NUAA; no interval here has a coverage guarantee.
- The outcomes of the evaluation, the influential unit included, were known when the protocol was written; the
  analysis without fold 3, seed 44 is a post hoc influence diagnostic.
- Equal or different AUROC does not show identical rankings or calibrated probabilities; no calibration was assessed.
- The re-estimation with augmentation and dropout (`PROTOCOL.md` §4 of the registration) is a later step.
- The last line of `run.log.txt` is written by the launch wrapper in Italian ("fork ancora pulito": the tree of the
  registered folder was still clean after the run).
