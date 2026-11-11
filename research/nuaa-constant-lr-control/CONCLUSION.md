# Conclusion: constant learning-rate control, runs `20261111-SESSABLRUN-8568ffa` and `20261111-LRCTRL-ff53279`

**What was done.** On 10 November 2026 the coordinator confirmed on the C2 card the control proposed in section 5 of
`delivery/synthesis/SYNTHESIS.md` (PR #24): the same 36 LivenessNet models (arms B, A\* and B' x 4 folds x seeds
42/43/44) and data as the patience-12 run `20261109-SESSABLRUN-6a50265`, `early_stopping_patience` 12, at most 12
epochs, the learning rate constant at 0.001 (no ReduceLROnPlateau), the cap of 360 minutes, the test separate; before
the run, configuration and runner updated so that only the scheduler changes and the original path stays unchanged; the
effective learning rate recorded at every epoch; every outcome kept; the identity of the histories checked before the
first learning-rate reduction in the reference; on a discrepancy or over budget, stop and report. The protocol fixed
before the run is `NOTE.md`. The configuration (`60b38e094da8`) is the patience-12 one (`ca039df5d311`) with
`training.scheduler` changed and the `amendment` block; the runner refuses it, before reading any data, unless both
SHA-256 are the pinned ones and every other field is equal. The training run `20261111-SESSABLRUN-8568ffa` (clean
commit `8568ffa`) ended PASS, 36 of 36 models complete, no discrepancy, within the cap. A separate script, which trains
nothing and reads no test score, compared its training cards and dev scores with those of the reference run in run
`20261111-LRCTRL-ff53279` (PASS), by the reading rule (a)-(e) fixed before the run. Numbers from its `aggregate.md` and
`comparison.csv` (`comparison/` in the package) and from the training cards for single epochs (`curves.csv` for the
control run).

**(a) Shared histories: identical, the comparison is valid.** For every model the epochs of the reference run
recorded at the recipe rate (0.001 as Keras records it, float32 0.0010000000474974513), up to the epoch at whose end
the first reduction happened, are identical in the control run in `loss`, `accuracy`, `val_loss`, `val_accuracy` and
`lr`: 36 of 36, largest absolute difference 0.0 in the five fields. The windows are 3 epochs in 12 models, 4 in 6, 5 in
5, 6 in 1, 8 in 7, 9 in 1, 11 in 3 and 12 in 1 (A\* fold 1 seed 42, never reduced in the reference run: its whole
history is identical). The rate of the control run is 0.001 at every epoch of every model (36 of 36; one recorded
value). The control was run as fixed.

**(b) Primary outcome: mixed, no overall verdict** (the third case of the rule). The ten original degenerate trainings
all recover within 12 epochs with the constant rate, and none of the 36 models is degenerate; but one model outside the
ten, B' fold 2 seed 42, which recovered in the reference run, does not recover here. By the rule fixed before the run (`NOTE.md`, (b): "A mixed outcome (for
example all ten recover but another model becomes degenerate or loses its recovery) is reported as such, model by
model"; `reading_rule.b` of the configuration of the comparison, fixed before the run as well, adds "with no overall
verdict"), the outcome is reported model by model and there is no overall verdict:

| the ten | first recovery (reference) | first reduced epoch (reference) | first recovery (control) | recovered at epoch 12 (control) | restored epoch (reference -> control) | dev accuracy at the checkpoint (control) | dev AUROC (control) |
|---|---|---|---|---|---|---|---|
| A\* fold 2 seed 43 | 7 | 4 | 6 | yes | 10 -> 9 | 1.0000 | 1.0000 |
| A\* fold 2 seed 44 | 10 | 4 | 5 | yes | 12 -> 10 | 1.0000 | 1.0000 |
| A\* fold 3 seed 42 | 9 | 4 | 6 | no | 9 -> 10 | 0.9900 | 1.0000 |
| A\* fold 3 seed 44 | 11 | 4 | 7 | yes | 11 -> 8 | 0.8750 | 1.0000 |
| A\* fold 4 seed 44 | 10 | 4 | 6 | yes | 12 -> 8 | 0.9975 | 1.0000 |
| B fold 1 seed 42 | 8 | 4 | 4 | yes | 10 -> 9 | 1.0000 | 1.0000 |
| B fold 4 seed 44 | 7 | 5 | 10 | yes | 11 -> 11 | 1.0000 | 1.0000 |
| B' fold 1 seed 44 | 7 | 4 | 6 | yes | 9 -> 6 | 0.9975 | 0.9975 |
| B' fold 2 seed 44 | 12 | 5 | 7 | yes | 12 -> 12 | 0.9950 | 1.0000 |
| B' fold 4 seed 44 | 10 | 4 | 10 | no | 11 -> 10 | 0.8900 | 0.9831 |

For the ten, the recovery of the reference run is reproduced without the reduction: first recovery at epoch 4 to 10
(median 6) against 7 to 12 (median 9.5) in the reference run, earlier in 8 of the ten, at the same epoch in one (B'
fold 4 seed 44, epoch 10) and later in one (B fold 4 seed 44, epoch 10 against 7). In the reference run every recovery
of the ten came after the first reduction (first reduced epoch 4 or 5); here they come at 0.001. The ten alone would meet
the condition of the first case of the rule (all ten recover, none of the 36 degenerate); the rule gives that reading
only when no recovery is lost outside the ten, which is not the case here. The recovery criterion holds at
epoch 12 itself in 8 of the ten (not in A\* fold 3 seed 42 and B' fold 4 seed 44); two of the ten keep a dev accuracy
below 0.9 at the checkpoint (A\* fold 3 seed 44 0.8750, B' fold 4 seed 44 0.8900; in the reference run A\* fold 3 seed
44 0.8900 and A\* fold 4 seed 44 0.8650).

Degenerate among the 36: none (same criterion: restored epoch 1 or 2 with dev accuracy 0.5).

Recovery of the reference run lost outside the ten: B' fold 2 seed 42. Its shared window is 8 epochs (first reduced
epoch 9): dev accuracy 0.5 at epochs 1-5, 7 and 8 and 0.6225 at epoch 6 in both runs. In the reference run, at 0.0005
from epoch 9, dev accuracy 0.65, 0.77, 0.8775 and 0.4625 at epochs 9-12 with val_loss 1.625, 4.431, 0.875 and 12.378:
the criterion (dev accuracy >= 0.75 and val_loss < 1.0) holds at epoch 11 only, the restored epoch (dev accuracy
0.8775, dev AUROC 0.9830). In the control run, at 0.001, dev accuracy 0.77, 0.855, 0.7875 and 0.8625 at epochs 9-12,
with val_loss 5.437, 1.685, 13.297 and 9.064: the accuracy part of the criterion holds from epoch 9 on, the val_loss
part never; restored epoch 10 (best val_loss 1.6847, dev accuracy 0.8550, dev AUROC 0.8678); not degenerate. In this single
training the absence of the reduction coincides with the lost recovery; its recovery in the reference run was itself
marginal (one epoch, followed by 0.4625 at epoch 12). It is the only model in which the control run loses a
recovery; no model gains or loses the degenerate status.

**(c) Secondary outcomes** (reference against control run; medians [min-max]; the table per model is in
`aggregate.md`).

| group | run | first recovery | restored epoch | best val_loss | dev accuracy | dev AUROC | epochs at 0.5 |
|---|---|---|---|---|---|---|---|
| the ten | reference | 9.5 [7-12] | 11 [9-12] | 0.1373 [0.0004-0.5239] | 0.9900 [0.8650-1.0000] | 0.9955 [0.9895-1.0000] | 4 [2-7] |
| the ten | control | 6 [4-10] | 9.5 [6-12] | 0.0142 [0.0001-0.3141] | 0.9975 [0.8750-1.0000] | 1.0000 [0.9831-1.0000] | 4 [2-5] |
| the other 26 | reference | 6 [1-11] (26 of 26) | 10 [2-12] | 0.0153 [0.0001-0.8750] | 0.9975 [0.7975-1.0000] | 1.0000 [0.9830-1.0000] | 3 [1-7] |
| the other 26 | control | 6 [1-9] (25 of 26) | 9.5 [3-12] | 0.0154 [0.0001-1.6847] | 0.9963 [0.8550-1.0000] | 1.0000 [0.8678-1.0000] | 3.5 [1-7] |

- Models with a recovery: 36 of 36 in the reference run, 35 of 36 here. The recovery criterion holds at epoch 12
  itself in 26 models (28 in the reference run): not in B fold 2 seed 44, B fold 4 seed 43, A\* fold 1 seed 43, A\*
  fold 3 seed 42, A\* fold 4 seed 43, B' fold 2 seed 42, B' fold 3 seed 43, B' fold 3 seed 44, B' fold 4 seed 42 and
  B' fold 4 seed 44. The dev curves after the recovery are not monotone in either run.
- The restored epoch changes in 23 of the 36 models (8 of the ten, 15 of the other 26). Only 5 checkpoints are the
  same weights in the two runs (restored epoch inside the shared window: A\* fold 1 seed 42, A\* fold 1 seed 43, A\*
  fold 4 seed 43, B' fold 3 seed 43, B' fold 4 seed 43; same best val_loss, dev accuracy and dev AUROC); the other 31
  are different trainings from the first epoch after the window, 8 of them with the same restored epoch. At the
  checkpoint the best val_loss is lower here in 19 models and higher in 12; the dev accuracy is higher in 14, lower in
  8, equal in 14; below 0.9 in 4 models in each run (here A\* fold 3 seed 44 0.8750, A\* fold 4 seed 42 0.8975, B' fold
  2 seed 42 0.8550, B' fold 4 seed 44 0.8900; in the reference run A\* fold 3 seed 44, A\* fold 4 seed 42 0.7975, A\*
  fold 4 seed 44, B' fold 2 seed 42). The epochs at 0.5 change in 12 models, in both directions (for example B fold 1
  seed 42 5 -> 3, B' fold 2 seed 44 7 -> 5, B' fold 3 seed 42 3 -> 5).
- The last recorded rate is 0.001 in every model here; in the reference run 0.0005 to 6.25e-05 (0.001 in A\* fold 1
  seed 42 only).
- Reported apart, as fixed. B fold 3 seed 44 (shared window 5): recovers at epoch 7 (10 in the reference run,
  with the rate halved from epochs 6, 8 and 10) and restores epoch 10 (11 there); dev accuracy at the checkpoint 0.9150
  (1.0000 there), dev AUROC 1.0000, best val_loss 0.1457 (0.0119 there); after the recovery its dev accuracy is 0.5025
  at epoch 8 and 0.88-0.915 at epochs 9-12; recovered at epoch 12. B' fold 4 seed 42 (shared window 4; the relapse):
  in the reference run it restored epoch 2 (dev accuracy 0.9175) and stayed at 0.50 at epochs 5-7 and 0.33-0.34 at
  epochs 8-9 while the rate was halved; here, at 0.001, dev accuracy 0.6725 at epoch 5, 0.5 at epochs 6-8, 0.5325 and
  0.57 at epochs 9-10, 0.97 at epoch 11 (val_loss 0.097, the restored epoch, dev AUROC 0.9940) and 0.5325 at epoch 12
  (val_loss 28.246). Not degenerate in either run; the relapse happens with and without the reduction, and the
  checkpoint moves from epoch 2 to epoch 11.

**(d) Test, reported separately and not used to choose or judge the recipe.** Read only from the public
`aggregate.json` of the two runs. The two primary verdicts (A\* - B, stratum 03) stay `inconclusive`: AUROC +0.1286
[-0.0576, 0.3794], +0.5694 [0.3222, 0.7760], +0.1945 [-0.0458, 0.4088] for seeds 42 / 43 / 44 (reference run +0.2186
[0.0643, 0.4011], +0.3971 [0.2309, 0.5873], +0.1689 [-0.0214, 0.3888]); ACER at the dev threshold -0.0343 [-0.1970,
0.1349], -0.2425 [-0.3920, -0.0661], -0.2293 [-0.3733, -0.0629] (reference run -0.2007 [-0.3589, 0.0024], -0.2772
[-0.3922, -0.1407], -0.0811 [-0.2545, 0.0999]). Of the 84 verdicts (3 comparisons x 4 strata x 7 metrics), 12 differ
from the reference run: A\* - B BPCER and ACER at the threshold 0.5 in strata 03 and all, consistent there (A\* lower),
become inconclusive; A\* - B' BPCER and ACER at the threshold 0.5 and at the dev threshold in strata 03 and all,
inconclusive there, become consistent (A\* lower). The other 72 are the same, A\* - B' AUROC in strata 03 and all
consistent (A\* higher) in both. These differences are not attributable to the ten alone: 31 of the 36 checkpoints are
different weights (23 with another restored epoch), in every arm. Nothing here is a verdict on the control or on the
session-03 question.

**(e) Times.** Wall time 15,217.0 s (about 254 of the 360 minutes), within the cap of 21,600 s; no write after the
cap; not stopped. The estimate was 17,940.9 s (the wall time of the reference run): -2,723.9 s (-15 %). Training
14,328.3 s (every model 12 epochs); seconds per epoch, weighted: B 36.5153, A\* 37.0486, B' 25.9382, against 42.6097,
40.0181 and 34.4986 in the reference run; the run was launched at night as the only heavy process, but the Mac was not
dedicated to it (other processes active, not measured), so this difference is not attributed to the change. Peak
resident memory 5,078 MB.

**Limits.** The criteria (degenerate, recovery) are those of the previous packages, defined after the patience-4 run
and declared as such; the reading rule was fixed before this run. A constant rate tests the absence of the reduction,
not the level of the rate (0.001 is never varied), and the reduction is removed in all 36 models, including the 16
whose first recovery in the reference run already came at 0.001. With a different rate path the checkpoint chosen by
the minimum dev val_loss changes (23 of 36), so dev and test values at the checkpoint differ for reasons other than the
recovery; the dev is small (2-3 subjects, 400 frames per fold) and chooses the checkpoint among 12 epochs. One dataset
(NUAA), one attack type, three seeds per fold; the folds are not independent. The recovery criterion is a threshold on
one dev epoch: B' fold 2 seed 42 fails it by its val_loss (>= 1.6 at every epoch from 9 on) while its dev accuracy is
0.77-0.8625 at epochs 9-12, and in the reference run it met it at one epoch only.

**What is not concluded.** An overall verdict: the rule gives none for a mixed outcome. That the reduction is
unnecessary in general: in B' fold 2 seed 42 its absence coincides with the lost recovery, in a single training, with
a marginal recovery in the reference run. That the reduction is necessary: the ten recover without it, earlier in
8 of 10. That every model is stable: the recovery holds at epoch 12 in 26 of 36, the relapse of B' fold 4 seed 42
appears with and without the reduction, and the checkpoint of 23 models moves. The cause of the early dev saturation
(BatchNormalization statistics, dropout, augmentation, level of the rate, small dev, initialisation; hypotheses of the
earlier notes) is not tested. Whether a lower constant rate, or the reduction alone without the extra epochs, would
also work. Anything about the session-03 question from the test section.

**Possible next steps** (possibilities only, none started, no change proposed here; each would need the
coordinator's agreement): more seeds per fold, so that a single training such as B' fold 2 seed 42 weighs less; a rule
for the trainings whose recovery does not hold at the last epoch, fixed before any further run; a check of the other
candidate causes of the saturation (BatchNormalization or the level of the rate), as already stated in the earlier
notes.

**Pre-commitment.** Configuration, reading rule, criteria and declarations were fixed before the run and not changed;
the rule was applied as written, including the "no overall verdict" of the mixed case of the configuration of the
comparison; every outcome is reported, the
unfavourable ones included; nothing was excluded or rerun; no test score was read by the comparison.
