# Conclusion on the comparisons: run `20261106-SESSABLRUN-d999458`

**Question.** Does representing session 03 in training change the error on held-out subjects, at fixed fitting
subjects, total count, test, preprocessing and recipe? Diagnostic, not causal; no statistical test.

**Design.** Arms B (sessions 01-02, 1,400 frames), A\* (sessions 01-03 of the same subjects, 1,400) and B' (sessions
01-02 at the per-cell counts of A\*, 934); 4 subject folds x seeds 42/43/44, common dev without session 03, test = every
frame of the held-out subjects. Outcome PASS, 36 of 36 trainings, wall time 10,209 s of the 21,600 s cap.

**Primary verdicts** (A\* - B, stratum 03: 16 subjects, 9,123 frames; difference [paired bootstrap 2.5-97.5]):

| metric | seed 42 | seed 43 | seed 44 | verdict |
|---|---|---|---|---|
| AUROC | +0.2658 [-0.0060, 0.4862] | +0.2064 [-0.0729, 0.4510] | -0.0103 [-0.2949, 0.2683] | inconclusive: the sign differs across the seeds |
| ACER (dev threshold) | -0.0854 [-0.3192, 0.1665] | -0.1703 [-0.3013, -0.0303] | -0.0008 [-0.1787, 0.2265] | inconclusive: an interval does not exclude 0 |

Both primary verdicts are inconclusive under the rule fixed before the run. Arm values in stratum 03: AUROC A\* 0.8444 /
0.6865 / 0.5614 against B 0.5786 / 0.4800 / 0.5717; ACER at the dev threshold A\* 0.2563 / 0.2496 / 0.4985 against B
0.3417 / 0.4199 / 0.4993.

**Secondary comparisons.** Of the 84 verdicts (3 comparisons x 4 strata x 7 metrics), 83 are inconclusive. The one
consistent verdict is B' - B in stratum 03, APCER at the dev threshold: B' higher, +0.2352 [0.0843, 0.3868], +0.0854
[0.0136, 0.1653], +0.2498 [0.0140, 0.5060]. In stratum 03, B' - B has AUROC +0.0333 / -0.0699 / -0.0815 (the sign
differs); A\* - B' has AUROC +0.2325 / +0.2764 / +0.0712, the same sign for the three seeds, with every interval
including 0 (inconclusive). Reference A (folds run, different fitting frames and dev, indicative only): AUROC 0.9377 /
0.9527 / 0.9664 and ACER at the dev threshold 0.0937 / 0.1532 / 0.1178 in stratum 03.

**Degenerate trainings.** Ten of the 36 models restored epoch 1 or 2 with dev accuracy 0.5 at that epoch (criterion
from the training cards only, defined after the run, descriptive): B 2, A\* 5, B' 3; seed 42: 2, seed 43: 1, seed 44:
7. Six of them give one constant test score (AUROC 0.5000 and ACER 0.5000 in every stratum: A\* fold 2 seed 43, fold 3
seed 44, fold 4 seed 44; B' folds 1, 2 and 4 seed 44); the other four are saturated near one class (minimum dev
val_loss 2.36-18.09). In the folds run every model also had dev accuracy 0.5 in its first epochs but recovered within
the 12 epochs; here, with patience 4, 19 trainings stopped early, at epoch 5 or 6 for the ten degenerate ones. The six
models shared with the failed attempt have identical per-epoch dev val_loss values in both attempts (retrained, not reused) (B fold 1
seed 42 degenerate in both). The reading rule uses every model; nothing was excluded or rerun.

**Diagnostic reading (not causal, not a verdict).** Six of the 12 A\* - B pairs (fold, seed) contain a degenerate model,
five of them on the A\* side; seed 44 of A\* has three of its four folds degenerate, and it is the seed with the negative
AUROC difference. In the six pairs without a degenerate model, the stratum-03 difference is positive for AUROC
(+0.2097 to +0.6558) and negative for ACER at the dev threshold (-0.4438 to -0.0518) in every pair; the non-degenerate
A\* models reach a stratum-03 AUROC of 0.8401-0.9909 per fold, all ten non-degenerate B models 0.2230-0.9779 (0.2230-0.7812 within those six pairs). This
selection is post hoc and is not part of the pre-declared rule: it describes where the instability sits, it does not
replace the verdicts. The B' comparisons do not show a consistent effect of the per-cell count on AUROC.

**Limits.** The session-03 bona fide cell of the fitting subjects comes from 1-2 subjects per fold; the dev has no
session 03, so the dev threshold and early stopping are calibrated on sessions 01-02 for every arm; three seeds per
fold, with degenerate trainings concentrated in seed 44; frames of one subject and session are correlated and the
bootstrap rests on 16 subjects (9 in strata 01 and 02); the folds are not independent; one dataset, one attack type
(print). Pairing is nominal: the same seed gives different batch orders and augmentation in different arms.

**What would help** (possibilities, none started): more seeds per fold, so that a degenerate training weighs less in
the pooled values; a rule for degenerate trainings fixed before any further run; the dev with session 03 already
declared as an alternative (dev-mix), for the threshold-based readings; arm A\*\*, which takes the session-03 bona fide
cell from all fitting subjects.

**Previous attempt.** Run `20261106-SESSABLRUN-3b4021f` ended FAIL at model 7 of 36 on a false alarm of the checkpoint
check (clipped crossentropy compared with the Keras val_loss: 0.1936 against 0.2308 for the checkpoint of minimum
val_loss). Its six complete models are kept in that folder and not used; this run started from scratch with the
corrected check (val_loss from the logits, within 1e-3 absolute or 1 % relative: maximum difference 2.9e-6, 36 of 36).

**Pre-commitment.** Arms, comparisons, reading rule, recipe, threshold criterion and counts were fixed in the
configuration before any training and were not changed; every outcome is reported, including the unfavourable ones.
