# Conclusion: `early_stopping_patience` 12, runs `20261109-SESSABLRUN-6a50265` and `20261109-PATVER-4f2b925`

**What was done.** On 7 November 2026 the coordinator agreed on the C2 card to the one change proposed in
`delivery/degenerate/NOTE.md` (PR #22): `early_stopping_patience` from 4 to 12, at most 12 epochs, the same 36 models
(arms B, A\* and B' x 4 folds x seeds 42/43/44), every other setting unchanged, the checkpoint of minimum dev val_loss
and the overall cap of 360 minutes kept; he asked for two precisions in the note before the run (primary outcome and
conclusion if a model does not recover), the check of the shared histories, and the test reported separately and not
used to choose the recipe. The note was amended (`NOTE.md`, [amended 9 Nov]); the new configuration (`ca039df5d311`)
differs from the parent (`4c2d29d13ded`) only in that field and in the `amendment` block, and the runner refuses it,
before reading any data, unless both SHA-256 are the pinned ones and every other field is equal. The training run
`20261109-SESSABLRUN-6a50265` (clean commit `6a50265`) ended PASS, 36 of 36 models complete. A separate script, which
trains nothing and reads no test score, compared its training cards and dev scores with those of the parent run
`20261106-SESSABLRUN-d999458` in run `20261109-PATVER-4f2b925` (PASS), by the reading rule (a)-(e) fixed before the run.
Numbers from its `aggregate.md` and `comparison.csv` (`comparison/` in the package) and from `curves.csv` for single
epochs.

**(a) Shared histories.** For every model the first `epochs_run` epochs of the parent run (5-6 for the ten, 5-11 for
the 9 non-degenerate models stopped before epoch 12, 12 for the 17 complete ones) are identical in the verification
run: 36 of 36, largest absolute difference 0.0 in `loss`, `accuracy`, `val_loss`, `val_accuracy` and `lr`. The
comparison is valid. As stated in the protocol, with identical first epochs no new degenerate model can appear.

**(b) Primary outcome: positive.** The ten original degenerate trainings all recover within 12 epochs (first epoch
with dev accuracy >= 0.75 and val_loss < 1.0) and none of the 36 models is degenerate.

| model | first recovery | restored epoch | dev accuracy at the checkpoint | dev AUROC at the checkpoint |
|---|---|---|---|---|
| A\* fold 2 seed 43 | 7 | 10 | 1.0000 | 1.0000 |
| A\* fold 2 seed 44 | 10 | 12 | 0.9975 | 0.9950 |
| A\* fold 3 seed 42 | 9 | 9 | 0.9425 | 0.9900 |
| A\* fold 3 seed 44 | 11 | 11 | 0.8900 | 0.9895 |
| A\* fold 4 seed 44 | 10 | 12 | 0.8650 | 0.9999 |
| B fold 1 seed 42 | 8 | 10 | 0.9950 | 0.9960 |
| B fold 4 seed 44 | 7 | 11 | 0.9925 | 1.0000 |
| B' fold 1 seed 44 | 7 | 9 | 0.9950 | 0.9950 |
| B' fold 2 seed 44 | 12 | 12 | 0.9875 | 0.9950 |
| B' fold 4 seed 44 | 10 | 11 | 0.9850 | 0.9988 |

The recovery does not separate the stop from the lr halved by ReduceLROnPlateau (declared confounder), and a positive
outcome does not by itself show that every model is stable.

Models without recovery, reported separately as agreed: none among the ten, none outside the ten. B fold 3 seed 44,
the only non-degenerate model without recovery in the parent run (restored epoch 3, dev accuracy 0.2375), recovers at
epoch 10 and restores epoch 11 (dev accuracy 1.0000, best val_loss 0.0119); in the parent run it had stopped at epoch
7.

**(c) Secondary outcomes** (medians [min-max]; the table per model is in `aggregate.md`).

| group | run | first recovery | restored epoch | best val_loss | dev accuracy | dev AUROC | epochs at 0.5 |
|---|---|---|---|---|---|---|---|
| the ten | parent | - (none of 10) | 1 [1-2] | 14.7178 [2.3605-60.9868] | 0.5000 [0.5000-0.5000] | 0.5000 [0.2114-0.9975] | 4 [2-6] |
| the ten | new | 9.5 [7-12] | 11 [9-12] | 0.1373 [0.0004-0.5239] | 0.9900 [0.8650-1.0000] | 0.9955 [0.9895-1.0000] | 4 [2-7] |
| the other 26 | parent | 6 [1-11] (25 of 26) | 9 [1-12] | 0.0220 [0.0001-3.2731] | 0.9963 [0.2375-1.0000] | 1.0000 [0.0950-1.0000] | 3 [1-7] |
| the other 26 | new | 6 [1-11] (26 of 26) | 10 [2-12] | 0.0153 [0.0001-0.8750] | 0.9975 [0.7975-1.0000] | 1.0000 [0.9830-1.0000] | 3 [1-7] |

Models with a recovery: 25 of 36 in the parent run, 36 of 36 here. The lr at the last epoch of the ten is 0.00025 or
0.000125 (1e-3 halved two or three times), against 0.0005 at their last epoch in the parent run.

- Late recoveries: five of the ten recover at epoch 10 or later (A\* fold 2 seed 44 and fold 4 seed 44 at 10, A\*
  fold 3 seed 44 at 11, B' fold 4 seed 44 at 10, B' fold 2 seed 44 at 12, the last epoch, which is also its restored
  epoch). Two of the ten keep a dev accuracy below 0.9 at the checkpoint (A\* fold 3 seed 44 0.8900, fold 4 seed 44
  0.8650). Saturation in the first epochs remains: the ten have 2-7 epochs at 0.5 (2-6 in the parent run, where their
  histories ended at epoch 5-6).
- The 9 non-degenerate models stopped before epoch 12 in the parent run continue to epoch 12, and in five of them the
  checkpoint changes: B fold 1 seed 44 (restored epoch 6 -> 11), B fold 3 seed 44 (3 -> 11), B fold 4 seed 43 (6 ->
  12), B' fold 1 seed 42 (2 -> 12), B' fold 3 seed 42 (1 -> 11); in the other four (B fold 2 seed 43, B' fold 1 seed
  43, fold 4 seed 42, fold 4 seed 43) the restored epoch is the same. The 17 models that already ran 12 epochs have
  identical histories and the same restored epoch.
- Observation, not a criterion: B' fold 4 seed 42 still restores epoch 2 (dev accuracy 0.9175, val_loss 0.1375). From
  epoch 5 to 7 its dev accuracy falls to 0.50 (epochs 5-6 already in the parent run, which stopped at epoch 6), then
  0.33-0.34 at epochs 8-9, and rises to 0.785 at epoch 12, with val_loss between 3.2 and 21.0 from epoch 5 on, while
  the lr is halved at epochs 5, 7, 9 and 11 (down to 6.25e-05). It is not degenerate by the criterion. More generally,
  the recovery criterion holds at epoch 12 itself in 28 of the 36 models (`curves.csv`): the checkpoint is the epoch of
  minimum dev val_loss, not the last one, and the dev curves after the recovery are not monotone.

**(d) Test, reported separately and not used to choose or judge the recipe.** Read only from the public
`aggregate.json` of the two runs. The two primary verdicts (A\* - B, stratum 03) stay `inconclusive`: AUROC +0.2186
[0.0643, 0.4011], +0.3971 [0.2309, 0.5873], +0.1689 [-0.0214, 0.3888] for seeds 42 / 43 / 44 (same sign, the interval
of seed 44 includes 0); ACER at the dev threshold -0.2007 [-0.3589, 0.0024], -0.2772 [-0.3922, -0.1407], -0.0811
[-0.2545, 0.0999] (the intervals of seeds 42 and 44 include 0). Of the 84 verdicts (3 comparisons x 4 strata x 7
metrics), 7 differ from the parent run, 6 consistent here against 1 there: A\* - B BPCER and ACER at the threshold 0.5
in strata 03 and all become consistent (A\* lower); A\* - B' AUROC in strata 03 and all becomes consistent (A\*
higher); B' - B APCER at the dev threshold in stratum 03, consistent in the parent run, becomes inconclusive. These
differences are not attributable to the recovery of the ten alone: 15 of the 36 checkpoints changed (the ten and five
of the nine non-degenerate models stopped early; in A\* - B, 10 of the 24 models), and with patience 12 the dev chooses
the checkpoint among 12 epochs. Nothing here is a verdict on the change.

**(e) Times.** Wall time 17,940.9 s (about 299 of the 360 minutes), within the cap of 21,600 s; no write after the
cap; not stopped. The estimate was 13,090 s: +4,850.9 s (+37 %). Seconds per epoch, weighted: B 42.6097, A\* 40.0181,
B' 34.4986, against 32.4537, 31.8532 and 21.2375 in the parent run; the Mac was not dedicated to the run (other
processes active, not measured), so this difference is not attributed to the change. Peak resident memory 5,121 MB.

**Limits.** The criteria (degenerate, recovery) are those of the previous package, defined after the parent run and
declared as such; the reading rule was fixed before this run. With patience 12 the checkpoint is chosen on the dev among
12 epochs instead of 5-12: a more optimistic selection on the dev, whose reuse (early stopping, ReduceLROnPlateau,
checkpoint, EER threshold) was already declared; the dev has 2-3 subjects (400 frames) per fold. One dataset (NUAA),
one attack type, three seeds per fold; the folds are not independent. With patience 12 the recipe is no longer
identical to that of the folds run (patience 4), which stays a reference, not a paired control.

**What is not concluded.** That every model is stable: the outcome concerns the ten and the degenerate criterion, and
the dev curves after the recovery are not monotone (B' fold 4 seed 42 above). That the stop was the cause: the lr is
halved together with the longer training, and the cause of the dev saturation in the first epochs (BatchNormalization
statistics, dropout, augmentation, lr, small dev, initialisation; hypotheses of `NOTE.md`) is not tested by this run.
That the change would recover the same models with other seeds, folds or datasets. Anything about the session-03
question from the test section.

**Possible next steps** (possibilities only, none started, no change proposed here; each would need the
coordinator's agreement): a run that separates the stop from the lr halved by ReduceLROnPlateau, which would require a
second change of the recipe; more seeds per fold, so that a single unstable training weighs less; a rule for unstable
trainings after the recovery fixed before any further run; a check of the other candidate causes of the saturation
(BatchNormalization or lr), as already stated in `NOTE.md`.

**Pre-commitment.** Configuration, reading rule, criteria and declarations were fixed before the run and not changed;
every outcome is reported, the unfavourable ones included; nothing was excluded or rerun; no test score was read by the
comparison.
