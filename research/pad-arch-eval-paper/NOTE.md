# Note for the first draft: LivenessNet and ResNet-18 on the NUAA test, patience 4, threshold 0.5

Source: the public files of the evaluation `20261121-ARCHEVAL-332fbdf` (PR #32), through `outputs/` of this folder.
Values in percentage points (proportions of the source files × 100, two decimals). Gap = ACER(ResNet-18) −
ACER(LivenessNet), mean over the 12 fold-seed units (4 folds × 3 seeds); contrast = gap with re-estimated BN − gap with
ordinary BN.

**The 16 subjects and the intervals.** The test holds 16 subjects, all from NUAA, pooled over the 4 folds. Each subject
interval is the 2.5–97.5 percentile interval of 1,000 resamples of these 16 subjects, with the same resamples for both
networks and both BN states (README of PR #32); 926 resamples are valid for ACER (950 for APCER, 940 for BPCER): a
resample is dropped when a unit's metric is undefined, that is when, in some fold, no subject is drawn or the subjects
drawn have no frame of one of the two classes. The intervals are individual 95% intervals, with no correction for
multiplicity. There are 3 seeds (42, 43, 44), paired only nominally: the same seed does not give the two networks the
same initial weights. The intervals are copied from `results.json`; they cannot be recomputed from the public files,
which hold no per-subject counts.

**Reuse of the LivenessNet cells.** Both LivenessNet cells of the primary analysis (ordinary and re-estimated BN) had
already been read on this test with the accepted checkpoints (PR #21, PR #28). The new LivenessNet trainings repeat the
accepted ones with the same recipe, data and seeds and follow the same training history (`history_check` in PR #31),
which does not prove identical weights. Only the ResNet-18 side is new; on NUAA the comparison is therefore not an
independent validation (`PROTOCOL.md` §5). Nothing was excluded after the test was seen: the two LivenessNet models
that, with ordinary BN, decide every test frame as attack at threshold 0.5 (ACER 1/2) enter the means like the others.

**After re-estimation the interval of the gap contains zero.** With ordinary BN the gap is −16.06 [−20.79, −7.41]
points (ResNet-18 lower in 11 of the 12 units); with re-estimated BN it is −6.07 [−12.53, +4.76] points (ResNet-18
lower in 9 of 12). In points of ACER, the mean gap goes from −16.06 with ordinary BN to −6.07 with re-estimated BN
(point estimates), and the contrast is +9.98 [+2.56, +16.32] points. With re-estimated BN the pre-registered rule reads
the gap as "unresolved": which network has the lower ACER in that state is not concluded. By the pre-registered
definition there is no change of sign between the two BN states, since a change of sign needs two resolved conditions
of opposite sign; the gap of 2 of the 12 units changes sign, which the same definition reports apart. The contrast
shows that, on these checkpoints and test frames, the signed ACER difference is algebraically higher with this
re-estimation; it identifies no cause. Per seed (appendix, reported apart), the contrast interval of each seed
contains zero. The results apply only to these two networks, this recipe and these data.

**Figure 1 (suggested caption).** Figure 1. ACER gap between ResNet-18 and LivenessNet on the NUAA test (16 subjects
pooled over 4 folds; patience 4, threshold 0.5) with ordinary BN and with BN re-estimated on the fitting set, and their
contrast (re-estimated − ordinary), in percentage points. Black: mean over the 12 fold-seed units and subject-bootstrap
95% interval (1,000 resamples, 926 valid; individual intervals). Grey: the 12 unit values, stacked vertically where
they are closer than 2 points. A negative gap means a lower ACER for ResNet-18.

## Addition of the author (not requested, not pre-registered)

Computed from `units.csv` alone: the 12 per-unit ACER contrasts range from −7.64 to +54.09 points; the largest is that
of fold 3, seed 44, whose gap goes from −54.32 to −0.22 points; the mean of the other 11 contrasts is +5.97 points. No
interval is available for these values from the public files, and this description was not part of the
pre-registration.
