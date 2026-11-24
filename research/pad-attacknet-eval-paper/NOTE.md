# Note for the first draft: the five networks on the NUAA test under BN re-estimation (C5), patience 4, threshold 0.5

Source: the public files of the evaluation `20261124-ANETEVAL-7c22b1a`, through `outputs/` of this folder. Values in
percentage points (proportions × 100, two decimals). Variation of a network = ACER with BN re-estimated on the fitting
set − ACER with ordinary BN, mean over the 12 fold-seed units (4 folds × 3 seeds); contrast of an AttackNet = its
variation − the variation of LivenessNet, per unit, on the same units and subjects.

**The 16 subjects and the intervals.** The test holds 16 NUAA subject identifiers, pooled over the 4 folds. Each
interval is the 2.5–97.5 percentile interval of 1,000 resamples of these 16 subjects, with the same resamples for every
network and both BN states; 926 resamples are valid for ACER (950 for APCER, 940 for BPCER). The intervals are
individual 95% intervals: no correction for multiplicity, no simultaneous coverage over the three contrasts, the five
variations or the comparisons. Whether the 16 identifiers correspond to 16 or to 15 participants is being clarified
with the curators of NUAA (`PROTOCOL.md` §10). There are 3 seeds (42, 43, 44), paired only nominally: the same seed does
not give the networks the same initial weights. The intervals are copied from `results.json`; they cannot be
recomputed from the public files, which hold no per-subject counts.

**Primary analysis.** The ACER variation is −7.07 [−13.33, −0.41] for LivenessNet, +8.07 [+3.47, +11.66] for
AttackNetV1, +6.87 [+3.35, +11.85] for AttackNetV2_1, +4.53 [+1.00, +8.82] for AttackNetV2_2 and +2.91 [−1.73, +6.63]
for ResNet-18; the five variations are reported with their intervals and no reading (`PROTOCOL.md` §6 (a)). The
contrasts are +15.14 [+4.96, +24.05] for AttackNetV1, +13.95 [+7.56, +21.63] for AttackNetV2_1 and +11.61 [+4.46,
+18.95] for AttackNetV2_2. **For each AttackNet the contrast interval lies entirely above zero** (one reading per
AttackNet; the protocol fixes no joint statement on the three, `PROTOCOL.md` §6 (c)). The conclusion written before
the test, for each AttackNet k: "On these checkpoints and test frames, the ACER variation with this BN re-estimation
(re-estimated minus ordinary BN) is algebraically higher for k than for LivenessNet; it does not identify a cause."

**Facts from the public files that bear on these numbers** (a choice of the author, descriptive, not in
`PROTOCOL.md`; means only, since subject intervals need the per-subject counts). The per-unit ACER variation is above
zero in 8, 8 and 7 of the 12 units for AttackNetV1, AttackNetV2_1 and AttackNetV2_2, and below zero in 5 of the 12
units for LivenessNet (median of its 12 values 1.28). For each AttackNet the mean APCER variation is positive (20.17,
23.96, 19.35) and the mean BPCER variation negative (−4.04, −10.21, −10.29). The mean variation of LivenessNet depends
on two units: fold 3 seed 44 (ACER 72.65 with ordinary BN and 18.91 with re-estimated BN, variation −53.73) and fold 1
seed 42 (one of the two one-class models, variation −30.12). Without fold 3 seed 44 the mean variation of LivenessNet
is −2.83 and the three contrasts are 8.50, 9.59 and 5.95 (Appendix C, leaving one unit out). At patience 12 with the
dev EER threshold (descriptive) the variation of LivenessNet is positive (3.19) and the three contrasts are close to
zero (−0.46, −0.33, 0.10), each interval containing zero. These results are kept as they are.

**Per seed** (Appendix A; `PROTOCOL.md` §6 (d): reported apart, outside the reading; 4 units per seed, the same 16
subjects, individual intervals): the 9 per-seed contrast means are all above zero (from 5.04 to 28.43 percentage
points); 4 of their 9 intervals contain zero (seed 43 AttackNetV1, seed 43 AttackNetV2_1, seed 44 AttackNetV2_1,
seed 44 AttackNetV2_2).

**Reuse and limits.** LivenessNet and ResNet-18 are the 48 checkpoints of C2, already read on this test; in each
contrast only the AttackNet side is new (`PROTOCOL.md` §5): not an independent validation, and a post hoc extension
written after the C2 results. Nothing was excluded after the test was seen: the two LivenessNet models that, with
ordinary BN, decide every test frame as attack at threshold 0.5 (ACER 1/2) enter the means like the others; both are
inside the ordinary-BN mean of LivenessNet, and so inside its variation and every contrast; the same two models
decide every dev frame as attack at threshold 0.5 (Appendix B); and no model is not evaluable. Patience 12, the EER
threshold, the comparison with ResNet-18, the rankings and leaving one unit out are descriptive analyses (Appendix C).
The results apply only to these networks, this recipe and these data.

**Figure 1 (suggested caption).** Figure 1. ACER variation (BN re-estimated on the fitting set − ordinary BN) of
LivenessNet, AttackNetV1, AttackNetV2_1, AttackNetV2_2 and ResNet-18 on the NUAA test, and contrast of each AttackNet
with LivenessNet (its variation − that of LivenessNet), patience 4, threshold 0.5, in percentage points. Black: mean
over the 12 fold-seed units and subject-bootstrap 95% interval (16 subjects pooled over 4 folds, 1,000 resamples, 926
valid; individual intervals, no correction for multiplicity). Grey: the 12 unit values, stacked vertically where they
are closer than 2 points. A positive variation means a higher ACER after re-estimation.
