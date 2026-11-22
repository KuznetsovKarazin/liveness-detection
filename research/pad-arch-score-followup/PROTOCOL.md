# Secondary analyses of the two-detector NUAA comparison

Version 0.1, proposed on 22 November 2026. **Draft for an author-approved, time-stamped registration.** No new NUAA score analysis or mechanistic test run has been executed under this plan. The published research snapshot is db90203ba56607e2443e2a402e743d690313e8eb. The earlier outcomes, including the influential fold, were known when this plan was written. Record deviations and preserve results for every condition.

## 1. Establish the grouping

The 12,614-frame manifest has 16 unique identifiers. Fourteen have both classes. One contributes 472 genuine frames in fold 1, another 468 photographic frames in fold 3. Both occur exclusively in session 3. The original dataset paper reports 15 people. Keep the working grouping labelled “by manifest identifier.” Check the original four client/imposter file lists, the parser and the independent count tables. Confirm the naming convention with the dataset maintainers. Do not match identities from face images. Keep identifiers and frame paths within the authorized environment.

- If the maintainers confirm distinct participants or give a corrected total, document the release and mapping and reconcile all class/subject counts.
- If they confirm aliases, recompute the membership by person and check every fitted split, development partition, held-out person and acquisition session. If fit/development separation changes, rebuild affected checkpoints and thresholds before re-evaluating. If the only alias is the two session-3-only labels, neither entered fitting or development. They may still require a common test fold, affected predictions and corrected cluster intervals. Specify the corrected allocation first. Preserve both the as-run result and the corrected analysis.
- If mapping remains unavailable, report identifier-level inferences and retain this limitation. Any sensitivity merging the two labels remains hypothetical. Do not rename them as participants.

## 2. Freeze the score analysis

Run on existing private scores only, with no training, after recording the plan and code hash. Inputs are the evaluation's restricted directory containing p4/p12 test score CSVs and the two training runs' restricted directories containing per-state development score CSVs. No images or weights are needed in this step. Check the upstream SHA-256 manifests before running. Retain restricted identifiers and score rows locally.

For each of 48 checkpoints and both BN states, use the attack probability, label 1 for attack, and attack if score is strictly greater than the threshold. Pair the same score rows across states and networks by split and frame ID. The program checks the parent checkpoint hash across both states and independently reproduces the original development EER threshold and the public test error rates at both thresholds. Any non-reproduction is a diagnostic failure to resolve before interpreting additional outputs.

1. Report test AUROC (half credit for ties) and interpolated descriptive test EER. Test EER estimates discrimination and does not set a deployment threshold. Preserve all one-class predictions and tied scores. A constant score with both classes present has AUROC and interpolated EER of 0.5.
2. Select two thresholds using the development scores separately for each state and checkpoint. For each target APCER (5%, 10%), choose the largest candidate threshold at which empirical development APCER does not exceed the target. Candidates are all distinct dev scores plus the just-below-minimum all-attack sentinel. Keep tied scores together. Report the chosen threshold, achieved development APCER/BPCER and achieved test APCER/BPCER. A development target does not guarantee the same test APCER. Report any all-attack operating point.
3. Show one held-out DET curve per checkpoint and BN state, with APCER horizontal and BPCER vertical. To permit aggregate-only export, the public plot uses minimum attainable BPCER on a common grid of APCER ceilings. Report the grid and the unweighted mean of the twelve curves. Interpolation of operating thresholds between the two cohorts is not used.
4. Report the two within-model changes, the change in the architecture gap, all fold–seed points, and whether AUROC changes at the same time. AUC stability is consistent with a shift of the operating point, but equal AUC does not demonstrate identical rankings or calibrated probabilities. If interpreting calibration explicitly, add a separately specified reliability assessment.

## 3. Magnitude and uncertainty

Retain the original 1,000-draw pooled percentile interval and its 926 valid ACER draws. The new analyses are sensitivity estimates. Use the same draws for both networks, BN states, classes and seeds. Sum class-specific frame errors and denominators within fold, recompute each unit's ACER, then average with equal fold–seed weight. Keep undefined draws as counts in the report and do not substitute zero.

- Run 10,000 attempted pooled identifier draws, with PCG64 seed 20261122, and a second set of 10,000 draws sampling four identifiers with replacement within each fold. The stratified variant conditions on the observed fold allocation and has a different distribution. Report attempted, valid and invalid draws for each.
- Compute pooled percentile and BCa intervals for the primary paired contrast and each within-model change. For BCa use the bias correction with half weight on ties and acceleration from deleting one identifier. Report an unavailable BCa result if required jackknifes or adjusted quantiles fail. Report the lack of coverage guarantees with 16 clusters. Expanding the number of draws reduces numerical error only.
- Report all-identifier-deletion point ranges and the pooled interval after omitting fold 3, seed 44 as an explicitly post hoc influence diagnostic. Do not test the 11/12 dependent signs as independent Bernoulli trials. Do not interpret overlapping within-model intervals as an estimate for the cross-model contrast. No permutation test is proposed without a defensible exchangeability assumption.
- Use the script on existing scores. Publication output contains only checkpoint-level metrics, a coarse DET grid and summary intervals. The initial group reconciliation may require revising the input grouping and publishing a separately labeled sensitivity run.

## 4. Mechanistic re-estimation with frozen weights

Register the following additional analysis before producing new held-out predictions. Use the existing 48 checkpoints, four folds, three seeds and both patience settings. The primary mechanistic subset is patience 4, all twelve fold–seed units for both networks. Patience 12 assesses the same effects descriptively. The only scoring threshold for primary mechanistic contrasts is 0.5. No gradient step or new optimizer state is used.

**Controls.** Preserve the original stored state and reproduce the existing clean/dropout-off re-estimation on development, with identical threshold decisions and maximum absolute score difference at most 1e-5. Check that trainable-weight hashes remain identical to the parent and that the comparator still contains no dropout. Distinguish dropout from the separate augmentation module's GaussianNoise.

| Input used to estimate BN moments | Dropout during estimation | Role |
|---|---|---|
| Unaugmented fitting frames | Off | Existing reference |
| Augmented fitting frames | Off | Isolate the change of fitting input distribution |
| Unaugmented fitting frames | On for upstream dropout layers only | Isolate the dropout contribution at clean inputs |
| Augmented fitting frames | On for upstream dropout layers only | Estimate the interaction |

Use exactly the original augmentation operations and ranges, the unaugmented fitting frame list as the base, the same channel conventions, and the original dropout rates. Use eight deterministic views per fitting frame per stochastic condition. Fix seeds 101–108, deriving stateless augmentation draws by frame and view and dropout masks by frame, view and layer. Reuse augmentation draws across comparable cells and dropout masks across comparable cells. A clean/dropout-off cell has identical repeated views and may be computed once. Combine counts, sums and squared deviations in float64 across views. Preserve sample and spatial weighting, and write the same population variance convention as the original estimator. At every layer, earlier BNs use the assigned frozen moments. Enable only explicit upstream dropout for the corresponding cell. Calling the whole model with training=True would also change BN behavior and invalidate the comparison. During final development/test scoring all dropout and augmentation are off. For dropout-free ResNet-18, duplicates with identical augmentation must match within the reference tolerance and may be reused.

Define E00, E10, E01 and E11 for clean/off, augmented/off, clean/on and augmented/on. Report augmentation E10−E00, dropout E01−E00 and interaction E11−E10−E01+E00 within each architecture for ACER, APCER and BPCER. Report within-unit values and descriptive same-draw intervals. Report the same differences of the architecture gaps. All mechanistic comparisons are exploratory. Compare the absolute percentage-point changes with stochastic-view Monte Carlo variability using pre-fixed views 1–4 versus 5–8 on development. Keep any Monte Carlo sensitivity independent of test selection.

**EMA and the restored epoch.** Record the selected epoch, 88 updates per epoch, the initialization coefficient 0.99^(88e) and pre/post BN moment discrepancies by layer. Relate the restored epoch to the change as a descriptive, paired diagnostic. Plot the new cells against the recorded restored epochs. Neither this factorization nor an epoch correlation isolates an EMA-only causal effect: stored moments also depend on the trajectory of the learned weights. An exact replay would require training-state histories or a separate prospective training experiment.

**Order.** Implement and test on artificial arrays and the fitting/development partition. Freeze source, numerical tolerances, seeds, estimator, estimands, thresholds, aggregation, group policy and exclusion rules before applying the new states to held-out data. Archive all outcomes, including ties, null and contradictory results. No target test frames may inform fitting statistics in this inductive analysis.

## 5. Other architectures and new data

Keep the AttackNet comparison as the architecture-extension task. List the actual implementations and their BN/dropout layouts (AttackNet V1, V2.1 and V2.2), confirm the forward pass and parameter counts, and freeze the run matrix before reading their NUAA scores. Keep one comparison protocol across architectures and distinguish any model-specific tuning as a separate study. Extra architectures on the familiar NUAA data expand architecture coverage.

Independent confirmation needs a separate authorized dataset and a sequestered split not previously examined for this hypothesis. Document any previous project use, identities, devices, attack species, metric and primary analysis before test access. Use an untouched, independently grouped test split. If undertaking target-domain AdaBN, reserve unlabeled target calibration subjects separately from those used for evaluation and report this separately from the inductive experiment.
