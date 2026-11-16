# Report: BatchNormalization recalibration of the 36 patience-4 checkpoints, run `20261114-BNRECAL-a10a082`

Numbers in this report are read from the public outputs of the run in this folder: `recalibration/aggregate.md`,
`recalibration/results.csv`, `recalibration/baseline.csv`, `recalibration/run_record.json`, `run/status.json` and
`run/run.log.txt` (cell values, counts of rows, smallest and largest values of a column). No mean, median or
percentage is computed here. Values are written with four decimals as in `aggregate.md`; values below 0.0001 are
written in scientific notation with four significant digits; the worsening sizes are copied as `aggregate.md` prints
them. Models are named as in `NOTE.md` (B, A\*, B'); the files use `arm_b`, `arm_a_star`, `arm_b_prime`.

## What was done

On 13 November 2026 the coordinator approved, on the C2 card, the control proposed in `NOTE.md` (commit `dd7a893`) on
the 36 existing checkpoints of the patience-4 run `20261106-SESSABLRUN-d999458`, without new training and without
reading the test, with three points to specify in this pull request before the run: the estimator (global per-channel
moments over the whole fitting set, layer by layer, dropout and augmentation off, batch sizes, last batch and spatial
dimensions counted, only `moving_mean` and `moving_variance` changed, on copies); the baseline tolerances fixed in
advance, with the rule to stop and document a discrepancy without interpreting the recalibration; conclusions limited
in both directions, the 2×2 table included. The revised note (sections 2 and 4 of `NOTE.md`), the runner, its
configuration, its tests and its mutation check were committed as `65b65ed` (13:18:25 UTC) and pushed to this pull
request before the run (GitHub recorded the pull request as updated at 13:18:31 UTC with head `65b65ed`). The run
started at 14:18:46 CET (13:18:46 UTC; first line of `run/run.log.txt`) from the clean commit `a10a082` of the working
repository (private; `dirty: false` in `run/status.json`) and finished at 14:47:54 CET (`finished` in
`run/status.json`) with status **PASS**, 36 of 36 models completed, wall time 1748.6 s (29.1 minutes). The run records
configuration SHA-256 `d97e2e7497fb038708f09613bab6b8602ddac53976a31d84f8771c87c3d52865` and note SHA-256
`55ddd83901503286c31e7c5d352b9fcd3b62cc69f12872a0700663fc7f644af7`, equal to those of
`livenesslab/delivery/bnrecal/nuaa_bn_recalibration.json` and `NOTE.md` in this folder; the SHA-256 of the six code
files and of the architecture recorded in `run_record.json` (`code_sha256`; key `architecture`) are equal to those of
the files in `livenesslab/scripts/` and of `src/architectures.py` of this repository. Device: CPU (GPU hidden; the
package tensorflow-metal 1.1.0 is installed), TensorFlow 2.15.0, Keras 2.15.0, NumPy 1.26.4, Python 3.11.16
(`run_record.json`). The field `checks.environment.tensorflow_state` ("not imported") is recorded at the end of phase
0, before TensorFlow is imported for phases 1 and 2.

No training was run; no trained parameter was changed (kernels, biases, gamma and beta compared byte for byte, see
below); the recalibration was made on in-memory copies and no `.keras` file was written. What was read of the test,
as recorded in the field `test_read` of `run_record.json`: "the fold schema (fold<k>_test.csv, the test role of our
folds) is never opened and no <model>_test_scores.csv is opened; the M2 manifests official_test and cnn_test (NUAA's
official image lists, not the test role of our folds) are read only to index the cache; as in the parent run the frame
files (cache and extracted frames) are hashed and loaded whole, but only fitting and dev frames are assembled; the
public aggregate of the parent run, which holds test metrics, is read only as bytes for its SHA-256 and never parsed;
no public file of the folds run is opened; no test frame is assembled or evaluated and no test score or metric is
read, computed or written".

## Baseline: 36 of 36 within every tolerance

Before any recalibration, all 36 checkpoints were evaluated on the dev with their original statistics (CPU, batch 16,
frames / 255, inference mode), after checking each file against its `weights_sha256`. Every check passed for every
model (`baseline_pass` = 1 in the 36 rows of `baseline.csv`; `baseline_failed` empty in `status.json`), so the stop
rule did not apply. Largest deviation observed against each tolerance (`baseline.csv`):

| check | column(s) | largest value over the 36 models | model | tolerance |
|---|---|---|---|---|
| attack probability, parent's scorer vs stored dev scores | `prob_parent_max_abs_diff` | 0.0 (all 36) | - | ≤ 1e-5 |
| attack probability, this evaluator vs stored dev scores | `prob_eval_max_abs_diff` | 0.0 (all 36) | - | ≤ 1e-5 |
| decisions at 0.5 changed (both scorers) | `prob_parent_discordant_05`, `prob_eval_discordant_05` | 0 (all 36) | - | 0 |
| val_loss from the logits vs recorded best val_loss | `val_loss_diff` | 2.876e-06 | A\* fold 2 seed 44 | ≤ 1e-3 absolute |
| val_loss vs the parent's own recomputation (reported, not binding) | `val_loss_diff_recomputed` | 0.0 (all 36) | - | - |
| dev accuracy (argmax) vs recorded | `accuracy_diff`; `correct` vs `correct_card` | 0.0 (all 36); equal in 36 of 36 | - | ≤ 1e-6, same correct frames |
| softmax of the float64 logits vs float32 probabilities | `softmax_gap` | 3.329e-06 | B' fold 1 seed 42 | ≤ 1e-4 |

The fitting set has no recorded reference in inference mode (`fitting_reference`: "none (no recorded value in inference
mode)"); its accuracy and loss before are those of `results.csv`.

## Checks of the implementation in the run

These checks are binding: any failure stops the run as FAIL, with every output removed (`README.md`, "The control
runner"). The run is PASS with 36 of 36 models completed, so each of them passed for each model:

- tensors: after the five layers every weight tensor of the copy was compared byte for byte with the original; exactly
  the 10 moving tensors (`moving_mean` and `moving_variance` of `bn1_1`, `bn1_2`, `bn2_1`, `bn2_2`, `bn_dense`) differ,
  each equal to the installed float32 statistic, and every other tensor (kernels, biases, gamma, beta) is identical
  (`estimator.copy` in `run_record.json`);
- consistency: one more pass over the fitting set with the final copy recomputed the moments of the five inputs, which
  had to equal the installed statistics within 1e-4 relative plus 1e-6 absolute (`tolerances.consistency_rel`,
  `consistency_abs`). The largest differences per layer are recorded only in the restricted statistics files
  (`max_abs_diff_mean`, `max_abs_diff_variance`); the public outputs do not report them;
- softmax of the logits within 1e-4 after the recalibration, on the dev and on the fitting set;
- at the end, the SHA-256 of the 36 parent `.keras` files and of the parent restricted `SHA256SUMS` are unchanged
  (`checks.final` in `run_record.json`: `parent_checkpoints_unchanged` 36, `parent_restricted_sha256sums_unchanged`
  true).

## Primary outcome: all ten

The reading rule fixed before the run, applied by the script (`aggregate.md`, "Primary reading"), gives the case "all
ten":

> all ten (10 of 10) meet the recovery criterion on the dev after the re-estimation: on these checkpoints, re-estimating
> the statistics on the fitting set, every trained parameter unchanged, is sufficient to meet the criterion (the moving
> statistics, dropout and the augmentation of the fitting set, hypotheses 1-3, are not separated); the later saturated
> epochs, at residual <= 0.0704, are not examined. A recovery shows the effect of this recalibration on the examined
> checkpoints, not a single cause of the saturation

The ten, before -> after (recovery criterion: dev accuracy ≥ 0.75 and val_loss < 1.0; dev of 400 frames; fitting of
1,400 frames for B and A\*, 934 for B'):

| model | restored epoch | dev accuracy | dev correct (of 400) | dev val_loss | dev AUROC | fitting accuracy | fitting loss |
|---|---|---|---|---|---|---|---|
| B fold 1 seed 42 | 1 | 0.5000 -> 0.9950 | 200 -> 398 | 7.0611 -> 0.0244 | 0.9975 -> 0.9998 | 0.5000 -> 0.9993 | 7.3138 -> 0.0015 |
| A\* fold 2 seed 43 | 1 | 0.5000 -> 1.0000 | 200 -> 400 | 60.9868 -> 0.0031 | 0.5000 -> 1.0000 | 0.5000 -> 0.9886 | 60.2927 -> 0.0371 |
| A\* fold 2 seed 44 | 1 | 0.5000 -> 0.9800 | 200 -> 392 | 18.0934 -> 0.3553 | 0.2114 -> 0.9814 | 0.5000 -> 0.9957 | 15.9745 -> 0.0492 |
| A\* fold 3 seed 42 | 1 | 0.5000 -> 1.0000 | 200 -> 400 | 3.1296 -> 0.0137 | 0.4690 -> 1.0000 | 0.5000 -> 0.9986 | 2.8493 -> 0.0147 |
| A\* fold 3 seed 44 | 1 | 0.5000 -> 0.8275 | 200 -> 331 | 33.4090 -> 0.2594 | 0.5000 -> 1.0000 | 0.5000 -> 0.9971 | 31.6305 -> 0.0172 |
| B fold 4 seed 44 | 2 | 0.5000 -> 1.0000 | 200 -> 400 | 2.3605 -> 0.0013 | 0.9652 -> 1.0000 | 0.5000 -> 0.9971 | 2.2382 -> 0.0143 |
| A\* fold 4 seed 44 | 1 | 0.5000 -> 0.8100 | 200 -> 324 | 22.8274 -> 0.6411 | 0.5000 -> 0.9909 | 0.5000 -> 0.9986 | 22.9596 -> 0.0063 |
| B' fold 1 seed 44 | 1 | 0.5000 -> 0.9950 | 200 -> 398 | 12.1449 -> 0.0146 | 0.4975 -> 0.9998 | 0.5000 -> 0.9989 | 13.0656 -> 0.0037 |
| B' fold 2 seed 44 | 2 | 0.5000 -> 0.9975 | 200 -> 399 | 11.4259 -> 0.0065 | 0.5000 -> 1.0000 | 0.5000 -> 1.0000 | 11.6160 -> 0.0003 |
| B' fold 4 seed 44 | 1 | 0.5000 -> 1.0000 | 200 -> 400 | 17.2907 -> 0.0281 | 0.5000 -> 1.0000 | 0.5000 -> 0.9979 | 16.4776 -> 0.0075 |

The ten do not reach the same values after the re-estimation. Four have dev accuracy 1.0000 after (A\* fold 2 seed 43,
A\* fold 3 seed 42, B fold 4 seed 44, B' fold 4 seed 44); two stay well below the others: A\* fold 3 seed 44 at 0.8275
(331 of 400 correct) and A\* fold 4 seed 44 at 0.8100 (324 of 400). The dev val_loss after ranges from 0.0013 (B fold
4 seed 44) to 0.6411 (A\* fold 4 seed 44); A\* fold 2 seed 44 has 0.3553 and A\* fold 3 seed 44 0.2594. The dev AUROC
after ranges from 0.9814 (A\* fold 2 seed 44) to 1.0000. On the fitting set after, the accuracy ranges from 0.9886 (A\*
fold 2 seed 43) to 1.0000 (B' fold 2 seed 44). For these ten no worsening indicator is set (`results.csv`).

## Controls: 25 of 25 keep the criterion; every worsening listed

`aggregate.md`: "25 of the 25 controls keep the criterion, 0 lose it." Worsening indicators are set in 17 of the 25
controls and in none of the ten or the model apart (`results.csv`, columns `lost_criterion` to `fitting_loss_higher`).
Number of models per indicator (`aggregate.md`, "Worsening"): `lost_criterion` 0, `dev_accuracy_lower` 11,
`val_loss_higher` 15, `dev_auroc_lower` 7, `fitting_accuracy_lower` 3, `fitting_loss_higher` 4 (and `gained_criterion`
11: the ten and the model apart). Any change in the worsening direction counts, however small. Every worsening, as
`aggregate.md` lists it (dev accuracy as correct frames out of 400, with the dev accuracy from the "All models" table):

| model | indicator | before -> after | difference |
|---|---|---|---|
| A\* fold 1 seed 42 | val_loss_higher | 0.0104333 -> 0.0134321 | 0.00299882 |
| B fold 1 seed 43 | dev_accuracy_lower | 399 -> 397 correct frames (dev accuracy 0.9975 -> 0.9925) | -2 |
| B fold 1 seed 43 | val_loss_higher | 0.00558041 -> 0.182965 | 0.177385 |
| B fold 1 seed 43 | dev_auroc_lower | 0.999975 -> 0.995 | -0.004975 |
| A\* fold 1 seed 43 | dev_accuracy_lower | 399 -> 398 correct frames (dev accuracy 0.9975 -> 0.9950) | -1 |
| B fold 1 seed 44 | val_loss_higher | 0.029843 -> 0.344459 | 0.314616 |
| B fold 1 seed 44 | dev_auroc_lower | 0.998475 -> 0.994687 | -0.0037875 |
| B fold 1 seed 44 | fitting_accuracy_lower | 1 -> 0.999286 | -0.000714302 |
| B fold 1 seed 44 | fitting_loss_higher | 0.00126252 -> 0.00144349 | 0.000180971 |
| A\* fold 1 seed 44 | val_loss_higher | 0.0345145 -> 0.532653 | 0.498139 |
| A\* fold 1 seed 44 | dev_auroc_lower | 0.99915 -> 0.98945 | -0.0097 |
| A\* fold 1 seed 44 | fitting_accuracy_lower | 1 -> 0.998571 | -0.00142854 |
| A\* fold 1 seed 44 | fitting_loss_higher | 0.000694501 -> 0.00280055 | 0.00210605 |
| A\* fold 2 seed 42 | val_loss_higher | 0.000107111 -> 0.000317227 | 0.000210116 |
| A\* fold 2 seed 42 | fitting_loss_higher | 3.77205e-05 -> 0.000341481 | 0.000303761 |
| B fold 3 seed 43 | dev_accuracy_lower | 397 -> 387 correct frames (dev accuracy 0.9925 -> 0.9675) | -10 |
| B fold 3 seed 43 | val_loss_higher | 0.0530546 -> 0.0639548 | 0.0109002 |
| A\* fold 3 seed 43 | dev_accuracy_lower | 399 -> 367 correct frames (dev accuracy 0.9975 -> 0.9175) | -32 |
| A\* fold 3 seed 43 | val_loss_higher | 0.00883641 -> 0.166968 | 0.158132 |
| B fold 4 seed 42 | dev_accuracy_lower | 400 -> 399 correct frames (dev accuracy 1.0000 -> 0.9975) | -1 |
| B fold 4 seed 42 | val_loss_higher | 0.00054232 -> 0.162219 | 0.161677 |
| B fold 4 seed 42 | dev_auroc_lower | 1 -> 0.995 | -0.005 |
| A\* fold 4 seed 42 | dev_accuracy_lower | 319 -> 315 correct frames (dev accuracy 0.7975 -> 0.7875) | -4 |
| A\* fold 4 seed 42 | dev_auroc_lower | 0.999 -> 0.991625 | -0.007375 |
| A\* fold 4 seed 42 | fitting_accuracy_lower | 0.998571 -> 0.997143 | -0.0014286 |
| B fold 4 seed 43 | dev_accuracy_lower | 400 -> 399 correct frames (dev accuracy 1.0000 -> 0.9975) | -1 |
| B fold 4 seed 43 | val_loss_higher | 0.00128923 -> 0.00466482 | 0.00337558 |
| A\* fold 4 seed 43 | dev_accuracy_lower | 398 -> 367 correct frames (dev accuracy 0.9950 -> 0.9175) | -31 |
| A\* fold 4 seed 43 | val_loss_higher | 0.0271906 -> 0.151603 | 0.124413 |
| A\* fold 4 seed 43 | dev_auroc_lower | 0.9993 -> 0.99915 | -0.00015 |
| B' fold 2 seed 43 | dev_accuracy_lower | 400 -> 399 correct frames (dev accuracy 1.0000 -> 0.9975) | -1 |
| B' fold 2 seed 43 | val_loss_higher | 0.00423285 -> 0.00931295 | 0.00508009 |
| B' fold 3 seed 42 | val_loss_higher | 0.321121 -> 0.344856 | 0.0237347 |
| B' fold 3 seed 43 | dev_accuracy_lower | 400 -> 399 correct frames (dev accuracy 1.0000 -> 0.9975) | -1 |
| B' fold 3 seed 43 | val_loss_higher | 0.00113761 -> 0.0279358 | 0.0267982 |
| B' fold 3 seed 44 | dev_accuracy_lower | 399 -> 367 correct frames (dev accuracy 0.9975 -> 0.9175) | -32 |
| B' fold 3 seed 44 | val_loss_higher | 0.0340957 -> 0.12177 | 0.0876745 |
| B' fold 4 seed 42 | val_loss_higher | 0.137541 -> 0.361438 | 0.223897 |
| B' fold 4 seed 42 | dev_auroc_lower | 0.99655 -> 0.948475 | -0.048075 |
| B' fold 4 seed 42 | fitting_loss_higher | 0.0259913 -> 0.0603952 | 0.0344039 |

The largest worsenings: three controls lose 31-32 correct dev frames and end at dev accuracy 0.9175 (A\* fold 3 seed 43,
A\* fold 4 seed 43, B' fold 3 seed 44); B fold 3 seed 43 loses 10 (0.9925 -> 0.9675); the largest val_loss increases
are A\* fold 1 seed 44 (0.0345 -> 0.5327) and B fold 1 seed 44 (0.0298 -> 0.3445); the largest AUROC decrease is B'
fold 4 seed 42 (0.9966 -> 0.9485). A\* fold 4 seed 42 keeps the criterion at dev accuracy 0.7875 after (threshold 0.75;
val_loss 0.7175). The 8 controls with no worsening indicator: B fold 2 seed 42, B fold 2 seed 43, B fold 2 seed 44, B
fold 3 seed 42, B' fold 1 seed 42, B' fold 1 seed 43, B' fold 2 seed 42, B' fold 4 seed 43.

Counts over the 25 controls, by comparing the "after" and "before" columns of `results.csv` in each row (higher /
equal / lower after):

| quantity | columns | higher after | equal | lower after |
|---|---|---|---|---|
| correct dev frames | `dev_correct_before`, `dev_correct_after` | 8 | 6 | 11 |
| dev val_loss | `dev_val_loss_before`, `dev_val_loss_after` | 15 | 0 | 10 |
| dev AUROC | `dev_auroc_before`, `dev_auroc_after` | 5 | 13 | 7 |
| fitting accuracy | `fitting_accuracy_before`, `fitting_accuracy_after` | 14 | 8 | 3 |
| fitting loss | `fitting_loss_before`, `fitting_loss_after` | 4 | 0 | 21 |

## All 36 models, before -> after

Every model, dev and fitting before -> after, as `aggregate.md` ("All models") prints them, with the correct dev frames
of `results.csv` (`dev_correct_before`, `dev_correct_after`; dev of 400 frames for every model); in the order of the
run; dev criterion 1 = met; the 2×2 cell as in the table below:

| model | group | restored epoch | dev accuracy | dev correct (of 400) | dev val_loss | dev AUROC | fitting accuracy | fitting loss | dev criterion | 2×2 cell |
|---|---|---|---|---|---|---|---|---|---|---|
| B fold 1 seed 42 | ten | 1 | 0.5000 -> 0.9950 | 200 -> 398 | 7.0611 -> 0.0244 | 0.9975 -> 0.9998 | 0.5000 -> 0.9993 | 7.3138 -> 0.0015 | 0 -> 1 | dev after only |
| A\* fold 1 seed 42 | control | 10 | 0.9950 -> 0.9975 | 398 -> 399 | 0.0104 -> 0.0134 | 1.0000 -> 1.0000 | 0.9986 -> 1.0000 | 0.0232 -> 0.0001 | 1 -> 1 | both |
| B fold 1 seed 43 | control | 11 | 0.9975 -> 0.9925 | 399 -> 397 | 0.0056 -> 0.1830 | 1.0000 -> 0.9950 | 1.0000 -> 1.0000 | 0.0001 -> 0.0000 | 1 -> 1 | both |
| A\* fold 1 seed 43 | control | 9 | 0.9975 -> 0.9950 | 399 -> 398 | 0.0165 -> 0.0128 | 0.9995 -> 1.0000 | 0.9943 -> 1.0000 | 0.0421 -> 0.0005 | 1 -> 1 | both |
| B fold 1 seed 44 | control | 6 | 0.9950 -> 0.9950 | 398 -> 398 | 0.0298 -> 0.3445 | 0.9985 -> 0.9947 | 1.0000 -> 0.9993 | 0.0013 -> 0.0014 | 1 -> 1 | both |
| A\* fold 1 seed 44 | control | 11 | 0.9875 -> 0.9875 | 395 -> 395 | 0.0345 -> 0.5327 | 0.9991 -> 0.9895 | 1.0000 -> 0.9986 | 0.0007 -> 0.0028 | 1 -> 1 | both |
| B fold 2 seed 42 | control | 9 | 0.9475 -> 1.0000 | 379 -> 400 | 0.2308 -> 0.0087 | 0.9888 -> 1.0000 | 0.9857 -> 0.9971 | 0.2986 -> 0.0076 | 1 -> 1 | both |
| A\* fold 2 seed 42 | control | 11 | 1.0000 -> 1.0000 | 400 -> 400 | 0.0001 -> 0.0003 | 1.0000 -> 1.0000 | 1.0000 -> 1.0000 | 0.0000 -> 0.0003 | 1 -> 1 | both |
| B fold 2 seed 43 | control | 7 | 1.0000 -> 1.0000 | 400 -> 400 | 0.0011 -> 0.0004 | 1.0000 -> 1.0000 | 0.9950 -> 0.9993 | 0.2597 -> 0.0043 | 1 -> 1 | both |
| A\* fold 2 seed 43 | ten | 1 | 0.5000 -> 1.0000 | 200 -> 400 | 60.9868 -> 0.0031 | 0.5000 -> 1.0000 | 0.5000 -> 0.9886 | 60.2927 -> 0.0371 | 0 -> 1 | dev after only |
| B fold 2 seed 44 | control | 10 | 0.9975 -> 1.0000 | 399 -> 400 | 0.0030 -> 0.0029 | 1.0000 -> 1.0000 | 0.9893 -> 0.9986 | 0.0313 -> 0.0031 | 1 -> 1 | both |
| A\* fold 2 seed 44 | ten | 1 | 0.5000 -> 0.9800 | 200 -> 392 | 18.0934 -> 0.3553 | 0.2114 -> 0.9814 | 0.5000 -> 0.9957 | 15.9745 -> 0.0492 | 0 -> 1 | dev after only |
| B fold 3 seed 42 | control | 12 | 1.0000 -> 1.0000 | 400 -> 400 | 0.0168 -> 0.0075 | 1.0000 -> 1.0000 | 0.9993 -> 1.0000 | 0.0051 -> 0.0005 | 1 -> 1 | both |
| A\* fold 3 seed 42 | ten | 1 | 0.5000 -> 1.0000 | 200 -> 400 | 3.1296 -> 0.0137 | 0.4690 -> 1.0000 | 0.5000 -> 0.9986 | 2.8493 -> 0.0147 | 0 -> 1 | dev after only |
| B fold 3 seed 43 | control | 10 | 0.9925 -> 0.9675 | 397 -> 387 | 0.0531 -> 0.0640 | 1.0000 -> 1.0000 | 1.0000 -> 1.0000 | 0.0002 -> 0.0001 | 1 -> 1 | both |
| A\* fold 3 seed 43 | control | 12 | 0.9975 -> 0.9175 | 399 -> 367 | 0.0088 -> 0.1670 | 1.0000 -> 1.0000 | 1.0000 -> 1.0000 | 0.0014 -> 0.0002 | 1 -> 1 | both |
| B fold 3 seed 44 | apart | 3 | 0.2375 -> 0.9950 | 95 -> 398 | 3.2731 -> 0.0417 | 0.0950 -> 0.9986 | 0.2343 -> 0.9979 | 2.4545 -> 0.0130 | 0 -> 1 | dev after only |
| A\* fold 3 seed 44 | ten | 1 | 0.5000 -> 0.8275 | 200 -> 331 | 33.4090 -> 0.2594 | 0.5000 -> 1.0000 | 0.5000 -> 0.9971 | 31.6305 -> 0.0172 | 0 -> 1 | dev after only |
| B fold 4 seed 42 | control | 11 | 1.0000 -> 0.9975 | 400 -> 399 | 0.0005 -> 0.1622 | 1.0000 -> 0.9950 | 1.0000 -> 1.0000 | 0.0000 -> 0.0000 | 1 -> 1 | both |
| A\* fold 4 seed 42 | control | 10 | 0.7975 -> 0.7875 | 319 -> 315 | 0.8106 -> 0.7175 | 0.9990 -> 0.9916 | 0.9986 -> 0.9971 | 0.0580 -> 0.0111 | 1 -> 1 | both |
| B fold 4 seed 43 | control | 6 | 1.0000 -> 0.9975 | 400 -> 399 | 0.0013 -> 0.0047 | 1.0000 -> 1.0000 | 0.9993 -> 1.0000 | 0.0071 -> 0.0001 | 1 -> 1 | both |
| A\* fold 4 seed 43 | control | 9 | 0.9950 -> 0.9175 | 398 -> 367 | 0.0272 -> 0.1516 | 0.9993 -> 0.9991 | 0.9993 -> 0.9993 | 0.0036 -> 0.0024 | 1 -> 1 | both |
| B fold 4 seed 44 | ten | 2 | 0.5000 -> 1.0000 | 200 -> 400 | 2.3605 -> 0.0013 | 0.9652 -> 1.0000 | 0.5000 -> 0.9971 | 2.2382 -> 0.0143 | 0 -> 1 | dev after only |
| A\* fold 4 seed 44 | ten | 1 | 0.5000 -> 0.8100 | 200 -> 324 | 22.8274 -> 0.6411 | 0.5000 -> 0.9909 | 0.5000 -> 0.9986 | 22.9596 -> 0.0063 | 0 -> 1 | dev after only |
| B' fold 1 seed 42 | control | 2 | 0.7725 -> 0.9950 | 309 -> 398 | 0.6101 -> 0.3244 | 0.9937 -> 0.9948 | 0.7976 -> 0.9979 | 0.5912 -> 0.0121 | 1 -> 1 | both |
| B' fold 1 seed 43 | control | 4 | 0.9400 -> 0.9975 | 376 -> 399 | 0.8647 -> 0.0096 | 0.9899 -> 1.0000 | 0.8126 -> 0.9989 | 9.4140 -> 0.0020 | 1 -> 1 | dev after only |
| B' fold 1 seed 44 | ten | 1 | 0.5000 -> 0.9950 | 200 -> 398 | 12.1449 -> 0.0146 | 0.4975 -> 0.9998 | 0.5000 -> 0.9989 | 13.0656 -> 0.0037 | 0 -> 1 | dev after only |
| B' fold 2 seed 42 | control | 11 | 0.8775 -> 0.9875 | 351 -> 395 | 0.8750 -> 0.0215 | 0.9830 -> 1.0000 | 0.7784 -> 0.9989 | 1.0785 -> 0.0015 | 1 -> 1 | dev after only |
| B' fold 2 seed 43 | control | 10 | 1.0000 -> 0.9975 | 400 -> 399 | 0.0042 -> 0.0093 | 1.0000 -> 1.0000 | 1.0000 -> 1.0000 | 0.0052 -> 0.0001 | 1 -> 1 | both |
| B' fold 2 seed 44 | ten | 2 | 0.5000 -> 0.9975 | 200 -> 399 | 11.4259 -> 0.0065 | 0.5000 -> 1.0000 | 0.5000 -> 1.0000 | 11.6160 -> 0.0003 | 0 -> 1 | dev after only |
| B' fold 3 seed 42 | control | 1 | 0.8325 -> 0.8800 | 333 -> 352 | 0.3211 -> 0.3449 | 1.0000 -> 1.0000 | 0.7216 -> 1.0000 | 0.4431 -> 0.0027 | 1 -> 1 | dev after only |
| B' fold 3 seed 43 | control | 9 | 1.0000 -> 0.9975 | 400 -> 399 | 0.0011 -> 0.0279 | 1.0000 -> 1.0000 | 0.9925 -> 1.0000 | 0.1127 -> 0.0001 | 1 -> 1 | both |
| B' fold 3 seed 44 | control | 9 | 0.9975 -> 0.9175 | 399 -> 367 | 0.0341 -> 0.1218 | 1.0000 -> 1.0000 | 0.9968 -> 1.0000 | 0.0385 -> 0.0001 | 1 -> 1 | both |
| B' fold 4 seed 42 | control | 2 | 0.9175 -> 0.9500 | 367 -> 380 | 0.1375 -> 0.3614 | 0.9966 -> 0.9485 | 0.9968 -> 0.9968 | 0.0260 -> 0.0604 | 1 -> 1 | both |
| B' fold 4 seed 43 | control | 3 | 1.0000 -> 1.0000 | 400 -> 400 | 0.0142 -> 0.0037 | 1.0000 -> 1.0000 | 0.9722 -> 0.9989 | 0.0761 -> 0.0024 | 1 -> 1 | both |
| B' fold 4 seed 44 | ten | 1 | 0.5000 -> 1.0000 | 200 -> 400 | 17.2907 -> 0.0281 | 0.5000 -> 1.0000 | 0.5000 -> 0.9979 | 16.4776 -> 0.0075 | 0 -> 1 | dev after only |

## Model reported apart

B fold 3 seed 44 (restored epoch 3, not degenerate, outside the criterion at its checkpoint), `aggregate.md`: dev
accuracy 0.2375 -> 0.9950 (95 -> 398 correct, `results.csv`), val_loss 3.2731 -> 0.0417, AUROC 0.0950 -> 0.9986;
fitting accuracy 0.2343 -> 0.9979, fitting loss 2.4545 -> 0.0130. It meets the criterion after; this is not part of
the primary reading.

## Fitting before × dev after (2×2)

Each cell counts checkpoints; no cell names a cause (`aggregate.md`):

| fitting before (original statistics, inference mode) | dev after: meets the criterion | dev after: does not meet it |
|---|---|---|
| meets the criterion | both: 0 of the ten, 22 of all | fitting only: 0 of the ten, 0 of all |
| does not meet it | dev after only: 10 of the ten, 14 of all | neither: 0 of the ten, 0 of all |

The 14 in "dev after only" (`results.csv`, `cell` = `fitting_not_met_dev_met`) are the ten, the model apart and three
controls: B' fold 1 seed 43 (fitting before: accuracy 0.8126, loss 9.4140), B' fold 2 seed 42 (0.7784, 1.0785) and B'
fold 3 seed 42 (0.7216, 0.4431). The 22 in "both" are the other 22 controls. All 36 checkpoints meet the criterion on
the dev after and on the fitting set after (`dev_criterion_after` and `fitting_criterion_after` = 1 in 36 of 36 rows).

## Fitting before

With the original statistics, in inference mode, each of the ten has fitting accuracy 0.5000
(`fitting_accuracy_before` = 0.5 in each of the ten rows of `results.csv`) and fitting loss from 2.2382 (B fold 4 seed
44) to 60.2927 (A\* fold 2 seed 43); their dev accuracy before is 0.5000 too (200 of 400). This is what the evaluation
gives on these checkpoints; it is reported without interpretation.

## Per-layer change of the statistics

Definitions, as the runner computes them (`change_summary` in `livenesslab/scripts/nuaa_bn_recalibration.py`), per
channel, with the installed (float32) and original values of each layer: `var_ratio` = installed moving variance /
original moving variance; `mean_shift` = |installed moving mean − original moving mean| / sqrt(original moving variance
+ epsilon), epsilon = 0.001. Each row of `results.csv` holds, per layer, the minimum, median and maximum over the
channels of that model (`<layer>_var_ratio_min`, `_median`, `_max`; `<layer>_mean_shift_min`, `_median`, `_max`).
Below, for each group and layer, only the smallest per-model minimum and the largest per-model maximum over the models
of the group:

| group (models) | layer | var_ratio: smallest min | var_ratio: largest max | mean_shift: smallest min | mean_shift: largest max |
|---|---|---|---|---|---|
| ten (10) | `bn1_1` | 2.484e-11 | 0.2133 | 2.988e-07 | 0.5227 |
| ten (10) | `bn1_2` | 0.0280 | 1.1455 | 0.0025 | 0.4927 |
| ten (10) | `bn2_1` | 0.1215 | 1.6696 | 0.0018 | 0.3813 |
| ten (10) | `bn2_2` | 0.2410 | 2.0870 | 0.0250 | 0.5477 |
| ten (10) | `bn_dense` | 0.0 | 131.3149 | 3.997e-06 | 7.3796 |
| control (25) | `bn1_1` | 0.0 | 2.0019 | 2.345e-06 | 1.3277 |
| control (25) | `bn1_2` | 0.0551 | 2.0860 | 0.0002 | 0.3847 |
| control (25) | `bn2_1` | 0.1042 | 1.3961 | 0.0002 | 0.3933 |
| control (25) | `bn2_2` | 0.2781 | 2.7373 | 0.0002 | 0.5329 |
| control (25) | `bn_dense` | 0.0 | 22070.4519 | 1.191e-07 | 14.5464 |
| apart (1) | `bn1_1` | 6.057e-09 | 0.3845 | 1.352e-05 | 0.2369 |
| apart (1) | `bn1_2` | 0.3286 | 1.0664 | 0.0103 | 0.2840 |
| apart (1) | `bn2_1` | 0.3091 | 1.1353 | 0.0042 | 0.3740 |
| apart (1) | `bn2_2` | 0.5483 | 2.0259 | 0.0045 | 0.4418 |
| apart (1) | `bn_dense` | 0.0 | 18.7599 | 0.0005 | 1.2725 |

A `var_ratio` minimum of 0.0 means that at least one channel has installed variance 0.0; it occurs in `bn_dense` for
10 of the ten, 23 of the 25 controls and the model apart, and in `bn1_1` for 7 of the 25 controls (`results.csv`,
`<layer>_var_ratio_min` = 0.0). The full per-channel values (original, float64, installed, recomputed) are in the
restricted statistics files.

## Time

`run_record.json`, `timings`; the note gave one estimate for the whole run (about 32 minutes, upper estimate about 140,
frame reading not priced; cap 240 minutes), not one per phase:

| phase | seconds |
|---|---|
| phase 0 inputs | 2.8 |
| phase 1 baseline (36 models) | 342.0 |
| phase 2 recalibration and evaluation (36 models) | 1400.9 |
| final checks and writes | 0.6 |
| wall time | 1748.6 (29.1 minutes) |

The four phases add up to 1746.3 s; the remaining 2.3 s are not attributed to a phase.

Per model (`timings.per_model`, smallest and largest): baseline 9.2-12.3 s for B, 9.2-10.3 s for A\*, 7.8-10.3 s for
B'; recalibration (frame assembly, loading, copy, five fitting passes, tensor comparison) 26.7-32.2 s for B,
26.8-31.2 s for A\*, 18.0-19.1 s for B'; evaluation after (consistency and fitting pass, dev pass) 14.9-16.6 s for B,
14.8-17.8 s for A\*, 10.3-10.7 s for B'. Peak resident memory 2029 MB (`run.log.txt`). Not stopped (`stopped`: null);
within the cap.

## What this shows and what it does not

On these 36 checkpoints, with this estimator, the ten degenerate checkpoints meet the recovery criterion on the dev
after the re-estimation, the 25 controls keep it, and 17 controls show at least one worsening indicator (listed above).
The limits fixed before the run (`NOTE.md`, sections 4 and "Limits"; `reading_rule.limits` of the configuration)
apply as written:

- a recovery shows the effect of this recalibration on the examined checkpoints, not a single cause of the saturation;
  a missing recovery would not, in general, exclude a role of BatchNormalization or dropout;
- the moving statistics, dropout and the augmentation of the fitting set (hypotheses 1-3) are not separated by this
  control;
- the later saturated epochs, at residual ≤ 0.0704, are not examined: only the restored checkpoint of each training is
  saved;
- the checkpoints were selected on the same dev (minimum dev val_loss);
- small dev (2-3 subjects per fold, 400 frames);
- one estimator;
- the criteria were defined after the patience-4 run;
- each cell and each case states what is observed with these statistics on these checkpoints and names no cause.

Nothing in this report concerns the test.

**Not proposed here.** As in `NOTE.md`: "Possibilities only, to be agreed after the report; none follows from the
outcome alone: the momentum, the learning rate, the initialisation."
