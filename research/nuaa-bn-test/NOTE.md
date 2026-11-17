# BatchNormalization re-estimation of PR #27 on the NUAA test: before/after (pre-run note, 16 November 2026)

Assigned by the coordinator on the C2 card, 16 November 2026. A **post hoc diagnostic extension on a test already
analysed**: the "before" test metrics of these checkpoints are published (parent run `20261106-SESSABLRUN-d999458`),
and the re-estimation ("recalibration" in PR #27) was chosen and first measured on the dev of the same folds. It is not
an independent validation. Note, configuration and code are pushed before the run (compare the push time with `started`
in `status.json`); the procedure is not changed after any test result is seen.

## 1 Planned comparisons

All 36 checkpoints, including the 17 with a dev or fitting indicator worsened in PR #27 (marked in `results.csv` and in
the `aggregate.md` table of all models and list of worsenings). Per arm, fold and seed, same test frames, original vs
re-estimated statistics, scopes `test` (whole fold) and `session_03`: class counts, APCER, BPCER and ACER at the fixed
threshold 0.5, AUROC, after − before, worsening flags; per-subject results (restricted). Paired summaries: the ten
degenerate checkpoints, the 25 controls, the model apart, all 36, the three arms. Deliverables: before/after table,
figure of the differences, short conclusion (the reading).

## 2 What stays fixed

- The 36 `.keras` files (SHA-256 checked before each load and at the end); no training. The fold schema; test = every
  frame of the held-out subjects (3,242, 3,114, 3,439, 2,819 per fold, checked).
- Evaluation and estimator of PR #27: `nuaa_bn_recalibration.py` imported unchanged, pinned by SHA-256 (`8c228ef4…`);
  re-estimation on the fitting set of the model's arm and fold, on an in-memory copy, only `moving_mean` and
  `moving_variance` changed. Shared configuration sections equal those of PR #27 key by key, except two descriptive
  keys, rewritten because false here (`parent_run.used`, `evaluation.probability`).
- Threshold 0.5, attack if score > 0.5 (tie = bona fide); no threshold selected.

## 3 Phases, tolerances and stop rules (each phase for all 36 before the next)

| Phase | Checks | On failure |
|---|---|---|
| 0 inputs | SHA-256 of inputs, code and the PR #27 run | `FAIL` |
| 1 baseline | dev: the PR #27 baseline checks (probability 1e-5, no changed decision, val_loss 1e-3, accuracy 1e-6, softmax 1e-4); test: same frames as the stored test scores (ids, labels, subjects, sessions, file SHA-256, count, order; no realignment), probability within 1e-5, no decision changed at 0.5 (R2 tolerance, not measured on the test), softmax within 1e-4 | `BASELINE_FAILED`, `baseline.csv`; nothing after |
| 2 re-estimation | PR #27 internal checks (only the 10 moving tensors change; consistency; softmax) | `FAIL` |
| | same as the PR #27 run: statistics within 1e-4 relative + 1e-6 absolute (largest difference, bit-identity reported), dev probabilities within 1e-5, no decision changed | `RECALIBRATION_MISMATCH`, `recalibration_check.csv`; no re-estimated test score |
| 3a scoring | copy rebuilt and verified, same test frames, softmax; then the 36 checkpoints and the three anchoring SHA256SUMS re-verified | `FAIL` |
| 3b metrics | metrics, rows, summaries, outputs | `FAIL` |

`FAIL` removes every output; `status.json` keeps the reason and `run.log` the printed lines, which hold test metrics
only after an error in 3b or in the final writes. The 240-minute cap (checked before each model, never inside 3b) or a
voluntary stop gives `INCOMPLETE`: the checks completed so far are published, no test metric is computed or published.

## 4 What is read of the test

The fold schema with its test manifests is validated; the parent's `<model>_test_scores.csv` are hashed in phase 0 and
parsed in phase 1, which scores the test with the original statistics (PR #27 evaluator only) and writes or logs no
test metric (an accuracy and AUROC computed in memory are discarded). Phase 3a reassembles the same frames and scores
them re-estimated; phase 3b computes the metrics, also from the stored scores as a reported check. The parent's public
aggregate is only hashed; NUAA's official test lists index the frame cache.

## 5 Reading rule (applied by the script)

Primary: the ten, scope `test`, ACER at 0.5, mean over the ten of after − before and its subject-bootstrap interval:

- entirely below zero: on this test, the effect of this re-estimation on these checkpoints is a lower mean ACER;
- containing zero (a bound at zero counts as containing it): no direction shown, nor its absence;
- entirely above zero: the effect is a higher mean ACER;
- `not_available`: a metric undefined for a model, or no interval (fewer than 5 subjects or no valid resample).

Plus the count of the ten with ACER lower, equal or higher. Secondary, descriptive, same statements: session 03, other
groups, arms, APCER, BPCER, AUROC (threshold-free). Every worsening is listed with its size. No causal statement on the
saturation.

Bootstrap: the project's recipe unchanged (1,000 resamples of the held-out subjects pooled over the folds, seed 0,
percentiles 2.5–97.5, minimum 5 subjects), same resamples before and after. New, declared: a resample's statistic is the
mean over the models of their paired differences, dropped if any model's metric is undefined (drops reported). The
intervals rest on 16 subjects (15 with bona fide, 15 with attack frames; in session 03 nine with bona fide: 2, 3, 2, 2
per fold); with so few clusters a percentile interval may be too narrow. The model apart (four subjects) has none.

## 6 Outputs

Pull request: code, configuration, public outputs (`results.csv`, `baseline.csv`, `recalibration_check.csv`,
`summary.csv`, `aggregate.json`, `aggregate.md`, `differences.png`, run record, `SHA256SUMS`; no subject or frame
identifier). Shared restricted N1 folder: per-frame scores, statistics, per-subject results.

## 7 Time

From the PR #27 run on this machine (2.05 s per model + 3.61 ms per frame; re-estimation plus evaluation 1,400.9 s):
588 s (phase 1, 142,326 frames) + 1,401 s + 447 s (phase 3a) + about 90 s unmeasured ≈ 42 minutes; upper estimate
about 112 (slowest rates, doubled). As before, the four-hour limit is applied as elapsed time (the cap above); process
CPU time over all threads is recorded, not capped or estimated, and can exceed the elapsed time.
