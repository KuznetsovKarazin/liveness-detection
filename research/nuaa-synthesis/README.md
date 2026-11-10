# NUAA synthesis: Methods/Results of C1, M1, N1, N2, the C1 crop check, the subject-held-out folds run, the session ablation run and the patience verification, in one table

Written on the coordinator's request of 2 November 2026 (C2 card), revised on 3 November after the coordinator's review
(three precisions, labels of the generated tables, checksums) and on 4 November on the coordinator's request of that
day, after the merge of the subject-held-out folds run (PR #18): its results in the unified table (every fold and seed:
12 fold × seed rows, 3 rows pooled over the folds and 3 rows of the fold models on the 3,000 N1 test frames, with the
dev threshold next to 0.5 and the bootstrap over subjects; 39 rows, from 21), two annexes (every session stratum of the
run; the comparison with N1 on the same 3,000 frames), the run in Methods and Results, the descriptive comparison with
N1 (§3.6; run `20261104-FOLDSCOMPARE-3b3bcca`, source `research/nuaa-folds-n1-compare/tables/aggregate.json`), a short
analysis of fold 1 seed 43 and fold 4 (§3.7), a fourth reading and the section on the next experiment rewritten as its
state. Revised on 10 November on the coordinator's request of that day, after the merge of the session ablation run (PR
#21, run `20261106-SESSABLRUN-d999458`, early-stopping patience 4), the analysis of its degenerate trainings (PR #22)
and the patience verification (PR #23: run `20261109-SESSABLRUN-6a50265` with patience 12 and the comparison of the two
runs `20261109-PATVER-4f2b925`): both runs in the unified table (18 rows pooled over the four folds, one per arm, seed
and run, at 0.5 and at the dev threshold, with the bootstrap over subjects; 57 rows, from 39) and two annexes,
`tables/patience_table.md` (the 36 models with patience 4 and 12 side by side: train and dev, then the test, each
followed by counts computed from its rows; its compact form, `tables/patience_table_compact.md`, with the 36 models and
the same counts, is embedded in `SYNTHESIS.md`) and `tables/session_ablation_table.md` (every arm, seed and session
stratum of both runs and the 84 verdicts of the reading rule of each run side by side, the 7 that differ marked). This
folder holds the synthesis document, the unified table rebuilt from the public aggregates of `research/pad-2026`, of the
update of 4 November and of the packages of PR #21 and PR #23, the script that rebuilds and checks it, its tests and
their logs. No inference and no training: every cell comes from a file of this branch, accepted in `research/pad-2026`,
delivered with the update of 4 November (FOLDS-N1 rows: run `20261104-FOLDSCOMPARE-3b3bcca`, folder
`research/nuaa-folds-n1-compare`) or accepted with PR #21 and PR #23 (SESSABL rows and the two new annexes), named with
its SHA-256 in `tables/summary_table.json` (twenty-five sources: the C1 and M1 tables, the M2 manifest aggregate, the
N1, N2 and crop-check aggregates, the N1 subject file, the public fold schema, the aggregate and the run record of the
folds run, the aggregate of the comparison with N1; for each of the two session ablation runs its aggregate, run record,
public `SHA256SUMS` and `status.json`; for the comparison of the two runs its aggregate, `comparison.csv`, `curves.csv`,
`SHA256SUMS` and `status.json`). Besides copied values, the patience annex holds counts derived from the public rows and
epoch curves (the recovery criterion of the comparison at the last epoch run, restored epochs changed, test values
lower, equal or higher with patience 12, and two counts at the cut-offs used in `CONCLUSION.md` of the patience
verification, first recovery at epoch 10 or later and dev accuracy below 0.9, cut-offs taken from it): counts of models,
not new metrics. "Restored epoch changed" compares the restored epochs only: with the same restored epoch the weight and
threshold file hashes still differ in every model, and the identity of the 21 unchanged models is shown by identical
test values and dev threshold, which the script checks. C1, M1, M2, N1, N2, the C1 crop check, the fold schema, the step
zero, the folds run, the C2 runner, the R2 reference, the session ablation run, the degenerate-trainings package and the
patience verification are unchanged. The script reads nothing restricted; §3.7 quotes values computed read-only from the
restricted run files (sources given): descriptive calculations on the existing scores, no inference, no training, not
verified by `check`.

## Contents

| path | what it is |
|---|---|
| `SYNTHESIS.md` | the synthesis: question, methods (with the subject-held-out folds run), results (unified table in compact form, with the rows of the session ablation run and of its patience-12 verification; same sessions, session transfer, dataset transfer, context, the folds run by seed, stratum and fold, the comparison with N1 on the same 3,000 frames, the analysis of fold 1 seed 43 and fold 4, then the session ablation run and the patience comparison, with the 36 models in the compact table of `tables/patience_table_compact.md`), readings and limits, state on 10 November and the proposed control |
| `tables/summary_table.md` | the unified table, one row per result accepted in `research/pad-2026` or delivered with this update (FOLDS-N1 rows: run `20261104-FOLDSCOMPARE-3b3bcca`, folder `research/nuaa-folds-n1-compare`); 57 rows: C1 4, M1 4, N1 3, N2 sessions 6, N2 combined 2, crop check 2, folds run 15 = 12 fold × seed and 3 pooled per seed, folds run on the N1 test frames 3, session ablation run with patience 4 9 and with patience 12 9 (arms B, A*, B' × seeds 42/43/44, pooled over the four folds, stratum all); with model, training data, early stopping, test set, unit, extractor, threshold and its origin, denominators, error counts, APCER/BPCER/ACER/AUROC, for the folds-run and session-ablation rows also the dev threshold with its error counts and rates and, for the pooled and the N1-frames rows, the bootstrap over subjects; run id, same-session or transfer |
| `tables/summary_table_compact.md` | the same rows with codes and a legend (the form used in `SYNTHESIS.md`) |
| `tables/summary_table.csv` | the same rows, machine-readable, with a `source` column (file and SHA-256 prefix) |
| `tables/summary_table.json` | the rows plus the sources with full SHA-256, the run ids, the crop paired statistics, the N2 subsets (fitting and early-stopping subjects), the N1 C1-subset rows |
| `tables/cross_dataset_table.md` | annex: the 16 cross-dataset CNN rows of the C1 table (dataset transfer to NUAA at 0.5) |
| `tables/context_rows_table.md` | annex: the other analyzers of the C1 table on the same 300 images (pre-trained models, probes, hand-crafted features) |
| `tables/subject_sensitivity_table.md` | annex: N1 part 1, sensitivity of the M1 result to the 16 test subjects |
| `tables/n1_same_ids_table.md` | annex: the folds run versus N1 on the same 3,000 frames, per seed with the bootstrap, differences with the paired bootstrap, median and range, paired decisions per frame, subject counts, control values of the coordinator |
| `tables/folds_run_table.md` | annex: the folds run, every session stratum (01, 02, 03, all) per fold and seed, median and range over the seeds per fold, the pooled rows per seed and stratum with the bootstrap, median and range of the pooled rows, the 12 models (training and dev thresholds) |
| `tables/patience_table.md` | annex: the 36 models of the session ablation run (arm, fold, seed, one of the ten original degenerate trainings), patience 4 and 12 side by side: epochs run, restored epoch, degenerate, first recovery, recovery holding at the last epoch, best val_loss, dev accuracy and AUROC at the checkpoint, epochs at 0.5, last lr (fixed decimals), shared epochs and whether identical, restored epoch changed, then counts computed from the rows; a second table, same order, with the test per model (ACER at 0.5 and at the dev threshold, AUROC; strata 03 and all) in both runs, then counts of models lower, equal or higher with patience 12 |
| `tables/patience_table_compact.md` | the 36 models in compact form for `SYNTHESIS.md` (model, in the ten, restored epoch, first recovery, recovery holding at the last epoch and dev accuracy at the checkpoint as "p4 → p12", restored epoch changed), a legend line and the same counts as the train and dev table of `patience_table.md`, built from the same rows |
| `tables/session_ablation_table.md` | annex: the two session ablation runs (patience 4 and 12), pooled over the four folds per arm, seed, stratum (01, 02, 03, all) and patience with the bootstrap, median and range over the seeds, the 84 verdicts of the reading rule of each run side by side with the per-seed differences and their paired bootstrap intervals, the verdicts that differ marked, then counts |
| `livenesslab/scripts/nuaa_synthesis.py` | `build` and `check` (with `--synthesis`, `check` also requires every table line of the compact table and every table and summary line of the compact patience table to appear verbatim in `SYNTHESIS.md`); standard library only |
| `livenesslab/scripts/check_nuaa_synthesis.py` | 123 tests on a synthetic `research/` tree (no data, no dependency; the two session ablation runs and their comparison with 36 models, epoch curves and 84 verdicts each, with boundary cases: a val_loss tie, val_accuracy 0.75 and val_loss 1.0 exactly, a dev accuracy of 0.9 and a first recovery at epoch 10 exactly, an NA cell, an lr of 6.25e-05, different verdict counts in the two runs): every row and annex cell against the fixture, every consistency check against a faulty source (folds run, comparison with N1, session ablation runs and patience comparison included), trial runs (dirty tree, run id ending with `-dev`) refused, the reading rule at its boundaries, byte-exact `check`, `check --synthesis` failing when a line of either compact table is missing or altered, every summary line asserted in full, error paths without traceback |
| `test_logs/check_nuaa_synthesis.txt`, `test_logs/linux.txt` | outputs of the tests on macOS and in a Linux container |
| `SHA256SUMS` | SHA-256 of every committed file of this folder |

## Sources (all in this branch; merge commits of the accepted pull requests)

| block | folder | file read | accepted |
|---|---|---|---|
| C1 | `research/c1-livenesslab/` | `tables/nuaa_table.md` (run `20260928-C1-seed42-a80f6f9`) | PR #1, merge `7731076` |
| M1 | `research/m1-metrics/` | `tables/nuaa_test_dev_threshold_table.md` (run `20261003-M1-seed42-e67ecc7`) | PR #2 (metric module, merge `0b06e69`) and PR #3 (dev threshold, merge `b9aed95`) |
| M2 | `research/m2-nuaa-manifests/` | `nuaa_manifests_aggregate.json` (counts and subjects of the NUAA manifests; accepted package `20261006-M2-seed42-36c677a`, aggregate regenerated at `5fcc1f8`) | PR #4 (merge `4f9f585`) and PR #6 (merge `21e1ad7`) |
| N1 | `research/n1-nuaa-stability/` | `tables/aggregate.json` (study `20261024-N1-LivenessNet-fb93496`), `tables/subject_sensitivity/n1_subject_sensitivity_public.json` (run `20261024-N1-subjects-a5fcbbc`) | PR #10 (merge `c9f1e3f`) and PR #11 (merge `688afc5`) |
| N2 | `research/n2-nuaa-check/` | `tables/sessions/aggregate.json` (study `20261029-N2-sessions-2ac3e29`), `tables/combined/aggregate.json` (study `20261029-N2-combined-247dfd7`) | PR #13, merge `027a3ca` |
| crop check | `research/c1-crop-check/` | `tables/aggregate.json` (study `20261102-C1CROP-f6747cc`) | PR #14, merge `5254c5c` |
| fold schema | `research/nuaa-subject-folds/` | `tables/fold_schema_public.json` (run `20261103-FOLDS-67b7f3d`; test counts per fold, role and class × session cell) | PR #16, merge `cee1a6c` |
| folds run | `research/nuaa-folds-run/` | `tables/aggregate.json` and `tables/run_record.json` (run `20261103-FOLDSRUN-e174812`) | PR #18, merge `96d3fc1` |
| comparison with N1 | `research/nuaa-folds-n1-compare/` | `tables/aggregate.json` (run `20261104-FOLDSCOMPARE-3b3bcca`, commit `3b3bcca`, clean tree; SHA-256 `d191efdf3d38f45386bd8af4756270244758c61275c6fe960769b76b84aaff4a`, as in its `tables/SHA256SUMS`) | delivered with the update of 4 November |
| session ablation run (patience 4) | `research/nuaa-session-ablation-run/` | `tables/aggregate.json`, `tables/run_record.json`, `tables/SHA256SUMS`, `run/status.json` (run `20261106-SESSABLRUN-d999458`) | PR #21, merge `67f641c` |
| degenerate trainings | `research/nuaa-degenerate-trainings/` | not read: its criteria (degenerate, saturation, recovery, checkpoint agreement) are read from the comparison aggregate of PR #23, which copies them with the same values | PR #22, merge `12dc78a` |
| patience verification (patience 12) and comparison of the two runs | `research/nuaa-patience-verification/` | `tables/aggregate.json`, `tables/run_record.json`, `tables/SHA256SUMS`, `run/status.json` (run `20261109-SESSABLRUN-6a50265`); `comparison/aggregate.json`, `comparison/comparison.csv`, `comparison/curves.csv`, `comparison/SHA256SUMS`, `comparison_run/status.json` (run `20261109-PATVER-4f2b925`) | PR #23, merge `ec6c023` |

The C1 rows of the unified table take their counts from the M1 table (which carries the counts at 0.5 and at the dev
threshold for the same scores); the script checks them against the C1 table (rates, AUC, denominators), against the
rates themselves at four decimals, and against the error counts of the N1 subject file at both thresholds. The
descriptive texts (frames and subjects of each set) are built from the M2 aggregate and from the N2 target block, and
every row's denominators are compared with them. The script also checks that every aggregate was produced from a clean
tree with `passed` true (the N1 subject file, which has no `passed` field: clean tree and accepted M1 table reproduced),
that each run id ends with the recorded commit, that the N2 target equals the exclusions recorded by M2 and the subject
counts agree between M2 and the texts, that the N2 subject groups have the same denominators for every seed and add up
to their sets, that every JSON row's rates equal its counts at four decimals, that no Markdown table has a duplicated
row, that the N2 seeds, weights and session 03 rows equal N1, that the N1 reference inside the N2 combined aggregate
equals N1, that the C1 reference inside the crop-check aggregate equals the C1 row (counts, rates, run) and is
reproduced by the C1-extractor row, and that the M1 thresholds in the N1 subject file equal the M1 table. For the folds
run it checks that the aggregate, the run record and the fold schema come from clean trees, that aggregate and record
describe the same run of the delivered configuration (run id, commit, configuration, bootstrap) and the approved schema
(run id, configuration SHA-256, test rows per fold), that the denominators of every fold and stratum equal the schema
counts, that the four test folds add up to the official lists of M2 (12,614 frames, 16 subjects), to the originals
verified by the run and to the cache plus the extracted frames, that the stratum all is the sum of the session strata,
that the pooled rows are the sum of the four folds, that medians and ranges over the seeds are those of the per-seed
values, that the bootstrap has the declared parameters, that the 12 models have the schema sizes, a valid restored epoch
and a timing in the record, and that the run stayed within the cap. For the comparison with N1 it requires an official
run (clean tree, run id without `-dev`) of the delivered configuration, and checks its inputs against M2 (manifest
SHA-256 and counts), the folds run, N1 and the fold schema, its N1 values against the N1 rows of the unified table, its
folds counts against the pooled session 03 of the folds run (a subset: consistent, not equal), the differences (folds
minus N1), the paired decisions and subject counts against the counts, the medians and ranges, the bootstrap parameters
and the control values of the coordinator (agreement at the digits he gave: 100 × ACER rounded to two decimals, AUROC to
five; no tolerance). For each of the two session ablation runs it checks that aggregate, run record and `status.json`
come from a clean tree with status `PASS` (`passed` true in `status.json`), describe the same run of the delivered
configuration (run id ending with the commit and without `-dev`, commit, configuration SHA-256, end time, amendment)
and are anchored (`status.json` gives the SHA-256 of the public `SHA256SUMS`, which gives those of the aggregate and the
record); that the 36 models (3 arms × 4 folds × 3 seeds) are complete, none missing, not stopped, within the cap; that
the record names the approved fold schema and the folds run read here (run id, SHA-256 of its aggregate and record),
with the same bootstrap; that the fitting and dev cells of the record are the same in both runs and give the sizes of
every training card; that every rate equals its counts at four decimals, the denominators of every model and stratum
equal the fold schema, the stratum all is the sum of the sessions, the pooled rows are the sum of the four folds and
have the subjects of the folds run, medians and ranges equal the per-seed values and the bootstrap has the declared
parameters; that every difference of the three comparisons (per model and pooled) is minuend minus subtrahend; and that
each of the 84 verdicts of each run is the reading rule applied to the per-seed differences and intervals (verdict,
direction, reason, primary or secondary label), the primary verdicts included. The amendment of the patience-12 run must
name the configuration of the parent run and `training.early_stopping_patience` from the parent patience to the new
one, as given by the comparison. For the comparison of the two runs it requires an official run (clean tree, status
`PASS`, run id without `-dev`, delivered configuration, the declarations true), anchored (`status.json` → `SHA256SUMS` →
aggregate, `comparison.csv`, `curves.csv`), whose inputs are the two runs read here (run id, commit, configuration,
status, SHA-256 of their `status.json` and `SHA256SUMS`, 36 complete models, patience 4 and 12) and whose
`status.json` gives `verification_status` `PASS`, the status of the patience-12 run; it rebuilds every value of
`comparison.csv` except the dev AUROC at the checkpoint (copied; checked against the JSON rows and the group medians)
from `curves.csv` with the criteria of the comparison aggregate (restored epoch = first epoch of minimum dev val_loss,
epochs run = the early-stopping replay with the patience of the run, degenerate, first recovery, epochs at 0.5, last
lr, dev accuracy at the checkpoint within the checkpoint tolerance; the largest shared-history differences are required
to be 0) and checks it against the JSON rows of the comparison and the training cards of both aggregates; it checks the
shared histories (identical 36/36), the ten (the degenerate models of the patience-4 run), the verdict (`positive`,
reproduced from the rows), the group medians, section (d) (the verdicts and medians of both aggregates, 7 verdicts
changed) and section (e) (times equal to the run records, within the cap), and that every model whose restored epoch
did not change has identical test values and dev threshold in both runs (21 of 21). Every verdict direction is
checked in full (arm, sign, metric name as in the runner, arm), and the primary verdicts in verdict, direction, reason
and label. No generated Markdown table may hold a duplicated row. Any difference
stops the build and nothing is written.

## Commands (from the repository root)

Rebuild the tables:

```bash
python research/nuaa-synthesis/livenesslab/scripts/nuaa_synthesis.py build --research research --out research/nuaa-synthesis/tables
```

Check that the committed tables are byte for byte what the sources give and that every line of the compact table, and
every table and summary line of the compact patience table, is in `SYNTHESIS.md` verbatim (exit code 1 on any
difference):

```bash
python research/nuaa-synthesis/livenesslab/scripts/nuaa_synthesis.py check --research research --out research/nuaa-synthesis/tables --synthesis research/nuaa-synthesis/SYNTHESIS.md
```

The check covers the cells of the tables and the table inside the document; the numbers quoted in the prose of
`SYNTHESIS.md` were verified by hand against the tables and the sources.

Tests (synthetic sources written in a temporary folder; no data):

```bash
python research/nuaa-synthesis/livenesslab/scripts/check_nuaa_synthesis.py
```

Checksums of this folder:

```bash
cd research/nuaa-synthesis && shasum -a 256 -c SHA256SUMS
```

Dependencies: Python 3.11 or later, standard library only (tested on 3.12 in a `python:3.12` Linux container,
`test_logs/linux.txt`, and on 3.14 on macOS, `test_logs/check_nuaa_synthesis.txt`). `check` compares bytes, and `build`
writes UTF-8 with LF line endings on every platform.

## What the synthesis does not do

It adds no result, no uncertainty estimate and no statistical test to the accepted runs (the bootstraps of the folds
run, of the comparison and of the two session ablation runs are copied from their aggregates; the counts of the patience
annex are counts of models derived from the public rows and curves, not new metrics); the script does not re-read the
restricted per-image scores (§3.7 quotes read-only calculations on the restricted run files, with sources); it does not
change any accepted document; it runs no training and no inference. Section 5 of `SYNTHESIS.md` records the state on 10
November: folds run and step zero accepted (PR #16 to #18), session ablation run (PR #21), degenerate trainings (PR
#22), patience verification (PR #23), this update; the extension to the other session directions stays left out; it ends
with one proposed control (learning-rate reduction removed, everything else as in the patience-12 run), to be agreed
before any training, not started.
