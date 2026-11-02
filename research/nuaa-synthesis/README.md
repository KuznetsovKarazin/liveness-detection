# NUAA synthesis: Methods/Results of the accepted runs C1, M1, N1, N2 and the C1 crop check, in one table

Written on the coordinator's request of 2 November 2026 (C2 card). This folder holds the synthesis document, the
unified table rebuilt from the public aggregates already merged in `research/pad-2026`, the script that rebuilds and
checks it, its tests and their logs. No inference, no training, no new number: every cell comes from an accepted file
of this branch, named with its SHA-256 in `tables/summary_table.json` (eight sources: the C1 and M1 tables, the M2 manifest aggregate, the
N1, N2 and crop-check aggregates, the N1 subject file). C1, M1, M2, N1, N2, the C1 crop check, the C2
runner and the R2 reference are unchanged. Nothing restricted is needed or used.

## Contents

| path | what it is |
|---|---|
| `SYNTHESIS.md` | the synthesis: question, methods, results (unified table in compact form, three readings: same sessions, session transfer, dataset transfer), limits, proposed next experiment |
| `tables/summary_table.md` | the unified table, one row per accepted result (21 rows: C1 4, M1 4, N1 3, N2 sessions 6, N2 combined 2, crop check 2) with model, training data, early stopping, test set, unit, extractor, threshold and its origin, denominators, error counts, APCER/BPCER/ACER/AUROC, run id, same-session or transfer |
| `tables/summary_table_compact.md` | the same rows with codes and a legend (the form used in `SYNTHESIS.md`) |
| `tables/summary_table.csv` | the same rows, machine-readable, with a `source` column (file and SHA-256 prefix) |
| `tables/summary_table.json` | the rows plus the sources with full SHA-256, the run ids, the crop paired statistics, the N2 subsets (fitting and early-stopping subjects), the N1 C1-subset rows |
| `tables/cross_dataset_table.md` | annex: the 16 cross-dataset CNN rows of the C1 table (dataset transfer to NUAA at 0.5) |
| `tables/context_rows_table.md` | annex: the other analyzers of the C1 table on the same 300 images (pre-trained models, probes, hand-crafted features) |
| `tables/subject_sensitivity_table.md` | annex: N1 part 1, sensitivity of the M1 result to the 16 test subjects |
| `livenesslab/scripts/nuaa_synthesis.py` | `build` and `check` (with `--synthesis`, `check` also requires every line of the compact table to appear verbatim in `SYNTHESIS.md`); standard library only |
| `livenesslab/scripts/check_nuaa_synthesis.py` | 63 tests on a synthetic `research/` tree (no data, no dependency): every row and annex cell against the fixture, every consistency check against a faulty source, byte-exact `check`, error paths without traceback |
| `test_logs/check_nuaa_synthesis.txt`, `test_logs/linux.txt` | outputs of the tests on macOS and in a Linux container |
| `SHA256SUMS` | SHA-256 of every file of this folder |

## Sources (all in this branch; merge commits of the accepted pull requests)

| block | folder | file read | accepted |
|---|---|---|---|
| C1 | `research/c1-livenesslab/` | `tables/nuaa_table.md` (run `20260928-C1-seed42-a80f6f9`) | PR #1, merge `7731076` |
| M1 | `research/m1-metrics/` | `tables/nuaa_test_dev_threshold_table.md` (run `20261003-M1-seed42-e67ecc7`) | PR #2 (metric module, merge `0b06e69`) and PR #3 (dev threshold, merge `b9aed95`) |
| M2 | `research/m2-nuaa-manifests/` | `nuaa_manifests_aggregate.json` (counts and subjects of the NUAA manifests; accepted package `20261006-M2-seed42-36c677a`, aggregate regenerated at `5fcc1f8`) | PR #4 (merge `4f9f585`) and PR #6 (merge `21e1ad7`) |
| N1 | `research/n1-nuaa-stability/` | `tables/aggregate.json` (study `20261024-N1-LivenessNet-fb93496`), `tables/subject_sensitivity/n1_subject_sensitivity_public.json` (run `20261024-N1-subjects-a5fcbbc`) | PR #10 (merge `c9f1e3f`) and PR #11 (merge `688afc5`) |
| N2 | `research/n2-nuaa-check/` | `tables/sessions/aggregate.json` (study `20261029-N2-sessions-2ac3e29`), `tables/combined/aggregate.json` (study `20261029-N2-combined-247dfd7`) | PR #13, merge `027a3ca` |
| crop check | `research/c1-crop-check/` | `tables/aggregate.json` (study `20261102-C1CROP-f6747cc`) | PR #14, merge `5254c5c` |

The C1 rows of the unified table take their counts from the M1 table (which carries the counts at 0.5 and at the dev
threshold for the same scores); the script checks them against the C1 table (rates, AUC, denominators), against the
rates themselves at four decimals, and against the error counts of the N1 subject file at both thresholds. The
descriptive texts (frames and subjects of each set) are built from the M2 aggregate and from the N2 target block, and
every row's denominators are compared with them. The script also checks that every aggregate was produced from a clean
tree with `passed` true (the N1 subject file, which has no `passed` field: clean tree and accepted M1 table reproduced),
that each run id ends with the recorded commit, that the N2 target equals the exclusions recorded by M2 and the
subject counts agree between M2 and the texts, that the N2 subject groups have the same denominators for every seed and
add up to their sets, that every JSON row's rates equal its counts at four decimals, that no Markdown table has a duplicated row, that the N2 seeds, weights and session 03 rows
equal N1, that the N1 reference inside the N2 combined aggregate equals N1, that the C1 reference inside the crop-check
aggregate equals the C1 row (counts, rates, run) and is reproduced by the C1-extractor row, and that the M1 thresholds in
the N1 subject file equal the M1 table. Any difference stops the build and nothing is written.

## Commands (from the repository root)

Rebuild the tables:

```bash
python research/nuaa-synthesis/livenesslab/scripts/nuaa_synthesis.py build --research research --out research/nuaa-synthesis/tables
```

Check that the committed tables are byte for byte what the sources give and that every line of the compact table is in
`SYNTHESIS.md` verbatim (exit code 1 on any difference):

```bash
python research/nuaa-synthesis/livenesslab/scripts/nuaa_synthesis.py check --research research --out research/nuaa-synthesis/tables --synthesis research/nuaa-synthesis/SYNTHESIS.md
```

The check covers the cells of the tables and the table inside the document; the numbers quoted in the prose of
`SYNTHESIS.md` were verified by hand against the tables and the sources.

Tests (synthetic sources written in a temporary folder; no data):

```bash
python research/nuaa-synthesis/livenesslab/scripts/check_nuaa_synthesis.py
```

Dependencies: Python 3.11 or 3.12, standard library only (verified in a `python:3.12` Linux container,
`test_logs/linux.txt`). `check` compares bytes, and `build` writes UTF-8 with LF line endings on every platform.

## What the synthesis does not do

It adds no result, no uncertainty estimate and no statistical test to the accepted runs; it does not re-read the
restricted per-image scores; it does not change any accepted document. The proposal in section 5 of `SYNTHESIS.md` is a
protocol to agree on, not a run.
