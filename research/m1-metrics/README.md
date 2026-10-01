# M1 — PAD metrics: validation and source-dev threshold

Task **M1** of the PAD research workflow (`research/README.md`): verify the metric module of LivenessLab with
diagnostic cases whose expected values are derived by hand, fix the conventions for edge cases (constant and
near-constant scores, missing class, ties at the threshold, EER, BPCER@APCER10), define a threshold protocol on the
source development set only, and recompute the C1 table with the verified module. No new training; no development-set
score was computed for the delivered materials or appears in them: dev scoring is pending authorisation.

This folder is a filtered export of the operator's development repository (private; development commit `980ca52`,
run `20261001-M1-seed42-980ca52`): the metric module and the scripts needed to rerun the checks, the documents, the configuration and
the aggregate tables, without data, weights, per-subject records or working notes. As in C1, one file is redacted:
`src/tesi_app/usage_report.py` has its list of the operator's IP addresses emptied. The web application front end
(`src/tesi_app/static/`) is not repeated here: it is in `research/c1-livenesslab/`. Code comments, command-line
help and progress messages are in Italian by the developer's convention; reports, error messages and every document
of this folder are in English.

**Status of the source-dev threshold: protocol and code delivered, dev scoring pending authorisation.** The dev set
is built and its separation checks pass (see below); computing its scores is new inference on images and waits for
explicit authorisation, so `config/threshold.json` does not exist yet (`config/threshold.md` says so).

Contents of this folder:

| path | what it is |
|---|---|
| `run-report.md` | run report compiled from `research/templates/run-report.md` |
| `metrics.md` | definitions and conventions of every metric, with numerical examples, and the threshold selection protocol |
| `metric_cases.md` | the diagnostic cases of `scripts/check_metrics.py` with the hand derivation of every expected value |
| `table_diff.md` | row-by-row comparison of the recomputed NUAA table with the C1 table, with the explanation of every difference |
| `config/nuaa_config.json` | the C1 evaluation configuration (unchanged) |
| `config/threshold.md` | status of the source-dev threshold (`threshold.json` after the authorised dev scoring) |
| `manifests/` | counts and SHA-256 of the source-dev manifest (the per-image file is a per-subject record and stays in the restricted folder) |
| `tables/` | the NUAA table recomputed with the verified module, the APCER by attack type, the automatic row comparison with C1 |
| `livenesslab/` | `src/tesi_app` (metric module in `evaluation.py`), `scripts/check_metrics.py`, `scripts/select_threshold.py`, `scripts/check_select_threshold.py`, `scripts/check_c1_consistency.py`, `scripts/eval_dataset.py`, `requirements*.txt` |
| `SHA256SUMS` | SHA-256 of every other file of this folder, `livenesslab/` included |

Per-sample scores (`scores.csv` in the schema of `research/templates/score-schema.json`), the per-image dev manifest,
the dev summary and separation report, the logs and the check reports stay out of this public repository: they are
in the project workspace under `02_Experiments/M1/20261001-M1-seed42-980ca52/`, with the SHA-256 listed in `run-report.md` and in
`SHA256SUMS` of that folder. The run report is kept only in this public folder.

## Integrity

Completeness of a folder (this one, or the restricted run folder): every listed file matches its SHA-256 and no file
is missing from, or added to, the list. From inside the folder:

```bash
shasum -a 256 -c SHA256SUMS && test -z "$(find . -type l)" && diff <(awk '{print $2}' SHA256SUMS | sed 's|^\*||' | sort) <(find . -type f ! -type l ! -name SHA256SUMS | sed 's|^\./||' | sort) && echo COMPLETE
```

The last line printed must be `COMPLETE`. Run it on the folders as delivered, before the commands below: step 0
creates working folders (`livenesslab/results/`, `livenesslab/models/`) that are not in the list. A symbolic link anywhere in the folder makes the command fail (`test -z`),
so a listed file cannot be replaced by a link to another file.

Completeness alone does not detect a file added to this folder together with its line in `SHA256SUMS`. The
restricted run folder anchors this one: its file `public_SHA256SUMS.sha256` records the SHA-256, and its `README.md`
the SHA-256 and the file count, of this folder's `SHA256SUMS`; both were written before the restricted `SHA256SUMS`, which
therefore lists the anchor too. From inside this folder, with `M1_RUN` pointing at the restricted run folder (step 0
of "Commands"), compare the two:

```bash
shasum -a 256 -c "$M1_RUN/public_SHA256SUMS.sha256" && echo MATCH || echo MISMATCH
```

The last line printed must be `MATCH`. The opposite anchor does not exist: `run-report.md` does not record the SHA-256
of the restricted `SHA256SUMS` (that file lists the anchor of this folder, so the record would be circular); it
records the number of files of the restricted folder and the SHA-256 of its main files (`scores.csv`, manifests,
score cache), and the restricted folder is checked with the completeness command above.

## Relation to `research/c1-livenesslab`

`research/c1-livenesslab/` stays frozen as the evidence of C1 and is not modified by this task. From M1 on, the code
in `livenesslab/` of this folder supersedes the C1 copy of the same files; the application, the training scripts, the
front end and the checkpoint cards exist only in the C1 folder. Files of `livenesslab/` that differ from the C1 copy
(C1 code at development commit `73d00d3`) or are new, with their SHA-256:

| file (under `livenesslab/`) | status | SHA-256 in C1 | SHA-256 in M1 |
|---|---|---|---|
| `scripts/check_c1_consistency.py` | changed | `99f56ca0c08b36883230448ea27ab8b9a1c4af27ca21744324578064263550df` | `c0676623ab215eca4e48748aabc635ae2ee64b2f0f80f28a40791680d88ddd13` |
| `scripts/check_metrics.py` | new in M1 | — | `01c0ffef74f2b7f9b237c272365212890230eca0cf35ce25b964418f3ec34817` |
| `scripts/check_select_threshold.py` | new in M1 | — | `0ce70ccd71657bc23990928aed8dfbfa37c320995493a6ef470726e250a36374` |
| `scripts/eval_dataset.py` | changed | `84ff096b2d2a8dc5ea87e4ec3bdfabfb6b97182c0d8d765b45c696c9a546be89` | `47428dc0cd0c2e3135ba4381278865ce734f270bb323e2296e77894241a5cd24` |
| `scripts/select_threshold.py` | new in M1 | — | `690fa4dcc40f9895bf09dee2e7d321d4f97735fea79486f4b405a497e8072a9b` |
| `src/tesi_app/evaluation.py` | changed | `2b17f97a395d9ca4acf65ad3da7d459417c9470e004fa0a5337b6c6d984814aa` | `5deb4426538322807878895d5423a336b7a928288393b95e071cb35b783fa94e` |

6 files changed or new; 26 files identical to C1.

Every other file of `livenesslab/` is byte-identical to its copy in `research/c1-livenesslab/livenesslab/`. To repeat
the comparison from this folder of the group repository:

```bash
diff -rq ../c1-livenesslab/livenesslab livenesslab | grep -v '^Only in ../c1-livenesslab/livenesslab'
```

On the folder as delivered it prints one line per changed file (`Files … differ`) and one per new file (`Only in
livenesslab/…`); the files that exist only in the C1 folder are filtered out. After the commands below, the working
folders `livenesslab/results` and `livenesslab/models` also appear as `Only in livenesslab`.

## Conventions in one paragraph

Label 0 = bona fide, 1 = attack; score = probability of attack; attack if `score > threshold` (a tie is bona fide);
unit = image, with `n`, `n_bona_fide`, `n_attack` always reported. Scores are constant only if `max − min == 0`
exactly. A metric that cannot be computed is reported as not available (`—` in the tables, `null` in JSON) next to
its counts, never as 0. EER on the full ROC without interpolation, minimum of `|BPCER − APCER|` searched on integer
counts; BPCER@APCER10 at the empirical 0.10 quantile of the attack scores. Full definitions, formulas with
denominators and examples: `metrics.md`.

## Commands

All commands run inside `livenesslab/` with the environment of `research/c1-livenesslab/README.md` (Python 3.11,
`pip install -r requirements.txt`). One path scheme: leave `LIVENESSLAB_RESULTS_DIR` and `LIVENESSLAB_WEIGHTS_DIR`
unset, so that results live in `livenesslab/results/` and checkpoint cards in `livenesslab/models/weights/` (the
defaults); every path below is relative to `livenesslab/`.

```bash
cd research/m1-metrics/livenesslab
unset LIVENESSLAB_RESULTS_DIR LIVENESSLAB_WEIGHTS_DIR
```

**0. Restricted inputs.** Point two variables at the restricted run folders of the project workspace, verify their
integrity first, then copy exactly these files (nothing else is needed for commands 1, 2 and the checks):

```bash
C1_RUN=/path/to/02_Experiments/C1/20260928-C1-seed42-a80f6f9
M1_RUN=/path/to/02_Experiments/M1/20261001-M1-seed42-980ca52
(cd "$C1_RUN" && shasum -a 256 -c SHA256SUMS)
(cd "$M1_RUN" && shasum -a 256 -c SHA256SUMS && test -z "$(find . -type l)" && diff <(awk '{print $2}' SHA256SUMS | sed 's|^\*||' | sort) <(find . -type f ! -type l ! -name SHA256SUMS | sed 's|^\./||' | sort) && echo COMPLETE)
(cd .. && shasum -a 256 -c "$M1_RUN/public_SHA256SUMS.sha256") && echo MATCH || echo MISMATCH   # anchor of this public folder: must print MATCH
mkdir -p results/eval results/c1 results/m1 models/weights
cp "$C1_RUN/nuaa_score_cache.json" results/eval/nuaa.json
cp "$C1_RUN/nuaa_manifest.csv" "$C1_RUN/nuaa_table.md" results/c1/
cp ../../c1-livenesslab/checkpoints/*.json models/weights/
cp "$M1_RUN/nuaa_dev_manifest.csv" "$M1_RUN/nuaa_dev_summary.json" results/m1/
cp -R "$M1_RUN" results/m1/20261001-M1-seed42-980ca52
```

The last line makes a working copy of the M1 run folder: the consistency check writes its report into the run
folder, so it must never run on the delivered folder itself.

**1. Metric checks** (no data, no weights, a few seconds; exit code 1 if any case fails):

```bash
python scripts/check_metrics.py --require-restricted
```

Expected with the restricted inputs of step 0: 31 test cases, 989 checks, none skipped, last line `RESULT: PASS`.
Without them, `python scripts/check_metrics.py` skips the two real-data cases: 276 checks, last line
`RESULT: PASS (2 cases skipped: restricted inputs not found)`; with `--require-restricted` the same run fails (exit
code 1). The export of this package uses `--require-restricted`, so no skipped case is accepted in the run report.
Every case prints its expected value and the value obtained. The expected values of the
synthetic cases are derived by hand as exact fractions (`metric_cases.md`); the real-data cases use the independent
reference implementation of the check script, itself checked against those fractions in case `reference_self_check`;
none comes from scikit-learn or from the module under test.

**2. Recompute the NUAA table from the saved scores** (score cache and checkpoint cards only; no images, no weight
files):

```bash
python scripts/eval_dataset.py --dataset nuaa --from-cache --lang en \
    --export results/m1/nuaa_scores.csv --table results/m1/nuaa_table.md
python scripts/eval_dataset.py --dataset nuaa --from-cache --lang en --by-attack-type \
    --manifest results/c1/nuaa_manifest.csv --table results/m1/nuaa_table_by_attack_type.md
```

If a CNN of the cache has no JSON card in `models/weights/`, the script prints a warning on stderr and leaves that
row out of the table: copy all the cards as in step 0. Outside a git repository the header of the regenerated table
reports `Commit: n/d`; inside a clone of the group repository it reports that repository's commit. The table is tied
to the development commit of this run by `run-report.md`.

Compare the rows with the table of this folder (the header lines carry date and commit and differ by design):

```bash
diff <(grep '^| ' results/m1/nuaa_table.md) <(grep '^| ' ../tables/nuaa_table.md) && echo "table rows identical"
```

**3. Source-dev threshold** (`metrics.md`, "Threshold selection protocol"). Apart from `score-dev`, which runs
only after authorisation, no command of this step computes a score on an image. `build-dev` lists and hashes the
NUAA training images; `check-separation` lists and hashes the training and the test images; `select` and `synthetic-trial` list the file names of the official test split (and apply the
rule on the fifth field of the name) only to refuse any test image. `build-dev` and `check-separation` need the
official NUAA folders (`NUAA/images/{bonafide,attack}_{training,validation}`) under `LIVENESSLAB_DATA_DIR`. `synthetic-trial`,
`select --synthetic` and `apply` on a synthetic threshold use them when present; without them they print a `NOTE:` and
refuse test images by the file-name rule (fifth field `03`) and the C1 manifest only, and check the dev manifest against
the summary and the separation report only. A real `select` or `apply` requires the official folders and stops without
them; the file-name rule and the C1 manifest are applied in every case. The outputs of `build-dev` and `check-separation` are delivered in the M1 run
folder (`nuaa_dev_manifest.csv`, `nuaa_dev_summary.json`, `nuaa_dev_separation.json`); the export rebuilds them at
its own commit and requires the same manifest and the same separation outcome. `check-separation` also verifies the
dev manifest row by row against the reconstruction of the training split and against the files on disk, and refuses
any dev image whose name carries the test session `03` in the fifth field. `select` accepts a real (non-synthetic)
dev cache only with the provenance file written by `score-dev`; `apply` validates `threshold.json` and recomputes the
selection from the dev cache before using it, refuses a C1 score cache whose SHA-256 differs from the value recorded at
the C1 delivery (`--expected-c1-cache-sha256`, default `9b2b63b467e5e413c1fd5051e7c1c478f482301dc17070d60950d2e1daa34e65`;
an empty or malformed value is an error before any step, upper case is accepted),
flags as `non-informative` a row whose dev AUC is below 0.5 or whose dev EER is above 0.5 (with the reason; no score
inversion), and adds a sensitivity block (test metrics without the 22 attacks of the two dev subjects; no subject
identifier is written in the table).

```bash
python scripts/select_threshold.py build-dev
python scripts/select_threshold.py check-separation
```

The plan of the dev scoring can be printed without images, from the two files copied in step 0:

```bash
python scripts/select_threshold.py score-dev --dry-run
```

The next steps run only after authorisation: `score-dev --i-am-authorized` is new inference on the 574 dev images
with the 4 eligible CNNs (images and weights required), `select` freezes `results/m1/threshold.json`, `apply`
recomputes the 300 test images at the dev threshold next to the table at 0.5
(`results/m1/nuaa_test_dev_threshold_table.md`):

```bash
# after authorisation only:
# python scripts/select_threshold.py score-dev --i-am-authorized
# python scripts/select_threshold.py select
# python scripts/select_threshold.py apply
```

Until then, `select` and `apply` are exercised only on synthetic dev scores. The trial below builds a deterministic
fake dev cache (seed 42; bona fide ~ Beta(2, 5), attacks ~ Beta(5, 2); fingerprints copied from the C1 cache), runs
`select --synthetic` and `apply` on it and writes everything to a folder outside `results/`; its outputs are **not
results**. It needs the dev manifest, summary and separation report of the M1 run folder and the C1 score cache and
manifest of step 0; the official NUAA folders are optional (see above). The copy delivered
with this run is in `02_Experiments/M1/20261001-M1-seed42-980ca52/synthetic_threshold_trial/`.

```bash
cp "$M1_RUN/nuaa_dev_separation.json" results/m1/
python scripts/select_threshold.py synthetic-trial --out-dir "${TMPDIR:-/tmp}/m1-synthetic-trial"
```

## Rerun the checks

Consistency of the recomputed artefacts (score cache, exported CSV, `scores.csv` of the working copy of the run
folder, table), after steps 0 and 2:

```bash
python scripts/check_c1_consistency.py --dataset nuaa --c1 results/m1 \
    --run-dir results/m1/20261001-M1-seed42-980ca52 \
    --by-type-table results/m1/nuaa_table_by_attack_type.md \
    --expected-cache-sha256 9b2b63b467e5e413c1fd5051e7c1c478f482301dc17070d60950d2e1daa34e65 \
    --expected-config-sha256 f0dfdb90a1f61b86fa21e1519a742f180c632bdf677a6c9c45b022526ca8fa89 \
    --expected-manifest-sha256 86095da1531b1f24b1233c9668ac1ab37714978e444b313456732c028ccfb40c --self-test
```

The run folder holds `scores.csv`, `nuaa_manifest.csv` and `nuaa_config.json`; the checker finds the last two there
or in `--c1`. It compares by sets (analyzers, samples, pairs, duplicates) and as text (every value is compared with
the text the producing program writes: scores and thresholds as the round-trip text of the float, integer cells as
plain integers, metrics with the four-decimal format of the table), and exits with status 1 on any problem, printing
`N problem(s)`. It also checks: the SHA-256 of the manifest against the configuration and, with `--expected-manifest-sha256`,
against the given value; the SHA-256 of the configuration with `--expected-config-sha256`; the dataset of the cache
and, with `--expected-cache-sha256`, its SHA-256 (the three values above are those of the C1 cache, configuration and
manifest, also in `run-report.md`); the `subject` of every manifest row against the 4-digit prefix of the file name;
the analyzers of the configuration against the rows of the table (with and without score) and its `excluded` marks;
the checkpoint SHA-256 of the configuration against the fingerprint of each analyzer (full SHA-256 or 16-character
prefix); the fingerprints of the cache, of the exported CSV and of the configuration, and `checkpoint_sha256` of
`scores.csv`; every column of the group schema in
`scores.csv` (`run_id` equal to the run folder name, split `test`, modality `RGB`, subject and session from the
manifest, `attack_type` `print` for attacks and empty for bona fide); the Markdown tables with a strict parser
(canonical rows, exact header and separator, note rows of the analyzers without score); the per-type table,
recomputed with `metrics_by_attack_type`. `--self-test` builds 80 faulty copies of the artefacts and requires for
each one its expected message, not just any problem; the report ends with `self-test result: all faults detected
(80 faulty copies)`; a separate line of the self-test checks the handling of the `--expected-*-sha256` values
(not counted among the faulty copies). Each `--expected-*-sha256` value is taken without surrounding whitespace and
in lower case (upper case is accepted) and must then be exactly 64 hexadecimal characters: an empty, whitespace-only,
truncated or otherwise malformed value stops the checker with `ERROR:` and exit status 1 before any check, without
writing the report. The checker rewrites `consistency_check.txt` in the working copy; compare it with the delivered
one (only the date line differs).

The M1 run folder also holds `consistency_check_m1.txt`, regenerated by the export with the same checker: the same
check run on a temporary copy of the C1 folder in which the table, the exported CSV and the per-type table were
replaced by the recomputed ones. Its title names the C1 run (`20260928-C1-seed42-a80f6f9`) because the group-schema
`scores.csv` it checks is the C1 one: the scores are the same.

Offline tests of the threshold script (synthetic data in a temporary folder; no images, no restricted input, nothing
written under `results/`):

```bash
python scripts/check_select_threshold.py
```

Expected: `tests run: 39, failures: 0, errors: 0, skipped: 0` and `RESULT: PASS`.

## Differences

From C1 (`research/c1-livenesslab/`, merged at `7731076`):

- Metric module (`src/tesi_app/evaluation.py`): five documented differences from the reference code
  (`src/evaluation_utils.py`), each covered by a case of `scripts/check_metrics.py`:
  1. BPCER@APCER10 counts the rejected bona fide (score >= threshold), not the accepted ones;
  2. missing class: APCER or BPCER, ACER, AUC, EER and BPCER@APCER10 are `null`, never 0, with the counts reported;
  3. EER on the full ROC (`drop_intermediate=False`), with the minimum of `|BPCER − APCER|` searched on integer
     counts and ties resolved to the first point in decreasing threshold order;
  4. exact constancy test (`max − min == 0`): near-constant scores such as `[0.5, 0.5000001]` are ordinary scores
     (AUC 1, EER 0 in that example), while the C1 code (`np.allclose`) lost AUC and EER;
  5. constant scores with both classes: AUC 0.5 and EER 0.5, EER threshold and BPCER@APCER10 `null`.
  The inclusive decision `>=` exists only as a diagnostic (`decisions_at(..., inclusive=True)`), never in the tables.
  Invalid input (labels other than 0/1, also non-integer such as 0.7; NaN or infinite scores; different lengths)
  raises `ValueError` in `compute_metrics` and `metrics_by_attack_type` instead of being truncated or propagated.
- APCER by attack type: `metrics_by_attack_type` and `scripts/eval_dataset.py --by-attack-type`; on NUAA the single
  type `print` has denominator 150 and APCER equal to the global APCER on every row.
- Table: identical to the C1 table at four decimals on all 35 rows; the exported CSV is byte-identical. At full
  precision only the EER changes, on 12 of 30 rows, by at most 7 ulp (the new value is the correctly rounded exact
  fraction). Details in `table_diff.md`.
- Consistency checker: the three minor findings noted after the third C1 review are closed, and the checks listed
  in "Rerun the checks" were added (strict table parser, values compared as text, fingerprints, every column of the
  group schema, manifest and cache hashes, per-type table). No column read can abort the check: empty values, missing
  columns and malformed rows give dedicated problem messages and exit status 1. The self-test has 80 faulty copies,
  each with its expected message. On the valid C1 artefacts the result is unchanged (no inconsistency); the report
  has one line per check, so it is longer than the C1 one.
- Threshold script: `scripts/select_threshold.py` (new) and its offline tests `scripts/check_select_threshold.py`
  (39 tests on synthetic data).
- Table header: the commit of an external code folder is reported only if the folder is the root of its own git
  repository; otherwise `n/d` (never the commit of the enclosing repository).

## Scope and limits

- The recomputed table is a descriptive evaluation of existing checkpoints on the same 300-image subset of the
  official NUAA test set used in C1, at the fixed threshold 0.5; image-level counts with correlated frames; single
  seed; no confidence intervals.
- Source dev: 574 images (180 bona fide, 394 attacks) of the 2 subjects held out as validation subjects when the 4
  NUAA CNNs were trained (the per-subject list stays in the restricted folder). Separation checks passed: no shared
  image hash with the C1 manifest, with the official test split or with the images of the fitted subjects; no dev
  subject among the fitted subjects.
- Of the 574 dev images, 493 were used only for early stopping and 81 were never read by the training; none was used
  to fit the weights. With only two dev subjects, the dev threshold will carry high uncertainty.
- NUAA's official protocol is not subject-disjoint between train and test: the two dev subjects also appear among the
  300 C1 test images (as attacks of session 03), with no shared image but a shared identity. The separation from the
  test set is by official split, acquisition session (fifth field of the NUAA file name: 01/02 in training, 03 in test;
  not the `session_id` column of `scores.csv`, which keeps the third field as in C1) and image hash.
- The `session_id` column of `scores.csv` is inherited unchanged from C1: it holds the third field of the NUAA file
  name, not the acquisition session (fifth field). The correction is planned for M2; it is not made here, so that
  `scores.csv` stays identical to the C1 one except for `run_id`.
- The source-dev threshold applies only to the 4 CNNs trained on NUAA with subject-wise validation; all other rows
  stay at 0.5. It is not computed yet: the selection code was verified on synthetic scores only (see the synthetic
  trial in step 3), and the dev scoring waits for authorisation.
