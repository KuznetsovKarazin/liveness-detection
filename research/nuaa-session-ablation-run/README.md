# NUAA training with and without session 03: LivenessNet run, arms B / A* / B', 4 folds x 3 seeds on the CPU

Run authorised by the coordinator on the C2 card on 5 November 2026, after the protocol proposal
(`delivery/sessionablation/PROPOSAL.md`): primary comparison A* against B, B' as secondary control, no A** and no
dev-mix; the three arms, the comparisons and the reading rule fixed in the configuration before the run; for every
primary metric the consistency condition must hold for all three seeds, each with its own bootstrap interval; test,
preprocessing, recipe and common dev as proposed; input checks first, then 36 trainings with an overall cap of 360
minutes; if the time is not enough, stop and report the run as incomplete, keeping the logs and the results available.

The question (proposal): does representing session 03 in training change the error on held-out subjects, at fixed
fitting subjects, total count, test, preprocessing and recipe? Diagnostic, not causal; no statistical test.
Pre-commitment: every rule below was fixed before any training; every number is reported whatever its value.

## What it does and what it does not do

- Trains 36 LivenessNet models on the CPU with the recipe of the folds run unchanged (the trainer of
  `nuaa_folds_run.py`, imported, not copied): 3 arms x 4 subject folds x seeds 42/43/44, each with the common dev of
  its fold; scores the held-out subjects of the fold; aggregates per arm and compares the arms on the same test frames.
- Extracts nothing: every frame comes from the training cache `nuaa_1500.npz` or from the frames extracted by the
  folds run (`nuaa_folds_frames.20261103-FOLDSRUN-e174812.npz`), both verified by SHA-256, the extracted frames also
  frame by frame. The detector and the frozen extractor are therefore not inputs of this run.
- Writes the dev threshold of every model before any test frame is scored and records the SHA-256 of the threshold
  file, read back, in every score row.
- Does not read `devmix_fold<k>.csv` (dev-mix excluded by the coordinator) and has no A** arm.
- Does not write in the accepted runs, in `data/` or in `export/`; the restricted outputs (scores, per-subject table,
  weights, thresholds, cards) go to the shared folder only.

## Run

- Run `20261106-SESSABLRUN-d999458` (clean commit `d999458` of the development repository, delivered configuration
  `4c2d29d13ded`), started 2026-11-06 08:41:20+0100, finished 11:31:29+0100: `PASS`, 36 of 36 trainings complete. Cap
  not reached: wall time 10,209 s (about 170 of 360 minutes); aggregation 2.0 s; peak resident memory 6,758 MB.
- Primary verdicts printed by the runner (A* - B, stratum 03), both `inconclusive`: AUROC (the sign differs across the
  seeds); ACER at the dev threshold (the bootstrap interval of a seed does not exclude 0 on the side of the
  difference). The secondary verdicts, the two secondary comparisons, reference A and every number are in
  `public/aggregate.md` and `public/aggregate.json`.
- Previous attempt `20261106-SESSABLRUN-3b4021f`: `FAIL` at model 7 of 36 on the false alarm of the earlier
  checkpoint check, as declared in the configuration (`previous_attempt`; see "Previous attempt" under the steps of
  `run`); its folder is kept and none of its models is used.
- Where the outcomes are: in the run folder, `public/` (`aggregate.json`, `aggregate.md`, `run_record.json` and its
  `SHA256SUMS`, SHA-256 `89cb1385...` recorded in `status.json`), `status.json` and `run.log`; in the package of the
  code repository, `public/` copied as `tables/`, `status.json` as `run/status.json` and `run.log` as
  `run/run.log.txt` (same bytes: `*.log` is ignored by git there); `restricted/` goes to the shared folder only. The
  reading of the results is in `CONCLUSION.md`.

## Arms, dev and test (fixed in the configuration)

| Arm | Fitting set (per fold) | Cells (01/bf, 01/att, 02/bf, 02/att, 03/bf, 03/att) | Frames |
|---|---|---|---|
| B (`arm_b`) | sessions 01-02 of the S12 subjects (fitting subjects with frames in sessions 01-02) | 350/350/350/350 | 1,400 |
| A* (`arm_a_star`) | sessions 01-03 of the same S12 subjects | 234/233/233/234/233/233 | 1,400 |
| B' (`arm_b_prime`) | sessions 01-02 of S12 at the per-cell counts of A* | 234/233/233/234 | 934 |

- Manifests: the cells run `20261104-SESSABL-5d89888` (`armB_fitting_fold<k>.csv`, `armAstar_fitting_fold<k>.csv`,
  `armBprime_fitting_fold<k>.csv`, roles `arm_b_fitting`, `arm_a_star_fitting` and `arm_b_prime_fitting_optional`, the
  role name written by the cells run).
- Common dev (`dev_fold<k>.csv`, role `dev_common`): sessions 01-02 of the dev-fold subjects (fold k mod 4 + 1), 100
  per cell, 400 frames, the same for the three arms; it drives early stopping, ReduceLROnPlateau, the restored epoch
  and the EER threshold.
- Test: `fold<k>_test.csv` of the fold schema run `20261103-FOLDS-67b7f3d`, every frame of the held-out subjects
  (3,242/3,114/3,439/2,819), strata 01, 02, 03 and all.
- Nesting, verified for every fold before the run: B = the session 01-02 rows of the schema fitting set; the session
  01-02 rows of A* and all of B' inside B; B' = the session 01-02 rows of A*; the subjects of A* and B' inside S12;
  the common dev = the session 01-02 rows of the schema dev (no session 03); fitting, dev and test subjects disjoint.
- Order of the 36 trainings (fixed, listed in the configuration): for fold 1..4 and seed 42, 43, 44, B then A* (the
  24 trainings of the primary comparison complete pair by pair); then B' for fold 1..4 and seed 42, 43, 44. Every
  model is trained, thresholded and scored before the next one starts.

## Inputs (every one pinned by SHA-256 in the configuration and verified by `check-inputs` before anything is written)

| Input | What is checked |
|---|---|
| The five M2 manifests and the twelve fold manifests of the schema run (functions of the folds run) | SHA-256 against the configuration and the restricted `SHA256SUMS` (`f3b30ae7...`), counts, roles disjoint by subject and key, every row equal to the official lists, `in_cache` flags, the test folds partition the 12,614 images |
| The schema configuration `delivery/folds/nuaa_subject_folds.json` | SHA-256 `d1d43c14...` (in the code and in the configuration); seeds, threshold criterion, recipe and number of folds equal to this run |
| The cells run `20261104-SESSABL-5d89888` | `status.json` PASS with its configuration and the two `SHA256SUMS` anchors (restricted `0904ee72...`, public `45d12575...`); its configuration (`2b86ef8e...`) with the same allocations as this configuration; the 16 manifests used: SHA-256 against the restricted `SHA256SUMS`, fold and role of every row, no duplicated key, counts per cell, every row equal to the official lists, `in_cache` flag; the nesting above |
| The folds run `20261103-FOLDSRUN-e174812` | `status.json` PASS with its configuration (`7aaf0be2...`) and `SHA256SUMS` anchors (public `11abd9e8...`, restricted `1204039d...`); `extracted_frames.csv` (restricted), `aggregate.json` (public, reference A) and `run_record.json` (public) equal to their `SHA256SUMS` rows; the rows of `extracted_frames.csv` equal to the official lists; the code imported from the folds run (`nuaa_folds_run.py` with its trainer and scorer, `m2_writers.py`, `path_guard.py`, `package_rules.py`) and the architecture with the SHA-256 recorded in `code_sha256` of that `run_record.json` |
| The training cache `data/processed/nuaa_1500.npz` | SHA-256 `dc286dc3...`, members and shapes from the `.npy` headers without pickle, labels and subjects equal to the manifests in the M2 sampling order (the position of every cache frame) |
| The extracted frames `data/processed/nuaa_folds_frames.20261103-FOLDSRUN-e174812.npz` | SHA-256 `c3fd9903...`, members `X` (uint8, 6,614 x 256 x 256 x 3) and `keys` without pickle, keys = the frames outside the cache in key order, the SHA-256 of every frame equal to `extracted_frames.csv` |
| Every row of every manifest used (cells, common dev, test) | found exactly once, in the cache or in the extracted frames, consistently with its `in_cache` flag (real inputs: 19,206 rows from the cache, 9,944 from the extracted frames) |
| The 12,614 original images | located, regular files, size and SHA-256 of their manifest row |
| The architecture `src/livedetection/src/architectures.py` | SHA-256 `a83c02b0...` at the checks and again just before loading it |
| The environment | OpenCV 4.11.0 and TensorFlow 2.15.0 installed (package metadata, TensorFlow not imported during the checks); `LIVENESSLAB_SILENT_FACE` and `LIVENESSLAB_LIVEDETECTION` unset |

`check-inputs` on the real inputs takes seconds, not minutes (4-11 s measured) and writes nothing. The GPU is not
checked here (TensorFlow is not imported): it is hidden, and the visible devices verified to be `["CPU"]`, by the
trainer before every training and by the scorer before every scoring, and the card of every model is refused unless
it records the CPU.

## Steps of `run` (one process, in this order; every step timed)

1. `check-inputs`: everything above.
2. For every model, in the fixed order: `train` (the folds-run trainer: Adam legacy 1e-3 with clipnorm 1, categorical
   crossentropy on one-hot labels, class weights balanced on the fitting set, augmentation on the fitting set only,
   shuffle(4096, seed), batch 16, at most 12 epochs, EarlyStopping on the val_loss of the common dev with patience 4 and
   restored best weights, ReduceLROnPlateau 0.5/2/1e-6, seed for random, numpy and TensorFlow, CPU only,
   `clear_session` before every model; checkpoint `.keras` saved through a temporary name; card with the history per
   epoch), `threshold` (common dev scored with the saved checkpoint, CPU, batch 16, BatchNorm in inference; val_loss
   recomputed from the output-layer logits against the card, tolerance 1e-3 absolute or 1 % relative, see below; EER threshold of the M1 protocol written to
   `<arm>_fold<k>_seed<s>_threshold.json`, read back, SHA-256 recorded), `score` (only now the test frames are
   assembled and scored; decisions at 0.5 and at the frozen dev threshold).
3. `aggregate`: see below; public files scanned for identifiers and forbidden text; checksums.

Check of the checkpoint (`threshold.val_loss_check` in the configuration). Keras 2.15 records the val_loss of an
output layer with softmax activation from its logits (softmax cross-entropy from logits, no clipping). The check
therefore recomputes it in the same way from the saved checkpoint: h = the input of the `output` layer (sub-model up
to that layer, CPU, batch 16, `training=False`), z = h @ W + b in float64 with the weights of that layer, val_loss =
mean over the dev frames of logsumexp(z) - z[y]. It must equal the recorded minimum within 1e-3 absolute or 1 %
relative, otherwise the run fails; the softmax of z must equal the scorer probabilities within 1e-4. The crossentropy
on probabilities limited to [1e-7, 1 - 1e-7] caps the loss of a frame at -ln 1e-7 = 16.118 and differs from the Keras
value as soon as one dev frame is misclassified with a larger loss: the card keeps it as `val_loss_recomputed_clipped`
(informative only) next to `val_loss_recomputed` (from the logits, compared).

Previous attempt (`previous_attempt` in the configuration): run `20261106-SESSABLRUN-3b4021f` (commit `3b4021f`,
started 2026-11-06 07:32:57+0100) ended FAIL at model 7 of 36 (`arm_b_fold2_seed42`) on a false alarm of the earlier
check, which compared the clipped crossentropy with the Keras value (one dev frame beyond -ln 1e-7: 0.1936 against
0.2308, with the checkpoint of minimum val_loss). Its six complete models are kept in that folder and not used; no
result was read beyond the training cards. This run starts from scratch with the corrected check; protocol, inputs,
recipe, order and cap are unchanged.

## Wall-time cap and outcomes

The cap (360 minutes from the start; elapsed time = the larger of the monotonic and the wall clock) is checked before
every training, at the end of every epoch, before every threshold and before every test scoring. Unlike the folds
run, exceeding it is not a failure: the model in progress is interrupted (neither saved nor evaluated: its weights,
threshold and dev scores, if any, are removed), no other training starts, and the run goes on to the aggregate of the
complete models, to the final writes and to `status.json` = `INCOMPLETE` with the lists of complete and missing
trainings and the step at which it stopped; logs and available results are kept. The aggregation and the final
writes run after the last model also when the cap has been exceeded: `run_record.json` and `status.json` record
`final_writes_after_cap`, true when the cap was already exceeded before the aggregation started (measured there,
whatever stopped the run). They take a few seconds, measured 2-5 s at real scale (dry run at real scale with a stub trainer, 5 November 2026: the 36 models, 113,526 test score rows, aggregation 2.0 s, aggregation and final writes 2.2 s, the public scan with the real identifiers and both `SHA256SUMS` passed).

Voluntary stop: `kill -USR1 <pid>` (the pid is in `status.json` while `RUNNING` and in the opening `== NUAA session ablation run` line of `run.log`) is
treated as the cap. The signal handler only records the request; the run stops at the next check (before a training,
at the end of an epoch, before a threshold or a test scoring, so at most about one epoch later), discards the model in
progress, aggregates the complete models and ends `INCOMPLETE` (exit code 2) with `stopped.cause` = `SIGUSR1`. A
request after the last check of the last model has no effect.

| Outcome | When | Exit code | Outputs |
|---|---|---|---|
| `PASS` | 36 of 36 trainings complete | 0 | all |
| `INCOMPLETE` | the cap or `SIGUSR1` stopped the run | 2 | the complete models, the aggregate over them (pooled values only for arm and seed with all four folds, comparisons only where both arms are pooled, median and range only with the three seeds; the other verdicts are `inconclusive`, reason "incomplete run") |
| `FAIL` | an error, a refused input, a refused public file, `SIGTERM` or `SIGHUP` | 1 | public folder and `per_subject.csv` removed, and the files of the model in progress; with complete models, their five restricted files kept with a restricted `SHA256SUMS` over them only, not aggregated, no verdict (`status.json`: `kept_complete_models`, note "public removed; restricted outputs of N complete models kept, not aggregated; no verdict"); with no complete model every output removed, as in the folds run; `status.json` with the reason and `run.log` remain |
| `FAIL` (interrupted) | Ctrl-C (`KeyboardInterrupt`) | 130 | as `FAIL` |

The order of the trainings puts the 24 trainings of the primary comparison first, so a run stopped by the cap during
B' still has the complete primary comparison.

## Metrics, comparisons and reading rule

Per arm, fold, seed and session stratum (01, 02, 03, all; NA where a class is absent): n, bona fide and attacks,
subjects, APCER, BPCER and ACER at 0.5 and at the dev threshold, AUROC (Mann-Whitney, ties 0.5). Pooled over the four
folds per arm and seed (every subject held out once; the decisions at the dev threshold use the threshold of the
fold of each frame), with a bootstrap over the held-out subjects; median and range over the seeds.

Comparisons (paired on the same test frames and the same seeds; nominal pairing, since batch order and augmentation
differ with the data): primary `a_star_minus_b` (A* - B); secondary `b_prime_minus_b` (the per-cell count of sessions
01-02) and `a_star_minus_b_prime` (the added session at equal per-cell count). Per comparison, fold, seed and stratum:
the difference of every metric. Pooled per comparison, seed and stratum: the difference with a paired bootstrap over
the subjects of the stratum.

Bootstrap: `default_rng(0)` restarted for every seed, comparison (or arm) and stratum; subjects in ascending order;
1,000 resamples `integers(0, n, n)`; the same indices for both arms of a comparison and for every metric; metrics on
the pooled frames of the resampled subjects; percentiles 2.5-97.5 with linear interpolation; a resample without a
class does not count for the metrics that need it; no interval with fewer than 5 subjects (the pooled strata hold 9
subjects in 01 and 02, 16 in 03 and all). The metrics of a resample are computed from per-subject counts and pairwise
Mann-Whitney counts weighted by the resampling multiplicities: this is algebraically identical to recomputing them on
the concatenated frames, and the suite checks the equality exactly (also against the per-arm bootstrap of the folds
run).

Reading rule (in the configuration, applied mechanically): for each comparison, stratum and metric, the pooled
difference is `consistent` only if it has the same non-zero sign for all three seeds and the paired bootstrap interval
of every seed excludes 0 on the side of that sign; otherwise `inconclusive` (also when a seed has no pooled value, no
difference or no interval). A consistent verdict states the direction (for example "consistent: A* higher AUROC than
B"). Primary: AUROC and ACER at the dev threshold in stratum 03 of A* - B; the same rule, labelled secondary, for the
other strata, the other metrics and the two secondary comparisons. Every verdict is reported, with its reason.

Reference A: the pooled values per seed and stratum of the folds run (with their bootstrap intervals), copied from its
public `aggregate.json` with the label "reference, different fitting frames and dev: indicative"; not compared with a
bootstrap.

## Budget

From the proposal (scaled from the folds run): the 24 trainings of B and A* (1,400 fitting + 400 dev frames) take
470.5-561.8 s each, with their scoring 213.8-237.4 min; the 12 trainings of B' 80.6-89.6 min; total 294.4-327.0 min
under the cap of 360 min, plus the checks and the loading of the frames (seconds, not minutes: 4-11 s measured). The
estimate holds on an idle machine. Memory: the cache and the extracted frames stay in memory as uint8 (about 2.5 GB),
as in the folds run; the peak resident memory is printed in `run.log` after every model (`peak RSS`). Dry run at real scale with a stub trainer (no TensorFlow): peak resident memory 4.1 GB (frames, assembled
sets and the aggregation of 36 models); TensorFlow adds its own memory during the real trainings.

## Limits

The session-03 bona fide cell of S12 comes from 1-2 subjects per fold; two dev cells rest on one subject; frames of
one subject and session are correlated (n counts frames); three seeds give a range; the four folds are not
independent; the dev without session 03 calibrates early stopping and the EER threshold on the session 01-02
distribution for every arm, which can penalise A* in stratum 03 at the dev threshold (AUROC, threshold-free, is the
cleaner reading); one dataset, one attack type (print).

## Commands (from the development repository root, the data present)

```bash
.venv/bin/python -I scripts/nuaa_session_ablation_run.py check-inputs
nohup caffeinate -dims .venv/bin/python -I scripts/nuaa_session_ablation_run.py run --i-am-authorized > logs/sessablrun-console.txt 2>&1 &
```

The first command verifies only and writes nothing. The run is started with `nohup` and `caffeinate -dims`, the
console redirected to a file under `logs/` (ignored by git: a file in the repository root would make the tree dirty
and the run id `-dev`), never through `| tee`. Under `nohup` SIGHUP stays ignored. To stop the run and keep an
aggregate of the complete models: `kill -USR1 <pid>` (`INCOMPLETE`, see above); `kill <pid>` (SIGTERM) stops it with
`status.json` = FAIL (complete models kept, no aggregate, no verdict). Sleep time counts towards the cap (wall clock).
`--out-root` (default `results/sessionablationrun`) is for tests and dry runs only.

Run folder `results/sessionablationrun/YYYYMMDD-SESSABLRUN-<commit>[-dev]` (`-dev` when `git status --porcelain`,
untracked files included, is not empty): `status.json` (RUNNING, then PASS, INCOMPLETE or FAIL; it records whether
the configuration is the delivered one, by SHA-256), `run.log` (one log per attempt, its SHA-256 in the final
`status.json`), `public/` (`aggregate.json`, `aggregate.md`, `run_record.json`, `SHA256SUMS`), `restricted/` (per
model `<arm>_fold<k>_seed<s>_card.json`, `_threshold.json`, `_dev_scores.csv`, `_test_scores.csv`,
`_LivenessNet.keras`; `per_subject.csv` with an `arm` column; `public_SHA256SUMS.sha256`; `SHA256SUMS` over the files
present). A folder that already holds a `status.json` is refused: every attempt keeps its outcome; to try again,
archive it first.

## Files of the package

| File | Content |
|---|---|
| `README.md` | this document |
| `livenesslab/delivery/sessionablationrun/nuaa_session_ablation_run.json` | the configuration: inputs, arms, dev, nesting, order of the trainings, recipe, threshold, metrics, comparisons, bootstrap, reading rule, reference A, budget, declarations |
| `livenesslab/scripts/nuaa_session_ablation_run.py` | the script (`check-inputs`, `run`) |
| `livenesslab/scripts/nuaa_folds_run.py` | the folds-run script, imported for its unchanged functions (manifests, cache, originals, environment, frame store, trainer, scorer, threshold, metrics, writers) |
| `livenesslab/scripts/check_nuaa_session_ablation_run.py` | 81 tests on a synthetic world (identifiers and structure unlike NUAA, fake trainer and scorer, injected clock); the only test on the real inputs is the read-only `check-inputs` |
| `livenesslab/scripts/check_nuaa_session_ablation_run_mutations.py` | 158 declared mutations of the script, each detected by the suite on a copy, plus a control and three declared redundant; one mutant at a time by default (`--jobs 1`, at most 2), never on the real data (`SESSABLRUN_INPUT_ROOT` = the copy, which holds the code and the configuration only, so the real-input test is skipped in the unmutated copy and in every mutant) |
| `livenesslab/scripts/check_nuaa_folds_run.py` | the suite of the folds run, which provides the base synthetic world and the fake TensorFlow |
| `livenesslab/scripts/m2_writers.py`, `path_guard.py`, `package_rules.py` | the shared modules, unchanged |
| `test_logs/` | `check_nuaa_session_ablation_run.txt` (macOS: suite and the real `check-inputs`), `check_nuaa_session_ablation_run_mutations.txt` (`--jobs 2`), `linux_package_no_data.txt` (python:3.12 container, package layout, no data, no TensorFlow: suite with the declared skip and mutation check); each log opens with the SHA-256 of the scripts tested and of the configuration |
| `tables/aggregate.json`, `tables/aggregate.md`, `tables/run_record.json`, `tables/SHA256SUMS` | the public outputs of run `20261106-SESSABLRUN-d999458` (its `public/`) |
| `run/status.json`, `run/run.log.txt` | the final status and the log (`run.log`, renamed) of that run |
| `CONCLUSION.md` | the reading of the results |
| `SHA256SUMS` | SHA-256 of every file of the package |

Dependencies: Python 3.11 or 3.12, NumPy, OpenCV 4.11.0 (its version is checked), TensorFlow 2.15 (training and
scoring, CPU only). The tests need NumPy only.

Expected test lines. `tests run: 81, failures: 0, errors: 0, skipped: 0` with the real inputs and the delivered
configuration present; without the data `skipped: 1` (declared: `real inputs`; `delivered configuration` only when the
configuration file is absent too); `RESULT: PASS`. Mutation check (every copy without data, so `skipped: 1` in every
line): `mutations killed: 158/158`, `control mutations survived: 4/4 (1 control, 3 declared redundant)`,
`RESULT: PASS`.
Linux container without data (`test_logs/linux_package_no_data.txt`): the suite passes with that one skip, the
mutation check passes, and `check-inputs` fails as expected (`RESULT: FAIL (check-inputs)`: no manifest in the container).
