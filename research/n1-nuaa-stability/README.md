# N1: NUAA stability (part 1: subject sensitivity of the M1 result; part 2: small LivenessNet training pilot)

Card N1. Part 1 (item 1): sensitivity of the accepted M1 result to the 16 test subjects, from existing scores only (no
training, no inference). Part 2 (item 2): one architecture (LivenessNet), three trainings with training seeds 42, 43
and 44, on the local cache `nuaa_1500.npz` with the M2 preprocessing and split (2,507 fitting / 493 validation / 3,000
test frames); split seed 42 for every run. This folder holds the code, the configuration saved before the runs, the
run reports, the publishable aggregate tables and the test logs. Per-image scores, per-subject tables, logs,
checkpoint cards and hashes are in the restricted area.


## Delivery status

- Restricted folders, uploaded on 26 October 2026 to the N1 folder created and shared by the coordinator (92 files;
  no images, cache or weights): `20261024-N1-subjects-a5fcbbc/` and `20261024-N1-LivenessNet-fb93496/` and, as first
  execution, `20261024-N1-subjects-2c44c64-dev/` and `20261024-N1-LivenessNet-2c44c64/` (with
  `20261024-N1-LivenessNet-2c44c64.SHA256SUMS` next to it).
- Part 1: run `20261024-N1-subjects-a5fcbbc` on 24 October 2026, from the clean commit `a5fcbbc` of the
  development repository (dirty tree: no). An earlier run from a modified tree (`20261024-N1-subjects-2c44c64-dev`)
  gave the same values; only run id, commit, date and tree state differ.
- Part 2: study `20261024-N1-LivenessNet-fb93496` on 24 October 2026, from the clean commit `fb93496` (pre-run checks
  PASS, three trainings on the CPU, CPU scoring, aggregate; recorded tree state clean, the nine recorded code SHA-256
  equal to the files of this folder). The tables below come from it.
- A first execution of part 2 (`20261024-N1-LivenessNet-2c44c64`, before the NumPy 2 fix of `n1_train.py`, modified
  tree) is kept unchanged in the restricted area. The re-run from the clean commit reproduced it bit for bit on this
  machine (CPU, same software): same weights SHA-256 for the three seeds, `scores_test.csv` identical except the
  `run_id` column, aggregate identical except the training wall times. Nothing is claimed beyond this machine and
  software.
- Images and cache stay local. Checkpoint cards and SHA-256 are in the restricted folders; the weights (`.h5`,
  `.keras`) are kept by the operator, within the already authorised scope, available to the coordinator on request,
  never uploaded publicly.

## What is delivered

Run reports (the group template `templates/run-report.md`, one per run): `run-report-subjects.md` (part 1,
`20261024-N1-subjects-a5fcbbc`) and `run-report-training.md` (part 2, `20261024-N1-LivenessNet-fb93496`). The two
run ids omit the seed of the `YYYYMMDD-task-seed-shortcommit` format: part 1 has no random component, part 2 holds
three training seeds in `seed42/`, `seed43/`, `seed44/` (stated in each report).

Part 1:
- `livenesslab/scripts/n1_subject_sensitivity.py` (`check-inputs`, `run`), its configuration
  `livenesslab/delivery/n1/subject_sensitivity/n1_subject_sensitivity.json` (at the default path of the script),
  `livenesslab/scripts/check_n1_subject_sensitivity.py` (tests) and `check_n1_subject_sensitivity_mutations.py`.
- `tables/subject_sensitivity/n1_subject_sensitivity_public.{json,md}`: the public aggregate (no subject identifiers).

Part 2:
- `n1_nuaa_stability.json`: the configuration, written before the runs (SHA-256 in `SHA256SUMS`).
- `livenesslab/scripts/n1_train.py`: subcommands `check`, `train`, `score`, `aggregate`.
- `livenesslab/scripts/n1_trainer.py`: copy of the `train` function of `train_cnn.py` (which is unchanged) with the
  training seed separate from the split seed, CPU only, and the score export moved to `n1_train.py score`.
- `n1_trainer_vs_train_cnn.diff`: the full difference from the current trainer. Functional changes: (1) training
  seed separate from the split seed (GroupShuffleSplit uses only the split seed); (2) hyperparameters read from the
  validated configuration, with the values of the card: 12 epochs instead of the command-line default 10 of
  `train_cnn.py`; batch 16, learning rate 0.001, patience 4, optimizer, loss, class weights, augmentation and
  scheduler as in `train_cnn.py`; (3) only
  `X_train`, `y_train`, `s_train` reach the trainer, no test evaluation, split by subject mandatory and checked by
  SHA-256; (4) CPU only, GPU hidden and checked; (5) card with per-epoch history and file hashes, existing outputs
  refused. The removed lines are image loading and caching (`build_arrays`, `preprocess`), the command line, the
  `models/weights` and `manifest.json` handling and the output-folder checks, moved to or replaced by `n1_train.py`.
  Reproducible with
  `diff -u --label research/m2-c2-pilot-impl/livenesslab/scripts/train_cnn.py --label research/n1-nuaa-stability/livenesslab/scripts/n1_trainer.py research/m2-c2-pilot-impl/livenesslab/scripts/train_cnn.py research/n1-nuaa-stability/livenesslab/scripts/n1_trainer.py`
  (run from the repository root; exit code 1 means "files differ").
- `livenesslab/scripts/check_n1_train.py` (tests on synthetic data, no TensorFlow) and
  `livenesslab/scripts/check_n1_mutations.py` (declared mutations of `n1_train.py` and `n1_trainer.py`).
- `livenesslab/scripts/n1_metal_relu_repro.py`: minimal reproducer of the GPU defect (synthetic only);
  `livenesslab/scripts/n1_metal_diagnostic.py`: the layer-by-layer measurement on LivenessNet with initial weights that
  first showed it; `livenesslab/scripts/n1_metal_checks.py`: `fit-steps` (synthetic: weights after 5 fit steps on CPU
  and GPU), `c1-rescore` (the C1 LivenessNet checkpoint scored on the CPU against the accepted C1 scores; restricted
  inputs, only aggregates printed) and `c1-linear` (C1 weights, random inputs: GPU `predict` against the CPU network
  with and without the `dense1` ReLU). See "Device" and "Finding on C1 LivenessNet".
- `tables/aggregate.json`, `tables/aggregate.md`: the aggregate table (no identifiers, no per-image rows).
- `test_logs/`: outputs of the test commands of both parts and of the three GPU scripts (reproducer, diagnostic, checks).
- Unchanged copies, byte-identical to those published in `research/pad-2026` (packages `m2-writers-maintenance`,
  `m1-metrics`, `m2-c2-pilot-impl`): `nuaa_manifests.py`, `select_threshold.py`, `path_guard.py`, `m2_writers.py`,
  `package_rules.py`, `src/tesi_app/{__init__,paths,evaluation,core}.py`, `requirements.txt`.

## Part 1: subject sensitivity of the M1 result

Inputs (restricted, verified by SHA-256 against the configuration and against the `SHA256SUMS` of their folders,
whose SHA-256 is also fixed): the C1 scores and manifest of the 300 NUAA test images (run
`20260928-C1-seed42-a80f6f9`) and `scores_test_dev_threshold.csv`, `threshold.json`, the accepted table and the dev
manifest of the M1 run `20261003-M1-seed42-e67ecc7`. For each of the four NUAA CNNs: per subject, samples, false
accepts and false rejects at 0.5 and at the frozen M1 threshold; ΔACER = ACER_M1 − ACER_0.5 on the full set (which must
reproduce the accepted M1 table exactly, also the block without the two dev subjects) and with one subject at a time
excluded from the computation only (16 values): range, sign changes, denominators. Exact fractions; decision rule
attack if `score_attack > threshold` (tie = bona fide); a class absent gives NA, never 0; no recalibration on the test.

Run `20261024-N1-subjects-a5fcbbc` (`tables/subject_sensitivity/`):

| analyzer | M1 threshold | n bona fide | n attack | FA@0.5 | FR@0.5 | ACER@0.5 | FA@M1 | FR@M1 | ACER@M1 | ΔACER full | LOSO ΔACER min | LOSO ΔACER max | sign changes | of which opposite sign | zero | NA | n bona fide after exclusion | n attack after exclusion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| livenessnet__nuaa | 0.181055 | 150 | 150 | 24 | 109 | 0.443333 | 5 | 118 | 0.410000 | -0.033333 | -0.051750 | -0.009568 | 0/16 | 0 | 0 | 0 | 128–150 | 138–150 |
| attacknet_v1__nuaa | 0.329237 | 150 | 150 | 42 | 40 | 0.273333 | 37 | 44 | 0.270000 | -0.003333 | -0.010165 | 0.002542 | 2/16 | 2 | 0 | 0 | 128–150 | 138–150 |
| attacknet_v2_1__nuaa | 0.870329 | 150 | 150 | 3 | 115 | 0.393333 | 11 | 91 | 0.340000 | -0.053333 | -0.067033 | -0.019845 | 0/16 | 0 | 0 | 0 | 128–150 | 138–150 |
| attacknet_v2_2__nuaa | 0.696933 | 150 | 150 | 22 | 118 | 0.466667 | 38 | 112 | 0.500000 | 0.033333 | 0.019910 | 0.048762 | 0/16 | 0 | 0 | 0 | 128–150 | 138–150 |

Class composition of the subjects (same for every CNN): 8 with both classes, 7 with attacks only, 1 with bona fide
only. Full-set values reproduced exactly from the accepted M1 table (56 + 56 cells).

Reading (descriptive, no statistical test): for livenessnet, attacknet_v2_1 and attacknet_v2_2 the sign of ΔACER does
not change when any single subject is excluded, although its size varies; for attacknet_v1 the full ΔACER (−1/300) is
smaller in magnitude than the largest change from excluding a single subject, and the sign changes in 2 of 16
exclusions (two subjects with attacks only).

Limits of part 1: the test images include persons already present in training (NUAA is not subject-disjoint between
its official train and test splits), so the result does not show generalisation; 16 subjects with unbalanced class
composition; the 16 leave-one-out values are not independent and are not a confidence interval; this is a sensitivity
analysis of existing results, not cross-validation and not a new independent validation; the thresholds were not
recalibrated; the 300 C1 images are verified against the official source (M2), the verification is not extended
beyond them.

Commands (from `livenesslab/`, with the restricted C1 and M1 run folders at the paths of the configuration):
`python scripts/n1_subject_sensitivity.py check-inputs`, then `python scripts/n1_subject_sensitivity.py run`.

## Part 2: small LivenessNet training pilot

### Configuration (saved before the runs)

Frozen in `n1_train.py` (`FROZEN`, types included) and in the configuration file; any other value is refused:
LivenessNet, NUAA, official protocol; training seeds `[42, 43, 44]`; split seed 42
(`GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` on the subject array); at most 12 epochs, batch 16,
initial learning rate 0.001, early-stopping patience 4 on `val_loss`; checkpoint = weights of the epoch with minimum
`val_loss` (`restore_best_weights`, and `set_weights` when the 12 epochs end first); optimizer
`legacy.Adam(clipnorm=1.0)`, categorical cross-entropy, balanced class weights, the same augmentation and
`ReduceLROnPlateau(factor 0.5, patience 2, min_lr 1e-6)` as the current trainer; device CPU. Scores on CPU, inference
mode, batch 16. Threshold 0.5 on the attack probability, attack if `score_attack > 0.5` (tie = bona fide).
The data block (cache SHA-256 `dc286dc3…177920`, the six M2 manifests with SHA-256 and counts, the four previous
LivenessNet NUAA checkpoints to refuse) is required for a non-trial run (`CARD_DATA`).

The training seed sets `random`, NumPy and TensorFlow seeds (weight initialisation, augmentation) and the `tf.data`
shuffle seed; the split seed is used only by the split. The trainer receives only `X_train`, `y_train`, `s_train`: the
test frames never enter the training process.

### Pre-run checks (`check`, repeated at the start of `train` and `score`)

No TensorFlow, no weights, no detector. Each must pass, otherwise the command stops with `RESULT: FAIL` and keeps its log:
cache file name and SHA-256; cache members, dtypes and shapes read from the `.npy` headers (no pickled objects);
SHA-256, rows, classes and subjects of the six manifests (`official_train`, `official_test`, `cnn_train`, `cnn_es`,
`cnn_test`, `c1`), strict row validation of `nuaa_manifests.py`, same fields for rows sharing a key across manifests;
sampling order = `spread(sorted names, 1500)` per class (bona fide first) on the official lists, equal to
`cnn_train ∪ cnn_es` and to `cnn_test`; `y_*` and `s_*` equal element by element to label and subject of the manifest
rows in that order; the split recomputed with the split seed equal to `cnn_train` / `cnn_es` by key and by file
SHA-256; the C1 subset inside the 3,000 test frames; no duplicated derived frame and none shared between fitting,
validation and test.

Result on the real data (`check.log`, restricted): all PASS; 2,507 fitting frames from 7 subjects, 493 validation
frames from 2 subjects; the 16 test subjects include the 7 fitting and the 2 validation subjects; 0 duplicated or
shared frames.

### Device

With TensorFlow 2.15 and tensorflow-metal, the ReLU after a Dense layer (MatMul + BiasAdd + ReLU) is dropped on the GPU
in graph mode (`tf.function`, hence Keras `fit`, `evaluate` and `predict`): the output equals the pre-activation.
Eager mode on the GPU and the CPU (eager and graph) are correct; only `disable_meta_optimizer` fixes it (the individual
grappler options tried do not); the AttackNets are unaffected (their dense layers use tanh). In LivenessNet this is
`dense1`, after `Flatten`. Reproducer on synthetic data: `scripts/n1_metal_relu_repro.py`
(`test_logs/n1_metal_relu_repro.txt`: error 3.59 in graph mode on the GPU with ReLU, about 1e-7 without activation, with
tanh, in eager mode and on the CPU; `MISMATCH`, exit code 1; with `--disable-meta-optimizer` `NO MISMATCH`, exit code
0). On LivenessNet with initial weights (`test_logs/n1_metal_diagnostic.txt`): up to 0.31 on the `dense1` activations
and 0.089 on the probabilities (initial weights with a fixed seed).

With the GPU the `val_loss` that drives early stopping would not be the one of the weights scored on the CPU, so the
pilot trains on the CPU only (GPU hidden and checked), with the optimizer, augmentation and scheduler unchanged. `score`
recomputes the `val_loss` of the restored weights on the CPU and refuses a difference above both 1e-3 and 1 % of the
recorded minimum.

### Commands

From `livenesslab/` of this folder; in the development repository, from its root with the paths below. With the
restricted inputs (M2 manifests folder, the cache) and the reference architectures (`LIVENESSLAB_LIVEDETECTION` pointing to the folder with
`src/architectures.py`); the array form works in bash and zsh:

```
C=(--config ../n1_nuaa_stability.json --manifests <M2 manifests folder> --cache <path>/nuaa_1500.npz --study-dir <outside the repository>/YYYYMMDD-N1-LivenessNet-<shortcommit>)
python scripts/n1_train.py check "${C[@]}"
for s in 42 43 44; do
  python scripts/n1_train.py train "${C[@]}" --train-seed $s --i-am-authorized
  python scripts/n1_train.py score "${C[@]}" --train-seed $s --i-am-authorized
done
python scripts/n1_train.py aggregate "${C[@]}"
```

In the development repository the configuration is `delivery/n1/n1_nuaa_stability.json` and the study folder is
under `results/n1/` (the runs of this report).

Every command ends with `RESULT: PASS (<command>)` or `RESULT: FAIL (<command>)` (exit code 1). The study folder must
be new at `check`; inside the development repository only `results/n1/` is accepted, inside the group repository
(the tree with the accepted packages) no output folder is accepted. A non-trial study requires the data of the card and
the current commit in its name; the working-tree state and the SHA-256 of every code file are recorded at `check`
and compared at every later step.

### Results (study `20261024-N1-LivenessNet-fb93496`, `tables/aggregate.md`)

Split seed 42 for every run (2,507 fitting / 493 validation frames, M2 manifests); training seeds 42, 43, 44.
Threshold 0.5 on the attack probability, attack if score_attack > 0.5 (tie = bona fide). Scores on CPU. No seed
selection: every seed of the configuration is reported. Median and range (minimum-maximum) over the three seeds.

#### Test (3,000 frames of nuaa_cnn_test)

| metric | seed 42 | seed 43 | seed 44 | median | range |
|---|---|---|---|---|---|
| APCER | 0.1067 | 0.2607 | 0.0480 | 0.1067 | 0.0480-0.2607 |
| BPCER | 0.7700 | 0.7680 | 0.7540 | 0.7680 | 0.7540-0.7700 |
| ACER | 0.4383 | 0.5143 | 0.4010 | 0.4383 | 0.4010-0.5143 |
| AUROC | 0.3485 | 0.3754 | 0.4582 | 0.3754 | 0.3485-0.4582 |
| accepted attacks / attacks | 160/1500 | 391/1500 | 72/1500 | | |
| rejected bona fide / bona fide | 1155/1500 | 1152/1500 | 1131/1500 | | |

#### C1 subset (secondary analysis) (300 frames of nuaa_c1_v2, inside the test)

| metric | seed 42 | seed 43 | seed 44 | median | range |
|---|---|---|---|---|---|
| APCER | 0.0867 | 0.2533 | 0.0400 | 0.0867 | 0.0400-0.2533 |
| BPCER | 0.7800 | 0.7800 | 0.7667 | 0.7800 | 0.7667-0.7800 |
| ACER | 0.4333 | 0.5167 | 0.4033 | 0.4333 | 0.4033-0.5167 |
| AUROC | 0.3374 | 0.3714 | 0.4641 | 0.3714 | 0.3374-0.4641 |
| accepted attacks / attacks | 13/150 | 38/150 | 6/150 | | |
| rejected bona fide / bona fide | 117/150 | 117/150 | 115/150 | | |

#### Training

| | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| epochs run (max 12) | 9 | 12 | 12 |
| restored epoch (minimum val_loss) | 5 | 12 | 8 |
| minimum val_loss | 0.000898 | 0.012337 | 0.000542 |
| val_loss recomputed on CPU | 0.000899 | 0.012337 | 0.000542 |
| training wall time (s) | 496.0 | 671.0 | 671.4 |
| weights SHA-256 (.h5) | `c63a858107b6…` | `34bac22cafd3…` | `41c128484ff7…` |

NA = not computable (class absent). Values are rounded to four decimals; aggregate.json holds the exact values.

Checkpoint cards (restricted, `seed<N>/checkpoints/LivenessNet__nuaa.json`): restored by early stopping for seeds 42
and 44, at the end of the 12 epochs for seed 43. Full SHA-256 of the `.h5` and `.keras` files in `tables/aggregate.json`.
`score` recomputed the minimum `val_loss` on the CPU with a largest difference of 5.96e-08 over the three seeds
(refused above both 1e-3 and 1 % of the recorded minimum).

#### Interpretation

- The three runs reach near-zero validation loss but behave badly on the test: BPCER is 0.754-0.770 (about three
  bona fide frames in four rejected), AUROC is below 0.5 for every seed (0.348-0.458), so on the test the attack
  score ranks bona fide frames above attacks more often than the reverse.
- The variation between seeds is mostly in APCER (0.048-0.261) and therefore ACER (0.401-0.514); BPCER changes by
  less than 0.02. The validation curves are unstable (single epochs with `val_loss` above 9 in every run), and the
  epoch of minimum `val_loss` ranges from 5 to 12.
- The C1 subset (300 of the 3,000 test frames) gives values within about 0.02 of the full test for every seed and
  metric; being a subset of the test, it is not independent evidence.
- Session shift, a possible explanation to be verified: fitting and validation frames come from sessions 01-02, the
  whole test from session 03. The poor test results are not limited to new people: on the 1,325 test frames of the 7
  fitting subjects BPCER is 0.687 for every seed and AUROC 0.319-0.453 (on the other 1,675 frames BPCER 0.800-0.827,
  AUROC 0.412-0.471).
- Weak validation, a possible explanation to be verified: 2 subjects and 493 frames from the training sessions;
  `val_loss` is very unstable, and `val_accuracy` is 0.6856 (= 338/493, accuracy matching an all-attack baseline) in
  5 of the 33 epochs. The chosen epoch may therefore be noisy, which could contribute to the spread of APCER between
  seeds.
- What these runs measure is the instability of this training recipe on NUAA with this validation, not a property of
  LivenessNet in general; session shift and weak validation are possible explanations to be verified, not tested
  causes.

#### Historical comparison (descriptive only)

The earlier LivenessNet NUAA checkpoint (C1, 12 September, last-epoch weights), scored through the same CPU path as
these runs (`n1_metal_checks.py c1-rescore`, `test_logs/n1_metal_checks.txt`), gives AUROC 0.3910 on the 3,000 test
frames and 0.3813 on the C1 300, which agrees closely with the accepted C1 scores (AUROC 0.3782; Pearson correlation
0.9995, largest difference 0.0515; the C1 scores come from the application pipeline on the images, these from the
cache frames). Its card reports AUC 0.6023 on the 3,000 frames: that value was computed with Keras `predict` (graph
mode) and is not compared here (see "Finding on C1 LivenessNet"). Checkpoint rule and training device differ from
the new runs: no conclusion is drawn from the difference, and no earlier threshold is used.

### Finding on C1 LivenessNet

Established, with the scripts and logs of this folder:
- the defect described in "Device": ReLU after MatMul + BiasAdd dropped on the GPU in graph mode; eager GPU and CPU
  correct; only `disable_meta_optimizer` fixes it; tanh, used by the AttackNet dense layers, is not affected
  (`n1_metal_relu_repro.py`);
- with the C1 LivenessNet weights, Keras `predict` on the GPU equals the same network on the CPU with `dense1` linear
  (largest difference 2.4e-7, against 0.29 from the correct network; `n1_metal_checks.py c1-linear`, random inputs);
- `fit` on the GPU does not train the same network as on the CPU: on a small synthetic Dense(relu) network the
  weights differ by 0.10 after 5 steps, 1.2e-7 with the meta-optimizer disabled (`n1_metal_checks.py fit-steps`);
- the C1 LivenessNet checkpoint scored on the CPU agrees closely with the accepted C1 scores (Pearson 0.9995, AUROC
  0.3813 against 0.3782; `n1_metal_checks.py c1-rescore`). From reading the code, the C1 scores were computed by the
  application in eager mode, one image at a time, so they are not affected by the defect.

Not established:
- the device of the computation behind the card metrics (AUC 0.6023 on the 3,000) is not recorded: they were
  computed with Keras `predict` (graph mode); if, as is likely but not recorded, on the GPU, they describe the network
  with `dense1` effectively linear and are not comparable with the C1 table or with these runs;
- the training device of the C1 LivenessNet checkpoint is not recorded (GPU likely); if it was the GPU, its weights
  were optimised with `dense1` effectively linear, which is not a correct LivenessNet training;
- the effect on the real-data metrics of C1 LivenessNet was not quantified;
- the cause inside tensorflow-metal (a plugin optimisation pass is a hypothesis, not proven).

### Same defect in accepted code

From reading the code and from the minimal reproduction (`n1_metal_relu_repro.py`); not run on C2 data (the C2 pilot
has not been run on real data):
- `train_cnn.py` (accepted, unchanged) sets only `remapping: False` and does not hide the GPU: on an Apple-silicon GPU
  with tensorflow-metal, `fit` of LivenessNet runs in graph mode with `dense1` effectively linear, so early stopping
  and the restored epoch rest on the wrong `val_loss`; its test evaluation uses `model.predict` (graph mode), so the
  card metrics of a GPU run describe that variant.
- `research/m2-c2-pilot-impl/` `c2_pilot.py` lists LivenessNet in `ARCHS` and trains through `train_cnn.train`, so
  the same applies to its LivenessNet training; only its scoring (`tf_scorer`) hides the GPU and runs on the CPU.
- The AttackNets (dense layers with tanh) are not affected.

N1 does not change these files, as the card requires; a separate task is proposed (for example, train on the CPU or
hide the GPU in `train_cnn.py` and the C2 runner, with a test based on the reproducer).

## Restricted area (not in this repository)

Restricted folders in the project document workspace (see "Delivery status"):
`02_Experiments/N1/20261024-N1-subjects-a5fcbbc/` and `02_Experiments/N1/20261024-N1-LivenessNet-fb93496/`, plus the
first execution, kept as it was:
`02_Experiments/N1/20261024-N1-subjects-2c44c64-dev/` (with its own `SHA256SUMS` in `public/` and `restricted/`,
written by the script) and `02_Experiments/N1/20261024-N1-LivenessNet-2c44c64/`, whose checksum file
`20261024-N1-LivenessNet-2c44c64.SHA256SUMS` (weights excluded) was produced at upload time, outside the original
folder, and uploaded next to it (never committed into it). Weights excluded (see "Delivery status").

Part 1, run folder `20261024-N1-subjects-a5fcbbc/` (with its own `SHA256SUMS` files, written by the script): `status.json`;
`restricted/`: `n1_per_image.csv`, `n1_per_subject.csv`, `n1_loso.csv`, `n1_subject_sensitivity_restricted.json`,
`n1_subject_sensitivity_restricted.md`, `public_SHA256SUMS.sha256` (anchor of the public folder), `SHA256SUMS`;
`public/`: `n1_subject_sensitivity_public.json`, `n1_subject_sensitivity_public.md`, `SHA256SUMS`.

Part 2, study folder `20261024-N1-LivenessNet-fb93496/` (with a `SHA256SUMS` of every file except the weights,
34 files, `668e3fec…`): `check.json`, `check.log`, `status.json`, `aggregate.log`, `aggregate.json`,
`aggregate.md`, `per_subject_all_seeds.csv`; per seed `seed<N>/`: `status.json`, `train.log`, `train.json` (card
SHA-256, weights and Keras SHA-256, epochs, restored epoch, device), `checkpoints/LivenessNet__nuaa.json` (card with
the per-epoch history), `score.log`, `scores_test.csv` (3,000 rows: sample key, subject, session, label,
`score_attack` with exact round trip, decision at 0.5, C1 flag, file and checkpoint SHA-256), `per_subject.csv` (per
subject and set: counts, rejected bona fide, accepted attacks, BPCER/APCER/ACER, `NA` when a class is absent),
`metrics.json`, `score.json`; and, local only, `checkpoints/LivenessNet__nuaa.{h5,keras}` (their SHA-256 are in the
cards, in `train.json` and in `tables/aggregate.json`).

Per-image records and `research/templates/score-schema.json`: `scores_test.csv` uses its own columns (`run_id`,
`train_seed`, `sample_id`, `subject_id`, `session_id`, `label`, `score_attack`, `prediction`, `in_c1_subset`,
`file_sha256`, `checkpoint_sha256`, `config_sha256`), a declared deviation from the schema. Mapping to the schema
fields: `run_id`, `sample_id` (the manifest key), `subject_id`, `session_id`, `label`, `score_attack`, `prediction`,
`checkpoint_sha256` and `config_sha256` are the same; `dataset_id` = `nuaa`, `split` = `test`, `video_id` = null,
`attack_type` = `print` for label 1 and null for label 0, `modality` = `RGB` (constant for every row, not written);
`threshold` = 0.5, fixed a priori (not frozen from source dev; not written; `prediction` = 1 iff `score_attack >
0.5`). Extra columns: `train_seed`, `in_c1_subset`, `file_sha256` (as `file_sha256` in the C2 records). A
schema-conformant export of the same records can be added as a follow-up if wanted.

## Tests

From `research/n1-nuaa-stability/livenesslab/` (`python` is Python 3.11 or later with the packages of
`requirements.txt`; data, weights, results folders and TensorFlow are not needed). Part 2:

```
python scripts/check_n1_train.py
python scripts/check_n1_mutations.py
```

Expected last lines (also in `test_logs/`):

```
tests run: 91, failures: 0, errors: 0, skipped: 0
RESULT: PASS
```

```
unmutated copy (after): PASS
mutations killed: 72/72
control mutations survived: 1/1
RESULT: PASS
```

Every mutation runs in a fresh copy of the code, removed afterwards; the unmutated suite must pass in a fresh copy
before and after the mutations, and a control mutation (a comment only) must survive. A mutation counts as killed
only when the suite runs and its summary line reports failures or errors (same number of tests as the unmutated
suite); an import crash, a syntax error or a missing summary line is a harness error, never a killed mutation. The
copy includes the delivered configuration, read from `delivery/n1/` in the development repository or from the
package root (`../n1_nuaa_stability.json`) here, so that `test_delivered_configuration` runs instead of being skipped.

Dependencies: NumPy (1.x or 2.x: only public functions, checked by a test) and scikit-learn (OpenCV optional:
without it the suite registers an empty `cv2` module and prints a `NOTE` line). No data, no TensorFlow (the scripts
that need it are run with TensorFlow blocked and must print `SKIP: TensorFlow not installed (missing input)`, exit
code 2). The only possible skip is `test_delivered_configuration` when the configuration
is not next to the code.

Part 1:

```
python scripts/check_n1_subject_sensitivity.py
python scripts/check_n1_subject_sensitivity_mutations.py
```

Expected last lines from this package layout (the two skips are the tests on the restricted C1 and M1 run folders,
reported with the reason `missing input`):

```
tests run: 47, failures: 0, errors: 0, skipped: 2
RESULT: PASS
```

In the development repository, with the restricted folders present: `tests run: 47, failures: 0, errors: 0, skipped: 0`.

The mutation harness accepts skips on the unmutated copy only when every skip reason is a declared missing input
(`MISSING_INPUT` of `check_n1_subject_sensitivity.py`) and reports them; a mutation counts as killed only by failures or
errors of the tests that run, never by a skip (a mutated copy that skips other tests without failing is invalid). Expected
last lines, both from this package layout (2 declared skips on the unmutated copy) and in the development repository (no
skip) (`test_logs/check_n1_subject_sensitivity_mutations.txt`, both runs):

```
mutations killed: 40/40
control mutations survived: 1/1
RESULT: PASS
```

GPU reproducer (needs TensorFlow; a measurement, not a test): `python scripts/n1_metal_relu_repro.py` prints
`MISMATCH: ReLU after MatMul+BiasAdd dropped on GPU in graph mode` (exit code 1) where the defect is present,
`NO MISMATCH` (exit code 0) otherwise, `NO GPU: nothing to compare` (exit code 2) without a GPU and
`SKIP: TensorFlow not installed (missing input)` (exit code 2) without TensorFlow. The same SKIP line and exit code
apply to `n1_metal_diagnostic.py` and to `n1_metal_checks.py fit-steps` / `c1-linear`.

## Limits

- The new seed 42 is not an exact replica of the earlier C1 checkpoint: the current trainer restores the weights of the
  minimum `val_loss`, the earlier LivenessNet kept the last epoch; its training device is not recorded (GPU likely),
  these were trained on the CPU. The three new runs are comparable with each other; the historical comparison is
  descriptive only.
- C1 subset: the 300 C1 images are verified against the official source (M2); this verification is not extended to
  the whole cache, whose frames are linked to the manifests by sampling order and by the label and subject arrays.
- The test contains people already present in training (all 16 test subjects; 7 of them in fitting, 2 in validation):
  NUAA official train and test differ by session, not by subject.
- Validation has only 2 subjects, from the training sessions.
- Three seeds give a range, not a confidence interval.
- CPU training: seeds are fixed; the two executions on this machine gave identical weights, but parallel reductions
  on the CPU are not guaranteed to be deterministic on other machines or software versions. The reference
  `architectures.py` (`a83c02b0…`) is recorded but is not part of this folder.
- From this package layout the two part 1 tests on the restricted C1 and M1 run folders are skipped (missing input):
  here the agreement with the accepted M1 table is checked only on the synthetic table of the tests; the check on the
  real scores ran in the development repository (`test_logs/check_n1_subject_sensitivity_mutations.txt` has both runs).

## Files and integrity

`SHA256SUMS` lists the 37 files of `research/n1-nuaa-stability/` other than itself (38 files in all); check with
`shasum -a 256 -c SHA256SUMS` from this folder (37 lines `OK`). No file of the folder is ignored by the `.gitignore`
of the group repository (`git check-ignore --no-index`, exit code 1); the aggregate tables are in `tables/` because
`results/` is ignored there.
