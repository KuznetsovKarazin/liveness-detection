# PROTOCOL.md §4, the 2x2 BN re-estimation on fitting/dev: choices fixed before the trial

Written with the implementation (`scripts/pad_arch_bn_factorial.py`), before any real checkpoint, frame or score was
read by it. The whole text of this note, Section 6 included, is fixed by its SHA-256 when the trial is launched: it is
committed privately before the trial, the SHA-256 is written in `analysis.sha256` of the configuration, and `trial`
refuses to start while that field is null. Sections 1-5 and 7-8 are fixed choices. Section 6 is the analysis proposed
for the registration: it becomes binding at the public registration, before any test score of the new states is
produced; any change to it between the trial and the registration is declared in the registration with its diff. Every
later change to any section is listed as a deviation, with a reason that does not depend on development outcomes.

Sources: `delivery/pad-arch-score-followup/PROTOCOL.md` §4 (lines 33-52), the coordinator's request of 22 November 2026
(point 3: verify the 2x2 combination on fitting/dev with the saved checkpoints before any new test score; register code
and configuration after the trial with the measured time; both options off in the final evaluation), the plan of 23
November and the independent reviews of the plan and of the code. Where plan and protocol differ, the protocol
prevails.

## 1. Scope and sequence

- **This trial: `--scope pilot`.** Two checkpoints: patience 4, fold 1, seed 42, one per network (the first unit of
  `plan/run_order.csv`). The rule is neutral and was fixed before any outcome; it is not chosen on results.
- **Sequence.** The trial; then the public registration of code, configuration and measured time (wall and CPU); then
  the complete runs on fitting/dev; then the test, with both options off in the final evaluation.
- **Complete runs.** The 48 checkpoints in two processes, `--scope p4-24` then `--scope p12-24`. The Monte Carlo
  comparison between views 1-4 and 5-8 is made at patience 4 only. If the coordinator asks for a different scope, the
  change is registered as a declared modification; it is not chosen from the numbers of the trial.

| Run | Units | Monte Carlo halves | Explicit duplicates | Cap |
|---|---|---|---|---|
| `pilot` (this trial) | p4, fold 1, seed 42, both networks | yes | every unit | 200 min |
| `p4-24` | the 24 checkpoints at patience 4 | yes | first unit of each network | 1,500 min |
| `p12-24` | the 24 checkpoints at patience 12 | no | first unit of each network | 820 min |

- **Development metrics known before the registration.** The `pilot` reports its cell metrics publicly
  (`report_cell_metrics` true): the development values of its two checkpoints are therefore known before the public
  registration of §6. They are in-sample (§4) and no choice of this note depends on them; any change between the trial
  and the registration is declared with its diff.
- **Launch conditions.** Every run is started with `delivery/archbn2x2/launch.zsh` from the root of the repository, in a
  zsh with `setopt nobgnice`, the power adapter connected, powermode 0 and no other heavy process. The script runs
  `check-inputs` first and then `trial --attempt <k> --i-am-authorized` under `caffeinate -dims`, and propagates its
  exit code. It does not start the trial if the nice value is not 0, if `pmset -g batt` does not report the power
  adapter, if powermode is not 0 or low power mode is on, if a views folder of an earlier trial is left in the temporary
  folder, if the current folder is not the root of a git work tree, if `git status --porcelain` fails or the work tree
  is not clean, or if `check-inputs` fails. A launch that is refused leaves a separate "not launched" conditions file
  and does not use the attempt. The same holds when the trial exits with an error without printing its start line, which
  it prints right after creating its run folder and its first status.json: the conditions file and the console output
  take the "not launched" names, and the trial's exit code is propagated. A trial that has printed its start line uses
  the attempt, whatever its outcome. An error between the creation of the run folder and the start line would leave that
  folder with its conditions under the "not launched" names; the trial then requires the next attempt number. The script
  writes, outside the repository, a conditions file with, before and after the command: date, load, CPU usage, power
  source, powermode and memory pressure; and once: system version, powermode and lowpowermode (when reported), the nice
  value, the number of `TF_NUM_` variables set, whether the temporary folder is the default one of macOS, views folders
  left and free space, the commit and the number of lines of `git status --porcelain`, the output of `check-inputs`, the
  command and its exit code. The file holds no path. The measured wall and CPU times of the trial depend on these
  conditions and are reported with them.

## 2. Choices

| # | Choice | Value fixed |
|---|---|---|
| S1 | Scope | §1 |
| S2 | Views and seeds | 8 views per fitting frame per stochastic cell, view seeds 101-108 (view v uses seed 100 + v); halves 101-104 and 105-108 (PROTOCOL r.46, r.48) |
| S3 | Augmentation | explicit, stateless re-implementation of the training Sequential (`pad_arch_comparison.trial_trainer`): same seven operations and order, ranges derived from the recipe, fill mode reflect, bilinear interpolation, the same clips; applied after one division by 255; **contrast factor drawn per frame** (Keras 2.15 draws one per batch of 16), §3 |
| S4 | Random draws (choice of the author) | NumPy Philox4x64 keyed by the first 128 bits (big-endian) of SHA-256(`pad-arch-bn-factorial/1\|<domain>\|<view seed>\|<SHA-256 hex of the frame key>\|<operation or Dropout layer>`), domain `augmentation` or `dropout`; one generator per draw; §3 |
| S5 | Dropout during estimation | only the Dropout layers upstream of a BatchNormalization (`dropout1`, `dropout2`, rate 0.25 read from the checkpoint and compared with the configuration; `noise_shape` None checked); `dropout_dense` never active; every layer, BN included, called with `training=False` (PROTOCOL r.46) |
| S6 | Views storage | `numpy.memmap` in a new folder `archbn2x2-views-*` of the temporary folder of the process (TMPDIR), never inside the repository or the input root (checked), after a free-space check (views plus 2 GB); removed at the end of each fold and at every stop, signals included; a folder left by an earlier trial stops the start; peak resident memory limit 18 GB (FAIL above) |
| S7 | Duplicates | reuse rule of §4; explicit check as in §1; a duplicate outside the reference tolerance stops the command (FAIL) |
| S8 | Monte Carlo | two separate re-estimations per stochastic cell (views 101-104, 105-108), each scored on the dev; pilot and `p4-24` only |
| S9 | Tolerances | E00 statistics within 1e-4 relative + 1e-6 absolute (bit identity reported); E00 and baseline dev scores within 1e-5, 0 discordant decisions at 0.5 and at the saved dev EER threshold, recomputed EER threshold within 1e-5 of the saved one; consistency 1e-4 + 1e-6; softmax 1e-4; duplicates as E00 |
| S10 | Determinism | E11 repeated on the first LivenessNet unit: statistics (float64 and float32) and dev probabilities bit-identical, and the first batch of every view regenerated from the draws byte-identical to the stored view; otherwise FAIL |
| S11 | Cap | about twice the synthetic estimate (§1, §8), checked before every layer, pass, dev scoring and view batch; INCOMPLETE at the cap |
| S12 | Metrics | threshold 0.5 only for every effect (attack if score > 0.5, a tie is bona fide); dev EER threshold reported descriptively |

## 3. Augmentation and draws: what is the original and what differs

- Same functions as the Keras layers in training mode: `transform`, `get_rotation_matrix`, `get_zoom_matrix`,
  `get_translation_matrix` (Keras 2.15 module pinned by SHA-256), the batched flip formula
  `f * reverse(x) + (1 - f) * x` with `f = round(u)`, `x + delta` then clip to [0, 1] for brightness,
  `tf.image.adjust_contrast` then clip to [0, 255] for contrast, `x + stddev * z` for the noise. The suite forces the
  Keras layers of the recipe to the same parameters and checks that each operation, and the whole Sequential, give
  bit-identical images.
- The ranges are derived from the recipe (rotation ±0.03·2π, zoom [0.88, 1.12] for height and width, translation ±0.06
  of the side, brightness ±0.15·(1 − 0), contrast [0.8, 1.2], noise standard deviation 0.02); the recipe is compared with
  the pinned source of `pad_arch_comparison.py` read with the AST, and the suite compares minimum and maximum of 4,000
  draws (within 1% of each bound) and the noise standard deviation (1.2 million values, within 1%) with the Keras layers
  built from that source.
- The clip of the contrast is not inert on [0, 1] data: its lower bound acts when the factor exceeds 1. The upper bound
  (255) never acts. The noise is not clipped.
- Differences from training: the random draws, and the contrast factor per frame instead of per batch. The marginal
  distribution of every parameter is the same; in training the composition of a batch was random (`shuffle(4096)`), so
  a per-batch factor has no reproducible counterpart; the BN moments are sums over frames, so only the correlation
  between frames of a batch changes.
- Why NumPy Philox with a SHA-256 key (choice of the author): a counter-based generator keyed by (domain, view, frame,
  operation) makes every draw a pure function of these four values, independent of batch size, frame order and every
  other draw (tested), and of the TensorFlow version; it is also cheaper (one eager TensorFlow scalar draw cost about
  2.7 ms; six draws for each of 11,200 frame-views of a fold would add about 3 minutes per fold). The stateful Keras
  generators, and the GaussianNoise layer, which has no resettable generator, cannot give per-frame draws. The values
  drawn therefore differ from those that Keras would draw; the operations applied to them are the same.
- The AST check verifies that the training frames are divided by 255 once and that the Sequential is called with
  `training=True`; it does not verify that this call is the one applied to the fitting dataset (it is, in the pinned
  source, but the check does not prove it).
- No per-frame seed, parameter, view or mask is written to any output: the frame key identifies subject and file.

## 4. Reuse, cells and controls

- E00 is the existing estimator (`nuaa_bn_recalibration.recalibrate`, unchanged, one clean pass). E10, E01, E11 use eight
  views each. A layer whose set of active stochastic operations upstream is the same as in the reference cell
  (E01 -> E00, E11 -> E10) takes the reference statistics: `bn1_1` and `bn1_2` of LivenessNet in E01 and E11, every layer
  of ResNet-18 in E01 and E11 (a fully reusable cell is the reference cell). PROTOCOL r.46: identical repeated views may
  be computed once; ResNet-18 duplicates must match within the reference tolerance and may be reused.
- Explicit duplicates: E01 on eight identical clean copies against E00 (the duplicate of PROTOCOL r.46), and E11 on the
  same eight augmented views against E10. For ResNet-18 the explicit E11 is the same computation as E10 (same views, same
  path): it is a determinism check of ResNet-18, reported under that meaning; in the complete runs it is made for the
  first ResNet-18 unit only.
- Order per checkpoint: checkpoint SHA-256 and architecture; topology (LivenessNet chain, or no Dropout/noise layer in
  ResNet-18); EMA facts (fitting frames and batch giving the declared updates per epoch); ordinary dev against
  `_dev_ordinary.csv`; E00 against `_bn_statistics.json` (re-estimated and stored moments) and `_dev_re-estimated.csv`;
  for LivenessNet E00 through the layer-by-layer chain; E10, E01, E11 with the checks of PR #27 (only the moving tensors
  changed; a multi-output consistency pass over the same views and masks; counts views x frames x H x W; softmax); finite
  views within [-0.5, 1.7]; mask zero frequency within 5 binomial standard deviations of the rate (counted once per layer
  per re-estimation); explicit duplicates; Monte Carlo halves; repetition; at the end the original model and the
  checkpoint file unchanged.
- Identity with the parent: it rests on the SHA-256 of the `.keras` file, checked before loading and at the end, and on
  the byte-for-byte comparison of every non-moving tensor of each copy with the loaded checkpoint. The hash of the
  trainable weights is recorded only: no parent hash of that kind exists to compare it with.
- Any failed control stops the command as FAIL with `failed_check` (check and values) in `status.json`; no stochastic
  cell is reported. Cap or SIGUSR1: INCOMPLETE, complete units kept. SIGTERM, SIGHUP, memory, disk space: FAIL. No
  automatic resume.
- The development frames were also the early-stopping set of the checkpoints: every dev number is in-sample.

## 5. Protection of the test and what this code does not contain

- Files read: listed in `TEST_READ` and in the configuration; of each ARCHRUN unit only `.keras`, `_card.json`,
  `_bn_statistics.json`, `_dev_ordinary.csv`, `_dev_re-estimated.csv` (anything else is refused before opening).
- An audit hook on every Python `open` compares the opened file by identity (`os.stat` of the file and of every folder
  that contains it, along the path as written and along the path resolved by the system, so that symbolic links to files
  and `..` after a symbolic link are followed; names of both paths in lower case) and refuses: a relative path given to
  `os.open` (its audit event does not carry `dir_fd`), always (the views folder is removed file by file with `os.remove` and `os.rmdir`, which open nothing); `results/archeval` of the repository and of the input root, and any
  `--forbid-outside` folder; names matching `fold[0-9]*_test.csv`, `*_test_scores.csv`, `*scores_test*`,
  `*test*_scores*`; any file with more than one hard link outside the Python installation and the system folders; under
  `results/` and `data/`, any file that is not in a declared input or in the run folder; outside the repository and the
  input root, anything other than the Python installation, the system folders, the temporary folder, the Keras
  configuration folder, the folder of this command and of the configuration. Everything under an allowed folder is
  allowed, so at the start of `check-inputs` and `trial` the command refuses a temporary folder (TMPDIR) inside the
  repository, the input root or the folder that contains the repository (the default temporary folder of macOS, under
  `/var/folders`, is accepted), and any allowed folder outside the repository (TMPDIR, the folders of the configuration,
  of the command and of the output root) that contains the repository, the input root or the home folder. The folder of
  the score analyses, outside the repository, is therefore refused without any flag: the opening would arise only if TMPDIR
  were that folder or contained it, and both cases are refused at the start (TMPDIR inside the folder that contains the
  repository, or containing the repository or the home folder). With
  the official output root, `--input-root` and a configuration outside the repository are refused. A refused open that a
  library swallowed is caught at the next step (FAIL, `failed_check` read_guard) and again at the end of the trial.
- Limits: a hard link is recognised by its link count, so a file linked into a non-exempt place is refused but a copy is
  not (a copy is a separate file with its own content); reads done outside Python are not audited: TensorFlow's C++ code
  (only the declared checkpoints are given to it) and `git` in `repo_state` (`git status` may read tracked files,
  including under `results/archeval`, to refresh its index; no content reaches the process). The opens of TensorFlow,
  Keras and their libraries in the real environment pass through this guard only in `trial` (`check-inputs` does not
  import TensorFlow): a wrongly refused library file would show only there, early in the log.
- **No test evaluation command exists in this code.** It will be written after the trial, registered with the rest, and
  will refuse to start without a registration record (pull request, merged commit, SHA-256 of the registered
  configuration, status and SHA-256 of the trial in PASS).
- The frame files (cache and extracted frames) are hashed and loaded whole, as in every earlier run, and hold frames of
  the held-out subjects; only fitting and development frames of arm B are assembled.

## 6. Analysis proposed for the registration (fixed with this note at the trial, binding at the registration, before any test score)

All mechanistic comparisons are exploratory (PROTOCOL r.48). Primary subset: patience 4, the twelve fold-seed units of
each network; patience 12 descriptive. Threshold 0.5 only. Percentage points (pp) throughout.

### 6.1 Effects within a unit

For unit u (fold, seed), network, metric m in {ACER, APCER, BPCER} at 0.5 on the held-out test of the unit's fold:
augmentation A_u = m(E10) − m(E00); dropout D_u = m(E01) − m(E00); interaction I_u = m(E11) − m(E10) − m(E01) + m(E00).
Aggregate: unweighted mean over the twelve units (equal fold-seed weight), per network; all twelve values reported.
For ResNet-18 E01 = E00 and E11 = E10 by construction (no dropout), so D_u = I_u = 0.

### 6.2 Architecture gaps

G_u(cell) = m_ResNet-18(cell) − m_LivenessNet(cell) on the same unit; the same differences of the gaps:
G_u(E10) − G_u(E00), G_u(E01) − G_u(E00) and G_u(E11) − G_u(E10) − G_u(E01) + G_u(E00); unweighted mean over the twelve
units.

### 6.3 Groups and intervals

- **Group policy** (PROTOCOL r.52), taken as registered for the score analysis: the working grouping "by manifest
  identifier" of PROTOCOL §1, that is the 16 values of `subject_id` of the score rows (the folder of the NUAA list line,
  equal to the ID field of the file name), each in the test of exactly one fold (`delivery/pad-arch-score-followup/
  NOTES.md`, section "The four original NUAA lists"; `score_analysis.py`: identifiers in sorted order, an identifier that
  spans folds stops the script). The two identifiers present in one class only and only in session 3 (472 genuine
  frames in the test of fold 1, 468 photographic frames in the test of fold 3) are kept as two clusters of their own, as
  in the registered analysis. Their relation to the maintainers' answer follows PROTOCOL §1 to the letter: if the
  grouping is corrected before the test evaluation of the new states, the intervals are given for the as-run grouping
  and for the corrected one, both labelled; if no answer is available, the identifier-level grouping and its limitation
  remain.
- **Draws** (as `score_analysis.bootstrap`): `numpy.random.default_rng(20261122)`, then
  `rng.multinomial(16, [1/16] * 16, size=10000)`, the pooled identifier weights in the sorted order of the identifiers.
  The within-fold stratified draws that the same generator produces next in the registered script are not used here.
  The same 10,000 draws are used for both networks, every cell, every effect, both classes and all seeds.
- **Statistic of a draw**: for each unit and cell, the class-specific frame errors and denominators summed over the
  identifiers of the unit's fold with their weights; the unit metric recomputed at 0.5; the effect per unit; the
  unweighted mean over the twelve units (PROTOCOL §3).
- **Valid draw, metric by metric.** ACER: the weighted denominators of both classes are positive in every one of the
  twelve units (the registered rule, `valid = (d > 0).all(...)` in `score_analysis.bootstrap`). APCER: the weighted
  attack denominator is positive in every unit. BPCER: the weighted bona fide denominator is positive in every unit. The
  registered script defines validity for ACER only; the APCER and BPCER rules are the corresponding simplest variant
  (choice of the author). Invalid draws are counted and reported, never replaced by zero.
- **Interval**: 95% percentile interval of the valid draws (`numpy.quantile` at 0.025 and 0.975, default method as in the
  registered script). No BCa, no test of significance.

### 6.4 Monte Carlo comparison

- **Where it is computed: development only** (PROTOCOL r.48: "views 1-4 versus 5-8 on development"), from the effects
  per view half of the trial and of `p4-24` (`effects.csv`, rows `101-104` and `105-108`; E00 is the same single clean
  estimate in both halves); the command writes the quantities below in `montecarlo.csv`. It never uses test data and
  does not depend on any test selection.
- **Difference.** For unit u, network, metric m and effect X in {A, D, I}: X_u^(h) is the effect with the stochastic cells
  re-estimated on view half h; H_u(X, m) = X_u^(101-104) − X_u^(105-108) (signed, pp). The interaction has its own H_u,
  computed from its own half-effects; no maximum is taken anywhere.
- **Scale of a single unit.** With the two halves independent, Var(H) = 2 Var(X^(4 views)) and
  Var(X^(8 views)) ≈ Var(X^(4 views)) / 2, so the standard deviation of an eight-view effect is SD(H) / 2. Statistic:
  s_MC(X, m) = sqrt(mean over the units of H_u²) / 2, over the units of the run and network (twelve at patience 4; one in
  the pilot, descriptive only). The relation Var ∝ 1/views holds to first order only: the metric at 0.5 is a count of
  decisions, not a smooth function of the moments.
- **Scale of the mean over the units.** The units are not independent with respect to the views: the draws depend only
  on the frame and the view, so the three units of a fold share views and masks, and different folds share fitting
  frames (a frame in the fitting set of several folds gets the same draws in each). No reduction by sqrt(12), nor by
  sqrt(4) at the level of the fold, is applied. For the mean over the units the command reports (i) the difference
  between the halves of the mean itself, H̄ = mean over the units of H_u, with |H̄| / 2, and (ii) s_MC without any
  reduction, as the prudent scale, because the shared frames correlate the units positively.
- **Comparison.** For each unit, |X_u| next to s_MC with the ratio |X_u| / s_MC; for the mean over the units, |mean
  effect| next to s_MC (not reduced) and to |H̄| / 2, with the ratio |mean effect| / s_MC. An effect, of a unit or of
  the mean, with magnitude at most 2 × s_MC is described as within the view-sampling variability; the ratio is always
  reported with the label. If s_MC = 0 (no variability observed between the halves), the ratio is NA, and the label is NA
  for a non-zero effect (within for a zero effect). On the dev (trial and complete runs) both terms come from the same partition. For the test
  effects of the final evaluation the scale is the development scale: this is a comparison across partitions, reported
  as such and descriptive only.

### 6.5 Figure

Per network, the within-unit changes A_u, D_u and E11 − E00 in ACER at 0.5 (test) against the restored epoch of the
checkpoint (p4 filled, p12 open markers), with 0.99^(88e) on a secondary axis; one panel per network. Per BN layer, the
pre/post moment discrepancies (median variance ratio and median standardized mean shift against the stored moments)
for each cell.

### 6.6 Exclusions, one class and ties

- No exclusion. A unit failing a control stops the command; the failure is reported; a new attempt repeats every unit
  in a new folder; no unit is dropped from the aggregates. A unit that cannot be reproduced at all is reported as
  missing and the aggregates are given on the available units with the count of missing units.
- A tie at 0.5 is bona fide. One-class predictions are kept and counted. APCER and BPCER are defined when the class is
  present (both classes are present in the test of every fold); a missing class gives NA and the unit is counted as not
  evaluable for that metric.

## 7. Tests (summary)

The plan's tests T1-T14 are implemented, with one correction: the plan's "all-one masks equal dropout off" is wrong for
inverted dropout (kept values are scaled by 1/(1 − rate)) and is replaced by: all-one masks give x/(1 − rate) at the
active layers and leave the layers upstream of any dropout unchanged. Every control that stops the command has a test
that makes it fire inside a synthetic trial; the read guard is tested with upper-case names, hard and symbolic links,
symbolic links to files, `..` after a symbolic link, `os.open` with `dir_fd`, undeclared files under `results/` and
`data/`, files outside the allowed places, a temporary folder inside or around the roots, and a refused open swallowed
by a library; the draw ranges and the noise
standard deviation are tested against the Keras layers built from the source. The launcher is tested with fake `pmset` and `ps` (no
dependence on the power source or the nice value of the machine running the suite). Without TensorFlow and Keras the
tests that need them are skipped with a declared reason. Every mutation of the mutation check declares the test that
must kill it, and only that test is run on the mutated copy.

## 8. Time and memory (synthetic measurement)

`pad_arch_bn_factorial.py measure`: random uint8 arrays 256x256x3 and random weights (no data read), batch 16, CPU,
10 + 10 threads; minimum of two calls after a warm-up.

- Per batch of 16: one augmented view batch 0.027 s (1.7 ms per frame-view); LivenessNet evaluation 0.039 s,
  estimator passes 0.195 s summed over its 5 layers, masks 7.9 ms (`dropout1`) and 4.0 ms (`dropout2`); ResNet-18
  evaluation 0.124 s, estimator passes 1.42 s summed over its 20 layers. The measurement itself took 8.2 s wall and
  55.2 s CPU.
- Cross-check: E00 per checkpoint from these costs is 27 s (LivenessNet) and 143 s (ResNet-18), against 28-29 s and
  148-152 s of `phase_2_seconds` in ARCHRUN, which include the same re-estimation plus loading and writing: the estimates
  below are optimistic by about 10% against the recorded times.
- Estimates: `pilot` 6,009 s (1.67 h; ResNet-18 4,419 s, of which 1,995 s are the explicit duplicates, the E11 one being
  a determinism check); `p4-24` 46,716 s (12.98 h); `p12-24` 25,534 s (7.09 h). The caps of §1 are about twice these.
- Memory: the measuring process peaked at 1.3 GB; the trial adds the frame store (2.5 GB, loaded whole), the fitting part
  (0.3 GB) and the views of a fold (8.8 GB as memmap pages, which count as resident when read): about 14 GB expected,
  limit 18 GB. The temporary folder needs about 8.8 GB free plus the 2 GB margin.
- These are estimates, optimistic by about 10% against the recorded ARCHRUN times (minimum of two calls, no sustained
  load); the trial measures and reports the real wall and CPU time per trial, unit, cell and layer.
