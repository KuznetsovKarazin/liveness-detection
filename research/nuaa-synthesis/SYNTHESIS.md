# NUAA 2026 — Methods/Results synthesis, from C1 to the BatchNormalization control, and towards the article

Synthesis of the NUAA results run between 28 September and 17 November 2026 and accepted in `research/pad-2026` (29
September – 4 November) or in the later packages of this repository (folds run, PR #18; comparison with N1, delivered
with the update of 4 November; session ablation run, PR #21; degenerate trainings, PR #22; patience verification, PR
#23; constant learning-rate control, PR #25; BatchNormalization statistics on the dev, PR #27) or delivered in PR #28
(the same statistics on the test), written on the coordinator's request of 2 November. Every number comes from the
public aggregate, comparison table, per-epoch table, summary table or run record of the run named next to it, or from
the note of PR #22 (`research/nuaa-degenerate-trainings/NOTE.md`) where the text says so (§3.7 also quotes, with their
source, values computed read-only from the restricted files of the folds run; §6 also cites the public READMEs of the
CASIA-FASD manifests (`research/m2-rgb-manifests/`) and of the C2 pilot protocol (`research/m2-c2-pilot-protocol/`),
the manifest aggregate, and the publications it names); the unified table
(`tables/summary_table.md`; compact form below) and the annexes are rebuilt by `nuaa_synthesis.py` from those files and
`check` fails if a cell differs. No inference and no training; §3.7 also reports read-only descriptive calculations on
the existing restricted scores, which `check` does not verify. Revised on 3 November after the coordinator's review
(three precisions: the subject × session cell already measured by N2, the dev threshold, the provenance of the external
models; labels of the generated tables; checksums).

Revised on 4 November on the coordinator's request of the same day, after the merge of the subject-held-out run (PR
#18), with no new training and no other model:

- the run in Methods (§2, "Subject-held-out run");
- its results in the unified table, every fold and seed kept: 12 fold × seed rows, 3 rows pooled over the four folds and
  3 rows of the fold models on the 3,000 N1 test frames, with the dev threshold next to 0.5 and the bootstrap over
  subjects (39 rows, from 21); the annexes `tables/folds_run_table.md` (every session stratum) and
  `tables/n1_same_ids_table.md` (the comparison with N1);
- §3.5, the results of the run by seed, stratum and fold;
- §3.6, a descriptive comparison with N1 on the same 3,000 frames (run `20261104-FOLDSCOMPARE-3b3bcca`), from the
  existing scores and the thresholds already frozen;
- §3.7, a short analysis of fold 1 seed 43 and of fold 4 from the logs and tables already available, observations kept
  apart from possible explanations;
- §4, a fourth reading (subjects never seen, sessions represented); §5 rewritten as the state of the experiment.

Revised on 10 November on the coordinator's request of the same day, after the merge of the patience verification (PR
#23), with no new training and no other model:

- the session ablation run, the analysis of its degenerate trainings and the verification of the one agreed change
  (`early_stopping_patience` 4 → 12) in Methods (§2, "Session ablation run", "Degenerate trainings", "Patience
  verification");
- both runs in the unified table: 18 rows pooled over the four folds, one per arm, seed and run, stratum all, at 0.5 and
  at the dev threshold, with the bootstrap over subjects (57 rows, from 39); the annexes `tables/patience_table.md` (the
  36 models with patience 4 and 12 side by side: train and dev, then the test) and `tables/session_ablation_table.md`
  (every arm, seed and session stratum of both runs, and the 84 verdicts of each run side by side); the compact form
  `tables/patience_table_compact.md`, embedded in §3.9;
- §3.8, the session ablation run with patience 4: pooled results per arm and seed, the primary verdicts, the 84
  verdicts, the 10 degenerate trainings;
- §3.9, the same 36 models with patience 4 and 12, with all 36 models in a compact table, read in two blocks: training
  and dev first, then the test, reported separately and not used to choose or judge the recipe; every model included,
  with the unfavourable facts listed in (i) and (ii);
- §4, a fifth reading and its limits; §5 rewritten as the state on 10 November with one proposed control, to be agreed
  before any training.

Revised on 11 November on the coordinator's request of the same day, after the merge of the constant learning-rate
control (PR #25, merge `a1ca91f`), with no new training, no inference, no new seed and no other model:

- the control in Methods (§2, "Constant learning-rate control");
- its run in the unified table: 9 rows SESSABL-P12C pooled over the four folds, one per arm and seed, stratum all, at
  0.5 and at the dev threshold, with the bootstrap over subjects (66 rows, from 57); the annexes
  `tables/three_regimes_table.md` (the 36 models in the three regimes side by side: train and dev, then the test) and
  `tables/session_ablation_table_constant_lr.md` (every arm, seed and session stratum of the control run, and the 84
  verdicts of the three runs side by side), a separate annex so that `tables/session_ablation_table.md` stays as it was;
  the compact form `tables/three_regimes_table_compact.md`, embedded in §3.10;
- §3.10, the same 36 models in the three regimes (p4: patience 4; p12: patience 12 with the learning-rate reduction;
  p12c: patience 12 at a constant rate), every model included, read in two blocks: training and dev first, by recovery,
  stability and selected checkpoint, then the test, reported separately and never used to choose or judge the recipe; a
  short conclusion;
- §1 and §4, the question and the fifth reading extended to the three regimes; §5 rewritten as the state on 11 November.

Revised on 17 November on the coordinator's request of 16 November, after the merge of the BatchNormalization control
on the dev (PR #27, merge `c234064`) and the run of the same re-estimation on the test (PR #28, run
`20261117-BNTEST-6245b10`, a post hoc diagnostic extension on a test already analysed, not an independent validation),
with no new training and no other model:

- the two runs in Methods (§2, "BatchNormalization statistics, dev and test") and in Results (§3.11), with the compact
  table `tables/bn_recalibration_table_compact.md` (the 36 checkpoints, dev and test side by side, and the paired
  summaries of the test) and the annex `tables/bn_recalibration_table.md` (every checkpoint, scope and summary, with the
  sources and their SHA-256); the unified table gets no row (§3.11 says why) and the 15 earlier tables are unchanged;
- §1 extended; §4 rewritten as two lists, what the results show (each item with its scope) and the hypotheses still
  open, every unfavourable result kept; §5 rewritten as the state on 17 November; a new §6, "Towards the article".

Corrected on 17 November after the coordinator's review of the same day (C2 card): in §6 (c) the state of the M2
manifests (the 21 restricted CSV are received and verified; the canonical deposit and the final approval of M2 and the
CASIA-FASD agreement with decision D-4 stay open), and in §5 the two deviations of the test run recorded as such. No
number, table or reading changed.

## 1. Question

Restricted to NUAA, the group's question reads: how reliably do detectors trained on NUAA (and one trained elsewhere)
keep their discrimination and their error rates at the chosen threshold when the capture session changes (sessions 01–02
→ 03) or the dataset changes (five datasets → NUAA), once splits, metrics and preprocessing are under control. C1 fixed
the starting point, M1 the metric module and the source-dev threshold, M2 the manifests and overlap rules, N1 the seed
and subject sensitivity, N2 the session check and the historical combined model, the crop check the one documented
preprocessing difference, the subject-held-out run the cell no earlier run measured: subjects never seen, with every
session represented in training; the session ablation run whether representing session 03 in training changes that error
at fixed fitting subjects and total count, the patience verification whether the recipe trains stably on those 36
models, the constant learning-rate control whether the recovery seen with the longer patience still occurs without
the learning-rate reduction, and the BatchNormalization control what the ten degenerate patience-4 checkpoints hold:
whether re-estimating their stored normalisation statistics on the fitting set, every trained parameter unchanged, lets
them meet the recovery criterion on the dev (PR #27), and what the same re-estimation does on the test (PR #28).

## 2. Methods

**Data and units.** NUAA Photograph Imposter Database (Tan et al., ECCV 2010), public mirror
`akahana/anti-spoofing-nuaaaa` (archive SHA-256 `c0780c6f…`; accepted M2 package `20261006-M2-seed42-36c677a`; counts
read here from the public aggregate regenerated at `5fcc1f8`, PR #6). The official split is a *session* split: training
= sessions 01–02 of 9 subjects (1,743 bona fide + 1,748 print attacks), test = session 03 of 16 subjects (3,362 +
5,761); the 9 training subjects reappear in the test, 7 subjects are test-only. Sampled sets, versioned as manifests
with per-image SHA-256 (15 overlap rules, PASS: no image shared between any training-side and test-side manifest by key
or content): `cnn_train` 2,507 frames of 7 subjects (fitting), `cnn_es` 493 frames of the 2 other training subjects
(early stopping), `m1_dev` 574 = all official training frames of those 2 subjects, `cnn_test` 3,000 (1,500 + 1,500, one
every k over the official lists), `c1` 300 (150 + 150, same rule, inside `cnn_test`), N2 target 491 = official training
list minus fitting minus early stopping (243 + 248, 9 subjects; 81 of them are in `m1_dev`). Unit = image frame; frames
of one subject and session are correlated; no video exists.

**Preprocessing.** RetinaFace (Haar fallback, whole frame without a face), square crop with margin 0.15, 256×256
`INTER_AREA`, BGR→RGB, the reference enhancement of the historical code, /255. Two extractor versions: `cache_0ac7143`
built the training cache of 12 September (6,000/6,000 cache frames reproduced bit for bit, N2) and `c1_a80f6f9`, the
application module since 24 September, used for the C1 scores (RetinaFace box 1 px narrower and 1 px shorter). The
combined model keeps its historical chain (whole image or face crop, Lanczos 256, enhancement, [−1, 1]).

**Models and training.** Four reference CNNs (LivenessNet, AttackNet V1, V2.1, V2.2; `architectures.py` of this
repository) trained on 12 September on `cnn_train` with early stopping on `cnn_es` (Adam 1e-3, batch 16, ≤ 12 epochs,
patience 4; saved weights: minimum `val_loss` epoch for the AttackNets, last epoch for LivenessNet, which reached the
limit; split seed 42, no global seed; tensorflow-metal installed, training device not recorded, GPU likely). N1
established that on the Metal GPU the `fit` path of LivenessNet drops the ReLU after the dense layer in graph mode: if
the C1 LivenessNet checkpoint was trained on the GPU, it was trained with that layer effectively linear (the AttackNets
are unaffected). N1 retrained LivenessNet three times on the CPU (training seeds 42, 43, 44; split seed 42; same
hyper-parameters, minimum-`val_loss` weights restored); every later scoring runs on the CPU. The combined model is the
coordinator's AttackNet V2.2 checkpoint (SHA-256 `3dd0c853…`), trained per the historical code on 3DMAD, CSMAD,
Replay-Attack, MSSpoof and Our Dataset (NUAA is not among the training datasets named in that code, so there is no
NUAA training in our experiment; the checkpoint was trained by the coordinator's group and the provenance of its
training images is not verified here), loaded as saved.

**Thresholds and metrics.** Score = probability of attack (label 0 bona fide, 1 attack); attack if score > t, tie =
bona fide. t = 0.5 fixed a priori in C1, N1, N2 and the crop check; in M1 t = the EER threshold of `m1_dev`, frozen
before any test score was read. APCER = accepted attacks / attacks, BPCER = rejected bona fide / bona fide, ACER = their
mean, with denominators; AUROC threshold-free (`auc` in the C1 table, `test AUC` in M1: the same statistic); an absent
class gives NA. The metric module was validated in M1 on edge cases (ties, constant scores, missing class).

**Subject-held-out run (folds run).** Protocol: the fold schema approved by the coordinator (PR #16, run
`20261103-FOLDS-67b7f3d`, configuration SHA-256 `d1d43c14…`); run `20261103-FOLDSRUN-e174812` (configuration
`7aaf0be2…`, the delivered one, clean commit; accepted with PR #18, merge `96d3fc1`). The 16 subjects are dealt into 4
folds of 4 by a fixed stratified rule (frames in sessions 01–02; bona fide frames in session 03). For held-out fold k
the dev fold is fold k mod 4 + 1 and the fitting folds are the other two: fitting = 2,100 frames of 8 subjects, dev =
600 frames of 4 subjects, 350 and 100 per class × session cell drawn with the deterministic `spread` rule of M2, all
three sessions; test = every frame of the 4 held-out subjects in every session (3,242, 3,114, 3,439 and 2,819 frames:
the 12,614 official frames, each subject held out once). Frames: the 6,000 cache frames, reproduced bit for bit from the
originals with the frozen cache extractor (`face.py` 0ac7143; the N2 check repeated), and the other 6,614 frames
extracted once with the same extractor and chain. Model and recipe: LivenessNet with the N1 recipe unchanged (CPU; Adam
1e-3 with clipnorm 1.0, batch 16, ≤ 12 epochs, early stopping on the dev `val_loss` with patience 4 and the best weights
restored, learning-rate reduction on plateau, balanced class weights, augmentation on the fitting frames only), training
seeds 42, 43, 44: 12 trainings of 702–838 s each (model cards), 9,703 s for the whole run against a cap of 6 hours.
Thresholds: 0.5, and the EER threshold of the dev fold only (M1 criterion), written and hashed before any test frame was
scored. Metrics per fold, seed and session stratum (01, 02, 03, all; NA where a class is absent) and pooled over the
four folds per seed (every subject held out once; four models and four dev thresholds); uncertainty: bootstrap over
subjects on the pooled strata (1,000 resamples with replacement, seed 0, percentiles 2.5–97.5, no interval below 5
subjects). Pre-commitment: every rule was fixed before any training and every number is reported whatever its value.
Diagnostic design, not a benchmark and not a complete causal isolation: it mixes the three sessions in training and does
not preserve the official NUAA protocol.

**Session ablation run.** Protocol proposed on 4 November (`research/nuaa-session-ablation/PROPOSAL.md`, PR #20) and
authorised on 5 November with A\* − B as the primary comparison and B' as the secondary control; run
`20261106-SESSABLRUN-d999458` (configuration `4c2d29d1…`, clean commit; accepted with PR #21, merge `67f641c`).
Question: does representing session 03 in training change the error on held-out subjects, at fixed fitting subjects,
total count, test, preprocessing and recipe? Same four subject folds as the folds run; for held-out fold k the fitting
subjects are the subjects of the two fitting folds with frames in sessions 01–02 (4, 5, 5 and 4 per fold), the same for
every arm. Arms: B, sessions 01–02 of those subjects, 350 frames per class × session cell, 1,400 frames; A\*, sessions
01–03 of the same subjects, 234/233/233/234/233/233 per cell, 1,400 frames; B', sessions 01–02 at the per-cell counts of
A\*, 934 frames (it separates the per-cell count of sessions 01–02 from the added session). Nested data, verified for
every fold before the run: B = the session 01–02 part of the fitting set of the folds run, the session 01–02 frames of
A\* and all of B' inside B, B' = the session 01–02 frames of A\*. Dev common to the three arms, without the test and
without session 03: sessions 01–02 of the dev-fold subjects, 100 frames per cell, 400 frames; it drives early stopping,
the learning-rate reduction on plateau, the restored epoch and the EER threshold, written and hashed before any test
frame was scored (0.5 also reported). Test: every frame of the held-out subjects, strata 01, 02, 03 and all, as in the
folds run. Recipe of the folds run unchanged (patience 4). 36 trainings (3 arms × 4 folds × seeds 42, 43, 44) on the
CPU, overall cap 360 minutes; a run stopped by the cap is reported as incomplete. Comparisons paired on the same test
frames and seeds (nominal pairing: batch order and augmentation differ with the data), pooled over the four folds per
seed, with a paired bootstrap over the held-out subjects of the stratum (1,000 resamples, seed 0, percentiles 2.5–97.5;
16 subjects in strata 03 and all, 9 in 01 and 02). Reading rule, fixed in the configuration before any training and
applied mechanically: a difference is `consistent` only if it has the same non-zero sign for all three seeds and the
interval of every seed excludes 0 on that side, otherwise `inconclusive`. Primary: AUROC and ACER at the dev threshold
in stratum 03 of A\* − B; the same rule, labelled secondary, for the other strata and metrics and for B' − B and A\* −
B': 3 comparisons × 4 strata × 7 metrics = 84 verdicts. No statistical test; diagnostic, not causal.

**Degenerate trainings (analysis after the run).** Run `20261107-DEGEN-a3b4dc0` (PR #22, merge `12dc78a`): no training,
no inference, no test score read; it reads the training cards (per epoch: `loss`, `accuracy`, `val_loss`,
`val_accuracy`, `lr`) and the dev scores of the 36 models, of the 12 models of the folds run and of the 6 complete
models of the failed first attempt (determinism only). The criteria were defined after the parent run, from the cards,
and are declared as such: degenerate = restored epoch 1 or 2 with dev accuracy 0.5 (within 1e-6) at that epoch;
saturated epoch = dev accuracy 0.5 (within 1e-6); recovery = first epoch with dev accuracy ≥ 0.75 and dev `val_loss` <
1.0 (sensitivity grid 0.70/0.75/0.80 × 0.7/1.0/1.5). The study replays the Keras `EarlyStopping` exactly on the recorded
histories (54 of 54 cards reproduced; note of PR #22) and lists seven possible explanations of the early dev saturation
(BatchNormalization moving statistics, dropout before BatchNormalization, augmentation, learning rate, early stopping
counted from a saturated epoch, small dev, initialisation), none of them tested. It proposed one change, chosen on
train/dev only: `early_stopping_patience` from 4 to 12 (= the maximum number of epochs), every other setting unchanged;
declared confounder: with ReduceLROnPlateau as configured (factor 0.5, patience 2, minimum 1e-6) a longer training also
halves the learning rate more times, so a recovery would not separate the stop from the learning-rate reduction.

**Patience verification.** Change and protocol agreed by the coordinator on 7 November, with two precisions written into
the note before the run (`research/nuaa-patience-verification/NOTE.md`, passages marked "[amended 9 Nov]").
Configuration `ca039df5…` = the parent configuration `4c2d29d1…` with `training.early_stopping_patience` 4 → 12 and one
more block, `amendment` (parent SHA-256, changed field, from, to, the coordinator's authorisation, the two precisions,
the time estimate, the shared-history check); every other field equal. The runner, before reading any data, refuses an
amended configuration unless its SHA-256 and that of its parent are the pinned ones and the two are equal in every field
except that one and the block. With patience 12 = the maximum number of epochs, every model runs 12 epochs and the
weights of the epoch of minimum dev `val_loss` are restored; same 36 models, data, order, dev, threshold rule and cap.
Run `20261109-SESSABLRUN-6a50265` (clean commit; PASS, 36 of 36). The test is computed and sealed by the runner as
before and reported separately. A separate comparison script, `nuaa_patience_verify.py` (run `20261109-PATVER-4f2b925`,
PASS), trains nothing and reads no test score: it reads the training cards, the dev scores and the frozen thresholds of
the two runs, and for the test only the public `aggregate.json` of each run. Its reading rule (a)–(e) was fixed before
the training run: (a) for every model, the epochs shared with the parent run (its first `epochs_run` there) must have
identical histories in the five fields, otherwise the comparison is not valid; (b) primary outcome positive if the ten
original degenerate trainings all recover within 12 epochs and none of the 36 models is degenerate, otherwise negative
("this change is not enough"), every model without recovery reported separately; (c) secondary: first recovery, restored
epoch, best `val_loss`, dev accuracy and dev AUROC at the checkpoint, epochs at 0.5, learning rate per epoch; (d) the
test, reported in a separate section, never part of the verdict; (e) an incomplete or failed run gets no verdict. For
both runs the synthesis script also checks the public files against each other (anchors, schema denominators, sums over
sessions and folds, every verdict against the reading rule, every value of the comparison rebuilt from the epoch
curves); the counts in the summary lines of `tables/patience_table.md` are counts of models derived from the public rows
and curves, not new metrics.

**Constant learning-rate control.** Proposed in §5 of the update of 10 November (PR #24, merge `1e340fc`) and agreed by
the coordinator on 10 November, with the protocol and the reading rule written in
`research/nuaa-constant-lr-control/NOTE.md` before the run. Question: in the patience-12 setting, do the ten recover
because training lasts 12 epochs, or does the learning-rate reduction that came with it contribute? Configuration
`60b38e09…` = the patience-12 configuration `ca039df5…` with `training.scheduler` changed from
`ReduceLROnPlateau(monitor=val_loss, factor=0.5, patience=2, min_lr=1e-6)` to a constant rate and the `amendment` block
rewritten for this control; every other field equal. The runner refuses it, before reading any data, unless both SHA-256
are the pinned ones and every other field is equal. The trainer, in which that field had been descriptive only, now
reads it (two accepted values, any other refused); with the constant value ReduceLROnPlateau is replaced by a callback
that does not change the rate and writes the rate in effect into the logs of every epoch, so that `history.lr` is
recorded in both modes; a training card with a recorded rate other than 0.001 is refused. Everything else as in the
patience-12 run: the same 36 models in the same order, data, dev, threshold rule and test, `early_stopping_patience` 12,
at most 12 epochs, checkpoint at the epoch of minimum dev `val_loss`, CPU, cap of 360 minutes; no seed added. Run
`20261111-SESSABLRUN-8568ffa` (clean commit `8568ffa`; 11 November, 00:07–04:21): PASS, 36 of 36, wall time 15,217.0 s
within the cap of 21,600 s (estimate 17,940.9 s); the rate was 0.001 at every epoch of every model. The test was
computed and sealed by the runner as before and is reported separately. The comparison with the patience-12 run,
`nuaa_patience_verify.py` with a second configuration schema (run `20261111-LRCTRL-ff53279`, clean commit, PASS), trains
nothing and reads no test score: it reads the training cards, the dev scores and the frozen thresholds of the two runs
and, for the test, only the public `aggregate.json` of each. Its reading rule (a)–(e) was fixed before the run: (a) for
every model the epochs of the patience-12 run recorded at 0.001, up to the epoch at whose end the first reduction
happened (3 to 12 epochs; all 12 for A\* fold 1 seed 42, never reduced), must be identical in the five fields, and the
rate of the control must be 0.001 at every epoch, otherwise the comparison is not valid; (b) primary: if all ten recover
within 12 epochs and none of the 36 is degenerate, the outcome is consistent with the extra epochs being sufficient; if
at least one of the ten does not recover, it is consistent with a contribution of the reduction; a mixed outcome (for
example all ten recover but another model becomes degenerate or loses its recovery) is reported model by model
(`NOTE.md`), with no overall verdict (`reading_rule.b` of the configuration of the comparison, fixed before the run as
well); (c) secondary, for every model: first recovery, recovery at epoch 12, restored epoch, dev accuracy and dev AUROC
at the checkpoint, epochs at 0.5, with B fold 3 seed 44 and B' fold 4 seed 42 reported apart; (d) the test in a separate
section, never used to choose or judge; (e) time against the estimate, no verdict for an incomplete run. The criteria
(degenerate, recovery) are those of the earlier packages, defined after the patience-4 run. Accepted with PR #25 (merge
`a1ca91f`, 11 November). For the control run and its comparison the synthesis script checks the public files as for the
two earlier runs (anchors, schema denominators, sums over sessions and folds, every verdict against the reading rule,
every value of the comparison rebuilt from the epoch curves except the dev AUROC at the checkpoint (copied)), and also
the amendment block (parent = the patience-12 configuration, changed field `training.scheduler`, ReduceLROnPlateau →
constant), the patience-12 columns of this comparison against those of the patience comparison, the shared window
(identical histories, rate 0.001 at every epoch of the control), the outcome of the reading rule recomputed from the
rows, and that the checkpoints with the same restored epoch inside the shared window have identical test values and dev
threshold in both runs (the hashes of their weight and threshold files differ, as different archives); the counts in the
summary lines of `tables/three_regimes_table.md` are counts of models derived from the public rows and curves, not new
metrics.

**BatchNormalization statistics, dev and test.** One control on the early dev saturation, proposed on 12 November and
approved by the coordinator on 13 November with three precisions written into the note before the run
(`research/nuaa-saturation-control/NOTE.md`), accepted with PR #27 (merge `c234064`); its extension to the test,
assigned on 16 November, in PR #28 (`research/nuaa-bn-test/NOTE.md`). No training. Both runs use the 36 checkpoints of
the patience-4 run `20261106-SESSABLRUN-d999458` as saved (the ten degenerate trainings; 25 control checkpoints, the
other models meeting the recovery criterion of §2 at their checkpoint; B fold 3 seed 44, which does not, reported
apart), on in-memory copies. Estimator, fixed before the dev run: per channel, the global mean and population variance
(divisor N; frames × height × width for the convolutional layers) of the input of each of the five BatchNormalization
layers over the whole fitting set of the model's arm and fold (1,400 frames for B and A\*, 934 for B'), layer by layer
in network order, in inference mode (dropout off), on frames / 255 without the augmentation of the trainer, accumulated
in float64 over batches of 16 weighted by their size; only `moving_mean` and `moving_variance` are replaced, every other
tensor (kernels, biases, gamma, beta) is compared byte for byte with the original, and one more pass over the fitting
set recomputes the moments of the five inputs. Dev run `20261114-BNRECAL-a10a082` (clean commit `a10a082`): a baseline
of all 36 checkpoints on the dev with their original statistics, against tolerances and a stop rule fixed before the run
(attack probability within 1e-5 of the stored dev scores and no decision changed at 0.5, val_loss within 1e-3, dev
accuracy within 1e-6, softmax of the logits within 1e-4); primary reading, fixed before the run: how many of the ten
meet the recovery criterion on the dev after the re-estimation, with a pre-written statement for all, some or none, each
limited to the effect of this re-estimation on these checkpoints; secondary: the control checkpoints, every worsening
indicator (any change in that direction, however small), the model apart and a 2×2 count, fitting before × dev after,
that names no cause. Test run `20261117-BNTEST-6245b10` (clean commit `6245b10`): the estimator of PR #27 imported
unchanged, its statistics required to agree with those of PR #27 within 1e-4 relative plus 1e-6 absolute, with the dev
probabilities within 1e-5 and no decision changed (observed: identical in 36 of 36); note, configuration and code were
published before the run (pull request #28 opened at 07:55:14 CET on 17 November with head `9aca72b`; the run started at
07:55:51 CET, first line of its `run/run.log.txt`; the final `run/status.json` lacks the `started` field, a defect of
the runner, so the start time rests on that log). Phases: a baseline on the dev (the checks of PR #27) and on the test
(the same frames as the stored test scores of the parent run, attack probability within 1e-5, no decision changed at
0.5), then the re-estimation, then the scoring with the re-estimated statistics; metrics on the same test frames before
and after, for every model, on the whole test fold (scope "test": every frame of the held-out subjects) and on session
03: APCER, BPCER and ACER at the fixed threshold 0.5 (no threshold selected), AUROC. Reading rule fixed in the note:
primary, the ten, scope test, ACER at 0.5, the mean over the ten of the paired difference after − before with a
bootstrap over subjects (the project's recipe unchanged: 1,000 resamples of the held-out subjects pooled over the folds,
seed 0, percentiles 2.5–97.5, no interval below 5 subjects, the same resamples before and after; new and declared: the
statistic of a resample is the mean over the models of their paired differences), read as entirely below zero,
containing zero (a bound at zero included) or entirely above zero, plus the count of the ten lower, equal or higher;
secondary and descriptive, with the same statements: session 03, the control checkpoints, the model apart, all 36, the
three arms, APCER, BPCER and AUROC; every worsening listed. Declared before the run, in the run's own words: "post hoc
diagnostic extension on a test already analysed", not an independent validation (the values before were published with
the parent run, and the re-estimation was chosen and first measured on the dev of the same folds); no statement on the
cause of the saturation. The synthesis script reads the public outputs of both runs and checks them against the parent
run and the patience comparison (§3.11; the checks are listed in `tables/bn_recalibration_table.md`).

**Verification practice.** Configuration hashed before each run; inputs, checkpoints and code bound by SHA-256;
per-image scores restricted, aggregates public; synthetic tests and declared mutations; a reproduction check where a
chain had to be rebuilt (N2: 24 parity keys, difference 0; crop check: 300/300 C1 scores within 1e-5, same decisions);
each package reviewed by the coordinator before merging (C1: recompute from the score cache reproduced; N2 and crop
check: metrics recomputed). For the folds run, the synthesis script also checks the public aggregate against the
approved schema (denominators of every fold and stratum), the run record (run, commit, configuration, cache check,
extraction, timings) and itself (stratum all = sum of the sessions, pooled = sum of the folds, medians and ranges).

## 3. Results

Unified table, compact form (codes: `T-NUAA` = fitting on `cnn_train`, early stopping on `cnn_es`; `T-COMB` = the five
historical datasets; `S-300` = `c1`; `S-3000` = `cnn_test`; `S-491` = the N2 target; `S-03/9` = the session 03 test
frames of the 9 training subjects, 614 + 924; `E-C1` / `E-CACHE` = the two extractors; `E-WHOLE` / `E-CROP` = the two
preparations of the combined model; `P-REF` = the reference chain, `P-HIST` = the historical chain; `N2-S` / `N2-C` = N2
part 1 / part 2; `T-FOLD` = fitting 2,100 + dev 600 frames of 8 + 4 subjects outside the held-out fold, all three
sessions; `S-FOLD-k` = every frame of the 4 subjects of held-out fold k, sessions 01–03; `S-FOLD-ALL` = the four
held-out folds pooled, 12,614 frames of the 16 subjects; `(EER, fold dev)` = the EER threshold of the dev fold of each
model; FOLDS-N1 = the fold models of each seed on the 3,000 frames of `S-3000` (each frame scored by the model of its
fold; the N1 rows above are the comparison); SESSABL-P4 / SESSABL-P12 / SESSABL-P12C = the session ablation run with
early-stopping patience 4, its verification run with patience 12 and its control run with patience 12 at a constant
learning rate 0.001, one row per arm and seed pooled over the four folds, stratum all; `T-ABL-B` / `T-ABL-A*` /
`T-ABL-B'` = the fitting sets of arms B (sessions 01–02, 1,400 frames per fold), A\* (sessions 01–03 of the same
subjects, 1,400) and B' (sessions 01–02 at the per-cell counts of A\*, 934), each with the common dev of the fold
(sessions 01–02, 400 frames); `(EER, common dev)` = the EER threshold of the common dev of each model; "subject-held-out
(session ablation)" = held-out subjects, sessions in training as in the arm, dev without session 03; ckpt = checkpoint;
"rejected bf / accepted att." = numerators of BPCER and APCER; the columns from "dev thr." to "bootstrap" apply to the
FOLDS, FOLDS-N1 and SESSABL rows only, — elsewhere; full legend in `tables/summary_table_compact.md`, row ids and
sources in `tables/summary_table.md`):

| block | model | training | test set | extractor | prep. | threshold | rejected bf / accepted att. | APCER | BPCER | ACER | AUROC | dev thr. | rejected bf / accepted att. (dev) | APCER dev | BPCER dev | ACER dev | bootstrap (ACER 0.5; ACER dev; AUROC) | run | transfer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | LivenessNet (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | 0.5 (fixed) | 109 / 24 | 0.1600 | 0.7267 | 0.4433 | 0.3782 | — | — | — | — | — | — | 20260928-C1-seed42-a80f6f9 | session |
| C1 | AttackNet V1 (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | 0.5 (fixed) | 40 / 42 | 0.2800 | 0.2667 | 0.2733 | 0.7547 | — | — | — | — | — | — | 20260928-C1-seed42-a80f6f9 | session |
| C1 | AttackNet V2.1 (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | 0.5 (fixed) | 115 / 3 | 0.0200 | 0.7667 | 0.3933 | 0.7103 | — | — | — | — | — | — | 20260928-C1-seed42-a80f6f9 | session |
| C1 | AttackNet V2.2 (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | 0.5 (fixed) | 118 / 22 | 0.1467 | 0.7867 | 0.4667 | 0.4985 | — | — | — | — | — | — | 20260928-C1-seed42-a80f6f9 | session |
| M1 | LivenessNet (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | 0.181055 (dev EER) | 118 / 5 | 0.0333 | 0.7867 | 0.4100 | 0.3782 | — | — | — | — | — | — | 20261003-M1-seed42-e67ecc7 | session |
| M1 | AttackNet V1 (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | 0.329237 (dev EER) | 44 / 37 | 0.2467 | 0.2933 | 0.2700 | 0.7547 | — | — | — | — | — | — | 20261003-M1-seed42-e67ecc7 | session |
| M1 | AttackNet V2.1 (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | 0.870329 (dev EER) | 91 / 11 | 0.0733 | 0.6067 | 0.3400 | 0.7103 | — | — | — | — | — | — | 20261003-M1-seed42-e67ecc7 | session |
| M1 | AttackNet V2.2 (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | 0.696933 (dev EER) | 112 / 38 | 0.2533 | 0.7467 | 0.5000 | 0.4985 | — | — | — | — | — | — | 20261003-M1-seed42-e67ecc7 | session |
| N1 | LivenessNet, new CPU training, seed 42 | T-NUAA | S-3000 | E-CACHE | P-REF | 0.5 (fixed) | 1155 / 160 | 0.1067 | 0.7700 | 0.4383 | 0.3485 | — | — | — | — | — | — | 20261024-N1-LivenessNet-fb93496 | session |
| N1 | LivenessNet, new CPU training, seed 43 | T-NUAA | S-3000 | E-CACHE | P-REF | 0.5 (fixed) | 1152 / 391 | 0.2607 | 0.7680 | 0.5143 | 0.3754 | — | — | — | — | — | — | 20261024-N1-LivenessNet-fb93496 | session |
| N1 | LivenessNet, new CPU training, seed 44 | T-NUAA | S-3000 | E-CACHE | P-REF | 0.5 (fixed) | 1131 / 72 | 0.0480 | 0.7540 | 0.4010 | 0.4582 | — | — | — | — | — | — | 20261024-N1-LivenessNet-fb93496 | session |
| N2-S | LivenessNet, N1 ckpt seed 42 | T-NUAA | S-491 | E-CACHE | P-REF | 0.5 (fixed) | 0 / 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | — | — | — | — | — | — | 20261029-N2-sessions-2ac3e29 | same |
| N2-S | LivenessNet, N1 ckpt seed 42 | T-NUAA | S-03/9 | E-CACHE | P-REF | 0.5 (fixed) | 422 / 154 | 0.1667 | 0.6873 | 0.4270 | 0.3230 | — | — | — | — | — | — | 20261029-N2-sessions-2ac3e29 | session/9 |
| N2-S | LivenessNet, N1 ckpt seed 43 | T-NUAA | S-491 | E-CACHE | P-REF | 0.5 (fixed) | 0 / 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | — | — | — | — | — | — | 20261029-N2-sessions-2ac3e29 | same |
| N2-S | LivenessNet, N1 ckpt seed 43 | T-NUAA | S-03/9 | E-CACHE | P-REF | 0.5 (fixed) | 422 / 359 | 0.3885 | 0.6873 | 0.5379 | 0.3171 | — | — | — | — | — | — | 20261029-N2-sessions-2ac3e29 | session/9 |
| N2-S | LivenessNet, N1 ckpt seed 44 | T-NUAA | S-491 | E-CACHE | P-REF | 0.5 (fixed) | 0 / 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | — | — | — | — | — | — | 20261029-N2-sessions-2ac3e29 | same |
| N2-S | LivenessNet, N1 ckpt seed 44 | T-NUAA | S-03/9 | E-CACHE | P-REF | 0.5 (fixed) | 422 / 63 | 0.0682 | 0.6873 | 0.3777 | 0.4569 | — | — | — | — | — | — | 20261029-N2-sessions-2ac3e29 | session/9 |
| N2-C | AttackNet V2.2 combined (coordinator's ckpt) | T-COMB | S-3000 | E-WHOLE | P-HIST | 0.5 (fixed) | 1492 / 222 | 0.1480 | 0.9947 | 0.5713 | 0.2169 | — | — | — | — | — | — | 20261029-N2-combined-247dfd7 | dataset |
| N2-C | AttackNet V2.2 combined (coordinator's ckpt) | T-COMB | S-3000 | E-CROP | P-HIST | 0.5 (fixed) | 1224 / 353 | 0.2353 | 0.8160 | 0.5257 | 0.3706 | — | — | — | — | — | — | 20261029-N2-combined-247dfd7 | dataset |
| CROP | AttackNet V2.1 (C1 ckpt 12/09), CPU batch 16 | T-NUAA | S-300 | E-CACHE | P-REF | 0.5 (fixed) | 121 / 4 | 0.0267 | 0.8067 | 0.4167 | 0.7092 | — | — | — | — | — | — | 20261102-C1CROP-f6747cc | session |
| CROP | AttackNet V2.1 (C1 ckpt 12/09), CPU batch 16 | T-NUAA | S-300 | E-C1 | P-REF | 0.5 (fixed) | 115 / 3 | 0.0200 | 0.7667 | 0.3933 | 0.7103 | — | — | — | — | — | — | 20261102-C1CROP-f6747cc | session |
| FOLDS | LivenessNet, fold 1, seed 42 | T-FOLD | S-FOLD-1 | E-CACHE | P-REF | 0.5 (fixed) | 117 / 1 | 0.0005 | 0.0822 | 0.0414 | 0.9920 | 0.602415 (EER, fold dev) | 115 / 1 | 0.0005 | 0.0808 | 0.0407 | — | 20261103-FOLDSRUN-e174812 | subject-held-out (sessions represented) |
| FOLDS | LivenessNet, fold 1, seed 43 | T-FOLD | S-FOLD-1 | E-CACHE | P-REF | 0.5 (fixed) | 414 / 467 | 0.2567 | 0.2909 | 0.2738 | 0.8632 | 0.639853 (EER, fold dev) | 363 / 576 | 0.3167 | 0.2551 | 0.2859 | — | 20261103-FOLDSRUN-e174812 | subject-held-out (sessions represented) |
| FOLDS | LivenessNet, fold 1, seed 44 | T-FOLD | S-FOLD-1 | E-CACHE | P-REF | 0.5 (fixed) | 143 / 1 | 0.0005 | 0.1005 | 0.0505 | 0.9949 | 0.982276 (EER, fold dev) | 93 / 9 | 0.0049 | 0.0654 | 0.0352 | — | 20261103-FOLDSRUN-e174812 | subject-held-out (sessions represented) |
| FOLDS | LivenessNet, fold 2, seed 42 | T-FOLD | S-FOLD-2 | E-CACHE | P-REF | 0.5 (fixed) | 315 / 54 | 0.0310 | 0.2299 | 0.1304 | 0.9144 | 0.576769 (EER, fold dev) | 302 / 73 | 0.0419 | 0.2204 | 0.1311 | — | 20261103-FOLDSRUN-e174812 | subject-held-out (sessions represented) |
| FOLDS | LivenessNet, fold 2, seed 43 | T-FOLD | S-FOLD-2 | E-CACHE | P-REF | 0.5 (fixed) | 117 / 3 | 0.0017 | 0.0854 | 0.0436 | 0.9963 | 0.738281 (EER, fold dev) | 81 / 22 | 0.0126 | 0.0591 | 0.0359 | — | 20261103-FOLDSRUN-e174812 | subject-held-out (sessions represented) |
| FOLDS | LivenessNet, fold 2, seed 44 | T-FOLD | S-FOLD-2 | E-CACHE | P-REF | 0.5 (fixed) | 419 / 24 | 0.0138 | 0.3058 | 0.1598 | 0.9651 | 0.656081 (EER, fold dev) | 412 / 46 | 0.0264 | 0.3007 | 0.1636 | — | 20261103-FOLDSRUN-e174812 | subject-held-out (sessions represented) |
| FOLDS | LivenessNet, fold 3, seed 42 | T-FOLD | S-FOLD-3 | E-CACHE | P-REF | 0.5 (fixed) | 23 / 31 | 0.0151 | 0.0166 | 0.0158 | 0.9972 | 0.467128 (EER, fold dev) | 24 / 27 | 0.0132 | 0.0173 | 0.0152 | — | 20261103-FOLDSRUN-e174812 | subject-held-out (sessions represented) |
| FOLDS | LivenessNet, fold 3, seed 43 | T-FOLD | S-FOLD-3 | E-CACHE | P-REF | 0.5 (fixed) | 1 / 39 | 0.0190 | 0.0007 | 0.0099 | 0.9995 | 0.283827 (EER, fold dev) | 5 / 30 | 0.0146 | 0.0036 | 0.0091 | — | 20261103-FOLDSRUN-e174812 | subject-held-out (sessions represented) |
| FOLDS | LivenessNet, fold 3, seed 44 | T-FOLD | S-FOLD-3 | E-CACHE | P-REF | 0.5 (fixed) | 98 / 44 | 0.0214 | 0.0707 | 0.0460 | 0.9959 | 0.678218 (EER, fold dev) | 49 / 52 | 0.0253 | 0.0353 | 0.0303 | — | 20261103-FOLDSRUN-e174812 | subject-held-out (sessions represented) |
| FOLDS | LivenessNet, fold 4, seed 42 | T-FOLD | S-FOLD-4 | E-CACHE | P-REF | 0.5 (fixed) | 0 / 596 | 0.3147 | 0.0000 | 0.1573 | 0.9658 | 0.040219 (EER, fold dev) | 17 / 421 | 0.2223 | 0.0184 | 0.1203 | — | 20261103-FOLDSRUN-e174812 | subject-held-out (sessions represented) |
| FOLDS | LivenessNet, fold 4, seed 43 | T-FOLD | S-FOLD-4 | E-CACHE | P-REF | 0.5 (fixed) | 0 / 342 | 0.1806 | 0.0000 | 0.0903 | 0.9855 | 0.672415 (EER, fold dev) | 0 / 390 | 0.2059 | 0.0000 | 0.1030 | — | 20261103-FOLDSRUN-e174812 | subject-held-out (sessions represented) |
| FOLDS | LivenessNet, fold 4, seed 44 | T-FOLD | S-FOLD-4 | E-CACHE | P-REF | 0.5 (fixed) | 1 / 172 | 0.0908 | 0.0011 | 0.0459 | 0.9950 | 0.958593 (EER, fold dev) | 0 / 467 | 0.2466 | 0.0000 | 0.1233 | — | 20261103-FOLDSRUN-e174812 | subject-held-out (sessions represented) |
| FOLDS | LivenessNet, 4 fold models, seed 42 (pooled) | T-FOLD | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 455 / 682 | 0.0908 | 0.0891 | 0.0900 | 0.9520 | 4 fold values (EER, fold dev) | 458 / 522 | 0.0695 | 0.0897 | 0.0796 | 0.0325-0.1718; 0.0284-0.1503; 0.8975-0.9906 | 20261103-FOLDSRUN-e174812 | subject-held-out (sessions represented) |
| FOLDS | LivenessNet, 4 fold models, seed 43 (pooled) | T-FOLD | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 532 / 851 | 0.1133 | 0.1042 | 0.1088 | 0.9740 | 4 fold values (EER, fold dev) | 449 / 1018 | 0.1356 | 0.0880 | 0.1118 | 0.0343-0.2090; 0.0399-0.2072; 0.9218-0.9982 | 20261103-FOLDSRUN-e174812 | subject-held-out (sessions represented) |
| FOLDS | LivenessNet, 4 fold models, seed 44 (pooled) | T-FOLD | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 661 / 241 | 0.0321 | 0.1295 | 0.0808 | 0.9755 | 4 fold values (EER, fold dev) | 554 / 574 | 0.0764 | 0.1085 | 0.0925 | 0.0257-0.1668; 0.0328-0.1862; 0.9351-0.9970 | 20261103-FOLDSRUN-e174812 | subject-held-out (sessions represented) |
| FOLDS-N1 | LivenessNet, 4 fold models, seed 42 (N1 test frames) | T-FOLD | S-3000 | E-CACHE | P-REF | 0.5 (fixed) | 157 / 170 | 0.1133 | 0.1047 | 0.1090 | 0.9385 | 4 fold values (EER, fold dev) | 153 / 133 | 0.0887 | 0.1020 | 0.0953 | 0.0244-0.2148; 0.0215-0.1932; 0.8650-0.9961 | 20261104-FOLDSCOMPARE-3b3bcca | subject-held-out, N1 test frames |
| FOLDS-N1 | LivenessNet, 4 fold models, seed 43 (N1 test frames) | T-FOLD | S-3000 | E-CACHE | P-REF | 0.5 (fixed) | 237 / 213 | 0.1420 | 0.1580 | 0.1500 | 0.9527 | 4 fold values (EER, fold dev) | 203 / 256 | 0.1707 | 0.1353 | 0.1530 | 0.0453-0.2943; 0.0520-0.2898; 0.8602-0.9968 | 20261104-FOLDSCOMPARE-3b3bcca | subject-held-out, N1 test frames |
| FOLDS-N1 | LivenessNet, 4 fold models, seed 44 (N1 test frames) | T-FOLD | S-3000 | E-CACHE | P-REF | 0.5 (fixed) | 235 / 65 | 0.0433 | 0.1567 | 0.1000 | 0.9670 | 4 fold values (EER, fold dev) | 212 / 139 | 0.0927 | 0.1413 | 0.1170 | 0.0182-0.2309; 0.0308-0.2438; 0.9075-0.9990 | 20261104-FOLDSCOMPARE-3b3bcca | subject-held-out, N1 test frames |
| SESSABL-P4 | LivenessNet, arm B, 4 fold models, seed 42 (pooled) | T-ABL-B | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 2788 / 1176 | 0.1566 | 0.5461 | 0.3514 | 0.6739 | 4 fold values (EER, common dev) | 1835 / 1204 | 0.1603 | 0.3595 | 0.2599 | 0.2023-0.4713; 0.1359-0.3813; 0.5166-0.8598 | 20261106-SESSABLRUN-d999458 | subject-held-out (session ablation) |
| SESSABL-P4 | LivenessNet, arm B, 4 fold models, seed 43 (pooled) | T-ABL-B | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 2682 / 1181 | 0.1573 | 0.5254 | 0.3413 | 0.7114 | 4 fold values (EER, common dev) | 2716 / 260 | 0.0346 | 0.5320 | 0.2833 | 0.2101-0.4605; 0.1550-0.3915; 0.5528-0.8547 | 20261106-SESSABLRUN-d999458 | subject-held-out (session ablation) |
| SESSABL-P4 | LivenessNet, arm B, 4 fold models, seed 44 (pooled) | T-ABL-B | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 3506 / 1334 | 0.1777 | 0.6868 | 0.4322 | 0.6696 | 4 fold values (EER, common dev) | 3126 / 1761 | 0.2345 | 0.6123 | 0.4234 | 0.2947-0.5570; 0.2782-0.5550; 0.4842-0.8564 | 20261106-SESSABLRUN-d999458 | subject-held-out (session ablation) |
| SESSABL-P4 | LivenessNet, arm A*, 4 fold models, seed 42 (pooled) | T-ABL-A* | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 455 / 2663 | 0.3546 | 0.0891 | 0.2219 | 0.8572 | 4 fold values (EER, common dev) | 1248 / 1649 | 0.2196 | 0.2445 | 0.2320 | 0.1244-0.3239; 0.1106-0.3655; 0.7404-0.9531 | 20261106-SESSABLRUN-d999458 | subject-held-out (session ablation) |
| SESSABL-P4 | LivenessNet, arm A*, 4 fold models, seed 43 (pooled) | T-ABL-A* | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 1575 / 1336 | 0.1779 | 0.3085 | 0.2432 | 0.7316 | 4 fold values (EER, common dev) | 179 / 3087 | 0.4111 | 0.0351 | 0.2231 | 0.1203-0.3827; 0.1330-0.3461; 0.5325-0.9205 | 20261106-SESSABLRUN-d999458 | subject-held-out (session ablation) |
| SESSABL-P4 | LivenessNet, arm A*, 4 fold models, seed 44 (pooled) | T-ABL-A* | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 2648 / 1834 | 0.2442 | 0.5187 | 0.3815 | 0.5494 | 4 fold values (EER, common dev) | 1673 / 4334 | 0.5772 | 0.3277 | 0.4524 | 0.2785-0.4990; 0.3443-0.5615; 0.4330-0.6648 | 20261106-SESSABLRUN-d999458 | subject-held-out (session ablation) |
| SESSABL-P4 | LivenessNet, arm B', 4 fold models, seed 42 (pooled) | T-ABL-B' | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 1510 / 2028 | 0.2701 | 0.2958 | 0.2829 | 0.7248 | 4 fold values (EER, common dev) | 1574 / 2573 | 0.3427 | 0.3083 | 0.3255 | 0.1646-0.4153; 0.1773-0.4806; 0.5518-0.8813 | 20261106-SESSABLRUN-d999458 | subject-held-out (session ablation) |
| SESSABL-P4 | LivenessNet, arm B', 4 fold models, seed 43 (pooled) | T-ABL-B' | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 2892 / 1267 | 0.1687 | 0.5665 | 0.3676 | 0.5891 | 4 fold values (EER, common dev) | 3062 / 836 | 0.1113 | 0.5998 | 0.3556 | 0.2485-0.4706; 0.2349-0.4570; 0.4036-0.7904 | 20261106-SESSABLRUN-d999458 | subject-held-out (session ablation) |
| SESSABL-P4 | LivenessNet, arm B', 4 fold models, seed 44 (pooled) | T-ABL-B' | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 4268 / 20 | 0.0027 | 0.8360 | 0.4194 | 0.5213 | 4 fold values (EER, common dev) | 1964 / 3661 | 0.4875 | 0.3847 | 0.4361 | 0.3073-0.5019; 0.2957-0.5700; 0.4207-0.6421 | 20261106-SESSABLRUN-d999458 | subject-held-out (session ablation) |
| SESSABL-P12 | LivenessNet, arm B, 4 fold models, seed 42 (pooled) | T-ABL-B | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 1799 / 1210 | 0.1611 | 0.3524 | 0.2568 | 0.8517 | 4 fold values (EER, common dev) | 2212 / 1104 | 0.1470 | 0.4333 | 0.2902 | 0.1265-0.3780; 0.1686-0.4148; 0.7290-0.9467 | 20261109-SESSABLRUN-6a50265 | subject-held-out (session ablation) |
| SESSABL-P12 | LivenessNet, arm B, 4 fold models, seed 43 (pooled) | T-ABL-B | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 2598 / 1489 | 0.1983 | 0.5089 | 0.3536 | 0.7314 | 4 fold values (EER, common dev) | 2716 / 500 | 0.0666 | 0.5320 | 0.2993 | 0.2162-0.4836; 0.1649-0.4158; 0.5871-0.8645 | 20261109-SESSABLRUN-6a50265 | subject-held-out (session ablation) |
| SESSABL-P12 | LivenessNet, arm B, 4 fold models, seed 44 (pooled) | T-ABL-B | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 1962 / 864 | 0.1151 | 0.3843 | 0.2497 | 0.8602 | 4 fold values (EER, common dev) | 1373 / 1065 | 0.1418 | 0.2690 | 0.2054 | 0.1380-0.3821; 0.0908-0.3437; 0.7103-0.9618 | 20261109-SESSABLRUN-6a50265 | subject-held-out (session ablation) |
| SESSABL-P12 | LivenessNet, arm A*, 4 fold models, seed 42 (pooled) | T-ABL-A* | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 527 / 774 | 0.1031 | 0.1032 | 0.1032 | 0.9622 | 4 fold values (EER, common dev) | 389 / 1350 | 0.1798 | 0.0762 | 0.1280 | 0.0515-0.1647; 0.0547-0.2181; 0.9176-0.9892 | 20261109-SESSABLRUN-6a50265 | subject-held-out (session ablation) |
| SESSABL-P12 | LivenessNet, arm A*, 4 fold models, seed 43 (pooled) | T-ABL-A* | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 354 / 1360 | 0.1811 | 0.0693 | 0.1252 | 0.9457 | 4 fold values (EER, common dev) | 398 / 1344 | 0.1790 | 0.0780 | 0.1285 | 0.0557-0.2251; 0.0558-0.2289; 0.8770-0.9859 | 20261109-SESSABLRUN-6a50265 | subject-held-out (session ablation) |
| SESSABL-P12 | LivenessNet, arm A*, 4 fold models, seed 44 (pooled) | T-ABL-A* | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 434 / 746 | 0.0993 | 0.0850 | 0.0922 | 0.9526 | 4 fold values (EER, common dev) | 464 / 1796 | 0.2392 | 0.0909 | 0.1650 | 0.0444-0.1502; 0.0927-0.2566; 0.8914-0.9889 | 20261109-SESSABLRUN-6a50265 | subject-held-out (session ablation) |
| SESSABL-P12 | LivenessNet, arm B', 4 fold models, seed 42 (pooled) | T-ABL-B' | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 1175 / 788 | 0.1049 | 0.2302 | 0.1676 | 0.8728 | 4 fold values (EER, common dev) | 956 / 1800 | 0.2397 | 0.1873 | 0.2135 | 0.0724-0.2823; 0.1087-0.3291; 0.7530-0.9637 | 20261109-SESSABLRUN-6a50265 | subject-held-out (session ablation) |
| SESSABL-P12 | LivenessNet, arm B', 4 fold models, seed 43 (pooled) | T-ABL-B' | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 2892 / 1267 | 0.1687 | 0.5665 | 0.3676 | 0.5891 | 4 fold values (EER, common dev) | 3062 / 836 | 0.1113 | 0.5998 | 0.3556 | 0.2485-0.4706; 0.2349-0.4570; 0.4036-0.7904 | 20261109-SESSABLRUN-6a50265 | subject-held-out (session ablation) |
| SESSABL-P12 | LivenessNet, arm B', 4 fold models, seed 44 (pooled) | T-ABL-B' | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 2683 / 681 | 0.0907 | 0.5256 | 0.3081 | 0.7480 | 4 fold values (EER, common dev) | 2651 / 1461 | 0.1946 | 0.5193 | 0.3569 | 0.1918-0.4183; 0.2250-0.4738; 0.6079-0.8911 | 20261109-SESSABLRUN-6a50265 | subject-held-out (session ablation) |
| SESSABL-P12C | LivenessNet, arm B, 4 fold models, seed 42 (pooled) | T-ABL-B | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 1023 / 1670 | 0.2224 | 0.2004 | 0.2114 | 0.8473 | 4 fold values (EER, common dev) | 1411 / 741 | 0.0987 | 0.2764 | 0.1875 | 0.1165-0.3292; 0.0893-0.2930; 0.6925-0.9667 | 20261111-SESSABLRUN-8568ffa | subject-held-out (session ablation) |
| SESSABL-P12C | LivenessNet, arm B, 4 fold models, seed 43 (pooled) | T-ABL-B | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 2535 / 1730 | 0.2304 | 0.4966 | 0.3635 | 0.6424 | 4 fold values (EER, common dev) | 2618 / 493 | 0.0657 | 0.5128 | 0.2892 | 0.2262-0.4880; 0.1655-0.4025; 0.4761-0.8233 | 20261111-SESSABLRUN-8568ffa | subject-held-out (session ablation) |
| SESSABL-P12C | LivenessNet, arm B, 4 fold models, seed 44 (pooled) | T-ABL-B | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 1880 / 1957 | 0.2606 | 0.3683 | 0.3144 | 0.7998 | 4 fold values (EER, common dev) | 2071 / 1194 | 0.1590 | 0.4057 | 0.2823 | 0.1794-0.4571; 0.1394-0.4272; 0.6478-0.9155 | 20261111-SESSABLRUN-8568ffa | subject-held-out (session ablation) |
| SESSABL-P12C | LivenessNet, arm A*, 4 fold models, seed 42 (pooled) | T-ABL-A* | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 427 / 1764 | 0.2349 | 0.0836 | 0.1593 | 0.9102 | 4 fold values (EER, common dev) | 758 / 1375 | 0.1831 | 0.1485 | 0.1658 | 0.0900-0.2512; 0.0847-0.2547; 0.8429-0.9680 | 20261111-SESSABLRUN-8568ffa | subject-held-out (session ablation) |
| SESSABL-P12C | LivenessNet, arm A*, 4 fold models, seed 43 (pooled) | T-ABL-A* | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 235 / 1721 | 0.2292 | 0.0460 | 0.1376 | 0.9526 | 4 fold values (EER, common dev) | 198 / 1975 | 0.2630 | 0.0388 | 0.1509 | 0.0652-0.2324; 0.0748-0.2572; 0.8791-0.9905 | 20261111-SESSABLRUN-8568ffa | subject-held-out (session ablation) |
| SESSABL-P12C | LivenessNet, arm A*, 4 fold models, seed 44 (pooled) | T-ABL-A* | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 973 / 654 | 0.0871 | 0.1906 | 0.1388 | 0.8931 | 4 fold values (EER, common dev) | 809 / 745 | 0.0992 | 0.1585 | 0.1288 | 0.0420-0.2586; 0.0462-0.2372; 0.7289-0.9949 | 20261111-SESSABLRUN-8568ffa | subject-held-out (session ablation) |
| SESSABL-P12C | LivenessNet, arm B', 4 fold models, seed 42 (pooled) | T-ABL-B' | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 2086 / 1569 | 0.2089 | 0.4086 | 0.3088 | 0.6940 | 4 fold values (EER, common dev) | 2360 / 1858 | 0.2474 | 0.4623 | 0.3549 | 0.1620-0.4551; 0.2273-0.4862; 0.5139-0.8596 | 20261111-SESSABLRUN-8568ffa | subject-held-out (session ablation) |
| SESSABL-P12C | LivenessNet, arm B', 4 fold models, seed 43 (pooled) | T-ABL-B' | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 2171 / 2045 | 0.2723 | 0.4253 | 0.3488 | 0.6433 | 4 fold values (EER, common dev) | 2838 / 920 | 0.1225 | 0.5559 | 0.3392 | 0.2367-0.4547; 0.2313-0.4338; 0.4763-0.8223 | 20261111-SESSABLRUN-8568ffa | subject-held-out (session ablation) |
| SESSABL-P12C | LivenessNet, arm B', 4 fold models, seed 44 (pooled) | T-ABL-B' | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) | 2506 / 1374 | 0.1830 | 0.4909 | 0.3369 | 0.6894 | 4 fold values (EER, common dev) | 2304 / 1412 | 0.1880 | 0.4513 | 0.3197 | 0.2021-0.4565; 0.1816-0.4338; 0.5262-0.8772 | 20261111-SESSABLRUN-8568ffa | subject-held-out (session ablation) |

**3.1 Same sessions as training (01–02).** The three N1 checkpoints make no error on the 491 training-session frames
never used for fitting or early stopping: ACER 0.0000 and AUROC 1.0000 for every seed, on the 410 frames of the 7
fitting subjects and on the 81 frames of the 2 early-stopping subjects alike (N2 part 1). The four C1 checkpoints
separate the 574 source-dev frames of those 2 subjects completely as well: dev AUROC (`dev AUC`) 1.0000 and operational
dev EER 0.0000 for all four (M1), so the dev does not identify a unique threshold. Caveats: the 493 early-stopping
frames chose the saved epoch; 2 subjects; 81 of the 491 are also dev frames.

**3.2 Session transfer (01–02 → 03).** At 0.5 on the 300 C1 images the four C1 checkpoints give ACER 0.2733–0.4667 and
AUROC 0.3782–0.7547; three reject most bona fide frames (BPCER 0.7267–0.7867), AttackNet V1 does not (BPCER 0.2667).
The M1 dev threshold moves ACER by −0.0533 to +0.0333 and does not change the ranking; excluding one test subject at a
time never changes the sign of that change except for AttackNet V1 (2 of 16, two attack-only subjects; N1 part 1).
LivenessNet retrained on the CPU with three seeds gives on the 3,000 test frames ACER 0.4010–0.5143, BPCER
0.7540–0.7700, AUROC 0.3485–0.4582 (bona fide ranked above attacks) with a near-zero validation loss in every run (N1).
The failure is not confined to new people: on the session 03 frames of the 9 training subjects BPCER is 0.6873 for every
seed and AUROC 0.3171–0.4569, on the 7 fitting subjects alone BPCER 0.6873 and AUROC 0.3191–0.4526 (N2 part 1). The
documented preprocessing difference does not explain it: scoring AttackNet V2.1 with the training-consistent extractor
changes 9 of 300 decisions (7 bona fide accepted → rejected, 1 the reverse, 1 attack rejected → accepted), ACER +0.0233,
AUROC within 0.002; the C1 row is reproduced to four decimals by the C1 extractor (crop check).

**3.3 Dataset transfer (→ NUAA).** The combined AttackNet V2.2 on the 3,000 test frames: whole image ACER 0.5713, AUROC
0.2169 (almost every frame called an attack); face crop ACER 0.5257, AUROC 0.3706; neither variant improves on any N1
seed on ACER, and the face-crop AUROC falls inside the N1 range (above seed 42, below 43 and 44) (N2 part 2). The 16 C1
checkpoints trained on CASIA-FASD, CelebA-Spoof and SynthASpoof, applied to the 300 NUAA images at 0.5 without
adaptation, range from ACER 0.2000 (AttackNet V1 on CASIA-FASD, AUROC 0.8106) to 0.7667, most near or below chance
(`tables/cross_dataset_table.md`).

**3.4 Context (same 300 images, C1 table).** Two third-party pre-trained models with no NUAA training in our experiment
separate the session 03 images almost perfectly at 0.5 (MiniFASNet ACER 0.0000, CLIP zero-shot 0.0100), as do linear
probes on frozen third-party encoders fitted on the 3,000 sampled training images (DINOv2 0.0000, CLIP 0.0133);
hand-crafted features fitted on the same images give ACER 0.32–0.39, the untrained Fourier rule 0.12 and Depth Anything
0.36 (`tables/context_rows_table.md`). "No NUAA training in our experiment" is what C1 establishes for these models and
encoders ("zero-shot" names the C1 analyzer: classification by text prompts, no classifier fitted on NUAA; it says
nothing about the pretraining data); their training data are those declared by their authors (C1 `inventory.md`), and
whether NUAA appears in them is not verified here and would need a provenance check. The images are separable by other
analyzers at the same threshold; the failure is specific to these NUAA-trained CNN checkpoints and their recipe (a
reading, not a tested cause; the folds run (§3.5) used the same recipe with every session represented, see the fourth
reading in §4; descriptive rows: the probes saw the dev frames, one threshold, no seed).

**3.5 Subject-held-out, sessions represented (run 20261103-FOLDSRUN-e174812).** Pooled over the four folds (all 12,614
frames, 16 subjects; each frame scored by the model of its fold), seeds 42, 43 and 44 give ACER 0.0900, 0.1088 and
0.0808 at 0.5 and 0.0796, 0.1118 and 0.0925 at the dev thresholds, AUROC 0.9520, 0.9740 and 0.9755 (bootstrap over
subjects: ACER at 0.5 0.0325–0.1718, 0.0343–0.2090, 0.0257–0.1668; AUROC 0.8975–0.9906, 0.9218–0.9982, 0.9351–0.9970).
By session stratum, pooled (median over the seeds, range in brackets): session 01 ACER 0.0613 (0.0064–0.0641) at 0.5 and
0.0535 (0.0076–0.0613) at the dev threshold, AUROC 0.9901 (0.9887–0.9997); session 02 0.0053 (0.0017–0.0187) and 0.0034
(0.0018–0.0141), AUROC 0.9997 (0.9996–0.9998); session 03 0.1094 (0.0975–0.1515) and 0.1178 (0.0937–0.1532), AUROC
0.9527 (0.9377–0.9664), bootstrap of the session-03 AUROC 0.8618–0.9960, 0.8579–0.9972 and 0.9051–0.9989. Session 03 is
the stratum with the highest pooled ACER for every seed, at both thresholds. Per held-out fold (stratum all, median over
the seeds, range): fold 1 ACER at 0.5 0.0505 (0.0414–0.2738), AUROC 0.9920 (0.8632–0.9949); fold 2 0.1304
(0.0436–0.1598), 0.9651 (0.9144–0.9963); fold 3 0.0158 (0.0099–0.0460), 0.9972 (0.9959–0.9995); fold 4 0.0903
(0.0459–0.1573), 0.9855 (0.9658–0.9950). Of the 48 fold × seed × stratum cells, 45 have AUROC ≥ 0.9; the three below are
fold 1 seed 43 (session 03 0.7087, all 0.8632) and fold 2 seed 42 (session 03 0.8609). Cells that stand apart
(observations; analysis in §3.7): fold 1, seed 43, session 03, ACER 0.4095 at 0.5 and 0.4254 at the dev threshold (dev:
BPCER 0.3914, APCER 0.4594), AUROC 0.7087, while seeds 42 and 44 on the same frames give ACER 0.0020 and 0.0009 at 0.5;
fold 4, session 03, no bona fide frame rejected at either threshold by any seed (BPCER 0.0000) and APCER 0.2567–0.3113
at the dev threshold (0.1147–0.3967 at 0.5), AUROC 0.9780–0.9980. Also visible in the annex: fold 1, session 01, BPCER
at 0.5 0.4173 and 0.4449 for seeds 42 and 44 (0.0039 for seed 43); fold 2, session 03, BPCER at 0.5 0.1119–0.4082.
Per-fold strata rest on 4 subjects (2–3 in sessions 01 and 02), so a single subject can move a cell; every stratum of
every fold and seed is in `tables/folds_run_table.md`.

**3.6 Same 3,000 identifiers as N1 (descriptive).** On the 3,000 session-03 frames on which N1 was tested (`S-3000`:
1,500 bona fide and 1,500 attacks of 16 subjects), each scored by the fold model that held its subject out, the folds
run gives ACER 0.1090, 0.1500 and 0.1000 at 0.5 and 0.0953, 0.1530 and 0.1170 at its dev thresholds, against 0.4383,
0.5143 and 0.4010 for N1 at 0.5, and AUROC 0.9385, 0.9527 and 0.9670 against 0.3485, 0.3754 and 0.4582 (seeds 42, 43,
44; rows FOLDS-N1 and N1 of the table; run `20261104-FOLDSCOMPARE-3b3bcca`). N1 AUROC is below 0.5 on these frames
(0.3485–0.4582, as published by N1); the AUROC difference includes that. Differences, folds minus N1, with the paired
bootstrap over the 16 subjects (2.5–97.5 percentiles): ACER at 0.5 −0.3293 [−0.4796, −0.1527], −0.3643 [−0.5270,
−0.1841], −0.3010 [−0.4600, −0.0945]; ACER at the dev thresholds against N1 at 0.5 −0.3430 [−0.4847, −0.1684], −0.3613
[−0.5206, −0.1856], −0.2840 [−0.4439, −0.0815]; AUROC +0.5900 [0.2821, 0.8453], +0.5772 [0.3090, 0.8056], +0.5088
[0.2094, 0.7525]. Per frame, the two decisions differ on 1,438, 1,477 and 1,205 of the 3,000 frames at 0.5 (1,411, 1,510
and 1,286 at the dev thresholds); the bona fide frames correct only in the folds run are 1,066, 946 and 990, those
correct only in N1 68, 31 and 94; among the attacks the two counts are closer for seeds 42 and 44 (147/157, 64/57), not
for seed 43 (339/161). Per subject, the error rate at 0.5 is lower in the folds run for 12, 14 and 10 of the 16 subjects
(higher for 4, 2 and 4, equal for 0, 0 and 2). The values computed by the coordinator from the scores of the folds run
are reproduced: agreement at the digits given by the coordinator (100 × ACER rounded to two decimals, AUROC rounded to
five):

| measure | control, as given (seeds 42 / 43 / 44) | computed (seeds 42 / 43 / 44) |
|---|---|---|
| ACER at 0.5 | 10.90 % / 15.00 % / 10.00 % | 0.10900 / 0.15000 / 0.10000 |
| ACER at the dev thresholds | 9.53 % / 15.30 % / 11.70 % | 0.09533 / 0.15300 / 0.11700 |
| AUROC | 0.93851 / 0.95268 / 0.96701 | 0.93851 / 0.95268 / 0.96701 |

Declared with the comparison: descriptive and post hoc, from existing scores and thresholds already frozen (no training,
no inference, no new threshold); the training roles differ (N1 was trained on 9 of these 16 subjects, in sessions 01–02;
in the folds run no frame is scored by a model that saw its subject); the AUROC of the folds run pools the scores of
four models; N1 has only the threshold 0.5, so the dev difference sets the folds run at its frozen dev thresholds
against N1 at 0.5; the 3,000 frames are a subset of the pooled session 03 of §3.5 (9,123 frames), so the folds values
here are close to those of §3.5 (equal to four decimals for the seed-43 AUROC) but computed on a subset; frames
correlated within subject and capture, three seeds, a coarse bootstrap over 16 subjects. Indicative; no causal reading:
the two experiments differ in training subjects, sessions, sample sizes, dev sets and thresholds at once, so the
difference is associated with the whole design, not with one factor. Every table of the comparison is in
`tables/n1_same_ids_table.md`.

**3.7 Fold 1 seed 43 and fold 4: observed and possible explanations.** Files already available, no new training or
inference: read-only descriptive calculations on the existing restricted scores, not verified by `check`. Sources:
[card] model card, [thr] threshold file, [scores] per-frame scores, [subj] per-subject table, [fm] fold manifests, [szr]
raw-view image statistics of step zero (restricted); [agg] public aggregate, [sz] public step-zero aggregate (crop
view). F<k>-a…d = the four held-out subjects of fold k (anonymous); no error rate or count is given per subject, only
within-subject AUROC and correlations.

*Fold 1, seed 43. Observed.* - Training [card]: restored epoch 8 by early stopping, minimum dev `val_loss` 0.3401 (seeds
42 and 44: 0.0591, 0.3497; other folds 0.0136–0.1302); the dev `val_loss` oscillates until epoch 12, as it does in some
other models. - Dev already poorly separated [thr, scores]: dev EER 0.1567 (seeds 42, 44: 0.0100, 0.0567); dev session
03 AUROC 0.7291, close to the test session 03 (0.7087 [agg]); dev errors concentrated in session 03: one subject's
session-03 attacks almost all accepted, bona fide errors on two subjects. - Test session 03 [agg, subj]: ACER 0.4254 at
the dev threshold (seeds 42, 44: 0.0020, 0.0028); almost all of the rejected bona fide come from one of the two subjects
with session-03 bona fide (F1-b), one capture group; the accepted attacks come from all three subjects with attacks and
from all 15 capture groups; AUROC within F1-b 0.41 (seeds 42, 44: 1.00). - Same data, three seeds [scores]: the
session-03 errors of seed 43 share at most 6 frames with another seed; Spearman of the attack scores with seeds 42 and
44: 0.3547 and 0.0340.

*Fold 1, seed 43. Possible explanations (not verified).*
- Seed-dependent training. For: same data, the other seeds reach AUROC 1.0000 in session 03 with different errors.
  Against: oscillations also in models that generalise; seeds 42 and 44 have their own concentrated error (most of
  F1-b's session-01 bona fide frames rejected); three seeds cannot say how often this happens.
- Stopping on a poorly separating epoch. For: dev session 03 AUROC 0.7291. Against: seed 44 has a similar `val_loss` and
  an almost perfect session-03 test; checking it needs per-epoch test scores (new inference).
- Luminance. For [sz, crop view]: in the N1 sets the crop luminance separates the classes in sessions 01–02 and is
  reversed in session 03 (AUROC 0.9221 and 0.3531). Against [szr, raw view, not the network input, which is the
  processed crop]: within F1-b the luminance–score correlation is −0.03 (Spearman); the other seeds classify the same
  frames; step zero describes the N1 sets, not the fold fitting.

*Fold 4. Observed.*
- Session 03 [agg, thr]: BPCER 0.0000 at both thresholds for every seed (553 bona fide of 2 subjects); APCER 0.2800,
  0.2567, 0.3113 at the dev thresholds 0.040219, 0.672415, 0.958593 (0.3967, 0.2267, 0.1147 at 0.5); AUROC
  0.9780–0.9980.
- Threshold not identified [scores]: for seeds 42 and 43 the dev is fully separated (dev AUROC 1.0000) and every
  threshold of the zero-error interval gives a session-03 APCER of 0.2800–0.3107 and 0.2567–0.3100 (descriptive, not a
  protocol threshold).
- Two subjects [subj, scores]: two of the four held-out subjects (F4-c, F4-d), both without frames in sessions 01–02,
  account for nearly all of the accepted attacks, with AUROC within them 1.00 and 0.93–0.99; F4-d is also the subject
  with the most accepted attacks when its frames are the dev of fold 3 (models fitted on other subjects), F4-c does not
  stand out there.

*Fold 4. Possible explanations (not verified).* - Non-unique dev threshold. Against: over the whole zero-error interval
the APCER stays ≥ 0.2567: it explains why the threshold moves between seeds, not the level of the error. - Scores
shifted, ranking kept, for subjects seen only in session 03. For: AUROC ≥ 0.9780; the fold-4 dev has session-03 attacks
only from subjects with sessions 01–02 [fm]; aggregated over the four folds, the session-03 APCER at the dev thresholds
is 0.1868–0.2121 for the 6 such subjects with attacks and 0.0214–0.1661 for the other 9 (the seed-43 value, 0.1661,
includes fold 1 seed 43) [subj]. Against: three subjects of that type have almost no accepted attacks; 6 subjects cannot
separate the type from the individual. - Luminance [szr, raw view, not the network input, which is the processed crop]:
the accepted attacks of fold 4 are darker than the rejected ones (AUROC 0.2890–0.3473). Against: within F4-d and F4-c
the luminance–score correlation is weak (−0.34 to −0.18; −0.09 to 0.16).

For both: frames are correlated, and each of the two bona fide cells rests on 2 capture groups, so these cells carry far
less information than their denominators suggest. Hypotheses, not causes; no test was run.

**3.8 Session ablation run, patience 4 (run 20261106-SESSABLRUN-d999458).** PASS, 36 of 36 trainings, wall time 10,209 s
of the 21,600 s cap. Pooled over the four folds (each frame scored by the model of its fold; 9,123 frames of 16 subjects
in stratum 03, 12,614 in stratum all), per arm and seed (stratum all: rows SESSABL-P4 of the table; every stratum with
the bootstrap in `tables/session_ablation_table.md`):

| arm | seed | stratum 03: ACER 0.5 | ACER dev | AUROC | stratum all: ACER 0.5 | ACER dev | AUROC |
|---|---|---|---|---|---|---|---|
| B | 42 | 0.4386 | 0.3417 | 0.5786 | 0.3514 | 0.2599 | 0.6739 |
| B | 43 | 0.4904 | 0.4199 | 0.4800 | 0.3413 | 0.2833 | 0.7114 |
| B | 44 | 0.4634 | 0.4993 | 0.5717 | 0.4322 | 0.4234 | 0.6696 |
| A\* | 42 | 0.2430 | 0.2563 | 0.8444 | 0.2219 | 0.2320 | 0.8572 |
| A\* | 43 | 0.2838 | 0.2496 | 0.6865 | 0.2432 | 0.2231 | 0.7316 |
| A\* | 44 | 0.3813 | 0.4985 | 0.5614 | 0.3815 | 0.4524 | 0.5494 |
| B' | 42 | 0.3622 | 0.4319 | 0.6118 | 0.2829 | 0.3255 | 0.7248 |
| B' | 43 | 0.4957 | 0.4720 | 0.4101 | 0.3676 | 0.3556 | 0.5891 |
| B' | 44 | 0.4546 | 0.4683 | 0.4902 | 0.4194 | 0.4361 | 0.5213 |

Primary verdicts (A\* − B, stratum 03, seeds 42 / 43 / 44, difference [paired bootstrap]): AUROC +0.2658 [−0.0060,
0.4862], +0.2064 [−0.0729, 0.4510], −0.0103 [−0.2949, 0.2683], inconclusive (the sign differs across the seeds); ACER at
the dev threshold −0.0854 [−0.3192, 0.1665], −0.1703 [−0.3013, −0.0303], −0.0008 [−0.1787, 0.2265], inconclusive (two
intervals include 0). Of the 84 verdicts, 1 is consistent and 83 are inconclusive; the consistent one is secondary: B' −
B in stratum 03, APCER at the dev threshold, B' higher (+0.2352, +0.0854, +0.2498). Reference A (the folds run: other
fitting frames and a dev with session 03; indicative, not compared): stratum 03 AUROC 0.9377, 0.9527 and 0.9664.

*Degenerate trainings (criterion defined after the run, §2).* Ten of the 36 models restored epoch 1 or 2 with dev
accuracy 0.5 and stopped at epoch 5 or 6: B 2 (fold 1 seed 42, fold 4 seed 44), A\* 5 (fold 2 seeds 43 and 44, fold 3
seeds 42 and 44, fold 4 seed 44), B' 3 (folds 1, 2 and 4, seed 44); seed 42: 2, seed 43: 1, seed 44: 7. Six give one
constant test score (AUROC 0.5000 and ACER 0.5000 in every stratum: A\* fold 2 seed 43, fold 3 seed 44, fold 4 seed 44;
B' folds 1, 2 and 4 seed 44); the other four are saturated near one class (minimum dev `val_loss` 2.36–18.09). Outside
the criterion, B fold 3 seed 44 restored epoch 3 with dev accuracy 0.2375 and never recovered: a failure of the same
family that the criterion misses. With patience 4, 25 of the 36 models reach the recovery criterion at some epoch (none
of the ten) and 16 still meet it at their last epoch. The dev was saturated at 0.5 in the first epoch in most models,
not only in the ten (24 of the 26 others, 12 of 12 in the folds run); the training accuracy was above 0.9 in every
saturated epoch, so the failure is visible only on the dev (note of PR #22). The reading rule used every model; nothing
was excluded or rerun. A first attempt (`20261106-SESSABLRUN-3b4021f`) had ended FAIL at model 7 of 36 on a false alarm
of the checkpoint check; its models are kept and not used.

*Post-hoc reading (declared as such; not part of the rule, it does not replace the verdicts).* Six of the 12 A\* − B
pairs (fold, seed) contain a degenerate model, five of them on the A\* side; seed 44 of A\* has three of its four folds
degenerate and is the seed with the negative AUROC difference. In the six pairs without a degenerate model the
stratum-03 difference is positive for AUROC (+0.2097 to +0.6558) and negative for ACER at the dev threshold (−0.4438 to
−0.0518) in every pair. This describes where the instability sits; it is a selection made after the run.

**3.9 Patience 4 versus 12 on the same 36 models (runs 20261109-SESSABLRUN-6a50265 and 20261109-PATVER-4f2b925).** Both
runs use the same data, dev, threshold rule and seeds; only `early_stopping_patience` differs. All 36 models are in the
compact table below; the full columns are in `tables/patience_table.md`, one row per model in two tables, each followed
by its counts: "Train and dev, per model (patience 4, then patience 12)", with epochs run, restored epoch, degenerate,
first recovery, "recovery holds at last epoch", best val_loss, dev accuracy and dev AUROC at the checkpoint, epochs at
0.5 and last lr for each run, then shared epochs, shared identical and restored epoch changed; and "Test, per model (not
used to choose or judge the recipe; patience 4, then patience 12)", with ACER at 0.5, ACER at the dev threshold and
AUROC in strata 03 and all for each run. The pooled values of both runs per arm, seed and stratum and the 84 verdicts of
each run side by side, with a "differs" column, are in `tables/session_ablation_table.md`; stratum all is also in rows
SESSABL-P4 and SESSABL-P12 of the table.

*(i) Training and dev.* Compact table (`tables/patience_table_compact.md`; p4 = run `20261106-SESSABLRUN-d999458`,
patience 4; p12 = run `20261109-SESSABLRUN-6a50265`, patience 12; "in the ten" = one of the ten degenerate trainings of
the p4 run, — = not among the ten; first recovery = first epoch meeting the recovery criterion of §2, — when none;
"recovery holds at last epoch" = that criterion met at the last epoch run; dev acc at ckpt = dev accuracy at the
restored epoch; "restored epoch changed" = the two restored epochs differ, — when equal; train and dev only, the test
per model is in the annex). "Restored epoch changed" compares the restored epochs only: with the same restored epoch the
weight and threshold file hashes still differ in every model (36 of 36 each), and the identity of the 21 unchanged
models is shown by identical test values and dev threshold in both runs. The cut-offs "epoch 10 or later" and "below
0.9" used below and in the summary lines are those of `CONCLUSION.md` of the patience verification.

| model | in the ten | restored epoch p4 → p12 | first recovery p4 → p12 | recovery holds at last epoch p4 → p12 | dev acc at ckpt p4 → p12 | restored epoch changed |
|---|---|---|---|---|---|---|
| B fold 1 seed 42 | yes | 1 → 10 | — → 8 | no → yes | 0.5000 → 0.9950 | yes |
| B fold 1 seed 43 | — | 11 → 11 | 5 → 5 | yes → yes | 0.9975 → 0.9975 | — |
| B fold 1 seed 44 | — | 6 → 11 | 6 → 6 | yes → yes | 0.9950 → 0.9975 | yes |
| B fold 2 seed 42 | — | 9 → 9 | 5 → 5 | no → no | 0.9475 → 0.9475 | — |
| B fold 2 seed 43 | — | 7 → 7 | 7 → 7 | yes → yes | 1.0000 → 1.0000 | — |
| B fold 2 seed 44 | — | 10 → 10 | 6 → 6 | yes → yes | 0.9975 → 0.9975 | — |
| B fold 3 seed 42 | — | 12 → 12 | 6 → 6 | yes → yes | 1.0000 → 1.0000 | — |
| B fold 3 seed 43 | — | 10 → 10 | 6 → 6 | yes → yes | 0.9925 → 0.9925 | — |
| B fold 3 seed 44 | — | 3 → 11 | — → 10 | no → yes | 0.2375 → 1.0000 | yes |
| B fold 4 seed 42 | — | 11 → 11 | 6 → 6 | yes → yes | 1.0000 → 1.0000 | — |
| B fold 4 seed 43 | — | 6 → 12 | 6 → 6 | yes → yes | 1.0000 → 1.0000 | yes |
| B fold 4 seed 44 | yes | 2 → 11 | — → 7 | no → yes | 0.5000 → 0.9925 | yes |
| A* fold 1 seed 42 | — | 10 → 10 | 6 → 6 | yes → yes | 0.9950 → 0.9950 | — |
| A* fold 1 seed 43 | — | 9 → 9 | 7 → 7 | yes → yes | 0.9975 → 0.9975 | — |
| A* fold 1 seed 44 | — | 11 → 11 | 7 → 7 | yes → yes | 0.9875 → 0.9875 | — |
| A* fold 2 seed 42 | — | 11 → 11 | 5 → 5 | yes → yes | 1.0000 → 1.0000 | — |
| A* fold 2 seed 43 | yes | 1 → 10 | — → 7 | no → yes | 0.5000 → 1.0000 | yes |
| A* fold 2 seed 44 | yes | 1 → 12 | — → 10 | no → yes | 0.5000 → 0.9975 | yes |
| A* fold 3 seed 42 | yes | 1 → 9 | — → 9 | no → yes | 0.5000 → 0.9425 | yes |
| A* fold 3 seed 43 | — | 12 → 12 | 6 → 6 | yes → yes | 0.9975 → 0.9975 | — |
| A* fold 3 seed 44 | yes | 1 → 11 | — → 11 | no → no | 0.5000 → 0.8900 | yes |
| A* fold 4 seed 42 | — | 10 → 10 | 10 → 10 | no → no | 0.7975 → 0.7975 | — |
| A* fold 4 seed 43 | — | 9 → 9 | 7 → 7 | no → no | 0.9950 → 0.9950 | — |
| A* fold 4 seed 44 | yes | 1 → 12 | — → 10 | no → yes | 0.5000 → 0.8650 | yes |
| B' fold 1 seed 42 | — | 2 → 12 | 2 → 2 | no → yes | 0.7725 → 0.9975 | yes |
| B' fold 1 seed 43 | — | 4 → 4 | 4 → 4 | no → no | 0.9400 → 0.9400 | — |
| B' fold 1 seed 44 | yes | 1 → 9 | — → 7 | no → no | 0.5000 → 0.9950 | yes |
| B' fold 2 seed 42 | — | 11 → 11 | 11 → 11 | no → no | 0.8775 → 0.8775 | — |
| B' fold 2 seed 43 | — | 10 → 10 | 3 → 3 | yes → yes | 1.0000 → 1.0000 | — |
| B' fold 2 seed 44 | yes | 2 → 12 | — → 12 | no → yes | 0.5000 → 0.9875 | yes |
| B' fold 3 seed 42 | — | 1 → 11 | 1 → 1 | no → yes | 0.8325 → 0.9900 | yes |
| B' fold 3 seed 43 | — | 9 → 9 | 7 → 7 | yes → yes | 1.0000 → 1.0000 | — |
| B' fold 3 seed 44 | — | 9 → 9 | 7 → 7 | yes → yes | 0.9975 → 0.9975 | — |
| B' fold 4 seed 42 | — | 2 → 2 | 1 → 1 | no → no | 0.9175 → 0.9175 | — |
| B' fold 4 seed 43 | — | 3 → 3 | 3 → 3 | no → yes | 1.0000 → 1.0000 | — |
| B' fold 4 seed 44 | yes | 1 → 11 | — → 10 | no → yes | 0.5000 → 0.9850 | yes |

- Models: 36 (3 arms x 4 folds x 3 seeds); complete: 36 with patience 4, 36 with patience 12.
- In the ten original degenerate trainings: 10.
- Degenerate with patience 4: 10 of 36 (10 of the ten).
- Degenerate with patience 12: 0 of 36 (0 of the ten).
- Shared histories identical: 36 of 36 (shared epochs = epochs run with patience 4 in 36 of 36).
- Epochs run with patience 4: 5-12, 17 of 36 models at 12.
- Epochs run with patience 12: 12, 36 of 36 models at 12.
- With a recovery (an epoch meeting the recovery criterion) with patience 4: 25 of 36 (0 of the ten).
- With a recovery (an epoch meeting the recovery criterion) with patience 12: 36 of 36 (10 of the ten).
- Recovery holds at the last epoch with patience 4: 16 of 36 (0 of the ten).
- Recovery holds at the last epoch with patience 12: 28 of 36 (8 of the ten).
- First recovery of the ten with patience 12: median (min-max) 9.5 (7-12).
- First recovery of the ten at epoch 10 or later with patience 12: 5 of the ten (cut-off of CONCLUSION.md).
- Restored epoch changed: 15 of 36 (10 of the ten, 5 of the other 26).
- Stopped before epoch 12 with patience 4: 19 models, 9 outside the ten, of which 5 changed restored epoch.
- Dev accuracy at the checkpoint, the ten, median (min-max) with patience 4: 0.5000 (0.5000-0.5000).
- Dev accuracy at the checkpoint, the ten, median (min-max) with patience 12: 0.9900 (0.8650-1.0000).
- Dev accuracy at the checkpoint, the other 26, median (min-max) with patience 4: 0.9963 (0.2375-1.0000).
- Dev accuracy at the checkpoint, the other 26, median (min-max) with patience 12: 0.9975 (0.7975-1.0000).
- Dev accuracy at the checkpoint below 0.9 with patience 12, the ten: 2 (cut-off of CONCLUSION.md).
- Dev accuracy at the checkpoint below 0.9 with patience 12, the other 26: 2 (cut-off of CONCLUSION.md).

- (a) Shared histories: for every model the epochs of the parent run (5–6 for the ten, 5–11 for the 9 other models
  stopped before epoch 12, 12 for the 17 complete ones) are identical in the verification run, 36 of 36, largest
  absolute difference 0.0 in the five fields: the comparison is valid.
- (b) The ten original degenerate trainings all recover within 12 epochs: first recovery at epochs 7–12, median 9.5;
  five of them at epoch 10 or later (A\* fold 2 seed 44 and fold 4 seed 44 at 10, A\* fold 3 seed 44 at 11, B' fold 4
  seed 44 at 10, B' fold 2 seed 44 at 12, the last epoch, which is also its restored epoch). None of the 36 models is
  degenerate (10 of 36 with patience 4); models with a recovery 36 of 36 (25 of 36 with patience 4, none of the ten).
  Primary outcome of the agreed rule: positive. B fold 3 seed 44, the only non-degenerate model without recovery with
  patience 4, recovers at epoch 10 and restores epoch 11 (dev accuracy 1.0000, best `val_loss` 0.0119); with patience 4
  it had stopped at epoch 7.
- The recovery criterion holds at the last epoch in 28 of the 36 models (8 of the ten), against 16 of 36 (none of the
  ten) with patience 4. It does not hold at epoch 12 in B fold 2 seed 42, A\* fold 3 seed 44, A\* fold 4 seeds 42 and
  43, B' fold 1 seeds 43 and 44, B' fold 2 seed 42 and B' fold 4 seed 42 (two of them among the ten). The checkpoint is
  the epoch of minimum dev `val_loss`, not the last one, and the dev curves after the recovery are not monotone.
- The restored epoch changes in 15 of the 36 models: the ten and 5 of the other 26, all of them among the nine
  non-degenerate models stopped early with patience 4 (B fold 1 seed 44, restored epoch 6 → 11; B fold 3 seed 44, 3 →
  11; B fold 4 seed 43, 6 → 12; B' fold 1 seed 42, 2 → 12; B' fold 3 seed 42, 1 → 11); the other 21 keep their restored
  epoch.
- The ten at the checkpoint: dev accuracy median 0.9900 (0.8650–1.0000), two below 0.9 (A\* fold 3 seed 44 0.8900, A\*
  fold 4 seed 44 0.8650); dev AUROC 0.9895–1.0000; best `val_loss` 0.0004–0.5239. The other 26: dev accuracy median
  0.9975 (0.7975–1.0000), two below 0.9 (A\* fold 4 seed 42 0.7975 and B' fold 2 seed 42 0.8775, the same values in both
  runs). Saturation in the first epochs remains: the ten have 2–7 epochs at 0.5.
- B' fold 4 seed 42 relapses (observation, not a criterion): it still restores epoch 2 (dev accuracy 0.9175, `val_loss`
  0.1375); its dev accuracy is 0.50 at epochs 5–7 (epochs 5–6 already at 0.5 with patience 4, shared history), 0.33–0.34
  at epochs 8–9 and 0.785 at epoch 12, with `val_loss` between 3.2 and 21.0 from epoch 5 on, while the learning rate is
  halved at epochs 5, 7, 9 and 11 (down to 0.0000625). It is not degenerate by the criterion.
- Learning rate, the declared confounder: with patience 12 ReduceLROnPlateau halved the learning rate in 35 of the 36
  models (first epoch with a recorded rate below 0.001: epoch 4 to 12); only A\* fold 1 seed 42 kept 0.001 for all 12
  epochs. The last rate of the ten is 0.00025 or 0.000125 (0.001 halved two or three times), against 0.0005 at their
  last epoch with patience 4. In the ten the first reduction comes at epoch 4 (8 models) or 5 (2), before every first
  recovery, and the rate at the epoch of first recovery is 0.00025 (8) or 0.000125 (2); in the other 26 models the first
  recovery comes at 0.001 in 16, 0.0005 in 9 and 0.000125 in 1 (`comparison/curves.csv` of the patience verification).
  The longer training and the lower rate therefore come together in the epochs in which the ten recover; this run does
  not separate them.
- Time: wall 17,940.9 s against the estimate of 13,090 s (+4,850.9 s, +37 %), within the cap of 21,600 s; seconds per
  epoch, weighted, B 42.6097, A\* 40.0181, B' 34.4986 against 32.4537, 31.8532 and 21.2375 with patience 4. The machine
  was not dedicated to the run (other processes active, not measured), so this difference is not attributed to the
  change.

*(ii) Test, reported separately and not used to choose or judge the recipe.* Read only from the public `aggregate.json`
of the two runs.

- The two primary verdicts (A\* − B, stratum 03) stay inconclusive: AUROC +0.2186 [0.0643, 0.4011], +0.3971 [0.2309,
  0.5873], +0.1689 [−0.0214, 0.3888] for seeds 42 / 43 / 44 (same sign, the interval of seed 44 includes 0); ACER at the
  dev threshold −0.2007 [−0.3589, 0.0024], −0.2772 [−0.3922, −0.1407], −0.0811 [−0.2545, 0.0999] (the intervals of seeds
  42 and 44 include 0).
- Verdicts: 1 consistent and 83 inconclusive with patience 4, 6 and 78 with patience 12. 7 of the 84 differ, 6 becoming
  consistent and 1 inconclusive: A\* − B, BPCER and ACER at 0.5, strata 03 and all (four verdicts, A\* lower); A\* − B',
  AUROC, strata 03 and all (two verdicts, A\* higher); B' − B, APCER at the dev threshold, stratum 03, consistent with
  patience 4, inconclusive with patience 12.
- Per model (patience 12 against patience 4; lower / equal / higher, of 36): stratum 03 ACER at 0.5 11 / 21 / 4, ACER at
  the dev threshold 11 / 21 / 4, AUROC 1 / 21 / 14; stratum all ACER at 0.5 14 / 21 / 1, ACER at the dev threshold 13 /
  21 / 2, AUROC 0 / 21 / 15. The 21 models whose restored epoch did not change have identical test values and dev
  thresholds in both runs; every difference comes from the 15 models whose restored epoch changed.
- Unfavourable per-model facts. Stratum 03 ACER at 0.5 rises in B fold 4 seed 43 (0.6316 → 0.6493), B fold 4 seed 44
  (0.5000 → 0.5950), B' fold 1 seed 44 (0.5000 → 0.5170) and B' fold 4 seed 44 (0.5000 → 0.5703); at the dev threshold
  it rises in B fold 1 seed 42 (0.2937 → 0.4045), B fold 4 seed 43 (0.4686 → 0.5486), B' fold 1 seed 44 (0.5000 →
  0.5170) and B' fold 4 seed 44 (0.5000 → 0.6534); three of each four are among the ten. The stratum-03 AUROC falls in
  B' fold 4 seed 44, one of the ten (0.5000 → 0.2515: after its recovery on the dev, it ranks the session-03 frames
  worse than chance). In stratum all, B fold 4 seed 43 rises at both thresholds (0.4276 → 0.4635, 0.2874 → 0.3508) and B
  fold 1 seed 42 at the dev threshold (0.2650 → 0.3699).
- Stratum 03, pooled per arm and seed, patience 4 → 12 (seeds 42 / 43 / 44): AUROC B 0.5786 → 0.7314, 0.4800 → 0.5185,
  0.5717 → 0.7646; A\* 0.8444 → 0.9500, 0.6865 → 0.9155, 0.5614 → 0.9335; B' 0.6118 → 0.8020, 0.4101 → 0.4101, 0.4902 →
  0.5646. ACER at the dev threshold B 0.3417 → 0.3747, 0.4199 → 0.4408, 0.4993 → 0.2893; A\* 0.2563 → 0.1740, 0.2496 →
  0.1636, 0.4985 → 0.2082; B' 0.4319 → 0.2908, 0.4720 → 0.4720, 0.4683 → 0.5040. Three of the nine pooled ACER values
  rise (B seeds 42 and 43, B' seed 44); B' seed 43 is identical in the two runs, as none of its four models changed
  restored epoch.
- These differences follow from the change through the 15 models whose restored epoch changed (10 of the 24 models of
  A\* − B); they are not attributable to the recovery of the ten alone; with patience 12 the dev chooses the restored
  epoch among 12 epochs; they are not a verdict on the change.

*Conclusion.* Train/dev: in this trial the longer patience let the ten degenerate trainings, and B fold 3 seed 44, reach
the recovery criterion within 12 epochs, with no degenerate model among the 36. It does not separate the duration of the
training from the learning-rate reduction (the constant-rate control, §3.10), which came with it in 35 of 36 models and
preceded every recovery of the ten; it does not show that every model is stable (the recovery holds at the last epoch in
28 of 36, 8 of the ten; B' fold 4 seed 42 relapsed; two of the ten and two of the other 26 stay below 0.9 dev accuracy
at the checkpoint); and it does not explain the saturation of the dev in the first epochs, which remains. Test: it is
not used to choose the recipe; the primary question of the ablation (does session 03 in training change the error on
held-out subjects?) stays inconclusive under the rule fixed before the run, with patience 4 and with patience 12; the 7
changed secondary verdicts and the per-model changes, in both directions, come with 15 changed restored epochs and are
not evidence for the change or for the session effect.

**3.10 Three regimes on the same 36 models: patience 4, patience 12 with the learning-rate reduction, patience 12 at a
constant rate (runs 20261106-SESSABLRUN-d999458, 20261109-SESSABLRUN-6a50265 and 20261111-SESSABLRUN-8568ffa;
comparisons 20261109-PATVER-4f2b925 and 20261111-LRCTRL-ff53279).** The three runs use the same 36 models, data, order,
dev, threshold rule and seeds. From p4 to p12 only `early_stopping_patience` changes (4 → 12); from p12 to p12c only the
scheduler (ReduceLROnPlateau → a constant rate 0.001). Each step has its own comparison, with its reading rule fixed
before its run (§3.9 for p4 → p12; §2, the note of the control and the reading rule of its comparison configuration for
p12 → p12c); the table puts the three regimes side by side and adds no rule of its own. All 36 models are in the compact
table below; the full columns are in `tables/three_regimes_table.md`, one row per model in two tables, each followed by
its counts: train and dev (for each run epochs run, restored epoch, degenerate, first recovery, recovery holds at last
epoch, epochs at 0.5 and epochs at 0.5 after the first recovery, best val_loss, dev accuracy and dev AUROC at the
checkpoint, last lr; then the first reduced epoch of p12, the shared epochs p12/p12c, restored epoch changed p4→p12 and
p12→p12c, same weights p12/p12c), then the test per model (ACER at 0.5, ACER at the dev threshold and AUROC in strata 03
and all for each run). The pooled values of the control run per arm, seed and stratum and the 84 verdicts of the three
runs side by side are in `tables/session_ablation_table_constant_lr.md`; stratum all is also in rows SESSABL-P12C of the
table.

*(i) Training and dev.* Compact table (`tables/three_regimes_table_compact.md`; p4 = run `20261106-SESSABLRUN-d999458`,
patience 4; p12 = run `20261109-SESSABLRUN-6a50265`, patience 12 with ReduceLROnPlateau; p12c = run
`20261111-SESSABLRUN-8568ffa`, patience 12 at a constant learning rate 0.001; "in the ten" = one of the ten degenerate
trainings of the p4 run, — = not among the ten; first recovery = first epoch meeting the recovery criterion of §2, —
when none; "holds at last epoch" = that criterion met at the last epoch run; dev acc at ckpt = dev accuracy at the
restored epoch; "restored epoch changed" = the two restored epochs differ, — when equal; train and dev only, the test
per model is in the annex). "Restored epoch changed" compares the restored epochs only: from p12 to p12c the two
checkpoints are the same weights only when the same restored epoch lies inside the shared window (the epochs before the
first reduction of the rate in p12, identical in the two runs) (in the public files the hashes of the weight and
threshold files still differ, as different archives; the identity is shown by the restored epoch inside the shared
window and by identical test values and dev threshold, checked by the synthesis script); the same restored epoch after
the window belongs to another training. The summary lines below the table are computed from its rows and from the
curves.

| model | in the ten | restored epoch p4 → p12 → p12c | first recovery p4 → p12 → p12c | holds at last epoch p4 → p12 → p12c | dev acc at ckpt p4 → p12 → p12c | restored epoch changed p4→p12 / p12→p12c |
|---|---|---|---|---|---|---|
| B fold 1 seed 42 | yes | 1 → 10 → 9 | — → 8 → 4 | no → yes → yes | 0.5000 → 0.9950 → 1.0000 | yes / yes |
| B fold 1 seed 43 | — | 11 → 11 → 9 | 5 → 5 → 5 | yes → yes → yes | 0.9975 → 0.9975 → 1.0000 | — / yes |
| B fold 1 seed 44 | — | 6 → 11 → 11 | 6 → 6 → 6 | yes → yes → yes | 0.9950 → 0.9975 → 1.0000 | yes / — |
| B fold 2 seed 42 | — | 9 → 9 → 10 | 5 → 5 → 8 | no → no → yes | 0.9475 → 0.9475 → 0.9950 | — / yes |
| B fold 2 seed 43 | — | 7 → 7 → 7 | 7 → 7 → 6 | yes → yes → yes | 1.0000 → 1.0000 → 1.0000 | — / — |
| B fold 2 seed 44 | — | 10 → 10 → 9 | 6 → 6 → 5 | yes → yes → no | 0.9975 → 0.9975 → 0.9975 | — / yes |
| B fold 3 seed 42 | — | 12 → 12 → 12 | 6 → 6 → 7 | yes → yes → yes | 1.0000 → 1.0000 → 1.0000 | — / — |
| B fold 3 seed 43 | — | 10 → 10 → 9 | 6 → 6 → 6 | yes → yes → yes | 0.9925 → 0.9925 → 0.9250 | — / yes |
| B fold 3 seed 44 | — | 3 → 11 → 10 | — → 10 → 7 | no → yes → yes | 0.2375 → 1.0000 → 0.9150 | yes / yes |
| B fold 4 seed 42 | — | 11 → 11 → 11 | 6 → 6 → 6 | yes → yes → yes | 1.0000 → 1.0000 → 1.0000 | — / — |
| B fold 4 seed 43 | — | 6 → 12 → 6 | 6 → 6 → 6 | yes → yes → no | 1.0000 → 1.0000 → 1.0000 | yes / yes |
| B fold 4 seed 44 | yes | 2 → 11 → 11 | — → 7 → 10 | no → yes → yes | 0.5000 → 0.9925 → 1.0000 | yes / — |
| A* fold 1 seed 42 | — | 10 → 10 → 10 | 6 → 6 → 6 | yes → yes → yes | 0.9950 → 0.9950 → 0.9950 | — / — |
| A* fold 1 seed 43 | — | 9 → 9 → 9 | 7 → 7 → 7 | yes → yes → no | 0.9975 → 0.9975 → 0.9975 | — / — |
| A* fold 1 seed 44 | — | 11 → 11 → 9 | 7 → 7 → 7 | yes → yes → yes | 0.9875 → 0.9875 → 0.9975 | — / yes |
| A* fold 2 seed 42 | — | 11 → 11 → 7 | 5 → 5 → 5 | yes → yes → yes | 1.0000 → 1.0000 → 1.0000 | — / yes |
| A* fold 2 seed 43 | yes | 1 → 10 → 9 | — → 7 → 6 | no → yes → yes | 0.5000 → 1.0000 → 1.0000 | yes / yes |
| A* fold 2 seed 44 | yes | 1 → 12 → 10 | — → 10 → 5 | no → yes → yes | 0.5000 → 0.9975 → 1.0000 | yes / yes |
| A* fold 3 seed 42 | yes | 1 → 9 → 10 | — → 9 → 6 | no → yes → no | 0.5000 → 0.9425 → 0.9900 | yes / yes |
| A* fold 3 seed 43 | — | 12 → 12 → 12 | 6 → 6 → 6 | yes → yes → yes | 0.9975 → 0.9975 → 0.9725 | — / — |
| A* fold 3 seed 44 | yes | 1 → 11 → 8 | — → 11 → 7 | no → no → yes | 0.5000 → 0.8900 → 0.8750 | yes / yes |
| A* fold 4 seed 42 | — | 10 → 10 → 11 | 10 → 10 → 9 | no → no → yes | 0.7975 → 0.7975 → 0.8975 | — / yes |
| A* fold 4 seed 43 | — | 9 → 9 → 9 | 7 → 7 → 7 | no → no → no | 0.9950 → 0.9950 → 0.9950 | — / — |
| A* fold 4 seed 44 | yes | 1 → 12 → 8 | — → 10 → 6 | no → yes → yes | 0.5000 → 0.8650 → 0.9975 | yes / yes |
| B' fold 1 seed 42 | — | 2 → 12 → 10 | 2 → 2 → 2 | no → yes → yes | 0.7725 → 0.9975 → 0.9975 | yes / yes |
| B' fold 1 seed 43 | — | 4 → 4 → 12 | 4 → 4 → 4 | no → no → yes | 0.9400 → 0.9400 → 0.9675 | — / yes |
| B' fold 1 seed 44 | yes | 1 → 9 → 6 | — → 7 → 6 | no → no → yes | 0.5000 → 0.9950 → 0.9975 | yes / yes |
| B' fold 2 seed 42 | — | 11 → 11 → 10 | 11 → 11 → — | no → no → no | 0.8775 → 0.8775 → 0.8550 | — / yes |
| B' fold 2 seed 43 | — | 10 → 10 → 12 | 3 → 3 → 3 | yes → yes → yes | 1.0000 → 1.0000 → 0.9950 | — / yes |
| B' fold 2 seed 44 | yes | 2 → 12 → 12 | — → 12 → 7 | no → yes → yes | 0.5000 → 0.9875 → 0.9950 | yes / — |
| B' fold 3 seed 42 | — | 1 → 11 → 4 | 1 → 1 → 1 | no → yes → yes | 0.8325 → 0.9900 → 0.9900 | yes / yes |
| B' fold 3 seed 43 | — | 9 → 9 → 9 | 7 → 7 → 7 | yes → yes → no | 1.0000 → 1.0000 → 1.0000 | — / — |
| B' fold 3 seed 44 | — | 9 → 9 → 9 | 7 → 7 → 8 | yes → yes → no | 0.9975 → 0.9975 → 0.9850 | — / — |
| B' fold 4 seed 42 | — | 2 → 2 → 11 | 1 → 1 → 1 | no → no → no | 0.9175 → 0.9175 → 0.9700 | — / yes |
| B' fold 4 seed 43 | — | 3 → 3 → 3 | 3 → 3 → 3 | no → yes → yes | 1.0000 → 1.0000 → 1.0000 | — / — |
| B' fold 4 seed 44 | yes | 1 → 11 → 10 | — → 10 → 10 | no → yes → no | 0.5000 → 0.9850 → 0.8900 | yes / yes |

**Recovery** (train and dev only)

- Models: 36 (3 arms x 4 folds x 3 seeds); complete: p4 36, p12 36, p12c 36.
- The ten (degenerate with p4) with a recovery: p4 0 of 10, p12 10 of 10, p12c 10 of 10.
- Degenerate among the 36: p4 10, p12 0, p12c 0.
- With a recovery among the 36: p4 25, p12 36, p12c 35.
- First recovery of the ten, median (min-max): p4 —, p12 9.5 (7-12), p12c 6 (4-10).
- First recovery of the ten, p12c against p12: earlier in 8, at the same epoch in 1, later in 1.
- First recovery of the other 26, median (min-max) [with one]: p4 6 (1-11) [25], p12 6 (1-11) [26], p12c 6 (1-9) [25].
- Recovery lost (outside the ten): p4 to p12 none; p12 to p12c B' fold 2 seed 42.

**Stability** (train and dev only)

- Recovery holds at the last epoch: p4 16 of 36, p12 28 of 36, p12c 26 of 36.
- Recovery holds at the last epoch, the ten: p4 0, p12 8, p12c 8; the other 26: p4 16, p12 20, p12c 18.
- Holds at the last epoch with p12 but not with p12c: 7; with p12c but not with p12: 5.
- Models with an epoch at 0.5 after their first recovery (a relapse): p4 5, p12 5, p12c 6.
- B fold 3 seed 44 (reported apart): first recovery — → 10 → 7; holds at last no → yes → yes; at 0.5 after it — → 0 → 0.
- B' fold 2 seed 42 (recovery lost): first recovery 11 → 11 → —; holds at last no → no → no; at 0.5 after it 0 → 0 → —.
- B' fold 4 seed 42 (reported apart): first recovery 1 → 1 → 1; holds at last no → no → no; at 0.5 after it 2 → 3 → 3.

**Selected checkpoint** (train and dev only)

- Restored epoch changed: p4 to p12 15 of 36 (10 of the ten), p12 to p12c 23 of 36 (8 of the ten).
- Same restored epoch with p12 and p12c: 13; of these, same weights (inside the shared window): 5.
- Different checkpoints with p12 and p12c: 31 of 36 (another restored epoch 23, same epoch after the window 8).
- Best val_loss at the checkpoint, p12c against p12: higher in 12, lower in 19, equal in 5.
- Dev accuracy at the checkpoint, p12c against p12: higher in 14, lower in 8, equal in 14.
- Dev accuracy and AUROC at the checkpoint, next four lines: median (min-max) per regime.
- Dev acc at ckpt, the ten: p4 0.5000 (0.5000-0.5000), p12 0.9900 (0.8650-1.0000), p12c 0.9975 (0.8750-1.0000).
- Dev acc at ckpt, the other 26: p4 0.9963 (0.2375-1.0000), p12 0.9975 (0.7975-1.0000), p12c 0.9963 (0.8550-1.0000).
- Dev AUROC at ckpt, the ten: p4 0.5000 (0.2114-0.9975), p12 0.9955 (0.9895-1.0000), p12c 1.0000 (0.9831-1.0000).
- Dev AUROC at ckpt, the other 26: p4 1.0000 (0.0950-1.0000), p12 1.0000 (0.9830-1.0000), p12c 1.0000 (0.8678-1.0000).
- Dev acc at ckpt below 0.9: p4 15 of 36, p12 4 of 36, p12c 4 of 36 (the ten: p4 10, p12 2, p12c 2).
- B fold 3 seed 44: restored epoch 3 → 11 → 10; dev acc at ckpt 0.2375 → 1.0000 → 0.9150.
- B' fold 2 seed 42: restored epoch 11 → 11 → 10; dev acc at ckpt 0.8775 → 0.8775 → 0.8550.
- B' fold 4 seed 42: restored epoch 2 → 2 → 11; dev acc at ckpt 0.9175 → 0.9175 → 0.9700.

Read by block, the same counts with the models named:

- Recovery. The ten degenerate trainings of the p4 run recover in none of the ten with p4, in 10 of 10 with p12 and in
  10 of 10 with p12c; degenerate models among the 36: 10, 0 and 0; models with a recovery: 25, 36 and 35 of 36. First
  recovery of the ten: median 9.5 (7–12) with p12, every one after the first reduction of the rate (first reduced epoch
  4 or 5); median 6 (4–10) with p12c, at 0.001, earlier in 8 of the ten, at the same epoch in B' fold 4 seed 44 (10) and
  later in B fold 4 seed 44 (10 against 7). The other 26: median 6 (1–11) with p4 (25 of 26) and p12 (26 of 26), 6 (1–9)
  with p12c (25 of 26). One recovery is lost with p12c, outside the ten: B' fold 2 seed 42, whose first recovery was at
  epoch 11 with p4 and p12 (with p12 the only epoch meeting the criterion, at 0.0005: dev accuracy 0.8775, followed by
  0.4625 at epoch 12) and which with p12c has dev accuracy 0.77–0.8625 at epochs 9–12 with `val_loss` never below 1.6. B
  fold 3 seed 44, without recovery with p4 (restored epoch 3, dev accuracy 0.2375), recovers at epoch 10 with p12 and at
  epoch 7 with p12c. By the rule of the control the p12 → p12c outcome is mixed, reported model by model (`NOTE.md`),
  with no overall verdict (`reading_rule.b` of the configuration of its comparison).
- Stability. The recovery criterion holds at the last epoch in 16 of 36 models with p4 (none of the ten), 28 with p12 (8
  of the ten) and 26 with p12c (8 of the ten). It holds with p12 but not with p12c in 7 models (B fold 2 seed 44, B fold
  4 seed 43, A\* fold 1 seed 43, A\* fold 3 seed 42, B' fold 3 seeds 43 and 44, B' fold 4 seed 44), with p12c but not
  with p12 in 5 (B fold 2 seed 42, A\* fold 3 seed 44, A\* fold 4 seed 42, B' fold 1 seeds 43 and 44), in neither in 3
  (A\* fold 4 seed 43, B' fold 2 seed 42, B' fold 4 seed 42). Models with an epoch at 0.5 after their first recovery (a
  relapse; an observation, not a criterion): 5 with p4, 5 with p12, 6 with p12c. B' fold 4 seed 42 relapses in all three
  regimes: it restores epoch 2 with p4 and p12, and its dev accuracy is 0.5 at epochs 5–7 with p12 (5–6 already with p4,
  shared history), with the rate halved at epochs 5, 7, 9 and 11, and at epochs 6–8 with p12c, at 0.001, where it
  restores epoch 11 (dev accuracy 0.9700); it is not degenerate in any regime. Epochs at 0.5 for the ten: median 4 in
  all three regimes (2–6, 2–7 and 2–5); between p12 and p12c they change in 12 of the 36 models, in both directions. The
  saturation of the dev in the first epochs remains in every regime, and the dev curves after the recovery are not
  monotone.
- Selected checkpoint. In every regime the checkpoint is the epoch of minimum dev `val_loss`, chosen among
  the 5–12 epochs run with p4 and among 12 with p12 and p12c. The restored epoch changed in 15 of 36 models from p4 to
  p12 (10 of the ten, 5 of the other 26) and in 23 of 36 from p12 to p12c (8 of the ten, 15 of the other 26). From p4 to
  p12 the 21 models with an unchanged restored epoch have identical test values and dev threshold (§3.9). From p12 to
  p12c 13 models keep the restored epoch, but in only 5 of them does it lie inside the shared window (A\* fold 1 seeds
  42 and 43, A\* fold 4 seed 43, B' fold 3 seed 43, B' fold 4 seed 43): these are the same weights (in the public files
  the hashes of the weight and threshold files still differ, as different archives; the identity is shown by the
  restored epoch inside the shared window and by identical test values and dev threshold, checked by the synthesis
  script); the other 8 are different trainings after the window, so 31 of the 36 checkpoints differ. Dev accuracy at the
  checkpoint, the ten: median 0.5000 (0.5000–0.5000) with p4, 0.9900 (0.8650–1.0000) with p12, 0.9975 (0.8750–1.0000)
  with p12c; the other 26: 0.9963 (0.2375–1.0000), 0.9975 (0.7975–1.0000), 0.9963 (0.8550–1.0000); dev AUROC at the
  checkpoint, the ten: 0.5000 (0.2114–0.9975), 0.9955 (0.9895–1.0000), 1.0000 (0.9831–1.0000); the other 26: 1.0000
  (0.0950–1.0000), 1.0000 (0.9830–1.0000), 1.0000 (0.8678–1.0000). Dev accuracy below 0.9 at the checkpoint: 15 of 36
  models with p4 (10 of the ten), 4 with p12 (2 of the ten) and 4 with p12c (2 of the ten: A\* fold 3 seed 44 0.8750, B'
  fold 4 seed 44 0.8900; and A\* fold 4 seed 42 0.8975, B' fold 2 seed 42 0.8550). From p12 to p12c the best `val_loss`
  at the checkpoint is higher in 12 models, lower in 19 and equal in 5; the dev accuracy at the checkpoint is higher in
  14, lower in 8 and equal in 14. Last recorded rate: 0.0005 for the ten with p4; with p12, among the 36, 0.0005 to
  0.0000625 (0.001 in A\* fold 1 seed 42 only; the ten 0.00025 or 0.000125); 0.001 in every model with p12c.

*(ii) Test, reported separately and never used to choose or judge the recipe.* Read only from the public
`aggregate.json` of the three runs (primary comparison A\* − B, stratum 03, seeds 42 / 43 / 44, difference [paired
bootstrap]):

| regime | AUROC | ACER at the dev threshold | primary verdicts (AUROC; ACER dev) |
|---|---|---|---|
| p4 | +0.2658 [−0.0060, 0.4862], +0.2064 [−0.0729, 0.4510], −0.0103 [−0.2949, 0.2683] | −0.0854 [−0.3192, 0.1665], −0.1703 [−0.3013, −0.0303], −0.0008 [−0.1787, 0.2265] | inconclusive; inconclusive |
| p12 | +0.2186 [0.0643, 0.4011], +0.3971 [0.2309, 0.5873], +0.1689 [−0.0214, 0.3888] | −0.2007 [−0.3589, 0.0024], −0.2772 [−0.3922, −0.1407], −0.0811 [−0.2545, 0.0999] | inconclusive; inconclusive |
| p12c | +0.1286 [−0.0576, 0.3794], +0.5694 [0.3222, 0.7760], +0.1945 [−0.0458, 0.4088] | −0.0343 [−0.1970, 0.1349], −0.2425 [−0.3920, −0.0661], −0.2293 [−0.3733, −0.0629] | inconclusive; inconclusive |

- The two primary verdicts (A\* − B, stratum 03) are inconclusive in all three regimes: with p12c the AUROC differences
  have the same sign and the intervals of seeds 42 and 44 include 0; the ACER differences have the same sign and the
  interval of seed 42 includes 0.
- Of the 84 verdicts, consistent / inconclusive: 1 / 83 with p4, 6 / 78 with p12, 10 / 74 with p12c; 69 are the same in
  the three runs. From p4 to p12 7 differ (§3.9): A\* − B, BPCER and ACER at 0.5, strata 03 and all, become consistent
  (A\* lower); A\* − B', AUROC, strata 03 and all, become consistent (A\* higher); B' − B, APCER at the dev threshold,
  stratum 03, becomes inconclusive. From p12 to p12c 12 differ: the four A\* − B verdicts at 0.5 that became consistent
  with p12 are inconclusive again; A\* − B', BPCER and ACER at 0.5 and at the dev threshold, strata 03 and all, become
  consistent (A\* lower); A\* − B', AUROC, strata 03 and all, is consistent (A\* higher) in both.
- Per model (p12c against p12; lower / equal / higher, of 36): stratum 03 ACER at 0.5 8 / 5 / 23, ACER at the dev
  threshold 15 / 5 / 16,
  AUROC 19 / 5 / 12; stratum all ACER at 0.5 9 / 5 / 22, ACER at the dev threshold 16 / 5 / 15, AUROC 16 / 5 / 15. The 5
  models with the same weights (restored epoch inside the shared window; file hashes differ, as different archives) have
  identical test values and dev threshold in both runs, as the script checks; the other 31 differ, in every arm.
- Stratum 03, pooled per arm and seed, p12 → p12c (seeds 42 / 43 / 44): AUROC B 0.7314 → 0.7326, 0.5185 → 0.3608, 0.7646
  → 0.6470; A\* 0.9500 → 0.8612, 0.9155 → 0.9303, 0.9335 → 0.8415; B' 0.8020 → 0.5587, 0.4101 → 0.4416, 0.5646 → 0.5321.
  ACER at the dev threshold B 0.3747 → 0.2624, 0.4408 → 0.4265, 0.2893 → 0.3916; A\* 0.1740 → 0.2281, 0.1636 → 0.1840,
  0.2082 → 0.1623; B' 0.2908 → 0.4732, 0.4720 → 0.4681, 0.5040 → 0.4241. Six of the nine pooled AUROC values fall (B
  seeds 43 and 44, A\* seeds 42 and 44, B' seeds 42 and 44) and four of the nine pooled ACER values at the dev threshold
  rise (B seed 44, A\* seeds 42 and 43, B' seed 42).
- The test values per model differ in the 15 models whose restored epoch changed from p4 to p12 (21 identical) and in 31
  of 36 from p12 to p12c (5 identical), in every arm; these differences are not attributable to the ten alone and are
  not a verdict on any regime or on the session-03 question.

*Conclusion.* Train/dev: on the same 36 models, the ten degenerate trainings of patience 4 recover within 12 epochs both
with the learning-rate reduction and at a constant rate (10 of 10 in each, earlier at the constant rate in 8 of the
ten), and no model is degenerate in either 12-epoch regime; for the ten, the recovery of the patience-12 run is
reproduced without the reduction; the ten alone would meet the first case of the rule, which gives that reading only
when no recovery is lost outside the ten, which is not the case here. The conclusion stays mixed: at the constant rate
one recovery is lost outside the ten, B' fold 2 seed 42, whose recovery with the reduction was itself marginal (one
epoch); by the rule fixed before the run there is no overall verdict, and the three regimes show neither that the
reduction is unnecessary nor that it is necessary. No regime shows that every model is stable (recovery holding at the
last epoch in 16, 28 and 26 of 36; the relapse of B' fold 4 seed 42 in all three), and none explains the saturation of
the dev in the first epochs. The selected checkpoint moves with the rate path (23 of 36 restored epochs, 31 of 36
checkpoints from p12 to p12c), so dev and test values at the checkpoint differ for reasons beyond the recovery; neither
12-epoch regime is better than the other on every train/dev count (p12c against p12: models with a recovery 35 against
36, recovery holding at the last epoch 26 against 28; best val_loss lower in 19 of 36, dev accuracy at the checkpoint
higher in 14 and lower in 8). Test: the primary results stay inconclusive in all three regimes; the changed secondary
verdicts (7 and 12 of 84) come with changed checkpoints, and the recipe is not chosen on the test. Limits: the criteria
were defined after the first (patience-4) run, while the rules of the two later comparisons were fixed before their
runs; the constant rate tests the absence of the reduction, not the level of the rate (0.001 is never varied); the dev
is small (2–3 subjects, 400 frames per fold) and chooses the checkpoint among 12 epochs; one dataset, one attack type
(print), three seeds per fold, folds not independent; the cause of the early dev saturation is not tested.

**3.11 BatchNormalization statistics re-estimated on the fitting set, dev and test (runs 20261114-BNRECAL-a10a082 and
20261117-BNTEST-6245b10).** The 36 checkpoints of the patience-4 run, before and after the re-estimation of §2; no
training. All 36 are in the compact table below (`tables/bn_recalibration_table_compact.md`), with the paired summaries
of the test and summary lines; the annex `tables/bn_recalibration_table.md` has every column of the dev, every
checkpoint and scope of the test, the 56 paired summaries, and the 14 sources with their SHA-256. The script checks, for
both runs, identity and anchors (clean tree, `PASS`, run id ending with the commit, `status.json` → `SHA256SUMS` →
every file read, 36 models within the cap, the parent run read here by run id and by the SHA-256 of its public
aggregate); on the dev, each group against the patience comparison, the values before against the checkpoint (dev
accuracy within 1e-6, val_loss within 1e-3, AUROC equal), and the criterion, the 2×2 cell, every worsening indicator,
the readings and the counts recomputed from the values; on the test, every denominator and every value before equal to
the published per-model value of the parent run (72 of 72 model and scope rows), APCER and BPCER consistent with whole
counts over the denominators and ACER their mean, differences equal to after − before, the indicators, and in every
paired summary the models, folds, subjects, means, lower / equal / higher counts and the reading recomputed from the
rows (the subjects with each class and the valid resamples are copied); for both runs the largest baseline differences
against the tolerances of the run. The bootstrap intervals are copied, since they need the per-subject results, which
are restricted. The unified table gets no row: its rows of the 36-model design (SESSABL) are pooled per arm and seed
over the four folds, at 0.5 and at the dev threshold, with a bootstrap per seed; the dev control has no test metric, and
the test control publishes per-model rows and paired means over models at 0.5 only, so a pooled or bootstrapped BN row
would be a new number. The unified table does hold per-model rows (the folds run), but the per-model rows of the BN
controls are kept, by choice, in the BN annex.

| model | group | restored epoch | dev accuracy before → after | dev val_loss before → after | dev criterion before → after | worsened on the dev | test ACER 0.5 before → after | session 03 ACER 0.5 before → after | test AUROC before → after |
|---|---|---|---|---|---|---|---|---|---|
| B fold 1 seed 42 | ten | 1 | 0.5000 → 0.9950 | 7.0611 → 0.0244 | no → yes | — | 0.5000 → 0.1988 | 0.5000 → 0.2919 | 0.7389 → 0.9217 |
| B fold 1 seed 43 | control | 11 | 0.9975 → 0.9925 | 0.0056 → 0.1830 | yes → yes | dev acc, dev val_loss, dev AUROC | 0.3234 → 0.3538 | 0.4715 → 0.5009 | 0.8948 → 0.7598 |
| B fold 1 seed 44 | control | 6 | 0.9950 → 0.9950 | 0.0298 → 0.3445 | yes → yes | dev val_loss, dev AUROC, fitting acc, fitting loss | 0.3565 → 0.3135 | 0.4970 → 0.4633 | 0.6400 → 0.7759 |
| B fold 2 seed 42 | control | 9 | 0.9475 → 1.0000 | 0.2308 → 0.0087 | yes → yes | — | 0.2757 → 0.2103 | 0.3643 → 0.2686 | 0.8368 → 0.9406 |
| B fold 2 seed 43 | control | 7 | 1.0000 → 1.0000 | 0.0011 → 0.0004 | yes → yes | — | 0.3479 → 0.3706 | 0.4650 → 0.4916 | 0.6701 → 0.6939 |
| B fold 2 seed 44 | control | 10 | 0.9975 → 1.0000 | 0.0030 → 0.0029 | yes → yes | — | 0.1659 → 0.0789 | 0.2219 → 0.1015 | 0.9865 → 0.9706 |
| B fold 3 seed 42 | control | 12 | 1.0000 → 1.0000 | 0.0168 → 0.0075 | yes → yes | — | 0.2091 → 0.2456 | 0.3154 → 0.3617 | 0.8298 → 0.6852 |
| B fold 3 seed 43 | control | 10 | 0.9925 → 0.9675 | 0.0531 → 0.0640 | yes → yes | dev acc, dev val_loss | 0.2555 → 0.2584 | 0.3726 → 0.3762 | 0.7776 → 0.7653 |
| B fold 3 seed 44 | apart | 3 | 0.2375 → 0.9950 | 3.2731 → 0.0417 | no → yes | — | 0.7265 → 0.1891 | 0.6554 → 0.2968 | 0.2781 → 0.8301 |
| B fold 4 seed 42 | control | 11 | 1.0000 → 0.9975 | 0.0005 → 0.1622 | yes → yes | dev acc, dev val_loss, dev AUROC | 0.3936 → 0.4227 | 0.5710 → 0.6140 | 0.7502 → 0.6143 |
| B fold 4 seed 43 | control | 6 | 1.0000 → 0.9975 | 0.0013 → 0.0047 | yes → yes | dev acc, dev val_loss | 0.4276 → 0.4677 | 0.6316 → 0.6810 | 0.5968 → 0.6553 |
| B fold 4 seed 44 | ten | 2 | 0.5000 → 1.0000 | 2.3605 → 0.0013 | no → yes | — | 0.5000 → 0.5236 | 0.5000 → 0.7424 | 0.4672 → 0.6272 |
| A* fold 1 seed 42 | control | 10 | 0.9950 → 0.9975 | 0.0104 → 0.0134 | yes → yes | dev val_loss | 0.0839 → 0.1317 | 0.1243 → 0.1571 | 0.9828 → 0.9555 |
| A* fold 1 seed 43 | control | 9 | 0.9975 → 0.9950 | 0.0165 → 0.0128 | yes → yes | dev acc | 0.0513 → 0.1560 | 0.0281 → 0.1795 | 0.9921 → 0.9934 |
| A* fold 1 seed 44 | control | 11 | 0.9875 → 0.9875 | 0.0345 → 0.5327 | yes → yes | dev val_loss, dev AUROC, fitting acc, fitting loss | 0.1428 → 0.1731 | 0.1972 → 0.1986 | 0.8963 → 0.9165 |
| A* fold 2 seed 42 | control | 11 | 1.0000 → 1.0000 | 0.0001 → 0.0003 | yes → yes | dev val_loss, fitting loss | 0.0937 → 0.0964 | 0.1216 → 0.1259 | 0.9701 → 0.9607 |
| A* fold 2 seed 43 | ten | 1 | 0.5000 → 1.0000 | 60.9868 → 0.0031 | no → yes | — | 0.5000 → 0.0616 | 0.5000 → 0.0697 | 0.5000 → 0.9972 |
| A* fold 2 seed 44 | ten | 1 | 0.5000 → 0.9800 | 18.0934 → 0.3553 | no → yes | — | 0.5000 → 0.0542 | 0.5000 → 0.0677 | 0.2782 → 0.9981 |
| A* fold 3 seed 42 | ten | 1 | 0.5000 → 1.0000 | 3.1296 → 0.0137 | no → yes | — | 0.5000 → 0.1109 | 0.5000 → 0.1433 | 0.5255 → 0.9727 |
| A* fold 3 seed 43 | control | 12 | 0.9975 → 0.9175 | 0.0088 → 0.1670 | yes → yes | dev acc, dev val_loss | 0.1836 → 0.0720 | 0.2333 → 0.0930 | 0.9534 → 0.9727 |
| A* fold 3 seed 44 | ten | 1 | 0.5000 → 0.8275 | 33.4090 → 0.2594 | no → yes | — | 0.5000 → 0.1731 | 0.5000 → 0.2552 | 0.5000 → 0.9023 |
| A* fold 4 seed 42 | control | 10 | 0.7975 → 0.7875 | 0.8106 → 0.7175 | yes → yes | dev acc, dev AUROC, fitting acc | 0.1763 → 0.1576 | 0.1270 → 0.1013 | 0.9271 → 0.9416 |
| A* fold 4 seed 43 | control | 9 | 0.9950 → 0.9175 | 0.0272 → 0.1516 | yes → yes | dev acc, dev val_loss, dev AUROC | 0.1777 → 0.2148 | 0.2251 → 0.3152 | 0.9227 → 0.7956 |
| A* fold 4 seed 44 | ten | 1 | 0.5000 → 0.8100 | 22.8274 → 0.6411 | no → yes | — | 0.5000 → 0.1052 | 0.5000 → 0.0224 | 0.5000 → 0.9799 |
| B' fold 1 seed 42 | control | 2 | 0.7725 → 0.9950 | 0.6101 → 0.3244 | yes → yes | — | 0.4933 → 0.2127 | 0.7216 → 0.2768 | 0.5029 → 0.9156 |
| B' fold 1 seed 43 | control | 4 | 0.9400 → 0.9975 | 0.8647 → 0.0096 | yes → yes | — | 0.4131 → 0.3557 | 0.5057 → 0.5154 | 0.5857 → 0.6022 |
| B' fold 1 seed 44 | ten | 1 | 0.5000 → 0.9950 | 12.1449 → 0.0146 | no → yes | — | 0.5000 → 0.3271 | 0.5000 → 0.4751 | 0.5000 → 0.8025 |
| B' fold 2 seed 42 | control | 11 | 0.8775 → 0.9875 | 0.8750 → 0.0215 | yes → yes | — | 0.1241 → 0.1813 | 0.0994 → 0.2314 | 0.9281 → 0.9271 |
| B' fold 2 seed 43 | control | 10 | 1.0000 → 0.9975 | 0.0042 → 0.0093 | yes → yes | dev acc, dev val_loss | 0.3759 → 0.4343 | 0.4986 → 0.5721 | 0.6391 → 0.6623 |
| B' fold 2 seed 44 | ten | 2 | 0.5000 → 0.9975 | 11.4259 → 0.0065 | no → yes | — | 0.5000 → 0.1386 | 0.5000 → 0.1769 | 0.5000 → 0.9845 |
| B' fold 3 seed 42 | control | 1 | 0.8325 → 0.8800 | 0.3211 → 0.3449 | yes → yes | dev val_loss | 0.3314 → 0.4369 | 0.4593 → 0.5795 | 0.7411 → 0.7787 |
| B' fold 3 seed 43 | control | 9 | 1.0000 → 0.9975 | 0.0011 → 0.0279 | yes → yes | dev acc, dev val_loss | 0.2715 → 0.2860 | 0.3984 → 0.4160 | 0.6528 → 0.7142 |
| B' fold 3 seed 44 | control | 9 | 0.9975 → 0.9175 | 0.0341 → 0.1218 | yes → yes | dev acc, dev val_loss | 0.2031 → 0.2052 | 0.3228 → 0.2742 | 0.7675 → 0.9255 |
| B' fold 4 seed 42 | control | 2 | 0.9175 → 0.9500 | 0.1375 → 0.3614 | yes → yes | dev val_loss, dev AUROC, fitting loss | 0.1606 → 0.3692 | 0.2207 → 0.5701 | 0.8717 → 0.7445 |
| B' fold 4 seed 43 | control | 3 | 1.0000 → 1.0000 | 0.0142 → 0.0037 | yes → yes | — | 0.3959 → 0.4333 | 0.5756 → 0.6500 | 0.4826 → 0.7240 |
| B' fold 4 seed 44 | ten | 1 | 0.5000 → 1.0000 | 17.2907 → 0.0281 | no → yes | — | 0.5000 → 0.3201 | 0.5000 → 0.5096 | 0.5000 → 0.6134 |

Legend. Dev = run `20261114-BNRECAL-a10a082` (PR #27), test = run `20261117-BNTEST-6245b10` (PR #28), both on the 36
checkpoints of run `20261106-SESSABLRUN-d999458` (early-stopping patience 4); before = original statistics, after =
moving mean and variance of the five BatchNormalization layers re-estimated on the fitting set of the model's arm and
fold, every trained parameter unchanged; group: ten = the ten degenerate trainings, control = the 25 control checkpoints
(the other models meeting the recovery criterion at their checkpoint), apart = the 1 that do not; dev criterion =
recovery criterion (dev accuracy >= 0.75 and val_loss < 1.0); worsened on the dev = indicators set in PR #27 (any change
in the worsening direction, however small; — none); test = every frame of the held-out subjects of the fold; ACER at the
fixed threshold 0.5; AUROC threshold-free. The test reading is a post hoc diagnostic extension on a test already
analysed, not an independent validation.

**Dev (PR #27)**

- Models: 36 (the ten degenerate trainings 10, control checkpoints 25, reported apart 1: B fold 3 seed 44); dev 400
  frames per model; fitting 1,400 frames (B, A*) or 934 (B').
- Baseline with the original statistics, before any re-estimation: 36 of 36 within every tolerance (largest
  attack-probability difference 0, changed decisions at 0.5 0, largest dev val_loss difference 2.876e-06, largest dev
  accuracy difference 0, largest softmax gap 3.329e-06).
- The ten meeting the recovery criterion on the dev: before 0 of 10, after 10 of 10 (primary reading of the run: all).
- The ten after the re-estimation: dev accuracy 0.8100-1.0000, dev val_loss 0.0013-0.6411, dev AUROC 0.9814-1.0000;
  fitting accuracy before 0.5000-0.5000, after 0.9886-1.0000.
- Control checkpoints meeting the criterion on the dev: before 25 of 25, after 25 of 25 (lost: 0).
- Control checkpoints with at least one worsened indicator: 17 of 25 (dev accuracy lower 11, dev val_loss higher 15, dev
  AUROC lower 7, fitting accuracy lower 3, fitting loss higher 4); with none: 8. The ten and the model apart with a
  worsened indicator: 0.
- Control checkpoints, correct dev frames after against before: higher 8, equal 6, lower 11; largest loss 32 of 400 (A*
  fold 3 seed 43, B' fold 3 seed 44); dev accuracy after 0.7875-1.0000.
- Model reported apart, B fold 3 seed 44: dev accuracy 0.2375 → 0.9950, dev val_loss 3.2731 → 0.0417, criterion no →
  yes.
- Fitting before x dev after (checkpoints counted, no cause named): both 22, dev after only 14 (the ten 10, apart 1,
  control checkpoints 3), fitting only 0, neither 0.

| summary | scope | models | subjects (bona fide / attack) | ACER 0.5 | APCER 0.5 | BPCER 0.5 | AUROC | valid resamples (ACER / APCER / BPCER / AUROC) |
|---|---|---|---|---|---|---|---|---|
| group ten | test | 10 | 16 (15 / 15) | -0.2987 [-0.4033, -0.2147] below; 9/0/1 | -0.0406 [-0.0948, 0.0279] contains; 2/0/8 | -0.5568 [-0.7246, -0.4507] below; 8/0/2 | 0.3790 [0.2635, 0.4344] above; 0/0/10 | 926 / 950 / 940 / 926 |
| group ten | session 03 | 10 | 16 (9 / 15) | -0.2246 [-0.3759, -0.1425] below; 8/0/2 | -0.0033 [-0.0734, 0.0651] contains; 2/0/8 | -0.4459 [-0.6941, -0.3185] below; 8/0/2 | 0.3153 [0.2027, 0.4447] above; 1/0/9 | 650 / 950 / 661 / 650 |
| group control | test | 25 | 16 (15 / 15) | 0.0082 [-0.0220, 0.0316] contains; 7/0/18 | 0.0409 [0.0026, 0.0714] above; 8/1/16 | -0.0246 [-0.0600, 0.0130] contains; 13/1/11 | 0.0237 [0.0039, 0.0396] above; 10/0/15 | 926 / 950 / 940 / 926 |
| group control | session 03 | 25 | 16 (9 / 15) | 0.0138 [-0.0295, 0.0424] contains; 7/0/18 | 0.0490 [-0.0004, 0.0973] contains; 8/1/16 | -0.0214 [-0.0794, 0.0195] contains; 13/3/9 | 0.0228 [0.0018, 0.0448] above; 12/0/13 | 650 / 950 / 661 / 650 |
| group apart | test | 1 | 4 (3 / 4) | -0.5373 no interval, not_available; 1/0/0 | -0.6067 no interval, not_available; 1/0/0 | -0.4679 no interval, not_available; 1/0/0 | 0.5520 no interval, not_available; 0/0/1 | 0 / 0 / 0 / 0 |
| group apart | session 03 | 1 | 4 (2 / 4) | -0.3586 no interval, not_available; 1/0/0 | -0.5219 no interval, not_available; 1/0/0 | -0.1954 no interval, not_available; 1/0/0 | 0.2815 no interval, not_available; 0/0/1 | 0 / 0 / 0 / 0 |
| group all | test | 36 | 16 (15 / 15) | -0.0922 [-0.1399, -0.0533] below; 17/0/19 | 0.0003 [-0.0334, 0.0383] contains; 11/1/24 | -0.1848 [-0.2502, -0.1359] below; 22/1/13 | 0.1370 [0.0863, 0.1598] above; 10/0/26 | 926 / 950 / 940 / 926 |
| group all | session 03 | 36 | 16 (9 / 15) | -0.0627 [-0.1322, -0.0232] below; 16/0/20 | 0.0186 [-0.0219, 0.0661] contains; 11/1/24 | -0.1441 [-0.2525, -0.0928] below; 22/3/11 | 0.1113 [0.0645, 0.1586] above; 13/0/23 | 650 / 950 / 661 / 650 |
| arm B | test | 12 | 16 (15 / 15) | -0.0707 [-0.1333, -0.0041] below; 5/0/7 | 0.0541 [0.0087, 0.1012] above; 3/0/9 | -0.1956 [-0.2880, -0.1083] below; 8/1/3 | 0.0644 [-0.0282, 0.1224] contains; 5/0/7 | 926 / 950 / 940 / 926 |
| arm B | session 03 | 12 | 16 (9 / 15) | -0.0313 [-0.1071, 0.0292] contains; 5/0/7 | 0.0850 [0.0305, 0.1442] above; 3/0/9 | -0.1477 [-0.2818, -0.0457] below; 8/1/3 | 0.0449 [-0.0333, 0.1262] contains; 5/0/7 | 650 / 950 / 661 / 650 |
| arm A* | test | 12 | 16 (15 / 15) | -0.1586 [-0.1937, -0.1160] below; 7/0/5 | -0.1554 [-0.1919, -0.1112] below; 7/1/4 | -0.1618 [-0.2083, -0.1069] below; 4/0/8 | 0.2032 [0.1418, 0.2394] above; 3/0/9 | 926 / 950 / 940 / 926 |
| arm A* | session 03 | 12 | 16 (9 / 15) | -0.1523 [-0.1981, -0.1179] below; 7/0/5 | -0.1541 [-0.1903, -0.1117] below; 7/1/4 | -0.1506 [-0.2170, -0.1101] below; 5/1/6 | 0.1943 [0.1364, 0.2574] above; 4/0/8 | 650 / 950 / 661 / 650 |
| arm B' | test | 12 | 16 (15 / 15) | -0.0474 [-0.1223, 0.0072] contains; 5/0/7 | 0.1022 [0.0414, 0.1643] above; 1/0/11 | -0.1969 [-0.3181, -0.1204] below; 10/0/2 | 0.1436 [0.0811, 0.1895] above; 2/0/10 | 926 / 950 / 940 / 926 |
| arm B' | session 03 | 12 | 16 (9 / 15) | -0.0046 [-0.1064, 0.0427] contains; 4/0/8 | 0.1250 [0.0473, 0.2090] above; 1/0/11 | -0.1341 [-0.2746, -0.0995] below; 9/1/2 | 0.0945 [0.0063, 0.2002] above; 4/0/8 | 650 / 950 / 661 / 650 |

Legend. Paired summaries of the test control, copied from `summary.csv` (means, model counts, folds, subjects and
readings recomputed from its rows; subjects with each class, valid resamples and intervals copied): per metric, mean
over the models of the difference after - before, with the subject-bootstrap interval [2.5, 97.5] (1,000 resamples of
the held-out subjects pooled over the folds, seed 0, the same resamples before and after; no interval with fewer than 5
subjects) and the reading fixed before the run (below / contains / above zero, not_available without an interval), then
the models lower / equal / higher after (for AUROC, higher = AUROC higher); for APCER, BPCER and ACER a negative value
is a lower error rate. The summaries share models and subjects and are not independent.

**Test (PR #28)**

- Baseline on the test with the original statistics: 36 of 36 within every tolerance (largest attack-probability
  difference 0, changed decisions at 0.5 0); the values before equal the published per-model values of the parent run in
  72 of 72 model and scope rows (checked here); re-estimated statistics identical to those of PR #27 in 36 of 36.
- Primary reading (fixed before the run): the ten, scope test, ACER at 0.5, mean difference after - before -0.2987
  [-0.4033, -0.2147] below; lower 9, equal 0, higher 1 (higher: B fold 4 seed 44 0.5000 → 0.5236).
- The ten, session 03, ACER at 0.5: mean difference -0.2246 [-0.3759, -0.1425] below; lower 8, equal 0, higher 2
  (higher: B fold 4 seed 44 0.5000 → 0.7424, B' fold 4 seed 44 0.5000 → 0.5096).
- Control checkpoints, models higher after: ACER at 0.5 18 of 25 on the test and 18 in session 03; APCER at 0.5 16 and
  16; BPCER at 0.5 11 and 9; AUROC lower 10 and 12.
- Control checkpoints with a worsened indicator on the dev in PR #27 (17): ACER at 0.5 higher on the test in 14 of them,
  in session 03 in 13.
- Indicators set, models (test / session 03): ACER higher 19 / 20, APCER higher 24 / 24, BPCER higher 13 / 11, AUROC
  lower 10 / 13.
- Subjects behind the intervals: 16 held-out subjects, 15 with bona fide and 15 with attack frames on the test, 9 and 15
  in session 03; valid resamples for ACER 926 of 1,000 on the test and 650 in session 03; the model apart (1 fold, 4
  subjects) has no interval.

**Time**

- Dev control: wall 1,748.6 s (29.1 minutes); estimate 32 minutes (upper 140); cap 240 minutes.
- Test control: wall 3,027.4 s (50.5 minutes), process CPU time over all threads 18,438.4 s; estimate 42 minutes (upper
  112); cap 240 minutes of elapsed time.
- Parent run (patience 4): 36 trainings, 326 epochs, 9,437.7 s of training (wall 10,209.2 s); seconds per epoch B
  32.4537, A* 31.8532, B' 21.2375 (fitting + dev frames 1,800, 1,800, 1,334).

*(i) Dev (PR #27).* The baseline passed for all 36 checkpoints, so the stop rule did not apply. With their original
statistics, in inference mode, the ten have accuracy 0.5000 on the dev and on the fitting set (both balanced by class).
After the re-estimation, every trained parameter unchanged, all ten meet the recovery criterion (primary reading "all";
dev accuracy 0.8100–1.0000, val_loss 0.0013–0.6411, fitting accuracy 0.9886–1.0000); they do not reach the same level:
A\* fold 3 seed 44 stays at 0.8275 and A\* fold 4 seed 44 at 0.8100. The 25 control checkpoints keep the criterion, and
17 of them have at least one worsened indicator: three lose 31–32 of 400 correct dev frames and end at dev accuracy
0.9175 (A\* fold 3 seed 43, A\* fold 4 seed 43, B' fold 3 seed 44), B fold 3 seed 43 loses 10, A\* fold 4 seed 42 keeps
the criterion at 0.7875, and the dev AUROC of B' fold 4 seed 42 falls from 0.9966 to 0.9485 (annex). B fold 3 seed 44,
reported apart, goes from dev accuracy 0.2375 to 0.9950. In the 2×2 count (fitting before × dev after), 14 checkpoints
do not meet the criterion on the fitting set before the re-estimation and meet it on the dev after: the ten, the model
apart and three control checkpoints of arm B' (B' fold 1 seed 43, fold 2 seed 42, fold 3 seed 42); the other 22 meet it
in both. The run reads this as its pre-written case "all": "on these checkpoints, re-estimating the statistics on the
fitting set, every trained parameter unchanged, is sufficient to meet the criterion"; the moving statistics, dropout and
the augmentation of the fitting set (hypotheses 1–3 of the note) are not separated, the later saturated epochs are not
examined, and a recovery shows the effect of this re-estimation on these checkpoints, not a single cause of the
saturation (`REPORT.md` of PR #27).

*(ii) Test (PR #28), a post hoc diagnostic extension on a test already analysed, not an independent validation;
threshold fixed at 0.5.* The baseline passed for all 36, and the values before equal those published with the parent
run. Primary reading: for the ten, on the whole test, the mean ACER goes from 0.5000 to 0.2013, a paired difference of
−0.2987 with subject-bootstrap interval [−0.4033, −0.2147], entirely below zero; in the run's pre-written words, "on
this test, the effect of this re-estimation on these checkpoints is a lower mean value". It is lower for 9 of the ten
and higher for one, B fold 4 seed 44 (0.5000 → 0.5236; its APCER goes from 0.0000 to 0.5370, its BPCER from 1.0000 to
0.5103). In session 03 the difference for the ten is −0.2246 [−0.3759, −0.1425], with 8 lower and 2 higher (B fold 4
seed 44, 0.5000 → 0.7424; B' fold 4 seed 44, 0.5000 → 0.5096, whose session-03 AUROC falls from 0.5000 to 0.2328). For
the ten, the BPCER falls (−0.5568 [−0.7246, −0.4507]), the APCER is higher for 8 of them with an interval containing
zero, and the AUROC is higher for all ten (+0.3790 [0.2635, 0.4344]). Unfavourable results: for the 25 control
checkpoints the ACER interval contains zero (+0.0082 [−0.0220, 0.0316]) and the ACER is higher for 18 of 25; their APCER
is higher (+0.0409 [0.0026, 0.0714], an interval above zero); of the 17 control checkpoints with a worsened indicator on
the dev, 14 have a higher ACER on the test. By arm, A\* has every interval on the favourable side, on the test and in
session 03; for B the ACER interval is below zero on the test and contains zero in session 03; for B' it contains zero
in both; in B and B' the APCER is higher, with intervals above zero in both scopes. Over all 36, ACER −0.0922 [−0.1399,
−0.0533], lower for 17 and higher for 19. The session-03 intervals rest on 9 held-out subjects with bona fide frames and
on 650 valid resamples of 1,000 for ACER (926 on the whole test); with so few clusters a percentile interval may be too
narrow (limits of the run). The model apart has no interval (one fold, four subjects): ACER 0.7265 → 0.1891. Every model
and scope whose ACER, APCER or BPCER is higher, or whose AUROC is lower, is listed with its size in the run's
`aggregate.md`. Time: 3,027.4 s of elapsed time (estimate 42 minutes, cap 240); the process CPU time over all threads
was 18,438.4 s, more than the four hours of CPU time set by the coordinator, which the note had applied as elapsed time
(`REPORT.md` of PR #28); the final `status.json` lacks the `started` field (a defect of the runner), so the start time
rests on `run/run.log.txt`.

*Conclusion.* Dev: on these 36 checkpoints, with this estimator, the ten have accuracy 0.5000 on dev and fitting set
with their stored statistics, and all ten meet the recovery criterion once the statistics are re-estimated on the
fitting set, every trained parameter unchanged; the control checkpoints keep the criterion, and 17 of the 25 have at
least one worsened indicator. Test (post hoc, at 0.5): the mean ACER of the ten is lower after the re-estimation, with
one of the ten higher; the control checkpoints show no direction on ACER (interval containing zero, 18 of 25 higher) and
a higher APCER; arms B and B' also have a higher APCER. Neither run separates the lag of the moving averages, dropout
before BatchNormalization and the augmentation of the fitting pipeline, and neither says anything about other
checkpoints, thresholds or data.

## 4. Reading and limits

Two lists: what the accepted runs show, each item with its scope, and the hypotheses and questions they leave open.
Nothing in the second list has been tested by any of our runs.

**(i) What the results show**

- Same sessions and session change (C1, M1, N1, N2, crop check; NUAA, the official session split, these NUAA-trained
  checkpoints and recipe, thresholds 0.5 and the M1 dev threshold). Within the training sessions the recipe makes no
  error on the frames it never used, including the 2 subjects it never fitted; across the session change it fails, for
  the subjects it was trained on as well as for new ones (N1 on the 3,000 test frames: ACER 0.4010–0.5143, AUROC
  0.3485–0.4582; §3.1–§3.2). The design difference between the two readings is
  the session shift. The failure is not explained by (accepted runs): the training seed (N1 range), the composition of
  the 300 (N1 part 1), the threshold choice (M1: the dev does not identify a unique threshold, dev EER 0, dev AUROC 1),
  the 1 px box (crop check), the numerical path (CPU/GPU parity within 1e-5, decisions unchanged). Scope and limits:
  image-level counts with correlated frames; three seeds give a range, not a confidence interval; one dataset and one
  attack type (print); NUAA is not subject-disjoint by design; the 12 September checkpoints come from a script that
  predates the repository; the historical preparation of the combined model is not documented (the two variants bracket
  it); the C1 subset is one session; the source-dev threshold is not uniquely identified (every threshold between the
  two dev classes gives EER 0) and the dev gives no guarantee of transfer to session 03.
- Subjects never seen, sessions represented (§3.5–§3.6; four subject folds, three seeds, the N1 recipe). With the three
  sessions represented in training, the recipe reaches on the held-out subjects a pooled AUROC of 0.9377–0.9998 over
  strata and seeds; session 03 remains the stratum with the highest error (pooled ACER 0.0975–0.1515 at 0.5), and the
  results vary across folds and seeds (fold 1 seed 43, fold 4, §3.7). On the same 3,000 session-03 frames as N1 the
  difference from N1 remains for every seed (ACER at 0.5 −0.3643 to −0.3010, AUROC +0.5088 to +0.5900, paired bootstrap
  intervals not containing zero), with different training roles, so it is associated with the whole design, not with one
  factor, and is indicative only. Scope and limits: the pooled AUROC is computed on the scores of four models, not of
  one detector; three seeds give a range, and the bootstrap over subjects is the only interval; the dev of a held-out
  fold is the test fold of the next one, so the four fold results are not independent (no leakage); session 03 is
  under-sampled in fitting (about 18–25 % of its bona fide and 11–13 % of its attacks); data and roles differ from N1
  (sessions 01–03 against 01–02 in training, 2,100 against 2,507 fitting frames, a dev fold of 4 subjects against 2
  early-stopping subjects, every frame of the held-out subjects against the 3,000 sampled session-03 frames).
- Recipe stability (§3.8–§3.10; the 36 trainings of the session ablation design, three seeds per fold). With patience 4,
  10 of 36 trainings were degenerate; with patience 12, with or without the learning-rate reduction, the ten recover and
  none of the 36 is degenerate; the two 12-epoch regimes support the longer patience for these 36 trainings on
  train/dev, and none of the regimes identifies why the dev saturates; the primary comparison A\* − B is inconclusive in
  all three regimes. The result of the constant-rate control is mixed (the ten recover at a constant rate, earlier in 8
  of them, but B' fold 2 seed 42 loses its marginal recovery), with no overall verdict. The recovery holds at the last
  epoch in 28 (p12) and 26 (p12c) of 36 models only; B' fold 4 seed 42 relapses in all three regimes; the restored epoch
  changed in 15 of 36 models from p4 to p12 and in 23 of 36 from p12 to p12c, with 31 of 36 different checkpoints, so
  the test differences between the runs come from those models, not from the recovery of the ten alone, and are not
  evidence for or against any regime. Scope and limits: the criteria (degenerate, saturated epoch, recovery) were
  defined after the patience-4 run, from its cards, and the degenerate criterion misses at least one failure of the same
  family (B fold 3 seed 44 with patience 4); the recovery criterion is a threshold on one dev epoch (B' fold 2 seed 42
  fails it at the constant rate by its `val_loss` alone); the dev of 2–3 subjects (400 frames per fold) drives early
  stopping, the learning-rate reduction where present, the checkpoint and the threshold, and with patience 12 it chooses
  the checkpoint among 12 epochs instead of 5–12, a more optimistic selection on the dev; the constant rate tests the
  absence of the reduction, not the level of the rate; with patience 12 the recipe is no longer the one of the folds run
  (patience 4), which stays a reference, not a paired control; one dataset, one attack type (print), three seeds per
  fold, folds not independent.
- BatchNormalization statistics, dev (§3.11, PR #27; these 36 checkpoints, this estimator, the dev of each fold). With
  their stored statistics the ten have accuracy 0.5000 on dev and fitting set; with the moving mean and variance
  re-estimated on the fitting set, every trained parameter unchanged, all ten meet the recovery criterion on the dev
  (dev accuracy 0.8100–1.0000). The 25 control checkpoints keep the criterion, and 17 of them have at least one worsened
  dev or fitting indicator (three lose 31–32 of 400 correct dev frames). Scope and limits: only the restored checkpoint
  of each training is saved, so the later saturated epochs are not examined; the checkpoints were selected on the same
  dev; small dev (2–3 subjects per fold); one estimator; criteria defined after the patience-4 run.
- BatchNormalization statistics, test (§3.11, PR #28; a post hoc diagnostic extension on a test already analysed, not an
  independent validation; these 36 checkpoints, this test, the fixed threshold 0.5). The mean ACER of the ten is lower
  after the re-estimation (0.5000 → 0.2013, −0.2987 [−0.4033, −0.2147]), lower for 9 of them and higher for B fold 4
  seed 44 (0.5000 → 0.5236; in session 03 also B' fold 4 seed 44); the APCER of 8 of the ten is higher (interval
  containing zero). For the 25 control checkpoints the ACER interval contains zero and the ACER is higher for 18 of 25,
  and their APCER is higher (interval above zero); in arms B and B' the APCER is higher (intervals above zero, on the
  test and in session 03). Scope and limits: the values before were published and the re-estimation was chosen on the
  dev of the same folds; one estimator, one threshold, the restored checkpoints only; the intervals rest on 16 held-out
  subjects (in session 03, 9 with bona fide frames, and 650 valid resamples of 1,000 for ACER) and may be too narrow;
  the summaries share models and subjects; the model apart has no interval. Process: the CPU time over all threads
  (18,438.4 s) exceeded the four CPU hours set, which the note had applied as elapsed time; the final `status.json`
  lacks `started`, so the start time rests on `run.log.txt`.

**(ii) Hypotheses and questions still open**

- Why the recipe fails on session 03. The result is consistent with a session-shift explanation and does not prove it;
  the weak validation (2 subjects, sessions 01–02, near-zero loss) remains an untested alternative explanation (N1, N2).
  The folds run is consistent, as declared before it, with a failure due to "session not represented, or training data
  too narrow" rather than to the session alone: adding session 03 to training also adds variety, and that design does
  not separate the two. Not tested: the effect of the GPU defect on the training of the 12 September LivenessNet
  checkpoint (its C1 scores are not affected: eager scoring, and the CPU rescoring agrees, N1; N1 retrained on the CPU
  and found the same picture), the enhancement chain, the recipe itself (the folds run, §3.5, used the same recipe with
  every session represented).
- Whether representing session 03 in training changes the error on held-out subjects at fixed fitting subjects: the
  primary comparison is inconclusive in every regime, and the BatchNormalization control says nothing new about it.
- Which mechanism produces stored statistics with which the ten reach only accuracy 0.5000 (the note of PR #22 lists
  seven possible explanations of the early dev saturation, none tested as a cause). Three candidates, each consistent
  with §3.11 and none separated by our controls, because the re-estimation changes all three at once: the moving
  averages lagging the weights at an epoch-1 or epoch-2 checkpoint (momentum 0.99); the variance of the inputs of the
  two BatchNormalization layers that follow a dropout layer, estimated with dropout active and used without it; the
  augmentation of the fitting pipeline, absent from the re-estimation (hypotheses 1–3 of the note of PR #27). The
  momentum, the learning rate and the initialisation are named there as possible later controls, none agreed.
- Why the dev is at 0.5 in later epochs too (the note of PR #27 counts 24 of the other 26 models at 0.5 at epoch 3 or
  later): only the restored checkpoint of each training is saved, so the re-estimation could not be tried there.
- Why the same re-estimation worsens 17 of the 25 control checkpoints on the dev, raises their APCER and that of arms B
  and B' on the test, and raises the ACER of one of the ten (B fold 4 seed 44); whether a statistic other than the
  global moments of the fitting set would avoid it.
- Whether the learning-rate reduction contributes to the recovery with patience 12 (mixed result, no overall verdict).
- Whether any of this, the degenerate trainings or the effect of the re-estimation, occurs with other data, other
  subjects, other architectures or other recipes (a longer patience, another momentum, dropout placed after the last
  BatchNormalization), and whether a recipe change prevents it rather than repairing it afterwards.

## 5. State on 17 November

**Done.** The subject-held-out folds run and the step zero were accepted by 4 November (PR #16 to #18; §2, §3.5). The
session ablation was proposed on 4 November (PR #20, merge `3f682a4`), authorised on 5 November and run with patience 4
(PR #21, merge `67f641c`; §2, §3.8). Its ten degenerate trainings were studied from the train/dev curves, with one
change proposed and agreed on 7 November (PR #22, merge `12dc78a`). The change was verified on the same 36 models and
the comparison accepted on 10 November (PR #23, merge `ec6c023`; §3.9); the update of the synthesis of that day (PR #24,
merge `1e340fc`) proposed one control, run and accepted on 11 November (PR #25, merge `a1ca91f`; §3.10), and the
synthesis was updated with the three regimes (PR #26, merge `5a2087b`). One control on the early dev saturation, the
re-estimation of the BatchNormalization statistics of the 36 patience-4 checkpoints, was proposed on 12 November,
approved with three precisions on 13 November, run on 14 November and merged on 16 November (PR #27, merge `c234064`;
§2, §3.11). The same re-estimation on the test, assigned on 16 November as a post hoc diagnostic extension on a test
already analysed, not an independent validation, was published before the run in PR #28 and run on 17 November (run
`20261117-BNTEST-6245b10`, PASS; §3.11); PR #28 is open, and this update of the synthesis is a later commit of it. This
update runs no training and no inference and adds no seed and no model. The extension to the other session directions
stays left out (coordinator, 3 November).

**Recorded deviations of the test run (PR #28).** The process CPU time over all threads was 18,438.4 s (5.12 hours)
against the limit of four CPU hours, which the note had applied as elapsed time (cap 240 minutes; the run took 50.5
minutes); the final `status.json` has no `started` field, so the start time is read from the run log. After the
coordinator's review of 17 November the run is neither altered nor repeated for these two points; the protocols that
follow state the unit of every time limit (elapsed or CPU time) and record the start in the final status.

**Not started.** The independent verification proposed in §6 is a proposal only, to be agreed before any data are
prepared or any model is trained; no other control is proposed. Of the possible next steps listed on 11 November (from
`CONCLUSION.md` of the constant learning-rate control): the check of another candidate cause of the saturation was
taken up for BatchNormalization (PR #27 and PR #28), not for the level of the rate; more seeds per fold were not
taken up for NUAA (the proposal of §6 uses 36 trainings on another dataset); the rule, fixed before any run, for the
trainings whose recovery does not hold at the last epoch is carried into the proposal of §6.

## 6. Towards the article

**(a) Question and contribution.** Question: in a small CNN with BatchNormalization and dropout, early-stopped on a
small subject-disjoint dev, how often does face PAD training end at a checkpoint at chance in inference mode, and do its
trained weights separate the classes once the stored moving statistics are replaced? Supported, on NUAA, LivenessNet,
this recipe and these 36 trainings only: (1) with patience 4, 10 of 36 trainings met the degenerate criterion (defined
after that run; it misses at least one failure of the same family, §3.8), each at accuracy 0.5000 on its dev and its
fitting set with the stored statistics; (2) with only the moving mean and variance re-estimated on the fitting set,
every trained parameter unchanged, all ten met the recovery criterion on the dev that had selected them (accuracy
0.8100–1.0000); (3) the re-estimation is not neutral: 17 of the 25 control checkpoints had a worsened dev or fitting
indicator, and on the test (a post hoc diagnostic extension on a test already analysed, not an independent validation)
their ACER was higher for 18 of 25 (interval containing zero) and their APCER higher (interval above zero), as in arms B
and B', and among the ten the APCER was higher for 8 (interval containing zero) and the ACER for one (two in session 03)
(§3.11). The session-shift results (§3.1–§3.10) are not part of this claim. Not claimable: the mechanism (§4 (ii)); the
operation, or recomputing the statistics before evaluating a non-converged model, as new (Wu and Johnson, Sec. 3.3); any
frequency beyond these 36 trainings; any benefit for other checkpoints, thresholds or data; the test results as a
validation.

The question of the article is the coordinator's choice. (A) The case above: it has §3.8–§3.11 on NUAA; it still needs
the same count and check on a second dataset, with every reading fixed before any result. (B) The session-shift results
(§3.1–§3.10), with the BatchNormalization finding as a methodological caution: it has C1 to N2, the folds run and the
session ablation; it still needs to separate the session from the weak validation and from the variety of the training
data (§4 (ii)), and its primary comparison is inconclusive in every regime. (C) Both, as one case study on evaluation
validity for small PAD datasets: it has all of §3 and needs what (A) needs and a framing that keeps the two questions
apart. The verification in (c) serves (A) and (C); none is proposed for (B).

**(b) Directly pertinent works** (re-read at the primary source; numbering of the version named). Dataset reference:
Tan, Li, Liu, Jiang, ECCV 2010, pp. 504–517, DOI 10.1007/978-3-642-15567-3_37.

| work | what it shows (where) | relation to §3.11 | what we add; what we cannot claim |
|---|---|---|---|
| Ioffe, Szegedy, ICML 2015, PMLR 37:448–456 (arXiv:1502.03167) | at inference, normalisation with population statistics averaged over training mini-batches, parameters frozen; moving averages named as a way to track accuracy during training (Sec. 3.1, Algorithm 2) | our re-estimation is such a population estimate (divisor N; Algorithm 2 uses m/(m−1) times the mean batch variance) | nothing on the operation, which is the original definition |
| Li, Chen, Hu, Yang, CVPR 2019, pp. 2677–2685, DOI 10.1109/CVPR.2019.00279 (arXiv:1801.05134 v1) | dropout before BatchNormalization shifts the variance between train and test mode (Sec. 3); with all parameters fixed, an accuracy gap on the training data between train and evaluation mode (Sec. 4, Fig. 5: PreResNet and DenseNet, CIFAR-10/100); re-running the moving-average estimate on the training data in test mode, all other parameters fixed, lowers the test error of every dropout-0.5 model (Table 3, CIFAR-10/100, 5 runs) | a related, much milder symptom and the same remedy; in LivenessNet the two BatchNormalization layers that follow a dropout layer do so through a convolution or dense layer and a ReLU, and the last dropout is followed by no BatchNormalization | a count under early stopping in face PAD, the harm on control checkpoints; not the symptom, not the remedy, not dropout as our cause |
| Wu, Johnson, 2021, arXiv:2105.07576 (technical report) | moving-average statistics fall outside the distribution of batch statistics, mostly early in training, and lag the current weights (Sec. 3.1, Fig. 2; their momentum 0.9, ours 0.99); statistics from one fixed model state with equally weighted samples are the original formulation, to be used when a non-converged model is evaluated (Sec. 3.2–3.3); ResNet-50 on ImageNet | the ten were restored at epoch 1 or 2; our estimator is of this kind | a small-data PAD case; not the lag as our cause, not the advice to recompute before evaluating a non-converged model (Sec. 3.3); recomputing on the evaluation inputs with inference preprocessing instead of training augmentation made their error slightly worse (Sec. 5.1, Table 4), which bears on the worsened control checkpoints |
| Izmailov, Podoprikhin, Garipov, Vetrov, Wilson, UAI 2018 (arXiv:1803.05407) | after the weights are averaged, one additional pass over the data computes the BatchNormalization statistics for the averaged weights, which training does not collect (Sec. 3.2, Algorithm 1) | the same kind of operation (one more pass over the data to recompute the statistics, in training mode in their description; ours in inference mode, layer by layer), used when weights and stored statistics do not match | another occasion for it; not the operation |
| Kuznetsov et al., Multimedia Tools and Applications 85(3), 2026, article 222, DOI 10.1007/s11042-026-21445-w (arXiv:2508.09094 v1) | LivenessNet, the architecture as described there (originally Rosebrock, 2019): convolutional blocks with BatchNormalization after each ReLU, each block ending in dropout, then dense, ReLU, BatchNormalization, dropout (Sec. 3.2.1); 3DMAD, CSMAD, MSSpoof, Replay-Attack and an own dataset, not NUAA (Sec. 3.1); average accuracy 99 % within and 52 % across datasets, ACER 0.6 % and 47.7 % (Sec. 4.4, item 2) | source of the architecture; their recipe: patience 15–20 within 20 epochs, rates 5e-9 to 1e-6, batch 8 (Sec. 3.4) | a failure observed with our recipe (patience 4 within 12 epochs, Adam 1e-3, batch 16), which differs from theirs; nothing about their models or datasets; their averages are not NUAA results |
| Chingovska, Anjos, Marcel, BIOSIG 2012 (Idiap publication) | the NUAA protocol has training and test data but no development data; 15 identities; CASIA-FASD inherits the lack of a complete protocol (Sec. II); for NUAA, folds of the training data serve as development set (Sec. V) | our dev is carved from 2–3 subjects per fold and drives stopping, checkpoint and criteria | nothing on the protocol; their 15 identities against the 16 subjects of our copy (M2) is not resolved here |

**(c) What is missing, and one independent verification.** Missing: the same count and check on other data, with every
reading fixed before any result (the test reading of §3.11 is post hoc); the mechanism, not proposed now. A proposal
only, to be pre-registered (note, configuration, code) before any data preparation and agreed before any training:
CASIA-FASD. Declared differences from NUAA, seed variant and cost derivation: package README.

- Already known, not used for any choice (pilot README, `research/m2-c2-pilot-protocol/README.md`): the 12 September
  CASIA-FASD LivenessNet checkpoint of C1, from a script older than the repository, "stopped after 5 epochs with
  patience 4"; the dev accuracies of the LivenessNet and AttackNet V2.2 CASIA-FASD checkpoints equal one class share
  (71/316, 245/316), "consistent with, not proof of, a constant decision at 0.5"; the LivenessNet card gives APCER
  1.000, BPCER 0.000 on the CASIA-FASD test. The count below is therefore not blind. Re-estimating that checkpoint needs
  no training, but its provenance is weak; it is not the proposal.
- Access. In our local data: an unofficial mirror copy, frames only, 4,063 colour frames (995 bona fide, 3,068 attacks),
  50 subject numbers local to the two splits (20 + 30), 12 videos per subject; manifests merged with PR #5
  (`research/m2-rgb-manifests/`, README and aggregate); the 21 restricted manifest CSV are received and verified (21
  hashes, 12 splits reconstructed, 57 tests: C2 card description, read on 17 November 2026; the pilot README, written
  earlier, still marks that check as pending). Open: the canonical deposit and the final approval of M2; that the two
  splits hold different persons (documented, not verified); use for a new run, since the agreement of 9 October 2026
  is "to be signed through the University" and decision D-4 is open (pilot README): the coordinator's. Rejected:
  CelebA-Spoof (identity overlap not measurable in our copy); 3DMAD, CSMAD, MSSpoof, Replay-Attack (not in our local
  copies; access would depend on the coordinator); a mechanism control on NUAA (not independent).
- Protocol. Four subject-disjoint folds of 12–13 subjects with the fold rule of §2; test = every frame of the held-out
  fold; fitting and dev balanced by class with the spread rule, the attack frames of each subject spread over its nine
  attack videos, because the criteria of PR #22 read a constant prediction as dev accuracy 0.5 (the dev of the 12
  September checkpoints was unbalanced, 71 and 245 of 316). LivenessNet, patience-4 recipe unchanged; 36 trainings,
  seeds 42–50. Readings: the number of degenerate checkpoints over 36, by fold and seed, as a frequency; the estimator
  and reading of PR #27 on every checkpoint (confirming: all degenerate checkpoints meet the recovery criterion after
  the re-estimation; not confirming: some or none; none degenerate: nothing said on the re-estimation); a rule fixed
  before any run for trainings recovering at some epoch but not at the last one (reported apart, with both epochs); the
  test reading of PR #28 and the effects on the control checkpoints descriptive; persons shared across the two splits,
  if found, reported as such. A confirmation would show the effect on a second dataset, not a mechanism; a
  non-confirmation would not exclude BatchNormalization or dropout in general.
- Cost (elapsed time, from our measured runs): about 11,400 s (3.2 hours), up to about 20,000 s (5.6 hours), with all
  432 epochs at the slowest training rate measured. Process CPU time over all threads is not recorded for our trainings
  and is not estimated; PR #28 alone used 18,438.4 s (5.1 CPU hours, 6.1 times its elapsed time), above the four CPU
  hours set for it. The cap and its unit are for the coordinator to set.

Remaining activities, in order, if (A) or (C) is chosen: the CASIA-FASD agreement and decision D-4, and the canonical
deposit and final approval of M2 (the coordinator's); the pre-registration of (c), after these; the verification and
its report; figures and tables; writing.