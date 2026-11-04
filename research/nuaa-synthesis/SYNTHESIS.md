# NUAA 2026 — Methods and Results synthesis (C1, M1, N1, N2, C1 crop check, subject-held-out folds run)

Synthesis of the NUAA results run between 28 September and 4 November 2026 and accepted in `research/pad-2026` (29
September – 4 November) or delivered with this update (FOLDS-N1 rows: run `20261104-FOLDSCOMPARE-3b3bcca`, folder
`research/nuaa-folds-n1-compare`), written on the coordinator's request of 2 November. Every number comes from the
public aggregate of the run named next to it (§3.7 also quotes, with their source, values computed read-only from the
restricted files of the folds run); the unified table (`tables/summary_table.md`; compact form below) is rebuilt by
`nuaa_synthesis.py` from those files and `check` fails if a cell differs. No inference and no training; §3.7 also
reports read-only descriptive calculations on the existing restricted scores, which `check` does not verify. Revised on
3 November after the coordinator's review (three precisions: the subject × session cell already measured by N2, the dev
threshold, the provenance of the external models; labels of the generated tables; checksums).

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

## 1. Question

Restricted to NUAA, the group's question reads: how reliably do detectors trained on NUAA (and one trained elsewhere)
keep their discrimination and their error rates at the chosen threshold when the capture session changes (sessions
01–02 → 03) or the dataset changes (five datasets → NUAA), once splits, metrics and preprocessing are under control. C1
fixed the starting point, M1 the metric module and the source-dev threshold, M2 the manifests and overlap rules, N1 the
seed and subject sensitivity, N2 the session check and the historical combined model, the crop check the one documented
preprocessing difference, the subject-held-out run the cell no earlier run measured: subjects never seen, with every
session represented in training.

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
fold; the N1 rows above are the comparison); ckpt = checkpoint; "rejected bf / accepted att." = numerators of BPCER and
APCER; the columns from "dev thr." to "bootstrap" apply to the FOLDS rows only, — elsewhere; full legend in
`tables/summary_table_compact.md`, row ids and sources in `tables/summary_table.md`):

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

## 4. Reading and limits

- Within the training sessions the recipe makes no error on the frames it never used, including the 2 subjects it never
  fitted; across the session change it collapses, for the subjects it was trained on as well as for new ones. The design
  difference between the two readings is the session shift. The result is consistent with a session-shift explanation
  and does not prove it; the weak validation (2 subjects, sessions 01–02, near-zero loss) remains an untested
  alternative explanation (N1, N2); no causal test has been run. - Not explained by (accepted runs): the training seed
  (N1 range), the composition of the 300 (N1 part 1), the threshold choice (M1: the dev does not identify a unique
  threshold, dev EER 0, dev AUROC 1), the 1 px box (crop check), the numerical path (CPU/GPU parity within 1e-5,
  decisions unchanged). Not tested: the effect of the GPU defect on the training of the 12 September LivenessNet
  checkpoint (its C1 scores are not affected: eager scoring, and the CPU rescoring agrees, N1; N1 retrained on the CPU
  and found the same picture), the enhancement chain, the recipe itself (the folds run, §3.5, used the same recipe with
  every session represented; see the fourth reading). - Limits: image-level counts with correlated frames; three seeds
  give a range, not a confidence interval; one dataset and one attack type (print); NUAA is not subject-disjoint by
  design; the 12 September checkpoints come from a script that predates the repository; the historical preparation of
  the combined model is not documented (the two variants bracket it); the C1 subset is one session; the source-dev
  threshold is not uniquely identified (dev EER 0, dev AUROC 1: every threshold between the two dev classes gives EER 0)
  and the dev gives no guarantee of transfer to session 03.
- Subjects never seen, sessions represented (§3.5): with the three sessions represented in training, the N1 recipe
  reaches on the held-out subjects a pooled AUROC of 0.9377–0.9998 over strata and seeds; session 03 remains the stratum
  with the highest error (pooled ACER 0.0975–0.1515 at 0.5), and the results vary across folds and seeds (fold 1 seed
  43, fold 4, §3.7). As declared in the protocol before the run, a low error on the held-out subjects is consistent with
  an N1/C1 failure due to "session not represented, or training data too narrow" rather than to the session alone:
  adding session 03 to training also adds variety, and this design does not separate the two; it does not prove either,
  and no causal isolation has been run. The pooled AUROC is computed on the scores of four different models (one per
  held-out fold), not of one detector; three seeds give a range and the bootstrap over subjects is the only interval;
  the dev of a held-out fold is the test fold of the next one, so the four fold results are not independent (no
  leakage); session 03 is under-sampled in fitting (about 18–25 % of its bona fide and 11–13 % of its attacks). The
  difference from N1 (ACER 0.4010–0.5143 and AUROC 0.3485–0.4582 on the 3,000 session-03 frames, §3.2) is indicative
  only, because data and roles differ (sessions 01–03 against 01–02 in training, 2,100 against 2,507 fitting frames, a
  dev fold of 4 subjects against 2 early-stopping subjects, every frame of the held-out subjects against the 3,000
  sampled session-03 frames); on the same 3,000 frames (§3.6) the difference remains for every seed (ACER at 0.5 −0.3643
  to −0.3010, AUROC +0.5088 to +0.5900, paired bootstrap intervals not containing zero), still with different training
  roles, so it is associated with the whole design, not with one factor.

## 5. Next experiment: state on 4 November

**Cells of the subject × session design.** Before the folds run, every accepted session-03 result had session 03 out of
training and 7 of its 16 subjects never in training. N2 measured seen subjects in the new session (the 9 training
subjects in session 03: BPCER 0.6873 for every seed, AUROC 0.3171–0.4569, §3.2); the folds run measures the opposite
cell, subjects never seen with every session represented in training (§3.5). It remains a diagnostic design, not a
complete causal isolation (reading caveat in §4).

**Done.** The subject-held-out, session-matched check proposed here on 3 November went through three steps, each
reviewed by the coordinator before the next. (1) Architecture and scope decided on 3 November (C2 card): LivenessNet, to
stay comparable with N1, with the recipe and preprocessing of N1. (2) The concrete fold schema (four subject folds,
three seeds, counts per class and session, fitting/dev/test separation, budget) delivered and approved before any
training (PR #16, merge `cee1a6c`, run `20261103-FOLDS-67b7f3d`). (3) The run, authorised on 3 November with the schema
and configuration unchanged, executed on the CPU within the cap and accepted (PR #18, merge `96d3fc1`, run
`20261103-FOLDSRUN-e174812`; §2, §3.5). The step zero (per-class and per-session image statistics of the authorised NUAA
materials against the delivered scores; associations, not causes) was executed and accepted before the run (PR #17,
merge `d341520`, run `20261103-STEPZERO-78ca9e4`).

**This update (4 November).** On the coordinator's request: the results with every fold and seed (§3.5 and the unified
table), the descriptive comparison with N1 on the same 3,000 frames from the existing scores and the thresholds already
frozen (§3.6), and a short analysis of fold 1 seed 43 and of fold 4 from the logs and tables already available, observed
results kept apart from possible explanations (§3.7). No new training, no other model, no new threshold.

**Left out.** The extension to the other session directions (train on 01+03 and test on 02, train on 02+03 and test on
01, next to the official 01+02 → 03 measured by N1; same recipe, three seeds, 6 more trainings) stays left out
(coordinator, 3 November). It would show whether session 03 is special or whether any session change breaks the recipe,
and with the folds run it would form a two-factor design (subject × session). Complement, not alternative: the C2 pilot
(CASIA-FASD → NUAA, independent source units, source-dev threshold), already protocolled and implemented, awaiting the
CASIA-FASD agreement and the coordinator's go-ahead for the real run, covers the dataset transfer.
