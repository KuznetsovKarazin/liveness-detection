# NUAA 2026 — Methods and Results synthesis (C1, M1, N1, N2, C1 crop check)

Synthesis of the NUAA results run between 28 September and 2 November 2026 and merged in `research/pad-2026`
(29 September – 2 November), written on the coordinator's request of 2 November. Every number comes from the public
aggregate of the run named next to it; the unified table (`tables/summary_table.md`; compact form below) is rebuilt by
`nuaa_synthesis.py` from those files and `check` fails if a cell differs. No inference, no training, no new number.

## 1. Question

Restricted to NUAA, the group's question reads: how reliably do detectors trained on NUAA (and one trained elsewhere)
keep their discrimination and their error rates at the chosen threshold when the capture session changes (sessions
01–02 → 03) or the dataset changes (five datasets → NUAA), once splits, metrics and preprocessing are under control. C1
fixed the starting point, M1 the metric module and the source-dev threshold, M2 the manifests and overlap rules, N1 the
seed and subject sensitivity, N2 the session check and the historical combined model, the crop check the one documented
preprocessing difference.

## 2. Methods

**Data and units.** NUAA Photograph Imposter Database (Tan et al., ECCV 2010), public mirror
`akahana/anti-spoofing-nuaaaa` (archive SHA-256 `c0780c6f…`; accepted M2 package `20261006-M2-seed42-36c677a`; counts
read here from the public aggregate regenerated at `5fcc1f8`, PR #6). The official split is a *session* split: training = sessions 01–02 of 9 subjects
(1,743 bona fide + 1,748 print attacks), test = session 03 of 16 subjects (3,362 + 5,761); the 9 training subjects
reappear in the test, 7 subjects are test-only. Sampled sets, versioned as manifests with per-image SHA-256 (15 overlap
rules, PASS: no image shared between any training-side and test-side manifest by key or content): `cnn_train` 2,507
frames of 7 subjects (fitting), `cnn_es` 493 frames of the 2 other training subjects (early stopping), `m1_dev` 574 =
all official training frames of those 2 subjects, `cnn_test` 3,000 (1,500 + 1,500, one every k over the official
lists), `c1` 300 (150 + 150, same rule, inside `cnn_test`), N2 target 491 = official training list minus fitting minus
early stopping (243 + 248, 9 subjects; 81 of them are in `m1_dev`). Unit = image frame; frames of one subject and
session are correlated; no video exists.

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
Replay-Attack, MSSpoof and Our Dataset, never on NUAA, loaded as saved.

**Thresholds and metrics.** Score = probability of attack (label 0 bona fide, 1 attack); attack if score > t, tie =
bona fide. t = 0.5 fixed a priori in C1, N1, N2 and the crop check; in M1 t = the EER threshold of `m1_dev`, frozen
before any test score was read. APCER = accepted attacks / attacks, BPCER = rejected bona fide / bona fide, ACER = their
mean, with denominators; AUROC threshold-free (`auc` in the C1 table, `test AUC` in M1: the same statistic); an absent
class gives NA. The metric module was validated in M1 on edge cases (ties, constant scores, missing class).

**Verification practice.** Configuration hashed before each run; inputs, checkpoints and code bound by SHA-256;
per-image scores restricted, aggregates public; synthetic tests and declared mutations; a reproduction check where a
chain had to be rebuilt (N2: 24 parity keys, difference 0; crop check: 300/300 C1 scores within 1e-5, same decisions);
each package reviewed by the coordinator before merging (C1: recompute from the score cache reproduced; N2 and crop
check: metrics recomputed).

## 3. Results

Unified table, compact form (codes: `T-NUAA` = fitting on `cnn_train`, early stopping on `cnn_es`; `T-COMB` = the five
historical datasets; `S-300` = `c1`; `S-3000` = `cnn_test`; `S-491` = the N2 target; `S-03/9` = the session 03 test
frames of the 9 training subjects, 614 + 924; `E-C1` / `E-CACHE` = the two extractors; `E-WHOLE` / `E-CROP` = the two
preparations of the combined model; `P-REF` = the reference chain, `P-HIST` = the historical chain; `N2-S` / `N2-C` = N2
part 1 / part 2; ckpt = checkpoint; "rejected bf / accepted att." = numerators of BPCER and APCER; full legend in
`tables/summary_table_compact.md`, row ids and sources in `tables/summary_table.md`):

| block | model | training | test set | extractor | prep. | threshold | rejected bf / accepted att. | APCER | BPCER | ACER | AUROC | run | transfer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | LivenessNet (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | 0.5 (fixed) | 109 / 24 | 0.1600 | 0.7267 | 0.4433 | 0.3782 | 20260928-C1-seed42-a80f6f9 | session |
| C1 | AttackNet V1 (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | 0.5 (fixed) | 40 / 42 | 0.2800 | 0.2667 | 0.2733 | 0.7547 | 20260928-C1-seed42-a80f6f9 | session |
| C1 | AttackNet V2.1 (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | 0.5 (fixed) | 115 / 3 | 0.0200 | 0.7667 | 0.3933 | 0.7103 | 20260928-C1-seed42-a80f6f9 | session |
| C1 | AttackNet V2.2 (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | 0.5 (fixed) | 118 / 22 | 0.1467 | 0.7867 | 0.4667 | 0.4985 | 20260928-C1-seed42-a80f6f9 | session |
| M1 | LivenessNet (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | 0.181055 (dev EER) | 118 / 5 | 0.0333 | 0.7867 | 0.4100 | 0.3782 | 20261003-M1-seed42-e67ecc7 | session |
| M1 | AttackNet V1 (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | 0.329237 (dev EER) | 44 / 37 | 0.2467 | 0.2933 | 0.2700 | 0.7547 | 20261003-M1-seed42-e67ecc7 | session |
| M1 | AttackNet V2.1 (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | 0.870329 (dev EER) | 91 / 11 | 0.0733 | 0.6067 | 0.3400 | 0.7103 | 20261003-M1-seed42-e67ecc7 | session |
| M1 | AttackNet V2.2 (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | 0.696933 (dev EER) | 112 / 38 | 0.2533 | 0.7467 | 0.5000 | 0.4985 | 20261003-M1-seed42-e67ecc7 | session |
| N1 | LivenessNet, new CPU training, seed 42 | T-NUAA | S-3000 | E-CACHE | P-REF | 0.5 (fixed) | 1155 / 160 | 0.1067 | 0.7700 | 0.4383 | 0.3485 | 20261024-N1-LivenessNet-fb93496 | session |
| N1 | LivenessNet, new CPU training, seed 43 | T-NUAA | S-3000 | E-CACHE | P-REF | 0.5 (fixed) | 1152 / 391 | 0.2607 | 0.7680 | 0.5143 | 0.3754 | 20261024-N1-LivenessNet-fb93496 | session |
| N1 | LivenessNet, new CPU training, seed 44 | T-NUAA | S-3000 | E-CACHE | P-REF | 0.5 (fixed) | 1131 / 72 | 0.0480 | 0.7540 | 0.4010 | 0.4582 | 20261024-N1-LivenessNet-fb93496 | session |
| N2-S | LivenessNet, N1 ckpt seed 42 | T-NUAA | S-491 | E-CACHE | P-REF | 0.5 (fixed) | 0 / 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 20261029-N2-sessions-2ac3e29 | same |
| N2-S | LivenessNet, N1 ckpt seed 42 | T-NUAA | S-03/9 | E-CACHE | P-REF | 0.5 (fixed) | 422 / 154 | 0.1667 | 0.6873 | 0.4270 | 0.3230 | 20261029-N2-sessions-2ac3e29 | session/9 |
| N2-S | LivenessNet, N1 ckpt seed 43 | T-NUAA | S-491 | E-CACHE | P-REF | 0.5 (fixed) | 0 / 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 20261029-N2-sessions-2ac3e29 | same |
| N2-S | LivenessNet, N1 ckpt seed 43 | T-NUAA | S-03/9 | E-CACHE | P-REF | 0.5 (fixed) | 422 / 359 | 0.3885 | 0.6873 | 0.5379 | 0.3171 | 20261029-N2-sessions-2ac3e29 | session/9 |
| N2-S | LivenessNet, N1 ckpt seed 44 | T-NUAA | S-491 | E-CACHE | P-REF | 0.5 (fixed) | 0 / 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 20261029-N2-sessions-2ac3e29 | same |
| N2-S | LivenessNet, N1 ckpt seed 44 | T-NUAA | S-03/9 | E-CACHE | P-REF | 0.5 (fixed) | 422 / 63 | 0.0682 | 0.6873 | 0.3777 | 0.4569 | 20261029-N2-sessions-2ac3e29 | session/9 |
| N2-C | AttackNet V2.2 combined (coordinator's ckpt) | T-COMB | S-3000 | E-WHOLE | P-HIST | 0.5 (fixed) | 1492 / 222 | 0.1480 | 0.9947 | 0.5713 | 0.2169 | 20261029-N2-combined-247dfd7 | dataset |
| N2-C | AttackNet V2.2 combined (coordinator's ckpt) | T-COMB | S-3000 | E-CROP | P-HIST | 0.5 (fixed) | 1224 / 353 | 0.2353 | 0.8160 | 0.5257 | 0.3706 | 20261029-N2-combined-247dfd7 | dataset |
| CROP | AttackNet V2.1 (C1 ckpt 12/09), CPU batch 16 | T-NUAA | S-300 | E-CACHE | P-REF | 0.5 (fixed) | 121 / 4 | 0.0267 | 0.8067 | 0.4167 | 0.7092 | 20261102-C1CROP-f6747cc | session |
| CROP | AttackNet V2.1 (C1 ckpt 12/09), CPU batch 16 | T-NUAA | S-300 | E-C1 | P-REF | 0.5 (fixed) | 115 / 3 | 0.0200 | 0.7667 | 0.3933 | 0.7103 | 20261102-C1CROP-f6747cc | session |

**3.1 Same sessions as training (01–02).** The three N1 checkpoints make no error on the 491 training-session frames
never used for fitting or early stopping: ACER 0.0000 and AUROC 1.0000 for every seed, on the 410 frames of the 7
fitting subjects and on the 81 frames of the 2 early-stopping subjects alike (N2 part 1). The four C1 checkpoints
separate the 574 source-dev frames of those 2 subjects completely as well: dev AUROC (`dev AUC`) 1.0000 and operational
dev EER 0.0000 for all four (M1), so the dev does not identify a unique threshold. Caveats: the 493 early-stopping frames
chose the saved epoch; 2 subjects; 81 of the 491 are also dev frames.

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

**3.4 Context (same 300 images, C1 table).** Two pre-trained models never trained on NUAA separate the session 03 images
almost perfectly at 0.5 (MiniFASNet ACER 0.0000, CLIP zero-shot 0.0100), as do linear probes on frozen encoders fitted
on the 3,000 sampled training images (DINOv2 0.0000, CLIP 0.0133); hand-crafted features fitted on the same images give
ACER 0.32–0.39, the untrained Fourier rule 0.12 and Depth Anything 0.36 (`tables/context_rows_table.md`). The images are
separable by other analyzers at the same threshold; the failure is specific to these NUAA-trained CNN checkpoints and
their recipe (a reading, not a tested cause; descriptive rows: the probes saw the dev frames, one threshold, no seed).

## 4. Reading and limits

- Within the training sessions the recipe makes no error on the frames it never used, including the 2 subjects it never
  fitted; across the session change it collapses, for the subjects it was trained on as well as for new ones. The
  session shift separates the two readings. The result is consistent with a session-shift explanation and does not prove
  it; the weak validation (2 subjects, sessions 01–02, near-zero loss) remains an untested alternative explanation (N1,
  N2); no causal test has been run.
- Not explained by (accepted runs): the training seed (N1 range), the composition of the 300 (N1 part 1), the threshold
  choice (M1: the dev does not identify a unique threshold, dev EER 0, dev AUROC 1), the 1 px box (crop check), the
  numerical path (CPU/GPU parity within 1e-5, decisions unchanged). Not tested: the effect of the GPU defect on the
  training of the 12 September LivenessNet checkpoint (its C1 scores are not affected: eager scoring, and the CPU
  rescoring agrees, N1; N1 retrained on the CPU and found the same picture), the enhancement chain, the recipe itself.
- Limits: image-level counts with correlated frames; three seeds give a range, not a confidence interval; one dataset
  and one attack type (print); NUAA is not subject-disjoint by design; the 12 September checkpoints come from a script
  that predates the repository; the historical preparation of the combined model is not documented (the two variants
  bracket it); the C1 subset is one session; the source-dev threshold carries no information (dev EER 0).

## 5. Proposed next experiment (to be agreed before any training)

In every accepted session-03 result the two factors change together: session 03 is never in training, and 7 of its 16
subjects are never in training. The proposal keeps one factor fixed and moves the other.

**Recommended: subject-held-out, session-matched check on NUAA.** One architecture (AttackNet V1, the best AUROC and
ACER among the NUAA CNNs in C1, or LivenessNet to stay comparable with N1: one of the two, to be chosen together), the
recipe and preprocessing of N1 (CPU, cache extractor, ≤ 12 epochs, three training seeds). Question: when the capture
session is represented in training, does the recipe generalise to subjects it has never seen? Unit = subject. The 16
subjects go into K = 4 folds, fixed before the run and stratified so that every fold holds both classes in session 03
(in session 03, 9 subjects have bona fide frames and 15 attacks: 8 both, 7 attacks only, 1 bona fide only — N1 part 1,
C2 protocol; only 3 of the 9 training subjects have bona fide frames there, N2; the 7 test-only subjects have no
session 01–02 frames). For each fold: fitting on the frames of the other subjects from all three sessions (a fixed
count per class and session, drawn with the deterministic `spread` rule of M2; cache frames where present, otherwise
extracted with the frozen cache extractor, `face.py` 0ac7143, N2), early stopping and EER threshold on a
subject-disjoint part of those subjects, test on the held-out subjects in every session; per fold and seed, per-session
strata (NA where a class is absent), APCER/BPCER/ACER at 0.5 and at the dev threshold, AUROC, a bootstrap over subjects
as the uncertainty; nothing chosen on the test. Declared as a diagnostic protocol, not a benchmark: it mixes sessions in
training and does not preserve the official NUAA protocol; per-subject counts are unbalanced; frames stay correlated.
Budget: 12 trainings of about the N1 size (LivenessNet: about 8–11 min each on the CPU in N1; AttackNet V1 not timed on
the CPU) plus scoring. Why this one: the subject is the independent unit of NUAA (the frames of one subject in one
session are correlated: consecutive frames of one capture, M2), it addresses the question the accepted runs leave open,
and it needs no new dataset and no CASIA-FASD agreement. Reading caveat: adding session 03 to training also adds
variety, so a low error on the held-out subjects would locate the N1/C1 failure in "session not represented, or
training data too narrow", not in the session alone (the extension below separates the two); a high error would point
at the recipe.

**Step zero, without training or inference.** Per-class and per-session image statistics (brightness, contrast,
saturation, sharpness, face-box size) on the training frames and on the 300 and 3,000 test frames, set against the
per-image scores already delivered with C1, N1 and N2 (restricted folders). An AUROC below 0.5 in session 03 (N1 on
the 3,000; N2 on the 9 training subjects; two of the four C1 CNNs on the 300: LivenessNet 0.3782, AttackNet V2.2 0.4985
at chance level) means that the checkpoints give the session-03 bona fide frames a higher attack score than the
attacks (bona fide ranked above attacks, as in §3.2); this is consistent with a cue that is class-correlated in
sessions 01–02 and reversed in 03. The statistics would say which cue, if any, and sharpen the design above (for
instance illumination normalisation as a factor). It would precede the training, on the coordinator's word.

**Optional extension: session-held-out, every subject of the held-out session present in training.** Train on 01+03
and test on 02, train on 02+03 and test on 01, next to the official 01+02 → 03 already measured by N1: same recipe,
three seeds, 6 more trainings. It would show whether session 03 is special or whether any session change breaks the
recipe; with the recommended check it forms a two-factor design (subject × session).

**To decide together:** the scope (the recommended check alone, or with the extension) and the architecture (AttackNet
V1 or LivenessNet). Complement, not alternative: the C2 pilot (CASIA-FASD → NUAA, independent source units, source-dev
threshold), already protocolled and implemented, awaiting the CASIA-FASD agreement and the coordinator's go-ahead for
the real run, covers the dataset transfer.
