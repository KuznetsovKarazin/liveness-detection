# NUAA: training with and without session 03, protocol proposal

Asked by the coordinator on the C2 card on 4 November 2026. Proposal only: no training before it is agreed.

**Question.** Does representing session 03 in training change the error on held-out subjects, at fixed fitting
subjects, total count, test, preprocessing and recipe?

**Design.** The four subject folds of the approved schema. Held-out fold k: test = its 4 subjects, every frame, strata
01/02/03 (unchanged); dev = fold k mod 4 + 1; fitting subjects S12 = the subjects of the two fitting folds with frames
in sessions 01-02: 4/5/5/4 per fold, of which 1/1/2/2 with session-03 bona fide and 4/5/5/4 with session-03 attacks.

| Arm | Fitting set (cells = session x class; bf = bona fide, att = attack) | Frames |
|---|---|---|
| B, without 03 | sessions 01, 02 of S12, 350 per cell x 4 cells | 1,400 |
| A\*, with 03 | sessions 01, 02, 03 of the same S12; 234/233/233/234/233/233 (01/bf, 01/att, 02/bf, ...) | 1,400 |
| A, reference | the completed folds run (`20261103-FOLDSRUN-e174812`): 350 x 6 cells, all 8 fitting subjects | 2,100 |

- 350 = the largest multiple of 50 not above the smallest availability over cells and folds (361). Cells are sampled
  with `spread` (equally spaced items of the sorted file names): the session 01-02 frames of A\* are nested in those
  of B; its 03 cells spread over S12. The 03/bf cell of A\* comes from 1 subject in folds 1 and 2, 2 in folds 3 and 4.
- B is exactly the session 01-02 part of A's fitting set, the common dev the session 01-02 part of A's dev (same
  frames): A - B adds 700 session-03 frames of all 8 fitting subjects and a dev with 03; A is not rerun (indicative).
- **Dev, common to both arms, without the test**: sessions 01-02 of the dev-fold subjects, 100 per cell x 4 = 400 (more
  than 100 frames available in every cell). As in the folds run, the dev drives early stopping, the learning-rate
  reduction (ReduceLROnPlateau), the restored epoch and the EER (equal error rate) threshold, frozen with SHA-256 before
  any test scoring; 0.5 also reported. No session-03 information in either arm's threshold (one factor at a time).
  Declared alternative, not used: "dev-mix", sessions 01-03 of the dev subjects, 67/67/66/67/67/66 over 6 cells (400).
- **Fixed**: frozen extractor and cache (no new extraction), the recipe of the folds run (LivenessNet, at most 12
  epochs, batch 16, Adam 1e-3, early-stopping patience 4, CPU), seeds 42/43/44, test sets of the schema.
  Pre-commitment: every outcome reported; nothing changed or chosen after a score is read.
- **Outcome**: per arm, fold, seed and stratum: APCER, BPCER, ACER at 0.5 and at the dev threshold, AUROC. Paired
  differences A\* - B per fold and seed (same test frames; same seeds, nominal pairing: batch order and augmentation
  differ with the data); pooled over the 4 folds per seed, with a bootstrap of the difference over the held-out
  subjects of the stratum (16 in stratum 03, 9 in 01 and 02; 1,000 resamples, seed 0, percentiles 2.5-97.5), added
  to the run configuration before any training. Primary: AUROC and ACER at the dev threshold in stratum 03; a
  difference is read as consistent only if it has the same sign for all three seeds and its bootstrap interval
  excludes 0, otherwise inconclusive; strata 01 and 02 secondary; no statistical test. Diagnostic, not causal: the
  design varies the session composition at fixed subjects and total; the per-cell count of sessions 01-02 changes
  too (A\* 233-234, B 350): B' separates it.
- **Optional, only if the coordinator wants them**: B' (12 trainings) = sessions 01-02 of S12 at the per-cell count of
  A\* (234/233/233/234 = 934, nested in B); A\*\* = A\* with session-03 bona fide from all fitting subjects.

**Counts available** (frames). Fold: held out (dev; fitting folds). S12: subjects (with 03/bf, with 03/att). R1 = 1
subject, count withheld; R1s = 1 subject, count printed in the fold schema; both with the frames the cell must supply.

| Fold | S12 | 01/bf | 01/att | 02/bf | 02/att | 03/bf | 03/att | dev 01/bf | 01/att | 02/bf | 02/att | Test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (2; 3+4) | 4 (1, 4) | 401 | 381 | 493 | 400 | R1 (>= 233) | 1,632 | 234 | 187 | R1s (>= 100) | 193 | 3,242 |
| 2 (3; 1+4) | 5 (1, 5) | 379 | 481 | 491 | 500 | R1 (>= 233) | 2,051 | 276 | 187 | 246 | 200 | 3,114 |
| 3 (4; 1+2) | 5 (2, 5) | 488 | 474 | 361 | 493 | 945 | 1,913 | R1s (>= 100) | 194 | 247 | 200 | 3,439 |
| 4 (1; 2+3) | 4 (2, 4) | 510 | 374 | 363 | 393 | 922 | 1,494 | 254 | 287 | 244 | 300 | 2,819 |
| **Chosen B/A\*** | | 350/234 | 350/233 | 350/233 | 350/234 | -/233 | -/233 | 100 | 100 | 100 | 100 | all |

**Limits.** 03/bf of S12 from 1-2 subjects per fold; two dev cells of one subject; correlated frames (n counts frames);
three seeds give a range; the four folds are not independent; the dev without session 03 calibrates early stopping
and the EER threshold on the session 01-02 distribution for both arms, which can penalise A\* in stratum 03 at the dev
threshold: AUROC, threshold-free, is the cleaner reading; one dataset, one attack type (print).

**Budget.** 24 trainings (2 arms x 4 folds x 3 seeds) on 1,400 fitting + 400 dev frames, scaled by 1,800/2,700 from the
folds run (705.8-842.7 s): 470.5-561.8 s each; 101.7 (mean) to 112.4 (maximum) min per arm. Scoring 75,684 test + 9,600
dev frames: 10.3-12.7 min. Total 213.8-237.4 min; cap 360 min; no extraction (frames already extracted). B': 80.6-89.6
min; with B' the total is 294.4-327.0 min, under the cap. A\*\*: no counts yet: it changes the fitting subjects (it
leaves the fixed-subject design), must be counted first, and with both optional arms the 360-min cap must be revised.

**Deliverables** as in the folds run: configuration first, check-inputs, public aggregate, restricted outputs, SHA-256.
