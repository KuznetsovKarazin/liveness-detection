# NUAA training with and without session 03: available counts (run `20261104-SESSABL-5d89888`)

Status: counts for a proposed protocol, to be agreed with the coordinator before any training; nothing trained,
nothing scored, no image read

Training with and without session 03 on NUAA (coordinator, C2 card, 4 November 2026): a short protocol proposal that
compares the two trainings keeping subjects, sample size, test, preprocessing and recipe fixed as far as possible,
with the dev and the threshold chosen without the test; this package gives the available counts, the limits and the
compute budget. Same four subject folds as the approved fold schema.

## Design

- Arm B (without session 03): 350 frames per cell over 01/bona_fide, 01/attack, 02/bona_fide, 02/attack of the S12
subjects, 1400 in all (largest multiple of rounding_step not above the smallest availability over the cells and the
held-out folds, step 50).
- Arm A* (with session 03, same S12 subjects): 1400 frames, balanced: 01/bona_fide 234, 01/attack 233, 02/bona_fide
233, 02/attack 234, 03/bona_fide 233, 03/attack 233.
- Arm B′ (optional control of the per-cell size): sessions 01-02 of the S12 subjects with the per-cell counts of arm
A*: 01/bona_fide 234, 01/attack 233, 02/bona_fide 233, 02/attack 234 (934 in all).
- Nesting: arm B: spread over the available frames of each session 01-02 cell; arm A* and arm B' session 01-02 cells:
spread over the frames chosen for arm B in that cell (nested in B); arm A* session 03 cells: spread over the available
frames of the S12 subjects.
- Arm A (reference, already run): 350 frames per cell, six cells, all eight fitting subjects.
- Dev (common to both arms): 100 frames per cell over the four session 01-02 cells of the dev subjects, 400 in all;
alternative with the session mix of arm A*: balanced, 01/bona_fide 67, 01/attack 67, 02/bona_fide 66, 02/attack 67,
03/bona_fide 67, 03/attack 66.
- Test: unchanged from the fold schema, every frame of the held-out subjects: 3242, 3114, 3439, 2819 (12614).

Notation: R1 = cell of one subject; R1s = cell of one subject already printed in the fold schema; Rd = derivable by
differencing; >=n = at least the count the design needs. Rule: a frame count is public only if its cell holds at least
two subjects and, with the numbers already public in the fold schema and those admitted here, it makes no new
single-subject quantity derivable by differencing (exact rank check on subject x session x class atoms); a cell of one
subject that the fold schema already prints is not printed here and stays derivable from the schema (R1s).

## Held-out fold 1 (dev fold 2, fitting folds 3 and 4)

S12: 4 subjects (1 with bona fide in session 03, 4 with attacks in session 03); fitting subjects without sessions
01-02 (arm A only): 4; dev subjects with sessions 01-02: 2 of 4; test subjects: 4. Feasible: arm_b yes, arm_a_star
yes, dev_common yes, dev_mix_alternative yes.

| row | 01/bf | 01/att | 02/bf | 02/att | 03/bf | 03/att | 01_02 | all |
|---|---|---|---|---|---|---|---|---|
| S12 subjects with frames | 3 | 4 | 4 | 4 | 1 | 4 | | |
| S12 available | 401 | 381 | 493 | 400 | >=233 R1 | 1632 | 1675 | Rd |
| arm B fitting (subj.) | 350 (3) | 350 (4) | 350 (4) | 350 (4) | - | - | 1400 | 1400 |
| arm A* fitting (subj.) | 234 (3) | 233 (4) | 233 (4) | 234 (4) | 233 (1) | 233 (4) | 934 | 1400 |
| arm B′ (optional) (subj.) | 234 (3) | 233 (4) | 233 (4) | 234 (4) | - | - | 934 | 934 |
| arm A fitting, run (subj.) | 350 (3) | 350 (4) | 350 (4) | 350 (4) | 350 (4) | 350 (8) | 1400 | 2100 |
| all fitting subjects available | 401 | 381 | 493 | 400 | 1418 | 3165 | - | 6258 |
| dev subjects with frames | 2 | 2 | 1 | 2 | 3 | 4 | | |
| dev available | 234 | 187 | >=100 R1s | 193 | 1019 | 1364 | 731 | 3114 |
| dev common (subj.) | 100 (2) | 100 (2) | 100 (1) | 100 (2) | - | - | 400 | 400 |
| dev mix, alternative (subj.) | 67 (2) | 67 (2) | 66 (1) | 67 (2) | 67 (3) | 66 (4) | 267 | 400 |
| test (all frames) | 254 | 287 | 244 | 300 | 925 | 1232 | - | 3242 |

## Held-out fold 2 (dev fold 3, fitting folds 1 and 4)

S12: 5 subjects (1 with bona fide in session 03, 5 with attacks in session 03); fitting subjects without sessions
01-02 (arm A only): 3; dev subjects with sessions 01-02: 2 of 4; test subjects: 4. Feasible: arm_b yes, arm_a_star
yes, dev_common yes, dev_mix_alternative yes.

| row | 01/bf | 01/att | 02/bf | 02/att | 03/bf | 03/att | 01_02 | all |
|---|---|---|---|---|---|---|---|---|
| S12 subjects with frames | 4 | 5 | 4 | 5 | 1 | 5 | | |
| S12 available | 379 | 481 | 491 | 500 | >=233 R1 | 2051 | 1851 | Rd |
| arm B fitting (subj.) | 350 (4) | 350 (5) | 350 (4) | 350 (5) | - | - | 1400 | 1400 |
| arm A* fitting (subj.) | 234 (4) | 233 (5) | 233 (4) | 234 (5) | 233 (1) | 233 (5) | 934 | 1400 |
| arm B′ (optional) (subj.) | 234 (4) | 233 (5) | 233 (4) | 234 (5) | - | - | 934 | 934 |
| arm A fitting, run (subj.) | 350 (4) | 350 (5) | 350 (4) | 350 (5) | 350 (4) | 350 (7) | 1400 | 2100 |
| all fitting subjects available | 379 | 481 | 491 | 500 | 1478 | 2732 | - | 6061 |
| dev subjects with frames | 2 | 2 | 2 | 2 | 2 | 4 | | |
| dev available | 276 | 187 | 246 | 200 | 865 | 1665 | 909 | 3439 |
| dev common (subj.) | 100 (2) | 100 (2) | 100 (2) | 100 (2) | - | - | 400 | 400 |
| dev mix, alternative (subj.) | 67 (2) | 67 (2) | 66 (2) | 67 (2) | 67 (2) | 66 (4) | 267 | 400 |
| test (all frames) | 234 | 187 | R1s | 193 | 1019 | 1364 | - | 3114 |

## Held-out fold 3 (dev fold 4, fitting folds 1 and 2)

S12: 5 subjects (2 with bona fide in session 03, 5 with attacks in session 03); fitting subjects without sessions
01-02 (arm A only): 3; dev subjects with sessions 01-02: 2 of 4; test subjects: 4. Feasible: arm_b yes, arm_a_star
yes, dev_common yes, dev_mix_alternative yes.

| row | 01/bf | 01/att | 02/bf | 02/att | 03/bf | 03/att | 01_02 | all |
|---|---|---|---|---|---|---|---|---|
| S12 subjects with frames | 5 | 5 | 3 | 5 | 2 | 5 | | |
| S12 available | 488 | 474 | 361 | 493 | 945 | 1913 | 1816 | 4674 |
| arm B fitting (subj.) | 350 (5) | 350 (5) | 350 (3) | 350 (5) | - | - | 1400 | 1400 |
| arm A* fitting (subj.) | 234 (5) | 233 (5) | 233 (3) | 234 (5) | 233 (2) | 233 (5) | 934 | 1400 |
| arm B′ (optional) (subj.) | 234 (5) | 233 (5) | 233 (3) | 234 (5) | - | - | 934 | 934 |
| arm A fitting, run (subj.) | 350 (5) | 350 (5) | 350 (3) | 350 (5) | 350 (5) | 350 (7) | 1400 | 2100 |
| all fitting subjects available | 488 | 474 | 361 | 493 | 1944 | 2596 | - | 6356 |
| dev subjects with frames | 1 | 2 | 2 | 2 | 2 | 4 | | |
| dev available | >=100 R1s | 194 | 247 | 200 | 553 | 1500 | 766 | 2819 |
| dev common (subj.) | 100 (1) | 100 (2) | 100 (2) | 100 (2) | - | - | 400 | 400 |
| dev mix, alternative (subj.) | 67 (1) | 67 (2) | 66 (2) | 67 (2) | 67 (2) | 66 (4) | 267 | 400 |
| test (all frames) | 276 | 187 | 246 | 200 | 865 | 1665 | - | 3439 |

## Held-out fold 4 (dev fold 1, fitting folds 2 and 3)

S12: 4 subjects (2 with bona fide in session 03, 4 with attacks in session 03); fitting subjects without sessions
01-02 (arm A only): 4; dev subjects with sessions 01-02: 3 of 4; test subjects: 4. Feasible: arm_b yes, arm_a_star
yes, dev_common yes, dev_mix_alternative yes.

| row | 01/bf | 01/att | 02/bf | 02/att | 03/bf | 03/att | 01_02 | all |
|---|---|---|---|---|---|---|---|---|
| S12 subjects with frames | 4 | 4 | 3 | 4 | 2 | 4 | | |
| S12 available | 510 | 374 | 363 | 393 | 922 | 1494 | 1640 | 4056 |
| arm B fitting (subj.) | 350 (4) | 350 (4) | 350 (3) | 350 (4) | - | - | 1400 | 1400 |
| arm A* fitting (subj.) | 234 (4) | 233 (4) | 233 (3) | 234 (4) | 233 (2) | 233 (4) | 934 | 1400 |
| arm B′ (optional) (subj.) | 234 (4) | 233 (4) | 233 (3) | 234 (4) | - | - | 934 | 934 |
| arm A fitting, run (subj.) | 350 (4) | 350 (4) | 350 (3) | 350 (4) | 350 (5) | 350 (8) | 1400 | 2100 |
| all fitting subjects available | 510 | 374 | 363 | 393 | 1884 | 3029 | - | 6553 |
| dev subjects with frames | 3 | 3 | 2 | 3 | 2 | 3 | | |
| dev available | 254 | 287 | 244 | 300 | 925 | 1232 | 1085 | 3242 |
| dev common (subj.) | 100 (3) | 100 (3) | 100 (2) | 100 (3) | - | - | 400 | 400 |
| dev mix, alternative (subj.) | 67 (3) | 67 (3) | 66 (2) | 67 (3) | 67 (2) | 66 (3) | 267 | 400 |
| test (all frames) | R1s | 194 | 247 | 200 | 553 | 1500 | - | 2819 |

## Budget

24 trainings (2 arms x 4 folds x 3 seeds). Per training: arm_b 1400 fitting + 400 dev frames, 470.5-561.8 s,
101.7-112.4 min for its 12 trainings; arm_a_star 1400 fitting + 400 dev frames, 470.5-561.8 s, 101.7-112.4 min for its
12 trainings. Scoring: 75684 test frames (each model scores its held-out fold only) and 9600 dev frames, 10.3-12.7
min. Total 213.8-237.4 min against a cap of 360 min; times scaled by frames per epoch from the reference run
20261103-FOLDSRUN-e174812.

Optional arm B′ (not in the base budget): 12 trainings of 934 fitting + 400 dev frames, 348.7-416.4 s each, 75.4-83.3
min of training, 5.2-6.3 min of scoring, 80.6-89.6 min in all.

Restricted numbers in this document: 8; subjects with a single-subject quantity already derivable from the public fold
schema: 2 (this document adds none).
