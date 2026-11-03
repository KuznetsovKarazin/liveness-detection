# NUAA subject folds: schema of the subject-held-out check (run `20261103-FOLDS-67b7f3d`)

Status: proposed schema, to be fixed with the coordinator before any training; nothing trained, nothing scored, no image read

Subject-held-out, session-matched check on NUAA with LivenessNet (coordinator, C2 card, 3 November 2026): four subject folds, three training seeds, counts per class and session, fitting/dev/test separation and budget, fixed before the training. Diagnostic design, not a benchmark and not a complete causal isolation: it mixes the three sessions in training and does not preserve the official NUAA protocol. Exploratory: the NUAA session-03 results of C1, M1, N1 and N2 are already known; every choice below is a rule fixed before any training, none is chosen on a result.

Strata (subjects): A = 3, B = 6, C = 6, D = 1; folds of 4, 4, 4, 4 subjects (members restricted, SHA-256 `38730dbf9033…`).

**Held-out fold 1 (test), dev fold 2, fitting folds [3, 4]; strata of the held-out subjects: A 1, B 2, C 1; subjects with bona fide in session 03: 2, with attacks in session 03: 3, with frames in sessions 01-02: 3**

| role | 01/bona_fide | 01/attack | 02/bona_fide | 02/attack | 03/bona_fide | 03/attack | total |
|---|---|---|---|---|---|---|---|
| fitting | 350 (3 subj.) | 350 (4 subj.) | 350 (4 subj.) | 350 (4 subj.) | 350 (4 subj.) | 350 (8 subj.) | 2100 (8 subj.) |
| fitting, available | 401 | 381 | 493 | 400 | 1418 | 3165 | 6258 |
| dev | 100 (2 subj.) | 100 (2 subj.) | 100 (1 subj.) | 100 (2 subj.) | 100 (3 subj.) | 100 (4 subj.) | 600 (4 subj.) |
| dev, available | 234 | 187 | 117 | 193 | 1019 | 1364 | 3114 |
| test | 254 (3 subj.) | 287 (3 subj.) | 244 (2 subj.) | 300 (3 subj.) | 925 (2 subj.) | 1232 (3 subj.) | 3242 (4 subj.) |
| to extract (not in cache) | 97 | 102 | 103 | 107 | 758 | 1247 | 2414 |

**Held-out fold 2 (test), dev fold 3, fitting folds [1, 4]; strata of the held-out subjects: A 1, B 1, C 2; subjects with bona fide in session 03: 3, with attacks in session 03: 4, with frames in sessions 01-02: 2**

| role | 01/bona_fide | 01/attack | 02/bona_fide | 02/attack | 03/bona_fide | 03/attack | total |
|---|---|---|---|---|---|---|---|
| fitting | 350 (4 subj.) | 350 (5 subj.) | 350 (4 subj.) | 350 (5 subj.) | 350 (4 subj.) | 350 (7 subj.) | 2100 (8 subj.) |
| fitting, available | 379 | 481 | 491 | 500 | 1478 | 2732 | 6061 |
| dev | 100 (2 subj.) | 100 (2 subj.) | 100 (2 subj.) | 100 (2 subj.) | 100 (2 subj.) | 100 (4 subj.) | 600 (4 subj.) |
| dev, available | 276 | 187 | 246 | 200 | 865 | 1665 | 3439 |
| test | 234 (2 subj.) | 187 (2 subj.) | 117 (1 subj.) | 193 (2 subj.) | 1019 (3 subj.) | 1364 (4 subj.) | 3114 (4 subj.) |
| to extract (not in cache) | 97 | 88 | 76 | 90 | 805 | 1334 | 2490 |

**Held-out fold 3 (test), dev fold 4, fitting folds [1, 2]; strata of the held-out subjects: A 1, B 1, C 1, D 1; subjects with bona fide in session 03: 2, with attacks in session 03: 4, with frames in sessions 01-02: 2**

| role | 01/bona_fide | 01/attack | 02/bona_fide | 02/attack | 03/bona_fide | 03/attack | total |
|---|---|---|---|---|---|---|---|
| fitting | 350 (5 subj.) | 350 (5 subj.) | 350 (3 subj.) | 350 (5 subj.) | 350 (5 subj.) | 350 (7 subj.) | 2100 (8 subj.) |
| fitting, available | 488 | 474 | 361 | 493 | 1944 | 2596 | 6356 |
| dev | 100 (1 subj.) | 100 (2 subj.) | 100 (2 subj.) | 100 (2 subj.) | 100 (2 subj.) | 100 (4 subj.) | 600 (4 subj.) |
| dev, available | 125 | 194 | 247 | 200 | 553 | 1500 | 2819 |
| test | 276 (2 subj.) | 187 (2 subj.) | 246 (2 subj.) | 200 (2 subj.) | 865 (2 subj.) | 1665 (4 subj.) | 3439 (4 subj.) |
| to extract (not in cache) | 106 | 91 | 99 | 92 | 742 | 1573 | 2703 |

**Held-out fold 4 (test), dev fold 1, fitting folds [2, 3]; strata of the held-out subjects: B 2, C 2; subjects with bona fide in session 03: 2, with attacks in session 03: 4, with frames in sessions 01-02: 2**

| role | 01/bona_fide | 01/attack | 02/bona_fide | 02/attack | 03/bona_fide | 03/attack | total |
|---|---|---|---|---|---|---|---|
| fitting | 350 (4 subj.) | 350 (4 subj.) | 350 (3 subj.) | 350 (4 subj.) | 350 (5 subj.) | 350 (8 subj.) | 2100 (8 subj.) |
| fitting, available | 510 | 374 | 363 | 393 | 1884 | 3029 | 6553 |
| dev | 100 (3 subj.) | 100 (3 subj.) | 100 (2 subj.) | 100 (3 subj.) | 100 (2 subj.) | 100 (3 subj.) | 600 (4 subj.) |
| dev, available | 254 | 287 | 244 | 300 | 925 | 1232 | 3242 |
| test | 125 (1 subj.) | 194 (2 subj.) | 247 (2 subj.) | 200 (2 subj.) | 553 (2 subj.) | 1500 (4 subj.) | 2819 (4 subj.) |
| to extract (not in cache) | 78 | 92 | 98 | 94 | 567 | 1438 | 2367 |

Distinct frames used: 12614 of 12614; in the training cache 6000; to extract with the frozen extractor: 6614.

Budget: 12 trainings (4 folds × 3 seeds), about 8-11 min per training on the CPU for 2,507 fitting frames in N1 (wall 496-671 s); 2,100 fitting frames here, so at most about 2.5 h of training; extraction of the frames outside the cache with the frozen extractor, once, before the trainings (the N2 cache check of 6,000 frames and the extraction of 491 ran in minutes); scoring of every frame of the held-out subjects, 12 passes; wall-time cap 6 h, else stop, no result, reported as a failed run
