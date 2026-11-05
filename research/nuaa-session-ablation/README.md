# NUAA training with and without session 03: available counts

Counts for the protocol proposal asked by the coordinator on the C2 card on 4 November 2026: compare the training
with and without session 03, keeping subjects, sample size, test, preprocessing and recipe fixed as far as possible,
with the dev and the threshold chosen without the test. This package derives the available counts, the proposed
cell sizes, the limits and the compute budget. Nothing is trained or scored and no image is read: it only reads the
accepted manifests. No training starts before the protocol is agreed.

## Design that the counts quantify

Same four subject folds as the approved fold schema (run `20261103-FOLDS-67b7f3d`). For the held-out fold k:
test = every frame of its subjects (unchanged); dev = fold k mod 4 + 1; fitting = the other two folds.

- **S12**: the fitting subjects with frames in sessions 01-02 (the official training subjects of the two folds).
- **Arm B** (without session 03): sessions 01 and 02 of the S12 subjects, four cells (session x class), n frames per
  cell; n = the largest multiple of 50 not above the smallest availability over cells and folds.
- **Arm A\*** (with session 03, same S12 subjects): six cells, the same total 4n split evenly; the remainder goes one
  frame per cell in a fixed order that keeps bona fide and attacks equal. If a cell were short in some fold it would
  take the smallest availability and the rest would be split again (declared "unbalanced" variant).
- **Arm B′** (optional control of the per-cell size): sessions 01-02 of the S12 subjects with the per-cell counts of
  arm A\* (234/233/233/234, 934 frames); its budget is estimated separately, not added to the base budget.
- **Nested samples**: arm B takes each session 01-02 cell with `spread` over the available frames; the session
  01-02 cells of arm A\* and arm B′ take `spread` over the frames chosen for B (subsets of B, so B′ equals the
  session 01-02 part of A\*); the session 03 cells of A\* take `spread` over the available frames of the S12 subjects.
- **Arm A** (reference, already run as `20261103-FOLDSRUN-e174812`): 350 frames per cell, six cells, all eight
  fitting subjects; reported for comparison, not rerun.
- **Dev**: common to both arms, sessions 01-02 of the dev subjects, 100 frames per cell (same rounding rule); the
  alternative dev with the session mix of arm A\* (six cells, same total) is reported, not proposed.
- `spread` is the rule of M2 and of the fold schema (equally spaced items of the sorted file names).
- The configuration states n, the arm A\* allocation and the dev sizes; the script recomputes them from the rules
  and fails if they differ.

## Disclosure rule of the public tables

No subject identifier, no image key. A frame count is printed only if its cell has at least two subjects and, with
the numbers already public in the fold schema and those already admitted, it makes no new single-subject quantity
derivable by differencing (exact rank check on subject x session x class atoms). Withheld numbers are marked `R1`
(cell of one subject), `R1s` (cell of one subject already printed in the fold schema) or `Rd` (derivable by
differencing), with `>=n` where the design needs n frames. Everything is in the restricted folder.

## Contents

| Path | What |
|---|---|
| `livenesslab/scripts/nuaa_session_ablation.py` | `check-inputs` (writes nothing) and `run --i-am-authorized` |
| `livenesslab/scripts/check_nuaa_session_ablation.py` | 53 offline tests on synthetic data and one on the real inputs |
| `livenesslab/scripts/check_nuaa_session_ablation_mutations.py` | 51 declared mutations plus one control |
| `livenesslab/scripts/nuaa_subject_folds.py` | approved fold schema code (manifest reading, `spread`) |
| `livenesslab/scripts/check_nuaa_subject_folds.py` | its test suite (synthetic fold schema fixture of the tests) |
| `livenesslab/scripts/m2_writers.py` | atomic writes and output-folder checks |
| `livenesslab/scripts/path_guard.py` | path checks (protected and allowed folders) |
| `livenesslab/scripts/package_rules.py` | forbidden-content rules of the public outputs |
| `livenesslab/delivery/sessionablation/nuaa_session_ablation.json` | configuration: inputs, rules, sizes, budget |
| `livenesslab/delivery/folds/nuaa_subject_folds.json` | approved fold schema configuration (input) |
| `tables/` | public outputs of the run: JSON, Markdown and their `SHA256SUMS` |
| `test_logs/` | macOS suite and real `check-inputs`, mutations, Linux container without data |

Inputs (read only, all pinned): the fold schema configuration (SHA-256 `d1d43c14...`), through it the two official
M2 manifests (SHA-256 and counts); the fold schema run: `status.json` PASS, `public/SHA256SUMS` and
`restricted/SHA256SUMS` (anchors of its `status.json`), `fold_schema_public.json` and the twelve manifests checked
against them and row by row against the official lists. Standard library only.

Commands, run from `livenesslab/` in this package or from the root of the development repository:

```
python scripts/nuaa_session_ablation.py check-inputs
python scripts/nuaa_session_ablation.py run --i-am-authorized
python scripts/check_nuaa_session_ablation.py
python scripts/check_nuaa_session_ablation_mutations.py
```

Expected test lines. `tests run: 54, failures: 0, errors: 0, skipped: 0` with the real inputs present; without the
data `skipped: 1` (declared: `real inputs`); `RESULT: PASS`. Mutation check: `mutations killed: 51/51; survived: 0;
invalid: 0; control survived: yes` (the one control mutation survives), `RESULT: PASS`. Linux container without data
(`test_logs/linux_package_no_data.txt`): the suite passes with that one skip, and `check-inputs` fails as expected
(`RESULT: FAIL (check-inputs)`: no manifest in the container).

## Delivery status

- Run `20261104-SESSABL-5d89888` (clean commit `5d89888` of the development repository): `public/` copied here as
  `tables/` (its `SHA256SUMS` is the anchor recorded in `status.json`). The restricted folder
  (subject lists, per-subject counts, the twenty proposed manifests with keys), with `status.json`
 , was uploaded on 2026-11-04 to the shared N1 folder as the run folder `20261104-SESSABL-5d89888/`
  (provisional destination, as for the folds run).
