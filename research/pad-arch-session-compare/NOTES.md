# Registration of the session comparison: what was read and what was run before the commit

Date of this note: 22–23 November 2026. Card C2. Nothing in this folder contains a score, a frame identifier or a
subject identifier of the study; `check_synthetic.py` uses invented four-digit identifiers only.

## What was not run

When this folder was committed, `session_compare.py` and `session_summary.py` had not been run on any score file of
the study, and no other program had computed an APCER or a BPCER by session group on the score files of the 48 checkpoints. Before the
commit, `check_synthetic.py` and `check_mutations.py` were run on invented data only (section "Checks of this
folder").

## What was read before the commit

Five passes, read only, on the restricted score files of `20261121-ARCHEVAL-332fbdf` (48 test files),
`20261120-ARCHRUN-p4-7faa49f` and `20261120-ARCHRUN-p12-ea7fdca` (96 development files) and on the public SM and PS
of pull request #36. No score column was accessed in any restricted file.

1. Author of the plan. Test files (all 48): `sample_id`, `subject_id`, `session_id`, `label`, `bn_state`.
   Development files (all 96): `session_id`. SM and PS.
2. Independent review of the plan. Test files (all 48): `sample_id`, `subject_id`, `session_id`, `label` and the
   metadata columns `network`, `fold`, `train_seed`, `patience`, `bn_state`, `role`. Development files (all 96):
   `session_id` and `label`. SM.
3. Confirmation pass on the revised plan. SM, PS and the four test files of patience 4, LivenessNet, seed 42:
   `subject_id`, `label`, `session_id`, `bn_state`.
4. Independent review of the code of this folder. The four test files of patience 4, LivenessNet, seed 42 and the
   two development files of LivenessNet, fold 1, seed 42: `network`, `fold`, `train_seed`, `patience`, `bn_state`,
   `role`, `sample_id`, `subject_id`, `session_id`, `label` (where present); counts only. SHA-256 of SM, PS and
   `expected_counts.csv`.
5. Confirmation of the code review. All 48 test files: `sample_id`, `subject_id`, `bn_state`, `patience`; all 96
   development files: `sample_id`, `subject_id`; counts only (no frame or identifier shared by development and test
   in any of the 96 checkpoint-states; 400 development frames in each file; the patience of every test row equal to
   the file name).

The author of the scripts read the header line (column names only) of one test file and one development file.

Obtained from the restricted files (counts only, no identifier printed):

- On all 302,736 rows of the 48 test files (both states): no `sample_id` outside the form
  `<class>/<ID>_<ID>_<glasses>_<pos>_<session>_<pic>.jpg`; `session_id` always equal to the session field of the file
  name, two characters, values `01`, `02` and `03` only; `label` consistent with the class prefix; `subject_id`
  equal to both identifier fields.
- The test denominators per fold, class and group of `PROTOCOL.md` §3 (`expected_counts.csv`), the same in all 24
  conditions of each fold.
- Manifest identifiers per fold and class: in sessions 1–2, in session 3, in both.

  | Fold | Photograph, sessions 1–2 | Photograph, session 3 | Photograph, both | Bona fide, sessions 1–2 | Bona fide, session 3 | Bona fide, both |
  |---|---|---|---|---|---|---|
  | 1 | 3 | 3 | 3 | 3 | 2 | 1 |
  | 2 | 2 | 4 | 2 | 2 | 3 | 1 |
  | 3 | 2 | 4 | 2 | 2 | 2 | 1 |
  | 4 | 2 | 4 | 2 | 2 | 2 | 0 |

- The rows of the two one-class identifiers have `session_id` `03` only.
- In each of the 96 development files: 100 frames per session (`01`, `02`) and class; no other session.

Checked on the public files of pull request #36 only:

- The 384 test rates of SM are exactly k/n, with k an integer and n the class denominator of the fold.
- The 192 thresholds of SM read back bit for bit (`repr(float(x)) == x`).
- The 192 development BPCER of SM all equal `1.0 − k/200`, as `profile` of SA computes them, with k the bona fide
  frames at or below the threshold; 9 of them differ from `(200 − k)/200`. Check C7 therefore uses `profile` copied
  unchanged.
- The 32 means of PS for the test rates at the development thresholds are reproduced exactly from SM with
  `np.array(...).mean()` over the twelve units in (fold, seed) order, as in SA.

## Implementation notes

Fixed before the run; none changes a quantity defined in `PROTOCOL.md`.

1. `session_compare.py` takes the path of PS as `--paired` (check C12 reads it) and an optional `--traceback`, the
   restricted file that receives the full traceback of a failure.
2. `session_summary.py` uses the check functions of `session_compare.py`, loaded from the file next to it, so that
   each check has one implementation.
3. Order of the checks in `session_compare.py`: before any restricted score file is opened, C11, C1, the PS part
   of C12 (SHA-256 and the 16 rows needed), the `expected_counts.csv` part of C6 (SHA-256 and the 16 cells) and the
   existence of the 144 input files (C10); then, for each test and development file, C2, C4, C5, C3, C7; then C10
   (rows), C6, C9, C8, C12 and C11 again before writing. In `session_summary.py`: the output files, C1, the PS part of
   C12, the `expected_counts.csv` part of C6, then C10, C6, C8, C9, C12.
4. Check C3 also compares the development partition of each fold across conditions and requires development and test
   to share no frame and no identifier, with 400 development frames, as SA does (lines 199–202 and 211); check C2
   also requires labels `0` or `1`, only the two BN states and the patience of the file name in a test file; check
   C10 also requires the 144 input files.
5. `session_summary.py` repeats C6 on the frame counts only: `session_counts.csv` has no identifier counts.
6. The rule for undefined quantities cannot be reached through the scripts on any input, because C6 stops on a
   denominator 0 in both scripts. `check_synthetic.py` exercises it by calling the functions of
   `session_summary.py` directly (summary, table, CSV and figure) with one unit whose session 3 photograph
   denominator is 0.
7. Every fold–seed unit of a fold has the same test frames, so the twelve units of a combination have four distinct
   sets of denominators, one per fold; the same holds in the invented data of `check_synthetic.py`.
8. Exit codes: 4 for a failed check, 5 for an unexpected error, 2 for an argument error. Library warnings are
   silenced in both scripts, because a warning prints the path of a library source file. The scripts set
   `sys.dont_write_bytecode`, so they leave no `__pycache__` in this folder. Outputs are built completely, written to
   a temporary folder inside the output folder and moved into place at the end.
9. The figure has its columns in the order of the rows of Table A3 (patience 4 LivenessNet, patience 4 ResNet-18,
   patience 12 LivenessNet, patience 12 ResNet-18), the same order as the table. `session_summary.py` separates the
   values of the figure (`figure_data`) from the drawing, and `check_synthetic.py` checks every panel, label, point and
   mean against an independent computation, and every cell of the table.

## Checks of this folder

Environment: Python 3.11.16, NumPy 1.26.4, Matplotlib 3.11.2.

- `check_synthetic.py`: passed, 192 operating points, 16 combinations, table and figure, 41 stop cases.
- `check_mutations.py`: mutations killed 56/56, removal mutations killed by their own stop case 12/12.

Both run on 23 November 2026 on an Apple M5 Pro; outputs, with their start times, in `test_logs/`.
