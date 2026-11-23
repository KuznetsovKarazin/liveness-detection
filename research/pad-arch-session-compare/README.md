# Session comparison at the registered development thresholds

Protocol: `PROTOCOL.md`. Notes on what was read before this registration: `NOTES.md`. Card C2.

## Files

| File | Role |
|---|---|
| `session_compare.py` | Reads the restricted score files, SM, PS and the configuration; runs checks C1–C12; writes `session_counts.csv` only. |
| `session_summary.py` | Reads public files only; repeats C1, C6, C8, C9, C10, C12; writes `session_summary.csv`, `session_table.md`, `session_differences.pdf`. |
| `session_config.json` | Registered configuration: SHA-256 of SM, PS and `expected_counts.csv`, groups, targets, highlighted outcome. The keys `metrics_file` and `ps_file` are informational; the scripts take the paths as arguments. |
| `expected_counts.csv` | The 16 test denominators (fold × class × group) and the number of manifest identifiers per cell. |
| `check_synthetic.py` | End-to-end run on invented scores, with an independent check of every count, table cell and figure value, and 41 stop cases (every check and sub-check). Reads no file of the study. |
| `check_mutations.py` | The 56 mutations of `PROTOCOL.md` §7, each on a temporary copy of the scripts; invented data only. |
| `test_logs/` | Outputs of the two checks on the files of this folder. |
| `NOTES.md`, `SHA256SUMS` | What was read before this registration and the author's choices; SHA-256 of every file of this folder. |

## Environment

Python 3.11.16, NumPy 1.26.4, Matplotlib 3.11.2 (figure only). No SciPy, no TensorFlow, no network access.
`check_synthetic.py` and `check_mutations.py` need no data and use no platform-specific call; they have been run on
macOS only.

## Commands

From this folder as committed (`SM` and `PS` below are the two public files of
`research/pad-arch-score-results/runs/20261122-ARCHSCORE-fe87b3d/public/`):

~~~sh
python -I -B check_synthetic.py
python -I -B check_mutations.py

python -I -B session_compare.py \
  --test /path/to/20261121-ARCHEVAL-332fbdf/restricted \
  --dev4 /path/to/20261120-ARCHRUN-p4-7faa49f/restricted \
  --dev12 /path/to/20261120-ARCHRUN-p12-ea7fdca/restricted \
  --metrics /path/to/score_metrics.csv \
  --paired /path/to/paired_summary.csv \
  --traceback /path/to/<run>/restricted/traceback.txt \
  --out /path/to/<run>/public

python -I -B session_summary.py \
  --counts /path/to/<run>/public/session_counts.csv \
  --metrics /path/to/score_metrics.csv \
  --paired /path/to/paired_summary.csv \
  --traceback /path/to/<run>/restricted/traceback.txt \
  --out /path/to/<run>/public
~~~

The registered run does not pass `--config`: each script reads `session_config.json` next to itself and prints its
SHA-256 as the first line of its output. `--out` of `session_compare.py` must be new or empty; `session_summary.py`
refuses to overwrite its three files. Exit code 0 on success, 4 when a check fails (one line
`CHECK FAILED <check>: <reason> [<key>]`), 5 on an unexpected error (its type only), 2 on an argument error
(argparse); in the failure cases nothing is written and the full traceback goes only to the `--traceback` file. The
public inputs (configuration, SM, PS, `expected_counts.csv`, output folder, existence of the input files) are checked
before any restricted score file is opened. The scripts write no bytecode cache, also without `-B`.

Expected outputs: `session_counts.csv` (192 rows), `session_summary.csv` (16 rows), `session_table.md` (16 rows and
header), `session_differences.pdf`.

## Checks of this folder

- `check_synthetic.py`: `Synthetic end-to-end check passed: 192 operating points, 16 combinations, table and figure, 41 stop cases.`
- `check_mutations.py`: mutations killed: `56/56`; removal mutations killed by their own stop case: `12/12`.

The outputs of the two checks, run on 23 November 2026 on the files of this folder, are in `test_logs/`
(`check_synthetic.log.txt`, `check_mutations.log.txt`). Each log starts with the date, the versions and the SHA-256 of
the scripts, the configuration and `expected_counts.csv` it ran on.
