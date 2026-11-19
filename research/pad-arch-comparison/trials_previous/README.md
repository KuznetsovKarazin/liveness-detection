# Earlier technical trials

The official trial of the final pre-registration is in `../trial/` (`20261119-ARCHTRIAL-b8f7cca`). The five earlier
trials are kept here as they were written (the public files of each run: `status.json`, `trial.json`, `trial.md`,
`run.log.txt`, which is `run.log` renamed); none of their files is modified. All of them are the same unit: NUAA, arm
B, fold 1, seed 42, fitting and dev only, 3 epochs per network, CPU only, batch 16; no test frame assembled, no
checkpoint saved.

| run | date | machine | TensorFlow threads | code | LivenessNet history against the accepted card | role |
|---|---|---|---|---|---|---|
| `20261118-ARCHTRIAL-3f059b7` | 2026-11-18 | Apple M4, 10 logical CPUs, 16 GB RAM | not set by the runner; recorded 0 = not set through `tf.config.threading`; environment variables not recorded; according to the author, none was set (no record of the command line or of the environment) | code and configuration of PR #29 (commit `3f059b7`) | reproduced | the trial of PR #29 |
| `20261118-ARCHTRIAL-0858ae7` | 2026-11-18 | Apple M5 Pro, 15 logical CPUs, 24 GB RAM | not set by the runner; recorded 0 = not set through `tf.config.threading`; environment variables not recorded; according to the author, none was set (no record of the command line or of the environment) | code and configuration of PR #29 (commit `0858ae7`) | not reproduced | the new machine with the TensorFlow defaults: the reason for fixing the threads |
| `20261119-ARCHTRIAL-b8e78b2` | 2026-11-19 | Apple M5 Pro, 15 logical CPUs, 24 GB RAM | intra-op 10 and inter-op 10 through `TF_NUM_INTRAOP_THREADS=10 TF_NUM_INTEROP_THREADS=10` on the command line, declared by the conditions file written by the author (`../trial_comparison/reference_conditions.txt`), not by the record | code and configuration of PR #29 (commit `b8e78b2`) | reproduced | unofficial; the reference of the check requested by the coordinator (`../trial_comparison/`) |
| `20261119-ARCHTRIAL-1865f54` | 2026-11-19 | Apple M5 Pro, 15 logical CPUs, 24 GB RAM | intra-op 10 and inter-op 10, set and recorded by the runner | commit `1865f54`, configuration `2ade32c6d9bd` | reproduced | repeated because of erratum 2 (machine record) |
| `20261119-ARCHTRIAL-571b812` | 2026-11-19 | Apple M5 Pro, 15 logical CPUs, 24 GB RAM | intra-op 10 and inter-op 10, set and recorded by the runner | commit `571b812`, configuration `2ade32c6d9bd` | reproduced | repeated because the code and the configuration changed after it (corrections after our own second independent review; no change to the trainer or the recipe) |
| `20261119-ARCHTRIAL-b8f7cca` | 2026-11-19 | Apple M5 Pro, 15 logical CPUs, 24 GB RAM | intra-op 10 and inter-op 10, set and recorded by the runner | commit `b8f7cca`, configuration `5a83325e13d2` (this package) | reproduced | official |

The code and configuration of `3f059b7`, `0858ae7`, `b8e78b2` are the same (`code_sha256` and `config_sha256` of their
`trial.json`). The trials of format `pad-arch-trial/2` give their own commit and configuration; for `1865f54`,
`571b812` their code and configuration are not in this folder; `code_sha256` of their records names them.
`../trial_comparison/reference_conditions.txt`: besides the omitted lines declared in its heading, the trailing spaces
at the end of lines and the indentation before the count of `git status --short` were removed with respect to the
private log.

## Comparisons with the official trial

`compare-trials` (no tolerance), the official trial `20261119-ARCHTRIAL-b8f7cca` against each earlier trial, outputs
as written:

| earlier trial | outcome | folder |
|---|---|---|
| `20261118-ARCHTRIAL-3f059b7` | reproduced: difference 0 in the five fields over the 3 epochs of both networks | `20261118-ARCHTRIAL-3f059b7/comparison_with_official/` |
| `20261118-ARCHTRIAL-0858ae7` | not reproduced: LivenessNet differs from epoch 1 in `loss`, `accuracy`, `val_loss` (largest differences at epoch 2); ResNet-18 differs from epoch 1 in `loss`, `accuracy`, `val_loss`, `val_accuracy` (largest differences at epoch 1 for `loss` and `accuracy`, at epoch 3 for `val_loss` and `val_accuracy`) | `20261118-ARCHTRIAL-0858ae7/comparison_with_official/` |
| `20261119-ARCHTRIAL-b8e78b2` | reproduced: difference 0 in the five fields over the 3 epochs of both networks | `../trial_comparison/` |
| `20261119-ARCHTRIAL-1865f54` | reproduced: difference 0 in the five fields over the 3 epochs of both networks | `20261119-ARCHTRIAL-1865f54/comparison_with_official/` |
| `20261119-ARCHTRIAL-571b812` | reproduced: difference 0 in the five fields over the 3 epochs of both networks | `20261119-ARCHTRIAL-571b812/comparison_with_official/` |

The history of the official trial equals the accepted card (`../trial/trial.md`), so the differences of an earlier
trial from the official trial are also its differences from the card.

## Errata (our defects; the files are not modified)

1. **`first_differing_epoch` in the records of format `pad-arch-trial/1`** (`3f059b7`, `0858ae7`, `b8e78b2`): the
   field of `baseline_check` in `trial.json` holds the epoch at which the largest absolute difference is first
   reached, not the first epoch at which the two values differ (our defect in PR #29, corrected with a test in the
   final pre-registration; the trial reports of this format do not print the field). In `0858ae7` it reads 2 for
   `loss`, `accuracy`, `val_loss`; the first differing epoch is 1
   (`20261118-ARCHTRIAL-0858ae7/comparison_with_official/`). In `3f059b7`, `b8e78b2` there is no difference and the
   field is empty, as it should be.
2. **Names of the core groups in the records made on the Apple M5 Pro before the correction** (`0858ae7`, `b8e78b2`,
   `1865f54`): `performance_cores` 5 and `efficiency_cores` 10 in `trial.json`, and the line "5 performance and 10
   efficiency cores" in `trial.md`, are false: on the Apple M5 Pro the system names the two groups "Super" (5 cores)
   and "Performance" (10 cores) (our defect: the names were fixed in the code; corrected with a test in commit
   `4069c77` of our working repository: field `core_levels`, with the names given by the system).
3. **Threads in the records of format `pad-arch-trial/1`**: `intra_op_parallelism_threads` and
   `inter_op_parallelism_threads` 0 mean "not set through `tf.config.threading`"; the format does not record
   environment variables, so the record of `20261119-ARCHTRIAL-b8e78b2` does not show the two variables that the
   conditions file reports on its command line.