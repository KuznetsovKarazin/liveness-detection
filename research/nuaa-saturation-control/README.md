# NUAA: one control on the early dev saturation (proposal, no training, no inference)

Written on the coordinator's request of 12 November 2026 (C2 card): a brief note, at most one page, choosing ONE control
on the early dev saturation of the LivenessNet trainings (dev accuracy 0.5 in the first epochs with confident logits of
one class, while the train accuracy is above 0.9 from epoch 1), present in the three accepted runs on the same 36
models (patience 4 `20261106-SESSABLRUN-d999458`, patience 12 with ReduceLROnPlateau `20261109-SESSABLRUN-6a50265`,
patience 12 at a constant learning rate `20261111-SESSABLRUN-8568ffa`). Starting from the code and the recorded curves,
`NOTE.md` states the hypothesis, the one modification, what stays fixed, the reading rule and the expected cost. Only
the proposal: the control is to be agreed before any inference, without using the test.

The proposed control re-estimates the statistics of the five BatchNormalization layers of each patience-4 checkpoint on
the fitting set of its arm and fold (inference mode, layer by layer) and evaluates the dev before and after, with the
recovery criterion already in use as reading rule (hypotheses 1 and 2 of `research/nuaa-degenerate-trainings/NOTE.md`
against 4 and 7), after a validity condition (checkpoint SHA-256 and the recorded dev values reproduced). As a secondary
reading that does not decide, it also measures accuracy and loss on the fitting set in inference mode before and after
the recalibration (one more pass each), read with the same criterion in a 2×2 table (fitting before × dev after). Per
model: five recalibration passes and two evaluation passes over the fitting set, two passes over the dev (10,600 frame
passes for B and A*, 7,338 for B'); derived cost about 32 minutes for the 36 models, upper estimate about 140 minutes,
the reading of the fitting frames not priced.

## What it does not do

No training, no inference, no model loaded, no image read, no test score or test metric read. The script reads only
public tables already merged in `research/pad-2026` and the public `src/architectures.py`; of the public aggregate of the
patience-4 run it uses only the `training` block (restored epoch, sizes, `weights_sha256`; the file is read and hashed
whole), and of the public `comparison.csv` of the patience verification only the `parent_*` columns (dev AUROC at the
patience-4 checkpoint). The
cost of the control is derived from recorded timings, not measured. The checkpoint files (`.keras`) are private and were
not read: their SHA-256 are copied from the public aggregate.

## Contents

| path | what it is |
|---|---|
| `NOTE.md` | the one-page note: hypothesis (facts for and against), the one modification, what stays fixed, reading rule with each outcome's conclusion written in advance (validity first; primary on the dev; secondary, not deciding: the 25 control models, the non-degenerate model outside the criterion, dev AUROC, fitting accuracy and loss in inference mode with the reading of the 2×2 table) and the expected outcome (not observed), expected cost, limits, what is not proposed, pre-commitment |
| `tables/checkpoints.csv` | the 36 checkpoints of the patience-4 run, one row per model: arm, fold, seed, fitting frames, steps per epoch, epochs run, restored epoch, dev accuracy and val_loss at the checkpoint, degenerate, train accuracy at epoch 1 and at the restored epoch, dev accuracy at the restored epoch, epochs at 0.5, BatchNormalization residual 0.99^(steps) after epoch 1 and 0.99^(steps × restored epoch), first recovery epoch (`NA` if none), at the recovery criterion at the checkpoint, dev AUROC at the checkpoint, `weights_sha256` |
| `tables/facts.json`, `tables/facts.md` | the facts cited in the note: the ten degenerate models and their restored epochs, train accuracy at epoch 1 (median and range, the ten and the 26), dev saturation counts, dev AUROC at the checkpoint, BatchNormalization residuals per arm, the five BatchNormalization and three Dropout layers with positions, fitting and dev sizes, epochs at 0.5 of the ten in the two 12-epoch runs, the dev subjects per fold, the non-degenerate model outside the recovery criterion (dev AUROC included), the cost estimate with its inputs (per model five recalibration passes and two evaluation passes over the fitting set, two dev passes; the slowest per-epoch mean of each arm = maximum over its models of `train_seconds / epochs_run`) |
| `tables/layers.md` | the layer order of LivenessNet parsed from `src/architectures.py`, BatchNormalization and Dropout marked |
| `tables/sources.json` | SHA-256 of every source read at build time |
| `livenesslab/scripts/nuaa_saturation_note.py` | `build` and `check` (with `--note`, `check` also requires a fixed list of numbers, formatted from the rebuilt facts, to appear in `NOTE.md`, line breaks ignored); standard library only |
| `livenesslab/scripts/check_nuaa_saturation_note.py` | 83 tests: 80 on a synthetic tree written in a temporary folder (parsers, short and long CSV rows, the residual formula, the layer parsing with the total of 22 layers and the `name=` check, every count against hand-computed values, the val_loss branch of the recovery criterion, a restored epoch 3 at dev accuracy 0.5 (not degenerate), 22 error paths, outputs, `build`/`check`/`check --note` with every fragment of the note list required, `--help`) and 3 end-to-end tests on the real public files when present |
| `test_logs/check_nuaa_saturation_note.txt` | output of the tests and of `check --note` on macOS |
| `test_logs/linux.txt` | the same tests, `check --note` and the checksum verification in a Linux container (python:3.12) |
| `SHA256SUMS` | SHA-256 of every other file of this folder |

## Sources (all in this branch)

| source | used for | SHA-256 |
|---|---|---|
| `research/nuaa-degenerate-trainings/tables/models.csv` | the 36 models of the patience-4 run (rows `sessablrun`) | `50f19b53d1e955313998a8881fadbd328298e065cc6eb5feec9a88a60948f5cf` |
| `research/nuaa-degenerate-trainings/tables/curves.csv` | their per-epoch train/dev histories | `234d8b92f6de0bada7d5d8efff917916c776e10794fbbba7a4d7bbef7cbf7f60` |
| `research/nuaa-degenerate-trainings/tables/aggregate.json` | criteria (degenerate, saturation, recovery), BatchNormalization momentum | `7902a523f9677f4eea14b206b51960e49edd1535ab318d67ed16e2092dc0698b` |
| `research/nuaa-session-ablation-run/tables/run_record.json` | run id, architecture SHA-256, timings of the threshold step | `76cd5b5a195247a9c526f32a2c6e63b046a81379244b922233041632cf4b789a` |
| `research/nuaa-session-ablation-run/tables/aggregate.json` | `training` block only (`weights_sha256`, cross-check of restored epoch and sizes) | `ccfffc7dea951cdc228d59fc727b3807f57548863de3bcbdd450e545bd5f6418` |
| `research/nuaa-patience-verification/tables/run_record.json` | run id and architecture SHA-256 of the patience-12 run | `3435441fe5430eccfd11d961f0a4ed9f98d780497869cf291733a9771e475553` |
| `research/nuaa-patience-verification/comparison/curves.csv` | per-epoch curves of the patience-12 run | `a97c3af06ed016e42a4939d0e94a1259535153e53127ab315f92ac5e1cb951ec` |
| `research/nuaa-patience-verification/comparison/comparison.csv` | `parent_restored_epoch`, `parent_dev_auroc_at_checkpoint` only | `b300a440665cd336b9c6be7c7c197472111c7ccd6f728206e8d9b4cec93e4dd2` |
| `research/nuaa-constant-lr-control/tables/run_record.json` | run id and architecture SHA-256 of the constant-lr run | `dd1628171328972ec5ef95184146dccab84fa55f99e0f801296ea38b4eef6c68` |
| `research/nuaa-constant-lr-control/comparison/curves.csv` | per-epoch curves of the constant-lr run | `e188b3d09d49bafcaf53dbf761aa9f17900f41d1664e702f172d9ab64533815c` |
| `research/nuaa-constant-lr-control/livenesslab/delivery/constlr/nuaa_session_ablation_run_constant_lr.json` | fitting and dev sizes, batch, scoring batch and device, architecture SHA-256 | `60b38e094da8c2243f3f9386f69d216bf8b853a3511b73538feec39e09b0d58f` |
| `src/architectures.py` | LivenessNet layer order (parsed as text) | `a83c02b0788bd6f98262abf00f5fe970abed501f231edba348820482aea64d30` |

The per-epoch curves of the two 12-epoch runs are in two files: `comparison/curves.csv` of the constant-lr control holds
only the constant-lr run, `comparison/curves.csv` of the patience verification the patience-12 run.

Checks made by `build` (any failure stops it with an explicit message and nothing is written): every source present; the
three run records `PASS` from a clean tree, with the architecture SHA-256 equal to that of `src/architectures.py` (also
equal to the one in the configuration); the degenerate study reads the patience-4 run of the record; the degenerate
criterion is restored epoch 1-2 with dev accuracy 0.5; exactly the 36 models (3 arms × 4 folds × 3 seeds) in
`models.csv` and `curves.csv`, epochs 1..n without gaps; for every model the fitting size equals the configuration, the
steps per epoch equal ceil(fitting / batch), the restored epoch is the first epoch of minimum val_loss, `models.csv` and
`curves.csv` agree at that epoch, the degenerate flag equals the criterion applied to the curves, the epochs at 0.5 and
the first recovery epoch equal those recomputed from the curves, the `training` block of the patience-4 aggregate agrees
(restored epoch, sizes, best val_loss, 400 dev frames) and gives a SHA-256, `comparison.csv` has the same restored epoch,
the run record has the threshold timing; in the two 12-epoch curves, `in_ten` equals the degenerate models of the
patience-4 run; `seconds_per_epoch` equals `train_seconds / epochs_run`; every CSV row has as many fields as its header;
LivenessNet has 22 layers, each with `name=`, of which 5 BatchNormalization and 3 Dropout, and no `momentum` argument;
training and scoring device CPU.

## Commands (from the repository root)

Rebuild the tables:

```bash
python3 -I -B research/nuaa-saturation-control/livenesslab/scripts/nuaa_saturation_note.py build --research research --out research/nuaa-saturation-control/tables
```

Check that every source still has the SHA-256 recorded in `tables/sources.json`, that the committed tables are byte for
byte what the sources give (rebuilt in a temporary folder) and that the numbers of the note are in `NOTE.md` (exit code
1 on any difference):

```bash
python3 -I -B research/nuaa-saturation-control/livenesslab/scripts/nuaa_saturation_note.py check --research research --out research/nuaa-saturation-control/tables --note research/nuaa-saturation-control/NOTE.md
```

Tests (synthetic sources written in a temporary folder; the end-to-end test runs on the public files when present):

```bash
python3 -I -B research/nuaa-saturation-control/livenesslab/scripts/check_nuaa_saturation_note.py -v
```

Checksums of this folder:

```bash
cd research/nuaa-saturation-control && shasum -a 256 -c SHA256SUMS
```

Dependencies: Python 3.11 or later, standard library only (tested on macOS, `test_logs/check_nuaa_saturation_note.txt`, and in a Linux container, `test_logs/linux.txt`).
`build` writes UTF-8 with LF line endings and JSON with sorted keys; `check` compares bytes.
