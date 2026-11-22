# Registration of the protocol and of the script, and comparison with the stored files

Date of this note: 22 November 2026. Card C2. Nothing in this folder contains a score, a frame identifier or a subject
identifier.

## What this folder fixes

The five files `PROTOCOL.md`, `README.md`, `score_analysis.py`, `check_synthetic.py` and `units.csv` are the content of
`PAD_Valerio_Followup_v0.1.zip`, attached by the coordinator to the C2 card on 22 November 2026 (17,379 bytes, SHA-256
`c4abe0cccb0c530660b4bef3405ca447f703a6c17dc15b037ece2a751bbeaa57`). They are here byte for byte, with no change
(`SHA256SUMS`). The commit of this folder is the dated version of the protocol and of the script asked for on the card
before any new result is computed. Version 0.1 is registered as it is: the remarks of this note do not modify it.

When this folder was committed, `score_analysis.py` had not been run on any score file of the study, in whole or in
part, and no other program had been run on these score files to compute the quantities it adds (AUROC, EER, DET grid,
thresholds at 5% and 10% development APCER, the 10,000 draws). What was done before the commit is listed below: it
reads names, columns, formats, pairings and counts, and computes no metric. Earlier deliveries on the NUAA test
already report AUROC values for LivenessNet models (public files of the fold runs and of the session-ablation runs);
`PROTOCOL.md` states that the earlier outcomes were known when the plan was written.

`check_synthetic.py`, which runs the whole of `score_analysis.py` on invented scores only and reads no file of the
study, was run before the commit on 22 November 2026, on the extracted copy of the archive (the same bytes as this folder),
in the environment recorded below and ended with "Synthetic
end-to-end check passed."

## The stored score files against the input the script expects

Folders: the evaluation `20261121-ARCHEVAL-332fbdf` and the two training runs `20261120-ARCHRUN-p4-7faa49f` and
`20261120-ARCHRUN-p12-ea7fdca`, each with its `restricted/` subdirectory in the authorised environment.

- The three `restricted/SHA256SUMS` manifests validate (49, 144 and 144 lines).
- Test scores: 48 files `p4|p12_<network>_fold<k>_seed<s>_test_scores.csv`, each with the two BN states in the column
  `bn_state`. Development scores: 96 files `<network>_fold<k>_seed<s>_dev_<state>.csv` for the states `ordinary` and
  `re-estimated`. The names are the ones the script builds.
- Every column the script reads is present (`network`, `fold`, `train_seed`, `bn_state`, `role`, `sample_id`,
  `subject_id`, `label`, `score_attack`, `prediction_05`, `checkpoint_sha256`), with `role` equal to `test` or `dev`,
  `label` and `prediction_05` equal to 0 or 1, label 1 for the attack and the stored decision rule "attack if
  score_attack > threshold".
- In all 144 files: the metadata of each row agree with the file name; no duplicated `sample_id` within a checkpoint
  and state; both classes present; every `score_attack` finite and in [0, 1]; 400 development rows per file; 3,242,
  3,114, 3,439 and 2,819 test frames in folds 1 to 4 (12,614 in all); within each fold the same frames, identifiers and
  labels for both networks, the three seeds, the two patience settings and the two states, on the test and on the
  development partition; no frame and no identifier shared by the development and the test partition of a fold; one
  checkpoint SHA-256 per model, the same in the two states and in the development and test files (48 distinct values);
  16 identifiers on the test, four per fold, none in two folds.
- The `checkpoint_sha256` of the score rows is the SHA-256 of the stored checkpoint file and the `weights_sha256` of
  its training card, for all 48 models (the files were hashed, no model was loaded).
- `units.csv` of this folder has the 96 rows of the published
  `research/pad-arch-eval-results/runs/20261121-ARCHEVAL-332fbdf/public/units.csv`, in the same order, and is equal to
  it cell by cell on its 17 columns. The published file has three more columns (`frame_ids_sha256`,
  `delta_acer_value`, `tie`) and a different column order, so the two files do not have the same SHA-256.
- The development threshold rule of `original_dev_threshold` is the rule of the runs: candidates are the distinct
  development scores plus the value just below the minimum when it is not negative; the minimum of
  |fp * n_attack - fn * n_bona_fide| on integer counts; ties to the highest value.
- Facts quoted by `PROTOCOL.md` that agree with the stored files: the snapshot
  `db90203ba56607e2443e2a402e743d690313e8eb`; the original interval from 1,000 draws of which 926 valid; the primary
  change of +9.983325 percentage points (0.0998332546851536 from the published `units.csv`); 88 updates per epoch
  (1,400 fitting frames and batch 16 in all 48 training cards); BatchNormalization momentum 0.99; three dropout layers
  in LivenessNet (0.25, 0.25, 0.5) and none in ResNet-18; `GaussianNoise(0.02)` as the last operation of the
  augmentation.

## The four original NUAA lists

Counted from `client_train_raw.txt`, `client_test_raw.txt`, `imposter_train_raw.txt` and `imposter_test_raw.txt` with a
script written for this check, independent of the parser of the manifests.

- 1,743, 3,362, 1,748 and 5,761 lines, 12,614 in all: 5,105 client and 7,509 imposter.
- Every line is a path ending in `<ClientRaw|ImposterRaw>/<ID>/<ID>_<glasses>_<pos>_<session>_<picNo>.jpg`; the folder
  is always equal to the ID field; the class of the list always agrees with the folder; no line is repeated.
- The two train lists contain sessions 01 and 02 only; the two test lists contain session 03 only.
- 16 distinct identifiers. 14 are in both classes. Two are in one class only: one with 472 client frames, all in
  session 03, in the test of fold 1; one with 468 imposter frames, all in session 03, in the test of fold 3. This is
  what `PROTOCOL.md` §1 states.
- In addition: 15 identifiers appear among the client lists and 15 among the imposter lists. Identifiers per class and
  session: client 8, 7 and 9 in sessions 01, 02 and 03; imposter 9, 9 and 15.
- Association of the identifiers between client and imposter: the subject of our manifests is the folder of the list
  line, which equals the ID field of the file name, in both classes. A client frame and an imposter frame have the
  same identifier when their lines carry the same ID, and there is no other correspondence between the two classes.
  No identity was matched from the images.
- Parser: the 12,614 `sample_id` of the test score files of the four folds (class, ID and original file name)
  correspond one to one to the 12,614 lines of the four lists, with `subject_id` equal to the folder and `label`
  equal to the class of the list for every frame.
- Fitting and development: for all 48 models the hashes of the fitting keys and of the development keys in the
  training cards are the hashes of the frames of sessions 01 and 02 of the fold manifests (1,400 and 400 frames). The
  two identifiers that exist in session 03 only are therefore in the fitting and development frames of none of the 48
  models, as `PROTOCOL.md` §1 states (the fold manifests also list session-03 frames in those roles, which these runs
  do not use).

## Differences and remarks

None changes a computation of the script.

1. `units.csv` is a subset and reordering of the columns of the published file, not the same bytes (see above).
2. The stored `restricted/` folders also hold files the script does not read: a third development file per model
   (`*_dev_scorer.csv`), `loso_values.csv`, the checkpoints, their cards and BN statistics.
3. `PROTOCOL.md` §2, point 3 asks for the grid and the unweighted mean of the twelve curves: the script writes the grid
   per checkpoint and state (`det_grid.csv`) and draws the mean in `det_curves.pdf`, but does not write the mean as a
   table. Fixed here, before the run: the mean is the arithmetic mean of `minimum_attainable_test_bpcer` over the
   twelve fold-seed units, for each network, patience, state and APCER ceiling of `det_grid.csv`.
4. `PROTOCOL.md` §2, point 2 asks to report any all-attack operating point: the outputs have no column for it. Fixed
   here, before the run: it is a row of `score_metrics.csv` with `5pct_dev_apcer` 0 and `5pct_dev_bpcer` 1 (or the same
   two columns at 10%), which on the development scores happens only at the threshold just below the minimum.
5. `PROTOCOL.md` §3 asks for the identifier-deletion ranges; for the two within-model changes the script keeps the
   pooled summary only (percentile and BCa intervals): the deletion range, the within-fold variant and the variant
   without fold 3, seed 44 are written for the primary contrast only.
6. The draws depend on the NumPy generator: the versions of the environment in which the script will be run are
   Python 3.11.16, NumPy 1.26.4, SciPy 1.16.1, Matplotlib 3.11.2.

## What follows

From this folder as committed: `check_synthetic.py` is run again first, as in `README.md`, and its outcome is reported; then
`score_analysis.py` is run once on the stored scores, with the command of `README.md`, `units.csv` of this folder and a
new output folder. The five outputs are aggregate only and are delivered as they are written, with any failed
reproduction reported before any estimate is read. The re-estimation with augmentation and dropout (`PROTOCOL.md` §4)
is a later step and is not part of this registration.
