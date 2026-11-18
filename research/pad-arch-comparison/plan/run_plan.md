# Planned runs: LivenessNet and ResNet-18

Generated from the configuration 9264ec0201c1 and the trial 20261118-ARCHTRIAL-3f059b7 (trial.json 56d20023ddb6, status PASS).

Both networks trained from scratch with the same recipe, subject splits, seeds and preprocessing; patience 4 and 12, at most 12 epochs; evaluation cells per training: BN ordinary / re-estimated x threshold fixed_0.5 / dev_eer. Every training of the plan is new (existing = 0, a choice of the author); for LivenessNet the accepted runs are the reference runs of the history reproduction check, not checkpoints to reuse.

Hours are hours of wall time on the machine of the trial; CPU hours use the CPU / wall ratio measured in the trial. The rates are means over the measured epochs, the first included (graph tracing: a cautious estimate); each training adds its measured cost outside the epochs; the BN re-estimation is counted as (BN layers + 1) full passes over the fitting set, an upper bound (at most: its passes stop at the input of each layer); outside the estimate: saving the checkpoints, assembling the test, loading the inputs (once per process). «to be fixed in the pre-registration»: a value of the configuration not fixed yet; an empty cell: no trial measurement.

| dataset | access | network | patience | units | existing_trainings | new_trainings | reference_run | epochs_min | epochs_max | fitting_frames_per_unit | dev_frames_per_unit | test_frames_per_split | evaluation_cells | training_wall_hours_min | training_wall_hours_max | evaluation_wall_hours | wall_hours_min | wall_hours_max | cpu_hours_min | cpu_hours_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NUAA | available | LivenessNet | 4 | 12 | 0 | 12 | 20261106-SESSABLRUN-d999458 | 5 | 12 | 1400 | 400 | 3242 3114 3439 2819 | 4 | 0.46 | 1.10 | 0.27 | 0.73 | 1.37 | 5.06 | 9.44 |
| NUAA | available | LivenessNet | 12 | 12 | 0 | 12 | 20261109-SESSABLRUN-6a50265 | 12 | 12 | 1400 | 400 | 3242 3114 3439 2819 | 4 | 1.10 | 1.10 | 0.27 | 1.37 | 1.37 | 9.44 | 9.44 |
| NUAA | available | ResNet-18 | 4 | 12 | 0 | 12 | none | 5 | 12 | 1400 | 400 | 3242 3114 3439 2819 | 4 | 1.16 | 2.78 | 1.66 | 2.82 | 4.44 | 22.86 | 35.98 |
| NUAA | available | ResNet-18 | 12 | 12 | 0 | 12 | none | 12 | 12 | 1400 | 400 | 3242 3114 3439 2819 | 4 | 2.78 | 2.78 | 1.66 | 4.44 | 4.44 | 35.98 | 35.98 |
| CelebA-Spoof | access missing | LivenessNet | 4 | 3 | 0 | 3 | none | 5 | 12 | 1400 | 400 | to be fixed in the pre-registration | 4 | 0.12 | 0.27 |  |  |  |  |  |
| CelebA-Spoof | access missing | LivenessNet | 12 | 3 | 0 | 3 | none | 12 | 12 | 1400 | 400 | to be fixed in the pre-registration | 4 | 0.27 | 0.27 |  |  |  |  |  |
| CelebA-Spoof | access missing | ResNet-18 | 4 | 3 | 0 | 3 | none | 5 | 12 | 1400 | 400 | to be fixed in the pre-registration | 4 | 0.29 | 0.70 |  |  |  |  |  |
| CelebA-Spoof | access missing | ResNet-18 | 12 | 3 | 0 | 3 | none | 12 | 12 | 1400 | 400 | to be fixed in the pre-registration | 4 | 0.70 | 0.70 |  |  |  |  |  |
| Replay-Attack | access missing | LivenessNet | 4 | 3 | 0 | 3 | none | 5 | 12 | 1400 | 400 | to be fixed in the pre-registration | 4 | 0.12 | 0.27 |  |  |  |  |  |
| Replay-Attack | access missing | LivenessNet | 12 | 3 | 0 | 3 | none | 12 | 12 | 1400 | 400 | to be fixed in the pre-registration | 4 | 0.27 | 0.27 |  |  |  |  |  |
| Replay-Attack | access missing | ResNet-18 | 4 | 3 | 0 | 3 | none | 5 | 12 | 1400 | 400 | to be fixed in the pre-registration | 4 | 0.29 | 0.70 |  |  |  |  |  |
| Replay-Attack | access missing | ResNet-18 | 12 | 3 | 0 | 3 | none | 12 | 12 | 1400 | 400 | to be fixed in the pre-registration | 4 | 0.70 | 0.70 |  |  |  |  |  |

## Totals (hours of wall time on this machine; CPU hours apart)

| scope | rows with every number | training_wall_hours_min | training_wall_hours_max | evaluation_wall_hours | wall_hours_min | wall_hours_max | cpu_hours_min | cpu_hours_max |
|---|---|---|---|---|---|---|---|---|
| NUAA | 4 of 4 | 5.51 | 7.76 | 3.86 | 9.36 | 11.62 | 73.34 | 90.84 |
| CelebA-Spoof | 0 of 4 |  |  |  |  |  |  |  |
| Replay-Attack | 0 of 4 |  |  |  |  |  |  |  |
| all | 4 of 12 | 5.51 | 7.76 | 3.86 | 9.36 | 11.62 | 73.34 | 90.84 |

## Datasets

- NUAA (available): official source: NUAA Photograph Imposter Database (Tan, Li, Liu, Jiang, ECCV 2010), the copy and official lists used since M2 (manifests official_train and official_test); identity split: the four subject folds of the accepted fold schema (run 20261103-FOLDS-67b7f3d): four held-out test subjects per fold; arm B fitting and common dev of the fold from the cells run 20261104-SESSABL-5d89888; missing access: none.
- CelebA-Spoof (access missing): official source: Zhang et al., CelebA-Spoof (ECCV 2020, DOI 10.1007/978-3-030-58610-2_5); repository https://github.com/ZhangYuanhan-AI/CelebA-Spoof, download folder linked from its README; identity split: the official training / validation / test split, ratio 8 : 1 : 1, with no overlap on subjects (paper, section 3.3), taken as one split; how the identity is encoded in the archive is not verified; fitting and dev subsampled to the NUAA arm B budget (proposal, to be fixed in the pre-registration with the frame sampling and the test size); missing access: the official archive is not held (the local copy is one shard of an unofficial mirror with no identity column); according to the coordinator's D1 card as of 17 November 2026 the conditions of use were confirmed by the first author of the dataset on 8 October 2026 and the university channel is still to be agreed.
- Replay-Attack (access missing): official source: Chingovska, Anjos, Marcel, Replay-Attack (BIOSIG 2012); Idiap, https://www.idiap.ch/en/scientific-research/data/replayattack, distributed through Zenodo, DOI 10.34777/cwcg-7r82; identity split: the official train / devel / test sets, whose identities do not overlap (paper, section III; 15 devel and 20 test clients, 15 train clients by difference), taken as one split; fitting and dev frames sampled from the train and devel videos to the NUAA arm B budget (proposal, to be fixed in the pre-registration with the frame sampling, the per-video aggregation and the test size); missing access: no copy is held; the Zenodo record is restricted and access depends on an End-User License Agreement signed for the university by a signatory in a permanent position; according to the coordinator's D1 card as of 17 November 2026 no new agreement or access is confirmed.

## Costs

- rates: from the trial.json of a trial produced with this configuration and this code, per network: training seconds per 1,000 fitting frames per epoch = mean over the measured epochs, the first included (it includes the graph tracing: a cautious estimate), of (epoch seconds - validation seconds) / fitting frames x 1,000; validation seconds per 1,000 dev frames (same mean); inference seconds per 1,000 frames from the two inference passes; seconds per training outside the epochs (the whole trainer minus the sum of its epochs: model building, data preparation before the fit, time between and after the epochs); CPU / wall ratio of all of these
- training: per row: new trainings x (epochs x (training rate x fitting frames + validation rate x dev frames) / 1,000 + seconds per training outside the epochs); minimum epochs min(patience + 1, 12) (best epoch 1, then patience epochs without improvement), maximum 12
- evaluation: per training (all trainings of a row are evaluated): inference rate x ((BN layers + 1) x fitting frames + 2 x dev frames + 2 x test frames of its split) / 1,000; (BN layers + 1) full passes is an upper bound (at most): the PR #27 re-estimation makes one pass per BN layer truncated at the input of that layer, plus one consistency pass; the test and the dev are scored in the two BN states; the two thresholds add no pass
- totals: hours of wall time on the machine of the trial, interval from the minimum to the maximum epochs; CPU hours apart, wall hours x the measured CPU / wall ratio; a row with a value not fixed in the configuration, or without a trial measurement, has no number and is left out of the totals (the totals state how many rows they include); outside the estimate: saving the checkpoints, assembling the test, loading the inputs (once per process)
