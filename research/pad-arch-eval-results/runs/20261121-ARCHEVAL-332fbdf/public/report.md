# NUAA test: LivenessNet and ResNet-18 (20261121-ARCHEVAL-332fbdf)

Status: **PASS**. Runs `20261120-ARCHRUN-p4-7faa49f` (patience 4) and `20261120-ARCHRUN-p12-ea7fdca` (patience 12); 48 models, 48 evaluated; every model enters, one-class models included; nothing excluded after the test is seen.

## Primary analysis (requested by the coordinator: patience 4, threshold 0.5)

ACER, APCER and BPCER are proportions between 0 and 1 (not percentages). Gap = ACER(ResNet-18) - ACER(LivenessNet), mean over the 12 fold-seed units, with the subject-bootstrap interval (percentiles 2.5-97.5 of 1,000 resamples, seed 0, of the 16 test subjects pooled over the 4 folds; in parentheses the valid resamples, those in which the metric is defined in every unit); contrast = gap(re-estimated BN) - gap(ordinary BN).

| | mean | subject interval (valid resamples) | units: ResNet-18 lower / LivenessNet lower / ties |
|---|---|---|---|
| ACER gap, ordinary BN | -0.1606 | [-0.2079, -0.0741] (926) | 11 / 1 / 0 |
| ACER gap, re-estimated BN | -0.0607 | [-0.1253, 0.0476] (926) | 9 / 3 / 0 |
| contrast | 0.0998 | [0.0256, 0.1632] (926) | units whose gap changes sign: 2 of 12 |

Reading of the contrast (a choice of the author, pad_arch_definitions.contrast_reading): **ACER difference (ResNet-18 - LivenessNet) higher with re-estimated BN** (interval entirely above zero).

Conclusion (written before the test; quoted from PROTOCOL.md §6, with the alternative resolved): "on these checkpoints and test frames the signed ACER difference (ResNet-18 − LivenessNet) is algebraically higher with this re-estimation; it does not identify a cause and says nothing on which network has the lower ACER in each BN state (that is read from the two gaps, reported with their intervals)"

### Per seed (reported apart; not part of the reading)

Each seed has 4 units (one per fold) and the bootstrap of each seed uses the same 16 test subjects.

| seed | ACER gap, ordinary [interval] | ACER gap, re-estimated [interval] | contrast [interval] | contrast reading |
|---|---|---|---|---|
| 42 | -0.1778 [-0.2821, -0.0227] | -0.0762 [-0.1557, 0.0479] | 0.1016 [-0.0226, 0.2263] | unresolved |
| 43 | -0.0970 [-0.1961, 0.0615] | -0.0905 [-0.1850, 0.0453] | 0.0065 [-0.0819, 0.0794] | unresolved |
| 44 | -0.2069 [-0.3472, -0.0259] | -0.0156 [-0.0796, 0.0896] | 0.1913 [-0.0108, 0.3771] | unresolved |

Checks requested by the coordinator, passed before the primary analysis: (a) on the evaluation plan, before any test list or frame was used: the 12 fold-seed combinations for each network in both BN states and the threshold to be applied 0.5 for both networks; (b) on the units passed to pad_arch_definitions.primary_analysis: [4, 'ordinary', 'fixed_0.5']: 12 units, effective threshold {'LivenessNet': [0.5], 'ResNet-18': [0.5]}; [4, 're-estimated', 'fixed_0.5']: 12 units, effective threshold {'LivenessNet': [0.5], 'ResNet-18': [0.5]}. The same completeness check passed on the other 6 conditions (choice of the author).

## Primary analysis: APCER and BPCER

| | APCER gap [interval] | BPCER gap [interval] |
|---|---|---|
| ordinary BN | 0.1801 [0.0871, 0.3372] | -0.5013 [-0.6144, -0.3330] |
| re-estimated BN | 0.1556 [0.0581, 0.3211] | -0.2771 [-0.4333, -0.0856] |
| contrast | -0.0245 [-0.0912, 0.0415] | 0.2242 [0.1202, 0.3088] |

## The eight conditions (secondary except the two primary ones; descriptive)

| patience | BN | threshold | mean ACER gap [interval] | ResNet-18 lower / LivenessNet lower / ties | sign | per-seed reading |
|---|---|---|---|---|---|---|
| 4 | ordinary | fixed_0.5 | -0.1606 [-0.2079, -0.0741] | 11 / 1 / 0 | ResNet-18 lower | inconclusive |
| 4 | ordinary | dev_eer | -0.1442 [-0.1865, -0.0560] | 11 / 1 / 0 | ResNet-18 lower | inconclusive |
| 4 | re-estimated | fixed_0.5 | -0.0607 [-0.1253, 0.0476] | 9 / 3 / 0 | unresolved | inconclusive |
| 4 | re-estimated | dev_eer | -0.0949 [-0.1596, 0.0111] | 9 / 3 / 0 | unresolved | inconclusive |
| 12 | ordinary | fixed_0.5 | -0.1031 [-0.1701, 0.0176] | 10 / 2 / 0 | unresolved | inconclusive |
| 12 | ordinary | dev_eer | -0.0973 [-0.1498, -0.0020] | 10 / 2 / 0 | ResNet-18 lower | inconclusive |
| 12 | re-estimated | fixed_0.5 | -0.0360 [-0.1016, 0.0613] | 8 / 4 / 0 | unresolved | inconclusive |
| 12 | re-estimated | dev_eer | -0.0915 [-0.1562, 0.0125] | 10 / 2 / 0 | unresolved | inconclusive |

## Between conditions (one factor changed; contrast = gap(first) - gap(second))

| first | second | signs | sign change | point-estimate inversion | ACER contrast [interval] | units changing sign |
|---|---|---|---|---|---|---|
| [4, 're-estimated', 'fixed_0.5'] | [4, 'ordinary', 'fixed_0.5'] | unresolved / ResNet-18 lower | no | no | 0.0998 [0.0256, 0.1632] | 2 of 12 |
| [4, 're-estimated', 'dev_eer'] | [4, 'ordinary', 'dev_eer'] | unresolved / ResNet-18 lower | no | no | 0.0492 [-0.0220, 0.1216] | 4 of 12 |
| [12, 're-estimated', 'fixed_0.5'] | [12, 'ordinary', 'fixed_0.5'] | unresolved / unresolved | no | no | 0.0671 [0.0127, 0.1015] | 2 of 12 |
| [12, 're-estimated', 'dev_eer'] | [12, 'ordinary', 'dev_eer'] | unresolved / ResNet-18 lower | no | no | 0.0058 [-0.0385, 0.0509] | 2 of 12 |
| [4, 'ordinary', 'dev_eer'] | [4, 'ordinary', 'fixed_0.5'] | ResNet-18 lower / ResNet-18 lower | no | no | 0.0164 [-0.0152, 0.0528] | 0 of 12 |
| [4, 're-estimated', 'dev_eer'] | [4, 're-estimated', 'fixed_0.5'] | unresolved / unresolved | no | no | -0.0342 [-0.0676, 0.0115] | 4 of 12 |
| [12, 'ordinary', 'dev_eer'] | [12, 'ordinary', 'fixed_0.5'] | ResNet-18 lower / unresolved | no | no | 0.0058 [-0.0246, 0.0248] | 2 of 12 |
| [12, 're-estimated', 'dev_eer'] | [12, 're-estimated', 'fixed_0.5'] | unresolved / unresolved | no | no | -0.0555 [-0.1016, -0.0018] | 2 of 12 |
| [12, 'ordinary', 'fixed_0.5'] | [4, 'ordinary', 'fixed_0.5'] | unresolved / ResNet-18 lower | no | no | 0.0575 [-0.0075, 0.1286] | 1 of 12 |
| [12, 'ordinary', 'dev_eer'] | [4, 'ordinary', 'dev_eer'] | ResNet-18 lower / ResNet-18 lower | no | no | 0.0469 [-0.0008, 0.0898] | 1 of 12 |
| [12, 're-estimated', 'fixed_0.5'] | [4, 're-estimated', 'fixed_0.5'] | unresolved / unresolved | no | no | 0.0247 [0.0046, 0.0385] | 1 of 12 |
| [12, 're-estimated', 'dev_eer'] | [4, 're-estimated', 'dev_eer'] | unresolved / unresolved | no | no | 0.0034 [-0.0149, 0.0187] | 1 of 12 |

## Subject sensitivity (mean ACER gap excluding one test subject at a time)

Aggregates only, in the public form of N1 (PR #10); the value of each exclusion is in the restricted outputs.

| condition | with all subjects | minimum | maximum | exclusions changing the sign | NA | n attack after exclusion | n bona fide after exclusion |
|---|---|---|---|---|---|---|---|
| [4, 'ordinary', 'fixed_0.5'] | -0.1606 | -0.1749 | -0.1334 | 0 of 16 | 0 | 20685-22527 | 13029-15315 |
| [4, 'ordinary', 'dev_eer'] | -0.1442 | -0.1600 | -0.1138 | 0 of 16 | 0 | 20685-22527 | 13029-15315 |
| [4, 're-estimated', 'fixed_0.5'] | -0.0607 | -0.0939 | -0.0312 | 0 of 16 | 0 | 20685-22527 | 13029-15315 |
| [4, 're-estimated', 'dev_eer'] | -0.0949 | -0.1230 | -0.0699 | 0 of 16 | 0 | 20685-22527 | 13029-15315 |
| [12, 'ordinary', 'fixed_0.5'] | -0.1031 | -0.1286 | -0.0671 | 0 of 16 | 0 | 20685-22527 | 13029-15315 |
| [12, 'ordinary', 'dev_eer'] | -0.0973 | -0.1227 | -0.0738 | 0 of 16 | 0 | 20685-22527 | 13029-15315 |
| [12, 're-estimated', 'fixed_0.5'] | -0.0360 | -0.0687 | -0.0057 | 0 of 16 | 0 | 20685-22527 | 13029-15315 |
| [12, 're-estimated', 'dev_eer'] | -0.0915 | -0.1193 | -0.0671 | 0 of 16 | 0 | 20685-22527 | 13029-15315 |

## One-class runs at threshold 0.5 on the dev (definition (a))

| network | patience | BN state | count |
|---|---|---|---|
| LivenessNet | 4 | ordinary | 2 of 12 evaluable (all bona fide 0, all attack 2, not evaluable 0) |
| LivenessNet | 4 | re-estimated | 0 of 12 evaluable (all bona fide 0, all attack 0, not evaluable 0) |
| LivenessNet | 12 | ordinary | 0 of 12 evaluable (all bona fide 0, all attack 0, not evaluable 0) |
| LivenessNet | 12 | re-estimated | 0 of 12 evaluable (all bona fide 0, all attack 0, not evaluable 0) |
| ResNet-18 | 4 | ordinary | 0 of 12 evaluable (all bona fide 0, all attack 0, not evaluable 0) |
| ResNet-18 | 4 | re-estimated | 0 of 12 evaluable (all bona fide 0, all attack 0, not evaluable 0) |
| ResNet-18 | 12 | ordinary | 0 of 12 evaluable (all bona fide 0, all attack 0, not evaluable 0) |
| ResNet-18 | 12 | re-estimated | 0 of 12 evaluable (all bona fide 0, all attack 0, not evaluable 0) |

## The earlier "degenerate" count, apart (PROTOCOL.md §6)

a choice of the author: PROTOCOL.md §6 asks for the earlier "degenerate" count apart from the one-class count: it is computed on the training cards of the 48 new models (fitting and dev information only) with the accepted rule of the group "ten" of PR #27, nuaa_bn_recalibration.group_of with the groups section of the pinned configuration of PR #27 (restored_epoch in {1, 2} and the dev accuracy at the checkpoint, val_acc_at_best of the card, equal to 0.5 within 1e-6, the bound included; source: the degenerate criterion of the degenerate study 20261107-DEGEN-a3b4dc0), per network x patience; applied to ResNet-18 as written, since its cards have the same fields; the rule was derived from LivenessNet histories; the study's own script, nuaa_degenerate_trainings.is_degenerate, uses a strict bound (< 1e-6) where group_of includes it (<= 1e-6): they differ only for a dev accuracy exactly 1e-6 from 0.5

| network | patience | degenerate |
|---|---|---|
| LivenessNet | 4 | 2 of 12 |
| LivenessNet | 12 | 0 of 12 |
| ResNet-18 | 4 | 0 of 12 |
| ResNet-18 | 12 | 0 of 12 |

Dev reproduced before the test: 96 of 96 model-state pairs within 1e-05 with no discordant decision (largest difference 0.0000000000); per model: reproduction.csv.

Test frames per fold: 3242, 3114, 3439, 2819; public outputs hold only counts and the SHA-256 of the sorted frame identifiers.

Started 2026-11-21T09:11:55+0100, finished 2026-11-21T09:42:42+0100; elapsed time 1848 s of a cap of 3.50 hours of wall time (elapsed real time, not CPU time); process CPU time (all threads) 15480 s.

TensorFlow threads set by this runner: requested intra-op 10 and inter-op 10, effective (read back from tf.config.threading) intra-op 10 and inter-op 10; environment variables: TF_NUM_INTRAOP_THREADS absent, TF_NUM_INTEROP_THREADS absent; logical CPUs 15.

What was read of the test: the NUAA test is read once, by evaluate, after every check that does not need it. What each command reads. check-inputs: this configuration; PROTOCOL.md and plan/run_plan.csv of the pre-registration (each equal to its line of the SHA256SUMS of the delivered package); the configuration of the two runs (validated by pad_arch_run.load_config) and what pad_arch_run.load_pins reads with it (the configuration of the pre-registration, the code, the status.json and trial.json of the official trial and the package SHA256SUMS, run_order.csv and run_caps.csv, the configurations of PRs #27 and #28); the code files pinned here and this file; what pad_arch_comparison.load_inputs reads (its TEST_READ: the five M2 manifests, read to index the cache, the cache and the extracted frames file, loaded whole, which also hold frames of the held-out subjects, the cells run, the folds run without its public files, the LivenessNet architecture file, the status.json and restricted SHA256SUMS of the two accepted runs and one training card); with --run-p4 and --run-p12, of each run folder: status.json, run.log (bytes, for its SHA-256), the public SHA256SUMS and every file it lists (summary.json and one_class.csv parsed, the others as bytes), the restricted SHA256SUMS and every file it lists (the checkpoints and the phase-1 dev scores as bytes, for the SHA-256; per model the card, the BN statistics and the two dev score files parsed); no fold schema, no test frame, TensorFlow not imported, nothing written. evaluate, in this order: the same files; the dev frames of each fold assembled (no other role) and every checkpoint loaded and scored on the dev in both BN states; only after the 48 dev reproductions have passed and after a re-verification (this configuration; the configuration of the runs and the files it pins, through pad_arch_run.final_checks; the code; PROTOCOL.md and run_plan.csv; of each run folder status.json, run.log, the two SHA256SUMS and every file they list; the inputs of pad_arch_comparison.load_inputs, already in memory, are not read again), the fold schema of the accepted folds run (fold<k>_fitting.csv, fold<k>_dev.csv and fold<k>_test.csv, loaded and checked as nuaa_bn_test.load_inputs does: every fold 1..k, the approved schema configuration, every row found once in the cache or in the extracted frames) and then, fold by fold, the test frames of the fold assembled once (every frame of its held-out subjects, from the cache and the extracted frames, in key order) and scored by the 12 models of the fold in the two BN states (attack probabilities and logits only: no test metric is computed while scoring); then the same re-verification is repeated, the checks requested by the coordinator are applied to the units, and only then the test metrics are computed, the primary analysis first. No aggregate.json, run_record.json or <model>_test_scores.csv of any earlier run is opened.

Command: `python -I -B scripts/pad_arch_eval.py evaluate --config delivery/archeval/pad_arch_eval.json --run-p4 results/archrun/20261120-ARCHRUN-p4-7faa49f --run-p12 results/archrun/20261120-ARCHRUN-p12-ea7fdca --out-root results/archeval --attempt 1 --i-am-authorized`
