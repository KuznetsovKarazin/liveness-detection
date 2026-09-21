# PAD research workflow — 2026

## Research question
How reliably does face presentation attack detection transfer across datasets, sensor modalities and iOS deployment after controlling splits, metrics and preprocessing?

## Milestones
1. **Traceable starting point:** map publications to code, data, checkpoints and tables; reproduce one existing experiment.
2. **Verified baseline:** audit metrics and splits, run a small RGB cross-dataset pilot, and check Python/CoreML numerical parity.
3. **Thesis results:** controlled VIS/NIR and RGB/measured-depth comparisons, plus an iOS field evaluation with error and latency analysis.

An extended leave-one-dataset-out study follows the verified pilot and an explicit compute budget.

## Starting a task
- Use branch `research/pad-2026` as the integration base for this work.
- Create a task branch, for example `task/M1-metrics`, from the latest integration branch.
- Keep one primary task in progress; link the task identifier in the pull request.
- Deliver code/configuration, a run report and aggregate results. Request review before merging into the integration branch.
- The historical main branch starting point is `4334db59d87855f73a060e2f6bc220e707003a72`. This workflow does not certify existing numerical claims.

## Artifact contract
Copy `templates/run-report.md` for each run. A run identifier is `YYYYMMDD-task-seed-shortcommit`.
Keep code, environment definitions, experiment configurations and approved aggregate tables here.
Keep licensed datasets, images, per-subject records, model weights and private links out of this public repository. Subject pseudonyms alone do not make data suitable for public release.
Store permitted experiment files in the project document workspace under `02_Experiments/<task-id>/<run-id>/`; keep access decisions in its dataset register.
Record SHA-256 hashes for data manifests, checkpoints and configuration. Use local environment variables for dataset roots, never personal paths or credentials.

## Minimum evaluation rules
- Explicitly record labels and the score direction. The shared score convention is probability of attack, label 0 bona fide and label 1 attack.
- Preserve official protocols. Split by real subject/session/video identifiers before extracting or augmenting frames; detect overlaps and duplicates.
- Select model, calibration and operating threshold using source development data only. Freeze them before testing an unseen target.
- APCER is attacks accepted divided by attacks; BPCER is bona fide samples rejected divided by bona fide samples. ACER is their mean when both are defined.
- Report missing-class metrics as unavailable, with counts; never replace them with zero.
- State the evaluation unit and sample counts. Adjacent frames are not independent participants.
- Distinguish measured depth from depth estimated from RGB.
- For CoreML, first compare identical tensors, then the full preprocessing pipelines on identical images.
- A diagnostic single-seed pilot is not a definitive estimate of generalization.

## Reviews
The scientific reviewer checks protocol, leakage, score semantics and conclusions. The independent reproduction task checks that another person can reproduce a central result from the documented procedure.
Corrections must identify the affected artifact and include a repeatable check. Published experiments are assessed from their original evidence, not inferred from a newer code snapshot.
