# Annex: the other analyzers of the C1 table on the same 300 NUAA images (threshold 0.5, run `20260928-C1-seed42-a80f6f9`)

Context only: third-party pre-trained models with no NUAA training in our experiment (whether NUAA appears in their pretraining data is not verified here: it would need a provenance check; "zero-shot" names the C1 analyzer, classification by text prompts with no classifier fitted on NUAA, and says nothing about the pretraining data), hand-crafted rules, linear probes on third-party encoders (same caveat for the encoders) and SVMs on hand-crafted features, the probes and the SVMs fitted on all 3,000 sampled NUAA training images (the dev images included; no early stopping involved). Source: `c1-livenesslab/tables/nuaa_table.md`.

| analyzer | model | trained on | n bona fide | n attack | APCER | BPCER | ACER | EER | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| minifasnet | MiniFASNet (Silent-Face, pre-trained) | no NUAA training in our experiment; third-party pretraining, NUAA in its data not verified | 150 | 150 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| clip_zeroshot | CLIP ViT-B/32 zero-shot | no NUAA training in our experiment; third-party pretraining, NUAA in its data not verified | 150 | 150 | 0.0133 | 0.0067 | 0.0100 | 0.0133 | 0.9982 |
| clip_probe | linear probe on CLIP ViT-B/32 | probe: all 3,000 sampled NUAA training images (dev included); encoder: third-party pretraining, NUAA in its data not verified | 150 | 150 | 0.0267 | 0.0000 | 0.0133 | 0.0000 | 1.0000 |
| dinov2_probe | linear probe on DINOv2-small | probe: all 3,000 sampled NUAA training images (dev included); encoder: third-party pretraining, NUAA in its data not verified | 150 | 150 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| depth | Depth Anything V2 (depth statistics) | no NUAA training in our experiment; third-party pretraining, NUAA in its data not verified | 150 | 150 | 0.6933 | 0.0200 | 0.3567 | 0.5533 | 0.4470 |
| lbp | LBP + SVM | SVM: all 3,000 sampled NUAA training images (dev included); no pre-trained component | 150 | 150 | 0.3067 | 0.3333 | 0.3200 | 0.3200 | 0.7819 |
| fourier | Fourier spectrum rule | no training (hand-crafted rule) | 150 | 150 | 0.2400 | 0.0000 | 0.1200 | 0.0667 | 0.9780 |
| ida | IDA features + SVM | SVM: all 3,000 sampled NUAA training images (dev included); no pre-trained component | 150 | 150 | 0.2733 | 0.4400 | 0.3567 | 0.3000 | 0.7910 |
| dog | DoG features + SVM | SVM: all 3,000 sampled NUAA training images (dev included); no pre-trained component | 150 | 150 | 0.0267 | 0.7533 | 0.3900 | 0.3333 | 0.7120 |
| iqa | IQA features + SVM | SVM: all 3,000 sampled NUAA training images (dev included); no pre-trained component | 150 | 150 | 0.3200 | 0.4133 | 0.3667 | 0.3400 | 0.7218 |
