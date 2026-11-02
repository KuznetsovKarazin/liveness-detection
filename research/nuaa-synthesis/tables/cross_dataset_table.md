# Annex: the 16 cross-dataset CNN rows of the C1 table on the same 300 NUAA images (threshold 0.5, run `20260928-C1-seed42-a80f6f9`)

Checkpoints of 2026-09-12 (CASIA-FASD, CelebA-Spoof) and 2026-09-25 (SynthASpoof) applied to NUAA without any adaptation (dataset transfer, fixed threshold 0.5, no source-dev threshold). Source: `c1-livenesslab/tables/nuaa_table.md`.

| analyzer | model | trained on | protocol | n bona fide | n attack | APCER | BPCER | ACER | EER | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| livenessnet__casia_fasd | LivenessNet | CASIA-FASD | official split of the source dataset | 150 | 150 | 1.0000 | 0.0000 | 0.5000 | 0.2800 | 0.7468 |
| livenessnet__casia_fasd-pooled | LivenessNet | CASIA-FASD | pooled 80/20 (random over images) | 150 | 150 | 0.7067 | 0.8267 | 0.7667 | 0.7667 | 0.1634 |
| attacknet_v1__casia_fasd | AttackNet V1 | CASIA-FASD | official split of the source dataset | 150 | 150 | 0.1133 | 0.2867 | 0.2000 | 0.1667 | 0.8106 |
| attacknet_v2_1__casia_fasd | AttackNet V2.1 | CASIA-FASD | official split of the source dataset | 150 | 150 | 0.3667 | 0.9800 | 0.6733 | 0.6933 | 0.2304 |
| attacknet_v2_2__casia_fasd | AttackNet V2.2 | CASIA-FASD | official split of the source dataset | 150 | 150 | 0.0000 | 1.0000 | 0.5000 | 0.3267 | 0.6283 |
| attacknet_v2_2__casia_fasd-pooled | AttackNet V2.2 | CASIA-FASD | pooled 80/20 (random over images) | 150 | 150 | 0.1533 | 0.9733 | 0.5633 | 0.4667 | 0.5232 |
| livenessnet__celeba_spoof | LivenessNet | CelebA-Spoof | official split of the source dataset | 150 | 150 | 0.9133 | 0.0733 | 0.4933 | 0.5467 | 0.4997 |
| livenessnet__celeba_spoof-pooled | LivenessNet | CelebA-Spoof | pooled 80/20 (random over images) | 150 | 150 | 0.9067 | 0.3533 | 0.6300 | 0.6533 | 0.2791 |
| attacknet_v1__celeba_spoof | AttackNet V1 | CelebA-Spoof | official split of the source dataset | 150 | 150 | 0.0000 | 1.0000 | 0.5000 | 0.3867 | 0.6812 |
| attacknet_v2_1__celeba_spoof | AttackNet V2.1 | CelebA-Spoof | official split of the source dataset | 150 | 150 | 0.3333 | 0.7800 | 0.5567 | 0.6400 | 0.3351 |
| attacknet_v2_2__celeba_spoof | AttackNet V2.2 | CelebA-Spoof | official split of the source dataset | 150 | 150 | 0.0000 | 0.9000 | 0.4500 | 0.2533 | 0.7881 |
| attacknet_v2_2__celeba_spoof-pooled | AttackNet V2.2 | CelebA-Spoof | pooled 80/20 (random over images) | 150 | 150 | 0.8200 | 0.0000 | 0.4100 | 0.3833 | 0.6854 |
| livenessnet__synthaspoof | LivenessNet | SynthASpoof | own split by synthetic identity | 150 | 150 | 0.4800 | 0.7733 | 0.6267 | 0.6600 | 0.3135 |
| attacknet_v1__synthaspoof | AttackNet V1 | SynthASpoof | own split by synthetic identity | 150 | 150 | 0.0133 | 0.7933 | 0.4033 | 0.5467 | 0.4757 |
| attacknet_v2_1__synthaspoof | AttackNet V2.1 | SynthASpoof | own split by synthetic identity | 150 | 150 | 0.2733 | 0.7600 | 0.5167 | 0.6667 | 0.3230 |
| attacknet_v2_2__synthaspoof | AttackNet V2.2 | SynthASpoof | own split by synthetic identity | 150 | 150 | 0.5667 | 0.5600 | 0.5633 | 0.5667 | 0.3627 |
