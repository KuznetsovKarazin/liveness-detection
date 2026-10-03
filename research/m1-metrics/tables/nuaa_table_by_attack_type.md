# Evaluation `nuaa` · 2026-10-03 12:48

Command: `scripts/eval_dataset.py --dataset nuaa --from-cache --lang en --by-attack-type --manifest results/c1/nuaa_manifest.csv --table results/m1/nuaa_table_by_attack_type.md`  
Commit: `acf6fdaea06f363ac3071f35eed6114a234beb21` · livedetection `4334db59` · Silent-Face `b6d5f04a`  
Python 3.11.16 · TensorFlow 2.15.0 · Keras 2.15.0 · torch 2.14.0 · transformers 5.17.0 · numpy 1.26.4 · scikit-learn 1.3.2 · OpenCV 4.11.0 · TF devices ['CPU', 'GPU']  
Threshold 0.5 on the attack score (tie = bona fide); labels 0 = bona fide, 1 = attack; unit = image; APCER = accepted attacks / n_attack, BPCER = rejected bona fide / n_bona_fide, ACER = their mean; BPCER@APCER10 at the 0.1 quantile of the attack scores; EER on the full ROC curve, no interpolation; non-computable values = —. Model fingerprints (SHA-256 of the weights or pinned revision): see the CSV.
Cache mode: metrics recomputed from the saved scores; fingerprints read from the cache, not verified against local model files.

| analyzer | n | n_bona_fide | n_attack | apcer | bpcer | acer | eer | bpcer_at_apcer10 | auc | accuracy | n_errors | n_noface |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| livenessnet__nuaa | 300 | 150 | 150 | 0.1600 | 0.7267 | 0.4433 | 0.6600 | 0.7267 | 0.3782 | 0.5567 | 0 | 0 |
| attacknet_v1__nuaa | 300 | 150 | 150 | 0.2800 | 0.2667 | 0.2733 | 0.2800 | 0.6133 | 0.7547 | 0.7267 | 0 | 0 |
| attacknet_v2_1__nuaa | 300 | 150 | 150 | 0.0200 | 0.7667 | 0.3933 | 0.3467 | 0.5733 | 0.7103 | 0.6067 | 0 | 0 |
| attacknet_v2_2__nuaa | 300 | 150 | 150 | 0.1467 | 0.7867 | 0.4667 | 0.4867 | 0.8000 | 0.4985 | 0.5333 | 0 | 0 |
| livenessnet__casia_fasd | 300 | 150 | 150 | 1.0000 | 0.0000 | 0.5000 | 0.2800 | 0.5467 | 0.7468 | 0.5000 | 0 | 0 |
| livenessnet__casia_fasd-pooled | 300 | 150 | 150 | 0.7067 | 0.8267 | 0.7667 | 0.7667 | 1.0000 | 0.1634 | 0.2333 | 0 | 0 |
| attacknet_v1__casia_fasd | 300 | 150 | 150 | 0.1133 | 0.2867 | 0.2000 | 0.1667 | 0.3000 | 0.8106 | 0.8000 | 0 | 0 |
| attacknet_v2_1__casia_fasd | 300 | 150 | 150 | 0.3667 | 0.9800 | 0.6733 | 0.6933 | 1.0000 | 0.2304 | 0.3267 | 0 | 0 |
| attacknet_v2_2__casia_fasd | 300 | 150 | 150 | 0.0000 | 1.0000 | 0.5000 | 0.3267 | 0.9933 | 0.6283 | 0.5000 | 0 | 0 |
| attacknet_v2_2__casia_fasd-pooled | 300 | 150 | 150 | 0.1533 | 0.9733 | 0.5633 | 0.4667 | 0.9867 | 0.5232 | 0.4367 | 0 | 0 |
| livenessnet__celeba_spoof | 300 | 150 | 150 | 0.9133 | 0.0733 | 0.4933 | 0.5467 | 0.6867 | 0.4997 | 0.5067 | 0 | 0 |
| livenessnet__celeba_spoof-pooled | 300 | 150 | 150 | 0.9067 | 0.3533 | 0.6300 | 0.6533 | 0.9600 | 0.2791 | 0.3700 | 0 | 0 |
| attacknet_v1__celeba_spoof | 300 | 150 | 150 | 0.0000 | 1.0000 | 0.5000 | 0.3867 | 0.6600 | 0.6812 | 0.5000 | 0 | 0 |
| attacknet_v2_1__celeba_spoof | 300 | 150 | 150 | 0.3333 | 0.7800 | 0.5567 | 0.6400 | 0.9800 | 0.3351 | 0.4433 | 0 | 0 |
| attacknet_v2_2__celeba_spoof | 300 | 150 | 150 | 0.0000 | 0.9000 | 0.4500 | 0.2533 | 0.6667 | 0.7881 | 0.5500 | 0 | 0 |
| attacknet_v2_2__celeba_spoof-pooled | 300 | 150 | 150 | 0.8200 | 0.0000 | 0.4100 | 0.3833 | 0.8133 | 0.6854 | 0.5900 | 0 | 0 |
| livenessnet__synthaspoof | 300 | 150 | 150 | 0.4800 | 0.7733 | 0.6267 | 0.6600 | 0.9400 | 0.3135 | 0.3733 | 0 | 0 |
| attacknet_v1__synthaspoof | 300 | 150 | 150 | 0.0133 | 0.7933 | 0.4033 | 0.5467 | 0.6067 | 0.4757 | 0.5967 | 0 | 0 |
| attacknet_v2_1__synthaspoof | 300 | 150 | 150 | 0.2733 | 0.7600 | 0.5167 | 0.6667 | 0.8133 | 0.3230 | 0.4833 | 0 | 0 |
| attacknet_v2_2__synthaspoof | 300 | 150 | 150 | 0.5667 | 0.5600 | 0.5633 | 0.5667 | 0.8467 | 0.3627 | 0.4367 | 0 | 0 |
| minifasnet | 300 | 150 | 150 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0 | 0 |
| clip_zeroshot | 300 | 150 | 150 | 0.0133 | 0.0067 | 0.0100 | 0.0133 | 0.0000 | 0.9982 | 0.9900 | 0 | 0 |
| clip_probe | 300 | 150 | 150 | 0.0267 | 0.0000 | 0.0133 | 0.0000 | 0.0000 | 1.0000 | 0.9867 | 0 | 0 |
| depth | 300 | 150 | 150 | 0.6933 | 0.0200 | 0.3567 | 0.5533 | 1.0000 | 0.4470 | 0.6433 | 0 | 0 |
| dinov2_probe | 300 | 150 | 150 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0 | 0 |
| lbp | 300 | 150 | 150 | 0.3067 | 0.3333 | 0.3200 | 0.3200 | 0.5733 | 0.7819 | 0.6800 | 0 | 0 |
| fourier | 300 | 150 | 150 | 0.2400 | 0.0000 | 0.1200 | 0.0667 | 0.0267 | 0.9780 | 0.8800 | 0 | 0 |
| ida | 300 | 150 | 150 | 0.2733 | 0.4400 | 0.3567 | 0.3000 | 0.6600 | 0.7910 | 0.6433 | 0 | 0 |
| dog | 300 | 150 | 150 | 0.0267 | 0.7533 | 0.3900 | 0.3333 | 0.5400 | 0.7120 | 0.6100 | 0 | 0 |
| iqa | 300 | 150 | 150 | 0.3200 | 0.4133 | 0.3667 | 0.3400 | 0.8333 | 0.7218 | 0.6333 | 0 | 0 |
| livenessnet__nuaa-pooled | 0 | not evaluated here: with the 80/20 pooled split of the reference code the images of this dataset are part of the model's training set (its held-out 20 % test is reported in the checkpoint JSON card) |
| attacknet_v1__nuaa-pooled | 0 | not evaluated here: with the 80/20 pooled split of the reference code the images of this dataset are part of the model's training set (its held-out 20 % test is reported in the checkpoint JSON card) |
| attacknet_v2_1__nuaa-pooled | 0 | not evaluated here: with the 80/20 pooled split of the reference code the images of this dataset are part of the model's training set (its held-out 20 % test is reported in the checkpoint JSON card) |
| attacknet_v2_2__nuaa-pooled | 0 | not evaluated here: with the 80/20 pooled split of the reference code the images of this dataset are part of the model's training set (its held-out 20 % test is reported in the checkpoint JSON card) |
| facemesh | 0 | no score (untrained or descriptive model) |

## APCER by attack type

| analyzer | attack_type | n_attack | accepted | apcer | n_bona_fide | rejected_bona_fide | bpcer |
|---|---|---|---|---|---|---|---|
| livenessnet__nuaa | print | 150 | 24 | 0.1600 | 150 | 109 | 0.7267 |
| attacknet_v1__nuaa | print | 150 | 42 | 0.2800 | 150 | 40 | 0.2667 |
| attacknet_v2_1__nuaa | print | 150 | 3 | 0.0200 | 150 | 115 | 0.7667 |
| attacknet_v2_2__nuaa | print | 150 | 22 | 0.1467 | 150 | 118 | 0.7867 |
| livenessnet__casia_fasd | print | 150 | 150 | 1.0000 | 150 | 0 | 0.0000 |
| livenessnet__casia_fasd-pooled | print | 150 | 106 | 0.7067 | 150 | 124 | 0.8267 |
| attacknet_v1__casia_fasd | print | 150 | 17 | 0.1133 | 150 | 43 | 0.2867 |
| attacknet_v2_1__casia_fasd | print | 150 | 55 | 0.3667 | 150 | 147 | 0.9800 |
| attacknet_v2_2__casia_fasd | print | 150 | 0 | 0.0000 | 150 | 150 | 1.0000 |
| attacknet_v2_2__casia_fasd-pooled | print | 150 | 23 | 0.1533 | 150 | 146 | 0.9733 |
| livenessnet__celeba_spoof | print | 150 | 137 | 0.9133 | 150 | 11 | 0.0733 |
| livenessnet__celeba_spoof-pooled | print | 150 | 136 | 0.9067 | 150 | 53 | 0.3533 |
| attacknet_v1__celeba_spoof | print | 150 | 0 | 0.0000 | 150 | 150 | 1.0000 |
| attacknet_v2_1__celeba_spoof | print | 150 | 50 | 0.3333 | 150 | 117 | 0.7800 |
| attacknet_v2_2__celeba_spoof | print | 150 | 0 | 0.0000 | 150 | 135 | 0.9000 |
| attacknet_v2_2__celeba_spoof-pooled | print | 150 | 123 | 0.8200 | 150 | 0 | 0.0000 |
| livenessnet__synthaspoof | print | 150 | 72 | 0.4800 | 150 | 116 | 0.7733 |
| attacknet_v1__synthaspoof | print | 150 | 2 | 0.0133 | 150 | 119 | 0.7933 |
| attacknet_v2_1__synthaspoof | print | 150 | 41 | 0.2733 | 150 | 114 | 0.7600 |
| attacknet_v2_2__synthaspoof | print | 150 | 85 | 0.5667 | 150 | 84 | 0.5600 |
| minifasnet | print | 150 | 0 | 0.0000 | 150 | 0 | 0.0000 |
| clip_zeroshot | print | 150 | 2 | 0.0133 | 150 | 1 | 0.0067 |
| clip_probe | print | 150 | 4 | 0.0267 | 150 | 0 | 0.0000 |
| depth | print | 150 | 104 | 0.6933 | 150 | 3 | 0.0200 |
| dinov2_probe | print | 150 | 0 | 0.0000 | 150 | 0 | 0.0000 |
| lbp | print | 150 | 46 | 0.3067 | 150 | 50 | 0.3333 |
| fourier | print | 150 | 36 | 0.2400 | 150 | 0 | 0.0000 |
| ida | print | 150 | 41 | 0.2733 | 150 | 66 | 0.4400 |
| dog | print | 150 | 4 | 0.0267 | 150 | 113 | 0.7533 |
| iqa | print | 150 | 48 | 0.3200 | 150 | 62 | 0.4133 |

Threshold 0.5 (tie = bona fide); APCER = accepted attacks of the type / n_attack of the type; global BPCER = rejected bona fide / n_bona_fide (bona fide samples have no attack type). Type `print` assigned to all NUAA attacks (the manifest has no attack_type column).
