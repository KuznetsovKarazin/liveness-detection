# Valutazione `nuaa` · 2026-09-26 18:51

Comando: `scripts/eval_dataset.py --dataset nuaa --from-cache --export results/c1/nuaa_scores.csv --table results/c1/nuaa_table.md`  
Commit: `eb9b32690bd0f60d96e0379774c80e18afcf7430` · livedetection `4334db59` · Silent-Face `b6d5f04a`  
Python 3.11.16 · TensorFlow 2.15.0 · Keras 2.15.0 · torch 2.14.0 · transformers 5.17.0 · numpy 1.26.4 · scikit-learn 1.3.2 · OpenCV 4.11.0 · dispositivi TF ['CPU', 'GPU']  
Soglia 0,5 sul punteggio di attacco (pareggio = bona fide); BPCER@APCER10 con soglia al quantile 0,1 degli attacchi; EER senza interpolazione. Impronte dei modelli (SHA-256 dei pesi o id versione): vedi CSV.

| analizzatore | n | apcer | bpcer | acer | eer | bpcer_at_apcer10 | auc | accuracy | n_errors | n_noface |
|---|---|---|---|---|---|---|---|---|---|---|
| livenessnet__nuaa | 300 | 0.1600 | 0.7267 | 0.4433 | 0.6600 | 0.7267 | 0.3782 | 0.5567 | 0 | 0 |
| attacknet_v1__nuaa | 300 | 0.2800 | 0.2667 | 0.2733 | 0.2800 | 0.6133 | 0.7547 | 0.7267 | 0 | 0 |
| attacknet_v2_1__nuaa | 300 | 0.0200 | 0.7667 | 0.3933 | 0.3467 | 0.5733 | 0.7103 | 0.6067 | 0 | 0 |
| attacknet_v2_2__nuaa | 300 | 0.1467 | 0.7867 | 0.4667 | 0.4867 | 0.8000 | 0.4985 | 0.5333 | 0 | 0 |
| livenessnet__casia_fasd | 300 | 1.0000 | 0.0000 | 0.5000 | 0.2800 | 0.5467 | 0.7468 | 0.5000 | 0 | 0 |
| livenessnet__casia_fasd-pooled | 300 | 0.7067 | 0.8267 | 0.7667 | 0.7667 | 1.0000 | 0.1634 | 0.2333 | 0 | 0 |
| attacknet_v1__casia_fasd | 300 | 0.1133 | 0.2867 | 0.2000 | 0.1667 | 0.3000 | 0.8106 | 0.8000 | 0 | 0 |
| attacknet_v2_1__casia_fasd | 300 | 0.3667 | 0.9800 | 0.6733 | 0.6933 | 1.0000 | 0.2304 | 0.3267 | 0 | 0 |
| attacknet_v2_2__casia_fasd | 300 | 0.0000 | 1.0000 | 0.5000 | 0.3267 | 0.9933 | 0.6283 | 0.5000 | 0 | 0 |
| attacknet_v2_2__casia_fasd-pooled | 300 | 0.1533 | 0.9733 | 0.5633 | 0.4667 | 0.9867 | 0.5232 | 0.4367 | 0 | 0 |
| livenessnet__celeba_spoof | 300 | 0.9133 | 0.0733 | 0.4933 | 0.5467 | 0.6867 | 0.4997 | 0.5067 | 0 | 0 |
| livenessnet__celeba_spoof-pooled | 300 | 0.9067 | 0.3533 | 0.6300 | 0.6533 | 0.9600 | 0.2791 | 0.3700 | 0 | 0 |
| attacknet_v1__celeba_spoof | 300 | 0.0000 | 1.0000 | 0.5000 | 0.3867 | 0.6600 | 0.6812 | 0.5000 | 0 | 0 |
| attacknet_v2_1__celeba_spoof | 300 | 0.3333 | 0.7800 | 0.5567 | 0.6400 | 0.9800 | 0.3351 | 0.4433 | 0 | 0 |
| attacknet_v2_2__celeba_spoof | 300 | 0.0000 | 0.9000 | 0.4500 | 0.2533 | 0.6667 | 0.7881 | 0.5500 | 0 | 0 |
| attacknet_v2_2__celeba_spoof-pooled | 300 | 0.8200 | 0.0000 | 0.4100 | 0.3833 | 0.8133 | 0.6854 | 0.5900 | 0 | 0 |
| livenessnet__synthaspoof | 300 | 0.4800 | 0.7733 | 0.6267 | 0.6600 | 0.9400 | 0.3135 | 0.3733 | 0 | 0 |
| attacknet_v1__synthaspoof | 300 | 0.0133 | 0.7933 | 0.4033 | 0.5467 | 0.6067 | 0.4757 | 0.5967 | 0 | 0 |
| attacknet_v2_1__synthaspoof | 300 | 0.2733 | 0.7600 | 0.5167 | 0.6667 | 0.8133 | 0.3230 | 0.4833 | 0 | 0 |
| attacknet_v2_2__synthaspoof | 300 | 0.5667 | 0.5600 | 0.5633 | 0.5667 | 0.8467 | 0.3627 | 0.4367 | 0 | 0 |
| minifasnet | 300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0 | 0 |
| clip_zeroshot | 300 | 0.0133 | 0.0067 | 0.0100 | 0.0133 | 0.0000 | 0.9982 | 0.9900 | 0 | 0 |
| clip_probe | 300 | 0.0267 | 0.0000 | 0.0133 | 0.0000 | 0.0000 | 1.0000 | 0.9867 | 0 | 0 |
| depth | 300 | 0.6933 | 0.0200 | 0.3567 | 0.5533 | 1.0000 | 0.4470 | 0.6433 | 0 | 0 |
| dinov2_probe | 300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0 | 0 |
| lbp | 300 | 0.3067 | 0.3333 | 0.3200 | 0.3200 | 0.5733 | 0.7819 | 0.6800 | 0 | 0 |
| fourier | 300 | 0.2400 | 0.0000 | 0.1200 | 0.0667 | 0.0267 | 0.9780 | 0.8800 | 0 | 0 |
| ida | 300 | 0.2733 | 0.4400 | 0.3567 | 0.3000 | 0.6600 | 0.7910 | 0.6433 | 0 | 0 |
| dog | 300 | 0.0267 | 0.7533 | 0.3900 | 0.3333 | 0.5400 | 0.7120 | 0.6100 | 0 | 0 |
| iqa | 300 | 0.3200 | 0.4133 | 0.3667 | 0.3400 | 0.8333 | 0.7218 | 0.6333 | 0 | 0 |
| livenessnet__nuaa-pooled | 0 | non valutabile qui: con lo split 80/20 del docente le immagini di questo dataset fanno parte del training del modello (il test del 20 % tenuto fuori è riportato in models/weights/*.json) |
| attacknet_v1__nuaa-pooled | 0 | non valutabile qui: con lo split 80/20 del docente le immagini di questo dataset fanno parte del training del modello (il test del 20 % tenuto fuori è riportato in models/weights/*.json) |
| attacknet_v2_1__nuaa-pooled | 0 | non valutabile qui: con lo split 80/20 del docente le immagini di questo dataset fanno parte del training del modello (il test del 20 % tenuto fuori è riportato in models/weights/*.json) |
| attacknet_v2_2__nuaa-pooled | 0 | non valutabile qui: con lo split 80/20 del docente le immagini di questo dataset fanno parte del training del modello (il test del 20 % tenuto fuori è riportato in models/weights/*.json) |
| facemesh | 0 | nessun punteggio (modello non addestrato o descrittivo) |
