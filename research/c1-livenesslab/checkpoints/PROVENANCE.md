# Provenienza dei pesi in questa cartella

I file `.h5` / `.keras` (esclusi dal repository, ~3 GB) sono stati addestrati il 12/09/2026 sul Mac della tesi (Apple Silicon,
TensorFlow 2.15 + tensorflow-metal) con una versione di `scripts/train_cnn.py` precedente alla creazione del repository
(16/09/2026). La catena di preprocessing (RetinaFace, ritaglio 15 %, 256×256 INTER_AREA, enhancement del docente, /255),
gli iperparametri (Adam 1e-3, batch 16, early stopping su val_loss con pazienza 4, `--per-class 1500`, max 12 epoche) e i
protocolli (official / pooled) sono gli stessi dello script attuale; le differenze dello script attuale sono: seed globale
registrato, ripristino dei pesi migliori anche senza early stopping, hash SHA-256 scritto nel JSON.

Per ogni file `<Arch>__<dataset>[-pooled].json`:
- `weights_sha256`: SHA-256 del file `.h5` corrispondente (calcolato il 24/09/2026), la stessa impronta usata dalla cache di valutazione;
- `test`: metriche sul test (split ufficiale, oppure il 20 % tenuto fuori per il protocollo pooled) calcolate al momento
  dell'addestramento; `bpcer_at_apcer10` è stata rimossa (formula invertita all'epoca) e `hter` tolto (coincideva con ACER);
- `epochs_run` = 12 significa che il training ha raggiunto il limite di epoche senza early stopping: i pesi sono dell'ultima
  epoca, non di quella con la val_loss minima (`best_val_loss` si riferisce a quest'ultima).

I dataset di addestramento (`data/<DS>/images`) derivano dai mirror pubblici indicati in `data/eval/<id>/info.json` tramite
`scripts/prepare_datasets.py` (NUAA, CASIA-FASD) e dalla conversione dello shard 0 del mirror Hugging Face di CelebA-Spoof.

## SynthASpoof (25/09/2026)

I file `<Arch>__synthaspoof.*` sono stati addestrati il 25/09/2026 con lo script attuale (`scripts/train_cnn.py --dataset synthaspoof
--per-class 1500 --epochs 12`, seed 42, protocollo official): training sulle identità sintetiche 1–9 di ogni decina, test sulla
decima (`scripts/prepare_datasets.py --only synthaspoof`). Dataset: Fang, Huber, Damer, CVPR Workshops 2023, Google Drive
ufficiale, CC BY-NC-SA 4.0. Il JSON contiene già `weights_sha256`, `epochs_run` e le metriche sul test sintetico; i pesi salvati sono
quelli dell'epoca con la val_loss minima (validazione per identità).

## Classificatori dei metodi classici e delle sonde lineari (25/09/2026)

I file `<id>_clf.joblib` (lbp, dog, iqa, ida, clip_probe, dinov2_probe) sono prodotti da `scripts/train_classic.py --dataset nuaa
--per-class 1500` (seed 42, protocollo official: split ufficiale train/test di NUAA, feature sul ritaglio RetinaFace con il `CROP`
di ciascun analizzatore). SVM RBF con standardizzazione per le feature fatte a mano, regressione logistica per gli embedding;
`class_weight="balanced"`. La scheda `<id>_clf.json` riporta dataset, numeri, classificatore, metriche sul test ufficiale,
SHA-256 dello script e del joblib. L'impronta dell'analizzatore è l'hash del joblib: sostituendo il file la cache dei punteggi
si azzera da sola. Encoder congelati: CLIP ViT-B/32 (`openai/clip-vit-base-patch32@3d74acf9`) e DINOv2-small
(`facebook/dinov2-small@ed25f3a3`), scaricati con `scripts/download_models.py`.
