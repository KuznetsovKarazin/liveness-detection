# Provenance of the checkpoints (models/weights)

English version for the C1 delivery of the note kept with the weights (`models/weights/PROVENANCE.md`, in Italian).
The weight files are not in the repository; their cards (`*.json`) and SHA-256 are.

## CNN checkpoints of 2026-09-12 (NUAA, CASIA-FASD, CelebA-Spoof; official and pooled protocols)

- Trained on 2026-09-12 on the thesis Mac (Apple Silicon, TensorFlow 2.15 + tensorflow-metal) with a version of
  `scripts/train_cnn.py` that predates the creation of the repository (2026-09-16). Preprocessing (RetinaFace crop
  with 15 % margin, 256×256 INTER_AREA, BGR→RGB, reference enhancement, /255), hyper-parameters (Adam 1e-3, batch 16,
  early stopping on val_loss with patience 4, `--per-class 1500`, at most 12 epochs) and protocols (official / pooled)
  are those of the current script. Differences of the current script: global seed recorded in the card, best weights
  restored even without early stopping, SHA-256 written in the card at save time. No seed is recorded for these
  checkpoints (seed 42 was used only for splitting and shuffling).
- `epochs_run` = 12 means the run reached the epoch limit without early stopping: the weights are those of the last
  epoch, not of the epoch with the minimum validation loss (`best_val_loss` refers to the latter).
- `test` metrics in the cards were computed at training time on the checkpoint's own test split (official test set,
  or the held-out 20 % for the pooled protocol); `bpcer_at_apcer10` was removed (inverted formula at the time) and
  `hter` was dropped (identical to ACER).
- Files: each checkpoint exists as `<Arch>__<dataset>[-pooled].h5` (classic HDF5 weights, the file loaded by the
  application; `weights_sha256` in the card is its hash, computed on 2026-09-24) and as a `.keras` copy saved by the
  same training run (hash listed in `config/nuaa_config.json`, `other_formats`; not used by the evaluation).
- **Reconciliation of 2026-09-28.** Seven checkpoints trained in the morning of 2026-09-12 (`LivenessNet__nuaa`,
  `LivenessNet__nuaa-pooled`, `LivenessNet__casia_fasd`, `LivenessNet__celeba_spoof`, `AttackNetV2_2__nuaa`,
  `AttackNetV2_2__casia_fasd`, `AttackNetV2_2__celeba_spoof`) have a `.keras` file dated at training time
  (12:31–13:19) and an `.h5` file written at 15:21 of the same day, when the HDF5 export was added to the training
  script and applied to the checkpoints already trained; the other seventeen were written in both formats at
  training time. A tensor-by-tensor comparison made on 2026-09-28 (models loaded from both files, `np.array_equal`
  on every weight array) found the seven `.h5` files identical to their `.keras` sources. The `saved` field of the
  cards is the training time; the file modification time of those seven `.h5` files is 15:21.

## CNN checkpoints on SynthASpoof (2026-09-25)

Trained on 2026-09-25 with the current script (`scripts/train_cnn.py --dataset synthaspoof --per-class 1500 --epochs 12`,
seed 42, protocol "official" meaning our own split by synthetic identity: identities 1–9 of every ten for training, the
tenth for test, `scripts/prepare_datasets.py --only synthaspoof`). Dataset: Fang, Huber, Damer, CVPR Workshops 2023,
authors' public Google Drive link, CC BY-NC-SA 4.0. Cards contain `weights_sha256`, `epochs_run` and test metrics on
the synthetic held-out part; saved weights are those of the epoch with the minimum validation loss (validation by
identity). `.h5` and `.keras` written together at training time.

## Classifiers of the classic methods and linear probes (2026-09-25)

`<id>_clf.joblib` (lbp, dog, iqa, ida, clip_probe, dinov2_probe) produced by `scripts/train_classic.py --dataset nuaa
--per-class 1500` (seed 42, official NUAA train/test split, features on the RetinaFace crop with each analyzer's `CROP`).
SVM RBF with standardisation for hand-crafted features, logistic regression for embeddings; `class_weight="balanced"`.
Cards (`<id>_clf.json`) report dataset, counts, classifier, hyper-parameters, test metrics on the official test set,
SHA-256 of the training script and of the joblib (`clf_sha256`, the analyzer fingerprint: replacing the file
invalidates the cached scores). Frozen encoders: CLIP ViT-B/32 (`openai/clip-vit-base-patch32@3d74acf9`) and
DINOv2-small (`facebook/dinov2-small@ed25f3a3`), downloaded with `scripts/download_models.py`.

## Training data

`data/<DS>/images` derive from the public mirrors named in `inventory.md` through `scripts/prepare_datasets.py`
(NUAA, CASIA-FASD, SynthASpoof) and from the conversion of shard 0 of the Hugging Face mirror of CelebA-Spoof.
