# CASIA-FASD and CelebA-Spoof: transformations from the image files to the CNN inputs

How the images listed in the M2 manifests of CASIA-FASD and CelebA-Spoof become the derived frames of the training
caches and the inputs of the twelve checkpoints of these two datasets (four "official" and two pooled per dataset).
Described from the code of the repository; parameters are quoted from the source, nothing is measured here.

The checkpoints were trained on 12 September 2026 with a version of `scripts/train_cnn.py` that precedes the
repository (`models/weights/PROVENANCE.md`); the closest versioned version is commit `0ac7143`, whose split logic is
used to rebuild the splits. The preprocessing chain is the one of the NUAA tranche (`transformations.md`), which
records the same statement for the NUAA checkpoints; the individual settings below are read from the code and
inferred to apply to the 12 September weights, they are not recorded separately for that run.

## 1. Files before the CNN pipeline

- **CASIA-FASD.** The mirror author extracted the frames from the original videos and cropped them (256 x 256 JPEG,
  method not documented); no video is available, so no frame is extracted by us. `scripts/prepare_datasets.py --only
  casia` links (hard links, same names) the colour frames of `train_img` into `{bonafide,attack}_training` and those of
  `test_img` into `{bonafide,attack}_validation`, by the class suffix; the `depth/` maps are not used. The 300
  evaluation images are `spread(test_img, 150)` per class, also hard links.
- **CelebA-Spoof.** The mirror author cropped the faces (method not documented). The parquet was converted once outside
  the repository: the PNG bytes of each row were written unchanged under `celeba_<row>.jpg` (byte-identical, verified);
  the row sets follow the rule of `naming_schema_rgb.md`. The 300 evaluation images are separate, byte-identical
  copies of `_validation` files.

## 2. Sampling and preprocessing (`scripts/train_cnn.py`, `build_arrays` and `preprocess`)

1. Per class, `spread(sorted file list, 1500)`, bona fide first, for the training and for the test folders. CASIA-FASD:
   all 1,655 training frames, all 591 bona fide test frames and 1,500 of the 1,817 attack test frames. CelebA-Spoof:
   all 3,000 training and 900 test images.
2. Each sampled image: `cv2.imread`; face detection (RetinaFace through OpenCV DNN, Haar cascade as fallback, whole
   image if both fail); square crop with a margin of 0.15; resize to 256 x 256 with `cv2.INTER_AREA`; BGR to RGB;
   enhancement of the reference code (bilateral filter, CLAHE, unsharp masking, gamma, `convertScaleAbs`). Details in
   `transformations.md` of the NUAA tranche; the code is the same for every dataset. CelebA-Spoof files are PNG with a
   `.jpg` extension: `cv2.imread` decodes them by content.
3. Stored as `uint8` in `data/processed/casia_fasd_1500.npz` and `celeba_spoof_1500.npz` (`X_train`, `X_test`), with the
   labels (`y_*`) and the groups (`s_*`) in the same order: CASIA-FASD the subject number (prefix of the name),
   CelebA-Spoof the file name. No image was skipped (cache sizes equal to the sampled lists). These arrays are the
   **derived frames**: `build` hashes each frame as stored and counts duplicates; frame bytes are never exported.

## 3. Splits of the checkpoints (logic of commit `0ac7143`, `train`)

1. **Official.** Fitting and early-stopping images from the training array: `GroupShuffleSplit(n_splits=1,
   test_size=0.2, random_state=42)` on `s_train` when it has at least 5 distinct groups. CASIA-FASD: by subject (16
   subjects fit, 4 held out; 1,339 / 316 images). CelebA-Spoof: the groups are file names, one image per group, so the
   split is per image (2,400 / 600). Test: the test array (CASIA-FASD 2,091, CelebA-Spoof 900). The split is made after
   the per-image preprocessing and before the augmentation.
2. **Pooled.** Training and test arrays concatenated; `train_test_split(train_size=0.8, stratify, random_state=42)`;
   on the training part, without groups, `train_test_split(test_size=0.1, stratify, random_state=42)`. Per frame or per
   image: CASIA-FASD 2,696 / 300 / 750, CelebA-Spoof 2,808 / 312 / 780. The current script uses 20 % of validation also
   in the pooled protocol and would not reproduce these counts.
3. Augmentation on the fitting images only (flip, rotation, zoom, translation, brightness, contrast, Gaussian noise;
   shuffle buffer 4096, seed 42, batch 16), class weights, Adam (1e-3, `clipnorm=1.0`), categorical cross-entropy.
   Early-stopping images: no gradient updates; their `val_loss` drives `EarlyStopping(patience=4,
   restore_best_weights=True)` and `ReduceLROnPlateau(factor=0.5, patience=2)`. Test images: divided by 255, no
   augmentation; the card metrics come from them.

## 4. Pretraining

The CNNs are trained from scratch; the derived frames use the pre-trained RetinaFace detector (or the Haar cascade)
for the face crop.

## Not recorded

- The detector used for each image is not stored in the cache.
- The caches store no file names: frames are linked to the files only through the sampling order and the label and
  group arrays.
- The training run did not save its file list or its split: both are rebuilt and verified against the card counts and
  the cache arrays.
