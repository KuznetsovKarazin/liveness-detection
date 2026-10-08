# NUAA: transformations from the image files to the CNN inputs

How the images listed in the M2 manifests become the derived frames of the training cache and the inputs of the four
NUAA "official" CNNs (LivenessNet, AttackNetV1, AttackNetV2_1, AttackNetV2_2, `__nuaa`). Described from the code of
the repository; parameters are quoted from the source, nothing is measured here.

The checkpoints were trained on 12 September 2026 with an earlier version of `scripts/train_cnn.py`.
The public note `research/c1-livenesslab/checkpoints/PROVENANCE.md` (C1 package) records that the preprocessing chain and the hyper-parameters are the same as in the
current script, which differs only in the global seed, the restore of the best weights also when early stopping does
not fire, and the SHA-256 written in the card. The individual settings listed below are therefore read from the current
script and inferred to apply to the 12 September weights from that statement; they are not recorded separately for that run.

## 1. Files and sampling (`scripts/prepare_datasets.py`, `scripts/train_cnn.py`)

1. The split lists of the mirror (copy of the official lists) define the splits: train lists to `{bonafide,attack}_training`, test lists to
   `{bonafide,attack}_validation` (hard links to the files extracted from the archive; see `naming_schema.md`).
2. `build_arrays`: per class, `spread(sorted file list, 1500)` (1,500 files at a constant step over the sorted list),
   bona fide first, for the training and for the test folders. Images not sampled are the exclusions of the report.

## 2. Preprocessing of each sampled image (`scripts/train_cnn.py`, `preprocess`)

1. `cv2.imread` (BGR, 8 bit). An unreadable image is skipped (none was: the cache holds 3,000 + 3,000 frames).
2. Face detection (`src/tesi_app/face.py`, `get_detector`): RetinaFace (WIDER FACE, Caffe model distributed with
   Silent-Face-Anti-Spoofing) through OpenCV DNN on the image reduced to about 192 px per side, mean BGR (104, 117,
   123) subtracted, most confident detection kept if confidence >= 0.6 and box >= 16 px; otherwise the Haar cascade of
   OpenCV (largest face); if both fail, the whole image is used.
3. Crop around the face box enlarged by a margin of 0.15 per side and made square on the longer side, clipped to the
   image borders (`Face.crop(margin=0.15, square=True)`).
4. Resize to 256 x 256 with `cv2.INTER_AREA`, then BGR to RGB.
5. Enhancement of the reference code, unchanged (`src/livedetection/scripts/create_datasets.py`,
   `EnhancedDatasetCreator.advanced_image_enhancement`): bilateral filter (9, 75, 75); CLAHE on the L channel in LAB
   (clip limit 3.0, tiles 8 x 8); unsharp masking (Gaussian sigma 2.0, weights 1.5 / -0.5); gamma 1.2; final
   `convertScaleAbs(alpha=1.1, beta=5)`.
6. Stored as `uint8` in the cache `data/processed/nuaa_1500.npz` (`X_train`, `X_test`), with the labels (`y_*`) and the
   subject prefixes (`s_*`) in the same order. These arrays are the **derived frames** checked by
   `nuaa_manifests.py build` (hash of each frame, duplicates inside and across arrays).

## 3. Training-time transformations (`scripts/train_cnn.py`, `train`)

1. Pixels divided by 255 (`float32`), labels one-hot (index 0 bona fide, 1 attack).
2. Split of the training array by subject: `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` on the
   subject prefix, giving 2,507 images that fit the weights and 493 validation images (2 subjects). The split is made
   by subject, after the per-image preprocessing of section 2 and before the augmentation; no frame is extracted from
   video (NUAA images are still frames). M2 defines no new split.
3. Augmentation on the fitting images only, inside the `tf.data` pipeline: `RandomFlip("horizontal")`,
   `RandomRotation(0.03)`, `RandomZoom(0.12, 0.12)`, `RandomTranslation(0.06, 0.06)`,
   `RandomBrightness(0.15, value_range=(0, 1))`, `RandomContrast(0.2)`, `GaussianNoise(0.02)`; shuffle buffer 4096,
   seed 42, batch 16. Validation and test images are not augmented.
4. Class weights balancing the two classes of the fitting images.
5. Validation images: no gradient updates; their `val_loss` drives `EarlyStopping(patience=4,
   restore_best_weights=True)` and `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6)`. Adam (learning rate
   1e-3, `clipnorm=1.0`), categorical cross-entropy, at most 12 epochs (C1 `checkpoints/PROVENANCE.md`). When early stopping fires
   the saved weights are those of the epoch with the lowest `val_loss`; when the epoch limit is reached (LivenessNet,
   12 epochs) they are those of the last epoch, as stated in the card note.
6. Test: the 3,000 test frames divided by 255, no augmentation; the card metrics come from them.

## 4. Pretraining

The CNNs are trained from scratch (random initialisation; C1 `inventory.md`); the derived frames use the pre-trained
RetinaFace detector (or the Haar cascade of OpenCV) for the face crop.

## 5. Inference in the application

`scripts/train_cnn.py` states that it uses the same inference path as the application (face crop, 256 x 256,
enhancement, /255); the application loads the `.h5` whose SHA-256 is in the card.

## Not recorded

- The detector used for each image (RetinaFace, Haar or whole image) is not stored in the cache.
- The cache stores no file names: frames are linked to the files only through the sampling order and the subject and
  label arrays.
