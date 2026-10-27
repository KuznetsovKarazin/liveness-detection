# C2 technical fix: verified CPU execution for LivenessNet training, training device recorded and checked (synthetic tests only)

Task assigned on the N1 card on 27 October 2026, after the merge of PR #11 (`688afc5`), and described on the C2 card: a
separate PR from `research/pad-2026` at `688afc5`, without waiting for CASIA. In `train_cnn.py` and in the C2 runner,
LivenessNet must be created, trained and validated on the CPU only, with the actual device verified and a clear stop
when it is not guaranteed; the device policy of AttackNet is kept and verified, with no global disabling of the GPUs;
the requested and actual devices are recorded in the new training metadata and in the checkpoint card and checked before
scoring; synthetic regressions are added to the existing test harness.

**Technical change only: no dataset, no full training, no C2 experiment, no inference on real data.** Every test uses
synthetic inputs in temporary folders. The accepted packages (`research/c1-livenesslab/`, `research/m1-metrics/`,
`research/m2-*`, `research/n1-nuaa-stability/`) and their results are unchanged; the approved configuration
`research/m2-c2-pilot-protocol/c2_pilot_casia_to_nuaa.json` is unchanged (SHA-256 `4065d853…f84f7a`). No new workaround
of the graph optimizer: the existing `remapping: False` of `train_cnn.py` (for the AttackNets) stays as it was.

## Delivery status

This folder (`research/c2-livenessnet-device/`) is the content of the PR; nothing is uploaded and no run folder exists.
The accepted copies of `train_cnn.py` (in `research/c1-livenesslab/` and `research/m2-c2-pilot-impl/`) and of
`c2_pilot.py` keep the behaviour described in the N1 README ("Same defect in accepted code"); **the copies in this
folder are the ones to use for the C2 run and for any future training with `train_cnn.py`.**

## Why

N1 found that, with tensorflow-metal on an Apple-silicon GPU, the ReLU after the `Dense` that follows the `Flatten` of
LivenessNet is skipped in graph mode (`fit`, `evaluate`, `predict`): the network trains and is validated as a different
model. Minimal reproduction and logs: `research/n1-nuaa-stability/` (`n1_metal_relu_repro.py`, sections "Device" and
"Same defect in accepted code" of its README). The defect is documented by the author's reproducer and logs, not re-run
independently. The AttackNets (dense layers with tanh) are not affected. The device of the historical C1 LivenessNet
training is not recorded and is not proven here.

## What changed

Compared with `research/m2-c2-pilot-impl/livenesslab/` (accepted, PR #8):

- **`scripts/train_cnn.py`**
  - `DEVICE_POLICY`: `"CPU"` for LivenessNet, `"default"` (the device TensorFlow chooses, as before) for AttackNetV1,
    AttackNetV2_1 and AttackNetV2_2. An architecture without a policy is refused.
  - For the CPU policy, model creation, compile, the augmentation layers, `fit` with its validation, the restoration of
    the best weights, the optional test evaluation (`predict`) and the saving of weights and card all run inside
    `tf.device("/CPU:0")`. The GPU is never hidden from the process (`tf.config.set_visible_devices` is not called), so
    an AttackNet trained afterwards in the same process (as `c2_pilot.py train` does) still uses the default device.
  - A GPU present but hidden from the process (`list_physical_devices` shows it, `get_visible_devices` does not) is
    refused before the model is created: under the default policy an AttackNet would otherwise train on the CPU
    silently.
  - Verification, with the actual device:
    - before `fit`: every model variable on the expected device;
    - after `fit`: every model and optimizer variable on the expected device;
    - after `fit`, a functional check of the defect itself, on 4 validation frames inside the same scope: graph mode
      (where the ReLU is skipped) against eager mode for (a) the outputs of every layer, through a new `tf.function`,
      (b) the probabilities of `model.predict` and (c) the loss of `model.evaluate`. (b) and (c) are the functions Keras
      itself builds for the test evaluation and for the validation inside `fit`; the eager loss is `model.compute_loss`
      on the eager outputs. For the CPU policy the largest relative difference must be within `GRAPH_EAGER_TOL` = 1e-5;
      for LivenessNet on the CPU it was exactly 0 on every machine tried (below). Variable placement alone does not
      prove where the kernels ran (a function can compute on the GPU with its variables on the CPU); this comparison
      detects the defect in the graphs it runs, whatever the placement of the variables. It runs in inference mode
      (`training=False`) after training and changes no weights or history. A comparison that is not finite (outputs not
      finite after training) also stops the training, with its own message.
    - Expected device: `CPU:0` for the CPU policy. For `default` it is `GPU:0` when a GPU is visible, otherwise `CPU:0`;
      for the AttackNets this verifies the default policy, so a CPU scope left open would be detected. Their
      graph/eager difference is recorded, not enforced (they are not affected by the defect).
    - Every failure stops with an explicit `ERROR: …` line, before any weights, Keras file, card or `manifest.json`
      entry is written. A device or graph/eager failure of LivenessNet ends with `(CPU execution not guaranteed:
      training stopped, no weights saved)`; a non-finite comparison, a hidden GPU and an architecture without a policy
      have their own messages.
  - The card (and, without `--out-dir`, `manifest.json`) gains one field, `training_device`, placed before `test`; every
    other field and its order are unchanged.

    ```json
    "training_device": {"requested": "CPU", "expected": "CPU:0", "effective": "CPU:0", "visible": ["CPU", "GPU"],
                        "physical": ["CPU", "GPU"], "graph_eager_max_rel_diff": 0.0, "graph_eager_layer": null}
    ```

  - The body of `train` after the device choice moved unchanged into `_train_in_scope`, called inside the scope (with
    the `sklearn` import, which now comes after the seeding instead of before it).
- **`scripts/c2_pilot.py`**
  - `TRAINING_DEVICE` (LivenessNet `"CPU"`, AttackNetV2_2 `"default"`, equal to `DEVICE_POLICY`; a test compares them).
  - `train.json` records `training_device` for each model.
  - The weights check (end of `train`, then before every `score`, `select` and `apply`) adds `device_problems`:
    - the card must record `training_device`;
    - `requested` must follow `TRAINING_DEVICE`;
    - exactly the fields written by `train_cnn.py` (`DEVICE_KEYS`), no more and no fewer;
    - `visible` and `physical` must be sorted lists without repeats, containing `CPU`, made only of `CPU` and `GPU`,
      with every visible device among the physical ones and no GPU present but hidden;
    - `expected` and `effective` must equal the expected device (`CPU:0` for LivenessNet; for AttackNet V2.2 `GPU:0` if
      `GPU` was visible, else `CPU:0`);
    - `graph_eager_max_rel_diff` must be a finite, non-negative number, within `GRAPH_EAGER_TOL` for LivenessNet;
      `graph_eager_layer` must be `null` when it is 0 and a layer name otherwise;
    - the entry in `train.json` must be equal to the card.

  The scorer `tf_scorer` is unchanged (PR #8): scoring is on the CPU for both models, as the approved configuration
  requires (`scoring.device`); it runs after training, inside `tf.device("/CPU:0")`, and hides the GPU only for
  scoring (with `c2_pilot.py all`, in the same process after training, that call has no effect and the CPU scope
  remains). The rule "the GPU is never hidden" applies to training.
- **`scripts/check_c2_pilot.py`**: the fake TensorFlow module gains simulated devices: a visible GPU, none, or one
  present but hidden; a `tf.device` scope stack; graph and eager calls whose difference imitates tensorflow-metal (large
  for LivenessNet when computing on the GPU, about 4e-6 for the AttackNets, 0 on the CPU); and injected faults (scope
  ignored by variable placement, computation on the GPU with variables on the CPU, optimizer variables elsewhere,
  Keras's own `predict` or `evaluate` on the GPU while a new `tf.function` stays in the CPU scope, Dropout noise in
  training mode, an imposed graph/eager difference). New class `TestTrainingDevice`, 31 tests, including both training
  orders in the same process. The behavioural test of the default card now expects the single new field, and the test
  without test evaluation now allows the single `predict` on 4 validation frames of the device check.
- **`scripts/check_c2_mutations.py`**: 48 new declared mutations of the device code in `train_cnn.py` and
  `c2_pilot.py`; the pattern of one earlier mutation (`sweep 55`) follows the extended `train.json` line;
  `m2_writers.py` is copied into each mutated tree when present (`OPTIONAL`), since the maintained manifest modules
  import it.
- **`scripts/check_c2_device_tf.py`** (new): check with the real TensorFlow, synthetic data only (see below).
- Not changed by this task, included so that the folder runs on its own: `c2_pilot.py` imports `rgb_manifests.py` and
  `nuaa_manifests.py`; the copies here are the maintained ones of `research/m2-writers-maintenance/` (byte-identical,
  with the `m2_writers.py` they import), which that PR named as the ones to use for future runs. The other copies are
  byte-identical to `research/m2-c2-pilot-impl/`.

## Check with the real TensorFlow (synthetic data)

`check_c2_device_tf.py` calls `train_cnn.train` as `c2_pilot.py` does (`out_dir`, `test_eval=False`) on 40 random
256×256 frames of 8 fake subjects, 1 epoch, in a temporary folder:

1. LivenessNet then AttackNetV2_2 in the same process;
2. in a subprocess with the GPU hidden (the N1 trainer path): `train` must refuse (GPU present but hidden); then, with
   that one check bypassed in that subprocess only, the same LivenessNet training, comparing the saved weights;
3. negative control: LivenessNet in a subprocess with the `default` policy (on the GPU, as before this change): its
   recorded graph/eager difference must exceed `GRAPH_EAGER_TOL` (the defect) and its weights must differ.

On macOS 27.0.1, Apple M4, Python 3.11.16, TensorFlow 2.15.0, tensorflow-metal 1.1.0:

```
TensorFlow 2.15.0, GPU visible: True
  LivenessNet: requested CPU, effective CPU:0, visible ['CPU', 'GPU'], graph/eager max relative difference 0.000e+00 (None)
  AttackNetV2_2: requested default, effective GPU:0, visible ['CPU', 'GPU'], graph/eager max relative difference 4.647e-06 (evaluate)
  LivenessNet weights, CPU scope with GPU visible vs GPU hidden: 32 arrays, max |diff| = 0.000e+00; that run: effective CPU:0, graph/eager 0.000e+00 (None)
  LivenessNet weights, CPU scope with GPU visible vs GPU, policy default (negative control): 32 arrays, max |diff| = 6.502e+00; that run: effective GPU:0, graph/eager 1.689e+00 (bn_dense)
RESULT: PASS
```

`None` means that every difference is 0. On the GPU the largest graph/eager difference of LivenessNet is in `bn_dense`,
the BatchNormalization right after `dense1`, as expected if its ReLU is skipped. The 6.5 weight difference alone would
not show the defect (random data, GPU non-determinism); the graph/eager difference does.

On Linux without a GPU, in the official `python:3.11` container with TensorFlow 2.15.0, NumPy 1.26.4, scikit-learn
1.3.2, h5py 3.14.0 and OpenCV (`opencv-python-headless` 4.11.0.86), on aarch64 and on x86_64 (emulated on the same
Mac), points 2 and 3 are skipped and point 1 gives `RESULT: PASS`:

```
aarch64: LivenessNet: requested CPU, effective CPU:0, visible ['CPU'], graph/eager max relative difference 0.000e+00 (None)
aarch64: AttackNetV2_2: requested default, effective CPU:0, visible ['CPU'], graph/eager max relative difference 1.213e-06 (evaluate)
x86_64:  LivenessNet: requested CPU, effective CPU:0, visible ['CPU'], graph/eager max relative difference 0.000e+00 (None)
x86_64:  AttackNetV2_2: requested default, effective CPU:0, visible ['CPU'], graph/eager max relative difference 7.174e-07 (evaluate)
```

It needs TensorFlow, h5py, OpenCV (imported by `train_cnn.py`) and the reference architectures
(`LIVENESSLAB_LIVEDETECTION`); without TensorFlow it prints `SKIP` and exits with 2, and without a GPU it skips points 2
and 3 (declared) and expects `CPU:0` for both models. It is not part of the offline suite. In its log, the trainer's own
lines are in Italian and the cards are labelled `casia_fasd` (the dataset argument of `train_cnn.train`): the data are
random.

## Difference from the approved protocol

The approved configuration says `training.device`: "Metal GPU (not deterministic; evidence = saved weights SHA-256)"
for both trainings. With this change LivenessNet is trained on the CPU, as requested; AttackNet V2.2 stays on the
default device. The configuration file is not edited (its SHA-256 is a constant of `c2_pilot.py` and of the approval);
the device actually used is recorded in each card and in `train.json`. Whether to amend the configuration text is the
coordinator's decision.

Expected cost, an estimate not measured on C2 data: in N1 the CPU training of LivenessNet took 496–671 s (9–12
epochs, early stopping) on 2,507 fitting frames; C2 fits on 1,339 CASIA-FASD frames, so the LivenessNet training is
expected to stay well within the 120-minute cap of the budget, which covers the whole `train` step.

## Verification without data, weights or TensorFlow

From inside `livenesslab/`, with numpy and scikit-learn (TensorFlow is never imported: the training goes through the
real `train_cnn.train` with a fake `tensorflow` module):

```bash
python scripts/check_c2_pilot.py
python scripts/check_c2_mutations.py
```

In a git work tree with `../../m2-c2-pilot-protocol/` next to this folder (as in `research/` of the group repository)
the last two lines of `check_c2_pilot.py` are:

```
tests run: 216, failures: 0, errors: 0, skipped: 0
RESULT: PASS
```

`check_c2_mutations.py` applies 362 declared mutations (314 of PR #8 and 48 new ones), each to a fresh copy of the code,
and counts a mutation only if the suite ran to its summary with at least one failure or error:

```
mutations killed: 362/362
RESULT: PASS
```

With `--block tensorflow,keras,h5py,cv2` the suite prints the OpenCV `NOTE`, `blocked imports attempted: 1; blocked
packages loaded: none` and the same `tests run: 216, failures: 0, errors: 0, skipped: 0`, `RESULT: PASS`.

Verified on macOS 27.0.1 (Apple M4, Python 3.11.16, NumPy 1.26.4, scikit-learn 1.3.2). Also run on Linux, in the
official `python:3.12` container (Python 3.12.15, aarch64, NumPy 2.5.4, scikit-learn 1.9.1, without TensorFlow and
OpenCV), from this folder laid out as in `research/`: after the OpenCV `NOTE` line, `tests run: 216, failures: 0,
errors: 0, skipped: 0`, `RESULT: PASS`; there `check_c2_mutations.py` gives `mutations killed: 362/362`. The offline
suite was not run on x86_64 by the author (only the check with the real TensorFlow above).

## Test logs

Run from this package layout (git work tree, `../../m2-c2-pilot-protocol/` next to it), except where stated:

- `test_logs/check_c2_pilot.txt`: `python scripts/check_c2_pilot.py`;
- `test_logs/check_c2_pilot_block.txt`: `python scripts/check_c2_pilot.py --block tensorflow,keras,h5py,cv2`;
- `test_logs/check_c2_mutations.txt`: `python scripts/check_c2_mutations.py` (the bare copy it tests has neither the
  configuration nor git, so its first line ends with `skipped: 2`);
- `test_logs/check_c2_device_tf.txt`: `python scripts/check_c2_device_tf.py` (with `LIVENESSLAB_LIVEDETECTION` set);
- `test_logs/linux.txt`: the Linux runs above (`python:3.12`: last lines of the suite and of the mutations;
  `python:3.11`: the check with the real TensorFlow on aarch64 and x86_64), after the installation lines; Keras
  progress bars removed.

## Not done, limits

- No training, scoring or check on real data; the real training was exercised only on 40 synthetic frames for one
  epoch.
- The run-time check proves where the variables are and that, after training, graph mode computes the same layer
  outputs, `predict` probabilities and `evaluate` loss as eager mode on 4 validation frames; it does not log the
  placement of every kernel, and it cannot run Keras's training step itself without changing the weights. That the CPU
  scope also covers the training step is shown on this Mac by the weights equal, bit for bit, to the GPU-hidden run
  (synthetic data, one epoch). The input pipeline (`tf.data`, augmentation) runs on the host CPU for both models,
  as before.
- The AttackNet default policy is checked against the devices of the training process: a GPU present but hidden is
  refused, but whether a machine should have a GPU at all is not checked (on a machine without a GPU both models train
  on the CPU and record it).
- The Metal defect itself is documented by the author's reproducer and logs (N1), not re-run independently; the
  historical C1 device is not proven.
- The real TensorFlow was run on macOS with a GPU (Apple M4) and on Linux without a GPU (containers, aarch64 and
  emulated x86_64); not on Windows or with a CUDA GPU.
- Versions as in `requirements.txt` (TensorFlow and Keras 2.15). Keras 3 (TensorFlow 2.16 or later) is not supported by
  `train_cnn.py`, as before this change.

## Files and integrity

This `README.md`, `SHA256SUMS`, `test_logs/` (5 files) and `livenesslab/` (`requirements.txt`; `scripts/`:
`c2_pilot.py`, `check_c2_pilot.py`, `check_c2_mutations.py`, `check_c2_device_tf.py`, `train_cnn.py`,
`select_threshold.py`, `path_guard.py`, `rgb_manifests.py`, `nuaa_manifests.py`, `m2_writers.py`, `package_rules.py`;
`src/tesi_app/`: `__init__.py`, `paths.py`, `evaluation.py`, `core.py`, `face.py`): 23 files listed in `SHA256SUMS`,
plus `SHA256SUMS`. From inside this folder:

```bash
shasum -a 256 -c SHA256SUMS && test -z "$(find . -type l)" && diff <(awk '{print $2}' SHA256SUMS | sed 's|^\*||' | sort) <(find . -type f ! -type l ! -name SHA256SUMS ! -path '*/__pycache__/*' | sed 's|^\./||' | sort) && echo COMPLETE
```
