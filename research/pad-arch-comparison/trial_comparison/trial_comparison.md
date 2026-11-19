# Comparison of two technical trials: 20261119-ARCHTRIAL-b8f7cca against 20261119-ARCHTRIAL-b8e78b2

Outcome: **reproduced** (reproduced only if both networks have the same, non-zero number of epochs and difference 0 in loss, accuracy, val_loss, val_accuracy, lr; no tolerance).

| trial | role | format | status | commit | configuration SHA-256 | machine | threads as recorded |
|---|---|---|---|---|---|---|---|
| 20261119-ARCHTRIAL-b8f7cca | new | pad-arch-trial/2 | PASS | b8f7cca | 5a83325e13d29d1a67dcda4f129838d19ec39b8f119f5de647693b68c5128b0a | Apple M5 Pro, 15 logical CPUs, 24.0 GB RAM, macOS 27.2.1 | {"effective": {"inter_op": 10, "intra_op": 10}, "environment": {"TF_NUM_INTEROP_THREADS": null, "TF_NUM_INTRAOP_THREADS": null}, "logical_cpus": 15, "requested": {"inter_op": 10, "intra_op": 10}, "set_by_this_runner": true} |
| 20261119-ARCHTRIAL-b8e78b2 | reference | pad-arch-trial/1 | PASS | b8e78b2 | 9264ec0201c148519b3cd07a5e8493ff49b25aadfbd2680636d0a4b0d9bcaab8 | Apple M5 Pro, 15 logical CPUs, 24.0 GB RAM, macOS 27.2.1 | {"inter_op_parallelism_threads": 0, "intra_op_parallelism_threads": 0, "logical_cpus": 15, "set_by_this_runner": false} |

Notes on the threads: 20261119-ARCHTRIAL-b8f7cca: format pad-arch-trial/2: threads set by the runner; requested, effective (read back) and environment variables recorded; 20261119-ARCHTRIAL-b8e78b2: format pad-arch-trial/1: the runner did not set the threads; the recorded values are those read from the getters of tf.config.threading (0 = not set through tf.config.threading); format /1 does not record the environment variables.

How the threads were imposed in the reference trial 20261119-ARCHTRIAL-b8e78b2: command line `TF_NUM_INTRAOP_THREADS=10 TF_NUM_INTEROP_THREADS=10 .venv/bin/python -I -B scripts/pad_arch_comparison.py trial --i-am-authorized --out-root <folder outside the repository>` in the conditions file reference_conditions.txt (SHA-256 591e60bfd5644d356e0579866ff2d4fad089cf9ffab4a7f04cb6adbd545c0e41); declared by a conditions file written by the author around the reference trial, not by its record (the format pad-arch-trial/1 does not record the environment variables); checked by the code: the file holds the commit of the reference record as a whole word, and its one line with both variables sets TF_NUM_INTRAOP_THREADS and TF_NUM_INTEROP_THREADS to integers equal to the threads required of the configuration.

| network | field | epochs (new / reference) | epochs compared | first differing epoch | largest absolute difference | epoch of the largest difference |
|---|---|---|---|---|---|---|
| LivenessNet | loss | 3 / 3 | 3 | none | 0 | none |
| LivenessNet | accuracy | 3 / 3 | 3 | none | 0 | none |
| LivenessNet | val_loss | 3 / 3 | 3 | none | 0 | none |
| LivenessNet | val_accuracy | 3 / 3 | 3 | none | 0 | none |
| LivenessNet | lr | 3 / 3 | 3 | none | 0 | none |
| ResNet-18 | loss | 3 / 3 | 3 | none | 0 | none |
| ResNet-18 | accuracy | 3 / 3 | 3 | none | 0 | none |
| ResNet-18 | val_loss | 3 / 3 | 3 | none | 0 | none |
| ResNet-18 | val_accuracy | 3 / 3 | 3 | none | 0 | none |
| ResNet-18 | lr | 3 / 3 | 3 | none | 0 | none |

Files read (SHA-256): 20261119-ARCHTRIAL-b8f7cca status.json eabd17a7f2431cf792a29598002012e2431db75ef543c37474841805263fb094; 20261119-ARCHTRIAL-b8f7cca trial.json d44a82c80bae0f5d54e3df8b07135993e79df75929b2eabd079b97547cf43c80; 20261119-ARCHTRIAL-b8e78b2 status.json 8e77ee8e449b44e845c60ae5c5c0b1a0fe5c2ecc39e508622e730eab1264b1ed; 20261119-ARCHTRIAL-b8e78b2 trial.json 7c252328261367cf7fbfefbedcb0d20ba5350d3e1d7ed55dfc125b8e39e96021.
