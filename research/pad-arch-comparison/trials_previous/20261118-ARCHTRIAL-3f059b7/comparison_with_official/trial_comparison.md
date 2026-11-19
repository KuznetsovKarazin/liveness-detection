# Comparison of two technical trials: 20261119-ARCHTRIAL-b8f7cca against 20261118-ARCHTRIAL-3f059b7

Outcome: **reproduced** (reproduced only if both networks have the same, non-zero number of epochs and difference 0 in loss, accuracy, val_loss, val_accuracy, lr; no tolerance).

| trial | role | format | status | commit | configuration SHA-256 | machine | threads as recorded |
|---|---|---|---|---|---|---|---|
| 20261119-ARCHTRIAL-b8f7cca | new | pad-arch-trial/2 | PASS | b8f7cca | 5a83325e13d29d1a67dcda4f129838d19ec39b8f119f5de647693b68c5128b0a | Apple M5 Pro, 15 logical CPUs, 24.0 GB RAM, macOS 27.2.1 | {"effective": {"inter_op": 10, "intra_op": 10}, "environment": {"TF_NUM_INTEROP_THREADS": null, "TF_NUM_INTRAOP_THREADS": null}, "logical_cpus": 15, "requested": {"inter_op": 10, "intra_op": 10}, "set_by_this_runner": true} |
| 20261118-ARCHTRIAL-3f059b7 | reference | pad-arch-trial/1 | PASS | 3f059b7 | 9264ec0201c148519b3cd07a5e8493ff49b25aadfbd2680636d0a4b0d9bcaab8 | Apple M4, 10 logical CPUs, 16.0 GB RAM, macOS 27.2.1 | {"inter_op_parallelism_threads": 0, "intra_op_parallelism_threads": 0, "logical_cpus": 10, "set_by_this_runner": false} |

Notes on the threads: 20261119-ARCHTRIAL-b8f7cca: format pad-arch-trial/2: threads set by the runner; requested, effective (read back) and environment variables recorded; 20261118-ARCHTRIAL-3f059b7: format pad-arch-trial/1: the runner did not set the threads; the recorded values are those read from the getters of tf.config.threading (0 = not set through tf.config.threading); format /1 does not record the environment variables.

How the threads were imposed in the reference trial 20261118-ARCHTRIAL-3f059b7: not recorded.

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

Files read (SHA-256): 20261119-ARCHTRIAL-b8f7cca status.json eabd17a7f2431cf792a29598002012e2431db75ef543c37474841805263fb094; 20261119-ARCHTRIAL-b8f7cca trial.json d44a82c80bae0f5d54e3df8b07135993e79df75929b2eabd079b97547cf43c80; 20261118-ARCHTRIAL-3f059b7 status.json b722183c10cdd4890373f9b27dc37af1d9ebe350bb9f344a122c3077ded0d13a; 20261118-ARCHTRIAL-3f059b7 trial.json 56d20023ddb62ca49d2f2ea972b2dd5c5d9161f7d2d33b71beae2ae47611c13b.
