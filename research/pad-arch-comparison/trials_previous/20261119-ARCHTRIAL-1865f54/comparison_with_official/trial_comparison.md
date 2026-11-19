# Comparison of two technical trials: 20261119-ARCHTRIAL-b8f7cca against 20261119-ARCHTRIAL-1865f54

Outcome: **reproduced** (reproduced only if both networks have the same, non-zero number of epochs and difference 0 in loss, accuracy, val_loss, val_accuracy, lr; no tolerance).

| trial | role | format | status | commit | configuration SHA-256 | machine | threads as recorded |
|---|---|---|---|---|---|---|---|
| 20261119-ARCHTRIAL-b8f7cca | new | pad-arch-trial/2 | PASS | b8f7cca | 5a83325e13d29d1a67dcda4f129838d19ec39b8f119f5de647693b68c5128b0a | Apple M5 Pro, 15 logical CPUs, 24.0 GB RAM, macOS 27.2.1 | {"effective": {"inter_op": 10, "intra_op": 10}, "environment": {"TF_NUM_INTEROP_THREADS": null, "TF_NUM_INTRAOP_THREADS": null}, "logical_cpus": 15, "requested": {"inter_op": 10, "intra_op": 10}, "set_by_this_runner": true} |
| 20261119-ARCHTRIAL-1865f54 | reference | pad-arch-trial/2 | PASS | 1865f54 | 2ade32c6d9bd48b1ba63cd29cbcd811f2f8ec85e0d2816e0603351d94f03b539 | Apple M5 Pro, 15 logical CPUs, 24.0 GB RAM, macOS 27.2.1 | {"effective": {"inter_op": 10, "intra_op": 10}, "environment": {"TF_NUM_INTEROP_THREADS": null, "TF_NUM_INTRAOP_THREADS": null}, "logical_cpus": 15, "requested": {"inter_op": 10, "intra_op": 10}, "set_by_this_runner": true} |

Notes on the threads: 20261119-ARCHTRIAL-b8f7cca: format pad-arch-trial/2: threads set by the runner; requested, effective (read back) and environment variables recorded; 20261119-ARCHTRIAL-1865f54: format pad-arch-trial/2: threads set by the runner; requested, effective (read back) and environment variables recorded.

How the threads were imposed in the reference trial 20261119-ARCHTRIAL-1865f54: not recorded.

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

Files read (SHA-256): 20261119-ARCHTRIAL-b8f7cca status.json eabd17a7f2431cf792a29598002012e2431db75ef543c37474841805263fb094; 20261119-ARCHTRIAL-b8f7cca trial.json d44a82c80bae0f5d54e3df8b07135993e79df75929b2eabd079b97547cf43c80; 20261119-ARCHTRIAL-1865f54 status.json 1e30d86977ee520aa5ae9a071ab8638549d2d0a3f19a50b53e69954f1943ffbf; 20261119-ARCHTRIAL-1865f54 trial.json 7818108cfeddd2986f4c9197cd339bfbdf4a8cff5dd4611b89a8af26ad20a78b.
