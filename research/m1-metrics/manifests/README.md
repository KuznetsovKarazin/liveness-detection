# NUAA source-dev manifest (summary)

The per-image development manifest (`nuaa_dev_manifest.csv`) is a per-subject record and is kept out of this public repository, as required by `research/README.md`. It is delivered in the restricted project folder `02_Experiments/M1/20261003-M1-seed42-acf6fda/`.

- Rows: 574 images (180 bona fide, 394 attacks) from 2 subjects; one SHA-256 per image.
- SHA-256 of the manifest file: `d4637ea67f30da2bb0e6ddcdec688c97aa3833c41709417ac3e93ee00b6fa938` (also in `nuaa_dev_manifest.sha256`, in `run-report.md` and, once the threshold is selected, in `config/threshold.json`).
- Selection rule: see `metrics.md`, section "Threshold selection protocol".
- Evaluation manifest (300 test images, unchanged since C1): SHA-256 `86095da1531b1f24b1233c9668ac1ab37714978e444b313456732c028ccfb40c` (`config/nuaa_config.json`).
