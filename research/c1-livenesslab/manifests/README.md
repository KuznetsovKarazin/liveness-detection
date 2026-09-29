# NUAA evaluation manifest (summary)

The per-image manifest (`nuaa_manifest.csv`: file, label, subject, session, SHA-256, bytes) is a per-subject record and is kept out of this public repository, as required by `research/README.md`. It is delivered in the restricted project folder `02_Experiments/C1/<run-id>/`.

- Rows: 300 images (150 bona fide, 150 print attacks); columns `file,label,subject,session,sha256,bytes` (`subject` and `session` are the first and third fields of the original NUAA file name); LF line endings; one SHA-256 per image, no duplicates.
- SHA-256 of the manifest file: `86095da1531b1f24b1233c9668ac1ab37714978e444b313456732c028ccfb40c` (also in `nuaa_manifest.sha256`, in `config/nuaa_config.json` and in `run-report.md`).
- Selection rule: from the official NUAA test lists (3,362 bona fide and 5,761 attack images), sorted as in the official files, 150 bona fide and 150 attack images taken at a constant step over each list (`scripts/prepare_datasets.py`, `--per-class 150`). The bona fide images come from 9 subjects and the attack images from 15, which is the composition of the official test set.
- Source: NUAA Photograph Imposter Database (Tan, Li, Liu, Jiang, ECCV 2010), obtained from the public Hugging Face mirror `akahana/anti-spoofing-nuaaaa`; academic use. Attacks are printed A4 photographs re-captured by webcam.
- Verification: with the dataset at hand, `scripts/make_c1_manifest.py` regenerates the manifest; its hash must match the value above.
