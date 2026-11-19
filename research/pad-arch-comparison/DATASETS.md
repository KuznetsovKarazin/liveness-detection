# External datasets for the architecture comparison: CelebA-Spoof and Replay-Attack

**Status, 18 November 2026.** Neither dataset is used in this delivery. Sources were read at the primary source on 18
November 2026; what is marked "not verified" was not read at a primary source and is not stated as fact. The state of
the authorisations is that of the coordinator's D1 card as of 17 November 2026 (the procedures are the coordinator's).

## CelebA-Spoof

| item | content |
|---|---|
| reference | Zhang Y., Yin Z., Li Y., Yin G., Yan J., Shao J., Liu Z., "CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset with Rich Annotations", *Computer Vision – ECCV 2020*, LNCS, Springer, Cham, pp. 70–85, DOI 10.1007/978-3-030-58610-2_5; arXiv:2007.12342. LNCS volume number: not verified |
| official source | repository https://github.com/ZhangYuanhan-AI/CelebA-Spoof (the paper gives the earlier name of the same repository, github.com/Davidzhangyuanhan/CelebA-Spoof); download folder linked from its README |
| how it is obtained | direct download, no form or agreement to sign: "Google Drive and Baidu Drive are available now!" (README); the README's "Registration" concerns the 2020 challenge, not the download |
| conditions of use | the README's "Dataset Agreement" list (its "License and Citation" section concerns the software), silent on who must accept it: "available for non-commercial research purposes only"; "You agree not to further copy, publish or distribute any portion"; internal copies "for internal use at a single site within the same organization" |
| official numbers | "625,537 pictures of 10,177 subjects" (abstract); live : spoof "1 : 3" (§3.3; Fig. 4(a): 24 % live, 76 % spoof); absolute live and spoof counts not reported in the paper; live images "directly inherited from CelebA dataset" (§3.1); 43 attributes per image; "4 macro-types and 11 micro-types of spoof type" (Fig. 3); 8 sessions (2 environments × 4 illuminations), more than 10 sensors (§3.2, Table 1) |
| identity split: source | "a ratio of 8 : 1 : 1" for training, validation and test; "all three sets are guaranteed to have no overlap on subjects"; live images split as "defined in the CelebA dataset" (§3.3). Intra-dataset benchmark: training and test sets, test labels in `metas/intra_test/test_label.json` (benchmark code README) |
| identity split: derived | none yet; the planned split is the official one, by subject, as stated in §3.3 |
| identity split: not verified | how the identity is encoded on disk; whether the archive holds a distinct validation set; label files other than the test one; the folder structure (described by neither the README, the repository tree nor the paper); that the CelebA split itself is by identity |
| local state | one shard of the test part of an unofficial mirror, 6,717 rows (6,678 with an image), with no identity column; the existing manifests have no subject column. It cannot support an identity split; the official archive is not held |
| access, D1 card (17 November 2026) | conditions of use confirmed by the first author of the dataset on 8 October 2026 according to the coordinator's D1 card: remote access for students, a restricted institutional Drive and local copies for non-commercial research within the group; no additional registration; no redistribution to third parties. Still to be agreed: the university channel. The permission does not verify the provenance or equivalence of the mirror copy used in earlier experiments. No new agreement or access confirmed as of 17 November 2026 |
| missing materially | the official archive; the check of how the identity is encoded |
| other points not verified | total size (the download folder lists 50 parts `.001`–`.050`, each displayed as 1,000 MB: about 50 GB if that is the real size); whether bounding-box files are provided (in AENet the faces are cropped with RetinaFace, §5) |

## Replay-Attack (Idiap)

| item | content |
|---|---|
| reference | Chingovska I., Anjos A., Marcel S., "On the Effectiveness of Local Binary Patterns in Face Anti-spoofing", *Proceedings of the 11th International Conference of the Biometrics Special Interest Group (BIOSIG)*, September 2012 (Idiap publication 2447). Pages and DOI of the article: not verified |
| official source | https://www.idiap.ch/en/scientific-research/data/replayattack ("Get data" leads to Zenodo); https://zenodo.org/records/4593128, dataset DOI 10.34777/cwcg-7r82. The Idiap page header describes "A speech dataset with 44 speakers", apparently in error: cite the DOI as well |
| how it is obtained | Zenodo: "Restricted", "files are restricted"; access "based on an End-User License Agreement"; the request names an "authorized signatory (MUST hold a permanent position)" and an institutional address of the signatory's organisation; it follows that a student cannot sign (our reading). Full EULA text: not verified (not visible without access) |
| conditions of use | "strictly restricted to non-commercial research" (Zenodo); "If you use this database, please cite the following publication" (BIOSIG 2012) |
| official numbers | "1,300 video clips of photo and video attack attempts of 50 clients" (Idiap page): 1,200 in the protocol and 100 for enrolment. Lighting: controlled and adverse. Attacks (paper, §III): print (A4), mobile (iPhone 3GS), highdef (iPad), each hand-based and fixed-support; "20 attack videos were registered for each client" (Idiap page). Table I: real access 60 / 60 / 80 and attacks 300 / 300 / 400 (train / devel / test), 1,200 in all. Video: "320 by 240 pixels (QVGA), at 25 frames-per-second"; real-access videos 15 s each (375 frames), attack videos about 10 s (paper, §III); "for at least 9 seconds", `.mov` (Idiap page). Face locations are provided, annotated by a cascade of classifiers |
| identity split: source | "Identities for each subset were chosen randomly, but do not overlap" (paper, §III); clients of one set "do not appear in any other set" (Idiap page); devel "15 clients", test "20 clients"; the test is to be used solely to report results (§III), with the threshold at the EER of the devel set (§V-A); 18 protocols, including grandtest |
| identity split: derived | 15 training clients (50 − 15 − 20; consistent with 60 real-access videos at 4 per client) |
| identity split: not verified | the client lists, until the archive is held |
| local state | nothing: no copy, no EULA form prepared |
| access, D1 card (17 November 2026) | no new agreement or access confirmed as of 17 November 2026. Per the D1 card: each Idiap dataset has its own EULA and Zenodo procedure; the licence is institutional and covers employees and students under the responsibility of the institution and the signatory, with no individual licences; the minimum requirement is a permanent position, distinct from the internal confirmation of the power to sign; the EULA cannot be modified; the coordinator's Zenodo requests cover MSSpoof and 3DMAD now, Replay-Attack later; an earlier access held at another university does not authorise this group |
| missing materially | the Idiap EULA for the university, with a signatory in a permanent position and the internal confirmation of the power to sign; the archive |
| other points not verified | the archive size ("3.1 TB" on Zenodo, to be clarified); pages and DOI of the article. To be fixed in the pre-registration: frame sampling and per-video aggregation (the paper treats every frame as an independent sample and normalises faces to 64×64) |

## ResNet-18 in the PAD works checked

- AENet (CelebA-Spoof paper, note 10): "For fair comparison, ResNet-18 is used in all experiments" (the note goes on to a heavier Xception backbone), with the backbone pre-trained on ImageNet (§5); its numbers are not comparable with a training from scratch.
- SSDG (Jia et al., CVPR 2020, pp. 8481–8490, DOI 10.1109/CVPR42600.2020.00851) uses a ResNet-18 variant (SSDG-R) and Replay-Attack; SSAN (Wang et al., CVPR 2022, arXiv:2203.05340; pages and DOI not verified) has a ResNet-18 variant (SSAN-R).
- He et al. do not give the parameter count of ResNet-18; the torchvision documentation gives 11,689,512 with the 1000-way head.
