"""Genera delivery/anetcmp/pad_anet_comparison.json con gli SHA-256 attuali dei file fissati e i testi della
configurazione (TEST_READ dello script, le note delle definizioni). Da rilanciare dopo ogni modifica del codice fissato
o dei testi, dalla radice del repository:

  .venv/bin/python -I -B delivery/anetcmp/make_config.py

Non sovrascrive una configurazione diversa senza --force; stampa lo SHA-256 della configurazione scritta."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import nuaa_folds_run as FR  # noqa: E402
import nuaa_session_ablation_run as SR  # noqa: E402
import pad_anet_comparison as C  # noqa: E402
import pad_anet_definitions as DEF  # noqa: E402
import pad_anet_models as AN  # noqa: E402
import pad_arch_definitions as D  # noqa: E402

OUT = ROOT / "delivery/anetcmp/pad_anet_comparison.json"
A = "a choice of the author"
PREREG = "delivery/archcmp/pad_arch_comparison.json"
BNCFG = "delivery/bnrecal/nuaa_bn_recalibration.json"
C2_TRIAL = "delivery/pad-arch-comparison/trial"
EVAL = "delivery/pad-arch-eval-results/runs/20261121-ARCHEVAL-332fbdf"


def sha(p: str) -> str:
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def doc(pr: dict = None, ev_status: dict = None, data_sha=sha, code_sha=sha, PREREG=PREREG, BNCFG=BNCFG, C2_TRIAL=C2_TRIAL, EVAL=EVAL) -> dict:
    """La configurazione. Senza argomenti: i file del repository; la suite passa i file del suo mondo sintetico (data_sha
    per i file di dati e configurazioni, code_sha per il codice, percorsi relativi alla radice degli ingressi)."""
    pr = pr if pr is not None else json.loads((ROOT / PREREG).read_text(encoding="utf-8"))
    ev_status = ev_status if ev_status is not None else json.loads((ROOT / EVAL / "status.json").read_text(encoding="utf-8"))
    sha = data_sha
    return {
        "schema": C.CONFIG_SCHEMA,
        "task": ("First step of the C5 card (21 November 2026): a pull request with the C5 protocol and the run matrix; a short trial for each "
                 "AttackNet on fitting and dev only (arm B, fold 1, seed 42), with the check that every BN layer is re-estimated, the device, the "
                 "time per epoch and the planned hours; configuration and evaluation code fixed before any new test result is examined. The "
                 "comparison: AttackNetV1, AttackNetV2_1 and AttackNetV2_2 trained from scratch with the project implementations, 4 folds x 3 "
                 "seeds (42, 43, 44) x 2 patiences (4 and 12) = 24 trainings each, 72 in all, 36 per patience; the 48 C2 checkpoints of LivenessNet "
                 "and ResNet-18 reused; every checkpoint with ordinary BN and BN re-estimated on the fitting set only, at threshold 0.5 and at the "
                 "EER threshold estimated on the dev of the fold in the same BN state."),
        "authorisation": ("The C5 card asks for this pull request and for the three short trials as its first step; the trial is launched with "
                          "--i-am-authorized from a clean committed tree after an independent review of code and texts. The 72 trainings start "
                          "after the pull request with the command and configuration of the runs is published and its link is on the C5 card, "
                          "unless the coordinator states otherwise on the card."),
        "declarations": {
            "no_test": ("no test frame is assembled or evaluated and no test score or metric of the new trainings is read in this step; the "
                        "only trainings are the 3-epoch trials on the fitting and dev frames of arm B, fold 1, seed 42"),
            "already_known": ("the NUAA test and the C2 results are already known (C2 evaluation 20261121-ARCHEVAL-332fbdf, pull request #32); "
                              "the LivenessNet and ResNet-18 cells of this comparison are those cells, reused"),
            "not_independent": "on NUAA this comparison is not an independent validation",
            "post_hoc": ("the C5 design was written after the C2 contrast was seen (+9.98 points of ACER): a post hoc extension to three "
                         "more networks"),
            "technical_trial": ("3 epochs per network, patience 4, seed 42, NUAA arm B fold 1, fitting 1,400 and dev 400 frames; LivenessNet "
                                "first as a control of the machine, not counted among the three trials"),
            "cpu_only": "CPU only, GPU hidden and the remapping optimizer off (nuaa_folds_run.hide_gpu), as in C2",
            "checkpoints": ("the three trial checkpoints are saved outside the repository, an unofficial output: only their names and "
                            "SHA-256 enter the record"),
            "threads": "10 intra-op and 10 inter-op TensorFlow threads, set and read back by the runner (pad_arch_comparison.set_threads), as in C2"},
        "preregistration": {"path": PREREG, "sha256": sha(PREREG)},
        "c2_trial": {"run_id": "20261119-ARCHTRIAL-b8f7cca", "folder": C2_TRIAL, "status_sha256": sha(f"{C2_TRIAL}/status.json"),
                     "trial_json_sha256": sha(f"{C2_TRIAL}/trial.json")},
        "bn_recalibration_config": {"path": BNCFG, "sha256": sha(BNCFG)},
        "c2_results": {"evaluation_run_id": "20261121-ARCHEVAL-332fbdf", "status": {"path": f"{EVAL}/status.json", "sha256": sha(f"{EVAL}/status.json")},
                       "results": {"path": f"{EVAL}/public/results.json", "sha256": sha(f"{EVAL}/public/results.json")},
                       "runs": {k: ev_status["runs"][k]["run_id"] for k in C.C2_RUNS},
                       "restricted_sha256sums_sha256": ev_status["restricted_sha256sums_sha256"]},
        "code": {f: code_sha(f) for f in C.CODE_FILES},
        "architectures": {"path": pr["architectures"]["LivenessNet"]["path"], "sha256": pr["architectures"]["LivenessNet"]["sha256"],
                          "input_shape": list(FR.FRAME_SHAPE),
                          "networks": {n: {"trainable_parameters": AN.expected_counts(n, FR.FRAME_SHAPE)["trainable"],
                                           "non_trainable_parameters": AN.expected_counts(n, FR.FRAME_SHAPE)["non_trainable"],
                                           "batchnorm_layers": AN.expected_counts(n, FR.FRAME_SHAPE)["batchnorm_layers"], "batchnorm_names": AN.bn_order(n),
                                           "constructor": AN.CONSTRUCTOR} for n in AN.NETWORKS}},
        "training": {
            "recipe": pr["training"]["recipe"],
            "threads": pr["training"]["threads"],
            "trial_overrides": {"max_epochs": C.TRIAL_EPOCHS},
            "trainer": ("anet_trainer, a declared copy of pad_arch_comparison.trial_trainer with the substitutions ANET_TRAINER_SUBSTITUTIONS "
                        "(restored epoch = the epoch of the smallest finite val_loss; non-finite history values written as null with the list "
                        "of quantity and first epoch; the trainer saves no checkpoint); on finite histories it gives the card and weights of "
                        "the original"),
            "non_finite_rule": ("rule (n), fixed before the runs, a choice of the author and a departure from C2, where a non-finite value in "
                                "a new history ended the run as FAIL (here one network must not stop the trainings of the others): a "
                                "non-finite value in the history does not stop the run; the quantity and "
                                "the first non-finite epoch are recorded; if a finite val_loss exists, the restored epoch is that of the smallest "
                                "finite val_loss and the checkpoint is saved and evaluated, provided every restored tensor is finite; otherwise the "
                                "model is not evaluable (no checkpoint, no phase 2, no test); also not evaluable: non-finite scores or logits "
                                "on the fitting or the dev in either BN state, or probabilities non-finite, outside [0, 1] or of the wrong "
                                "shape in the timed inference pass (one message of the frozen C2 function, all counted as not evaluable), or "
                                "non-finite re-estimated statistics; no new attempt with the same seed for a non-finite case")},
        "trial": {"dataset": "NUAA", "arm": "arm_b", "fold": 1, "seed": 42, "roles": list(C.TRIAL_ROLES), "networks": list(AN.NETWORKS),
                  "control": C.CONTROL,
                  "control_rule": ("LivenessNet is trained first, with the same trainer, as a control of the machine and threads (not one of the "
                                   "three trials): the five quantities of epochs 1-3 must be identical, difference 0, to the accepted card "
                                   "arm_b_fold1_seed42 and to the C2 trial 20261119-ARCHTRIAL-b8f7cca, and the machine and versions equal, field "
                                   "by field, to those of the C2 trial (the rule of pad_arch_run.machine_rule, recorded); otherwise the outcome "
                                   "is CONTROL_FAILED"),
                  "after_outcome": ("pre-registered, a choice of the author: whatever the outcome (PASS, FAIL, CONTROL_FAILED, NOT_EVALUABLE, "
                                    "INCOMPLETE), it is reported on the C5 card before anything else; no new attempt changes the recipe, the "
                                    "constructor or the seed; an attempt in which the LivenessNet control was not reproduced or an AttackNet "
                                    "was not evaluable, as recorded in its status.json, trial.json or run.log, counts as CONTROL_FAILED or "
                                    "NOT_EVALUABLE whatever its final status, FAIL and INCOMPLETE included; after CONTROL_FAILED or "
                                    "NOT_EVALUABLE no new attempt and no run before the coordinator's reply; an INCOMPLETE is reported with "
                                    "its cause (cap or SIGUSR1) and step; otherwise a FAIL or INCOMPLETE is corrected in the code or the "
                                    "conditions, reported, and the trial repeated as a new attempt from a clean committed tree, the earlier "
                                    "record kept"),
                  "attempts": ("--attempt N must be 1 + the attempts found in the output and checkpoint roots (any date and commit); from the "
                               "second the run id ends with -r<N>; attempt and earlier attempts with their status recorded in status.json "
                               "and trial.json; the date of the run id is today's"),
                  "conditions_file": ("trial/conditions.txt (uptime, CPU usage, power mode, command, number of TF_NUM_ variables set, commit "
                                      "and number of lines of git status --short), written by the author around the trial, not by the runner"),
                  "epochs": C.TRIAL_EPOCHS, "patience": C.TRIAL_PATIENCE, "inference_batch": FR.SCORE_BATCH, "estimator_batch": 16,
                  "wall_time_cap_minutes": 60, "cap_unit": C.CAP_UNIT, "estimate_minutes": 20, "upper_estimate_minutes": 40,
                  "checkpoints": ("saved with pad_arch_run.save_checkpoint under --checkpoint-root/<run id>, a folder outside the repository, "
                                  "reloaded with nuaa_bn_recalibration.load_keras; only names and SHA-256 are recorded"),
                  "bn_check": ("five checks, one failed = FAIL: (1) the BatchNormalization layers read from the graph are the five of the "
                               "registry, in the order of the re-estimation, with an admitted order on the graph and the moving statistics "
                               "equal to all the non-trainable variables (544 or 448); (2) values per channel = frames x H x W of the registry "
                               "(91,750,400 for bn1_1 and bn1_final, 22,937,600 for bn2_1 and bn2_final, 1,400 for bn_dense) and the channels of "
                               "the registry; (3) every moving_mean and moving_variance changed and equal to the installed float32 values, every "
                               "other tensor identical, the original unchanged; (4) consistency and softmax of the logits within the tolerances "
                               "of PR #27; (5) finite statistics and dev scores different between the two BN states")},
        "plan": {"networks": list(AN.NETWORKS), "patiences": list(D.PATIENCES), "folds": list(DEF.FOLDS), "seeds": list(D.PLAN_SEEDS), "arm": "arm_b",
                 "max_epochs": FR.FROZEN_TRAINING["max_epochs"], "runs": C.run_names(),
                 "order": (f"{A}: six runs of 12 trainings, one process each, patience 4 then 12, in each AttackNetV1, AttackNetV2_1, "
                           "AttackNetV2_2; in each run folds 1-4 and seeds 42, 43, 44; the trainer clears the session and resets the seeds "
                           "before every model, a failure costs 12 trainings and not 36"),
                 "caps": {"rule": (f"{A}: cap of a run = (largest estimate + (anomaly factor - 1) x the largest cost of one training) x "
                                   "multiplier, rounded up to a multiple of the rounding step; the anomaly factor covers one training at about three "
                                   "times its cost, as the first model of a C2 run (49.0 s per epoch against 14.4-16.3, 3.0-3.4 times)"),
                          "unit": C.RUN_CAP_UNIT, "anomaly_factor": 3, "wall_multiplier": 1.5, "round_up_hours": 0.5},
                 "machine_rule": ("every training and evaluation of the plan runs on the same machine with the same threads and versions as "
                                  "the official C5 trial, as in C2"),
                 "reference": {"networks": list(DEF.REUSED), "cells": C.REFERENCE_CELLS,
                               "rule": ("the 48 C2 checkpoints of LivenessNet and ResNet-18 (runs 20261120-ARCHRUN-p4-7faa49f and "
                                        "20261120-ARCHRUN-p12-ea7fdca) are not trained and not re-evaluated: their per-frame test scores "
                                        "already computed by the C2 evaluation are read, each file checked against the restricted SHA256SUMS "
                                        "whose SHA-256 is in the public status.json of that evaluation")},
                 "evaluation": ("the NUAA test is read once, by one C5 evaluation command published before it is run; the test frames of each "
                                "fold are assembled once and scored by the evaluable new models of that fold in both BN states")},
        "definitions": {
            "bootstrap": dict(SR.BOOT), "conditions": [list(c) for c in D.CONDITIONS], "decision": FR.DECISION_RULE,
            "eer_threshold": FR.THRESHOLD_CRITERION, "fixed_threshold": FR.FIXED_THRESHOLD,
            "frame_ids": {"field": D.FRAME_ID_FIELD, "rule": ("per unit the networks present are scored on the same frame identifiers in the "
                                                             "same order, also between the two BN states and between new and reused cells")},
            "primary": {"patience": 4, "threshold": "fixed_0.5", "ordinary": list(D.PRIMARY_ORDINARY), "re_estimated": list(D.PRIMARY_RE_ESTIMATED),
                        "variation": "per network and unit, ACER with re-estimated BN minus ACER with ordinary BN, as an exact fraction",
                        "contrast": ("per AttackNet and unit, its variation minus the variation of LivenessNet, as an exact fraction, on the same "
                                     "12 fold-seed pairs and the same subjects"),
                        "per_seed": "the same values on the units of each seed, reported apart, outside the reading",
                        "intervals": DEF.INTERVALS_NOTE, "reading_names": {k: list(DEF.reading_names(k)) for k in AN.NETWORKS}},
            "presentation": DEF.PP_NOTE,
            "non_evaluable": ("rule (n): a model is not evaluable without a finite val_loss, with non-finite restored weights, non-finite "
                              "scores or logits on the fitting or the dev in either BN state, or probabilities non-finite, outside [0, 1] "
                              "or of the wrong shape in the timed inference pass (one message of the frozen C2 function, all counted as not "
                              "evaluable), or non-finite re-estimated statistics"),
            "guardian": ("before the primary analysis: the effective threshold of every network 0.5 in both BN states, otherwise stop; "
                         "LivenessNet and ResNet-18 with the 12 fold-seed combinations in both BN states, otherwise stop; an AttackNet with "
                         "fewer than 12 pairs evaluable in both states: its variation, contrast and comparison with ResNet-18 not computed (k "
                         "of 12 pairs), the other networks go on"),
            "descriptive_vs_resnet18": ("for each of LivenessNet and the three AttackNets, per BN state the difference metric(network) - "
                                        "metric(ResNet-18), and the variation of the network minus that of ResNet-18, with individual subject "
                                        "intervals; no reading"),
            "rankings": ("per condition the networks present in every unit ordered by mean ACER as exact fractions, ties with the same "
                         "lowest rank, the others listed as not ranked; per unit the counts of the ranks; no interval"),
            "descriptive_conditions": ("the variations, contrasts and comparisons with ResNet-18 also at patience 12 with threshold 0.5 and at "
                                       "the EER threshold at both patiences, with their intervals and no reading; the rankings for each of the 8 "
                                       "conditions (pad_anet_definitions.descriptive_analysis); all three pairs of conditions are computed, a "
                                       "missing condition stops the evaluation"),
            "denominators": ("units, subjects, test frames summed over the units with attack and bona fide frames per fold and in total; the "
                             "valid resamples reported with each interval"),
            "one_class": ("per model and BN state on the dev of its fold at threshold 0.5, the one-class rule of C2; on the test, as a "
                          "descriptive count, the models that decide every frame one class, at threshold 0.5 and at the EER threshold of "
                          "its fold, per BN state"),
            "reuse": ("LivenessNet = the 12 C2 models at patience 4 (and 12 at patience 12); ResNet-18 the same; their test scores are read from "
                      "the C2 evaluation files, never recomputed")},
        "outputs": {"trial": [FR.STATUS, FR.LOG, *C.TRIAL_OUTPUTS], "plan": list(C.PLAN_OUTPUTS)},
        "environment": pr["environment"],
        "choices_of_the_author": {
            "constructors": f"{A}: create_model with its defaults (dropout 0.5, no L2), the rule C2 applied to LivenessNet, not the values of model_configs.py",
            "livenessnet_control": f"{A}: LivenessNet trained first in the trial as a control of the machine, not counted among the three trials",
            "six_runs": f"{A}: six runs of 12 trainings instead of two of 36",
            "order": f"{A}: patience 4 then 12; AttackNetV1, AttackNetV2_1, AttackNetV2_2; folds 1-4, seeds 42-44 in each run",
            "non_evaluable_rule": f"{A}: rule (n) for non-finite trainings and not evaluable models",
            "guardian": f"{A}: the per-network guardian, extending the guardian of C2 to the three AttackNets",
            "reading": f"{A}: the reading of each contrast from its interval and the conclusions written before the test, one per AttackNet",
            "rankings": f"{A}: the descriptive rankings of the five networks",
            "denominators": f"{A}: the denominators reported with the table",
            "test_one_class": f"{A}: the descriptive count of the models that decide every test frame one class",
            "percentage_points": f"{A}: percentage points with two decimals in tables, figure and texts; exact fractions and full-precision proportions in the computations",
            "descriptive_vs_resnet18": f"{A}: the descriptive comparison of the four networks with ResNet-18 in the form of definition (m)",
            "caps": f"{A}: the cap rule of the runs with the anomaly factor",
            "trial_outcomes": f"{A}: the outcomes CONTROL_FAILED and NOT_EVALUABLE of the trial, with their outputs written, and what follows each outcome",
            "reference_cells": f"{A}: the public reference_cells.csv holds the name of each reused test score file and the anchor of the restricted SHA256SUMS, not the per-file SHA-256",
            "installed_copy": f"{A}: the copy with the installed statistics must agree with the re-estimated copy within the tolerance of PR #27 (1e-5, no discordant decision)",
            "descriptive_conditions": f"{A}: the descriptive conditions of definition (o)",
            "departure_from_c2": f"{A}: rule (n) instead of the C2 rule, which ended the run as FAIL on a non-finite history",
            "attempts": f"{A}: the attempt rule of the trial (--attempt N, run id -r<N>)"},
        "test": C.TEST_READ,
    }


def main() -> int:
    text = json.dumps(doc(), indent=1, ensure_ascii=False) + "\n"
    if OUT.exists() and OUT.read_text(encoding="utf-8") != text and "--force" not in sys.argv:
        print(f"{OUT.name} exists and differs: pass --force to rewrite it")
        return 1
    OUT.write_text(text, encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)} {hashlib.sha256(text.encode('utf-8')).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
