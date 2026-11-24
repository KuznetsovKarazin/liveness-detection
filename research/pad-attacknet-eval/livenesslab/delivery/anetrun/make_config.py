"""Genera delivery/anetrun/pad_anet_run.json con gli SHA-256 attuali dei file fissati e i testi della configurazione dei
sei run C5. Da rilanciare dopo ogni modifica del codice fissato o dei testi, dalla radice del repository:

  .venv/bin/python -I -B delivery/anetrun/make_config.py

Non sovrascrive una configurazione diversa senza --force; stampa lo SHA-256 della configurazione scritta."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import pad_anet_run as NR  # noqa: E402
import pad_arch_comparison as PAC  # noqa: E402
import pad_arch_run as R  # noqa: E402

OUT = ROOT / "delivery/anetrun/pad_anet_run.json"
A = "a choice of the author"
COMPARISON = "delivery/anetcmp/pad_anet_comparison.json"
PACKAGE = "delivery/pad-attacknet-comparison"
TRIAL_ID = "20261122-ANETTRIAL-a951ed0"
BNCFG = "delivery/bnrecal/nuaa_bn_recalibration.json"
BTCFG = "delivery/bntest/nuaa_bn_test.json"


def sha(p: str) -> str:
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def doc(data_sha=sha, code_sha=sha, COMPARISON=COMPARISON, PACKAGE=PACKAGE, TRIAL_ID=TRIAL_ID, BNCFG=BNCFG, BTCFG=BTCFG) -> dict:
    """La configurazione. Senza argomenti: i file del repository; la suite passa i file del suo mondo sintetico."""
    sha = data_sha
    return {
        "schema": NR.CONFIG_SCHEMA,
        "task": ("The six NUAA runs of the C5 comparison pre-registered in the PR A package (PROTOCOL.md §4, §6 (n), §8, §11): 72 new "
                 "trainings of AttackNetV1, AttackNetV2_1 and AttackNetV2_2, in six runs of 12 (anet_v1_p4, anet_v2_1_p4, anet_v2_2_p4, then the "
                 "same at patience 12), one process each, positions of plan/run_matrix.csv (folds 1-4, seeds 42, 43, 44), arm B, fitting 1,400 "
                 "and dev 400 frames per unit, the recipe and trainer of the technical trial (pad_anet_comparison.anet_trainer), CPU only, 10 "
                 "intra-op and 10 inter-op threads; fitting and dev only, no test frame"),
        "authorisation": ("PROTOCOL.md §11 of the PR A package: the 72 trainings start after the pull request with the command and "
                          "configuration of the runs is published and its link is on the C5 card, unless the coordinator states otherwise on "
                          "the card; the evaluation code and configuration are published, with the link on the card, before the NUAA test is "
                          "read. Order of the official commands: this pull request published and its link on the C5 card; then "
                          "check-estimator (PROTOCOL.md §3), not before the link; then the six runs in the order of plan/run_order.csv"),
        "operations": ("results/ is versioned: after check-estimator, and after each run, the work tree is not clean, and check-estimator "
                       "and run, the only commands that check the work tree, refuse until a commit is made (after the link of this pull request "
                       "on the C5 card: check-estimator, commit, then each of the six runs in the order of run_order.csv followed by a commit); "
                       "check-estimator and each run are launched from the repository root, from a zsh with setopt nobgnice, as nohup zsh "
                       "delivery/anetrun/launch.zsh <arguments> > <a log file outside the repository> 2>&1 &, which runs "
                       "caffeinate -dims .venv/bin/python -I -B scripts/pad_anet_run.py <arguments>; SIGUSR1 to the pid in status.json stops the command as INCOMPLETE at the next check; "
                       "SIGUSR1 is sent only for a cause outside the results (for example the machine must be released), never after looking "
                       "at the histories or the dev outcomes of the run, and the cause is stated in the report; "
                       "one heavy process at a time on the machine; check-machine and check-inputs before each command. The launch script "
                       "delivery/anetrun/launch.zsh stops if its nice value is not 0, runs check-machine, check-inputs and the command in this "
                       "order (the command only if both pass) and writes, outside the repository, a conditions file per command, copied "
                       "unchanged by the author after each official command into the folder of the records of that run: operating system, power mode and power source (pmset), the nice value of the "
                       "shell, the number of TF_NUM_ variables set, commit and number of lines of git status --porcelain, load and CPU usage at "
                       "the start and at the end, the outputs of check-machine and check-inputs, the command and its exit code"),
        "comparison": {"path": COMPARISON, "sha256": sha(COMPARISON)},
        "trial_record": {"run_id": TRIAL_ID, "folder": f"{PACKAGE}/trial", "status_sha256": sha(f"{PACKAGE}/trial/status.json"),
                         "trial_json_sha256": sha(f"{PACKAGE}/trial/trial.json"),
                         "package_sums": {"path": f"{PACKAGE}/SHA256SUMS", "sha256": sha(f"{PACKAGE}/SHA256SUMS")}},
        "plan_files": {k: {"path": f"{PACKAGE}/plan/{k}.csv", "sha256": sha(f"{PACKAGE}/plan/{k}.csv")} for k in NR.PLAN_KEYS},
        "code": {f: code_sha(f) for f in NR.CODE_FILES},
        "estimator_check": {
            "bn_recalibration_config": {"path": BNCFG, "sha256": sha(BNCFG)},
            "bn_test_config": {"path": BTCFG, "sha256": sha(BTCFG)},
            "bnrecal_run_id": "20261114-BNRECAL-a10a082", "models": 36, "cap_minutes": 240, "cap_unit": PAC.CAP_UNIT,
            "rule": ("PROTOCOL.md §3: as in C2, the estimator first reproduces the statistics of run 20261114-BNRECAL-a10a082 within the "
                     "tolerance of PRs #27 and #28 before the runs, otherwise the work stops; the check is that of pad_arch_run "
                     "check-estimator (the 36 models of the run, both criteria of nuaa_bn_test.recal_check, cap of the PR #27 configuration, "
                     "240 minutes of wall time); a run needs a PASS of it made with this configuration and this code")},
        "runs": list(NR.RUNS),
        "machine_rule": {"fields": list(R.MACHINE_FIELDS), "versions": list(R.VERSION_FIELDS), "threads": list(R.THREAD_FIELDS),
                         "consequence": ("the rule of C2 (pad_arch_run.machine_rule, thread_environment_rule, threads_rule) applied to the "
                                         f"record of the official C5 trial {TRIAL_ID}: the same machine model, the same versions, the same "
                                         "thread record (requested and effective 10 and 10, set by the runner, environment variables absent); "
                                         "before any folder, in check-machine, check-inputs, check-estimator and run; a difference stops the "
                                         "command, nothing trained")},
        "sequence": ("phase 1, for each of the 12 positions in order: the fitting and dev frames of the unit assembled and their keys checked "
                     "against the card of the accepted run; the training (pad_anet_comparison.anet_trainer, the model built with the registry); "
                     "rule (n) on the history and on the restored weights; the in-memory dev scored; the checkpoint saved; the card; the dev "
                     "scored with the saved checkpoint (nuaa_folds_run.tf_scorer), its logits and the val_loss checked against the card. Phase "
                     "2, after the 12 trainings, for each evaluable model: the checkpoint reloaded and checked against the registry; the ordinary "
                     "dev; the BN statistics re-estimated on its fitting set (copy in memory); the consistency pass; the re-estimated dev; the "
                     "checks of PR #27; the agreement with the phase-1 scores; thresholds, decisions and counts of both BN states. The sequence "
                     "of pad_arch_run (C2) with the AttackNet trainer, the registry and rule (n)"),
        "checkpoint": {"format": ".keras (model.save of Keras 2.15), the format of the accepted runs; read back with tf.keras.models.load_model(compile=False)",
                       "when": R.CHECKPOINT_TEXT},
        "non_evaluable": {
            "rule": ("rule (n) of the pre-registration (PROTOCOL.md §6 (k)/(n), pad_anet_definitions.training_outcome): a non-finite value "
                     "in the history does not stop the run; the quantity and the first non-finite epoch are recorded; the restored epoch is "
                     "that of the smallest finite val_loss; without a finite val_loss, or with non-finite restored weights, the model is not "
                     "evaluable: no checkpoint, no phase 2"),
            "in_memory_dev": ("before the checkpoint is saved, the dev and the fitting are scored with the model in memory in the ordinary state "
                              "(pad_anet_comparison.forward_checked, inference only: no state of the model, no seed and no recorded epoch time "
                              "changes): non-finite scores or logits make the model not evaluable, without checkpoint"),
            "fitting_scores": ("the fitting is scored in both BN states: in the ordinary state by the in-memory pass of phase 1 (non_evaluable."
                               "in_memory_dev), in the re-estimated state by the consistency pass of phase 2; non-finite scores or logits there, or "
                               "on the dev in either BN state, make the model not evaluable"),
            "statistics": ("non-finite re-estimated statistics (the message of nuaa_bn_recalibration) make the model not evaluable, before any "
                           "output of its phase 2 is written"),
            "record": ("a not evaluable model keeps its card in restricted/; a model found not evaluable in phase 2 keeps its checkpoint and "
                       "phase-1 dev scores in restricted/, marked not evaluable in models.csv, and the evaluation does not score it on the test "
                       "(amended PROTOCOL.md §6 (k)/(n)); models.csv, summary.json and report.md give the status 'not evaluable' with the reason; "
                       "it does not change the outcome of the run; no new attempt with the same seed for it")},
        "estimator": {"estimator": ("the estimator of PR #27 (nuaa_bn_recalibration.recalibrate), extended to networks with branches by "
                                    "pad_arch_run.reestimate (the order checked on the graph), on an in-memory copy; only moving_mean and "
                                    "moving_variance assigned; every other tensor checked bit-identical"),
                      "batch": 16, "orders": NR.expected_orders(),
                      "order_rule": ("LivenessNet (check-estimator only): the order of PR #27; each AttackNet: the order of the registry "
                                     "(pad_anet_models.bn_order), the order of the technical trial, checked on the graph by "
                                     "pad_arch_run.reestimate")},
        "dev_evaluation": {"states": ["ordinary", "re-estimated"],
                           "threshold": ("the EER threshold of the project (pad_arch_definitions.condition_threshold) on the dev of the fold "
                                         "with the scores of the model in the same BN state; attack if score > threshold, also at 0.5"),
                           "one_class": "definition (a) (pad_arch_definitions.one_class_run): per model and BN state, on the dev of its fold, at 0.5",
                           "agreement": ("the binding agreement of C2 between the phase-1 scorer and the evaluator on the ordinary dev "
                                         "(pad_arch_run.evaluator_agreement, tolerances of PR #27): a disagreement gives the outcome MISMATCH "
                                         "and no public result")},
        "cap": {"source": "plan/run_caps.csv of the PR A package, column cap_wall_hours of the run, read, not rewritten",
                "unit": PAC.RUN_CAP_UNIT,
                "checks": ("as in C2: before every training, at the end of every epoch and before every inference pass; at the cap the run "
                           "stops as INCOMPLETE, the model in progress in phase 1 is discarded, the complete models are kept and reported")},
        "after_stop": (f"{A}, recorded in the amended PROTOCOL.md §6 (k)/(n) of this delivery, which replaces the pre-registered sentence "
                       "'resumed only for the missing trainings': a run stopped at its cap, by SIGUSR1, by a FAIL, a MISMATCH or an "
                       "interruption is not resumed within its folder; it keeps its folder and outcome and, after its outcome has been reported "
                       "on the C5 card, is repeated as a new attempt in a new folder from position 1, with the same seeds, code and "
                       "configuration, before the evaluation, as in C2; the repetition retrains every position, those not evaluable included, "
                       "and their outcome in the last attempt is the one evaluated; a run is never repeated because of a not evaluable model; "
                       "the evaluation reads, for each of the six runs, only the last attempt, which must be PASS, with phase 1 complete for the "
                       "12 positions (not evaluable models included)"),
        "attempts": (f"{A}: as in C2: --attempt N required, the last attempt of the same run in the output root + 1; suffix -r<k>; official "
                     f"output root {NR.DEFAULT_OUT_ROOT}; another root outside the repository only with --unofficial-out-root, recorded as not "
                     "official; --date only with --unofficial-out-root (an official run id carries the date of the launch); an official run needs an official "
                     "check-estimator"),
        "outputs": {"run": [NR.STATUS, NR.LOG], "public": list(NR.PUBLIC_OUTPUTS),
                    "restricted": [f"<model>{x}" for x in R.PER_MODEL] + ["SHA256SUMS"]},
        "choices_of_the_author": {
            "sequence": f"{A}: the two phases and their order, as in C2 (sequence)",
            "in_memory_dev": (f"{A}: the dev and the fitting scored with the model in memory in the ordinary state before the checkpoint is "
                              "saved, for rule (n) (non_evaluable.in_memory_dev)"),
            "checkpoint": f"{A}: the checkpoint saved by this runner right after the trainer returns, as in C2 (checkpoint.when)",
            "card_check": (f"{A}: the card check of C2 (pad_arch_run.card_problems) with the non-finite history values read as +inf, so that "
                           "the restored epoch expected is that of the smallest finite val_loss"),
            "estimator_check_process": f"{A}: check-estimator as a command of its own before the runs, the check of C2 (estimator_check.rule)",
            "after_stop": f"{A}: no resume within the folder, a new attempt from position 1, the last attempt evaluated (after_stop)",
            "launch_conditions": (f"{A}: the launch script delivery/anetrun/launch.zsh, which writes a conditions file per command with the "
                                  "launch conditions (operations)"),
            "attempts": f"{A}: the attempt rule and the unofficial output root (attempts)",
            "machine_scope": f"{A}: the machine model and versions compared with the C5 trial, not the physical unit (machine_rule)",
            "threads_record": f"{A}: the whole thread record of the C5 trial compared, environment variables included (machine_rule)",
            "evaluator_agreement": f"{A}: the binding agreement between the phase-1 scorer and the evaluator, outcome MISMATCH (dev_evaluation.agreement)",
            "dev_frames": (f"{A}: the dev frames checked per fold across the models of the run and the two BN states (one network per run; "
                           "in C2 the check was between the two networks of a unit)"),
            "outputs": (f"{A}: the names and columns of the public and restricted files, the run id "
                        f"YYYYMMDD-ANETRUN-<v1|v2_1|v2_2>-p<4|12>-<commit>[-r<k>] and the folder {NR.DEFAULT_OUT_ROOT}"),
            "deviation_record": (f"{A}: the two amendments of PROTOCOL.md of PR A (§7, the order of publication of the trial; §6 (k)/(n), "
                                 "the repetition of a stopped run) recorded in an amended copy in this delivery, the original kept unchanged")},
        "test": NR.TEST_READ,
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
