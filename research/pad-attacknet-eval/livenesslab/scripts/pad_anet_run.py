"""
I sei run NUAA del confronto C5 (scheda C5 del coordinatore, 21/11/2026; PROTOCOL.md della PR A, pubblicato nella PR
#34): 72 addestramenti nuovi delle tre AttackNet (AttackNetV1, AttackNetV2_1, AttackNetV2_2), in sei run da 12 (uno per
rete e patience: anet_v1_p4, anet_v2_1_p4, anet_v2_2_p4, poi gli stessi a patience 12), un processo ciascuno, nell'ordine
di plan/run_order.csv e con le posizioni di plan/run_matrix.csv (fold 1-4, seed 42, 43, 44), braccio B, fitting 1.400 e dev
400 frame per unità, la ricetta e il trainer della prova tecnica (pad_anet_comparison.anet_trainer, la copia dichiarata
del trainer di C2 con la regola (n)), solo CPU, 10 + 10 thread. Nessun frame di test è assemblato o valutato.

Sottocomandi:
- check-inputs: configurazione, file fissati (configurazione della PR A con quello che fissa, codice, record della prova
  tecnica ufficiale e SHA256SUMS del pacchetto della PR A, i tre file del piano uguali a quelli rigenerati con
  pad_anet_comparison.plan_texts), regola della macchina e dei thread contro la prova C5, ingressi della preregistrazione
  C2 (PAC.load_inputs) e le 24 schede di riferimento del braccio B (chiavi del fitting e del dev); con --estimator-check
  anche il record di check-estimator. Legge soltanto; non importa TensorFlow, non assembla frame, non scrive nulla.
- check-machine: la sola regola della macchina, delle versioni e delle variabili dei thread contro la prova C5 (nessun
  dato, TensorFlow non importato, nulla scritto).
- check-estimator: i 36 modelli del run 20261114-BNRECAL-a10a082 ristimati con lo stimatore della PR #27 esteso
  (pad_arch_run.reestimate) e confrontati con le statistiche e i punteggi del dev del run BNRECAL con le tolleranze
  delle PR #27 e #28 (PROTOCOL.md §3), come in C2 (pad_arch_run.cmd_check_estimator, copia con le sole sostituzioni di
  configurazione, schema e run id). Prima dei run.
- run --run <nome>: un run del piano. Fase 1, per ogni posizione nell'ordine: frame dell'unità, addestramento, regola (n)
  (pad_anet_definitions.training_outcome: un valore non finito della storia non ferma il run; senza val_loss finita o
  con pesi ripristinati non finiti il modello è non valutabile), punteggi del dev in memoria (regola (n)), checkpoint
  salvato e ricaricato, scheda, punteggi del dev con lo scorer dei run accettati, controllo del val_loss. Fase 2, per
  ogni modello valutabile: checkpoint ricaricato e controllato con il registro, stato ordinario sul dev, ristima sul
  fitting, passaggio di coerenza, stato ristimato sul dev, i controlli della PR #27, accordo fra scorer e valutatore. Un
  modello non valutabile è registrato con il motivo e non ferma il run. Tetto del run da plan/run_caps.csv; SIGUSR1 ->
  INCOMPLETE. Esiti: PASS 0, FAIL 1, INCOMPLETE 2, MISMATCH 4 (come C2).

Uscite (results/anetrun/<run-id>): status.json, run.log; public/ models.csv, one_class.csv, summary.json, report.md,
SHA256SUMS (aggregati: nessun identificativo di soggetto o di frame, nessun percorso); restricted/ <rete>_fold<k>_seed<s>.keras,
_card.json, _dev_scorer.csv, _dev_ordinary.csv, _dev_re-estimated.csv, _bn_statistics.json, SHA256SUMS (contengono gli ID
dei frame). I modelli non valutabili hanno solo la scheda (e, se il motivo è emerso dopo il salvataggio, il checkpoint e
i punteggi della fase 1).

Agganci dei test (variabili di modulo a None): solo punti d'iniezione; ogni comando ufficiale (configurazione consegnata
o radice d'uscita ufficiale) controlla all'avvio che tutti gli agganci di questo modulo e dei moduli del progetto che usa
siano None (check_hooks) e lo registra in status.json.

  .venv/bin/python -I -B scripts/pad_anet_run.py check-machine
  .venv/bin/python -I -B scripts/pad_anet_run.py check-inputs [--estimator-check results/anetrun/<id>]
  nohup caffeinate -dims .venv/bin/python -I -B scripts/pad_anet_run.py check-estimator --attempt 1 --i-am-authorized > <log fuori dal repository> 2>&1 &
  nohup caffeinate -dims .venv/bin/python -I -B scripts/pad_anet_run.py run --run anet_v1_p4 --estimator-check results/anetrun/<id> --attempt 1 --i-am-authorized > <log fuori dal repository> 2>&1 &

Dipendenze: NumPy; TensorFlow 2.15 (solo check-estimator e run, solo CPU). Formati: configurazione
pad-anet-run-config/1, run pad-anet-run/1, controllo dello stimatore pad-anet-estimator-check/1.
"""
import argparse
import gc
import json
import math
import os
import re
import shlex
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import m2_writers as mw  # noqa: E402
import nuaa_bn_recalibration as BR  # noqa: E402
import nuaa_bn_test as BT  # noqa: E402
import nuaa_folds_run as FR  # noqa: E402
import nuaa_session_ablation_run as SR  # noqa: E402
import pad_anet_comparison as C  # noqa: E402
import pad_anet_definitions as DEF  # noqa: E402
import pad_anet_models as AN  # noqa: E402
import pad_arch_comparison as PAC  # noqa: E402
import pad_arch_definitions as D  # noqa: E402
import pad_arch_run as R  # noqa: E402
import path_guard as pg  # noqa: E402

Failure, fail = FR.Failure, FR.fail
sha256_bytes, sha256_file, read_regular, rel_path, csv_text, parse_csv = (FR.sha256_bytes, FR.sha256_file, FR.read_regular,
                                                                          FR.rel_path, FR.csv_text, FR.parse_csv)
now = FR.now

DEFAULT_CONFIG = "delivery/anetrun/pad_anet_run.json"
DEFAULT_OUT_ROOT = "results/anetrun"
CONFIG_SCHEMA = "pad-anet-run-config/1"
RUN_SCHEMA = "pad-anet-run/1"
ESTIMATOR_SCHEMA = "pad-anet-estimator-check/1"
RUNS = {r: (n, p) for r, n, p in ((C.run_name(n, p), n, p) for p in DEF.PATIENCES for n in AN.NETWORKS)}   # nome -> (rete, patience)
RUN_TAG = {r: f"{C.RUN_TAG[n]}-p{p}" for r, (n, p) in RUNS.items()}                                          # anet_v2_1_p4 -> v2_1-p4
RUN_ID_RE = re.compile(r"^\d{8}-ANETRUN-(v1|v2_1|v2_2)-p(4|12)-[0-9a-f]{7,40}(?:-r([2-9]|[1-9]\d+))?$")
EST_ID_RE = re.compile(r"^\d{8}-ANETEST-[0-9a-f]{7,40}(?:-r([2-9]|[1-9]\d+))?$")
LOOKS_LIKE_RUN = re.compile(r"^\d{8}-ANETRUN-([^-]*)-p([^-]*)-", re.IGNORECASE)       # una voce che somiglia a un run id (forma da controllare)
LOOKS_LIKE_EST = re.compile(r"^\d{8}-ANETEST-", re.IGNORECASE)
THIS_FILE = "scripts/pad_anet_run.py"
CODE_FILES = C.CODE_FILES + ("scripts/pad_anet_comparison.py",)
TOP_KEYS = ("schema", "task", "authorisation", "operations", "comparison", "trial_record", "plan_files", "code", "estimator_check", "runs",
            "machine_rule", "sequence", "checkpoint", "non_evaluable", "estimator", "dev_evaluation", "cap", "after_stop", "attempts", "outputs",
            "choices_of_the_author", "test")
CHOICE_KEYS = ("sequence", "in_memory_dev", "checkpoint", "card_check", "estimator_check_process", "after_stop", "attempts", "machine_scope",
               "threads_record", "evaluator_agreement", "dev_frames", "outputs", "deviation_record", "launch_conditions")
PLAN_KEYS = ("run_matrix", "run_order", "run_caps")
PACKAGE_ENTRIES = {"status_sha256": "trial/status.json", "trial_json_sha256": "trial/trial.json", "run_matrix": "plan/run_matrix.csv",
                   "run_order": "plan/run_order.csv", "run_caps": "plan/run_caps.csv"}
ARM = R.ARM
HISTORY_KEYS = PAC.HISTORY_KEYS
STATUS_CODE = R.STATUS_CODE                                # PASS 0, FAIL 1, INCOMPLETE 2, MISMATCH 4
STATUS, LOG, PUBLIC, RESTRICTED = FR.STATUS, FR.LOG, R.PUBLIC, R.RESTRICTED
PUBLIC_OUTPUTS = R.PUBLIC_OUTPUTS
EST_OUTPUTS = R.EST_OUTPUTS
CAP_UNIT = PAC.RUN_CAP_UNIT
DISCARDED = R.DISCARDED
NOT_EVALUABLE = DEF.NOT_EVALUABLE
STATE_FIELDS = R.STATE_FIELDS
MODEL_COLUMNS = (("run", "position", "network", "fold", "seed", "patience", "model_status", "phase_1", "phase_2", "evaluable",
                  "not_evaluable_reason", "epochs_run", "restored_epoch", "restored_by", "best_val_loss", "history_non_finite")
                 + tuple(f"{s.replace('-', '_')}_{f}" for s in D.BN_STATES for f in STATE_FIELDS)
                 + ("evaluator_max_abs_diff", "evaluator_discordant_05", "evaluator_discordant_eer", "evaluator_agreement", "train_seconds",
                    "train_cpu_seconds", "phase_1_seconds", "phase_2_seconds"))
ONE_CLASS_COLUMNS = R.ONE_CLASS_COLUMNS
EST_COLUMNS = R.EST_COLUMNS
EXTRA_PROTECTED = R.EXTRA_PROTECTED + ("results/archrun", "results/archeval", "results/anetcmp")
CHECKPOINT_TEXT = R.CHECKPOINT_TEXT
NOT_EVALUABLE_CHECKPOINT = "none: the model is not evaluable under rule (n) of the pre-registration, no checkpoint saved"
TEST_SENTENCE = R.TEST_SENTENCE
TEST_READ = (
    f"{TEST_SENTENCE}. What each command reads. check-machine: this configuration and, through pad_anet_comparison.load_config, the "
    "configuration of PR A with the files it pins (no frame, no checkpoint); the trial record of PR A. check-inputs and run: this "
    "configuration; the configuration of PR A and what pad_anet_comparison.load_config and load_inputs read (the C2 pre-registration and "
    "the configurations it pins, the PR #27 configuration, the C2 trial record, the public status.json and results.json of the C2 "
    "evaluation, the architecture file, the code); the code files pinned here and this file; the status.json and trial.json of the "
    "official C5 trial and the SHA256SUMS of the PR A package that lists them; run_matrix.csv, run_order.csv and run_caps.csv, compared "
    "with the files regenerated from the configuration of PR A, the public files of the C2 evaluation and the trial; the PR #28 "
    "configuration (bytes, for its SHA-256); what pad_arch_comparison.load_inputs reads (its TEST_READ: the five M2 manifests, the cache "
    "hashed whole, the extracted frames file loaded whole, both frame files also holding frames of the held-out subjects, the cells and "
    "folds runs, the two accepted LivenessNet runs); the 12 training cards of arm B of each accepted run (keys of the fitting and dev "
    "frames); with --estimator-check, the status.json and estimator_check.json of a check-estimator record. run, in addition: the image "
    "arrays of the cache, loaded whole into the frame store; of all the frames it assembles only the fitting and dev frames of arm B (an "
    "explicit check refuses any other role); it reads back its own checkpoints. check-estimator: what pad_arch_run.cmd_check_estimator "
    "reads (the configurations of PRs #27 and #28, what nuaa_bn_recalibration.load_inputs and nuaa_bn_test.load_bnrecal read: the M2 "
    "manifests, cache and extracted frames file, the cells and folds runs, the parent run at patience 4 with per model card, dev scores, "
    "threshold and checkpoint, its aggregate.json only as bytes, the BNRECAL run with dev and fitting rows only); it assembles only "
    "fitting and dev frames. No command loads the fold schema (fold<k>_test.csv) or opens a <model>_test_scores.csv")

# punti di iniezione (solo i test)
REPO_STATE = None       # (root) -> {"commit", "dirty"} (default: SR.repo_state)
ARCH_LOADER = None      # (path) -> modulo con create_model (default: FR.default_arch_loader)
TRAINER = None          # come pad_anet_comparison.anet_trainer
SAVE_MODEL = None       # (model, path) -> None (default: pad_arch_run.save_checkpoint)
MAKE_SCORER = None      # (weights_path, sha256) -> funzione(X uint8) -> probabilità N x 2 (default: FR.tf_scorer)
REESTIMATE = None       # (ctx, rec, fit, dev) -> risultato della fase 2 (default: reestimate_model)
ESTIMATOR_BASE = None   # radice degli ingressi di check-estimator (default: --input-root)
MY_HOOKS = ("REPO_STATE", "ARCH_LOADER", "TRAINER", "SAVE_MODEL", "MAKE_SCORER", "REESTIMATE", "ESTIMATOR_BASE")
HOOKS = {**C.HOOKS, "pad_anet_run": MY_HOOKS}
PINNED_VALUES = C.PINNED_VALUES


# ============================================================================ configurazione

def _text(v, what: str) -> None:
    if not isinstance(v, str) or not v.strip():
        fail(f"configuration: {what} must be a non-empty text")


def _keys(d, keys, what: str) -> None:
    if not isinstance(d, dict) or sorted(d) != sorted(keys):
        fail(f"configuration: {what} needs exactly {', '.join(sorted(keys))}")


def _pin(e, what: str) -> None:
    _keys(e, ["path", "sha256"], what)
    _text(e["path"], f"{what}.path")
    if not SR._hex(e["sha256"]):
        fail(f"configuration: {what}.sha256 malformed")


def _author(v, what: str) -> None:
    _text(v, what)
    if "choice of the author" not in v:
        fail(f"configuration: {what} must be declared a choice of the author")


def expected_orders() -> dict:
    """L'ordine della ristima: LivenessNet (solo check-estimator) quello della PR #27 (AM.LIVENESSNET_BN, controllato
    contro la sua configurazione); ogni AttackNet quello del registro (AN.bn_order), lo stesso della prova tecnica."""
    return {"LivenessNet": list(R.AM.LIVENESSNET_BN), **{n: list(AN.bn_order(n)) for n in AN.NETWORKS}}


def load_config(p: Path) -> tuple:
    """(configurazione, SHA-256 dei byte letti). Insieme chiuso a ogni livello e valori fissati controllati qui, prima di
    leggere qualunque altro file; i testi sono controllati come presenti (il loro contenuto è quello di make_config, la
    suite lo verifica)."""
    b = read_regular(p, "configuration")
    cfg = SR.read_json(b, "configuration")
    _keys(cfg, TOP_KEYS, "the top level")
    if cfg["schema"] != CONFIG_SCHEMA:
        fail(f"configuration: schema must be {CONFIG_SCHEMA}")
    for k in ("task", "authorisation", "operations", "sequence", "after_stop", "attempts", "test"):
        _text(cfg[k], k)
    if TEST_SENTENCE not in cfg["test"]:
        fail("configuration: test must state that no test frame is assembled or evaluated")
    _pin(cfg["comparison"], "comparison")
    tr = cfg["trial_record"]
    _keys(tr, ["run_id", "folder", "status_sha256", "trial_json_sha256", "package_sums"], "trial_record")
    if not isinstance(tr["run_id"], str) or not C.RUN_ID_RE.match(tr["run_id"]):
        fail("configuration: trial_record.run_id must be an ANETTRIAL run id")
    _text(tr["folder"], "trial_record.folder")
    for k in ("status_sha256", "trial_json_sha256"):
        if not SR._hex(tr[k]):
            fail(f"configuration: trial_record.{k} malformed")
    _pin(tr["package_sums"], "trial_record.package_sums")
    _keys(cfg["plan_files"], PLAN_KEYS, "plan_files")
    for k in PLAN_KEYS:
        _pin(cfg["plan_files"][k], f"plan_files.{k}")
    if not isinstance(cfg["code"], dict) or sorted(cfg["code"]) != sorted(CODE_FILES) or not all(SR._hex(v) for v in cfg["code"].values()):
        fail(f"configuration: code must pin exactly {', '.join(CODE_FILES)} by SHA-256")
    ec = cfg["estimator_check"]
    _keys(ec, ["bn_recalibration_config", "bn_test_config", "bnrecal_run_id", "models", "cap_minutes", "cap_unit", "rule"], "estimator_check")
    _pin(ec["bn_recalibration_config"], "estimator_check.bn_recalibration_config")
    _pin(ec["bn_test_config"], "estimator_check.bn_test_config")
    if ec["bnrecal_run_id"] != "20261114-BNRECAL-a10a082" or ec["models"] != 36:
        fail("configuration: estimator_check must name the run 20261114-BNRECAL-a10a082 and its 36 models (PROTOCOL.md §3)")
    if isinstance(ec["cap_minutes"], bool) or not isinstance(ec["cap_minutes"], int) or ec["cap_minutes"] <= 0 or ec["cap_unit"] != PAC.CAP_UNIT:
        fail(f"configuration: estimator_check.cap_minutes must be a positive integer of {PAC.CAP_UNIT}")
    _text(ec["rule"], "estimator_check.rule")
    if cfg["runs"] != list(RUNS):
        fail(f"configuration: runs must be {', '.join(RUNS)} (PROTOCOL.md §4, in this order)")
    mr = cfg["machine_rule"]
    _keys(mr, ["fields", "versions", "threads", "consequence"], "machine_rule")
    if (mr["fields"], mr["versions"], mr["threads"]) != (list(R.MACHINE_FIELDS), list(R.VERSION_FIELDS), list(R.THREAD_FIELDS)):
        fail("configuration: machine_rule must list the fields of pad_arch_run.machine_rule and threads_rule")
    _text(mr["consequence"], "machine_rule.consequence")
    _keys(cfg["checkpoint"], ["format", "when"], "checkpoint")
    for k in ("format", "when"):
        _text(cfg["checkpoint"][k], f"checkpoint.{k}")
    ne = cfg["non_evaluable"]
    _keys(ne, ["rule", "in_memory_dev", "fitting_scores", "statistics", "record"], "non_evaluable")
    for k, v in ne.items():
        _text(v, f"non_evaluable.{k}")
    es = cfg["estimator"]
    _keys(es, ["estimator", "batch", "orders", "order_rule"], "estimator")
    _text(es["estimator"], "estimator.estimator")
    _text(es["order_rule"], "estimator.order_rule")
    if isinstance(es["batch"], bool) or not isinstance(es["batch"], int) or es["batch"] <= 0:
        fail("configuration: estimator.batch must be a positive integer")
    if es["orders"] != expected_orders():
        fail("configuration: estimator.orders must be the PR #27 order for LivenessNet and the registry order (pad_anet_models.bn_order) "
             "for each AttackNet")
    de = cfg["dev_evaluation"]
    _keys(de, ["states", "threshold", "one_class", "agreement"], "dev_evaluation")
    if de["states"] != list(D.BN_STATES):
        fail(f"configuration: dev_evaluation.states must be {list(D.BN_STATES)}")
    for k in ("threshold", "one_class", "agreement"):
        _text(de[k], f"dev_evaluation.{k}")
    _keys(cfg["cap"], ["source", "unit", "checks"], "cap")
    if cfg["cap"]["unit"] != CAP_UNIT:
        fail(f"configuration: cap.unit must be {CAP_UNIT}")
    for k in ("source", "checks"):
        _text(cfg["cap"][k], f"cap.{k}")
    out = cfg["outputs"]
    _keys(out, ["run", "public", "restricted"], "outputs")
    if out["run"] != [STATUS, LOG] or out["public"] != list(PUBLIC_OUTPUTS) or out["restricted"] != [f"<model>{x}" for x in R.PER_MODEL] + ["SHA256SUMS"]:
        fail("configuration: outputs differ from the outputs of this runner")
    _keys(cfg["choices_of_the_author"], CHOICE_KEYS, "choices_of_the_author")
    for k, v in cfg["choices_of_the_author"].items():
        _author(v, f"choices_of_the_author.{k}")
    return cfg, sha256_bytes(b)


class Pins:
    """Ciò che la configurazione fissa, letto e controllato: configurazione della PR A (cfg_c5, sha_c5, con la
    preregistrazione C2 cfg_pr, pcfg e la configurazione della PR #27 bn_cfg), codice, record della prova C5 (trial),
    posizioni e tetti dei sei run, configurazione della PR #28 (bt_cfg, solo in check-estimator), SHA-256 dei file letti."""


def check_code(cfg: dict) -> dict:
    """Ogni file di CODE_FILES uguale allo SHA-256 fissato nella configurazione; più lo SHA-256 attuale di questo file."""
    got = {f: sha256_bytes(read_regular(ROOT / f, "code file")) for f in CODE_FILES}
    bad = [f for f in CODE_FILES if got[f] != cfg["code"][f]]
    if bad:
        fail(f"code files {bad} differ from the SHA-256 pinned in the configuration")
    return {**got, THIS_FILE: sha256_bytes(read_regular(Path(__file__), "code file"))}


def read_pinned(base: Path, e: dict, what: str) -> tuple:
    return R.read_pinned(base, e, what)


def package_check(base: Path, cfg: dict) -> dict:
    """Il record della prova e i tre file del piano ancorati al SHA256SUMS del pacchetto della PR A (fissato per SHA-256):
    le righe trial/status.json, trial/trial.json e plan/<file> uguali agli SHA-256 della configurazione."""
    tr = cfg["trial_record"]
    _, b = read_pinned(base, tr["package_sums"], "package SHA256SUMS")
    sums = FR.read_sums(b, "package SHA256SUMS")
    want = {"status_sha256": tr["status_sha256"], "trial_json_sha256": tr["trial_json_sha256"],
            **{k: cfg["plan_files"][k]["sha256"] for k in PLAN_KEYS}}
    bad = [entry for key, entry in PACKAGE_ENTRIES.items() if sums.get(entry) != want[key]]
    if bad:
        fail(f"PR A package: {', '.join(bad)} of its SHA256SUMS differ from the SHA-256 of the configuration")
    return {entry: sums[entry] for entry in PACKAGE_ENTRIES.values()}


def run_positions(matrix_bytes: bytes, run: str) -> list:
    """Le 12 posizioni di un run lette da run_matrix.csv (già uguale al file rigenerato): posizioni 1..12 consecutive,
    la rete e la patience del run, fold 1-4 × seed 42, 43, 44 nell'ordine del piano, braccio B, nessun riferimento."""
    net, patience = RUNS[run]
    rows = [r for r in parse_csv(matrix_bytes, C.MATRIX_COLUMNS, "run_matrix.csv") if r["run"] == run]
    want = [(f, s) for f in DEF.FOLDS for s in DEF.PLAN_SEEDS]
    if len(rows) != len(want):
        fail(f"run_matrix.csv: {len(rows)} positions for {run}, the plan has {len(want)}")
    out = []
    for i, (r, (f, s)) in enumerate(zip(rows, want), start=1):
        got = (r["position"], r["network"], r["fold"], r["seed"], r["patience"], r["arm"], r["reference"])
        if got != (str(i), net, str(f), str(s), str(patience), ARM, "none"):
            fail(f"run_matrix.csv: position {i} of {run} is {got}, the plan has {(i, net, f, s, patience, ARM, 'none')}")
        out.append({"position": i, "network": net, "fold": f, "seed": s, "patience": patience})
    return out


def run_cap_hours(caps_bytes: bytes, run: str) -> float:
    rows = [r for r in parse_csv(caps_bytes, C.CAPS_COLUMNS, "run_caps.csv") if r["run"] == run]
    if len(rows) != 1:
        fail(f"run_caps.csv: {len(rows)} rows for {run} (exactly one expected)")
    try:
        cap = float(rows[0]["cap_wall_hours"])
    except ValueError:
        fail(f"run_caps.csv: cap_wall_hours of {run} is not a number")
    if not math.isfinite(cap) or cap <= 0:
        fail(f"run_caps.csv: cap_wall_hours of {run} must be a finite positive number")
    return cap


def load_pins(cfg: dict, base: Path, est_base: Path = None) -> Pins:
    """I file fissati, ciascuno con il suo SHA-256: configurazione della PR A (pad_anet_comparison.load_config, che
    valida anche la preregistrazione C2 e la configurazione della PR #27), codice (anche pad_anet_comparison.check_code),
    record della prova C5 (uguale alle righe del SHA256SUMS del pacchetto e accettato da pad_anet_comparison.load_trial:
    stessa configurazione e stesso codice della PR A), i tre file del piano uguali byte per byte a quelli rigenerati
    (pad_anet_comparison.plan_texts con i file pubblici della valutazione C2 e la prova), la configurazione della PR #27
    uguale a quella della PR A e, per byte, quella della PR #28; con est_base (solo check-estimator) la PR #28 validata."""
    pins = Pins()
    pins.files = {}
    cp, _ = read_pinned(base, cfg["comparison"], "PR A configuration")
    pins.cfg_c5, pins.sha_c5, pins.cfg_pr, pins.pcfg, pins.bn_cfg = C.load_config(cp, base)
    pins.files["comparison"] = pins.sha_c5
    pins.code = check_code(cfg)
    code_c5 = C.check_code(pins.cfg_c5)
    if {k: v for k, v in pins.code.items() if k in code_c5} != code_c5:
        fail("code: the files pinned by the PR A configuration differ from those pinned here")
    tr = cfg["trial_record"]
    pins.package = package_check(base, cfg)
    pins.files["package_sums"] = tr["package_sums"]["sha256"]
    folder = rel_path(base, tr["folder"])
    for fn, key in ((STATUS, "status_sha256"), (PAC.TRIAL_JSON, "trial_json_sha256")):
        h = sha256_bytes(read_regular(folder / fn, f"trial record {fn}"))
        if h != tr[key]:
            fail(f"trial record {fn}: SHA-256 {h} differs from trial_record.{key}")
    pins.trial, pins.trial_sha = C.load_trial(folder, pins.cfg_c5, pins.sha_c5, code_c5)
    if pins.trial["run_id"] != tr["run_id"] or pins.trial.get("hooks", {}).get("official_configuration") is not True:
        fail(f"trial record: run id {pins.trial['run_id']} or an unofficial trial; the configuration requires the official {tr['run_id']}")
    pins.files["trial_status"], pins.files["trial_json"] = tr["status_sha256"], tr["trial_json_sha256"]
    cells = C.load_c2_results(pins.cfg_c5, base)
    want = C.plan_texts(pins.cfg_c5, pins.sha_c5, pins.pcfg, cells, pins.trial, pins.trial_sha)
    plan = {}
    for k in PLAN_KEYS:
        _, b = read_pinned(base, cfg["plan_files"][k], f"{k}.csv")
        if b != want[f"{k}.csv"].encode("utf-8"):
            fail(f"{k}.csv differs from the file regenerated from the PR A configuration, the C2 evaluation and the trial "
                 "(pad_anet_comparison.plan_texts)")
        plan[k] = b
        pins.files[k] = sha256_bytes(b)
    order = parse_csv(plan["run_order"], C.ORDER_COLUMNS, "run_order.csv")
    if [r["run"] for r in order] != list(RUNS):
        fail("run_order.csv: the six runs are not in the order of this configuration")
    pins.positions = {r: run_positions(plan["run_matrix"], r) for r in RUNS}
    pins.caps = {r: run_cap_hours(plan["run_caps"], r) for r in RUNS}
    ec = cfg["estimator_check"]
    brp, brb = read_pinned(base, ec["bn_recalibration_config"], "PR #27 configuration")
    if sha256_bytes(brb) != pins.cfg_c5["bn_recalibration_config"]["sha256"]:
        fail("estimator_check: the PR #27 configuration differs from the one of the PR A configuration")
    btp, _ = read_pinned(base, ec["bn_test_config"], "PR #28 configuration")
    pins.bt_cfg = None
    if est_base is not None:
        pins.bt_cfg, _, bn_cfg = BT.load_config(btp, est_base)
        if bn_cfg != pins.bn_cfg or pins.bt_cfg["bnrecal_run"]["config_sha256"] != ec["bn_recalibration_config"]["sha256"]:
            fail("estimator_check: the PR #27 configuration pinned here is not the one of the PR #28 configuration")
        if pins.bt_cfg["bnrecal_run"]["run_id"] != ec["bnrecal_run_id"] or pins.bt_cfg["bnrecal_run"]["models"] != ec["models"]:
            fail("estimator_check: the BNRECAL run or its number of models differ from the PR #28 configuration")
    if pins.bn_cfg["batchnorm"]["layers"] != cfg["estimator"]["orders"]["LivenessNet"] or pins.bn_cfg["evaluation"]["batch"] != cfg["estimator"]["batch"]:
        fail("estimator: the LivenessNet order or the batch differ from the PR #27 configuration")
    if cfg["estimator"]["batch"] != pins.cfg_c5["trial"]["estimator_batch"]:
        fail("estimator: the batch differs from the estimator batch of the trial (PR A configuration)")
    for n in AN.NETWORKS:
        if cfg["estimator"]["orders"][n] != pins.cfg_c5["architectures"]["networks"][n]["batchnorm_names"]:
            fail(f"estimator: the order of {n} differs from the PR A configuration")
    if pins.bn_cfg["budget"]["wall_time_cap_minutes"] != ec["cap_minutes"]:
        fail("estimator_check.cap_minutes must be the cap of the PR #27 configuration (budget.wall_time_cap_minutes)")
    pins.files["bn_recalibration_config"] = ec["bn_recalibration_config"]["sha256"]
    pins.files["bn_test_config"] = ec["bn_test_config"]["sha256"]
    return pins


# ============================================================================ agganci, configurazione ufficiale

def check_hooks(official: bool) -> dict:
    """Gli agganci dei test di questo modulo e dei moduli del progetto (HOOKS: quelli della PR A più questi) tutti None e
    i valori che solo i test sostituiscono (PINNED_VALUES) quelli dell'import; nei comandi ufficiali altrimenti Failure
    prima di ogni lavoro. Restituisce quelli impostati o cambiati (registrati in status.json)."""
    mods = {m: sys.modules[__name__] if m == "pad_anet_run" else sys.modules[m] for m in HOOKS}
    setv = sorted(f"{m}.{k}" for m, names in HOOKS.items() for k in names if getattr(mods[m], k) is not None)
    changed = sorted(f"{m}.{k}" for (m, k), v in PINNED_VALUES.items() if getattr(sys.modules[m], k) is not v)
    if official and (setv or changed):
        fail(f"test hooks are set or test values changed ({', '.join(setv + changed)}): the official commands run with every hook None")
    return {"official_configuration": official, "set": setv, "changed": changed}


def is_official_config(cfg_p: Path) -> bool:
    try:
        return os.path.samefile(cfg_p, ROOT / DEFAULT_CONFIG)
    except OSError:
        return False


# ============================================================================ macchina e thread (contro la prova C5)

def machine_rules(pins: Pins) -> tuple:
    """La regola della macchina e delle variabili dei thread di C2 (pad_arch_run.machine_rule e thread_environment_rule)
    applicata al record della prova tecnica C5: stesso modello di macchina, stesse versioni, stesse variabili dei thread
    (assenti). Una differenza -> Failure prima di ogni cartella."""
    env = R.thread_environment_rule(pins.cfg_pr, pins.trial)
    return env, R.machine_rule(pins.trial)


# ============================================================================ fase 1 e fase 2

class Ctx:
    """Lo stato di un run in corso."""


def card_problems(card: dict, training: dict, seed: int, weights: Path, network: str) -> list:
    """I controlli della scheda dei run accettati (pad_arch_run.card_problems) con una sola sostituzione dichiarata: una
    storia con valori non finiti (scritti come null dalla regola (n)) è controllata sostituendo ogni null con +inf,
    così l'epoca ripristinata attesa è quella della val_loss finita minima (pad_anet_definitions.finite_argmin) e non il
    primo NaN di np.argmin; su storie finite il controllo è identico a quello di C2."""
    h = card.get("history")
    if isinstance(h, dict) and any(v is None for k in HISTORY_KEYS for v in (h.get(k) or [])):
        h = {k: ([math.inf if v is None else v for v in h[k]] if k in HISTORY_KEYS and isinstance(h.get(k), list) else h[k]) for k in h}
        card = {**card, "history": h}
    return R.card_problems(card, training, seed, weights, network)


def history_matches(epochs: list, history: dict, n: int) -> bool:
    """La storia registrata dalla callback dei tempi uguale a quella della scheda (valori non finiti come None, la forma
    della regola (n)), epoca per epoca."""
    if len(epochs) != n:
        return False
    return [[DEF.finite_or_none(e[k]) for k in HISTORY_KEYS] for e in epochs] == [[history[k][i] for k in HISTORY_KEYS] for i in range(n)]


def not_evaluable(ctx, rec: dict, reason: str) -> None:
    """Un modello non valutabile per la regola (n): registrato con il motivo (run.log, rec), il run continua."""
    rec["evaluable"] = False
    rec["reason"] = reason
    print(f"   {rec['name']}: {reason}", flush=True)


def phase_one(ctx, pos: dict, rec: dict) -> None:
    """Fase 1 di una posizione (la fase 1 di pad_arch_run.phase_one per le AttackNet): addestramento con il trainer della
    prova (anet_trainer) e il costruttore del registro; regola (n) sulla storia e sui pesi ripristinati
    (DEF.training_outcome); punteggi del dev in memoria (regola (n), C.forward_checked); checkpoint; scheda; punteggi del
    dev con lo scorer dei run accettati e controllo del val_loss. rec["phase_1"] = True quando l'esito della fase 1 è
    scritto (anche per un modello non valutabile: allora solo la scheda, rec["evaluable"] False)."""
    net, fold, seed = pos["network"], pos["fold"], pos["seed"]
    name = rec["name"]
    t_phase = time.monotonic()
    ctx.budget.check(f"train {name}")
    fit, dev = R.unit_frames(ctx.store, ctx.inp, "fitting", fold), R.unit_frames(ctx.store, ctx.inp, "dev", fold)
    ref = ctx.refs[(fold, seed)]
    for role, part in (("fitting", fit), ("dev", dev)):
        PAC.keys_check(ref["card"], role, part["rows"])
    rec["threads_before"] = PAC.check_threads(ctx.tf, ctx.cfg_pr, f"before {name}")
    epochs = []
    timer = PAC.EpochTimer(epochs)
    t0, c0 = time.monotonic(), time.process_time()
    card, model = (TRAINER or C.anet_trainer)({"X_fit": fit["X"], "y_fit": fit["y"], "X_dev": dev["X"], "y_dev": dev["y"]}, ctx.training,
                                              net, seed, lambda: C.build_checked(ctx.cfg_c5, net, ctx.arch),
                                              lambda step: ctx.budget.check(f"{step} of {name}"), timer)
    rec["train_seconds"], rec["train_cpu_seconds"] = time.monotonic() - t0, time.process_time() - c0
    if not history_matches(epochs, card["history"], card["epochs_run"]):
        fail(f"{name}: the history recorded by the timing callback differs from the Keras history")
    weights_finite = DEF.arrays_finite(model.get_weights()) if card["restored_epoch"] is not None else None
    outcome = DEF.training_outcome(card["history"], weights_finite)
    rec["outcome"] = outcome
    rec["card"] = {k: card[k] for k in ("epochs_run", "restored_epoch", "restored_by", "best_val_loss", "non_finite")}
    base_card = {**card, "test_evaluation": "none (no test frame is assembled in this run)", "network": net, "arm": ARM, "fold": fold,
                 "run": ctx.run_name, "position": pos["position"], "fitting_keys_sha256": BR.keys_sha(fit["rows"]),
                 "dev_keys_sha256": BR.keys_sha(dev["rows"]), "epoch_timing": [{k: (DEF.finite_or_none(v) if k in HISTORY_KEYS else v)
                                                                                for k, v in e.items()} for e in epochs],
                 "training_overhead": {"seconds": rec["train_seconds"] - sum(e["seconds_total"] for e in epochs),
                                       "cpu_seconds": rec["train_cpu_seconds"] - sum(e["cpu_seconds"] for e in epochs)},
                 "rule_n": outcome}
    reason = None if outcome["evaluable"] else outcome["reason"]
    if reason is None:
        # regola (n), stato ordinario: dev e fitting valutati con il modello in memoria (inferenza, nessuno stato del modello
        # cambia; dopo l'addestramento e la registrazione dei tempi per epoca); un valore non finito -> non valutabile
        try:
            for role, part in (("dev", dev), ("fitting", fit)):
                ctx.budget.check(f"in-memory {role} scoring {name}")
                C.forward_checked(model, part, ctx.cfg["estimator"]["batch"], f"in-memory {role}", "ordinary")
        except C.NotEvaluable as e:
            reason = str(e)
    if reason is not None:
        del model
        gc.collect()
        card = {**base_card, "rule_n": {**outcome, "evaluable": False, "reason": reason}, "not_evaluable_reason": reason,
                "checkpoint": NOT_EVALUABLE_CHECKPOINT, "weights_file": None, "weights_sha256": None, "saved": None}
        rec["weights_sha256"] = None
        rec["threads_after"] = PAC.check_threads(ctx.tf, ctx.cfg_pr, f"after {name}")
        mw.write_all(ctx.rdir, {f"{name}_card.json": PAC.json_text(card, f"{name}_card.json")})
        not_evaluable(ctx, rec, reason)
        rec["phase_1"] = True
        rec["phase_1_seconds"] = time.monotonic() - t_phase
        return
    weights = ctx.rdir / f"{name}.keras"
    (SAVE_MODEL or R.save_checkpoint)(model, weights)
    del model
    gc.collect()
    card = {**base_card, "checkpoint": CHECKPOINT_TEXT, "weights_file": weights.name, "weights_sha256": sha256_file(weights), "saved": now()}
    rec["weights_sha256"] = card["weights_sha256"]
    if problems := card_problems(card, ctx.training, seed, weights, net):
        fail(f"{name}: " + "; ".join(problems))
    if card["n_fitting"] != ctx.pcfg["arms"][ARM]["total"] or card["n_dev"] != ctx.pcfg["dev"]["total"]:
        fail(f"{name}: the card reports {card['n_fitting']} fitting and {card['n_dev']} dev frames, expected "
             f"{ctx.pcfg['arms'][ARM]['total']} and {ctx.pcfg['dev']['total']}")
    ctx.budget.check(f"dev scoring {name}")
    scorer = (MAKE_SCORER or FR.tf_scorer)(weights, card["weights_sha256"])
    p = FR.probabilities(scorer, dev["X"])
    z = SR.dev_logits(weights, card, dev["X"], p, name)
    vl = SR.val_loss_logits(z, dev["y"])
    diff = abs(vl - card["best_val_loss"])
    if not (diff <= FR.VAL_LOSS_TOL[0] or diff <= FR.VAL_LOSS_TOL[1] * card["best_val_loss"]):
        fail(f"{name}: val_loss recomputed on the dev {vl:.6f}, recorded minimum {card['best_val_loss']:.6f}: the checkpoint is not the "
             "one of minimum finite val_loss")
    card["val_loss_recomputed"] = vl
    rec["scorer_p"] = p[:, 1].copy()
    rec["evaluable"] = True
    rec["threads_after"] = PAC.check_threads(ctx.tf, ctx.cfg_pr, f"after {name}")
    mw.write_all(ctx.rdir, {f"{name}_dev_scorer.csv": csv_text(R.SCORER_COLUMNS, R.scorer_rows(rec, dev, p[:, 1], z))}, newline="")
    mw.write_all(ctx.rdir, {f"{name}_card.json": PAC.json_text(card, f"{name}_card.json")})
    rec["phase_1"] = True
    rec["phase_1_seconds"] = time.monotonic() - t_phase
    nf = "; non-finite " + ", ".join(f"{k} from epoch {e}" for k, e in card["non_finite"]) if card["non_finite"] else ""
    print(f"== {name}: {card['epochs_run']} epochs, restored epoch {card['restored_epoch']}, val_loss {vl:.6f} recomputed on the dev{nf}",
          flush=True)


def reestimate_model(ctx, rec: dict, fit: dict, dev: dict) -> dict:
    """Fase 2 di un modello (copia dichiarata di pad_arch_run.reestimate_model): sostituzioni pad_anet_comparison.check_loaded
    (il registro nuovo) al posto di check_loaded, pad_anet_comparison.forward_checked (regola (n): punteggi o logit non
    finiti -> NotEvaluable) al posto di BR.evaluate, e le statistiche ristimate non finite tradotte in NotEvaluable,
    come nella prova tecnica. Ogni altro difetto -> Failure."""
    tf = BR.tf_module()
    name, net = rec["name"], rec["network"]
    bn_cfg, batch = ctx.bn_cfg, ctx.cfg["estimator"]["batch"]
    order = ctx.cfg["estimator"]["orders"][net]
    tf.keras.backend.clear_session()
    orig = BR.load_model(ctx.rdir / f"{name}.keras", rec["weights_sha256"])
    C.check_loaded(orig, ctx.cfg_c5, net)
    ctx.budget.check(f"ordinary dev scoring {name}")
    ev_ord, _ = C.forward_checked(orig, dev, batch, "dev", "ordinary")
    if ev_ord["softmax_gap"] > BR.tol(bn_cfg, "softmax_of_logits_max_abs_diff"):
        fail(f"{name}: the softmax of the logits differs from the ordinary dev probabilities by {ev_ord['softmax_gap']:.2e}")
    copy = tf.keras.models.clone_model(orig)
    copy.set_weights(orig.get_weights())
    snap = BR.snapshot_weights(orig)
    if BR.snapshot_weights(copy) != snap:
        fail(f"{name}: the in-memory copy differs from the original before the re-estimation")
    try:
        stats = R.reestimate(copy, fit, order, batch, ctx.budget, name)
    except Failure as e:
        if str(e).endswith("the re-estimated mean or variance is not finite"):
            raise C.NotEvaluable(f"{NOT_EVALUABLE}: non-finite re-estimated statistics") from None
        raise
    if BR.snapshot_weights(orig) != snap:
        fail(f"{name}: the original model changed during the re-estimation")
    ctx.budget.check(f"consistency pass {name}")
    fit_after, final = C.forward_checked(copy, fit, batch, "fitting", "re-estimated", layers=order)
    ctx.budget.check(f"re-estimated dev scoring {name}")
    dev_after, _ = C.forward_checked(copy, dev, batch, "dev", "re-estimated")
    cons = R.pr27_checks(name, stats, snap, copy, order, fit_after, final, dev_after, bn_cfg)
    changes = {n: BR.change_summary(stats[n], bn_cfg["batchnorm"]["epsilon"]) for n in order}
    out = {"stats": stats, "cons": cons, "changes": changes, "ordinary": {"p": ev_ord["p"][:, 1], "z": ev_ord["z"]},
           "re_estimated": {"p": dev_after["p"][:, 1], "z": dev_after["z"]},
           "fitting_after": {k: fit_after[k] for k in ("accuracy", "val_loss", "auroc", "n")}}
    del orig, copy, final
    gc.collect()
    return out


def phase_two(ctx, rec: dict) -> None:
    """Fase 2 di un modello valutabile (pad_arch_run.phase_two, con la ristima di reestimate_model): i due stati della BN,
    soglie, decisioni e conteggi; accordo fra lo scorer della fase 1 e il valutatore sullo stato ordinario (registrato:
    un disaccordo dà MISMATCH alla fine). Un modello non valutabile qui (regola (n)) è registrato e il run continua."""
    name = rec["name"]
    t0 = time.monotonic()
    ctx.budget.check(f"re-estimation {name}")
    fit, dev = R.unit_frames(ctx.store, ctx.inp, "fitting", rec["fold"]), R.unit_frames(ctx.store, ctx.inp, "dev", rec["fold"])
    try:
        res = (REESTIMATE or reestimate_model)(ctx, rec, fit, dev)
    except C.NotEvaluable as e:
        not_evaluable(ctx, rec, str(e))
        rec["phase_2_seconds"] = time.monotonic() - t0
        return
    summ_o, unit_o = R.dev_state(dev, res["ordinary"]["p"], res["ordinary"]["z"], "ordinary", name)
    agreement = R.evaluator_agreement(rec["scorer_p"], res["ordinary"]["p"], summ_o["threshold_eer"], ctx.bn_cfg)
    summ_r, unit_r = R.dev_state(dev, res["re_estimated"]["p"], res["re_estimated"]["z"], "re-estimated", name)
    order = ctx.cfg["estimator"]["orders"][rec["network"]]
    doc = R.stats_doc(rec, res, summ_r, order, ctx.cfg)
    doc["dev_ordinary"], doc["evaluator_agreement"] = summ_o, agreement
    mw.write_all(ctx.rdir, {f"{name}_dev_ordinary.csv": csv_text(R.SCORE_COLUMNS, R.score_rows(rec, "ordinary", dev, res["ordinary"]["p"],
                                                                                             res["ordinary"]["z"], summ_o)),
                            f"{name}_dev_re-estimated.csv": csv_text(R.SCORE_COLUMNS, R.score_rows(rec, "re-estimated", dev, res["re_estimated"]["p"],
                                                                                                 res["re_estimated"]["z"], summ_r))}, newline="")
    mw.write_all(ctx.rdir, {f"{name}_bn_statistics.json": PAC.json_text(doc, f"{name}_bn_statistics.json")})
    rec["states"] = {"ordinary": summ_o, "re-estimated": summ_r}
    rec["evaluator_agreement"] = agreement
    rec["phase_2"] = True
    rec["phase_2_seconds"] = time.monotonic() - t0
    print(f"== {name}: dev EER threshold ordinary {R.fmt_thr(summ_o)}, re-estimated {R.fmt_thr(summ_r)}; evaluator agreement "
          f"{'yes' if agreement['pass'] else 'NO'}", flush=True)


def check_dev_frames(recs: list) -> dict:
    """Gli ID dei frame del dev (solo conteggi e impronte nelle uscite pubbliche): per ogni fold, tutti i modelli del run
    nei due stati della BN sugli stessi frame nello stesso ordine (stessa impronta D.ids_sha256 e stesso numero). In C2
    il confronto era fra le due reti di un'unità (pad_arch_run.check_dev_frames, che vuole le due reti di C2); qui un run
    ha una sola rete. Una differenza -> Failure (nessun ID nel messaggio)."""
    per_fold = {}
    for r in recs:
        for st in D.BN_STATES:
            s = (r.get("states") or {}).get(st)
            if s is not None:
                per_fold.setdefault(r["fold"], set()).add((s["n_frame_ids"], s["frame_ids_sha256"]))
    for f, v in per_fold.items():
        if len(v) != 1:
            fail(f"dev frames of fold {f}: the models of the run or the two BN states do not use the same frames in the same order")
    return {"folds_compared": len(per_fold), "models_compared": len([r for r in recs if r.get("states")]),
            "folds": {f"fold{f}": {"n_frames": next(iter(v))[0], "frame_ids_sha256": next(iter(v))[1]} for f, v in sorted(per_fold.items())}}


# ============================================================================ uscite

def model_status(rec: dict) -> str:
    if rec.get("discarded"):
        return DISCARDED
    if rec.get("evaluable") is False:
        return NOT_EVALUABLE
    if rec.get("phase_2"):
        return "complete"
    if rec.get("phase_1"):
        return "phase 1 complete, BN states not evaluated"
    return "not started"


def model_row(rec: dict, run: str, patience: int) -> list:
    c = rec.get("card") or {}
    ag = rec.get("evaluator_agreement") or {}
    nf = c.get("non_finite")
    v = {"run": run, "position": rec["position"], "network": rec["network"], "fold": rec["fold"], "seed": rec["seed"], "patience": patience,
         "model_status": model_status(rec), "phase_1": bool(rec.get("phase_1")), "phase_2": bool(rec.get("phase_2")),
         "evaluable": rec.get("evaluable"), "not_evaluable_reason": rec.get("reason"), "epochs_run": c.get("epochs_run"),
         "restored_epoch": c.get("restored_epoch"), "restored_by": c.get("restored_by"), "best_val_loss": c.get("best_val_loss"),
         "history_non_finite": "; ".join(f"{k} from epoch {e}" for k, e in nf) if nf else None,
         "evaluator_max_abs_diff": ag.get("max_abs_diff"), "evaluator_discordant_05": ag.get("discordant_05"),
         "evaluator_discordant_eer": ag.get("discordant_eer"), "evaluator_agreement": ag.get("pass"),
         "train_seconds": round(rec["train_seconds"], 1) if "train_seconds" in rec else None,
         "train_cpu_seconds": round(rec["train_cpu_seconds"], 1) if "train_cpu_seconds" in rec else None,
         "phase_1_seconds": round(rec["phase_1_seconds"], 1) if "phase_1_seconds" in rec else None,
         "phase_2_seconds": round(rec["phase_2_seconds"], 1) if "phase_2_seconds" in rec else None}
    for s in D.BN_STATES:
        st = (rec.get("states") or {}).get(s) or {}
        for f in STATE_FIELDS:
            v[f"{s.replace('-', '_')}_{f}"] = R.one_class_text(st.get(f)) if f == "one_class_05" else (st.get(f) if st else None)
    return [R.cell(v[k]) for k in MODEL_COLUMNS]


def one_class_rows(recs: list, patience: int) -> list:
    """Il conteggio della definizione (a) (pad_anet_definitions.one_class_counts, le cinque reti) per rete × patience ×
    stato della BN, sui modelli con quello stato valutato."""
    records = [{"network": r["network"], "patience": patience, "bn_state": s, "result": r["states"][s]["one_class_05"]}
               for r in recs for s in D.BN_STATES if s in (r.get("states") or {})]
    return DEF.one_class_counts(records) if records else []


def public_model(rec: dict) -> dict:
    """Ciò che il riepilogo pubblico riporta di un modello: nessun ID di frame o di soggetto; un modello tolto al tetto ha
    solo posizione, rete, fold, seed e stato."""
    base = {k: rec[k] for k in ("position", "name", "network", "fold", "seed") if k in rec}
    base["model_status"] = model_status(rec)
    if rec.get("discarded"):
        return base
    return base | {"phase_1": bool(rec.get("phase_1")), "phase_2": bool(rec.get("phase_2")), "evaluable": rec.get("evaluable"),
                   "reason": rec.get("reason"), "card": rec.get("card"), "dev": rec.get("states"),
                   "evaluator_agreement": rec.get("evaluator_agreement"), "threads_before": rec.get("threads_before"),
                   "threads_after": rec.get("threads_after"),
                   "seconds": {k: rec.get(k) for k in ("train_seconds", "train_cpu_seconds", "phase_1_seconds", "phase_2_seconds")}}


def report_md(doc: dict) -> str:
    L = [f"# NUAA run {doc['run']} ({doc['run_id']}): {doc['network']}, patience {doc['patience']}", ""]
    if not doc["official"]:
        L += [f"**Not an official run**: written outside {DEFAULT_OUT_ROOT} with --unofficial-out-root (a test of the command).", ""]
    ne = [m for m in doc["models"] if m.get("evaluable") is False]
    L += [f"Status: **{doc['status']}**. Attempt {doc['attempt']['attempt']}. {doc['positions']} positions of `run_matrix.csv`; phase 1 "
          f"(training, rule (n), checkpoint, dev scored with the scorer of the accepted runs) complete for {doc['complete_phase_1']}, phase 2 "
          f"(both BN states on the dev, BN re-estimated on the fitting set) for {doc['complete_phase_2']}; not evaluable under rule (n): "
          f"{len(ne)}. {TEST_SENTENCE[0].upper() + TEST_SENTENCE[1:]}: the frame files, hashed and loaded whole, also hold frames of the "
          "held-out subjects, of which only the fitting and dev frames of arm B are assembled; every number below is a dev or training "
          "number.", "",
          f"Started {doc['started']}, finished {doc['finished']}; wall time {doc['wall_seconds']:.0f} s of a cap of {doc['cap_hours']:.2f} "
          f"{doc['cap_unit']}; process CPU time (all threads) {doc['process_cpu_seconds']:.0f} s.", ""]
    if doc["stopped"]:
        L += [f"Stopped: {doc['stopped']['cause']} at {doc['stopped']['step']} ({doc['stopped']['elapsed_minutes']} min)"
              + (f"; {doc['stopped']['discarded']} {DISCARDED}" if doc["stopped"].get("discarded") else "") + ".", ""]
    m = doc["machine"]
    L += [f"Machine: {m['processor']}, cores as named by the system: {PAC.core_levels_text(m)}; {m['logical_cpus']} logical CPUs, "
          f"{PAC.ram_text(m)}, {m['os']}; equal to the official trial {doc['machine_rule']['trial_run']} in "
          f"{', '.join(R.MACHINE_FIELDS)} and in the versions {', '.join(R.VERSION_FIELDS)} (machine model and versions, not the physical "
          "unit).", "", PAC.threads_text(doc["threads"]), "",
          f"BN estimator: checked by {doc['estimator_check']['run_id']} ({doc['estimator_check']['models']} models of the BNRECAL run "
          "reproduced).", "", "## Models", "",
          "| position | fold | seed | status | epochs | restored epoch | non-finite history | ordinary: EER threshold | ordinary: one-class "
          "at 0.5 | re-estimated: EER threshold | re-estimated: one-class at 0.5 | train s |", "|" + "---|" * 12]
    for r in doc["models"]:
        c = r.get("card") or {}
        st = r.get("dev") or {}

        def thr(s):
            return "NA" if s not in st else R.fmt_thr(st[s])

        def oc(s):
            return "NA" if s not in st else R.one_class_text(st[s]["one_class_05"])
        nf = ", ".join(f"{k} from epoch {e}" for k, e in c["non_finite"]) if c.get("non_finite") else "none"
        status = r["model_status"] + (f" ({r['reason']})" if r.get("reason") else "")
        L.append(f"| {r['position']} | {r['fold']} | {r['seed']} | {status} | {c.get('epochs_run', 'NA')} | {c.get('restored_epoch', 'NA')} | "
                 f"{nf} | {thr('ordinary')} | {oc('ordinary')} | {thr('re-estimated')} | {oc('re-estimated')} | "
                 f"{PAC.fmt((r.get('seconds') or {}).get('train_seconds')) or 'NA'} |")
    L += ["", "Rule (n) of the pre-registration (PROTOCOL.md §6): a non-finite value in the history does not stop the run; without a finite "
          "val_loss, with non-finite restored weights, non-finite scores or logits, or non-finite re-estimated statistics the model is not "
          "evaluable and the run goes on.", "",
          "## One-class runs at threshold 0.5 on the dev (definition (a))", "", "| network | patience | BN state | count |", "|---|---|---|---|"]
    for o in doc["one_class"]:
        L.append(f"| {o['network']} | {o['patience']} | {o['bn_state']} | {o['text']} (all bona fide {o['all_bona_fide']}, all attack "
                 f"{o['all_attack']}, not evaluable {o['not_evaluable']}) |")
    if not doc["one_class"]:
        L.append("| none | | | no model has its BN states evaluated |")
    f = doc["dev_frames"]
    L += ["", f"Dev frame identifiers (public outputs hold only counts and SHA-256): {f['models_compared']} models compared over "
          f"{f['folds_compared']} folds, both BN states; every model of a fold used the same frames in the same order.", "",
          f"Command: `{doc['command']}`", ""]
    return "\n".join(L)


def final_checks(cfg: dict, cfg_p: Path, cfg_sha: str, base: Path, pins: Pins, refs: dict = None, est_folder: Path = None) -> dict:
    """Alla fine: configurazione, codice, configurazione della PR A, record della prova e SHA256SUMS del pacchetto, file
    del piano, configurazioni delle PR #27 e #28, schede di riferimento e record dello stimatore con gli stessi SHA-256
    dell'inizio."""
    if sha256_bytes(read_regular(cfg_p, "configuration")) != cfg_sha:
        fail("the configuration changed during the run")
    if check_code(cfg) != pins.code:
        fail("the code changed during the run")
    for e in (cfg["comparison"], cfg["trial_record"]["package_sums"], cfg["estimator_check"]["bn_recalibration_config"],
              cfg["estimator_check"]["bn_test_config"], *cfg["plan_files"].values()):
        read_pinned(base, e, "pinned file")
    folder = rel_path(base, cfg["trial_record"]["folder"])
    for fn, key in ((STATUS, "status_sha256"), (PAC.TRIAL_JSON, "trial_json_sha256")):
        if sha256_bytes(read_regular(folder / fn, f"trial record {fn}")) != cfg["trial_record"][key]:
            fail(f"the trial record {fn} changed during the run")
    if refs:
        for ref in refs.values():
            key = f"patience_{ref['card']['early_stopping_patience']}"
            run = rel_path(base, pins.cfg_pr["parent_runs"][key]["folder"])
            if sha256_bytes(read_regular(BR.parent_path(run, ref["model"], "card.json"), "reference card")) != ref["sha256"]:
                fail("a reference card changed during the run")
    if est_folder is not None:
        b = read_regular(est_folder / PUBLIC / "estimator_check.json", "estimator_check.json")
        if sha256_bytes(b) != pins.estimator["estimator_check_json_sha256"]:
            fail("the estimator check record changed during the run")
    return {"configuration_unchanged": True, "code_unchanged": len(pins.code), "pinned_inputs_unchanged": True,
            "reference_cards_unchanged": len(refs or {})}


# ============================================================================ cartella del run, tentativi, comando

def repo_state(root: Path) -> dict:
    return REPO_STATE(root) if REPO_STATE is not None else SR.repo_state(root)


def run_id_for(kind: str, date: str, state: dict, attempt: int, run: str = None) -> str:
    """Run id: YYYYMMDD-ANETRUN-<v1|v2_1|v2_2>-p<4|12>-<commit> (run) o YYYYMMDD-ANETEST-<commit> (check-estimator), con
    il suffisso -r<k> per il tentativo k > 1."""
    if not re.fullmatch(r"\d{8}", date or ""):
        fail("--date must be YYYYMMDD")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        fail("--attempt must be a positive integer")
    suffix = "" if attempt == 1 else f"-r{attempt}"
    rid = (f"{date}-ANETRUN-{RUN_TAG[run]}-{state['commit']}" if kind == "run" else f"{date}-ANETEST-{state['commit']}") + suffix
    if not (RUN_ID_RE if kind == "run" else EST_ID_RE).match(rid):
        fail(f"run id {rid} malformed")
    return rid


def previous_attempts(out_root: Path, kind: str, run: str = None) -> list:
    """Le cartelle dello stesso run nella radice d'uscita (qualunque data e commit), con tentativo e stato."""
    out = []
    if not out_root.is_dir():
        return out
    for e in sorted(os.listdir(out_root)):
        loose = (LOOKS_LIKE_RUN if kind == "run" else LOOKS_LIKE_EST).match(e)
        if not loose:
            continue
        if kind == "run":
            tag = RUN_TAG[run].rsplit("-p", 1)
            if (loose.group(1).lower(), loose.group(2).lower()) != (tag[0], tag[1]):
                continue
        m = (RUN_ID_RE if kind == "run" else EST_ID_RE).match(e)
        if not m:
            fail(f"output root: the entry {e!r} looks like a run id of this run but does not have the expected form "
                 "(YYYYMMDD-ANETRUN-<v1|v2_1|v2_2>-p<4|12>-<commit> or YYYYMMDD-ANETEST-<commit>, suffix -r<k> with k >= 2): rename or move it")
        g = m.group(3) if kind == "run" else m.group(1)
        k = int(g) if g else 1
        try:
            st = json.loads((out_root / e / STATUS).read_text(encoding="utf-8")).get("status", "unreadable")
        except (OSError, ValueError):
            st = "unreadable"
        out.append({"run_id": e, "attempt": k, "status": st})
    return out


def attempt_rule(out_root: Path, kind: str, run: str, attempt) -> dict:
    """La regola dei tentativi: --attempt obbligatorio, uguale all'ultimo tentativo dello stesso run + 1."""
    if attempt is None:
        fail("--attempt N is required (1 for the first attempt; every attempt keeps its outcome)")
    prev = previous_attempts(out_root, kind, run)
    need = max((p["attempt"] for p in prev), default=0) + 1
    if attempt != need:
        fail(f"--attempt {attempt} refused: {len(prev)} earlier attempts of this run in the output root "
             f"({', '.join(p['run_id'] + ' ' + p['status'] for p in prev) or 'none'}); this attempt must be --attempt {need}")
    return {"attempt": attempt, "previous": prev}


def folder_arg(p) -> Path:
    return Path(p) if Path(p).is_absolute() else ROOT / p


def resolve(a) -> tuple:
    base = Path(a.input_root).resolve() if a.input_root else ROOT
    cfg_p = Path(a.config) if Path(a.config).is_absolute() else ROOT / a.config
    return base, cfg_p


def command_text(a) -> str:
    """Il comando registrato, dai valori interpretati da argparse, con i percorsi scritti con pad_arch_run.path_text."""
    parts = [a.cmd, "--config", R.path_text(folder_arg(a.config))]
    if a.input_root is not None:
        parts += ["--input-root", R.path_text(Path(a.input_root).resolve())]
    parts += ["--out-root", R.path_text(folder_arg(a.out_root))]
    if a.unofficial_out_root:
        parts.append("--unofficial-out-root")
    if a.run is not None:
        parts += ["--run", a.run]
    if a.estimator_check is not None:
        parts += ["--estimator-check", R.path_text(folder_arg(a.estimator_check))]
    if a.date is not None:
        parts += ["--date", a.date]
    if a.attempt is not None:
        parts += ["--attempt", str(a.attempt)]
    if a.i_am_authorized:
        parts.append("--i-am-authorized")
    return "python -I -B scripts/pad_anet_run.py " + " ".join(shlex.quote(x) for x in parts)


def input_paths(cfg: dict, pins: Pins, base: Path, cfg_p: Path, est_base: Path = None) -> dict:
    out = C.input_paths(pins.cfg_c5, pins.cfg_pr, pins.pcfg, base, rel_path(base, cfg["comparison"]["path"]))
    out["the configuration of this runner"] = cfg_p
    out["the trial record"] = rel_path(base, cfg["trial_record"]["folder"])
    out["the PR A package SHA256SUMS"] = rel_path(base, cfg["trial_record"]["package_sums"]["path"])
    for k in PLAN_KEYS:
        out[f"the plan file {k}"] = rel_path(base, cfg["plan_files"][k]["path"])
    out["the PR #28 configuration"] = rel_path(base, cfg["estimator_check"]["bn_test_config"]["path"])
    if est_base is not None:
        out["the BNRECAL run"] = rel_path(est_base, pins.bt_cfg["bnrecal_run"]["folder"])
        out["the parent run of PR #27"] = rel_path(est_base, pins.bn_cfg["parent_run"]["folder"])
    return out


def run_dir_problem(run: Path, out_root: Path, inputs: dict) -> str:
    """Motivo per cui la cartella del run non è ammessa ("" se lo è): pad_arch_run.run_dir_problem con la radice ufficiale
    results/anetrun."""
    if os.path.lexists(run):
        return f"the run folder {run.name} already exists: every attempt keeps its outcome"
    if out_root.is_symlink():
        return "the output root is a symbolic link"
    for label, p in inputs.items():
        if pg.within(p, run) or (Path(p).is_dir() and pg.within(run, p)):
            return f"{label} {Path(p).name} is inside the run folder, or contains it: inputs and outputs must be separate"
    return mw.output_problem(run, ROOT, pg.BASE_PROTECTED, [ROOT / DEFAULT_OUT_ROOT], [ROOT / x for x in EXTRA_PROTECTED], inputs,
                             [STATUS, LOG], [run / PUBLIC, run / RESTRICTED])


def official_root(a) -> bool:
    out_root, official, unofficial = folder_arg(a.out_root), ROOT / DEFAULT_OUT_ROOT, a.unofficial_out_root
    if pg.within(out_root, official) and pg.within(official, out_root):
        if unofficial:
            fail(f"--unofficial-out-root with the official output root {DEFAULT_OUT_ROOT}: refused")
        return True
    if pg.within(out_root, ROOT) or pg.within(ROOT, out_root):
        fail(f"an output root other than {DEFAULT_OUT_ROOT} inside the repository (or containing it) is refused")
    if not unofficial:
        fail(f"--out-root other than {DEFAULT_OUT_ROOT} needs --unofficial-out-root: the run is then recorded as not official")
    return False


def load_estimator_record(folder: Path, cfg_sha: str, code: dict, official: bool) -> dict:
    """Il record di un controllo dello stimatore (pad_arch_run.load_estimator_record con lo schema e il run id di questo
    runner): PASS, stessa configurazione e stesso codice, ufficiale se il run è ufficiale, tutti i 36 modelli passati."""
    st = SR.read_json(read_regular(folder / STATUS, "estimator check status.json"), "estimator check status.json")
    if st.get("status") != "PASS" or not isinstance(st.get("run_id"), str) or not EST_ID_RE.match(st["run_id"]):
        fail("estimator check: status.json is not a PASS of check-estimator (the statistics of the BNRECAL run must be reproduced first)")
    b = read_regular(folder / PUBLIC / "estimator_check.json", "estimator_check.json")
    if sha256_bytes(b) != st.get("estimator_check_json_sha256"):
        fail("estimator check: estimator_check.json differs from the SHA-256 in its status.json")
    doc = SR.read_json(b, "estimator_check.json")
    if doc.get("schema") != ESTIMATOR_SCHEMA or doc.get("status") != "PASS" or doc.get("run_id") != st["run_id"]:
        fail("estimator check: schema, status or run id of estimator_check.json differ")
    if doc.get("config_sha256") != cfg_sha:
        fail("estimator check: produced with another configuration of this runner")
    if doc.get("code_sha256") != code:
        fail("estimator check: produced with other code (code_sha256 differs from the current files)")
    if official and doc.get("official") is not True:
        fail(f"estimator check: an official run needs an official check-estimator (in {DEFAULT_OUT_ROOT})")
    rows = doc.get("models")
    if not isinstance(rows, list) or not rows or len(rows) != doc.get("expected_models") or any(r.get("recalibration_pass") is not True for r in rows):
        fail("estimator check: not every model of the BNRECAL run was checked and passed")
    if (doc.get("machine_rule") or {}).get("equal") is not True or (doc.get("threads_rule") or {}).get("equal") is not True:
        fail("estimator check: the machine and thread rule was not met")
    return {"run_id": st["run_id"], "estimator_check_json_sha256": st["estimator_check_json_sha256"], "models": len(rows),
            "official": doc.get("official") is True}


def prepare(a, kind: str) -> tuple:
    """I controlli comuni prima di creare una cartella (pad_arch_run.prepare): configurazione, agganci (tutti None nei
    comandi ufficiali), file fissati, variabili dei thread e regola della macchina contro la prova C5, radice d'uscita,
    record dello stimatore (run), albero pulito, regola dei tentativi, run id e cartella nuova ammessa."""
    base, cfg_p = resolve(a)
    cfg, cfg_sha = load_config(cfg_p)
    official = official_root(a)
    if a.date is not None and official:
        fail("--date is accepted only with --unofficial-out-root (tests of the command): an official run id carries the date of the launch")
    hooks = check_hooks(official or is_official_config(cfg_p))
    est_base = (ESTIMATOR_BASE or base) if kind == "estimator" else None
    command = command_text(a)
    if probs := PAC.output_problems(command, FR.Inputs()):
        fail("the recorded command holds text that a publishable output must not hold: " + "; ".join(probs))
    pins = load_pins(cfg, base, est_base)
    env, mrule = machine_rules(pins)
    pins.estimator = None
    est_folder = None
    if kind == "run":
        if not a.estimator_check:
            fail("run needs --estimator-check: the folder of a PASS of check-estimator (PROTOCOL.md §3: the statistics of the BNRECAL run "
                 "reproduced before the runs)")
        est_folder = folder_arg(a.estimator_check)
        pins.estimator = load_estimator_record(est_folder, cfg_sha, pins.code, official)
    state = repo_state(ROOT)
    if state["dirty"]:
        fail("the git work tree is not clean: the runs need a committed state (commit the changes first; results/ is versioned)")
    out_root = folder_arg(a.out_root)
    attempt = attempt_rule(out_root, kind, a.run if kind == "run" else None, a.attempt)
    rid = run_id_for(kind, a.date or time.strftime("%Y%m%d"), state, a.attempt, a.run if kind == "run" else None)
    run = out_root / rid
    inputs = input_paths(cfg, pins, base, cfg_p, est_base)
    if est_folder is not None:
        inputs["the estimator check"] = est_folder
    if msg := run_dir_problem(run, out_root, inputs):
        fail(f"refusing the run folder: {msg}")
    return base, cfg_p, cfg, cfg_sha, pins, env, mrule, state, rid, run, est_folder, official, attempt, command, hooks


# ============================================================================ comandi

def cmd_check_machine(a) -> int:
    base, cfg_p = resolve(a)
    cfg, cfg_sha = load_config(cfg_p)
    check_hooks(is_official_config(cfg_p))
    pins = load_pins(cfg, base)
    env, mrule = machine_rules(pins)
    tfs = FR.tf_state()
    n = len(R.MACHINE_FIELDS) + len(R.VERSION_FIELDS)
    print(f"check-machine: machine and versions equal to the C5 trial {pins.trial['run_id']} ({n} fields); thread environment variables "
          f"{R.env_text(env)}, as in the trial; TensorFlow {tfs}; nothing written")
    print("RESULT: PASS (check-machine)")
    return 0


def cmd_check_inputs(a) -> int:
    base, cfg_p = resolve(a)
    cfg, cfg_sha = load_config(cfg_p)
    hooks = check_hooks(is_official_config(cfg_p))
    pins = load_pins(cfg, base)
    env, mrule = machine_rules(pins)
    est = None
    if a.estimator_check:
        est = load_estimator_record(folder_arg(a.estimator_check), cfg_sha, pins.code, False)
    inp = C.load_inputs(pins.cfg_c5, pins.cfg_pr, pins.pcfg, base, frames=True)
    refs = {p: R.load_reference_cards(pins.cfg_pr, base, p) for p in DEF.PATIENCES}
    units = {p: R.check_unit_keys(inp, refs[p], pins.pcfg) for p in refs}
    PAC.assert_no_test(inp)
    tfs = FR.tf_state()
    print(f"CHECK-INPUTS: configuration {cfg_sha[:12]}; PR A configuration {pins.sha_c5[:12]}; code pinned ({len(CODE_FILES)} files); "
          f"hooks {'none set' if not hooks['set'] and not hooks['changed'] else ', '.join(hooks['set'] + hooks['changed'])}; trial "
          f"{pins.trial['run_id']} PASS, accepted by pad_anet_comparison.load_trial and equal to the PR A package; run_matrix.csv, "
          "run_order.csv and run_caps.csv equal to the files regenerated from the PR A configuration, the C2 evaluation and the trial; "
          + "; ".join(f"{r}: {len(pins.positions[r])} positions, cap {pins.caps[r]:.2f} h" for r in RUNS)
          + f"; machine model and versions equal to the C5 trial ({', '.join(R.MACHINE_FIELDS + R.VERSION_FIELDS)}); environment variables "
          + R.env_text(env)
          + f"; inputs of the pre-registration PASS; {sum(len(v) for v in refs.values())} reference cards with keys equal to the manifests "
          f"({sum(len(v) for v in units.values())} unit checks); estimator check "
          + (f"{est['run_id']} PASS ({est['models']} models, {'official' if est['official'] else 'not official'})" if est else "not given")
          + f"; {TEST_SENTENCE}; no frame assembled; TensorFlow {tfs}; nothing written")
    print("RESULT: PASS (check-inputs)")
    return 0


def cmd_check_estimator(a) -> int:
    """pad_arch_run.cmd_check_estimator con le sole sostituzioni di configurazione (questa), schema e run id
    (pad-anet-estimator-check/1, ANETEST), regola della macchina contro la prova C5 e agganci registrati."""
    if not a.i_am_authorized:
        fail("check-estimator loads the 36 accepted checkpoints and the restricted NUAA frames: pass --i-am-authorized")
    base, cfg_p, cfg, cfg_sha, pins, env, mrule, state, rid, run, _, official, attempt, command, hooks = prepare(a, "estimator")
    est_base = ESTIMATOR_BASE or base
    run.mkdir(parents=True)
    head = {"run_id": rid, "commit": state["commit"], "dirty": False, "config_sha256": cfg_sha, "command": "check-estimator",
            "command_line": command, "official": official, "attempt": attempt, "hooks": hooks}
    scan = FR.Inputs()
    ec = cfg["estimator_check"]
    budget = SR.Budget(ec["cap_minutes"])
    SR.signal_handlers(budget)
    started, c0 = now(), PAC.cpu_seconds()
    log, saved_warning = None, warnings.showwarning
    threads = {"requested": dict(pins.trial["threads"]["requested"]), "effective": None, "set_by_this_runner": False, "environment": env,
               "logical_cpus": os.cpu_count()}
    extra = {"threads": threads, "machine_rule": mrule, "cap_minutes": ec["cap_minutes"], "cap_unit": ec["cap_unit"]}
    rows, status, stopped = [], "PASS", None
    try:
        R.write_status(run, {**head, "status": "RUNNING", "passed": False, "started": started, "pid": os.getpid(), **extra,
                             "note": "RUNNING without a live process means a failed run; SIGUSR1 stops it as INCOMPLETE"}, scan)
        log = FR.open_log(run)
        warnings.showwarning = PAC.show_warning
        print(f"== estimator check {rid}: configuration {cfg_sha[:12]}, cap {ec['cap_minutes']} {ec['cap_unit']}, pid {os.getpid()}", flush=True)
        bn_cfg, bt_cfg = pins.bn_cfg, pins.bt_cfg
        inp = BR.load_inputs(bn_cfg, est_base)
        scan = inp
        inp.bn_cfg, inp.bnrecal = bn_cfg, {}
        BT.load_bnrecal(bt_cfg, est_base, inp)
        if len(inp.order) != ec["models"]:
            fail(f"the parent run holds {len(inp.order)} models, the configuration expects {ec['models']}")
        store = FR.FrameStore(inp, inp.X_extracted, inp.extracted_keys)
        tf = R.tf_module()
        threads = PAC.set_threads(tf, pins.cfg_pr)
        extra["threads"] = threads
        extra["threads_rule"] = R.threads_rule(pins.trial, threads)
        R.require_threads(extra["threads_rule"])
        FR.hide_gpu(tf)
        order, batch = cfg["estimator"]["orders"]["LivenessNet"], cfg["estimator"]["batch"]
        try:
            for name in inp.order:
                budget.check(f"estimator {name}")
                m = inp.parent[name]
                fit, dev = BR.frames(store, inp, "fitting", m["arm"], m["fold"]), BR.frames(store, inp, "dev", m["arm"], m["fold"])
                BR.tf_module().keras.backend.clear_session()
                orig = BR.load_model(m["weights"], m["weights_sha256"])
                BR.check_architecture(orig, bn_cfg, name)
                copy = tf.keras.models.clone_model(orig)
                copy.set_weights(orig.get_weights())
                snap = BR.snapshot_weights(orig)
                if BR.snapshot_weights(copy) != snap:
                    fail(f"{name}: the in-memory copy differs from the original before the re-estimation")
                stats = R.reestimate(copy, fit, order, batch, budget, name)
                budget.check(f"consistency pass {name}")
                fit_after, final = BR.evaluate(copy, fit, batch, f"{name} fitting after", layers=order)
                budget.check(f"dev after {name}")
                dev_after, _ = BR.evaluate(copy, dev, batch, f"{name} dev after")
                cons = R.pr27_checks(name, stats, snap, copy, order, fit_after, final, dev_after, bn_cfg)
                check = BT.recal_check(bt_cfg, stats, inp.bnrecal[name]["stats"], dev_after["p"][:, 1], inp.bnrecal[name]["dev_after"])
                rows.append({"arm": m["arm"], "fold": m["fold"], "seed": m["seed"], "group": inp.groups[name], **check,
                             "consistency_channels_outside": sum(c["channels_outside_tolerance"] for c in cons.values())})
                print(f"   {name}: {'reproduced' if check['recalibration_pass'] else 'MISMATCH'} (statistics "
                      f"{'identical' if check['statistics_identical'] else 'max diff ' + format(max(check['statistics_max_abs_diff_mean'], check['statistics_max_abs_diff_variance']), '.1e')})",
                      flush=True)
                del orig, copy, fit, dev, final
                gc.collect()
        except SR.CapExceeded as e:
            status = "INCOMPLETE"
            stopped = {"step": e.step, "cause": e.cause, "elapsed_minutes": round(e.minutes, 1)}
        status = R.estimator_status(status, rows)
        if inp.folds:
            fail("the fold schema was loaded: this command must never open the test manifests")
        if sha256_bytes(read_regular(cfg_p, "configuration")) != cfg_sha or check_code(cfg) != pins.code:
            fail("the configuration or the code changed during the check")
        wall, cpu_total, finished = budget.elapsed(), round(PAC.cpu_seconds() - c0, 1), now()
        doc = {"schema": ESTIMATOR_SCHEMA, **head, "status": status, "started": started, "finished": finished, "wall_seconds": round(wall, 1),
               "process_cpu_seconds": cpu_total, "stopped": stopped, "bnrecal_run": ec["bnrecal_run_id"], "expected_models": ec["models"],
               "rule": ec["rule"], "tolerances": {k: BT.tol(bt_cfg, k) for k in ("recalibration_statistics_rel", "recalibration_statistics_abs",
                                                                                 "recalibration_dev_probability_max_abs_diff",
                                                                                 "recalibration_dev_decisions_discordant_at_0_5")},
               "order": order, **extra, "machine": (R.MACHINE or PAC.machine)(), "environment": (R.ENVIRONMENT or FR.environment)(),
               "code_sha256": pins.code, "models": rows, "test_read": TEST_READ}
        texts = {"estimator_check.csv": csv_text(EST_COLUMNS, [[BR.cell_text(r[k]) for k in EST_COLUMNS] for r in rows]),
                 "estimator_check.json": PAC.json_text(doc, "estimator_check.json")}
        for n, text in texts.items():
            if probs := PAC.output_problems(text, inp):
                fail(f"{n} refused: " + "; ".join(probs))
        mw.write_all(run / PUBLIC, texts, newline="")
        print(f"RESULT: {status} (check-estimator)")
        warnings.showwarning = saved_warning
        FR.close_log(log)
        log = None
        R.write_status(run, {**head, "status": status, "passed": status == "PASS", "started": started, "finished": finished,
                             "wall_seconds": round(wall, 1), "process_cpu_seconds": cpu_total, "stopped": stopped, **extra,
                             "models_checked": len(rows), "models_reproduced": sum(r["recalibration_pass"] for r in rows),
                             "estimator_check_json_sha256": sha256_bytes(texts["estimator_check.json"].encode("utf-8")),
                             "estimator_check_csv_sha256": sha256_bytes(texts["estimator_check.csv"].encode("utf-8")),
                             "run_log_sha256": sha256_file(run / LOG)}, scan)
    except BaseException as e:
        warnings.showwarning = saved_warning
        R.fail_cleanup(run, list(EST_OUTPUTS), log, head, started, budget, c0, extra, e, scan)
        raise
    return STATUS_CODE[status]


def cmd_run(a) -> int:
    """Un run del piano (pad_arch_run.cmd_run per le AttackNet): fase 1 per le 12 posizioni, poi fase 2 per i modelli
    valutabili; i modelli non valutabili per la regola (n) restano registrati e non cambiano l'esito del run."""
    if not a.i_am_authorized:
        fail("run trains 12 networks on the restricted NUAA frames: pass --i-am-authorized (only after the publication of this command and "
             "configuration and the link on the C5 card)")
    if a.run not in RUNS:
        fail(f"--run must be one of {', '.join(RUNS)}")
    base, cfg_p, cfg, cfg_sha, pins, env, mrule, state, rid, run, est_folder, official, attempt, command, hooks = prepare(a, "run")
    net, patience = RUNS[a.run]
    positions, cap_h = pins.positions[a.run], pins.caps[a.run]
    run.mkdir(parents=True)
    scan = FR.Inputs()
    head = {"run_id": rid, "run": a.run, "network": net, "commit": state["commit"], "dirty": False, "config_sha256": cfg_sha, "command": "run",
            "command_line": command, "official": official, "attempt": attempt, "hooks": hooks}
    budget = SR.Budget(cap_h * 60)
    SR.signal_handlers(budget)
    started, c0 = now(), PAC.cpu_seconds()
    log, saved_warning = None, warnings.showwarning
    threads = {"requested": dict(pins.trial["threads"]["requested"]), "effective": None, "set_by_this_runner": False, "environment": env,
               "logical_cpus": os.cpu_count()}
    recs = [{"position": p["position"], "network": p["network"], "fold": p["fold"], "seed": p["seed"],
             "name": PAC.training_name(p["network"], p["fold"], p["seed"])} for p in positions]
    ctx = Ctx()
    ctx.current = None
    extra = {"threads": threads, "machine_rule": mrule, "estimator_check": pins.estimator, "cap_hours": cap_h, "cap_unit": CAP_UNIT}
    try:
        R.write_status(run, {**head, "status": "RUNNING", "passed": False, "started": started, "pid": os.getpid(), **extra,
                             "note": "RUNNING without a live process means a failed run; SIGUSR1 to this pid stops it as INCOMPLETE"}, scan)
        log = FR.open_log(run)
        warnings.showwarning = PAC.show_warning
        print(f"== NUAA run {a.run} ({rid}, attempt {attempt['attempt']}{'' if official else ', NOT OFFICIAL'}): configuration {cfg_sha[:12]}, "
              f"cap {cap_h:.2f} {CAP_UNIT}, pid {os.getpid()}", flush=True)
        inp = C.load_inputs(pins.cfg_c5, pins.cfg_pr, pins.pcfg, base, frames=True)
        scan = inp
        refs = R.load_reference_cards(pins.cfg_pr, base, patience)
        R.check_unit_keys(inp, refs, pins.pcfg)
        store = FR.FrameStore(inp, inp.X_extracted, inp.extracted_keys)
        tf = R.tf_module()
        threads = PAC.set_threads(tf, pins.cfg_pr)            # subito dopo l'import, prima di qualunque operazione
        extra["threads"] = threads
        extra["threads_rule"] = R.threads_rule(pins.trial, threads)
        R.require_threads(extra["threads_rule"])
        print(f"   {PAC.threads_text(threads)}", flush=True)
        FR.hide_gpu(tf)
        ctx.cfg, ctx.cfg_c5, ctx.cfg_pr, ctx.pcfg, ctx.bn_cfg, ctx.inp, ctx.store, ctx.budget, ctx.tf, ctx.refs = (
            cfg, pins.cfg_c5, pins.cfg_pr, pins.pcfg, pins.bn_cfg, inp, store, budget, tf, refs)
        ctx.run_name, ctx.rdir = a.run, run / RESTRICTED
        ctx.rdir.mkdir()
        ctx.training = {**pins.pcfg["training"], "early_stopping_patience": patience}
        ctx.arch = (ARCH_LOADER or FR.default_arch_loader)(rel_path(base, pins.cfg_c5["architectures"]["path"]))
        status, stopped = "PASS", None
        try:
            for pos, rec in zip(positions, recs):
                ctx.current = rec
                phase_one(ctx, pos, rec)
            ctx.current = None
            for rec in recs:
                if rec.get("evaluable"):
                    ctx.current = rec
                    phase_two(ctx, rec)
            ctx.current = None
        except SR.CapExceeded as e:
            status = "INCOMPLETE"
            stopped = {"step": e.step, "cause": e.cause, "elapsed_minutes": round(e.minutes, 1)}
            if ctx.current is not None and not ctx.current.get("phase_1"):
                stopped["discarded"] = ctx.current["name"]
                R.discard_model(ctx.rdir, ctx.current["name"])
                R.mark_discarded(ctx.current)
            ctx.current = None
            print(f"== {'CAP' if e.cause == 'cap' else 'STOP'}: {e}; the complete models are kept and reported", flush=True)
        PAC.assert_no_test(inp)
        frames = check_dev_frames([r for r in recs if r.get("phase_2")])
        checks = final_checks(cfg, cfg_p, cfg_sha, base, pins, refs, est_folder)
        mismatches = [{"name": r["name"], **r["evaluator_agreement"]} for r in recs if r.get("phase_2") and not r["evaluator_agreement"]["pass"]]
        if mismatches:
            status = "MISMATCH"
        wall, cpu_total, finished = budget.elapsed(), round(PAC.cpu_seconds() - c0, 1), now()
        done1, done2 = [r for r in recs if r.get("phase_1")], [r for r in recs if r.get("phase_2")]
        ne = [{"name": r["name"], "reason": r["reason"]} for r in recs if r.get("evaluable") is False]
        res_sha = R.write_restricted_sums(ctx.rdir)
        pub_sha = None
        if status != "MISMATCH":
            doc = {"schema": RUN_SCHEMA, **head, "status": status, "patience": patience, "started": started, "finished": finished,
                   "wall_seconds": round(wall, 1), "process_cpu_seconds": cpu_total, "cap_hours": cap_h, "cap_unit": CAP_UNIT, "stopped": stopped,
                   "positions": len(recs), "complete_phase_1": len(done1), "complete_phase_2": len(done2), "not_evaluable": ne,
                   "machine": (R.MACHINE or PAC.machine)(), "environment": (R.ENVIRONMENT or FR.environment)(), "machine_rule": mrule,
                   "threads": threads, "threads_rule": extra["threads_rule"], "estimator_check": pins.estimator, "training": ctx.training,
                   "sequence": cfg["sequence"], "non_evaluable": cfg["non_evaluable"], "after_stop": cfg["after_stop"],
                   "models": [public_model(r) for r in recs], "one_class": one_class_rows(done2, patience), "dev_frames": frames,
                   "checks": checks, "code_sha256": pins.code, "inputs_sha256": {**pins.files, "configuration": cfg_sha},
                   "peak_rss_mb": round(SR.peak_rss_mb(), 1), "test_read": TEST_READ, "command": command}
            texts = {"models.csv": csv_text(MODEL_COLUMNS, [model_row(r, a.run, patience) for r in recs]),
                     "one_class.csv": csv_text(ONE_CLASS_COLUMNS, [[R.cell(o[k]) for k in ONE_CLASS_COLUMNS] for o in doc["one_class"]]),
                     "summary.json": PAC.json_text(doc, "summary.json"), "report.md": report_md(doc)}
            pub_sha = R.write_public(run, inp, texts)
        print(f"run folder: {run.name}, wall {wall:.0f} s ({wall / 3600:.2f} h of the {cap_h:.2f}-hour cap), process CPU {cpu_total:.0f} s; "
              f"not evaluable {len(ne)}")
        if mismatches:
            print(f"== MISMATCH: the scorer of phase 1 and the evaluator of phase 2 disagree on the ordinary dev for {len(mismatches)} models; "
                  "no public result is written", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        log[0].flush()
        if probs := PAC.output_problems((run / LOG).read_text(encoding="utf-8", errors="replace"), inp):
            fail(f"{LOG} holds text that a publishable output must not hold: " + "; ".join(probs))
        print(f"RESULT: {status} (run {a.run})")                # dopo il controllo del run.log: nessun «RESULT: PASS» prima di un FAIL
        warnings.showwarning = saved_warning
        FR.close_log(log)
        log = None
        R.write_status(run, {**head, "status": status, "passed": status == "PASS", "started": started, "finished": finished,
                             "wall_seconds": round(wall, 1), "process_cpu_seconds": cpu_total, **extra, "stopped": stopped,
                             "complete_phase_1": [r["name"] for r in done1], "complete_phase_2": [r["name"] for r in done2],
                             "missing_phase_1": [r["name"] for r in recs if not r.get("phase_1")], "not_evaluable": ne,
                             "evaluator_mismatches": mismatches, "dev_frames": frames,
                             "machine": (R.MACHINE or PAC.machine)(), "code_sha256": pins.code, "inputs_sha256": {**pins.files, "configuration": cfg_sha},
                             "public_sha256sums_sha256": pub_sha, "restricted_sha256sums_sha256": res_sha, "run_log_sha256": sha256_file(run / LOG)}, scan)
    except BaseException as e:
        warnings.showwarning = saved_warning
        cur = ctx.current
        if cur is not None and not cur.get("phase_1") and (run / RESTRICTED).is_dir():
            R.discard_model(run / RESTRICTED, cur["name"])
        extra["complete_phase_1"] = [r["name"] for r in recs if r.get("phase_1")]
        extra["not_evaluable"] = [{"name": r["name"], "reason": r["reason"]} for r in recs if r.get("evaluable") is False]
        extra["restricted"] = R.restricted_note(run / RESTRICTED)
        R.fail_cleanup(run, list(PUBLIC_OUTPUTS), log, head, started, budget, c0, extra, e, scan)
        raise
    return STATUS_CODE[status]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="The six NUAA runs of the C5 comparison (72 AttackNet trainings, fitting and dev only, CPU), "
                                             "the check of the BN estimator, the check of the inputs and of the machine")
    ap.add_argument("cmd", choices=["check-machine", "check-inputs", "check-estimator", "run"])
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="configuration JSON (relative to the repository root)")
    ap.add_argument("--input-root", default=None, help="base folder of the input paths of the configuration (default: repository root)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help=f"folder that receives <run-id>/ (official: {DEFAULT_OUT_ROOT}; another folder "
                                                                 "outside the repository only with --unofficial-out-root; never a symbolic link)")
    ap.add_argument("--unofficial-out-root", action="store_true", help=f"accept an --out-root other than {DEFAULT_OUT_ROOT}; the run is "
                                                                      "recorded as not official (tests of the command)")
    ap.add_argument("--run", default=None, choices=list(RUNS), help="run: the run of the plan")
    ap.add_argument("--estimator-check", default=None, help="run (required) and check-inputs (optional): the folder of a PASS of check-estimator")
    ap.add_argument("--date", default=None, help="YYYYMMDD of the run id, only with --unofficial-out-root (default and official: today)")
    ap.add_argument("--attempt", type=int, default=None, help="check-estimator and run (required): the last attempt of the same run in the "
                                                              "output root + 1 (1 if none)")
    ap.add_argument("--i-am-authorized", action="store_true", help="required by check-estimator and run")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return {"check-machine": cmd_check_machine, "check-inputs": cmd_check_inputs, "check-estimator": cmd_check_estimator,
                "run": cmd_run}[a.cmd](a)
    except Failure as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(f"RESULT: FAIL ({a.cmd})")
        return 1
    except KeyboardInterrupt:
        print(f"RESULT: FAIL ({a.cmd}, interrupted)")
        return 130
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: unexpected error {type(e).__name__}: {e}", file=sys.stderr)
        print(f"RESULT: FAIL ({a.cmd})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
