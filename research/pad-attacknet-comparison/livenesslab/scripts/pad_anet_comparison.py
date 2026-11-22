"""
Confronto C5 fra le cinque reti PAD su NUAA (scheda C5 del coordinatore, 21/11/2026): LivenessNet e le tre AttackNet
(AttackNetV1, AttackNetV2_1, AttackNetV2_2) con ResNet-18 come riferimento esterno. In questa consegna (primo passo
della scheda): configurazione, matrice dei 72 addestramenti nuovi e delle 48 celle riusate di C2 (plan, check-plan),
controllo degli ingressi (check-inputs) e la prova tecnica breve di ciascuna AttackNet sul solo fitting/dev (trial),
con la verifica della ristima di tutte le BN. Nessun frame di test è assemblato o valutato e nessun punteggio o metrica
del test nuovo è letto o calcolato; dei risultati di C2, già pubblicati, si leggono solo i file pubblici della
valutazione (TEST_READ dice esattamente che cosa legge ciascun comando).

Comandi:
  check-inputs   sola lettura, TensorFlow non importato, nulla scritto: configurazione a insieme chiuso; la
                 configurazione della preregistrazione C2 (SHA-256 fissato, validata da pad_arch_comparison.load_config) e
                 quella della PR #27 (SHA-256 fissato, validata da nuaa_bn_recalibration.load_config); codice fissato;
                 architetture e conteggi del registro (pad_anet_models); il record della prova C2 (status.json e
                 trial.json con gli SHA-256 fissati, validati da pad_arch_comparison.read_trial_record); i file pubblici
                 della valutazione C2 (status.json e results.json con gli SHA-256 fissati: run, ancora del SHA256SUMS
                 riservato, i 48 checkpoint); pad_arch_comparison.load_inputs (manifest, cache, frame estratti, scheda
                 di riferimento, chiavi del fitting e del dev della prova); thread dell'ambiente.
  check-machine  sola lettura, TensorFlow non importato, nulla scritto, nessun dato: macchina e versioni di adesso contro
                 il record della prova C2 (machine_comparison), da eseguire subito prima della prova (PASS solo se uguali).
  trial          la prova tecnica (vedi sotto), da un albero git pulito, con --i-am-authorized e --checkpoint-root fuori
                 dal repository.
  plan           plan/run_matrix.csv (72 righe), reference_cells.csv (48), run_order.csv, run_plan.csv, run_plan.md,
                 run_caps.csv dalla configurazione, dai file pubblici della valutazione C2 e, se dato, dal trial.json
                 di una prova PASS di questo codice (--trial-run); senza prova le ore e i tetti sono TRIAL_PENDING. Mai
                 sopra file esistenti.
  check-plan     sola lettura: rigenera in memoria i sei file e li confronta byte per byte con quelli di --plan-dir (con
                 lo stesso --trial-run, o senza); controlla conteggi e chiavi della matrice (check_matrix) e delle celle
                 (check_reference_cells).

Prova tecnica (un processo, in quest'ordine): agganci dei test tutti None con la configurazione consegnata, riconosciuta
per identità del file (check_hooks, is_official); variabili d'ambiente dei thread; albero git pulito; regola dei
tentativi (attempt_rule: --attempt = tentativi precedenti + 1, run id con -r<N> dal secondo; i precedenti con il loro
stato registrati); configurazione e ingressi (check-inputs); fitting e dev di NUAA,
braccio B, fold 1 (solo questi due ruoli, pad_arch_comparison.trial_frames) con le chiavi uguali alla scheda accettata;
TensorFlow con 10 thread intra-op e 10 inter-op (pad_arch_comparison.set_threads), GPU nascosta; poi
  1. LivenessNet, controllo della macchina, non contato fra le tre prove: 3 epoche con anet_trainer (copia dichiarata
     di pad_arch_comparison.trial_trainer, ANET_TRAINER_SUBSTITUTIONS); le cinque grandezze delle epoche 1-3 identiche
     (scarto 0) alla scheda accettata (pad_arch_comparison.baseline_check) e alla prova C2 20261119-ARCHTRIAL-b8f7cca
     (pad_arch_comparison.history_differences), e macchina e versioni uguali a quelle della prova C2 (machine_comparison,
     la regola di pad_arch_run.machine_rule, registrata campo per campo); altrimenti esito CONTROL_FAILED;
  2. per AttackNetV1, AttackNetV2_1, AttackNetV2_2, seed 42, 3 epoche, patience 4, con anet_trainer e il costruttore
     del registro; regola (n) di pad_anet_definitions (valore non finito registrato; modello non valutabile senza
     val_loss finita, con pesi non finiti, con punteggi o logit non finiti sul fitting o sul dev in uno dei due stati, o
     con statistiche ristimate non finite); passate d'inferenza cronometrate su fitting e dev;
     salvataggio del .keras fuori dal repository (pad_arch_run.save_checkpoint) e SHA-256; ricarica
     (nuaa_bn_recalibration.load_keras, SHA verificato subito prima), controllo con il registro, pesi uguali sui byte a
     quelli in memoria, punteggi del dev del ricaricato contro quelli in memoria (pad_arch_run.evaluator_agreement);
     scorer dei run accettati (nuaa_folds_run.tf_scorer) contro il valutatore (stato ordinario); fase 2 dei run sul
     ricaricato: dev ordinario (nuaa_bn_recalibration.evaluate), copia in memoria uguale sui byte, ristima
     (pad_arch_run.reestimate, ordine controllato sul grafo), originale invariato, passaggio di coerenza e dev
     ristimato, controlli della PR #27 (pad_arch_run.pr27_checks); copia del percorso del test con le statistiche
     installate (nuaa_bn_test.install), uguale sui byte alla copia ristimata e con gli stessi punteggi del dev;
  3. il controllo «ristima di tutte le BN» (bn_check: cinque controlli, uno fallito = FAIL), i tempi e le ore previste
     (projection, la stessa regola di plan).
Esiti: PASS (0), FAIL (1: trial.json e trial.md tolti; restano status.json con la ragione e run.log), INCOMPLETE (2:
tetto di tempo reale o SIGUSR1), CONTROL_FAILED (3: storia di LivenessNet non riprodotta; uscite scritte),
NOT_EVALUABLE (5: un'AttackNet non valutabile per la regola (n); uscite scritte). Precedenza INCOMPLETE >
NOT_EVALUABLE > CONTROL_FAILED; un controllo non riprodotto o un'AttackNet non valutabile sono scritti subito in run.log
(«FINDING: …») e in status.json (findings), e restano nello stato finale anche se poi l'esito è FAIL o INCOMPLETE. La riga «RESULT: <esito> (trial)» si scrive nel run.log solo dopo l'ultimo controllo
delle uscite (run.log compreso).

Uscite della prova (results/anetcmp/YYYYMMDD-ANETTRIAL-<commit>[-r<N>], data di oggi; --out-root fuori dal repository o, dentro, solo sotto
results/anetcmp): status.json, run.log, trial.json, trial.md, solo aggregati (nessun identificativo, percorso, nome
dell'host o dell'utente: controllati prima della scrittura). I tre checkpoint delle AttackNet vanno in
<--checkpoint-root>/<run id>/, fuori dal repository, uscita non ufficiale: nel record solo nomi e SHA-256.

  .venv/bin/python -I -B scripts/pad_anet_comparison.py check-inputs
  .venv/bin/python -I -B scripts/pad_anet_comparison.py check-machine
  caffeinate -dims .venv/bin/python -I -B scripts/pad_anet_comparison.py trial --attempt 1 --i-am-authorized --checkpoint-root <cartella fuori dal repository>
  .venv/bin/python -I -B scripts/pad_anet_comparison.py plan [--trial-run results/anetcmp/<run-id>] [--out-dir <cartella nuova>]
  .venv/bin/python -I -B scripts/pad_anet_comparison.py check-plan --plan-dir <cartella> [--trial-run <prova>]

Riuso per importazione, moduli C2 congelati: pad_arch_comparison (PAC: configurazione della preregistrazione,
ingressi, thread, macchina, EpochTimer, inference_pass, baseline_check, history_differences, read_trial_record,
scansione delle uscite), pad_arch_run (R: reestimate, bn_order_problems, pr27_checks, evaluator_agreement,
save_checkpoint), nuaa_bn_recalibration (BR), nuaa_bn_test (BT.install), nuaa_folds_run (FR), nuaa_session_ablation_run
(SR), pad_arch_models (AM.model_topology, model_counts), m2_writers, path_guard; moduli nuovi pad_anet_models (AN) e
pad_anet_definitions (DEF). Dipendenze: NumPy; TensorFlow 2.15 (solo trial, solo CPU).
"""
import argparse
import gc
import json
import math
import os
import re
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
import pad_anet_definitions as DEF  # noqa: E402
import pad_anet_models as AN  # noqa: E402
import pad_arch_comparison as PAC  # noqa: E402
import pad_arch_definitions as D  # noqa: E402
import pad_arch_models as AM  # noqa: E402
import pad_arch_run as R  # noqa: E402
import path_guard as pg  # noqa: E402

Failure, fail = FR.Failure, FR.fail
sha256_bytes, sha256_file, read_regular, rel_path, csv_text = FR.sha256_bytes, FR.sha256_file, FR.read_regular, FR.rel_path, FR.csv_text
now = FR.now
hide_gpu, training_callbacks, check_scheduler = FR.hide_gpu, FR.training_callbacks, FR.check_scheduler
finite_argmin, finite_or_none, non_finite_entries = DEF.finite_argmin, DEF.finite_or_none, DEF.non_finite_entries

DEFAULT_CONFIG = "delivery/anetcmp/pad_anet_comparison.json"
DEFAULT_OUT_ROOT = "results/anetcmp"
DEFAULT_PLAN_DIR = "delivery/anetcmp/plan"
CONFIG_SCHEMA = "pad-anet-comparison-config/1"
TRIAL_SCHEMA = "pad-anet-trial/1"
RUN_ID_RE = re.compile(r"^\d{8}-ANETTRIAL-[0-9a-f]{7,40}(?:-r([2-9]|[1-9]\d+))?$")
LOOKS_LIKE_RUN = re.compile(r"^\d{8}-ANETTRIAL-", re.IGNORECASE)      # una voce che somiglia a un run id (forma da controllare)
THIS_FILE = "scripts/pad_anet_comparison.py"
CODE_FILES = PAC.CODE_FILES + ("scripts/pad_arch_comparison.py", "scripts/pad_arch_run.py", "scripts/nuaa_bn_test.py", "scripts/pad_anet_models.py",
                               "scripts/pad_anet_definitions.py")
TOP_KEYS = ("schema", "task", "authorisation", "declarations", "preregistration", "c2_trial", "bn_recalibration_config", "c2_results", "code",
            "architectures", "training", "trial", "plan", "definitions", "outputs", "environment", "choices_of_the_author", "test")
DECLARATION_KEYS = ("no_test", "already_known", "not_independent", "post_hoc", "technical_trial", "cpu_only", "checkpoints", "threads")
DECLARATION_PHRASES = {"no_test": "no test frame is assembled or evaluated", "already_known": "already known",
                       "not_independent": "not an independent validation", "post_hoc": "post hoc"}
CHOICE_KEYS = ("constructors", "livenessnet_control", "six_runs", "order", "non_evaluable_rule", "guardian", "reading", "rankings", "denominators",
               "test_one_class", "percentage_points", "descriptive_vs_resnet18", "caps", "trial_outcomes", "reference_cells", "installed_copy",
               "descriptive_conditions", "departure_from_c2", "attempts")
TRIAL_UNIT = PAC.TRIAL_UNIT
TRIAL_ROLES = PAC.TRIAL_ROLES
TRIAL_EPOCHS = 3
TRIAL_PATIENCE = 4
CONTROL = DEF.REFERENCE
CAP_LIMIT_MINUTES = 60
CAP_UNIT = PAC.CAP_UNIT
RUN_CAP_UNIT = PAC.RUN_CAP_UNIT
STATUS_CODE = {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2, "CONTROL_FAILED": 3, "NOT_EVALUABLE": 5}
STATUS, LOG, TRIAL_JSON, TRIAL_MD = FR.STATUS, FR.LOG, "trial.json", "trial.md"
TRIAL_OUTPUTS = (TRIAL_JSON, TRIAL_MD)
PLAN_OUTPUTS = ("run_matrix.csv", "reference_cells.csv", "run_order.csv", "run_plan.csv", "run_plan.md", "run_caps.csv")
TRIAL_PENDING = "TRIAL_PENDING"
EXTRA_PROTECTED = R.EXTRA_PROTECTED + ("results/archrun", "results/archeval")
HISTORY_KEYS = PAC.HISTORY_KEYS
RUN_TAG = {"AttackNetV1": "v1", "AttackNetV2_1": "v2_1", "AttackNetV2_2": "v2_2"}
C2_RUNS = ("nuaa_patience_4", "nuaa_patience_12")
C2_RUN_PATIENCE = {"nuaa_patience_4": 4, "nuaa_patience_12": 12}
C2_TAG = {"nuaa_patience_4": "p4", "nuaa_patience_12": "p12"}
REFERENCE_CELLS = len(DEF.REUSED) * len(DEF.FOLDS) * len(DEF.PLAN_SEEDS) * len(DEF.PATIENCES)          # 48
NEW_TRAININGS = len(AN.NETWORKS) * len(DEF.FOLDS) * len(DEF.PLAN_SEEDS) * len(DEF.PATIENCES)           # 72
MATRIX_COLUMNS = ("run", "position", "network", "fold", "seed", "patience", "arm", "fitting_frames", "dev_frames", "max_epochs",
                  "test_frames_of_fold", "reference")
CELL_COLUMNS = ("network", "fold", "seed", "patience", "run_id", "checkpoint_sha256", "test_scores_file", "evaluation_run_id",
                "evaluation_restricted_sha256sums_sha256")
ORDER_COLUMNS = ("sequence", "run", "network", "patience", "trainings", "process")
PLAN_COLUMNS = ("network", "patience", "units", "new_trainings", "epochs_min", "epochs_max", "fitting_frames_per_unit", "dev_frames_per_unit",
                "seconds_per_epoch", "seconds_outside_epochs", "seconds_phase_2", "training_wall_hours_min", "training_wall_hours_max",
                "test_scores", "dev_reproductions", "evaluation_wall_hours", "cpu_over_wall", "training_cpu_hours_min", "training_cpu_hours_max",
                "evaluation_cpu_hours")
CAPS_COLUMNS = ("run", "network", "patience", "trainings", "wall_hours_min", "wall_hours_max", "cap_wall_hours", "cpu_hours_min", "cpu_hours_max")
BANNED = PAC.BANNED
PHASE2_STEPS = ("save", "scorer", "reload", "ordinary_dev", "reestimate", "consistency", "re_estimated_dev")
C_STEPS = PHASE2_STEPS + ("installed_copy",)
TEST_READ = (
    "no test frame is assembled or evaluated and no test score or metric of the new trainings is read, computed or written. check-inputs "
    "and trial read what pad_arch_comparison.load_inputs reads (its TEST_READ: the five M2 manifests, the cache hashed whole with its label "
    "and subject members read, the extracted frames file loaded whole with the SHA-256 of every frame, the cells run, the folds run, the "
    "configurations, status.json and restricted SHA256SUMS of the two accepted LivenessNet runs and the card arm_b_fold1_seed42 of the run "
    "at patience 4; the fold schema is never loaded); in addition this configuration, the configuration of the pre-registration of C2, "
    "the configuration of PR #27 (for its tolerances and BN settings), the code files pinned here and the status.json and trial.json of "
    "the official C2 trial. check-inputs, trial, plan and check-plan read the public status.json and results.json of the C2 evaluation "
    "(published files; results.json holds the aggregated C2 test results, already known: it is parsed and only its list of the 48 "
    "checkpoints, with run, network, fold, seed and checkpoint SHA-256, is used); no restricted file of the C2 runs or of the C2 "
    "evaluation and no <model>_test_scores.csv is opened. trial, in addition: the image arrays of the cache, loaded whole into the frame "
    "store; of all the frames it assembles only the fitting and dev frames of arm B, fold 1 (an explicit check refuses any other role); "
    "it reads back its own checkpoints, written outside the repository. plan and check-plan read also, if given, the status.json and "
    "trial.json of a trial of this code")

# punti di iniezione (solo i test): con la configurazione consegnata trial li vuole tutti None (check_hooks)
REPO_STATE = None       # (root) -> {"commit", "dirty"} (default: SR.repo_state)
ARCH_LOADER = None      # (path) -> modulo con create_model (default: FR.default_arch_loader)
TRAINER = None          # come anet_trainer (default: anet_trainer)
MACHINE = None          # () -> dizionario della macchina (default: PAC.machine)
# tutti gli agganci (variabili di modulo maiuscole che valgono None) dei moduli del progetto importati da questo, diretti
# e indiretti: la suite li ritrova con ast nei sorgenti e controlla che l'elenco sia completo
HOOKS = {"pad_anet_comparison": ("REPO_STATE", "ARCH_LOADER", "TRAINER", "MACHINE"),
         "pad_arch_comparison": ("REPO_STATE", "ARCH_LOADER", "BUILD", "TRAINER", "MACHINE", "SYSCTL"),
         "pad_arch_run": ("REPO_STATE", "MACHINE", "ENVIRONMENT", "TF_MODULE", "ARCH_LOADER", "TRAINER", "SAVE_MODEL", "MAKE_SCORER",
                          "REESTIMATE", "ESTIMATOR_BASE"),
         "nuaa_bn_recalibration": ("REPO_STATE", "MAKE_SCORER", "LOAD_MODEL"),
         "nuaa_bn_test": ("REPO_STATE", "MAKE_SCORER"),
         "nuaa_folds_run": ("REPO_STATE", "PREPROCESSOR", "TRAINER", "MAKE_SCORER", "CLOCK", "ARCH_LOADER"),
         "nuaa_session_ablation_run": ("REPO_STATE", "TRAINER", "MAKE_SCORER", "MAKE_LOGITS", "CLOCK", "ARCH_LOADER"),
         "nuaa_patience_verify": ("REPO_STATE", "PLOTTER")}
# valori che solo i test sostituiscono (il mondo sintetico): con la configurazione consegnata devono essere quelli dell'import
PINNED_VALUES = {("nuaa_folds_run", "FRAME_SHAPE"): FR.FRAME_SHAPE, ("nuaa_session_ablation_run", "SCHEMA_CONFIG_SHA256"): SR.SCHEMA_CONFIG_SHA256,
                 ("nuaa_session_ablation_run", "CELLS_CONFIG_SHA256"): SR.CELLS_CONFIG_SHA256,
                 ("nuaa_session_ablation_run", "peak_rss_mb"): SR.peak_rss_mb, ("nuaa_folds_run", "environment"): FR.environment}

ANET_TRAINER_SUBSTITUTIONS = (
    ("def trial_trainer(data: dict, training: dict, network: str, train_seed: int, build_model, cap_check, timer) -> tuple:\n",
     "def anet_trainer(data: dict, training: dict, network: str, train_seed: int, build_model, cap_check, timer) -> tuple:\n"),
    ("    \"\"\"Copia dichiarata di nuaa_folds_run.default_trainer (la ricetta di n1_trainer.train, riga per riga), con le sole\n"
     "    sostituzioni di TRAINER_SUBSTITUTIONS: il modello viene da build_model() (una rete del registro) invece che da\n"
     "    create_model(\"LivenessNet\"); nessun checkpoint salvato (nessun weights_path); la callback di timer.callback(tf),\n"
     "    che registra i tempi e la storia e non tocca il training, è inserita prima di Cap; la scheda dice la rete e che\n"
     "    nessun checkpoint è salvato; restituisce (scheda, modello) per le passate di inferenza della prova.\"\"\"\n",
     "    \"\"\"Copia dichiarata di pad_arch_comparison.trial_trainer (C2, congelata) con le sole sostituzioni di\n"
     "    ANET_TRAINER_SUBSTITUTIONS, per la regola (n) di pad_anet_definitions: l'epoca ripristinata è quella della\n"
     "    val_loss finita minima (finite_argmin; np.argmin di un NaN dà il primo NaN), None se nessuna è finita; i valori\n"
     "    non finiti della storia sono scritti come null, con l'elenco (grandezza, prima epoca) in non_finite; la scheda\n"
     "    dice che il trainer non salva il checkpoint (lo salva chi chiama). Su storie finite scheda e pesi sono quelli\n"
     "    dell'originale.\"\"\"\n"),
    ("    best = int(np.argmin(h[\"val_loss\"]))\n", "    best = finite_argmin(h[\"val_loss\"])\n"),
    ("\"restored_epoch\": best + 1,\n", "\"restored_epoch\": None if best is None else best + 1,\n"),
    ("            \"best_val_loss\": float(h[\"val_loss\"][best]), \"val_acc_at_best\": float(h[\"val_accuracy\"][best]),\n"
     "            \"history\": {k: [float(v) for v in h[k]] for k in (\"loss\", \"accuracy\", \"val_loss\", \"val_accuracy\", \"lr\") if k in h},\n",
     "            \"best_val_loss\": None if best is None else float(h[\"val_loss\"][best]),\n"
     "            \"val_acc_at_best\": None if best is None else finite_or_none(h[\"val_accuracy\"][best]),\n"
     "            \"history\": {k: [finite_or_none(v) for v in h[k]] for k in (\"loss\", \"accuracy\", \"val_loss\", \"val_accuracy\", \"lr\") if k in h},\n"
     "            \"non_finite\": non_finite_entries(h),\n"),
    ("\"checkpoint\": \"none saved (trial)\"}\n", "\"checkpoint\": \"none saved by the trainer (the caller saves it)\"}\n"),
    ("f\"(val_loss {card['best_val_loss']:.6f})\", flush=True)\n",
     "f\"(val_loss {'none finite' if card['best_val_loss'] is None else format(card['best_val_loss'], '.6f')})\", flush=True)\n"),
)


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


def _int_eq(v, want: int, what: str) -> None:
    if isinstance(v, bool) or v != want or not isinstance(v, int):
        fail(f"configuration: {what} must be {want}")


def _number(v, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        fail(f"configuration: {what} must be a finite positive number")
    return float(v)


def _author(v, what: str) -> None:
    _text(v, what)
    if "a choice of the author" not in v:
        fail(f"configuration: {what} must be declared a choice of the author")


def pinned_bytes(base: Path, e: dict, what: str) -> tuple:
    """(percorso, byte) di un file fissato con il suo SHA-256: diverso -> Failure."""
    p = rel_path(base, e["path"])
    b = read_regular(p, what)
    if sha256_bytes(b) != e["sha256"]:
        fail(f"{what} {p.name}: SHA-256 {sha256_bytes(b)} differs from the configuration")
    return p, b


def run_name(network: str, patience: int) -> str:
    return f"anet_{RUN_TAG[network]}_p{patience}"


def run_names() -> list:
    """I sei run, nell'ordine fissato (scelta dell'autore, plan.order): patience 4 poi 12; in ciascuna AttackNetV1,
    AttackNetV2_1, AttackNetV2_2."""
    return [run_name(n, p) for p in DEF.PATIENCES for n in AN.NETWORKS]


def check_architectures(ar: dict, cfg_pr: dict) -> None:
    """Le tre AttackNet: lo stesso file di riferimento fissato da C2 per LivenessNet, la forma dei frame, conteggi e
    strati BN uguali all'aritmetica del registro, il costruttore del registro."""
    _keys(ar, ["input_shape", "networks", "path", "sha256"], "architectures")
    if ar["path"] != cfg_pr["architectures"]["LivenessNet"]["path"] or ar["sha256"] != cfg_pr["architectures"]["LivenessNet"]["sha256"]:
        fail("configuration: architectures must be the reference architecture file pinned by the C2 pre-registration")
    if ar["input_shape"] != list(FR.FRAME_SHAPE):
        fail(f"configuration: architectures.input_shape must be {list(FR.FRAME_SHAPE)} (the frame shape)")
    _keys(ar["networks"], list(AN.NETWORKS), "architectures.networks")
    for n in AN.NETWORKS:
        e = ar["networks"][n]
        _keys(e, ["batchnorm_layers", "batchnorm_names", "constructor", "non_trainable_parameters", "trainable_parameters"], f"architectures.networks.{n}")
        exp = AN.expected_counts(n, ar["input_shape"])
        got = {"trainable": e["trainable_parameters"], "non_trainable": e["non_trainable_parameters"], "batchnorm_layers": e["batchnorm_layers"]}
        if any(got[k] != exp[k] or not FR.is_count(got[k]) for k in got):
            fail(f"configuration: architectures.networks.{n} counts {got} differ from the arithmetic of the registry {exp}")
        if e["batchnorm_names"] != AN.bn_order(n):
            fail(f"configuration: architectures.networks.{n}.batchnorm_names must be {AN.bn_order(n)} (the registry)")
        if e["constructor"] != AN.CONSTRUCTOR:
            fail(f"configuration: architectures.networks.{n}.constructor must be {AN.CONSTRUCTOR!r}")


def check_trial_section(t: dict, cfg_pr: dict, bn_cfg: dict) -> None:
    _keys(t, ["after_outcome", "arm", "attempts", "bn_check", "cap_unit", "checkpoints", "conditions_file", "control", "control_rule", "dataset",
              "epochs", "estimate_minutes", "estimator_batch", "fold", "inference_batch", "networks", "patience", "roles", "seed",
              "upper_estimate_minutes", "wall_time_cap_minutes"], "trial")
    if {k: t[k] for k in TRIAL_UNIT} != TRIAL_UNIT or any(isinstance(t[k], bool) for k in TRIAL_UNIT):
        fail(f"configuration: the trial unit must be {TRIAL_UNIT} (arm B, fold 1, seed 42, as the card asks)")
    if t["roles"] != list(TRIAL_ROLES):
        fail(f"configuration: trial.roles must be {list(TRIAL_ROLES)} (never the test)")
    if t["networks"] != list(AN.NETWORKS):
        fail(f"configuration: trial.networks must be {list(AN.NETWORKS)}, in this order")
    if t["control"] != CONTROL:
        fail(f"configuration: trial.control must be {CONTROL}")
    _int_eq(t["epochs"], TRIAL_EPOCHS, "trial.epochs")
    _int_eq(t["patience"], TRIAL_PATIENCE, "trial.patience")
    _int_eq(t["inference_batch"], FR.SCORE_BATCH, "trial.inference_batch")
    _int_eq(t["estimator_batch"], bn_cfg["evaluation"]["batch"], "trial.estimator_batch")
    if not FR.is_count(t["wall_time_cap_minutes"]) or not 1 <= t["wall_time_cap_minutes"] <= CAP_LIMIT_MINUTES:
        fail(f"configuration: trial.wall_time_cap_minutes must be an integer from 1 to {CAP_LIMIT_MINUTES}")
    if t["cap_unit"] != CAP_UNIT:
        fail(f"configuration: trial.cap_unit must be {CAP_UNIT!r}")
    _number(t["estimate_minutes"], "trial.estimate_minutes")
    _number(t["upper_estimate_minutes"], "trial.upper_estimate_minutes")
    if not t["estimate_minutes"] <= t["upper_estimate_minutes"] < t["wall_time_cap_minutes"]:
        fail("configuration: trial needs estimate_minutes <= upper_estimate_minutes < wall_time_cap_minutes")
    for k in ("after_outcome", "attempts", "bn_check", "checkpoints", "conditions_file", "control_rule"):
        _text(t[k], f"trial.{k}")
    if any(x not in t["after_outcome"] for x in ("PASS", "FAIL", "CONTROL_FAILED", "NOT_EVALUABLE", "no new attempt")):
        fail("configuration: trial.after_outcome must state what follows PASS, FAIL, CONTROL_FAILED and NOT_EVALUABLE (no new attempt ...)")
    if "machine" not in t["control_rule"]:
        fail("configuration: trial.control_rule must include the comparison of machine and versions with the C2 trial")
    if "outside the repository" not in t["checkpoints"]:
        fail("configuration: trial.checkpoints must say that the checkpoints are saved outside the repository")
    if "CONTROL_FAILED" not in t["control_rule"]:
        fail("configuration: trial.control_rule must state the outcome CONTROL_FAILED when the history is not reproduced")


def check_plan_section(pl: dict) -> None:
    _keys(pl, ["arm", "caps", "evaluation", "folds", "machine_rule", "max_epochs", "networks", "order", "patiences", "reference", "runs", "seeds"], "plan")
    if pl["networks"] != list(AN.NETWORKS) or pl["patiences"] != list(DEF.PATIENCES) or pl["folds"] != list(DEF.FOLDS) \
            or pl["seeds"] != list(DEF.PLAN_SEEDS) or pl["arm"] != TRIAL_UNIT["arm"]:
        fail("configuration: plan networks, patiences, folds, seeds and arm must be those of the card (3 AttackNets, 4 and 12, folds 1-4, "
             "seeds 42-44, arm B)")
    _int_eq(pl["max_epochs"], FR.FROZEN_TRAINING["max_epochs"], "plan.max_epochs")
    if pl["runs"] != run_names():
        fail(f"configuration: plan.runs must be {run_names()} (six runs of 12, in this order)")
    _keys(pl["reference"], ["cells", "networks", "rule"], "plan.reference")
    if pl["reference"]["networks"] != list(DEF.REUSED):
        fail(f"configuration: plan.reference.networks must be {list(DEF.REUSED)}")
    _int_eq(pl["reference"]["cells"], REFERENCE_CELLS, "plan.reference.cells")
    _text(pl["reference"]["rule"], "plan.reference.rule")
    if "not re-evaluated" not in pl["reference"]["rule"]:
        fail("configuration: plan.reference.rule must say that the 48 C2 checkpoints are not re-evaluated")
    ca = pl["caps"]
    _keys(ca, ["anomaly_factor", "round_up_hours", "rule", "unit", "wall_multiplier"], "plan.caps")
    _number(ca["round_up_hours"], "plan.caps.round_up_hours")
    if _number(ca["wall_multiplier"], "plan.caps.wall_multiplier") < 1 or _number(ca["anomaly_factor"], "plan.caps.anomaly_factor") < 1:
        fail("configuration: plan.caps.wall_multiplier and anomaly_factor must be at least 1")
    if ca["unit"] != RUN_CAP_UNIT:
        fail(f"configuration: plan.caps.unit must be {RUN_CAP_UNIT!r}")
    _text(ca["rule"], "plan.caps.rule")
    for k in ("evaluation", "order", "machine_rule"):
        _text(pl[k], f"plan.{k}")
    if "same machine" not in pl["machine_rule"] or "same threads" not in pl["machine_rule"]:
        fail("configuration: plan.machine_rule must state the same machine and the same threads")
    if "once" not in pl["evaluation"]:
        fail("configuration: plan.evaluation must say that the NUAA test is read once")


def check_definitions(df: dict) -> None:
    _keys(df, ["bootstrap", "conditions", "decision", "denominators", "descriptive_conditions", "descriptive_vs_resnet18", "eer_threshold",
               "fixed_threshold", "frame_ids", "guardian", "non_evaluable", "one_class", "presentation", "primary", "rankings", "reuse"], "definitions")
    if df["bootstrap"] != SR.BOOT:
        fail(f"configuration: definitions.bootstrap must be the recipe of the project {SR.BOOT}")
    if df["conditions"] != [list(c) for c in D.CONDITIONS]:
        fail("configuration: definitions.conditions must list the 8 conditions of C2")
    if df["decision"] != FR.DECISION_RULE or df["eer_threshold"] != FR.THRESHOLD_CRITERION:
        fail("configuration: definitions.decision and eer_threshold must be the rule and the EER criterion of the project")
    if df["fixed_threshold"] != FR.FIXED_THRESHOLD or isinstance(df["fixed_threshold"], bool):
        fail(f"configuration: definitions.fixed_threshold must be {FR.FIXED_THRESHOLD}")
    _keys(df["frame_ids"], ["field", "rule"], "definitions.frame_ids")
    if df["frame_ids"]["field"] != D.FRAME_ID_FIELD:
        fail(f"configuration: definitions.frame_ids.field must be {D.FRAME_ID_FIELD!r}")
    pr = df["primary"]
    _keys(pr, ["contrast", "intervals", "ordinary", "patience", "per_seed", "re_estimated", "reading_names", "threshold", "variation"], "definitions.primary")
    _int_eq(pr["patience"], DEF.PRIMARY_PATIENCE, "definitions.primary.patience")
    if pr["threshold"] != DEF.PRIMARY_THRESHOLD or pr["ordinary"] != list(DEF.PRIMARY_ORDINARY) or pr["re_estimated"] != list(DEF.PRIMARY_RE_ESTIMATED):
        fail("configuration: definitions.primary must fix patience 4, threshold 0.5 and the two BN states of pad_anet_definitions")
    if pr["intervals"] != DEF.INTERVALS_NOTE:
        fail("configuration: definitions.primary.intervals must be pad_anet_definitions.INTERVALS_NOTE (individual intervals)")
    if pr["reading_names"] != {k: list(DEF.reading_names(k)) for k in AN.NETWORKS}:
        fail("configuration: definitions.primary.reading_names must be those of pad_anet_definitions.reading_names")
    for k in ("contrast", "per_seed", "variation"):
        _text(pr[k], f"definitions.primary.{k}")
    if "minus" not in pr["variation"] or "minus" not in pr["contrast"]:
        fail("configuration: definitions.primary.variation and contrast must state the subtractions")
    if df["presentation"] != DEF.PP_NOTE:
        fail("configuration: definitions.presentation must be pad_anet_definitions.PP_NOTE (percentage points at presentation only)")
    for k in ("denominators", "descriptive_conditions", "descriptive_vs_resnet18", "guardian", "non_evaluable", "one_class", "rankings", "reuse"):
        _text(df[k], f"definitions.{k}")
    if "not evaluable" not in df["non_evaluable"]:
        fail("configuration: definitions.non_evaluable must define the not evaluable model")
    if "no reading" not in df["descriptive_vs_resnet18"]:
        fail("configuration: definitions.descriptive_vs_resnet18 must say that the comparison has no reading")


def load_config(p: Path, base: Path) -> tuple:
    """(configurazione, SHA-256 dei byte letti, configurazione della preregistrazione C2, configurazione del run
    accettato a patience 4, configurazione della PR #27). Insiemi chiusi, valori fissati controllati qui, prima di
    leggere qualunque dato; nessun valore predefinito."""
    data = read_regular(p, "configuration")
    cfg = SR.read_json(data, f"configuration {p.name}")
    if cfg.get("schema") != CONFIG_SCHEMA:
        fail(f"configuration {p.name}: schema must be {CONFIG_SCHEMA!r}")
    if sorted(cfg) != sorted(TOP_KEYS):
        fail(f"configuration: fields must be exactly {', '.join(TOP_KEYS)} (missing {sorted(set(TOP_KEYS) - set(cfg))}, "
             f"unknown {sorted(set(cfg) - set(TOP_KEYS))})")
    _text(cfg["task"], "task")
    _text(cfg["authorisation"], "authorisation")
    if cfg["test"] != TEST_READ:
        fail("configuration: test must state exactly what is read (TEST_READ of the script)")
    de = cfg["declarations"]
    _keys(de, DECLARATION_KEYS, "declarations")
    for k in DECLARATION_KEYS:
        _text(de[k], f"declarations.{k}")
    for k, phrase in DECLARATION_PHRASES.items():
        if phrase not in de[k]:
            fail(f"configuration: declarations.{k} must say {phrase!r}")
    _pin(cfg["preregistration"], "preregistration")
    pp_, _ = pinned_bytes(base, cfg["preregistration"], "pre-registration configuration")
    cfg_pr, sha_pr, pcfg = PAC.load_config(pp_, base)
    _pin(cfg["bn_recalibration_config"], "bn_recalibration_config")
    bp, _ = pinned_bytes(base, cfg["bn_recalibration_config"], "configuration of PR #27")
    bn_cfg, _ = BR.load_config(bp)
    ct = cfg["c2_trial"]
    _keys(ct, ["folder", "run_id", "status_sha256", "trial_json_sha256"], "c2_trial")
    _text(ct["folder"], "c2_trial.folder")
    if not isinstance(ct["run_id"], str) or not PAC.RUN_ID_RE.match(ct["run_id"]) or not SR._hex(ct["status_sha256"]) or not SR._hex(ct["trial_json_sha256"]):
        fail("configuration: c2_trial needs an ARCHTRIAL run id and the SHA-256 of its status.json and trial.json")
    cr = cfg["c2_results"]
    _keys(cr, ["evaluation_run_id", "restricted_sha256sums_sha256", "results", "runs", "status"], "c2_results")
    _pin(cr["status"], "c2_results.status")
    _pin(cr["results"], "c2_results.results")
    if not isinstance(cr["evaluation_run_id"], str) or not re.fullmatch(r"\d{8}-ARCHEVAL-[0-9a-f]{7,40}", cr["evaluation_run_id"]):
        fail("configuration: c2_results.evaluation_run_id must be an ARCHEVAL run id")
    if not SR._hex(cr["restricted_sha256sums_sha256"]):
        fail("configuration: c2_results.restricted_sha256sums_sha256 malformed")
    _keys(cr["runs"], list(C2_RUNS), "c2_results.runs")
    for k in C2_RUNS:
        if not isinstance(cr["runs"][k], str) or not R.RUN_ID_RE.match(cr["runs"][k]) or R.RUN_ID_RE.match(cr["runs"][k]).group(1) != str(C2_RUN_PATIENCE[k]):
            fail(f"configuration: c2_results.runs.{k} must be an ARCHRUN run id at patience {C2_RUN_PATIENCE[k]}")
    co = cfg["code"]
    _keys(co, list(CODE_FILES), "code")
    if not all(SR._hex(v) for v in co.values()):
        fail("configuration: code SHA-256 values malformed")
    check_architectures(cfg["architectures"], cfg_pr)
    tr = cfg["training"]
    _keys(tr, ["non_finite_rule", "recipe", "threads", "trainer", "trial_overrides"], "training")
    for k in ("non_finite_rule", "recipe", "trainer"):
        _text(tr[k], f"training.{k}")
    if tr["threads"] != cfg_pr["training"]["threads"]:
        fail("configuration: training.threads must be those of the C2 pre-registration (10 intra-op and 10 inter-op)")
    PAC.check_threads_config(tr["threads"])
    if tr["trial_overrides"] != {"max_epochs": TRIAL_EPOCHS} or isinstance(tr["trial_overrides"].get("max_epochs"), bool):
        fail(f"configuration: training.trial_overrides must be exactly {{\"max_epochs\": {TRIAL_EPOCHS}}}")
    if "not evaluable" not in tr["non_finite_rule"]:
        fail("configuration: training.non_finite_rule must state when a model is not evaluable")
    check_trial_section(cfg["trial"], cfg_pr, bn_cfg)
    if cfg["trial"]["patience"] != pcfg["training"]["early_stopping_patience"]:
        fail("configuration: the trial patience must be that of the accepted run at patience 4")
    check_plan_section(cfg["plan"])
    check_definitions(cfg["definitions"])
    banned = [w for w in BANNED if w in json.dumps([cfg["definitions"], de]).lower()]
    if banned:
        fail(f"configuration: the definitions and declarations must not use causal or conclusive wording ({', '.join(banned)})")
    _keys(cfg["outputs"], ["plan", "trial"], "outputs")
    if cfg["outputs"]["trial"] != [STATUS, LOG, *TRIAL_OUTPUTS] or cfg["outputs"]["plan"] != list(PLAN_OUTPUTS):
        fail(f"configuration: outputs must be trial {[STATUS, LOG, *TRIAL_OUTPUTS]} and plan {list(PLAN_OUTPUTS)}")
    if cfg["environment"] != cfg_pr["environment"]:
        fail("configuration: environment must be that of the C2 pre-registration")
    ch = cfg["choices_of_the_author"]
    _keys(ch, CHOICE_KEYS, "choices_of_the_author")
    for k in CHOICE_KEYS:
        _author(ch[k], f"choices_of_the_author.{k}")
    return cfg, sha256_bytes(data), cfg_pr, pcfg, bn_cfg


def trial_training(cfg: dict, pcfg: dict) -> dict:
    """La ricetta della prova: quella del run accettato a patience 4 (la ricetta di C2) con le sole sostituzioni di
    training.trial_overrides (3 epoche)."""
    return {**pcfg["training"], **cfg["training"]["trial_overrides"]}


# ============================================================================ ingressi

def check_code(cfg: dict) -> dict:
    got = {f: sha256_bytes(read_regular(ROOT / f, "code file")) for f in CODE_FILES}
    bad = [f for f in CODE_FILES if got[f] != cfg["code"][f]]
    if bad:
        fail(f"code files {bad} differ from the SHA-256 pinned in the configuration")
    return {**got, THIS_FILE: sha256_bytes(read_regular(Path(__file__), "code file"))}


def load_c2_trial(cfg: dict, cfg_sha_pr: str, base: Path) -> dict:
    """Il record della prova ufficiale di C2 (riferimento del controllo LivenessNet): status.json e trial.json con gli
    SHA-256 fissati, validati da PAC.read_trial_record (formato pad-arch-trial/2), PASS, con il run id fissato e la
    configurazione della preregistrazione; la storia di LivenessNet con almeno le epoche della prova."""
    ct = cfg["c2_trial"]
    folder = rel_path(base, ct["folder"])
    doc, files = PAC.read_trial_record(folder, "C2 trial")
    if files[STATUS] != ct["status_sha256"] or files[PAC.TRIAL_JSON] != ct["trial_json_sha256"]:
        fail("C2 trial: status.json or trial.json differ from the SHA-256 of the configuration")
    if doc["schema"] != PAC.TRIAL_SCHEMA or doc["status"] != "PASS" or doc["run_id"] != ct["run_id"]:
        fail(f"C2 trial: not the PASS record {ct['run_id']} of format {PAC.TRIAL_SCHEMA}")
    if doc["config_sha256"] != cfg_sha_pr:
        fail("C2 trial: produced with another configuration than the pinned pre-registration")
    if len(doc["networks"][CONTROL]["epochs"]) < TRIAL_EPOCHS:
        fail(f"C2 trial: the {CONTROL} history has fewer than {TRIAL_EPOCHS} epochs")
    return doc


def c2_reference_cells(status: dict, results: dict, cfg: dict) -> list:
    """Le 48 celle riusate di C2 dai soli file pubblici della valutazione: status.json PASS, ufficiale, con il run id
    fissato, i due run fissati e l'ancora del SHA256SUMS riservato fissata; results.json con lo stesso run id e i 48
    modelli (run, rete, fold, seed, weights_sha256): le due reti riusate × 4 fold × 3 seed × 2 patience, ciascuna una
    volta. Per riga il nome del file dei punteggi di test riservato (<p4|p12>_<rete>_fold<k>_seed<s>_test_scores.csv),
    il cui SHA-256 si controlla nella valutazione dal SHA256SUMS riservato ancorato: nessun file riservato è letto qui."""
    cr = cfg["c2_results"]
    if status.get("status") != "PASS" or status.get("run_id") != cr["evaluation_run_id"] or status.get("official") is not True:
        fail(f"C2 evaluation: status.json is not the official PASS {cr['evaluation_run_id']}")
    if status.get("restricted_sha256sums_sha256") != cr["restricted_sha256sums_sha256"]:
        fail("C2 evaluation: restricted_sha256sums_sha256 of status.json differs from the configuration")
    runs = status.get("runs") or {}
    if sorted(runs) != sorted(C2_RUNS) or any((runs[k] or {}).get("run_id") != cr["runs"][k] for k in C2_RUNS):
        fail("C2 evaluation: the runs of status.json differ from c2_results.runs")
    if results.get("run_id") != cr["evaluation_run_id"] or results.get("status") != "PASS":
        fail("C2 evaluation: results.json is not the PASS record of the evaluation run")
    models = results.get("models")
    if not isinstance(models, list) or len(models) != REFERENCE_CELLS:
        fail(f"C2 evaluation: results.json must list {REFERENCE_CELLS} models")
    out, seen = [], set()
    for m in models:
        if not isinstance(m, dict) or m.get("run") not in C2_RUNS or m.get("network") not in DEF.REUSED or m.get("fold") not in DEF.FOLDS \
                or m.get("seed") not in DEF.PLAN_SEEDS or isinstance(m.get("fold"), bool) or isinstance(m.get("seed"), bool) or not SR._hex(m.get("weights_sha256")):
            fail("C2 evaluation: a model of results.json is outside the plan (run, network, fold, seed) or lacks its checkpoint SHA-256")
        key = (m["network"], m["fold"], m["seed"], C2_RUN_PATIENCE[m["run"]])
        if key in seen:
            fail("C2 evaluation: a model of results.json is listed twice")
        seen.add(key)
        out.append({"network": m["network"], "fold": m["fold"], "seed": m["seed"], "patience": key[3], "run_id": cr["runs"][m["run"]],
                    "checkpoint_sha256": m["weights_sha256"],
                    "test_scores_file": f"{C2_TAG[m['run']]}_{PAC.training_name(m['network'], m['fold'], m['seed'])}_test_scores.csv",
                    "evaluation_run_id": cr["evaluation_run_id"], "evaluation_restricted_sha256sums_sha256": cr["restricted_sha256sums_sha256"]})
    out.sort(key=lambda r: (DEF.PATIENCES.index(r["patience"]), DEF.REUSED.index(r["network"]), r["fold"], r["seed"]))
    check_reference_cells(out)
    return out


def load_c2_results(cfg: dict, base: Path) -> list:
    cr = cfg["c2_results"]
    _, sb = pinned_bytes(base, cr["status"], "C2 evaluation status.json")
    _, rb = pinned_bytes(base, cr["results"], "C2 evaluation results.json")
    return c2_reference_cells(SR.read_json(sb, "C2 evaluation status.json"), SR.read_json(rb, "C2 evaluation results.json"), cfg)


def load_inputs(cfg: dict, cfg_pr: dict, pcfg: dict, base: Path, frames: bool = True):
    """Ingressi: codice e architettura per SHA-256, record della prova C2, file pubblici della valutazione C2 e, con
    frames, PAC.load_inputs con la configurazione della preregistrazione C2 (accetta solo quella: la configurazione C5
    non ha la sua forma, quindi qui si passa la configurazione C2 fissata e validata). Nessun frame assemblato,
    TensorFlow non importato."""
    code = check_code(cfg)
    ap = rel_path(base, cfg["architectures"]["path"])
    h = sha256_bytes(read_regular(ap, "architecture file"))
    if h != cfg["architectures"]["sha256"]:
        fail(f"architecture file {ap.name}: SHA-256 {h} differs from the configuration")
    pr_sha = sha256_bytes(read_regular(rel_path(base, cfg["preregistration"]["path"]), "pre-registration configuration"))
    c2_trial = load_c2_trial(cfg, pr_sha, base)
    cells = load_c2_results(cfg, base)
    inp = PAC.load_inputs(cfg_pr, pcfg, base) if frames else FR.Inputs()
    inp.code_c5 = {"architecture": h, **code}
    inp.c2_trial = c2_trial
    inp.reference_cells = cells
    return inp


# ============================================================================ matrice, celle, piano

def matrix_rows(cfg: dict, pcfg: dict) -> list:
    """Le 72 righe della matrice: sei run (run_names), in ciascuno fold 1-4 × seed 42, 43, 44 nelle posizioni 1-12; per
    riga braccio, frame del fitting e del dev dell'unità (configurazione accettata), epoche massime, frame di test del
    fold (configurazione accettata, dove il test è contato e non letto) e reference «none» (ogni addestramento è
    nuovo)."""
    pl = cfg["plan"]
    test = pcfg["schema_expected"]["test_per_fold"]
    rows = []
    for p in pl["patiences"]:
        for n in pl["networks"]:
            pos = 0
            for f in pl["folds"]:
                for s in pl["seeds"]:
                    pos += 1
                    rows.append({"run": run_name(n, p), "position": pos, "network": n, "fold": f, "seed": s, "patience": p, "arm": pl["arm"],
                                 "fitting_frames": pcfg["arms"][pl["arm"]]["total"], "dev_frames": pcfg["dev"]["total"], "max_epochs": pl["max_epochs"],
                                 "test_frames_of_fold": test[f - 1], "reference": "none"})
    check_matrix(rows)
    return rows


def check_matrix(rows: list) -> dict:
    """La matrice dei nuovi addestramenti: 72 righe con (rete, fold, seed, patience) tutte diverse; 36 per patience, 24
    per rete, 12 per cella rete × patience; sei run da 12 con le posizioni 1-12 e una sola rete e patience ciascuno;
    reti, fold, seed e patience del piano; nessun riferimento. Altrimenti Failure."""
    keys = [(r["network"], r["fold"], r["seed"], r["patience"]) for r in rows]
    if len(rows) != NEW_TRAININGS or len(set(keys)) != NEW_TRAININGS:
        fail(f"run matrix: {len(rows)} rows, {len(set(keys))} distinct trainings ({NEW_TRAININGS} required)")
    for r in rows:
        if r["network"] not in AN.NETWORKS or r["fold"] not in DEF.FOLDS or r["seed"] not in DEF.PLAN_SEEDS or r["patience"] not in DEF.PATIENCES \
                or r["reference"] != "none" or r["run"] != run_name(r["network"], r["patience"]):
            fail("run matrix: a row is outside the plan or names a reference")
    per_p = {p: sum(1 for k in keys if k[3] == p) for p in DEF.PATIENCES}
    per_n = {n: sum(1 for k in keys if k[0] == n) for n in AN.NETWORKS}
    per_cell = {(n, p): sum(1 for k in keys if k[0] == n and k[3] == p) for n in AN.NETWORKS for p in DEF.PATIENCES}
    if set(per_p.values()) != {36} or set(per_n.values()) != {24} or set(per_cell.values()) != {12}:
        fail(f"run matrix: per patience {per_p}, per network {per_n}: 36 per patience, 24 per network and 12 per cell required")
    runs = {}
    for r in rows:
        runs.setdefault(r["run"], []).append(r["position"])
    if sorted(runs) != sorted(run_names()) or any(sorted(v) != list(range(1, 13)) for v in runs.values()):
        fail("run matrix: six runs of 12 positions (1-12) required")
    return {"rows": len(rows), "per_patience": per_p, "per_network": per_n, "runs": len(runs)}


def check_reference_cells(rows: list) -> dict:
    """Le 48 celle riusate: LivenessNet e ResNet-18 × 4 fold × 3 seed × 2 patience, ciascuna una volta, con il run della
    sua patience e lo SHA-256 del checkpoint. Altrimenti Failure."""
    keys = [(r["network"], r["fold"], r["seed"], r["patience"]) for r in rows]
    want = {(n, f, s, p) for n in DEF.REUSED for f in DEF.FOLDS for s in DEF.PLAN_SEEDS for p in DEF.PATIENCES}
    if len(rows) != REFERENCE_CELLS or set(keys) != want or len(set(keys)) != len(keys):
        fail(f"reference cells: {len(rows)} rows; the {REFERENCE_CELLS} cells (2 networks x 4 folds x 3 seeds x 2 patiences) are required, each once")
    for r in rows:
        tag = {4: "p4", 12: "p12"}[r["patience"]]
        if not SR._hex(r["checkpoint_sha256"]) or not R.RUN_ID_RE.match(r["run_id"]) or f"-ARCHRUN-{tag}-" not in r["run_id"]:
            fail("reference cells: a row lacks the checkpoint SHA-256 or the run of its patience")
    return {"rows": len(rows)}


def order_rows() -> list:
    """L'ordine dei sei run: un processo ciascuno, patience 4 poi 12, AttackNetV1, AttackNetV2_1, AttackNetV2_2."""
    return [{"sequence": i, "run": run_name(n, p), "network": n, "patience": p, "trainings": len(DEF.FOLDS) * len(DEF.PLAN_SEEDS),
             "process": f"one process (run {i} of {len(run_names())})"}
            for i, (p, n) in enumerate(((p, n) for p in DEF.PATIENCES for n in AN.NETWORKS), start=1)]


def network_rates(rec: dict, n_fit: int, n_dev: int) -> dict:
    """Velocità di un'AttackNet dalla prova (unica regola, usata da trial.md e da plan): secondi per epoca = media del
    tempo totale delle epoche misurate (validazione compresa; la prima include il tracing del grafo: stima prudente);
    secondi fuori dalle epoche = trainer meno le sue epoche; secondi della fase 2 dei run = salvataggio, scorer del dev,
    ricarica, dev ordinario, ristima, passaggio di coerenza e dev ristimato (PHASE2_STEPS); inferenza in secondi per
    1.000 frame = passate su fitting e dev / frame; rapporto CPU / tempo reale su epoche, passate e passi. None dove una
    misura manca."""
    ep = rec.get("epochs") or []
    steps = rec.get("steps") or {}
    inf = rec.get("inference") or {}
    ov = rec.get("training_overhead")
    out = {"seconds_per_epoch": float(np.mean([e["seconds_total"] for e in ep])) if ep else None,
           "seconds_outside_epochs": None if ov is None else ov["seconds"],
           "seconds_phase_2": sum(steps[k]["seconds"] for k in PHASE2_STEPS) if all(k in steps for k in PHASE2_STEPS) else None,
           "inference_seconds_per_1000_frames": (sum(inf[r]["seconds"] for r in TRIAL_ROLES) / sum(inf[r]["frames"] for r in TRIAL_ROLES) * 1000)
           if all(r in inf for r in TRIAL_ROLES) else None, "epochs_measured": len(ep), "cpu_over_wall": None}
    wall = sum(e["seconds_total"] for e in ep) + sum(inf[r]["seconds"] for r in inf) + sum(v["seconds"] for v in steps.values()) + (ov["seconds"] if ov else 0.0)
    cpu = sum(e["cpu_seconds"] for e in ep) + sum(inf[r]["cpu_seconds"] for r in inf) + sum(v["cpu_seconds"] for v in steps.values()) + (ov["cpu_seconds"] if ov else 0.0)
    if wall > 0:
        out["cpu_over_wall"] = cpu / wall
    return out


def hours(seconds):
    return None if seconds is None else seconds / 3600


def plan_rows(cfg: dict, pcfg: dict, trial) -> list:
    """Righe del piano per AttackNet × patience (regola in run_plan.md): 12 addestramenti; epoche minime min(patience + 1,
    12), massime 12; ore di addestramento = 12 × (epoche × s/epoca + s fuori dalle epoche + s della fase 2); valutazione
    (fatta una volta, nella valutazione C5) = (punteggi di test + riproduzioni del dev) × s per 1.000 frame / 1.000, con
    punteggi di test = 3 seed × frame di test dei 4 fold × 2 stati BN e riproduzioni del dev = 12 × frame del dev × 2
    stati; ore di CPU = ore × rapporto CPU / tempo reale della prova. Senza prova le ore restano None (TRIAL_PENDING)."""
    pl = cfg["plan"]
    test = pcfg["schema_expected"]["test_per_fold"]
    n_fit, n_dev = pcfg["arms"][pl["arm"]]["total"], pcfg["dev"]["total"]
    units = len(pl["folds"]) * len(pl["seeds"])
    rows = []
    for n in pl["networks"]:
        r = network_rates(trial["networks"][n], n_fit, n_dev) if trial is not None else {}
        for p in pl["patiences"]:
            ep_min, ep_max = min(p + 1, pl["max_epochs"]), pl["max_epochs"]
            spe, ov, ph2, inf, ratio = (r.get(k) for k in ("seconds_per_epoch", "seconds_outside_epochs", "seconds_phase_2",
                                                            "inference_seconds_per_1000_frames", "cpu_over_wall"))
            t_min = None if None in (spe, ov, ph2) else units * (ep_min * spe + ov + ph2)
            t_max = None if None in (spe, ov, ph2) else units * (ep_max * spe + ov + ph2)
            scores = len(pl["seeds"]) * sum(test) * len(DEF.BN_STATES)
            devrep = units * n_dev * len(DEF.BN_STATES)
            ev = None if inf is None else (scores + devrep) * inf / 1000
            rows.append({"network": n, "patience": p, "units": units, "new_trainings": units, "epochs_min": ep_min, "epochs_max": ep_max,
                         "fitting_frames_per_unit": n_fit, "dev_frames_per_unit": n_dev, "seconds_per_epoch": spe, "seconds_outside_epochs": ov,
                         "seconds_phase_2": ph2, "training_wall_hours_min": hours(t_min), "training_wall_hours_max": hours(t_max),
                         "test_scores": scores, "dev_reproductions": devrep, "evaluation_wall_hours": hours(ev), "cpu_over_wall": ratio,
                         "training_cpu_hours_min": None if None in (t_min, ratio) else hours(t_min * ratio),
                         "training_cpu_hours_max": None if None in (t_max, ratio) else hours(t_max * ratio),
                         "evaluation_cpu_hours": None if None in (ev, ratio) else hours(ev * ratio),
                         "_per_training_max": None if t_max is None else t_max / units})
    return rows


def cap_of(cfg: dict, wall_max_hours, per_training_max_hours):
    """Tetto di un run (regola di plan.caps, scelta dell'autore): (stima massima + (anomaly_factor − 1) × costo massimo
    di un addestramento) × wall_multiplier, arrotondato per eccesso a un multiplo di round_up_hours; None senza stima."""
    ca = cfg["plan"]["caps"]
    if wall_max_hours is None or per_training_max_hours is None:
        return None
    v = (wall_max_hours + (ca["anomaly_factor"] - 1) * per_training_max_hours) * ca["wall_multiplier"]
    return math.ceil(round(v / ca["round_up_hours"], 9)) * ca["round_up_hours"]


def run_caps(cfg: dict, rows: list) -> list:
    """Per ciascuno dei sei run (un'AttackNet a una patience, 12 addestramenti): ore reali minime e massime di
    addestramento con la fase 2 (la valutazione sul test non fa parte dei run), tetto (cap_of) e stima in ore di CPU."""
    out = []
    for r in rows:
        out.append({"run": run_name(r["network"], r["patience"]), "network": r["network"], "patience": r["patience"], "trainings": r["new_trainings"],
                    "wall_hours_min": r["training_wall_hours_min"], "wall_hours_max": r["training_wall_hours_max"],
                    "cap_wall_hours": cap_of(cfg, r["training_wall_hours_max"], hours(r["_per_training_max"]) if r["_per_training_max"] is not None else None),
                    "cpu_hours_min": r["training_cpu_hours_min"], "cpu_hours_max": r["training_cpu_hours_max"]})
    order = {n: i for i, n in enumerate(run_names())}
    return sorted(out, key=lambda c: order[c["run"]])


def num_cell(v, nd: int = 2) -> str:
    """Una cella numerica del piano: TRIAL_PENDING se manca la misura della prova."""
    if v is None:
        return TRIAL_PENDING
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def plan_totals(rows: list, caps: list) -> dict:
    keys = ("training_wall_hours_min", "training_wall_hours_max", "evaluation_wall_hours", "training_cpu_hours_min", "training_cpu_hours_max",
            "evaluation_cpu_hours")
    out = {k: None if any(r[k] is None for r in rows) else sum(r[k] for r in rows) for k in keys}
    out["cap_wall_hours"] = None if any(c["cap_wall_hours"] is None for c in caps) else sum(c["cap_wall_hours"] for c in caps)
    out["test_scores"] = sum(r["test_scores"] for r in rows)
    out["dev_reproductions"] = sum(r["dev_reproductions"] for r in rows)
    return out


def plan_md(cfg: dict, cfg_sha: str, rows: list, caps: list, trial, trial_sha) -> str:
    pl = cfg["plan"]
    t = plan_totals(rows, caps)
    head = (f"Generated from the configuration {cfg_sha[:12]}" + (f" and the trial {trial['run_id']} (trial.json {trial_sha[:12]})." if trial else
            f" without a trial: every hour and cap is {TRIAL_PENDING}."))
    L = ["# Planned runs: AttackNetV1, AttackNetV2_1 and AttackNetV2_2 on NUAA", "", head, "",
         f"72 new trainings ({len(AN.NETWORKS)} AttackNets x {len(pl['folds'])} folds x {len(pl['seeds'])} seeds x {len(pl['patiences'])} patiences, "
         f"36 per patience), arm B, at most {pl['max_epochs']} epochs, in six runs of 12 (`run_matrix.csv`, `run_order.csv`); 48 reused C2 cells "
         "(LivenessNet and ResNet-18, `reference_cells.csv`), not trained and not re-evaluated.", "",
         "Hours are hours of wall time on the machine of the trial; CPU hours use the CPU / wall ratio measured in the trial. Training of a "
         "network at a patience = 12 x (epochs x seconds per epoch + seconds outside the epochs + seconds of phase 2), epochs from "
         "min(patience + 1, 12) to 12; seconds per epoch: mean of the measured epochs, the first included (graph tracing: a cautious "
         "estimate), validation included; phase 2: saving, dev scorer, reloading, ordinary dev, re-estimation, consistency pass, "
         "re-estimated dev, as measured in the trial. Evaluation (done once, by the C5 evaluation, not by the runs): (test scores + dev "
         "reproductions) x inference seconds per 1,000 frames; test scores = 3 seeds x test frames of the 4 folds x 2 BN states; dev "
         "reproductions = 12 x dev frames x 2 BN states. Outside the estimate: loading the inputs (once per process), assembling the test, "
         "loading each checkpoint and installing its statistics in the evaluation. Wall time is elapsed real time (the \"elapsed time\" of C2).", "",
         "| " + " | ".join(PLAN_COLUMNS) + " |", "|" + "---|" * len(PLAN_COLUMNS)]
    for r in rows:
        L.append("| " + " | ".join(num_cell(r[k]) for k in PLAN_COLUMNS) + " |")
    L += ["", "## Totals", "",
          f"- Training, 72 trainings with phase 2: {num_cell(t['training_wall_hours_min'])}-{num_cell(t['training_wall_hours_max'])} hours of wall "
          f"time; {num_cell(t['training_cpu_hours_min'])}-{num_cell(t['training_cpu_hours_max'])} CPU hours.",
          f"- Evaluation (once): {t['test_scores']} test scores and {t['dev_reproductions']} dev reproductions, {num_cell(t['evaluation_wall_hours'])} "
          f"hours of wall time; {num_cell(t['evaluation_cpu_hours'])} CPU hours.", "",
          "## Caps of the six runs (`run_caps.csv`)", "",
          f"Rule: {pl['caps']['rule']}; unit: {pl['caps']['unit']}; anomaly factor {pl['caps']['anomaly_factor']}, multiplier "
          f"{pl['caps']['wall_multiplier']}, rounded up to a multiple of {pl['caps']['round_up_hours']} hours.", "",
          "| " + " | ".join(CAPS_COLUMNS) + " |", "|" + "---|" * len(CAPS_COLUMNS)]
    for c in caps:
        L.append("| " + " | ".join(num_cell(c[k]) for k in CAPS_COLUMNS) + " |")
    L += ["", f"Sum of the six caps: {num_cell(t['cap_wall_hours'])} hours of wall time.", "",
          f"Order: {pl['order']}.", "", f"Machine rule: {pl['machine_rule']}.", "",
          f"Reused cells: {pl['reference']['rule']}.", "", f"Evaluation: {pl['evaluation']}.", ""]
    if trial is not None:
        m = trial["machine"]
        L += [f"Machine of the trial {trial['run_id']}: {m['processor']}, {m['logical_cpus']} logical CPUs, {PAC.ram_text(m)}, {m['os']}.", ""]
    return "\n".join(L)


def plan_texts(cfg: dict, cfg_sha: str, pcfg: dict, cells: list, trial, trial_sha) -> dict:
    rows = plan_rows(cfg, pcfg, trial)
    caps = run_caps(cfg, rows)
    return {"run_matrix.csv": csv_text(MATRIX_COLUMNS, [[str(r[k]) for k in MATRIX_COLUMNS] for r in matrix_rows(cfg, pcfg)]),
            "reference_cells.csv": csv_text(CELL_COLUMNS, [[str(r[k]) for k in CELL_COLUMNS] for r in cells]),
            "run_order.csv": csv_text(ORDER_COLUMNS, [[str(r[k]) for k in ORDER_COLUMNS] for r in order_rows()]),
            "run_plan.csv": csv_text(PLAN_COLUMNS, [[num_cell(r[k]) for k in PLAN_COLUMNS] for r in rows]),
            "run_plan.md": plan_md(cfg, cfg_sha, rows, caps, trial, trial_sha),
            "run_caps.csv": csv_text(CAPS_COLUMNS, [[num_cell(c[k]) for k in CAPS_COLUMNS] for c in caps])}


def load_trial(folder: Path, cfg: dict, cfg_sha: str, code: dict) -> tuple:
    """Il trial.json di una prova di questo codice per plan e check-plan: status.json PASS con il run id ANETTRIAL e lo
    SHA-256 di trial.json; trial.json con lo schema, lo stesso run id e stato, la configurazione di adesso
    (config_sha256) e il codice di adesso (code_sha256, questo file compreso), le tre AttackNet misurate. (trial.json,
    SHA-256)."""
    st = SR.read_json(read_regular(folder / STATUS, "trial status.json"), "trial status.json")
    if st.get("status") != "PASS" or not isinstance(st.get("run_id"), str) or not RUN_ID_RE.match(st["run_id"]):
        fail("trial status.json: not a PASS trial with an ANETTRIAL run id (plan uses only a PASS trial)")
    b = read_regular(folder / TRIAL_JSON, "trial.json")
    if sha256_bytes(b) != st.get("trial_json_sha256"):
        fail("trial.json: SHA-256 differs from trial_json_sha256 of its status.json")
    doc = SR.read_json(b, "trial.json")
    if doc.get("schema") != TRIAL_SCHEMA or doc.get("run_id") != st["run_id"] or doc.get("status") != "PASS":
        fail("trial.json: schema, run id or status differ from its status.json")
    if doc.get("config_sha256") != cfg_sha:
        fail("trial.json: produced with another configuration (config_sha256 differs from the current configuration)")
    if doc.get("code_sha256") != {"architecture": cfg["architectures"]["sha256"], **code}:
        fail("trial.json: produced with other code (code_sha256 differs from the current files)")
    nets = doc.get("networks")
    if not isinstance(nets, dict) or sorted(nets) != sorted(AN.NETWORKS):
        fail(f"trial.json: the networks must be exactly {', '.join(AN.NETWORKS)}")
    PAC.check_machine(doc.get("machine"), "trial.json")
    return doc, sha256_bytes(b)


# ============================================================================ trainer (copia dichiarata)

def anet_trainer(data: dict, training: dict, network: str, train_seed: int, build_model, cap_check, timer) -> tuple:
    """Copia dichiarata di pad_arch_comparison.trial_trainer (C2, congelata) con le sole sostituzioni di
    ANET_TRAINER_SUBSTITUTIONS, per la regola (n) di pad_anet_definitions: l'epoca ripristinata è quella della
    val_loss finita minima (finite_argmin; np.argmin di un NaN dà il primo NaN), None se nessuna è finita; i valori
    non finiti della storia sono scritti come null, con l'elenco (grandezza, prima epoca) in non_finite; la scheda
    dice che il trainer non salva il checkpoint (lo salva chi chiama). Su storie finite scheda e pesi sono quelli
    dell'originale."""
    check_scheduler(training)
    if set(data) != {"X_fit", "y_fit", "X_dev", "y_dev"}:
        fail(f"the trainer receives X_fit, y_fit, X_dev and y_dev, got {sorted(data)}")
    import random
    import tensorflow as tf
    if training["device"] != "CPU":
        fail(f"unknown training device {training['device']!r}")
    devices = hide_gpu(tf)
    tf.keras.backend.clear_session()                 # memoria: grafo del modello precedente liberato (non cambia la ricetta)
    random.seed(train_seed); np.random.seed(train_seed); tf.random.set_seed(train_seed)
    model = build_model()
    Xtr = data["X_fit"].astype(np.float32)
    Xtr /= 255.0
    Xva = data["X_dev"].astype(np.float32)
    Xva /= 255.0
    ytr, yva = np.asarray(data["y_fit"]), np.asarray(data["y_dev"])
    if len(ytr) == 0 or len(yva) == 0 or (ytr == 0).sum() == 0 or (ytr == 1).sum() == 0:
        fail("fitting and dev must be non-empty and the fitting set must hold both classes")
    Ytr, Yva = np.eye(2)[ytr], np.eye(2)[yva]           # one-hot: indice 0 bona fide, 1 attacco
    lr, batch, epochs, patience = training["initial_learning_rate"], training["batch"], training["max_epochs"], training["early_stopping_patience"]
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr, clipnorm=1.0),
                  loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")])
    cw = {0: len(ytr) / (2 * (ytr == 0).sum()), 1: len(ytr) / (2 * (ytr == 1).sum())}
    aug = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.03), tf.keras.layers.RandomZoom(0.12, 0.12),
                               tf.keras.layers.RandomTranslation(0.06, 0.06), tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
                               tf.keras.layers.RandomContrast(0.2), tf.keras.layers.GaussianNoise(0.02)])
    ds_tr = tf.data.Dataset.from_tensor_slices((Xtr, Ytr)).shuffle(4096, seed=train_seed).batch(batch).map(lambda x, y: (aug(x, training=True), y)).prefetch(2)
    ds_va = tf.data.Dataset.from_tensor_slices((Xva, Yva)).batch(batch)
    cbs = training_callbacks(tf, training, cap_check)
    cbs.insert(len(cbs) - 1, timer.callback(tf))      # tempi e storia per epoca, prima di Cap
    t0 = time.time()
    hist = model.fit(ds_tr, validation_data=ds_va, epochs=epochs, class_weight=cw, callbacks=cbs, verbose=2)
    train_s = time.time() - t0
    es = cbs[0]
    h = hist.history
    del ds_tr, ds_va, Xtr, Xva
    gc.collect()
    best = finite_argmin(h["val_loss"])
    if es.stopped_epoch == 0 and es.best_weights is not None:
        model.set_weights(es.best_weights)           # epoche finite prima dell'early stopping: pesi migliori anche qui
    card = {"arch": network, "train_seed": int(train_seed), "n_fitting": int(len(ytr)), "n_dev": int(len(yva)),
            "max_epochs": int(epochs), "epochs_run": len(h["loss"]), "restored_epoch": None if best is None else best + 1,
            "restored_by": "early stopping" if es.stopped_epoch else "end of epochs (set_weights)",
            "best_val_loss": None if best is None else float(h["val_loss"][best]),
            "val_acc_at_best": None if best is None else finite_or_none(h["val_accuracy"][best]),
            "history": {k: [finite_or_none(v) for v in h[k]] for k in ("loss", "accuracy", "val_loss", "val_accuracy", "lr") if k in h},
            "non_finite": non_finite_entries(h),
            "device": "CPU", "visible_devices": devices, "lr": lr, "batch": batch, "early_stopping_patience": patience,
            "scheduler": training["scheduler"], "class_weight": {str(k): float(v) for k, v in cw.items()}, "train_seconds": round(train_s, 1),
            "test_evaluation": "none (no test frame is assembled in the trial)", "cpu_nondeterminism": "declared (N1): parallel reductions on the CPU "
            "may give minimal differences between two runs with the same seed", "checkpoint": "none saved by the trainer (the caller saves it)"}
    print(f"== {network} seed {train_seed}: {card['epochs_run']} epochs in {train_s:.0f}s, restored epoch {card['restored_epoch']} "
          f"(val_loss {'none finite' if card['best_val_loss'] is None else format(card['best_val_loss'], '.6f')})", flush=True)
    return card, model


# ============================================================================ prova tecnica

def check_hooks(official: bool) -> dict:
    """Gli agganci dei test (HOOKS: tutte le variabili di modulo che valgono None nei moduli del progetto importati) e i
    valori che solo i test sostituiscono (PINNED_VALUES): con la configurazione consegnata gli agganci devono essere tutti
    None e i valori quelli dell'import, altrimenti Failure prima di ogni lavoro; restituisce quelli impostati o cambiati
    (registrati in status.json e trial.json, stampati da check-inputs)."""
    mods = {m: sys.modules[__name__] if m == "pad_anet_comparison" else sys.modules[m] for m in HOOKS}
    setv = sorted(f"{m}.{k}" for m, names in HOOKS.items() for k in names if getattr(mods[m], k) is not None)
    changed = sorted(f"{m}.{k}" for (m, k), v in PINNED_VALUES.items() if getattr(sys.modules[m], k) is not v)
    if official and (setv or changed):
        fail(f"test hooks are set or test values changed ({', '.join(setv + changed)}): the official commands run with every hook None")
    return {"official_configuration": official, "set": setv, "changed": changed}


def is_official(cfg_p: Path) -> bool:
    """La configurazione consegnata, riconosciuta per identità del file (percorsi risolti, collegamenti e «..» compresi),
    non dalla stringa del percorso."""
    try:
        return os.path.samefile(cfg_p, ROOT / DEFAULT_CONFIG)
    except OSError:
        return False


def machine_comparison(c2: dict) -> dict:
    """La macchina e le versioni di adesso confrontate campo per campo con il record della prova C2 (la regola di
    pad_arch_run.machine_rule, R.MACHINE_FIELDS e R.VERSION_FIELDS: uguali solo se presenti e non nulli in entrambi e
    identici), senza fallire: l'esito entra nel controllo LivenessNet (trial_control)."""
    m_now, e_now = (MACHINE or PAC.machine)(), FR.environment()
    tm, te = c2.get("machine") or {}, c2.get("environment") or {}
    rec = {"c2_trial": c2.get("run_id"),
           "machine": {f: {"c2_trial": tm.get(f), "now": m_now.get(f), "equal": R._same(tm, m_now, f)} for f in R.MACHINE_FIELDS},
           "versions": {f: {"c2_trial": te.get(f), "now": e_now.get(f), "equal": R._same(te, e_now, f)} for f in R.VERSION_FIELDS}}
    rec["differing"] = [f for part in ("machine", "versions") for f, v in rec[part].items() if not v["equal"]]
    rec["equal"] = not rec["differing"]
    return rec


def previous_attempts(out_root: Path, ck_root: Path) -> list:
    """I tentativi precedenti della prova: le voci di out_root e della radice dei checkpoint con la forma di un run id
    ANETTRIAL (qualunque data e commit), ciascuna con lo stato del suo status.json (o «no status.json»). Una voce che
    somiglia a un run id ma non ne ha la forma ferma il comando."""
    names = set()
    for root in (out_root, ck_root):
        if root.is_dir():
            for e in os.listdir(root):
                if LOOKS_LIKE_RUN.match(e):
                    if not RUN_ID_RE.match(e):
                        fail(f"{e!r} looks like a trial run id but does not have its form: rename or archive it before a new attempt")
                    names.add(e)
    out = []
    for rid in sorted(names):
        st = "no status.json"
        p = out_root / rid / STATUS
        if p.is_file() and not p.is_symlink():
            try:
                st = json.loads(p.read_text(encoding="utf-8")).get("status") or "no status"
            except (ValueError, OSError):
                st = "unreadable status.json"
        out.append({"run_id": rid, "status": st})
    return out


def attempt_rule(out_root: Path, ck_root: Path, attempt) -> dict:
    """Regola dei tentativi: --attempt deve essere il numero dei tentativi precedenti + 1 (ogni tentativo conserva il suo
    esito: nessuna cartella si riusa o si toglie); altrimenti Failure prima di creare cartelle."""
    prev = previous_attempts(out_root, ck_root)
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt != len(prev) + 1:
        fail(f"--attempt must be {len(prev) + 1}: {len(prev)} earlier attempts found ({', '.join(p['run_id'] + ' ' + p['status'] for p in prev) or 'none'})")
    return {"attempt": attempt, "previous_attempts": prev}


def build_checked(cfg: dict, net: str, arch):
    """Il modello di un'AttackNet costruito con il registro (AN.build: create_model con i suoi default) e controllato
    (AN.check_model) contro la configurazione, prima del training."""
    shape = cfg["architectures"]["input_shape"]
    a = cfg["architectures"]["networks"][net]
    try:
        model = AN.build(net, shape, arch)
        AN.check_model(model, net, shape, {"trainable": a["trainable_parameters"], "non_trainable": a["non_trainable_parameters"],
                                           "batchnorm_layers": a["batchnorm_layers"]})
    except AN.ModelProblem as e:
        fail(str(e))
    return model


def check_loaded(model, cfg: dict, net: str) -> dict:
    """Un checkpoint ricaricato contro il registro (la funzione di pad_arch_run.check_loaded sul registro nuovo)."""
    a = cfg["architectures"]["networks"][net]
    try:
        return AN.check_model(model, net, cfg["architectures"]["input_shape"], {"trainable": a["trainable_parameters"],
                                                                                "non_trainable": a["non_trainable_parameters"],
                                                                                "batchnorm_layers": a["batchnorm_layers"]})
    except AN.ModelProblem as e:
        fail(str(e))


class NotEvaluable(Exception):
    """Un modello non valutabile per la regola (n) (pad_anet_definitions): il run continua, il modello non è valutato."""


def step(rec: dict, name: str, t0: float, c0: float) -> None:
    rec["steps"][name] = {"seconds": time.monotonic() - t0, "cpu_seconds": time.process_time() - c0}


def clock() -> tuple:
    return time.monotonic(), time.process_time()


def forward_checked(model, part: dict, batch: int, what: str, state: str, layers=()) -> tuple:
    """BR.evaluate con il controllo della regola (n) prima del record: punteggi o logit non finiti -> NotEvaluable."""
    f = BR.forward(model, part["X"], batch, layers=layers, scores=True)
    if not DEF.arrays_finite([f["p"], f["z"]]):
        raise NotEvaluable(f"{DEF.NOT_EVALUABLE}: non-finite scores or logits on the {what} ({state} BN)")
    return BR.eval_record(f["p"], f["z"], part["y"], f"{what} {state}"), f["moments"]


def eer_or_none(y, p_att):
    oc = D.one_class_run(y, p_att)
    return D.condition_threshold("dev_eer", y, p_att) if oc["evaluable"] else None


def max_abs(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def bn_check(net: str, n_fit: int, order: list, orig, copy, stats: dict, cons: dict, snap: dict, ev_ord: dict, dev_after: dict, cfg: dict,
             bn_cfg: dict) -> dict:
    """«Verifichi la ristima di tutte le BN»: cinque controlli, ciascuno registrato; uno fallito -> Failure (prova FAIL).
      1. gli strati BatchNormalization letti dal grafo (AM.model_counts) sono i nomi del registro e della configurazione
         e coincidono con l'ordine della ristima; pad_arch_run.bn_order_problems vuoto; le statistiche mobili delle BN
         del livello superiore sono tutte le variabili non addestrabili (544 o 448: nessuna BN fuori);
      2. valori per canale accumulati = frame × H × W del registro, canali del registro (BR.recalibrate fallisce già se il
         conteggio non è frame × H × W: controllo esistente, ripetuto qui contro il registro);
      3. ogni moving_mean e moving_variance dei cinque strati diverso dall'originale e uguale sui byte al float32
         installato, ogni altro tensore identico sui byte, originale invariato (BR.tensor_problems, già in pr27_checks);
      4. coerenza entro consistency_rel e consistency_abs e softmax dei logit entro la tolleranza (pr27_checks, già
         passati: qui registrati);
      5. statistiche finite e punteggi del dev diversi fra i due stati (scarto massimo > 0): lo stato valutato è quello
         ristimato."""
    counts = AM.model_counts(orig)
    want = cfg["architectures"]["networks"][net]["batchnorm_names"]
    c1 = {"graph": counts["batchnorm_names"], "registry": AN.bn_order(net), "configuration": want, "order": list(order),
          "order_problems": R.bn_order_problems(list(order), AM.model_topology(orig)), "moving_values": sum(AN.moving_sizes(orig).values()),
          "non_trainable": counts["non_trainable"]}
    c1["pass"] = c1["graph"] == c1["registry"] == c1["configuration"] == c1["order"] and not c1["order_problems"] \
        and c1["moving_values"] == c1["non_trainable"] == cfg["architectures"]["networks"][net]["non_trainable_parameters"]
    shape = cfg["architectures"]["input_shape"]
    exp = AN.expected_recalibration_counts(net, n_fit, shape)
    chans = {b["name"]: b["channels"] for b in AN.batchnorm_layers(net, shape)}
    layers = {}
    for n in order:
        s = stats[n]
        layers[n] = {"kind": "conv" if s["spatial"] > 1 else "dense", "channels": int(len(s["mean64"])), "channels_registry": chans[n],
                     "count": int(s["count"]), "count_expected": exp[n],
                     "moving_mean_changed": np.asarray(s["mean32"], np.float32).tobytes() != np.asarray(s["old_mean"], np.float32).tobytes(),
                     "moving_variance_changed": np.asarray(s["var32"], np.float32).tobytes() != np.asarray(s["old_var"], np.float32).tobytes(),
                     "max_abs_change_mean": max_abs(s["mean32"], s["old_mean"]), "max_abs_change_variance": max_abs(s["var32"], s["old_var"]),
                     "consistency_max_abs_diff_mean": cons[n]["max_abs_diff_mean"], "consistency_max_abs_diff_variance": cons[n]["max_abs_diff_var"],
                     "consistency_channels_outside": cons[n]["channels_outside_tolerance"],
                     "statistics_finite": DEF.arrays_finite([s["mean64"], s["var64"], s["mean32"], s["var32"]]),
                     "change_summary": BR.change_summary(s, bn_cfg["batchnorm"]["epsilon"])}
    c2 = all(v["count"] == v["count_expected"] and v["channels"] == v["channels_registry"] for v in layers.values())
    c3 = all(v["moving_mean_changed"] and v["moving_variance_changed"] for v in layers.values()) \
        and not BR.tensor_problems(snap, BR.snapshot_weights(copy), BR.moving_keys(copy, order), stats) and BR.snapshot_weights(orig) == snap
    c4 = all(v["consistency_channels_outside"] == 0 for v in layers.values())
    dev_diff = max_abs(ev_ord["p"][:, 1], dev_after["p"][:, 1])
    c5 = all(v["statistics_finite"] for v in layers.values()) and dev_diff > 0
    checks = {"1_bn_layers_from_the_graph": c1, "2_values_per_channel": {"pass": c2}, "3_only_moving_tensors_changed": {"pass": c3},
              "4_consistency_and_softmax": {"pass": c4, "softmax_gap_ordinary": ev_ord["softmax_gap"], "softmax_gap_re_estimated": dev_after["softmax_gap"]},
              "5_states_differ": {"pass": c5, "dev_max_abs_score_difference": dev_diff}}
    passed = all(v["pass"] for v in checks.values())
    if not passed:
        fail(f"{net}: BN check failed: {', '.join(k for k, v in checks.items() if not v['pass'])}")
    return {"layers": layers, "checks": checks, "passed": True}


def trial_control(ctx, rec: dict) -> None:
    """Il controllo LivenessNet (non contato fra le tre prove): 3 epoche con anet_trainer e il costruttore di C2
    (PAC.build_checked); storia contro la scheda accettata (PAC.baseline_check, scarto 0) e contro la prova C2
    (PAC.history_differences sulle prime 3 epoche, scarto 0); macchina e versioni contro la prova C2 (machine_comparison).
    Esito «reproduced» solo se le due storie sono identiche e macchina e versioni uguali."""
    tf = ctx.tf
    ctx.budget.check(f"train {CONTROL}")
    rec["threads_before"] = PAC.check_threads(tf, ctx.cfg_pr, f"before {CONTROL}")
    epochs = []
    timer = PAC.EpochTimer(epochs)
    t0, c0 = clock()
    card, model = (TRAINER or anet_trainer)(ctx.data, ctx.training, CONTROL, TRIAL_UNIT["seed"], lambda: PAC.build_checked(ctx.cfg_pr, CONTROL, ctx.arch),
                                            lambda s: ctx.budget.check(f"{s} of {CONTROL}"), timer)
    rec["train_seconds"], rec["train_cpu_seconds"] = time.monotonic() - t0, time.process_time() - c0
    rec["epochs"] = [{k: (finite_or_none(v) if k in HISTORY_KEYS else v) for k, v in e.items()} for e in epochs]
    rec["card"] = {k: card[k] for k in ("arch", "epochs_run", "restored_epoch", "best_val_loss", "non_finite", "lr", "batch", "early_stopping_patience")}
    if any(e[k] is None for e in rec["epochs"] for k in HISTORY_KEYS):
        rec["outcome"] = "not reproduced"
        rec["reason"] = "non-finite value in the LivenessNet history"
    else:
        accepted = PAC.baseline_check(ctx.cfg_pr, ctx.inp.reference_card, {"epochs": rec["epochs"]})
        c2 = ctx.inp.c2_trial["networks"][CONTROL]["epochs"][:TRIAL_EPOCHS]
        diffs = {k: PAC.history_differences([e[k] for e in c2], [e[k] for e in rec["epochs"][:TRIAL_EPOCHS]]) for k in HISTORY_KEYS}
        c2_same = len(rec["epochs"]) >= TRIAL_EPOCHS and all(d["epochs_compared"] == TRIAL_EPOCHS and d["first_differing_epoch"] is None for d in diffs.values())
        rec["accepted_card"] = accepted
        rec["c2_trial"] = {"run_id": ctx.inp.c2_trial["run_id"], "fields": diffs, "outcome": "reproduced" if c2_same else "not reproduced"}
        rec["machine_and_versions"] = machine_comparison(ctx.inp.c2_trial)
        rec["outcome"] = "reproduced" if accepted["outcome"] == "reproduced" and c2_same and rec["machine_and_versions"]["equal"] else "not reproduced"
    print(f"   {CONTROL} control: {rec['outcome']}", flush=True)
    # l'esito è noto: registrato subito, prima del controllo dei thread «after» (un suo FAIL non lo cancella)
    if rec["outcome"] != "reproduced":
        ctx.record_finding("control", rec.get("reason") or (
            f"accepted card {rec['accepted_card']['outcome']}, C2 trial {rec['c2_trial']['outcome']}, machine and versions "
            + ("equal" if rec["machine_and_versions"]["equal"] else "differing: " + ", ".join(rec["machine_and_versions"]["differing"]))))
    rec["threads_after"] = PAC.check_threads(tf, ctx.cfg_pr, f"after {CONTROL}")
    del model
    tf.keras.backend.clear_session()
    gc.collect()


def trial_attacknet(ctx, net: str, rec: dict) -> None:
    """Una delle tre prove (vedi il docstring del modulo, passo 2); rec è riempito man mano. Un modello non valutabile
    per la regola (n) registra il motivo e non continua; ogni altro difetto -> Failure."""
    tf, cfg, bn_cfg = ctx.tf, ctx.cfg, ctx.bn_cfg
    batch, ibatch = cfg["trial"]["estimator_batch"], cfg["trial"]["inference_batch"]
    order = cfg["architectures"]["networks"][net]["batchnorm_names"]
    name = f"{net}_fold{TRIAL_UNIT['fold']}_seed{TRIAL_UNIT['seed']}"
    ctx.budget.check(f"train {net}")
    rec["threads_before"] = PAC.check_threads(tf, ctx.cfg_pr, f"before {net}")
    epochs = []
    timer = PAC.EpochTimer(epochs)
    t0, c0 = clock()
    card, model = (TRAINER or anet_trainer)(ctx.data, ctx.training, net, TRIAL_UNIT["seed"], lambda: build_checked(cfg, net, ctx.arch),
                                            lambda s: ctx.budget.check(f"{s} of {net}"), timer)
    rec["train_seconds"], rec["train_cpu_seconds"] = time.monotonic() - t0, time.process_time() - c0
    rec["epochs"] = [{k: (finite_or_none(v) if k in HISTORY_KEYS else v) for k, v in e.items()} for e in epochs]
    if [{k: e[k] for k in HISTORY_KEYS} for e in rec["epochs"]] != [{k: card["history"][k][i] for k in HISTORY_KEYS} for i in range(card["epochs_run"])]:
        fail(f"{net}: the history recorded by the timing callback differs from the Keras history")
    rec["training_overhead"] = {"seconds": rec["train_seconds"] - sum(e["seconds_total"] for e in epochs),
                                "cpu_seconds": rec["train_cpu_seconds"] - sum(e["cpu_seconds"] for e in epochs)}
    rec["card"] = {k: card[k] for k in ("arch", "train_seed", "n_fitting", "n_dev", "max_epochs", "epochs_run", "restored_epoch", "restored_by",
                                        "best_val_loss", "non_finite", "device", "visible_devices", "lr", "batch", "early_stopping_patience",
                                        "scheduler", "class_weight", "train_seconds", "checkpoint")}
    counts = AM.model_counts(model)
    rec.update({"parameters": {"trainable": counts["trainable"], "non_trainable": counts["non_trainable"]}, "batchnorm_layers": counts["batchnorm_layers"],
                "steps_per_epoch": math.ceil(card["n_fitting"] / card["batch"])})
    weights_finite = DEF.arrays_finite(model.get_weights()) if card["restored_epoch"] is not None else None
    out = DEF.training_outcome(card["history"], weights_finite)
    rec["outcome"] = out
    try:
        if not out["evaluable"]:
            raise NotEvaluable(out["reason"])
        mem, _ = forward_checked(model, ctx.dev, batch, "in-memory dev", "ordinary")
        for role, part in (("fitting", ctx.fit), ("dev", ctx.dev)):
            ctx.budget.check(f"inference on the {role} of {net}")
            try:
                rec["inference"][role] = PAC.inference_pass(model, part["X"], ibatch)
            except Failure as e:
                if str(e).startswith("inference pass: probabilities non-finite"):
                    # il messaggio unico di PAC.inference_pass copre anche «outside [0, 1] or of the wrong shape»: tutti non valutabili
                    # (la forma è già garantita da AN.check_model), dichiarato nella regola (n)
                    raise NotEvaluable(f"{DEF.NOT_EVALUABLE}: probabilities non-finite, outside [0, 1] or of the wrong shape on the {role} "
                                       "in the timed inference pass (ordinary BN)") from None
                raise
        snap_mem = BR.snapshot_weights(model)
        path = ctx.ckpt / f"{name}.keras"
        t0, c0 = clock()
        R.save_checkpoint(model, path)
        sha = sha256_file(path)
        step(rec, "save", t0, c0)
        rec["checkpoint"] = {"file": path.name, "sha256": sha, "location": "outside the repository (unofficial output, not published)"}
        del model
        tf.keras.backend.clear_session()
        gc.collect()
        ctx.budget.check(f"scorer {net}")
        t0, c0 = clock()
        p_scorer = FR.probabilities((R.MAKE_SCORER or FR.tf_scorer)(path, sha), ctx.dev["X"])
        step(rec, "scorer", t0, c0)
        t0, c0 = clock()
        orig = BR.load_model(path, sha)
        check_loaded(orig, cfg, net)
        snap = BR.snapshot_weights(orig)
        step(rec, "reload", t0, c0)
        ctx.budget.check(f"ordinary dev {net}")
        t0, c0 = clock()
        ev_ord, _ = forward_checked(orig, ctx.dev, batch, "dev", "ordinary")
        step(rec, "ordinary_dev", t0, c0)
        if ev_ord["softmax_gap"] > BR.tol(bn_cfg, "softmax_of_logits_max_abs_diff"):
            fail(f"{net}: the softmax of the logits differs from the ordinary dev probabilities by {ev_ord['softmax_gap']:.2e}")
        thr = eer_or_none(ctx.dev["y"], ev_ord["p"][:, 1])
        rec["reload"] = {"weights_equal_bytes": snap == snap_mem, "checked_against_registry": True,
                         "agreement_in_memory": R.evaluator_agreement(mem["p"][:, 1], ev_ord["p"][:, 1], thr, bn_cfg)}
        rec["scorer_agreement"] = R.evaluator_agreement(p_scorer[:, 1], ev_ord["p"][:, 1], thr, bn_cfg)
        if not rec["reload"]["weights_equal_bytes"]:
            fail(f"{net}: the reloaded weights differ from the weights in memory")
        if not rec["reload"]["agreement_in_memory"]["pass"] or not rec["scorer_agreement"]["pass"]:
            fail(f"{net}: the dev scores of the reloaded model, of the scorer and of the evaluator do not agree")
        copy = tf.keras.models.clone_model(orig)
        copy.set_weights(orig.get_weights())
        if BR.snapshot_weights(copy) != snap:
            fail(f"{net}: the in-memory copy differs from the reloaded model before the re-estimation")
        t0, c0 = clock()
        try:
            stats = R.reestimate(copy, ctx.fit, order, batch, ctx.budget, name)
        except Failure as e:
            if str(e).endswith("the re-estimated mean or variance is not finite"):
                raise NotEvaluable(f"{DEF.NOT_EVALUABLE}: non-finite re-estimated statistics") from None
            raise
        step(rec, "reestimate", t0, c0)
        if BR.snapshot_weights(orig) != snap:
            fail(f"{net}: the reloaded model changed during the re-estimation")
        ctx.budget.check(f"consistency pass {net}")
        t0, c0 = clock()
        fit_after, final = forward_checked(copy, ctx.fit, batch, "fitting", "re-estimated", layers=order)
        step(rec, "consistency", t0, c0)
        ctx.budget.check(f"re-estimated dev {net}")
        t0, c0 = clock()
        dev_after, _ = forward_checked(copy, ctx.dev, batch, "dev", "re-estimated")
        step(rec, "re_estimated_dev", t0, c0)
        cons = R.pr27_checks(name, stats, snap, copy, order, fit_after, final, dev_after, bn_cfg)
        t0, c0 = clock()
        inst = tf.keras.models.clone_model(orig)
        inst.set_weights(orig.get_weights())
        installed = BT.install(inst, {n: (stats[n]["mean32"], stats[n]["var32"]) for n in order})
        probs = BR.tensor_problems(snap, BR.snapshot_weights(inst), BR.moving_keys(inst, order), installed)
        same = BR.snapshot_weights(inst) == BR.snapshot_weights(copy)
        ev_inst, _ = forward_checked(inst, ctx.dev, batch, "dev", "re-estimated")
        step(rec, "installed_copy", t0, c0)
        thr_re = eer_or_none(ctx.dev["y"], dev_after["p"][:, 1])
        rec["installed_copy"] = {"tensor_problems": probs, "equal_bytes_to_re_estimated_copy": same,
                                 "max_abs_dev_score_difference": max_abs(ev_inst["p"][:, 1], dev_after["p"][:, 1]),
                                 "agreement": R.evaluator_agreement(dev_after["p"][:, 1], ev_inst["p"][:, 1], thr_re, bn_cfg)}
        if probs or not same or not rec["installed_copy"]["agreement"]["pass"]:
            fail(f"{net}: the copy with the installed statistics (the path of the test) differs from the re-estimated copy")
        rec["bn_check"] = bn_check(net, len(ctx.fit["y"]), order, orig, copy, stats, cons, snap, ev_ord, dev_after, cfg, bn_cfg)
        rec["dev"] = {s: {"accuracy_05": r["accuracy_05"], "val_loss": r["val_loss"], "auroc": r["auroc"], "n": r["n"]}
                      for s, r in (("ordinary", ev_ord), ("re-estimated", dev_after))}
        del orig, copy, inst, final
    except NotEvaluable as e:
        rec["outcome"] = {**rec["outcome"], "evaluable": False, "reason": str(e)}
        print(f"   {net}: {e}", flush=True)
        ctx.record_finding(net, rec["outcome"]["reason"])           # subito, prima del controllo dei thread «after»
    rec["threads_after"] = PAC.check_threads(tf, ctx.cfg_pr, f"after {net}")
    rec["peak_rss_mb_after"] = round(SR.peak_rss_mb(), 1)
    tf.keras.backend.clear_session()
    gc.collect()


def projection(cfg: dict, pcfg: dict, doc: dict) -> dict:
    """Le ore previste dalla prova, con le funzioni del piano (plan_rows, run_caps, plan_totals)."""
    if any(not (doc["networks"][n].get("outcome") or {}).get("evaluable") or not doc["networks"][n].get("bn_check") for n in AN.NETWORKS):
        return {"available": False, "reason": "not every AttackNet completed the trial"}
    rows = plan_rows(cfg, pcfg, doc)
    caps = run_caps(cfg, rows)
    return {"available": True, "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows], "caps": caps,
            "totals": plan_totals(rows, caps)}


def fmt(v, nd: int = 1) -> str:
    return "" if v is None else f"{v:.{nd}f}"


def trial_md(doc: dict) -> str:
    u = doc["unit"]
    L = [f"# Technical trial {doc['run_id']}: AttackNetV1, AttackNetV2_1, AttackNetV2_2", "",
         f"Status: **{doc['status']}**. NUAA, arm B, fold {u['fold']}, seed {u['seed']}: fitting {u['frames']['fitting']['n']} frames, dev "
         f"{u['frames']['dev']['n']} frames; no test frame assembled. {doc['epochs_fixed']} epochs, patience {doc['patience']}, batch {doc['batch']}, "
         f"device {doc['device']}. Wall-time cap {doc['cap_minutes']} {doc['cap_unit']}.", "",
         f"Started {doc['started']}, finished {doc['finished']}; wall time {doc['wall_seconds']:.0f} s, process CPU time (all threads) "
         f"{doc['process_cpu_seconds']:.0f} s; peak RSS {doc['peak_rss_mb']:.0f} MB. Attempt {doc['attempt']}; earlier attempts: "
         + (", ".join(f"{p['run_id']} ({p['status']})" for p in doc["previous_attempts"]) or "none") + ".", "",
         f"Machine: {doc['machine']['processor']}, cores as named by the system: {PAC.core_levels_text(doc['machine'])}; "
         f"{doc['machine']['logical_cpus']} logical CPUs, {PAC.ram_text(doc['machine'])}, {doc['machine']['os']}; TensorFlow "
         f"{doc['environment'].get('tensorflow_installed')}, Keras {doc['environment'].get('keras')}, Python {doc['environment'].get('python')}, "
         f"NumPy {doc['environment'].get('numpy')}.", "", PAC.threads_text(doc["threads"]), ""]
    if doc["stopped"]:
        L += [f"Stopped: {doc['stopped']['cause']} at {doc['stopped']['step']} ({doc['stopped']['elapsed_minutes']} min of wall time).", ""]
    c = doc["control"]
    L += ["## LivenessNet control (not one of the three trials)", "", f"Outcome: **{c.get('outcome', 'not run')}**. Rule: {doc['control_rule']}", ""]
    if c.get("accepted_card"):
        L += [f"- against the accepted card ({c['accepted_card']['model']} of {c['accepted_card']['reference_run']}): {c['accepted_card']['outcome']}",
              f"- against the C2 trial {c['c2_trial']['run_id']}: {c['c2_trial']['outcome']}",
              f"- machine and versions against the C2 trial: " + ("equal" if c["machine_and_versions"]["equal"] else
                                                                  "differing: " + ", ".join(c["machine_and_versions"]["differing"])), ""]
    L += ["## Epochs", "", "| network | epoch | loss | accuracy | val_loss | val_accuracy | lr | seconds | validation seconds | CPU seconds |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for net, rec in [(CONTROL, c)] + list(doc["networks"].items()):
        for e in rec.get("epochs", []):
            vals = [("none" if e[k] is None else (f"{e[k]:.6g}" if k == "lr" else f"{e[k]:.6f}")) for k in HISTORY_KEYS]
            L.append(f"| {net} | {e['epoch']} | " + " | ".join(vals) + f" | {e['seconds_total']:.1f} | {e['seconds_validation']:.1f} | {e['cpu_seconds']:.1f} |")
    L += ["", "The histories are technical outputs of the trial, not results.", "", "## Outcome, BN check and timings per AttackNet", "",
          "| network | evaluable | trainable parameters | BN layers | BN check | s per epoch | s outside the epochs | s of phase 2 | "
          "inference s per 1,000 frames | CPU / wall |", "|---|---|---|---|---|---|---|---|---|---|"]
    for net, rec in doc["networks"].items():
        r = rec.get("rates") or {}
        o = rec.get("outcome") or {}
        L.append(f"| {net} | {'yes' if o.get('evaluable') else (o.get('reason') or 'not run')} | {rec.get('parameters', {}).get('trainable', '')} | "
                 f"{rec.get('batchnorm_layers', '')} | {'passed' if (rec.get('bn_check') or {}).get('passed') else 'not passed'} | "
                 f"{fmt(r.get('seconds_per_epoch'))} | {fmt(r.get('seconds_outside_epochs'))} | {fmt(r.get('seconds_phase_2'))} | "
                 f"{fmt(r.get('inference_seconds_per_1000_frames'))} | {fmt(r.get('cpu_over_wall'), 2)} |")
    L += ["", "Save, reload and copies (seconds; largest absolute differences of the dev attack scores; agreement: at most 1e-5 and no discordant "
          "decision at 0.5 and at the EER threshold, the rule of PR #27):", "",
          "| network | inference fitting s | inference dev s | " + " | ".join(f"{k} s" for k in C_STEPS) + " | reloaded weights equal | reloaded vs memory | "
          "scorer vs evaluator | installed copy vs re-estimated copy |", "|" + "---|" * (len(C_STEPS) + 7)]
    for net, rec in doc["networks"].items():
        st, inf = rec.get("steps") or {}, rec.get("inference") or {}
        ag = [((rec.get("reload") or {}).get("agreement_in_memory") or {}).get("max_abs_diff"), (rec.get("scorer_agreement") or {}).get("max_abs_diff"),
              (rec.get("installed_copy") or {}).get("max_abs_dev_score_difference")]
        L.append(f"| {net} | {fmt((inf.get('fitting') or {}).get('seconds'))} | {fmt((inf.get('dev') or {}).get('seconds'))} | "
                 + " | ".join(fmt((st.get(k) or {}).get("seconds")) for k in C_STEPS)
                 + f" | {(rec.get('reload') or {}).get('weights_equal_bytes', '')} | " + " | ".join("" if v is None else f"{v:.3g}" for v in ag) + " |")
    L += ["", "BN check per layer (values per channel counted against frames x H x W of the registry; changes of the installed float32 "
          "statistics against the original ones):", "",
          "| network | layer | channels | values per channel | expected | mean changed | variance changed | largest change of the mean | "
          "largest change of the variance |", "|---|---|---|---|---|---|---|---|---|"]
    for net, rec in doc["networks"].items():
        for n, v in ((rec.get("bn_check") or {}).get("layers") or {}).items():
            L.append(f"| {net} | {n} | {v['channels']} | {v['count']} | {v['count_expected']} | {v['moving_mean_changed']} | {v['moving_variance_changed']} | "
                     f"{v['max_abs_change_mean']:.3g} | {v['max_abs_change_variance']:.3g} |")
    pj = doc["projection"]
    L += ["", "## Hours of the plan from this trial", ""]
    if pj["available"]:
        t = pj["totals"]
        L += [f"Training of the 72 trainings with phase 2: {t['training_wall_hours_min']:.2f}-{t['training_wall_hours_max']:.2f} hours of wall time, "
              f"{t['training_cpu_hours_min']:.2f}-{t['training_cpu_hours_max']:.2f} CPU hours; evaluation (once): {t['evaluation_wall_hours']:.2f} hours "
              f"of wall time, {t['evaluation_cpu_hours']:.2f} CPU hours; caps of the six runs: "
              + ", ".join(f"{cc['run']} {cc['cap_wall_hours']:.1f} h" for cc in pj["caps"]) + ". Rule: `plan/run_plan.md`."]
    else:
        L += [f"Not available: {pj['reason']}."]
    L += ["", f"Checkpoints: {doc['checkpoints']['location']}; SHA-256: " + ", ".join(f"{k} {v}" for k, v in doc["checkpoints"]["sha256"].items()) + ".",
          "", f"Command: `{doc['command']}`", ""]
    return "\n".join(L)


def repo_state(root: Path) -> dict:
    return REPO_STATE(root) if REPO_STATE is not None else SR.repo_state(root)


def resolve(a) -> tuple:
    base = Path(a.input_root).resolve() if a.input_root else ROOT
    cfg_p = Path(a.config) if Path(a.config).is_absolute() else ROOT / a.config
    return base, cfg_p


def input_paths(cfg: dict, cfg_pr: dict, pcfg: dict, base: Path, cfg_p: Path) -> dict:
    out = PAC.input_paths(cfg_pr, pcfg, base, cfg_p)
    out.update({"the C2 pre-registration": rel_path(base, cfg["preregistration"]["path"]), "the configuration of PR #27":
                rel_path(base, cfg["bn_recalibration_config"]["path"]), "the C2 trial": rel_path(base, cfg["c2_trial"]["folder"]),
                "the C2 evaluation status": rel_path(base, cfg["c2_results"]["status"]["path"]),
                "the C2 evaluation results": rel_path(base, cfg["c2_results"]["results"]["path"])})
    return out


def run_dir_problem(run: Path, out_root: Path, inputs: dict) -> str:
    """Motivo per cui la cartella della prova non è ammessa ("" se lo è): radice non collegamento simbolico, ingressi
    separati, dentro il repository solo sotto results/anetcmp/."""
    if out_root.is_symlink():
        return "the output root is a symbolic link"
    for label, p in inputs.items():
        if pg.within(p, run) or (Path(p).is_dir() and pg.within(run, p)):
            return f"{label} {Path(p).name} is inside the run folder, or contains it: inputs and outputs must be separate"
    return mw.output_problem(run, ROOT, pg.BASE_PROTECTED, [ROOT / DEFAULT_OUT_ROOT], [ROOT / x for x in EXTRA_PROTECTED], inputs,
                             [STATUS, LOG, *TRIAL_OUTPUTS])


def checkpoint_dir_problem(root: Path, inputs: dict) -> str:
    """La radice dei checkpoint della prova: fuori dal repository, non un collegamento simbolico, separata dagli ingressi."""
    if root.is_symlink():
        return "the checkpoint root is a symbolic link"
    if pg.within(root, ROOT) or pg.within(ROOT, root):
        return "the checkpoint root must be outside the repository"
    for label, p in inputs.items():
        if pg.within(p, root) or pg.within(root, p):
            return f"{label} {Path(p).name} is inside the checkpoint root, or contains it"
    return ""


def final_checks(cfg: dict, cfg_p: Path, cfg_sha: str, base: Path, cfg_pr: dict) -> dict:
    """Alla fine: configurazione, codice e scheda di riferimento con gli stessi SHA-256 dell'inizio."""
    if sha256_bytes(read_regular(cfg_p, "configuration")) != cfg_sha:
        fail("the configuration changed during the run")
    check_code(cfg)
    run = rel_path(base, cfg_pr["parent_runs"]["patience_4"]["folder"])
    bc = cfg_pr["baseline_check"]
    if sha256_bytes(read_regular(BR.parent_path(run, bc["model"], "card.json"), "reference card")) != bc["card_sha256"]:
        fail("the reference card changed during the run")
    return {"configuration_unchanged": True, "code_unchanged": len(CODE_FILES), "reference_card_unchanged": True}


class Ctx:
    """Lo stato della prova in corso."""


def outcome_status(nets: dict, control: dict, stopped) -> str:
    if stopped:
        return "INCOMPLETE"
    if any(not (nets[n].get("outcome") or {}).get("evaluable") for n in AN.NETWORKS):
        return "NOT_EVALUABLE"
    if control.get("outcome") != "reproduced":
        return "CONTROL_FAILED"
    return "PASS"


def cmd_check_inputs(a) -> int:
    base, cfg_p = resolve(a)
    hooks = check_hooks(is_official(cfg_p))
    cfg, cfg_sha, cfg_pr, pcfg, _ = load_config(cfg_p, base)
    env = PAC.thread_environment(cfg_pr)
    inp = load_inputs(cfg, cfg_pr, pcfg, base, frames=True)
    tfs = FR.tf_state()
    f = cfg_pr["trial"]["frames"]
    print(f"CHECK-INPUTS: configuration {cfg_sha[:12]}, C2 pre-registration {cfg['preregistration']['sha256'][:12]} and configuration of PR #27 "
          f"{cfg['bn_recalibration_config']['sha256'][:12]} validated, code pinned ({len(CODE_FILES)} files), architecture file "
          f"{cfg['architectures']['sha256'][:12]}, registry counts " + ", ".join(f"{n} {AN.expected_counts(n)['trainable']}" for n in AN.NETWORKS) + "; "
          f"C2 trial {inp.c2_trial['run_id']} PASS; C2 evaluation {cfg['c2_results']['evaluation_run_id']} PASS with {len(inp.reference_cells)} reused "
          f"checkpoints; trial fitting {f['fitting']['n']} and dev {f['dev']['n']} frames with keys equal to the reference card; fold schema not "
          f"loaded, no frame assembled; TensorFlow {tfs}; threads intra-op {cfg['training']['threads']['intra_op']} and inter-op "
          f"{cfg['training']['threads']['inter_op']}, environment variables " + ", ".join(f"{k} {'absent' if v is None else repr(v)}" for k, v in env.items())
          + f"; {'delivered' if hooks['official_configuration'] else 'test'} configuration, test hooks set: "
          + (", ".join(hooks["set"] + hooks["changed"]) or "none") + "; nothing written")
    print("RESULT: PASS (check-inputs)")
    return 0


def cmd_check_machine(a) -> int:
    """Sola lettura, prima della prova: la macchina e le versioni di adesso contro il record della prova C2
    (machine_comparison, la stessa funzione del controllo LivenessNet), senza dati, senza TensorFlow, nulla scritto.
    Legge la configurazione, la preregistrazione C2 e la configurazione della PR #27 (load_config), il codice fissato,
    il record della prova C2 e i file pubblici della valutazione C2 (load_inputs senza frame). Esito PASS solo se tutti
    i campi sono uguali."""
    base, cfg_p = resolve(a)
    cfg, cfg_sha, cfg_pr, pcfg, _ = load_config(cfg_p, base)
    inp = load_inputs(cfg, cfg_pr, pcfg, base, frames=False)
    r = machine_comparison(inp.c2_trial)
    FR.tf_state()
    n = len(R.MACHINE_FIELDS) + len(R.VERSION_FIELDS)
    if r["equal"]:
        print(f"check-machine: machine and versions equal to the C2 trial {inp.c2_trial['run_id']} ({n} fields); TensorFlow not imported; "
              "nothing written")
        print("RESULT: PASS (check-machine)")
        return 0
    print(f"check-machine: {', '.join(r['differing'])} differ from the C2 trial {inp.c2_trial['run_id']}: the LivenessNet control would give "
          "CONTROL_FAILED; do not launch the trial")
    print("RESULT: FAIL (check-machine)")
    return 1


def cmd_trial(a) -> int:
    if not a.i_am_authorized:
        fail("trial trains on the restricted NUAA frames: pass --i-am-authorized (first step of the C5 card, launched after an independent review)")
    if not a.checkpoint_root:
        fail("trial needs --checkpoint-root, a folder outside the repository for the three checkpoints")
    if a.attempt is None:
        fail("trial needs --attempt N (1 for the first attempt; every attempt keeps its outcome)")
    base, cfg_p = resolve(a)
    official = is_official(cfg_p)
    hooks = check_hooks(official)
    cfg, cfg_sha, cfg_pr, pcfg, bn_cfg = load_config(cfg_p, base)
    env = PAC.thread_environment(cfg_pr)
    state = repo_state(ROOT)
    if state["dirty"]:
        fail("the git work tree is not clean: the trial needs a committed state (commit the changes first)")
    out_root = Path(a.out_root) if Path(a.out_root).is_absolute() else ROOT / a.out_root
    ck_root = Path(a.checkpoint_root).absolute()
    attempts = attempt_rule(out_root, ck_root, a.attempt)
    rid = f"{time.strftime('%Y%m%d')}-ANETTRIAL-{state['commit']}" + ("" if a.attempt == 1 else f"-r{a.attempt}")
    if not RUN_ID_RE.match(rid):
        fail(f"run id {rid} does not follow YYYYMMDD-ANETTRIAL-<commit>[-r<attempt>]")
    run = out_root / rid
    ckpt = ck_root / rid
    if os.path.lexists(run):
        fail(f"the run folder {rid} already exists: every attempt keeps its outcome (archive it or use another commit)")
    if os.path.lexists(ckpt):
        fail(f"the checkpoint folder {rid} already exists under the checkpoint root")
    inputs = input_paths(cfg, cfg_pr, pcfg, base, cfg_p)
    if msg := run_dir_problem(run, out_root, inputs):
        fail(f"refusing the run folder: {msg}")
    if msg := checkpoint_dir_problem(ck_root, {**inputs, "the run folder": run}):
        fail(f"refusing the checkpoint root: {msg}")
    run.mkdir(parents=True)
    command = (f"python -I -B scripts/pad_anet_comparison.py trial --attempt {a.attempt} --i-am-authorized --checkpoint-root <a folder outside the "
               "repository>" + ("" if official else " --config <test configuration>"))
    head = {"run_id": rid, "commit": state["commit"], "dirty": False, "config_sha256": cfg_sha, "command": "trial", "hooks": hooks, **attempts}
    cap = cfg["trial"]["wall_time_cap_minutes"]
    budget = SR.Budget(cap)
    SR.signal_handlers(budget)
    started, c0 = now(), PAC.cpu_seconds()
    log, saved_warning = None, warnings.showwarning
    threads = {"requested": {k: cfg["training"]["threads"][k] for k in PAC.THREAD_KEYS}, "effective": None, "set_by_this_runner": False,
               "environment": env, "logical_cpus": os.cpu_count()}
    findings = {"control_not_reproduced": None, "not_evaluable": {}}

    def running_status():
        FR.write_status(run, {**head, "status": "RUNNING", "passed": False, "started": started, "pid": os.getpid(), "threads": threads,
                              "findings": findings,
                              "note": "RUNNING without a live process means a failed run; SIGUSR1 to this pid stops it as INCOMPLETE"})

    def record_finding(what: str, reason: str) -> None:
        """Un controllo LivenessNet non riprodotto o un'AttackNet non valutabile, registrati appena noti in run.log e in
        status.json (e poi nello stato finale, anche FAIL o INCOMPLETE): un esito successivo non li cancella."""
        if what == "control":
            findings["control_not_reproduced"] = reason
            print(f"FINDING: {CONTROL} control not reproduced ({reason})", flush=True)
        else:
            findings["not_evaluable"][what] = reason
            print(f"FINDING: {what} {reason}", flush=True)
        running_status()
    try:
        running_status()
        log = FR.open_log(run)
        warnings.showwarning = PAC.show_warning
        print(f"== AttackNet trial {rid}: configuration {cfg_sha[:12]}, cap {cap} {CAP_UNIT}, pid {os.getpid()}", flush=True)
        t = time.monotonic()
        inp = load_inputs(cfg, cfg_pr, pcfg, base, frames=True)
        store = FR.FrameStore(inp, inp.X_extracted, inp.extracted_keys)
        fit, dev = PAC.trial_frames(store, inp, "fitting"), PAC.trial_frames(store, inp, "dev")
        data_check = {f"{r}_keys_equal_to_reference_card": PAC.keys_check(inp.reference_card, r, part["rows"]) for r, part in (("fitting", fit), ("dev", dev))}
        print(f"   inputs PASS in {time.monotonic() - t:.1f} s: fitting {len(fit['keys'])} and dev {len(dev['keys'])} frames assembled, no test frame "
              f"assembled, keys equal to the reference card; peak RSS {SR.peak_rss_mb():.0f} MB", flush=True)
        import tensorflow as tf
        threads = PAC.set_threads(tf, cfg_pr)
        print(f"   {PAC.threads_text(threads)}", flush=True)
        hide_gpu(tf)
        arch = (ARCH_LOADER or FR.default_arch_loader)(rel_path(base, cfg["architectures"]["path"]))
        ckpt.mkdir(parents=True)
        ctx = Ctx()
        ctx.tf, ctx.cfg, ctx.cfg_pr, ctx.bn_cfg, ctx.inp, ctx.arch, ctx.budget, ctx.ckpt = tf, cfg, cfg_pr, bn_cfg, inp, arch, budget, ckpt
        ctx.fit, ctx.dev, ctx.record_finding = fit, dev, record_finding
        ctx.data = {"X_fit": fit["X"], "y_fit": fit["y"], "X_dev": dev["X"], "y_dev": dev["y"]}
        ctx.training = trial_training(cfg, pcfg)
        control = {}
        nets = {n: {"epochs": [], "inference": {}, "steps": {}} for n in AN.NETWORKS}
        stopped = None
        try:
            trial_control(ctx, control)                         # i finding sono registrati dentro, appena noti
            for net in AN.NETWORKS:
                trial_attacknet(ctx, net, nets[net])
        except SR.CapExceeded as e:
            stopped = {"step": e.step, "cause": e.cause, "elapsed_minutes": round(e.minutes, 1)}
            print(f"== {'CAP' if e.cause == 'cap' else 'STOP'}: {e}; the epochs measured so far are reported", flush=True)
        PAC.assert_no_test(inp)
        checks = final_checks(cfg, cfg_p, cfg_sha, base, cfg_pr)
        for net in AN.NETWORKS:
            nets[net]["rates"] = network_rates(nets[net], len(fit["keys"]), len(dev["keys"]))
        status = outcome_status(nets, control, stopped)
        wall, cpu_total, finished = budget.elapsed(), round(PAC.cpu_seconds() - c0, 1), now()
        doc = {"schema": TRIAL_SCHEMA, **head, "status": status, "started": started, "finished": finished, "wall_seconds": round(wall, 1),
               "process_cpu_seconds": cpu_total, "cap_minutes": cap, "cap_unit": CAP_UNIT, "stopped": stopped,
               "unit": {**TRIAL_UNIT, "frames": cfg_pr["trial"]["frames"]}, "epochs_fixed": TRIAL_EPOCHS, "patience": ctx.training["early_stopping_patience"],
               "batch": ctx.training["batch"], "inference_batch": cfg["trial"]["inference_batch"], "estimator_batch": cfg["trial"]["estimator_batch"],
               "device": "CPU (GPU hidden)", "visible_devices": sorted({d.device_type for d in tf.config.get_visible_devices()}), "threads": threads,
               "machine": (MACHINE or PAC.machine)(), "environment": inp.environment, "peak_rss_mb": round(SR.peak_rss_mb(), 1),
               "control": control, "control_rule": cfg["trial"]["control_rule"], "networks": nets, "data_check": data_check,
               "checkpoints": {"location": "outside the repository (unofficial output, not published)",
                               "sha256": {n: nets[n]["checkpoint"]["sha256"] for n in AN.NETWORKS if "checkpoint" in nets[n]}},
               "training": ctx.training, "code_sha256": inp.code_c5, "checks": checks, "test_read": TEST_READ, "declarations": cfg["declarations"],
               "command": command}
        doc["projection"] = projection(cfg, pcfg, doc) if status == "PASS" else {"available": False, "reason": f"trial {status}"}
        texts = {TRIAL_JSON: PAC.json_text(doc, TRIAL_JSON), TRIAL_MD: trial_md(doc)}
        for n, text in texts.items():
            if probs := PAC.output_problems(text, inp):
                fail(f"{n} refused: " + "; ".join(probs))
        mw.write_all(run, texts)
        print(f"run folder: {run.name}, wall {wall:.0f} s ({wall / 60:.1f} min of the {cap}-minute wall-time cap), process CPU {cpu_total:.0f} s, "
              f"peak RSS {doc['peak_rss_mb']:.0f} MB")
        sys.stdout.flush()
        sys.stderr.flush()
        log[0].flush()
        if probs := PAC.output_problems((run / LOG).read_text(encoding="utf-8", errors="replace"), inp):
            fail(f"{LOG} holds text that a publishable output must not hold: " + "; ".join(probs))
        print(f"RESULT: {status} (trial)")                 # dopo ogni controllo: nessun «RESULT: PASS» prima di un FAIL
        warnings.showwarning = saved_warning
        FR.close_log(log)
        log = None
        FR.write_status(run, {**head, "status": status, "passed": status == "PASS", "started": started, "finished": finished,
                              "wall_seconds": round(wall, 1), "process_cpu_seconds": cpu_total, "cap_minutes": cap, "cap_unit": CAP_UNIT,
                              "stopped": stopped, "threads": threads, "control": control.get("outcome"), "findings": findings,
                              "networks": {n: (nets[n].get("outcome") or {}).get("reason") or "evaluable" for n in AN.NETWORKS},
                              "trial_json_sha256": sha256_bytes(texts[TRIAL_JSON].encode("utf-8")),
                              "trial_md_sha256": sha256_bytes(texts[TRIAL_MD].encode("utf-8")), "run_log_sha256": sha256_file(run / LOG)})
    except BaseException as e:
        warnings.showwarning = saved_warning
        reason = str(e) if isinstance(e, Failure) else f"unexpected error {type(e).__name__}: {e}"
        left = mw.invalidate_each(run, list(TRIAL_OUTPUTS), None, FR.FAIL_DOC)
        outputs = "removed" if not left else f"left behind: {', '.join(left)}"
        tail = f"ERROR: {reason}\nFAIL outputs: {outputs}\nRESULT: FAIL (trial)\n"
        if log is not None:
            log[0].write(tail)
            FR.close_log(log)
        elif (run / LOG).is_file() and not (run / LOG).is_symlink():
            with open(run / LOG, "a", encoding="utf-8") as fh:
                fh.write(tail)
        try:
            FR.write_status(run, {**head, "status": "FAIL", "passed": False, "started": started, "finished": now(),
                                  "wall_seconds": round(budget.elapsed(), 1), "process_cpu_seconds": round(PAC.cpu_seconds() - c0, 1),
                                  "cap_minutes": cap, "cap_unit": CAP_UNIT, "threads": threads, "reason": reason, "outputs": outputs,
                                  "findings": findings,
                                  "run_log_sha256": sha256_file(run / LOG) if (run / LOG).is_file() else "none"})
        except Exception:  # noqa: BLE001
            print("ERROR: status.json could not be written", file=sys.stderr)
        raise
    return STATUS_CODE[status]


def plan_inputs(a) -> tuple:
    base, cfg_p = resolve(a)
    cfg, cfg_sha, cfg_pr, pcfg, _ = load_config(cfg_p, base)
    code = check_code(cfg)
    cells = load_c2_results(cfg, base)
    trial = trial_sha = None
    inputs = {"the configuration": cfg_p}
    if a.trial_run:
        folder = Path(a.trial_run) if Path(a.trial_run).is_absolute() else ROOT / a.trial_run
        trial, trial_sha = load_trial(folder, cfg, cfg_sha, code)
        inputs["the trial run"] = folder
    return cfg, cfg_sha, pcfg, cells, trial, trial_sha, inputs


def cmd_plan(a) -> int:
    cfg, cfg_sha, pcfg, cells, trial, trial_sha, inputs = plan_inputs(a)
    out = Path(a.out_dir) if Path(a.out_dir).is_absolute() else ROOT / a.out_dir
    existing = [n for n in PLAN_OUTPUTS if os.path.lexists(out / n)]
    if existing:
        fail(f"{', '.join(existing)} already exist in {out.name}: the plan is written only to new files")
    if msg := mw.output_problem(out, ROOT, pg.BASE_PROTECTED, [ROOT / DEFAULT_PLAN_DIR], [ROOT / x for x in EXTRA_PROTECTED], inputs, list(PLAN_OUTPUTS)):
        fail(f"refusing the plan folder: {msg}")
    texts = plan_texts(cfg, cfg_sha, pcfg, cells, trial, trial_sha)
    for n, text in texts.items():
        if probs := PAC.output_problems(text, FR.Inputs()):
            fail(f"{n} refused: " + "; ".join(probs))
    mw.write_all(out, texts, newline="")
    print(f"plan: {NEW_TRAININGS} new trainings in {len(run_names())} runs, {len(cells)} reused cells; " +
          ("hours from the trial " + trial["run_id"] if trial else f"hours {TRIAL_PENDING}") + f"; files {', '.join(PLAN_OUTPUTS)} in {out.name}")
    print("RESULT: PASS (plan)")
    return 0


def cmd_check_plan(a) -> int:
    if not a.plan_dir:
        fail("check-plan needs --plan-dir")
    cfg, cfg_sha, pcfg, cells, trial, trial_sha, _ = plan_inputs(a)
    folder = Path(a.plan_dir) if Path(a.plan_dir).is_absolute() else ROOT / a.plan_dir
    texts = plan_texts(cfg, cfg_sha, pcfg, cells, trial, trial_sha)
    bad = []
    for n, text in texts.items():
        got = read_regular(folder / n, f"plan file {n}")
        if got != text.encode("utf-8"):
            bad.append(n)
    if bad:
        fail(f"plan files {', '.join(bad)} differ from the files regenerated from the configuration" + (f" and the trial {trial['run_id']}" if trial else ""))
    m = check_matrix([{**r, "position": int(r["position"]), "fold": int(r["fold"]), "seed": int(r["seed"]), "patience": int(r["patience"])}
                      for r in FR.parse_csv(read_regular(folder / "run_matrix.csv", "run_matrix.csv"), MATRIX_COLUMNS, "run_matrix.csv")])
    c = check_reference_cells([{**r, "fold": int(r["fold"]), "seed": int(r["seed"]), "patience": int(r["patience"])}
                               for r in FR.parse_csv(read_regular(folder / "reference_cells.csv", "reference_cells.csv"), CELL_COLUMNS, "reference_cells.csv")])
    print(f"check-plan: {len(PLAN_OUTPUTS)} files equal byte for byte to the regenerated ones; run matrix {m['rows']} rows ({m['per_patience']} per "
          f"patience, 24 per network, {m['runs']} runs of 12), {c['rows']} reused cells; " + (f"trial {trial['run_id']}" if trial else f"hours {TRIAL_PENDING}"))
    print("RESULT: PASS (check-plan)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="C5: the three AttackNets against LivenessNet and ResNet-18 on NUAA: inputs check, technical trial "
                                             "(arm B fold 1 seed 42, fitting and dev only, CPU), run matrix and plan")
    ap.add_argument("cmd", choices=["check-inputs", "check-machine", "trial", "plan", "check-plan"])
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="configuration JSON (relative to the repository root)")
    ap.add_argument("--input-root", default=None, help="base folder of the input paths of the configuration (default: repository root)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="trial: folder that receives <run-id>/ (inside the repository only results/anetcmp)")
    ap.add_argument("--checkpoint-root", default=None, help="trial: folder outside the repository that receives <run-id>/ with the three checkpoints")
    ap.add_argument("--attempt", type=int, default=None, help="trial: the number of this attempt, 1 + the earlier attempts found in the output "
                                                              "and checkpoint roots (required)")
    ap.add_argument("--i-am-authorized", action="store_true", help="required by trial: training on the restricted frames")
    ap.add_argument("--trial-run", default=None, help="plan, check-plan: the run folder of a PASS trial of this code")
    ap.add_argument("--out-dir", default=DEFAULT_PLAN_DIR, help="plan: folder of the six plan files (never overwritten)")
    ap.add_argument("--plan-dir", default=None, help="check-plan: folder of the six plan files to compare")
    a = ap.parse_args(argv)
    try:
        return {"check-inputs": cmd_check_inputs, "check-machine": cmd_check_machine, "trial": cmd_trial, "plan": cmd_plan,
                "check-plan": cmd_check_plan}[a.cmd](a)
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
