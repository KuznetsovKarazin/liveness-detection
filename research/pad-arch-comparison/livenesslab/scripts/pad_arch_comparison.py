"""
Confronto tra reti PAD, LivenessNet e ResNet-18, entrambe addestrate da zero con la stessa ricetta, gli stessi split
per soggetto, gli stessi seed e lo stesso preprocessing (incarico del coordinatore sulla scheda C2, commento del
17/11/2026; sono scelte dell'autore, non del coordinatore, l'analisi leave-one-subject-out nella forma del suo protocollo
N1 e il riaddestramento di tutte le reti del piano). In questa consegna: le definizioni operative
(scripts/pad_arch_definitions.py), la tabella dei run previsti generata dalla configurazione (plan) e una prova tecnica
breve (trial). Nessun frame di test è assemblato o valutato e nessun punteggio o metrica del test è letto. I file dei
frame contengono anche i frame dei soggetti tenuti fuori: la cache è sottoposta a hash per intero (e solo trial ne
carica gli array di immagini, nel FrameStore); il file dei frame estratti è caricato per intero con lo SHA-256 di ogni
frame (in entrambi i comandi); solo trial assembla, e solo i frame del fitting e del dev (TEST_READ dice esattamente che
cosa è letto). Nessun dataset esterno è toccato.

Comandi:
  check-inputs   configurazione a insieme chiuso e controlli incrociati (configurazioni dei run accettati a patience 4 e
                 12 con lo SHA-256 fissato e validate dal loro runner, numeri di NUAA uguali alle loro sezioni, ricetta
                 uguale a nuaa_folds_run.FROZEN_TRAINING, conteggi dei parametri uguali all'aritmetica del registro);
                 codice importato e architettura di LivenessNet per SHA-256; status dei due run accettati (PASS, run id,
                 configurazione, SHA256SUMS riservato, i 12 modelli del braccio B completati); la scheda del modello di
                 riferimento B fold 1 seed 42 (run a patience 4) uguale al SHA256SUMS riservato e alla configurazione;
                 gli ingressi del fitting e del dev con la sequenza di nuaa_bn_recalibration.load_inputs (i cinque
                 manifest M2, cache sottoposta a hash per intero con etichette e soggetti letti, run delle celle, run dei
                 fold, frame estratti caricati per intero con lo SHA-256 di ogni frame, sorgenti delle righe; lo schema
                 dei fold, che contiene il ruolo test, non è mai caricato); conteggi del fitting e del dev della prova e impronte delle loro chiavi uguali a
                 fitting_keys_sha256 e dev_keys_sha256 della scheda di riferimento; le variabili d'ambiente dei thread di
                 TensorFlow non contraddicono training.threads (thread_environment). Nessun frame assemblato, TensorFlow
                 non importato, nulla scritto.
  trial          la prova tecnica, in un solo processo e in quest'ordine: le variabili d'ambiente dei thread
                 (TF_NUM_INTRAOP_THREADS, TF_NUM_INTEROP_THREADS) presenti e diverse da training.threads rifiutate prima
                 di ogni lavoro; gli stessi controlli di check-inputs; fitting
                 e dev di NUAA, braccio B, fold 1 assemblati (solo questi due ruoli: assert_no_test) e le loro chiavi
                 confrontate di nuovo con la scheda di riferimento; import di TensorFlow e subito i thread intra-op e
                 inter-op fissati a training.threads (10 e 10, coordinatore, 19/11/2026) con set_threads, riletti con i
                 getter (diversi, o TensorFlow già inizializzato: FAIL), poi la GPU nascosta; per ciascuna rete,
                 LivenessNet poi ResNet-18, seed 42, 3 epoche fisse con trial_trainer (copia dichiarata di
                 nuaa_folds_run.default_trainer, nessun checkpoint salvato) e la callback dei tempi; poi una passata di
                 sola inferenza (batch 16, training=False) sul fitting e una sul dev. Controllo della storia di
                 LivenessNet contro le prime 3 epoche della scheda accettata (regola di nuaa_patience_verify.shared_history:
                 cinque grandezze identiche, scarto 0; per grandezza la prima epoca diversa, lo scarto massimo e la sua
                 epoca, da history_differences): esito registrato, non cambia lo stato. Thread effettivi riletti prima e
                 dopo ogni rete. Riverifica finale di configurazione, codice e scheda.
  plan           run_plan.csv, run_plan.md, run_order.csv e run_caps.csv dalla configurazione e, se dato, dal trial.json
                 di una prova (--trial-run) del formato pad-arch-trial/2 (il formato precedente è rifiutato con un
                 messaggio), che deve essere stato prodotto con la stessa configurazione e lo stesso codice (i file
                 fissati sono prima confrontati con la configurazione, come in trial: check_code) e con i thread impostati
                 dal runner ai valori della configurazione: per dataset × rete × patience unità, addestramenti, epoche
                 minime e massime, ore stimate; l'ordine fissato dei 48 addestramenti NUAA nei due run (uno per patience,
                 plan_order) con la posizione del modello di riferimento nel run accettato; per run il tetto in ore di
                 tempo reale e, a parte, la stima in ore di CPU (run_caps); la macchina della prova, i thread e la regola
                 della stessa macchina.
  compare-trials confronto di sola lettura fra due prove (--trial-run la nuova, --reference-run quella di riferimento,
                 formato pad-arch-trial/1 o /2): per rete e per campo della storia epoche confrontate, prima epoca
                 diversa, scarto massimo e sua epoca (history_differences); esito «reproduced» solo con lo stesso numero
                 non nullo di epoche e scarto 0 ovunque; per ciascuna prova run id, commit, SHA-256 della configurazione,
                 macchina, thread come registrati dal suo formato e SHA-256 dei file letti. trial_comparison.json e
                 trial_comparison.md in --compare-out-dir (fuori dal repository o sotto delivery/archcmp/trial-comparison;
                 mai dentro le cartelle lette), mai sovrascritti; su ogni errore nessuna uscita.

Un solo processo per la prova (decisione motivata nella configurazione, trial.process): LivenessNet è la prima rete
addestrata nel processo, come il modello B fold 1 seed 42 nel run accettato (primo dei 36 del suo processo); i run del
piano addestrano più modelli in sequenza in un processo, come i run accettati; gli ingressi (circa 2,5 GB di frame)
sono caricati e verificati una volta; fra le due reti il trainer chiama clear_session e il modello precedente è liberato.

Esiti della prova: PASS (0), FAIL (1: trial.json e trial.md tolti; restano status.json con la ragione e run.log),
INCOMPLETE (2: tetto di tempo reale o SIGUSR1, controllati prima di ogni rete, a fine epoca e prima di ciascuna delle
due passate d'inferenza, quindi superabili al più del costo di un addestramento fuori dalle epoche (costruzione del
modello, preparazione) più un'epoca con la sua validazione, oppure di una passata d'inferenza; trial.json e
trial.md riportano le epoche e le passate misurate). Albero git non pulito, cartella del run esistente o
fuori dalla radice prevista: rifiuto prima di scrivere. Lo status.json finale ha sempre started, finished, il tempo
reale e il tempo di CPU del processo (impegno del 17/11 col coordinatore: il runner precedente scriveva started solo
nello stato RUNNING).

Uscite della prova (results/archcmp/YYYYMMDD-ARCHTRIAL-<commit>; --out-root accetta un'altra radice purché fuori dal
repository o, dentro il repository, solo sotto results/archcmp/, e non un collegamento simbolico): status.json,
run.log, trial.json, trial.md; solo aggregati (nessun identificativo di soggetto o di frame, nessun percorso, nome
dell'host o dell'utente: controllati prima della scrittura). Nessuna uscita riservata.

  .venv/bin/python -I -B scripts/pad_arch_comparison.py check-inputs
  .venv/bin/python -I -B scripts/pad_arch_comparison.py trial --i-am-authorized
  .venv/bin/python -I -B scripts/pad_arch_comparison.py plan [--trial-run results/archcmp/<run-id>] [--out-dir <cartella nuova>]
  .venv/bin/python -I -B scripts/pad_arch_comparison.py compare-trials --trial-run <prova nuova> --reference-run <prova di riferimento> \
      [--compare-out-dir <cartella nuova>]

Thread di TensorFlow: dal 19/11/2026 (coordinatore) fissati a 10 intra-op e 10 inter-op per l'intero piano NUAA
(training.threads); set_threads è l'unica funzione che li imposta, riusabile dai comandi futuri del piano; record in
status.json (anche RUNNING e FAIL), trial.json, trial.md e nel record di ogni rete.

Riuso per importazione: nuaa_folds_run (FR: cache, FrameStore/assemble, ricetta, callback, hide_gpu, stato, log),
nuaa_session_ablation_run (SR: configurazioni dei run accettati, celle, frame estratti, Budget, segnali, stato git),
nuaa_bn_recalibration (BR: Inputs, frames, load_folds_frames_rows, parent_path), nuaa_patience_verify (PV:
HISTORY_KEYS; la regola di shared_history è riprodotta in history_differences), pad_arch_models (AM: registro), pad_arch_definitions (definizioni, nessun uso qui oltre al
piano delle condizioni), m2_writers e path_guard. Dipendenze: NumPy; TensorFlow 2.15 (solo trial, solo CPU).
Formati: configurazione pad-arch-comparison-config/2, prova pad-arch-trial/2 (thread impostati e registrati, prima
epoca diversa corretta); compare-trials legge anche pad-arch-trial/1 (PR #29).
"""
import argparse
import gc
import getpass
import importlib.metadata
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import m2_writers as mw  # noqa: E402
import nuaa_bn_recalibration as BR  # noqa: E402
import nuaa_folds_run as FR  # noqa: E402
import nuaa_patience_verify as PV  # noqa: E402
import nuaa_session_ablation_run as SR  # noqa: E402
import pad_arch_definitions as D  # noqa: E402
import pad_arch_models as AM  # noqa: E402
import path_guard as pg  # noqa: E402

Failure, fail = FR.Failure, FR.fail
sha256_bytes, sha256_file, read_regular, rel_path, csv_text = FR.sha256_bytes, FR.sha256_file, FR.read_regular, FR.rel_path, FR.csv_text
now = FR.now
hide_gpu, training_callbacks, check_scheduler = FR.hide_gpu, FR.training_callbacks, FR.check_scheduler

DEFAULT_CONFIG = "delivery/archcmp/pad_arch_comparison.json"
DEFAULT_OUT_ROOT = "results/archcmp"
DEFAULT_PLAN_DIR = "delivery/archcmp/plan"
DEFAULT_COMPARE_DIR = "delivery/archcmp/trial-comparison"
CONFIG_SCHEMA = "pad-arch-comparison-config/2"
TRIAL_SCHEMA = "pad-arch-trial/2"
TRIAL_SCHEMA_V1 = "pad-arch-trial/1"              # il formato della PR #29 (thread non impostati), letto solo da compare-trials
COMPARE_SCHEMA = "pad-arch-trial-comparison/1"
THREAD_KEYS = ("intra_op", "inter_op")
TF_THREAD_ENV = {"intra_op": "TF_NUM_INTRAOP_THREADS", "inter_op": "TF_NUM_INTEROP_THREADS"}
REQUIRED_THREADS = 10                              # coordinatore, 19/11/2026: intra-op e inter-op a 10 per l'intero piano NUAA
V1_THREAD_KEYS = ("inter_op_parallelism_threads", "intra_op_parallelism_threads", "logical_cpus", "set_by_this_runner")
V2_THREAD_KEYS = ("effective", "environment", "logical_cpus", "requested", "set_by_this_runner")
MACHINE_KEYS = ("logical_cpus", "os", "processor", "ram_gb")
V1_THREADS_NOTE = ("format pad-arch-trial/1: the runner did not set the threads; the recorded values are those read from the getters of "
                   "tf.config.threading (0 = not set through tf.config.threading); format /1 does not record the environment variables")
V2_THREADS_NOTE = "format pad-arch-trial/2: threads set by the runner; requested, effective (read back) and environment variables recorded"
RUN_ID_RE = re.compile(r"^\d{8}-ARCHTRIAL-[0-9a-f]{7,40}$")
TOP_KEYS = ("schema", "task", "authorisation", "declarations", "parent_runs", "code", "architectures", "training", "trial",
            "baseline_check", "definitions", "plan", "outputs", "environment", "test")
CODE_FILES = ("scripts/nuaa_folds_run.py", "scripts/nuaa_session_ablation_run.py", "scripts/nuaa_bn_recalibration.py",
              "scripts/nuaa_patience_verify.py", "scripts/m2_writers.py", "scripts/path_guard.py", "scripts/package_rules.py",
              "scripts/pad_arch_models.py", "scripts/pad_arch_definitions.py")
DECLARATION_KEYS = ("no_test", "technical_trial", "cpu_only", "no_checkpoint", "threads")
PARENT_KEYS = ("patience_4", "patience_12")
PARENT_PATIENCE = {"patience_4": 4, "patience_12": 12}
TRIAL_UNIT = {"dataset": "NUAA", "arm": "arm_b", "fold": 1, "seed": 42}
TRIAL_ROLES = ("fitting", "dev")
TRIAL_EPOCHS = 3
CAP_LIMIT_MINUTES = 60
CAP_UNIT = "minutes of wall time (elapsed real time, not CPU time)"
RUN_CAP_UNIT = "hours of wall time (elapsed real time, not CPU time)"
DATASETS = ("NUAA", "CelebA-Spoof", "Replay-Attack")
ACCESS = ("available", "access missing")
UNIT_KINDS = ("fold", "split")
TBF = "to be fixed in the pre-registration"
BANNED = ("because", "caused by", "proves", "demonstrates", "attributable", "due to")
STATUS_CODE = {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}
STATUS, LOG, TRIAL_JSON, TRIAL_MD = FR.STATUS, FR.LOG, "trial.json", "trial.md"
TRIAL_OUTPUTS = (TRIAL_JSON, TRIAL_MD)
PLAN_OUTPUTS = ("run_plan.csv", "run_plan.md", "run_order.csv", "run_caps.csv")
COMPARE_OUTPUTS = ("trial_comparison.json", "trial_comparison.md")
ORDER_COLUMNS = ("run", "position", "network", "fold", "seed", "patience", "reference_run", "reference_model", "reference_position")
CAPS_COLUMNS = ("run", "dataset", "patience", "trainings", "wall_hours_min", "wall_hours_max", "cap_wall_hours", "cpu_hours_min",
                "cpu_hours_max")
EXTRA_PROTECTED = BR.EXTRA_PROTECTED + ("results/bnrecal", "results/bntest", "results/patienceverify", "results/lrcontrol")
HISTORY_KEYS = PV.HISTORY_KEYS
PLAN_COLUMNS = ("dataset", "access", "network", "patience", "units", "existing_trainings", "new_trainings", "reference_run",
                "epochs_min", "epochs_max", "fitting_frames_per_unit", "dev_frames_per_unit", "test_frames_per_split",
                "evaluation_cells", "training_wall_hours_min", "training_wall_hours_max", "evaluation_wall_hours",
                "wall_hours_min", "wall_hours_max", "cpu_hours_min", "cpu_hours_max")
TEST_READ = ("no test frame is assembled or evaluated and no test score or metric is read, computed or written. check-inputs and "
             "trial read: the five M2 manifests (official_train, official_test, cnn_train, cnn_es and cnn_test: NUAA's official "
             "image lists and the cache order, read to index the cache and to check the rows of extracted_frames.csv against the official lists; "
             "official_test and cnn_test are not the test role of the folds); the cache, hashed whole with its label and subject "
             "members read (only trial loads its image arrays whole, in the frame store); the extracted frames file, loaded whole "
             "with the SHA-256 of every frame (both commands); these frame files also hold frames of the held-out subjects, but only "
             "the fitting and dev frames of arm B, fold 1 are assembled, by trial only (an explicit check refuses any other role); "
             "of the cells run status.json, the public and restricted SHA256SUMS, its configuration and the cell manifests of the "
             "arms and of the common dev; of the folds run status.json, the restricted SHA256SUMS and extracted_frames.csv (no "
             "public file opened); of the accepted run at patience 4 status.json, the restricted SHA256SUMS and one training card "
             "(arm_b_fold1_seed42); of the accepted run at patience 12 status.json and the restricted SHA256SUMS; the configurations "
             "of the two accepted runs and the LivenessNet architecture file; the fold schema (fold<k>_test.csv, the test role of "
             "the folds) is never loaded and no <model>_test_scores.csv, public aggregate or run record of any run is opened; plan "
             "reads only this configuration, the configurations of the two accepted runs, the code files pinned in this "
             "configuration and, if given, the status.json and trial.json of a trial; compare-trials reads only the status.json and "
             "trial.json of the two trials it compares and, if given, the reference conditions file")
TRAINER_SUBSTITUTIONS = (
    ("def default_trainer(data: dict, training: dict, weights_path: Path, train_seed: int, arch_module, cap_check) -> dict:\n",
     "def trial_trainer(data: dict, training: dict, network: str, train_seed: int, build_model, cap_check, timer) -> tuple:\n"),
    ("    \"\"\"La ricetta di n1_trainer.train, riga per riga, con fitting e dev già separati (nessuno split interno); il\n"
     "    checkpoint è l'epoca con val_loss minimo sul dev; cap_check() è chiamata alla fine di ogni epoca (tetto di tempo).\n"
     "    Lo scheduler è quello di training.scheduler (training_callbacks), controllato prima di importare TensorFlow.\"\"\"\n",
     "    \"\"\"Copia dichiarata di nuaa_folds_run.default_trainer (la ricetta di n1_trainer.train, riga per riga), con le sole\n"
     "    sostituzioni di TRAINER_SUBSTITUTIONS: il modello viene da build_model() (una rete del registro) invece che da\n"
     "    create_model(\"LivenessNet\"); nessun checkpoint salvato (nessun weights_path); la callback di timer.callback(tf),\n"
     "    che registra i tempi e la storia e non tocca il training, è inserita prima di Cap; la scheda dice la rete e che\n"
     "    nessun checkpoint è salvato; restituisce (scheda, modello) per le passate di inferenza della prova.\"\"\"\n"),
    ("    if weights_path.exists() or weights_path.is_symlink():\n        fail(f\"{weights_path.name} already exists (never overwritten)\")\n", ""),
    ("    model = arch_module.create_model(\"LivenessNet\").get_model()\n", "    model = build_model()\n"),
    ("    cbs = training_callbacks(tf, training, cap_check)\n",
     "    cbs = training_callbacks(tf, training, cap_check)\n    cbs.insert(len(cbs) - 1, timer.callback(tf))      # tempi e storia per epoca, prima di Cap\n"),
    ("    tmp = weights_path.with_name(f\".{weights_path.name}.{os.getpid()}.tmp.keras\")\n    try:\n        model.save(tmp)\n"
     "        os.replace(tmp, weights_path)                 # scrittura atomica del checkpoint\n    except BaseException:\n"
     "        tmp.unlink(missing_ok=True)                   # nessun temporaneo resta in restricted/ dopo un errore\n        raise\n", ""),
    ("    card = {\"arch\": \"LivenessNet\", ", "    card = {\"arch\": network, "),
    ("\"test_evaluation\": \"none (scores computed afterwards on CPU)\", ", "\"test_evaluation\": \"none (no test frame is assembled in the trial)\", "),
    (", \"weights_file\": weights_path.name,\n            \"weights_sha256\": sha256_file(weights_path), \"saved\": now()}\n",
     ", \"checkpoint\": \"none saved (trial)\"}\n"),
    ("    print(f\"== LivenessNet seed {train_seed}: ", "    print(f\"== {network} seed {train_seed}: "),
    ("    return card\n", "    return card, model\n"),
)

# punti di iniezione (solo i test): stato git, modulo di LivenessNet, costruttore del modello, trainer, macchina
REPO_STATE = None       # (root) -> {"commit", "dirty"} (default: SR.repo_state)
ARCH_LOADER = None      # (path) -> modulo con create_model (default: FR.default_arch_loader)
BUILD = None            # (network, input_shape, arch_module) -> modello Keras (default: AM.build)
TRAINER = None          # come trial_trainer (default: trial_trainer)
MACHINE = None          # () -> dizionario della macchina (default: machine)
SYSCTL = None           # (chiave) -> valore testuale o None (default: sysctl), solo per machine()


# ============================================================================ configurazione

def _text(v, what: str) -> None:
    if not isinstance(v, str) or not v.strip():
        fail(f"configuration: {what} must be a non-empty text")


def _keys(d, keys, what: str) -> None:
    if not isinstance(d, dict) or sorted(d) != sorted(keys):
        fail(f"configuration: {what} needs exactly {', '.join(sorted(keys))}")


def _int(v, what: str, low: int = 0) -> int:
    if not FR.is_count(v) or v < low:
        fail(f"configuration: {what} must be an integer >= {low}")
    return v


def _number(v, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        fail(f"configuration: {what} must be a finite positive number")
    return float(v)


def _opt_int(v, what: str, low: int = 0):
    return None if v is None else _int(v, what, low)


def _opt_text(v, what: str):
    if v is not None:
        _text(v, what)


def check_architectures(ar) -> list:
    """Le due architetture: stessa forma d'ingresso (quella dei frame, FR.FRAME_SHAPE), conteggi uguali all'aritmetica
    del registro (AM.expected_counts) per quella forma."""
    _keys(ar, list(AM.NETWORKS), "architectures")
    _keys(ar["LivenessNet"], ["batchnorm_layers", "builder", "input_shape", "non_trainable_parameters", "path", "sha256",
                              "trainable_parameters"], "architectures.LivenessNet")
    _keys(ar["ResNet-18"], ["batchnorm", "batchnorm_layers", "definition", "input_shape", "module", "non_trainable_parameters",
                            "trainable_parameters"], "architectures.ResNet-18")
    for k in ("builder", "path"):
        _text(ar["LivenessNet"][k], f"architectures.LivenessNet.{k}")
    if not SR._hex(ar["LivenessNet"]["sha256"]):
        fail("configuration: architectures.LivenessNet.sha256 malformed")
    for k in ("batchnorm", "definition"):
        _text(ar["ResNet-18"][k], f"architectures.ResNet-18.{k}")
    if ar["ResNet-18"]["module"] != "scripts/pad_arch_models.py":
        fail("configuration: architectures.ResNet-18.module must be scripts/pad_arch_models.py (pinned in code)")
    shape = ar["LivenessNet"]["input_shape"]
    if ar["ResNet-18"]["input_shape"] != shape or shape != list(FR.FRAME_SHAPE):
        fail(f"configuration: both networks need input_shape {list(FR.FRAME_SHAPE)} (the frame shape)")
    for net in AM.NETWORKS:
        try:
            exp = AM.expected_counts(net, shape)
        except AM.ModelProblem as e:
            fail(f"configuration: {e}")
        got = {k: ar[net][f"{k}_parameters"] for k in ("trainable", "non_trainable")}
        got["batchnorm_layers"] = ar[net]["batchnorm_layers"]
        if any(got[k] != exp[k] or not FR.is_count(got[k]) for k in got):
            fail(f"configuration: architectures.{net} counts {got} differ from the arithmetic of the registry {exp}")
    return shape


def check_threads_config(th) -> dict:
    """training.threads: esattamente intra_op, inter_op e rule; i due valori interi positivi (non booleani), uguali a
    REQUIRED_THREADS = 10, il valore fissato dal coordinatore il 19/11/2026 per l'intero piano NUAA; nessun valore
    predefinito (un campo mancante è un errore)."""
    _keys(th, [*THREAD_KEYS, "rule"], "training.threads")
    for k in THREAD_KEYS:
        v = th[k]
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            fail(f"configuration: training.threads.{k} must be a positive integer")
        if v != REQUIRED_THREADS:
            fail(f"configuration: training.threads.{k} must be {REQUIRED_THREADS} (fixed by the coordinator on 19 November 2026 for the "
                 "whole NUAA plan)")
    _text(th["rule"], "training.threads.rule")
    return {k: th[k] for k in THREAD_KEYS}


def check_definitions_added(df: dict) -> None:
    """Le definizioni aggiunte con la preregistrazione definitiva: frame_ids (il campo degli ID dei frame uguale a
    D.FRAME_ID_FIELD), primary_analysis (patience, soglia e le due condizioni uguali alle costanti di
    pad_arch_definitions, i nomi della lettura del contrasto, la regola di lettura dichiarata scelta dell'autore),
    per_seed_reading (testo)."""
    fi = df["frame_ids"]
    _keys(fi, ["field", "rule"], "definitions.frame_ids")
    if fi["field"] != D.FRAME_ID_FIELD:
        fail(f"configuration: definitions.frame_ids.field must be {D.FRAME_ID_FIELD!r} (the frame key of the score files)")
    _text(fi["rule"], "definitions.frame_ids.rule")
    pa = df["primary_analysis"]
    _keys(pa, ["contrast", "ordinary", "patience", "per_seed", "re_estimated", "reading", "reading_names", "secondary", "threshold"],
          "definitions.primary_analysis")
    if pa["patience"] != D.PRIMARY_PATIENCE or isinstance(pa["patience"], bool) or pa["threshold"] != D.PRIMARY_THRESHOLD:
        fail(f"configuration: definitions.primary_analysis must fix patience {D.PRIMARY_PATIENCE} and threshold {D.PRIMARY_THRESHOLD!r}")
    if pa["ordinary"] != list(D.PRIMARY_ORDINARY) or pa["re_estimated"] != list(D.PRIMARY_RE_ESTIMATED):
        fail("configuration: definitions.primary_analysis conditions must be those of pad_arch_definitions (ordinary and re-estimated BN, "
             "patience 4, threshold 0.5)")
    if pa["reading_names"] != [D.CONTRAST_LOWER, D.CONTRAST_HIGHER, D.CONTRAST_UNRESOLVED]:
        fail("configuration: definitions.primary_analysis.reading_names must be the names of pad_arch_definitions.contrast_reading")
    for k in ("contrast", "per_seed", "reading", "secondary"):
        _text(pa[k], f"definitions.primary_analysis.{k}")
    if "re-estimated" not in pa["contrast"] or "minus" not in pa["contrast"]:
        fail("configuration: definitions.primary_analysis.contrast must state the subtraction (re-estimated minus ordinary)")
    if "choice of the author" not in pa["reading"]:
        fail("configuration: definitions.primary_analysis.reading must be declared a choice of the author")
    _text(df["per_seed_reading"], "definitions.per_seed_reading")


def check_plan_added(pl: dict, nuaa: dict) -> None:
    """Il piano della preregistrazione definitiva: order (motivazione), runs (i run NUAA, uno per patience, con
    l'elenco ordinato dei loro addestramenti uguale a nuaa_runs), caps (regola dei tetti: moltiplicatore >= 1 e passo
    di arrotondamento > 0, unità RUN_CAP_UNIT), machine_rule (stessa macchina e stessi thread)."""
    _text(pl["order"], "plan.order")
    if pl["runs"] != nuaa_runs(nuaa):
        fail("configuration: plan.runs must list the NUAA runs (one per patience) with the trainings in the fixed order of nuaa_runs")
    ca = pl["caps"]
    _keys(ca, ["round_up_hours", "rule", "unit", "wall_multiplier"], "plan.caps")
    _number(ca["round_up_hours"], "plan.caps.round_up_hours")
    if _number(ca["wall_multiplier"], "plan.caps.wall_multiplier") < 1:
        fail("configuration: plan.caps.wall_multiplier must be at least 1 (a cap below the estimate is refused)")
    if ca["unit"] != RUN_CAP_UNIT:
        fail(f"configuration: plan.caps.unit must be {RUN_CAP_UNIT!r}")
    _text(ca["rule"], "plan.caps.rule")
    _text(pl["machine_rule"], "plan.machine_rule")
    if "same machine" not in pl["machine_rule"] or "same threads" not in pl["machine_rule"]:
        fail("configuration: plan.machine_rule must state that every training and evaluation runs on the same machine with the same threads")


def check_dataset(ds: dict, i: int) -> None:
    """Una voce del piano: campi obbligatori sempre presenti; per un dataset disponibile ogni valore è fissato, per uno
    con accesso mancante i valori non ancora fissati sono null; nessun addestramento esistente (il piano riaddestra
    tutto: existing = 0); nessun numero inventato."""
    what = f"plan.datasets[{i}]"
    _keys(ds, ["access", "frames_per_unit", "identity_split", "missing_access", "name", "official_source", "seeds", "splits",
               "trainings", "unit_kind"], what)
    if ds["name"] != DATASETS[i]:
        fail(f"configuration: {what}.name must be {DATASETS[i]!r} (datasets {', '.join(DATASETS)}, in this order)")
    if ds["access"] not in ACCESS:
        fail(f"configuration: {what}.access must be one of {', '.join(ACCESS)}")
    avail = ds["access"] == "available"
    for k in ("official_source", "identity_split", "missing_access"):
        _opt_text(ds[k], f"{what}.{k}")
    if ds["unit_kind"] is not None and ds["unit_kind"] not in UNIT_KINDS:
        fail(f"configuration: {what}.unit_kind must be one of {', '.join(UNIT_KINDS)} or null")
    for k in ("splits", "seeds"):
        v = ds[k]
        if v is not None and (not isinstance(v, list) or not v or not all(FR.is_count(x) for x in v) or len(set(v)) != len(v)):
            fail(f"configuration: {what}.{k} must be a list of distinct non-negative integers, or null")
    if ds["seeds"] is not None and ds["seeds"] != list(D.PLAN_SEEDS):
        fail(f"configuration: {what}.seeds must be {list(D.PLAN_SEEDS)} (pad_arch_definitions.PLAN_SEEDS, the same seeds for every dataset)")
    fp = ds["frames_per_unit"]
    _keys(fp, ["dev", "fitting", "test_per_split"], f"{what}.frames_per_unit")
    _opt_int(fp["fitting"], f"{what}.frames_per_unit.fitting", 1)
    _opt_int(fp["dev"], f"{what}.frames_per_unit.dev", 1)
    t = fp["test_per_split"]
    if t is not None and (not isinstance(t, list) or ds["splits"] is None or len(t) != len(ds["splits"])
                          or not all(FR.is_count(x) and x > 0 for x in t)):
        fail(f"configuration: {what}.frames_per_unit.test_per_split must list a positive count per split, or be null")
    units = len(ds["splits"]) * len(ds["seeds"]) if ds["splits"] is not None and ds["seeds"] is not None else None
    tr = ds["trainings"]
    want = [(n, p) for n in AM.NETWORKS for p in D.PATIENCES]
    if not isinstance(tr, list) or len(tr) != len(want):
        fail(f"configuration: {what}.trainings must list {len(want)} entries (network x patience)")
    for j, (e, (net, pat)) in enumerate(zip(tr, want)):
        _keys(e, ["existing", "network", "new", "patience", "reference_run"], f"{what}.trainings[{j}]")
        if (e["network"], e["patience"]) != (net, pat):
            fail(f"configuration: {what}.trainings[{j}] must be {net} with patience {pat}")
        if e["existing"] != 0 or isinstance(e["existing"], bool):
            fail(f"configuration: {what}.trainings[{j}].existing must be 0 (every training of the plan is new)")
        _opt_int(e["new"], f"{what}.trainings[{j}].new")
        _opt_text(e["reference_run"], f"{what}.trainings[{j}].reference_run")
        if units is not None and e["new"] is not None and e["new"] != units:
            fail(f"configuration: {what}.trainings[{j}].new must equal the units ({units})")
    if avail:
        nulls = [k for k in ("official_source", "identity_split", "unit_kind", "splits", "seeds") if ds[k] is None] + \
                [f"frames_per_unit.{k}" for k, v in fp.items() if v is None] + [f"trainings[{j}].new" for j, e in enumerate(tr) if e["new"] is None]
        if nulls or ds["missing_access"] is not None:
            fail(f"configuration: {what} is available: {', '.join(nulls) or 'missing_access'} must be fixed (missing_access null)")
    elif ds["missing_access"] is None:
        fail(f"configuration: {what}: missing_access must state what access is missing")


def load_config(p: Path, base: Path) -> tuple:
    """(configurazione, SHA-256 dei byte letti, configurazione del run accettato a patience 4). Insiemi chiusi, tipi e
    valori fissati controllati qui, prima di leggere qualunque dato; poi le configurazioni dei due run accettati (SHA-256
    fissato, validate da SR.load_config) e i numeri di NUAA contro di esse."""
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
    if "no test frame is assembled or evaluated" not in de["no_test"]:
        fail("configuration: declarations.no_test must say that no test frame is assembled or evaluated")
    pr_ = cfg["parent_runs"]
    _keys(pr_, PARENT_KEYS, "parent_runs")
    for k in PARENT_KEYS:
        e = pr_[k]
        _keys(e, ["config_path", "config_sha256", "early_stopping_patience", "folder", "restricted_sha256sums_sha256", "run_id"], f"parent_runs.{k}")
        _text(e["config_path"], f"parent_runs.{k}.config_path")
        _text(e["folder"], f"parent_runs.{k}.folder")
        if not isinstance(e["run_id"], str) or not SR.RUN_ID_RE.match(e["run_id"]):
            fail(f"configuration: parent_runs.{k}.run_id must be a SESSABLRUN run id")
        if not SR._hex(e["config_sha256"]) or not SR._hex(e["restricted_sha256sums_sha256"]):
            fail(f"configuration: parent_runs.{k} SHA-256 values malformed")
        if e["early_stopping_patience"] != PARENT_PATIENCE[k] or isinstance(e["early_stopping_patience"], bool):
            fail(f"configuration: parent_runs.{k}.early_stopping_patience must be {PARENT_PATIENCE[k]}")
    co = cfg["code"]
    _keys(co, list(CODE_FILES), "code")
    if not all(SR._hex(v) for v in co.values()):
        fail("configuration: code SHA-256 values malformed")
    check_architectures(cfg["architectures"])
    tr = cfg["training"]
    _keys(tr, ["checkpoint", "recipe", "threads", "timing", "trainer", "trial_overrides"], "training")
    for k in ("checkpoint", "recipe", "timing", "trainer"):
        _text(tr[k], f"training.{k}")
    check_threads_config(tr["threads"])
    if tr["trial_overrides"] != {"max_epochs": TRIAL_EPOCHS} or isinstance(tr["trial_overrides"].get("max_epochs"), bool):
        fail(f"configuration: training.trial_overrides must be exactly {{\"max_epochs\": {TRIAL_EPOCHS}}} (the rest of the recipe unchanged)")
    if "none" not in tr["checkpoint"]:
        fail("configuration: training.checkpoint must say that no checkpoint is saved in the trial")
    t = cfg["trial"]
    _keys(t, ["arm", "cap_checks", "cap_unit", "dataset", "estimate", "estimate_minutes", "fold", "frames", "inference", "inference_batch",
              "networks", "process", "roles", "seed", "upper_estimate_minutes", "wall_time_cap_minutes"], "trial")
    if {k: t[k] for k in TRIAL_UNIT} != TRIAL_UNIT or any(isinstance(t[k], bool) for k in TRIAL_UNIT):
        fail(f"configuration: the trial unit must be {TRIAL_UNIT}")
    if t["roles"] != list(TRIAL_ROLES):
        fail(f"configuration: trial.roles must be {list(TRIAL_ROLES)} (never the test)")
    if t["networks"] != list(AM.NETWORKS):
        fail(f"configuration: trial.networks must be {list(AM.NETWORKS)}, in this order")
    if t["inference_batch"] != FR.SCORE_BATCH or isinstance(t["inference_batch"], bool):
        fail(f"configuration: trial.inference_batch must be {FR.SCORE_BATCH} (the scoring batch of the project)")
    if not FR.is_count(t["wall_time_cap_minutes"]) or not 1 <= t["wall_time_cap_minutes"] <= CAP_LIMIT_MINUTES:
        fail(f"configuration: trial.wall_time_cap_minutes must be an integer from 1 to {CAP_LIMIT_MINUTES}")
    if t["cap_unit"] != CAP_UNIT:
        fail(f"configuration: trial.cap_unit must be {CAP_UNIT!r}")
    _number(t["estimate_minutes"], "trial.estimate_minutes")
    _number(t["upper_estimate_minutes"], "trial.upper_estimate_minutes")
    if not t["estimate_minutes"] <= t["upper_estimate_minutes"] < t["wall_time_cap_minutes"]:
        fail("configuration: trial needs estimate_minutes <= upper_estimate_minutes < wall_time_cap_minutes")
    for k in ("cap_checks", "estimate", "inference", "process"):
        _text(t[k], f"trial.{k}")
    _keys(t["frames"], list(TRIAL_ROLES), "trial.frames")
    for r in TRIAL_ROLES:
        f = t["frames"][r]
        _keys(f, ["attack", "bona_fide", "n", "subjects"], f"trial.frames.{r}")
        for k in f:
            _int(f[k], f"trial.frames.{r}.{k}", 1)
        if f["n"] != f["attack"] + f["bona_fide"]:
            fail(f"configuration: trial.frames.{r}.n must be attack + bona_fide")
    bc = cfg["baseline_check"]
    _keys(bc, ["card_sha256", "consequence", "difference", "epochs", "fields", "model", "reference_run", "rule", "source"], "baseline_check")
    if bc["reference_run"] != pr_["patience_4"]["run_id"]:
        fail("configuration: baseline_check.reference_run must be the accepted run at patience 4")
    if bc["model"] != SR.model_name(TRIAL_UNIT["arm"], TRIAL_UNIT["fold"], TRIAL_UNIT["seed"]):
        fail("configuration: baseline_check.model must be the model of the trial unit")
    if bc["epochs"] != TRIAL_EPOCHS or isinstance(bc["epochs"], bool):
        fail(f"configuration: baseline_check.epochs must be {TRIAL_EPOCHS}")
    if bc["fields"] != list(HISTORY_KEYS):
        fail(f"configuration: baseline_check.fields must be {list(HISTORY_KEYS)} (the shared-history fields of the project)")
    if bc["difference"] != 0 or isinstance(bc["difference"], bool):
        fail("configuration: baseline_check.difference must be 0 (identical histories, the rule of the project)")
    if not SR._hex(bc["card_sha256"]):
        fail("configuration: baseline_check.card_sha256 malformed")
    for k in ("consequence", "rule", "source"):
        _text(bc[k], f"baseline_check.{k}")
    if "not reproduced" not in bc["consequence"] or "pre-registration" not in bc["consequence"]:
        fail("configuration: baseline_check.consequence must state what follows if the history is not reproduced (pre-registration)")
    df = cfg["definitions"]
    _keys(df, ["bootstrap", "comparison", "conditions", "decision", "eer_threshold", "fixed_threshold", "frame_ids", "one_class",
               "per_seed_reading", "primary_analysis", "sign", "subject_sensitivity"], "definitions")
    check_definitions_added(df)
    if df["bootstrap"] != SR.BOOT:
        fail(f"configuration: definitions.bootstrap must be the recipe of the project {SR.BOOT}")
    if df["conditions"] != [list(c) for c in D.CONDITIONS]:
        fail("configuration: definitions.conditions must list the 8 conditions (patience, BN state, threshold) of the definitions")
    if df["decision"] != FR.DECISION_RULE or df["eer_threshold"] != FR.THRESHOLD_CRITERION:
        fail("configuration: definitions.decision and eer_threshold must be the rule and the EER criterion of the project")
    if df["fixed_threshold"] != FR.FIXED_THRESHOLD or isinstance(df["fixed_threshold"], bool):
        fail(f"configuration: definitions.fixed_threshold must be {FR.FIXED_THRESHOLD}")
    for k in ("comparison", "one_class", "sign", "subject_sensitivity"):
        _text(df[k], f"definitions.{k}")
    banned = [w for w in BANNED if w in json.dumps([df, de]).lower()]
    if banned:
        fail(f"configuration: the definitions and declarations must not use causal or conclusive wording ({', '.join(banned)})")
    pl = cfg["plan"]
    _keys(pl, ["bn_states", "caps", "costs", "datasets", "machine_rule", "max_epochs", "networks", "order", "patiences", "runs", "thresholds"],
          "plan")
    if (pl["networks"], pl["patiences"], pl["bn_states"], pl["thresholds"]) != (list(AM.NETWORKS), list(D.PATIENCES), list(D.BN_STATES),
                                                                                  list(D.THRESHOLD_RULES)):
        fail("configuration: plan networks, patiences, BN states and thresholds must be those of the definitions")
    if pl["max_epochs"] != FR.FROZEN_TRAINING["max_epochs"] or isinstance(pl["max_epochs"], bool):
        fail(f"configuration: plan.max_epochs must be {FR.FROZEN_TRAINING['max_epochs']} (the recipe)")
    _keys(pl["costs"], ["evaluation", "rates", "totals", "training"], "plan.costs")
    for k in pl["costs"]:
        _text(pl["costs"][k], f"plan.costs.{k}")
    if not isinstance(pl["datasets"], list) or len(pl["datasets"]) != len(DATASETS):
        fail(f"configuration: plan.datasets must list {', '.join(DATASETS)}")
    for i, ds in enumerate(pl["datasets"]):
        check_dataset(ds, i)
    if pl["datasets"][0]["access"] != "available":
        fail("configuration: plan.datasets[0] (NUAA) must be available")
    check_plan_added(pl, pl["datasets"][0])
    _keys(cfg["outputs"], ["compare_trials", "plan", "trial"], "outputs")
    if cfg["outputs"]["trial"] != [STATUS, LOG, *TRIAL_OUTPUTS] or cfg["outputs"]["plan"] != list(PLAN_OUTPUTS) \
            or cfg["outputs"]["compare_trials"] != list(COMPARE_OUTPUTS):
        fail(f"configuration: outputs must be trial {[STATUS, LOG, *TRIAL_OUTPUTS]}, plan {list(PLAN_OUTPUTS)} and compare_trials "
             f"{list(COMPARE_OUTPUTS)}")
    _keys(cfg["environment"], ["keras", "tensorflow"], "environment")
    _text(cfg["environment"]["tensorflow"], "environment.tensorflow")
    _text(cfg["environment"]["keras"], "environment.keras")
    pcfg = check_parent_configs(cfg, base)
    return cfg, sha256_bytes(data), pcfg


def check_parent_configs(cfg: dict, base: Path) -> dict:
    """Le configurazioni dei due run accettati: SHA-256 fissato, validate da SR.load_config; la seconda è l'emendamento
    della prima (parent_config_sha256) con il solo campo training.early_stopping_patience cambiato (4 -> 12); la ricetta
    è FR.FROZEN_TRAINING; i numeri di NUAA nel piano (split, seed, frame del fitting, del dev e del test per split; i
    run di riferimento della riga LivenessNet) sono quelli di queste configurazioni."""
    pc = {}
    for k in PARENT_KEYS:
        e = cfg["parent_runs"][k]
        cp = rel_path(base, e["config_path"])
        b = read_regular(cp, f"configuration of the accepted run {k}")
        if sha256_bytes(b) != e["config_sha256"]:
            fail(f"configuration of the accepted run {k} ({cp.name}): SHA-256 {sha256_bytes(b)} differs from parent_runs.{k}.config_sha256")
        pc[k], _ = SR.load_config(cp)
    p4, p12 = pc["patience_4"], pc["patience_12"]
    if p4["training"] != FR.FROZEN_TRAINING:
        fail("configuration of the accepted run patience_4: training is not the recipe (FR.FROZEN_TRAINING)")
    am = p12.get(SR.AMENDMENT_KEY) or {}
    if am.get("parent_config_sha256") != cfg["parent_runs"]["patience_4"]["config_sha256"] or am.get("changed_field") != "training.early_stopping_patience":
        fail("configuration of the accepted run patience_12: not the amendment of the patience_4 configuration on early_stopping_patience")
    if p12["training"] != {**p4["training"], "early_stopping_patience": 12}:
        fail("configuration of the accepted run patience_12: training differs from patience_4 in more than early_stopping_patience 4 -> 12")
    nuaa = cfg["plan"]["datasets"][0]
    want_fp = {"fitting": p4["arms"][TRIAL_UNIT["arm"]]["total"], "dev": p4["dev"]["total"], "test_per_split": p4["schema_expected"]["test_per_fold"]}
    if nuaa["splits"] != list(range(1, p4["schema_expected"]["k"] + 1)) or nuaa["seeds"] != p4["train_seeds"] or nuaa["frames_per_unit"] != want_fp \
            or nuaa["unit_kind"] != "fold":
        fail(f"configuration: plan NUAA must have folds 1-{p4['schema_expected']['k']}, seeds {p4['train_seeds']} and frames per unit {want_fp} "
             "(the accepted configurations)")
    ref = {(e["network"], e["patience"]): e["reference_run"] for e in nuaa["trainings"]}
    want_ref = {("LivenessNet", 4): cfg["parent_runs"]["patience_4"]["run_id"], ("LivenessNet", 12): cfg["parent_runs"]["patience_12"]["run_id"],
                ("ResNet-18", 4): None, ("ResNet-18", 12): None}
    if ref != want_ref:
        fail("configuration: plan NUAA reference_run must be the accepted runs for LivenessNet (history reproduction check) and null for ResNet-18")
    if cfg["trial"]["frames"]["fitting"]["n"] != want_fp["fitting"] or cfg["trial"]["frames"]["dev"]["n"] != want_fp["dev"]:
        fail("configuration: trial.frames differ from the arm B fitting and the common dev of the accepted configuration")
    if cfg["architectures"]["LivenessNet"]["path"] != p4["architecture"]["path"] or cfg["architectures"]["LivenessNet"]["sha256"] != p4["architecture"]["sha256"]:
        fail("configuration: architectures.LivenessNet differs from the architecture of the accepted run")
    return p4


def trial_training(cfg: dict, pcfg: dict) -> dict:
    """La ricetta della prova: quella del run accettato a patience 4 con le sole sostituzioni di training.trial_overrides."""
    return {**pcfg["training"], **cfg["training"]["trial_overrides"]}


# ============================================================================ ingressi

def check_code(cfg: dict) -> dict:
    got = {f: sha256_bytes(read_regular(ROOT / f, "code file")) for f in CODE_FILES}
    bad = [f for f in CODE_FILES if got[f] != cfg["code"][f]]
    if bad:
        fail(f"code files {bad} differ from the SHA-256 pinned in the configuration")
    return got


def load_parent_status(cfg: dict, base: Path, key: str, inp) -> dict:
    """status.json di un run accettato (PASS, run id, configurazione, SHA256SUMS riservato; i 12 modelli del braccio B
    completati) e il suo SHA256SUMS riservato ancorato. Nessun altro file del run è aperto qui."""
    e = cfg["parent_runs"][key]
    run = rel_path(base, e["folder"])
    st = SR.read_json(read_regular(run / STATUS, f"{key} status.json"), f"{key} status.json")
    if st.get("status") != "PASS" or st.get("run_id") != e["run_id"] or st.get("config_sha256") != e["config_sha256"] \
            or st.get("restricted_sums_sha256") != e["restricted_sha256sums_sha256"]:
        fail(f"accepted run {e['run_id']}: status.json is not PASS with the run id, configuration and restricted SHA256SUMS of the configuration")
    arm_b = [SR.model_name(TRIAL_UNIT["arm"], f, s) for f in cfg["plan"]["datasets"][0]["splits"] for s in cfg["plan"]["datasets"][0]["seeds"]]
    if not isinstance(st.get("completed"), list) or any(n not in st["completed"] for n in arm_b):
        fail(f"accepted run {e['run_id']}: the {len(arm_b)} arm B models are not all completed")
    b = read_regular(run / BR.RESTRICTED / "SHA256SUMS", f"{key} restricted SHA256SUMS")
    if sha256_bytes(b) != e["restricted_sha256sums_sha256"]:
        fail(f"accepted run {e['run_id']}: restricted SHA256SUMS has SHA-256 {sha256_bytes(b)}, expected {e['restricted_sha256sums_sha256']}")
    return FR.read_sums(b, f"{key} restricted SHA256SUMS")


def load_reference_card(cfg: dict, base: Path, sums: dict) -> dict:
    """La scheda del modello di riferimento (B fold 1 seed 42, run a patience 4): elencata nel SHA256SUMS riservato con
    lo SHA-256 della configurazione, braccio, fold e seed giusti, storia con le cinque grandezze e almeno 3 epoche."""
    bc = cfg["baseline_check"]
    run = rel_path(base, cfg["parent_runs"]["patience_4"]["folder"])
    fn = f"{bc['model']}_card.json"
    if sums.get(fn) != bc["card_sha256"]:
        fail(f"accepted run {bc['reference_run']}: {fn} is not listed in the restricted SHA256SUMS with baseline_check.card_sha256")
    b = read_regular(BR.parent_path(run, bc["model"], "card.json"), f"reference {fn}")
    if sha256_bytes(b) != bc["card_sha256"]:
        fail(f"reference {fn}: SHA-256 {sha256_bytes(b)} differs from baseline_check.card_sha256")
    card = SR.read_json(b, f"reference {fn}")
    if (card.get("arm"), card.get("fold"), card.get("train_seed")) != (TRIAL_UNIT["arm"], TRIAL_UNIT["fold"], TRIAL_UNIT["seed"]):
        fail(f"reference {fn}: arm, fold or seed differ from the trial unit")
    if not SR._hex(card.get("fitting_keys_sha256")) or not SR._hex(card.get("dev_keys_sha256")):
        fail(f"reference {fn}: fitting_keys_sha256 or dev_keys_sha256 missing or malformed")
    h = card.get("history")
    if not isinstance(h, dict) or any(not isinstance(h.get(k), list) or len(h[k]) < bc["epochs"] for k in HISTORY_KEYS):
        fail(f"reference {fn}: the history lacks one of {', '.join(HISTORY_KEYS)} for the first {bc['epochs']} epochs")
    return card


def keys_check(card: dict, role: str, rows) -> bool:
    """Le chiavi di un ruolo della prova contro la scheda di riferimento: impronta come nel runner accettato (BR.keys_sha:
    chiavi ordinate unite da a capo) uguale a <ruolo>_keys_sha256 della scheda; altrimenti FAIL (nessuna chiave nel
    messaggio)."""
    if BR.keys_sha(rows) != card[f"{role}_keys_sha256"]:
        fail(f"trial {role}: the keys differ from {role}_keys_sha256 of the reference card (not the data of the accepted model)")
    return True


def role_counts(rows) -> dict:
    return {"n": len(rows), "bona_fide": sum(1 for r in rows if r["label"] == "0"), "attack": sum(1 for r in rows if r["label"] == "1"),
            "subjects": len({r["subject"] for r in rows})}


def assert_no_test(inp) -> None:
    """Controllo esplicito: lo schema dei fold (che contiene il test) non è caricato e nessun ruolo diverso da fitting e
    dev è stato assemblato."""
    if inp.folds:
        fail("the fold schema was loaded: this runner must never open the test manifests")
    other = sorted(set(inp.assembled) - set(TRIAL_ROLES))
    if other:
        fail(f"roles {other} were assembled: only the fitting and dev frames may be assembled")


def trial_frames(store, inp, role: str) -> dict:
    """I frame di un ruolo della prova (BR.frames: celle del braccio e dev comune del fold); ogni altro ruolo è rifiutato
    prima di assemblare."""
    inp.assembled.append(role)
    assert_no_test(inp)
    return BR.frames(store, inp, role, TRIAL_UNIT["arm"], TRIAL_UNIT["fold"])


def load_inputs(cfg: dict, pcfg: dict, base: Path):
    """Ingressi: codice e architettura per SHA-256, status dei due run accettati, scheda di riferimento, poi la sequenza
    di nuaa_bn_recalibration.load_inputs per il fitting e il dev (manifest M2, cache, celle, frame estratti, sorgenti),
    conteggi della prova dalle righe. Nessun frame assemblato, TensorFlow non importato."""
    refused = [v for v in FR.ENV_REFUSED if os.environ.get(v, "").strip()]
    if refused:
        fail(f"environment variables {refused} are set: they would redirect the reference code (unset them)")
    inp = BR.Inputs()
    inp.assembled = []
    code = check_code(cfg)
    ap = rel_path(base, cfg["architectures"]["LivenessNet"]["path"])
    h = sha256_bytes(read_regular(ap, "architecture file"))
    if h != cfg["architectures"]["LivenessNet"]["sha256"]:
        fail(f"architecture file {ap.name}: SHA-256 {h} differs from the configuration")
    sums = {k: load_parent_status(cfg, base, k, inp) for k in PARENT_KEYS}
    inp.reference_card = load_reference_card(cfg, base, sums["patience_4"])
    # la sequenza di nuaa_bn_recalibration.load_inputs (righe 535-545), invariata
    v = SR.fr_view(pcfg)
    FR.load_m2_manifests(v, base, inp)
    cache_keys = {r["key"] for n in FR.CACHE_MANIFESTS for r in inp.rows[n]}
    inp.to_extract = sorted(k for k in inp.official if k not in cache_keys)
    FR.load_cache(v, base, inp)
    SR.load_cells_run(pcfg, base, inp)
    BR.load_folds_frames_rows(pcfg, base, inp)
    SR.load_frames(pcfg, base, inp)
    SR.check_sources(pcfg, inp)
    assert_no_test(inp)
    for role in TRIAL_ROLES:
        got = role_counts(BR.role_rows(inp, role, TRIAL_UNIT["arm"], TRIAL_UNIT["fold"]))
        if got != cfg["trial"]["frames"][role]:
            fail(f"trial {role}: the manifest holds {got}, the configuration {cfg['trial']['frames'][role]}")
        keys_check(inp.reference_card, role, BR.role_rows(inp, role, TRIAL_UNIT["arm"], TRIAL_UNIT["fold"]))
    inp.code = {"architecture": h, **code, "scripts/pad_arch_comparison.py": sha256_bytes(read_regular(Path(__file__), "code file"))}
    inp.environment = FR.environment()
    for k in ("tensorflow", "keras"):
        got = inp.environment["tensorflow_installed"] if k == "tensorflow" else inp.environment.get("keras")
        if got != cfg["environment"][k]:
            fail(f"{k} {got} installed, configuration {cfg['environment'][k]} (checked without importing TensorFlow)")
    inp.checks["test"] = TEST_READ
    return inp


# ============================================================================ macchina, tempi, inferenza

def machine() -> dict:
    """La macchina senza nome dell'host, utente o percorsi: processore, gruppi di core con il nome che dà loro il
    sistema (macOS: hw.perflevelN.name, per esempio «Performance» ed «Efficiency» sull'Apple M4, «Super» e
    «Performance» sull'Apple M5 Pro) e il numero di core fisici di ciascuno (lista vuota se il sistema non li riporta),
    RAM, sistema, numero di CPU logiche. Nella PR #29 i due gruppi erano registrati con i nomi fissi
    performance_cores / efficiency_cores, sbagliati sull'Apple M5 Pro (difetto nostro, corretto qui). Il numero dei
    gruppi è hw.nperflevels (assente, per esempio su Intel o Linux: lista vuota); il nome di ciascun gruppo è
    hw.perflevelN.name, i suoi core hw.perflevelN.physicalcpu. Senza sysctl (Linux) o con il comando fallito: lista
    vuota e ram_gb None."""
    sysctl = SYSCTL or sysctl_value
    mem = sysctl("hw.memsize")
    num = lambda x: int(x) if x is not None and x.isdigit() else None      # noqa: E731
    mac = platform.mac_ver()[0]
    return {"processor": sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown",
            "core_levels": [{"name": sysctl(f"hw.perflevel{i}.name"), "physical_cores": num(sysctl(f"hw.perflevel{i}.physicalcpu"))}
                            for i in range(num(sysctl("hw.nperflevels")) or 0)],
            "ram_gb": round(int(mem) / 2 ** 30, 1) if mem and mem.isdigit() else None,
            "os": f"macOS {mac}" if mac else f"{platform.system()} {platform.release()}", "logical_cpus": os.cpu_count(),
            "architecture": platform.machine()}


def sysctl_value(k: str):
    """Il valore di sysctl -n <k> come testo, oppure None (comando assente, fallito o valore vuoto)."""
    try:
        r = subprocess.run(["sysctl", "-n", k], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def core_levels_text(m: dict) -> str:
    """I gruppi di core di un record della macchina come testo («5 Super, 10 Performance»); «not reported» se mancano;
    un nome o un numero di core mancante in un gruppo è «not reported» al suo posto."""
    levels = m.get("core_levels") or []
    nr = "not reported"
    return ", ".join(f"{nr if lv.get('physical_cores') is None else lv['physical_cores']} {nr if lv.get('name') is None else lv['name']}"
                     for lv in levels) if levels else nr


def ram_text(m: dict) -> str:
    """La RAM di un record della macchina come testo: «24.0 GB RAM», oppure «RAM not reported» (ram_gb None)."""
    return "RAM not reported" if m.get("ram_gb") is None else f"{m['ram_gb']} GB RAM"


def json_text(doc, what: str) -> str:
    """Un JSON pubblico: nessun NaN né infinito (allow_nan=False); un valore non serializzabile o non finito -> Failure
    prima di ogni scrittura."""
    try:
        return json.dumps(doc, indent=1, allow_nan=False) + "\n"
    except (ValueError, TypeError) as e:
        fail(f"{what}: not written ({e}): a value is not finite or not serializable")


def thread_environment(cfg: dict) -> dict:
    """Le variabili d'ambiente dei thread di TensorFlow (TF_NUM_INTRAOP_THREADS, TF_NUM_INTEROP_THREADS): per ciascuna il
    valore trovato, come testo, oppure None se assente. Una variabile presente con un valore diverso da quello di
    training.threads (testo senza spazi ai lati diverso dall'intero della configurazione; anche vuota) -> Failure, da
    chiamare prima di ogni lavoro; presente e uguale, oppure assente: si procede."""
    th = cfg["training"]["threads"]
    env = {}
    for k in THREAD_KEYS:
        var = TF_THREAD_ENV[k]
        if var in os.environ:
            if os.environ[var].strip() != str(th[k]):
                fail(f"environment variable {var}={os.environ[var]!r} contradicts training.threads.{k} = {th[k]} of the configuration "
                     f"(unset it, or set it to {th[k]})")
            env[var] = os.environ[var]
        else:
            env[var] = None
    return env


def read_threads(tf) -> dict:
    """I thread effettivi letti da tf.config.threading."""
    return {"intra_op": tf.config.threading.get_intra_op_parallelism_threads(), "inter_op": tf.config.threading.get_inter_op_parallelism_threads()}


def check_threads(tf, cfg: dict, what: str) -> dict:
    """I thread effettivi riletti dai getter uguali a quelli di training.threads; altrimenti Failure."""
    want = {k: cfg["training"]["threads"][k] for k in THREAD_KEYS}
    got = read_threads(tf)
    if got != want:
        fail(f"{what}: TensorFlow threads {got} differ from training.threads {want} of the configuration")
    return got


def set_threads(tf, cfg: dict) -> dict:
    """Unico punto in cui un comando che usa TensorFlow ne fissa i thread (oggi trial; lo stesso per i comandi futuri del
    piano), da chiamare subito dopo l'import di TensorFlow e prima di qualunque sua operazione: controlla le variabili
    d'ambiente (thread_environment), imposta intra-op e inter-op con tf.config.threading ai valori di training.threads
    (un RuntimeError, cioè TensorFlow già inizializzato con altri valori, diventa Failure), li rilegge con i getter e
    fallisce se diversi; nient'altro del threading di TensorFlow è impostato. Restituisce il record: valori richiesti, effettivi, set_by_this_runner true, variabili
    d'ambiente trovate (o None), os.cpu_count()."""
    want = {k: cfg["training"]["threads"][k] for k in THREAD_KEYS}
    env = thread_environment(cfg)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(want["intra_op"])
        tf.config.threading.set_inter_op_parallelism_threads(want["inter_op"])
    except RuntimeError as e:
        fail(f"the TensorFlow threads cannot be set ({e}): TensorFlow was initialised before the runner set them")
    got = check_threads(tf, cfg, "after setting")
    return {"requested": want, "effective": got, "set_by_this_runner": True, "environment": env, "logical_cpus": os.cpu_count()}


def cpu_seconds() -> float:
    """Tempo CPU del processo su tutti i thread (os.times: utente + sistema)."""
    t = os.times()
    return t.user + t.system


class EpochTimer:
    """Registra per ogni epoca la storia (le cinque grandezze dei log a fine epoca, già con il lr scritto dallo
    scheduler) e i tempi: secondi totali dell'epoca (orologio monotono, da on_epoch_begin a on_epoch_end), secondi della
    sola validazione sul dev (on_test_begin / on_test_end), secondi di CPU del processo su tutti i thread
    (time.process_time). Non legge né cambia pesi, ottimizzatore o dati."""

    def __init__(self, record: list):
        self.record = record

    def callback(self, tf):
        timer = self

        class EpochTiming(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                timer.t0, timer.c0, timer.val = time.monotonic(), time.process_time(), 0.0

            def on_test_begin(self, logs=None):
                timer.v0 = time.monotonic()

            def on_test_end(self, logs=None):
                timer.val += time.monotonic() - timer.v0

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                missing = [k for k in HISTORY_KEYS if k not in logs]
                if missing:
                    fail(f"epoch {epoch + 1}: the logs lack {missing}")
                timer.record.append({"epoch": epoch + 1, **{k: float(logs[k]) for k in HISTORY_KEYS},
                                     "seconds_total": time.monotonic() - timer.t0, "seconds_validation": timer.val,
                                     "cpu_seconds": time.process_time() - timer.c0})
        return EpochTiming()


def inference_pass(model, X: np.ndarray, batch: int) -> dict:
    """Una passata di sola inferenza (training=False, frame / 255 per batch, come lo scorer del progetto), su CPU:
    secondi (orologio monotono) e secondi di CPU; le probabilità sono controllate (finite, in [0, 1]) e scartate."""
    import tensorflow as tf
    t0, c0 = time.monotonic(), time.process_time()
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].astype(np.float32) / 255.0
        with tf.device("/CPU:0"):
            p = np.asarray(model(xb, training=False), dtype=np.float64)
        if p.shape != (len(xb), 2) or not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
            fail("inference pass: probabilities non-finite, outside [0, 1] or of the wrong shape")
    return {"frames": int(len(X)), "batch": batch, "seconds": time.monotonic() - t0, "cpu_seconds": time.process_time() - c0}


def network_rates(rec: dict, n_fit: int, n_dev: int) -> dict:
    """Velocità di una rete dalla prova (unica regola, usata da trial.md e da plan): secondi di addestramento per 1.000
    frame di fitting per epoca = media sulle epoche misurate, la prima compresa (che include il tracing del grafo: stima
    prudente), di (totale − validazione) / frame del fitting × 1.000; secondi di validazione per 1.000 frame = media di
    validazione / frame del dev × 1.000; secondi d'inferenza per 1.000 frame = (passata sul fitting + passata sul dev) /
    frame valutati × 1.000; secondi per addestramento fuori dalle epoche = training_overhead della prova; rapporto CPU /
    tempo reale = CPU di epoche, passate e costo fuori dalle epoche / loro tempo reale. None dove una misura manca."""
    ep = rec.get("epochs") or []
    out = {"train_seconds_per_1000_fitting_frames_per_epoch": None, "validation_seconds_per_1000_frames": None,
           "inference_seconds_per_1000_frames": None, "overhead_seconds_per_training": None, "cpu_over_wall": None,
           "epochs_measured": len(ep)}
    if ep:
        out["train_seconds_per_1000_fitting_frames_per_epoch"] = float(np.mean([(e["seconds_total"] - e["seconds_validation"]) / n_fit * 1000 for e in ep]))
        out["validation_seconds_per_1000_frames"] = float(np.mean([e["seconds_validation"] / n_dev * 1000 for e in ep]))
    inf = rec.get("inference") or {}
    if all(r in inf for r in TRIAL_ROLES):
        out["inference_seconds_per_1000_frames"] = sum(inf[r]["seconds"] for r in TRIAL_ROLES) / sum(inf[r]["frames"] for r in TRIAL_ROLES) * 1000
    ov = rec.get("training_overhead")
    if ov is not None:
        out["overhead_seconds_per_training"] = ov["seconds"]
    wall = sum(e["seconds_total"] for e in ep) + sum(inf[r]["seconds"] for r in inf) + (ov["seconds"] if ov else 0.0)
    cpu = sum(e["cpu_seconds"] for e in ep) + sum(inf[r]["cpu_seconds"] for r in inf) + (ov["cpu_seconds"] if ov else 0.0)
    if wall > 0:
        out["cpu_over_wall"] = cpu / wall
    return out


def build_checked(cfg: dict, net: str, arch):
    """Il modello di una rete costruito e controllato (AM.check_model) prima del training: nessun numero casuale usato
    dal controllo."""
    shape = cfg["architectures"][net]["input_shape"]
    try:
        model = (BUILD or AM.build)(net, shape, arch)
        AM.check_model(model, net, shape, {"trainable": cfg["architectures"][net]["trainable_parameters"],
                                           "non_trainable": cfg["architectures"][net]["non_trainable_parameters"],
                                           "batchnorm_layers": cfg["architectures"][net]["batchnorm_layers"]})
    except AM.ModelProblem as e:
        fail(str(e))
    return model


def trial_network(cfg: dict, pcfg: dict, net: str, arch, fit: dict, dev: dict, budget, rec: dict) -> None:
    """Una rete della prova: training di 3 epoche con trial_trainer (timer nel record, che resta con le epoche misurate
    anche se il tetto ferma il training), poi le due passate d'inferenza; alla fine i thread effettivi riletti
    (check_threads, uguali a training.threads) nel record della rete, già scritti prima del training (una rete fermata
    dal tetto li ha comunque); rec è riempito man mano."""
    import tensorflow as tf
    budget.check(f"train {net}")
    rec["threads"] = check_threads(tf, cfg, f"before {net}")
    training = trial_training(cfg, pcfg)
    data = {"X_fit": fit["X"], "y_fit": fit["y"], "X_dev": dev["X"], "y_dev": dev["y"]}
    timer = EpochTimer(rec["epochs"])
    t0, c0 = time.monotonic(), time.process_time()
    card, model = (TRAINER or trial_trainer)(data, training, net, TRIAL_UNIT["seed"], lambda: build_checked(cfg, net, arch),
                                             lambda step: budget.check(f"{step} of {net}"), timer)
    if [{k: e[k] for k in HISTORY_KEYS} for e in rec["epochs"]] != [{k: card["history"][k][i] for k in HISTORY_KEYS} for i in range(card["epochs_run"])]:
        fail(f"{net}: the history recorded by the timing callback differs from the Keras history")
    # costo dell'addestramento fuori dalle epoche: costruzione del modello, preparazione dei dati prima del fit, tempi fra
    # le epoche e dopo l'ultima (ripristino dei pesi), cioè tutto il trainer meno la somma delle epoche
    rec["training_overhead"] = {"seconds": time.monotonic() - t0 - sum(e["seconds_total"] for e in rec["epochs"]),
                                "cpu_seconds": time.process_time() - c0 - sum(e["cpu_seconds"] for e in rec["epochs"])}
    counts = AM.model_counts(model)
    rec.update({"parameters": {"trainable": counts["trainable"], "non_trainable": counts["non_trainable"]},
                "batchnorm_layers": counts["batchnorm_layers"], "steps_per_epoch": math.ceil(card["n_fitting"] / card["batch"]),
                "card": {k: card[k] for k in ("arch", "train_seed", "n_fitting", "n_dev", "max_epochs", "epochs_run", "restored_epoch",
                                              "restored_by", "device", "visible_devices", "lr", "batch", "early_stopping_patience", "scheduler",
                                              "class_weight", "train_seconds", "checkpoint")}})
    for e in rec["epochs"]:
        print(f"   {net} epoch {e['epoch']}: {e['seconds_total']:.1f} s (validation {e['seconds_validation']:.1f} s, process CPU "
              f"{e['cpu_seconds']:.1f} s)", flush=True)
    for role, part in (("fitting", fit), ("dev", dev)):
        budget.check(f"inference on the {role} of {net}")
        rec["inference"][role] = inference_pass(model, part["X"], cfg["trial"]["inference_batch"])
        print(f"   {net} inference on the {role}: {part['X'].shape[0]} frames in {rec['inference'][role]['seconds']:.1f} s", flush=True)
    rec["threads"] = check_threads(tf, cfg, f"after {net}")       # thread effettivi riletti a fine rete, uguali alla configurazione
    del model
    tf.keras.backend.clear_session()
    gc.collect()


def trial_trainer(data: dict, training: dict, network: str, train_seed: int, build_model, cap_check, timer) -> tuple:
    """Copia dichiarata di nuaa_folds_run.default_trainer (la ricetta di n1_trainer.train, riga per riga), con le sole
    sostituzioni di TRAINER_SUBSTITUTIONS: il modello viene da build_model() (una rete del registro) invece che da
    create_model("LivenessNet"); nessun checkpoint salvato (nessun weights_path); la callback di timer.callback(tf),
    che registra i tempi e la storia e non tocca il training, è inserita prima di Cap; la scheda dice la rete e che
    nessun checkpoint è salvato; restituisce (scheda, modello) per le passate di inferenza della prova."""
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
    best = int(np.argmin(h["val_loss"]))
    if es.stopped_epoch == 0 and es.best_weights is not None:
        model.set_weights(es.best_weights)           # epoche finite prima dell'early stopping: pesi migliori anche qui
    card = {"arch": network, "train_seed": int(train_seed), "n_fitting": int(len(ytr)), "n_dev": int(len(yva)),
            "max_epochs": int(epochs), "epochs_run": len(h["loss"]), "restored_epoch": best + 1,
            "restored_by": "early stopping" if es.stopped_epoch else "end of epochs (set_weights)",
            "best_val_loss": float(h["val_loss"][best]), "val_acc_at_best": float(h["val_accuracy"][best]),
            "history": {k: [float(v) for v in h[k]] for k in ("loss", "accuracy", "val_loss", "val_accuracy", "lr") if k in h},
            "device": "CPU", "visible_devices": devices, "lr": lr, "batch": batch, "early_stopping_patience": patience,
            "scheduler": training["scheduler"], "class_weight": {str(k): float(v) for k, v in cw.items()}, "train_seconds": round(train_s, 1),
            "test_evaluation": "none (no test frame is assembled in the trial)", "cpu_nondeterminism": "declared (N1): parallel reductions on the CPU "
            "may give minimal differences between two runs with the same seed", "checkpoint": "none saved (trial)"}
    print(f"== {network} seed {train_seed}: {card['epochs_run']} epochs in {train_s:.0f}s, restored epoch {card['restored_epoch']} "
          f"(val_loss {card['best_val_loss']:.6f})", flush=True)
    return card, model


# ============================================================================ controllo della storia di LivenessNet

def history_differences(ref: list, new: list) -> dict:
    """Un campo della storia di due addestramenti confrontato epoca per epoca, sulle epoche presenti in entrambi (le
    prime min(len) epoche, numerate da 1): epoche confrontate; prima epoca in cui i due valori differiscono (None se in
    nessuna); scarto assoluto massimo (0.0 se nessuna differenza) e prima epoca in cui è raggiunto (None se nessuna
    differenza). Due valori sono uguali solo se identici (scarto 0, nessuna tolleranza). Valori ammessi: reali finiti
    (int o float di Python, intero o reale di NumPy, non booleani: D.finite_real); un valore non finito (NaN o
    infinito) in una delle due storie date, anche oltre le epoche confrontate, un booleano o un valore non numerico ->
    Failure; anche uno scarto non finito (due valori finiti enormi di segno opposto) -> Failure. Unica regola della prima epoca diversa, usata dal controllo della storia della prova
    (baseline_check) e da compare-trials (compare_histories)."""
    for h in (ref, new):
        if any(not D.finite_real(v) or isinstance(v, Fraction) for v in h):
            fail("history: a value is not a finite number (NaN, infinite or not numeric): the difference is not defined")
    m = min(len(ref), len(new))
    first = at = None
    mx = 0.0
    for i in range(m):
        if new[i] != ref[i]:
            d = float(abs(float(new[i]) - float(ref[i])))
            if not math.isfinite(d):
                fail("history: the difference of two values is not finite")
            if first is None:
                first = i + 1
            if at is None or d > mx:
                mx, at = d, i + 1
    return {"epochs_compared": m, "first_differing_epoch": first, "max_abs_difference": float(mx), "epoch_of_max_abs_difference": at}


def baseline_check(cfg: dict, card: dict, rec: dict) -> dict:
    """Le prime epoche della storia di LivenessNet nella prova contro quelle della scheda accettata, con la regola delle
    «storie comuni» del progetto (quella di PV.shared_history: le cinque grandezze identiche nelle epoche fissate,
    scarto 0, nessuna tolleranza), calcolata con una sola fonte, history_differences: per grandezza la prima epoca in cui
    i valori differiscono (first_differing_epoch), lo scarto assoluto massimo (max_abs_difference) e la prima epoca in
    cui è raggiunto (epoch_of_max_abs_difference); esito «reproduced» solo se nessuna grandezza ha un'epoca diversa
    (un test confronta l'esito con PV.shared_history, che qui non è chiamata: il suo campo «epoch» è l'epoca dello scarto
    massimo, non la prima epoca diversa). Un valore non finito nelle epoche confrontate -> Failure (la prova è FAIL, senza
    uscite). Esito «not reproduced»; «not checked» con meno epoche misurate di quelle fissate (prova INCOMPLETE); non
    cambia lo stato della prova."""
    bc = cfg["baseline_check"]
    n = bc["epochs"]
    ep = rec["epochs"]
    base = {"reference_run": bc["reference_run"], "model": bc["model"], "epochs_compared": n, "fields": list(HISTORY_KEYS),
            "rule": bc["rule"], "consequence": bc["consequence"]}
    if len(ep) < n:
        return {**base, "outcome": "not checked", "reason": f"{len(ep)} of {n} epochs measured", "first_differing_epoch": None,
                "max_abs_difference": None, "epoch_of_max_abs_difference": None}
    hn = {k: [e[k] for e in ep] for k in HISTORY_KEYS}
    f = {k: history_differences(card["history"][k][:n], hn[k][:n]) for k in HISTORY_KEYS}
    same = all(f[k]["epochs_compared"] == n and f[k]["first_differing_epoch"] is None for k in HISTORY_KEYS)
    return {**base, "outcome": "reproduced" if same else "not reproduced",
            "first_differing_epoch": {k: f[k]["first_differing_epoch"] for k in HISTORY_KEYS},
            "max_abs_difference": {k: f[k]["max_abs_difference"] for k in HISTORY_KEYS},
            "epoch_of_max_abs_difference": {k: f[k]["epoch_of_max_abs_difference"] for k in HISTORY_KEYS}}


# ============================================================================ uscite della prova

def fmt(v, nd: int = 1) -> str:
    return "" if v is None else f"{v:.{nd}f}"


def trial_md(doc: dict) -> str:
    u = doc["unit"]
    L = [f"# Technical trial {doc['run_id']}", "",
         f"Status: **{doc['status']}**. NUAA, arm B, fold {u['fold']}, seed {u['seed']}: fitting {u['frames']['fitting']['n']} frames, dev "
         f"{u['frames']['dev']['n']} frames; no test frame assembled. {doc['epochs_fixed']} fixed epochs per network, batch {doc['batch']}, "
         f"device {doc['device']}; no checkpoint saved. Wall-time cap {doc['cap_minutes']} {doc['cap_unit']}.", "",
         f"Started {doc['started']}, finished {doc['finished']}; wall time {doc['wall_seconds']:.0f} s, process CPU time (all threads) "
         f"{doc['process_cpu_seconds']:.0f} s; peak RSS {doc['peak_rss_mb']:.0f} MB.", "",
         f"Machine: {doc['machine']['processor']}, cores as named by the system: {core_levels_text(doc['machine'])}; "
         f"{doc['machine']['logical_cpus']} logical CPUs, {ram_text(doc['machine'])}, {doc['machine']['os']}; TensorFlow {doc['environment'].get('tensorflow_installed')}, "
         f"Keras {doc['environment'].get('keras')}, Python {doc['environment'].get('python')}, NumPy {doc['environment'].get('numpy')}.", "",
         threads_text(doc["threads"]), ""]
    if doc["stopped"]:
        L += [f"Stopped: {doc['stopped']['cause']} at {doc['stopped']['step']} ({doc['stopped']['elapsed_minutes']} min of wall time).", ""]
    L += ["## Epochs", "", "| network | epoch | loss | accuracy | val_loss | val_accuracy | lr | seconds | validation seconds | CPU seconds |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for net, rec in doc["networks"].items():
        for e in rec["epochs"]:
            L.append(f"| {net} | {e['epoch']} | {e['loss']:.6f} | {e['accuracy']:.4f} | {e['val_loss']:.6f} | {e['val_accuracy']:.4f} | "
                     f"{e['lr']:.6g} | {e['seconds_total']:.1f} | {e['seconds_validation']:.1f} | {e['cpu_seconds']:.1f} |")
    L += ["", "The histories are technical outputs of the trial, not results.", "", "## Inference and rates", "",
          "| network | trainable parameters | BN layers | steps per epoch | inference fitting s | inference dev s | train s per 1,000 fitting frames per epoch | "
          "validation s per 1,000 frames | inference s per 1,000 frames | s per training outside the epochs | CPU / wall |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for net, rec in doc["networks"].items():
        r, inf = rec["rates"], rec["inference"]
        L.append(f"| {net} | {rec.get('parameters', {}).get('trainable', '')} | {rec.get('batchnorm_layers', '')} | {rec.get('steps_per_epoch', '')} | "
                 f"{fmt(inf.get('fitting', {}).get('seconds'))} | {fmt(inf.get('dev', {}).get('seconds'))} | "
                 f"{fmt(r['train_seconds_per_1000_fitting_frames_per_epoch'])} | {fmt(r['validation_seconds_per_1000_frames'])} | "
                 f"{fmt(r['inference_seconds_per_1000_frames'])} | {fmt(r['overhead_seconds_per_training'])} | {fmt(r['cpu_over_wall'], 2)} |")
    L += ["", "Rates: mean over the measured epochs, the first included (it includes the graph tracing: a cautious estimate); "
          "outside the epochs: the whole trainer minus the sum of its epochs (model building, data preparation before the fit, "
          "time between and after the epochs).", "",
          "Data: the assembled fitting and dev keys are equal to fitting_keys_sha256 and dev_keys_sha256 of the reference card: "
          + ", ".join(f"{k} {v}" for k, v in doc["data_check"].items()) + "."]
    b = doc["baseline_check"]
    L += ["", "## LivenessNet history against the accepted card", "", f"Outcome: **{b['outcome']}** (first {b['epochs_compared']} epochs of "
          f"{b['model']} in {b['reference_run']}; fields {', '.join(b['fields'])}). Rule: {b['rule']}", ""]
    if b["max_abs_difference"] is not None:
        L += ["| field | first differing epoch | largest absolute difference | epoch of the largest difference |", "|---|---|---|---|"]
        for k in b["fields"]:
            L.append(f"| {k} | {epoch_text(b['first_differing_epoch'][k])} | {b['max_abs_difference'][k]:.3g} | "
                     f"{epoch_text(b['epoch_of_max_abs_difference'][k])} |")
        L += ["", "First differing epoch: the first epoch at which the two values differ; the largest difference is reported with "
              "the first epoch at which it is reached.", ""]
    L += [f"Consequence (fixed before the trial): {b['consequence']}", "", f"Command: `{doc['command']}`", ""]
    return "\n".join(L)


def epoch_text(e) -> str:
    return "none" if e is None else str(e)


def threads_text(th: dict) -> str:
    """Una riga sui thread di TensorFlow di un record del formato pad-arch-trial/2."""
    env = ", ".join(f"{k} {'absent' if v is None else repr(v)}" for k, v in th["environment"].items())
    return (f"TensorFlow threads {'set by this runner' if th['set_by_this_runner'] else 'not set by the runner'}: requested intra-op "
            f"{th['requested']['intra_op']} and inter-op {th['requested']['inter_op']}, effective (read back from tf.config.threading) "
            f"intra-op {th['effective']['intra_op']} and inter-op {th['effective']['inter_op']}; environment variables: {env}; logical "
            f"CPUs {th['logical_cpus']}.")


def personal_tokens() -> list:
    """Testi che nessuna uscita pubblicabile può contenere: cartella home, radice del repository, utente, nome dell'host."""
    out = [str(Path.home()), str(ROOT)]
    for f in (getpass.getuser, socket.gethostname):
        try:
            v = f()
        except Exception:  # noqa: BLE001
            v = ""
        if v and len(v) >= 3:
            out += [v, v.split(".")[0]]
    return sorted(set(t for t in out if t))


def output_problems(text: str, inp) -> list:
    """Contenuto vietato in un'uscita pubblicabile: le regole del progetto (FR.public_problems) e i testi personali."""
    bad = FR.public_problems(text, inp)
    bad += [f"personal text {t[:3]}..." for t in personal_tokens() if t in text]
    return bad


def show_warning(message, category, filename, lineno, file=None, line=None):
    """Avvisi Python nel run.log senza il percorso del file che li emette (nessun percorso personale nelle uscite)."""
    print(f"warning: {category.__name__}: {message}", file=sys.stderr, flush=True)


# ============================================================================ cartella del run

def repo_state(root: Path) -> dict:
    return REPO_STATE(root) if REPO_STATE is not None else SR.repo_state(root)


def run_id_for(date: str, state: dict) -> str:
    if not re.fullmatch(r"\d{8}", date or ""):
        fail("--date must be YYYYMMDD")
    rid = f"{date}-ARCHTRIAL-{state['commit']}"
    if not RUN_ID_RE.match(rid):
        fail(f"run id {rid} does not follow YYYYMMDD-ARCHTRIAL-<commit>")
    return rid


def resolve(a) -> tuple:
    base = Path(a.input_root).resolve() if a.input_root else ROOT
    cfg_p = Path(a.config) if Path(a.config).is_absolute() else ROOT / a.config
    return base, cfg_p


def input_paths(cfg: dict, pcfg: dict, base: Path, cfg_p: Path) -> dict:
    out = {"the configuration": cfg_p, "the LivenessNet architecture": rel_path(base, cfg["architectures"]["LivenessNet"]["path"]),
           "the manifests folder": rel_path(base, pcfg["manifests"]["folder"]), "the cache": rel_path(base, pcfg["cache"]["path"]),
           "the cells run": rel_path(base, pcfg["cells_run"]["folder"]), "the folds run": rel_path(base, pcfg["folds_run"]["folder"]),
           "the extracted frames": rel_path(base, pcfg["folds_run"]["frames_file"]["path"])}
    for k in PARENT_KEYS:
        out[f"the accepted run {k}"] = rel_path(base, cfg["parent_runs"][k]["folder"])
        out[f"the configuration of {k}"] = rel_path(base, cfg["parent_runs"][k]["config_path"])
    return out


def run_dir_problem(run: Path, out_root: Path, inputs: dict) -> str:
    """Motivo per cui la cartella del run non è ammessa ("" se lo è), prima di qualunque scrittura: radice d'uscita non
    collegamento, ingressi separati, dentro il repository solo sotto results/archcmp/."""
    if out_root.is_symlink():
        return "the output root is a symbolic link"
    for label, p in inputs.items():
        if pg.within(p, run) or (Path(p).is_dir() and pg.within(run, p)):
            return f"{label} {Path(p).name} is inside the run folder, or contains it: inputs and outputs must be separate"
    return mw.output_problem(run, ROOT, pg.BASE_PROTECTED, [ROOT / DEFAULT_OUT_ROOT], [ROOT / x for x in EXTRA_PROTECTED], inputs,
                             [STATUS, LOG, *TRIAL_OUTPUTS])


def final_checks(cfg: dict, cfg_p: Path, cfg_sha: str, base: Path) -> dict:
    """Alla fine: configurazione, codice importato e scheda di riferimento con gli stessi SHA-256 dell'inizio."""
    if sha256_bytes(read_regular(cfg_p, "configuration")) != cfg_sha:
        fail("the configuration changed during the run")
    check_code(cfg)
    run = rel_path(base, cfg["parent_runs"]["patience_4"]["folder"])
    bc = cfg["baseline_check"]
    if sha256_bytes(read_regular(BR.parent_path(run, bc["model"], "card.json"), "reference card")) != bc["card_sha256"]:
        fail("the reference card changed during the run")
    return {"configuration_unchanged": True, "code_unchanged": len(CODE_FILES), "reference_card_unchanged": True}


# ============================================================================ comandi

def cmd_check_inputs(a) -> int:
    base, cfg_p = resolve(a)
    cfg, cfg_sha, pcfg = load_config(cfg_p, base)
    env = thread_environment(cfg)
    inp = load_inputs(cfg, pcfg, base)
    tfs = FR.tf_state()
    f = cfg["trial"]["frames"]
    print(f"CHECK-INPUTS: configuration {cfg_sha[:12]}, accepted runs {cfg['parent_runs']['patience_4']['run_id']} (patience 4) and "
          f"{cfg['parent_runs']['patience_12']['run_id']} (patience 12) PASS with their configurations and restricted SHA256SUMS, reference "
          f"card {cfg['baseline_check']['model']} {cfg['baseline_check']['card_sha256'][:12]}, code pinned ({len(CODE_FILES)} files), "
          f"LivenessNet architecture {cfg['architectures']['LivenessNet']['sha256'][:12]}, trial fitting {f['fitting']['n']} frames "
          f"({f['fitting']['bona_fide']}/{f['fitting']['attack']}, {f['fitting']['subjects']} subjects) and dev {f['dev']['n']} "
          f"({f['dev']['bona_fide']}/{f['dev']['attack']}, {f['dev']['subjects']} subjects) from the manifests, {len(inp.extracted_keys)} "
          f"extracted frames loaded whole and verified frame by frame, the cache hashed (its image arrays not loaded), fitting and dev keys "
          f"equal to the reference "
          f"card; fold schema not loaded, no frame assembled; TensorFlow {inp.environment['tensorflow_installed']} "
          f"installed and {tfs}; TensorFlow threads of the configuration intra-op {cfg['training']['threads']['intra_op']} and inter-op "
          f"{cfg['training']['threads']['inter_op']}, environment variables "
          + ", ".join(f"{k} {'absent' if v is None else repr(v)}" for k, v in env.items()) + " (not contradicting them); nothing written")
    print("RESULT: PASS (check-inputs)")
    return 0


def cmd_trial(a) -> int:
    if not a.i_am_authorized:
        fail("trial trains two networks on the restricted NUAA frames: pass --i-am-authorized (trial authorised by the coordinator on "
             "17 November 2026, launched after an independent review)")
    base, cfg_p = resolve(a)
    cfg, cfg_sha, pcfg = load_config(cfg_p, base)
    env = thread_environment(cfg)                      # variabili d'ambiente dei thread discordanti: rifiuto prima di ogni lavoro
    state = repo_state(ROOT)
    if state["dirty"]:
        fail("the git work tree is not clean: the trial needs a committed state (commit the changes first)")
    rid = run_id_for(a.date or time.strftime("%Y%m%d"), state)
    out_root = Path(a.out_root) if Path(a.out_root).is_absolute() else ROOT / a.out_root
    run = out_root / rid
    if os.path.lexists(run):
        fail(f"the run folder {rid} already exists: every attempt keeps its outcome (archive it or use another commit)")
    if msg := run_dir_problem(run, out_root, input_paths(cfg, pcfg, base, cfg_p)):
        fail(f"refusing the run folder: {msg}")
    run.mkdir(parents=True)
    command = "python -I -B scripts/pad_arch_comparison.py trial --i-am-authorized" + ("" if a.config == DEFAULT_CONFIG else " --config <test configuration>")
    head = {"run_id": rid, "commit": state["commit"], "dirty": False, "config_sha256": cfg_sha, "command": "trial"}
    cap = cfg["trial"]["wall_time_cap_minutes"]
    budget = SR.Budget(cap)
    SR.signal_handlers(budget)
    started, c0 = now(), cpu_seconds()
    log, saved_warning = None, warnings.showwarning
    threads = {"requested": {k: cfg["training"]["threads"][k] for k in THREAD_KEYS}, "effective": None, "set_by_this_runner": False,
               "environment": env, "logical_cpus": os.cpu_count()}
    try:
        FR.write_status(run, {**head, "status": "RUNNING", "passed": False, "started": started, "pid": os.getpid(), "threads": threads,
                              "note": "RUNNING without a live process means a failed run; SIGUSR1 to this pid stops it as INCOMPLETE"})
        log = FR.open_log(run)
        warnings.showwarning = show_warning
        print(f"== architecture trial {rid}: configuration {cfg_sha[:12]}, cap {cap} {CAP_UNIT}, pid {os.getpid()}", flush=True)
        t = time.monotonic()
        inp = load_inputs(cfg, pcfg, base)
        store = FR.FrameStore(inp, inp.X_extracted, inp.extracted_keys)
        fit, dev = trial_frames(store, inp, "fitting"), trial_frames(store, inp, "dev")
        data_check = {f"{r}_keys_equal_to_reference_card": keys_check(inp.reference_card, r, part["rows"])
                      for r, part in (("fitting", fit), ("dev", dev))}
        print(f"   inputs PASS in {time.monotonic() - t:.1f} s: fitting {len(fit['keys'])} and dev {len(dev['keys'])} frames assembled, "
              f"no test frame assembled, keys equal to the reference card; peak RSS {SR.peak_rss_mb():.0f} MB", flush=True)
        import tensorflow as tf
        threads = set_threads(tf, cfg)                 # thread fissati subito dopo l'import, prima di qualunque operazione
        print(f"   {threads_text(threads)}", flush=True)
        hide_gpu(tf)                                   # solo CPU, prima di qualunque operazione di TensorFlow
        arch = (ARCH_LOADER or FR.default_arch_loader)(rel_path(base, cfg["architectures"]["LivenessNet"]["path"]))
        nets = {n: {"epochs": [], "inference": {}} for n in AM.NETWORKS}
        status, stopped = "PASS", None
        try:
            for net in AM.NETWORKS:
                trial_network(cfg, pcfg, net, arch, fit, dev, budget, nets[net])
        except SR.CapExceeded as e:
            status = "INCOMPLETE"
            stopped = {"step": e.step, "cause": e.cause, "elapsed_minutes": round(e.minutes, 1)}
            print(f"== {'CAP' if e.cause == 'cap' else 'STOP'}: {e}; the epochs measured so far are reported", flush=True)
        assert_no_test(inp)
        checks = final_checks(cfg, cfg_p, cfg_sha, base)
        n_fit, n_dev = len(fit["keys"]), len(dev["keys"])
        for net in AM.NETWORKS:
            nets[net]["rates"] = network_rates(nets[net], n_fit, n_dev)
        bcheck = baseline_check(cfg, inp.reference_card, nets["LivenessNet"])
        wall, cpu_total, finished = budget.elapsed(), round(cpu_seconds() - c0, 1), now()   # misurati una volta, per tutti i record
        doc = {"schema": TRIAL_SCHEMA, **head, "status": status, "started": started, "finished": finished, "wall_seconds": round(wall, 1),
               "process_cpu_seconds": cpu_total, "cap_minutes": cap, "cap_unit": CAP_UNIT, "stopped": stopped,
               "unit": {**TRIAL_UNIT, "frames": cfg["trial"]["frames"]}, "epochs_fixed": TRIAL_EPOCHS, "batch": trial_training(cfg, pcfg)["batch"],
               "inference_batch": cfg["trial"]["inference_batch"], "device": "CPU (GPU hidden)",
               "visible_devices": sorted({d.device_type for d in tf.config.get_visible_devices()}),
               "threads": threads,
               "machine": (MACHINE or machine)(), "environment": inp.environment, "peak_rss_mb": round(SR.peak_rss_mb(), 1),
               "networks": nets, "data_check": data_check, "baseline_check": bcheck, "checkpoint": "none saved", "training": trial_training(cfg, pcfg),
               "process": cfg["trial"]["process"], "code_sha256": inp.code, "checks": checks, "test_read": TEST_READ,
               "declarations": cfg["declarations"], "command": command}
        texts = {TRIAL_JSON: json_text(doc, TRIAL_JSON), TRIAL_MD: trial_md(doc)}
        for n, text in texts.items():
            if probs := output_problems(text, inp):
                fail(f"{n} refused: " + "; ".join(probs))
        mw.write_all(run, texts)
        print(f"   LivenessNet history against {bcheck['model']} of {bcheck['reference_run']}: {bcheck['outcome']}", flush=True)
        print(f"run folder: {run.name}, wall {wall:.0f} s ({wall / 60:.1f} min of the {cap}-minute wall-time cap), process CPU "
              f"{cpu_total:.0f} s, peak RSS {doc['peak_rss_mb']:.0f} MB")
        print(f"RESULT: {status} (trial)")
        warnings.showwarning = saved_warning
        FR.close_log(log)
        log = None
        if probs := output_problems((run / LOG).read_text(encoding="utf-8", errors="replace"), inp):
            fail(f"{LOG} holds text that a publishable output must not hold: " + "; ".join(probs))
        FR.write_status(run, {**head, "status": status, "passed": status == "PASS", "started": started, "finished": finished,
                              "wall_seconds": round(wall, 1), "process_cpu_seconds": cpu_total,
                              "cap_minutes": cap, "cap_unit": CAP_UNIT, "stopped": stopped, "threads": threads, "baseline_check": bcheck["outcome"],
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
                                  "wall_seconds": round(budget.elapsed(), 1), "process_cpu_seconds": round(cpu_seconds() - c0, 1),
                                  "cap_minutes": cap, "cap_unit": CAP_UNIT, "threads": threads, "reason": reason, "outputs": outputs,
                                  "run_log_sha256": sha256_file(run / LOG) if (run / LOG).is_file() else "none"})
        except Exception:  # noqa: BLE001
            print("ERROR: status.json could not be written", file=sys.stderr)
        raise
    return STATUS_CODE[status]


# ============================================================================ piano

def load_trial(folder: Path, cfg: dict, cfg_sha: str) -> dict:
    """Il trial.json di una prova: status.json PASS o INCOMPLETE con lo SHA-256 di trial.json, schema, run id e unità
    della prova uguali alla configurazione; prodotto con questa configurazione (config_sha256 uguale allo SHA-256 dei byte
    letti ora) e con lo stesso codice: code_sha256 uguale alle impronte della configurazione per i file fissati (che
    cmd_plan, come trial, confronta prima con i file attuali con check_code) e allo SHA-256 attuale di questo script."""
    st = SR.read_json(read_regular(folder / STATUS, "trial status.json"), "trial status.json")
    if st.get("status") not in ("PASS", "INCOMPLETE") or not isinstance(st.get("run_id"), str) or not RUN_ID_RE.match(st["run_id"]):
        fail("trial status.json: not a PASS or INCOMPLETE trial with an ARCHTRIAL run id")
    b = read_regular(folder / TRIAL_JSON, "trial.json")
    if sha256_bytes(b) != st.get("trial_json_sha256"):
        fail("trial.json: SHA-256 differs from trial_json_sha256 of its status.json")
    doc = SR.read_json(b, "trial.json")
    if doc.get("schema") == TRIAL_SCHEMA_V1:
        fail(f"trial.json: schema {TRIAL_SCHEMA_V1} is the format before the runner set the TensorFlow threads and before the first "
             f"differing epoch was corrected; plan needs a trial of this code ({TRIAL_SCHEMA}); compare-trials reads both formats")
    if doc.get("schema") != TRIAL_SCHEMA or doc.get("run_id") != st["run_id"] or doc.get("status") != st["status"]:
        fail("trial.json: schema, run id or status differ from its status.json")
    if doc.get("unit") != {**TRIAL_UNIT, "frames": cfg["trial"]["frames"]} or sorted(doc.get("networks", {})) != sorted(AM.NETWORKS):
        fail("trial.json: the trial unit, its frames or the networks differ from the configuration")
    if doc.get("config_sha256") != cfg_sha:
        fail("trial.json: produced with another configuration (config_sha256 differs from the current configuration)")
    code = doc.get("code_sha256")
    now_code = {**{f: cfg["code"][f] for f in CODE_FILES}, "scripts/pad_arch_comparison.py": sha256_bytes(read_regular(Path(__file__), "code file"))}
    if not isinstance(code, dict) or any(code.get(f) != h for f, h in now_code.items()):
        fail("trial.json: produced with other code (code_sha256 differs from the current files)")
    check_trial_threads(doc, cfg)
    check_machine(doc.get("machine"), "trial.json")
    return doc


def check_machine(m, what: str) -> dict:
    """La macchina di un record di prova: dizionario con processor e os testi, logical_cpus intero positivo, ram_gb numero
    finito positivo (o None se il sistema non l'ha data); altrimenti Failure."""
    if not isinstance(m, dict) or any(k not in m for k in MACHINE_KEYS):
        fail(f"{what}: the machine record must hold {', '.join(MACHINE_KEYS)}")
    if not isinstance(m["processor"], str) or not isinstance(m["os"], str) or not FR.is_count(m["logical_cpus"]) or m["logical_cpus"] < 1:
        fail(f"{what}: machine processor, os or logical_cpus malformed")
    if m["ram_gb"] is not None and (isinstance(m["ram_gb"], bool) or not isinstance(m["ram_gb"], (int, float)) or not math.isfinite(m["ram_gb"])
                                    or m["ram_gb"] <= 0):
        fail(f"{what}: machine ram_gb malformed")
    return {k: m[k] for k in MACHINE_KEYS}


def check_trial_threads(doc: dict, cfg: dict) -> dict:
    """I thread di un trial.json del formato pad-arch-trial/2: record con i campi di V2_THREAD_KEYS, impostati dal
    runner, richiesti uguali a training.threads della configurazione, effettivi uguali ai richiesti; lo stesso per i
    thread effettivi registrati per ogni rete misurata."""
    th = doc.get("threads")
    want = {k: cfg["training"]["threads"][k] for k in THREAD_KEYS}
    if not isinstance(th, dict) or sorted(th) != sorted(V2_THREAD_KEYS):
        fail(f"trial.json: the threads record must hold exactly {', '.join(V2_THREAD_KEYS)}")
    if th["set_by_this_runner"] is not True or th["requested"] != want or th["effective"] != want:
        fail(f"trial.json: the TensorFlow threads were not set by the runner to training.threads {want} (requested and effective)")
    for net, rec in doc["networks"].items():
        if rec.get("epochs") and rec.get("threads") != want:
            fail(f"trial.json: the threads recorded for {net} differ from training.threads {want}")
    return th


def hours(seconds):
    return None if seconds is None else seconds / 3600


def training_name(network: str, fold: int, seed: int) -> str:
    return f"{network}_fold{fold}_seed{seed}"


def plan_order(splits, seeds) -> list:
    """L'ordine fissato degli addestramenti di un run NUAA del piano (scelta dell'autore, motivata in plan.order): per
    fold e seed, nell'ordine del run accettato (nuaa_session_ablation_run.training_order: fold, poi seed), LivenessNet
    poi ResNet-18. Così ogni LivenessNet occupa nel suo processo la posizione che il modello del braccio B con lo stesso
    fold e seed occupava nel run accettato (dove a ogni B seguiva A*), preceduto dallo stesso numero di addestramenti;
    il primo addestramento del processo è LivenessNet fold 1 seed 42, come nella prova."""
    return [(net, f, s) for f in splits for s in seeds for net in AM.NETWORKS]


def nuaa_runs(nuaa: dict) -> list:
    """I run NUAA del piano: uno per patience (4, poi 12), un processo ciascuno, con i 24 addestramenti nell'ordine di
    plan_order (la forma di plan.runs nella configurazione)."""
    return [{"name": f"nuaa_patience_{p}", "dataset": nuaa["name"], "patience": p,
             "trainings": [training_name(*x) for x in plan_order(nuaa["splits"], nuaa["seeds"])]} for p in D.PATIENCES]


def order_rows(cfg: dict) -> list:
    """Le righe di run_order.csv: per run e posizione (1-based) la rete, il fold e il seed; per LivenessNet il run
    accettato alla stessa patience, il modello del braccio B con lo stesso fold e seed e la sua posizione (1-based) nel
    processo del run accettato (nuaa_session_ablation_run.training_order), che coincide con la posizione nel piano."""
    nuaa = cfg["plan"]["datasets"][0]
    ref = {(e["network"], e["patience"]): e["reference_run"] for e in nuaa["trainings"]}
    accepted = SR.training_order(len(nuaa["splits"]), nuaa["seeds"])
    rows = []
    for run in nuaa_runs(nuaa):
        for i, (net, f, s) in enumerate(plan_order(nuaa["splits"], nuaa["seeds"]), start=1):
            liv = ref[(net, run["patience"])] is not None
            rows.append({"run": run["name"], "position": i, "network": net, "fold": f, "seed": s, "patience": run["patience"],
                         "reference_run": ref[(net, run["patience"])], "reference_model": SR.model_name("arm_b", f, s) if liv else None,
                         "reference_position": accepted.index(("arm_b", f, s)) + 1 if liv else None})
    return rows


def run_caps(cfg: dict, rows: list) -> list:
    """Tetto e stima di ciascun run NUAA del piano (regola di plan.caps): ore reali minime e massime = somma delle ore
    reali (addestramento più valutazione) delle righe del piano della stessa patience, che devono contare insieme gli
    addestramenti del run; tetto in ore di tempo reale = massimo × wall_multiplier arrotondato per eccesso al multiplo di
    round_up_hours; a parte, la stima in ore di CPU (minimo e massimo, somma delle righe: ore reali × rapporto CPU / reale
    misurato nella prova). Senza i numeri della prova (o con una riga incompleta) le celle restano vuote."""
    ca = cfg["plan"]["caps"]
    nuaa = cfg["plan"]["datasets"][0]
    out = []
    for run in nuaa_runs(nuaa):
        rs = [r for r in rows if r["dataset"] == nuaa["name"] and r["patience"] == run["patience"]]
        if sum(r["new_trainings"] for r in rs) != len(run["trainings"]):
            fail(f"plan: the rows of patience {run['patience']} do not count the {len(run['trainings'])} trainings of {run['name']}")
        tot = {k: (None if any(r[k] is None for r in rs) else sum(r[k] for r in rs))
               for k in ("wall_hours_min", "wall_hours_max", "cpu_hours_min", "cpu_hours_max")}
        cap = None
        if tot["wall_hours_max"] is not None:
            cap = math.ceil(round(tot["wall_hours_max"] * ca["wall_multiplier"] / ca["round_up_hours"], 9)) * ca["round_up_hours"]
        out.append({"run": run["name"], "dataset": run["dataset"], "patience": run["patience"], "trainings": len(run["trainings"]), **tot,
                    "cap_wall_hours": cap})
    return out


def plan_rows(cfg: dict, trial) -> list:
    """Righe del piano per dataset × rete × patience. Epoche: minimo min(patience + 1, max_epochs) (la prima epoca
    migliore e patience epoche senza miglioramento), massimo max_epochs. Tempo reale di un addestramento = epoche ×
    (velocità di addestramento × frame del fitting + velocità di validazione × frame del dev) + costo fuori dalle epoche
    misurato nella prova; valutazione di ogni addestramento = velocità d'inferenza × ((strati BN + 1) × fitting [al più:
    la ristima della PR #27 fa per ogni strato una passata troncata all'ingresso dello strato, contata qui come intera, più
    la passata di controllo] + 2 × dev + 2 × test dello split [score nei due stati della BN; le due soglie non aggiungono
    passate]). Fuori dalla stima: salvataggio dei checkpoint, assemblaggio del test, caricamento degli ingressi (una volta
    per processo). Ore di CPU = ore reali × rapporto CPU / reale misurato nella prova. Un valore mancante lascia la cella
    senza numero."""
    pl = cfg["plan"]
    rates = {}
    if trial is not None:
        n_fit, n_dev = trial["unit"]["frames"]["fitting"]["n"], trial["unit"]["frames"]["dev"]["n"]
        rates = {n: network_rates(trial["networks"][n], n_fit, n_dev) for n in AM.NETWORKS}
    rows = []
    for ds in pl["datasets"]:
        fp = ds["frames_per_unit"]
        units = len(ds["splits"]) * len(ds["seeds"]) if ds["splits"] is not None and ds["seeds"] is not None else None
        for e in ds["trainings"]:
            net, p = e["network"], e["patience"]
            r = rates.get(net, {})
            tr, va, inf, ov, ratio = (r.get("train_seconds_per_1000_fitting_frames_per_epoch"), r.get("validation_seconds_per_1000_frames"),
                                      r.get("inference_seconds_per_1000_frames"), r.get("overhead_seconds_per_training"), r.get("cpu_over_wall"))
            ep_min, ep_max = min(p + 1, pl["max_epochs"]), pl["max_epochs"]
            per_epoch = None if None in (tr, va, fp["fitting"], fp["dev"]) else (tr * fp["fitting"] + va * fp["dev"]) / 1000
            t_min = None if None in (per_epoch, ov, e["new"]) else e["new"] * (ep_min * per_epoch + ov)
            t_max = None if None in (per_epoch, ov, e["new"]) else e["new"] * (ep_max * per_epoch + ov)
            nbn = cfg["architectures"][net]["batchnorm_layers"]
            ev = None
            if None not in (inf, fp["fitting"], fp["dev"], fp["test_per_split"], ds["seeds"]):
                ev = len(ds["seeds"]) * sum(inf * ((nbn + 1) * fp["fitting"] + 2 * fp["dev"] + 2 * t) / 1000 for t in fp["test_per_split"])
            w_min = None if None in (t_min, ev) else t_min + ev
            w_max = None if None in (t_max, ev) else t_max + ev
            rows.append({"dataset": ds["name"], "access": ds["access"], "network": net, "patience": p, "units": units,
                         "existing_trainings": e["existing"], "new_trainings": e["new"], "reference_run": e["reference_run"],
                         "epochs_min": ep_min, "epochs_max": ep_max, "fitting_frames_per_unit": fp["fitting"], "dev_frames_per_unit": fp["dev"],
                         "test_frames_per_split": fp["test_per_split"], "evaluation_cells": len(D.BN_STATES) * len(D.THRESHOLD_RULES),
                         "training_wall_hours_min": hours(t_min), "training_wall_hours_max": hours(t_max), "evaluation_wall_hours": hours(ev),
                         "wall_hours_min": hours(w_min), "wall_hours_max": hours(w_max),
                         "cpu_hours_min": None if None in (w_min, ratio) else hours(w_min * ratio),
                         "cpu_hours_max": None if None in (w_max, ratio) else hours(w_max * ratio)})
    return rows


CONFIG_VALUES = ("units", "new_trainings", "fitting_frames_per_unit", "dev_frames_per_unit", "test_frames_per_split")
HOUR_KEYS = ("training_wall_hours_min", "training_wall_hours_max", "evaluation_wall_hours", "wall_hours_min", "wall_hours_max",
             "cpu_hours_min", "cpu_hours_max")


def cell(row: dict, k: str) -> str:
    """Testo di una cella: un valore della configurazione non ancora fissato è TBF; un'ora senza misura resta vuota."""
    v = row[k]
    if v is None:
        if k in CONFIG_VALUES:
            return TBF
        if k == "reference_run":
            return "none"
        return ""
    if k in HOUR_KEYS:
        return f"{v:.2f}"
    if isinstance(v, list):
        return " ".join(str(x) for x in v)
    return str(v)


def totals(rows: list) -> dict:
    """Somme delle ore sulle righe che hanno tutti i numeri, con il numero di righe incluse."""
    full = [r for r in rows if all(r[k] is not None for k in HOUR_KEYS)]
    return {"rows": len(full), "of": len(rows), **{k: sum(r[k] for r in full) if full else None for k in HOUR_KEYS}}


def plan_md(cfg: dict, cfg_sha: str, rows: list, trial, trial_sha, caps: list) -> str:
    pl = cfg["plan"]
    L = ["# Planned runs: LivenessNet and ResNet-18", "",
         f"Generated from the configuration {cfg_sha[:12]}" + (f" and the trial {trial['run_id']} (trial.json {trial_sha[:12]}, status "
                                                               f"{trial['status']})." if trial else " without a trial: no time estimate."), "",
         f"Both networks trained from scratch with the same recipe, subject splits, seeds and preprocessing; patience {' and '.join(map(str, pl['patiences']))}, "
         f"at most {pl['max_epochs']} epochs; evaluation cells per training: BN {' / '.join(pl['bn_states'])} x threshold {' / '.join(pl['thresholds'])}. "
         "Every training of the plan is new (existing = 0, a choice of the author); for LivenessNet the accepted runs are the reference "
         "runs of the history reproduction check, not checkpoints to reuse.", "",
         "Hours are hours of wall time on the machine of the trial; CPU hours use the CPU / wall ratio measured in the trial. "
         "The rates are means over the measured epochs, the first included (graph tracing: a cautious estimate); each training adds "
         "its measured cost outside the epochs; the BN re-estimation is counted as (BN layers + 1) full passes over the fitting set, an "
         "upper bound (at most: its passes stop at the input of each layer); outside the estimate: saving the checkpoints, assembling "
         "the test, loading the inputs (once per process). "
         f"«{TBF}»: a value of the configuration not fixed yet; an empty cell: no trial measurement.", "",
         "| " + " | ".join(PLAN_COLUMNS) + " |", "|" + "---|" * len(PLAN_COLUMNS)]
    for r in rows:
        L.append("| " + " | ".join(cell(r, k) for k in PLAN_COLUMNS) + " |")
    L += ["", "## Totals (hours of wall time on the machine of the trial; CPU hours apart)", "",
          "| scope | rows with every number | " + " | ".join(HOUR_KEYS) + " |",
          "|" + "---|" * (len(HOUR_KEYS) + 2)]
    for name in [*DATASETS, "all"]:
        t = totals([r for r in rows if name == "all" or r["dataset"] == name])
        L.append(f"| {name} | {t['rows']} of {t['of']} | " + " | ".join("" if t[k] is None else f"{t[k]:.2f}" for k in HOUR_KEYS) + " |")
    L += ["", "## Datasets", ""]
    for ds in pl["datasets"]:
        L.append(f"- {ds['name']} ({ds['access']}): official source: {ds['official_source'] or TBF}; identity split: {ds['identity_split'] or TBF}; "
                 f"missing access: {ds['missing_access'] or 'none'}.")
    L += ["", "## Costs", ""] + [f"- {k}: {v}" for k, v in pl["costs"].items()]
    th = cfg["training"]["threads"]
    L += ["", "## NUAA runs: order, caps, machine and threads", "",
          f"Order of the trainings (`run_order.csv`): {pl['order']}.", "",
          f"Caps (`run_caps.csv`): {pl['caps']['rule']}; unit of the cap: {pl['caps']['unit']}; multiplier {pl['caps']['wall_multiplier']}, "
          f"rounded up to a multiple of {pl['caps']['round_up_hours']} hours.", "",
          "| " + " | ".join(CAPS_COLUMNS) + " |", "|" + "---|" * len(CAPS_COLUMNS)]
    for c in caps:
        L.append("| " + " | ".join(caps_cell(c, k) for k in CAPS_COLUMNS) + " |")
    if trial is not None:
        m = trial["machine"]
        L += ["", f"Machine of the trial {trial['run_id']}: {m['processor']}, {m['logical_cpus']} logical CPUs, {ram_text(m)}, {m['os']}."]
    else:
        L += ["", "Machine: no trial given."]
    L += ["", f"TensorFlow threads (training.threads): intra-op {th['intra_op']}, inter-op {th['inter_op']}, set by the runner; rule: {th['rule']}.", "",
          f"Machine rule: {pl['machine_rule']}.", ""]
    return "\n".join(L)


def caps_cell(c: dict, k: str) -> str:
    """Testo di una cella di run_caps.csv: ore con due decimali (il tetto con due decimali del suo multiplo), vuota senza misura."""
    v = c[k]
    if v is None:
        return ""
    if k in ("wall_hours_min", "wall_hours_max", "cap_wall_hours", "cpu_hours_min", "cpu_hours_max"):
        return f"{v:.2f}"
    return str(v)


def order_cell(r: dict, k: str) -> str:
    v = r[k]
    return "none" if v is None else str(v)


def cmd_plan(a) -> int:
    base, cfg_p = resolve(a)
    cfg, cfg_sha, _ = load_config(cfg_p, base)
    check_code(cfg)                                    # i file fissati di ora contro la configurazione, come in trial
    trial = trial_sha = None
    inputs = {"the configuration": cfg_p}
    if a.trial_run:
        folder = Path(a.trial_run) if Path(a.trial_run).is_absolute() else ROOT / a.trial_run
        trial = load_trial(folder, cfg, cfg_sha)
        trial_sha = sha256_bytes(read_regular(folder / TRIAL_JSON, "trial.json"))
        inputs["the trial run"] = folder
    out = Path(a.out_dir) if Path(a.out_dir).is_absolute() else ROOT / a.out_dir
    existing = [n for n in PLAN_OUTPUTS if os.path.lexists(out / n)]
    if existing:
        fail(f"{', '.join(existing)} already exist in {out.name}: the plan is written only to new files")
    if msg := mw.output_problem(out, ROOT, pg.BASE_PROTECTED, [ROOT / DEFAULT_PLAN_DIR], [ROOT / x for x in EXTRA_PROTECTED], inputs, list(PLAN_OUTPUTS)):
        fail(f"refusing the plan folder: {msg}")
    rows = plan_rows(cfg, trial)
    caps = run_caps(cfg, rows)
    texts = {"run_plan.csv": csv_text(PLAN_COLUMNS, [[cell(r, k) for k in PLAN_COLUMNS] for r in rows]),
             "run_plan.md": plan_md(cfg, cfg_sha, rows, trial, trial_sha, caps),
             "run_order.csv": csv_text(ORDER_COLUMNS, [[order_cell(r, k) for k in ORDER_COLUMNS] for r in order_rows(cfg)]),
             "run_caps.csv": csv_text(CAPS_COLUMNS, [[caps_cell(c, k) for k in CAPS_COLUMNS] for c in caps])}
    empty = FR.Inputs()
    for n, text in texts.items():
        if probs := output_problems(text, empty):
            fail(f"{n} refused: " + "; ".join(probs))
    mw.write_all(out, texts, newline="")
    t = totals(rows)
    print(f"plan: {len(rows)} rows, {t['rows']} with every number; files {', '.join(PLAN_OUTPUTS)} in {out.name}")
    print("RESULT: PASS (plan)")
    return 0


# ============================================================================ confronto fra due prove

def _number_value(v) -> bool:
    """Un numero finito (non booleano, non NaN, non infinito)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def read_trial_record(folder: Path, label: str) -> tuple:
    """Una prova letta da compare-trials, nel formato pad-arch-trial/1 (PR #29) o pad-arch-trial/2: status.json PASS o
    INCOMPLETE con un run id ARCHTRIAL (RUN_ID_RE) e lo SHA-256 di trial.json; trial.json con quello SHA-256, schema fra
    i due, run id e stato uguali a status.json, unità della prova (NUAA, braccio B, fold 1, seed 42); in entrambi i
    formati commit di 7-40 cifre esadecimali e config_sha256 di 64 cifre esadecimali (ben formati; config_sha256 non è
    confrontato con la configurazione attuale, perché le due prove possono averne di diverse); macchina
    (check_machine); il record dei thread del suo formato (check_thread_record: campi esatti e valori); per ciascuna
    delle due reti le epoche numerate da 1 (interi, non booleani) con le cinque grandezze reali e finite (un NaN o un
    infinito -> Failure). Restituisce (trial.json, SHA-256 dei due file letti). Ogni difetto dei due record -> Failure
    (nessuna uscita scritta)."""
    sb = read_regular(folder / STATUS, f"{label} status.json")
    st = SR.read_json(sb, f"{label} status.json")
    if st.get("status") not in ("PASS", "INCOMPLETE") or not isinstance(st.get("run_id"), str) or not RUN_ID_RE.match(st["run_id"]):
        fail(f"{label} status.json: not a PASS or INCOMPLETE trial with an ARCHTRIAL run id")
    tb = read_regular(folder / TRIAL_JSON, f"{label} trial.json")
    if sha256_bytes(tb) != st.get("trial_json_sha256"):
        fail(f"{label} trial.json: SHA-256 differs from trial_json_sha256 of its status.json")
    doc = SR.read_json(tb, f"{label} trial.json")
    if doc.get("schema") not in (TRIAL_SCHEMA_V1, TRIAL_SCHEMA):
        fail(f"{label} trial.json: schema must be {TRIAL_SCHEMA_V1} or {TRIAL_SCHEMA}")
    if doc.get("run_id") != st["run_id"] or doc.get("status") != st["status"]:
        fail(f"{label} trial.json: run id or status differ from its status.json")
    u = doc.get("unit")
    if not isinstance(u, dict) or any(u.get(k) != v for k, v in TRIAL_UNIT.items()):
        fail(f"{label} trial.json: not a trial of the unit {TRIAL_UNIT}")
    if not isinstance(doc.get("commit"), str) or not re.fullmatch(r"[0-9a-f]{7,40}", doc["commit"]) or not SR._hex(doc.get("config_sha256")):
        fail(f"{label} trial.json: commit or config_sha256 malformed")
    check_machine(doc.get("machine"), f"{label} trial.json")
    check_thread_record(doc.get("threads"), doc["schema"], f"{label} trial.json")
    nets = doc.get("networks")
    if not isinstance(nets, dict) or sorted(nets) != sorted(AM.NETWORKS):
        fail(f"{label} trial.json: the networks must be exactly {', '.join(AM.NETWORKS)}")
    for net in AM.NETWORKS:
        ep = nets[net].get("epochs") if isinstance(nets[net], dict) else None
        if not isinstance(ep, list) or any(not isinstance(e, dict) or e.get("epoch") != i + 1 or isinstance(e.get("epoch"), bool)
                                           or any(not _number_value(e.get(k)) for k in HISTORY_KEYS) for i, e in enumerate(ep)):
            fail(f"{label} trial.json: the epochs of {net} must be numbered from 1 with finite numeric {', '.join(HISTORY_KEYS)}")
    return doc, {STATUS: sha256_bytes(sb), TRIAL_JSON: sha256_bytes(tb)}


def check_thread_record(th, schema: str, what: str) -> dict:
    """Il record dei thread di una prova, nel suo formato, con i campi esatti e i valori validati:
      - pad-arch-trial/1: inter_op_parallelism_threads e intra_op_parallelism_threads interi >= 0 (0 = non impostati con
        tf.config.threading), logical_cpus intero >= 1, set_by_this_runner esattamente False;
      - pad-arch-trial/2: requested ed effective con esattamente intra_op e inter_op interi >= 0 (requested >= 1),
        set_by_this_runner booleano, environment con esattamente le due variabili (testo o None), logical_cpus intero >= 1.
    Altrimenti Failure."""
    keys = V1_THREAD_KEYS if schema == TRIAL_SCHEMA_V1 else V2_THREAD_KEYS
    if not isinstance(th, dict) or sorted(th) != sorted(keys):
        fail(f"{what}: the threads record of {schema} must hold exactly {', '.join(keys)}")
    ok = FR.is_count(th["logical_cpus"]) and th["logical_cpus"] >= 1
    if schema == TRIAL_SCHEMA_V1:
        ok = ok and th["set_by_this_runner"] is False and all(FR.is_count(th[k]) for k in ("inter_op_parallelism_threads", "intra_op_parallelism_threads"))
    else:
        pairs = [th["requested"], th["effective"]]
        ok = ok and isinstance(th["set_by_this_runner"], bool) and all(isinstance(x, dict) and sorted(x) == sorted(THREAD_KEYS)
                                                                        and all(FR.is_count(x[k]) for k in THREAD_KEYS) for x in pairs) \
            and all(th["requested"][k] >= 1 for k in THREAD_KEYS) and isinstance(th["environment"], dict) \
            and sorted(th["environment"]) == sorted(TF_THREAD_ENV.values()) and all(v is None or isinstance(v, str) for v in th["environment"].values())
    if not ok:
        fail(f"{what}: the threads record of {schema} holds malformed values")
    return th


def compare_histories(new: dict, ref: dict) -> dict:
    """Le storie delle due reti di due prove, per rete e per campo (HISTORY_KEYS) con history_differences (la regola della
    prima epoca diversa, la stessa del controllo della prova): epoche confrontate, prima epoca diversa, scarto assoluto
    massimo e sua epoca. Una rete è riprodotta solo se le due prove hanno lo stesso numero, non nullo, di epoche e
    scarto 0 in ogni campo; esito «reproduced» solo se lo sono tutte le reti, altrimenti «not reproduced»."""
    out = {}
    for net in AM.NETWORKS:
        a, b = new["networks"][net]["epochs"], ref["networks"][net]["epochs"]
        fields = {k: history_differences([e[k] for e in b], [e[k] for e in a]) for k in HISTORY_KEYS}
        same = len(a) == len(b)
        out[net] = {"epochs_new": len(a), "epochs_reference": len(b), "same_number_of_epochs": same, "fields": fields,
                    "reproduced": same and len(a) > 0 and all(f["first_differing_epoch"] is None for f in fields.values())}
    return {"networks": out, "outcome": "reproduced" if all(v["reproduced"] for v in out.values()) else "not reproduced"}


CONDITIONS_NOTE = ("declared by a conditions file written by the author around the reference trial, not by its record (the format "
                   "pad-arch-trial/1 does not record the environment variables); checked by the code: the file holds the commit of the "
                   "reference record as a whole word, and its one line with both variables sets TF_NUM_INTRAOP_THREADS and "
                   "TF_NUM_INTEROP_THREADS to integers equal to the threads required of the configuration")


def reference_conditions(path, commit: str) -> dict:
    """Come erano stati imposti i thread nella prova di riferimento, quando il suo record non lo dice (formato /1): da un
    file di testo delle condizioni della prova (comando e condizioni), scritto dall'autore attorno alla prova e dato con
    --reference-conditions. Controlli: il file contiene il commit del record di riferimento come parola intera (non
    preceduto né seguito da una cifra esadecimale o da una lettera); ha una e una sola riga con sia
    TF_NUM_INTRAOP_THREADS sia TF_NUM_INTEROP_THREADS (la riga del comando); in quella riga ciascuna variabile ha un
    solo valore, un intero uguale a REQUIRED_THREADS (10: il valore che load_config richiede a training.threads per
    intra_op e inter_op). Ogni altro caso -> Failure, nessuna uscita. Si riportano solo il nome del file, il suo
    SHA-256, la riga (spazi ai lati tolti), i valori letti e la nota CONDITIONS_NOTE, che dice da dove viene e che cosa
    è stato controllato; il resto del file non è riportato e la riga passa dalla scansione dei testi personali. Senza il
    file: «not recorded»."""
    if path is None:
        return {"source": "not recorded"}
    q = Path(path) if Path(path).is_absolute() else ROOT / path
    b = read_regular(q, "reference conditions file")
    try:
        text = b.decode("utf-8")
    except UnicodeDecodeError:
        fail(f"reference conditions file {q.name}: not UTF-8 text")
    if not re.search(rf"(?<![0-9A-Za-z]){re.escape(commit)}(?![0-9A-Za-z])", text):
        fail(f"reference conditions file {q.name}: it does not hold the commit {commit} of the reference record (another trial?)")
    lines = [l.strip() for l in text.splitlines() if all(v in l for v in TF_THREAD_ENV.values())]
    if len(lines) != 1:
        fail(f"reference conditions file {q.name}: {len(lines)} lines hold both {' and '.join(TF_THREAD_ENV.values())} (exactly one, the "
             "command, is required)")
    values = {}
    for k, var in TF_THREAD_ENV.items():
        found = re.findall(rf"(?<![A-Za-z0-9_]){var}=(\S*)", lines[0])
        if len(found) != 1 or not re.fullmatch(r"[0-9]+", found[0]) or int(found[0]) != REQUIRED_THREADS:
            fail(f"reference conditions file {q.name}: {var} must be set once, to the integer {REQUIRED_THREADS} (training.threads)")
        values[var] = int(found[0])
    return {"source": "conditions file", "file": q.name, "sha256": sha256_bytes(b), "command_line": lines[0], "values": values,
            "note": CONDITIONS_NOTE}


def trial_summary(doc: dict, files: dict) -> dict:
    """Ciò che il confronto riporta di una prova: run id, formato, stato, commit, SHA-256 della configurazione, macchina,
    i thread come registrati dal suo formato (nulla aggiunto) e gli SHA-256 dei due file letti."""
    return {"run_id": doc["run_id"], "schema": doc["schema"], "status": doc["status"], "commit": doc["commit"],
            "config_sha256": doc["config_sha256"], "machine": {k: doc["machine"][k] for k in MACHINE_KEYS},
            "threads": {"recorded": doc["threads"], "note": V1_THREADS_NOTE if doc["schema"] == TRIAL_SCHEMA_V1 else V2_THREADS_NOTE},
            "files_sha256": files}


def comparison_md(doc: dict) -> str:
    L = [f"# Comparison of two technical trials: {doc['new']['run_id']} against {doc['reference']['run_id']}", "",
         f"Outcome: **{doc['outcome']}** (reproduced only if both networks have the same, non-zero number of epochs and difference 0 in "
         f"{', '.join(HISTORY_KEYS)}; no tolerance).", "",
         "| trial | role | format | status | commit | configuration SHA-256 | machine | threads as recorded |", "|---|---|---|---|---|---|---|---|"]
    for role in ("new", "reference"):
        t = doc[role]
        m = t["machine"]
        L.append(f"| {t['run_id']} | {role} | {t['schema']} | {t['status']} | {t['commit']} | {t['config_sha256']} | {m['processor']}, "
                 f"{m['logical_cpus']} logical CPUs, {ram_text(m)}, {m['os']} | {json.dumps(t['threads']['recorded'], sort_keys=True)} |")
    c = doc["reference"]["threads_as_imposed"]
    imposed = ("not recorded" if c["source"] == "not recorded" else
               f"command line `{c['command_line']}` in the conditions file {c['file']} (SHA-256 {c['sha256']}); {c['note']}")
    L += ["", "Notes on the threads: " + "; ".join(f"{doc[r]['run_id']}: {doc[r]['threads']['note']}" for r in ("new", "reference")) + ".", "",
          f"How the threads were imposed in the reference trial {doc['reference']['run_id']}: {imposed}.", "",
          "| network | field | epochs (new / reference) | epochs compared | first differing epoch | largest absolute difference | "
          "epoch of the largest difference |", "|---|---|---|---|---|---|---|"]
    for net, v in doc["networks"].items():
        for k, f in v["fields"].items():
            L.append(f"| {net} | {k} | {v['epochs_new']} / {v['epochs_reference']} | {f['epochs_compared']} | {epoch_text(f['first_differing_epoch'])} | "
                     f"{f['max_abs_difference']:.3g} | {epoch_text(f['epoch_of_max_abs_difference'])} |")
    L += ["", "Files read (SHA-256): " + "; ".join(f"{doc[r]['run_id']} {n} {h}" for r in ("new", "reference")
                                                    for n, h in doc[r]["files_sha256"].items()) + ".", ""]
    return "\n".join(L)


def cmd_compare_trials(a) -> int:
    """compare-trials: confronto di sola lettura fra due prove (--trial-run, la nuova; --reference-run, quella di
    riferimento; facoltativo --reference-conditions, il file delle condizioni della prova di riferimento: reference_conditions);
    scrive trial_comparison.json e trial_comparison.md in --compare-out-dir (cartella nuova o senza quei
    file; dentro il repository solo sotto delivery/archcmp/trial-comparison; mai dentro una delle due cartelle lette né
    dentro le cartelle protette), solo dopo tutti i controlli e la scansione delle uscite pubbliche: su ogni errore
    nessuna uscita."""
    if not a.trial_run or not a.reference_run:
        fail("compare-trials needs --trial-run (the new trial) and --reference-run (the reference trial)")
    folders = {lab: (Path(p) if Path(p).is_absolute() else ROOT / p) for lab, p in (("the new trial", a.trial_run), ("the reference trial", a.reference_run))}
    if pg.within(folders["the new trial"], folders["the reference trial"]) or pg.within(folders["the reference trial"], folders["the new trial"]):
        fail("compare-trials: the two trial folders must be different and not nested")
    out = Path(a.compare_out_dir) if Path(a.compare_out_dir).is_absolute() else ROOT / a.compare_out_dir
    existing = [n for n in COMPARE_OUTPUTS if os.path.lexists(out / n)]
    if existing:
        fail(f"{', '.join(existing)} already exist in {out.name}: the comparison is written only to new files")
    inputs = dict(folders)
    if a.reference_conditions is not None:
        inputs["the reference conditions file"] = Path(a.reference_conditions) if Path(a.reference_conditions).is_absolute() \
            else ROOT / a.reference_conditions
    if msg := mw.output_problem(out, ROOT, pg.BASE_PROTECTED, [ROOT / DEFAULT_COMPARE_DIR], [ROOT / x for x in EXTRA_PROTECTED], inputs,
                                list(COMPARE_OUTPUTS)):
        fail(f"refusing the comparison folder: {msg}")
    new, new_files = read_trial_record(folders["the new trial"], "new trial")
    ref, ref_files = read_trial_record(folders["the reference trial"], "reference trial")
    conditions = reference_conditions(a.reference_conditions, ref["commit"])
    cmp = compare_histories(new, ref)
    doc = {"schema": COMPARE_SCHEMA, "outcome": cmp["outcome"], "rule":
           "per network and field (loss, accuracy, val_loss, val_accuracy, lr), over the epochs present in both trials: epochs compared, "
           "first differing epoch, largest absolute difference and the first epoch at which it is reached; reproduced only if both "
           "networks have the same, non-zero number of epochs and difference 0 in every field (no tolerance)",
           "new": trial_summary(new, new_files), "reference": {**trial_summary(ref, ref_files), "threads_as_imposed": conditions},
           "networks": cmp["networks"]}
    texts = {"trial_comparison.json": json_text(doc, "trial_comparison.json"), "trial_comparison.md": comparison_md(doc)}
    for n, text in texts.items():
        if probs := output_problems(text, FR.Inputs()):
            fail(f"{n} refused: " + "; ".join(probs))
    mw.write_all(out, texts)
    print(f"compare-trials: {new['run_id']} against {ref['run_id']}: {cmp['outcome']}; files {', '.join(COMPARE_OUTPUTS)} in {out.name}")
    print("RESULT: PASS (compare-trials)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LivenessNet and ResNet-18: inputs check, technical trial (NUAA arm B fold 1 seed 42, fitting "
                                             "and dev only, CPU), the table of the planned runs and the comparison of two trials")
    ap.add_argument("cmd", choices=["check-inputs", "trial", "plan", "compare-trials"])
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="configuration JSON (relative to the repository root)")
    ap.add_argument("--input-root", default=None, help="base folder of the input paths of the configuration (default: repository root)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="trial: folder that receives <run-id>/ (default results/archcmp; inside the repository only results/archcmp, "
                         "outside it any folder that is not an input; never a symbolic link)")
    ap.add_argument("--date", default=None, help="trial: YYYYMMDD of the run id (default: today)")
    ap.add_argument("--i-am-authorized", action="store_true", help="required by trial: training on the restricted frames")
    ap.add_argument("--trial-run", default=None, help="plan: the run folder of a trial whose trial.json gives the times (same configuration and code as now); "
                                                      "compare-trials: the new trial")
    ap.add_argument("--out-dir", default=DEFAULT_PLAN_DIR, help="plan: folder of run_plan.csv, run_plan.md, run_order.csv and run_caps.csv (never overwritten)")
    ap.add_argument("--reference-run", default=None, help="compare-trials: the run folder of the reference trial")
    ap.add_argument("--reference-conditions", default=None, help="compare-trials: text file with the command and conditions of the reference "
                                                                 "trial; its name, SHA-256 and the one line holding both TF_NUM_ variables are reported")
    ap.add_argument("--compare-out-dir", default=DEFAULT_COMPARE_DIR, help="compare-trials: folder of trial_comparison.json and trial_comparison.md "
                                                                          "(never overwritten; inside the repository only delivery/archcmp/trial-comparison)")
    a = ap.parse_args(argv)
    try:
        return {"check-inputs": cmd_check_inputs, "trial": cmd_trial, "plan": cmd_plan, "compare-trials": cmd_compare_trials}[a.cmd](a)
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
