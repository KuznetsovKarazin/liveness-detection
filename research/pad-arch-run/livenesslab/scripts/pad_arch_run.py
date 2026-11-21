"""
I due run NUAA del confronto LivenessNet / ResNet-18 (48 addestramenti, preregistrazione definitiva della PR #29,
delivery/pad-arch-comparison/PROTOCOL.md §3, §4, §7, §8, §11): il comando promesso nella preregistrazione («it will
be published, with its configuration, in a new pull request before the runs, and it will apply and record the machine
and thread rule (§7)»). No test frame is assembled or evaluated, no test score or metric is read: i file dei frame (la
cache, sottoposta a hash per intero, e il file dei frame estratti, caricato per intero con lo SHA-256 di ogni frame;
nel run anche le immagini della cache, caricate per intero nel FrameStore) contengono anche frame dei soggetti tenuti
fuori, ma se ne assemblano solo i frame del fitting e del dev del braccio B. TEST_READ dice esattamente che cosa legge
ciascun comando. Il test si leggerà dopo, con un comando di valutazione pubblicato prima; le uscite riservate di questo
comando gli bastano: checkpoint, statistiche BN ristimate, punteggi del dev nei due stati della BN con l'ID di ogni frame
e le soglie EER.

Tutto ciò che la preregistrazione definisce è importato dai moduli congelati, fissati per SHA-256 nella configurazione
(una sola definizione per regola): pad_arch_comparison (PAC: configurazione della preregistrazione, ingressi, thread,
macchina, trainer della prova trial_trainer, EpochTimer, history_differences, ordine del piano, tetti, scansione delle
uscite), pad_arch_definitions (D: one_class_run, condition_threshold, attack_decisions, frame_ids, ids_sha256,
prepare_units, same_frames, one_class_counts), pad_arch_models (AM: registro e controllo dei modelli),
nuaa_bn_recalibration (BR: lo stimatore della PR #27, recalibrate, il valutatore evaluate e i controlli), nuaa_bn_test
(BT: il controllo «stessa ristima della PR #27» della PR #28, recal_check, e il caricamento del run BNRECAL),
nuaa_session_ablation_run (SR: Budget, segnali, logit del dev), nuaa_folds_run (FR: FrameStore/assemble, scorer, soglia
EER, controlli della scheda), m2_writers e path_guard.

Comandi:
  check-inputs     sola lettura, TensorFlow non importato, nulla scritto: configurazione a insieme chiuso; la
                   configurazione della preregistrazione (SHA-256 fissato, validata da PAC.load_config); codice fissato;
                   il record della prova ufficiale (status.json e trial.json con lo SHA-256 fissato, uguale alle righe
                   trial/status.json e trial/trial.json del SHA256SUMS del pacchetto consegnato, a sua volta fissato;
                   validato da PAC.load_trial); run_order.csv e run_caps.csv fissati e uguali, byte per byte, ai file
                   rigenerati dalle funzioni della preregistrazione; le configurazioni della PR #27 (validata da
                   BR.load_config) e della PR #28 (per SHA-256); regola della macchina; variabili d'ambiente dei thread
                   (thread_environment_rule: nessuna discordante e tutte uguali al record della prova, come in
                   check-estimator e run); PAC.load_inputs; le 24 schede di riferimento e, per ogni unità, chiavi del
                   fitting e del dev uguali a quelle delle schede; con --estimator-check, il record del controllo. Non
                   controlla l'albero git (lo fanno check-estimator e run).
  check-estimator  il controllo che il protocollo impone prima dell'uso della ristima sui modelli nuovi (§3): su una
                   copia in memoria di ciascuno dei 36 checkpoint accettati di LivenessNet a patience 4 (run genitore della
                   PR #27), lo stimatore di questo comando (reestimate) e i controlli della PR #27: copia uguale prima della
                   ristima, solo i tensori mobili cambiati (BR.tensor_problems), passaggio di coerenza (BR.consistency) e
                   conteggi per canale, softmax dei logit sul fitting e sul dev; poi statistiche installate e probabilità
                   del dev dopo contro il run 20261114-BNRECAL-a10a082 (BT.recal_check, tolleranze della PR #28). Esiti:
                   PASS (0), MISMATCH (4: almeno un modello fuori; documentato in estimator_check.csv; i run sono
                   rifiutati), FAIL (1), INCOMPLETE (2: tetto estimator_check.cap_minutes o SIGUSR1). Un processo a sé.
  run              un run (--run nuaa_patience_4 | nuaa_patience_12), un processo, con --i-am-authorized e
                   --estimator-check <cartella di un controllo PASS, ufficiale, con la stessa configurazione e lo stesso
                   codice>.

Sequenza di un run (in un solo processo, in quest'ordine):
  0. prima di creare la cartella: i controlli di check-inputs sui file fissati; variabili d'ambiente dei thread uguali
     al record della prova (thread_environment_rule); regola della macchina; record del controllo dello stimatore;
     albero git pulito; regola dei tentativi (attempt_rule); cartella del run nuova.
  1. status.json RUNNING e run.log; ingressi (PAC.load_inputs), schede di riferimento e chiavi; FrameStore; import di
     TensorFlow e subito PAC.set_threads (10 e 10); il confronto con il record della prova (threads_rule) è registrato
     in status.json e, se diverso, FAIL prima di ogni addestramento; GPU nascosta.
  2. fase 1, nelle 24 posizioni di run_order.csv, le stesse operazioni di TensorFlow fra due addestramenti dei run
     accettati tranne l'assemblaggio e i punteggi del test: tetto; fitting e dev dell'unità (unit_frames: solo questi due
     ruoli) e chiavi uguali alla scheda di riferimento; thread riletti; addestramento con PAC.trial_trainer (ricetta del
     run accettato con la patience del run, al più 12 epoche, EpochTimer, tetto a fine epoca); un valore non finito nella
     storia nuova (qualunque rete) -> FAIL con grandezza ed epoca, prima del checkpoint (scelta dell'autore: la scheda,
     che contiene la storia, non si può scrivere con PAC.json_text); checkpoint .keras salvato subito dopo
     (save_checkpoint, le righe di nuaa_folds_run.default_trainer); controlli della scheda; per LivenessNet il controllo
     della storia (history_check: una differenza finita è riportata, il run continua e il modello resta, §7); tetto; punteggi del
     dev con il checkpoint salvato (FR.tf_scorer), logit e val_loss ricalcolato contro il minimo della scheda (come
     nuaa_session_ablation_run.train_one); thread riletti; scheda e punteggi della fase 1 (_dev_scorer.csv) scritti.
  3. fase 2, dopo i 24 addestramenti, nello stesso ordine: tetto; checkpoint ricaricato e controllato; stato «ordinary»
     sul dev con il valutatore della PR #27 (BR.evaluate) prima della ristima; accordo vincolante con i punteggi della
     fase 1 (evaluator_agreement: scarto <= 1e-5, nessuna decisione discordante a 0,5 e alla soglia EER, tolleranze della
     configurazione della PR #27); copia in memoria, ristima (reestimate), controlli della PR #27 (tensori, coerenza e
     conteggi, softmax), stato «re-estimated» sul dev con lo stesso valutatore; soglie EER, decisioni e conteggi
     one-class dai punteggi di BR.evaluate nei due stati.
  4. ID dei frame del dev: fra le due reti di ogni unità e fra le unità della stessa fold (D.prepare_units) e fra i due
     stati della BN (D.same_frames), sui confronti possibili; PAC.assert_no_test; riverifica di configurazione, codice e
     ingressi fissati; uscite.
Tetto: le ore di tempo reale di run_caps.csv per il run (5,00), controllate prima di ogni addestramento, a fine epoca,
prima dei punteggi del dev, prima della fase 2 di ogni modello, prima di ogni strato della ristima, del passaggio di
coerenza e del dev ristimato. Al tetto, o con SIGUSR1: INCOMPLETE; i file del modello in corso nella fase 1 sono tolti e
il modello è marcato «discarded at the cap» (né salvato né valutato, come nei run accettati); nessun altro passo parte.
Con la ristima in fase 2, un arresto nella fase 1 lascia tutti i modelli senza gli stati della BN valutati (la fase 2 non
parte). Nessuna ripresa: un nuovo tentativo ripete le 24 posizioni in un processo e in una cartella nuovi.
Esiti del run: PASS (0); FAIL (1: uscite pubbliche tolte; restano status.json con la ragione, run.log e i file riservati
dei modelli completi); INCOMPLETE (2: uscite pubbliche dei modelli completi); MISMATCH (4: accordo fra i due valutatori
non rispettato per almeno un modello; gli addestramenti non sono interrotti, ma nessuna uscita pubblica di risultati: la
discrepanza è documentata per modello in status.json; i file riservati restano con il loro SHA256SUMS). Un MISMATCH
prevale su INCOMPLETE (lo stato registra anche l'arresto).

Regola dei tentativi (attempt_rule, scelta dell'autore): nella radice d'uscita si cercano le cartelle dello stesso run
(qualunque data e commit); --attempt deve essere l'ultimo tentativo + 1 (1, senza suffisso, solo se non ce n'è nessuno);
lo stato registra il numero del tentativo e i run id precedenti con il loro stato. La radice d'uscita ufficiale è
results/archrun, riconosciuta per identità dei file (path_guard: altre grafie e collegamenti simbolici compresi); un'altra
radice, solo fuori dal repository, è ammessa solo con --unofficial-out-root, e il run è marcato non ufficiale nello stato e
nelle uscite; un run ufficiale richiede un controllo dello stimatore ufficiale. Una voce della radice che somiglia a un
run id dello stesso run ma non ne ha la forma (per esempio -r02, -r1, -r3.old) ferma il comando.

Procedura operativa (configurazione, operations): results/ è versionato, quindi dopo check-estimator e dopo il primo run
l'albero non è pulito e check-estimator e run (i soli comandi che controllano l'albero; check-inputs no) rifiutano finché
non c'è un commit; SIGHUP e SIGTERM danno FAIL, SIGUSR1 dà INCOMPLETE al controllo successivo. Lancio con
nohup caffeinate -dims e l'uscita rediretta in un file fuori dal repository: senza redirezione nohup crea nohup.out nella
cartella corrente (la radice del repository), l'albero non è più pulito e il comando rifiuta; il tetto conta il tempo
reale (SR.Budget: il massimo fra orologio monotono e di sistema), quindi anche il tempo in cui la macchina dorme, e
caffeinate lo impedisce. I run partono solo dopo il via del coordinatore e la pubblicazione chiesta (authorisation).

Uscite (results/archrun/<run-id>): status.json (started, finished, tempo reale e CPU del processo misurati una volta,
macchina, regola della macchina e dei thread, commit, stato dell'albero, SHA-256 di configurazione, codice e ingressi,
tentativo, comando con tutti gli argomenti presi dai valori interpretati, senza percorsi personali; status.json passa
dalla scansione di PAC.output_problems, e il comando prima di creare la cartella), run.log; public/ models.csv, one_class.csv, summary.json, report.md,
SHA256SUMS (aggregati, nessun identificativo di soggetto o di frame, nessun percorso: PAC.output_problems prima della
scrittura); restricted/ <rete>_fold<k>_seed<s>.keras, _card.json, _dev_scorer.csv (fase 1), _dev_ordinary.csv,
_dev_re-estimated.csv, _bn_statistics.json, SHA256SUMS. I file riservati contengono gli ID dei frame.

  .venv/bin/python -I -B scripts/pad_arch_run.py check-inputs [--estimator-check results/archrun/<id>]
  nohup caffeinate -dims .venv/bin/python -I -B scripts/pad_arch_run.py check-estimator --i-am-authorized > <log fuori dal repository> 2>&1 &
  nohup caffeinate -dims .venv/bin/python -I -B scripts/pad_arch_run.py run --run nuaa_patience_4 --estimator-check results/archrun/<id> --i-am-authorized > <log fuori dal repository> 2>&1 &
  nohup caffeinate -dims .venv/bin/python -I -B scripts/pad_arch_run.py run --run nuaa_patience_12 --estimator-check results/archrun/<id> --i-am-authorized > <log fuori dal repository> 2>&1 &

Dipendenze: NumPy; TensorFlow 2.15 (solo check-estimator e run, solo CPU). Formati: configurazione
pad-arch-run-config/1, run pad-arch-run/1, controllo dello stimatore pad-arch-estimator-check/1.
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
import pad_arch_comparison as PAC  # noqa: E402
import pad_arch_definitions as D  # noqa: E402
import pad_arch_models as AM  # noqa: E402
import path_guard as pg  # noqa: E402

Failure, fail = FR.Failure, FR.fail
sha256_bytes, sha256_file, read_regular, rel_path, csv_text, parse_csv = (FR.sha256_bytes, FR.sha256_file, FR.read_regular,
                                                                          FR.rel_path, FR.csv_text, FR.parse_csv)
now = FR.now

DEFAULT_CONFIG = "delivery/archrun/pad_arch_run.json"
DEFAULT_OUT_ROOT = "results/archrun"
CONFIG_SCHEMA = "pad-arch-run-config/1"
RUN_SCHEMA = "pad-arch-run/1"
ESTIMATOR_SCHEMA = "pad-arch-estimator-check/1"
RUNS = {"nuaa_patience_4": 4, "nuaa_patience_12": 12}
RUN_TAG = {"nuaa_patience_4": "p4", "nuaa_patience_12": "p12"}
RUN_ID_RE = re.compile(r"^\d{8}-ARCHRUN-p(4|12)-[0-9a-f]{7,40}(?:-r([2-9]|[1-9]\d+))?$")
EST_ID_RE = re.compile(r"^\d{8}-ARCHEST-[0-9a-f]{7,40}(?:-r([2-9]|[1-9]\d+))?$")
LOOKS_LIKE_RUN = re.compile(r"^\d{8}-ARCHRUN-p(4|12)-", re.IGNORECASE)      # una voce che somiglia a un run id (forma da controllare)
LOOKS_LIKE_EST = re.compile(r"^\d{8}-ARCHEST-", re.IGNORECASE)
THIS_FILE = "scripts/pad_arch_run.py"
CODE_FILES = PAC.CODE_FILES + ("scripts/pad_arch_comparison.py", "scripts/nuaa_bn_test.py")
TOP_KEYS = ("schema", "task", "authorisation", "operations", "preregistration", "trial_record", "plan_files", "code", "estimator_check",
            "runs", "machine_rule", "sequence", "checkpoint", "history_check", "estimator", "dev_evaluation", "cap", "after_stop",
            "attempts", "outputs", "choices_of_the_author", "test")
CHOICE_KEYS = ("sequence", "checkpoint", "estimator_order", "estimator_check_process", "estimator_check_models", "estimator_check_cap",
               "after_stop", "attempts", "machine_versions", "machine_scope", "threads_record", "evaluator", "evaluator_agreement",
               "history_outcome", "history_first_differing_epoch", "history_non_finite", "dev_threshold_missing_class", "outputs")
PACKAGE_ENTRIES = {"status_sha256": "trial/status.json", "trial_json_sha256": "trial/trial.json"}
MACHINE_FIELDS = ("processor", "core_levels", "ram_gb", "os", "logical_cpus", "architecture")
VERSION_FIELDS = ("python", "numpy", "tensorflow_installed", "keras", "platform")
THREAD_FIELDS = ("requested", "effective", "environment", "set_by_this_runner")
ROLES = PAC.TRIAL_ROLES                              # ("fitting", "dev"): gli unici ruoli assemblati
ARM = "arm_b"
DATASET = "NUAA"
HISTORY_KEYS = PAC.HISTORY_KEYS
STATUS_CODE = {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2, "MISMATCH": 4}
STATUS, LOG, PUBLIC, RESTRICTED = FR.STATUS, FR.LOG, "public", "restricted"
PUBLIC_OUTPUTS = ("models.csv", "one_class.csv", "summary.json", "report.md", "SHA256SUMS")
PER_MODEL = (".keras", "_card.json", "_dev_scorer.csv", "_dev_ordinary.csv", "_dev_re-estimated.csv", "_bn_statistics.json")
EST_OUTPUTS = ("estimator_check.csv", "estimator_check.json")
CAP_UNIT = PAC.RUN_CAP_UNIT
EST_CAP_UNIT = PAC.CAP_UNIT                         # minuti di tempo reale
DISCARDED = "discarded at the cap"
SCORER_COLUMNS = ("network", "fold", "train_seed", "role", D.FRAME_ID_FIELD, "subject_id", "session_id", "label", "score_attack",
                  "logit_bona_fide", "logit_attack", "file_sha256", "checkpoint_sha256")
SCORE_COLUMNS = ("network", "fold", "train_seed", "bn_state", "role", D.FRAME_ID_FIELD, "subject_id", "session_id", "label", "score_attack",
                 "logit_bona_fide", "logit_attack", "prediction_05", "threshold_dev_eer", "prediction_dev_eer", "file_sha256",
                 "checkpoint_sha256")
STATE_FIELDS = ("n", "n_attack", "n_bona_fide", "threshold_eer", "decided_attack_05", "correct_05", "accuracy_05", "one_class_05",
                "decided_attack_eer", "correct_eer", "accuracy_eer")
MODEL_COLUMNS = (("run", "position", "network", "fold", "seed", "patience", "model_status", "phase_1", "phase_2", "epochs_run",
                  "restored_epoch", "restored_by", "best_val_loss", "history_check", "epochs_reference", "epochs_new", "first_differing_epoch",
                  "largest_abs_difference", "field_of_largest_difference", "epoch_of_largest_difference", "history_non_finite")
                 + tuple(f"{s.replace('-', '_')}_{f}" for s in D.BN_STATES for f in STATE_FIELDS)
                 + ("evaluator_max_abs_diff", "evaluator_discordant_05", "evaluator_discordant_eer", "evaluator_agreement", "train_seconds",
                    "train_cpu_seconds", "phase_1_seconds", "phase_2_seconds"))
ONE_CLASS_COLUMNS = ("network", "patience", "bn_state", "one_class", "evaluable", "not_evaluable", "all_bona_fide", "all_attack", "text")
EST_COLUMNS = ("arm", "fold", "seed", "group") + BT.RECAL_COLUMNS[4:]
EXTRA_PROTECTED = PAC.EXTRA_PROTECTED + ("results/archcmp",)
CHECKPOINT_TEXT = ("saved by this runner right after the trainer returned: the weights restored by the trainer (EarlyStopping "
                   "restore_best_weights, or set_weights when the epochs end first), model.save of Keras 2.15 to a .keras file, "
                   "written to a temporary name and renamed, as nuaa_folds_run.default_trainer")
TEST_SENTENCE = "no test frame is assembled or evaluated, no test score or metric is read"
TEST_READ = (
    f"{TEST_SENTENCE}. What each command reads. check-inputs and run: this configuration; the configuration of the pre-registration; "
    "the code files pinned here and this file; the status.json and trial.json of the official trial and the SHA256SUMS of the delivered "
    "package that lists them; run_order.csv and run_caps.csv; the configuration of PR #27 (validated) and the configuration of PR #28 "
    "(bytes, for its SHA-256); what pad_arch_comparison.load_inputs reads (its TEST_READ): the five M2 manifests (read to index the cache; "
    "official_test and cnn_test are NUAA's official lists, not the test role of the folds), the cache, hashed whole with its label and "
    "subject members read, the extracted frames file, loaded whole with the SHA-256 of every frame (both frame files also hold frames of "
    "the held-out subjects), the cells run (status.json, both SHA256SUMS, its configuration, the cell manifests of the arms and of the "
    "common dev), the folds run (status.json, restricted SHA256SUMS, extracted_frames.csv), the LivenessNet architecture file, the "
    "configurations, status.json and restricted SHA256SUMS of the two accepted runs and the card arm_b_fold1_seed42 of the run at "
    "patience 4; in addition the 12 training cards of arm B of each accepted run (history and keys of the fitting and dev frames); with "
    "--estimator-check, the status.json and estimator_check.json of a check-estimator record. run, in addition: the image arrays of the "
    "cache, loaded whole into the frame store; of all the frames it assembles only the fitting and dev frames of arm B (an explicit check "
    "refuses any other role); it reads back its own checkpoints. check-estimator: this configuration, the configuration of the "
    "pre-registration (for the threads) and, through pad_arch_comparison.load_config, the configurations of the two accepted runs, the code, the official trial record and the package SHA256SUMS, run_order.csv and run_caps.csv, "
    "the configurations of PRs #27 and #28 (both validated), what nuaa_bn_recalibration.load_inputs reads (the same M2 manifests, cache "
    "and extracted frames file, loaded whole, the cells run, the status.json, restricted SHA256SUMS and extracted_frames.csv of the folds "
    "run, the revised note of PR #27, the architecture file, and of the parent run at patience 4 status.json, both SHA256SUMS, its public "
    "aggregate.json only as bytes for the SHA-256 (it holds test metrics, never parsed), and per model card, dev scores, threshold and "
    "checkpoint), of the run 20261114-BNRECAL-a10a082 what nuaa_bn_test.load_bnrecal reads (status.json, both SHA256SUMS, public "
    "results.csv and run_record.json, restricted public_SHA256SUMS.sha256, per model _scores.csv and _bn_statistics.json, which hold dev "
    "and fitting rows only); it assembles only fitting and dev frames. No command loads the fold schema (fold<k>_test.csv) or opens a "
    "<model>_test_scores.csv")

# punti di iniezione (solo i test)
REPO_STATE = None       # (root) -> {"commit", "dirty"} (default: SR.repo_state)
MACHINE = None          # () -> macchina (default: PAC.machine)
ENVIRONMENT = None      # () -> versioni (default: FR.environment)
TF_MODULE = None        # () -> modulo TensorFlow (default: import tensorflow)
ARCH_LOADER = None      # (path) -> modulo con create_model (default: FR.default_arch_loader)
TRAINER = None          # come PAC.trial_trainer
SAVE_MODEL = None       # (model, path) -> None (default: save_checkpoint)
MAKE_SCORER = None      # (weights_path, sha256) -> funzione(X uint8) -> probabilità N x 2 (default: FR.tf_scorer)
REESTIMATE = None       # (ctx, rec, fit, dev) -> risultato della fase 2 (default: reestimate_model)
ESTIMATOR_BASE = None   # radice degli ingressi di check-estimator (default: --input-root)


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
    """L'ordine della ristima per rete (scelta dell'autore): LivenessNet, i cinque strati nell'ordine della rete e della
    PR #27 (AM.LIVENESSNET_BN); ResNet-18, i 20 strati nell'ordine della specifica AM.resnet18_spec (stem, poi per stadio
    e blocco bn1, bn2 e, nel primo blocco degli stadi 2-4, downsample_bn). Che sia un ordine topologico è controllato
    sul grafo (bn_order_problems)."""
    return {"LivenessNet": list(AM.LIVENESSNET_BN), "ResNet-18": [n for n, kind, _ in AM.resnet18_spec() if kind == "bn"]}


def load_config(p: Path) -> tuple:
    """(configurazione, SHA-256 dei byte letti). Insieme chiuso a ogni livello, tipi e valori fissati controllati qui,
    prima di leggere qualunque altro file; nessun valore predefinito (un campo mancante o in più è un errore)."""
    data = read_regular(p, "configuration")
    cfg = SR.read_json(data, f"configuration {p.name}")
    if cfg.get("schema") != CONFIG_SCHEMA:
        fail(f"configuration {p.name}: schema must be {CONFIG_SCHEMA!r}")
    if sorted(cfg) != sorted(TOP_KEYS):
        fail(f"configuration: fields must be exactly {', '.join(TOP_KEYS)} (missing {sorted(set(TOP_KEYS) - set(cfg))}, "
             f"unknown {sorted(set(cfg) - set(TOP_KEYS))})")
    for k in ("task", "authorisation", "operations"):
        _text(cfg[k], k)
    if "merge" not in cfg["authorisation"] or "go-ahead" not in cfg["authorisation"]:
        fail("configuration: authorisation must say that the runs start after the verification, the merge and the go-ahead")
    if "nohup" not in cfg["operations"] or "commit" not in cfg["operations"]:
        fail("configuration: operations must state the commit between the commands and the launch with nohup")
    if cfg["test"] != TEST_READ:
        fail("configuration: test must state exactly what is read (TEST_READ of the script)")
    _pin(cfg["preregistration"], "preregistration")
    tr = cfg["trial_record"]
    _keys(tr, ["folder", "package_sums", "run_id", "status_sha256", "trial_json_sha256"], "trial_record")
    _text(tr["folder"], "trial_record.folder")
    _pin(tr["package_sums"], "trial_record.package_sums")
    if not isinstance(tr["run_id"], str) or not PAC.RUN_ID_RE.match(tr["run_id"]):
        fail("configuration: trial_record.run_id must be an ARCHTRIAL run id")
    if not SR._hex(tr["status_sha256"]) or not SR._hex(tr["trial_json_sha256"]):
        fail("configuration: trial_record SHA-256 values malformed")
    pf = cfg["plan_files"]
    _keys(pf, ["run_caps", "run_order"], "plan_files")
    for k in pf:
        _pin(pf[k], f"plan_files.{k}")
    _keys(cfg["code"], list(CODE_FILES), "code")
    if not all(SR._hex(v) for v in cfg["code"].values()):
        fail("configuration: code SHA-256 values malformed")
    ec = cfg["estimator_check"]
    _keys(ec, ["bn_recalibration_config", "bn_test_config", "bnrecal_run_id", "cap_minutes", "cap_rule", "cap_unit", "models", "rule",
               "tolerances", "where"], "estimator_check")
    _pin(ec["bn_recalibration_config"], "estimator_check.bn_recalibration_config")
    _pin(ec["bn_test_config"], "estimator_check.bn_test_config")
    if not isinstance(ec["bnrecal_run_id"], str) or not BR.RUN_ID_RE.match(ec["bnrecal_run_id"]):
        fail("configuration: estimator_check.bnrecal_run_id must be a BNRECAL run id")
    if not FR.is_count(ec["models"]) or ec["models"] < 1:
        fail("configuration: estimator_check.models must be a positive integer")
    if not FR.is_count(ec["cap_minutes"]) or ec["cap_minutes"] < 1:
        fail("configuration: estimator_check.cap_minutes must be a positive integer")
    if ec["cap_unit"] != EST_CAP_UNIT:
        fail(f"configuration: estimator_check.cap_unit must be {EST_CAP_UNIT!r}")
    for k in ("rule", "tolerances"):
        _text(ec[k], f"estimator_check.{k}")
    if "stops" not in ec["rule"]:
        fail("configuration: estimator_check.rule must state that the work stops when the statistics are not reproduced")
    _author(ec["where"], "estimator_check.where")
    _author(ec["cap_rule"], "estimator_check.cap_rule")
    if cfg["runs"] != list(RUNS):
        fail(f"configuration: runs must be {list(RUNS)} (one per patience, in the order of the plan)")
    mr = cfg["machine_rule"]
    _keys(mr, ["consequence", "fields", "scope", "threads", "versions"], "machine_rule")
    if mr["fields"] != list(MACHINE_FIELDS) or mr["versions"] != list(VERSION_FIELDS) or mr["threads"] != list(THREAD_FIELDS):
        fail(f"configuration: machine_rule must compare fields {list(MACHINE_FIELDS)}, versions {list(VERSION_FIELDS)} and threads "
             f"{list(THREAD_FIELDS)} with the record of the official trial")
    _author(mr["consequence"], "machine_rule.consequence")
    if "before any training" not in mr["consequence"]:
        fail("configuration: machine_rule.consequence must say that a difference stops the work before any training")
    _author(mr["scope"], "machine_rule.scope")
    if "not the identity of the physical unit" not in mr["scope"]:
        fail("configuration: machine_rule.scope must say that the rule compares the machine model, not the identity of the physical unit")
    sq = cfg["sequence"]
    _keys(sq, ["phase_1", "phase_2", "reason"], "sequence")
    _text(sq["phase_1"], "sequence.phase_1")
    _text(sq["phase_2"], "sequence.phase_2")
    _author(sq["reason"], "sequence.reason")
    if "threshold.json" not in sq["reason"]:
        fail("configuration: sequence.reason must state the operations of the accepted runs that are left out (threshold.json)")
    ck = cfg["checkpoint"]
    _keys(ck, ["file", "format", "when"], "checkpoint")
    if ck["when"] != CHECKPOINT_TEXT:
        fail("configuration: checkpoint.when must be CHECKPOINT_TEXT of the script")
    _text(ck["file"], "checkpoint.file")
    _text(ck["format"], "checkpoint.format")
    hc = cfg["history_check"]
    _keys(hc, ["consequence", "difference", "fields", "first_differing_epoch", "non_finite", "reference", "rule"], "history_check")
    if hc["fields"] != list(HISTORY_KEYS) or hc["difference"] != 0 or isinstance(hc["difference"], bool):
        fail(f"configuration: history_check must compare {list(HISTORY_KEYS)} with difference 0")
    for k in ("reference", "rule", "consequence"):
        _text(hc[k], f"history_check.{k}")
    _author(hc["first_differing_epoch"], "history_check.first_differing_epoch")
    _author(hc["non_finite"], "history_check.non_finite")
    if "stays in the comparison" not in hc["consequence"] or "continues" not in hc["consequence"]:
        fail("configuration: history_check.consequence must say that the run continues and the model stays in the comparison")
    es = cfg["estimator"]
    _keys(es, ["batch", "consistency", "estimator", "order_rule", "orders"], "estimator")
    if es["orders"] != expected_orders():
        fail("configuration: estimator.orders must be the BN layers of each network in the order of expected_orders")
    if es["batch"] != FR.SCORE_BATCH or isinstance(es["batch"], bool):
        fail(f"configuration: estimator.batch must be {FR.SCORE_BATCH} (the batch of the estimator of PR #27)")
    for k in ("consistency", "estimator"):
        _text(es[k], f"estimator.{k}")
    _author(es["order_rule"], "estimator.order_rule")
    de = cfg["dev_evaluation"]
    _keys(de, ["agreement", "evaluator", "frame_ids", "one_class", "states", "threshold"], "dev_evaluation")
    if de["states"] != list(D.BN_STATES):
        fail(f"configuration: dev_evaluation.states must be {list(D.BN_STATES)}")
    for k in ("one_class", "threshold"):
        _text(de[k], f"dev_evaluation.{k}")
    _text(de["frame_ids"], "dev_evaluation.frame_ids")
    if "public outputs" not in de["frame_ids"] or "restricted" not in de["frame_ids"]:
        fail("configuration: dev_evaluation.frame_ids must say that the public outputs hold only counts and the restricted files the identifiers")
    _author(de["evaluator"], "dev_evaluation.evaluator")
    _author(de["agreement"], "dev_evaluation.agreement")
    if "MISMATCH" not in de["agreement"] or "1e-5" not in de["agreement"]:
        fail("configuration: dev_evaluation.agreement must state the tolerance and the MISMATCH outcome")
    cp = cfg["cap"]
    _keys(cp, ["checks", "on_cap", "source", "unit"], "cap")
    if cp["unit"] != CAP_UNIT:
        fail(f"configuration: cap.unit must be {CAP_UNIT!r}")
    for k in ("checks", "on_cap", "source"):
        _text(cp[k], f"cap.{k}")
    if "INCOMPLETE" not in cp["on_cap"] or DISCARDED not in cp["on_cap"]:
        fail(f"configuration: cap.on_cap must say that a run at its cap stops as INCOMPLETE and the model in progress is {DISCARDED!r}")
    _author(cfg["after_stop"], "after_stop")
    if "no automatic resume" not in cfg["after_stop"]:
        fail("configuration: after_stop must state that there is no automatic resume")
    _author(cfg["attempts"], "attempts")
    if "--unofficial-out-root" not in cfg["attempts"]:
        fail("configuration: attempts must state the rule of the attempts and of the unofficial output root")
    ou = cfg["outputs"]
    _keys(ou, ["public", "restricted", "run"], "outputs")
    if ou["public"] != list(PUBLIC_OUTPUTS) or ou["restricted"] != [f"<model>{x}" for x in PER_MODEL] + ["SHA256SUMS"] \
            or ou["run"] != [STATUS, LOG]:
        fail("configuration: outputs must list the public, restricted and run files of the script")
    ch = cfg["choices_of_the_author"]
    _keys(ch, list(CHOICE_KEYS), "choices_of_the_author")
    for k in CHOICE_KEYS:
        _text(ch[k], f"choices_of_the_author.{k}")
    return cfg, sha256_bytes(data)


# ============================================================================ file fissati

class Pins:
    """Ciò che la configurazione fissa, letto e controllato: preregistrazione (cfg_pr, sha_pr, pcfg), codice, record
    della prova, file del piano (posizioni e tetti per run), configurazioni della PR #27 (bn_cfg) e della PR #28
    (bt_cfg, solo in check-estimator) e gli SHA-256 dei file letti (files)."""


def check_code(cfg: dict) -> dict:
    """Ogni file di CODE_FILES uguale allo SHA-256 fissato nella configurazione; più lo SHA-256 attuale di questo file."""
    got = {f: sha256_bytes(read_regular(ROOT / f, "code file")) for f in CODE_FILES}
    bad = [f for f in CODE_FILES if got[f] != cfg["code"][f]]
    if bad:
        fail(f"code files {bad} differ from the SHA-256 pinned in the configuration")
    return {**got, THIS_FILE: sha256_bytes(read_regular(Path(__file__), "code file"))}


def read_pinned(base: Path, e: dict, what: str) -> tuple:
    """(percorso, byte) di un file fissato: file regolare con lo SHA-256 della configurazione."""
    p = rel_path(base, e["path"])
    b = read_regular(p, what)
    if sha256_bytes(b) != e["sha256"]:
        fail(f"{what} {p.name}: SHA-256 {sha256_bytes(b)} differs from the configuration")
    return p, b


def order_text(cfg_pr: dict) -> str:
    """run_order.csv rigenerato dalle funzioni della preregistrazione (PAC.order_rows, PAC.order_cell)."""
    return csv_text(PAC.ORDER_COLUMNS, [[PAC.order_cell(r, k) for k in PAC.ORDER_COLUMNS] for r in PAC.order_rows(cfg_pr)])


def caps_text(cfg_pr: dict, trial: dict) -> str:
    """run_caps.csv rigenerato dalle funzioni della preregistrazione (PAC.run_caps su PAC.plan_rows con il trial.json)."""
    caps = PAC.run_caps(cfg_pr, PAC.plan_rows(cfg_pr, trial))
    return csv_text(PAC.CAPS_COLUMNS, [[PAC.caps_cell(c, k) for k in PAC.CAPS_COLUMNS] for c in caps])


def run_positions(order_bytes: bytes, cfg_pr: dict, run: str) -> list:
    """Le 24 posizioni di un run lette da run_order.csv: posizioni 1..n consecutive, ciascuna con rete, fold, seed e
    patience uguali a PAC.plan_order e con il run di riferimento della preregistrazione per LivenessNet."""
    rows = [r for r in parse_csv(order_bytes, PAC.ORDER_COLUMNS, "run_order.csv") if r["run"] == run]
    nuaa = cfg_pr["plan"]["datasets"][0]
    want = PAC.plan_order(nuaa["splits"], nuaa["seeds"])
    patience = RUNS[run]
    ref = cfg_pr["parent_runs"][f"patience_{patience}"]["run_id"]
    out = []
    if len(rows) != len(want):
        fail(f"run_order.csv: {len(rows)} positions for {run}, the plan has {len(want)}")
    for i, (r, (net, f, s)) in enumerate(zip(rows, want), start=1):
        got = (r["position"], r["network"], r["fold"], r["seed"], r["patience"])
        if got != (str(i), net, str(f), str(s), str(patience)):
            fail(f"run_order.csv: position {i} of {run} is {got[1:]} with position {got[0]}, the plan has {(net, f, s, patience)}")
        if r["reference_run"] != (ref if net == "LivenessNet" else "none"):
            fail(f"run_order.csv: position {i} of {run}: reference run differs from the pre-registration")
        out.append({"position": i, "network": net, "fold": f, "seed": s, "patience": patience,
                    "reference_model": r["reference_model"] if net == "LivenessNet" else None})
    return out


def run_cap_hours(caps_bytes: bytes, run: str) -> float:
    rows = [r for r in parse_csv(caps_bytes, PAC.CAPS_COLUMNS, "run_caps.csv") if r["run"] == run]
    if len(rows) != 1:
        fail(f"run_caps.csv: {len(rows)} rows for {run} (exactly one expected)")
    try:
        cap = float(rows[0]["cap_wall_hours"])
    except ValueError:
        fail(f"run_caps.csv: cap_wall_hours of {run} is not a number")
    if not math.isfinite(cap) or cap <= 0:
        fail(f"run_caps.csv: cap_wall_hours of {run} must be a finite positive number")
    return cap


def trial_package_check(base: Path, tr: dict) -> dict:
    """Il record della prova ancorato al pacchetto congelato: SHA256SUMS del pacchetto (fissato per SHA-256) con le righe
    trial/status.json e trial/trial.json uguali agli SHA-256 della configurazione."""
    _, b = read_pinned(base, tr["package_sums"], "package SHA256SUMS")
    sums = FR.read_sums(b, "package SHA256SUMS")
    bad = [entry for key, entry in PACKAGE_ENTRIES.items() if sums.get(entry) != tr[key]]
    if bad:
        fail(f"trial record: {', '.join(bad)} of the delivered package SHA256SUMS differ from the SHA-256 of the configuration")
    return {entry: sums[entry] for entry in PACKAGE_ENTRIES.values()}


def load_pins(cfg: dict, base: Path, est_base: Path = None) -> Pins:
    """I file fissati dalla configurazione, ciascuno con il suo SHA-256: preregistrazione (validata da PAC.load_config),
    codice (anche PAC.check_code contro la preregistrazione), record della prova (uguale alle righe del SHA256SUMS del
    pacchetto e validato da PAC.load_trial; stato PASS e run id della configurazione), run_order.csv e run_caps.csv
    uguali byte per byte ai file rigenerati, configurazione della PR #27 (BR.load_config: tolleranze, epsilon, ordine,
    batch e tetto) e, per byte, quella della PR #28. Con est_base (solo check-estimator) la configurazione della PR #28 è
    anche validata da BT.load_config, con i percorsi relativi a est_base, e deve nominare la configurazione della PR #27
    fissata qui (contenuto e SHA-256), il run BNRECAL e il numero dei modelli della configurazione."""
    pins = Pins()
    pins.files = {}
    pp, _ = read_pinned(base, cfg["preregistration"], "pre-registration configuration")
    pins.cfg_pr, pins.sha_pr, pins.pcfg = PAC.load_config(pp, base)
    pins.files["preregistration"] = pins.sha_pr
    pins.code = check_code(cfg)
    PAC.check_code(pins.cfg_pr)
    tr = cfg["trial_record"]
    pins.files["package_sums"] = tr["package_sums"]["sha256"]
    trial_package_check(base, tr)
    folder = rel_path(base, tr["folder"])
    for fn, key in ((STATUS, "status_sha256"), (PAC.TRIAL_JSON, "trial_json_sha256")):
        h = sha256_bytes(read_regular(folder / fn, f"trial record {fn}"))
        if h != tr[key]:
            fail(f"trial record {fn}: SHA-256 {h} differs from trial_record.{key}")
    pins.trial = PAC.load_trial(folder, pins.cfg_pr, pins.sha_pr)
    if pins.trial["status"] != "PASS" or pins.trial["run_id"] != tr["run_id"]:
        fail(f"trial record: status {pins.trial['status']} and run id {pins.trial['run_id']}, the configuration requires PASS and {tr['run_id']}")
    pins.files["trial_status"], pins.files["trial_json"] = tr["status_sha256"], tr["trial_json_sha256"]
    _, ob = read_pinned(base, cfg["plan_files"]["run_order"], "run_order.csv")
    if ob.decode("utf-8", "replace") != order_text(pins.cfg_pr):
        fail("run_order.csv differs from the order regenerated from the pre-registration (PAC.order_rows)")
    _, cb = read_pinned(base, cfg["plan_files"]["run_caps"], "run_caps.csv")
    if cb.decode("utf-8", "replace") != caps_text(pins.cfg_pr, pins.trial):
        fail("run_caps.csv differs from the caps regenerated from the pre-registration and the trial (PAC.run_caps)")
    pins.positions = {r: run_positions(ob, pins.cfg_pr, r) for r in RUNS}
    pins.caps = {r: run_cap_hours(cb, r) for r in RUNS}
    pins.files["run_order"], pins.files["run_caps"] = sha256_bytes(ob), sha256_bytes(cb)
    ec = cfg["estimator_check"]
    brp, _ = read_pinned(base, ec["bn_recalibration_config"], "PR #27 configuration")
    pins.bn_cfg, _ = BR.load_config(brp)
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
    if pins.bn_cfg["budget"]["wall_time_cap_minutes"] != ec["cap_minutes"]:
        fail("estimator_check.cap_minutes must be the cap of the PR #27 configuration (budget.wall_time_cap_minutes)")
    pins.files["bn_recalibration_config"] = ec["bn_recalibration_config"]["sha256"]
    pins.files["bn_test_config"] = ec["bn_test_config"]["sha256"]
    return pins


# ============================================================================ regola della macchina e dei thread

def _same(a: dict, b: dict, f: str) -> bool:
    """Uguali solo se il campo c'è in entrambi, non è nullo in nessuno dei due e ha lo stesso valore."""
    return f in a and f in b and a[f] is not None and b[f] is not None and a[f] == b[f]


def machine_rule(trial: dict) -> dict:
    """Regola della macchina (PROTOCOL.md §7, plan.machine_rule, con le aggiunte dell'autore): la macchina di adesso
    (PAC.machine) e le versioni (FR.environment, senza importare TensorFlow) confrontate campo per campo con il record
    della prova ufficiale (MACHINE_FIELDS, VERSION_FIELDS); uguali solo se presenti e non nulli in entrambi e identici.
    Il confronto è sul modello di macchina e sulle versioni, non sull'identità dell'unità fisica. Una differenza ->
    Failure prima di ogni addestramento. Restituisce il confronto registrato."""
    m_now, e_now = (MACHINE or PAC.machine)(), (ENVIRONMENT or FR.environment)()
    tm, te = trial.get("machine") or {}, trial.get("environment") or {}
    rec = {"trial_run": trial["run_id"],
           "machine": {f: {"trial": tm.get(f), "now": m_now.get(f), "equal": _same(tm, m_now, f)} for f in MACHINE_FIELDS},
           "versions": {f: {"trial": te.get(f), "now": e_now.get(f), "equal": _same(te, e_now, f)} for f in VERSION_FIELDS}}
    bad = [f for part in ("machine", "versions") for f, v in rec[part].items() if not v["equal"]]
    if bad:
        fail(f"machine rule: {', '.join(bad)} differ from the official trial {trial['run_id']} or are missing (same machine model, "
             "versions and threads required): nothing trained")
    rec["equal"] = True
    return rec


def env_text(env: dict) -> str:
    return ", ".join(f"{k} {'absent' if v is None else repr(v)}" for k, v in env.items())


def thread_environment_rule(cfg_pr: dict, trial: dict) -> dict:
    """Le variabili d'ambiente dei thread (TF_NUM_INTRAOP_THREADS, TF_NUM_INTEROP_THREADS), uguale in check-inputs,
    check-estimator e run e prima di creare qualunque cartella: una variabile discordante da training.threads è rifiutata
    da PAC.thread_environment; poi le variabili trovate (testo, o None se assente) devono essere uguali a quelle del
    record della prova ufficiale. Scelta dell'autore, più stretta di training.threads.rule della preregistrazione (che
    ammette variabili presenti e uguali ai valori): i run ripetono esattamente le condizioni della prova, in cui
    entrambe erano assenti. Restituisce le variabili trovate."""
    env = PAC.thread_environment(cfg_pr)
    want = (trial.get("threads") or {}).get("environment")
    if env != want:
        fail(f"thread rule: environment variables {env_text(env)} differ from the official trial {trial.get('run_id')} "
             f"({env_text(want) if isinstance(want, dict) else 'not recorded'}): the runs repeat the conditions of the trial exactly; "
             "refused before any folder is created")
    return env


def threads_rule(trial: dict, threads: dict) -> dict:
    """I thread di adesso (record di PAC.set_threads) confrontati con quelli della prova ufficiale: richiesti, effettivi,
    variabili d'ambiente trovate e impostati dal runner (THREAD_FIELDS). Ogni campo deve esserci in entrambi i record
    e avere lo stesso valore (le variabili d'ambiente assenti valgono None, come nel record della prova). Restituisce il
    confronto con equal; chi chiama lo registra e poi fallisce se equal è False."""
    tt = trial.get("threads") or {}
    rec = {f: {"trial": tt.get(f), "now": threads.get(f), "equal": f in tt and f in threads and tt[f] == threads[f]} for f in THREAD_FIELDS}
    for f in ("requested", "effective", "set_by_this_runner"):
        if rec[f]["trial"] is None or rec[f]["now"] is None:
            rec[f]["equal"] = False
    return {"fields": rec, "equal": all(v["equal"] for v in rec.values()), "trial_run": trial.get("run_id")}


def require_threads(rule: dict) -> None:
    if not rule["equal"]:
        bad = [f for f, v in rule["fields"].items() if not v["equal"]]
        fail(f"thread rule: {', '.join(bad)} differ from the official trial {rule['trial_run']}: nothing trained")


def tf_module():
    """TensorFlow importato qui (o il modulo dei test)."""
    if TF_MODULE is not None:
        return TF_MODULE()
    import tensorflow as tf
    return tf


# ============================================================================ ingressi

def load_reference_cards(cfg_pr: dict, base: Path, patience: int) -> dict:
    """Le 12 schede del braccio B del run accettato alla patience data: ciascuna elencata nel SHA256SUMS riservato del
    run (ancorato da PAC.load_parent_status) e uguale alla sua riga; braccio, fold, seed e patience giusti; storia con
    le cinque grandezze, stessa lunghezza (almeno un'epoca); impronte delle chiavi del fitting e del dev. (fold, seed) ->
    {"card", "sha256", "model"}."""
    key = f"patience_{patience}"
    sums = PAC.load_parent_status(cfg_pr, base, key, None)
    run = rel_path(base, cfg_pr["parent_runs"][key]["folder"])
    nuaa = cfg_pr["plan"]["datasets"][0]
    out = {}
    for f in nuaa["splits"]:
        for s in nuaa["seeds"]:
            model = SR.model_name(ARM, f, s)
            fn = f"{model}_card.json"
            if fn not in sums:
                fail(f"accepted run {cfg_pr['parent_runs'][key]['run_id']}: {fn} not listed in its restricted SHA256SUMS")
            b = read_regular(BR.parent_path(run, model, "card.json"), f"reference {fn}")
            if sha256_bytes(b) != sums[fn]:
                fail(f"reference {fn}: SHA-256 differs from the restricted SHA256SUMS of the accepted run")
            card = SR.read_json(b, f"reference {fn}")
            if (card.get("arm"), card.get("fold"), card.get("train_seed"), card.get("early_stopping_patience")) != (ARM, f, s, patience):
                fail(f"reference {fn}: arm, fold, seed or patience differ from the file name and the run")
            if not SR._hex(card.get("fitting_keys_sha256")) or not SR._hex(card.get("dev_keys_sha256")):
                fail(f"reference {fn}: fitting_keys_sha256 or dev_keys_sha256 missing or malformed")
            h = card.get("history")
            if not isinstance(h, dict) or any(not isinstance(h.get(k), list) or not h[k] for k in HISTORY_KEYS) \
                    or len({len(h[k]) for k in HISTORY_KEYS}) != 1:
                fail(f"reference {fn}: the history must hold {', '.join(HISTORY_KEYS)} with the same, non-zero number of epochs")
            out[(f, s)] = {"card": card, "sha256": sums[fn], "model": model}
    return out


def unit_frames(store, inp, role: str, fold: int) -> dict:
    """I frame di un ruolo di un'unità del braccio B: solo fitting e dev (ogni altro ruolo rifiutato prima di assemblare;
    PAC.assert_no_test dopo averlo registrato), con BR.frames (celle del braccio, dev comune del fold, ordine di chiave)."""
    if role not in ROLES:
        fail(f"role {role!r} refused: this runner assembles only the fitting and dev frames")
    inp.assembled.append(role)
    PAC.assert_no_test(inp)
    return BR.frames(store, inp, role, ARM, fold)


def check_unit_keys(inp, refs: dict, pcfg: dict) -> dict:
    """Per ogni unità (fold, seed) delle schede: conteggi del fitting e del dev uguali alla configurazione del run
    accettato e impronte delle chiavi uguali alla scheda (PAC.keys_check)."""
    out = {}
    for (f, s), ref in sorted(refs.items()):
        for role, n in (("fitting", pcfg["arms"][ARM]["total"]), ("dev", pcfg["dev"]["total"])):
            rows = BR.role_rows(inp, role, ARM, f)
            if len(rows) != n:
                fail(f"fold {f} {role}: {len(rows)} frames in the manifests, {n} in the accepted configuration")
            PAC.keys_check(ref["card"], role, rows)
        out[f"fold{f}_seed{s}"] = True
    return out


def load_estimator_record(folder: Path, cfg_sha: str, code: dict, official: bool) -> dict:
    """Il record di un controllo dello stimatore: status.json PASS con run id ARCHEST e lo SHA-256 di
    estimator_check.json; estimator_check.json con lo schema, lo stesso stato e run id, la stessa configurazione
    (config_sha256) e lo stesso codice di adesso (code_sha256, questo file compreso), ufficiale se il run è ufficiale,
    regola della macchina e dei thread rispettata, tutti i modelli controllati e passati. Altrimenti Failure."""
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
        fail("estimator check: an official run needs an official check-estimator (in results/archrun)")
    rows = doc.get("models")
    if not isinstance(rows, list) or not rows or len(rows) != doc.get("expected_models") or any(r.get("recalibration_pass") is not True for r in rows):
        fail("estimator check: not every model of the BNRECAL run was checked and passed")
    if (doc.get("machine_rule") or {}).get("equal") is not True or (doc.get("threads_rule") or {}).get("equal") is not True:
        fail("estimator check: the machine and thread rule was not met")
    return {"run_id": st["run_id"], "estimator_check_json_sha256": st["estimator_check_json_sha256"], "models": len(rows),
            "official": doc.get("official") is True}


# ============================================================================ ristima (fattore BN, §3)

def bn_ancestors(topology: dict, name: str) -> set:
    """Gli strati BatchNormalization da cui dipende l'ingresso dello strato name (risalendo gli strati in ingresso della
    topologia: nome -> (classe, ingressi, attributi), la forma di AM.model_topology)."""
    seen, stack, out = set(), list(topology[name][1]), set()
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        if n not in topology:
            fail(f"topology: layer {n} is not in the model")
        if topology[n][0] == "BatchNormalization":
            out.add(n)
        stack.extend(topology[n][1])
    return out


def bn_order_problems(order, topology: dict) -> list:
    """Un ordine di ristima ammesso: esattamente gli strati BatchNormalization del modello, ciascuno una volta, e ogni
    strato dopo tutti gli strati BatchNormalization da cui dipende il suo ingresso (ordine topologico); gli strati che
    non si alimentano a vicenda possono stare in qualunque ordine fra loro (la suite dimostra che il risultato non
    cambia). Elenco dei problemi (vuoto se l'ordine è ammesso)."""
    bn = sorted(n for n, v in topology.items() if v[0] == "BatchNormalization")
    if sorted(order) != bn or len(set(order)) != len(order):
        return [f"the order must hold each of the {len(bn)} BatchNormalization layers of the model exactly once"]
    pos = {n: i for i, n in enumerate(order)}
    return [f"{n} comes before {a}, on which its input depends" for n in order for a in sorted(bn_ancestors(topology, n)) if pos[a] > pos[n]]


def reestimate(copy, part: dict, order, batch: int, budget=None, name: str = "") -> dict:
    """Lo stimatore della PR #27 esteso alle reti con rami: l'ordine dato è controllato sul grafo della copia
    (bn_order_problems su AM.model_topology), poi BR.recalibrate, invariato: per ogni strato, nell'ordine, momenti
    globali per canale dell'ingresso sul fitting in inferenza con gli strati precedenti già ristimati (unione esatta dei
    batch in float64, ultimo batch col suo conteggio, varianza di popolazione), cast a float32, assegnazione dei soli
    moving_mean e moving_variance; il tetto (budget) è controllato prima di ogni strato."""
    if probs := bn_order_problems(list(order), AM.model_topology(copy)):
        fail(f"{name}: BN order refused: " + "; ".join(probs))
    return BR.recalibrate(copy, part, list(order), batch, budget, name)


def pr27_checks(name: str, stats: dict, snap: dict, copy, order, fit_after: dict, final: dict, other: dict, bn_cfg: dict) -> dict:
    """I controlli della PR #27 dopo una ristima, uguali in run e in check-estimator: solo i tensori mobili cambiati e
    uguali alle statistiche installate (BR.tensor_problems), coerenza degli ingressi delle BN entro consistency_rel/abs
    e stessi conteggi per canale (BR.consistency), softmax dei logit entro softmax_of_logits_max_abs_diff sul fitting e
    sull'altro passaggio (other, il dev). Ogni difetto -> Failure; restituisce il controllo di coerenza."""
    if probs := BR.tensor_problems(snap, BR.snapshot_weights(copy), BR.moving_keys(copy, order), stats):
        fail(f"{name}: " + "; ".join(probs))
    rel, abs_ = BR.tol(bn_cfg, "consistency_rel"), BR.tol(bn_cfg, "consistency_abs")
    cons = BR.consistency(stats, final, rel, abs_)
    if bad := {n: c["channels_outside_tolerance"] for n, c in cons.items() if c["channels_outside_tolerance"]}:
        fail(f"{name}: consistency check failed: {bad} channel statistics outside rel {rel!r} + abs {abs_!r}")
    if any(c["count"] != stats[n]["count"] for n, c in cons.items()):
        fail(f"{name}: the consistency pass counted a different number of values per channel")
    sg = BR.tol(bn_cfg, "softmax_of_logits_max_abs_diff")
    for r, what in ((fit_after, "fitting"), (other, "dev")):
        if r["softmax_gap"] > sg:
            fail(f"{name}: after the re-estimation the softmax of the logits differs from the {what} probabilities by {r['softmax_gap']:.2e}")
    return cons


def check_loaded(model, cfg_pr: dict, network: str) -> dict:
    """Un checkpoint ricaricato contro il registro e la configurazione della preregistrazione (AM.check_model: struttura,
    conteggi, strati BN, topologia di ResNet-18)."""
    a = cfg_pr["architectures"][network]
    try:
        return AM.check_model(model, network, a["input_shape"], {"trainable": a["trainable_parameters"], "non_trainable": a["non_trainable_parameters"],
                                                                 "batchnorm_layers": a["batchnorm_layers"]})
    except AM.ModelProblem as e:
        fail(str(e))


# ============================================================================ dev: stati della BN, soglia, conteggi

def dev_state(part: dict, p_att, z, state: str, what: str) -> tuple:
    """Uno stato della BN di un modello sul dev del suo fold: (riepilogo, unità per D.prepare_units). ID dei frame dalle
    chiavi (D.frame_ids); etichette e punteggi d'attacco nel dominio (D.labels_scores); soglia EER del progetto nello
    stesso stato (D.condition_threshold, controllata contro FR.eer_threshold; None se una classe manca: run non
    valutabile); decisioni con la regola del progetto (D.attack_decisions: attacco se score > soglia) a 0,5 e alla
    soglia EER; conteggio «one-class» della definizione (a) (D.one_class_run); accuratezze come frazioni dei frame."""
    ids = D.frame_ids([r["key"] for r in part["rows"]], len(part["y"]), f"{what} dev")
    y, s = D.labels_scores(part["y"], p_att, f"{what} dev {state}")
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (len(y), 2) or not np.all(np.isfinite(z)):
        fail(f"{what} dev {state}: the logits have shape {z.shape} or non-finite values")
    oc = D.one_class_run(y, s)
    thr = D.condition_threshold("dev_eer", y, s) if oc["evaluable"] else None
    eer = FR.eer_threshold(y, s) if oc["evaluable"] else None
    if eer is not None and float(eer["value"]) != thr:
        fail(f"{what} dev {state}: the EER threshold differs between pad_arch_definitions and nuaa_folds_run")
    d05 = D.attack_decisions(s, FR.FIXED_THRESHOLD)
    deer = D.attack_decisions(s, thr) if thr is not None else None
    n = len(y)
    summ = {"state": state, "n": n, "n_attack": int(np.sum(y == 1)), "n_bona_fide": int(np.sum(y == 0)), "threshold_eer": thr,
            "eer_fp_rejected_bona_fide": None if eer is None else eer["fp"], "eer_fn_accepted_attacks": None if eer is None else eer["fn"],
            "decided_attack_05": int(np.sum(d05)), "correct_05": int(np.sum(d05 == (y == 1))), "accuracy_05": float(np.mean(d05 == (y == 1))),
            "one_class_05": oc, "decided_attack_eer": None if deer is None else int(np.sum(deer)),
            "correct_eer": None if deer is None else int(np.sum(deer == (y == 1))),
            "accuracy_eer": None if deer is None else float(np.mean(deer == (y == 1))), "n_frame_ids": len(ids),
            "frame_ids_sha256": D.ids_sha256(ids)}
    unit = {"y": y, "s": s, "subject": np.array([r["subject"] for r in part["rows"]]),
            "threshold": float(FR.FIXED_THRESHOLD) if thr is None else thr, D.FRAME_ID_FIELD: list(ids)}
    return summ, unit


def score_rows(rec: dict, state: str, part: dict, p_att, z, summ: dict) -> list:
    """Le righe per frame di uno stato della BN, con le decisioni della regola del progetto (D.attack_decisions)."""
    thr = summ["threshold_eer"]
    p = np.asarray(p_att, dtype=np.float64)
    d05 = D.attack_decisions(p, FR.FIXED_THRESHOLD)
    deer = D.attack_decisions(p, thr) if thr is not None else None
    rows = []
    for i, r in enumerate(part["rows"]):
        rows.append([rec["network"], str(rec["fold"]), str(rec["seed"]), state, "dev", r["key"], r["subject"], r["session"], r["label"],
                     repr(float(p[i])), repr(float(z[i][0])), repr(float(z[i][1])), "1" if d05[i] else "0",
                     "NA" if thr is None else repr(float(thr)), "NA" if deer is None else ("1" if deer[i] else "0"), r["sha256"],
                     rec["weights_sha256"]])
    return rows


def scorer_rows(rec: dict, part: dict, p_att, z) -> list:
    """Le righe per frame dei punteggi della fase 1 (FR.tf_scorer), senza soglia né decisioni."""
    return [[rec["network"], str(rec["fold"]), str(rec["seed"]), "dev", r["key"], r["subject"], r["session"], r["label"], repr(float(p_att[i])),
             repr(float(z[i][0])), repr(float(z[i][1])), r["sha256"], rec["weights_sha256"]] for i, r in enumerate(part["rows"])]


def evaluator_agreement(p_scorer, p_eval, threshold, bn_cfg: dict) -> dict:
    """Accordo, sullo stato ordinario, fra i punteggi d'attacco della fase 1 (FR.tf_scorer, lo scorer dei run accettati)
    e quelli della fase 2 (BR.evaluate, il valutatore usato nei due stati): stesso numero di frame, scarto assoluto
    massimo <= dev_probability_max_abs_diff e decisioni discordanti <= dev_decisions_discordant_at_0_5 sia a 0,5 sia
    alla soglia EER dello stato ordinario (la regola delle PR #27 e #28; con la configurazione della PR #27: 1e-5 e 0).
    Senza soglia EER (classe assente) il confronto alla soglia non si fa (None)."""
    a, b = np.asarray(p_scorer, np.float64), np.asarray(p_eval, np.float64)
    tol_p, tol_d = BR.tol(bn_cfg, "dev_probability_max_abs_diff"), BR.tol(bn_cfg, "dev_decisions_discordant_at_0_5")
    if a.shape != b.shape:
        return {"max_abs_diff": None, "discordant_05": None, "discordant_eer": None, "threshold_eer": threshold, "tolerance": tol_p,
                "discordant_tolerance": tol_d, "pass": False, "reason": f"{a.shape} scores in phase 1, {b.shape} in phase 2"}
    diff = float(np.max(np.abs(a - b))) if len(a) else 0.0
    d05 = int(np.sum(D.attack_decisions(a, FR.FIXED_THRESHOLD) != D.attack_decisions(b, FR.FIXED_THRESHOLD)))
    deer = None if threshold is None else int(np.sum(D.attack_decisions(a, threshold) != D.attack_decisions(b, threshold)))
    ok = diff <= tol_p and d05 <= tol_d and (deer is None or deer <= tol_d)
    return {"max_abs_diff": diff, "discordant_05": d05, "discordant_eer": deer, "threshold_eer": threshold, "tolerance": tol_p,
            "discordant_tolerance": tol_d, "pass": bool(ok), "reason": None}


def check_dev_frames(recs: list) -> dict:
    """Gli ID dei frame del dev con le funzioni della preregistrazione (una sola definizione per regola): per ogni stato
    della BN, le unità (NUAA, fold, seed) con entrambe le reti in quello stato passano da D.prepare_units (le due reti
    sugli stessi frame: soggetti ed etichette, poi ID nello stesso ordine; unità della stessa fold sugli stessi frame;
    un ID in una sola fold); per ogni unità presente nei due stati, D.same_frames fra i due stati. Una differenza ->
    Failure (nessun ID nel messaggio). Restituisce i confronti eseguiti e, per unità, numero dei frame e impronta."""
    prepared, units_out = {}, {}
    for st in D.BN_STATES:
        units = {}
        for r in recs:
            u = (r.get("dev_units") or {}).get(st)
            if u is not None:
                units.setdefault((r["fold"], r["seed"]), {})[r["network"]] = u
        complete = [{"unit": (DATASET, f, s), "networks": nets} for (f, s), nets in sorted(units.items()) if sorted(nets) == sorted(AM.NETWORKS)]
        prepared[st] = {pu["unit"]: pu for pu in D.prepare_units(complete)} if complete else {}
        for u, pu in prepared[st].items():
            units_out[f"fold{u[1]}_seed{u[2]}"] = {"n_frames": pu["n_frames"], "frame_ids_sha256": pu["frame_ids_sha256"]}
    both = [u for u in prepared[D.BN_STATES[0]] if u in prepared[D.BN_STATES[1]]]
    for u in both:
        D.same_frames(prepared[D.BN_STATES[0]][u], prepared[D.BN_STATES[1]][u], "dev frames, the two BN states")
    return {"units_two_networks_compared": {st: len(prepared[st]) for st in D.BN_STATES}, "units_two_states_compared": len(both),
            "units": units_out}


# ============================================================================ controllo della storia di LivenessNet (§7)

def _first_non_finite(h) -> int:
    for i, v in enumerate(h):
        if not D.finite_real(v):
            return i + 1
    return None


def non_finite_history(h: dict) -> list:
    """I valori non finiti (NaN o infinito) di una storia nuova, uno per grandezza (la prima epoca): nel run (phase_one,
    qualunque rete) un elenco non vuoto ferma il run come FAIL subito dopo l'addestramento, prima del checkpoint. Scelta
    dell'autore: la scheda, che contiene la storia, non si può scrivere (PAC.json_text rifiuta i valori non finiti, come
    nella prova tecnica); PROTOCOL.md §7 (storia diversa riportata, il modello resta nel confronto) riguarda le
    differenze finite."""
    return [f"non-finite value in {k}, new history, epoch {e}" for k in HISTORY_KEYS if (e := _first_non_finite(h[k])) is not None]


def field_differences(k: str, ref: list, new: list) -> dict:
    """Una grandezza della storia (definizioni dell'autore): si confrontano con PAC.history_differences le epoche in comune
    che precedono il primo valore non finito (NaN o infinito, in una delle due storie) e la prima epoca in cui la
    differenza di due valori finiti non è rappresentabile (due valori enormi di segno opposto); la prima epoca diversa è
    il minimo fra la prima differenza trovata lì, l'epoca del valore non finito e l'epoca della differenza non
    rappresentabile; lo scarto massimo e la sua epoca sono quelli delle epoche finite confrontate; i casi non finiti sono
    riportati a parte (non_finite: grandezza, storia, epoca). Nessun Failure. Nel run una storia nuova non finita non
    arriva qui (phase_one si ferma prima: non_finite_history), quindi il ramo non finito vale per la scheda di
    riferimento (le 24 schede dei run accettati non ne hanno: riportato come differenza, il run continua) e resta come
    difesa, perché history_check non sollevi mai e non confronti mai un valore non finito."""
    reasons = []
    e_ref, e_new = _first_non_finite(ref), _first_non_finite(new)
    cut = min([x for x in (e_ref, e_new) if x is not None], default=None)
    m = min(len(ref), len(new))
    if cut is not None:
        reasons.append(f"non-finite value in {k}, {'new' if e_new == cut else 'reference'} history, epoch {cut}")
        m = min(m, cut - 1)
    over = None
    for i in range(m):
        if not math.isfinite(abs(float(new[i]) - float(ref[i]))):
            over = i + 1
            break
    if over is not None:
        reasons.append(f"difference not representable in {k} at epoch {over}")
        m = over - 1
    d = PAC.history_differences(list(ref[:m]), list(new[:m]))
    firsts = [x for x in (d["first_differing_epoch"], cut, over) if x is not None]
    return {**d, "first_differing_epoch": min(firsts) if firsts else None, "non_finite": reasons or None}


def history_check(ref: dict, card: dict) -> dict:
    """La storia di un addestramento di LivenessNet contro la scheda del run accettato alla stessa patience, per
    ciascuna delle cinque grandezze con field_differences (PAC.history_differences sulle epoche finite in comune: prima
    epoca diversa, scarto assoluto massimo e sua epoca; nessuna tolleranza). Definizioni dell'autore: la prima epoca
    diversa del modello è la prima epoca in cui le due storie differiscono in una grandezza qualunque, dove un'epoca
    presente in una sola delle due conta come diversa (min(epoche) + 1 se i numeri di epoche differiscono), e un valore
    non finito o una differenza non rappresentabile contano come diversi nella loro epoca; esito «identical» solo se non
    c'è nessuna epoca diversa, altrimenti «different». Lo scarto più grande è il massimo fra le grandezze (a pari valore
    la prima in HISTORY_KEYS), con la grandezza e la sua epoca. L'esito non ferma il run."""
    rh, nh = ref["card"]["history"], card["history"]
    fields = {k: field_differences(k, rh[k], nh[k]) for k in HISTORY_KEYS}
    reasons = [x for k in HISTORY_KEYS for x in (fields[k]["non_finite"] or [])]
    n_ref, n_new = len(rh["loss"]), len(nh["loss"])
    firsts = [v["first_differing_epoch"] for v in fields.values() if v["first_differing_epoch"] is not None]
    if n_ref != n_new:
        firsts.append(min(n_ref, n_new) + 1)
    big = max(HISTORY_KEYS, key=lambda k: (fields[k]["max_abs_difference"], -HISTORY_KEYS.index(k)))
    largest = fields[big]["max_abs_difference"]
    return {"reference_model": ref["model"], "reference_card_sha256": ref["sha256"], "epochs_reference": n_ref, "epochs_new": n_new,
            "epochs_compared": min(n_ref, n_new), "outcome": "different" if firsts else "identical",
            "first_differing_epoch": min(firsts) if firsts else None, "largest_abs_difference": largest,
            "field_of_largest_difference": big if largest else None,
            "epoch_of_largest_difference": fields[big]["epoch_of_max_abs_difference"] if largest else None,
            "non_finite": reasons or None, "fields": fields}


def card_problems(card: dict, training: dict, seed: int, weights: Path, network: str) -> list:
    """I controlli della scheda dei run accettati (FR.card_problems: campi, seed, iperparametri, scheduler, epoca
    ripristinata = minimo del val_loss, lr per epoca, checkpoint con il suo SHA-256), con la rete controllata a parte
    (FR.card_problems vuole «LivenessNet»)."""
    out = [] if card.get("arch") == network else [f"architecture {card.get('arch')!r} differs from the position ({network})"]
    return out + FR.card_problems({**card, "arch": "LivenessNet"}, training, seed, weights)


def same_history_records(epochs: list, history: dict, n: int) -> bool:
    """La storia registrata dalla callback dei tempi uguale a quella di Keras, epoca per epoca e grandezza per grandezza,
    confrontando le rappresentazioni (repr) dei valori: un NaN è uguale a un NaN."""
    a = [[repr(float(e[k])) for k in HISTORY_KEYS] for e in epochs]
    b = [[repr(float(history[k][i])) for k in HISTORY_KEYS] for i in range(n)]
    return a == b


# ============================================================================ fase 1 e fase 2

class Ctx:
    """Lo stato di un run in corso (configurazioni, ingressi, frame, cartella, tetto, TensorFlow, schede di riferimento)."""


def save_checkpoint(model, path: Path) -> None:
    """Il salvataggio del checkpoint dei run accettati (nuaa_folds_run.default_trainer): model.save su un nome temporaneo
    e os.replace; mai sopra un file esistente; nessun temporaneo resta dopo un errore."""
    if os.path.lexists(path):
        fail(f"{path.name} already exists (never overwritten)")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.keras")
    try:
        model.save(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def model_files(name: str) -> list:
    return [f"{name}{x}" for x in PER_MODEL]


def discard_model(rdir: Path, name: str) -> list:
    """I file di un modello interrotto (tetto, arresto o errore) tolti, temporanei del checkpoint compresi."""
    if not rdir.is_dir():
        return []
    removed = mw.invalidate(rdir, model_files(name))
    for e in sorted(os.listdir(rdir)):
        if e.startswith(f".{name}.keras.") and e.endswith(".tmp.keras"):
            (rdir / e).unlink()
            removed.append(e)
    return removed


def mark_discarded(rec: dict) -> None:
    """Un modello tolto al tetto: restano solo posizione, nome, rete, fold e seed, con lo stato DISCARDED (nessuna epoca,
    storia o tempo come se fosse completo)."""
    keep = {k: rec[k] for k in ("position", "name", "network", "fold", "seed")}
    rec.clear()
    rec.update(keep, discarded=True)


def phase_one(ctx, pos: dict, rec: dict) -> None:
    """Fase 1 di una posizione: addestramento, checkpoint, scheda, controllo della storia (LivenessNet), punteggi del dev
    con lo scorer dei run accettati e controllo del val_loss. rec["phase_1"] = True solo quando checkpoint, scheda e
    punteggi della fase 1 sono scritti."""
    net, fold, seed = pos["network"], pos["fold"], pos["seed"]
    name = rec["name"]
    t_phase = time.monotonic()
    ctx.budget.check(f"train {name}")
    fit, dev = unit_frames(ctx.store, ctx.inp, "fitting", fold), unit_frames(ctx.store, ctx.inp, "dev", fold)
    ref = ctx.refs[(fold, seed)]
    for role, part in (("fitting", fit), ("dev", dev)):
        PAC.keys_check(ref["card"], role, part["rows"])
    rec["threads_before"] = PAC.check_threads(ctx.tf, ctx.cfg_pr, f"before {name}")
    epochs = []
    timer = PAC.EpochTimer(epochs)
    t0, c0 = time.monotonic(), time.process_time()
    card, model = (TRAINER or PAC.trial_trainer)({"X_fit": fit["X"], "y_fit": fit["y"], "X_dev": dev["X"], "y_dev": dev["y"]}, ctx.training,
                                                 net, seed, lambda: PAC.build_checked(ctx.cfg_pr, net, ctx.arch),
                                                 lambda step: ctx.budget.check(f"{step} of {name}"), timer)
    rec["train_seconds"], rec["train_cpu_seconds"] = time.monotonic() - t0, time.process_time() - c0
    if not same_history_records(epochs, card["history"], card["epochs_run"]) or len(epochs) != card["epochs_run"]:
        fail(f"{name}: the history recorded by the timing callback differs from the Keras history")
    if bad := non_finite_history(card["history"]):
        fail(f"{name}: {'; '.join(bad)}: the run ends as FAIL (a card with a non-finite value cannot be written)")
    weights = ctx.rdir / f"{name}.keras"
    (SAVE_MODEL or save_checkpoint)(model, weights)
    del model
    gc.collect()
    card = {**card, "checkpoint": CHECKPOINT_TEXT, "test_evaluation": "none (no test frame is assembled in this run)",
            "weights_file": weights.name, "weights_sha256": sha256_file(weights), "saved": now(), "network": net, "arm": ARM, "fold": fold,
            "run": ctx.run_name, "position": pos["position"], "fitting_keys_sha256": BR.keys_sha(fit["rows"]),
            "dev_keys_sha256": BR.keys_sha(dev["rows"]), "epoch_timing": epochs,
            "training_overhead": {"seconds": rec["train_seconds"] - sum(e["seconds_total"] for e in epochs),
                                  "cpu_seconds": rec["train_cpu_seconds"] - sum(e["cpu_seconds"] for e in epochs)}}
    rec["weights_sha256"] = card["weights_sha256"]
    if problems := card_problems(card, ctx.training, seed, weights, net):
        fail(f"{name}: " + "; ".join(problems))
    if card["n_fitting"] != ctx.pcfg["arms"][ARM]["total"] or card["n_dev"] != ctx.pcfg["dev"]["total"]:
        fail(f"{name}: the card reports {card['n_fitting']} fitting and {card['n_dev']} dev frames, expected "
             f"{ctx.pcfg['arms'][ARM]['total']} and {ctx.pcfg['dev']['total']}")
    rec["card"] = {k: card[k] for k in ("epochs_run", "restored_epoch", "restored_by", "best_val_loss")}
    if net == "LivenessNet":
        rec["history_check"] = history_check(ref, card)
        card["history_check"] = rec["history_check"]
        print(f"   {name}: history against {ref['model']} of the accepted run: {rec['history_check']['outcome']}"
              + ("" if rec["history_check"]["first_differing_epoch"] is None else f" (first differing epoch {rec['history_check']['first_differing_epoch']})"),
              flush=True)
    ctx.budget.check(f"dev scoring {name}")
    scorer = (MAKE_SCORER or FR.tf_scorer)(weights, card["weights_sha256"])
    p = FR.probabilities(scorer, dev["X"])
    z = SR.dev_logits(weights, card, dev["X"], p, name)
    vl = SR.val_loss_logits(z, dev["y"])
    diff = abs(vl - card["best_val_loss"])
    if not (diff <= FR.VAL_LOSS_TOL[0] or diff <= FR.VAL_LOSS_TOL[1] * card["best_val_loss"]):
        fail(f"{name}: val_loss recomputed on the dev {vl:.6f}, recorded minimum {card['best_val_loss']:.6f}: the checkpoint is not the "
             "one of minimum val_loss")
    card["val_loss_recomputed"] = vl
    rec["scorer_p"] = p[:, 1].copy()
    rec["threads_after"] = PAC.check_threads(ctx.tf, ctx.cfg_pr, f"after {name}")
    mw.write_all(ctx.rdir, {f"{name}_dev_scorer.csv": csv_text(SCORER_COLUMNS, scorer_rows(rec, dev, p[:, 1], z))}, newline="")
    mw.write_all(ctx.rdir, {f"{name}_card.json": PAC.json_text(card, f"{name}_card.json")})
    rec["phase_1"] = True
    rec["phase_1_seconds"] = time.monotonic() - t_phase
    print(f"== {name}: {card['epochs_run']} epochs, restored epoch {card['restored_epoch']}, val_loss {vl:.6f} recomputed on the dev", flush=True)


def reestimate_model(ctx, rec: dict, fit: dict, dev: dict) -> dict:
    """Fase 2 di un modello: checkpoint ricaricato (SHA-256 verificato subito prima, BR.load_model) e controllato
    (check_loaded); tetto; stato ordinario sul dev con il valutatore della PR #27 (BR.evaluate); copia in memoria
    (clone_model + set_weights, uguale sui byte prima della ristima); ristima (reestimate, tetto prima di ogni strato);
    originale invariato; tetto; passaggio di coerenza sul fitting (che dà anche la valutazione del fitting dopo); tetto;
    dev ristimato (BR.evaluate); i controlli della PR #27 (pr27_checks). Ogni difetto -> Failure."""
    tf = BR.tf_module()
    name, net = rec["name"], rec["network"]
    bn_cfg, batch = ctx.bn_cfg, ctx.cfg["estimator"]["batch"]
    order = ctx.cfg["estimator"]["orders"][net]
    tf.keras.backend.clear_session()
    orig = BR.load_model(ctx.rdir / f"{name}.keras", rec["weights_sha256"])
    check_loaded(orig, ctx.cfg_pr, net)
    ctx.budget.check(f"ordinary dev scoring {name}")
    ev_ord, _ = BR.evaluate(orig, dev, batch, f"{name} dev ordinary")
    if ev_ord["softmax_gap"] > BR.tol(bn_cfg, "softmax_of_logits_max_abs_diff"):
        fail(f"{name}: the softmax of the logits differs from the ordinary dev probabilities by {ev_ord['softmax_gap']:.2e}")
    copy = tf.keras.models.clone_model(orig)
    copy.set_weights(orig.get_weights())
    snap = BR.snapshot_weights(orig)
    if BR.snapshot_weights(copy) != snap:
        fail(f"{name}: the in-memory copy differs from the original before the re-estimation")
    stats = reestimate(copy, fit, order, batch, ctx.budget, name)
    if BR.snapshot_weights(orig) != snap:
        fail(f"{name}: the original model changed during the re-estimation")
    ctx.budget.check(f"consistency pass {name}")
    fit_after, final = BR.evaluate(copy, fit, batch, f"{name} fitting after", layers=order)
    ctx.budget.check(f"re-estimated dev scoring {name}")
    dev_after, _ = BR.evaluate(copy, dev, batch, f"{name} dev after")
    cons = pr27_checks(name, stats, snap, copy, order, fit_after, final, dev_after, bn_cfg)
    changes = {n: BR.change_summary(stats[n], bn_cfg["batchnorm"]["epsilon"]) for n in order}
    out = {"stats": stats, "cons": cons, "changes": changes, "ordinary": {"p": ev_ord["p"][:, 1], "z": ev_ord["z"]},
           "re_estimated": {"p": dev_after["p"][:, 1], "z": dev_after["z"]},
           "fitting_after": {k: fit_after[k] for k in ("accuracy", "val_loss", "auroc", "n")}}
    del orig, copy, final
    gc.collect()
    return out


def stats_doc(rec: dict, res: dict, summ: dict, order, cfg: dict) -> dict:
    """Il file riservato delle statistiche ristimate (la forma di BR.stats_json, con la rete e l'ordine): per strato,
    nell'ordine della ristima, statistiche vecchie, in float64, installate in float32, controllo di coerenza; con le
    installate la copia ristimata si ricostruisce dal checkpoint (SHA-256 nel file)."""
    lay = {}
    for n in order:
        s, c = res["stats"][n], res["cons"][n]
        lay[n] = {"kind": "conv" if s["spatial"] > 1 else "dense", "frames": s["frames"], "values_per_frame_per_channel": s["spatial"], "count": s["count"],
                  "channels": int(len(s["mean64"])), "old_moving_mean": [float(x) for x in s["old_mean"]],
                  "old_moving_variance": [float(x) for x in s["old_var"]], "mean_float64": [float(x) for x in s["mean64"]],
                  "variance_float64": [float(x) for x in s["var64"]], "installed_moving_mean_float32": [float(x) for x in s["mean32"]],
                  "installed_moving_variance_float32": [float(x) for x in s["var32"]],
                  "consistency": {"max_abs_diff_mean": c["max_abs_diff_mean"], "max_abs_diff_variance": c["max_abs_diff_var"], "count": c["count"],
                                  "channels_outside_tolerance": c["channels_outside_tolerance"]}, "summary": res["changes"][n]}
    return {"model": rec["name"], "network": rec["network"], "weights_sha256": rec["weights_sha256"], "order": list(order),
            "estimator": cfg["estimator"]["estimator"], "layers": lay, "fitting_after": res["fitting_after"], "dev_re_estimated": summ,
            "reproduce": "load the .keras checkpoint (SHA-256 above) and assign installed_moving_mean_float32 and "
                         "installed_moving_variance_float32 to moving_mean and moving_variance of each layer; every other tensor unchanged"}


def phase_two(ctx, rec: dict) -> None:
    """Fase 2 di un modello completo nella fase 1: i due stati della BN dallo stesso valutatore (BR.evaluate), soglie,
    decisioni e conteggi; accordo vincolante fra lo scorer della fase 1 e il valutatore sullo stato ordinario
    (evaluator_agreement, registrato: un disaccordo dà l'esito MISMATCH alla fine, non interrompe il run)."""
    name = rec["name"]
    t0 = time.monotonic()
    ctx.budget.check(f"re-estimation {name}")
    fit, dev = unit_frames(ctx.store, ctx.inp, "fitting", rec["fold"]), unit_frames(ctx.store, ctx.inp, "dev", rec["fold"])
    res = (REESTIMATE or reestimate_model)(ctx, rec, fit, dev)
    summ_o, unit_o = dev_state(dev, res["ordinary"]["p"], res["ordinary"]["z"], "ordinary", name)
    agreement = evaluator_agreement(rec["scorer_p"], res["ordinary"]["p"], summ_o["threshold_eer"], ctx.bn_cfg)
    summ_r, unit_r = dev_state(dev, res["re_estimated"]["p"], res["re_estimated"]["z"], "re-estimated", name)
    order = ctx.cfg["estimator"]["orders"][rec["network"]]
    doc = stats_doc(rec, res, summ_r, order, ctx.cfg)
    doc["dev_ordinary"], doc["evaluator_agreement"] = summ_o, agreement
    mw.write_all(ctx.rdir, {f"{name}_dev_ordinary.csv": csv_text(SCORE_COLUMNS, score_rows(rec, "ordinary", dev, res["ordinary"]["p"],
                                                                                           res["ordinary"]["z"], summ_o)),
                            f"{name}_dev_re-estimated.csv": csv_text(SCORE_COLUMNS, score_rows(rec, "re-estimated", dev, res["re_estimated"]["p"],
                                                                                               res["re_estimated"]["z"], summ_r))}, newline="")
    mw.write_all(ctx.rdir, {f"{name}_bn_statistics.json": PAC.json_text(doc, f"{name}_bn_statistics.json")})
    rec["states"] = {"ordinary": summ_o, "re-estimated": summ_r}
    rec["dev_units"] = {"ordinary": unit_o, "re-estimated": unit_r}
    rec["evaluator_agreement"] = agreement
    rec["phase_2"] = True
    rec["phase_2_seconds"] = time.monotonic() - t0
    print(f"== {name}: dev EER threshold ordinary {fmt_thr(summ_o)}, re-estimated {fmt_thr(summ_r)}; evaluator agreement "
          f"{'yes' if agreement['pass'] else 'NO'}", flush=True)


def fmt_thr(summ: dict) -> str:
    return "NA" if summ["threshold_eer"] is None else format(summ["threshold_eer"], ".6f")


# ============================================================================ uscite

def cell(v) -> str:
    if v is None:
        return "NA"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def one_class_text(oc) -> str:
    if oc is None:
        return "NA"
    if not oc["evaluable"]:
        return "not evaluable"
    return oc["class"] if oc["one_class"] else "no"


def model_status(rec: dict) -> str:
    if rec.get("discarded"):
        return DISCARDED
    if rec.get("phase_2"):
        return "complete"
    if rec.get("phase_1"):
        return "phase 1 complete, BN states not evaluated"
    return "not started"


def model_row(rec: dict, run: str, patience: int) -> list:
    c = rec.get("card") or {}
    h = rec.get("history_check") or {}
    ag = rec.get("evaluator_agreement") or {}
    hist = ("not applicable" if rec["network"] != "LivenessNet" else None) if not h else h["outcome"]
    v = {"run": run, "position": rec["position"], "network": rec["network"], "fold": rec["fold"], "seed": rec["seed"], "patience": patience,
         "model_status": model_status(rec), "phase_1": bool(rec.get("phase_1")), "phase_2": bool(rec.get("phase_2")),
         "epochs_run": c.get("epochs_run"), "restored_epoch": c.get("restored_epoch"), "restored_by": c.get("restored_by"),
         "best_val_loss": c.get("best_val_loss"), "history_check": hist, "epochs_reference": h.get("epochs_reference"),
         "epochs_new": h.get("epochs_new"), "first_differing_epoch": h.get("first_differing_epoch"),
         "largest_abs_difference": h.get("largest_abs_difference"), "field_of_largest_difference": h.get("field_of_largest_difference"),
         "epoch_of_largest_difference": h.get("epoch_of_largest_difference"),
         "history_non_finite": "; ".join(h["non_finite"]) if h.get("non_finite") else None,
         "evaluator_max_abs_diff": ag.get("max_abs_diff"), "evaluator_discordant_05": ag.get("discordant_05"),
         "evaluator_discordant_eer": ag.get("discordant_eer"), "evaluator_agreement": ag.get("pass"),
         "train_seconds": round(rec["train_seconds"], 1) if "train_seconds" in rec else None,
         "train_cpu_seconds": round(rec["train_cpu_seconds"], 1) if "train_cpu_seconds" in rec else None,
         "phase_1_seconds": round(rec["phase_1_seconds"], 1) if "phase_1_seconds" in rec else None,
         "phase_2_seconds": round(rec["phase_2_seconds"], 1) if "phase_2_seconds" in rec else None}
    for s in D.BN_STATES:
        st = (rec.get("states") or {}).get(s) or {}
        for f in STATE_FIELDS:
            v[f"{s.replace('-', '_')}_{f}"] = one_class_text(st.get(f)) if f == "one_class_05" else (st.get(f) if st else None)
    return [cell(v[k]) for k in MODEL_COLUMNS]


def one_class_rows(recs: list, patience: int) -> list:
    """Il conteggio (a) della preregistrazione (D.one_class_counts) per rete × patience × stato della BN, sui modelli con
    quello stato valutato."""
    records = [{"network": r["network"], "patience": patience, "bn_state": s, "result": r["states"][s]["one_class_05"]}
               for r in recs for s in D.BN_STATES if s in (r.get("states") or {})]
    return D.one_class_counts(records) if records else []


def public_model(rec: dict) -> dict:
    """Ciò che il riepilogo pubblico riporta di un modello: nessun ID di frame o di soggetto (solo conteggi e impronte);
    un modello tolto al tetto ha solo posizione, rete, fold, seed e stato."""
    base = {k: rec[k] for k in ("position", "name", "network", "fold", "seed") if k in rec}
    base["model_status"] = model_status(rec)
    if rec.get("discarded"):
        return base
    return base | {"phase_1": bool(rec.get("phase_1")), "phase_2": bool(rec.get("phase_2")), "card": rec.get("card"),
                   "history_check": rec.get("history_check"), "dev": rec.get("states"), "evaluator_agreement": rec.get("evaluator_agreement"),
                   "threads_before": rec.get("threads_before"), "threads_after": rec.get("threads_after"),
                   "seconds": {k: rec.get(k) for k in ("train_seconds", "train_cpu_seconds", "phase_1_seconds", "phase_2_seconds")}}


def report_md(doc: dict) -> str:
    L = [f"# NUAA run {doc['run']} ({doc['run_id']})", ""]
    if not doc["official"]:
        L += ["**Not an official run**: written outside results/archrun with --unofficial-out-root (a test of the command).", ""]
    L += [f"Status: **{doc['status']}**. Attempt {doc['attempt']['attempt']}. Patience {doc['patience']}; {doc['positions']} positions of "
          f"`run_order.csv`; phase 1 (training, checkpoint, dev scored with the scorer of the accepted runs) complete for "
          f"{doc['complete_phase_1']}, phase 2 (both BN states on the dev with the evaluator of PR #27, BN re-estimated on the fitting set) "
          f"for {doc['complete_phase_2']}. {TEST_SENTENCE[0].upper() + TEST_SENTENCE[1:]}: the frame files, hashed and loaded whole, also hold "
          "frames of the held-out subjects, of which only the fitting and dev frames of arm B are assembled; every number below is a dev or "
          "training number.", "",
          f"Started {doc['started']}, finished {doc['finished']}; elapsed time {doc['wall_seconds']:.0f} s of a cap of {doc['cap_hours']:.2f} "
          f"{doc['cap_unit']}; process CPU time (all threads) {doc['process_cpu_seconds']:.0f} s.", ""]
    if doc["stopped"]:
        L += [f"Stopped: {doc['stopped']['cause']} at {doc['stopped']['step']} ({doc['stopped']['elapsed_minutes']} min)"
              + (f"; {doc['stopped']['discarded']} {DISCARDED}" if doc["stopped"].get("discarded") else "")
              + ("; the stop came before phase 2: no model has its BN states evaluated" if doc["complete_phase_2"] == 0 else "") + ".", ""]
    m = doc["machine"]
    L += [f"Machine: {m['processor']}, cores as named by the system: {PAC.core_levels_text(m)}; {m['logical_cpus']} logical CPUs, "
          f"{PAC.ram_text(m)}, {m['os']}; equal to the official trial {doc['machine_rule']['trial_run']} in "
          f"{', '.join(MACHINE_FIELDS)} and in the versions {', '.join(VERSION_FIELDS)} (machine model and versions, not the physical unit).",
          "", PAC.threads_text(doc["threads"]), "",
          f"BN estimator: checked by {doc['estimator_check']['run_id']} ({doc['estimator_check']['models']} models of the accepted run "
          "reproduced).", "", "## Models", "",
          "| position | network | fold | seed | status | epochs | restored epoch | history check | first differing epoch | ordinary: EER "
          "threshold | ordinary: one-class at 0.5 | re-estimated: EER threshold | re-estimated: one-class at 0.5 | train s |", "|" + "---|" * 14]
    for r in doc["models"]:
        c = r.get("card") or {}
        h = r.get("history_check") or {}
        st = r.get("dev") or {}

        def thr(s):
            return "NA" if s not in st else fmt_thr(st[s])

        def oc(s):
            return "NA" if s not in st else one_class_text(st[s]["one_class_05"])
        hist = h.get("outcome", "not applicable" if r["network"] != "LivenessNet" else "NA")
        first = "NA" if not h else ("none" if h.get("first_differing_epoch") is None else h["first_differing_epoch"])
        L.append(f"| {r['position']} | {r['network']} | {r['fold']} | {r['seed']} | {r['model_status']} | {c.get('epochs_run', 'NA')} | "
                 f"{c.get('restored_epoch', 'NA')} | {hist} | {first} | {thr('ordinary')} | {oc('ordinary')} | {thr('re-estimated')} | "
                 f"{oc('re-estimated')} | {PAC.fmt((r.get('seconds') or {}).get('train_seconds')) or 'NA'} |")
    L += ["", "History check (LivenessNet, against the card of the same arm B fold and seed in the accepted run at the same patience; "
          "five fields, difference 0; an epoch present in one history only counts as differing): a different history is reported and the "
          "model stays in the comparison (PROTOCOL.md §7).", "",
          "## One-class runs at threshold 0.5 on the dev (definition (a))", "", "| network | patience | BN state | count |", "|---|---|---|---|"]
    for o in doc["one_class"]:
        L.append(f"| {o['network']} | {o['patience']} | {o['bn_state']} | {o['text']} (all bona fide {o['all_bona_fide']}, all attack "
                 f"{o['all_attack']}, not evaluable {o['not_evaluable']}) |")
    if not doc["one_class"]:
        L.append("| none | | | no model has its BN states evaluated |")
    f = doc["dev_frames"]
    L += ["", f"Dev frame identifiers (public outputs hold only counts and SHA-256): comparisons made between the two networks of a unit: "
          f"{f['units_two_networks_compared']['ordinary']} units with ordinary BN and {f['units_two_networks_compared']['re-estimated']} with "
          f"re-estimated BN; between the two BN states: {f['units_two_states_compared']} units; every comparison made found the same frames "
          "in the same order.", "", f"Command: `{doc['command']}`", ""]
    return "\n".join(L)


def write_public(run: Path, inp, texts: dict) -> str:
    """Le uscite pubbliche dopo la scansione (PAC.output_problems), con il loro SHA256SUMS; restituisce lo SHA-256 del
    SHA256SUMS."""
    for n, text in texts.items():
        if probs := PAC.output_problems(text, inp):
            fail(f"{n} refused: " + "; ".join(probs))
    files = {**texts, "SHA256SUMS": FR.sums_text(texts)}
    mw.write_all(run / PUBLIC, files, newline="")
    return sha256_bytes(files["SHA256SUMS"].encode("utf-8"))


def write_restricted_sums(rdir: Path) -> str:
    """SHA256SUMS dei file riservati presenti (modelli completi)."""
    names = sorted(e for e in os.listdir(rdir) if e != "SHA256SUMS" and not e.startswith(".")) if rdir.is_dir() else []
    text = "".join(f"{sha256_file(rdir / n)}  {n}\n" for n in names)
    mw.write_all(rdir, {"SHA256SUMS": text}, newline="")
    return sha256_bytes(text.encode("utf-8"))


def final_checks(cfg: dict, cfg_p: Path, cfg_sha: str, base: Path, pins: Pins, refs: dict = None, est_folder: Path = None) -> dict:
    """Alla fine: configurazione, codice, preregistrazione, record della prova e SHA256SUMS del pacchetto, file del piano,
    configurazioni della PR #27 e #28, schede di riferimento e record del controllo dello stimatore con gli stessi SHA-256
    dell'inizio."""
    if sha256_bytes(read_regular(cfg_p, "configuration")) != cfg_sha:
        fail("the configuration changed during the run")
    if check_code(cfg) != pins.code:
        fail("the code changed during the run")
    read_pinned(base, cfg["preregistration"], "pre-registration configuration")
    for e in (cfg["plan_files"]["run_order"], cfg["plan_files"]["run_caps"], cfg["estimator_check"]["bn_recalibration_config"],
              cfg["estimator_check"]["bn_test_config"], cfg["trial_record"]["package_sums"]):
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
    """Run id: YYYYMMDD-ARCHRUN-p4|p12-<commit> (run) o YYYYMMDD-ARCHEST-<commit> (check-estimator), con il suffisso
    -r<k> per il tentativo k > 1."""
    if not re.fullmatch(r"\d{8}", date or ""):
        fail("--date must be YYYYMMDD")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        fail("--attempt must be a positive integer")
    suffix = "" if attempt == 1 else f"-r{attempt}"
    rid = (f"{date}-ARCHRUN-{RUN_TAG[run]}-{state['commit']}" if kind == "run" else f"{date}-ARCHEST-{state['commit']}") + suffix
    if not (RUN_ID_RE if kind == "run" else EST_ID_RE).match(rid):
        fail(f"run id {rid} malformed")
    return rid


def previous_attempts(out_root: Path, kind: str, run: str = None) -> list:
    """Le cartelle dello stesso run nella radice d'uscita (qualunque data e commit): run id, numero del tentativo (1 senza
    suffisso, k con -r<k>) e stato letto dal loro status.json («unreadable» se manca o non si legge)."""
    out = []
    if not out_root.is_dir():
        return out
    for e in sorted(os.listdir(out_root)):
        loose = (LOOKS_LIKE_RUN if kind == "run" else LOOKS_LIKE_EST).match(e)
        if not loose or (kind == "run" and loose.group(1) != str(RUNS[run])):
            continue
        m = (RUN_ID_RE if kind == "run" else EST_ID_RE).match(e)
        if not m:
            fail(f"output root: the entry {e!r} looks like a run id of this run but does not have the expected form "
                 "(YYYYMMDD-ARCHRUN-p4|p12-<commit> or YYYYMMDD-ARCHEST-<commit>, suffix -r<k> with k >= 2): rename or move it")
        g = m.group(2) if kind == "run" else m.group(1)
        k = int(g) if g else 1
        try:
            st = json.loads((out_root / e / STATUS).read_text(encoding="utf-8")).get("status", "unreadable")
        except (OSError, ValueError):
            st = "unreadable"
        out.append({"run_id": e, "attempt": k, "status": st})
    return out


def attempt_rule(out_root: Path, kind: str, run: str, attempt: int) -> dict:
    """La regola dei tentativi: --attempt deve essere l'ultimo tentativo dello stesso run + 1 (1 se non ce n'è nessuno)."""
    prev = previous_attempts(out_root, kind, run)
    need = max((p["attempt"] for p in prev), default=0) + 1
    if attempt != need:
        fail(f"--attempt {attempt} refused: {len(prev)} earlier attempts of this run in the output root "
             f"({', '.join(p['run_id'] + ' ' + p['status'] for p in prev) or 'none'}); this attempt must be --attempt {need}")
    return {"attempt": attempt, "previous": prev}


def path_text(p: Path) -> str:
    """Un percorso nel comando registrato: dentro il repository (per identità, pg.within) relativo alla radice, fuori come
    «<outside the repository>/<nome>»."""
    if pg.within(p, ROOT):
        try:
            return str(Path(p).resolve().relative_to(ROOT.resolve()))
        except ValueError:
            return f"<inside the repository>/{Path(p).name}"
    return f"<outside the repository>/{Path(p).name}"


def command_text(a) -> str:
    """Il comando registrato, costruito dai valori interpretati da argparse (non dalla riga grezza: nessuna forma
    --opt=valore sfugge): tutti gli argomenti, con i percorsi risolti come li usa il comando (--config, --out-root,
    --estimator-check rispetto alla radice del repository, --input-root rispetto alla cartella corrente) e scritti con
    path_text."""
    parts = [a.cmd, "--config", path_text(folder_arg(a.config))]
    if a.input_root is not None:
        parts += ["--input-root", path_text(Path(a.input_root).resolve())]
    parts += ["--out-root", path_text(folder_arg(a.out_root))]
    if a.unofficial_out_root:
        parts.append("--unofficial-out-root")
    if a.run is not None:
        parts += ["--run", a.run]
    if a.estimator_check is not None:
        parts += ["--estimator-check", path_text(folder_arg(a.estimator_check))]
    if a.date is not None:
        parts += ["--date", a.date]
    parts += ["--attempt", str(a.attempt)]
    if a.i_am_authorized:
        parts.append("--i-am-authorized")
    return "python -I -B scripts/pad_arch_run.py " + " ".join(shlex.quote(x) for x in parts)


def input_paths(cfg: dict, pins: Pins, base: Path, cfg_p: Path, est_base: Path = None) -> dict:
    out = PAC.input_paths(pins.cfg_pr, pins.pcfg, base, rel_path(base, cfg["preregistration"]["path"]))
    out["the configuration of this runner"] = cfg_p
    out["the trial record"] = rel_path(base, cfg["trial_record"]["folder"])
    out["the package SHA256SUMS"] = rel_path(base, cfg["trial_record"]["package_sums"]["path"])
    for k in ("run_order", "run_caps"):
        out[f"the plan file {k}"] = rel_path(base, cfg["plan_files"][k]["path"])
    out["the PR #28 configuration"] = rel_path(base, cfg["estimator_check"]["bn_test_config"]["path"])
    if est_base is not None:
        out["the BNRECAL run"] = rel_path(est_base, pins.bt_cfg["bnrecal_run"]["folder"])
        out["the parent run of PR #27"] = rel_path(est_base, pins.bn_cfg["parent_run"]["folder"])
    return out


def run_dir_problem(run: Path, out_root: Path, inputs: dict) -> str:
    """Motivo per cui la cartella del run non è ammessa ("" se lo è), prima di qualunque scrittura: esistente, radice
    d'uscita collegamento, ingressi dentro la cartella o che la contengono, dentro il repository solo sotto
    results/archrun (path_guard e m2_writers.output_problem)."""
    if os.path.lexists(run):
        return f"the run folder {run.name} already exists: every attempt keeps its outcome"
    if out_root.is_symlink():
        return "the output root is a symbolic link"
    for label, p in inputs.items():
        if pg.within(p, run) or (Path(p).is_dir() and pg.within(run, p)):
            return f"{label} {Path(p).name} is inside the run folder, or contains it: inputs and outputs must be separate"
    return mw.output_problem(run, ROOT, pg.BASE_PROTECTED, [ROOT / DEFAULT_OUT_ROOT], [ROOT / x for x in EXTRA_PROTECTED], inputs,
                             [STATUS, LOG], [run / PUBLIC, run / RESTRICTED])


def resolve(a) -> tuple:
    base = Path(a.input_root).resolve() if a.input_root else ROOT
    cfg_p = Path(a.config) if Path(a.config).is_absolute() else ROOT / a.config
    return base, cfg_p


def folder_arg(p) -> Path:
    return Path(p) if Path(p).is_absolute() else ROOT / p


def root_kind(out_root: Path, repo: Path, official: Path, unofficial_flag: bool) -> bool:
    """La radice d'uscita decisa per identità dei file (path_guard.within nelle due direzioni: stessa cartella anche con
    un'altra grafia o attraverso un collegamento simbolico): ufficiale solo se è results/archrun del repository (allora
    il flag --unofficial-out-root è rifiutato); ogni altra radice dentro il repository, o che lo contiene, è rifiutata;
    fuori dal repository è ammessa solo con --unofficial-out-root. Restituisce True per la radice ufficiale."""
    if pg.within(out_root, official) and pg.within(official, out_root):
        if unofficial_flag:
            fail("--unofficial-out-root with the official output root results/archrun: refused")
        return True
    if pg.within(out_root, repo) or pg.within(repo, out_root):
        fail("an output root other than results/archrun inside the repository (or containing it) is refused")
    if not unofficial_flag:
        fail("--out-root other than results/archrun needs --unofficial-out-root: the run is then recorded as not official")
    return False


def official_root(a) -> bool:
    return root_kind(folder_arg(a.out_root), ROOT, ROOT / DEFAULT_OUT_ROOT, a.unofficial_out_root)


def prepare(a, kind: str) -> tuple:
    """I controlli comuni prima di creare una cartella: configurazione, file fissati, variabili dei thread (discordanti o
    diverse dal record della prova: thread_environment_rule), regola della
    macchina, radice d'uscita ufficiale o dichiarata non ufficiale, record dello stimatore (run), albero pulito, regola dei
    tentativi, run id e cartella nuova ammessa."""
    base, cfg_p = resolve(a)
    cfg, cfg_sha = load_config(cfg_p)
    est_base = (ESTIMATOR_BASE or base) if kind == "estimator" else None
    command = command_text(a)
    if probs := PAC.output_problems(command, FR.Inputs()):
        fail("the recorded command holds text that a publishable output must not hold: " + "; ".join(probs))
    pins = load_pins(cfg, base, est_base)
    env = thread_environment_rule(pins.cfg_pr, pins.trial)   # variabili dei thread diverse dalla prova: rifiuto prima di ogni cartella
    mrule = machine_rule(pins.trial)
    official = official_root(a)
    pins.estimator = None
    est_folder = None
    if kind == "run":
        if not a.estimator_check:
            fail("run needs --estimator-check: the folder of a PASS of check-estimator (PROTOCOL.md §3: the statistics of the BNRECAL run "
                 "reproduced before the re-estimation is used on the new models)")
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
    return base, cfg_p, cfg, cfg_sha, pins, env, mrule, state, rid, run, est_folder, official, attempt, command


def write_status(run: Path, doc: dict, inp, redact: bool = False) -> None:
    """status.json dopo la scansione di PAC.output_problems (come le uscite pubbliche): un testo vietato -> Failure; con
    redact (stato FAIL scritto dopo un errore) i testi personali sono sostituiti e lo stato è scritto comunque."""
    text = json.dumps(doc, indent=1) + "\n"
    if probs := PAC.output_problems(text, inp):
        if not redact:
            fail(f"{STATUS} refused: " + "; ".join(probs))
        for t in PAC.personal_tokens():
            text = text.replace(t, "<redacted>")
    mw.write_all(run, {STATUS: text})


def fail_cleanup(run: Path, names: list, log, head: dict, started: str, budget, c0: float, extra: dict, e: BaseException, inp=None) -> None:
    """Dopo un errore: uscite pubbliche tolte, riga d'errore nel log, status.json FAIL con la ragione e i tempi."""
    reason = str(e) if isinstance(e, Failure) else f"unexpected error {type(e).__name__}: {e}"
    left = mw.invalidate_each(run / PUBLIC, names, None, FR.FAIL_DOC) if (run / PUBLIC).is_dir() else []
    outputs = "public removed" if not left else f"left behind: {', '.join(left)}"
    tail = f"ERROR: {reason}\nFAIL outputs: {outputs}\nRESULT: FAIL ({head['command']})\n"
    if log is not None:
        log[0].write(tail)
        FR.close_log(log)
    elif (run / LOG).is_file() and not (run / LOG).is_symlink():
        with open(run / LOG, "a", encoding="utf-8") as fh:
            fh.write(tail)
    try:
        write_status(run, {**head, "status": "FAIL", "passed": False, "started": started, "finished": now(),
                           "wall_seconds": round(budget.elapsed(), 1), "process_cpu_seconds": round(PAC.cpu_seconds() - c0, 1),
                           **extra, "reason": reason, "outputs": outputs,
                           "run_log_sha256": sha256_file(run / LOG) if (run / LOG).is_file() else "none"}, inp or FR.Inputs(), redact=True)
    except Exception:  # noqa: BLE001
        print("ERROR: status.json could not be written", file=sys.stderr)


# ============================================================================ comandi

def cmd_check_inputs(a) -> int:
    base, cfg_p = resolve(a)
    cfg, cfg_sha = load_config(cfg_p)
    pins = load_pins(cfg, base)
    env = thread_environment_rule(pins.cfg_pr, pins.trial)
    machine_rule(pins.trial)
    est = None
    if a.estimator_check:
        est = load_estimator_record(folder_arg(a.estimator_check), cfg_sha, pins.code, False)
    inp = PAC.load_inputs(pins.cfg_pr, pins.pcfg, base)
    refs = {p: load_reference_cards(pins.cfg_pr, base, p) for p in RUNS.values()}
    units = {p: check_unit_keys(inp, refs[p], pins.pcfg) for p in refs}
    PAC.assert_no_test(inp)
    tfs = FR.tf_state()
    print(f"CHECK-INPUTS: configuration {cfg_sha[:12]}; pre-registration {pins.sha_pr[:12]}; code pinned ({len(CODE_FILES)} files); trial "
          f"{pins.trial['run_id']} PASS and equal to the delivered package; run_order.csv and run_caps.csv equal to the files regenerated "
          "from the pre-registration; "
          + "; ".join(f"{r}: {len(pins.positions[r])} positions, cap {pins.caps[r]:.2f} h" for r in RUNS)
          + f"; machine model and versions equal to the trial ({', '.join(MACHINE_FIELDS + VERSION_FIELDS)}); environment variables "
          + ", ".join(f"{k} {'absent' if v is None else repr(v)}" for k, v in env.items())
          + f"; inputs of the pre-registration PASS; {sum(len(v) for v in refs.values())} reference cards with keys equal to the manifests "
          f"({sum(len(v) for v in units.values())} unit checks); estimator check "
          + (f"{est['run_id']} PASS ({est['models']} models, {'official' if est['official'] else 'not official'})" if est else "not given")
          + f"; {TEST_SENTENCE}; no frame assembled; TensorFlow {tfs}; nothing written")
    print("RESULT: PASS (check-inputs)")
    return 0


def estimator_status(status: str, rows: list) -> str:
    """L'esito di check-estimator dopo i modelli controllati: MISMATCH se l'esito era PASS e almeno un modello non
    riproduce il run BNRECAL (recalibration_pass falso); altrimenti l'esito invariato (INCOMPLETE resta INCOMPLETE)."""
    if status == "PASS" and any(not r["recalibration_pass"] for r in rows):
        return "MISMATCH"
    return status


def cmd_check_estimator(a) -> int:
    if not a.i_am_authorized:
        fail("check-estimator loads the 36 accepted checkpoints and the restricted NUAA frames: pass --i-am-authorized")
    base, cfg_p, cfg, cfg_sha, pins, env, mrule, state, rid, run, _, official, attempt, command = prepare(a, "estimator")
    est_base = ESTIMATOR_BASE or base
    run.mkdir(parents=True)
    head = {"run_id": rid, "commit": state["commit"], "dirty": False, "config_sha256": cfg_sha, "command": "check-estimator",
            "command_line": command, "official": official, "attempt": attempt}
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
        write_status(run, {**head, "status": "RUNNING", "passed": False, "started": started, "pid": os.getpid(), **extra,
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
        tf = tf_module()
        threads = PAC.set_threads(tf, pins.cfg_pr)
        extra["threads"] = threads
        extra["threads_rule"] = threads_rule(pins.trial, threads)
        require_threads(extra["threads_rule"])
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
                stats = reestimate(copy, fit, order, batch, budget, name)
                budget.check(f"consistency pass {name}")
                fit_after, final = BR.evaluate(copy, fit, batch, f"{name} fitting after", layers=order)
                budget.check(f"dev after {name}")
                dev_after, _ = BR.evaluate(copy, dev, batch, f"{name} dev after")
                cons = pr27_checks(name, stats, snap, copy, order, fit_after, final, dev_after, bn_cfg)
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
        status = estimator_status(status, rows)
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
               "order": order, **extra, "machine": (MACHINE or PAC.machine)(), "environment": (ENVIRONMENT or FR.environment)(),
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
        write_status(run, {**head, "status": status, "passed": status == "PASS", "started": started, "finished": finished,
                           "wall_seconds": round(wall, 1), "process_cpu_seconds": cpu_total, "stopped": stopped, **extra,
                           "models_checked": len(rows), "models_reproduced": sum(r["recalibration_pass"] for r in rows),
                           "estimator_check_json_sha256": sha256_bytes(texts["estimator_check.json"].encode("utf-8")),
                           "estimator_check_csv_sha256": sha256_bytes(texts["estimator_check.csv"].encode("utf-8")),
                           "run_log_sha256": sha256_file(run / LOG)}, scan)
    except BaseException as e:
        warnings.showwarning = saved_warning
        fail_cleanup(run, list(EST_OUTPUTS), log, head, started, budget, c0, extra, e, scan)
        raise
    return STATUS_CODE[status]


def restricted_note(rdir: Path) -> str:
    if (rdir / "SHA256SUMS").is_file():
        return "the files of the complete models are kept, with the restricted SHA256SUMS written before the error"
    return "the files of the complete models are kept, not validated (no restricted SHA256SUMS)"


def cmd_run(a) -> int:
    if not a.i_am_authorized:
        fail("run trains 24 networks on the restricted NUAA frames: pass --i-am-authorized (only after the coordinator's verification, merge "
             "and go-ahead)")
    if a.run not in RUNS:
        fail(f"--run must be one of {', '.join(RUNS)}")
    base, cfg_p, cfg, cfg_sha, pins, env, mrule, state, rid, run, est_folder, official, attempt, command = prepare(a, "run")
    patience = RUNS[a.run]
    positions, cap_h = pins.positions[a.run], pins.caps[a.run]
    run.mkdir(parents=True)
    scan = FR.Inputs()
    head = {"run_id": rid, "run": a.run, "commit": state["commit"], "dirty": False, "config_sha256": cfg_sha, "command": "run",
            "command_line": command, "official": official, "attempt": attempt}
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
        write_status(run, {**head, "status": "RUNNING", "passed": False, "started": started, "pid": os.getpid(), **extra,
                           "note": "RUNNING without a live process means a failed run; SIGUSR1 to this pid stops it as INCOMPLETE"}, scan)
        log = FR.open_log(run)
        warnings.showwarning = PAC.show_warning
        print(f"== NUAA run {a.run} ({rid}, attempt {attempt['attempt']}{'' if official else ', NOT OFFICIAL'}): configuration {cfg_sha[:12]}, "
              f"cap {cap_h:.2f} {CAP_UNIT}, pid {os.getpid()}", flush=True)
        inp = PAC.load_inputs(pins.cfg_pr, pins.pcfg, base)
        scan = inp
        refs = load_reference_cards(pins.cfg_pr, base, patience)
        check_unit_keys(inp, refs, pins.pcfg)
        store = FR.FrameStore(inp, inp.X_extracted, inp.extracted_keys)
        tf = tf_module()
        threads = PAC.set_threads(tf, pins.cfg_pr)            # subito dopo l'import, prima di qualunque operazione
        extra["threads"] = threads
        extra["threads_rule"] = threads_rule(pins.trial, threads)
        require_threads(extra["threads_rule"])
        print(f"   {PAC.threads_text(threads)}", flush=True)
        FR.hide_gpu(tf)
        ctx.cfg, ctx.cfg_pr, ctx.pcfg, ctx.bn_cfg, ctx.inp, ctx.store, ctx.budget, ctx.tf, ctx.refs = (cfg, pins.cfg_pr, pins.pcfg, pins.bn_cfg, inp,
                                                                                                    store, budget, tf, refs)
        ctx.run_name, ctx.rdir = a.run, run / RESTRICTED
        ctx.rdir.mkdir()
        ctx.training = {**pins.pcfg["training"], "early_stopping_patience": patience}
        ctx.arch = (ARCH_LOADER or FR.default_arch_loader)(rel_path(base, pins.cfg_pr["architectures"]["LivenessNet"]["path"]))
        status, stopped = "PASS", None
        try:
            for pos, rec in zip(positions, recs):
                ctx.current = rec
                phase_one(ctx, pos, rec)
            ctx.current = None
            for rec in recs:
                ctx.current = rec
                phase_two(ctx, rec)
            ctx.current = None
        except SR.CapExceeded as e:
            status = "INCOMPLETE"
            stopped = {"step": e.step, "cause": e.cause, "elapsed_minutes": round(e.minutes, 1)}
            if ctx.current is not None and not ctx.current.get("phase_1"):
                stopped["discarded"] = ctx.current["name"]
                discard_model(ctx.rdir, ctx.current["name"])
                mark_discarded(ctx.current)
            ctx.current = None
            print(f"== {'CAP' if e.cause == 'cap' else 'STOP'}: {e}; the complete models are kept and reported", flush=True)
        PAC.assert_no_test(inp)
        frames = check_dev_frames([r for r in recs if r.get("phase_2")])
        checks = final_checks(cfg, cfg_p, cfg_sha, base, pins, refs, est_folder)
        mismatches = [{"name": r["name"], **r["evaluator_agreement"]} for r in recs if r.get("phase_2") and not r["evaluator_agreement"]["pass"]]
        if mismatches:
            status = "MISMATCH"
        wall, cpu_total, finished = budget.elapsed(), round(PAC.cpu_seconds() - c0, 1), now()   # misurati una volta, per tutte le uscite
        done1, done2 = [r for r in recs if r.get("phase_1")], [r for r in recs if r.get("phase_2")]
        res_sha = write_restricted_sums(ctx.rdir)
        pub_sha = None
        if status != "MISMATCH":
            doc = {"schema": RUN_SCHEMA, **head, "status": status, "patience": patience, "started": started, "finished": finished,
                   "wall_seconds": round(wall, 1), "process_cpu_seconds": cpu_total, "cap_hours": cap_h, "cap_unit": CAP_UNIT, "stopped": stopped,
                   "positions": len(recs), "complete_phase_1": len(done1), "complete_phase_2": len(done2), "machine": (MACHINE or PAC.machine)(),
                   "environment": (ENVIRONMENT or FR.environment)(), "machine_rule": mrule, "threads": threads, "threads_rule": extra["threads_rule"],
                   "estimator_check": pins.estimator, "training": ctx.training, "sequence": cfg["sequence"], "after_stop": cfg["after_stop"],
                   "models": [public_model(r) for r in recs], "one_class": one_class_rows(done2, patience), "dev_frames": frames,
                   "checks": checks, "code_sha256": pins.code, "inputs_sha256": {**pins.files, "configuration": cfg_sha},
                   "peak_rss_mb": round(SR.peak_rss_mb(), 1), "test_read": TEST_READ, "command": command}
            texts = {"models.csv": csv_text(MODEL_COLUMNS, [model_row(r, a.run, patience) for r in recs]),
                     "one_class.csv": csv_text(ONE_CLASS_COLUMNS, [[cell(o[k]) for k in ONE_CLASS_COLUMNS] for o in doc["one_class"]]),
                     "summary.json": PAC.json_text(doc, "summary.json"), "report.md": report_md(doc)}
            pub_sha = write_public(run, inp, texts)
        print(f"run folder: {run.name}, elapsed {wall:.0f} s ({wall / 3600:.2f} h of the {cap_h:.2f}-hour cap), process CPU {cpu_total:.0f} s")
        if mismatches:
            print(f"== MISMATCH: the scorer of phase 1 and the evaluator of phase 2 disagree on the ordinary dev for {len(mismatches)} models; "
                  "no public result is written", flush=True)
        print(f"RESULT: {status} (run {a.run})")
        warnings.showwarning = saved_warning
        FR.close_log(log)
        log = None
        if probs := PAC.output_problems((run / LOG).read_text(encoding="utf-8", errors="replace"), inp):
            fail(f"{LOG} holds text that a publishable output must not hold: " + "; ".join(probs))
        write_status(run, {**head, "status": status, "passed": status == "PASS", "started": started, "finished": finished,
                              "wall_seconds": round(wall, 1), "process_cpu_seconds": cpu_total, **extra, "stopped": stopped,
                              "complete_phase_1": [r["name"] for r in done1], "complete_phase_2": [r["name"] for r in done2],
                              "missing_phase_1": [r["name"] for r in recs if not r.get("phase_1")],
                              "evaluator_mismatches": mismatches, "dev_frames": frames,
                              "machine": (MACHINE or PAC.machine)(), "code_sha256": pins.code, "inputs_sha256": {**pins.files, "configuration": cfg_sha},
                              "public_sha256sums_sha256": pub_sha, "restricted_sha256sums_sha256": res_sha, "run_log_sha256": sha256_file(run / LOG)}, scan)
    except BaseException as e:
        warnings.showwarning = saved_warning
        cur = ctx.current
        if cur is not None and not cur.get("phase_1") and (run / RESTRICTED).is_dir():
            discard_model(run / RESTRICTED, cur["name"])
        extra["complete_phase_1"] = [r["name"] for r in recs if r.get("phase_1")]
        extra["restricted"] = restricted_note(run / RESTRICTED)
        fail_cleanup(run, list(PUBLIC_OUTPUTS), log, head, started, budget, c0, extra, e, scan)
        raise
    return STATUS_CODE[status]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="The two NUAA runs of the LivenessNet / ResNet-18 comparison (48 trainings, fitting and dev only, CPU), "
                                             "the check of the BN estimator and the check of the inputs")
    ap.add_argument("cmd", choices=["check-inputs", "check-estimator", "run"])
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="configuration JSON (relative to the repository root)")
    ap.add_argument("--input-root", default=None, help="base folder of the input paths of the configuration (default: repository root)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="folder that receives <run-id>/ (official: results/archrun; another folder "
                                                                 "outside the repository only with --unofficial-out-root; never a symbolic link)")
    ap.add_argument("--unofficial-out-root", action="store_true", help="accept an --out-root other than results/archrun; the run is recorded "
                                                                      "as not official (tests of the command)")
    ap.add_argument("--run", default=None, choices=list(RUNS), help="run: the run of the plan")
    ap.add_argument("--estimator-check", default=None, help="run (required) and check-inputs (optional): the folder of a PASS of check-estimator")
    ap.add_argument("--date", default=None, help="YYYYMMDD of the run id (default: today)")
    ap.add_argument("--attempt", type=int, default=1, help="attempt number: the last attempt of the same run in the output root + 1 (1 if none)")
    ap.add_argument("--i-am-authorized", action="store_true", help="required by check-estimator and run")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return {"check-inputs": cmd_check_inputs, "check-estimator": cmd_check_estimator, "run": cmd_run}[a.cmd](a)
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
