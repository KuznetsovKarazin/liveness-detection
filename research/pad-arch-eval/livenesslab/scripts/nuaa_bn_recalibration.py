"""
Controllo concordato sulla saturazione iniziale del dev (C2, 13/11/2026): ristima delle statistiche mobili delle cinque
BatchNormalization dei 36 checkpoint salvati dal run 20261106-SESSABLRUN-d999458 (pazienza 4) sul fitting del loro
braccio e fold, e nuova valutazione del dev e del fitting prima e dopo. Nessun training, nessun peso addestrato
modificato, nessun file .keras riscritto, nessun manifest del ruolo test dei fold, punteggio o metrica del test letto
(le liste ufficiali NUAA official_test e cnn_test dei manifest M2 servono solo a indicizzare la cache; l'aggregato
pubblico del genitore, che contiene metriche del test, è letto solo come byte per lo SHA-256, mai decodificato).

Fasi, in un solo processo e in quest'ordine (ognuna con il tempo impiegato):
  0. check-inputs   configurazione a insieme chiuso (tolleranze, regola di arresto, criterio, gruppi, regola di lettura,
                    uscite, tetto, SHA-256 della nota rivista: null rifiutato); configurazione del run genitore (SHA-256
                    fissato, validata dal suo runner); run genitore PASS con i due SHA256SUMS ancorati; per ognuno dei 36
                    modelli scheda, punteggi del dev, soglia e .keras uguali alle righe del SHA256SUMS riservato, SHA-256
                    dei pesi uguale alla scheda; aggregato pubblico del genitore solo come byte contro il suo SHA256SUMS;
                    manifest M2, cache e run delle celle (solo i manifest dei bracci e del dev comune) riusati dal runner
                    genitore; dal run dei fold solo status, SHA256SUMS riservato ed extracted_frames.csv (nessun suo
                    file pubblico aperto); frame estratti con lo SHA-256 di ogni frame; chiavi del fitting e del dev uguali a fitting_keys_sha256 e dev_keys_sha256 della
                    scheda; gruppi (i dieci, i controlli, il modello a parte) derivati dalle schede e uguali alla
                    configurazione. I manifest del test dello schema dei fold non sono mai aperti (questo runner non
                    carica lo schema) e i *_test_scores.csv del genitore nemmeno (parent_path li rifiuta). Nulla scritto,
                    TensorFlow non importato.
  1. baseline       per ognuno dei 36 modelli, PRIMA di qualunque ricalibrazione: SHA-256 verificato subito prima di
                    ogni caricamento; probabilità del dev con lo scorer del genitore (FR.tf_scorer) e con il valutatore
                    di questo script (stesso caricamento, CPU, batch 16, frame/255, training=False; logit dell'ingresso
                    dello strato output come SR.tf_logits); confronti con le tolleranze della configurazione: punteggi
                    salvati del dev (scarto massimo, decisioni discordanti a 0,5), val_loss dai logit contro
                    best_val_loss (solo assoluto; lo scarto da val_loss_recomputed è riportato), accuratezza all'argmax contro
                    val_acc_at_best, softmax dei logit contro le probabilità. Valutazione del fitting con le statistiche
                    originali (senza riferimento). Se un solo controllo di un solo modello non passa: public/baseline.csv
                    con le discrepanze, status BASELINE_FAILED (codice 3), nessuna ricalibrazione calcolata o scritta.
                    Un errore che impedisce la valutazione (architettura inattesa, valori non finiti, SHA-256 diverso) è
                    invece FAIL, con la ragione in status.json e nessuna tabella.
  2. recalibration  solo con la baseline superata per tutti i 36: per ogni modello, su una copia in memoria
                    (clone_model + set_weights; il file non è mai aperto in scrittura), per ogni BatchNormalization
                    nell'ordine della rete l'ingresso dello strato su tutto il fitting in inferenza (dropout spento,
                    nessuna augmentation, gli strati precedenti già ristimati), momenti globali per canale in float64
                    (finiti, altrimenti FAIL)
                    (conteggio frame × H × W per gli strati convoluzionali, frame per il denso; ultimo batch con il suo
                    conteggio; varianza di popolazione dall'unione esatta dei batch, non la media delle varianze dei
                    batch), cast a float32 e assegnazione dei soli moving_mean e moving_variance. Poi: confronto di ogni
                    tensore della copia con l'originale (esattamente i 10 tensori mobili diversi, ogni altro identico sui
                    byte); un passaggio a più uscite sul fitting con la copia finale (controllo di coerenza: i momenti
                    degli ingressi delle cinque BN uguali a quelli installati, entro la tolleranza della configurazione;
                    altrimenti FAIL) che dà anche accuratezza e loss del fitting dopo; dev dopo (stesso valutatore).
  3. aggregate      results.csv (tutti i 36), aggregate.json e aggregate.md con la regola di lettura della
                    configurazione applicata meccanicamente, run_record.json, SHA256SUMS; SHA-256 di ogni .keras e del
                    SHA256SUMS riservato del genitore riverificati alla fine.
Esiti: PASS (codice 0), BASELINE_FAILED (3), INCOMPLETE (tetto o SIGUSR1, 2: uscite dei modelli completi, nessuna
lettura), FAIL (errore, SIGTERM o SIGHUP, 1: ogni uscita tolta). Albero git non pulito o cartella del run esistente:
rifiuto prima di scrivere.

Uscite (cartella del run results/bnrecal/YYYYMMDD-BNRECAL-<commit>; --out-root diverso solo per i test):
  status.json, run.log
  public/      results.csv, baseline.csv, aggregate.json, aggregate.md, run_record.json, SHA256SUMS (nessun identificativo)
  restricted/  <braccio>_fold<k>_seed<s>_scores.csv (dev e fitting, prima e dopo, colonna statistics), _bn_statistics.json
               (statistiche vecchie, stimate in float64, installate in float32 e ricontrollate, per canale; con esse la
               copia ricalibrata si ricostruisce dal .keras del genitore: nessun .keras salvato qui), run_record.json,
               public_SHA256SUMS.sha256, SHA256SUMS

  .venv/bin/python -I -B scripts/nuaa_bn_recalibration.py check-inputs
  .venv/bin/python -I -B scripts/nuaa_bn_recalibration.py run --i-am-authorized --config delivery/bnrecal/nuaa_bn_recalibration.json

Riuso per importazione: nuaa_session_ablation_run (configurazione del genitore, run delle celle, frame estratti, sorgenti, tf_logits come modello dei logit, val_loss_logits, softmax, score_rows, Budget,
segnali, stato git), nuaa_folds_run (manifest M2, cache, FrameStore/assemble, tf_scorer, probabilities, auroc,
hide_gpu, scrittura dello stato, log, controllo del pubblico), m2_writers e path_guard (scritture atomiche, cartelle).
Locale, non importato: load_folds_frames_rows, la parte di SR.load_folds_run che ancora extracted_frames.csv, senza
decodificare l'aggregato pubblico del run dei fold (metriche del test) né il suo run_record.
Indicatori di peggioramento: ogni valore prima e dopo deve essere finito (con AUROC definita), altrimenti FAIL; nessun
indicatore è calcolato da un valore non finito.
Dipendenze: NumPy; TensorFlow 2.15 (fasi 1 e 2, solo CPU).
"""
import argparse
import gc
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import m2_writers as mw  # noqa: E402
import nuaa_folds_run as FR  # noqa: E402
import nuaa_session_ablation_run as SR  # noqa: E402
import path_guard as pg  # noqa: E402

Failure, fail = FR.Failure, FR.fail
sha256_bytes, sha256_file, read_regular, rel_path, parse_csv, csv_text = (FR.sha256_bytes, FR.sha256_file, FR.read_regular,
                                                                          FR.rel_path, FR.parse_csv, FR.csv_text)
now, fmt = FR.now, FR.fmt

DEFAULT_CONFIG = "delivery/bnrecal/nuaa_bn_recalibration.json"
DEFAULT_OUT_ROOT = "results/bnrecal"
CONFIG_SCHEMA = "nuaa-bn-recalibration-config/1"
RUN_ID_RE = re.compile(r"^\d{8}-BNRECAL-[0-9a-f]{7,40}$")
TOP_KEYS = ("schema", "task", "authorisation", "parent_run", "architecture", "batchnorm", "estimator", "evaluation", "tolerances",
            "stop_rule", "recovery_criterion", "groups", "reading_rule", "outputs", "budget", "environment", "note_path",
            "note_sha256", "test")
TOLERANCE_KEYS = ("dev_probability_max_abs_diff", "dev_decisions_discordant_at_0_5", "val_loss_abs", "dev_accuracy_abs",
                  "softmax_of_logits_max_abs_diff", "consistency_rel", "consistency_abs")
ROLES = ("fitting", "dev")                  # gli unici ruoli che questo runner assembla
PARENT_SUFFIXES = ("card.json", "dev_scores.csv", "threshold.json", "LivenessNet.keras")   # mai test_scores.csv
LAYER_CLASSES = ("InputLayer", "Conv2D", "Activation", "BatchNormalization", "MaxPooling2D", "Dropout", "Flatten", "Dense")
MOVING = ("moving_mean", "moving_variance")
STATISTICS = ("moving", "re-estimated")
CELLS = ("fitting_met_dev_met", "fitting_met_dev_not_met", "fitting_not_met_dev_met", "fitting_not_met_dev_not_met")
CASES = ("all", "none", "some")
GROUPS = ("ten", "control", "apart")
FLAGS = ("gained_criterion", "lost_criterion", "dev_accuracy_lower", "val_loss_higher", "dev_auroc_lower", "fitting_accuracy_lower",
         "fitting_loss_higher")
STATUS_CODE = {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2, "BASELINE_FAILED": 3}
STATUS, LOG, PUBLIC, RESTRICTED = FR.STATUS, FR.LOG, "public", "restricted"
PUBLIC_OUTPUTS = ("results.csv", "baseline.csv", "aggregate.json", "aggregate.md", "run_record.json", "SHA256SUMS")
PER_MODEL = ("scores.csv", "bn_statistics.json")
RESTRICTED_FIXED = ("run_record.json", "public_SHA256SUMS.sha256", "SHA256SUMS")
SCORE_COLUMNS = SR.SCORE_COLUMNS + ("statistics", "logit_bona_fide", "logit_attack")
EXTRA_PROTECTED = SR.EXTRA_PROTECTED + ("results/sessionablationrun",)
FAIL_DOC = FR.FAIL_DOC
TEST_READ = ("the fold schema (fold<k>_test.csv, the test role of our folds) is never opened and no <model>_test_scores.csv is "
             "opened; the M2 manifests official_test and cnn_test (NUAA's official image lists, not the test role of our "
             "folds) are read only to index the cache; as in the parent run the frame files (cache and extracted frames) are "
             "hashed and loaded whole, but only fitting and dev frames are assembled; the public aggregate of the parent run, "
             "which holds test metrics, is read only as bytes for its SHA-256 and never parsed; no public file of the folds "
             "run is opened; no test frame is assembled or evaluated and no test score or metric is read, computed or written")
EVAL_RECORD = ("p", "z", "correct", "accuracy", "accuracy_05", "val_loss", "auroc", "softmax_gap", "n")

# punti di iniezione (solo i test): stato git, scorer del genitore, caricamento dei modelli
REPO_STATE = None       # (root) -> {"commit", "dirty"} (default: SR.repo_state)
MAKE_SCORER = None      # (weights_path, sha256) -> funzione(X uint8) -> probabilità N x 2 (default: FR.tf_scorer)
LOAD_MODEL = None       # (weights_path, sha256) -> modello Keras (default: load_keras)


# ============================================================================ configurazione

def _text(v, what: str) -> None:
    if not isinstance(v, str) or not v.strip():
        fail(f"configuration: {what} must be a non-empty text")


def _keys(d, keys, what: str) -> None:
    if not isinstance(d, dict) or sorted(d) != sorted(keys):
        fail(f"configuration: {what} needs exactly {', '.join(keys)}")


def _number(v, what: str, positive: bool = True) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or (v <= 0 if positive else v < 0):
        fail(f"configuration: {what} must be a finite {'positive' if positive else 'non-negative'} number")
    return float(v)


def _names(v, what: str) -> list:
    if not isinstance(v, list) or not all(isinstance(x, str) and re.fullmatch(r"arm_(b|a_star|b_prime)_fold\d+_seed\d+", x) for x in v) \
            or len(set(v)) != len(v):
        fail(f"configuration: {what} must list distinct model names <arm>_fold<k>_seed<s>")
    return v


def load_config(p: Path) -> tuple:
    """(configurazione, SHA-256 dei byte letti). Insiemi chiusi, tipi e testi obbligatori controllati qui, prima di tutto;
    i valori numerici (tolleranze, criterio, tetto) sono letti dalla configurazione, non fissati nel codice."""
    data = read_regular(p, "configuration")
    cfg = SR.read_json(data, f"configuration {p.name}")
    if cfg.get("schema") != CONFIG_SCHEMA:
        fail(f"configuration {p.name}: schema must be {CONFIG_SCHEMA!r}")
    if sorted(cfg) != sorted(TOP_KEYS):
        fail(f"configuration: fields must be exactly {', '.join(TOP_KEYS)} (missing {sorted(set(TOP_KEYS) - set(cfg))}, "
             f"unknown {sorted(set(cfg) - set(TOP_KEYS))})")
    for k in ("task", "authorisation", "stop_rule", "note_path"):
        _text(cfg[k], k)
    if cfg["test"] != TEST_READ:
        fail("configuration: test must state exactly what is read of the test (TEST_READ of the script)")
    if "BASELINE_FAILED" not in cfg["stop_rule"]:
        fail("configuration: stop_rule must declare the BASELINE_FAILED outcome")
    if cfg["note_sha256"] is not None and not SR._hex(cfg["note_sha256"]):
        fail("configuration: note_sha256 must be null or a SHA-256")
    pr = cfg["parent_run"]
    _keys(pr, ["config_path", "config_sha256", "folder", "models", "public_sha256sums_sha256", "restricted_sha256sums_sha256", "run_id",
               "used"], "parent_run")
    for k in ("run_id", "folder", "config_path", "used"):
        _text(pr[k], f"parent_run.{k}")
    if not all(SR._hex(pr[k]) for k in ("config_sha256", "public_sha256sums_sha256", "restricted_sha256sums_sha256")):
        fail("configuration: parent_run SHA-256 values malformed")
    if not FR.is_count(pr["models"]) or pr["models"] < 1:
        fail("configuration: parent_run.models must be a positive count")
    ar = cfg["architecture"]
    _keys(ar, ["augmentation", "input_shape", "layers", "name", "path", "sha256"], "architecture")
    if ar["name"] != "LivenessNet" or not SR._hex(ar["sha256"]):
        fail("configuration: architecture needs name LivenessNet and a SHA-256")
    _text(ar["path"], "architecture.path")
    _text(ar["augmentation"], "architecture.augmentation")
    if not isinstance(ar["input_shape"], list) or len(ar["input_shape"]) != 3 or not all(FR.is_count(x) and x > 0 for x in ar["input_shape"]):
        fail("configuration: architecture.input_shape must be [height, width, channels]")
    lay = ar["layers"]
    if not isinstance(lay, list) or not lay or not all(isinstance(x, list) and len(x) == 2 and all(isinstance(y, str) for y in x) for x in lay) \
            or len({x[0] for x in lay}) != len(lay) or any(x[1] not in LAYER_CLASSES for x in lay):
        fail(f"configuration: architecture.layers must list [name, class] pairs with distinct names and classes in {', '.join(LAYER_CLASSES)}")
    bn = cfg["batchnorm"]
    _keys(bn, ["epsilon", "inference", "kinds", "layers", "momentum", "running_variance_in_training"], "batchnorm")
    order = bn["layers"]
    declared_bn = [n for n, c in lay if c == "BatchNormalization"]
    if not isinstance(order, list) or order != declared_bn or not order:
        fail("configuration: batchnorm.layers must list the BatchNormalization layers of architecture.layers in network order")
    if not isinstance(bn["kinds"], dict) or list(bn["kinds"]) != order or any(v not in ("conv", "dense") for v in bn["kinds"].values()):
        fail("configuration: batchnorm.kinds must give conv or dense for every BatchNormalization layer, in order")
    _number(bn["momentum"], "batchnorm.momentum")
    _number(bn["epsilon"], "batchnorm.epsilon")
    _text(bn["inference"], "batchnorm.inference")
    _text(bn["running_variance_in_training"], "batchnorm.running_variance_in_training")
    es = cfg["estimator"]
    _keys(es, ["cast", "changed_tensors", "consistency", "copy", "counts", "mode", "order", "statistics", "variance"], "estimator")
    for k in es:
        if k != "changed_tensors":
            _text(es[k], f"estimator.{k}")
    if es["changed_tensors"] != list(MOVING):
        fail(f"configuration: estimator.changed_tensors must be {list(MOVING)}")
    ev = cfg["evaluation"]
    _keys(ev, ["accuracy", "auroc", "batch", "decision", "device", "input", "logits", "mode", "probability", "val_loss"], "evaluation")
    if ev["device"] != "CPU" or not FR.is_count(ev["batch"]) or ev["batch"] < 1:
        fail("configuration: evaluation must be on the CPU with a positive batch")
    for k in ev:
        if k not in ("batch", "device"):
            _text(ev[k], f"evaluation.{k}")
    tol = cfg["tolerances"]
    _keys(tol, list(TOLERANCE_KEYS), "tolerances")
    for k in TOLERANCE_KEYS:
        _keys(tol[k], ["source", "value"], f"tolerances.{k}")
        _text(tol[k]["source"], f"tolerances.{k}.source")
        if k == "dev_decisions_discordant_at_0_5":
            if not FR.is_count(tol[k]["value"]):
                fail("configuration: tolerances.dev_decisions_discordant_at_0_5.value must be a non-negative integer")
        else:
            _number(tol[k]["value"], f"tolerances.{k}.value")
    rc = cfg["recovery_criterion"]
    _keys(rc, ["accuracy_min", "accuracy_min_inclusive", "source", "text", "val_loss_max", "val_loss_max_inclusive"], "recovery_criterion")
    _number(rc["accuracy_min"], "recovery_criterion.accuracy_min")
    _number(rc["val_loss_max"], "recovery_criterion.val_loss_max")
    if not isinstance(rc["accuracy_min_inclusive"], bool) or not isinstance(rc["val_loss_max_inclusive"], bool):
        fail("configuration: recovery_criterion inclusive flags must be true or false")
    _text(rc["text"], "recovery_criterion.text")
    _text(rc["source"], "recovery_criterion.source")
    gr = cfg["groups"]
    _keys(gr, ["apart", "control", "degenerate", "ten"], "groups")
    _keys(gr["degenerate"], ["dev_accuracy", "restored_epochs", "source", "within"], "groups.degenerate")
    if not isinstance(gr["degenerate"]["restored_epochs"], list) or not gr["degenerate"]["restored_epochs"] \
            or not all(FR.is_count(x) and x > 0 for x in gr["degenerate"]["restored_epochs"]):
        fail("configuration: groups.degenerate.restored_epochs must list positive epochs")
    _number(gr["degenerate"]["dev_accuracy"], "groups.degenerate.dev_accuracy")
    _number(gr["degenerate"]["within"], "groups.degenerate.within")
    _text(gr["degenerate"]["source"], "groups.degenerate.source")
    for g in ("ten", "apart"):
        _keys(gr[g], ["definition", "expected"], f"groups.{g}")
        _text(gr[g]["definition"], f"groups.{g}.definition")
        _names(gr[g]["expected"], f"groups.{g}.expected")
    _keys(gr["control"], ["definition", "expected_count"], "groups.control")
    _text(gr["control"]["definition"], "groups.control.definition")
    if not FR.is_count(gr["control"]["expected_count"]):
        fail("configuration: groups.control.expected_count must be a non-negative integer")
    if set(gr["ten"]["expected"]) & set(gr["apart"]["expected"]):
        fail("configuration: a model is listed both among the ten and apart")
    rr = cfg["reading_rule"]
    _keys(rr, ["apart", "cells", "cells_header", "controls", "limits", "primary", "report", "worsening", *CASES], "reading_rule")
    for k in rr:
        if k != "cells":
            _text(rr[k], f"reading_rule.{k}")
    for k in CASES:
        for ph in ("{k}", "{n}"):
            if ph not in rr[k]:
                fail(f"configuration: reading_rule.{k} must contain the placeholders {{k}} and {{n}}")
    _keys(rr["cells"], list(CELLS), "reading_rule.cells")
    for k in CELLS:
        _text(rr["cells"][k], f"reading_rule.cells.{k}")
    banned = [w for w in ("attributable", "a reading against", "fails", "penalise", "caused by", "because") if w in json.dumps(rr).lower()]
    if banned:
        fail(f"configuration: the reading rule must not use causal wording ({', '.join(banned)})")
    ou = cfg["outputs"]
    _keys(ou, ["public", "restricted"], "outputs")
    if ou["public"] != list(PUBLIC_OUTPUTS):
        fail(f"configuration: outputs.public must be {list(PUBLIC_OUTPUTS)}")
    if ou["restricted"] != [f"<model>_{x}" for x in PER_MODEL] + list(RESTRICTED_FIXED):
        fail("configuration: outputs.restricted must list the per-model files and the fixed restricted files of this script")
    bu = cfg["budget"]
    _keys(bu, ["estimate", "estimate_minutes", "on_cap_exceeded", "upper_estimate_minutes", "voluntary_stop", "wall_time_cap_minutes"], "budget")
    if not FR.is_count(bu["wall_time_cap_minutes"]) or bu["wall_time_cap_minutes"] < 1:
        fail("configuration: budget.wall_time_cap_minutes must be a positive integer")
    _number(bu["estimate_minutes"], "budget.estimate_minutes")
    _number(bu["upper_estimate_minutes"], "budget.upper_estimate_minutes")
    if not bu["estimate_minutes"] <= bu["upper_estimate_minutes"]:
        fail("configuration: budget.estimate_minutes must not exceed budget.upper_estimate_minutes")
    for k in ("estimate", "on_cap_exceeded", "voluntary_stop"):
        _text(bu[k], f"budget.{k}")
    if "INCOMPLETE" not in bu["on_cap_exceeded"] or "SIGUSR1" not in bu["voluntary_stop"]:
        fail("configuration: budget must declare the INCOMPLETE outcome and the SIGUSR1 stop")
    _keys(cfg["environment"], ["tensorflow"], "environment")
    _text(cfg["environment"]["tensorflow"], "environment.tensorflow")
    return cfg, sha256_bytes(data)


def tol(cfg: dict, key: str):
    return cfg["tolerances"][key]["value"]


# ============================================================================ ingressi (fase 0)

class Inputs(SR.Inputs):
    """Gli ingressi del runner genitore senza lo schema dei fold (folds resta vuoto: nessun manifest del test aperto) più
    il run genitore: schede, punteggi del dev, soglie e pesi dei 36 modelli."""

    def __init__(self):
        super().__init__()
        self.parent_cfg = None
        self.parent = {}         # nome -> {"card", "dev_rows", "threshold", "weights", "arm", "fold", "seed"}
        self.order = []
        self.groups = {}
        self.parent_sums_sha = None
        self.note_sha = None


def parent_path(run: Path, name: str, suffix: str) -> Path:
    """Percorso di un file per modello del run genitore: solo scheda, punteggi del dev, soglia e pesi; ogni altro nome (in
    particolare _test_scores.csv) è rifiutato prima di aprire qualunque file."""
    if suffix not in PARENT_SUFFIXES:
        fail(f"parent file {name}_{suffix} refused: this runner opens only {', '.join(PARENT_SUFFIXES)} of the parent run (never the test scores)")
    return run / RESTRICTED / f"{name}_{suffix}"


def role_rows(inp: Inputs, role: str, arm: str, fold: int) -> list:
    """Righe del manifest di un ruolo: fitting del braccio e fold o dev comune del fold; nessun altro ruolo (il test no)."""
    if role not in ROLES:
        fail(f"role {role!r} refused: this runner assembles only the fitting and dev frames")
    return inp.cells[(arm, fold) if role == "fitting" else ("dev", fold)]


def frames(store, inp: Inputs, role: str, arm: str, fold: int) -> dict:
    return FR.assemble(store, role_rows(inp, role, arm, fold))


def keys_sha(rows) -> str:
    """Impronta delle chiavi come nel runner genitore (righe ordinate per chiave, unite da a capo)."""
    return sha256_bytes("\n".join(sorted(r["key"] for r in rows)).encode())


def meets(acc: float, loss: float, rc: dict) -> bool:
    """Criterio di recupero della configurazione (estremi come dichiarati: accuratezza >= minimo, loss < massimo)."""
    a_ok = acc >= rc["accuracy_min"] if rc["accuracy_min_inclusive"] else acc > rc["accuracy_min"]
    l_ok = loss <= rc["val_loss_max"] if rc["val_loss_max_inclusive"] else loss < rc["val_loss_max"]
    return bool(a_ok and l_ok)


def group_of(card: dict, cfg: dict) -> str:
    """Gruppo dalla scheda del genitore: degenere (i dieci) se l'epoca ripristinata è fra quelle dichiarate e
    l'accuratezza del dev vale 0,5 entro la tolleranza; altrimenti a parte se fuori dal criterio al checkpoint; controllo
    altrimenti."""
    dg = cfg["groups"]["degenerate"]
    if card["restored_epoch"] in dg["restored_epochs"] and abs(card["val_acc_at_best"] - dg["dev_accuracy"]) <= dg["within"]:
        return "ten"
    if not meets(card["val_acc_at_best"], card["best_val_loss"], cfg["recovery_criterion"]):
        return "apart"
    return "control"


def check_groups(cfg: dict, inp: Inputs) -> None:
    got = {g: [n for n in inp.order if inp.groups[n] == g] for g in GROUPS}
    gr = cfg["groups"]
    if sorted(got["ten"]) != sorted(gr["ten"]["expected"]):
        fail(f"groups: the degenerate models derived from the cards {got['ten']} differ from the configuration {gr['ten']['expected']}")
    if sorted(got["apart"]) != sorted(gr["apart"]["expected"]):
        fail(f"groups: the models outside the criterion derived from the cards {got['apart']} differ from the configuration "
             f"{gr['apart']['expected']}")
    if len(got["control"]) != gr["control"]["expected_count"]:
        fail(f"groups: {len(got['control'])} control models derived from the cards, the configuration expects {gr['control']['expected_count']}")
    inp.checks["groups"] = {g: (len(v) if g == "control" else v) for g, v in got.items()}


def load_parent(cfg: dict, base: Path, inp: Inputs) -> None:
    """Il run genitore: configurazione con lo SHA-256 fissato (validata dal suo runner), status PASS, SHA256SUMS pubblico e
    riservato ancorati, aggregato pubblico solo come byte per lo SHA-256 (mai decodificato), e per ogni modello dell'ordine di training scheda,
    punteggi del dev, soglia e pesi uguali al SHA256SUMS riservato e fra loro."""
    pr = cfg["parent_run"]
    cp = rel_path(base, pr["config_path"])
    cb = read_regular(cp, "parent configuration")
    if sha256_bytes(cb) != pr["config_sha256"]:
        fail(f"parent configuration {cp.name}: SHA-256 {sha256_bytes(cb)} differs from the configuration ({pr['config_sha256'][:12]}...)")
    pcfg, _ = SR.load_config(cp)
    inp.parent_cfg = pcfg
    if pcfg["architecture"]["sha256"] != cfg["architecture"]["sha256"] or pcfg["architecture"]["path"] != cfg["architecture"]["path"]:
        fail("parent configuration: the architecture differs from this configuration")
    if pcfg["scoring"]["batch"] != cfg["evaluation"]["batch"] or pcfg["scoring"]["device"] != cfg["evaluation"]["device"]:
        fail("parent configuration: the scoring batch or device differ from this evaluation")
    run = rel_path(base, pr["folder"])
    st = SR.read_json(read_regular(run / STATUS, "parent status.json"), "parent status.json")
    if st.get("status") != "PASS" or st.get("run_id") != pr["run_id"] or st.get("config_sha256") != pr["config_sha256"] \
            or st.get("public_sums_sha256") != pr["public_sha256sums_sha256"] or st.get("restricted_sums_sha256") != pr["restricted_sha256sums_sha256"]:
        fail(f"parent run {pr['run_id']}: status.json is not PASS with the run id, configuration and SHA256SUMS of this configuration")
    order = [SR.model_name(*x) for x in SR.training_order(pcfg["schema_expected"]["k"], pcfg["train_seeds"])]
    if pcfg["training_order"] != order or st.get("completed") != order or st.get("missing") != [] or len(order) != pr["models"]:
        fail(f"parent run {pr['run_id']}: the completed models differ from the training order of its configuration ({pr['models']} expected)")
    sums = {}
    for sub, key in ((PUBLIC, "public_sha256sums_sha256"), (RESTRICTED, "restricted_sha256sums_sha256")):
        b = read_regular(run / sub / "SHA256SUMS", f"parent {sub} SHA256SUMS")
        if sha256_bytes(b) != pr[key]:
            fail(f"parent run {pr['run_id']}: {sub} SHA256SUMS has SHA-256 {sha256_bytes(b)}, expected {pr[key]}")
        sums[sub] = FR.read_sums(b, f"parent {sub} SHA256SUMS")
    if "aggregate.json" not in sums[PUBLIC]:
        fail("parent public SHA256SUMS: aggregate.json not listed")
    # l'aggregato pubblico contiene metriche del test: letto solo come byte per l'hash, mai decodificato
    ab = read_regular(run / PUBLIC / "aggregate.json", "parent aggregate.json")
    if sha256_bytes(ab) != sums[PUBLIC]["aggregate.json"]:
        fail("parent aggregate.json: SHA-256 differs from its SHA256SUMS")
    del ab
    inp.parent_sums_sha = pr["restricted_sha256sums_sha256"]
    inp.parent_aggregate_sha = sums[PUBLIC]["aggregate.json"]
    for arm, fold, seed in SR.training_order(pcfg["schema_expected"]["k"], pcfg["train_seeds"]):
        name = SR.model_name(arm, fold, seed)
        files = {}
        for suf in PARENT_SUFFIXES:
            fn = f"{name}_{suf}"
            if fn not in sums[RESTRICTED]:
                fail(f"parent restricted SHA256SUMS: {fn} not listed")
            p = parent_path(run, name, suf)
            if suf == "LivenessNet.keras":
                h = sha256_file(FR.regular_path(p, f"parent {fn}"))
            else:
                files[suf] = read_regular(p, f"parent {fn}")
                h = sha256_bytes(files[suf])
            if h != sums[RESTRICTED][fn]:
                fail(f"parent {fn}: SHA-256 {h} differs from the parent restricted SHA256SUMS")
        card = SR.read_json(files["card.json"], f"parent {name}_card.json")
        need = ("arm", "fold", "train_seed", "weights_sha256", "best_val_loss", "val_acc_at_best", "restored_epoch", "val_loss_recomputed",
                "n_fitting", "n_dev", "fitting_keys_sha256", "dev_keys_sha256", "history")
        if any(k not in card for k in need):
            fail(f"parent {name}_card.json: fields {[k for k in need if k not in card]} missing")
        if (card["arm"], card["fold"], card["train_seed"]) != (arm, fold, seed):
            fail(f"parent {name}_card.json: arm, fold or seed differ from the file name")
        w = sums[RESTRICTED][f"{name}_LivenessNet.keras"]
        if card["weights_sha256"] != w:
            fail(f"{name}: weights_sha256 of the card differs from the checkpoint in the restricted SHA256SUMS")
        h = card["history"]
        e = card["restored_epoch"]
        if not isinstance(h, dict) or not isinstance(e, int) or not 1 <= e <= len(h.get("val_loss", [])) \
                or h["val_loss"][e - 1] != card["best_val_loss"] or h.get("val_accuracy", [None] * e)[e - 1] != card["val_acc_at_best"]:
            fail(f"{name}: the card's best val_loss or dev accuracy is not its history at the restored epoch")
        fit_rows, dev_rows = role_rows(inp, "fitting", arm, fold), role_rows(inp, "dev", arm, fold)
        if card["n_fitting"] != len(fit_rows) or card["n_dev"] != len(dev_rows):
            fail(f"{name}: the card reports {card['n_fitting']} fitting and {card['n_dev']} dev frames, the manifests hold "
                 f"{len(fit_rows)} and {len(dev_rows)}")
        if card["fitting_keys_sha256"] != keys_sha(fit_rows) or card["dev_keys_sha256"] != keys_sha(dev_rows):
            fail(f"{name}: the fitting or dev keys of the manifests differ from fitting_keys_sha256 / dev_keys_sha256 of the card")
        thr = SR.read_json(files["threshold.json"], f"parent {name}_threshold.json")
        if thr.get("checkpoint_sha256") != w or not isinstance(thr.get("value"), float):
            fail(f"{name}: the parent threshold does not belong to the checkpoint")
        rows = parse_csv(files["dev_scores.csv"], SR.SCORE_COLUMNS, f"parent {name}_dev_scores.csv")
        want_keys = sorted(r["key"] for r in dev_rows)
        bad = [r for r in rows if (r["arm"], r["fold"], r["train_seed"], r["role"], r["checkpoint_sha256"]) != (arm, str(fold), str(seed), "dev", w)]
        if bad or [r["sample_id"] for r in rows] != want_keys:
            fail(f"parent {name}_dev_scores.csv: rows of another model or role, or keys different from the dev manifest in key order")
        try:
            stored = np.array([float(r["score_attack"]) for r in rows], np.float64)
        except ValueError:
            fail(f"parent {name}_dev_scores.csv: score_attack not numeric")
        if not np.all(np.isfinite(stored)) or np.any(stored < 0) or np.any(stored > 1):
            fail(f"parent {name}_dev_scores.csv: score_attack outside [0, 1]")
        inp.parent[name] = {"arm": arm, "fold": fold, "seed": seed, "card": card, "stored": stored,
                            "threshold": {"value": thr["value"], "sha256": sums[RESTRICTED][f"{name}_threshold.json"]},
                            "weights": run / RESTRICTED / f"{name}_LivenessNet.keras", "weights_sha256": w}
        inp.groups[name] = group_of(card, cfg)
    inp.order = order
    check_groups(cfg, inp)
    inp.checks["parent_run"] = {"run_id": pr["run_id"], "config_sha256": pr["config_sha256"], "public_sha256sums_sha256": pr["public_sha256sums_sha256"],
                                "restricted_sha256sums_sha256": pr["restricted_sha256sums_sha256"], "models": len(order),
                                "per_model_files_checked": list(PARENT_SUFFIXES), "not_opened": "<model>_test_scores.csv",
                                "aggregate_json": "read as bytes and compared with its SHA256SUMS line; never parsed (it holds test metrics)",
                                "keys": "fitting_keys_sha256 and dev_keys_sha256 of every card equal to the manifests"}


def load_folds_frames_rows(pcfg: dict, base: Path, inp: Inputs) -> None:
    """Dal run dei fold solo ciò che serve ai frame estratti: status.json PASS con run id, configurazione e SHA256SUMS
    riservato della configurazione del genitore, SHA256SUMS riservato ancorato, extracted_frames.csv uguale alla sua riga
    e alle liste ufficiali. Al posto di SR.load_folds_run, che decodifica anche l'aggregato pubblico (metriche del test,
    riferimento A) e il run_record (controllo del codice del trainer, che qui non si usa): nessun file pubblico del run dei
    fold è aperto. inp.reference resta None."""
    fr_ = pcfg["folds_run"]
    folder = rel_path(base, fr_["folder"])
    st = SR.read_json(read_regular(folder / STATUS, "folds run status.json"), "folds run status.json")
    if st.get("status") != "PASS" or st.get("run_id") != fr_["run_id"] or st.get("config_sha256") != fr_["config_sha256"] \
            or st.get("restricted_sums_sha256") != fr_["restricted_sha256sums_sha256"]:
        fail(f"folds run {fr_['run_id']}: status.json is not PASS with the run id, configuration and restricted SHA256SUMS of the parent configuration")
    b = read_regular(folder / RESTRICTED / "SHA256SUMS", "folds run restricted SHA256SUMS")
    if sha256_bytes(b) != fr_["restricted_sha256sums_sha256"]:
        fail(f"folds run {fr_['run_id']}: restricted SHA256SUMS has SHA-256 {sha256_bytes(b)}, expected {fr_['restricted_sha256sums_sha256']}")
    listed = FR.read_sums(b, "folds run restricted SHA256SUMS")
    if "extracted_frames.csv" not in listed:
        fail("folds run restricted SHA256SUMS: extracted_frames.csv not listed")
    eb = read_regular(folder / RESTRICTED / "extracted_frames.csv", "folds run extracted_frames.csv")
    if sha256_bytes(eb) != listed["extracted_frames.csv"]:
        fail("folds run extracted_frames.csv: SHA-256 differs from its SHA256SUMS")
    rows = parse_csv(eb, FR.FRAME_COLUMNS, "folds run extracted_frames.csv")
    keys = [r["sample_id"] for r in rows]
    if keys != sorted(keys) or keys != inp.to_extract:
        fail("folds run extracted_frames.csv: its keys are not the frames outside the cache, in key order")
    for r in rows:
        o = inp.official[r["sample_id"]]
        if (r["subject_id"], r["session_id"], r["label"], r["file_sha256"]) != (o["subject"], o["session"], o["label"], o["sha256"]) \
                or not SR._hex(r["frame_sha256"]):
            fail("folds run extracted_frames.csv: a row differs from the official lists or has no frame SHA-256")
    inp.extracted_rows = rows
    inp.reference = None
    inp.checks["folds_run"] = {"run_id": fr_["run_id"], "restricted_sha256sums_sha256": fr_["restricted_sha256sums_sha256"],
                               "extracted_frames_csv_sha256": listed["extracted_frames.csv"],
                               "public_files": "not opened (its aggregate holds test metrics)"}


def load_inputs(cfg: dict, base: Path, need_note: bool = True) -> Inputs:
    """Fase 0. Riusa i caricatori del runner genitore senza lo schema dei fold: manifest M2, cache, run delle celle
    (bracci e dev comune), run dei fold e frame estratti (SHA-256 di ogni frame), sorgenti delle righe; poi il run
    genitore. Nessun modello caricato, TensorFlow non importato."""
    refused = [v for v in FR.ENV_REFUSED if os.environ.get(v, "").strip()]
    if refused:
        fail(f"environment variables {refused} are set: they would redirect the reference code (unset them)")
    inp = Inputs()
    if cfg["note_sha256"] is None:
        if need_note:
            fail("configuration: note_sha256 is null: the revised NOTE.md must be frozen and its SHA-256 written before the run")
    else:
        h = sha256_bytes(read_regular(rel_path(ROOT, cfg["note_path"]), "revised note"))
        if h != cfg["note_sha256"]:
            fail(f"revised note {Path(cfg['note_path']).name}: SHA-256 {h} differs from note_sha256 of the configuration")
        inp.note_sha = h
    pr = cfg["parent_run"]
    cp = rel_path(base, pr["config_path"])
    if sha256_bytes(read_regular(cp, "parent configuration")) != pr["config_sha256"]:
        fail(f"parent configuration {cp.name}: SHA-256 differs from the configuration ({pr['config_sha256'][:12]}...)")
    pcfg, _ = SR.load_config(cp)
    v = SR.fr_view(pcfg)
    FR.load_m2_manifests(v, base, inp)
    cache_keys = {r["key"] for n in FR.CACHE_MANIFESTS for r in inp.rows[n]}
    inp.to_extract = sorted(k for k in inp.official if k not in cache_keys)       # come FR.load_fold_manifests, senza lo schema
    FR.load_cache(v, base, inp)
    SR.load_cells_run(pcfg, base, inp)
    load_folds_frames_rows(pcfg, base, inp)
    SR.load_frames(pcfg, base, inp)
    SR.check_sources(pcfg, inp)              # inp.folds è vuoto: solo le righe dei bracci e del dev comune
    if inp.folds:
        fail("the fold schema was loaded: this runner must not open the test manifests")
    load_parent(cfg, base, inp)
    ap = rel_path(base, cfg["architecture"]["path"])
    h = sha256_bytes(read_regular(ap, "architecture file"))
    if h != cfg["architecture"]["sha256"]:
        fail(f"architecture file {ap.name}: SHA-256 {h} differs from the configuration")
    inp.code = {"architecture": h, **{f: sha256_bytes(read_regular(ROOT / f, "code file")) for f in code_files()}}
    inp.environment = FR.environment()
    if inp.environment["tensorflow_installed"] != cfg["environment"]["tensorflow"]:
        fail(f"TensorFlow {inp.environment['tensorflow_installed']} installed, configuration {cfg['environment']['tensorflow']} "
             "(checked without importing it)")
    inp.checks["environment"] = {**inp.environment, "tensorflow_state": "imported later (phases 1 and 2)" if "tensorflow" in sys.modules
                                 else FR.tf_state()}
    inp.checks["test"] = TEST_READ
    return inp


def code_files() -> tuple:
    return ("scripts/nuaa_bn_recalibration.py",) + SR.CODE_FILES


# ============================================================================ TensorFlow: modelli, momenti, valutazione

def tf_module():
    """TensorFlow solo su CPU (difetto della ReLU su Metal): GPU nascosta prima di ogni uso."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        import tensorflow as tf
    except ModuleNotFoundError as e:
        fail(f"missing Python package {e.name!r}: the recalibration needs TensorFlow")
    FR.hide_gpu(tf)
    return tf


def load_keras(path, sha: str):
    """Caricamento del checkpoint come nello scorer del genitore (compile=False, CPU), con lo SHA-256 verificato subito prima."""
    tf = tf_module()
    if sha256_file(path) != sha:
        fail(f"{Path(path).name}: SHA-256 changed just before loading")
    with tf.device("/CPU:0"):
        return tf.keras.models.load_model(str(path), compile=False)


def load_model(path, sha: str):
    return (LOAD_MODEL or load_keras)(path, sha)


def check_architecture(model, cfg: dict, name: str) -> None:
    """Strati del modello salvato uguali alla configurazione (nomi e classi, in ordine; nessuno strato di augmentation o
    di pre-elaborazione: solo le classi di LAYER_CLASSES); ingresso; BatchNormalization con momentum ed epsilon
    registrati, asse dei canali, gamma e beta presenti."""
    got = [[l.name, type(l).__name__] for l in model.layers]
    if got != cfg["architecture"]["layers"]:
        fail(f"{name}: the layers of the saved model differ from the configuration (augmentation or preprocessing layers are refused)")
    if list(model.input_shape[1:]) != cfg["architecture"]["input_shape"]:
        fail(f"{name}: input shape {model.input_shape[1:]} differs from the configuration")
    bn = cfg["batchnorm"]
    for n in bn["layers"]:
        layer = model.get_layer(n)
        c = layer.get_config()
        rank = len(layer.input_shape)
        axis = c["axis"] if isinstance(c["axis"], list) else [c["axis"]]
        if (rank == 4) != (bn["kinds"][n] == "conv") or axis not in ([rank - 1], [-1]) or not c["center"] or not c["scale"] \
                or c["momentum"] != bn["momentum"] or c["epsilon"] != bn["epsilon"]:
            fail(f"{name}: BatchNormalization {n} differs from the recorded facts (kind, channel axis, gamma/beta, momentum, epsilon)")


class Moments:
    """Momenti globali per canale in float64, aggiornati batch per batch con l'unione esatta (Chan et al.): per un batch
    di n_b valori per canale (frame × H × W per uno strato convoluzionale, frame per il denso) con media m_b e somma dei
    quadrati degli scarti M2_b, n = n_a + n_b, d = m_b − m_a, media = m_a + d·n_b/n, M2 = M2_a + M2_b + d²·n_a·n_b/n.
    Alla fine media e varianza di popolazione M2/n: le stesse di somma/N e somma dei quadrati/N − media² sui valori
    concatenati, cioè Σ_b n_b (var_b + (m_b − media)²)/N, senza la cancellazione numerica della seconda forma. Ogni batch
    pesa per il suo conteggio (anche l'ultimo, più corto); non è la media delle varianze dei batch."""

    def __init__(self):
        self.n, self.mean, self.m2 = 0, None, None

    def add(self, a) -> None:
        x = np.asarray(a, dtype=np.float64)
        x = x.reshape(-1, x.shape[-1])
        nb = x.shape[0]
        if nb == 0:
            return
        mb = x.mean(axis=0)
        m2b = ((x - mb) ** 2).sum(axis=0)
        if self.n == 0:
            self.n, self.mean, self.m2 = nb, mb, m2b
            return
        n = self.n + nb
        d = mb - self.mean
        self.mean = self.mean + d * (nb / n)
        self.m2 = self.m2 + m2b + d * d * (self.n * nb / n)
        self.n = n

    def result(self) -> tuple:
        if self.n == 0:
            fail("moments of an empty set")
        return self.mean, self.m2 / self.n


def forward(model, X, batch: int, layers=(), scores: bool = True) -> dict:
    """Un passaggio in inferenza (training=False: dropout spento, BatchNormalization con le statistiche mobili correnti)
    sui frame /255 in float32, batch per batch: momenti degli ingressi degli strati richiesti e, con scores, probabilità
    dello strato output e logit z = h @ W + b in float64 dall'ingresso dello strato output (come SR.tf_logits)."""
    tf = tf_module()
    outs = [model.get_layer(n).input for n in layers]
    out_layer = model.get_layer("output")
    if scores:
        outs += [out_layer.input, model.output]
    sub = tf.keras.Model(model.input, outs)
    mom = {n: Moments() for n in layers}
    hs, ps = [], []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].astype(np.float32) / 255.0
        with tf.device("/CPU:0"):
            res = sub(xb, training=False)
        res = list(res) if isinstance(res, (list, tuple)) else [res]
        for n, a in zip(layers, res):
            mom[n].add(np.asarray(a))
        if scores:
            hs.append(np.asarray(res[-2], dtype=np.float64))
            ps.append(np.asarray(res[-1], dtype=np.float64))
    out = {"moments": mom}
    if scores:
        W, b = (np.asarray(v, dtype=np.float64) for v in out_layer.get_weights())
        out["z"] = np.concatenate(hs) @ W + b if hs else np.zeros((0, 2))
        out["p"] = np.concatenate(ps) if ps else np.zeros((0, 2))
    return out


def eval_record(p: np.ndarray, z: np.ndarray, y: np.ndarray, what: str) -> dict:
    """Valori di una valutazione: accuratezza all'argmax delle probabilità (come CategoricalAccuracy di Keras: pareggio =
    indice 0, risultato in float32), accuratezza alla decisione 0,5 sul punteggio d'attacco, val_loss dai logit (SR),
    AUROC del punteggio d'attacco (FR), scarto fra softmax dei logit e probabilità."""
    p = FR.probabilities(lambda X: p, np.zeros(len(y)))
    if z.shape != (len(y), 2) or not np.all(np.isfinite(z)):
        fail(f"{what}: the logits have shape {z.shape} or non-finite values")
    y = np.asarray(y, np.int64)
    correct = int((np.argmax(p, axis=1) == y).sum())
    c05 = int(((p[:, 1] > FR.FIXED_THRESHOLD).astype(np.int64) == y).sum())
    n = len(y)
    return {"p": p, "z": z, "correct": correct, "accuracy": float(np.float32(correct) / np.float32(n)),
            "accuracy_05": float(np.float32(c05) / np.float32(n)), "val_loss": SR.val_loss_logits(z, y),
            "auroc": FR.auroc(p[y == 1, 1], p[y == 0, 1]), "softmax_gap": float(np.max(np.abs(SR.softmax(z) - p))) if n else 0.0, "n": n}


def evaluate(model, part: dict, batch: int, what: str, layers=()) -> tuple:
    """La stessa valutazione per dev e fitting, prima e dopo: (record, momenti degli strati richiesti)."""
    f = forward(model, part["X"], batch, layers=layers, scores=True)
    return eval_record(f["p"], f["z"], part["y"], what), f["moments"]


def snapshot_weights(model) -> dict:
    """Ogni tensore dei pesi per (strato, indice): nome breve, dtype, forma e byte."""
    out = {}
    for layer in model.layers:
        for i, w in enumerate(layer.weights):
            a = np.asarray(w.numpy())
            out[(layer.name, i)] = {"name": w.name.split("/")[-1].split(":")[0], "dtype": a.dtype.str, "shape": a.shape, "bytes": a.tobytes()}
    return out


def moving_keys(model, bn_layers) -> dict:
    """(strato, indice) dei tensori moving_mean e moving_variance di ogni BatchNormalization."""
    out = {}
    for n in bn_layers:
        layer = model.get_layer(n)
        for attr in MOVING:
            var = getattr(layer, attr)
            idx = [i for i, w in enumerate(layer.weights) if w is var]
            if len(idx) != 1:
                fail(f"BatchNormalization {n}: {attr} not found among its weights")
            out[(n, idx[0])] = (n, attr)
    return out


def recalibrate(copy, part: dict, bn_layers, batch: int, budget=None, name: str = "") -> dict:
    """Ristima strato per strato sulla copia: per ogni BatchNormalization, nell'ordine dato, i momenti dell'ingresso sul
    fitting con gli strati precedenti già ristimati, poi l'assegnazione dei soli moving_mean e moving_variance (cast a
    float32). Restituisce per strato le statistiche vecchie, quelle in float64, quelle installate e il conteggio."""
    out = {}
    for n in bn_layers:
        if budget is not None:
            budget.check(f"recalibration {n} of {name}")
        layer = copy.get_layer(n)
        old_mean, old_var = (np.asarray(getattr(layer, a).numpy()).copy() for a in MOVING)
        mom = forward(copy, part["X"], batch, layers=(n,), scores=False)["moments"][n]
        mean, var = mom.result()
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(var))):
            fail(f"{name}: {n}: the re-estimated mean or variance is not finite")
        m32, v32 = mean.astype(np.float32), var.astype(np.float32)
        layer.moving_mean.assign(m32)
        layer.moving_variance.assign(v32)
        spatial = int(np.prod(layer.input_shape[1:-1])) if len(layer.input_shape) > 2 else 1
        if mom.n != len(part["X"]) * spatial:
            fail(f"{name}: {n} accumulated {mom.n} values per channel, expected frames x H x W = {len(part['X'])} x {spatial}")
        out[n] = {"count": mom.n, "frames": len(part["X"]), "spatial": spatial, "old_mean": old_mean, "old_var": old_var,
                  "mean64": mean, "var64": var, "mean32": m32, "var32": v32}
    return out


def tensor_problems(orig: dict, copy: dict, moving: dict, stats: dict) -> list:
    """Confronto della copia con l'originale: stessi tensori; ogni tensore non mobile identico sui byte; ogni tensore
    mobile diverso dall'originale e uguale sui byte al valore installato. Elenco dei problemi (vuoto se tutto torna)."""
    out = []
    if sorted(orig) != sorted(copy):
        return ["the copy does not have the same weight tensors as the original"]
    differ = []
    for k in sorted(orig):
        a, b = orig[k], copy[k]
        if (a["dtype"], a["shape"]) != (b["dtype"], b["shape"]):
            out.append(f"{k[0]}/{a['name']}: dtype or shape changed")
            continue
        if a["bytes"] != b["bytes"]:
            differ.append(k)
        if k in moving:
            n, attr = moving[k]
            want = stats[n]["mean32" if attr == "moving_mean" else "var32"]
            if b["bytes"] != np.asarray(want, np.float32).tobytes():
                out.append(f"{n}/{attr}: the copy does not hold the installed float32 statistic")
    extra = [f"{k[0]}/{orig[k]['name']}" for k in differ if k not in moving]
    if extra:
        out.append("trained tensors changed: " + ", ".join(extra))
    same = [f"{moving[k][0]}/{moving[k][1]}" for k in moving if k not in differ]
    if same:
        out.append("moving tensors equal to the original after the recalibration: " + ", ".join(same))
    if not out and len(differ) != len(moving):
        out.append(f"{len(differ)} tensors differ, {len(moving)} expected")
    return out


def consistency(stats: dict, final: dict, rel: float, abs_: float) -> dict:
    """Per strato: scarto massimo fra i momenti degli ingressi ricalcolati con la copia finale (float64) e le statistiche
    installate (float32), e superamenti di |installato − ricalcolato| <= rel·|ricalcolato| + abs."""
    out = {}
    for n, s in stats.items():
        m, v = final[n].result()
        dm = np.abs(s["mean32"].astype(np.float64) - m)
        dv = np.abs(s["var32"].astype(np.float64) - v)
        bad = int(np.sum(~(dm <= rel * np.abs(m) + abs_)) + np.sum(~(dv <= rel * np.abs(v) + abs_)))     # NaN conta come fuori
        out[n] = {"recomputed_mean": m, "recomputed_var": v, "max_abs_diff_mean": float(dm.max()), "max_abs_diff_var": float(dv.max()),
                  "count": final[n].n, "channels_outside_tolerance": bad}
    return out


def change_summary(s: dict, eps: float) -> dict:
    """Riassunto pubblico del cambiamento di uno strato: min, mediana e massimo sui canali di var nuova / var vecchia e
    di |media nuova − media vecchia| / sqrt(var vecchia + epsilon) (valori installati e originali, float32)."""
    old_v = s["old_var"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = s["var32"].astype(np.float64) / old_v
    shift = np.abs(s["mean32"].astype(np.float64) - s["old_mean"].astype(np.float64)) / np.sqrt(old_v + eps)
    return {f"var_ratio_{k}": float(f(ratio)) for k, f in (("min", np.min), ("median", np.median), ("max", np.max))} | \
           {f"mean_shift_{k}": float(f(shift)) for k, f in (("min", np.min), ("median", np.median), ("max", np.max))}


# ============================================================================ baseline (fase 1)

def baseline_checks(cfg: dict, p_parent: np.ndarray, ev: dict, stored: np.ndarray, card: dict) -> dict:
    """I controlli della baseline di un modello con le tolleranze della configurazione: probabilità d'attacco del dev
    (scorer del genitore e valutatore) contro i punteggi salvati, decisioni discordanti a 0,5, softmax dei logit,
    val_loss dai logit contro best_val_loss (solo assoluto), accuratezza all'argmax contro val_acc_at_best.
    Riportati senza vincolo: scarto da val_loss_recomputed, accuratezza a 0,5, conteggi. pass = tutti i vincolanti."""
    pt, dt = tol(cfg, "dev_probability_max_abs_diff"), tol(cfg, "dev_decisions_discordant_at_0_5")
    d_par = float(np.max(np.abs(p_parent - stored))) if len(stored) else 0.0
    d_ev = float(np.max(np.abs(ev["p"][:, 1] - stored))) if len(stored) else 0.0
    dis_par = int(np.sum((p_parent > FR.FIXED_THRESHOLD) != (stored > FR.FIXED_THRESHOLD)))
    dis_ev = int(np.sum((ev["p"][:, 1] > FR.FIXED_THRESHOLD) != (stored > FR.FIXED_THRESHOLD)))
    va = tol(cfg, "val_loss_abs")
    dvl = abs(ev["val_loss"] - card["best_val_loss"])
    acc_t = tol(cfg, "dev_accuracy_abs")
    dacc = abs(ev["accuracy"] - card["val_acc_at_best"])
    sg = tol(cfg, "softmax_of_logits_max_abs_diff")
    c = {"prob_parent_max_abs_diff": d_par, "prob_parent_discordant_05": dis_par, "prob_eval_max_abs_diff": d_ev,
         "prob_eval_discordant_05": dis_ev, "prob_tolerance": pt, "discordant_tolerance": dt,
         "prob_pass": d_par <= pt and d_ev <= pt and dis_par <= dt and dis_ev <= dt,
         "softmax_gap": ev["softmax_gap"], "softmax_tolerance": sg, "softmax_pass": ev["softmax_gap"] <= sg,
         "val_loss": ev["val_loss"], "val_loss_card_best": card["best_val_loss"], "val_loss_diff": dvl,
         "val_loss_tolerance": f"abs {va!r}", "val_loss_pass": dvl <= va,
         "val_loss_card_recomputed": card["val_loss_recomputed"], "val_loss_diff_recomputed": abs(ev["val_loss"] - card["val_loss_recomputed"]),
         "accuracy": ev["accuracy"], "accuracy_card": card["val_acc_at_best"], "accuracy_diff": dacc, "accuracy_tolerance": acc_t,
         "correct": ev["correct"], "correct_card": int(round(card["val_acc_at_best"] * card["n_dev"])), "accuracy_pass": dacc <= acc_t,
         "accuracy_05": ev["accuracy_05"]}
    c["baseline_pass"] = bool(c["prob_pass"] and c["softmax_pass"] and c["val_loss_pass"] and c["accuracy_pass"])
    return c


BASELINE_COLUMNS = ("arm", "fold", "seed", "n_dev", "prob_parent_max_abs_diff", "prob_parent_discordant_05", "prob_eval_max_abs_diff",
                    "prob_eval_discordant_05", "prob_tolerance", "discordant_tolerance", "prob_pass", "softmax_gap", "softmax_tolerance",
                    "softmax_pass", "val_loss", "val_loss_card_best", "val_loss_diff", "val_loss_tolerance", "val_loss_pass",
                    "val_loss_card_recomputed", "val_loss_diff_recomputed", "accuracy", "accuracy_card", "accuracy_diff", "accuracy_tolerance",
                    "correct", "correct_card", "accuracy_pass", "accuracy_05", "fitting_accuracy", "fitting_loss", "fitting_reference",
                    "baseline_pass")


def cell_text(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return fmt(v)
    return "NA" if v is None else str(v)


def baseline_csv(inp: Inputs, base: dict) -> str:
    rows = []
    for n in inp.order:
        if n not in base:
            continue
        m = inp.parent[n]
        c = {**base[n]["checks"], "arm": m["arm"], "fold": m["fold"], "seed": m["seed"], "n_dev": m["card"]["n_dev"],
             "fitting_accuracy": base[n]["fit"]["accuracy"], "fitting_loss": base[n]["fit"]["val_loss"],
             "fitting_reference": "none (no recorded value in inference mode)"}
        rows.append([cell_text(c[k]) for k in BASELINE_COLUMNS])
    return csv_text(BASELINE_COLUMNS, rows)


# ============================================================================ righe, regola di lettura, aggregato

def cell_of(fit_met: bool, dev_met: bool) -> str:
    return f"fitting_{'met' if fit_met else 'not_met'}_dev_{'met' if dev_met else 'not_met'}"


def result_row(cfg: dict, name: str, m: dict, group: str, before: dict, after: dict, changes: dict, seconds: dict) -> dict:
    """Riga pubblica di un modello: valori prima e dopo, criterio sul dev e sul fitting, cella 2×2 (fitting prima × dev
    dopo), ogni peggioramento come indicatore, riassunti per strato, tempi."""
    rc = cfg["recovery_criterion"]
    db, da, fb, fa = before["dev"], after["dev"], before["fit"], after["fit"]
    dev_b, dev_a = meets(db["accuracy"], db["val_loss"], rc), meets(da["accuracy"], da["val_loss"], rc)
    fit_b, fit_a = meets(fb["accuracy"], fb["val_loss"], rc), meets(fa["accuracy"], fa["val_loss"], rc)
    au_b, au_a = db["auroc"], da["auroc"]
    vals = [db["accuracy"], da["accuracy"], fb["accuracy"], fa["accuracy"], db["val_loss"], da["val_loss"], fb["val_loss"], fa["val_loss"], au_b, au_a]
    if any(v is None or not math.isfinite(v) for v in vals):
        fail(f"{name}: a dev or fitting value before or after is not finite (or the dev AUROC is undefined): no worsening flag may be "
             "computed from it")
    r = {"arm": m["arm"], "fold": m["fold"], "seed": m["seed"], "group": group, "restored_epoch": m["card"]["restored_epoch"],
         "n_fitting": fb["n"], "n_dev": db["n"],
         "dev_accuracy_before": db["accuracy"], "dev_accuracy_after": da["accuracy"], "dev_correct_before": db["correct"],
         "dev_correct_after": da["correct"], "dev_val_loss_before": db["val_loss"], "dev_val_loss_after": da["val_loss"],
         "dev_auroc_before": au_b, "dev_auroc_after": au_a,
         "fitting_accuracy_before": fb["accuracy"], "fitting_accuracy_after": fa["accuracy"], "fitting_loss_before": fb["val_loss"],
         "fitting_loss_after": fa["val_loss"],
         "dev_criterion_before": dev_b, "dev_criterion_after": dev_a, "fitting_criterion_before": fit_b, "fitting_criterion_after": fit_a,
         "cell": cell_of(fit_b, dev_a),
         "gained_criterion": not dev_b and dev_a, "lost_criterion": dev_b and not dev_a,
         "dev_accuracy_lower": da["correct"] < db["correct"], "val_loss_higher": da["val_loss"] > db["val_loss"],
         "dev_auroc_lower": au_a < au_b,
         "fitting_accuracy_lower": fa["correct"] < fb["correct"], "fitting_loss_higher": fa["val_loss"] > fb["val_loss"]}
    for n in cfg["batchnorm"]["layers"]:
        for k, v in changes[n].items():
            r[f"{n}_{k}"] = v
    r.update({f"{k}_seconds": round(v, 1) for k, v in seconds.items()})
    return r


def result_columns(cfg: dict) -> tuple:
    head = ("arm", "fold", "seed", "group", "restored_epoch", "n_fitting", "n_dev", "dev_accuracy_before", "dev_accuracy_after",
            "dev_correct_before", "dev_correct_after", "dev_val_loss_before", "dev_val_loss_after", "dev_auroc_before", "dev_auroc_after",
            "fitting_accuracy_before", "fitting_accuracy_after", "fitting_loss_before", "fitting_loss_after", "dev_criterion_before",
            "dev_criterion_after", "fitting_criterion_before", "fitting_criterion_after", "cell") + FLAGS
    layers = tuple(f"{n}_{k}_{s}" for n in cfg["batchnorm"]["layers"] for k in ("var_ratio", "mean_shift") for s in ("min", "median", "max"))
    return head + layers + ("baseline_seconds", "recalibration_seconds", "evaluation_seconds")


def results_csv(cfg: dict, rows: list) -> str:
    cols = result_columns(cfg)
    return csv_text(cols, [[cell_text(r[c]) for c in cols] for r in rows])


def name_of(r: dict) -> str:
    return SR.model_name(r["arm"], r["fold"], r["seed"])


def aggregate(cfg: dict, rows: list) -> dict:
    """La regola di lettura della configurazione applicata meccanicamente alle righe dei modelli completi."""
    rr = cfg["reading_rule"]
    ten = [r for r in rows if r["group"] == "ten"]
    ctrl = [r for r in rows if r["group"] == "control"]
    apart = [r for r in rows if r["group"] == "apart"]
    k, n = sum(r["dev_criterion_after"] for r in ten), len(ten)
    case = "all" if n and k == n else "none" if k == 0 else "some"
    keep = [name_of(r) for r in ctrl if r["dev_criterion_before"] and r["dev_criterion_after"]]
    lose = [name_of(r) for r in ctrl if r["dev_criterion_before"] and not r["dev_criterion_after"]]
    cells = {g: {c: sum(1 for r in rs if r["cell"] == c) for c in CELLS} for g, rs in (("ten", ten), ("all", rows))}
    worse = [{"model": name_of(r), "group": r["group"], "flags": [f for f in FLAGS if r[f] and f != "gained_criterion"],
              "sizes": {f: worsening_size(r, f) for f in FLAGS if r[f] and f != "gained_criterion"}} for r in rows]
    worse = [w for w in worse if w["flags"]]
    return {"primary": {"rule": rr["primary"], "positives": k, "of": n, "case": case, "statement": rr[case].format(k=k, n=n),
                        "recovered": [name_of(r) for r in ten if r["dev_criterion_after"]],
                        "not_recovered": [name_of(r) for r in ten if not r["dev_criterion_after"]]},
            "controls": {"rule": rr["controls"], "n": len(ctrl), "meeting_before": sum(r["dev_criterion_before"] for r in ctrl),
                         "keep": len(keep), "lose": len(lose), "lost": lose,
                         "not_meeting_before": [name_of(r) for r in ctrl if not r["dev_criterion_before"]]},
            "apart": {"rule": rr["apart"], "models": [{"model": name_of(r), **{x: r[x] for x in (
                "dev_accuracy_before", "dev_accuracy_after", "dev_val_loss_before", "dev_val_loss_after", "dev_auroc_before",
                "dev_auroc_after", "dev_criterion_after")}} for r in apart]},
            "cells": {"counts": cells, "statements": rr["cells"], "header": rr["cells_header"]},
            "worsening": {"rule": rr["worsening"], "models": worse, "counts": {f: sum(1 for r in rows if r[f]) for f in FLAGS}},
            "limits": rr["limits"], "report": rr["report"]}


WORSE_VALUES = {"lost_criterion": ("dev_criterion_before", "dev_criterion_after"), "dev_accuracy_lower": ("dev_correct_before", "dev_correct_after"),
                "val_loss_higher": ("dev_val_loss_before", "dev_val_loss_after"), "dev_auroc_lower": ("dev_auroc_before", "dev_auroc_after"),
                "fitting_accuracy_lower": ("fitting_accuracy_before", "fitting_accuracy_after"),
                "fitting_loss_higher": ("fitting_loss_before", "fitting_loss_after")}


def worsening_size(r: dict, flag: str) -> dict:
    """Entità di un peggioramento: valore prima, dopo e differenza (dopo − prima), per riconoscere il rumore numerico."""
    b, a = (r[k] for k in WORSE_VALUES[flag])
    b, a = (int(x) if isinstance(x, bool) else x for x in (b, a))
    return {"before": b, "after": a, "difference": a - b}


def table(headers, rows) -> str:
    return SR.table(headers, rows)


def aggregate_md(cfg: dict, doc: dict, rows: list) -> str:
    a = doc.get("reading")
    L = [f"# BatchNormalization recalibration of the {doc['models']} patience-4 checkpoints ({doc['run_id']})", "",
         f"Status: **{doc['status']}**. Parent run `{doc['parent_run_id']}`; configuration `{doc['config_sha256'][:12]}`; revised note "
         f"`{(doc['note_sha256'] or 'none')[:12]}`. No training. {doc['test_read']}.", ""]
    if a is None:
        L += [f"Reading not applied: {doc['reading_not_applied']}", ""]
    else:
        p = a["primary"]
        L += ["## Primary reading (fixed before the run)", "", p["rule"], "", f"**{p['statement']}**", "",
              f"Meeting the criterion after: {', '.join(p['recovered']) or 'none'}; not meeting it: {', '.join(p['not_recovered']) or 'none'}.", "",
              "## Controls and the model reported apart", "", a["controls"]["rule"], "",
              f"{a['controls']['keep']} of the {a['controls']['n']} controls keep the criterion, {a['controls']['lose']} lose it"
              + (f" ({', '.join(a['controls']['lost'])})" if a["controls"]["lost"] else "") + ".", "", a["apart"]["rule"], ""]
        for x in a["apart"]["models"]:
            L.append(f"- {x['model']}: dev accuracy {fmt4(x['dev_accuracy_before'])} -> {fmt4(x['dev_accuracy_after'])}, val_loss "
                     f"{fmt4(x['dev_val_loss_before'])} -> {fmt4(x['dev_val_loss_after'])}, AUROC {fmt4(x['dev_auroc_before'])} -> "
                     f"{fmt4(x['dev_auroc_after'])}")
        cc, st = a["cells"]["counts"], a["cells"]["statements"]
        def cell(c):
            return f"{st[c]}: {cc['ten'][c]} of the ten, {cc['all'][c]} of all"
        L += ["", "## Fitting before x dev after (2x2)", "", a["cells"]["header"], "",
              table(["fitting before (original statistics, inference mode)", "dev after: meets the criterion", "dev after: does not meet it"],
                    [["meets the criterion", cell("fitting_met_dev_met"), cell("fitting_met_dev_not_met")],
                     ["does not meet it", cell("fitting_not_met_dev_met"), cell("fitting_not_met_dev_not_met")]]), "",
              "## Worsening (every case listed)", "", a["worsening"]["rule"], "",
              table(["flag", "models"], [[f, a["worsening"]["counts"][f]] for f in FLAGS]), ""]
        L += [f"- {w['model']} ({w['group']}): " + "; ".join(f"{f} {fmt_any(z['before'])} -> {fmt_any(z['after'])} (difference "
                                                            f"{fmt_any(z['difference'])})" for f, z in w["sizes"].items())
              for w in a["worsening"]["models"]] or ["- none"]
        L += ["", "## Limits", "", a["limits"], "", a["report"], ""]
    if rows:
        L += ["## All models", "", table(["model", "group", "epoch", "dev acc", "dev val_loss", "dev AUROC", "fitting acc", "fitting loss",
                                          "dev crit", "cell"],
              [[name_of(r), r["group"], r["restored_epoch"], f"{fmt4(r['dev_accuracy_before'])} -> {fmt4(r['dev_accuracy_after'])}",
                f"{fmt4(r['dev_val_loss_before'])} -> {fmt4(r['dev_val_loss_after'])}", f"{fmt4(r['dev_auroc_before'])} -> {fmt4(r['dev_auroc_after'])}",
                f"{fmt4(r['fitting_accuracy_before'])} -> {fmt4(r['fitting_accuracy_after'])}",
                f"{fmt4(r['fitting_loss_before'])} -> {fmt4(r['fitting_loss_after'])}",
                f"{int(r['dev_criterion_before'])} -> {int(r['dev_criterion_after'])}", r["cell"]] for r in rows]), ""]
    t = doc["timings"]
    L += ["## Time", "", f"Wall time {t['wall_seconds']:.0f} s ({t['wall_seconds'] / 60:.1f} min) against the note's estimate of about "
          f"{t['estimate_central_minutes']} minutes (upper estimate about {t['estimate_upper_minutes']}); cap {t['cap_minutes']} minutes.", "",
          table(["phase", "seconds"], [[k, f"{v:.1f}"] for k, v in t["phases"].items()]), ""]
    return "\n".join(L)


def fmt_any(v) -> str:
    return str(v) if isinstance(v, int) else f"{v:.6g}"


def fmt4(v) -> str:
    return FR.f4(v)


# ============================================================================ cartella del run

def repo_state(root: Path) -> dict:
    return REPO_STATE(root) if REPO_STATE is not None else SR.repo_state(root)


def run_id_for(date: str, state: dict) -> str:
    if not re.fullmatch(r"\d{8}", date or ""):
        fail("--date must be YYYYMMDD")
    rid = f"{date}-BNRECAL-{state['commit']}"
    if not RUN_ID_RE.match(rid):
        fail(f"run id {rid} does not follow YYYYMMDD-BNRECAL-<commit>")
    return rid


def restricted_outputs(order) -> tuple:
    return tuple(f"{n}_{x}" for n in order for x in PER_MODEL) + RESTRICTED_FIXED


def input_paths(cfg: dict, pcfg: dict, base: Path, cfg_p: Path) -> dict:
    out = {"the configuration": cfg_p, "the parent run folder": rel_path(base, cfg["parent_run"]["folder"]),
           "the parent configuration": rel_path(base, cfg["parent_run"]["config_path"]), "the revised note": rel_path(ROOT, cfg["note_path"])}
    out.update({f"manifest {m}": rel_path(base, pcfg["manifests"]["folder"]) / pcfg["manifests"]["files"][m]["file"] for m in FR.MANIFESTS})
    out["the cells run folder"] = rel_path(base, pcfg["cells_run"]["folder"])
    out["the folds run folder"] = rel_path(base, pcfg["folds_run"]["folder"])
    out["the extracted frames file"] = rel_path(base, pcfg["folds_run"]["frames_file"]["path"])
    out["the cache"] = rel_path(base, pcfg["cache"]["path"])
    out["the architecture file"] = rel_path(base, cfg["architecture"]["path"])
    return out


def run_dir_problem(run: Path, root: Path, inputs: dict, order, out_root: Path) -> str:
    """Motivo per cui la cartella del run non è ammessa ("" se lo è), prima di qualunque modifica."""
    if out_root.is_symlink():
        return "the output root is a symbolic link"
    for label, p in inputs.items():
        if pg.within(p, run) or (Path(p).is_dir() and pg.within(run, p)):
            return f"{label} {Path(p).name} is inside the run folder, or contains it: inputs and outputs must be separate"
    extra = [root / x for x in EXTRA_PROTECTED]
    msg = mw.output_problem(run, root, pg.BASE_PROTECTED, [root / DEFAULT_OUT_ROOT], extra, inputs, [STATUS, LOG], [run / PUBLIC, run / RESTRICTED])
    for sub, names in ((PUBLIC, PUBLIC_OUTPUTS), (RESTRICTED, restricted_outputs(order))):
        msg = msg or mw.output_problem(run / sub, root, pg.BASE_PROTECTED, [root / DEFAULT_OUT_ROOT], extra, inputs, list(names))
    return msg


def invalidate_outputs(run: Path, order) -> list:
    left = []
    for sub, names in ((PUBLIC, PUBLIC_OUTPUTS), (RESTRICTED, restricted_outputs(order))):
        d = run / sub
        if d.is_dir():
            left += [f"{sub}/{n}" for n in mw.invalidate_each(d, list(names), None, FAIL_DOC)]
    return left


def resolve(a) -> tuple:
    base = Path(a.input_root).resolve() if a.input_root else ROOT
    cfg_p = Path(a.config) if Path(a.config).is_absolute() else ROOT / a.config
    return base, cfg_p


# ============================================================================ comandi

def cmd_check_inputs(a) -> int:
    base, cfg_p = resolve(a)
    cfg, cfg_sha = load_config(cfg_p)
    inp = load_inputs(cfg, base)
    g = inp.checks["groups"]
    print(f"CHECK-INPUTS: configuration {cfg_sha[:12]}, revised note {inp.note_sha[:12]}, parent run {cfg['parent_run']['run_id']} PASS "
          f"({len(inp.order)} models: card, dev scores, threshold and weights equal to its restricted SHA256SUMS; public "
          f"aggregate hashed only, never parsed; fitting and dev keys equal to every card), {len(FR.MANIFESTS)} M2 manifests, {len(inp.cells_sha)} cell manifests "
          f"(devmix not read), cache {len(inp.cache_index)} frames, extracted frames {len(inp.extracted_keys)} with every frame SHA-256 "
          f"verified, groups: {len(g['ten'])} degenerate, {g['control']} controls, {len(g['apart'])} apart; fold schema and test "
          f"manifests not opened; TensorFlow {inp.environment['tensorflow_installed']} installed and not imported; nothing written")
    print("RESULT: PASS (check-inputs)")
    return 0


def write_public(run: Path, inp: Inputs, public: dict) -> str:
    """Scansione di ogni file pubblico (nessun identificativo, chiave o hash d'immagine), SHA256SUMS, scrittura atomica;
    restituisce lo SHA-256 del SHA256SUMS pubblico."""
    for pname, text in public.items():
        if probs := FR.public_problems(text, inp, pname.endswith(".md")):
            fail(f"public file {pname} refused: " + "; ".join(probs))
    public = {**public, "SHA256SUMS": FR.sums_text(public)}
    mw.write_all(run / PUBLIC, public)
    return sha256_bytes(public["SHA256SUMS"].encode("utf-8"))


def write_restricted_sums(run: Path, names: list) -> str:
    rdir = run / RESTRICTED
    missing = [n for n in names if not (rdir / n).is_file()]
    if missing:
        fail(f"restricted outputs missing: {missing}")
    extra = sorted(e for e in os.listdir(rdir) if e not in names and e != "SHA256SUMS")
    if extra:
        fail(f"restricted folder holds files that are not outputs of this run: {extra}")
    sums = "".join(f"{sha256_file(rdir / n)}  {n}\n" for n in sorted(names))
    mw.write_all(rdir, {"SHA256SUMS": sums})
    return sha256_bytes(sums.encode("utf-8"))


def score_rows(m: dict, role: str, part: dict, rec: dict, statistics: str) -> list:
    rows = SR.score_rows(m["arm"], m["fold"], m["seed"], role, part, rec["p"][:, 1], m["threshold"], m["card"])
    for r, z in zip(rows, rec["z"]):
        r.update({"statistics": statistics, "logit_bona_fide": repr(float(z[0])), "logit_attack": repr(float(z[1]))})
    return rows


def stats_json(name: str, m: dict, stats: dict, cons: dict, changes: dict, cfg: dict) -> str:
    lay = {}
    for n, s in stats.items():
        lay[n] = {"kind": cfg["batchnorm"]["kinds"][n], "frames": s["frames"], "values_per_frame_per_channel": s["spatial"], "count": s["count"],
                  "channels": int(len(s["mean64"])), "old_moving_mean": [float(x) for x in s["old_mean"]],
                  "old_moving_variance": [float(x) for x in s["old_var"]], "mean_float64": [float(x) for x in s["mean64"]],
                  "variance_float64": [float(x) for x in s["var64"]], "installed_moving_mean_float32": [float(x) for x in s["mean32"]],
                  "installed_moving_variance_float32": [float(x) for x in s["var32"]],
                  "consistency": {"recomputed_mean": [float(x) for x in cons[n]["recomputed_mean"]],
                                  "recomputed_variance": [float(x) for x in cons[n]["recomputed_var"]],
                                  "max_abs_diff_mean": cons[n]["max_abs_diff_mean"], "max_abs_diff_variance": cons[n]["max_abs_diff_var"],
                                  "count": cons[n]["count"], "channels_outside_tolerance": cons[n]["channels_outside_tolerance"]},
                  "summary": changes[n]}
    return json.dumps({"model": name, "weights_sha256": m["weights_sha256"], "estimator": cfg["estimator"], "layers": lay,
                       "reproduce": "load the parent .keras checkpoint (SHA-256 above) and assign installed_moving_mean_float32 and "
                                    "installed_moving_variance_float32 to moving_mean and moving_variance of each layer; every other "
                                    "tensor unchanged"}, indent=1) + "\n"


def baseline_phase(cfg: dict, inp: Inputs, store, budget, base: dict) -> None:
    """Fase 1: tutti i modelli prima di qualunque ricalibrazione (base: nome -> valori e controlli)."""
    batch = cfg["evaluation"]["batch"]
    for name in inp.order:
        budget.check(f"baseline {name}")
        t0 = time.time()
        m = inp.parent[name]
        fit, dev = frames(store, inp, "fitting", m["arm"], m["fold"]), frames(store, inp, "dev", m["arm"], m["fold"])
        if dev["keys"] != sorted(r["key"] for r in role_rows(inp, "dev", m["arm"], m["fold"])):
            fail(f"{name}: the dev frames are not in the key order of the stored scores")
        tf_module().keras.backend.clear_session()        # memoria: grafo del modello precedente liberato
        scorer = (MAKE_SCORER or FR.tf_scorer)(m["weights"], m["weights_sha256"])
        p_parent = FR.probabilities(scorer, dev["X"])[:, 1]
        del scorer
        model = load_model(m["weights"], m["weights_sha256"])
        check_architecture(model, cfg, name)
        ev_dev, _ = evaluate(model, dev, batch, f"{name} dev")
        ev_fit, _ = evaluate(model, fit, batch, f"{name} fitting")
        checks = baseline_checks(cfg, p_parent, ev_dev, m["stored"], m["card"])
        base[name] = {"dev": ev_dev, "fit": ev_fit, "checks": checks, "seconds": time.time() - t0}
        del model, fit, dev
        gc.collect()
        print(f"   baseline {name}: {'pass' if checks['baseline_pass'] else 'DISCREPANCY'} (probability {checks['prob_parent_max_abs_diff']:.1e}"
              f"/{checks['prob_eval_max_abs_diff']:.1e}, val_loss diff {checks['val_loss_diff']:.1e}, correct {checks['correct']}/"
              f"{checks['correct_card']}), {base[name]['seconds']:.1f} s", flush=True)


def recalibration_phase(cfg: dict, inp: Inputs, store, budget, base: dict, run: Path, rows: list, completed: list) -> None:
    """Fase 2: ricalibrazione su copia, controlli dei tensori e di coerenza, valutazione dopo, uscite riservate."""
    tf = tf_module()
    batch, bn = cfg["evaluation"]["batch"], cfg["batchnorm"]["layers"]
    rel, abs_ = tol(cfg, "consistency_rel"), tol(cfg, "consistency_abs")
    sg = tol(cfg, "softmax_of_logits_max_abs_diff")
    for name in inp.order:
        budget.check(f"recalibration {name}")
        m = inp.parent[name]
        t0 = time.time()
        fit, dev = frames(store, inp, "fitting", m["arm"], m["fold"]), frames(store, inp, "dev", m["arm"], m["fold"])
        tf.keras.backend.clear_session()
        orig = load_model(m["weights"], m["weights_sha256"])
        check_architecture(orig, cfg, name)
        copy = tf.keras.models.clone_model(orig)
        copy.set_weights(orig.get_weights())
        snap = snapshot_weights(orig)
        if snapshot_weights(copy) != snap:
            fail(f"{name}: the in-memory copy differs from the original before the recalibration")
        stats = recalibrate(copy, fit, bn, batch, budget, name)
        if probs := tensor_problems(snap, snapshot_weights(copy), moving_keys(copy, bn), stats):
            fail(f"{name}: " + "; ".join(probs))
        if snapshot_weights(orig) != snap:
            fail(f"{name}: the original model changed during the recalibration")
        t1 = time.time()
        budget.check(f"evaluation {name}")
        fit_after, final = evaluate(copy, fit, batch, f"{name} fitting after", layers=bn)
        cons = consistency(stats, final, rel, abs_)
        if bad := {n: c["channels_outside_tolerance"] for n, c in cons.items() if c["channels_outside_tolerance"]}:
            fail(f"{name}: consistency check failed: the inputs of {sorted(bad)} recomputed with the recalibrated copy do not have the "
                 f"installed moments within rel {rel!r} + abs {abs_!r} ({bad} channel statistics outside)")
        if any(c["count"] != stats[n]["count"] for n, c in cons.items()):
            fail(f"{name}: the consistency pass counted a different number of values per channel")
        dev_after, _ = evaluate(copy, dev, batch, f"{name} dev after")
        for rec, what in ((fit_after, "fitting"), (dev_after, "dev")):
            if rec["softmax_gap"] > sg:
                fail(f"{name}: after the recalibration the softmax of the logits differs from the {what} probabilities by {rec['softmax_gap']:.2e}")
        changes = {n: change_summary(stats[n], cfg["batchnorm"]["epsilon"]) for n in bn}
        seconds = {"baseline": base[name]["seconds"], "recalibration": t1 - t0, "evaluation": time.time() - t1}
        before = {"dev": base[name]["dev"], "fit": base[name]["fit"]}
        r = result_row(cfg, name, m, inp.groups[name], before, {"dev": dev_after, "fit": fit_after}, changes, seconds)
        srows = (score_rows(m, "dev", dev, base[name]["dev"], STATISTICS[0]) + score_rows(m, "dev", dev, dev_after, STATISTICS[1]) +
                 score_rows(m, "fitting", fit, base[name]["fit"], STATISTICS[0]) + score_rows(m, "fitting", fit, fit_after, STATISTICS[1]))
        mw.write_all(run / RESTRICTED, {f"{name}_scores.csv": csv_text(SCORE_COLUMNS, [[x[c] for c in SCORE_COLUMNS] for x in srows]),
                                        f"{name}_bn_statistics.json": stats_json(name, m, stats, cons, changes, cfg)}, newline="")
        rows.append(r)
        completed.append(name)
        del orig, copy, fit, dev
        gc.collect()
        print(f"== {name} ({inp.groups[name]}): dev accuracy {fmt4(r['dev_accuracy_before'])} -> {fmt4(r['dev_accuracy_after'])}, val_loss "
              f"{fmt4(r['dev_val_loss_before'])} -> {fmt4(r['dev_val_loss_after'])}, fitting accuracy {fmt4(r['fitting_accuracy_before'])} -> "
              f"{fmt4(r['fitting_accuracy_after'])}, criterion {int(r['dev_criterion_before'])} -> {int(r['dev_criterion_after'])}, "
              f"{seconds['recalibration'] + seconds['evaluation']:.1f} s", flush=True)


def final_checks(cfg: dict, inp: Inputs, base_dir: Path) -> dict:
    """Alla fine: SHA-256 di ogni .keras del genitore e del suo SHA256SUMS riservato invariati."""
    changed = [n for n in inp.order if sha256_file(inp.parent[n]["weights"]) != inp.parent[n]["weights_sha256"]]
    if changed:
        fail(f"parent checkpoints changed during the run: {changed}")
    run = rel_path(base_dir, cfg["parent_run"]["folder"])
    if sha256_bytes(read_regular(run / RESTRICTED / "SHA256SUMS", "parent restricted SHA256SUMS")) != inp.parent_sums_sha:
        fail("the parent restricted SHA256SUMS changed during the run")
    return {"parent_checkpoints_unchanged": len(inp.order), "parent_restricted_sha256sums_unchanged": True}


def cmd_run(a) -> int:
    if not a.i_am_authorized:
        fail("run loads the restricted checkpoints and frames: pass --i-am-authorized (control approved by the coordinator on 13 "
             "November 2026, after the revised note and the checks of the implementation and of the baseline)")
    base_dir, cfg_p = resolve(a)
    cfg, cfg_sha = load_config(cfg_p)
    if cfg["note_sha256"] is None:
        fail("configuration: note_sha256 is null: the revised NOTE.md must be frozen and its SHA-256 written before the run")
    state = repo_state(ROOT)
    if state["dirty"]:
        fail("the git work tree is not clean: the run needs a committed state (commit or stash the changes first)")
    rid = run_id_for(a.date or time.strftime("%Y%m%d"), state)
    out_root = Path(a.out_root) if Path(a.out_root).is_absolute() else ROOT / a.out_root
    run = out_root / rid
    pcfg, _ = SR.load_config(rel_path(base_dir, cfg["parent_run"]["config_path"]))
    order = [SR.model_name(*x) for x in SR.training_order(pcfg["schema_expected"]["k"], pcfg["train_seeds"])]
    if os.path.lexists(run):
        fail(f"the run folder {rid} already exists: every attempt keeps its outcome (archive it or use another commit)")
    if msg := run_dir_problem(run, ROOT, input_paths(cfg, pcfg, base_dir, cfg_p), order, out_root):
        fail(f"refusing the run folder: {msg}")
    run.mkdir(parents=True)
    head = {"run_id": rid, "commit": state["commit"], "dirty": False, "config_sha256": cfg_sha, "note_sha256": cfg["note_sha256"],
            "parent_run_id": cfg["parent_run"]["run_id"], "command": "run"}
    budget = SR.Budget(cfg["budget"]["wall_time_cap_minutes"])
    SR.signal_handlers(budget)
    log, phases, base, rows, completed, stopped = None, {}, {}, [], [], None
    try:
        FR.write_status(run, {**head, "status": "RUNNING", "passed": False, "started": now(), "pid": os.getpid(),
                              "note": "RUNNING without a live process means a failed run; SIGUSR1 to this pid stops it as INCOMPLETE"})
        log = FR.open_log(run)
        print(f"== NUAA BatchNormalization recalibration {rid}: configuration {cfg_sha[:12]}, note {cfg['note_sha256'][:12]}, parent "
              f"{cfg['parent_run']['run_id']}, cap {cfg['budget']['wall_time_cap_minutes']} min, pid {os.getpid()}", flush=True)
        t = time.time()
        inp = load_inputs(cfg, base_dir)
        phases["phase 0 inputs"] = time.time() - t
        store = FR.FrameStore(inp, inp.X_extracted, inp.extracted_keys)
        print(f"   phase 0 PASS: {len(inp.order)} models, {len(inp.cells_sha)} cell manifests, peak RSS {SR.peak_rss_mb():.0f} MB", flush=True)
        status, phase = "PASS", "baseline"
        try:
            t = time.time()
            baseline_phase(cfg, inp, store, budget, base)
            phases["phase 1 baseline"] = time.time() - t
            failed = [n for n in inp.order if not base[n]["checks"]["baseline_pass"]]
            if failed:
                status = "BASELINE_FAILED"
                print(f"== BASELINE_FAILED: {len(failed)} models do not reproduce the recorded dev values ({', '.join(failed)}); the "
                      "recalibration is not computed (stop rule)", flush=True)
            else:
                phase = "recalibration"
                t = time.time()
                recalibration_phase(cfg, inp, store, budget, base, run, rows, completed)
                phases["phase 2 recalibration and evaluation"] = time.time() - t
        except SR.CapExceeded as e:
            status = "INCOMPLETE"
            stopped = {"phase": phase, "step": e.step, "cause": e.cause, "elapsed_minutes": round(e.minutes, 1)}
            phases[f"phase {'1 baseline' if phase == 'baseline' else '2 recalibration and evaluation'} (stopped)"] = time.time() - t
            print(f"== {'CAP' if e.cause == 'cap' else 'STOP'}: {e}; no further model (the files of a model are written only when "
                  "it is complete)", flush=True)
        t = time.time()
        checks = {**inp.checks, "final": final_checks(cfg, inp, base_dir)}
        reading = None
        if status == "PASS":
            reading = aggregate(cfg, rows)
        wall = budget.elapsed()
        phases["final checks and writes"] = time.time() - t
        timings = {"wall_seconds": round(wall, 1), "phases": {k: round(v, 1) for k, v in phases.items()},
                   "estimate_central_minutes": cfg["budget"]["estimate_minutes"], "estimate_upper_minutes": cfg["budget"]["upper_estimate_minutes"],
                   "cap_minutes": cfg["budget"]["wall_time_cap_minutes"],
                   "per_model": {n: {"baseline": round(base[n]["seconds"], 1)} for n in base}}
        for r in rows:
            timings["per_model"][name_of(r)].update({"recalibration": r["recalibration_seconds"], "evaluation": r["evaluation_seconds"]})
        record = {**head, "status": status, "finished": now(), "timings": timings, "stopped": stopped, "models": len(inp.order),
                  "baseline_failed": [n for n in base if not base[n]["checks"]["baseline_pass"]], "completed": completed,
                  "tolerances": cfg["tolerances"], "stop_rule": cfg["stop_rule"], "recovery_criterion": cfg["recovery_criterion"],
                  "estimator": cfg["estimator"], "evaluation": cfg["evaluation"], "batchnorm": cfg["batchnorm"],
                  "checks": checks, "environment": inp.environment, "device": "CPU (GPU hidden)", "code_sha256": inp.code,
                  "cells_manifests_sha256": inp.cells_sha, "test": cfg["test"], "test_read": TEST_READ,
                  "parent_public_aggregate_sha256": inp.parent_aggregate_sha}
        public = {"baseline.csv": baseline_csv(inp, base), "run_record.json": json.dumps(record, indent=1) + "\n"}
        if status in ("PASS", "INCOMPLETE") and phase == "recalibration":
            doc = {**head, "status": status, "models": len(inp.order), "completed": len(completed), "timings": timings,
                   "groups": inp.checks["groups"], "recovery_criterion": cfg["recovery_criterion"]["text"], "reading": reading,
                   "reading_not_applied": None if reading else f"run {status}: {len(completed)} of {len(inp.order)} models complete",
                   "test": cfg["test"], "test_read": TEST_READ[0].upper() + TEST_READ[1:]}
            public["results.csv"] = results_csv(cfg, rows)
            public["aggregate.json"] = json.dumps(doc, indent=1, default=float) + "\n"
            public["aggregate.md"] = aggregate_md(cfg, doc, rows)
        pub_sha = write_public(run, inp, public)
        rsums = None
        if completed:
            mw.write_all(run / RESTRICTED, {"run_record.json": public["run_record.json"],
                                            "public_SHA256SUMS.sha256": pub_sha + "  public/SHA256SUMS\n"})
            rsums = write_restricted_sums(run, [f"{n}_{x}" for n in completed for x in PER_MODEL] + ["run_record.json", "public_SHA256SUMS.sha256"])
        if reading:
            print(f"primary: {reading['primary']['statement']}")
            print(f"controls: {reading['controls']['keep']} of {reading['controls']['n']} keep the criterion, {reading['controls']['lose']} lose it")
        print(f"run folder: {run.name}, wall {wall:.0f} s ({wall / 60:.1f} min; estimate about {cfg['budget']['estimate_minutes']:g}, upper "
              f"about {cfg['budget']['upper_estimate_minutes']:g}), peak RSS "
              f"{SR.peak_rss_mb():.0f} MB")
        print(f"RESULT: {status} (run)")
        FR.close_log(log)
        log = None
        FR.write_status(run, {**head, "status": status, "passed": status == "PASS", "finished": record["finished"], "wall_seconds": round(wall, 1),
                              "stopped": stopped, "baseline_failed": record["baseline_failed"], "completed": completed,
                              "public_sums_sha256": pub_sha, "restricted_sums_sha256": rsums or "none", "run_log_sha256": sha256_file(run / LOG)})
    except BaseException as e:
        reason = str(e) if isinstance(e, Failure) else f"unexpected error {type(e).__name__}: {e}"
        left = invalidate_outputs(run, order)
        outputs = "removed" if not left else f"left behind: {', '.join(left)}"
        tail = f"ERROR: {reason}\nFAIL outputs: {outputs}\nRESULT: FAIL (run)\n"
        if log is not None:
            log[0].write(tail)
            FR.close_log(log)
        elif (run / LOG).is_file() and not (run / LOG).is_symlink():
            with open(run / LOG, "a", encoding="utf-8") as fh:
                fh.write(tail)
        try:
            FR.write_status(run, {**head, "status": "FAIL", "passed": False, "reason": reason, "outputs": outputs,
                                  "run_log_sha256": sha256_file(run / LOG) if (run / LOG).is_file() else "none"})
        except Exception:  # noqa: BLE001
            print("ERROR: status.json could not be written", file=sys.stderr)
        raise
    return STATUS_CODE[status]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="NUAA LivenessNet: BatchNormalization statistics re-estimated on the fitting set of the 36 "
                                             "patience-4 checkpoints, dev and fitting evaluated before and after (CPU, no training, no test)")
    ap.add_argument("cmd", choices=["check-inputs", "run"])
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="configuration JSON (relative to the repository root)")
    ap.add_argument("--input-root", default=None, help="base folder of the input paths of the configuration (default: repository root)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="folder that receives <run-id>/ (default results/bnrecal; another folder for tests only)")
    ap.add_argument("--date", default=None, help="YYYYMMDD of the run id (default: today)")
    ap.add_argument("--i-am-authorized", action="store_true", help="required by run: inference on the restricted checkpoints and frames")
    a = ap.parse_args(argv)
    try:
        return cmd_check_inputs(a) if a.cmd == "check-inputs" else cmd_run(a)
    except Failure as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(f"RESULT: FAIL ({a.cmd})")
        return 1
    except KeyboardInterrupt:
        print(f"RESULT: FAIL ({a.cmd}, interrupted)")
        return 130
    except SystemExit as e:
        print(f"RESULT: FAIL ({a.cmd})")
        return e.code if isinstance(e.code, int) and e.code else 1
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: unexpected error {type(e).__name__}: {e}", file=sys.stderr)
        print(f"RESULT: FAIL ({a.cmd})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
