"""
Confronto del run di verifica con early_stopping_patience 12 (20261109-SESSABLRUN-6a50265) con il run genitore
20261106-SESSABLRUN-d999458 (pazienza 4), stessi 36 modelli LivenessNet, secondo la regola di lettura concordata con il
coordinatore il 7/11/2026 (delivery/patienceverify/NOTE.md, «Verification protocol (agreed on 7 November 2026)»):
  (a) storie comuni: per ogni modello le prime epochs_run epoche del genitore devono essere identiche (loss, accuracy,
      val_loss, val_accuracy, lr; scarto 0) nel nuovo run; altrimenti lo scarto massimo (modello, epoca, campo) e il
      confronto NON VALIDO, nessun verdetto;
  (b) esito primario: positivo se i dieci degeneri originari recuperano tutti entro 12 epoche e nessuno dei 36 è
      degenere, altrimenti negativo («this change is not enough»); ogni modello senza recupero elencato a parte;
  (c) secondari per modello, genitore contro nuovo;
  (d) il test, letto SOLO dai public/aggregate.json dei due run (mai dagli score di test riservati), in una sezione
      separata e dichiarata «not used to choose or judge the recipe»;
  (e) run di verifica INCOMPLETE o FAIL: nessun verdetto, si riporta ciò che è completo.
In più i tempi (s/epoca per braccio, wall time) contro la stima di 13.090 s.

Solo lettura di uscite già scritte dai due run: nessun training, nessuna inferenza, nessuno score di test letto (i file
*_test_scores.csv elencati nei SHA256SUMS riservati non vengono mai aperti).

Ingressi (fissati nella configurazione per SHA-256; ogni file anche contro il SHA256SUMS della sua cartella):
  - run genitore: status.json PASS ancorato, SHA256SUMS pubblico e riservato ancorati; per modello _card.json,
    _dev_scores.csv e _threshold.json dal riservato; public/aggregate.json (solo la sezione del test);
  - run di verifica: stesse ancore, con valore null finché il run non è finito («to be filled after the run»): con i
    null `run` rifiuta e `check-inputs` controlla solo la configurazione e il genitore, e lo dice. status.json deve
    essere del run, della configurazione emendata (configuration_name, config_sha256, commit, amendment 4 -> 12) e PASS,
    INCOMPLETE o FAIL (con FAIL il pubblico non c'è: public_sums_sha256 = "none").
Controlli su ogni scheda come nel runner delle curve (chiavi a insieme chiuso, ricetta con la pazienza del suo run,
storia, restored_epoch = argmin, replay esatto dell'EarlyStopping, pesi nel SHA256SUMS); stessi dati del genitore
(fitting_keys_sha256 e dev_keys_sha256 uguali); dev_scores riga per riga come nel runner delle curve, stesso dev per
tutti i modelli del fold e per lo stesso modello nei due run; i dieci degeneri e i modelli senza recupero del genitore
dichiarati nella configurazione devono essere quelli che i criteri danno sulle schede del genitore.

Copiato da scripts/nuaa_degenerate_trainings.py (runner delle curve, non importato: il pacchetto è autonomo), con gli
stessi controlli: Failure/fail, sha256_bytes, sha256_file, read_regular, read_json, parse_csv, read_sums, csv_text,
sums_text, rel_path, parse_time, i validatori della configurazione (_number, _hex, _text, _exact, _int_list), is_half,
is_degenerate, first_recovery, simulate_es, auroc, dev_diag, name_fields, validate_card (con la pazienza del run come
parametro), listed_file, load_threshold, load_dev (più il confronto con il dev dell'altro run), public_problems,
write_binary, write_public, Tee, open_log, close_log, repo_state, run_id_for, run_dir_problem, write_status,
invalidate_outputs, environment, fmt, table, f4. Nuovo qui: configurazione, ancore null, status del run di verifica,
lettura della sezione del test dagli aggregati pubblici, confronto delle storie, verdetto, tabelle, figure, rifiuto di
una cartella del run che ha già uno status.json.

Uscite (cartella del run <out-root>/YYYYMMDD-PATVER-<commit>[-dev], -dev se l'albero git non è pulito):
  status.json                RUNNING all'avvio, poi PASS o FAIL (unico stato corrente)
  run.log                    copia di stdout e stderr
  public/                    comparison.csv, curves.csv, aggregate.json, aggregate.md (sezioni a-e e dichiarazioni),
                             figure_ten.png, figure_no_recovery.png, figure_val_accuracy_all.png (solo con matplotlib;
                             altrimenti salto dichiarato), SHA256SUMS
Nessuna cartella restricted/. I file pubblici non contengono date né tempi d'esecuzione di questo comando: due run con
lo stesso id danno file pubblici uguali byte per byte. Dentro il repository la cartella del run è ammessa solo sotto
results/patienceverify/; una cartella del run che ha già uno status.json è rifiutata (ogni tentativo conserva il suo esito).

Scansione del pubblico (public_problems, come nel runner delle curve), prima di ogni scrittura, su tutti i testi
pubblici e sulle etichette delle figure: chiavi d'immagine, identificativi di soggetto letti dai dev_scores (parola
intera), regole di package_rules, IPv4 non locali, segni di lavoro non finito nei .md. Un risultato qualunque: nessun
pubblico e status FAIL.

  .venv/bin/python -I scripts/nuaa_patience_verify.py check-inputs               # solo verifica, nulla scritto
  .venv/bin/python -I scripts/nuaa_patience_verify.py run --i-am-authorized      # confronto, figure e uscite

Dipendenze: solo libreria standard; matplotlib facoltativo (figure); scripts/m2_writers.py, scripts/path_guard.py,
scripts/package_rules.py. Nessun NumPy, nessun TensorFlow.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT                    # radice del codice registrato (i test possono spostare ROOT, mai questa)
sys.path.insert(0, str(ROOT / "scripts"))
import m2_writers as mw  # noqa: E402
import package_rules as pr  # noqa: E402
import path_guard as pg  # noqa: E402

DEFAULT_CONFIG = "delivery/patienceverify/nuaa_patience_verify.json"
DEFAULT_OUT_ROOT = "results/patienceverify"
CONFIG_SCHEMA = "nuaa-patience-verify-config/1"
# configurazione consegnata con le ancore del run di verifica ancora null: da aggiornare dopo averle riempite
CONFIG_SHA256 = "24c3f75bab49ee9506245c4b922a92f26388c2f21a7ba9d97919c6bbb0f53f48"
RUN_ID_RE = re.compile(r"^\d{8}-PATVER-[0-9a-f]{7,40}(-dev)?$")
TOP_KEYS = ("schema", "task", "request", "inputs", "recipe", "criteria", "original_degenerate", "models_without_recovery_parent",
            "reading_rule", "time", "declarations", "outputs")
RUNS = ("parent_run", "verification_run")
ANCHORS = ("status_sha256", "public_sums_sha256", "restricted_sums_sha256")
NO_PUBLIC = "none"                  # public_sums_sha256 di un run FAIL (pubblico tolto dal runner)
CONFIGURATION_NAME = "patience12_verification"
AMENDMENT = {"changed_field": "training.early_stopping_patience", "from": 4, "to": 12}
RUN_STATUSES = ("PASS", "INCOMPLETE", "FAIL")
HISTORY_KEYS = ("loss", "accuracy", "val_loss", "val_accuracy", "lr")
CARD_KEYS = tuple(sorted(("arch", "arm", "batch", "best_val_loss", "class_weight", "cpu_nondeterminism", "dev_keys_sha256", "device",
                          "early_stopping_patience", "epochs_run", "fitting_keys_sha256", "fold", "history", "lr", "max_epochs", "n_dev",
                          "n_fitting", "restored_by", "restored_epoch", "saved", "test_evaluation", "test_scored_after_threshold",
                          "train_seconds", "train_seed", "val_acc_at_best", "val_loss_check", "val_loss_recomputed",
                          "val_loss_recomputed_clipped", "visible_devices", "weights_file", "weights_sha256")))
DEV_COLUMNS = ("arm", "fold", "train_seed", "role", "sample_id", "subject_id", "session_id", "label", "score_attack", "prediction_05",
               "prediction_dev", "threshold_dev", "file_sha256", "checkpoint_sha256", "threshold_sha256")
RESTORED_BY = {True: "early stopping", False: "end of epochs (set_weights)"}   # fermato dalla regola / storia completa
DECISION = 0.5                      # Keras accuracy al checkpoint: attacco se score_attack > 0,5 (pareggio = bona fide)
TIME_FMT = "%Y-%m-%dT%H:%M:%S%z"
ARM_LABEL = {"arm_b": "B", "arm_a_star": "A*", "arm_b_prime": "B'"}
COMPARISONS = {"a_star_minus_b": ("arm_a_star", "arm_b"), "b_prime_minus_b": ("arm_b_prime", "arm_b"),
               "a_star_minus_b_prime": ("arm_a_star", "arm_b_prime")}
COMPARISON_LABEL = {"a_star_minus_b": "A* - B", "b_prime_minus_b": "B' - B", "a_star_minus_b_prime": "A* - B'"}
METRICS = ("apcer_05", "bpcer_05", "acer_05", "apcer_dev", "bpcer_dev", "acer_dev", "auroc")
PRIMARY_METRICS = ("auroc", "acer_dev")
TEST_VERDICTS = ("consistent", "inconclusive")
VERDICT_POSITIVE, VERDICT_NEGATIVE, VERDICT_NOT_VALID, VERDICT_NONE = "positive", "negative", "not valid", "no verdict"
TEST_LABEL = "not used to choose or judge the recipe"
DECLARATION_FLAGS = ("no_training", "no_inference", "no_test_scores_read", "test_from_public_aggregate_only",
                     "test_not_used_to_judge_the_recipe", "reading_rule_fixed_before_the_run", "criteria_defined_in_the_previous_package")
DECLARATION_TEXTS = ("no_identifiers", "dev_columns_read", "limits")
COMPARISON_COLUMNS = ("arm", "fold", "seed", "in_ten", "new_status",
                      "parent_epochs_run", "parent_restored_epoch", "parent_degenerate", "parent_recovered", "parent_first_recovery",
                      "parent_best_val_loss", "parent_dev_acc_at_checkpoint", "parent_dev_auroc_at_checkpoint", "parent_epochs_at_half",
                      "parent_lr_last",
                      "new_epochs_run", "new_restored_epoch", "new_degenerate", "new_recovered", "new_first_recovery",
                      "new_best_val_loss", "new_dev_acc_at_checkpoint", "new_dev_auroc_at_checkpoint", "new_epochs_at_half",
                      "new_lr_last", "shared_epochs", "shared_identical") + tuple(f"max_abs_diff_shared_{k}" for k in HISTORY_KEYS)
CURVE_COLUMNS = ("arm", "fold", "seed", "in_ten", "epoch", "loss", "accuracy", "val_loss", "val_accuracy", "lr", "is_restored",
                 "saturated", "shared_with_parent", "degenerate", "recovered_by_epoch")
FIGURES = ("figure_ten.png", "figure_no_recovery.png", "figure_val_accuracy_all.png")
CODE_FILES = ("scripts/nuaa_patience_verify.py", "scripts/m2_writers.py", "scripts/path_guard.py", "scripts/package_rules.py")
STATUS = "status.json"
LOG = "run.log"
PUBLIC = "public"
PUBLIC_OUTPUTS = ("comparison.csv", "curves.csv", "aggregate.json", "aggregate.md") + FIGURES + ("SHA256SUMS",)
FAIL_DOC = {"stage": "invalidation-failed", "note": "this output could not be removed after a failure: no verdict is valid"}
EXTRA_PROTECTED = ("models", "results/c1", "results/m1", "results/m2", "results/m2_rgb", "results/eval", "results/n1",
                   "results/n2", "results/c1crop", "results/folds", "results/foldsrun", "results/foldscompare", "results/stepzero",
                   "results/sessionablation", "results/sessionablationrun", "results/degenerate", "export")
RESTRICTED_NOTE = "none: this comparison writes no restricted output (the per-frame dev scores are read in place and never copied)"

REPO_STATE = None       # (root) -> {"commit", "dirty"}; None: git nel repository (sostituibile solo dai test)
PLOTTER = None          # (kind, data) -> byte PNG; None: matplotlib (sostituibile solo dai test)


class Failure(Exception):
    """Rifiuto o errore con motivo esplicito (riportato in status.json)."""


def fail(msg: str):
    raise Failure(msg)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return time.strftime(TIME_FMT)


def read_regular(p: Path, what: str) -> bytes:
    """Byte di un file regolare (non un collegamento simbolico, non una cartella, non un FIFO), letti una sola volta."""
    try:
        st = os.lstat(p)
    except OSError:
        fail(f"{what} {Path(p).name} not found")
    if stat.S_ISLNK(st.st_mode):
        fail(f"{what} {Path(p).name} is a symbolic link: inputs must be regular files")
    if not stat.S_ISREG(st.st_mode):
        fail(f"{what} {Path(p).name} is not a regular file")
    with open(p, "rb") as f:
        return f.read()


def read_json(b: bytes, what: str) -> dict:
    try:
        doc = json.loads(b.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        fail(f"{what} is not valid JSON: {e}")
    if not isinstance(doc, dict):
        fail(f"{what} is not a JSON object")
    return doc


def parse_csv(b: bytes, cols, what: str) -> list:
    try:
        text = b.decode("utf-8")
    except UnicodeDecodeError:
        fail(f"{what} is not UTF-8")
    r = csv.DictReader(io.StringIO(text, newline=""))
    if r.fieldnames != list(cols):
        fail(f"{what}: columns {r.fieldnames} differ from {list(cols)}")
    rows = list(r)
    if any(None in x or None in x.values() for x in rows):
        fail(f"{what}: rows with a wrong number of fields")
    return rows


def read_sums(b: bytes, what: str) -> dict:
    out = {}
    for line in b.decode("utf-8", errors="replace").splitlines():
        m = re.fullmatch(r"([0-9a-f]{64})  (\S.*)", line)
        if not m:
            fail(f"{what}: malformed line {line[:60]!r}")
        if m.group(2) in out:
            fail(f"{what}: {m.group(2)} listed twice")
        out[m.group(2)] = m.group(1)
    return out


def csv_text(cols, rows) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    w.writerows(rows)
    return buf.getvalue()


def sums_text(files: dict) -> str:
    return "".join(f"{sha256_bytes(t if isinstance(t, bytes) else t.encode('utf-8'))}  {n}\n" for n, t in sorted(files.items()))


def rel_path(base: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else base / q


def parse_time(v, what: str) -> datetime:
    try:
        return datetime.strptime(v, TIME_FMT)
    except (TypeError, ValueError):
        fail(f"{what}: {v!r} is not a time of the form YYYY-MM-DDTHH:MM:SS+zzzz")


# ============================================================================ configurazione

def is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _hex(v) -> bool:
    return isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) is not None


def _text(v, what: str) -> None:
    if not isinstance(v, str) or not v.strip():
        fail(f"configuration: {what} must be a non-empty text")


def _exact(d, keys, what: str) -> dict:
    if not isinstance(d, dict) or sorted(d) != sorted(keys):
        fail(f"configuration: {what} needs exactly {', '.join(sorted(keys))}")
    return d


def _int_list(v, what: str, lo: int = None) -> None:
    if not isinstance(v, list) or not v or not all(is_int(x) for x in v) or len(set(v)) != len(v) or \
            (lo is not None and min(v) < lo):
        fail(f"configuration: {what} must be a non-empty list of distinct integers" + (f" >= {lo}" if lo is not None else ""))


def model_names(cfg: dict) -> list:
    """I 36 modelli nell'ordine delle tabelle: braccio (ordine della configurazione), fold, seed."""
    i = cfg["inputs"]
    return [f"{a}_fold{f}_seed{s}" for a in i["arms"] for f in i["folds"] for s in i["seeds"]]


def anchors_filled(cfg: dict) -> bool:
    """True se le ancore del run di verifica sono riempite (load_config ammette solo tutte null o tutte riempite)."""
    return cfg["inputs"]["verification_run"]["status_sha256"] is not None


def load_config(p: Path) -> tuple:
    """(configurazione, SHA-256 dei byte letti). Insiemi chiusi e valori ammessi controllati qui, prima di tutto."""
    data = read_regular(p, "configuration")
    cfg = read_json(data, f"configuration {p.name}")
    if cfg.get("schema") != CONFIG_SCHEMA:
        fail(f"configuration {p.name}: schema must be {CONFIG_SCHEMA!r}")
    if sorted(cfg) != sorted(TOP_KEYS):
        fail(f"configuration: fields must be exactly {', '.join(TOP_KEYS)} (missing {sorted(set(TOP_KEYS) - set(cfg))}, "
             f"unknown {sorted(set(cfg) - set(TOP_KEYS))})")
    _text(cfg["task"], "task")
    for k, v in _exact(cfg["request"], ("card", "date", "text"), "request").items():
        _text(v, f"request.{k}")
    inputs = _exact(cfg["inputs"], RUNS + ("arms", "folds", "seeds", "models", "dev_rows"), "inputs")
    if not isinstance(inputs["arms"], list) or sorted(inputs["arms"]) != sorted(ARM_LABEL) or len(inputs["arms"]) != len(ARM_LABEL):
        fail(f"configuration: inputs.arms must list {', '.join(ARM_LABEL)} once each")
    _int_list(inputs["folds"], "inputs.folds", 1)
    _int_list(inputs["seeds"], "inputs.seeds", 0)
    n = len(inputs["arms"]) * len(inputs["folds"]) * len(inputs["seeds"])
    if not is_int(inputs["models"]) or inputs["models"] != n:
        fail(f"configuration: inputs.models must be {n} (arms x folds x seeds)")
    if not is_int(inputs["dev_rows"]) or inputs["dev_rows"] < 2 or inputs["dev_rows"] % 2:
        fail("configuration: inputs.dev_rows must be a positive even integer (balanced dev)")
    common = ["commit", "config_sha256", "early_stopping_patience", "folder", "label", "run_id"] + list(ANCHORS)
    par = _exact(inputs["parent_run"], common, "inputs.parent_run")
    ver = _exact(inputs["verification_run"], common + ["amendment", "anchors_note", "configuration_name"], "inputs.verification_run")
    for g, spec in (("parent_run", par), ("verification_run", ver)):
        for k in ("folder", "label", "run_id"):
            _text(spec[k], f"inputs.{g}.{k}")
        if not re.fullmatch(r"\d{8}-SESSABLRUN-[0-9a-f]{7,40}", spec["run_id"]):
            fail(f"configuration: inputs.{g}.run_id must be a run id of the session ablation runner without -dev")
        if not isinstance(spec["commit"], str) or not spec["run_id"].endswith("-" + spec["commit"]):
            fail(f"configuration: inputs.{g}.commit must be the commit of its run id")
        if not _hex(spec["config_sha256"]):
            fail(f"configuration: inputs.{g}.config_sha256 must be a SHA-256")
        if not is_int(spec["early_stopping_patience"]) or spec["early_stopping_patience"] < 1:
            fail(f"configuration: inputs.{g}.early_stopping_patience must be a positive integer")
    for k in ANCHORS:
        if not _hex(par[k]):
            fail(f"configuration: inputs.parent_run.{k} must be a SHA-256")
    filled = [ver[k] is not None for k in ANCHORS]
    if any(filled) and not all(filled):
        fail("configuration: the anchors of inputs.verification_run must be all null (to be filled after the run) or all filled")
    if all(filled):
        if not _hex(ver["status_sha256"]) or not _hex(ver["restricted_sums_sha256"]) or \
                not (_hex(ver["public_sums_sha256"]) or ver["public_sums_sha256"] == NO_PUBLIC):
            fail(f"configuration: the anchors of inputs.verification_run must be SHA-256 values ('{NO_PUBLIC}' allowed for "
                 "public_sums_sha256 of a FAIL run)")
    _text(ver["anchors_note"], "inputs.verification_run.anchors_note")
    if ver["configuration_name"] != CONFIGURATION_NAME:
        fail(f"configuration: inputs.verification_run.configuration_name must be {CONFIGURATION_NAME!r}")
    am = _exact(ver["amendment"], ["changed_field", "from", "parent_config_sha256", "to"], "inputs.verification_run.amendment")
    if (am["changed_field"], am["from"], am["to"]) != (AMENDMENT["changed_field"], AMENDMENT["from"], AMENDMENT["to"]) or \
            not is_int(am["from"]) or not is_int(am["to"]):
        fail(f"configuration: inputs.verification_run.amendment must declare {AMENDMENT['changed_field']} "
             f"{AMENDMENT['from']} -> {AMENDMENT['to']}")
    if am["parent_config_sha256"] != par["config_sha256"]:
        fail("configuration: inputs.verification_run.amendment.parent_config_sha256 must be the config_sha256 of the parent run")
    if (par["early_stopping_patience"], ver["early_stopping_patience"]) != (am["from"], am["to"]):
        fail("configuration: the patience of the two runs must be the from and to values of the amendment")
    if par["run_id"] == ver["run_id"] or par["label"] == ver["label"] or par["folder"] == ver["folder"]:
        fail("configuration: the two runs must have distinct run ids, labels and folders")
    rc = _exact(cfg["recipe"], ["batch", "lr", "max_epochs", "min_delta", "monitor", "restore_best_weights", "source"], "recipe")
    for k in ("batch", "max_epochs"):
        if not is_int(rc[k]) or rc[k] < 1:
            fail(f"configuration: recipe.{k} must be a positive integer")
    if not _number(rc["lr"]) or rc["lr"] <= 0:
        fail("configuration: recipe.lr must be positive")
    if rc["min_delta"] != 0 or rc["monitor"] != "val_loss" or rc["restore_best_weights"] is not True:
        fail("configuration: the recipe of the runs is monitor val_loss, min_delta 0, restore_best_weights true")
    if ver["early_stopping_patience"] != rc["max_epochs"]:
        fail("configuration: the patience of the verification run must equal recipe.max_epochs (no early stop)")
    _text(rc["source"], "recipe.source")
    cr = _exact(cfg["criteria"], ["checkpoint_agreement_tolerance", "degenerate", "recovery", "saturation", "shared_history_difference",
                                  "source"], "criteria")
    dg = _exact(cr["degenerate"], ["restored_epochs", "text", "tolerance", "val_acc_at_best"], "criteria.degenerate")
    _int_list(dg["restored_epochs"], "criteria.degenerate.restored_epochs", 1)
    if not _number(dg["val_acc_at_best"]) or not 0 <= dg["val_acc_at_best"] <= 1 or not _number(dg["tolerance"]) or \
            not 0 < dg["tolerance"] < 1e-3:
        fail("configuration: criteria.degenerate needs val_acc_at_best in [0, 1] and a tolerance in (0, 1e-3)")
    _text(dg["text"], "criteria.degenerate.text")
    sa = _exact(cr["saturation"], ["text", "tolerance", "val_accuracy"], "criteria.saturation")
    if sa["val_accuracy"] != dg["val_acc_at_best"] or sa["tolerance"] != dg["tolerance"]:
        fail("configuration: criteria.saturation must use the value and tolerance of criteria.degenerate (one definition)")
    _text(sa["text"], "criteria.saturation.text")
    rv = _exact(cr["recovery"], ["text", "val_accuracy_min", "val_loss_max"], "criteria.recovery")
    for k in ("val_accuracy_min", "val_loss_max"):
        if not _number(rv[k]) or rv[k] <= 0:
            fail(f"configuration: criteria.recovery.{k} must be a positive number")
    _text(rv["text"], "criteria.recovery.text")
    if not _number(cr["checkpoint_agreement_tolerance"]) or not 0 < cr["checkpoint_agreement_tolerance"] < 1e-3:
        fail("configuration: criteria.checkpoint_agreement_tolerance must be in (0, 1e-3)")
    if cr["shared_history_difference"] != 0 or not is_int(cr["shared_history_difference"]):
        fail("configuration: criteria.shared_history_difference must be 0 (identical shared histories, as agreed)")
    _text(cr["source"], "criteria.source")
    names = model_names(cfg)
    ten = cfg["original_degenerate"]
    if not isinstance(ten, list) or len(ten) != 10 or len(set(ten)) != 10 or not set(ten) <= set(names) or ten != sorted(ten):
        fail("configuration: original_degenerate must list the ten degenerate models of the parent run, sorted, once each")
    nr = cfg["models_without_recovery_parent"]
    if not isinstance(nr, list) or len(set(nr)) != len(nr) or not set(nr) <= set(names) or set(nr) & set(ten) or nr != sorted(nr):
        fail("configuration: models_without_recovery_parent must list models of the run outside the ten, sorted, once each")
    _exact(cfg["reading_rule"], ["a", "b", "c", "d", "e"], "reading_rule")
    for k, v in cfg["reading_rule"].items():
        _text(v, f"reading_rule.{k}")
    tm = _exact(cfg["time"], ["cap_seconds", "estimate_seconds", "source"], "time")
    if not _number(tm["estimate_seconds"]) or not _number(tm["cap_seconds"]) or not 0 < tm["estimate_seconds"] <= tm["cap_seconds"]:
        fail("configuration: time.estimate_seconds must be positive and not above time.cap_seconds")
    _text(tm["source"], "time.source")
    dec = _exact(cfg["declarations"], DECLARATION_FLAGS + DECLARATION_TEXTS, "declarations")
    for k in DECLARATION_FLAGS:
        if dec[k] is not True:
            fail(f"configuration: declarations.{k} must be true")
    for k in DECLARATION_TEXTS:
        _text(dec[k], f"declarations.{k}")
    for k, v in _exact(cfg["outputs"], ["public", "restricted"], "outputs").items():
        _text(v, f"outputs.{k}")
    return cfg, sha256_bytes(data)


# ============================================================================ regole (una sola definizione ciascuna)

def is_half(v: float, crit: dict) -> bool:
    """Epoca satura: val_accuracy uguale al valore del criterio entro la tolleranza (0,5 ± 1e-6)."""
    return abs(v - crit["saturation"]["val_accuracy"]) < crit["saturation"]["tolerance"]


def is_degenerate(restored: int, val_acc: float, crit: dict) -> bool:
    """Criterio dei degeneri: epoca ripristinata fra quelle del criterio e val_accuracy al checkpoint = 0,5."""
    d = crit["degenerate"]
    return restored in d["restored_epochs"] and abs(val_acc - d["val_acc_at_best"]) < d["tolerance"]


def first_recovery(h: dict, acc: float, loss: float):
    """Prima epoca (da 1) con val_accuracy >= acc e val_loss < loss; None se nessuna."""
    for i, (va, vl) in enumerate(zip(h["val_accuracy"], h["val_loss"])):
        if va >= acc and vl < loss:
            return i + 1
    return None


def simulate_es(h: dict, n_max: int, monitor: str = "val_loss", patience: int = 4, start: int = 0):
    """EarlyStopping di Keras 2.15 (min_delta 0, baseline None, restore_best_weights) rigiocato sulla storia registrata
    (copiato dal runner delle curve): prima di start_from_epoch nulla; wait += 1; un miglioramento stretto aggiorna il
    migliore e azzera wait; altrimenti con wait >= patience e epoca > 0 si ferma e ripristina il migliore. None se la
    storia finisce prima della regola senza arrivare a n_max (non determinabile)."""
    vals = h[monitor]
    better = (lambda a, b: a < b) if monitor == "val_loss" else (lambda a, b: a > b)
    best = math.inf if monitor == "val_loss" else -math.inf
    best_ep, wait = None, 0
    for e, v in enumerate(vals):
        if e < start:
            continue
        wait += 1
        if better(v, best):
            best, best_ep, wait = v, e, 0
            continue
        if wait >= patience and e > 0:
            return {"stop": e + 1, "restored": best_ep + 1, "early": True}
    if len(vals) == n_max and best_ep is not None:
        return {"stop": n_max, "restored": best_ep + 1, "early": False}
    return None


def auroc(labels, scores) -> float:
    """AUROC (attacco = 1 positivo, score più alto -> attacco) con i ranghi medi per i pareggi (Mann-Whitney)."""
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    pos = sum(1 for _, y in pairs if y == 1)
    neg = len(pairs) - pos
    rs = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (rs - pos * (pos + 1) / 2) / (pos * neg)


def dev_diag(labels: list, scores: list) -> dict:
    """Diagnostica del checkpoint sul dev: accuratezza alla soglia di Keras (attacco se score > 0,5) e AUROC."""
    return {"n": len(scores), "acc_05": sum(1 for s, y in zip(scores, labels) if (s > DECISION) == (y == 1)) / len(scores),
            "auroc": auroc(labels, scores)}


def shared_history(hp: dict, hn: dict, n_shared: int) -> dict:
    """Confronto (a) delle prime n_shared epoche: identico solo se ogni valore dei cinque campi è uguale (scarto 0, nessuna
    tolleranza); per campo lo scarto assoluto massimo e la prima epoca in cui è raggiunto. Una storia nuova più corta
    delle epoche comuni non è identica."""
    out = {"shared_epochs": n_shared, "complete": all(len(hn[k]) >= n_shared for k in HISTORY_KEYS), "fields": {}}
    for k in HISTORY_KEYS:
        mx, at = 0.0, None
        for i in range(min(n_shared, len(hn[k]))):
            d = abs(hn[k][i] - hp[k][i])
            if hn[k][i] != hp[k][i] and (at is None or d > mx):
                mx, at = d, i + 1
        out["fields"][k] = {"max_abs_difference": mx, "epoch": at}
    out["identical"] = out["complete"] and all(v["epoch"] is None for v in out["fields"].values())
    return out


# ============================================================================ ingressi

class Run:
    """Un run letto: status, modelli completi (scheda validata, storia, dev), sezione del test dal pubblico."""

    def __init__(self, label: str):
        self.label = label
        self.status = None
        self.st = {}
        self.models = {}
        self.test = None


class Inputs:
    def __init__(self):
        self.parent = None
        self.new = None
        self.subjects = set()               # identificativi di soggetto dei dev (solo per la scansione)
        self.checks = {}


def name_fields(name: str) -> tuple:
    m = re.fullmatch(r"([a-z][a-z0-9_]*)_fold(\d+)_seed(\d+)", name)
    if not m:
        fail(f"model name {name!r} does not follow <arm>_fold<k>_seed<s>")
    return m.group(1), int(m.group(2)), int(m.group(3))


def validate_card(c: dict, name: str, patience: int, cfg: dict, listed: dict, what: str) -> None:
    """Scheda di training (come nel runner delle curve): chiavi, ricetta con la pazienza del run, storia, checkpoint e
    replay dell'EarlyStopping che riproduce arresto ed epoca ripristinata."""
    what = f"{what} card {name}"
    rc = cfg["recipe"]
    if sorted(c) != list(CARD_KEYS):
        fail(f"{what}: keys differ from the expected set (missing {sorted(set(CARD_KEYS) - set(c))}, "
             f"unknown {sorted(set(c) - set(CARD_KEYS))})")
    arm, fold, seed = name_fields(name)
    if c["arch"] != "LivenessNet":
        fail(f"{what}: arch is not LivenessNet")
    if c["fold"] != fold or c["train_seed"] != seed or c["arm"] != arm:
        fail(f"{what}: arm, fold or train_seed differ from the file name")
    if not is_int(c["fold"]) or not is_int(c["train_seed"]):      # true == 1 per Python: un booleano non è un intero
        fail(f"{what}: fold and train_seed must be integers (not booleans)")
    want = {"max_epochs": rc["max_epochs"], "early_stopping_patience": patience, "batch": rc["batch"], "lr": rc["lr"]}
    for k, v in want.items():
        if c[k] != v or isinstance(c[k], bool):
            fail(f"{what}: {k} = {c[k]!r} differs from the recipe of its run ({v!r})")
    if not is_int(c["n_fitting"]) or c["n_fitting"] < 1:
        fail(f"{what}: n_fitting must be a positive integer")
    if not is_int(c["n_dev"]) or c["n_dev"] < 2:
        fail(f"{what}: n_dev must be an integer >= 2")
    if not _number(c["train_seconds"]) or c["train_seconds"] <= 0:
        fail(f"{what}: train_seconds must be a positive number")
    n = c["epochs_run"]
    if not is_int(n) or not 1 <= n <= rc["max_epochs"]:
        fail(f"{what}: epochs_run must be an integer between 1 and max_epochs")
    h = c["history"]
    if not isinstance(h, dict) or sorted(h) != sorted(HISTORY_KEYS):
        fail(f"{what}: history must have exactly the keys {', '.join(HISTORY_KEYS)}")
    for k in HISTORY_KEYS:
        if not isinstance(h[k], list) or len(h[k]) != n:
            fail(f"{what}: history.{k} has {len(h[k]) if isinstance(h[k], list) else 'no'} values, epochs_run is {n}")
        if not all(_number(v) for v in h[k]):
            fail(f"{what}: history.{k} holds a value that is not a finite number")
    if not all(0 <= v <= 1 for k in ("accuracy", "val_accuracy") for v in h[k]):
        fail(f"{what}: accuracy outside [0, 1] in the history")
    if not all(v >= 0 for k in ("loss", "val_loss") for v in h[k]):
        fail(f"{what}: negative loss in the history")
    if not all(v > 0 for v in h["lr"]):
        fail(f"{what}: non-positive lr in the history")
    r = c["restored_epoch"]
    argmin = h["val_loss"].index(min(h["val_loss"])) + 1
    if not is_int(r) or r != argmin:
        fail(f"{what}: restored_epoch {r!r} is not the epoch of minimum val_loss ({argmin})")
    if c["best_val_loss"] != h["val_loss"][r - 1] or c["val_acc_at_best"] != h["val_accuracy"][r - 1]:
        fail(f"{what}: best_val_loss or val_acc_at_best differ from the history at the restored epoch")
    sim = simulate_es(h, rc["max_epochs"], rc["monitor"], patience, 0)
    if sim is None or sim["stop"] != n or sim["restored"] != r:
        got = "not determinable" if sim is None else f"stop {sim['stop']}, restored {sim['restored']}"
        fail(f"{what}: the replay of the Keras EarlyStopping (patience {patience}) gives {got}, the card records epochs_run {n}, "
             f"restored_epoch {r}")
    if c["restored_by"] != RESTORED_BY[sim["early"]]:
        fail(f"{what}: restored_by {c['restored_by']!r} differs from the replay ({RESTORED_BY[sim['early']]!r})")
    if c["weights_file"] != f"{name}_LivenessNet.keras" or not _hex(c["weights_sha256"]) or listed.get(c["weights_file"]) != c["weights_sha256"]:
        fail(f"{what}: weights_file or weights_sha256 differ from the restricted SHA256SUMS of its folder")
    if not _hex(c["dev_keys_sha256"]) or not _hex(c["fitting_keys_sha256"]):
        fail(f"{what}: dev_keys_sha256 and fitting_keys_sha256 must be SHA-256 values")
    parse_time(c["saved"], f"{what} saved")
    parse_time(c["test_scored_after_threshold"], f"{what} test_scored_after_threshold")


def listed_file(folder: Path, listed: dict, fn: str, what: str) -> bytes:
    if fn not in listed:
        fail(f"{what} restricted SHA256SUMS: {fn} not listed")
    b = read_regular(folder / "restricted" / fn, f"{what} {fn}")
    if sha256_bytes(b) != listed[fn]:
        fail(f"{what} {fn}: SHA-256 {sha256_bytes(b)} differs from the restricted SHA256SUMS ({listed[fn]})")
    return b


def load_threshold(b: bytes, card: dict, name: str, what: str) -> dict:
    """threshold.json del checkpoint: braccio, fold, seed e checkpoint uguali alla scheda; value un numero finito;
    frozen_at fra il salvataggio dei pesi e lo score del test; dev_scores_sha256 controllato con il dev."""
    what = f"{what} {name}_threshold.json"
    thr = read_json(b, what)
    arm, fold, seed = name_fields(name)
    if thr.get("arm") != arm or thr.get("fold") != fold or thr.get("train_seed") != seed:
        fail(f"{what}: arm, fold or train_seed differ from the card")
    if not is_int(thr.get("fold")) or not is_int(thr.get("train_seed")):      # true == 1 per Python
        fail(f"{what}: fold and train_seed must be integers (not booleans)")
    if thr.get("checkpoint_sha256") != card["weights_sha256"]:
        fail(f"{what}: checkpoint_sha256 differs from the weights of the card")
    if not _number(thr.get("value")):
        fail(f"{what}: value is not a finite number")
    if not _hex(thr.get("dev_scores_sha256")):
        fail(f"{what}: dev_scores_sha256 must be a SHA-256")
    saved = parse_time(card["saved"], f"{name} saved")
    frozen = parse_time(thr.get("frozen_at"), f"{what} frozen_at")
    tested = parse_time(card["test_scored_after_threshold"], f"{name} test_scored_after_threshold")
    if not saved <= frozen <= tested:
        fail(f"{what}: saved, frozen_at and test_scored_after_threshold are not in order")
    return {"value": float(thr["value"]), "dev_scores_sha256": thr["dev_scores_sha256"], "sha256": sha256_bytes(b)}


def load_dev(b: bytes, card: dict, name: str, cfg: dict, thr: dict, inp: Inputs, what: str) -> tuple:
    """dev_scores.csv del checkpoint, riga per riga (come nel runner delle curve): ruolo, braccio, fold, seed e checkpoint
    della scheda; etichetta e score nel dominio; decisioni e soglia della riga coerenti con la soglia congelata (attacco
    se score > soglia, pareggio bona fide; prediction_05 alla soglia fissa 0,5); sample_id e file_sha256 unici; chiavi
    uguali a dev_keys_sha256 della scheda; score uguali a dev_scores_sha256 della soglia; dev bilanciato; accuratezza
    alla soglia 0,5 = val_acc_at_best. Restituisce la diagnostica e le chiavi (solo in memoria)."""
    what = f"{what} {name}_dev_scores.csv"
    rows = parse_csv(b, DEV_COLUMNS, what)
    n_dev = cfg["inputs"]["dev_rows"]
    if len(rows) != n_dev:
        fail(f"{what}: {len(rows)} rows; the configuration expects {n_dev}")
    if card["n_dev"] != n_dev:
        fail(f"{what}: n_dev of the card is {card['n_dev']}; the configuration expects {n_dev}")
    arm, fold, seed = name_fields(name)
    value = thr["value"]
    labels, scores, subjects, keys = [], [], set(), []
    for r in rows:
        if r["role"] != "dev":
            fail(f"{what}: a row carries another role")
        if r["fold"] != str(fold) or r["train_seed"] != str(seed) or r["arm"] != arm:
            fail(f"{what}: a row carries another arm, fold or seed")
        if r["checkpoint_sha256"] != card["weights_sha256"]:
            fail(f"{what}: checkpoint_sha256 differs from the weights of the card")
        if r["label"] not in ("0", "1"):
            fail(f"{what}: label out of range")
        try:
            s = float(r["score_attack"])
        except ValueError:
            fail(f"{what}: score_attack is not a number")
        if not math.isfinite(s) or not 0.0 <= s <= 1.0:
            fail(f"{what}: score_attack outside [0, 1]")
        if r["threshold_dev"] != repr(value) or r["threshold_sha256"] != thr["sha256"]:
            fail(f"{what}: threshold_dev or threshold_sha256 differ from the frozen threshold {name}_threshold.json")
        if r["prediction_05"] != ("1" if s > DECISION else "0"):
            fail(f"{what}: prediction_05 does not follow 'attack if score_attack > 0.5 (tie = bona fide)'")
        if r["prediction_dev"] != ("1" if s > value else "0"):
            fail(f"{what}: prediction_dev does not follow 'attack if score_attack > threshold (tie = bona fide)'")
        labels.append(int(r["label"]))
        scores.append(s)
        subjects.add(r["subject_id"])           # solo per la scansione del pubblico
        keys.append((r["sample_id"], r["file_sha256"], r["label"], r["subject_id"]))
    if len({k[0] for k in keys}) != len(keys) or len({k[1] for k in keys}) != len(keys):
        fail(f"{what}: a sample_id or a file_sha256 appears in more than one row")
    if sha256_bytes("\n".join(k[0] for k in keys).encode()) != card["dev_keys_sha256"]:
        fail(f"{what}: the sample_id list differs from dev_keys_sha256 of the card")
    if sha256_bytes("\n".join(repr(float(r["score_attack"])) for r in rows).encode()) != thr["dev_scores_sha256"]:
        fail(f"{what}: the scores differ from dev_scores_sha256 of {name}_threshold.json")
    if labels.count(0) != labels.count(1):
        fail(f"{what}: the dev is not balanced ({labels.count(0)} bona fide, {labels.count(1)} attacks)")
    d = dev_diag(labels, scores)
    if abs(d["acc_05"] - card["val_acc_at_best"]) > cfg["criteria"]["checkpoint_agreement_tolerance"]:
        fail(f"{what}: accuracy at 0.5 of the checkpoint {d['acc_05']:.6f} differs from val_acc_at_best {card['val_acc_at_best']:.6f}")
    inp.subjects |= subjects
    return d, tuple(keys)


def load_test(agg: dict, what: str) -> dict:
    """Sezione del test dall'aggregate.json pubblico (solo questa, sola lettura): verdetti primari e, per i tre
    confronti, verdetto e mediana sui seed della differenza per strato e metrica. Struttura controllata a insieme chiuso
    sulle chiavi usate; nessun altro campo letto."""
    strata = agg.get("strata")
    if not isinstance(strata, list) or not strata or not all(isinstance(s, str) for s in strata) or len(set(strata)) != len(strata):
        fail(f"{what}: strata must be a list of distinct names")

    def verdict(v, where):
        if not isinstance(v, dict) or v.get("verdict") not in TEST_VERDICTS or not isinstance(v.get("reason"), str):
            fail(f"{what}: {where} is not a verdict ({', '.join(TEST_VERDICTS)}) with a reason")
        if (v["verdict"] == "consistent") != isinstance(v.get("direction"), str):
            fail(f"{what}: {where} has a direction only when consistent")
        return {"verdict": v["verdict"], "direction": v.get("direction"), "reason": v["reason"]}
    pv = agg.get("primary_verdicts")
    if not isinstance(pv, dict) or sorted(pv) != sorted(PRIMARY_METRICS):
        fail(f"{what}: primary_verdicts must hold exactly {', '.join(PRIMARY_METRICS)}")
    out = {"strata": list(strata), "primary": {k: verdict(pv[k], f"primary_verdicts.{k}") for k in PRIMARY_METRICS}, "rows": {}}
    comps = agg.get("comparisons")
    if not isinstance(comps, dict) or sorted(comps) != sorted(COMPARISONS):
        fail(f"{what}: comparisons must hold exactly {', '.join(COMPARISONS)}")
    for c, (mi, su) in COMPARISONS.items():
        cc = comps[c]
        if not isinstance(cc, dict) or cc.get("minuend") != mi or cc.get("subtrahend") != su:
            fail(f"{what}: comparisons.{c} must be {mi} - {su}")
        for s in strata:
            for m in METRICS:
                try:
                    v, med = cc["verdicts"][s][m], cc["pooled_summary_over_seeds"][s][m]["median"]
                except (KeyError, TypeError):
                    fail(f"{what}: comparisons.{c} has no verdict or median for stratum {s}, metric {m}")
                if med is not None and not _number(med):
                    fail(f"{what}: comparisons.{c} median of stratum {s}, metric {m} is not a number or null")
                out["rows"][(c, s, m)] = {**verdict(v, f"comparisons.{c} stratum {s} metric {m}"), "median": med}
    return out


def load_run(cfg: dict, base: Path, g: str, inp: Inputs) -> Run:
    """Run genitore o di verifica: status ancorato e coerente con la configurazione, SHA256SUMS ancorati, schede,
    soglie e dev dei modelli completi per SHA-256 registrato, sezione del test dal pubblico (se c'è)."""
    spec = cfg["inputs"][g]
    run = Run(spec["label"])
    folder = rel_path(base, spec["folder"])
    sb = read_regular(folder / STATUS, f"{g} status.json")
    if sha256_bytes(sb) != spec["status_sha256"]:
        fail(f"{g} status.json: SHA-256 {sha256_bytes(sb)} differs from the configuration ({spec['status_sha256']})")
    st = read_json(sb, f"{g} status.json")
    names = model_names(cfg)
    status = st.get("status")
    allowed = ("PASS",) if g == "parent_run" else RUN_STATUSES
    if status not in allowed or st.get("passed") is not (status == "PASS"):
        fail(f"{g} status.json: status {status!r} (passed {st.get('passed')!r}) is not one of {', '.join(allowed)}")
    for k in ("run_id", "commit", "config_sha256"):
        if st.get(k) != spec[k]:
            fail(f"{g} status.json: {k} {st.get(k)!r} differs from the configuration ({spec[k]!r})")
    if st.get("dirty") not in (None, False):
        fail(f"{g} status.json: a run from a tree that is not clean")
    if g == "verification_run":
        if st.get("configuration_name") != spec["configuration_name"]:
            fail(f"{g} status.json: configuration_name {st.get('configuration_name')!r} differs from {spec['configuration_name']!r}")
        am = st.get("amendment")
        if not isinstance(am, dict) or any(am.get(k) != v for k, v in spec["amendment"].items()):
            fail(f"{g} status.json: amendment differs from the configuration (parent SHA-256, field, from and to)")
    if status == "FAIL":
        completed = st.get("kept_complete_models")
        missing = [n for n in names if n not in (completed or [])]
    else:
        completed, missing = st.get("completed"), st.get("missing")
    if not isinstance(completed, list) or not isinstance(missing, list) or not completed or len(set(completed)) != len(completed) \
            or sorted(completed + missing) != sorted(names):
        fail(f"{g} status.json: the complete and missing models must split the {len(names)} models of the configuration, with "
             "at least one complete")
    if status == "PASS" and missing:
        fail(f"{g} status.json: PASS with missing models")
    if status == "INCOMPLETE" and not missing:
        fail(f"{g} status.json: INCOMPLETE without missing models")
    if st.get("restricted_sums_sha256") != spec["restricted_sums_sha256"]:
        fail(f"{g} status.json: the restricted SHA256SUMS anchor differs from the configuration")
    has_public = status != "FAIL"
    if has_public != (spec["public_sums_sha256"] != NO_PUBLIC):
        fail(f"{g}: public_sums_sha256 of the configuration must be '{NO_PUBLIC}' exactly when the run is FAIL")
    if has_public and st.get("public_sums_sha256") != spec["public_sums_sha256"]:
        fail(f"{g} status.json: the public SHA256SUMS anchor differs from the configuration")
    wall = st.get("wall_seconds")
    if not _number(wall) or wall <= 0:
        fail(f"{g} status.json: wall_seconds must be a positive number")
    rb = read_regular(folder / "restricted" / "SHA256SUMS", f"{g} restricted SHA256SUMS")
    if sha256_bytes(rb) != spec["restricted_sums_sha256"]:
        fail(f"{g} restricted SHA256SUMS: SHA-256 {sha256_bytes(rb)} differs from the configuration ({spec['restricted_sums_sha256']})")
    listed = read_sums(rb, f"{g} restricted SHA256SUMS")
    cards = sorted(n[: -len("_card.json")] for n in listed if n.endswith("_card.json"))
    if cards != sorted(completed):
        fail(f"{g} restricted SHA256SUMS: {len(cards)} training cards listed, status.json declares {len(completed)} complete models "
             f"(missing {len(set(completed) - set(cards))}, unexpected {len(set(cards) - set(completed))})")
    if has_public:
        pb = read_regular(folder / PUBLIC / "SHA256SUMS", f"{g} public SHA256SUMS")
        if sha256_bytes(pb) != spec["public_sums_sha256"]:
            fail(f"{g} public SHA256SUMS: SHA-256 {sha256_bytes(pb)} differs from the configuration ({spec['public_sums_sha256']})")
        plisted = read_sums(pb, f"{g} public SHA256SUMS")
        if "aggregate.json" not in plisted:
            fail(f"{g} public SHA256SUMS: aggregate.json not listed")
        ab = read_regular(folder / PUBLIC / "aggregate.json", f"{g} public aggregate.json")
        if sha256_bytes(ab) != plisted["aggregate.json"]:
            fail(f"{g} public aggregate.json: SHA-256 differs from the public SHA256SUMS")
        agg = read_json(ab, f"{g} public aggregate.json")
        for k in ("run_id", "config_sha256", "status"):
            if agg.get(k) != st.get(k):
                fail(f"{g} public aggregate.json: {k} differs from status.json")
        run.test = load_test(agg, f"{g} public aggregate.json")
    patience = spec["early_stopping_patience"]
    dev_of_fold = {}
    for name in [n for n in names if n in completed]:
        card = read_json(listed_file(folder, listed, f"{name}_card.json", g), f"{g} {name}_card.json")
        validate_card(card, name, patience, cfg, listed, g)
        thr = load_threshold(listed_file(folder, listed, f"{name}_threshold.json", g), card, name, g)
        d, keys = load_dev(listed_file(folder, listed, f"{name}_dev_scores.csv", g), card, name, cfg, thr, inp, g)
        arm, fold, seed = name_fields(name)
        # dev comune: stesse righe (chiave, file, etichetta, soggetto) nello stesso ordine per tutti i modelli del fold
        if dev_of_fold.setdefault(fold, keys) != keys:
            fail(f"{g} fold {fold}: the models of the fold do not share the same dev (keys, files, labels or subjects differ)")
        run.models[name] = {"name": name, "arm": arm, "fold": fold, "seed": seed, "card": card, "h": card["history"], "dev": d,
                            "keys": keys}
    run.status, run.st = status, st
    inp.checks[g] = {"run_id": spec["run_id"], "status": status, "config_sha256": spec["config_sha256"], "commit": spec["commit"],
                     "status_sha256": spec["status_sha256"], "public_sums_sha256": spec["public_sums_sha256"],
                     "restricted_sums_sha256": spec["restricted_sums_sha256"], "early_stopping_patience": patience,
                     "complete_models": len(run.models), "missing_models": len(missing),
                     "early_stopping_replay": f"{len(run.models)}/{len(run.models)} reproduced",
                     "test_section": "public aggregate.json" if has_public else "none (FAIL run: no public folder)"}
    return run


def check_parent_lists(cfg: dict, par: Run) -> None:
    """I dieci degeneri e i modelli senza recupero del genitore dichiarati nella configurazione devono essere quelli che
    i criteri danno sulle schede del genitore (nessuna lista scritta a mano diversa dai dati)."""
    crit = cfg["criteria"]
    rv = crit["recovery"]
    deg = sorted(n for n, m in par.models.items() if is_degenerate(m["card"]["restored_epoch"], m["card"]["val_acc_at_best"], crit))
    if deg != cfg["original_degenerate"]:
        fail(f"configuration: original_degenerate differs from the degenerate models of the parent run by the criterion "
             f"({len(deg)} found)")
    nr = sorted(n for n, m in par.models.items() if n not in deg and first_recovery(m["h"], rv["val_accuracy_min"], rv["val_loss_max"]) is None)
    if nr != cfg["models_without_recovery_parent"]:
        fail(f"configuration: models_without_recovery_parent differs from the parent run ({len(nr)} non-degenerate models without recovery)")


def load_inputs(cfg: dict, base: Path, need_verification: bool = True) -> Inputs:
    """Genitore sempre; run di verifica solo con le ancore riempite (con i null: None, e `run` rifiuta prima)."""
    inp = Inputs()
    inp.parent = load_run(cfg, base, "parent_run", inp)
    if len(inp.parent.models) != cfg["inputs"]["models"]:
        fail("parent_run: not every model is complete")
    check_parent_lists(cfg, inp.parent)
    if not anchors_filled(cfg):
        if need_verification:
            fail("the anchors of the verification run are null (to be filled after the run): run refused")
        inp.checks["verification_run"] = {"run_id": cfg["inputs"]["verification_run"]["run_id"],
                                          "status": "not read: anchors null (to be filled after the run)"}
        return inp
    inp.new = load_run(cfg, base, "verification_run", inp)
    for name, m in inp.new.models.items():
        p = inp.parent.models[name]
        if m["card"]["fitting_keys_sha256"] != p["card"]["fitting_keys_sha256"] or m["card"]["n_fitting"] != p["card"]["n_fitting"]:
            fail(f"verification_run {name}: the fitting set differs from the parent run")
        if m["keys"] != p["keys"]:
            fail(f"verification_run {name}: the dev differs from the parent run (keys, files, labels or subjects)")
    return inp


# ============================================================================ analisi

def model_view(m: dict, crit: dict) -> dict:
    """Campi derivati di un modello (una sola volta, con le regole sopra)."""
    c, h = m["card"], m["h"]
    rv = crit["recovery"]
    rec = first_recovery(h, rv["val_accuracy_min"], rv["val_loss_max"])
    return {"epochs_run": c["epochs_run"], "restored_epoch": c["restored_epoch"], "best_val_loss": c["best_val_loss"],
            "val_acc_at_best": c["val_acc_at_best"], "degenerate": is_degenerate(c["restored_epoch"], c["val_acc_at_best"], crit),
            "first_recovery": rec, "recovered": rec is not None, "dev_acc": m["dev"]["acc_05"], "dev_auroc": m["dev"]["auroc"],
            "epochs_at_half": sum(1 for v in h["val_accuracy"] if is_half(v, crit)), "lr": list(h["lr"]), "lr_last": h["lr"][-1],
            "train_seconds": c["train_seconds"]}


def label(name: str) -> str:
    arm, fold, seed = name_fields(name)
    return f"{ARM_LABEL[arm]} fold {fold} seed {seed}"


def analyse(cfg: dict, inp: Inputs) -> dict:
    """Sezioni (a)-(e) come dati; verdetto con la precedenza (e) > (a) > (b)."""
    crit = cfg["criteria"]
    names = model_names(cfg)
    ten = cfg["original_degenerate"]
    par, new = inp.parent, inp.new
    P = {n: model_view(par.models[n], crit) for n in names}
    N = {n: model_view(new.models[n], crit) for n in names if n in new.models}
    # (a) storie comuni
    shared = {n: shared_history(par.models[n]["h"], new.models[n]["h"], P[n]["epochs_run"]) for n in N}
    worst = None
    for n in N:
        for k in HISTORY_KEYS:
            f = shared[n]["fields"][k]
            if f["epoch"] is not None and (worst is None or f["max_abs_difference"] > worst["max_abs_difference"]):
                worst = {"model": label(n), "epoch": f["epoch"], "field": k, "max_abs_difference": f["max_abs_difference"]}
    shorter = [label(n) for n in N if not shared[n]["complete"]]
    identical = sum(1 for n in N if shared[n]["identical"])
    valid = identical == len(N)
    A = {"rule": cfg["reading_rule"]["a"], "models_compared": len(N), "identical": identical, "valid": valid,
         "largest_difference": worst, "new_history_shorter_than_shared": shorter,
         "max_abs_difference_per_field": {k: max([shared[n]["fields"][k]["max_abs_difference"] for n in N], default=0.0)
                                          for k in HISTORY_KEYS}}
    # (b) esito primario
    ten_not_rec = [label(n) for n in ten if n in N and not N[n]["recovered"]]
    ten_deg = [label(n) for n in ten if n in N and N[n]["degenerate"]]
    deg_new = [label(n) for n in names if n in N and N[n]["degenerate"]]
    out_not_rec = [label(n) for n in names if n in N and n not in ten and not N[n]["recovered"]]
    parent_not_rec = [label(n) for n in names if not P[n]["recovered"]]
    missing = [label(n) for n in names if n not in N]
    ten_ok = not ten_not_rec and all(n in N for n in ten)          # i degeneri (dieci compresi) sono in deg_new
    if new.status != "PASS":
        verdict = VERDICT_NONE
        concl = (f"No verdict: the verification run is {new.status} ({len(N)} of {len(names)} models complete; missing: "
                 f"{', '.join(missing)}); what is complete is reported below.")
    elif not valid:
        verdict = VERDICT_NOT_VALID
        w = worst or {"model": shorter[0] if shorter else "-", "epoch": None, "field": "history length", "max_abs_difference": None}
        concl = (f"Not valid: the shared histories are not identical in {len(N) - identical} of {len(N)} models (largest difference "
                 f"{w['max_abs_difference']!r} in {w['model']}, epoch {w['epoch']}, field {w['field']}); no verdict.")
    elif ten_ok and not deg_new:
        verdict = VERDICT_POSITIVE
        concl = ("Positive: the ten original degenerate trainings recover within 12 epochs and none of the 36 models is degenerate. "
                 "The recovery does not separate the stop from the lr halved by ReduceLROnPlateau (declared confounder), and a "
                 "positive outcome does not by itself show that every model is stable.")
    else:
        verdict = VERDICT_NEGATIVE
        concl = (f"Negative: this change is not enough (among the ten, without recovery: {', '.join(ten_not_rec) or 'none'}; "
                 f"degenerate among the 36: {', '.join(deg_new) or 'none'}). This does not rule out a role of early stopping "
                 "(the lr halved by ReduceLROnPlateau remains the declared confounder).")
    if verdict in (VERDICT_POSITIVE, VERDICT_NEGATIVE) and out_not_rec:
        concl += (f" Outside the ten, without recovery within 12 epochs: {', '.join(out_not_rec)}; this change is not enough to make "
                  "every model stable.")
        if verdict == VERDICT_POSITIVE:     # NOTE (b): «in either case» anche per un modello fuori dai dieci
            concl += (" This does not rule out a role of early stopping (the lr halved by ReduceLROnPlateau remains the declared "
                      "confounder).")
    B = {"rule": cfg["reading_rule"]["b"], "verdict": verdict, "conclusion": concl, "verification_status": new.status,
         "ten": [label(n) for n in ten], "ten_recovered": sum(1 for n in ten if n in N and N[n]["recovered"]),
         "ten_complete": sum(1 for n in ten if n in N), "ten_without_recovery": ten_not_rec, "ten_degenerate": ten_deg,
         "degenerate_among_complete": deg_new, "without_recovery_outside_ten": out_not_rec,
         "without_recovery_parent": parent_not_rec, "missing_models": missing,
         "ten_first_recovery": {label(n): N[n]["first_recovery"] for n in ten if n in N}}
    # (c) secondari
    rows = []
    for n in names:
        arm, fold, seed = name_fields(n)
        p, q, s = P[n], N.get(n), shared.get(n)
        rows.append({"model": label(n), "arm": arm, "fold": fold, "seed": seed, "in_ten": n in ten,
                     "new_status": "complete" if q else "missing", "parent": p, "new": q,
                     "shared_epochs": p["epochs_run"], "shared_identical": s["identical"] if s else None,
                     "max_abs_diff_shared": {k: s["fields"][k]["max_abs_difference"] for k in HISTORY_KEYS} if s else None})

    def med(group, side, key):
        v = [r[side][key] for r in rows if r[side] is not None and group(r) and r[side][key] is not None]
        return {"models": len(v), "median": statistics.median(v) if v else None, "min": min(v) if v else None, "max": max(v) if v else None}
    groups = (("the ten", lambda r: r["in_ten"]), ("the other 26", lambda r: not r["in_ten"]), ("all 36", lambda r: True))
    C = {"rule": cfg["reading_rule"]["c"],
         "summary": [{"group": gl, "side": side, **{k: med(gf, side, k) for k in ("first_recovery", "restored_epoch", "best_val_loss",
                                                                                  "dev_acc", "dev_auroc", "epochs_at_half", "lr_last")}}
                     for gl, gf in groups for side in ("parent", "new")],
         "recovered": {"parent": sum(1 for n in names if P[n]["recovered"]), "new": sum(1 for n in N if N[n]["recovered"]),
                       "new_complete": len(N)}}
    # (d) test dal pubblico
    if new.status == "PASS" and new.test is not None:
        D = {"label": TEST_LABEL, "rule": cfg["reading_rule"]["d"], "reported": True,
             "source": "public/aggregate.json of the two runs (never the restricted test scores)",
             "primary": [{"metric": k, "parent": par.test["primary"][k], "new": new.test["primary"][k]} for k in PRIMARY_METRICS],
             "rows": [{"comparison": COMPARISON_LABEL[c], "stratum": s, "metric": m, "parent": par.test["rows"].get((c, s, m)),
                       "new": new.test["rows"].get((c, s, m))}
                      for c in COMPARISONS for s in new.test["strata"] for m in METRICS]}
        D["verdicts_changed"] = sum(1 for r in D["rows"] if r["parent"] is None or r["parent"]["verdict"] != r["new"]["verdict"])
    else:
        D = {"label": TEST_LABEL, "rule": cfg["reading_rule"]["d"], "reported": False,
             "note": f"not reported: the verification run is {new.status} (its aggregate, if any, covers only the complete models)"}
    # (e) tempi
    tm = cfg["time"]

    def per_arm(run, which):
        out = []
        for a in cfg["inputs"]["arms"]:
            ms = [run.models[n] for n in names if n in run.models and name_fields(n)[0] == a]
            ep = sum(m["card"]["epochs_run"] for m in ms)
            ts = sum(m["card"]["train_seconds"] for m in ms)
            out.append({"run": which, "arm": ARM_LABEL[a], "models": len(ms), "epochs": ep, "train_seconds": round(ts, 1),
                        "seconds_per_epoch": ts / ep if ep else None})
        return out
    tot_new = sum(m["card"]["train_seconds"] for m in new.models.values())
    E = {"rule": cfg["reading_rule"]["e"], "per_arm": per_arm(par, "parent") + per_arm(new, "new"),
         "new_train_seconds": round(tot_new, 1), "new_wall_seconds": new.st["wall_seconds"], "parent_wall_seconds": par.st["wall_seconds"],
         "estimate_seconds": tm["estimate_seconds"], "cap_seconds": tm["cap_seconds"],
         "wall_minus_estimate_seconds": round(new.st["wall_seconds"] - tm["estimate_seconds"], 1),
         "within_cap": new.st["wall_seconds"] <= tm["cap_seconds"],
         "final_writes_after_cap": new.st.get("final_writes_after_cap"), "stopped": new.st.get("stopped")}
    return {"P": P, "N": N, "shared": shared, "a": A, "b": B, "c": C, "d": D, "e": E, "rows": rows}


def comparison_rows(T: dict) -> list:
    def side(v, keys):
        if v is None:
            return [""] * len(keys)
        return ["" if v[k] is None else (int(v[k]) if isinstance(v[k], bool) else (repr(v[k]) if isinstance(v[k], float) else v[k]))
                for k in keys]
    keys = ("epochs_run", "restored_epoch", "degenerate", "recovered", "first_recovery", "best_val_loss", "dev_acc", "dev_auroc",
            "epochs_at_half", "lr_last")
    out = []
    for r in T["rows"]:
        diffs = [repr(r["max_abs_diff_shared"][k]) if r["max_abs_diff_shared"] else "" for k in HISTORY_KEYS]
        out.append([r["arm"], r["fold"], r["seed"], int(r["in_ten"]), r["new_status"]] + side(r["parent"], keys) + side(r["new"], keys)
                   + [r["shared_epochs"], "" if r["shared_identical"] is None else int(r["shared_identical"])] + diffs)
    return out


def curve_rows(cfg: dict, inp: Inputs, T: dict) -> list:
    crit = cfg["criteria"]
    rv = crit["recovery"]
    rows = []
    for n in model_names(cfg):
        if n not in inp.new.models:
            continue
        m = inp.new.models[n]
        h, v = m["h"], T["N"][n]
        arm, fold, seed = name_fields(n)
        for i in range(m["card"]["epochs_run"]):
            rows.append([arm, fold, seed, int(n in cfg["original_degenerate"]), i + 1] + [repr(h[k][i]) for k in HISTORY_KEYS] +
                        [int(i + 1 == m["card"]["restored_epoch"]), int(is_half(h["val_accuracy"][i], crit)),
                         int(i + 1 <= T["P"][n]["epochs_run"]), int(v["degenerate"]),
                         int(h["val_accuracy"][i] >= rv["val_accuracy_min"] and h["val_loss"][i] < rv["val_loss_max"])])
    return rows


# ============================================================================ testi pubblici

def fmt(v, nd: int = 4) -> str:
    """Formato delle celle: float a 4 decimali, interi e testi come sono, None come «-»."""
    if v is None or v == "":
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def md(s) -> str:
    return fmt(s).replace("*", "\\*")


def table(head, rows) -> str:
    out = ["| " + " | ".join(md(h) for h in head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(md(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def f4(v) -> str:
    return "-" if v is None else f"{float(v):.4f}"


def tv(v) -> str:
    """Verdetto del test con la direzione (se consistente)."""
    if v is None:
        return "-"
    return v["verdict"] + (f" ({v['direction']})" if v["direction"] else "")


def aggregate_md(doc: dict) -> str:
    A, B, C, D, E = (doc["sections"][k] for k in ("a", "b", "c", "d", "e"))
    ins = doc["inputs"]
    L = [f"# Verification run with early_stopping_patience 12 against the parent run ({doc['run_id']})", "",
         f"Parent run {ins['parent_run']['run_id']} (patience {ins['parent_run']['early_stopping_patience']}, {ins['parent_run']['status']}, "
         f"{ins['parent_run']['complete_models']} models); verification run {ins['verification_run']['run_id']} (patience "
         f"{ins['verification_run']['early_stopping_patience']}, {ins['verification_run']['status']}, "
         f"{ins['verification_run']['complete_models']} complete models). Reading rule agreed on 7 November 2026 and fixed before the "
         "run; criteria of the previous package: " + doc["criteria"]["degenerate"]["text"] + "; recovery: " +
         doc["criteria"]["recovery"]["text"] + ". No training, no inference, no test score read. "
         f"Figures: {doc['figures']['note']}.", ""]
    L += ["## (a) Shared histories", "", md(A["rule"]) + ".", "",
          f"Models compared: {A['models_compared']}; identical shared histories: {A['identical']} of {A['models_compared']}; "
          f"valid: {'yes' if A['valid'] else 'no'}.",
          "Largest absolute difference per field: " + ", ".join(f"{k} {v!r}" for k, v in A["max_abs_difference_per_field"].items()) + "."]
    if A["largest_difference"]:
        w = A["largest_difference"]
        L += [f"Largest difference: {md(w['model'])}, epoch {w['epoch']}, field {w['field']}, {w['max_abs_difference']!r}."]
    if A["new_history_shorter_than_shared"]:
        L += ["New history shorter than the shared epochs: " + ", ".join(md(x) for x in A["new_history_shorter_than_shared"]) + "."]
    L += ["", "## (b) Primary outcome", "", md(B["rule"]) + ".", "", f"**Verdict: {B['verdict']}.** {md(B['conclusion'])}", "",
          f"The ten original degenerate trainings: {B['ten_recovered']} of {B['ten_complete']} complete recover within 12 epochs "
          f"(first recovery: " + ", ".join(f"{md(k)} {fmt(v)}" for k, v in B["ten_first_recovery"].items()) + ").",
          "Without recovery among the ten: " + (", ".join(md(x) for x in B["ten_without_recovery"]) or "none") + ".",
          "Degenerate among the complete models of the verification run: " + (", ".join(md(x) for x in B["degenerate_among_complete"]) or "none") + ".",
          "Without recovery outside the ten (listed separately): " + (", ".join(md(x) for x in B["without_recovery_outside_ten"]) or "none") + ".",
          "Without recovery in the parent run: " + (", ".join(md(x) for x in B["without_recovery_parent"]) or "none") + "."]
    if B["missing_models"]:
        L += ["Missing in the verification run: " + ", ".join(md(x) for x in B["missing_models"]) + "."]
    L += ["", "## (c) Secondary outcomes", "", md(C["rule"]) + ".", "",
          f"Models with a recovery: parent {C['recovered']['parent']} of 36, verification run {C['recovered']['new']} of "
          f"{C['recovered']['new_complete']} complete.", ""]
    head = ["model", "ten", "epochs run", "restored", "degenerate", "first recovery", "best val_loss", "dev acc", "dev AUROC",
            "epochs at 0.5", "lr last", "shared identical"]
    trs = []
    for r in doc["rows"]:
        for side in ("parent", "new"):
            v = r[side]
            trs.append([f"{r['model']} ({side})", "yes" if r["in_ten"] else "-"] +
                       (["missing"] + ["-"] * 8 if v is None else
                        [v["epochs_run"], v["restored_epoch"], v["degenerate"], v["first_recovery"], v["best_val_loss"], v["dev_acc"],
                         v["dev_auroc"], v["epochs_at_half"], f"{v['lr_last']:.3g}"]) +
                       [fmt(r["shared_identical"]) if side == "new" else "-"])
    L += [table(head, trs), "", "Medians [min-max] per group:", ""]
    keys = ("first_recovery", "restored_epoch", "best_val_loss", "dev_acc", "dev_auroc", "epochs_at_half", "lr_last")
    L += [table(["group", "run"] + list(keys),
                [[s["group"], s["side"]] + [f"{f4(s[k]['median'])} [{f4(s[k]['min'])}-{f4(s[k]['max'])}] (n {s[k]['models']})" for k in keys]
                 for s in C["summary"]]), ""]
    L += [f"## (d) Test, {D['label']}", "", md(D["rule"]) + ".", ""]
    if D["reported"]:
        L += [f"Source: {D['source']}. Verdicts that differ from the parent run: {D['verdicts_changed']} of {len(D['rows'])}.", "",
              table(["primary (A* - B, stratum 03)", "parent", "verification run"],
                    [[p["metric"], tv(p["parent"]), tv(p["new"])] for p in D["primary"]]), "",
              table(["comparison", "stratum", "metric", "parent verdict", "parent median difference", "new verdict", "new median difference"],
                    [[r["comparison"], r["stratum"], r["metric"], tv(r["parent"]), f4(r["parent"]["median"]) if r["parent"] else "-",
                      tv(r["new"]), f4(r["new"]["median"])] for r in D["rows"]]), ""]
    else:
        L += [D["note"] + ".", ""]
    L += ["## (e) Times", "", md(E["rule"]) + ".", "",
          table(["run", "arm", "models", "epochs", "train_seconds", "s/epoch (weighted)"],
                [[r["run"], r["arm"], r["models"], r["epochs"], r["train_seconds"], r["seconds_per_epoch"]] for r in E["per_arm"]]), "",
          f"Verification run: training {E['new_train_seconds']:.1f} s, wall {E['new_wall_seconds']:.1f} s against the estimate of "
          f"{E['estimate_seconds']:.0f} s (difference {E['wall_minus_estimate_seconds']:+.1f} s), cap {E['cap_seconds']:.0f} s "
          f"(within: {'yes' if E['within_cap'] else 'no'}); final writes after the cap: {fmt(E['final_writes_after_cap'])}; "
          f"stopped: {'no' if not E['stopped'] else 'yes'}. Parent run wall {E['parent_wall_seconds']:.1f} s.", ""]
    L += ["## Declarations", ""] + [f"- {k}: {md(v)}" for k, v in doc["declarations"].items()] + [""]
    return "\n".join(L)


def public_problems(text: str, inp: Inputs, markdown: bool = False) -> list:
    """Contenuto vietato in un file pubblico (come nel runner delle curve): chiavi d'immagine, identificativi di soggetto
    dei dev (parola intera), regole di package_rules, IPv4 non locali, segni di lavoro non finito (solo nei .md).
    Restituisce solo le categorie, mai il testo trovato."""
    bad = []
    if re.search(r"\b(real|attack)/\d{4}_", text):
        bad.append("image keys")
    for s in sorted(inp.subjects):
        if re.search(rf"(?<![\w.]){re.escape(s)}(?![\w.])", text):
            bad.append("subject id")
            break
    for lab, rx in pr.FORBIDDEN_TEXT:
        if rx.search(text):
            bad.append(lab)
    if any(ip not in pr.IP_ALLOW for ip in (m.group(0) for m in pr.IPV4.finditer(text))):
        bad.append("IP address")
    if markdown and pr.MARKERS.search(text):
        bad.append("unfinished-work marker")
    return bad


# ============================================================================ figure

def figure_data(cfg: dict, inp: Inputs, T: dict) -> dict:
    """Dati delle tre figure (solo campi validati: braccio, fold, seed, storie, epoche ripristinate, recupero)."""
    def curve(n, run, view):
        m = run.models[n]
        return {"restored": m["card"]["restored_epoch"], "recovered": view["first_recovery"], "degenerate": view["degenerate"],
                "val_accuracy": m["h"]["val_accuracy"], "val_loss": m["h"]["val_loss"]}

    def pair(n):
        return {"label": label(n), "in_ten": n in cfg["original_degenerate"], "parent": curve(n, inp.parent, T["P"][n]),
                "new": curve(n, inp.new, T["N"][n]) if n in inp.new.models else None, "max_epochs": cfg["recipe"]["max_epochs"]}
    names = model_names(cfg)
    no_rec = [n for n in names if not T["P"][n]["recovered"] or (n in T["N"] and not T["N"][n]["recovered"])]
    return {"ten": [pair(n) for n in cfg["original_degenerate"]], "no_recovery": [pair(n) for n in no_rec],
            "all": [pair(n) for n in names]}


def default_plotter(kind: str, data: dict) -> bytes:
    """Figure pubbliche (matplotlib, Agg): genitore tratteggiato, run di verifica pieno; val_loss in scala log."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    grey, red, blue, green = "#7f7f7f", "#c0392b", "#1f5fa8", "#2e8b57"
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 8})
    items = data["ten"] if kind == "ten" else data["no_recovery"] if kind == "no_recovery" else data["all"]
    keys = ("val_accuracy",) if kind == "val_accuracy_all" else ("val_accuracy", "val_loss")
    per_row = 4 if kind == "val_accuracy_all" else 2
    n = max(len(items), 1)
    rows = (n + per_row - 1) // per_row
    cols = per_row * len(keys)
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.3 * rows + 1.0), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    if not items:
        axes[0][0].text(0.0, 0.5, "no model without recovery in either run", fontsize=10)
    for i, it in enumerate(items):
        r, c0 = divmod(i, per_row)
        for j, k in enumerate(keys):
            ax = axes[r][c0 * len(keys) + j]
            ax.axis("on")
            # nuovo run pieno sotto, genitore tratteggiato sopra (sulle epoche comuni le due curve coincidono)
            for side, ls, col, z in (("new", "-", red if it["new"] and it["new"]["degenerate"] else blue, 2), ("parent", "--", grey, 3)):
                cv = it[side]
                if cv is None:
                    continue
                x = list(range(1, len(cv[k]) + 1))
                ax.plot(x, cv[k], ls=ls, color=col, marker="o" if side == "new" else None, ms=2.2, lw=1.1 if side == "new" else 1.6, zorder=z)
                ax.plot([cv["restored"]], [cv[k][cv["restored"] - 1]], marker="*", ms=8, color=green if side == "new" else grey, ls="none")
            if k == "val_loss":
                ax.set_yscale("log")
                ax.axhline(math.log(2), color=grey, lw=0.6, alpha=0.5)
            else:
                ax.axhline(0.5, color=grey, lw=0.6, alpha=0.5)
                ax.set_ylim(-0.03, 1.05)
            ax.set_xlim(0.5, it["max_epochs"] + 0.5)
            ax.set_xticks([1, 4, 8, 12] if it["max_epochs"] == 12 else range(1, it["max_epochs"] + 1, 2))
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2)
            nr = it["new"]["recovered"] if it["new"] else None
            ax.set_title(f"{it['label']}{' (ten)' if it['in_ten'] and kind != 'ten' else ''}\n{k}; recovery parent "
                         f"{it['parent']['recovered'] or '-'}, new {nr or '-'}", fontsize=7)
    handles = [Line2D([], [], color=grey, ls="--", label="parent run (patience 4)"),
               Line2D([], [], color=blue, ls="-", label="verification run (patience 12)"),
               Line2D([], [], color=red, ls="-", label="verification run, degenerate"),
               Line2D([], [], color=green, marker="*", ls="none", label="restored epoch (verification run)"),
               Line2D([], [], color=grey, marker="*", ls="none", label="restored epoch (parent run)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    title = {"ten": "The ten original degenerate trainings: dev accuracy and dev loss (log scale) per epoch",
             "no_recovery": "Trainings without recovery in at least one of the two runs: dev accuracy and dev loss (log scale)",
             "val_accuracy_all": "Dev accuracy per epoch, all 36 trainings"}[kind]
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, metadata={"Software": None})
    plt.close(fig)
    return buf.getvalue()


def figure_bytes(data: dict) -> tuple:
    """({nome: byte PNG}, nota): matplotlib assente -> salto dichiarato, mai un errore."""
    kinds = {"figure_ten.png": "ten", "figure_no_recovery.png": "no_recovery", "figure_val_accuracy_all.png": "val_accuracy_all"}
    try:
        pngs = {n: (PLOTTER or default_plotter)(k, data) for n, k in kinds.items()}
    except ImportError as e:
        return {}, f"skipped: missing Python package {e.name!r}"
    for n, b in pngs.items():
        if not isinstance(b, bytes) or not b:
            fail(f"figure {n}: the plotter returned no PNG bytes")
    return pngs, f"{', '.join(pngs)} written"


# ============================================================================ uscite

def environment() -> dict:
    out = {"python": sys.version.split()[0]}
    try:
        import matplotlib
        out["matplotlib"] = str(matplotlib.__version__)
    except ImportError:
        out["matplotlib"] = "absent"
    return out


def code_sha256() -> dict:
    return {f: sha256_file(CODE_ROOT / f) for f in CODE_FILES if (CODE_ROOT / f).is_file()}


def build_public(cfg: dict, cfg_sha: str, inp: Inputs, head: dict) -> tuple:
    """File pubblici (testi e PNG), scansionati prima della scrittura; nessuna data né tempo d'esecuzione di questo comando."""
    T = analyse(cfg, inp)
    fig_data = figure_data(cfg, inp, T)
    pngs, note = figure_bytes(fig_data)
    doc = {"run_id": head["run_id"], "commit": head["commit"], "dirty": head["dirty"], "config_sha256": cfg_sha,
           "delivered_configuration": cfg_sha == CONFIG_SHA256, "request": cfg["request"], "task": cfg["task"],
           "reading_rule": cfg["reading_rule"], "criteria": cfg["criteria"], "recipe": cfg["recipe"],
           "original_degenerate": [label(n) for n in cfg["original_degenerate"]],
           "models_without_recovery_parent": [label(n) for n in cfg["models_without_recovery_parent"]],
           "inputs": {g: {**inp.checks[g], "folder": cfg["inputs"][g]["folder"]} for g in RUNS},
           "verdict": T["b"]["verdict"], "sections": {k: T[k] for k in ("a", "b", "c", "d", "e")},
           "rows": [{k: v for k, v in r.items() if k not in ("arm", "fold", "seed")} for r in T["rows"]],
           "restricted": RESTRICTED_NOTE, "figures": {"files": sorted(pngs), "note": note}, "declarations": cfg["declarations"],
           "environment": environment(), "code_sha256": code_sha256()}
    public = {"comparison.csv": csv_text(COMPARISON_COLUMNS, comparison_rows(T)), "curves.csv": csv_text(CURVE_COLUMNS, curve_rows(cfg, inp, T)),
              "aggregate.json": json.dumps(doc, indent=1) + "\n", "aggregate.md": aggregate_md(doc)}
    scan = dict(public)
    scan["figure labels"] = json.dumps(fig_data)
    for name, text in scan.items():
        if probs := public_problems(text, inp, name.endswith(".md")):
            fail(f"public file {name} refused: " + "; ".join(probs))
    public.update(pngs)
    public["SHA256SUMS"] = sums_text(public)
    return public, doc


def repo_state(root: Path) -> dict:
    if REPO_STATE is not None:
        return REPO_STATE(root)
    try:
        c = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], cwd=root, capture_output=True, text=True, timeout=30)
        s = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        fail(f"git not available: {type(e).__name__}")
    if c.returncode or s.returncode or not c.stdout.strip():
        fail("git state not readable: the run id needs the commit")
    return {"commit": c.stdout.strip(), "dirty": bool(s.stdout.strip())}


def run_id_for(date: str, state: dict) -> str:
    if not re.fullmatch(r"\d{8}", date or ""):
        fail("--date must be YYYYMMDD")
    rid = f"{date}-PATVER-{state['commit']}" + ("-dev" if state["dirty"] else "")
    if not RUN_ID_RE.match(rid):
        fail(f"run id {rid} does not follow YYYYMMDD-PATVER-<commit>[-dev]")
    return rid


def protected_extra(root: Path) -> list:
    return [root / x for x in EXTRA_PROTECTED]


def input_paths(cfg: dict, base: Path, cfg_p: Path) -> dict:
    return {"the configuration": cfg_p, **{f"the {g.replace('_', ' ')}": rel_path(base, cfg["inputs"][g]["folder"]) for g in RUNS}}


def run_dir_problem(run: Path, root: Path, inputs: dict, out_root: Path) -> str:
    """Motivo per cui la cartella del run non è ammessa ("" se lo è), prima di qualunque modifica."""
    if out_root.is_symlink():
        return "the output root is a symbolic link"
    if (run / STATUS).exists() or (run / STATUS).is_symlink():
        return f"the run folder {run.name} already holds {STATUS}: every attempt keeps its outcome (archive that folder first)"
    for lab, p in inputs.items():
        if pg.within(p, run) or (Path(p).is_dir() and pg.within(run, p)):
            return f"{lab} {Path(p).name} is inside the run folder, or contains it: inputs and outputs must be separate"
    msg = mw.output_problem(run, root, pg.BASE_PROTECTED, [root / DEFAULT_OUT_ROOT], protected_extra(root), inputs, [STATUS, LOG], [run / PUBLIC])
    if msg:
        return msg
    if msg := mw.output_problem(run / PUBLIC, root, pg.BASE_PROTECTED, [root / DEFAULT_OUT_ROOT], protected_extra(root), inputs, list(PUBLIC_OUTPUTS)):
        return msg
    for sub, names in (("", [STATUS, LOG, PUBLIC]), (PUBLIC, PUBLIC_OUTPUTS)):
        d = run / sub if sub else run
        if d.is_dir():
            for e in sorted(os.listdir(d)):
                if not any(mw.owned_name(e, n) for n in names):
                    return f"{(Path(sub) / e).as_posix()} in the run folder is not an output of this command: refused, nothing removed"
    return ""


def write_status(run: Path, doc: dict) -> None:
    mw.write_all(run, {STATUS: json.dumps(doc, indent=1) + "\n"})


def invalidate_outputs(run: Path) -> list:
    d = run / PUBLIC
    return [f"{PUBLIC}/{n}" for n in mw.invalidate_each(d, list(PUBLIC_OUTPUTS), None, FAIL_DOC)] if d.is_dir() else []


def write_binary(d: Path, files: dict) -> None:
    tmps, done = {}, []
    try:
        for n in files:
            q = d / n
            if q.is_symlink() or (q.exists() and not q.is_file()):
                fail(f"{n} is a link or not a regular file: refused")
        for n, data in files.items():
            fd, tp = tempfile.mkstemp(prefix=f".{n}.", suffix=".tmp", dir=d)
            tmps[n] = Path(tp)
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o666 & ~mw._UMASK)            # stessi permessi dei testi (write_all)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
        for n, tp in tmps.items():
            os.replace(tp, d / n)
            done.append(n)
    except BaseException:
        for tp in tmps.values():
            tp.unlink(missing_ok=True)
        for n in done:
            (d / n).unlink(missing_ok=True)
        raise


def write_public(d: Path, files: dict) -> None:
    """Testi con write_all (newline ""), PNG con temporanei e rinomine; SHA256SUMS per ultimo; nulla resta su errore."""
    texts = {n: v for n, v in files.items() if isinstance(v, str) and n != "SHA256SUMS"}
    blobs = {n: v for n, v in files.items() if isinstance(v, bytes)}
    mw.write_all(d, texts, newline="")
    try:
        if blobs:
            write_binary(d, blobs)
        mw.write_all(d, {"SHA256SUMS": files["SHA256SUMS"]}, newline="")
    except BaseException:
        for n in list(texts) + list(blobs):
            (d / n).unlink(missing_ok=True)
        raise


class Tee:
    """Copia di stdout/stderr nel log del run (aperto con O_NOFOLLOW: un collegamento è rifiutato)."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        try:
            self.fh.write(s)
            self.fh.flush()
        except (OSError, ValueError):
            pass
        return len(s)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, n):
        return getattr(self.stream, n)


def open_log(run: Path):
    p = run / LOG
    if p.is_symlink() or (p.exists() and not p.is_file()):
        fail(f"{LOG} is a link or not a regular file: refused")
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o644)
    fh = os.fdopen(fd, "w", encoding="utf-8")
    fh.write(f"===== {now()} run =====\n")
    saved = (sys.stdout, sys.stderr)
    sys.stdout, sys.stderr = Tee(sys.stdout, fh), Tee(sys.stderr, fh)
    return fh, saved


def close_log(handle) -> None:
    fh, (out, err) = handle
    sys.stdout, sys.stderr = out, err
    fh.close()


def resolve(a) -> tuple:
    base = Path(a.input_root).resolve() if a.input_root else ROOT
    cfg_p = Path(a.config) if Path(a.config).is_absolute() else ROOT / a.config
    return base, cfg_p


def summary_line(inp: Inputs) -> str:
    c = inp.checks
    p, v = c["parent_run"], c["verification_run"]
    left = f"parent run {p['run_id']} ({p['status']}, {p['complete_models']} cards, dev scores and thresholds; early stopping replay {p['early_stopping_replay']})"
    if inp.new is None:
        return left + f"; verification run {v['run_id']} {v['status']}"
    return left + (f"; verification run {v['run_id']} ({v['status']}, {v['complete_models']} complete models, {v['missing_models']} missing; "
                   f"early stopping replay {v['early_stopping_replay']}; test section: {v['test_section']})")


def cmd_check_inputs(a) -> int:
    base, cfg_p = resolve(a)
    cfg, cfg_sha = load_config(cfg_p)
    inp = load_inputs(cfg, base, need_verification=False)
    print(f"CHECK-INPUTS: configuration {cfg_sha[:12]} (delivered: {cfg_sha == CONFIG_SHA256}), {summary_line(inp)}; nothing written")
    if inp.new is None:
        print("RESULT: PASS (check-inputs, structural only: the anchors of the verification run are null, to be filled after the run)")
    else:
        print("RESULT: PASS (check-inputs)")
    return 0


def cmd_run(a) -> int:
    if not a.i_am_authorized:
        fail("run reads the restricted training cards, dev scores and thresholds of two runs: pass --i-am-authorized (comparison "
             "agreed with the coordinator on 7 November 2026)")
    base, cfg_p = resolve(a)
    cfg, cfg_sha = load_config(cfg_p)
    if not anchors_filled(cfg):
        fail("the anchors of the verification run are null (to be filled after the run): run refused")
    state = repo_state(ROOT)
    rid = run_id_for(a.date or time.strftime("%Y%m%d"), state)
    out_root = Path(a.out_root) if Path(a.out_root).is_absolute() else ROOT / a.out_root
    run = out_root / rid
    inputs = input_paths(cfg, base, cfg_p)
    if msg := run_dir_problem(run, ROOT, inputs, out_root):
        fail(f"refusing the run folder: {msg}")
    run.mkdir(parents=True, exist_ok=True)
    head = {"run_id": rid, "commit": state["commit"], "dirty": state["dirty"], "config_sha256": cfg_sha,
            "delivered_configuration": cfg_sha == CONFIG_SHA256, "command": "run", "restricted": RESTRICTED_NOTE}
    log = None
    t0 = time.time()
    try:
        write_status(run, {**head, "status": "RUNNING", "passed": False, "started": now(),
                           "note": "RUNNING without a live process means a failed run"})
        log = open_log(run)
        left = invalidate_outputs(run)
        if left:
            fail(f"previous outputs could not be removed: {', '.join(left)}")
        print(f"== NUAA patience 12 verification {rid}: configuration {cfg_sha[:12]} (delivered: {head['delivered_configuration']})", flush=True)
        inp = load_inputs(cfg, base)
        print(f"   check-inputs PASS: {summary_line(inp)}", flush=True)
        public, doc = build_public(cfg, cfg_sha, inp, head)
        write_public(run / PUBLIC, public)
        s = doc["sections"]
        print(f"(a) shared histories identical: {s['a']['identical']}/{s['a']['models_compared']} (valid: {s['a']['valid']})")
        print(f"(b) verdict: {s['b']['verdict']}; ten recovered {s['b']['ten_recovered']}/{s['b']['ten_complete']}")
        print(f"figures: {doc['figures']['note']}")
        print(f"run folder: {run.name} (public/ only; no restricted outputs)")
        print("RESULT: PASS (run)")
        close_log(log)
        log = None
        write_status(run, {**head, "status": "PASS", "passed": True, "finished": now(), "wall_seconds": round(time.time() - t0, 1),
                           "verdict": doc["verdict"], "verification_status": s["b"]["verification_status"],
                           "figures": doc["figures"]["note"], "public_files": len(public) - 1,
                           "public_sums_sha256": sha256_bytes(public["SHA256SUMS"].encode("utf-8")),
                           "run_log_sha256": sha256_file(run / LOG),
                           "timestamps": "status.json and run.log only; the public files hold no date or run time of this command"})
    except BaseException as e:
        reason = str(e) if isinstance(e, Failure) else f"unexpected error {type(e).__name__}: {e}"
        left = invalidate_outputs(run)
        if log is not None:
            log[0].write(f"ERROR: {reason}\nRESULT: FAIL (run)\n")
            close_log(log)
        try:
            write_status(run, {**head, "status": "FAIL", "passed": False, "reason": reason, "finished": now(),
                               "outputs": "removed" if not left else f"left behind: {', '.join(left)}",
                               "run_log_sha256": sha256_file(run / LOG) if (run / LOG).is_file() else "none"})
        except Exception:  # noqa: BLE001
            print("ERROR: status.json could not be written", file=sys.stderr)
        raise
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="NUAA: comparison of the verification run (patience 12) with the parent run (patience 4)")
    ap.add_argument("cmd", choices=["check-inputs", "run"])
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="configuration JSON (relative to the repository root)")
    ap.add_argument("--input-root", default=None, help="base folder of the input paths of the configuration (default: repository root)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="folder that receives <run-id>/ (default results/patienceverify)")
    ap.add_argument("--date", default=None, help="YYYYMMDD of the run id (default: today)")
    ap.add_argument("--i-am-authorized", action="store_true", help="required by run: reads the restricted training cards and dev scores")
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
    except SystemExit as e:                     # fail() di m2_writers: già stampato
        print(f"RESULT: FAIL ({a.cmd})")
        return e.code if isinstance(e.code, int) and e.code else 1
    except Exception as e:  # noqa: BLE001  (errore imprevisto: le uscite sono già state tolte da cmd_run)
        print(f"ERROR: unexpected error {type(e).__name__}: {e}", file=sys.stderr)
        print(f"RESULT: FAIL ({a.cmd})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
