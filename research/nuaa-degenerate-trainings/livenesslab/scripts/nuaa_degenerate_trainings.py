"""
Training degeneri del run di ablazione 20261106-SESSABLRUN-d999458 studiati dalle curve train/dev già registrate,
chiesto dal coordinatore il 6/11/2026 sulla scheda C2: «approfondire i 10 training degeneri usando le curve train/dev
già disponibili», «Li confronti con alcuni training riusciti e mostri pochi esempi rappresentativi, distinguendo ciò
che osserviamo dalle possibili spiegazioni» (le spiegazioni non sono calcolate qui).

Solo lettura di uscite già scritte da altri run: nessun training, nessuna inferenza, nessuno score né metrica del test
letti. Lo studio è descrittivo e i criteri (degenere, saturazione, recupero) sono stati fissati dopo il run.

Ingressi (fissati nella configurazione per SHA-256; ogni file anche contro il SHA256SUMS della sua cartella):
  - run di ablazione (36 modelli): status.json PASS ancorato, SHA256SUMS pubblico e riservato ancorati, run.log
    ancorato (solo la riga del tempo totale); per modello <arm>_fold<k>_seed<s>_card.json (storia per epoca),
    _dev_scores.csv (label e score_attack per le misure; subject_id solo per contare i soggetti del dev e per la
    scansione del pubblico, mai scritto; arm, fold, train_seed, role, checkpoint_sha256, sample_id, file_sha256,
    prediction_05, prediction_dev, threshold_dev e threshold_sha256 solo come controlli per riga) e _threshold.json
    (frozen_at per i tempi; arm, fold, train_seed, checkpoint_sha256, value e dev_scores_sha256 come controlli);
  - run dei fold (12 modelli): stesse ancore; fold<k>_seed<s>_card.json, _dev_scores.csv e _threshold.json (solo
    controlli di coerenza);
  - tentativo fallito 20261106-SESSABLRUN-3b4021f: cartella FAIL, le sei schede complete ancorate una per una
    (configurazione, SHA256SUMS riservato della cartella, status.json FAIL) e usate solo per il confronto di
    determinismo.

Controlli su ogni scheda: chiavi a insieme chiuso; ricetta uguale alla configurazione; storie con le cinque chiavi,
lunghezza epochs_run, numeri finiti; restored_epoch = argmin del val_loss (primo minimo); best_val_loss e
val_acc_at_best letti all'epoca ripristinata; la simulazione esatta dell'EarlyStopping di Keras 2.15 (min_delta 0,
miglioramento stretto, restore_best_weights, simulate_es) riproduce epochs_run, restored_epoch e restored_by;
weights_file e weights_sha256 elencati nel SHA256SUMS. Sul dev_scores, riga per riga (load_dev): ruolo, braccio, fold,
seed e checkpoint della scheda; etichetta in {0, 1}, score in [0, 1]; prediction_05, prediction_dev, threshold_dev e
threshold_sha256 coerenti con la soglia congelata e con la regola «attacco se score > soglia» (pareggio = bona fide);
sample_id e file_sha256 unici; elenco dei sample_id uguale a dev_keys_sha256 della scheda e score uguali a
dev_scores_sha256 della soglia (stesse formule dei runner); righe = n_dev; dev bilanciato; accuratezza alla soglia 0,5
uguale a val_acc_at_best entro la tolleranza; stesse righe (chiave, file, etichetta, soggetto, nello stesso ordine) per
tutti i modelli di un fold.

Uscite (cartella del run <out-root>/YYYYMMDD-DEGEN-<commit>[-dev], -dev se l'albero git non è pulito):
  status.json                RUNNING all'avvio, poi PASS o FAIL (unico stato corrente)
  run.log                    copia di stdout e stderr
  public/                    curves.csv, models.csv, aggregate.json, aggregate.md (tabelle T1-T13),
                             figure_examples.png, figure_val_accuracy_all.png, figure_val_loss_all.png (solo con
                             matplotlib; altrimenti salto dichiarato in aggregate e status), SHA256SUMS
Nessuna cartella restricted/: lo studio non scrive uscite riservate (lo dichiara lo status). I file pubblici non
contengono date né tempi di esecuzione (solo status.json e run.log): due run con lo stesso id danno file pubblici
uguali byte per byte. Dentro il repository la cartella del run è ammessa solo sotto results/degenerate/.

Scansione del pubblico (public_problems), prima di ogni scrittura, su tutti i testi pubblici e sulle etichette delle
figure: chiavi d'immagine, identificativi di soggetto letti dai dev_scores (regola degli altri runner: l'identificativo
come parola intera, non preceduto né seguito da lettera, cifra, trattino basso o punto, così «0.0030» non è
«0030»), regole di package_rules (percorsi assoluti, credenziali, modelli locali), IPv4 non locali, segni di lavoro
non finito nei .md. Un risultato qualunque: nessun pubblico e status FAIL.

  .venv/bin/python -I scripts/nuaa_degenerate_trainings.py check-inputs               # solo verifica, nulla scritto
  .venv/bin/python -I scripts/nuaa_degenerate_trainings.py run --i-am-authorized      # tabelle, figure e uscite

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

DEFAULT_CONFIG = "delivery/degenerate/nuaa_degenerate_trainings.json"
DEFAULT_OUT_ROOT = "results/degenerate"
CONFIG_SCHEMA = "nuaa-degenerate-trainings-config/1"
CONFIG_SHA256 = "8edda999f3f47f84e90ef32d4c1166db51d3fc9650747f58028cfdda7d6195e4"   # configurazione consegnata, fissata prima del run
RUN_ID_RE = re.compile(r"^\d{8}-DEGEN-[0-9a-f]{7,40}(-dev)?$")
TOP_KEYS = ("schema", "task", "request", "inputs", "recipe", "criteria", "alternative_rules", "examples", "time_scenarios",
            "declarations", "outputs")
GROUPS = ("ablation_run", "folds_run", "failed_attempt")
HISTORY_KEYS = ("loss", "accuracy", "val_loss", "val_accuracy", "lr")
BASE_CARD_KEYS = ("arch", "batch", "best_val_loss", "class_weight", "cpu_nondeterminism", "dev_keys_sha256", "device",
                  "early_stopping_patience", "epochs_run", "fitting_keys_sha256", "fold", "history", "lr", "max_epochs", "n_dev",
                  "n_fitting", "restored_by", "restored_epoch", "saved", "test_evaluation", "test_scored_after_threshold",
                  "train_seconds", "train_seed", "val_acc_at_best", "val_loss_recomputed", "visible_devices", "weights_file",
                  "weights_sha256")
CARD_KEYS = {"ablation_run": tuple(sorted(BASE_CARD_KEYS + ("arm", "val_loss_check", "val_loss_recomputed_clipped"))),
             "folds_run": tuple(sorted(BASE_CARD_KEYS)),
             "failed_attempt": tuple(sorted(BASE_CARD_KEYS + ("arm",)))}
DEV_COMMON = ("fold", "train_seed", "role", "sample_id", "subject_id", "session_id", "label", "score_attack", "prediction_05",
              "prediction_dev", "threshold_dev", "file_sha256", "checkpoint_sha256", "threshold_sha256")
DEV_COLUMNS = {"ablation_run": ("arm",) + DEV_COMMON, "folds_run": DEV_COMMON}
RESTORED_BY = {True: "early stopping", False: "end of epochs (set_weights)"}   # fermato dalla regola / storia completa
FOLDS_ARM = "folds"                 # braccio dei modelli del run dei fold (come nel prototipo)
DECISION = 0.5                      # Keras accuracy al checkpoint: attacco se score_attack > 0,5 (pareggio = bona fide)
TIME_FMT = "%Y-%m-%dT%H:%M:%S%z"
WALL_RE = re.compile(r"^run folder: .*\bwall (\d+(?:\.\d+)?) s\b")
WALL_TOL = 1.0                      # tempo del run.log (secondi interi) contro wall_seconds dello status.json
SELECTIONS = ("all", "degenerate", "stopped before max_epochs")
DECLARATION_FLAGS = ("no_training", "no_inference", "no_test_scores_read", "descriptive", "criteria_defined_after_the_run")
DECLARATION_TEXTS = ("dev_columns_read", "observations_not_explanations", "limits")
EPOCH_COLUMNS = ("run", "arm", "fold", "seed", "n_fitting", "steps_per_epoch", "epoch", "loss", "accuracy", "val_loss",
                 "val_accuracy", "lr", "is_best", "degenerate", "run_id", "stopped_epoch")
MODEL_COLUMNS = ("run", "arm", "fold", "seed", "n_fitting", "steps_per_epoch", "epochs_run", "restored_epoch", "restored_by",
                 "best_val_loss", "val_acc_at_best", "degenerate", "first_recovery_epoch", "epochs_with_val_acc_0.5",
                 "first_val_loss_below_epoch1", "min_val_loss_after_epoch_2", "last_val_accuracy", "last_val_loss",
                 "lr_at_last_epoch", "train_seconds", "seconds_per_epoch", "run_id")
NOT_DEG_LABEL = "not degenerate (by the criterion)"   # non uno stato di riuscita: fra i non degeneri c'è chi non recupera
FIGURES = ("figure_examples.png", "figure_val_accuracy_all.png", "figure_val_loss_all.png")
CODE_FILES = ("scripts/nuaa_degenerate_trainings.py", "scripts/m2_writers.py", "scripts/path_guard.py", "scripts/package_rules.py")
STATUS = "status.json"
LOG = "run.log"
PUBLIC = "public"
PUBLIC_OUTPUTS = ("curves.csv", "models.csv", "aggregate.json", "aggregate.md") + FIGURES + ("SHA256SUMS",)
FAIL_DOC = {"stage": "invalidation-failed", "note": "this output could not be removed after a failure: no verdict is valid"}
EXTRA_PROTECTED = ("models", "results/c1", "results/m1", "results/m2", "results/m2_rgb", "results/eval", "results/n1",
                   "results/n2", "results/c1crop", "results/folds", "results/foldsrun", "results/foldscompare", "results/stepzero",
                   "results/sessionablation", "results/sessionablationrun", "export")
RESTRICTED_NOTE = "none: this study writes no restricted output (the per-frame dev scores are read in place and never copied)"

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
    inputs = _exact(cfg["inputs"], GROUPS, "inputs")
    run_keys = ["dev_rows", "folder", "folds", "label", "models", "public_sums_sha256", "restricted_sums_sha256", "run_id",
                "run_log_sha256", "seeds", "status_sha256"]
    for g in ("ablation_run", "folds_run"):
        spec = _exact(inputs[g], run_keys + (["arms"] if g == "ablation_run" else []), f"inputs.{g}")
        for k in ("folder", "label", "run_id"):
            _text(spec[k], f"inputs.{g}.{k}")
        for k in ("public_sums_sha256", "restricted_sums_sha256", "run_log_sha256", "status_sha256"):
            if not _hex(spec[k]):
                fail(f"configuration: inputs.{g}.{k} must be a SHA-256")
        _int_list(spec["folds"], f"inputs.{g}.folds", 1)
        _int_list(spec["seeds"], f"inputs.{g}.seeds", 0)
        if g == "ablation_run":
            if not isinstance(spec["arms"], list) or not spec["arms"] or len(set(spec["arms"])) != len(spec["arms"]) or \
                    not all(isinstance(a, str) and re.fullmatch(r"[a-z][a-z0-9_]*", a) and a != FOLDS_ARM for a in spec["arms"]):
                fail("configuration: inputs.ablation_run.arms must list distinct arm names")
        n = len(spec["folds"]) * len(spec["seeds"]) * (len(spec["arms"]) if g == "ablation_run" else 1)
        if not is_int(spec["models"]) or spec["models"] != n:
            fail(f"configuration: inputs.{g}.models must be {n} (arms x folds x seeds)")
        if not is_int(spec["dev_rows"]) or spec["dev_rows"] < 2 or spec["dev_rows"] % 2:
            fail(f"configuration: inputs.{g}.dev_rows must be a positive even integer (balanced dev)")
    fa = _exact(inputs["failed_attempt"], ["cards", "folder", "label", "note", "restricted_sums_sha256", "run_id", "status_sha256"],
                "inputs.failed_attempt")
    for k in ("folder", "label", "note", "run_id"):
        _text(fa[k], f"inputs.failed_attempt.{k}")
    for k in ("restricted_sums_sha256", "status_sha256"):
        if not _hex(fa[k]):
            fail(f"configuration: inputs.failed_attempt.{k} must be a SHA-256")
    if not isinstance(fa["cards"], dict) or not fa["cards"] or \
            not all(isinstance(n, str) and n.endswith("_card.json") and _hex(h) for n, h in fa["cards"].items()):
        fail("configuration: inputs.failed_attempt.cards must map <model>_card.json names to SHA-256 values")
    labels = [inputs[g]["label"] for g in GROUPS]
    if len(set(labels)) != len(labels):
        fail("configuration: the input labels must be distinct")
    rc = _exact(cfg["recipe"], ["batch", "bn_momentum", "early_stopping_patience", "lr", "max_epochs", "min_delta", "monitor",
                                "restore_best_weights", "source"], "recipe")
    for k in ("batch", "early_stopping_patience", "max_epochs"):
        if not is_int(rc[k]) or rc[k] < 1:
            fail(f"configuration: recipe.{k} must be a positive integer")
    if not _number(rc["lr"]) or rc["lr"] <= 0 or not _number(rc["bn_momentum"]) or not 0 < rc["bn_momentum"] < 1:
        fail("configuration: recipe.lr must be positive and recipe.bn_momentum in (0, 1)")
    if rc["min_delta"] != 0 or rc["monitor"] != "val_loss" or rc["restore_best_weights"] is not True:
        fail("configuration: the recipe of the runs is monitor val_loss, min_delta 0, restore_best_weights true")
    _text(rc["source"], "recipe.source")
    cr = _exact(cfg["criteria"], ["below_chance_val_accuracy", "bn_residual_epochs", "boundary_restored_epoch_max", "checkpoint_agreement_tolerance",
                                  "defined_after_the_run", "degenerate", "late_recovery_after_epoch", "pair_train_epochs", "recovery",
                                  "saturation", "train_accuracy_reference", "train_side_epochs"], "criteria")
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
    rv = _exact(cr["recovery"], ["sensitivity", "text", "val_accuracy_min", "val_loss_max"], "criteria.recovery")
    se = _exact(rv["sensitivity"], ["val_accuracy_min", "val_loss_max"], "criteria.recovery.sensitivity")
    for k in ("val_accuracy_min", "val_loss_max"):
        if not _number(rv[k]) or rv[k] <= 0:
            fail(f"configuration: criteria.recovery.{k} must be a positive number")
        if not isinstance(se[k], list) or not se[k] or not all(_number(x) and x > 0 for x in se[k]) or rv[k] not in se[k] \
                or se[k] != sorted(set(se[k])):
            fail(f"configuration: criteria.recovery.sensitivity.{k} must be increasing positive numbers including the declared value")
    _text(rv["text"], "criteria.recovery.text")
    for k in ("checkpoint_agreement_tolerance",):
        if not _number(cr[k]) or not 0 < cr[k] < 1e-3:
            fail(f"configuration: criteria.{k} must be in (0, 1e-3)")
    for k in ("below_chance_val_accuracy", "train_accuracy_reference"):
        if not _number(cr[k]) or not 0 < cr[k] < 1:
            fail(f"configuration: criteria.{k} must be in (0, 1)")
    for k in ("bn_residual_epochs", "boundary_restored_epoch_max", "late_recovery_after_epoch", "pair_train_epochs", "train_side_epochs"):
        if not is_int(cr[k]) or not 1 <= cr[k] <= rc["max_epochs"]:
            fail(f"configuration: criteria.{k} must be an epoch between 1 and recipe.max_epochs")
    if cr["defined_after_the_run"] is not True:
        fail("configuration: criteria.defined_after_the_run must be true")
    rules = cfg["alternative_rules"]
    if not isinstance(rules, list) or not rules:
        fail("configuration: alternative_rules must be a non-empty list")
    for i, r in enumerate(rules):
        _exact(r, ["label", "monitor", "patience", "start_from_epoch"], f"alternative_rules[{i}]")
        _text(r["label"], f"alternative_rules[{i}].label")
        if r["monitor"] not in ("val_loss", "val_accuracy") or not is_int(r["patience"]) or r["patience"] < 1 or \
                not is_int(r["start_from_epoch"]) or r["start_from_epoch"] < 0:
            fail(f"configuration: alternative_rules[{i}] needs monitor val_loss or val_accuracy, patience >= 1, start_from_epoch >= 0")
    if (rules[0]["monitor"], rules[0]["patience"], rules[0]["start_from_epoch"]) != ("val_loss", rc["early_stopping_patience"], 0):
        fail("configuration: alternative_rules[0] must be the rule of the run (val_loss, the recipe patience, start 0)")
    if len({r["label"] for r in rules}) != len(rules):
        fail("configuration: the labels of alternative_rules must be distinct")
    ex = cfg["examples"]
    if not isinstance(ex, list) or not ex:
        fail("configuration: examples must be a non-empty list")
    for i, e in enumerate(ex):
        _exact(e, ["arm", "fold", "run", "seed", "why"], f"examples[{i}]")
        _text(e["why"], f"examples[{i}].why")
        if e["run"] not in (inputs["ablation_run"]["label"], inputs["folds_run"]["label"]):
            fail(f"configuration: examples[{i}].run must be the label of the ablation run or of the folds run")
    ts = cfg["time_scenarios"]
    if not isinstance(ts, list) or not ts:
        fail("configuration: time_scenarios must be a non-empty list")
    for i, s in enumerate(ts):
        _exact(s, ["epochs", "label", "selection", "with_overhead"], f"time_scenarios[{i}]")
        _text(s["label"], f"time_scenarios[{i}].label")
        if not is_int(s["epochs"]) or s["epochs"] < 1 or not isinstance(s["with_overhead"], bool):
            fail(f"configuration: time_scenarios[{i}] needs epochs >= 1 and with_overhead true or false")
        sel = s["selection"]
        arms = sel[len("arms: "):].split(", ") if isinstance(sel, str) and sel.startswith("arms: ") else None
        if sel not in SELECTIONS and not (arms and set(arms) <= set(inputs["ablation_run"]["arms"]) and len(set(arms)) == len(arms)):
            fail(f"configuration: time_scenarios[{i}].selection must be one of {', '.join(SELECTIONS)} or 'arms: <arm>, <arm>'")
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
    """EarlyStopping di Keras 2.15 (min_delta 0, baseline None, restore_best_weights) rigiocato sulla storia registrata,
    come keras/src/callbacks.py: prima di start_from_epoch nulla; wait += 1; un miglioramento stretto (val_loss minore,
    val_accuracy maggiore) aggiorna il migliore e azzera wait; altrimenti con wait >= patience e epoca > 0 si ferma e
    ripristina il migliore. La traiettoria dei pesi non dipende dall'EarlyStopping finché non ferma (ReduceLROnPlateau
    resta uguale), quindi il risultato è esatto se la regola si ferma entro le epoche registrate o se la storia arriva a
    n_max; altrimenti None (non determinabile). early: True se la regola ha fermato il training."""
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
            return {"stop": e + 1, "restored": best_ep + 1, "early": True, "val_acc": h["val_accuracy"][best_ep],
                    "val_loss": h["val_loss"][best_ep]}
    if len(vals) == n_max and best_ep is not None:
        return {"stop": n_max, "restored": best_ep + 1, "early": False, "val_acc": h["val_accuracy"][best_ep],
                "val_loss": h["val_loss"][best_ep]}
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


def dev_diag(labels: list, scores: list, n_subjects: int) -> dict:
    """Diagnostica del checkpoint sul dev: accuratezza alla soglia di Keras (attacco se score > 0,5), quote predette
    come attacco (tutte, per classe), score medi per classe, AUROC, score distinti."""
    bf = [s for s, y in zip(scores, labels) if y == 0]
    at = [s for s, y in zip(scores, labels) if y == 1]
    return {"n": len(scores), "n_subjects": n_subjects, "acc_05": sum(1 for s, y in zip(scores, labels) if (s > DECISION) == (y == 1)) / len(scores),
            "pred_attack_05": sum(1 for s in scores if s > DECISION) / len(scores),
            "pred_attack_05_bona_fide": sum(1 for s in bf if s > DECISION) / len(bf),
            "pred_attack_05_attack": sum(1 for s in at if s > DECISION) / len(at),
            "mean_bona_fide": statistics.fmean(bf), "mean_attack": statistics.fmean(at), "auroc": auroc(labels, scores),
            "distinct": len(set(scores))}


# ============================================================================ ingressi

class Inputs:
    """Modelli (scheda validata, storia, dev), tempi e l'esito di ogni controllo (checks)."""

    def __init__(self):
        self.models = {g: [] for g in GROUPS}   # gruppo -> modelli in ordine di nome del file
        self.subjects = set()                   # identificativi di soggetto dei dev (solo per la scansione)
        self.wall = None                        # tempo totale del run di ablazione (riga del run.log, secondi interi)
        self.wall_status = None                 # lo stesso tempo da status.json (un decimale)
        self.checks = {}


def model_names(spec: dict, group: str) -> list:
    if group == "ablation_run":
        return sorted(f"{a}_fold{f}_seed{s}" for a in spec["arms"] for f in spec["folds"] for s in spec["seeds"])
    return sorted(f"fold{f}_seed{s}" for f in spec["folds"] for s in spec["seeds"])


def name_fields(name: str) -> tuple:
    m = re.fullmatch(r"(?:([a-z][a-z0-9_]*)_)?fold(\d+)_seed(\d+)", name)
    if not m:
        fail(f"model name {name!r} does not follow [<arm>_]fold<k>_seed<s>")
    return m.group(1) or FOLDS_ARM, int(m.group(2)), int(m.group(3))


def validate_card(c: dict, name: str, group: str, cfg: dict, listed: dict) -> None:
    """Scheda di training: chiavi, ricetta, storia, checkpoint e simulazione dell'EarlyStopping che riproduce l'arresto."""
    what = f"{group} card {name}"
    rc = cfg["recipe"]
    if sorted(c) != list(CARD_KEYS[group]):
        fail(f"{what}: keys differ from the expected set (missing {sorted(set(CARD_KEYS[group]) - set(c))}, "
             f"unknown {sorted(set(c) - set(CARD_KEYS[group]))})")
    arm, fold, seed = name_fields(name)
    if c["arch"] != "LivenessNet":
        fail(f"{what}: arch is not LivenessNet")
    if c["fold"] != fold:
        fail(f"{what}: fold differs from the file name")
    if c["train_seed"] != seed:
        fail(f"{what}: train_seed differs from the file name")
    if "arm" in c and c["arm"] != arm:
        fail(f"{what}: arm differs from the file name")
    for k in ("max_epochs", "early_stopping_patience", "batch", "lr"):
        if c[k] != rc[k] or isinstance(c[k], bool):
            fail(f"{what}: {k} = {c[k]!r} differs from the recipe of the configuration ({rc[k]!r})")
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
    sim = simulate_es(h, rc["max_epochs"], rc["monitor"], rc["early_stopping_patience"], 0)
    if sim is None or sim["stop"] != n or sim["restored"] != r:
        got = "not determinable" if sim is None else f"stop {sim['stop']}, restored {sim['restored']}"
        fail(f"{what}: the replay of the Keras EarlyStopping gives {got}, the card records epochs_run {n}, restored_epoch {r}")
    if c["restored_by"] != RESTORED_BY[sim["early"]]:
        fail(f"{what}: restored_by {c['restored_by']!r} differs from the replay ({RESTORED_BY[sim['early']]!r})")
    if c["weights_file"] != f"{name}_LivenessNet.keras" or not _hex(c["weights_sha256"]) or listed.get(c["weights_file"]) != c["weights_sha256"]:
        fail(f"{what}: weights_file or weights_sha256 differ from the restricted SHA256SUMS of its folder")
    parse_time(c["saved"], f"{what} saved")
    parse_time(c["test_scored_after_threshold"], f"{what} test_scored_after_threshold")


def load_status(folder: Path, spec: dict, what: str, want: str) -> dict:
    sb = read_regular(folder / STATUS, f"{what} status.json")
    if sha256_bytes(sb) != spec["status_sha256"]:
        fail(f"{what} status.json: SHA-256 {sha256_bytes(sb)} differs from the configuration ({spec['status_sha256']})")
    st = read_json(sb, f"{what} status.json")
    if st.get("status") != want or st.get("passed") is not (want == "PASS") or st.get("run_id") != spec["run_id"]:
        fail(f"{what} status.json: not a {want} of run {spec['run_id']}")
    if st.get("restricted_sums_sha256") != spec["restricted_sums_sha256"]:
        fail(f"{what} status.json: the restricted SHA256SUMS anchor differs from the configuration")
    return st


def listed_file(folder: Path, listed: dict, fn: str, what: str) -> bytes:
    if fn not in listed:
        fail(f"{what} restricted SHA256SUMS: {fn} not listed")
    b = read_regular(folder / "restricted" / fn, f"{what} {fn}")
    if sha256_bytes(b) != listed[fn]:
        fail(f"{what} {fn}: SHA-256 {sha256_bytes(b)} differs from the restricted SHA256SUMS ({listed[fn]})")
    return b


def load_threshold(b: bytes, group: str, card: dict, name: str) -> dict:
    """threshold.json del checkpoint: braccio, fold, seed e checkpoint uguali alla scheda; value un numero finito;
    frozen_at un tempo fra il salvataggio dei pesi e lo score del test; dev_scores_sha256 controllato con il dev."""
    what = f"{group} {name}_threshold.json"
    thr = read_json(b, what)
    arm, fold, seed = name_fields(name)
    if group == "ablation_run" and thr.get("arm") != arm:
        fail(f"{what}: arm differs from the card")
    if thr.get("fold") != fold or thr.get("train_seed") != seed:
        fail(f"{what}: fold or train_seed differ from the card")
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
        fail(f"{group} {name}: saved, frozen_at and test_scored_after_threshold are not in order")
    return {"value": float(thr["value"]), "dev_scores_sha256": thr["dev_scores_sha256"], "sha256": sha256_bytes(b),
            "delays": ((frozen - saved).total_seconds(), (tested - frozen).total_seconds())}


def load_dev(b: bytes, group: str, card: dict, name: str, spec: dict, cfg: dict, thr: dict, inp: Inputs) -> dict:
    """dev_scores.csv del checkpoint, riga per riga: ruolo, braccio, fold, seed e checkpoint della scheda; etichetta e
    score nel dominio; decisioni e soglia della riga coerenti con la soglia congelata (attacco se score > soglia, il
    pareggio è bona fide; prediction_05 alla soglia fissa 0,5); sample_id e file_sha256 unici; elenco delle chiavi
    uguale a dev_keys_sha256 della scheda (SHA-256 delle chiavi unite da «\\n», come nei runner); score uguali a
    dev_scores_sha256 della soglia (SHA-256 di repr(float(score)) uniti da «\\n», come nei runner); dev bilanciato;
    accuratezza alla soglia 0,5 = val_acc_at_best. Restituisce la diagnostica e le chiavi (solo in memoria)."""
    what = f"{group} {name}_dev_scores.csv"
    rows = parse_csv(b, DEV_COLUMNS[group], what)
    if len(rows) != spec["dev_rows"]:
        fail(f"{what}: {len(rows)} rows; the configuration expects {spec['dev_rows']}")
    if card["n_dev"] != spec["dev_rows"]:
        fail(f"{what}: n_dev of the card is {card['n_dev']}; the configuration expects {spec['dev_rows']}")
    arm, fold, seed = name_fields(name)
    value = thr["value"]
    labels, scores, subjects, keys = [], [], set(), []
    for r in rows:
        if r["role"] != "dev":
            fail(f"{what}: a row carries another role")
        if r["fold"] != str(fold) or r["train_seed"] != str(seed):
            fail(f"{what}: a row carries another fold or seed")
        if group == "ablation_run" and r["arm"] != arm:
            fail(f"{what}: a row carries another arm")
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
        subjects.add(r["subject_id"])           # solo per contare i soggetti e per la scansione del pubblico
        keys.append((r["sample_id"], r["file_sha256"], r["label"], r["subject_id"]))
    if len({k[0] for k in keys}) != len(keys) or len({k[1] for k in keys}) != len(keys):
        fail(f"{what}: a sample_id or a file_sha256 appears in more than one row")
    if sha256_bytes("\n".join(k[0] for k in keys).encode()) != card["dev_keys_sha256"]:
        fail(f"{what}: the sample_id list differs from dev_keys_sha256 of the card")
    if sha256_bytes("\n".join(repr(float(r["score_attack"])) for r in rows).encode()) != thr["dev_scores_sha256"]:
        fail(f"{what}: the scores differ from dev_scores_sha256 of {name}_threshold.json")
    if labels.count(0) != labels.count(1):
        fail(f"{what}: the dev is not balanced ({labels.count(0)} bona fide, {labels.count(1)} attacks)")
    d = dev_diag(labels, scores, len(subjects))
    if abs(d["acc_05"] - card["val_acc_at_best"]) > cfg["criteria"]["checkpoint_agreement_tolerance"]:
        fail(f"{what}: accuracy at 0.5 of the checkpoint {d['acc_05']:.6f} differs from val_acc_at_best {card['val_acc_at_best']:.6f}")
    inp.subjects |= subjects
    return d, tuple(keys)


def load_run(cfg: dict, base: Path, group: str, inp: Inputs) -> None:
    """Run di ablazione o run dei fold: status PASS, SHA256SUMS ancorati, run.log ancorato, schede e dev_scores (e, per
    l'ablazione, soglie e tempo totale) per SHA-256 registrato; nessun'altra scheda elencata."""
    spec = cfg["inputs"][group]
    folder = rel_path(base, spec["folder"])
    st = load_status(folder, spec, group, "PASS")
    if st.get("public_sums_sha256") != spec["public_sums_sha256"] or st.get("run_log_sha256") != spec["run_log_sha256"]:
        fail(f"{group} status.json: the public SHA256SUMS or run.log anchor differs from the configuration")
    pb = read_regular(folder / PUBLIC / "SHA256SUMS", f"{group} public SHA256SUMS")
    if sha256_bytes(pb) != spec["public_sums_sha256"]:
        fail(f"{group} public SHA256SUMS: SHA-256 {sha256_bytes(pb)} differs from the configuration ({spec['public_sums_sha256']})")
    rb = read_regular(folder / "restricted" / "SHA256SUMS", f"{group} restricted SHA256SUMS")
    if sha256_bytes(rb) != spec["restricted_sums_sha256"]:
        fail(f"{group} restricted SHA256SUMS: SHA-256 {sha256_bytes(rb)} differs from the configuration ({spec['restricted_sums_sha256']})")
    listed = read_sums(rb, f"{group} restricted SHA256SUMS")
    names = model_names(spec, group)
    cards = sorted(n[: -len("_card.json")] for n in listed if n.endswith("_card.json"))
    if cards != names:
        fail(f"{group} restricted SHA256SUMS: {len(cards)} training cards listed, the configuration expects exactly {len(names)} "
             f"(missing {len(set(names) - set(cards))}, unexpected {len(set(cards) - set(names))})")
    lb = read_regular(folder / LOG, f"{group} run.log")
    if sha256_bytes(lb) != spec["run_log_sha256"]:
        fail(f"{group} run.log: SHA-256 {sha256_bytes(lb)} differs from the configuration ({spec['run_log_sha256']})")
    dev_of_fold, subjects_per_fold, frames_per_fold = {}, {}, {}
    for name in names:
        card = read_json(listed_file(folder, listed, f"{name}_card.json", group), f"{group} {name}_card.json")
        validate_card(card, name, group, cfg, listed)
        thr = load_threshold(listed_file(folder, listed, f"{name}_threshold.json", group), group, card, name)
        d, keys = load_dev(listed_file(folder, listed, f"{name}_dev_scores.csv", group), group, card, name, spec, cfg, thr, inp)
        arm, fold, seed = name_fields(name)
        # dev comune: stesse righe (chiave, file, etichetta, soggetto) nello stesso ordine per tutti i modelli del fold
        if dev_of_fold.setdefault(fold, keys) != keys:
            fail(f"{group} fold {fold}: the models of the fold do not share the same dev (keys, files, labels or subjects differ)")
        subjects_per_fold[fold], frames_per_fold[fold] = d["n_subjects"], d["n"]
        inp.models[group].append({"run": spec["label"], "run_id": spec["run_id"], "group": group, "name": name, "arm": arm,
                                  "fold": fold, "seed": seed, "card": card, "h": card["history"], "dev": d, "delays": thr["delays"]})
    if group == "ablation_run":
        walls = [float(mm.group(1)) for line in lb.decode("utf-8", errors="replace").splitlines() if (mm := WALL_RE.match(line))]
        if len(walls) != 1:
            fail(f"{group} run.log: {len(walls)} lines 'run folder: ... wall <s> s' found, exactly one expected")
        if not _number(st.get("wall_seconds")) or abs(walls[0] - st["wall_seconds"]) > WALL_TOL:
            fail(f"{group} run.log: wall {walls[0]} s differs from wall_seconds of status.json by more than {WALL_TOL} s")
        inp.wall = walls[0]
        inp.wall_status = st["wall_seconds"]
    inp.checks[group] = {"run_id": spec["run_id"], "status": "PASS", "status_sha256": spec["status_sha256"],
                         "public_sums_sha256": spec["public_sums_sha256"], "restricted_sums_sha256": spec["restricted_sums_sha256"],
                         "run_log_sha256": spec["run_log_sha256"], "cards": len(names), "dev_scores_files": len(names),
                         "threshold_files": len(names),
                         "dev_rows_per_model": spec["dev_rows"], "subjects_per_fold_dev": {str(f): v for f, v in sorted(subjects_per_fold.items())},
                         "frames_per_fold_dev": {str(f): v for f, v in sorted(frames_per_fold.items())},
                         "early_stopping_replay": f"{len(names)}/{len(names)} reproduced"}


def load_failed(cfg: dict, base: Path, inp: Inputs) -> None:
    """Tentativo fallito: status.json FAIL ancorato, SHA256SUMS riservato ancorato; ogni scheda per SHA-256 della
    configurazione, uguale alla riga del SHA256SUMS, ed elencata fra i modelli completi dello status; stessi controlli
    della scheda degli altri run."""
    spec = cfg["inputs"]["failed_attempt"]
    folder = rel_path(base, spec["folder"])
    st = load_status(folder, spec, "failed_attempt", "FAIL")
    rb = read_regular(folder / "restricted" / "SHA256SUMS", "failed_attempt restricted SHA256SUMS")
    if sha256_bytes(rb) != spec["restricted_sums_sha256"]:
        fail(f"failed_attempt restricted SHA256SUMS: SHA-256 {sha256_bytes(rb)} differs from the configuration")
    listed = read_sums(rb, "failed_attempt restricted SHA256SUMS")
    names = sorted(n[: -len("_card.json")] for n in spec["cards"])
    if sorted(st.get("kept_complete_models") or []) != names:
        fail("failed_attempt status.json: kept_complete_models differ from the cards of the configuration")
    main = set(model_names(cfg["inputs"]["ablation_run"], "ablation_run"))
    for name in names:
        fn = f"{name}_card.json"
        if name not in main:
            fail(f"failed_attempt {fn}: not a model of the ablation run")
        if listed.get(fn) != spec["cards"][fn]:
            fail(f"failed_attempt {fn}: the SHA-256 of the configuration differs from the restricted SHA256SUMS")
        card = read_json(listed_file(folder, listed, fn, "failed_attempt"), f"failed_attempt {fn}")
        validate_card(card, name, "failed_attempt", cfg, listed)
        arm, fold, seed = name_fields(name)
        inp.models["failed_attempt"].append({"run": spec["label"], "run_id": spec["run_id"], "group": "failed_attempt", "name": name,
                                             "arm": arm, "fold": fold, "seed": seed, "card": card, "h": card["history"], "dev": None})
    inp.checks["failed_attempt"] = {"run_id": spec["run_id"], "status": "FAIL", "status_sha256": spec["status_sha256"],
                                    "restricted_sums_sha256": spec["restricted_sums_sha256"], "cards": len(names),
                                    "use": "determinism comparison only", "early_stopping_replay": f"{len(names)}/{len(names)} reproduced"}


def load_inputs(cfg: dict, base: Path) -> Inputs:
    inp = Inputs()
    load_run(cfg, base, "ablation_run", inp)
    load_run(cfg, base, "folds_run", inp)
    load_failed(cfg, base, inp)
    for e in cfg["examples"]:
        if not any(m["run"] == e["run"] and m["arm"] == e["arm"] and m["fold"] == e["fold"] and m["seed"] == e["seed"]
                   for g in ("ablation_run", "folds_run") for m in inp.models[g]):
            fail(f"configuration: example {e['run']} {e['arm']} fold {e['fold']} seed {e['seed']} is not a model of the inputs")
    return inp


# ============================================================================ analisi

def median(v):
    return statistics.median(v) if v else None


def enrich(cfg: dict, inp: Inputs) -> None:
    """Campi derivati di ogni modello (una sola volta, con le regole sopra)."""
    crit, rc = cfg["criteria"], cfg["recipe"]
    rv = crit["recovery"]
    for g in GROUPS:
        for m in inp.models[g]:
            c, h = m["card"], m["h"]
            m["ep"] = c["epochs_run"]
            m["n_fitting"] = c["n_fitting"]
            m["steps"] = math.ceil(c["n_fitting"] / rc["batch"])
            m["deg"] = is_degenerate(c["restored_epoch"], c["val_acc_at_best"], crit)
            m["rec"] = first_recovery(h, rv["val_accuracy_min"], rv["val_loss_max"])
            m["sec_ep"] = c["train_seconds"] / c["epochs_run"]
            m["sat"] = sum(1 for v in h["val_accuracy"] if is_half(v, crit))
            m["first_impr"] = next((i + 1 for i in range(1, len(h["val_loss"])) if h["val_loss"][i] < h["val_loss"][0]), None)


def epoch_rows(inp: Inputs) -> list:
    rows = []
    for g in GROUPS:
        for m in inp.models[g]:
            h = m["h"]
            for i in range(m["ep"]):
                rows.append([m["run"], m["arm"], m["fold"], m["seed"], m["n_fitting"], m["steps"], i + 1, repr(h["loss"][i]),
                             repr(h["accuracy"][i]), repr(h["val_loss"][i]), repr(h["val_accuracy"][i]), repr(h["lr"][i]),
                             int(i + 1 == m["card"]["restored_epoch"]), int(m["deg"]), m["run_id"], m["ep"]])
    return rows


def model_rows(inp: Inputs) -> list:
    rows = []
    for g in GROUPS:
        for m in inp.models[g]:
            c, h = m["card"], m["h"]
            rows.append([m["run"], m["arm"], m["fold"], m["seed"], m["n_fitting"], m["steps"], m["ep"], c["restored_epoch"],
                         c["restored_by"], repr(c["best_val_loss"]), repr(c["val_acc_at_best"]), int(m["deg"]),
                         m["rec"] if m["rec"] else "", m["sat"], m["first_impr"] or "",
                         repr(min(h["val_loss"][2:])) if m["ep"] > 2 else "", repr(h["val_accuracy"][-1]), repr(h["val_loss"][-1]),
                         repr(h["lr"][-1]), c["train_seconds"], round(m["sec_ep"], 2), m["run_id"]])
    return rows


def mkey(m: dict) -> dict:
    return {"run": m["run"], "arm": m["arm"], "fold": m["fold"], "seed": m["seed"]}


def analyse(cfg: dict, inp: Inputs) -> dict:
    """Tabelle T1-T13 come dati (stessi conteggi e stesse formule della scheda dei fatti)."""
    crit, rc = cfg["criteria"], cfg["recipe"]
    spec = cfg["inputs"]["ablation_run"]
    arms, seeds, folds = spec["arms"], spec["seeds"], spec["folds"]
    main = inp.models["ablation_run"]
    fr = inp.models["folds_run"]
    failed = inp.models["failed_attempt"]
    deg = [m for m in main if m["deg"]]
    ok = [m for m in main if not m["deg"]]
    groups = [("degenerate", deg), ("non-degenerate", ok), ("folds run", fr)]
    half = lambda v: is_half(v, crit)   # noqa: E731
    late = crit["late_recovery_after_epoch"]
    T = {}

    # T1
    T["T1"] = {"title": "Degenerate trainings (criterion re-applied)", "criterion": crit["degenerate"]["text"],
               "rows": [{**mkey(m), "epochs_run": m["ep"], "restored_epoch": m["card"]["restored_epoch"],
                         "best_val_loss": m["card"]["best_val_loss"], "val_acc_at_best": m["card"]["val_acc_at_best"]} for m in deg],
               "count": len(deg), "of": len(main),
               "per_arm": {a: sum(1 for m in deg if m["arm"] == a) for a in arms},
               "per_seed": {str(s): sum(1 for m in deg if m["seed"] == s) for s in seeds},
               "per_fold": {str(f): sum(1 for m in deg if m["fold"] == f) for f in folds},
               "boundary_rule": f"non-degenerate with restored_epoch <= {crit['boundary_restored_epoch_max']}",
               "boundary": [{**mkey(m), "restored_epoch": m["card"]["restored_epoch"], "val_acc_at_best": round(m["card"]["val_acc_at_best"], 4),
                             "best_val_loss": round(m["card"]["best_val_loss"], 3)} for m in ok
                            if m["card"]["restored_epoch"] <= crit["boundary_restored_epoch_max"]]}
    # T2
    rows = []
    for a in arms:
        for s in seeds:
            r = {"arm": a, "seed": s}
            for f in folds:
                m = next(x for x in main if x["arm"] == a and x["seed"] == s and x["fold"] == f)
                r[f"fold{f}"] = ("D" if m["deg"] else "") + str(m["card"]["restored_epoch"])
            rows.append(r)
    T["T2"] = {"title": "Fold x seed per arm (D = degenerate, number = restored epoch)", "rows": rows}
    # T3
    rows = []
    for lab, grp in groups:
        sat_ep = [(m, i) for m in grp for i, v in enumerate(m["h"]["val_accuracy"]) if half(v)]
        vls = [m["h"]["val_loss"][i] for m, i in sat_ep]
        tra = [m["h"]["accuracy"][i] for m, i in sat_ep]
        trl = [m["h"]["loss"][i] for m, i in sat_ep]
        rows.append({"group": lab, "models": len(grp), "epoch1_saturated": sum(1 for m in grp if half(m["h"]["val_accuracy"][0])),
                     "epoch2_saturated": sum(1 for m in grp if len(m["h"]["val_accuracy"]) > 1 and half(m["h"]["val_accuracy"][1])),
                     "saturated_epochs": len(sat_ep), "epochs": sum(m["ep"] for m in grp),
                     "val_loss_min": min(vls) if vls else None, "val_loss_median": median(vls), "val_loss_max": max(vls) if vls else None,
                     "train_accuracy_min": min(tra) if tra else None, "train_accuracy_median": median(tra),
                     "train_loss_max": max(trl) if trl else None})
    allm = main + fr
    ref = crit["train_accuracy_reference"]
    T["T3"] = {"title": "Dev saturation (val_accuracy = 0.5) and the training side in the same epochs", "rows": rows,
               "models_with_a_saturated_epoch": sum(1 for m in allm if any(half(v) for v in m["h"]["val_accuracy"])),
               "models": len(allm), "train_accuracy_reference": ref,
               "saturated_epochs_with_train_accuracy_above_reference": sum(1 for m in allm for i, v in enumerate(m["h"]["val_accuracy"])
                                                                          if half(v) and m["h"]["accuracy"][i] > ref),
               "saturated_epochs": sum(1 for m in allm for v in m["h"]["val_accuracy"] if half(v)),
               "below_chance_threshold": crit["below_chance_val_accuracy"],
               "below_chance_epochs": [{**mkey(m), "epoch": i + 1, "val_accuracy": round(v, 4)} for m in allm
                                       for i, v in enumerate(m["h"]["val_accuracy"]) if v < crit["below_chance_val_accuracy"]]}

    # T4
    def rec_row(lab, grp):
        r = [m["rec"] for m in grp if m["rec"]]
        return {"group": lab, "models": len(grp), "with_recovery": len(r), "min": min(r) if r else None, "median": median(r),
                "max": max(r) if r else None, "median_steps": median([m["rec"] * m["steps"] for m in grp if m["rec"]]),
                f"recovery_after_epoch_{late}": sum(1 for x in r if x > late)}
    ab = [m for m in ok if m["arm"] != arms[-1]]
    bp = [m for m in ok if m["arm"] == arms[-1]]
    steps_label = lambda grp: "/".join(str(x) for x in sorted({m["steps"] for m in grp}))   # noqa: E731
    rows = [rec_row(a, [m for m in main if m["arm"] == a]) for a in arms]
    rows += [rec_row(f"seed {s}", [m for m in main if m["seed"] == s]) for s in seeds]
    rows += [rec_row(f"fold {f}", [m for m in main if m["fold"] == f]) for f in folds]
    rows += [rec_row("non-degenerate", ok), rec_row("degenerate", deg), rec_row("folds run", fr)]
    rows += [rec_row(f"{' and '.join(arms[:-1])} non-degenerate ({steps_label(ab)} steps)", ab),
             rec_row(f"{arms[-1]} non-degenerate ({steps_label(bp)} steps)", bp)]
    T["T4"] = {"title": "First recovery on the dev, per group", "criterion": crit["recovery"]["text"], "rows": rows,
               "late_after_epoch": late,
               "first_recovery_per_model": [{"group": "folds run", "epochs": sorted(m["rec"] for m in fr if m["rec"])},
                                            {"group": f"{' and '.join(arms[:-1])} non-degenerate", "epochs": sorted(m["rec"] for m in ab if m["rec"])},
                                            {"group": f"{arms[-1]} non-degenerate", "epochs": sorted(m["rec"] for m in bp if m["rec"])}]}
    # T5
    rows = []
    se = crit["recovery"]["sensitivity"]
    for a in se["val_accuracy_min"]:
        for l in se["val_loss_max"]:
            r = {"val_accuracy_min": a, "val_loss_max": l}
            for lab, grp in groups:
                v = [x for x in (first_recovery(m["h"], a, l) for m in grp) if x]
                r[lab] = {"with_recovery": len(v), "models": len(grp), "median": median(v)}
            rows.append(r)
    T["T5"] = {"title": "Sensitivity of the recovery criterion", "rows": rows}
    # T6
    rows = []
    for m in deg:
        vl, va = m["h"]["val_loss"], m["h"]["val_accuracy"]
        rows.append({**mkey(m), "restored_epoch": m["card"]["restored_epoch"], "epochs_run": m["ep"], "val_loss": vl, "val_accuracy": va,
                     "last_val_loss_vs_previous": "down" if vl[-1] < vl[-2] else "up",
                     "last_val_accuracy_vs_previous": "up" if va[-1] > va[-2] + 1e-9 else ("=" if abs(va[-1] - va[-2]) < 1e-9 else "down")})
    fi = []
    for lab, grp in [("non-degenerate", ok), ("folds run", fr), ("degenerate", deg)]:
        fi.append({"group": lab, "first_val_loss_below_epoch1": sorted([m["first_impr"] for m in grp], key=lambda x: (x is None, x or 0)),
                   "with_val_accuracy_0.5_in_that_epoch": sum(1 for m in grp if m["first_impr"] and half(m["h"]["val_accuracy"][m["first_impr"] - 1])),
                   "with_val_accuracy_0.5_at_epoch1": sum(1 for m in grp if half(m["h"]["val_accuracy"][0]))})
    nondeg = ok + fr                 # non degeneri per il criterio (run di ablazione e run dei fold): non tutti recuperano
    T["T6"] = {"title": "Mechanics of the stop (degenerate trainings)", "rows": rows, "first_improvement": fi,
               "non_degenerate_with_recovery_after_epoch": late,
               "non_degenerate_recovered_late": sum(1 for m in nondeg if m["rec"] and m["rec"] > late),
               "non_degenerate_with_recovery": sum(1 for m in nondeg if m["rec"]),
               "non_degenerate_without_recovery": [mkey(m) for m in nondeg if not m["rec"]]}
    # T7
    rows, changes, become = [], [], []
    for rule in cfg["alternative_rules"]:
        kw = {"monitor": rule["monitor"], "patience": rule["patience"], "start": rule["start_from_epoch"]}
        r = {"rule": rule["label"]}
        for lab, grp in groups:
            sims = [simulate_es(m["h"], m["card"]["max_epochs"], **kw) for m in grp]
            known = [s for s in sims if s]
            r[lab] = {"determinable": len(known), "models": len(grp),
                      "degenerate": sum(1 for s in known if is_degenerate(s["restored"], s["val_acc"], crit)),
                      "restored_median": median([s["restored"] for s in known])}
        rows.append(r)
        ch = []
        for m in ok:
            s = simulate_es(m["h"], m["card"]["max_epochs"], **kw)
            if s is None:
                ch.append({**mkey(m), "restored_run": m["card"]["restored_epoch"], "restored_rule": None})
            elif s["restored"] != m["card"]["restored_epoch"]:
                ch.append({**mkey(m), "restored_run": m["card"]["restored_epoch"], "restored_rule": s["restored"],
                           "val_acc_run": m["card"]["val_acc_at_best"], "val_acc_rule": s["val_acc"]})
        changes.append({"rule": rule["label"], "non_degenerate_changes": ch})
        become.append({"rule": rule["label"], "non_degenerate_that_become_degenerate": [
            mkey(m) for m in nondeg if (s := simulate_es(m["h"], rc["max_epochs"], **kw)) and is_degenerate(s["restored"], s["val_acc"], crit)]})
    T["T7"] = {"title": "Alternative early stopping rules replayed on the recorded histories (exact or not determinable)",
               "rows": rows, "changes": changes, "become_degenerate": become}
    # T8
    rows = []
    for lab, grp in groups:
        firsts = [next((i + 1 for i, v in enumerate(m["h"]["lr"]) if v < rc["lr"] * 0.99), None) for m in grp]
        fired_sat = sum(1 for m, e in zip(grp, firsts) if e and half(m["h"]["val_accuracy"][e - 2]))
        last = {}
        for m in grp:
            k = f"{m['h']['lr'][-1]:.2e}"
            last[k] = last.get(k, 0) + 1
        rows.append({"group": lab, "with_reduction": sum(1 for e in firsts if e), "first_reduced_epoch": sorted(e for e in firsts if e),
                     "decided_with_dev_at_0.5": fired_sat, "lr_at_last_epoch": dict(sorted(last.items(), reverse=True))})
    T["T8"] = {"title": "Learning rate (history lr = the lr used in the epoch)", "rows": rows}
    # T9
    rows = []
    for e in range(crit["train_side_epochs"]):
        r = {"epoch": e + 1}
        for lab, grp in groups:
            a = [m["h"]["accuracy"][e] for m in grp if m["ep"] > e]
            l = [m["h"]["loss"][e] for m in grp if m["ep"] > e]
            r[lab] = {"accuracy_median": median(a), "accuracy_min": min(a) if a else None, "accuracy_max": max(a) if a else None,
                      "loss_median": median(l), "loss_min": min(l) if l else None, "loss_max": max(l) if l else None, "models": len(a)}
        rows.append(r)
    pe = crit["pair_train_epochs"]
    pairs = [{"arm": m["arm"], "fold": m["fold"], "degenerate_seed": m["seed"], "other_seed": o["seed"],
              "degenerate_train_accuracy": [round(x, 4) for x in m["h"]["accuracy"][:pe]],
              "other_train_accuracy": [round(x, 4) for x in o["h"]["accuracy"][:pe]]}
             for m in deg for o in main if o["arm"] == m["arm"] and o["fold"] == m["fold"] and not o["deg"]]
    T["T9"] = {"title": "Training side per epoch (median [min-max])", "rows": rows, "pair_epochs": pe, "pairs": pairs}
    # T10
    rows = [{**mkey(m), "degenerate": m["deg"], "restored_epoch": m["card"]["restored_epoch"],
             **{k: m["dev"][k] for k in ("acc_05", "pred_attack_05", "pred_attack_05_bona_fide", "pred_attack_05_attack",
                                         "mean_bona_fide", "mean_attack", "auroc", "distinct")}}
            for m in deg + [x for x in ok if x["card"]["restored_epoch"] <= crit["boundary_restored_epoch_max"]]]
    au_ok = [m["dev"]["auroc"] for m in ok]
    au_f = [m["dev"]["auroc"] for m in fr]
    T["T10"] = {"title": "Checkpoint on the dev (dev_scores.csv: threshold 0.5 and dev AUROC)", "rows": rows,
                "auroc_non_degenerate": {"median": median(au_ok), "min": min(au_ok), "max": max(au_ok)},
                "auroc_folds_run": {"median": median(au_f), "min": min(au_f), "max": max(au_f)},
                "dev_subjects_per_fold": inp.checks["ablation_run"]["subjects_per_fold_dev"],
                "dev_frames_per_fold": inp.checks["ablation_run"]["frames_per_fold_dev"],
                "dev_subjects_per_fold_folds_run": inp.checks["folds_run"]["subjects_per_fold_dev"],
                "max_abs_difference_acc_05_val_acc_at_best": max(abs(m["dev"]["acc_05"] - m["card"]["val_acc_at_best"]) for m in main),
                "max_abs_difference_acc_05_val_acc_at_best_folds_run": max(abs(m["dev"]["acc_05"] - m["card"]["val_acc_at_best"]) for m in fr)}
    # T11: determinismo come segnalazione (identical true/false), mai come errore: è un'osservazione dello studio
    rows = []
    for m in failed:
        o = next(x for x in main if x["name"] == m["name"])
        same_len = m["ep"] == o["ep"]
        diffs = {k: (max(abs(a - b) for a, b in zip(m["h"][k], o["h"][k])) if same_len else None) for k in HISTORY_KEYS}
        mx = max(diffs.values()) if same_len else None
        rows.append({"model": m["name"], "same_epochs": same_len, "same_restored_epoch": m["card"]["restored_epoch"] == o["card"]["restored_epoch"],
                     "max_abs_history_difference": mx, "identical_history": same_len and mx == 0.0,
                     "same_weights_sha256": m["card"]["weights_sha256"] == o["card"]["weights_sha256"],
                     "train_seconds_failed_attempt": m["card"]["train_seconds"], "train_seconds_run": o["card"]["train_seconds"]})
    T["T11"] = {"title": "Determinism: failed attempt against the run", "rows": rows,
                "identical_histories": sum(1 for r in rows if r["identical_history"]), "models": len(rows)}
    # T12
    rows = []
    for a in list(arms) + [FOLDS_ARM]:
        grp = [m for m in main + fr if m["arm"] == a]
        sp = [m["sec_ep"] for m in grp]
        rows.append({"arm": a, "models": len(grp), "steps_per_epoch": steps_label(grp), "epochs": sum(m["ep"] for m in grp),
                     "train_seconds": sum(m["card"]["train_seconds"] for m in grp), "seconds_per_epoch_median": median(sp),
                     "seconds_per_epoch_min": min(sp), "seconds_per_epoch_max": max(sp),
                     "seconds_per_epoch_weighted": sum(m["card"]["train_seconds"] for m in grp) / sum(m["ep"] for m in grp)})
    tot = sum(m["card"]["train_seconds"] for m in main)
    over = (inp.wall - tot) / len(main)
    spe = {a: sum(m["card"]["train_seconds"] for m in main if m["arm"] == a) / sum(m["ep"] for m in main if m["arm"] == a) for a in arms}
    d_dev = [m["delays"][0] for m in main]
    d_test = [m["delays"][1] for m in main]
    scen = []
    for s in cfg["time_scenarios"]:
        sel = s["selection"]
        if sel == "all":
            chosen = main
        elif sel == "degenerate":
            chosen = deg
        elif sel == "stopped before max_epochs":
            chosen = [m for m in main if m["ep"] < m["card"]["max_epochs"]]
        else:
            chosen = [m for m in main if m["arm"] in sel[len("arms: "):].split(", ")]
        tr = sum(s["epochs"] * spe[m["arm"]] for m in chosen)
        scen.append({"label": s["label"], "selection": sel, "models": len(chosen), "degenerate": sum(1 for m in chosen if m["deg"]),
                     "epochs": s["epochs"], "training_seconds": tr, "training_hours": tr / 3600,
                     "with_overhead_seconds": tr + len(chosen) * over if s["with_overhead"] else None,
                     "with_overhead_hours": (tr + len(chosen) * over) / 3600 if s["with_overhead"] else None})
    T["T12"] = {"title": "Times", "rows": rows, "run_train_seconds": tot, "run_wall_seconds": inp.wall,
                "run_wall_seconds_status": inp.wall_status,
                "run_epochs": sum(m["ep"] for m in main), "overhead_seconds": inp.wall - tot, "overhead_seconds_per_model": over,
                "weighted_seconds_per_epoch": spe,
                "saved_to_threshold_seconds": {"median": median(d_dev), "min": min(d_dev), "max": max(d_dev)},
                "threshold_to_test_scores_seconds": {"median": median(d_test), "min": min(d_test), "max": max(d_test)},
                "time_resolution_seconds": 1, "scenarios": scen}
    # T13
    mom = rc["bn_momentum"]
    rows = []
    for lab, grp in [(f"{arms[-1]}", [m for m in main if m["arm"] == arms[-1]]), (" and ".join(arms[:-1]), [m for m in main if m["arm"] != arms[-1]]),
                     ("folds run", fr)]:
        for st in sorted({m["steps"] for m in grp}):
            rows.append({"group": f"{lab} ({st} steps)", "steps_per_epoch": st,
                         "residual": [mom ** (st * e) for e in range(1, crit["bn_residual_epochs"] + 1)]})
    T["T13"] = {"title": f"Residual weight of the initial value in the moving statistics of BatchNormalization (momentum {mom})",
                "rows": rows, "note": f"{mom}^steps: share of the moving average still due to the initial value (mean 0, variance 1) "
                                      "if the batch statistic were constant; with changing weights the moving average also chases a moving target"}
    return T


# ============================================================================ testi pubblici

def fmt(v, nd: int = 4) -> str:
    """Formato delle celle (come la scheda dei fatti): float a 4 decimali, interi e testi come sono, None come «-»."""
    if v is None or v == "":
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def table(head, rows) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(fmt(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def f4(v) -> str:
    """Mediane e altri valori: sempre quattro decimali (i conteggi restano interi); None come «-»."""
    return "-" if v is None else f"{float(v):.4f}"


def counts(d: dict, prefix: str = "") -> str:
    return ", ".join(f"{prefix}{k} {v}" for k, v in d.items())


def who(d: dict) -> str:
    return f"{d['arm']} f{d['fold']} s{d['seed']}" + ("" if d["run"] == "sessablrun" else f" ({d['run']})")


def aggregate_md(doc: dict) -> str:
    T = doc["tables"]
    L = [f"# Degenerate trainings of the session ablation run: train/dev curves ({doc['run_id']})", "",
         "Observations from the training cards already recorded; no training, no inference, no test score or test metric read. "
         "The possible explanations are not computed here. Criteria defined after the run: " + doc["criteria"]["degenerate"]["text"] +
         "; recovery: " + doc["criteria"]["recovery"]["text"] + ".", "",
         f"Inputs: {doc['inputs']['ablation_run']['run_id']} ({doc['inputs']['ablation_run']['cards']} cards), "
         f"{doc['inputs']['folds_run']['run_id']} ({doc['inputs']['folds_run']['cards']} cards), "
         f"{doc['inputs']['failed_attempt']['run_id']} ({doc['inputs']['failed_attempt']['cards']} cards, determinism only). "
         f"Early stopping replay: {doc['counts']['early_stopping_replay']}. Figures: {doc['figures']['note']}.", ""]
    t = T["T1"]
    L += [f"## T1. {t['title']}", "", table(["arm", "fold", "seed", "epochs_run", "restored_epoch", "best_val_loss", "val_acc_at_best"],
                                            [[r["arm"], r["fold"], r["seed"], r["epochs_run"], r["restored_epoch"], r["best_val_loss"],
                                              r["val_acc_at_best"]] for r in t["rows"]]), "",
          f"Count: {t['count']} of {t['of']}. Per arm: {counts(t['per_arm'])}; per seed: {counts(t['per_seed'], 'seed ')}; "
          f"per fold: {counts(t['per_fold'], 'fold ')}.", "",
          f"Boundary cases ({t['boundary_rule']}): " + "; ".join(f"{who(r)} restored {r['restored_epoch']}, val_acc {r['val_acc_at_best']}, "
                                                                 f"val_loss {r['best_val_loss']}" for r in t["boundary"]) + ".", ""]
    t = T["T2"]
    keys = [k for k in t["rows"][0] if k.startswith("fold")]
    L += [f"## T2. {t['title']}", "", table(["arm", "seed"] + keys, [[r["arm"], r["seed"]] + [r[k] for k in keys] for r in t["rows"]]), ""]
    t = T["T3"]
    L += [f"## T3. {t['title']}", "", table(["group", "models", "epoch 1 at 0.5", "epoch 2 at 0.5", "epochs at 0.5", "epochs", "val_loss min",
                                            "val_loss median", "val_loss max", "train acc min", "train acc median", "train loss max"],
                                           [[r["group"], r["models"], r["epoch1_saturated"], r["epoch2_saturated"], r["saturated_epochs"], r["epochs"],
                                             r["val_loss_min"], r["val_loss_median"], r["val_loss_max"], r["train_accuracy_min"],
                                             r["train_accuracy_median"], r["train_loss_max"]] for r in t["rows"]]), "",
          f"Models ({t['models']}) with at least one epoch at 0.5: {t['models_with_a_saturated_epoch']}; epochs at 0.5 with train accuracy > "
          f"{t['train_accuracy_reference']}: {t['saturated_epochs_with_train_accuracy_above_reference']} of {t['saturated_epochs']}.", "",
          f"Epochs with val_accuracy < {t['below_chance_threshold']}: " + "; ".join(f"{who(r)} epoch {r['epoch']} ({r['val_accuracy']})"
                                                                                   for r in t["below_chance_epochs"]) + ".", ""]
    t = T["T4"]
    late = t["late_after_epoch"]
    L += [f"## T4. {t['title']}", "", f"Criterion: {t['criterion']}.", "",
          table(["group", "models", "with recovery", "min", "median", "max", "median in steps", f"recovery after epoch {late}"],
                [[r["group"], r["models"], r["with_recovery"], r["min"], f4(r["median"]), r["max"], f4(r["median_steps"]), r[f"recovery_after_epoch_{late}"]]
                 for r in t["rows"]]), ""]
    L += [f"First recovery per model, {r['group']}: {r['epochs']}." for r in t["first_recovery_per_model"]]
    L += [""]
    t = T["T5"]
    labs = [k for k in t["rows"][0] if k not in ("val_accuracy_min", "val_loss_max")]
    L += [f"## T5. {t['title']}", "", table(["criterion"] + labs, [[f"acc>={r['val_accuracy_min']:.2f}, loss<{r['val_loss_max']}"] +
                                                                   [f"{r[k]['with_recovery']}/{r[k]['models']} med {f4(r[k]['median'])}"
                                                                    for k in labs] for r in t["rows"]]), ""]
    t = T["T6"]
    L += [f"## T6. {t['title']}", "", table(["arm", "fold", "seed", "restored", "epochs_run", "val_loss per epoch", "val_acc per epoch",
                                            "val_loss last vs previous", "val_acc last vs previous"],
                                           [[r["arm"], r["fold"], r["seed"], r["restored_epoch"], r["epochs_run"],
                                             " ".join(f"{v:.1f}" for v in r["val_loss"]), " ".join(f"{v:.3f}" for v in r["val_accuracy"]),
                                             r["last_val_loss_vs_previous"], r["last_val_accuracy_vs_previous"]] for r in t["rows"]]), ""]
    for r in t["first_improvement"]:
        L += [f"{r['group']}: first epoch with val_loss < val_loss of epoch 1 -> {r['first_val_loss_below_epoch1']}; with val_accuracy "
              f"still 0.5 in that epoch: {r['with_val_accuracy_0.5_in_that_epoch']}; models with val_accuracy 0.5 at epoch 1: "
              f"{r['with_val_accuracy_0.5_at_epoch1']}.", ""]
    L += [f"Non-degenerate trainings (by the criterion; ablation run and folds run) with the first recovery after epoch "
          f"{t['non_degenerate_with_recovery_after_epoch']}: {t['non_degenerate_recovered_late']} of {t['non_degenerate_with_recovery']} "
          f"with recovery; never recovered: " + ("; ".join(who(r) for r in t["non_degenerate_without_recovery"]) or "none") + ".", ""]
    t = T["T7"]
    labs = [k for k in t["rows"][0] if k != "rule"]
    L += [f"## T7. {t['title']}", "", table(["rule"] + labs, [[r["rule"]] + [f"{r[k]['determinable']}/{r[k]['models']} determinable, "
                                                                            f"{r[k]['degenerate']} degenerate, restored median "
                                                                            f"{f4(r[k]['restored_median'])}"
                                                                            for k in labs] for r in t["rows"]]), "",
          "Non-degenerate: restored epoch of the run -> of the rule ('?' = not determinable):"]
    for c in t["changes"][1:]:
        items = [f"{who(x)}: {x['restored_run']} -> " + ("?" if x["restored_rule"] is None else
                                                          f"{x['restored_rule']} (val_acc {x['val_acc_run']:.4f} -> {x['val_acc_rule']:.4f})")
                 for x in c["non_degenerate_changes"]]
        L += [f"- {c['rule']}: " + ("; ".join(items) if items else "no change")]
    L += ["", "Non-degenerate trainings (by the criterion; ablation run and folds run) that would become degenerate under the rule:"]
    for b in t["become_degenerate"][1:]:
        L += [f"- {b['rule']}: " + ("; ".join(who(x) for x in b["non_degenerate_that_become_degenerate"]) or "none")]
    L += [""]
    t = T["T8"]
    L += [f"## T8. {t['title']}", "", table(["group", "models with a reduction", "first epoch at reduced lr", "reductions decided with the dev at 0.5",
                                            "lr at the last epoch"],
                                           [[r["group"], r["with_reduction"], r["first_reduced_epoch"], r["decided_with_dev_at_0.5"],
                                             ", ".join(f"{k}: {v}" for k, v in r["lr_at_last_epoch"].items())] for r in t["rows"]]), ""]
    t = T["T9"]
    labs = [k for k in t["rows"][0] if k != "epoch"]
    L += [f"## T9. {t['title']}", "", table(["epoch"] + labs, [[r["epoch"]] + [
        f"acc {r[k]['accuracy_median']:.4f} [{r[k]['accuracy_min']:.4f}-{r[k]['accuracy_max']:.4f}]; loss {r[k]['loss_median']:.4f} "
        f"[{r[k]['loss_min']:.4f}-{r[k]['loss_max']:.4f}]" if r[k]["models"] else "-" for k in labs] for r in t["rows"]]), "",
          f"Pairs with the same arm and fold (degenerate against another seed), train accuracy of epochs 1-{t['pair_epochs']}:"]
    L += [f"- {p['arm']} f{p['fold']}: degenerate s{p['degenerate_seed']} {p['degenerate_train_accuracy']} against s{p['other_seed']} "
          f"{p['other_train_accuracy']}" for p in t["pairs"]]
    L += [""]
    t = T["T10"]
    L += [f"## T10. {t['title']}", "", table(["arm", "fold", "seed", "deg", "restored", "acc 0.5", "share predicted attack", "mean score bona fide",
                                             "mean score attack", "dev AUROC", "distinct scores"],
                                            [[r["arm"], r["fold"], r["seed"], "D" if r["degenerate"] else "-", r["restored_epoch"], r["acc_05"],
                                              r["pred_attack_05"], r["mean_bona_fide"], r["mean_attack"], r["auroc"], r["distinct"]] for r in t["rows"]]), "",
          f"Dev AUROC of the non-degenerate: median {t['auroc_non_degenerate']['median']:.4f} [{t['auroc_non_degenerate']['min']:.4f}-"
          f"{t['auroc_non_degenerate']['max']:.4f}]; folds run: median {t['auroc_folds_run']['median']:.4f} [{t['auroc_folds_run']['min']:.4f}-"
          f"{t['auroc_folds_run']['max']:.4f}].", "",
          f"Distinct subjects in the dev per fold (count only): {counts(t['dev_subjects_per_fold'], 'fold ')}; dev frames: "
          f"{counts(t['dev_frames_per_fold'], 'fold ')}; folds run: {counts(t['dev_subjects_per_fold_folds_run'], 'fold ')}.", "",
          f"Agreement of the accuracy at 0.5 of the checkpoint (dev_scores) with val_acc_at_best of the card, largest difference: "
          f"{t['max_abs_difference_acc_05_val_acc_at_best']!r} (folds run {t['max_abs_difference_acc_05_val_acc_at_best_folds_run']!r}).", ""]
    t = T["T11"]
    L += [f"## T11. {t['title']}", "", table(["model", "same epochs", "same restored epoch", "largest history difference", "identical history",
                                             "same weights SHA-256", "train_s failed attempt", "train_s run"],
                                            [[r["model"], r["same_epochs"], r["same_restored_epoch"], r["max_abs_history_difference"],
                                              r["identical_history"], r["same_weights_sha256"], r["train_seconds_failed_attempt"],
                                              r["train_seconds_run"]] for r in t["rows"]]), "",
          f"Identical histories: {t['identical_histories']} of {t['models']} (a difference is reported, not an error).", ""]
    t = T["T12"]
    L += [f"## T12. {t['title']}", "", table(["arm", "models", "steps/epoch", "epochs", "train_seconds", "s/epoch median", "min", "max",
                                             "s/epoch weighted"],
                                            [[r["arm"], r["models"], r["steps_per_epoch"], r["epochs"], r["train_seconds"], r["seconds_per_epoch_median"],
                                              r["seconds_per_epoch_min"], r["seconds_per_epoch_max"], r["seconds_per_epoch_weighted"]] for r in t["rows"]]), "",
          f"Run: train_seconds {t['run_train_seconds']:.1f} s, wall {t['run_wall_seconds_status']:.1f} s (status.json; the run.log line "
          f"gives {t['run_wall_seconds']:.0f} s, whole seconds, used for the rest below), rest (checks, loading, scoring, aggregate) "
          f"{t['overhead_seconds']:.1f} s = {t['overhead_seconds_per_model']:.1f} s per model.",
          f"From the saved weights to the threshold (dev scores): median {t['saved_to_threshold_seconds']['median']:.0f} s "
          f"[{t['saved_to_threshold_seconds']['min']:.0f}-{t['saved_to_threshold_seconds']['max']:.0f}]; from the threshold to the test "
          f"scores: median {t['threshold_to_test_scores_seconds']['median']:.0f} s [{t['threshold_to_test_scores_seconds']['min']:.0f}-"
          f"{t['threshold_to_test_scores_seconds']['max']:.0f}] (resolution {t['time_resolution_seconds']} s)."]
    for s in t["scenarios"]:
        L += [f"- Scenario {s['label']} ({s['models']} models, {s['degenerate']} degenerate): training {s['training_seconds']:.0f} s "
              f"({s['training_hours']:.2f} h)" + (f", with the rest per model of the run {s['with_overhead_seconds']:.0f} s "
                                                   f"({s['with_overhead_hours']:.2f} h)" if s["with_overhead_seconds"] is not None else "") + "."]
    L += [f"Original run: {t['run_train_seconds']:.0f} s of training over {t['run_epochs']} epochs, wall {t['run_wall_seconds']:.0f} s "
          f"({t['run_wall_seconds'] / 3600:.2f} h).", ""]
    t = T["T13"]
    n = len(t["rows"][0]["residual"])
    L += [f"## T13. {t['title']}", "", table(["group", "after epoch 1"] + [f"epoch {e}" for e in range(2, n + 1)],
                                            [[r["group"]] + r["residual"] for r in t["rows"]]), "", f"({t['note']}.)", ""]
    L += ["## Declarations", ""] + [f"- {k}: {v}" for k, v in doc["declarations"].items()] + [""]
    return "\n".join(L)


def public_problems(text: str, inp: Inputs, markdown: bool = False) -> list:
    """Contenuto vietato in un file pubblico: chiavi d'immagine, identificativi di soggetto dei dev (parola intera, non
    dentro numeri o nomi), regole di package_rules, IPv4 non locali, segni di lavoro non finito (solo nei .md).
    Restituisce solo le categorie, mai il testo trovato: il motivo finisce in status.json e run.log."""
    bad = []
    if re.search(r"\b(real|attack)/\d{4}_", text):
        bad.append("image keys")
    for s in sorted(inp.subjects):
        if re.search(rf"(?<![\w.]){re.escape(s)}(?![\w.])", text):
            bad.append("subject id")
            break
    for label, rx in pr.FORBIDDEN_TEXT:
        if rx.search(text):
            bad.append(label)
    if any(ip not in pr.IP_ALLOW for ip in (m.group(0) for m in pr.IPV4.finditer(text))):
        bad.append("IP address")
    if markdown and pr.MARKERS.search(text):
        bad.append("unfinished-work marker")
    return bad


# ============================================================================ figure

def figure_data(cfg: dict, inp: Inputs) -> dict:
    """Dati delle tre figure (solo campi validati: braccio, fold, seed, storie, epoche ripristinata e di arresto)."""
    def curve(m):
        return {"label": (m["arm"] if m["group"] == "ablation_run" else "folds run") + f" fold {m['fold']} seed {m['seed']}",
                "status": "degenerate" if m["deg"] else "not degenerate", "recovered": m["rec"],
                "restored": m["card"]["restored_epoch"], "stopped": m["ep"],
                "max_epochs": m["card"]["max_epochs"], "steps": m["steps"], **{k: m["h"][k] for k in HISTORY_KEYS}}
    pool = inp.models["ablation_run"] + inp.models["folds_run"]
    ex = []
    for e in cfg["examples"]:
        m = next(x for x in pool if x["run"] == e["run"] and x["arm"] == e["arm"] and x["fold"] == e["fold"] and x["seed"] == e["seed"])
        ex.append({**curve(m), "why": e["why"]})
    spec = cfg["inputs"]["ablation_run"]
    order = [(a, s) for a in spec["arms"] for s in spec["seeds"]] + [(FOLDS_ARM, s) for s in cfg["inputs"]["folds_run"]["seeds"]]
    grid = [[curve(next(m for m in pool if m["arm"] == a and m["seed"] == s and m["fold"] == f)) for f in spec["folds"]] for a, s in order]
    return {"examples": ex, "grid": grid}


def default_plotter(kind: str, data: dict) -> bytes:
    """Figure pubbliche (matplotlib, Agg): esempi (accuratezza e loss per epoca, loss in scala log, epoca ripristinata e
    di arresto marcate) e piccoli multipli di val_accuracy e val_loss per tutti i modelli del run e del run dei fold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    red, blue, grey, green = "#c0392b", "#1f5fa8", "#7f7f7f", "#2e8b57"
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8})

    def marks(ax, c):
        ax.axvline(c["restored"], color=green, lw=1.2, ls="--")
        ax.axvline(c["stopped"], color=grey, lw=1.2, ls=":")

    if kind == "examples":
        ex = data["examples"]
        rows = (len(ex) + 1) // 2
        fig, axes = plt.subplots(rows, 4, figsize=(17, 4.4 * rows), squeeze=False)
        for i, c in enumerate(ex):
            r, k = divmod(i, 2)
            a1, a2 = axes[r][2 * k], axes[r][2 * k + 1]
            x = list(range(1, len(c["loss"]) + 1))
            col = red if c["status"] == "degenerate" else blue
            a1.plot(x, c["accuracy"], color=col, ls="--", marker="o", ms=3.5, label="train accuracy (training mode)")
            a1.plot(x, c["val_accuracy"], color=col, marker="s", ms=4, label="dev accuracy (inference)")
            a1.axhline(0.5, color=grey, lw=0.8, ls="-", alpha=0.6)
            a1.set_ylim(-0.02, 1.04)
            a1.set_ylabel("accuracy")
            a2.semilogy(x, c["loss"], color=col, ls="--", marker="o", ms=3.5, label="train loss")
            a2.semilogy(x, c["val_loss"], color=col, marker="s", ms=4, label="dev loss (val_loss)")
            a2.axhline(math.log(2), color=grey, lw=0.8, alpha=0.6)
            a2.set_ylabel("loss (log scale)")
            for ax in (a1, a2):
                marks(ax, c)
                ax.set_xlim(0.5, c["max_epochs"] + 0.5)
                ax.set_xticks(range(1, c["max_epochs"] + 1))
                ax.set_xlabel("epoch")
                ax.grid(alpha=0.25)
            st = "degenerate" if c["status"] == "degenerate" else NOT_DEG_LABEL
            a1.set_title(f"{i + 1}. {c['label']}\n{st}", loc="left", fontweight="bold", color=col)
            rec = f"recovered at epoch {c['recovered']}" if c["recovered"] else "never recovered"
            a2.set_title(f"restored epoch {c['restored']}, stopped at epoch {c['stopped']}\n{rec}", loc="left")
        for j in range(len(ex), rows * 2):
            r, k = divmod(j, 2)
            axes[r][2 * k].axis("off")
            axes[r][2 * k + 1].axis("off")
        handles = [Line2D([], [], color="k", ls="--", marker="o", ms=3.5, label="train (training mode: augmentation, dropout, batch statistics)"),
                   Line2D([], [], color="k", marker="s", ms=4, label="dev (inference mode)"),
                   Line2D([], [], color=green, ls="--", label="restored epoch (checkpoint)"),
                   Line2D([], [], color=grey, ls=":", label="last epoch run (stop)"),
                   Line2D([], [], color=grey, alpha=0.6, label="0.5 (accuracy) / ln 2 (loss)"),
                   Line2D([], [], color=red, lw=2, label="degenerate"), Line2D([], [], color=blue, lw=2, label=NOT_DEG_LABEL)]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
        fig.suptitle("Representative trainings: train and dev curves per epoch (left: accuracy; right: loss, log scale)", fontsize=11)
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    else:
        grid = data["grid"]
        key = "val_accuracy" if kind == "val_accuracy_all" else "val_loss"
        nr, nc = len(grid), len(grid[0])
        fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc + 1, 1.55 * nr + 1.2), sharex=True, sharey=True, squeeze=False)
        for r, row in enumerate(grid):
            for k, c in enumerate(row):
                ax = axes[r][k]
                x = list(range(1, len(c[key]) + 1))
                deg = c["status"] == "degenerate"
                if deg:
                    ax.set_facecolor("#fbeaea")
                ax.plot(x, c[key], color=red if deg else blue, lw=1.6 if deg else 1.1, marker="o", ms=2.2)
                ax.plot([c["restored"]], [c[key][c["restored"] - 1]], marker="*", ms=9, color=green, ls="none")
                if key == "val_loss":
                    ax.set_yscale("log")
                    ax.axhline(math.log(2), color=grey, lw=0.7, alpha=0.6)
                else:
                    ax.axhline(0.5, color=grey, lw=0.7, alpha=0.6)
                    ax.set_ylim(-0.03, 1.05)
                ax.set_xlim(0.5, c["max_epochs"] + 0.5)
                ax.set_xticks([1, 4, 8, 12] if c["max_epochs"] == 12 else range(1, c["max_epochs"] + 1, 2))
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.2)
                ax.set_title(c["label"] + (" - degenerate" if deg else ""), fontsize=7.5, color=red if deg else "black",
                             fontweight="bold" if deg else "normal")
        handles = [Line2D([], [], color=blue, lw=1.1, label=NOT_DEG_LABEL), Line2D([], [], color=red, lw=1.6, label="degenerate (shaded)"),
                   Line2D([], [], color=green, marker="*", ms=9, ls="none", label="restored epoch"),
                   Line2D([], [], color=grey, alpha=0.6, label="0.5" if key == "val_accuracy" else "ln 2")]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
        what = "dev accuracy (val_accuracy)" if key == "val_accuracy" else "dev loss (val_loss, log scale)"
        fig.suptitle(f"{what} per epoch: the 36 trainings of the ablation run (rows: arm and seed) and the 12 of the folds run "
                     "(last three rows); columns: fold", fontsize=10)
        fig.supxlabel("epoch (star: restored epoch; the curve ends at the last epoch run)", fontsize=9, y=0.028)
        fig.supylabel(what, fontsize=9)
        fig.tight_layout(rect=(0, 0.03, 1, 0.975))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, metadata={"Software": None})
    plt.close(fig)
    return buf.getvalue()


def figure_bytes(cfg: dict, inp: Inputs) -> tuple:
    """({nome: byte PNG}, nota, dati delle figure): matplotlib assente -> salto dichiarato, mai un errore."""
    data = figure_data(cfg, inp)
    kinds = {"figure_examples.png": "examples", "figure_val_accuracy_all.png": "val_accuracy_all", "figure_val_loss_all.png": "val_loss_all"}
    try:
        pngs = {n: (PLOTTER or default_plotter)(k, data) for n, k in kinds.items()}
    except ImportError as e:
        return {}, f"skipped: missing Python package {e.name!r}", data
    for n, b in pngs.items():
        if not isinstance(b, bytes) or not b:
            fail(f"figure {n}: the plotter returned no PNG bytes")
    return pngs, f"{', '.join(pngs)} written", data


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


def build_public(cfg: dict, cfg_sha: str, inp: Inputs, head: dict) -> dict:
    """File pubblici (testi e PNG), scansionati prima della scrittura; nessuna data né tempo d'esecuzione."""
    enrich(cfg, inp)
    tables = analyse(cfg, inp)
    pngs, note, fig_data = figure_bytes(cfg, inp)
    n_models = {inp.checks[g]["run_id"]: len(inp.models[g]) for g in GROUPS}
    curves = epoch_rows(inp)
    doc = {"run_id": head["run_id"], "commit": head["commit"], "dirty": head["dirty"], "config_sha256": cfg_sha,
           "delivered_configuration": cfg_sha == CONFIG_SHA256, "request": cfg["request"], "task": cfg["task"],
           "declarations": cfg["declarations"], "recipe": cfg["recipe"], "criteria": cfg["criteria"],
           "alternative_rules": cfg["alternative_rules"], "examples": cfg["examples"],
           "inputs": {g: {**inp.checks[g], "folder": cfg["inputs"][g]["folder"]} for g in GROUPS},
           "counts": {"models": n_models, "curves_rows": len(curves), "early_stopping_replay": f"{sum(n_models.values())}/{sum(n_models.values())} cards reproduced"},
           "restricted": RESTRICTED_NOTE, "figures": {"files": sorted(pngs), "note": note},
           "environment": environment(), "code_sha256": code_sha256(), "tables": tables}
    public = {"curves.csv": csv_text(EPOCH_COLUMNS, curves), "models.csv": csv_text(MODEL_COLUMNS, model_rows(inp)),
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
    rid = f"{date}-DEGEN-{state['commit']}" + ("-dev" if state["dirty"] else "")
    if not RUN_ID_RE.match(rid):
        fail(f"run id {rid} does not follow YYYYMMDD-DEGEN-<commit>[-dev]")
    return rid


def protected_extra(root: Path) -> list:
    return [root / x for x in EXTRA_PROTECTED]


def input_paths(cfg: dict, base: Path, cfg_p: Path) -> dict:
    return {"the configuration": cfg_p, **{f"the {g.replace('_', ' ')}": rel_path(base, cfg["inputs"][g]["folder"]) for g in GROUPS}}


def run_dir_problem(run: Path, root: Path, inputs: dict, out_root: Path) -> str:
    """Motivo per cui la cartella del run non è ammessa ("" se lo è), prima di qualunque modifica."""
    if out_root.is_symlink():
        return "the output root is a symbolic link"
    for label, p in inputs.items():
        if pg.within(p, run) or (Path(p).is_dir() and pg.within(run, p)):
            return f"{label} {Path(p).name} is inside the run folder, or contains it: inputs and outputs must be separate"
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
    return (f"ablation run {c['ablation_run']['run_id']} ({c['ablation_run']['cards']} cards and dev scores, {c['ablation_run']['threshold_files']} "
            f"thresholds), folds run {c['folds_run']['run_id']} ({c['folds_run']['cards']} cards and dev scores, {c['folds_run']['threshold_files']} "
            f"thresholds), failed attempt "
            f"{c['failed_attempt']['run_id']} ({c['failed_attempt']['cards']} cards); early stopping replay reproduced on "
            f"{sum(len(inp.models[g]) for g in GROUPS)}/{sum(len(inp.models[g]) for g in GROUPS)} cards")


def cmd_check_inputs(a) -> int:
    base, cfg_p = resolve(a)
    cfg, cfg_sha = load_config(cfg_p)
    inp = load_inputs(cfg, base)
    print(f"CHECK-INPUTS: configuration {cfg_sha[:12]} (delivered: {cfg_sha == CONFIG_SHA256}), {summary_line(inp)}; nothing written")
    print("RESULT: PASS (check-inputs)")
    return 0


def cmd_run(a) -> int:
    if not a.i_am_authorized:
        fail("run reads the restricted training cards and dev scores of two runs: pass --i-am-authorized (study requested by the "
             "coordinator on 6 November 2026)")
    base, cfg_p = resolve(a)
    cfg, cfg_sha = load_config(cfg_p)
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
        print(f"== NUAA degenerate trainings {rid}: configuration {cfg_sha[:12]} (delivered: {head['delivered_configuration']})", flush=True)
        inp = load_inputs(cfg, base)
        print(f"   check-inputs PASS: {summary_line(inp)}", flush=True)
        public, doc = build_public(cfg, cfg_sha, inp, head)
        write_public(run / PUBLIC, public)
        t = doc["tables"]
        print(f"degenerate: {t['T1']['count']}/{t['T1']['of']} (per arm {t['T1']['per_arm']}, per seed {t['T1']['per_seed']}, "
              f"per fold {t['T1']['per_fold']}); curves rows {doc['counts']['curves_rows']}; identical histories "
              f"{t['T11']['identical_histories']}/{t['T11']['models']}")
        print(f"figures: {doc['figures']['note']}")
        print(f"run folder: {run.name} (public/ only; no restricted outputs)")
        print("RESULT: PASS (run)")
        close_log(log)
        log = None
        write_status(run, {**head, "status": "PASS", "passed": True, "finished": now(), "wall_seconds": round(time.time() - t0, 1),
                           "figures": doc["figures"]["note"], "public_files": len(public) - 1,
                           "public_sums_sha256": sha256_bytes(public["SHA256SUMS"].encode("utf-8")),
                           "run_log_sha256": sha256_file(run / LOG),
                           "timestamps": "status.json and run.log only; the public files hold no date or run time"})
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
    ap = argparse.ArgumentParser(description="NUAA: the degenerate trainings of the session ablation run from the recorded train/dev curves")
    ap.add_argument("cmd", choices=["check-inputs", "run"])
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="configuration JSON (relative to the repository root)")
    ap.add_argument("--input-root", default=None, help="base folder of the input paths of the configuration (default: repository root)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="folder that receives <run-id>/ (default results/degenerate)")
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
