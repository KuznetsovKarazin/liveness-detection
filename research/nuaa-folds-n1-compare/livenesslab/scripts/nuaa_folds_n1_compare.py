"""
Confronto descrittivo fra il run LivenessNet a quattro fold per soggetto × tre seed (20261103-FOLDSRUN-e174812) e N1
(20261024-N1-LivenessNet-fb93496) sugli stessi 3.000 frame del manifest nuaa_cnn_test, chiesto dal coordinatore il
4/11/2026 sulla scheda C2: «il confronto descrittivo con N1 sugli stessi 3.000 ID, usando gli score esistenti e le
soglie già congelate».

Solo lettura di punteggi già scritti dai due run e delle soglie già congelate: nessun training, nessuna inferenza,
nessuna immagine letta, nessuna soglia nuova. Il confronto è successivo ai due run e descrittivo (nessuna causalità).

Passi, in un solo processo:
  1. check-inputs   configurazione a insieme chiuso con gli SHA-256 ancorati; manifest cnn_test (SHA-256 uguale a
                    quello della configurazione dello schema dei fold, chiave manifests.cnn_test) e manifest di
                    training di N1 (cnn_train, cnn_es); run dei fold: status.json, SHA256SUMS pubblico e riservato
                    ancorati, i 12 file di punteggio del test e le 12 soglie per SHA-256 registrato, ogni riga coerente
                    con la sua soglia congelata e con la regola di decisione; per seed ogni chiave di cnn_test nel test
                    di esattamente un fold, ogni soggetto nel test di un solo fold; N1: SHA256SUMS ancorato, i tre file
                    di punteggio con le stesse 3.000 chiavi; label, soggetto, sessione e file_sha256 uguali al
                    manifest; ancore: le metriche pubblicate dai due run ricalcolate dai punteggi; valori di controllo
                    del coordinatore riprodotti. Nulla scritto.
  2. run            gli stessi controlli, poi per seed: metriche dei due esperimenti sui 3.000 frame, differenze
                    (fold − N1), mediana e intervallo sui seed, bootstrap appaiato sui soggetti, tabella 2×2 delle
                    decisioni per frame, conteggi per soggetto; uscite pubbliche senza identificativi e riservate.

Uscite (cartella del run <out-root>/YYYYMMDD-FOLDSCOMPARE-<commit>[-dev], -dev se l'albero git non è pulito):
  status.json, run.log
  public/      aggregate.json, aggregate.md, SHA256SUMS            (nessun identificativo, nessuna chiave)
  restricted/  paired_frames.csv, per_subject.csv, public_SHA256SUMS.sha256, SHA256SUMS
Dentro il repository la cartella del run è ammessa solo sotto results/foldscompare/.

  .venv/bin/python -I scripts/nuaa_folds_n1_compare.py check-inputs                  # solo verifica, nulla scritto
  .venv/bin/python -I scripts/nuaa_folds_n1_compare.py run --i-am-authorized         # confronto e uscite

Dipendenze: NumPy; scripts/m2_writers.py, scripts/path_guard.py, scripts/package_rules.py, scripts/nuaa_manifests.py
(load_manifest, con scripts/select_threshold.py e src/tesi_app/paths.py).
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
import textwrap
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT                    # radice del codice registrato (i test possono spostare ROOT, mai questa)
sys.path.insert(0, str(ROOT / "scripts"))
import m2_writers as mw  # noqa: E402
import nuaa_manifests as nm  # noqa: E402  (load_manifest; importa select_threshold e src/tesi_app/paths)
import package_rules as pr  # noqa: E402
import path_guard as pg  # noqa: E402

DEFAULT_CONFIG = "delivery/foldscompare/nuaa_folds_n1_compare.json"
DEFAULT_OUT_ROOT = "results/foldscompare"
CONFIG_SCHEMA = "nuaa-folds-n1-compare-config/1"
CONFIG_SHA256 = "78fd1424959f807328ad979aa1381cce2488453b5315f030e60888dcc4b003ae"   # configurazione consegnata, fissata prima del run
SCHEMA_CONFIG_SHA256 = "d1d43c148f762965c5ae367a0de1c59770324f7840034fd5a6c7fad0a14e73e8"   # delivery/folds/nuaa_subject_folds.json
RUN_ID_RE = re.compile(r"^\d{8}-FOLDSCOMPARE-[0-9a-f]{7,40}(-dev)?$")
TOP_KEYS = ("schema", "task", "request", "manifest", "n1_training_manifests", "schema_config", "folds_run", "n1_run",
            "train_seeds", "decision", "metrics", "bootstrap", "control_values", "declarations", "outputs")
TRAIN_SEEDS = [42, 43, 44]          # seed dei due run (fissati dal coordinatore)
FIXED_THRESHOLD = 0.5               # soglia fissa di C1, N1 e N2
DECISION_RULE = "attack if score_attack > threshold (tie = bona fide)"
BOOT_RECIPE = ("generator numpy.random.default_rng(0) (PCG64); subjects in ascending identifier order; at every resample "
               "integers(0, n, n) with n = the number of subjects (16); the generator restarts from 0 for every training seed "
               "(the same 1,000 resamples for 42, 43 and 44); every metric computed on the pooled frames of the resampled "
               "subjects, not as a mean of per-subject values; the same subject indices for both experiments; percentiles "
               "2.5 and 97.5 with np.percentile, linear interpolation; a resample without a class does not count for the "
               "metrics that need it; no interval with fewer than 5 subjects")
BOOT = {"resamples": 1000, "seed": 0, "percentiles": [2.5, 97.5], "min_subjects": 5, "recipe": BOOT_RECIPE}
DECLARATION_FLAGS = ("descriptive", "post_hoc", "no_training", "no_inference", "no_images_read", "no_new_threshold")
DECLARATION_TEXTS = ("training_roles", "auroc_four_models", "n1_threshold", "indicative", "no_causality", "limits")
LABELS = ("0", "1")
ANCHOR_TOL = 1e-12
# valori di controllo del coordinatore: accordo alle cifre che ha dato (ACER in percentuale a 2 decimali, AUROC a 5)
CONTROL_DIGITS = {"acer": 2, "auroc": 5}
CONTROL_KEYS = {"acer_05": "acer_05_percent", "acer_dev": "acer_dev_percent", "auroc": "auroc"}
CONTROL_COMPARISON = ("agreement at the digits given by the coordinator: round(100 * ACER, 2) equal to the given percentage, "
                      "round(AUROC, 5) equal to the given AUROC")                  # ricalcolo delle metriche pubblicate dai due run (stessi punteggi, stesso criterio)
MANIFEST_SPEC_KEYS = ["file", "folder", "n_attack", "n_bona_fide", "n_subjects", "rows", "session", "sha256"]
FOLD_SCORE_COLUMNS = ("fold", "train_seed", "role", "sample_id", "subject_id", "session_id", "label", "score_attack",
                      "prediction_05", "prediction_dev", "threshold_dev", "file_sha256", "checkpoint_sha256", "threshold_sha256")
N1_SCORE_COLUMNS = ("run_id", "train_seed", "sample_id", "subject_id", "session_id", "label", "score_attack", "prediction",
                    "in_c1_subset", "file_sha256", "checkpoint_sha256", "config_sha256")
PAIRED_COLUMNS = ("train_seed", "sample_id", "subject_id", "session_id", "label", "n1_score_attack", "n1_prediction_05",
                  "fold", "fold_score_attack", "fold_threshold_dev", "fold_prediction_05", "fold_prediction_dev")
SUBJECT_COLUMNS = ("train_seed", "subject_id", "fold", "n_bona_fide", "n_attack", "n1_errors_05", "n1_error_rate_05",
                   "n1_acer_05", "folds_errors_05", "folds_error_rate_05", "folds_acer_05", "folds_errors_dev",
                   "folds_error_rate_dev", "folds_acer_dev", "n1_auroc", "folds_auroc")
# metriche riassunte sui seed e intervalli del bootstrap (esperimento_metrica; diff = fold − N1)
SUMMARY_KEYS = ("folds_bpcer_05", "folds_apcer_05", "folds_acer_05", "folds_bpcer_dev", "folds_apcer_dev", "folds_acer_dev",
                "folds_auroc", "n1_bpcer_05", "n1_apcer_05", "n1_acer_05", "n1_auroc", "diff_acer_05", "diff_acer_dev",
                "diff_auroc")
BOOT_KEYS = ("folds_acer_05", "folds_acer_dev", "folds_auroc", "n1_acer_05", "n1_auroc", "diff_acer_05", "diff_acer_dev",
             "diff_auroc")
CODE_FILES = ("scripts/nuaa_folds_n1_compare.py", "scripts/m2_writers.py", "scripts/path_guard.py", "scripts/package_rules.py",
              "scripts/nuaa_manifests.py", "scripts/select_threshold.py", "src/tesi_app/paths.py")
STATUS = "status.json"
LOG = "run.log"
PUBLIC, RESTRICTED = "public", "restricted"
PUBLIC_OUTPUTS = ("aggregate.json", "aggregate.md", "SHA256SUMS")
RESTRICTED_OUTPUTS = ("paired_frames.csv", "per_subject.csv", "public_SHA256SUMS.sha256", "SHA256SUMS")
FAIL_DOC = {"stage": "invalidation-failed", "note": "this output could not be removed after a failure: no verdict is valid"}
EXTRA_PROTECTED = ("models", "results/c1", "results/m1", "results/m2", "results/m2_rgb", "results/eval", "results/n1",
                   "results/n2", "results/c1crop", "results/folds", "results/foldsrun", "results/stepzero", "export")
NA = "NA"
MD_WIDTH = 118

REPO_STATE = None       # (root) -> {"commit", "dirty"}; None: git nel repository (sostituibile solo dai test)


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
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


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
    return "".join(f"{sha256_bytes(t.encode('utf-8'))}  {n}\n" for n, t in sorted(files.items()))


def f4(v) -> str:
    return NA if v is None else f"{v:.4f}"


def f5(v) -> str:
    return NA if v is None else f"{v:.5f}"


def fmt(v) -> str:
    return NA if v is None else repr(float(v))


def rel_path(base: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else base / q


def guarded(what: str, fn, *args):
    """Chiama una funzione di nuaa_manifests.py (che si ferma con sys.exit dopo una riga ERROR) come Failure."""
    try:
        return fn(*args)
    except SystemExit:
        fail(f"{what}: invalid input (see the ERROR line above)")


# ============================================================================ configurazione

def is_count(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _hex(v) -> bool:
    return isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) is not None


def _text(v, what: str) -> None:
    if not isinstance(v, str) or not v.strip():
        fail(f"configuration: {what} must be a non-empty text")


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def load_config(p: Path) -> tuple:
    """(configurazione, SHA-256 dei byte letti). Insiemi chiusi e valori fissati controllati qui, prima di tutto."""
    data = read_regular(p, "configuration")
    cfg = read_json(data, f"configuration {p.name}")
    if cfg.get("schema") != CONFIG_SCHEMA:
        fail(f"configuration {p.name}: schema must be {CONFIG_SCHEMA!r}")
    if sorted(cfg) != sorted(TOP_KEYS):
        fail(f"configuration: fields must be exactly {', '.join(TOP_KEYS)} (missing {sorted(set(TOP_KEYS) - set(cfg))}, "
             f"unknown {sorted(set(cfg) - set(TOP_KEYS))})")
    _text(cfg["task"], "task")
    rq = cfg["request"]
    if not isinstance(rq, dict) or sorted(rq) != ["card", "date", "text"]:
        fail("configuration: request needs date, card and text")
    for k in rq:
        _text(rq[k], f"request.{k}")
    m = cfg["manifest"]
    if not isinstance(m, dict) or sorted(m) != MANIFEST_SPEC_KEYS:
        fail(f"configuration: manifest needs exactly {', '.join(MANIFEST_SPEC_KEYS)}")
    for k in ("file", "folder", "session"):
        _text(m[k], f"manifest.{k}")
    if not _hex(m["sha256"]):
        fail("configuration: manifest.sha256 must be a SHA-256")
    for k in ("rows", "n_bona_fide", "n_attack", "n_subjects"):
        if not is_count(m[k]) or m[k] == 0:
            fail(f"configuration: manifest.{k} must be a positive integer")
    if m["n_bona_fide"] + m["n_attack"] != m["rows"]:
        fail("configuration: manifest.n_bona_fide + manifest.n_attack must equal manifest.rows")
    tm = cfg["n1_training_manifests"]
    if not isinstance(tm, dict) or sorted(tm) != ["files", "sessions", "training_subjects_in_set"]:
        fail("configuration: n1_training_manifests needs files, sessions and training_subjects_in_set")
    if not isinstance(tm["files"], dict) or sorted(tm["files"]) != ["cnn_es", "cnn_train"]:
        fail("configuration: n1_training_manifests.files must define exactly cnn_train and cnn_es")
    for k, v in tm["files"].items():
        if not isinstance(v, dict) or sorted(v) != ["file", "sha256"] or not _hex(v["sha256"]):
            fail(f"configuration: n1_training_manifests.files.{k} needs file and sha256")
        _text(v["file"], f"n1_training_manifests.files.{k}.file")
    if not isinstance(tm["sessions"], list) or not tm["sessions"] or not all(isinstance(s, str) for s in tm["sessions"]):
        fail("configuration: n1_training_manifests.sessions must list sessions")
    if not is_count(tm["training_subjects_in_set"]):
        fail("configuration: n1_training_manifests.training_subjects_in_set must be a non-negative integer")
    sc = cfg["schema_config"]
    if not isinstance(sc, dict) or sorted(sc) != ["path", "sha256"] or not _hex(sc["sha256"]):
        fail("configuration: schema_config needs path and sha256")
    _text(sc["path"], "schema_config.path")
    fr = cfg["folds_run"]
    fr_keys = ["folder", "k", "public_sums_sha256", "restricted_sums_sha256", "run_id", "status_sha256"]
    if not isinstance(fr, dict) or sorted(fr) != fr_keys:
        fail(f"configuration: folds_run needs exactly {', '.join(fr_keys)}")
    for k in ("folder", "run_id"):
        _text(fr[k], f"folds_run.{k}")
    for k in ("public_sums_sha256", "restricted_sums_sha256", "status_sha256"):
        if not _hex(fr[k]):
            fail(f"configuration: folds_run.{k} must be a SHA-256")
    if not is_count(fr["k"]) or fr["k"] < 2:
        fail("configuration: folds_run.k must be an integer >= 2")
    n1 = cfg["n1_run"]
    if not isinstance(n1, dict) or sorted(n1) != ["folder", "run_id", "sha256sums_sha256", "threshold"]:
        fail("configuration: n1_run needs folder, run_id, sha256sums_sha256 and threshold")
    for k in ("folder", "run_id"):
        _text(n1[k], f"n1_run.{k}")
    if not _hex(n1["sha256sums_sha256"]):
        fail("configuration: n1_run.sha256sums_sha256 must be a SHA-256")
    if n1["threshold"] != FIXED_THRESHOLD:
        fail(f"configuration: n1_run.threshold must be {FIXED_THRESHOLD} (the fixed threshold of N1)")
    seeds = cfg["train_seeds"]
    if seeds != TRAIN_SEEDS:
        fail(f"configuration: train_seeds must be {TRAIN_SEEDS} (the seeds of both runs)")
    if cfg["decision"] != DECISION_RULE:
        fail(f"configuration: decision must be {DECISION_RULE!r}")
    me = cfg["metrics"]
    me_keys = ["auroc", "differences", "paired_frames", "per_seed", "per_subject", "summary"]
    if not isinstance(me, dict) or sorted(me) != me_keys:
        fail(f"configuration: metrics needs exactly {', '.join(me_keys)}")
    for k in me_keys:
        _text(me[k], f"metrics.{k}")
    if cfg["bootstrap"] != BOOT:
        fail(f"configuration: bootstrap must be {json.dumps(BOOT)} (the recipe of the folds run)")
    cv = cfg["control_values"]
    if not isinstance(cv, dict) or sorted(cv) != ["acer_05_percent", "acer_dev_percent", "auroc", "comparison", "source"]:
        fail("configuration: control_values needs source, acer_05_percent, acer_dev_percent, auroc and comparison")
    _text(cv["source"], "control_values.source")
    for k, top in (("acer_05_percent", 100), ("acer_dev_percent", 100), ("auroc", 1)):
        if not isinstance(cv[k], list) or len(cv[k]) != len(seeds) or not all(_number(v) and 0 <= v <= top for v in cv[k]):
            fail(f"configuration: control_values.{k} must list one value in [0, {top}] per seed")
    if cv["comparison"] != CONTROL_COMPARISON:
        fail(f"configuration: control_values.comparison must be {CONTROL_COMPARISON!r}")
    dec = cfg["declarations"]
    if not isinstance(dec, dict) or sorted(dec) != sorted(DECLARATION_FLAGS + DECLARATION_TEXTS):
        fail(f"configuration: declarations must have exactly the keys {', '.join(DECLARATION_FLAGS + DECLARATION_TEXTS)}")
    for k in DECLARATION_FLAGS:
        if dec[k] is not True:
            fail(f"configuration: declarations.{k} must be true")
    for k in DECLARATION_TEXTS:
        _text(dec[k], f"declarations.{k}")
    ou = cfg["outputs"]
    if not isinstance(ou, dict) or sorted(ou) != ["public", "restricted"]:
        fail("configuration: outputs needs public and restricted")
    for k in ou:
        _text(ou[k], f"outputs.{k}")
    return cfg, sha256_bytes(data)


# ============================================================================ ingressi

class Inputs:
    """Manifest, punteggi dei due run per seed, soglie congelate e l'esito di ogni controllo (checks)."""

    def __init__(self):
        self.manifest = []          # righe di cnn_test (load_manifest)
        self.by_key = {}
        self.subjects = set()       # soggetti di cnn_test
        self.n1_training_subjects = set()
        self.folds = {}             # seed -> chiave -> riga del run dei fold (solo cnn_test)
        self.folds_all = {}         # seed -> tutte le righe dei 4 file del test (ancora pubblica)
        self.fold_of_subject = {}   # seed -> soggetto -> fold
        self.thresholds = {}        # (fold, seed) -> {"value", "sha256"}
        self.n1 = {}                # seed -> chiave -> riga di N1
        self.folds_aggregate = {}
        self.n1_aggregate = {}
        self.n1_metrics = {}
        self.checks = {}

    @property
    def image_sha(self) -> set:
        return {r["sha256"] for r in self.manifest}


def counts_of(rows) -> dict:
    return {"rows": len(rows), "n_bona_fide": sum(1 for r in rows if r["label"] == "0"),
            "n_attack": sum(1 for r in rows if r["label"] == "1"), "n_subjects": len({r["subject"] for r in rows})}


def load_manifests(cfg: dict, base: Path, inp: Inputs) -> None:
    """Configurazione dello schema dei fold (SHA-256 fissato) e, per SHA-256 uguale a quello registrato lì, il manifest
    cnn_test (conteggi, sessione, chiavi e contenuti unici) e i manifest di training di N1 (cnn_train, cnn_es)."""
    sc = cfg["schema_config"]
    p = rel_path(base, sc["path"])
    b = read_regular(p, "schema configuration")
    h = sha256_bytes(b)
    if h != sc["sha256"] or h != SCHEMA_CONFIG_SHA256:
        fail(f"schema configuration {p.name}: SHA-256 {h} differs from the approved schema ({SCHEMA_CONFIG_SHA256[:12]}...)")
    schema = read_json(b, f"schema configuration {p.name}")
    reg = schema.get("manifests") if isinstance(schema.get("manifests"), dict) else {}
    m = cfg["manifest"]
    spec = reg.get("cnn_test")
    if not isinstance(spec, dict) or spec.get("sha256") != m["sha256"] or spec.get("file") != m["file"]:
        fail("manifest cnn_test: file and SHA-256 differ from manifests.cnn_test of the schema configuration")
    for k, v in cfg["n1_training_manifests"]["files"].items():
        s2 = reg.get(k)
        if not isinstance(s2, dict) or s2.get("sha256") != v["sha256"] or s2.get("file") != v["file"]:
            fail(f"manifest {k}: file and SHA-256 differ from manifests.{k} of the schema configuration")
    mdir = rel_path(base, m["folder"])
    mp = mdir / m["file"]
    hb = sha256_bytes(read_regular(mp, "manifest cnn_test"))
    if hb != m["sha256"]:
        fail(f"manifest cnn_test ({m['file']}): SHA-256 {hb} differs from the configuration ({m['sha256']})")
    rows = guarded("manifest cnn_test", nm.load_manifest, mp)
    c = counts_of(rows)
    for k, v in c.items():
        if m[k] != v:
            fail(f"manifest cnn_test: {k} = {v} differs from the configuration ({m[k]})")
    if any(r["session"] != m["session"] for r in rows):
        fail(f"manifest cnn_test: a row has a session other than {m['session']}")
    if len({r["key"] for r in rows}) != len(rows) or len({r["sha256"] for r in rows}) != len(rows):
        fail("manifest cnn_test: two rows share a key or a file SHA-256")
    inp.manifest = rows
    inp.by_key = {r["key"]: r for r in rows}
    inp.subjects = {r["subject"] for r in rows}
    tm = cfg["n1_training_manifests"]
    train_subjects, train_keys = set(), set()
    for k, v in sorted(tm["files"].items()):
        tp = mdir / v["file"]
        th = sha256_bytes(read_regular(tp, f"manifest {k}"))
        if th != v["sha256"]:
            fail(f"manifest {k} ({v['file']}): SHA-256 {th} differs from the configuration ({v['sha256']})")
        trows = guarded(f"manifest {k}", nm.load_manifest, tp)
        if any(r["session"] not in tm["sessions"] for r in trows):
            fail(f"manifest {k}: a row has a session outside {tm['sessions']}")
        train_subjects |= {r["subject"] for r in trows}
        train_keys |= {r["key"] for r in trows}
    if train_keys & set(inp.by_key):
        fail("the N1 training manifests share keys with cnn_test")
    inp.n1_training_subjects = train_subjects
    overlap = len(train_subjects & inp.subjects)
    if overlap != tm["training_subjects_in_set"]:
        fail(f"N1 training subjects in cnn_test: {overlap}, the configuration declares {tm['training_subjects_in_set']}")
    inp.checks["manifest"] = {"name": "cnn_test", "sha256": m["sha256"], **c, "session": m["session"],
                              "schema_config_sha256": h, "n1_training_subjects": len(train_subjects),
                              "n1_training_subjects_in_set": overlap}


def finite_score(v: str, what: str) -> float:
    try:
        x = float(v)
    except ValueError:
        fail(f"{what}: score_attack {v[:20]!r} is not a number")
    if not math.isfinite(x) or not 0.0 <= x <= 1.0:
        fail(f"{what}: score_attack {v[:20]!r} outside [0, 1]")
    return x


def same_as_manifest(r: dict, o: dict) -> bool:
    return (r["label"] == o["label"] and r["subject_id"] == o["subject"] and r["session_id"] == o["session"]
            and r["file_sha256"] == o["sha256"])


def load_folds_run(cfg: dict, base: Path, inp: Inputs) -> None:
    """Run dei fold: status.json, SHA256SUMS pubblico e riservato ancorati; per fold e seed il file dei punteggi del test
    e la soglia congelata per SHA-256 registrato; righe coerenti (fold, seed, ruolo, soglia, decisioni, checkpoint); per
    seed ogni chiave di cnn_test nel test di esattamente un fold, uguale al manifest; ogni soggetto in un solo fold."""
    fr = cfg["folds_run"]
    folder = rel_path(base, fr["folder"])
    sb = read_regular(folder / STATUS, "folds run status.json")
    if sha256_bytes(sb) != fr["status_sha256"]:
        fail(f"folds run status.json: SHA-256 {sha256_bytes(sb)} differs from the configuration ({fr['status_sha256']})")
    st_doc = read_json(sb, "folds run status.json")
    if st_doc.get("status") != "PASS" or st_doc.get("passed") is not True or st_doc.get("run_id") != fr["run_id"]:
        fail(f"folds run status.json: not a PASS of run {fr['run_id']}")
    if st_doc.get("public_sums_sha256") != fr["public_sums_sha256"] or st_doc.get("restricted_sums_sha256") != fr["restricted_sums_sha256"]:
        fail("folds run status.json: the SHA256SUMS anchors differ from the configuration")
    pb = read_regular(folder / PUBLIC / "SHA256SUMS", "folds run public SHA256SUMS")
    if sha256_bytes(pb) != fr["public_sums_sha256"]:
        fail(f"folds run public SHA256SUMS: SHA-256 {sha256_bytes(pb)} differs from status.json ({fr['public_sums_sha256']})")
    plisted = read_sums(pb, "folds run public SHA256SUMS")
    ab = read_regular(folder / PUBLIC / "aggregate.json", "folds run aggregate.json")
    if plisted.get("aggregate.json") != sha256_bytes(ab):
        fail("folds run aggregate.json: SHA-256 differs from the public SHA256SUMS")
    inp.folds_aggregate = read_json(ab, "folds run aggregate.json")
    if inp.folds_aggregate.get("run_id") != fr["run_id"]:
        fail("folds run aggregate.json: run_id differs from the configuration")
    rb = read_regular(folder / RESTRICTED / "SHA256SUMS", "folds run restricted SHA256SUMS")
    if sha256_bytes(rb) != fr["restricted_sums_sha256"]:
        fail(f"folds run restricted SHA256SUMS: SHA-256 {sha256_bytes(rb)} differs from status.json ({fr['restricted_sums_sha256']})")
    listed = read_sums(rb, "folds run restricted SHA256SUMS")
    k = fr["k"]
    for seed in cfg["train_seeds"]:
        inp.folds[seed], inp.folds_all[seed], inp.fold_of_subject[seed] = {}, [], {}
        seen = Counter()
        for fold in range(1, k + 1):
            name = f"fold{fold}_seed{seed}"
            files = {}
            for suffix in ("test_scores.csv", "threshold.json"):
                fn = f"{name}_{suffix}"
                if fn not in listed:
                    fail(f"folds run restricted SHA256SUMS: {fn} not listed")
                b = read_regular(folder / RESTRICTED / fn, f"folds run {fn}")
                if sha256_bytes(b) != listed[fn]:
                    fail(f"folds run {fn}: SHA-256 {sha256_bytes(b)} differs from the restricted SHA256SUMS ({listed[fn]})")
                files[suffix] = b
            ckpt = listed.get(f"{name}_LivenessNet.keras")
            if ckpt is None:
                fail(f"folds run restricted SHA256SUMS: {name}_LivenessNet.keras not listed")
            thr_sha = sha256_bytes(files["threshold.json"])
            thr = read_json(files["threshold.json"], f"folds run {name}_threshold.json")
            value = thr.get("value")
            if thr.get("fold") != fold or thr.get("train_seed") != seed or thr.get("set") != "dev fold only" \
                    or not _number(value) or thr.get("checkpoint_sha256") != ckpt:
                fail(f"folds run {name}_threshold.json: fold, seed, set, value or checkpoint differ from {name}")
            inp.thresholds[(fold, seed)] = {"value": float(value), "sha256": thr_sha}
            what = f"folds run {name}_test_scores.csv"
            rows = parse_csv(files["test_scores.csv"], FOLD_SCORE_COLUMNS, what)
            if not rows:
                fail(f"{what}: no rows")
            for r in rows:
                if r["fold"] != str(fold) or r["train_seed"] != str(seed) or r["role"] != "test":
                    fail(f"{what}: a row carries another fold, seed or role")
                if r["label"] not in LABELS:
                    fail(f"{what}: label out of range")
                s = finite_score(r["score_attack"], what)
                if r["threshold_dev"] != repr(float(value)) or float(r["threshold_dev"]) != float(value):
                    fail(f"{what}: threshold_dev differs from the frozen threshold of {name}")
                if r["threshold_sha256"] != thr_sha:
                    fail(f"{what}: threshold_sha256 differs from the SHA-256 of {name}_threshold.json")
                if r["checkpoint_sha256"] != ckpt:
                    fail(f"{what}: checkpoint_sha256 differs from the restricted SHA256SUMS")
                if r["prediction_05"] != ("1" if s > FIXED_THRESHOLD else "0") or r["prediction_dev"] != ("1" if s > float(value) else "0"):
                    fail(f"{what}: a prediction does not follow '{DECISION_RULE}'")
                prev = inp.fold_of_subject[seed].setdefault(r["subject_id"], fold)
                if prev != fold:
                    fail(f"folds run seed {seed}: a subject is in the test of two folds")
                inp.folds_all[seed].append({**r, "score": s})
                if r["sample_id"] in inp.by_key:
                    seen[r["sample_id"]] += 1
                    if not same_as_manifest(r, inp.by_key[r["sample_id"]]):
                        fail(f"{what}: a cnn_test row differs from the manifest in label, subject, session or file SHA-256")
                    inp.folds[seed][r["sample_id"]] = {**r, "score": s, "threshold": float(value), "fold": fold}
        missing = len(set(inp.by_key) - set(seen))
        twice = sum(1 for v in seen.values() if v > 1)
        if missing or twice or len(inp.folds[seed]) != len(inp.by_key):
            fail(f"folds run seed {seed}: {len(seen)} cnn_test keys found, {missing} missing, {twice} in more than one row "
                 f"(each must be in the test of exactly one fold)")
    parts = {seed: {f: sorted(s for s, x in m.items() if x == f) for f in range(1, k + 1)} for seed, m in inp.fold_of_subject.items()}
    if any(parts[s] != parts[cfg["train_seeds"][0]] for s in cfg["train_seeds"]):
        fail("folds run: the held-out subjects of a fold differ between seeds")
    inp.checks["folds_run"] = {"run_id": fr["run_id"], "status_sha256": fr["status_sha256"], "public_sums_sha256": fr["public_sums_sha256"],
                               "restricted_sums_sha256": fr["restricted_sums_sha256"], "score_files": k * len(cfg["train_seeds"]),
                               "rows_per_seed": {str(s): len(inp.folds_all[s]) for s in cfg["train_seeds"]},
                               "cnn_test_rows_per_seed": {str(s): len(inp.folds[s]) for s in cfg["train_seeds"]},
                               "test_subjects_per_fold": {str(f): len(v) for f, v in parts[cfg["train_seeds"][0]].items()}}


def load_n1_run(cfg: dict, base: Path, inp: Inputs) -> None:
    """N1: SHA256SUMS ancorato; status.json, aggregate.json, metrics.json e scores_test.csv per SHA-256 registrato; le
    3.000 chiavi di cnn_test una volta ciascuna, uguali al manifest; decisioni coerenti con la soglia fissa; checkpoint e
    configurazione uguali a quelli registrati da N1."""
    n1 = cfg["n1_run"]
    folder = rel_path(base, n1["folder"])
    sb = read_regular(folder / "SHA256SUMS", "N1 SHA256SUMS")
    if sha256_bytes(sb) != n1["sha256sums_sha256"]:
        fail(f"N1 SHA256SUMS: SHA-256 {sha256_bytes(sb)} differs from the configuration ({n1['sha256sums_sha256']})")
    listed = read_sums(sb, "N1 SHA256SUMS")

    def listed_file(rel: str) -> bytes:
        if rel not in listed:
            fail(f"N1 SHA256SUMS: {rel} not listed")
        b = read_regular(folder / rel, f"N1 {rel}")
        if sha256_bytes(b) != listed[rel]:
            fail(f"N1 {rel}: SHA-256 {sha256_bytes(b)} differs from the N1 SHA256SUMS ({listed[rel]})")
        return b
    status = read_json(listed_file("status.json"), "N1 status.json")
    if status.get("result") != "PASS":
        fail("N1 status.json: not a PASS")
    agg = read_json(listed_file("aggregate.json"), "N1 aggregate.json")
    if agg.get("passed") is not True or agg.get("run_id") != n1["run_id"] or agg.get("threshold") != FIXED_THRESHOLD:
        fail(f"N1 aggregate.json: not a passed aggregate of run {n1['run_id']} at threshold {FIXED_THRESHOLD}")
    inp.n1_aggregate = agg
    for seed in cfg["train_seeds"]:
        met = read_json(listed_file(f"seed{seed}/metrics.json"), f"N1 seed{seed}/metrics.json")
        if met.get("run_id") != n1["run_id"] or met.get("train_seed") != seed:
            fail(f"N1 seed{seed}/metrics.json: run_id or train_seed differ")
        inp.n1_metrics[seed] = met
        b = listed_file(f"seed{seed}/scores_test.csv")
        sa = (agg.get("seeds") or {}).get(str(seed)) or {}
        if sa.get("scores_sha256") != sha256_bytes(b):
            fail(f"N1 seed{seed}/scores_test.csv: SHA-256 differs from scores_sha256 of the N1 aggregate")
        weights = met.get("weights_sha256")
        if not _hex(weights) or (sa.get("training") or {}).get("weights_sha256") != weights:
            fail(f"N1 seed{seed}: weights_sha256 of metrics.json and of the aggregate differ")
        what = f"N1 seed{seed}/scores_test.csv"
        rows = parse_csv(b, N1_SCORE_COLUMNS, what)
        out = {}
        for r in rows:
            if r["run_id"] != n1["run_id"] or r["train_seed"] != str(seed):
                fail(f"{what}: a row carries another run_id or train_seed")
            if r["checkpoint_sha256"] != weights or r["config_sha256"] != agg.get("config_sha256"):
                fail(f"{what}: checkpoint_sha256 or config_sha256 differ from the ones recorded by N1")
            if r["label"] not in LABELS:
                fail(f"{what}: label out of range")
            s = finite_score(r["score_attack"], what)
            if r["prediction"] != ("1" if s > FIXED_THRESHOLD else "0"):
                fail(f"{what}: a prediction does not follow '{DECISION_RULE}' at {FIXED_THRESHOLD}")
            o = inp.by_key.get(r["sample_id"])
            if o is None:
                fail(f"{what}: a key is not in cnn_test")
            if r["sample_id"] in out:
                fail(f"{what}: a key appears twice")
            if not same_as_manifest(r, o):
                fail(f"{what}: a row differs from the manifest in label, subject, session or file SHA-256")
            out[r["sample_id"]] = {**r, "score": s}
        if set(out) != set(inp.by_key):
            fail(f"{what}: {len(out)} keys, {len(set(inp.by_key) - set(out))} cnn_test keys missing")
        inp.n1[seed] = out
    inp.checks["n1_run"] = {"run_id": n1["run_id"], "sha256sums_sha256": n1["sha256sums_sha256"],
                            "config_sha256": agg.get("config_sha256"), "rows_per_seed": {str(s): len(inp.n1[s]) for s in cfg["train_seeds"]}}


def load_inputs(cfg: dict, base: Path) -> Inputs:
    inp = Inputs()
    load_manifests(cfg, base, inp)
    load_folds_run(cfg, base, inp)
    load_n1_run(cfg, base, inp)
    check_anchors(cfg, inp)
    return inp


# ============================================================================ metriche

def auroc(att, bf):
    """AUROC di score_attack come separatore (attacco positivo), Mann-Whitney con i pareggi a 0,5 (rango medio dei
    valori uguali); None senza una classe."""
    att, bf = np.asarray(att, float), np.asarray(bf, float)
    if len(att) == 0 or len(bf) == 0:
        return None
    allv = np.concatenate([att, bf])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0          # rango medio (1-based) di ogni valore distinto
    ranks = avg[inv]
    u = ranks[:len(att)].sum() - len(att) * (len(att) + 1) / 2.0
    return float(u / (len(att) * len(bf)))


def rates(y: np.ndarray, pred: np.ndarray) -> dict:
    n_a, n_b = int((y == 1).sum()), int((y == 0).sum())
    accepted, rejected = int(((y == 1) & (pred == 0)).sum()), int(((y == 0) & (pred == 1)).sum())
    apcer = accepted / n_a if n_a else None
    bpcer = rejected / n_b if n_b else None
    acer = (apcer + bpcer) / 2 if apcer is not None and bpcer is not None else None
    return {"accepted_attacks": accepted, "rejected_bona_fide": rejected, "apcer": apcer, "bpcer": bpcer, "acer": acer}


def metrics(y: np.ndarray, s: np.ndarray, preds: dict) -> dict:
    """Conteggi e tassi di un insieme di frame: per ogni decisione (nome -> predizioni) accettati, rifiutati, APCER,
    BPCER, ACER; AUROC dei punteggi."""
    out = {"n": int(len(y)), "n_bona_fide": int((y == 0).sum()), "n_attack": int((y == 1).sum())}
    for name, pred in preds.items():
        for k, v in rates(y, pred).items():
            out[f"{k}_{name}"] = v
    out["auroc"] = auroc(s[y == 1], s[y == 0])
    return out


def paired_arrays(inp: Inputs, seed: int) -> dict:
    """Le due righe di ogni frame di cnn_test, in ordine di chiave, come array."""
    keys = sorted(inp.by_key)
    fr, n1 = inp.folds[seed], inp.n1[seed]
    return {"keys": keys,
            "y": np.array([int(inp.by_key[k]["label"]) for k in keys], int),
            "subject": np.array([inp.by_key[k]["subject"] for k in keys]),
            "session": [inp.by_key[k]["session"] for k in keys],
            "n1_s": np.array([n1[k]["score"] for k in keys], float),
            "n1_p": np.array([int(n1[k]["prediction"]) for k in keys], int),
            "f_s": np.array([fr[k]["score"] for k in keys], float),
            "f_p05": np.array([int(fr[k]["prediction_05"]) for k in keys], int),
            "f_pdev": np.array([int(fr[k]["prediction_dev"]) for k in keys], int),
            "fold": np.array([fr[k]["fold"] for k in keys], int),
            "thr": np.array([fr[k]["threshold"] for k in keys], float)}


def pair_metrics(a: dict, ix=None) -> dict:
    """Metriche dei due esperimenti e differenze (fold − N1) su un sottoinsieme di frame (ix, tutti se None)."""
    sel = (lambda v: v) if ix is None else (lambda v: v[ix])
    y = sel(a["y"])
    f = metrics(y, sel(a["f_s"]), {"05": sel(a["f_p05"]), "dev": sel(a["f_pdev"])})
    n = metrics(y, sel(a["n1_s"]), {"05": sel(a["n1_p"])})
    d = lambda x, z: None if x is None or z is None else x - z   # noqa: E731
    flat = {**{f"folds_{k}": v for k, v in f.items()}, **{f"n1_{k}": v for k, v in n.items()}}
    flat["diff_acer_05"] = d(f["acer_05"], n["acer_05"])
    flat["diff_acer_dev"] = d(f["acer_dev"], n["acer_05"])
    flat["diff_auroc"] = d(f["auroc"], n["auroc"])
    return {"folds": f, "n1": n, "flat": flat}


def bootstrap(a: dict) -> dict:
    """Bootstrap appaiato sui soggetti (ricetta in BOOT["recipe"]): generatore default_rng(BOOT["seed"]) creato a ogni
    chiamata, quindi ripartito da 0 per ogni seed di training; soggetti in ordine crescente; BOOT["resamples"]
    ricampionamenti integers(0, n, n) con reinserimento; metriche sui frame uniti dei soggetti estratti; gli stessi
    indici per i due esperimenti e per ogni metrica, così la differenza è appaiata per soggetto; percentili
    BOOT["percentiles"] con np.percentile lineare; nessun intervallo con meno di BOOT["min_subjects"] soggetti; un
    ricampionamento senza una classe non conta per le metriche che la richiedono."""
    subjects = sorted(set(a["subject"].tolist()))
    out = {"n_subjects": len(subjects), "resamples": BOOT["resamples"], "seed": BOOT["seed"], "percentiles": BOOT["percentiles"]}
    if len(subjects) < BOOT["min_subjects"]:
        out["note"] = f"no interval: fewer than {BOOT['min_subjects']} subjects"
        for k in BOOT_KEYS:
            out[k] = {"low": None, "high": None, "valid_resamples": 0}
        return out
    index = [np.flatnonzero(a["subject"] == s) for s in subjects]
    rng = np.random.default_rng(BOOT["seed"])
    vals = {k: [] for k in BOOT_KEYS}
    for _ in range(BOOT["resamples"]):
        pick = rng.integers(0, len(subjects), len(subjects))
        ix = np.concatenate([index[i] for i in pick])
        m = pair_metrics(a, ix)["flat"]
        for k in BOOT_KEYS:
            if m[k] is not None:
                vals[k].append(m[k])
    for k in BOOT_KEYS:
        if vals[k]:
            lo, hi = np.percentile(vals[k], BOOT["percentiles"], method="linear")
            out[k] = {"low": float(lo), "high": float(hi), "valid_resamples": len(vals[k])}
        else:
            out[k] = {"low": None, "high": None, "valid_resamples": 0}
    return out


def decision_table(a: dict) -> dict:
    """Tabella 2×2 per frame (corretto/sbagliato in N1 × nel run dei fold), per classe, a 0,5 e alla soglia dev del fold
    (N1 sempre a 0,5); decisions_changed = frame su cui i due esperimenti decidono diversamente."""
    y = a["y"]
    n1_ok = a["n1_p"] == y
    out = {}
    for name, fp in (("05", a["f_p05"]), ("dev", a["f_pdev"])):
        f_ok = fp == y
        t = {}
        for lab, cls in ((0, "bona_fide"), (1, "attack")):
            m = y == lab
            t[cls] = {"both_correct": int((m & n1_ok & f_ok).sum()), "n1_only_correct": int((m & n1_ok & ~f_ok).sum()),
                      "folds_only_correct": int((m & ~n1_ok & f_ok).sum()), "both_wrong": int((m & ~n1_ok & ~f_ok).sum())}
        t["decisions_changed"] = int((a["n1_p"] != fp).sum())
        t["n"] = int(len(y))
        out[name] = t
    return out


def per_subject(seed: int, a: dict) -> tuple:
    """(righe riservate per soggetto, conteggi pubblici di soggetti). Errori per soggetto: frame sbagliati (sempre
    definiti); ACER solo con entrambe le classi."""
    rows, cmp = [], {}
    tallies = {(name, meas): Counter() for name in ("05", "dev") for meas in ("error_rate", "acer")}
    for subj in sorted(set(a["subject"].tolist())):
        ix = np.flatnonzero(a["subject"] == subj)
        y = a["y"][ix]
        folds = sorted(set(a["fold"][ix].tolist()))
        n1 = rates(y, a["n1_p"][ix])
        f05, fdev = rates(y, a["f_p05"][ix]), rates(y, a["f_pdev"][ix])
        err = lambda r: r["accepted_attacks"] + r["rejected_bona_fide"]   # noqa: E731
        n = len(ix)
        rows.append([seed, subj, "+".join(map(str, folds)), int((y == 0).sum()), int((y == 1).sum()), err(n1), fmt(err(n1) / n), fmt(n1["acer"]),
                     err(f05), fmt(err(f05) / n), fmt(f05["acer"]), err(fdev), fmt(err(fdev) / n), fmt(fdev["acer"]),
                     fmt(auroc(a["n1_s"][ix][y == 1], a["n1_s"][ix][y == 0])), fmt(auroc(a["f_s"][ix][y == 1], a["f_s"][ix][y == 0]))])
        for name, fr in (("05", f05), ("dev", fdev)):
            for meas, fv, nv in (("error_rate", err(fr) / n, err(n1) / n), ("acer", fr["acer"], n1["acer"])):
                c = tallies[(name, meas)]
                if fv is None or nv is None:
                    c["not_defined"] += 1
                else:
                    c["folds_lower"] += fv < nv
                    c["equal"] += fv == nv
                    c["folds_higher"] += fv > nv
    for (name, meas), c in tallies.items():
        cmp.setdefault(name, {})[meas] = {k: int(c.get(k, 0)) for k in ("folds_lower", "equal", "folds_higher", "not_defined")}
    return rows, {"n_subjects": len(rows), "with_both_classes": int(sum(1 for r in rows if r[3] and r[4])), "comparison": cmp}


def summary(values: list) -> dict:
    if any(v is None for v in values):
        return {"values": values, "median": None, "min": None, "max": None}
    return {"values": values, "median": float(statistics.median(values)), "min": float(min(values)), "max": float(max(values))}


def seed_metrics(cfg: dict, inp: Inputs) -> dict:
    return {seed: pair_metrics(paired_arrays(inp, seed)) for seed in cfg["train_seeds"]}


def control_agrees(g, w, kind: str) -> bool:
    """Accordo con un valore di controllo alle cifre date dal coordinatore: ACER come percentuale a 2 decimali
    (round(100 * g, 2)), AUROC a 5 decimali."""
    if g is None:
        return False
    if kind == "auroc":
        return round(g, CONTROL_DIGITS["auroc"]) == round(w, CONTROL_DIGITS["auroc"])
    return round(100 * g, CONTROL_DIGITS["acer"]) == round(w, CONTROL_DIGITS["acer"])


def check_anchors(cfg: dict, inp: Inputs) -> None:
    """Ancore della lettura: le metriche pubblicate dai due run ricalcolate dai loro punteggi (run dei fold: tutti i frame
    tenuti fuori dei quattro fold, per seed, contro pooled_per_seed.all di aggregate.json; N1: i 3.000 frame contro
    metrics.json e aggregate.json), e i valori di controllo del coordinatore riprodotti sui 3.000 frame."""
    problems = []

    def compare(what, got: dict, want: dict, pairs):
        for gk, wk in pairs:
            g, w = got.get(gk), (want or {}).get(wk)
            if isinstance(g, int) and not isinstance(g, bool):
                ok = g == w
            else:
                ok = g is not None and _number(w) and abs(g - w) <= ANCHOR_TOL
            if not ok:
                problems.append(f"{what} {wk}: recomputed {g}, published {w}")
    pooled = inp.folds_aggregate.get("pooled_per_seed") or {}
    for seed in cfg["train_seeds"]:
        rows = inp.folds_all[seed]
        y = np.array([int(r["label"]) for r in rows], int)
        s = np.array([r["score"] for r in rows], float)
        m = metrics(y, s, {"05": np.array([int(r["prediction_05"]) for r in rows]), "dev": np.array([int(r["prediction_dev"]) for r in rows])})
        compare(f"folds run seed {seed} (all held-out frames)", m, (pooled.get(str(seed)) or {}).get("all"),
                [(k, k) for k in ("n", "n_bona_fide", "n_attack", "accepted_attacks_05", "rejected_bona_fide_05", "acer_05",
                                  "accepted_attacks_dev", "rejected_bona_fide_dev", "acer_dev", "auroc")])
        n1 = inp.n1[seed]
        keys = sorted(n1)
        y = np.array([int(n1[k]["label"]) for k in keys], int)
        m = metrics(y, np.array([n1[k]["score"] for k in keys]), {"05": np.array([int(n1[k]["prediction"]) for k in keys])})
        pairs = [("n", "n"), ("n_bona_fide", "n_bona_fide"), ("n_attack", "n_attack"), ("accepted_attacks_05", "accepted_attacks"),
                 ("rejected_bona_fide_05", "rejected_bona_fide"), ("apcer_05", "apcer"), ("bpcer_05", "bpcer"), ("acer_05", "acer"),
                 ("auroc", "auroc")]
        compare(f"N1 seed {seed} metrics.json", m, ((inp.n1_metrics[seed].get("metrics") or {}).get("test_3000")), pairs)
        sa = ((inp.n1_aggregate.get("seeds") or {}).get(str(seed)) or {})
        compare(f"N1 seed {seed} aggregate.json", m, (sa.get("metrics") or {}).get("test_3000"), pairs)
    if problems:
        fail("anchors not reproduced: " + "; ".join(problems[:6]))
    per = seed_metrics(cfg, inp)
    cv = cfg["control_values"]
    got = {"acer_05": [per[s]["folds"]["acer_05"] for s in cfg["train_seeds"]],
           "acer_dev": [per[s]["folds"]["acer_dev"] for s in cfg["train_seeds"]],
           "auroc": [per[s]["folds"]["auroc"] for s in cfg["train_seeds"]]}
    for k, vals in got.items():
        for seed, g, w in zip(cfg["train_seeds"], vals, cv[CONTROL_KEYS[k]]):
            if not control_agrees(g, w, k):
                problems.append(f"{k} seed {seed}: computed {g}, control value {w} ({CONTROL_KEYS[k]})")
    if problems:
        fail("control values of the coordinator not reproduced: " + "; ".join(problems))
    inp.checks["anchors"] = {"folds_run": "pooled_per_seed.all of aggregate.json recomputed from the 12 test score files",
                             "n1": "test_3000 of metrics.json and aggregate.json recomputed from scores_test.csv",
                             "tolerance": ANCHOR_TOL}
    inp.checks["control_values"] = {"source": cv["source"], "control": {CONTROL_KEYS[k]: cv[CONTROL_KEYS[k]] for k in got},
                                    "computed": got, "comparison": CONTROL_COMPARISON, "reproduced": True}


# ============================================================================ aggregato

def aggregate(cfg: dict, inp: Inputs) -> tuple:
    """(documento pubblico, righe appaiate riservate, righe per soggetto riservate)."""
    per_seed, boot, subj_rows, paired = {}, {}, [], []
    flat = {}
    for seed in cfg["train_seeds"]:
        a = paired_arrays(inp, seed)
        pm = pair_metrics(a)
        flat[seed] = pm["flat"]
        rows, counts = per_subject(seed, a)
        subj_rows += rows
        per_seed[str(seed)] = {"folds": pm["folds"], "n1": pm["n1"],
                               "difference_folds_minus_n1": {k: pm["flat"][k] for k in ("diff_acer_05", "diff_acer_dev", "diff_auroc")},
                               "paired_decisions": decision_table(a), "per_subject_counts": counts,
                               "test_subjects_per_fold": {str(f): int(len(set(a["subject"][a["fold"] == f].tolist())))
                                                          for f in range(1, cfg["folds_run"]["k"] + 1)}}
        boot[str(seed)] = bootstrap(a)
        for i, k in enumerate(a["keys"]):
            paired.append([seed, k, str(a["subject"][i]), a["session"][i], int(a["y"][i]), fmt(a["n1_s"][i]), int(a["n1_p"][i]),
                           int(a["fold"][i]), fmt(a["f_s"][i]), fmt(a["thr"][i]), int(a["f_p05"][i]), int(a["f_pdev"][i])])
    over = {k: summary([flat[s][k] for s in cfg["train_seeds"]]) for k in SUMMARY_KEYS}
    doc = {"per_seed": per_seed, "summary_over_seeds": over, "bootstrap_per_seed": boot,
           "control_values": inp.checks["control_values"]}
    return doc, paired, subj_rows


def table(headers, rows) -> str:
    return "| " + " | ".join(headers) + " |\n|" + "---|" * len(headers) + "\n" + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)


def para(text: str) -> str:
    return textwrap.fill(text, MD_WIDTH, break_long_words=False, break_on_hyphens=False)


def ci(b: dict, k: str) -> str:
    return f"[{f4(b[k]['low'])}, {f4(b[k]['high'])}]" if b[k]["low"] is not None else "no interval"


def n1_auroc_note(agg: dict) -> str:
    """Frase fattuale, senza interpretazione, quando l'AUROC di N1 è sotto 0,5 per ogni seed."""
    v = agg["summary_over_seeds"]["n1_auroc"]
    if v["max"] is None or v["max"] >= 0.5:
        return ""
    return (f"N1 AUROC is below 0.5 on these frames ({f4(v['min'])}-{f4(v['max'])}, as published by N1); the AUROC "
            "difference includes that.")


def aggregate_md(cfg: dict, agg: dict, head: dict) -> str:
    seeds = cfg["train_seeds"]
    fr, n1, dec, m = cfg["folds_run"], cfg["n1_run"], cfg["declarations"], cfg["manifest"]
    L = ["# NUAA: folds run versus N1 on the same 3,000 frames (descriptive, no identifiers)", "",
         para(f"Run `{head['run_id']}` (commit `{head['commit']}`, dirty tree: {'yes' if head['dirty'] else 'no'}), "
              f"{head['finished']}; configuration `{head['config_sha256'][:12]}`. Requested by the coordinator on "
              f"{cfg['request']['date']} ({cfg['request']['card']} card). Folds run `{fr['run_id']}` (status "
              f"`{fr['status_sha256'][:12]}`), N1 `{n1['run_id']}` (SHA256SUMS `{n1['sha256sums_sha256'][:12]}`); frames: "
              f"the {m['rows']:,} rows of the manifest `{m['file']}` (`{m['sha256'][:12]}`), {m['n_bona_fide']:,} bona fide and "
              f"{m['n_attack']:,} attacks of {m['n_subjects']} subjects, session {m['session']}."), "",
         para("Descriptive and post hoc: computed after both runs from their existing scores and their already frozen "
              "thresholds; no training, no inference, no image read, no new threshold. Indicative; no causal reading. "
              + dec["training_roles"]), "",
         para(f"Decision: {DECISION_RULE}. `05` = the fixed threshold 0.5; `dev` = the EER threshold of the fold whose model "
              "scored the frame, frozen on its dev fold before the test (" + dec["n1_threshold"] + "). AUROC: Mann-Whitney, "
              "ties 0.5. " + dec["auroc_four_models"]), "",
         "## Per seed", ""]
    rows = []
    for s in seeds:
        p, b = agg["per_seed"][str(s)], agg["bootstrap_per_seed"][str(s)]
        f, n = p["folds"], p["n1"]
        rows.append([s, "folds, 0.5", f"{f['rejected_bona_fide_05']} / {f['accepted_attacks_05']}", f4(f["bpcer_05"]), f4(f["apcer_05"]),
                     f"{f4(f['acer_05'])} {ci(b, 'folds_acer_05')}", f"{f4(f['auroc'])} {ci(b, 'folds_auroc')}"])
        rows.append([s, "folds, dev", f"{f['rejected_bona_fide_dev']} / {f['accepted_attacks_dev']}", f4(f["bpcer_dev"]), f4(f["apcer_dev"]),
                     f"{f4(f['acer_dev'])} {ci(b, 'folds_acer_dev')}", "as above"])
        rows.append([s, "N1, 0.5", f"{n['rejected_bona_fide_05']} / {n['accepted_attacks_05']}", f4(n["bpcer_05"]), f4(n["apcer_05"]),
                     f"{f4(n['acer_05'])} {ci(b, 'n1_acer_05')}", f"{f4(n['auroc'])} {ci(b, 'n1_auroc')}"])
    L.append(table(["seed", "experiment", "rej. bf / acc. att.", "BPCER", "APCER", "ACER [bootstrap]", "AUROC [bootstrap]"], rows))
    f0 = agg["per_seed"][str(seeds[0])]["folds"]
    L += [para(f"Every seed: {f0['n_bona_fide']:,} bona fide and {f0['n_attack']:,} attacks. Bootstrap: "
               f"{BOOT['resamples']:,} resamples of the {agg['bootstrap_per_seed'][str(seeds[0])]['n_subjects']} subjects with "
               f"replacement. Recipe: {BOOT['recipe']}."), "",
          "## Differences, folds minus N1 (paired bootstrap over subjects)", ""]
    rows = []
    for s in seeds:
        d, b = agg["per_seed"][str(s)]["difference_folds_minus_n1"], agg["bootstrap_per_seed"][str(s)]
        rows.append([s, f"{f4(d['diff_acer_05'])} {ci(b, 'diff_acer_05')}", f"{f4(d['diff_acer_dev'])} {ci(b, 'diff_acer_dev')}",
                     f"{f4(d['diff_auroc'])} {ci(b, 'diff_auroc')}"])
    L.append(table(["seed", "ACER folds 0.5 - N1 0.5", "ACER folds dev - N1 0.5", "AUROC folds - N1"], rows))
    L += ["## Median and range over the three seeds", ""]
    names = {"folds_acer_05": "folds ACER 0.5", "folds_acer_dev": "folds ACER dev", "folds_auroc": "folds AUROC",
             "n1_acer_05": "N1 ACER 0.5", "n1_auroc": "N1 AUROC", "diff_acer_05": "difference ACER 0.5",
             "diff_acer_dev": "difference ACER dev vs N1 0.5", "diff_auroc": "difference AUROC"}
    L.append(table(["measure", "median", "min", "max"],
                   [[v, f4(agg["summary_over_seeds"][k]["median"]), f4(agg["summary_over_seeds"][k]["min"]),
                     f4(agg["summary_over_seeds"][k]["max"])] for k, v in names.items()]))
    L += ["## Paired decisions per frame", "",
          para("Per class: frames correct in both experiments, only in N1, only in the folds run, in neither; N1 always at "
               "0.5, the folds run at 0.5 and at its dev thresholds. Changed decisions = frames of either class on which the "
               "two decisions differ, one count per seed and threshold."), ""]
    rows = []
    cell = lambda c: f"{c['both_correct']} / {c['n1_only_correct']} / {c['folds_only_correct']} / {c['both_wrong']}"   # noqa: E731
    for s in seeds:
        for t in ("05", "dev"):
            pd = agg["per_seed"][str(s)]["paired_decisions"][t]
            rows.append([s, "0.5" if t == "05" else "dev", cell(pd["bona_fide"]), cell(pd["attack"]), pd["decisions_changed"]])
    L.append(table(["seed", "folds at", "bona fide: both ok / N1 only ok / folds only ok / both wrong", "attack: same cells",
                    "changed decisions (both classes)"], rows))
    L += ["## Subjects (counts only)", "",
          para("Per subject, error rate = wrong frames / frames (defined for every subject); ACER only for subjects with "
               "both classes. Counts of subjects where the folds run (at 0.5 or at its dev thresholds) is lower, equal or higher "
               "than N1 at 0.5."), ""]
    rows = []
    for s in seeds:
        pc = agg["per_seed"][str(s)]["per_subject_counts"]
        for t in ("05", "dev"):
            for meas in ("error_rate", "acer"):
                c = pc["comparison"][t][meas]
                rows.append([s, "0.5" if t == "05" else "dev", meas.replace("_", " "), c["folds_lower"], c["equal"], c["folds_higher"],
                             c["not_defined"]])
    L.append(table(["seed", "folds at", "measure", "folds lower", "equal", "folds higher", "not defined"], rows))
    pc0 = agg["per_seed"][str(seeds[0])]["per_subject_counts"]
    L += [para(f"{pc0['n_subjects']} subjects, {pc0['with_both_classes']} with both classes."), "",
          "## Control values of the coordinator", ""]
    cv = agg["control_values"]
    given = lambda k, v: f"{v:.5f}" if k == "auroc" else f"{v:.2f} %"   # noqa: E731
    rows = [[k, s, given(k, cv["control"][CONTROL_KEYS[k]][i]), f5(cv["computed"][k][i])] for k in ("acer_05", "acer_dev", "auroc")
            for i, s in enumerate(seeds)]
    L.append(table(["measure", "seed", "control (as given)", "computed"], rows))
    L += [para(f"Source: {cv['source']}. Comparison: {cv['comparison']}."), "",
          "## Limits", "", para(" ".join(x for x in (dec["indicative"], n1_auroc_note(agg), dec["no_causality"], dec["limits"]) if x)), ""]
    return "\n".join(L)


def public_problems(text: str, inp: Inputs, markdown: bool = False) -> list:
    """Contenuto vietato in un file pubblico: chiavi o SHA-256 delle immagini, identificativi di soggetto (fuori da
    numeri e hash), regole di package_rules, IPv4 non locali, segni di lavoro non finito (solo nei .md)."""
    bad = []
    if re.search(r"\b(real|attack)/\d{4}_", text):
        bad.append("image keys")
    if set(re.findall(r"[0-9a-f]{64}", text)) & inp.image_sha:
        bad.append("image SHA-256 values")
    for s in sorted(inp.subjects | inp.n1_training_subjects):
        if re.search(rf"(?<![\w.]){re.escape(s)}(?![\w.])", text):
            bad.append(f"subject id {s}")
            break
    for label, rx in pr.FORBIDDEN_TEXT:
        if m := rx.search(text):
            bad.append(f"{label}: {m.group(0)[:40]!r}")
    for ip in (m.group(0) for m in pr.IPV4.finditer(text)):
        if ip not in pr.IP_ALLOW:
            bad.append(f"IP address: {ip}")
            break
    if markdown and (m := pr.MARKERS.search(text)):
        bad.append(f"unfinished-work marker: {m.group(0)!r}")
    return bad


# ============================================================================ run

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
    rid = f"{date}-FOLDSCOMPARE-{state['commit']}" + ("-dev" if state["dirty"] else "")
    if not RUN_ID_RE.match(rid):
        fail(f"run id {rid} does not follow YYYYMMDD-FOLDSCOMPARE-<commit>[-dev]")
    return rid


def protected_extra(root: Path) -> list:
    return [root / x for x in EXTRA_PROTECTED]


def input_paths(cfg: dict, base: Path, cfg_p: Path) -> dict:
    return {"the configuration": cfg_p, "the schema configuration": rel_path(base, cfg["schema_config"]["path"]),
            "the manifests folder": rel_path(base, cfg["manifest"]["folder"]), "the folds run": rel_path(base, cfg["folds_run"]["folder"]),
            "the N1 run": rel_path(base, cfg["n1_run"]["folder"])}


def run_dir_problem(run: Path, root: Path, inputs: dict, out_root: Path = None) -> str:
    """Motivo per cui la cartella del run non è ammessa ("" se lo è), prima di qualunque modifica."""
    if (out_root if out_root is not None else run.parent).is_symlink():
        return "the output root is a symbolic link"
    for label, p in inputs.items():
        if pg.within(p, run) or (Path(p).is_dir() and pg.within(run, p)):
            return f"{label} {Path(p).name} is inside the run folder, or contains it: inputs and outputs must be separate"
    msg = mw.output_problem(run, root, pg.BASE_PROTECTED, [root / DEFAULT_OUT_ROOT], protected_extra(root), inputs, [STATUS, LOG],
                            [run / PUBLIC, run / RESTRICTED])
    if msg:
        return msg
    for sub, names in ((PUBLIC, PUBLIC_OUTPUTS), (RESTRICTED, RESTRICTED_OUTPUTS)):
        if msg := mw.output_problem(run / sub, root, pg.BASE_PROTECTED, [root / DEFAULT_OUT_ROOT], protected_extra(root), inputs, list(names)):
            return msg
    allowed = {"": [STATUS, LOG, PUBLIC, RESTRICTED], PUBLIC: PUBLIC_OUTPUTS, RESTRICTED: RESTRICTED_OUTPUTS}
    for sub, names in allowed.items():
        d = run / sub if sub else run
        if d.is_dir():
            for e in sorted(os.listdir(d)):
                if not any(mw.owned_name(e, n) for n in names):
                    return f"{(Path(sub) / e).as_posix()} in the run folder is not an output of this command: refused, nothing removed"
    return ""


def write_status(run: Path, doc: dict) -> None:
    mw.write_all(run, {STATUS: json.dumps(doc, indent=1) + "\n"})


def invalidate_outputs(run: Path) -> list:
    left = []
    for sub, names in ((PUBLIC, PUBLIC_OUTPUTS), (RESTRICTED, RESTRICTED_OUTPUTS)):
        d = run / sub
        if d.is_dir():
            left += [f"{sub}/{n}" for n in mw.invalidate_each(d, list(names), None, FAIL_DOC)]
    return left


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


def code_sha256() -> dict:
    return {f: sha256_file(CODE_ROOT / f) for f in CODE_FILES if (CODE_ROOT / f).is_file()}


def control_line(inp: Inputs) -> str:
    c = inp.checks["control_values"]["computed"]
    return "; ".join(f"{k} " + "/".join(f5(v) for v in vals) for k, vals in c.items())


def cmd_check_inputs(a) -> int:
    base, cfg_p = resolve(a)
    cfg, cfg_sha = load_config(cfg_p)
    inp = load_inputs(cfg, base)
    m, f = inp.checks["manifest"], inp.checks["folds_run"]
    per_seed = lambda d: "/".join(str(v) for v in d.values())   # noqa: E731
    print(f"CHECK-INPUTS: configuration {cfg_sha[:12]} (delivered: {cfg_sha == CONFIG_SHA256}), manifest cnn_test {m['rows']} rows ({m['n_bona_fide']}/{m['n_attack']}, "
          f"{m['n_subjects']} subjects, {m['n1_training_subjects_in_set']} of them N1 training subjects), folds run "
          f"{f['run_id']} ({f['score_files']} test score files, {per_seed(f['cnn_test_rows_per_seed'])} cnn_test rows per seed), N1 "
          f"{inp.checks['n1_run']['run_id']} ({per_seed(inp.checks['n1_run']['rows_per_seed'])} rows per seed), anchors reproduced; "
          f"control values reproduced: {control_line(inp)}; nothing written")
    print("RESULT: PASS (check-inputs)")
    return 0


def cmd_run(a) -> int:
    if not a.i_am_authorized:
        fail("run reads the restricted per-frame scores of the folds run and of N1: pass --i-am-authorized (comparison "
             "requested by the coordinator on 4 November 2026)")
    base, cfg_p = resolve(a)
    cfg, cfg_sha = load_config(cfg_p)
    state = repo_state(ROOT)
    rid = run_id_for(a.date or time.strftime("%Y%m%d"), state)
    out_root = Path(a.out_root) if Path(a.out_root).is_absolute() else ROOT / a.out_root
    run = out_root / rid
    inputs = input_paths(cfg, base, cfg_p)
    if (run / STATUS).exists() or (run / STATUS).is_symlink():
        fail(f"the run folder {rid} already holds {STATUS}: every attempt keeps its outcome (archive that folder first)")
    if msg := run_dir_problem(run, ROOT, inputs, out_root):
        fail(f"refusing the run folder: {msg}")
    run.mkdir(parents=True, exist_ok=True)
    head = {"run_id": rid, "commit": state["commit"], "dirty": state["dirty"], "config_sha256": cfg_sha,
            "delivered_configuration": cfg_sha == CONFIG_SHA256, "command": "run"}
    log = None
    try:
        write_status(run, {**head, "status": "RUNNING", "passed": False, "started": now(),
                           "note": "RUNNING without a live process means a failed run"})
        log = open_log(run)
        left = invalidate_outputs(run)
        if left:
            fail(f"previous outputs could not be removed: {', '.join(left)}")
        print(f"== NUAA folds run versus N1 {rid}: configuration {cfg_sha[:12]} (delivered: {head['delivered_configuration']})", flush=True)
        inp = load_inputs(cfg, base)
        print(f"   check-inputs PASS; control values reproduced: {control_line(inp)}", flush=True)
        agg, paired, subj_rows = aggregate(cfg, inp)
        finished = now()
        head_out = {**head, "finished": finished}
        agg_doc = {**head_out, "request": cfg["request"], "task": cfg["task"], "declarations": cfg["declarations"],
                   "decision": DECISION_RULE, "bootstrap": BOOT, "metrics": cfg["metrics"],
                   "inputs": {"manifest": inp.checks["manifest"], "folds_run": inp.checks["folds_run"], "n1_run": inp.checks["n1_run"],
                              "anchors": inp.checks["anchors"]},
                   "code_sha256": code_sha256(), **agg}
        public = {"aggregate.json": json.dumps(agg_doc, indent=1) + "\n", "aggregate.md": aggregate_md(cfg, agg, head_out)}
        for name, text in public.items():
            if probs := public_problems(text, inp, name.endswith(".md")):
                fail(f"public file {name} refused: " + "; ".join(probs))
        public["SHA256SUMS"] = sums_text(public)
        restricted = {"paired_frames.csv": csv_text(PAIRED_COLUMNS, paired), "per_subject.csv": csv_text(SUBJECT_COLUMNS, subj_rows),
                      "public_SHA256SUMS.sha256": sha256_bytes(public["SHA256SUMS"].encode("utf-8")) + "  public/SHA256SUMS\n"}
        restricted["SHA256SUMS"] = sums_text(restricted)
        mw.write_all(run / RESTRICTED, restricted, newline="")
        mw.write_all(run / PUBLIC, public)
        for k in ("folds_acer_05", "folds_acer_dev", "folds_auroc", "n1_acer_05", "n1_auroc", "diff_acer_05", "diff_acer_dev", "diff_auroc"):
            v = agg["summary_over_seeds"][k]
            print(f"{k}: median {f4(v['median'])} ({f4(v['min'])}-{f4(v['max'])}), seeds {'/'.join(f4(x) for x in v['values'])}")
        print(f"run folder: {run.name} (public/ and restricted/)")
        print("RESULT: PASS (run)")
        close_log(log)
        log = None
        write_status(run, {**head, "status": "PASS", "passed": True, "finished": finished,
                           "public_sums_sha256": sha256_bytes(public["SHA256SUMS"].encode("utf-8")),
                           "restricted_sums_sha256": sha256_bytes(restricted["SHA256SUMS"].encode("utf-8")),
                           "run_log_sha256": sha256_file(run / LOG)})
    except BaseException as e:
        reason = str(e) if isinstance(e, Failure) else f"unexpected error {type(e).__name__}: {e}"
        left = invalidate_outputs(run)
        if log is not None:
            log[0].write(f"ERROR: {reason}\nRESULT: FAIL (run)\n")
            close_log(log)
        try:
            write_status(run, {**head, "status": "FAIL", "passed": False, "reason": reason,
                               "outputs": "removed" if not left else f"left behind: {', '.join(left)}",
                               "run_log_sha256": sha256_file(run / LOG) if (run / LOG).is_file() else "none"})
        except Exception:  # noqa: BLE001
            print("ERROR: status.json could not be written", file=sys.stderr)
        raise
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="NUAA: descriptive comparison of the folds run and N1 on the 3,000 cnn_test frames")
    ap.add_argument("cmd", choices=["check-inputs", "run"])
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="configuration JSON (relative to the repository root)")
    ap.add_argument("--input-root", default=None, help="base folder of the input paths of the configuration (default: repository root)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="folder that receives <run-id>/ (default results/foldscompare)")
    ap.add_argument("--date", default=None, help="YYYYMMDD of the run id (default: today)")
    ap.add_argument("--i-am-authorized", action="store_true", help="required by run: reads the restricted per-frame scores")
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
    except SystemExit as e:                     # fail() di m2_writers o di nuaa_manifests: già stampato
        print(f"RESULT: FAIL ({a.cmd})")
        return e.code if isinstance(e.code, int) and e.code else 1
    except Exception as e:  # noqa: BLE001  (errore imprevisto: le uscite sono già state tolte da cmd_run)
        print(f"ERROR: unexpected error {type(e).__name__}: {e}", file=sys.stderr)
        print(f"RESULT: FAIL ({a.cmd})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
