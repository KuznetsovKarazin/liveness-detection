"""
Step zero NUAA (autorizzato dal coordinatore sulla scheda C2 il 3/11/2026): statistiche d'immagine descrittive dei
materiali NUAA già autorizzati, messe a confronto con gli score per immagine già consegnati. Nessun training, nessuna
inferenza, nessuna nuova rilevazione dei volti: ogni numero è una statistica delle immagini o un confronto fra quelle
statistiche e score già accettati (C1, N1, N2). Le associazioni trovate sono associazioni, non cause.

Ingressi (sola lettura, fissati nella configurazione per SHA-256; gli score e i riquadri anche contro il SHA256SUMS
della loro cartella, a sua volta fissato per SHA-256):
  - i manifest M2 delle liste ufficiali NUAA (official_train 3.491 e official_test 9.123; cnn_train, cnn_es, cnn_test,
    c1 v2 per i sottoinsiemi) e gli originali (localizzati per chiave nella disposizione <classe>_<split>/<nome> e
    accettati solo con SHA-256 e byte uguali al manifest);
  - la cache di training nuaa_1500.npz (6.000 frame uint8 256x256x3 RGB: X_train = cnn_train + cnn_es nell'ordine
    spread della lista ufficiale di training, X_test = cnn_test nell'ordine spread della lista ufficiale di test);
  - gli score per immagine dei run accettati: N1 (tre seed, 3.000), N2 sessioni (tre seed, 491), N2 combinato (due
    varianti, 3.000), C1 (quattro CNN NUAA, 300);
  - i riquadri RetinaFace del confronto del ritaglio (300 immagini, due estrattori).

Statistiche per immagine (vista raw: file originale letto con cv2.imread, BGR; vista crop: frame della cache, RGB):
  lum_mean, lum_std   media e deviazione standard (popolazione) del grigio OpenCV (0,299 R + 0,587 G + 0,114 B)
  sat_mean            media del canale S di HSV (OpenCV, 0-255)
  lap_var             varianza (popolazione) del Laplaciano 3x3 del grigio (cv2.Laplacian, CV_64F, ksize 1)
  width, height, bytes (solo raw: dimensioni dell'immagine e byte del file dal manifest)
Associazioni (tutte descrittive): per insieme x sessione x classe n, mediana e quartili; AUROC di ogni statistica come
separatore delle classi (1 = attacco, statistica più alta -> attacco: un valore sotto 0,5 significa bona fide con
statistica più alta); correlazione di Spearman fra ogni statistica e score_attack per file di score x sessione x classe;
AUROC di ogni statistica fra i bona fide della sessione 03 rifiutati e accettati (N1 e C1); area relativa del riquadro
per estrattore. Per ogni insieme x sessione anche l'AUROC di classe ENTRO soggetto (calcolato dentro ogni soggetto con
entrambe le classi; nel pubblico mediana, minimo e massimo fra quei soggetti con il loro numero; valori per soggetto
nel riservato) e un intervallo bootstrap sui soggetti dell'AUROC aggregato (BOOT: 1.000 ricampionamenti con
reinserimento dei soggetti, seme 0, percentili 2,5 e 97,5; stessi indici per ogni statistica; nella riga «all» le
sessioni di un soggetto sono messe insieme). AUROC = statistica U di Mann-Whitney con i pareggi a 0,5; Spearman =
Pearson dei ranghi medi.

Uscite (cartella del run <out-root>/YYYYMMDD-STEPZERO-<commit>[-dev], -dev se l'albero git non è pulito):
  status.json                      RUNNING all'avvio, poi PASS o FAIL (unico stato corrente)
  public/                          aggregati senza chiavi, hash delle immagini o identificativi di soggetto
    aggregate.json, aggregate.md, figure_distributions.png (solo con matplotlib; altrimenti salto dichiarato), SHA256SUMS
  restricted/                      righe per immagine e per soggetto: area riservata, mai nella PR
    per_image_raw.csv, per_image_crop.csv, per_subject.csv (mediane per soggetto), per_subject_auroc.csv (AUROC entro
    soggetto), public_SHA256SUMS.sha256 (ancora del pubblico), SHA256SUMS
Dentro il repository la cartella del run è ammessa solo sotto results/stepzero/; fuori (per esempio scratch) dove
path_guard lo consente. Cartella di uscita e ingressi validati per identità prima di qualunque modifica
(m2_writers.output_problem); scritture atomiche (m2_writers.write_all); le uscite precedenti sono tolte subito dopo
l'ammissione (nessun PASS vecchio accanto a un'esecuzione che fallisce) e su qualunque errore restano solo
status.json = FAIL e nessuna uscita. I testi pubblici sono scansionati (contenuto vietato, chiavi, hash, soggetti)
prima della scrittura.

  .venv/bin/python scripts/nuaa_step_zero.py check-inputs                 # solo verifica degli ingressi, nessuna scrittura
  .venv/bin/python scripts/nuaa_step_zero.py run --i-am-authorized        # statistiche e uscite in results/stepzero/<run-id>/

Dipendenze: numpy e OpenCV (statistiche), matplotlib facoltativo (figura); scripts/m2_writers.py, scripts/path_guard.py,
scripts/package_rules.py. Nessun TensorFlow.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import numpy.lib.format as npy_format

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import m2_writers as mw  # noqa: E402
import path_guard as pg  # noqa: E402
from package_rules import FORBIDDEN_TEXT, IP_ALLOW, IPV4, MARKERS  # noqa: E402

DEFAULT_CONFIG = "delivery/stepzero/nuaa_step_zero.json"
DEFAULT_OUT_ROOT = "results/stepzero"
CONFIG_SCHEMA = "nuaa-step-zero-config/1"
RUN_ID_RE = re.compile(r"^\d{8}-STEPZERO-[0-9a-f]{7,40}(-dev)?$")
CODE_FILES = ("scripts/nuaa_step_zero.py", "scripts/m2_writers.py", "scripts/path_guard.py", "scripts/package_rules.py")
DECISION_RULE = "attack if score_attack > threshold (tie = bona fide)"
DECLARATIONS = ("no_training", "no_inference", "no_face_detection", "no_new_threshold", "associations_not_causes")
TOP_KEYS = ("schema", "task", "nature", "authorisation", "declarations", "threshold", "decision_rule", "statistics", "views", "measures",
            "bootstrap", "manifests", "images", "cache", "scores", "face_boxes", "sets", "expected", "outputs")
MEASURE_KEYS = ("quartiles", "class_auroc", "class_auroc_ci", "within_subject_auroc", "all_row", "spearman", "rejected_vs_accepted",
                "crop_view", "reading")
LIST_SPLIT = {"official_train": "train", "cnn_train": "train", "cnn_es": "train", "official_test": "test", "cnn_test": "test", "c1": "test"}
BOOT = {"resamples": 1000, "seed": 0, "percentiles": [2.5, 97.5]}     # bootstrap sui soggetti dell'AUROC di classe
NA = "NA"
STATS = ("lum_mean", "lum_std", "sat_mean", "lap_var")
RAW_EXTRA = ("width", "height", "bytes")
RAW_STATS = STATS + RAW_EXTRA
MANIFEST_COLUMNS = ("key", "label", "subject", "glasses", "pos", "session", "pic", "official_split", "sha256", "bytes")
MANIFESTS = ("official_train", "official_test", "cnn_train", "cnn_es", "cnn_test", "c1")
SESSIONS = ("01", "02", "03")
SESSION_OF_SPLIT = {"01": "train", "02": "train", "03": "test"}
CACHE_MEMBERS = ("X_train", "y_train", "s_train", "X_test", "y_test", "s_test")
NPY_HEADERS = {(1, 0): "read_array_header_1_0", (2, 0): "read_array_header_2_0"}
SCORE_COLUMNS = ("sample_id", "subject_id", "label", "score_attack", "prediction")
BOX_COLUMNS = ("extractor", "sample_id", "label", "bbox_w", "bbox_h")
RAW_SETS = ("official_train", "official_test", "cnn_train", "cnn_es", "cnn_test", "c1", "target_491")
CROP_SETS = ("cnn_train", "cnn_es", "cnn_test", "c1")
NAME_RE = re.compile(r"^(\d{4})_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d+)\.jpg$")
KEY_DIR_OF_LABEL = {0: "real", 1: "attack"}
LAYOUT = {("0", "train"): "bonafide_training", ("1", "train"): "attack_training",
          ("0", "test"): "bonafide_validation", ("1", "test"): "attack_validation"}

STATUS = "status.json"
PUBLIC, RESTRICTED = "public", "restricted"
FIGURE = "figure_distributions.png"
PUBLIC_OUTPUTS = ("aggregate.json", "aggregate.md", FIGURE, "SHA256SUMS")
RESTRICTED_OUTPUTS = ("per_image_raw.csv", "per_image_crop.csv", "per_subject.csv", "per_subject_auroc.csv", "public_SHA256SUMS.sha256", "SHA256SUMS")
FAIL_DOC = {"stage": "invalidation-failed", "note": "this output could not be removed after a failure: no verdict is valid"}
# cartelle protette con percorso assoluto (dati, pesi, risultati consegnati, export), in entrambe le direzioni
EXTRA_PROTECTED = ("data", "models", "results/c1", "results/c1crop", "results/m1", "results/m2", "results/m2_rgb", "results/eval",
                   "results/n1", "results/n2", "export", "delivery")
REPO_STATE = None       # (root) -> {"commit", "dirty"}; None: git nel repository (sostituibile solo dai test)
PLOTTER = None          # (doc, arrays) -> byte PNG; None: matplotlib (sostituibile solo dai test)


class Failure(Exception):
    """Rifiuto o errore con motivo esplicito (riportato in status.json)."""


def fail(msg: str):
    raise Failure(msg)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


# ============================================================================ configurazione

def load_config(p: Path) -> tuple:
    """(configurazione, SHA-256 dei byte letti). Campi obbligatori e insiemi chiusi controllati qui, prima di tutto."""
    data = read_regular(p, "configuration")
    try:
        cfg = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        fail(f"configuration {p.name} is not valid JSON: {e}")
    if not isinstance(cfg, dict) or cfg.get("schema") != CONFIG_SCHEMA:
        fail(f"configuration {p.name}: schema must be {CONFIG_SCHEMA!r}")
    if sorted(cfg) != sorted(TOP_KEYS):
        fail(f"configuration: fields must be exactly {', '.join(TOP_KEYS)} (missing {sorted(set(TOP_KEYS) - set(cfg))}, "
             f"unknown {sorted(set(cfg) - set(TOP_KEYS))})")
    for k in ("task", "nature", "authorisation"):
        _text(cfg[k], k)
    _texts(cfg["views"], ("raw", "crop"), "views")
    _texts(cfg["measures"], MEASURE_KEYS, "measures")
    _texts(cfg["sets"], RAW_SETS, "sets")
    _texts(cfg["outputs"], ("public", "restricted"), "outputs")
    if cfg["bootstrap"] != BOOT:
        fail(f"configuration: bootstrap must be {json.dumps(BOOT)} (the recipe of the script)")
    dec = cfg["declarations"]
    if not isinstance(dec, dict):
        fail("configuration: declarations must be a mapping")
    for k in DECLARATIONS:
        if dec.get(k) is not True:
            fail(f"configuration: declarations.{k} must be true")
    for k, v in dec.items():
        if k not in DECLARATIONS and not (isinstance(v, str) and v):
            fail(f"configuration: declarations.{k} must be a text (the boolean declarations are {', '.join(DECLARATIONS)})")
    if cfg.get("threshold") != 0.5 or cfg.get("decision_rule") != DECISION_RULE:
        fail(f"configuration: threshold must be 0.5 and decision_rule {DECISION_RULE!r}")
    st = cfg["statistics"]
    if not isinstance(st, dict) or sorted(st) != sorted(RAW_STATS):
        fail(f"configuration: statistics must define exactly {', '.join(RAW_STATS)}")
    man = cfg["manifests"]
    if not isinstance(man, dict) or sorted(man) != ["files", "folder", "origin"] or not isinstance(man["folder"], str) \
            or not isinstance(man["files"], dict) or sorted(man["files"]) != sorted(MANIFESTS):
        fail(f"configuration: manifests needs exactly folder, origin and the files {', '.join(MANIFESTS)}")
    _text(man["origin"], "manifests.origin")
    for k, v in man["files"].items():
        if not isinstance(v, dict) or not isinstance(v.get("file"), str) or not _hex(v.get("sha256")):
            fail(f"configuration: manifest {k} needs file and sha256")
        if not all(is_int(v.get(f)) and v[f] >= 0 for f in ("rows", "n_bona_fide", "n_attack", "n_subjects")):
            fail(f"configuration: manifest {k} needs the integer counts rows, n_bona_fide, n_attack, n_subjects")
        if not isinstance(v.get("sessions"), list) or not v["sessions"] or not set(v["sessions"]) <= set(SESSIONS):
            fail(f"configuration: manifest {k} needs its sessions (subset of {', '.join(SESSIONS)})")
    if not isinstance(cfg["images"], dict) or sorted(cfg["images"]) != ["folder", "layout"] or not isinstance(cfg["images"]["folder"], str):
        fail("configuration: images needs exactly folder and layout")
    _text(cfg["images"]["layout"], "images.layout")
    c = cfg["cache"]
    if not isinstance(c, dict) or sorted(c) != ["built", "members", "order", "path", "per_class", "sha256"] or not isinstance(c["path"], str) \
            or not _hex(c["sha256"]) or not is_int(c["per_class"]):
        fail("configuration: cache needs exactly path, sha256, per_class, members, order and built")
    _text(c["order"], "cache.order")
    _text(c["built"], "cache.built")
    if not isinstance(c.get("members"), dict) or sorted(c["members"]) != sorted(CACHE_MEMBERS):
        fail(f"configuration: cache.members must be exactly {', '.join(CACHE_MEMBERS)}")
    for k, m in c["members"].items():
        if not isinstance(m, dict) or not isinstance(m.get("dtype"), str) or not isinstance(m.get("shape"), list):
            fail(f"configuration: cache member {k} needs dtype and shape")
    sc = cfg["scores"]
    if not isinstance(sc, dict) or not sc:
        fail("configuration: scores must be a non-empty mapping")
    for k, v in sc.items():
        _hashed_input(v, f"score file {k}")
        if v.get("set") not in RAW_SETS or not is_int(v.get("rows")) or not isinstance(v.get("filter"), dict):
            fail(f"configuration: score file {k} needs set (one of {', '.join(RAW_SETS)}), rows and filter")
        if not all(isinstance(x, str) and isinstance(y, str) for x, y in v["filter"].items()):
            fail(f"configuration: score file {k}: filter must map column names to text values")
        if not isinstance(v.get("rejected_vs_accepted"), bool) or not isinstance(v.get("label"), str):
            fail(f"configuration: score file {k} needs rejected_vs_accepted (bool) and label")
    fb = cfg["face_boxes"]
    _hashed_input(fb, "face_boxes")
    if sorted(fb) != ["extractors", "origin", "path", "rows", "set", "sha256", "sums", "sums_sha256"] or fb["set"] not in RAW_SETS \
            or not isinstance(fb["extractors"], list) or not fb["extractors"] or not is_int(fb["rows"]):
        fail("configuration: face_boxes needs exactly path, sha256, sums, sums_sha256, set, extractors, rows and origin")
    _text(fb["origin"], "face_boxes.origin")
    for k, v in cfg["scores"].items():
        if sorted(v) != ["filter", "label", "path", "rejected_vs_accepted", "rows", "set", "sha256", "sums", "sums_sha256"]:
            fail(f"configuration: score file {k} has unknown or missing fields")
    if set(man["files"]["official_train"]["sessions"]) & set(man["files"]["official_test"]["sessions"]):
        fail("configuration: the declared sessions of the official train and test lists overlap")
    exp = cfg["expected"]
    need = ("n_images", "n_subjects", "n_cache_frames", "n_target_491")
    if not isinstance(exp, dict) or not all(is_int(exp.get(k)) and exp[k] >= 0 for k in need):
        fail(f"configuration: expected must give the non-negative integers {', '.join(need)}")
    _relative(man["folder"], "manifests.folder")
    _relative(cfg["images"]["folder"], "images.folder")
    _relative(c["path"], "cache.path")
    return cfg, sha256_bytes(data)


def _text(v, what: str) -> None:
    if not isinstance(v, str) or not v.strip():
        fail(f"configuration: {what} must be a non-empty text")


def _texts(d, keys, what: str) -> None:
    if not isinstance(d, dict) or sorted(d) != sorted(keys):
        fail(f"configuration: {what} must have exactly the fields {', '.join(keys)}")
    for k, v in d.items():
        _text(v, f"{what}.{k}")


def _hex(v) -> bool:
    return isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) is not None


def _relative(p: str, what: str) -> None:
    if not isinstance(p, str) or not p or Path(p).is_absolute() or ".." in Path(p).parts:
        fail(f"configuration: {what} must be a relative path without '..'")


def _hashed_input(v, what: str) -> None:
    if not isinstance(v, dict) or not all(isinstance(v.get(f), str) and v[f] for f in ("path", "sha256", "sums", "sums_sha256")):
        fail(f"configuration: {what} needs path, sha256, sums and sums_sha256")
    for f in ("sha256", "sums_sha256"):
        if not _hex(v[f]):
            fail(f"configuration: {what}: {f} is not a SHA-256")
    for f in ("path", "sums"):
        _relative(v[f], f"{what}.{f}")


# ============================================================================ ingressi

def read_regular(p: Path, what: str) -> bytes:
    """Byte di un file regolare (non un collegamento simbolico, non una cartella, non un FIFO), letti una sola volta."""
    try:
        st = os.lstat(p)
    except OSError:
        fail(f"{what} {p.name} not found")
    if stat.S_ISLNK(st.st_mode):
        fail(f"{what} {p.name} is a symbolic link: inputs must be regular files")
    if not stat.S_ISREG(st.st_mode):
        fail(f"{what} {p.name} is not a regular file")
    with open(p, "rb") as f:
        return f.read()


def sums_entry(text: str, name: str) -> str:
    """SHA-256 della riga di SHA256SUMS per `name` (formato shasum, anche con '*'); None se assente o ripetuta."""
    hits = []
    for line in text.splitlines():
        m = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        if m and m.group(2) == name:
            hits.append(m.group(1))
    return hits[0] if len(hits) == 1 else None


def verify_hashed(spec: dict, base: Path, what: str) -> bytes:
    """Byte verificati di un ingresso con cartella: SHA-256 del file = configurazione = riga del SHA256SUMS della sua
    cartella, e SHA-256 del SHA256SUMS = configurazione. Qualunque differenza: FAIL (nessun calcolo)."""
    p, sp = base / spec["path"], base / spec["sums"]
    b = read_regular(p, what)
    h = sha256_bytes(b)
    if h != spec["sha256"]:
        fail(f"{what} ({p.name}): SHA-256 mismatch: {h} instead of the accepted {spec['sha256']}")
    sb = read_regular(sp, f"SHA256SUMS of {what}")
    sh = sha256_bytes(sb)
    if sh != spec["sums_sha256"]:
        fail(f"SHA256SUMS of {what}: SHA-256 mismatch: {sh} instead of the accepted {spec['sums_sha256']}")
    try:
        rel = p.resolve().relative_to(sp.resolve().parent).as_posix()
    except ValueError:
        fail(f"{what}: {p.name} is not inside the folder of its SHA256SUMS")
    listed = sums_entry(sb.decode("utf-8", errors="replace"), rel)
    if listed is None:
        fail(f"{what}: {rel} is not listed exactly once in its SHA256SUMS")
    if listed != h:
        fail(f"{what}: SHA-256 mismatch with its SHA256SUMS ({listed} listed, {h} on disk)")
    return b


def parse_csv(b: bytes, what: str, columns) -> list:
    try:
        text = b.decode("utf-8-sig")
    except UnicodeDecodeError:
        fail(f"{what} is not UTF-8")
    try:
        rd = csv.reader(io.StringIO(text, newline=""))
        header = next(rd, [])
        body = list(rd)
    except csv.Error as e:
        fail(f"{what}: unreadable CSV ({e})")
    if len(set(header)) != len(header):
        fail(f"{what}: duplicated columns in the header")
    missing = [c for c in columns if c not in header]
    if missing:
        fail(f"{what}: missing columns {missing}")
    ragged = [i + 2 for i, r in enumerate(body) if len(r) != len(header)]
    if ragged:
        fail(f"{what}: rows with a wrong number of fields (lines {ragged[:3]})")
    return [dict(zip(header, r)) for r in body]


def parse_name(key: str, what: str) -> dict:
    """Campi del nome <soggetto>_<ID>_<glasses>_<pos>_<session>_<picNo>.jpg di una chiave real/<nome> o attack/<nome>."""
    parts = key.split("/", 1)
    if len(parts) != 2 or parts[0] not in ("real", "attack"):
        fail(f"{what}: key {key!r} is not real/<name> or attack/<name>")
    m = NAME_RE.match(parts[1])
    if not m or m.group(1) != m.group(2):
        fail(f"{what}: {key!r} is not a NUAA file name <subject>_<ID>_<glasses>_<pos>_<session>_<picNo>.jpg with subject = ID")
    return {"dir": parts[0], "name": parts[1], "subject": m.group(1), "session": m.group(5)}


def parse_number(text: str, what: str) -> float:
    try:
        v = float(text)
    except (TypeError, ValueError):
        fail(f"{what}: {text!r} is not a number")
    if v != v or v in (float("inf"), float("-inf")):
        fail(f"{what}: {text!r} is not finite")
    return v


def spread(items, k):
    """Copia di scripts/train_cnn.py: k elementi a passo costante dall'elenco ORDINATO."""
    items = sorted(items)
    if k is None or k >= len(items):
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def spread_keys(rows, per_class: int) -> list:
    """Chiavi di spread per classe, bona fide e poi attacchi (copia di scripts/nuaa_manifests.py, M2)."""
    out = []
    for lab in ("0", "1"):
        names = [r["key"].split("/", 1)[1] for r in rows if r["label"] == lab]
        out += [f"{KEY_DIR_OF_LABEL[int(lab)]}/{n}" for n in spread(names, per_class)]
    return out


class Inputs:
    """Manifest, insiemi, cache, score e riquadri verificati; ogni controllo in checks."""


def load_manifests(cfg: dict, base: Path, inp: Inputs) -> None:
    """I sei manifest M2: SHA-256, colonne, righe valide e coerenti con il nome (soggetto, sessione, split, classe),
    conteggi della configurazione; liste ufficiali disgiunte per chiave e per hash; sottoinsiemi contenuti e derivati
    come dichiarato (cnn_train + cnn_es = spread della lista di training; cnn_test = spread della lista di test, c1
    dentro cnn_test); target_491 = official_train meno cnn_train meno cnn_es."""
    folder = base / cfg["manifests"]["folder"]
    inp.rows, inp.by_key, inp.checks = {}, {}, {}
    for name in MANIFESTS:
        spec = cfg["manifests"]["files"][name]
        p = folder / spec["file"]
        b = read_regular(p, f"manifest {name}")
        h = sha256_bytes(b)
        if h != spec["sha256"]:
            fail(f"manifest {name} ({p.name}): SHA-256 mismatch: {h} instead of the accepted {spec['sha256']}")
        rows = parse_csv(b, f"manifest {name}", MANIFEST_COLUMNS)
        seen = {}
        for i, r in enumerate(rows, 2):
            where = f"manifest {name} line {i}"
            f = parse_name(r["key"], where)
            if r["label"] not in ("0", "1") or KEY_DIR_OF_LABEL[int(r["label"])] != f["dir"]:
                fail(f"{where}: label {r['label']!r} does not match the key folder {f['dir']}")
            if r["subject"] != f["subject"]:
                fail(f"{where}: subject {r['subject']!r} differs from the name ({f['subject']})")
            if r["session"] != f["session"] or r["session"] not in spec["sessions"]:
                fail(f"{where}: session {r['session']!r} differs from the name ({f['session']}) or is outside {spec['sessions']}")
            if r["official_split"] != LIST_SPLIT[name]:
                fail(f"{where}: official_split {r['official_split']!r} differs from the list {name} ({LIST_SPLIT[name]})")
            if r["official_split"] != SESSION_OF_SPLIT[r["session"]]:
                fail(f"{where}: official_split {r['official_split']!r} does not follow from session {r['session']}")
            if not _hex(r["sha256"]) or not r["bytes"].isdigit() or int(r["bytes"]) <= 0:
                fail(f"{where}: sha256 or bytes malformed")
            if r["key"] in seen:
                fail(f"{where}: duplicated key {r['key']}")
            seen[r["key"]] = r
        nb = sum(1 for r in rows if r["label"] == "0")
        got = {"rows": len(rows), "n_bona_fide": nb, "n_attack": len(rows) - nb, "n_subjects": len({r["subject"] for r in rows})}
        for k, v in got.items():
            if spec[k] != v:
                fail(f"manifest {name}: {k} {v} differs from the configuration ({spec[k]})")
        inp.rows[name], inp.by_key[name] = rows, seen
        inp.checks[f"manifest_{name}"] = {**got, "sha256": h}
    tr, te = inp.by_key["official_train"], inp.by_key["official_test"]
    if set(tr) & set(te):
        fail("official train and test lists share a key")
    if {r["sha256"] for r in tr.values()} & {r["sha256"] for r in te.values()}:
        fail("official train and test lists share a file SHA-256")
    for sub, src in (("cnn_train", "official_train"), ("cnn_es", "official_train"), ("cnn_test", "official_test"), ("c1", "official_test")):
        for k, r in inp.by_key[sub].items():
            if k not in inp.by_key[src] or inp.by_key[src][k]["sha256"] != r["sha256"] or inp.by_key[src][k]["bytes"] != r["bytes"]:
                fail(f"manifest {sub}: {k} is not a row of {src} with the same SHA-256 and bytes")
    if set(inp.by_key["cnn_train"]) & set(inp.by_key["cnn_es"]):
        fail("cnn_train and cnn_es share a key")
    if not set(inp.by_key["c1"]) <= set(inp.by_key["cnn_test"]):
        fail("c1 is not inside cnn_test")
    k = cfg["cache"]["per_class"]
    inp.order = {"train": spread_keys(inp.rows["official_train"], k), "test": spread_keys(inp.rows["official_test"], k)}
    if sorted(inp.order["train"]) != sorted(list(inp.by_key["cnn_train"]) + list(inp.by_key["cnn_es"])):
        fail("the sampling order of the official train list does not give cnn_train + cnn_es")
    if inp.order["test"] != spread_keys(inp.rows["cnn_test"], k) or set(inp.order["test"]) != set(inp.by_key["cnn_test"]):
        fail("the sampling order of the official test list does not give cnn_test")
    used = set(inp.by_key["cnn_train"]) | set(inp.by_key["cnn_es"])
    inp.rows["target_491"] = sorted((r for r in inp.rows["official_train"] if r["key"] not in used), key=lambda r: r["key"])
    inp.by_key["target_491"] = {r["key"]: r for r in inp.rows["target_491"]}
    inp.all_rows = sorted(list(tr.values()) + list(te.values()), key=lambda r: r["key"])
    inp.subjects = sorted({r["subject"] for r in inp.all_rows})
    exp = cfg["expected"]
    got = {"n_images": len(inp.all_rows), "n_subjects": len(inp.subjects), "n_target_491": len(inp.rows["target_491"])}
    for k, v in got.items():
        if exp[k] != v:
            fail(f"official lists: {k} {v} differs from the configuration ({exp[k]})")
    inp.checks["official_lists"] = {**got,
                                    "sampling_order": "spread per class (bona fide first) on the official lists: equal to cnn_train + cnn_es and to cnn_test"}


def locate(root: Path, r: dict) -> Path:
    """Il file originale di una riga del manifest nella disposizione <classe>_<split>/<nome>: accettato solo se file
    regolare con SHA-256 e dimensione uguali alla riga (la provenienza è l'hash, non il percorso)."""
    p = root / LAYOUT[(r["label"], r["official_split"])] / r["key"].split("/", 1)[1]
    if p.is_symlink():
        fail(f"original of {r['key']}: symbolic link refused")
    if not p.is_file():
        fail(f"original of {r['key']} not found under the images folder")
    if p.stat().st_size != int(r["bytes"]):
        fail(f"original of {r['key']}: size differs from the manifest")
    if sha256_file(p) != r["sha256"]:
        fail(f"original of {r['key']}: SHA-256 differs from the manifest")
    return p


def npy_header(fh, what: str) -> tuple:
    ver = npy_format.read_magic(fh)
    if ver not in NPY_HEADERS:
        fail(f"{what}: .npy format version {ver[0]}.{ver[1]} not supported (1.0 or 2.0 expected)")
    return getattr(npy_format, NPY_HEADERS[ver])(fh)


def npz_members(p: Path) -> dict:
    """Membri della cache letti dalle intestazioni .npy (tipo e forma), senza caricare gli array; nessun pickle."""
    out = {}
    try:
        with zipfile.ZipFile(p) as zf:
            names = zf.namelist()
            for n in names:
                if not n.endswith(".npy"):
                    fail(f"cache {p.name}: member {n!r} is not an .npy array")
                with zf.open(n) as fh:
                    shape, fortran, dtype = npy_header(fh, f"cache {p.name}: member {n!r}")
                if dtype.hasobject:
                    fail(f"cache {p.name}: member {n!r} holds Python objects (pickle refused)")
                out[n[:-4]] = {"dtype": dtype.str, "shape": list(shape), "fortran": bool(fortran)}
            if len(set(names)) != len(names):
                fail(f"cache {p.name}: duplicated members")
    except (OSError, zipfile.BadZipFile, ValueError) as e:
        fail(f"cache {p.name} unreadable ({e})")
    return out


def load_cache_schema(cfg: dict, base: Path, inp: Inputs) -> None:
    c = cfg["cache"]
    cp = base / c["path"]
    if cp.is_symlink() or not cp.is_file():
        fail(f"cache {cp.name} is missing or is not a regular file")
    inp.cache_path = cp
    inp.cache_sha = sha256_file(cp)
    if inp.cache_sha != c["sha256"]:
        fail(f"cache {cp.name}: SHA-256 {inp.cache_sha} differs from the expected {c['sha256']}")
    mem = npz_members(cp)
    if set(mem) != set(CACHE_MEMBERS):
        fail(f"cache {cp.name}: members {sorted(mem)} differ from {sorted(CACHE_MEMBERS)}")
    for k, want in c["members"].items():
        got = {"dtype": mem[k]["dtype"], "shape": mem[k]["shape"]}
        if got != want or mem[k]["fortran"]:
            fail(f"cache {cp.name}: {k} has dtype {got['dtype']}, shape {got['shape']}, expected {want['dtype']}, {want['shape']}")
    n = mem["X_train"]["shape"][0] + mem["X_test"]["shape"][0]
    if n != cfg["expected"]["n_cache_frames"] or mem["X_train"]["shape"][0] != len(inp.order["train"]) or mem["X_test"]["shape"][0] != len(inp.order["test"]):
        fail(f"cache {cp.name}: {n} frames do not match the expected count and the sampling orders")
    inp.checks["cache"] = {"sha256": inp.cache_sha, "frames": n, "members": {k: {"dtype": v["dtype"], "shape": v["shape"]} for k, v in sorted(mem.items())},
                           "reading": "each member decompressed once, in sampling order (compressed npz: no memory mapping)"}


def load_scores(cfg: dict, base: Path, inp: Inputs) -> None:
    """Ogni file di score: verificato per SHA-256 e SHA256SUMS, filtrato alle righe dichiarate, chiavi = insieme
    dichiarato, soggetto ed etichetta uguali al manifest, score finito, previsione salvata = regola score > 0,5."""
    inp.scores = {}
    t = cfg["threshold"]
    for name, spec in cfg["scores"].items():
        b = verify_hashed(spec, base, f"score file {name}")
        rows = [r for r in parse_csv(b, f"score file {name}", SCORE_COLUMNS) if all(r.get(c) == v for c, v in spec["filter"].items())]
        keys = inp.by_key[spec["set"]]
        out = {}
        for r in rows:
            where = f"score file {name}, sample {r['sample_id']}"
            if r["sample_id"] in out:
                fail(f"{where}: duplicated sample")
            m = keys.get(r["sample_id"])
            if m is None:
                fail(f"{where}: not a key of the set {spec['set']}")
            if r["subject_id"] != m["subject"] or r["label"] != m["label"]:
                fail(f"{where}: subject or label differ from the manifest")
            s = parse_number(r["score_attack"], where)
            if r["prediction"] not in ("0", "1") or int(r["prediction"]) != int(s > t):
                fail(f"{where}: stored prediction {r['prediction']!r} differs from the rule score > {t}")
            out[r["sample_id"]] = (s, int(r["prediction"]))
        if len(out) != spec["rows"] or len(out) != len(keys):
            fail(f"score file {name}: {len(out)} rows after the filter instead of {spec['rows']} (set {spec['set']}: {len(keys)} keys)")
        inp.scores[name] = out
        inp.checks[f"scores_{name}"] = {"rows": len(out), "set": spec["set"], "sha256": spec["sha256"]}


def load_boxes(cfg: dict, base: Path, inp: Inputs) -> None:
    """Riquadri del confronto del ritaglio: per estrattore, una riga per immagine dell'insieme dichiarato."""
    spec = cfg["face_boxes"]
    b = verify_hashed(spec, base, "face boxes")
    rows = parse_csv(b, "face boxes", BOX_COLUMNS)
    keys = inp.by_key[spec["set"]]
    out = {e: {} for e in spec["extractors"]}
    for r in rows:
        where = f"face boxes, {r['extractor']} {r['sample_id']}"
        if r["extractor"] not in out:
            fail(f"{where}: extractor not declared in the configuration")
        if r["sample_id"] in out[r["extractor"]]:
            fail(f"{where}: duplicated sample")
        m = keys.get(r["sample_id"])
        if m is None or r["label"] != m["label"]:
            fail(f"{where}: not a key of the set {spec['set']}, or label different from the manifest")
        w, h = parse_number(r["bbox_w"], where), parse_number(r["bbox_h"], where)
        if w <= 0 or h <= 0:
            fail(f"{where}: box width and height must be positive")
        out[r["extractor"]][r["sample_id"]] = (w, h)
    if sum(len(d) for d in out.values()) != spec["rows"] or any(len(d) != len(keys) for d in out.values()):
        fail(f"face boxes: {sum(len(d) for d in out.values())} rows instead of {spec['rows']}, or an extractor without every key of {spec['set']}")
    inp.boxes = out
    inp.box_set = spec["set"]
    inp.checks["face_boxes"] = {"rows": spec["rows"], "extractors": list(spec["extractors"]), "sha256": spec["sha256"]}


def check_originals(cfg: dict, base: Path, inp: Inputs) -> None:
    root = base / cfg["images"]["folder"]
    if root.is_symlink() or not root.is_dir():
        fail("images folder missing or a symbolic link")
    inp.images_root = root
    inp.located = {r["key"]: locate(root, r) for r in inp.all_rows}
    inp.checks["originals"] = {"located_and_hashed": len(inp.all_rows), "layout": "<class>_<split>/<name>"}


def load_inputs(cfg: dict, base: Path) -> Inputs:
    inp = Inputs()
    load_manifests(cfg, base, inp)
    load_cache_schema(cfg, base, inp)
    load_scores(cfg, base, inp)
    load_boxes(cfg, base, inp)
    check_originals(cfg, base, inp)
    return inp


def input_paths(cfg: dict, base: Path) -> dict:
    """Etichetta -> percorso di ogni ingresso (per la validazione per identità della cartella del run)."""
    out = {"manifests folder": base / cfg["manifests"]["folder"], "images folder": base / cfg["images"]["folder"], "cache": base / cfg["cache"]["path"]}
    for k, v in cfg["scores"].items():
        out[f"score file {k}"] = base / v["path"]
        out[f"SHA256SUMS of score file {k}"] = base / v["sums"]
    out["face boxes"] = base / cfg["face_boxes"]["path"]
    out["SHA256SUMS of face boxes"] = base / cfg["face_boxes"]["sums"]
    return out


# ============================================================================ statistiche

def image_stats(bgr_or_rgb, rgb: bool) -> dict:
    """Le quattro statistiche di un'immagine uint8 HxWx3 (BGR da cv2.imread, oppure RGB per i frame della cache)."""
    import cv2
    a = np.asarray(bgr_or_rgb)
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        fail("image statistics need a uint8 HxWx3 image")
    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY if rgb else cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV if rgb else cv2.COLOR_BGR2HSV)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=1)
    g = gray.astype(np.float64)
    return {"lum_mean": float(g.mean()), "lum_std": float(g.std()), "sat_mean": float(hsv[:, :, 1].astype(np.float64).mean()),
            "lap_var": float(lap.var())}


def raw_stats(inp: Inputs) -> dict:
    """Chiave -> statistiche raw (le quattro più larghezza, altezza e byte del manifest) di ogni immagine ufficiale."""
    import cv2
    out = {}
    for r in inp.all_rows:
        img = cv2.imread(str(inp.located[r["key"]]), cv2.IMREAD_COLOR)
        if img is None:
            fail(f"original of {r['key']}: not decodable as an image")
        s = image_stats(img, rgb=False)
        s.update(width=int(img.shape[1]), height=int(img.shape[0]), bytes=int(r["bytes"]))
        out[r["key"]] = s
    return out


def crop_stats(inp: Inputs) -> dict:
    """Chiave -> statistiche crop dei 6.000 frame della cache, membro per membro, nell'ordine di campionamento; le
    etichette e i soggetti salvati nella cache devono coincidere con il manifest."""
    out = {}
    with np.load(inp.cache_path) as z:
        for member, order in (("train", inp.order["train"]), ("test", inp.order["test"])):
            X, y, s = z[f"X_{member}"], z[f"y_{member}"], z[f"s_{member}"]
            src = inp.by_key["official_train"] if member == "train" else inp.by_key["official_test"]
            for i, key in enumerate(order):
                r = src[key]
                if str(int(y[i])) != r["label"] or str(s[i]) != r["subject"]:
                    fail(f"cache X_{member} row {i}: label or subject differ from the manifest row of {key}")
                out[key] = {**image_stats(X[i], rgb=True), "member": f"X_{member}", "index": i}
            del X
    return out


# ============================================================================ misure di associazione

def ranks(x) -> np.ndarray:
    """Ranghi 1..n con i pareggi al rango medio."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    _, inv, counts = np.unique(x[order], return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)                                   # posizione (da 1) dell'ultimo di ogni gruppo di pareggi
    avg = (ends - counts + 1 + ends) / 2
    r = np.empty(len(x), dtype=np.float64)
    r[order] = avg[inv]
    return r


def auroc(pos, neg):
    """AUROC = P(x_pos > x_neg) + 0,5 P(x_pos = x_neg) (statistica U di Mann-Whitney sui ranghi medi); None se una
    classe è vuota."""
    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return None
    r = ranks(np.concatenate([pos, neg]))
    u = r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def spearman(x, y):
    """Correlazione di Spearman: Pearson dei ranghi medi; None con meno di 3 coppie o con una variabile costante."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 3 or len(x) != len(y):
        return None
    rx, ry = ranks(x), ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if d == 0:
        return None
    return float((rx * ry).sum() / d)


def quartiles(v) -> dict:
    v = np.asarray(v, dtype=np.float64)
    if len(v) == 0:
        return {"n": 0, "median": None, "q1": None, "q3": None}
    return {"n": int(len(v)), "median": float(np.median(v)), "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75))}


def by_session_class(rows, stats: dict, names) -> dict:
    """sessione -> classe -> statistica -> valori, per le righe del manifest con statistiche."""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        s = stats[r["key"]]
        for n in names:
            out[r["session"]]["attack" if r["label"] == "1" else "bona_fide"][n].append(s[n])
    return out


def by_subject(rows, stats: dict, names) -> dict:
    """sessione (e «all») -> soggetto -> classe -> statistica -> array dei valori."""
    g = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for r in rows:
        s = stats[r["key"]]
        c = "attack" if r["label"] == "1" else "bona_fide"
        for ses in (r["session"], "all"):
            for n in names:
                g[ses][r["subject"]][c][n].append(s[n])
    return {ses: {subj: {c: {n: np.asarray(v, dtype=np.float64) for n, v in d.items()} for c, d in cls.items()} for subj, cls in subs.items()}
            for ses, subs in g.items()}


def subject_aurocs(subs: dict, names) -> dict:
    """AUROC di classe entro soggetto: soggetto -> (n bona fide, n attacchi, {statistica: AUROC}) per i soli soggetti
    con entrambe le classi nella sessione."""
    out = {}
    for subj, cls in sorted(subs.items()):
        att, bf = cls.get("attack", {}), cls.get("bona_fide", {})
        if not att or not bf:
            continue
        out[subj] = (int(len(bf[names[0]])), int(len(att[names[0]])), {n: auroc(att[n], bf[n]) for n in names})
    return out


def bootstrap_auroc(subs: dict, names) -> dict:
    """Intervallo per soggetto dell'AUROC di classe aggregato: BOOT["resamples"] ricampionamenti con reinserimento dei
    soggetti della sessione (stesso generatore, seme BOOT["seed"], per ogni statistica: stessi indici), AUROC sui frame
    dei soggetti estratti, percentili BOOT["percentiles"]; un ricampionamento senza una delle classi non conta."""
    subjects = sorted(subs)
    out = {}
    for n in names:
        att = [subs[s]["attack"][n] if "attack" in subs[s] else np.empty(0) for s in subjects]
        bf = [subs[s]["bona_fide"][n] if "bona_fide" in subs[s] else np.empty(0) for s in subjects]
        rng = np.random.default_rng(BOOT["seed"])
        vals = []
        for _ in range(BOOT["resamples"]):
            idx = rng.integers(0, len(subjects), len(subjects))
            a = auroc(np.concatenate([att[i] for i in idx]), np.concatenate([bf[i] for i in idx]))
            if a is not None:
                vals.append(a)
        if vals:
            lo, hi = np.percentile(vals, BOOT["percentiles"])
            out[n] = {"low": float(lo), "high": float(hi), "valid_resamples": len(vals)}
        else:
            out[n] = {"low": None, "high": None, "valid_resamples": 0}
    return out


def set_summary(rows, stats: dict, names, boot: bool = True) -> dict:
    """Per sessione (e «all» quando le sessioni sono più di una): quartili per classe, AUROC di classe per statistica
    con l'intervallo bootstrap sui soggetti, AUROC entro soggetto (mediana e min-max fra i soggetti con entrambe le
    classi)."""
    g = by_session_class(rows, stats, names)
    sessions = sorted(g)
    per_subject = by_subject(rows, stats, names)
    out = {}
    for ses in sessions + (["all"] if len(sessions) > 1 else []):
        if ses == "all":
            cls = {c: {n: sum((g[x][c][n] for x in sessions), []) for n in names} for c in ("bona_fide", "attack")}
        else:
            cls = {c: {n: g[ses][c][n] for n in names} for c in ("bona_fide", "attack")}
        within = subject_aurocs(per_subject[ses], names)
        out[ses] = {"n_bona_fide": len(cls["bona_fide"][names[0]]), "n_attack": len(cls["attack"][names[0]]),
                    "n_subjects": len(per_subject[ses]),
                    "classes": {c: {n: quartiles(cls[c][n]) for n in names} for c in cls},
                    "class_auroc": {n: auroc(cls["attack"][n], cls["bona_fide"][n]) for n in names},
                    "class_auroc_ci": bootstrap_auroc(per_subject[ses], names) if boot else "not computed",
                    "within_subject_auroc": {"n_subjects": len(within),
                                             **{n: ({"median": float(np.median([v[2][n] for v in within.values()])),
                                                     "min": float(min(v[2][n] for v in within.values())),
                                                     "max": float(max(v[2][n] for v in within.values()))} if within else None) for n in names}}}
    return out


def score_associations(rows, scores: dict, stats: dict, names) -> dict:
    """Per sessione (e «all») e per classe (bona fide, attacco, tutte): n e Spearman fra statistica e score."""
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        c = "attack" if r["label"] == "1" else "bona_fide"
        g[r["session"]][c].append((stats[r["key"]], scores[r["key"]][0]))
    sessions = sorted(g)
    out = {}
    for ses in sessions + (["all"] if len(sessions) > 1 else []):
        part = {c: sum((g[x][c] for x in sessions), []) if ses == "all" else g[ses][c] for c in ("bona_fide", "attack")}
        part["all"] = part["bona_fide"] + part["attack"]
        out[ses] = {c: {"n": len(v), "spearman": {n: spearman([s[n] for s, _ in v], [sc for _, sc in v]) for n in names}} for c, v in part.items()}
    return out


def rejected_vs_accepted(rows, scores: dict, stats: dict, names, session: str = "03") -> dict:
    """Bona fide della sessione data: rifiutati (previsione 1) contro accettati, AUROC di ogni statistica (rifiutati =
    positivi)."""
    rej = [stats[r["key"]] for r in rows if r["label"] == "0" and r["session"] == session and scores[r["key"]][1] == 1]
    acc = [stats[r["key"]] for r in rows if r["label"] == "0" and r["session"] == session and scores[r["key"]][1] == 0]
    return {"session": session, "n_rejected": len(rej), "n_accepted": len(acc),
            "auroc_rejected_vs_accepted": {n: auroc([s[n] for s in rej], [s[n] for s in acc]) for n in names}}


def box_annex(inp: Inputs, raw: dict) -> dict:
    """Area del riquadro relativa all'immagine, per estrattore: quartili per classe e AUROC come separatore."""
    out = {}
    keys = inp.by_key[inp.box_set]
    for e, d in inp.boxes.items():
        rel = {"bona_fide": [], "attack": []}
        for k, (w, h) in d.items():
            r = raw[k]
            rel["attack" if keys[k]["label"] == "1" else "bona_fide"].append(w * h / (r["width"] * r["height"]))
        out[e] = {"n_bona_fide": len(rel["bona_fide"]), "n_attack": len(rel["attack"]),
                  "relative_area": {c: quartiles(v) for c, v in rel.items()}, "class_auroc": auroc(rel["attack"], rel["bona_fide"])}
    return out


# ============================================================================ uscite

def f6(v) -> str:
    return NA if v is None else f"{v:.6f}"


def f4(v) -> str:
    return NA if v is None else f"{v:.4f}"


def csv_text(header, rows) -> str:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def sums_text(files: dict) -> str:
    return "".join(f"{sha256_bytes(b if isinstance(b, bytes) else b.encode('utf-8'))}  {n}\n" for n, b in sorted(files.items()))


def public_problems(text: str, inp: Inputs, markdown: bool = False) -> list:
    """Contenuto vietato in un file pubblico: chiavi o SHA-256 delle immagini, identificativi di soggetto (fuori da
    numeri e hash), regole di package_rules (percorsi, credenziali, strumenti), IPv4 non locali, segni di lavoro non
    finito (solo nei .md)."""
    bad = []
    if re.search(r"\b(real|attack)/\d{4}_", text):
        bad.append("image keys")
    if set(re.findall(r"[0-9a-f]{64}", text)) & {r["sha256"] for r in inp.all_rows}:
        bad.append("image SHA-256 values")
    for s in inp.subjects:
        if re.search(rf"(?<![\w.]){re.escape(s)}(?![\w.])", text):
            bad.append(f"subject id {s}")
            break
    for label, rx in FORBIDDEN_TEXT:
        if m := rx.search(text):
            bad.append(f"{label}: {m.group(0)[:40]!r}")
    for ip in (m.group(0) for m in IPV4.finditer(text)):
        if ip not in IP_ALLOW:
            bad.append(f"IP address: {ip}")
            break
    if markdown and (m := MARKERS.search(text)):
        bad.append(f"unfinished-work marker: {m.group(0)!r}")
    return bad


def aggregate_doc(ctx: dict) -> dict:
    cfg, inp, raw, crop = ctx["cfg"], ctx["inp"], ctx["raw"], ctx["crop"]
    sets_raw = {s: set_summary(inp.rows[s], raw, RAW_STATS) for s in RAW_SETS}
    sets_crop = {s: set_summary(inp.rows[s], crop, STATS) for s in CROP_SETS}
    assoc, rva = {}, {}
    for name, spec in cfg["scores"].items():
        rows, sc = inp.rows[spec["set"]], inp.scores[name]
        in_cache = all(r["key"] in crop for r in rows)
        assoc[name] = {"label": spec["label"], "set": spec["set"], "n": len(rows),
                       "raw": score_associations(rows, sc, raw, RAW_STATS),
                       "crop": score_associations(rows, sc, crop, STATS) if in_cache else "NA: frames not in the training cache"}
        if spec["rejected_vs_accepted"]:
            rva[name] = {"label": spec["label"], "raw": rejected_vs_accepted(rows, sc, raw, RAW_STATS),
                         "crop": rejected_vs_accepted(rows, sc, crop, STATS) if in_cache else "NA: frames not in the training cache"}
    bf03 = {r["subject"] for r in inp.rows["official_test"] if r["label"] == "0"}
    at03 = {r["subject"] for r in inp.rows["official_test"] if r["label"] == "1"}
    s03 = {"bona_fide": len(bf03), "attack": len(at03), "both_classes": len(bf03 & at03)}
    return {
        "task": cfg["task"], "nature": cfg["nature"], "declarations": cfg["declarations"],
        "run_id": ctx["run_id"], "commit": ctx["state"]["commit"], "dirty": ctx["state"]["dirty"], "date": ctx["date_iso"],
        "threshold": cfg["threshold"], "decision_rule": DECISION_RULE,
        "statistics": cfg["statistics"], "views": cfg["views"],
        "measures": cfg["measures"], "bootstrap": BOOT,
        "counts": {"official_images": len(inp.all_rows), "subjects": len(inp.subjects), "cache_frames": len(crop),
                   "sets": {s: len(inp.rows[s]) for s in RAW_SETS}, "subjects_session_03": s03},
        "sets_raw": sets_raw, "sets_crop": sets_crop,
        "score_associations": assoc, "rejected_vs_accepted": rva,
        "face_box_annex": {"set": inp.box_set, "relative_area": "bbox_w * bbox_h / (width * height) of the original image",
                           "extractors": box_annex(inp, raw)},
        "figure": ctx["figure_note"],
        "inputs_sha256": {"manifests": {k: v["sha256"] for k, v in cfg["manifests"]["files"].items()}, "cache": cfg["cache"]["sha256"],
                          "scores": {k: v["sha256"] for k, v in cfg["scores"].items()}, "face_boxes": cfg["face_boxes"]["sha256"]},
        "checks": inp.checks, "config_sha256": ctx["config_sha256"], "code_sha256": ctx["code"], "environment": environment(),
        "limits": ["frames of one subject and session are correlated: n counts frames, not independent units",
                   "descriptive measures without statistical tests; the only uncertainty is the bootstrap over subjects of the class AUROC",
                   f"the aggregate class AUROC and the within-class Spearman confound subject and class: in session 03 the bona fide frames "
                   f"come from {s03['bona_fide']} subjects and the attacks from {s03['attack']} subjects, {s03['both_classes']} subjects "
                   f"have both classes; the within-subject AUROC is the reading that holds the subject fixed",
                   "the statistics are global image properties; they do not identify what a network uses",
                   "the only face-level view is the crop view; the raw view is the whole frame (background included; NUAA attacks are "
                   "re-photographed prints with a border)",
                   cfg["measures"]["crop_view"], cfg["measures"]["all_row"],
                   "the crop view covers the 6,000 cache frames only (cnn_train, cnn_es, cnn_test); the 491 target frames have the raw view only",
                   "face boxes exist only for the 300 C1 images (two extractors); no boxes for the 3,000 without a new detection pass, "
                   "which is outside this step",
                   "associations, not causes"],
    }


def md_quartile_table(title: str, summary: dict, names) -> list:
    L = [f"### {title}", "", "| session | class | n | " + " | ".join(f"{n} median (Q1–Q3)" for n in names) + " |",
         "|---|---|---|" + "---|" * len(names)]
    for ses, d in summary.items():
        for c in ("bona_fide", "attack"):
            q = d["classes"][c]
            L.append(f"| {ses} | {c.replace('_', ' ')} | {q[names[0]]['n']} | " + " | ".join(
                NA if q[n]["median"] is None else f"{q[n]['median']:.2f} ({q[n]['q1']:.2f}–{q[n]['q3']:.2f})" for n in names) + " |")
    L += ["", "| session | n bona fide | n attack | n subjects | " + " | ".join(f"AUROC {n} [bootstrap]" for n in names) + " |", "|---|---|---|---|" + "---|" * len(names)]
    for ses, d in summary.items():
        ci = d["class_auroc_ci"]
        L.append(f"| {ses} | {d['n_bona_fide']} | {d['n_attack']} | {d['n_subjects']} | " + " | ".join(
            f4(d["class_auroc"][n]) + (f" [{f4(ci[n]['low'])}–{f4(ci[n]['high'])}]" if isinstance(ci, dict) else "") for n in names) + " |")
    L += ["", "| session | subjects with both classes | " + " | ".join(f"within-subject AUROC {n}: median (min–max)" for n in names) + " |", "|---|---|" + "---|" * len(names)]
    for ses, d in summary.items():
        w = d["within_subject_auroc"]
        L.append(f"| {ses} | {w['n_subjects']} | " + " | ".join(
            NA if w[n] is None else f"{w[n]['median']:.4f} ({w[n]['min']:.4f}–{w[n]['max']:.4f})" for n in names) + " |")
    return L + [""]


def aggregate_md(doc: dict) -> str:
    L = [f"# NUAA step zero: image statistics against the delivered scores (aggregate, no identifiers)", "",
         f"Run `{doc['run_id']}` (commit `{doc['commit']}`, dirty tree: {'yes' if doc['dirty'] else 'no'}), {doc['date']}. "
         f"{doc['counts']['official_images']} official NUAA images ({doc['counts']['subjects']} subjects), {doc['counts']['cache_frames']} cache frames; "
         "no training, no inference, no face detection. Every measure is descriptive: associations, not causes.", "",
         "Statistics: " + "; ".join(f"`{k}` = {v}" for k, v in doc["statistics"].items()) + ".", "",
         f"Class AUROC: {doc['measures']['class_auroc']}. Bootstrap: {doc['measures']['class_auroc_ci']}. "
         f"Within-subject AUROC: {doc['measures']['within_subject_auroc']}. Spearman: {doc['measures']['spearman']}.", "",
         f"Crop view: {doc['measures']['crop_view']}. All row: {doc['measures']['all_row']}.", "",
         "## Sets, raw view (original images)", ""]
    for s, d in doc["sets_raw"].items():
        L += md_quartile_table(f"{s} ({doc['counts']['sets'][s]} images)", d, RAW_STATS)
    L += ["## Sets, crop view (training-cache frames, 256x256 after preprocessing)", ""]
    for s, d in doc["sets_crop"].items():
        L += md_quartile_table(f"{s} ({doc['counts']['sets'][s]} frames)", d, STATS)
    L += ["## Spearman between each statistic and score_attack (per score file, session and class)", ""]
    for name, a in doc["score_associations"].items():
        for view in ("raw", "crop"):
            v = a[view]
            if isinstance(v, str):
                L += [f"### {a['label']} — {view} view: {v}", ""]
                continue
            names = RAW_STATS if view == "raw" else STATS
            L += [f"### {a['label']} — {view} view ({a['set']}, {a['n']} frames)", "",
                  "| session | class | n | " + " | ".join(names) + " |", "|---|---|---|" + "---|" * len(names)]
            for ses, d in v.items():
                for c in ("bona_fide", "attack", "all"):
                    L.append(f"| {ses} | {c.replace('_', ' ')} | {d[c]['n']} | " + " | ".join(f4(d[c]["spearman"][n]) for n in names) + " |")
            L.append("")
    L += ["## Bona fide of session 03: rejected against accepted (AUROC of each statistic, rejected = positive)", ""]
    for name, a in doc["rejected_vs_accepted"].items():
        for view in ("raw", "crop"):
            v = a[view]
            if isinstance(v, str):
                L += [f"- {a['label']}, {view} view: {v}"]
                continue
            names = RAW_STATS if view == "raw" else STATS
            L += [f"- {a['label']}, {view} view: rejected {v['n_rejected']}, accepted {v['n_accepted']}; " +
                  ", ".join(f"{n} {f4(v['auroc_rejected_vs_accepted'][n])}" for n in names)]
    L += ["", f"## Face-box annex ({doc['face_box_annex']['set']}; relative area = {doc['face_box_annex']['relative_area']})", "",
          "| extractor | n bona fide | n attack | bona fide median (Q1–Q3) | attack median (Q1–Q3) | class AUROC |", "|---|---|---|---|---|---|"]
    for e, d in doc["face_box_annex"]["extractors"].items():
        q = d["relative_area"]
        L.append(f"| {e} | {d['n_bona_fide']} | {d['n_attack']} | " + " | ".join(
            NA if q[c]["median"] is None else f"{q[c]['median']:.4f} ({q[c]['q1']:.4f}–{q[c]['q3']:.4f})" for c in ("bona_fide", "attack")) +
            f" | {f4(d['class_auroc'])} |")
    L += ["", f"Figure: {doc['figure']}.", "", "Limits: " + " ".join(x[0].upper() + x[1:] + "." for x in doc["limits"]), ""]
    return "\n".join(L)


def default_plotter(doc: dict, arrays: dict) -> bytes:
    """Figura pubblica: distribuzioni (boxplot) delle quattro statistiche raw per sessione x classe sulle liste
    ufficiali; nessun punto per soggetto, nessun identificativo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(STATS), figsize=(4 * len(STATS), 4))
    for ax, n in zip(axes, STATS):
        data, labels = [], []
        for ses in SESSIONS:
            for c in ("bona_fide", "attack"):
                data.append(arrays[ses][c][n])
                labels.append(f"{ses}\n{'bf' if c == 'bona_fide' else 'att'}")
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=7)
        ax.set_title(n)
    fig.suptitle(f"NUAA official lists, raw view, per session and class ({doc['run_id']})", fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def figure_bytes(doc: dict, inp: Inputs, raw: dict):
    """(byte PNG o None, nota): matplotlib assente -> salto dichiarato, mai un errore."""
    arrays = by_session_class(inp.all_rows, raw, STATS)
    try:
        return (PLOTTER or default_plotter)(doc, arrays), f"{FIGURE}: boxplots of the four raw statistics per session and class on the official lists"
    except ImportError as e:
        return None, f"{FIGURE} skipped: missing Python package {e.name!r}"


def build_outputs(ctx: dict) -> tuple:
    """(file pubblici, file riservati) come testi o byte; i testi pubblici sono scansionati prima."""
    inp, raw, crop = ctx["inp"], ctx["raw"], ctx["crop"]
    ctx["figure_note"] = "pending"
    doc = aggregate_doc(ctx)
    png, note = figure_bytes(doc, inp, raw)
    doc["figure"] = note
    js = json.dumps(doc, indent=1) + "\n"
    md = aggregate_md(doc)
    for name, text, is_md in (("aggregate.json", js, False), ("aggregate.md", md, True)):
        if probs := public_problems(text, inp, is_md):
            fail(f"public {name} would contain forbidden content: {'; '.join(probs)}")
    public = {"aggregate.json": js, "aggregate.md": md}
    if png is not None:
        public[FIGURE] = png
    public["SHA256SUMS"] = sums_text(public)
    rows_raw = [[r["key"], r["subject"], r["session"], r["label"], r["official_split"]] +
                [int(r["key"] in inp.by_key[s]) for s in ("cnn_train", "cnn_es", "cnn_test", "c1", "target_491")] +
                [raw[r["key"]][n] for n in RAW_EXTRA] + [f6(raw[r["key"]][n]) for n in STATS] for r in inp.all_rows]
    rows_crop = [[crop[k]["member"], crop[k]["index"], k, inp.by_key["official_train" if crop[k]["member"] == "X_train" else "official_test"][k]["subject"],
                  k.split("/", 1)[1].split("_")[4], inp.by_key["official_train" if crop[k]["member"] == "X_train" else "official_test"][k]["label"]] +
                 [f6(crop[k][n]) for n in STATS] for k in sorted(crop, key=lambda x: (crop[x]["member"] != "X_train", crop[x]["index"]))]
    per_subject = []
    for view, stats, rows, names in (("raw", raw, inp.all_rows, RAW_STATS), ("crop", crop, [r for r in inp.all_rows if r["key"] in crop], STATS)):
        g = defaultdict(list)
        for r in rows:
            g[(r["subject"], r["session"], r["label"])].append(stats[r["key"]])
        for (subj, ses, lab), vals in sorted(g.items()):
            per_subject.append([view, subj, ses, lab, len(vals)] + [f6(float(np.median([v[n] for v in vals]))) for n in names] +
                               ([NA] * (len(RAW_STATS) - len(names))))
    auroc_rows = []
    for view, stats, names, sets in (("raw", raw, RAW_STATS, RAW_SETS), ("crop", crop, STATS, CROP_SETS)):
        for s in sets:
            per = by_subject(inp.rows[s], stats, names)
            for ses in sorted(per):
                for subj, (nb, na, vals) in subject_aurocs(per[ses], names).items():
                    auroc_rows.append([view, s, ses, subj, nb, na] + [f6(vals[n]) for n in names] + ([NA] * (len(RAW_STATS) - len(names))))
    restricted = {
        "per_subject_auroc.csv": csv_text(["view", "set", "session", "subject_id", "n_bona_fide", "n_attack"] + [f"auroc_{n}" for n in RAW_STATS], auroc_rows),
        "per_image_raw.csv": csv_text(["sample_id", "subject_id", "session_id", "label", "official_split", "in_cnn_train", "in_cnn_es",
                                       "in_cnn_test", "in_c1", "in_target_491"] + list(RAW_EXTRA) + list(STATS), rows_raw),
        "per_image_crop.csv": csv_text(["cache_member", "cache_index", "sample_id", "subject_id", "session_id", "label"] + list(STATS), rows_crop),
        "per_subject.csv": csv_text(["view", "subject_id", "session_id", "label", "n"] + [f"median_{n}" for n in RAW_STATS], per_subject),
        "public_SHA256SUMS.sha256": sha256_bytes(public["SHA256SUMS"].encode("utf-8")) + "  public/SHA256SUMS\n",
    }
    restricted["SHA256SUMS"] = sums_text(restricted)
    return public, restricted, doc


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
    rid = f"{date}-STEPZERO-{state['commit']}" + ("-dev" if state["dirty"] else "")
    if not RUN_ID_RE.match(rid):
        fail(f"run id {rid} does not follow YYYYMMDD-STEPZERO-<commit>[-dev]")
    return rid


def protected_extra(root: Path) -> list:
    return [root / x for x in EXTRA_PROTECTED]


def run_dir_problem(run: Path, root: Path, inputs: dict) -> str:
    """Motivo per cui la cartella del run non è ammessa ("" se lo è), prima di qualunque modifica."""
    if run.parent.is_symlink():
        return f"the output root {run.parent.name} is a symbolic link"
    for label, p in inputs.items():
        if label.startswith("SHA256SUMS of") and pg.within(run, Path(p).parent):
            return f"the run folder is inside the folder of {label} ({Path(p).parent.name}): inputs and outputs must be separate"
        if pg.within(p, run):
            return f"{label} {Path(p).name} is inside the run folder: inputs and outputs must be separate"
    msg = mw.output_problem(run, root, pg.BASE_PROTECTED, [root / DEFAULT_OUT_ROOT], protected_extra(root), inputs, [STATUS],
                            [run / PUBLIC, run / RESTRICTED])
    if msg:
        return msg
    for sub, names in ((PUBLIC, PUBLIC_OUTPUTS), (RESTRICTED, RESTRICTED_OUTPUTS)):
        if msg := mw.output_problem(run / sub, root, pg.BASE_PROTECTED, [root / DEFAULT_OUT_ROOT], protected_extra(root), inputs, list(names)):
            return msg
    allowed = {"": [STATUS, PUBLIC, RESTRICTED], PUBLIC: PUBLIC_OUTPUTS, RESTRICTED: RESTRICTED_OUTPUTS}
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


def write_group(d: Path, files: dict) -> None:
    """Testi con write_all (newline ""), byte (figura) con temporanei e rinomine; nulla resta su errore."""
    texts = {n: v for n, v in files.items() if isinstance(v, str)}
    blobs = {n: v for n, v in files.items() if isinstance(v, bytes)}
    mw.write_all(d, texts, newline="")
    if blobs:
        try:
            write_binary(d, blobs)
        except BaseException:
            for n in texts:
                (d / n).unlink(missing_ok=True)
            raise


def write_binary(d: Path, files: dict) -> None:
    import tempfile
    tmps, done = {}, []
    try:
        for n in files:
            q = d / n
            if q.is_symlink() or (q.exists() and not q.is_file()):
                fail(f"{n} is a link or not a regular file: refused")
        for n, data in files.items():
            fd, tp = tempfile.mkstemp(prefix=f".{n}.", suffix=".tmp", dir=d)
            tmps[n] = Path(tp)
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


def environment() -> dict:
    out = {"python": sys.version.split()[0], "numpy": np.__version__}
    for mod in ("cv2", "matplotlib"):
        try:
            m = __import__(mod)
            out[mod] = str(m.__version__)
        except (ImportError, AttributeError):
            out[mod] = "absent"
    return out


def code_hashes() -> dict:
    return {f: sha256_bytes((ROOT / f).read_bytes()) for f in CODE_FILES}


def resolve(a) -> tuple:
    base = Path(a.input_root).resolve() if a.input_root else ROOT
    cfg_p = Path(a.config) if Path(a.config).is_absolute() else ROOT / a.config
    return base, cfg_p


def cmd_check_inputs(a) -> int:
    base, cfg_p = resolve(a)
    cfg, _ = load_config(cfg_p)
    inp = load_inputs(cfg, base)
    print(f"inputs: {len(MANIFESTS)} manifests, {inp.checks['cache']['frames']} cache frames, {len(cfg['scores'])} score files, "
          f"face boxes, {inp.checks['originals']['located_and_hashed']} originals verified against the configuration")
    print("RESULT: PASS (check-inputs)")
    return 0


def cmd_run(a) -> int:
    if not a.i_am_authorized:
        fail("run writes the run folder: pass --i-am-authorized (the coordinator authorised the step zero on 3 November 2026)")
    base, cfg_p = resolve(a)
    cfg, cfg_sha = load_config(cfg_p)                   # solo per i percorsi da validare: nessuna scrittura prima
    state = repo_state(ROOT)
    date = a.date or time.strftime("%Y%m%d")
    rid = run_id_for(date, state)
    out_root = Path(a.out_root) if Path(a.out_root).is_absolute() else ROOT / a.out_root
    run = out_root / rid
    inputs = {**input_paths(cfg, base), "the configuration": cfg_p}
    if msg := run_dir_problem(run, ROOT, inputs):
        fail(f"refusing the run folder: {msg}")
    run.mkdir(parents=True, exist_ok=True)
    head = {"run_id": rid, "commit": state["commit"], "dirty": state["dirty"], "command": "run"}
    write_status(run, {**head, "status": "RUNNING", "passed": False})
    try:
        left = invalidate_outputs(run)
        if left:
            fail(f"previous outputs could not be removed: {', '.join(left)}")
        inp = load_inputs(cfg, base)
        raw = raw_stats(inp)
        crop = crop_stats(inp)
        ctx = {"cfg": cfg, "inp": inp, "raw": raw, "crop": crop, "run_id": rid, "state": state, "config_sha256": cfg_sha,
               "date_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "code": code_hashes()}
        public, restricted, doc = build_outputs(ctx)
        write_group(run / RESTRICTED, restricted)
        write_group(run / PUBLIC, public)
        write_status(run, {**head, "status": "PASS", "passed": True, "date": ctx["date_iso"], "figure": doc["figure"],
                           "public_sums_sha256": sha256_bytes(public["SHA256SUMS"].encode("utf-8")),
                           "restricted_sums_sha256": sha256_bytes(restricted["SHA256SUMS"].encode("utf-8"))})
    except BaseException as e:
        reason = str(e) if isinstance(e, Failure) else f"unexpected error {type(e).__name__}"
        left = invalidate_outputs(run)
        try:
            write_status(run, {**head, "status": "FAIL", "passed": False, "reason": reason,
                               "outputs": "removed" if not left else f"left behind: {', '.join(left)}"})
        except Exception:  # noqa: BLE001
            print("ERROR: status.json could not be written", file=sys.stderr)
        raise
    for s in ("official_train", "official_test"):
        for ses, d in doc["sets_raw"][s].items():
            print(f"{s} session {ses}: n {d['n_bona_fide']}/{d['n_attack']}; class AUROC " + ", ".join(f"{n} {f4(d['class_auroc'][n])}" for n in RAW_STATS))
    print(f"figure: {doc['figure']}")
    print(f"run folder: {run.name} (public/ and restricted/)")
    print("RESULT: PASS (run)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="NUAA step zero: descriptive image statistics against the delivered scores (no training, no inference)")
    ap.add_argument("cmd", choices=["check-inputs", "run"])
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="configuration JSON (relative to the repository root)")
    ap.add_argument("--input-root", default=None, help="base folder of the input paths of the configuration (default: repository root)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="folder that receives <run-id>/ (default results/stepzero)")
    ap.add_argument("--date", default=None, help="YYYYMMDD of the run id (default: today)")
    ap.add_argument("--i-am-authorized", action="store_true", help="required by run: the step zero was authorised by the coordinator")
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
