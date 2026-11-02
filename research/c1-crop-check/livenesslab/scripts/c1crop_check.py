"""
C1 crop check: confronto appaiato di due estrattori di volto sullo stesso checkpoint AttackNet V2.1 (official NUAA, quello
di C1) e sulle stesse 300 immagini di C1. L'estrattore "cache_0ac7143" (face.py al commit 0ac7143) è quello con cui è
stata costruita la cache di addestramento; l'estrattore "c1_a80f6f9" (face.py al commit a80f6f9, lo stesso file dell'app
da allora) è quello della valutazione C1. Unica differenza numerica: il riquadro RetinaFace è 1 px più stretto e 1 px
più basso con l'estrattore C1. Pesi, immagini, catena (train_cnn.preprocess, invariato), normalizzazione (/255), CPU e
soglia 0,5 sono gli stessi per i due estrattori; nessun addestramento, nessuna taratura, nessuna inversione.

Sottocomandi, sempre con --config <JSON> --manifests <cartella dei manifest M2 NUAA> --images <radice degli originali>
--cache <nuaa_1500.npz> --weights <AttackNetV2_1__nuaa.h5> --c1-scores <scores.csv del run C1> --face-cache <face_0ac7143.py>
--face-c1 <face_a80f6f9.py> --study-dir <cartella dello studio>:
  check      nessun rilevatore, nessun TensorFlow, nessun pixel: configurazione (SHA-256 vincolato nel codice per i run non
             -trial), manifest (official_test, cnn_test, c1) per SHA-256, righe e conteggi, c1 dentro cnn_test riga per
             riga, ordine di campionamento della cache, originali dei 300 presenti con SHA-256 e dimensione uguali al
             manifest, schema e SHA-256 della cache, pesi per SHA-256 e dimensione, le due copie congelate di face.py per
             SHA-256 (e face.py attuale uguale alla copia C1), risorse del rilevatore e codice di riferimento per SHA-256,
             score C1 di riferimento (file per SHA-256; le 300 righe di attacknet_v2_1__nuaa legate al checkpoint, alla
             configurazione C1 e al manifest riga per riga). Crea la cartella dello studio (nuova) e scrive check.json.
  frames     i due rilevatori, nessun TensorFlow: per ogni immagine e per ciascun estrattore il rilevatore usato, il
             riquadro, l'hash del frame pre-elaborato; i frame dell'estrattore della cache confrontati bit per bit con le
             righe di X_test della cache (ogni differenza ferma lo studio: la catena della cache non si ricostruisce);
             frame uguali e diversi fra i due estrattori. Scrive frames.csv (riservato) e frames.json.
  parity     estrattore C1 e TensorFlow: i 300 score ricalcolati su CPU a batch 16 (lo scoring dello studio) entro la
             tolleranza dagli score C1 salvati e con le stesse decisioni (ogni violazione ferma lo studio: la catena di
             C1 non si ricostruisce); a titolo informativo anche la stessa riproduzione nelle condizioni di C1
             (dispositivo predefinito, un'immagine alla volta) e su CPU un'immagine alla volta. Scrive parity.csv
             (riservato) e parity.json, con la riga C1 di riferimento ricalcolata dagli score salvati.
  score      --i-am-authorized: i due estrattori sullo stesso checkpoint, CPU; score appaiati per immagine, decisioni,
             differenze; scores_paired.csv e per_subject.csv (riservati), metrics.json, score.json.
  aggregate  rilegge score, ricalcola metriche e statistiche appaiate, scrive aggregate.json, aggregate.md (pubblicabili,
             solo aggregati, scansione del contenuto vietato) e figure_paired.png (pubblicabile: un punto per immagine
             senza identificativi; i valori restano nel riservato scores_paired.csv).

Infrastruttura (scritture atomiche, log, record, metriche, scansione del pubblico) da scripts/n1_train.py e
scripts/n2_sessions.py, invariati. Comportamento su errore come n1_train.py (status.json RUNNING e PASS / FAIL, log in
coda, uscite scritte solo alla fine, uscite di frames, parity e score mai sovrascritte; aggregate toglie le proprie).
La cartella dello studio si chiama YYYYMMDD-C1CROP-<commit breve>[-trial]; dentro il repository di sviluppo è ammessa
solo sotto results/c1crop/. Un run non -trial richiede la configurazione consegnata (SHA-256 nel codice) e il commit
corrente nel nome.

Dipendenze: numpy (tutti i passi); scikit-learn e OpenCV (importati da src/tesi_app/evaluation.py per le metriche) per
parity, score e aggregate; OpenCV e il codice di riferimento per il rilevatore in frames, parity e score;
TensorFlow/Keras 2.15 per parity e score; matplotlib per la figura di aggregate.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np  # noqa: E402
    import n1_train as N1  # noqa: E402
    import n2_sessions as N2  # noqa: E402
    import nuaa_manifests as nm  # noqa: E402
    import path_guard as pg  # noqa: E402
except ModuleNotFoundError as _e:
    print(f"ERROR: missing Python package {_e.name!r}: numpy is needed by every subcommand", file=sys.stderr)
    print("RESULT: FAIL (import)")
    sys.exit(1)
from tesi_app.paths import LIVEDETECTION, load_module  # noqa: E402

fail = N1.fail
CropError = N1.N1Error
sha256_file, sha256_text, now, write_all = N1.sha256_file, N1.sha256_text, N1.now, N1.write_all
THRESHOLD, DECISION_RULE, NA = N1.THRESHOLD, N1.DECISION_RULE, N1.NA

CONFIG_SCHEMA = "c1-crop-check-config/1"
# SHA-256 della configurazione consegnata (delivery/c1crop/c1crop.json): un run non -trial accetta solo questa
CONFIG_SHA256 = "caa1156ac7608089790c12b337040cfc4f2443611be7735599ebe32b75d8e4ae"
RUN_ID_RE = re.compile(r"^(\d{8})-C1CROP-([0-9a-f]{7,40})(-trial)?$")
EXTRACTORS = ("cache_0ac7143", "c1_a80f6f9")          # ordine fisso: prima la cache di addestramento, poi C1
ARCH = "AttackNetV2_1"
SET = "c1_300"
MANIFESTS = ("official_test", "cnn_test", "c1")
COUNT_FIELDS = N2.COUNT_FIELDS
TARGET_SIZE = (256, 256)
CODE_FILES = ("scripts/c1crop_check.py", "scripts/n2_sessions.py", "scripts/n1_train.py", "scripts/train_cnn.py", "scripts/nuaa_manifests.py",
              "scripts/select_threshold.py", "scripts/path_guard.py", "scripts/m2_writers.py", "scripts/package_rules.py",
              "src/tesi_app/evaluation.py", "src/tesi_app/paths.py", "src/tesi_app/face.py")
REFERENCE_CODE = {"LIVENESSLAB_LIVEDETECTION/scripts/create_datasets.py": ("scripts", "create_datasets.py"),
                  "LIVENESSLAB_LIVEDETECTION/src/architectures.py": ("src", "architectures.py")}
C1_REQUIRED_COLUMNS = ("run_id", "sample_id", "dataset_id", "split", "subject_id", "label", "score_attack", "threshold", "prediction",
                       "checkpoint_sha256", "config_sha256", "analyzer_id")
CACHE_MEMBERS = ("X_train", "y_train", "s_train", "X_test", "y_test", "s_test")
FRAMES_OUTPUTS = ("frames.csv", "frames.json")
PARITY_OUTPUTS = ("parity.csv", "parity.json")
SCORE_OUTPUTS = ("scores_paired.csv", "per_subject.csv", "metrics.json", "score.json")
AGG_OUTPUTS = ("aggregate.json", "aggregate.md", "figure_paired.png")
FRAME_COLUMNS = ["run_id", "extractor", "sample_id", "subject_id", "session_id", "label", "detector", "confidence", "bbox_x", "bbox_y", "bbox_w",
                 "bbox_h", "file_sha256", "frame_sha256", "equal_to_cache"]
PARITY_COLUMNS = ["run_id", "sample_id", "label", "score_stored", "score_recomputed", "abs_diff", "prediction_stored", "prediction_recomputed",
                  "score_c1_conditions", "frame_sha256"]
SCORE_COLUMNS = ["run_id", "sample_id", "subject_id", "session_id", "label", "file_sha256", "detector_cache", "detector_c1", "frame_sha256_cache",
                 "frame_sha256_c1", "frame_equal", "score_cache", "score_c1", "abs_diff", "signed_diff", "prediction_cache", "prediction_c1",
                 "decision_change", "weights_sha256", "config_sha256"]
SUBJECT_COLUMNS = ["run_id", "extractor", "set", "subject_id", "n_bona_fide", "n_attack", "rejected_bona_fide", "accepted_attacks", "bpcer", "apcer", "acer"]
DIFF_BINS = ((0.0, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.0))
CHANGES = ("unchanged", "attack_to_bona_fide", "bona_fide_to_attack")   # decisione con l'estrattore C1 -> decisione con l'estrattore della cache

# punti di iniezione (test)
PREPROCESSOR = None      # (name, face_path, cfg) -> funzione(path) -> (frame uint8 256x256x3, nome breve del rilevatore, riquadro o None)
MAKE_SCORER = None       # (weights_path, weights_sha256) -> funzione(X uint8) -> probabilità N x 2 (CPU, batch 16: vincolante)
MAKE_SCORER_C1 = None    # (weights_path, weights_sha256) -> idem nelle condizioni di C1 (dispositivo predefinito, batch 1: informativo)
MAKE_SCORER_CPU1 = None  # (weights_path, weights_sha256) -> idem su CPU un'immagine alla volta (informativo: una replica CPU a batch 1)
PLOTTER = None           # (doc) -> {"figure_paired.png": byte PNG}
CURRENT_FACE = None      # percorso del face.py attuale dell'app; None: src/tesi_app/face.py


# ============================================================================ configurazione

def load_config(p) -> tuple:
    p = Path(p)
    if not p.is_file():
        fail(f"configuration not found: {p.name}")
    raw = p.read_bytes()
    try:
        cfg = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        fail(f"configuration {p.name} is not valid JSON ({e})")
    if not isinstance(cfg, dict):
        fail(f"configuration {p.name} is not a JSON object")
    if problems := config_problems(cfg):
        fail(f"configuration {p.name} refused: " + "; ".join(problems))
    return cfg, hashlib.sha256(raw).hexdigest()


def config_problems(cfg: dict) -> list:
    want = {"schema", "study", "architecture", "dataset", "protocol", "weights", "extractors", "difference", "chain", "detector_resources",
            "reference_code_sha256", "scoring", "metrics", "cache", "c1_reference", "data"}
    if set(cfg) != want:
        return [f"keys {sorted(set(cfg) ^ want)} missing or not allowed"]
    try:
        return _config_problems(cfg)
    except (AttributeError, TypeError, KeyError) as e:      # un blocco che non è un oggetto JSON: rifiuto pulito, mai un traceback
        return [f"malformed configuration block ({type(e).__name__}: {e})"]


def _config_problems(cfg: dict) -> list:
    out = []
    if cfg["schema"] != CONFIG_SCHEMA:
        out.append(f"schema {cfg['schema']!r} is not {CONFIG_SCHEMA!r}")
    if cfg["architecture"] != ARCH or cfg["dataset"] != "nuaa" or cfg["protocol"] != "official":
        out.append(f"architecture {ARCH}, dataset nuaa and protocol official are fixed")
    w = cfg["weights"]
    if not isinstance(w, dict) or not N2._hex(w.get("sha256")) or not N2._int(w.get("bytes")) or not isinstance(w.get("file"), str) \
            or w.get("classes") != {"0": "bona fide", "1": "attack"}:
        out.append("weights needs file, sha256, bytes and classes 0 bona fide / 1 attack")
    ex = cfg["extractors"]
    if not isinstance(ex, dict) or tuple(ex) != EXTRACTORS:
        out.append(f"extractors must be exactly {list(EXTRACTORS)} in this order")
    else:
        for n, e in ex.items():
            if not isinstance(e, dict) or not isinstance(e.get("file"), str) or not N2._hex(e.get("sha256")) or not re.fullmatch(r"[0-9a-f]{40}", str(e.get("commit", ""))):
                out.append(f"extractors.{n} needs file, commit (40 hex) and sha256")
        if not out and ex[EXTRACTORS[0]]["sha256"] == ex[EXTRACTORS[1]]["sha256"]:
            out.append("the two extractors must be different files")
    if not isinstance(cfg["detector_resources"], dict) or not all(N2._hex(x) for x in cfg["detector_resources"].values()):
        out.append("detector_resources must map file names to SHA-256 values")
    if set(cfg["reference_code_sha256"]) != set(REFERENCE_CODE) or not all(N2._hex(x) for x in cfg["reference_code_sha256"].values()):
        out.append("reference_code_sha256 must give the SHA-256 of create_datasets.py and architectures.py")
    sc = cfg["scoring"]
    if not isinstance(sc, dict) or sc.get("device") != "CPU" or sc.get("batch") != N1.SCORE_BATCH:
        out.append("scoring must be CPU with the N1 batch")
    m = cfg["metrics"]
    if not isinstance(m, dict) or m.get("threshold") != THRESHOLD or m.get("decision") != DECISION_RULE or m.get("set") != SET:
        out.append(f"metrics must fix threshold {THRESHOLD}, the decision rule and the set {SET}")
    c = cfg["cache"]
    if not isinstance(c, dict) or not isinstance(c.get("file"), str) or not N2._hex(c.get("sha256")) or not N2._int(c.get("per_class")) \
            or not isinstance(c.get("members"), dict) or set(c["members"]) != set(CACHE_MEMBERS) \
            or c["members"]["X_test"] != {"dtype": "|u1", "shape": [c["per_class"] * 2] + list(TARGET_SIZE) + [3]} or "rule" not in c:
        out.append("cache needs file, sha256, per_class, the six members with X_test uint8 [2*per_class, 256, 256, 3] and the rule")
    r = cfg["c1_reference"]
    par = r.get("parity") if isinstance(r, dict) else None
    if not isinstance(r, dict) or not isinstance(r.get("run_id"), str) or not isinstance(r.get("file"), str) or not N2._hex(r.get("sha256")) \
            or r.get("analyzer_id") != "attacknet_v2_1__nuaa" or not N2._int(r.get("rows")) or not N2._hex(r.get("config_sha256")) \
            or not isinstance(r.get("columns"), list) or not set(C1_REQUIRED_COLUMNS) <= set(r["columns"]) or not isinstance(par, dict) \
            or not isinstance(par.get("tolerance_abs"), float) or not (0 < par["tolerance_abs"] <= 1e-5) or "rule" not in par:
        out.append("c1_reference needs run_id, file, sha256, analyzer_id attacknet_v2_1__nuaa, rows, config_sha256, columns (including "
                   + ", ".join(C1_REQUIRED_COLUMNS) + ") and parity (tolerance_abs in (0, 1e-5], rule)")
    d = cfg["data"]
    if not isinstance(d, dict) or set(d) != {"m2_run", "manifests", "originals"} or set(d.get("manifests", {})) != set(MANIFESTS):
        out.append("data must hold m2_run, originals and the three manifests")
    else:
        for n, e in d["manifests"].items():
            if not isinstance(e, dict) or not isinstance(e.get("file"), str) or not N2._hex(e.get("sha256")) or not all(N2._int(e.get(f)) for f in COUNT_FIELDS):
                out.append(f"data.manifests.{n}: file, sha256 and integer counts expected")
        if not out and (d["manifests"]["c1"]["rows"] != cfg["c1_reference"]["rows"] or d["manifests"]["cnn_test"]["rows"] != cfg["cache"]["per_class"] * 2):
            out.append("c1 rows must equal c1_reference.rows and cnn_test rows must equal 2 * cache.per_class")
    return out


def run_problems(cfg_sha: str, trial: bool) -> list:
    if trial:
        return []
    if cfg_sha != CONFIG_SHA256:
        return [f"non-trial run: the configuration SHA-256 {cfg_sha[:12]}… is not the delivered one ({CONFIG_SHA256[:12]}…)"]
    return []


# ============================================================================ cartelle e ingressi

def study_dir_problem(d) -> str:
    d = Path(d)
    if not RUN_ID_RE.match(d.name):
        return f"study folder name {d.name!r} is not YYYYMMDD-C1CROP-<shortcommit>[-trial]"
    if d.is_symlink():
        return f"{d.name} is a symbolic link: the study folder must be a real folder"
    if d.exists() and not d.is_dir():
        return f"{d.name} exists and is not a folder"
    for g in N1.outer_roots():
        if pg.within(d, g) or pg.contains(d, g):
            return f"{d} is inside, or contains, the work tree {g.name} with the accepted packages: outputs go outside it"
    if msg := pg.deletion_problem(d, ROOT, pg.BASE_PROTECTED, [ROOT / "results" / "c1crop"]):
        return msg
    for q in N2.protected_extra() + [ROOT / "results" / "n2"]:
        if pg.within(d, q) or (Path(q).exists() and pg.contains(d, q)):
            return f"{d} is, contains or is inside a protected folder ({q.name})"
    return ""


INPUT_OPTIONS = (("--config", "config", "file"), ("--manifests", "manifests", "dir"), ("--images", "images", "dir"), ("--cache", "cache", "file"),
                 ("--weights", "weights", "file"), ("--c1-scores", "c1_scores", "file"), ("--face-cache", "face_cache", "file"),
                 ("--face-c1", "face_c1", "file"))


def inputs_problem(a, d) -> str:
    for opt, attr, kind in INPUT_OPTIONS:
        p = Path(getattr(a, attr))
        if pg.within(p, d) or pg.contains(p, d):
            return f"{opt} {p.name} is, contains or is inside the study folder: inputs and outputs must be separate"
        if kind == "dir" and (p.is_symlink() or not p.is_dir()):
            return f"{opt} {p.name} is a symbolic link or not a folder"
        if kind == "file" and (p.is_symlink() or not p.is_file()):
            return f"{opt} {p.name} not found or not a regular file"
    return ""


def is_trial(d) -> bool:
    return Path(d).name.endswith("-trial")


def face_paths(a) -> dict:
    return {EXTRACTORS[0]: Path(a.face_cache), EXTRACTORS[1]: Path(a.face_c1)}


def code_hashes(a) -> dict:
    out = {f: sha256_file(ROOT / f) for f in CODE_FILES}
    for key, rel in REFERENCE_CODE.items():
        p = Path(LIVEDETECTION).joinpath(*rel)
        out[key] = sha256_file(p) if p.is_file() else "absent"
    for n, p in face_paths(a).items():
        out[f"extractor:{n}"] = sha256_file(p)
    return out


def frozen_code_problems(cfg: dict, a) -> list:
    """Pesi, copie congelate di face.py (e face.py attuale uguale alla copia C1), risorse del rilevatore e codice di
    riferimento: SHA-256 uguali alla configurazione."""
    out = []
    wp = Path(a.weights)
    if wp.name != cfg["weights"]["file"]:
        out.append(f"--weights {wp.name} is not {cfg['weights']['file']}")
    elif wp.stat().st_size != cfg["weights"]["bytes"]:
        out.append(f"weights file size {wp.stat().st_size} differs from the configuration ({cfg['weights']['bytes']})")
    elif sha256_file(wp) != cfg["weights"]["sha256"]:
        out.append("weights file: SHA-256 differs from the configuration")
    for n, p in face_paths(a).items():
        e = cfg["extractors"][n]
        if p.name != e["file"]:
            out.append(f"extractor {n}: {p.name} is not {e['file']}")
        elif sha256_file(p) != e["sha256"]:
            out.append(f"extractor {n} ({p.name}): SHA-256 differs from the configuration")
    cur = Path(CURRENT_FACE) if CURRENT_FACE else ROOT / "src" / "tesi_app" / "face.py"
    if not cur.is_file() or sha256_file(cur) != cfg["extractors"][EXTRACTORS[1]]["sha256"]:
        out.append("the current application face.py is not the C1 extractor copy (SHA-256 differs from the configuration)")
    paths = N2.resource_paths()
    for n, want in cfg["detector_resources"].items():
        p = paths.get(n)
        if p is None or not Path(p).is_file():
            out.append(f"detector resource {n} not found")
        elif sha256_file(p) != want:
            out.append(f"detector resource {n}: SHA-256 differs from the configuration")
    for key, rel in REFERENCE_CODE.items():
        p = Path(LIVEDETECTION).joinpath(*rel)
        if not p.is_file():
            out.append(f"reference code {rel[-1]} not found under LIVENESSLAB_LIVEDETECTION")
        elif sha256_file(p) != cfg["reference_code_sha256"][key]:
            out.append(f"reference code {rel[-1]}: SHA-256 differs from the configuration")
    return out


class Inputs:
    pass


def load_manifests(cfg: dict, a, inp: Inputs) -> None:
    d = cfg["data"]
    inp.rows, inp.sha = {}, {}
    for name in MANIFESTS:
        e = d["manifests"][name]
        p = Path(a.manifests) / e["file"]
        if p.is_symlink() or not p.is_file():
            fail(f"manifest {e['file']} ({name}) not found or not a regular file")
        got = sha256_file(p)
        if got != e["sha256"]:
            fail(f"manifest {e['file']} ({name}): SHA-256 {got} differs from the configuration ({e['sha256']})")
        rows = N1.guarded(e["file"], nm.load_manifest, p)
        c = N1.counts_of(rows)
        for f in COUNT_FIELDS:
            if c[f] != e[f]:
                fail(f"manifest {e['file']} ({name}): {f} {c[f]} differs from the configuration ({e[f]})")
        for dim in ("key", "sha256"):
            if len({r[dim] for r in rows}) != len(rows):
                fail(f"manifest {e['file']} ({name}): duplicated {dim} values")
        if any(r["official_split"] != "test" for r in rows):
            fail(f"manifest {name}: rows outside the official test split")
        inp.rows[name], inp.sha[name] = rows, got
    inp.by_key = {n: {r["key"]: r for r in rows} for n, rows in inp.rows.items()}
    for sub, sup in (("cnn_test", "official_test"), ("c1", "cnn_test")):
        for k, r in inp.by_key[sub].items():
            if k not in inp.by_key[sup]:
                fail(f"manifest {sub}: {sum(1 for x in inp.by_key[sub] if x not in inp.by_key[sup])} keys are not in {sup}")
            if inp.by_key[sup][k] != r:
                fail(f"manifest {sub}: rows with the same key as {sup} but different fields")
    inp.checks["manifests"] = {n: {"file": d["manifests"][n]["file"], "sha256": inp.sha[n], **N1.counts_of(inp.rows[n])} for n in MANIFESTS}
    k = cfg["cache"]["per_class"]
    inp.order = nm.spread_keys(inp.rows["official_test"], k)
    if inp.order != nm.spread_keys(inp.rows["cnn_test"], k):
        fail("the sampling order of the official test list does not give cnn_test")
    inp.pos = {key: i for i, key in enumerate(inp.order)}
    inp.c1 = sorted(inp.rows["c1"], key=lambda r: r["key"])
    inp.c1_keys = [r["key"] for r in inp.c1]
    inp.checks["sampling_order"] = "spread per class (bona fide first) on the official test list: equal to cnn_test; X_test rows in this order"


def load_cache_schema(cfg: dict, a, inp: Inputs) -> None:
    c = cfg["cache"]
    cp = Path(a.cache)
    if cp.name != c["file"]:
        fail(f"cache {cp.name} is not the file of the configuration ({c['file']})")
    got = sha256_file(cp)
    if got != c["sha256"]:
        fail(f"cache {cp.name}: SHA-256 {got} differs from the configuration ({c['sha256']})")
    mem = N1.npz_members(cp)
    if set(mem) != set(CACHE_MEMBERS):
        fail(f"cache {cp.name}: members {sorted(mem)} differ from {sorted(CACHE_MEMBERS)}")
    for k, want in c["members"].items():
        if {"dtype": mem[k]["dtype"], "shape": mem[k]["shape"]} != want or mem[k]["fortran"]:
            fail(f"cache {cp.name}: {k} has dtype {mem[k]['dtype']}, shape {mem[k]['shape']}, expected {want['dtype']}, {want['shape']}")
    inp.checks["cache"] = {"file": c["file"], "sha256": got, "X_test": c["members"]["X_test"]}


def load_c1_reference(cfg: dict, a, inp: Inputs) -> None:
    """Gli score C1 salvati (scores.csv del run C1, schema del gruppo): file per SHA-256; le righe dell'analizzatore
    legate al run, alla configurazione C1, al checkpoint e al manifest c1 riga per riga (chiave, soggetto, etichetta;
    la colonna session_id di C1 segue il manifest C1 del 2026-09, cioè il terzo campo del nome, e non viene confrontata)."""
    r = cfg["c1_reference"]
    p = Path(a.c1_scores)
    if p.name != r["file"]:
        fail(f"--c1-scores {p.name} is not {r['file']}")
    got = sha256_file(p)
    if got != r["sha256"]:
        fail(f"C1 scores {p.name}: SHA-256 {got} differs from the configuration ({r['sha256']})")
    rows = [x for x in N1.read_csv_rows(p, r["columns"]) if x["analyzer_id"] == r["analyzer_id"]]
    if len(rows) != r["rows"]:
        fail(f"C1 scores: {len(rows)} rows of {r['analyzer_id']} instead of {r['rows']}")
    keys = [x["sample_id"] for x in rows]
    if sorted(keys) != inp.c1_keys:
        fail(f"C1 scores: the {r['analyzer_id']} rows do not hold exactly the c1 manifest keys")
    inp.c1_stored = {}
    for x in rows:
        m = inp.by_key["c1"][x["sample_id"]]
        try:
            v = float(x["score_attack"])
        except ValueError:
            v = float("nan")
        if x["run_id"] != r["run_id"] or x["dataset_id"] != "nuaa" or x["split"] != "test" or x["subject_id"] != m["subject"] \
                or x["label"] != m["label"] or x["threshold"] != repr(THRESHOLD) \
                or x["checkpoint_sha256"] != cfg["weights"]["sha256"] or x["config_sha256"] != r["config_sha256"] \
                or not (0 <= v <= 1) or x["prediction"] != ("1" if v > THRESHOLD else "0"):
            fail("C1 scores: rows inconsistent with the C1 run, the C1 configuration, the checkpoint, the c1 manifest or the decision rule")
        inp.c1_stored[x["sample_id"]] = v
    inp.checks["c1_reference"] = {"run_id": r["run_id"], "file": r["file"], "sha256": got, "analyzer_id": r["analyzer_id"], "rows": len(rows),
                                  "checkpoint_sha256": cfg["weights"]["sha256"], "config_sha256": r["config_sha256"],
                                  "linkage": "every row bound to the run, the C1 configuration, the checkpoint and its c1 manifest row (key, subject, label); "
                                             "session_id not compared (C1 manifest rule of 2026-09: third field of the file name)"}


def load_inputs(cfg: dict, a, originals: bool = True) -> Inputs:
    inp = Inputs()
    inp.checks = {}
    load_manifests(cfg, a, inp)
    if originals:
        layouts = Counter()
        for r in inp.c1:
            N2.locate(Path(a.images), r, layouts)
        inp.checks["originals"] = {"n_verified": len(inp.c1), "by_layout": dict(sorted(layouts.items())),
                                   "rule": "file regular, size and SHA-256 equal to the manifest row"}
    load_cache_schema(cfg, a, inp)
    if problems := frozen_code_problems(cfg, a):
        fail("frozen code check failed: " + "; ".join(problems))
    inp.checks["frozen_code"] = {"weights": {k: cfg["weights"][k] for k in ("file", "sha256", "bytes")},
                                 "extractors": {n: {k: cfg["extractors"][n][k] for k in ("file", "commit", "sha256")} for n in EXTRACTORS},
                                 "current_face_py": cfg["extractors"][EXTRACTORS[1]]["sha256"], "detector_resources": cfg["detector_resources"],
                                 "reference_code_sha256": cfg["reference_code_sha256"]}
    load_c1_reference(cfg, a, inp)
    return inp


def same_inputs(inp: Inputs, check: dict) -> None:
    """Gli ingressi rivalidati contro la configurazione coincidono con quelli registrati al check. Dopo frames il
    controllo è ridondante (ogni record successivo è legato al check.json per SHA-256), ma resta per il passo frames."""
    for k in ("manifests", "cache", "frozen_code", "c1_reference"):
        if inp.checks.get(k) != check["checks"].get(k):
            fail(f"{k} differ from those recorded at check")


def parse_common(a, need_study: bool = True):
    d = Path(a.study_dir)
    if msg := study_dir_problem(d):
        fail(f"study folder refused: {msg}")
    if msg := inputs_problem(a, d):
        fail(f"inputs refused: {msg}")
    cfg, cfg_sha = load_config(a.config)
    if msg := "; ".join(run_problems(cfg_sha, is_trial(d))):
        fail(msg)
    if need_study and not d.is_dir():
        fail("study folder does not exist: run check first")
    return d, cfg, cfg_sha


def open_study(a) -> tuple:
    d, cfg, cfg_sha = parse_common(a)
    check = N1.load_record(d / "check.json", "check", d.name, cfg_sha)
    if check.get("code_sha256") != code_hashes(a):
        fail("the code changed since check (SHA-256 of the scripts or the frozen copies differ from check.json): start a new study")
    if not is_trial(d) and N1.repo_state(d).get("commit") != check.get("commit"):
        fail("non-trial run: the repository commit differs from check")
    return d, cfg, cfg_sha, check, sha256_file(d / "check.json")


# ============================================================================ pre-elaborazione e scorer

def load_extractor(name: str, path, sha: str):
    """La copia congelata di face.py come modulo tesi_app.face_<nome> (gli import relativi del file funzionano),
    SHA-256 controllato subito prima del caricamento."""
    p = Path(path)
    if sha256_file(p) != sha:
        fail(f"extractor {name} ({p.name}): SHA-256 changed just before loading")
    mod_name = f"tesi_app.face_{name}"
    mod = sys.modules.get(mod_name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(mod_name, str(p))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    return mod.FaceDetector()


def default_preprocessor(name: str, face_path, cfg: dict):
    """train_cnn.preprocess (invariato) con l'estrattore congelato e l'enhancement del codice di riferimento."""
    try:
        import train_cnn as tc
    except ModuleNotFoundError as e:
        fail(f"missing Python package {e.name!r}: the preprocessing needs OpenCV")
    det = N2.RecordingDetector(load_extractor(name, face_path, cfg["extractors"][name]["sha256"]))
    cd = load_module("kuznetsov_create_datasets", Path(LIVEDETECTION) / "scripts" / "create_datasets.py")
    creator = cd.EnhancedDatasetCreator.__new__(cd.EnhancedDatasetCreator)   # istanza vuota: serve solo advanced_image_enhancement

    def pre(path):
        det.last = None
        arr = tc.preprocess(Path(path), det, creator)
        if arr is None:
            fail(f"{Path(path).name}: unreadable or empty crop")
        f = det.last
        box = None if f is None else {"x": int(f.x), "y": int(f.y), "w": int(f.w), "h": int(f.h), "confidence": float(f.confidence)}
        return arr, N2.detector_short(f), box
    return pre


def preprocessors(a, cfg: dict) -> dict:
    make = PREPROCESSOR or default_preprocessor
    return {n: make(n, p, cfg) for n, p in face_paths(a).items()}


def preprocess_all(pre, inp: Inputs, images) -> tuple:
    """I 300 frame (ordine di chiave) con un estrattore: array, nomi brevi dei rilevatori, riquadri, hash dei frame."""
    frames, dets, boxes = [], [], []
    for r in inp.c1:
        arr, det, box = pre(N2.locate(Path(images), r))
        if arr.shape != TARGET_SIZE + (3,) or arr.dtype != np.uint8:
            fail(f"{r['key']}: preprocessed frame has shape {arr.shape} {arr.dtype}, expected {TARGET_SIZE + (3,)} uint8")
        frames.append(arr)
        dets.append(det)
        boxes.append(box)
    frames = np.stack(frames)
    return frames, dets, boxes, nm.frame_hashes(frames)


def tf_scorer(weights_path, weights_sha: str, device: str = "/CPU:0", batch: int = N1.SCORE_BATCH):
    """Scorer vero: architettura AttackNetV2_1 del codice di riferimento, pesi verificati per SHA-256, remapping di
    Grappler disattivato come nell'app, ingresso frame / 255, BatchNorm in inferenza; probabilità softmax N x 2.
    Vincolante: CPU (GPU nascosta) a batch 16 (lo scoring dello studio). Informativo: `device` None e batch 1, cioè il
    dispositivo predefinito del processo un'immagine alla volta, le condizioni in cui l'app ha prodotto gli score C1
    (su questo Mac la GPU Metal; le AttackNet non sono toccate dal difetto Metal della ReLU, che riguarda LivenessNet)."""
    try:
        import tensorflow as tf
    except ModuleNotFoundError as e:
        fail(f"missing Python package {e.name!r}: parity and score need TensorFlow")
    if device == "/CPU:0":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            tf.config.set_visible_devices([], "GPU")     # CPU anche con tensorflow-metal (dopo il primo calcolo non è più modificabile: resta tf.device)
        except (RuntimeError, ValueError):
            pass
    tf.config.optimizer.set_experimental_options({"remapping": False})
    if sha256_file(weights_path) != weights_sha:
        fail(f"{Path(weights_path).name}: SHA-256 changed just before loading")
    arch = Path(LIVEDETECTION) / "src" / "architectures.py"
    if not arch.is_file():
        fail("the reference architectures (LIVENESSLAB_LIVEDETECTION/src/architectures.py) are missing")
    mod = load_module("kuznetsov_architectures", arch)
    import contextlib
    place = (lambda: tf.device(device)) if device else contextlib.nullcontext
    with place():
        model = mod.create_model(ARCH).get_model()
        model.load_weights(str(weights_path))

    def score(X):
        out = []
        for i in range(0, len(X), batch):
            xb = X[i:i + batch].astype(np.float32) / 255.0
            with place():
                out.append(np.asarray(model(xb, training=False), dtype=np.float64))
        return np.concatenate(out) if out else np.zeros((0, 2))
    return score


def default_device() -> str:
    """Tipo del dispositivo predefinito di TensorFlow (per il record della riproduzione informativa)."""
    try:
        import tensorflow as tf
        kinds = {d.device_type for d in tf.config.list_logical_devices()}
    except Exception:  # noqa: BLE001
        return "unknown"
    return "GPU" if "GPU" in kinds else "CPU"


def attack_scores(cfg: dict, a, frames) -> list:
    p = N1.probabilities((MAKE_SCORER or tf_scorer)(a.weights, cfg["weights"]["sha256"]), frames)[:, 1]
    return [float(v) for v in p]


def prediction(v: float) -> str:
    return "1" if v > THRESHOLD else "0"


def change_of(pred_c1: str, pred_cache: str) -> str:
    if pred_c1 == pred_cache:
        return CHANGES[0]
    return CHANGES[1] if pred_c1 == "1" else CHANGES[2]


# ============================================================================ check

def cmd_check(a) -> None:
    d, cfg, cfg_sha = parse_common(a, need_study=False)
    if d.exists() and any(d.iterdir()):
        fail("the study folder must be new (missing or empty): refused, nothing written")
    state = N1.repo_state(d)
    m = RUN_ID_RE.match(d.name)
    if not is_trial(d) and not str(state.get("commit", "")).startswith(m.group(2)):
        fail(f"non-trial run: the study id must carry the current commit ({str(state.get('commit'))[:7]})")
    if not is_trial(d) and state.get("dirty"):
        fail(f"non-trial run: the work tree is not clean ({len(state.get('dirty_paths', []))} modified or unversioned paths): commit or stash first")
    d.mkdir(parents=True, exist_ok=True)
    N1.accept(d, "check", "check.log")
    print(f"== C1CROP check: study {d.name}, configuration SHA-256 {cfg_sha}")
    inp = load_inputs(cfg, a)
    for k, v in inp.checks.items():
        print(f"   PASS {k}: {json.dumps(v, sort_keys=True) if not isinstance(v, str) else v}")
    doc = {"passed": True, "step": "check", "run_id": d.name, "trial": is_trial(d), "config_sha256": cfg_sha,
           "delivered_configuration": cfg_sha == CONFIG_SHA256, "checks": inp.checks, "detector": "not used", "weights": "not loaded",
           "tensorflow": N2.tf_state(), "code_sha256": code_hashes(a), "environment": N2.environment(), **state, "created": now()}
    write_all(d, {"check.json": json.dumps(doc, indent=1) + "\n"})
    print(f"== check PASS: 3 manifests, {len(inp.c1)} originals verified, cache schema, weights, the two extractor copies, detector resources, "
          f"reference code and the {len(inp.c1_stored)} stored C1 scores bound")


# ============================================================================ frames

def cmd_frames(a) -> None:
    d, cfg, cfg_sha, check, check_sha = open_study(a)
    N2.refuse_existing(d, FRAMES_OUTPUTS, "frames")
    N1.accept(d, "frames", "frames.log")
    print(f"== C1CROP frames: study {d.name}, two extractors, no TensorFlow")
    inp = load_inputs(cfg, a, originals=False)
    same_inputs(inp, check)
    pres = preprocessors(a, cfg)
    z = np.load(a.cache, allow_pickle=False)
    X = z["X_test"]                      # schema e SHA-256 della cache verificati al check e ricontrollati da load_inputs
    cache_rows = np.stack([X[inp.pos[k]] for k in inp.c1_keys])
    del X
    cache_hashes = nm.frame_hashes(cache_rows)
    per, rows = {}, []
    for n in EXTRACTORS:
        frames, dets, boxes, hashes = preprocess_all(pres[n], inp, a.images)
        equal = [h == c for h, c in zip(hashes, cache_hashes)]
        worst = int(np.abs(frames.astype(np.int16) - cache_rows.astype(np.int16)).reshape(len(frames), -1).max(axis=1).max()) if not all(equal) else 0
        per[n] = {"frames": frames, "dets": dets, "boxes": boxes, "hashes": hashes, "equal": equal, "worst_vs_cache": worst}
        for r, det, box, h, e in zip(inp.c1, dets, boxes, hashes, equal):
            b = box or {}
            rows.append({"run_id": d.name, "extractor": n, "sample_id": r["key"], "subject_id": r["subject"], "session_id": r["session"], "label": r["label"],
                         "detector": det, "confidence": "" if box is None else repr(float(b["confidence"])),
                         "bbox_x": "" if box is None else str(b["x"]), "bbox_y": "" if box is None else str(b["y"]),
                         "bbox_w": "" if box is None else str(b["w"]), "bbox_h": "" if box is None else str(b["h"]),
                         "file_sha256": r["sha256"], "frame_sha256": h, "equal_to_cache": "1" if e else "0"})
        print(f"   {n}: detectors {dict(sorted(Counter(dets).items()))}, frames equal to the cache rows {sum(equal)}/{len(equal)}")
    c0, c1 = per[EXTRACTORS[0]], per[EXTRACTORS[1]]
    if not all(c0["equal"]):
        bad = [k for k, e in zip(inp.c1_keys, c0["equal"]) if not e]
        fail(f"the cache chain is not reconstructed: {len(bad)} of {len(inp.c1_keys)} frames preprocessed with {EXTRACTORS[0]} differ from their "
             f"X_test rows (max abs difference {c0['worst_vs_cache']}); study stopped before any score")
    if len(set(c0["hashes"])) != len(c0["hashes"]) or len(set(c1["hashes"])) != len(c1["hashes"]):
        fail("duplicated derived frames within an extractor (exact pixel hash)")
    pair_equal = [h0 == h1 for h0, h1 in zip(c0["hashes"], c1["hashes"])]
    diff_px = np.abs(c0["frames"].astype(np.int16) - c1["frames"].astype(np.int16)).reshape(len(pair_equal), -1)
    pairs = Counter(f"{a0}/{a1}" for a0, a1 in zip(c0["dets"], c1["dets"]))
    box_delta = Counter()
    for b0, b1 in zip(c0["boxes"], c1["boxes"]):
        if b0 is None or b1 is None:
            box_delta["no box in one or both"] += 1
        else:
            box_delta[f"dx {b1['x'] - b0['x']}, dy {b1['y'] - b0['y']}, dw {b1['w'] - b0['w']}, dh {b1['h'] - b0['h']}"] += 1
    summary = {"n": len(pair_equal), "detectors": {n: dict(sorted(Counter(per[n]["dets"]).items())) for n in EXTRACTORS},
               "detector_pairs": dict(sorted(pairs.items())), "equal_to_cache": {n: sum(per[n]["equal"]) for n in EXTRACTORS},
               "frames_equal_between_extractors": sum(pair_equal), "frames_different_between_extractors": len(pair_equal) - sum(pair_equal),
               "max_abs_pixel_difference": int(diff_px.max()) if len(diff_px) else 0,
               "mean_abs_pixel_difference_over_different_frames": float(np.mean([diff_px[i].mean() for i, e in enumerate(pair_equal) if not e])) if not all(pair_equal) else 0.0,
               "box_delta_c1_minus_cache": dict(sorted(box_delta.items())),
               "rule": cfg["cache"]["rule"]}
    frames_txt = N1.csv_text(rows, FRAME_COLUMNS)
    doc = {"passed": True, "step": "frames", "run_id": d.name, "config_sha256": cfg_sha, "check_sha256": check_sha, "summary": summary,
           "extractors": {n: {k: cfg["extractors"][n][k] for k in ("file", "commit", "sha256")} for n in EXTRACTORS},
           "tensorflow": N2.tf_state(), "environment": N2.environment(), "files": {"frames.csv": sha256_text(frames_txt)}, "created": now()}
    write_all(d, {"frames.csv": frames_txt, "frames.json": json.dumps(doc, indent=1) + "\n"})
    print(f"== frames PASS: {summary['n']} images, cache chain reconstructed bit for bit with {EXTRACTORS[0]}, "
          f"{summary['frames_different_between_extractors']} frames differ between the extractors, detector pairs {summary['detector_pairs']}")


# ============================================================================ parity

def expected_frames(d: Path, inp: Inputs) -> dict:
    rows = N1.read_csv_rows(d / "frames.csv", FRAME_COLUMNS)
    out = {n: {} for n in EXTRACTORS}
    for r in rows:
        if r["extractor"] not in out:
            fail("frames.csv holds an unknown extractor")
        out[r["extractor"]][r["sample_id"]] = r
    for n in EXTRACTORS:
        if sorted(out[n]) != inp.c1_keys:
            fail(f"frames.csv does not hold exactly the c1 keys for {n}")
    return out


def recompute_checked(pre, inp: Inputs, images, expected: dict, name: str):
    frames, dets, boxes, hashes = preprocess_all(pre, inp, images)
    for k, det, h in zip(inp.c1_keys, dets, hashes):
        if h != expected[k]["frame_sha256"] or det != expected[k]["detector"]:
            fail(f"{name}: preprocessed frame or detector of an image differ from frames.csv")
    return frames, dets, hashes


def cmd_parity(a) -> None:
    d, cfg, cfg_sha, check, check_sha = open_study(a)
    N2.refuse_existing(d, PARITY_OUTPUTS, "parity")
    N2.load_step(d, "frames.json", "frames", d.name, cfg_sha, check_sha)
    frames_sha = sha256_file(d / "frames.json")
    tol = cfg["c1_reference"]["parity"]["tolerance_abs"]
    N1.accept(d, "parity", "parity.log")
    print(f"== C1CROP parity: study {d.name}, extractor {EXTRACTORS[1]} against the stored C1 scores, tolerance {tol}, CPU")
    inp = load_inputs(cfg, a, originals=False)
    same_inputs(inp, check)
    expected = expected_frames(d, inp)
    pre = preprocessors(a, cfg)[EXTRACTORS[1]]
    frames, dets, hashes = recompute_checked(pre, inp, a.images, expected[EXTRACTORS[1]], EXTRACTORS[1])
    stored_v = [inp.c1_stored[r["key"]] for r in inp.c1]
    # informativo, prima del vincolante (il dispositivo predefinito va usato prima di nascondere la GPU): le condizioni di C1
    c1cond = N1.probabilities((MAKE_SCORER_C1 or (lambda w, h: tf_scorer(w, h, device=None, batch=1)))(a.weights, cfg["weights"]["sha256"]), frames)[:, 1]
    cdiff = [abs(float(v) - s0) for v, s0 in zip(c1cond, stored_v)]
    c1cond_doc = {"device": default_device(), "batch": 1, "max_abs_diff": max(cdiff), "within_1e-6": sum(1 for x in cdiff if x <= 1e-6),
                  "within_1e-7": sum(1 for x in cdiff if x <= 1e-7), "same_decision": sum(prediction(float(v)) == prediction(s0) for v, s0 in zip(c1cond, stored_v)),
                  "note": "informative only: the default device of this process, one image at a time, as the application computed the C1 scores; not the scoring of the study"}
    print(f"   C1 conditions ({c1cond_doc['device']}, batch 1): max |recomputed - stored| {max(cdiff):.3e}, within 1e-6 {c1cond_doc['within_1e-6']}, "
          f"same decision {c1cond_doc['same_decision']}")
    cpu1 = N1.probabilities((MAKE_SCORER_CPU1 or (lambda w, h: tf_scorer(w, h, batch=1)))(a.weights, cfg["weights"]["sha256"]), frames)[:, 1]
    c1diff = [abs(float(v) - s0) for v, s0 in zip(cpu1, stored_v)]
    cpu1_doc = {"device": "CPU", "batch": 1, "max_abs_diff": max(c1diff), "within_1e-5": sum(1 for x in c1diff if x <= 1e-5),
                "within_1e-6": sum(1 for x in c1diff if x <= 1e-6), "same_decision": sum(prediction(float(v)) == prediction(s0) for v, s0 in zip(cpu1, stored_v)),
                "note": "informative only: the CPU one image at a time (a CPU replication of the stored scores one image at a time, as an R2-like run would do); not the scoring of the study"}
    print(f"   CPU, batch 1 (informative): max |recomputed - stored| {max(c1diff):.3e}, within 1e-5 {cpu1_doc['within_1e-5']}, same decision {cpu1_doc['same_decision']}")
    scores = attack_scores(cfg, a, frames)
    rows, diffs, same_dec = [], [], 0
    for r, h, v, s0, vc in zip(inp.c1, hashes, scores, stored_v, c1cond):
        diffs.append(abs(v - s0))
        same_dec += prediction(v) == prediction(s0)
        rows.append({"run_id": d.name, "sample_id": r["key"], "label": r["label"], "score_stored": repr(s0), "score_recomputed": repr(v),
                     "abs_diff": repr(diffs[-1]), "prediction_stored": prediction(s0), "prediction_recomputed": prediction(v),
                     "score_c1_conditions": repr(float(vc)), "frame_sha256": h})
    within = sum(1 for x in diffs if x <= tol)
    print(f"   CPU, batch {N1.SCORE_BATCH}: max |recomputed - stored| {max(diffs):.3e} over {len(diffs)} images, within tolerance {within}, same decision {same_dec}")
    if within != len(diffs) or same_dec != len(diffs):
        fail(f"the C1 chain is not reconstructed: {len(diffs) - within} scores outside the tolerance {tol} and {len(diffs) - same_dec} decisions differ "
             f"from the stored C1 scores (max abs difference {max(diffs):.3e}); study stopped before any comparison")
    ref = N1.set_metrics([int(r["label"]) for r in inp.c1], stored_v)
    par_txt = N1.csv_text(rows, PARITY_COLUMNS)
    doc = {"passed": True, "step": "parity", "run_id": d.name, "config_sha256": cfg_sha, "check_sha256": check_sha, "frames_sha256": frames_sha,
           "rule": cfg["c1_reference"]["parity"]["rule"], "tolerance_abs": tol, "n": len(diffs), "max_abs_diff": max(diffs),
           "median_abs_diff": float(np.median(diffs)), "within_tolerance": within, "within_1e-6": sum(1 for x in diffs if x <= 1e-6),
           "within_1e-7": sum(1 for x in diffs if x <= 1e-7), "same_decision": same_dec, "device": "CPU", "batch": N1.SCORE_BATCH,
           "c1_conditions": c1cond_doc, "cpu_batch1": cpu1_doc, "stored_device": cfg["c1_reference"]["device"], "c1_reference_metrics": ref,
           "detectors": dict(sorted(Counter(dets).items())), "weights_sha256": cfg["weights"]["sha256"], "environment": N2.environment(),
           "files": {"parity.csv": sha256_text(par_txt)}, "created": now()}
    write_all(d, {"parity.csv": par_txt, "parity.json": json.dumps(doc, indent=1) + "\n"})
    print(f"== parity PASS: {len(diffs)} scores within {tol} of the stored C1 scores, same decisions; C1 reference row ACER {N1.fmt_na(ref['acer'])}, "
          f"AUROC {N1.fmt_na(ref['auroc'])}")


# ============================================================================ score

def paired_stats(rows) -> dict:
    """Statistiche appaiate dalle righe di scores_paired.csv (stesso calcolo in score e in aggregate)."""
    ad = [float(r["abs_diff"]) for r in rows]
    sd = [float(r["signed_diff"]) for r in rows]
    bins = {"zero": sum(1 for x in ad if x == 0)}
    for lo, hi in DIFF_BINS:
        bins[f"({lo:g}, {hi:g}]"] = sum(1 for x in ad if lo < x <= hi)
    changes = {c: {"all": 0, "bona_fide": 0, "attack": 0} for c in CHANGES}
    for r in rows:
        if r["decision_change"] not in changes:
            fail("scores_paired.csv holds an unknown decision_change value")
        changes[r["decision_change"]]["all"] += 1
        changes[r["decision_change"]]["bona_fide" if r["label"] == "0" else "attack"] += 1
    return {"n": len(rows), "max_abs_diff": max(ad) if ad else None, "median_abs_diff": float(np.median(ad)) if ad else None,
            "mean_abs_diff": float(np.mean(ad)) if ad else None, "max_signed_diff_cache_minus_c1": max(sd) if sd else None,
            "min_signed_diff_cache_minus_c1": min(sd) if sd else None, "mean_signed_diff_cache_minus_c1": float(np.mean(sd)) if sd else None,
            "abs_diff_bins": bins, "decision_changes_c1_to_cache": changes,
            "frames_equal": sum(1 for r in rows if r["frame_equal"] == "1"),
            "detector_pairs": dict(sorted(Counter(f"{r['detector_cache']}/{r['detector_c1']}" for r in rows).items()))}


def metrics_from_rows(rows) -> dict:
    y = [int(r["label"]) for r in rows]
    return {EXTRACTORS[0]: N1.set_metrics(y, [float(r["score_cache"]) for r in rows]), EXTRACTORS[1]: N1.set_metrics(y, [float(r["score_c1"]) for r in rows])}


def subject_rows(run_id: str, rows) -> list:
    out = []
    for n, col in zip(EXTRACTORS, ("prediction_cache", "prediction_c1")):
        view = [{"subject_id": r["subject_id"], "label": r["label"], "prediction": r[col]} for r in rows]
        out += [{"run_id": run_id, "extractor": n, **x} for x in N1.per_subject(view, SET)]
    return out


def cmd_score(a) -> None:
    d, cfg, cfg_sha, check, check_sha = open_study(a)
    if not a.i_am_authorized:
        fail("score needs --i-am-authorized (inference on the restricted NUAA data)")
    N2.refuse_existing(d, SCORE_OUTPUTS, "score")
    N2.load_step(d, "frames.json", "frames", d.name, cfg_sha, check_sha)
    par = N2.load_step(d, "parity.json", "parity", d.name, cfg_sha, check_sha)
    if par.get("frames_sha256") != sha256_file(d / "frames.json"):
        fail("parity.json does not belong to this frames.json (SHA-256 differs): run frames and parity again in a new study")
    N1.accept(d, "score", "score.log")
    print(f"== C1CROP score: study {d.name}, two extractors, same checkpoint, CPU")
    inp = load_inputs(cfg, a, originals=False)
    same_inputs(inp, check)
    expected = expected_frames(d, inp)
    pres = preprocessors(a, cfg)
    per = {}
    for n in EXTRACTORS:
        frames, dets, hashes = recompute_checked(pres[n], inp, a.images, expected[n], n)
        per[n] = {"scores": attack_scores(cfg, a, frames), "dets": dets, "hashes": hashes}
    c0, c1 = per[EXTRACTORS[0]], per[EXTRACTORS[1]]
    rows = []
    for i, r in enumerate(inp.c1):
        v0, v1 = c0["scores"][i], c1["scores"][i]
        rows.append({"run_id": d.name, "sample_id": r["key"], "subject_id": r["subject"], "session_id": r["session"], "label": r["label"],
                     "file_sha256": r["sha256"], "detector_cache": c0["dets"][i], "detector_c1": c1["dets"][i], "frame_sha256_cache": c0["hashes"][i],
                     "frame_sha256_c1": c1["hashes"][i], "frame_equal": "1" if c0["hashes"][i] == c1["hashes"][i] else "0",
                     "score_cache": repr(v0), "score_c1": repr(v1), "abs_diff": repr(abs(v0 - v1)), "signed_diff": repr(v0 - v1),
                     "prediction_cache": prediction(v0), "prediction_c1": prediction(v1), "decision_change": change_of(prediction(v1), prediction(v0)),
                     "weights_sha256": cfg["weights"]["sha256"], "config_sha256": cfg_sha})
    scores_txt = N1.csv_text(rows, SCORE_COLUMNS)
    back = list(csv.DictReader(io.StringIO(scores_txt)))      # metriche dalle righe rilette dal testo scritto (repr esatto)
    met, paired = metrics_from_rows(back), paired_stats(back)
    subj = subject_rows(d.name, back)
    subj_txt = N1.csv_text(subj, SUBJECT_COLUMNS)
    met_doc = {"run_id": d.name, "set": SET, "metrics": met, "paired": paired, "weights_sha256": cfg["weights"]["sha256"]}
    met_txt = json.dumps(met_doc, indent=1) + "\n"
    doc = {"passed": True, "step": "score", "run_id": d.name, "config_sha256": cfg_sha, "check_sha256": check_sha,
           "parity_sha256": sha256_file(d / "parity.json"), "weights_sha256": cfg["weights"]["sha256"], "device": "CPU", "n_images": len(rows),
           "detectors": {n: dict(sorted(Counter(per[n]["dets"]).items())) for n in EXTRACTORS}, "environment": N2.environment(),
           "files": {"scores_paired.csv": sha256_text(scores_txt), "per_subject.csv": sha256_text(subj_txt), "metrics.json": sha256_text(met_txt)},
           "created": now()}
    write_all(d, {"scores_paired.csv": scores_txt, "per_subject.csv": subj_txt, "metrics.json": met_txt, "score.json": json.dumps(doc, indent=1) + "\n"})
    for n in EXTRACTORS:
        m = met[n]
        print(f"   {n}: APCER {N1.fmt_na(m['apcer'])} ({m['accepted_attacks']}/{m['n_attack']}), BPCER {N1.fmt_na(m['bpcer'])} "
              f"({m['rejected_bona_fide']}/{m['n_bona_fide']}), ACER {N1.fmt_na(m['acer'])}, AUROC {N1.fmt_na(m['auroc'])}")
    ch = paired["decision_changes_c1_to_cache"]
    print(f"== score PASS: {len(rows)} paired scores, max abs difference {paired['max_abs_diff']:.4g}, median {paired['median_abs_diff']:.4g}, "
          f"decisions changed {len(rows) - ch['unchanged']['all']} ({CHANGES[1]} {ch[CHANGES[1]]['all']}, {CHANGES[2]} {ch[CHANGES[2]]['all']})")


# ============================================================================ aggregate

def default_plotter(doc: dict, pts: dict) -> dict:
    """La figura dai punti per immagine (score appaiati per classe, senza identificativi): i valori non entrano in
    aggregate.json e restano nel riservato scores_paired.csv."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        fail(f"missing Python package {e.name!r}: the figure needs matplotlib")
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    for cls, col in (("bona_fide", "#2b6cb0"), ("attack", "#c05621")):
        xs, ys = pts[cls]["score_c1"], pts[cls]["score_cache"]
        ax.scatter(xs, ys, s=12, alpha=0.6, color=col, label=f"{cls.replace('_', ' ')} (n={len(xs)})", edgecolors="none")
    ax.plot([0, 1], [0, 1], color="k", linewidth=0.8)
    ax.axvline(THRESHOLD, color="k", linestyle=":", linewidth=1)
    ax.axhline(THRESHOLD, color="k", linestyle=":", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("attack score, C1 extractor (face.py a80f6f9)")
    ax.set_ylabel("attack score, cache extractor (face.py 0ac7143)")
    ax.set_title("AttackNet V2.1 official NUAA on the 300 C1 images: paired scores\n(threshold 0.5 dotted, diagonal = equal scores)", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return {"figure_paired.png": buf.getvalue()}


def table_md(doc: dict) -> str:
    f4 = N1.f4
    met, paired, ref = doc["metrics"], doc["paired"], doc["c1_reference"]["metrics"]
    L = [f"# C1 crop check: the cache extractor against the C1 extractor, AttackNet V2.1 official NUAA, the 300 C1 images (study `{doc['run_id']}`)", "",
         f"Checkpoint `{doc['weights']['file']}` (SHA-256 `{doc['weights']['sha256'][:12]}…`), the one of C1, loaded as is. Same 300 images (manifest "
         f"`nuaa_c1_v2.csv`), same chain (`train_cnn.py` preprocess: crop with margin 0.15, square, whole frame without a face; 256×256 INTER_AREA; "
         f"reference enhancement; /255), both extractors scored on the CPU in this study (the stored C1 scores came from the application on the GPU, one image at a time: see the parity line below), threshold {THRESHOLD} fixed in C1, {DECISION_RULE}; only the face extractor varies: "
         f"`{EXTRACTORS[0]}` built the training cache, `{EXTRACTORS[1]}` was used in the C1 evaluation (RetinaFace box 1 px narrower and 1 px shorter). "
         "No training, no calibration, no score inversion.", "",
         "## Metrics per extractor (same 300 images)", "",
         "| extractor | n bona fide | n attack | rejected bona fide | accepted attacks | APCER | BPCER | ACER | AUROC |", "|---|---|---|---|---|---|---|---|---|"]
    for n in EXTRACTORS:
        m = met[n]
        L.append(f"| {n} | {m['n_bona_fide']} | {m['n_attack']} | {m['rejected_bona_fide']} | {m['accepted_attacks']} | {f4(m['apcer'])} | {f4(m['bpcer'])} | "
                 f"{f4(m['acer'])} | {f4(m['auroc'])} |")
    L.append(f"| C1 delivered (stored scores, {doc['c1_reference']['run_id']}) | {ref['n_bona_fide']} | {ref['n_attack']} | {ref['rejected_bona_fide']} | "
             f"{ref['accepted_attacks']} | {f4(ref['apcer'])} | {f4(ref['bpcer'])} | {f4(ref['acer'])} | {f4(ref['auroc'])} |")
    ch = paired["decision_changes_c1_to_cache"]
    cc, cb = doc["parity"]["c1_conditions"], doc["parity"]["cpu_batch1"]
    L += ["", f"Parity of the C1 chain (CPU, batch 16, the scoring of this study): {doc['parity']['within_tolerance']}/{doc['parity']['n']} recomputed scores within "
          f"{doc['parity']['tolerance_abs']:g} of the stored C1 scores (max abs difference {doc['parity']['max_abs_diff']:.3e}; within 1e-6: {doc['parity']['within_1e-6']}), "
          f"{doc['parity']['same_decision']}/{doc['parity']['n']} same decisions. In the C1 conditions ({cc['device']}, one image at a time, informative): max abs "
          f"difference {cc['max_abs_diff']:.3e}, within 1e-6 {cc['within_1e-6']}/{doc['parity']['n']}, same decisions {cc['same_decision']}/{doc['parity']['n']}. "
          f"On the CPU one image at a time (informative): max abs difference {cb['max_abs_diff']:.3e}, within 1e-5 {cb['within_1e-5']}/{doc['parity']['n']}, "
          f"same decisions {cb['same_decision']}/{doc['parity']['n']}. "
          f"Cache chain: {doc['frames']['equal_to_cache'][EXTRACTORS[0]]}/{doc['frames']['n']} frames of `{EXTRACTORS[0]}` equal bit for bit to the cache rows.", "",
          "## Paired differences (cache extractor minus C1 extractor, per image)", "",
          "| statistic | value |", "|---|---|",
          f"| images | {paired['n']} |",
          f"| frames identical between the extractors | {paired['frames_equal']} |",
          f"| max abs score difference | {paired['max_abs_diff']:.6f} |",
          f"| median abs score difference | {paired['median_abs_diff']:.6f} |",
          f"| mean abs score difference | {paired['mean_abs_diff']:.6f} |",
          f"| signed difference: min / mean / max | {paired['min_signed_diff_cache_minus_c1']:.6f} / {paired['mean_signed_diff_cache_minus_c1']:.6f} / "
          f"{paired['max_signed_diff_cache_minus_c1']:.6f} |",
          f"| decisions changed at {THRESHOLD} | {paired['n'] - ch['unchanged']['all']} |",
          f"| attack (C1 extractor) to bona fide (cache extractor): all / bona fide / attack images | {ch[CHANGES[1]]['all']} / {ch[CHANGES[1]]['bona_fide']} / {ch[CHANGES[1]]['attack']} |",
          f"| bona fide (C1 extractor) to attack (cache extractor): all / bona fide / attack images | {ch[CHANGES[2]]['all']} / {ch[CHANGES[2]]['bona_fide']} / {ch[CHANGES[2]]['attack']} |",
          "", "| abs score difference | images |", "|---|---|"]
    for k, v in paired["abs_diff_bins"].items():
        L.append(f"| {k} | {v} |")
    L += ["", "## Detector per image", "", "| detector pair (cache / C1) | images |", "|---|---|"]
    for k, v in paired["detector_pairs"].items():
        L.append(f"| {k} | {v} |")
    L += ["", "| box delta (C1 minus cache: dx, dy, dw, dh) | images |", "|---|---|"]
    for k, v in doc["frames"]["box_delta_c1_minus_cache"].items():
        L.append(f"| {k} | {v} |")
    L += ["", f"Pixel differences between the two preprocessed frames: max abs {doc['frames']['max_abs_pixel_difference']}, mean abs over the different frames "
          f"{doc['frames']['mean_abs_pixel_difference_over_different_frames']:.3f} (levels on 0-255).", "",
          "NA = not computable (class absent). Values are rounded to four decimals (metrics), six (score differences), three significant digits (parity maximum) "
          "or three decimals (pixel mean); aggregate.json holds the exact values.", ""]
    return "\n".join(L)


def cmd_aggregate(a) -> None:
    d, cfg, cfg_sha, check, check_sha = open_study(a)
    for n in AGG_OUTPUTS:
        q = d / n
        if os.path.lexists(q) and not q.is_symlink() and not q.is_file():
            fail(f"{n} in the study folder is not a regular file: refused, nothing removed")
    N1.accept(d, "aggregate", "aggregate.log")
    for n in AGG_OUTPUTS:
        q = d / n
        if q.is_symlink() or q.is_file():
            q.unlink()
            print(f"   removed previous {n}")
    inp = load_inputs(cfg, a, originals=False)
    same_inputs(inp, check)
    fr = N2.load_step(d, "frames.json", "frames", d.name, cfg_sha, check_sha)
    par = N2.load_step(d, "parity.json", "parity", d.name, cfg_sha, check_sha)
    sc = N2.load_step(d, "score.json", "score", d.name, cfg_sha, check_sha)
    if par.get("frames_sha256") != sha256_file(d / "frames.json") or sc.get("parity_sha256") != sha256_file(d / "parity.json"):
        fail("frames.json, parity.json and score.json do not form one chain (SHA-256 differ)")
    if sc.get("weights_sha256") != cfg["weights"]["sha256"] or par.get("weights_sha256") != cfg["weights"]["sha256"]:
        fail("score.json or parity.json record another checkpoint")
    rows = N1.read_csv_rows(d / "scores_paired.csv", SCORE_COLUMNS)
    if [r["sample_id"] for r in rows] != inp.c1_keys:
        fail("scores_paired.csv does not hold exactly the c1 rows in key order")
    expected = expected_frames(d, inp)
    for r in rows:
        m = inp.by_key["c1"][r["sample_id"]]
        e0, e1 = expected[EXTRACTORS[0]][r["sample_id"]], expected[EXTRACTORS[1]][r["sample_id"]]
        try:
            v0, v1 = float(r["score_cache"]), float(r["score_c1"])
        except ValueError:
            v0 = v1 = float("nan")
        exp = {"run_id": d.name, "subject_id": m["subject"], "session_id": m["session"], "label": m["label"], "file_sha256": m["sha256"],
               "detector_cache": e0["detector"], "detector_c1": e1["detector"], "frame_sha256_cache": e0["frame_sha256"], "frame_sha256_c1": e1["frame_sha256"],
               "frame_equal": "1" if e0["frame_sha256"] == e1["frame_sha256"] else "0", "prediction_cache": prediction(v0), "prediction_c1": prediction(v1),
               "decision_change": change_of(prediction(v1), prediction(v0)), "weights_sha256": cfg["weights"]["sha256"], "config_sha256": cfg_sha}
        if any(r[k] != x for k, x in exp.items()) or not (0 <= v0 <= 1 and 0 <= v1 <= 1) or float(r["abs_diff"]) != abs(v0 - v1) or float(r["signed_diff"]) != v0 - v1:
            fail("scores_paired.csv: rows inconsistent with the manifests, the frames, the checkpoint or the decision rule")
    if sc.get("detectors") != {n: dict(Counter(r[c] for r in rows)) for n, c in zip(EXTRACTORS, ("detector_cache", "detector_c1"))}:
        fail("detector counts of score.json differ from the rows")
    met, paired = metrics_from_rows(rows), paired_stats(rows)
    stored = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    if json.dumps(met, sort_keys=True) != json.dumps(stored.get("metrics"), sort_keys=True) or json.dumps(paired, sort_keys=True) != json.dumps(stored.get("paired"), sort_keys=True):
        fail("metrics or paired statistics recomputed from the scores differ from metrics.json")
    subj = N1.read_csv_rows(d / "per_subject.csv", SUBJECT_COLUMNS)
    if N1.csv_text(subj, SUBJECT_COLUMNS) != N1.csv_text(subject_rows(d.name, rows), SUBJECT_COLUMNS):
        fail("per_subject.csv differs from the table recomputed from the scores")
    pts = {cls: {"score_c1": [float(r["score_c1"]) for r in rows if r["label"] == lab], "score_cache": [float(r["score_cache"]) for r in rows if r["label"] == lab]}
           for cls, lab in (("bona_fide", "0"), ("attack", "1"))}
    doc = {"passed": True, "step": "aggregate", "run_id": d.name, "config_sha256": cfg_sha, "check_sha256": check_sha, "commit": check.get("commit"),
           "dirty_at_check": check.get("dirty"), "code_sha256": check["code_sha256"], "weights": {k: cfg["weights"][k] for k in ("file", "sha256", "bytes", "source")},
           "extractors": {n: {k: cfg["extractors"][n][k] for k in ("file", "commit", "sha256", "role", "retinaface_box")} for n in EXTRACTORS},
           "difference": cfg["difference"], "chain": cfg["chain"], "set": SET, "n": len(rows), "threshold": THRESHOLD, "decision": DECISION_RULE,
           "threshold_origin": cfg["metrics"]["threshold_origin"], "metrics": met, "paired": paired,
           "frames": fr["summary"], "parity": {k: par[k] for k in ("tolerance_abs", "n", "max_abs_diff", "median_abs_diff", "within_tolerance", "within_1e-6", "within_1e-7", "same_decision",
                                          "stored_device", "c1_conditions", "cpu_batch1")},
           "c1_reference": {"run_id": cfg["c1_reference"]["run_id"], "scores_sha256": cfg["c1_reference"]["sha256"], "metrics": par["c1_reference_metrics"]},
           "selection": "none: both extractors reported on the same images with the same checkpoint and threshold; no score inversion",
           "figures": {"figure_paired.png": "paired attack scores per image (C1 extractor on x, cache extractor on y), one colour per class, no identifiers"},
           "created": now()}
    js, md = json.dumps(doc, indent=1) + "\n", table_md(doc)
    for name, text in (("aggregate.json", js), ("aggregate.md", md)):
        if probs := N1.public_problems(text, inp):
            fail(f"{name} would hold {', '.join(probs)}: not written")
    figs = (PLOTTER or default_plotter)(doc, pts)
    if set(figs) != {"figure_paired.png"} or not all(isinstance(v, bytes) and v[:8] == b"\x89PNG\r\n\x1a\n" for v in figs.values()):
        fail("the plotter did not return the PNG figure")
    write_all(d, {"aggregate.json": js, "aggregate.md": md})
    try:
        N2.write_binary(d, figs)
    except BaseException:
        for n in ("aggregate.json", "aggregate.md"):
            (d / n).unlink(missing_ok=True)
        raise
    print(md)
    print(f"== aggregate PASS: {len(rows)} paired scores, aggregate.json, aggregate.md and the figure (publishable, scanned)")


# ============================================================================ main

COMMANDS = {"check": cmd_check, "frames": cmd_frames, "parity": cmd_parity, "score": cmd_score, "aggregate": cmd_aggregate}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="C1 crop check: the cache extractor against the C1 extractor on AttackNet V2.1 and the 300 C1 images")
    ap.add_argument("command", choices=list(COMMANDS))
    ap.add_argument("--config", required=True, help="configuration JSON (c1crop.json)")
    ap.add_argument("--manifests", required=True, help="folder with the M2 NUAA manifests")
    ap.add_argument("--images", required=True, help="root folder of the original NUAA images")
    ap.add_argument("--cache", required=True, help="nuaa_1500.npz (training cache)")
    ap.add_argument("--weights", required=True, help="AttackNetV2_1__nuaa.h5 (the C1 checkpoint)")
    ap.add_argument("--c1-scores", required=True, help="scores.csv of the C1 run (group schema, restricted)")
    ap.add_argument("--face-cache", required=True, help="face_0ac7143.py (frozen copy of the cache extractor)")
    ap.add_argument("--face-c1", required=True, help="face_a80f6f9.py (frozen copy of the C1 extractor)")
    ap.add_argument("--study-dir", required=True, help="study folder YYYYMMDD-C1CROP-<shortcommit>[-trial]")
    ap.add_argument("--i-am-authorized", action="store_true", help="inference on the restricted data is authorised (score)")
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    N1.RUN_STATE.update(dir=None, cmd=a.command, log=None)
    code = 0
    try:
        COMMANDS[a.command](a)
        N1.write_status("PASS")
        print(f"RESULT: PASS ({a.command})")
    except (CropError, SystemExit, Exception, KeyboardInterrupt) as e:  # noqa: BLE001
        if isinstance(e, SystemExit) and e.code in (None, 0):
            raise
        msg = str(e) if isinstance(e, (CropError, SystemExit)) else f"unexpected {type(e).__name__}: {e}"
        if isinstance(e, KeyboardInterrupt):
            msg = "interrupted"
        print(f"ERROR: {msg}", file=sys.stderr)
        try:
            N1.write_status("FAIL", msg)
        except BaseException as e2:  # noqa: BLE001
            print(f"ERROR: status.json not written ({type(e2).__name__})", file=sys.stderr)
        print(f"RESULT: FAIL ({a.command})")
        code = 130 if isinstance(e, KeyboardInterrupt) else 1
    finally:
        N1.close_log()
    return code


if __name__ == "__main__":
    sys.exit(main())
