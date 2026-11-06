"""
Run LivenessNet su NUAA con quattro fold per soggetto e tre seed di training (controllo subject-held-out con la
sessione rappresentata nel training), autorizzato dal coordinatore il 3/11/2026 sulla scheda C2 dopo l'approvazione
dello schema dei fold (PR #16, run 20261103-FOLDS-67b7f3d): «può procedere con l'implementazione e l'esecuzione
dell'esperimento LivenessNet su CPU: 4 fold per soggetto, seed 42/43/44 e limite di calcolo complessivo di 6 ore,
secondo la configurazione consegnata. Esegua prima i controlli previsti sugli input e sulla cache. Mantenga fissi
preprocessing, split e ricetta; determini la soglia sul solo dev e la congeli prima del test. Conservi tutti gli
esiti, anche negativi».

Passi del run, in un solo processo e nell'ordine (ognuno registrato con il tempo impiegato):
  1. check-inputs   configurazione a insieme chiuso; i cinque manifest M2 (SHA-256, conteggi, liste disgiunte); i
                    dodici manifest fold<k>_<ruolo>.csv dello schema accettato (SHA256SUMS riservato, conteggi,
                    ruoli disgiunti per soggetto e per chiave, ogni riga uguale alla lista ufficiale, flag in_cache);
                    cache nuaa_1500.npz (SHA-256, schema .npy senza pickle, etichette e soggetti uguali ai manifest
                    nell'ordine di campionamento); estrattore congelato, risorse del rilevatore e codice di
                    riferimento per SHA-256; ambiente (OpenCV registrato, TensorFlow non importato). Nulla scritto.
  2. cache-check    ognuno dei 6.000 frame della cache rifatto dall'originale con l'estrattore congelato e confrontato
                    bit per bit (il controllo N2, ripetuto); rilevatore usato per frame contato.
  3. extract        i frame delle liste ufficiali che non stanno nella cache (6.614) estratti una volta con la stessa
                    catena, salvati in data/processed/nuaa_folds_frames.npz (chiavi ordinate, uint8); SHA-256 del
                    file e di ogni frame; rilevatore per frame (riservato).
  4. train          per ogni fold k e seed: fitting (2.100 frame) e dev (600) dai manifest dello schema, ricetta N1
                    invariata su CPU (Adam legacy, 12 epoche al massimo, early stopping sul val_loss del dev con
                    ripristino dei pesi migliori, scheduler, augmentation, pesi di classe bilanciati); pesi .keras
                    riservati con SHA-256; scheda con la storia per epoca.
  5. threshold      punteggi del dev con il checkpoint (CPU), val_loss ricalcolato contro la scheda, soglia EER con
                    il criterio M1 scritta in threshold.json, riletta e ancorata per SHA-256 PRIMA di qualunque
                    punteggio del test.
  6. score          punteggi di ogni frame del test del fold (tutti i frame dei 4 soggetti tenuti fuori), decisione a
                    0,5 e alla soglia dev congelata.
  7. aggregate      per fold, seed e strato di sessione (01, 02, 03, all): n, errori, APCER/BPCER/ACER alle due soglie,
                    AUROC; per soggetto (riservato); mediana e intervallo sui seed per fold; conteggi aggregati sui
                    quattro fold per sessione e seed; bootstrap sui 16 soggetti tenuti fuori (1.000 ricampionamenti,
                    seme 0, percentili 2,5–97,5 di ACER e AUROC); pubblico senza identificativi.
Tetto di tempo (budget.wall_time_cap_minutes, 360): controllato a ogni confine di passo e alla fine di ogni epoca; se
superato il run si ferma, le uscite sono tolte e lo stato è FAIL (protocollo: «stop, no results; report as a failed
run»). Nessuna scelta dipende da un risultato: fold, conteggi, ricetta e criterio della soglia sono fissati nella
configurazione prima del training (pre-commitment).

Uscite (cartella del run <out-root>/YYYYMMDD-FOLDSRUN-<commit>[-dev], -dev se l'albero git non è pulito):
  status.json, run.log
  public/      aggregate.json, aggregate.md, run_record.json, SHA256SUMS   (nessun identificativo, nessuna chiave)
  restricted/  fold<k>_seed<s>_card.json, _threshold.json, _dev_scores.csv, _test_scores.csv, _LivenessNet.keras;
               per_subject.csv, extracted_frames.csv, public_SHA256SUMS.sha256, SHA256SUMS
Dentro il repository la cartella del run è ammessa solo sotto results/foldsrun/; i frame estratti solo in data/processed/.

  .venv/bin/python -I scripts/nuaa_folds_run.py check-inputs                  # solo verifica, nulla scritto
  .venv/bin/python -I scripts/nuaa_folds_run.py run --i-am-authorized         # tutti i passi

Dipendenze: NumPy, OpenCV (estrazione), TensorFlow 2.15 (training e scoring, solo CPU); scripts/m2_writers.py,
scripts/path_guard.py, scripts/package_rules.py, scripts/train_cnn.py (preprocess, invariato), il codice di riferimento
del coordinatore (architectures.py, create_datasets.py) e l'estrattore congelato delivery/n2/face_0ac7143.py.
"""
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import re
import signal
import stat
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import m2_writers as mw  # noqa: E402
import package_rules as pr  # noqa: E402
import path_guard as pg  # noqa: E402

DEFAULT_CONFIG = "delivery/foldsrun/nuaa_folds_run.json"
DEFAULT_OUT_ROOT = "results/foldsrun"
CONFIG_SCHEMA = "nuaa-folds-run-config/1"
RUN_ID_RE = re.compile(r"^\d{8}-FOLDSRUN-[0-9a-f]{7,40}(-dev)?$")
TOP_KEYS = ("schema", "task", "authorisation", "schema_run", "manifests", "images", "expected", "cache", "extraction",
            "environment", "architecture", "train_seeds", "training", "threshold", "scoring", "metrics", "bootstrap", "budget",
            "declarations", "outputs")
CONFIG_SHA256 = "7aaf0be21356631e1541f233b6e483a41646aad2e43bbcd15bbefb8d2c09bf4f"       # SHA-256 della configurazione consegnata (delivery/foldsrun/nuaa_folds_run.json), fissato prima del run
SCHEMA_CONFIG_SHA256 = "d1d43c148f762965c5ae367a0de1c59770324f7840034fd5a6c7fad0a14e73e8"   # delivery/folds/nuaa_subject_folds.json
TRAIN_SEEDS = [42, 43, 44]       # fissati dal coordinatore (3 novembre 2026)
CAP_MINUTES = 360                # tetto complessivo di calcolo fissato dal coordinatore
DECLARATION_KEYS = ("pre_commitment", "dev_reuse", "cpu_only", "all_outcomes_kept", "cpu_nondeterminism", "diagnostic", "exploratory",
                    "limits", "question")
DECLARATION_FLAGS = ("pre_commitment", "dev_reuse", "cpu_only", "all_outcomes_kept")
RESOURCE_NAMES = ("deploy.prototxt", "Widerface-RetinaFace.caffemodel", "haarcascade_frontalface_default.xml")
ENV_REFUSED = ("LIVENESSLAB_SILENT_FACE", "LIVENESSLAB_LIVEDETECTION")   # ridirigerebbero le risorse del rilevatore e il codice di riferimento
MANIFESTS = ("official_train", "official_test", "cnn_train", "cnn_es", "cnn_test")
CACHE_MANIFESTS = ("cnn_train", "cnn_es", "cnn_test")
COLUMNS = ("key", "label", "subject", "glasses", "pos", "session", "pic", "official_split", "sha256", "bytes")
FOLD_COLUMNS = ("fold", "role", "key", "label", "subject", "session", "sha256", "bytes", "in_cache")
ROLES = ("fitting", "dev", "test")
SESSIONS = ("01", "02", "03")
STRATA = ("01", "02", "03", "all")
LABELS = ("0", "1")
CLASS_NAME = {"0": "bona_fide", "1": "attack"}
NAME_RE = re.compile(r"^(\d{4})_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d+)\.jpg$")
KEY_DIR_OF_LABEL = {"0": "real", "1": "attack"}
MEMBERS = ("X_train", "y_train", "s_train", "X_test", "y_test", "s_test")
NPY_HEADERS = {(1, 0): "read_array_header_1_0", (2, 0): "read_array_header_2_0"}
FRAME_SHAPE = (256, 256, 3)          # forma di un frame della cache e dei frame estratti (i test la riducono)
FIXED_THRESHOLD = 0.5
DECISION_RULE = "attack if score_attack > threshold (tie = bona fide)"
SCORE_BATCH = 16
VAL_LOSS_TOL = (1e-3, 1e-2)          # |val_loss ricalcolato - minimo registrato| <= 1e-3 oppure <= 1 % del minimo
EPS = 1e-7                           # clipping di Keras nella categorical crossentropy
BOOT = {"resamples": 1000, "seed": 0, "percentiles": [2.5, 97.5], "min_subjects": 5}
NA = "NA"
# ricetta N1 (n1_train.FROZEN["training"], studio 20261024-N1-LivenessNet-fb93496), invariata: la configurazione deve averla identica
FROZEN_TRAINING = {
    "recipe": "N1 part 2 (n1_nuaa_stability.json, study 20261024-N1-LivenessNet-fb93496), unchanged",
    "max_epochs": 12, "batch": 16, "initial_learning_rate": 0.001, "early_stopping_patience": 4,
    "checkpoint": "weights of the epoch with minimum val_loss on the dev frames (EarlyStopping restore_best_weights; "
                  "set_weights when the epochs end first)",
    "optimizer": "tf.keras.optimizers.legacy.Adam(learning_rate=initial_learning_rate, clipnorm=1.0)",
    "loss": "CategoricalCrossentropy, one-hot labels (index 0 bona fide, 1 attack)",
    "class_weight": "balanced on the fitting set: n / (2 * n_class)",
    "augmentation": "RandomFlip(horizontal), RandomRotation(0.03), RandomZoom(0.12, 0.12), RandomTranslation(0.06, 0.06), "
                    "RandomBrightness(0.15, value_range=(0, 1)), RandomContrast(0.2), GaussianNoise(0.02); fitting set only",
    "scheduler": "ReduceLROnPlateau(monitor=val_loss, factor=0.5, patience=2, min_lr=1e-6)",
    "shuffle": "tf.data shuffle(4096, seed=train_seed)",
    "seeds": "train_seed for random, numpy and TensorFlow (weight initialisation, augmentation, batch order); no split seed: "
             "the folds and the samples are deterministic",
    "device": "CPU",
}
THRESHOLD_CRITERION = ("eer, the M1 protocol: candidates = distinct dev scores plus nextafter(min, -inf) if >= 0; attack if "
                       "score > value (tie = bona fide); minimum of |fp * n_attack - fn * n_bona_fide| on integers, ties to "
                       "the highest value")
CODE_FILES = ("scripts/nuaa_folds_run.py", "scripts/m2_writers.py", "scripts/path_guard.py", "scripts/package_rules.py",
              "scripts/train_cnn.py")
STATUS = "status.json"
LOG = "run.log"
PUBLIC, RESTRICTED = "public", "restricted"
PUBLIC_OUTPUTS = ("aggregate.json", "aggregate.md", "run_record.json", "SHA256SUMS")
PER_MODEL = ("card.json", "threshold.json", "dev_scores.csv", "test_scores.csv", "LivenessNet.keras")
RESTRICTED_FIXED = ("per_subject.csv", "extracted_frames.csv", "public_SHA256SUMS.sha256", "SHA256SUMS")
SCORE_COLUMNS = ("fold", "train_seed", "role", "sample_id", "subject_id", "session_id", "label", "score_attack",
                 "prediction_05", "prediction_dev", "threshold_dev", "file_sha256", "checkpoint_sha256", "threshold_sha256")
SUBJECT_COLUMNS = ("fold", "train_seed", "subject_id", "stratum", "n_bona_fide", "n_attack", "rejected_bona_fide_05",
                   "accepted_attacks_05", "apcer_05", "bpcer_05", "acer_05", "rejected_bona_fide_dev", "accepted_attacks_dev",
                   "apcer_dev", "bpcer_dev", "acer_dev", "auroc")
FRAME_COLUMNS = ("sample_id", "subject_id", "session_id", "label", "official_split", "detector", "file_sha256", "frame_sha256")
FAIL_DOC = {"stage": "invalidation-failed", "note": "this output could not be removed after a failure: no verdict is valid"}
EXTRA_PROTECTED = ("models", "results/c1", "results/m1", "results/m2", "results/m2_rgb", "results/eval", "results/n1",
                   "results/n2", "results/c1crop", "results/folds", "results/stepzero", "export")
METRIC_KEYS = ("apcer_05", "bpcer_05", "acer_05", "apcer_dev", "bpcer_dev", "acer_dev", "auroc")
FAILED_FRAMES = "nuaa_folds_frames.{run_id}.failed.npz"

# punti di iniezione (solo i test): stato git, pre-elaborazione congelata, trainer, scorer, orologio, architettura
REPO_STATE = None       # (root) -> {"commit", "dirty"}
PREPROCESSOR = None     # (cfg, base) -> funzione(percorso) -> (frame uint8, nome del rilevatore)
TRAINER = None          # (data, training, weights_path, train_seed, arch_module, cap_callback) -> scheda
MAKE_SCORER = None      # (weights_path, sha256) -> funzione(X uint8) -> probabilità N x 2 (default: TensorFlow su CPU)
CLOCK = None            # () -> secondi (default time.monotonic)
ARCH_LOADER = None      # (path) -> modulo con create_model


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


def clock() -> float:
    return (CLOCK or time.monotonic)()


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


def regular_path(p: Path, what: str) -> Path:
    """Percorso di un file regolare (stessa regola di read_regular, senza leggerlo: per i file grandi)."""
    try:
        st = os.lstat(p)
    except OSError:
        fail(f"{what} {p.name} not found")
    if stat.S_ISLNK(st.st_mode):
        fail(f"{what} {p.name} is a symbolic link: inputs must be regular files")
    if not stat.S_ISREG(st.st_mode):
        fail(f"{what} {p.name} is not a regular file")
    return p


def spread(items, k):
    """k elementi distribuiti uniformemente sull'elenco ordinato (la regola di train_cnn.py e dello schema dei fold)."""
    items = sorted(items)
    if k >= len(items):
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def spread_keys(rows, per_class: int) -> list:
    """Chiavi di spread per classe, bona fide e poi attacchi (nuaa_manifests.spread_keys, l'ordine della cache)."""
    out = []
    for lab in LABELS:
        names = [r["key"].split("/", 1)[1] for r in rows if r["label"] == lab]
        out += [f"{KEY_DIR_OF_LABEL[lab]}/{n}" for n in spread(names, per_class)]
    return out


def frame_hashes(arr) -> list:
    """SHA-256 di ogni frame (nuaa_manifests.frame_hashes: tipo e forma in testa, poi i byte in ordine C)."""
    head = f"{arr.dtype.str}{arr.shape[1:]}".encode()
    return [hashlib.sha256(head + np.ascontiguousarray(x).tobytes()).hexdigest() for x in arr]


def csv_text(cols, rows) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    w.writerows(rows)
    return buf.getvalue()


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


def sums_text(files: dict) -> str:
    return "".join(f"{sha256_bytes(t.encode('utf-8'))}  {n}\n" for n, t in sorted(files.items()))


def f4(v) -> str:
    return NA if v is None else f"{v:.4f}"


def fmt(v) -> str:
    return NA if v is None else repr(float(v))


def rel_path(base: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else base / q


# ============================================================================ configurazione

def is_count(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _hex(v) -> bool:
    return isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) is not None


def _text(v, what: str) -> None:
    if not isinstance(v, str) or not v.strip():
        fail(f"configuration: {what} must be a non-empty text")


def _check_manifest_spec(spec, name: str) -> None:
    if not isinstance(spec, dict) or sorted(spec) != ["file", "n_attack", "n_bona_fide", "n_subjects", "rows", "sessions", "sha256"]:
        fail(f"configuration: manifests.files.{name} needs file, sha256, rows, n_bona_fide, n_attack, n_subjects, sessions")
    _text(spec["file"], f"manifests.files.{name}.file")
    if not _hex(spec["sha256"]):
        fail(f"configuration: manifests.files.{name}.sha256 must be a SHA-256")
    for k in ("rows", "n_bona_fide", "n_attack", "n_subjects"):
        if not is_count(spec[k]):
            fail(f"configuration: manifests.files.{name}.{k} must be a non-negative integer")
    if not isinstance(spec["sessions"], list) or not spec["sessions"] or any(s not in SESSIONS for s in spec["sessions"]):
        fail(f"configuration: manifests.files.{name}.sessions must list sessions among {', '.join(SESSIONS)}")


def load_config(p: Path) -> tuple:
    """(configurazione, SHA-256 dei byte letti). Insiemi chiusi e valori fissati controllati qui, prima di tutto."""
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
    for k in ("task", "authorisation"):
        _text(cfg[k], k)
    sr = cfg["schema_run"]
    if not isinstance(sr, dict) or sorted(sr) != ["config_path", "config_sha256", "folder", "restricted_sha256sums_sha256", "run_id"]:
        fail("configuration: schema_run needs run_id, folder, config_path, config_sha256, restricted_sha256sums_sha256")
    _text(sr["run_id"], "schema_run.run_id")
    _text(sr["folder"], "schema_run.folder")
    _text(sr["config_path"], "schema_run.config_path")
    if not _hex(sr["config_sha256"]) or not _hex(sr["restricted_sha256sums_sha256"]):
        fail("configuration: schema_run SHA-256 values malformed")
    man = cfg["manifests"]
    if not isinstance(man, dict) or sorted(man) != ["files", "folder", "origin"] or not isinstance(man["files"], dict) \
            or sorted(man["files"]) != sorted(MANIFESTS):
        fail(f"configuration: manifests needs folder, origin and the files {', '.join(MANIFESTS)}")
    _text(man["folder"], "manifests.folder")
    _text(man["origin"], "manifests.origin")
    for k, v in man["files"].items():
        _check_manifest_spec(v, k)
    if set(man["files"]["official_train"]["sessions"]) & set(man["files"]["official_test"]["sessions"]):
        fail("configuration: the declared sessions of the official lists must be disjoint")
    img = cfg["images"]
    if not isinstance(img, dict) or sorted(img) != ["folder", "layout"]:
        fail("configuration: images needs folder and layout")
    _text(img["folder"], "images.folder")
    if img["layout"] not in LAYOUTS:
        fail(f"configuration: images.layout must be one of {sorted(LAYOUTS)}")
    exp = cfg["expected"]
    exp_keys = ["dev_per_fold", "fitting_per_fold", "k", "n_cache_frames", "n_images", "n_subjects", "n_to_extract", "test_per_fold"]
    if not isinstance(exp, dict) or sorted(exp) != exp_keys:
        fail(f"configuration: expected needs exactly {', '.join(exp_keys)}")
    for k in exp_keys:
        if k != "test_per_fold" and not is_count(exp[k]):
            fail(f"configuration: expected.{k} must be a non-negative integer")
    if exp["k"] < 2:
        fail("configuration: expected.k must be at least 2")
    if not isinstance(exp["test_per_fold"], list) or len(exp["test_per_fold"]) != exp["k"] or not all(is_count(v) for v in exp["test_per_fold"]):
        fail("configuration: expected.test_per_fold must list one count per fold")
    if sum(exp["test_per_fold"]) != exp["n_images"] or exp["n_images"] != exp["n_cache_frames"] + exp["n_to_extract"]:
        fail("configuration: expected counts are inconsistent (test folds partition the images; images = cache + to extract)")
    c = cfg["cache"]
    if not isinstance(c, dict) or sorted(c) != ["built", "detectors_n2", "members", "order", "path", "per_class", "sha256"]:
        fail("configuration: cache needs path, sha256, per_class, members, order, built, detectors_n2")
    dn = c["detectors_n2"]
    if not isinstance(dn, dict) or sorted(dn) != ["test", "train"] or not all(isinstance(v, dict) and v and all(is_count(x) for x in v.values()) for v in dn.values()):
        fail("configuration: cache.detectors_n2 needs the detector counts of train and test (N2 cache check)")
    _text(c["path"], "cache.path")
    if not _hex(c["sha256"]) or not is_count(c["per_class"]) or c["per_class"] < 1:
        fail("configuration: cache.sha256 or cache.per_class malformed")
    if not isinstance(c["members"], dict) or sorted(c["members"]) != sorted(MEMBERS):
        fail(f"configuration: cache.members must define exactly {', '.join(MEMBERS)}")
    for k, v in c["members"].items():
        if not isinstance(v, dict) or sorted(v) != ["dtype", "shape"] or not isinstance(v["dtype"], str) \
                or not isinstance(v["shape"], list) or not all(is_count(x) for x in v["shape"]):
            fail(f"configuration: cache.members.{k} needs dtype and shape")
    if 4 * c["per_class"] != exp["n_cache_frames"] or \
            c["members"]["X_train"]["shape"][0] + c["members"]["X_test"]["shape"][0] != exp["n_cache_frames"]:
        fail("configuration: cache members (two classes, two splits of per_class) do not add up to expected.n_cache_frames")
    ex = cfg["extraction"]
    ex_keys = ["chain", "detector_resources", "frames_file", "frozen_face", "reference_code"]
    if not isinstance(ex, dict) or sorted(ex) != ex_keys:
        fail(f"configuration: extraction needs exactly {', '.join(ex_keys)}")
    _text(ex["chain"], "extraction.chain")
    _text(ex["frames_file"], "extraction.frames_file")
    env = cfg["environment"]
    if not isinstance(env, dict) or sorted(env) != ["opencv", "tensorflow"]:
        fail("configuration: environment needs opencv and tensorflow")
    _text(env["opencv"], "environment.opencv")
    _text(env["tensorflow"], "environment.tensorflow")
    ff = ex["frozen_face"]
    if not isinstance(ff, dict) or sorted(ff) != ["path", "sha256"] or not _hex(ff["sha256"]):
        fail("configuration: extraction.frozen_face needs path and sha256")
    _text(ff["path"], "extraction.frozen_face.path")
    for group in ("detector_resources", "reference_code"):
        g = ex[group]
        if not isinstance(g, dict) or not g:
            fail(f"configuration: extraction.{group} must be a non-empty mapping")
        for k, v in g.items():
            if not isinstance(v, dict) or sorted(v) != ["path", "sha256"] or not _hex(v["sha256"]):
                fail(f"configuration: extraction.{group}.{k} needs path and sha256")
            _text(v["path"], f"extraction.{group}.{k}.path")
    if sorted(ex["reference_code"]) != ["architectures", "create_datasets"]:
        fail("configuration: extraction.reference_code must define architectures and create_datasets")
    if sorted(ex["detector_resources"]) != sorted(RESOURCE_NAMES):
        fail(f"configuration: extraction.detector_resources must define exactly {', '.join(RESOURCE_NAMES)}")
    ar = cfg["architecture"]
    if not isinstance(ar, dict) or sorted(ar) != ["chosen", "name", "path", "sha256"] or ar["name"] != "LivenessNet" \
            or not _hex(ar["sha256"]):
        fail("configuration: architecture needs name LivenessNet, path, sha256, chosen")
    _text(ar["path"], "architecture.path")
    _text(ar["chosen"], "architecture.chosen")
    if ar["sha256"] != ex["reference_code"]["architectures"]["sha256"] or ar["path"] != ex["reference_code"]["architectures"]["path"]:
        fail("configuration: architecture and extraction.reference_code.architectures must name the same file")
    seeds = cfg["train_seeds"]
    if not isinstance(seeds, list) or not seeds or not all(is_count(s) for s in seeds) or len(set(seeds)) != len(seeds):
        fail("configuration: train_seeds must be a list of distinct non-negative integers")
    if seeds != TRAIN_SEEDS:
        fail(f"configuration: train_seeds must be {TRAIN_SEEDS} (fixed by the coordinator)")
    if cfg["training"] != FROZEN_TRAINING:
        fail("configuration: training must be the N1 recipe, unchanged (FROZEN_TRAINING of the script)")
    th = cfg["threshold"]
    th_keys = ["also_reported", "criterion", "dev_reuse_declared", "frozen_before_test", "set"]
    if not isinstance(th, dict) or sorted(th) != th_keys or th["criterion"] != THRESHOLD_CRITERION or th["set"] != "dev fold only":
        fail("configuration: threshold must declare set 'dev fold only' and the EER criterion of the M1 protocol")
    for k in ("also_reported", "frozen_before_test", "dev_reuse_declared"):
        _text(th[k], f"threshold.{k}")
    sc = cfg["scoring"]
    if not isinstance(sc, dict) or sorted(sc) != ["batch", "batchnorm", "device", "input", "score"] or sc["device"] != "CPU" \
            or sc["batch"] != SCORE_BATCH:
        fail(f"configuration: scoring must be on the CPU with batch {SCORE_BATCH}")
    me = cfg["metrics"]
    me_keys = ["aggregation", "auroc", "decision", "per_subject", "per_unit", "pre_commitment", "strata", "uncertainty"]
    if not isinstance(me, dict) or sorted(me) != me_keys or me["strata"] != list(STRATA) or me["decision"] != DECISION_RULE:
        fail(f"configuration: metrics needs {', '.join(me_keys)}, strata {list(STRATA)} and the decision rule {DECISION_RULE!r}")
    for k in me_keys:
        if k != "strata":
            _text(me[k], f"metrics.{k}")
    if cfg["bootstrap"] != BOOT:
        fail(f"configuration: bootstrap must be {json.dumps(BOOT)} (the recipe of the script)")
    bu = cfg["budget"]
    if not isinstance(bu, dict) or sorted(bu) != ["estimate", "on_cap_exceeded", "trainings", "wall_time_cap_minutes"] \
            or not is_count(bu["wall_time_cap_minutes"]) or bu["wall_time_cap_minutes"] < 1:
        fail("configuration: budget needs trainings, estimate, wall_time_cap_minutes (positive) and on_cap_exceeded")
    if bu["trainings"] != exp["k"] * len(seeds):
        fail("configuration: budget.trainings must be k folds x the number of seeds")
    if bu["wall_time_cap_minutes"] != CAP_MINUTES:
        fail(f"configuration: budget.wall_time_cap_minutes must be {CAP_MINUTES} (fixed by the coordinator)")
    _text(bu["estimate"], "budget.estimate")
    _text(bu["on_cap_exceeded"], "budget.on_cap_exceeded")
    dec = cfg["declarations"]
    if not isinstance(dec, dict) or sorted(dec) != sorted(DECLARATION_KEYS):
        fail(f"configuration: declarations must have exactly the keys {', '.join(DECLARATION_KEYS)}")
    for k in DECLARATION_FLAGS:
        if dec.get(k) is not True:
            fail(f"configuration: declarations.{k} must be true")
    for k, v in dec.items():
        if k not in DECLARATION_FLAGS and not (isinstance(v, str) and v):
            fail(f"configuration: declarations.{k} must be a text")
    ou = cfg["outputs"]
    if not isinstance(ou, dict) or sorted(ou) != ["public", "restricted"]:
        fail("configuration: outputs needs public and restricted")
    for k in ou:
        _text(ou[k], f"outputs.{k}")
    return cfg, sha256_bytes(data)


# ============================================================================ ingressi

class Inputs:
    """Manifest, cache, schema dei fold e l'esito di ogni controllo (checks)."""

    def __init__(self):
        self.rows = {}          # manifest M2: nome -> righe
        self.by_key = {}
        self.official = {}      # chiave -> riga delle liste ufficiali
        self.folds = {}         # fold -> ruolo -> righe
        self.fold_sha = {}
        self.cache_sha = None
        self.order = {}
        self.cache_index = {}   # chiave -> (split, indice nella cache)
        self.z = None
        self.to_extract = []
        self.checks = {}
        self.code = {}
        self.environment = {}

    @property
    def subjects(self) -> set:
        return {r["subject"] for r in self.official.values()}


def counts_of(rows) -> dict:
    return {"rows": len(rows), "n_bona_fide": sum(1 for r in rows if r["label"] == "0"),
            "n_attack": sum(1 for r in rows if r["label"] == "1"), "n_subjects": len({r["subject"] for r in rows})}


def parse_manifest(b: bytes, what: str) -> list:
    rows = parse_csv(b, COLUMNS, what)
    for x in rows:
        if x["label"] not in LABELS or x["official_split"] not in ("train", "test") or x["session"] not in SESSIONS:
            fail(f"{what}: label, official_split or session out of range")
        m = NAME_RE.match(x["key"].split("/", 1)[1] if "/" in x["key"] else "")
        if not m or m.group(1) != x["subject"] or m.group(5) != x["session"] or x["key"].split("/", 1)[0] != KEY_DIR_OF_LABEL[x["label"]]:
            fail(f"{what}: a key does not carry the subject, session and class of its row")
        if not _hex(x["sha256"]) or not x["bytes"].isdigit():
            fail(f"{what}: sha256 or bytes malformed")
    return rows


def load_m2_manifests(cfg: dict, base: Path, inp: Inputs) -> None:
    """I cinque manifest M2: SHA-256 e conteggi uguali alla configurazione; liste ufficiali disgiunte per chiave e per
    contenuto; manifest della cache dentro le liste ufficiali, con le stesse righe (nuaa_subject_folds.load_manifests)."""
    mdir = rel_path(base, cfg["manifests"]["folder"])
    for k in MANIFESTS:
        spec = cfg["manifests"]["files"][k]
        p = mdir / spec["file"]
        b = read_regular(p, f"manifest {k}")
        h = sha256_bytes(b)
        if h != spec["sha256"]:
            fail(f"manifest {k} ({spec['file']}): SHA-256 {h} differs from the configuration ({spec['sha256']})")
        rows = parse_manifest(b, f"manifest {k}")
        c = counts_of(rows)
        for x, v in c.items():
            if spec[x] != v:
                fail(f"manifest {k}: {x} = {v} differs from the configuration ({spec[x]})")
        if any(r["session"] not in spec["sessions"] for r in rows):
            fail(f"manifest {k}: a row has a session outside the declared {spec['sessions']}")
        if len({r["key"] for r in rows}) != len(rows) or len({r["sha256"] for r in rows}) != len(rows):
            fail(f"manifest {k}: two rows share a key or a file SHA-256")
        inp.rows[k] = rows
    tr, te = inp.rows["official_train"], inp.rows["official_test"]
    if {r["key"] for r in tr} & {r["key"] for r in te} or {r["sha256"] for r in tr} & {r["sha256"] for r in te}:
        fail("the official lists share a key or a file SHA-256")
    if any(r["official_split"] != "train" for r in tr) or any(r["official_split"] != "test" for r in te):
        fail("official_split of the official lists differs from the list")
    inp.official = {r["key"]: r for r in tr + te}
    inp.by_key = {k: {r["key"]: r for r in inp.rows[k]} for k in MANIFESTS}
    for k in CACHE_MANIFESTS:
        for r in inp.rows[k]:
            o = inp.official.get(r["key"])
            if o is None or any(o[c] != r[c] for c in COLUMNS):
                fail(f"manifest {k}: a key is not in the official lists with the same values in every column")
    cache = [r["key"] for k in CACHE_MANIFESTS for r in inp.rows[k]]
    if len(cache) != len(set(cache)):
        fail("the cache manifests share keys")
    exp = cfg["expected"]
    if len(cache) != exp["n_cache_frames"]:
        fail(f"cache manifests: {len(cache)} frames, expected {exp['n_cache_frames']}")
    if len(inp.official) != exp["n_images"]:
        fail(f"official lists: {len(inp.official)} images, expected {exp['n_images']}")
    if len(inp.subjects) != exp["n_subjects"]:
        fail("official lists: number of subjects differs from the configuration")
    inp.checks["manifests"] = {k: {"sha256": cfg["manifests"]["files"][k]["sha256"], **counts_of(inp.rows[k])} for k in MANIFESTS}


def check_schema_config(cfg: dict, base: Path, inp: Inputs) -> None:
    """La configurazione dello schema approvato (PR #16): SHA-256 fissato nel codice e nella configurazione; seed, tetto,
    criterio della soglia, ricetta del training, numeri del bootstrap e conteggi di campionamento uguali a quelli di
    questo run."""
    sr = cfg["schema_run"]
    p = rel_path(base, sr["config_path"])
    b = read_regular(p, "schema configuration")
    h = sha256_bytes(b)
    if h != sr["config_sha256"] or h != SCHEMA_CONFIG_SHA256:
        fail(f"schema configuration {p.name}: SHA-256 {h} differs from the approved schema ({SCHEMA_CONFIG_SHA256[:12]}...)")
    try:
        sc = json.loads(b.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        fail(f"schema configuration {p.name} is not valid JSON: {e}")
    if sc.get("train_seeds") != cfg["train_seeds"]:
        fail("schema configuration: train_seeds differ from this run")
    if sc.get("budget", {}).get("wall_time_cap_minutes") != cfg["budget"]["wall_time_cap_minutes"]:
        fail("schema configuration: wall_time_cap_minutes differs from this run")
    if sc.get("threshold", {}).get("criterion") != cfg["threshold"]["criterion"] or sc["threshold"].get("set") != cfg["threshold"]["set"]:
        fail("schema configuration: threshold criterion or set differ from this run")
    tr = sc.get("training", {})
    diff = [k for k, v in FROZEN_TRAINING.items() if k != "device" and tr.get(k) != v]
    if diff or not str(tr.get("device", "")).startswith("CPU"):
        fail(f"schema configuration: training fields {diff or ['device']} differ from the recipe of this run")
    unc = sc.get("metrics", {}).get("uncertainty", "")
    for needle in (f"{BOOT['resamples']:,} resamples", f"random seed {BOOT['seed']}", f"percentile {BOOT['percentiles'][0]}-{BOOT['percentiles'][1]}",
                   f"fewer than {BOOT['min_subjects']} subjects"):
        if needle not in unc:
            fail(f"schema configuration: the bootstrap declaration does not state {needle!r}")
    sa, exp = sc.get("sampling", {}), cfg["expected"]
    cells = len(SESSIONS) * len(LABELS)
    if sa.get("fitting_per_cell", 0) * cells != exp["fitting_per_fold"] or sa.get("dev_per_cell", 0) * cells != exp["dev_per_fold"]:
        fail("schema configuration: fitting or dev counts per cell do not give the counts of this run")
    if sc.get("folds", {}).get("k") != exp["k"]:
        fail("schema configuration: number of folds differs from this run")
    inp.checks["schema_config"] = {"path": sr["config_path"], "sha256": h, "train_seeds": sc["train_seeds"],
                                   "wall_time_cap_minutes": sc["budget"]["wall_time_cap_minutes"],
                                   "fitting_per_cell": sa["fitting_per_cell"], "dev_per_cell": sa["dev_per_cell"], "k": exp["k"]}


def read_sums(b: bytes, what: str) -> dict:
    out = {}
    for line in b.decode("utf-8", errors="replace").splitlines():
        m = re.fullmatch(r"([0-9a-f]{64})  (\S.*)", line)
        if not m:
            fail(f"{what}: malformed line {line[:60]!r}")
        out[m.group(2)] = m.group(1)
    return out


def load_fold_manifests(cfg: dict, base: Path, inp: Inputs) -> None:
    """I dodici manifest dello schema accettato: SHA256SUMS riservato ancorato per SHA-256, ogni file uguale alla sua
    riga, conteggi della configurazione, fold e ruolo coerenti con il nome, ruoli disgiunti per chiave e per soggetto,
    dev del fold k = test del fold k mod K + 1, ogni riga uguale alla lista ufficiale, in_cache = chiave nei manifest
    della cache, i test dei fold partizionano le immagini ufficiali."""
    sr, exp = cfg["schema_run"], cfg["expected"]
    folder = rel_path(base, sr["folder"])
    sums_b = read_regular(folder / "SHA256SUMS", "schema SHA256SUMS")
    h = sha256_bytes(sums_b)
    if h != sr["restricted_sha256sums_sha256"]:
        fail(f"schema run {sr['run_id']}: SHA256SUMS of the restricted folder has SHA-256 {h}, expected {sr['restricted_sha256sums_sha256']}")
    listed = read_sums(sums_b, "schema SHA256SUMS")
    k = exp["k"]
    cache_keys = {r["key"] for n in CACHE_MANIFESTS for r in inp.rows[n]}
    test_subjects = {}
    for i in range(1, k + 1):
        inp.folds[i] = {}
        for role in ROLES:
            name = f"fold{i}_{role}.csv"
            if name not in listed:
                fail(f"schema SHA256SUMS: {name} not listed")
            b = read_regular(folder / name, f"fold manifest {name}")
            hh = sha256_bytes(b)
            if hh != listed[name]:
                fail(f"fold manifest {name}: SHA-256 {hh} differs from SHA256SUMS ({listed[name]})")
            inp.fold_sha[name] = hh
            rows = parse_csv(b, FOLD_COLUMNS, f"fold manifest {name}")
            want = {"fitting": exp["fitting_per_fold"], "dev": exp["dev_per_fold"], "test": exp["test_per_fold"][i - 1]}[role]
            if len(rows) != want:
                fail(f"fold manifest {name}: {len(rows)} rows, expected {want}")
            if len({r["key"] for r in rows}) != len(rows):
                fail(f"fold manifest {name}: duplicated keys")
            for r in rows:
                if r["fold"] != str(i) or r["role"] != role:
                    fail(f"fold manifest {name}: a row carries another fold or role")
                o = inp.official.get(r["key"])
                if o is None or any(o[c] != r[c] for c in ("label", "subject", "session", "sha256", "bytes")):
                    fail(f"fold manifest {name}: a key is not in the official lists with the same label, subject, session, SHA-256 and size")
                if r["in_cache"] not in ("0", "1") or (r["in_cache"] == "1") != (r["key"] in cache_keys):
                    fail(f"fold manifest {name}: in_cache does not match the cache manifests")
            inp.folds[i][role] = rows
        subj = {role: {r["subject"] for r in inp.folds[i][role]} for role in ROLES}
        keys = {role: {r["key"] for r in inp.folds[i][role]} for role in ROLES}
        for a, b_ in (("fitting", "dev"), ("fitting", "test"), ("dev", "test")):
            if subj[a] & subj[b_] or keys[a] & keys[b_]:
                fail(f"fold {i}: {a} and {b_} share a subject or a key")
        test_subjects[i] = subj["test"]
    for i in range(1, k + 1):
        nxt = i % k + 1
        if {r["subject"] for r in inp.folds[i]["dev"]} != test_subjects[nxt]:
            fail(f"fold {i}: the dev subjects are not the test subjects of fold {nxt}")
        fit = {r["subject"] for r in inp.folds[i]["fitting"]}
        others = set().union(*(test_subjects[j] for j in range(1, k + 1) if j not in (i, nxt)))
        if fit != others:
            fail(f"fold {i}: the fitting subjects are not the subjects of the other folds")
    # ridondante per costruzione (ogni altro fold sta nel fitting o nel dev del fold k, quindi una chiave estranea
    # è già rifiutata come condivisa): tenuto come dichiarazione esplicita della proprietà
    all_test = Counter(r["key"] for i in inp.folds for r in inp.folds[i]["test"])
    if set(all_test) != set(inp.official) or any(v != 1 for v in all_test.values()):
        fail("the test folds do not partition the official images (every image held out exactly once)")
    subjects = {i: sorted(test_subjects[i]) for i in test_subjects}
    if sum(len(v) for v in subjects.values()) != exp["n_subjects"]:
        fail("the folds do not partition the subjects")
    inp.checks["schema"] = {"run_id": sr["run_id"], "folds": k, "subjects_per_fold": {i: len(v) for i, v in subjects.items()},
                            "fitting_per_fold": exp["fitting_per_fold"], "dev_per_fold": exp["dev_per_fold"],
                            "test_per_fold": exp["test_per_fold"], "restricted_sha256sums_sha256": h,
                            "in_cache": {f"fold{i}_{role}": sum(1 for r in inp.folds[i][role] if r["in_cache"] == "1")
                                         for i in inp.folds for role in ROLES}}
    inp.to_extract = sorted(k_ for k_ in inp.official if k_ not in cache_keys)
    if len(inp.to_extract) != exp["n_to_extract"]:
        fail(f"{len(inp.to_extract)} frames to extract, expected {exp['n_to_extract']}")


def npy_header(fh, what: str) -> tuple:
    fmt = npy_format
    ver = fmt.read_magic(fh)
    if ver not in NPY_HEADERS:
        fail(f"{what}: .npy format version {ver[0]}.{ver[1]} not supported (1.0 or 2.0 expected)")
    return getattr(fmt, NPY_HEADERS[ver])(fh)


def npz_members(p: Path) -> dict:
    """Membri di un .npz dalle intestazioni .npy (tipo e forma), senza caricare gli array; nessun pickle."""
    out = {}
    try:
        with zipfile.ZipFile(p) as zf:
            names = zf.namelist()
            for n in names:
                if not n.endswith(".npy"):
                    fail(f"{p.name}: member {n!r} is not an .npy array")
                with zf.open(n) as fh:
                    shape, fortran, dtype = npy_header(fh, f"{p.name}: member {n!r}")
                if dtype.hasobject:
                    fail(f"{p.name}: member {n!r} holds Python objects (pickle refused)")
                out[n[:-4]] = {"dtype": dtype.str, "shape": list(shape), "fortran": bool(fortran)}
            if len(set(names)) != len(names):
                fail(f"{p.name}: duplicated members")
    except (OSError, zipfile.BadZipFile, ValueError) as e:
        fail(f"{p.name} unreadable ({e})")
    return out


def load_cache(cfg: dict, base: Path, inp: Inputs) -> None:
    """Cache di training: SHA-256, schema, ordine di campionamento (spread per classe) e collegamento etichette e
    soggetti con i manifest; indice chiave -> (split, posizione)."""
    c = cfg["cache"]
    cp = regular_path(rel_path(base, c["path"]), "cache")
    inp.cache_sha = sha256_file(cp)
    if inp.cache_sha != c["sha256"]:
        fail(f"cache {cp.name}: SHA-256 {inp.cache_sha} differs from the expected {c['sha256']}")
    mem = npz_members(cp)
    if set(mem) != set(MEMBERS):
        fail(f"cache {cp.name}: members {sorted(mem)} differ from {sorted(MEMBERS)}")
    for k, want in c["members"].items():
        got = {"dtype": mem[k]["dtype"], "shape": mem[k]["shape"]}
        if got != want or mem[k]["fortran"]:
            fail(f"cache {cp.name}: {k} has dtype {got['dtype']}, shape {got['shape']}"
                 f"{' (Fortran order)' if mem[k]['fortran'] else ''}, expected {want['dtype']}, {want['shape']}")
    if tuple(mem["X_train"]["shape"][1:]) != FRAME_SHAPE or tuple(mem["X_test"]["shape"][1:]) != FRAME_SHAPE:
        fail(f"cache {cp.name}: frames are not {FRAME_SHAPE}")
    order_train = spread_keys(inp.rows["official_train"], c["per_class"])
    order_test = spread_keys(inp.rows["official_test"], c["per_class"])
    if sorted(order_train) != sorted(list(inp.by_key["cnn_train"]) + list(inp.by_key["cnn_es"])):
        fail("the sampling order of the official train list does not give cnn_train + cnn_es")
    if sorted(order_test) != sorted(inp.by_key["cnn_test"]):
        fail("the sampling order of the official test list does not give cnn_test")
    inp.order = {"train": order_train, "test": order_test}
    z = np.load(cp, allow_pickle=False)
    for split in ("train", "test"):
        keys = inp.order[split]
        y, s = z[f"y_{split}"], z[f"s_{split}"]
        exp_y = np.array([int(inp.official[q]["label"]) for q in keys], np.int64)
        exp_s = np.array([inp.official[q]["subject"] for q in keys])
        if len(y) != len(keys) or y.dtype.kind not in "iu" or not np.array_equal(y, exp_y):
            fail(f"cache y_{split}: labels differ from the manifests in sampling order (linkage refused)")
        if len(s) != len(keys) or s.dtype.kind != "U" or not np.array_equal(s, exp_s):
            fail(f"cache s_{split}: subjects differ from the manifests in sampling order (linkage refused)")
        for i, q in enumerate(keys):
            inp.cache_index[q] = (split, i)
    inp.z = z
    inp.checks["cache"] = {"sha256": inp.cache_sha, "frames": len(inp.cache_index),
                           "members": {k: {"dtype": v["dtype"], "shape": v["shape"]} for k, v in sorted(mem.items())},
                           "linkage": "y/s equal to label and subject of the manifest rows in sampling order (spread per class)"}


def frozen_code_checks(cfg: dict, base: Path, inp: Inputs) -> None:
    """Estrattore congelato, risorse del rilevatore e codice di riferimento: SHA-256 uguali alla configurazione."""
    ex = cfg["extraction"]
    out = {}
    ff = rel_path(base, ex["frozen_face"]["path"])
    h = sha256_bytes(read_regular(ff, "frozen face module"))
    if h != ex["frozen_face"]["sha256"]:
        fail(f"frozen face module {ff.name}: SHA-256 {h} differs from the configuration")
    out["frozen_face"] = h
    for group in ("detector_resources", "reference_code"):
        for k, v in ex[group].items():
            p = resource_path(v["path"], base)
            hh = sha256_bytes(read_regular(p, f"{group} {k}"))
            if hh != v["sha256"]:
                fail(f"{group} {k} ({p.name}): SHA-256 {hh} differs from the configuration")
            out[f"{group}.{k}"] = hh
    for f in CODE_FILES:
        out[f] = sha256_bytes(read_regular(ROOT / f, "code file"))
    inp.code = out
    inp.checks["frozen_code"] = {k: v for k, v in out.items() if not k.startswith("scripts/")}


def resource_path(spec: str, base: Path) -> Path:
    """Percorso di una risorsa: relativo alla radice degli ingressi, oppure «cv2.data.haarcascades/<nome>»."""
    if spec.startswith("cv2.data.haarcascades/"):
        try:
            import cv2
            return Path(cv2.data.haarcascades) / spec.split("/", 1)[1]
        except (ImportError, AttributeError):
            fail("OpenCV (cv2.data.haarcascades) is needed to locate the Haar cascade")
    return rel_path(base, spec)


def environment(cfg: dict = None) -> dict:
    out = {"python": platform.python_version(), "numpy": np.__version__, "platform": sys.platform}
    try:
        import importlib.metadata as md
        for p in ("tensorflow", "tensorflow-metal", "keras"):      # le versioni pip di OpenCV (quattro numeri) restano fuori dal pubblico
            try:
                out[p] = md.version(p)
            except md.PackageNotFoundError:
                out[p] = "absent"
    except ImportError:
        pass
    try:
        import cv2
        out["opencv"] = str(cv2.__version__)
    except (ImportError, AttributeError):
        out["opencv"] = "absent"
    try:
        out["tensorflow_installed"] = importlib.metadata.version("tensorflow")
    except importlib.metadata.PackageNotFoundError:
        out["tensorflow_installed"] = "absent"
    if cfg is not None:
        if out["opencv"] != cfg["environment"]["opencv"]:
            fail(f"OpenCV {out['opencv']} differs from the configuration ({cfg['environment']['opencv']}): the cache would not reproduce")
        if out["tensorflow_installed"] != cfg["environment"]["tensorflow"]:
            fail(f"TensorFlow {out['tensorflow_installed']} installed, configuration {cfg['environment']['tensorflow']} (checked without importing it)")
    return out


def tf_state() -> str:
    """Per i passi che non devono usare TensorFlow: lo stato è accertato, non dichiarato."""
    if "tensorflow" in sys.modules:
        fail("TensorFlow was imported in a step that must not use it")
    return "not imported"


def check_originals(cfg: dict, base: Path, inp: Inputs) -> None:
    """Ogni originale delle liste ufficiali: presente, file regolare, dimensione e SHA-256 della riga."""
    images, layout = rel_path(base, cfg["images"]["folder"]), cfg["images"]["layout"]
    if images.is_symlink() or not images.is_dir():
        fail(f"images folder {images.name} missing, or a symbolic link")
    n = 0
    for i, key in enumerate(sorted(inp.official)):
        locate(images, inp.official[key], layout, i)
        n += 1
    inp.checks["originals"] = {"located_and_hashed": n, "layout": layout}


def load_inputs(cfg: dict, base: Path) -> Inputs:
    refused = [v for v in ENV_REFUSED if os.environ.get(v, "").strip()]
    if refused:
        fail(f"environment variables {refused} are set: they would redirect the detector resources or the reference code (unset them)")
    inp = Inputs()
    load_m2_manifests(cfg, base, inp)
    load_fold_manifests(cfg, base, inp)
    check_schema_config(cfg, base, inp)
    load_cache(cfg, base, inp)
    frozen_code_checks(cfg, base, inp)
    inp.environment = environment(cfg)
    inp.checks["environment"] = {**inp.environment, "tensorflow_state": tf_state()}
    check_originals(cfg, base, inp)
    frames_path = rel_path(base, cfg["extraction"]["frames_file"])
    if frames_path.exists() or frames_path.is_symlink():
        fail(f"{frames_path.name} already exists in {frames_path.parent.name}: a previous extraction is in the way; move it away "
             "(the run never overwrites it)")
    return inp


# ============================================================================ originali e pre-elaborazione congelata

LAYOUTS = {"images": lambda r: f"{'bonafide' if r['label'] == '0' else 'attack'}_{'training' if r['official_split'] == 'train' else 'validation'}/"
                               f"{r['key'].split('/', 1)[1]}",
           "raw": lambda r: f"{'ClientRaw' if r['label'] == '0' else 'ImposterRaw'}/{r['subject']}/{r['key'].split('/', 1)[1].split('_', 1)[1]}"}


def locate(images: Path, r: dict, layout: str, index=None) -> Path:
    """Il file originale di una riga: accettato solo se file regolare con SHA-256 e dimensione uguali alla riga. I
    messaggi non portano identificativi (finiscono in status.json): solo la posizione nell'elenco ordinato delle chiavi."""
    p = images / LAYOUTS[layout](r)
    what = f"original image {index if index is not None else '?'} of the sorted official list (session {r['session']})"
    if p.is_symlink():
        fail(f"{what}: symbolic link refused")
    if not p.is_file():
        fail(f"{what}: not found under the images folder")
    if p.stat().st_size != int(r["bytes"]) or sha256_file(p) != r["sha256"]:
        fail(f"{what}: size or SHA-256 differ from the manifest")
    return p


def load_module(name: str, path: Path):
    """Carica un file Python come modulo registrato in sys.modules (una sola volta per processo)."""
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


class RecordingDetector:
    """Il rilevatore congelato, con l'ultimo esito registrato (quale rilevatore ha trovato il volto)."""

    def __init__(self, det):
        self.det, self.last = det, None

    def detect(self, img, *args, **kwargs):
        self.last = self.det.detect(img, *args, **kwargs)
        return self.last


def detector_short(face) -> str:
    return "none" if face is None else str(face.detector).split()[0]


def default_preprocessor(cfg: dict, base: Path):
    """train_cnn.preprocess (invariato) con il rilevatore congelato e l'enhancement del codice di riferimento."""
    try:
        import train_cnn as tc
    except ModuleNotFoundError as e:
        fail(f"missing Python package {e.name!r}: the preprocessing needs OpenCV")
    ex = cfg["extraction"]
    ff = rel_path(base, ex["frozen_face"]["path"])
    if sha256_file(ff) != ex["frozen_face"]["sha256"]:
        fail(f"frozen face module {ff.name}: SHA-256 changed just before loading")
    det = RecordingDetector(load_module("tesi_app.face_0ac7143", ff).FaceDetector())
    cd = load_module("kuznetsov_create_datasets", rel_path(base, ex["reference_code"]["create_datasets"]["path"]))
    creator = cd.EnhancedDatasetCreator.__new__(cd.EnhancedDatasetCreator)   # istanza vuota: serve solo advanced_image_enhancement

    def pre(path):
        det.last = None
        arr = tc.preprocess(Path(path), det, creator)
        if arr is None:
            fail(f"{Path(path).name}: unreadable or empty crop")
        return arr, detector_short(det.last)
    return pre


def preprocessor(cfg: dict, base: Path):
    return (PREPROCESSOR or default_preprocessor)(cfg, base)


def check_frame(arr, what: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape != FRAME_SHAPE or arr.dtype != np.uint8:
        fail(f"{what}: frame has shape {arr.shape} {arr.dtype}, expected {FRAME_SHAPE} uint8")
    return arr


def cache_check(cfg: dict, base: Path, inp: Inputs, pre) -> dict:
    """Ogni frame della cache rifatto dall'originale con l'estrattore congelato e confrontato bit per bit (N2)."""
    images, layout = rel_path(base, cfg["images"]["folder"]), cfg["images"]["layout"]
    result, hashes = {}, {}
    for split in ("train", "test"):
        X = inp.z[f"X_{split}"]
        hashes[split] = frame_hashes(X)
        equal = different = worst = 0
        dets = Counter()
        for i, key in enumerate(inp.order[split]):
            r = inp.official[key]
            arr, det = pre(locate(images, r, layout, i))
            arr = check_frame(arr, f"reproduced X_{split}[{i}]")
            dets[det] += 1
            diff = int(np.abs(arr.astype(np.int16) - X[i].astype(np.int16)).max())
            if diff == 0:
                equal += 1
            else:
                different += 1
                worst = max(worst, diff)
                print(f"   DIFF X_{split}[{i}]: max abs difference {diff}", flush=True)
        result[split] = {"compared": equal + different, "equal": equal, "different": different, "worst_max_abs_diff": worst,
                         "detectors": dict(sorted(dets.items())), "duplicates_within": len(hashes[split]) - len(set(hashes[split]))}
        print(f"   X_{split}: {equal} equal, {different} different, worst max abs diff {worst}, detectors {dict(sorted(dets.items()))}", flush=True)
        if different or result[split]["duplicates_within"]:
            fail(f"cache check: X_{split} has {different} frames that differ from the frozen preprocessing (worst max abs diff {worst}) and "
                 f"{result[split]['duplicates_within']} duplicated frames")
        if result[split]["detectors"] != cfg["cache"]["detectors_n2"][split]:
            fail(f"cache check: detectors of X_{split} {result[split]['detectors']} differ from the N2 cache check "
                 f"({cfg['cache']['detectors_n2'][split]})")
    shared = len(set(hashes["train"]) & set(hashes["test"]))
    if shared:
        fail(f"cache check: {shared} frames are shared between X_train and X_test (exact pixel hash)")
    if sum(result[s]["compared"] for s in result) != len(inp.cache_index):
        fail("cache check: not every cache frame was compared")
    return {"rule": "every cache frame reproduced bit for bit from its original with the frozen extractor; detectors equal to the N2 "
                    "cache check; no frame shared between X_train and X_test", "cache_train_test_shared": shared, **result}


def extract_frames(cfg: dict, base: Path, inp: Inputs, pre, frames_path: Path) -> tuple:
    """I frame fuori dalla cache estratti una volta e salvati in frames_path (chiavi ordinate, uint8); (record, righe)."""
    if frames_path.exists() or frames_path.is_symlink():
        fail(f"{frames_path.name} already exists: extracted frames are never overwritten (move it away first)")
    if msg := mw.output_problem(frames_path.parent, ROOT, pg.BASE_PROTECTED, [ROOT / "data" / "processed"], [], None, [frames_path.name]):
        fail(f"frames folder refused: {msg}")
    images, layout = rel_path(base, cfg["images"]["folder"]), cfg["images"]["layout"]
    keys = list(inp.to_extract)
    X = np.zeros((len(keys),) + FRAME_SHAPE, np.uint8)
    rows, dets = [], Counter()
    cache_subject = {}
    for split in ("train", "test"):
        for h, key in zip(frame_hashes(inp.z[f"X_{split}"]), inp.order[split]):
            cache_subject.setdefault(h, set()).add(inp.official[key]["subject"])
    for i, key in enumerate(keys):
        r = inp.official[key]
        arr, det = pre(locate(images, r, layout, i))
        X[i] = check_frame(arr, f"extracted frame {i}")
        dets[det] += 1
        rows.append([key, r["subject"], r["session"], r["label"], r["official_split"], det, r["sha256"], frame_hashes(X[i:i + 1])[0]])
        if (i + 1) % 500 == 0:
            print(f"   extracted {i + 1}/{len(keys)}", flush=True)
    # frame identici (hash dei pixel): fra soggetti diversi è un errore (un volto non può essere di due persone), dentro
    # lo stesso soggetto è registrato (fotogrammi consecutivi di una cattura possono coincidere)
    by_hash = defaultdict(set)
    for x in rows:
        by_hash[x[-1]].add(x[1])
    across = sum(1 for h, subs in by_hash.items() if len(subs) > 1 or (h in cache_subject and cache_subject[h] != subs))
    within_same = sum(1 for x in rows if len([y for y in rows if y[-1] == x[-1]]) > 1 or x[-1] in cache_subject)
    dup = {"identical_frames_same_subject": within_same, "identical_frames_across_subjects_or_with_the_cache_of_another_subject": across}
    if across:
        fail(f"extracted frames: {across} identical frames across different subjects (or identical to a cache frame of another subject)")
    frames_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tp = tempfile.mkstemp(prefix=f".{frames_path.name}.", suffix=".tmp", dir=frames_path.parent)
    os.close(fd)
    try:
        with open(tp, "wb") as f:
            np.savez(f, X=X, keys=np.array(keys))
        os.replace(tp, frames_path)
    except BaseException:
        Path(tp).unlink(missing_ok=True)
        raise
    record = {"n": len(keys), "file": frames_path.name, "sha256": sha256_file(frames_path), "detectors": dict(sorted(dets.items())),
              "identical_frames": dup, "members": {"X": {"dtype": "|u1", "shape": list(X.shape)}, "keys": {"shape": [len(keys)]}}}
    return record, rows, X


class FrameStore:
    """Frame per chiave: dalla cache (per indice di campionamento) o dall'array dei frame estratti."""

    def __init__(self, inp: Inputs, X_extracted: np.ndarray, extracted_keys):
        self.X = {"train": inp.z["X_train"], "test": inp.z["X_test"], "extracted": X_extracted}
        self.index = dict(inp.cache_index)
        for i, k in enumerate(extracted_keys):
            self.index[k] = ("extracted", i)

    def get(self, key: str) -> np.ndarray:
        split, i = self.index[key]
        return self.X[split][i]


def assemble(store: FrameStore, rows) -> dict:
    """Array di un ruolo nell'ordine delle righe del manifest (ordinate per chiave): X uint8, y int64, s, chiavi."""
    rows = sorted(rows, key=lambda r: r["key"])
    X = np.stack([store.get(r["key"]) for r in rows]) if rows else np.zeros((0,) + FRAME_SHAPE, np.uint8)
    y = np.array([int(r["label"]) for r in rows], np.int64)
    s = np.array([r["subject"] for r in rows])
    return {"X": X, "y": y, "s": s, "keys": [r["key"] for r in rows], "rows": rows}


# ============================================================================ training (ricetta N1, fitting e dev dati)

def default_arch_loader(path: Path):
    return load_module("kuznetsov_architectures", path)


def hide_gpu(tf) -> list:
    """Solo CPU: su GPU Metal, in grafo, la ReLU del Dense dopo il Flatten di LivenessNet viene saltata (N1)."""
    try:
        tf.config.set_visible_devices([], "GPU")
    except (RuntimeError, ValueError) as e:
        fail(f"the GPU cannot be hidden ({e}): TensorFlow was initialised before the trainer")
    devices = sorted({d.device_type for d in tf.config.get_visible_devices()})
    if devices != ["CPU"]:
        fail(f"visible devices {devices}, CPU only expected")
    tf.config.optimizer.set_experimental_options({"remapping": False})
    return devices


def default_trainer(data: dict, training: dict, weights_path: Path, train_seed: int, arch_module, cap_check) -> dict:
    """La ricetta di n1_trainer.train, riga per riga, con fitting e dev già separati (nessuno split interno); il
    checkpoint è l'epoca con val_loss minimo sul dev; cap_check() è chiamata alla fine di ogni epoca (tetto di tempo)."""
    if set(data) != {"X_fit", "y_fit", "X_dev", "y_dev"}:
        fail(f"the trainer receives X_fit, y_fit, X_dev and y_dev, got {sorted(data)}")
    if weights_path.exists() or weights_path.is_symlink():
        fail(f"{weights_path.name} already exists (never overwritten)")
    import random
    import tensorflow as tf
    if training["device"] != "CPU":
        fail(f"unknown training device {training['device']!r}")
    devices = hide_gpu(tf)
    tf.keras.backend.clear_session()                 # memoria: grafo del modello precedente liberato (non cambia la ricetta)
    random.seed(train_seed); np.random.seed(train_seed); tf.random.set_seed(train_seed)
    model = arch_module.create_model("LivenessNet").get_model()
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

    class Cap(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            cap_check(f"epoch {epoch + 1}")
    cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6), Cap()]
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
    tmp = weights_path.with_name(f".{weights_path.name}.{os.getpid()}.tmp.keras")
    try:
        model.save(tmp)
        os.replace(tmp, weights_path)                 # scrittura atomica del checkpoint
    except BaseException:
        tmp.unlink(missing_ok=True)                   # nessun temporaneo resta in restricted/ dopo un errore
        raise
    card = {"arch": "LivenessNet", "train_seed": int(train_seed), "n_fitting": int(len(ytr)), "n_dev": int(len(yva)),
            "max_epochs": int(epochs), "epochs_run": len(h["loss"]), "restored_epoch": best + 1,
            "restored_by": "early stopping" if es.stopped_epoch else "end of epochs (set_weights)",
            "best_val_loss": float(h["val_loss"][best]), "val_acc_at_best": float(h["val_accuracy"][best]),
            "history": {k: [float(v) for v in h[k]] for k in ("loss", "accuracy", "val_loss", "val_accuracy", "lr") if k in h},
            "device": "CPU", "visible_devices": devices, "lr": lr, "batch": batch, "early_stopping_patience": patience,
            "class_weight": {str(k): float(v) for k, v in cw.items()}, "train_seconds": round(train_s, 1),
            "test_evaluation": "none (scores computed afterwards on CPU)", "cpu_nondeterminism": "declared (N1): parallel reductions on the CPU "
            "may give minimal differences between two runs with the same seed", "weights_file": weights_path.name,
            "weights_sha256": sha256_file(weights_path), "saved": now()}
    print(f"== LivenessNet seed {train_seed}: {card['epochs_run']} epochs in {train_s:.0f}s, restored epoch {card['restored_epoch']} "
          f"(val_loss {card['best_val_loss']:.6f})", flush=True)
    return card


def card_problems(card: dict, training: dict, seed: int, weights_path: Path) -> list:
    """Scheda del training: campi, seed, iperparametri della ricetta, epoca ripristinata = minimo del val_loss, pesi."""
    out = []
    need = ("arch", "train_seed", "n_fitting", "n_dev", "max_epochs", "epochs_run", "restored_epoch", "restored_by", "best_val_loss",
            "history", "device", "visible_devices", "lr", "batch", "early_stopping_patience", "train_seconds", "weights_file", "weights_sha256")
    for k in need:
        if k not in card:
            out.append(f"card without {k}")
    if out:
        return out
    if card["arch"] != "LivenessNet" or card["train_seed"] != seed or card["device"] != "CPU" or card["visible_devices"] != ["CPU"]:
        out.append("architecture, seed or device differ from the protocol")
    if (card["max_epochs"], card["lr"], card["batch"], card["early_stopping_patience"]) != \
            (training["max_epochs"], training["initial_learning_rate"], training["batch"], training["early_stopping_patience"]):
        out.append("hyperparameters differ from the recipe")
    h = card["history"]
    vl = h.get("val_loss") or []
    if not vl or len(vl) != card["epochs_run"] or card["epochs_run"] > card["max_epochs"]:
        out.append("history length differs from epochs_run, or more epochs than allowed")
    elif card["restored_epoch"] != int(np.argmin(vl)) + 1 or card["best_val_loss"] != float(vl[card["restored_epoch"] - 1]):
        out.append("restored epoch is not the epoch of minimum val_loss")
    if card["weights_file"] != weights_path.name or not weights_path.is_file() or sha256_file(weights_path) != card["weights_sha256"]:
        out.append("weights file missing or its SHA-256 differs from the card")
    return out


# ============================================================================ scoring, soglia, metriche

def tf_scorer(weights_path, weights_sha: str):
    """Scorer vero: il modello salvato (.keras, soli strati standard), pesi verificati per SHA-256, CPU, BatchNorm in
    inferenza (training=False); probabilità softmax N x 2."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        import tensorflow as tf
    except ModuleNotFoundError as e:
        fail(f"missing Python package {e.name!r}: scoring needs TensorFlow")
    hide_gpu(tf)
    if sha256_file(weights_path) != weights_sha:
        fail(f"{Path(weights_path).name}: SHA-256 changed just before loading")
    with tf.device("/CPU:0"):
        model = tf.keras.models.load_model(str(weights_path), compile=False)

    def score(X):
        out = []
        for i in range(0, len(X), SCORE_BATCH):
            xb = X[i:i + SCORE_BATCH].astype(np.float32) / 255.0
            with tf.device("/CPU:0"):
                out.append(np.asarray(model(xb, training=False), dtype=np.float64))
        return np.concatenate(out) if out else np.zeros((0, 2))
    return score


def probabilities(scorer, X) -> np.ndarray:
    p = np.asarray(scorer(X), dtype=np.float64)
    if p.shape != (len(X), 2) or not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        fail(f"the scorer returned {p.shape} values, non-finite or outside [0, 1] ({len(X)} x 2 probabilities expected)")
    return p


def val_loss(p: np.ndarray, y: np.ndarray) -> float:
    """Categorical crossentropy media come Keras (probabilità normalizzate e limitate a [1e-7, 1 - 1e-7])."""
    q = p / p.sum(axis=1, keepdims=True)
    q = np.clip(q, EPS, 1 - EPS)
    return float(np.mean(-np.log(q[np.arange(len(y)), y])))


def candidate_thresholds(s) -> list:
    """select_threshold.candidate_thresholds: punteggi distinti in ordine decrescente più nextafter(min, -inf) se >= 0."""
    vals = sorted({float(v) for v in s}, reverse=True)
    low = float(np.nextafter(vals[-1], -np.inf))
    if low >= 0.0:
        vals.append(low)
    return vals


def eer_threshold(y, s) -> dict:
    """select_threshold.select_threshold(y, s, "eer"), copiato: minimo di |fp·n_a − fn·n_b| sui conteggi interi con la
    regola attacco se s > t (pareggio = bona fide), a pari merito la soglia più alta."""
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    att, bf = np.sort(s[y == 1]), np.sort(s[y == 0])
    n_a, n_b = int(len(att)), int(len(bf))
    if n_a == 0 or n_b == 0:
        fail("both classes are required to select the dev threshold")
    best = None
    for t in candidate_thresholds(s):
        fn = int(np.searchsorted(att, t, side="right"))
        fp = n_b - int(np.searchsorted(bf, t, side="right"))
        key = abs(fp * n_a - fn * n_b)
        if best is None or key < best[0]:
            best = (key, t, fp, fn)
    return {"value": best[1], "fp": best[2], "fn": best[3], "n_attack": n_a, "n_bona_fide": n_b, "objective": best[0]}


def auroc(att, bf):
    """AUROC di score_attack come separatore (attacco positivo), Mann-Whitney con i pareggi a 0,5; None senza una classe."""
    att, bf = np.asarray(att, float), np.asarray(bf, float)
    if len(att) == 0 or len(bf) == 0:
        return None
    allv = np.concatenate([att, bf])
    order = np.argsort(allv, kind="mergesort")
    sorted_v = allv[order]
    ranks = np.empty(len(allv), float)
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2          # rango medio (1-based)
        i = j + 1
    u = ranks[:len(att)].sum() - len(att) * (len(att) + 1) / 2
    return float(u / (len(att) * len(bf)))


def stratum_metrics(rows) -> dict:
    """Metriche di un insieme di righe di punteggio: alle due soglie (0,5 e dev congelata del fold) e AUROC."""
    y = np.array([int(r["label"]) for r in rows], int)
    s = np.array([float(r["score_attack"]) for r in rows], float)
    out = {"n": len(rows), "n_bona_fide": int((y == 0).sum()), "n_attack": int((y == 1).sum()),
           "n_subjects": len({r["subject_id"] for r in rows})}
    pred05 = np.array([int(r["prediction_05"]) for r in rows], int)
    preddev = np.array([int(r["prediction_dev"]) for r in rows], int)
    for name, pred in (("05", pred05), ("dev", preddev)):
        accepted, rejected = int(np.sum((y == 1) & (pred == 0))), int(np.sum((y == 0) & (pred == 1)))
        apcer = accepted / out["n_attack"] if out["n_attack"] else None
        bpcer = rejected / out["n_bona_fide"] if out["n_bona_fide"] else None
        out[f"accepted_attacks_{name}"], out[f"rejected_bona_fide_{name}"] = accepted, rejected
        out[f"apcer_{name}"], out[f"bpcer_{name}"] = apcer, bpcer
        out[f"acer_{name}"] = (apcer + bpcer) / 2 if apcer is not None and bpcer is not None else None
    out["auroc"] = auroc(s[y == 1], s[y == 0])
    return out


def rows_of_stratum(rows, stratum: str) -> list:
    return list(rows) if stratum == "all" else [r for r in rows if r["session_id"] == stratum]


def summary(values: list) -> dict:
    if any(v is None for v in values):
        return {"values": values, "median": None, "min": None, "max": None}
    return {"values": values, "median": float(statistics.median(values)), "min": float(min(values)), "max": float(max(values))}


def bootstrap(rows, keys) -> dict:
    """Intervallo per soggetto delle metriche aggregate di uno strato: BOOT["resamples"] ricampionamenti con
    reinserimento dei soggetti (seme BOOT["seed"], gli stessi indici per ogni metrica), percentili BOOT["percentiles"];
    nessun intervallo con meno di BOOT["min_subjects"] soggetti; un ricampionamento senza una classe non conta."""
    subjects = sorted({r["subject_id"] for r in rows})
    out = {"n_subjects": len(subjects), "resamples": BOOT["resamples"], "seed": BOOT["seed"], "percentiles": BOOT["percentiles"]}
    if len(subjects) < BOOT["min_subjects"]:
        out["note"] = f"no interval: fewer than {BOOT['min_subjects']} subjects"
        for k in keys:
            out[k] = {"low": None, "high": None, "valid_resamples": 0}
        return out
    by_subject = defaultdict(list)
    for r in rows:
        by_subject[r["subject_id"]].append(r)
    rng = np.random.default_rng(BOOT["seed"])
    vals = {k: [] for k in keys}
    for _ in range(BOOT["resamples"]):
        idx = rng.integers(0, len(subjects), len(subjects))
        sample = [r for i in idx for r in by_subject[subjects[i]]]
        m = stratum_metrics(sample)
        for k in keys:
            if m[k] is not None:
                vals[k].append(m[k])
    for k in keys:
        if vals[k]:
            lo, hi = np.percentile(vals[k], BOOT["percentiles"])
            out[k] = {"low": float(lo), "high": float(hi), "valid_resamples": len(vals[k])}
        else:
            out[k] = {"low": None, "high": None, "valid_resamples": 0}
    return out


# ============================================================================ aggregato

def per_subject_rows(fold: int, seed: int, rows) -> list:
    out = []
    for subj in sorted({r["subject_id"] for r in rows}):
        sel = [r for r in rows if r["subject_id"] == subj]
        for stratum in STRATA:
            part = rows_of_stratum(sel, stratum)
            if not part:
                continue
            m = stratum_metrics(part)
            out.append([fold, seed, subj, stratum, m["n_bona_fide"], m["n_attack"], m["rejected_bona_fide_05"], m["accepted_attacks_05"],
                        fmt(m["apcer_05"]), fmt(m["bpcer_05"]), fmt(m["acer_05"]), m["rejected_bona_fide_dev"], m["accepted_attacks_dev"],
                        fmt(m["apcer_dev"]), fmt(m["bpcer_dev"]), fmt(m["acer_dev"]), fmt(m["auroc"])])
    return out


def aggregate(cfg: dict, scores: dict, cards: dict, thresholds: dict) -> tuple:
    """scores[(fold, seed)] = righe del test; (documento aggregato pubblico, righe per soggetto riservate)."""
    seeds, k = cfg["train_seeds"], cfg["expected"]["k"]
    per = {}
    subject_rows = []
    for fold in range(1, k + 1):
        per[fold] = {}
        for seed in seeds:
            rows = scores[(fold, seed)]
            per[fold][seed] = {st: (stratum_metrics(rows_of_stratum(rows, st)) if rows_of_stratum(rows, st) else None) for st in STRATA}
            subject_rows += per_subject_rows(fold, seed, rows)
    fold_summary = {}
    for fold in range(1, k + 1):
        fold_summary[fold] = {}
        for st in STRATA:
            ms = [per[fold][seed][st] for seed in seeds]
            if any(m is None for m in ms):
                fold_summary[fold][st] = None
                continue
            fold_summary[fold][st] = {key: summary([m[key] for m in ms]) for key in METRIC_KEYS}
    pooled, boot = {}, {}
    for seed in seeds:
        rows = [r for fold in range(1, k + 1) for r in scores[(fold, seed)]]
        pooled[seed] = {st: stratum_metrics(rows_of_stratum(rows, st)) for st in STRATA if rows_of_stratum(rows, st)}
        boot[seed] = {st: bootstrap(rows_of_stratum(rows, st), ("acer_dev", "acer_05", "auroc")) for st in pooled[seed]}
    pooled_summary = {st: {key: summary([pooled[seed][st][key] for seed in seeds]) for key in METRIC_KEYS}
                      for st in STRATA if all(st in pooled[seed] for seed in seeds)}
    training = {f"fold{fold}_seed{seed}": {"epochs_run": cards[(fold, seed)]["epochs_run"], "restored_epoch": cards[(fold, seed)]["restored_epoch"],
                                           "restored_by": cards[(fold, seed)]["restored_by"], "best_val_loss": cards[(fold, seed)]["best_val_loss"],
                                           "train_seconds": cards[(fold, seed)]["train_seconds"], "n_fitting": cards[(fold, seed)]["n_fitting"],
                                           "n_dev": cards[(fold, seed)]["n_dev"], "weights_sha256": cards[(fold, seed)]["weights_sha256"],
                                           "dev_threshold": thresholds[(fold, seed)]["value"], "dev_threshold_sha256": thresholds[(fold, seed)]["sha256"],
                                           "dev_apcer_at_threshold": thresholds[(fold, seed)]["fn"] / thresholds[(fold, seed)]["n_attack"],
                                           "dev_bpcer_at_threshold": thresholds[(fold, seed)]["fp"] / thresholds[(fold, seed)]["n_bona_fide"]}
                for fold in range(1, k + 1) for seed in seeds}
    doc = {"per_fold_seed": {str(f): {str(s): per[f][s] for s in seeds} for f in per},
           "per_fold_summary_over_seeds": {str(f): fold_summary[f] for f in fold_summary},
           "pooled_per_seed": {str(s): pooled[s] for s in seeds},
           "pooled_summary_over_seeds": pooled_summary,
           "bootstrap_per_seed": {str(s): boot[s] for s in seeds},
           "training": training}
    return doc, subject_rows


def table(headers, rows) -> str:
    return "| " + " | ".join(headers) + " |\n|" + "---|" * len(headers) + "\n" + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)


def aggregate_md(cfg: dict, agg: dict, record: dict, run_id: str) -> str:
    seeds, k = cfg["train_seeds"], cfg["expected"]["k"]
    L = [f"# NUAA subject folds: LivenessNet, {k} subject folds x {len(seeds)} seeds (aggregate, no identifiers)", "",
         f"Run `{run_id}` (commit `{record['commit']}`, dirty tree: {'yes' if record['dirty'] else 'no'}), {record['finished']}. "
         f"Schema {cfg['schema_run']['run_id']} (configuration `{cfg['schema_run']['config_sha256'][:12]}`), configuration of this run "
         f"`{record['config_sha256'][:12]}`. Training on the CPU with the N1 recipe unchanged; every number below is reported whatever "
         "its value (pre-commitment). Diagnostic design, not a benchmark and not a causal isolation; exploratory.", "",
         "Decision: attack if score_attack > threshold (tie = bona fide). `05` = the fixed threshold 0.5 of C1, N1 and N2; `dev` = the "
         "EER threshold selected on the dev fold only (M1 criterion), frozen before any test scoring. AUROC: threshold-free "
         "(Mann-Whitney, ties 0.5). NA where a class is absent in the stratum. Frames of one subject and session are correlated: "
         "n counts frames, not independent units.", "",
         "## Per fold and seed (held-out subjects, by session stratum)", ""]
    rows = []
    for f in range(1, k + 1):
        for s in seeds:
            for st in STRATA:
                m = agg["per_fold_seed"][str(f)][str(s)][st]
                if m is None:
                    continue
                rows.append([f, s, st, m["n_bona_fide"], m["n_attack"], m["n_subjects"],
                             f"{m['rejected_bona_fide_05']} / {m['accepted_attacks_05']}", f4(m["bpcer_05"]), f4(m["apcer_05"]), f4(m["acer_05"]),
                             f"{m['rejected_bona_fide_dev']} / {m['accepted_attacks_dev']}", f4(m["bpcer_dev"]), f4(m["apcer_dev"]), f4(m["acer_dev"]),
                             f4(m["auroc"])])
    L.append(table(["fold", "seed", "stratum", "n bona fide", "n attack", "subjects", "rejected bf / accepted att. (05)", "BPCER 05", "APCER 05",
                    "ACER 05", "rejected bf / accepted att. (dev)", "BPCER dev", "APCER dev", "ACER dev", "AUROC"], rows))
    L += ["", "## Per fold: median and range over the seeds", ""]
    rows = []
    for f in range(1, k + 1):
        for st in STRATA:
            fs = agg["per_fold_summary_over_seeds"][str(f)][st]
            if fs is None:
                continue
            rows.append([f, st] + [f"{f4(fs[key]['median'])} ({f4(fs[key]['min'])}-{f4(fs[key]['max'])})" for key in ("acer_05", "acer_dev", "auroc")])
    L.append(table(["fold", "stratum", "ACER 05: median (min-max)", "ACER dev: median (min-max)", "AUROC: median (min-max)"], rows))
    L += ["", f"## Pooled over the {k} folds, per seed (every subject held out once; {k} models, {k} dev thresholds)", ""]
    rows = []
    for s in seeds:
        for st in STRATA:
            m = agg["pooled_per_seed"][str(s)].get(st)
            if m is None:
                continue
            b = agg["bootstrap_per_seed"][str(s)][st]
            ci = lambda key: f"[{f4(b[key]['low'])}-{f4(b[key]['high'])}]" if b[key]["low"] is not None else "no interval"
            rows.append([s, st, m["n_bona_fide"], m["n_attack"], m["n_subjects"], f4(m["bpcer_05"]), f4(m["apcer_05"]),
                         f"{f4(m['acer_05'])} {ci('acer_05')}", f4(m["bpcer_dev"]), f4(m["apcer_dev"]), f"{f4(m['acer_dev'])} {ci('acer_dev')}",
                         f"{f4(m['auroc'])} {ci('auroc')}"])
    L.append(table(["seed", "stratum", "n bona fide", "n attack", "subjects", "BPCER 05", "APCER 05", "ACER 05 [bootstrap]", "BPCER dev",
                    "APCER dev", "ACER dev [bootstrap]", "AUROC [bootstrap]"], rows))
    L += ["", "## Pooled: median and range over the seeds", ""]
    rows = [[st] + [f"{f4(v[key]['median'])} ({f4(v[key]['min'])}-{f4(v[key]['max'])})" for key in ("acer_05", "acer_dev", "auroc")]
            for st, v in agg["pooled_summary_over_seeds"].items()]
    L.append(table(["stratum", "ACER 05: median (min-max)", "ACER dev: median (min-max)", "AUROC: median (min-max)"], rows))
    L += ["", "## Training and dev thresholds", ""]
    rows = [[name, t["n_fitting"], t["n_dev"], t["epochs_run"], t["restored_epoch"], t["restored_by"], f"{t['best_val_loss']:.4f}",
             f"{t['train_seconds']:.0f}", f"{t['dev_threshold']:.6f}", f4(t["dev_bpcer_at_threshold"]), f4(t["dev_apcer_at_threshold"])]
            for name, t in agg["training"].items()]
    L.append(table(["model", "n fitting", "n dev", "epochs", "restored epoch", "restored by", "min val_loss", "seconds", "dev threshold",
                    "dev BPCER at threshold", "dev APCER at threshold"], rows))
    bt = BOOT
    L += ["", "## Steps and budget", "",
          table(["step", "seconds (wall clock)", "seconds (monotonic)"],
                [[s, f"{v['wall_clock_seconds']:.0f}", f"{v['monotonic_seconds']:.0f}"] for s, v in record["timings"].items()]),
          f"Wall time {record['wall_seconds']:.0f} s of the cap {cfg['budget']['wall_time_cap_minutes'] * 60} s. Cache check: "
          f"{record['cache_check']['train']['equal'] + record['cache_check']['test']['equal']} frames reproduced bit for bit. Extraction: "
          f"{record['extraction']['n']} frames, detectors {record['extraction']['detectors']}.", "",
          "## Limits", "",
          f"Bootstrap: {bt['resamples']} resamples with replacement of the held-out subjects of the stratum (pooled over the four folds, per "
          f"seed), seed {bt['seed']}, percentiles {bt['percentiles'][0]}-{bt['percentiles'][1]}; no interval with fewer than {bt['min_subjects']} "
          "subjects; the pooled estimate is a cross-validation aggregate scored by four models with four dev thresholds. "
          + cfg["declarations"]["limits"], ""]
    return "\n".join(L)


def public_problems(text: str, inp: Inputs, markdown: bool = False) -> list:
    """Contenuto vietato in un file pubblico: chiavi o SHA-256 delle immagini, identificativi di soggetto (fuori da
    numeri e hash), regole di package_rules, IPv4 non locali, segni di lavoro non finito (solo nei .md)."""
    bad = []
    if re.search(r"\b(real|attack)/\d{4}_", text):
        bad.append("image keys")
    if set(re.findall(r"[0-9a-f]{64}", text)) & {r["sha256"] for r in inp.official.values()}:
        bad.append("image SHA-256 values")
    for s in sorted(inp.subjects):
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
    rid = f"{date}-FOLDSRUN-{state['commit']}" + ("-dev" if state["dirty"] else "")
    if not RUN_ID_RE.match(rid):
        fail(f"run id {rid} does not follow YYYYMMDD-FOLDSRUN-<commit>[-dev]")
    return rid


def restricted_outputs(cfg: dict) -> tuple:
    return tuple(f"fold{f}_seed{s}_{x}" for f in range(1, cfg["expected"]["k"] + 1) for s in cfg["train_seeds"] for x in PER_MODEL) + RESTRICTED_FIXED


def protected_extra(root: Path) -> list:
    return [root / x for x in EXTRA_PROTECTED]


def input_paths(cfg: dict, base: Path) -> dict:
    out = {"the configuration": None}
    out.update({f"manifest {m}": rel_path(base, cfg["manifests"]["folder"]) / cfg["manifests"]["files"][m]["file"] for m in MANIFESTS})
    out["the schema folder"] = rel_path(base, cfg["schema_run"]["folder"])
    out["the cache"] = rel_path(base, cfg["cache"]["path"])
    out["the images folder"] = rel_path(base, cfg["images"]["folder"])
    out["the frozen face module"] = rel_path(base, cfg["extraction"]["frozen_face"]["path"])
    return out


def run_dir_problem(run: Path, root: Path, inputs: dict, cfg: dict, out_root: Path = None) -> str:
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
    for sub, names in ((PUBLIC, PUBLIC_OUTPUTS), (RESTRICTED, restricted_outputs(cfg))):
        if msg := mw.output_problem(run / sub, root, pg.BASE_PROTECTED, [root / DEFAULT_OUT_ROOT], protected_extra(root), inputs, list(names)):
            return msg
    allowed = {"": [STATUS, LOG, PUBLIC, RESTRICTED], PUBLIC: PUBLIC_OUTPUTS, RESTRICTED: restricted_outputs(cfg)}
    for sub, names in allowed.items():
        d = run / sub if sub else run
        if d.is_dir():
            for e in sorted(os.listdir(d)):
                if not any(mw.owned_name(e, n) for n in names):
                    return f"{(Path(sub) / e).as_posix()} in the run folder is not an output of this command: refused, nothing removed"
    return ""


def write_status(run: Path, doc: dict) -> None:
    mw.write_all(run, {STATUS: json.dumps(doc, indent=1) + "\n"})


def invalidate_outputs(run: Path, cfg: dict) -> list:
    left = []
    for sub, names in ((PUBLIC, PUBLIC_OUTPUTS), (RESTRICTED, restricted_outputs(cfg))):
        d = run / sub
        if d.is_dir():
            left += [f"{sub}/{n}" for n in mw.invalidate_each(d, list(names), None, FAIL_DOC)]
    return left


class Tee:
    """Copia di stdout/stderr nel log del run (aperto in coda, O_NOFOLLOW: un collegamento è rifiutato)."""

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


class Budget:
    """Tetto di tempo del run: elapsed() dall'avvio come massimo fra l'orologio monotono e quello di sistema (su macOS il
    monotono non avanza durante lo stop), check(step) fallisce se il tetto è superato; lap registra entrambi i tempi."""

    def __init__(self, cap_minutes: int):
        self.cap = cap_minutes * 60
        self.start = (clock(), time.time())
        self.timings = {}

    def elapsed(self) -> float:
        return max(clock() - self.start[0], time.time() - self.start[1])

    def check(self, step: str) -> None:
        if self.elapsed() > self.cap:
            fail(f"wall-time cap of {self.cap / 60:.0f} min exceeded at {step} ({self.elapsed() / 60:.1f} min): stopped, no result "
                 "(protocol: reported as a failed run)")

    def mark(self) -> tuple:
        return (clock(), time.time())

    def lap(self, step: str, t0: tuple) -> None:
        self.timings[step] = {"monotonic_seconds": round(clock() - t0[0], 1), "wall_clock_seconds": round(time.time() - t0[1], 1)}


def resolve(a) -> tuple:
    base = Path(a.input_root).resolve() if a.input_root else ROOT
    cfg_p = Path(a.config) if Path(a.config).is_absolute() else ROOT / a.config
    return base, cfg_p


def cmd_check_inputs(a) -> int:
    base, cfg_p = resolve(a)
    cfg, cfg_sha = load_config(cfg_p)
    inp = load_inputs(cfg, base)
    sc = inp.checks["schema"]
    print(f"CHECK-INPUTS: configuration {cfg_sha[:12]}, {len(MANIFESTS)} M2 manifests, {sc['folds'] * len(ROLES)} fold manifests of "
          f"{sc['run_id']}, cache {inp.checks['cache']['frames']} frames linked, {len(inp.to_extract)} frames to extract, frozen code and "
          f"resources verified, OpenCV {inp.environment['opencv']}, TensorFlow not imported; nothing written")
    print("RESULT: PASS (check-inputs)")
    return 0


def score_rows(fold: int, seed: int, role: str, part: dict, p_att: np.ndarray, thr: dict, card: dict) -> list:
    rows = []
    for i, r in enumerate(part["rows"]):
        v = float(p_att[i])
        rows.append({"fold": str(fold), "train_seed": str(seed), "role": role, "sample_id": r["key"], "subject_id": r["subject"],
                     "session_id": r["session"], "label": r["label"], "score_attack": repr(v),
                     "prediction_05": "1" if v > FIXED_THRESHOLD else "0", "prediction_dev": "1" if v > thr["value"] else "0",
                     "threshold_dev": repr(float(thr["value"])), "file_sha256": r["sha256"], "checkpoint_sha256": card["weights_sha256"],
                     "threshold_sha256": thr["sha256"]})
    return rows


def rows_text(rows) -> str:
    return csv_text(SCORE_COLUMNS, [[r[c] for c in SCORE_COLUMNS] for r in rows])


def train_one(cfg: dict, run: Path, store: FrameStore, inp: Inputs, fold: int, seed: int, budget: Budget, arch) -> tuple:
    """Un modello: training, scheda, punteggi del dev, soglia congelata, punteggi del test; (scheda, soglia, righe test)."""
    rdir = run / RESTRICTED
    rdir.mkdir(exist_ok=True)
    name = f"fold{fold}_seed{seed}"
    fit, dev = assemble(store, inp.folds[fold]["fitting"]), assemble(store, inp.folds[fold]["dev"])
    weights = rdir / f"{name}_LivenessNet.keras"
    t0 = budget.mark()
    budget.check(f"train {name}")
    card = (TRAINER or default_trainer)({"X_fit": fit["X"], "y_fit": fit["y"], "X_dev": dev["X"], "y_dev": dev["y"]}, cfg["training"], weights,
                                        seed, arch, budget.check)
    budget.lap(f"train {name}", t0)
    if problems := card_problems(card, cfg["training"], seed, weights):
        fail(f"{name}: " + "; ".join(problems))
    card = {**card, "fold": fold, "fitting_keys_sha256": sha256_bytes("\n".join(fit["keys"]).encode()),
            "dev_keys_sha256": sha256_bytes("\n".join(dev["keys"]).encode())}
    del fit
    gc.collect()
    # punteggi del dev con il checkpoint salvato, val_loss ricalcolato, soglia EER scritta e riletta prima del test
    t0 = budget.mark()
    scorer = (MAKE_SCORER or tf_scorer)(weights, card["weights_sha256"])
    p_dev = probabilities(scorer, dev["X"])
    vl = val_loss(p_dev, dev["y"])
    diff = abs(vl - card["best_val_loss"])
    if not (diff <= VAL_LOSS_TOL[0] or diff <= VAL_LOSS_TOL[1] * card["best_val_loss"]):
        fail(f"{name}: val_loss recomputed on the dev {vl:.6f}, recorded minimum {card['best_val_loss']:.6f}: the checkpoint is not the "
             "one of minimum val_loss")
    card["val_loss_recomputed"] = vl
    sel = eer_threshold(dev["y"], p_dev[:, 1])
    thr = {"fold": fold, "train_seed": seed, "set": "dev fold only", "criterion": cfg["threshold"]["criterion"], "value": float(sel["value"]),
           "fp_rejected_bona_fide": sel["fp"], "fn_accepted_attacks": sel["fn"], "n_attack": sel["n_attack"], "n_bona_fide": sel["n_bona_fide"],
           "objective": sel["objective"], "dev_scores_sha256": sha256_bytes("\n".join(repr(float(v)) for v in p_dev[:, 1]).encode()),
           "checkpoint_sha256": card["weights_sha256"], "frozen_at": now()}
    thr_text = json.dumps(thr, indent=1) + "\n"
    mw.write_all(rdir, {f"{name}_threshold.json": thr_text}, newline="")
    back_bytes = read_regular(rdir / f"{name}_threshold.json", "threshold")
    back = json.loads(back_bytes.decode("utf-8"))
    if back != thr:
        fail(f"{name}: threshold.json does not read back identical")
    thr["sha256"] = sha256_bytes(back_bytes)         # l'impronta della soglia congelata è quella dei byte riletti
    thr["fp"], thr["fn"] = sel["fp"], sel["fn"]
    dev_rows = score_rows(fold, seed, "dev", dev, p_dev[:, 1], thr, card)
    mw.write_all(rdir, {f"{name}_dev_scores.csv": rows_text(dev_rows)}, newline="")
    del dev
    budget.lap(f"threshold {name}", t0)
    # punteggi del test, solo ora: i frame del test sono assemblati dopo la soglia (memoria, e nessun accesso prima)
    t0 = budget.mark()
    budget.check(f"score {name}")
    test = assemble(store, inp.folds[fold]["test"])
    p_test = probabilities(scorer, test["X"])
    test_rows = score_rows(fold, seed, "test", test, p_test[:, 1], thr, card)
    card["test_scored_after_threshold"] = now()
    mw.write_all(rdir, {f"{name}_test_scores.csv": rows_text(test_rows)}, newline="")
    mw.write_all(rdir, {f"{name}_card.json": json.dumps(card, indent=1) + "\n"})
    del test, scorer
    gc.collect()
    budget.lap(f"score {name}", t0)
    m = stratum_metrics(test_rows)
    print(f"== {name}: test {m['n_bona_fide']}/{m['n_attack']}, ACER 05 {f4(m['acer_05'])}, ACER dev {f4(m['acer_dev'])} (threshold "
          f"{thr['value']:.6f}), AUROC {f4(m['auroc'])}", flush=True)
    return card, thr, test_rows


def signal_handlers():
    """SIGTERM (e SIGHUP, se non è già ignorato: sotto nohup resta ignorato, così la chiusura del terminale non ferma
    il run) fermano il run come un'interruzione: lo stato diventa FAIL e le uscite sono tolte."""
    def handler(signum, frame):
        raise Failure(f"signal {signal.Signals(signum).name} received: run stopped")
    sigs = [signal.SIGTERM]
    if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:
        sigs.append(signal.SIGHUP)
    for sig in sigs:
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):      # non nel thread principale (test): nessun gestore
            pass


def park_failed_frames(frames_path: Path, run_id: str) -> str:
    """Dopo un errore il file dei frame estratti da questo run, se c'è, prende il nome del run fallito: non blocca il
    tentativo successivo e resta disponibile all'ispezione."""
    if not frames_path.is_file() or frames_path.is_symlink():
        return "none"
    target = frames_path.with_name(FAILED_FRAMES.format(run_id=run_id))
    if target.exists() or target.is_symlink():
        return f"left in place ({target.name} already exists)"
    os.replace(frames_path, target)
    return target.name


def cmd_run(a) -> int:
    if not a.i_am_authorized:
        fail("run trains and scores on the restricted NUAA data: pass --i-am-authorized (the coordinator authorised the run on "
             "3 November 2026)")
    base, cfg_p = resolve(a)
    cfg, cfg_sha = load_config(cfg_p)
    state = repo_state(ROOT)
    rid = run_id_for(a.date or time.strftime("%Y%m%d"), state)
    out_root = Path(a.out_root) if Path(a.out_root).is_absolute() else ROOT / a.out_root
    run = out_root / rid
    inputs = input_paths(cfg, base)
    inputs["the configuration"] = cfg_p
    inputs["the schema configuration"] = rel_path(base, cfg["schema_run"]["config_path"])
    frames_path = rel_path(base, cfg["extraction"]["frames_file"])
    inputs["the frames file"] = frames_path
    if frames_path.exists() or frames_path.is_symlink():
        fail(f"{frames_path.name} already exists in {frames_path.parent.name}: a previous extraction is in the way; move it away "
             "(the run never overwrites it)")
    if (run / STATUS).exists() or (run / STATUS).is_symlink():
        fail(f"the run folder {rid} already holds {STATUS}: every attempt keeps its outcome (archive that folder first)")
    if msg := run_dir_problem(run, ROOT, inputs, cfg, out_root):
        fail(f"refusing the run folder: {msg}")
    run.mkdir(parents=True, exist_ok=True)
    head = {"run_id": rid, "commit": state["commit"], "dirty": state["dirty"], "config_sha256": cfg_sha,
            "delivered_configuration": cfg_sha == CONFIG_SHA256, "command": "run"}
    log = None
    budget = Budget(cfg["budget"]["wall_time_cap_minutes"])
    signal_handlers()
    try:
        write_status(run, {**head, "status": "RUNNING", "passed": False, "started": now(),
                           "note": "RUNNING without a live process means a failed run"})
        log = open_log(run)
        left = invalidate_outputs(run, cfg)
        if left:
            fail(f"previous outputs could not be removed: {', '.join(left)}")
        print(f"== NUAA folds run {rid}: configuration {cfg_sha[:12]} (delivered: {head['delivered_configuration']}), cap "
              f"{cfg['budget']['wall_time_cap_minutes']} min", flush=True)
        t0 = budget.mark()
        inp = load_inputs(cfg, base)
        budget.lap("check-inputs", t0)
        print(f"   check-inputs PASS: {len(inp.cache_index)} cache frames, {len(inp.to_extract)} to extract", flush=True)
        pre = preprocessor(cfg, base)
        t0 = budget.mark()
        cc = cache_check(cfg, base, inp, pre)
        budget.lap("cache-check", t0)
        budget.check("cache-check")
        t0 = budget.mark()
        ext, frame_rows, X_ext = extract_frames(cfg, base, inp, pre, frames_path)
        budget.lap("extract", t0)
        budget.check("extract")
        print(f"   extract PASS: {ext['n']} frames in {frames_path.name} ({ext['sha256'][:12]}), detectors {ext['detectors']}", flush=True)
        store = FrameStore(inp, X_ext, list(inp.to_extract))
        arch = (ARCH_LOADER or default_arch_loader)(rel_path(base, cfg["architecture"]["path"]))
        cards, thresholds, scores = {}, {}, {}
        for fold in range(1, cfg["expected"]["k"] + 1):
            for seed in cfg["train_seeds"]:
                cards[(fold, seed)], thresholds[(fold, seed)], scores[(fold, seed)] = train_one(cfg, run, store, inp, fold, seed, budget, arch)
        budget.check("aggregate")
        t0 = budget.mark()
        agg, subject_rows = aggregate(cfg, scores, cards, thresholds)
        budget.lap("aggregate", t0)
        budget.check("write")
        record = {**head, "finished": now(), "wall_seconds": round(budget.elapsed(), 1), "timings": budget.timings,
                  "checks": inp.checks, "cache_check": cc, "extraction": ext, "environment": inp.environment, "code_sha256": inp.code,
                  "schema_run": cfg["schema_run"], "train_seeds": cfg["train_seeds"], "folds": cfg["expected"]["k"],
                  "models": len(cards), "bootstrap": BOOT, "fold_manifests_sha256": inp.fold_sha}
        agg_doc = {**head, "finished": record["finished"], "bootstrap": BOOT, "decision": DECISION_RULE, "strata": list(STRATA), **agg}
        public = {"aggregate.json": json.dumps(agg_doc, indent=1) + "\n", "run_record.json": json.dumps(record, indent=1) + "\n"}
        public["aggregate.md"] = aggregate_md(cfg, agg, record, rid)
        for name, text in public.items():
            if probs := public_problems(text, inp, name.endswith(".md")):
                fail(f"public file {name} refused: " + "; ".join(probs))
        public["SHA256SUMS"] = sums_text(public)
        rdir = run / RESTRICTED
        restricted_text = {"per_subject.csv": csv_text(SUBJECT_COLUMNS, subject_rows), "extracted_frames.csv": csv_text(FRAME_COLUMNS, frame_rows),
                           "public_SHA256SUMS.sha256": sha256_bytes(public["SHA256SUMS"].encode("utf-8")) + "  public/SHA256SUMS\n"}
        budget.check("write")
        mw.write_all(rdir, restricted_text, newline="")
        names = [n for n in restricted_outputs(cfg) if n != "SHA256SUMS"]
        missing = [n for n in names if not (rdir / n).is_file()]
        if missing:
            fail(f"restricted outputs missing: {missing}")
        sums = {n: sha256_file(rdir / n) for n in names}
        budget.check("write")
        mw.write_all(rdir, {"SHA256SUMS": "".join(f"{h}  {n}\n" for n, h in sorted(sums.items()))})
        mw.write_all(run / PUBLIC, public)
        for st in STRATA:
            v = agg["pooled_summary_over_seeds"].get(st)
            if v:
                print(f"pooled {st}: ACER 05 {f4(v['acer_05']['median'])} ({f4(v['acer_05']['min'])}-{f4(v['acer_05']['max'])}), ACER dev "
                      f"{f4(v['acer_dev']['median'])} ({f4(v['acer_dev']['min'])}-{f4(v['acer_dev']['max'])}), AUROC {f4(v['auroc']['median'])} "
                      f"({f4(v['auroc']['min'])}-{f4(v['auroc']['max'])})")
        print(f"run folder: {run.name} (public/ and restricted/), wall {record['wall_seconds']:.0f} s")
        print("RESULT: PASS (run)")
        close_log(log)
        log = None
        write_status(run, {**head, "status": "PASS", "passed": True, "finished": record["finished"], "wall_seconds": record["wall_seconds"],
                           "public_sums_sha256": sha256_bytes(public["SHA256SUMS"].encode("utf-8")),
                           "restricted_sums_sha256": sha256_bytes((rdir / "SHA256SUMS").read_bytes()),
                           "run_log_sha256": sha256_file(run / LOG)})
    except BaseException as e:
        reason = str(e) if isinstance(e, Failure) else f"unexpected error {type(e).__name__}: {e}"
        left = invalidate_outputs(run, cfg)
        try:
            parked = park_failed_frames(frames_path, rid)
        except OSError as pe:                     # lo stato FAIL va scritto comunque, con l'errore annotato
            parked = f"not parked ({type(pe).__name__}: {pe})"
        if log is not None:
            log[0].write(f"ERROR: {reason}\nRESULT: FAIL (run)\n")
            close_log(log)
        try:
            write_status(run, {**head, "status": "FAIL", "passed": False, "reason": reason, "wall_seconds": round(budget.elapsed(), 1),
                               "outputs": "removed" if not left else f"left behind: {', '.join(left)}", "extracted_frames_file": parked,
                               "run_log_sha256": sha256_file(run / LOG) if (run / LOG).is_file() else "none"})
        except Exception:  # noqa: BLE001
            print("ERROR: status.json could not be written", file=sys.stderr)
        raise
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="NUAA subject folds: LivenessNet training run, 4 folds x 3 seeds on the CPU")
    ap.add_argument("cmd", choices=["check-inputs", "run"])
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="configuration JSON (relative to the repository root)")
    ap.add_argument("--input-root", default=None, help="base folder of the input paths of the configuration (default: repository root)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="folder that receives <run-id>/ (default results/foldsrun)")
    ap.add_argument("--date", default=None, help="YYYYMMDD of the run id (default: today)")
    ap.add_argument("--i-am-authorized", action="store_true", help="required by run: training and inference on the restricted NUAA data")
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
