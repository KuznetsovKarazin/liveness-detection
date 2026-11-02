"""
Pilota N1, parte 2: piccolo pilota di training NUAA (LivenessNet, seed di training 42, 43 e 44; seed dello split sempre
42), con la cache locale nuaa_1500.npz, il preprocessing e lo split di M2 (2.507 fitting, 493 validazione, 3.000 test).

Sottocomandi, sempre con --config <file JSON> --manifests <cartella dei manifest M2 NUAA> --cache <nuaa_1500.npz>
--study-dir <cartella dello studio>:
  check      controlli prima di qualunque training (nessun TensorFlow, nessun peso, nessun rilevatore): configurazione
             (valori della scheda, schema chiuso), SHA-256 e schema della cache (membri, tipi, forme), SHA-256, righe e
             conteggi dei sei manifest M2 (validazione stretta di nuaa_manifests.py), derivazione dei manifest dalle
             liste ufficiali con il campionamento `spread` (ordine di campionamento, bona fide e poi attacchi),
             collegamento fra gli array y/s della cache e le righe dei manifest nello stesso ordine, split per soggetto
             con il seed dello split (GroupShuffleSplit) uguale a cnn_train / cnn_early_stopping per chiave e per SHA-256
             del file, sottoinsieme C1 dentro il test, frame derivati senza duplicati né condivisi fra fitting,
             validazione e test. Crea la cartella dello studio (nuova) e scrive check.json; il log in check.log.
  train      --train-seed N --i-am-authorized: ripete i controlli, crea la cartella seed<N> (nuova: mai riusata) e
             addestra con scripts/n1_trainer.py, solo su CPU (solo X_train, y_train, s_train; checkpoint all'epoca di
             val_loss minimo) in seed<N>/checkpoints/; controlla pesi e scheda e scrive train.json; il log in
             seed<N>/train.log.
  score      --train-seed N --i-am-authorized: su CPU (BatchNorm in inferenza), punteggi di attacco dei 3.000 frame del
             test e ricalcolo di val_loss sui 493 di validazione (deve coincidere con il minimo registrato); scrive
             scores_test.csv (per immagine, con il flag del sottoinsieme C1), per_subject.csv, metrics.json e
             score.json (area riservata); il log in seed<N>/score.log.
  aggregate  rilegge i tre seed (record legati a questo studio, SHA-256 dei punteggi, metriche ricalcolate) e scrive
             aggregate.json e aggregate.md (pubblicabili: nessun identificativo, nessuna riga per immagine, controllati
             con la scansione del contenuto vietato) e per_subject_all_seeds.csv (riservato); il log in aggregate.log.

Metriche: soglia fissa 0,5 sulla probabilità di attacco (attacco se score_attack > 0,5, pareggio = bona fide), APCER,
BPCER e ACER con i denominatori, AUROC (src/tesi_app/evaluation.py, invariato); classe assente = NA, mai 0. Nessuna
soglia M1 riusata, nessuna scelta del seed migliore: tutti i seed della configurazione, valori individuali, mediana e
intervallo (minimo-massimo).

Comportamento su errore: rifiuti prima che una cartella sia accettata (nome, posizione protetta, cartella non nuova,
ingressi non validi, opzioni) non scrivono nulla; dopo, status.json della cartella diventa RUNNING e poi PASS o FAIL
con il passo e il motivo; i log restano sempre (in coda, mai sovrascritti); i record e le uscite di un passo si
scrivono solo alla fine (temporanei con mkstemp e rinomine). Le uscite di train e score non si sovrascrivono mai: una
cartella seed<N> esistente è rifiutata, e score è rifiutato se le sue uscite esistono già. aggregate toglie le proprie
uscite precedenti prima di ricalcolarle (dopo aver verificato che siano file regolari).

La cartella dello studio si chiama YYYYMMDD-N1-LivenessNet-<commit breve>[-trial]; dentro il repository di sviluppo è
ammessa solo sotto results/n1/; nell'albero del repository del gruppo (pacchetti accettati accanto) non è ammessa.
Un run non -trial richiede i dati della scheda (SHA-256 della cache e dei manifest, conteggi) e il commit corrente nel
nome; lo stato dell'albero (dirty) è registrato.

Dipendenze: numpy (tutti i passi); scikit-learn per check, train e aggregate/score (split, AUROC); TensorFlow solo per
train e score con i modelli veri; OpenCV importato da src/tesi_app/evaluation.py.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np  # noqa: E402
    import numpy.lib.format as npy_format  # noqa: E402  (intestazioni .npy: solo funzioni pubbliche)
    import path_guard as pg  # noqa: E402
    import nuaa_manifests as nm  # noqa: E402  (lettura e validazione dei manifest NUAA, spread_keys, frame_hashes)
    from package_rules import FORBIDDEN_TEXT  # noqa: E402
except ModuleNotFoundError as _e:   # dipendenza mancante: messaggio chiaro, mai un traceback
    print(f"ERROR: missing Python package {_e.name!r}: numpy is needed by every subcommand", file=sys.stderr)
    print("RESULT: FAIL (import)")
    sys.exit(1)
from tesi_app.paths import DATA, LIVEDETECTION, RESULTS, WEIGHTS, load_module  # noqa: E402

CONFIG_SCHEMA = "n1-nuaa-stability-config/1"
RUN_ID_RE = re.compile(r"^(\d{8})-N1-LivenessNet-([0-9a-f]{7,40})(-trial)?$")
SEED_DIR_RE = re.compile(r"^seed(\d+)$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
THRESHOLD = 0.5
DECISION_RULE = "attack if score_attack > 0.5 (tie = bona fide)"
SCORE_BATCH = 16
NA = "NA"

# valori della scheda N1 (Trello cezonCpA): ogni configurazione deve averli identici, anche nei run -trial
FROZEN = {
    "schema": CONFIG_SCHEMA,
    "study": "N1 part 2: small NUAA training pilot",
    "architecture": "LivenessNet",
    "dataset": "nuaa",
    "protocol": "official",
    "train_seeds": [42, 43, 44],
    "split_seed": 42,
    "split": "GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=split_seed) on the subject array s_train",
    "training": {
        "max_epochs": 12, "batch": 16, "initial_learning_rate": 0.001, "early_stopping_patience": 4,
        "checkpoint": "weights of the epoch with minimum val_loss (EarlyStopping restore_best_weights; set_weights when the epochs end first)",
        "optimizer": "tf.keras.optimizers.legacy.Adam(learning_rate=initial_learning_rate, clipnorm=1.0)",
        "loss": "CategoricalCrossentropy, one-hot labels (index 0 bona fide, 1 attack)",
        "class_weight": "balanced on the fitting set: n / (2 * n_class)",
        "augmentation": "RandomFlip(horizontal), RandomRotation(0.03), RandomZoom(0.12, 0.12), RandomTranslation(0.06, 0.06), "
                        "RandomBrightness(0.15, value_range=(0, 1)), RandomContrast(0.2), GaussianNoise(0.02); fitting set only",
        "scheduler": "ReduceLROnPlateau(monitor=val_loss, factor=0.5, patience=2, min_lr=1e-6)",
        "shuffle": "tf.data shuffle(4096, seed=train_seed)",
        "seeds": "train_seed for random, numpy and TensorFlow (weight initialisation, augmentation, batch order); split_seed only for the split",
        "device": "CPU",
    },
    "scoring": {"device": "CPU", "batch": SCORE_BATCH, "batchnorm": "inference (training=False)", "input": "frames of the cache / 255",
                "score": "softmax output, column 1 (attack probability)"},
    "metrics": {"threshold": THRESHOLD, "decision": DECISION_RULE,
                "reported": ["APCER", "BPCER", "ACER", "AUROC"], "sets": ["test_3000", "c1_300"],
                "aggregation": "values of every seed, median and range (minimum-maximum); no seed selection"},
}
# dati della scheda: obbligatori per un run non -trial (SHA-256 della cache dalla scheda, manifest del run M2 accettato)
CARD_DATA = {
    "m2_run": "20261006-M2-seed42-36c677a",
    "cache": {"file": "nuaa_1500.npz", "sha256": "dc286dc371e61d638da1842c062a38c5368a9ec4da07ccbc36b7fff6f3177920", "per_class": 1500,
              "members": {"X_train": {"dtype": "|u1", "shape": [3000, 256, 256, 3]}, "y_train": {"dtype": "<i8", "shape": [3000]},
                          "s_train": {"dtype": "<U4", "shape": [3000]}, "X_test": {"dtype": "|u1", "shape": [3000, 256, 256, 3]},
                          "y_test": {"dtype": "<i8", "shape": [3000]}, "s_test": {"dtype": "<U4", "shape": [3000]}}},
    "manifests": {
        "official_train": {"file": "nuaa_official_train.csv", "sha256": "2a7e8906f98b8e3cf0c42cd0a2cf10e15868750b1ac82e85b62a9e0b59aad648",
                           "rows": 3491, "n_bona_fide": 1743, "n_attack": 1748, "n_subjects": 9},
        "official_test": {"file": "nuaa_official_test.csv", "sha256": "2fad23ff295cb2c1d3d0be238deef02faad252aa717933c30588bd5a9c4106c7",
                          "rows": 9123, "n_bona_fide": 3362, "n_attack": 5761, "n_subjects": 16},
        "cnn_train": {"file": "nuaa_cnn_train.csv", "sha256": "6d46ec7da7bfc32b1cce9ed3e331f8ed4377e5fc2df31e1b77ac7f6a0c2bc33c",
                      "rows": 2507, "n_bona_fide": 1345, "n_attack": 1162, "n_subjects": 7},
        "cnn_es": {"file": "nuaa_cnn_early_stopping.csv", "sha256": "17282be8ef300e5e178ae87cd137ffd875786f1b353924e7cfb6eb14db1584c1",
                   "rows": 493, "n_bona_fide": 155, "n_attack": 338, "n_subjects": 2},
        "cnn_test": {"file": "nuaa_cnn_test.csv", "sha256": "ce9669dccae37041e778ff7bcba5f4cef08fa79f50d65be84806b084c5d95748",
                     "rows": 3000, "n_bona_fide": 1500, "n_attack": 1500, "n_subjects": 16},
        "c1": {"file": "nuaa_c1_v2.csv", "sha256": "1c2cf97cb7115e7ae98640103670c628a3910a37c25b89f76369cc7ecad87426",
               "rows": 300, "n_bona_fide": 150, "n_attack": 150, "n_subjects": 16},
    },
    # checkpoint precedenti di LivenessNet su NUAA (C1, 12/09; official e pooled): i nuovi pesi non possono coincidere
    "previous_checkpoints_sha256": ["d861c5969f24376aff5aaac30c2b2b8793e0cd142dd016756dc82f188ccd99a5",
                                    "5d503ea2acf76a0f64578df420063db8470be5810a41e586219cdce249427606",
                                    "2456c9f3bda446badf519de2b2d80ba99b99add32c5c9e452e0d7a208023ba9b",
                                    "16d462e6c3537158ee5c06e9b8d027202a5dd9cc948fd2ed36ab60175f80e7f5"],
}
MANIFESTS = tuple(CARD_DATA["manifests"])
MEMBERS = tuple(CARD_DATA["cache"]["members"])
COUNT_FIELDS = ("rows", "n_bona_fide", "n_attack", "n_subjects")
CODE_FILES = ("scripts/n1_train.py", "scripts/n1_trainer.py", "scripts/nuaa_manifests.py", "scripts/select_threshold.py",
              "scripts/path_guard.py", "scripts/m2_writers.py", "scripts/package_rules.py", "src/tesi_app/evaluation.py",
              "src/tesi_app/paths.py")
ARCH_KEY = "LIVENESSLAB_LIVEDETECTION/src/architectures.py"
CHECKPOINTS = "checkpoints"
CKPT_KEY = "LivenessNet__nuaa"
STATUS = "status.json"
SCORE_OUTPUTS = ("scores_test.csv", "per_subject.csv", "metrics.json", "score.json")
AGG_OUTPUTS = ("aggregate.json", "aggregate.md", "per_subject_all_seeds.csv")
SCORE_COLUMNS = ["run_id", "train_seed", "sample_id", "subject_id", "session_id", "label", "score_attack", "prediction",
                 "in_c1_subset", "file_sha256", "checkpoint_sha256", "config_sha256"]
SUBJECT_COLUMNS = ["run_id", "train_seed", "set", "subject_id", "n_bona_fide", "n_attack", "rejected_bona_fide", "accepted_attacks",
                   "bpcer", "apcer", "acer"]
SETS = {"test_3000": "test (3,000 frames of nuaa_cnn_test)", "c1_300": "C1 subset (300 frames of nuaa_c1_v2, inside the test)"}
VAL_LOSS_TOL = (1e-3, 1e-2)      # |val_loss ricalcolato - minimo registrato| <= 1e-3 oppure <= 1 % del minimo
EPS = 1e-7                       # clipping di Keras nella categorical crossentropy

# punti di iniezione (test): addestramento, scorer, stato del repository
TRAINER = None          # (data, training, out_dir, train_seed, split_seed, split_check) -> scheda; None: n1_trainer.train
MAKE_SCORER = None      # (weights_path, weights_sha256) -> funzione(X uint8) -> probabilità N x 2; None: TensorFlow su CPU
REPO_STATE = None       # (study_dir) -> {"commit", "dirty", ...}; None: git

RUN_STATE = {"dir": None, "cmd": None, "log": None}


class N1Error(Exception):
    """Errore esplicito di un passo: messaggio senza traceback, RESULT: FAIL, codice 1."""


def fail(msg: str) -> None:
    raise N1Error(msg)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def guarded(what: str, fn, *args):
    """Chiama una funzione di nuaa_manifests.py (che si ferma con sys.exit dopo una riga ERROR) come N1Error."""
    try:
        return fn(*args)
    except SystemExit as e:
        fail(f"{what}: invalid input (see the ERROR line above)" if e.code in (None, 0, 1) else f"{what}: {e.code}")


# ============================================================================ scritture e log

_UMASK = os.umask(0o022)
os.umask(_UMASK)


def write_all(d, files: dict) -> None:
    """Scrive un gruppo di file (nome -> testo) solo alla fine: temporanei con tempfile.mkstemp (nome nuovo, O_EXCL,
    nessun collegamento seguito), poi le rinomine; un'uscita che è un collegamento o non è un file regolare è rifiutata
    prima di scrivere. Su qualunque eccezione non resta nessun file del gruppo né alcun temporaneo."""
    d = Path(d)
    tmps, done = {}, []
    try:
        for n in files:
            q = d / n
            if q.is_symlink() or (q.exists() and not q.is_file()):
                raise N1Error(f"{n} is a link or not a regular file: refused")
        for n, text in files.items():
            fd, tp = tempfile.mkstemp(prefix=f".{n}.", suffix=".tmp", dir=d)
            tmps[n] = Path(tp)
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o666 & ~_UMASK)
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                f.write(text)
        for n, tp in tmps.items():
            os.replace(tp, d / n)
            done.append(n)
    except BaseException:
        for tp in tmps.values():
            tp.unlink(missing_ok=True)
        for n in done:
            (d / n).unlink(missing_ok=True)
        raise


class Tee:
    """Copia di stdout/stderr nel log della cartella accettata (aperto in coda con O_NOFOLLOW: un collegamento è rifiutato)."""

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


def open_log(d: Path, name: str) -> None:
    p = Path(d) / name
    if p.is_symlink() or (p.exists() and not p.is_file()):
        fail(f"{name} is a link or not a regular file: refused")
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0), 0o666 & ~_UMASK)
    fh = os.fdopen(fd, "a", encoding="utf-8")
    fh.write(f"\n===== {now()} {' '.join(sys.argv[1:2])} =====\n")
    RUN_STATE["log"] = (fh, sys.stdout, sys.stderr)
    sys.stdout, sys.stderr = Tee(sys.stdout, fh), Tee(sys.stderr, fh)


def close_log() -> None:
    if RUN_STATE["log"]:
        fh, out, err = RUN_STATE["log"]
        sys.stdout, sys.stderr = out, err
        fh.close()
        RUN_STATE["log"] = None


def write_status(result: str, reason: str = "") -> None:
    d = RUN_STATE["dir"]
    if d is None:
        return
    doc = {"result": result, "step": RUN_STATE["cmd"], "folder": Path(d).name, "updated": now()}
    if reason:
        doc["reason"] = reason
    write_all(d, {STATUS: json.dumps(doc, indent=1) + "\n"})


def accept(d: Path, cmd: str, log: str) -> None:
    """Cartella accettata: da qui status.json = RUNNING e il log riceve tutto."""
    RUN_STATE.update(dir=Path(d), cmd=cmd)
    write_status("RUNNING")
    open_log(d, log)


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
    problems = config_problems(cfg)
    if problems:
        fail(f"configuration {p.name} refused: " + "; ".join(problems))
    return cfg, hashlib.sha256(raw).hexdigest()


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def config_problems(cfg: dict) -> list:
    """Schema chiuso: i valori della scheda (FROZEN) identici, tipi compresi (12 e non 12.0, true e non 1), e un blocco
    data ben formato; la coerenza con la scheda dei dati (CARD_DATA) è richiesta da run_problems per i run non -trial."""
    out = []
    want = set(FROZEN) | {"data"}
    if set(cfg) != want:
        out.append(f"keys {sorted(set(cfg) ^ want)} missing or not allowed")
    for k, v in FROZEN.items():
        if k in cfg and json.dumps(cfg[k], sort_keys=True) != json.dumps(v, sort_keys=True):
            out.append(f"{k} differs from the N1 card")
    d = cfg.get("data")
    if not isinstance(d, dict) or set(d) != set(CARD_DATA):
        return out + ["data must hold exactly " + ", ".join(sorted(CARD_DATA))]
    if not isinstance(d["m2_run"], str) or not d["m2_run"]:
        out.append("data.m2_run must be a run id")
    c = d["cache"]
    if not isinstance(c, dict) or set(c) != {"file", "sha256", "per_class", "members"}:
        return out + ["data.cache must hold file, sha256, per_class, members"]
    if not isinstance(c["file"], str) or not re.fullmatch(r"nuaa_[1-9]\d*\.npz", c["file"]):
        out.append("data.cache.file must be nuaa_<per_class>.npz")
    elif not _int(c["per_class"]) or c["file"] != f"nuaa_{c['per_class']}.npz":
        out.append("data.cache.per_class must be the integer in the file name")
    if not isinstance(c["sha256"], str) or not HEX64.match(c["sha256"]):
        out.append("data.cache.sha256 must be 64 lower-case hexadecimal characters")
    m = c["members"]
    if not isinstance(m, dict) or set(m) != set(MEMBERS):
        out.append(f"data.cache.members must be exactly {list(MEMBERS)}")
    else:
        for k, v in m.items():
            if (not isinstance(v, dict) or set(v) != {"dtype", "shape"} or not isinstance(v["dtype"], str) or not isinstance(v["shape"], list)
                    or not v["shape"] or not all(_int(x) and x > 0 for x in v["shape"])):
                out.append(f"data.cache.members.{k} must be {{dtype, shape}} with positive integer sizes")
    ms = d["manifests"]
    if not isinstance(ms, dict) or set(ms) != set(MANIFESTS):
        return out + [f"data.manifests must be exactly {list(MANIFESTS)}"]
    for k, e in ms.items():
        if not isinstance(e, dict) or set(e) != {"file", *COUNT_FIELDS, "sha256"}:
            out.append(f"data.manifests.{k} must hold file, sha256, {', '.join(COUNT_FIELDS)}"); continue
        if not isinstance(e["file"], str) or not re.fullmatch(r"nuaa_[a-z0-9_]+\.csv", e["file"]):
            out.append(f"data.manifests.{k}.file must be a nuaa_*.csv name")
        if not isinstance(e["sha256"], str) or not HEX64.match(e["sha256"]):
            out.append(f"data.manifests.{k}.sha256 must be 64 lower-case hexadecimal characters")
        if not all(_int(e[f]) and e[f] >= 0 for f in COUNT_FIELDS):
            out.append(f"data.manifests.{k}: counts must be non-negative integers")
    prev = d["previous_checkpoints_sha256"]
    if not isinstance(prev, list) or not all(isinstance(x, str) and HEX64.match(x) for x in prev):
        out.append("data.previous_checkpoints_sha256 must be a list of SHA-256 values")
    return out


def run_problems(cfg: dict, trial: bool) -> list:
    """Un run non -trial accetta solo i dati della scheda (cache, manifest, conteggi, checkpoint precedenti)."""
    if trial:
        return []
    if json.dumps(cfg["data"], sort_keys=True) != json.dumps(CARD_DATA, sort_keys=True):
        return ["non-trial run: data differs from the N1 card (cache SHA-256, M2 manifests and counts)"]
    return []


# ============================================================================ cartelle e ingressi

def protected_extra() -> list:
    """Cartelle protette spostabili con le variabili d'ambiente (pesi, dati, risultati consegnati)."""
    return [WEIGHTS, DATA] + [RESULTS / x for x in ("c1", "m1", "m2", "m2_rgb", "eval", "c2")]


def outer_roots() -> tuple:
    import m2_writers as mw
    return mw.outer_roots(ROOT)


def study_dir_problem(d) -> str:
    """Motivo per cui `d` non può essere la cartella dello studio ("" se può): nome, collegamento, file, alberi con i
    pacchetti accettati, path_guard (radice, home, repository, cartelle protette; dentro il repository solo results/n1/),
    cartelle dei pesi, dei dati e dei risultati consegnati anche spostate (in entrambe le direzioni)."""
    d = Path(d)
    if not RUN_ID_RE.match(d.name):
        return f"study folder name {d.name!r} is not YYYYMMDD-N1-LivenessNet-<shortcommit>[-trial]"
    if d.is_symlink():
        return f"{d.name} is a symbolic link: the study folder must be a real folder"
    if d.exists() and not d.is_dir():
        return f"{d.name} exists and is not a folder"
    for g in outer_roots():
        if pg.within(d, g) or pg.contains(d, g):
            return f"{d} is inside, or contains, the work tree {g.name} with the accepted packages: outputs go outside it"
    if msg := pg.deletion_problem(d, ROOT, pg.BASE_PROTECTED, [ROOT / "results" / "n1"]):
        return msg
    for q in protected_extra():
        if pg.within(d, q) or (Path(q).exists() and pg.contains(d, q)):
            return f"{d} is, contains or is inside a protected folder ({q.name})"
    return ""


def inputs_problem(a, d) -> str:
    """Ingressi controllati prima di qualunque modifica: esistono, sono del tipo giusto e sono separati dalla cartella
    dello studio per identità dei file (né dentro, né uguali, né antenati)."""
    for opt, p in (("--config", a.config), ("--manifests", a.manifests), ("--cache", a.cache)):
        p = Path(p)
        if pg.within(p, d) or pg.contains(p, d):
            return f"{opt} {p.name} is, contains or is inside the study folder: inputs and outputs must be separate"
        if opt == "--manifests" and not p.is_dir():
            return f"--manifests {p.name} is not a folder"
        if opt != "--manifests" and not p.is_file():
            return f"{opt} {p.name} not found"
    return ""


def is_trial(d) -> bool:
    return Path(d).name.endswith("-trial")


def git(*args) -> str:
    try:
        r = subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def repo_state(d) -> dict:
    """Commit e stato dell'albero (file non versionati compresi) senza la cartella dello studio, se sta nel repository."""
    if REPO_STATE is not None:
        return REPO_STATE(d)
    commit = (git("rev-parse", "HEAD") or "").strip()
    if not commit:
        return {"commit": "n/d", "dirty": False, "dirty_paths": [], "dirty_excludes": None}
    excl = None
    try:
        excl = Path(d).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        pass
    args = ["status", "--porcelain", "--untracked-files=all", "--", "."] + ([f":(exclude){excl}"] if excl else [])
    st = git(*args)
    paths = sorted(line[3:] for line in (st or "").splitlines() if line.strip())
    return {"commit": commit, "dirty": st is None or bool(paths), "dirty_paths": paths if st is not None else ["git status failed"],
            "dirty_excludes": excl}


def code_hashes() -> dict:
    out = {f: sha256_file(ROOT / f) for f in CODE_FILES}
    arch = Path(LIVEDETECTION) / "src" / "architectures.py"
    out[ARCH_KEY] = sha256_file(arch) if arch.is_file() else "absent"
    return out


# ============================================================================ ingressi: manifest, cache, split

class Inputs:
    pass


def counts_of(rows) -> dict:
    lab = Counter(r["label"] for r in rows)
    return {"rows": len(rows), "n_bona_fide": lab["0"], "n_attack": lab["1"], "n_subjects": len({r["subject"] for r in rows})}


NPY_HEADERS = {(1, 0): "read_array_header_1_0", (2, 0): "read_array_header_2_0"}


def npy_header(fh, what: str) -> tuple:
    """Forma, ordine Fortran e tipo di un .npy dall'intestazione, con le sole funzioni pubbliche di numpy.lib.format
    (read_magic e read_array_header_1_0 / 2_0, presenti in NumPy 1.x e 2.x); la versione 3.0 del formato (nomi dei
    campi in UTF-8) non ha una funzione pubblica ed è rifiutata, come ogni altra versione."""
    fmt = npy_format
    ver = fmt.read_magic(fh)
    if ver not in NPY_HEADERS:
        fail(f"{what}: .npy format version {ver[0]}.{ver[1]} not supported (1.0 or 2.0 expected)")
    return getattr(fmt, NPY_HEADERS[ver])(fh)


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


def load_inputs(cfg: dict, a, frames: bool = True) -> Inputs:
    """Tutti i controlli prima del training (vedi check). Restituisce manifest, cache, ordini di campionamento, split e
    l'esito di ogni controllo (inp.checks)."""
    from sklearn.model_selection import GroupShuffleSplit  # noqa: F401  (dipendenza dichiarata: errore chiaro se manca)
    import n1_trainer
    d = cfg["data"]
    inp = Inputs()
    inp.checks = {}
    # ---------------------------------------------------------------- cache: nome, SHA-256, schema
    cp = Path(a.cache)
    if cp.name != d["cache"]["file"]:
        fail(f"cache {cp.name} is not the file of the configuration ({d['cache']['file']})")
    inp.cache_sha = sha256_file(cp)
    if inp.cache_sha != d["cache"]["sha256"]:
        fail(f"cache {cp.name}: SHA-256 {inp.cache_sha} differs from the expected {d['cache']['sha256']}")
    inp.checks["cache_sha256"] = inp.cache_sha
    mem = npz_members(cp)
    if set(mem) != set(MEMBERS):
        fail(f"cache {cp.name}: members {sorted(mem)} differ from {sorted(MEMBERS)}")
    for k, want in d["cache"]["members"].items():
        got = {"dtype": mem[k]["dtype"], "shape": mem[k]["shape"]}
        if got != want or mem[k]["fortran"]:
            fail(f"cache {cp.name}: {k} has dtype {got['dtype']}, shape {got['shape']}{' (Fortran order)' if mem[k]['fortran'] else ''}, "
                 f"expected {want['dtype']}, {want['shape']}")
    inp.checks["cache_schema"] = {k: {"dtype": v["dtype"], "shape": v["shape"]} for k, v in sorted(mem.items())}
    # ---------------------------------------------------------------- manifest: SHA-256, righe valide, conteggi
    inp.rows, inp.sha = {}, {}
    for name, e in d["manifests"].items():
        p = Path(a.manifests) / e["file"]
        if p.is_symlink() or not p.is_file():
            fail(f"manifest {e['file']} ({name}) not found or not a regular file")
        got = sha256_file(p)
        if got != e["sha256"]:
            fail(f"manifest {e['file']} ({name}): SHA-256 {got} differs from the configuration ({e['sha256']})")
        rows = guarded(e["file"], nm.load_manifest, p)
        c = counts_of(rows)
        for f in COUNT_FIELDS:
            if c[f] != e[f]:
                fail(f"manifest {e['file']} ({name}): {f} {c[f]} differs from the configuration ({e[f]})")
        for dim in ("key", "sha256"):
            if len({r[dim] for r in rows}) != len(rows):
                fail(f"manifest {e['file']} ({name}): duplicated {dim} values")
        inp.rows[name], inp.sha[name] = rows, got
    inp.by_key = {n: {r["key"]: r for r in rows} for n, rows in inp.rows.items()}
    # stessa chiave in due manifest: stessi valori in tutti i campi (controllo per riga, non solo per insiemi)
    for sub, sup in (("cnn_train", "official_train"), ("cnn_es", "official_train"), ("cnn_test", "official_test"), ("c1", "cnn_test")):
        for k, r in inp.by_key[sub].items():
            if k not in inp.by_key[sup]:
                fail(f"manifest {sub}: {sum(1 for x in inp.by_key[sub] if x not in inp.by_key[sup])} keys are not in {sup}")
            if inp.by_key[sup][k] != r:
                fail(f"manifest {sub}: rows with the same key as {sup} but different fields")
    for name, split in (("cnn_train", "train"), ("cnn_es", "train"), ("cnn_test", "test"), ("c1", "test")):
        if any(r["official_split"] != split for r in inp.rows[name]):
            fail(f"manifest {name}: rows outside the official {split} split")
    for dim in ("key", "sha256", "subject"):
        shared = {r[dim] for r in inp.rows["cnn_train"]} & {r[dim] for r in inp.rows["cnn_es"]}
        if shared:
            fail(f"cnn_train and cnn_es share {len(shared)} values of {dim}")
    for dim in ("key", "sha256"):
        shared = {r[dim] for r in inp.rows["official_train"]} & {r[dim] for r in inp.rows["official_test"]}
        if shared:
            fail(f"official train and test share {len(shared)} values of {dim}")
    inp.checks["manifests"] = {n: {"file": d["manifests"][n]["file"], "sha256": inp.sha[n], **counts_of(inp.rows[n])} for n in MANIFESTS}
    # ---------------------------------------------------------------- ordine di campionamento (spread per classe)
    k = d["cache"]["per_class"]
    order_train = nm.spread_keys(inp.rows["official_train"], k)
    order_test = nm.spread_keys(inp.rows["official_test"], k)
    if sorted(order_train) != sorted(list(inp.by_key["cnn_train"]) + list(inp.by_key["cnn_es"])):
        fail("the sampling order of the official train list does not give cnn_train + cnn_es")
    if order_test != nm.spread_keys(inp.rows["cnn_test"], k):
        fail("the sampling order of the official test list does not give cnn_test")
    if not set(inp.by_key["c1"]) <= set(order_test):
        fail("the C1 subset is not inside the 3,000 test frames")
    inp.order = {"train": order_train, "test": order_test}
    # ---------------------------------------------------------------- array y/s della cache contro i manifest
    z = np.load(cp, allow_pickle=False)
    inp.arrays = {}
    for split, src in (("train", "official_train"), ("test", "official_test")):
        keys = inp.order[split]
        y, s = z[f"y_{split}"], z[f"s_{split}"]
        exp_y = np.array([int(inp.by_key[src][q]["label"]) for q in keys], np.int64)
        exp_s = np.array([inp.by_key[src][q]["subject"] for q in keys])
        if len(y) != len(keys) or y.dtype.kind not in "iu" or not np.array_equal(y, exp_y):
            fail(f"cache y_{split}: labels differ from the manifests in sampling order (linkage refused)")
        if len(s) != len(keys) or s.dtype.kind != "U" or not np.array_equal(s, exp_s):
            fail(f"cache s_{split}: subjects differ from the manifests in sampling order (linkage refused)")
        inp.arrays[f"y_{split}"], inp.arrays[f"s_{split}"] = y, s
    inp.checks["linkage"] = ("y_train/s_train and y_test/s_test equal, element by element, to label and subject of the manifest rows in "
                             "sampling order (spread per class, bona fide first)")
    # ---------------------------------------------------------------- split per soggetto con il seed dello split
    tr_i, va_i = n1_trainer.split_indices(inp.arrays["y_train"], inp.arrays["s_train"], cfg["split_seed"])
    fit = [order_train[i] for i in tr_i]
    val = [order_train[i] for i in va_i]
    if sorted(fit) != sorted(inp.by_key["cnn_train"]) or sorted(val) != sorted(inp.by_key["cnn_es"]):
        fail(f"split seed {cfg['split_seed']}: GroupShuffleSplit gives {len(fit)} fitting / {len(val)} validation frames that do not "
             f"reproduce cnn_train ({len(inp.rows['cnn_train'])}) / cnn_es ({len(inp.rows['cnn_es'])}) by key")
    sha_of = {q: inp.by_key["official_train"][q]["sha256"] for q in order_train}
    inp.split_check = {"sha256": [sha_of[q] for q in order_train], "train": [r["sha256"] for r in inp.rows["cnn_train"]],
                       "val": [r["sha256"] for r in inp.rows["cnn_es"]]}
    n1_trainer.check_split(tr_i, va_i, inp.split_check)
    inp.tr_i, inp.va_i = tr_i, va_i
    fit_subj, val_subj = set(inp.arrays["s_train"][tr_i].tolist()), set(inp.arrays["s_train"][va_i].tolist())
    if fit_subj & val_subj:
        fail("fitting and validation share subjects")
    test_subj = set(inp.arrays["s_test"].tolist())
    inp.checks["split"] = {"split_seed": cfg["split_seed"], "n_fitting": len(fit), "n_validation": len(val),
                           "n_subjects_fitting": len(fit_subj), "n_subjects_validation": len(val_subj),
                           "equal_to_manifests": "cnn_train and cnn_es, by key and by file SHA-256 (multisets)",
                           "n_test_subjects": len(test_subj), "n_test_subjects_in_fitting": len(test_subj & fit_subj),
                           "n_test_subjects_in_validation": len(test_subj & val_subj)}
    inp.c1_keys = set(inp.by_key["c1"])
    inp.checks["c1_subset"] = {"n": len(inp.c1_keys), "inside_test": True,
                               "selected_by": "keys of nuaa_c1_v2.csv (SHA-256 in the configuration), each row equal to its cnn_test row"}
    # ---------------------------------------------------------------- frame derivati
    inp.z, inp._X = z, {}
    if frames:
        h_tr, h_te = nm.frame_hashes(frames_of(inp, "train")), nm.frame_hashes(frames_of(inp, "test"))
        out = {"X_train_duplicates": len(h_tr) - len(set(h_tr)), "X_test_duplicates": len(h_te) - len(set(h_te)),
               "fitting_validation_shared": len({h_tr[i] for i in tr_i} & {h_tr[i] for i in va_i}),
               "train_test_shared": len(set(h_tr) & set(h_te))}
        bad = {k: v for k, v in out.items() if v}
        if bad:
            fail(f"derived frames: {bad} (exact pixel hash)")
        inp.checks["derived_frames"] = out
    return inp


def frames_of(inp: Inputs, split: str) -> np.ndarray:
    """X_train o X_test della cache, letto una sola volta per processo."""
    if split not in inp._X:
        inp._X[split] = inp.z[f"X_{split}"]
    return inp._X[split]


# ============================================================================ check

def parse_common(a, need_study: bool = True):
    d = Path(a.study_dir)
    if msg := study_dir_problem(d):
        fail(f"study folder refused: {msg}")
    if msg := inputs_problem(a, d):
        fail(f"inputs refused: {msg}")
    cfg, cfg_sha = load_config(a.config)
    if msg := "; ".join(run_problems(cfg, is_trial(d))):
        fail(msg)
    if need_study and not d.is_dir():
        fail("study folder does not exist: run check first")
    return d, cfg, cfg_sha


def cmd_check(a) -> None:
    d, cfg, cfg_sha = parse_common(a, need_study=False)
    if d.exists() and any(d.iterdir()):
        fail("the study folder must be new (missing or empty): refused, nothing written")
    state = repo_state(d)
    m = RUN_ID_RE.match(d.name)
    if not is_trial(d) and not str(state.get("commit", "")).startswith(m.group(2)):
        fail(f"non-trial run: the study id must carry the current commit ({str(state.get('commit'))[:7]})")
    d.mkdir(parents=True, exist_ok=True)
    accept(d, "check", "check.log")
    print(f"== N1 check: study {d.name}, configuration SHA-256 {cfg_sha}")
    inp = load_inputs(cfg, a, frames=True)
    for k, v in inp.checks.items():
        print(f"   PASS {k}: {json.dumps(v, sort_keys=True) if not isinstance(v, str) else v}")
    doc = {"passed": True, "step": "check", "run_id": d.name, "trial": is_trial(d), "config_sha256": cfg_sha,
           "card_data": cfg["data"] == CARD_DATA, "checks": inp.checks, "detector": "not used", "weights": "not loaded",
           "tensorflow": "not imported", "code_sha256": code_hashes(), **state, "created": now()}
    write_all(d, {"check.json": json.dumps(doc, indent=1) + "\n"})
    print(f"== check PASS: cache, 6 manifests, sampling order, linkage, split seed {cfg['split_seed']} "
          f"({inp.checks['split']['n_fitting']}/{inp.checks['split']['n_validation']}), C1 subset, derived frames")


def load_record(path: Path, step: str, run_id: str, cfg_sha: str, check_sha: str = None) -> dict:
    """Record di un passo: file regolare, JSON, passo superato e legato a questo studio (run_id, passo, configurazione e,
    dopo il check, SHA-256 del check.json di questo studio)."""
    if path.is_symlink() or not path.is_file():
        fail(f"missing {path.parent.name}/{path.name}: run the previous step first")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, ValueError):
        fail(f"{path.name} is not valid JSON")
    if not isinstance(doc, dict) or doc.get("passed") is not True:
        fail(f"{path.name} does not record a passed step")
    if doc.get("step") != step or doc.get("run_id") != run_id or doc.get("config_sha256") != cfg_sha:
        fail(f"{path.name} belongs to another step, study or configuration: records of another run are refused")
    if check_sha is not None and doc.get("check_sha256") != check_sha:
        fail(f"{path.name} was not written after the check of this study (check.json SHA-256 differs)")
    return doc


def open_study(a) -> tuple:
    """Cartella dello studio dopo check: stessa configurazione, stesso codice e (run non -trial) stesso commit."""
    d, cfg, cfg_sha = parse_common(a)
    check = load_record(d / "check.json", "check", d.name, cfg_sha)
    if check.get("code_sha256") != code_hashes():
        fail("the code changed since check (SHA-256 of the scripts differ from check.json): start a new study")
    if not is_trial(d) and repo_state(d).get("commit") != check.get("commit"):
        fail("non-trial run: the repository commit differs from check")
    return d, cfg, cfg_sha, check, sha256_file(d / "check.json")


def seed_of(a, cfg) -> int:
    if a.train_seed not in cfg["train_seeds"]:
        fail(f"--train-seed {a.train_seed} is not one of the configuration seeds {cfg['train_seeds']}")
    return a.train_seed


def same_inputs(inp: Inputs, check: dict) -> None:
    if inp.checks.get("cache_sha256") != check["checks"]["cache_sha256"] or inp.checks["manifests"] != check["checks"]["manifests"]:
        fail("cache or manifests differ from those recorded at check")


# ============================================================================ train

def default_trainer(data, training, out_dir, train_seed, split_seed, split_check):
    try:
        import n1_trainer
        arch = Path(LIVEDETECTION) / "src" / "architectures.py"
        if not arch.is_file():
            fail("the reference architectures (LIVENESSLAB_LIVEDETECTION/src/architectures.py) are missing")
        mod = load_module("kuznetsov_architectures", arch)
    except ModuleNotFoundError as e:
        fail(f"missing Python package {e.name!r}: train needs TensorFlow and scikit-learn")
    return n1_trainer.train(data, training, out_dir, train_seed, split_seed, split_check, mod)


def weights_problems(ck: Path, cfg: dict, seed: int) -> list:
    """Scheda e pesi di un seed: SHA-256 dei file uguali alla scheda, campi della configurazione, epoca ripristinata =
    epoca di val_loss minimo, arresto coerente con la patience, pesi diversi dai checkpoint precedenti."""
    out = []
    card_p = ck / f"{CKPT_KEY}.json"
    if card_p.is_symlink() or not card_p.is_file():
        return [f"{card_p.name} missing"]
    try:
        card = json.loads(card_p.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, ValueError):
        return [f"{card_p.name} is not valid JSON"]
    for f, ext, key in (("weights_file", ".h5", "weights_sha256"), ("keras_file", ".keras", "keras_sha256")):
        p = ck / f"{CKPT_KEY}{ext}"
        if card.get(f) != p.name or p.is_symlink() or not p.is_file():
            out.append(f"{p.name} missing or not the file of the card"); continue
        if sha256_file(p) != card.get(key):
            out.append(f"{p.name}: SHA-256 differs from the card")
        if card.get(key) in cfg["data"]["previous_checkpoints_sha256"]:
            out.append(f"{p.name} is a previous checkpoint (C1), not a new training")
    t = cfg["training"]
    sp = cfg["data"]["manifests"]
    want = {"arch": "LivenessNet", "dataset": "nuaa", "protocol": "official", "train_seed": seed, "split_seed": cfg["split_seed"],
            "n_train": sp["cnn_train"]["rows"], "n_val": sp["cnn_es"]["rows"], "val_split": "by subject", "max_epochs": t["max_epochs"],
            "lr": t["initial_learning_rate"], "batch": t["batch"], "early_stopping_patience": t["early_stopping_patience"],
            "device": t["device"], "split_check": "passed: fitting and early-stopping frames equal to the manifests by SHA-256"}
    for k, v in want.items():
        if card.get(k) != v or type(card.get(k)) is not type(v):
            out.append(f"card field {k} = {card.get(k)!r}, expected {v!r}")
    vl = (card.get("history") or {}).get("val_loss")
    n = card.get("epochs_run")
    if not isinstance(vl, list) or not vl or not all(isinstance(x, float) and math.isfinite(x) for x in vl) or not _int(n) or len(vl) != n:
        return out + ["card history.val_loss missing, non-finite or not one value per epoch"]
    if not 1 <= n <= t["max_epochs"]:
        out.append(f"epochs_run {n} outside 1..{t['max_epochs']}")
    best = int(np.argmin(vl)) + 1
    if card.get("restored_epoch") != best or card.get("best_val_loss") != vl[best - 1]:
        out.append(f"restored epoch {card.get('restored_epoch')} is not the epoch of minimum val_loss ({best})")
    # Keras: l'early stopping scatta dopo `patience` epoche senza miglioramento, cioè all'epoca best + patience
    fired = n == best + t["early_stopping_patience"]
    if n < t["max_epochs"] and not fired:
        out.append(f"stopped after {n} epochs, not {t['early_stopping_patience']} epochs after the minimum (epoch {best})")
    if card.get("restored_by") != ("early stopping" if fired else "end of epochs (set_weights)"):
        out.append(f"restored_by {card.get('restored_by')!r} inconsistent with {n} epochs and minimum at epoch {best}")
    return out


def device_info() -> dict:
    out = {}
    try:
        import importlib.metadata as md
        for p in ("tensorflow", "tensorflow-metal", "keras", "numpy", "scikit-learn"):
            try:
                out[p] = md.version(p)
            except md.PackageNotFoundError:
                out[p] = "absent"
    except ImportError:
        pass
    if "tensorflow" in sys.modules:
        tf = sys.modules["tensorflow"]
        try:
            out["visible_devices"] = [x.device_type for x in tf.config.get_visible_devices()]
        except Exception:  # noqa: BLE001  (diagnostica: mai un errore)
            out["visible_devices"] = "n/d"
    out["python"] = sys.version.split()[0]
    out["platform"] = sys.platform
    return out


def cmd_train(a) -> None:
    d, cfg, cfg_sha, check, check_sha = open_study(a)
    seed = seed_of(a, cfg)
    if not a.i_am_authorized:
        fail("train needs --i-am-authorized (training on the restricted NUAA data)")
    sd = d / f"seed{seed}"
    if os.path.lexists(sd):
        fail(f"{sd.name} already exists: every seed has a new folder, never reused (refused, nothing changed)")
    sd.mkdir()
    accept(sd, "train", "train.log")
    print(f"== N1 train: study {d.name}, train seed {seed}, split seed {cfg['split_seed']}")
    inp = load_inputs(cfg, a, frames=True)
    same_inputs(inp, check)
    data = {"X_train": frames_of(inp, "train"), "y_train": inp.arrays["y_train"], "s_train": inp.arrays["s_train"]}
    ck = sd / CHECKPOINTS
    ck.mkdir()
    t0 = time.time()
    card = (TRAINER or default_trainer)(data, cfg["training"], ck, seed, cfg["split_seed"], inp.split_check)
    wall = time.time() - t0
    problems = weights_problems(ck, cfg, seed)
    if problems:
        fail("weights check failed: " + "; ".join(problems))
    card_p = ck / f"{CKPT_KEY}.json"
    doc = {"passed": True, "step": "train", "run_id": d.name, "train_seed": seed, "split_seed": cfg["split_seed"],
           "config_sha256": cfg_sha, "check_sha256": check_sha, "commit": check.get("commit"),
           "card_file": f"{CHECKPOINTS}/{card_p.name}", "card_sha256": sha256_file(card_p),
           "weights_file": f"{CHECKPOINTS}/{CKPT_KEY}.h5", "weights_sha256": card["weights_sha256"],
           "keras_file": f"{CHECKPOINTS}/{CKPT_KEY}.keras", "keras_sha256": card["keras_sha256"],
           "epochs_run": card["epochs_run"], "restored_epoch": card["restored_epoch"], "restored_by": card["restored_by"],
           "best_val_loss": card["best_val_loss"], "wall_seconds": round(wall, 1), "device": device_info(),
           "determinism": "seeds fixed (python, numpy, TensorFlow, tf.data shuffle); training on the CPU only (GPU hidden); parallel CPU "
                          "reductions can still give small differences between two runs with the same seed",
           "created": now()}
    write_all(sd, {"train.json": json.dumps(doc, indent=1) + "\n"})
    print(f"== train PASS: seed {seed}, {card['epochs_run']} epochs, restored epoch {card['restored_epoch']}, wall {wall:.0f}s, "
          f"weights {card['weights_sha256']}")


# ============================================================================ score

def tf_scorer(weights_path, weights_sha: str):
    """Scorer vero: architettura del codice di riferimento, pesi verificati per SHA-256, CPU, BatchNorm in inferenza;
    restituisce le probabilità softmax N x 2."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        import tensorflow as tf
    except ModuleNotFoundError as e:
        fail(f"missing Python package {e.name!r}: score needs TensorFlow")
    try:
        tf.config.set_visible_devices([], "GPU")     # CPU anche con tensorflow-metal
    except (RuntimeError, ValueError):
        pass
    tf.config.optimizer.set_experimental_options({"remapping": False})
    if sha256_file(weights_path) != weights_sha:
        fail(f"{Path(weights_path).name}: SHA-256 changed just before loading")
    arch = Path(LIVEDETECTION) / "src" / "architectures.py"
    if not arch.is_file():
        fail("the reference architectures (LIVENESSLAB_LIVEDETECTION/src/architectures.py) are missing")
    mod = load_module("kuznetsov_architectures", arch)
    with tf.device("/CPU:0"):
        model = mod.create_model("LivenessNet").get_model()
        model.load_weights(str(weights_path))

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


def metrics_api():
    try:
        from tesi_app.evaluation import compute_metrics
    except ModuleNotFoundError as e:
        fail(f"missing Python package {e.name!r}: metrics use src/tesi_app/evaluation.py (numpy, scikit-learn, OpenCV)")
    return compute_metrics


def set_metrics(y, s) -> dict:
    """Metriche a soglia 0,5 con i denominatori e AUROC; classe assente = None (NA)."""
    compute_metrics = metrics_api()
    y, s = np.asarray(y, dtype=int), np.asarray(s, dtype=float)
    m = compute_metrics(y, s, THRESHOLD)
    pred = s > THRESHOLD
    return {"n": m["n"], "n_bona_fide": m["n_real"], "n_attack": m["n_attack"],
            "accepted_attacks": int(np.sum((y == 1) & ~pred)), "rejected_bona_fide": int(np.sum((y == 0) & pred)),
            "ties_at_threshold": int(np.sum(s == THRESHOLD)), "threshold": THRESHOLD, "decision": DECISION_RULE,
            "apcer": m["apcer"], "bpcer": m["bpcer"], "acer": m["acer"], "auroc": m["auc"]}


def fmt_na(v) -> str:
    return NA if v is None else repr(float(v))


def per_subject(rows, set_name: str) -> list:
    out = []
    for subj in sorted({r["subject_id"] for r in rows}):
        sel = [r for r in rows if r["subject_id"] == subj]
        bf = [r for r in sel if r["label"] == "0"]
        at = [r for r in sel if r["label"] == "1"]
        rej = sum(1 for r in bf if r["prediction"] == "1")
        acc = sum(1 for r in at if r["prediction"] == "0")
        bpcer = rej / len(bf) if bf else None
        apcer = acc / len(at) if at else None
        acer = (apcer + bpcer) / 2 if bf and at else None
        out.append({"set": set_name, "subject_id": subj, "n_bona_fide": len(bf), "n_attack": len(at), "rejected_bona_fide": rej,
                    "accepted_attacks": acc, "bpcer": fmt_na(bpcer), "apcer": fmt_na(apcer), "acer": fmt_na(acer)})
    return out


def csv_text(rows, cols) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def read_csv_rows(p: Path, cols) -> list:
    if p.is_symlink() or not p.is_file():
        fail(f"{p.parent.name}/{p.name} missing or not a regular file")
    with p.open(newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        header = next(rd, [])
        body = list(rd)
    if header != cols or any(len(r) != len(cols) for r in body):
        fail(f"{p.name}: header or rows differ from the declared columns")
    return [dict(zip(cols, r)) for r in body]


def score_rows(d: Path, seed: int, inp: Inputs, s_att: np.ndarray, train: dict, cfg_sha: str) -> list:
    rows = []
    for i, k in enumerate(inp.order["test"]):
        r = inp.by_key["cnn_test"][k]
        v = float(s_att[i])
        rows.append({"run_id": d.name, "train_seed": str(seed), "sample_id": k, "subject_id": r["subject"], "session_id": r["session"],
                     "label": r["label"], "score_attack": repr(v), "prediction": "1" if v > THRESHOLD else "0",
                     "in_c1_subset": "1" if k in inp.c1_keys else "0", "file_sha256": r["sha256"],
                     "checkpoint_sha256": train["weights_sha256"], "config_sha256": cfg_sha})
    return sorted(rows, key=lambda r: r["sample_id"])


def metrics_from_rows(rows) -> dict:
    """Metriche dei due insiemi dalle righe dei punteggi (stesso calcolo in score e in aggregate)."""
    out = {}
    for name, sel in (("test_3000", rows), ("c1_300", [r for r in rows if r["in_c1_subset"] == "1"])):
        out[name] = set_metrics([int(r["label"]) for r in sel], [float(r["score_attack"]) for r in sel])
    return out


def cmd_score(a) -> None:
    d, cfg, cfg_sha, check, check_sha = open_study(a)
    seed = seed_of(a, cfg)
    if not a.i_am_authorized:
        fail("score needs --i-am-authorized (inference on the restricted NUAA data)")
    sd = d / f"seed{seed}"
    if sd.is_symlink() or not sd.is_dir():
        fail(f"{sd.name} missing or not a folder: run train first")
    found = [n for n in SCORE_OUTPUTS if os.path.lexists(sd / n)]
    if found:
        fail(f"{sd.name} already holds {', '.join(found)}: scores are never recomputed in place (refused, nothing changed)")
    train = load_record(sd / "train.json", "train", d.name, cfg_sha, check_sha)
    if train.get("train_seed") != seed:
        fail("train.json records another seed")
    accept(sd, "score", "score.log")
    print(f"== N1 score: study {d.name}, train seed {seed}, CPU")
    if problems := weights_problems(sd / CHECKPOINTS, cfg, seed):
        fail("weights check failed: " + "; ".join(problems))
    if sha256_file(sd / train["weights_file"]) != train["weights_sha256"]:
        fail("weights differ from train.json")
    inp = load_inputs(cfg, a, frames=False)
    same_inputs(inp, check)
    scorer = (MAKE_SCORER or tf_scorer)(sd / train["weights_file"], train["weights_sha256"])
    pv = probabilities(scorer, frames_of(inp, "train")[inp.va_i])
    vl = val_loss(pv, inp.arrays["y_train"][inp.va_i])
    diff = abs(vl - train["best_val_loss"])
    if not (diff <= VAL_LOSS_TOL[0] or diff <= VAL_LOSS_TOL[1] * train["best_val_loss"]):
        fail(f"restored weights: val_loss recomputed on CPU {vl:.6f}, recorded minimum {train['best_val_loss']:.6f}: "
             "the checkpoint is not the one of minimum val_loss")
    pt = probabilities(scorer, frames_of(inp, "test"))
    rows = score_rows(d, seed, inp, pt[:, 1], train, cfg_sha)
    scores_txt = csv_text(rows, SCORE_COLUMNS)
    back = list(csv.DictReader(io.StringIO(scores_txt)))
    if any(float(b["score_attack"]) != float(r["score_attack"]) for b, r in zip(back, rows)) or len(back) != len(rows):
        fail("scores do not survive the CSV round trip")
    met = metrics_from_rows(back)
    subj = [{"run_id": d.name, "train_seed": str(seed), **x} for name in SETS
            for x in per_subject(back if name == "test_3000" else [r for r in back if r["in_c1_subset"] == "1"], name)]
    subj_txt = csv_text(subj, SUBJECT_COLUMNS)
    met_doc = {"run_id": d.name, "train_seed": seed, "sets": SETS, "metrics": met, "weights_sha256": train["weights_sha256"]}
    met_txt = json.dumps(met_doc, indent=1) + "\n"
    doc = {"passed": True, "step": "score", "run_id": d.name, "train_seed": seed, "config_sha256": cfg_sha, "check_sha256": check_sha,
           "train_sha256": sha256_file(sd / "train.json"), "weights_sha256": train["weights_sha256"], "device": "CPU",
           "val_loss_recomputed": vl, "val_loss_recorded": train["best_val_loss"], "val_loss_abs_diff": diff,
           "files": {"scores_test.csv": sha256_text(scores_txt), "per_subject.csv": sha256_text(subj_txt), "metrics.json": sha256_text(met_txt)},
           "n_scores": len(rows), "n_c1": sum(1 for r in rows if r["in_c1_subset"] == "1"), "created": now()}
    write_all(sd, {"scores_test.csv": scores_txt, "per_subject.csv": subj_txt, "metrics.json": met_txt,
                   "score.json": json.dumps(doc, indent=1) + "\n"})
    for name in SETS:
        m = met[name]
        print(f"   {name}: APCER {fmt_na(m['apcer'])} ({m['accepted_attacks']}/{m['n_attack']}), BPCER {fmt_na(m['bpcer'])} "
              f"({m['rejected_bona_fide']}/{m['n_bona_fide']}), ACER {fmt_na(m['acer'])}, AUROC {fmt_na(m['auroc'])}")
    print(f"== score PASS: seed {seed}, {len(rows)} test scores (C1 {doc['n_c1']}), val_loss check |diff| {diff:.2e}")


# ============================================================================ aggregate

def summary(values: list) -> dict:
    if any(v is None for v in values):
        return {"values": values, "median": None, "min": None, "max": None}
    return {"values": values, "median": statistics.median(values), "min": min(values), "max": max(values)}


def f4(v) -> str:
    return NA if v is None else f"{v:.4f}"


def public_problems(text: str, inp: Inputs) -> list:
    """Contenuto vietato nei file pubblicabili: chiavi o SHA-256 delle immagini, identificativi di soggetto NUAA, testo
    vietato del pacchetto (percorsi personali, credenziali, indirizzi, nomi di strumenti)."""
    out = []
    if re.search(r"\b(real|attack)/\d{4}_", text):
        out.append("image keys")
    shas = {r["sha256"] for rows in inp.rows.values() for r in rows}
    if set(re.findall(r"[0-9a-f]{64}", text)) & shas:
        out.append("image SHA-256 values")
    subjects = {r["subject"] for rows in inp.rows.values() for r in rows}
    if any(re.search(rf"(?<![\w.]){re.escape(s)}(?![\w.])", text) for s in subjects):
        out.append("subject identifiers")
    for label, rx in FORBIDDEN_TEXT:
        if rx.search(text):
            out.append(label)
    return out


def table_md(doc: dict) -> str:
    seeds = [str(s) for s in doc["train_seeds"]]
    L = [f"# N1 part 2: LivenessNet on NUAA, three training seeds (study `{doc['run_id']}`)", "",
         f"Split seed {doc['split_seed']} for every run (2,507 fitting / 493 validation frames, M2 manifests); training seeds "
         f"{', '.join(seeds)}. Threshold {THRESHOLD} on the attack probability, {DECISION_RULE}. Scores on CPU. "
         "No seed selection: every seed of the configuration is reported. Median and range (minimum-maximum) over the three seeds.",
         ""]
    for name, title in SETS.items():
        L += [f"## {title}", "", "| metric | " + " | ".join(f"seed {s}" for s in seeds) + " | median | range |",
              "|---|" + "---|" * (len(seeds) + 2)]
        for met in ("apcer", "bpcer", "acer", "auroc"):
            s = doc["summary"][name][met]
            L.append(f"| {met.upper()} | " + " | ".join(f4(v) for v in s["values"]) + f" | {f4(s['median'])} | "
                     + (NA if s["min"] is None else f"{s['min']:.4f}-{s['max']:.4f}") + " |")
        cnt = [doc["seeds"][s]["metrics"][name] for s in seeds]
        L.append("| accepted attacks / attacks | " + " | ".join(f"{c['accepted_attacks']}/{c['n_attack']}" for c in cnt) + " | | |")
        L.append("| rejected bona fide / bona fide | " + " | ".join(f"{c['rejected_bona_fide']}/{c['n_bona_fide']}" for c in cnt) + " | | |")
        L.append("")
    L += ["## Training", "", "| | " + " | ".join(f"seed {s}" for s in seeds) + " |", "|---|" + "---|" * len(seeds)]
    for k, lab in (("epochs_run", "epochs run (max 12)"), ("restored_epoch", "restored epoch (minimum val_loss)"),
                   ("best_val_loss", "minimum val_loss"), ("val_loss_recomputed", "val_loss recomputed on CPU"),
                   ("wall_seconds", "training wall time (s)")):
        vals = [doc["seeds"][s]["training"][k] for s in seeds]
        L.append(f"| {lab} | " + " | ".join(f"{v:.6f}" if isinstance(v, float) and k != "wall_seconds" else str(v) for v in vals) + " |")
    L.append("| weights SHA-256 (.h5) | " + " | ".join(f"`{doc['seeds'][s]['training']['weights_sha256'][:12]}…`" for s in seeds) + " |")
    L += ["", "NA = not computable (class absent). Values are rounded to four decimals; aggregate.json holds the exact values.", ""]
    return "\n".join(L)


def cmd_aggregate(a) -> None:
    d, cfg, cfg_sha, check, check_sha = open_study(a)
    if a.train_seed is not None:
        fail("aggregate takes every seed of the configuration (no --train-seed: no seed selection)")
    for n in AGG_OUTPUTS:
        q = d / n
        if os.path.lexists(q) and not q.is_symlink() and not q.is_file():
            fail(f"{n} in the study folder is not a regular file: refused, nothing removed")
    accept(d, "aggregate", "aggregate.log")
    for n in AGG_OUTPUTS:                         # uscite precedenti tolte prima del ricalcolo (nessun PASS vecchio)
        q = d / n
        if q.is_symlink() or q.is_file():
            q.unlink()
            print(f"   removed previous {n}")
    inp = load_inputs(cfg, a, frames=False)
    same_inputs(inp, check)
    seeds, subj_all, weights = {}, [], set()
    for seed in cfg["train_seeds"]:
        sd = d / f"seed{seed}"
        train = load_record(sd / "train.json", "train", d.name, cfg_sha, check_sha)
        sc = load_record(sd / "score.json", "score", d.name, cfg_sha, check_sha)
        if train.get("train_seed") != seed or sc.get("train_seed") != seed:
            fail(f"seed{seed}: records of another seed")
        if sc.get("train_sha256") != sha256_file(sd / "train.json") or sc.get("weights_sha256") != train["weights_sha256"]:
            fail(f"seed{seed}: score.json does not belong to its train.json")
        if problems := weights_problems(sd / CHECKPOINTS, cfg, seed):
            fail(f"seed{seed}: weights check failed: " + "; ".join(problems))
        for n, h in sc["files"].items():
            if (sd / n).is_symlink() or not (sd / n).is_file() or sha256_file(sd / n) != h:
                fail(f"seed{seed}/{n} changed after scoring (SHA-256 differs from score.json)")
        rows = read_csv_rows(sd / "scores_test.csv", SCORE_COLUMNS)
        keys = [r["sample_id"] for r in rows]
        if sorted(keys) != sorted(inp.by_key["cnn_test"]) or len(set(keys)) != len(keys):
            fail(f"seed{seed}/scores_test.csv does not hold exactly the cnn_test rows")
        for r in rows:
            m = inp.by_key["cnn_test"][r["sample_id"]]
            exp = {"run_id": d.name, "train_seed": str(seed), "subject_id": m["subject"], "session_id": m["session"], "label": m["label"],
                   "in_c1_subset": "1" if r["sample_id"] in inp.c1_keys else "0", "file_sha256": m["sha256"],
                   "checkpoint_sha256": train["weights_sha256"], "config_sha256": cfg_sha}
            try:
                v = float(r["score_attack"])
            except ValueError:
                v = float("nan")
            if any(r[k] != x for k, x in exp.items()) or not (0 <= v <= 1) or r["prediction"] != ("1" if v > THRESHOLD else "0"):
                fail(f"seed{seed}/scores_test.csv: rows inconsistent with the manifests, the checkpoint or the decision rule")
        met = metrics_from_rows(rows)
        stored = json.loads((sd / "metrics.json").read_text(encoding="utf-8"))["metrics"]
        if json.dumps(met, sort_keys=True) != json.dumps(stored, sort_keys=True):
            fail(f"seed{seed}: metrics recomputed from the scores differ from metrics.json")
        weights.add(train["weights_sha256"])
        subj_all += read_csv_rows(sd / "per_subject.csv", SUBJECT_COLUMNS)
        seeds[str(seed)] = {"metrics": met, "training": {k: train[k] for k in ("epochs_run", "restored_epoch", "restored_by", "best_val_loss",
                                                                              "wall_seconds", "weights_sha256", "keras_sha256")}
                            | {"val_loss_recomputed": sc["val_loss_recomputed"], "device": train["device"]},
                            "scores_sha256": sc["files"]["scores_test.csv"]}
    if len(weights) != len(cfg["train_seeds"]):
        fail("two seeds have the same weights")
    summ = {name: {met: summary([seeds[str(s)]["metrics"][name][met] for s in cfg["train_seeds"]]) for met in ("apcer", "bpcer", "acer", "auroc")}
            for name in SETS}
    doc = {"passed": True, "step": "aggregate", "run_id": d.name, "config_sha256": cfg_sha, "check_sha256": check_sha,
           "commit": check.get("commit"), "dirty_at_check": check.get("dirty"), "code_sha256": check["code_sha256"],
           "train_seeds": cfg["train_seeds"], "split_seed": cfg["split_seed"], "sets": SETS, "threshold": THRESHOLD, "decision": DECISION_RULE,
           "seeds": seeds, "summary": summ, "selection": "none: every seed of the configuration, no seed chosen on the test",
           "created": now()}
    js, md = json.dumps(doc, indent=1) + "\n", table_md(doc)
    for name, text in (("aggregate.json", js), ("aggregate.md", md)):
        if probs := public_problems(text, inp):
            fail(f"{name} would hold {', '.join(probs)}: not written")
    write_all(d, {"aggregate.json": js, "aggregate.md": md, "per_subject_all_seeds.csv": csv_text(subj_all, SUBJECT_COLUMNS)})
    print(md)
    print(f"== aggregate PASS: {len(seeds)} seeds, aggregate.json and aggregate.md (publishable), per_subject_all_seeds.csv (restricted)")


# ============================================================================ main

COMMANDS = {"check": cmd_check, "train": cmd_train, "score": cmd_score, "aggregate": cmd_aggregate}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="N1 part 2: LivenessNet on NUAA with three training seeds")
    ap.add_argument("command", choices=list(COMMANDS))
    ap.add_argument("--config", required=True, help="configuration JSON (n1_nuaa_stability.json)")
    ap.add_argument("--manifests", required=True, help="folder with the six M2 NUAA manifests")
    ap.add_argument("--cache", required=True, help="the local cache nuaa_1500.npz")
    ap.add_argument("--study-dir", required=True, help="study folder YYYYMMDD-N1-LivenessNet-<shortcommit>[-trial]")
    ap.add_argument("--train-seed", type=int, default=None, help="training seed (train and score)")
    ap.add_argument("--i-am-authorized", action="store_true", help="training and inference on the restricted data are authorised")
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    RUN_STATE.update(dir=None, cmd=a.command, log=None)
    code = 0
    try:
        if a.command in ("train", "score") and a.train_seed is None:
            fail(f"{a.command} needs --train-seed")
        COMMANDS[a.command](a)
        write_status("PASS")
        print(f"RESULT: PASS ({a.command})")
    except (N1Error, SystemExit, Exception, KeyboardInterrupt) as e:  # noqa: BLE001
        if isinstance(e, SystemExit) and e.code in (None, 0):
            raise
        msg = str(e) if isinstance(e, (N1Error, SystemExit)) else f"unexpected {type(e).__name__}: {e}"
        if isinstance(e, KeyboardInterrupt):
            msg = "interrupted"
        print(f"ERROR: {msg}", file=sys.stderr)
        try:
            write_status("FAIL", msg)
        except BaseException as e2:  # noqa: BLE001
            print(f"ERROR: status.json not written ({type(e2).__name__})", file=sys.stderr)
        print(f"RESULT: FAIL ({a.command})")
        code = 130 if isinstance(e, KeyboardInterrupt) else 1
    finally:
        close_log()
    return code


if __name__ == "__main__":
    sys.exit(main())
