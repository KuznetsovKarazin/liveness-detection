"""
Pilota C2 (CASIA-FASD → NUAA, LivenessNet e AttackNet V2.2, seed 42): implementazione minima del protocollo approvato
(delivery/c2/pilot/, o ../../m2-c2-pilot-protocol/ nel pacchetto: README.md, c2_pilot_casia_to_nuaa.json,
manifest_table.md), con la configurazione emendata nel solo dispositivo del training (c2_pilot_casia_to_nuaa_v2.json:
delivery/c2/livenessnet_device/, o ../c2_pilot_casia_to_nuaa_v2.json nel pacchetto; l'originale approvata resta invariata).

Sottocomandi, sempre con --config <file JSON> e --run-dir <cartella del run>:
  check    solo manifest, cache e split, prima di qualunque training: SHA-256 e conteggi dei manifest contro la
           configurazione (righe, classi, soggetti, video), validazione stretta delle righe (codice delle tranche M2),
           disgiunzione di fitting, early stopping e test sorgente per chiave, SHA-256 del file, soggetto e video (fra
           train_img e test_img i soggetti sono numerati localmente allo split: la loro separazione è quella dello split
           del mirror, verificata, mentre la disgiunzione delle persone è documentata dal protocollo), separazione fra
           dataset (nessun file né frame derivato CASIA uguale a uno NUAA), collegamento cache ↔ manifest (ordine di
           campionamento, array y/s, conteggi), nessun checkpoint pooled o esistente nella configurazione. Nessun
           rilevatore, nessun peso, nessun TensorFlow. Scrive check.json (SHA-256 di configurazione, manifest, cache e
           codice; stato del repository), letto da tutti i passi successivi.
  train    delega a scripts/train_cnn.py (train, con out_dir, split_check e test_eval=False) i due training con gli
           iperparametri della configurazione; i pesi vanno in <run>/checkpoints, mai in models/weights; dopo il
           training controlla pesi e schede (SHA-256 del file contro la scheda, campi, epoca ripristinata, pesi esistenti
           o pooled rifiutati, dispositivo del training) e scrive train.json, con il dispositivo richiesto ed effettivo
           di ogni modello. Parte solo con --i-am-authorized.
  score    --set dev oppure --set source_test,target: punteggi grezzi (score_attack con precisione di andata e ritorno,
           threshold e prediction null) nei campi di research/templates/score-schema.json più model_id e file_sha256;
           su CPU, con BatchNorm in inferenza (training=False). Prima di ogni punteggio ripete il controllo dei pesi,
           compreso il dispositivo del training registrato (LivenessNet solo CPU con scarto grafo/eager entro
           GRAPH_EAGER_TOL, TRAINING_DEVICE). Il
           secondo punteggio è rifiutato senza threshold.json e ne registra lo SHA-256 prima di calcolare i punteggi dei
           test. Parte solo con --i-am-authorized.
  select   legge solo raw_scores_dev.csv; rifiuta ogni riga la cui chiave o il cui SHA-256 appartiene al test sorgente o
           a NUAA, punteggi incompleti o incoerenti con il manifest del dev e un dev con una sola classe; sceglie la
           soglia con le funzioni di scripts/select_threshold.py importate invariate (criterio eer dell'emendamento M1
           del 2/10) e scrive threshold.json (valori, conteggi aggregati del dev, SHA-256; nessun identificativo).
  apply    rifiuta un threshold.json cambiato dopo la registrazione o diverso dal ricalcolo sul dev e punteggi grezzi
           incompleti o incoerenti (righe, chiavi, duplicati, valori non finiti, campi diversi dal manifest); scrive i
           record di decisione scores_E1..E4.csv e scores_dev.csv (threshold = valore congelato, prediction = 1 se e
           solo se score_attack > threshold) e la tabella aggregata c2_table.json / c2_table.md (alla soglia del dev e a
           0,5, etichettata), con APCER aggregato e per tipo di attacco, BPCER, ACER, denominatori e soggetti per classe.
  all      i passi in ordine, fermandosi al primo errore (con --i-am-authorized e --epochs).

Comportamento su errore e su rifiuto (unico stato corrente: status.json nella cartella del run):
  - rifiuti prima che la cartella del run sia accettata (nome, posizione protetta, cartella non -trial non nuova al
    check, cartella inesistente) e errori negli argomenti: nessun file scritto né cancellato;
  - appena la cartella è accettata status.json diventa RUNNING (prima di qualunque cancellazione);
  - rifiuti dell'identità del run (check.json mancante, non superato o di un'altra cartella, configurazione, codice o
    commit diversi da quelli registrati al check): le uscite esistenti restano come sono (appartengono al run
    registrato) e status.json diventa FAIL con il passo, il motivo e outputs = "untouched";
  - ogni altro rifiuto o errore (autorizzazione, --epochs, ingressi, pesi, punteggi, soglia, eccezioni impreviste,
    interruzione): il passo ha già cancellato le proprie uscite e quelle dei passi successivi (cartelle con
    scripts/path_guard.py) e status.json diventa FAIL con outputs = "removed";
  - i file di un passo si scrivono solo alla fine (temporanei e rinomine; su qualunque eccezione non resta nessun file
    del gruppo); su errore "ERROR: ..." e "RESULT: FAIL (<comando>)", codice 1 (130 su KeyboardInterrupt); su
    successo status.json = PASS (se non si può scrivere, il comando fallisce).
La cartella del run deve chiamarsi YYYYMMDD-C2-seed42-<commit breve> ed essere nuova al check (inesistente, vuota o con
il solo file regolare status.json: anche un .DS_Store la rende non nuova), oppure finire con -trial; dentro il
repository è ammessa solo sotto results/c2/. Un run non -trial richiede la configurazione emendata
(APPROVED_CONFIG_SHA256), un albero git pulito, il commit del nome del run e lo stesso commit in ogni passo.

Dipendenze: numpy (tutti i passi); scikit-learn per train (split) e per select/apply (AUC, tramite
src/tesi_app/evaluation.py, che importa anche OpenCV); TensorFlow solo per train e score con i modelli veri.
"""
import argparse
import csv
import functools
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np  # noqa: E402
    import path_guard as pg  # noqa: E402
    import rgb_manifests as rm  # noqa: E402  (CASIA-FASD: lettura e validazione dei manifest, ordine di campionamento)
    import nuaa_manifests as nm  # noqa: E402  (NUAA: lettura e validazione dei manifest, hash dei frame)
    import select_threshold as st  # noqa: E402  (selettore M1, invariato: threshold_counts, candidate_thresholds, select_threshold)
except ModuleNotFoundError as _e:   # dipendenza mancante: messaggio chiaro, mai un traceback
    print(f"ERROR: missing Python package {_e.name!r}: numpy is needed by every subcommand", file=sys.stderr)
    print("RESULT: FAIL (import)")
    sys.exit(1)
from tesi_app.paths import DATA, LIVEDETECTION, RESULTS, WEIGHTS, load_module  # noqa: E402

# SHA-256 della configurazione emendata (delivery/c2/livenessnet_device/SHA256SUMS, PR #12): quella approvata con la PR #7
# (4065d853…f84f7a) con il solo training.device per architettura, uguale a TRAINING_DEVICE; un run non -trial accetta
# solo questa
APPROVED_CONFIG_SHA256 = "4905f950c05512ce60412acab3b65fc7dde445a99158f7303b5b87bdb36f14e9"
CONFIG_SCHEMA = "c2-pilot-config/draft-1"
SEED = 42
RUN_ID_RE = re.compile(r"^(\d{8})-C2-seed42-([0-9a-f]{7,40})(-trial)?$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ARCHS = ("LivenessNet", "AttackNetV2_2")
# dispositivo del training richiesto per architettura, uguale a DEVICE_POLICY di scripts/train_cnn.py: LivenessNet solo
# su CPU (tensorflow-metal salta la ReLU del suo Dense dopo il Flatten sulla GPU in modalità grafo, pilota N1);
# AttackNet V2.2 sul dispositivo predefinito di TensorFlow, come nel protocollo
TRAINING_DEVICE = {"LivenessNet": "CPU", "AttackNetV2_2": "default"}
GRAPH_EAGER_TOL = 1e-5      # uguale a scripts/train_cnn.py: scarto relativo massimo grafo/eager per la politica CPU
# campi della registrazione del dispositivo scritta da train_cnn.py (né più né meno)
DEVICE_KEYS = ("requested", "expected", "effective", "visible", "physical", "graph_eager_max_rel_diff", "graph_eager_layer")
# ruolo -> (dataset, manifest M2): nessun nuovo split, nessun manifest pooled (protocollo approvato, D-3)
ROLES = {"source_train": ("casia_fasd", "cnn_train"), "source_dev": ("casia_fasd", "cnn_es"),
         "source_test": ("casia_fasd", "cnn_test"), "target_test": ("nuaa", "cnn_test")}
SOURCE_ROLES = ("source_train", "source_dev", "source_test")
EVALUATIONS = {"E1": (0, "source_test"), "E2": (1, "source_test"), "E3": (0, "target_test"), "E4": (1, "target_test")}
# tipo di attacco CASIA-FASD per token video (interfaccia Idiap bob.db.casia_fasd v2.0.9, get_type; dichiarato nel protocollo)
CASIA_ATTACK_TYPE = {"3": "warped", "4": "warped", "HR_2": "warped", "5": "cut", "6": "cut", "HR_3": "cut",
                     "7": "video", "8": "video", "HR_4": "video"}
ATTACK_TYPES = {"casia_fasd": ("cut", "video", "warped"), "nuaa": ("print",)}
NULL = "null"
SCORE_COLUMNS = ["run_id", "sample_id", "dataset_id", "split", "subject_id", "session_id", "video_id", "attack_type", "modality",
                 "label", "score_attack", "threshold", "prediction", "checkpoint_sha256", "config_sha256", "model_id", "file_sha256"]
RAW = {"dev": "raw_scores_dev.csv", "source_test": "raw_scores_source_test.csv", "target_test": "raw_scores_target.csv"}
RAW_SPLIT = {"source_dev": "dev", "source_test": "test", "target_test": "test"}
SCORE_BATCH = 16
DECISION_RULE = "attack if score_attack > value (tie = bona fide)"
SELECTION_RULE = ("criterion eer (M1 amendment of 2 October 2026): candidates = distinct dev scores plus nextafter(min, -inf) "
                  "if >= 0; attack if score > value (tie = bona fide); minimum of |fp*n_attack - fn*n_bona_fide| on integers, "
                  "ties -> highest value; functions threshold_counts, candidate_thresholds, select_threshold of "
                  "scripts/select_threshold.py, unchanged")
CODE_FILES = ("scripts/c2_pilot.py", "scripts/train_cnn.py", "scripts/select_threshold.py", "scripts/path_guard.py",
              "scripts/rgb_manifests.py", "scripts/nuaa_manifests.py", "scripts/package_rules.py", "src/tesi_app/evaluation.py",
              "src/tesi_app/paths.py")
# architetture del codice di riferimento, usate da train (train_cnn.py) e da score: anche fuori dal repository
ARCH_KEY = "LIVENESSLAB_LIVEDETECTION/src/architectures.py"
# uscite di ogni passo, nell'ordine dei passi: un passo cancella le proprie e quelle dei passi successivi
STEPS = ("check", "train", "score_dev", "select", "score_tests", "apply")
OUTPUTS = {
    "check": ["check.json"],
    "train": ["train.json"],
    "score_dev": [RAW["dev"], "score_dev.json"],
    "select": ["threshold.json"],
    "score_tests": [RAW["source_test"], RAW["target_test"], "score_tests.json"],
    "apply": [f"scores_{e}.csv" for e in EVALUATIONS] + ["scores_dev.csv", "c2_table.json", "c2_table.md", "subject_dispersion.json",
              "apply.json"],
}
CHECKPOINTS = "checkpoints"
# punti di iniezione (test e run): addestramento, costruzione dello scorer, stato del repository
TRAINER = None          # funzione con la firma di train_cnn.train; None: scripts/train_cnn.py
MAKE_SCORER = None      # (arch, weights_path, weights_sha256) -> funzione(X uint8) -> punteggi di attacco; None: TensorFlow su CPU
REPO_STATE = None       # (run_dir) -> {"commit", "dirty", "dirty_excludes"}; None: rgb_manifests.repo_state_excluding


STATUS = "status.json"
# stato dell'esecuzione corrente: cartella del run accettata (dove scrivere status.json), passo, uscite cancellate
RUN_STATE = {"run": None, "step": None, "removed": False, "cmd": None}


class C2Error(Exception):
    """Errore esplicito di un passo: messaggio senza traceback, RESULT: FAIL, codice 1."""


def fail(msg: str) -> None:
    raise C2Error(msg)


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
    """Chiama una funzione delle tranche M2 (che si ferma con sys.exit o ManifestError) trasformando l'errore in C2Error."""
    try:
        return fn(*args)
    except rm.ManifestError as e:
        fail(f"{what}: {e}")
    except SystemExit as e:
        fail(f"{what}: invalid input (see the ERROR line above)" if e.code in (None, 0, 1) else f"{what}: {e.code}")


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
    """Controlli strutturali della configurazione: insiemi chiusi, nessun ripiego. Lista vuota se valida."""
    bad = []
    if cfg.get("schema") != CONFIG_SCHEMA:
        bad.append(f"schema {cfg.get('schema')!r} is not {CONFIG_SCHEMA!r}")
    if cfg.get("seed") != SEED or not _int(cfg.get("seed")):
        bad.append(f"seed {cfg.get('seed')!r} is not {SEED}")
    if cfg.get("direction") != {"source": "casia_fasd", "target": "nuaa"}:
        bad.append("direction is not casia_fasd -> nuaa")
    models = cfg.get("models")
    if not isinstance(models, list) or len(models) != 2:
        bad.append("models must list exactly two models")
    else:
        for m in models:
            if not isinstance(m, dict) or set(m) != {"id", "arch"}:
                bad.append(f"model entry {m!r} must have exactly the fields id and arch (no checkpoint: new training, D-1)")
            elif not re.fullmatch(r"[a-z0-9_]+", str(m["id"])) or "pooled" in str(m["id"]):
                bad.append(f"model id {m['id']!r} not allowed")
        if all(isinstance(m, dict) for m in models):
            if sorted(m.get("arch") for m in models) != sorted(ARCHS):
                bad.append(f"model architectures {[m.get('arch') for m in models]} are not {list(ARCHS)}")
            if len({m.get("id") for m in models}) != len(models):
                bad.append("model ids are not unique")
    ck = cfg.get("checkpoints")
    if not isinstance(ck, dict):
        bad.append("checkpoints block missing")
    else:
        if not str(ck.get("proposed", "")).startswith("new training"):
            bad.append("checkpoints.proposed is not a new training (D-1)")
        for blk in ("excluded_pooled", "alternative_not_recommended"):
            d = ck.get(blk)
            hashes = [v for k, v in (d or {}).items() if k.endswith(".h5")] if isinstance(d, dict) else []
            if len(hashes) != 2 or not all(isinstance(h, str) and HEX64.match(h) for h in hashes):
                bad.append(f"checkpoints.{blk} must list the SHA-256 of two .h5 files")
        for k in ck:
            if k not in ("proposed", "alternative_not_recommended", "excluded_pooled", "output_dir", "reason_source_side_only",
                         "pre_commitment", "check_of_new_weights"):
                bad.append(f"checkpoints.{k} not allowed")
    mans = cfg.get("manifests")
    if not isinstance(mans, dict):
        bad.append("manifests block missing")
    else:
        for role, (ds, name) in ROLES.items():
            e = mans.get(role)
            if not isinstance(e, dict):
                bad.append(f"manifests.{role} missing"); continue
            if e.get("name") != f"{ds}/{name}":
                bad.append(f"manifests.{role}.name {e.get('name')!r} is not {ds}/{name}")
            if "pooled" in str(e.get("file", "")) or not str(e.get("file", "")).endswith(".csv") or "/" in str(e.get("file", "")):
                bad.append(f"manifests.{role}.file {e.get('file')!r} not allowed")
            if not isinstance(e.get("sha256"), str) or not HEX64.match(e["sha256"]):
                bad.append(f"manifests.{role}.sha256 is not a SHA-256")
            for f in ("rows", "n_bona_fide", "n_attack", "n_subjects"):
                if not _int(e.get(f)) or e[f] < 0:
                    bad.append(f"manifests.{role}.{f} is not a non-negative integer")
            if _int(e.get("rows")) and _int(e.get("n_bona_fide")) and _int(e.get("n_attack")) and e["rows"] != e["n_bona_fide"] + e["n_attack"]:
                bad.append(f"manifests.{role}: rows differ from n_bona_fide + n_attack")
            if ds == "casia_fasd" and (not _int(e.get("n_videos")) or e["n_videos"] < 0):
                bad.append(f"manifests.{role}.n_videos is not a non-negative integer")
    df = cfg.get("derived_frames")
    if not isinstance(df, dict) or not isinstance(df.get("source"), dict) or not isinstance(df.get("target"), dict):
        bad.append("derived_frames block missing")
    else:
        for side, members in (("source", ("X_train", "X_test")), ("target", ("X_test",))):
            e = df[side]
            if not re.fullmatch(r"[a-z_]+_[1-9]\d*\.npz", str(e.get("file", ""))):
                bad.append(f"derived_frames.{side}.file {e.get('file')!r} is not <dataset>_<per class>.npz")
            if not isinstance(e.get("sha256"), str) or not HEX64.match(e["sha256"]):
                bad.append(f"derived_frames.{side}.sha256 is not a SHA-256")
            for mbr in members:
                if not _int(e.get(mbr)) or e[mbr] <= 0:
                    bad.append(f"derived_frames.{side}.{mbr} is not a positive integer")
    b = cfg.get("budget")
    if not isinstance(b, dict) or not _int(b.get("max_epochs")) or b["max_epochs"] < 1 or not _int(b.get("batch")) or b["batch"] < 1 \
            or not isinstance(b.get("learning_rate"), float) or not b["learning_rate"] > 0 or not _int(b.get("wall_time_cap_minutes")):
        bad.append("budget: max_epochs, batch, learning_rate and wall_time_cap_minutes required")
    t = cfg.get("threshold")
    if not isinstance(t, dict) or t.get("criterion") != "eer" or t.get("set") != "source_dev only":
        bad.append("threshold must be criterion eer on source_dev only")
    ev = cfg.get("evaluations")
    ids = [m.get("id") for m in models] if isinstance(models, list) and all(isinstance(m, dict) for m in models) else []
    want = [{"id": e, "model": ids[i] if len(ids) > i else None, "set": s} for e, (i, s) in EVALUATIONS.items()]
    if ev != want:
        bad.append("evaluations must be E1-E4: each model on source_test and on target_test, in the declared order")
    if not isinstance(cfg.get("scoring"), dict) or cfg["scoring"].get("device") != "CPU":
        bad.append("scoring.device must be CPU")
    tdev = cfg.get("training", {}).get("device") if isinstance(cfg.get("training"), dict) else None
    if not isinstance(tdev, dict) or sorted(tdev) != sorted([*ARCHS, "note"]) or {a: tdev[a] for a in ARCHS} != TRAINING_DEVICE \
            or not isinstance(tdev["note"], str) or not tdev["note"].strip():
        bad.append(f"training.device must give exactly the device policy of each architecture, {TRAINING_DEVICE}, and a note "
                   "(amended configuration c2_pilot_casia_to_nuaa_v2.json)")
    return bad


def model_list(cfg: dict) -> list:
    return [(m["id"], m["arch"]) for m in cfg["models"]]


# ============================================================================ cartella del run e stato del repository

def protected_extra() -> list:
    """Cartelle protette spostabili con le variabili d'ambiente (pesi, dati, risultati consegnati)."""
    return [WEIGHTS, DATA, RESULTS / "c1", RESULTS / "m1", RESULTS / "m2", RESULTS / "m2_rgb", RESULTS / "eval"]


@functools.lru_cache(maxsize=None)
def _outer_roots(root: str) -> tuple:
    """Alberi che contengono il codice senza esserne la radice: l'albero di lavoro git che contiene ROOT (git rev-parse
    --show-toplevel, se diverso da ROOT) e ogni antenato X di ROOT con ROOT dentro X/research (disposizione del
    repository del gruppo, anche in una copia senza git). Lì stanno i pacchetti accettati: nessuna uscita è ammessa."""
    root = Path(root).resolve()
    out = []
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root, capture_output=True, text=True, timeout=30)
        top = Path(r.stdout.strip()).resolve() if r.returncode == 0 and r.stdout.strip() else None
        if top is not None and not pg.same(top, root):
            out.append(top)
    except (OSError, subprocess.SubprocessError):
        pass
    for x in root.parents:
        if (x / "research").is_dir() and any(pg.same(x / "research", y) for y in root.parents) and not any(pg.same(x, o) for o in out):
            out.append(x)
    return tuple(out)


def outer_roots() -> tuple:
    return _outer_roots(str(ROOT))


def run_dir_problem(run) -> str:
    """Motivo per cui `run` non può essere la cartella del run ("" se può): nome, collegamento simbolico, alberi esterni
    con i pacchetti accettati (outer_roots), path_guard (radice, home, repository, cartelle protette; dentro il repository
    solo results/c2/), cartelle dei pesi, dei dati e dei risultati consegnati anche se spostate con le variabili
    d'ambiente (in entrambe le direzioni)."""
    run = Path(run)
    if not RUN_ID_RE.match(run.name):
        return f"run folder name {run.name!r} is not YYYYMMDD-C2-seed42-<shortcommit>[-trial]"
    if run.is_symlink():
        return f"{run.name} is a symbolic link: the run folder must be a real folder"
    for g in outer_roots():
        if pg.within(run, g) or pg.contains(run, g):
            return f"{run} is inside, or contains, the work tree {g.name} with the accepted packages: outputs go outside it"
    msg = pg.deletion_problem(run, ROOT, pg.BASE_PROTECTED, [ROOT / "results" / "c2"])
    if msg:
        return msg
    for q in protected_extra():
        if pg.within(run, q) or (Path(q).exists() and pg.contains(run, q)):
            return f"{run} is, contains or is inside a protected folder ({q.name})"
    return ""


def input_paths(a, caches: bool) -> dict:
    out = {"--config": a.config, "--source-manifests": a.source_manifests, "--target-manifests": a.target_manifests}
    if caches:
        out |= {"--source-cache": a.source_cache, "--target-cache": a.target_cache}
    return out


def inputs_problem(a, run, caches: bool) -> str:
    """Ingressi del passo controllati prima di qualunque modifica: esistono, sono del tipo giusto e sono separati dalla
    cartella del run per identità dei file (né dentro, né uguali, né antenati; anche tramite collegamenti o alias)."""
    run = Path(run)
    for opt, p in input_paths(a, caches).items():
        p = Path(p)
        if pg.within(p, run) or pg.contains(p, run):
            return f"{opt} {p.name} is, contains or is inside the run folder: inputs and outputs must be separate"
        if opt == "--config" and not p.is_file():
            return f"configuration not found: {p.name}"
        if opt.endswith("manifests") and not p.is_dir():
            return f"{opt} {p.name} is not a folder"
        if opt.endswith("cache") and not p.is_file():
            return f"{opt.split('-')[2]} cache {p.name} not found"
    return ""


def is_trial(run) -> bool:
    return Path(run).name.endswith("-trial")


def repo_state(run) -> dict:
    fn = REPO_STATE or rm.repo_state_excluding
    return fn(run)


def code_hashes() -> dict:
    """SHA-256 del codice che decide gli esiti, registrati al check e ricontrollati a ogni passo; le architetture del
    codice di riferimento con il loro SHA-256, o "absent" se il file non c'è (pacchetto pubblico, test)."""
    out = {f: sha256_file(ROOT / f) for f in CODE_FILES}
    arch = Path(LIVEDETECTION) / "src" / "architectures.py"
    out[ARCH_KEY] = sha256_file(arch) if arch.is_file() else "absent"
    return out


def remove_outputs(run, step: str) -> list:
    """Cancella le uscite del passo e dei passi successivi (e i temporanei), con path_guard per le cartelle."""
    run = Path(run)
    removed = []
    RUN_STATE["step"], RUN_STATE["removed"] = step, True
    entries = sorted(os.listdir(run))
    for s in STEPS[STEPS.index(step):]:
        for n in OUTPUTS[s]:
            temps = [e for e in entries if e == n + ".tmp" or (e.startswith(f".{n}.") and e.endswith(".tmp"))]
            for q in [run / n] + [run / e for e in temps]:
                if q.is_symlink():              # il collegamento si toglie, il suo bersaglio mai
                    q.unlink(); removed.append(q.name)
                elif q.is_dir():
                    fail(f"{q.name} in the run folder is a folder, not an output file: refused")
                elif q.exists():
                    q.unlink(); removed.append(q.name)
        if s == "train":
            dirs = [CHECKPOINTS, CHECKPOINTS + ".tmp"] + [e for e in entries if e.startswith(f".{CHECKPOINTS}.")]
            for d in (run / e for e in dirs):
                if d.is_symlink():
                    d.unlink(); removed.append(d.name)
                elif d.is_dir():
                    pg.safe_rmtree(d, ROOT, pg.BASE_PROTECTED, [ROOT / "results" / "c2"], ignore_errors=False)
                    removed.append(d.name)
                elif d.exists():
                    d.unlink(); removed.append(d.name)
    return removed


_UMASK = os.umask(0o022)
os.umask(_UMASK)


def write_all(run, files: dict) -> None:
    """Scrive un gruppo di file (nome -> testo) solo alla fine: prima tutti i temporanei, creati in modo sicuro
    (tempfile.mkstemp: nome nuovo, O_CREAT | O_EXCL, nessun collegamento seguito), poi le rinomine; un'uscita che è un
    collegamento o non è un file regolare è rifiutata prima di scrivere. Su qualunque eccezione non resta nessun file
    del gruppo né alcun temporaneo, e nessun file esterno è toccato."""
    run = Path(run)
    tmps, done = {}, []
    try:
        for n in files:
            q = run / n
            if q.is_symlink() or (q.exists() and not q.is_file()):
                raise C2Error(f"{n} in the run folder is a link or not a regular file: refused")
        for n, text in files.items():
            fd, tp = tempfile.mkstemp(prefix=f".{n}.", suffix=".tmp", dir=run)
            tmps[n] = Path(tp)
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            os.chmod(tp, 0o666 & ~_UMASK)
        for n, tp in tmps.items():
            os.replace(tp, run / n)
            done.append(n)
    except BaseException:
        for tp in tmps.values():
            tp.unlink(missing_ok=True)
        for n in done:
            (run / n).unlink(missing_ok=True)
        raise


def record_base(ctx, step: str) -> dict:
    """Provenienza di un record: run, passo, configurazione, check.json di questo run (SHA-256) e commit."""
    return {"passed": True, "step": step, "run_id": ctx.run.name, "config_sha256": ctx.cfg_sha,
            "check_sha256": sha256_file(ctx.run / "check.json"), "commit": ctx.check.get("commit")}


def load_record(run, name: str, ctx=None, step: str = None) -> dict:
    """Record di un passo precedente: file regolare, JSON valido, passo superato e, con ctx, scritto da questo run
    (run_id, passo, configurazione e SHA-256 del check.json di questo run): un record copiato da un altro run, anche
    con la stessa configurazione, è rifiutato."""
    p = Path(run) / name
    if p.is_symlink() or not p.is_file():
        fail(f"missing {name} in the run folder: run the previous step first")
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, ValueError):
        fail(f"{name} is not valid JSON")
    if not isinstance(d, dict) or d.get("passed") is not True:
        fail(f"{name} does not record a passed step")
    if ctx is not None:
        if d.get("run_id") != ctx.run.name:
            fail(f"{name} belongs to another run ({d.get('run_id')!r}): records and checkpoints of another run are refused")
        if d.get("step") != step:
            fail(f"{name} records the step {d.get('step')!r}, expected {step!r}")
        if d.get("config_sha256") != ctx.cfg_sha:
            fail(f"{name} was written with another configuration")
        if d.get("check_sha256") != sha256_file(ctx.run / "check.json"):
            fail(f"{name} was not written after the check of this run (check.json SHA-256 differs)")
        if d.get("commit") != ctx.check.get("commit"):
            fail(f"{name} was written at another commit than the check of this run")
    return d


class Ctx:
    def __init__(self, run, cfg, cfg_sha, check, trial):
        self.run, self.cfg, self.cfg_sha, self.check, self.trial = run, cfg, cfg_sha, check, trial


def open_run(a, step: str, caches: bool) -> Ctx:
    """Cartella del run e configurazione di un passo dopo check: prima, senza scrivere nulla, cartella e ingressi
    (inputs_problem); poi stessa configurazione (SHA-256), stesso codice e, per un run non -trial, stesso commit del
    check con albero pulito."""
    run = Path(a.run_dir)
    RUN_STATE.update(step=step, removed=False)
    if msg := run_dir_problem(run):
        fail(f"run folder refused: {msg}")
    if not run.is_dir():
        fail("run folder does not exist or is not a folder: run check first")
    if msg := inputs_problem(a, run, caches):
        fail(f"inputs refused: {msg}")
    accept_run(run, step)
    cfg, cfg_sha = load_config(a.config)
    check = load_record(run, "check.json")
    if check.get("config_sha256") != cfg_sha:
        fail(f"configuration SHA-256 {cfg_sha[:12]}… differs from the one recorded at check ({str(check.get('config_sha256'))[:12]}…): refused")
    if check.get("code_sha256") != code_hashes():
        fail("the code changed since check (SHA-256 of the scripts differ from check.json): run check again")
    if check.get("run_id") != run.name:
        fail("check.json belongs to another run folder")
    trial = is_trial(run)
    if not trial:
        state = repo_state(run)
        if state.get("dirty") or state.get("commit") != check.get("commit"):
            fail("non-trial run: the repository commit or state differs from check (clean tree and same commit required)")
    return Ctx(run, cfg, cfg_sha, check, trial)


# ============================================================================ ingressi: manifest e cache

class Inputs:
    pass


def counts_of(ds: str, rows) -> dict:
    lab = Counter(r["label"] for r in rows)
    subj = "subject_key" if ds == "casia_fasd" else "subject"
    out = {"rows": len(rows), "n_bona_fide": lab["0"], "n_attack": lab["1"], "n_subjects": len({r[subj] for r in rows}),
           "n_subjects_bona_fide": len({r[subj] for r in rows if r["label"] == "0"}),
           "n_subjects_attack": len({r[subj] for r in rows if r["label"] == "1"})}
    if ds == "casia_fasd":
        out["n_videos"] = len({r["video_key"] for r in rows})
    return out


def per_class(cache_file: str) -> int:
    return int(re.search(r"_([1-9]\d*)\.npz$", cache_file).group(1))


def load_inputs(cfg: dict, a, caches: bool) -> Inputs:
    """Manifest (SHA-256 contro la configurazione, validazione stretta, conteggi, disgiunzioni) e, con caches, le cache
    dei frame (SHA-256, membri, collegamento con i manifest). Nessun frame è hashato qui (solo in check)."""
    inp = Inputs()
    inp.rows, inp.sha = {}, {}
    dirs = {"casia_fasd": Path(a.source_manifests), "nuaa": Path(a.target_manifests)}
    for role, (ds, _) in ROLES.items():
        e = cfg["manifests"][role]
        p = dirs[ds] / e["file"]
        if not p.is_file():
            fail(f"manifest {e['file']} ({role}) not found")
        got = sha256_file(p)
        if got != e["sha256"]:
            fail(f"manifest {e['file']} ({role}): SHA-256 {got[:12]}… differs from the configuration ({e['sha256'][:12]}…)")
        rows = guarded(e["file"], rm.load_manifest, "casia_fasd", p) if ds == "casia_fasd" else guarded(e["file"], nm.load_manifest, p)
        c = counts_of(ds, rows)
        for f in ("rows", "n_bona_fide", "n_attack", "n_subjects") + (("n_videos",) if ds == "casia_fasd" else ()):
            if c[f] != e[f]:
                fail(f"manifest {e['file']} ({role}): {f} {c[f]} differs from the configuration ({e[f]})")
        if len({r['key'] for r in rows}) != len(rows):
            fail(f"manifest {e['file']} ({role}): duplicated keys")
        inp.rows[role], inp.sha[role] = rows, got
    separation(inp)
    inp.by_key = {role: {r["key"]: r for r in rows} for role, rows in inp.rows.items()}
    if caches:
        link_caches(cfg, a, inp)
    return inp


def separation(inp: Inputs) -> None:
    """Disgiunzioni: fitting, early stopping e test sorgente per chiave, SHA-256, soggetto e video; lato dello split del
    mirror (train_img per fitting ed early stopping, test_img per il test); nessun SHA-256 di file CASIA uguale a NUAA."""
    want_split = {"source_train": "train", "source_dev": "train", "source_test": "test"}
    for role, split in want_split.items():
        wrong = sum(1 for r in inp.rows[role] if r["mirror_split"] != split)
        if wrong:
            fail(f"{role}: {wrong} rows outside the mirror split {split}_img")
    roles = list(SOURCE_ROLES)
    for i, ra in enumerate(roles):
        for rb in roles[i + 1:]:
            for dim in ("key", "sha256", "subject_key", "video_key"):
                shared = {r[dim] for r in inp.rows[ra]} & {r[dim] for r in inp.rows[rb]}
                if shared:
                    fail(f"split overlap: {ra} and {rb} share {len(shared)} values of {dim}")
    casia = {r["sha256"] for role in roles for r in inp.rows[role]}
    nuaa = {r["sha256"] for r in inp.rows["target_test"]}
    if casia & nuaa:
        fail(f"cross-dataset overlap: {len(casia & nuaa)} CASIA-FASD files have the SHA-256 of a NUAA file")


def link_caches(cfg: dict, a, inp: Inputs) -> None:
    """Cache dei frame: SHA-256 contro la configurazione, membri, tipo e forma, e collegamento con i manifest per ordine
    di campionamento (spread per classe, bona fide e poi attacchi) e array y/s; nessun file name nella cache."""
    df = cfg["derived_frames"]
    paths = {"source": Path(a.source_cache), "target": Path(a.target_cache)}
    inp.cache, inp.cache_sha, inp.order = {}, {}, {}
    for side in ("source", "target"):
        p = paths[side]
        if p.name != df[side]["file"]:
            fail(f"{side} cache {p.name} is not the file of the configuration ({df[side]['file']})")
        if not p.is_file():
            fail(f"{side} cache {p.name} not found")
        got = sha256_file(p)
        if got != df[side]["sha256"]:
            fail(f"{side} cache {p.name}: SHA-256 {got[:12]}… differs from the configuration ({df[side]['sha256'][:12]}…)")
        try:
            z = np.load(p, allow_pickle=False)
            members = set(z.files)
        except (OSError, ValueError) as e:
            fail(f"{side} cache {p.name} unreadable ({e})")
        need = {"X_train", "y_train", "s_train", "X_test", "y_test", "s_test"} if side == "source" else {"X_test", "y_test", "s_test"}
        if not need <= members:
            fail(f"{side} cache {p.name}: missing members {sorted(need - members)}")
        inp.cache[side], inp.cache_sha[side] = z, got
    k_src, k_tgt = per_class(df["source"]["file"]), per_class(df["target"]["file"])
    train_rows = inp.rows["source_train"] + inp.rows["source_dev"]
    links = {("source", "train"): ("casia_fasd", rm.spread_keys(train_rows, k_src), {r["key"]: r for r in train_rows}),
             ("source", "test"): ("casia_fasd", rm.spread_keys(inp.rows["source_test"], k_src), inp.by_key["source_test"]),
             ("target", "test"): ("nuaa", nm.spread_keys(inp.rows["target_test"], k_tgt), inp.by_key["target_test"])}
    want_n = {("source", "train"): df["source"]["X_train"], ("source", "test"): df["source"]["X_test"], ("target", "test"): df["target"]["X_test"]}
    shapes = set()
    for (side, split), (ds, keys, rows) in links.items():
        z = inp.cache[side]
        if len(keys) != len(rows):
            fail(f"{side} cache X_{split}: {len(rows) - len(keys)} manifest rows are not in the sampling order (per-class limit)")
        y, s = z[f"y_{split}"], z[f"s_{split}"]
        exp_y = np.array([int(rows[k]["label"]) for k in keys], np.int64)
        exp_s = np.array([rows[k]["subject"] for k in keys])        # CASIA: numero locale del soggetto; NUAA: soggetto
        if len(y) != want_n[(side, split)] or len(keys) != want_n[(side, split)]:
            fail(f"{side} cache X_{split}: {len(y)} frames, configuration {want_n[(side, split)]}, manifests {len(keys)}")
        if y.dtype.kind not in "iu" or not np.array_equal(y, exp_y) or s.dtype.kind != "U" or not np.array_equal(s, exp_s):
            fail(f"{side} cache X_{split}: label or group arrays differ from the manifests in sampling order (linkage refused)")
        X = z[f"X_{split}"]     # lettura dell'array: forma e tipo
        if X.dtype != np.uint8 or X.ndim != 4 or X.shape[0] != len(keys) or X.shape[3] != 3:
            fail(f"{side} cache X_{split}: dtype {X.dtype}, shape {X.shape} (uint8 frames N x H x W x 3 expected)")
        shapes.add(X.shape[1:])
        inp.order[(side, split)] = keys
    if len(shapes) != 1:
        fail(f"frame shapes differ between source and target caches: {sorted(shapes)}")
    inp.index = {"train": {k: i for i, k in enumerate(inp.order[("source", "train")])},
                 "source_test": {k: i for i, k in enumerate(inp.order[("source", "test")])},
                 "target_test": {k: i for i, k in enumerate(inp.order[("target", "test")])}}


def frame_separation(inp: Inputs) -> dict:
    """Frame derivati (hash dei pixel come salvati): fitting, early stopping e test sorgente disgiunti, e nessun frame
    CASIA uguale a un frame del test NUAA."""
    src = inp.cache["source"]
    h_train = nm.frame_hashes(src["X_train"])
    h_test = nm.frame_hashes(src["X_test"])
    h_tgt = nm.frame_hashes(inp.cache["target"]["X_test"])
    idx = inp.index["train"]
    fit = {h_train[idx[r["key"]]] for r in inp.rows["source_train"]}
    dev = {h_train[idx[r["key"]]] for r in inp.rows["source_dev"]}
    test = set(h_test)
    out = {"fit_dev_shared": len(fit & dev), "train_test_shared": len((fit | dev) & test), "casia_nuaa_shared": len((fit | dev | test) & set(h_tgt)),
           "internal_duplicates": {"X_train": len(h_train) - len(set(h_train)), "X_test": len(h_test) - len(test),
                                   "target_X_test": len(h_tgt) - len(set(h_tgt))}}
    for k, what in (("fit_dev_shared", "fitting and early-stopping"), ("train_test_shared", "training and source-test"),
                    ("casia_nuaa_shared", "CASIA-FASD and NUAA")):
        if out[k]:
            fail(f"derived frames: {out[k]} frames shared between {what} (exact pixel hash)")
    return out


# ============================================================================ check

def cmd_check(a) -> None:
    run = Path(a.run_dir)
    if msg := run_dir_problem(run):
        fail(f"run folder refused: {msg}")
    trial = is_trial(run)
    if run.exists() and not run.is_dir():
        fail("run folder path exists and is not a folder")
    if msg := inputs_problem(a, run, caches=True):
        fail(f"inputs refused: {msg}")
    if run.is_dir() and not trial and any(not (e.name == STATUS and e.is_file() and not e.is_symlink()) for e in run.iterdir()):
        fail("a non-trial run folder must be new (missing, empty or holding only the regular file status.json): refused, nothing removed")
    run.mkdir(parents=True, exist_ok=True)
    accept_run(run, "check")
    cfg, cfg_sha = load_config(a.config)
    state = repo_state(run)
    m = RUN_ID_RE.match(run.name)
    if not trial:
        if cfg_sha != APPROVED_CONFIG_SHA256:
            fail(f"non-trial run: configuration SHA-256 {cfg_sha[:12]}… is not the approved (amended) one ({APPROVED_CONFIG_SHA256[:12]}…)")
        if state.get("dirty") or not str(state.get("commit", "")).startswith(m.group(2)):
            fail("non-trial run: the working tree must be clean and the run id must carry the current commit")
    remove_outputs(run, "check")
    inp = load_inputs(cfg, a, caches=True)
    frames = frame_separation(inp)
    roles = {}
    for role, (ds, name) in ROLES.items():
        e = cfg["manifests"][role]
        roles[role] = {"name": e["name"], "file": e["file"], "sha256": inp.sha[role], **counts_of(ds, inp.rows[role])}
    doc = {"passed": True, "step": "check", "run_id": run.name, "trial": trial, "approved_config": cfg_sha == APPROVED_CONFIG_SHA256,
           "config_sha256": cfg_sha, "manifests": roles,
           "caches": {side: {"file": cfg["derived_frames"][side]["file"], "sha256": inp.cache_sha[side]} for side in ("source", "target")},
           "derived_frames": frames, "detector": "not used", "weights": "not loaded",
           "code_sha256": code_hashes(), **state, "created": now()}
    write_all(run, {"check.json": json.dumps(doc, indent=1) + "\n"})
    print(f"== check: 4 manifests, 2 caches and the splits verified; {'trial' if trial else 'non-trial'} run {run.name}")


# ============================================================================ train e controllo dei pesi

def load_trainer():
    if TRAINER is not None:
        return TRAINER
    try:
        import train_cnn
    except ModuleNotFoundError as e:
        fail(f"missing Python package {e.name!r}: train needs scripts/train_cnn.py and its dependencies (TensorFlow, OpenCV, scikit-learn)")
    return train_cnn.train


def all_listed_hashes(cfg: dict) -> dict:
    out = {}
    for blk in ("excluded_pooled", "alternative_not_recommended"):
        for k, v in cfg["checkpoints"][blk].items():
            if k.endswith(".h5"):
                out[v] = f"{blk} {k}"
    return out


def device_problems(mid: str, arch: str, card: dict, entry: dict) -> list:
    """Dispositivo del training registrato nella scheda e in train.json: richiesto secondo TRAINING_DEVICE; dispositivi
    visibili e fisici registrati (liste ordinate, senza ripetizioni, di CPU e GPU, con la CPU; i visibili fra i fisici) e
    nessuna GPU presente ma nascosta; atteso ed effettivo uguali al dispositivo della politica (CPU:0 per la politica CPU;
    per "default" GPU:0 se fra i visibili c'era una GPU, altrimenti CPU:0); scarto grafo/eager registrato (numero finito,
    non negativo) ed entro GRAPH_EAGER_TOL per la politica CPU, con lo strato (nome) solo se lo scarto non è 0; esattamente
    i campi DEVICE_KEYS; stessa registrazione nei due file."""
    d = card.get("training_device")
    if not isinstance(d, dict):
        return [f"{mid}: the card does not record the training device"]
    bad = []
    want = TRAINING_DEVICE[arch]
    vis, phys, gap = d.get("visible"), d.get("physical"), d.get("graph_eager_max_rel_diff")
    devs_ok = lambda v: isinstance(v, list) and "CPU" in v and v == sorted(set(v)) and set(v) <= {"CPU", "GPU"}   # noqa: E731
    if sorted(d) != sorted(DEVICE_KEYS):
        bad.append(f"{mid}: training device fields {sorted(d)}, expected {sorted(DEVICE_KEYS)}")
    if d.get("requested") != want:
        bad.append(f"{mid}: training device requested {d.get('requested')!r}, expected {want!r}")
    elif not devs_ok(vis) or not devs_ok(phys):
        bad.append(f"{mid}: visible devices {vis!r} or physical devices {phys!r} not recorded correctly")
    elif not set(vis) <= set(phys):
        bad.append(f"{mid}: visible devices {vis!r} not among the physical devices {phys!r}")
    elif "GPU" in phys and "GPU" not in vis:
        bad.append(f"{mid}: a GPU was present but hidden from the training process")
    else:
        exp = "CPU:0" if want == "CPU" or "GPU" not in vis else "GPU:0"
        for k in ("expected", "effective"):
            if d.get(k) != exp:
                bad.append(f"{mid}: training device {k} {d.get(k)!r}, expected {exp!r}"
                           + (" (CPU execution not guaranteed)" if want == "CPU" else ""))
    if not isinstance(gap, float) or not math.isfinite(gap) or gap < 0:
        bad.append(f"{mid}: graph/eager difference {gap!r} not recorded correctly")
    elif want == "CPU" and gap > GRAPH_EAGER_TOL:
        bad.append(f"{mid}: graph-mode outputs differed from eager by {gap!r} (relative), above {GRAPH_EAGER_TOL} "
                   "(CPU execution not guaranteed)")
    layer = d.get("graph_eager_layer")
    if isinstance(gap, float) and (layer is None) != (gap == 0) or layer is not None and not (isinstance(layer, str) and layer):
        bad.append(f"{mid}: graph/eager layer {layer!r} inconsistent with the difference {gap!r}")
    if entry.get("training_device") != d:
        bad.append(f"{mid}: train.json and card disagree on the training device")
    return bad


def weights_problems(ckdir, cfg: dict, rec: dict, inp_counts: dict) -> list:
    """Pesi nuovi prima di qualunque punteggio: SHA-256 del file contro la scheda e train.json, campi della scheda
    (architettura, dataset, protocollo, seed, conteggi, split per soggetto, nessuna valutazione del test, controllo dello
    split, iperparametri), epoca ripristinata = epoca di val_loss minimo, pesi esistenti o pooled rifiutati,
    dispositivo del training (device_problems)."""
    bad = []
    ckdir = Path(ckdir)
    listed = all_listed_hashes(cfg)
    b = cfg["budget"]
    models = rec.get("models") or {}
    if sorted(models) != sorted(i for i, _ in model_list(cfg)):
        return [f"train.json lists models {sorted(models)}, configuration {sorted(i for i, _ in model_list(cfg))}"]
    seen = set()
    for mid, arch in model_list(cfg):
        e = models[mid]
        stem = f"{arch}__casia_fasd"
        card_p, h5, ker = ckdir / f"{stem}.json", ckdir / f"{stem}.h5", ckdir / f"{stem}.keras"
        if e.get("weights_file") != h5.name or e.get("card_file") != card_p.name or e.get("keras_file") != ker.name:
            bad.append(f"{mid}: file names in train.json differ from {stem}.*"); continue
        if not (card_p.is_file() and h5.is_file() and ker.is_file()):
            bad.append(f"{mid}: missing weights, Keras file or card in {CHECKPOINTS}/"); continue
        if sha256_file(card_p) != e.get("card_sha256"):
            bad.append(f"{mid}: card SHA-256 differs from train.json (card changed)")
        try:
            card = json.loads(card_p.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, ValueError):
            bad.append(f"{mid}: card is not valid JSON"); continue
        w = sha256_file(h5)
        if w != card.get("weights_sha256") or w != e.get("weights_sha256"):
            bad.append(f"{mid}: weights SHA-256 differs from the card or from train.json (checkpoint changed)")
        if sha256_file(ker) != card.get("keras_sha256") or card.get("keras_sha256") != e.get("keras_sha256"):
            bad.append(f"{mid}: Keras file SHA-256 differs from the card or from train.json")
        if w in listed:
            bad.append(f"{mid}: weights are an existing checkpoint refused by the configuration ({listed[w]})")
        if w in seen:
            bad.append(f"{mid}: the two models have the same weights")
        seen.add(w)
        expect = {"arch": arch, "dataset": "casia_fasd", "protocol": "official", "seed": SEED, "n_train": inp_counts["source_train"],
                  "n_val": inp_counts["source_dev"], "val_split": "per soggetto", "lr": b["learning_rate"], "batch": b["batch"],
                  "test_evaluation": "disabled"}
        for k, v in expect.items():
            if card.get(k) != v:
                bad.append(f"{mid}: card field {k} = {card.get(k)!r}, expected {v!r}")
        if "test" in card or "n_test" in card:
            bad.append(f"{mid}: the card holds a test evaluation (test evaluation must be disabled)")
        if not str(card.get("split_check", "")).startswith("passed"):
            bad.append(f"{mid}: the card does not record a passed split check")
        hist = card.get("val_loss_history")
        n_ep = card.get("epochs_run")
        if not _int(n_ep) or not 1 <= n_ep <= b["max_epochs"] or not isinstance(hist, list) or len(hist) != n_ep \
                or not all(isinstance(v, float) and math.isfinite(v) for v in hist):
            bad.append(f"{mid}: epochs_run {n_ep!r} or val_loss history not consistent with the budget ({b['max_epochs']} epochs)")
        else:
            best = int(np.argmin(hist)) + 1
            if card.get("restored_epoch") != best or card.get("best_epoch") != best or card.get("best_val_loss") != hist[best - 1]:
                bad.append(f"{mid}: restored epoch {card.get('restored_epoch')!r} is not the epoch of minimum val_loss ({best})")
        if e.get("restored_epoch") != card.get("restored_epoch") or e.get("epochs_run") != n_ep:
            bad.append(f"{mid}: train.json and card disagree on epochs")
        bad += device_problems(mid, arch, card, e)
    return bad


def cmd_train(a) -> None:
    ctx = open_run(a, "train", caches=True)
    if not a.i_am_authorized:
        fail("train needs --i-am-authorized (new training on the restricted data)")
    b = ctx.cfg["budget"]
    if a.epochs != b["max_epochs"]:
        fail(f"--epochs {a.epochs} differs from the configuration budget ({b['max_epochs']})")
    trainer = load_trainer()
    remove_outputs(ctx.run, "train")
    inp = load_inputs(ctx.cfg, a, caches=True)
    z = inp.cache["source"]
    order = inp.order[("source", "train")]
    rows = {r["key"]: r for r in inp.rows["source_train"] + inp.rows["source_dev"]}
    split_check = {"sha256": [rows[k]["sha256"] for k in order], "train": [r["sha256"] for r in inp.rows["source_train"]],
                   "val": [r["sha256"] for r in inp.rows["source_dev"]]}
    data = {"X_train": z["X_train"], "y_train": z["y_train"], "s_train": z["s_train"]}   # nessun X_test: nessuna valutazione del test
    final = ctx.run / CHECKPOINTS
    tmp = Path(tempfile.mkdtemp(prefix=f".{CHECKPOINTS}.", dir=ctx.run))      # cartella temporanea nuova, mai un nome fisso
    try:
        rec = {**record_base(ctx, "train"), "epochs_argument": a.epochs,
               "batch": b["batch"], "learning_rate": b["learning_rate"], "test_evaluation": "disabled", "models": {}}
        t_all = time.time()
        for mid, arch in model_list(ctx.cfg):
            try:
                info = trainer(arch, "casia_fasd", data, a.epochs, b["batch"], b["learning_rate"], "official",
                               out_dir=tmp, split_check=split_check, test_eval=False)
            except SystemExit as e:
                fail(f"{mid}: training stopped: {str(e.code).removeprefix('ERROR: ')}")
            stem = f"{arch}__casia_fasd"
            rec["models"][mid] = {"arch": arch, "weights_file": f"{stem}.h5", "card_file": f"{stem}.json", "keras_file": f"{stem}.keras",
                                  "weights_sha256": sha256_file(tmp / f"{stem}.h5"), "card_sha256": sha256_file(tmp / f"{stem}.json"),
                                  "keras_sha256": sha256_file(tmp / f"{stem}.keras"),
                                  "epochs_run": info.get("epochs_run"), "restored_epoch": info.get("restored_epoch"),
                                  "train_seconds": info.get("train_seconds"), "training_device": info.get("training_device")}
        rec["wall_seconds"] = round(time.time() - t_all)
        if rec["wall_seconds"] > 60 * b["wall_time_cap_minutes"]:
            fail(f"wall time {rec['wall_seconds']} s above the cap of {b['wall_time_cap_minutes']} minutes: failed run, no results")
        problems = weights_problems(tmp, ctx.cfg, rec, {r: len(inp.rows[r]) for r in ("source_train", "source_dev")})
        if problems:
            fail("weights check after training failed: " + "; ".join(problems))
        os.replace(tmp, final)
        rec["created"] = now()
        write_all(ctx.run, {"train.json": json.dumps(rec, indent=1) + "\n"})
    except BaseException:
        for d in (tmp, final):
            if d.is_dir() and not d.is_symlink() and not (ctx.run / "train.json").exists():
                pg.safe_rmtree(d, ROOT, pg.BASE_PROTECTED, [ROOT / "results" / "c2"], ignore_errors=False)
        raise
    print(f"== train: {len(rec['models'])} models trained, weights and cards in {CHECKPOINTS}/ checked")


def checked_weights(ctx: Ctx, inp: Inputs) -> dict:
    rec = load_record(ctx.run, "train.json", ctx, "train")
    if (ctx.run / CHECKPOINTS).is_symlink():
        fail(f"{CHECKPOINTS}/ is a symbolic link: the weights of this run must be in its own folder")
    problems = weights_problems(ctx.run / CHECKPOINTS, ctx.cfg, rec, {r: len(inp.rows[r]) for r in ("source_train", "source_dev")})
    if problems:
        fail("weights check failed: " + "; ".join(problems))
    return rec


# ============================================================================ punteggi grezzi

def tf_scorer(arch: str, weights_path, weights_sha: str):
    """Scorer vero: architettura del codice di riferimento, pesi verificati per SHA-256, CPU, BatchNorm in inferenza."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        import tensorflow as tf
    except ModuleNotFoundError as e:
        fail(f"missing Python package {e.name!r}: score needs TensorFlow")
    try:
        tf.config.set_visible_devices([], "GPU")     # CPU anche con tensorflow-metal; già inizializzato: resta tf.device
    except (RuntimeError, ValueError):
        pass
    tf.config.optimizer.set_experimental_options({"remapping": False})
    if sha256_file(weights_path) != weights_sha:
        fail(f"{Path(weights_path).name}: SHA-256 changed just before loading")
    arch_mod = load_module("kuznetsov_architectures", LIVEDETECTION / "src" / "architectures.py")
    with tf.device("/CPU:0"):
        model = arch_mod.create_model(arch).get_model()
        model.load_weights(str(weights_path))

    def score(X):
        out = []
        for i in range(0, len(X), SCORE_BATCH):
            xb = X[i:i + SCORE_BATCH].astype(np.float32) / 255.0
            with tf.device("/CPU:0"):
                p = np.asarray(model(xb, training=False))
            out.append(p[:, 1])
        return np.concatenate(out) if out else np.zeros(0)
    return score


def expected_fields(role: str, row: dict) -> dict:
    """Campi di un record di punteggio ricavati dal manifest (tutti tranne score, soglia, decisione e hash)."""
    ds = ROLES[role][0]
    att = row["label"] == "1"
    if ds == "casia_fasd":
        return {"sample_id": row["key"], "dataset_id": ds, "split": RAW_SPLIT[role], "subject_id": row["subject_key"], "session_id": NULL,
                "video_id": row["video_key"], "attack_type": CASIA_ATTACK_TYPE[row["video"]] if att else NULL, "modality": "RGB",
                "label": row["label"], "file_sha256": row["sha256"]}
    return {"sample_id": row["key"], "dataset_id": ds, "split": RAW_SPLIT[role], "subject_id": row["subject"], "session_id": row["session"],
            "video_id": NULL, "attack_type": "print" if att else NULL, "modality": "RGB", "label": row["label"], "file_sha256": row["sha256"]}


def csv_text(rows) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=SCORE_COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def score_role(ctx: Ctx, inp: Inputs, role: str, scorers: dict, train_rec: dict) -> list:
    if role == "source_dev":
        X, idx = inp.cache["source"]["X_train"], inp.index["train"]
    elif role == "source_test":
        X, idx = inp.cache["source"]["X_test"], inp.index["source_test"]
    else:
        X, idx = inp.cache["target"]["X_test"], inp.index["target_test"]
    keys = sorted(inp.by_key[role])
    sel = np.array([idx[k] for k in keys], dtype=np.int64)
    out = []
    for mid, _ in model_list(ctx.cfg):
        s = np.asarray(scorers[mid](X[sel]), dtype=np.float64)
        if s.shape != (len(keys),) or not np.all(np.isfinite(s)) or np.any(s < 0) or np.any(s > 1):
            fail(f"{mid} on {role}: the scorer returned {s.shape} values, non-finite or outside [0, 1]")
        for k, v in zip(keys, s):
            out.append({"run_id": ctx.run.name, **expected_fields(role, inp.by_key[role][k]), "score_attack": repr(float(v)),
                        "threshold": NULL, "prediction": NULL, "checkpoint_sha256": train_rec["models"][mid]["weights_sha256"],
                        "config_sha256": ctx.cfg_sha, "model_id": mid})
    return [{c: r[c] for c in SCORE_COLUMNS} for r in out]


def make_scorers(ctx: Ctx, train_rec: dict) -> dict:
    make = MAKE_SCORER or tf_scorer
    return {mid: make(arch, ctx.run / CHECKPOINTS / train_rec["models"][mid]["weights_file"], train_rec["models"][mid]["weights_sha256"])
            for mid, arch in model_list(ctx.cfg)}


def parse_sets(text: str) -> str:
    sets = sorted(x.strip() for x in str(text).split(","))
    if sets == ["dev"]:
        return "score_dev"
    if sets == ["source_test", "target"]:
        return "score_tests"
    fail(f"--set {text!r}: use 'dev' or 'source_test,target'")


def cmd_score(a) -> None:
    step = parse_sets(a.set)
    ctx = open_run(a, step, caches=True)
    if not a.i_am_authorized:
        fail("score needs --i-am-authorized (inference on the restricted data)")
    thr_p = ctx.run / "threshold.json"
    if step == "score_tests" and (thr_p.is_symlink() or not thr_p.is_file()):
        fail("source_test and target scoring refused: threshold.json is missing (run score --set dev and select first)")
    remove_outputs(ctx.run, step)
    inp = load_inputs(ctx.cfg, a, caches=True)
    train_rec = checked_weights(ctx, inp)
    files = {}
    if step == "score_dev":
        rows = score_role(ctx, inp, "source_dev", make_scorers(ctx, train_rec), train_rec)
        text = csv_text(rows)
        rec = {**record_base(ctx, step), "device": "CPU", "train_sha256": sha256_file(ctx.run / "train.json"),
               "files": {RAW["dev"]: {"sha256": sha256_text(text), "rows": len(rows)}}, "created": now()}
        files = {RAW["dev"]: text, "score_dev.json": json.dumps(rec, indent=1) + "\n"}
    else:
        thr_sha = sha256_file(thr_p)                 # registrato prima di qualunque punteggio dei test
        verify_threshold(ctx, inp, train_rec)
        scorers = make_scorers(ctx, train_rec)
        texts = {RAW[r]: csv_text(score_role(ctx, inp, r, scorers, train_rec)) for r in ("source_test", "target_test")}
        if sha256_file(thr_p) != thr_sha:
            fail("threshold.json changed while the test sets were scored")
        rec = {**record_base(ctx, step), "device": "CPU", "train_sha256": sha256_file(ctx.run / "train.json"),
               "score_dev_sha256": sha256_file(ctx.run / "score_dev.json"), "threshold_sha256_before_scoring": thr_sha,
               "files": {n: {"sha256": sha256_text(t), "rows": t.count("\n") - 1} for n, t in texts.items()}, "created": now()}
        files = {**texts, "score_tests.json": json.dumps(rec, indent=1) + "\n"}
    write_all(ctx.run, files)
    print(f"== score ({step}): raw scores written, threshold and prediction null")


# ============================================================================ lettura e validazione dei punteggi

def read_scores(p) -> list:
    p = Path(p)
    if not p.is_file():
        fail(f"missing {p.name}")
    raw = p.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        fail(f"{p.name}: byte-order mark or CR line endings")
    try:
        rd = csv.reader(io.StringIO(raw.decode("utf-8")))
        header = next(rd, [])
        body = list(rd)
    except (UnicodeDecodeError, csv.Error) as e:
        fail(f"{p.name}: unreadable CSV ({e})")
    if header != SCORE_COLUMNS:
        fail(f"{p.name}: header differs from the score record fields")
    if any(len(r) != len(header) for r in body):
        fail(f"{p.name}: rows with a wrong number of fields")
    return [dict(zip(header, r)) for r in body]


def parse_score(text: str):
    """Punteggio di un record: numero finito in [0, 1] scritto con repr (andata e ritorno esatta); None se non valido."""
    try:
        v = float(text)
    except ValueError:
        return None
    if not math.isfinite(v) or not 0.0 <= v <= 1.0 or repr(v) != text:
        return None
    return v


def test_rows_in_selection(rows, inp: Inputs) -> str:
    """Righe del test sorgente o di NUAA fra i punteggi della selezione, per chiave e per SHA-256 (non per sessione: la
    regola del nome di NUAA non si applica ai nomi CASIA)."""
    keys = set(inp.by_key["source_test"]) | set(inp.by_key["target_test"])
    hashes = {r["sha256"] for role in ("source_test", "target_test") for r in inp.rows[role]}
    by_key = sum(1 for r in rows if r.get("sample_id") in keys)
    by_sha = sum(1 for r in rows if r.get("file_sha256") in hashes)
    if by_key or by_sha:
        return (f"selection refused: rows of the source test or of NUAA among the dev scores "
                f"({by_key} by key, {by_sha} by SHA-256)")
    return ""


def score_problems(rows, role: str, ctx: Ctx, inp: Inputs, train_rec: dict, phase: str = "raw", thresholds: dict = None,
                   only: str = None) -> list:
    """Righe di punteggio di un insieme contro il manifest, per ogni modello: completezza, duplicati, chiavi fuori dal
    manifest, ogni campo uguale a quello atteso, punteggio valido, hash di pesi e configurazione; phase raw: soglia e
    decisione null; phase decision: soglia = valore congelato e decisione = 1 se e solo se score > soglia."""
    bad = []
    want = inp.by_key[role]
    mids = [mid for mid, _ in model_list(ctx.cfg) if only is None or mid == only]
    seen = Counter((r.get("model_id"), r.get("sample_id")) for r in rows)
    dup = sum(n - 1 for n in seen.values() if n > 1)
    if dup:
        bad.append(f"{dup} duplicated (model, sample) rows")
    other = sorted({r.get("model_id") for r in rows} - set(mids))
    if other:
        bad.append(f"rows of models outside the configuration: {other}")
    for mid in mids:
        got = {r["sample_id"] for r in rows if r.get("model_id") == mid}
        if got != set(want):
            bad.append(f"{mid}: {len(set(want) - got)} manifest rows without score, {len(got - set(want))} rows outside the manifest")
    wrong, invalid, ckpt, cfgs, phase_bad = 0, 0, 0, 0, 0
    for r in rows:
        mid, k = r.get("model_id"), r.get("sample_id")
        if mid not in mids or k not in want:
            continue
        exp = {"run_id": ctx.run.name, **expected_fields(role, want[k])}
        if any(r.get(c) != v for c, v in exp.items()):
            wrong += 1
        v = parse_score(r.get("score_attack", ""))
        if v is None:
            invalid += 1
        if r.get("checkpoint_sha256") != train_rec["models"][mid]["weights_sha256"]:
            ckpt += 1
        if r.get("config_sha256") != ctx.cfg_sha:
            cfgs += 1
        if phase == "raw":
            phase_bad += (r.get("threshold"), r.get("prediction")) != (NULL, NULL)
        else:
            t = thresholds[mid]
            phase_bad += r.get("threshold") != repr(t) or v is None or r.get("prediction") != str(int(v > t))
    for n, what in ((wrong, "rows with fields different from the manifest"), (invalid, "scores not finite, outside [0, 1] or not round-trip"),
                    (ckpt, "rows with another checkpoint SHA-256"), (cfgs, "rows with another configuration SHA-256"),
                    (phase_bad, "rows with threshold or prediction not null" if phase == "raw" else "rows with a wrong threshold or prediction")):
        if n:
            bad.append(f"{n} {what}")
    return bad


def check_raw(ctx: Ctx, inp: Inputs, train_rec: dict, role: str, record: str) -> list:
    name = RAW["dev"] if role == "source_dev" else RAW[role]
    rec = load_record(ctx.run, record, ctx, record.removesuffix(".json"))
    if rec.get("train_sha256") != sha256_file(ctx.run / "train.json"):
        fail(f"{record} was written with another train.json")
    if record == "score_tests.json" and rec.get("score_dev_sha256") != sha256_file(ctx.run / "score_dev.json"):
        fail("score_tests.json was written with another score_dev.json")
    p = ctx.run / name
    rows = read_scores(p)
    if (rec.get("files") or {}).get(name, {}).get("sha256") != sha256_file(p):
        fail(f"{name}: SHA-256 differs from {record} (scores changed after scoring)")
    if role == "source_dev" and (msg := test_rows_in_selection(rows, inp)):
        fail(msg)
    problems = score_problems(rows, role, ctx, inp, train_rec)
    if problems:
        fail(f"{name} refused: " + "; ".join(problems))
    return rows


def arrays(rows, mid: str):
    sel = sorted((r for r in rows if r["model_id"] == mid), key=lambda r: r["sample_id"])
    return (np.array([int(r["label"]) for r in sel]), np.array([float(r["score_attack"]) for r in sel]),
            [r["subject_id"] for r in sel], [r["attack_type"] for r in sel])


def metrics_api():
    try:
        from tesi_app.evaluation import compute_metrics, metrics_by_attack_type
    except ModuleNotFoundError as e:
        fail(f"missing Python package {e.name!r}: select and apply use src/tesi_app/evaluation.py (numpy, scikit-learn, OpenCV)")
    return compute_metrics, metrics_by_attack_type


# ============================================================================ select

def subjects(y, subj) -> dict:
    return {"n_subjects_bona_fide": len({s for s, l in zip(subj, y) if l == 0}),
            "n_subjects_attack": len({s for s, l in zip(subj, y) if l == 1})}


def selection(ctx: Ctx, rows) -> dict:
    """Soglia di ogni modello sul solo dev, con le funzioni del selettore M1 importate invariate."""
    compute_metrics, _ = metrics_api()
    out = {}
    for mid, arch in model_list(ctx.cfg):
        y, s, subj, _ = arrays(rows, mid)
        if not (np.any(y == 0) and np.any(y == 1)):
            fail(f"{mid}: the dev scores contain a single class: no threshold can be selected (metrics undefined, null)")
        sel = st.select_threshold(y, s, "eer")
        value = float(sel["value"])
        if value not in st.candidate_thresholds(s):
            fail(f"{mid}: selected value is not a candidate threshold")
        op = st.threshold_counts(y, s, value)
        if (op["accepted_attacks"], op["rejected_bona_fide"]) != (sel["fn"], sel["fp"]):
            fail(f"{mid}: counts at the selected value differ from the selection")
        m = compute_metrics(y, s)
        reasons = [r for r, c in (("dev AUC < 0.5", m.get("auc") is not None and m["auc"] < 0.5),
                                  ("operational dev EER > 0.5", op["acer"] is not None and op["acer"] > 0.5)) if c]
        out[mid] = {"arch": arch, "value": value, "decision_rule": DECISION_RULE,
                    "dev": {**op, **subjects(y, subj), "operational_dev_eer": op["acer"], "auc": m.get("auc"),
                            "roc_eer_information_only": m.get("eer")},
                    "flag": f"non-informative ({', '.join(reasons)})" if reasons else ""}
    return out


def public_ids(inp: Inputs) -> dict:
    casia = [r for role in SOURCE_ROLES for r in inp.rows[role]]
    ids = rm.identifiers("casia_fasd", {"all": casia})
    ids["names"] |= {r["key"] for r in inp.rows["target_test"]} | {r["key"] for r in casia}
    ids["hashes"] |= {r["sha256"] for r in inp.rows["target_test"]}
    return ids


def public_scan(text: str, inp: Inputs, what: str, markdown: bool = False) -> None:
    bad = rm.public_text_problems(text, public_ids(inp), markdown=markdown)
    nuaa = sorted(n for r in inp.rows["target_test"] for n in (r["key"], nm.name_of(r["key"])) if n in text)
    if nuaa:
        bad.append(f"{len(nuaa)} NUAA image names or keys")
    if bad:
        fail(f"{what}: identifiers in a public file: " + "; ".join(bad))


def threshold_doc(ctx: Ctx, models: dict, raw_sha: str, train_rec: dict) -> dict:
    return {"schema": "c2-threshold/1", "passed": True, "run_id": ctx.run.name, "set": "source_dev", "criterion": "eer",
            "decision_rule": DECISION_RULE, "selection_rule": SELECTION_RULE, "config_sha256": ctx.cfg_sha,
            "check_sha256": sha256_file(ctx.run / "check.json"),
            "raw_scores_dev_sha256": raw_sha, "select_threshold_py_sha256": sha256_file(ROOT / "scripts" / "select_threshold.py"),
            "models": {mid: {**e, "checkpoint_sha256": train_rec["models"][mid]["weights_sha256"]} for mid, e in models.items()},
            "also_reported": "fixed threshold 0.5, in the aggregate table only, labelled"}


def cmd_select(a) -> None:
    ctx = open_run(a, "select", caches=False)
    remove_outputs(ctx.run, "select")
    inp = load_inputs(ctx.cfg, a, caches=False)
    train_rec = checked_weights(ctx, inp)
    rows = check_raw(ctx, inp, train_rec, "source_dev", "score_dev.json")
    models = selection(ctx, rows)
    doc = threshold_doc(ctx, models, sha256_file(ctx.run / RAW["dev"]), train_rec)
    doc["created"] = now()
    text = json.dumps(doc, indent=1) + "\n"
    public_scan(text, inp, "threshold.json")
    write_all(ctx.run, {"threshold.json": text})
    try:   # rilettura dal disco e ricalcolo dei conteggi con score > valore
        back = json.loads((ctx.run / "threshold.json").read_text(encoding="utf-8"))
        for mid in models:
            y, s, _, _ = arrays(rows, mid)
            v = back["models"][mid]["value"]
            if not isinstance(v, float) or v != models[mid]["value"] or \
                    {k: back["models"][mid]["dev"][k] for k in ("accepted_attacks", "rejected_bona_fide")} != \
                    {k: st.threshold_counts(y, s, v)[k] for k in ("accepted_attacks", "rejected_bona_fide")}:
                fail(f"{mid}: threshold.json read back differs from the selection")
    except BaseException:
        (ctx.run / "threshold.json").unlink(missing_ok=True)
        raise
    for mid, e in models.items():
        d = e["dev"]
        print(f"== select {mid}: eer threshold {e['value']!r}; dev APCER {d['apcer']} ({d['accepted_attacks']}/{d['n_attack']}), "
              f"BPCER {d['bpcer']} ({d['rejected_bona_fide']}/{d['n_bona_fide']}), ties {d['ties_at_value']}"
              + (f"; {e['flag']}" if e["flag"] else ""))


def verify_threshold(ctx: Ctx, inp: Inputs, train_rec: dict) -> dict:
    """threshold.json uguale al ricalcolo di select sui punteggi grezzi del dev (valori, conteggi, hash): un file
    modificato a mano è rifiutato anche se i record a valle sono stati aggiornati. Restituisce i valori per modello."""
    p = ctx.run / "threshold.json"
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        fail("threshold.json missing or not valid JSON")
    if p.is_symlink() or not isinstance(doc, dict) or doc.get("run_id") != ctx.run.name:
        fail("threshold.json belongs to another run or is a link: refused")
    rows = check_raw(ctx, inp, train_rec, "source_dev", "score_dev.json")
    want = threshold_doc(ctx, selection(ctx, rows), sha256_file(ctx.run / RAW["dev"]), train_rec)
    got = {k: v for k, v in doc.items() if k != "created"} if isinstance(doc, dict) else {}
    if json.loads(json.dumps(want)) != got:
        fail("threshold.json differs from the selection recomputed on the raw dev scores (edited or stale): refused")
    return {mid: e["value"] for mid, e in want["models"].items()}


# ============================================================================ apply

def cell(compute_metrics, by_type, y, s, subj, types, t: float, ds: str) -> dict:
    op = st.threshold_counts(y, s, t)
    bt = by_type(y, s, types, threshold=t, expected_types=list(ATTACK_TYPES[ds])) if len(y) else None
    return {"threshold": t, **{k: op[k] for k in ("n", "n_bona_fide", "n_attack", "accepted_attacks", "rejected_bona_fide")},
            "ties_at_threshold": op["ties_at_value"], "apcer": op["apcer"], "bpcer": op["bpcer"], "acer": op["acer"],
            **subjects(y, subj),
            "apcer_by_attack_type": {k: {"n_attack": v["n_attack"], "accepted": v["accepted"], "apcer": v["apcer"]}
                                     for k, v in (bt or {"by_type": {}})["by_type"].items()},
            "apcer_worst_attack_type": max((v["apcer"] for v in (bt or {"by_type": {}})["by_type"].values() if v["apcer"] is not None), default=None)}


def dispersion(y, s, subj, t: float) -> dict:
    """Dispersione per soggetto alla soglia (uscita riservata, senza identificativi): BPCER di ogni soggetto con bona
    fide e APCER di ogni soggetto con attacchi, come valori ordinati con minimo, mediana e massimo."""
    out = {}
    for lab, name in ((0, "bpcer_per_subject"), (1, "apcer_per_subject")):
        per = {}
        for l, v, k in zip(y, s, subj):
            if l == lab:
                n, e = per.get(k, (0, 0))
                per[k] = (n + 1, e + int((v > t) if lab == 0 else (v <= t)))
        vals = sorted(e / n for n, e in per.values())
        out[name] = {"n_subjects": len(vals), "values_sorted": vals, "min": vals[0] if vals else None,
                     "median": float(np.median(vals)) if vals else None, "max": vals[-1] if vals else None}
    return out


def fmt(v) -> str:
    return "null" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def table_md(doc: dict) -> str:
    L = ["# C2 pilot: CASIA-FASD -> NUAA, aggregate results", "",
         f"Run `{doc['run_id']}`; configuration SHA-256 `{doc['config_sha256']}`; threshold.json SHA-256 `{doc['threshold_sha256']}`.",
         "Threshold frozen on the source dev (criterion eer); decision: attack if score_attack > threshold (tie = bona fide). "
         "The rows at 0.5 are the fixed threshold, reported for reference only. Unit: frame / image. Missing class: null.", "",
         "| id | model | set | threshold | n | bona fide | attack | accepted attacks | rejected bona fide | ties | APCER | BPCER | ACER | "
         "subjects bf / attack | APCER by attack type (accepted/n) |", "|" + "---|" * 15]
    for e in doc["evaluations"]:
        for label, c in (("dev threshold", e["at_dev_threshold"]), ("0.5 (fixed, reference)", e["at_fixed_0_5"])):
            types = ", ".join(f"{k} {fmt(v['apcer'])} ({v['accepted']}/{v['n_attack']})" for k, v in c["apcer_by_attack_type"].items())
            L.append(f"| {e['id']} | {e['model']} | {e['set']} | {label}: {c['threshold']!r} | {c['n']} | {c['n_bona_fide']} | {c['n_attack']} | "
                     f"{c['accepted_attacks']} | {c['rejected_bona_fide']} | {c['ties_at_threshold']} | {fmt(c['apcer'])} | {fmt(c['bpcer'])} | "
                     f"{fmt(c['acer'])} | {c['n_subjects_bona_fide']} / {c['n_subjects_attack']} | {types} |")
    L += ["", "Descriptive (threshold-free): " + "; ".join(f"{e['id']} AUC {fmt(e['descriptive']['auc'])}, EER {fmt(e['descriptive']['eer'])}, "
                                                          f"BPCER@APCER10 {fmt(e['descriptive']['bpcer_at_apcer10'])}" for e in doc["evaluations"]), ""]
    return "\n".join(L)


def cmd_apply(a) -> None:
    ctx = open_run(a, "apply", caches=False)
    remove_outputs(ctx.run, "apply")
    inp = load_inputs(ctx.cfg, a, caches=False)
    train_rec = checked_weights(ctx, inp)
    st_rec = load_record(ctx.run, "score_tests.json", ctx, "score_tests")
    thr_p = ctx.run / "threshold.json"
    if not thr_p.is_file():
        fail("threshold.json is missing")
    thr_sha = sha256_file(thr_p)
    if st_rec.get("threshold_sha256_before_scoring") != thr_sha:
        fail("threshold.json changed since its SHA-256 was recorded before scoring the test sets: refused")
    values = verify_threshold(ctx, inp, train_rec)
    raw = {"source_dev": check_raw(ctx, inp, train_rec, "source_dev", "score_dev.json")}
    for role in ("source_test", "target_test"):
        raw[role] = check_raw(ctx, inp, train_rec, role, "score_tests.json")
    compute_metrics, by_type = metrics_api()

    def decided(rows):
        out = []
        for r in rows:
            t = values[r["model_id"]]
            out.append({**r, "threshold": repr(t), "prediction": str(int(float(r["score_attack"]) > t))})
        return out
    files = {}
    evals = []
    for e, (i, role) in EVALUATIONS.items():
        mid, arch = model_list(ctx.cfg)[i]
        rows = [r for r in raw[role] if r["model_id"] == mid]
        dec = decided(rows)
        if p := score_problems(dec, role, ctx, inp, train_rec, "decision", values, only=mid):
            fail(f"decision records of {e}: " + "; ".join(p))
        files[f"scores_{e}.csv"] = csv_text(sorted(dec, key=lambda r: r["sample_id"]))
        y, s, subj, types = arrays(rows, mid)
        ds = ROLES[role][0]
        m = compute_metrics(y, s) if len(y) else {}
        evals.append({"id": e, "model": mid, "arch": arch, "set": role, "dataset": ds,
                      "at_dev_threshold": cell(compute_metrics, by_type, y, s, subj, types, values[mid], ds),
                      "at_fixed_0_5": {**cell(compute_metrics, by_type, y, s, subj, types, 0.5, ds), "label": "fixed threshold 0.5 (reference only)"},
                      "descriptive": {"auc": m.get("auc"), "eer": m.get("eer"), "bpcer_at_apcer10": m.get("bpcer_at_apcer10")}})
    files["scores_dev.csv"] = csv_text(decided(raw["source_dev"]))
    disp = {"schema": "c2-subject-dispersion/1", "run_id": ctx.run.name, "restricted": True, "threshold": "frozen dev threshold",
            "note": "per-subject error rates at the frozen threshold, no identifiers; frames of a subject are not independent",
            "evaluations": {}}
    for e, (i, role) in EVALUATIONS.items():
        mid = model_list(ctx.cfg)[i][0]
        y, s, subj, _ = arrays([r for r in raw[role] if r["model_id"] == mid], mid)
        disp["evaluations"][e] = {"model": mid, "set": role, **dispersion(y, s, subj, values[mid])}
    files["subject_dispersion.json"] = json.dumps(disp, indent=1) + "\n"
    doc = {"schema": "c2-table/1", "run_id": ctx.run.name, "config_sha256": ctx.cfg_sha, "threshold_sha256": thr_sha,
           "unit": "frame / image", "decision_rule": DECISION_RULE,
           "apcer": "pooled over all attacks (group convention); per attack type as ISO/IEC 30107-3 (worst case reported)",
           "attack_types": {"casia_fasd": "token mapping of the Idiap bob.db.casia_fasd interface (v2.0.9, get_type): 3, 4, HR_2 warped; "
                                          "5, 6, HR_3 cut; 7, 8, HR_4 video; assumed for the mirror's file names (declared)",
                            "nuaa": "print"},
           "missing_class": "null with counts, never 0", "evaluations": evals}
    jtext = json.dumps(doc, indent=1) + "\n"
    mtext = table_md(doc)
    public_scan(jtext, inp, "c2_table.json")
    public_scan(mtext, inp, "c2_table.md", markdown=True)
    files["c2_table.json"], files["c2_table.md"] = jtext, mtext
    files["apply.json"] = json.dumps({**record_base(ctx, "apply"), "threshold_sha256": thr_sha, "files": {n: sha256_text(t) for n, t in files.items()},
                                      "created": now()}, indent=1) + "\n"
    write_all(ctx.run, files)
    for e in evals:
        c = e["at_dev_threshold"]
        print(f"== {e['id']} {e['model']} on {e['set']}: APCER {fmt(c['apcer'])} BPCER {fmt(c['bpcer'])} ACER {fmt(c['acer'])} "
              f"(n {c['n']}, bona fide {c['n_bona_fide']}, attack {c['n_attack']})")


# ============================================================================ main

def cmd_all(a) -> None:
    cmd_check(a)
    cmd_train(a)
    a.set = "dev"
    cmd_score(a)
    cmd_select(a)
    a.set = "source_test,target"
    cmd_score(a)
    cmd_apply(a)


COMMANDS = {"check": cmd_check, "train": cmd_train, "score": cmd_score, "select": cmd_select, "apply": cmd_apply, "all": cmd_all}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="C2 pilot CASIA-FASD -> NUAA: check, train, score, select, apply")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in COMMANDS:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, help="configuration JSON (c2_pilot_casia_to_nuaa_v2.json)")
        p.add_argument("--run-dir", required=True, help="run folder YYYYMMDD-C2-seed42-<shortcommit>[-trial]")
        p.add_argument("--source-manifests", default=str(RESULTS / "m2_rgb" / "casia_fasd" / "manifests"))
        p.add_argument("--target-manifests", default=str(RESULTS / "m2" / "manifests"))
        p.add_argument("--source-cache", default=str(DATA / "processed" / "casia_fasd_1500.npz"))
        p.add_argument("--target-cache", default=str(DATA / "processed" / "nuaa_1500.npz"))
        if name in ("train", "all"):
            p.add_argument("--epochs", type=int, required=True)
        if name in ("train", "score", "all"):
            p.add_argument("--i-am-authorized", action="store_true", help="explicit authorisation for training and inference")
        if name == "score":
            p.add_argument("--set", required=True, help="dev | source_test,target")
    return ap


def accept_run(run, step: str) -> None:
    """Cartella del run accettata (nome e posizione controllati): da qui status.json dice RUNNING finché il comando
    finisce, così un PASS precedente non resta come stato corrente se il comando si interrompe."""
    RUN_STATE.update(run=Path(run), step=step)
    write_status("RUNNING", RUN_STATE.get("cmd") or step)


def write_status(result: str, cmd: str, reason: str = "") -> None:
    """status.json della cartella del run accettata: esito dell'ultimo comando, passo, motivo e stato delle uscite."""
    run = RUN_STATE["run"]
    if run is None or not Path(run).is_dir():
        return
    if (Path(run) / STATUS).is_symlink() or ((Path(run) / STATUS).exists() and not (Path(run) / STATUS).is_file()):
        raise C2Error("status.json in the run folder is not a regular file")
    for e in os.listdir(run):                       # temporanei vecchi di status.json: si tolgono, mai seguiti
        if e == STATUS + ".tmp" or (e.startswith(f".{STATUS}.") and e.endswith(".tmp")):
            q = Path(run) / e
            if q.is_symlink() or q.is_file():
                q.unlink()
    doc = {"result": result, "command": cmd, "step": RUN_STATE["step"], "created": now()}
    if result == "FAIL":
        doc["reason"] = reason
        doc["outputs"] = ("removed: outputs of this step and of the later steps" if RUN_STATE["removed"]
                          else "untouched: refused before any change (options, identity), the existing outputs are not current")
    write_all(run, {STATUS: json.dumps(doc, indent=1) + "\n"})


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    RUN_STATE.update(run=None, step=None, removed=False, cmd=a.cmd)
    code = 1
    try:
        COMMANDS[a.cmd](a)
        write_status("PASS", a.cmd)
    except C2Error as e:
        reason = str(e)
    except SystemExit as e:     # fail() di un modulo importato: messaggio già stampato
        if e.code in (0, None):
            raise
        reason = f"invalid input (exit {e.code})"
    except KeyboardInterrupt:   # interruzione: FAIL, mai il PASS di un comando precedente
        reason, code = "interrupted (KeyboardInterrupt)", 130
    except Exception as e:      # noqa: BLE001  errore imprevisto (OSError, KeyError, errori di TensorFlow): mai un PASS
        reason = f"unexpected error {type(e).__name__}: {e}"
    else:
        print(f"RESULT: PASS ({a.cmd})", flush=True)
        return 0
    print(f"ERROR: {reason}", file=sys.stderr, flush=True)
    try:
        write_status("FAIL", a.cmd, reason)
    except (OSError, C2Error) as e:
        print(f"ERROR: status.json not written ({e})", file=sys.stderr, flush=True)
    print(f"RESULT: FAIL ({a.cmd})", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
