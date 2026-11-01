"""
N2, parte 2: il checkpoint storico AttackNetV2_2_combined_all_final.h5 (addestrato dal gruppo del coordinatore su
combined_all, senza NUAA) applicato senza riaddestramento alle stesse 3.000 immagini del test NUAA di N1, con due
varianti di preparazione delle immagini fissate prima del test e riportate entrambe: immagine intera (come
create_datasets.py tratta i file che riceve) e volto ritagliato (rilevatore attuale dell'app, margine 0,15, quadrato,
immagine intera quando non trova un volto). Catena storica per entrambe: BGR->RGB, Lanczos 256x256,
advanced_image_enhancement, /255, (x-0,5)*2. Soglia 0,5 fissata prima del test (regola argmax del codice storico);
nessuna calibrazione, nessuna inversione, nessun filtro di qualità.

Sottocomandi, sempre con --config <JSON> --manifests <cartella dei manifest M2 NUAA> --images <radice degli originali>
--model <file .h5> --n1-study <cartella dello studio N1> --study-dir <cartella dello studio>:
  check      nessun rilevatore, nessun TensorFlow: configurazione (SHA-256 vincolato nel codice per i run non -trial),
             manifest (official_test, cnn_test, c1) per SHA-256, righe e conteggi, c1 dentro cnn_test riga per riga,
             originali dei 3.000 presenti con SHA-256 e dimensione uguali al manifest, file del modello per SHA-256 e
             dimensione, face.py attuale, risorse del rilevatore e codice di riferimento per SHA-256, aggregato N1
             (SHA-256) con le righe di confronto. Crea la cartella dello studio (nuova) e scrive check.json.
  score      --variant whole_image|face_crop --i-am-authorized: pre-elaborazione dei 3.000 secondo la variante,
             rilevatore usato e ripiego registrati per frame, duplicati esatti di pixel rifiutati, punteggi su CPU con il
             modello caricato così com'è; <variante>/scores_test.csv, frames.csv, per_subject.csv (riservati),
             metrics.json, score.json; la cartella della variante deve essere nuova.
  aggregate  rilegge le due varianti, ricalcola le metriche, affianca le tre righe N1 (aggregate.json di N1, SHA-256
             vincolato) e scrive aggregate.json, aggregate.md e figure_distributions.png (pubblicabili, scansione del
             contenuto vietato) e per_subject_all_variants.csv (riservato).

Infrastruttura (scritture, log, record, metriche, scansione del pubblico) da scripts/n1_train.py e
scripts/n2_sessions.py, invariati. Comportamento su errore come n1_train.py. La cartella dello studio si chiama
YYYYMMDD-N2-combined-<commit breve>[-trial]; dentro il repository di sviluppo è ammessa solo sotto results/n2/.

Dipendenze: numpy, scikit-learn (metriche); OpenCV e il codice di riferimento per score; TensorFlow/Keras 2.15 e h5py
per score (il modello è un HDF5 di Keras 2.15); matplotlib per la figura di aggregate.
"""
import argparse
import csv
import hashlib
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
from tesi_app.paths import DATA, LIVEDETECTION, RESULTS, WEIGHTS, load_module  # noqa: E402

fail = N1.fail
N2Error = N1.N1Error
sha256_file, sha256_text, now, write_all = N1.sha256_file, N1.sha256_text, N1.now, N1.write_all
THRESHOLD, DECISION_RULE, NA = N1.THRESHOLD, N1.DECISION_RULE, N1.NA

CONFIG_SCHEMA = "n2-nuaa-combined-config/1"
# SHA-256 della configurazione consegnata (delivery/n2/n2_combined.json): un run non -trial accetta solo questa
CONFIG_SHA256 = "0527cc91c88700b94550a178bb78f890e8cc1d987bf1ef932b30afb61567c154"
RUN_ID_RE = re.compile(r"^(\d{8})-N2-combined-([0-9a-f]{7,40})(-trial)?$")
VARIANTS = ("whole_image", "face_crop")
SETS = dict(N1.SETS)
MANIFESTS = ("official_test", "cnn_test", "c1")
COUNT_FIELDS = N2.COUNT_FIELDS
TARGET_SIZE = (256, 256)
CODE_FILES = ("scripts/n2_combined.py", "scripts/n2_sessions.py", "scripts/n1_train.py", "scripts/nuaa_manifests.py", "scripts/select_threshold.py",
              "scripts/path_guard.py", "scripts/m2_writers.py", "scripts/package_rules.py", "src/tesi_app/evaluation.py", "src/tesi_app/paths.py",
              "src/tesi_app/face.py")
REFERENCE_CODE = {"LIVENESSLAB_LIVEDETECTION/scripts/create_datasets.py": ("scripts", "create_datasets.py"),
                  "LIVENESSLAB_LIVEDETECTION/src/dataset_loader.py": ("src", "dataset_loader.py")}
SCORE_OUTPUTS = ("scores_test.csv", "frames.csv", "per_subject.csv", "metrics.json", "score.json")
AGG_OUTPUTS = ("aggregate.json", "aggregate.md", "per_subject_all_variants.csv", "figure_distributions.png")
SCORE_COLUMNS = ["run_id", "variant", "sample_id", "subject_id", "session_id", "label", "score_attack", "prediction", "in_c1_subset", "detector",
                 "file_sha256", "frame_sha256", "model_sha256", "config_sha256"]
FRAME_COLUMNS = ["run_id", "variant", "sample_id", "detector", "file_sha256", "frame_sha256"]
SUBJECT_COLUMNS = ["run_id", "variant", "set", "subject_id", "n_bona_fide", "n_attack", "rejected_bona_fide", "accepted_attacks", "bpcer", "apcer", "acer"]
N1_METRICS = ("apcer", "bpcer", "acer", "auroc", "n_bona_fide", "n_attack", "rejected_bona_fide", "accepted_attacks")

# punti di iniezione (test)
PREPROCESSOR = None      # (variant, cfg) -> funzione(path) -> (frame uint8 256x256x3, nome breve del rilevatore o "n/a")
MAKE_SCORER = None       # (model_path, model_sha256) -> funzione(X uint8) -> probabilità N x 2
PLOTTER = None           # (doc) -> {"figure_distributions.png": byte PNG}


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
    want = {"schema", "study", "architecture", "dataset", "protocol", "model", "variants", "detector_resources", "reference_code_sha256", "quality_filter",
            "scoring", "metrics", "n1_reference", "data"}
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
    mdl = cfg["model"]
    if not isinstance(mdl, dict) or not N2._hex(mdl.get("sha256")) or not N2._int(mdl.get("bytes")) or not isinstance(mdl.get("file"), str) \
            or mdl.get("input_shape") != list(TARGET_SIZE) + [3] or mdl.get("classes") != {"0": "bona fide", "1": "attack"}:
        out.append("model needs file, sha256, bytes, input_shape [256, 256, 3] and classes 0 bona fide / 1 attack")
    v = cfg["variants"]
    if not isinstance(v, dict) or tuple(v) != VARIANTS:
        out.append(f"variants must be exactly {list(VARIANTS)} in this order")
    else:
        fc = v["face_crop"]
        if not isinstance(fc.get("detector"), dict) or not N2._hex(fc["detector"].get("sha256")) or "fallback" not in fc:
            out.append("variants.face_crop needs the detector SHA-256 and the fallback rule")
    if not isinstance(cfg["detector_resources"], dict) or not all(N2._hex(x) for x in cfg["detector_resources"].values()):
        out.append("detector_resources must map file names to SHA-256 values")
    if set(cfg["reference_code_sha256"]) != set(REFERENCE_CODE) or not all(N2._hex(x) for x in cfg["reference_code_sha256"].values()):
        out.append("reference_code_sha256 must give the SHA-256 of create_datasets.py and dataset_loader.py")
    sc = cfg["scoring"]
    if not isinstance(sc, dict) or sc.get("device") != "CPU" or sc.get("batch") != N1.SCORE_BATCH:
        out.append("scoring must be CPU with the N1 batch")
    m = cfg["metrics"]
    if not isinstance(m, dict) or m.get("threshold") != THRESHOLD or m.get("decision") != DECISION_RULE or m.get("sets") != list(SETS):
        out.append(f"metrics must fix threshold {THRESHOLD}, the decision rule and the sets {list(SETS)}")
    n1 = cfg["n1_reference"]
    if not isinstance(n1, dict) or not N2._hex(n1.get("aggregate_sha256")) or n1.get("train_seeds") != [42, 43, 44] or not N1.RUN_ID_RE.match(str(n1.get("run_id", ""))):
        out.append("n1_reference needs run_id, aggregate_sha256 and train_seeds [42, 43, 44]")
    d = cfg["data"]
    if not isinstance(d, dict) or set(d) != {"m2_run", "manifests", "originals"} or set(d.get("manifests", {})) != set(MANIFESTS):
        out.append("data must hold m2_run, originals and the three manifests")
    else:
        for n, e in d["manifests"].items():
            if not isinstance(e, dict) or not isinstance(e.get("file"), str) or not N2._hex(e.get("sha256")) or not all(N2._int(e.get(f)) for f in COUNT_FIELDS):
                out.append(f"data.manifests.{n}: file, sha256 and integer counts expected")
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
        return f"study folder name {d.name!r} is not YYYYMMDD-N2-combined-<shortcommit>[-trial]"
    if d.is_symlink():
        return f"{d.name} is a symbolic link: the study folder must be a real folder"
    if d.exists() and not d.is_dir():
        return f"{d.name} exists and is not a folder"
    for g in N1.outer_roots():
        if pg.within(d, g) or pg.contains(d, g):
            return f"{d} is inside, or contains, the work tree {g.name} with the accepted packages: outputs go outside it"
    if msg := pg.deletion_problem(d, ROOT, pg.BASE_PROTECTED, [ROOT / "results" / "n2"]):
        return msg
    for q in N2.protected_extra():
        if pg.within(d, q) or (Path(q).exists() and pg.contains(d, q)):
            return f"{d} is, contains or is inside a protected folder ({q.name})"
    return ""


INPUT_OPTIONS = (("--config", "config", "file"), ("--manifests", "manifests", "dir"), ("--images", "images", "dir"), ("--model", "model", "file"),
                 ("--n1-study", "n1_study", "dir"))


def inputs_problem(a, d) -> str:
    for opt, attr, kind in INPUT_OPTIONS:
        p = Path(getattr(a, attr))
        if pg.within(p, d) or pg.contains(p, d):
            return f"{opt} {p.name} is, contains or is inside the study folder: inputs and outputs must be separate"
        if kind == "dir" and not p.is_dir():
            return f"{opt} {p.name} is not a folder"
        if kind == "file" and not p.is_file():
            return f"{opt} {p.name} not found"
    return ""


def is_trial(d) -> bool:
    return Path(d).name.endswith("-trial")


def code_hashes() -> dict:
    out = {f: sha256_file(ROOT / f) for f in CODE_FILES}
    for key, rel in REFERENCE_CODE.items():
        p = Path(LIVEDETECTION).joinpath(*rel)
        out[key] = sha256_file(p) if p.is_file() else "absent"
    return out


def frozen_code_problems(cfg: dict, a) -> list:
    out = []
    mp = Path(a.model)
    if mp.name != cfg["model"]["file"]:
        out.append(f"--model {mp.name} is not {cfg['model']['file']}")
    if mp.is_symlink() or not mp.is_file():
        out.append("model file missing or not a regular file")
    else:
        if mp.stat().st_size != cfg["model"]["bytes"]:
            out.append(f"model file size {mp.stat().st_size} differs from the configuration ({cfg['model']['bytes']})")
        if sha256_file(mp) != cfg["model"]["sha256"]:
            out.append("model file: SHA-256 differs from the configuration")
    if sha256_file(ROOT / "src" / "tesi_app" / "face.py") != cfg["variants"]["face_crop"]["detector"]["sha256"]:
        out.append("src/tesi_app/face.py: SHA-256 differs from the configuration (face_crop detector)")
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


# ============================================================================ ingressi

class Inputs:
    pass


def load_inputs(cfg: dict, a, originals: bool = True) -> Inputs:
    inp = Inputs()
    inp.checks = {}
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
    inp.test = sorted(inp.rows["cnn_test"], key=lambda r: r["key"])
    inp.c1_keys = set(inp.by_key["c1"])
    if originals:
        layouts = Counter()
        for r in inp.test:
            N2.locate(Path(a.images), r, layouts)
        inp.checks["originals"] = {"n_verified": len(inp.test), "by_layout": dict(sorted(layouts.items())),
                                   "rule": "file regular, size and SHA-256 equal to the manifest row"}
    if problems := frozen_code_problems(cfg, a):
        fail("frozen code check failed: " + "; ".join(problems))
    inp.checks["frozen_code"] = {"model": {k: cfg["model"][k] for k in ("file", "sha256", "bytes")}, "face_crop_detector": cfg["variants"]["face_crop"]["detector"],
                                 "detector_resources": cfg["detector_resources"], "reference_code_sha256": cfg["reference_code_sha256"]}
    load_n1_reference(cfg, a, inp)
    return inp


def load_n1_reference(cfg: dict, a, inp: Inputs) -> None:
    """aggregate.json dello studio N1 (SHA-256 vincolato): le righe LivenessNet dei tre seed per i due insiemi."""
    n1 = cfg["n1_reference"]
    d = Path(a.n1_study)
    if d.name != n1["run_id"]:
        fail(f"--n1-study {d.name} is not the N1 study {n1['run_id']}")
    p = d / n1["aggregate_file"]
    if p.is_symlink() or not p.is_file() or sha256_file(p) != n1["aggregate_sha256"]:
        fail("N1 aggregate.json missing or SHA-256 differs from the configuration")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("passed") is not True or doc.get("step") != "aggregate" or doc.get("run_id") != n1["run_id"] or doc.get("train_seeds") != n1["train_seeds"] \
            or doc.get("threshold") != THRESHOLD or doc.get("decision") != DECISION_RULE or set(doc.get("sets", {})) != set(SETS):
        fail("N1 aggregate.json does not record the N1 study with the same threshold, decision rule and sets")
    inp.n1 = {}
    for s in n1["train_seeds"]:
        try:
            met = doc["seeds"][str(s)]["metrics"]
            inp.n1[str(s)] = {name: {k: met[name][k] for k in N1_METRICS} for name in SETS}
            inp.n1[str(s)]["weights_sha256"] = doc["seeds"][str(s)]["training"]["weights_sha256"]
        except (KeyError, TypeError):
            fail(f"N1 aggregate.json: metrics of seed {s} missing")
    inp.checks["n1_reference"] = {"run_id": n1["run_id"], "aggregate_sha256": n1["aggregate_sha256"], "seeds": sorted(inp.n1)}


def same_inputs(inp: Inputs, check: dict) -> None:
    for k in ("manifests", "frozen_code", "n1_reference"):
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
    check = N1.load_record(d / "check.json", "check", d.name, cfg_sha)      # il check di questa parte non scrive altri file
    if check.get("code_sha256") != code_hashes():
        fail("the code changed since check (SHA-256 of the scripts differ from check.json): start a new study")
    if not is_trial(d) and N1.repo_state(d).get("commit") != check.get("commit"):
        fail("non-trial run: the repository commit differs from check")
    return d, cfg, cfg_sha, check, sha256_file(d / "check.json")


# ============================================================================ pre-elaborazione e scorer

def historical_chain(rgb: np.ndarray, creator) -> np.ndarray:
    """Catena di create_datasets.process_single_image dopo la conversione BGR->RGB: Lanczos 256x256, enhancement, uint8."""
    import cv2
    resized = cv2.resize(rgb, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
    return np.asarray(creator.advanced_image_enhancement(resized), dtype=np.uint8)


def default_preprocessor(variant: str, cfg: dict):
    try:
        import cv2
    except ModuleNotFoundError as e:
        fail(f"missing Python package {e.name!r}: the preprocessing needs OpenCV")
    cd = load_module("kuznetsov_create_datasets", Path(LIVEDETECTION) / "scripts" / "create_datasets.py")
    creator = cd.EnhancedDatasetCreator.__new__(cd.EnhancedDatasetCreator)
    det = None
    if variant == "face_crop":
        from tesi_app.face import FaceDetector   # il rilevatore attuale dell'app, SHA-256 verificato al check
        det = FaceDetector()

    def pre(path):
        img = cv2.imread(str(path))
        if img is None:
            fail(f"{Path(path).name}: unreadable image")
        name = "n/a"
        if det is not None:
            f = det.detect(img)
            name = N2.detector_short(f)
            if f is not None:
                img = f.crop(img, margin=0.15, square=True)
                if img.size == 0:
                    fail(f"{Path(path).name}: empty crop")
        return historical_chain(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), creator), name
    return pre


def preprocessor(variant: str, cfg: dict):
    return (PREPROCESSOR or default_preprocessor)(variant, cfg)


def tf_model_scorer(model_path, model_sha: str):
    """Il modello storico caricato così com'è (Keras 2.15 HDF5, compile=False) su CPU; ingresso (x/255 - 0,5) * 2."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        import tensorflow as tf
    except ModuleNotFoundError as e:
        fail(f"missing Python package {e.name!r}: score needs TensorFlow")
    try:
        tf.config.set_visible_devices([], "GPU")
    except (RuntimeError, ValueError):
        pass
    if sha256_file(model_path) != model_sha:
        fail(f"{Path(model_path).name}: SHA-256 changed just before loading")
    with tf.device("/CPU:0"):
        model = tf.keras.models.load_model(str(model_path), compile=False)
    shape = tuple(model.input_shape[1:])
    if shape != TARGET_SIZE + (3,) or tuple(model.output_shape[1:]) != (2,):
        fail(f"model input {shape} / output {model.output_shape[1:]} differ from the expected (256, 256, 3) / (2,)")

    def score(X):
        out = []
        for i in range(0, len(X), N1.SCORE_BATCH):
            xb = (X[i:i + N1.SCORE_BATCH].astype(np.float32) / 255.0 - 0.5) * 2.0
            with tf.device("/CPU:0"):
                out.append(np.asarray(model(xb, training=False), dtype=np.float64))
        return np.concatenate(out) if out else np.zeros((0, 2))
    return score


# ============================================================================ check

def cmd_check(a) -> None:
    d, cfg, cfg_sha = parse_common(a, need_study=False)
    if d.exists() and any(d.iterdir()):
        fail("the study folder must be new (missing or empty): refused, nothing written")
    state = N1.repo_state(d)
    m = RUN_ID_RE.match(d.name)
    if not is_trial(d) and not str(state.get("commit", "")).startswith(m.group(2)):
        fail(f"non-trial run: the study id must carry the current commit ({str(state.get('commit'))[:7]})")
    d.mkdir(parents=True, exist_ok=True)
    N1.accept(d, "check", "check.log")
    print(f"== N2 combined check: study {d.name}, configuration SHA-256 {cfg_sha}")
    inp = load_inputs(cfg, a)
    for k, v in inp.checks.items():
        print(f"   PASS {k}: {json.dumps(v, sort_keys=True)}")
    doc = {"passed": True, "step": "check", "run_id": d.name, "trial": is_trial(d), "config_sha256": cfg_sha,
           "delivered_configuration": cfg_sha == CONFIG_SHA256, "checks": inp.checks, "detector": "not used", "model": "not loaded",
           "tensorflow": N2.tf_state(), "code_sha256": code_hashes(), "environment": N2.environment(), **state, "created": now()}
    write_all(d, {"check.json": json.dumps(doc, indent=1) + "\n"})
    print(f"== check PASS: 3 manifests, {len(inp.test)} originals verified, model file, detector and reference code bound, N1 aggregate bound")


# ============================================================================ score

def metrics_from_rows(rows) -> dict:
    out = {}
    for name, sel in (("test_3000", rows), ("c1_300", [r for r in rows if r["in_c1_subset"] == "1"])):
        out[name] = N1.set_metrics([int(r["label"]) for r in sel], [float(r["score_attack"]) for r in sel])
    return out


def cmd_score(a) -> None:
    d, cfg, cfg_sha, check, check_sha = open_study(a)
    if a.variant not in VARIANTS:
        fail(f"--variant must be one of {list(VARIANTS)}")
    if not a.i_am_authorized:
        fail("score needs --i-am-authorized (inference on the restricted NUAA data)")
    vd = d / a.variant
    if os.path.lexists(vd):
        fail(f"{vd.name} already exists: a variant folder is never reused (refused, nothing changed)")
    inp = load_inputs(cfg, a, originals=False)
    same_inputs(inp, check)
    vd.mkdir()
    N1.accept(vd, "score", "score.log")
    print(f"== N2 combined score: study {d.name}, variant {a.variant}, CPU")
    pre = preprocessor(a.variant, cfg)
    frames, dets = [], []
    for r in inp.test:
        arr, det = pre(N2.locate(Path(a.images), r))
        if arr.shape != TARGET_SIZE + (3,) or arr.dtype != np.uint8:
            fail(f"{r['key']}: preprocessed frame has shape {arr.shape} {arr.dtype}, expected {TARGET_SIZE + (3,)} uint8")
        frames.append(arr)
        dets.append(det)
    frames = np.stack(frames)
    hashes = nm.frame_hashes(frames)
    if len(set(hashes)) != len(hashes):
        fail(f"duplicated derived frames in variant {a.variant}: {len(hashes) - len(set(hashes))} (exact pixel hash)")
    p = N1.probabilities((MAKE_SCORER or tf_model_scorer)(a.model, cfg["model"]["sha256"]), frames)[:, 1]
    rows, frows, values = [], [], []
    for r, det, h, v in zip(inp.test, dets, hashes, p):
        v = float(v)
        values.append(v)
        rows.append({"run_id": d.name, "variant": a.variant, "sample_id": r["key"], "subject_id": r["subject"], "session_id": r["session"], "label": r["label"],
                     "score_attack": repr(v), "prediction": "1" if v > THRESHOLD else "0", "in_c1_subset": "1" if r["key"] in inp.c1_keys else "0",
                     "detector": det, "file_sha256": r["sha256"], "frame_sha256": h, "model_sha256": cfg["model"]["sha256"], "config_sha256": cfg_sha})
        frows.append({"run_id": d.name, "variant": a.variant, "sample_id": r["key"], "detector": det, "file_sha256": r["sha256"], "frame_sha256": h})
    scores_txt = N1.csv_text(rows, SCORE_COLUMNS)
    back = list(csv.DictReader(io.StringIO(scores_txt)))
    if len(back) != len(values) or any(float(b["score_attack"]) != v for b, v in zip(back, values)):
        fail("scores do not survive the CSV round trip")
    met = metrics_from_rows(back)
    subj = [{"run_id": d.name, "variant": a.variant, **x} for name in SETS
            for x in N1.per_subject(back if name == "test_3000" else [r for r in back if r["in_c1_subset"] == "1"], name)]
    frames_txt, subj_txt = N1.csv_text(frows, FRAME_COLUMNS), N1.csv_text(subj, SUBJECT_COLUMNS)
    met_doc = {"run_id": d.name, "variant": a.variant, "sets": SETS, "metrics": met, "model_sha256": cfg["model"]["sha256"]}
    met_txt = json.dumps(met_doc, indent=1) + "\n"
    detectors = dict(sorted(Counter(dets).items()))
    doc = {"passed": True, "step": "score", "run_id": d.name, "variant": a.variant, "config_sha256": cfg_sha, "check_sha256": check_sha,
           "model_sha256": cfg["model"]["sha256"], "device": "CPU", "n_scores": len(rows), "detectors": detectors,
           "fallback_whole_image": detectors.get("none", 0) if a.variant == "face_crop" else None, "environment": N2.environment(),
           "files": {"scores_test.csv": sha256_text(scores_txt), "frames.csv": sha256_text(frames_txt), "per_subject.csv": sha256_text(subj_txt),
                     "metrics.json": sha256_text(met_txt)}, "created": now()}
    write_all(vd, {"scores_test.csv": scores_txt, "frames.csv": frames_txt, "per_subject.csv": subj_txt, "metrics.json": met_txt,
                   "score.json": json.dumps(doc, indent=1) + "\n"})
    for name in SETS:
        m = met[name]
        print(f"   {name}: APCER {N1.fmt_na(m['apcer'])} ({m['accepted_attacks']}/{m['n_attack']}), BPCER {N1.fmt_na(m['bpcer'])} "
              f"({m['rejected_bona_fide']}/{m['n_bona_fide']}), ACER {N1.fmt_na(m['acer'])}, AUROC {N1.fmt_na(m['auroc'])}")
    print(f"== score PASS: variant {a.variant}, {len(rows)} scores, detectors {detectors}")


# ============================================================================ aggregate

def default_plotter(doc: dict) -> dict:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        fail(f"missing Python package {e.name!r}: the figure needs matplotlib")
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(5.2 * len(VARIANTS), 3.6), squeeze=False)
    for j, var in enumerate(VARIANTS):
        h = doc["variants"][var]["distributions"]["test_3000"]
        edges = h["bin_edges"]
        width = (edges[1] - edges[0]) / 2.2
        ax = axes[0][j]
        for cls, off, col in (("bona_fide", 0, "#2b6cb0"), ("attack", width, "#c05621")):
            tot = max(1, sum(h[cls]))
            ax.bar([e + off for e in edges[:-1]], [c / tot for c in h[cls]], width=width, align="edge", color=col, label=f"{cls.replace('_', ' ')} (n={sum(h[cls])})")
        ax.set_title(f"{var.replace('_', ' ')}: test 3,000")
        ax.set_xlabel("attack score")
        ax.set_ylabel("fraction of the class")
        ax.set_xlim(0, 1)
        ax.axvline(THRESHOLD, color="k", linestyle=":", linewidth=1)
        ax.legend(fontsize=8)
    fig.suptitle("Historical combined AttackNetV2.2 on the NUAA test: attack-score distributions per class (threshold 0.5 dotted)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return {"figure_distributions.png": buf.getvalue()}


def table_md(doc: dict) -> str:
    f4 = N1.f4
    L = [f"# N2 part 2: the historical combined AttackNetV2.2 on the 3,000 NUAA test frames of N1, two image-preparation variants (study `{doc['run_id']}`)", "",
         f"Checkpoint `{doc['model']['file']}` (SHA-256 `{doc['model']['sha256'][:12]}…`), loaded as saved, no retraining. Same 3,000 frames as N1 (manifest "
         f"`nuaa_cnn_test.csv`), every frame scored (no quality filter). Threshold {THRESHOLD} fixed before the test, {DECISION_RULE}; no calibration, no "
         "score inversion. The N1 rows are the three LivenessNet trainings of study "
         f"`{doc['n1_reference']['run_id']}`, read from its aggregate. The comparison with N1 changes architecture, training recipe and training data at once; "
         "the two variants are a sensitivity analysis to the image preparation, not a reproduction of the historical preparation.", "",
         "## Single table", "",
         "| model / variant | set | n bona fide | n attack | rejected bona fide | accepted attacks | APCER | BPCER | ACER | AUROC |", "|---|---|---|---|---|---|---|---|---|---|"]
    for var in VARIANTS:
        for name, title in SETS.items():
            m = doc["variants"][var]["metrics"][name]
            L.append(f"| combined AttackNetV2.2, {var.replace('_', ' ')} | {title} | {m['n_bona_fide']} | {m['n_attack']} | {m['rejected_bona_fide']} | "
                     f"{m['accepted_attacks']} | {f4(m['apcer'])} | {f4(m['bpcer'])} | {f4(m['acer'])} | {f4(m['auroc'])} |")
    for s in doc["n1_reference"]["train_seeds"]:
        for name, title in SETS.items():
            m = doc["n1_reference"]["seeds"][str(s)][name]
            L.append(f"| N1 LivenessNet, seed {s} | {title} | {m['n_bona_fide']} | {m['n_attack']} | {m['rejected_bona_fide']} | {m['accepted_attacks']} | "
                     f"{f4(m['apcer'])} | {f4(m['bpcer'])} | {f4(m['acer'])} | {f4(m['auroc'])} |")
    L += ["", "## Image preparation per variant", "", "| variant | detector counts on the 3,000 frames | frames without a face (whole image used) |", "|---|---|---|"]
    for var in VARIANTS:
        v = doc["variants"][var]
        L.append(f"| {var.replace('_', ' ')} | " + ", ".join(f"{k} {n}" for k, n in v["detectors"].items()) + f" | {NA if v['fallback_whole_image'] is None else v['fallback_whole_image']} |")
    L += ["", "## Score distributions per class, test 3,000 (counts per bin of width 0.1)", "",
          "| variant | class | " + " | ".join(f"[{e:.1f}, {e + 1 / N2.BINS:.1f}{')' if i < N2.BINS - 1 else ']'}" for i, e in
                                              enumerate(doc["variants"][VARIANTS[0]]["distributions"]["test_3000"]["bin_edges"][:-1])) + " |",
          "|---|---|" + "---|" * N2.BINS]
    for var in VARIANTS:
        for cls in ("bona_fide", "attack"):
            L.append(f"| {var.replace('_', ' ')} | {cls.replace('_', ' ')} | " + " | ".join(str(c) for c in doc["variants"][var]["distributions"]["test_3000"][cls]) + " |")
    L += ["", "NA = not computable (class absent). Values are rounded to four decimals; aggregate.json holds the exact values.", ""]
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
    variants, subj_all = {}, []
    for var in VARIANTS:
        vd = d / var
        sc = N1.load_record(vd / "score.json", "score", d.name, cfg_sha, check_sha)
        if sc.get("variant") != var or sc.get("model_sha256") != cfg["model"]["sha256"]:
            fail(f"{var}: score.json records another variant or model")
        for n, h in sc["files"].items():
            if (vd / n).is_symlink() or not (vd / n).is_file() or sha256_file(vd / n) != h:
                fail(f"{var}/{n} changed after scoring (SHA-256 differs from score.json)")
        rows = N1.read_csv_rows(vd / "scores_test.csv", SCORE_COLUMNS)
        if sc.get("detectors") != dict(Counter(r["detector"] for r in rows)):
            fail(f"{var}: detector counts of score.json differ from the rows")
        if [r["sample_id"] for r in rows] != [r["key"] for r in inp.test]:
            fail(f"{var}/scores_test.csv does not hold exactly the cnn_test rows in key order")
        frames = {r["sample_id"]: r for r in N1.read_csv_rows(vd / "frames.csv", FRAME_COLUMNS)}
        for r in rows:
            m = inp.by_key["cnn_test"][r["sample_id"]]
            exp = {"run_id": d.name, "variant": var, "subject_id": m["subject"], "session_id": m["session"], "label": m["label"],
                   "in_c1_subset": "1" if r["sample_id"] in inp.c1_keys else "0", "file_sha256": m["sha256"], "frame_sha256": frames[r["sample_id"]]["frame_sha256"],
                   "detector": frames[r["sample_id"]]["detector"], "model_sha256": cfg["model"]["sha256"], "config_sha256": cfg_sha}
            try:
                v = float(r["score_attack"])
            except ValueError:
                v = float("nan")
            if any(r[k] != x for k, x in exp.items()) or not (0 <= v <= 1) or r["prediction"] != ("1" if v > THRESHOLD else "0"):
                fail(f"{var}/scores_test.csv: rows inconsistent with the manifests, the frames, the model or the decision rule")
        met = metrics_from_rows(rows)
        stored = json.loads((vd / "metrics.json").read_text(encoding="utf-8"))["metrics"]
        if json.dumps(met, sort_keys=True) != json.dumps(stored, sort_keys=True):
            fail(f"{var}: metrics recomputed from the scores differ from metrics.json")
        subj_rows = N1.read_csv_rows(vd / "per_subject.csv", SUBJECT_COLUMNS)
        recomputed = [{"run_id": d.name, "variant": var, **x} for name in SETS
                      for x in N1.per_subject(rows if name == "test_3000" else [r for r in rows if r["in_c1_subset"] == "1"], name)]
        if N1.csv_text(subj_rows, SUBJECT_COLUMNS) != N1.csv_text(recomputed, SUBJECT_COLUMNS):
            fail(f"{var}: per_subject.csv differs from the table recomputed from the scores")
        subj_all += subj_rows
        variants[var] = {"metrics": met, "distributions": {"test_3000": N2.histogram(rows), "c1_300": N2.histogram([r for r in rows if r["in_c1_subset"] == "1"])},
                         "detectors": sc["detectors"], "fallback_whole_image": sc.get("fallback_whole_image"), "scores_sha256": sc["files"]["scores_test.csv"],
                         "chain": cfg["variants"][var]["chain"]}
    doc = {"passed": True, "step": "aggregate", "run_id": d.name, "config_sha256": cfg_sha, "check_sha256": check_sha, "commit": check.get("commit"),
           "dirty_at_check": check.get("dirty"), "code_sha256": check["code_sha256"], "model": {k: cfg["model"][k] for k in ("file", "sha256", "bytes", "source")},
           "sets": SETS, "threshold": THRESHOLD, "decision": DECISION_RULE, "threshold_origin": cfg["metrics"]["threshold_origin"],
           "quality_filter": cfg["quality_filter"], "variants": variants,
           "n1_reference": {"run_id": cfg["n1_reference"]["run_id"], "aggregate_sha256": cfg["n1_reference"]["aggregate_sha256"],
                            "train_seeds": cfg["n1_reference"]["train_seeds"], "seeds": {s: {n: inp.n1[s][n] for n in SETS} for s in inp.n1}},
           "selection": "none: both variants reported, threshold 0.5 fixed before the test, no score inversion; the choice of a candidate variant is a "
                        "recommendation to be verified on separate data, not a result of this test",
           "figures": {"figure_distributions.png": "attack-score distributions per class on the 3,000 test frames, one panel per variant"}, "created": now()}
    js, md = json.dumps(doc, indent=1) + "\n", table_md(doc)
    for name, text in (("aggregate.json", js), ("aggregate.md", md)):
        if probs := N1.public_problems(text, inp):
            fail(f"{name} would hold {', '.join(probs)}: not written")
    figs = (PLOTTER or default_plotter)(doc)
    if set(figs) != {"figure_distributions.png"} or not all(isinstance(v, bytes) and v[:8] == b"\x89PNG\r\n\x1a\n" for v in figs.values()):
        fail("the plotter did not return the PNG figure")
    write_all(d, {"aggregate.json": js, "aggregate.md": md, "per_subject_all_variants.csv": N1.csv_text(subj_all, SUBJECT_COLUMNS)})
    try:
        N2.write_binary(d, figs)
    except BaseException:
        for n in ("aggregate.json", "aggregate.md", "per_subject_all_variants.csv"):
            (d / n).unlink(missing_ok=True)
        raise
    print(md)
    print(f"== aggregate PASS: {len(variants)} variants, aggregate.json, aggregate.md and the figure (publishable), per_subject_all_variants.csv (restricted)")


# ============================================================================ main

COMMANDS = {"check": cmd_check, "score": cmd_score, "aggregate": cmd_aggregate}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="N2 part 2: the historical combined AttackNetV2.2 on the NUAA test of N1, two image-preparation variants")
    ap.add_argument("command", choices=list(COMMANDS))
    ap.add_argument("--config", required=True, help="configuration JSON (n2_combined.json)")
    ap.add_argument("--manifests", required=True, help="folder with the M2 NUAA manifests")
    ap.add_argument("--images", required=True, help="root folder of the original NUAA images")
    ap.add_argument("--model", required=True, help="AttackNetV2_2_combined_all_final.h5")
    ap.add_argument("--n1-study", required=True, help="the N1 study folder (aggregate.json)")
    ap.add_argument("--study-dir", required=True, help="study folder YYYYMMDD-N2-combined-<shortcommit>[-trial]")
    ap.add_argument("--variant", default=None, help="whole_image or face_crop (score)")
    ap.add_argument("--i-am-authorized", action="store_true", help="inference on the restricted data is authorised")
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    N1.RUN_STATE.update(dir=None, cmd=a.command, log=None)
    code = 0
    try:
        if a.command == "score" and a.variant is None:
            fail("score needs --variant")
        COMMANDS[a.command](a)
        N1.write_status("PASS")
        print(f"RESULT: PASS ({a.command})")
    except (N2Error, SystemExit, Exception, KeyboardInterrupt) as e:  # noqa: BLE001
        if isinstance(e, SystemExit) and e.code in (None, 0):
            raise
        msg = str(e) if isinstance(e, (N2Error, SystemExit)) else f"unexpected {type(e).__name__}: {e}"
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
