"""
N2, parte 1: i tre checkpoint LivenessNet di N1 (seed di training 42, 43 e 44, solo CPU) applicati ai 491 frame NUAA
delle sessioni 01-02 che stanno nella lista ufficiale di training ma non in cnn_train (fitting) né in
cnn_early_stopping (validazione), con il preprocessing con cui è stata costruita la cache di training (face.py
congelato al commit 0ac7143) e il confronto con la sessione 03 (i 3.000 punteggi del test N1, letti così come sono).

Sottocomandi, sempre con --config <JSON> --manifests <cartella dei manifest M2 NUAA> --cache <nuaa_1500.npz>
--images <radice delle immagini originali> --n1-study <cartella dello studio N1> --frozen-face <face_0ac7143.py>
--study-dir <cartella dello studio>:
  check        nessun rilevatore, nessun TensorFlow, nessun pixel: configurazione (SHA-256 vincolato nel codice per i run
               non -trial), SHA-256 e schema della cache, SHA-256, righe e conteggi dei sette manifest M2 (validazione
               stretta di nuaa_manifests.py, stessi campi per le chiavi condivise), derivazione dei 491 (official_train
               meno cnn_train meno cnn_es) con conteggi, disgiunzione per chiave e per SHA-256 da fitting, validazione,
               test e C1, sovrapposizione con il dev M1, gruppi di soggetti (fitting / validazione), originali presenti
               con SHA-256 e dimensione uguali al manifest, studio N1 (record, pesi, punteggi legati per SHA-256),
               face.py congelato, risorse del rilevatore e codice di riferimento per SHA-256. Crea la cartella dello
               studio (nuova) e scrive target_manifest.csv (riservato) e check.json; il log in check.log.
  cache-check  rilevatore congelato, nessun TensorFlow: ogni frame della cache (6.000, in ordine di campionamento)
               rifatto dall'originale e confrontato bit per bit; i 491 pre-elaborati, con il rilevatore usato per
               ciascuno e i duplicati esatti di pixel (interni, con la cache, dentro la cache). Scrive cache_check.json
               e frames_target.csv (riservato); il log in cache_check.log. --step N solo nei run -trial.
  parity       rilevatore e TensorFlow su CPU: per le chiavi N1 della lista di parità (regola e SHA-256 della lista fissati nella configurazione), frame rifatto uguale alla
               riga della cache e punteggio ricalcolato con ciascun checkpoint entro la tolleranza dal punteggio salvato
               in N1. Scrive parity.json e parity.csv (riservato); il log in parity.log.
  score        --train-seed N --i-am-authorized: i 491 pre-elaborati (hash dei frame uguali a cache-check), punteggi su
               CPU con il checkpoint N1 del seed (percorso di scoring di N1, invariato), scores_sessions.csv (per
               immagine, riservato), per_subject.csv, metrics.json e score.json; il log in seed<N>/score.log. La
               cartella seed<N> deve essere nuova.
  aggregate    rilegge i tre seed e i punteggi N1 della sessione 03 (SHA-256 vincolati), ricalcola le metriche, scrive
               aggregate.json e aggregate.md (pubblicabili: tabella unica, strati per gruppo, distribuzioni per classe,
               scansione del contenuto vietato), figure_distributions.png e figure_strata.png (pubblicabili, senza
               identificativi), per_subject_all_seeds.csv e strata_all_seeds.csv (riservati, strati soggetto x classe);
               il log in aggregate.log.

Metriche: soglia fissa 0,5 sulla probabilità di attacco (attacco se score_attack > 0,5, pareggio = bona fide), APCER,
BPCER, ACER con i denominatori, AUROC; classe assente = NA, mai 0. Nessuna scelta del seed, nessuna soglia nuova,
nessuna inversione del punteggio. Infrastruttura (scritture atomiche, log, record, scorer CPU, metriche) importata da
scripts/n1_train.py, invariato.

Comportamento su errore: come n1_train.py (rifiuti prima dell'accettazione della cartella non scrivono nulla; poi
status.json RUNNING e PASS / FAIL; log in coda; uscite scritte solo alla fine; uscite di cache-check, parity e score
mai sovrascritte; aggregate toglie le proprie uscite precedenti).

La cartella dello studio si chiama YYYYMMDD-N2-sessions-<commit breve>[-trial]; dentro il repository di sviluppo è
ammessa solo sotto results/n2/. Un run non -trial richiede la configurazione consegnata (SHA-256 nel codice) e il
commit corrente nel nome.

Dipendenze: numpy (tutti i passi), scikit-learn (metriche), OpenCV e il codice di riferimento (cache-check, parity,
score), TensorFlow solo per parity e score; matplotlib solo per le figure di aggregate.
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
import tempfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np  # noqa: E402
    import n1_train as N1  # noqa: E402  (infrastruttura e percorso di scoring di N1, invariati)
    import nuaa_manifests as nm  # noqa: E402
    import path_guard as pg  # noqa: E402
    import select_threshold as st  # noqa: E402
except ModuleNotFoundError as _e:   # dipendenza mancante: messaggio chiaro, mai un traceback
    print(f"ERROR: missing Python package {_e.name!r}: numpy is needed by every subcommand", file=sys.stderr)
    print("RESULT: FAIL (import)")
    sys.exit(1)
from tesi_app.paths import DATA, LIVEDETECTION, RESULTS, SILENT_FACE, WEIGHTS, load_module  # noqa: E402

fail = N1.fail
N2Error = N1.N1Error
sha256_file, sha256_text, now, write_all = N1.sha256_file, N1.sha256_text, N1.now, N1.write_all
THRESHOLD, DECISION_RULE, NA = N1.THRESHOLD, N1.DECISION_RULE, N1.NA

CONFIG_SCHEMA = "n2-nuaa-sessions-config/1"
# SHA-256 della configurazione consegnata (delivery/n2/n2_sessions.json): un run non -trial accetta solo questa
CONFIG_SHA256 = "41e6bfe7cff404fc69117610499fa7158575414d214fd708a618f38aa824e43f"
RUN_ID_RE = re.compile(r"^(\d{8})-N2-sessions-([0-9a-f]{7,40})(-trial)?$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TOLERANCE = 1e-6
# titoli degli insiemi riportati: i conteggi vengono dai dati del check (set_titles), mai da numeri scritti nel codice
TARGET_SETS = {"sessions_01_02_491": "sessions 01-02: the {n_target} target frames",
               "sessions_01_02_fitting_subjects": "sessions 01-02: target frames of the {n_fit} fitting subjects",
               "sessions_01_02_validation_subjects": "sessions 01-02: target frames of the {n_val} validation subjects",
               "sessions_01_02_not_in_m1_dev": "sessions 01-02: target frames outside the M1 dev set"}
REFERENCE_SETS = {"session_03_3000": "session 03: the {n_test} nuaa_cnn_test frames (N1 scores, read as stored)",
                  "session_03_9_subjects": "session 03: test frames of the {n_subj} target subjects (N1 scores)",
                  "session_03_fitting_subjects": "session 03: test frames of the {n_fit} fitting subjects (N1 scores)",
                  "session_03_validation_subjects": "session 03: test frames of the {n_val} validation subjects (N1 scores)"}
SETS = {**TARGET_SETS, **REFERENCE_SETS}


def set_titles(t: dict, n_test: int) -> dict:
    """Titoli degli insiemi con i conteggi del check (target e manifest), per metrics.json e per l'aggregato."""
    v = {"n_target": t["rows"], "n_fit": t["n_fitting_subjects"], "n_val": t["n_validation_subjects"], "n_subj": t["n_subjects"], "n_test": f"{n_test:,}"}
    return {k: tpl.format(**v) for k, tpl in SETS.items()}
PAIRS = (("sessions_01_02_491", "session_03_9_subjects"), ("sessions_01_02_fitting_subjects", "session_03_fitting_subjects"),
         ("sessions_01_02_validation_subjects", "session_03_validation_subjects"))
MANIFESTS = ("official_train", "official_test", "cnn_train", "cnn_es", "cnn_test", "c1", "m1_dev")
COUNT_FIELDS = ("rows", "n_bona_fide", "n_attack", "n_subjects")
CODE_FILES = ("scripts/n2_sessions.py", "scripts/n1_train.py", "scripts/train_cnn.py", "scripts/nuaa_manifests.py",
              "scripts/select_threshold.py", "scripts/path_guard.py", "scripts/m2_writers.py", "scripts/package_rules.py",
              "src/tesi_app/evaluation.py", "src/tesi_app/paths.py", "src/tesi_app/face.py")
REFERENCE_CODE = {"LIVENESSLAB_LIVEDETECTION/scripts/create_datasets.py": ("scripts", "create_datasets.py"),
                  "LIVENESSLAB_LIVEDETECTION/src/architectures.py": ("src", "architectures.py")}
STATUS = "status.json"
TARGET_MANIFEST = "target_manifest.csv"
PARITY_KEYS = "parity_keys.csv"
PARITY_KEY_COLUMNS = ["sample_id", "subject_id", "label"]
FRAMES_TARGET = "frames_target.csv"
CHECK_OUTPUTS = (TARGET_MANIFEST, PARITY_KEYS, "check.json")
CACHE_OUTPUTS = ("cache_check.json", FRAMES_TARGET)
PARITY_OUTPUTS = ("parity.json", "parity.csv")
SCORE_OUTPUTS = ("scores_sessions.csv", "per_subject.csv", "metrics.json", "score.json")
PUBLIC_FIGURES = ("figure_distributions.png", "figure_groups.png")        # costruite dal solo aggregate.json (già scansionato)
RESTRICTED_FIGURES = ("figure_strata.png",)                              # tassi per soggetto, senza identificativi: riservata come gli strati
AGG_OUTPUTS = ("aggregate.json", "aggregate.md", "per_subject_all_seeds.csv", "strata_all_seeds.csv") + PUBLIC_FIGURES + RESTRICTED_FIGURES
SCORE_COLUMNS = ["run_id", "train_seed", "sample_id", "subject_id", "session_id", "label", "group", "in_m1_dev", "score_attack",
                 "prediction", "detector", "file_sha256", "frame_sha256", "checkpoint_sha256", "config_sha256"]
FRAME_COLUMNS = ["run_id", "sample_id", "subject_id", "session_id", "label", "detector", "file_sha256", "frame_sha256"]
PARITY_COLUMNS = ["run_id", "train_seed", "sample_id", "label", "frame_equal", "score_stored", "score_recomputed", "abs_diff"]
STRATA_COLUMNS = ["run_id", "train_seed", "set", "subject_id", "group", "class", "n", "errors", "error_rate"]
SUBJECT_COLUMNS = N1.SUBJECT_COLUMNS
BINS = 10

# punti di iniezione (test): pre-elaborazione, risorse del rilevatore, figure (lo scorer e lo stato git sono quelli di n1_train)
PREPROCESSOR = None     # (frozen_face_path, cfg) -> funzione(path) -> (frame uint8 HxWx3, nome breve del rilevatore)
RESOURCE_PATHS = None   # {nome del file -> percorso} delle risorse del rilevatore; None: Silent-Face e OpenCV
PLOTTER = None          # (doc, strata_rows) -> {nome del file -> byte PNG}; None: matplotlib


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


def _hex(v) -> bool:
    return isinstance(v, str) and bool(HEX64.match(v))


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def config_problems(cfg: dict) -> list:
    """Schema chiuso della configurazione: chiavi, regole fisse (soglia 0,5, decisione, tolleranza 1e-6, CPU, seed N1,
    passo 1 del controllo della cache, insiemi riportati), SHA-256 ben formati, chiavi di parità valide e uniche."""
    want = {"schema", "study", "architecture", "dataset", "protocol", "n1_study", "preprocessing", "cache_check", "parity", "target",
            "reference", "scoring", "metrics", "data"}
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
    if (cfg["architecture"], cfg["dataset"], cfg["protocol"]) != ("LivenessNet", "nuaa", "official"):
        out.append("architecture, dataset or protocol differ from LivenessNet / nuaa / official")
    n1 = cfg["n1_study"]
    if not isinstance(n1, dict) or n1.get("train_seeds") != [42, 43, 44] or sorted(n1.get("seeds", {})) != ["42", "43", "44"]:
        out.append("n1_study.train_seeds must be [42, 43, 44] with one block per seed")
    else:
        for s, b in n1["seeds"].items():
            if not isinstance(b, dict) or not all(_hex(b.get(k)) for k in ("weights_sha256", "card_sha256", "scores_test_sha256")):
                out.append(f"n1_study.seeds.{s}: three SHA-256 values expected")
        if not all(_hex(n1.get(k)) for k in ("config_sha256", "check_sha256")) or not RUN_ID_N1.match(str(n1.get("run_id", ""))):
            out.append("n1_study: run_id, config_sha256 and check_sha256 expected")
        if n1.get("scores_columns") != N1.SCORE_COLUMNS:
            out.append("n1_study.scores_columns differ from the N1 score columns")
    pre = cfg["preprocessing"]
    ff = pre.get("frozen_face") if isinstance(pre, dict) else None
    if not isinstance(ff, dict) or not _hex(ff.get("sha256")) or not isinstance(ff.get("file"), str):
        out.append("preprocessing.frozen_face needs file and sha256")
    if not isinstance(pre.get("detector_resources"), dict) or not all(_hex(v) for v in pre.get("detector_resources", {}).values()):
        out.append("preprocessing.detector_resources must map file names to SHA-256 values")
    if set(pre.get("reference_code_sha256", {})) != set(REFERENCE_CODE) or not all(_hex(v) for v in pre.get("reference_code_sha256", {}).values()):
        out.append("preprocessing.reference_code_sha256 must give the SHA-256 of create_datasets.py and architectures.py")
    cc = cfg["cache_check"]
    try:
        n_cache = sum(cfg["data"]["cache"]["members"][k]["shape"][0] for k in ("X_train", "X_test"))
    except (KeyError, TypeError, IndexError):
        n_cache = None
    if not isinstance(cc, dict) or n_cache is None or cc.get("frames") != n_cache or cc.get("step") != 1:
        out.append("cache_check must cover every cache frame (X_train + X_test of data.cache.members) with step 1")
    par = cfg["parity"]
    if not isinstance(par, dict) or par.get("tolerance_abs") != TOLERANCE:
        out.append(f"parity.tolerance_abs must be {TOLERANCE}")
    if not isinstance(par, dict) or not _int(par.get("n_keys")) or par.get("n_keys") < 1 or not _hex(par.get("keys_sha256")) or "keys" in par:
        out.append("parity needs n_keys, keys_sha256 and the selection rule (no key list: identifiers stay in the restricted area)")
    t = cfg["target"]
    if not isinstance(t, dict) or not _hex(t.get("manifest_sha256")) or t.get("manifest_file") != TARGET_MANIFEST:
        out.append("target needs manifest_file target_manifest.csv and manifest_sha256")
    elif not all(_int(t.get(k)) for k in ("rows", "n_bona_fide", "n_attack", "n_subjects", "in_m1_dev", "in_c1", "in_cnn_test")) \
            or not isinstance(t.get("by_session"), dict) or t["rows"] != t["n_bona_fide"] + t["n_attack"] \
            or not all(_int(v) for v in t["by_session"].values()) or t["rows"] != sum(t["by_session"].values()) or t["in_c1"] != 0 or t["in_cnn_test"] != 0:
        out.append("target counts are not integers or not consistent (rows = classes = sessions; no C1 or test frame allowed)")
    sc = cfg["scoring"]
    if not isinstance(sc, dict) or sc.get("device") != "CPU" or sc.get("batch") != N1.SCORE_BATCH:
        out.append("scoring must be CPU with the N1 batch")
    m = cfg["metrics"]
    if not isinstance(m, dict) or m.get("threshold") != THRESHOLD or m.get("decision") != DECISION_RULE or m.get("sets") != list(SETS):
        out.append(f"metrics must fix threshold {THRESHOLD}, the decision rule and the reported sets")
    d = cfg["data"]
    if not isinstance(d, dict) or set(d) != {"m2_run", "cache", "manifests"} or set(d.get("manifests", {})) != set(MANIFESTS):
        out.append("data must hold m2_run, cache and the seven manifests")
    else:
        c = d["cache"]
        if not isinstance(c, dict) or not isinstance(c.get("file"), str) or not _hex(c.get("sha256")) or set(c.get("members", {})) != set(N1.MEMBERS) \
                or not _int(c.get("per_class")):
            out.append("data.cache needs file, sha256, per_class and the six members")
        for n, e in d["manifests"].items():
            if not isinstance(e, dict) or not isinstance(e.get("file"), str) or not _hex(e.get("sha256")) or not all(_int(e.get(f)) for f in COUNT_FIELDS):
                out.append(f"data.manifests.{n}: file, sha256 and integer counts expected")
    return out


RUN_ID_N1 = N1.RUN_ID_RE


def run_problems(cfg_sha: str, trial: bool) -> list:
    """Un run non -trial accetta solo la configurazione consegnata (SHA-256 vincolato nel codice)."""
    if trial:
        return []
    if cfg_sha != CONFIG_SHA256:
        return [f"non-trial run: the configuration SHA-256 {cfg_sha[:12]}… is not the delivered one ({CONFIG_SHA256[:12]}…)"]
    return []


# ============================================================================ cartelle e ingressi

def protected_extra() -> list:
    return [WEIGHTS, DATA] + [RESULTS / x for x in ("c1", "m1", "m2", "m2_rgb", "eval", "c2", "n1")]


def study_dir_problem(d) -> str:
    """Nome, collegamento, file, alberi con i pacchetti accettati, path_guard (dentro il repository solo results/n2/),
    cartelle protette anche spostate (pesi, dati, risultati consegnati, studio N1), in entrambe le direzioni."""
    d = Path(d)
    if not RUN_ID_RE.match(d.name):
        return f"study folder name {d.name!r} is not YYYYMMDD-N2-sessions-<shortcommit>[-trial]"
    if d.is_symlink():
        return f"{d.name} is a symbolic link: the study folder must be a real folder"
    if d.exists() and not d.is_dir():
        return f"{d.name} exists and is not a folder"
    for g in N1.outer_roots():
        if pg.within(d, g) or pg.contains(d, g):
            return f"{d} is inside, or contains, the work tree {g.name} with the accepted packages: outputs go outside it"
    if msg := pg.deletion_problem(d, ROOT, pg.BASE_PROTECTED, [ROOT / "results" / "n2"]):
        return msg
    for q in protected_extra():
        if pg.within(d, q) or (Path(q).exists() and pg.contains(d, q)):
            return f"{d} is, contains or is inside a protected folder ({q.name})"
    return ""


INPUT_OPTIONS = (("--config", "config", "file"), ("--manifests", "manifests", "dir"), ("--cache", "cache", "file"),
                 ("--images", "images", "dir"), ("--n1-study", "n1_study", "dir"), ("--frozen-face", "frozen_face", "file"))


def inputs_problem(a, d) -> str:
    """Ingressi controllati prima di qualunque modifica: esistono, sono del tipo giusto e sono separati dalla cartella
    dello studio per identità dei file (né dentro, né uguali, né antenati)."""
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


def code_hashes(a) -> dict:
    out = {f: sha256_file(ROOT / f) for f in CODE_FILES}
    for key, rel in REFERENCE_CODE.items():
        p = Path(LIVEDETECTION).joinpath(*rel)
        out[key] = sha256_file(p) if p.is_file() else "absent"
    out["frozen_face"] = sha256_file(a.frozen_face)
    return out


def resource_paths() -> dict:
    if RESOURCE_PATHS is not None:
        return dict(RESOURCE_PATHS)
    out = {n: Path(SILENT_FACE) / "resources" / "detection_model" / n for n in ("deploy.prototxt", "Widerface-RetinaFace.caffemodel")}
    try:
        import cv2
        out["haarcascade_frontalface_default.xml"] = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    except (ImportError, AttributeError):
        out["haarcascade_frontalface_default.xml"] = Path("absent")
    return out


def frozen_code_problems(cfg: dict, a) -> list:
    """face.py congelato, risorse del rilevatore e codice di riferimento: SHA-256 uguali alla configurazione."""
    out = []
    pre = cfg["preprocessing"]
    ff = Path(a.frozen_face)
    if ff.name != pre["frozen_face"]["file"]:
        out.append(f"--frozen-face {ff.name} is not {pre['frozen_face']['file']}")
    if sha256_file(ff) != pre["frozen_face"]["sha256"]:
        out.append(f"frozen face module {ff.name}: SHA-256 differs from the configuration")
    paths = resource_paths()
    for n, want in pre["detector_resources"].items():
        p = paths.get(n)
        if p is None or not Path(p).is_file():
            out.append(f"detector resource {n} not found")
        elif sha256_file(p) != want:
            out.append(f"detector resource {n}: SHA-256 differs from the configuration")
    for key, rel in REFERENCE_CODE.items():
        p = Path(LIVEDETECTION).joinpath(*rel)
        if not p.is_file():
            out.append(f"reference code {rel[-1]} not found under LIVENESSLAB_LIVEDETECTION")
        elif sha256_file(p) != pre["reference_code_sha256"][key]:
            out.append(f"reference code {rel[-1]}: SHA-256 differs from the configuration")
    return out


# ============================================================================ ingressi: manifest, cache, target, studio N1

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
        inp.rows[name], inp.sha[name] = rows, got
    inp.by_key = {n: {r["key"]: r for r in rows} for n, rows in inp.rows.items()}
    for sub, sup in (("cnn_train", "official_train"), ("cnn_es", "official_train"), ("m1_dev", "official_train"),
                     ("cnn_test", "official_test"), ("c1", "cnn_test")):
        for k, r in inp.by_key[sub].items():
            if k not in inp.by_key[sup]:
                fail(f"manifest {sub}: {sum(1 for x in inp.by_key[sub] if x not in inp.by_key[sup])} keys are not in {sup}")
            if inp.by_key[sup][k] != r:
                fail(f"manifest {sub}: rows with the same key as {sup} but different fields")
    for dim in ("key", "sha256", "subject"):
        if {r[dim] for r in inp.rows["cnn_train"]} & {r[dim] for r in inp.rows["cnn_es"]}:
            fail(f"cnn_train and cnn_es share values of {dim}")
    for dim in ("key", "sha256"):
        if {r[dim] for r in inp.rows["official_train"]} & {r[dim] for r in inp.rows["official_test"]}:
            fail(f"official train and test share values of {dim}")
    inp.checks["manifests"] = {n: {"file": d["manifests"][n]["file"], "sha256": inp.sha[n], **N1.counts_of(inp.rows[n])} for n in MANIFESTS}
    k = d["cache"]["per_class"]
    inp.order = {"train": nm.spread_keys(inp.rows["official_train"], k), "test": nm.spread_keys(inp.rows["official_test"], k)}
    if sorted(inp.order["train"]) != sorted(list(inp.by_key["cnn_train"]) + list(inp.by_key["cnn_es"])):
        fail("the sampling order of the official train list does not give cnn_train + cnn_es")
    if inp.order["test"] != nm.spread_keys(inp.rows["cnn_test"], k):
        fail("the sampling order of the official test list does not give cnn_test")
    inp.checks["sampling_order"] = "spread per class (bona fide first) on the official lists: equal to cnn_train + cnn_es and to cnn_test"


def load_cache_schema(cfg: dict, a, inp: Inputs) -> None:
    d = cfg["data"]
    cp = Path(a.cache)
    if cp.name != d["cache"]["file"]:
        fail(f"cache {cp.name} is not the file of the configuration ({d['cache']['file']})")
    inp.cache_sha = sha256_file(cp)
    if inp.cache_sha != d["cache"]["sha256"]:
        fail(f"cache {cp.name}: SHA-256 {inp.cache_sha} differs from the expected {d['cache']['sha256']}")
    mem = N1.npz_members(cp)
    if set(mem) != set(N1.MEMBERS):
        fail(f"cache {cp.name}: members {sorted(mem)} differ from {sorted(N1.MEMBERS)}")
    for k, want in d["cache"]["members"].items():
        got = {"dtype": mem[k]["dtype"], "shape": mem[k]["shape"]}
        if got != want or mem[k]["fortran"]:
            fail(f"cache {cp.name}: {k} has dtype {got['dtype']}, shape {got['shape']}, expected {want['dtype']}, {want['shape']}")
    inp.checks["cache_sha256"] = inp.cache_sha
    inp.checks["cache_schema"] = {k: {"dtype": v["dtype"], "shape": v["shape"]} for k, v in sorted(mem.items())}


def derive_target(cfg: dict, inp: Inputs) -> None:
    """I 491: official_train meno cnn_train meno cnn_es, ordinati per chiave; conteggi, disgiunzione, gruppi."""
    t = cfg["target"]
    used = set(inp.by_key["cnn_train"]) | set(inp.by_key["cnn_es"])
    rows = sorted((r for r in inp.rows["official_train"] if r["key"] not in used), key=lambda r: r["key"])
    c = N1.counts_of(rows)
    by_session = dict(sorted(Counter(r["session"] for r in rows).items()))
    got = {**c, "by_session": by_session, "in_m1_dev": sum(1 for r in rows if r["key"] in inp.by_key["m1_dev"]),
           "in_c1": sum(1 for r in rows if r["key"] in inp.by_key["c1"]), "in_cnn_test": sum(1 for r in rows if r["key"] in inp.by_key["cnn_test"])}
    for k, v in got.items():
        if t[k] != v:
            fail(f"target {k}: {v} differs from the configuration ({t[k]})")
    fit = {r["subject"] for r in inp.rows["cnn_train"]}
    val = {r["subject"] for r in inp.rows["cnn_es"]}
    if {r["subject"] for r in rows} - (fit | val):      # difensivo: un soggetto solo nel target non avrebbe gruppo (load_manifests ha già escluso fit & val)
        fail("target subjects outside the fitting and validation subjects")
    inp.target = rows
    inp.group = {s: "fitting" for s in fit} | {s: "validation" for s in val}
    inp.m1_dev_keys = set(inp.by_key["m1_dev"])
    inp.target_text = N1.csv_text(rows, nm.COLUMNS)
    inp.target_sha = sha256_text(inp.target_text)
    if inp.target_sha != t["manifest_sha256"]:
        fail(f"target manifest: SHA-256 {inp.target_sha} differs from the configuration ({t['manifest_sha256']})")
    inp.checks["target"] = {**got, "manifest_sha256": inp.target_sha, "n_fitting_subjects": len(fit), "n_validation_subjects": len(val),
                            "n_frames_fitting_subjects": sum(1 for r in rows if r["subject"] in fit),
                            "n_frames_validation_subjects": sum(1 for r in rows if r["subject"] in val),
                            "disjoint_from": "cnn_train and cnn_es by construction (keys); cnn_test and c1 because the official train and test "
                                             "lists share no key and no file SHA-256 and cnn_test is inside official_test (checked above)"}


LAYOUTS = {"images": lambda r: f"{'bonafide' if r['label'] == '0' else 'attack'}_{'training' if r['official_split'] == 'train' else 'validation'}/{nm.name_of(r['key'])}",
           "raw": lambda r: f"{'ClientRaw' if r['label'] == '0' else 'ImposterRaw'}/{r['subject']}/{nm.name_of(r['key']).split('_', 1)[1]}"}


def locate(root: Path, r: dict, layouts: Counter = None) -> Path:
    """Il file originale di una riga del manifest: cercato nelle due disposizioni note, accettato solo se file regolare
    con SHA-256 e dimensione uguali alla riga (la provenienza è l'hash, non il percorso)."""
    for name, rel in LAYOUTS.items():
        p = Path(root) / rel(r)
        if p.is_symlink():
            fail(f"original of {r['key']}: symbolic link refused")
        if p.is_file():
            if p.stat().st_size != int(r["bytes"]) or sha256_file(p) != r["sha256"]:
                fail(f"original of {r['key']}: size or SHA-256 differ from the manifest")
            if layouts is not None:
                layouts[name] += 1
            return p
    fail(f"original of {r['key']} not found under the images folder")


def check_originals(inp: Inputs, a) -> None:
    layouts = Counter()
    for r in inp.target:
        locate(Path(a.images), r, layouts)
    inp.checks["originals"] = {"n_verified": len(inp.target), "by_layout": dict(sorted(layouts.items())),
                               "rule": "file regular, size and SHA-256 equal to the manifest row"}


def n1_scores_path(a, seed: int, cfg: dict) -> Path:
    return Path(a.n1_study) / f"seed{seed}" / cfg["n1_study"]["scores_file"]


def load_n1_study(cfg: dict, a, inp: Inputs) -> None:
    """Studio N1: check.json, train.json, score.json, pesi e punteggi legati per SHA-256 alla configurazione; le righe dei
    punteggi coincidono con cnn_test riga per riga (chiave, soggetto, sessione, etichetta, SHA-256 del file, checkpoint)."""
    n1 = cfg["n1_study"]
    d = Path(a.n1_study)
    if d.name != n1["run_id"]:
        fail(f"--n1-study {d.name} is not the N1 study {n1['run_id']}")
    ck = d / "check.json"
    if ck.is_symlink() or not ck.is_file() or sha256_file(ck) != n1["check_sha256"]:
        fail("N1 check.json missing or SHA-256 differs from the configuration")
    N1.load_record(ck, "check", n1["run_id"], n1["config_sha256"])
    inp.n1 = {}
    for seed in n1["train_seeds"]:
        b = n1["seeds"][str(seed)]
        sd = d / f"seed{seed}"
        train = N1.load_record(sd / "train.json", "train", n1["run_id"], n1["config_sha256"], n1["check_sha256"])
        if train.get("train_seed") != seed or train.get("weights_sha256") != b["weights_sha256"] or train.get("card_sha256") != b["card_sha256"]:
            fail(f"N1 seed{seed}/train.json: seed, weights or card SHA-256 differ from the configuration")
        w = sd / n1["weights_file"]
        if w.is_symlink() or not w.is_file():
            fail(f"N1 seed{seed} weights {w.name} missing")
        if sha256_file(w) != b["weights_sha256"]:
            fail(f"N1 seed{seed} weights {w.name}: SHA-256 differs from the configuration")
        score = N1.load_record(sd / "score.json", "score", n1["run_id"], n1["config_sha256"], n1["check_sha256"])
        if score.get("train_seed") != seed or score.get("weights_sha256") != b["weights_sha256"] \
                or score.get("files", {}).get(n1["scores_file"]) != b["scores_test_sha256"]:
            fail(f"N1 seed{seed}/score.json: seed, weights or scores SHA-256 differ from the configuration")
        sp = n1_scores_path(a, seed, cfg)
        if sp.is_symlink() or not sp.is_file() or sha256_file(sp) != b["scores_test_sha256"]:
            fail(f"N1 seed{seed}/{sp.name}: missing or SHA-256 differs from the configuration")
        rows = N1.read_csv_rows(sp, n1["scores_columns"])
        keys = [r["sample_id"] for r in rows]
        if sorted(keys) != sorted(inp.by_key["cnn_test"]) or len(set(keys)) != len(keys):
            fail(f"N1 seed{seed} scores do not hold exactly the cnn_test rows")
        for r in rows:
            m = inp.by_key["cnn_test"][r["sample_id"]]
            try:
                v = float(r["score_attack"])
            except ValueError:
                v = float("nan")
            if r["run_id"] != n1["run_id"] or r["train_seed"] != str(seed) or r["subject_id"] != m["subject"] or r["session_id"] != m["session"] \
                    or r["label"] != m["label"] or r["file_sha256"] != m["sha256"] or r["checkpoint_sha256"] != b["weights_sha256"] \
                    or r["config_sha256"] != n1["config_sha256"] or r["in_c1_subset"] != ("1" if r["sample_id"] in inp.by_key["c1"] else "0") \
                    or not (0 <= v <= 1) or r["prediction"] != ("1" if v > THRESHOLD else "0"):
                fail(f"N1 seed{seed} scores: rows inconsistent with cnn_test, the C1 subset, the checkpoint or the decision rule")
        inp.n1[seed] = {"weights": w, "weights_sha256": b["weights_sha256"], "scores": {r["sample_id"]: float(r["score_attack"]) for r in rows},
                        "rows": rows}
    inp.checks["n1_study"] = {"run_id": n1["run_id"], "check_sha256": n1["check_sha256"], "config_sha256": n1["config_sha256"],
                              "seeds": {str(s): {"weights_sha256": inp.n1[s]["weights_sha256"], "n_scores": len(inp.n1[s]["rows"])} for s in inp.n1},
                              "linkage": "train.json, score.json, weights and scores_test.csv bound by SHA-256; every score row equal to its cnn_test row"}


def load_inputs(cfg: dict, a, originals: bool = True) -> Inputs:
    """Tutti i controlli di check (nessun rilevatore, nessun TensorFlow, nessun pixel della cache)."""
    inp = Inputs()
    inp.checks = {}
    load_cache_schema(cfg, a, inp)
    load_manifests(cfg, a, inp)
    derive_target(cfg, inp)
    if originals:
        check_originals(inp, a)
    load_n1_study(cfg, a, inp)
    if problems := frozen_code_problems(cfg, a):
        fail("frozen code check failed: " + "; ".join(problems))
    inp.checks["frozen_code"] = {"frozen_face": cfg["preprocessing"]["frozen_face"], "detector_resources": cfg["preprocessing"]["detector_resources"],
                                 "reference_code_sha256": cfg["preprocessing"]["reference_code_sha256"]}
    inp.parity_keys = parity_keys(inp.rows["cnn_test"])
    inp.parity_sha = sha256_text("\n".join(inp.parity_keys) + "\n")
    par = cfg["parity"]
    if len(inp.parity_keys) != par["n_keys"] or inp.parity_sha != par["keys_sha256"]:
        fail(f"parity list derived from cnn_test: {len(inp.parity_keys)} keys, SHA-256 {inp.parity_sha[:12]}…; the configuration fixes "
             f"{par['n_keys']} keys and {par['keys_sha256'][:12]}…")
    inp.parity_text = N1.csv_text([{"sample_id": k, "subject_id": inp.by_key["cnn_test"][k]["subject"], "label": inp.by_key["cnn_test"][k]["label"]}
                                   for k in inp.parity_keys], PARITY_KEY_COLUMNS)
    inp.checks["parity_keys"] = {"n": len(inp.parity_keys), "keys_sha256": inp.parity_sha, "selection": "first bona fide and first attack key per test "
                                 "subject, key order", "tolerance_abs": par["tolerance_abs"]}
    return inp


def parity_keys(rows) -> list:
    """La lista di parità derivata dalla regola: per ogni soggetto del test, la prima chiave bona fide e la prima chiave di
    attacco in ordine di chiave (ordine per soggetto e classe)."""
    by = {}
    for r in sorted(rows, key=lambda r: r["key"]):
        by.setdefault((r["subject"], r["label"]), r["key"])
    return [by[k] for k in sorted(by)]


def same_inputs(inp: Inputs, check: dict) -> None:
    for k in ("cache_sha256", "manifests", "target", "n1_study", "frozen_code", "parity_keys"):
        if inp.checks.get(k) != check["checks"].get(k):
            fail(f"{k} differ from those recorded at check")


# ============================================================================ passi comuni

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


def verify_files(d: Path, doc: dict, step: str, record: str) -> None:
    """I file elencati nel record di un passo devono essere ancora quelli (file regolari, stesso SHA-256)."""
    for n, h in doc.get("files", {}).items():
        q = d / n
        if q.is_symlink() or not q.is_file() or sha256_file(q) != h:
            fail(f"{n} changed after {step} (SHA-256 differs from {record})")


def open_study(a) -> tuple:
    d, cfg, cfg_sha = parse_common(a)
    check = N1.load_record(d / "check.json", "check", d.name, cfg_sha)
    verify_files(d, check, "check", "check.json")
    if check.get("code_sha256") != code_hashes(a):
        fail("the code changed since check (SHA-256 of the scripts differ from check.json): start a new study")
    if not is_trial(d) and N1.repo_state(d).get("commit") != check.get("commit"):
        fail("non-trial run: the repository commit differs from check")
    return d, cfg, cfg_sha, check, sha256_file(d / "check.json")


def refuse_existing(d: Path, names, step: str) -> None:
    found = [n for n in names if os.path.lexists(d / n)]
    if found:
        fail(f"{d.name} already holds {', '.join(found)}: {step} outputs are never recomputed in place (refused, nothing changed)")


def load_step(d: Path, name: str, step: str, run_id: str, cfg_sha: str, check_sha: str) -> dict:
    """Record di un passo precedente dello studio; i file che elenca devono essere ancora quelli (SHA-256)."""
    doc = N1.load_record(d / name, step, run_id, cfg_sha, check_sha)
    verify_files(d, doc, step, name)
    return doc


def write_binary(d, files: dict) -> None:
    """Come write_all, per file binari (figure): temporanei e rinomine, nulla resta su errore."""
    d = Path(d)
    tmps, done = {}, []
    try:
        for n in files:
            q = d / n
            if q.is_symlink() or (q.exists() and not q.is_file()):
                raise N2Error(f"{n} is a link or not a regular file: refused")
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


# ============================================================================ pre-elaborazione congelata

def load_frozen_detector(path, sha: str):
    p = Path(path)
    if sha256_file(p) != sha:
        fail(f"frozen face module {p.name}: SHA-256 changed just before loading")
    name = "tesi_app.face_0ac7143"
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, str(p))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod.FaceDetector()


class RecordingDetector:
    """Il rilevatore congelato, con l'ultimo esito registrato (per sapere quale rilevatore ha trovato il volto)."""

    def __init__(self, det):
        self.det, self.last = det, None

    def detect(self, img, *args, **kwargs):
        self.last = self.det.detect(img, *args, **kwargs)
        return self.last


def detector_short(face) -> str:
    """Nome breve del rilevatore ("RetinaFace", "Haar") dalla descrizione del volto; "none" senza volto (immagine intera)."""
    return "none" if face is None else str(face.detector).split()[0]


def default_preprocessor(frozen_face, cfg: dict):
    """train_cnn.preprocess (invariato) con il rilevatore congelato e l'enhancement del codice di riferimento."""
    try:
        import train_cnn as tc
    except ModuleNotFoundError as e:
        fail(f"missing Python package {e.name!r}: the preprocessing needs OpenCV")
    det = RecordingDetector(load_frozen_detector(frozen_face, cfg["preprocessing"]["frozen_face"]["sha256"]))
    cd = load_module("kuznetsov_create_datasets", Path(LIVEDETECTION) / "scripts" / "create_datasets.py")
    creator = cd.EnhancedDatasetCreator.__new__(cd.EnhancedDatasetCreator)   # istanza vuota: serve solo advanced_image_enhancement

    def pre(path):
        det.last = None
        arr = tc.preprocess(Path(path), det, creator)
        if arr is None:
            fail(f"{Path(path).name}: unreadable or empty crop")
        return arr, detector_short(det.last)
    return pre


def preprocessor(a, cfg: dict):
    return (PREPROCESSOR or default_preprocessor)(a.frozen_face, cfg)


def frame_hash(arr) -> str:
    return nm.frame_hashes(np.asarray(arr)[None])[0]


def tf_state() -> str:
    """Per i passi che non devono usare TensorFlow: lo stato è accertato, non dichiarato."""
    if "tensorflow" in sys.modules:
        fail("TensorFlow was imported in a step that must not use it")
    return "not imported"


def environment() -> dict:
    out = N1.device_info()
    try:
        import cv2
        out["opencv"] = str(cv2.__version__)
    except (ImportError, AttributeError):      # assente, oppure il modulo vuoto registrato dalle suite di test
        out["opencv"] = "absent"
    return out


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
    print(f"== N2 check: study {d.name}, configuration SHA-256 {cfg_sha}")
    inp = load_inputs(cfg, a)
    for k, v in inp.checks.items():
        print(f"   PASS {k}: {json.dumps(v, sort_keys=True) if not isinstance(v, str) else v}")
    doc = {"passed": True, "step": "check", "run_id": d.name, "trial": is_trial(d), "config_sha256": cfg_sha,
           "delivered_configuration": cfg_sha == CONFIG_SHA256, "checks": inp.checks, "detector": "not used", "weights": "not loaded",
           "tensorflow": tf_state(), "code_sha256": code_hashes(a), "files": {TARGET_MANIFEST: inp.target_sha, PARITY_KEYS: sha256_text(inp.parity_text)},
           "environment": environment(), **state, "created": now()}
    write_all(d, {TARGET_MANIFEST: inp.target_text, PARITY_KEYS: inp.parity_text, "check.json": json.dumps(doc, indent=1) + "\n"})
    t = inp.checks["target"]
    print(f"== check PASS: cache, 7 manifests, target {t['rows']} frames ({t['n_bona_fide']} bona fide, {t['n_attack']} attacks, "
          f"{t['n_subjects']} subjects, {t['in_m1_dev']} in the M1 dev set), originals verified, N1 study bound, frozen code verified")


# ============================================================================ cache-check

def cmd_cache_check(a) -> None:
    d, cfg, cfg_sha, check, check_sha = open_study(a)
    if a.step != 1 and not is_trial(d):
        fail("--step is allowed only in -trial runs: the delivered check covers every frame")
    if a.step < 1:
        fail("--step must be a positive integer")
    refuse_existing(d, CACHE_OUTPUTS, "cache-check")
    N1.accept(d, "cache-check", "cache_check.log")
    print(f"== N2 cache-check: study {d.name}, frozen detector, step {a.step}")
    inp = load_inputs(cfg, a, originals=False)
    same_inputs(inp, check)
    pre = preprocessor(a, cfg)
    z = np.load(a.cache, allow_pickle=False)
    result, cache_hashes = {}, {}
    for split, src in (("train", "official_train"), ("test", "official_test")):
        X = z[f"X_{split}"]                      # len(X) = chiavi in ordine di campionamento: lo schema della cache è verificato al check
        hashes = nm.frame_hashes(X)
        cache_hashes[split] = hashes
        equal = different = 0
        worst = 0
        dets = Counter()
        for i in range(0, len(X), a.step):
            r = inp.by_key[src][inp.order[split][i]]
            arr, det = pre(locate(Path(a.images), r))
            dets[det] += 1
            if arr.shape != X[i].shape or arr.dtype != X[i].dtype:
                fail(f"X_{split}[{i}]: reproduced frame has shape {arr.shape} {arr.dtype}, cache {X[i].shape} {X[i].dtype}")
            diff = int(np.abs(arr.astype(np.int16) - X[i].astype(np.int16)).max())
            if diff == 0:
                equal += 1
            else:
                different += 1
                worst = max(worst, diff)
                print(f"   DIFF X_{split}[{i}] {r['key']}: max abs difference {diff}")
        result[split] = {"compared": equal + different, "equal": equal, "different": different, "worst_max_abs_diff": worst,
                         "detectors": dict(sorted(dets.items())), "duplicates_within": len(hashes) - len(set(hashes))}
        print(f"   X_{split}: {equal} equal, {different} different (step {a.step}), detectors {dict(sorted(dets.items()))}")
        del X
    frames_rows, dets = [], Counter()
    for r in inp.target:
        arr, det = pre(locate(Path(a.images), r))
        dets[det] += 1
        frames_rows.append({"run_id": d.name, "sample_id": r["key"], "subject_id": r["subject"], "session_id": r["session"], "label": r["label"],
                            "detector": det, "file_sha256": r["sha256"], "frame_sha256": frame_hash(arr)})
    th = [x["frame_sha256"] for x in frames_rows]
    cache_all = set(cache_hashes["train"]) | set(cache_hashes["test"])
    target = {"n": len(frames_rows), "detectors": dict(sorted(dets.items())), "duplicates_within": len(th) - len(set(th)),
              "shared_with_cache": sum(1 for h in th if h in cache_all),
              "cache_train_test_shared": len(set(cache_hashes["train"]) & set(cache_hashes["test"]))}
    print(f"   target: {target['n']} frames, detectors {target['detectors']}, duplicates within {target['duplicates_within']}, "
          f"shared with the cache {target['shared_with_cache']}")
    total_diff = sum(result[s]["different"] for s in result)
    compared = sum(result[s]["compared"] for s in result)
    n_frames = sum(len(cache_hashes[s]) for s in cache_hashes)      # = cache_check.frames: lo schema della cache è già verificato
    dup = {k: v for k, v in {**{f"cache_{s}_duplicates": result[s]["duplicates_within"] for s in result},
                             "cache_train_test_shared": target["cache_train_test_shared"], "target_duplicates": target["duplicates_within"],
                             "target_shared_with_cache": target["shared_with_cache"]}.items() if v}
    if total_diff or (a.step == 1 and compared != n_frames):
        fail(f"cache check: {total_diff} frames differ from the cache ({compared} compared): the frozen preprocessing does not reproduce the cache")
    if dup:
        fail(f"duplicated derived frames: {dup} (exact pixel hash)")
    frames_txt = N1.csv_text(frames_rows, FRAME_COLUMNS)
    doc = {"passed": True, "step": "cache-check", "run_id": d.name, "config_sha256": cfg_sha, "check_sha256": check_sha, "step_size": a.step,
           "rule": cfg["cache_check"]["rule"], "cache": result, "target_frames": target, "frozen_face_sha256": cfg["preprocessing"]["frozen_face"]["sha256"],
           "tensorflow": tf_state(), "environment": environment(), "files": {FRAMES_TARGET: sha256_text(frames_txt)}, "created": now()}
    write_all(d, {FRAMES_TARGET: frames_txt, "cache_check.json": json.dumps(doc, indent=1) + "\n"})
    print(f"== cache-check PASS: {compared} cache frames reproduced bit for bit (step {a.step}), {target['n']} target frames preprocessed, "
          "no duplicated derived frame")


# ============================================================================ parity

def scorer_for(inp: Inputs, seed: int):
    return (N1.MAKE_SCORER or N1.tf_scorer)(inp.n1[seed]["weights"], inp.n1[seed]["weights_sha256"])


def cmd_parity(a) -> None:
    d, cfg, cfg_sha, check, check_sha = open_study(a)
    refuse_existing(d, PARITY_OUTPUTS, "parity")
    load_step(d, "cache_check.json", "cache-check", d.name, cfg_sha, check_sha)
    cache_check_sha = sha256_file(d / "cache_check.json")
    N1.accept(d, "parity", "parity.log")
    print(f"== N2 parity: study {d.name}, {cfg['parity']['n_keys']} N1 keys, tolerance {cfg['parity']['tolerance_abs']}")
    inp = load_inputs(cfg, a, originals=False)
    same_inputs(inp, check)
    pre = preprocessor(a, cfg)
    z = np.load(a.cache, allow_pickle=False)
    X = z["X_test"]
    pos = {k: i for i, k in enumerate(inp.order["test"])}
    frames, equal = [], []
    for k in inp.parity_keys:
        r = inp.by_key["cnn_test"][k]
        arr, _ = pre(locate(Path(a.images), r))
        same = arr.shape == X[pos[k]].shape and bool(np.array_equal(arr, X[pos[k]]))
        equal.append(same)
        frames.append(X[pos[k]])
    del X
    if not all(equal):
        fail(f"parity: {sum(1 for e in equal if not e)} of {len(equal)} frames differ from the cache rows")
    frames = np.stack(frames)
    tol = cfg["parity"]["tolerance_abs"]
    rows, seeds = [], {}
    for seed in cfg["n1_study"]["train_seeds"]:
        p = N1.probabilities(scorer_for(inp, seed), frames)[:, 1]
        diffs = []
        for k, e, v in zip(inp.parity_keys, equal, p):
            stored = inp.n1[seed]["scores"][k]
            diffs.append(abs(float(v) - stored))
            rows.append({"run_id": d.name, "train_seed": str(seed), "sample_id": k, "label": inp.by_key["cnn_test"][k]["label"],
                         "frame_equal": "1" if e else "0", "score_stored": repr(stored), "score_recomputed": repr(float(v)), "abs_diff": repr(diffs[-1])})
        seeds[str(seed)] = {"n_keys": len(diffs), "max_abs_diff": max(diffs), "within_tolerance": sum(1 for x in diffs if x <= tol)}
        print(f"   seed {seed}: max |recomputed - stored| {max(diffs):.3e} over {len(diffs)} keys")
    bad = {s: v for s, v in seeds.items() if v["within_tolerance"] != v["n_keys"]}
    if bad:
        fail(f"parity: scores outside the tolerance {tol} for seeds {sorted(bad)}: resolve before any comparison")
    par_txt = N1.csv_text(rows, PARITY_COLUMNS)
    doc = {"passed": True, "step": "parity", "run_id": d.name, "config_sha256": cfg_sha, "check_sha256": check_sha, "cache_check_sha256": cache_check_sha,
           "rule": cfg["parity"]["rule"],
           "tolerance_abs": tol, "n_keys": len(inp.parity_keys), "frames_equal": sum(equal), "seeds": seeds, "device": "CPU",
           "environment": environment(), "files": {"parity.csv": sha256_text(par_txt)}, "created": now()}
    write_all(d, {"parity.csv": par_txt, "parity.json": json.dumps(doc, indent=1) + "\n"})
    print(f"== parity PASS: {len(inp.parity_keys)} frames equal to the cache, scores within {tol} for every seed")


# ============================================================================ score

def set_rows(rows):
    """Sottoinsiemi dei 491 per gli insiemi riportati (dalle colonne delle righe)."""
    return {"sessions_01_02_491": rows,
            "sessions_01_02_fitting_subjects": [r for r in rows if r["group"] == "fitting"],
            "sessions_01_02_validation_subjects": [r for r in rows if r["group"] == "validation"],
            "sessions_01_02_not_in_m1_dev": [r for r in rows if r["in_m1_dev"] == "0"]}


def metrics_from_rows(rows) -> dict:
    return {name: N1.set_metrics([int(r["label"]) for r in sel], [float(r["score_attack"]) for r in sel]) for name, sel in set_rows(rows).items()}


def cmd_score(a) -> None:
    d, cfg, cfg_sha, check, check_sha = open_study(a)
    if a.train_seed not in cfg["n1_study"]["train_seeds"]:
        fail(f"--train-seed {a.train_seed} is not one of the N1 seeds {cfg['n1_study']['train_seeds']}")
    seed = a.train_seed
    if not a.i_am_authorized:
        fail("score needs --i-am-authorized (inference on the restricted NUAA data)")
    cache_rec = load_step(d, "cache_check.json", "cache-check", d.name, cfg_sha, check_sha)
    par = load_step(d, "parity.json", "parity", d.name, cfg_sha, check_sha)
    if par.get("cache_check_sha256") != sha256_file(d / "cache_check.json"):
        fail("parity.json does not belong to this cache_check.json (SHA-256 differs): run cache-check and parity again in a new study")
    sd = d / f"seed{seed}"
    if os.path.lexists(sd):
        fail(f"{sd.name} already exists: a seed folder is never reused (refused, nothing changed)")
    inp = load_inputs(cfg, a, originals=False)
    same_inputs(inp, check)
    expected = {r["sample_id"]: r for r in N1.read_csv_rows(d / FRAMES_TARGET, FRAME_COLUMNS)}
    if sorted(expected) != [r["key"] for r in inp.target]:
        fail(f"{FRAMES_TARGET} does not hold exactly the target keys")
    sd.mkdir()
    N1.accept(sd, "score", "score.log")
    print(f"== N2 score: study {d.name}, N1 seed {seed}, CPU, frozen detector")
    pre = preprocessor(a, cfg)
    frames, dets = [], []
    for r in inp.target:
        arr, det = pre(locate(Path(a.images), r))
        h = frame_hash(arr)
        if h != expected[r["key"]]["frame_sha256"] or det != expected[r["key"]]["detector"]:
            fail(f"{r['key']}: preprocessed frame or detector differ from cache-check")
        frames.append(arr)
        dets.append(det)
    frames = np.stack(frames)
    p = N1.probabilities(scorer_for(inp, seed), frames)[:, 1]
    rows, values = [], []
    for r, det, v in zip(inp.target, dets, p):
        v = float(v)
        values.append(v)
        rows.append({"run_id": d.name, "train_seed": str(seed), "sample_id": r["key"], "subject_id": r["subject"], "session_id": r["session"],
                     "label": r["label"], "group": inp.group[r["subject"]], "in_m1_dev": "1" if r["key"] in inp.m1_dev_keys else "0",
                     "score_attack": repr(v), "prediction": "1" if v > THRESHOLD else "0", "detector": det, "file_sha256": r["sha256"],
                     "frame_sha256": expected[r["key"]]["frame_sha256"], "checkpoint_sha256": inp.n1[seed]["weights_sha256"], "config_sha256": cfg_sha})
    scores_txt = N1.csv_text(rows, SCORE_COLUMNS)
    back = list(csv.DictReader(io.StringIO(scores_txt)))
    if len(back) != len(values) or any(float(b["score_attack"]) != v for b, v in zip(back, values)):
        fail("scores do not survive the CSV round trip")
    met = metrics_from_rows(back)
    subj = [{"run_id": d.name, "train_seed": str(seed), **x} for x in N1.per_subject(back, "sessions_01_02_491")]
    subj_txt = N1.csv_text(subj, SUBJECT_COLUMNS)
    titles = set_titles(check["checks"]["target"], len(inp.rows["cnn_test"]))
    met_doc = {"run_id": d.name, "train_seed": seed, "sets": {k: titles[k] for k in TARGET_SETS}, "metrics": met, "weights_sha256": inp.n1[seed]["weights_sha256"]}
    met_txt = json.dumps(met_doc, indent=1) + "\n"
    doc = {"passed": True, "step": "score", "run_id": d.name, "train_seed": seed, "config_sha256": cfg_sha, "check_sha256": check_sha,
           "cache_check_sha256": sha256_file(d / "cache_check.json"), "weights_sha256": inp.n1[seed]["weights_sha256"], "device": "CPU",
           "n_scores": len(rows), "detectors": dict(sorted(Counter(dets).items())), "environment": environment(),
           "files": {"scores_sessions.csv": sha256_text(scores_txt), "per_subject.csv": sha256_text(subj_txt), "metrics.json": sha256_text(met_txt)},
           "created": now()}
    write_all(sd, {"scores_sessions.csv": scores_txt, "per_subject.csv": subj_txt, "metrics.json": met_txt, "score.json": json.dumps(doc, indent=1) + "\n"})
    for name in TARGET_SETS:
        m = met[name]
        print(f"   {name}: APCER {N1.fmt_na(m['apcer'])} ({m['accepted_attacks']}/{m['n_attack']}), BPCER {N1.fmt_na(m['bpcer'])} "
              f"({m['rejected_bona_fide']}/{m['n_bona_fide']}), ACER {N1.fmt_na(m['acer'])}, AUROC {N1.fmt_na(m['auroc'])}")
    print(f"== score PASS: seed {seed}, {len(rows)} scores of the sessions 01-02 target, cache_check {cache_rec['target_frames']['n']} frames matched")


# ============================================================================ aggregate

def reference_rows(inp: Inputs, seed: int) -> dict:
    """Insiemi della sessione 03 dai punteggi N1 (righe così come salvate, con il gruppo del soggetto)."""
    rows = [{**r, "group": inp.group.get(r["subject_id"], "other")} for r in inp.n1[seed]["rows"]]
    nine = [r for r in rows if r["group"] != "other"]
    return {"session_03_3000": rows, "session_03_9_subjects": nine,
            "session_03_fitting_subjects": [r for r in nine if r["group"] == "fitting"],
            "session_03_validation_subjects": [r for r in nine if r["group"] == "validation"]}


def histogram(rows) -> dict:
    """Conteggi di score_attack per classe in BINS intervalli di uguale larghezza su [0, 1] (l'ultimo chiuso a destra)."""
    out = {}
    for lab, name in (("0", "bona_fide"), ("1", "attack")):
        s = np.array([float(r["score_attack"]) for r in rows if r["label"] == lab], dtype=float)
        counts, _ = np.histogram(s, bins=BINS, range=(0.0, 1.0)) if len(s) else (np.zeros(BINS, int), None)
        out[name] = [int(c) for c in counts]
    out["bin_edges"] = [round(i / BINS, 2) for i in range(BINS + 1)]
    return out


def strata_rows(run_id: str, seed: int, set_name: str, rows, group: dict) -> list:
    """Strati soggetto x classe di un insieme: n, errori alla soglia e tasso (BPCER per i bona fide, APCER per gli attacchi), NA se vuoto."""
    out = []
    subjects = sorted(group)
    for subj in subjects:
        for lab, cls in (("0", "bona_fide"), ("1", "attack")):
            sel = [r for r in rows if r["subject_id"] == subj and r["label"] == lab]
            err = sum(1 for r in sel if r["prediction"] == ("1" if lab == "0" else "0"))
            out.append({"run_id": run_id, "train_seed": str(seed), "set": set_name, "subject_id": subj, "group": group[subj], "class": cls,
                        "n": str(len(sel)), "errors": str(err), "error_rate": N1.fmt_na(err / len(sel) if sel else None)})
    return out


def strata_summary(target_strata: list, reference_strata: list) -> dict:
    """Confronto pubblico degli strati (nessun identificativo): per gruppo e classe, quanti strati sono calcolabili in
    entrambi gli insiemi e in quanti il tasso di errore delle sessioni 01-02 è minore, uguale o maggiore di quello della
    sessione 03."""
    ref = {(r["subject_id"], r["class"]): r for r in reference_strata}
    out = {}
    for grp in ("fitting", "validation"):
        for cls in ("bona_fide", "attack"):
            sel = [r for r in target_strata if r["group"] == grp and r["class"] == cls]
            cmp = Counter()
            for r in sel:
                q = ref.get((r["subject_id"], r["class"]))
                if r["error_rate"] == NA or q is None or q["error_rate"] == NA:
                    cmp["not_comparable"] += 1
                else:
                    a, b = float(r["error_rate"]), float(q["error_rate"])
                    cmp["lower_in_01_02" if a < b else "equal" if a == b else "higher_in_01_02"] += 1
            out[f"{grp}/{cls}"] = {"strata": len(sel), **{k: cmp.get(k, 0) for k in ("lower_in_01_02", "equal", "higher_in_01_02", "not_comparable")}}
    return out


def default_plotter(doc: dict, strata: list) -> dict:
    """Figure: le due pubbliche dal solo `doc` (aggregate.json, già scansionato: nessun identificativo, nessuna riga per
    immagine); quella per soggetto dagli strati (riservata)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        fail(f"missing Python package {e.name!r}: the figures need matplotlib")
    seeds = [str(s) for s in doc["train_seeds"]]
    out = {}
    fig, axes = plt.subplots(len(seeds), 2, figsize=(10, 3.2 * len(seeds)), squeeze=False)
    edges = doc["seeds"][seeds[0]]["distributions"]["sessions_01_02_491"]["bin_edges"]
    width = (edges[1] - edges[0]) / 2.2
    for i, s in enumerate(seeds):
        for j, (name, title) in enumerate((("sessions_01_02_491", "sessions 01-02 (491 frames)"), ("session_03_3000", "session 03 (3,000 frames)"))):
            h = doc["seeds"][s]["distributions"][name]
            ax = axes[i][j]
            for k, (cls, off, col) in enumerate((("bona_fide", 0, "#2b6cb0"), ("attack", width, "#c05621"))):
                tot = max(1, sum(h[cls]))
                ax.bar([e + off for e in edges[:-1]], [c / tot for c in h[cls]], width=width, align="edge", color=col,
                       label=f"{cls.replace('_', ' ')} (n={sum(h[cls])})")
            ax.set_title(f"seed {s}: {title}")
            ax.set_xlabel("attack score")
            ax.set_ylabel("fraction of the class")
            ax.set_xlim(0, 1)
            ax.axvline(THRESHOLD, color="k", linestyle=":", linewidth=1)
            ax.legend(fontsize=8)
    fig.suptitle("Attack-score distributions per class (threshold 0.5 dotted)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    out["figure_distributions.png"] = buf.getvalue()
    # figura pubblica 2: tassi di errore per gruppo e classe, sessioni 01-02 contro sessione 03, per seed (dall'aggregato)
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.4 * len(seeds), 4.0), squeeze=False)
    groups = (("fitting", "sessions_01_02_fitting_subjects", "session_03_fitting_subjects"),
              ("validation", "sessions_01_02_validation_subjects", "session_03_validation_subjects"))
    for i, s in enumerate(seeds):
        ax = axes[0][i]
        labels, a_vals, b_vals = [], [], []
        for grp, ta, tb in groups:
            for cls, met in (("bona fide", "bpcer"), ("attack", "apcer")):
                labels.append(f"{grp}\n{cls}")
                a_vals.append(doc["seeds"][s]["metrics"][ta][met])
                b_vals.append(doc["seeds"][s]["metrics"][tb][met])
        xs = list(range(len(labels)))
        for off, vals, col, lab in ((-0.2, a_vals, "#2b6cb0", "sessions 01-02"), (0.2, b_vals, "#c05621", "session 03")):
            ax.bar([x + off for x in xs], [0 if v is None else v for v in vals], width=0.4, color=col, label=lab)
            for x, v in zip(xs, vals):
                ax.text(x + off, (0 if v is None else v) + 0.02, "NA" if v is None else f"{v:.3f}", ha="center", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("error rate at 0.5 (BPCER / APCER)")
        ax.set_title(f"seed {s}")
        ax.legend(fontsize=8)
    fig.suptitle("Error rate per group of subjects and class: sessions 01-02 (target) against session 03 (N1 test)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    out["figure_groups.png"] = buf.getvalue()
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.2 * len(seeds), 4.2), squeeze=False)
    for i, s in enumerate(seeds):
        ax = axes[0][i]
        ref = {(r["subject_id"], r["class"]): r for r in strata if r["train_seed"] == s and r["set"] == "session_03_9_subjects"}
        for grp, col in (("fitting", "#2b6cb0"), ("validation", "#c05621")):
            for cls, mk in (("bona_fide", "o"), ("attack", "^")):
                xs, ys = [], []
                for r in strata:
                    q = ref.get((r["subject_id"], r["class"]))
                    if r["train_seed"] == s and r["set"] == "sessions_01_02_491" and r["group"] == grp and r["class"] == cls \
                            and r["error_rate"] != NA and q is not None and q["error_rate"] != NA:
                        xs.append(float(q["error_rate"]))
                        ys.append(float(r["error_rate"]))
                if xs:
                    ax.scatter(xs, ys, marker=mk, color=col, label=f"{grp} subjects, {cls.replace('_', ' ')} ({len(xs)})", alpha=0.8)
        ax.plot([0, 1], [0, 1], color="grey", linewidth=0.8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("error rate, session 03 (N1 test)")
        ax.set_ylabel("error rate, sessions 01-02")
        ax.set_title(f"seed {s}")
        ax.legend(fontsize=7)
    fig.suptitle("Subject x class strata: error rate at 0.5 (BPCER for bona fide, APCER for attacks); strata with a class absent are omitted")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    out["figure_strata.png"] = buf.getvalue()
    return out


def f4(v) -> str:
    return N1.f4(v)


def range_text(s: dict) -> str:
    return NA if s["min"] is None else f"{s['min']:.4f}-{s['max']:.4f}"


def table_md(doc: dict) -> str:
    seeds = [str(s) for s in doc["train_seeds"]]
    L = [f"# N2 part 1: the three N1 LivenessNet checkpoints on the NUAA frames of sessions 01-02 outside fitting and validation (study `{doc['run_id']}`)", "",
         f"Target: {doc['target']['rows']} frames of the official training list ({doc['target']['n_bona_fide']} bona fide, {doc['target']['n_attack']} attacks, "
         f"{doc['target']['n_subjects']} subjects: {doc['target']['n_fitting_subjects']} fitting, {doc['target']['n_validation_subjects']} validation; "
         f"{doc['target']['in_m1_dev']} also in the M1 dev set), preprocessed with the frozen detector that reproduces the training cache "
         f"({doc['cache_check']['reproduced']} of {doc['cache_check']['frames']} cache frames bit for bit). Reference: the session 03 scores of the N1 study, "
         f"read as stored (parity on {doc['parity']['n_keys']} keys: max |difference| {doc['parity']['max_abs_diff']:.2e}, tolerance {doc['parity']['tolerance_abs']}). "
         f"Threshold {THRESHOLD} on the attack probability, {DECISION_RULE}; scores on CPU; no seed selection, no threshold change, no score inversion. "
         "Median and range (minimum-maximum) over the three seeds. NA = not computable (class absent).", "",
         "## Single table: every set, every seed", "",
         "| set | seed | n bona fide | n attack | rejected bona fide | accepted attacks | APCER | BPCER | ACER | AUROC |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, title in doc["sets"].items():
        for s in seeds:
            m = doc["seeds"][s]["metrics"][name]
            L.append(f"| {title} | {s} | {m['n_bona_fide']} | {m['n_attack']} | {m['rejected_bona_fide']} | {m['accepted_attacks']} | "
                     f"{f4(m['apcer'])} | {f4(m['bpcer'])} | {f4(m['acer'])} | {f4(m['auroc'])} |")
        su = doc["summary"][name]
        L.append(f"| {title} | median (range) | | | | | " + " | ".join(f"{f4(su[k]['median'])} ({range_text(su[k])})" for k in ("apcer", "bpcer", "acer", "auroc")) + " |")
    L += ["", "## Subject x class strata (sessions 01-02 against session 03, same subjects)", "",
          "Per seed, group and class: strata computable in both sets and how many have a lower, equal or higher error rate in sessions 01-02 than in session 03 "
          "(per-subject values are in the restricted `strata_all_seeds.csv` and `figure_strata.png`).", "",
          "| seed | group / class | strata | lower in 01-02 | equal | higher in 01-02 | not comparable |", "|---|---|---|---|---|---|---|"]
    for s in seeds:
        for k, v in doc["seeds"][s]["strata_summary"].items():
            L.append(f"| {s} | {k.replace('_', ' ')} | {v['strata']} | {v['lower_in_01_02']} | {v['equal']} | {v['higher_in_01_02']} | {v['not_comparable']} |")
    L += ["", "## Score distributions per class (counts per bin of width 0.1)", ""]
    for name in ("sessions_01_02_491", "session_03_3000"):
        L += [f"### {doc['sets'][name]}", "", "| seed | class | " + " | ".join(f"[{e:.1f}, {e + 1 / BINS:.1f}{')' if i < BINS - 1 else ']'}" for i, e in
                                                                         enumerate(doc["seeds"][seeds[0]]["distributions"][name]["bin_edges"][:-1])) + " |",
              "|---|---|" + "---|" * BINS]
        for s in seeds:
            for cls in ("bona_fide", "attack"):
                L.append(f"| {s} | {cls.replace('_', ' ')} | " + " | ".join(str(c) for c in doc["seeds"][s]["distributions"][name][cls]) + " |")
        L.append("")
    L += ["Detectors on the target frames: " + ", ".join(f"{k} {v}" for k, v in doc["cache_check"]["target_detectors"].items()) + ". "
          "Values are rounded to four decimals; aggregate.json holds the exact values.", ""]
    return "\n".join(L)


def cmd_aggregate(a) -> None:
    d, cfg, cfg_sha, check, check_sha = open_study(a)
    if a.train_seed is not None:
        fail("aggregate takes every seed of the configuration (no --train-seed: no seed selection)")
    for n in AGG_OUTPUTS:
        q = d / n
        if os.path.lexists(q) and not q.is_symlink() and not q.is_file():
            fail(f"{n} in the study folder is not a regular file: refused, nothing removed")
    cache_rec = load_step(d, "cache_check.json", "cache-check", d.name, cfg_sha, check_sha)
    par = load_step(d, "parity.json", "parity", d.name, cfg_sha, check_sha)
    N1.accept(d, "aggregate", "aggregate.log")
    for n in AGG_OUTPUTS:                         # uscite precedenti tolte prima del ricalcolo (nessun PASS vecchio)
        q = d / n
        if q.is_symlink() or q.is_file():
            q.unlink()
            print(f"   removed previous {n}")
    inp = load_inputs(cfg, a, originals=False)
    same_inputs(inp, check)
    seeds, subj_all, strata_all = {}, [], []
    for seed in cfg["n1_study"]["train_seeds"]:
        sd = d / f"seed{seed}"
        sc = N1.load_record(sd / "score.json", "score", d.name, cfg_sha, check_sha)
        if sc.get("train_seed") != seed or sc.get("weights_sha256") != inp.n1[seed]["weights_sha256"] \
                or sc.get("cache_check_sha256") != sha256_file(d / "cache_check.json"):
            fail(f"seed{seed}: score.json records another seed, checkpoint or cache-check")
        for n, h in sc["files"].items():
            if (sd / n).is_symlink() or not (sd / n).is_file() or sha256_file(sd / n) != h:
                fail(f"seed{seed}/{n} changed after scoring (SHA-256 differs from score.json)")
        rows = N1.read_csv_rows(sd / "scores_sessions.csv", SCORE_COLUMNS)
        keys = [r["sample_id"] for r in rows]
        if keys != [r["key"] for r in inp.target]:
            fail(f"seed{seed}/scores_sessions.csv does not hold exactly the target rows in key order")
        frames = {r["sample_id"]: r for r in N1.read_csv_rows(d / FRAMES_TARGET, FRAME_COLUMNS)}
        for r in rows:
            m = inp.by_key["official_train"][r["sample_id"]]
            exp = {"run_id": d.name, "train_seed": str(seed), "subject_id": m["subject"], "session_id": m["session"], "label": m["label"],
                   "group": inp.group[m["subject"]], "in_m1_dev": "1" if r["sample_id"] in inp.m1_dev_keys else "0", "file_sha256": m["sha256"],
                   "frame_sha256": frames[r["sample_id"]]["frame_sha256"], "detector": frames[r["sample_id"]]["detector"],
                   "checkpoint_sha256": inp.n1[seed]["weights_sha256"], "config_sha256": cfg_sha}
            try:
                v = float(r["score_attack"])
            except ValueError:
                v = float("nan")
            if any(r[k] != x for k, x in exp.items()) or not (0 <= v <= 1) or r["prediction"] != ("1" if v > THRESHOLD else "0"):
                fail(f"seed{seed}/scores_sessions.csv: rows inconsistent with the manifests, the frames, the checkpoint or the decision rule")
        met = metrics_from_rows(rows)
        stored = json.loads((sd / "metrics.json").read_text(encoding="utf-8"))["metrics"]
        if json.dumps(met, sort_keys=True) != json.dumps(stored, sort_keys=True):
            fail(f"seed{seed}: metrics recomputed from the scores differ from metrics.json")
        subj_rows = N1.read_csv_rows(sd / "per_subject.csv", SUBJECT_COLUMNS)
        recomputed = [{"run_id": d.name, "train_seed": str(seed), **x} for x in N1.per_subject(rows, "sessions_01_02_491")]
        if N1.csv_text(subj_rows, SUBJECT_COLUMNS) != N1.csv_text(recomputed, SUBJECT_COLUMNS):
            fail(f"seed{seed}: per_subject.csv differs from the table recomputed from the scores")
        if sc.get("detectors") != dict(Counter(r["detector"] for r in rows)) or sc.get("detectors") != cache_rec["target_frames"]["detectors"]:
            fail(f"seed{seed}: detector counts of score.json differ from the rows or from cache-check")
        ref = reference_rows(inp, seed)
        for name, sel in ref.items():
            met[name] = N1.set_metrics([int(r["label"]) for r in sel], [float(r["score_attack"]) for r in sel])
        dist = {name: histogram(sel) for name, sel in {**set_rows(rows), **ref}.items() if name in ("sessions_01_02_491", "session_03_3000", "session_03_9_subjects")}
        st_t = strata_rows(d.name, seed, "sessions_01_02_491", rows, inp.group)
        st_r = strata_rows(d.name, seed, "session_03_9_subjects", ref["session_03_9_subjects"], inp.group)
        strata_all += st_t + st_r
        subj_all += subj_rows
        seeds[str(seed)] = {"metrics": met, "distributions": dist, "strata_summary": strata_summary(st_t, st_r),
                            "weights_sha256": inp.n1[seed]["weights_sha256"], "scores_sha256": sc["files"]["scores_sessions.csv"],
                            "n1_scores_sha256": cfg["n1_study"]["seeds"][str(seed)]["scores_test_sha256"], "detectors": sc["detectors"]}
    if len({v["weights_sha256"] for v in seeds.values()}) != len(seeds):
        fail("two seeds have the same weights")
    summ = {name: {met: N1.summary([seeds[str(s)]["metrics"][name][met] for s in cfg["n1_study"]["train_seeds"]]) for met in ("apcer", "bpcer", "acer", "auroc")}
            for name in SETS}
    t = check["checks"]["target"]
    doc = {"passed": True, "step": "aggregate", "run_id": d.name, "config_sha256": cfg_sha, "check_sha256": check_sha, "commit": check.get("commit"),
           "dirty_at_check": check.get("dirty"), "code_sha256": check["code_sha256"], "train_seeds": cfg["n1_study"]["train_seeds"],
           "n1_study": cfg["n1_study"]["run_id"], "sets": set_titles(t, len(inp.rows["cnn_test"])), "pairs": PAIRS, "threshold": THRESHOLD, "decision": DECISION_RULE,
           "target": {k: t[k] for k in ("rows", "n_bona_fide", "n_attack", "n_subjects", "by_session", "in_m1_dev", "n_fitting_subjects",
                                       "n_validation_subjects", "n_frames_fitting_subjects", "n_frames_validation_subjects")},
           "cache_check": {"frames": cfg["cache_check"]["frames"], "reproduced": sum(cache_rec["cache"][s]["equal"] for s in cache_rec["cache"]),
                           "different": sum(cache_rec["cache"][s]["different"] for s in cache_rec["cache"]), "step_size": cache_rec["step_size"],
                           "target_detectors": cache_rec["target_frames"]["detectors"], "frozen_face_sha256": cache_rec["frozen_face_sha256"],
                           "opencv": cache_rec["environment"].get("opencv")},
           "parity": {"n_keys": par["n_keys"], "tolerance_abs": par["tolerance_abs"], "max_abs_diff": max(v["max_abs_diff"] for v in par["seeds"].values()),
                      "per_seed": {s: v["max_abs_diff"] for s, v in par["seeds"].items()}},
           "seeds": seeds, "summary": summ, "selection": "none: every N1 seed, threshold 0.5 fixed before the run, no score inversion",
           "figures": {"public": {"figure_distributions.png": "attack-score distributions per class, sessions 01-02 and session 03, per seed (from aggregate.json)",
                                  "figure_groups.png": "error rate at 0.5 per group (fitting / validation subjects) and class, sessions 01-02 against session 03, per seed (from aggregate.json)"},
                       "restricted": {"figure_strata.png": "subject x class strata: error rate in sessions 01-02 against session 03, per seed (per-subject values, no identifiers)"}},
           "created": now()}
    js, md = json.dumps(doc, indent=1) + "\n", table_md(doc)
    for name, text in (("aggregate.json", js), ("aggregate.md", md)):
        if probs := N1.public_problems(text, inp):
            fail(f"{name} would hold {', '.join(probs)}: not written")
    figs = (PLOTTER or default_plotter)(json.loads(js), strata_all)       # le figure pubbliche nascono dal solo aggregato scansionato
    if set(figs) != set(PUBLIC_FIGURES) | set(RESTRICTED_FIGURES) or not all(isinstance(v, bytes) and v[:8] == b"\x89PNG\r\n\x1a\n" for v in figs.values()):
        fail("the plotter did not return the three PNG figures")
    write_all(d, {"aggregate.json": js, "aggregate.md": md, "per_subject_all_seeds.csv": N1.csv_text(subj_all, SUBJECT_COLUMNS),
                  "strata_all_seeds.csv": N1.csv_text(strata_all, STRATA_COLUMNS)})
    try:
        write_binary(d, figs)
    except BaseException:
        for n in ("aggregate.json", "aggregate.md", "per_subject_all_seeds.csv", "strata_all_seeds.csv"):
            (d / n).unlink(missing_ok=True)
        raise
    print(md)
    print(f"== aggregate PASS: {len(seeds)} seeds, aggregate.json, aggregate.md and {len(PUBLIC_FIGURES)} figures (publishable), per_subject_all_seeds.csv, "
          f"strata_all_seeds.csv and {', '.join(RESTRICTED_FIGURES)} (restricted)")


# ============================================================================ main

COMMANDS = {"check": cmd_check, "cache-check": cmd_cache_check, "parity": cmd_parity, "score": cmd_score, "aggregate": cmd_aggregate}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="N2 part 1: the N1 LivenessNet checkpoints on the NUAA frames of sessions 01-02 outside fitting and validation")
    ap.add_argument("command", choices=list(COMMANDS))
    ap.add_argument("--config", required=True, help="configuration JSON (n2_sessions.json)")
    ap.add_argument("--manifests", required=True, help="folder with the seven M2 NUAA manifests")
    ap.add_argument("--cache", required=True, help="the local cache nuaa_1500.npz")
    ap.add_argument("--images", required=True, help="root folder of the original NUAA images")
    ap.add_argument("--n1-study", required=True, help="the N1 study folder (checkpoints, records, scores)")
    ap.add_argument("--frozen-face", required=True, help="the frozen face module face_0ac7143.py")
    ap.add_argument("--study-dir", required=True, help="study folder YYYYMMDD-N2-sessions-<shortcommit>[-trial]")
    ap.add_argument("--train-seed", type=int, default=None, help="N1 training seed (score)")
    ap.add_argument("--step", type=int, default=1, help="cache-check: one frame every N (only in -trial runs)")
    ap.add_argument("--i-am-authorized", action="store_true", help="inference on the restricted data is authorised")
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    N1.RUN_STATE.update(dir=None, cmd=a.command, log=None)
    code = 0
    try:
        if a.command == "score" and a.train_seed is None:
            fail("score needs --train-seed")
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
