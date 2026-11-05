"""
Schema dei fold per soggetto su NUAA (controllo subject-held-out con sessione rappresentata nel training), chiesto dal
coordinatore il 3/11/2026 sulla scheda C2: «prepari lo schema concreto dei quattro fold per soggetto, con tre seed,
conteggi per classe/sessione, separazione fitting/dev/test e budget. Fissiamo questo schema prima dei training».

Solo derivazione da manifest già accettati (M2): nessun training, nessuna inferenza, nessuna lettura di immagini.
Lo schema è deterministico (nessun numero casuale): regola di stratificazione e di assegnazione scritta nella
configurazione, applicata alle liste ufficiali NUAA, e manifest per fold e ruolo con SHA-256, così il coordinatore può
rifarlo dagli stessi manifest.

Regola dei fold (configurazione, `folds`): i 16 soggetti delle liste ufficiali in quattro strati, in ordine di
identificativo dentro ogni strato: A = soggetti con frame nelle sessioni 01–02 e bona fide nella sessione 03;
B = con frame nelle sessioni 01–02, senza bona fide nella 03; C = senza frame nelle sessioni 01–02, con bona fide nella
03; D = senza frame nelle sessioni 01–02, senza bona fide nella 03. Assegnazione bilanciata, un soggetto alla volta
nell'ordine A, B, C, D e identificativo crescente: al fold con meno soggetti; a parità al fold con meno soggetti che
condividono una proprietà con lui (frame nelle sessioni 01–02 per A e B, bona fide nella sessione 03 per A e C, la
somma dei due conteggi per A); a parità l'indice più basso. Ogni fold deve avere almeno due soggetti con bona fide
nella sessione 03, due con attacchi nella 03 e due con frame nelle sessioni 01–02; altrimenti FAIL (nessun ripiego).

Ruoli per ogni fold tenuto fuori k: test = fold k (tutti i suoi frame, per sessione); dev (early stopping e soglia EER)
= fold k mod K + 1; fitting = gli altri fold. Fitting e dev campionati con conteggi fissi per cella (classe × sessione)
con `spread` (elementi equidistanti dell'elenco ordinato dei nomi, la regola di M2 e di train_cnn.py); una cella con
meno frame del conteggio fissato è FAIL. I frame già nella cache di training (cnn_train + cnn_es + cnn_test) sono
contati come «in cache», gli altri come «da estrarre» con l'estrattore congelato (budget).

Uscite (cartella del run <out-root>/YYYYMMDD-FOLDS-<commit>[-dev], -dev se l'albero git non è pulito):
  status.json                      RUNNING all'avvio, poi PASS o FAIL
  public/                          solo aggregati e regole, nessun identificativo di soggetto né chiave
    fold_schema_public.json, fold_schema_public.md, SHA256SUMS
  restricted/                      identificativi: area riservata, mai nella PR
    fold_members.csv, per_subject_counts.csv, fold<k>_<ruolo>.csv (12 manifest), SHA256SUMS,
    public_SHA256SUMS.sha256 (ancora della cartella pubblica)
Dentro il repository la cartella del run è ammessa solo sotto results/folds/. Cartella di uscita e ingressi validati
per identità prima di qualunque modifica (m2_writers.output_problem); scritture atomiche (m2_writers.write_all); le
uscite precedenti sono tolte subito dopo l'ammissione e su qualunque errore resta solo status.json = FAIL.

  .venv/bin/python scripts/nuaa_subject_folds.py check-inputs     # solo verifica degli ingressi, nessuna scrittura
  .venv/bin/python scripts/nuaa_subject_folds.py run              # schema e uscite in results/folds/<run-id>/

Libreria standard più scripts/m2_writers.py, scripts/path_guard.py e scripts/package_rules.py.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import platform
import re
import stat
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import m2_writers as mw  # noqa: E402
import package_rules as pr  # noqa: E402
import path_guard as pg  # noqa: E402

DEFAULT_CONFIG = "delivery/folds/nuaa_subject_folds.json"
DEFAULT_OUT_ROOT = "results/folds"
CONFIG_SCHEMA = "nuaa-subject-folds-config/1"
RUN_ID_RE = re.compile(r"^\d{8}-FOLDS-[0-9a-f]{7,40}(-dev)?$")
MANIFESTS = ("official_train", "official_test", "cnn_train", "cnn_es", "cnn_test")
CACHE_MANIFESTS = ("cnn_train", "cnn_es", "cnn_test")
COLUMNS = ("key", "label", "subject", "glasses", "pos", "session", "pic", "official_split", "sha256", "bytes")
SESSIONS = ("01", "02", "03")
LABELS = ("0", "1")
CLASS_NAME = {"0": "bona_fide", "1": "attack"}
ROLES = ("fitting", "dev", "test")
STRATA = ("A", "B", "C", "D")
DEALING = ("balanced: subjects in stratum order A, B, C, D and ascending identifier, each to the fold with the fewest subjects, "
           "ties to the fold with the fewest subjects sharing a property (frames in sessions 01-02 for A and B, bona fide in "
           "session 03 for A and C, both counts for A), then the lowest index")
MIN_SUBJECTS = 2       # per fold: soggetti con bona fide nella 03, con attacchi nella 03, con frame nelle sessioni 01-02
STRATA_TEXT = {"A": "frames in sessions 01-02 and bona fide frames in session 03",
               "B": "frames in sessions 01-02, no bona fide frame in session 03",
               "C": "no frame in sessions 01-02, bona fide frames in session 03",
               "D": "no frame in sessions 01-02, no bona fide frame in session 03"}
CODE_FILES = ("scripts/nuaa_subject_folds.py", "scripts/m2_writers.py", "scripts/path_guard.py", "scripts/package_rules.py")
NAME_RE = re.compile(r"^(\d{4})_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d+)\.jpg$")

STATUS = "status.json"
PUBLIC, RESTRICTED = "public", "restricted"
PUBLIC_OUTPUTS = ("fold_schema_public.json", "fold_schema_public.md", "SHA256SUMS")
FAIL_DOC = {"stage": "invalidation-failed", "note": "this output could not be removed after a failure: no verdict is valid"}
EXTRA_PROTECTED = ("data", "models", "results/c1", "results/m1", "results/m2", "results/m2_rgb", "results/eval", "results/n1",
                   "results/n2", "results/c1crop", "export")
REPO_STATE = None       # (root) -> {"commit", "dirty"}; None: git nel repository (sostituibile solo dai test)


class Failure(Exception):
    """Rifiuto o errore con motivo esplicito (riportato in status.json)."""


def fail(msg: str):
    raise Failure(msg)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def spread(items, k):
    """k elementi distribuiti uniformemente sull'elenco ordinato (la regola di train_cnn.py e di prepare_datasets.py)."""
    items = sorted(items)
    if k >= len(items):
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def restricted_outputs(k: int) -> tuple:
    return (("fold_members.csv", "per_subject_counts.csv")
            + tuple(f"fold{i}_{role}.csv" for i in range(1, k + 1) for role in ROLES)
            + ("public_SHA256SUMS.sha256", "SHA256SUMS"))


# ============================================================================ configurazione e ingressi

def is_count(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def load_config(p: Path) -> tuple:
    """(configurazione, SHA-256 dei byte letti). Campi obbligatori controllati qui, prima di qualunque altra cosa."""
    data = read_regular(p, "configuration")
    try:
        cfg = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        fail(f"configuration {p.name} is not valid JSON: {e}")
    if not isinstance(cfg, dict) or cfg.get("schema") != CONFIG_SCHEMA:
        fail(f"configuration {p.name}: schema must be {CONFIG_SCHEMA!r}")
    f = cfg.get("folds")
    if not isinstance(f, dict) or not is_count(f.get("k")) or f["k"] < 2:
        fail("configuration: folds.k must be an integer >= 2")
    if f.get("strata") != list(STRATA) or f.get("dealing") != DEALING:
        fail("configuration: folds.strata and folds.dealing must state the documented rule")
    if f.get("dev_fold") != "fold (k mod K) + 1 for the held-out fold k" or f.get("fitting_folds") != "the other K - 2 folds":
        fail("configuration: folds.dev_fold and folds.fitting_folds must state the documented rule")
    s = cfg.get("sampling")
    if not isinstance(s, dict) or not all(is_count(s.get(x)) and s[x] > 0 for x in ("fitting_per_cell", "dev_per_cell")):
        fail("configuration: sampling.fitting_per_cell and sampling.dev_per_cell must be positive integers")
    if s.get("rule") != "spread(sorted file names of the cell, n): one every len/n, from the first" or s.get("test") != "all frames of the held-out subjects":
        fail("configuration: sampling.rule and sampling.test must state the documented rule")
    man = cfg.get("manifests")
    if not isinstance(man, dict) or sorted(man) != sorted(MANIFESTS):
        fail(f"configuration: manifests must have exactly the keys {', '.join(MANIFESTS)}")
    for k, v in man.items():
        if not isinstance(v, dict) or not isinstance(v.get("file"), str) or not v["file"] or not re.fullmatch(r"[0-9a-f]{64}", str(v.get("sha256") or "")):
            fail(f"configuration: manifest {k} needs file and sha256")
        if Path(v["file"]).name != v["file"]:
            fail(f"configuration: manifest {k}: file must be a bare file name")
        if not all(is_count(v.get(x)) for x in ("rows", "n_bona_fide", "n_attack", "n_subjects")):
            fail(f"configuration: manifest {k} needs the counts rows, n_bona_fide, n_attack, n_subjects")
    exp = cfg.get("expected")
    if not isinstance(exp, dict) or not all(is_count(exp.get(x)) for x in ("n_subjects", "n_images", "n_cache_frames")):
        fail("configuration: expected must give n_subjects, n_images and n_cache_frames")
    seeds = cfg.get("train_seeds")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds) or not all(is_count(x) for x in seeds):
        fail("configuration: train_seeds must be a non-empty list of distinct non-negative integers")
    for key in ("architecture", "training", "threshold", "metrics", "frames", "budget", "declarations"):
        if key not in cfg:
            fail(f"configuration: {key} is missing")
    return cfg, sha256_bytes(data)


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


def parse_manifest(b: bytes, what: str) -> list:
    try:
        text = b.decode("utf-8")
    except UnicodeDecodeError:
        fail(f"{what} is not UTF-8")
    r = csv.DictReader(io.StringIO(text, newline=""))
    if r.fieldnames != list(COLUMNS):
        fail(f"{what}: columns must be exactly {', '.join(COLUMNS)}")
    rows = list(r)
    keys = Counter(x["key"] for x in rows)
    if any(n > 1 for n in keys.values()):
        fail(f"{what}: duplicated keys")
    for x in rows:
        if x["label"] not in LABELS or x["session"] not in SESSIONS:
            fail(f"{what}: label or session outside the admitted values")
        m = NAME_RE.match(x["key"].split("/", 1)[1] if "/" in x["key"] else "")
        if not m or m.group(1) != x["subject"] or m.group(5) != x["session"]:
            fail(f"{what}: a key does not carry the subject and session of its row")
        if not re.fullmatch(r"[0-9a-f]{64}", x["sha256"]) or not x["bytes"].isdigit():
            fail(f"{what}: sha256 or bytes malformed")
    return rows


def counts_of(rows) -> dict:
    return {"rows": len(rows), "n_bona_fide": sum(1 for r in rows if r["label"] == "0"),
            "n_attack": sum(1 for r in rows if r["label"] == "1"), "n_subjects": len({r["subject"] for r in rows})}


def load_manifests(cfg: dict, mdir: Path) -> dict:
    """Manifest verificati: SHA-256 e conteggi uguali alla configurazione; liste ufficiali disgiunte per chiave e per
    contenuto; manifest della cache dentro le liste ufficiali."""
    out = {}
    for k in MANIFESTS:
        spec = cfg["manifests"][k]
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
        out[k] = rows
    tr, te = out["official_train"], out["official_test"]
    for k in ("official_train", "official_test"):
        if len({r["sha256"] for r in out[k]}) != len(out[k]):
            fail(f"manifest {k}: two rows share a file SHA-256")
    if {r["key"] for r in tr} & {r["key"] for r in te} or {r["sha256"] for r in tr} & {r["sha256"] for r in te}:
        fail("the official lists share a key or a file SHA-256")
    if any(r["session"] == "03" for r in tr) or any(r["session"] != "03" for r in te):
        fail("the official train list must hold sessions 01-02 only and the official test list session 03 only")
    official = {r["key"]: r for r in tr + te}
    for k in CACHE_MANIFESTS:
        for r in out[k]:
            o = official.get(r["key"])
            if o is None or any(o[c] != r[c] for c in COLUMNS):
                fail(f"manifest {k}: a key is not in the official lists with the same values in every column")
    cache = [r["key"] for k in CACHE_MANIFESTS for r in out[k]]
    if len(cache) != len(set(cache)):
        fail("the cache manifests share keys")
    if len(cache) != cfg["expected"]["n_cache_frames"]:
        fail(f"cache manifests: {len(cache)} frames, expected {cfg['expected']['n_cache_frames']}")
    if len(official) != cfg["expected"]["n_images"]:
        fail(f"official lists: {len(official)} images, expected {cfg['expected']['n_images']}")
    if len({r['subject'] for r in official.values()}) != cfg["expected"]["n_subjects"]:
        fail("official lists: number of subjects differs from the configuration")
    return out


# ============================================================================ schema

def subject_properties(official: list) -> dict:
    """Soggetto -> {stratum, sessions_01_02, bona_fide_03, attack_03} dalla presenza di frame nelle sessioni 01–02 e
    delle due classi nella sessione 03 (strato A, B, C, D come nella docstring)."""
    has_0102, bf_03, att_03 = set(), set(), set()
    for r in official:
        if r["session"] in ("01", "02"):
            has_0102.add(r["subject"])
        if r["session"] == "03":
            (bf_03 if r["label"] == "0" else att_03).add(r["subject"])
    out = {}
    for s in {r["subject"] for r in official}:
        st = "A" if s in has_0102 and s in bf_03 else "B" if s in has_0102 else "C" if s in bf_03 else "D"
        out[s] = {"stratum": st, "sessions_01_02": s in has_0102, "bona_fide_03": s in bf_03, "attack_03": s in att_03}
    return out


def strata_of(official: list) -> dict:
    """Strato di ogni soggetto (A, B, C, D)."""
    return {s: v["stratum"] for s, v in subject_properties(official).items()}


def deal_folds(props: dict, k: int) -> dict:
    """Fold -> soggetti con la regola bilanciata DEALING: un soggetto alla volta (strato A, B, C, D, identificativo
    crescente) al fold con meno soggetti; a parità a quello con meno soggetti che condividono una proprietà con lui
    (sessioni 01–02 per A e B, bona fide nella 03 per A e C, somma dei due conteggi per A); a parità l'indice minore."""
    folds = {i: [] for i in range(1, k + 1)}

    def shared(fold, s):
        n = 0
        if props[s]["sessions_01_02"]:
            n += sum(1 for x in fold if props[x]["sessions_01_02"])
        if props[s]["bona_fide_03"]:
            n += sum(1 for x in fold if props[x]["bona_fide_03"])
        return n

    for st in STRATA:
        for s in sorted(x for x, v in props.items() if v["stratum"] == st):
            best = min(range(1, k + 1), key=lambda i: (len(folds[i]), shared(folds[i], s), i))
            folds[best].append(s)
    return folds


def cell_frames(rows, subjects) -> dict:
    """(sessione, classe) -> righe dei soggetti dati, ordinate per nome del file."""
    out = defaultdict(list)
    for r in rows:
        if r["subject"] in subjects:
            out[(r["session"], r["label"])].append(r)
    for v in out.values():
        v.sort(key=lambda r: r["key"].split("/", 1)[1])
    return out


def build_schema(cfg: dict, mans: dict) -> dict:
    """Fold, ruoli e manifest per fold; FAIL se un fold non ha entrambe le classi nella sessione 03, nessun soggetto
    delle sessioni 01–02, o una cella di fitting/dev con meno frame del conteggio fissato."""
    k = cfg["folds"]["k"]
    official = mans["official_train"] + mans["official_test"]
    props = subject_properties(official)
    strata = {s: v["stratum"] for s, v in props.items()}
    folds = deal_folds(props, k)
    cache = {r["key"] for m in CACHE_MANIFESTS for r in mans[m]}
    by_subject = defaultdict(list)
    for r in official:
        by_subject[r["subject"]].append(r)
    for i, subs in folds.items():
        for prop, text in (("bona_fide_03", "bona fide frames in session 03"), ("attack_03", "attack frames in session 03"),
                           ("sessions_01_02", "frames in sessions 01-02")):
            n = sum(1 for s in subs if props[s][prop])
            if n < MIN_SUBJECTS:
                fail(f"fold {i} has {n} subject(s) with {text}, at least {MIN_SUBJECTS} required: the balanced dealing cannot satisfy the constraint")
    n_fit, n_dev = cfg["sampling"]["fitting_per_cell"], cfg["sampling"]["dev_per_cell"]
    per_fold = {}
    for i in range(1, k + 1):
        dev = i % k + 1
        fit = [j for j in range(1, k + 1) if j not in (i, dev)]
        roles = {"test": set(folds[i]), "dev": set(folds[dev]), "fitting": {s for j in fit for s in folds[j]}}
        manifests, availability = {}, {}
        for role in ROLES:
            cells = cell_frames(official, roles[role])
            chosen = []
            availability[role] = {}
            for s in SESSIONS:
                for lab in LABELS:
                    rows = cells.get((s, lab), [])
                    availability[role][f"{s}/{CLASS_NAME[lab]}"] = len(rows)
                    if role == "test":
                        chosen += rows
                    else:
                        n = n_fit if role == "fitting" else n_dev
                        if len(rows) < n:
                            fail(f"fold {i}, {role}, session {s}, {CLASS_NAME[lab]}: {len(rows)} frames available, {n} required")
                        names = spread([r["key"].split("/", 1)[1] for r in rows], n)
                        by_name = {r["key"].split("/", 1)[1]: r for r in rows}
                        chosen += [by_name[x] for x in names]
            chosen.sort(key=lambda r: r["key"])
            manifests[role] = chosen
        keys = [r["key"] for role in ROLES for r in manifests[role]]
        if len(keys) != len(set(keys)):
            fail(f"fold {i}: a frame appears in two roles")
        per_fold[i] = {"dev_fold": dev, "fitting_folds": fit, "subjects": roles, "manifests": manifests,
                       "availability": availability}
    return {"k": k, "strata": strata, "props": props, "folds": folds, "per_fold": per_fold, "cache": cache, "by_subject": by_subject}


# ============================================================================ uscite

def csv_text(header, rows) -> str:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


MANIFEST_HEADER = ["fold", "role", "key", "label", "subject", "session", "sha256", "bytes", "in_cache"]


def role_counts(rows, cache: set) -> dict:
    out = {}
    for s in SESSIONS:
        for lab in LABELS:
            cell = [r for r in rows if r["session"] == s and r["label"] == lab]
            out[f"{s}/{CLASS_NAME[lab]}"] = {"frames": len(cell), "subjects": len({r["subject"] for r in cell}),
                                            "in_cache": sum(1 for r in cell if r["key"] in cache),
                                            "to_extract": sum(1 for r in cell if r["key"] not in cache)}
    out["total"] = {"frames": len(rows), "subjects": len({r["subject"] for r in rows}),
                    "in_cache": sum(1 for r in rows if r["key"] in cache), "to_extract": sum(1 for r in rows if r["key"] not in cache)}
    return out


def md_counts(title: str, rows_by_role: dict, availability: dict) -> str:
    cols = [f"{s}/{CLASS_NAME[lab]}" for s in SESSIONS for lab in LABELS] + ["total"]
    out = [f"**{title}**", "", "| role | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for role in ROLES:
        c = rows_by_role[role]
        out.append(f"| {role} | " + " | ".join(f"{c[x]['frames']} ({c[x]['subjects']} subj.)" for x in cols) + " |")
        if role != "test":
            av = availability[role]
            out.append(f"| {role}, available | " + " | ".join(str(av[x]) for x in cols[:-1]) + f" | {sum(av.values())} |")
    out.append("| to extract (not in cache) | " + " | ".join(str(rows_by_role["fitting"][x]["to_extract"] + rows_by_role["dev"][x]["to_extract"]
                                                                    + rows_by_role["test"][x]["to_extract"]) for x in cols) + " |")
    return "\n".join(out) + "\n"


def build_outputs(cfg: dict, cfg_sha: str, schema: dict, state: dict, run_id: str, mdir: Path) -> tuple:
    """(file pubblici, file riservati) come testi. Il pubblico non contiene identificativi."""
    k, cache = schema["k"], schema["cache"]
    restricted, public_folds = {}, {}
    members = [[i, s, schema["strata"][s]] for i in range(1, k + 1) for s in schema["folds"][i]]
    restricted["fold_members.csv"] = csv_text(["fold", "subject", "stratum"], members)
    per_subject = []
    for s in sorted(schema["by_subject"]):
        c = Counter((r["session"], r["label"]) for r in schema["by_subject"][s])
        per_subject.append([s, schema["strata"][s]] + [c[(ses, lab)] for ses in SESSIONS for lab in LABELS])
    restricted["per_subject_counts.csv"] = csv_text(["subject", "stratum"] + [f"{ses}_{CLASS_NAME[lab]}" for ses in SESSIONS for lab in LABELS], per_subject)
    for i in range(1, k + 1):
        pf = schema["per_fold"][i]
        roles = {}
        for role in ROLES:
            rows = pf["manifests"][role]
            text = csv_text(MANIFEST_HEADER, [[i, role, r["key"], r["label"], r["subject"], r["session"], r["sha256"], r["bytes"],
                                               1 if r["key"] in cache else 0] for r in rows])
            restricted[f"fold{i}_{role}.csv"] = text
            roles[role] = {"manifest": f"fold{i}_{role}.csv", "manifest_sha256": sha256_bytes(text.encode("utf-8")),
                           "n_subjects": len(pf["subjects"][role]), "counts": role_counts(rows, cache),
                           "availability": pf["availability"][role]}
        subs = schema["folds"][i]
        public_folds[str(i)] = {"held_out_fold": i, "dev_fold": pf["dev_fold"], "fitting_folds": pf["fitting_folds"],
                                "stratum_counts": dict(sorted(Counter(schema["strata"][s] for s in subs).items())),
                                "subjects_with": {"bona_fide_03": sum(1 for s in subs if schema["props"][s]["bona_fide_03"]),
                                                  "attack_03": sum(1 for s in subs if schema["props"][s]["attack_03"]),
                                                  "sessions_01_02": sum(1 for s in subs if schema["props"][s]["sessions_01_02"])},
                                "roles": roles}
    to_extract = sorted({r["key"] for i in range(1, k + 1) for role in ROLES for r in schema["per_fold"][i]["manifests"][role] if r["key"] not in cache})
    trainings = k * len(cfg["train_seeds"])
    doc = {
        "schema": "nuaa-subject-folds/1", "run_id": run_id, "status": cfg["status"], "task": cfg["task"],
        "config_sha256": cfg_sha, "commit": state["commit"], "dirty": state["dirty"],
        "manifests": {m: {"file": cfg["manifests"][m]["file"], "sha256": cfg["manifests"][m]["sha256"]} for m in MANIFESTS},
        "code_sha256": state.get("code_sha256", {}),
        "folds": {"k": k, "strata": STRATA_TEXT, "strata_sizes": dict(sorted(Counter(schema["strata"].values()).items())),
                  "dealing": cfg["folds"]["dealing"], "min_subjects_per_property": MIN_SUBJECTS,
                  "dev_fold": cfg["folds"]["dev_fold"], "fitting_folds": cfg["folds"]["fitting_folds"],
                  "subjects_per_fold": {str(i): len(v) for i, v in schema["folds"].items()},
                  "members": "restricted/fold_members.csv", "members_sha256": sha256_bytes(restricted["fold_members.csv"].encode("utf-8"))},
        "sampling": cfg["sampling"], "per_fold": public_folds,
        "frames": {**cfg["frames"], "distinct_frames_used": len({r["key"] for i in range(1, k + 1) for role in ROLES for r in schema["per_fold"][i]["manifests"][role]}),
                   "distinct_frames_to_extract": len(to_extract), "cache_frames": len(cache)},
        "architecture": cfg["architecture"], "train_seeds": cfg["train_seeds"], "training": cfg["training"], "threshold": cfg["threshold"],
        "metrics": cfg["metrics"], "budget": {**cfg["budget"], "trainings": trainings},
        "declarations": cfg["declarations"], "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    md = [f"# NUAA subject folds: schema of the subject-held-out check (run `{run_id}`)", "",
          f"Status: {cfg['status']}", "", cfg["task"], "",
          f"Strata (subjects): " + ", ".join(f"{s} = {n}" for s, n in doc["folds"]["strata_sizes"].items())
          + f"; folds of {', '.join(str(v) for v in doc['folds']['subjects_per_fold'].values())} subjects (members restricted, SHA-256 `{doc['folds']['members_sha256'][:12]}…`).", ""]
    for i in range(1, k + 1):
        pf = public_folds[str(i)]
        sw = pf["subjects_with"]
        md.append(md_counts(f"Held-out fold {i} (test), dev fold {pf['dev_fold']}, fitting folds {pf['fitting_folds']}; strata of the held-out subjects: "
                            + ", ".join(f"{s} {n}" for s, n in pf["stratum_counts"].items())
                            + f"; subjects with bona fide in session 03: {sw['bona_fide_03']}, with attacks in session 03: {sw['attack_03']}, "
                              f"with frames in sessions 01-02: {sw['sessions_01_02']}",
                            {role: pf["roles"][role]["counts"] for role in ROLES},
                            {role: pf["roles"][role]["availability"] for role in ROLES}))
    md += [f"Distinct frames used: {doc['frames']['distinct_frames_used']} of {cfg['expected']['n_images']}; in the training cache {len(cache)}; "
           f"to extract with the frozen extractor: {len(to_extract)}.", "",
           f"Budget: {trainings} trainings ({k} folds × {len(cfg['train_seeds'])} seeds), " + cfg["budget"]["estimate"], ""]
    public = {"fold_schema_public.json": json.dumps(doc, indent=1) + "\n", "fold_schema_public.md": "\n".join(md)}
    return public, restricted


def sums_text(files: dict) -> str:
    return "".join(f"{sha256_bytes(t.encode('utf-8'))}  {n}\n" for n, t in sorted(files.items()))


def public_problems(public: dict, subjects, keys) -> list:
    """Identificativi (soggetti, chiavi, nomi di file delle immagini) o contenuto vietato nel pubblico: elenco dei motivi."""
    out = []
    forbidden = pr.forbidden_text(pr.load_local_patterns(), pr.load_local_patterns(key="tool_reference"))
    for name, text in public.items():
        for s in subjects:
            if f"{s}_{s}_" in text or f'"{s}"' in text or f"subject {s}" in text:
                out.append(f"{name}: subject identifier {s}")
        if any(k in text for k in keys):
            out.append(f"{name}: an image key")
        for label, rx in forbidden:
            if rx.search(text):
                out.append(f"{name}: {label}")
    return out


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
    rid = f"{date}-FOLDS-{state['commit']}" + ("-dev" if state["dirty"] else "")
    if not RUN_ID_RE.match(rid):
        fail(f"run id {rid} does not follow YYYYMMDD-FOLDS-<commit>[-dev]")
    return rid


def protected_extra(root: Path) -> list:
    return [root / x for x in EXTRA_PROTECTED]


def run_dir_problem(run: Path, root: Path, inputs: dict, k: int, out_root: Path = None) -> str:
    """Motivo per cui la cartella del run non è ammessa ("" se lo è), prima di qualunque modifica. `out_root` è la
    radice come data (non risolta): un collegamento simbolico è rifiutato."""
    if (out_root if out_root is not None else run.parent).is_symlink():
        return "the output root is a symbolic link"
    for label, p in inputs.items():
        if pg.within(p, run):
            return f"{label} {Path(p).name} is inside the run folder: inputs and outputs must be separate"
        if label.startswith("manifest") and pg.within(run, Path(p).parent):
            return f"the run folder is inside the manifests folder: inputs and outputs must be separate"
    msg = mw.output_problem(run, root, pg.BASE_PROTECTED, [root / DEFAULT_OUT_ROOT], protected_extra(root), inputs, [STATUS],
                            [run / PUBLIC, run / RESTRICTED])
    if msg:
        return msg
    for sub, names in ((PUBLIC, PUBLIC_OUTPUTS), (RESTRICTED, restricted_outputs(k))):
        if msg := mw.output_problem(run / sub, root, pg.BASE_PROTECTED, [root / DEFAULT_OUT_ROOT], protected_extra(root), inputs, list(names)):
            return msg
    allowed = {"": [STATUS, PUBLIC, RESTRICTED], PUBLIC: PUBLIC_OUTPUTS, RESTRICTED: restricted_outputs(k)}
    for sub, names in allowed.items():
        d = run / sub if sub else run
        if d.is_dir():
            for e in sorted(os.listdir(d)):
                if not any(mw.owned_name(e, n) for n in names):
                    return f"{(Path(sub) / e).as_posix()} in the run folder is not an output of this command: refused, nothing removed"
    return ""


def write_status(run: Path, doc: dict) -> None:
    mw.write_all(run, {STATUS: json.dumps(doc, indent=1) + "\n"})


def invalidate_outputs(run: Path, k: int) -> list:
    left = []
    for sub, names in ((PUBLIC, PUBLIC_OUTPUTS), (RESTRICTED, restricted_outputs(k))):
        d = run / sub
        if d.is_dir():
            left += [f"{sub}/{n}" for n in mw.invalidate_each(d, list(names), None, FAIL_DOC)]
    return left


def code_sha256(root: Path) -> dict:
    return {f: sha256_bytes(read_regular(root / f, "code file")) for f in CODE_FILES}


def cmd_check_inputs(a) -> int:
    root = ROOT
    cfg, cfg_sha = load_config(root / a.config)
    mans = load_manifests(cfg, (root / a.manifests).resolve())
    schema = build_schema(cfg, mans)
    print(f"CHECK-INPUTS: configuration {cfg_sha[:12]}, {len(MANIFESTS)} manifests verified, {schema['k']} folds, "
          f"{sum(len(v) for v in schema['folds'].values())} subjects, nothing written")
    print("RESULT: PASS (check-inputs)")
    return 0


def cmd_run(a) -> int:
    """Configurazione, stato git, ammissione della cartella, RUNNING e invalidazione prima di leggere i manifest:
    un manifest guasto trova le uscite precedenti già tolte (nessun PASS vecchio accanto a un FAIL)."""
    root = ROOT
    cfg, cfg_sha = load_config(root / a.config)
    mdir = (root / a.manifests).resolve()
    k = cfg["folds"]["k"]
    state = repo_state(root)
    state["code_sha256"] = code_sha256(root)
    run_id = run_id_for(a.date or time.strftime("%Y%m%d"), state)
    run = (root / a.out_root).resolve() / run_id
    inputs = {f"manifest {m}": mdir / cfg["manifests"][m]["file"] for m in MANIFESTS}
    inputs["configuration"] = (root / a.config).resolve()
    if msg := run_dir_problem(run, root, inputs, k, root / a.out_root):
        fail(msg)
    run.mkdir(parents=True, exist_ok=True)
    write_status(run, {"status": "RUNNING", "run_id": run_id, "config_sha256": cfg_sha, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    left = invalidate_outputs(run, k)
    if left:
        write_status(run, {"status": "FAIL", "run_id": run_id, "reason": f"previous outputs could not be removed: {left}"})
        fail(f"previous outputs could not be removed: {left}")
    try:
        mans = load_manifests(cfg, mdir)
        schema = build_schema(cfg, mans)
        public, restricted = build_outputs(cfg, cfg_sha, schema, state, run_id, mdir)
        keys = {r["key"] for m in MANIFESTS for r in mans[m]} | {r["key"].split("/", 1)[1] for m in MANIFESTS for r in mans[m]}
        problems = public_problems(public, sorted(schema["strata"]), keys)
        if problems:
            fail("public outputs refused: " + "; ".join(problems))
        public["SHA256SUMS"] = sums_text(public)
        restricted["public_SHA256SUMS.sha256"] = sha256_bytes(public["SHA256SUMS"].encode("utf-8")) + "  public/SHA256SUMS\n"
        restricted["SHA256SUMS"] = sums_text(restricted)
        (run / PUBLIC).mkdir(exist_ok=True)
        (run / RESTRICTED).mkdir(exist_ok=True)
        mw.write_all(run / PUBLIC, public)
        mw.write_all(run / RESTRICTED, restricted)
    except BaseException as e:  # noqa: BLE001  (su qualunque errore nessuna uscita resta accanto a status FAIL)
        left = invalidate_outputs(run, k)
        reason = str(e) if isinstance(e, Failure) else f"unexpected error {type(e).__name__}: {e}"
        write_status(run, {"status": "FAIL", "run_id": run_id, "reason": reason, "outputs_left": left})
        raise
    write_status(run, {"status": "PASS", "run_id": run_id, "config_sha256": cfg_sha, "commit": state["commit"], "dirty": state["dirty"],
                       "public_sha256sums": sha256_bytes(public["SHA256SUMS"].encode("utf-8")),
                       "restricted_sha256sums": sha256_bytes(restricted["SHA256SUMS"].encode("utf-8")),
                       "python": platform.python_version(), "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    print(f"RUN: {run_id}: {k} folds, {len(public)} public and {len(restricted)} restricted files in {run}")
    print("RESULT: PASS (run)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="NUAA subject folds: schema of the subject-held-out check (no training, no inference)")
    ap.add_argument("cmd", choices=["check-inputs", "run"])
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="configuration JSON (relative to the repository root)")
    ap.add_argument("--manifests", default="results/m2/manifests", help="folder of the M2 NUAA manifests (relative to the repository root)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="folder that receives <run-id>/ (default results/folds)")
    ap.add_argument("--date", default=None, help="YYYYMMDD of the run id (default: today)")
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
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: unexpected error {type(e).__name__}: {e}", file=sys.stderr)
        print(f"RESULT: FAIL ({a.cmd})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
