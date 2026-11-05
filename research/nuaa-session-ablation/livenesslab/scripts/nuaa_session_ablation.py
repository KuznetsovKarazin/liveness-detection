"""
Conteggi disponibili per la proposta di protocollo «training con e senza la sessione 03» su NUAA, chiesta dal
coordinatore il 4/11/2026 sulla scheda C2: «prepari una breve proposta di protocollo [...] per confrontare il training
con e senza la sessione 03. Mantenga fissi, per quanto possibile, soggetti, numerosità, test, preprocessing e ricetta;
specifichi anche come organizzare il dev e scegliere la soglia senza usare il test. Indichi i conteggi disponibili, gli
eventuali limiti e il budget di calcolo».

Solo derivazione da manifest già accettati: nessun training, nessuna inferenza, nessuna immagine letta. Il disegno è
deterministico (nessun numero casuale) e usa gli stessi quattro fold per soggetto dello schema approvato
(run 20261103-FOLDS-67b7f3d): per il fold tenuto fuori k, test = i suoi soggetti (tutti i frame, invariato), dev = fold
k mod K + 1, fitting = gli altri due fold.
  S12      soggetti del fitting con frame nelle sessioni 01–02 (i soggetti di training ufficiali dei due fold di fitting)
  braccio B   (senza sessione 03) fitting = sessioni 01 e 02 dei soggetti S12, celle 01/bf, 01/att, 02/bf, 02/att, n
              frame per cella; n = il più grande multiplo di rounding_step non sopra la disponibilità minima sui fold
  braccio A*  (con sessione 03, stessi soggetti S12) fitting = sessioni 01, 02 e 03 dei soggetti S12, 6 celle, totale
              uguale al braccio B (4n) ripartito in parti uguali; il resto, un frame per cella, nell'ordine
              remainder_order (bona fide e attacchi restano bilanciati); se una cella non basta in qualche fold prende la
              disponibilità minima sui fold e il resto si ridistribuisce (variante «sbilanciata», dichiarata)
  braccio A   (riferimento già eseguito, run dei fold) fitting = 350 per cella sui 6 strati di tutti gli 8 soggetti del
              fitting: conteggi riportati dai manifest dello schema
  braccio B′  (facoltativo, controllo della numerosità per cella) sessioni 01–02 dei soggetti S12 con le stesse
              quantità per cella delle celle 01–02 di A* (934 frame); budget stimato a parte
  annidamento le celle 01–02 di A* e di B′ sono spread sui frame già scelti per B nella stessa cella (sottoinsiemi
              di B); le celle 03 di A* sono spread sui frame disponibili dei soggetti S12
  dev comune  sessioni 01–02 dei soggetti del fold di dev, 4 celle, d per cella (stessa regola di arrotondamento);
              alternativa: dev con la stessa miscela di sessioni del braccio A* (tutti i soggetti del fold di dev,
              6 celle come il dev dello schema, totale 4d con la stessa ripartizione), solo riportata
Campionamento di ogni cella con `spread` (la regola di M2, di train_cnn.py e dello schema dei fold).

Riservatezza del pubblico: nessun identificativo di soggetto né chiave; un conteggio di frame è pubblicato solo se la
sua cella ha almeno due soggetti e se, insieme a quelli già pubblicati dallo schema approvato e a quelli già ammessi
qui, non rende ricavabile per differenza alcuna nuova quantità di un solo soggetto (controllo esatto di rango sugli
atomi soggetto × sessione × classe: per ogni soggetto s, d_s = rango(tutto) − rango(tutto senza le colonne di s) non
deve crescere rispetto allo schema). I conteggi trattenuti diventano «restricted» con il numero di soggetti e, dove
serve, il minimo richiesto dal disegno (≥ n). Nel riservato tutto.

Ingressi (sola lettura, ancorati): la configurazione dello schema dei fold (SHA-256 nella configurazione) e da essa i
manifest M2 delle liste ufficiali (SHA-256 e conteggi); il run dello schema: status.json PASS, public/SHA256SUMS e
restricted/SHA256SUMS con SHA-256 fissati nella configurazione, fold_schema_public.json e i 12 manifest per fold e ruolo
verificati contro quei SHA256SUMS e riga per riga contro le liste ufficiali.

Uscite (cartella del run <out-root>/YYYYMMDD-SESSABL-<commit>[-dev], -dev se l'albero git non è pulito):
  status.json      RUNNING all'avvio, poi PASS o FAIL
  public/          session_ablation_public.json, session_ablation_public.md, SHA256SUMS (nessun identificativo)
  restricted/      subjects.csv, counts_restricted.json, <braccio>_fold<k>.csv (20 manifest delle celle proposte),
                   public_SHA256SUMS.sha256 (ancora del pubblico), SHA256SUMS
Dentro il repository la cartella del run è ammessa solo sotto results/sessionablation/. Cartella e ingressi validati
prima di qualunque modifica (m2_writers.output_problem); scritture atomiche; le uscite precedenti sono tolte subito
dopo l'ammissione e su qualunque errore resta solo status.json = FAIL.

  .venv/bin/python scripts/nuaa_session_ablation.py check-inputs                # verifica e conteggi, nessuna scrittura
  .venv/bin/python scripts/nuaa_session_ablation.py run --i-am-authorized       # uscite in results/sessionablation/<run-id>/

Libreria standard più scripts/nuaa_subject_folds.py (lettura dei manifest, spread), scripts/m2_writers.py,
scripts/path_guard.py e scripts/package_rules.py.
"""
import argparse
import csv
import io
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import m2_writers as mw  # noqa: E402
import nuaa_subject_folds as NF  # noqa: E402
import path_guard as pg  # noqa: E402

Failure = NF.Failure
fail = NF.fail
sha256_bytes = NF.sha256_bytes

DEFAULT_CONFIG = "delivery/sessionablation/nuaa_session_ablation.json"
DEFAULT_OUT_ROOT = "results/sessionablation"
DEFAULT_FOLDS_ROOT = "results/folds"
CONFIG_SCHEMA = "nuaa-session-ablation-config/1"
RUN_ID_RE = re.compile(r"^\d{8}-SESSABL-[0-9a-f]{7,40}(-dev)?$")
OFFICIAL = ("official_train", "official_test")
SESSIONS, LABELS, CLASS_NAME = NF.SESSIONS, NF.LABELS, NF.CLASS_NAME
CELLS = [f"{s}/{CLASS_NAME[lab]}" for s in SESSIONS for lab in LABELS]
CELLS_12 = CELLS[:4]
CELLS_03 = CELLS[4:]
SCHEMA_ROLES = NF.ROLES
S12_RULE = "subjects of the two fitting folds with frames in sessions 01-02"
DEV_RULE = ("subjects of the dev fold (k mod K) + 1: sessions 01-02 for the common dev, all sessions for the "
            "alternative with the session mix of arm A*")
ROUNDING_RULE = "largest multiple of rounding_step not above the smallest availability over the cells and the held-out folds"
ALLOCATION_RULE = ("total split evenly over the cells, the remainder one frame each to the cells in remainder_order; a cell "
                   "short in some held-out fold takes the smallest availability over the folds and the rest is split again "
                   "over the other cells (unbalanced variant)")
SAMPLING_RULE = "spread(sorted file names of the cell, n): one every len/n, from the first"
NESTING_RULE = ("arm B: spread over the available frames of each session 01-02 cell; arm A* and arm B' session 01-02 "
                "cells: spread over the frames chosen for arm B in that cell (nested in B); arm A* session 03 cells: "
                "spread over the available frames of the S12 subjects")
ARMS = ("arm_b", "arm_a_star")
CODE_FILES = ("scripts/nuaa_session_ablation.py", "scripts/nuaa_subject_folds.py", "scripts/m2_writers.py",
              "scripts/path_guard.py", "scripts/package_rules.py")

STATUS = "status.json"
PUBLIC, RESTRICTED = "public", "restricted"
PUBLIC_OUTPUTS = ("session_ablation_public.json", "session_ablation_public.md", "SHA256SUMS")
MANIFEST_KINDS = ("armB_fitting", "armAstar_fitting", "armBprime_fitting", "dev", "devmix")
FAIL_DOC = {"stage": "invalidation-failed", "note": "this output could not be removed after a failure: no verdict is valid"}
EXTRA_PROTECTED = ("data", "models", "results/c1", "results/m1", "results/m2", "results/m2_rgb", "results/eval", "results/n1",
                   "results/n2", "results/c1crop", "results/folds", "results/foldsrun", "results/foldscompare",
                   "results/stepzero", "export")
REPO_STATE = None       # (root) -> {"commit", "dirty"}; None: git nel repository (sostituibile solo dai test)
RESTRICTED_NOTE = {"one": "restricted: one subject", "schema": "restricted: one subject, already public in the fold schema",
                   "diff": "restricted: derivable by differencing"}
RESTRICTED_TAG = {"restricted: one subject": "R1", "restricted: one subject, already public in the fold schema": "R1s",
                  "restricted: derivable by differencing": "Rd"}


def restricted_outputs(k: int) -> tuple:
    return (("subjects.csv", "counts_restricted.json")
            + tuple(f"{kind}_fold{i}.csv" for i in range(1, k + 1) for kind in MANIFEST_KINDS)
            + ("public_SHA256SUMS.sha256", "SHA256SUMS"))


def is_count(v) -> bool:
    return NF.is_count(v)


def is_hex(v) -> bool:
    return isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) is not None


# ============================================================================ configurazione

def load_config(p: Path) -> tuple:
    """(configurazione, SHA-256 dei byte letti). Campi e regole obbligatori controllati qui, prima di ogni altra cosa."""
    data = NF.read_regular(p, "configuration")
    try:
        cfg = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        fail(f"configuration {p.name} is not valid JSON: {e}")
    if not isinstance(cfg, dict) or cfg.get("schema") != CONFIG_SCHEMA:
        fail(f"configuration {p.name}: schema must be {CONFIG_SCHEMA!r}")
    inp = cfg.get("inputs")
    if not isinstance(inp, dict) or not isinstance(inp.get("folds_config"), dict) or not isinstance(inp.get("fold_schema"), dict):
        fail("configuration: inputs.folds_config and inputs.fold_schema are required")
    fc, fs = inp["folds_config"], inp["fold_schema"]
    if not isinstance(fc.get("path"), str) or not fc["path"] or Path(fc["path"]).is_absolute() or ".." in Path(fc["path"]).parts:
        fail("configuration: inputs.folds_config.path must be a relative path inside the repository")
    if not is_hex(fc.get("sha256")):
        fail("configuration: inputs.folds_config.sha256 must be a SHA-256")
    if not isinstance(fs.get("run_id"), str) or not NF.RUN_ID_RE.match(fs["run_id"]):
        fail("configuration: inputs.fold_schema.run_id must be a fold schema run id")
    if not (is_hex(fs.get("public_sha256sums")) and is_hex(fs.get("restricted_sha256sums"))):
        fail("configuration: inputs.fold_schema needs public_sha256sums and restricted_sha256sums")
    exp = cfg.get("expected")
    if not isinstance(exp, dict) or not all(is_count(exp.get(x)) for x in ("k", "n_subjects", "n_images")) or exp["k"] < 3:
        fail("configuration: expected must give k (>= 3), n_subjects and n_images")
    tf = exp.get("test_frames")
    if not isinstance(tf, list) or len(tf) != exp["k"] or not all(is_count(x) for x in tf):
        fail("configuration: expected.test_frames must list one count per fold")
    d = cfg.get("design")
    if not isinstance(d, dict):
        fail("configuration: design is missing")
    for key, text in (("s12", S12_RULE), ("dev_subjects", DEV_RULE), ("rounding_rule", ROUNDING_RULE),
                      ("allocation_rule", ALLOCATION_RULE), ("sampling", SAMPLING_RULE)):
        if d.get(key) != text:
            fail(f"configuration: design.{key} must state the documented rule")
    if not is_count(d.get("rounding_step")) or d["rounding_step"] < 1:
        fail("configuration: design.rounding_step must be a positive integer")
    if sorted(d.get("remainder_order") or []) != sorted(CELLS) or len(d["remainder_order"]) != len(CELLS):
        fail("configuration: design.remainder_order must list the six cells once each")
    if d.get("nesting") != NESTING_RULE:
        fail("configuration: design.nesting must state the documented rule")
    bp = d.get("arm_b_prime")
    if not isinstance(bp, dict) or bp.get("cells") != CELLS_12 or not isinstance(bp.get("allocation"), dict) \
            or sorted(bp["allocation"]) != sorted(CELLS_12) or not all(is_count(x) for x in bp["allocation"].values()):
        fail("configuration: design.arm_b_prime needs the four session 01-02 cells and an allocation per cell")
    for key, cells in (("arm_b", CELLS_12), ("dev", CELLS_12)):
        v = d.get(key)
        if not isinstance(v, dict) or v.get("cells") != cells or not is_count(v.get("per_cell")) or v["per_cell"] < 1:
            fail(f"configuration: design.{key} needs cells {cells} and a positive per_cell")
    for key in ("arm_a_star", "dev_mix"):
        v = d.get(key)
        if not isinstance(v, dict) or v.get("cells") != CELLS or not isinstance(v.get("allocation"), dict) \
                or sorted(v["allocation"]) != sorted(CELLS) or not all(is_count(x) for x in v["allocation"].values()):
            fail(f"configuration: design.{key} needs the six cells and an allocation per cell")
        if v.get("variant") not in ("balanced", "unbalanced"):
            fail(f"configuration: design.{key}.variant must be balanced or unbalanced")
        if not is_count(v.get("total")):
            fail(f"configuration: design.{key}.total must be a count")
    if d["arm_a_star"]["total"] != len(CELLS_12) * d["arm_b"]["per_cell"]:
        fail("configuration: design.arm_a_star.total must equal the arm B total (4 x arm_b.per_cell)")
    if d["dev_mix"]["total"] != len(CELLS_12) * d["dev"]["per_cell"]:
        fail("configuration: design.dev_mix.total must equal the common dev total (4 x dev.per_cell)")
    for key in ("arm_a_star", "dev_mix"):
        if sum(d[key]["allocation"].values()) != d[key]["total"]:
            fail(f"configuration: design.{key}.allocation must add up to its total")
    seeds = cfg.get("train_seeds")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds) or not all(is_count(x) for x in seeds):
        fail("configuration: train_seeds must be a non-empty list of distinct non-negative integers")
    b = cfg.get("budget")
    if not isinstance(b, dict):
        fail("configuration: budget is missing")
    for key in ("reference_train_seconds", "reference_score_seconds", "reference_threshold_seconds"):
        v = b.get(key)
        if not isinstance(v, list) or not v or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0 for x in v):
            fail(f"configuration: budget.{key} must be a non-empty list of positive seconds")
    for key in ("reference_fitting_frames", "reference_dev_frames", "reference_scored_frames_per_seed", "wall_time_cap_minutes"):
        if not is_count(b.get(key)) or b[key] < 1:
            fail(f"configuration: budget.{key} must be a positive integer")
    if not is_hex(b.get("reference_run_record_sha256")) or not isinstance(b.get("reference_run"), str):
        fail("configuration: budget needs reference_run and reference_run_record_sha256")
    for key in ("status", "task", "arms", "training", "threshold", "metrics", "declarations"):
        if key not in cfg:
            fail(f"configuration: {key} is missing")
    return cfg, sha256_bytes(data)


# ============================================================================ ingressi

def parse_sums(b: bytes, what: str) -> dict:
    """Nome -> SHA-256 da un file SHA256SUMS («<sha>  <nome>» per riga)."""
    out = {}
    try:
        text = b.decode("utf-8")
    except UnicodeDecodeError:
        fail(f"{what} is not UTF-8")
    for line in text.splitlines():
        m = re.fullmatch(r"([0-9a-f]{64})  (\S+)", line)
        if not m or m.group(2) in out:
            fail(f"{what}: malformed or duplicated line")
        out[m.group(2)] = m.group(1)
    return out


def read_listed(d: Path, name: str, sums: dict, what: str) -> bytes:
    """Byte di un file elencato in un SHA256SUMS ancorato, con lo stesso SHA-256."""
    if name not in sums:
        fail(f"{what}: {name} is not listed in its SHA256SUMS")
    b = NF.read_regular(d / name, what)
    if sha256_bytes(b) != sums[name]:
        fail(f"{what}: {name} differs from its SHA256SUMS")
    return b


def parse_fold_manifest(b: bytes, k: int, role: str, official: dict) -> list:
    """Righe di un manifest dello schema (fold, ruolo), ognuna uguale alla riga ufficiale della sua chiave."""
    what = f"fold schema manifest fold{k}_{role}.csv"
    try:
        text = b.decode("utf-8")
    except UnicodeDecodeError:
        fail(f"{what} is not UTF-8")
    r = csv.DictReader(io.StringIO(text, newline=""))
    if r.fieldnames != NF.MANIFEST_HEADER:
        fail(f"{what}: columns must be exactly {', '.join(NF.MANIFEST_HEADER)}")
    rows = list(r)
    if len({x["key"] for x in rows}) != len(rows):
        fail(f"{what}: duplicated keys")
    for x in rows:
        o = official.get(x["key"])
        if x["fold"] != str(k) or x["role"] != role:
            fail(f"{what}: a row with fold or role different from its file")
        if o is None or any(o[c] != x[c] for c in ("label", "subject", "session", "sha256", "bytes")):
            fail(f"{what}: a key is not in the official lists with the same label, subject, session, sha256 and bytes")
        if x["in_cache"] not in ("0", "1"):
            fail(f"{what}: in_cache must be 0 or 1")
    return rows


def load_inputs(cfg: dict, root: Path, mdir: Path, folds_root: Path) -> dict:
    """Ingressi verificati: configurazione dello schema, liste ufficiali M2, run dello schema (status, SHA256SUMS
    ancorati, JSON pubblico, 12 manifest) e coerenza fra loro. Nessuna scrittura."""
    fc = cfg["inputs"]["folds_config"]
    fb = NF.read_regular(root / fc["path"], "fold schema configuration")
    if sha256_bytes(fb) != fc["sha256"]:
        fail(f"fold schema configuration {Path(fc['path']).name}: SHA-256 {sha256_bytes(fb)} differs from the configuration")
    try:
        fcfg = json.loads(fb.decode("utf-8"))
        specs = {m: fcfg["manifests"][m] for m in OFFICIAL}
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        fail("fold schema configuration: unreadable, or without the official manifests")
    mans = {}
    for m in OFFICIAL:
        spec = specs[m]
        if not isinstance(spec, dict) or not isinstance(spec.get("file"), str) or Path(spec["file"]).name != spec["file"]:
            fail(f"fold schema configuration: manifest {m} needs a bare file name")
        b = NF.read_regular(mdir / spec["file"], f"manifest {m}")
        if sha256_bytes(b) != spec.get("sha256"):
            fail(f"manifest {m} ({spec['file']}): SHA-256 {sha256_bytes(b)} differs from the fold schema configuration")
        rows = NF.parse_manifest(b, f"manifest {m}")
        for x, v in NF.counts_of(rows).items():
            if spec.get(x) != v:
                fail(f"manifest {m}: {x} = {v} differs from the fold schema configuration ({spec.get(x)})")
        mans[m] = rows
    tr, te = mans["official_train"], mans["official_test"]
    if any(r["session"] == "03" for r in tr) or any(r["session"] != "03" for r in te):
        fail("the official train list must hold sessions 01-02 only and the official test list session 03 only")
    official = {r["key"]: r for r in tr + te}
    if len(official) != len(tr) + len(te) or len({r["sha256"] for r in tr + te}) != len(tr) + len(te):
        fail("the official lists share a key or a file SHA-256")
    exp = cfg["expected"]
    if len(official) != exp["n_images"]:
        fail(f"official lists: {len(official)} images, expected {exp['n_images']}")
    if len({r["subject"] for r in official.values()}) != exp["n_subjects"]:
        fail("official lists: number of subjects differs from the configuration")
    # run dello schema dei fold: status, SHA256SUMS ancorati, JSON pubblico e manifest
    fs = cfg["inputs"]["fold_schema"]
    run = folds_root / fs["run_id"]
    try:
        st = json.loads(NF.read_regular(run / STATUS, "fold schema status").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        fail("fold schema status.json is not valid JSON")
    if not isinstance(st, dict) or st.get("status") != "PASS" or st.get("run_id") != fs["run_id"]:
        fail("fold schema status.json: the run is not PASS or has another run id")
    if st.get("config_sha256") != fc["sha256"]:
        fail("fold schema status.json: the run was made with another fold schema configuration")
    if st.get("public_sha256sums") != fs["public_sha256sums"] or st.get("restricted_sha256sums") != fs["restricted_sha256sums"]:
        fail("fold schema status.json: SHA256SUMS anchors differ from the configuration")
    sums = {}
    for sub, pin in ((PUBLIC, fs["public_sha256sums"]), (RESTRICTED, fs["restricted_sha256sums"])):
        b = NF.read_regular(run / sub / "SHA256SUMS", f"fold schema {sub}/SHA256SUMS")
        if sha256_bytes(b) != pin:
            fail(f"fold schema {sub}/SHA256SUMS: SHA-256 differs from the configuration")
        sums[sub] = parse_sums(b, f"fold schema {sub}/SHA256SUMS")
    try:
        spub = json.loads(read_listed(run / PUBLIC, "fold_schema_public.json", sums[PUBLIC], "fold schema public").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        fail("fold_schema_public.json is not valid JSON")
    k = exp["k"]
    if not isinstance(spub, dict) or not isinstance(spub.get("per_fold"), dict) or sorted(spub["per_fold"]) != [str(i) for i in range(1, k + 1)]:
        fail(f"fold_schema_public.json: per_fold must hold the folds 1..{k}")
    schema_rows, file_sha = {}, {}
    for i in range(1, k + 1):
        for role in SCHEMA_ROLES:
            name = f"fold{i}_{role}.csv"
            b = read_listed(run / RESTRICTED, name, sums[RESTRICTED], "fold schema restricted")
            file_sha[(i, role)] = sha256_bytes(b)
            schema_rows[(i, role)] = parse_fold_manifest(b, i, role, official)
    members = {i: {r["subject"] for r in schema_rows[(i, "test")]} for i in range(1, k + 1)}
    tested = [r["key"] for i in range(1, k + 1) for r in schema_rows[(i, "test")]]
    if len(tested) != len(set(tested)) or set(tested) != set(official):
        fail("fold schema: the test manifests must hold every official frame exactly once")
    for i in range(1, k + 1):
        if {r["key"] for r in schema_rows[(i, "test")]} != {key for key, r in official.items() if r["subject"] in members[i]}:
            fail(f"fold schema: test of fold {i} is not every frame of its subjects")
        if any(members[i] & members[j] for j in range(i + 1, k + 1)):
            fail("fold schema: a subject is in two folds")
    return {"official": official, "members": members, "schema_rows": schema_rows, "schema_public": spub, "file_sha": file_sha,
            "folds_config_sha256": fc["sha256"]}


# ============================================================================ disegno

def floor_to(v: int, step: int) -> int:
    return (v // step) * step


def allocate(total: int, cells: list, order: list) -> dict:
    """Ripartizione uguale di total sulle celle; il resto, un frame ciascuna, alle celle nell'ordine dato."""
    base, rest = divmod(total, len(cells))
    out = {c: base for c in cells}
    for c in [x for x in order if x in out][:rest]:
        out[c] += 1
    return out


def allocate_capped(total: int, cells: list, order: list, caps: dict) -> tuple:
    """(allocazione, variante): bilanciata se ogni cella sta sotto il suo tetto (disponibilità minima sui fold);
    altrimenti le celle corte prendono il tetto e il resto si ripartisce di nuovo sulle altre (sbilanciata)."""
    fixed = {}
    while True:
        free = [c for c in cells if c not in fixed]
        if not free:
            fail(f"allocation of {total} frames: every cell is short, the total cannot be reached")
        alloc = allocate(total - sum(fixed.values()), free, order)
        short = [c for c in free if alloc[c] > caps[c]]
        if not short:
            out = {**fixed, **alloc}
            return {c: out[c] for c in cells}, ("unbalanced" if fixed else "balanced")
        for c in short:
            fixed[c] = caps[c]


def cell_rows(official: dict, subjects: set) -> dict:
    """Cella -> righe dei soggetti dati, ordinate per nome del file."""
    out = {c: [] for c in CELLS}
    for r in official.values():
        if r["subject"] in subjects:
            out[f"{r['session']}/{CLASS_NAME[r['label']]}"].append(r)
    for v in out.values():
        v.sort(key=lambda r: r["key"].split("/", 1)[1])
    return out


def sample(rows: list, n: int, what: str) -> list:
    if len(rows) < n:
        fail(f"{what}: {len(rows)} frames available, {n} required")
    by_name = {r["key"].split("/", 1)[1]: r for r in rows}
    return [by_name[x] for x in NF.spread(list(by_name), n)]


def build_design(cfg: dict, inp: dict) -> dict:
    """Ruoli, soggetti S12, disponibilità, numerosità con le regole della configurazione (FAIL se la configurazione
    dichiara valori diversi da quelli delle regole) e manifest delle celle proposte."""
    k, d = cfg["expected"]["k"], cfg["design"]
    official, members = inp["official"], inp["members"]
    count = Counter((r["subject"], f"{r['session']}/{CLASS_NAME[r['label']]}") for r in official.values())
    has12 = {s for (s, c) in count if c in CELLS_12}
    folds = {}
    for i in range(1, k + 1):
        dev = i % k + 1
        fit = [j for j in range(1, k + 1) if j not in (i, dev)]
        fit_subjects = set().union(*(members[j] for j in fit))
        s12 = fit_subjects & has12
        folds[i] = {"dev_fold": dev, "fitting_folds": fit, "test": set(members[i]), "dev": set(members[dev]),
                    "dev12": members[dev] & has12, "fitting": fit_subjects, "s12": s12, "s03only": fit_subjects - s12,
                    "cells_s12": cell_rows(official, s12), "cells_dev": cell_rows(official, members[dev])}
        if not s12 or not folds[i]["dev12"]:
            fail(f"held-out fold {i}: no fitting or dev subject with frames in sessions 01-02")
    step = d["rounding_step"]
    min_b = min(len(folds[i]["cells_s12"][c]) for i in folds for c in CELLS_12)
    n = floor_to(min_b, step)
    if n < 1:
        fail(f"arm B: the smallest availability ({min_b}) is below the rounding step")
    if d["arm_b"]["per_cell"] != n:
        fail(f"configuration: design.arm_b.per_cell = {d['arm_b']['per_cell']} differs from the rule ({n})")
    caps_a = {c: min(len(folds[i]["cells_s12"][c]) for i in folds) for c in CELLS}
    alloc_a, var_a = allocate_capped(len(CELLS_12) * n, CELLS, d["remainder_order"], caps_a)
    if d["arm_a_star"]["allocation"] != alloc_a or d["arm_a_star"]["variant"] != var_a:
        fail(f"configuration: design.arm_a_star allocation or variant differ from the rule ({var_a}: "
             + ", ".join(f"{c} {alloc_a[c]}" for c in CELLS) + ")")
    min_dev = min(len(folds[i]["cells_dev"][c]) for i in folds for c in CELLS_12)
    dn = floor_to(min_dev, step)
    if dn < 1:
        fail(f"dev: the smallest availability ({min_dev}) is below the rounding step")
    if d["dev"]["per_cell"] != dn:
        fail(f"configuration: design.dev.per_cell = {d['dev']['per_cell']} differs from the rule ({dn})")
    caps_m = {c: min(len(folds[i]["cells_dev"][c]) for i in folds) for c in CELLS}
    alloc_m, var_m = allocate_capped(len(CELLS_12) * dn, CELLS, d["remainder_order"], caps_m)
    if d["dev_mix"]["allocation"] != alloc_m or d["dev_mix"]["variant"] != var_m:
        fail(f"configuration: design.dev_mix allocation or variant differ from the rule ({var_m}: "
             + ", ".join(f"{c} {alloc_m[c]}" for c in CELLS) + ")")
    # braccio B′ (facoltativo): stesse quantità per cella delle celle 01-02 di A*, annidato in B
    alloc_bp = {c: alloc_a[c] for c in CELLS_12}
    if d["arm_b_prime"]["allocation"] != alloc_bp:
        fail("configuration: design.arm_b_prime.allocation must equal the session 01-02 cells of arm A* ("
             + ", ".join(f"{c} {alloc_bp[c]}" for c in CELLS_12) + ")")
    test_frames = cfg["expected"]["test_frames"]
    for i, f in folds.items():
        what = f"held-out fold {i}"
        b_cells = {c: sample(f["cells_s12"][c], n, f"{what}, armB_fitting, {c}") for c in CELLS_12}
        cells = {"armB_fitting": b_cells,
                 # campioni annidati: le celle 01-02 di A* e di B′ sono spread sui frame già scelti per B
                 "armAstar_fitting": {**{c: sample(b_cells[c], alloc_a[c], f"{what}, armAstar_fitting, {c}") for c in CELLS_12},
                                      **{c: sample(f["cells_s12"][c], alloc_a[c], f"{what}, armAstar_fitting, {c}") for c in CELLS_03}},
                 "armBprime_fitting": {c: sample(b_cells[c], alloc_bp[c], f"{what}, armBprime_fitting, {c}") for c in CELLS_12},
                 "dev": {c: sample(f["cells_dev"][c], dn, f"{what}, dev, {c}") for c in CELLS_12},
                 "devmix": {c: sample(f["cells_dev"][c], alloc_m[c], f"{what}, devmix, {c}") for c in CELLS if alloc_m[c]}}
        f["manifests"] = {kind: sorted((r for v in cells[kind].values() for r in v), key=lambda r: r["key"]) for kind in MANIFEST_KINDS}
        if f["s12"] & f["dev"] or f["s12"] & f["test"] or f["dev"] & f["test"]:
            fail(f"held-out fold {i}: fitting, dev and test subjects must be disjoint")
        f["test_rows"] = inp["schema_rows"][(i, "test")]
        if len(f["test_rows"]) != test_frames[i - 1]:
            fail(f"held-out fold {i}: {len(f['test_rows'])} test frames, expected {test_frames[i - 1]}")
    return {"k": k, "count": count, "has12": has12, "folds": folds, "n": n, "min_b": min_b, "alloc_a": alloc_a, "alloc_bp": alloc_bp,
            "variant_a": var_a, "caps_a": caps_a, "dev_n": dn, "min_dev": min_dev, "alloc_m": alloc_m, "variant_m": var_m,
            "caps_m": caps_m}


def check_schema_public(inp: dict, des: dict) -> list:
    """Numeri già pubblicati dallo schema approvato (disponibilità per ruolo e cella, totali, test), confrontati con il
    ricalcolo dalle appartenenze: [(id, valore, insieme di soggetti, celle)] per la linea di base della riservatezza."""
    spub, out = inp["schema_public"], []
    for i, f in des["folds"].items():
        pf = spub["per_fold"].get(str(i), {})
        if pf.get("dev_fold") != f["dev_fold"] or pf.get("fitting_folds") != f["fitting_folds"]:
            fail(f"fold_schema_public.json: roles of fold {i} differ from the rule (dev fold k mod K + 1)")
        for role, subs in (("fitting", f["fitting"]), ("dev", f["dev"]), ("test", f["test"])):
            if not {x["subject"] for x in inp["schema_rows"][(i, role)]} <= subs:
                fail(f"fold schema manifest fold{i}_{role}.csv: a subject outside the {role} folds of the rule")
            r = (pf.get("roles") or {}).get(role) or {}
            if r.get("manifest_sha256") != inp["file_sha"][(i, role)]:
                fail(f"fold_schema_public.json: manifest SHA-256 of fold {i}, {role} differs from the restricted manifest")
            av = r.get("availability") or {}
            for c in CELLS:
                v = sum(des["count"][(s, c)] for s in subs)
                if av.get(c) != v:
                    fail(f"fold_schema_public.json: availability of fold {i}, {role}, {c} differs from the recount")
                out.append((f"schema.fold{i}.{role}.{c}", v, set(subs), [c]))
            out.append((f"schema.fold{i}.{role}.all", sum(av[c] for c in CELLS), set(subs), list(CELLS)))
    return out


# ============================================================================ riservatezza (controllo esatto di rango)

class Span:
    """Spazio generato da vettori interi sparsi (colonna -> intero); eliminazione esatta senza frazioni (righe divise per
    il loro MCD), pivot = colonna minima di ogni riga."""

    def __init__(self):
        self.rows = {}

    def reduce(self, v: dict) -> dict:
        v = {c: x for c, x in v.items() if x}
        for p in sorted(self.rows):
            a = v.get(p, 0)
            if a:
                r = self.rows[p]
                b = r[p]
                v = {c: b * v.get(c, 0) - a * r.get(c, 0) for c in set(v) | set(r)}
                v = {c: x for c, x in v.items() if x}
        return v

    def add(self, v: dict) -> bool:
        v = self.reduce(v)
        if not v:
            return False
        g = 0
        for x in v.values():
            g = math.gcd(g, x)
        p = min(v)
        sign = 1 if v[p] > 0 else -1
        self.rows[p] = {c: sign * x // g for c, x in v.items()}
        return True

    def rank(self) -> int:
        return len(self.rows)


class Disclosure:
    """Per ogni soggetto s, d_s = rango(pubblicato) − rango(pubblicato senza le colonne di s): la dimensione delle
    quantità di un solo soggetto ricavabili da combinazioni lineari dei numeri pubblicati."""

    def __init__(self, atoms: dict):
        self.atoms = atoms                      # (soggetto, cella) -> colonna
        self.subjects = sorted({s for s, _ in atoms})
        self.own = {s: {col for (x, _), col in atoms.items() if x == s} for s in self.subjects}
        self.full = Span()
        self.minus = {s: Span() for s in self.subjects}

    def vector(self, subjects, cells) -> dict:
        """Indicatore degli atomi (soggetto, cella) con frame: la quantità «somma dei frame di quei soggetti e celle»."""
        return {self.atoms[(s, c)]: 1 for s in subjects for c in cells if (s, c) in self.atoms}

    def project(self, v: dict, s: str) -> dict:
        return {c: x for c, x in v.items() if c not in self.own[s]}

    def would_reveal(self, v: dict) -> list:
        """Soggetti il cui d_s crescerebbe aggiungendo v (indipendente nel pieno, dipendente senza le colonne di s)."""
        if not self.full.reduce(v):
            return []
        return [s for s in self.subjects if not self.minus[s].reduce(self.project(v, s))]

    def add(self, v: dict) -> None:
        self.full.add(v)
        for s in self.subjects:
            self.minus[s].add(self.project(v, s))

    def leaks(self) -> dict:
        return {s: self.full.rank() - self.minus[s].rank() for s in self.subjects}


def atoms_of(count: Counter) -> dict:
    """Colonne degli atomi soggetto × cella con almeno un frame (un atomo vuoto non porta informazione)."""
    return {a: i for i, a in enumerate(sorted(a for a, v in count.items() if v > 0))}


def quantities(des: dict) -> list:
    """Numeri candidati alla pubblicazione, in ordine di priorità:
    [(id, valore, soggetti, celle)]."""
    out = []
    cnt = des["count"]

    def add(qid, subs, cells):
        out.append((qid, sum(cnt[(s, c)] for s in subs for c in cells), set(subs), list(cells)))

    # priorità: totali del test, poi le celle che decidono la fattibilità (S12 e dev), poi le celle del test, poi i
    # subtotali e la disponibilità di tutti i soggetti del fitting (già pubblica nello schema)
    order = [("test", "test", "all"), ("s12", "s12", CELLS), ("dev", "dev", CELLS), ("test", "test", CELLS),
             ("s12", "s12", "s12tot"), ("s12", "s12", "all"), ("dev", "dev", "s12tot"), ("dev", "dev", "all"),
             ("fitting_all", "fitting", CELLS), ("fitting_all", "fitting", "all")]
    for row, key, cells in order:
        for i, f in des["folds"].items():
            if cells == "s12tot":
                add(f"fold{i}.{row}.sessions_01_02", f[key], CELLS_12)
            elif cells == "all":
                add(f"fold{i}.{row}.all", f[key], CELLS)
            else:
                for c in cells:
                    add(f"fold{i}.{row}.{c}", f[key], [c])
    return out


def decide_publication(des: dict, baseline: list) -> tuple:
    """(decisioni, d_s dello schema): per ogni quantità «published» o il motivo del riserbo. Regola dura: almeno due
    soggetti con frame (o valore 0); regola di rango: nessun d_s cresce rispetto allo schema più i numeri già ammessi.
    Una cella di un solo soggetto che lo schema approvato pubblica già è marcata come tale (resta ricavabile da lì)."""
    atoms = atoms_of(des["count"])
    dis = Disclosure(atoms)
    for _, _, subs, cells in baseline:
        dis.add(dis.vector(subs, cells))
    base_leaks = dis.leaks()
    decisions = {}
    for qid, value, subs, cells in quantities(des):
        with_frames = {s for s in subs if any(des["count"][(s, c)] for c in cells)}
        v = dis.vector(subs, cells)
        if value and len(with_frames) < 2:
            known = not dis.full.reduce(v)
            decisions[qid] = {"value": value, "subjects": len(with_frames), "status": RESTRICTED_NOTE["schema" if known else "one"]}
        elif dis.would_reveal(v):
            decisions[qid] = {"value": value, "subjects": len(with_frames), "status": RESTRICTED_NOTE["diff"]}
        else:
            dis.add(v)
            decisions[qid] = {"value": value, "subjects": len(with_frames), "status": "published"}
        decisions[qid]["vector"] = (subs, cells)
    return decisions, base_leaks


def disclosure_problems(des: dict, baseline: list, decisions: dict) -> list:
    """Verifica indipendente della pubblicazione, da zero: regola dura per ogni numero pubblicato, d_s finale non
    maggiore di quello dello schema per ogni soggetto."""
    atoms = atoms_of(des["count"])
    base, full = Disclosure(atoms), Disclosure(atoms)
    for _, _, subs, cells in baseline:
        base.add(base.vector(subs, cells))
        full.add(full.vector(subs, cells))
    out = []
    for qid, dd in decisions.items():
        if dd["status"] != "published":
            continue
        subs, cells = dd["vector"]
        if dd["value"] and len({s for s in subs if any(des["count"][(s, c)] for c in cells)}) < 2:
            out.append(f"{qid}: a published count of one subject")
        full.add(full.vector(subs, cells))
    b, f = base.leaks(), full.leaks()
    worse = [s for s in b if f[s] > b[s]]
    if worse:
        out.append(f"{len(worse)} subject(s) with a count derivable by differencing from the public numbers")
    return out


# ============================================================================ budget

def budget(cfg: dict, des: dict) -> dict:
    """Stima proporzionale ai frame per epoca (fitting + dev) dai tempi misurati nel run di riferimento; FAIL se il
    limite superiore supera il tetto della configurazione."""
    b, k, seeds = cfg["budget"], des["k"], len(cfg["train_seeds"])
    ref_frames = b["reference_fitting_frames"] + b["reference_dev_frames"]
    per_arm = {}
    total_mid = total_hi = 0.0
    dev_frames = len(CELLS_12) * des["dev_n"]
    fit = {"arm_b": len(CELLS_12) * des["n"], "arm_a_star": sum(des["alloc_a"].values())}
    tr = b["reference_train_seconds"]
    for arm in ARMS:
        ratio = (fit[arm] + dev_frames) / ref_frames
        mid = sum(tr) / len(tr) * ratio * k * seeds
        hi = max(tr) * ratio * k * seeds
        per_arm[arm] = {"fitting_frames": fit[arm], "dev_frames": dev_frames, "ratio_to_reference": round(ratio, 4),
                        "seconds_per_training": [round(min(tr) * ratio, 1), round(max(tr) * ratio, 1)],
                        "training_minutes": [round(mid / 60, 1), round(hi / 60, 1)], "trainings": k * seeds}
        total_mid += mid
        total_hi += hi
    n_ref_passes = len(b["reference_score_seconds"])
    per_frame_score = sum(b["reference_score_seconds"]) / (b["reference_scored_frames_per_seed"] * n_ref_passes / k)
    per_frame_score_hi = max(b["reference_score_seconds"]) * k / b["reference_scored_frames_per_seed"]
    per_dev_frame = sum(b["reference_threshold_seconds"]) / (len(b["reference_threshold_seconds"]) * b["reference_dev_frames"])
    per_dev_frame_hi = max(b["reference_threshold_seconds"]) / b["reference_dev_frames"]
    test_frames = sum(len(f["test_rows"]) for f in des["folds"].values())
    # braccio facoltativo B′: stima a parte, non sommata al budget base
    bp_fit = sum(des["alloc_bp"].values())
    ratio = (bp_fit + dev_frames) / ref_frames
    bp_train = (sum(tr) / len(tr) * ratio * k * seeds, max(tr) * ratio * k * seeds)
    bp_score = (seeds * test_frames * per_frame_score + seeds * k * dev_frames * per_dev_frame,
                seeds * test_frames * per_frame_score_hi + seeds * k * dev_frames * per_dev_frame_hi)
    optional = {"arm_b_prime": {"fitting_frames": bp_fit, "dev_frames": dev_frames, "ratio_to_reference": round(ratio, 4),
                                "trainings": k * seeds,
                                "seconds_per_training": [round(min(tr) * ratio, 1), round(max(tr) * ratio, 1)],
                                "training_minutes": [round(bp_train[0] / 60, 1), round(bp_train[1] / 60, 1)],
                                "scoring_minutes": [round(bp_score[0] / 60, 1), round(bp_score[1] / 60, 1)],
                                "total_minutes": [round((bp_train[0] + bp_score[0]) / 60, 1), round((bp_train[1] + bp_score[1]) / 60, 1)],
                                "note": "optional arm, estimated separately and not added to the base budget"}}
    scored = len(ARMS) * seeds * test_frames
    dev_scored = len(ARMS) * seeds * k * dev_frames
    score_mid = scored * per_frame_score + dev_scored * per_dev_frame
    score_hi = scored * per_frame_score_hi + dev_scored * per_dev_frame_hi
    out = {"trainings": len(ARMS) * k * seeds, "arms": per_arm, "test_frames_scored": scored, "dev_frames_scored": dev_scored,
           "scoring_minutes": [round(score_mid / 60, 1), round(score_hi / 60, 1)],
           "total_minutes": [round((total_mid + score_mid) / 60, 1), round((total_hi + score_hi) / 60, 1)],
           "optional": optional, "wall_time_cap_minutes": b["wall_time_cap_minutes"], "reference_run": b["reference_run"],
           "reference_run_record_sha256": b["reference_run_record_sha256"]}
    if out["total_minutes"][1] > b["wall_time_cap_minutes"]:
        fail(f"budget: upper estimate {out['total_minutes'][1]} min exceeds the cap of {b['wall_time_cap_minutes']} min")
    return out


# ============================================================================ uscite

def csv_text(header, rows) -> str:
    return NF.csv_text(header, rows)


def subjects_per_cell(rows) -> dict:
    by = defaultdict(set)
    for r in rows:
        by[f"{r['session']}/{CLASS_NAME[r['label']]}"].add(r["subject"])
    return {c: len(by[c]) for c in CELLS}


def frames_per_cell(rows) -> dict:
    c = Counter(f"{r['session']}/{CLASS_NAME[r['label']]}" for r in rows)
    return {x: c[x] for x in CELLS}


def render(dd: dict, at_least=None) -> str:
    if dd["status"] == "published":
        return str(dd["value"])
    tag = RESTRICTED_TAG[dd["status"]]
    return f">={at_least} {tag}" if at_least else tag


def public_cell(dd: dict, at_least=None) -> dict:
    out = {"subjects": dd["subjects"], "status": dd["status"],
           "frames": dd["value"] if dd["status"] == "published" else None}
    if dd["status"] != "published" and at_least:
        out["at_least"] = at_least
    return out


def wrap(text: str, width: int = 118) -> list:
    """Righe di al massimo `width` caratteri (le righe dei .md restano entro 120)."""
    out, line = [], ""
    for w in text.split():
        if line and len(line) + 1 + len(w) > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}" if line else w
    if line:
        out.append(line)
    return out


def build_outputs(cfg: dict, cfg_sha: str, inp: dict, des: dict, decisions: dict, base_leaks: dict, bud: dict,
                  state: dict, run_id: str) -> tuple:
    """(file pubblici, file riservati) come testi. Il pubblico non contiene identificativi né conteggi trattenuti."""
    k, cnt = des["k"], des["count"]
    restricted, per_fold = {}, {}
    subj_rows = []
    for i, f in des["folds"].items():
        for s in sorted(f["test"] | f["dev"] | f["fitting"]):
            role = "test" if s in f["test"] else "dev" if s in f["dev"] else "fitting"
            fold_of = next(j for j, m in inp["members"].items() if s in m)
            subj_rows.append([i, s, fold_of, role, 1 if s in f["s12"] else 0, 1 if s in des["has12"] else 0]
                             + [cnt[(s, c)] for c in CELLS])
    restricted["subjects.csv"] = csv_text(["held_out_fold", "subject", "subject_fold", "role", "s12", "sessions_01_02"]
                                          + [c.replace("/", "_") for c in CELLS], subj_rows)
    kind_role = {"armB_fitting": "arm_b_fitting", "armAstar_fitting": "arm_a_star_fitting",
                 "armBprime_fitting": "arm_b_prime_fitting_optional", "dev": "dev_common",
                 "devmix": "dev_mix_alternative"}
    in_cache = {r["key"]: r["in_cache"] for f in des["folds"].values() for r in f["test_rows"]}
    for i, f in des["folds"].items():
        mans = {}
        for kind in MANIFEST_KINDS:
            rows = f["manifests"][kind]
            text = csv_text(NF.MANIFEST_HEADER, [[i, kind_role[kind], r["key"], r["label"], r["subject"], r["session"], r["sha256"],
                                                  r["bytes"], in_cache[r["key"]]] for r in rows])
            restricted[f"{kind}_fold{i}.csv"] = text
            mans[kind_role[kind]] = {"manifest": f"{kind}_fold{i}.csv", "manifest_sha256": sha256_bytes(text.encode("utf-8")),
                                     "frames": frames_per_cell(rows), "subjects_per_cell": subjects_per_cell(rows),
                                     "total": len(rows), "subjects": len({r["subject"] for r in rows})}
        mans["test_unchanged"] = {"manifest": f"fold{i}_test.csv of the fold schema", "manifest_sha256": inp["file_sha"][(i, "test")],
                                  "total": len(f["test_rows"])}
        ref = {}
        for role in ("fitting", "dev"):
            rows = inp["schema_rows"][(i, role)]
            ref[role] = {"frames": frames_per_cell(rows), "subjects_per_cell": subjects_per_cell(rows), "total": len(rows),
                         "subjects": len({r["subject"] for r in rows}), "manifest_sha256": inp["file_sha"][(i, role)]}
        need = {"s12": {**{c: des["n"] for c in CELLS_12}, **{c: des["alloc_a"][c] for c in CELLS_03}},
                "dev": {**{c: des["dev_n"] for c in CELLS_12}, **{c: des["alloc_m"][c] for c in CELLS_03}}}
        avail = {}
        for row in ("s12", "dev", "fitting_all", "test"):
            avail[row] = {}
            for c in CELLS + ["sessions_01_02", "all"]:
                qid = f"fold{i}.{row}.{c}"
                if qid in decisions:
                    avail[row][c] = public_cell(decisions[qid], need.get(row, {}).get(c))
        feas = {"arm_b": all(len(f["cells_s12"][c]) >= des["n"] for c in CELLS_12),
                "arm_a_star": all(len(f["cells_s12"][c]) >= des["alloc_a"][c] for c in CELLS),
                "dev_common": all(len(f["cells_dev"][c]) >= des["dev_n"] for c in CELLS_12),
                "dev_mix_alternative": all(len(f["cells_dev"][c]) >= des["alloc_m"][c] for c in CELLS)}
        per_fold[str(i)] = {
            "held_out_fold": i, "dev_fold": f["dev_fold"], "fitting_folds": f["fitting_folds"],
            "subjects": {"test": len(f["test"]), "dev": len(f["dev"]), "dev_with_sessions_01_02": len(f["dev12"]),
                         "fitting": len(f["fitting"]), "s12": len(f["s12"]),
                         "s12_with_bona_fide_03": sum(1 for s in f["s12"] if cnt[(s, "03/bona_fide")]),
                         "s12_with_attack_03": sum(1 for s in f["s12"] if cnt[(s, "03/attack")]),
                         "fitting_without_sessions_01_02": len(f["s03only"])},
            "availability": avail, "feasible": feas, "proposed": mans, "arm_a_reference": ref}
    restricted["counts_restricted.json"] = json.dumps({
        "run_id": run_id, "per_fold_subjects": {str(i): {x: sorted(f[x]) for x in ("test", "dev", "dev12", "fitting", "s12", "s03only")}
                                                for i, f in des["folds"].items()},
        "quantities": {q: {"value": dd["value"], "subjects": sorted(dd["vector"][0]), "cells": dd["vector"][1], "status": dd["status"]}
                       for q, dd in decisions.items()},
        "smallest_availability": {"arm_b": des["min_b"], "arm_a_star_caps": des["caps_a"], "dev": des["min_dev"], "dev_mix_caps": des["caps_m"]},
        "schema_baseline_leaks": base_leaks}, indent=1) + "\n"
    leaky = sum(1 for v in base_leaks.values() if v > 0)
    held = [q for q, dd in decisions.items() if dd["status"] != "published"]
    d = cfg["design"]
    doc = {
        "schema": "nuaa-session-ablation/1", "run_id": run_id, "status": cfg["status"], "task": cfg["task"],
        "config_sha256": cfg_sha, "commit": state["commit"], "dirty": state["dirty"], "code_sha256": state.get("code_sha256", {}),
        "inputs": {"folds_config_sha256": inp["folds_config_sha256"], "fold_schema": cfg["inputs"]["fold_schema"]},
        "design": {"s12": d["s12"], "dev_subjects": d["dev_subjects"], "rounding_step": d["rounding_step"],
                   "rounding_rule": d["rounding_rule"], "allocation_rule": d["allocation_rule"], "remainder_order": d["remainder_order"],
                   "sampling": d["sampling"], "nesting": d["nesting"], "arm_b_per_cell": des["n"], "arm_b_total": len(CELLS_12) * des["n"],
                   "arm_b_prime_allocation_optional": des["alloc_bp"], "arm_b_prime_total_optional": sum(des["alloc_bp"].values()),
                   "arm_a_star_allocation": des["alloc_a"], "arm_a_star_variant": des["variant_a"],
                   "arm_a_star_total": sum(des["alloc_a"].values()), "dev_per_cell": des["dev_n"],
                   "dev_total": len(CELLS_12) * des["dev_n"], "dev_mix_allocation": des["alloc_m"], "dev_mix_variant": des["variant_m"],
                   "test": "unchanged from the fold schema: every frame of the held-out subjects",
                   "test_frames": [len(f["test_rows"]) for f in des["folds"].values()]},
        "arms": cfg["arms"], "per_fold": per_fold, "train_seeds": cfg["train_seeds"], "training": cfg["training"],
        "threshold": cfg["threshold"], "metrics": cfg["metrics"], "budget": bud,
        "disclosure": {"rule": "a frame count is public only if its cell holds at least two subjects and, with the numbers "
                               "already public in the fold schema and those admitted here, it makes no new single-subject "
                               "quantity derivable by differencing (exact rank check on subject x session x class atoms); "
                               "a cell of one subject that the fold schema already prints is not printed here and stays "
                               "derivable from the schema (R1s)",
                       "restricted_numbers": len(held),
                       "schema_subjects_with_derivable_counts": leaky,
                       "note": "R1 = cell of one subject; R1s = cell of one subject already printed in the fold schema; Rd = derivable "
                               "by differencing; >=n = at least the count the design needs"},
        "declarations": cfg["declarations"], "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    public = {"session_ablation_public.json": json.dumps(doc, indent=1) + "\n", "session_ablation_public.md": markdown(cfg, doc, decisions, des)}
    return public, restricted


def markdown(cfg: dict, doc: dict, decisions: dict, des: dict) -> str:
    dz = doc["design"]
    md = [f"# NUAA training with and without session 03: available counts (run `{doc['run_id']}`)", ""]
    md += wrap(f"Status: {cfg['status']}") + [""] + wrap(cfg["task"]) + [""]
    alloc = ", ".join(f"{c} {dz['arm_a_star_allocation'][c]}" for c in CELLS)
    mix = ", ".join(f"{c} {dz['dev_mix_allocation'][c]}" for c in CELLS)
    md += ["## Design", ""]
    md += wrap(f"- Arm B (without session 03): {dz['arm_b_per_cell']} frames per cell over 01/bona_fide, 01/attack, "
               f"02/bona_fide, 02/attack of the S12 subjects, {dz['arm_b_total']} in all ({dz['rounding_rule']}, step "
               f"{dz['rounding_step']}).")
    md += wrap(f"- Arm A* (with session 03, same S12 subjects): {dz['arm_a_star_total']} frames, {dz['arm_a_star_variant']}: "
               f"{alloc}.")
    bp = ", ".join(f"{c} {dz['arm_b_prime_allocation_optional'][c]}" for c in CELLS_12)
    md += wrap(f"- Arm B′ (optional control of the per-cell size): sessions 01-02 of the S12 subjects with the per-cell "
               f"counts of arm A*: {bp} ({dz['arm_b_prime_total_optional']} in all).")
    md += wrap(f"- Nesting: {dz['nesting']}.")
    md += wrap("- Arm A (reference, already run): 350 frames per cell, six cells, all eight fitting subjects.")
    md += wrap(f"- Dev (common to both arms): {dz['dev_per_cell']} frames per cell over the four session 01-02 cells of "
               f"the dev subjects, {dz['dev_total']} in all; alternative with the session mix of arm A*: "
               f"{dz['dev_mix_variant']}, {mix}.")
    md += wrap("- Test: unchanged from the fold schema, every frame of the held-out subjects: "
               + ", ".join(str(x) for x in dz["test_frames"]) + f" ({sum(dz['test_frames'])}).")
    md += [""] + wrap(f"Notation: {doc['disclosure']['note']}. Rule: {doc['disclosure']['rule']}.") + [""]
    cols = CELLS + ["sessions_01_02", "all"]
    head = "| row | " + " | ".join(c.replace("bona_fide", "bf").replace("attack", "att").replace("sessions_", "")
                                     for c in cols) + " |"
    for key, pf in doc["per_fold"].items():
        i = int(key)
        sj = pf["subjects"]
        md += [f"## Held-out fold {i} (dev fold {pf['dev_fold']}, fitting folds {pf['fitting_folds'][0]} and "
               f"{pf['fitting_folds'][1]})", ""]
        md += wrap(f"S12: {sj['s12']} subjects ({sj['s12_with_bona_fide_03']} with bona fide in session 03, "
                   f"{sj['s12_with_attack_03']} with attacks in session 03); fitting subjects without sessions 01-02 "
                   f"(arm A only): {sj['fitting_without_sessions_01_02']}; dev subjects with sessions 01-02: "
                   f"{sj['dev_with_sessions_01_02']} of {sj['dev']}; test subjects: {sj['test']}. Feasible: "
                   + ", ".join(f"{a} {'yes' if v else 'NO'}" for a, v in pf["feasible"].items()) + ".")
        md += ["", head, "|---|" + "---|" * len(cols)]

        def av_row(label, row, need):
            vals = []
            for c in cols:
                dd = decisions.get(f"fold{i}.{row}.{c}")
                vals.append("-" if dd is None else render(dd, need.get(c)))
            return f"| {label} | " + " | ".join(vals) + " |"

        def subj_row(label, subs):
            return f"| {label} | " + " | ".join(str(sum(1 for s in subs if des["count"][(s, c)])) for c in CELLS) + " | | |"

        def sample_row(label, m):
            fr, sp = m["frames"], m["subjects_per_cell"]
            vals = [f"{fr[c]} ({sp[c]})" if fr[c] else "-" for c in CELLS]
            return f"| {label} | " + " | ".join(vals) + f" | {sum(fr[c] for c in CELLS_12)} | {m['total']} |"

        f = des["folds"][i]
        need_s12 = {**{c: des["n"] for c in CELLS_12}, **{c: des["alloc_a"][c] for c in CELLS_03}}
        need_dev = {**{c: des["dev_n"] for c in CELLS_12}, **{c: des["alloc_m"][c] for c in CELLS_03}}
        pr_ = pf["proposed"]
        md += [subj_row("S12 subjects with frames", f["s12"]), av_row("S12 available", "s12", need_s12),
               sample_row("arm B fitting (subj.)", pr_["arm_b_fitting"]),
               sample_row("arm A* fitting (subj.)", pr_["arm_a_star_fitting"]),
               sample_row("arm B′ (optional) (subj.)", pr_["arm_b_prime_fitting_optional"]),
               sample_row("arm A fitting, run (subj.)", pf["arm_a_reference"]["fitting"]),
               av_row("all fitting subjects available", "fitting_all", {}),
               subj_row("dev subjects with frames", f["dev"]), av_row("dev available", "dev", need_dev),
               sample_row("dev common (subj.)", pr_["dev_common"]), sample_row("dev mix, alternative (subj.)", pr_["dev_mix_alternative"]),
               av_row("test (all frames)", "test", {}), ""]
    b = doc["budget"]
    md += ["## Budget", ""]
    md += wrap(f"{b['trainings']} trainings (2 arms x {des['k']} folds x {len(doc['train_seeds'])} seeds). Per training: "
               + "; ".join(f"{a} {v['fitting_frames']} fitting + {v['dev_frames']} dev frames, {v['seconds_per_training'][0]}-"
                           f"{v['seconds_per_training'][1]} s, {v['training_minutes'][0]}-{v['training_minutes'][1]} min for its "
                           f"{v['trainings']} trainings" for a, v in b["arms"].items())
               + f". Scoring: {b['test_frames_scored']} test frames (each model scores its held-out fold only) and "
               f"{b['dev_frames_scored']} dev frames, {b['scoring_minutes'][0]}-{b['scoring_minutes'][1]} min. Total "
               f"{b['total_minutes'][0]}-{b['total_minutes'][1]} min against a cap of {b['wall_time_cap_minutes']} min; "
               f"times scaled by frames per epoch from the reference run {b['reference_run']}.")
    o = b["optional"]["arm_b_prime"]
    md += [""] + wrap(f"Optional arm B′ (not in the base budget): {o['trainings']} trainings of {o['fitting_frames']} fitting + "
                      f"{o['dev_frames']} dev frames, {o['seconds_per_training'][0]}-{o['seconds_per_training'][1]} s each, "
                      f"{o['training_minutes'][0]}-{o['training_minutes'][1]} min of training, {o['scoring_minutes'][0]}-"
                      f"{o['scoring_minutes'][1]} min of scoring, {o['total_minutes'][0]}-{o['total_minutes'][1]} min in all.")
    md += [""] + wrap(f"Restricted numbers in this document: {doc['disclosure']['restricted_numbers']}; subjects with a "
                      f"single-subject quantity already derivable from the public fold schema: "
                      f"{doc['disclosure']['schema_subjects_with_derivable_counts']} (this document adds none).") + [""]
    return "\n".join(md)


def sums_text(files: dict) -> str:
    return NF.sums_text(files)


def public_problems(public: dict, subjects, keys) -> list:
    """Identificativi o contenuto vietato nel pubblico (regole dello schema dei fold)."""
    return NF.public_problems(public, subjects, keys)


# ============================================================================ run

def repo_state(root: Path) -> dict:
    if REPO_STATE is not None:
        return REPO_STATE(root)
    old = NF.REPO_STATE
    try:
        NF.REPO_STATE = None
        return NF.repo_state(root)
    finally:
        NF.REPO_STATE = old


def run_id_for(date: str, state: dict) -> str:
    if not re.fullmatch(r"\d{8}", date or ""):
        fail("--date must be YYYYMMDD")
    rid = f"{date}-SESSABL-{state['commit']}" + ("-dev" if state["dirty"] else "")
    if not RUN_ID_RE.match(rid):
        fail(f"run id {rid} does not follow YYYYMMDD-SESSABL-<commit>[-dev]")
    return rid


def protected_extra(root: Path) -> list:
    return [root / x for x in EXTRA_PROTECTED]


def run_dir_problem(run: Path, root: Path, inputs: dict, k: int, out_root: Path = None) -> str:
    """Motivo per cui la cartella del run non è ammessa ("" se lo è), prima di qualunque modifica."""
    if (out_root if out_root is not None else run.parent).is_symlink():
        return "the output root is a symbolic link"
    for label, p in inputs.items():
        if pg.within(p, run):
            return f"{label} {Path(p).name} is inside the run folder: inputs and outputs must be separate"
        if pg.within(run, p if Path(p).is_dir() else Path(p).parent):
            return f"the run folder is inside the folder of {label}: inputs and outputs must be separate"
    msg = mw.output_problem(run, root, pg.BASE_PROTECTED, [root / DEFAULT_OUT_ROOT], protected_extra(root), inputs, [STATUS],
                            [run / PUBLIC, run / RESTRICTED])
    if msg:
        return msg
    for sub, names in ((PUBLIC, PUBLIC_OUTPUTS), (RESTRICTED, restricted_outputs(k))):
        if msg := mw.output_problem(run / sub, root, pg.BASE_PROTECTED, [root / DEFAULT_OUT_ROOT], protected_extra(root), inputs, list(names)):
            return msg
    allowed = {"": [STATUS, PUBLIC, RESTRICTED], PUBLIC: PUBLIC_OUTPUTS, RESTRICTED: restricted_outputs(k)}
    for sub, names in allowed.items():
        dd = run / sub if sub else run
        if dd.is_dir():
            for e in sorted(os.listdir(dd)):
                if not any(mw.owned_name(e, n) for n in names):
                    return f"{(Path(sub) / e).as_posix()} in the run folder is not an output of this command: refused, nothing removed"
    return ""


def write_status(run: Path, doc: dict) -> None:
    mw.write_all(run, {STATUS: json.dumps(doc, indent=1) + "\n"})


def invalidate_outputs(run: Path, k: int) -> list:
    left = []
    for sub, names in ((PUBLIC, PUBLIC_OUTPUTS), (RESTRICTED, restricted_outputs(k))):
        dd = run / sub
        if dd.is_dir():
            left += [f"{sub}/{n}" for n in mw.invalidate_each(dd, list(names), None, FAIL_DOC)]
    return left


def code_sha256(root: Path) -> dict:
    return {f: sha256_bytes(NF.read_regular(root / f, "code file")) for f in CODE_FILES}


def resolve(root: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else root / q


def compute(cfg: dict, root: Path, a) -> tuple:
    inp = load_inputs(cfg, root, resolve(root, a.manifests).resolve(), resolve(root, a.folds_root).resolve())
    des = build_design(cfg, inp)
    baseline = check_schema_public(inp, des)
    decisions, base_leaks = decide_publication(des, baseline)
    if problems := disclosure_problems(des, baseline, decisions):
        fail("public counts refused: " + "; ".join(problems))
    bud = budget(cfg, des)
    return inp, des, baseline, decisions, base_leaks, bud


def cmd_check_inputs(a) -> int:
    root = ROOT
    cfg, cfg_sha = load_config(resolve(root, a.config))
    inp, des, _, decisions, base_leaks, bud = compute(cfg, root, a)
    s12 = ", ".join(f"{len(f['s12'])}/{sum(1 for s in f['s12'] if des['count'][(s, '03/bona_fide')])}" for f in des["folds"].values())
    print(f"CHECK-INPUTS: configuration {cfg_sha[:12]}, fold schema {cfg['inputs']['fold_schema']['run_id']} verified, "
          f"{des['k']} folds; S12 subjects / with bona fide 03 per fold: {s12}; arm B {des['n']} per cell; arm A* "
          f"{des['variant_a']} " + ",".join(str(des["alloc_a"][c]) for c in CELLS) + f"; dev {des['dev_n']} per cell; "
          f"{sum(1 for d in decisions.values() if d['status'] != 'published')} numbers restricted; budget "
          f"{bud['total_minutes'][0]}-{bud['total_minutes'][1]} min; nothing written")
    print("RESULT: PASS (check-inputs)")
    return 0


def cmd_run(a) -> int:
    """Configurazione, stato git, ammissione della cartella, RUNNING e invalidazione prima di leggere gli ingressi."""
    if not a.i_am_authorized:
        fail("run writes the run folder: pass --i-am-authorized (counts only, for the protocol proposal asked by the coordinator on 4 November 2026)")
    root = ROOT
    cfg, cfg_sha = load_config(resolve(root, a.config))
    k = cfg["expected"]["k"]
    state = repo_state(root)
    state["code_sha256"] = code_sha256(root)
    run_id = run_id_for(a.date or time.strftime("%Y%m%d"), state)
    out_root = resolve(root, a.out_root)
    run = out_root.resolve() / run_id
    mdir = resolve(root, a.manifests).resolve()
    froot = resolve(root, a.folds_root).resolve()
    inputs = {"configuration": resolve(root, a.config).resolve(), "manifests": mdir,
              "fold schema": froot / cfg["inputs"]["fold_schema"]["run_id"],
              "fold schema configuration": resolve(root, cfg["inputs"]["folds_config"]["path"]).resolve()}
    if msg := run_dir_problem(run, root, inputs, k, out_root):
        fail(msg)
    run.mkdir(parents=True, exist_ok=True)
    write_status(run, {"status": "RUNNING", "run_id": run_id, "config_sha256": cfg_sha, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    left = invalidate_outputs(run, k)
    if left:
        write_status(run, {"status": "FAIL", "run_id": run_id, "reason": f"previous outputs could not be removed: {left}"})
        fail(f"previous outputs could not be removed: {left}")
    try:
        inp, des, baseline, decisions, base_leaks, bud = compute(cfg, root, a)
        public, restricted = build_outputs(cfg, cfg_sha, inp, des, decisions, base_leaks, bud, state, run_id)
        keys = set(inp["official"]) | {x.split("/", 1)[1] for x in inp["official"]}
        problems = public_problems(public, sorted({r["subject"] for r in inp["official"].values()}), keys)
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
    print(f"RUN: {run_id}: {len(public)} public and {len(restricted)} restricted files in {run}")
    print("RESULT: PASS (run)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="NUAA training with and without session 03: available counts (no training, no inference)")
    ap.add_argument("cmd", choices=["check-inputs", "run"])
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="configuration JSON (relative to the repository root)")
    ap.add_argument("--manifests", default="results/m2/manifests", help="folder of the M2 NUAA manifests")
    ap.add_argument("--folds-root", default=DEFAULT_FOLDS_ROOT, help="folder that holds the fold schema run")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="folder that receives <run-id>/ (default results/sessionablation)")
    ap.add_argument("--date", default=None, help="YYYYMMDD of the run id (default: today)")
    ap.add_argument("--i-am-authorized", action="store_true", help="required by run: it writes the run folder")
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
