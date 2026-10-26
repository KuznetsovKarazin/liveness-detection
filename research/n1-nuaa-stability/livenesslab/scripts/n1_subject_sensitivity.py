"""
N1, parte 1: sensibilità del risultato M1 ai soggetti del test NUAA (scheda N1 del docente).

Analisi di sensibilità dei risultati esistenti: nessun training, nessuna inferenza, nessuna nuova soglia. Legge solo
file già consegnati e accettati, verificati per SHA-256 prima dell'uso:
  - gli score C1 delle 300 immagini di test (scores.csv del run C1 accettato, soglia fissa 0,5) e il manifest C1;
  - scores_test_dev_threshold.csv del run M1 accettato (stesse immagini, decise alla soglia M1 congelata), con
    threshold.json (le quattro soglie congelate), la tabella M1 accettata (nuaa_test_dev_threshold_table.md) e il
    manifest del dev M1 (solo per riprodurre il blocco di sensibilità già presente in M1, senza i due soggetti dev).
Ogni file d'ingresso deve avere lo SHA-256 della configurazione e comparire con lo stesso valore nel SHA256SUMS della
sua cartella, il cui SHA-256 è anch'esso fissato nella configurazione; altrimenti FAIL prima di qualunque calcolo.

Regola di decisione (metrics.md, M1): attacco se e solo se score_attack > soglia; un pareggio è bona fide. FA = attacchi
accettati (score <= soglia), FR = bona fide rifiutati (score > soglia); APCER = FA / n_attack, BPCER = FR / n_bona_fide,
ACER = (APCER + BPCER) / 2. Una classe assente dà NA (mai 0) per la sua metrica, per l'ACER e per il ΔACER.

Per ciascuna delle quattro CNN NUAA:
  - per soggetto: n bona fide, n attacchi, FA e FR a 0,5 e alla soglia M1 (APCER e BPCER del soggetto, NA se la classe
    manca);
  - ΔACER = ACER_M1 − ACER_0,5 sull'insieme completo, che deve riprodurre esattamente la tabella M1 accettata (conteggi
    esatti e valori a quattro decimali; anche il blocco senza i soggetti dev), altrimenti FAIL;
  - ΔACER escludendo dal solo calcolo un soggetto alla volta (16 valori): intervallo, cambi di segno rispetto al ΔACER
    completo, denominatori per classe. Le soglie restano quelle M1: nessuna ricalibrazione sul test.
I calcoli usano frazioni esatte (fractions.Fraction): il segno di un ΔACER non dipende dall'arrotondamento.

Uscite (cartella del run <out-root>/YYYYMMDD-N1-subjects-<commit>[-dev], -dev se l'albero git non è pulito):
  status.json                      RUNNING all'avvio, poi PASS o FAIL (unico stato corrente)
  restricted/                      identificativi dei soggetti: area riservata, mai nella PR
    n1_per_subject.csv, n1_loso.csv, n1_per_image.csv, n1_subject_sensitivity_restricted.json, .md,
    public_SHA256SUMS.sha256 (ancora della cartella pubblica), SHA256SUMS
  public/                          solo aggregati, nessun identificativo di soggetto
    n1_subject_sensitivity_public.json, n1_subject_sensitivity_public.md, SHA256SUMS
Dentro il repository la cartella del run è ammessa solo sotto results/n1/; fuori (per esempio scratch) dove path_guard
lo consente. Cartella di uscita e ingressi validati per identità prima di qualunque modifica (m2_writers.output_problem);
scritture atomiche (m2_writers.write_all); dopo l'ammissione le uscite precedenti sono tolte subito (nessun PASS vecchio
accanto a un'esecuzione che fallisce) e su qualunque errore restano solo status.json = FAIL e nessuna uscita.

  .venv/bin/python scripts/n1_subject_sensitivity.py check-inputs     # solo verifica degli ingressi, nessuna scrittura
  .venv/bin/python scripts/n1_subject_sensitivity.py run              # calcolo e uscite in results/n1/<run-id>/

Libreria standard più scripts/m2_writers.py e scripts/path_guard.py; nessuna dipendenza esterna.
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
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import m2_writers as mw  # noqa: E402
import path_guard as pg  # noqa: E402

DEFAULT_CONFIG = "delivery/n1/subject_sensitivity/n1_subject_sensitivity.json"
DEFAULT_OUT_ROOT = "results/n1"
CONFIG_SCHEMA = "n1-subject-sensitivity-config/1"
RUN_ID_RE = re.compile(r"^\d{8}-N1-subjects-[0-9a-f]{7,40}(-dev)?$")
INPUT_KEYS = ("c1_scores", "c1_manifest", "m1_scores_test_dev_threshold", "m1_threshold", "m1_table", "m1_dev_manifest")
CODE_FILES = ("scripts/n1_subject_sensitivity.py", "scripts/m2_writers.py", "scripts/path_guard.py")
DECISION_RULE = "attack if score_attack > threshold (tie = bona fide)"
NA = "NA"

STATUS = "status.json"
PUBLIC, RESTRICTED = "public", "restricted"
PUBLIC_OUTPUTS = ("n1_subject_sensitivity_public.json", "n1_subject_sensitivity_public.md", "SHA256SUMS")
RESTRICTED_OUTPUTS = ("n1_per_subject.csv", "n1_loso.csv", "n1_per_image.csv", "n1_subject_sensitivity_restricted.json",
                      "n1_subject_sensitivity_restricted.md", "public_SHA256SUMS.sha256", "SHA256SUMS")
FAIL_DOC = {"stage": "invalidation-failed", "note": "this output could not be removed after a failure: no verdict is valid"}

# cartelle protette con percorso assoluto (dati, pesi, risultati consegnati, export, pacchetti), in entrambe le direzioni
EXTRA_PROTECTED = ("data", "models", "results/c1", "results/m1", "results/m2", "results/m2_rgb", "results/eval", "export")
REPO_STATE = None       # (root) -> {"commit", "dirty"}; None: git nel repository (sostituibile solo dai test)


class Failure(Exception):
    """Rifiuto o errore con motivo esplicito (riportato in status.json)."""


def fail(msg: str):
    raise Failure(msg)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ============================================================================ configurazione e ingressi

def load_config(p: Path) -> tuple:
    """(configurazione, SHA-256 dei byte letti). Campi obbligatori controllati qui, prima di qualunque altra cosa."""
    data = read_regular(p, "configuration")
    try:
        cfg = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        fail(f"configuration {p.name} is not valid JSON: {e}")
    if not isinstance(cfg, dict) or cfg.get("schema") != CONFIG_SCHEMA:
        fail(f"configuration {p.name}: schema must be {CONFIG_SCHEMA!r}")
    an = cfg.get("analyzers")
    if not isinstance(an, list) or not an or len(set(an)) != len(an) or not all(isinstance(x, str) and x for x in an):
        fail("configuration: analyzers must be a non-empty list of distinct names")
    if cfg.get("fixed_threshold") != 0.5:
        fail("configuration: fixed_threshold must be 0.5 (the C1 and M1 table threshold)")
    if cfg.get("decision_rule") != DECISION_RULE:
        fail(f"configuration: decision_rule must be {DECISION_RULE!r}")
    exp = cfg.get("expected")
    need = ("n_images", "n_bona_fide", "n_attack", "n_subjects", "n_dev_subjects")
    if not isinstance(exp, dict) or not all(isinstance(exp.get(k), int) and not isinstance(exp.get(k), bool) and exp[k] >= 0 for k in need):
        fail(f"configuration: expected must give the non-negative integers {', '.join(need)}")
    inputs = cfg.get("inputs")
    if not isinstance(inputs, dict) or sorted(inputs) != sorted(INPUT_KEYS):
        fail(f"configuration: inputs must have exactly the keys {', '.join(INPUT_KEYS)}")
    for k, v in inputs.items():
        if not isinstance(v, dict) or not all(isinstance(v.get(f), str) and v[f] for f in ("path", "sha256", "sums", "sums_sha256")):
            fail(f"configuration: input {k} needs path, sha256, sums and sums_sha256")
        for f in ("sha256", "sums_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", v[f]):
                fail(f"configuration: input {k}: {f} is not a SHA-256")
        for f in ("path", "sums"):
            if Path(v[f]).is_absolute() or ".." in Path(v[f]).parts:
                fail(f"configuration: input {k}: {f} must be a relative path without '..'")
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


def input_paths(cfg: dict, base: Path) -> dict:
    """Etichetta -> percorso di ogni file d'ingresso e di ogni SHA256SUMS (per la validazione per identità)."""
    out = {}
    for k in INPUT_KEYS:
        out[f"input {k}"] = base / cfg["inputs"][k]["path"]
        out[f"SHA256SUMS of {k}"] = base / cfg["inputs"][k]["sums"]
    return out


def sums_entry(text: str, name: str) -> str:
    """SHA-256 della riga di SHA256SUMS per `name` (formato shasum, anche con '*'); None se assente o ripetuta."""
    hits = []
    for line in text.splitlines():
        m = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        if m and m.group(2) == name:
            hits.append(m.group(1))
    return hits[0] if len(hits) == 1 else None


def verify_inputs(cfg: dict, base: Path) -> dict:
    """Byte verificati di ogni ingresso: SHA-256 del file = configurazione = riga del SHA256SUMS della sua cartella, e
    SHA-256 del SHA256SUMS = configurazione. Qualunque differenza: FAIL (nessun calcolo)."""
    data = {}
    for k in INPUT_KEYS:
        spec = cfg["inputs"][k]
        p, sp = base / spec["path"], base / spec["sums"]
        b = read_regular(p, f"input {k}")
        h = sha256_bytes(b)
        if h != spec["sha256"]:
            fail(f"input {k} ({p.name}): SHA-256 mismatch: {h} instead of the accepted {spec['sha256']}")
        sb = read_regular(sp, f"SHA256SUMS of {k}")
        sh = sha256_bytes(sb)
        if sh != spec["sums_sha256"]:
            fail(f"SHA256SUMS of {k}: SHA-256 mismatch: {sh} instead of the accepted {spec['sums_sha256']}")
        try:
            rel = p.resolve().relative_to(sp.resolve().parent).as_posix()
        except ValueError:
            fail(f"input {k}: {p.name} is not inside the folder of its SHA256SUMS")
        listed = sums_entry(sb.decode("utf-8", errors="replace"), rel)
        if listed is None:
            fail(f"input {k}: {rel} is not listed exactly once in its SHA256SUMS")
        if listed != h:
            fail(f"input {k}: SHA-256 mismatch with its SHA256SUMS ({listed} listed, {h} on disk)")
        data[k] = b
    return data


# ============================================================================ lettura delle righe

def parse_csv(b: bytes, what: str, columns) -> list:
    try:
        text = b.decode("utf-8")
    except UnicodeDecodeError:
        fail(f"{what} is not UTF-8")
    r = csv.DictReader(io.StringIO(text, newline=""))
    if r.fieldnames is None or not set(columns) <= set(r.fieldnames):
        fail(f"{what}: missing columns {sorted(set(columns) - set(r.fieldnames or []))}")
    return list(r)


def parse_score(text: str, what: str) -> float:
    try:
        v = float(text)
    except (TypeError, ValueError):
        fail(f"{what}: score {text!r} is not a number")
    if v != v or v in (float("inf"), float("-inf")):
        fail(f"{what}: score {text!r} is not finite")
    return v


def parse_label(text: str, what: str) -> int:
    if text not in ("0", "1"):
        fail(f"{what}: label {text!r} is not 0 or 1")
    return int(text)


SCORE_COLUMNS = ("sample_id", "dataset_id", "split", "subject_id", "label", "score_attack", "threshold", "prediction", "analyzer_id")


def score_rows(b: bytes, what: str, analyzers, exp: dict, thresholds: dict) -> dict:
    """analizzatore -> {sample_id: (subject, label, score_text, score)} delle CNN richieste, con i controlli: righe per
    analizzatore, chiavi uniche, dataset e split, soglia della riga uguale a quella attesa, previsione salvata uguale
    alla regola score > soglia ricalcolata."""
    rows = parse_csv(b, what, SCORE_COLUMNS)
    out = {a: {} for a in analyzers}
    for i, r in enumerate(rows, 2):
        a = r["analyzer_id"]
        if a not in out:
            continue
        where = f"{what} line {i}"
        sid = r["sample_id"]
        if sid in out[a]:
            fail(f"{where}: duplicated sample {sid} for {a}")
        if r["dataset_id"] != exp.get("dataset_id", r["dataset_id"]) or r["split"] != exp.get("split", r["split"]):
            fail(f"{where}: dataset or split differs from the configuration")
        if not r["subject_id"]:
            fail(f"{where}: empty subject_id")
        lab = parse_label(r["label"], where)
        s = parse_score(r["score_attack"], where)
        t = parse_score(r["threshold"], where)
        if t != thresholds[a]:
            fail(f"{where}: threshold {r['threshold']} differs from the expected {thresholds[a]!r} for {a}")
        if r["prediction"] not in ("0", "1") or int(r["prediction"]) != int(s > thresholds[a]):
            fail(f"{where}: stored prediction {r['prediction']!r} differs from the rule score > threshold")
        out[a][sid] = (r["subject_id"], lab, r["score_attack"], s)
    for a, d in out.items():
        if len(d) != exp["n_images"]:
            fail(f"{what}: {len(d)} rows for {a} instead of {exp['n_images']}")
    return out


def m1_thresholds(b: bytes, analyzers) -> dict:
    try:
        doc = json.loads(b.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        fail(f"threshold.json is not valid JSON: {e}")
    if doc.get("synthetic") is not False:
        fail("threshold.json: synthetic must be false (frozen M1 thresholds only)")
    out = {}
    for a in analyzers:
        e = (doc.get("analyzers") or {}).get(a)
        if not isinstance(e, dict) or not isinstance(e.get("value"), float):
            fail(f"threshold.json: no frozen threshold for {a}")
        if e.get("decision_rule") != "attack if score > value (tie = bona fide)":
            fail(f"threshold.json: decision rule of {a} is not the M1 rule")
        if not 0.0 <= e["value"] <= 1.0:
            fail(f"threshold.json: threshold of {a} outside [0, 1]")
        out[a] = e["value"]
    return out


def joined(c1: dict, m1: dict, manifest: list, analyzers, exp: dict) -> dict:
    """analizzatore -> elenco ordinato di (sample_id, subject, label, score): stessi campioni, soggetti, etichette e
    score (testo identico) in C1 e M1, e uguali al manifest C1."""
    man = {}
    for r in manifest:
        if r["file"] in man:
            fail(f"C1 manifest: duplicated file {r['file']}")
        man[r["file"]] = (r["subject"], parse_label(r["label"], "C1 manifest"))
    if len(man) != exp["n_images"]:
        fail(f"C1 manifest: {len(man)} images instead of {exp['n_images']}")
    out = {}
    for a in analyzers:
        if set(c1[a]) != set(m1[a]) or set(c1[a]) != set(man):
            fail(f"{a}: the samples of the C1 scores, the M1 scores and the C1 manifest differ")
        rows = []
        for sid in sorted(c1[a]):
            sc, lc, tc, vc = c1[a][sid]
            sm, lm, tm, _ = m1[a][sid]
            if (sc, lc) != (sm, lm) or (sc, lc) != man[sid]:
                fail(f"{a} {sid}: subject or label differ between C1 scores, M1 scores and C1 manifest")
            if tc != tm:
                fail(f"{a} {sid}: score differs between C1 ({tc}) and M1 ({tm})")
            rows.append((sid, sc, lc, vc))
        nb = sum(1 for r in rows if r[2] == 0)
        if (nb, len(rows) - nb) != (exp["n_bona_fide"], exp["n_attack"]):
            fail(f"{a}: {nb} bona fide and {len(rows) - nb} attacks instead of {exp['n_bona_fide']} and {exp['n_attack']}")
        subj = {r[1] for r in rows}
        if len(subj) != exp["n_subjects"]:
            fail(f"{a}: {len(subj)} subjects instead of {exp['n_subjects']}")
        out[a] = rows
    return out


# ============================================================================ metriche esatte

def counts(rows, t: float) -> dict:
    """Conteggi con la regola operativa: FA = attacchi con score <= t, FR = bona fide con score > t."""
    nb = sum(1 for r in rows if r[2] == 0)
    na = sum(1 for r in rows if r[2] == 1)
    fa = sum(1 for r in rows if r[2] == 1 and not r[3] > t)
    fr = sum(1 for r in rows if r[2] == 0 and r[3] > t)
    return {"n_bona_fide": nb, "n_attack": na, "fa": fa, "fr": fr}


def rates(c: dict) -> dict:
    """APCER, BPCER, ACER esatti; None (NA) quando la classe manca, mai 0."""
    apcer = Fraction(c["fa"], c["n_attack"]) if c["n_attack"] else None
    bpcer = Fraction(c["fr"], c["n_bona_fide"]) if c["n_bona_fide"] else None
    acer = (apcer + bpcer) / 2 if apcer is not None and bpcer is not None else None
    return {"apcer": apcer, "bpcer": bpcer, "acer": acer}


def minus(a: dict, b: dict) -> dict:
    return {k: a[k] - b[k] for k in a}


def delta(r05: dict, rm1: dict):
    return rm1["acer"] - r05["acer"] if rm1["acer"] is not None and r05["acer"] is not None else None


def sign(x) -> str:
    if x is None:
        return NA
    return "+" if x > 0 else "-" if x < 0 else "0"


def dec(x, k: int = 6) -> str:
    """Valore decimale a k cifre ("NA" per None); lo zero esatto senza segno."""
    if x is None:
        return NA
    if x == 0:
        return f"{0:.{k}f}"
    return f"{float(x):.{k}f}"


def frac(x) -> str:
    return NA if x is None else f"{x.numerator}/{x.denominator}"


def table_float(c: dict) -> dict:
    """Valori come li scrive la tabella M1 (divisioni in virgola mobile di evaluation.py, quattro decimali)."""
    ap = c["fa"] / c["n_attack"] if c["n_attack"] else None
    bp = c["fr"] / c["n_bona_fide"] if c["n_bona_fide"] else None
    ac = (ap + bp) / 2 if ap is not None and bp is not None else None
    f = (lambda v: "—" if v is None else f"{v:.4f}")
    return {"APCER": f(ap), "BPCER": f(bp), "ACER": f(ac)}


def analyze(rows, t05: float, tm1: float) -> dict:
    """Insieme completo, per soggetto e lasciando fuori un soggetto alla volta (dal solo calcolo)."""
    subjects = sorted({r[1] for r in rows})
    full = {"t05": counts(rows, t05), "tm1": counts(rows, tm1)}
    per = {s: {"t05": counts([r for r in rows if r[1] == s], t05), "tm1": counts([r for r in rows if r[1] == s], tm1)} for s in subjects}
    for th in ("t05", "tm1"):           # i conteggi per soggetto si sommano a quelli completi
        tot = {k: sum(per[s][th][k] for s in subjects) for k in full[th]}
        if tot != full[th]:
            fail("internal check: per-subject counts do not add up to the full counts")
    fr05, frm1 = rates(full["t05"]), rates(full["tm1"])
    d_full = delta(fr05, frm1)
    loso = {}
    for s in subjects:
        c05, cm1 = minus(full["t05"], per[s]["t05"]), minus(full["tm1"], per[s]["tm1"])
        r05, rm1 = rates(c05), rates(cm1)
        d = delta(r05, rm1)
        loso[s] = {"t05": c05, "tm1": cm1, "r05": r05, "rm1": rm1, "delta": d,
                   "sign": sign(d), "sign_change": NA if d is None or d_full is None else sign(d) != sign(d_full)}
    return {"subjects": subjects, "full": full, "full_r05": fr05, "full_rm1": frm1, "delta_full": d_full, "per": per, "loso": loso}


def summary(res: dict) -> dict:
    """Aggregato pubblicabile (nessun identificativo di soggetto)."""
    ds = [v["delta"] for v in res["loso"].values() if v["delta"] is not None]
    d0 = res["delta_full"]
    sg = [v["sign"] for v in res["loso"].values()]
    defined = [v for v in res["loso"].values() if v["delta"] is not None]
    nbs = [v["t05"]["n_bona_fide"] for v in res["loso"].values()]
    nas = [v["t05"]["n_attack"] for v in res["loso"].values()]
    comp = [(res["per"][s]["t05"]["n_bona_fide"] > 0, res["per"][s]["t05"]["n_attack"] > 0) for s in res["subjects"]]
    out = {
        "n_exclusions": len(res["loso"]), "n_defined": len(ds), "n_na": len(res["loso"]) - len(ds),
        "delta_min": dec(min(ds)) if ds else NA, "delta_max": dec(max(ds)) if ds else NA,
        "delta_min_exact": frac(min(ds)) if ds else NA, "delta_max_exact": frac(max(ds)) if ds else NA,
        "deviation_from_full_min": dec(min(ds) - d0) if ds and d0 is not None else NA,
        "deviation_from_full_max": dec(max(ds) - d0) if ds and d0 is not None else NA,
        "n_sign_changes": sum(1 for v in defined if v["sign_change"] is True),
        "n_opposite_sign": sum(1 for v in defined if d0 is not None and v["sign"] in "+-" and sign(d0) in "+-" and v["sign"] != sign(d0)),
        "n_zero": sg.count("0"), "n_positive": sg.count("+"), "n_negative": sg.count("-"),
        "n_bona_fide_after_exclusion": [min(nbs), max(nbs)], "n_attack_after_exclusion": [min(nas), max(nas)],
        "subjects_both_classes": sum(1 for b, a in comp if b and a), "subjects_attack_only": sum(1 for b, a in comp if a and not b),
        "subjects_bona_fide_only": sum(1 for b, a in comp if b and not a),
    }
    return out


# ============================================================================ riproduzione della tabella M1 accettata

def md_tables(text: str) -> list:
    """Tabelle Markdown (righe come dizionari) nell'ordine del file; ogni tabella inizia con la riga '| analyzer |'."""
    tables, cur, head = [], None, None
    for line in text.splitlines():
        if line.startswith("| analyzer |"):
            head = [c.strip() for c in line.strip().strip("|").split("|")]
            cur = []
            tables.append(cur)
        elif cur is not None and line.startswith("|---"):
            continue
        elif cur is not None and line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) != len(head):
                fail("M1 table: a row has a different number of cells from its header")
            cur.append(dict(zip(head, cells)))
        else:
            cur = None
    return tables


def expected_cells(c05: dict, cm1: dict, tm1: float) -> dict:
    f05, fm1 = table_float(c05), table_float(cm1)
    return {"n": str(c05["n_bona_fide"] + c05["n_attack"]), "n_bona_fide": str(c05["n_bona_fide"]), "n_attack": str(c05["n_attack"]),
            "APCER@0.5": f05["APCER"], "BPCER@0.5": f05["BPCER"], "ACER@0.5": f05["ACER"],
            "accepted attacks@0.5": str(c05["fa"]), "rejected bona fide@0.5": str(c05["fr"]), "dev threshold": f"{tm1:.6f}",
            "APCER@dev": fm1["APCER"], "BPCER@dev": fm1["BPCER"], "ACER@dev": fm1["ACER"],
            "accepted attacks@dev": str(cm1["fa"]), "rejected bona fide@dev": str(cm1["fr"])}


def reproduce_table(text: str, data: dict, thr: dict, dev_subjects: set, analyzers) -> dict:
    """I conteggi e i valori ricalcolati devono coincidere esattamente con la tabella M1 accettata (blocco principale e
    blocco senza i soggetti dev); altrimenti FAIL. Restituisce il numero di celle confrontate per blocco."""
    tabs = md_tables(text)
    if len(tabs) != 2:
        fail(f"M1 table: {len(tabs)} tables found instead of 2 (main block and dev-subject sensitivity block)")
    checked = {}
    for name, tab, keep in (("main", tabs[0], None), ("without the dev subjects", tabs[1], dev_subjects)):
        by = {r["analyzer"]: r for r in tab}
        n = 0
        for a in analyzers:
            if a not in by:
                fail(f"M1 table ({name} block): no row for {a}")
            rows = data[a] if keep is None else [r for r in data[a] if r[1] not in keep]
            exp = expected_cells(counts(rows, 0.5), counts(rows, thr[a]), thr[a])
            for col, v in exp.items():
                if by[a].get(col) != v:
                    fail(f"M1 table ({name} block), {a}, {col}: recomputed {v} differs from the accepted {by[a].get(col)}")
                n += 1
        checked[name] = n
    return checked


# ============================================================================ stato del repository e run

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
    rid = f"{date}-N1-subjects-{state['commit']}" + ("-dev" if state["dirty"] else "")
    if not RUN_ID_RE.match(rid):
        fail(f"run id {rid} does not follow YYYYMMDD-N1-subjects-<commit>[-dev]")
    return rid


def protected_extra(root: Path) -> list:
    return [root / x for x in EXTRA_PROTECTED]


def run_dir_problem(run: Path, root: Path, inputs: dict) -> str:
    """Motivo per cui la cartella del run non è ammessa ("" se lo è), prima di qualunque modifica."""
    if run.parent.is_symlink():
        return f"the output root {run.parent.name} is a symbolic link"
    # la cartella del run non sta nella cartella di un pacchetto d'ingresso (quella del suo SHA256SUMS: ne cambierebbe
    # la completezza) e nessun ingresso sta nella cartella del run
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
    # nessuna voce estranea: solo le uscite dichiarate (e i loro temporanei); un file di un'altra versione è rifiutato
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


# ============================================================================ uscite

def csv_text(header, rows) -> str:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


PER_SUBJECT_HEADER = ["analyzer", "subject_id", "n_bona_fide", "n_attack", "fa_at_0_5", "fr_at_0_5", "apcer_at_0_5", "bpcer_at_0_5",
                      "m1_threshold", "fa_at_m1", "fr_at_m1", "apcer_at_m1", "bpcer_at_m1"]
LOSO_HEADER = ["analyzer", "excluded_subject", "n_bona_fide", "n_attack", "fa_at_0_5", "fr_at_0_5", "apcer_at_0_5", "bpcer_at_0_5",
               "acer_at_0_5", "m1_threshold", "fa_at_m1", "fr_at_m1", "apcer_at_m1", "bpcer_at_m1", "acer_at_m1", "delta_acer",
               "delta_acer_exact", "sign", "full_delta_acer", "sign_change"]
PER_IMAGE_HEADER = ["analyzer", "sample_id", "subject_id", "label", "score_attack", "prediction_at_0_5", "m1_threshold", "prediction_at_m1"]


def tf(v) -> str:
    return NA if v == NA else ("true" if v else "false")


def build_outputs(ctx: dict) -> tuple:
    """(file pubblici, file riservati) come testi."""
    an, res, thr, c1 = ctx["analyzers"], ctx["results"], ctx["thresholds"], ctx["c1"]
    per_rows, loso_rows, img_rows = [], [], []
    for a in an:
        r = res[a]
        for s in r["subjects"]:
            p05, pm1 = r["per"][s]["t05"], r["per"][s]["tm1"]
            q05, qm1 = rates(p05), rates(pm1)
            per_rows.append([a, s, p05["n_bona_fide"], p05["n_attack"], p05["fa"], p05["fr"], dec(q05["apcer"]), dec(q05["bpcer"]),
                             repr(thr[a]), pm1["fa"], pm1["fr"], dec(qm1["apcer"]), dec(qm1["bpcer"])])
            v = r["loso"][s]
            loso_rows.append([a, s, v["t05"]["n_bona_fide"], v["t05"]["n_attack"], v["t05"]["fa"], v["t05"]["fr"], dec(v["r05"]["apcer"]),
                              dec(v["r05"]["bpcer"]), dec(v["r05"]["acer"]), repr(thr[a]), v["tm1"]["fa"], v["tm1"]["fr"],
                              dec(v["rm1"]["apcer"]), dec(v["rm1"]["bpcer"]), dec(v["rm1"]["acer"]), dec(v["delta"]), frac(v["delta"]),
                              v["sign"], dec(r["delta_full"]), tf(v["sign_change"])])
        for sid, subj, lab, s in ctx["data"][a]:
            img_rows.append([a, sid, subj, lab, c1[a][sid][2], int(s > 0.5), repr(thr[a]), int(s > thr[a])])
    pub = public_doc(ctx)
    pub_md = public_md(ctx, pub)
    public = {"n1_subject_sensitivity_public.json": json.dumps(pub, indent=1) + "\n", "n1_subject_sensitivity_public.md": pub_md}
    public["SHA256SUMS"] = sums_text(public)
    restricted = {"n1_per_subject.csv": csv_text(PER_SUBJECT_HEADER, per_rows), "n1_loso.csv": csv_text(LOSO_HEADER, loso_rows),
                  "n1_per_image.csv": csv_text(PER_IMAGE_HEADER, img_rows)}
    rdoc = {**pub, "restricted": True, "subjects": res[an[0]]["subjects"],
            "loso_by_analyzer": {a: {s: {"delta_acer": dec(res[a]["loso"][s]["delta"]), "sign": res[a]["loso"][s]["sign"],
                                         "n_bona_fide": res[a]["loso"][s]["t05"]["n_bona_fide"], "n_attack": res[a]["loso"][s]["t05"]["n_attack"]}
                                     for s in res[a]["subjects"]} for a in an}}
    restricted["n1_subject_sensitivity_restricted.json"] = json.dumps(rdoc, indent=1) + "\n"
    restricted["n1_subject_sensitivity_restricted.md"] = restricted_md(ctx, per_rows, loso_rows)
    restricted["public_SHA256SUMS.sha256"] = f"{sha256_bytes(public['SHA256SUMS'].encode('utf-8'))}  SHA256SUMS\n"
    restricted["SHA256SUMS"] = sums_text(restricted)
    return public, restricted


def sums_text(files: dict) -> str:
    return "".join(f"{sha256_bytes(t.encode('utf-8'))}  {n}\n" for n, t in sorted(files.items()) if n != "SHA256SUMS")


def public_doc(ctx: dict) -> dict:
    an, res, thr = ctx["analyzers"], ctx["results"], ctx["thresholds"]
    rows = {}
    for a in an:
        r = res[a]
        f05, fm1 = r["full"]["t05"], r["full"]["tm1"]
        rows[a] = {
            "m1_threshold": thr[a],
            "full": {"n_bona_fide": f05["n_bona_fide"], "n_attack": f05["n_attack"],
                     "at_0_5": {"fa": f05["fa"], "fr": f05["fr"], **{k: dec(v) for k, v in r["full_r05"].items()}},
                     "at_m1": {"fa": fm1["fa"], "fr": fm1["fr"], **{k: dec(v) for k, v in r["full_rm1"].items()}},
                     "delta_acer": dec(r["delta_full"]), "delta_acer_exact": frac(r["delta_full"]), "sign": sign(r["delta_full"])},
            "leave_one_subject_out": summary(r),
        }
    return {
        "task": "N1 part 1: sensitivity of the accepted M1 result to the test subjects",
        "nature": "sensitivity analysis of existing results: not cross-validation, not a new independent validation",
        "run_id": ctx["run_id"], "commit": ctx["state"]["commit"], "dirty": ctx["state"]["dirty"], "date": ctx["date_iso"],
        "decision_rule": DECISION_RULE, "fixed_threshold": 0.5,
        "thresholds": "the four frozen M1 thresholds of threshold.json, applied unchanged (no recalibration on the test set)",
        "delta_definition": "delta ACER = ACER at the M1 threshold - ACER at 0.5; leave-one-subject-out: one subject excluded from the computation only",
        "sign_change": "sign (+, 0, -) of the leave-one-out delta different from the sign of the full delta; NA when a class is absent",
        "na_rule": "a class absent after the exclusion gives NA for its error rate, for ACER and for delta, never 0",
        "n_images": ctx["exp"]["n_images"], "n_subjects": ctx["exp"]["n_subjects"],
        "accepted_m1_table_reproduced": {"result": True, "cells_checked": ctx["reproduced"]},
        "inputs_sha256": {k: ctx["cfg"]["inputs"][k]["sha256"] for k in INPUT_KEYS},
        "config_sha256": ctx["config_sha256"], "code_sha256": ctx["code"],
        "analyzers": rows,
        "limits": ["the test images include persons already present in training (NUAA is not subject-disjoint between its official "
                   "train and test splits): the result does not show generalisation",
                   "16 subjects with unbalanced class composition: the leave-one-out values are not independent and are not a confidence interval"],
    }


def public_md(ctx: dict, pub: dict) -> str:
    L = ["# N1 part 1: sensitivity of the M1 result to the test subjects (aggregate, no subject identifiers)", "",
         f"Run `{pub['run_id']}` (commit `{pub['commit']}`, dirty tree: {'yes' if pub['dirty'] else 'no'}), {pub['date']}. "
         f"{pub['n_images']} NUAA test images, {pub['n_subjects']} subjects; existing C1 and M1 scores only, verified by SHA-256; "
         "no training, no inference, no new threshold.", "",
         f"Decision rule: {DECISION_RULE}. Thresholds: 0.5 and the four frozen M1 thresholds (unchanged). "
         "ΔACER = ACER_M1 − ACER_0.5. Leave-one-subject-out: one subject excluded from the computation only, 16 values per CNN. "
         "Sign change: sign of the leave-one-out ΔACER (+, 0, −) different from the full one. A class absent gives NA, never 0.", "",
         f"Full-set values reproduced exactly from the accepted M1 table ({ctx['reproduced']['main']} cells of the main block and "
         f"{ctx['reproduced']['without the dev subjects']} of the block without the dev subjects).", "",
         "| analyzer | M1 threshold | n bona fide | n attack | FA@0.5 | FR@0.5 | ACER@0.5 | FA@M1 | FR@M1 | ACER@M1 | ΔACER full | "
         "LOSO ΔACER min | LOSO ΔACER max | sign changes | of which opposite sign | zero | NA | n bona fide after exclusion | n attack after exclusion |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, r in pub["analyzers"].items():
        f, s = r["full"], r["leave_one_subject_out"]
        L.append(f"| {a} | {r['m1_threshold']:.6f} | {f['n_bona_fide']} | {f['n_attack']} | {f['at_0_5']['fa']} | {f['at_0_5']['fr']} | "
                 f"{f['at_0_5']['acer']} | {f['at_m1']['fa']} | {f['at_m1']['fr']} | {f['at_m1']['acer']} | {f['delta_acer']} | "
                 f"{s['delta_min']} | {s['delta_max']} | {s['n_sign_changes']}/{s['n_defined']} | {s['n_opposite_sign']} | {s['n_zero']} | "
                 f"{s['n_na']} | {s['n_bona_fide_after_exclusion'][0]}–{s['n_bona_fide_after_exclusion'][1]} | "
                 f"{s['n_attack_after_exclusion'][0]}–{s['n_attack_after_exclusion'][1]} |")
    s0 = next(iter(pub["analyzers"].values()))["leave_one_subject_out"]
    L += ["", f"Class composition of the subjects (same for every CNN): {s0['subjects_both_classes']} with both classes, "
          f"{s0['subjects_attack_only']} with attacks only, {s0['subjects_bona_fide_only']} with bona fide only.", "",
          "Limits: " + " ".join(x[0].upper() + x[1:] + "." for x in pub["limits"]),
          "This is a sensitivity analysis of existing results, not cross-validation and not a new independent validation.", ""]
    return "\n".join(L)


def restricted_md(ctx: dict, per_rows, loso_rows) -> str:
    L = ["# N1 part 1: per-subject detail (RESTRICTED: subject identifiers)", "",
         f"Run `{ctx['run_id']}`. Same rule and thresholds as the public aggregate. NA: class absent.", ""]
    for a in ctx["analyzers"]:
        r = ctx["results"][a]
        L += [f"## {a} (M1 threshold {ctx['thresholds'][a]:.6f}; full ΔACER {dec(r['delta_full'])})", "",
              "| subject | n bona fide | n attack | FA@0.5 | FR@0.5 | FA@M1 | FR@M1 | LOSO n bona fide | LOSO n attack | LOSO ACER@0.5 | LOSO ACER@M1 | LOSO ΔACER | sign | sign change |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for p, q in zip([x for x in per_rows if x[0] == a], [x for x in loso_rows if x[0] == a]):
            L.append(f"| {p[1]} | {p[2]} | {p[3]} | {p[4]} | {p[5]} | {p[9]} | {p[10]} | {q[2]} | {q[3]} | {q[8]} | {q[14]} | {q[15]} | {q[17]} | {q[19]} |")
        L.append("")
    return "\n".join(L)


# ============================================================================ comandi

def prepare(a) -> dict:
    """Configurazione, ingressi verificati e calcolo, senza scrivere nulla."""
    base = Path(a.input_root).resolve() if a.input_root else ROOT
    cfg_p = Path(a.config) if Path(a.config).is_absolute() else ROOT / a.config
    cfg, cfg_sha = load_config(cfg_p)
    data = verify_inputs(cfg, base)
    an, exp = cfg["analyzers"], cfg["expected"]
    thr = m1_thresholds(data["m1_threshold"], an)
    c1 = score_rows(data["c1_scores"], "C1 scores.csv", an, exp, {x: 0.5 for x in an})
    m1 = score_rows(data["m1_scores_test_dev_threshold"], "M1 scores_test_dev_threshold.csv", an, exp, thr)
    manifest = parse_csv(data["c1_manifest"], "C1 manifest", ("file", "label", "subject"))
    dev = parse_csv(data["m1_dev_manifest"], "M1 dev manifest", ("file", "label", "subject"))
    dev_subjects = {r["subject"] for r in dev}
    if len(dev_subjects) != exp["n_dev_subjects"]:
        fail(f"M1 dev manifest: {len(dev_subjects)} subjects instead of {exp['n_dev_subjects']}")
    rows = joined(c1, m1, manifest, an, exp)
    reproduced = reproduce_table(data["m1_table"].decode("utf-8"), rows, thr, dev_subjects, an)
    results = {x: analyze(rows[x], 0.5, thr[x]) for x in an}
    return {"cfg": cfg, "config_sha256": cfg_sha, "analyzers": an, "exp": exp, "thresholds": thr, "c1": c1, "data": rows,
            "reproduced": reproduced, "results": results, "base": base, "config_path": cfg_p}


def cmd_check_inputs(a) -> int:
    base = Path(a.input_root).resolve() if a.input_root else ROOT
    cfg_p = Path(a.config) if Path(a.config).is_absolute() else ROOT / a.config
    cfg, _ = load_config(cfg_p)
    verify_inputs(cfg, base)
    print(f"inputs: {len(INPUT_KEYS)} files verified against the configuration and their SHA256SUMS")
    print("RESULT: PASS (check-inputs)")
    return 0


def code_hashes() -> dict:
    return {f: sha256_bytes((ROOT / f).read_bytes()) for f in CODE_FILES}


def cmd_run(a) -> int:
    base = Path(a.input_root).resolve() if a.input_root else ROOT
    cfg_p = Path(a.config) if Path(a.config).is_absolute() else ROOT / a.config
    cfg, _ = load_config(cfg_p)                     # solo per i percorsi da validare: nessuna scrittura prima
    state = repo_state(ROOT)
    date = a.date or time.strftime("%Y%m%d")
    rid = run_id_for(date, state)
    out_root = Path(a.out_root) if Path(a.out_root).is_absolute() else ROOT / a.out_root
    run = out_root / rid
    inputs = {**input_paths(cfg, base), "the configuration": cfg_p}
    if msg := run_dir_problem(run, ROOT, inputs):
        fail(f"refusing the run folder: {msg}")
    # cartella ammessa: da qui ogni errore lascia solo status.json = FAIL
    run.mkdir(parents=True, exist_ok=True)
    head = {"run_id": rid, "commit": state["commit"], "dirty": state["dirty"], "command": "run"}
    write_status(run, {**head, "status": "RUNNING", "passed": False})
    try:
        left = invalidate_outputs(run)
        if left:
            fail(f"previous outputs could not be removed: {', '.join(left)}")
        ctx = prepare(a)
        ctx.update(run_id=rid, state=state, date_iso=time.strftime("%Y-%m-%dT%H:%M:%S%z"), code=code_hashes())
        public, restricted = build_outputs(ctx)
        mw.write_all(run / RESTRICTED, restricted, newline="")
        mw.write_all(run / PUBLIC, public, newline="")
        write_status(run, {**head, "status": "PASS", "passed": True, "date": ctx["date_iso"],
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
    for x in ctx["analyzers"]:
        r = ctx["results"][x]
        s = summary(r)
        print(f"{x}: M1 threshold {ctx['thresholds'][x]:.6f}; ACER@0.5 {dec(r['full_r05']['acer'], 4)} -> ACER@M1 "
              f"{dec(r['full_rm1']['acer'], 4)}; full delta {dec(r['delta_full'])}; LOSO [{s['delta_min']}, {s['delta_max']}], "
              f"sign changes {s['n_sign_changes']}/{s['n_defined']}, NA {s['n_na']}")
    print(f"accepted M1 table reproduced: {ctx['reproduced']}")
    print(f"run folder: {run.name} (public/ and restricted/)")
    print("RESULT: PASS (run)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="N1 part 1: sensitivity of the accepted M1 result to the test subjects")
    ap.add_argument("cmd", choices=["check-inputs", "run"])
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="configuration JSON (relative to the repository root)")
    ap.add_argument("--input-root", default=None, help="base folder of the input paths of the configuration (default: repository root)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="folder that receives <run-id>/ (default results/n1)")
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
    except Exception as e:  # noqa: BLE001  (errore imprevisto: le uscite sono già state tolte da cmd_run)
        print(f"ERROR: unexpected error {type(e).__name__}: {e}", file=sys.stderr)
        print(f"RESULT: FAIL ({a.cmd})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
