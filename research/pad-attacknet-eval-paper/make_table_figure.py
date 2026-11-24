"""
Materiali per l'articolo della scheda C5 (le cinque reti sul test NUAA: LivenessNet, le tre AttackNet e ResNet-18 come
riferimento esterno), rigenerati dai soli file pubblici della valutazione 20261124-ANETEVAL-7c22b1a e degli otto run che
essa valuta: la tabella delle cinque reti (APCER, BPCER, ACER nei due stati della BN, con i denominatori), le variazioni
e i tre contrasti con gli intervalli per soggetto, la figura delle variazioni di ACER, l'appendice per seed, per unità e
per fold, i conteggi dei modelli a classe unica sul test e sul dev e dei non valutabili, le analisi descrittive
(patience 12, soglia EER, confronto con ResNet-18, graduatorie, sensibilità togliendo un'unità) e un CSV aggregato in cui
sta ogni valore delle tabelle e della figura.

Dalla cartella del record (--record, la cartella che contiene status.json e public/) legge soltanto status.json,
public/SHA256SUMS e i file che questo elenca; nient'altro della cartella. Il SHA-256 di public/SHA256SUMS deve essere
l'impronta fissata qui (PUBLIC_SUMS_SHA256, uguale a public_sha256sums_sha256 di status.json) e ogni file elencato deve
avere il SHA-256 della sua riga; report.md e reproduction.csv sono verificati solo nell'impronta, non letti. Dei sei run
delle AttackNet (--anet-runs) e dei due run di C2 (--c2-runs) legge soltanto status.json, public/SHA256SUMS,
public/one_class.csv e public/models.csv: l'impronta di status.json è quella che results.json e status.json della
valutazione registrano (AttackNet) o quella fissata qui da status.json della valutazione C2 (C2_RUNS), public/SHA256SUMS
ha l'impronta registrata nel suo status.json e i due CSV quella della loro riga. Poi controlla status.json e
results.json, valida tutte le righe di units.csv e ricalcola dalle frazioni esatte di units.csv medie, conteggi, valori
per unità, tabella delle reti (table.csv), modelli a classe unica sul test e graduatorie che results.json riporta, e i
conteggi a classe unica sul dev di one_class.csv contro models.csv e units.csv: se un valore non torna si ferma. Gli
intervalli per soggetto non si possono ricalcolare dai file pubblici (servono i conteggi per soggetto): si copiano da
results.json dopo averne controllato forma, ricampionamenti validi, coerenza con la media e con le letture.

Tutte le uscite si calcolano in memoria: prima il CSV aggregato, poi le tabelle (Tabella 1, appendice A, analisi
descrittive) e la figura, costruite rileggendo i byte di quel CSV: ogni valore delle tabelle e della figura è un valore
del CSV aggregato (le note generate, i conteggi a classe unica e le graduatorie vengono dagli input controllati). Poi si
scrivono, con fsync, in una cartella temporanea nascosta accanto alla cartella di uscita, il nome di uscita è riservato
con un mkdir esclusivo, riconosciuto per inode subito prima del rename, e la cartella temporanea vi è rinominata sopra:
il rename è l'unico punto di consegna (la meccanica del pacchetto della tabella C2, PR #33). SIGINT, SIGTERM e SIGHUP
sono solo registrati e letti in punti di controllo; messaggio ed esito si decidono dallo stato sul disco. --check
rigenera in memoria e confronta con i file consegnati, senza scrivere nulla.

Arrotondamento (una sola funzione, points): proporzione × 100, due decimali, metà lontano dallo zero, sul valore
esatto (la frazione di units.csv, oppure il decimale scritto in results.json per gli estremi degli intervalli).
Tipografia: nelle tabelle Markdown, nella figura e nelle frasi generate il segno meno è U+2212; nei CSV e nel .tex il
trattino ASCII.

  python3 -I -B make_table_figure.py --out DIR [--record DIR] [--anet-runs DIR] [--c2-runs DIR]
  python3 -I -B make_table_figure.py --check [--against DIR] [--text-only] [--record DIR] [--anet-runs DIR] [--c2-runs DIR]
"""
import argparse
import csv
import errno
import hashlib
import io
import json
import math
import os
import re
import shutil
import signal
import stat
import sys
import tempfile
import textwrap
import zlib
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path

RUN_ID = "20261124-ANETEVAL-7c22b1a"
SCHEMA = "pad-anet-eval/1"
# SHA-256 di public/SHA256SUMS della valutazione (= public_sha256sums_sha256 di status.json)
PUBLIC_SUMS_SHA256 = "1f6815b08c04b94f54f99863b260ffd16e86da8684a748d1c27e06403061a252"
SUMS_FILES = ("conditions.csv", "report.md", "reproduction.csv", "results.json", "table.csv", "units.csv")

HERE = Path(os.path.abspath(__file__)).parent
DEFAULT_RECORD = HERE.parent / "pad-attacknet-eval-results" / "runs" / RUN_ID
DEFAULT_ANET_RUNS = HERE.parent / "pad-attacknet-eval" / "runs"
DEFAULT_C2_RUNS = HERE.parent / "pad-arch-eval" / "runs"
# i sei run delle AttackNet valutati (results.json, runs: le impronte dei loro status.json si leggono da lì)
ANET_RUNS = {("AttackNetV1", 4): "20261122-ANETRUN-v1-p4-b8e0b3c", ("AttackNetV2_1", 4): "20261122-ANETRUN-v2_1-p4-9dfca6b",
             ("AttackNetV2_2", 4): "20261122-ANETRUN-v2_2-p4-bd28e4b",
             ("AttackNetV1", 12): "20261123-ANETRUN-v1-p12-1ef418b",
             ("AttackNetV2_1", 12): "20261123-ANETRUN-v2_1-p12-903e3c9",
             ("AttackNetV2_2", 12): "20261123-ANETRUN-v2_2-p12-b5c214a"}
# i due run di C2 (LivenessNet e ResNet-18): run id e SHA-256 di status.json, da runs.nuaa_patience_<p>.status_sha256 di
# status.json della valutazione C2 20261121-ARCHEVAL-332fbdf (PR #32)
C2_RUNS = {4: ("20261120-ARCHRUN-p4-7faa49f", "4e17325da5a9506153a280538bb64c74d1d00632981dd736b17d8bc5648bc0e2"),
           12: ("20261120-ARCHRUN-p12-ea7fdca", "52a945cc85eae33507736896085058048ec9e31c283ec9fd9ea14a788de4c4e2")}
RUN_PUBLIC_FILES = ("models.csv", "one_class.csv", "report.md", "summary.json")
RUN_READ = ("one_class.csv", "models.csv")
ONE_CLASS_HEADER = ["network", "patience", "bn_state", "one_class", "evaluable", "not_evaluable", "all_bona_fide",
                    "all_attack", "text"]
MODELS_COLUMNS = ("network", "fold", "seed", "patience", "model_status", "ordinary_one_class_05",
                  "re_estimated_one_class_05")
DEFAULT_AGAINST = HERE / "outputs"

TEXT_OUTPUTS = ("aggregate.csv", "primary_table.md", "primary_table.tex", "appendix_per_seed.md",
                "appendix_single_class.md", "single_class.csv", "descriptive.md")
IMAGE_OUTPUTS = ("figure_acer_variations.png", "figure_acer_variations.pdf")
OUTPUTS = TEXT_OUTPUTS + IMAGE_OUTPUTS

REFERENCE, EXTERNAL = "LivenessNet", "ResNet-18"
ATTACKNETS = ("AttackNetV1", "AttackNetV2_1", "AttackNetV2_2")
NETWORKS = (REFERENCE,) + ATTACKNETS + (EXTERNAL,)
OURS = (REFERENCE,) + ATTACKNETS
REUSED = (REFERENCE, EXTERNAL)
RUN_TAG = {"AttackNetV1": "v1", "AttackNetV2_1": "v2_1", "AttackNetV2_2": "v2_2"}
METRICS = ("apcer", "bpcer", "acer")
STATES = ("ordinary", "re-estimated")
PATIENCES = (4, 12)
THRESHOLDS = ("fixed_0.5", "dev_eer")
FOLDS = (1, 2, 3, 4)
SEEDS = (42, 43, 44)
UNITS = tuple((f, s) for f in FOLDS for s in SEEDS)
N_UNITS = len(UNITS)
N_SUBJECTS = 16
PRIMARY = (4, "fixed_0.5")
DESCRIPTIVE = ((12, "fixed_0.5"), (4, "dev_eer"), (12, "dev_eer"))
PAIRS = (PRIMARY,) + DESCRIPTIVE
CONDITIONS = tuple((p, st, t) for p in PATIENCES for st in STATES for t in THRESHOLDS)
SCOPES = ("all seeds",) + tuple(f"seed {s}" for s in SEEDS)
MINUS = "−"

INTERVALS_NOTE = ("individual 95% intervals: no correction for multiplicity, no simultaneous coverage over the contrasts, "
                  "variations or comparisons")
PP_NOTE = ("percentage points = proportion x 100, rounded to two decimals half away from zero from the exact value, at "
           "presentation only; the computations use exact fractions")
INTERVAL_REASONS = ("interval entirely below zero", "interval entirely above zero", "interval contains zero",
                    "no interval (fewer subjects than the minimum, or no valid resample)")
UNRESOLVED = "unresolved"

UNITS_HEADER = ["patience", "bn_state", "threshold", "fold", "seed", "n_frames", "frame_ids_sha256", "network",
                "threshold_value", "apcer", "bpcer", "acer"]
TABLE_HEADER = ["network", "bn_state", "units", "apcer_mean", "bpcer_mean", "acer_mean", "n_attack", "n_bona_fide",
                "n_subjects", "computed"]
CONDITIONS_HEADER = ["condition", "quantity", "network", "computed", "acer_mean", "acer_low", "acer_high",
                     "acer_valid_resamples", "apcer_mean", "apcer_low", "apcer_high", "bpcer_mean", "bpcer_low",
                     "bpcer_high"]
AGG_HEADER = ["analysis", "patience", "threshold", "quantity", "network", "bn_state", "scope", "fold", "seed", "units",
              "apcer", "apcer_exact", "apcer_low", "apcer_high", "apcer_valid_resamples",
              "bpcer", "bpcer_exact", "bpcer_low", "bpcer_high", "bpcer_valid_resamples",
              "acer", "acer_exact", "acer_low", "acer_high", "acer_valid_resamples",
              "below_zero", "above_zero", "zero", "n_attack", "n_bona_fide", "n_subjects", "computed", "reading",
              "reading_reason", "note", "source"]
SINGLE_HEADER = ["split", "network", "patience", "bn_state", "threshold", "models", "one_class", "all_attack",
                 "all_bona_fide", "not_evaluable"]

QUANTITIES = ("variation", "contrast", "gap_vs_resnet18", "variation_difference_vs_resnet18")
VARIATION_STATE = "re-estimated minus ordinary"

FRACTION_RE = re.compile(r"-?(0|[1-9][0-9]*)/[1-9][0-9]*")
SUMS_LINE_RE = re.compile(r"([0-9a-f]{64})  ([A-Za-z0-9._-]+)")
HEX64_RE = re.compile(r"[0-9a-f]{64}")
INT_RE = re.compile(r"0|[1-9][0-9]*")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class Refusal(Exception):
    """Errore che ferma il comando senza lasciare uscite (messaggio in inglese, senza identificativi)."""


class OutputRace(Exception):
    """La cartella di uscita è comparsa dopo i controlli (per esempio un'altra esecuzione sulla stessa cartella)."""


class Interrupted(Exception):
    """Un segnale registrato trovato in un punto di controllo prima del rename: la scrittura si ferma e si pulisce."""


# ---------------------------------------------------------------------------------------------------------------------
# percorsi e letture

def same_file(a, b):
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def check_dir(path, what):
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        raise Refusal(f"{what} does not exist: {path}")
    if stat.S_ISLNK(st.st_mode):
        raise Refusal(f"{what} is a symbolic link: {path}")
    if not stat.S_ISDIR(st.st_mode):
        raise Refusal(f"{what} is not a directory: {path}")


def read_regular(path, what):
    """Legge un file regolare, mai un collegamento simbolico né un file speciale; i byte letti sono quelli verificati."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        raise Refusal(f"missing {what}: {path}")
    if stat.S_ISLNK(st.st_mode):
        raise Refusal(f"{what} is a symbolic link: {path}")
    if not stat.S_ISREG(st.st_mode):
        raise Refusal(f"{what} is not a regular file: {path}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        st2 = os.fstat(fd)
        if not stat.S_ISREG(st2.st_mode) or (st2.st_dev, st2.st_ino) != (st.st_dev, st.st_ino):
            raise Refusal(f"{what} changed while it was opened: {path}")
        chunks = []
        while True:
            b = os.read(fd, 1 << 20)
            if not b:
                break
            chunks.append(b)
    finally:
        os.close(fd)
    return b"".join(chunks)


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def parse_sums(sums, where, names):
    """Un SHA256SUMS: ASCII, a capo finale, righe ben formate, esattamente i file names una volta ciascuno."""
    try:
        text = sums.decode("ascii")
    except UnicodeDecodeError:
        raise Refusal(f"{where} is not ASCII")
    if not text.endswith("\n"):
        raise Refusal(f"{where} does not end with a newline")
    listed = {}
    for line in text[:-1].split("\n"):
        m = SUMS_LINE_RE.fullmatch(line)
        if not m:
            raise Refusal(f"{where}: malformed line {line!r}")
        if m.group(2) in listed:
            raise Refusal(f"{where} lists {m.group(2)} twice")
        listed[m.group(2)] = m.group(1)
    if sorted(listed) != sorted(names):
        raise Refusal(f"{where} lists {sorted(listed)}, expected {sorted(names)}")
    return listed


def run_folders(anet_runs, c2_runs):
    """Le otto cartelle dei run: chiave -> (cartella, run id). Chiavi: (AttackNet, patience) e ("C2", patience)."""
    out = {k: (os.path.join(anet_runs, rid), rid) for k, rid in ANET_RUNS.items()}
    out.update({("C2", p): (os.path.join(c2_runs, rid), rid) for p, (rid, _) in C2_RUNS.items()})
    return out


def load_input_bytes(record, anet_runs, c2_runs):
    """status.json; public/SHA256SUMS contro l'impronta fissata; ogni file elencato contro la sua riga. Dei run:
    status.json, public/SHA256SUMS e i due CSV letti, ciascuno contro la sua riga. Nient'altro delle cartelle del record e
    dei run è aperto né elencato."""
    check_dir(record, "the record directory")
    public = os.path.join(record, "public")
    check_dir(public, "the public directory of the record")
    out = {"status.json": read_regular(os.path.join(record, "status.json"), "status.json of the record")}
    sums = read_regular(os.path.join(public, "SHA256SUMS"), "public SHA256SUMS")
    if sha256(sums) != PUBLIC_SUMS_SHA256:
        raise Refusal(f"public/SHA256SUMS has SHA-256 {sha256(sums)}, expected the pinned {PUBLIC_SUMS_SHA256} "
                      f"(public_sha256sums_sha256 of status.json of {RUN_ID})")
    listed = parse_sums(sums, "public/SHA256SUMS", SUMS_FILES)
    for name in SUMS_FILES:
        b = read_regular(os.path.join(public, name), f"public {name}")
        if sha256(b) != listed[name]:
            raise Refusal(f"public/{name}: SHA-256 {sha256(b)} differs from its line in SHA256SUMS ({listed[name]})")
        out[name] = b
    out["_public_sums_sha256"] = sha256(sums)
    check_dir(anet_runs, "the directory of the AttackNet runs")
    check_dir(c2_runs, "the directory of the C2 runs")
    out["runs"] = {}
    for key, (folder, rid) in run_folders(anet_runs, c2_runs).items():
        check_dir(folder, f"the run directory {rid}")
        check_dir(os.path.join(folder, "public"), f"the public directory of {rid}")
        r = {"status.json": read_regular(os.path.join(folder, "status.json"), f"status.json of {rid}"),
             "SHA256SUMS": read_regular(os.path.join(folder, "public", "SHA256SUMS"), f"public SHA256SUMS of {rid}")}
        rl = parse_sums(r["SHA256SUMS"], f"public/SHA256SUMS of {rid}", RUN_PUBLIC_FILES)
        for name in RUN_READ:
            b = read_regular(os.path.join(folder, "public", name), f"public {name} of {rid}")
            if sha256(b) != rl[name]:
                raise Refusal(f"public/{name} of {rid}: SHA-256 {sha256(b)} differs from its line in SHA256SUMS")
            r[name] = b
        out["runs"][key] = r
    return out


# ---------------------------------------------------------------------------------------------------------------------
# lettura rigorosa dei valori

def no_duplicates(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise Refusal(f"duplicate key {k!r} in a JSON file")
        d[k] = v
    return d


def reject_constant(c):
    raise Refusal(f"non-finite number {c} in a JSON file")


def parse_json(b, name):
    try:
        d = json.loads(b.decode("utf-8"), object_pairs_hook=no_duplicates, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise Refusal(f"{name} is not valid JSON: {e}")
    if not isinstance(d, dict):
        raise Refusal(f"{name}: the top level is not an object")
    return d


def same(a, b):
    """Uguaglianza con i tipi (True non è 1, 12.0 non è 12)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def get(d, path, name="results.json"):
    cur = d
    for k in path.split("|"):
        if not isinstance(cur, dict) or k not in cur:
            raise Refusal(f"{name}: missing field {path.replace('|', '.')}")
        cur = cur[k]
    return cur


def expect(d, path, value, name="results.json"):
    if not same(get(d, path, name), value):
        raise Refusal(f"{name}: {path.replace('|', '.')} is {get(d, path, name)!r}, expected {value!r}")


def fraction(text, where):
    if not FRACTION_RE.fullmatch(text):
        raise Refusal(f"{where}: {text!r} is not an exact fraction")
    f = Fraction(text)
    if text != f"{f.numerator}/{f.denominator}":
        raise Refusal(f"{where}: {text!r} is not written reduced")
    return f


def exact_text(f):
    return f"{f.numerator}/{f.denominator}"


def read_csv(b, name, header):
    try:
        text = b.decode("utf-8")
    except UnicodeDecodeError:
        raise Refusal(f"{name} is not UTF-8")
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows or rows[0] != header:
        raise Refusal(f"{name}: unexpected header")
    out = []
    for i, r in enumerate(rows[1:], start=2):
        if len(r) != len(header):
            raise Refusal(f"{name}: line {i} has {len(r)} fields, expected {len(header)}")
        out.append(dict(zip(header, r)))
    return out


def cell(v):
    """La cella di un CSV del valutatore: NA per None, 1/0 per i booleani, n/d per le frazioni, repr per i float."""
    if v is None:
        return "NA"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, Fraction):
        return exact_text(v)
    if isinstance(v, float):
        return repr(v)
    return str(v)


def mean(vals):
    vals = list(vals)
    return sum(vals, Fraction(0)) / len(vals)


# ---------------------------------------------------------------------------------------------------------------------
# regole (PROTOCOL.md §6 (c): un estremo a zero contiene lo zero)

def reading_names(k):
    return (f"ACER variation of {k} minus that of LivenessNet lower than zero",
            f"ACER variation of {k} minus that of LivenessNet higher than zero", UNRESOLVED)


def reading_rule(k, low, high):
    """Lettura e ragione del contrasto di k dall'intervallo del suo ACER."""
    names = reading_names(k)
    if low is None or high is None:
        return names[2], INTERVAL_REASONS[3]
    if high < 0:
        return names[0], INTERVAL_REASONS[0]
    if low > 0:
        return names[1], INTERVAL_REASONS[1]
    return names[2], INTERVAL_REASONS[2]


def conclusion_rule(k, reading):
    names = reading_names(k)
    if reading in names[:2]:
        side = "lower" if reading == names[0] else "higher"
        return (f"On these checkpoints and test frames, the ACER variation with this BN re-estimation (re-estimated "
                f"minus ordinary BN) is algebraically {side} for {k} than for LivenessNet; it does not identify a cause.")
    return (f"This comparison does not show that the ACER variation with this BN re-estimation differs between {k} "
            f"and LivenessNet; it does not exclude it.")


def not_computed_text(pairs, reason):
    return f"not computed ({pairs} of {N_UNITS} pairs evaluable in both BN states): {reason}"


# termini delle quantità lineari per unità: (coefficiente, rete, stato)
def terms_of(quantity, net, state=None):
    if quantity == "variation":
        return ((1, net, "re-estimated"), (-1, net, "ordinary"))
    if quantity == "contrast":
        return ((1, net, "re-estimated"), (-1, net, "ordinary"), (-1, REFERENCE, "re-estimated"),
                (1, REFERENCE, "ordinary"))
    if quantity == "gap_vs_resnet18":
        return ((1, net, state), (-1, EXTERNAL, state))
    if quantity == "variation_difference_vs_resnet18":
        return ((1, net, "re-estimated"), (-1, net, "ordinary"), (-1, EXTERNAL, "re-estimated"),
                (1, EXTERNAL, "ordinary"))
    raise ValueError(quantity)


# ---------------------------------------------------------------------------------------------------------------------
# controlli e ricalcoli

def check_status(st, res, sums_sha):
    for path, value in (("run_id", RUN_ID), ("status", "PASS"), ("passed", True), ("official", True),
                        ("second_reading", False), ("public_sha256sums_sha256", sums_sha)):
        expect(st, path, value, "status.json")
    expect(st, "attempt", {"attempt": 1, "previous": [], "second_reading": False, "earlier_readings": []},
           "status.json")
    if not same(get(st, "not_evaluable", "status.json"), get(res, "not_evaluable")):
        raise Refusal("status.json: not_evaluable differs from not_evaluable of results.json")


def check_header(res, st):
    for path, value in (("schema", SCHEMA), ("run_id", RUN_ID), ("status", "PASS"), ("official", True),
                        ("second_reading", False)):
        expect(res, path, value)
    expect(res, "attempt", get(st, "attempt", "status.json"))
    expect(res, "hooks|set", [])
    expect(res, "c2_reproduction|exact", True)
    expect(res, "c2_opposite", {"gap_ordinary": True, "gap_re_estimated": True, "variation_difference": True})
    frames = get(res, "test_frames_per_fold")
    if not (isinstance(frames, list) and len(frames) == len(FOLDS)
            and all(type(n) is int and n > 0 for n in frames)):
        raise Refusal("results.json: test_frames_per_fold is not 4 positive integers")
    ids = get(res, "test_frame_ids_sha256")
    if not (isinstance(ids, dict) and sorted(ids) == [str(f) for f in FOLDS]
            and all(isinstance(v, str) and HEX64_RE.fullmatch(v) for v in ids.values())):
        raise Refusal("results.json: test_frame_ids_sha256 is not one SHA-256 per fold")
    return frames, ids


def analysis_of(p, t):
    """«primary» per la coppia dell'analisi primaria (patience 4, soglia 0,5), «descriptive» per le altre."""
    return "primary" if (p, t) == PRIMARY else "descriptive"


def pair_key(p, t):
    return f"patience_{p}_{t}"


def pair_block(res, p, t):
    """Il blocco di results.json di una coppia di condizioni (primaria o descrittiva)."""
    if (p, t) == PRIMARY:
        return get(res, "primary_analysis"), "primary_analysis"
    return get(res, f"descriptive|pairs|{pair_key(p, t)}"), f"descriptive.pairs.{pair_key(p, t)}"


def check_denominators(res, frames):
    """I denominatori per fold di ogni coppia e stato: uguali fra loro, frame = attacchi + bona fide = frame del fold,
    16 soggetti uniti. Restituisce per fold (frame, attacchi, bona fide, soggetti)."""
    found = None
    for p, t in PAIRS:
        blk, path = pair_block(res, p, t)
        for state in STATES:
            d = get(blk, f"denominators|{state}", path)
            per = d.get("per_fold") if isinstance(d, dict) else None
            if not isinstance(per, dict) or sorted(per) != [str(f) for f in FOLDS]:
                raise Refusal(f"results.json: {path}.denominators.{state}.per_fold is not one entry per fold")
            folds = {}
            for f in FOLDS:
                e = per[str(f)]
                if not isinstance(e, dict) or sorted(e) != ["attack", "bona_fide", "frames", "subjects"] \
                        or any(type(e[k]) is not int or e[k] <= 0 for k in e):
                    raise Refusal(f"results.json: {path}.denominators.{state}.per_fold.{f} is not four positive "
                                  f"integers")
                if e["frames"] != frames[f - 1] or e["attack"] + e["bona_fide"] != e["frames"]:
                    raise Refusal(f"results.json: {path}.denominators.{state}.per_fold.{f}: frames differ from "
                                  f"test_frames_per_fold or from attack + bona fide")
                folds[f] = (e["frames"], e["attack"], e["bona_fide"], e["subjects"])
            if sum(v[3] for v in folds.values()) != N_SUBJECTS:
                raise Refusal(f"results.json: {path}.denominators.{state}: the subjects of the folds do not add up to "
                              f"{N_SUBJECTS}")
            want = {"units": N_UNITS, "subjects": N_SUBJECTS,
                    "test_frames_summed_over_units": sum(folds[f][0] for f, _ in UNITS),
                    "attack_frames_summed_over_units": sum(folds[f][1] for f, _ in UNITS),
                    "bona_fide_frames_summed_over_units": sum(folds[f][2] for f, _ in UNITS)}
            if not same({k: d.get(k) for k in want}, want) or sorted(d) != sorted(list(want) + ["per_fold"]):
                raise Refusal(f"results.json: {path}.denominators.{state}: totals differ from the per-fold values")
            if found is None:
                found = folds
            elif folds != found:
                raise Refusal(f"results.json: {path}.denominators.{state} differs from the other conditions")
    return found


def check_unit_row(r, frames, ids, folds):
    """Una riga di units.csv: condizione, unità, rete, soglia, frame, frazioni (con i denominatori del fold), ACER."""
    where = f"units.csv ({r['patience']}, {r['bn_state']}, {r['threshold']}, fold {r['fold']}, seed {r['seed']}, " \
            f"{r['network']})"
    if r["patience"] not in [str(p) for p in PATIENCES] or r["bn_state"] not in STATES \
            or r["threshold"] not in THRESHOLDS:
        raise Refusal(f"{where}: condition outside the plan")
    if r["fold"] not in [str(f) for f in FOLDS] or r["seed"] not in [str(s) for s in SEEDS]:
        raise Refusal(f"{where}: fold or seed outside the plan")
    if r["network"] not in NETWORKS:
        raise Refusal(f"{where}: network outside the plan")
    fold = int(r["fold"])
    t = r["threshold_value"]
    if r["threshold"] == "fixed_0.5":
        if t != "0.5":
            raise Refusal(f"{where}: effective threshold {t!r}, expected 0.5")
    else:
        try:
            v = float(t)
        except ValueError:
            v = math.nan
        if not (math.isfinite(v) and 0 <= v <= 1 and t == repr(v)):
            raise Refusal(f"{where}: threshold is not a number between 0 and 1")
    if not INT_RE.fullmatch(r["n_frames"]) or int(r["n_frames"]) != frames[fold - 1]:
        raise Refusal(f"{where}: n_frames {r['n_frames']!r} differs from test_frames_per_fold of results.json")
    if r["frame_ids_sha256"] != ids[str(fold)]:
        raise Refusal(f"{where}: frame_ids_sha256 differs from test_frame_ids_sha256 of results.json")
    u = {}
    for m in METRICS:
        u[m] = fraction(r[m], where)
        if not 0 <= u[m] <= 1:
            raise Refusal(f"{where}: {m} outside [0, 1]")
    if u["acer"] != (u["apcer"] + u["bpcer"]) / 2:
        raise Refusal(f"{where}: acer is not the mean of APCER and BPCER")
    _, n_att, n_bf, _ = folds[fold]
    if (u["apcer"] * n_att).denominator != 1 or (u["bpcer"] * n_bf).denominator != 1:
        raise Refusal(f"{where}: APCER or BPCER is not a count over the attack or bona fide frames of the fold")
    u["threshold"] = t
    return u


def load_units(b, frames, ids, folds):
    """Tutte le righe di units.csv validate: {condizione: {(fold, seed): {rete: tassi}}}. LivenessNet e ResNet-18 in
    ogni unità di ogni condizione; una rete al più una volta per unità; ogni condizione con le 12 unità."""
    rows = read_csv(b, "units.csv", UNITS_HEADER)
    out = {c: {k: {} for k in UNITS} for c in CONDITIONS}
    for r in rows:
        u = check_unit_row(r, frames, ids, folds)
        c = (int(r["patience"]), r["bn_state"], r["threshold"])
        k = (int(r["fold"]), int(r["seed"]))
        if r["network"] in out[c][k]:
            raise Refusal(f"units.csv: {r['network']} twice in {c}, fold {k[0]} seed {k[1]}")
        out[c][k][r["network"]] = u
    for c in CONDITIONS:
        for k in UNITS:
            for n in REUSED:
                if n not in out[c][k]:
                    raise Refusal(f"units.csv: {n} missing in {c}, fold {k[0]} seed {k[1]} (every unit must hold "
                                  f"LivenessNet and ResNet-18)")
    return out


def presence(U):
    """Per ogni AttackNet e patience le unità in cui c'è: la stessa nelle quattro condizioni della patience (un modello
    non valutabile manca ovunque). Restituisce {(rete, patience): insieme di unità}."""
    out = {}
    for k in ATTACKNETS:
        for p in PATIENCES:
            sets = {(st, t): frozenset(u for u in UNITS if k in U[(p, st, t)][u]) for st in STATES for t in THRESHOLDS}
            if len(set(sets.values())) != 1:
                raise Refusal(f"units.csv: {k} at patience {p} is not present in the same units in the four conditions")
            out[(k, p)] = next(iter(sets.values()))
    for n in REUSED:
        for p in PATIENCES:
            out[(n, p)] = frozenset(UNITS)
    return out


def check_not_evaluable(res, pres):
    """not_evaluable di results.json: una voce per ogni modello che manca da units.csv e nessun'altra; restituisce il
    motivo per rete e patience (il testo del guardiano)."""
    ne = get(res, "not_evaluable")
    if not isinstance(ne, list):
        raise Refusal("results.json: not_evaluable is not a list")
    seen, reasons = set(), {}
    keys = ["fold", "name", "network", "patience", "reason", "run", "seed"]
    for i, e in enumerate(ne):
        if not isinstance(e, dict) or sorted(e) != keys or e["network"] not in ATTACKNETS \
                or e["patience"] not in PATIENCES or e["fold"] not in FOLDS or e["seed"] not in SEEDS \
                or type(e["patience"]) is not int or type(e["fold"]) is not int or type(e["seed"]) is not int \
                or not isinstance(e["reason"], str) or not e["reason"] \
                or e["name"] != f"{e['network']}_fold{e['fold']}_seed{e['seed']}" \
                or e["run"] != f"anet_{RUN_TAG[e['network']]}_p{e['patience']}":
            raise Refusal(f"results.json: not_evaluable[{i}] is not a model of the plan with its run, name and reason")
        key = (e["network"], e["patience"], e["fold"], e["seed"])
        if key in seen:
            raise Refusal(f"results.json: not_evaluable lists {e['name']} twice")
        seen.add(key)
        reasons.setdefault((e["network"], e["patience"]), []).append(f"{e['run']}/{e['name']}: {e['reason']}")
    for k in ATTACKNETS:
        for p in PATIENCES:
            missing = {(k, p) + u for u in UNITS if u not in pres[(k, p)]}
            listed = {x for x in seen if x[:2] == (k, p)}
            if missing != listed:
                raise Refusal(f"results.json: not_evaluable does not list exactly the {k} models missing from "
                              f"units.csv at patience {p}")
    return {key: "; ".join(v) for key, v in reasons.items()}


def guardian(pres, reasons, p):
    out = {}
    for n in NETWORKS:
        pairs = len(pres[(n, p)])
        computed = pairs == N_UNITS
        reason = None if computed else (reasons.get((n, p)) or "fewer than 12 fold-seed pairs evaluable in both BN "
                                                               "states")
        out[n] = {"pairs": pairs, "computed": computed, "reason": reason,
                  "text": "computed" if computed else not_computed_text(pairs, reason)}
    return out


def check_models(res, U, pres):
    """models di results.json: un modello valutabile per ogni AttackNet presente in units.csv, con le soglie EER del dev
    uguali alle soglie effettive di units.csv alla soglia EER."""
    models = get(res, "models")
    if not isinstance(models, list):
        raise Refusal("results.json: models is not a list")
    seen = set()
    for i, m in enumerate(models):
        if not isinstance(m, dict) or sorted(m) != ["dev_eer_threshold", "fold", "network", "run", "seed",
                                                    "weights_sha256"] \
                or m["network"] not in ATTACKNETS or m["fold"] not in FOLDS or m["seed"] not in SEEDS \
                or type(m["fold"]) is not int or type(m["seed"]) is not int or not isinstance(m["run"], str) or not re.fullmatch(r"anet_(v1|v2_1|v2_2)_p(4|12)", m["run"]) \
                or m["run"].split("_p")[0] != f"anet_{RUN_TAG[m['network']]}" \
                or not isinstance(m["weights_sha256"], str) or not HEX64_RE.fullmatch(m["weights_sha256"]) \
                or not isinstance(m["dev_eer_threshold"], dict) or sorted(m["dev_eer_threshold"]) != sorted(STATES):
            raise Refusal(f"results.json: models[{i}] is not an evaluated AttackNet of the plan")
        p = int(m["run"].rsplit("_p", 1)[1])
        key = (m["network"], p, m["fold"], m["seed"])
        if key in seen:
            raise Refusal(f"results.json: models lists {m['network']} fold {m['fold']} seed {m['seed']} at patience "
                          f"{p} twice")
        seen.add(key)
        if (m["fold"], m["seed"]) not in pres[(m["network"], p)]:
            raise Refusal(f"results.json: models[{i}] is not in units.csv")
        for st in STATES:
            v = m["dev_eer_threshold"][st]
            if type(v) is not float or repr(v) != U[(p, st, "dev_eer")][(m["fold"], m["seed"])][m["network"]]["threshold"]:
                raise Refusal(f"results.json: models[{i}].dev_eer_threshold.{st} differs from the threshold of "
                              f"units.csv at the EER threshold")
    want = {(k, p) + u for k in ATTACKNETS for p in PATIENCES for u in pres[(k, p)]}
    if seen != want:
        raise Refusal("results.json: models does not list exactly the AttackNet models of units.csv")


def unit_value(U, p, t, u, terms):
    """Il valore esatto di una quantità lineare nell'unità u (dict metrica -> Fraction), None se una rete manca."""
    out = {}
    for m in METRICS:
        tot = Fraction(0)
        for coef, net, state in terms:
            r = U[(p, state, t)][u].get(net)
            if r is None:
                return None
            tot += coef * r[m]
        out[m] = tot
    return out


def scope_units(scope):
    return list(UNITS) if scope == "all seeds" else [u for u in UNITS if u[1] == int(scope.split()[1])]


def check_interval(iv, path, mean_value):
    """Forma dell'intervallo copiato: {low, high, valid_resamples}; con 0 ricampionamenti validi estremi None, altrimenti
    low ≤ media ≤ high (controllo di plausibilità: i valori non si ricalcolano dai file pubblici)."""
    if not isinstance(iv, dict) or sorted(iv) != ["high", "low", "valid_resamples"]:
        raise Refusal(f"results.json: {path} is not an interval")
    valid = iv["valid_resamples"]
    if type(valid) is not int or not 0 <= valid <= 1000:
        raise Refusal(f"results.json: {path} does not have 0 to 1,000 valid resamples")
    low, high = iv["low"], iv["high"]
    if valid == 0:
        if low is not None or high is not None:
            raise Refusal(f"results.json: {path} has bounds without valid resamples")
        return None, None, 0
    if type(low) is not float or type(high) is not float or not math.isfinite(low) or not math.isfinite(high):
        raise Refusal(f"results.json: {path} has bounds that are not finite numbers")
    if not low <= high:
        raise Refusal(f"results.json: {path} has low > high")
    if not low <= mean_value <= high:
        raise Refusal(f"results.json: {path} does not contain its mean")
    return low, high, valid


def check_quantity(blk, path, U, p, t, quantity, net, state, keys, readings):
    """Un blocco calcolato (variazione, contrasto, divario o differenza di variazioni) contro units.csv: medie esatte,
    conteggi delle unità, forma degli intervalli e, per i contrasti con lettura, lettura, ragione e conclusione."""
    want_keys = ["computed", "counts", "interval", "mean"]
    if quantity == "contrast" and readings:
        want_keys += ["conclusion", "reading", "reason"]
    if quantity in ("gap_vs_resnet18", "variation_difference_vs_resnet18"):
        want_keys.remove("computed")
    if not isinstance(blk, dict) or sorted(blk) != sorted(want_keys):
        raise Refusal(f"results.json: {path} does not hold exactly {sorted(want_keys)}")
    if "computed" in blk and blk["computed"] is not True:
        raise Refusal(f"results.json: {path}.computed is not true")
    vals = [unit_value(U, p, t, u, terms_of(quantity, net, state)) for u in keys]
    out = {"mean": {}, "interval": {}}
    for m in METRICS:
        exact = mean(v[m] for v in vals)
        got = get(blk, f"mean|{m}", path)
        if type(got) is not float or got != float(exact):
            raise Refusal(f"results.json: {path}.mean.{m} = {got!r}, units.csv gives {float(exact)!r}")
        out["mean"][m] = exact
        out["interval"][m] = check_interval(get(blk, f"interval|{m}", path), f"{path}.interval.{m}", got)
    acer = [v["acer"] for v in vals]
    counts = {"below_zero": sum(x < 0 for x in acer), "above_zero": sum(x > 0 for x in acer),
              "zero": sum(x == 0 for x in acer), "undefined": 0, "units": len(keys)}
    if not same(blk["counts"], counts):
        raise Refusal(f"results.json: {path}.counts = {blk['counts']!r}, units.csv gives {counts!r}")
    out["counts"] = counts
    if quantity == "contrast" and readings:
        low, high, _ = out["interval"]["acer"]
        rd, reason = reading_rule(net, low, high)
        if (blk["reading"], blk["reason"]) != (rd, reason):
            raise Refusal(f"results.json: {path}.reading does not follow the reading rule on its ACER interval")
        if blk["conclusion"] != conclusion_rule(net, rd):
            raise Refusal(f"results.json: {path}.conclusion is not the conclusion written for its reading")
        out["reading"], out["reason"], out["conclusion"] = rd, reason, blk["conclusion"]
    return out


def check_not_computed(blk, path, quantity, net, status):
    want = {"computed": False, "text": status["text"]}
    if quantity == "contrast":
        want.update(reading="not computed",
                    conclusion=f"Not computed for {net} ({status['pairs']} of {N_UNITS} pairs evaluable in both BN "
                               f"states): {status['reason']}.")
    if not same(blk, want):
        raise Refusal(f"results.json: {path} is not the 'not computed' block of the guardian")


def check_analysis(res, U, pres, reasons):
    """Le quattro coppie di condizioni (primaria e descrittive), su tutte le unità e per seed: guardiano, quantità e
    righe per unità contro units.csv. Restituisce {(p, t): {"guardian", "scopes": {scope: {(quantità, rete, stato):
    valori}}}}."""
    out = {}
    for p, t in PAIRS:
        blk, path = pair_block(res, p, t)
        readings = (p, t) == PRIMARY
        want_keys = ["conditions", "denominators", "descriptive_vs_resnet18", "contrasts", "guardian", "intervals",
                     "n_subjects", "per_seed", "presentation", "rows", "variations"]
        if readings:
            want_keys += ["definition", "rankings"]
        if sorted(blk) != sorted(want_keys):
            raise Refusal(f"results.json: {path} does not hold exactly the fields of an analysis")
        expect(blk, "conditions", {"ordinary": [p, "ordinary", t], "re-estimated": [p, "re-estimated", t]}, path)
        expect(blk, "intervals", INTERVALS_NOTE, path)
        expect(blk, "presentation", PP_NOTE, path)
        status = guardian(pres, reasons, p)
        if not same(blk["guardian"], status):
            raise Refusal(f"results.json: {path}.guardian differs from the networks present in units.csv")
        if (p, t) == PRIMARY:
            for n in REUSED:
                if not status[n]["computed"]:
                    raise Refusal(f"results.json: {n} must have all {N_UNITS} units in the primary analysis")
        if not isinstance(blk["per_seed"], dict) or sorted(blk["per_seed"]) != [str(s) for s in SEEDS]:
            raise Refusal(f"results.json: {path}.per_seed does not hold exactly seeds 42, 43 and 44")
        scopes = {}
        for scope in SCOPES:
            sb = blk if scope == "all seeds" else blk["per_seed"][scope.split()[1]]
            spath = path if scope == "all seeds" else f"{path}.per_seed.{scope.split()[1]}"
            if scope != "all seeds" and sorted(sb) != ["contrasts", "descriptive_vs_resnet18", "n_subjects",
                                                       "variations"]:
                raise Refusal(f"results.json: {spath} does not hold exactly the fields of a seed")
            if not same(sb["n_subjects"], N_SUBJECTS):
                raise Refusal(f"results.json: {spath}.n_subjects is not {N_SUBJECTS}")
            keys = scope_units(scope)
            q = {}
            for name, nets in (("variations", NETWORKS), ("contrasts", ATTACKNETS),
                               ("descriptive_vs_resnet18", OURS)):
                group = sb[name]
                if not isinstance(group, dict) or list(group) != list(nets):
                    raise Refusal(f"results.json: {spath}.{name} does not hold {', '.join(nets)} in this order")
                for n in nets:
                    b = group[n]
                    bpath = f"{spath}.{name}.{n}"
                    if name == "descriptive_vs_resnet18":
                        if not status[n]["computed"]:
                            check_not_computed(b, bpath, "comparison", n, status[n])
                            continue
                        if not isinstance(b, dict) or sorted(b) != ["computed", "gap_ordinary", "gap_re_estimated",
                                                                    "variation_difference"] or b["computed"] is not True:
                            raise Refusal(f"results.json: {bpath} is not a computed comparison with ResNet-18")
                        for st in STATES:
                            q[("gap_vs_resnet18", n, st)] = check_quantity(
                                b[f"gap_{st.replace('-', '_')}"], f"{bpath}.gap_{st.replace('-', '_')}", U, p, t,
                                "gap_vs_resnet18", n, st, keys, False)
                        q[("variation_difference_vs_resnet18", n, VARIATION_STATE)] = check_quantity(
                            b["variation_difference"], f"{bpath}.variation_difference", U, p, t,
                            "variation_difference_vs_resnet18", n, None, keys, False)
                        continue
                    quantity = "variation" if name == "variations" else "contrast"
                    if not status[n]["computed"]:
                        check_not_computed(b, bpath, quantity, n, status[n])
                        continue
                    q[(quantity, n, VARIATION_STATE)] = check_quantity(b, bpath, U, p, t, quantity, n, None, keys,
                                                                       readings)
            scopes[scope] = q
        check_rows(blk["rows"], f"{path}.rows", U, p, t, status)
        out[(p, t)] = {"guardian": status, "scopes": scopes}
    return out


def check_rows(rows, path, U, p, t, status):
    """Le righe per unità di results.json: unità in ordine (fold, seed), frame, e per ogni quantità calcolata il valore
    esatto e il suo float, uguali a quelli di units.csv."""
    if not isinstance(rows, list) or len(rows) != N_UNITS:
        raise Refusal(f"results.json: {path} does not hold {N_UNITS} units")
    specs = {}
    for n in NETWORKS:
        if status[n]["computed"]:
            specs[f"variation/{n}"] = terms_of("variation", n)
    for k in ATTACKNETS:
        if status[k]["computed"]:
            specs[f"contrast/{k}"] = terms_of("contrast", k)
    for n in ATTACKNETS + (REFERENCE,):
        if status[n]["computed"]:
            for st in STATES:
                specs[f"gap/{n}/{st}"] = terms_of("gap_vs_resnet18", n, st)
            specs[f"variation_difference/{n}"] = terms_of("variation_difference_vs_resnet18", n)
    for row, u in zip(rows, UNITS):
        where = f"{path} fold {u[0]} seed {u[1]}"
        if not isinstance(row, dict) or not same(row.get("unit"), ["NUAA", u[0], u[1]]):
            raise Refusal(f"results.json: {path} is not in ascending (fold, seed) order of the plan")
        if sorted(row) != sorted(["unit", "n_frames", "frame_ids_sha256"] + list(specs)):
            raise Refusal(f"results.json: {where} does not hold exactly the quantities of the computed networks")
        for name, terms in specs.items():
            want = unit_value(U, p, t, u, terms)
            for m in METRICS:
                e = row[name].get(m) if isinstance(row[name], dict) else None
                if not isinstance(e, dict) or e.get("exact") != exact_text(want[m]) \
                        or type(e.get("value")) is not float or e["value"] != float(want[m]):
                    raise Refusal(f"results.json: {where} {name}.{m} differs from units.csv")


def check_c2_opposite(res):
    """Il confronto di LivenessNet con ResNet-18 dell'analisi primaria è l'opposto esatto del contrasto di C2 che la
    valutazione ha riprodotto (c2_reproduction.primary_contrast): media negata, estremi negati e scambiati."""
    c2 = get(res, "c2_reproduction|primary_contrast")
    mine = get(res, "primary_analysis|descriptive_vs_resnet18|LivenessNet|variation_difference")
    for m in METRICS:
        a, b = mine["interval"][m], get(c2, f"interval|{m}", "results.json c2_reproduction")
        if not (mine["mean"][m] == -get(c2, f"mean|{m}") and a["low"] == -b["high"] and a["high"] == -b["low"]
                and a["valid_resamples"] == b["valid_resamples"]):
            raise Refusal("results.json: the comparison of LivenessNet with ResNet-18 is not the exact opposite of the "
                          "C2 contrast reproduced by the evaluation")


def check_valid_resamples(analysis):
    """Le frasi generate danno un solo numero di ricampionamenti validi per metrica: deve valere in ogni blocco."""
    valid = {}
    for m in METRICS:
        seen = {q["interval"][m][2] for a in analysis.values() for s in a["scopes"].values() for q in s.values()}
        if len(seen) != 1:
            raise Refusal(f"results.json: the valid resamples of {m.upper()} differ between the blocks "
                          f"({sorted(seen)}); the generated texts state one number per metric")
        valid[m] = seen.pop()
    return valid


def rates_over(U, c, n, keys, folds):
    """Medie esatte sulle unità della rete (quelle in cui c'è) e denominatori sommati su quelle unità."""
    mine = [u for u in keys if n in U[c][u]]
    if not mine:
        return {"units": 0, "mean": None, "n_attack": None, "n_bona_fide": None, "n_subjects": None}
    subjects = sum(folds[f][3] for f in sorted({u[0] for u in mine}))
    return {"units": len(mine), "mean": {m: mean(U[c][u][n][m] for u in mine) for m in METRICS},
            "n_attack": sum(folds[u[0]][1] for u in mine), "n_bona_fide": sum(folds[u[0]][2] for u in mine),
            "n_subjects": subjects}


def check_table(b, U, folds):
    """table.csv del valutatore, campo per campo, contro le medie e i denominatori ricalcolati da units.csv."""
    rows = read_csv(b, "table.csv", TABLE_HEADER)
    want = []
    for st in STATES:
        for n in NETWORKS:
            r = rates_over(U, (4, st, "fixed_0.5"), n, UNITS, folds)
            mm = r["mean"] or {}
            want.append([n, st, str(r["units"]), cell(mm.get("apcer")), cell(mm.get("bpcer")), cell(mm.get("acer")),
                         cell(r["n_attack"]), cell(r["n_bona_fide"]), cell(r["n_subjects"]),
                         cell(r["units"] == N_UNITS)])
    got = [[r[h] for h in TABLE_HEADER] for r in rows]
    if len(got) != len(want):
        raise Refusal(f"table.csv: {len(got)} rows, expected {len(want)}")
    for g, w in zip(got, want):
        if g != w:
            raise Refusal(f"table.csv ({w[0]}, {w[1]}): a field differs from the means and denominators recomputed "
                          f"from units.csv")


def check_conditions_csv(b, res):
    """conditions.csv, campo per campo, contro i blocchi di results.json (variazioni e contrasti delle 4 coppie)."""
    rows = read_csv(b, "conditions.csv", CONDITIONS_HEADER)
    want = []
    for p, t in PAIRS:
        blk, _ = pair_block(res, p, t)
        label = pair_key(p, t) + (" (primary)" if (p, t) == PRIMARY else "")
        for kind, group in (("variation", blk["variations"]), ("contrast", blk["contrasts"])):
            for n, q in group.items():
                if not q.get("computed"):
                    want.append([label, kind, n, "0"] + ["NA"] * (len(CONDITIONS_HEADER) - 4))
                    continue
                iv = q["interval"]
                want.append([label, kind, n, "1", cell(q["mean"]["acer"]), cell(iv["acer"]["low"]),
                             cell(iv["acer"]["high"]), cell(iv["acer"]["valid_resamples"]), cell(q["mean"]["apcer"]),
                             cell(iv["apcer"]["low"]), cell(iv["apcer"]["high"]), cell(q["mean"]["bpcer"]),
                             cell(iv["bpcer"]["low"]), cell(iv["bpcer"]["high"])])
    got = [[r[h] for h in CONDITIONS_HEADER] for r in rows]
    if len(got) != len(want):
        raise Refusal(f"conditions.csv: {len(got)} rows, expected {len(want)}")
    for g, w in zip(got, want):
        if g != w:
            raise Refusal(f"conditions.csv ({w[0]}, {w[1]}, {w[2]}): a field differs from results.json")


def single_class(U):
    """I modelli che sul test decidono una sola classe per tutti i frame: APCER 0 e BPCER 1 (tutto attacco) o APCER 1 e
    BPCER 0 (tutto bona fide); con entrambe le classi presenti nel fold le due coppie sono equivalenti alla definizione."""
    cells, models = {}, []
    for c in CONDITIONS:
        for u in UNITS:
            for n, r in U[c][u].items():
                e = cells.setdefault((n, c[0], c[1], c[2]), {"models": 0, "one_class": 0, "all_bona_fide": 0,
                                                             "all_attack": 0})
                e["models"] += 1
                pair = (r["apcer"], r["bpcer"])
                if pair == (0, 1):
                    e["one_class"] += 1
                    e["all_attack"] += 1
                    models.append((c, u, n, "attack", r["acer"]))
                elif pair == (1, 0):
                    e["one_class"] += 1
                    e["all_bona_fide"] += 1
                    models.append((c, u, n, "bona fide", r["acer"]))
    order = sorted(cells, key=lambda k: (NETWORKS.index(k[0]), k[1], STATES.index(k[2]), k[3]))
    return [{"network": k[0], "patience": k[1], "bn_state": k[2], "threshold": k[3], **cells[k],
             "text": f"{cells[k]['one_class']} of {cells[k]['models']}"} for k in order], models


def check_single_class(res, U):
    counts, models = single_class(U)
    if not same(get(res, "test_one_class"), counts):
        raise Refusal("results.json: test_one_class differs from the models of units.csv that decide every test "
                      "frame as one class")
    return counts, models


def rankings(U, c):
    """Graduatoria descrittiva (PROTOCOL.md §6 (g)): reti presenti in tutte le unità per ACER medio esatto, pari merito
    con lo stesso rango minimo; per unità i conteggi dei ranghi con la stessa regola."""
    full = [n for n in NETWORKS if all(n in U[c][u] for u in UNITS)]
    means = {n: mean(U[c][u][n]["acer"] for u in UNITS) for n in full}

    def rank_of(vals):
        return {n: 1 + sum(1 for m in vals if vals[m] < vals[n]) for n in vals}
    r = rank_of(means)
    order = sorted(full, key=lambda n: (r[n], NETWORKS.index(n)))
    counts = {n: {str(i): 0 for i in range(1, len(full) + 1)} for n in full}
    for u in UNITS:
        for n, k in rank_of({n: U[c][u][n]["acer"] for n in full}).items():
            counts[n][str(k)] += 1
    return {"order": [{"network": n, "rank": r[n], "mean_acer": float(means[n])} for n in order],
            "not_ranked": [n for n in NETWORKS if n not in full], "unit_rank_counts": counts, "units_not_counted": 0,
            "units": N_UNITS}, means


def check_rankings(res, U):
    out = {}
    for c in CONDITIONS:
        mine, means = rankings(U, c)
        path = f"descriptive|rankings|{c[0]}/{c[1]}/{c[2]}"
        if not same(get(res, path), mine):
            raise Refusal(f"results.json: {path.replace('|', '.')} differs from the ranking recomputed from units.csv")
        if c[0] == 4 and c[2] == "fixed_0.5" and not same(get(res, f"primary_analysis|rankings|{c[1]}"), mine):
            raise Refusal(f"results.json: primary_analysis.rankings.{c[1]} differs from the ranking recomputed from "
                          f"units.csv")
        out[c] = (mine, means)
    return out


def run_key(key):
    """Il nome del run: anet_<tag>_p<p> per le AttackNet, nuaa_patience_<p> per C2."""
    net, p = key
    return f"nuaa_patience_{p}" if net == "C2" else f"anet_{RUN_TAG[net]}_p{p}"


def check_run_status(key, r, res, st):
    """status.json di un run: impronta (registrata dalla valutazione per le AttackNet, fissata per C2), esito,
    nome, rete, impronta di public/SHA256SUMS."""
    net, p = key
    rid = ANET_RUNS[key] if net != "C2" else C2_RUNS[p][0]
    h = sha256(r["status.json"])
    if net != "C2":
        want = get(res, f"runs|{run_key(key)}")
        if not same(want, get(st, f"runs|{run_key(key)}", "status.json")) or not isinstance(want, dict) \
                or want.get("run_id") != rid or want.get("official") is not True:
            raise Refusal(f"results.json: runs.{run_key(key)} is not the official run {rid} recorded in status.json")
        pin = want.get("status_sha256")
    else:
        pin = C2_RUNS[p][1]
    if h != pin:
        raise Refusal(f"status.json of {rid}: SHA-256 {h} differs from the pinned {pin}")
    rs = parse_json(r["status.json"], f"status.json of {rid}")
    for path, value in (("run_id", rid), ("status", "PASS"), ("passed", True), ("official", True),
                        ("run", run_key(key)), ("public_sha256sums_sha256", sha256(r["SHA256SUMS"]))):
        expect(rs, path, value, f"status.json of {rid}")
    if net != "C2":
        expect(rs, "network", net, f"status.json of {rid}")
    return rid


def check_run(key, r, res, st, pres):
    """I conteggi a classe unica sul dev di un run (one_class.csv), riga per riga contro models.csv e, per le AttackNet,
    contro le unità di units.csv e not_evaluable di results.json. Restituisce le righe e i modelli a classe unica."""
    net, p = key
    rid = check_run_status(key, r, res, st)
    nets = REUSED if net == "C2" else (net,)
    rows = read_csv(r["one_class.csv"], f"one_class.csv of {rid}", ONE_CLASS_HEADER)
    want_rows = [(n, str(p), s) for n in nets for s in STATES]
    if [(x["network"], x["patience"], x["bn_state"]) for x in rows] != want_rows:
        raise Refusal(f"one_class.csv of {rid}: the rows are not {want_rows}")
    mtext = r["models.csv"].decode("utf-8")
    mrows = list(csv.DictReader(io.StringIO(mtext, newline="")))
    if not mrows or any(c not in mrows[0] for c in MODELS_COLUMNS):
        raise Refusal(f"models.csv of {rid}: missing columns")
    out, named = [], []
    for x in rows:
        n, state = x["network"], x["bn_state"]
        where = f"one_class.csv of {rid} ({n}, {state})"
        vals = {}
        for k in ("one_class", "evaluable", "not_evaluable", "all_bona_fide", "all_attack"):
            if not INT_RE.fullmatch(x[k]):
                raise Refusal(f"{where}: {k} is not an integer")
            vals[k] = int(x[k])
        if vals["one_class"] != vals["all_attack"] + vals["all_bona_fide"]:
            raise Refusal(f"{where}: one_class is not all_attack + all_bona_fide")
        if vals["evaluable"] + vals["not_evaluable"] != N_UNITS:
            raise Refusal(f"{where}: evaluable + not_evaluable is not {N_UNITS}")
        if x["text"] != f"{vals['one_class']} of {vals['evaluable']} evaluable":
            raise Refusal(f"{where}: text differs from the counts")
        mine = [m for m in mrows if m["network"] == n]
        if sorted((m["fold"], m["seed"]) for m in mine) != sorted((str(f), str(s)) for f, s in UNITS) \
                or any(m["patience"] != str(p) for m in mine):
            raise Refusal(f"models.csv of {rid}: {n} does not hold the 12 fold-seed models at patience {p}")
        col = f"{state.replace('-', '_')}_one_class_05"
        complete = [m for m in mine if m["model_status"] == "complete"]
        if any(m[col] not in ("no", "attack", "bona fide") for m in complete):
            raise Refusal(f"models.csv of {rid}: {col} of a complete model is not no, attack or bona fide")
        got = {"evaluable": len(complete), "not_evaluable": N_UNITS - len(complete),
               "all_attack": sum(m[col] == "attack" for m in complete),
               "all_bona_fide": sum(m[col] == "bona fide" for m in complete)}
        if any(vals[k] != v for k, v in got.items()):
            raise Refusal(f"{where}: the counts differ from models.csv")
        units = {(int(m["fold"]), int(m["seed"])) for m in complete}
        if units != set(pres[(n, p)]):
            raise Refusal(f"{where}: the evaluable models differ from the units of units.csv at patience {p}")
        out.append({"split": "dev", "network": n, "patience": p, "bn_state": state, "threshold": "fixed_0.5",
                    "models": vals["evaluable"], "one_class": vals["one_class"], "all_attack": vals["all_attack"],
                    "all_bona_fide": vals["all_bona_fide"], "not_evaluable": vals["not_evaluable"]})
        named += [(n, p, state, int(m["fold"]), int(m["seed"]), m[col]) for m in complete if m[col] != "no"]
    return out, named


def check_runs(data, res, st, pres):
    """I conteggi sul dev delle cinque reti × 2 patience × 2 stati, nell'ordine rete, patience, stato."""
    rows, named = [], []
    for key, r in data["runs"].items():
        o, nm = check_run(key, r, res, st, pres)
        rows += o
        named += nm
    rows.sort(key=lambda x: (NETWORKS.index(x["network"]), x["patience"], STATES.index(x["bn_state"])))
    named.sort(key=lambda x: (NETWORKS.index(x[0]), x[1], STATES.index(x[2]), x[3], x[4]))
    return rows, named


def analyse(data):
    """Tutti i controlli e i valori delle uscite; nessuna scrittura."""
    res = parse_json(data["results.json"], "results.json")
    st = parse_json(data["status.json"], "status.json")
    check_status(st, res, data["_public_sums_sha256"])
    frames, ids = check_header(res, st)
    folds = check_denominators(res, frames)
    U = load_units(data["units.csv"], frames, ids, folds)
    pres = presence(U)
    reasons = check_not_evaluable(res, pres)
    check_models(res, U, pres)
    analysis = check_analysis(res, U, pres, reasons)
    check_c2_opposite(res)
    valid = check_valid_resamples(analysis)
    check_table(data["table.csv"], U, folds)
    check_conditions_csv(data["conditions.csv"], res)
    counts, models = check_single_class(res, U)
    ranks = check_rankings(res, U)
    dev_counts, dev_models = check_runs(data, res, st, pres)
    return {"U": U, "folds": folds, "analysis": analysis, "valid": valid, "single_counts": counts,
            "single_models": models, "ranks": ranks, "not_evaluable": get(res, "not_evaluable"), "res": res,
            "dev_counts": dev_counts, "dev_models": dev_models}


# ---------------------------------------------------------------------------------------------------------------------
# CSV aggregato (unica fonte di tabella, appendici e figura)

def agg_rows(a):
    """Le righe del CSV aggregato: tassi delle reti (8 condizioni; tutte le unità, per seed e per unità) e quantità
    delle 4 coppie (tutte le unità e per seed con medie, intervalli copiati e conteggi; per unità con i valori esatti)."""
    U, folds, res = a["U"], a["folds"], a["res"]
    rows = []
    for c in CONDITIONS:
        p, st, t = c
        analysis_name = analysis_of(p, t)
        for n in NETWORKS:
            for scope in SCOPES:
                r = rates_over(U, c, n, scope_units(scope), folds)
                row = {"analysis": analysis_name, "patience": str(p), "threshold": t, "quantity": "rate", "network": n,
                       "bn_state": st, "scope": scope, "units": str(r["units"]),
                       "computed": "1" if r["units"] == len(scope_units(scope)) else "0",
                       "source": "units.csv (mean of the exact per-unit fractions; denominators from results.json)"}
                if r["mean"] is not None:
                    for m in METRICS:
                        row[m], row[f"{m}_exact"] = repr(float(r["mean"][m])), exact_text(r["mean"][m])
                    row.update(n_attack=str(r["n_attack"]), n_bona_fide=str(r["n_bona_fide"]),
                               n_subjects=str(r["n_subjects"]))
                rows.append(row)
            for u in UNITS:
                if n not in U[c][u]:
                    continue
                row = {"analysis": analysis_name, "patience": str(p), "threshold": t, "quantity": "rate", "network": n,
                       "bn_state": st, "scope": "unit", "fold": str(u[0]), "seed": str(u[1]), "units": "1",
                       "computed": "1", "n_attack": str(folds[u[0]][1]), "n_bona_fide": str(folds[u[0]][2]),
                       "n_subjects": str(folds[u[0]][3]), "source": "units.csv"}
                for m in METRICS:
                    v = U[c][u][n][m]
                    row[m], row[f"{m}_exact"] = repr(float(v)), exact_text(v)
                rows.append(row)
    for p, t in PAIRS:
        analysis_name = analysis_of(p, t)
        an = a["analysis"][(p, t)]
        blk, path = pair_block(res, p, t)
        for quantity, nets, states in (("variation", NETWORKS, (VARIATION_STATE,)),
                                       ("contrast", ATTACKNETS, (VARIATION_STATE,)),
                                       ("gap_vs_resnet18", OURS, STATES),
                                       ("variation_difference_vs_resnet18", OURS, (VARIATION_STATE,))):
            for n in nets:
                status = an["guardian"][n]
                for st in states:
                    base = {"analysis": analysis_name, "patience": str(p), "threshold": t, "quantity": quantity,
                            "network": n, "bn_state": st}
                    for scope in SCOPES:
                        row = dict(base, scope=scope, units=str(len(scope_units(scope))))
                        if not status["computed"]:
                            row.update(computed="0", note=status["text"])
                            if quantity == "contrast" and scope == "all seeds" and analysis_name == "primary":
                                row["reading"] = "not computed"
                            rows.append(row)
                            continue
                        q = an["scopes"][scope][(quantity, n, st)]
                        row["computed"] = "1"
                        for m in METRICS:
                            low, high, valid = q["interval"][m]
                            row[m], row[f"{m}_exact"] = repr(float(q["mean"][m])), exact_text(q["mean"][m])
                            row[f"{m}_low"] = "" if low is None else repr(low)
                            row[f"{m}_high"] = "" if high is None else repr(high)
                            row[f"{m}_valid_resamples"] = str(valid)
                        row.update({k: str(q["counts"][k]) for k in ("below_zero", "above_zero", "zero")})
                        if "reading" in q and scope == "all seeds":
                            row["reading"], row["reading_reason"] = q["reading"], q["reason"]
                        sub = "" if scope == "all seeds" else f".per_seed.{scope.split()[1]}"
                        group = {"variation": "variations", "contrast": "contrasts"}.get(quantity,
                                                                                          "descriptive_vs_resnet18")
                        row["source"] = (f"results.json {path}{sub}.{group}.{n} (mean and counts checked against "
                                         f"units.csv; interval copied: subject bootstrap, not recomputable from public "
                                         f"files)")
                        rows.append(row)
                    if not status["computed"]:
                        continue
                    for u in UNITS:
                        v = unit_value(U, p, t, u, terms_of(quantity, n, st if st in STATES else None))
                        row = dict(base, scope="unit", fold=str(u[0]), seed=str(u[1]), units="1", computed="1",
                                   source="units.csv")
                        for m in METRICS:
                            row[m], row[f"{m}_exact"] = repr(float(v[m])), exact_text(v[m])
                        rows.append(row)
    return rows


def csv_bytes(rows, header):
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=header, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in header})
    return buf.getvalue().encode("utf-8")


class Agg:
    """Il CSV aggregato riletto dai suoi byte: le uscite si costruiscono da qui, non dalle strutture in memoria."""

    def __init__(self, b):
        self.rows = list(csv.DictReader(io.StringIO(b.decode("utf-8"), newline="")))
        self.index = {}
        for r in self.rows:
            key = (r["patience"], r["threshold"], r["quantity"], r["network"], r["bn_state"], r["scope"], r["fold"],
                   r["seed"])
            self.index[key] = r

    def row(self, p, t, quantity, n, st, scope="all seeds", unit=None):
        f, s = ("", "") if unit is None else (str(unit[0]), str(unit[1]))
        return self.index.get((str(p), t, quantity, n, st, scope, f, s))


def val(r, m):
    """Il valore esatto di una cella (frazione), None se vuota."""
    return Fraction(r[f"{m}_exact"]) if r and r[f"{m}_exact"] else None


def bound(r, m, side):
    return float(r[f"{m}_{side}"]) if r and r[f"{m}_{side}"] else None


# ---------------------------------------------------------------------------------------------------------------------
# uscite

def points(x):
    """Proporzione → punti percentuali, 2 decimali, metà lontano dallo zero, sul valore esatto (unica regola)."""
    if isinstance(x, float):
        d = Decimal(repr(x))
    elif isinstance(x, Fraction):
        q = abs(x) * 10000
        n = q.numerator // q.denominator
        if (q - n) * 2 >= 1:
            n += 1
        d = Decimal(-n if x < 0 else n).scaleb(-4)
    else:
        raise TypeError(type(x))
    return (d * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def fmt(x, signed=False, minus="-"):
    """Il segno è quello del valore esatto, anche quando l'arrotondamento dà 0.00. minus: '-' nei CSV e nel .tex,
    U+2212 altrove. None → testo vuoto."""
    if x is None:
        return ""
    s = f"{abs(points(x)):.2f}"
    if x < 0:
        return minus + s
    return ("+" + s) if signed and x > 0 else s


def fmt_u(x, signed=False):
    return fmt(x, signed, MINUS)


def mean_interval(r, m):
    """«media [low, high]» in punti, U+2212; «no interval» se non ci sono ricampionamenti validi."""
    low, high = bound(r, m, "low"), bound(r, m, "high")
    iv = "[no interval]" if low is None else f"[{fmt_u(low, True)}, {fmt_u(high, True)}]"
    return f"{fmt_u(val(r, m), True)} {iv}"


def wrap(text, first, rest="   "):
    """Frase generata spezzata a 117 colonne (rientro di 3 spazi dopo la prima riga)."""
    return textwrap.wrap(text, width=117, initial_indent=first, subsequent_indent=rest if first else "",
                         break_long_words=False, break_on_hyphens=False)


def valid_text(v):
    return f"{v['acer']:,} for ACER, {v['apcer']:,} for APCER and {v['bpcer']:,} for BPCER"


def near_zero_text(rows):
    """Valori non nulli che si arrotondano a 0.00: il segno resta; si dice quale valore è e quanto vale."""
    hits = []
    for label, r in rows:
        for m in METRICS:
            for part, x in (("mean", val(r, m)), ("lower bound", bound(r, m, "low")),
                            ("upper bound", bound(r, m, "high"))):
                if x is not None and x != 0 and points(x) == 0:
                    v = (Decimal(x.numerator) / Decimal(x.denominator) if isinstance(x, Fraction)
                         else Decimal(repr(x))) * 100
                    hits.append(f"{part} of the {m.upper()} {label}: "
                                f"{(MINUS if v < 0 else '+')}{abs(v).quantize(Decimal('0.0001'), ROUND_HALF_UP)} "
                                f"points")
    if not hits:
        return None
    return ("A value written " + MINUS + "0.00 or +0.00 is non-zero and rounds to zero; its sign is kept ("
            + "; ".join(hits) + ").")


def denominators_text(a):
    f = a["folds"]
    parts = [f"fold {k}: {f[k][0]:,} frames ({f[k][1]:,} attack, {f[k][2]:,} bona fide), {f[k][3]} subjects"
             for k in FOLDS]
    return (f"Test frames per fold (`denominators` of `results.json`, checked against `units.csv`): "
            f"{'; '.join(parts)}. Each fold is scored by its three seeds, so the attack and bona fide frames of a "
            f"network are summed over its units (the APCER and BPCER denominators); the {N_SUBJECTS} subjects are "
            f"pooled over the folds. Summed over the {N_UNITS} units: "
            f"{sum(f[u[0]][0] for u in UNITS):,} test frames, {sum(f[u[0]][1] for u in UNITS):,} attack and "
            f"{sum(f[u[0]][2] for u in UNITS):,} bona fide; each test frame counts once per seed "
            f"({sum(f[k][0] for k in FOLDS):,} distinct test frames: {sum(f[k][1] for k in FOLDS):,} attack and "
            f"{sum(f[k][2] for k in FOLDS):,} bona fide).")


def single_class_text(a, conditions):
    """Frase sui modelli che decidono una sola classe sul test nelle condizioni date (dai dati)."""
    ms = [x for x in a["single_models"] if x[0] in conditions]
    if not ms:
        return "No model decides every test frame as the same class in these conditions."
    parts = [f"{n} fold {u[0]} seed {u[1]}, patience {c[0]}, {c[1]} BN, {THRESHOLD_LABEL[c[2]]}: every frame {cls}"
             for c, u, n, cls, _ in ms]
    acers = sorted({exact_text(x) for *_, x in ms})
    inside = ""
    states = sorted({c[1] for c, *_ in ms}, key=STATES.index)
    if all(n == REFERENCE and (c[0], c[2]) == PRIMARY for c, _, n, *_ in ms):
        # modelli di LivenessNet nella condizione primaria: stanno nella media di LivenessNet, quindi nella sua
        # variazione e in ogni contrasto
        who = "both are" if len(ms) == 2 else ("it is" if len(ms) == 1 else f"all {len(ms)} are")
        where = f"the {states[0]}-BN mean" if len(states) == 1 else "the means in both BN states"
        inside = f"; {who} inside {where} of LivenessNet, and so inside its variation and every contrast"
    return (f"{len(ms)} model{'s' if len(ms) != 1 else ''} decide{'s' if len(ms) == 1 else ''} every test frame as "
            f"the same class ({'; '.join(parts)}; ACER {', '.join(acers)}); they enter the means like the others"
            f"{inside}: nothing was excluded after the test was seen.")


def dev_same_text(a):
    """Se i modelli a classe unica sul dev (condizione primaria) sono gli stessi del test, con la stessa classe, lo dice;
    altrimenti rimanda all'appendice B (dai dati)."""
    test = sorted((n, c[1], u[0], u[1], cls) for c, u, n, cls, _ in a["single_models"] if (c[0], c[2]) == PRIMARY)
    dev = sorted((n, st, f, s, cls) for n, p, st, f, s, cls in a["dev_models"] if p == PRIMARY[0])
    if test and test == dev:
        classes = sorted({x[4] for x in test})
        who = {1: "the same model", 2: "the same two models"}.get(len(test), f"the same {len(test)} models")
        verb = "decide" if len(test) > 1 else "decides"
        return (f"{who[0].upper()}{who[1:]} {verb} every dev frame as {' or '.join(classes)} at threshold 0.5 "
                f"(Appendix B).")
    return "The one-class counts on the dev are in Appendix B."


def not_evaluable_text(a):
    ne = a["not_evaluable"]
    g = {pk: a["analysis"][pk]["guardian"] for pk in PAIRS}
    short = [(n, p) for p in PATIENCES for n in NETWORKS if g[(p, "fixed_0.5")][n]["pairs"] != N_UNITS]
    pairs = (f"every network has {N_UNITS} of {N_UNITS} fold-seed pairs evaluable in both BN states at patience 4 "
             f"and at patience 12" if not short else
             "; ".join(f"{n} {g[(p, 'fixed_0.5')][n]['pairs']} of {N_UNITS} fold-seed pairs evaluable in both BN "
                       f"states at patience {p}" for n, p in short) + "; every other network and patience 12 of 12")
    if not ne:
        return f"Not evaluable models: 0 (`not_evaluable` of `results.json`); {pairs} (guardian of `results.json`)."
    names = "; ".join(f"{e['name']} at patience {e['patience']} ({e['reason']})" for e in ne)
    return (f"Not evaluable models: {len(ne)} (`not_evaluable` of `results.json`): {names}; {pairs} (guardian of "
            f"`results.json`).")


def thousands(x):
    """Un intero scritto con la virgola delle migliaia (come il README dell'esito); vuoto resta vuoto."""
    return f"{int(x):,}" if x else x


def rates_table(agg, p, t, scope="all seeds"):
    lines = ["| network | BN state | APCER (%) | BPCER (%) | ACER (%) | units | attack frames | bona fide frames | "
             "subjects |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n in NETWORKS:
        for st in STATES:
            r = agg.row(p, t, "rate", n, st, scope)
            lines.append(f"| {n} | {st} | {fmt_u(val(r, 'apcer'))} | {fmt_u(val(r, 'bpcer'))} | "
                         f"{fmt_u(val(r, 'acer'))} | {r['units']} | {thousands(r['n_attack'])} | "
                         f"{thousands(r['n_bona_fide'])} | {r['n_subjects']} |")
    return lines


def quantity_table(agg, p, t, quantity, nets, scope, first, reading=False, st=VARIATION_STATE):
    head = f"| {first} | APCER (points) | BPCER (points) | ACER (points) | ACER units above / below / at zero |"
    sep = "|---|---|---|---|---|"
    if reading:
        head += " reading (pre-registered rule) |"
        sep += "---|"
    lines = [head, sep]
    for n in nets:
        r = agg.row(p, t, quantity, n, st, scope)
        if r["computed"] != "1":
            lines.append(f"| {n} | {r['note']} | | | |" + (" not computed |" if reading else ""))
            continue
        cells = " | ".join(mean_interval(r, m) for m in METRICS)
        line = f"| {n} | {cells} | {r['above_zero']} / {r['below_zero']} / {r['zero']} |"
        if reading:
            line += f" \"{r['reading']}\" ({r['reading_reason']}) |"
        lines.append(line)
    return lines


def primary_md(a, agg):
    p, t = PRIMARY
    rows = [(f"variation of {n}", agg.row(p, t, "variation", n, VARIATION_STATE)) for n in NETWORKS] + \
           [(f"contrast of {k}", agg.row(p, t, "contrast", k, VARIATION_STATE)) for k in ATTACKNETS]
    nz = near_zero_text([(lab, r) for lab, r in rows if r["computed"] == "1"])
    res = a["res"]
    conclusions = [f"{k}: \"{get(res, f'primary_analysis|contrasts|{k}|conclusion')}\"" for k in ATTACKNETS]
    out = [
        "# Table 1. Five networks on the NUAA test: patience 4, threshold 0.5, two BN states",
        "",
        f"Generated by `make_table_figure.py` from `aggregate.csv`, itself built from the public files of the "
        f"evaluation",
        f"`{RUN_ID}`. Values in percentage points: the proportions (0 to 1) of the source files multiplied by 100",
        "and rounded to two decimals, half away from zero, from the exact value. Full precision in `aggregate.csv`.",
        "The rows follow the fixed order of `PROTOCOL.md` (LivenessNet, the three AttackNets, ResNet-18 as external",
        "reference); the order is not a ranking.",
        "",
        "**(a)** APCER, BPCER and ACER of each network: mean over the 12 fold-seed units (4 folds × seeds 42, 43, 44),",
        "with the denominators summed over the units of the network.",
        "",
        *rates_table(agg, p, t),
        "",
        "**(b)** ACER variation of each network: metric with re-estimated BN − metric with ordinary BN, mean over the",
        "same units [subject interval]; ACER units: how many of the 12 unit variations are above, below and at zero.",
        "",
        *quantity_table(agg, p, t, "variation", NETWORKS, "all seeds", "network"),
        "",
        "**(c)** Contrast of each AttackNet: its variation − the variation of LivenessNet, per unit on the same units",
        "and subjects, mean [subject interval]. Primary analysis; the reading follows the rule written before the test.",
        "",
        *quantity_table(agg, p, t, "contrast", ATTACKNETS, "all seeds", "AttackNet", reading=True),
        "",
        "Notes.",
        "",
        "1. Patience 4, fixed threshold 0.5 (effective threshold 0.5 for every model in both BN states, checked in",
        "   `units.csv`). BN states: ordinary, and re-estimated on the fitting set only. The means of (a) and every",
        "   mean of (b) and (c) are computed from the exact fractions of `units.csv`; the means of (b) and (c) are",
        "   also those of `results.json` (the script stops if they differ).",
        *wrap(f"Subject intervals: percentiles 2.5–97.5 of 1,000 resamples of the {N_SUBJECTS} test subjects pooled "
              f"over the 4 folds, the same resamples for every network and both BN states; valid resamples "
              f"{valid_text(a['valid'])}, in every block (checked). They are individual 95% intervals: no correction "
              f"for multiplicity, no simultaneous coverage over the contrasts, variations or comparisons. They are "
              f"copied from `results.json`: they cannot be recomputed from the public files, which hold no "
              f"per-subject counts. No interval is published for the rates of (a).", "2. "),
        "3. Reading of each contrast (`PROTOCOL.md` §6 (c), from the interval of its ACER over the three seeds",
        "   together): \"... lower than zero\" if the interval lies entirely below zero, \"... higher than zero\" if",
        "   entirely above, \"unresolved\" if it contains zero (a bound at zero contains it). One conclusion per",
        "   AttackNet, no joint statement on the three (quoted from `results.json`):",
        *[ln for c in conclusions for ln in wrap(c, "   - ", "     ")],
        "   The five variations are reported with their intervals and no reading.",
        *wrap(denominators_text(a), "4. "),
        *wrap(single_class_text(a, [(4, st, "fixed_0.5") for st in STATES]) + " " + dev_same_text(a)
              + " Counts for every condition in `appendix_single_class.md`.", "5. "),
        *wrap(not_evaluable_text(a), "6. "),
        "7. LivenessNet and ResNet-18 are the 48 checkpoints of C2, already read on this test; in each contrast only",
        "   the AttackNet side is new (`PROTOCOL.md` §5). Per-seed values: `appendix_per_seed.md`; patience 12, the",
        "   EER threshold, the comparison with ResNet-18 and the rankings: `descriptive.md` (descriptive analyses).",
    ]
    out += wrap(rounding_text(agg), "8. ")
    if nz:
        out += wrap(nz, "9. ")
    out.append("")
    return "\n".join(out).encode("utf-8")


def rounding_text(agg):
    """Nota 8 (dai dati): le variazioni e i contrasti sono calcolati dalle frazioni esatte; per ogni metrica si
    confrontano la variazione arrotondata con la differenza delle celle arrotondate di (a), il contrasto arrotondato con
    la differenza delle differenze di (a) e con la differenza delle variazioni arrotondate di (b); si elenca ogni caso
    diverso."""
    p, t = PRIMARY
    cases = []

    def rp(r, m):
        return points(val(r, m))
    for m in METRICS:
        rate = {(n, st): rp(agg.row(p, t, "rate", n, st), m) for n in NETWORKS for st in STATES}
        var = {}
        for n in NETWORKS:
            r = agg.row(p, t, "variation", n, VARIATION_STATE)
            if r["computed"] != "1":
                continue
            var[n] = rp(r, m)
            d = rate[(n, "re-estimated")] - rate[(n, "ordinary")]
            if d != var[n]:
                cases.append(f"{m.upper()} variation of {n}: {fmt_dec(var[n])} in (b), {fmt_dec(d)} from the "
                             f"cells of (a)")
        for k in ATTACKNETS:
            r = agg.row(p, t, "contrast", k, VARIATION_STATE)
            if r["computed"] != "1" or REFERENCE not in var:
                continue
            c = rp(r, m)
            d1 = (rate[(k, "re-estimated")] - rate[(k, "ordinary")]) - \
                (rate[(REFERENCE, "re-estimated")] - rate[(REFERENCE, "ordinary")])
            d2 = var[k] - var[REFERENCE]
            if d1 != c:
                cases.append(f"{m.upper()} contrast of {k}: {fmt_dec(c)} in (c), {fmt_dec(d1)} from the cells of (a)")
            if d2 != c:
                cases.append(f"{m.upper()} contrast of {k}: {fmt_dec(c)} in (c), {fmt_dec(d2)} from the variations "
                             f"of (b)")
    head = ("Variations and contrasts are computed from the exact fractions: a difference of rounded values can differ "
            "from them (here by 0.01)")
    if not cases:
        return head + "; in this table no such difference occurs."
    return f"{head} ({'; '.join(cases)})."


def fmt_dec(d):
    """Un valore già arrotondato (Decimal), con segno, U+2212 per il meno."""
    return (MINUS if d < 0 else "+" if d > 0 else "") + f"{abs(d):.2f}"


def tex_num(x, signed=False):
    s = fmt(x, signed)
    return f"${s}$" if s.startswith("-") or s.startswith("+") else s


def primary_tex(agg):
    p, t = PRIMARY
    out = [f"% Table 1, generated by make_table_figure.py from aggregate.csv (evaluation {RUN_ID}).",
           "% Percentage points, two decimals, half away from zero; individual 95% subject intervals; notes and",
           "% readings in primary_table.md.",
           "\\begin{tabular}{llrrrrrrr}", "\\hline",
           "Network & BN state & APCER (\\%) & BPCER (\\%) & ACER (\\%) & Units & Attack & Bona fide & Subjects \\\\",
           "\\hline"]
    for n in NETWORKS:
        for st in STATES:
            r = agg.row(p, t, "rate", n, st)
            out.append(f"{n.replace('_', chr(92) + '_')} & {st} & {tex_num(val(r, 'apcer'))} & "
                       f"{tex_num(val(r, 'bpcer'))} & {tex_num(val(r, 'acer'))} & {r['units']} & {r['n_attack']} & "
                       f"{r['n_bona_fide']} & {r['n_subjects']} \\\\")
    out += ["\\hline", "\\end{tabular}", ""]
    for quantity, nets, title in (("variation", NETWORKS, "Variation (re-estimated $-$ ordinary BN)"),
                                  ("contrast", ATTACKNETS, "Contrast with LivenessNet")):
        out += ["\\begin{tabular}{llll}", "\\hline",
                f"{title} & APCER (points) & BPCER (points) & ACER (points) \\\\", "\\hline"]
        for n in nets:
            r = agg.row(p, t, quantity, n, VARIATION_STATE)
            name = n.replace("_", "\\_")
            if r["computed"] != "1":
                out.append(f"{name} & \\multicolumn{{3}}{{l}}{{not computed}} \\\\")
                continue
            cells = []
            for m in METRICS:
                low, high = bound(r, m, "low"), bound(r, m, "high")
                iv = "[no interval]" if low is None else f"[{tex_num(low, True)}, {tex_num(high, True)}]"
                cells.append(f"{tex_num(val(r, m), True)} {iv}")
            out.append(f"{name} & {' & '.join(cells)} \\\\")
        out += ["\\hline", "\\end{tabular}", ""]
    return "\n".join(out).encode("utf-8")


def per_seed_md(a, agg):
    p, t = PRIMARY
    out = [
        "# Appendix A. Per seed, per unit and per fold: patience 4, threshold 0.5",
        "",
        "Reported apart, outside the reading (`PROTOCOL.md` §6 (d)). Values in percentage points, as in Table 1; full",
        "precision in `aggregate.csv`. Each seed has 4 units, one per fold.",
        "",
    ]
    nz_rows = []
    for s in SEEDS:
        scope = f"seed {s}"
        out += [f"## Seed {s}", "",
                "**(a)** APCER, BPCER and ACER of each network, mean over the 4 units of the seed.", "",
                *rates_table(agg, p, t, scope), "",
                "**(b)** ACER variations and contrasts, mean over the 4 units of the seed [subject interval]. The",
                "intervals are those of `primary_analysis.per_seed` of `results.json`, over the same 16 subjects; no",
                "reading is given per seed.", "",
                *quantity_table(agg, p, t, "variation", NETWORKS, scope, "variation of"), "",
                *quantity_table(agg, p, t, "contrast", ATTACKNETS, scope, "contrast of"), ""]
        nz_rows += [(f"variation of {n} (seed {s})", agg.row(p, t, "variation", n, VARIATION_STATE, scope))
                    for n in NETWORKS]
        nz_rows += [(f"contrast of {k} (seed {s})", agg.row(p, t, "contrast", k, VARIATION_STATE, scope))
                    for k in ATTACKNETS]
    out += ["## Per unit", "",
            "Each network × fold × seed: APCER, BPCER and ACER with ordinary (ord.) and re-estimated (re.) BN, the",
            "ACER variation and, for each AttackNet, its ACER contrast with LivenessNet in the same unit. No interval.",
            "",
            "| fold | seed | network | APCER ord. | BPCER ord. | ACER ord. | APCER re. | BPCER re. | ACER re. | ACER "
            "variation | ACER contrast |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for u in UNITS:
        for n in NETWORKS:
            o = agg.row(p, t, "rate", n, "ordinary", "unit", u)
            r = agg.row(p, t, "rate", n, "re-estimated", "unit", u)
            if o is None or r is None:
                out.append(f"| {u[0]} | {u[1]} | {n} | not evaluable | | | | | | | |")
                continue
            v = agg.row(p, t, "variation", n, VARIATION_STATE, "unit", u)
            c = agg.row(p, t, "contrast", n, VARIATION_STATE, "unit", u) if n in ATTACKNETS else None
            out.append(f"| {u[0]} | {u[1]} | {n} | {fmt_u(val(o, 'apcer'))} | {fmt_u(val(o, 'bpcer'))} | "
                       f"{fmt_u(val(o, 'acer'))} | {fmt_u(val(r, 'apcer'))} | {fmt_u(val(r, 'bpcer'))} | "
                       f"{fmt_u(val(r, 'acer'))} | {fmt_u(val(v, 'acer'), True) if v else ''} | "
                       f"{fmt_u(val(c, 'acer'), True) if c else ''} |")
    out += ["", *per_fold_lines(agg)]
    nz = near_zero_text([(lab, r) for lab, r in nz_rows if r["computed"] == "1"])
    if nz:
        out += wrap(nz, "") + [""]
    return "\n".join(out).encode("utf-8")


def unit_mean(agg, p, t, quantity, n, units):
    """Media esatta dell'ACER di una quantità sulle unità date, dalle righe per unità del CSV aggregato (None se la
    quantità non è calcolata)."""
    rows = [agg.row(p, t, quantity, n, VARIATION_STATE, "unit", u) for u in units]
    if any(r is None for r in rows):
        return None, None
    vals = [val(r, "acer") for r in rows]
    return sum(vals, Fraction(0)) / len(vals), vals


def per_fold_lines(agg):
    """Appendice A, per fold (scelta dell'autore): media sui 3 seed del fold delle variazioni e dei contrasti di ACER,
    con le unità sopra, sotto e a zero; nessun intervallo (results.json non ne ha per fold)."""
    p, t = PRIMARY
    out = ["## Per fold", "",
           "A choice of the author, outside the reading: for each fold, the mean over its 3 units (one per seed) of the",
           "ACER variations and contrasts, and how many of the 3 unit values are above, below and at zero. No interval:",
           "`results.json` holds none per fold.", "",
           "| fold | quantity | ACER (points) | units above / below / at zero |", "|---:|---|---:|---|"]
    for f in FOLDS:
        units = [u for u in UNITS if u[0] == f]
        for quantity, nets in (("variation", NETWORKS), ("contrast", ATTACKNETS)):
            for n in nets:
                m, vals = unit_mean(agg, p, t, quantity, n, units)
                if m is None:
                    out.append(f"| {f} | {quantity} of {n} | not computed | |")
                    continue
                out.append(f"| {f} | {quantity} of {n} | {fmt_u(m, True)} | {sum(v > 0 for v in vals)} / "
                           f"{sum(v < 0 for v in vals)} / {sum(v == 0 for v in vals)} |")
    out.append("")
    return out


def leave_one_out_lines(agg):
    """Appendice C, sensibilità togliendo un'unità alla volta (scelta dell'autore, descrittiva, non nel protocollo):
    medie sulle altre 11 unità delle cinque variazioni e dei tre contrasti di ACER, dal CSV aggregato."""
    p, t = PRIMARY
    cols = [("variation", n) for n in NETWORKS] + [("contrast", k) for k in ATTACKNETS]
    out = ["## Leaving one unit out", "",
           "Leaving one unit out (a choice of the author, descriptive, not in `PROTOCOL.md`): means over the other 11",
           "units; no interval, since subject intervals need the per-subject counts, which are not in the public files.",
           "Patience 4, threshold 0.5, ACER, percentage points.", "",
           "| unit left out | " + " | ".join(f"{'variation' if q == 'variation' else 'contrast'} of {n}"
                                             for q, n in cols) + " |",
           "|---|" + "---:|" * len(cols)]
    for left in [None] + list(UNITS):
        keep = [u for u in UNITS if u != left]
        cells = []
        for q, n in cols:
            m, _ = unit_mean(agg, p, t, q, n, keep)
            cells.append("not computed" if m is None else fmt_u(m, True))
        label = "none (all 12 units)" if left is None else f"fold {left[0]}, seed {left[1]}"
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out.append("")
    return out


THRESHOLD_LABEL = {"fixed_0.5": "threshold 0.5", "dev_eer": "dev EER threshold"}
CONDITION_LABEL = {(4, "fixed_0.5"): "patience 4, threshold 0.5", (12, "fixed_0.5"): "patience 12, threshold 0.5",
                   (4, "dev_eer"): "patience 4, dev EER threshold", (12, "dev_eer"): "patience 12, dev EER threshold"}


def single_class_csv(a):
    """Le 40 righe del test (not_evaluable = modelli mancanti da units.csv) e le 20 del dev (one_class.csv dei run)."""
    test = [dict(e, split="test", not_evaluable=N_UNITS - e["models"]) for e in a["single_counts"]]
    return csv_bytes([{k: str(e[k]) for k in SINGLE_HEADER} for e in test + a["dev_counts"]], SINGLE_HEADER)


def dev_lines(a):
    """Appendice B, sul dev: tabella rete × (patience, stato) e la frase con i modelli, dai run."""
    cols = [(p, st) for p in PATIENCES for st in STATES]
    counts = {(e["network"], e["patience"], e["bn_state"]): e for e in a["dev_counts"]}
    lines = ["| network | " + " | ".join(f"p{p}, {st}" for p, st in cols) + " |", "|---|" + "---:|" * len(cols)]
    for n in NETWORKS:
        cells = []
        for p, st in cols:
            e = counts[(n, p, st)]
            cells.append(f"{e['one_class']} of {e['models']} evaluable ({e['all_attack']} attack, "
                         f"{e['all_bona_fide']} bona fide)")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    dm = a["dev_models"]
    if dm:
        names = "; ".join(f"{n} fold {f} seed {sd}, patience {p}, {st} BN: every dev frame {cls}"
                          for n, p, st, f, sd, cls in dm)
        sentence = f"Models that decide every dev frame as one class at threshold 0.5 (`public/models.csv`): {names}."
    else:
        sentence = "No model decides every dev frame as one class at threshold 0.5 (`public/models.csv`)."
    return lines, sentence


def single_class_md(a):
    dev = dev_lines(a)
    counts = {(e["network"], e["patience"], e["bn_state"], e["threshold"]): e for e in a["single_counts"]}
    head = "| network | " + " | ".join(f"p{c[0]}, {c[1]}, {'0.5' if c[2] == 'fixed_0.5' else 'EER'}"
                                       for c in CONDITIONS) + " |"
    lines = [head, "|---|" + "---:|" * len(CONDITIONS)]
    for n in NETWORKS:
        cells = []
        for c in CONDITIONS:
            e = counts[(n,) + c]
            extra = []
            if e["all_attack"]:
                extra.append(f"{e['all_attack']} attack")
            if e["all_bona_fide"]:
                extra.append(f"{e['all_bona_fide']} bona fide")
            cells.append(f"{e['one_class']} of {e['models']}" + (f" ({', '.join(extra)})" if extra else ""))
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    out = [
        "# Appendix B. Models that decide every test or dev frame as one class, and not evaluable models",
        "",
        "Descriptive count of `PROTOCOL.md` §6 (f) on the test: per network, patience, BN state and threshold (0.5,",
        "or the EER threshold of the dev of the fold in the same BN state), the models that decide every test frame",
        "attack or every test frame bona fide (\"k of n\": k such models of the n evaluated). Computed by the script",
        "from `units.csv` (APCER 0 and BPCER 1: every frame attack; APCER 1 and BPCER 0: every frame bona fide; both",
        "classes are present in every fold) and checked against `test_one_class` of `results.json`; in",
        "`single_class.csv`. It enters no reading.",
        "",
        *lines,
        "",
        *wrap(single_class_text(a, CONDITIONS), ""),
        "",
        *wrap(not_evaluable_text(a), ""),
        "",
        "## On the dev (pre-registered, `PROTOCOL.md` §6 (f), first part)",
        "",
        "Per network, patience and BN state, at threshold 0.5 on the dev of the fold of each model: \"k of n evaluable",
        "(a attack, b bona fide)\", the models that decide every dev frame as one class. The dev count is computed by",
        "the C2 rule in each run and read from `public/one_class.csv` of the eight run records (six of C5, two of C2),",
        "checked against their `status.json`, the pins and `public/models.csv`; in `single_class.csv` (split `dev`).",
        "",
        *dev[0],
        "",
        *wrap(dev[1], ""),
        "",
    ]
    return "\n".join(out).encode("utf-8")


def descriptive_md(a, agg):
    out = [
        "# Appendix C. Descriptive analyses: patience 12, the dev EER threshold, the comparison with ResNet-18, leaving",
        "one unit out, rankings",
        "",
        "Descriptive analyses (`PROTOCOL.md` §6 (g), (m), (o)): reported with their intervals and no reading; not",
        "results of the primary analysis. Values in percentage points; intervals: individual 95% subject intervals,",
        f"copied from `results.json` (valid resamples {valid_text(a['valid'])}); means and unit counts",
        "recomputed from `units.csv`. The EER threshold is estimated on the dev of the fold (2–3 subjects) in the same",
        "BN state.",
        "",
    ]
    nz_rows = []
    for p, t in DESCRIPTIVE:
        lab = CONDITION_LABEL[(p, t)]
        out += [f"## {lab[0].upper()}{lab[1:]} (descriptive)", "",
                "APCER, BPCER and ACER of each network, mean over the 12 units:", "",
                *rates_table(agg, p, t), "",
                "ACER variations (re-estimated − ordinary BN) and contrasts (AttackNet − LivenessNet), mean [subject "
                "interval]:", "",
                *quantity_table(agg, p, t, "variation", NETWORKS, "all seeds", "variation of"), "",
                *quantity_table(agg, p, t, "contrast", ATTACKNETS, "all seeds", "contrast of"), ""]
        nz_rows += [(f"variation of {n} ({lab})", agg.row(p, t, "variation", n, VARIATION_STATE)) for n in NETWORKS]
        nz_rows += [(f"contrast of {k} ({lab})", agg.row(p, t, "contrast", k, VARIATION_STATE)) for k in ATTACKNETS]
    out += ["## Comparison with ResNet-18 (descriptive, `PROTOCOL.md` §6 (m))", "",
            "For each of LivenessNet and the three AttackNets: the gap metric(network) − metric(ResNet-18) in each BN",
            "state and the difference of the variations V(network) − V(ResNet-18), mean [subject interval]. For",
            "LivenessNet these are the C2 gaps and contrast with the opposite sign (checked against the reproduction",
            "of C2 recorded in `results.json`). No reading.", ""]
    for p, t in PAIRS:
        out += [f"**{CONDITION_LABEL[(p, t)]}**", "",
                "| network | quantity | APCER (points) | BPCER (points) | ACER (points) | ACER units above / below / at "
                "zero |", "|---|---|---|---|---|---|"]
        for n in OURS:
            for quantity, st, label in (("gap_vs_resnet18", "ordinary", "gap, ordinary BN"),
                                        ("gap_vs_resnet18", "re-estimated", "gap, re-estimated BN"),
                                        ("variation_difference_vs_resnet18", VARIATION_STATE,
                                         "variation difference")):
                r = agg.row(p, t, quantity, n, st)
                if r["computed"] != "1":
                    out.append(f"| {n} | {label} | {r['note']} | | | |")
                    continue
                nz_rows.append((f"{label} of {n} ({CONDITION_LABEL[(p, t)]})", r))
                out.append(f"| {n} | {label} | {' | '.join(mean_interval(r, m) for m in METRICS)} | "
                           f"{r['above_zero']} / {r['below_zero']} / {r['zero']} |")
        out.append("")
    out += leave_one_out_lines(agg)
    out += ["## Rankings (descriptive, `PROTOCOL.md` §6 (g))", "",
            "Not a result: no interval. Per condition the networks present in every unit, ordered by mean ACER (exact",
            "fractions; ties share the lowest rank), and, per network, how many of the 12 units give it each rank",
            "(same rule on the unit ACER). Recomputed from `units.csv` and checked against `results.json`.", "",
            "| condition | BN state | order by mean ACER (%) | units at rank 1 / 2 / 3 / 4 / 5 |", "|---|---|---|---|"]
    for c in sorted(CONDITIONS, key=lambda c: (c[0], THRESHOLDS.index(c[2]), STATES.index(c[1]))):
        mine, means = a["ranks"][c]
        order = ", ".join(f"{e['rank']}. {e['network']} {fmt_u(means[e['network']])}" for e in mine["order"])
        counts = "; ".join(f"{n} " + " / ".join(str(mine["unit_rank_counts"][n][str(i)])
                                                for i in range(1, len(mine["order"]) + 1))
                           for n in NETWORKS if n in mine["unit_rank_counts"])
        nr = f" Not ranked: {', '.join(mine['not_ranked'])}." if mine["not_ranked"] else ""
        out.append(f"| {CONDITION_LABEL[(c[0], c[2])]} | {c[1]} | {order}.{nr} | {counts} |")
    out.append("")
    nz = near_zero_text([(lab, r) for lab, r in nz_rows if r["computed"] == "1"])
    if nz:
        out += wrap(nz, "") + [""]
    return "\n".join(out).encode("utf-8")


# disposizione verticale dei punti delle unità (regola deterministica, dichiarata nel README)
LEVEL_GAP_POINTS = 2.0   # due punti più vicini di 2,0 punti percentuali sull'asse vanno su livelli diversi
LEVEL_BASE, LEVEL_STEP = 0.15, 0.065
LABEL_OFFSET = 0.74      # testo di ogni riga sopra la barra, sopra i punti impilati
ROW_STEP = 1.3           # distanza verticale fra due righe


def unit_levels(values):
    """Livello di ciascun punto: in ordine crescente di valore (pareggi per fold, seed), il primo livello senza un punto
    a meno di LEVEL_GAP_POINTS."""
    order = sorted(range(len(values)), key=lambda i: (values[i][0], values[i][1]))
    last = {}
    levels = [0] * len(values)
    for i in order:
        x = values[i][0]
        lvl = 0
        while lvl in last and x - last[lvl] < LEVEL_GAP_POINTS:
            lvl += 1
        last[lvl] = x
        levels[i] = lvl
    return levels


def figure_items(agg):
    """Le righe della figura dal CSV aggregato: (titolo, media, low, high, valori delle unità con fold e seed)."""
    p, t = PRIMARY
    items = []
    for quantity, nets in (("variation", NETWORKS), ("contrast", ATTACKNETS)):
        for n in nets:
            r = agg.row(p, t, quantity, n, VARIATION_STATE)
            if r["computed"] != "1":
                items.append((quantity, n, None, None, None, []))
                continue
            units = []
            for u in UNITS:
                ur = agg.row(p, t, quantity, n, VARIATION_STATE, "unit", u)
                units.append((float(points(val(ur, "acer"))), u))
            items.append((quantity, n, val(r, "acer"), bound(r, "acer", "low"), bound(r, "acer", "high"), units))
    # ogni valore disegnato è quello stampato: punti percentuali arrotondati con la regola unica (points)
    return [(q, n, None if mu is None else float(points(mu)), None if lo is None else float(points(lo)),
             None if hi is None else float(points(hi)), units, mu, lo, hi) for q, n, mu, lo, hi, units in items]


def figure_images(agg):
    """PNG e PDF della figura dai soli valori del CSV aggregato; nessun metadato di software o di data. Larghezza
    prevista: una colonna, 3,5 pollici (89 mm), al 100%."""
    try:
        import matplotlib
    except ImportError:
        raise Refusal("matplotlib is not available: the figure cannot be generated (with --check, --text-only "
                      "compares the text outputs without it)")
    matplotlib.use("Agg", force=True)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    items = figure_items(agg)
    with matplotlib.rc_context():
        matplotlib.rcdefaults()
        matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6,
                                    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.labelsize": 7,
                                    "pdf.fonttype": 42, "axes.unicode_minus": True})
        fig = Figure(figsize=(3.5, 6.0), dpi=300)
        FigureCanvasAgg(fig)
        ax = fig.add_axes((0.04, 0.105, 0.92, 0.885))
        allv = [x for it in items for x in ([it[2], it[3], it[4]] if it[2] is not None else []) + [v for v, _ in it[5]]]
        # asse simmetrico: il primo multiplo di 10 che lascia almeno 3 punti di margine al valore più lontano
        lim = 10 * math.ceil((max(abs(v) for v in allv) + 3) / 10)
        # posizioni verticali: le cinque variazioni, uno spazio con il titolo del gruppo, i tre contrasti
        ypos, y = {}, 0.0
        for quantity, n, *_ in reversed(items):  # dal basso: contrasti, poi variazioni
            ypos[(quantity, n)] = y
            y += ROW_STEP
            if quantity == "contrast" and n == ATTACKNETS[0]:
                y += 0.45
        top = y - ROW_STEP + 1.3
        first = True
        for quantity, n, mu, low, high, units, mu_x, low_x, high_x in items:
            yy = ypos[(quantity, n)]
            ax.plot([0, 0], [yy - 0.15, yy + 0.4], color="0.3", lw=0.7, ls=(0, (4, 3)), zorder=1,
                    label="zero" if first else None)
            if mu is None:
                ax.text(0.0, yy + LABEL_OFFSET, f"{n}: not computed", transform=ax.get_yaxis_transform(), ha="left",
                        va="center", fontsize=7)
                continue
            levels = unit_levels(units)
            ax.plot([v for v, _ in units], [yy + LEVEL_BASE + LEVEL_STEP * lv for lv in levels], ls="none",
                    marker="o", ms=2.4, mfc="0.55", mec="0.2", mew=0.35, zorder=2,
                    label="fold-seed units (12)" if first else None)
            ax.plot([low, high], [yy, yy], color="black", lw=1.7, solid_capstyle="butt", zorder=3)
            for end in (low, high):
                ax.plot([end, end], [yy - 0.07, yy + 0.07], color="black", lw=1.0, zorder=3)
            ax.plot([mu], [yy], ls="none", marker="o", ms=3.8, color="black", zorder=4,
                    label="mean, 95% subject interval (individual)" if first else None)
            ax.text(0.0, yy + LABEL_OFFSET, f"{n}: {fmt_u(mu_x, True)} [{fmt_u(low_x, True)}, {fmt_u(high_x, True)}]",
                    transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.8)
            first = False
        gv = ypos[("variation", NETWORKS[0])] + 1.08
        gc = ypos[("contrast", ATTACKNETS[0])] + 1.08
        ax.text(0.0, gv, "ACER variation: re-estimated − ordinary BN",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.5, fontweight="bold")
        ax.text(0.0, gc, "Contrast: variation of the AttackNet − variation of LivenessNet",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.5, fontweight="bold")
        ax.axhline(ypos[("contrast", ATTACKNETS[0])] + 1.4, color="0.8", lw=0.5, zorder=0)
        ax.set_xlim(-lim, lim)
        # griglia verticale leggera ogni 10 punti (tacche principali ogni 20): aiuta a leggere la larghezza degli
        # intervalli senza cambiare la scala, che contiene tutti i punti
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.grid(axis="x", which="both", color="0.88", lw=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.3, top)
        ax.set_yticks([])
        ax.set_xlabel("ACER, percentage points (patience 4, threshold 0.5)", fontsize=7)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False, fontsize=6.0,
                  handletextpad=0.3, columnspacing=0.8, handlelength=1.6)
        png, pdf = io.BytesIO(), io.BytesIO()
        fig.savefig(png, format="png", dpi=300, metadata={"Software": None})
        fig.savefig(pdf, format="pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None})
    return png.getvalue(), pdf.getvalue()


def build(data, images=True):
    """Tutte le uscite in memoria: {nome: byte}. Prima il CSV aggregato, poi le altre uscite dai suoi byte riletti.
    Con images False, i soli file di testo."""
    try:
        a = analyse(data)
    except (AttributeError, KeyError, TypeError, ValueError, IndexError) as e:
        raise Refusal(f"unexpected structure of the input files ({type(e).__name__}: {e})")
    agg_b = csv_bytes(agg_rows(a), AGG_HEADER)
    agg = Agg(agg_b)
    out = {"aggregate.csv": agg_b, "primary_table.md": primary_md(a, agg), "primary_table.tex": primary_tex(agg),
           "appendix_per_seed.md": per_seed_md(a, agg), "appendix_single_class.md": single_class_md(a),
           "single_class.csv": single_class_csv(a), "descriptive.md": descriptive_md(a, agg)}
    if images:
        out["figure_acer_variations.png"], out["figure_acer_variations.pdf"] = figure_images(agg)
    return out


def png_structure_ok(b):
    """Firma, CRC di ogni chunk e IEND finale (un byte cambiato rompe almeno un CRC)."""
    if not b.startswith(PNG_SIGNATURE):
        return False
    i = len(PNG_SIGNATURE)
    while i + 12 <= len(b):
        n = int.from_bytes(b[i:i + 4], "big")
        kind = b[i + 4:i + 8]
        end = i + 12 + n
        if end > len(b):
            return False
        if zlib.crc32(b[i + 4:i + 8 + n]) != int.from_bytes(b[i + 8 + n:end], "big"):
            return False
        if kind == b"IEND":
            return end == len(b)
        i = end
    return False


def pdf_structure_ok(b):
    """Intestazione %PDF- e %%EOF finale (controllo di forma, non del contenuto)."""
    return b.startswith(b"%PDF-") and b.rstrip(b"\r\n").endswith(b"%%EOF")


# ---------------------------------------------------------------------------------------------------------------------
# comandi

def inside(path, root):
    """path (risolto) è root o sta dentro root, per identità dei file sugli antenati."""
    p = os.path.realpath(path)
    while True:
        if same_file(p, root):
            return True
        nxt = os.path.dirname(p)
        if nxt == p:
            return False
        p = nxt


def results_package_root(record):
    """La cartella del pacchetto dei risultati quando il record è <root>/runs/<RUN_ID>, altrimenti None."""
    p = Path(os.path.realpath(record))
    if p.name == RUN_ID and p.parent.name == "runs":
        return str(p.parent.parent)
    return None


def check_out_path(out, record, run_roots=()):
    """Regola delle cartelle di uscita: l'uscita non esiste; non è il record né sta dentro il record, il pacchetto dei
    risultati che lo contiene o le due radici dei run; dentro questo pacchetto è ammessa soltanto outputs/ stessa."""
    if os.path.lexists(out) and inside(out, record):
        raise Refusal(f"the output directory is the record directory or inside it: {out}")
    parent = os.path.dirname(out)
    try:
        pst = os.stat(parent)
    except FileNotFoundError:
        raise Refusal(f"the parent of the output directory does not exist: {parent}")
    if not stat.S_ISDIR(pst.st_mode):
        raise Refusal(f"the parent of the output directory is not a directory: {parent}")
    if inside(parent, record):
        raise Refusal(f"the output directory is inside the record directory: {out}")
    for rr in run_roots:
        if inside(parent, rr) or (os.path.lexists(out) and inside(out, rr)):
            raise Refusal(f"the output directory is inside a directory of the runs: {out}")
    root = results_package_root(record)
    if root is not None and inside(parent, root):
        raise Refusal(f"the output directory is inside the package of the results that holds the record: {out}")
    if inside(parent, str(HERE)) and not (same_file(parent, str(HERE)) and os.path.basename(out) == "outputs"):
        raise Refusal(f"the output directory is inside this package; only {DEFAULT_AGAINST} itself is allowed: {out}")
    if os.path.lexists(out):
        raise Refusal(f"the output directory already exists (it must not exist): {out}")


SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
RACE_ERRNOS = (errno.EEXIST, errno.ENOTEMPTY, errno.ENOTDIR, errno.EISDIR)


def _identity(path):
    """(dispositivo, inode) di path senza seguire collegamenti, o None se non esiste."""
    try:
        st = os.lstat(path)
    except OSError:
        return None
    return st.st_dev, st.st_ino


def install_recorders(state):
    """Gestori di SIGINT, SIGTERM e SIGHUP che registrano soltanto il segnale: non sollevano mai eccezioni, quindi non
    possono interrompere la scrittura in un punto qualsiasi (con matplotlib il processo ha più thread, ma il gestore
    Python gira sempre nel thread principale)."""
    def record(signum, frame):
        state["received"].append(signum)
    return {s: signal.signal(s, record) for s in SIGNALS}


def restore_handlers(old):
    for s, h in old.items():
        signal.signal(s, h)


def _poll(state):
    """Punto di controllo: un segnale registrato interrompe la scrittura qui, prima del rename."""
    if state["received"]:
        raise Interrupted(f"signal {state['received'][0]}")


def committed(state, out):
    """Le uscite sono consegnate se e solo se il nome di uscita è la cartella temporanea di questa esecuzione (stesso
    dispositivo e inode: il rename li conserva). Lo stato si legge sul disco, non da dove è nata un'eccezione."""
    return state.get("tmp_id") is not None and _identity(out) == state["tmp_id"]


def cleanup(state, out):
    """Toglie solo ciò che questa esecuzione ha creato, riconosciuto per identità: la cartella temporanea e la riserva
    vuota; mai un oggetto di un'altra esecuzione con lo stesso nome."""
    tmp = state.get("tmp")
    if tmp is not None and state.get("tmp_id") is not None and _identity(tmp) == state["tmp_id"]:
        shutil.rmtree(tmp, ignore_errors=True)
    if state.get("reserved_id") is not None and _identity(out) == state["reserved_id"]:
        try:
            os.rmdir(out)
        except OSError:
            pass


def write_outputs(out, files, state):
    """Cartella temporanea nascosta accanto all'uscita (stesso genitore: il rename non attraversa file system), file
    creati in modo esclusivo e sincronizzati; il nome di uscita è riservato con un mkdir esclusivo, la riserva è
    riconosciuta per inode subito prima del rename e la cartella temporanea vi è rinominata sopra: il rename è l'unico
    punto di consegna. I segnali sono solo registrati (install_recorders) e letti nei punti di controllo: dopo ogni
    file, prima della riserva, prima del rename. Se non c'è consegna, la pulizia toglie ciò che questa esecuzione ha
    creato, per identità."""
    parent = os.path.dirname(out)
    try:
        tmp = tempfile.mkdtemp(prefix=".mtf-", dir=parent)
        state["tmp"], state["tmp_id"] = tmp, _identity(tmp)
        for name in OUTPUTS:
            # O_CREAT | O_EXCL non segue un collegamento simbolico (POSIX): un nome già presente fa fallire l'apertura
            fd = os.open(os.path.join(tmp, name), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(fd, "wb") as fh:
                fh.write(files[name])
                fh.flush()
                os.fsync(fh.fileno())
            _poll(state)
        os.chmod(tmp, 0o755)
        _poll(state)
        try:
            os.mkdir(out, 0o755)
        except OSError as e:
            if e.errno in RACE_ERRNOS:
                raise OutputRace(out)
            raise
        # i gestori non sollevano eccezioni: nulla interrompe il codice fra il mkdir e la lettura dell'identità
        state["reserved_id"] = _identity(out)
        _poll(state)
        if _identity(out) != state["reserved_id"] or os.listdir(out):
            raise OutputRace(out)
        try:
            os.rename(tmp, out)
        except OSError as e:
            if e.errno in RACE_ERRNOS:
                raise OutputRace(out)
            raise
    finally:
        if not committed(state, out):
            cleanup(state, out)


def finish(out, state, failure, extra=None):
    """Messaggio ed esito decisi dallo stato sul disco: consegna avvenuta oppure no."""
    sigs = list(state["received"]) + ([extra] if extra is not None else [])
    if committed(state, out):
        note = f"; signal {sigs[0]} received, handled after the outputs were complete" if sigs else ""
        print(f"written {len(OUTPUTS)} files to {out}{note}")
        print("RESULT: PASS")
        return 0
    cleanup(state, out)
    if failure is None and sigs:
        failure = f"interrupted by signal {sigs[0]}; nothing was left in {out}"
    elif failure is None:
        # nessun segnale e nessun errore, ma il nome di uscita non è la cartella di questa esecuzione: un altro processo
        # ha cambiato la cartella temporanea o il nome di uscita durante il rename
        failure = (f"the output name or the temporary directory changed during the rename; nothing was delivered by "
                   f"this run: {out}")
    print(f"ERROR: {failure}", file=sys.stderr)
    print("RESULT: FAIL")
    return 1


def check_against(against, files, text_only):
    names = sorted(os.listdir(against))
    if names != sorted(OUTPUTS):
        raise Refusal(f"the directory to check holds {names}, expected exactly {sorted(OUTPUTS)}")
    bad = []
    for name in OUTPUTS:
        delivered = read_regular(os.path.join(against, name), f"delivered output {name}")
        if name in IMAGE_OUTPUTS and text_only:
            ok = png_structure_ok(delivered) if name.endswith(".png") else pdf_structure_ok(delivered)
            print(f"{name}: {'structure valid' if ok else 'INVALID STRUCTURE'}, content not compared (--text-only)")
            if not ok:
                bad.append(name)
            continue
        same_bytes = delivered == files[name]
        print(f"{name}: {'identical' if same_bytes else 'DIFFERENT'}")
        if not same_bytes:
            bad.append(name)
    return bad


OPTIONS = ("--out", "--check", "--record", "--anet-runs", "--c2-runs", "--against", "--text-only")


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="make_table_figure.py", allow_abbrev=False,
        description="Regenerate the table of the five networks, the ACER variations and contrasts, the figure, the "
                    "per-seed appendix, the one-class counts, the descriptive analyses and the aggregated CSV of the "
                    f"C5 comparison on the NUAA test from the public files of the evaluation {RUN_ID} and of its "
                    "eight runs.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--out", help="output directory to create (it must not exist)")
    g.add_argument("--check", action="store_true", help="regenerate in memory and compare with --against")
    p.add_argument("--record", default=str(DEFAULT_RECORD),
                   help="record directory of the evaluation (holding status.json and public/)")
    p.add_argument("--anet-runs", default=str(DEFAULT_ANET_RUNS),
                   help="directory holding the records of the six AttackNet runs")
    p.add_argument("--c2-runs", default=str(DEFAULT_C2_RUNS), help="directory holding the records of the two C2 runs")
    p.add_argument("--against", default=None, help="directory of the delivered outputs (with --check)")
    p.add_argument("--text-only", action="store_true",
                   help="with --check: compare the text outputs only; check only the structure of the images")
    for o in OPTIONS:
        n = sum(1 for t in argv if t == o or t.startswith(o + "="))
        if n > 1:
            p.error(f"option {o} given more than once")
    a = p.parse_args(argv)
    if not a.check and (a.against is not None or a.text_only):
        p.error("--against and --text-only go with --check")
    if a.check and a.against is None:
        a.against = str(DEFAULT_AGAINST)
    return a


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    record = os.path.abspath(args.record)
    anet_runs, c2_runs = os.path.abspath(args.anet_runs), os.path.abspath(args.c2_runs)
    out = state = None
    try:
        check_dir(record, "the record directory")
        check_dir(anet_runs, "the directory of the AttackNet runs")
        check_dir(c2_runs, "the directory of the C2 runs")
        if args.check:
            against = os.path.abspath(args.against)
            check_dir(against, "the directory to check")
            if inside(against, record):
                raise Refusal(f"the directory to check is the record directory or inside it: {against}")
            if inside(against, anet_runs) or inside(against, c2_runs):
                raise Refusal(f"the directory to check is inside a directory of the runs: {against}")
        else:
            out = os.path.abspath(args.out)
            check_out_path(out, record, (anet_runs, c2_runs))
        data = load_input_bytes(record, anet_runs, c2_runs)
        files = build(data, images=not (args.check and args.text_only))
        if args.check:
            bad = check_against(against, files, args.text_only)
            if bad:
                print(f"ERROR: {len(bad)} file(s) differ from the regenerated ones or are invalid: {', '.join(bad)}")
                print("RESULT: FAIL")
                return 1
            print("RESULT: PASS")
            return 0
        state = {"received": []}
        old = install_recorders(state)
        failure = None
        try:
            write_outputs(out, files, state)
        except OutputRace:
            failure = (f"the output name or the temporary directory was taken by another process after the checks; "
                       f"nothing was written by this run: {out}")
        except Interrupted:
            failure = None
        except OSError as e:
            failure = f"could not write the outputs ({e.strerror}); nothing was left in {out}"
        finally:
            restore_handlers(old)
        return finish(out, state, failure)
    except KeyboardInterrupt:
        # SIGINT fuori dalla fase di scrittura (i registratori non sono installati): esito deciso dallo stato
        if state is not None:
            return finish(out, state, None, extra=int(signal.SIGINT))
        print(f"ERROR: interrupted by signal {int(signal.SIGINT)}; nothing was written", file=sys.stderr)
        print("RESULT: FAIL")
        return 1
    except Refusal as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("RESULT: FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
