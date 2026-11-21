"""
Tabella primaria, appendice per seed e figura dei divari di ACER di LivenessNet e ResNet-18 sul test NUAA (patience 4,
soglia 0,5), rigenerate dai soli file pubblici della valutazione 20261121-ARCHEVAL-332fbdf (PR #32).

Dalla cartella pubblica della valutazione legge soltanto `SHA256SUMS`, `units.csv`, `results.json` e `conditions.csv`:
il `SHA256SUMS` deve avere l'impronta fissata qui (PUBLIC_SUMS_SHA256, uguale a `public_sha256sums_sha256` di
`status.json` della valutazione) e ogni file letto la sua riga di quel `SHA256SUMS`; `report.md` e `reproduction.csv`
non sono aperti né sottoposti a hash. Poi controlla run, stato, schema, condizioni primarie, soglia effettiva e
completezza delle unità, valida tutte le righe di `units.csv`, e ricalcola dalle frazioni esatte di `units.csv` le
medie, i conteggi e i valori per seed che `results.json` riporta: se un valore non torna si ferma. Gli intervalli per
soggetto non si possono ricalcolare dai file pubblici (servono i conteggi per soggetto): si copiano da `results.json`
dopo averne controllato la forma e la coerenza con le etichette.

Ogni frase generata che enuncia un dato (ricampionamenti validi, segni, modelli che decidono una sola classe, frame per
fold, valori che si arrotondano a zero) è costruita dai dati e il comando si ferma se i dati la contraddicono.

Tutte le uscite si calcolano in memoria; poi si scrivono, con fsync, in una cartella temporanea nascosta accanto alla
cartella di uscita (stesso genitore: il rename non attraversa file system); il nome di uscita è riservato con un mkdir
esclusivo, riconosciuto per inode subito prima del rename, e la cartella temporanea vi è rinominata sopra: il rename è
l'unico punto di consegna. SIGINT, SIGTERM e SIGHUP sono solo registrati e letti in punti di controllo; messaggio ed
esito si decidono dallo stato sul disco (il nome di uscita è o non è la cartella temporanea di questa esecuzione, per
inode), e la pulizia toglie solo ciò che questa esecuzione ha creato, per identità. Dopo SIGKILL può restare la cartella
temporanea nascosta o la riserva vuota, mai una cartella di uscita incompleta. `--check` rigenera in memoria e
confronta con i file consegnati, senza scrivere nulla.

Regola delle cartelle di uscita: l'uscita non esiste; non è dentro la cartella degli input né dentro il pacchetto dei
risultati che la contiene (la cartella di cui gli input sono `runs/<RUN_ID>/public`); dentro questo pacchetto è ammessa
soltanto `outputs/` stessa (il bersaglio documentato della rigenerazione, dopo averla tolta).

Arrotondamento (una sola funzione, `points`): proporzione × 100, due decimali, metà lontano dallo zero, applicato al
valore esatto (frazione di `units.csv`, oppure il decimale scritto in `results.json`). Tipografia: nelle tabelle
Markdown, nella figura e nelle frasi generate il segno meno è U+2212; nei CSV e nelle etichette citate da
`results.json` il trattino ASCII.

  python3 -I -B make_table_figure.py --out DIR [--inputs DIR]
  python3 -I -B make_table_figure.py --check [--against DIR] [--text-only] [--inputs DIR]
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

RUN_ID = "20261121-ARCHEVAL-332fbdf"
SCHEMA = "pad-arch-eval/1"
# SHA-256 del SHA256SUMS della cartella pubblica della valutazione (= public_sha256sums_sha256 di status.json)
PUBLIC_SUMS_SHA256 = "c8a5d339d73da3826e95620dad586015989461ffe04b1cd26bb8c9b2bb19ffeb"
SUMS_FILES = ("conditions.csv", "report.md", "reproduction.csv", "results.json", "units.csv")
READ_FILES = ("units.csv", "results.json", "conditions.csv")

HERE = Path(os.path.abspath(__file__)).parent
DEFAULT_INPUTS = HERE.parent / "pad-arch-eval-results" / "runs" / RUN_ID / "public"
DEFAULT_AGAINST = HERE / "outputs"

TEXT_OUTPUTS = ("primary_table.csv", "primary_table.md", "primary_table.tex", "per_seed.csv", "per_seed.md",
                "figure_acer_gaps_data.csv")
IMAGE_OUTPUTS = ("figure_acer_gaps.png", "figure_acer_gaps.pdf")
OUTPUTS = TEXT_OUTPUTS + IMAGE_OUTPUTS

PATIENCE = 4
THRESHOLD = "fixed_0.5"
PATIENCES = ("4", "12")
THRESHOLDS = ("fixed_0.5", "dev_eer")
STATES = ("ordinary", "re-estimated")
GAP_KEY = {"ordinary": "gap_ordinary", "re-estimated": "gap_re_estimated"}
NETWORKS = (("livenessnet", "LivenessNet"), ("resnet18", "ResNet-18"))
METRICS = ("apcer", "bpcer", "acer")
FOLDS = (1, 2, 3, 4)
SEEDS = (42, 43, 44)
N_SUBJECTS = 16
MINUS = "−"

UNITS_HEADER = ["patience", "bn_state", "threshold", "fold", "seed", "n_frames", "frame_ids_sha256",
                "livenessnet_threshold", "livenessnet_apcer", "livenessnet_bpcer", "livenessnet_acer",
                "resnet18_threshold", "resnet18_apcer", "resnet18_bpcer", "resnet18_acer", "delta_apcer",
                "delta_bpcer", "delta_acer", "delta_acer_value", "tie"]
CONDITIONS_HEADER = ["patience", "bn_state", "threshold", "label", "scope", "units", "n_subjects", "mean_delta_acer",
                     "low_acer", "high_acer", "valid_resamples_acer", "mean_delta_apcer", "low_apcer", "high_apcer",
                     "mean_delta_bpcer", "low_bpcer", "high_bpcer", "resnet18_lower", "livenessnet_lower", "ties",
                     "undefined", "sign", "sign_reason"]
OUT_HEADER = ["scope", "row", "network", "bn_state", "units", "apcer", "bpcer", "acer", "apcer_exact", "bpcer_exact",
              "acer_exact", "apcer_low", "apcer_high", "apcer_valid_resamples", "bpcer_low", "bpcer_high",
              "bpcer_valid_resamples", "acer_low", "acer_high", "acer_valid_resamples", "resnet18_lower",
              "livenessnet_lower", "ties", "units_changing_sign", "label", "label_reason", "interval_source"]

SIGN_BELOW = ("ResNet-18 lower", "interval entirely below zero")
SIGN_ABOVE = ("LivenessNet lower", "interval entirely above zero")
SIGN_ZERO = ("unresolved", "interval contains zero")
CONTRAST_BELOW = ("ACER difference (ResNet-18 - LivenessNet) lower with re-estimated BN",
                  "interval entirely below zero")
CONTRAST_ABOVE = ("ACER difference (ResNet-18 - LivenessNet) higher with re-estimated BN",
                  "interval entirely above zero")
CONTRAST_ZERO = ("unresolved", "interval contains zero")

FRACTION_RE = re.compile(r"-?(0|[1-9][0-9]*)/[1-9][0-9]*")
SUMS_LINE_RE = re.compile(r"([0-9a-f]{64})  ([A-Za-z0-9._-]+)")
HEX64_RE = re.compile(r"[0-9a-f]{64}")
INT_RE = re.compile(r"[1-9][0-9]*")

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


def load_input_bytes(inputs):
    """SHA256SUMS contro l'impronta fissata, poi ogni file letto contro la sua riga di SHA256SUMS."""
    check_dir(inputs, "the input directory")
    sums = read_regular(os.path.join(inputs, "SHA256SUMS"), "input SHA256SUMS")
    if sha256(sums) != PUBLIC_SUMS_SHA256:
        raise Refusal(f"SHA256SUMS of the input directory has SHA-256 {sha256(sums)}, expected the pinned "
                      f"{PUBLIC_SUMS_SHA256} (public_sha256sums_sha256 of status.json of {RUN_ID})")
    try:
        text = sums.decode("ascii")
    except UnicodeDecodeError:
        raise Refusal("SHA256SUMS of the input directory is not ASCII")
    if not text.endswith("\n"):
        raise Refusal("SHA256SUMS of the input directory does not end with a newline")
    listed = {}
    for line in text[:-1].split("\n"):
        m = SUMS_LINE_RE.fullmatch(line)
        if not m:
            raise Refusal(f"SHA256SUMS of the input directory: malformed line {line!r}")
        if m.group(2) in listed:
            raise Refusal(f"SHA256SUMS of the input directory lists {m.group(2)} twice")
        listed[m.group(2)] = m.group(1)
    if sorted(listed) != sorted(SUMS_FILES):
        raise Refusal(f"SHA256SUMS of the input directory lists {sorted(listed)}, expected {sorted(SUMS_FILES)}")
    out = {}
    for name in READ_FILES:
        b = read_regular(os.path.join(inputs, name), f"input {name}")
        if sha256(b) != listed[name]:
            raise Refusal(f"{name}: SHA-256 {sha256(b)} differs from its line in SHA256SUMS ({listed[name]})")
        out[name] = b
    return out


# ---------------------------------------------------------------------------------------------------------------------
# lettura rigorosa dei valori

def no_duplicates(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise Refusal(f"results.json: duplicate key {k!r}")
        d[k] = v
    return d


def reject_constant(c):
    raise Refusal(f"results.json: non-finite number {c}")


def parse_json(b):
    try:
        return json.loads(b.decode("utf-8"), object_pairs_hook=no_duplicates, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise Refusal(f"results.json is not valid JSON: {e}")


def same(a, b):
    """Uguaglianza con i tipi (True non è 1, 12.0 non è 12)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def get(d, path):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise Refusal(f"results.json: missing field {path}")
        cur = cur[k]
    return cur


def number(d, path):
    v = get(d, path)
    if type(v) is not float:
        raise Refusal(f"results.json: {path} is not a floating-point number")
    return v


def integer(d, path):
    v = get(d, path)
    if type(v) is not int:
        raise Refusal(f"results.json: {path} is not an integer")
    return v


def expect(d, path, value):
    if not same(get(d, path), value):
        raise Refusal(f"results.json: {path} is {get(d, path)!r}, expected {value!r}")


def fraction(text, where):
    if not FRACTION_RE.fullmatch(text):
        raise Refusal(f"{where}: {text!r} is not an exact fraction")
    f = Fraction(text)
    if text != f"{f.numerator}/{f.denominator}":
        raise Refusal(f"{where}: {text!r} is not written reduced")
    return f


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


def sign_rule(low, high):
    """PROTOCOL.md §6: un estremo a zero contiene lo zero."""
    if high < 0:
        return SIGN_BELOW
    if low > 0:
        return SIGN_ABOVE
    return SIGN_ZERO


def contrast_rule(low, high):
    if high < 0:
        return CONTRAST_BELOW
    if low > 0:
        return CONTRAST_ABOVE
    return CONTRAST_ZERO


def changes_sign(o, e):
    """Il divario di un'unità cambia segno: strettamente negativo in uno stato e strettamente positivo nell'altro."""
    return (o < 0 < e) or (o > 0 > e)


def sign_change_between_states(label_o, label_e):
    """PROTOCOL.md §6: cambio di segno solo fra due condizioni risolte di segno opposto."""
    resolved = (SIGN_BELOW[0], SIGN_ABOVE[0])
    return label_o in resolved and label_e in resolved and label_o != label_e


# ---------------------------------------------------------------------------------------------------------------------
# controlli e ricalcoli

def check_results_header(res):
    if not isinstance(res, dict):
        raise Refusal("results.json: the top level is not an object")
    expect(res, "schema", SCHEMA)
    expect(res, "run_id", RUN_ID)
    expect(res, "status", "PASS")
    expect(res, "official", True)
    expect(res, "primary_analysis.conditions",
           {"ordinary": [PATIENCE, "ordinary", THRESHOLD], "re-estimated": [PATIENCE, "re-estimated", THRESHOLD]})
    cb = get(res, "checks_before_primary_analysis")
    for state in STATES:
        key = f"[{PATIENCE}, '{state}', '{THRESHOLD}']"
        for part in ("plan", "units"):
            if not isinstance(cb.get(part), dict) or key not in cb[part]:
                raise Refusal(f"results.json: checks_before_primary_analysis.{part} has no entry {key}")
        if not same(cb["plan"][key], {"combinations": 12, "threshold": 0.5}):
            raise Refusal(f"results.json: checks_before_primary_analysis.plan {key} is not 12 combinations at "
                          f"threshold 0.5")
        if not same(cb["units"][key], {"units": 12, "combinations_per_network": 12,
                                       "effective_threshold": {"LivenessNet": [0.5], "ResNet-18": [0.5]}}):
            raise Refusal(f"results.json: checks_before_primary_analysis.units {key} is not 12 units with effective "
                          f"threshold 0.5 for both networks")
    frames = get(res, "test_frames_per_fold")
    if not (isinstance(frames, list) and len(frames) == len(FOLDS)
            and all(type(n) is int and n > 0 for n in frames)):
        raise Refusal("results.json: test_frames_per_fold is not 4 positive integers")
    ids = get(res, "test_frame_ids_sha256")
    if not (isinstance(ids, dict) and sorted(ids) == [str(f) for f in FOLDS]
            and all(isinstance(v, str) and HEX64_RE.fullmatch(v) for v in ids.values())):
        raise Refusal("results.json: test_frame_ids_sha256 is not one SHA-256 per fold")
    return frames, ids


def check_unit_row(r, frames, ids):
    """Una riga di units.csv (qualsiasi condizione): piano, soglie, frame, frazioni, ACER, divari, valore, pareggio."""
    where = f"units.csv ({r['patience']}, {r['bn_state']}, {r['threshold']}, fold {r['fold']}, seed {r['seed']})"
    if r["patience"] not in PATIENCES or r["bn_state"] not in STATES or r["threshold"] not in THRESHOLDS:
        raise Refusal(f"{where}: condition outside the plan")
    if r["fold"] not in [str(f) for f in FOLDS] or r["seed"] not in [str(s) for s in SEEDS]:
        raise Refusal(f"{where}: fold or seed outside the plan")
    fold = int(r["fold"])
    for net, _ in NETWORKS:
        t = r[f"{net}_threshold"]
        if r["threshold"] == THRESHOLD:
            if t != "0.5":
                raise Refusal(f"{where}: effective threshold of {net} is {t!r}, expected 0.5")
        else:
            try:
                v = float(t)
            except ValueError:
                v = math.nan
            if not (math.isfinite(v) and 0 <= v <= 1 and t == repr(v)):
                raise Refusal(f"{where}: threshold of {net} is not a number between 0 and 1")
    if not INT_RE.fullmatch(r["n_frames"]) or int(r["n_frames"]) != frames[fold - 1]:
        raise Refusal(f"{where}: n_frames {r['n_frames']!r} differs from test_frames_per_fold of results.json")
    if r["frame_ids_sha256"] != ids[str(fold)]:
        raise Refusal(f"{where}: frame_ids_sha256 differs from test_frame_ids_sha256 of results.json")
    u = {}
    for net, _ in NETWORKS:
        for m in METRICS:
            u[(net, m)] = fraction(r[f"{net}_{m}"], where)
            if not 0 <= u[(net, m)] <= 1:
                raise Refusal(f"{where}: {net}_{m} outside [0, 1]")
        if u[(net, "acer")] != (u[(net, "apcer")] + u[(net, "bpcer")]) / 2:
            raise Refusal(f"{where}: {net}_acer is not the mean of APCER and BPCER")
    for m in METRICS:
        u[("delta", m)] = fraction(r[f"delta_{m}"], where)
        if u[("delta", m)] != u[("resnet18", m)] - u[("livenessnet", m)]:
            raise Refusal(f"{where}: delta_{m} is not ResNet-18 minus LivenessNet")
    if r["delta_acer_value"] != repr(float(u[("delta", "acer")])):
        raise Refusal(f"{where}: delta_acer_value is not the value of delta_acer")
    if r["tie"] != ("1" if u[("delta", "acer")] == 0 else "0"):
        raise Refusal(f"{where}: tie does not match delta_acer")
    return u


def load_units(b, frames, ids):
    """Tutte le 96 righe di units.csv validate; restituisce le 24 righe primarie (12 unità × 2 stati BN)."""
    rows = read_csv(b, "units.csv", UNITS_HEADER)
    keys = [(r["patience"], r["bn_state"], r["threshold"], r["fold"], r["seed"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise Refusal("units.csv: a condition and unit appears twice")
    units = {s: {} for s in STATES}
    for r in rows:
        u = check_unit_row(r, frames, ids)
        if (r["patience"], r["threshold"]) == (str(PATIENCE), THRESHOLD):
            units[r["bn_state"]][(int(r["fold"]), int(r["seed"]))] = u
    want = sorted((f, s) for f in FOLDS for s in SEEDS)
    for state in STATES:
        if sorted(units[state]) != want:
            raise Refusal(f"units.csv: the primary condition with {state} BN does not hold the 12 fold-seed units "
                          f"exactly once ({len(units[state])} found)")
    n_all = len(PATIENCES) * len(STATES) * len(THRESHOLDS) * len(want)
    if len(rows) != n_all:
        raise Refusal(f"units.csv: {len(rows)} rows, expected {n_all} (2 patiences × 2 BN states × 2 thresholds × "
                      f"12 units)")
    return units


def check_result_rows(res, units):
    """primary_analysis.rows di results.json contro units.csv, unità per unità e campo per campo."""
    rows = get(res, "primary_analysis.rows")
    if not isinstance(rows, list) or len(rows) != 12:
        raise Refusal("results.json: primary_analysis.rows does not hold 12 units")
    seen = set()
    for i, row in enumerate(rows):
        unit = row.get("unit") if isinstance(row, dict) else None
        if not (isinstance(unit, list) and len(unit) == 3 and unit[0] == "NUAA" and type(unit[1]) is int
                and type(unit[2]) is int and (unit[1], unit[2]) in units["ordinary"]):
            raise Refusal(f"results.json: primary_analysis.rows[{i}].unit is not a unit of the plan")
        key = (unit[1], unit[2])
        if key in seen:
            raise Refusal(f"results.json: primary_analysis.rows holds fold {key[0]} seed {key[1]} twice")
        seen.add(key)
        o, e = units["ordinary"][key], units["re-estimated"][key]
        for block in ("delta_ordinary", "delta_re_estimated", "contrast"):
            for m in METRICS:
                want = {"delta_ordinary": o[("delta", m)], "delta_re_estimated": e[("delta", m)],
                        "contrast": e[("delta", m)] - o[("delta", m)]}[block]
                exact = get(row, f"{block}.{m}.exact")
                value = get(row, f"{block}.{m}.value")
                if exact != f"{want.numerator}/{want.denominator}":
                    raise Refusal(f"results.json: primary_analysis.rows fold {key[0]} seed {key[1]} {block}.{m}.exact "
                                  f"differs from units.csv")
                if type(value) is not float or value != float(want):
                    raise Refusal(f"results.json: primary_analysis.rows fold {key[0]} seed {key[1]} {block}.{m}.value "
                                  f"differs from units.csv")


def counts(vals):
    return {"resnet18_lower": sum(v < 0 for v in vals), "livenessnet_lower": sum(v > 0 for v in vals),
            "ties": sum(v == 0 for v in vals)}


def mean(vals):
    vals = list(vals)
    return sum(vals, Fraction(0)) / len(vals)


def scope_keys(scope):
    if scope == "all seeds":
        return [(f, s) for f in FOLDS for s in SEEDS]
    seed = int(scope.split()[1])
    return [(f, seed) for f in FOLDS]


def interval(block, sub, path, mean_value):
    """Intervallo copiato: forma, ricampionamenti validi e media dentro l'intervallo (controllo di plausibilità; i valori
    non si possono ricalcolare dai file pubblici)."""
    low, high = number(block, f"{sub}.low"), number(block, f"{sub}.high")
    valid = integer(block, f"{sub}.valid_resamples")
    if not low <= high:
        raise Refusal(f"results.json: {path}.{sub} has low > high")
    if not low <= mean_value <= high:
        raise Refusal(f"results.json: {path}.{sub} does not contain its mean")
    if not 1 <= valid <= 1000:
        raise Refusal(f"results.json: {path}.{sub} does not have 1 to 1,000 valid resamples")
    return low, high, valid


def check_gap_block(block, path, keys, units, state):
    """Un divario di results.json contro le frazioni di units.csv: medie, conteggi, segno, intervalli ben formati."""
    out = {"mean": {}, "interval": {}}
    for m in METRICS:
        exact = mean(units[state][k][("delta", m)] for k in keys)
        got = number(block, f"mean_delta.{m}")
        if got != float(exact):
            raise Refusal(f"results.json: {path}.mean_delta.{m} = {got!r}, units.csv gives {float(exact)!r}")
        out["mean"][m] = (got, exact)
        out["interval"][m] = interval(block, f"interval.{m}", path, got)
    c = counts([units[state][k][("delta", "acer")] for k in keys])
    c.update(undefined=0, units=len(keys))
    if not same(get(block, "acer_counts"), c):
        raise Refusal(f"results.json: {path}.acer_counts = {get(block, 'acer_counts')!r}, units.csv gives {c!r}")
    out["counts"] = c
    low, high, _ = out["interval"]["acer"]
    if (get(block, "sign"), get(block, "sign_reason")) != sign_rule(low, high):
        raise Refusal(f"results.json: {path}.sign does not follow the sign rule on its ACER interval")
    out["label"] = sign_rule(low, high)
    if not same(get(block, "n_subjects"), N_SUBJECTS):
        raise Refusal(f"results.json: {path}.n_subjects is not {N_SUBJECTS}")
    return out


def check_contrast_block(block, path, keys, units):
    out = {"mean": {}, "interval": {}}
    for m in METRICS:
        exact = mean(units["re-estimated"][k][("delta", m)] - units["ordinary"][k][("delta", m)] for k in keys)
        got = number(block, f"mean.{m}")
        if got != float(exact):
            raise Refusal(f"results.json: {path}.mean.{m} = {got!r}, units.csv gives {float(exact)!r}")
        out["mean"][m] = (got, exact)
        out["interval"][m] = interval(block, f"interval.{m}", path, got)
    changing = sum(changes_sign(units["ordinary"][k][("delta", "acer")], units["re-estimated"][k][("delta", "acer")])
                   for k in keys)
    if not same(get(block, "units"), len(keys)):
        raise Refusal(f"results.json: {path}.units is not {len(keys)}")
    if not same(get(block, "units_changing_sign"), changing):
        raise Refusal(f"results.json: {path}.units_changing_sign is not {changing} (units.csv)")
    out["changing"] = changing
    low, high, _ = out["interval"]["acer"]
    if (get(block, "reading"), get(block, "reason")) != contrast_rule(low, high):
        raise Refusal(f"results.json: {path}.reading does not follow the reading rule on its ACER interval")
    out["label"] = contrast_rule(low, high)
    return out


def check_conditions(b, res):
    """Le 8 righe primarie di conditions.csv (2 stati × tutte le unità e 3 seed) contro results.json, campo per campo."""
    rows = read_csv(b, "conditions.csv", CONDITIONS_HEADER)
    prim = [r for r in rows if (r["patience"], r["threshold"]) == (str(PATIENCE), THRESHOLD)
            and r["label"] == "primary"]
    want = sorted((s, sc) for s in STATES for sc in ["all seeds"] + [f"seed {x}" for x in SEEDS])
    if sorted((r["bn_state"], r["scope"]) for r in prim) != want:
        raise Refusal("conditions.csv: the primary rows are not the 2 BN states × (all seeds, seed 42, 43, 44)")
    for r in prim:
        key = GAP_KEY[r["bn_state"]]
        path = f"primary_analysis.{key}" if r["scope"] == "all seeds" else \
            f"primary_analysis.per_seed.{r['scope'].split()[1]}.{key}"
        blk = get(res, path)
        exp = {"units": str(get(blk, "acer_counts.units")), "n_subjects": str(get(blk, "n_subjects")),
               "valid_resamples_acer": str(get(blk, "interval.acer.valid_resamples")),
               "resnet18_lower": str(get(blk, "acer_counts.resnet18_lower")),
               "livenessnet_lower": str(get(blk, "acer_counts.livenessnet_lower")),
               "ties": str(get(blk, "acer_counts.ties")), "undefined": str(get(blk, "acer_counts.undefined")),
               "sign": get(blk, "sign"), "sign_reason": get(blk, "sign_reason")}
        for m in METRICS:
            exp[f"mean_delta_{m}"] = repr(get(blk, f"mean_delta.{m}"))
            exp[f"low_{m}"] = repr(get(blk, f"interval.{m}.low"))
            exp[f"high_{m}"] = repr(get(blk, f"interval.{m}.high"))
        for k, v in exp.items():
            if r[k] != v:
                raise Refusal(f"conditions.csv ({r['bn_state']}, {r['scope']}): {k} differs from results.json {path}")


def check_valid_resamples(scopes):
    """Le frasi generate danno un solo numero di ricampionamenti validi per metrica: deve valere in ogni blocco e seed."""
    valid = {}
    for m in METRICS:
        seen = {s[part]["interval"][m][2] for s in scopes.values() for part in ("ordinary", "re-estimated", "contrast")}
        if len(seen) != 1:
            raise Refusal(f"results.json: the valid resamples of {m.upper()} differ between the blocks "
                          f"({sorted(seen)}); the generated texts state one number per metric")
        valid[m] = seen.pop()
    return valid


def check_one_class_dev(res):
    """Il conteggio preregistrato «one-class» (sul dev, soglia 0,5) per patience 4, da one_class di results.json."""
    oc = get(res, "one_class")
    if not isinstance(oc, list):
        raise Refusal("results.json: one_class is not a list")
    out = {}
    for e in oc:
        if not isinstance(e, dict):
            raise Refusal("results.json: one_class holds an entry that is not an object")
        if e.get("patience") == PATIENCE:
            key = (e.get("network"), e.get("bn_state"))
            if key in out or key[0] not in [n for _, n in NETWORKS] or key[1] not in STATES:
                raise Refusal("results.json: one_class entries for patience 4 are not one per network and BN state")
            k, n = e.get("one_class"), e.get("evaluable")
            if type(k) is not int or type(n) is not int or not 0 <= k <= n:
                raise Refusal("results.json: one_class has a count that is not k of n evaluable")
            out[key] = (k, n)
    if len(out) != len(NETWORKS) * len(STATES):
        raise Refusal("results.json: one_class entries for patience 4 are not one per network and BN state")
    return out


def analyse(data):
    """Tutti i controlli e i valori delle uscite; nessuna scrittura."""
    res = parse_json(data["results.json"])
    frames, ids = check_results_header(res)
    units = load_units(data["units.csv"], frames, ids)
    check_result_rows(res, units)
    check_conditions(data["conditions.csv"], res)
    scopes = {}
    for scope in ["all seeds"] + [f"seed {s}" for s in SEEDS]:
        keys = scope_keys(scope)
        base = "primary_analysis" if scope == "all seeds" else f"primary_analysis.per_seed.{scope.split()[1]}"
        s = {"keys": keys, "networks": {}}
        for state in STATES:
            for net, _ in NETWORKS:
                s["networks"][(net, state)] = {m: mean(units[state][k][(net, m)] for k in keys) for m in METRICS}
            s[state] = check_gap_block(get(res, f"{base}.{GAP_KEY[state]}"), f"{base}.{GAP_KEY[state]}", keys,
                                       units, state)
        s["contrast"] = check_contrast_block(get(res, f"{base}.contrast"), f"{base}.contrast", keys, units)
        scopes[scope] = s
    per_seed = get(res, "primary_analysis.per_seed")
    if not isinstance(per_seed, dict) or sorted(per_seed) != [str(s) for s in SEEDS]:
        raise Refusal("results.json: primary_analysis.per_seed does not hold exactly seeds 42, 43 and 44")
    valid = check_valid_resamples(scopes)
    single = []
    for state in STATES:
        for net, label in NETWORKS:
            for (f, s) in sorted(units[state]):
                pair = (units[state][(f, s)][(net, "apcer")], units[state][(f, s)][(net, "bpcer")])
                if pair == (0, 1):
                    single.append((state, label, f, s, "attack", units[state][(f, s)][(net, "acer")]))
                elif pair == (1, 0):
                    single.append((state, label, f, s, "bona fide", units[state][(f, s)][(net, "acer")]))
    return {"units": units, "scopes": scopes, "frames": frames, "single": single, "valid": valid,
            "one_class_dev": check_one_class_dev(res)}


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
    """Il segno è quello del valore esatto, anche quando l'arrotondamento dà 0.00. minus: '-' nei CSV, U+2212 altrove."""
    s = f"{abs(points(x)):.2f}"
    if x < 0:
        return minus + s
    return ("+" + s) if signed and x > 0 else s


def fmt_u(x, signed=False):
    return fmt(x, signed, MINUS)


def fmt_interval(iv):
    low, high, _ = iv
    return f"[{fmt_u(low, True)}, {fmt_u(high, True)}]"


def exact_text(f):
    return f"{f.numerator}/{f.denominator}"


def table_rows(scope, s):
    rows = []
    for state in STATES:
        for net, label in NETWORKS:
            v = s["networks"][(net, state)]
            rows.append({"scope": scope, "row": "network", "network": label, "bn_state": state,
                         "units": str(len(s["keys"])),
                         **{m: repr(float(v[m])) for m in METRICS},
                         **{f"{m}_exact": exact_text(v[m]) for m in METRICS}})
    src = "results.json primary_analysis" + ("" if scope == "all seeds" else f".per_seed.{scope.split()[1]}")
    for name, blk, state in (("gap_ordinary", s["ordinary"], "ordinary"),
                             ("gap_re_estimated", s["re-estimated"], "re-estimated"),
                             ("contrast", s["contrast"], "re-estimated minus ordinary")):
        r = {"scope": scope, "row": name, "network": "ResNet-18 minus LivenessNet", "bn_state": state,
             "units": str(len(s["keys"]))}
        for m in METRICS:
            r[m] = repr(blk["mean"][m][0])
            r[f"{m}_exact"] = exact_text(blk["mean"][m][1])
            low, high, valid = blk["interval"][m]
            r[f"{m}_low"], r[f"{m}_high"], r[f"{m}_valid_resamples"] = repr(low), repr(high), str(valid)
        if name != "contrast":
            r.update({k: str(blk["counts"][k]) for k in ("resnet18_lower", "livenessnet_lower", "ties")})
        else:
            r["units_changing_sign"] = str(blk["changing"])
        r["label"], r["label_reason"] = blk["label"]
        r["interval_source"] = f"{src}.{name}.interval (subject bootstrap; not recomputable from public files)"
        rows.append(r)
    return rows


def csv_bytes(rows, header):
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=header, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in header})
    return buf.getvalue().encode("utf-8")


def md_rates(scopes_list, a):
    multi = len(scopes_list) > 1
    lines = ["| " + ("seed | " if multi else "") + "network | BN state | APCER (%) | BPCER (%) | ACER (%) |",
             "|" + ("---|" if multi else "") + "---|---|---:|---:|---:|"]
    for scope in scopes_list:
        s = a["scopes"][scope]
        for state in STATES:
            for net, label in NETWORKS:
                v = s["networks"][(net, state)]
                lead = f"{scope.split()[1]} | " if multi else ""
                lines.append(f"| {lead}{label} | {state} | {fmt_u(v['apcer'])} | {fmt_u(v['bpcer'])} | "
                             f"{fmt_u(v['acer'])} |")
    return lines


def md_gaps(scopes_list, a, label_header):
    multi = len(scopes_list) > 1
    lines = ["| " + ("seed | " if multi else "") + "quantity | APCER (points) | BPCER (points) | ACER (points) | "
             f"ACER units | {label_header} |",
             "|" + ("---|" if multi else "") + "---|---|---|---|---|---|"]
    for scope in scopes_list:
        s = a["scopes"][scope]
        lead = f"{scope.split()[1]} | " if multi else ""
        for name, blk in (("gap, ordinary BN", s["ordinary"]), ("gap, re-estimated BN", s["re-estimated"]),
                          ("contrast, re-estimated minus ordinary", s["contrast"])):
            cells = [f"{fmt_u(blk['mean'][m][0], True)} {fmt_interval(blk['interval'][m])}" for m in METRICS]
            if "counts" in blk:
                c = blk["counts"]
                u = f"{c['resnet18_lower']} / {c['livenessnet_lower']} / {c['ties']}"
            else:
                u = f"gap changes sign in {blk['changing']} of {len(s['keys'])}"
            lines.append(f"| {lead}{name} | {' | '.join(cells)} | {u} | \"{blk['label'][0]}\" ({blk['label'][1]}) |")
    return lines


def wrap(text, first):
    """Frase generata spezzata a 117 colonne, come il resto delle note (rientro di 3 spazi dopo la prima riga)."""
    rest = "   " if first else ""
    return textwrap.wrap(text, width=117, initial_indent=first, subsequent_indent=rest, break_long_words=False,
                         break_on_hyphens=False)


def valid_text(v):
    return f"{v['acer']:,} for ACER, {v['apcer']:,} for APCER, {v['bpcer']:,} for BPCER"


def single_class_text(a):
    """Nota sui modelli che, sul test a soglia 0,5, decidono una sola classe per tutti i frame (dai dati)."""
    sc = a["single"]
    if not sc:
        first = "No model of the two primary conditions decides every test frame as the same class at threshold 0.5."
    else:
        parts = [f"{label} fold {f} seed {s} with {state} BN, every frame {cls}" for state, label, f, s, cls, _ in sc]
        acers = sorted({exact_text(x) for *_, x in sc})
        first = (f"{len(sc)} models decide every test frame as the same class at threshold 0.5 ({'; '.join(parts)}; "
                 f"ACER {', '.join(acers)} in `units.csv`); they enter the means like the others: nothing was "
                 f"excluded after the test was seen.")
    dev = a["one_class_dev"]
    dparts = [f"{label} with {state} BN {dev[(label, state)][0]} of {dev[(label, state)][1]}"
              for _, label in NETWORKS for state in STATES]
    return (f"{first} The pre-registered count of \"one-class\" runs is a different measure, on the dev at threshold "
            f"0.5 (`PROTOCOL.md` §6; `one_class` of `results.json`, patience 4): {'; '.join(dparts)}.")


def sign_change_text(a):
    s = a["scopes"]["all seeds"]
    lo, le = s["ordinary"]["label"][0], s["re-estimated"]["label"][0]
    verdict = "there is one" if sign_change_between_states(lo, le) else "there is none"
    return (f"A change of sign needs two resolved conditions of opposite sign: with labels \"{lo}\" (ordinary BN) and "
            f"\"{le}\" (re-estimated BN), {verdict} between the two BN states. The units whose gap changes sign "
            f"({s['contrast']['changing']} of {len(s['keys'])}) are reported apart, as the same definition asks.")


def near_zero_text(a, scopes_list):
    """Valori non nulli che si arrotondano a 0.00: il segno resta; si dice quale valore è e quanto vale."""
    hits = []
    for scope in scopes_list:
        s = a["scopes"][scope]
        for name, blk in (("gap with ordinary BN", s["ordinary"]), ("gap with re-estimated BN", s["re-estimated"]),
                          ("contrast", s["contrast"])):
            for m in METRICS:
                for part, x in (("mean", blk["mean"][m][0]), ("lower bound", blk["interval"][m][0]),
                                ("upper bound", blk["interval"][m][1])):
                    if x != 0 and points(x) == 0:
                        v = Decimal(repr(x)) * 100
                        where = "" if scope == "all seeds" else f" of {scope}"
                        hits.append(f"{part} of the {m.upper()} {name}{where}: "
                                    f"{(MINUS if v < 0 else '+')}{abs(v).quantize(Decimal('0.0001'), ROUND_HALF_UP)}"
                                    f" points")
    if not hits:
        return None
    return ("A value written " + MINUS + "0.00 or +0.00 is non-zero and rounds to zero; its sign is kept ("
            + "; ".join(hits) + ").")


def primary_md(a):
    s = a["scopes"]["all seeds"]
    fr = a["frames"]
    nz = near_zero_text(a, ["all seeds"])
    out = [
        "# Table 1. LivenessNet and ResNet-18 on the NUAA test: patience 4, threshold 0.5, two BN states",
        "",
        f"Generated by `make_table_figure.py` from the public files of the evaluation `{RUN_ID}` (PR #32):",
        "`units.csv` and `results.json`. Values in percentage points: the proportions (0 to 1) of the source files",
        "multiplied by 100 and rounded to two decimals, half away from zero, from the exact value. The full-precision",
        "proportions are in `primary_table.csv`.",
        "",
        f"**(a)** Each network: mean over the {len(s['keys'])} fold-seed units (4 folds × seeds 42, 43, 44).",
        "",
        *md_rates(["all seeds"], a),
        "",
        "**(b)** Gap = metric(ResNet-18) − metric(LivenessNet), mean over the same units, and contrast = gap with",
        "re-estimated BN − gap with ordinary BN; mean [subject interval]. ACER units: ResNet-18 lower / LivenessNet",
        "lower / ties; for the contrast, the units whose gap changes sign. Labels quoted from `results.json`.",
        "",
        *md_gaps(["all seeds"], a, "label (pre-registered rules)"),
        "",
        "Notes.",
        "",
        "1. Patience 4, fixed threshold 0.5; effective threshold 0.5 for both networks in both BN states and 12 of 12",
        "   fold-seed units in each (`checks_before_primary_analysis` of `results.json`, checked by the script). BN",
        "   states: ordinary, and re-estimated on the fitting set only.",
        "2. The means of (a) are computed by the script from the exact fractions of `units.csv`. The mean gaps and the",
        "   contrast of (b) are those of `results.json`; the script recomputes them from `units.csv` and stops if they",
        "   differ. Each gap equals the difference of the means of (a), up to the rounding of the displayed values.",
        "3. Subject intervals: percentiles 2.5–97.5 of 1,000 resamples of the "
        f"{N_SUBJECTS} test subjects pooled over the",
        f"   4 folds; valid resamples {valid_text(a['valid'])}, in every block (checked). They are copied",
        "   from `results.json`: they cannot be recomputed from the public files, which hold no per-subject counts.",
        "   Individual 95% intervals, with no correction for multiplicity. No interval is published for the rates of",
        "   each network in (a).",
        "4. Labels of the gaps: the sign rule of `PROTOCOL.md` §6: \"ResNet-18 lower\" if the interval of the ACER gap",
        "   lies entirely below zero, \"LivenessNet lower\" if entirely above, \"unresolved\" if it contains zero (a",
        "   bound at zero contains it). Label of the contrast: the reading of the contrast, a choice of the author",
        "   written before the test (`PROTOCOL.md` §6): \"... higher with re-estimated BN\" if the interval of the ACER",
        "   contrast lies entirely above zero, \"... lower ...\" if entirely below, \"unresolved\" if it contains zero.",
        *wrap(sign_change_text(a), "   "),
        *wrap(single_class_text(a), "5. "),
        f"6. Test frames per fold (`n_frames` of `units.csv`): {', '.join(str(n) for n in fr)}; in each fold both",
        "   networks and both BN states are scored on the same frames (same `frame_ids_sha256`, checked). The",
        "   number of frames per class is not in the public files: the fractions of `units.csv` are written reduced",
        "   and are not denominators.",
        "7. The LivenessNet cells repeat cells already read on this test with the accepted checkpoints (PR #21,",
        "   PR #28); only the ResNet-18 side is new (`PROTOCOL.md` §5).",
    ]
    if nz:
        out += wrap(nz, "8. ")
    out.append("")
    return "\n".join(out).encode("utf-8")


def per_seed_md(a):
    seeds = [f"seed {s}" for s in SEEDS]
    nz = near_zero_text(a, seeds)
    out = [
        "# Appendix. Per seed: LivenessNet and ResNet-18 on the NUAA test, patience 4, threshold 0.5",
        "",
        "Reported apart; not part of the reading (`PROTOCOL.md` §6). Each seed has 4 units, one per fold. Its subject",
        f"intervals are those of `primary_analysis.per_seed` of `results.json`, over the same {N_SUBJECTS} test subjects",
        f"(`n_subjects` of each block, checked), with valid resamples {valid_text(a['valid'])}",
        "(checked in every block); they are not recomputable from the public files. Values in percentage points, as",
        "in Table 1; full precision in `per_seed.csv`.",
        "",
        "**(a)** Each network: mean over the 4 units of the seed.",
        "",
        *md_rates(seeds, a),
        "",
        "**(b)** Gaps (ResNet-18 − LivenessNet) and contrast (re-estimated − ordinary), mean over the 4 units of the",
        "seed [subject interval]. Labels quoted from `results.json`: the rules of `PROTOCOL.md` §6 applied to each",
        "seed, which do not enter the reading.",
        "",
        *md_gaps(seeds, a, "label (rule of `PROTOCOL.md` §6 applied to the seed; not part of the reading)"),
        "",
    ]
    if nz:
        out += wrap(nz, "") + [""]
    return "\n".join(out).encode("utf-8")


def tex_num(x, signed=False):
    s = fmt(x, signed)
    return f"${s}$" if s.startswith("-") or s.startswith("+") else s


def primary_tex(a):
    s = a["scopes"]["all seeds"]
    out = ["% Table 1, generated by make_table_figure.py from the public files of " + RUN_ID + " (PR #32).",
           "% Percentage points, two decimals, half away from zero; unit counts, labels and notes are in",
           "% primary_table.md.",
           "\\begin{tabular}{llrrr}", "\\hline",
           "Network & BN state & APCER (\\%) & BPCER (\\%) & ACER (\\%) \\\\", "\\hline"]
    for state in STATES:
        for net, label in NETWORKS:
            v = s["networks"][(net, state)]
            out.append(f"{label} & {state} & {tex_num(v['apcer'])} & {tex_num(v['bpcer'])} & {tex_num(v['acer'])} "
                       f"\\\\")
    out += ["\\hline", "\\end{tabular}", "", "\\begin{tabular}{llll}", "\\hline",
            "Quantity & APCER (points) & BPCER (points) & ACER (points) \\\\", "\\hline"]
    for name, blk in (("Gap, ordinary BN", s["ordinary"]), ("Gap, re-estimated BN", s["re-estimated"]),
                      ("Contrast", s["contrast"])):
        cells = []
        for m in METRICS:
            low, high, _ = blk["interval"][m]
            cells.append(f"{tex_num(blk['mean'][m][0], True)} [{tex_num(low, True)}, {tex_num(high, True)}]")
        out.append(f"{name} & {' & '.join(cells)} \\\\")
    out += ["\\hline", "\\end{tabular}", ""]
    return "\n".join(out).encode("utf-8")


FIG_ITEMS = (("gap_ordinary", "ACER gap, ordinary BN", "ResNet-18 − LivenessNet; negative: ResNet-18 lower ACER"),
             ("gap_re_estimated", "ACER gap, re-estimated BN",
              "ResNet-18 − LivenessNet; negative: ResNet-18 lower ACER"),
             ("contrast", "Contrast, re-estimated − ordinary BN", "positive: gap higher with re-estimated BN"))
FIG_HEADER = ["item", "kind", "fold", "seed", "value", "value_points", "valid_resamples", "level"]
# regola della disposizione verticale dei punti delle unità (deterministica, dichiarata nel README)
LEVEL_GAP_POINTS = 2.0   # due punti più vicini di 2,0 punti percentuali sull'asse vanno su livelli diversi
LEVEL_BASE, LEVEL_STEP = 0.17, 0.09


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


def figure_rows(a):
    s, units = a["scopes"]["all seeds"], a["units"]
    rows = []
    for item, _, _ in FIG_ITEMS:
        blk = {"gap_ordinary": s["ordinary"], "gap_re_estimated": s["re-estimated"], "contrast": s["contrast"]}[item]
        mean_v = blk["mean"]["acer"][0]
        low, high, valid = blk["interval"]["acer"]
        for kind, v in (("mean", mean_v), ("low", low), ("high", high)):
            rows.append({"item": item, "kind": kind, "value": repr(v), "value_points": fmt(v, True),
                         "valid_resamples": str(valid) if kind != "mean" else ""})
        vals = []
        for k in s["keys"]:
            o, e = units["ordinary"][k][("delta", "acer")], units["re-estimated"][k][("delta", "acer")]
            vals.append(({"gap_ordinary": o, "gap_re_estimated": e, "contrast": e - o}[item], k))
        levels = unit_levels([(points(v), k) for v, k in vals])
        for (v, k), lvl in zip(vals, levels):
            rows.append({"item": item, "kind": "unit", "fold": str(k[0]), "seed": str(k[1]), "value": repr(float(v)),
                         "value_points": fmt(v, True), "level": str(lvl)})
    return rows


def figure_images(fig_rows):
    """PNG e PDF della figura dai soli valori del file dei dati; nessun metadato di software o di data.
    Larghezza prevista: una colonna, 3,5 pollici (89 mm), al 100%."""
    try:
        import matplotlib
    except ImportError:
        raise Refusal("matplotlib is not available: the figure cannot be generated (with --check, --text-only "
                      "compares the tables and the figure data without it)")
    matplotlib.use("Agg", force=True)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    with matplotlib.rc_context():
        matplotlib.rcdefaults()
        matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6,
                                    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.labelsize": 7,
                                    "pdf.fonttype": 42, "axes.unicode_minus": True})
        fig = Figure(figsize=(3.5, 3.6), dpi=300)
        FigureCanvasAgg(fig)
        ax = fig.add_axes((0.04, 0.19, 0.92, 0.79))
        ypos = {item: 2 - i for i, (item, _, _) in enumerate(FIG_ITEMS)}
        allv = [float(r["value"]) * 100 for r in fig_rows]
        lim = 10 * (int(max(abs(v) for v in allv) // 10) + 1)
        for y in (0.9, 1.9):
            ax.axhline(y, color="0.8", lw=0.5, zorder=0)
        first = True
        for item, title, hint in FIG_ITEMS:
            rr = [r for r in fig_rows if r["item"] == item]
            v = {r["kind"]: float(r["value"]) * 100 for r in rr if r["kind"] != "unit"}
            y = ypos[item]
            ax.plot([0, 0], [y - 0.18, y + 0.48], color="0.3", lw=0.7, ls=(0, (4, 3)), zorder=1,
                    label="zero" if first else None)
            ux = [float(r["value"]) * 100 for r in rr if r["kind"] == "unit"]
            uy = [y + LEVEL_BASE + LEVEL_STEP * int(r["level"]) for r in rr if r["kind"] == "unit"]
            ax.plot(ux, uy, ls="none", marker="o", ms=2.6, mfc="0.55", mec="0.2", mew=0.35, zorder=2,
                    label="fold-seed units (12)" if first else None)
            ax.plot([v["low"], v["high"]], [y, y], color="black", lw=1.2, solid_capstyle="butt", zorder=3)
            for end in ("low", "high"):
                ax.plot([v[end], v[end]], [y - 0.06, y + 0.06], color="black", lw=0.9, zorder=3)
            ax.plot([v["mean"]], [y], ls="none", marker="o", ms=4, color="black", zorder=4,
                    label="mean and subject interval (95%)" if first else None)
            lab = {r["kind"]: r["value_points"].replace("-", MINUS) for r in rr if r["kind"] != "unit"}
            ax.text(0.0, y + 0.73, f"{title}: {lab['mean']} [{lab['low']}, {lab['high']}]",
                    transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7)
            ax.text(0.0, y + 0.57, hint, transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=6.5, color="0.15")
            first = False
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.25, 2.9)
        ax.set_yticks([])
        ax.set_xlabel("ACER, percentage points", fontsize=7)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, frameon=False, fontsize=6.5,
                  handletextpad=0.3, columnspacing=0.7, handlelength=1.6)
        png, pdf = io.BytesIO(), io.BytesIO()
        fig.savefig(png, format="png", dpi=300, metadata={"Software": None})
        fig.savefig(pdf, format="pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None})
    return png.getvalue(), pdf.getvalue()


def build(data, images=True):
    """Tutte le uscite in memoria: {nome: byte}. Con images False, i soli file di testo e dei dati."""
    try:
        a = analyse(data)
    except (AttributeError, KeyError, TypeError, ValueError, IndexError) as e:
        raise Refusal(f"unexpected structure of the input files ({type(e).__name__}: {e})")
    rows = table_rows("all seeds", a["scopes"]["all seeds"])
    seed_rows = [r for s in SEEDS for r in table_rows(f"seed {s}", a["scopes"][f"seed {s}"])]
    fig_rows = figure_rows(a)
    out = {"primary_table.csv": csv_bytes(rows, OUT_HEADER), "primary_table.md": primary_md(a),
           "primary_table.tex": primary_tex(a), "per_seed.csv": csv_bytes(seed_rows, OUT_HEADER),
           "per_seed.md": per_seed_md(a), "figure_acer_gaps_data.csv": csv_bytes(fig_rows, FIG_HEADER)}
    if images:
        out["figure_acer_gaps.png"], out["figure_acer_gaps.pdf"] = figure_images(fig_rows)
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


def results_package_root(inputs):
    """La cartella del pacchetto dei risultati quando gli input sono <root>/runs/<RUN_ID>/public, altrimenti None."""
    p = Path(os.path.realpath(inputs))
    if p.name == "public" and p.parent.name == RUN_ID and p.parent.parent.name == "runs":
        return str(p.parent.parent.parent)
    return None


def check_out_path(out, inputs):
    """Regola delle cartelle di uscita (docstring del modulo)."""
    for name in READ_FILES + ("SHA256SUMS",):
        if os.path.lexists(out) and same_file(out, os.path.join(inputs, name)):
            raise Refusal(f"the output directory is an input file: {out}")
    if os.path.lexists(out) and same_file(out, inputs):
        raise Refusal(f"the output directory is the input directory: {out}")
    parent = os.path.dirname(out)
    try:
        pst = os.stat(parent)
    except FileNotFoundError:
        raise Refusal(f"the parent of the output directory does not exist: {parent}")
    if not stat.S_ISDIR(pst.st_mode):
        raise Refusal(f"the parent of the output directory is not a directory: {parent}")
    if inside(parent, inputs):
        raise Refusal(f"the output directory is inside the input directory: {out}")
    root = results_package_root(inputs)
    if root is not None and inside(parent, root):
        raise Refusal(f"the output directory is inside the package of the results that holds the inputs: {out}")
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
        committed = read_regular(os.path.join(against, name), f"committed output {name}")
        if name in IMAGE_OUTPUTS and text_only:
            ok = png_structure_ok(committed) if name.endswith(".png") else pdf_structure_ok(committed)
            print(f"{name}: {'structure valid' if ok else 'INVALID STRUCTURE'}, content not compared (--text-only)")
            if not ok:
                bad.append(name)
            continue
        same_bytes = committed == files[name]
        print(f"{name}: {'identical' if same_bytes else 'DIFFERENT'}")
        if not same_bytes:
            bad.append(name)
    return bad


OPTIONS = ("--out", "--check", "--inputs", "--against", "--text-only")


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="make_table_figure.py", allow_abbrev=False,
        description="Regenerate the primary table, the per-seed appendix and the figure of the ACER gaps of "
                    "LivenessNet and ResNet-18 on the NUAA test (patience 4, threshold 0.5) from the public files of "
                    f"the evaluation {RUN_ID} (PR #32).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--out", help="output directory to create (it must not exist)")
    g.add_argument("--check", action="store_true", help="regenerate in memory and compare with --against")
    p.add_argument("--inputs", default=str(DEFAULT_INPUTS), help="public directory of the evaluation")
    p.add_argument("--against", default=None, help="directory of the committed outputs (with --check)")
    p.add_argument("--text-only", action="store_true",
                   help="with --check: compare tables and figure data only; check only the structure of the images")
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
    inputs = os.path.abspath(args.inputs)
    out = state = None
    try:
        check_dir(inputs, "the input directory")
        if args.check:
            against = os.path.abspath(args.against)
            check_dir(against, "the directory to check")
            if same_file(against, inputs):
                raise Refusal(f"the directory to check is the input directory: {against}")
        else:
            out = os.path.abspath(args.out)
            check_out_path(out, inputs)
        data = load_input_bytes(inputs)
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
