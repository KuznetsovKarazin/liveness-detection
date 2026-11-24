"""
Test di make_table_figure.py (materiali per l'articolo della scheda C5) su un record SINTETICO costruito qui a mano:
conteggi di errori per fold, seed, rete e condizione scelti da questo file, e tutto ciò che il valutatore scrive
(units.csv, table.csv, conditions.csv, results.json, status.json, SHA256SUMS) ricalcolato qui in modo indipendente
dallo script (frazioni, medie, conteggi, righe per unità, letture, classe unica, graduatorie); gli intervalli, che i
file pubblici non permettono di ricalcolare, sono fissati qui con regole dichiarate. Il mondo contiene pareggi (una
variazione nulla, due reti a pari merito), un modello che decide tutto attacco e uno tutto bona fide, un modello non
valutabile (a patience 12, o a patience 4 nelle varianti con celle non calcolate), un valore che si arrotonda a zero.

Cosa provano:
  - i numeri e i testi delle uscite contro i valori calcolati qui (anche a mano per alcune celle) e la regola di
    arrotondamento (metà lontano dallo zero sul valore esatto, non sul float);
  - una fixture avversaria per ogni controllo, con SHA256SUMS e impronta riallineati (scatta la regola, non il
    checksum), e i casi d'arresto della lettura (file mancanti, collegamenti, file speciali, impronte);
  - restricted/ mai aperto né elencato;
  - i percorsi di uscita, la scrittura atomica, le gare, i segnali (processo vero, con e senza matplotlib), --check;
  - l'indipendenza dallo stato della macchina: locale, fuso orario, ora, HOME, cartella di lavoro, PYTHONHASHSEED,
    ordine dei file, matplotlibrc locale; nessun warning; nessun percorso o identificativo nelle uscite;
  - se il record reale è presente (--record, oppure il percorso predefinito dello script), le uscite contro table.csv
    e report.md del valutatore e contro outputs/ consegnato (altrimenti saltati, con il motivo dichiarato).
Salti ammessi: matplotlib assente (test delle immagini) e record reale assente (test del record reale).
TMPDIR deve stare fuori da questo pacchetto (controllato prima di creare qualsiasi cosa).

  python3 -I -B check_make_table_figure.py [--record DIR] [--verbose] [-f]
"""
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import logging
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
import warnings
from decimal import ROUND_HALF_UP, Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path
from unittest import mock

HERE = Path(os.path.abspath(__file__)).parent
SCRIPT = HERE / "make_table_figure.py"


def _inside(p, root):
    p, root = os.path.realpath(p), os.path.realpath(root)
    return p == root or p.startswith(root + os.sep)


_TMP = tempfile.gettempdir()
if _inside(_TMP, HERE):
    print("ERROR: TMPDIR is inside this package; set it elsewhere")
    print("RESULT: FAIL")
    sys.exit(1)

_ARGS = sys.argv[1:]
SUITE_OPTIONS = ("--record", "--anet-runs", "--c2-runs")


def _arg(name):
    return _ARGS[_ARGS.index(name) + 1] if name in _ARGS else None

# matplotlib: cartella di configurazione propria della suite (cache dei font), così l'esito non dipende da HOME né da
# una cache esistente; i suoi messaggi di log (la costruzione della cache) sono raccolti, non stampati
SUITE_TMP = Path(tempfile.mkdtemp(prefix="mtf-suite-"))
os.environ["MPLCONFIGDIR"] = str(SUITE_TMP / "mplconfig")
MPL_LOG = []


class _Collect(logging.Handler):
    def emit(self, record):
        MPL_LOG.append(record.getMessage())


_mpl_logger = logging.getLogger("matplotlib")
_mpl_logger.addHandler(_Collect())
_mpl_logger.propagate = False

if importlib.util.find_spec("matplotlib") is not None:
    import matplotlib.font_manager  # noqa: E402,F401  (cache dei font costruita una volta, prima dei sottoprocessi)

_spec = importlib.util.spec_from_file_location("make_table_figure", SCRIPT)
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

HAS_MPL = importlib.util.find_spec("matplotlib") is not None
REAL = Path(os.path.abspath(_arg("--record"))) if _arg("--record") else M.DEFAULT_RECORD
REAL_ANET = Path(os.path.abspath(_arg("--anet-runs"))) if _arg("--anet-runs") else M.DEFAULT_ANET_RUNS
REAL_C2 = Path(os.path.abspath(_arg("--c2-runs"))) if _arg("--c2-runs") else M.DEFAULT_C2_RUNS
REAL_ARGS = ["--record", str(REAL), "--anet-runs", str(REAL_ANET), "--c2-runs", str(REAL_C2)]
REAL_RUN_DIRS = [REAL_ANET / rid for rid in M.ANET_RUNS.values()] + [REAL_C2 / rid for rid, _ in M.C2_RUNS.values()]
HAS_REAL = (REAL / "status.json").is_file() and (REAL / "public" / "SHA256SUMS").is_file() \
    and all((d / "status.json").is_file() and (d / "public" / "SHA256SUMS").is_file() for d in REAL_RUN_DIRS)
DELIVERED = HERE / "outputs"
MISSING_INPUT = {}
if not HAS_MPL:
    MISSING_INPUT["matplotlib"] = "matplotlib is not installed: the tests that need it (figure images, real build in " \
                                  "the signal tests, machine state) are skipped"
if not HAS_REAL:
    MISSING_INPUT["record"] = "the records of the evaluation and of its runs are not present: the tests against the " \
                              "real records are skipped"
need_mpl = unittest.skipUnless(HAS_MPL, "missing input: " + MISSING_INPUT.get("matplotlib", ""))
need_real = unittest.skipUnless(HAS_REAL, "missing input: " + MISSING_INPUT.get("record", ""))
U = "−"
NETS = ("LivenessNet", "AttackNetV1", "AttackNetV2_1", "AttackNetV2_2", "ResNet-18")
ANETS = NETS[1:4]
STATES = ("ordinary", "re-estimated")
CONDS = [(p, st, t) for p in (4, 12) for st in STATES for t in ("fixed_0.5", "dev_eer")]
UNITS = [(f, s) for f in (1, 2, 3, 4) for s in (42, 43, 44)]
MET = ("apcer", "bpcer", "acer")


def snapshot(*roots):
    """File (dimensione, data, hash) e cartelle, esclusa test_logs/."""
    out = {}
    for root in roots:
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in ("test_logs", "restricted")]   # restricted/ mai aperto
            out[dp] = ("dir",)
            for f in fns:
                p = os.path.join(dp, f)
                st = os.lstat(p)
                if not os.path.isfile(p) or os.path.islink(p):
                    out[p] = ("special", st.st_mode)
                    continue
                out[p] = (st.st_size, st.st_mtime_ns, hashlib.sha256(Path(p).read_bytes()).hexdigest())
    return out


REAL_ROOTS = [REAL] + REAL_RUN_DIRS if HAS_REAL else []
BEFORE = snapshot(HERE, *REAL_ROOTS)


def differing(a, b):
    """I nomi dei file diversi fra due insiemi di uscite (un confronto per nome: un assertEqual sui byte farebbe
    calcolare a unittest il diff di file grandi)."""
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def fs(f):
    return f"{f.numerator}/{f.denominator}"


def pp(x, signed=False, minus=U):
    """Arrotondamento scritto qui: frazione (o decimale scritto) × 100, due decimali, metà lontano dallo zero."""
    with localcontext() as ctx:
        ctx.prec = 60
        d = (Decimal(x.numerator) / Decimal(x.denominator)) if isinstance(x, F) else Decimal(repr(x))
        q = (abs(d) * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    s = f"{q:.2f}"
    if x < 0:
        return minus + s
    return "+" + s if signed and x > 0 else s


def run(argv, pin=None, c2=None):
    """main() nello stesso processo, con l'impronta fissata e le impronte dei run di C2 sostituite da quelle del mondo
    sintetico."""
    out, err = io.StringIO(), io.StringIO()
    ctx = mock.patch.object(M, "PUBLIC_SUMS_SHA256", pin) if pin else contextlib.nullcontext()
    ctx2 = mock.patch.object(M, "C2_RUNS", c2) if c2 else contextlib.nullcontext()
    with ctx, ctx2, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = M.main(argv)
    return code, out.getvalue(), err.getvalue()


def child_env(**extra):
    """Ambiente minimo e dichiarato per i sottoprocessi: nessuna variabile della shell dell'utente."""
    env = {"PATH": "/usr/bin:/bin", "MPLCONFIGDIR": os.environ["MPLCONFIGDIR"], "LC_ALL": "C.UTF-8",
           "HOME": str(SUITE_TMP)}
    env.update(extra)
    return env


def cli(*args, **kw):
    kw.setdefault("env", child_env())
    return subprocess.run([sys.executable, "-I", "-B", str(SCRIPT), *args], capture_output=True, text=True, **kw)


RUN_WITH_PIN = r"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("m", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.PUBLIC_SUMS_SHA256 = sys.argv[2]
m.C2_RUNS = {int(k): tuple(v) for k, v in json.loads(sys.argv[3]).items()}
sys.exit(m.main(sys.argv[4:]))
"""


def cli_pin(w, *args, **kw):
    kw.setdefault("env", child_env())
    return subprocess.run([sys.executable, "-I", "-B", "-c", RUN_WITH_PIN, str(SCRIPT), w.pin, json.dumps(w.c2), *args],
                          capture_output=True, text=True, **kw)


# ---------------------------------------------------------------------------------------------------------------------
# il mondo sintetico

FOLDS = {1: (10, 8, 4), 2: (12, 6, 4), 3: (9, 9, 4), 4: (11, 5, 4)}         # attacchi, bona fide, soggetti per fold
VALID = {"acer": 900, "apcer": 950, "bpcer": 940}
INTERVALS = ("individual 95% intervals: no correction for multiplicity, no simultaneous coverage over the contrasts, "
             "variations or comparisons")
PRESENTATION = ("percentage points = proportion x 100, rounded to two decimals half away from zero from the exact "
                "value, at presentation only; the computations use exact fractions")
# scarto medio degli errori per rete e stato: AttackNetV1 peggiora con la ristima (contrasto > 0), AttackNetV2_1
# migliora (contrasto < 0), AttackNetV2_2 un poco peggio di LivenessNet (contrasto > 0, intervallo che contiene lo zero)
BIAS = {("LivenessNet", "ordinary"): 0.0, ("LivenessNet", "re-estimated"): 0.0,
        ("AttackNetV1", "ordinary"): 0.0, ("AttackNetV1", "re-estimated"): 0.3,
        ("AttackNetV2_1", "ordinary"): 0.3, ("AttackNetV2_1", "re-estimated"): 0.0,
        ("AttackNetV2_2", "ordinary"): 0.0, ("AttackNetV2_2", "re-estimated"): 0.1,
        ("ResNet-18", "ordinary"): -0.05, ("ResNet-18", "re-estimated"): 0.0}
ONE_ATTACK = ((4, "ordinary", "fixed_0.5"), (1, 42), "LivenessNet")        # tutto attacco
ONE_BONA = ((12, "re-estimated", "fixed_0.5"), (2, 43), "ResNet-18")         # tutto bona fide
ZERO_VARIATION = ((3, 43), "ResNet-18")                                       # variazione nulla a patience 4, soglia 0,5
RANK_TIE = (12, "ordinary", "dev_eer")                                        # AttackNetV2_2 uguale a AttackNetV1
NEAR_ZERO_LOW = -0.00004                                                      # estremo che si arrotonda a −0.00
# modelli a classe unica sul dev (nei models.csv sintetici dei run): rete, patience, stato, unità -> classe
DEV_ONE = {("LivenessNet", 4, "ordinary", (1, 42)): "attack", ("AttackNetV1", 12, "re-estimated", (2, 43)): "bona fide"}
RUN_MODEL_COLUMNS = ["run", "position", "network", "fold", "seed", "patience", "model_status", "ordinary_one_class_05",
                     "re_estimated_one_class_05"]


def reading_of(k, lo, hi):
    names = (f"ACER variation of {k} minus that of LivenessNet lower than zero",
             f"ACER variation of {k} minus that of LivenessNet higher than zero", "unresolved")
    if hi < 0:
        return names[0], "interval entirely below zero"
    if lo > 0:
        return names[1], "interval entirely above zero"
    return names[2], "interval contains zero"


def conclusion_of(k, rd):
    if rd.endswith("lower than zero") or rd.endswith("higher than zero"):
        side = "lower" if rd.endswith("lower than zero") else "higher"
        return (f"On these checkpoints and test frames, the ACER variation with this BN re-estimation (re-estimated "
                f"minus ordinary BN) is algebraically {side} for {k} than for LivenessNet; it does not identify a cause.")
    return (f"This comparison does not show that the ACER variation with this BN re-estimation differs between {k} "
            f"and LivenessNet; it does not exclude it.")


class World:
    """Un record sintetico completo in tmp/results/runs/<RUN_ID>: status.json, public/ (i sei file e SHA256SUMS) e
    restricted/ illeggibile; gli otto record dei run in tmp/anet/<id> e tmp/c2/<id> (status.json, public/ con
    models.csv, one_class.csv, report.md, summary.json e SHA256SUMS, restricted/ illeggibile), coerenti con units.csv;
    le impronte dei run delle AttackNet in results.json e status.json della valutazione, quelle di C2 in self.c2 (al posto
    di C2_RUNS). ne: il modello non valutabile (rete, patience, fold, seed) o None; bound_zero: l'estremo inferiore
    dell'intervallo di ACER del contrasto di AttackNetV1 esattamente a zero."""

    def __init__(self, ne=("AttackNetV2_1", 12, 3, 44), bound_zero=False):
        self.tmp = Path(tempfile.mkdtemp(prefix="mtf-"))
        self.root = self.tmp / "results"
        self.record = self.root / "runs" / M.RUN_ID
        self.public = self.record / "public"
        self.public.mkdir(parents=True)
        self.ne = ne
        self.bound_zero = bound_zero
        self.counts, self.thr = self.make_counts()
        self.run_pins, self.c2 = {}, {}
        for key in self.run_dirs():
            self.write_run(key)
        self.pin = self.write()

    def close(self):
        for r in [self.record / "restricted"] + [d / "restricted" for d in self.run_dirs().values()]:
            if r.exists():
                os.chmod(r, 0o755)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def args(self):
        return ["--record", str(self.record), "--anet-runs", str(self.tmp / "anet"), "--c2-runs", str(self.tmp / "c2")]

    # --- i record dei run -------------------------------------------------------------------------------------------
    def run_dirs(self):
        out = {k: self.tmp / "anet" / rid for k, rid in M.ANET_RUNS.items()}
        out.update({("C2", p): self.tmp / "c2" / rid for p, (rid, _) in M.C2_RUNS.items()})
        return out

    def run_nets(self, key):
        return ("LivenessNet", "ResNet-18") if key[0] == "C2" else (key[0],)

    def run_name(self, key):
        tag = {"AttackNetV1": "v1", "AttackNetV2_1": "v2_1", "AttackNetV2_2": "v2_2"}
        return f"nuaa_patience_{key[1]}" if key[0] == "C2" else f"anet_{tag[key[0]]}_p{key[1]}"

    def write_run(self, key):
        """models.csv e one_class.csv di un run, scritti qui: one_class.csv contato da models.csv."""
        d = self.run_dirs()[key]
        pub = d / "public"
        pub.mkdir(parents=True)
        p = key[1]
        rows, oc = [RUN_MODEL_COLUMNS], [M.ONE_CLASS_HEADER]
        for n in self.run_nets(key):
            for u in UNITS:
                ok = self.present(n, p, u)
                rows.append([self.run_name(key), str(len(rows)), n, str(u[0]), str(u[1]), str(p),
                             "complete" if ok else "not evaluable"]
                            + [(DEV_ONE.get((n, p, st, u), "no") if ok else "NA") for st in STATES])
            for st in STATES:
                ev = sum(self.present(n, p, u) for u in UNITS)
                att = sum(DEV_ONE.get((n, p, st, u)) == "attack" and self.present(n, p, u) for u in UNITS)
                bf = sum(DEV_ONE.get((n, p, st, u)) == "bona fide" and self.present(n, p, u) for u in UNITS)
                oc.append([n, str(p), st, str(att + bf), str(ev), str(12 - ev), str(bf), str(att),
                           f"{att + bf} of {ev} evaluable"])
        for name, data in (("models.csv", rows), ("one_class.csv", oc)):
            buf = io.StringIO(newline="")
            csv.writer(buf, lineterminator="\n").writerows(data)
            (pub / name).write_text(buf.getvalue())
        (pub / "report.md").write_text("# synthetic run\n")
        (pub / "summary.json").write_text("{}\n")
        st = {"run_id": d.name, "run": self.run_name(key), "status": "PASS", "passed": True, "official": True}
        if key[0] != "C2":
            st["network"] = key[0]
        (d / "status.json").write_text(json.dumps(st, indent=1))
        r = d / "restricted"
        r.mkdir()
        (r / "scores.csv").write_text("never read\n")
        os.chmod(r, 0)
        self.realign_run(key)

    def realign_run(self, key, sums=True):
        """SHA256SUMS del run e la sua impronta in status.json del run (con sums False soltanto l'impronta di
        status.json registrata)."""
        d = self.run_dirs()[key]
        pub = d / "public"
        if sums:
            lines = [f"{hashlib.sha256((pub / n).read_bytes()).hexdigest()}  {n}" for n in M.RUN_PUBLIC_FILES]
            b = ("\n".join(lines) + "\n").encode()
            (pub / "SHA256SUMS").write_bytes(b)
            st = json.loads((d / "status.json").read_text())
            st["public_sha256sums_sha256"] = hashlib.sha256(b).hexdigest()
            (d / "status.json").write_text(json.dumps(st, indent=1))
        h = hashlib.sha256((d / "status.json").read_bytes()).hexdigest()
        if key[0] == "C2":
            self.c2[key[1]] = (d.name, h)
        else:
            self.run_pins[self.run_name(key)] = {"run_id": d.name, "official": True, "status_sha256": h}

    def edit_run(self, key, fn, realign=True, sums=True):
        """Modifica avversaria di un run (fn riceve la cartella del run); poi SHA256SUMS (se sums), impronte e record
        della valutazione riallineati (con realign False solo la modifica)."""
        fn(self.run_dirs()[key])
        if not realign:
            return
        self.realign_run(key, sums)
        if key[0] != "C2":
            self.edit_json(lambda d: d["runs"].update(self.run_pins))
            self.edit_status(lambda d: d["runs"].update(self.run_pins))

    def path(self, name):
        return self.public / name

    # --- conteggi d'errore scelti qui ------------------------------------------------------------------------------
    def present(self, n, p, u):
        return not (self.ne and (n, p, u[0], u[1]) == self.ne)

    def make_counts(self):
        counts, thr = {}, {}
        for c in CONDS:
            for u in UNITS:
                a, b, _ = FOLDS[u[0]]
                for n in NETS:
                    if not self.present(n, c[0], u):
                        continue
                    rng = random.Random(f"{c}|{u}|{n}")
                    ra = min(0.95, max(0.0, rng.uniform(0.1, 0.4) + BIAS[(n, c[1])]))
                    rb = min(0.95, max(0.0, rng.uniform(0.1, 0.4) + BIAS[(n, c[1])]))
                    counts[(c, u, n)] = (round(ra * a), round(rb * b))
                    thr[(c, u, n)] = "0.5" if c[2] == "fixed_0.5" else repr(round(rng.uniform(0.05, 0.95), 6))
        c, u, n = ONE_ATTACK
        counts[(c, u, n)] = (0, FOLDS[u[0]][1])
        c, u, n = ONE_BONA
        counts[(c, u, n)] = (FOLDS[u[0]][0], 0)
        u, n = ZERO_VARIATION
        counts[((4, "re-estimated", "fixed_0.5"), u, n)] = counts[((4, "ordinary", "fixed_0.5"), u, n)]
        for u in UNITS:
            if (RANK_TIE, u, "AttackNetV1") in counts and (RANK_TIE, u, "AttackNetV2_2") in counts:
                counts[(RANK_TIE, u, "AttackNetV2_2")] = counts[(RANK_TIE, u, "AttackNetV1")]
        # le soglie EER di un modello sono le stesse nelle due condizioni EER della sua patience e del suo stato
        return counts, thr

    def rates(self, c, u, n):
        a, b, _ = FOLDS[u[0]]
        ea, eb = self.counts[(c, u, n)]
        ap, bp = F(ea, a), F(eb, b)
        return {"apcer": ap, "bpcer": bp, "acer": (ap + bp) / 2}

    def computed(self, n, p):
        return all(self.present(n, p, u) for u in UNITS)

    def unit_q(self, p, t, u, quantity, n, st=None):
        """Valore per unità di una quantità, con la definizione scritta qui."""
        o, r = (p, "ordinary", t), (p, "re-estimated", t)

        def R(net, c):
            return self.rates(c, u, net)
        if quantity == "variation":
            return {m: R(n, r)[m] - R(n, o)[m] for m in MET}
        if quantity == "contrast":
            return {m: (R(n, r)[m] - R(n, o)[m]) - (R("LivenessNet", r)[m] - R("LivenessNet", o)[m]) for m in MET}
        if quantity == "gap":
            c = (p, st, t)
            return {m: R(n, c)[m] - R("ResNet-18", c)[m] for m in MET}
        return {m: (R(n, r)[m] - R(n, o)[m]) - (R("ResNet-18", r)[m] - R("ResNet-18", o)[m]) for m in MET}

    def interval(self, key, mu, m):
        """Intervallo fissato qui: attorno alla media, con larghezze ricavate dalla chiave (contiene sempre la media)."""
        rng = random.Random(f"interval|{key}|{m}")
        lo, hi = mu - rng.uniform(0.01, 0.08), mu + rng.uniform(0.01, 0.08)
        return {"low": lo, "high": hi, "valid_resamples": VALID[m]}

    def block(self, p, t, keys, quantity, n, st, scope, readings):
        vals = [self.unit_q(p, t, u, quantity, n, st) for u in keys]
        mean = {m: sum((v[m] for v in vals), F(0)) / len(vals) for m in MET}
        b = {"mean": {m: float(mean[m]) for m in MET},
             "interval": {m: self.interval((p, t, quantity, n, st, scope), float(mean[m]), m) for m in MET}}
        acer = [v["acer"] for v in vals]
        b["counts"] = {"below_zero": sum(x < 0 for x in acer), "above_zero": sum(x > 0 for x in acer),
                       "zero": sum(x == 0 for x in acer), "undefined": 0, "units": len(keys)}
        if quantity == "contrast" and (p, t) == (4, "fixed_0.5") and scope == "all":
            iv, mu = b["interval"]["acer"], b["mean"]["acer"]
            if n == "AttackNetV1":
                assert mu > 0
                iv["low"], iv["high"] = (0.0 if self.bound_zero else mu / 2), mu + 0.05
            elif n == "AttackNetV2_1":
                assert mu < 0
                iv["low"], iv["high"] = mu - 0.05, mu / 2
            else:
                assert mu > 0
                iv["low"], iv["high"] = NEAR_ZERO_LOW, mu + 0.03
        if quantity in ("variation", "contrast"):
            b["computed"] = True
        if quantity == "contrast" and readings:
            rd, reason = reading_of(n, b["interval"]["acer"]["low"], b["interval"]["acer"]["high"])
            b.update(reading=rd, reason=reason, conclusion=conclusion_of(n, rd))
        return b, mean

    def analysis(self, p, t, readings, reasons):
        guardian = {}
        for n in NETS:
            pairs = sum(self.present(n, p, u) for u in UNITS)
            ok = pairs == 12
            reason = None if ok else reasons[(n, p)]
            guardian[n] = {"pairs": pairs, "computed": ok, "reason": reason,
                           "text": "computed" if ok else f"not computed ({pairs} of 12 pairs evaluable in both BN "
                                                         f"states): {reason}"}

        def scope_blocks(keys, scope):
            out = {"variations": {}, "contrasts": {}, "descriptive_vs_resnet18": {}}
            for n in NETS:
                if guardian[n]["computed"]:
                    out["variations"][n] = self.block(p, t, keys, "variation", n, None, scope, readings)[0]
                else:
                    out["variations"][n] = {"computed": False, "text": guardian[n]["text"]}
            for k in ANETS:
                if guardian[k]["computed"]:
                    out["contrasts"][k] = self.block(p, t, keys, "contrast", k, None, scope, readings)[0]
                else:
                    out["contrasts"][k] = {"computed": False, "text": guardian[k]["text"], "reading": "not computed",
                                           "conclusion": f"Not computed for {k} ({guardian[k]['pairs']} of 12 pairs "
                                                         f"evaluable in both BN states): {guardian[k]['reason']}."}
            for n in NETS[:4]:
                if not guardian[n]["computed"]:
                    out["descriptive_vs_resnet18"][n] = {"computed": False, "text": guardian[n]["text"]}
                    continue
                d = {"computed": True}
                for st in STATES:
                    d[f"gap_{st.replace('-', '_')}"] = self.block(p, t, keys, "gap", n, st, scope, False)[0]
                d["variation_difference"] = self.block(p, t, keys, "vdiff", n, None, scope, False)[0]
                out["descriptive_vs_resnet18"][n] = d
            out["n_subjects"] = 16
            return out
        res = {"conditions": {"ordinary": [p, "ordinary", t], "re-estimated": [p, "re-estimated", t]},
               "intervals": INTERVALS, "presentation": PRESENTATION, "guardian": guardian}
        res.update(scope_blocks(UNITS, "all"))
        res["per_seed"] = {str(s): scope_blocks([u for u in UNITS if u[1] == s], f"seed {s}") for s in (42, 43, 44)}
        den = {"units": 12, "subjects": 16,
               "test_frames_summed_over_units": sum(FOLDS[f][0] + FOLDS[f][1] for f, _ in UNITS),
               "attack_frames_summed_over_units": sum(FOLDS[f][0] for f, _ in UNITS),
               "bona_fide_frames_summed_over_units": sum(FOLDS[f][1] for f, _ in UNITS),
               "per_fold": {str(f): {"frames": a + b, "attack": a, "bona_fide": b, "subjects": s}
                            for f, (a, b, s) in FOLDS.items()}}
        res["denominators"] = {"ordinary": den, "re-estimated": json.loads(json.dumps(den))}
        rows = []
        for u in UNITS:
            row = {"unit": ["NUAA", u[0], u[1]], "n_frames": FOLDS[u[0]][0] + FOLDS[u[0]][1],
                   "frame_ids_sha256": self.ids()[str(u[0])]}
            for n in NETS:
                if guardian[n]["computed"]:
                    row[f"variation/{n}"] = self.jrow(self.unit_q(p, t, u, "variation", n))
            for k in ANETS:
                if guardian[k]["computed"]:
                    row[f"contrast/{k}"] = self.jrow(self.unit_q(p, t, u, "contrast", k))
            for n in ANETS + ("LivenessNet",):
                if guardian[n]["computed"]:
                    for st in STATES:
                        row[f"gap/{n}/{st}"] = self.jrow(self.unit_q(p, t, u, "gap", n, st))
                    row[f"variation_difference/{n}"] = self.jrow(self.unit_q(p, t, u, "vdiff", n))
            rows.append(row)
        res["rows"] = rows
        return res

    @staticmethod
    def jrow(v):
        return {m: {"exact": fs(v[m]), "value": float(v[m])} for m in MET}

    @staticmethod
    def ids():
        return {str(f): hashlib.sha256(f"synthetic fold {f}".encode()).hexdigest() for f in FOLDS}

    def ranking(self, c):
        full = [n for n in NETS if all((c, u, n) in self.counts for u in UNITS)]
        means = {n: sum((self.rates(c, u, n)["acer"] for u in UNITS), F(0)) / 12 for n in full}
        rank = {n: 1 + sum(means[m] < means[n] for m in full) for n in full}
        counts = {n: {str(i): 0 for i in range(1, len(full) + 1)} for n in full}
        for u in UNITS:
            acer = {n: self.rates(c, u, n)["acer"] for n in full}
            for n in full:
                counts[n][str(1 + sum(acer[m] < acer[n] for m in full))] += 1
        return {"order": [{"network": n, "rank": rank[n], "mean_acer": float(means[n])}
                          for n in sorted(full, key=lambda n: (rank[n], NETS.index(n)))],
                "not_ranked": [n for n in NETS if n not in full], "unit_rank_counts": counts, "units_not_counted": 0,
                "units": 12}

    def not_evaluable(self):
        if not self.ne:
            return []
        n, p, f, s = self.ne
        tag = {"AttackNetV1": "v1", "AttackNetV2_1": "v2_1", "AttackNetV2_2": "v2_2"}[n]
        return [{"run": f"anet_{tag}_p{p}", "name": f"{n}_fold{f}_seed{s}", "network": n, "fold": f, "seed": s,
                 "patience": p, "reason": "not evaluable: no finite val_loss"}]

    def results(self):
        ne = self.not_evaluable()
        reasons = {(e["network"], e["patience"]): f"{e['run']}/{e['name']}: {e['reason']}" for e in ne}
        pa = self.analysis(4, "fixed_0.5", True, reasons)
        pa["definition"] = "variation of a network = ..."
        pa["rankings"] = {st: self.ranking((4, st, "fixed_0.5")) for st in STATES}
        desc = {"pairs": {f"patience_{p}_{t}": self.analysis(p, t, False, reasons)
                          for p, t in ((12, "fixed_0.5"), (4, "dev_eer"), (12, "dev_eer"))},
                "rankings": {f"{c[0]}/{c[1]}/{c[2]}": self.ranking(c) for c in CONDS},
                "intervals": INTERVALS, "note": "descriptive: no reading"}
        lvd = pa["descriptive_vs_resnet18"]["LivenessNet"]["variation_difference"]
        c2 = {"mean": {m: -lvd["mean"][m] for m in MET},
              "interval": {m: {"low": -lvd["interval"][m]["high"], "high": -lvd["interval"][m]["low"],
                               "valid_resamples": VALID[m]} for m in MET}}
        one = {}
        for c in CONDS:
            for u in UNITS:
                for n in NETS:
                    if (c, u, n) not in self.counts:
                        continue
                    e = one.setdefault((n, c), {"models": 0, "one_class": 0, "all_bona_fide": 0, "all_attack": 0})
                    e["models"] += 1
                    r = self.rates(c, u, n)
                    if (r["apcer"], r["bpcer"]) in ((0, 1), (1, 0)):
                        e["one_class"] += 1
                        e["all_attack" if r["apcer"] == 0 else "all_bona_fide"] += 1
        toc = [{"network": n, "patience": c[0], "bn_state": c[1], "threshold": c[2], **one[(n, c)],
                "text": f"{one[(n, c)]['one_class']} of {one[(n, c)]['models']}"}
               for n in NETS for p in (4, 12) for st in STATES for t in ("dev_eer", "fixed_0.5")
               for c in [(p, st, t)]]
        models = []
        for p in (4, 12):
            for k in ANETS:
                tag = {"AttackNetV1": "v1", "AttackNetV2_1": "v2_1", "AttackNetV2_2": "v2_2"}[k]
                for u in UNITS:
                    if self.present(k, p, u):
                        models.append({"run": f"anet_{tag}_p{p}", "network": k, "fold": u[0], "seed": u[1],
                                       "weights_sha256": hashlib.sha256(f"{k}{p}{u}".encode()).hexdigest(),
                                       "dev_eer_threshold": {st: float(self.thr[((p, st, "dev_eer"), u, k)])
                                                             for st in STATES}})
        return {"schema": "pad-anet-eval/1", "run_id": M.RUN_ID, "status": "PASS", "official": True,
                "runs": json.loads(json.dumps(self.run_pins)),
                "attempt": {"attempt": 1, "previous": [], "second_reading": False, "earlier_readings": []},
                "second_reading": False, "hooks": {"official_configuration": True, "set": [], "changed": []},
                "primary_analysis": pa, "descriptive": desc,
                "c2_reproduction": {"primary_contrast": c2, "conditions": 8, "exact": True},
                "c2_opposite": {"gap_ordinary": True, "gap_re_estimated": True, "variation_difference": True},
                "not_evaluable": ne, "test_one_class": toc, "models": models,
                "test_frames_per_fold": [a + b for a, b, _ in FOLDS.values()], "test_frame_ids_sha256": self.ids()}

    def units_rows(self):
        rows = [M.UNITS_HEADER]
        for c in CONDS:
            for u in UNITS:
                for n in NETS:
                    if (c, u, n) not in self.counts:
                        continue
                    r = self.rates(c, u, n)
                    rows.append([str(c[0]), c[1], c[2], str(u[0]), str(u[1]), str(sum(FOLDS[u[0]][:2])),
                                 self.ids()[str(u[0])], n, self.thr[(c, u, n)], fs(r["apcer"]), fs(r["bpcer"]),
                                 fs(r["acer"])])
        return rows

    def table_rows(self):
        rows = [M.TABLE_HEADER]
        for st in STATES:
            c = (4, st, "fixed_0.5")
            for n in NETS:
                mine = [u for u in UNITS if (c, u, n) in self.counts]
                mm = {m: sum((self.rates(c, u, n)[m] for u in mine), F(0)) / len(mine) for m in MET}
                folds = sorted({u[0] for u in mine})
                rows.append([n, st, str(len(mine)), fs(mm["apcer"]), fs(mm["bpcer"]), fs(mm["acer"]),
                             str(sum(FOLDS[u[0]][0] for u in mine)), str(sum(FOLDS[u[0]][1] for u in mine)),
                             str(sum(FOLDS[f][2] for f in folds)), "1" if len(mine) == 12 else "0"])
        return rows

    def conditions_rows(self, res):
        rows = [M.CONDITIONS_HEADER]
        for p, t in ((4, "fixed_0.5"), (12, "fixed_0.5"), (4, "dev_eer"), (12, "dev_eer")):
            blk = res["primary_analysis"] if (p, t) == (4, "fixed_0.5") else \
                res["descriptive"]["pairs"][f"patience_{p}_{t}"]
            label = f"patience_{p}_{t}" + (" (primary)" if (p, t) == (4, "fixed_0.5") else "")
            for kind, group in (("variation", blk["variations"]), ("contrast", blk["contrasts"])):
                for n, q in group.items():
                    if not q["computed"]:
                        rows.append([label, kind, n, "0"] + ["NA"] * 10)
                        continue
                    iv = q["interval"]
                    rows.append([label, kind, n, "1", repr(q["mean"]["acer"]), repr(iv["acer"]["low"]),
                                 repr(iv["acer"]["high"]), str(iv["acer"]["valid_resamples"]),
                                 repr(q["mean"]["apcer"]), repr(iv["apcer"]["low"]), repr(iv["apcer"]["high"]),
                                 repr(q["mean"]["bpcer"]), repr(iv["bpcer"]["low"]), repr(iv["bpcer"]["high"])])
        return rows

    def write(self):
        res = self.results()
        self.path("results.json").write_text(json.dumps(res, indent=1))
        self.write_csv("units.csv", self.units_rows())
        self.write_csv("table.csv", self.table_rows())
        self.write_csv("conditions.csv", self.conditions_rows(res))
        self.path("report.md").write_text("# synthetic report\n")
        self.path("reproduction.csv").write_text("run,network\n")
        pin = self.realign()
        st = {"run_id": M.RUN_ID, "status": "PASS", "passed": True, "official": True, "second_reading": False,
              "attempt": res["attempt"], "not_evaluable": res["not_evaluable"], "public_sha256sums_sha256": pin,
              "runs": json.loads(json.dumps(self.run_pins))}
        (self.record / "status.json").write_text(json.dumps(st, indent=1))
        (self.record / "run.log").write_text("synthetic\n")
        r = self.record / "restricted"
        r.mkdir()
        (r / "scores.csv").write_text("never read\n")
        os.chmod(r, 0)
        return pin

    def write_csv(self, name, rows):
        buf = io.StringIO(newline="")
        csv.writer(buf, lineterminator="\n").writerows(rows)
        self.path(name).write_text(buf.getvalue())

    # --- modifiche avversarie, con SHA256SUMS e impronta riallineati -----------------------------------------------
    def realign(self):
        lines = [f"{hashlib.sha256(self.path(n).read_bytes()).hexdigest()}  {n}" for n in M.SUMS_FILES]
        b = ("\n".join(lines) + "\n").encode()
        self.path("SHA256SUMS").write_bytes(b)
        pin = hashlib.sha256(b).hexdigest()
        sp = self.record / "status.json"
        if sp.exists():
            st = json.loads(sp.read_text())
            st["public_sha256sums_sha256"] = pin
            sp.write_text(json.dumps(st, indent=1))
        self.pin = pin
        return pin

    def res(self):
        return json.loads(self.path("results.json").read_text())

    def edit_json(self, fn):
        d = self.res()
        fn(d)
        self.path("results.json").write_text(json.dumps(d, indent=1))
        return self.realign()

    def edit_status(self, fn):
        sp = self.record / "status.json"
        d = json.loads(sp.read_text())
        fn(d)
        sp.write_text(json.dumps(d, indent=1))

    def edit_csv(self, name, fn):
        rows = list(csv.reader(io.StringIO(self.path(name).read_text(), newline="")))
        rows = fn(rows)
        self.write_csv(name, rows)
        return self.realign()

    def build(self, images=False):
        """Le uscite del record sintetico; un rifiuto dello script su questo record, corretto per costruzione, è un
        fallimento del test (asserzione), non un errore."""
        with mock.patch.object(M, "PUBLIC_SUMS_SHA256", self.pin), mock.patch.object(M, "C2_RUNS", self.c2):
            try:
                return M.build(M.load_input_bytes(*self.args()[1::2]), images=images)
            except M.Refusal as e:
                raise AssertionError(f"the script refused the synthetic record, correct by construction: {e}")

    def analyse(self):
        with mock.patch.object(M, "PUBLIC_SUMS_SHA256", self.pin), mock.patch.object(M, "C2_RUNS", self.c2):
            try:
                return M.analyse(M.load_input_bytes(*self.args()[1::2]))
            except M.Refusal as e:
                raise AssertionError(f"the script refused the synthetic record, correct by construction: {e}")


def table_cells(md, first):
    """Le celle della riga di una tabella Markdown che comincia con «| first |»."""
    for line in md.splitlines():
        if line.startswith(f"| {first} |"):
            return [c.strip() for c in line.strip().strip("|").split("|")]
    raise AssertionError(f"no row {first!r}")


class Base(unittest.TestCase):
    world_kw = {}

    def setUp(self):
        self.w = World(**self.world_kw)
        self.addCleanup(self.w.close)

    def refused(self, msg, out_name="out", pin=None):
        """Il comando si ferma con il messaggio atteso e non lascia nulla (nessun file né cartella nuovi)."""
        before = snapshot(self.w.tmp)
        try:
            code, out, err = run([*self.w.args(), "--out", str(self.w.tmp / out_name)], pin or self.w.pin, self.w.c2)
        except Exception as e:                       # un'eccezione invece del rifiuto pulito è un fallimento del test
            self.fail(f"the script ended in an exception instead of a clean refusal: {type(e).__name__}: {e}")
        self.assertEqual(code, 1, out + err)
        self.assertIn(msg, err)
        self.assertIn("RESULT: FAIL", out)
        self.assertNotIn("Traceback", err)
        self.assertEqual(snapshot(self.w.tmp), before)
        return err

    def json_refused(self, fn, msg):
        self.w.edit_json(fn)
        return self.refused(msg)

    def csv_refused(self, name, fn, msg):
        self.w.edit_csv(name, fn)
        return self.refused(msg)


# ---------------------------------------------------------------------------------------------------------------------
# A. i numeri e i testi contro i valori calcolati qui

class A_Numbers(Base):
    def setUp(self):
        super().setUp()
        self.out = {k: v.decode("utf-8") for k, v in self.w.build().items()}
        self.agg = list(csv.DictReader(io.StringIO(self.out["aggregate.csv"], newline="")))

    def agg_row(self, p, t, q, n, st, scope="all seeds", unit=None):
        rows = [r for r in self.agg if (r["patience"], r["threshold"], r["quantity"], r["network"], r["bn_state"],
                                        r["scope"]) == (str(p), t, q, n, st, scope)
                and (unit is None or (r["fold"], r["seed"]) == (str(unit[0]), str(unit[1])))]
        self.assertEqual(len(rows), 1, (p, t, q, n, st, scope, unit))
        return rows[0]

    def test_rates_table(self):
        w = self.w
        for n in NETS:
            for st in STATES:
                c = (4, st, "fixed_0.5")
                mm = {m: sum((w.rates(c, u, n)[m] for u in UNITS), F(0)) / 12 for m in MET}
                cells = [x for x in self.out["primary_table.md"].splitlines()
                         if x.startswith(f"| {n} | {st} |")]
                self.assertEqual(len(cells), 1)
                want = f"| {n} | {st} | {pp(mm['apcer'])} | {pp(mm['bpcer'])} | {pp(mm['acer'])} | 12 | 126 | 84 | 16 |"
                self.assertEqual(cells[0], want)
                r = self.agg_row(4, "fixed_0.5", "rate", n, st)
                self.assertEqual([r[f"{m}_exact"] for m in MET], [fs(mm[m]) for m in MET])
                self.assertEqual((r["n_attack"], r["n_bona_fide"], r["n_subjects"], r["units"]), ("126", "84", "16",
                                                                                                    "12"))

    def test_rates_by_hand(self):
        # LivenessNet fold 1 seed 42 ordinario a 0,5: tutto attacco, APCER 0/10, BPCER 8/8, ACER 1/2 (scritto a mano)
        r = self.agg_row(4, "fixed_0.5", "rate", "LivenessNet", "ordinary", "unit", (1, 42))
        self.assertEqual((r["apcer_exact"], r["bpcer_exact"], r["acer_exact"]), ("0/1", "1/1", "1/2"))
        self.assertEqual((r["n_attack"], r["n_bona_fide"], r["n_subjects"]), ("10", "8", "4"))
        line = [x for x in self.out["appendix_per_seed.md"].splitlines() if x.startswith("| 1 | 42 | LivenessNet |")]
        self.assertTrue(line[0].startswith("| 1 | 42 | LivenessNet | 0.00 | 100.00 | 50.00 |"), line)
        # ResNet-18 fold 3 seed 43: variazione nulla, scritta senza segno
        line = [x for x in self.out["appendix_per_seed.md"].splitlines() if x.startswith("| 3 | 43 | ResNet-18 |")]
        self.assertTrue(line[0].endswith("| 0.00 |  |"), line)
        # 126 attacchi = 3 seed × (10 + 12 + 9 + 11); 84 bona fide = 3 × (8 + 6 + 9 + 5)
        self.assertIn("| LivenessNet | ordinary |", self.out["primary_table.md"])

    def test_table_is_the_primary_condition(self):
        w = self.w
        c = (4, "re-estimated", "fixed_0.5")
        mm = sum((w.rates(c, u, "AttackNetV1")["acer"] for u in UNITS), F(0)) / 12
        self.assertEqual(table_cells(self.out["primary_table.md"], "AttackNetV1")[4], pp(
            sum((w.rates((4, "ordinary", "fixed_0.5"), u, "AttackNetV1")["acer"] for u in UNITS), F(0)) / 12))
        self.assertIn(f"| AttackNetV1 | re-estimated | ", self.out["primary_table.md"])
        line = [x for x in self.out["primary_table.md"].splitlines()
                if x.startswith("| AttackNetV1 | re-estimated |")][0]
        self.assertEqual(line.split("|")[5].strip(), pp(mm))

    def test_rows_order(self):
        md = self.out["primary_table.md"]
        pos = [md.index(f"| {n} | ordinary |") for n in NETS]
        self.assertEqual(pos, sorted(pos))
        for n in NETS:
            self.assertLess(md.index(f"| {n} | ordinary |"), md.index(f"| {n} | re-estimated |"))
        heads = [x.split("|")[1].strip() for x in md.splitlines() if x.startswith("| ") and "[" in x]
        self.assertEqual(heads[:8], list(NETS) + list(ANETS))

    def variation_expected(self, n, keys, p=4, t="fixed_0.5"):
        vals = [self.w.unit_q(p, t, u, "variation", n) for u in keys]
        return {m: sum((v[m] for v in vals), F(0)) / len(keys) for m in MET}

    def test_variations(self):
        res = self.w.res()["primary_analysis"]["variations"]
        for n in NETS:
            mm = self.variation_expected(n, UNITS)
            cells = table_cells(self.out["primary_table.md"].split("**(b)**")[1], n)
            for i, m in enumerate(MET):
                iv = res[n]["interval"][m]
                self.assertEqual(cells[1 + i], f"{pp(mm[m], True)} [{pp(iv['low'], True)}, {pp(iv['high'], True)}]")
            r = self.agg_row(4, "fixed_0.5", "variation", n, "re-estimated minus ordinary")
            self.assertEqual(r["acer_exact"], fs(mm["acer"]))
        # segno: AttackNetV1 peggiora con la ristima nel mondo costruito qui
        self.assertTrue(table_cells(self.out["primary_table.md"].split("**(b)**")[1], "AttackNetV1")[3].startswith("+"))

    def test_contrasts_and_readings(self):
        part = self.out["primary_table.md"].split("**(c)**")[1]
        for k in ANETS:
            vals = [self.w.unit_q(4, "fixed_0.5", u, "contrast", k) for u in UNITS]
            mm = sum((v["acer"] for v in vals), F(0)) / 12
            cells = table_cells(part, k)
            self.assertTrue(cells[3].startswith(pp(mm, True) + " ["), cells)
        self.assertIn('"ACER variation of AttackNetV1 minus that of LivenessNet higher than zero" (interval entirely '
                      'above zero)', part)
        self.assertIn('"ACER variation of AttackNetV2_1 minus that of LivenessNet lower than zero" (interval entirely '
                      'below zero)', part)
        self.assertIn('"unresolved" (interval contains zero)', table_cells(part, "AttackNetV2_2")[5])
        # conclusioni citate, una per AttackNet
        flat = " ".join(self.out["primary_table.md"].split())
        self.assertIn("is algebraically higher for AttackNetV1 than for LivenessNet", flat)
        self.assertIn("is algebraically lower for AttackNetV2_1 than for LivenessNet", flat)
        self.assertIn("does not show that the ACER variation with this BN re-estimation differs between AttackNetV2_2",
                      flat)

    def test_contrast_reference_is_livenessnet(self):
        r = self.agg_row(4, "fixed_0.5", "contrast", "AttackNetV1", "re-estimated minus ordinary", "unit", (2, 42))
        v = self.w.unit_q(4, "fixed_0.5", (2, 42), "variation", "AttackNetV1")
        lv = self.w.unit_q(4, "fixed_0.5", (2, 42), "variation", "LivenessNet")
        self.assertEqual(r["acer_exact"], fs(v["acer"] - lv["acer"]))

    def test_unit_counts_and_tie(self):
        cells = table_cells(self.out["primary_table.md"].split("**(b)**")[1], "ResNet-18")
        vals = [self.w.unit_q(4, "fixed_0.5", u, "variation", "ResNet-18")["acer"] for u in UNITS]
        self.assertGreaterEqual(sum(v == 0 for v in vals), 1)          # almeno la variazione nulla costruita qui
        self.assertEqual(cells[4], f"{sum(v > 0 for v in vals)} / {sum(v < 0 for v in vals)} / "
                                   f"{sum(v == 0 for v in vals)}")
        self.assertIn("ACER units above / below / at zero", self.out["primary_table.md"])

    def test_intervals_copied(self):
        res = self.w.res()
        for p, t, blk in ((4, "fixed_0.5", res["primary_analysis"]),
                          (12, "dev_eer", res["descriptive"]["pairs"]["patience_12_dev_eer"])):
            for n in ("LivenessNet", "ResNet-18"):
                r = self.agg_row(p, t, "variation", n, "re-estimated minus ordinary")
                for m in MET:
                    iv = blk["variations"][n]["interval"][m]
                    self.assertEqual((r[f"{m}_low"], r[f"{m}_high"], r[f"{m}_valid_resamples"]),
                                     (repr(iv["low"]), repr(iv["high"]), str(iv["valid_resamples"])))
        seed = res["primary_analysis"]["per_seed"]["43"]["contrasts"]["AttackNetV2_2"]["interval"]["bpcer"]
        r = self.agg_row(4, "fixed_0.5", "contrast", "AttackNetV2_2", "re-estimated minus ordinary", "seed 43")
        self.assertEqual((r["bpcer_low"], r["bpcer_high"]), (repr(seed["low"]), repr(seed["high"])))

    def test_individual_intervals_declared(self):
        flat = " ".join(self.out["primary_table.md"].split())
        self.assertIn("They are individual 95% intervals: no correction for multiplicity, no simultaneous coverage over "
                      "the contrasts, variations or comparisons", flat)
        self.assertIn("valid resamples 900 for ACER, 950 for APCER and 940 for BPCER, in every block (checked)", flat)

    def test_rounding(self):
        self.assertEqual(f"{M.points(F(1, 800)):.2f}", "0.13")         # 0,125 punti: metà lontano dallo zero
        self.assertEqual(f"{M.points(F(-1, 800)):.2f}", "-0.13")
        self.assertEqual(f"{M.points(F(1, 3)):.2f}", "33.33")
        self.assertEqual(M.fmt(F(-1, 10 ** 7), True), "-0.00")           # il segno resta su 0.00
        self.assertEqual(M.fmt(F(1, 10 ** 7), True), "+0.00")
        self.assertEqual(M.fmt(F(0), True), "0.00")
        self.assertEqual(M.fmt(0.123456), "12.35")

    def test_rounding_from_exact(self):
        # 29/20000 = 0,145 punti esatti: 0.15; il float 0.00145 × 100 vale 0,14499999… (un arrotondamento del float
        # darebbe 0.14)
        self.assertEqual(f"{M.points(F(29, 20000)):.2f}", "0.15")
        self.assertEqual(f"{M.points(F(-29, 20000)):.2f}", "-0.15")
        self.assertEqual(M.fmt_u(F(-29, 20000), True), U + "0.15")

    def test_near_zero(self):
        flat = " ".join(self.out["primary_table.md"].split())
        self.assertIn(f"{U}0.00, +", table_cells(self.out["primary_table.md"].split("**(c)**")[1], "AttackNetV2_2")[3])
        self.assertIn(f"A value written {U}0.00 or +0.00 is non-zero and rounds to zero; its sign is kept (lower bound "
                      f"of the ACER contrast of AttackNetV2_2: {U}0.0040 points)", flat)

    def test_per_seed(self):
        md = self.out["appendix_per_seed.md"]
        res = self.w.res()["primary_analysis"]["per_seed"]
        for s in (42, 43, 44):
            part = md.split(f"## Seed {s}")[1].split("## ")[0]
            keys = [u for u in UNITS if u[1] == s]
            mm = self.variation_expected("LivenessNet", keys)
            iv = res[str(s)]["variations"]["LivenessNet"]["interval"]["acer"]
            cells = table_cells(part.split("| variation of |")[1], "LivenessNet")
            self.assertEqual(cells[3], f"{pp(mm['acer'], True)} [{pp(iv['low'], True)}, {pp(iv['high'], True)}]")
            rates = [x for x in part.splitlines() if x.startswith("| AttackNetV1 | ordinary |")][0]
            m4 = sum((self.w.rates((4, "ordinary", "fixed_0.5"), u, "AttackNetV1")["acer"] for u in keys), F(0)) / 4
            self.assertEqual(rates.split("|")[5].strip(), pp(m4))
        self.assertNotIn("reading (pre-registered rule)", md)
        rows = [x for x in md.splitlines() if re.match(r"\| [1-4] \| 4[234] \| ", x)]
        self.assertEqual(len(rows), 60)

    def test_single_class(self):
        md = self.out["appendix_single_class.md"]
        self.assertEqual(table_cells(md, "LivenessNet")[1], "1 of 12 (1 attack)")
        self.assertEqual(table_cells(md, "ResNet-18")[7], "1 of 12 (1 bona fide)")
        self.assertEqual(table_cells(md, "AttackNetV2_1")[5], "0 of 11")
        flat = " ".join(md.split())
        self.assertIn("LivenessNet fold 1 seed 42, patience 4, ordinary BN, threshold 0.5: every frame attack", flat)
        self.assertIn("ResNet-18 fold 2 seed 43, patience 12, re-estimated BN, threshold 0.5: every frame bona fide",
                      flat)
        rows = list(csv.DictReader(io.StringIO(self.out["single_class.csv"])))
        self.assertEqual([r["split"] for r in rows], ["test"] * 40 + ["dev"] * 20)
        got = {(r["network"], r["patience"], r["bn_state"], r["threshold"]): (r["one_class"], r["all_attack"],
                                                                              r["all_bona_fide"], r["not_evaluable"])
               for r in rows if r["split"] == "test"}
        self.assertEqual(got[("ResNet-18", "12", "re-estimated", "fixed_0.5")], ("1", "0", "1", "0"))
        self.assertEqual(got[("AttackNetV2_1", "12", "ordinary", "dev_eer")], ("0", "0", "0", "1"))
        flat5 = " ".join(self.out["primary_table.md"].split())
        self.assertIn("they enter the means like the others; it is inside the ordinary-BN mean of LivenessNet, and so "
                      "inside its variation and every contrast", flat5)
        self.assertIn("The same model decides every dev frame as attack at threshold 0.5 (Appendix B).", flat5)

    def test_single_class_dev(self):
        md = self.out["appendix_single_class.md"].split("## On the dev")[1]
        self.assertEqual(table_cells(md, "LivenessNet")[1], "1 of 12 evaluable (1 attack, 0 bona fide)")
        self.assertEqual(table_cells(md, "AttackNetV1")[4], "1 of 12 evaluable (0 attack, 1 bona fide)")
        self.assertEqual(table_cells(md, "AttackNetV2_1")[3], "0 of 11 evaluable (0 attack, 0 bona fide)")
        self.assertEqual(table_cells(md, "AttackNetV2_1")[1], "0 of 12 evaluable (0 attack, 0 bona fide)")
        flat = " ".join(md.split())
        self.assertIn("LivenessNet fold 1 seed 42, patience 4, ordinary BN: every dev frame attack; AttackNetV1 fold 2 "
                      "seed 43, patience 12, re-estimated BN: every dev frame bona fide.", flat)
        rows = [r for r in csv.DictReader(io.StringIO(self.out["single_class.csv"])) if r["split"] == "dev"]
        got = {(r["network"], r["patience"], r["bn_state"]): (r["models"], r["one_class"], r["all_attack"],
                                                              r["all_bona_fide"], r["not_evaluable"], r["threshold"])
               for r in rows}
        self.assertEqual(len(got), 20)
        self.assertEqual(got[("AttackNetV2_1", "12", "re-estimated")], ("11", "0", "0", "0", "1", "fixed_0.5"))
        self.assertEqual(got[("AttackNetV1", "12", "re-estimated")], ("12", "1", "0", "1", "0", "fixed_0.5"))
        self.assertEqual(got[("LivenessNet", "4", "ordinary")], ("12", "1", "1", "0", "0", "fixed_0.5"))

    def test_rounding_note(self):
        # la nota 8, ricalcolata qui: celle arrotondate di (a) e (b) contro i valori arrotondati di (b) e (c)
        w, c0 = self.w, (4, "fixed_0.5")
        D = lambda x: Decimal(pp(x, minus="-"))      # noqa: E731
        sgn = lambda d: (U if d < 0 else "+" if d > 0 else "") + f"{abs(d):.2f}"      # noqa: E731
        cases = []
        for m in MET:
            rate = {(n, st): D(sum((w.rates((4, st, "fixed_0.5"), u, n)[m] for u in UNITS), F(0)) / 12)
                    for n in NETS for st in STATES}
            var = {n: D(sum((w.unit_q(*c0, u, "variation", n)[m] for u in UNITS), F(0)) / 12) for n in NETS}
            for n in NETS:
                d = rate[(n, "re-estimated")] - rate[(n, "ordinary")]
                if d != var[n]:
                    cases.append(f"{m.upper()} variation of {n}: {sgn(var[n])} in (b), {sgn(d)} from the cells of (a)")
            for k in ANETS:
                c = D(sum((w.unit_q(*c0, u, "contrast", k)[m] for u in UNITS), F(0)) / 12)
                d1 = (rate[(k, "re-estimated")] - rate[(k, "ordinary")]) - \
                    (rate[("LivenessNet", "re-estimated")] - rate[("LivenessNet", "ordinary")])
                if d1 != c:
                    cases.append(f"{m.upper()} contrast of {k}: {sgn(c)} in (c), {sgn(d1)} from the cells of (a)")
                if var[k] - var["LivenessNet"] != c:
                    cases.append(f"{m.upper()} contrast of {k}: {sgn(c)} in (c), {sgn(var[k] - var['LivenessNet'])} "
                                 f"from the variations of (b)")
        self.assertTrue(cases)                       # il mondo sintetico contiene almeno un caso
        flat = " ".join(self.out["primary_table.md"].split())
        self.assertIn("Variations and contrasts are computed from the exact fractions: a difference of rounded values "
                      f"can differ from them (here by 0.01) ({'; '.join(cases)}).", flat)

    def test_dev_same_text_only_when_equal(self):
        # «The same … models decide every dev frame …» solo se i modelli a classe unica sul dev e sul test coincidono
        a = {"single_models": [((4, "ordinary", "fixed_0.5"), (1, 42), "LivenessNet", "attack", F(1, 2))],
             "dev_models": [("LivenessNet", 4, "ordinary", 2, 43, "attack")]}
        self.assertEqual(M.dev_same_text(a), "The one-class counts on the dev are in Appendix B.")
        a["dev_models"] = [("LivenessNet", 4, "ordinary", 1, 42, "attack")]
        self.assertEqual(M.dev_same_text(a), "The same model decides every dev frame as attack at threshold 0.5 "
                                             "(Appendix B).")

    def test_leave_one_out(self):
        part = self.out["descriptive.md"].split("## Leaving one unit out")[1].split("## ")[0]
        self.assertEqual(part.count("Leaving one unit out"), 1)
        keep = [u for u in UNITS if u != (1, 42)]
        lv = sum((self.w.unit_q(4, "fixed_0.5", u, "variation", "LivenessNet")["acer"] for u in keep), F(0)) / 11
        c2 = sum((self.w.unit_q(4, "fixed_0.5", u, "contrast", "AttackNetV2_1")["acer"] for u in keep), F(0)) / 11
        cells = table_cells(part, "fold 1, seed 42")
        self.assertEqual((cells[1], cells[7]), (pp(lv, True), pp(c2, True)))
        allv = sum((self.w.unit_q(4, "fixed_0.5", u, "variation", "ResNet-18")["acer"] for u in UNITS), F(0)) / 12
        self.assertEqual(table_cells(part, "none (all 12 units)")[5], pp(allv, True))
        self.assertEqual(len([x for x in part.splitlines() if x.startswith("| fold ")]), 12)
        self.assertIn("not in `PROTOCOL.md`", part)

    def test_per_fold(self):
        part = self.out["appendix_per_seed.md"].split("## Per fold")[1]
        vals = [self.w.unit_q(4, "fixed_0.5", (2, s), "contrast", "AttackNetV1")["acer"] for s in (42, 43, 44)]
        line = [x for x in part.splitlines() if x.startswith("| 2 | contrast of AttackNetV1 |")]
        self.assertEqual(line, [f"| 2 | contrast of AttackNetV1 | {pp(sum(vals, F(0)) / 3, True)} | "
                                f"{sum(v > 0 for v in vals)} / {sum(v < 0 for v in vals)} / {sum(v == 0 for v in vals)} |"])
        self.assertIn("No interval:", part)

    def test_per_unit_contrast_column(self):
        line = [x for x in self.out["appendix_per_seed.md"].splitlines() if x.startswith("| 2 | 42 | AttackNetV1 |")]
        c = self.w.unit_q(4, "fixed_0.5", (2, 42), "contrast", "AttackNetV1")["acer"]
        v = self.w.unit_q(4, "fixed_0.5", (2, 42), "variation", "AttackNetV1")["acer"]
        self.assertTrue(line[0].endswith(f"| {pp(v, True)} | {pp(c, True)} |"), line)

    def test_descriptive_values(self):
        desc = self.out["descriptive.md"]
        res = self.w.res()
        part = desc.split("**patience 4, threshold 0.5**")[1].split("**")[0]
        vals = [self.w.unit_q(4, "fixed_0.5", u, "gap", "AttackNetV2_2", "ordinary")["acer"] for u in UNITS]
        iv = res[PA]["descriptive_vs_resnet18"]["AttackNetV2_2"]["gap_ordinary"]["interval"]["acer"]
        row = [x for x in part.splitlines() if x.startswith("| AttackNetV2_2 | gap, ordinary BN |")][0]
        self.assertIn(f"| {pp(sum(vals, F(0)) / 12, True)} [{pp(iv['low'], True)}, {pp(iv['high'], True)}] |", row)
        part = desc.split("## Patience 12, dev EER threshold")[1].split("## ")[0]
        vals = [self.w.unit_q(12, "dev_eer", u, "variation", "LivenessNet")["acer"] for u in UNITS]
        iv = res["descriptive"]["pairs"]["patience_12_dev_eer"]["variations"]["LivenessNet"]["interval"]["acer"]
        cells = table_cells(part.split("| variation of |")[1], "LivenessNet")
        self.assertEqual(cells[3], f"{pp(sum(vals, F(0)) / 12, True)} [{pp(iv['low'], True)}, {pp(iv['high'], True)}]")

    @need_mpl
    def test_figure_labels(self):
        # i testi disegnati nella figura: gli 8 «rete: media [low, high]» e i 2 titoli
        import matplotlib.axes
        drawn = []
        real_text = matplotlib.axes.Axes.text

        def spy(ax, x, y, s, *a, **k):
            drawn.append(s)
            return real_text(ax, x, y, s, *a, **k)
        with mock.patch.object(matplotlib.axes.Axes, "text", autospec=True, side_effect=spy):
            self.w.build(images=True)
        res = self.w.res()[PA]
        want = []
        for q, nets in (("variations", NETS), ("contrasts", ANETS)):
            for n in nets:
                vals = [self.w.unit_q(4, "fixed_0.5", u, q[:-1], n)["acer"] for u in UNITS]
                iv = res[q][n]["interval"]["acer"]
                want.append(f"{n}: {pp(sum(vals, F(0)) / 12, True)} [{pp(iv['low'], True)}, {pp(iv['high'], True)}]")
        self.assertEqual([x for x in drawn if ": " in x and "[" in x], want)
        self.assertIn("ACER variation: re-estimated − ordinary BN", drawn)
        self.assertIn("Contrast: variation of the AttackNet − variation of LivenessNet", drawn)

    def test_thousands(self):
        self.assertEqual((M.thousands("22527"), M.thousands("925"), M.thousands("")), ("22,527", "925", ""))

    def test_not_evaluable_and_not_computed(self):
        md = " ".join(self.out["appendix_single_class.md"].split())
        self.assertIn("Not evaluable models: 1 (`not_evaluable` of `results.json`): AttackNetV2_1_fold3_seed44 at "
                      "patience 12 (not evaluable: no finite val_loss); AttackNetV2_1 11 of 12 fold-seed pairs", md)
        desc = self.out["descriptive.md"]
        part = desc.split("## Patience 12, threshold 0.5")[1].split("## ")[0]
        self.assertIn("| AttackNetV2_1 | not computed (11 of 12 pairs evaluable in both BN states): "
                      "anet_v2_1_p12/AttackNetV2_1_fold3_seed44: not evaluable: no finite val_loss | | | |", part)
        r = self.agg_row(12, "fixed_0.5", "contrast", "AttackNetV2_1", "re-estimated minus ordinary")
        self.assertEqual((r["computed"], r["acer"], r["acer_exact"], r["acer_low"]), ("0", "", "", ""))
        r = self.agg_row(12, "fixed_0.5", "rate", "AttackNetV2_1", "ordinary")
        self.assertEqual((r["units"], r["computed"], r["n_attack"]), ("11", "0", str(126 - 9)))

    def test_descriptive_marked(self):
        desc = self.out["descriptive.md"]
        self.assertIn("reported with their intervals and no reading", " ".join(desc.split()))
        for h in ("## Patience 12, threshold 0.5 (descriptive)", "## Patience 4, dev EER threshold (descriptive)",
                  "## Patience 12, dev EER threshold (descriptive)", "## Rankings (descriptive"):
            self.assertIn(h, desc)
        self.assertNotIn("higher than zero", desc)
        self.assertNotIn("reading (pre-registered rule)", desc)
        for r in self.agg:
            self.assertEqual(r["analysis"], "primary" if (r["patience"], r["threshold"]) == ("4", "fixed_0.5")
                             else "descriptive")
            if r["reading"]:
                self.assertEqual((r["patience"], r["threshold"], r["quantity"], r["scope"]),
                                 ("4", "fixed_0.5", "contrast", "all seeds"))

    def test_comparison_with_resnet18(self):
        r = self.agg_row(4, "fixed_0.5", "gap_vs_resnet18", "AttackNetV2_2", "ordinary")
        vals = [self.w.unit_q(4, "fixed_0.5", u, "gap", "AttackNetV2_2", "ordinary")["apcer"] for u in UNITS]
        self.assertEqual(r["apcer_exact"], fs(sum(vals, F(0)) / 12))
        r = self.agg_row(4, "dev_eer", "variation_difference_vs_resnet18", "LivenessNet",
                         "re-estimated minus ordinary")
        vals = [self.w.unit_q(4, "dev_eer", u, "vdiff", "LivenessNet")["bpcer"] for u in UNITS]
        self.assertEqual(r["bpcer_exact"], fs(sum(vals, F(0)) / 12))

    def test_rankings_with_tie(self):
        desc = self.out["descriptive.md"]
        line = [x for x in desc.splitlines() if x.startswith("| patience 12, dev EER threshold | ordinary |")][0]
        rk = self.w.ranking((12, "ordinary", "dev_eer"))
        ranks = {e["network"]: e["rank"] for e in rk["order"]}
        self.assertEqual(ranks["AttackNetV1"], ranks["AttackNetV2_2"])        # pari merito costruito qui
        self.assertIn(f"{ranks['AttackNetV1']}. AttackNetV1", line)
        self.assertIn(f"{ranks['AttackNetV2_2']}. AttackNetV2_2", line)
        c = rk["unit_rank_counts"]["AttackNetV1"]
        self.assertIn("AttackNetV1 " + " / ".join(str(c[str(i)]) for i in range(1, len(c) + 1)), line)
        # condizione con un modello non valutabile: AttackNetV2_1 non in graduatoria a patience 12
        line = [x for x in desc.splitlines() if x.startswith("| patience 12, threshold 0.5 | ordinary |")][0]
        self.assertIn("Not ranked: AttackNetV2_1.", line)

    def test_denominators_text(self):
        flat = " ".join(self.out["primary_table.md"].split())
        self.assertIn("Summed over the 12 units: 210 test frames, 126 attack and 84 bona fide; each test frame counts "
                      "once per seed (70 distinct test frames: 42 attack and 28 bona fide).", flat)
        self.assertIn("fold 1: 18 frames (10 attack, 8 bona fide), 4 subjects; fold 2: 18 frames (12 attack, 6 bona "
                      "fide), 4 subjects; fold 3: 18 frames (9 attack, 9 bona fide), 4 subjects; fold 4: 16 frames (11 "
                      "attack, 5 bona fide), 4 subjects", flat)

    def test_figure_values(self):
        agg = M.Agg(self.out["aggregate.csv"].encode())
        items = M.figure_items(agg)
        self.assertEqual([(q, n) for q, n, *_ in items], [("variation", n) for n in NETS] +
                         [("contrast", k) for k in ANETS])
        res = self.w.res()["primary_analysis"]
        for q, n, mu, lo, hi, units, *_ in items:
            blk = res["variations" if q == "variation" else "contrasts"][n]
            iv = blk["interval"]["acer"]
            vals = [self.w.unit_q(4, "fixed_0.5", u, q, n)["acer"] for u in UNITS]
            self.assertEqual(mu, float(pp(sum(vals, F(0)) / 12, minus="-")))
            self.assertEqual((lo, hi), (float(pp(iv["low"], minus="-")), float(pp(iv["high"], minus="-"))))
            self.assertEqual(sorted(v for v, _ in units), sorted(float(pp(v, minus="-")) for v in vals))

    def test_unit_levels(self):
        self.assertEqual(M.unit_levels([(0.0, (1, 42)), (1.0, (1, 43)), (1.5, (1, 44)), (10.0, (2, 42))]), [0, 1, 2, 0])
        self.assertEqual(M.unit_levels([(5.0, (2, 42)), (5.0, (1, 42))]), [1, 0])

    def test_tex(self):
        tex = self.out["primary_table.tex"]
        self.assertIn("AttackNetV2\\_1 & ordinary & ", tex)
        self.assertNotIn(U, tex)
        r = self.agg_row(4, "fixed_0.5", "contrast", "AttackNetV2_1", "re-estimated minus ordinary")
        self.assertIn(f"${pp(F(r['acer_exact']), True, '-')}$", tex)

    def test_aggregate_regenerates_tables(self):
        # le tabelle e la figura si rigenerano dal solo CSV aggregato: cambiato un valore del CSV, cambia la tabella
        agg_b = self.out["aggregate.csv"].encode()
        a = self.w.analyse()
        self.assertEqual(M.primary_tex(M.Agg(agg_b)).decode(), self.out["primary_table.tex"])
        rows = list(csv.DictReader(io.StringIO(self.out["aggregate.csv"], newline="")))
        for r in rows:
            if (r["quantity"], r["network"], r["scope"], r["patience"], r["threshold"]) == \
                    ("rate", "LivenessNet", "all seeds", "4", "fixed_0.5") and r["bn_state"] == "ordinary":
                r["acer_exact"] = "1/3"
        changed = M.csv_bytes(rows, M.AGG_HEADER)
        self.assertIn("33.33", table_cells(M.primary_md(a, M.Agg(changed)).decode(), "LivenessNet")[4])

    def test_no_identifiers_or_paths_in_outputs(self):
        ids = list(World.ids().values())
        for name, text in self.out.items():
            for s in ids + [str(self.w.tmp), str(SUITE_TMP), "/Users/", "/home/", "/private/", "/tmp/", "restricted"]:
                self.assertNotIn(s, text, name)


class A2_NotComputedPrimary(Base):
    """Variante: il modello non valutabile è a patience 4, quindi variazione, contrasto e confronto di AttackNetV2_1
    non sono calcolati nell'analisi primaria."""
    world_kw = {"ne": ("AttackNetV2_1", 4, 3, 44)}

    def test_not_computed_cells(self):
        out = {k: v.decode() for k, v in self.w.build().items()}
        md = out["primary_table.md"]
        cells = table_cells(md.split("**(b)**")[1], "AttackNetV2_1")
        self.assertEqual(cells[1], "not computed (11 of 12 pairs evaluable in both BN states): "
                                   "anet_v2_1_p4/AttackNetV2_1_fold3_seed44: not evaluable: no finite val_loss")
        self.assertEqual(cells[2:], ["", "", ""])
        cells = table_cells(md.split("**(c)**")[1], "AttackNetV2_1")
        self.assertEqual(cells[-1], "not computed")
        # il tasso della rete è la media sulle sue 11 unità, con i denominatori di quelle unità
        line = [x for x in md.splitlines() if x.startswith("| AttackNetV2_1 | ordinary |")][0]
        c = (4, "ordinary", "fixed_0.5")
        keys = [u for u in UNITS if u != (3, 44)]
        mm = sum((self.w.rates(c, u, "AttackNetV2_1")["acer"] for u in keys), F(0)) / 11
        self.assertEqual(line, line.split("|")[0] + f"| AttackNetV2_1 | ordinary | {line.split('|')[3].strip()} | "
                         f"{line.split('|')[4].strip()} | {pp(mm)} | 11 | {126 - 9} | {84 - 9} | 16 |")
        items = M.figure_items(M.Agg(out["aggregate.csv"].encode()))
        self.assertEqual(items[2][2:5], (None, None, None))
        tex = out["primary_table.tex"]
        self.assertIn("AttackNetV2\\_1 & \\multicolumn{3}{l}{not computed}", tex)
        self.assertNotIn("| AttackNetV2_1 | +0.00", md)
        line = [x for x in out["appendix_per_seed.md"].splitlines() if x.startswith("| 3 | 44 | AttackNetV2_1 |")][0]
        self.assertEqual(line, "| 3 | 44 | AttackNetV2_1 | not evaluable | | | | | | | |")

    @need_mpl
    def test_figure_with_not_computed(self):
        out = self.w.build(images=True)
        self.assertTrue(M.png_structure_ok(out["figure_acer_variations.png"]))


class A3_BoundAtZero(Base):
    world_kw = {"bound_zero": True}

    def test_valid_record_accepted(self):
        # il record con un estremo esattamente a zero, corretto per costruzione, passa
        ctx = contextlib.nullcontext() if HAS_MPL else mock.patch.object(M, "figure_images", return_value=(b"p", b"d"))
        with ctx:
            code, o, err = run([*self.w.args(), "--out", str(self.w.tmp / "o")], self.w.pin, self.w.c2)
        self.assertEqual((code, err), (0, ""), o + err)

    def test_bound_at_zero_contains_zero(self):
        md = self.w.build()["primary_table.md"].decode()
        cells = table_cells(md.split("**(c)**")[1], "AttackNetV1")
        self.assertEqual(cells[5], '"unresolved" (interval contains zero)')
        self.assertIn("[0.00, +", cells[3])


# ---------------------------------------------------------------------------------------------------------------------
# B. integrità degli input

class B_Integrity(Base):
    def test_pin(self):
        self.refused("expected the pinned", pin="0" * 64)

    def test_one_byte_in_each_listed_file(self):
        for name in M.SUMS_FILES:
            with self.subTest(name=name):
                p = self.w.path(name)
                orig = p.read_bytes()
                b = bytearray(orig)
                b[len(b) // 2] ^= 0x01
                p.write_bytes(bytes(b))
                self.refused(f"public/{name}: SHA-256")
                p.write_bytes(orig)

    def test_sums_rules(self):
        p = self.w.path("SHA256SUMS")
        orig = p.read_text()
        lines = orig.splitlines()
        for text, msg in ((orig[:-1], "does not end with a newline"),
                          ("\n".join(lines + [lines[0]]) + "\n", "lists conditions.csv twice"),
                          ("\n".join(lines[1:]) + "\n", "expected ['conditions.csv'"),
                          ("\n".join(lines + ["0" * 64 + "  extra.csv"]) + "\n", "expected ['conditions.csv'"),
                          (orig.replace("  ", " ", 1), "malformed line"),
                          (orig + "é\n", "not ASCII")):
            with self.subTest(msg=msg):
                p.write_bytes(text.encode("utf-8"))
                self.refused(msg, pin=hashlib.sha256(text.encode("utf-8")).hexdigest())
        p.write_text(orig)

    def test_sums_not_ascii(self):
        # un byte che non è ASCII (né UTF-8) nel SHA256SUMS pubblico: rifiuto pulito, nessun traceback
        p = self.w.path("SHA256SUMS")
        b = p.read_bytes() + b"\xff\n"
        p.write_bytes(b)
        self.refused("public/SHA256SUMS is not ASCII", pin=hashlib.sha256(b).hexdigest())

    def test_missing_files_and_types(self):
        name = "units.csv"
        real = self.w.tmp / "real_units.csv"
        shutil.copy(self.w.path(name), real)
        self.w.path(name).unlink()
        os.symlink(real, self.w.path(name))
        self.refused("public units.csv is a symbolic link", out_name="o1")
        self.w.path(name).unlink()
        self.w.path(name).mkdir()
        self.refused("public units.csv is not a regular file", out_name="o2")
        self.w.path(name).rmdir()
        os.mkfifo(self.w.path(name))
        self.refused("public units.csv is not a regular file", out_name="o3")
        self.w.path(name).unlink()
        self.refused("missing public units.csv", out_name="o4")
        shutil.copy(real, self.w.path(name))
        st = self.w.record / "status.json"
        st.rename(self.w.tmp / "status.json")
        self.refused("missing status.json of the record", out_name="o5")
        (self.w.tmp / "status.json").rename(st)
        shutil.rmtree(self.w.public)
        self.refused("the public directory of the record does not exist", out_name="o6")

    def test_input_changed_while_opened(self):
        real_fstat = os.fstat

        def other_inode(fd):
            st = real_fstat(fd)
            return os.stat_result((st.st_mode, st.st_ino + 1) + tuple(st)[2:])
        with mock.patch.object(M.os, "fstat", other_inode):
            self.refused("changed while it was opened")

    def test_restricted_never_opened(self):
        seen = []
        real_open, real_lstat = os.open, os.lstat

        def spy_open(path, *a, **k):
            seen.append(str(path))
            return real_open(path, *a, **k)

        def spy_lstat(path, *a, **k):
            seen.append(str(path))
            return real_lstat(path, *a, **k)
        with mock.patch.object(M.os, "open", spy_open), mock.patch.object(M.os, "lstat", spy_lstat), \
                mock.patch.object(M.os, "listdir", side_effect=AssertionError("listdir")):
            out = self.w.build()
        self.assertTrue(out)
        self.assertTrue(seen)
        self.assertFalse([p for p in seen if "restricted" in p], seen)
        self.assertEqual(os.stat(self.w.record / "restricted").st_mode & 0o777, 0)     # ancora illeggibile


# ---------------------------------------------------------------------------------------------------------------------
# C. una fixture avversaria per ogni controllo (SHA256SUMS e impronta riallineati)

def split_path(path):
    """Percorso a punti; «fixed_0.5» è una chiave sola."""
    return [k.replace("\0", "fixed_0.5") for k in path.replace("fixed_0.5", "\0").split(".")]


def set_path(d, path, value):
    cur = d
    keys = split_path(path)
    for k in keys[:-1]:
        cur = cur[k] if isinstance(cur, dict) else cur[int(k)]
    if isinstance(cur, list):
        cur[int(keys[-1])] = value
    else:
        cur[keys[-1]] = value


def get_path(d, path):
    cur = d
    for k in split_path(path):
        cur = cur[k] if isinstance(cur, dict) else cur[int(k)]
    return cur


class C_Status(Base):
    def test_status_fields(self):
        for path, value, msg in (("run_id", "other", "status.json: run_id"), ("status", "FAIL", "status.json: status"),
                                 ("passed", False, "status.json: passed"), ("official", False, "status.json: official"),
                                 ("second_reading", True, "status.json: second_reading"),
                                 ("attempt.attempt", 2, "status.json: attempt"),
                                 ("public_sha256sums_sha256", "0" * 64, "status.json: public_sha256sums_sha256"),
                                 ("not_evaluable", [], "not_evaluable differs")):
            with self.subTest(path=path):
                orig = json.loads((self.w.record / "status.json").read_text())
                self.w.edit_status(lambda d: set_path(d, path, value))
                self.refused(msg)
                (self.w.record / "status.json").write_text(json.dumps(orig))

    def test_status_not_json(self):
        (self.w.record / "status.json").write_text("{")
        self.refused("status.json is not valid JSON")


class C_Header(Base):
    def test_valid_record_accepted(self):
        # il record sintetico, corretto per costruzione, passa: le mutazioni che rifiutano dati corretti cadono qui
        # senza matplotlib la figura è sostituita da due byte (il test riguarda i controlli, non il disegno)
        ctx = contextlib.nullcontext() if HAS_MPL else mock.patch.object(M, "figure_images", return_value=(b"p", b"d"))
        with ctx:
            code, o, err = run([*self.w.args(), "--out", str(self.w.tmp / "o")], self.w.pin, self.w.c2)
        self.assertEqual((code, err), (0, ""), o + err)
        self.assertEqual(sorted(os.listdir(self.w.tmp / "o")), sorted(M.OUTPUTS))

    def test_results_fields(self):
        for path, value, msg in (("schema", "x", "results.json: schema"), ("run_id", "x", "results.json: run_id"),
                                 ("status", "FAIL", "results.json: status"),
                                 ("official", False, "results.json: official"),
                                 ("second_reading", True, "results.json: second_reading"),
                                 ("attempt.attempt", 2, "results.json: attempt"),
                                 ("hooks.set", ["x"], "results.json: hooks.set"),
                                 ("c2_reproduction.exact", False, "results.json: c2_reproduction.exact"),
                                 ("c2_opposite.gap_ordinary", False, "results.json: c2_opposite"),
                                 ("primary_analysis.conditions.ordinary", [12, "ordinary", "fixed_0.5"],
                                  "primary_analysis: conditions is")):
            with self.subTest(path=path):
                orig = self.w.path("results.json").read_text()
                self.w.edit_json(lambda d: set_path(d, path, value))
                self.refused(msg)
                self.w.path("results.json").write_text(orig)
                self.w.realign()

    def test_json_parsing(self):
        for text, msg in (('{"a": 1, "a": 2}', "duplicate key"), ('{"a": NaN}', "non-finite number"),
                          ("[1]", "the top level is not an object"), ("{", "not valid JSON")):
            with self.subTest(msg=msg):
                self.w.path("results.json").write_text(text)
                self.w.realign()
                self.refused(msg)

    def test_malformed_structure_fails_cleanly(self):
        self.json_refused(lambda d: set_path(d, "c2_reproduction.primary_contrast.interval.acer", 5),
                          "unexpected structure of the input files (TypeError")

    def test_frames_and_ids(self):
        self.json_refused(lambda d: set_path(d, "test_frames_per_fold.0", 0), "test_frames_per_fold is not 4 positive")

    def test_frame_ids_are_sha256(self):
        # un identificativo che non è uno SHA-256, uguale in results.json e in units.csv: lo ferma solo la forma
        bad = "x" * 64
        self.w.edit_json(lambda d: set_path(d, "test_frame_ids_sha256.4", bad))
        self.w.edit_csv("units.csv", lambda rows: [r if r[3] != "4" or r is rows[0] else r[:6] + [bad] + r[7:]
                                                   for r in rows])
        self.refused("test_frame_ids_sha256 is not one SHA-256 per fold")

    def test_frame_ids_one_per_fold(self):
        self.json_refused(lambda d: d["test_frame_ids_sha256"].pop("4"), "test_frame_ids_sha256 is not one SHA-256")

    def test_c2_opposite_values(self):
        self.json_refused(lambda d: set_path(d, "c2_reproduction.primary_contrast.interval.acer.low",
                                             -d["c2_reproduction"]["primary_contrast"]["interval"]["acer"]["low"]),
                          "not the exact opposite of the C2 contrast")

    def test_c2_opposite_mean(self):
        self.json_refused(lambda d: set_path(d, "c2_reproduction.primary_contrast.mean.bpcer",
                                             -d["c2_reproduction"]["primary_contrast"]["mean"]["bpcer"]),
                          "not the exact opposite of the C2 contrast")


class C_Denominators(Base):
    def test_attack_plus_bona(self):
        self.json_refused(lambda d: set_path(d, "primary_analysis.denominators.ordinary.per_fold.2.attack", 11),
                          "frames differ from test_frames_per_fold or from attack + bona fide")

    def test_frames_of_the_fold(self):
        def fn(d):
            e = d["primary_analysis"]["denominators"]["ordinary"]["per_fold"]["2"]
            e["attack"], e["frames"] = e["attack"] + 1, e["frames"] + 1
        self.json_refused(fn, "frames differ from test_frames_per_fold")

    def test_subjects_add_up(self):
        self.json_refused(lambda d: set_path(d, "primary_analysis.denominators.ordinary.per_fold.1.subjects", 5),
                          "do not add up to 16")

    def test_totals(self):
        self.json_refused(lambda d: set_path(
            d, "primary_analysis.denominators.re-estimated.attack_frames_summed_over_units", 127),
            "totals differ from the per-fold values")

    def test_same_in_every_condition(self):
        def fn(d):
            e = d["descriptive"]["pairs"]["patience_12_dev_eer"]["denominators"]["ordinary"]["per_fold"]
            e["1"]["subjects"], e["2"]["subjects"] = 3, 5
        self.json_refused(fn, "differs from the other conditions")


def unit_edit(match, field, value):
    """Funzione per edit_csv: cambia un campo delle righe di units.csv che soddisfano match."""
    def fn(rows):
        h = rows[0]
        hit = 0
        for r in rows[1:]:
            d = dict(zip(h, r))
            if match(d):
                r[h.index(field)] = value(d) if callable(value) else value
                hit += 1
        assert hit, "no row matched"
        return rows
    return fn


def is_unit(p, st, t, f, s, n):
    return lambda d: (d["patience"], d["bn_state"], d["threshold"], d["fold"], d["seed"], d["network"]) == \
        (str(p), st, t, str(f), str(s), n)


class C_Units(Base):
    def test_threshold_half(self):
        self.csv_refused("units.csv", unit_edit(is_unit(12, "ordinary", "fixed_0.5", 2, 42, "AttackNetV1"),
                                                "threshold_value", "0.50"), "effective threshold '0.50', expected 0.5")

    def test_dev_eer_threshold_form(self):
        for v in ("1.5", "nan", "0.30", "x"):
            with self.subTest(v=v):
                orig = self.w.path("units.csv").read_text()
                self.csv_refused("units.csv", unit_edit(is_unit(4, "re-estimated", "dev_eer", 1, 43, "ResNet-18"),
                                                        "threshold_value", v), "threshold is not a number between")
                self.w.path("units.csv").write_text(orig)
                self.w.realign()

    def test_condition_fold_network(self):
        for field, value, msg in (("patience", "8", "condition outside the plan"),
                                  ("threshold", "eer", "condition outside the plan"),
                                  ("seed", "45", "fold or seed outside the plan"),
                                  ("network", "VGG", "network outside the plan")):
            with self.subTest(field=field):
                orig = self.w.path("units.csv").read_text()
                self.csv_refused("units.csv", unit_edit(is_unit(4, "ordinary", "fixed_0.5", 1, 42, "ResNet-18"),
                                                        field, value), msg)
                self.w.path("units.csv").write_text(orig)
                self.w.realign()

    def test_n_frames(self):
        self.csv_refused("units.csv", unit_edit(is_unit(12, "re-estimated", "dev_eer", 4, 44, "LivenessNet"),
                                                "n_frames", "17"), "n_frames '17' differs")

    def test_frame_ids(self):
        self.csv_refused("units.csv", unit_edit(is_unit(12, "re-estimated", "dev_eer", 4, 44, "LivenessNet"),
                                                "frame_ids_sha256", World.ids()["1"]), "frame_ids_sha256 differs")

    def test_fractions(self):
        for v, msg in (("2/20", "is not written reduced"), ("1/0", "is not an exact fraction"),
                       ("0.1", "is not an exact fraction"), ("11/10", "apcer outside [0, 1]")):
            with self.subTest(v=v):
                orig = self.w.path("units.csv").read_text()
                self.csv_refused("units.csv", unit_edit(is_unit(4, "ordinary", "dev_eer", 1, 44, "AttackNetV1"),
                                                        "apcer", v), msg)
                self.w.path("units.csv").write_text(orig)
                self.w.realign()

    def test_acer_is_the_mean(self):
        self.csv_refused("units.csv", unit_edit(is_unit(4, "ordinary", "dev_eer", 1, 44, "AttackNetV1"), "acer",
                                                lambda d: fs(F(d["acer"]) + F(1, 1000))),
                         "acer is not the mean of APCER and BPCER")

    def test_rate_is_a_count_of_the_fold(self):
        # 1/7 non è un numero di attacchi su 10 (fold 1): ACER aggiornato perché scatti solo questo controllo
        def fn(rows):
            h = rows[0]
            for r in rows[1:]:
                d = dict(zip(h, r))
                if is_unit(4, "ordinary", "dev_eer", 1, 44, "AttackNetV1")(d):
                    r[h.index("apcer")] = "1/7"
                    r[h.index("acer")] = fs((F(1, 7) + F(d["bpcer"])) / 2)
            return rows
        self.csv_refused("units.csv", fn, "is not a count over the attack or bona fide frames of the fold")

    def test_bpcer_is_a_count_of_the_fold(self):
        def fn(rows):
            h = rows[0]
            for r in rows[1:]:
                d = dict(zip(h, r))
                if is_unit(4, "ordinary", "dev_eer", 1, 44, "AttackNetV1")(d):
                    r[h.index("bpcer")] = "1/7"
                    r[h.index("acer")] = fs((F(1, 7) + F(d["apcer"])) / 2)
            return rows
        self.csv_refused("units.csv", fn, "is not a count over the attack or bona fide frames of the fold")

    def test_reused_network_missing(self):
        self.csv_refused("units.csv", lambda rows: [r for r in rows
                                                    if not is_unit(12, "ordinary", "fixed_0.5", 3, 42, "ResNet-18")(
                                                        dict(zip(rows[0], r)))],
                         "ResNet-18 missing in (12, 'ordinary', 'fixed_0.5'), fold 3 seed 42")

    def test_network_twice(self):
        def fn(rows):
            h = rows[0]
            dup = [r for r in rows[1:] if is_unit(4, "ordinary", "fixed_0.5", 1, 42, "AttackNetV1")(dict(zip(h, r)))]
            return rows + dup
        self.csv_refused("units.csv", fn, "AttackNetV1 twice in")

    def test_presence_same_in_the_four_conditions(self):
        self.csv_refused("units.csv", lambda rows: [r for r in rows
                                                    if not is_unit(4, "re-estimated", "dev_eer", 2, 43, "AttackNetV1")(
                                                        dict(zip(rows[0], r)))],
                         "AttackNetV1 at patience 4 is not present in the same units in the four conditions")

    def test_csv_format(self):
        for fn, msg in ((lambda rows: [rows[0][:-1]] + rows[1:], "units.csv: unexpected header"),
                        (lambda rows: rows[:5] + [rows[5][:-1]] + rows[6:], "has 11 fields, expected 12")):
            with self.subTest(msg=msg):
                orig = self.w.path("units.csv").read_text()
                self.csv_refused("units.csv", fn, msg)
                self.w.path("units.csv").write_text(orig)
                self.w.realign()
        self.w.path("units.csv").write_bytes(b"\xff\xfe")
        self.w.realign()
        self.refused("units.csv is not UTF-8")


class C_NotEvaluable(Base):
    def test_list_matches_units(self):
        self.w.edit_json(lambda d: d["not_evaluable"].clear())
        self.w.edit_status(lambda d: d["not_evaluable"].clear())
        self.refused("not_evaluable does not list exactly")

    def test_entry_form(self):
        for path, value in (("not_evaluable.0.name", "AttackNetV2_1_fold3_seed43"), ("not_evaluable.0.run", "anet_v1_p12"),
                            ("not_evaluable.0.reason", ""), ("not_evaluable.0.fold", True)):
            with self.subTest(path=path):
                orig = self.w.path("results.json").read_text()

                def fn(d):
                    set_path(d, path, value)
                self.w.edit_json(fn)
                self.w.edit_status(lambda d: d.__setitem__("not_evaluable", self.w.res()["not_evaluable"]))
                self.refused("is not a model of the plan with its run, name and reason")
                self.w.path("results.json").write_text(orig)
                self.w.edit_status(lambda d: d.__setitem__("not_evaluable", json.loads(orig)["not_evaluable"]))
                self.w.realign()

    def test_entry_twice(self):
        def fn(d):
            d["not_evaluable"].append(dict(d["not_evaluable"][0]))
        self.w.edit_json(fn)
        self.w.edit_status(lambda d: d.__setitem__("not_evaluable", self.w.res()["not_evaluable"]))
        self.refused("lists AttackNetV2_1_fold3_seed44 twice")

    def test_guardian(self):
        for path, value in (("descriptive.pairs.patience_12_fixed_0.5.guardian.AttackNetV2_1.pairs", 12),
                            ("descriptive.pairs.patience_12_dev_eer.guardian.AttackNetV2_1.reason", "other"),
                            ("primary_analysis.guardian.AttackNetV1.text", "not computed")):
            with self.subTest(path=path):
                orig = self.w.path("results.json").read_text()
                self.json_refused(lambda d: set_path(d, path, value), "guardian differs from the networks present")
                self.w.path("results.json").write_text(orig)
                self.w.realign()

    def test_models_thresholds(self):
        self.json_refused(lambda d: set_path(d, "models.5.dev_eer_threshold.re-estimated",
                                             d["models"][5]["dev_eer_threshold"]["re-estimated"] + 1e-9),
                          "dev_eer_threshold.re-estimated differs from the threshold of units.csv")

    def test_models_list(self):
        for fn, msg in ((lambda d: d["models"].pop(0), "models does not list exactly"),
                        (lambda d: d["models"].append(dict(d["models"][0])), "twice"),
                        (lambda d: set_path(d, "models.3.run", "anet_v2_1_p4"), "is not an evaluated AttackNet"),
                        (lambda d: set_path(d, "models.3.seed", 45), "is not an evaluated AttackNet")):
            with self.subTest(msg=msg):
                orig = self.w.path("results.json").read_text()
                self.json_refused(fn, msg)
                self.w.path("results.json").write_text(orig)
                self.w.realign()

    def test_model_not_in_units(self):
        def fn(d):
            m = dict(d["models"][0])
            m["fold"], m["seed"], m["run"] = 3, 44, "anet_v2_1_p12"
            m["network"] = "AttackNetV2_1"
            d["models"].append(m)
        self.json_refused(fn, "is not in units.csv")


KEY = ("AttackNetV1", 4)                     # il run delle AttackNet usato dalle fixture
C2KEY = ("C2", 12)


def csv_edit(path, fn):
    rows = list(csv.reader(io.StringIO(path.read_text(), newline="")))
    rows = fn(rows)
    buf = io.StringIO(newline="")
    csv.writer(buf, lineterminator="\n").writerows(rows)
    path.write_text(buf.getvalue())


def status_edit(path, fn):
    d = json.loads(path.read_text())
    fn(d)
    path.write_text(json.dumps(d, indent=1))


class C_Runs(Base):
    """I record dei run e i conteggi sul dev: una fixture per controllo, con SHA256SUMS del run, impronta del suo
    status.json e record della valutazione riallineati, così che scatti la regola e non il checksum."""

    def rid(self, key=KEY):
        return self.w.run_dirs()[key].name

    def test_run_status_pin(self):
        # status.json del run cambiato e il suo SHA256SUMS riallineato, ma l'impronta registrata dalla valutazione no
        self.w.edit_run(KEY, lambda d: status_edit(d / "status.json", lambda x: x.update(extra=1)), realign=False)
        self.refused(f"status.json of {self.rid()}: SHA-256")

    def test_c2_run_pin(self):
        c2 = dict(self.w.c2)
        c2[12] = (c2[12][0], "0" * 64)
        before = snapshot(self.w.tmp)
        code, o, err = run([*self.w.args(), "--out", str(self.w.tmp / "o")], self.w.pin, c2)
        self.assertEqual(code, 1)
        self.assertIn(f"status.json of {self.rid(C2KEY)}: SHA-256", err)
        self.assertIn("differs from the pinned " + "0" * 64, err)
        self.assertEqual(snapshot(self.w.tmp), before)

    def test_runs_entry(self):
        self.w.edit_json(lambda d: d["runs"]["anet_v1_p4"].update(official=False))
        self.w.edit_status(lambda d: d["runs"]["anet_v1_p4"].update(official=False))
        self.refused("results.json: runs.anet_v1_p4 is not the official run")
        self.w.edit_status(lambda d: d["runs"]["anet_v1_p4"].update(official=True))
        self.refused("results.json: runs.anet_v1_p4 is not the official run")
        # le due voci entrambe ufficiali, ma con impronte diverse: vale solo se le due voci si confrontano
        self.w.edit_json(lambda d: d["runs"]["anet_v1_p4"].update(official=True))
        self.w.edit_status(lambda d: d["runs"]["anet_v1_p4"].update(status_sha256="0" * 64))
        self.refused("results.json: runs.anet_v1_p4 is not the official run")

    def test_run_sums_against_status(self):
        # public_sha256sums_sha256 del run diverso dal suo SHA256SUMS; impronta di status.json riallineata
        self.w.edit_run(KEY, lambda d: status_edit(d / "status.json",
                                                   lambda x: x.update(public_sha256sums_sha256="0" * 64)), sums=False)
        self.refused(f"status.json of {self.rid()}: public_sha256sums_sha256 is")

    def test_run_file_against_its_line(self):
        def position(d):
            # un campo che lo script non legge (position) cambiato in models.csv: lo vede solo l'impronta della riga
            csv_edit(d / "public" / "models.csv", lambda rows: [rows[0], [rows[1][0], "99"] + rows[1][2:]] + rows[2:])
        self.w.edit_run(KEY, position, realign=False)
        self.refused(f"public/models.csv of {self.rid()}: SHA-256")
        self.w.edit_run(KEY, lambda d: None)

        def one_byte(d):
            f = d / "public" / "one_class.csv"
            b = bytearray(f.read_bytes())
            b[-3] ^= 1
            f.write_bytes(bytes(b))
        self.w.edit_run(KEY, one_byte, realign=False)
        self.refused(f"public/one_class.csv of {self.rid()}: SHA-256")

    def test_run_sums_rules(self):
        def extra(d):
            f = d / "public" / "SHA256SUMS"
            f.write_text(f.read_text() + "0" * 64 + "  restricted.csv\n")
        self.w.edit_run(KEY, extra, sums=False)
        self.refused(f"public/SHA256SUMS of {self.rid()} lists")

    def test_run_status_fields(self):
        for field, value in (("status", "FAIL"), ("passed", False), ("official", False), ("run", "anet_v1_p12"),
                             ("network", "AttackNetV2_1"), ("run_id", "x")):
            with self.subTest(field=field):
                orig = (self.w.run_dirs()[KEY] / "status.json").read_text()
                self.w.edit_run(KEY, lambda d: status_edit(d / "status.json", lambda x: x.update({field: value})),
                                sums=False)
                self.refused(f"status.json of {self.rid()}: {field} is")
                self.w.edit_run(KEY, lambda d: (d / "status.json").write_text(orig), sums=False)

    def one_class_case(self, fn, msg, key=KEY):
        orig = (self.w.run_dirs()[key] / "public" / "one_class.csv").read_text()
        self.w.edit_run(key, lambda d: csv_edit(d / "public" / "one_class.csv", fn))
        self.refused(msg)
        self.w.edit_run(key, lambda d: (d / "public" / "one_class.csv").write_text(orig))

    def test_one_class_rules(self):
        def setf(i, col, v):
            def fn(rows):
                rows[i][rows[0].index(col)] = v
                return rows
            return fn
        r = self.rid()
        self.one_class_case(setf(1, "one_class", "1"), f"one_class.csv of {r} (AttackNetV1, ordinary): one_class is "
                                                       f"not all_attack + all_bona_fide")
        self.one_class_case(setf(1, "text", "0 of 12"), "text differs from the counts")
        self.one_class_case(setf(2, "not_evaluable", "1"), "evaluable + not_evaluable is not 12")
        self.one_class_case(setf(2, "evaluable", "x"), "evaluable is not an integer")
        self.one_class_case(lambda rows: rows[:-1], f"one_class.csv of {r}: the rows are not")
        self.one_class_case(lambda rows: rows + [rows[-1]], f"one_class.csv of {r}: the rows are not")
        self.one_class_case(setf(1, "network", "AttackNetV2_1"), f"one_class.csv of {r}: the rows are not")
        self.one_class_case(setf(1, "patience", "12"), f"one_class.csv of {r}: the rows are not")
        self.one_class_case(lambda rows: [rows[0][:-1]] + rows[1:], f"one_class.csv of {r}: unexpected header")

    def models_case(self, fn, msg, key=KEY, also_one_class=None):
        d = self.w.run_dirs()[key]
        orig = {n: (d / "public" / n).read_text() for n in ("models.csv", "one_class.csv")}

        def edit(d):
            csv_edit(d / "public" / "models.csv", fn)
            if also_one_class:
                csv_edit(d / "public" / "one_class.csv", also_one_class)
        self.w.edit_run(key, edit)
        self.refused(msg)
        self.w.edit_run(key, lambda d: [(d / "public" / n).write_text(t) for n, t in orig.items()])

    def test_one_class_against_models(self):
        def setm(fold, seed, col, v):
            def fn(rows):
                h = rows[0]
                for r in rows[1:]:
                    if (r[h.index("fold")], r[h.index("seed")]) == (str(fold), str(seed)):
                        r[h.index(col)] = v
                return rows
            return fn
        r = self.rid()
        self.models_case(setm(3, 43, "ordinary_one_class_05", "attack"),
                         f"one_class.csv of {r} (AttackNetV1, ordinary): the counts differ from models.csv")
        self.models_case(setm(3, 43, "re_estimated_one_class_05", "bona fide"),
                         f"one_class.csv of {r} (AttackNetV1, re-estimated): the counts differ from models.csv")
        self.models_case(setm(3, 43, "ordinary_one_class_05", "maybe"), "is not no, attack or bona fide")
        self.models_case(setm(3, 43, "seed", "42"), "does not hold the 12 fold-seed models at patience 4")
        self.models_case(setm(3, 43, "patience", "12"), "does not hold the 12 fold-seed models at patience 4")
        self.models_case(lambda rows: [[c for c in x[:-1]] for x in rows], "missing columns")

    def test_evaluable_against_units(self):
        # un modello presente in units.csv marcato non valutabile nel run, con one_class.csv coerente
        def models(rows):
            h = rows[0]
            for r in rows[1:]:
                if (r[h.index("fold")], r[h.index("seed")]) == ("1", "42"):
                    r[h.index("model_status")] = "not evaluable"
            return rows

        def oc(rows):
            h = rows[0]
            for r in rows[1:]:
                r[h.index("evaluable")], r[h.index("not_evaluable")] = "11", "1"
                r[h.index("text")] = f"{r[h.index('one_class')]} of 11 evaluable"
            return rows
        self.models_case(models, "the evaluable models differ from the units of units.csv at patience 4",
                         also_one_class=oc)

    def test_run_folder_symlink_and_missing(self):
        d = self.w.run_dirs()[KEY]
        moved = self.w.tmp / "moved_run"
        d.rename(moved)
        os.symlink(moved, d)
        self.refused(f"the run directory {d.name} is a symbolic link", out_name="o1")
        d.unlink()
        self.refused(f"the run directory {d.name} does not exist", out_name="o2")
        moved.rename(d)
        (self.w.tmp / "c2").rename(self.w.tmp / "c2_away")
        self.refused("the directory of the C2 runs does not exist", out_name="o3")
        (self.w.tmp / "c2_away").rename(self.w.tmp / "c2")


PA = "primary_analysis"


class C_Quantities(Base):
    def edit_refused(self, path, value_fn, msg):
        orig = self.w.path("results.json").read_text()
        self.json_refused(lambda d: set_path(d, path, value_fn(get_path(d, path))), msg)
        self.w.path("results.json").write_text(orig)
        self.w.realign()

    def test_mean_each_metric(self):
        for m in MET:
            with self.subTest(m=m):
                self.edit_refused(f"{PA}.variations.AttackNetV1.mean.{m}", lambda v: v + 1e-12,
                                  f"{PA}.variations.AttackNetV1.mean.{m} = ")

    def test_mean_of_each_quantity_and_scope(self):
        for path in (f"{PA}.contrasts.AttackNetV2_2.mean.acer",
                     f"{PA}.descriptive_vs_resnet18.AttackNetV1.gap_re_estimated.mean.apcer",
                     f"{PA}.descriptive_vs_resnet18.LivenessNet.gap_ordinary.mean.bpcer",
                     "descriptive.pairs.patience_4_dev_eer.variations.ResNet-18.mean.acer",
                     f"{PA}.per_seed.43.contrasts.AttackNetV1.mean.bpcer",
                     "descriptive.pairs.patience_12_dev_eer.per_seed.44.descriptive_vs_resnet18.AttackNetV2_2."
                     "variation_difference.mean.acer"):
            with self.subTest(path=path):
                self.edit_refused(path, lambda v: v + 1e-12, "units.csv gives")

    def test_mean_type(self):
        self.edit_refused(f"{PA}.variations.LivenessNet.mean.acer", lambda v: str(v), "units.csv gives")

    def test_counts(self):
        for k in ("below_zero", "above_zero", "zero", "units"):
            with self.subTest(k=k):
                self.edit_refused(f"{PA}.variations.ResNet-18.counts.{k}", lambda v: v + 1, "counts = ")

    def test_counts_per_seed(self):
        self.edit_refused(f"{PA}.per_seed.43.variations.ResNet-18.counts.zero", lambda v: v + 1, "counts = ")

    def test_interval_low_high(self):
        def swap(d):
            iv = d[PA]["variations"]["AttackNetV2_2"]["interval"]["bpcer"]
            iv["low"], iv["high"] = iv["high"], iv["low"]
        self.json_refused(swap, "has low > high")

    def test_interval_contains_mean(self):
        def fn(d):
            b = d[PA]["per_seed"]["42"]["variations"]["LivenessNet"]
            mu = b["mean"]["apcer"]
            b["interval"]["apcer"].update(low=mu + 0.001, high=mu + 0.002)
        self.json_refused(fn, "per_seed.42.variations.LivenessNet.interval.apcer does not contain its mean")

    def test_interval_form(self):
        for path, fn, msg in ((f"{PA}.variations.LivenessNet.interval.acer.valid_resamples", lambda v: 1001,
                               "does not have 0 to 1,000 valid resamples"),
                              (f"{PA}.variations.LivenessNet.interval.acer.valid_resamples", lambda v: -1,
                               "does not have 0 to 1,000 valid resamples"),
                              (f"{PA}.variations.LivenessNet.interval.acer.low", lambda v: None,
                               "bounds that are not finite numbers"),
                              (f"{PA}.variations.LivenessNet.interval.acer", lambda v: {"low": 0.0, "high": 0.0},
                               "is not an interval")):
            with self.subTest(msg=msg):
                self.edit_refused(path, fn, msg)

    def test_no_interval_without_resamples(self):
        def fn(d):
            for blk in self.all_blocks(d):
                blk["interval"]["acer"] = {"low": 0.0, "high": 0.0, "valid_resamples": 0}
        self.json_refused(fn, "has bounds without valid resamples")

    @staticmethod
    def all_blocks(d):
        out = []

        def walk(x):
            if isinstance(x, dict):
                if "interval" in x and "mean" in x:
                    out.append(x)
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(d["primary_analysis"])
        walk(d["descriptive"])
        return out

    def test_valid_resamples_uniform(self):
        self.edit_refused(f"{PA}.per_seed.44.contrasts.AttackNetV1.interval.bpcer.valid_resamples", lambda v: 939,
                          "the valid resamples of BPCER differ between the blocks")

    def test_valid_resamples_uniform_descriptive(self):
        self.edit_refused("descriptive.pairs.patience_12_fixed_0.5.descriptive_vs_resnet18.LivenessNet.gap_ordinary."
                          "interval.apcer.valid_resamples", lambda v: 949,
                          "the valid resamples of APCER differ between the blocks")

    def test_reading_rule(self):
        def flip(d):
            c = d[PA]["contrasts"]["AttackNetV2_1"]
            c["reading"] = c["reading"].replace("lower", "higher")
        self.json_refused(flip, "reading does not follow the reading rule")

    def test_reading_reason(self):
        self.edit_refused(f"{PA}.contrasts.AttackNetV2_2.reason", lambda v: "interval entirely above zero",
                          "reading does not follow the reading rule")

    def test_reading_rule_per_seed(self):
        def fn(d):
            c = d[PA]["per_seed"]["42"]["contrasts"]["AttackNetV1"]
            names = ("ACER variation of AttackNetV1 minus that of LivenessNet lower than zero",
                     "ACER variation of AttackNetV1 minus that of LivenessNet higher than zero", "unresolved")
            c["reading"] = [x for x in names if x != c["reading"]][0]
        self.json_refused(fn, "per_seed.42.contrasts.AttackNetV1.reading does not follow")

    def test_conclusion(self):
        self.edit_refused(f"{PA}.contrasts.AttackNetV1.conclusion", lambda v: v.replace("higher", "lower"),
                          "conclusion is not the conclusion written")

    def test_reading_in_descriptive_refused(self):
        def fn(d):
            d["descriptive"]["pairs"]["patience_12_fixed_0.5"]["contrasts"]["AttackNetV1"]["reading"] = "x"
        self.json_refused(fn, "does not hold exactly")

    def test_not_computed_block(self):
        self.edit_refused("descriptive.pairs.patience_12_fixed_0.5.contrasts.AttackNetV2_1.reading",
                          lambda v: "unresolved", "is not the 'not computed' block of the guardian")

    def test_per_seed_keys(self):
        def fn(d):
            d[PA]["per_seed"]["45"] = d[PA]["per_seed"]["44"]
        self.json_refused(fn, "per_seed does not hold exactly seeds 42, 43 and 44")

    def test_n_subjects(self):
        for path in (f"{PA}.n_subjects", f"{PA}.per_seed.43.n_subjects",
                     "descriptive.pairs.patience_4_dev_eer.per_seed.42.n_subjects"):
            with self.subTest(path=path):
                self.edit_refused(path, lambda v: 15, "n_subjects is not 16")

    def test_network_order(self):
        def fn(d):
            v = d[PA]["variations"]
            d[PA]["variations"] = {k: v[k] for k in reversed(list(v))}
        self.json_refused(fn, "in this order")

    def test_notes(self):
        for path, msg in ((f"{PA}.intervals", "intervals is"), ("descriptive.pairs.patience_12_dev_eer.presentation",
                                                                "presentation is"),
                          ("descriptive.pairs.patience_4_dev_eer.conditions.ordinary", "conditions is")):
            with self.subTest(path=path):
                self.edit_refused(path, lambda v: [4, "ordinary", "fixed_0.5"] if isinstance(v, list) else v + ".",
                                  msg)


class C_Rows(Base):
    def edit_refused(self, fn, msg):
        orig = self.w.path("results.json").read_text()
        self.json_refused(fn, msg)
        self.w.path("results.json").write_text(orig)
        self.w.realign()

    def test_value_and_exact(self):
        def value(d):
            d[PA]["rows"][4]["contrast/AttackNetV1"]["bpcer"]["value"] += 1e-12

        def exact(d):
            e = d["descriptive"]["pairs"]["patience_4_dev_eer"]["rows"][7]["gap/LivenessNet/ordinary"]["acer"]
            e["exact"] = fs(F(e["exact"]) + F(1, 10 ** 9))
        for fn in (value, exact):
            with self.subTest(fn=fn.__name__):
                self.edit_refused(fn, "differs from units.csv")

    def test_order_and_keys(self):
        def order(d):
            r = d[PA]["rows"]
            r[0], r[1] = r[1], r[0]

        def extra(d):
            d[PA]["rows"][2]["variation/Other"] = d[PA]["rows"][2]["variation/ResNet-18"]
        self.edit_refused(order, "ascending (fold, seed) order")
        self.edit_refused(extra, "does not hold exactly the quantities of the computed networks")


class C_OtherFiles(Base):
    def test_table_csv_each_field(self):
        for col in range(2, 10):
            with self.subTest(col=col):
                orig = self.w.path("table.csv").read_text()

                def fn(rows):
                    r = rows[3]
                    r[col] = "1/3" if col in (3, 4, 5) else ("0" if col == 9 else "999")
                    return rows
                self.csv_refused("table.csv", fn, "table.csv (AttackNetV2_1, ordinary): a field differs")
                self.w.path("table.csv").write_text(orig)
                self.w.realign()

    def test_table_csv_rows(self):
        self.csv_refused("table.csv", lambda rows: rows[:-1], "table.csv: 9 rows, expected 10")

    def test_conditions_csv_each_field(self):
        for col in range(3, 14):
            with self.subTest(col=col):
                orig = self.w.path("conditions.csv").read_text()

                def fn(rows):
                    rows[10][col] = "0.123"
                    return rows
                self.csv_refused("conditions.csv", fn, "a field differs from results.json")
                self.w.path("conditions.csv").write_text(orig)
                self.w.realign()

    def test_conditions_csv_rows(self):
        self.csv_refused("conditions.csv", lambda rows: rows[:-1], "conditions.csv: 31 rows, expected 32")

    def test_test_one_class(self):
        for path, value in (("test_one_class.0.one_class", 1), ("test_one_class.0.models", 11),
                            ("test_one_class.1.all_attack", 0), ("test_one_class.39.all_bona_fide", 0)):
            with self.subTest(path=path):
                orig = self.w.path("results.json").read_text()
                self.json_refused(lambda d: set_path(d, path, value), "test_one_class differs")
                self.w.path("results.json").write_text(orig)
                self.w.realign()

    def test_rankings(self):
        for path, value in (("descriptive.rankings.12/ordinary/dev_eer.order.2.rank", 99),
                            ("descriptive.rankings.4/ordinary/fixed_0.5.unit_rank_counts.LivenessNet.1", 99),
                            ("descriptive.rankings.12/re-estimated/fixed_0.5.not_ranked", []),
                            ("primary_analysis.rankings.re-estimated.order.0.mean_acer", 0.5)):
            with self.subTest(path=path):
                orig = self.w.path("results.json").read_text()
                self.json_refused(lambda d: set_path(d, path, value), "differs from the ranking recomputed")
                self.w.path("results.json").write_text(orig)
                self.w.realign()


# ---------------------------------------------------------------------------------------------------------------------
# D. percorsi, scrittura atomica, opzioni

class D_Paths(Base):
    def files(self):
        return {n: b"x" for n in M.OUTPUTS}

    def go(self, out, pin=True):
        return run([*self.w.args(), "--out", str(out)], self.w.pin if pin else None, self.w.c2 if pin else None)

    def test_record_directory(self):
        for rec, msg in ((self.w.tmp / "none", "the record directory does not exist"),
                         (self.w.path("units.csv"), "the record directory is not a directory")):
            code, o, err = run(["--record", str(rec), "--out", str(self.w.tmp / "o")])
            self.assertEqual(code, 1)
            self.assertIn(msg, err)
        os.symlink(self.w.record, self.w.tmp / "alias")
        code, o, err = run(["--record", str(self.w.tmp / "alias"), "--out", str(self.w.tmp / "o")])
        self.assertIn("the record directory is a symbolic link", err)
        self.assertFalse((self.w.tmp / "o").exists())

    def test_output_inside_record_or_results_package(self):
        for out, msg in ((self.w.record, "is the record directory or inside it"),
                         (self.w.public, "is the record directory or inside it"),
                         (self.w.public / "new", "inside the record directory"),
                         (self.w.record / "new", "inside the record directory"),
                         (self.w.root / "zz", "inside the package of the results"),
                         (self.w.root / "runs" / "zz", "inside the package of the results"),
                         (self.w.public / "sub" / "new", "parent of the output directory does not exist")):
            with self.subTest(out=out):
                before = snapshot(self.w.tmp)
                code, o, err = self.go(out)
                self.assertEqual(code, 1)
                self.assertIn(msg, err)
                self.assertEqual(snapshot(self.w.tmp), before)

    def test_output_inside_runs(self):
        for out in (self.w.tmp / "anet" / "zz", self.w.run_dirs()[KEY] / "zz", self.w.tmp / "c2" / "zz"):
            with self.subTest(out=out):
                code, o, err = self.go(out)
                self.assertEqual(code, 1)
                self.assertIn("inside a directory of the runs", err)
                self.assertFalse(out.exists())

    def test_output_inside_record_through_alias(self):
        os.symlink(self.w.record, self.w.tmp / "alias")
        code, o, err = self.go(self.w.tmp / "alias" / "new")
        self.assertEqual(code, 1)
        self.assertIn("inside the record directory", err)

    def test_output_inside_this_package(self):
        fake = self.w.tmp / "paper"
        (fake / "test_logs").mkdir(parents=True)
        with mock.patch.object(M, "HERE", fake), mock.patch.object(M, "DEFAULT_AGAINST", fake / "outputs"), \
                mock.patch.object(M, "build", return_value=self.files()):
            for out in (fake / "zz", fake / "test_logs" / "zz"):
                code, o, err = self.go(out)
                self.assertEqual(code, 1)
                self.assertIn("inside this package", err)
                self.assertFalse(out.exists())
            code, o, err = self.go(fake / "outputs")
            self.assertEqual(code, 0, o + err)
            self.assertEqual(sorted(os.listdir(fake / "outputs")), sorted(M.OUTPUTS))
            code, o, err = self.go(fake / "outputs" / "zz")
            self.assertIn("inside this package", err)

    def test_output_exists_or_symlink(self):
        (self.w.tmp / "empty").mkdir()
        self.refused("already exists", out_name="empty")
        os.symlink(self.w.tmp / "nowhere", self.w.tmp / "dangling")
        self.refused("already exists", out_name="dangling")
        (self.w.tmp / "afile").write_text("x")
        code, o, err = self.go(self.w.tmp / "afile" / "o")
        self.assertIn("parent of the output directory is not a directory", err)

    def test_options(self):
        o = str(self.w.tmp / "o")
        for args in (("--check", "--out", o), (), ("--out", o, "--text-only"), ("--out", o, "--against", o),
                     ("--bogus",), ("--out", o, "--out", o + "2"), ("--check", "--check"), ("--ch",),
                     ("--out=" + o, "--out", o + "2"), ("--out", o, "--record", "a", "--record=b")):
            with self.subTest(args=args):
                r = cli("--record", str(self.w.record), *args) if "--record" not in args else cli(*args)
                self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
                self.assertFalse((self.w.tmp / "o").exists())

    def test_help_in_english(self):
        r = cli("--help")
        self.assertEqual(r.returncode, 0)
        self.assertIn("Regenerate the table of the five networks", r.stdout)
        for w in ("Tabella", "rigenera", "soglia", "Materiali"):
            self.assertNotIn(w, r.stdout)

    def test_options_validated_before_reading_inputs(self):
        (self.w.tmp / "exists").mkdir()
        with mock.patch.object(M, "load_input_bytes", side_effect=AssertionError("inputs read")):
            code, o, err = self.go(self.w.tmp / "exists")
        self.assertEqual(code, 1)
        self.assertIn("already exists", err)

    def test_error_after_pass_in_same_folder(self):
        with mock.patch.object(M, "build", return_value=self.files()):
            code, o, err = self.go(self.w.tmp / "o")
        self.assertEqual(code, 0, o + err)
        before = snapshot(self.w.tmp / "o")
        p = self.w.path("results.json")
        p.write_bytes(p.read_bytes()[:-2])
        self.refused("public/results.json: SHA-256", out_name="o2")
        self.refused("already exists", out_name="o")
        self.assertEqual(snapshot(self.w.tmp / "o"), before)

    def test_failure_while_writing_leaves_nothing(self):
        real_open = os.open

        def failing(path, flags, *a, **kw):
            if str(path).endswith("descriptive.md") and flags & os.O_CREAT:
                raise OSError(28, "No space left on device")
            return real_open(path, flags, *a, **kw)
        with mock.patch.object(M, "build", return_value=self.files()), mock.patch.object(M.os, "open", failing):
            self.refused("could not write the outputs", out_name="o")
        self.assertNotIn(".mtf-", "".join(os.listdir(self.w.tmp)))

    def test_output_created_without_following_links(self):
        target = self.w.tmp / "target.txt"
        target.write_text("keep")
        planted = self.w.tmp / ".mtf-planted"
        planted.mkdir()
        os.symlink(target, planted / "aggregate.csv")
        with mock.patch.object(M, "build", return_value=self.files()), \
                mock.patch.object(M.tempfile, "mkdtemp", return_value=str(planted)):
            code, o, err = self.go(self.w.tmp / "o")
        self.assertEqual(code, 1)
        self.assertIn("could not write the outputs", err)
        self.assertEqual(target.read_text(), "keep")
        self.assertFalse((self.w.tmp / "o").exists())

    def test_two_runs_on_the_same_folder(self):
        def appear(kind, path):
            def side(data, images=True):
                if kind == "full":
                    path.mkdir()
                    (path / "aggregate.csv").write_text("other run")
                elif kind == "empty":
                    path.mkdir()
                elif kind == "file":
                    path.write_text("other run")
                else:
                    os.symlink(self.w.tmp / "elsewhere", path)
                return self.files()
            return side
        for kind in ("full", "empty", "file", "symlink"):
            with self.subTest(kind=kind):
                out = self.w.tmp / f"o_{kind}"
                with mock.patch.object(M, "build", side_effect=appear(kind, out)):
                    code, o, err = self.go(out)
                self.assertEqual(code, 1)
                self.assertIn("was taken by another process after the checks; nothing was written by this run", err)
                if kind == "empty":
                    self.assertEqual(os.listdir(out), [])
                self.assertNotIn(".mtf-", "".join(os.listdir(self.w.tmp)))
        real_rename = os.rename
        o3 = self.w.tmp / "o3"

        def rename_after_other_run(src, dst):
            (o3 / "aggregate.csv").write_text("other run")
            return real_rename(src, dst)
        with mock.patch.object(M, "build", return_value=self.files()), \
                mock.patch.object(M.os, "rename", rename_after_other_run):
            code, o, err = self.go(o3)
        self.assertEqual(code, 1)
        self.assertIn("was taken by another process after the checks", err)
        self.assertEqual(os.listdir(o3), ["aggregate.csv"])

    def test_reserved_name_replaced_by_a_file(self):
        real_rename = os.rename
        out = self.w.tmp / "o"

        def swap_then_rename(src, dst):
            out.rmdir()
            out.write_text("other run")
            return real_rename(src, dst)
        with mock.patch.object(M, "build", return_value=self.files()), \
                mock.patch.object(M.os, "rename", swap_then_rename):
            code, o, err = self.go(out)
        self.assertEqual(code, 1)
        self.assertIn("was taken by another process after the checks", err)
        self.assertEqual(out.read_text(), "other run")

    def test_write_next_to_output_mode_and_handlers(self):
        real_mkdtemp = tempfile.mkdtemp
        seen = []

        def spy(*a, **kw):
            seen.append(kw.get("dir"))
            return real_mkdtemp(*a, **kw)
        before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        out = self.w.tmp / "o"
        with mock.patch.object(M, "build", return_value=self.files()), mock.patch.object(M.tempfile, "mkdtemp", spy):
            code, o, err = self.go(out)
        self.assertEqual(code, 0, o + err)
        self.assertEqual([os.path.realpath(d) if d else None for d in seen], [os.path.realpath(self.w.tmp)])
        self.assertEqual(os.stat(out).st_mode & 0o777, 0o755)
        self.assertEqual({s: signal.getsignal(s) for s in before}, before)
        self.assertEqual(sorted(os.listdir(out)), sorted(M.OUTPUTS))

    def test_drawing_errors_keep_their_name(self):
        with mock.patch.object(M, "figure_images", side_effect=ValueError("drawing")):
            with self.assertRaises(ValueError):
                self.go(self.w.tmp / "o")
        self.assertFalse((self.w.tmp / "o").exists())

    def test_temporary_directory_replaced_before_the_rename(self):
        out = self.w.tmp / "o"
        real_poll = M._poll

        def poll(state):
            if state.get("reserved_id") is not None and "swapped" not in state:
                state["swapped"] = True
                shutil.rmtree(state["tmp"])
                os.mkdir(state["tmp"])
                Path(state["tmp"], "theirs.txt").write_text("other process")
            return real_poll(state)
        with mock.patch.object(M, "build", return_value=self.files()), mock.patch.object(M, "_poll", poll):
            code, o, err = self.go(out)
        self.assertEqual(code, 1)
        self.assertIn("changed during the rename; nothing was delivered by this run", err)
        self.assertNotIn("signal", err)
        self.assertEqual(os.listdir(out), ["theirs.txt"])

    def test_output_moved_away_after_the_rename(self):
        out, moved = self.w.tmp / "o", self.w.tmp / "moved"
        real_rename = os.rename

        def rename_then_move(src, dst):
            real_rename(src, dst)
            real_rename(dst, moved)
        with mock.patch.object(M, "build", return_value=self.files()), mock.patch.object(M.os, "rename",
                                                                                         rename_then_move):
            code, o, err = self.go(out)
        self.assertEqual(code, 1)
        self.assertIn("changed during the rename; nothing was delivered by this run", err)
        self.assertEqual(sorted(os.listdir(moved)), sorted(M.OUTPUTS))

    def test_temporary_directory_of_another_process_never_removed(self):
        out = self.w.tmp / "o"
        real_poll = M._poll

        def poll(state):
            if state.get("reserved_id") is not None and "swapped" not in state:
                state["swapped"] = True
                shutil.rmtree(state["tmp"])
                os.mkdir(state["tmp"])
                Path(state["tmp"], "theirs.txt").write_text("other process")
                (out / "x.txt").write_text("other process")
            return real_poll(state)
        with mock.patch.object(M, "build", return_value=self.files()), mock.patch.object(M, "_poll", poll):
            code, o, err = self.go(out)
        self.assertEqual(code, 1)
        tmps = [d for d in os.listdir(self.w.tmp) if d.startswith(".mtf-")]
        self.assertEqual(len(tmps), 1)
        self.assertEqual(os.listdir(self.w.tmp / tmps[0]), ["theirs.txt"])

    def test_cleanup_of_the_reservation_after_an_error(self):
        def denied(src, dst):
            raise PermissionError(13, "Permission denied")
        out = self.w.tmp / "o"
        with mock.patch.object(M, "build", return_value=self.files()), mock.patch.object(M.os, "rename", denied):
            code, o, err = self.go(out)
        self.assertEqual(code, 1)
        self.assertIn("could not write the outputs (Permission denied); nothing was left", err)
        self.assertFalse(out.exists())
        self.assertNotIn(".mtf-", "".join(os.listdir(self.w.tmp)))

    def test_reservation_replaced_by_another_empty_directory(self):
        out = self.w.tmp / "o"
        real_poll = M._poll
        theirs = {}

        def poll(state):
            if state.get("reserved_id") is not None and not theirs:
                out.rmdir()
                out.mkdir()
                theirs["id"] = M._identity(out)
            return real_poll(state)
        with mock.patch.object(M, "build", return_value=self.files()), mock.patch.object(M, "_poll", poll):
            code, o, err = self.go(out)
        self.assertEqual(code, 1)
        self.assertIn("was taken by another process after the checks", err)
        self.assertEqual(M._identity(out), theirs["id"])
        self.assertNotIn(".mtf-", "".join(os.listdir(self.w.tmp)))

    def test_without_matplotlib(self):
        out = self.w.tmp / "o"
        with mock.patch.dict(sys.modules, {"matplotlib": None}):
            code, o, err = self.go(out)
            self.assertEqual(code, 1)
            self.assertIn("matplotlib is not available", err)
            self.assertFalse(out.exists())


CHILD = r"""
import importlib.util, json, os, signal, sys, time
script, wargs, out, mode, point, pin, c2 = sys.argv[1:8]
spec = importlib.util.spec_from_file_location("m", script)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.PUBLIC_SUMS_SHA256 = pin
m.C2_RUNS = {int(k): tuple(v) for k, v in json.loads(c2).items()}
if mode == "fake":
    m.build = lambda data, images=True: {n: b"x" for n in m.OUTPUTS}
signal.signal(signal.SIGINT, signal.default_int_handler)
signal.signal(signal.SIGTERM, signal.SIG_DFL)
signal.signal(signal.SIGHUP, signal.SIG_DFL)
real_chmod, real_mkdir, real_rename, real_signal = os.chmod, os.mkdir, os.rename, signal.signal
state = {"installed": False, "restoring": False}
got = []


def ready():
    # si aspetta che il segnale sia registrato (non un tempo fisso): l'esito non dipende dal carico della macchina
    print("READY", "matplotlib" in sys.modules, flush=True)
    end = time.monotonic() + 60
    while not got and time.monotonic() < end:
        time.sleep(0.01)


def chmod(p, md, *a, **k):
    real_chmod(p, md, *a, **k)
    if point == "before_reservation" and os.path.basename(str(p)).startswith(".mtf-"):
        ready()


def mkdir(p, *a, **k):
    real_mkdir(p, *a, **k)
    if str(p) == out:
        print("reservation made", flush=True)
    if point == "after_mkdir" and str(p) == out:
        ready()


def rename(a, b):
    real_rename(a, b)
    if point == "after_rename" and str(b) == out:
        ready()


def sig(s, h):
    if getattr(h, "__name__", "") == "record":
        state["installed"] = True

        def wrapped(signum, frame, _h=h):
            _h(signum, frame)
            got.append(signum)
        wrapped.__name__ = "record"
        return real_signal(s, wrapped)
    if state["installed"] and point == "during_restore" and not state["restoring"]:
        state["restoring"] = True
        ready()
    r = real_signal(s, h)
    if state["installed"] and point == "after_restore" and s == signal.SIGHUP:
        ready()
    return r


os.chmod, os.mkdir, os.rename, signal.signal = chmod, mkdir, rename, sig
sys.exit(m.main(json.loads(wargs) + ["--out", out]))
"""


class S_Signals(Base):
    """Segnali mandati da fuori a un processo vero in cinque punti della scrittura, per SIGINT, SIGTERM e SIGHUP: con
    build finto (un solo thread, senza matplotlib) e con build vero (matplotlib importato, più thread)."""

    def one(self, mode, point, sig):
        out = self.w.tmp / f"o_{mode}_{point}_{int(sig)}"
        p = subprocess.Popen([sys.executable, "-I", "-B", "-c", CHILD, str(SCRIPT), json.dumps(self.w.args()), str(out),
                              mode, point, self.w.pin, json.dumps(self.w.c2)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             env=child_env())
        before = ""
        line = p.stdout.readline()
        while line and not line.startswith("READY"):
            before += line
            line = p.stdout.readline()
        if not line:
            so, se = p.communicate(timeout=120)
            self.fail(before + so + se)
        p.send_signal(sig)
        so, se = p.communicate(timeout=120)
        self.assertEqual(line.split()[1], "True" if mode == "real" else "False")
        return p.returncode, before + so, se, out

    def check(self, mode, point):
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            with self.subTest(sig=int(sig)):
                rc, so, se, out = self.one(mode, point, sig)
                if point in ("before_reservation", "after_mkdir"):
                    self.assertEqual(rc, 1, so + se)
                    self.assertIn(f"interrupted by signal {int(sig)}; nothing was left", se)
                    self.assertFalse(os.path.lexists(out))
                    if point == "before_reservation":
                        self.assertNotIn("reservation made", so)
                elif point == "after_restore" and sig != signal.SIGINT:
                    self.assertEqual(rc, -sig, so + se)
                    self.assertNotIn("nothing was left", so + se)
                    self.assertEqual(sorted(os.listdir(out)), sorted(M.OUTPUTS))
                else:
                    self.assertEqual(rc, 0, so + se)
                    self.assertIn(f"written 9 files to {out}; signal {int(sig)} received, handled after the outputs "
                                  f"were complete", so)
                    self.assertEqual(sorted(os.listdir(out)), sorted(M.OUTPUTS))
                    self.assertEqual(os.stat(out).st_mode & 0o777, 0o755)
                self.assertNotIn(".mtf-", "".join(os.listdir(self.w.tmp)))
                self.assertNotIn("Warning", se)
                if os.path.lexists(out):
                    shutil.rmtree(out)

    def test_fake_before_reservation(self):
        self.check("fake", "before_reservation")

    def test_fake_after_mkdir(self):
        self.check("fake", "after_mkdir")

    def test_fake_after_rename(self):
        self.check("fake", "after_rename")

    def test_fake_during_restore(self):
        self.check("fake", "during_restore")

    def test_fake_after_restore(self):
        self.check("fake", "after_restore")

    @need_mpl
    def test_real_before_reservation(self):
        self.check("real", "before_reservation")

    @need_mpl
    def test_real_after_rename(self):
        self.check("real", "after_rename")

    @need_mpl
    def test_real_after_restore(self):
        self.check("real", "after_restore")

    def test_sigkill_leaves_no_incomplete_output(self):
        rc, so, se, out = self.one("fake", "after_mkdir", signal.SIGKILL)
        self.assertEqual(rc, -signal.SIGKILL)
        self.assertTrue(not os.path.lexists(out) or os.listdir(out) == [])


# ---------------------------------------------------------------------------------------------------------------------
# E. --check

class E_Check(Base):
    def setUp(self):
        super().setUp()
        self.delivered = self.w.tmp / "delivered"
        files = self.w.build(images=HAS_MPL)
        if not HAS_MPL:
            files["figure_acer_variations.png"] = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x00IEND\xaeB`\x82"
            files["figure_acer_variations.pdf"] = b"%PDF-1.4\n%%EOF\n"
        self.delivered.mkdir()
        for n, b in files.items():
            (self.delivered / n).write_bytes(b)

    def check(self, against, *extra):
        return run([*self.w.args(), "--check", "--against", str(against), *extra], self.w.pin, self.w.c2)

    def test_text_only_passes(self):
        code, o, err = self.check(self.delivered, "--text-only")
        self.assertEqual(code, 0, o + err)
        self.assertEqual(o.count(": identical"), len(M.TEXT_OUTPUTS))
        self.assertEqual(o.count("structure valid"), 2)

    @need_mpl
    def test_full_check_passes(self):
        code, o, err = self.check(self.delivered)
        self.assertEqual(code, 0, o + err)
        self.assertEqual(o.count(": identical"), len(M.OUTPUTS))

    def mutate_each(self, names, extra=(), expect="DIFFERENT", where=None):
        for name in names:
            with self.subTest(name=name):
                d = Path(tempfile.mkdtemp(dir=self.w.tmp)) / "copy"
                shutil.copytree(self.delivered, d)
                p = d / name
                b = bytearray(p.read_bytes())
                b[where(len(b)) if where else len(b) // 2] ^= 0x01
                p.write_bytes(bytes(b))
                code, o, err = self.check(d, *extra)
                self.assertEqual(code, 1, o + err)
                self.assertIn(f"{name}: {expect}", o)
                shutil.rmtree(d)

    def test_one_byte_in_each_text_output(self):
        self.mutate_each(M.TEXT_OUTPUTS, ("--text-only",))

    @need_mpl
    def test_one_byte_in_each_image(self):
        self.mutate_each(M.IMAGE_OUTPUTS)

    def test_text_only_checks_image_structure(self):
        self.mutate_each(["figure_acer_variations.pdf"], ("--text-only",), "INVALID STRUCTURE", where=lambda n: 1)
        self.mutate_each(["figure_acer_variations.pdf"], ("--text-only",), "INVALID STRUCTURE", where=lambda n: n - 3)
        self.mutate_each(["figure_acer_variations.png"], ("--text-only",), "INVALID STRUCTURE", where=lambda n: n - 2)
        p = self.delivered / "figure_acer_variations.png"
        orig = p.read_bytes()
        p.write_bytes(orig + b"x")
        code, o, err = self.check(self.delivered, "--text-only")
        self.assertIn("figure_acer_variations.png: INVALID STRUCTURE", o)
        p.write_bytes(orig)
        self.assertFalse(M.png_structure_ok(b""))
        self.assertFalse(M.pdf_structure_ok(b"partial"))

    def test_against_refusals(self):
        (self.delivered / ".DS_Store").write_text("x")
        code, o, err = self.check(self.delivered, "--text-only")
        self.assertIn("expected exactly", err)
        (self.delivered / ".DS_Store").unlink()
        real = self.w.tmp / "real.csv"
        shutil.move(self.delivered / "single_class.csv", real)
        os.symlink(real, self.delivered / "single_class.csv")
        code, o, err = self.check(self.delivered, "--text-only")
        self.assertIn("delivered output single_class.csv is a symbolic link", err)
        code, o, err = self.check(self.w.public, "--text-only")
        self.assertIn("the directory to check is the record directory or inside it", err)
        code, o, err = self.check(self.w.tmp / "none", "--text-only")
        self.assertIn("the directory to check does not exist", err)

    def test_file_order_does_not_matter(self):
        real_listdir = os.listdir
        with mock.patch.object(M.os, "listdir", lambda p: list(reversed(sorted(real_listdir(p))))):
            code, o, err = self.check(self.delivered, "--text-only")
        self.assertEqual(code, 0, o + err)


# ---------------------------------------------------------------------------------------------------------------------
# G. indipendenza dallo stato della macchina, determinismo, nessun warning

class G_Determinism(Base):
    def test_two_text_generations(self):
        self.assertEqual(differing(self.w.build(), self.w.build()), [])

    def test_no_warnings(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.w.build(images=HAS_MPL)
        self.assertEqual([str(x.message) for x in caught], [])
        self.assertTrue(all("font cache" in m for m in MPL_LOG), MPL_LOG)

    @need_mpl
    def test_images_without_dates_or_software(self):
        a, b = self.w.build(images=True), self.w.build(images=True)
        self.assertEqual(differing(a, b), [])
        pdf, png = a["figure_acer_variations.pdf"], a["figure_acer_variations.png"]
        self.assertTrue(M.png_structure_ok(png) and M.pdf_structure_ok(pdf))
        for k in (b"CreationDate", b"ModDate", b"Producer", b"Creator", b"matplotlib"):
            self.assertNotIn(k, pdf)
        self.assertNotIn(b"Software", png)
        self.assertNotIn(b"tIME", png)

    def env_outputs(self, out, cwd, **env):
        r = cli_pin(self.w, *self.w.args(), "--out", str(out), cwd=cwd, env=child_env(**env))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stderr, "")
        return {n: (out / n).read_bytes() for n in M.OUTPUTS if (out / n).exists()}

    @need_mpl
    def test_machine_state_does_not_change_outputs(self):
        cases = [dict(), dict(LC_ALL="de_DE.UTF-8", LANG="tr_TR.UTF-8"), dict(TZ="Pacific/Kiritimati"),
                 dict(TZ="America/Adak", PYTHONHASHSEED="12345"), dict(HOME=str(self.w.tmp), USER="someone",
                                                                       LOGNAME="someone")]
        ref = None
        for i, env in enumerate(cases):
            with self.subTest(env=env):
                cwd = "/" if i % 2 else str(self.w.tmp)
                got = self.env_outputs(self.w.tmp / f"o{i}", cwd, **env)
                if ref is None:
                    ref = got
                self.assertEqual(differing(got, ref), [])

    @need_mpl
    def test_local_matplotlibrc_ignored(self):
        cwd = self.w.tmp / "cwd"
        cwd.mkdir()
        a = self.env_outputs(self.w.tmp / "a", str(cwd))
        (cwd / "matplotlibrc").write_text("axes.facecolor: red\nlines.linewidth: 5\nfont.size: 20\n")
        b = self.env_outputs(self.w.tmp / "b", str(cwd), MATPLOTLIBRC=str(cwd / "matplotlibrc"))
        self.assertEqual(differing(a, b), [])

    def test_midnight_and_clock(self):
        # nessuna uscita dipende dall'ora: con l'orologio spostato a mezzanotte i file di testo sono identici
        import time as _t
        ref = self.w.build()
        with mock.patch.object(_t, "time", return_value=1767225599.999), \
                mock.patch.object(_t, "localtime", return_value=_t.gmtime(1767225600)):
            self.assertEqual(differing(self.w.build(), ref), [])


# ---------------------------------------------------------------------------------------------------------------------
# R. il record reale, se presente

class R_RealRecord(unittest.TestCase):
    @need_real
    def test_pin_equals_sums_and_status(self):
        sums = (REAL / "public" / "SHA256SUMS").read_bytes()
        st = json.loads((REAL / "status.json").read_text())
        self.assertEqual(hashlib.sha256(sums).hexdigest(), M.PUBLIC_SUMS_SHA256)
        self.assertEqual(st["public_sha256sums_sha256"], M.PUBLIC_SUMS_SHA256)

    def real_outputs(self):
        return {k: v.decode() for k, v in M.build(M.load_input_bytes(*REAL_ARGS[1::2]), images=False).items()}

    @need_real
    def test_against_the_table_of_the_evaluator(self):
        md = self.real_outputs()["primary_table.md"]
        rows = list(csv.DictReader(io.StringIO((REAL / "public" / "table.csv").read_text())))
        self.assertEqual(len(rows), 10)
        for r in rows:
            line = [x for x in md.splitlines() if x.startswith(f"| {r['network']} | {r['bn_state']} |")]
            self.assertEqual(len(line), 1)
            want = (f"| {r['network']} | {r['bn_state']} | {pp(F(r['apcer_mean']))} | {pp(F(r['bpcer_mean']))} | "
                    f"{pp(F(r['acer_mean']))} | {r['units']} | {int(r['n_attack']):,} | {int(r['n_bona_fide']):,} | "
                    f"{r['n_subjects']} |")
            self.assertEqual(line[0], want)

    @need_real
    def test_against_the_report_of_the_evaluator(self):
        md = self.real_outputs()["primary_table.md"]
        report = (REAL / "public" / "report.md").read_text()
        n = 0
        for line in report.splitlines():
            m = re.match(r"\| (\S+) \| yes \| (-?[\d.]+) \| \[(-?[\d.]+), (-?[\d.]+)\] \|", line)
            if not m:
                continue
            name = m.group(1)
            # la tabella dei contrasti del valutatore ha sei colonne, quella delle variazioni quattro
            part = md.split("**(c)**")[1] if line.count("|") > 6 else md.split("**(b)**")[1].split("**(c)**")[0]
            acer = table_cells(part, name)[3]
            mine = re.match(r"([+−]?[\d.]+) \[([+−]?[\d.]+), ([+−]?[\d.]+)\]", acer).groups()
            norm = [Decimal(x.replace(U, "-").replace("+", "")) for x in mine]
            self.assertEqual(norm, [Decimal(m.group(i)) for i in (2, 3, 4)], name)
            n += 1
        self.assertEqual(n, 8)

    @need_real
    def test_numbers_of_readme_and_note(self):
        # ogni «media [low, high]» scritta a mano nel README e nella nota è una cella delle uscite generate
        out = " ".join(" ".join(self.real_outputs()[n].split()) for n in ("primary_table.md", "appendix_per_seed.md"))
        pat = re.compile(r"[+−]\d+\.\d\d \[[+−]\d+\.\d\d, [+−]\d+\.\d\d\]")
        n = 0
        for name in ("README.md", "NOTE.md"):
            for x in pat.findall(" ".join((HERE / name).read_text(encoding="utf-8").split())):
                self.assertIn(x, out, name)
                n += 1
        self.assertGreaterEqual(n, 16)

    @need_real
    def test_single_numbers_of_readme_and_note(self):
        # ogni numero a due decimali scritto a mano nel README e nella nota (anche senza intervallo) è una cella delle
        # uscite generate, a meno del «+»; l'unica eccezione dichiarata è la mediana delle 12 variazioni per unità di
        # LivenessNet, ricalcolata qui dal CSV aggregato
        out = self.real_outputs()
        num = re.compile(r"(?<![\d.])[+−-]?\d+\.\d\d(?![\d.])")     # il trattino ASCII vale come segno meno
        cells = {x.replace("+", "") for n in ("primary_table.md", "appendix_per_seed.md", "descriptive.md",
                                              "appendix_single_class.md") for x in num.findall(out[n])}
        rows = list(csv.DictReader(io.StringIO(out["aggregate.csv"], newline="")))
        lv = sorted(F(r["acer_exact"]) for r in rows if (r["patience"], r["threshold"], r["quantity"], r["network"],
                                                         r["scope"]) == ("4", "fixed_0.5", "variation", "LivenessNet",
                                                                         "unit"))
        self.assertEqual(len(lv), 12)
        cells.add(pp((lv[5] + lv[6]) / 2))
        n = 0
        for name in ("README.md", "NOTE.md"):
            for x in num.findall((HERE / name).read_text(encoding="utf-8")):
                self.assertIn(x.replace("+", "").replace("-", "−"), cells, f"{name}: {x}")
                n += 1
        self.assertGreaterEqual(n, 50)

    @need_real
    def test_delivered_outputs(self):
        if not DELIVERED.is_dir():
            self.fail("outputs/ is missing")
        # solo i file di testo byte per byte e la forma delle immagini: i byte di PNG e PDF dipendono dalle versioni di
        # matplotlib e FreeType (il confronto completo è nel log di --check, con le versioni dichiarate)
        code, o, err = run([*REAL_ARGS, "--check", "--text-only"])
        self.assertEqual(code, 0, o + err)
        self.assertEqual(o.count(": identical"), len(M.TEXT_OUTPUTS))
        self.assertEqual(o.count("structure valid"), 2)


# ---------------------------------------------------------------------------------------------------------------------
# H. file del pacchetto

class H_PublicFiles(unittest.TestCase):
    def test_no_identifiers_or_private_strings(self):
        # percorsi personali e indirizzi di posta
        bad = ["/Users/", "/home/", "/private/", "@"]
        if HAS_REAL:
            bad += list(json.loads((REAL / "public" / "results.json").read_text())["test_frame_ids_sha256"].values())
        files = [p for p in HERE.rglob("*") if p.is_file() and p.suffix in (".md", ".csv", ".tex", ".py", ".txt")
                 and p.name != "check_make_table_figure.py"]
        self.assertGreaterEqual(len(files), 3)
        for p in files:
            t = p.read_text(encoding="utf-8")
            for s in bad:
                with self.subTest(file=p.name, s=s):
                    self.assertNotIn(s, t)

    def test_scripts_are_not_executables(self):
        for p in HERE.glob("*.py"):
            self.assertFalse(p.read_text().startswith("#!"), p.name)
            self.assertFalse(os.stat(p).st_mode & 0o111, p.name)

    def test_script_never_names_restricted_files(self):
        code = [ln for ln in SCRIPT.read_text().splitlines() if not ln.lstrip().startswith("#")]
        body = "\n".join(code).split('"""', 2)[2]                 # senza la docstring del modulo
        self.assertNotIn("restricted", body)


class ZZ_Untouched(unittest.TestCase):
    def test_package_and_record_untouched(self):
        self.assertEqual(differing(snapshot(HERE, *REAL_ROOTS), BEFORE), [])

    def test_declared_skips_only(self):
        self.assertTrue(set(MISSING_INPUT) <= {"matplotlib", "record"})


def main():
    argv = [sys.argv[0]] + [a for i, a in enumerate(sys.argv[1:], start=1)
                            if a not in ("--verbose", "--record") and sys.argv[i - 1] != "--record"]
    if "--verbose" in sys.argv:
        argv.append("-v")
    try:
        result = unittest.main(argv=argv, exit=False, verbosity=1).result
    finally:
        shutil.rmtree(SUITE_TMP, ignore_errors=True)
    skipped = [f"{t.id().split('.', 1)[1]}: {r}" for t, r in result.skipped]
    print(f"tests run: {result.testsRun}, failures: {len(result.failures)}, errors: {len(result.errors)}, "
          f"skipped: {len(skipped)}")
    for s in skipped:
        print(f"skipped {s}")
    ok = result.wasSuccessful()
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
