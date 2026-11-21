"""
Test di make_table_figure.py (tabella, appendice per seed e figura dai file pubblici della PR #32), con la sola
libreria standard; matplotlib serve solo ai test che generano o confrontano le immagini.

Cosa provano:
  - i numeri contro le fonti: medie delle reti ricalcolate qui in modo indipendente dalle frazioni di units.csv,
    divari, contrasto, intervalli, ricampionamenti validi (926, 950, 940), conteggi (11/1/0, 9/3/0, 2 su 12) e i
    valori scritti dal coordinatore sulla C2 (16,06; 6,07; 9,98; 2,56; 16,32); la regola di arrotondamento;
  - l'impronta fissata nello script uguale al SHA-256 del SHA256SUMS pubblico e a public_sha256sums_sha256 di
    status.json;
  - l'integrità: un byte cambiato in ciascun input letto e nel SHA256SUMS ferma il comando;
  - una fixture avversaria per ogni controllo dichiarato, con SHA256SUMS riallineato e impronta sostituita (scatta la
    regola, non il checksum); per i conteggi, i pareggi e i modelli che decidono una sola classe un mondo coerente
    ricostruito qui (frazioni, medie, conteggi e righe ricalcolati da questo file, in modo indipendente);
  - le frasi generate costruite dai dati (ricampionamenti validi, cambio di segno, modelli a una sola classe, valori
    che si arrotondano a zero);
  - i percorsi: uscita uguale agli input, dentro gli input o il pacchetto dei risultati, dentro questo pacchetto (salvo
    outputs/), già esistente, collegamenti simbolici, file non regolari, genitore mancante, opzioni ripetute o
    incompatibili; nessuna cartella di uscita dopo un errore, a metà scrittura, dopo SIGTERM, SIGHUP e SIGKILL, con due
    esecuzioni sulla stessa cartella;
  - --check: PASS sui file consegnati, FAIL dopo un byte cambiato in ciascuno; --text-only controlla la forma delle
    immagini; un matplotlibrc locale non cambia la figura; due generazioni identiche byte per byte;
  - nessun identificativo né percorso privato nelle uscite e nei documenti; il pacchetto e gli input invariati, file e
    cartelle.
Unico salto ammesso: matplotlib assente (MISSING_INPUT), per i test che generano o confrontano le immagini.
TMPDIR deve stare fuori da questo pacchetto e dal pacchetto dei risultati (controllato prima di creare qualsiasi cosa).

  python3 -I -B check_make_table_figure.py [--verbose]
"""
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from unittest import mock

HERE = Path(os.path.abspath(__file__)).parent
SCRIPT = HERE / "make_table_figure.py"
RESULTS_PKG = HERE.parent / "pad-arch-eval-results"


def _inside(p, root):
    p, root = os.path.realpath(p), os.path.realpath(root)
    return p == root or p.startswith(root + os.sep)


_TMP = tempfile.gettempdir()
if _inside(_TMP, HERE) or _inside(_TMP, RESULTS_PKG):
    print(f"ERROR: TMPDIR ({_TMP}) is inside the package or the results package; set it elsewhere")
    print("RESULT: FAIL")
    sys.exit(1)

_spec = importlib.util.spec_from_file_location("make_table_figure", SCRIPT)
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

INPUTS = M.DEFAULT_INPUTS
STATUS = INPUTS.parent / "status.json"
COMMITTED = HERE / "outputs"
HAS_MPL = importlib.util.find_spec("matplotlib") is not None
MISSING_INPUT = {} if HAS_MPL else {
    "matplotlib": "matplotlib is not installed: the tests that generate or compare the figure images are skipped"}
need_mpl = unittest.skipUnless(HAS_MPL, "missing input: " + MISSING_INPUT.get("matplotlib", ""))
U = M.MINUS


F = Fraction


def flat(t):
    """Testo con gli a capo delle note spezzate ridotti a spazi."""
    return " ".join(t.split())


def fstr(f):
    return f"{f.numerator}/{f.denominator}"


def dict_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def snapshot(*roots):
    """File (dimensione, data, hash) e cartelle, esclusa test_logs/."""
    out = {}
    for root in roots:
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d != "test_logs"]
            out[dp] = ("dir",)
            for f in fns:
                p = os.path.join(dp, f)
                st = os.lstat(p)
                if not os.path.isfile(p) or os.path.islink(p):
                    out[p] = ("special", st.st_mode)
                    continue
                out[p] = (st.st_size, st.st_mtime_ns, hashlib.sha256(Path(p).read_bytes()).hexdigest())
    return out


BEFORE = snapshot(HERE, RESULTS_PKG)


def run(argv, pin=None):
    """main() nello stesso processo; con pin, l'impronta fissata sostituita (fixture riallineate)."""
    out, err = io.StringIO(), io.StringIO()
    ctx = mock.patch.object(M, "PUBLIC_SUMS_SHA256", pin) if pin else contextlib.nullcontext()
    with ctx, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = M.main(argv)
    return code, out.getvalue(), err.getvalue()


def cli(*args, **kw):
    return subprocess.run([sys.executable, "-I", "-B", str(SCRIPT), *args], capture_output=True, text=True, **kw)


class World:
    """Copia della cartella pubblica in tmp/results/runs/<RUN_ID>/public, con SHA256SUMS riallineabile."""

    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mtf-"))
        self.root = self.tmp / "results"
        self.inputs = self.root / "runs" / M.RUN_ID / "public"
        shutil.copytree(INPUTS, self.inputs, symlinks=True)

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def path(self, name):
        return self.inputs / name

    def realign(self):
        lines = []
        for line in self.path("SHA256SUMS").read_text().splitlines():
            name = line.split("  ", 1)[1]
            lines.append(f"{hashlib.sha256(self.path(name).read_bytes()).hexdigest()}  {name}")
        b = ("\n".join(lines) + "\n").encode()
        self.path("SHA256SUMS").write_bytes(b)
        return hashlib.sha256(b).hexdigest()

    def results(self):
        return json.loads(self.path("results.json").read_text())

    def edit_json(self, fn):
        d = self.results()
        fn(d)
        self.path("results.json").write_text(json.dumps(d, indent=1))

    def edit_csv(self, name, fn):
        rows = list(csv.reader(io.StringIO(self.path(name).read_text(), newline="")))
        rows = fn(rows)
        buf = io.StringIO(newline="")
        csv.writer(buf, lineterminator="\n").writerows(rows)
        self.path(name).write_text(buf.getvalue())

    def edit_units(self, fn):
        self.edit_csv("units.csv", lambda rows: [rows[0]] + fn(rows[0], rows[1:]))

    def edit_conditions(self, state, scope, field, value):
        def fn(rows):
            h = rows[0]
            hit = 0
            for r in rows[1:]:
                d = dict(zip(h, r))
                if (d["patience"], d["threshold"], d["label"], d["bn_state"], d["scope"]) == \
                        ("4", "fixed_0.5", "primary", state, scope):
                    r[h.index(field)] = value
                    hit += 1
            assert hit == 1
            return rows
        self.edit_csv("conditions.csv", fn)

    def build(self, pin, images=False):
        with mock.patch.object(M, "PUBLIC_SUMS_SHA256", pin):
            return M.build(M.load_input_bytes(str(self.inputs)), images=images)

    def consistent(self, changes):
        """Cambia i tassi di alcune unità primarie e ricalcola qui, in modo indipendente dallo script, tutto ciò che ne
        dipende in units.csv, results.json e conditions.csv. changes: (stato, fold, seed, rete, APCER, BPCER)."""
        def fix(h, rows):
            for r in rows:
                d = dict(zip(h, r))
                for st, fo, se, net, a, b in changes:
                    if (d["patience"], d["threshold"], d["bn_state"], d["fold"], d["seed"]) == \
                            ("4", "fixed_0.5", st, str(fo), str(se)):
                        r[h.index(f"{net}_apcer")], r[h.index(f"{net}_bpcer")] = fstr(F(a)), fstr(F(b))
                        r[h.index(f"{net}_acer")] = fstr((F(a) + F(b)) / 2)
                d = dict(zip(h, r))
                for m in ("apcer", "bpcer", "acer"):
                    r[h.index(f"delta_{m}")] = fstr(F(d[f"resnet18_{m}"]) - F(d[f"livenessnet_{m}"]))
                dacer = F(r[h.index("delta_acer")])
                r[h.index("delta_acer_value")] = repr(float(dacer))
                r[h.index("tie")] = "1" if dacer == 0 else "0"
            return rows
        self.edit_units(fix)
        un = {s: {} for s in ("ordinary", "re-estimated")}
        for d in dict_rows(self.path("units.csv")):
            if (d["patience"], d["threshold"]) == ("4", "fixed_0.5"):
                un[d["bn_state"]][(int(d["fold"]), int(d["seed"]))] = {
                    m: F(d[f"resnet18_{m}"]) - F(d[f"livenessnet_{m}"]) for m in ("apcer", "bpcer", "acer")}
        res = self.results()
        pa = res["primary_analysis"]
        for row in pa["rows"]:
            k = (row["unit"][1], row["unit"][2])
            for blk, val in (("delta_ordinary", un["ordinary"][k]), ("delta_re_estimated", un["re-estimated"][k]),
                             ("contrast", {m: un["re-estimated"][k][m] - un["ordinary"][k][m]
                                           for m in ("apcer", "bpcer", "acer")})):
                for m, v in val.items():
                    row[blk][m] = {"exact": fstr(v), "value": float(v)}
        cond = {}
        for scope in ("all seeds", "seed 42", "seed 43", "seed 44"):
            keys = [(f, s) for f in (1, 2, 3, 4) for s in (42, 43, 44) if scope == "all seeds" or s == int(scope[5:])]
            base = pa if scope == "all seeds" else pa["per_seed"][scope[5:]]
            for st, key in (("ordinary", "gap_ordinary"), ("re-estimated", "gap_re_estimated")):
                vals = [un[st][k] for k in keys]
                for m in ("apcer", "bpcer", "acer"):
                    base[key]["mean_delta"][m] = float(sum((v[m] for v in vals), F(0)) / len(keys))
                ac = [v["acer"] for v in vals]
                base[key]["acer_counts"].update(resnet18_lower=sum(x < 0 for x in ac),
                                                livenessnet_lower=sum(x > 0 for x in ac),
                                                ties=sum(x == 0 for x in ac))
                cond[(st, scope)] = base[key]
            for m in ("apcer", "bpcer", "acer"):
                base["contrast"]["mean"][m] = float(sum((un["re-estimated"][k][m] - un["ordinary"][k][m]
                                                         for k in keys), F(0)) / len(keys))
            base["contrast"]["units_changing_sign"] = sum(
                1 for k in keys if (un["ordinary"][k]["acer"] < 0 < un["re-estimated"][k]["acer"])
                or (un["ordinary"][k]["acer"] > 0 > un["re-estimated"][k]["acer"]))
        # le medie nuove possono uscire dagli intervalli copiati: si allargano gli intervalli e si rifanno le etichette
        # con le regole di PROTOCOL.md §6, scritte qui
        for scope in ("all seeds", "seed 42", "seed 43", "seed 44"):
            base = pa if scope == "all seeds" else pa["per_seed"][scope[5:]]
            for key in ("gap_ordinary", "gap_re_estimated", "contrast"):
                blk = base[key]
                means = blk["mean_delta"] if key != "contrast" else blk["mean"]
                for m in ("apcer", "bpcer", "acer"):
                    iv = blk["interval"][m]
                    iv["low"], iv["high"] = min(iv["low"], means[m]), max(iv["high"], means[m])
                lo, hi = blk["interval"]["acer"]["low"], blk["interval"]["acer"]["high"]
                where = "below" if hi < 0 else "above" if lo > 0 else None
                if key != "contrast":
                    blk["sign"] = {"below": "ResNet-18 lower", "above": "LivenessNet lower"}.get(where, "unresolved")
                    blk["sign_reason"] = f"interval entirely {where} zero" if where else "interval contains zero"
                else:
                    blk["reading"] = (f"ACER difference (ResNet-18 - LivenessNet) "
                                      f"{'lower' if where == 'below' else 'higher'} with re-estimated BN"
                                      if where else "unresolved")
                    blk["reason"] = f"interval entirely {where} zero" if where else "interval contains zero"
        self.path("results.json").write_text(json.dumps(res, indent=1))
        for (st, scope), blk in cond.items():
            for m in ("apcer", "bpcer", "acer"):
                self.edit_conditions(st, scope, f"mean_delta_{m}", repr(blk["mean_delta"][m]))
                self.edit_conditions(st, scope, f"low_{m}", repr(blk["interval"][m]["low"]))
                self.edit_conditions(st, scope, f"high_{m}", repr(blk["interval"][m]["high"]))
            for k in ("resnet18_lower", "livenessnet_lower", "ties"):
                self.edit_conditions(st, scope, k, str(blk["acer_counts"][k]))
            self.edit_conditions(st, scope, "sign", blk["sign"])
            self.edit_conditions(st, scope, "sign_reason", blk["sign_reason"])
        return self.realign()


def primary(h, r, state=None, fold=None, seed=None):
    d = dict(zip(h, r))
    return (d["patience"], d["threshold"]) == ("4", "fixed_0.5") and (state is None or d["bn_state"] == state) \
        and (fold is None or d["fold"] == str(fold)) and (seed is None or d["seed"] == str(seed))


class Base(unittest.TestCase):
    def setUp(self):
        self.w = World()
        self.addCleanup(self.w.close)

    def refused(self, msg, pin=None, out_name="out", extra=()):
        """Il comando si ferma con il messaggio atteso e non lascia nulla (nessun file né cartella nuovi)."""
        before = snapshot(self.w.tmp)
        code, out, err = run(["--inputs", str(self.w.inputs), "--out", str(self.w.tmp / out_name), *extra], pin)
        self.assertEqual(code, 1, out + err)
        self.assertIn(msg, err)
        self.assertIn("RESULT: FAIL", out)
        self.assertNotIn("Traceback", err)
        self.assertEqual(snapshot(self.w.tmp), before)
        return err


# ---------------------------------------------------------------------------------------------------------------------

class A_Numbers(unittest.TestCase):
    """I numeri delle uscite consegnate contro le fonti."""

    @classmethod
    def setUpClass(cls):
        cls.rows = dict_rows(COMMITTED / "primary_table.csv")
        cls.seed_rows = dict_rows(COMMITTED / "per_seed.csv")
        cls.res = json.loads((INPUTS / "results.json").read_text())
        cls.units = [r for r in dict_rows(INPUTS / "units.csv")
                     if r["patience"] == "4" and r["threshold"] == "fixed_0.5"]

    def row(self, name, net=None, state=None):
        hit = [r for r in self.rows if r["row"] == name and (net is None or r["network"] == net)
               and (state is None or r["bn_state"] == state)]
        self.assertEqual(len(hit), 1)
        return hit[0]

    def test_network_means_recomputed_here(self):
        for state in ("ordinary", "re-estimated"):
            us = [u for u in self.units if u["bn_state"] == state]
            self.assertEqual(len(us), 12)
            for net, label in (("livenessnet", "LivenessNet"), ("resnet18", "ResNet-18")):
                r = self.row("network", label, state)
                for m in ("apcer", "bpcer", "acer"):
                    exact = sum((F(u[f"{net}_{m}"]) for u in us), F(0)) / 12
                    self.assertEqual(F(r[f"{m}_exact"]), exact)
                    self.assertEqual(float(r[m]), float(exact))

    def test_gaps_and_contrast_equal_results_json(self):
        pa = self.res["primary_analysis"]
        for name in ("gap_ordinary", "gap_re_estimated", "contrast"):
            r = self.row(name)
            blk = pa[name]
            means = blk["mean_delta"] if name != "contrast" else blk["mean"]
            for m in ("apcer", "bpcer", "acer"):
                self.assertEqual(r[m], repr(means[m]))
                self.assertEqual(r[f"{m}_low"], repr(blk["interval"][m]["low"]))
                self.assertEqual(r[f"{m}_high"], repr(blk["interval"][m]["high"]))
                self.assertEqual(r[f"{m}_valid_resamples"], {"acer": "926", "apcer": "950", "bpcer": "940"}[m])

    def test_gap_is_difference_of_network_means(self):
        for state, name in (("ordinary", "gap_ordinary"), ("re-estimated", "gap_re_estimated")):
            for m in ("apcer", "bpcer", "acer"):
                d = F(self.row("network", "ResNet-18", state)[f"{m}_exact"]) \
                    - F(self.row("network", "LivenessNet", state)[f"{m}_exact"])
                self.assertEqual(F(self.row(name)[f"{m}_exact"]), d)

    def test_counts_and_labels(self):
        o, e, c = self.row("gap_ordinary"), self.row("gap_re_estimated"), self.row("contrast")
        self.assertEqual((o["resnet18_lower"], o["livenessnet_lower"], o["ties"]), ("11", "1", "0"))
        self.assertEqual((e["resnet18_lower"], e["livenessnet_lower"], e["ties"]), ("9", "3", "0"))
        self.assertEqual(c["units_changing_sign"], "2")
        self.assertEqual(o["label"], "ResNet-18 lower")
        self.assertEqual((e["label"], e["label_reason"]), ("unresolved", "interval contains zero"))
        self.assertEqual(c["label"], "ACER difference (ResNet-18 - LivenessNet) higher with re-estimated BN")

    def test_values_written_on_the_card_in_points(self):
        md = (COMMITTED / "primary_table.md").read_text()
        for s in (f"{U}16.06 [{U}20.79, {U}7.41]", f"{U}6.07 [{U}12.53, +4.76]", "+9.98 [+2.56, +16.32]",
                  "| LivenessNet | ordinary | 15.74 | 58.95 | 37.35 |",
                  "| ResNet-18 | re-estimated | 36.72 | 11.69 | 24.20 |"):
            self.assertIn(s, md)
        self.assertEqual(M.fmt(-0.16057264797151244), "-16.06")
        self.assertEqual(M.fmt(-0.060739393286358835), "-6.07")
        self.assertEqual(M.fmt(0.0998332546851536, True), "+9.98")
        self.assertEqual(M.fmt(0.02561418818588727), "2.56")
        self.assertEqual(M.fmt(0.1632356240297182), "16.32")

    def test_per_seed_equal_results_json(self):
        pa = self.res["primary_analysis"]["per_seed"]
        for seed in ("42", "43", "44"):
            for name in ("gap_ordinary", "gap_re_estimated", "contrast"):
                r = [x for x in self.seed_rows if x["scope"] == f"seed {seed}" and x["row"] == name]
                self.assertEqual(len(r), 1)
                blk = pa[seed][name]
                means = blk["mean_delta"] if name != "contrast" else blk["mean"]
                self.assertEqual(r[0]["acer"], repr(means["acer"]))
                self.assertEqual(r[0]["acer_low"], repr(blk["interval"]["acer"]["low"]))
                self.assertEqual(r[0]["acer_valid_resamples"], "926")
        md = (COMMITTED / "per_seed.md").read_text()
        self.assertIn("Reported apart; not part of the reading", md)
        self.assertIn("rule of `PROTOCOL.md` §6 applied to the seed; not part of the reading", md)
        self.assertNotIn("same resamples", md)
        self.assertEqual([x["units_changing_sign"] for x in self.seed_rows if x["row"] == "contrast"], ["1", "0", "1"])

    def test_rounding_and_typography(self):
        self.assertEqual(M.fmt(F(1, 800)), "0.13")
        self.assertEqual(M.fmt(F(-1, 800)), "-0.13")
        self.assertEqual(M.fmt(0.00125), "0.13")
        self.assertEqual(M.fmt(0.0012499999999), "0.12")
        self.assertEqual(M.fmt(F(1, 3)), "33.33")
        self.assertEqual(M.fmt(-3.698851188355394e-05), "-0.00")
        self.assertEqual(M.fmt_u(-3.698851188355394e-05), U + "0.00")
        self.assertEqual(M.fmt(0.0, True), "0.00")
        self.assertEqual(M.points(F(123456789, 1000000000)), Decimal("12.35"))
        for name in ("primary_table.md", "per_seed.md"):
            t = (COMMITTED / name).read_text()
            self.assertNotRegex(t, r"(?<![A-Za-z0-9])-[0-9]")     # numeri negativi solo con U+2212
        for name in ("primary_table.csv", "per_seed.csv", "figure_acer_gaps_data.csv"):
            self.assertNotIn(U, (COMMITTED / name).read_text())
        nz = flat((COMMITTED / "per_seed.md").read_text())
        self.assertIn(f"lower bound of the BPCER contrast of seed 44: {U}0.0037 points", nz)
        self.assertNotIn("rounds to zero", (COMMITTED / "primary_table.md").read_text())

    def test_sign_rules_bound_at_zero(self):
        self.assertEqual(M.sign_rule(-0.1, 0.0), M.SIGN_ZERO)
        self.assertEqual(M.sign_rule(0.0, 0.1), M.SIGN_ZERO)
        self.assertEqual(M.sign_rule(-0.2, -1e-300), M.SIGN_BELOW)
        self.assertEqual(M.sign_rule(1e-300, 0.2), M.SIGN_ABOVE)
        self.assertEqual(M.contrast_rule(0.0, 0.1), M.CONTRAST_ZERO)
        self.assertEqual(M.contrast_rule(-0.1, 0.0), M.CONTRAST_ZERO)
        self.assertEqual(M.contrast_rule(1e-300, 0.1), M.CONTRAST_ABOVE)
        self.assertEqual(M.contrast_rule(-0.1, -1e-300), M.CONTRAST_BELOW)
        self.assertFalse(M.changes_sign(F(-1, 2), F(0)))
        self.assertFalse(M.changes_sign(F(0), F(1, 2)))
        self.assertTrue(M.changes_sign(F(-1, 2), F(1, 3)))
        self.assertTrue(M.changes_sign(F(1, 2), F(-1, 3)))
        self.assertFalse(M.sign_change_between_states("ResNet-18 lower", "unresolved"))
        self.assertTrue(M.sign_change_between_states("ResNet-18 lower", "LivenessNet lower"))
        self.assertFalse(M.sign_change_between_states("unresolved", "unresolved"))

    def test_generated_notes(self):
        md = flat((COMMITTED / "primary_table.md").read_text())
        self.assertIn("LivenessNet fold 1 seed 42 with ordinary BN, every frame attack; LivenessNet fold 4 seed 44 "
                      "with ordinary BN, every frame attack", md)
        self.assertIn("LivenessNet with ordinary BN 2 of 12", md)
        self.assertIn("valid resamples 926 for ACER, 950 for APCER, 940 for BPCER, in every block", md)
        self.assertIn("with labels \"ResNet-18 lower\" (ordinary BN) and \"unresolved\" (re-estimated BN), there is "
                      "none", md)
        self.assertNotIn("one-class models", md.lower())

    def test_figure_data(self):
        rows = dict_rows(COMMITTED / "figure_acer_gaps_data.csv")
        self.assertEqual(len(rows), 3 * (3 + 12))
        for item in ("gap_ordinary", "gap_re_estimated", "contrast"):
            units = [F(r["value"]) for r in rows if r["item"] == item and r["kind"] == "unit"]
            mean = [float(r["value"]) for r in rows if r["item"] == item and r["kind"] == "mean"][0]
            self.assertEqual(len(units), 12)
            self.assertAlmostEqual(float(sum(units) / 12), mean, places=15)
            pts = [(float(r["value_points"]), int(r["level"])) for r in rows if r["item"] == item
                   and r["kind"] == "unit"]
            for i, (x, lv) in enumerate(pts):        # nessuna coppia sullo stesso livello a meno di 2,0 punti
                for x2, lv2 in pts[i + 1:]:
                    if lv == lv2:
                        self.assertGreaterEqual(abs(x - x2), M.LEVEL_GAP_POINTS)
        contrast = {(r["fold"], r["seed"]): r["value_points"] for r in rows if r["item"] == "contrast"
                    and r["kind"] == "unit"}
        self.assertEqual(contrast[("3", "44")], "+54.09")
        self.assertEqual(M.unit_levels([(0, 1), (1, 2), (5, 3), (1.5, 4)]), [0, 1, 0, 2])

    def test_note(self):
        note = (HERE / "NOTE.md").read_text()
        for s in ("16 subjects, all from NUAA", "926", f"{U}6.07 [{U}12.53, +4.76]", "+9.98 [+2.56, +16.32]",
                  "PR #21", "PR #28", "`history_check`", "Figure 1."):
            self.assertIn(s, note)
        self.assertNotIn("advantage", note)
        if "## Addition of the author" in note:
            add = note.split("## Addition of the author", 1)[1]
            rows = dict_rows(COMMITTED / "figure_acer_gaps_data.csv")
            c = {(r["fold"], r["seed"]): F(r["value"]) for r in rows if r["item"] == "contrast"
                 and r["kind"] == "unit"}
            top = max(c, key=c.get)
            rest = [v for k, v in c.items() if k != top]
            for s in (M.fmt_u(min(c.values()), True), M.fmt_u(max(c.values()), True), f"fold {top[0]}, seed {top[1]}",
                      M.fmt_u(sum(rest, F(0)) / 11, True)):
                self.assertIn(s, add)
            self.assertNotIn("+54.09", note.split("## Addition of the author", 1)[0])


class B_Pin(unittest.TestCase):
    def test_pin_equals_file_and_status(self):
        b = (INPUTS / "SHA256SUMS").read_bytes()
        self.assertEqual(hashlib.sha256(b).hexdigest(), M.PUBLIC_SUMS_SHA256)
        st = json.loads(STATUS.read_text())
        self.assertEqual(st["public_sha256sums_sha256"], M.PUBLIC_SUMS_SHA256)
        self.assertEqual(st["run_id"], M.RUN_ID)


class C_Integrity(Base):
    def flip(self, name):
        p = self.w.path(name)
        b = bytearray(p.read_bytes())
        b[len(b) // 2] ^= 0x01
        p.write_bytes(bytes(b))

    def test_one_byte_in_each_read_input(self):
        for name in M.READ_FILES:
            with self.subTest(name=name):
                self.setUp()
                self.flip(name)
                self.refused(f"{name}: SHA-256")

    def test_one_byte_in_sums(self):
        self.flip("SHA256SUMS")
        self.refused("expected the pinned")

    def test_realigned_sums_without_new_pin(self):
        self.flip("units.csv")
        self.w.realign()
        self.refused("expected the pinned")

    def sums_case(self, text, msg):
        p = self.w.path("SHA256SUMS")
        p.write_bytes(text)
        self.refused(msg, pin=hashlib.sha256(text).hexdigest())

    def test_sums_rules(self):
        orig = self.w.path("SHA256SUMS").read_bytes()
        self.sums_case(orig + b"0" * 64 + b"  extra.csv\n", "lists")
        self.sums_case(b"".join(orig.splitlines(True)[:-1]), "lists")
        self.sums_case(orig.replace(b"  units.csv", b" units.csv"), "malformed line")
        self.sums_case(orig[:-1], "does not end with a newline")
        self.sums_case(orig + [ln for ln in orig.splitlines(True) if b"units.csv" in ln][0], "lists units.csv twice")
        self.sums_case(orig + "é".encode(), "is not ASCII")


class D_Hostile(Base):
    """Fixture riallineate: SHA256SUMS riscritto e impronta sostituita, quindi deve scattare la regola."""

    def json_case(self, fn, msg, conditions=()):
        self.w.edit_json(fn)
        for args in conditions:
            self.w.edit_conditions(*args)
        return self.refused(msg, pin=self.w.realign())

    def fresh(self):
        self.setUp()

    def test_run_status_schema_official(self):
        for key, val in (("run_id", "20261121-ARCHEVAL-0000000"), ("status", "FAIL"), ("schema", "pad-arch-eval/2"),
                         ("official", False)):
            with self.subTest(key=key):
                self.fresh()
                self.json_case(lambda d: d.__setitem__(key, val), f"results.json: {key} is")

    def test_json_parsing(self):
        p = self.w.path("results.json")
        t = p.read_text()
        p.write_text(t.replace('"status": "PASS",', '"status": "PASS", "status": "PASS",', 1))
        self.refused("duplicate key", pin=self.w.realign())
        self.fresh()
        self.w.path("results.json").write_text(t.replace("-0.16057264797151244", "NaN", 1))
        self.refused("non-finite", pin=self.w.realign())
        self.fresh()
        self.w.path("results.json").write_text(t[:-10])
        self.refused("is not valid JSON", pin=self.w.realign())
        self.fresh()
        self.w.path("results.json").write_bytes(t.encode().replace(b'"PASS"', b'"PA\xffSS"', 1))
        self.refused("is not valid JSON", pin=self.w.realign())
        self.fresh()
        self.w.path("results.json").write_text("[]")
        self.refused("the top level is not an object", pin=self.w.realign())

    def test_malformed_structure_fails_cleanly(self):
        self.json_case(lambda d: d.__setitem__("checks_before_primary_analysis", []), "unexpected structure")
        self.fresh()
        self.json_case(lambda d: d["primary_analysis"]["rows"][0].__setitem__("contrast", []), "missing field")
        self.fresh()
        self.json_case(lambda d: d["primary_analysis"]["rows"].__setitem__(0, 7), "is not a unit of the plan")

    def test_condition_patience_or_threshold(self):
        self.json_case(lambda d: d["primary_analysis"]["conditions"].__setitem__(
            "ordinary", [12, "ordinary", "fixed_0.5"]), "primary_analysis.conditions")
        self.fresh()
        self.json_case(lambda d: d["primary_analysis"]["conditions"]["re-estimated"].__setitem__(2, "dev_eer"),
                       "primary_analysis.conditions")

    def test_effective_threshold_each_network(self):
        for state in ("ordinary", "re-estimated"):
            for net in ("LivenessNet", "ResNet-18"):
                with self.subTest(state=state, net=net):
                    self.fresh()
                    key = f"[4, '{state}', 'fixed_0.5']"
                    self.json_case(lambda d: d["checks_before_primary_analysis"]["units"][key]["effective_threshold"]
                                   .__setitem__(net, [0.49]), "effective threshold 0.5 for both networks")

    def test_units_and_plan_checks(self):
        key = "[4, 're-estimated', 'fixed_0.5']"
        self.json_case(lambda d: d["checks_before_primary_analysis"]["units"][key].__setitem__("units", 11),
                       "is not 12 units")
        self.fresh()
        self.json_case(lambda d: d["checks_before_primary_analysis"]["plan"][key].__setitem__("threshold", 0.4),
                       "is not 12 combinations")
        self.fresh()
        self.json_case(lambda d: d["checks_before_primary_analysis"]["units"][key].__setitem__("units", 12.0),
                       "is not 12 units")
        self.fresh()
        self.json_case(lambda d: d["checks_before_primary_analysis"]["plan"].pop(key), "plan has no entry")

    def test_frames_and_ids_in_results(self):
        self.json_case(lambda d: d["test_frames_per_fold"].__setitem__(0, 0), "not 4 positive integers")
        self.fresh()
        self.json_case(lambda d: d["test_frame_ids_sha256"].__setitem__("2", "x"), "one SHA-256 per fold")

    def test_units_csv_missing_duplicate_count(self):
        def drop(h, rows):
            hit = [i for i, r in enumerate(rows) if primary(h, r, "re-estimated", 3, 43)]
            del rows[hit[0]]
            return rows
        self.w.edit_units(drop)
        self.refused("does not hold the 12 fold-seed units", pin=self.w.realign())
        self.fresh()
        self.w.edit_units(lambda h, rows: rows + [rows[0]])
        self.refused("appears twice", pin=self.w.realign())
        self.fresh()
        self.w.edit_units(lambda h, rows: [r for r in rows if not (r[0] == "12" and r[3] == "4" and r[4] == "44"
                                                                   and r[2] == "dev_eer" and r[1] == "ordinary")])
        self.refused("95 rows, expected 96", pin=self.w.realign())

    def unit_field(self, field, value, msg, state="ordinary", fold=2, seed=43, patience="4", threshold="fixed_0.5"):
        def fn(h, rows):
            for r in rows:
                d = dict(zip(h, r))
                if (d["patience"], d["threshold"], d["bn_state"], d["fold"], d["seed"]) == \
                        (patience, threshold, state, str(fold), str(seed)):
                    r[h.index(field)] = value if not callable(value) else value(r[h.index(field)])
            return rows
        self.w.edit_units(fn)
        return self.refused(msg, pin=self.w.realign())

    def test_units_csv_thresholds(self):
        self.unit_field("resnet18_threshold", "0.49", "effective threshold of resnet18")
        self.fresh()
        self.unit_field("livenessnet_threshold", "0.5000001", "effective threshold of livenessnet", "re-estimated")
        self.fresh()
        self.unit_field("livenessnet_threshold", "nan", "threshold of livenessnet is not a number",
                        patience="12", threshold="dev_eer")
        self.fresh()
        self.unit_field("resnet18_threshold", "1.5", "threshold of resnet18 is not a number", threshold="dev_eer")

    def test_units_csv_fractions(self):
        self.unit_field("resnet18_acer", "1/3", "resnet18_acer is not the mean")
        self.fresh()
        self.unit_field("delta_bpcer", "1/3", "delta_bpcer is not ResNet-18 minus LivenessNet")
        self.fresh()
        self.unit_field("livenessnet_apcer", lambda v: "/".join(str(2 * int(x)) for x in v.split("/")),
                        "is not written reduced")
        self.fresh()
        self.unit_field("livenessnet_apcer", lambda v: v + "x", "is not an exact fraction")
        self.fresh()
        self.unit_field("livenessnet_bpcer", "3/2", "livenessnet_bpcer outside [0, 1]")
        self.fresh()
        self.unit_field("delta_acer_value", lambda v: repr(float(v) + 1e-12), "delta_acer_value")
        self.fresh()
        self.unit_field("tie", "1", "tie does not match")
        self.fresh()
        self.unit_field("fold", "5", "fold or seed outside the plan")
        self.fresh()
        self.unit_field("patience", "7", "condition outside the plan")
        self.fresh()
        # una riga non primaria (patience 12) è validata come le altre
        self.unit_field("resnet18_bpcer", "1/3", "resnet18_acer is not the mean", patience="12")

    def test_units_csv_frames(self):
        self.unit_field("n_frames", "3115", "n_frames")
        self.fresh()
        self.unit_field("n_frames", "-1", "n_frames", patience="12")
        self.fresh()
        self.unit_field("frame_ids_sha256", "0" * 64, "frame_ids_sha256")

    def test_csv_format(self):
        self.w.edit_csv("units.csv", lambda rows: [[c.upper() if c == "tie" else c for c in rows[0]]] + rows[1:])
        self.refused("units.csv: unexpected header", pin=self.w.realign())
        self.fresh()
        self.w.edit_csv("units.csv", lambda rows: rows[:5] + [rows[5] + ["x"]] + rows[6:])
        self.refused("units.csv: line 6 has 21 fields", pin=self.w.realign())
        self.fresh()
        self.w.path("conditions.csv").write_bytes(self.w.path("conditions.csv").read_bytes() + b"\xff\n")
        self.refused("conditions.csv is not UTF-8", pin=self.w.realign())

    def test_result_rows(self):
        self.json_case(lambda d: d["primary_analysis"]["rows"][0]["contrast"]["acer"].__setitem__("exact", "1/2"),
                       "contrast.acer.exact differs from units.csv")
        self.fresh()
        self.json_case(lambda d: d["primary_analysis"]["rows"][3]["delta_ordinary"]["bpcer"].__setitem__(
            "value", d["primary_analysis"]["rows"][3]["delta_ordinary"]["bpcer"]["value"] + 1e-12),
            "delta_ordinary.bpcer.value differs from units.csv")
        self.fresh()
        self.json_case(lambda d: d["primary_analysis"]["rows"].__setitem__(1, d["primary_analysis"]["rows"][0]),
                       "twice")
        self.fresh()
        self.json_case(lambda d: d["primary_analysis"]["rows"].pop(), "does not hold 12 units")

    def nudge(self, path, delta, conditions=(), msg=None):
        def fn(d):
            cur = d
            keys = path.split(".")
            for k in keys[:-1]:
                cur = cur[k]
            cur[keys[-1]] = cur[keys[-1]] + delta
            self.new = cur[keys[-1]]
        self.w.edit_json(fn)
        for st, sc, field in conditions:
            self.w.edit_conditions(st, sc, field, repr(self.new))
        return self.refused(msg or path, pin=self.w.realign())

    def test_mean_cross_checks_each_metric(self):
        for m in ("apcer", "bpcer", "acer"):
            with self.subTest(m=m):
                self.fresh()
                self.nudge(f"primary_analysis.gap_ordinary.mean_delta.{m}", 1e-12,
                           [("ordinary", "all seeds", f"mean_delta_{m}")])
                self.fresh()
                self.nudge(f"primary_analysis.contrast.mean.{m}", 1e-12)
                self.fresh()
                self.nudge(f"primary_analysis.per_seed.43.gap_re_estimated.mean_delta.{m}", -1e-9,
                           [("re-estimated", "seed 43", f"mean_delta_{m}")])

    def test_interval_form(self):
        def swap(d):
            iv = d["primary_analysis"]["gap_ordinary"]["interval"]["apcer"]
            iv["low"], iv["high"] = iv["high"], iv["low"]
            self.iv = iv
        self.w.edit_json(swap)
        self.w.edit_conditions("ordinary", "all seeds", "low_apcer", repr(self.iv["low"]))
        self.w.edit_conditions("ordinary", "all seeds", "high_apcer", repr(self.iv["high"]))
        self.refused("gap_ordinary.interval.apcer has low > high", pin=self.w.realign())
        self.fresh()
        self.json_case(lambda d: d["primary_analysis"]["contrast"]["interval"]["bpcer"].__setitem__(
            "valid_resamples", 1001), "does not have 1 to 1,000 valid resamples")
        self.fresh()
        self.json_case(lambda d: d["primary_analysis"]["contrast"]["interval"]["acer"].__setitem__(
            "valid_resamples", 0), "does not have 1 to 1,000 valid resamples")
        self.fresh()
        self.json_case(lambda d: d["primary_analysis"]["contrast"]["interval"]["acer"].__setitem__(
            "valid_resamples", 926.0), "not an integer")
        self.fresh()
        self.json_case(lambda d: d["primary_analysis"]["contrast"]["interval"]["acer"].__setitem__("low", 0),
                       "is not a floating-point number")

    def test_mean_inside_its_interval(self):
        def fn(d):
            g = d["primary_analysis"]["gap_ordinary"]
            m = g["mean_delta"]["bpcer"]
            g["interval"]["bpcer"].update(low=m + 0.01, high=m + 0.2)
            self.lh = g["interval"]["bpcer"]
        self.w.edit_json(fn)
        self.w.edit_conditions("ordinary", "all seeds", "low_bpcer", repr(self.lh["low"]))
        self.w.edit_conditions("ordinary", "all seeds", "high_bpcer", repr(self.lh["high"]))
        self.refused("gap_ordinary.interval.bpcer does not contain its mean", pin=self.w.realign())
        self.fresh()
        self.json_case(lambda d: d["primary_analysis"]["per_seed"]["43"]["contrast"]["interval"]["apcer"].__setitem__(
            "high", d["primary_analysis"]["per_seed"]["43"]["contrast"]["mean"]["apcer"] - 1e-9),
            "per_seed.43.contrast.interval.apcer does not contain its mean")

    def test_valid_resamples_equal_in_every_block(self):
        self.json_case(lambda d: d["primary_analysis"]["per_seed"]["43"]["contrast"]["interval"]["acer"]
                       .__setitem__("valid_resamples", 500), "valid resamples of ACER differ between the blocks")
        self.fresh()
        self.json_case(lambda d: d["primary_analysis"]["contrast"]["interval"]["apcer"]
                       .__setitem__("valid_resamples", 999), "valid resamples of APCER differ between the blocks")

    def test_counts_cross_check(self):
        self.json_case(lambda d: d["primary_analysis"]["gap_re_estimated"]["acer_counts"].update(
            resnet18_lower=10, livenessnet_lower=2), "acer_counts",
            [("re-estimated", "all seeds", "resnet18_lower", "10"),
             ("re-estimated", "all seeds", "livenessnet_lower", "2")])
        self.fresh()
        self.json_case(lambda d: d["primary_analysis"]["contrast"].__setitem__("units_changing_sign", 3),
                       "contrast.units_changing_sign is not 2")
        self.fresh()
        self.json_case(lambda d: d["primary_analysis"]["contrast"].__setitem__("units", 11), "contrast.units is not 12")
        self.fresh()
        self.json_case(lambda d: d["primary_analysis"]["per_seed"]["44"]["contrast"]
                       .__setitem__("units_changing_sign", 2), "per_seed.44.contrast.units_changing_sign")
        self.fresh()
        self.json_case(lambda d: d["primary_analysis"]["per_seed"].__setitem__("45", {}), "seeds 42, 43 and 44")

    def test_sign_and_reading_rules_all_and_per_seed(self):
        self.json_case(lambda d: d["primary_analysis"]["gap_ordinary"].update(
            sign="unresolved", sign_reason="interval contains zero"), "gap_ordinary.sign does not follow",
            [("ordinary", "all seeds", "sign", "unresolved"),
             ("ordinary", "all seeds", "sign_reason", "interval contains zero")])
        self.fresh()
        self.json_case(lambda d: d["primary_analysis"]["contrast"].update(
            reading="unresolved", reason="interval contains zero"), "contrast.reading does not follow")
        self.fresh()
        self.json_case(lambda d: d["primary_analysis"]["per_seed"]["43"]["gap_ordinary"].update(
            sign="ResNet-18 lower", sign_reason="interval entirely below zero"), "per_seed.43.gap_ordinary.sign",
            [("ordinary", "seed 43", "sign", "ResNet-18 lower"),
             ("ordinary", "seed 43", "sign_reason", "interval entirely below zero")])
        self.fresh()
        self.json_case(lambda d: d["primary_analysis"]["per_seed"]["42"]["contrast"].update(
            reading=M.CONTRAST_ABOVE[0], reason=M.CONTRAST_ABOVE[1]), "per_seed.42.contrast.reading")

    def test_subjects_all_and_per_seed(self):
        self.json_case(lambda d: d["primary_analysis"]["gap_ordinary"].__setitem__("n_subjects", 15),
                       "gap_ordinary.n_subjects", [("ordinary", "all seeds", "n_subjects", "15")])
        self.fresh()
        self.json_case(lambda d: d["primary_analysis"]["per_seed"]["42"]["gap_re_estimated"].__setitem__(
            "n_subjects", 15), "per_seed.42.gap_re_estimated.n_subjects",
            [("re-estimated", "seed 42", "n_subjects", "15")])

    def test_conditions_csv_each_field(self):
        for field, value in (("low_acer", "-0.08"), ("low_apcer", "0.1"), ("high_bpcer", "-0.1"), ("ties", "1"),
                             ("valid_resamples_acer", "925"), ("mean_delta_bpcer", "0.1"), ("units", "3")):
            with self.subTest(field=field):
                self.fresh()
                self.w.edit_conditions("re-estimated", "seed 44", field, value)
                self.refused(f"conditions.csv (re-estimated, seed 44): {field}", pin=self.w.realign())
        self.fresh()
        self.w.edit_csv("conditions.csv", lambda rows: [r for r in rows if r[4] != "seed 43" or r[3] != "primary"
                                                        or r[0] != "4" or r[2] != "fixed_0.5"])
        self.refused("conditions.csv: the primary rows", pin=self.w.realign())

    def test_one_class_dev_entries(self):
        self.json_case(lambda d: d.__setitem__("one_class", {}), "one_class is not a list")
        self.fresh()
        self.json_case(lambda d: d["one_class"].pop(0), "one per network and BN state")
        self.fresh()
        self.json_case(lambda d: d["one_class"][0].__setitem__("one_class", 13), "not k of n evaluable")
        self.fresh()
        self.json_case(lambda d: d["one_class"].append(d["one_class"][0]), "one per network and BN state")


class D2_ConsistentWorlds(Base):
    """Mondi coerenti ricostruiti da questo file: lo script deve accettarli e dire ciò che i dati dicono."""

    def test_tie_and_zero_gap_counts(self):
        # ResNet-18 uguale a LivenessNet in (ristimata, fold 2, seed 43): divario 0, pareggio, nessun cambio di segno
        row = [r for r in dict_rows(INPUTS / "units.csv") if (r["patience"], r["threshold"], r["bn_state"], r["fold"],
               r["seed"]) == ("4", "fixed_0.5", "re-estimated", "2", "43")][0]
        pin = self.w.consistent([("re-estimated", 2, 43, "resnet18", row["livenessnet_apcer"],
                                  row["livenessnet_bpcer"])])
        files = self.w.build(pin)
        rows = list(csv.DictReader(io.StringIO(files["primary_table.csv"].decode())))
        e = [r for r in rows if r["row"] == "gap_re_estimated"][0]
        c = [r for r in rows if r["row"] == "contrast"][0]
        self.assertEqual((e["resnet18_lower"], e["livenessnet_lower"], e["ties"]), ("8", "3", "1"))
        self.assertEqual(c["units_changing_sign"], "2")
        self.assertIn("8 / 3 / 1", files["primary_table.md"].decode())

    def test_single_class_bona_fide_named(self):
        pin = self.w.consistent([("ordinary", 3, 42, "resnet18", "1/1", "0/1")])
        md = flat(self.w.build(pin)["primary_table.md"].decode())
        self.assertIn("ResNet-18 fold 3 seed 42 with ordinary BN, every frame bona fide", md)
        self.assertIn("3 models decide every test frame as the same class", md)

    def test_no_single_class(self):
        pin = self.w.consistent([("ordinary", 1, 42, "livenessnet", "1/5", "1/5"),
                                 ("ordinary", 4, 44, "livenessnet", "1/5", "1/5")])
        md = flat(self.w.build(pin)["primary_table.md"].decode())
        self.assertIn("No model of the two primary conditions decides every test frame as the same class", md)

    def test_sign_change_sentence_follows_labels(self):
        # un intervallo interamente sopra lo zero richiede una media positiva: la frase si prova sulle etichette
        def a(lo, le):
            return {"scopes": {"all seeds": {"keys": [0] * 12, "ordinary": {"label": (lo, "")},
                                             "re-estimated": {"label": (le, "")}, "contrast": {"changing": 2}}}}
        self.assertIn("there is one between", M.sign_change_text(a("ResNet-18 lower", "LivenessNet lower")))
        self.assertIn("there is none between", M.sign_change_text(a("ResNet-18 lower", "unresolved")))
        self.assertIn("there is none between", M.sign_change_text(a("ResNet-18 lower", "ResNet-18 lower")))

    def test_near_zero_sentence(self):
        self.w.edit_json(lambda d: d["primary_analysis"]["gap_ordinary"]["interval"]["apcer"].__setitem__(
            "low", 4e-05))
        self.w.edit_conditions("ordinary", "all seeds", "low_apcer", "4e-05")
        md = flat(self.w.build(self.w.realign())["primary_table.md"].decode())
        self.assertIn("lower bound of the APCER gap with ordinary BN: +0.0040 points", md)
        self.assertIn("[+0.00, +33.72]", md)


class E_Paths(Base):
    def test_input_directory(self):
        code, out, err = run(["--inputs", str(self.w.tmp / "none"), "--out", str(self.w.tmp / "o")])
        self.assertIn("the input directory does not exist", err)
        os.symlink(self.w.inputs, self.w.tmp / "alias")
        code, out, err = run(["--inputs", str(self.w.tmp / "alias"), "--out", str(self.w.tmp / "o")])
        self.assertEqual(code, 1)
        self.assertIn("the input directory is a symbolic link", err)
        code, out, err = run(["--inputs", str(self.w.path("units.csv")), "--out", str(self.w.tmp / "o")])
        self.assertIn("the input directory is not a directory", err)
        self.assertFalse((self.w.tmp / "o").exists())

    def test_input_file_symlink_dir_fifo(self):
        name = "units.csv"
        real = self.w.tmp / "real_units.csv"
        shutil.copy(self.w.path(name), real)
        self.w.path(name).unlink()
        os.symlink(real, self.w.path(name))
        self.refused("input units.csv is a symbolic link", out_name="o1")
        self.w.path(name).unlink()
        self.w.path(name).mkdir()
        self.refused("input units.csv is not a regular file", out_name="o2")
        self.w.path(name).rmdir()
        os.mkfifo(self.w.path(name))
        self.refused("input units.csv is not a regular file", out_name="o3")
        self.w.path(name).unlink()
        self.refused("missing input units.csv", out_name="o4")

    def test_input_changed_while_opened(self):
        real_fstat = os.fstat

        def other_inode(fd):
            st = real_fstat(fd)
            return os.stat_result((st.st_mode, st.st_ino + 1) + tuple(st)[2:])
        with mock.patch.object(M.os, "fstat", other_inode):
            self.refused("changed while it was opened")

    def test_output_is_or_inside_inputs_or_results_package(self):
        for out, msg in ((self.w.inputs, "is the input directory"), (self.w.path("units.csv"), "is an input file"),
                         (self.w.path("SHA256SUMS"), "is an input file"),
                         (self.w.inputs / "new", "inside the input directory"),
                         (self.w.inputs.parent / "new", "inside the package of the results"),
                         (self.w.root / "zz", "inside the package of the results"),
                         (self.w.inputs / "sub" / "new", "parent of the output directory does not exist")):
            with self.subTest(out=out):
                before = snapshot(self.w.tmp)
                code, o, err = run(["--inputs", str(self.w.inputs), "--out", str(out)])
                self.assertEqual(code, 1)
                self.assertIn(msg, err)
                self.assertEqual(snapshot(self.w.tmp), before)

    def test_output_inside_inputs_through_alias(self):
        os.symlink(self.w.inputs, self.w.tmp / "alias")
        code, o, err = run(["--inputs", str(self.w.inputs), "--out", str(self.w.tmp / "alias" / "new")])
        self.assertEqual(code, 1)
        self.assertIn("inside the input directory", err)
        self.assertFalse((self.w.inputs / "new").exists())

    def test_output_inside_this_package(self):
        fake = self.w.tmp / "paper"
        (fake / "test_logs").mkdir(parents=True)
        files = {n: b"x" for n in M.OUTPUTS}
        with mock.patch.object(M, "HERE", fake), mock.patch.object(M, "DEFAULT_AGAINST", fake / "outputs"), \
                mock.patch.object(M, "build", return_value=files):
            for out, msg in ((fake / "zz", "inside this package"), (fake / "test_logs" / "zz", "inside this package")):
                code, o, err = run(["--inputs", str(self.w.inputs), "--out", str(out)])
                self.assertEqual(code, 1)
                self.assertIn(msg, err)
                self.assertFalse(out.exists())
            code, o, err = run(["--inputs", str(self.w.inputs), "--out", str(fake / "outputs")])
            self.assertEqual(code, 0, o + err)
            self.assertEqual(sorted(os.listdir(fake / "outputs")), sorted(M.OUTPUTS))
            code, o, err = run(["--inputs", str(self.w.inputs), "--out", str(fake / "outputs" / "zz")])
            self.assertEqual(code, 1)
            self.assertIn("inside this package", err)
        code, o, err = run(["--inputs", str(self.w.inputs), "--out", str(HERE / "zz")])
        self.assertIn("inside this package", err)
        code, o, err = run(["--inputs", str(self.w.inputs), "--out", str(COMMITTED)])
        self.assertIn("already exists", err)

    def test_output_exists_or_symlink(self):
        (self.w.tmp / "empty").mkdir()
        self.refused("already exists", out_name="empty")
        os.symlink(self.w.tmp / "nowhere", self.w.tmp / "dangling")
        self.refused("already exists", out_name="dangling")
        os.symlink(self.w.tmp / "empty", self.w.tmp / "todir")
        self.refused("already exists", out_name="todir")
        self.assertEqual(os.listdir(self.w.tmp / "empty"), [])
        (self.w.tmp / "afile").write_text("x")
        code, o, err = run(["--inputs", str(self.w.inputs), "--out", str(self.w.tmp / "afile" / "o")])
        self.assertIn("parent of the output directory is not a directory", err)

    def test_options(self):
        o = str(self.w.tmp / "o")
        for args in (("--check", "--out", o), (), ("--out", o, "--text-only"), ("--out", o, "--against", o),
                     ("--bogus",), ("--out", o, "--out", o + "2"), ("--check", "--check"), ("--ch",),
                     ("--out=" + o, "--out", o + "2")):
            with self.subTest(args=args):
                r = cli("--inputs", str(self.w.inputs), *args)
                self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
                self.assertFalse((self.w.tmp / "o").exists())
                self.assertFalse((self.w.tmp / "o2").exists())

    def test_help_in_english(self):
        r = cli("--help")
        self.assertEqual(r.returncode, 0)
        self.assertIn("Regenerate the primary table", r.stdout)
        for w in ("Tabella", "rigenera", "soglia"):
            self.assertNotIn(w, r.stdout)

    def test_options_validated_before_reading_inputs(self):
        (self.w.tmp / "exists").mkdir()
        with mock.patch.object(M, "load_input_bytes", side_effect=AssertionError("inputs read")):
            code, o, err = run(["--inputs", str(self.w.inputs), "--out", str(self.w.tmp / "exists")])
        self.assertEqual(code, 1)
        self.assertIn("already exists", err)

    def test_error_after_pass_in_same_folder(self):
        files = {n: b"x" for n in M.OUTPUTS}
        with mock.patch.object(M, "build", return_value=files):
            code, o, err = run(["--inputs", str(self.w.inputs), "--out", str(self.w.tmp / "o")])
        self.assertEqual(code, 0, o + err)
        before = snapshot(self.w.tmp / "o")
        p = self.w.path("results.json")
        p.write_bytes(p.read_bytes()[:-2])
        self.refused("results.json: SHA-256", out_name="o2")
        self.refused("already exists", out_name="o")
        self.assertEqual(snapshot(self.w.tmp / "o"), before)

    def test_failure_while_writing_leaves_nothing(self):
        real_open = os.open
        files = {n: b"x" for n in M.OUTPUTS}

        def failing(path, flags, *a, **kw):
            if str(path).endswith("per_seed.md") and flags & os.O_CREAT:
                raise OSError(28, "No space left on device")
            return real_open(path, flags, *a, **kw)
        with mock.patch.object(M, "build", return_value=files), mock.patch.object(M.os, "open", failing):
            err = self.refused("could not write the outputs", out_name="o")
        self.assertNotIn(".mtf-", "".join(os.listdir(self.w.tmp)))
        del err

    def test_output_created_without_following_links(self):
        # O_CREAT | O_EXCL non segue un collegamento simbolico: un collegamento già presente fa fallire la scrittura
        target = self.w.tmp / "target.txt"
        target.write_text("keep")
        planted = self.w.tmp / ".mtf-planted"
        planted.mkdir()
        os.symlink(target, planted / "primary_table.csv")
        files = {n: b"x" for n in M.OUTPUTS}
        with mock.patch.object(M, "build", return_value=files), \
                mock.patch.object(M.tempfile, "mkdtemp", return_value=str(planted)):
            code, o, err = run(["--inputs", str(self.w.inputs), "--out", str(self.w.tmp / "o")])
        self.assertEqual(code, 1)
        self.assertIn("could not write the outputs", err)
        self.assertEqual(target.read_text(), "keep")
        self.assertFalse((self.w.tmp / "o").exists())
        self.assertFalse(planted.exists())

    def test_two_runs_on_the_same_folder(self):
        files = {n: b"x" for n in M.OUTPUTS}

        def appear(kind, path):
            def side(data, images=True):
                if kind == "full":
                    path.mkdir()
                    (path / "primary_table.csv").write_text("other run")
                elif kind == "empty":
                    path.mkdir()
                elif kind == "file":
                    path.write_text("other run")
                else:
                    os.symlink(self.w.tmp / "elsewhere", path)
                return files
            return side
        # il nome di uscita compare fra i controlli e la scrittura: cartella piena, vuota, file, collegamento
        for kind in ("full", "empty", "file", "symlink"):
            with self.subTest(kind=kind):
                out = self.w.tmp / f"o_{kind}"
                with mock.patch.object(M, "build", side_effect=appear(kind, out)):
                    code, o, err = run(["--inputs", str(self.w.inputs), "--out", str(out)])
                self.assertEqual(code, 1)
                self.assertIn("was taken by another process after the checks; nothing was written by this run", err)
                if kind == "full":
                    self.assertEqual(os.listdir(out), ["primary_table.csv"])
                elif kind == "empty":
                    self.assertEqual(os.listdir(out), [])          # non sostituita
                elif kind == "file":
                    self.assertEqual(out.read_text(), "other run")
                else:
                    self.assertTrue(out.is_symlink())
                self.assertNotIn(".mtf-", "".join(os.listdir(self.w.tmp)))

        # l'altra esecuzione scrive nella cartella riservata prima del rename: il rename fallisce (cartella non vuota)
        real_rename = os.rename
        o3 = self.w.tmp / "o3"

        def rename_after_other_run(src, dst):
            (o3 / "primary_table.csv").write_text("other run")
            return real_rename(src, dst)
        with mock.patch.object(M, "build", return_value=files), mock.patch.object(M.os, "rename",
                                                                                  rename_after_other_run):
            code, o, err = run(["--inputs", str(self.w.inputs), "--out", str(o3)])
        self.assertEqual(code, 1)
        self.assertIn("was taken by another process after the checks; nothing was written by this run", err)
        self.assertEqual(os.listdir(o3), ["primary_table.csv"])
        self.assertNotIn(".mtf-", "".join(os.listdir(self.w.tmp)))

    def test_reserved_name_replaced_by_a_file(self):
        # un altro processo toglie la cartella riservata e mette un file al suo posto prima del rename: ENOTDIR è la gara
        files = {n: b"x" for n in M.OUTPUTS}
        real_rename = os.rename
        out = self.w.tmp / "o"

        def swap_then_rename(src, dst):
            out.rmdir()
            out.write_text("other run")
            return real_rename(src, dst)
        with mock.patch.object(M, "build", return_value=files), mock.patch.object(M.os, "rename", swap_then_rename):
            code, o, err = run(["--inputs", str(self.w.inputs), "--out", str(out)])
        self.assertEqual(code, 1)
        self.assertIn("was taken by another process after the checks; nothing was written by this run", err)
        self.assertEqual(out.read_text(), "other run")
        self.assertNotIn(".mtf-", "".join(os.listdir(self.w.tmp)))

    def test_write_next_to_output_mode_and_handlers(self):
        # cartella temporanea accanto all'uscita (stesso file system), uscita in 0755, gestori dei segnali ripristinati
        files = {n: b"x" for n in M.OUTPUTS}
        real_mkdtemp = tempfile.mkdtemp
        seen = []

        def spy(*a, **kw):
            seen.append(kw.get("dir"))
            return real_mkdtemp(*a, **kw)
        before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        out = self.w.tmp / "o"
        with mock.patch.object(M, "build", return_value=files), mock.patch.object(M.tempfile, "mkdtemp", spy):
            code, o, err = run(["--inputs", str(self.w.inputs), "--out", str(out)])
        self.assertEqual(code, 0, o + err)
        self.assertEqual([os.path.realpath(d) for d in seen], [os.path.realpath(self.w.tmp)])
        self.assertEqual(os.stat(out).st_mode & 0o777, 0o755)
        self.assertEqual({s: signal.getsignal(s) for s in before}, before)
        with mock.patch.object(M, "build", return_value=files):
            code, o, err = run(["--inputs", str(self.w.inputs), "--out", str(out)])
        self.assertEqual(code, 1)
        self.assertEqual({s: signal.getsignal(s) for s in before}, before)

    def test_drawing_errors_keep_their_name(self):
        with mock.patch.object(M, "figure_images", side_effect=ValueError("drawing")):
            with self.assertRaises(ValueError):
                run(["--inputs", str(self.w.inputs), "--out", str(self.w.tmp / "o")])
        self.assertFalse((self.w.tmp / "o").exists())

    def test_temporary_directory_replaced_before_the_rename(self):
        # un'altra cartella prende il posto della cartella temporanea prima del rename: il rename la porta sul nome di
        # uscita; nessun segnale è arrivato, quindi il messaggio non ne nomina uno; la cartella altrui resta intatta
        files = {n: b"x" for n in M.OUTPUTS}
        out = self.w.tmp / "o"
        real_poll = M._poll

        def poll(state):
            if state.get("reserved_id") is not None and "swapped" not in state:
                state["swapped"] = True
                shutil.rmtree(state["tmp"])
                os.mkdir(state["tmp"])
                Path(state["tmp"], "theirs.txt").write_text("other process")
            return real_poll(state)
        with mock.patch.object(M, "build", return_value=files), mock.patch.object(M, "_poll", poll):
            code, o, err = run(["--inputs", str(self.w.inputs), "--out", str(out)])
        self.assertEqual(code, 1)
        self.assertIn("the output name or the temporary directory changed during the rename; nothing was delivered by "
                      "this run", err)
        self.assertNotIn("signal", err)
        self.assertEqual(os.listdir(out), ["theirs.txt"])

    def test_output_moved_away_after_the_rename(self):
        files = {n: b"x" for n in M.OUTPUTS}
        out = self.w.tmp / "o"
        moved = self.w.tmp / "moved"
        real_rename = os.rename

        def rename_then_move(src, dst):
            real_rename(src, dst)
            real_rename(dst, moved)
        with mock.patch.object(M, "build", return_value=files), mock.patch.object(M.os, "rename", rename_then_move):
            code, o, err = run(["--inputs", str(self.w.inputs), "--out", str(out)])
        self.assertEqual(code, 1)
        self.assertIn("changed during the rename; nothing was delivered by this run", err)
        self.assertNotIn("signal", err)
        self.assertFalse(os.path.lexists(out))
        self.assertEqual(sorted(os.listdir(moved)), sorted(M.OUTPUTS))      # non cancellate: non sono più di questa corsa

    def test_temporary_directory_of_another_process_never_removed(self):
        # la cartella temporanea è sostituita da una cartella altrui e il rename non avviene (la riserva non è più
        # vuota): la pulizia riconosce la cartella temporanea per identità e lascia intatta quella altrui
        files = {n: b"x" for n in M.OUTPUTS}
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
        with mock.patch.object(M, "build", return_value=files), mock.patch.object(M, "_poll", poll):
            code, o, err = run(["--inputs", str(self.w.inputs), "--out", str(out)])
        self.assertEqual(code, 1)
        self.assertIn("was taken by another process after the checks", err)
        tmps = [d for d in os.listdir(self.w.tmp) if d.startswith(".mtf-")]
        self.assertEqual(len(tmps), 1)
        self.assertEqual(os.listdir(self.w.tmp / tmps[0]), ["theirs.txt"])
        self.assertEqual(os.listdir(out), ["x.txt"])

    def test_cleanup_of_the_reservation_after_an_error(self):
        # errore dopo la riserva (rename che fallisce con EACCES): la riserva vuota di questa esecuzione sparisce
        files = {n: b"x" for n in M.OUTPUTS}

        def denied(src, dst):
            raise PermissionError(13, "Permission denied")
        out = self.w.tmp / "o"
        with mock.patch.object(M, "build", return_value=files), mock.patch.object(M.os, "rename", denied):
            code, o, err = run(["--inputs", str(self.w.inputs), "--out", str(out)])
        self.assertEqual(code, 1)
        self.assertIn("could not write the outputs (Permission denied); nothing was left", err)
        self.assertFalse(out.exists())
        self.assertNotIn(".mtf-", "".join(os.listdir(self.w.tmp)))

    def test_reservation_replaced_by_another_empty_directory(self):
        # un'altra cartella vuota prende il posto della riserva prima del rename: riconosciuta per inode, non sostituita
        # e non rimossa
        files = {n: b"x" for n in M.OUTPUTS}
        out = self.w.tmp / "o"
        real_poll = M._poll
        theirs = {}

        def poll(state):
            if state.get("reserved_id") is not None and not theirs:
                out.rmdir()
                out.mkdir()
                theirs["id"] = M._identity(out)
            return real_poll(state)
        with mock.patch.object(M, "build", return_value=files), mock.patch.object(M, "_poll", poll):
            code, o, err = run(["--inputs", str(self.w.inputs), "--out", str(out)])
        self.assertEqual(code, 1)
        self.assertIn("was taken by another process after the checks; nothing was written by this run", err)
        self.assertEqual(M._identity(out), theirs["id"])
        self.assertEqual(os.listdir(out), [])
        self.assertNotIn(".mtf-", "".join(os.listdir(self.w.tmp)))

    def test_without_matplotlib(self):
        out = self.w.tmp / "o"
        with mock.patch.dict(sys.modules, {"matplotlib": None}):
            code, o, err = run(["--inputs", str(self.w.inputs), "--out", str(out)])
            self.assertEqual(code, 1)
            self.assertIn("matplotlib is not available", err)
            self.assertFalse(out.exists())
            code, o, err = run(["--inputs", str(self.w.inputs), "--check"])
            self.assertEqual(code, 1)
            self.assertIn("--text-only", err)
            code, o, err = run(["--inputs", str(self.w.inputs), "--check", "--text-only"])
            self.assertEqual(code, 0, o + err)
            self.assertIn("figure_acer_gaps.png: structure valid, content not compared (--text-only)", o)


CHILD = r"""
import importlib.util, os, signal, sys, time
script, inputs, out, mode, point = sys.argv[1:6]
spec = importlib.util.spec_from_file_location("m", script)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
if mode == "fake":
    m.build = lambda data, images=True: {n: b"x" for n in m.OUTPUTS}
# gestori di partenza come da un terminale, anche se il processo che lancia la suite ignora SIGINT (job in background)
signal.signal(signal.SIGINT, signal.default_int_handler)
signal.signal(signal.SIGTERM, signal.SIG_DFL)
signal.signal(signal.SIGHUP, signal.SIG_DFL)
real_chmod, real_mkdir, real_rename, real_signal = os.chmod, os.mkdir, os.rename, signal.signal
state = {"installed": False, "restoring": False}


def ready():
    print("READY", "matplotlib" in sys.modules, flush=True)
    time.sleep(1.5)


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
        return real_signal(s, h)
    if state["installed"] and point == "during_restore" and not state["restoring"]:
        state["restoring"] = True
        ready()
    r = real_signal(s, h)
    if state["installed"] and point == "after_restore" and s == signal.SIGHUP:
        ready()
    return r


os.chmod, os.mkdir, os.rename, signal.signal = chmod, mkdir, rename, sig
sys.exit(m.main(["--inputs", inputs, "--out", out]))
"""


class S_Signals(Base):
    """Segnali mandati da fuori a un processo vero in cinque punti della scrittura, per SIGINT, SIGTERM e SIGHUP: con
    build finto (un solo thread, senza matplotlib) e con build vero (matplotlib importato, più thread)."""

    def one(self, mode, point, sig):
        out = self.w.tmp / f"o_{mode}_{point}_{int(sig)}"
        p = subprocess.Popen([sys.executable, "-I", "-B", "-c", CHILD, str(SCRIPT), str(self.w.inputs), str(out), mode,
                              point], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        before = ""
        line = p.stdout.readline()
        while line and not line.startswith("READY"):
            before += line
            line = p.stdout.readline()
        if not line:                                # il messaggio si legge solo dopo: stderr si chiude a fine processo
            so, se = p.communicate(timeout=120)
            self.fail(before + so + se)
        p.send_signal(sig)
        so, se = p.communicate(timeout=120)
        self.assertEqual(line.split()[1], "True" if mode == "real" else "False")
        return p.returncode, before + so, se, out

    def check(self, mode, point):
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):      # elenco proprio, non quello dello script
            with self.subTest(sig=int(sig)):
                rc, so, se, out = self.one(mode, point, sig)
                if point in ("before_reservation", "after_mkdir"):
                    self.assertEqual(rc, 1, so + se)
                    self.assertIn(f"interrupted by signal {int(sig)}; nothing was left", se)
                    self.assertFalse(os.path.lexists(out))
                    if point == "before_reservation":        # interrotto prima: la riserva non viene neppure creata
                        self.assertNotIn("reservation made", so)
                elif point == "after_restore" and sig != signal.SIGINT:
                    # gestore già ripristinato (quello di partenza, predefinito): il processo finisce per il segnale,
                    # a uscite complete, senza il messaggio "nothing was left"
                    self.assertEqual(rc, -sig, so + se)
                    self.assertNotIn("nothing was left", so + se)
                    self.assertEqual(sorted(os.listdir(out)), sorted(M.OUTPUTS))
                else:
                    self.assertEqual(rc, 0, so + se)
                    self.assertIn(f"written 8 files to {out}; signal {int(sig)} received, handled after the outputs "
                                  f"were complete", so)
                    self.assertIn("RESULT: PASS", so)
                    self.assertEqual(sorted(os.listdir(out)), sorted(M.OUTPUTS))
                    self.assertEqual(os.stat(out).st_mode & 0o777, 0o755)
                self.assertNotIn(".mtf-", "".join(os.listdir(self.w.tmp)))
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
    def test_real_after_mkdir(self):
        self.check("real", "after_mkdir")

    @need_mpl
    def test_real_after_rename(self):
        self.check("real", "after_rename")

    @need_mpl
    def test_real_during_restore(self):
        self.check("real", "during_restore")

    @need_mpl
    def test_real_after_restore(self):
        self.check("real", "after_restore")

    def test_sigkill_leaves_no_incomplete_output(self):
        rc, so, se, out = self.one("fake", "after_mkdir", signal.SIGKILL)
        self.assertEqual(rc, -signal.SIGKILL)
        self.assertTrue(not os.path.lexists(out) or os.listdir(out) == [])   # al più la riserva vuota (dichiarato)
        for d in os.listdir(self.w.tmp):
            if d.startswith(".mtf-"):
                shutil.rmtree(self.w.tmp / d)



class F_Check(Base):
    def copy_committed(self):
        d = self.w.tmp / "committed"
        shutil.copytree(COMMITTED, d)
        return d

    def check(self, against, *extra):
        return run(["--inputs", str(self.w.inputs), "--check", "--against", str(against), *extra])

    def test_text_only_passes_on_committed(self):
        code, o, err = run(["--check", "--text-only"])
        self.assertEqual(code, 0, o + err)
        self.assertEqual(o.count(": identical"), len(M.TEXT_OUTPUTS))
        self.assertEqual(o.count("structure valid"), 2)

    @need_mpl
    def test_full_check_passes_on_committed(self):
        code, o, err = run(["--check"])
        self.assertEqual(code, 0, o + err)
        self.assertEqual(o.count(": identical"), len(M.OUTPUTS))

    def mutate_each(self, names, extra=(), expect="DIFFERENT", where=None):
        for name in names:
            with self.subTest(name=name):
                d = self.copy_committed()
                p = d / name
                b = bytearray(p.read_bytes())
                b[where(len(b)) if where else len(b) // 2] ^= 0x01
                p.write_bytes(bytes(b))
                code, o, err = self.check(d, *extra)
                self.assertEqual(code, 1, o + err)
                self.assertIn(f"{name}: {expect}", o)
                self.assertIn("RESULT: FAIL", o)
                shutil.rmtree(d)

    def test_one_byte_in_each_text_output(self):
        self.mutate_each(M.TEXT_OUTPUTS, ("--text-only",))

    def test_text_only_checks_image_structure(self):
        self.mutate_each(["figure_acer_gaps.png"], ("--text-only",), "INVALID STRUCTURE")
        self.mutate_each(["figure_acer_gaps.pdf"], ("--text-only",), "INVALID STRUCTURE", where=lambda n: 1)
        self.mutate_each(["figure_acer_gaps.pdf"], ("--text-only",), "INVALID STRUCTURE", where=lambda n: n - 3)
        d = self.copy_committed()
        (d / "figure_acer_gaps.png").write_bytes((d / "figure_acer_gaps.png").read_bytes()[:-12])
        code, o, err = self.check(d, "--text-only")
        self.assertIn("figure_acer_gaps.png: INVALID STRUCTURE", o)
        d2 = self.w.tmp / "committed2"
        shutil.copytree(COMMITTED, d2)
        (d2 / "figure_acer_gaps.png").write_bytes((d2 / "figure_acer_gaps.png").read_bytes() + b"x")
        code, o, err = self.check(d2, "--text-only")
        self.assertIn("figure_acer_gaps.png: INVALID STRUCTURE", o)
        self.assertFalse(M.png_structure_ok(b""))
        self.assertFalse(M.pdf_structure_ok(b"partial"))

    @need_mpl
    def test_one_byte_in_each_output(self):
        self.mutate_each(M.OUTPUTS)

    def test_against_refusals(self):
        d = self.copy_committed()
        (d / ".DS_Store").write_text("x")
        code, o, err = self.check(d, "--text-only")
        self.assertIn("expected exactly", err)
        (d / ".DS_Store").unlink()
        real = self.w.tmp / "real.csv"
        shutil.move(d / "per_seed.csv", real)
        os.symlink(real, d / "per_seed.csv")
        code, o, err = self.check(d, "--text-only")
        self.assertIn("committed output per_seed.csv is a symbolic link", err)
        os.symlink(COMMITTED, self.w.tmp / "alias")
        code, o, err = self.check(self.w.tmp / "alias", "--text-only")
        self.assertIn("the directory to check is a symbolic link", err)
        code, o, err = self.check(self.w.inputs, "--text-only")
        self.assertIn("the directory to check is the input directory", err)
        code, o, err = self.check(self.w.tmp / "none", "--text-only")
        self.assertIn("the directory to check does not exist", err)
        (self.w.tmp / "f").write_text("x")
        code, o, err = self.check(self.w.tmp / "f", "--text-only")
        self.assertIn("the directory to check is not a directory", err)


class G_Determinism(unittest.TestCase):
    def test_two_text_generations(self):
        data = M.load_input_bytes(str(INPUTS))
        self.assertEqual(M.build(data, images=False), M.build(data, images=False))

    @need_mpl
    def test_two_full_generations(self):
        data = M.load_input_bytes(str(INPUTS))
        a, b = M.build(data), M.build(data)
        self.assertEqual(a, b)
        self.assertTrue(M.png_structure_ok(a["figure_acer_gaps.png"]))
        self.assertTrue(M.pdf_structure_ok(a["figure_acer_gaps.pdf"]))
        for k in (b"CreationDate", b"Producer", b"Creator", b"matplotlib"):
            self.assertNotIn(k, a["figure_acer_gaps.pdf"])
        self.assertNotIn(b"Software", a["figure_acer_gaps.png"])

    @need_mpl
    def test_local_matplotlibrc_ignored(self):
        with tempfile.TemporaryDirectory() as cwd, tempfile.TemporaryDirectory() as cfg:
            Path(cwd, "matplotlibrc").write_text("axes.facecolor: red\nlines.linewidth: 5\nfont.size: 20\n")
            out = Path(cwd) / "o"
            env = {k: v for k, v in os.environ.items() if k != "MATPLOTLIBRC"}
            env["MPLCONFIGDIR"] = cfg
            r = cli("--out", str(out), cwd=cwd, env=env)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            for name in M.OUTPUTS:
                self.assertEqual((out / name).read_bytes(), (COMMITTED / name).read_bytes(), name)


class H_PublicFiles(unittest.TestCase):
    def test_no_identifiers_or_private_strings(self):
        ids = list(json.loads((INPUTS / "results.json").read_text())["test_frame_ids_sha256"].values())
        bad = ids + ["/Users/", "/home/", "/private/", "valecass", "@", "Claude", "Anthropic", "ChatGPT", "assistant",
                     " agent", "Plesk", "server", "virtual environment", "working repository"]
        files = [p for p in HERE.rglob("*") if p.is_file() and p.suffix in (".md", ".csv", ".tex", ".py", ".txt")
                 and p.name != "check_make_table_figure.py"]
        self.assertGreaterEqual(len(files), 9)
        for p in files:
            t = p.read_text(encoding="utf-8")
            for s in bad:
                with self.subTest(file=p.name, s=s):
                    self.assertNotIn(s, t)

    def test_scripts_are_not_executables(self):
        for p in (SCRIPT, Path(__file__)):
            self.assertFalse(p.read_text().startswith("#!"))
            self.assertFalse(os.stat(p).st_mode & 0o111)


class ZZ_Untouched(unittest.TestCase):
    def test_package_and_inputs_untouched(self):
        self.assertEqual(snapshot(HERE, RESULTS_PKG), BEFORE)

    def test_declared_skips_only(self):
        self.assertTrue(set(MISSING_INPUT) <= {"matplotlib"})


def main():
    argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a != "--verbose"] + (["-v"] if "--verbose" in sys.argv else [])
    result = unittest.main(argv=argv, exit=False, verbosity=1).result
    skipped = [f"{t.id()}: {r}" for t, r in result.skipped]
    print(f"tests run: {result.testsRun}, failures: {len(result.failures)}, errors: {len(result.errors)}, "
          f"skipped: {len(skipped)}")
    for s in skipped:
        print(f"skipped {s}")
    ok = result.wasSuccessful()
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
