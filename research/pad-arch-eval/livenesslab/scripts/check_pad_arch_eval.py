"""
Test di scripts/pad_arch_eval.py (la valutazione sul test NUAA dei 48 modelli del confronto LivenessNet / ResNet-18),
offline: nessun dato reale, nessun checkpoint reale.

Mondo sintetico in cartelle temporanee: il mondo della suite del comando dei run (check_pad_arch_run.World: manifest
M2, cache 8×8, run delle celle, schema dei fold con il test, frame estratti, configurazione della preregistrazione, record
della prova, file del piano, configurazione dei run) più un PROTOCOL.md sintetico con le due conclusioni nella forma del
§6, run_plan.csv rigenerato, entrambi nelle righe del SHA256SUMS finto del pacchetto, la configurazione di questo comando
e le due cartelle dei run prodotte da scripts/pad_arch_run.py stesso (run non ufficiali, con il TensorFlow, il trainer, lo
scorer e la ristima finti della sua suite): la valutazione legge esattamente il formato che il runner scrive.

Due livelli:
  - senza TensorFlow (classi A-N): configurazione, file fissati, tetto, conclusioni, coppie; il controllo chiesto dal
    coordinatore nei due punti, un test per caso (soglia effettiva diversa da 0,5 per LivenessNet sola o per ResNet-18
    sola, combinazione mancante nel solo stato ordinario o nel solo stato ristimato, combinazione duplicata, rete
    mancante, condizione secondaria incompleta), ciascuno con il messaggio e nessun file di risultati; la validazione
    delle cartelle dei run con fixture avversarie (checksum riallineati: scatta la regola, non l'impronta); la
    riproduzione del dev (MISMATCH, nessuna lettura del test); tetto e SIGUSR1; ordine delle fasi (nessuna lista del
    test prima delle 48 riproduzioni); ID dei frame fra reti e fra stati della BN; uscite senza identificativi; i
    risultati primari ricalcolati in modo indipendente dai file riservati; tentativi e cartella;
  - con TensorFlow (classi Z*, ultime): score_model su LivenessNet del codice di riferimento e su ResNet-18 veri
    (statistiche reinstallate uguali a BR.evaluate con le statistiche assegnate a mano; statistiche sbagliate
    rifiutate), e una valutazione vera su due run veri del runner su CPU (frame 8×8, trainer accorciato).
Senza TensorFlow i test che costruiscono un modello Keras sono saltati con il motivo dichiarato in MISSING_INPUT; gli
altri girano. TMPDIR deve stare fuori dalla radice del repository (controllato). Nessuna scrittura nel repository
(controllo finale su scripts, docs, delivery, results, export, src).

  .venv/bin/python -I -B scripts/check_pad_arch_eval.py [--verbose] [-f]
"""
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_nuaa_folds_run as CFR  # noqa: E402
import check_pad_arch_comparison as CPA  # noqa: E402
import check_pad_arch_run as CR  # noqa: E402
import nuaa_bn_recalibration as BR  # noqa: E402
import nuaa_folds_run as FR  # noqa: E402
import nuaa_session_ablation_run as SR  # noqa: E402
import pad_arch_comparison as PAC  # noqa: E402
import pad_arch_definitions as D  # noqa: E402
import pad_arch_eval as E  # noqa: E402
import pad_arch_models as AM  # noqa: E402
import pad_arch_run as R  # noqa: E402

TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
MISSING_INPUT = {
    "tensorflow": "TensorFlow is not installed: the tests that build or score a Keras model are skipped",
    "reference architecture": "src/livedetection/src/architectures.py (LivenessNet of the reference code) is not present",
    "delivered configuration": "delivery/archeval/pad_arch_eval.json or the files it pins are not present",
}
DATE, STATE = CR.DATE, CR.STATE
EVAL_ID = f"{DATE}-ARCHEVAL-{STATE['commit']}"
CONCLUSIONS = ('Conclusions, written before the test:\nif the interval excludes zero, "on these checkpoints the signed difference is '
               'algebraically lower [higher] with this re-estimation; no cause"; otherwise\n"this comparison does not show it".')
QUALITY = {"LivenessNet": 0.12, "ResNet-18": 0.2}       # separazione delle classi dei punteggi finti del test, per rete


def skip_missing(test: unittest.TestCase, what: str):
    """Unico modo di saltare un test di questa suite: un ingresso mancante dichiarato in MISSING_INPUT."""
    test.skipTest(f"missing input: {MISSING_INPUT[what]}")


def need_tf(test: unittest.TestCase):
    if not TF_AVAILABLE:
        skip_missing(test, "tensorflow")
    if not CPA.REF_ARCH.is_file():
        skip_missing(test, "reference architecture")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def author(t: str = "x") -> str:
    return f"a choice of the author: {t}"


# ============================================================================ mondo sintetico

class World(CR.World):
    """Il mondo del comando dei run con PROTOCOL.md e run_plan.csv nel pacchetto finto, la configurazione di questo
    comando e le due cartelle dei run prodotte dal runner con i suoi finti."""

    def build(self):
        super().build()
        self.eval_cfg = self.base / "archeval.json"
        self.write_eval_cfg(self.eval_doc())
        self.make_runs()

    def write_prereg_files(self):
        (self.base / "PROTOCOL.md").write_text(f"# Protocol\n\n## 6 Definitions\n\n{CONCLUSIONS}\n", encoding="utf-8")
        cfg_pr = json.loads(self.cfg_path.read_text())
        (self.base / "run_plan.csv").write_text(E.plan_text(cfg_pr, self.trial), encoding="utf-8", newline="")

    def package_sums(self, over=None):
        """Il SHA256SUMS finto del pacchetto, con anche PROTOCOL.md e plan/run_plan.csv del mondo (scritti la prima volta;
        salvo over)."""
        if not (self.base / "PROTOCOL.md").is_file():
            self.write_prereg_files()
        lines = {"PROTOCOL.md": sha((self.base / "PROTOCOL.md").read_bytes()), "plan/run_plan.csv": sha((self.base / "run_plan.csv").read_bytes())}
        super().package_sums({**lines, **(over or {})})

    def eval_doc(self) -> dict:
        pin = lambda p: {"path": p.name, "sha256": sha(p.read_bytes())}  # noqa: E731
        cfg_pr = json.loads(self.cfg_path.read_text())
        return {
            "schema": E.CONFIG_SCHEMA, "task": "t", "authorisation": "published before the test is read", "operations": "commit; nohup",
            "preregistration": {"protocol": pin(self.base / "PROTOCOL.md"), "run_plan": pin(self.base / "run_plan.csv")},
            "run_configuration": pin(self.run_cfg), "code": {f: sha((ROOT / f).read_bytes()) for f in E.CODE_FILES},
            "runs": list(R.RUNS), "run_checks": "PASS, SHA256SUMS",
            "primary_check": {"requested": "requested by the coordinator: q", "where": author("before any test frame; before primary_analysis"),
                              "effective_threshold": author()},
            "condition_checks": author(), "bn_states": {"ordinary": "o", "re_estimated": author("re-installed")},
            "dev_reproduction": author("1e-5, MISMATCH"), "thresholds": "t", "test_scoring": author(),
            "comparisons": {"pairs": E._pairs_json(), "rule": author()}, "conclusions": {"source": "s", "rule": author()},
            "cap": {"hours": E.cap_from_plan(cfg_pr, self.trial)["cap_hours"], "unit": E.CAP_UNIT, "derivation": author(), "checks": "c",
                    "on_cap": "INCOMPLETE"},
            "after_stop": author("no automatic resume; test_read_started; C2 card"), "attempts": author("--unofficial-out-root"),
            "machine_rule": "the threads; a difference stops the work before any evaluation", "late_failures": "run.log",
            "degenerate_apart": author("the rule of the group ten of PR #27"),
            "outputs": {"run": ["status.json", "run.log"], "public": list(E.PUBLIC_OUTPUTS),
                        "restricted": [f"<run>_<model>{E.TEST_SUFFIX}"] + list(E.RESTRICTED_FIXED)},
            "report": "r", "choices_of_the_author": {k: "c" for k in E.CHOICE_KEYS}, "test": E.TEST_READ}

    def write_eval_cfg(self, doc: dict) -> Path:
        self.eval_cfg.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        return self.eval_cfg

    def make_runs(self, out: Path = None, hooks=None):
        """Le due cartelle dei run (nuaa_patience_4 e nuaa_patience_12) scritte da pad_arch_run.py con i finti della sua suite."""
        out = out or self.tmp / "archrun"
        shutil.rmtree(out, ignore_errors=True)
        self.run_dirs = {}
        for run in R.RUNS:
            rc, o, e = CR.run_main(self.argv("run", run=run, out=out), hooks or CR.flow_hooks(self))
            if rc != 0:
                raise RuntimeError(f"the runner did not produce {run}: {(o + e)[-2000:]}")
            self.run_dirs[run] = CR.run_folder(self, run).parent.parent / out.name / CR.run_folder(self, run).name
        return self.run_dirs

    def copy_runs(self) -> dict:
        """Una copia delle due cartelle dei run in una cartella nuova (stessi nomi), da alterare in un test."""
        d = Path(tempfile.mkdtemp(dir=self.tmp, prefix="runs-"))
        return {k: Path(shutil.copytree(v, d / v.name, symlinks=True)) for k, v in self.run_dirs.items()}

    def eval_argv(self, cmd="evaluate", runs=None, out=None, extra=(), attempt=None, unofficial=True) -> list:
        runs = runs or self.run_dirs
        a = [cmd, "--config", str(self.eval_cfg), "--input-root", str(self.base), "--run-p4", str(runs["nuaa_patience_4"]),
             "--run-p12", str(runs["nuaa_patience_12"])]
        if cmd == "evaluate":
            a += ["--out-root", str(out or self.tmp / "eval"), "--date", DATE, "--i-am-authorized"] + (["--unofficial-out-root"] if unofficial else [])
        if attempt is not None:
            a += ["--attempt", str(attempt)]
        return a + list(extra)


WORLD = None


def world() -> World:
    global WORLD
    if WORLD is None:
        WORLD = World()
    return WORLD


# ============================================================================ finti e iniezioni

class Log:
    """Registro delle chiamate del valutatore finto: (ruolo, chiave del modello, schema dei fold caricato)."""

    def __init__(self):
        self.calls = []


def fake_test_scores(rec: dict, state: str, part: dict) -> np.ndarray:
    """Punteggi d'attacco finti del test, deterministici per modello e stato: separazione delle classi secondo la rete,
    rumore dal seed, fold e patience; nessuna metrica nota prima."""
    rng = np.random.default_rng(rec["seed"] * 1000 + rec["fold"] * 10 + rec["patience"] + (5 if state == "re-estimated" else 0))
    y = np.asarray(part["y"], np.float64)
    q = QUALITY[rec["network"]] + (0.05 if state == "re-estimated" else 0.0)
    return np.clip(0.5 + (2 * y - 1) * q + rng.normal(0, 0.15, len(y)), 0.001, 0.999)


def lz(p) -> np.ndarray:
    p = np.asarray(p, np.float64)
    return np.log(np.stack([1 - p, p], axis=1))


def fake_score(log: Log = None, dev_over=None, on_call=None):
    """Il valutatore finto con la firma di score_model: sul dev restituisce i punteggi salvati dal run (riproduzione
    esatta, salvo dev_over), sul test i punteggi finti; registra ogni chiamata."""
    def score(ctx, rec, parts):
        out = {s: {} for s in D.BN_STATES}
        for role, part in parts.items():
            if log is not None:
                log.calls.append((role, rec["key"], len(ctx.inp.folds)))
            if on_call:
                on_call(ctx, rec, role)
            for s in D.BN_STATES:
                if role == "dev":
                    p = np.array(rec["dev"][s]["s"], np.float64)
                    if dev_over:
                        p = dev_over(rec, s, p)
                else:
                    p = fake_test_scores(rec, s, part)
                out[s][role] = {"p": p, "z": lz(p), "softmax_gap": 0.0}
        return out
    return score


def eval_hooks(w: World, log: Log = None, score=None, plan_hook=None, units_hook=None, tf=None, **r) -> dict:
    h = CR.base_hooks(w, TF_MODULE=lambda: tf or CR.FakeTF(), **r)
    h["e"] = {"SCORE_MODEL": score or fake_score(log), "PLAN_HOOK": plan_hook, "UNITS_HOOK": units_hook}
    return h


@contextlib.contextmanager
def injected(e=None, **hooks):
    """Le iniezioni della suite del runner (r, p, sr, fr, br) e di questo comando (e), ripristinate all'uscita."""
    saved = [(k, getattr(E, k)) for k in (e or {})]
    try:
        for k, v in (e or {}).items():
            setattr(E, k, v)
        with CR.injected(**hooks):
            yield
    finally:
        for k, v in saved:
            setattr(E, k, v)


def run_eval(argv, hooks: dict) -> tuple:
    out, err = io.StringIO(), io.StringIO()
    with injected(**hooks), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = E.main(argv)
    return rc, out.getvalue(), err.getvalue()


def tree(folder: Path) -> dict:
    return {str(p.relative_to(folder)): sha(p.read_bytes()) for p in sorted(folder.rglob("*")) if p.is_file()} if folder.exists() else {}


def fresh_out(w: World) -> Path:
    return Path(tempfile.mkdtemp(dir=w.tmp, prefix="eval-"))


def no_results(test: unittest.TestCase, folder: Path):
    """Nessun file di risultati: nella cartella solo status.json e run.log; nessuna uscita pubblica o riservata."""
    test.assertEqual(sorted(p.name for p in folder.iterdir() if p.is_file()), ["run.log", "status.json"])
    for sub in ("public", "restricted"):
        test.assertEqual(sorted(os.listdir(folder / sub)) if (folder / sub).is_dir() else [], [], sub)


FLOW = {}


def flow_pass() -> dict:
    """La valutazione completa nel mondo con il valutatore finto (una volta, condivisa dai test)."""
    if not FLOW:
        w = world()
        log = Log()
        out = fresh_out(w)
        rc, o, e = run_eval(w.eval_argv(out=out), eval_hooks(w, log))
        FLOW.update(rc=rc, out=o, err=e, log=log, folder=out / EVAL_ID)
    return FLOW


def flow_ok(test: unittest.TestCase) -> dict:
    """La valutazione condivisa, con l'esito controllato prima di leggere qualunque uscita (un'uscita mancante è un
    fallimento dell'asserzione, non un errore)."""
    f = flow_pass()
    test.assertEqual(f["rc"], 0, (f["out"] + f["err"])[-3000:])
    return f


def assert_stops(test: unittest.TestCase, fn, msg: str):
    """fn deve fermarsi con un Failure il cui messaggio corrisponde a msg; qualunque altra eccezione, o nessuna, è
    un'asserzione fallita (un mutante che toglie il controllo è ucciso da un'asserzione, non da un errore)."""
    try:
        fn()
    except FR.Failure as e:
        test.assertRegex(str(e), msg)
        return
    except Exception as e:  # noqa: BLE001
        test.fail(f"not stopped by the expected check ({msg}): {type(e).__name__}: {e}")
    test.fail(f"not stopped by the expected check ({msg})")


def read_csv(p: Path) -> list:
    return list(csv.DictReader(io.StringIO(p.read_text(encoding="utf-8"))))


def rewrite(run: Path, sub: str, name: str, change) -> None:
    """Fixture avversaria: altera un file di una cartella di run e riallinea il SHA256SUMS della sua sottocartella e lo
    SHA-256 registrato in status.json, così l'impronta non ferma il comando e deve scattare la regola."""
    p = run / sub / name
    p.write_bytes(change(p.read_bytes()))
    sums = FR.read_sums((run / sub / "SHA256SUMS").read_bytes(), "s")
    sums[name] = sha(p.read_bytes())
    text = "".join(f"{h}  {n}\n" for n, h in sums.items())
    (run / sub / "SHA256SUMS").write_text(text, encoding="utf-8")
    st = json.loads((run / "status.json").read_text())
    st[f"{sub}_sha256sums_sha256"] = sha(text.encode())
    (run / "status.json").write_text(json.dumps(st, indent=1) + "\n", encoding="utf-8")


def rewrite_status(run: Path, change) -> None:
    st = json.loads((run / "status.json").read_text())
    change(st)
    (run / "status.json").write_text(json.dumps(st, indent=1) + "\n", encoding="utf-8")


def csv_change(fn):
    """Una modifica delle righe di un CSV (lista di dizionari) come funzione sui byte."""
    def change(b: bytes) -> bytes:
        rows = list(csv.DictReader(io.StringIO(b.decode())))
        cols = list(rows[0])
        fn(rows)
        return FR.csv_text(cols, [[r[c] for c in cols] for r in rows]).encode()
    return change


def json_change(fn):
    def change(b: bytes) -> bytes:
        d = json.loads(b)
        fn(d)
        return (json.dumps(d, indent=1) + "\n").encode()
    return change


def plan_full() -> list:
    """Un piano completo sintetico: 2 patience × 2 reti × 4 fold × 3 seed × 2 stati × 2 regole, soglia 0,5 per la fissa."""
    return [{"key": f"p{p}_{n}_fold{f}_seed{s}", "patience": p, "network": n, "fold": f, "seed": s, "bn_state": b, "rule": t,
             "threshold": 0.5 if t == "fixed_0.5" else 0.37} for p in D.PATIENCES for n in AM.NETWORKS for f in range(1, 5) for s in D.PLAN_SEEDS
            for b in D.BN_STATES for t in D.THRESHOLD_RULES]


def units_full(threshold=0.5, n_per=6) -> list:
    """Unità sintetiche complete di una condizione: 12 unità (NUAA, fold, seed), due reti, frame con ID unici per fold."""
    out = []
    for f in range(1, 5):
        ids = [f"real/f{f}_{i:03d}.jpg" for i in range(n_per)]
        y = np.array([i % 2 for i in range(n_per)])
        subj = np.array([f"s{f}{i % 3}" for i in range(n_per)])
        for s in D.PLAN_SEEDS:
            out.append({"unit": ("NUAA", f, s), "networks": {n: {"y": y, "s": np.linspace(0.1, 0.9, n_per), "subject": subj, "threshold": threshold,
                                                                 "sample_id": list(ids)} for n in AM.NETWORKS}})
    return out


# ============================================================================ A configurazione

class TestAConfig(unittest.TestCase):
    def setUp(self):
        self.w = world()

    def refused(self, change, msg):
        doc = self.w.eval_doc()
        change(doc)
        p = self.w.tmp / "bad_eval.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaisesRegex(FR.Failure, msg):
            E.load_config(p)

    def test_world_configuration_loads(self):
        cfg, h = E.load_config(self.w.eval_cfg)
        self.assertEqual(h, sha(self.w.eval_cfg.read_bytes()))
        self.assertEqual(cfg["comparisons"]["pairs"], E._pairs_json())

    def test_refusals(self):
        self.refused(lambda d: d.pop("report"), "fields must be exactly")
        self.refused(lambda d: d.__setitem__("extra", 1), "fields must be exactly")
        self.refused(lambda d: d.__setitem__("schema", "x"), "schema must be")
        self.refused(lambda d: d.__setitem__("test", E.TEST_READ + " "), "test must state exactly")
        self.refused(lambda d: d.__setitem__("authorisation", "after the merge"), "published before the test")
        self.refused(lambda d: d.__setitem__("operations", "run it"), "nohup")
        self.refused(lambda d: d.__setitem__("runs", ["nuaa_patience_12", "nuaa_patience_4"]), "runs must be")
        self.refused(lambda d: d["code"].pop(E.RUN_FILE), "code needs exactly")
        self.refused(lambda d: d["code"].__setitem__(E.RUN_FILE, "x"), "code SHA-256 values malformed")
        self.refused(lambda d: d["primary_check"].__setitem__("requested", "a wish"), "requested by the coordinator")
        self.refused(lambda d: d["primary_check"].__setitem__("where", author("before primary_analysis")), "two places")
        self.refused(lambda d: d["primary_check"].__setitem__("where", "before any test frame; primary_analysis"), "choice of the author")
        self.refused(lambda d: d["primary_check"].__setitem__("effective_threshold", "the field"), "choice of the author")
        self.refused(lambda d: d.__setitem__("condition_checks", "all"), "choice of the author")
        self.refused(lambda d: d["bn_states"].__setitem__("re_estimated", author("re-estimated again")), "re-installed")
        self.refused(lambda d: d.__setitem__("dev_reproduction", author("1e-4, MISMATCH")), "tolerance")
        self.refused(lambda d: d["comparisons"].__setitem__("pairs", E._pairs_json()[:-1]), "comparisons.pairs")
        self.refused(lambda d: d["comparisons"].__setitem__("pairs", [[b, a] for a, b in E._pairs_json()]), "comparisons.pairs")
        self.refused(lambda d: d["cap"].__setitem__("unit", "hours of CPU time"), "cap.unit")
        self.refused(lambda d: d["cap"].__setitem__("hours", True), "finite positive")
        self.refused(lambda d: d["cap"].__setitem__("on_cap", "stops"), "INCOMPLETE")
        self.refused(lambda d: d.__setitem__("after_stop", author("resume")), "no automatic resume")
        self.refused(lambda d: d.__setitem__("attempts", author("attempts")), "unofficial output root")
        self.refused(lambda d: d.__setitem__("machine_rule", "same machine"), "machine_rule")
        self.refused(lambda d: d["outputs"].__setitem__("public", ["report.md"]), "outputs must list")
        self.refused(lambda d: d["choices_of_the_author"].pop("cap"), "choices_of_the_author needs exactly")
        self.refused(lambda d: d["preregistration"]["protocol"].__setitem__("sha256", "z"), "malformed")

    def test_pairs(self):
        """Le 12 coppie: ciascuna cambia un solo fattore, la prima condizione ha il secondo livello; la prima coppia è il
        contrasto primario (ristimata meno ordinaria a patience 4 e soglia 0,5); nessuna coppia ripetuta."""
        p = E.condition_pairs()
        self.assertEqual(len(p), 12)
        self.assertEqual(len(set(p)), 12)
        self.assertEqual(p[0], (D.PRIMARY_RE_ESTIMATED, D.PRIMARY_ORDINARY))
        levels = (D.PATIENCES, D.BN_STATES, D.THRESHOLD_RULES)
        for a, b in p:
            diff = [i for i in range(3) if a[i] != b[i]]
            self.assertEqual(len(diff), 1, (a, b))
            self.assertEqual((a[diff[0]], b[diff[0]]), (levels[diff[0]][1], levels[diff[0]][0]))


class TestADeliveredConfig(unittest.TestCase):
    def test_delivered_configuration(self):
        """La configurazione consegnata: si carica, i suoi SHA-256 del codice sono quelli dei file, i file fissati e il
        tetto tornano con i file del repository (solo configurazioni e file del piano, nessun dato)."""
        p = ROOT / E.DEFAULT_CONFIG
        if not p.is_file():
            skip_missing(self, "delivered configuration")
        cfg, _ = E.load_config(p)
        pins = E.load_pins(cfg, ROOT)
        self.assertEqual(pins.cap["cap_hours"], 3.5)
        self.assertAlmostEqual(pins.cap["evaluation_hours_estimate"], 2.1284494927430355, places=12)
        self.assertEqual(pins.test_per_fold, [3242, 3114, 3439, 2819])
        self.assertIn("lower [higher]", pins.conclusions["excludes_zero"])
        self.assertIn("does not exclude it", pins.conclusions["otherwise"])
        self.assertEqual(cfg["primary_check"]["requested"].count("requested by the coordinator"), 1)


# ============================================================================ B file fissati, tetto, conclusioni

class TestBPins(unittest.TestCase):
    def setUp(self):
        self.w = world()

    def pins(self):
        cfg, _ = E.load_config(self.w.eval_cfg)
        with injected(**CR.base_hooks(self.w)):
            return E.load_pins(cfg, self.w.base)

    def test_pins_pass(self):
        p = self.pins()
        self.assertEqual(p.folds, [1, 2, 3, 4])
        self.assertEqual(tuple(p.seeds), D.PLAN_SEEDS)
        self.assertEqual(p.cap["cap_hours"], json.loads(self.w.eval_cfg.read_text())["cap"]["hours"])
        self.assertEqual(p.run_code[E.RUN_FILE], sha((ROOT / E.RUN_FILE).read_bytes()))

    @contextlib.contextmanager
    def altered(self, change):
        saved = self.w.eval_cfg.read_bytes()
        doc = json.loads(saved)
        change(doc)
        self.w.write_eval_cfg(doc)
        try:
            yield
        finally:
            self.w.eval_cfg.write_bytes(saved)

    def test_cap_other_than_the_plan_refused(self):
        with self.altered(lambda d: d["cap"].__setitem__("hours", d["cap"]["hours"] + 0.5)):
            with self.assertRaisesRegex(FR.Failure, "differs from the cap derived from the plan"):
                self.pins()

    def test_cap_rule(self):
        """La regola dei tetti del piano sulla stima della valutazione: 2,1284 × 1,5 = 3,19 -> 3,5; un massimo già
        multiplo resta (2,0 × 1,5 = 3,0)."""
        cfg_pr = json.loads(self.w.cfg_path.read_text())
        rows = [{"dataset": "NUAA", "network": n, "patience": p, "new_trainings": 12, "evaluation_wall_hours": h}
                for (n, p), h in zip([(a, b) for a in AM.NETWORKS for b in D.PATIENCES], (0.12431274977228118, 0.12431274977228118,
                                                                                          0.9399119965992365, 0.9399119965992365))]
        with mock.patch.object(E.PAC, "plan_rows", lambda c, t: rows):
            self.assertEqual(E.cap_from_plan(cfg_pr, None)["cap_hours"], 3.5)
        rows2 = [{**r, "evaluation_wall_hours": 0.5} for r in rows]
        with mock.patch.object(E.PAC, "plan_rows", lambda c, t: rows2):
            self.assertEqual(E.cap_from_plan(cfg_pr, None)["cap_hours"], 3.0)
        rows3 = [{**r, "evaluation_wall_hours": None} for r in rows]
        with mock.patch.object(E.PAC, "plan_rows", lambda c, t: rows3):
            with self.assertRaisesRegex(FR.Failure, "not all estimated"):
                E.cap_from_plan(cfg_pr, None)

    def test_protocol_and_plan_pinned_and_anchored(self):
        p = self.w.base / "PROTOCOL.md"
        saved = p.read_bytes()
        try:
            p.write_bytes(saved + b"\n")
            with self.assertRaisesRegex(FR.Failure, "differs from the configuration"):
                self.pins()
            p.write_bytes(saved)
            # stesso file, riga del pacchetto diversa (pacchetto e configurazione dei run riallineati): scatta l'ancora
            self.w.package_sums({"PROTOCOL.md": "1" * 64})
            self.w.write_run_cfg(self.run_doc_keep_sums())
            self.w.write_estimator_record()
            with self.altered(lambda d: d.__setitem__("run_configuration", {"path": self.w.run_cfg.name, "sha256": sha(self.w.run_cfg.read_bytes())})):
                with self.assertRaisesRegex(FR.Failure, "line of the delivered package SHA256SUMS"):
                    self.pins()
        finally:
            p.write_bytes(saved)
            self.w.write_run_cfg(self.w.run_doc())
            self.w.write_estimator_record()

    def run_doc_keep_sums(self) -> dict:
        """La configurazione dei run con il SHA256SUMS del pacchetto così com'è (senza rigenerarlo)."""
        doc = json.loads(self.w.run_cfg.read_text())
        doc["trial_record"]["package_sums"]["sha256"] = sha((self.w.base / "package_SHA256SUMS").read_bytes())
        return doc

    def test_run_plan_regenerated(self):
        p = self.w.base / "run_plan.csv"
        saved = p.read_bytes()
        try:
            p.write_bytes(saved.replace(b"NUAA", b"NUAB", 1))
            self.w.package_sums({"plan/run_plan.csv": sha(p.read_bytes())})
            self.w.write_run_cfg(self.run_doc_keep_sums())
            self.w.write_estimator_record()
            with self.altered(lambda d: (d.__setitem__("run_configuration", {"path": self.w.run_cfg.name, "sha256": sha(self.w.run_cfg.read_bytes())}),
                                         d["preregistration"].__setitem__("run_plan", {"path": "run_plan.csv", "sha256": sha(p.read_bytes())}))):
                with self.assertRaisesRegex(FR.Failure, "run_plan.csv differs from the plan regenerated"):
                    self.pins()
        finally:
            p.write_bytes(saved)
            self.w.write_run_cfg(self.w.run_doc())
            self.w.write_estimator_record()

    def test_code_pins_compared_with_the_run_configuration(self):
        doc = json.loads(self.w.run_cfg.read_text())
        f = R.CODE_FILES[0]
        saved = self.w.run_cfg.read_bytes()
        try:
            with self.altered(lambda d: d["code"].__setitem__(f, "0" * 64)):
                with self.assertRaisesRegex(FR.Failure, "differ from the SHA-256 pinned in the configuration"):
                    self.pins()
            self.assertEqual(doc["code"][f], sha((ROOT / f).read_bytes()))
        finally:
            self.w.run_cfg.write_bytes(saved)

    def test_code_pinned_here_differs_from_the_run_configuration(self):
        """Un file del codice fissato qui uguale al file ma diverso dalla configurazione dei run: rifiutato."""
        saved = self.w.run_cfg.read_bytes()
        try:
            with mock.patch.object(E, "check_code", lambda cfg: {f: cfg["code"][f] for f in E.CODE_FILES}):
                doc2 = json.loads(self.w.eval_cfg.read_text())
                doc2["code"][R.CODE_FILES[1]] = "a" * 64
                saved_e = self.w.eval_cfg.read_bytes()
                self.w.write_eval_cfg(doc2)
                try:
                    with self.assertRaisesRegex(FR.Failure, "differ from the code of the run configuration"):
                        self.pins()
                finally:
                    self.w.eval_cfg.write_bytes(saved_e)
        finally:
            self.w.run_cfg.write_bytes(saved)

    def test_conclusions_quoted(self):
        """Le conclusioni del §6 citate dal testo con gli a capo del file (spazi normalizzati): un parser senza la
        normalizzazione non le trova, e qui è un'asserzione fallita, non un errore."""
        try:
            c = E.protocol_conclusions(CONCLUSIONS)
        except FR.Failure as e:
            self.fail(f"the conclusions written over several lines are not found: {e}")
        self.assertEqual(E.conclusion_for(c, D.CONTRAST_LOWER), "on these checkpoints the signed difference is algebraically lower with this "
                                                                 "re-estimation; no cause")
        self.assertEqual(E.conclusion_for(c, D.CONTRAST_HIGHER), "on these checkpoints the signed difference is algebraically higher with this "
                                                                  "re-estimation; no cause")
        self.assertEqual(E.conclusion_for(c, D.CONTRAST_UNRESOLVED), "this comparison does not show it")
        with self.assertRaisesRegex(FR.Failure, "unknown reading"):
            E.conclusion_for(c, "lower")
        with self.assertRaisesRegex(FR.Failure, "not found in the expected form"):
            E.protocol_conclusions(CONCLUSIONS.replace("lower [higher]", "lower"))
        with self.assertRaisesRegex(FR.Failure, "not found in the expected form"):
            E.protocol_conclusions(CONCLUSIONS.replace("otherwise", "else"))


# ============================================================================ C controllo (a) sul piano

class TestCPlanCheck(unittest.TestCase):
    """(a) Il controllo chiesto dal coordinatore sul piano, prima di qualunque lista o frame del test: un test per caso,
    costruito in modo che scatti quel controllo e nessun altro."""

    F, S = [1, 2, 3, 4], list(D.PLAN_SEEDS)

    def check(self, plan):
        return E.check_plan(plan, self.F, self.S)

    def test_complete_plan_passes(self):
        out = self.check(plan_full())
        self.assertEqual(list(out)[:2], list(E.PRIMARY))
        self.assertEqual({v["combinations"] for v in out.values()}, {12})

    def drop(self, net, state, fold=2, seed=43, rule="fixed_0.5", patience=4):
        return [e for e in plan_full() if not (e["network"], e["bn_state"], e["fold"], e["seed"], e["rule"], e["patience"]) ==
                (net, state, fold, seed, rule, patience)]

    def test_one_network_missing_in_the_ordinary_state_only(self):
        """Tolte le voci di LivenessNet per fold 2 seed 43 nel solo stato ordinario; ResNet-18 resta: la combinazione
        manca per una sola rete e il controllo lo dice per quella rete."""
        with self.assertRaisesRegex(FR.Failure, r"^primary analysis stopped before the test: 1 of the 12 fold-seed combinations missing for "
                                                r"LivenessNet in the ordinary BN state \(patience 4, threshold fixed_0.5\): fold 2 seed 43$"):
            self.check(self.drop("LivenessNet", "ordinary"))

    def test_one_network_missing_in_the_re_estimated_state_only(self):
        """Tolte le voci di ResNet-18 per fold 2 seed 43 nel solo stato ristimato; LivenessNet resta."""
        with self.assertRaisesRegex(FR.Failure, r"^primary analysis stopped before the test: 1 of the 12 fold-seed combinations missing for "
                                                r"ResNet-18 in the re-estimated BN state \(patience 4, threshold fixed_0.5\): fold 2 seed 43$"):
            self.check(self.drop("ResNet-18", "re-estimated"))

    def test_duplicated_combination(self):
        plan = plan_full()
        dup = next(e for e in plan if (e["network"], e["bn_state"], e["rule"], e["patience"], e["fold"], e["seed"]) ==
                   ("ResNet-18", "ordinary", "fixed_0.5", 4, 3, 44))
        with self.assertRaisesRegex(FR.Failure, r"^primary analysis stopped before the test: 1 of the 12 fold-seed combinations duplicated for "
                                                r"ResNet-18 in the ordinary BN state .*fold 3 seed 44$"):
            self.check(plan + [dict(dup)])

    def test_combination_outside_the_plan(self):
        extra = {**plan_full()[0], "fold": 5}
        with self.assertRaisesRegex(FR.Failure, r"^primary analysis stopped before the test: 1 of the 12 fold-seed combinations not in the plan "
                                                r"for LivenessNet .*fold 5 seed 42$"):
            self.check(plan_full() + [extra])

    def threshold(self, net, value, state="ordinary"):
        return [{**e, "threshold": value} if (e["network"], e["bn_state"], e["rule"], e["patience"]) == (net, state, "fixed_0.5", 4) else e
                for e in plan_full()]

    def test_threshold_not_0_5_for_livenessnet_only(self):
        with self.assertRaisesRegex(FR.Failure, r"^primary analysis stopped before the test: the effective threshold of LivenessNet is 0.4, "
                                                r"not 0.5, in the ordinary BN state \(patience 4, threshold fixed_0.5; 12 entries\)$"):
            self.check(self.threshold("LivenessNet", 0.4))

    def test_threshold_not_0_5_for_resnet18_only(self):
        with self.assertRaisesRegex(FR.Failure, r"^primary analysis stopped before the test: the effective threshold of ResNet-18 is 0.6, "
                                                r"not 0.5, in the re-estimated BN state"):
            self.check(self.threshold("ResNet-18", 0.6, "re-estimated"))

    def test_threshold_one_entry_and_odd_values(self):
        plan = plan_full()
        i = next(i for i, e in enumerate(plan) if (e["network"], e["bn_state"], e["rule"], e["patience"]) == ("ResNet-18", "ordinary", "fixed_0.5", 4))
        for bad in (0.5000000001, float("nan"), True, "0.5", None):
            p2 = [dict(e) for e in plan]
            p2[i]["threshold"] = bad
            with self.assertRaisesRegex(FR.Failure, r"effective threshold of ResNet-18 .* 1 entries"):
                self.check(p2)
        p3 = [dict(e) for e in plan]
        p3[i]["threshold"] = np.float64(0.5)
        self.check(p3)

    def test_secondary_conditions(self):
        """Le altre sei condizioni: completezza (messaggio «condition (...)»); soglia 0,5 solo nelle fisse, nessun
        controllo di valore nelle EER."""
        with self.assertRaisesRegex(FR.Failure, r"^condition \(12, 'ordinary', 'dev_eer'\) stopped before the test: 1 of the 12 fold-seed "
                                                r"combinations missing for LivenessNet"):
            self.check(self.drop("LivenessNet", "ordinary", rule="dev_eer", patience=12))
        p = [{**e, "threshold": 0.7} if (e["rule"], e["patience"]) == ("fixed_0.5", 12) and e["network"] == "ResNet-18" else e for e in plan_full()]
        with self.assertRaisesRegex(FR.Failure, r"^condition \(12, 'ordinary', 'fixed_0.5'\) stopped before the test: the effective threshold "
                                                r"of ResNet-18 is 0.7"):
            self.check(p)
        p = [{**e, "threshold": 0.91} if e["rule"] == "dev_eer" else e for e in plan_full()]
        self.check(p)

    def test_network_outside_the_plan(self):
        p = plan_full() + [{**plan_full()[0], "network": "VGG"}]
        with self.assertRaisesRegex(FR.Failure, "networks outside the plan"):
            self.check(p)


# ============================================================================ D controllo (b) sulle unità

class TestDUnitsCheck(unittest.TestCase):
    """(b) Lo stesso controllo sulle unità passate a D.primary_analysis, sui valori effettivi."""

    F, S = [1, 2, 3, 4], list(D.PLAN_SEEDS)

    def check(self, units, cond=D.PRIMARY_ORDINARY):
        return E.check_units(cond, units, self.F, self.S)

    def test_complete_units_pass(self):
        out = self.check(units_full())
        self.assertEqual(out["units"], 12)
        self.assertEqual(out["effective_threshold"], {"LivenessNet": [0.5], "ResNet-18": [0.5]})

    def test_threshold_livenessnet_only(self):
        u = units_full()
        for x in u:
            x["networks"]["LivenessNet"]["threshold"] = 0.45
        with self.assertRaisesRegex(FR.Failure, r"^primary analysis stopped: the effective threshold of LivenessNet is 0.45, not 0.5, in the "
                                                r"ordinary BN state \(patience 4, threshold fixed_0.5; 12 entries\)$"):
            self.check(u)

    def test_threshold_resnet18_only(self):
        u = units_full()
        u[5]["networks"]["ResNet-18"]["threshold"] = 0.55
        with self.assertRaisesRegex(FR.Failure, r"^primary analysis stopped: the effective threshold of ResNet-18 is 0.55, not 0.5, in the "
                                                r"re-estimated BN state \(patience 4, threshold fixed_0.5; 1 entries\)$"):
            self.check(u, D.PRIMARY_RE_ESTIMATED)

    def test_missing_unit_ordinary(self):
        u = [x for x in units_full() if x["unit"] != ("NUAA", 1, 42)]
        with self.assertRaisesRegex(FR.Failure, r"^primary analysis stopped: 1 of the 12 fold-seed combinations missing for LivenessNet in the "
                                                r"ordinary BN state .*fold 1 seed 42; 1 of the 12 fold-seed combinations missing for ResNet-18 in "
                                                r"the ordinary BN state .*fold 1 seed 42$"):
            self.check(u)

    def test_missing_unit_re_estimated(self):
        u = [x for x in units_full() if x["unit"] != ("NUAA", 4, 44)]
        with self.assertRaisesRegex(FR.Failure, r"^primary analysis stopped: 1 of the 12 fold-seed combinations missing for LivenessNet in the "
                                                r"re-estimated BN state .*fold 4 seed 44"):
            self.check(u, D.PRIMARY_RE_ESTIMATED)

    def test_network_missing_in_a_unit(self):
        u = units_full()
        del u[3]["networks"]["ResNet-18"]
        with self.assertRaisesRegex(FR.Failure, r"^primary analysis stopped: 1 of the 12 fold-seed combinations missing for ResNet-18 in the "
                                                r"ordinary BN state .*fold 2 seed 42$"):
            self.check(u)

    def test_duplicated_unit(self):
        u = units_full()
        u.append(u[0])
        with self.assertRaisesRegex(FR.Failure, r"^primary analysis stopped: 1 of the 12 fold-seed combinations duplicated for LivenessNet .*"
                                                r"fold 1 seed 42; 1 of the 12 fold-seed combinations duplicated for ResNet-18"):
            self.check(u)

    def test_unit_of_another_dataset(self):
        u = units_full()
        u[0] = {**u[0], "unit": ("CelebA-Spoof", 1, 42)}
        with self.assertRaisesRegex(FR.Failure, r"a unit is not a \(NUAA, fold, seed\) unit"):
            self.check(u)

    def test_threshold_read_from_the_unit_used_by_prepare_units(self):
        """La soglia controllata è il campo che D.prepare_units usa per le decisioni: con 0,5 nel campo le decisioni di
        D sono quelle a 0,5; con un altro valore il controllo ferma prima di D."""
        u = units_full()
        pus = D.prepare_units(u)
        c = pus[0]["counts"]["LivenessNet"]
        y, s = u[0]["networks"]["LivenessNet"]["y"], u[0]["networks"]["LivenessNet"]["s"]
        self.assertEqual(int(c["rej"].sum()), int(np.sum((y == 0) & (s > 0.5))))
        u[0]["networks"]["LivenessNet"]["threshold"] = 0.2
        with mock.patch.object(E.D, "prepare_units", side_effect=AssertionError("prepare_units reached")):
            with self.assertRaisesRegex(FR.Failure, "effective threshold of LivenessNet is 0.2"):
                self.check(u)

    def test_eer_condition_completeness_only(self):
        u = units_full(threshold=0.83)
        out = self.check(u, (4, "ordinary", "dev_eer"))
        self.assertEqual(out["effective_threshold"], "per model (dev EER)")
        with self.assertRaisesRegex(FR.Failure, r"^condition \(4, 'ordinary', 'dev_eer'\) stopped: 1 of the 12"):
            self.check(u[1:], (4, "ordinary", "dev_eer"))


# ============================================================================ E flusso completo (valutatore finto)

class TestEFlow(unittest.TestCase):
    def setUp(self):
        self.w = world()

    def test_pass_outputs(self):
        f = flow_ok(self)
        self.assertEqual(f["rc"], 0, (f["out"] + f["err"])[-4000:])
        run = f["folder"]
        self.assertEqual(sorted(os.listdir(run)), ["public", "restricted", "run.log", "status.json"])
        self.assertEqual(sorted(os.listdir(run / "public")), sorted(E.PUBLIC_OUTPUTS))
        names = [f"{R.RUN_TAG[r]}_{PAC.training_name(n, fo, s)}{E.TEST_SUFFIX}" for r in R.RUNS for n, fo, s in PAC.plan_order([1, 2, 3, 4], [42, 43, 44])]
        self.assertEqual(sorted(os.listdir(run / "restricted")), sorted(names + list(E.RESTRICTED_FIXED)))
        for sub in ("public", "restricted"):
            sums = FR.read_sums((run / sub / "SHA256SUMS").read_bytes(), "s")
            self.assertEqual(sorted(sums), sorted(n for n in os.listdir(run / sub) if n != "SHA256SUMS"))
            for n, h in sums.items():
                self.assertEqual(sha((run / sub / n).read_bytes()), h)
        st = json.loads((run / "status.json").read_text())
        self.assertEqual(st["status"], "PASS")
        self.assertEqual(st["public_sha256sums_sha256"], sha((run / "public" / "SHA256SUMS").read_bytes()))
        self.assertEqual(st["restricted_sha256sums_sha256"], sha((run / "restricted" / "SHA256SUMS").read_bytes()))
        self.assertEqual(st["run_log_sha256"], sha((run / "run.log").read_bytes()))
        for k in ("started", "finished", "wall_seconds", "process_cpu_seconds", "threads", "threads_rule", "machine_rule", "commit", "dirty",
                  "config_sha256", "code_sha256", "inputs_sha256", "cap_hours", "cap_unit", "runs", "attempt", "dev_reproduction", "checks"):
            self.assertIn(k, st)
        self.assertTrue(st["test_read_started"])
        self.assertEqual(st["threads"]["effective"], {"intra_op": 10, "inter_op": 10})
        self.assertEqual(st["code_sha256"][E.THIS_FILE], sha((ROOT / E.THIS_FILE).read_bytes()))
        self.assertEqual(st["runs"]["nuaa_patience_4"]["run_id"], self.w.run_dirs["nuaa_patience_4"].name)
        self.assertEqual(len(st["dev_reproduction"]), 96)

    def test_order_dev_before_test_lists(self):
        """Le 48 riproduzioni del dev (due stati ciascuna) prima di qualunque lista del test (schema dei fold vuoto a ogni
        chiamata sul dev); poi il test fold per fold, ciascun fold assemblato una volta e valutato dai suoi 12 modelli."""
        f = flow_ok(self)
        calls = f["log"].calls
        devs = [c for c in calls if c[0] == "dev"]
        tests = [c for c in calls if c[0] == "test"]
        self.assertEqual(len(devs), 48)
        self.assertEqual(len(tests), 48)
        self.assertEqual({c[2] for c in devs}, {0})
        self.assertEqual({c[2] for c in tests}, {4})
        self.assertLess(max(i for i, c in enumerate(calls) if c[0] == "dev"), min(i for i, c in enumerate(calls) if c[0] == "test"))
        self.assertEqual([c[1] for c in devs], [f"{R.RUN_TAG[r]}_{PAC.training_name(n, fo, s)}" for r in R.RUNS
                                                for n, fo, s in PAC.plan_order([1, 2, 3, 4], [42, 43, 44])])
        folds = [int(c[1].split("fold")[1][0]) for c in tests]
        self.assertEqual(folds, sorted(folds))

    def test_test_assembled_once_per_fold(self):
        w = self.w
        real = FR.assemble
        test_keys = set()
        calls = []

        def counting(store, rows):
            out = real(store, rows)
            calls.append(len(rows))
            return out
        with mock.patch.object(E.FR, "assemble", counting):
            rc, o, e = run_eval(w.eval_argv(out=fresh_out(w)), eval_hooks(w))
        self.assertEqual(rc, 0, (o + e)[-3000:])
        n_test = json.loads(w.cfg_path.read_text())["plan"]["datasets"][0]["frames_per_unit"]["test_per_split"]
        self.assertEqual(sorted(c for c in calls if c in n_test), sorted(n_test))
        self.assertEqual(len([c for c in calls if c in n_test]), 4, (calls, test_keys))

    def test_report_leads_with_the_primary_analysis(self):
        f = flow_ok(self)
        text = (f["folder"] / "public" / "report.md").read_text()
        heads = [l for l in text.splitlines() if l.startswith("#")]
        self.assertEqual(heads[1], "## Primary analysis (requested by the coordinator: patience 4, threshold 0.5)")
        self.assertEqual(heads[2], "### Per seed (reported apart; not part of the reading)")
        body = text.split(heads[1])[1]
        self.assertLess(body.index("| ACER gap, ordinary BN |"), body.index("| ACER gap, re-estimated BN |"))
        self.assertLess(body.index("| ACER gap, re-estimated BN |"), body.index("| contrast |"))
        self.assertLess(body.index("| contrast |"), body.index("Reading of the contrast"))
        self.assertLess(body.index("Conclusion (written before the test; quoted from PROTOCOL.md §6, with the alternative resolved)"),
                        body.index("### Per seed"))
        self.assertIn("ACER, APCER and BPCER are proportions between 0 and 1 (not percentages)", body)
        self.assertIn("of the 16 test subjects pooled over the 4 folds; in parentheses the valid resamples", body)
        self.assertIn("Each seed has 4 units (one per fold) and the bootstrap of each seed uses the same 16 test subjects.", body)
        doc = json.loads((f["folder"] / "public" / "results.json").read_text())
        keys = list(doc)
        self.assertEqual(keys[:5], ["schema", "run_id", "status", "official", "primary_analysis"])
        self.assertEqual(list(doc["primary_analysis"])[:7], ["requested", "conditions", "definition", "gap_ordinary", "gap_re_estimated", "contrast",
                                                             "conclusion"])
        reading = doc["primary_analysis"]["contrast"]["reading"]
        self.assertEqual(doc["primary_analysis"]["conclusion"], E.conclusion_for(E.protocol_conclusions(CONCLUSIONS), reading))
        self.assertIn(doc["primary_analysis"]["conclusion"], text)
        self.assertEqual(sorted(doc["primary_analysis"]["per_seed"]), ["42", "43", "44"])

    def test_primary_results_recomputed_from_the_restricted_files(self):
        """Ricalcolo indipendente: dai soli file riservati per frame, ACER per unità contati qui (frazioni esatte) e
        l'analisi primaria di D sulle unità ricostruite; uguali a results.json e units.csv."""
        f = flow_ok(self)
        rdir = f["folder"] / "restricted"
        units = {c: {} for c in E.PRIMARY}
        acer = {}
        for p in sorted(rdir.glob(f"p4_*{E.TEST_SUFFIX}")):
            rows = read_csv(p)
            for state in D.BN_STATES:
                rs = [r for r in rows if r["bn_state"] == state]
                y = np.array([int(r["label"]) for r in rs])
                s = np.array([float(r["score_attack"]) for r in rs])
                net, fold, seed = rs[0]["network"], int(rs[0]["fold"]), int(rs[0]["train_seed"])
                dec = s > 0.5
                self.assertEqual([r["prediction_05"] for r in rs], ["1" if d else "0" for d in dec])
                apcer = Fraction(int(np.sum((y == 1) & ~dec)), int(np.sum(y == 1)))
                bpcer = Fraction(int(np.sum((y == 0) & dec)), int(np.sum(y == 0)))
                acer[(state, net, fold, seed)] = (apcer + bpcer) / 2
                cond = (4, state, "fixed_0.5")
                u = units[cond].setdefault((fold, seed), {"unit": ("NUAA", fold, seed), "networks": {}})
                u["networks"][net] = {"y": y, "s": s, "subject": np.array([r["subject_id"] for r in rs]), "threshold": 0.5,
                                      "sample_id": [r["sample_id"] for r in rs]}
        pa = D.primary_analysis({c: list(v.values()) for c, v in units.items()})
        doc = json.loads((f["folder"] / "public" / "results.json").read_text())["primary_analysis"]
        for k in ("gap_ordinary", "gap_re_estimated"):
            self.assertEqual(doc[k]["mean_delta"], pa[k]["mean_delta"])
            self.assertEqual(doc[k]["interval"], pa[k]["interval"])
        self.assertEqual(doc["contrast"]["mean"], pa["contrast"]["mean"])
        self.assertEqual(doc["contrast"]["interval"], pa["contrast"]["interval"])
        self.assertEqual(doc["contrast"]["reading"], pa["contrast"]["reading"])
        ur = [r for r in read_csv(f["folder"] / "public" / "units.csv") if (r["patience"], r["threshold"]) == ("4", "fixed_0.5")]
        self.assertEqual(len(ur), 24)
        for r in ur:
            key = (r["bn_state"], int(r["fold"]), int(r["seed"]))
            d = acer[(key[0], "ResNet-18", key[1], key[2])] - acer[(key[0], "LivenessNet", key[1], key[2])]
            self.assertEqual(Fraction(r["delta_acer"]), d)
            self.assertEqual(r["tie"], "1" if d == 0 else "0")
            self.assertEqual((r["livenessnet_threshold"], r["resnet18_threshold"]), ("0.5", "0.5"))
        means = {s: float(sum(Fraction(r["delta_acer"]) for r in ur if r["bn_state"] == s) / 12) for s in D.BN_STATES}
        self.assertEqual(means["ordinary"], doc["gap_ordinary"]["mean_delta"]["acer"])

    def test_eer_thresholds_from_the_saved_dev(self):
        """Nelle condizioni EER la soglia di ogni rete è D.condition_threshold sul dev salvato dal run nello stesso stato."""
        f = flow_ok(self)
        ur = read_csv(f["folder"] / "public" / "units.csv")
        run = self.w.run_dirs["nuaa_patience_12"]
        rows = read_csv(run / "restricted" / "ResNet-18_fold3_seed44_dev_re-estimated.csv")
        thr = D.condition_threshold("dev_eer", [int(r["label"]) for r in rows], [float(r["score_attack"]) for r in rows])
        got = [r for r in ur if (r["patience"], r["bn_state"], r["threshold"], r["fold"], r["seed"]) == ("12", "re-estimated", "dev_eer", "3", "44")]
        self.assertEqual(len(got), 1)
        self.assertEqual(float(got[0]["resnet18_threshold"]), thr)
        tr = read_csv(f["folder"] / "restricted" / f"p12_ResNet-18_fold3_seed44{E.TEST_SUFFIX}")
        self.assertEqual({r["threshold_dev_eer"] for r in tr if r["bn_state"] == "re-estimated"}, {repr(thr)})

    def test_primary_analysis_on_the_checked_units(self):
        """L'analisi primaria è calcolata sugli stessi oggetti controllati (b): punteggi di LivenessNet alterati nelle unità
        (valori validi, soglia 0,5) -> il divario pubblicato è quello di D.primary_analysis sulle unità alterate, diverso da
        quello del flusso senza alterazione."""
        seen = {}

        def hook(conds):
            for u in conds[D.PRIMARY_ORDINARY]:
                n = u["networks"]["LivenessNet"]
                u["networks"]["LivenessNet"] = {**n, "s": np.clip(1 - np.asarray(n["s"]), 0, 1)}
            seen["pa"] = D.primary_analysis({c: conds[c] for c in E.PRIMARY})
            return conds
        out = fresh_out(self.w)
        rc, o, e = run_eval(self.w.eval_argv(out=out), eval_hooks(self.w, units_hook=hook))
        self.assertEqual(rc, 0, (o + e)[-2000:])
        got = json.loads((out / EVAL_ID / "public" / "results.json").read_text())["primary_analysis"]
        self.assertEqual(got["gap_ordinary"]["mean_delta"], seen["pa"]["gap_ordinary"]["mean_delta"])
        base = json.loads((flow_ok(self)["folder"] / "public" / "results.json").read_text())["primary_analysis"]
        self.assertNotEqual(got["gap_ordinary"]["mean_delta"]["acer"], base["gap_ordinary"]["mean_delta"]["acer"])

    def test_conditions_comparisons_and_sensitivity(self):
        f = flow_ok(self)
        doc = json.loads((f["folder"] / "public" / "results.json").read_text())
        self.assertEqual([tuple(c["condition"]) for c in doc["conditions"]], list(D.CONDITIONS))
        self.assertEqual([c["label"] for c in doc["conditions"]].count("primary"), 2)
        self.assertEqual(len(doc["comparisons"]), 12)
        self.assertEqual(doc["comparisons"][0]["contrast_mean"], doc["primary_analysis"]["contrast"]["mean"])
        self.assertEqual(len(doc["subject_sensitivity"]), 8)
        keys = ["condition", "full_mean_delta_acer", "full_sign", "min", "max", "n_sign_differs", "n_not_available", "n_excluded",
                "n_attack_after_exclusion", "n_bona_fide_after_exclusion"]
        for x in doc["subject_sensitivity"]:
            self.assertEqual(list(x), keys)                 # solo aggregati: nessuna riga per soggetto o per esclusione
            self.assertEqual(x["n_excluded"], 16)
        lo = read_csv(f["folder"] / "restricted" / "loso_values.csv")
        self.assertEqual(len(lo), 8 * 16)
        self.assertEqual(set(r["excluded_subject"] for r in lo), set(CFR.SUBJECTS))
        lo4 = [r for r in lo if (r["patience"], r["bn_state"], r["threshold"]) == ("4", "ordinary", "fixed_0.5")]
        s4 = doc["subject_sensitivity"][0]
        self.assertEqual(min(float(r["mean_delta_acer"]) for r in lo4), s4["min"])
        self.assertEqual([min(int(r["n_attack_total"]) for r in lo4), max(int(r["n_attack_total"]) for r in lo4)], s4["n_attack_after_exclusion"])
        self.assertEqual([min(int(r["n_bona_fide_total"]) for r in lo4), max(int(r["n_bona_fide_total"]) for r in lo4)],
                         s4["n_bona_fide_after_exclusion"])
        self.assertLess(s4["n_bona_fide_after_exclusion"][0], s4["n_bona_fide_after_exclusion"][1])    # un intervallo, non un valore
        oc = doc["one_class"]
        self.assertEqual(len(oc), 8)
        self.assertEqual(sum(o["evaluable"] for o in oc), 96)
        # stati della BN diversi nel mondo: LivenessNet ordinaria dà tutto bona fide (0,3), la ristimata no
        got = {(o["network"], o["patience"], o["bn_state"]): o["text"] for o in oc}
        self.assertEqual(got[("LivenessNet", 4, "ordinary")], "12 of 12 evaluable")
        self.assertEqual(got[("LivenessNet", 4, "re-estimated")], "0 of 12 evaluable")
        self.assertEqual(got[("ResNet-18", 12, "re-estimated")], "0 of 12 evaluable")
        self.assertEqual(doc["degenerate_apart"]["counts"], [{"network": n, "patience": p, "degenerate": 0, "trainings": 12}
                                                              for n in AM.NETWORKS for p in D.PATIENCES])
        cr = read_csv(f["folder"] / "public" / "conditions.csv")
        self.assertEqual(len(cr), 8 * 4)
        self.assertEqual({r["label"] for r in cr if (r["patience"], r["threshold"]) == ("4", "fixed_0.5")}, {"primary"})
        checks = doc["checks_before_primary_analysis"]
        self.assertEqual(len(checks["units"]), 8)
        self.assertEqual(checks["units"][str(list(D.PRIMARY_ORDINARY))]["effective_threshold"], {"LivenessNet": [0.5], "ResNet-18": [0.5]})
        self.assertEqual(len(doc["models"]), 48)

    def test_public_outputs_hold_no_identifier(self):
        f = flow_ok(self)
        keys = {r["key"] for r in self.w_rows()}
        for n in E.PUBLIC_OUTPUTS:
            text = (f["folder"] / "public" / n).read_text()
            self.assertNotIn(str(self.w.tmp), text, n)
            self.assertFalse([k for k in keys if k in text or k.split("/", 1)[1] in text], n)
            for s in CFR.SUBJECTS:
                self.assertNotRegex(text, rf"(?<![\w.]){s}(?![\w.])", n)
        st = (f["folder"] / "status.json").read_text()
        self.assertNotIn(str(self.w.tmp), st)
        self.assertNotIn(str(Path.home()), (f["folder"] / "run.log").read_text())

    def w_rows(self):
        cfg, _ = E.load_config(self.w.eval_cfg)
        with injected(**CR.base_hooks(self.w)):
            pins = E.load_pins(cfg, self.w.base)
            inp = PAC.load_inputs(pins.cfg_pr, pins.pcfg, self.w.base)
            FR.load_fold_manifests(SR.fr_view(pins.pcfg), self.w.base, inp)
        return [r for f in inp.folds for r in inp.folds[f]["test"]]

    def test_restricted_test_scores_carry_frame_ids(self):
        f = flow_ok(self)
        rows = read_csv(f["folder"] / "restricted" / f"p4_LivenessNet_fold2_seed43{E.TEST_SUFFIX}")
        test = sorted([r for r in self.w_rows() if r["key"] in {x["sample_id"] for x in rows}], key=lambda r: r["key"])
        n = len(test)
        self.assertEqual(len(rows), 2 * n)
        self.assertEqual([r["sample_id"] for r in rows[:n]], [r["key"] for r in test])
        self.assertEqual([r["sample_id"] for r in rows[n:]], [r["key"] for r in test])
        self.assertEqual([r["bn_state"] for r in rows], ["ordinary"] * n + ["re-estimated"] * n)
        self.assertEqual({r["role"] for r in rows}, {"test"})
        self.assertEqual([r["subject_id"] for r in rows[:n]], [r["subject"] for r in test])


# ============================================================================ F controlli chiesti, nel flusso

class TestFRequestedChecksInTheFlow(unittest.TestCase):
    """Il controllo chiesto dal coordinatore nel comando: ogni caso con il suo messaggio, l'analisi primaria non
    calcolata, esito FAIL e nessun file di risultati."""

    def setUp(self):
        self.w = world()

    def failing(self, msg, plan_hook=None, units_hook=None, primary_calls=0):
        log = Log()
        out = fresh_out(self.w)
        calls = []
        real = D.primary_analysis

        def counted(c):
            calls.append(1)
            return real(c)
        with mock.patch.object(E.D, "primary_analysis", counted):
            rc, o, e = run_eval(self.w.eval_argv(out=out), eval_hooks(self.w, log, plan_hook=plan_hook, units_hook=units_hook))
        self.assertEqual(rc, 1, (o + e)[-3000:])
        self.assertRegex(e, msg)
        folder = out / EVAL_ID
        no_results(self, folder)
        st = json.loads((folder / "status.json").read_text())
        self.assertEqual(st["status"], "FAIL")
        self.assertRegex(st["reason"], msg)
        self.assertEqual(len(calls), primary_calls)
        return st, log

    def plan_drop(self, net, state, fold=2, seed=43):
        return lambda plan: [e for e in plan if not ((e["network"], e["bn_state"], e["fold"], e["seed"], e["rule"], e["patience"]) ==
                                                     (net, state, fold, seed, "fixed_0.5", 4))]

    def plan_threshold(self, net, value, state="ordinary"):
        return lambda plan: [{**e, "threshold": value} if (e["network"], e["bn_state"], e["rule"], e["patience"]) == (net, state, "fixed_0.5", 4)
                             else e for e in plan]

    def assert_test_not_read(self, st, log):
        self.assertFalse(st["test_read_started"])
        self.assertEqual([c for c in log.calls if c[0] != "dev"], [])
        self.assertEqual(log.calls, [])                    # il controllo (a) precede anche la riproduzione del dev

    def test_a_one_network_missing_in_the_ordinary_state_only(self):
        """Nel flusso: tolte le voci di LivenessNet per una combinazione nel solo stato ordinario, ResNet-18 resta."""
        st, log = self.failing(r"primary analysis stopped before the test: 1 of the 12 fold-seed combinations missing for LivenessNet in the "
                               r"ordinary BN state", plan_hook=self.plan_drop("LivenessNet", "ordinary"))
        self.assert_test_not_read(st, log)

    def test_a_one_network_missing_in_the_re_estimated_state_only(self):
        """Nel flusso: tolte le voci di ResNet-18 per una combinazione nel solo stato ristimato, LivenessNet resta."""
        st, log = self.failing(r"primary analysis stopped before the test: 1 of the 12 fold-seed combinations missing for ResNet-18 in the "
                               r"re-estimated BN state", plan_hook=self.plan_drop("ResNet-18", "re-estimated"))
        self.assert_test_not_read(st, log)

    def test_a_duplicated_combination(self):
        def dup(plan):
            e = next(x for x in plan if (x["network"], x["bn_state"], x["rule"], x["patience"]) == ("LivenessNet", "re-estimated", "fixed_0.5", 4))
            return plan + [dict(e)]
        st, log = self.failing(r"primary analysis stopped before the test: 1 of the 12 fold-seed combinations duplicated for LivenessNet in the "
                               r"re-estimated BN state", plan_hook=dup)
        self.assert_test_not_read(st, log)

    def test_a_threshold_livenessnet_only(self):
        st, log = self.failing(r"primary analysis stopped before the test: the effective threshold of LivenessNet is 0.4, not 0.5",
                               plan_hook=self.plan_threshold("LivenessNet", 0.4))
        self.assert_test_not_read(st, log)

    def test_a_threshold_resnet18_only(self):
        st, log = self.failing(r"primary analysis stopped before the test: the effective threshold of ResNet-18 is 0.6, not 0.5, in the "
                               r"re-estimated BN state", plan_hook=self.plan_threshold("ResNet-18", 0.6, "re-estimated"))
        self.assert_test_not_read(st, log)

    def units_threshold(self, net, value, cond=D.PRIMARY_ORDINARY):
        def hook(conds):
            for u in conds[cond]:
                u["networks"][net] = {**u["networks"][net], "threshold": value}
            return conds
        return hook

    def test_b_threshold_livenessnet_only(self):
        self.failing(r"primary analysis stopped: the effective threshold of LivenessNet is 0.4, not 0.5, in the ordinary BN state",
                     units_hook=self.units_threshold("LivenessNet", 0.4))

    def test_b_threshold_resnet18_only(self):
        self.failing(r"primary analysis stopped: the effective threshold of ResNet-18 is 0.45, not 0.5, in the re-estimated BN state",
                     units_hook=self.units_threshold("ResNet-18", 0.45, D.PRIMARY_RE_ESTIMATED))

    def test_b_missing_in_the_ordinary_state_only(self):
        def hook(conds):
            conds[D.PRIMARY_ORDINARY] = [u for u in conds[D.PRIMARY_ORDINARY] if u["unit"] != ("NUAA", 3, 43)]
            return conds
        self.failing(r"primary analysis stopped: 1 of the 12 fold-seed combinations missing for LivenessNet in the ordinary BN state .*fold 3 "
                     r"seed 43", units_hook=hook)

    def test_b_missing_in_the_re_estimated_state_only(self):
        def hook(conds):
            conds[D.PRIMARY_RE_ESTIMATED] = [u for u in conds[D.PRIMARY_RE_ESTIMATED] if u["unit"] != ("NUAA", 1, 44)]
            return conds
        self.failing(r"primary analysis stopped: 1 of the 12 fold-seed combinations missing for LivenessNet in the re-estimated BN state .*fold 1 "
                     r"seed 44", units_hook=hook)

    def test_b_duplicated_combination(self):
        def hook(conds):
            conds[D.PRIMARY_ORDINARY] = conds[D.PRIMARY_ORDINARY] + [conds[D.PRIMARY_ORDINARY][0]]
            return conds
        self.failing(r"primary analysis stopped: 1 of the 12 fold-seed combinations duplicated for LivenessNet in the ordinary BN state",
                     units_hook=hook)

    def test_b_network_missing_in_a_unit(self):
        def hook(conds):
            u = conds[D.PRIMARY_RE_ESTIMATED][4]
            conds[D.PRIMARY_RE_ESTIMATED][4] = {**u, "networks": {"LivenessNet": u["networks"]["LivenessNet"]}}
            return conds
        self.failing(r"primary analysis stopped: 1 of the 12 fold-seed combinations missing for ResNet-18 in the re-estimated BN state",
                     units_hook=hook)

    def test_secondary_condition_incomplete(self):
        """Una condizione secondaria incompleta: l'analisi primaria è già calcolata, ma il comando si ferma e non scrive
        nulla (tutto o niente)."""
        def hook(conds):
            conds[(12, "ordinary", "dev_eer")] = conds[(12, "ordinary", "dev_eer")][1:]
            return conds
        self.failing(r"condition \(12, 'ordinary', 'dev_eer'\) stopped: 1 of the 12 fold-seed combinations missing", units_hook=hook, primary_calls=1)

    def test_frame_ids_differ_between_the_networks(self):
        """Stessi soggetti ed etichette, un ID diverso per una rete in un'unità: D.prepare_units ferma il confronto."""
        def hook(conds):
            u = conds[D.PRIMARY_ORDINARY][2]
            ids = list(u["networks"]["ResNet-18"]["sample_id"])
            ids[0] = ids[0] + "x"
            u["networks"]["ResNet-18"] = {**u["networks"]["ResNet-18"], "sample_id": ids}
            return conds
        self.failing(r"the two networks are not scored on the same frame identifiers in the same order", units_hook=hook)

    def test_frame_ids_differ_between_the_bn_states(self):
        """Gli stessi ID alterati per le due reti di tutte le unità di un fold nello stato ristimato: ogni condizione da
        sola passa, il contrasto fra i due stati della BN (D.same_frames) ferma."""
        def hook(conds):
            for u in conds[D.PRIMARY_RE_ESTIMATED]:
                if u["unit"][1] == 2:
                    for n in AM.NETWORKS:
                        ids = list(u["networks"][n]["sample_id"])
                        ids[0], ids[1] = ids[1], ids[0]
                        u["networks"][n] = {**u["networks"][n], "sample_id": ids}
            return conds
        self.failing(r"the two conditions are not scored on the same frame identifiers in the same order", units_hook=hook, primary_calls=1)


# ============================================================================ G cartelle dei run

class TestGRunFolders(unittest.TestCase):
    """Validazione delle cartelle dei run prima di ogni uso; le fixture riallineano i checksum, così scatta la regola."""

    def setUp(self):
        self.w = world()

    def refused(self, runs, msg, cmd="check-inputs"):
        out = fresh_out(self.w)
        rc, o, e = run_eval(self.w.eval_argv(cmd, runs=runs, out=out), eval_hooks(self.w))
        self.assertEqual(rc, 1, (o + e)[-3000:])
        self.assertRegex(e, msg)
        self.assertEqual(os.listdir(out), [])
        return e

    def test_check_inputs_pass(self):
        rc, o, e = run_eval(self.w.eval_argv("check-inputs"), eval_hooks(self.w))
        self.assertEqual(rc, 0, (o + e)[-3000:])
        self.assertIn("check (a) passed on 192 plan entries (8 conditions)", o)
        self.assertIn("RESULT: PASS (check-inputs)", o)
        self.assertIn("TensorFlow not imported; nothing written", o)

    def test_check_inputs_without_runs_and_with_one(self):
        a = self.w.eval_argv("check-inputs")[:5]
        rc, o, e = run_eval(a, eval_hooks(self.w))
        self.assertEqual(rc, 0, e[-2000:])
        self.assertIn("run folders not given", o)
        rc, o, e = run_eval(a + ["--run-p4", str(self.w.run_dirs["nuaa_patience_4"])], eval_hooks(self.w))
        self.assertEqual(rc, 1)
        self.assertIn("needs both --run-p4 and --run-p12", e)

    def test_swapped_and_same_folders(self):
        r = self.w.run_dirs
        self.refused({"nuaa_patience_4": r["nuaa_patience_12"], "nuaa_patience_12": r["nuaa_patience_4"]}, "is not the folder of a run nuaa_patience_4")
        r2 = self.w.copy_runs()
        p = Path(tempfile.mkdtemp(dir=self.w.tmp)) / r2["nuaa_patience_12"].name
        os.symlink(r2["nuaa_patience_12"], p)
        self.refused({"nuaa_patience_4": r2["nuaa_patience_4"], "nuaa_patience_12": p}, "missing, not a folder or a symbolic link")

    def test_status_not_pass(self):
        r = self.w.copy_runs()
        rewrite_status(r["nuaa_patience_12"], lambda st: st.__setitem__("status", "INCOMPLETE"))
        self.refused(r, "status.json is not a PASS of nuaa_patience_12")

    def test_other_configuration_or_code(self):
        r = self.w.copy_runs()
        rewrite_status(r["nuaa_patience_4"], lambda st: st.__setitem__("config_sha256", "0" * 64))
        self.refused(r, "produced with another configuration of the runs")
        r = self.w.copy_runs()
        rewrite_status(r["nuaa_patience_4"], lambda st: st["code_sha256"].__setitem__(E.RUN_FILE, "0" * 64))
        self.refused(r, "produced with other code")

    def test_stopped_or_incomplete_run(self):
        for change in (lambda st: st.__setitem__("stopped", {"step": "x"}), lambda st: st.__setitem__("missing_phase_1", ["a"]),
                       lambda st: st.__setitem__("evaluator_mismatches", [{"name": "a"}])):
            r = self.w.copy_runs()
            rewrite_status(r["nuaa_patience_4"], change)
            self.refused(r, "stopped, with a model missing or with an evaluator mismatch")
        r = self.w.copy_runs()
        rewrite_status(r["nuaa_patience_4"], lambda st: st["complete_phase_2"].pop())
        self.refused(r, "complete models differ from the 24 positions")

    def test_machine_and_threads_of_the_run(self):
        r = self.w.copy_runs()
        rewrite_status(r["nuaa_patience_4"], lambda st: st["threads_rule"].__setitem__("equal", False))
        self.refused(r, "machine and thread rule was not met")
        r = self.w.copy_runs()
        rewrite_status(r["nuaa_patience_4"], lambda st: st["threads"].__setitem__("effective", {"intra_op": 15, "inter_op": 10}))
        self.refused(r, "effective threads")

    def test_dirty_run(self):
        r = self.w.copy_runs()
        rewrite_status(r["nuaa_patience_4"], lambda st: st.__setitem__("dirty", True))
        self.refused(r, "work tree that was not clean")

    def test_unofficial_runs_refused_by_an_official_evaluation(self):
        """Valutazione ufficiale (radice results/archeval del repository, simulata per identità con alias_of) con run non
        ufficiali: rifiutata prima di creare qualunque cartella."""
        w = self.w
        root = w.tmp / "fake_repo_eval_root"
        root.mkdir(exist_ok=True)
        with CR.alias_of(ROOT / E.DEFAULT_OUT_ROOT, root):
            rc, o, e = run_eval(w.eval_argv(out=ROOT / E.DEFAULT_OUT_ROOT, unofficial=False), eval_hooks(w))
        self.assertEqual(rc, 1)
        self.assertIn("an official evaluation needs an official run in results/archrun", e)
        self.assertEqual(os.listdir(root), [])

    def test_file_differs_from_its_sums(self):
        r = self.w.copy_runs()
        p = r["nuaa_patience_4"] / "restricted" / "ResNet-18_fold1_seed42.keras"
        p.write_bytes(p.read_bytes() + b"x")
        self.refused(r, r"restricted/ResNet-18_fold1_seed42.keras differs from its SHA256SUMS line")
        r = self.w.copy_runs()
        (r["nuaa_patience_12"] / "public" / "report.md").write_text("x", encoding="utf-8")
        self.refused(r, r"public/report.md differs from its SHA256SUMS line")

    def test_sums_of_the_status(self):
        r = self.w.copy_runs()
        rewrite_status(r["nuaa_patience_4"], lambda st: st.__setitem__("restricted_sha256sums_sha256", "0" * 64))
        self.refused(r, "the restricted SHA256SUMS differs from restricted_sha256sums_sha256")
        r = self.w.copy_runs()
        (r["nuaa_patience_4"] / "run.log").write_text("x", encoding="utf-8")
        self.refused(r, "run.log differs from run_log_sha256")

    def test_file_missing_from_the_sums(self):
        r = self.w.copy_runs()
        run = r["nuaa_patience_4"]
        sums = FR.read_sums((run / "restricted" / "SHA256SUMS").read_bytes(), "s")
        del sums["LivenessNet_fold3_seed43_dev_re-estimated.csv"]
        text = "".join(f"{h}  {n}\n" for n, h in sums.items())
        (run / "restricted" / "SHA256SUMS").write_text(text, encoding="utf-8")
        rewrite_status(run, lambda st: st.__setitem__("restricted_sha256sums_sha256", sha(text.encode())))
        self.refused(r, "restricted SHA256SUMS does not list exactly the 144 restricted files")

    def test_summary_of_another_run(self):
        r = self.w.copy_runs()
        rewrite(r["nuaa_patience_12"], "public", "summary.json", json_change(lambda d: d.__setitem__("patience", 4)))
        self.refused(r, "summary.json is not the PASS summary")

    def test_card_of_another_model(self):
        r = self.w.copy_runs()
        rewrite(r["nuaa_patience_4"], "restricted", "LivenessNet_fold2_seed42_card.json", json_change(lambda d: d.__setitem__("train_seed", 43)))
        self.refused(r, "LivenessNet_fold2_seed42_card.json differs from its position")
        r = self.w.copy_runs()
        rewrite(r["nuaa_patience_12"], "restricted", "ResNet-18_fold2_seed42_card.json", json_change(lambda d: d.__setitem__("early_stopping_patience", 4)))
        self.refused(r, "ResNet-18_fold2_seed42_card.json differs from its position")

    def test_statistics_file(self):
        r = self.w.copy_runs()
        rewrite(r["nuaa_patience_4"], "restricted", "ResNet-18_fold1_seed43_bn_statistics.json",
                json_change(lambda d: d.__setitem__("order", list(reversed(d["order"])))))
        self.refused(r, "differs from the model, its checkpoint or the order of its BN layers")
        r = self.w.copy_runs()
        rewrite(r["nuaa_patience_4"], "restricted", "ResNet-18_fold1_seed43_bn_statistics.json",
                json_change(lambda d: d.__setitem__("layers", dict(reversed(list(d["layers"].items()))))))
        self.refused(r, "differs from the model, its checkpoint or the order of its BN layers")
        r = self.w.copy_runs()

        def not32(d):
            d["layers"]["bn1_1"]["installed_moving_mean_float32"][0] = 0.1
        rewrite(r["nuaa_patience_4"], "restricted", "LivenessNet_fold1_seed43_bn_statistics.json", json_change(not32))
        self.refused(r, "bn1_1 mean: values not finite or not float32 values")
        r = self.w.copy_runs()

        def negative(d):
            d["layers"]["bn1_1"]["installed_moving_variance_float32"][0] = -1.0
        rewrite(r["nuaa_patience_4"], "restricted", "LivenessNet_fold1_seed43_bn_statistics.json", json_change(negative))
        self.refused(r, "a negative variance")
        r = self.w.copy_runs()

        def zero(d):                                      # un'unità ReLU mai attiva sul fitting: varianza 0, ammessa
            d["layers"]["bn_dense"]["installed_moving_variance_float32"][0] = 0.0
        rewrite(r["nuaa_patience_4"], "restricted", "LivenessNet_fold1_seed43_bn_statistics.json", json_change(zero))
        rc, o, e = run_eval(self.w.eval_argv("check-inputs", runs=r), eval_hooks(self.w))
        self.assertEqual(rc, 0, e[-2000:])
        r = self.w.copy_runs()
        rewrite(r["nuaa_patience_4"], "restricted", "LivenessNet_fold1_seed43_bn_statistics.json",
                json_change(lambda d: d["evaluator_agreement"].__setitem__("pass", False)))
        self.refused(r, "records no agreement of the evaluators")

    def test_dev_file_identifiers_swapped(self):
        """Due ID del dev scambiati (soggetto ed etichetta restano quelli della riga): l'ordine degli ID non è quello del
        dev del fold."""
        r = self.w.copy_runs()

        def swap(rows):
            rows[0]["sample_id"], rows[1]["sample_id"] = rows[1]["sample_id"], rows[0]["sample_id"]
        rewrite(r["nuaa_patience_4"], "restricted", "ResNet-18_fold2_seed44_dev_ordinary.csv", csv_change(swap))
        self.refused(r, "dev ordinary: the frame identifiers are not those of the dev of fold 2 in key order")

    def test_dev_file_subject_or_label(self):
        r = self.w.copy_runs()

        def label(rows):
            rows[3]["label"] = "1" if rows[3]["label"] == "0" else "0"
        rewrite(r["nuaa_patience_12"], "restricted", "LivenessNet_fold4_seed42_dev_re-estimated.csv", csv_change(label))
        self.refused(r, "dev re-estimated: subject, session, label or file SHA-256 differ from the manifest")

    def test_dev_file_threshold_column(self):
        r = self.w.copy_runs()

        def thr(rows):
            for x in rows:
                x["threshold_dev_eer"] = "0.5"
        rewrite(r["nuaa_patience_4"], "restricted", "LivenessNet_fold1_seed42_dev_re-estimated.csv", csv_change(thr))
        self.refused(r, "threshold_dev_eer or prediction_dev_eer differ from the EER threshold of the saved scores")

    def test_dev_file_prediction_at_0_5(self):
        r = self.w.copy_runs()

        def pred(rows):
            rows[0]["prediction_05"] = "1" if rows[0]["prediction_05"] == "0" else "0"
        rewrite(r["nuaa_patience_4"], "restricted", "ResNet-18_fold3_seed42_dev_ordinary.csv", csv_change(pred))
        self.refused(r, "prediction_05 differs from the rule of the project")

    def test_dev_file_of_another_state(self):
        r = self.w.copy_runs()

        def state(rows):
            for x in rows:
                x["bn_state"] = "ordinary"
        rewrite(r["nuaa_patience_4"], "restricted", "ResNet-18_fold3_seed42_dev_re-estimated.csv", csv_change(state))
        self.refused(r, "rows of another model, state, role or checkpoint")

    def test_dev_without_a_class(self):
        """Un dev senza una classe (etichette alterate nel file e nel manifest finto): nessuna soglia EER, rifiuto prima
        del test (scelta dell'autore)."""
        rec = {"key": "k", "fold": 1, "network": "LivenessNet", "seed": 42, "weights_sha256": "w"}
        rows = [{"key": f"real/a{i}", "subject": "s", "session": "01", "label": "0", "sha256": "h"} for i in range(4)]
        got = [{"network": "LivenessNet", "fold": "1", "train_seed": "42", "bn_state": s, "role": "dev", "sample_id": r["key"], "subject_id": "s",
                "session_id": "01", "label": "0", "score_attack": "0.2", "logit_bona_fide": "0", "logit_attack": "0", "prediction_05": "0",
                "threshold_dev_eer": "NA", "prediction_dev_eer": "NA", "file_sha256": "h", "checkpoint_sha256": "w", "logit": None}
               for s in D.BN_STATES for r in rows]
        tmp = Path(tempfile.mkdtemp(dir=self.w.tmp))
        rec["dev_files"] = {}
        for s in D.BN_STATES:
            text = FR.csv_text(R.SCORE_COLUMNS, [[g[c] for c in R.SCORE_COLUMNS] for g in got if g["bn_state"] == s])
            (tmp / f"{s}.csv").write_text(text, encoding="utf-8")
            rec["dev_files"][s] = (tmp / f"{s}.csv", sha(text.encode()))
        with mock.patch.object(E.BR, "role_rows", lambda inp, role, arm, fold: rows):
            with self.assertRaisesRegex(FR.Failure, "the dev lacks a class: no EER threshold"):
                E.load_dev(rec, None)

    def test_run_changed_during_the_evaluation(self):
        """Un file di un run cambiato dopo la lettura del test e prima dei risultati: la riverifica ferma il comando."""
        r = self.w.copy_runs()
        target = r["nuaa_patience_12"] / "restricted" / "LivenessNet_fold4_seed44_card.json"

        def touch(ctx, rec, role):
            if role == "test" and rec["key"] == "p12_ResNet-18_fold4_seed44":
                target.write_bytes(target.read_bytes() + b" ")
        out = fresh_out(self.w)
        rc, o, e = run_eval(self.w.eval_argv(runs=r, out=out), eval_hooks(self.w, score=fake_score(on_call=touch)))
        self.assertEqual(rc, 1, (o + e)[-2000:])
        self.assertIn("LivenessNet_fold4_seed44_card.json changed during the evaluation", e)
        no_results(self, out / EVAL_ID)


# ============================================================================ H riproduzione del dev

class TestHDevReproduction(unittest.TestCase):
    def setUp(self):
        self.w = world()

    def test_mismatch_stops_before_the_test(self):
        """Un modello fuori tolleranza nello stato ristimato (2e-5): MISMATCH (codice 4), tutti i 48 modelli controllati e
        documentati, nessuna lista e nessun frame del test, nessun risultato."""
        log = Log()

        def over(rec, state, p):
            return p + 2e-5 if (rec["key"], state) == ("p12_LivenessNet_fold3_seed43", "re-estimated") else p
        out = fresh_out(self.w)
        rc, o, e = run_eval(self.w.eval_argv(out=out), eval_hooks(self.w, score=fake_score(log, dev_over=over)))
        self.assertEqual(rc, 4, (o + e)[-3000:])
        folder = out / EVAL_ID
        no_results(self, folder)
        st = json.loads((folder / "status.json").read_text())
        self.assertEqual(st["status"], "MISMATCH")
        self.assertFalse(st["test_read_started"])
        bad = [r for r in st["dev_reproduction"] if not r["pass"]]
        self.assertEqual([(r["run"], r["network"], r["fold"], r["seed"], r["bn_state"]) for r in bad],
                         [("nuaa_patience_12", "LivenessNet", 3, 43, "re-estimated")])
        self.assertEqual(len(st["dev_reproduction"]), 96)
        self.assertEqual([c for c in log.calls if c[0] == "test"], [])
        self.assertEqual({c[2] for c in log.calls}, {0})

    def test_disagreement_at_the_eer_threshold_only(self):
        """Uno scarto di 1e-7 sui soli punteggi del dev uguali alla soglia EER (dentro la tolleranza, nessuna decisione
        diversa a 0,5): la decisione alla soglia EER cambia (pareggio = bona fide) e la riproduzione non passa."""
        hit = []

        def over(rec, s, p):
            t = rec["thresholds"][(s, "dev_eer")]
            if (rec["key"], s) == ("p4_ResNet-18_fold2_seed42", "re-estimated") and t != 0.5:
                eq = p == t
                hit.append(int(eq.sum()))
                return np.where(eq, p + 1e-7, p)
            return p
        out = fresh_out(self.w)
        rc, o, e = run_eval(self.w.eval_argv(out=out), eval_hooks(self.w, score=fake_score(dev_over=over)))
        self.assertTrue(hit and hit[0] > 0, hit)
        self.assertEqual(rc, 4, (o + e)[-2000:])
        st = json.loads((out / EVAL_ID / "status.json").read_text())
        bad = [r for r in st["dev_reproduction"] if not r["pass"]]
        self.assertEqual([(r["network"], r["fold"], r["seed"], r["bn_state"], r["discordant_05"], r["discordant_eer"]) for r in bad],
                         [("ResNet-18", 2, 42, "re-estimated", 0, hit[0])])
        self.assertLessEqual(bad[0]["max_abs_diff"], 1e-5)

    def test_tolerance_and_decisions(self):
        """Dentro la tolleranza (5e-6) la riproduzione passa; una decisione discordante a 0,5 con uno scarto minimo no."""
        def away(rec, s, p):
            t = rec["thresholds"][(s, "dev_eer")]
            return np.where(p > max(t, 0.5), p + 5e-6, np.where(p <= min(t, 0.5), p - 5e-6, p))
        out = fresh_out(self.w)
        rc, o, e = run_eval(self.w.eval_argv(out=out), eval_hooks(self.w, score=fake_score(dev_over=away)))
        self.assertEqual(rc, 0, (o + e)[-2000:])
        st = json.loads((out / EVAL_ID / "status.json").read_text())
        self.assertGreater(max(r["max_abs_diff"] for r in st["dev_reproduction"]), 4e-6)
        ag = R.evaluator_agreement(np.array([0.5, 0.2]), np.array([0.5000001, 0.2]), 0.3, json.loads(self.w.br_cfg.read_text()))
        self.assertFalse(ag["pass"])
        self.assertEqual(ag["discordant_05"], 1)

    def test_assembled_dev_must_match_the_saved_identifiers(self):
        rec = {"key": "k", "fold": 1, "dev": {"ordinary": {"ids": ["b", "a"]}}}
        ctx = E.Ctx()
        ctx.budget = SR.Budget(60)
        ctx.inp, ctx.store = types_ns(folds={}, assembled=[]), None
        with mock.patch.object(E.R, "unit_frames", lambda store, inp, role, fold: {"keys": ["a", "b"]}):
            with self.assertRaisesRegex(FR.Failure, "the assembled dev frames are not those of the saved dev scores"):
                E.reproduce_dev(ctx, [rec])


def types_ns(**k):
    import types
    return types.SimpleNamespace(**k)


# ============================================================================ I tetto e arresto

class TestIStops(unittest.TestCase):
    def setUp(self):
        self.w = world()

    def stop_at(self, role, key):
        def on_call(ctx, rec, r):
            if (r, rec["key"]) == (role, key):
                ctx.budget.stop_requested = "SIGUSR1"
        return on_call

    def test_sigusr1_during_the_test(self):
        out = fresh_out(self.w)
        rc, o, e = run_eval(self.w.eval_argv(out=out), eval_hooks(self.w, score=fake_score(on_call=self.stop_at("test", "p4_LivenessNet_fold2_seed42"))))
        self.assertEqual(rc, 2, (o + e)[-2000:])
        folder = out / EVAL_ID
        no_results(self, folder)
        st = json.loads((folder / "status.json").read_text())
        self.assertEqual(st["status"], "INCOMPLETE")
        self.assertTrue(st["test_read_started"])
        self.assertEqual(st["stopped"]["cause"], "SIGUSR1")
        self.assertEqual(st["stopped"]["step"], "test scoring p4_ResNet-18_fold2_seed42")

    def test_cap_during_the_dev(self):
        out = fresh_out(self.w)

        def on_call(ctx, rec, role):
            if rec["key"] == "p4_LivenessNet_fold1_seed43":
                ctx.budget.cap = -1
        rc, o, e = run_eval(self.w.eval_argv(out=out), eval_hooks(self.w, score=fake_score(on_call=on_call)))
        self.assertEqual(rc, 2, (o + e)[-2000:])
        st = json.loads((out / EVAL_ID / "status.json").read_text())
        self.assertEqual((st["status"], st["stopped"]["cause"], st["stopped"]["step"]), ("INCOMPLETE", "cap", "dev reproduction p4_ResNet-18_fold1_seed43"))
        self.assertFalse(st["test_read_started"])
        no_results(self, out / EVAL_ID)

    def test_cap_checked_before_assembly_and_analysis(self):
        for key, step in (("p12_ResNet-18_fold1_seed44", "test assembly fold 2"), ("p12_ResNet-18_fold4_seed44", "analysis")):
            out = fresh_out(self.w)

            def on_call(ctx, rec, role, key=key):
                if role == "test" and rec["key"] == key:
                    ctx.budget.cap = -1
            rc, o, e = run_eval(self.w.eval_argv(out=out), eval_hooks(self.w, score=fake_score(on_call=on_call)))
            self.assertEqual(rc, 2, (o + e)[-2000:])
            st = json.loads((out / EVAL_ID / "status.json").read_text())
            self.assertEqual(st["stopped"]["step"], step)
            no_results(self, out / EVAL_ID)


# ============================================================================ J cartella, tentativi, comando

class TestJFolderAndAttempts(unittest.TestCase):
    def setUp(self):
        self.w = world()

    def test_attempt_rule_and_existing_folder(self):
        out = fresh_out(self.w)
        rc, o, e = run_eval(self.w.eval_argv(out=out), eval_hooks(self.w))
        self.assertEqual(rc, 0, e[-2000:])
        before = tree(out)
        rc, o, e = run_eval(self.w.eval_argv(out=out), eval_hooks(self.w))
        self.assertEqual(rc, 1)
        self.assertIn("this attempt must be --attempt 2", e)
        self.assertEqual(tree(out), before)
        rc, o, e = run_eval(self.w.eval_argv(out=out, attempt=2), eval_hooks(self.w))
        self.assertEqual(rc, 0, e[-2000:])
        st = json.loads((out / f"{EVAL_ID}-r2" / "status.json").read_text())
        self.assertEqual(st["attempt"]["previous"], [{"run_id": EVAL_ID, "attempt": 1, "status": "PASS"}])
        (out / f"{DATE}-ARCHEVAL-{STATE['commit']}-r1").mkdir()
        rc, o, e = run_eval(self.w.eval_argv(out=out, attempt=3), eval_hooks(self.w))
        self.assertEqual(rc, 1)
        self.assertIn("does not have the expected form", e)

    def test_dirty_tree_and_roots(self):
        out = fresh_out(self.w)
        h = eval_hooks(self.w)
        h["r"]["REPO_STATE"] = lambda root: {"commit": "abcdef0", "dirty": True}
        rc, o, e = run_eval(self.w.eval_argv(out=out), h)
        self.assertEqual(rc, 1)
        self.assertIn("the git work tree is not clean", e)
        rc, o, e = run_eval(self.w.eval_argv(out=out, unofficial=False), eval_hooks(self.w))
        self.assertEqual(rc, 1)
        self.assertIn("needs --unofficial-out-root", e)
        self.assertEqual(os.listdir(out), [])
        rc, o, e = run_eval(self.w.eval_argv(out=ROOT / "results" / "archeval_test_never"), eval_hooks(self.w))
        self.assertEqual(rc, 1)
        self.assertIn("inside the repository", e)
        self.assertFalse((ROOT / "results" / "archeval_test_never").exists())

    def test_authorisation_required(self):
        a = [x for x in self.w.eval_argv(out=fresh_out(self.w)) if x != "--i-am-authorized"]
        rc, o, e = run_eval(a, eval_hooks(self.w))
        self.assertEqual(rc, 1)
        self.assertIn("pass --i-am-authorized", e)

    def test_machine_and_threads(self):
        out = fresh_out(self.w)
        h = eval_hooks(self.w)
        h["r"]["MACHINE"] = lambda: {**CR.MACH, "logical_cpus": 14}
        rc, o, e = run_eval(self.w.eval_argv(out=out), h)
        self.assertEqual(rc, 1)
        self.assertIn("machine rule: logical_cpus differ", e)
        self.assertEqual(os.listdir(out), [])
        out = fresh_out(self.w)
        rc, o, e = run_eval(self.w.eval_argv(out=out), eval_hooks(self.w, tf=CR.FakeTF(offset=1)))
        self.assertEqual(rc, 1)
        self.assertIn("TensorFlow threads", e)
        st = json.loads((out / EVAL_ID / "status.json").read_text())
        self.assertEqual(st["status"], "FAIL")
        no_results(self, out / EVAL_ID)
        with mock.patch.dict(os.environ, {"TF_NUM_INTRAOP_THREADS": "8"}):
            out = fresh_out(self.w)
            rc, o, e = run_eval(self.w.eval_argv(out=out), eval_hooks(self.w))
        self.assertEqual(rc, 1)
        self.assertIn("contradicts training.threads.intra_op", e)
        self.assertEqual(os.listdir(out), [])
        with mock.patch.dict(os.environ, {"TF_NUM_INTRAOP_THREADS": "10"}):
            out = fresh_out(self.w)
            rc, o, e = run_eval(self.w.eval_argv(out=out), eval_hooks(self.w))
        self.assertEqual(rc, 1)
        self.assertIn("differ from the official trial", e)
        self.assertEqual(os.listdir(out), [])

    def test_thread_rule_enforced(self):
        """Thread impostati e riletti uguali alla configurazione, ma il confronto con la prova (R.threads_rule) diverso: il
        comando si ferma prima di ogni valutazione, con il confronto registrato in status.json e nessun risultato."""
        rule = {"fields": {"effective": {"trial": {"intra_op": 10, "inter_op": 10}, "now": {"intra_op": 10, "inter_op": 9}, "equal": False}},
                "equal": False, "trial_run": "20261119-ARCHTRIAL-abcdef0"}
        log = Log()
        out = fresh_out(self.w)
        with mock.patch.object(E.R, "threads_rule", lambda trial, threads: rule):
            rc, o, e = run_eval(self.w.eval_argv(out=out), eval_hooks(self.w, log))
        self.assertEqual(rc, 1, (o + e)[-2000:])
        self.assertIn("thread rule: effective differ from the official trial", e)
        st = json.loads((out / EVAL_ID / "status.json").read_text())
        self.assertEqual(st["threads_rule"], rule)
        self.assertEqual(log.calls, [])
        no_results(self, out / EVAL_ID)

    def test_public_texts_scanned(self):
        """Un identificativo di soggetto in un testo pubblico: rifiutato prima di ogni scrittura, nessun risultato."""
        out = fresh_out(self.w)
        real = E.report_md
        with mock.patch.object(E, "report_md", lambda doc: real(doc) + f"\nsubject {CFR.SUBJECTS[0]}\n"):
            rc, o, e = run_eval(self.w.eval_argv(out=out), eval_hooks(self.w))
        self.assertEqual(rc, 1, (o + e)[-2000:])
        self.assertIn("report.md refused: subject id", e)
        no_results(self, out / EVAL_ID)

    def test_failure_after_the_restricted_files_removes_them(self):
        """Un errore nella scrittura delle uscite pubbliche, dopo quelle riservate: i file riservati scritti sono tolti."""
        out = fresh_out(self.w)
        real = E.mw.write_all

        def failing(folder, files, newline=None):
            if Path(folder).name == "public":
                raise OSError("disk full (simulated)")
            return real(folder, files, newline=newline)
        with mock.patch.object(E.mw, "write_all", failing):
            rc, o, e = run_eval(self.w.eval_argv(out=out), eval_hooks(self.w))
        self.assertEqual(rc, 1, (o + e)[-2000:])
        self.assertIn("disk full (simulated)", e)
        no_results(self, out / EVAL_ID)
        st = json.loads((out / EVAL_ID / "status.json").read_text())
        self.assertRegex(st["restricted"], r"^\d+ restricted result files were written and then removed after the failure$")
        self.assertEqual(st["public"], "no public result file was written")

    def test_command_text(self):
        f = flow_ok(self)
        st = json.loads((f["folder"] / "status.json").read_text())
        self.assertTrue(st["command_line"].startswith("python -I -B scripts/pad_arch_eval.py evaluate --config '<outside the repository>/archeval.json'"))
        self.assertIn(f"--run-p4 '<outside the repository>/{self.w.run_dirs['nuaa_patience_4'].name}'", st["command_line"])
        self.assertNotIn(str(self.w.tmp), st["command_line"])
        self.assertIn("--unofficial-out-root", st["command_line"])
        self.assertFalse(st["official"])

    def test_pass_then_failure_in_the_same_root_leaves_the_pass_intact(self):
        """Pattern 12: dopo un PASS, lo stesso comando con un ingresso guasto nella stessa radice: nessun file del PASS
        cambia, il nuovo tentativo è FAIL e senza risultati."""
        out = fresh_out(self.w)
        rc, o, e = run_eval(self.w.eval_argv(out=out), eval_hooks(self.w))
        self.assertEqual(rc, 0)
        before = tree(out / EVAL_ID)
        rc, o, e = run_eval(self.w.eval_argv(out=out, attempt=2), eval_hooks(self.w, plan_hook=lambda p: p[1:]))
        self.assertEqual(rc, 1)
        self.assertEqual(tree(out / EVAL_ID), before)
        no_results(self, out / f"{EVAL_ID}-r2")


# ============================================================================ K unità

class TestKUnits(unittest.TestCase):
    def test_build_conditions_from_the_plan(self):
        """Le unità prendono la soglia dalla voce del piano (stesso valore) e una voce ripetuta resta visibile come unità
        in più."""
        recs = []
        meta = {f: {"y": np.array([0, 1]), "subject": np.array(["s1", "s2"]), "ids": [f"real/a{f}", f"attack/b{f}"]} for f in range(1, 5)}
        for p in D.PATIENCES:
            for n in AM.NETWORKS:
                for f in range(1, 5):
                    for s in D.PLAN_SEEDS:
                        recs.append({"key": f"p{p}_{n}_fold{f}_seed{s}", "fold": f, "seed": s, "network": n,
                                     "test": {b: {"p": np.array([0.2, 0.7])} for b in D.BN_STATES}})
        plan = plan_full()
        conds = E.build_conditions(plan, recs, meta)
        self.assertEqual({len(v) for v in conds.values()}, {12})
        u = conds[(4, "ordinary", "dev_eer")][0]
        self.assertEqual(u["networks"]["LivenessNet"]["threshold"], 0.37)
        self.assertEqual(sorted(u["networks"]["ResNet-18"]), sorted(D.NETWORK_FIELDS))
        conds2 = E.build_conditions(plan + [plan[0]], recs, meta)
        self.assertEqual(len(conds2[(plan[0]["patience"], plan[0]["bn_state"], plan[0]["rule"])]), 13)

    def test_tie_is_on_acer_exact_fractions(self):
        """Pareggio di un'unità = ACER uguali come frazioni esatte, anche con APCER e BPCER diversi: LivenessNet respinge un
        bona fide su due, ResNet-18 accetta un attacco su due; ACER 1/4 per entrambe."""
        y = np.array([0, 0, 1, 1])
        ids = ["real/t0", "real/t1", "attack/t2", "attack/t3"]
        subj = np.array(["a", "b", "a", "b"])
        units = []
        for f in range(1, 5):
            for sd in D.PLAN_SEEDS:
                units.append({"unit": ("NUAA", f, sd), "networks": {
                    "LivenessNet": {"y": y, "s": np.array([0.9, 0.1, 0.9, 0.9]), "subject": subj, "threshold": 0.5, "sample_id": [f"{i}{f}" for i in ids]},
                    "ResNet-18": {"y": y, "s": np.array([0.1, 0.1, 0.1, 0.9]), "subject": subj, "threshold": 0.5, "sample_id": [f"{i}{f}" for i in ids]}}})
        for f in range(1, 5):                              # soggetti diversi per fold (i test dei fold sono disgiunti)
            for u in units:
                if u["unit"][1] == f:
                    for n in AM.NETWORKS:
                        u["networks"][n] = {**u["networks"][n], "subject": np.array([f"s{f}a", f"s{f}b", f"s{f}a", f"s{f}b"])}
        cond = D.PRIMARY_ORDINARY
        c = D.compare_condition(units)
        rows = E.unit_rows(cond, c, {cond: units})
        self.assertEqual({E.cell(r["tie"]) for r in rows}, {"1"})
        self.assertEqual({E.cell(r["delta_apcer"]) for r in rows}, {"1/2"})
        self.assertEqual({E.cell(r["delta_acer"]) for r in rows}, {"0/1"})

    def test_test_decisions_tie_at_the_threshold(self):
        """Nei file riservati del test la decisione segue la regola del progetto: un punteggio uguale alla soglia è bona
        fide, a 0,5 e alla soglia EER."""
        rec = {"run": "nuaa_patience_4", "network": "LivenessNet", "fold": 1, "seed": 42, "patience": 4, "weights_sha256": "w",
               "thresholds": {(s, r): (0.5 if r == "fixed_0.5" else 0.25) for s in D.BN_STATES for r in D.THRESHOLD_RULES},
               "test": {s: {"p": np.array([0.5, 0.25, 0.5000001]), "z": np.zeros((3, 2))} for s in D.BN_STATES}}
        meta = {1: {"rows": [{"key": f"real/k{i}", "subject": "s", "session": "01", "label": "0", "sha256": "h"} for i in range(3)]}}
        rows = E.test_score_rows(rec, meta)
        cols = list(E.TEST_SCORE_COLUMNS)
        self.assertEqual([r[cols.index("prediction_05")] for r in rows[:3]], ["0", "0", "1"])
        self.assertEqual([r[cols.index("prediction_dev_eer")] for r in rows[:3]], ["1", "0", "1"])

    def test_jsonable_and_cells(self):
        self.assertEqual(E.jsonable({(1, 2): Fraction(1, 3), 4: np.float64(0.5), "t": (np.int64(2), True)}),
                         {"[1, 2]": {"exact": "1/3", "value": 1 / 3}, "4": 0.5, "t": [2, True]})
        self.assertEqual([E.cell(x) for x in (None, True, Fraction(2, 4), 0.25, 3)], ["NA", "1", "1/2", "0.25", "3"])


# ============================================================================ L liste del test

class TestLTestLists(unittest.TestCase):
    def setUp(self):
        self.w = world()

    def ctx(self):
        cfg, _ = E.load_config(self.w.eval_cfg)
        ctx = E.Ctx()
        with injected(**CR.base_hooks(self.w)):
            pins = E.load_pins(cfg, self.w.base)
            ctx.inp = PAC.load_inputs(pins.cfg_pr, pins.pcfg, self.w.base)
        ctx.pcfg, ctx.base, ctx.folds, ctx.test_per_fold = pins.pcfg, self.w.base, pins.folds, pins.test_per_fold
        return ctx

    def test_lists_pass_and_counts(self):
        ctx = self.ctx()
        with injected(**CR.base_hooks(self.w)):
            lists = E.load_test_lists(ctx)
        self.assertEqual([len(lists[f]) for f in ctx.folds], ctx.test_per_fold)
        ctx2 = self.ctx()
        ctx2.test_per_fold = [n + (1 if i == 2 else 0) for i, n in enumerate(ctx2.test_per_fold)]
        with injected(**CR.base_hooks(self.w)), self.assertRaisesRegex(FR.Failure, r"fold 3: \d+ test frames, the pre-registration plans"):
            E.load_test_lists(ctx2)

    def test_test_disjoint_from_fitting_and_dev(self):
        ctx = self.ctx()
        real = BR.role_rows

        def leaking(inp, role, arm, fold):
            rows = real(inp, role, arm, fold)
            return rows + ([inp.folds[fold]["test"][0]] if role == "dev" and fold == 2 and inp.folds else [])
        with injected(**CR.base_hooks(self.w)), mock.patch.object(E.BR, "role_rows", leaking):
            with self.assertRaisesRegex(FR.Failure, "fold 2: the test shares a frame or a subject"):
                E.load_test_lists(ctx)


    def test_schema_configuration_checked(self):
        """Lo schema dei fold come in nuaa_bn_test.load_inputs: la configurazione dello schema approvato è controllata."""
        ctx = self.ctx()
        h = CR.base_hooks(self.w)
        h["sr"]["SCHEMA_CONFIG_SHA256"] = "0" * 64
        with injected(**h), self.assertRaisesRegex(FR.Failure, "schema configuration .*differs from the approved schema"):
            E.load_test_lists(ctx)

    def test_sources_of_the_test_rows_checked(self):
        """Una riga del test che non sta né nella cache né nei frame estratti: SR.check_sources ferma il comando."""
        ctx = self.ctx()
        real = FR.load_fold_manifests

        def extra_row(v, base, inp):
            real(v, base, inp)
            inp.folds[1]["test"] = inp.folds[1]["test"] + [{**inp.folds[1]["test"][0], "key": "real/9999_not_a_frame.jpg"}]
        with injected(**CR.base_hooks(self.w)), mock.patch.object(E.FR, "load_fold_manifests", extra_row):
            with self.assertRaisesRegex(FR.Failure, r"test fold 1: a frame is found 0 times"):
                E.load_test_lists(ctx)

    def test_every_fold_loaded(self):
        ctx = self.ctx()
        real = FR.load_fold_manifests

        def three(v, base, inp):
            real(v, base, inp)
            del inp.folds[4]
        with injected(**CR.base_hooks(self.w)), mock.patch.object(E.FR, "load_fold_manifests", three):
            assert_stops(self, lambda: E.load_test_lists(ctx), r"the fold schema was not loaded for every fold 1..4")


# ============================================================================ M correzioni della revisione indipendente

class TestMPointBMore(unittest.TestCase):
    """Il controllo (b), altri casi: valori strani della soglia, 13 unità, fold e seed scambiati, una rete fuori dal piano."""

    F, S = [1, 2, 3, 4], list(D.PLAN_SEEDS)

    def check(self, units, cond=D.PRIMARY_ORDINARY):
        return E.check_units(cond, units, self.F, self.S)

    def test_odd_threshold_values(self):
        for bad in ("0.5", 0.5000001, float("nan"), True):
            u = units_full()
            u[2]["networks"]["ResNet-18"] = {**u[2]["networks"]["ResNet-18"], "threshold": bad}
            with self.assertRaisesRegex(FR.Failure, r"^primary analysis stopped: the effective threshold of ResNet-18 is .*, not 0.5, in the ordinary BN "
                                                    r"state \(patience 4, threshold fixed_0.5; 1 entries\)$", msg=repr(bad)):
                self.check(u)

    def test_thirteen_units(self):
        u = units_full()
        u.append({**u[0], "unit": ("NUAA", 5, 42)})
        with self.assertRaisesRegex(FR.Failure, r"^primary analysis stopped: 1 of the 12 fold-seed combinations not in the plan for LivenessNet .*"
                                                r"fold 5 seed 42; 1 of the 12 fold-seed combinations not in the plan for ResNet-18 .*fold 5 seed 42$"):
            self.check(u)

    def test_fold_and_seed_swapped(self):
        u = units_full()
        self.assertEqual(u[0]["unit"], ("NUAA", 1, 42))
        u[0] = {**u[0], "unit": ("NUAA", 42, 1)}
        with self.assertRaisesRegex(FR.Failure, r"^primary analysis stopped: 1 of the 12 fold-seed combinations missing for LivenessNet .*fold 1 seed 42; "
                                                r"1 of the 12 fold-seed combinations not in the plan for LivenessNet .*fold 42 seed 1;"):
            self.check(u)

    def test_network_outside_the_plan(self):
        u = units_full()
        u[0]["networks"]["VGG"] = dict(u[0]["networks"]["LivenessNet"])
        with self.assertRaisesRegex(FR.Failure, r"^primary analysis stopped: networks outside the plan in the ordinary BN state .*\['VGG'\]$"):
            self.check(u)

    def test_point_a_fold_and_seed_swapped(self):
        plan = [{**e, "fold": 42, "seed": 1} if (e["network"], e["bn_state"], e["rule"], e["patience"], e["fold"], e["seed"]) ==
                ("ResNet-18", "re-estimated", "fixed_0.5", 4, 1, 42) else e for e in plan_full()]
        with self.assertRaisesRegex(FR.Failure, r"^primary analysis stopped before the test: 1 of the 12 fold-seed combinations missing for ResNet-18 in "
                                                r"the re-estimated BN state .*fold 1 seed 42; 1 of the 12 fold-seed combinations not in the plan for "
                                                r"ResNet-18 .*fold 42 seed 1$"):
            E.check_plan(plan, self.F, self.S)

    def test_plan_threshold_from_the_record(self):
        """La soglia della voce del piano è quella del record del modello (da D.condition_threshold), non una costante."""
        rec = {"key": "k", "patience": 4, "network": "LivenessNet", "fold": 1, "seed": 42,
               "thresholds": {(s, r): (0.45 if r == "fixed_0.5" else 0.3) for s in D.BN_STATES for r in D.THRESHOLD_RULES}}
        plan = E.evaluation_plan([rec])
        self.assertEqual({e["threshold"] for e in plan if e["rule"] == "fixed_0.5"}, {0.45})
        self.assertEqual({e["threshold"] for e in plan if e["rule"] == "dev_eer"}, {0.3})

    def test_degenerate_rule(self):
        """Il conteggio «degenere» di prima (gruppo «ten» della PR #27, nuaa_bn_recalibration.group_of): epoca ripristinata
        1 o 2 e accuratezza del dev al checkpoint 0,5 entro 1e-6. Esempio sintetico: 2 su 5 per LivenessNet a patience 4."""
        bn = json.loads(world().br_cfg.read_text())
        cards = [(1, 0.5), (2, 0.5 + 5e-7), (3, 0.5), (1, 0.6), (2, 0.5 - 2e-6)]
        recs = [{"network": "LivenessNet", "patience": 4, "card": {"restored_epoch": r, "val_acc_at_best": a, "best_val_loss": 0.69}} for r, a in cards]
        out = E.degenerate_counts(recs, bn)
        self.assertEqual(out[0], {"network": "LivenessNet", "patience": 4, "degenerate": 2, "trainings": 5})
        self.assertEqual([x["degenerate"] for x in out[1:]], [0, 0, 0])


class TestNReviewFlow(unittest.TestCase):
    """Le correzioni nel flusso completo: run ufficiali in una valutazione non ufficiale, numeri mai nel log, riverifiche
    prima e dopo il test, tetto effettivo, controllo incrociato dei conteggi one-class, lunghezze delle statistiche."""

    def setUp(self):
        self.w = world()

    def eval_fail(self, msg, runs=None, score=None, argv=None, **h):
        out = fresh_out(self.w)
        rc, o, e = run_eval(argv(out) if argv else self.w.eval_argv(runs=runs, out=out), eval_hooks(self.w, score=score, **h))
        self.assertEqual(rc, 1, (o + e)[-3000:])
        self.assertRegex(e, msg)
        return out, o

    def test_unofficial_evaluation_refuses_an_official_run(self):
        r = self.w.copy_runs()
        rewrite_status(r["nuaa_patience_4"], lambda st: st.__setitem__("official", True))
        out, _ = self.eval_fail("an official run is never read by an unofficial evaluation \\(--unofficial-out-root\\)", runs=r)
        self.assertEqual(os.listdir(out), [])
        rc, o, e = run_eval(self.w.eval_argv("check-inputs", runs=r), eval_hooks(self.w))
        self.assertEqual(rc, 0, e[-2000:])                  # check-inputs non legge il test: accetta entrambi

    def test_unofficial_evaluation_refuses_a_status_without_the_official_field(self):
        r = self.w.copy_runs()
        rewrite_status(r["nuaa_patience_12"], lambda st: st.pop("official"))
        out, _ = self.eval_fail("an official run is never read by an unofficial evaluation", runs=r)
        self.assertEqual(os.listdir(out), [])

    def test_unofficial_evaluation_refuses_a_folder_in_the_official_run_root(self):
        r = self.w.copy_runs()
        with mock.patch.object(E, "OFFICIAL_RUNS_DIR", r["nuaa_patience_12"].parent):
            out, _ = self.eval_fail("a folder in results/archrun is never read by an unofficial evaluation \\(--unofficial-out-root\\)", runs=r)
        self.assertEqual(os.listdir(out), [])

    def test_official_run_root_recognised_by_identity(self):
        """La radice dei run ufficiali riconosciuta per identità dei file, non per prefisso del percorso: un altro nome
        della stessa cartella (alias_of, come una grafia diversa su un filesystem che non distingue le maiuscole).
        Alias e cartella sono percorsi risolti: path_guard.within risolve i percorsi prima di os.stat, e alias_of confronta
        il prefisso come testo, quindi con una TMPDIR raggiunta tramite un collegamento (/var/folders su macOS) un alias non
        risolto non scatterebbe mai."""
        r = self.w.copy_runs()
        alias = (self.w.tmp / "OTHER-SPELLING-OF-THE-RUN-ROOT").resolve()
        target = r["nuaa_patience_12"].parent.resolve()
        self.assertNotEqual(str(alias), str(target))
        with mock.patch.object(E, "OFFICIAL_RUNS_DIR", alias), CR.alias_of(alias, target):
            self.assertTrue(E.pg.within(r["nuaa_patience_12"], alias))      # l'alias scatta davvero
            self.assertFalse(str(r["nuaa_patience_12"]).startswith(str(alias)))  # e non per prefisso di testo
            out, _ = self.eval_fail("a folder in results/archrun is never read by an unofficial evaluation", runs=r)
        self.assertEqual(os.listdir(out), [])

    def test_unofficial_evaluation_refuses_the_inputs_of_the_repository(self):
        """La base degli ingressi che è il repository (simulata con alias_of): una valutazione non ufficiale si ferma prima
        di leggere qualunque file. L'alias è costruito sul percorso risolto della base (vedi
        test_official_run_root_recognised_by_identity: path_guard.within risolve prima di os.stat)."""
        out = fresh_out(self.w)
        base = self.w.base.resolve()
        with CR.alias_of(base, ROOT):
            self.assertTrue(E.pg.within(self.w.base, ROOT) and E.pg.within(ROOT, self.w.base))   # l'alias scatta davvero
            rc, o, e = run_eval(self.w.eval_argv(out=out), eval_hooks(self.w))
        self.assertEqual(rc, 1, (o + e)[-2000:])
        self.assertIn("an unofficial evaluation (--unofficial-out-root) does not use the inputs of the repository", e)
        self.assertEqual(os.listdir(out), [])

    def primary_numbers(self) -> list:
        p = json.loads((flow_ok(self)["folder"] / "public" / "results.json").read_text())["primary_analysis"]
        vals = [p["gap_ordinary"]["mean_delta"]["acer"], p["gap_re_estimated"]["mean_delta"]["acer"], p["contrast"]["mean"]["acer"],
                p["contrast"]["interval"]["acer"]["low"], p["contrast"]["interval"]["acer"]["high"]]
        return [x for v in vals for x in (f"{v:.4f}", repr(v))]

    def test_no_number_in_the_log_after_a_pass(self):
        f = flow_ok(self)
        log = (f["folder"] / "run.log").read_text()
        self.assertEqual([x for x in self.primary_numbers() if x in log], [])
        self.assertIn("results written to public/ and restricted/", log)
        self.assertNotIn("analysis computed", log)
        self.assertIn("analysis computed; report:", f["out"])

    def test_late_failure_leaves_no_number_and_says_what_was_removed(self):
        """Un FAIL forzato all'ultimo passo (lo status.json finale): uscite tolte, e né run.log né status.json contengono
        un numero dell'analisi; lo stato dice che i file sono stati scritti e tolti."""
        real = R.write_status

        def late(run, doc, inp, redact=False):
            if doc.get("status") == "PASS":
                raise OSError("status not written (simulated)")
            return real(run, doc, inp, redact)
        out = fresh_out(self.w)
        with mock.patch.object(E.R, "write_status", late):
            rc, o, e = run_eval(self.w.eval_argv(out=out), eval_hooks(self.w))
        self.assertEqual(rc, 1, (o + e)[-2000:])
        folder = out / EVAL_ID
        no_results(self, folder)
        text = (folder / "run.log").read_text() + (folder / "status.json").read_text()
        self.assertEqual([x for x in self.primary_numbers() if x in text], [])
        st = json.loads((folder / "status.json").read_text())
        self.assertEqual(st["status"], "FAIL")
        self.assertRegex(st["restricted"], r"^\d+ restricted result files were written and then removed after the failure$")
        self.assertEqual(st["public"], "the public result files were written and then removed after the failure")
        self.assertNotIn("analysis computed", o)

    def test_log_scanned_before_the_numbers(self):
        """Un testo vietato nel log durante il test: il comando si ferma prima di calcolare l'analisi."""
        calls = []
        real = D.primary_analysis

        def say(ctx, rec, role):
            if role == "test" and rec["key"] == "p4_LivenessNet_fold1_seed42":
                print(f"subject {CFR.SUBJECTS[0]}", flush=True)
        with mock.patch.object(E.D, "primary_analysis", lambda c: calls.append(1) or real(c)):
            out, _ = self.eval_fail("run.log holds text that a publishable output must not hold", score=fake_score(on_call=say))
        self.assertEqual(calls, [])
        no_results(self, out / EVAL_ID)

    def test_effective_cap_in_seconds(self):
        seen = []
        out = fresh_out(self.w)
        rc, o, e = run_eval(self.w.eval_argv(out=out), eval_hooks(self.w, score=fake_score(on_call=lambda ctx, rec, role: seen.append(ctx.budget.cap))))
        self.assertEqual(rc, 0, e[-2000:])
        hours = json.loads(self.w.eval_cfg.read_text())["cap"]["hours"]
        self.assertEqual(set(seen), {hours * 3600})

    def config_copy(self) -> Path:
        d = Path(tempfile.mkdtemp(dir=self.w.tmp))
        p = d / "archeval.json"
        p.write_bytes(self.w.eval_cfg.read_bytes())
        return p

    def argv_with(self, cfg: Path):
        def make(out):
            a = self.w.eval_argv(out=out)
            a[a.index("--config") + 1] = str(cfg)
            return a
        return make

    def test_configuration_changed_before_the_test(self):
        cfg = self.config_copy()
        log = Log()

        def touch(ctx, rec, role):
            if (role, rec["key"]) == ("dev", "p12_ResNet-18_fold4_seed44"):
                cfg.write_bytes(cfg.read_bytes() + b" ")
        out, _ = self.eval_fail("the configuration changed during the evaluation", argv=self.argv_with(cfg), score=fake_score(log, on_call=touch))
        st = json.loads((out / EVAL_ID / "status.json").read_text())
        self.assertFalse(st["test_read_started"])
        self.assertEqual([c for c in log.calls if c[0] == "test"], [])

    def test_configuration_changed_during_the_test(self):
        cfg = self.config_copy()

        def touch(ctx, rec, role):
            if (role, rec["key"]) == ("test", "p4_ResNet-18_fold4_seed44"):
                cfg.write_bytes(cfg.read_bytes() + b" ")
        out, _ = self.eval_fail("the configuration changed during the evaluation", argv=self.argv_with(cfg), score=fake_score(on_call=touch))
        st = json.loads((out / EVAL_ID / "status.json").read_text())
        self.assertTrue(st["test_read_started"])
        no_results(self, out / EVAL_ID)

    def test_run_file_changed_before_the_test(self):
        """Un file di un run cambiato durante la riproduzione del dev: la riverifica prima delle liste del test ferma il comando."""
        r = self.w.copy_runs()
        target = r["nuaa_patience_4"] / "restricted" / "ResNet-18_fold1_seed42_card.json"
        log = Log()

        def touch(ctx, rec, role):
            if (role, rec["key"]) == ("dev", "p12_LivenessNet_fold1_seed42"):
                target.write_bytes(target.read_bytes() + b" ")
        lists = []
        real = E.load_test_lists
        with mock.patch.object(E, "load_test_lists", lambda ctx: lists.append(1) or real(ctx)):
            out, _ = self.eval_fail("ResNet-18_fold1_seed42_card.json changed during the evaluation", runs=r, score=fake_score(log, on_call=touch))
        self.assertFalse(json.loads((out / EVAL_ID / "status.json").read_text())["test_read_started"])
        self.assertEqual([c for c in log.calls if c[0] == "test"], [])
        self.assertEqual(lists, [])                         # la riverifica viene prima delle liste del test, non dopo

    def test_test_read_started_before_the_test_lists(self):
        """Un errore nel caricamento delle liste del test (configurazione dello schema diversa): lo stato registra che la
        lettura del test era cominciata (test_read_started è vero prima di aprire le liste)."""
        h = eval_hooks(self.w)
        h["sr"]["SCHEMA_CONFIG_SHA256"] = "0" * 64
        out = fresh_out(self.w)
        rc, o, e = run_eval(self.w.eval_argv(out=out), h)
        self.assertEqual(rc, 1, (o + e)[-2000:])
        self.assertIn("schema configuration", e)
        st = json.loads((out / EVAL_ID / "status.json").read_text())
        self.assertTrue(st["test_read_started"])
        no_results(self, out / EVAL_ID)

    def test_run_log_of_a_run_changed_during_the_test(self):
        r = self.w.copy_runs()
        target = r["nuaa_patience_4"] / "run.log"

        def touch(ctx, rec, role):
            if (role, rec["key"]) == ("test", "p12_ResNet-18_fold2_seed44"):
                target.write_bytes(target.read_bytes() + b"\n")
        out, _ = self.eval_fail("run.log changed during the evaluation", runs=r, score=fake_score(on_call=touch))
        no_results(self, out / EVAL_ID)

    def test_final_scan_of_the_run_log(self):
        """Un testo vietato scritto nel log dopo la prima scansione (durante l'analisi): la scansione finale ferma il
        comando e le uscite già scritte sono tolte."""
        real = E.analyse

        def loud(*args):
            out = real(*args)
            print(f"subject {CFR.SUBJECTS[0]}", flush=True)
            return out
        with mock.patch.object(E, "analyse", loud):
            out, _ = self.eval_fail("run.log holds text that a publishable output must not hold")
        no_results(self, out / EVAL_ID)

    def test_cap_checked_before_the_re_verification(self):
        out = fresh_out(self.w)

        def on_call(ctx, rec, role):
            if (role, rec["key"]) == ("dev", "p12_ResNet-18_fold4_seed44"):
                ctx.budget.cap = -1
        rc, o, e = run_eval(self.w.eval_argv(out=out), eval_hooks(self.w, score=fake_score(on_call=on_call)))
        self.assertEqual(rc, 2, (o + e)[-2000:])
        st = json.loads((out / EVAL_ID / "status.json").read_text())
        self.assertEqual(st["stopped"]["step"], "re-verification before the test")
        self.assertFalse(st["test_read_started"])
        no_results(self, out / EVAL_ID)

    def test_report_numbers_against_the_results(self):
        """Ogni numero del report che va al coordinatore riletto dal testo e confrontato con results.json: i due divari e
        il contrasto con media, intervallo, ricampionamenti validi e conteggi; la lettura; la tabella per seed con la lettura
        per seed; la lettura per seed delle otto condizioni; i conteggi one-class e «degeneri»."""
        f = flow_ok(self)
        rep = (f["folder"] / "public" / "report.md").read_text().splitlines()
        doc = json.loads((f["folder"] / "public" / "results.json").read_text())
        p = doc["primary_analysis"]

        primary = rep[rep.index("## Primary analysis (requested by the coordinator: patience 4, threshold 0.5)"):
                      rep.index("### Per seed (reported apart; not part of the reading)")]

        def row(prefix, lines=rep):
            got = [l for l in lines if l.startswith(prefix)]
            self.assertEqual(len(got), 1, prefix)
            return [c.strip() for c in got[0].split("|")[1:-1]]

        def close(text, v):
            self.assertLessEqual(abs(float(text) - v), 5e-5 + 1e-12, (text, v))

        def mean_iv(cell_text, mean, iv):
            m = re.fullmatch(r"(-?\d+\.\d{4}) \[(-?\d+\.\d{4}), (-?\d+\.\d{4})\]", cell_text)
            self.assertIsNotNone(m, cell_text)
            close(m[1], mean)
            close(m[2], iv["low"])
            close(m[3], iv["high"])
        for label, g, key in (("ACER gap, ordinary BN", p["gap_ordinary"], "mean_delta"), ("ACER gap, re-estimated BN", p["gap_re_estimated"], "mean_delta"),
                              ("contrast", p["contrast"], "mean")):
            cells = row(f"| {label} |", primary)
            close(cells[1], g[key]["acer"])
            m = re.fullmatch(r"\[(-?\d+\.\d{4}), (-?\d+\.\d{4})\] \((\d+)\)", cells[2])
            self.assertIsNotNone(m, cells[2])
            close(m[1], g["interval"]["acer"]["low"])
            close(m[2], g["interval"]["acer"]["high"])
            self.assertEqual(int(m[3]), g["interval"]["acer"]["valid_resamples"])
            if label != "contrast":
                self.assertEqual(cells[3], f"{g['acer_counts']['resnet18_lower']} / {g['acer_counts']['livenessnet_lower']} / {g['acer_counts']['ties']}")
            else:
                self.assertEqual(cells[3], f"units whose gap changes sign: {g['units_changing_sign']} of {g['units']}")
        self.assertEqual(len([l for l in rep if l.startswith("Reading of the contrast") and f"**{p['contrast']['reading']}** ({p['contrast']['reason']})" in l]), 1)
        self.assertEqual(sorted(p["per_seed"]), ["42", "43", "44"])
        for s, v in p["per_seed"].items():
            cells = row(f"| {s} |")
            mean_iv(cells[1], v["gap_ordinary"]["mean_delta"]["acer"], v["gap_ordinary"]["interval"]["acer"])
            mean_iv(cells[2], v["gap_re_estimated"]["mean_delta"]["acer"], v["gap_re_estimated"]["interval"]["acer"])
            mean_iv(cells[3], v["contrast"]["mean"]["acer"], v["contrast"]["interval"]["acer"])
            self.assertEqual(cells[4], v["contrast"]["reading"])
        self.assertGreater(len({v["contrast"]["reading"] for v in p["per_seed"].values()} | {p["contrast"]["reading"]}), 1)   # letture diverse
        for x in doc["conditions"]:
            k = x["condition"]
            cells = row(f"| {k[0]} | {k[1]} | {k[2]} |")
            mean_iv(cells[3], x["mean_delta"]["acer"], x["interval"]["acer"])
            self.assertEqual(cells[5], x["sign"])
            self.assertEqual(cells[6], x["per_seed_reading"]["verdict"])
        for o in doc["one_class"]:
            cells = row(f"| {o['network']} | {o['patience']} | {o['bn_state']} |")
            self.assertEqual(cells[3], f"{o['text']} (all bona fide {o['all_bona_fide']}, all attack {o['all_attack']}, not evaluable {o['not_evaluable']})")
        for x in doc["degenerate_apart"]["counts"]:
            cells = row(f"| {x['network']} | {x['patience']} | {x['degenerate']} of")
            self.assertEqual(cells[2], f"{x['degenerate']} of {x['trainings']}")

    def test_run_status_changed_during_the_test(self):
        r = self.w.copy_runs()
        target = r["nuaa_patience_12"] / "status.json"

        def touch(ctx, rec, role):
            if (role, rec["key"]) == ("test", "p4_LivenessNet_fold3_seed42"):
                target.write_bytes(target.read_bytes() + b"\n")
        out, _ = self.eval_fail("status.json changed during the evaluation", runs=r, score=fake_score(on_call=touch))
        no_results(self, out / EVAL_ID)

    def test_one_class_counts_cross_checked_with_the_runs(self):
        r = self.w.copy_runs()

        def change(rows):
            rows[0]["one_class"] = str(int(rows[0]["one_class"]) - 1)
        rewrite(r["nuaa_patience_4"], "public", "one_class.csv", csv_change(change))
        rc, o, e = run_eval(self.w.eval_argv("check-inputs", runs=r), eval_hooks(self.w))
        self.assertEqual(rc, 1, (o + e)[-2000:])
        self.assertIn("one-class counts of nuaa_patience_4 differ from public/one_class.csv", e)
        r = self.w.copy_runs()                               # il secondo run controllato anche lui
        rewrite(r["nuaa_patience_12"], "public", "one_class.csv", csv_change(change))
        rc, o, e = run_eval(self.w.eval_argv("check-inputs", runs=r), eval_hooks(self.w))
        self.assertEqual(rc, 1, (o + e)[-2000:])
        self.assertIn("one-class counts of nuaa_patience_12 differ from public/one_class.csv", e)
        r = self.w.copy_runs()                               # anche la colonna di testo

        def text(rows):
            rows[-1]["text"] = rows[-1]["text"] + " "
        rewrite(r["nuaa_patience_4"], "public", "one_class.csv", csv_change(text))
        rc, o, e = run_eval(self.w.eval_argv("check-inputs", runs=r), eval_hooks(self.w))
        self.assertEqual(rc, 1, (o + e)[-2000:])
        self.assertIn("one-class counts of nuaa_patience_4 differ from public/one_class.csv", e)

    def test_installed_statistics_of_different_lengths(self):
        r = self.w.copy_runs()

        def shorter(d):
            d["layers"]["bn1_2"]["installed_moving_mean_float32"] = d["layers"]["bn1_2"]["installed_moving_mean_float32"][:-1]
        rewrite(r["nuaa_patience_12"], "restricted", "LivenessNet_fold2_seed44_bn_statistics.json", json_change(shorter))
        rc, o, e = run_eval(self.w.eval_argv("check-inputs", runs=r), eval_hooks(self.w))
        self.assertEqual(rc, 1, (o + e)[-2000:])
        self.assertIn("bn1_2: installed mean and variance of different lengths", e)

    def test_card_without_the_fields_of_the_degenerate_count(self):
        r = self.w.copy_runs()
        rewrite(r["nuaa_patience_4"], "restricted", "ResNet-18_fold2_seed43_card.json", json_change(lambda d: d.pop("val_acc_at_best")))
        rc, o, e = run_eval(self.w.eval_argv("check-inputs", runs=r), eval_hooks(self.w))
        self.assertEqual(rc, 1, (o + e)[-2000:])
        self.assertIn("ResNet-18_fold2_seed43_card.json lacks restored_epoch, val_acc_at_best or best_val_loss", e)


# ============================================================================ Z TensorFlow

def tf_with_threads():
    return CR.tf_with_threads()


class TestZAScoreModel(unittest.TestCase):
    """score_model sui modelli veri: stato ordinario = BR.evaluate del checkpoint; stato ristimato = BR.evaluate dopo
    l'assegnazione a mano delle statistiche salvate; statistiche sbagliate o di un altro checkpoint rifiutate."""

    def setUp(self):
        need_tf(self)
        self.tf = tf_with_threads()
        self.w = world()

    def model_rec(self, network, shape=(8, 8, 3)):
        import tensorflow as tf
        m = AM.build(network, shape, CPA.ref_arch() if network == "LivenessNet" else None)
        rng = np.random.default_rng(7)
        m.set_weights([w + rng.normal(0, 0.05, w.shape).astype(np.float32) for w in m.get_weights()])
        d = Path(tempfile.mkdtemp(dir=self.w.tmp))
        path = d / f"{network}.keras"
        m.save(path)
        order = R.expected_orders()[network]
        fit = CR.frames_part(23, 1)
        copy = CR.clone(tf, m)
        stats = R.reestimate(copy, fit, order, 16)
        rec = {"key": f"p4_{network}", "network": network, "weights": path, "weights_sha256": sha(path.read_bytes()), "order": order,
               "installed": {n: (stats[n]["mean32"], stats[n]["var32"]) for n in order}}
        return rec, m, copy

    def ctx(self):
        cfg_pr = json.loads(self.w.cfg_path.read_text())
        ctx = E.Ctx()
        ctx.cfg_pr, ctx.batch, ctx.sg = cfg_pr, 16, 1e-4
        return ctx

    def test_two_states_equal_to_the_evaluator(self):
        for net in AM.NETWORKS:
            rec, m, copy = self.model_rec(net)
            dev = CR.frames_part(19, 3)
            try:
                out = E.score_model(self.ctx(), rec, {"dev": dev})
            except Exception as e:  # noqa: BLE001  (un difetto di score_model è un'asserzione fallita, non un errore)
                self.fail(f"{net}: score_model stopped on a correct checkpoint and correct statistics: {type(e).__name__}: {e}")
            ev, _ = BR.evaluate(m, dev, 16, "o")
            self.assertEqual(out["ordinary"]["dev"]["p"].tobytes(), ev["p"][:, 1].tobytes(), net)
            ev2, _ = BR.evaluate(copy, dev, 16, "r")
            self.assertEqual(out["re-estimated"]["dev"]["p"].tobytes(), ev2["p"][:, 1].tobytes(), net)
            self.assertFalse(np.array_equal(out["ordinary"]["dev"]["p"], out["re-estimated"]["dev"]["p"]), net)

    def test_softmax_tolerance_applied(self):
        """Il controllo del softmax dei logit della PR #27 si applica ai punteggi: con una tolleranza negativa (nessuno
        scarto ammesso, nemmeno 0) la valutazione si ferma."""
        rec, _, _ = self.model_rec("LivenessNet")
        ctx = self.ctx()
        ctx.sg = -1.0
        with self.assertRaisesRegex(FR.Failure, "the softmax of the logits differs from the probabilities"):
            E.score_model(ctx, rec, {"dev": CR.frames_part(5, 2)})

    def test_wrong_statistics_refused(self):
        """Statistiche salvate di forma sbagliata: rifiutate; statistiche uguali a quelle del checkpoint (nessuna ristima
        reinstallata) per uno strato: BR.tensor_problems ferma la valutazione."""
        rec, m, _ = self.model_rec("LivenessNet")
        n = rec["order"][0]
        mean, var = rec["installed"][n]
        with self.assertRaises(Exception):
            E.score_model(self.ctx(), {**rec, "installed": {**rec["installed"], n: (mean, var[:-1])}}, {"dev": CR.frames_part(5, 2)})
        own = (np.asarray(m.get_layer(n).moving_mean.numpy(), np.float32), np.asarray(m.get_layer(n).moving_variance.numpy(), np.float32))
        with self.assertRaisesRegex(FR.Failure, "moving tensors equal to the original"):
            E.score_model(self.ctx(), {**rec, "installed": {**rec["installed"], n: own}}, {"dev": CR.frames_part(5, 2)})

    def test_checkpoint_sha_checked_before_loading(self):
        rec, _, _ = self.model_rec("LivenessNet")
        rec["weights_sha256"] = "0" * 64
        with self.assertRaisesRegex(FR.Failure, "SHA-256 changed just before loading"):
            E.score_model(self.ctx(), rec, {"dev": CR.frames_part(5, 2)})


class TestZBRealEvaluation(unittest.TestCase):
    """Una valutazione vera: due run veri del runner su CPU nel mondo 8×8 (trial_trainer vero accorciato a una epoca,
    checkpoint e ristima veri), poi evaluate con TensorFlow vero: PASS, riproduzione del dev con scarto 0, punteggi
    riservati del test uguali a BR.evaluate del checkpoint con e senza le statistiche reinstallate."""

    def setUp(self):
        need_tf(self)
        tf_with_threads()

    def test_real_runs_and_evaluation(self):
        import tensorflow as tf
        w = world()

        def short(data, training, network, seed, build_model, cap_check, timer):
            card, model = PAC.trial_trainer(data, {**training, "max_epochs": 1}, network, seed, build_model, cap_check, timer)
            return {**card, "max_epochs": training["max_epochs"]}, model
        out_runs = w.tmp / "archrun_real"
        h = CR.base_hooks(w, ARCH_LOADER=lambda path: CPA.ref_arch(), TRAINER=short, TF_MODULE=lambda: tf)
        saved = dict(w.run_dirs)
        try:
            runs = w.make_runs(out_runs, h)
        finally:
            w.run_dirs = saved
        out = fresh_out(w)
        hooks = CR.base_hooks(w, TF_MODULE=lambda: tf)
        hooks["e"] = {"SCORE_MODEL": None, "PLAN_HOOK": None, "UNITS_HOOK": None}
        rc, o, e = run_eval(w.eval_argv(runs=runs, out=out), hooks)
        self.assertEqual(rc, 0, (o + e)[-4000:])
        folder = out / EVAL_ID
        st = json.loads((folder / "status.json").read_text())
        self.assertEqual(st["status"], "PASS")
        self.assertEqual(max(r["max_abs_diff"] for r in st["dev_reproduction"]), 0.0)
        cfg, _ = E.load_config(w.eval_cfg)
        with injected(**CR.base_hooks(w)):
            pins = E.load_pins(cfg, w.base)
            inp = PAC.load_inputs(pins.cfg_pr, pins.pcfg, w.base)
            FR.load_fold_manifests(SR.fr_view(pins.pcfg), w.base, inp)
        store = FR.FrameStore(inp, inp.X_extracted, inp.extracted_keys)
        name, net, fold = "ResNet-18_fold2_seed43", "ResNet-18", 2
        test = FR.assemble(store, inp.folds[fold]["test"])
        rd = runs["nuaa_patience_12"] / "restricted"
        m = tf.keras.models.load_model(str(rd / f"{name}.keras"), compile=False)
        ev, _ = BR.evaluate(m, test, 16, "o")
        rows = read_csv(folder / "restricted" / f"p12_{name}{E.TEST_SUFFIX}")
        n = len(test["keys"])
        self.assertEqual([r["score_attack"] for r in rows[:n]], [repr(float(v)) for v in ev["p"][:, 1]])
        s = json.loads((rd / f"{name}_bn_statistics.json").read_text())
        for layer in s["order"]:
            m.get_layer(layer).moving_mean.assign(np.array(s["layers"][layer]["installed_moving_mean_float32"], np.float32))
            m.get_layer(layer).moving_variance.assign(np.array(s["layers"][layer]["installed_moving_variance_float32"], np.float32))
        ev2, _ = BR.evaluate(m, test, 16, "r")
        self.assertEqual([r["score_attack"] for r in rows[n:]], [repr(float(v)) for v in ev2["p"][:, 1]])
        self.assertEqual([r["sample_id"] for r in rows[:n]], test["keys"])
        self.assertEqual(rows[0]["network"], net)


REPO_DIRS = ("scripts", "docs", "delivery", "results", "export", "src")
REPO_BEFORE = CFR.snapshot(*(ROOT / d for d in REPO_DIRS))


class ZZRepositoryUntouched(unittest.TestCase):
    """Ultima classe: il repository uguale, file per file (dimensione e data), a com'era all'import della suite; gli
    unici salti possibili sono quelli dichiarati."""

    def test_repository_untouched(self):
        self.assertEqual(CFR.snapshot(*(ROOT / d for d in REPO_DIRS)), REPO_BEFORE)

    def test_declared_skips_only(self):
        self.assertEqual(sorted(MISSING_INPUT), ["delivered configuration", "reference architecture", "tensorflow"])


def main() -> int:
    CR.tmp_outside_repo()
    argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a != "--verbose"] + (["-v"] if "--verbose" in sys.argv else [])
    try:
        result = unittest.main(argv=argv, exit=False, verbosity=1).result
    finally:
        for w in (WORLD, CR.WORLD):
            if w is not None:
                w.close()
    changed = CFR.snapshot(*(ROOT / d for d in REPO_DIRS)) != REPO_BEFORE
    skipped = [f"{t.id()}: {r}" for t, r in result.skipped]
    print(f"tests run: {result.testsRun}, failures: {len(result.failures)}, errors: {len(result.errors)}, skipped: {len(skipped)}")
    for s in skipped:
        print(f"skipped {s}")
    if changed:
        print("ERROR: the repository (scripts, docs, delivery, results, export, src) changed during the tests")
    ok = result.wasSuccessful() and not changed
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
