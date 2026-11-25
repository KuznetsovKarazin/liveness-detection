"""
Test di scripts/pad_arch_bn_factorial.py (la ristima BN 2x2 di PROTOCOL.md §4 su fitting/dev), offline e sintetici: nessun
checkpoint, frame o punteggio vero è letto.

Due livelli:
  - senza un modello Keras (classi A-D): configurazione (insiemi chiusi, valori fissati, ambiti di make_config), la ricetta
    dell'augmentation letta dal sorgente con l'AST (il vero pad_arch_comparison.py e copie alterate), le estrazioni senza
    stato (funzioni pure di dominio, vista, frame e operazione; indipendenti da batch e ordine; viste 101-108), la guardia
    delle letture (identità dei file, maiuscole, collegamenti fisici e simbolici, zone ammesse, regola fuori dal
    repository) e il rifiuto di ogni file dei genitori fuori dall'elenco, i confronti (tolleranze, pareggi a 0,5, dev a
    una classe, banda della frequenza delle maschere);
  - con TensorFlow su CPU (classi E-Z): l'augmentation operazione per operazione e intera contro gli strati Keras della
    ricetta forzati sugli stessi parametri (stessi parametri -> stessa immagine), gli estremi delle estrazioni contro gli
    strati Keras costruiti dalla ricetta letta dal sorgente, la deviazione standard del rumore, le clip, il dropout contro
    tf.nn.experimental.stateless_dropout, i controlli della struttura, i momenti (E00 per la nuova strada identico bit per
    bit a BR.recalibrate, catena = sotto-modello, unione sulle viste, conteggi, BN mai in training, coerenza che fallisce
    con altre maschere, stesse maschere per E01 ed E11), e la prova completa nel mondo sintetico della suite ARCHRUN
    (check_pad_arch_run.World: manifest, cache, frame 8x8, preregistrazione, configurazione ARCHRUN) con run genitori
    sintetici scritti qui nel formato ARCHRUN (veri checkpoint minuscoli di LivenessNet e ResNet-18 con pesi casuali e
    statistiche mobili perturbate, i loro file del dev e delle statistiche calcolati con pad_arch_run): PASS, metriche
    sigillate, stati ricostruiti dai file riservati, e ogni caso d'arresto (topologia, baseline, E00, E00 lungo la catena,
    tensori, coerenza, softmax, maschere, viste, duplicati, determinismo, fatti EMA, pesi e checkpoint cambiati, memoria,
    spazio su disco, cartella delle viste lasciata, guardia, tetto, SIGUSR1, SIGTERM e SIGHUP veri, albero sporco, nota
    nulla, tentativi, radice ufficiale con ingressi esterni, schede incoerenti).
Le cartelle temporanee stanno in TMPDIR, che deve stare fuori dalla radice del repository. Nulla è scritto nel repository
(controllo finale su results/archbn2x2, delivery/archbn2x2 e sul primo livello di results/, data/ ed export/).

  .venv/bin/python -I -B scripts/check_pad_arch_bn_factorial.py [--verbose] [-f]
"""
import contextlib
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
import types
import unittest
import warnings
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Keras 2.15 chiama random.randint con un float (keras/src/backend.py) dopo set_random_seed: l'avviso, con il percorso
# dell'installazione, finirebbe nell'uscita della suite; è filtrato solo lui (anche dentro il runner, vedi QuietRunner)
RANDRANGE_WARNING = "non-integer arguments to randrange"
warnings.filterwarnings("ignore", message=RANDRANGE_WARNING, category=DeprecationWarning)
HERE = Path(__file__).resolve()
# ARCHBN2X2_SUITE_WITHOUT_TF=1 simula una macchina senza TensorFlow e Keras (i loro import falliscono): serve a provare i
# salti dichiarati dove TensorFlow non c'è
WITHOUT_TF = os.environ.get("ARCHBN2X2_SUITE_WITHOUT_TF") == "1"
if WITHOUT_TF:
    sys.modules["tensorflow"] = None
    sys.modules["keras"] = None
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
import check_pad_arch_comparison as CPA  # noqa: E402
import check_pad_arch_run as CPR  # noqa: E402
import nuaa_bn_recalibration as BR  # noqa: E402
import nuaa_folds_run as FR  # noqa: E402
import nuaa_session_ablation_run as SR  # noqa: E402
import pad_arch_bn_factorial as F  # noqa: E402
import pad_arch_comparison as PAC  # noqa: E402
import pad_arch_models as AM  # noqa: E402
import pad_arch_run as R  # noqa: E402

ROOT = F.ROOT
MAKE_CONFIG = HERE.parents[1] / "delivery" / "archbn2x2" / "make_config.py"
MISSING_INPUT = {"reference architecture": "src/livedetection/src/architectures.py (LivenessNet of the reference code) is not present",
                 "tensorflow": "TensorFlow or Keras is not installed: the tests that build, re-estimate or score a Keras model, and those that "
                               "generate a configuration (the Keras module is pinned by SHA-256), are skipped",
                 "macOS tools": "zsh, caffeinate, pmset, sw_vers, top, memory_pressure or git is missing: the tests of launch.zsh are skipped"}
TF_AVAILABLE = (not WITHOUT_TF) and importlib.util.find_spec("tensorflow") is not None and importlib.util.find_spec("keras") is not None
STATE = {"commit": "abcdef0", "dirty": False}
DATE = "20261123"
SHAPE = (8, 8, 3)
UNITS_ALL = [{"run": "nuaa_patience_4", "network": "LivenessNet", "fold": 1, "seed": 42},
             {"run": "nuaa_patience_4", "network": "ResNet-18", "fold": 1, "seed": 42},
             {"run": "nuaa_patience_12", "network": "LivenessNet", "fold": 1, "seed": 42},
             {"run": "nuaa_patience_4", "network": "LivenessNet", "fold": 2, "seed": 43}]
LN1, RN1, LN12, LN2 = UNITS_ALL
# scritti per esteso, non copiati dal codice: un nome tolto da REQUIRED_PATTERNS deve far fallire la suite
SUITE_PATTERNS = ["fold[0-9]*_test.csv", "*_test_scores.csv", "*scores_test*", "*test*_scores*"]


def suite_guard():
    """La guardia di tutta la suite: results/archeval del repository e i nomi vietati (nessuna zona, nessuna regola esterna)."""
    F.install_guard([ROOT / "results" / "archeval"], SUITE_PATTERNS, link_exempt=F.link_exempt_paths())


def skip_missing(test: unittest.TestCase, what: str):
    test.skipTest(f"missing input: {MISSING_INPUT[what]}")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tmp_outside_repo():
    t = Path(tempfile.gettempdir()).resolve()
    if t == ROOT.resolve() or ROOT.resolve() in t.parents:
        raise SystemExit("TMPDIR is inside the repository: set it outside the repository root")


def load_make_config():
    spec = importlib.util.spec_from_file_location("make_config_archbn2x2", MAKE_CONFIG)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


TF = None


def tf():
    """TensorFlow con i thread della configurazione impostati prima di ogni operazione (la prova li reimposta uguali)."""
    global TF
    if TF is None:
        TF = CPR.tf_with_threads()
    return TF


def need_tf(test):
    if not TF_AVAILABLE:
        skip_missing(test, "tensorflow")


def need_arch(test):
    need_tf(test)
    if not CPA.REF_ARCH.is_file():
        skip_missing(test, "reference architecture")


def livenessnet(seed: int = 1):
    t = tf()
    t.keras.utils.set_random_seed(seed)
    return AM.build("LivenessNet", SHAPE, CPA.ref_arch())


def perturb(model, seed: int):
    """Statistiche mobili diverse da 0 e 1 iniziali (come dopo l'addestramento)."""
    rng = np.random.default_rng(seed)
    for l in model.layers:
        if type(l).__name__ == "BatchNormalization":
            g, b, m, v = l.get_weights()
            l.set_weights([g * np.float32(1.1), b + np.float32(0.05), rng.normal(0.2, 0.3, m.shape).astype(np.float32),
                           rng.uniform(0.5, 2.0, v.shape).astype(np.float32)])
    return model


def frames(n: int = 20, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (n,) + SHAPE, dtype=np.uint8)


def digests(n: int, prefix: str = "k") -> list:
    return [F.frame_digest(f"{prefix}{i:04d}") for i in range(n)]


def stale_left() -> list:
    return F.stale_views(env().vt) if ENV is not None else []


class FakeCtx:
    """Il ctx di una ristima fuori da una prova: nessun tetto, nessun limite di memoria."""

    def step(self, what):
        pass

    def span_start(self):
        return (F.now(), 0.0, 0.0)

    def span_end(self, t0, cell, layer):
        return {"wall_seconds": 0.0}


@contextlib.contextmanager
def injected(f=None, r=None, br=None, sr=None):
    mods = ((F, f), (R, r), (BR, br), (SR, sr))
    saved = [(m, k, getattr(m, k)) for m, d in mods for k in (d or {})]
    try:
        for m, d in mods:
            for k, v in (d or {}).items():
                setattr(m, k, v)
        yield
    finally:
        for m, k, v in saved:
            setattr(m, k, v)


@contextlib.contextmanager
def mock_attr(obj, name, value):
    saved = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, saved)


# ============================================================================ mondo sintetico e run genitori

class Env:
    """check_pad_arch_run.World più: la struttura del repository che make_config.py si aspetta (delivery/archrun/
    pad_arch_run.json, scripts/pad_arch_run.py) sotto la radice degli ingressi, due run ARCHRUN sintetici sotto
    results/archrun con le unità di UNITS_ALL (veri checkpoint minuscoli, file del dev e statistiche scritti con
    pad_arch_run) e una nota fissata."""

    def __init__(self):
        active = F.GUARD["active"]
        F.GUARD["active"] = False                  # il mondo scrive lo schema dei fold (fold<k>_test.csv): la prova non lo legge mai
        try:
            self.w = CPR.World()
        finally:
            F.GUARD["active"] = active
        self.base = self.w.base
        self.tmp = self.w.tmp
        self.hooks = CPR.base_hooks(self.w)
        (self.base / "delivery" / "archrun").mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.w.run_cfg, self.base / "delivery" / "archrun" / "pad_arch_run.json")
        (self.base / "scripts").mkdir(exist_ok=True)
        shutil.copy2(R.__file__, self.base / "scripts" / "pad_arch_run.py")
        (self.tmp / "cfg").mkdir()
        self.note = self.tmp / "cfg" / "NOTE.md"
        self.note.write_text("# frozen note (synthetic)\n", encoding="utf-8")
        self.mc = load_make_config()
        self.parents_built = False
        # la cartella temporanea della prova: non deve contenere la radice degli ingressi (il mondo sta in TMPDIR)
        self.vt = self.tmp / "trial_tmp"
        self.vt.mkdir()

    def inputs(self):
        cfg_run, _ = R.load_config(self.base / "delivery" / "archrun" / "pad_arch_run.json")
        with CPR.injected(**self.hooks):
            pins = R.load_pins(cfg_run, self.base)
            inp = PAC.load_inputs(pins.cfg_pr, pins.pcfg, self.base)
        return cfg_run, pins, inp

    def run_dir(self, r: str) -> Path:
        return self.base / "results" / "archrun" / self.mc.PARENT_IDS[r]

    def build_parents(self):
        if self.parents_built:
            return
        t = tf()
        cfg_run, pins, inp = self.inputs()
        store = FR.FrameStore(inp, inp.X_extracted, inp.extracted_keys)
        for r in self.mc.PARENT_IDS:
            (self.run_dir(r) / "restricted").mkdir(parents=True, exist_ok=True)
            for k, u in enumerate([x for x in UNITS_ALL if x["run"] == r]):
                self.write_unit(t, cfg_run, pins, inp, store, self.run_dir(r) / "restricted", u, seed=10 * k + len(r))
            self.reseal(r)
        self.parents_built = True

    def reseal(self, r: str):
        """SHA256SUMS riservato e status.json di un run genitore dopo una modifica dei suoi file."""
        run = self.run_dir(r)
        sums = CPA.write_sums(run / "restricted")
        cfg_sha = sha((self.base / "delivery" / "archrun" / "pad_arch_run.json").read_bytes())
        st = {"run_id": self.mc.PARENT_IDS[r], "run": r, "status": "PASS", "config_sha256": cfg_sha, "restricted_sha256sums_sha256": sums}
        (run / "status.json").write_text(json.dumps(st, indent=1) + "\n", encoding="utf-8")

    @contextlib.contextmanager
    def altered_card(self, u: dict, mutate):
        """La scheda di un'unità alterata (con SHA256SUMS e status.json coerenti), ripristinata all'uscita."""
        self.build_parents()
        p = self.run_dir(u["run"]) / "restricted" / f"{PAC.training_name(u['network'], u['fold'], u['seed'])}_card.json"
        b = p.read_bytes()
        card = json.loads(b)
        mutate(card)
        p.write_text(json.dumps(card, indent=1) + "\n", encoding="utf-8")
        self.reseal(u["run"])
        try:
            yield
        finally:
            p.write_bytes(b)
            self.reseal(u["run"])

    @contextlib.contextmanager
    def altered_file(self, u: dict, suffix: str, mutate):
        """Un file per unità di un run genitore alterato (mutate: testo -> testo), con SHA256SUMS e status.json coerenti,
        ripristinato all'uscita."""
        self.build_parents()
        p = self.run_dir(u["run"]) / "restricted" / f"{PAC.training_name(u['network'], u['fold'], u['seed'])}{suffix}"
        b = p.read_bytes()
        p.write_text(mutate(b.decode("utf-8")), encoding="utf-8", newline="")
        self.reseal(u["run"])
        try:
            yield
        finally:
            p.write_bytes(b)
            self.reseal(u["run"])

    def write_unit(self, t, cfg_run, pins, inp, store, rdir: Path, u: dict, seed: int):
        net, fold = u["network"], u["fold"]
        name = PAC.training_name(net, fold, u["seed"])
        fit, dev = BR.frames(store, inp, "fitting", "arm_b", fold), BR.frames(store, inp, "dev", "arm_b", fold)
        t.keras.backend.clear_session()
        t.keras.utils.set_random_seed(seed)
        model = perturb(PAC.build_checked(pins.cfg_pr, net, CPA.ref_arch()), seed)
        path = rdir / f"{name}.keras"
        model.save(path)
        w = sha(path.read_bytes())
        orig = BR.load_model(path, w)
        order = R.expected_orders()[net]
        ev_o, _ = BR.evaluate(orig, dev, 16, "ordinary")
        copy = t.keras.models.clone_model(orig)
        copy.set_weights(orig.get_weights())
        stats = R.reestimate(copy, fit, order, 16)
        fit_after, final = BR.evaluate(copy, fit, 16, "fitting after", layers=order)
        dev_after, _ = BR.evaluate(copy, dev, 16, "dev after")
        cons = BR.consistency(stats, final, 1e-4, 1e-6)
        res = {"stats": stats, "cons": cons, "changes": {n: BR.change_summary(stats[n], 1e-3) for n in order},
               "fitting_after": {k: fit_after[k] for k in ("accuracy", "val_loss", "auroc", "n")}}
        rec = {"name": name, "network": net, "fold": fold, "seed": u["seed"], "weights_sha256": w}
        so, _ = R.dev_state(dev, ev_o["p"][:, 1], ev_o["z"], "ordinary", name)
        sr, _ = R.dev_state(dev, dev_after["p"][:, 1], dev_after["z"], "re-estimated", name)
        doc = R.stats_doc(rec, res, sr, order, cfg_run)
        doc["dev_ordinary"] = so
        (rdir / f"{name}_bn_statistics.json").write_text(PAC.json_text(doc, "stats"), encoding="utf-8")
        for state, ev, summ in (("ordinary", ev_o, so), ("re-estimated", dev_after, sr)):
            rows = R.score_rows(rec, state, dev, ev["p"][:, 1], ev["z"], summ)
            (rdir / f"{name}_dev_{state}.csv").write_text(FR.csv_text(R.SCORE_COLUMNS, rows), encoding="utf-8", newline="")
        card = {"network": net, "arch": net, "fold": fold, "train_seed": u["seed"], "early_stopping_patience": F.PARENT_RUNS[u["run"]],
                "weights_sha256": w, "restored_epoch": 2 + seed % 3, "n_fitting": len(fit["X"]), "batch": 16,
                "fitting_keys_sha256": BR.keys_sha(fit["rows"]), "dev_keys_sha256": BR.keys_sha(dev["rows"])}
        (rdir / f"{name}_card.json").write_text(json.dumps(card, indent=1) + "\n", encoding="utf-8")
        (rdir / f"{name}_dev_scorer.csv").write_text("not opened by the trial\n", encoding="utf-8")

    def card(self, u: dict) -> dict:
        p = self.run_dir(u["run"]) / "restricted" / f"{PAC.training_name(u['network'], u['fold'], u['seed'])}_card.json"
        return json.loads(p.read_text())

    def config(self, units=None, report=True, storage="memmap", duplicate="all", mc=("nuaa_patience_4", "nuaa_patience_12"), note=True,
               mem=1000.0, cap=60, mutate=None, path=None) -> Path:
        self.build_parents()
        a = types.SimpleNamespace(note=None, cap_minutes=cap, storage=storage, report="yes" if report else "no", duplicate_check=duplicate,
                                  mc_runs=",".join(mc) if mc else "", memory_limit_gb=mem)
        units = units or [LN1, RN1]
        doc = self.mc.build(self.base, "custom", sorted(units, key=lambda u: u["fold"]), a)
        doc["ema"] = {"updates_per_epoch": 2, "fitting_frames": 32, "train_batch": 16, "momentum": 0.99}
        doc["analysis"]["path"] = str(self.note)
        doc["analysis"]["sha256"] = sha(self.note.read_bytes()) if note else None
        if mutate:
            mutate(doc)
        p = path or (self.tmp / "cfg" / "archbn2x2.json")
        p.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        return p

    def argv(self, cfg: Path, cmd="trial", out=None, attempt=None, extra=(), official=False) -> list:
        a = [cmd, "--config", str(cfg), "--input-root", str(self.base)]
        if cmd == "trial":
            a += ["--out-root", str(out or (self.tmp / "runs"))] + ([] if official else ["--unofficial-out-root"]) + ["--date", DATE, "--i-am-authorized"]
        if attempt is not None:
            a += ["--attempt", str(attempt)]
        return a + list(extra)

    def run(self, cfg: Path, cmd="trial", out=None, f=None, r=None, br=None, sr=None, extra=(), attempt=None, argv=None, tmpdir=None) -> tuple:
        out = out or Path(tempfile.mkdtemp(prefix="runs-", dir=self.tmp))
        # la cartella che contiene il repository sostituita da una estranea: qui TMPDIR può stare accanto al repository
        fh = {"REPO_STATE": lambda root: dict(STATE), "TF_MODULE": tf, "REPO_PARENT": self.tmp / "not_the_repository_parent", **(f or {})}
        buf_o, buf_e = io.StringIO(), io.StringIO()
        saved_tmp = tempfile.tempdir
        tempfile.tempdir = str(tmpdir or self.vt)
        try:
            with CPR.injected(**self.hooks), injected(f=fh, r=r, br=br, sr=sr), contextlib.redirect_stdout(buf_o), contextlib.redirect_stderr(buf_e):
                rc = F.main(argv or self.argv(cfg, cmd, out, attempt, extra))
        finally:
            tempfile.tempdir = saved_tmp
            suite_guard()
        folder = next(iter(sorted(out.iterdir())), None) if out.is_dir() else None
        return rc, buf_o.getvalue(), buf_e.getvalue(), folder

    def close(self):
        self.w.close()


ENV = None


def env() -> Env:
    global ENV
    if ENV is None:
        ENV = Env()
    return ENV


def read_csv(p: Path) -> list:
    import csv
    return list(csv.DictReader(io.StringIO(p.read_text(encoding="utf-8"))))


# ============================================================================ A. configurazione

class TestAConfig(unittest.TestCase):
    def setUp(self):
        need_arch(self)
        self.e = env()

    def refused(self, mutate, needle):
        p = self.e.config(mutate=mutate, path=self.e.tmp / "bad.json")
        with self.assertRaises(F.Failure) as c:
            F.load_config(p)
        self.assertIn(needle, str(c.exception))

    def test_generated_configuration_accepted(self):
        cfg, h = F.load_config(self.e.config())
        self.assertEqual(cfg["views"]["seeds"], list(range(101, 109)))       # 101-108, non 100-107
        self.assertEqual(len(h), 64)

    def test_closed_sets_and_fixed_values(self):
        self.refused(lambda d: d.update(extra=1), "fields must be exactly")
        self.refused(lambda d: d["views"].update(seeds=list(range(100, 108))), "views.seeds")
        self.refused(lambda d: d["views"].update(halves=[[101, 102, 103, 104], [101, 102, 103, 104]]), "views.halves")
        self.refused(lambda d: d.update(threshold=0.4), "threshold must be 0.5")
        self.refused(lambda d: d["threads"].update(intra_op=8), "threads must be 10")
        self.refused(lambda d: d["tolerances"].update(statistics_rel="1e-4"), "tolerances.statistics_rel")
        self.refused(lambda d: d["recipe"]["operations"][6].update(args=[0.03]), "recipe.operations")
        self.refused(lambda d: d["recipe"].update(differences="none"), "per frame")
        self.refused(lambda d: d["dropout"]["networks"]["ResNet-18"].update(layers={"d": 0.1}), "ResNet-18 has no Dropout")
        self.refused(lambda d: d["dropout"]["networks"]["LivenessNet"].update(active=["dropout9"]), "dropout.networks.LivenessNet")
        self.refused(lambda d: d["read_guard"].update(forbidden_paths=[]), "read_guard")
        self.refused(lambda d: d["read_guard"].update(forbidden_name_patterns=["*_test_scores.csv"]), "read_guard")
        self.refused(lambda d: d["read_guard"].update(allowlisted_zones=["results"]), "read_guard")
        self.refused(lambda d: d["cells"].update(E11="x"), "cells")
        self.refused(lambda d: d.update(test="nothing"), "test must state exactly")
        self.refused(lambda d: d["scope"].update(units=[LN2, LN1]), "grouped by fold")
        self.refused(lambda d: d["scope"]["units"].append(dict(d["scope"]["units"][-1])), "listed twice")
        self.refused(lambda d: d["ema"].update(updates_per_epoch=3), "ema.updates_per_epoch")
        self.refused(lambda d: d["estimator"].update(batch=32), "estimator.batch")
        self.refused(lambda d: d["keras_functions"].update(functions=["transform"]), "keras_functions")
        self.refused(lambda d: d["parent_runs"]["nuaa_patience_4"].update(run_id="20261120-ARCHRUN-p12-7faa49f"), "parent_runs.nuaa_patience_4")
        self.refused(lambda d: d["views"].pop("free_space_margin_gb"), "views needs exactly")
        self.refused(lambda d: d["scope"].update(name="other"), "scope.name")

    def test_scopes_and_units(self):
        mc = self.e.mc
        self.assertEqual(mc.units_of("pilot"), [{"run": "nuaa_patience_4", "network": n, "fold": 1, "seed": 42} for n in AM.NETWORKS])
        self.assertEqual({u["run"] for u in mc.units_of("p4-24")}, {"nuaa_patience_4"})
        self.assertEqual(len(mc.units_of("p12-24")), 24)
        folds = [u["fold"] for u in mc.units_of("p4-24")]
        self.assertEqual(folds, sorted(folds))
        self.assertEqual(mc.units_of("pilot", "p12:ResNet-18:3:44,p4:LivenessNet:1:42")[0]["fold"], 1)
        self.assertEqual((mc.SCOPES["p4-24"]["mc"], mc.SCOPES["p12-24"]["mc"]), (["nuaa_patience_4"], []))
        a = types.SimpleNamespace(cap_minutes=None, duplicate_check="all", mc_runs="")
        with self.assertRaises(SystemExit):                       # un elenco esplicito non prende i valori di un altro ambito
            mc.scope_values("custom", a)


# ============================================================================ B. ricetta dal sorgente

class TestBRecipe(unittest.TestCase):
    def src(self):
        return Path(PAC.__file__).read_text(encoding="utf-8")

    def test_real_source_equals_the_recipe(self):
        r = F.check_recipe(self.src())
        self.assertEqual(r["operations"], [c for c, _, _ in F.RECIPE])

    def refused(self, old, new):
        s = self.src()
        self.assertEqual(s.count(old), 1, old)
        with self.assertRaises(F.Failure):
            F.check_recipe(s.replace(old, new))

    def test_altered_sources_refused(self):
        self.refused("tf.keras.layers.RandomRotation(0.03)", "tf.keras.layers.RandomRotation(0.05)")
        self.refused("tf.keras.layers.GaussianNoise(0.02)])", "tf.keras.layers.GaussianNoise(0.03)])")
        self.refused(", tf.keras.layers.GaussianNoise(0.02)])", "])")
        self.refused("value_range=(0.0, 1.0)", "value_range=(0.0, 255.0)")
        self.refused("    Xtr /= 255.0\n", "    Xtr /= 255.0\n    Xtr /= 255.0\n")                  # doppia divisione
        self.refused("    Xtr /= 255.0\n", "    pass\n")
        self.refused("aug(x, training=True)", "aug(x)")

    def test_ranges_follow_the_recipe(self):
        r = F.recipe_ranges()
        self.assertEqual(r["zoom_h"], (1 - 0.12, 1 + 0.12))
        alt = tuple((c, (0.08, 0.08), kw) if c == "RandomZoom" else (c, a, kw) for c, a, kw in F.RECIPE)
        self.assertEqual(F.recipe_ranges(alt)["zoom_w"], (1 - 0.08, 1 + 0.08))
        with self.assertRaises(F.Failure):
            F.recipe_ranges(tuple((c, ((0.1, 0.2),), kw) if c == "RandomRotation" else (c, a, kw) for c, a, kw in F.RECIPE))


# ============================================================================ C. estrazioni senza stato

class TestCDraws(unittest.TestCase):
    def test_pure_function_of_its_arguments(self):
        d = digests(3)
        a = F.draw_params(d, 101)
        b = F.draw_params(list(reversed(d)), 101)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k][::-1])                   # l'ordine dei frame non conta
        c = F.draw_params(d[1:2], 101)
        for k in a:
            self.assertEqual(a[k][1], c[k][0])                                # nemmeno il batch
        e = F.draw_params(d, 102)
        self.assertFalse(np.array_equal(a["angle"], e["angle"]))              # un'altra vista, altre estrazioni

    def test_flip_and_independent_sides(self):
        p = F.draw_params(digests(400), 105)
        self.assertTrue(set(np.unique(p["flip"])) <= {0.0, 1.0})
        self.assertTrue(0.4 < p["flip"].mean() < 0.6)                         # entrambi gli esiti, metà ciascuno
        self.assertFalse(np.array_equal(p["zoom_h"], p["zoom_w"]))           # altezza e larghezza estratte indipendenti
        self.assertFalse(np.array_equal(p["trans_h"], p["trans_w"]))

    def test_domains_and_keys(self):
        d = F.frame_digest("a")
        u1 = F.philox("augmentation", 101, d, "dropout1").random(4)
        u2 = F.philox("dropout", 101, d, "dropout1").random(4)
        self.assertFalse(np.array_equal(u1, u2))
        self.assertEqual(F.frame_digest("a"), hashlib.sha256(b"a").hexdigest())
        np.testing.assert_array_equal(F.philox("dropout", 101, d, "x").random(4), F.philox("dropout", 101, d, "x").random(4))
        with self.assertRaises(F.Failure):
            F.philox("other", 101, d, "x")

    def test_masks_by_frame_view_layer(self):
        d = digests(2)
        m1 = F.mask_of(d[0], 101, "dropout1", (4, 4, 2), 0.25)
        self.assertTrue(np.array_equal(m1, F.mask_of(d[0], 101, "dropout1", (4, 4, 2), 0.25)))
        self.assertFalse(np.array_equal(m1, F.mask_of(d[1], 101, "dropout1", (4, 4, 2), 0.25)))
        self.assertFalse(np.array_equal(m1, F.mask_of(d[0], 102, "dropout1", (4, 4, 2), 0.25)))
        self.assertFalse(np.array_equal(m1, F.mask_of(d[0], 101, "dropout2", (4, 4, 2), 0.25)))
        big = F.mask_of(d[0], 101, "dropout1", (200, 200), 0.25)
        self.assertAlmostEqual(1 - big.mean(), 0.25, delta=0.01)

    def test_digests_from_the_keys(self):
        keys = ["attack/a.jpg", "client/b.jpg"]
        self.assertEqual(F.fit_digests(keys), [hashlib.sha256(k.encode()).hexdigest() for k in keys])
        self.assertEqual(F.fit_digests(keys[::-1]), F.fit_digests(keys)[::-1])

    def test_view_seeds(self):
        self.assertEqual(F.VIEW_SEEDS, tuple(range(101, 109)))
        self.assertEqual(F.HALVES, ((101, 102, 103, 104), (105, 106, 107, 108)))


# ============================================================================ D. guardia e confronti

class TestDGuardAndComparisons(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="guard-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(suite_guard)
        F.GUARD["active"] = False                       # la guardia della suite vieta questi nomi: si creano con la guardia spenta
        for rel in ("archeval/x.csv", "ok/fold3_test.csv", "ok/a_test_scores.csv", "ok/official_test.csv", "ok/scores_test_dev.csv",
                    "repo/results/declared/a.csv", "repo/results/other/b.csv", "repo/data/c.npz", "repo/code.py", "outside/allowed/d.txt",
                    "outside/private/e.txt", "archeval/y.csv", "archeval/sub/z.csv"):
            (self.tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.tmp / rel).write_text("x")
        os.link(self.tmp / "archeval" / "x.csv", self.tmp / "ok" / "innocent.csv")      # collegamento fisico
        os.symlink(self.tmp / "archeval", self.tmp / "ok" / "alias")                     # collegamento simbolico
        os.symlink(self.tmp / "archeval" / "y.csv", self.tmp / "outside" / "allowed" / "flink.csv")          # simbolico a un file vietato
        os.symlink(self.tmp / "repo" / "results" / "other" / "b.csv", self.tmp / "outside" / "allowed" / "rlink.csv")   # a un file non dichiarato
        os.symlink(self.tmp / "archeval" / "sub", self.tmp / "outside" / "allowed" / "sublink")             # per il caso '..' 
        suite_guard()

    def probe(self, paths) -> dict:
        """Per ogni percorso, se l'apertura è rifiutata, con la guardia ristretta attiva solo durante le aperture (un
        fallimento del test deve poter leggere i sorgenti per la traccia: la guardia della suite è rimessa prima degli assert)."""
        r = self.tmp / "repo"
        out = {}
        F.install_guard([self.tmp / "archeval"], SUITE_PATTERNS, zones=[(r / "results", [r / "results" / "declared"]), (r / "data", [])],
                        inside=[r, self.tmp / "ok", self.tmp / "archeval"], outside=[self.tmp / "outside" / "allowed"], link_exempt=[])
        try:
            for p in paths:
                try:
                    with open(p) as fh:
                        fh.read()
                    out[p] = False
                except F.Failure:
                    out[p] = True
            try:
                os.close(os.open(self.tmp / "archeval" / "x.csv", os.O_RDONLY))
                out["os.open"] = False
            except F.Failure:
                out["os.open"] = True
            # il caso del revisore: cartella corrente in un posto ammesso, dir_fd sul repository, nome relativo di un file non
            # dichiarato sotto results/; risolto sulla cartella corrente sembrerebbe ammesso: solo la regola sui percorsi
            # relativi di os.open lo rifiuta
            fd = os.open(self.tmp / "repo", os.O_RDONLY)
            cwd = os.getcwd()
            os.chdir(self.tmp / "outside" / "allowed")
            try:
                os.close(os.open("results/other/b.csv", os.O_RDONLY, dir_fd=fd))
                out["dir_fd"] = False
            except F.Failure:
                out["dir_fd"] = True
            finally:
                os.chdir(cwd)
                os.close(fd)
            hits = len(F.GUARD["hits"])
        finally:
            suite_guard()
        out["hits"] = hits
        return out

    def test_guard_by_identity_and_lower_case(self):
        t = self.tmp
        refused = [t / "archeval" / "x.csv", t / "ARCHEVAL" / "x.csv", t / "ok" / "fold3_test.csv", t / "ok" / "FOLD3_TEST.csv",
                   t / "ok" / "fold3_test.CSV", t / "ok" / "a_test_scores.csv", t / "ok" / "scores_test_dev.csv", t / "ok" / "innocent.csv",
                   t / "ok" / "alias" / "x.csv", t / "outside" / "allowed" / "flink.csv", t / "outside" / "allowed" / "rlink.csv",
                   Path(str(t / "outside" / "allowed" / "sublink") + "/../y.csv")]
        r = self.probe(refused + [t / "ok" / "official_test.csv"])
        self.assertEqual({str(p.relative_to(t)) for p in refused if not r[p]}, set())
        self.assertFalse(r[t / "ok" / "official_test.csv"])                   # la lista ufficiale di NUAA: ammessa
        self.assertTrue(r["os.open"])
        self.assertTrue(r["dir_fd"])
        self.assertGreaterEqual(r["hits"], 14)

    def test_zones_and_outside(self):
        t = self.tmp
        ok = [t / "repo" / "results" / "declared" / "a.csv", t / "repo" / "code.py", t / "outside" / "allowed" / "d.txt"]
        no = [t / "repo" / "results" / "other" / "b.csv", t / "REPO" / "Results" / "other" / "b.csv", t / "repo" / "data" / "c.npz",
              t / "outside" / "private" / "e.txt"]
        r = self.probe(ok + no)
        self.assertEqual([str(p.relative_to(t)) for p in ok if r[p]], [])
        self.assertEqual([str(p.relative_to(t)) for p in no if not r[p]], [])
        F.GUARD["active"] = False
        try:
            inert = (t / "archeval" / "x.csv").read_text()
        finally:
            F.GUARD["active"] = True
        self.assertEqual(inert, "x")                                           # spenta: inerte

    def test_parent_files_listed(self):
        for s in ("_dev_scorer.csv", "_test_scores.csv", "_card.json.bak"):
            with self.assertRaises(F.Failure):
                F.parent_file(Path("/x"), "LivenessNet_fold1_seed42", s)
        self.assertEqual(F.parent_file(Path("/x"), "n", ".keras").name, "n.keras")

    def test_rates_at_05_and_ties(self):
        y = np.array([1, 1, 0, 0])
        r = F.rates_05(y, [0.5, 0.9, 0.5, 0.2])                    # 0,5 è bona fide
        self.assertEqual((r["apcer_05"], r["bpcer_05"], r["acer_05"]), (0.5, 0.0, 0.25))
        one = F.rates_05(np.array([1, 1]), [0.1, 0.9])
        self.assertIsNone(one["bpcer_05"])
        self.assertIsNone(one["acer_05"])

    def test_score_compare(self):
        cfg = {"tolerances": {"dev_probability_max_abs_diff": 1e-5, "dev_decisions_discordant": 0}}
        y = np.array([1, 0, 1, 0])
        q = np.array([0.9, 0.1, 0.6, 0.4])
        thr = F.eer_or_none(y, q)
        p = q.copy()
        p[0] += 5e-6
        self.assertTrue(F.score_compare(p, q, y, thr, cfg)["pass"])
        p[0] += 2e-5
        self.assertFalse(F.score_compare(p, q, y, thr, cfg)["pass"])
        self.assertFalse(F.score_compare(q + 5e-6, q, y, thr, cfg)["pass"])        # un punteggio sulla soglia EER la attraversa
        self.assertFalse(F.score_compare(q, q, y, thr + 2e-5, cfg)["pass"])        # il valore della soglia ricalcolata conta
        c = F.score_compare(np.array([0.9, 0.1, 0.6, 0.5000001]), np.array([0.9, 0.1, 0.6, 0.4999999]), y, thr, cfg)
        self.assertEqual(c["discordant_05"], 1)
        self.assertFalse(c["pass"])

    def test_stat_compare(self):
        a = {"l": {"mean32": np.float32([1.0, 2.0]), "var32": np.float32([1.0, 1.0])}}
        b = {"l": {"mean32": np.float32([1.0, 2.0001]), "var32": np.float32([1.0, 1.0])}}
        self.assertTrue(F.stat_compare(a, a, 1e-4, 1e-6)["identical"])
        c = F.stat_compare(a, b, 1e-4, 1e-6)
        self.assertFalse(c["identical"])
        self.assertEqual(c["outside"], 0)
        self.assertEqual(F.stat_compare(a, b, 1e-6, 1e-7)["outside"], 1)

    def test_mask_frequency_band(self):
        c = F.MaskCounter()
        c.add("d", np.array([True] * 7500 + [False] * 2500))
        self.assertTrue(F.mask_frequency(c, {"d": 0.25}, 5)["d"]["pass"])
        c2 = F.MaskCounter()
        c2.add("d", np.array([True] * 7000 + [False] * 3000))
        self.assertFalse(F.mask_frequency(c2, {"d": 0.25}, 5)["d"]["pass"])
        self.assertFalse(F.mask_frequency(F.MaskCounter(), {"d": 0.25}, 5)["d"]["pass"])     # nulla contato

    def test_montecarlo_scale_without_reduction(self):
        recs = [{"run": "r", "network": "n", "label": f"u{i}", "metric": "acer_05", "effect": "augmentation", "value": v, "h": h}
                for i, (v, h) in enumerate(((0.10, 0.04), (0.03, -0.02), (0.06, 0.06)))]
        rows = F.montecarlo_rows(recs)
        s = (sum(h * h for h in (0.04, -0.02, 0.06)) / 3) ** 0.5 / 2
        mean = [r for r in rows if r[4] == "mean"][0]
        self.assertAlmostEqual(float(mean[9]), s)                                   # nessuna divisione per la radice delle unità
        self.assertAlmostEqual(float(mean[6]), 0.19 / 3)
        self.assertAlmostEqual(float(mean[7]), 0.08 / 3)
        self.assertAlmostEqual(float(mean[8]), 0.04 / 3)
        self.assertAlmostEqual(float(mean[10]), 0.19 / 3 / s)
        self.assertEqual(mean[11], "0")                                        # 0,063 > 2 x 0,0216
        u1 = [r for r in rows if r[4] == "u1"][0]
        self.assertTrue(s < 0.03 <= 2 * s)
        self.assertEqual(u1[11], "1")                                          # fra s e 2 s: entro la variabilità
        zero = F.montecarlo_rows([{**recs[0], "h": 0.0}])                  # s_MC = 0 con un effetto non nullo: NA, non «fuori»
        self.assertEqual((zero[0][10], zero[0][11]), ("NA", "NA"))
        nil = F.montecarlo_rows([{**recs[0], "h": 0.0, "value": 0.0}])
        self.assertEqual((nil[0][10], nil[0][11]), ("NA", "1"))
        u0 = [r for r in rows if r[4] == "u0"][0]
        self.assertAlmostEqual(float(u0[10]), 0.10 / s)
        self.assertEqual(u0[11], "1" if 0.10 <= 2 * s else "0")

    def test_reuse_rule(self):
        st = {"upstream": {"bn1_1": [], "bn1_2": [], "bn2_1": ["dropout1"], "bn2_2": ["dropout1"], "bn_dense": ["dropout1", "dropout2"]},
              "active": {"dropout1": 0.25, "dropout2": 0.25}}
        order = R.expected_orders()["LivenessNet"]
        self.assertEqual(F.reuse_eligible(st, "E01", order), ["bn1_1", "bn1_2"])
        self.assertEqual(F.reuse_eligible(st, "E11", order), ["bn1_1", "bn1_2"])
        self.assertEqual(F.reuse_eligible(st, "E10", order), [])
        rn = {"upstream": {}, "active": {}}
        o2 = R.expected_orders()["ResNet-18"]
        self.assertEqual(F.reuse_eligible(rn, "E01", o2), o2)
        self.assertEqual(F.reuse_eligible(rn, "E11", o2), o2)
        self.assertEqual(F.stochastic_upstream(st, "E11", "bn_dense"), ("augmentation", "dropout1", "dropout2"))


# ============================================================================ E. augmentation contro Keras (TensorFlow)

def keras_layer(cls, args, kwargs):
    t = tf()
    return getattr(t.keras.layers, cls)(*args, **kwargs)


def forced(layer, uniforms=(), seed=None, normal=None):
    """Lo strato Keras con le estrazioni sostituite: random_uniform restituisce i tensori dati in ordine,
    make_seed_for_stateless_op il seme dato, random_normal il rumore dato."""
    it = iter(uniforms)
    g = layer._random_generator
    g.random_uniform = lambda **k: next(it)
    if seed is not None:
        g.make_seed_for_stateless_op = lambda: seed
    if normal is not None:
        g.random_normal = lambda **k: normal
    return layer


class TestEAugmentation(unittest.TestCase):
    def setUp(self):
        need_tf(self)
        self.t = tf()
        rng = np.random.default_rng(3)
        self.x = rng.random((5, 16, 12, 3)).astype(np.float32)
        self.b = 5

    def params(self, **over):
        p = F.draw_params(digests(self.b), 101)
        p.update({k: np.asarray(v, np.float32) for k, v in over.items()})
        return p

    def same(self, layer, op, p, noise=None):
        got = layer(self.t.constant(self.x), training=True).numpy()
        mine = F.apply_op(self.t, op, self.t.constant(self.x), p, noise).numpy()
        np.testing.assert_array_equal(mine, got, err_msg=op)

    def test_each_operation_equals_the_keras_layer(self):
        t = self.t
        p = self.params()
        b = self.b
        seed = t.constant([7, 11], t.int64)
        u = t.random.stateless_uniform([b], seed=seed, minval=0, maxval=1.0)
        self.same(forced(keras_layer("RandomFlip", ("horizontal",), {}), seed=seed), "flip", self.params(flip=np.rint(u.numpy())))
        self.same(forced(keras_layer("RandomRotation", (0.03,), {}), [t.constant(p["angle"])]), "rotation", p)
        self.same(forced(keras_layer("RandomZoom", (0.12, 0.12), {}), [t.constant(p["zoom_h"][:, None]), t.constant(p["zoom_w"][:, None])]), "zoom", p)
        self.same(forced(keras_layer("RandomTranslation", (0.06, 0.06), {}), [t.constant(p["trans_h"][:, None]), t.constant(p["trans_w"][:, None])]),
                  "translation", p)
        self.same(forced(keras_layer("RandomBrightness", (0.15,), {"value_range": (0.0, 1.0)}), [t.constant(p["brightness"].reshape(b, 1, 1, 1))]),
                  "brightness", p)
        f = t.random.stateless_uniform(shape=[], minval=0.8, maxval=1.2, seed=seed)
        self.same(forced(keras_layer("RandomContrast", (0.2,), {}), seed=seed), "contrast", self.params(contrast=[f.numpy()] * b))
        noise = F.noise_of(digests(b), 101, self.x.shape[1:])
        self.same(forced(keras_layer("GaussianNoise", (0.02,), {}), normal=t.constant(noise)), "noise", p, noise)

    def test_whole_recipe_equals_the_keras_sequential(self):
        t = self.t
        p = self.params()
        b = self.b
        seed_f, seed_c = t.constant([1, 2], t.int64), t.constant([3, 4], t.int64)
        p["flip"] = np.rint(t.random.stateless_uniform([b], seed=seed_f, minval=0, maxval=1.0).numpy()).astype(np.float32)
        p["contrast"] = np.full(b, t.random.stateless_uniform(shape=[], minval=0.8, maxval=1.2, seed=seed_c).numpy(), np.float32)
        noise = F.noise_of(digests(b), 101, self.x.shape[1:])
        layers = [keras_layer(c, a, k) for c, a, k in F.RECIPE]
        aug = t.keras.Sequential(layers)
        aug.build((None,) + self.x.shape[1:])            # la costruzione simbolica chiama gli strati: si forzano le estrazioni dopo
        forced(layers[0], seed=seed_f)
        forced(layers[1], [t.constant(p["angle"])])
        forced(layers[2], [t.constant(p["zoom_h"][:, None]), t.constant(p["zoom_w"][:, None])])
        forced(layers[3], [t.constant(p["trans_h"][:, None]), t.constant(p["trans_w"][:, None])])
        forced(layers[4], [t.constant(p["brightness"].reshape(b, 1, 1, 1))])
        forced(layers[5], seed=seed_c)
        forced(layers[6], normal=t.constant(noise))
        got = aug(t.constant(self.x), training=True).numpy()
        x = t.constant(self.x)
        for op in F.OPS:
            x = F.apply_op(t, op, x, p, noise if op == "noise" else None)
        np.testing.assert_array_equal(x.numpy(), got)

    def test_draw_ranges_against_the_keras_layers_of_the_source(self):
        """Gli estremi delle estrazioni contro gli attributi degli strati Keras costruiti dalla ricetta letta dal sorgente:
        su 4.000 frame minimo e massimo entro l'1% dell'ampiezza dagli estremi, nessun valore fuori."""
        ops = {c: keras_layer(c, a, k) for c, a, k in F.recipe_from_source(Path(PAC.__file__).read_text())["operations"]}
        z, tr, ro, br, ct = ops["RandomZoom"], ops["RandomTranslation"], ops["RandomRotation"], ops["RandomBrightness"], ops["RandomContrast"]
        span = br._value_range[1] - br._value_range[0]
        want = {"angle": (ro.lower * 2 * np.pi, ro.upper * 2 * np.pi), "zoom_h": (1 + z.height_lower, 1 + z.height_upper),
                "zoom_w": (1 + z.width_lower, 1 + z.width_upper), "trans_h": (tr.height_lower, tr.height_upper),
                "trans_w": (tr.width_lower, tr.width_upper), "brightness": (br._factor[0], br._factor[1]), "contrast": (1 - ct.lower, 1 + ct.upper)}
        self.assertEqual(span, 1.0)
        p = F.draw_params(digests(4000, "r"), 103)
        for k, (lo, hi) in want.items():
            w = hi - lo
            self.assertGreaterEqual(p[k].min(), np.float32(lo) - 1e-6, k)
            self.assertLessEqual(p[k].max(), np.float32(hi) + 1e-6, k)
            self.assertLess(p[k].min() - lo, 0.01 * w, k)
            self.assertLess(hi - p[k].max(), 0.01 * w, k)
        n = F.noise_of(digests(6, "n"), 104, (256, 256, 3))
        self.assertAlmostEqual(float(n.std()) / ops["GaussianNoise"].stddev, 1.0, delta=0.01)
        self.assertAlmostEqual(float(n.mean()), 0.0, delta=1e-3)

    def test_augment_is_the_recipe_in_order(self):
        t = self.t
        X = frames(4)
        d = digests(4)
        p = F.draw_params(d, 107)
        x = t.constant(F.scaled(X))
        for op in F.OPS:
            x = F.apply_op(t, op, x, p, F.noise_of(d, 107, SHAPE) if op == "noise" else None)
        np.testing.assert_array_equal(F.augment(t, F.scaled(X), d, 107), x.numpy())
        self.assertEqual(F.OPS, ("flip", "rotation", "zoom", "translation", "brightness", "contrast", "noise"))

    def test_clips(self):
        t = self.t
        x = t.constant(np.full((2, 4, 4, 3), 0.95, np.float32))
        p = {"brightness": np.float32([0.15, -0.15])}
        out = F.apply_op(t, "brightness", x, p).numpy()
        self.assertEqual(out[0].max(), 1.0)                                  # luminosità con clip a [0, 1]
        y = np.zeros((1, 4, 4, 3), np.float32)
        y[0, 0, 0, :] = 1.0
        c = F.apply_op(t, "contrast", t.constant(y), {"contrast": np.float32([1.2])}).numpy()
        self.assertEqual(c.min(), 0.0)                                       # il limite inferiore della clip del contrasto agisce
        raw = t.image.adjust_contrast(t.constant(y), 1.2).numpy()
        self.assertLess(raw.min(), 0.0)
        self.assertGreater(c.max(), 1.0)                                     # il limite superiore (255) non agisce su dati in [0, 1]
        n = F.apply_op(t, "noise", t.constant(np.zeros((1, 2, 2, 1), np.float32)), {}, np.float32([[[[-0.5], [0.5]], [[0.0], [0.1]]]])).numpy()
        self.assertEqual(n.min(), -0.5)                                      # il rumore non ha clip

    def test_identity_parameters_give_the_scaled_frames(self):
        t = self.t
        X = frames(3)
        p = {"flip": np.zeros(3, np.float32), "angle": np.zeros(3, np.float32), "zoom_h": np.ones(3, np.float32), "zoom_w": np.ones(3, np.float32),
             "trans_h": np.zeros(3, np.float32), "trans_w": np.zeros(3, np.float32), "brightness": np.zeros(3, np.float32), "contrast": np.ones(3, np.float32)}
        x = t.constant(F.scaled(X))
        for op in F.OPS:
            x = F.apply_op(t, op, x, p, np.zeros((3,) + SHAPE, np.float32) if op == "noise" else None)
        np.testing.assert_allclose(x.numpy(), X.astype(np.float32) / 255.0, atol=1e-6)    # una divisione per 255, non di più
        self.assertLessEqual(x.numpy().max(), 1.0)

    def test_views_independent_of_batch_and_order(self):
        t = self.t
        X = frames(6)
        d = digests(6)
        a = F.augment(t, F.scaled(X), d, 103)
        b = np.concatenate([F.augment(t, F.scaled(X[i:i + 4]), d[i:i + 4], 103) for i in (0, 4)])
        np.testing.assert_array_equal(a, b)
        idx = [5, 0, 3, 1, 4, 2]
        c = F.augment(t, F.scaled(X[idx]), [d[i] for i in idx], 103)
        np.testing.assert_array_equal(a[idx], c)
        np.testing.assert_array_equal(a, F.augment(t, F.scaled(X), d, 103))    # ripetuta: identica (anche il rumore)
        self.assertFalse(np.array_equal(a, F.augment(t, F.scaled(X), d, 104)))

    def test_views_store_and_checks(self):
        t = self.t
        X = frames(5)
        tmp = Path(tempfile.mkdtemp(prefix="views-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        for storage in ("memory", "memmap"):
            v = F.Views(t, X, digests(5), (101, 102), 2, storage, tmp, (-0.5, 1.7), 0.0)
            np.testing.assert_array_equal(np.asarray(v.view(102)), F.augment(t, F.scaled(X), digests(5), 102))
            self.assertTrue(all(v.regenerated_first_batches(t).values()))
            v.close()
        self.assertEqual(list(tmp.iterdir()), [])                               # la cartella del memmap è tolta
        r = env().vt                                                            # e tolta anche con la guardia di un comando
        F.install_guard([], SUITE_PATTERNS, inside=[env().base], outside=[r], link_exempt=F.link_exempt_paths())
        try:
            v = F.Views(t, X, digests(5), (101,), 2, "memmap", r, (-0.5, 1.7), 0.0)
            v.close()
            hits = list(F.GUARD["hits"])
        finally:
            suite_guard()
        self.assertEqual((hits, F.stale_views(r)), ([], []))
        with injected(f={"VIEW_HOOK": lambda a: a * np.nan}):
            with self.assertRaises(F.ControlFailure) as c:
                F.Views(t, X, digests(5), (101,), 2, "memmap", tmp, (-0.5, 1.7), 0.0)
        self.assertEqual(c.exception.check, "views")
        self.assertEqual(list(tmp.iterdir()), [])                               # tolta anche dopo un arresto
        with injected(f={"VIEW_HOOK": lambda a: a + 5.0}):
            with self.assertRaises(F.ControlFailure):
                F.Views(t, X, digests(5), (101,), 2, "memory", tmp, (-0.5, 1.7), 0.0)
        with injected(f={"DISK_FREE": lambda p: 10}):
            with self.assertRaises(F.ControlFailure) as c:
                F.Views(t, X, digests(5), (101,), 2, "memmap", tmp, (-0.5, 1.7), 0.0)
        self.assertEqual(c.exception.check, "disk_space")
        with self.assertRaises(F.ControlFailure):                              # mai dentro una radice vietata
            F.Views(t, X, digests(5), (101,), 2, "memmap", tmp, (-0.5, 1.7), 0.0, refuse_inside=(tmp.parent,))


# ============================================================================ F. dropout e struttura

class TestFDropoutStructure(unittest.TestCase):
    def setUp(self):
        need_arch(self)
        self.t = tf()
        self.e = env()
        self.cfg, _ = F.load_config(self.e.config())

    def test_dropout_equals_tf_stateless_dropout(self):
        t = self.t
        x = t.constant(np.random.default_rng(0).random((3, 5, 5, 4)).astype(np.float32))
        seed = t.constant([5, 9], t.int64)
        u = t.random.stateless_uniform(x.shape, seed=seed, dtype=t.float32, alg="philox")
        ref = t.nn.experimental.stateless_dropout(x, 0.25, seed=seed, rng_alg="philox")
        np.testing.assert_array_equal(F.apply_dropout(t, x, (u >= 0.25).numpy(), 0.25).numpy(), ref.numpy())
        allkeep = F.apply_dropout(t, x, np.ones(x.shape, bool), 0.25).numpy()
        np.testing.assert_array_equal(allkeep, (x * t.convert_to_tensor(1 / 0.75, t.float32)).numpy())    # invertito: scala 1/(1-rate)

    def test_livenessnet_chain_accepted(self):
        m = livenessnet()
        st = F.structure(m, "LivenessNet", self.cfg)
        self.assertEqual(st["active"], {"dropout1": 0.25, "dropout2": 0.25})
        self.assertEqual(st["upstream"]["bn1_2"], [])
        self.assertEqual(st["upstream"]["bn2_1"], ["dropout1"])
        self.assertEqual(st["upstream"]["bn_dense"], ["dropout1", "dropout2"])

    def refused(self, model, net, cfg=None):
        with self.assertRaises(F.ControlFailure) as c:
            F.structure(model, net, cfg or self.cfg)
        self.assertEqual(c.exception.check, "topology")
        return str(c.exception)

    def test_structure_refusals(self):
        t = self.t
        L = t.keras.layers
        m = livenessnet()
        c = json.loads(json.dumps(self.cfg))
        c["dropout"]["networks"]["LivenessNet"]["layers"]["dropout1"] = 0.3
        self.refused(m, "LivenessNet", c)                                     # rate del checkpoint diverso
        m.get_layer("dropout1").noise_shape = (None, 1, 1, 16)
        self.refused(m, "LivenessNet")                                        # maschere per canale
        inp = L.Input(SHAPE, name="input_layer")
        a = L.Conv2D(2, 3, padding="same", name="c1")(inp)
        b = L.BatchNormalization(name="bn1_1")(a)
        x = L.Add(name="add")([a, b])
        self.refused(t.keras.Model(inp, x), "LivenessNet")                    # un ramo (e un Add)
        inp = L.Input(SHAPE, name="input_layer")
        a = L.Conv2D(2, 3, padding="same", name="c1")(inp)
        b = L.Conv2D(2, 3, padding="same", name="c2")(inp)
        self.refused(t.keras.Model(inp, [a, b]), "LivenessNet")               # un ramo con sole classi della catena
        inp = L.Input(SHAPE, name="input_layer")
        x = L.Conv2D(2, 3, padding="same", name="c1")(inp)
        x = L.BatchNormalization(name="bn1")(x)
        x = L.Dropout(0.1, name="d")(x)
        x = L.GlobalAveragePooling2D(name="gap")(x)
        out = L.Dense(2, activation="softmax", name="output")(x)
        self.assertIn("no dropout", self.refused(t.keras.Model(inp, out), "ResNet-18"))    # un comparatore con un Dropout
        inp = L.Input(SHAPE, name="input_layer")
        x = L.GaussianNoise(0.02, name="noise")(inp)
        x = L.GlobalAveragePooling2D(name="gap")(x)
        self.refused(t.keras.Model(inp, x), "ResNet-18")
        self.refused(t.keras.Model(inp, x), "LivenessNet")


# ============================================================================ G. momenti e celle (TensorFlow)

def stats_equal(a: dict, b: dict, keys=("mean64", "var64", "mean32", "var32")) -> bool:
    return all(np.asarray(a[n][k]).tobytes() == np.asarray(b[n][k]).tobytes() for n in b for k in keys)


class TestGMoments(unittest.TestCase):
    def setUp(self):
        need_arch(self)
        self.t = tf()
        self.e = env()
        self.cfg, _ = F.load_config(self.e.config())
        self.m = perturb(livenessnet(3), 3)
        self.st = F.structure(self.m, "LivenessNet", self.cfg)
        self.X = frames(21, 4)
        self.d = digests(21)
        self.order = R.expected_orders()["LivenessNet"]

    def copy(self):
        c = self.t.keras.models.clone_model(self.m)
        c.set_weights(self.m.get_weights())
        return c

    def test_T1_e00_by_the_new_path_is_the_estimator(self):
        a, b = self.copy(), self.copy()
        ref = BR.recalibrate(a, {"X": self.X}, self.order, 16)
        new = F.reestimate_cell(self.t, b, F.Source.clean(self.X, self.d, (0,)), self.order, 16, self.st, False, {}, FakeCtx(), "E00")
        self.assertTrue(stats_equal(new, ref))

    def test_T2_chain_equals_sub_model(self):
        a, b = self.copy(), self.copy()
        sub = F.reestimate_cell(self.t, a, F.Source.clean(self.X, self.d, (0,)), self.order, 16, self.st, False, {}, FakeCtx(), "sub")
        ch = F.reestimate_cell(self.t, b, F.Source.clean(self.X, self.d, (0,)), self.order, 16, self.st, False, {}, FakeCtx(), "chain",
                               force_chain=True)
        self.assertTrue(stats_equal(ch, sub))

    def test_T3_bn_in_inference_and_only_upstream_dropout(self):
        c = self.copy()
        before = BR.snapshot_weights(c)
        cnt = F.MaskCounter()
        F.moments_pass(self.t, c, F.Source.clean(self.X, self.d, (101, 102)), self.order, 16, self.st, True, cnt)
        self.assertEqual(BR.snapshot_weights(c), before)                      # nessuna statistica mobile aggiornata nel passaggio
        self.assertEqual(sorted(cnt.counts), ["dropout1", "dropout2"])        # dropout_dense mai attivo
        with injected(f={"UNIFORM": lambda rng, shape: np.ones(shape, np.float32)}):     # si tiene tutto: x / (1 - rate) agli strati attivi
            on = F.moments_pass(self.t, c, F.Source.clean(self.X, self.d, (101,)), ["bn1_2", "bn2_1"], 16, self.st, True)
        off = F.moments_pass(self.t, c, F.Source.clean(self.X, self.d, (101,)), ["bn1_2", "bn2_1"], 16, self.st, False)
        np.testing.assert_array_equal(on["bn1_2"].result()[0], off["bn1_2"].result()[0])     # a monte di nessun dropout: invariato
        self.assertFalse(np.array_equal(on["bn2_1"].result()[0], off["bn2_1"].result()[0]))

    def test_T4_masks_independent_of_batch_and_order(self):
        c = self.copy()
        a = F.moments_pass(self.t, c, F.Source.clean(self.X, self.d, (101,)), ["bn2_1"], 16, self.st, True)["bn2_1"]
        idx = list(range(20, -1, -1))
        b = F.moments_pass(self.t, c, F.Source.clean(self.X[idx], [self.d[i] for i in idx], (101,)), ["bn2_1"], 7, self.st, True)["bn2_1"]
        np.testing.assert_allclose(a.result()[0], b.result()[0], rtol=1e-12)
        np.testing.assert_allclose(a.result()[1], b.result()[1], rtol=1e-12)

    def test_T7_same_masks_for_clean_and_augmented_inputs(self):
        c = self.copy()
        keys = {"clean": [], "augmented": []}
        for kind in keys:
            src = F.Source(F.HALVES[0], 21, (lambda v, i, j: F.scaled(self.X[i:j]) * (0.5 if kind == "augmented" else 1.0)), self.d, kind)

            def rec(rng, shape, kind=kind):
                keys[kind].append((tuple(rng.bit_generator.state["state"]["key"]), shape))
                return rng.random(shape, dtype=np.float32)
            with injected(f={"UNIFORM": rec}):
                F.moments_pass(self.t, c, src, ["bn_dense"], 16, self.st, True)
        self.assertEqual(len(keys["clean"]), 4 * 21 * 2)                      # viste x frame x i due strati attivi
        self.assertEqual(keys["clean"], keys["augmented"])                     # E01 ed E11 condividono le maschere
        self.assertEqual({s for _, s in keys["clean"]}, {(4, 4, 16), (2, 2, 32)})     # maschere per elemento

    def test_T6_union_over_views(self):
        c = self.copy()
        sub = self.t.keras.Model(c.input, c.get_layer("bn2_1").input)
        views = [F.scaled(self.X)] * 3
        mom = F.moments_pass(self.t, c, F.Source.clean(self.X, self.d, (101, 102, 103)), ["bn2_1"], 16, self.st, False)["bn2_1"]
        allv = np.concatenate([np.asarray(sub(v, training=False), np.float64).reshape(-1, 32) for v in views])
        np.testing.assert_allclose(mom.result()[0], allv.mean(0), rtol=1e-12)
        np.testing.assert_allclose(mom.result()[1], allv.var(0), rtol=1e-10)
        self.assertEqual(mom.n, len(allv))
        seq = BR.Moments()                                                   # gli stessi batch nello stesso ordine: identico bit per bit
        for v in views:
            for i in range(0, 21, 16):
                seq.add(np.asarray(sub(v[i:i + 16], training=False)))
        self.assertEqual(seq.result()[0].tobytes(), mom.result()[0].tobytes())

    def test_counts_and_reuse(self):
        c = self.copy()
        ref = F.reestimate_cell(self.t, self.copy(), F.Source.clean(self.X, self.d, (0,)), self.order, 16, self.st, False, {}, FakeCtx(), "E00")
        reuse = {n: ref[n] for n in F.reuse_eligible(self.st, "E01", self.order)}
        out = F.reestimate_cell(self.t, c, F.Source.clean(self.X, self.d, F.HALVES[0]), self.order, 16, self.st, True, reuse, FakeCtx(), "E01")
        self.assertTrue(out["bn1_1"]["reused"] and out["bn1_2"]["reused"] and not out["bn2_1"]["reused"])
        self.assertEqual(out["bn2_1"]["count"], 4 * 21 * 4 * 4)                    # viste x frame x H x W
        self.assertEqual(out["bn_dense"]["count"], 4 * 21)
        self.assertEqual(out["bn1_1"]["mean32"].tobytes(), ref["bn1_1"]["mean32"].tobytes())

    def test_T8_only_moving_tensors_change(self):
        snap = BR.snapshot_weights(self.m)
        c = self.copy()
        st = F.reestimate_cell(self.t, c, F.Source.clean(self.X, self.d, (101, 102)), self.order, 16, self.st, True, {}, FakeCtx(), "E01")
        self.assertEqual(BR.tensor_problems(snap, BR.snapshot_weights(c), BR.moving_keys(c, self.order), st), [])
        self.assertEqual(BR.snapshot_weights(self.m), snap)

    def test_T9_consistency_fails_with_other_masks(self):
        c = self.copy()
        src = F.Source.clean(self.X, self.d, (101, 102))
        st = F.reestimate_cell(self.t, c, src, self.order, 16, self.st, True, {}, FakeCtx(), "E01")
        same = BR.consistency(st, F.moments_pass(self.t, c, src, self.order, 16, self.st, True), 1e-4, 1e-6)
        self.assertEqual(sum(v["channels_outside_tolerance"] for v in same.values()), 0)
        other = F.Source.clean(self.X, self.d, (103, 104))
        diff = BR.consistency(st, F.moments_pass(self.t, c, other, self.order, 16, self.st, True), 1e-4, 1e-6)
        self.assertGreater(sum(v["channels_outside_tolerance"] for v in diff.values()), 0)


# ============================================================================ Z. la prova completa (mondo sintetico)

PASS_RUN = {}


def pass_run() -> dict:
    """Una prova su LivenessNet e ResNet-18 del fold 1 a p4 e su LivenessNet del fold 2, tutto acceso (duplicati per ogni
    unità, metà, ripetizione), metriche delle celle pubbliche."""
    if not PASS_RUN:
        e = env()
        cfg = e.config(units=[LN1, RN1, LN2], duplicate="all")
        rc, out, err, run = e.run(cfg)
        PASS_RUN.update(rc=rc, out=out, err=err, run=run, cfg=cfg)
    return PASS_RUN


class TestZATrialPass(unittest.TestCase):
    def setUp(self):
        need_arch(self)
        self.p = pass_run()
        self.assertEqual(self.p["rc"], 0, (self.p["out"] + self.p["err"])[-4000:])
        self.run_ = self.p["run"]

    def test_status_and_times(self):
        st = json.loads((self.run_ / "status.json").read_text())
        self.assertEqual(st["status"], "PASS")
        for k in ("started", "finished", "wall_seconds", "process_cpu_seconds", "peak_rss_gb", "threads_rule", "machine_rule", "read_guard"):
            self.assertIn(k, st)
        self.assertEqual(st["complete_units"], ["p4_LivenessNet_fold1_seed42", "p4_ResNet-18_fold1_seed42", "p4_LivenessNet_fold2_seed43"])
        rows = read_csv(self.run_ / "public" / "timing.csv")
        self.assertTrue({"(unit)", "(views)"} <= {r["layer"] for r in rows})
        self.assertTrue(any(r["layer"] == "bn2_1" for r in rows))
        for r in rows:
            for k in ("started", "finished", "wall_seconds", "process_cpu_seconds"):
                self.assertNotEqual(r[k], "NA", (r, k))
        self.assertEqual(sorted(p.name for p in self.run_.iterdir()), ["public", "restricted", "run.log", "status.json"])   # nessuna vista qui
        self.assertEqual(stale_left(), [])                                    # e nessuna lasciata nella cartella temporanea

    def test_controls_pass(self):
        rows = read_csv(self.run_ / "public" / "controls.csv")
        self.assertTrue(all(r["pass"] == "1" for r in rows), [r for r in rows if r["pass"] != "1"])
        names = {(r["network"], r["control"]) for r in rows}
        for c in ("baseline", "E00", "structure", "weights"):
            self.assertIn(("LivenessNet", c), names)
            self.assertIn(("ResNet-18", c), names)
        self.assertIn(("LivenessNet", "E00_chain"), names)
        self.assertIn(("LivenessNet", "determinism"), names)
        self.assertIn(("LivenessNet", "duplicates_E01"), names)
        self.assertIn(("ResNet-18", "duplicates_E11"), names)
        e00 = [json.loads(r["detail"]) for r in rows if r["control"] == "E00"]
        self.assertTrue(all(d["statistics"]["identical"] and d["dev"]["identical"] for d in e00))     # riproduzione bit per bit
        det = next(json.loads(r["detail"]) for r in rows if r["control"] == "determinism")
        self.assertTrue(det["views_regenerated_identical"])

    def test_cells_effects_layers(self):
        cells = read_csv(self.run_ / "public" / "cells.csv")
        ln = [r for r in cells if r["network"] == "LivenessNet" and r["fold"] == "1"]
        self.assertEqual(sorted({(r["cell"], r["views"]) for r in ln}),
                         sorted([("E00", "all")] + [(c, v) for c in ("E10", "E01", "E11") for v in ("all", "101-104", "105-108")]))
        rn = {r["cell"]: r for r in cells if r["network"] == "ResNet-18" and r["views"] == "all"}
        for k in ("apcer_05", "bpcer_05", "acer_05"):
            self.assertEqual(rn["E01"][k], rn["E00"][k])                     # nessun dropout in ResNet-18: E01 = E00
            self.assertEqual(rn["E11"][k], rn["E10"][k])
        self.assertEqual(rn["E01"]["layers_reused"], "20")
        lnc = {r["cell"]: r for r in ln if r["views"] == "all"}
        self.assertEqual((lnc["E10"]["layers_reused"], lnc["E01"]["layers_reused"], lnc["E11"]["layers_reused"]), ("0", "2", "2"))
        self.assertNotEqual(lnc["E10"]["max_abs_dp_vs_e00"], "0.0")
        e = env().card(LN1)["restored_epoch"]
        self.assertEqual({r["ema_coefficient"] for r in ln}, {repr(0.99 ** (2 * e))})     # 0,99^(aggiornamenti per epoca x epoca)
        self.assertEqual({r["restored_epoch"] for r in ln}, {str(e)})
        eff = read_csv(self.run_ / "public" / "effects.csv")
        self.assertEqual({r["metric"] for r in eff}, {"acer_05", "apcer_05", "bpcer_05"})
        r0 = next(r for r in eff if r["network"] == "LivenessNet" and r["views"] == "all" and r["metric"] == "acer_05" and r["fold"] == "1")
        v = {c: float(lnc[c]["acer_05"]) for c in F.CELLS}
        self.assertAlmostEqual(float(r0["interaction_E11_minus_E10_minus_E01_plus_E00"]), v["E11"] - v["E10"] - v["E01"] + v["E00"])
        layers = read_csv(self.run_ / "public" / "layers.csv")
        bn11 = [r for r in layers if r["network"] == "LivenessNet" and r["cell"] == "E11" and r["views"] == "all" and r["layer"] == "bn1_1"]
        self.assertEqual(bn11[0]["reused"], "1")

    def test_montecarlo_from_the_effects(self):
        eff = read_csv(self.run_ / "public" / "effects.csv")
        mc = read_csv(self.run_ / "public" / "montecarlo.csv")
        cols = {"augmentation": "augmentation_E10_minus_E00", "dropout": "dropout_E01_minus_E00",
                "interaction": "interaction_E11_minus_E10_minus_E01_plus_E00"}
        for e, col in cols.items():
            sel = lambda v: {r["fold"]: float(r[col]) for r in eff if r["network"] == "LivenessNet" and r["metric"] == "acer_05" and r["views"] == v}  # noqa: E731
            a, h1, h2 = sel("all"), sel("101-104"), sel("105-108")
            hs = [h1[f] - h2[f] for f in a]
            s = (sum(h * h for h in hs) / len(hs)) ** 0.5 / 2
            row = next(r for r in mc if r["network"] == "LivenessNet" and r["metric"] == "acer_05" and r["effect"] == e and r["level"] == "mean")
            self.assertEqual(int(row["units"]), 2)
            self.assertAlmostEqual(float(row["s_mc"]), s)
            self.assertAlmostEqual(float(row["h_halves"]), sum(hs) / len(hs))
            self.assertAlmostEqual(float(row["effect_value"]), sum(a.values()) / len(a))

    def test_halves_use_four_views(self):
        for cell in ("E10", "E01", "E11"):
            for tag in ("101-104", "105-108"):
                doc = json.loads((self.run_ / "restricted" / f"p4_LivenessNet_fold1_seed42_{cell}_{tag}_bn.json").read_text())
                own = {n: v for n, v in doc["layers"].items() if not v["reused"]}
                self.assertTrue(own)
                self.assertEqual({v["views"] for v in own.values()}, {4}, (cell, tag))
                self.assertEqual(own["bn_dense"]["count"], 4 * 32)
        full = json.loads((self.run_ / "restricted" / "p4_LivenessNet_fold1_seed42_E10_all_bn.json").read_text())
        self.assertEqual(full["layers"]["bn_dense"]["views"], 8)

    def test_states_rebuilt_from_the_restricted_files(self):
        """Ogni stato salvato, assegnato al checkpoint genitore, dà i punteggi del dev salvati (il dev è valutato pulito)."""
        t = tf()
        e = env()
        _, _, inp = e.inputs()
        store = FR.FrameStore(inp, inp.X_extracted, inp.extracted_keys)
        for u, cell in ((LN1, "E11"), (LN1, "E10"), (RN1, "E10")):
            name = PAC.training_name(u["network"], u["fold"], u["seed"])
            label = F.unit_label(u)
            dev = BR.frames(store, inp, "dev", "arm_b", u["fold"])
            ck = e.run_dir(u["run"]) / "restricted" / f"{name}.keras"
            m = t.keras.models.load_model(str(ck), compile=False)
            doc = json.loads((self.run_ / "restricted" / f"{label}_{cell}_all_bn.json").read_text())
            for n, v in doc["layers"].items():
                m.get_layer(n).moving_mean.assign(np.array(v["installed_moving_mean_float32"], np.float32))
                m.get_layer(n).moving_variance.assign(np.array(v["installed_moving_variance_float32"], np.float32))
            ev, _ = BR.evaluate(m, dev, 16, "rebuilt")
            rows = read_csv(self.run_ / "restricted" / f"{label}_{cell}_all_dev.csv")
            self.assertEqual([r["sample_id"] for r in rows], dev["keys"])
            self.assertEqual([r["score_attack"] for r in rows], [repr(float(x)) for x in ev["p"][:, 1]], (label, cell))

    def test_restricted_and_public_texts(self):
        rdir = self.run_ / "restricted"
        names = sorted(p.name for p in rdir.iterdir())
        self.assertIn("p4_LivenessNet_fold1_seed42_E11_all_dev.csv", names)
        self.assertIn("p4_ResNet-18_fold1_seed42_E10_101-104_bn.json", names)
        self.assertIn("SHA256SUMS", names)
        doc = json.loads((rdir / "p4_LivenessNet_fold1_seed42_E10_all_bn.json").read_text())
        self.assertEqual(list(doc["layers"]), R.expected_orders()["LivenessNet"])
        _, _, inp = env().inputs()
        keys = list(inp.official)[:200]
        for p in (self.run_ / "public").iterdir():
            text = p.read_text(encoding="utf-8")
            self.assertFalse(any(k in text for k in keys), p.name)
            self.assertNotIn(str(Path.home()), text, p.name)


def stop_cfg(**kw) -> Path:
    e = env()
    return e.config(units=kw.pop("units", [LN1]), duplicate=kw.pop("duplicate", "none"), mc=kw.pop("mc", ()), **kw)


def nth_call(fn, n: int, replace):
    """fn con la chiamata n-esima (da 1) sostituita da replace(*args)."""
    calls = {"n": 0}

    def w(*a, **k):
        calls["n"] += 1
        return replace(*a, **k) if calls["n"] == n else fn(*a, **k)
    return w


class TestZBTrialStops(unittest.TestCase):
    def setUp(self):
        need_arch(self)
        self.e = env()

    def failed(self, rc, run, check=None):
        self.assertEqual(rc, 1)
        st = json.loads((run / "status.json").read_text())
        self.assertEqual(st["status"], "FAIL")
        self.assertFalse((run / "public" / "cells.csv").exists())
        if check:
            self.assertEqual((st.get("failed_check") or {}).get("check"), check, st.get("reason"))
        self.assertEqual(stale_left(), [])                                   # la cartella delle viste è tolta anche qui
        return st

    def test_sealed_metrics(self):
        rc, out, err, run = self.e.run(stop_cfg(report=False))
        self.assertEqual(rc, 0, (out + err)[-3000:])
        self.assertEqual(read_csv(run / "public" / "cells.csv"), [])
        self.assertEqual(read_csv(run / "public" / "effects.csv"), [])
        self.assertTrue(read_csv(run / "restricted" / "sealed_cells.csv"))
        self.assertTrue(read_csv(run / "public" / "controls.csv"))

    def test_topology_in_a_trial(self):
        def mut(d):
            d["dropout"]["networks"]["LivenessNet"]["layers"]["dropout1"] = 0.3
        rc, out, err, run = self.e.run(stop_cfg(mutate=mut))
        self.failed(rc, run, "topology")

    def test_e00_not_reproduced(self):
        orig = F.saved_stats
        rc, out, err, run = self.e.run(stop_cfg(), f={"saved_stats": lambda doc, order: {n: {**v, "mean32": v["mean32"] + np.float32(1e-3)}
                                                                                      for n, v in orig(doc, order).items()}})
        st = self.failed(rc, run, "E00")
        self.assertIn("statistics", st["failed_check"]["values"])

    def test_stored_moments_differ(self):
        orig = F.saved_stats
        rc, out, err, run = self.e.run(stop_cfg(), f={"saved_stats": lambda doc, order: {n: {**v, "old_mean": v["old_mean"] + np.float32(1.0)}
                                                                                      for n, v in orig(doc, order).items()}})
        st = self.failed(rc, run, "E00")
        self.assertFalse(st["failed_check"]["values"]["stored_statistics_identical"])

    def test_baseline_not_reproduced(self):
        orig = F.saved_scores

        def shifted(rows, dev, what):
            s, t = orig(rows, dev, what)
            return (s + 1e-3, t) if "ordinary" in what else (s, t)
        rc, out, err, run = self.e.run(stop_cfg(), f={"saved_scores": shifted})
        self.failed(rc, run, "baseline")

    def test_e00_chain_not_reproduced(self):
        orig = F.reestimate_cell

        def bumped(*a, **k):
            out = orig(*a, **k)
            if a[9].endswith("E00 chain"):
                n = next(iter(out))
                out[n] = {**out[n], "mean32": out[n]["mean32"] + np.float32(1.0)}
            return out
        rc, out, err, run = self.e.run(stop_cfg(), f={"reestimate_cell": bumped})
        self.failed(rc, run, "E00_chain")

    def test_consistency(self):
        orig = F.moments_pass

        def shifted(*a, **k):
            out = orig(*a, **k)
            if len(a[3]) > 1:                                                  # il passaggio di coerenza a più uscite
                m = out[a[3][-1]]
                m.mean = m.mean + 1.0
            return out
        rc, out, err, run = self.e.run(stop_cfg(), f={"moments_pass": shifted})
        self.failed(rc, run, "consistency")

    def test_trained_tensor_changed_in_a_cell(self):
        orig = F.reestimate_cell

        def touched(*a, **k):
            out = orig(*a, **k)
            if a[9].endswith(" E10"):
                g = a[1].get_layer("bn1_1").gamma
                g.assign(g.numpy() + 1.0)
            return out
        rc, out, err, run = self.e.run(stop_cfg(), f={"reestimate_cell": touched})
        self.failed(rc, run, "tensors")

    def test_softmax(self):
        orig = BR.evaluate

        def gap(model, part, batch, what, layers=()):
            r, m = orig(model, part, batch, what, layers)
            return ({**r, "softmax_gap": 1.0} if what.endswith("E10 dev") else r), m
        rc, out, err, run = self.e.run(stop_cfg(), br={"evaluate": gap})
        self.failed(rc, run, "softmax")

    def test_memory(self):
        rc, out, err, run = self.e.run(stop_cfg(mem=1.0), f={"MEMORY_PROBE": lambda: 2.0})
        st = self.failed(rc, run, "memory")
        self.assertEqual(st["failed_check"]["values"]["peak_rss_gb"], 2.0)

    def test_disk_space(self):
        rc, out, err, run = self.e.run(stop_cfg(), f={"DISK_FREE": lambda p: 1})
        self.failed(rc, run, "disk_space")

    def test_stale_views_folder(self):
        left = Path(tempfile.mkdtemp(prefix=F.VIEWS_PREFIX, dir=self.e.vt))
        try:
            rc, out, err, run = self.e.run(stop_cfg())
        finally:
            shutil.rmtree(left)
        self.assertEqual((rc, run), (1, None))
        self.assertIn("views folder", err)

    def test_non_finite_views(self):
        # NaN, non infinito: un NaN passa il controllo dei limiti (i confronti con NaN sono falsi), solo quello dei valori
        # finiti lo ferma
        rc, out, err, run = self.e.run(stop_cfg(), f={"VIEW_HOOK": lambda a: np.where(a > 0.9, np.nan, a)})
        st = self.failed(rc, run, "views")
        self.assertIn("non_finite_values", st["failed_check"]["values"])

    def test_mask_frequency(self):
        rc, out, err, run = self.e.run(stop_cfg(), f={"UNIFORM": lambda rng, shape: np.ones(shape, np.float32)})
        st = self.failed(rc, run, "masks")
        self.assertEqual(st["failed_check"]["values"]["layers"]["dropout1"]["zeros"], 0)

    def test_determinism(self):
        def bump(stats):
            n = next(iter(k for k, v in stats.items() if not v["reused"]))
            return {**stats, n: {**stats[n], "mean64": stats[n]["mean64"] * (1 + 1e-12)}}
        rc, out, err, run = self.e.run(stop_cfg(), f={"REPEAT_HOOK": bump})
        self.failed(rc, run, "determinism")

    def test_views_not_regenerated_identically(self):
        class V(F.Views):
            def regenerated_first_batches(self, tf_):
                return {**super().regenerated_first_batches(tf_), 101: False}
        rc, out, err, run = self.e.run(stop_cfg(), f={"Views": V})
        st = self.failed(rc, run, "determinism")
        self.assertFalse(st["failed_check"]["values"]["views_regenerated_identical"])

    def test_duplicates(self):
        orig = F.reestimate_cell

        def bumped(*a, **k):
            out = orig(*a, **k)
            if a[9].endswith("explicit"):
                n = next(iter(out))
                out[n] = {**out[n], "mean32": out[n]["mean32"] + np.float32(1.0)}
            return out
        rc, out, err, run = self.e.run(stop_cfg(duplicate="all"), f={"reestimate_cell": bumped})
        st = self.failed(rc, run, "duplicates")
        self.assertEqual(st["failed_check"]["values"]["cell"], "E01")

    def test_ema_facts(self):
        with self.e.altered_card(LN1, lambda c: c.update(n_fitting=31)):
            rc, out, err, run = self.e.run(stop_cfg())
        self.failed(rc, run, "ema")

    def test_model_or_checkpoint_changed_at_the_end(self):
        rc, out, err, run = self.e.run(stop_cfg(), f={"trainable_sha256": nth_call(F.trainable_sha256, 2, lambda m: "0" * 64)})
        st = self.failed(rc, run, "weights")
        self.assertEqual(st["failed_check"]["values"]["part"], "model")
        real = F.sha256_file
        calls = {"n": 0}

        def late(p):
            if str(p).endswith(".keras"):
                calls["n"] += 1
                if calls["n"] == 2:
                    return "0" * 64
            return real(p)
        rc, out, err, run = self.e.run(stop_cfg(), f={"sha256_file": late})
        st = self.failed(rc, run, "weights")
        self.assertEqual(st["failed_check"]["values"]["part"], "checkpoint file")

    def test_one_class_dev(self):
        real_frames, real_saved = R.unit_frames, F.saved_scores

        def one_class(store, inp, role, fold):
            part = real_frames(store, inp, role, fold)
            return {**part, "y": np.ones_like(part["y"])} if role == "dev" else part
        rc, out, err, run = self.e.run(stop_cfg(), r={"unit_frames": one_class},
                                       f={"saved_scores": lambda rows, dev, what: (real_saved(rows, dev, what)[0], None)})
        self.assertEqual(rc, 0, (out + err)[-3000:])
        cells = read_csv(run / "public" / "cells.csv")
        self.assertEqual({(r["bpcer_05"], r["acer_05"], r["eer_threshold"]) for r in cells}, {("NA", "NA", "NA")})
        self.assertTrue(all(r["apcer_05"] != "NA" for r in cells))

    def test_cap_incomplete_keeps_complete_units(self):
        real = SR.Budget.check

        def check(self_, step):
            if "p4_LivenessNet_fold2_seed43" in step:
                raise SR.CapExceeded(step, 61.0)
            return real(self_, step)
        with injected(sr={"Budget": type("B", (SR.Budget,), {"check": check})}):
            rc, out, err, run = self.e.run(stop_cfg(units=[LN1, LN2]))
        self.assertEqual(rc, 2, (out + err)[-2000:])
        st = json.loads((run / "status.json").read_text())
        self.assertEqual((st["status"], st["complete_units"]), ("INCOMPLETE", ["p4_LivenessNet_fold1_seed42"]))
        self.assertFalse(any("fold2" in p.name for p in (run / "restricted").iterdir()))
        self.assertEqual(stale_left(), [])

    def real_signal(self, sig, at):
        """Un segnale vero, inviato al processo durante il controllo di un passo."""
        real = SR.Budget.check
        sent = {"done": False}

        def check(self_, step):
            if at in step and not sent["done"]:
                sent["done"] = True
                os.kill(os.getpid(), sig)
            return real(self_, step)
        saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1)}
        for s in saved:
            signal.signal(s, signal.SIG_DFL if s != signal.SIGUSR1 else signal.SIG_IGN)    # SIGHUP non ignorato (anche sotto nohup)
        try:
            with injected(sr={"Budget": type("B", (SR.Budget,), {"check": check})}):
                return self.e.run(stop_cfg())
        finally:
            for s, h in saved.items():
                signal.signal(s, h)

    def test_real_signals(self):
        rc, out, err, run = self.real_signal(signal.SIGUSR1, "E10")
        self.assertEqual(rc, 2, (out + err)[-2000:])
        self.assertEqual(json.loads((run / "status.json").read_text())["stopped"]["cause"], "SIGUSR1")
        self.assertEqual(stale_left(), [])
        for sig, name in ((signal.SIGTERM, "SIGTERM"), (signal.SIGHUP, "SIGHUP")):
            rc, out, err, run = self.real_signal(sig, "E01")
            st = self.failed(rc, run)
            self.assertIn(name, st["reason"])

    def test_read_guard_during_the_trial(self):
        F.GUARD["active"] = False
        forbidden = self.e.base / "results" / "archeval"
        forbidden.mkdir(parents=True, exist_ok=True)
        (forbidden / "x.txt").write_text("x")
        other = self.e.base / "results" / "other"
        other.mkdir(parents=True, exist_ok=True)
        (other / "y.txt").write_text("y")
        suite_guard()
        for target, why in ((forbidden / "x.txt", "forbidden folder"), (other / "y.txt", "undeclared file under results")):
            def peek(a, target=target):
                open(target).close()
                return a
            rc, out, err, run = self.e.run(stop_cfg(), f={"VIEW_HOOK": peek})
            st = self.failed(rc, run)
            self.assertIn(why, st["reason"])                                   # results/archeval della radice degli ingressi
            self.assertNotIn(str(self.e.base), json.dumps(st))

    def test_refusals_before_any_folder(self):
        out = Path(tempfile.mkdtemp(prefix="runs-", dir=self.e.tmp))
        rc, o, err, run = self.e.run(stop_cfg(note=False), out=out)
        self.assertEqual((rc, run), (1, None))
        self.assertIn("analysis.sha256 is null", err)
        rc, o, err, run = self.e.run(stop_cfg(), out=out, f={"REPO_STATE": lambda root: {"commit": "abcdef0", "dirty": True}})
        self.assertEqual((rc, run), (1, None))
        self.assertIn("not clean", err)
        rc, o, err, run = self.e.run(stop_cfg(), out=out, attempt=2)
        self.assertEqual((rc, run), (1, None))
        with self.assertRaises(F.Failure):
            R.unit_frames(None, types.SimpleNamespace(assembled=[], folds={}), "test", 1)     # il ruolo test è rifiutato
        os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
        try:
            rc, o, err, run = self.e.run(stop_cfg(), out=out)
        finally:
            del os.environ["TF_NUM_INTRAOP_THREADS"]
        self.assertEqual((rc, run), (1, None))
        self.assertIn("TF_NUM_INTRAOP_THREADS", err)
        rc, o, err, run = self.e.run(stop_cfg(), out=out)                        # prima prova di questo ambito
        self.assertEqual(rc, 0, err[-2000:])
        self.assertRegex(run.name, r"^20261123-ARCHBN2X2-custom-abcdef0$")
        rc, o, err, run2 = self.e.run(stop_cfg(), out=out)                       # stesso ambito: serve --attempt 2
        self.assertEqual(rc, 1)
        self.assertIn("--attempt 2", err)

    def test_temporary_folder_checked_at_the_start(self):
        """TMPDIR che contiene la radice degli ingressi o la cartella home, o che sta dentro la radice degli ingressi:
        rifiutato all'avvio, prima di ogni cartella (anche check-inputs)."""
        sub = self.e.base / "tmp_inside"
        sub.mkdir(exist_ok=True)
        for tmp, needle in ((self.e.tmp, "contains the input root"), (sub, "inside the input root"), (Path.home(), "contains")):
            for cmd in ("trial", "check-inputs"):
                out = Path(tempfile.mkdtemp(prefix="runs-", dir=self.e.tmp))
                with mock_attr(FR, "tf_state", lambda: "not imported (test)"):
                    rc, o, err, run = self.e.run(stop_cfg(), cmd=cmd, out=out, tmpdir=tmp)
                self.assertEqual((rc, run), (1, None), (tmp, cmd))
                self.assertIn(needle, err)

    def test_temporary_folder_beside_the_repository(self):
        """Un TMPDIR dentro la cartella che contiene il repository è rifiutato all'avvio; la cartella temporanea predefinita
        di macOS no."""
        out = Path(tempfile.mkdtemp(prefix="runs-", dir=self.e.tmp))
        rc, o, err, run = self.e.run(stop_cfg(), out=out, f={"REPO_PARENT": self.e.tmp})        # vt sta dentro questa cartella
        self.assertEqual((rc, run), (1, None))
        self.assertIn("the folder that contains the repository", err)
        default = os.confstr("CS_DARWIN_USER_TEMP_DIR") if "CS_DARWIN_USER_TEMP_DIR" in os.confstr_names else "/var/folders/x/T"
        saved = tempfile.tempdir
        tempfile.tempdir = default
        try:
            self.assertEqual(F.outside_problem(ROOT), "")
        finally:
            tempfile.tempdir = saved

    def test_attempts_counted_per_scope(self):
        out = Path(tempfile.mkdtemp(prefix="runs-", dir=self.e.tmp))
        (out / "20261120-ARCHBN2X2-pilot-1234567").mkdir()                       # un'altra prova, di un altro ambito
        rc, o, err, run = self.e.run(stop_cfg(), out=out)
        self.assertEqual(rc, 0, err[-2000:])
        self.assertTrue((out / "20261123-ARCHBN2X2-custom-abcdef0").is_dir())

    def test_swallowed_refusal_stops_at_the_next_step(self):
        F.GUARD["active"] = False
        forbidden = self.e.base / "results" / "archeval"
        forbidden.mkdir(parents=True, exist_ok=True)
        (forbidden / "w.txt").write_text("x")
        suite_guard()

        def swallow(a):
            try:
                open(forbidden / "w.txt").close()
            except Exception:                                                   # una libreria che ingoia l'eccezione
                pass
            return a
        rc, out, err, run = self.e.run(stop_cfg(), f={"VIEW_HOOK": swallow})
        st = self.failed(rc, run, "read_guard")
        self.assertGreaterEqual(st["failed_check"]["values"]["refused_opens"], 1)

    def test_swallowed_refusal_after_the_last_step(self):
        F.GUARD["active"] = False
        forbidden = self.e.base / "results" / "archeval"
        forbidden.mkdir(parents=True, exist_ok=True)
        (forbidden / "w.txt").write_text("x")
        suite_guard()
        real = F.restricted_files

        def late(rec, dev, parent):                                            # dopo l'ultimo passo dell'ultima unità
            try:
                open(forbidden / "w.txt").close()
            except Exception:
                pass
            return real(rec, dev, parent)
        rc, out, err, run = self.e.run(stop_cfg(), f={"restricted_files": late})
        st = self.failed(rc, run)
        self.assertIn("refused opens during the trial", st["reason"])

    def test_cap_during_the_views(self):
        real = SR.Budget.check

        def check(self_, step):
            if step.startswith("views 101 batch"):
                raise SR.CapExceeded(step, 61.0)
            return real(self_, step)
        with injected(sr={"Budget": type("B", (SR.Budget,), {"check": check})}):
            rc, out, err, run = self.e.run(stop_cfg())
        self.assertEqual(rc, 2, (out + err)[-2000:])
        st = json.loads((run / "status.json").read_text())
        self.assertEqual((st["status"], st["stopped"]["step"], st["complete_units"]), ("INCOMPLETE", "views 101 batch 1", []))
        self.assertEqual(stale_left(), [])

    def test_frozen_note_changed(self):
        cfg = stop_cfg()
        b = self.e.note.read_bytes()
        try:
            self.e.note.write_bytes(b + b"changed after the configuration\n")
            rc, out, err, run = self.e.run(cfg)
        finally:
            self.e.note.write_bytes(b)
        self.assertEqual((rc, run), (1, None))
        self.assertIn("frozen note", err)

    def test_keras_module_pin(self):
        def mut(d):
            d["keras_functions"]["sha256"] = "0" * 64
        rc, out, err, run = self.e.run(stop_cfg(mutate=mut))
        self.assertEqual((rc, run), (1, None))
        self.assertIn("another Keras version", err)

    def test_parent_status_pin(self):
        cfg = stop_cfg()
        p = self.e.run_dir("nuaa_patience_4") / "status.json"
        b = p.read_bytes()
        try:
            p.write_text(json.dumps(json.loads(b), indent=2) + "\n", encoding="utf-8")          # stessi campi, altri byte
            with mock_attr(FR, "tf_state", lambda: "not imported (test)"):
                rc, out, err, run = self.e.run(cfg, cmd="check-inputs")
        finally:
            p.write_bytes(b)
        self.assertEqual(rc, 1)
        self.assertIn("differs from the pinned SHA-256", err)

    def test_configuration_changed_during_the_trial(self):
        cfg = stop_cfg()
        done = {"n": 0}

        def touch(a):
            if not done["n"]:
                done["n"] = 1
                cfg.write_text(cfg.read_text() + " ", encoding="utf-8")
            return a
        rc, out, err, run = self.e.run(cfg, f={"VIEW_HOOK": touch})
        st = self.failed(rc, run)
        self.assertIn("changed during the trial", st["reason"])

    def test_authorisation_flag_required(self):
        cfg = stop_cfg()
        out = Path(tempfile.mkdtemp(prefix="runs-", dir=self.e.tmp))
        argv = ["trial", "--config", str(cfg), "--input-root", str(self.e.base), "--out-root", str(out), "--unofficial-out-root", "--date", DATE]
        rc, o, err, run = self.e.run(cfg, argv=argv, out=out)
        self.assertEqual((rc, run), (1, None))
        self.assertIn("--i-am-authorized", err)

    def test_saved_dev_rows_of_the_test_role(self):
        def to_test(text):
            lines = text.splitlines(keepends=True)
            head = lines[0].rstrip("\r\n").split(",")
            row = lines[1].split(",")
            row[head.index("role")] = "test"
            return "".join([lines[0], ",".join(row)] + lines[2:])
        cfg = stop_cfg()
        with self.e.altered_file(LN1, "_dev_ordinary.csv", to_test), mock_attr(FR, "tf_state", lambda: "not imported (test)"):
            rc, out, err, run = self.e.run(stop_cfg(), cmd="check-inputs")
        self.assertEqual(rc, 1)
        self.assertIn("rows of another unit, state or role", err)
        del cfg

    def test_saved_dev_key_order(self):
        def swap(text):
            lines = text.splitlines(keepends=True)
            return "".join([lines[0], lines[2], lines[1]] + lines[3:])
        with self.e.altered_file(LN1, "_dev_ordinary.csv", swap):
            rc, out, err, run = self.e.run(stop_cfg())
        st = self.failed(rc, run)
        self.assertIn("key order", st["reason"])

    def test_required_name_patterns(self):
        self.assertEqual(F.REQUIRED_PATTERNS, ("fold[0-9]*_test.csv", "*_test_scores.csv", "*scores_test*", "*test*_scores*"))

    def test_official_root_refuses_outside_inputs(self):
        """Con la radice ufficiale, --input-root e una configurazione fuori dal repository sono rifiutati prima di ogni
        scrittura (nulla scritto in results/archbn2x2: lo verifica il controllo finale della suite)."""
        cfg = stop_cfg()
        official = ROOT / F.DEFAULT_OUT_ROOT
        argv = ["trial", "--config", str(cfg), "--input-root", str(self.e.base), "--out-root", str(official), "--i-am-authorized"]
        rc, o, err, run = self.e.run(cfg, argv=argv, out=self.e.tmp / "never")
        self.assertEqual(rc, 1)
        self.assertIn("--input-root is refused", err)
        argv = ["trial", "--config", str(cfg), "--out-root", str(official), "--i-am-authorized"]
        rc, o, err, run = self.e.run(cfg, argv=argv, out=self.e.tmp / "never")
        self.assertEqual(rc, 1)
        self.assertIn("configuration inside the repository", err)

    def test_check_inputs(self):
        # TensorFlow è già importato in questo processo: FR.tf_state (che lo rifiuta) è sostituita solo qui
        self.enterContext(mock_attr(FR, "tf_state", lambda: "not imported (test)"))
        rc, out, err, run = self.e.run(stop_cfg(), cmd="check-inputs")
        self.assertEqual(rc, 0, err[-2000:])
        self.assertIn("RESULT: PASS (check-inputs)", out)
        self.e.build_parents()
        p = self.e.run_dir("nuaa_patience_4") / "restricted" / "LivenessNet_fold1_seed42_card.json"
        b = p.read_bytes()
        try:
            p.write_bytes(b.replace(b'"restored_epoch"', b'"restored_epoch_"'))
            rc, out, err, run = self.e.run(stop_cfg(), cmd="check-inputs")
            self.assertEqual(rc, 1)
            self.assertIn("SHA-256 differs", err)
        finally:
            p.write_bytes(b)
        for mutate, needle in ((lambda c: c.update(fitting_keys_sha256="0" * 64), "keys of the manifests differ"),
                               (lambda c: c.update(network="ResNet-18"), "network, fold, seed, patience"),
                               (lambda c: c.update(early_stopping_patience=12), "network, fold, seed, patience")):
            with self.e.altered_card(LN1, mutate):
                rc, out, err, run = self.e.run(stop_cfg(), cmd="check-inputs")
            self.assertEqual(rc, 1, needle)
            self.assertIn(needle, err)


LAUNCH = "delivery/archbn2x2/launch.zsh"
FAKE_COMMAND = """import json, os, sys
with open(os.environ["FAKE_LOG"], "a") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\\n")
print("fake", sys.argv[1])
if sys.argv[1] == "trial" and not os.environ.get("FAKE_NO_START"):
    print("== 2x2 trial 20260101-ARCHBN2X2-pilot-0000000: fake start line", flush=True)
sys.exit(int(os.environ.get("FAKE_RC_" + sys.argv[1].replace("-", "_"), "0")))
"""


class TestHLaunch(unittest.TestCase):
    """delivery/archbn2x2/launch.zsh eseguito davvero in un repository finto, con pad_arch_bn_factorial.py finto (registra i
    propri argomenti; trial stampa la riga d'avvio «== 2x2 trial », se FAKE_NO_START non è impostata) e pmset e ps finti in
    testa al PATH: i test non dipendono dall'alimentazione né dal nice di chi lancia la suite. Il repository finto è un
    repository git appena creato (nessun commit) con i propri file esclusi in .git/info/exclude, quindi pulito; git gira
    senza configurazione globale o di sistema e senza risalire oltre la cartella temporanea del test."""

    def setUp(self):
        if any(shutil.which(t) is None for t in ("zsh", "caffeinate", "pmset", "sw_vers", "top", "memory_pressure", "git")):
            skip_missing(self, "macOS tools")
        self.tmp = Path(tempfile.mkdtemp(prefix="archbn2x2-launch-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # il repository finto sta in una sottocartella («outer»), così un test può farne il contenitore di un altro repository
        self.repo = self.tmp / "outer" / "repo"
        for d in ("scripts", "delivery/archbn2x2", ".venv/bin"):
            (self.repo / d).mkdir(parents=True)
        shutil.copy2(HERE.parents[1] / LAUNCH, self.repo / LAUNCH)
        (self.repo / "scripts" / "pad_arch_bn_factorial.py").write_text(FAKE_COMMAND, encoding="utf-8")
        os.symlink(sys.executable, self.repo / ".venv" / "bin" / "python")
        self.logs, self.calls, self.ttmp = self.tmp / "logs", self.tmp / "calls.jsonl", self.tmp / "t"
        self.ttmp.mkdir()
        subprocess.run(["git", "-c", "init.defaultBranch=main", "init", "-q", str(self.repo)], env=self.git_env(), check=True,
                       capture_output=True, timeout=60)
        (self.repo / ".git" / "info").mkdir(exist_ok=True)
        (self.repo / ".git" / "info" / "exclude").write_text("/scripts/\n/delivery/\n/.venv/\n", encoding="utf-8")

    def git_env(self) -> dict:
        """L'ambiente senza variabili GIT_ e TF_NUM_ del chiamante, con git senza configurazione globale e di sistema e senza
        ricerca del repository sopra la cartella temporanea del test."""
        env = {k: v for k, v in os.environ.items() if not k.startswith(("TF_NUM_", "ARCHBN2X2_", "GIT_"))}
        env.update({"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CEILING_DIRECTORIES": str(self.tmp.resolve())})
        return env

    def launch(self, *args, attempt="1", nice=0, battery=False, powermode=0, lowpower=None, log_dir=None, start=True, **rc):
        env = self.git_env()
        if not start:
            env["FAKE_NO_START"] = "1"
        fake = self.tmp / "fakebin"
        fake.mkdir(exist_ok=True)
        source = "Battery Power" if battery else "AC Power"
        # come la macchina vera: pmset -g ha la voce powermode; lowpowermode solo se chiesta (su questo Mac non compare)
        extra = "" if lowpower is None else f"; echo ' lowpowermode {lowpower}'"
        (fake / "pmset").write_text(f"#!/bin/sh\nif [ \"$2\" = batt ]; then echo \"Now drawing from '{source}'\"; "
                                    f"else echo ' powermode {powermode}'{extra}; fi\n", encoding="utf-8")
        (fake / "ps").write_text(f"#!/bin/sh\necho '  {nice}'\n", encoding="utf-8")
        for tool in ("pmset", "ps"):
            (fake / tool).chmod(0o755)
        env["PATH"] = f"{fake}{os.pathsep}{env.get('PATH', '')}"
        env["ZDOTDIR"] = str(fake)                                           # nessun file d'avvio di zsh dell'utente
        env["TMPDIR"] = str(self.ttmp)
        env.update({"ARCHBN2X2_LOG_DIR": str(log_dir or self.logs), "FAKE_LOG": str(self.calls), **({"ARCHBN2X2_ATTEMPT": attempt} if attempt else {})})
        env.update({f"FAKE_RC_{k}": str(v) for k, v in rc.items()})
        r = subprocess.run(["zsh", LAUNCH, *args], cwd=self.repo, env=env, capture_output=True, text=True, timeout=120)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []
        return r, calls

    def conditions(self) -> str:
        return (self.logs / "trial-attempt1-conditions.txt").read_text()

    def test_trial_launched_after_check_inputs_with_every_condition(self):
        r, calls = self.launch("trial", trial=4)
        self.assertEqual(r.returncode, 4, r.stderr)
        self.assertEqual(calls, [["check-inputs"], ["trial", "--attempt", "1", "--i-am-authorized"]])
        c = self.conditions()
        for part in ("== conditions before", "== system", "== power", "'AC Power'", "== nice of the launching shell\n0\n",
                     "== environment", "== temporary folder", "== git", "check-inputs exit 0", "== command", "trial exit 4", "== conditions after",
                     "== views folders left in TMPDIR\n0"):
            self.assertIn(part, c)
        before, after = c.split("== conditions after")
        self.assertIn("\npowermode 0\n", before.split("== system")[0])        # powermode nelle condizioni prima e dopo
        self.assertIn("\npowermode 0\n", after)
        self.assertNotRegex(c, r"(?m)^ ?lowpowermode ")                     # la voce che la macchina vera non ha
        self.assertEqual(list(self.logs.glob("*not-launched*")), [])
        self.assertEqual(PAC.output_problems(c, FR.Inputs()), [])
        self.assertNotIn(str(self.tmp), c)
        self.assertTrue((self.logs / "trial-attempt1.console.log").is_file())
        self.assertNotIn("not started", c)

    def refused(self, needle, **kw):
        """Un mancato avvio: il motivo sta nell'unico file «not launched», e il file del tentativo non esiste."""
        r, calls = self.launch("trial", **kw)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertNotIn(["trial", "--attempt", "1", "--i-am-authorized"], calls)
        self.assertFalse((self.logs / "trial-attempt1-conditions.txt").exists())
        files = list(self.logs.glob("trial-attempt1-not-launched-*-conditions.txt"))
        self.assertEqual(len(files), 1, files)
        text = files[0].read_text()
        self.assertIn("== not launched", text)
        self.assertIn(needle, text)
        return calls

    def test_nice_not_zero(self):
        self.assertEqual(self.refused("nice value is not 0", nice=5), [])

    def test_battery(self):
        self.assertEqual(self.refused("power adapter", battery=True), [])
        shutil.rmtree(self.logs)
        r, calls = self.launch("check-inputs", battery=True)                 # check-inputs a batteria sì
        self.assertEqual((r.returncode, calls), (0, [["check-inputs"]]))

    def test_low_power_mode(self):
        self.assertEqual(self.refused("low power mode", lowpower=1), [])

    def test_powermode_not_zero(self):
        self.assertEqual(self.refused("powermode is not 0", powermode=1), [])

    def test_refusal_then_launch_with_the_same_attempt(self):
        self.assertEqual(self.refused("nice value is not 0", nice=5), [])
        r, calls = self.launch("trial", trial=4)                            # stesso tentativo, ora con nice 0
        self.assertEqual(r.returncode, 4, r.stderr)
        self.assertEqual(calls, [["check-inputs"], ["trial", "--attempt", "1", "--i-am-authorized"]])
        self.assertIn("trial exit 4", self.conditions())

    def test_views_folder_left(self):
        (self.ttmp / "archbn2x2-views-x").mkdir()
        self.assertEqual(self.refused("views folder of an earlier trial"), [])

    def test_failed_check_inputs_stops_the_trial(self):
        self.assertEqual(self.refused("check-inputs failed", check_inputs=1), [["check-inputs"]])

    def test_dirty_work_tree(self):
        (self.repo / "notes.txt").write_text("x\n", encoding="utf-8")       # un file non tracciato
        self.assertEqual(self.refused("== not launched: the work tree is not clean"), [])
        text = next(self.logs.glob("trial-attempt1-not-launched-*-conditions.txt")).read_text()
        self.assertIn("== lines of git status --porcelain\n1\n", text)
        self.assertNotIn("notes.txt", text)

    def test_git_status_fails(self):
        # un indice di 13 byte: git status --porcelain esce con 128 («index file smaller than expected») senza righe,
        # mentre git rev-parse --show-toplevel (che non legge l'indice) dà ancora la radice
        (self.repo / ".git" / "index").write_bytes(b"not an index\n")
        self.assertEqual(self.refused("== not launched: git status failed"), [])
        text = next(self.logs.glob("trial-attempt1-not-launched-*-conditions.txt")).read_text()
        self.assertIn("== git status --porcelain\nfailed\n", text)

    def test_not_the_root_of_a_git_work_tree(self):
        shutil.rmtree(self.repo / ".git")                                  # nessun repository: git non risale oltre il test
        self.assertEqual(self.refused("the folder is not the root of a git work tree"), [])

    def test_inside_a_clean_git_work_tree_but_not_its_root(self):
        """Il repository finto senza .git proprio, sottocartella di un repository git pulito: git status funziona e non ha
        righe, quindi solo il controllo della radice ferma il lancio."""
        shutil.rmtree(self.repo / ".git")
        outer = self.repo.parent
        subprocess.run(["git", "-c", "init.defaultBranch=main", "init", "-q", str(outer)], env=self.git_env(), check=True,
                       capture_output=True, timeout=60)
        (outer / ".git" / "info").mkdir(exist_ok=True)
        (outer / ".git" / "info" / "exclude").write_text("/repo/\n", encoding="utf-8")
        st = subprocess.run(["git", "status", "--porcelain"], cwd=self.repo, env=self.git_env(), capture_output=True, text=True, timeout=60)
        self.assertEqual((st.returncode, st.stdout), (0, ""))              # premessa: albero pulito visto dalla sottocartella
        self.assertEqual(self.refused("== not launched: the folder is not the root of a git work tree"), [])

    def test_trial_refused_before_its_run_folder(self):
        """trial esce con 1 senza la riga d'avvio (come per un rifiuto prima della cartella del run): l'uscita è propagata,
        condizioni e console vanno sui nomi «not-launched» e lo stesso tentativo si rilancia."""
        r, calls = self.launch("trial", start=False, trial=1)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertEqual(calls, [["check-inputs"], ["trial", "--attempt", "1", "--i-am-authorized"]])
        self.assertIn("not started", r.stderr)
        self.assertFalse((self.logs / "trial-attempt1-conditions.txt").exists())
        self.assertFalse((self.logs / "trial-attempt1.console.log").exists())
        files = list(self.logs.glob("trial-attempt1-not-launched-*-conditions.txt"))
        self.assertEqual(len(files), 1, files)
        text = files[0].read_text()
        self.assertIn("check-inputs exit 0", text)
        self.assertIn("trial exit 1\n== conditions after\n", text)                # condizioni dopo anche qui
        self.assertTrue(text.endswith("\n== not started: the trial refused before creating its run folder\n"), text[-200:])
        consoles = list(self.logs.glob("trial-attempt1-not-launched-*.console.log"))
        self.assertEqual([p.name for p in consoles], [files[0].name.replace("-conditions.txt", ".console.log")])
        self.assertIn("fake trial", consoles[0].read_text())
        r, calls = self.launch("trial", trial=4)                            # stesso tentativo: parte
        self.assertEqual(r.returncode, 4, r.stderr)
        self.assertEqual(calls[2:], [["check-inputs"], ["trial", "--attempt", "1", "--i-am-authorized"]])
        self.assertIn("trial exit 4", self.conditions())
        self.assertTrue((self.logs / "trial-attempt1.console.log").is_file())

    def test_started_trial_that_fails_uses_the_attempt(self):
        """trial esce con 1 dopo la riga d'avvio: la prova è partita, il file del tentativo resta e il rilancio è rifiutato."""
        r, calls = self.launch("trial", trial=1)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertEqual(list(self.logs.glob("*not-launched*")), [])
        before = self.conditions()
        self.assertIn("trial exit 1\n== conditions after", before)
        self.assertNotIn("not started", before)
        self.assertTrue((self.logs / "trial-attempt1.console.log").is_file())
        r, calls2 = self.launch("trial")                                    # stesso tentativo: rifiutato
        self.assertEqual((r.returncode, calls2), (9, calls))
        self.assertIn("already exists", r.stderr)
        self.assertEqual(self.conditions(), before)

    def test_refusals_of_the_arguments(self):
        r, calls = self.launch("trial", log_dir=self.repo / "logs")
        self.assertEqual((r.returncode, calls), (9, []))
        self.assertIn("outside the repository", r.stderr)
        r, calls = self.launch("trial", attempt="")
        self.assertEqual((r.returncode, calls), (9, []))
        r, calls = self.launch("trial")
        self.assertEqual(r.returncode, 0)
        before = self.conditions()
        r, calls = self.launch("trial")                                     # il file delle condizioni non si sovrascrive
        self.assertEqual(r.returncode, 9)
        self.assertEqual(self.conditions(), before)


class ZYSuiteOutput(unittest.TestCase):
    def test_no_warning_or_personal_path_in_the_output(self):
        """Una parte della suite (strutture, augmentation con Keras e guardia) eseguita in un processo nuovo: la sua uscita non
        contiene avvisi né percorsi personali o del repository."""
        need_arch(self)
        tmp = Path(tempfile.mkdtemp(prefix="archbn2x2-output-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        env = {**os.environ, "TMPDIR": str(tmp)}
        # TestFDropoutStructure prima: set_random_seed seguito dagli strati di Keras è ciò che fa partire l'avviso di randrange
        r = subprocess.run([sys.executable, "-I", "-B", str(HERE), "TestFDropoutStructure", "TestEAugmentation", "TestDGuardAndComparisons"],
                           cwd=str(tmp), env=env,
                           capture_output=True, text=True, timeout=600)
        text = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, text[-2000:])
        self.assertNotIn("Warning", text)
        for t in (str(Path.home()), str(ROOT), "/Users/", "/home/"):
            self.assertNotIn(t, text)


class ZZRepositoryUntouched(unittest.TestCase):
    def test_declared_skips_only(self):
        self.assertEqual(sorted(MISSING_INPUT), ["macOS tools", "reference architecture", "tensorflow"])


def snapshot() -> dict:
    """Le uscite che un test potrebbe toccare per errore: il primo livello di results/, data/ ed export/ (solo nomi, nessuna
    cartella di dati tenuti da parte aperta) e le cartelle di questo comando, ricorsivamente."""
    out = {}
    for d in ("results", "data", "export"):
        p = ROOT / d
        out[d] = sorted(os.listdir(p)) if p.is_dir() else None
    for d in ("results/archbn2x2", "delivery/archbn2x2"):
        p = ROOT / d
        out[d] = sorted((str(q), q.stat().st_size, q.stat().st_mtime_ns) for q in p.rglob("*")) if p.is_dir() else None
    return out


class QuietRunner(unittest.TextTestRunner):
    """Il runner di unittest con gli avvisi «default» come sempre, più il solo filtro dell'avviso di Keras su randrange
    (TextTestRunner rimette i filtri a «default» e cancellerebbe quello impostato in testa al file)."""

    def run(self, test):
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            warnings.filterwarnings("ignore", message=RANDRANGE_WARNING, category=DeprecationWarning)
            self.warnings = None
            return super().run(test)


def main() -> int:
    tmp_outside_repo()
    before = snapshot()
    suite_guard()                       # tutta la suite gira con la guardia su results/archeval del repository
    argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a != "--verbose"] + (["-v"] if "--verbose" in sys.argv else [])
    try:
        result = unittest.main(argv=argv, exit=False, verbosity=1, testRunner=QuietRunner).result
    finally:
        if ENV is not None:
            ENV.close()
    after = snapshot()
    changed = before != after
    skipped = [f"{t.id()}: {r}" for t, r in result.skipped]
    print(f"tests run: {result.testsRun}, failures: {len(result.failures)}, errors: {len(result.errors)}, skipped: {len(skipped)}")
    for s in skipped:
        print(f"skipped {s}")
    if changed:
        print("ERROR: the repository outputs changed during the tests")
    ok = result.wasSuccessful() and not changed
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
