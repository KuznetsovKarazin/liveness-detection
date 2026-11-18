"""
Test di scripts/pad_arch_comparison.py (confronto tra LivenessNet e ResNet-18: configurazione, prova tecnica, piano),
scripts/pad_arch_definitions.py (definizioni operative) e scripts/pad_arch_models.py (registro delle reti), offline.

Mondo sintetico in cartelle temporanee, costruito sul mondo della suite del run di ablazione
(check_nuaa_session_ablation_run.World: manifest M2, cache 8×8, run delle celle, run dei fold, frame estratti, la
configurazione a patience 4 del mondo) più: la configurazione emendata a patience 12, i due run «accettati» finti
(status.json, SHA256SUMS riservato, la scheda del modello B fold 1 seed 42) e la configurazione di questo runner. Le
definizioni sono provate su dati sintetici costruiti a mano (perfetto, costante, quasi costante, pareggi esatti, classe
assente, un solo soggetto, meno di 5 soggetti, intervallo che tocca lo zero, valori fuori dominio) e contro un bootstrap
di riferimento scritto qui in modo indipendente. Il trainer è confrontato con il sorgente di nuaa_folds_run.default_trainer
(solo le sostituzioni dichiarate) e, con TensorFlow, nel comportamento (stessa storia). La prova tecnica gira davvero, su
CPU, con LivenessNet del codice di riferimento e ResNet-18 su frame 8×8 (mai 256×256 in addestramento; ResNet-18 è
costruita a 256×256 solo per contare i parametri). Nessuna scrittura nel repository (controllo finale su results/,
export/, delivery/).

Senza TensorFlow i test che costruiscono o addestrano un modello Keras sono saltati, con un motivo dichiarato in
MISSING_INPUT; gli altri girano.

  .venv/bin/python -I -B scripts/check_pad_arch_comparison.py [--verbose] [-f]
"""
import contextlib
import hashlib
import importlib.util
import inspect
import io
import json
import os
import shutil
import sys
import types
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_nuaa_folds_run as CFR  # noqa: E402
import check_nuaa_session_ablation_run as SAR  # noqa: E402
import nuaa_bn_recalibration as BR  # noqa: E402
import nuaa_folds_run as FR  # noqa: E402
import nuaa_session_ablation_run as SR  # noqa: E402
import pad_arch_comparison as P  # noqa: E402
import pad_arch_definitions as D  # noqa: E402
import pad_arch_models as AM  # noqa: E402

TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
REF_ARCH = ROOT / "src" / "livedetection" / "src" / "architectures.py"
MISSING_INPUT = {
    "tensorflow": "TensorFlow is not installed: the tests that build or train a Keras model are skipped",
    "reference architecture": "src/livedetection/src/architectures.py (LivenessNet of the reference code) is not present",
    "delivered configuration": "delivery/archcmp/pad_arch_comparison.json or the accepted configurations it pins are not present",
}
STATE = {"commit": "abcdef0", "dirty": False}
DATE = "20261118"
SHAPE = SAR.SHAPE
P4, P12 = "20261106-SESSABLRUN-abcdef0", "20261109-SESSABLRUN-abcdef1"
REF_MODEL = "arm_b_fold1_seed42"
RUN = f"{DATE}-ARCHTRIAL-{STATE['commit']}"


def skip_missing(test: unittest.TestCase, what: str):
    """Unico modo di saltare un test di questa suite: un ingresso mancante dichiarato in MISSING_INPUT."""
    test.skipTest(f"missing input: {MISSING_INPUT[what]}")


def need_tf(test: unittest.TestCase):
    if not TF_AVAILABLE:
        skip_missing(test, "tensorflow")
    if not REF_ARCH.is_file():
        skip_missing(test, "reference architecture")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_sums(folder: Path) -> str:
    return SAR.write_sums(folder)


@contextlib.contextmanager
def injected(p=None, sr=None, fr=None):
    """Punti di iniezione di pad_arch_comparison (p), nuaa_session_ablation_run (sr) e nuaa_folds_run (fr)."""
    saved = [(m, k, getattr(m, k)) for m, d in ((P, p), (SR, sr), (FR, fr)) for k in (d or {})]
    try:
        for m, d in ((P, p), (SR, sr), (FR, fr)):
            for k, v in (d or {}).items():
                setattr(m, k, v)
        yield
    finally:
        for m, k, v in saved:
            setattr(m, k, v)


def run_main(argv) -> tuple:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = P.main(argv)
    return rc, out.getvalue(), err.getvalue()


def ref_arch():
    return FR.load_module("kuznetsov_architectures_test", REF_ARCH)


# ============================================================================ mondo sintetico

class World(SAR.World):
    """Il mondo del run di ablazione più la configurazione a patience 12, i due run accettati finti e la configurazione."""

    def build(self):
        super().build()
        self.config12 = self.base / "config12.json"
        self.write_config(SAR.amended_doc(self.sr_doc(), sha(self.config.read_bytes())), self.config12)
        self.history = {"loss": [0.7, 0.6, 0.5, 0.4], "accuracy": [0.5, 0.6, 0.7, 0.8], "val_loss": [0.71, 0.69, 0.65, 0.6],
                        "val_accuracy": [0.5, 0.5, 0.75, 0.8], "lr": [0.0010000000474974513] * 4}
        self.cfg_path = self.base / "archcmp.json"
        self.write_parents()

    def write_parents(self, history=None, card_over=None):
        """I due run accettati finti (status, SHA256SUMS riservato; la scheda del modello di riferimento nel primo) e la
        configurazione di questo runner coerente con essi."""
        if history is not None:
            self.history = history
        names = [SR.model_name(*x) for x in SR.training_order(SAR.K, SAR.SEEDS)]
        self.sums = {}
        for key, rid, cfgp in (("patience_4", P4, self.config), ("patience_12", P12, self.config12)):
            run = self.base / key
            shutil.rmtree(run, ignore_errors=True)
            (run / "restricted").mkdir(parents=True)
            if key == "patience_4":
                card = {"arch": "LivenessNet", "arm": "arm_b", "fold": 1, "train_seed": 42, "history": self.history,
                        "fitting_keys_sha256": self.keys_sha("armB_fitting"), "dev_keys_sha256": self.keys_sha("dev"), **(card_over or {})}
                (run / "restricted" / f"{REF_MODEL}_card.json").write_text(json.dumps(card, indent=1) + "\n", encoding="utf-8")
                self.card_sha = sha((run / "restricted" / f"{REF_MODEL}_card.json").read_bytes())
            (run / "restricted" / "other.txt").write_text(key, encoding="utf-8")
            self.sums[key] = write_sums(run / "restricted")
            st = {"run_id": rid, "status": "PASS", "config_sha256": sha(cfgp.read_bytes()), "restricted_sums_sha256": self.sums[key],
                  "completed": names}
            (run / "status.json").write_text(json.dumps(st, indent=1) + "\n", encoding="utf-8")
        self.write_cfg(self.doc())

    def keys_sha(self, name: str) -> str:
        """Impronta delle chiavi di un manifest delle celle del fold 1 come nel runner accettato (chiavi ordinate)."""
        return sha("\n".join(sorted(r[2] for r in self.cells[(name, 1)])).encode())

    def counts(self, name: str) -> dict:
        rows = self.cells[(name, 1)]
        return {"n": len(rows), "bona_fide": sum(1 for r in rows if r[3] == "0"), "attack": sum(1 for r in rows if r[3] == "1"),
                "subjects": len({r[4] for r in rows})}

    def doc(self) -> dict:
        sr = self.sr_doc()
        env = FR.environment()
        nets = {n: AM.expected_counts(n, SHAPE) for n in AM.NETWORKS}

        def arch(n):
            c = nets[n]
            return {"trainable_parameters": c["trainable"], "non_trainable_parameters": c["non_trainable"], "batchnorm_layers": c["batchnorm_layers"],
                    "input_shape": list(SHAPE)}
        fp = {"fitting": sr["arms"]["arm_b"]["total"], "dev": sr["dev"]["total"], "test_per_split": list(sr["schema_expected"]["test_per_fold"])}
        nuaa = {"name": "NUAA", "access": "available", "official_source": "o", "identity_split": "i", "missing_access": None, "unit_kind": "fold",
                "splits": [1, 2, 3, 4], "seeds": list(SAR.SEEDS), "frames_per_unit": fp,
                "trainings": [{"network": n, "patience": p, "existing": 0, "new": 12,
                               "reference_run": ({4: P4, 12: P12}[p] if n == "LivenessNet" else None)} for n in AM.NETWORKS for p in D.PATIENCES]}

        def missing(name):
            return {"name": name, "access": "access missing", "official_source": None, "identity_split": None, "missing_access": "m",
                    "unit_kind": None, "splits": None, "seeds": list(SAR.SEEDS), "frames_per_unit": {"fitting": None, "dev": None, "test_per_split": None},
                    "trainings": [{"network": n, "patience": p, "existing": 0, "new": None, "reference_run": None} for n in AM.NETWORKS for p in D.PATIENCES]}
        return {
            "schema": P.CONFIG_SCHEMA, "task": "t", "authorisation": "a",
            "declarations": {k: ("no test frame is assembled or evaluated" if k == "no_test" else "d") for k in P.DECLARATION_KEYS},
            "parent_runs": {"patience_4": {"run_id": P4, "folder": "patience_4", "config_path": "config.json", "config_sha256": sha(self.config.read_bytes()),
                                           "restricted_sha256sums_sha256": self.sums["patience_4"], "early_stopping_patience": 4},
                            "patience_12": {"run_id": P12, "folder": "patience_12", "config_path": "config12.json",
                                            "config_sha256": sha(self.config12.read_bytes()), "restricted_sha256sums_sha256": self.sums["patience_12"],
                                            "early_stopping_patience": 12}},
            "code": {f: sha((ROOT / f).read_bytes()) for f in P.CODE_FILES},
            "architectures": {"LivenessNet": {"path": "ref/architectures.py", "sha256": self.code["ref/architectures.py"], "builder": "b", **arch("LivenessNet")},
                              "ResNet-18": {"module": "scripts/pad_arch_models.py", "definition": "d", "batchnorm": "b", **arch("ResNet-18")}},
            "training": {"recipe": "r", "trial_overrides": {"max_epochs": 3}, "trainer": "t", "timing": "t", "checkpoint": "none saved", "threads": "t"},
            "trial": {**P.TRIAL_UNIT, "roles": ["fitting", "dev"], "frames": {"fitting": self.counts("armB_fitting"), "dev": self.counts("dev")},
                      "networks": list(AM.NETWORKS), "inference_batch": 16, "inference": "i", "process": "p", "wall_time_cap_minutes": 60,
                      "cap_unit": P.CAP_UNIT, "cap_checks": "c", "estimate_minutes": 1, "upper_estimate_minutes": 2, "estimate": "e"},
            "baseline_check": {"reference_run": P4, "model": REF_MODEL, "card_sha256": self.card_sha, "epochs": 3, "fields": list(P.HISTORY_KEYS),
                               "difference": 0, "rule": "r", "source": "s",
                               "consequence": "if not reproduced, reported and explained before the pre-registration is closed"},
            "definitions": {"one_class": "o", "comparison": "c", "sign": "s", "subject_sensitivity": "s", "fixed_threshold": 0.5,
                            "decision": FR.DECISION_RULE, "eer_threshold": FR.THRESHOLD_CRITERION, "bootstrap": dict(SR.BOOT),
                            "conditions": [list(c) for c in D.CONDITIONS]},
            "plan": {"networks": list(AM.NETWORKS), "patiences": [4, 12], "bn_states": list(D.BN_STATES), "thresholds": list(D.THRESHOLD_RULES),
                     "max_epochs": 12, "datasets": [nuaa, missing("CelebA-Spoof"), missing("Replay-Attack")],
                     "costs": {"rates": "r", "training": "t", "evaluation": "e", "totals": "t"}},
            "outputs": {"trial": ["status.json", "run.log", "trial.json", "trial.md"], "plan": ["run_plan.csv", "run_plan.md"]},
            "environment": {"tensorflow": env["tensorflow_installed"], "keras": env.get("keras")},
            "test": P.TEST_READ}

    def write_cfg(self, doc: dict, path: Path = None) -> Path:
        p = path or self.cfg_path
        p.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        return p

    def argv(self, cmd: str, *extra, out=None, date=DATE, cfg=None) -> list:
        a = [cmd, "--config", str(cfg or self.cfg_path), "--input-root", str(self.base)]
        if cmd == "trial":
            a += ["--out-root", str(out or self.out), "--date", date, "--i-am-authorized"]
        return a + list(extra)


def hooks(w: World, **p) -> dict:
    """Le iniezioni di base: stato git, forma dei frame 8×8, SHA-256 delle configurazioni dello schema e delle celle del
    mondo, LivenessNet del codice di riferimento al posto del file finto del mondo."""
    return {"p": {"REPO_STATE": lambda root: dict(STATE), "ARCH_LOADER": lambda path: ref_arch(), **p},
            "sr": {"SCHEMA_CONFIG_SHA256": w.schema_sha, "CELLS_CONFIG_SHA256": w.cells_cfg_sha}, "fr": {"FRAME_SHAPE": SHAPE}}


WORLD = None


def world() -> World:
    global WORLD
    if WORLD is None:
        WORLD = World()
    return WORLD


# ============================================================================ dati sintetici delle definizioni

def unit(split, seed, frames, s_res, s_liv, t_res=0.5, t_liv=0.5, dataset="D"):
    """frames: lista di (soggetto, etichetta); punteggi d'attacco delle due reti."""
    subj = [f for f, _ in frames]
    y = [lab for _, lab in frames]
    return {"unit": (dataset, split, seed), "networks": {
        "ResNet-18": {"y": y, "s": list(s_res), "subject": subj, "threshold": t_res},
        "LivenessNet": {"y": y, "s": list(s_liv), "subject": subj, "threshold": t_liv}}}


def frames_of(subjects, per=2):
    """Per soggetto per frame bona fide e per frame d'attacco."""
    return [(s, lab) for s in subjects for lab in (0, 1) for _ in range(per)]


def perfect(frames):
    return [0.9 if lab else 0.1 for _, lab in frames]


def ref_acer(y, s, t):
    y, s = np.asarray(y), np.asarray(s, float)
    d = s > t
    a, b = y == 1, y == 0
    if not a.any() or not b.any():
        return None
    return (np.sum(a & ~d) / a.sum() + np.sum(b & d) / b.sum()) / 2


def ref_bootstrap(units, units_b=None):
    """Bootstrap di riferimento scritto qui: soggetti uniti, default_rng(0), integers(0, n, n), metriche ricalcolate sui
    frame ripetuti dei soggetti scelti, media sulle unità di Δ (o di Δ_a − Δ_b con units_b, stessi indici),
    ricampionamento scartato se un valore non è definito."""
    split_subj = {}
    for u in units:
        split_subj.setdefault(u["unit"][:2], sorted(set(u["networks"]["ResNet-18"]["subject"])))
    pooled = sorted({s for v in split_subj.values() for s in v})
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(1000):
        pick = rng.integers(0, len(pooled), len(pooled))
        mult = np.bincount(pick, minlength=len(pooled))
        def delta(u):
            out = []
            for net in ("ResNet-18", "LivenessNet"):
                n = u["networks"][net]
                idx = [i for i, s in enumerate(n["subject"]) for _ in range(mult[pooled.index(s)])]
                out.append(ref_acer([n["y"][i] for i in idx], [n["s"][i] for i in idx], n["threshold"]))
            return None if None in out else out[0] - out[1]
        ds = []
        for i, u in enumerate(units):
            d = delta(u)
            if units_b is not None and d is not None:
                e = delta(units_b[i])
                d = None if e is None else d - e
            ds.append(d)
        if None not in ds:
            vals.append(float(np.mean(ds)))
    return np.percentile(vals, [2.5, 97.5]) if vals else None, len(vals)


# ============================================================================ (a) run a una sola classe

class TestAOneClass(unittest.TestCase):
    y = [0] * 5 + [1] * 5

    def test_perfect_is_not_one_class(self):
        r = D.one_class_run(self.y, [0.1] * 5 + [0.9] * 5)
        self.assertEqual((r["evaluable"], r["one_class"], r["class"], r["n_decided_attack"]), (True, False, None, 5))

    def test_constant_bona_fide_and_attack(self):
        self.assertEqual(D.one_class_run(self.y, [0.2] * 10)["class"], "bona fide")
        self.assertEqual(D.one_class_run(self.y, [0.9] * 10)["class"], "attack")

    def test_tie_at_threshold_is_bona_fide(self):
        r = D.one_class_run(self.y, [0.5] * 10)
        self.assertEqual((r["one_class"], r["class"], r["n_decided_attack"]), (True, "bona fide", 0))

    def test_quasi_constant_is_not_one_class(self):
        r = D.one_class_run(self.y, [0.2] * 9 + [0.51])
        self.assertEqual((r["one_class"], r["n_decided_attack"]), (False, 1))
        r = D.one_class_run(self.y, [0.9] * 9 + [0.5])
        self.assertEqual((r["one_class"], r["n_decided_attack"]), (False, 9))

    def test_absent_class_is_not_evaluable(self):
        for y in ([0] * 4, [1] * 4):
            r = D.one_class_run(y, [0.2] * 4)
            self.assertEqual((r["evaluable"], r["one_class"], r["class"]), (False, None, None))

    def test_counts_k_of_n_with_not_evaluable_apart(self):
        recs = [{"network": "ResNet-18", "patience": 4, "bn_state": "ordinary", "result": D.one_class_run(self.y, s)}
                for s in ([0.2] * 10, [0.3] * 10, [0.9] * 10, [0.1] * 5 + [0.9] * 5)]
        recs.append({"network": "ResNet-18", "patience": 4, "bn_state": "ordinary", "result": D.one_class_run([0, 0], [0.9, 0.9])})
        recs.append({"network": "LivenessNet", "patience": 12, "bn_state": "re-estimated", "result": D.one_class_run(self.y, [0.4] * 10)})
        out = D.one_class_counts(recs)
        self.assertEqual([(c["network"], c["patience"], c["bn_state"]) for c in out],
                         [("LivenessNet", 12, "re-estimated"), ("ResNet-18", 4, "ordinary")])
        c = out[1]
        self.assertEqual((c["one_class"], c["evaluable"], c["not_evaluable"], c["all_bona_fide"], c["all_attack"], c["text"]),
                         (3, 4, 1, 2, 1, "3 of 4 evaluable"))
        with self.assertRaises(FR.Failure):
            D.one_class_counts([{"network": "VGG", "patience": 4, "bn_state": "ordinary", "result": recs[0]["result"]}])
        with self.assertRaises(FR.Failure):
            D.one_class_counts([{"network": "ResNet-18", "patience": 5, "bn_state": "ordinary", "result": recs[0]["result"]}])

    def test_out_of_domain(self):
        for y, s in (([0, 2], [0.1, 0.2]), ([0, 1], [0.1, float("nan")]), ([0, 1], [0.1, 1.5]), ([0, 1], [-0.1, 0.5]), ([0, 1], [0.1]),
                     ([], [])):
            with self.assertRaises(FR.Failure):
                D.one_class_run(y, s)


# ============================================================================ (b) confronto tra reti

class TestBComparison(unittest.TestCase):
    def test_perfect_against_constant(self):
        units = []
        for split, subs in ((1, ["a", "b", "c"]), (2, ["d", "e", "f"])):
            f = frames_of(subs)
            for seed in (42, 43):
                units.append(unit(split, seed, f, perfect(f), [0.2] * len(f)))
        out = D.compare_condition(units)
        self.assertEqual(out["mean_delta"], {"acer": -0.5, "apcer": -1.0, "bpcer": 0.0})
        self.assertEqual(out["acer_counts"], {"resnet18_lower": 4, "livenessnet_lower": 0, "ties": 0, "undefined": 0, "units": 4})
        self.assertEqual((out["interval"]["acer"]["low"], out["interval"]["acer"]["high"]), (-0.5, -0.5))
        self.assertEqual(out["interval"]["acer"]["valid_resamples"], ref_bootstrap(units)[1])
        self.assertGreater(out["interval"]["acer"]["valid_resamples"], 900)
        self.assertEqual(out["sign"], D.SIGN_RESNET_LOWER)
        self.assertEqual(sorted(out["per_seed"]), [42, 43])
        self.assertEqual(out["per_seed"][42]["acer_counts"]["units"], 2)
        self.assertEqual(out["bootstrap"]["n_subjects"], 6)

    def test_exact_tie_where_floats_differ(self):
        f = [("s", 0)] * 10 + [("s", 1)] * 10
        # ResNet-18: APCER 1/10, BPCER 2/10; LivenessNet: APCER 3/10, BPCER 0 -> ACER 3/20 per entrambe
        s_res = [0.9, 0.9] + [0.1] * 8 + [0.1] + [0.9] * 9
        s_liv = [0.1] * 10 + [0.1] * 3 + [0.9] * 7
        self.assertNotEqual((0.1 + 0.2) / 2, (0.3 + 0.0) / 2)
        out = D.compare_condition([unit(1, 42, f, s_res, s_liv)])
        row = out["rows"][0]
        self.assertEqual((row["metrics"]["ResNet-18"]["acer"], row["metrics"]["LivenessNet"]["acer"]), (Fraction(3, 20), Fraction(3, 20)))
        self.assertTrue(row["tie"])
        self.assertEqual(out["acer_counts"]["ties"], 1)
        self.assertEqual(out["mean_delta"]["acer"], 0.0)

    def test_identical_networks_interval_touching_zero(self):
        units = []
        for split, subs in ((1, list("abc")), (2, list("def"))):
            f = frames_of(subs)
            s = [0.2, 0.7, 0.4, 0.8] * (len(f) // 4)
            units.append(unit(split, 42, f, s, s))
        out = D.compare_condition(units)
        self.assertEqual((out["interval"]["acer"]["low"], out["interval"]["acer"]["high"]), (0.0, 0.0))
        self.assertEqual(out["sign"], D.SIGN_UNRESOLVED)
        self.assertEqual(out["acer_counts"], {"resnet18_lower": 0, "livenessnet_lower": 0, "ties": 2, "undefined": 0, "units": 2})

    def test_absent_class_undefined_and_dropped(self):
        f_ok = frames_of(list("abcde"))
        f_att = [(s, 1) for s in "fgh" for _ in range(2)]
        units = [unit(1, 42, f_ok, perfect(f_ok), [0.2] * len(f_ok)), unit(2, 42, f_att, [0.9] * 6, [0.2] * 6)]
        out = D.compare_condition(units)
        self.assertIsNone(out["mean_delta"]["acer"])
        self.assertIsNone(out["mean_delta"]["bpcer"])
        self.assertIsNotNone(out["mean_delta"]["apcer"])
        self.assertEqual(out["acer_counts"]["undefined"], 1)
        self.assertEqual(out["interval"]["acer"]["valid_resamples"], 0)
        self.assertEqual(out["sign"], D.SIGN_UNRESOLVED)

    def test_fewer_than_five_subjects_no_interval(self):
        for subs in (["a"], list("abcd")):
            f = frames_of(subs)
            out = D.compare_condition([unit(1, 42, f, perfect(f), [0.2] * len(f))])
            self.assertIsNone(out["interval"]["acer"]["low"])
            self.assertIn("fewer than 5", out["bootstrap"]["note"])
            self.assertEqual(out["sign"], D.SIGN_UNRESOLVED)
            self.assertEqual(out["mean_delta"]["acer"], -0.5)

    def test_bootstrap_against_an_independent_reference(self):
        rng = np.random.default_rng(7)
        units = []
        for split, subs in ((1, list("abc")), (2, list("defg"))):
            f = frames_of(subs, per=3)
            for seed in (42, 43):
                units.append(unit(split, seed, f, rng.uniform(0, 1, len(f)).round(2), rng.uniform(0, 1, len(f)).round(2),
                                  t_res=0.45, t_liv=0.55))
        out = D.compare_condition(units)
        (lo, hi), n = ref_bootstrap(units)
        self.assertAlmostEqual(out["interval"]["acer"]["low"], lo, places=12)
        self.assertAlmostEqual(out["interval"]["acer"]["high"], hi, places=12)
        self.assertEqual(out["interval"]["acer"]["valid_resamples"], n)
        want = np.mean([ref_acer(u["networks"]["ResNet-18"]["y"], u["networks"]["ResNet-18"]["s"], 0.45)
                        - ref_acer(u["networks"]["LivenessNet"]["y"], u["networks"]["LivenessNet"]["s"], 0.55) for u in units])
        self.assertAlmostEqual(out["mean_delta"]["acer"], want, places=12)
        seed42 = [u for u in units if u["unit"][2] == 42]
        (lo, hi), _ = ref_bootstrap(seed42)
        self.assertAlmostEqual(out["per_seed"][42]["interval"]["acer"]["low"], lo, places=12)
        self.assertAlmostEqual(out["per_seed"][42]["interval"]["acer"]["high"], hi, places=12)

    def test_thresholds_per_network_and_eer(self):
        f = frames_of(list("abcde"))
        s = [0.3 if lab else 0.2 for _, lab in f]
        out = D.compare_condition([unit(1, 42, f, s, s, t_res=0.25, t_liv=0.5)])
        self.assertEqual(out["rows"][0]["delta"]["acer"], Fraction(-1, 2))
        y = [lab for _, lab in f]
        self.assertEqual(D.condition_threshold("dev_eer", y, s), FR.eer_threshold(np.array(y), np.array(s))["value"])
        self.assertEqual(D.condition_threshold("fixed_0.5", y, s), 0.5)
        with self.assertRaises(FR.Failure):
            D.condition_threshold("youden", y, s)
        with self.assertRaises(FR.Failure):
            D.condition_threshold("dev_eer", [0, 0], [0.1, 0.2])

    def test_out_of_domain(self):
        f = frames_of(list("ab"))
        good = perfect(f)
        bad_cases = [
            [unit(1, 42, f, good[:-1] + [float("nan")], good)],
            [unit(1, 42, f, good[:-1] + [1.2], good)],
            [unit(1, 42, f, good, good, t_res=1.5)],
            [unit(1, 42, f, good, good, t_liv=float("inf"))],
            [unit(1, 42, f, good, good, t_res=True)],
            [unit(1, 42, f, good, good), unit(1, 42, f, good, good)],
            [unit(1, 42, f, good, good), unit(2, 42, f, good, good)],
            [unit(1, 42, f, good, good), unit(1, 43, frames_of(list("ac")), good, good)],
            [],
        ]
        for units in bad_cases:
            with self.assertRaises(FR.Failure):
                D.compare_condition(units)
        u = unit(1, 42, f, good, good)
        u["networks"]["LivenessNet"]["y"] = [1 - x for x in u["networks"]["LivenessNet"]["y"]]
        with self.assertRaises(FR.Failure):
            D.compare_condition([u])
        u = unit(1, 42, f, good, good)
        u["networks"]["LivenessNet"]["subject"] = list(reversed(u["networks"]["LivenessNet"]["subject"]))
        with self.assertRaises(FR.Failure):
            D.compare_condition([u])
        u = unit(1, 42, f, good, good)
        u["networks"]["LivenessNet"]["y"][0] = 2
        u["networks"]["ResNet-18"]["y"][0] = 2
        with self.assertRaises(FR.Failure):
            D.compare_condition([u])
        u = unit(1, 42, f, good, good)
        u["networks"]["VGG"] = u["networks"].pop("LivenessNet")
        with self.assertRaises(FR.Failure):
            D.compare_condition([u])


# ============================================================================ (c) segno e cambio di segno

def condition(delta_sign: int, subjects=list("abcdef"), flip_unit=None):
    """Una condizione con ResNet-18 migliore (delta_sign -1), peggiore (+1) o uguale (0) in ogni unità, salvo flip_unit."""
    units = []
    for split, subs in ((1, subjects[:3]), (2, subjects[3:])):
        f = frames_of(subs)
        good, bad = perfect(f), [0.2] * len(f)
        sign = -delta_sign if flip_unit == split else delta_sign
        units.append(unit(split, 42, f, good if sign < 0 else bad if sign > 0 else good, bad if sign < 0 else good if sign > 0 else good))
    return D.compare_condition(units)


class TestCSign(unittest.TestCase):
    def test_sign_of_bounds(self):
        self.assertEqual(D.sign_of({"low": -0.3, "high": -0.01}), D.SIGN_RESNET_LOWER)
        self.assertEqual(D.sign_of({"low": 0.01, "high": 0.3}), D.SIGN_LIVENESSNET_LOWER)
        self.assertEqual(D.sign_of({"low": -0.3, "high": 0.0}), D.SIGN_UNRESOLVED)
        self.assertEqual(D.sign_of({"low": 0.0, "high": 0.3}), D.SIGN_UNRESOLVED)
        self.assertEqual(D.sign_of({"low": -0.1, "high": 0.1}), D.SIGN_UNRESOLVED)
        self.assertEqual(D.sign_of({"low": None, "high": None}), D.SIGN_UNRESOLVED)

    def test_sign_change_only_between_resolved_opposite_signs(self):
        a, b = condition(-1), condition(+1)
        r = D.compare_conditions(a, b)
        self.assertEqual((r["sign_a"], r["sign_b"], r["sign_change"], r["point_estimate_inversion"]),
                         (D.SIGN_RESNET_LOWER, D.SIGN_LIVENESSNET_LOWER, True, False))
        self.assertEqual(r["units_changing_sign"], 2)
        self.assertEqual(r["contrast_mean"]["acer"], -1.0)
        self.assertEqual((r["contrast_interval"]["acer"]["low"], r["contrast_interval"]["acer"]["high"]), (-1.0, -1.0))
        r = D.compare_conditions(a, condition(-1))
        self.assertEqual((r["sign_change"], r["point_estimate_inversion"], r["units_changing_sign"]), (False, False, 0))

    def test_point_estimate_inversion_with_an_unresolved_condition(self):
        a = condition(-1)
        b = condition(+1, flip_unit=2)       # unità di segno opposto, media zero: non risolta
        self.assertEqual((b["sign"], b["mean_delta"]["acer"]), (D.SIGN_UNRESOLVED, 0.0))
        r = D.compare_conditions(a, b)
        self.assertEqual((r["sign_change"], r["point_estimate_inversion"], r["units_changing_sign"]), (False, False, 1))
        f1, f2 = frames_of(list("abc")), frames_of(list("def"))
        s1 = [0.2 if (s == "a" and lab) else (0.9 if lab else 0.1) for s, lab in f1]
        b = D.compare_condition([unit(1, 42, f1, s1, perfect(f1)), unit(2, 42, f2, perfect(f2), perfect(f2))])
        self.assertEqual((b["mean_delta"]["acer"], b["sign"], b["interval"]["acer"]["low"]), (1 / 12, D.SIGN_UNRESOLVED, 0.0))
        r = D.compare_conditions(a, b)
        self.assertEqual((r["sign_change"], r["point_estimate_inversion"], r["units_changing_sign"]), (False, True, 1))

    def test_contrast_interval_against_an_independent_reference(self):
        rng = np.random.default_rng(11)
        ua, ub = [], []
        for split, subs in ((1, list("abc")), (2, list("defg"))):
            f = frames_of(subs, per=3)
            sa, sl = rng.uniform(0, 1, len(f)).round(2), rng.uniform(0, 1, len(f)).round(2)
            ua.append(unit(split, 42, f, sa, sl))
            ub.append(unit(split, 42, f, rng.uniform(0, 1, len(f)).round(2), sl, t_res=0.4))
        r = D.compare_conditions(D.compare_condition(ua), D.compare_condition(ub))
        (lo, hi), n = ref_bootstrap(ua, ub)
        self.assertAlmostEqual(r["contrast_interval"]["acer"]["low"], lo, places=12)
        self.assertAlmostEqual(r["contrast_interval"]["acer"]["high"], hi, places=12)
        self.assertEqual(r["contrast_interval"]["acer"]["valid_resamples"], n)
        self.assertNotEqual(lo, hi)

    def test_conditions_on_different_units_refused(self):
        a = condition(-1)
        f = frames_of(list("abc"))
        with self.assertRaises(FR.Failure):
            D.compare_conditions(a, D.compare_condition([unit(1, 42, f, perfect(f), perfect(f))]))
        b = condition(-1, subjects=list("abcdeg"))
        with self.assertRaises(FR.Failure):
            D.compare_conditions(a, b)


# ============================================================================ (d) sensibilità ai soggetti

class TestDLeaveOneSubjectOut(unittest.TestCase):
    def test_one_subject(self):
        f = frames_of(["a"])
        r = D.leave_one_subject_out([unit(1, 42, f, perfect(f), [0.2] * 4)])
        self.assertEqual((r["full_mean_delta_acer"], r["full_sign"], r["n_excluded"], r["n_not_available"], r["min"], r["max"]),
                         (-0.5, -1, 1, 1, None, None))
        self.assertIsNone(r["values"][0]["mean_delta_acer"])
        self.assertIsNone(r["values"][0]["sign_differs"])
        self.assertEqual(r["values"][0]["denominators"], [{"unit": ("D", 1, 42), "n_attack": 0, "n_bona_fide": 0}])

    def test_class_absent_after_the_exclusion(self):
        f = [("a", 0), ("a", 0), ("a", 1), ("b", 1), ("b", 1), ("c", 1)]
        r = D.leave_one_subject_out([unit(1, 42, f, [0.1, 0.1, 0.9, 0.9, 0.9, 0.9], [0.2] * 6)])
        v = {x["excluded_subject"]: x for x in r["values"]}
        self.assertIsNone(v["a"]["mean_delta_acer"])
        self.assertEqual(v["a"]["denominators"][0], {"unit": ("D", 1, 42), "n_attack": 3, "n_bona_fide": 0})
        self.assertEqual(r["n_not_available"], 1)
        self.assertEqual((v["b"]["mean_delta_acer"], v["c"]["mean_delta_acer"]), (-0.5, -0.5))
        self.assertEqual(v["b"]["denominators"][0], {"unit": ("D", 1, 42), "n_attack": 2, "n_bona_fide": 2})

    def test_sign_changes_in_exactly_one_exclusion(self):
        # ResNet-18 sbaglia solo gli attacchi del soggetto a, LivenessNet nessuno: Δ > 0, e = 0 senza a
        f2 = frames_of(list("abcd"), per=2)
        s_res = [0.1 if lab == 0 else (0.2 if s == "a" else 0.9) for s, lab in f2]
        s_liv = [0.1 if lab == 0 else 0.9 for s, lab in f2]
        r = D.leave_one_subject_out([unit(1, 42, f2, s_res, s_liv)])
        self.assertEqual(r["full_sign"], 1)
        self.assertEqual([x["sign"] for x in r["values"]], [0, 1, 1, 1])
        self.assertEqual(r["n_sign_differs"], 1)
        self.assertEqual((r["min"], r["max"]), (0.0, 1 / 6))

    def test_zero_delta_and_two_units(self):
        f1, f2 = frames_of(list("ab")), frames_of(list("cd"))
        s1 = perfect(f1)
        units = [unit(1, 42, f1, s1, s1), unit(2, 42, f2, perfect(f2), perfect(f2))]
        r = D.leave_one_subject_out(units)
        self.assertEqual((r["full_mean_delta_acer"], r["full_sign"], r["n_sign_differs"], r["n_excluded"]), (0.0, 0, 0, 4))
        self.assertTrue(all(len(x["denominators"]) == 2 for x in r["values"]))
        self.assertEqual(r["values"][0]["denominators"], [{"unit": ("D", 1, 42), "n_attack": 2, "n_bona_fide": 2},
                                                          {"unit": ("D", 2, 42), "n_attack": 4, "n_bona_fide": 4}])
        s2 = [0.9 if lab == 0 and s == "c" else x for (s, lab), x in zip(f2, perfect(f2))]
        r = D.leave_one_subject_out([unit(1, 42, f1, s1, s1), unit(2, 42, f2, s2, perfect(f2))])
        self.assertEqual(r["full_sign"], 1)
        v = {x["excluded_subject"]: x for x in r["values"]}
        self.assertEqual((v["c"]["sign"], v["c"]["sign_differs"], v["a"]["sign_differs"]), (0, True, False))

    def test_not_part_of_the_sign_rule(self):
        src = inspect.getsource(D.sign_of) + inspect.getsource(D.compare_conditions) + inspect.getsource(D.compare_condition)
        self.assertNotIn("leave_one_subject_out", src)

    def test_out_of_domain(self):
        f = frames_of(list("ab"))
        with self.assertRaises(FR.Failure):
            D.leave_one_subject_out([unit(1, 42, f, [2.0] * 8, [0.1] * 8)])
        with self.assertRaises(FR.Failure):
            D.leave_one_subject_out([])


# ============================================================================ registro: aritmetica senza TensorFlow

class TestEModelArithmetic(unittest.TestCase):
    def test_resnet18_counts_from_the_specification(self):
        c = AM.spec_counts(AM.resnet18_spec())
        self.assertEqual((c["trunk"], c["dense"], c["trainable"], c["non_trainable"], c["batchnorm_layers"]),
                         (11_176_512, 1_026, 11_177_538, 9_600, 20))
        # verifica a mano: stem + 4 stadi, ognuno sommato a parte
        stem = 7 * 7 * 3 * 64 + 2 * 64
        s1 = 2 * (2 * 9 * 64 * 64 + 2 * 2 * 64)
        s = stem + s1
        cin = 64
        for f in (128, 256, 512):
            s += 9 * cin * f + 9 * f * f + cin * f + 3 * 2 * f + 2 * 9 * f * f + 2 * 2 * f
            cin = f
        self.assertEqual(s, AM.RESNET18_TRUNK)
        self.assertEqual(512 * 2 + 2, AM.RESNET18_DENSE)
        self.assertEqual(2 * (64 + 4 * 64 + 5 * 128 + 5 * 256 + 5 * 512), AM.RESNET18_MOVING)

    def test_spatial_sides(self):
        self.assertEqual(AM.resnet18_spatial(256), [128, 64, 64, 32, 16, 8])

    def test_livenessnet_counts(self):
        self.assertEqual(AM.livenessnet_counts((256, 256, 3)), {"trainable": 8405778, "non_trainable": 320, "batchnorm_layers": 5})
        self.assertEqual(AM.livenessnet_counts((8, 8, 3))["trainable"], 8405778 - 64 * 64 * 32 * 64 + 2 * 2 * 32 * 64)
        with self.assertRaises(AM.ModelProblem):
            AM.livenessnet_counts((10, 8, 3))
        with self.assertRaises(AM.ModelProblem):
            AM.expected_counts("VGG", (8, 8, 3))

    def test_topology_of_the_specification(self):
        t = AM.resnet18_topology()
        users = {}
        for name, (_, ins, _) in t.items():
            for i in ins:
                users.setdefault(i, []).append(name)
        adds = [n for n, v in t.items() if v[0] == "Add"]
        self.assertEqual(len(adds), 8)
        for a in adds:                                       # ogni somma seguita da una ReLU (e solo da quella)
            (nxt,) = users[a]
            self.assertEqual(t[nxt][0:1] + t[nxt][2], ("Activation", "relu"))
        for n, (k, ins, _) in t.items():                     # ogni BN preceduta dalla sua convoluzione
            if k == "BatchNormalization":
                self.assertEqual(t[ins[0]][0], "Conv2D")
        self.assertEqual(sum(1 for v in t.values() if v[0] == "BatchNormalization"), 20)
        for stage, _ in AM.STAGES:                            # conv-BN-ReLU-conv-BN; scorciatoia all'ingresso del blocco
            for block in AM.BLOCKS:
                p = f"layer{stage}_block{block}"
                chain = [f"{p}_conv1", f"{p}_bn1", f"{p}_relu1", f"{p}_conv2", f"{p}_bn2"]
                self.assertEqual([t[x][0] for x in chain], ["Conv2D", "BatchNormalization", "Activation", "Conv2D", "BatchNormalization"])
                for a, b in zip(chain[1:], chain):
                    self.assertEqual(t[a][1], (b,))
                first = t[f"{p}_conv1"][1][0]
                block_input = t[first][1][0] if first.endswith("_pad") else first
                short = t[f"{p}_add"][1][1]
                self.assertEqual(t[short][1][0] if short.endswith("_downsample_bn") else short,
                                 f"{p}_downsample_conv" if short.endswith("_downsample_bn") else block_input)
                if short.endswith("_downsample_bn"):
                    self.assertEqual(t[f"{p}_downsample_conv"][1], (block_input,))
                    self.assertEqual(t[f"{p}_downsample_conv"][2], ((1, 1), (2, 2), "valid", False, "HeNormal"))
                self.assertEqual(t[f"{p}_add"][1][0], f"{p}_bn2")
        self.assertTrue(all(v[2][4] == "HeNormal" and v[2][3] is False for v in t.values() if v[0] == "Conv2D"))
        self.assertEqual(t["output"], ("Dense", ("avgpool",), (2, "softmax", "GlorotUniform")))

    def test_unknown_kind_refused(self):
        with self.assertRaises(AM.ModelProblem):
            AM.spec_counts([("x", "pool", (1,))])


# ============================================================================ trainer: copia dichiarata

class TestFTrainerCopy(unittest.TestCase):
    def test_copy_has_only_the_declared_substitutions(self):
        src = inspect.getsource(FR.default_trainer)
        for old, new in P.TRAINER_SUBSTITUTIONS:
            self.assertEqual(src.count(old), 1, old[:80])
            src = src.replace(old, new)
        self.assertEqual(src, inspect.getsource(P.trial_trainer))

    def test_names_of_the_copy_are_the_originals(self):
        for n in ("hide_gpu", "training_callbacks", "check_scheduler", "fail"):
            self.assertIs(getattr(P, n), getattr(FR, n))

    def test_recipe_of_the_trial(self):
        w = world()
        with injected(**hooks(w)):
            cfg, _, pcfg = P.load_config(w.cfg_path, w.base)
        self.assertEqual(P.trial_training(cfg, pcfg), {**FR.FROZEN_TRAINING, "max_epochs": 3})


# ============================================================================ configurazione

class TestGConfig(unittest.TestCase):
    def setUp(self):
        self.w = world()
        self.ctx = injected(**hooks(self.w))
        self.ctx.__enter__()
        self.addCleanup(self.ctx.__exit__, None, None, None)

    def refused(self, mutate, needle: str):
        doc = self.w.doc()
        mutate(doc)
        p = self.w.write_cfg(doc, self.w.tmp / "bad.json")
        with self.assertRaises(FR.Failure) as cm:
            P.load_config(p, self.w.base)
        self.assertIn(needle, str(cm.exception))

    def test_synthetic_configuration_accepted(self):
        cfg, h, pcfg = P.load_config(self.w.cfg_path, self.w.base)
        self.assertEqual(h, sha(self.w.cfg_path.read_bytes()))
        self.assertEqual(pcfg["training"]["early_stopping_patience"], 4)

    def test_top_keys_schema_and_test_statement(self):
        self.refused(lambda d: d.update(extra=1), "fields must be exactly")
        self.refused(lambda d: d.pop("plan"), "fields must be exactly")
        self.refused(lambda d: d.update(schema="x"), "schema must be")
        self.refused(lambda d: d.update(test="reads the test"), "test must state exactly")
        self.refused(lambda d: d["declarations"].update(no_test="the test is fine"), "no test frame")
        self.refused(lambda d: d["declarations"].pop("threads"), "declarations needs exactly")

    def test_parent_runs(self):
        self.refused(lambda d: d["parent_runs"]["patience_4"].update(run_id="x"), "SESSABLRUN run id")
        self.refused(lambda d: d["parent_runs"]["patience_4"].update(early_stopping_patience=12), "early_stopping_patience must be 4")
        self.refused(lambda d: d["parent_runs"]["patience_12"].update(config_sha256="1" * 64), "differs from parent_runs.patience_12.config_sha256")
        self.refused(lambda d: d["parent_runs"]["patience_4"].update(config_sha256="zz"), "SHA-256 values malformed")
        doc = SAR.amended_doc(self.w.sr_doc(), sha(self.w.config.read_bytes()))
        doc["training"]["early_stopping_patience"] = 12
        doc["training"]["batch"] = 32
        p = self.w.write_config(doc, self.w.tmp / "c12bad.json")

        def other_child(d):
            d["parent_runs"]["patience_12"].update(config_path=str(p), config_sha256=sha(p.read_bytes()))
        with self.assertRaises(FR.Failure):
            other_child(d := self.w.doc())
            P.load_config(self.w.write_cfg(d, self.w.tmp / "bad.json"), self.w.base)
        doc = SAR.amended_doc(self.w.sr_doc(), "2" * 64)
        p = self.w.write_config(doc, self.w.tmp / "c12other.json")
        self.refused(lambda d: d["parent_runs"]["patience_12"].update(config_path=str(p), config_sha256=sha(p.read_bytes())),
                     "not the amendment of the patience_4 configuration")

    def test_code_and_architectures(self):
        self.refused(lambda d: d["code"].pop("scripts/pad_arch_models.py"), "code needs exactly")
        self.refused(lambda d: d["code"].update({"scripts/pad_arch_models.py": "x"}), "code SHA-256 values malformed")
        self.refused(lambda d: d["architectures"]["ResNet-18"].update(trainable_parameters=11177537), "differ from the arithmetic")
        self.refused(lambda d: d["architectures"]["LivenessNet"].update(non_trainable_parameters=0), "differ from the arithmetic")
        self.refused(lambda d: d["architectures"]["ResNet-18"].update(batchnorm_layers=21), "differ from the arithmetic")
        self.refused(lambda d: d["architectures"]["ResNet-18"].update(input_shape=[16, 16, 3]), "input_shape")
        self.refused(lambda d: d["architectures"]["ResNet-18"].update(module="scripts/other.py"), "module must be")
        self.refused(lambda d: d["architectures"]["LivenessNet"].update(sha256="3" * 64), "differs from the architecture of the accepted run")

    def test_training_and_trial(self):
        self.refused(lambda d: d["training"].update(trial_overrides={"max_epochs": 4}), "trial_overrides must be exactly")
        self.refused(lambda d: d["training"].update(trial_overrides={"max_epochs": 3, "batch": 32}), "trial_overrides must be exactly")
        self.refused(lambda d: d["training"].update(trial_overrides={"max_epochs": True}), "trial_overrides must be exactly")
        self.refused(lambda d: d["training"].update(checkpoint="saved"), "no checkpoint")
        self.refused(lambda d: d["trial"].update(fold=2), "trial unit")
        self.refused(lambda d: d["trial"].update(seed=True), "trial unit")
        self.refused(lambda d: d["trial"].update(roles=["fitting", "dev", "test"]), "never the test")
        self.refused(lambda d: d["trial"].update(networks=["ResNet-18", "LivenessNet"]), "in this order")
        self.refused(lambda d: d["trial"].update(inference_batch=32), "inference_batch")
        self.refused(lambda d: d["trial"].update(wall_time_cap_minutes=61), "from 1 to 60")
        self.refused(lambda d: d["trial"].update(wall_time_cap_minutes=0), "from 1 to 60")
        self.refused(lambda d: d["trial"].update(cap_unit="minutes"), "cap_unit")
        self.refused(lambda d: d["trial"].update(upper_estimate_minutes=60), "upper_estimate_minutes < wall_time_cap_minutes")
        self.refused(lambda d: d["trial"].update(estimate_minutes=3), "estimate_minutes <= upper_estimate_minutes")
        self.refused(lambda d: d["trial"]["frames"]["dev"].update(n=99), "n must be attack + bona_fide")
        self.refused(lambda d: d["trial"]["frames"]["fitting"].update(n=31, bona_fide=15), "trial.frames differ")

    def test_baseline_check(self):
        self.refused(lambda d: d["baseline_check"].update(reference_run=P12), "accepted run at patience 4")
        self.refused(lambda d: d["baseline_check"].update(model="arm_b_fold2_seed42"), "model of the trial unit")
        self.refused(lambda d: d["baseline_check"].update(epochs=2), "baseline_check.epochs")
        self.refused(lambda d: d["baseline_check"].update(fields=["loss", "val_loss"]), "shared-history fields")
        self.refused(lambda d: d["baseline_check"].update(difference=1e-6), "difference must be 0")
        self.refused(lambda d: d["baseline_check"].update(consequence="nothing"), "pre-registration")

    def test_definitions(self):
        self.refused(lambda d: d["definitions"].update(bootstrap={**SR.BOOT, "resamples": 500}), "recipe of the project")
        self.refused(lambda d: d["definitions"].update(conditions=d["definitions"]["conditions"][:7]), "8 conditions")
        self.refused(lambda d: d["definitions"].update(decision="attack if score >= threshold"), "rule and the EER criterion")
        self.refused(lambda d: d["definitions"].update(fixed_threshold=0.4), "fixed_threshold")
        self.refused(lambda d: d["definitions"].update(sign="ResNet-18 is better because of depth"), "causal")

    def test_plan_datasets(self):
        self.refused(lambda d: d["plan"].update(max_epochs=13), "plan.max_epochs")
        self.refused(lambda d: d["plan"].update(patiences=[4]), "plan networks, patiences")
        self.refused(lambda d: d["plan"]["datasets"].reverse(), "name must be")
        self.refused(lambda d: d["plan"]["datasets"][0].update(access="access missing", missing_access="m"), "NUAA) must be available")
        self.refused(lambda d: d["plan"]["datasets"][1].update(access="open"), "access must be one of")
        self.refused(lambda d: d["plan"]["datasets"][1].update(missing_access=None), "missing_access must state")
        self.refused(lambda d: d["plan"]["datasets"][0].update(official_source=None), "must be fixed")
        self.refused(lambda d: d["plan"]["datasets"][0]["frames_per_unit"].update(dev=None), "must be fixed")
        self.refused(lambda d: d["plan"]["datasets"][0]["frames_per_unit"].update(dev=17), "frames per unit")
        self.refused(lambda d: d["plan"]["datasets"][0]["frames_per_unit"].update(test_per_split=[1, 2]), "test_per_split")
        self.refused(lambda d: d["plan"]["datasets"][0].update(seeds=[42, 43]), "seeds must be")
        self.refused(lambda d: d["plan"]["datasets"][1].update(seeds=[1, 2, 3]), "seeds must be")
        self.refused(lambda d: d["plan"]["datasets"][0].update(splits=[1, 1, 2, 3]), "distinct")
        self.refused(lambda d: d["plan"]["datasets"][0]["trainings"][0].update(existing=12, new=0), "existing must be 0")
        self.refused(lambda d: d["plan"]["datasets"][0]["trainings"][2].update(new=11), "must equal the units")
        self.refused(lambda d: d["plan"]["datasets"][0]["trainings"][1].update(reference_run=P4), "reference_run")
        self.refused(lambda d: d["plan"]["datasets"][0]["trainings"][2].update(reference_run=P4), "reference_run")
        self.refused(lambda d: d["plan"]["datasets"][0]["trainings"].reverse(), "trainings[0] must be")
        self.refused(lambda d: d["plan"]["datasets"][2]["frames_per_unit"].update(fitting=0), "frames_per_unit.fitting")
        self.refused(lambda d: d["plan"]["datasets"][2].update(unit_kind="session"), "unit_kind")
        self.refused(lambda d: d["outputs"].update(plan=["plan.csv"]), "outputs must be")
        self.refused(lambda d: d["environment"].pop("keras"), "environment needs exactly")


# ============================================================================ ingressi e check-inputs

class TestHInputs(unittest.TestCase):
    def setUp(self):
        self.w = world()
        self.ctx = injected(**hooks(self.w))
        self.ctx.__enter__()
        self.addCleanup(self.ctx.__exit__, None, None, None)

    def load(self, doc=None):
        p = self.w.write_cfg(doc or self.w.doc(), self.w.tmp / "in.json")
        cfg, _, pcfg = P.load_config(p, self.w.base)
        return P.load_inputs(cfg, pcfg, self.w.base)

    def refused(self, needle: str, doc=None):
        with self.assertRaises(FR.Failure) as cm:
            self.load(doc)
        self.assertIn(needle, str(cm.exception))

    def test_inputs_pass_without_the_test_and_without_tensorflow(self):
        inp = self.load()
        self.assertEqual(inp.folds, {})
        self.assertEqual(inp.assembled, [])
        self.assertEqual(inp.reference_card["history"], self.w.history)
        rc, out, err = run_main(self.w.argv("check-inputs"))
        self.assertEqual(rc, 0, err)
        self.assertIn("fold schema not loaded, no frame assembled", out)
        self.assertNotIn("results", os.listdir(self.w.tmp))

    def test_test_guard(self):
        inp = self.load()
        store = FR.FrameStore(inp, inp.X_extracted, inp.extracted_keys)
        part = P.trial_frames(store, inp, "fitting")
        self.assertEqual(len(part["keys"]), self.w.counts("armB_fitting")["n"])
        with self.assertRaises(FR.Failure) as cm:
            P.trial_frames(store, inp, "test")
        self.assertIn("were assembled", str(cm.exception))
        inp.assembled = ["fitting"]
        inp.folds = {1: {"test": []}}
        with self.assertRaises(FR.Failure) as cm:
            P.assert_no_test(inp)
        self.assertIn("never open the test manifests", str(cm.exception))

    def test_code_pins(self):
        doc = self.w.doc()
        doc["code"]["scripts/pad_arch_definitions.py"] = "4" * 64
        self.refused("differ from the SHA-256 pinned", doc)

    def test_parent_status_and_sums(self):
        st = self.w.base / "patience_12" / "status.json"
        saved = st.read_text()
        try:
            st.write_text(json.dumps({**json.loads(saved), "status": "FAIL"}), encoding="utf-8")
            self.refused("is not PASS")
            st.write_text(json.dumps({**json.loads(saved), "completed": ["arm_b_fold1_seed42"]}), encoding="utf-8")
            self.refused("arm B models are not all completed")
        finally:
            st.write_text(saved, encoding="utf-8")
        doc = self.w.doc()
        doc["parent_runs"]["patience_4"]["restricted_sha256sums_sha256"] = "5" * 64
        self.refused("status.json is not PASS", doc)

    def test_reference_card(self):
        card = self.w.base / "patience_4" / "restricted" / f"{REF_MODEL}_card.json"
        saved = card.read_bytes()
        try:
            card.write_bytes(saved.replace(b"0.71", b"0.72"))
            self.refused("SHA-256")
        finally:
            card.write_bytes(saved)
        doc = self.w.doc()
        doc["baseline_check"]["card_sha256"] = "6" * 64
        self.refused("not listed in the restricted SHA256SUMS", doc)
        h = dict(self.w.history)
        try:
            self.w.write_parents({**h, "lr": h["lr"][:2]})
            self.refused("the history lacks")
        finally:
            self.w.write_parents(h)

    def test_keys_against_the_reference_card(self):
        try:
            self.w.write_parents(card_over={"dev_keys_sha256": "a" * 64})
            self.refused("trial dev: the keys differ from dev_keys_sha256 of the reference card")
            self.w.write_parents(card_over={"fitting_keys_sha256": "b" * 64})
            self.refused("trial fitting: the keys differ from fitting_keys_sha256")
            self.w.write_parents(card_over={"fitting_keys_sha256": None})
            self.refused("fitting_keys_sha256 or dev_keys_sha256 missing")
        finally:
            self.w.write_parents()
        inp = self.load()
        self.assertTrue(P.keys_check(inp.reference_card, "fitting", BR.role_rows(inp, "fitting", "arm_b", 1)))

    def test_assembled_keys_checked_against_the_card(self):
        real = FR.assemble

        def short(store, rows):
            return real(store, sorted(rows, key=lambda r: r["key"])[1:])
        out = self.w.tmp / "assemble_out"
        self.addCleanup(shutil.rmtree, out, True)
        with injected(fr={"assemble": short}):
            rc, o, err = run_main(self.w.argv("trial", out=out))
        self.assertEqual(rc, 1)
        self.assertIn("trial fitting: the keys differ from fitting_keys_sha256 of the reference card", err)
        run = out / RUN
        st = json.loads((run / "status.json").read_text())
        self.assertEqual(st["status"], "FAIL")
        self.assertEqual(sorted(os.listdir(run)), ["run.log", "status.json"])

    def test_output_root_symbolic_link_refused(self):
        target = self.w.tmp / "real_out"
        target.mkdir(exist_ok=True)
        link = self.w.tmp / "linked_out"
        if not link.is_symlink():
            link.symlink_to(target)
        rc, out, err = run_main(self.w.argv("trial", out=link))
        self.assertEqual(rc, 1)
        self.assertIn("the output root is a symbolic link", err)
        self.assertEqual(os.listdir(target), [])

    def test_card_of_another_model(self):
        try:
            self.w.write_parents(card_over={"fold": 2})
            self.refused("arm, fold or seed differ")
        finally:
            self.w.write_parents()

    def test_restricted_sums_file_anchored(self):
        f = self.w.base / "patience_12" / "restricted" / "SHA256SUMS"
        saved = f.read_bytes()
        try:
            f.write_bytes(saved + b"\n")
            self.refused("restricted SHA256SUMS has SHA-256")
        finally:
            f.write_bytes(saved)

    def test_architecture_file_pinned(self):
        a = self.w.base / "ref" / "architectures.py"
        saved = a.read_bytes()
        try:
            a.write_bytes(saved + b"# x\n")
            self.refused("architecture file")
        finally:
            a.write_bytes(saved)

    def test_final_checks(self):
        p = self.w.write_cfg(self.w.doc(), self.w.tmp / "final.json")
        cfg, h, _ = P.load_config(p, self.w.base)
        self.assertEqual(P.final_checks(cfg, p, h, self.w.base)["code_unchanged"], len(P.CODE_FILES))
        p.write_text(p.read_text() + " ", encoding="utf-8")
        with self.assertRaises(FR.Failure) as cm:
            P.final_checks(cfg, p, h, self.w.base)
        self.assertIn("configuration changed", str(cm.exception))
        p = self.w.write_cfg(self.w.doc(), self.w.tmp / "final.json")
        card = self.w.base / "patience_4" / "restricted" / f"{REF_MODEL}_card.json"
        saved = card.read_bytes()
        try:
            card.write_bytes(saved + b" ")
            with self.assertRaises(FR.Failure) as cm:
                P.final_checks(cfg, p, h, self.w.base)
            self.assertIn("reference card changed", str(cm.exception))
        finally:
            card.write_bytes(saved)

    def test_frames_and_cache(self):
        doc = self.w.doc()
        doc["trial"]["frames"]["fitting"]["subjects"] += 1
        self.refused("trial fitting: the manifest holds", doc)
        c = self.w.base / SR.fr_view(json.loads(self.w.config.read_text()))["cache"]["path"]
        saved = c.read_bytes()
        try:
            c.write_bytes(saved[:-1] + bytes([saved[-1] ^ 1]))
            self.refused("SHA-256")
        finally:
            c.write_bytes(saved)

    def test_environment(self):
        doc = self.w.doc()
        doc["environment"]["keras"] = "0.0.1"
        self.refused("configuration 0.0.1", doc)


# ============================================================================ piano

def fake_trial(w: World, folder: Path, nets=None, status="PASS", over=None) -> dict:
    """Una prova finta (status.json e trial.json coerenti) con tempi scelti a mano."""
    cfg = json.loads(w.cfg_path.read_text())
    nets = nets or {
        "LivenessNet": {"epochs": [{"epoch": i, "seconds_total": 10.0, "seconds_validation": 2.0, "cpu_seconds": 40.0} for i in (1, 2, 3)],
                        "inference": {"fitting": {"frames": 32, "seconds": 0.8, "cpu_seconds": 3.2}, "dev": {"frames": 16, "seconds": 0.4, "cpu_seconds": 1.6}}},
        "ResNet-18": {"epochs": [{"epoch": i, "seconds_total": 30.0, "seconds_validation": 6.0, "cpu_seconds": 90.0} for i in (1, 2, 3)],
                      "inference": {"fitting": {"frames": 32, "seconds": 3.2, "cpu_seconds": 9.6}, "dev": {"frames": 16, "seconds": 1.6, "cpu_seconds": 4.8}}}}
    for v in nets.values():
        v.setdefault("training_overhead", {"seconds": 5.0, "cpu_seconds": 10.0} if v["epochs"] else None)
    code = {**cfg["code"], "scripts/pad_arch_comparison.py": sha((ROOT / "scripts/pad_arch_comparison.py").read_bytes())}
    doc = {"schema": P.TRIAL_SCHEMA, "run_id": RUN, "status": status, "unit": {**P.TRIAL_UNIT, "frames": cfg["trial"]["frames"]}, "networks": nets,
           "config_sha256": sha(w.cfg_path.read_bytes()), "code_sha256": code, **(over or {})}
    folder.mkdir(parents=True, exist_ok=True)
    b = (json.dumps(doc, indent=1) + "\n").encode()
    (folder / "trial.json").write_bytes(b)
    (folder / "status.json").write_text(json.dumps({"run_id": RUN, "status": status, "trial_json_sha256": sha(b)}), encoding="utf-8")
    return doc


class TestIPlan(unittest.TestCase):
    def setUp(self):
        self.w = world()
        self.ctx = injected(**hooks(self.w))
        self.ctx.__enter__()
        self.addCleanup(self.ctx.__exit__, None, None, None)
        self.out = self.w.tmp / "plan_out"
        self.addCleanup(shutil.rmtree, self.out, True)
        self.addCleanup(shutil.rmtree, self.w.tmp / "trialfake", True)

    def plan(self, *extra, out=None):
        return run_main(self.w.argv("plan", "--out-dir", str(out or self.out), *extra))

    def test_plan_without_trial(self):
        rc, out, err = self.plan()
        self.assertEqual(rc, 0, err)
        text = (self.out / "run_plan.csv").read_text()
        lines = text.splitlines()
        self.assertEqual(lines[0], ",".join(P.PLAN_COLUMNS))
        self.assertEqual(len(lines), 13)
        self.assertIn("NUAA,available,LivenessNet,4,12,0,12," + P4 + ",5,12,32,16,", text)
        self.assertIn(f"CelebA-Spoof,access missing,ResNet-18,12,{P.TBF},0,{P.TBF},none,12,12,{P.TBF},{P.TBF},{P.TBF},4,,,,,,,", text)

    def test_plan_with_a_trial_numbers(self):
        folder = self.w.tmp / "trialfake"
        fake_trial(self.w, folder)
        rc, out, err = self.plan("--trial-run", str(folder))
        self.assertEqual(rc, 0, err)
        rows = {(r[0], r[2], r[3]): r for r in (l.split(",") for l in (self.out / "run_plan.csv").read_text().splitlines()[1:])}
        col = {k: i for i, k in enumerate(P.PLAN_COLUMNS)}
        sr = self.w.sr_doc()
        F, Dv, T = sr["arms"]["arm_b"]["total"], sr["dev"]["total"], sr["schema_expected"]["test_per_fold"]
        # LivenessNet: (10 - 2) / 32 * 1000 s per 1.000 frame di fitting, 2 / 16 * 1000 di validazione, 1.2 / 48 * 1000 d'inferenza
        tr, va, inf, ratio = 8 / F * 1000, 2 / Dv * 1000, 1.2 / 48 * 1000, (120 + 4.8 + 10) / (30 + 1.2 + 5)
        per_epoch = (tr * F + va * Dv) / 1000
        ev = 3 * sum(inf * (6 * F + 2 * Dv + 2 * t) / 1000 for t in T)
        r = rows[("NUAA", "LivenessNet", "4")]
        self.assertEqual(r[col["training_wall_hours_min"]], f"{12 * (5 * per_epoch + 5) / 3600:.2f}")
        self.assertEqual(r[col["training_wall_hours_max"]], f"{12 * (12 * per_epoch + 5) / 3600:.2f}")
        self.assertEqual(r[col["evaluation_wall_hours"]], f"{ev / 3600:.2f}")
        self.assertEqual(r[col["cpu_hours_max"]], f"{(12 * (12 * per_epoch + 5) + ev) * ratio / 3600:.2f}")
        r = rows[("NUAA", "ResNet-18", "12")]
        inf_r = 4.8 / 48 * 1000
        ev_r = 3 * sum(inf_r * (21 * F + 2 * Dv + 2 * t) / 1000 for t in T)
        self.assertEqual(r[col["evaluation_wall_hours"]], f"{ev_r / 3600:.2f}")
        self.assertEqual((r[col["epochs_min"]], r[col["epochs_max"]]), ("12", "12"))
        md = (self.out / "run_plan.md").read_text()
        self.assertIn("| NUAA | 4 of 4 |", md)
        self.assertIn("| CelebA-Spoof | 0 of 4 |", md)
        self.assertIn(f"the trial {RUN}", md)

    def test_plan_is_deterministic_and_never_overwritten(self):
        folder = self.w.tmp / "trialfake"
        fake_trial(self.w, folder)
        self.assertEqual(self.plan("--trial-run", str(folder))[0], 0)
        first = {n: (self.out / n).read_bytes() for n in P.PLAN_OUTPUTS}
        rc, out, err = self.plan("--trial-run", str(folder))
        self.assertEqual(rc, 1)
        self.assertIn("already exist", err)
        self.assertEqual(first, {n: (self.out / n).read_bytes() for n in P.PLAN_OUTPUTS})
        out2 = self.w.tmp / "plan_out2"
        self.addCleanup(shutil.rmtree, out2, True)
        self.assertEqual(self.plan("--trial-run", str(folder), out=out2)[0], 0)
        self.assertEqual(first, {n: (out2 / n).read_bytes() for n in P.PLAN_OUTPUTS})

    def test_incomplete_trial_leaves_cells_empty(self):
        folder = self.w.tmp / "trialfake"
        nets = {"LivenessNet": {"epochs": [{"epoch": 1, "seconds_total": 10.0, "seconds_validation": 2.0, "cpu_seconds": 40.0}], "inference": {}},
                "ResNet-18": {"epochs": [], "inference": {}}}
        fake_trial(self.w, folder, nets, status="INCOMPLETE")
        self.assertEqual(self.plan("--trial-run", str(folder))[0], 0)
        rows = [l.split(",") for l in (self.out / "run_plan.csv").read_text().splitlines()[1:]]
        col = {k: i for i, k in enumerate(P.PLAN_COLUMNS)}
        self.assertNotEqual(rows[0][col["training_wall_hours_min"]], "")
        self.assertEqual(rows[0][col["evaluation_wall_hours"]], "")
        self.assertEqual(rows[2][col["training_wall_hours_min"]], "")

    def test_trial_refused(self):
        folder = self.w.tmp / "trialfake"
        fake_trial(self.w, folder)
        (folder / "trial.json").write_text((folder / "trial.json").read_text().replace('"PASS"', '"PASS" '), encoding="utf-8")
        rc, out, err = self.plan("--trial-run", str(folder))
        self.assertEqual((rc, os.path.exists(self.out / "run_plan.csv")), (1, False))
        self.assertIn("SHA-256 differs", err)
        fake_trial(self.w, folder, status="FAIL")
        self.assertIn("not a PASS or INCOMPLETE trial", self.plan("--trial-run", str(folder))[2])
        doc = fake_trial(self.w, folder)
        doc["unit"]["fold"] = 2
        b = (json.dumps(doc) + "\n").encode()
        (folder / "trial.json").write_bytes(b)
        (folder / "status.json").write_text(json.dumps({"run_id": RUN, "status": "PASS", "trial_json_sha256": sha(b)}), encoding="utf-8")
        self.assertIn("trial unit", self.plan("--trial-run", str(folder))[2])
        doc = fake_trial(self.w, folder, over={"run_id": "20261117-ARCHTRIAL-1234567"})
        self.assertIn("run id or status differ", self.plan("--trial-run", str(folder))[2])
        fake_trial(self.w, folder, over={"config_sha256": "8" * 64})
        self.assertIn("another configuration", self.plan("--trial-run", str(folder))[2])
        cfg = json.loads(self.w.cfg_path.read_text())
        fake_trial(self.w, folder, over={"code_sha256": {**cfg["code"], "scripts/pad_arch_comparison.py": "9" * 64}})
        self.assertIn("other code", self.plan("--trial-run", str(folder))[2])
        fake_trial(self.w, folder, over={"code_sha256": {**cfg["code"], "scripts/pad_arch_models.py": "9" * 64,
                                                        "scripts/pad_arch_comparison.py": sha((ROOT / "scripts/pad_arch_comparison.py").read_bytes())}})
        self.assertIn("other code", self.plan("--trial-run", str(folder))[2])
        self.assertFalse(self.out.exists())

    def test_plan_checks_the_pinned_code(self):
        doc = self.w.doc()
        doc["code"]["scripts/pad_arch_definitions.py"] = "c" * 64
        bad = self.w.write_cfg(doc, self.w.tmp / "plan_badcode.json")
        rc, out, err = run_main(self.w.argv("plan", "--out-dir", str(self.out), cfg=bad))
        self.assertEqual(rc, 1)
        self.assertIn("differ from the SHA-256 pinned", err)
        self.assertFalse(self.out.exists())

    def test_plan_folder_inside_the_repository_refused(self):
        rc, out, err = self.plan(out=ROOT / "scripts" / "plan_should_not_exist")
        self.assertEqual(rc, 1)
        self.assertIn("refusing the plan folder", err)
        self.assertFalse((ROOT / "scripts" / "plan_should_not_exist").exists())


# ============================================================================ configurazione consegnata

class TestJDeliveredConfiguration(unittest.TestCase):
    def test_delivered_configuration_loads(self):
        p = ROOT / P.DEFAULT_CONFIG
        if not p.is_file() or not (ROOT / "delivery/sessionablationrun/nuaa_session_ablation_run.json").is_file():
            skip_missing(self, "delivered configuration")
        with injected(fr={"FRAME_SHAPE": (256, 256, 3)}):
            cfg, _, pcfg = P.load_config(p, ROOT)
        self.assertEqual(cfg["code"], {f: sha((ROOT / f).read_bytes()) for f in P.CODE_FILES})
        self.assertEqual(cfg["architectures"]["ResNet-18"]["trainable_parameters"], 11_177_538)
        self.assertEqual(cfg["training"]["trial_overrides"], {"max_epochs": 3})
        self.assertEqual([e["new"] for e in cfg["plan"]["datasets"][0]["trainings"]], [12, 12, 12, 12])
        self.assertEqual([e["existing"] for d in cfg["plan"]["datasets"] for e in d["trainings"]], [0] * 12)


# ============================================================================ TensorFlow: modelli

class TestKModels(unittest.TestCase):
    def setUp(self):
        need_tf(self)
        import tensorflow as tf
        FR.hide_gpu(tf)
        self.tf = tf

    def test_resnet18_built_at_256(self):
        m = AM.resnet18((256, 256, 3))
        got = AM.check_model(m, "ResNet-18", (256, 256, 3), AM.expected_counts("ResNet-18", (256, 256, 3)))
        self.assertEqual((got["trainable"], got["non_trainable"], got["batchnorm_layers"]), (11_177_538, 9_600, 20))
        sides = {l.name: l.output.shape[1] for l in m.layers}
        self.assertEqual([sides[n] for n in ("conv1", "pool1", "layer1_block2_relu2", "layer2_block2_relu2", "layer3_block2_relu2",
                                             "layer4_block2_relu2")], AM.resnet18_spatial(256))
        self.assertFalse(any(isinstance(l, self.tf.keras.Model) for l in m.layers))
        self.assertTrue(all(l.momentum == 0.99 and l.epsilon == 1e-3 for l in m.layers if isinstance(l, self.tf.keras.layers.BatchNormalization)))
        self.assertTrue(all(not l.use_bias for l in m.layers if isinstance(l, self.tf.keras.layers.Conv2D)))
        names = [l.name for l in m.layers]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names[-1], "output")
        del m
        self.tf.keras.backend.clear_session()

    def test_livenessnet_has_the_same_batchnorm_defaults(self):
        m = AM.build("LivenessNet", (256, 256, 3), ref_arch())
        got = AM.check_model(m, "LivenessNet", (256, 256, 3), AM.expected_counts("LivenessNet", (256, 256, 3)))
        self.assertEqual(tuple(got["batchnorm_names"]), AM.LIVENESSNET_BN)
        bn = [l for l in m.layers if isinstance(l, self.tf.keras.layers.BatchNormalization)]
        self.assertEqual({(l.momentum, l.epsilon) for l in bn}, {(AM.BN_MOMENTUM, AM.BN_EPSILON)})

    def test_check_model_refusals(self):
        tf = self.tf
        exp = AM.expected_counts("ResNet-18", SHAPE)
        m = AM.resnet18(SHAPE)
        with self.assertRaises(AM.ModelProblem):
            AM.check_model(m, "ResNet-18", SHAPE, {**exp, "trainable": exp["trainable"] + 1})
        m.get_layer("bn1").momentum = 0.9
        with self.assertRaises(AM.ModelProblem):
            AM.check_model(m, "ResNet-18", SHAPE, exp)
        inner = AM.resnet18(SHAPE)
        x = tf.keras.layers.Input(SHAPE)
        nested = tf.keras.Model(x, inner(x))
        with self.assertRaises(AM.ModelProblem) as cm:
            AM.check_model(nested, "ResNet-18", SHAPE, exp)
        self.assertIn("nested", str(cm.exception))
        x = tf.keras.layers.Input(SHAPE)
        y = tf.keras.layers.Dense(2, activation="sigmoid", name="output")(tf.keras.layers.Flatten()(x))
        with self.assertRaises(AM.ModelProblem):
            AM.check_model(tf.keras.Model(x, y), "ResNet-18", SHAPE, exp)
        with self.assertRaises(AM.ModelProblem):
            AM.check_model(AM.resnet18(SHAPE), "ResNet-18", (16, 16, 3), exp)
        cfg = AM.resnet18(SHAPE).get_config()
        for l in cfg["layers"]:
            if l["name"] == "layer2_block1_conv2":
                l["config"]["kernel_initializer"] = {"class_name": "GlorotUniform", "config": {"seed": None}}
        other = tf.keras.Model.from_config(cfg)
        self.assertEqual(AM.model_topology(other)["layer2_block1_conv2"][2][4], "GlorotUniform")
        with self.assertRaises(AM.ModelProblem) as cm:
            AM.check_model(other, "ResNet-18", SHAPE, exp)
        self.assertIn("layer2_block1_conv2", str(cm.exception))
        real = AM.resnet18_topology
        changed = real()
        changed["layer3_block1_relu2"] = ("Activation", ("layer3_block1_add",), ("linear",))
        try:
            AM.resnet18_topology = lambda: changed
            with self.assertRaises(AM.ModelProblem) as cm:
                AM.check_model(AM.resnet18(SHAPE), "ResNet-18", SHAPE, exp)
            self.assertIn("layer3_block1_relu2", str(cm.exception))
        finally:
            AM.resnet18_topology = real
        tf.keras.backend.clear_session()


# ============================================================================ TensorFlow: la prova tecnica

class TestLTrial(unittest.TestCase):
    """La prova vera su CPU nel mondo 8×8. La scheda di riferimento del mondo è scritta con la storia di LivenessNet
    prodotta da trial_trainer nello stesso modo della prova (3 epoche, seed 42): la prova deve riprodurla."""

    @classmethod
    def setUpClass(cls):
        if not TF_AVAILABLE or not REF_ARCH.is_file():
            return
        w = world()
        with injected(**hooks(w)):
            cfg, _, pcfg = P.load_config(w.cfg_path, w.base)
            inp = P.load_inputs(cfg, pcfg, w.base)
            store = FR.FrameStore(inp, inp.X_extracted, inp.extracted_keys)
            fit, dev = P.trial_frames(store, inp, "fitting"), P.trial_frames(store, inp, "dev")
            rec = []
            with contextlib.redirect_stdout(io.StringIO()):
                card, _ = P.trial_trainer({"X_fit": fit["X"], "y_fit": fit["y"], "X_dev": dev["X"], "y_dev": dev["y"]}, P.trial_training(cfg, pcfg),
                                          "LivenessNet", 42, lambda: P.build_checked(cfg, "LivenessNet", ref_arch()), lambda s: None, P.EpochTimer(rec))
        cls.history = card["history"]
        w.write_parents(card["history"])
        cls.fit, cls.dev, cls.cfg, cls.pcfg = fit, dev, cfg, pcfg

    def setUp(self):
        need_tf(self)
        self.w = world()
        self.out = self.w.tmp / "trial_out"
        shutil.rmtree(self.out, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.out, True)

    def go(self, argv=None, **p):
        h = hooks(self.w, **{k: v for k, v in p.items() if k not in ("sr", "fr")})
        # memoria di picco fissata: nel mondo sintetico i soggetti sono 1101-1116 e un picco di circa 1.100 MB scritto
        # come intero coinciderebbe con un identificativo (il controllo delle uscite pubbliche lo rifiuterebbe)
        h["sr"]["peak_rss_mb"] = lambda: 2048.0
        h["sr"].update(p.get("sr", {}))
        h["fr"].update(p.get("fr", {}))
        with injected(**h):
            rc, out, err = run_main(argv or self.w.argv("trial", out=self.out))
        return rc, out, err, self.out / RUN

    def assert_final_status(self, run: Path, status: str):
        st = json.loads((run / "status.json").read_text())
        self.assertEqual(st["status"], status)
        for k in ("started", "finished", "wall_seconds", "process_cpu_seconds", "cap_minutes", "cap_unit"):
            self.assertIn(k, st)
        self.assertEqual(st["cap_unit"], P.CAP_UNIT)
        self.assertIsInstance(st["process_cpu_seconds"], float)
        self.assertGreaterEqual(st["process_cpu_seconds"], 0)
        return st

    def test_trial_pass_then_plan_then_refused_again(self):
        rc, out, err, run = self.go()
        self.assertEqual(rc, 0, err[-3000:])
        self.assertEqual(sorted(os.listdir(run)), ["run.log", "status.json", "trial.json", "trial.md"])
        st = self.assert_final_status(run, "PASS")
        self.assertEqual(st["trial_json_sha256"], sha((run / "trial.json").read_bytes()))
        self.assertEqual(st["run_log_sha256"], sha((run / "run.log").read_bytes()))
        doc = json.loads((run / "trial.json").read_text())
        self.assertEqual((doc["status"], doc["started"], doc["batch"], doc["inference_batch"], doc["device"]),
                         ("PASS", st["started"], 16, 16, "CPU (GPU hidden)"))
        self.assertEqual(doc["visible_devices"], ["CPU"])
        for net, (tr, nt, bn) in (("LivenessNet", (AM.livenessnet_counts(SHAPE)["trainable"], 320, 5)), ("ResNet-18", (11_177_538, 9_600, 20))):
            r = doc["networks"][net]
            self.assertEqual(len(r["epochs"]), 3)
            self.assertEqual((r["parameters"]["trainable"], r["parameters"]["non_trainable"], r["batchnorm_layers"]), (tr, nt, bn))
            self.assertEqual(r["steps_per_epoch"], -(-self.w.counts("armB_fitting")["n"] // 16))
            self.assertEqual(sorted(r["inference"]), ["dev", "fitting"])
            self.assertTrue(all(v is not None for v in r["rates"].values()))
            self.assertTrue(all(e["seconds_validation"] < e["seconds_total"] for e in r["epochs"]))
            self.assertEqual(r["card"]["checkpoint"], "none saved (trial)")
            self.assertEqual(r["card"]["max_epochs"], 3)
        self.assertEqual(doc["networks"]["LivenessNet"]["threads"]["set_by_this_runner"], False)
        self.assertEqual(doc["baseline_check"]["outcome"], "reproduced")
        self.assertEqual(set(doc["baseline_check"]["max_abs_difference"].values()), {0.0})
        self.assertEqual(doc["data_check"], {"fitting_keys_equal_to_reference_card": True, "dev_keys_equal_to_reference_card": True})
        for net in AM.NETWORKS:
            ov = doc["networks"][net]["training_overhead"]
            self.assertGreater(ov["seconds"], 0)
            self.assertEqual(doc["networks"][net]["rates"]["overhead_seconds_per_training"], ov["seconds"])
        self.assertIn("logical_cpus", doc["machine"])
        for text in ((run / "trial.json").read_text(), (run / "trial.md").read_text(), (run / "run.log").read_text()):
            self.assertNotIn("real/", text)
            self.assertNotIn("attack/", text)
            self.assertNotIn(str(Path.home()), text)
        self.assertFalse(any(self.out.rglob("*.keras")))
        plan = self.w.tmp / "plan_from_trial"
        self.addCleanup(shutil.rmtree, plan, True)
        with injected(**hooks(self.w)):
            rc, out, err = run_main(self.w.argv("plan", "--trial-run", str(run), "--out-dir", str(plan)))
        self.assertEqual(rc, 0, err)
        self.assertIn("| NUAA | 4 of 4 |", (plan / "run_plan.md").read_text())
        before = {n: (run / n).read_bytes() for n in os.listdir(run)}
        rc, out, err, _ = self.go()
        self.assertEqual(rc, 1)
        self.assertIn("already exists", err)
        self.assertEqual(before, {n: (run / n).read_bytes() for n in os.listdir(run)})

    def test_not_reproduced_is_recorded_and_status_unchanged(self):
        h = json.loads(json.dumps(self.history))
        h["val_loss"][2] += 1e-6                 # solo la terza epoca: il confronto deve coprire tutte le 3
        self.w.write_parents(h)
        try:
            rc, out, err, run = self.go()
        finally:
            self.w.write_parents(self.history)
        self.assertEqual(rc, 0, err[-3000:])
        doc = json.loads((run / "trial.json").read_text())
        b = doc["baseline_check"]
        self.assertEqual((b["outcome"], b["first_differing_epoch"]["val_loss"], b["first_differing_epoch"]["loss"]), ("not reproduced", 3, None))
        self.assertAlmostEqual(b["max_abs_difference"]["val_loss"], 1e-6, places=9)
        self.assertEqual(doc["status"], "PASS")

    def test_stop_during_resnet_epochs_is_incomplete(self):
        real = SR.Budget.check

        def check(self_, step):
            if step == "epoch 2 of ResNet-18":
                self_.stop_requested = "SIGUSR1"
            return real(self_, step)
        rc, out, err, run = self.go(sr={"Budget": type("B", (SR.Budget,), {"check": check})})
        self.assertEqual(rc, 2, err[-3000:])
        st = self.assert_final_status(run, "INCOMPLETE")
        self.assertEqual((st["stopped"]["cause"], st["stopped"]["step"]), ("SIGUSR1", "epoch 2 of ResNet-18"))
        doc = json.loads((run / "trial.json").read_text())
        self.assertEqual([len(doc["networks"][n]["epochs"]) for n in AM.NETWORKS], [3, 2])
        self.assertEqual(doc["networks"]["ResNet-18"]["inference"], {})
        self.assertEqual(doc["baseline_check"]["outcome"], "reproduced")
        self.assertIsNone(doc["networks"]["ResNet-18"]["rates"]["inference_seconds_per_1000_frames"])

    def test_stop_between_the_two_inference_passes(self):
        real = SR.Budget.check

        def check(self_, step):
            if step == "inference on the dev of ResNet-18":
                self_.stop_requested = "SIGUSR1"
            return real(self_, step)
        rc, out, err, run = self.go(sr={"Budget": type("B", (SR.Budget,), {"check": check})})
        self.assertEqual(rc, 2, err[-3000:])
        doc = json.loads((run / "trial.json").read_text())
        self.assertEqual(sorted(doc["networks"]["ResNet-18"]["inference"]), ["fitting"])
        self.assertEqual(sorted(doc["networks"]["LivenessNet"]["inference"]), ["dev", "fitting"])
        self.assertEqual(doc["stopped"]["step"], "inference on the dev of ResNet-18")

    def test_wall_time_cap_before_any_epoch(self):
        doc = self.w.doc()
        doc["trial"].update(wall_time_cap_minutes=1, estimate_minutes=0.1, upper_estimate_minutes=0.2)
        self.w.write_cfg(doc)
        t = [0.0]
        try:
            rc, out, err, run = self.go(sr={"CLOCK": lambda: t.__setitem__(0, t[0] + 100.0) or t[0]})
        finally:
            self.w.write_cfg(self.w.doc())
        self.assertEqual(rc, 2, err[-3000:])
        st = self.assert_final_status(run, "INCOMPLETE")
        self.assertEqual((st["stopped"]["cause"], st["stopped"]["step"], st["baseline_check"]), ("cap", "train LivenessNet", "not checked"))
        doc = json.loads((run / "trial.json").read_text())
        self.assertEqual([len(doc["networks"][n]["epochs"]) for n in AM.NETWORKS], [0, 0])
        self.assertIn("wall-time cap", (run / "run.log").read_text())

    def test_overhead_is_the_trainer_minus_its_epochs(self):
        """Orologio finto: il trainer finto dura S + 7 secondi e registra epoche per S secondi; il costo fuori dalle epoche
        deve essere 7, non S + 7 (nessun doppio conteggio delle epoche)."""
        import tensorflow as tf
        from unittest import mock
        x = tf.keras.layers.Input(SHAPE)
        tiny = tf.keras.Model(x, tf.keras.layers.Dense(2, activation="softmax", name="output")(tf.keras.layers.Flatten()(x)))
        clock = [1000.0]
        hist = {k: [0.5, 0.4] for k in P.HISTORY_KEYS}

        def fake(data, training, network, seed, build_model, cap_check, timer):
            for i, sec in enumerate((10.0, 20.0)):
                timer.record.append({"epoch": i + 1, **{k: hist[k][i] for k in P.HISTORY_KEYS}, "seconds_total": sec,
                                     "seconds_validation": 1.0, "cpu_seconds": sec})
            clock[0] += 30.0 + 7.0
            card = {"arch": network, "train_seed": seed, "n_fitting": len(data["y_fit"]), "n_dev": len(data["y_dev"]), "max_epochs": 3,
                    "epochs_run": 2, "restored_epoch": 2, "restored_by": "end of epochs (set_weights)", "device": "CPU",
                    "visible_devices": ["CPU"], "lr": 0.001, "batch": 16, "early_stopping_patience": 4, "scheduler": "s",
                    "class_weight": {}, "train_seconds": 30.0, "checkpoint": "none saved (trial)", "history": hist}
            return card, tiny
        rec = {"epochs": [], "inference": {}}
        with injected(p={"TRAINER": fake}), mock.patch.object(P.time, "monotonic", lambda: clock[0]):
            budget = SR.Budget(60)
            with contextlib.redirect_stdout(io.StringIO()):
                P.trial_network(self.cfg, self.pcfg, "LivenessNet", None, self.fit, self.dev, budget, rec)
        self.assertEqual(rec["training_overhead"]["seconds"], 7.0)
        self.assertEqual(P.network_rates(rec, 32, 16)["overhead_seconds_per_training"], 7.0)

    def test_trainer_matches_the_original_trainer(self):
        """Stessa storia del trainer originale (3 epoche, stesso seed, stessi dati): la callback dei tempi e la copia non
        cambiano il training."""
        tmp = self.w.tmp / "orig_trainer"
        tmp.mkdir(exist_ok=True)
        self.addCleanup(shutil.rmtree, tmp, True)
        arch = types.SimpleNamespace(create_model=lambda name: types.SimpleNamespace(get_model=lambda: AM.build(name, SHAPE, ref_arch())))
        data = {"X_fit": self.fit["X"], "y_fit": self.fit["y"], "X_dev": self.dev["X"], "y_dev": self.dev["y"]}
        with contextlib.redirect_stdout(io.StringIO()):
            orig = FR.default_trainer(data, P.trial_training(self.cfg, self.pcfg), tmp / "m.keras", 42, arch, lambda s: None)
        self.assertEqual(orig["history"], self.history)

    def test_errors_before_and_during_the_trial(self):
        rc, out, err = run_main(self.w.argv("trial", out=self.out)[:-1])
        self.assertEqual(rc, 1)
        self.assertIn("--i-am-authorized", err)
        rc, out, err, run = self.go(REPO_STATE=lambda root: {"commit": "abcdef0", "dirty": True})
        self.assertEqual(rc, 1)
        self.assertIn("not clean", err)
        self.assertFalse(self.out.exists())
        rc, out, err, run = self.go(self.w.argv("trial", out=ROOT / "scripts" / "archcmp_should_not_exist"))
        self.assertEqual(rc, 1)
        self.assertIn("refusing the run folder", err)
        self.assertFalse((ROOT / "scripts" / "archcmp_should_not_exist").exists())
        rc, out, err, run = self.go(self.w.argv("trial", out=self.out, date="2026111"))
        self.assertIn("--date must be YYYYMMDD", err)
        doc = self.w.doc()
        doc["code"]["scripts/pad_arch_models.py"] = "7" * 64
        bad = self.w.write_cfg(doc, self.w.tmp / "badcode.json")
        rc, out, err, run = self.go(self.w.argv("trial", out=self.out, cfg=bad))
        self.assertEqual(rc, 1)
        self.assertIn("differ from the SHA-256 pinned", err)
        self.assertTrue((run / "status.json").is_file())
        self.assert_final_status(run, "FAIL")
        shutil.rmtree(self.out)
        doc["code"]["scripts/pad_arch_models.py"] = "x"
        bad = self.w.write_cfg(doc, self.w.tmp / "badcode.json")
        rc, out, err, run = self.go(self.w.argv("trial", out=self.out, cfg=bad))
        self.assertEqual((rc, self.out.exists()), (1, False))

        def broken(*a, **k):
            raise FR.Failure("synthetic trainer failure")
        rc, out, err, run = self.go(TRAINER=broken)
        self.assertEqual(rc, 1)
        st = self.assert_final_status(run, "FAIL")
        self.assertIn("synthetic trainer failure", st["reason"])
        self.assertEqual(sorted(os.listdir(run)), ["run.log", "status.json"])
        self.assertIn("RESULT: FAIL (trial)", (run / "run.log").read_text())

    def test_personal_text_fails_and_removes_the_outputs(self):
        home = str(Path.home())
        rc, out, err, run = self.go(MACHINE=lambda: {**P.machine(), "processor": home})
        self.assertEqual(rc, 1)
        st = self.assert_final_status(run, "FAIL")
        self.assertIn("trial.json refused", st["reason"])
        self.assertEqual(sorted(os.listdir(run)), ["run.log", "status.json"])
        shutil.rmtree(self.out)

        def chatty(*a, **k):
            print(f"working in {home}")
            return P.trial_trainer(*a, **k)
        rc, out, err, run = self.go(TRAINER=chatty)
        self.assertEqual(rc, 1)
        st = self.assert_final_status(run, "FAIL")
        self.assertIn("run.log holds text", st["reason"])
        self.assertEqual(sorted(os.listdir(run)), ["run.log", "status.json"])

    def test_same_command_after_a_pass_with_a_broken_input(self):
        rc, out, err, run = self.go()
        self.assertEqual(rc, 0, err[-3000:])
        frames = self.w.frames_file
        saved = frames.read_bytes()
        try:
            frames.write_bytes(saved[:-1] + bytes([saved[-1] ^ 1]))
            rc, out, err, run2 = self.go(self.w.argv("trial", out=self.out, date="20261119"))
        finally:
            frames.write_bytes(saved)
        self.assertEqual(rc, 1)
        st = self.assert_final_status(self.out / f"20261119-ARCHTRIAL-{STATE['commit']}", "FAIL")
        self.assertIn("SHA-256", st["reason"])
        self.assertEqual(sorted(os.listdir(self.out / f"20261119-ARCHTRIAL-{STATE['commit']}")), ["run.log", "status.json"])
        self.assertEqual(json.loads((run / "status.json").read_text())["status"], "PASS")


class TestMOutputs(unittest.TestCase):
    def test_personal_text_refused(self):
        inp = FR.Inputs()
        self.assertEqual(P.output_problems("{\"a\": 1}", inp), [])
        self.assertTrue(P.output_problems(f"path {Path.home()}/x", inp))
        self.assertIn(str(Path.home()), P.personal_tokens())
        self.assertTrue(P.output_problems(f"root {ROOT}", inp))
        self.assertTrue(P.output_problems("image real/0001_00_00_01_0.jpg", inp))

    def test_machine_has_no_host_or_user(self):
        m = P.machine()
        self.assertEqual(sorted(m), ["architecture", "efficiency_cores", "logical_cpus", "os", "performance_cores", "processor", "ram_gb"])
        self.assertEqual(P.output_problems(json.dumps(m), FR.Inputs()), [])

    def test_rates_rule(self):
        rec = {"epochs": [{"seconds_total": 10.0, "seconds_validation": 2.0, "cpu_seconds": 30.0},
                          {"seconds_total": 12.0, "seconds_validation": 2.0, "cpu_seconds": 30.0}],
               "inference": {"fitting": {"frames": 100, "seconds": 1.0, "cpu_seconds": 2.0}, "dev": {"frames": 50, "seconds": 2.0, "cpu_seconds": 1.0}},
               "training_overhead": {"seconds": 4.0, "cpu_seconds": 6.0}}
        r = P.network_rates(rec, 200, 50)
        self.assertEqual(r["train_seconds_per_1000_fitting_frames_per_epoch"], np.mean([8 / 200 * 1000, 10 / 200 * 1000]))
        self.assertEqual(r["validation_seconds_per_1000_frames"], 2 / 50 * 1000)
        self.assertEqual(r["inference_seconds_per_1000_frames"], 3.0 / 150 * 1000)
        self.assertNotEqual(r["inference_seconds_per_1000_frames"], 1.0 / 100 * 1000)
        self.assertEqual(r["overhead_seconds_per_training"], 4.0)
        self.assertEqual(r["cpu_over_wall"], 69 / 29.0)
        r = P.network_rates({"epochs": [], "inference": {}}, 200, 50)
        self.assertEqual([v for k, v in r.items() if k != "epochs_measured"], [None] * 5)

    def test_cpu_time_is_user_plus_system(self):
        import posix
        from unittest import mock
        with mock.patch.object(P.os, "times", return_value=posix.times_result((1.5, 2.25, 0.0, 0.0, 0.0))):
            self.assertEqual(P.cpu_seconds(), 3.75)

    def test_personal_tokens(self):
        import getpass
        t = P.personal_tokens()
        self.assertIn(str(Path.home()), t)
        self.assertIn(str(ROOT), t)
        if len(getpass.getuser()) >= 3:
            self.assertIn(getpass.getuser(), t)


class ZZRepositoryUntouched(unittest.TestCase):
    def test_declared_skips_only(self):
        self.assertEqual(sorted(MISSING_INPUT), ["delivered configuration", "reference architecture", "tensorflow"])


def main() -> int:
    before = CFR.snapshot(ROOT / "results", ROOT / "export", ROOT / "delivery")
    argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a != "--verbose"] + (["-v"] if "--verbose" in sys.argv else [])
    try:
        result = unittest.main(argv=argv, exit=False, verbosity=1).result
    finally:
        if WORLD is not None:
            WORLD.close()
    after = CFR.snapshot(ROOT / "results", ROOT / "export", ROOT / "delivery")
    changed = before != after
    skipped = [f"{t.id()}: {r}" for t, r in result.skipped]
    print(f"tests run: {result.testsRun}, failures: {len(result.failures)}, errors: {len(result.errors)}, skipped: {len(skipped)}")
    for s in skipped:
        print(f"skipped {s}")
    if changed:
        print("ERROR: results/, export/ or delivery/ changed during the tests")
    ok = result.wasSuccessful() and not changed
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
