"""
Test di scripts/pad_anet_comparison.py (C5: configurazione, ingressi, matrice e piano, prova tecnica),
scripts/pad_anet_definitions.py (definizioni operative) e scripts/pad_anet_models.py (registro delle tre AttackNet),
offline.

Mondo sintetico in cartelle temporanee: il mondo della suite di C2 (check_pad_arch_comparison.World: manifest M2, cache
8×8, run delle celle, run dei fold, frame estratti, i due run accettati finti con la scheda B fold 1 seed 42 e la
configurazione della preregistrazione C2) più: la configurazione della PR #27 (copia di quella consegnata), il record
finto della prova C2, i file pubblici finti della valutazione C2 (48 checkpoint) e la configurazione C5, costruita con
delivery/anetcmp/make_config.doc sui file del mondo. Le definizioni sono provate su dati sintetici costruiti a mano
(frazioni esatte, NaN e infiniti, unità incomplete, soglia diversa da 0,5, intervallo che tocca lo zero, ID dei frame
scambiati a parità di soggetto e classe) e contro il codice congelato di C2 (pad_arch_definitions.primary_analysis:
contrasti, divari e intervalli uguali esattamente). Il trainer è confrontato con il sorgente di
pad_arch_comparison.trial_trainer (solo le sostituzioni dichiarate) e, con TensorFlow, nel comportamento (stessa
scheda e stessi pesi su storie finite). La prova tecnica gira davvero, su CPU, con LivenessNet e le tre AttackNet del
codice di riferimento su frame 8×8 (le AttackNet sono costruite a 256×256 solo per contarne i parametri). Nessuna
scrittura nel repository (controllo finale su results/, export/, delivery/).

Senza TensorFlow i test che costruiscono o addestrano un modello Keras sono saltati, con un motivo dichiarato in
MISSING_INPUT; gli altri girano. TMPDIR deve stare fuori dalla radice del repository.

  .venv/bin/python -I -B scripts/check_pad_anet_comparison.py [--verbose] [-f]
"""
import contextlib
import copy as cp
import hashlib
import importlib.util
import inspect
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_nuaa_folds_run as CFR  # noqa: E402
import check_pad_arch_comparison as CPC  # noqa: E402
import nuaa_bn_recalibration as BR  # noqa: E402
import nuaa_folds_run as FR  # noqa: E402
import nuaa_session_ablation_run as SR  # noqa: E402
import pad_anet_comparison as C  # noqa: E402
import pad_anet_definitions as DEF  # noqa: E402
import pad_anet_models as AN  # noqa: E402
import pad_arch_comparison as PAC  # noqa: E402
import pad_arch_definitions as D  # noqa: E402
import pad_arch_run as R  # noqa: E402

TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
REF_ARCH = ROOT / "src" / "livedetection" / "src" / "architectures.py"
MAKE_CONFIG = ROOT / "delivery" / "anetcmp" / "make_config.py"
DELIVERED_BN = ROOT / "delivery" / "bnrecal" / "nuaa_bn_recalibration.json"
MISSING_INPUT = {
    "tensorflow": "TensorFlow is not installed: the tests that build or train a Keras model are skipped",
    "reference architecture": "src/livedetection/src/architectures.py (the reference code) is not present",
    "delivered configuration": "delivery/anetcmp/pad_anet_comparison.json or the files it pins are not present",
}
STATE = {"commit": "abcdef0", "dirty": False}
DATE = time.strftime("%Y%m%d")                     # il run id della prova porta la data di oggi
SHAPE = CPC.SHAPE
RUN = f"{DATE}-ANETTRIAL-{STATE['commit']}"
C2_TRIAL_ID = "20261119-ARCHTRIAL-b8f7cca"
EVAL_ID = "20261121-ARCHEVAL-332fbdf"
# macchina e versioni fisse del mondo sintetico (un campo che il sistema non riporta diventa un testo): il confronto con
# la prova C2 non dipende dal sistema su cui gira la suite (Linux senza sysctl, Python senza TensorFlow)
FIXED_M = {k: (v if v is not None else (8.0 if k == "ram_gb" else "not reported")) for k, v in PAC.machine().items()}
FIXED_E = {k: (v if v is not None else "not reported") for k, v in FR.environment().items()}
C2_RUN_IDS = {"nuaa_patience_4": "20261120-ARCHRUN-p4-7faa49f", "nuaa_patience_12": "20261120-ARCHRUN-p12-ea7fdca"}

if (os.path.realpath(tempfile.gettempdir()) + os.sep).startswith(os.path.realpath(ROOT) + os.sep):
    raise SystemExit("TMPDIR is inside the repository root: set it outside (the synthetic world must never be written in the repository)")


def skip_missing(test: unittest.TestCase, what: str):
    """Unico modo di saltare un test: un ingresso mancante dichiarato in MISSING_INPUT."""
    test.skipTest(f"missing input: {MISSING_INPUT[what]}")


def need_tf(test):
    if not TF_AVAILABLE:
        skip_missing(test, "tensorflow")
    if not REF_ARCH.is_file():
        skip_missing(test, "reference architecture")
    tf_threads()


def tf_threads():
    """TensorFlow con i thread della configurazione (10 e 10) e la GPU nascosta prima di qualunque operazione, come fa il
    runner, una sola volta per processo: la prova poi reimposta gli stessi valori (nessun cambiamento, nessun errore)."""
    import tensorflow as tf
    if (tf.config.threading.get_intra_op_parallelism_threads(), tf.config.threading.get_inter_op_parallelism_threads()) != (10, 10):
        CPC.tf_with_threads()
        FR.hide_gpu(tf)                                     # come il runner: thread, poi GPU nascosta, prima di ogni operazione
    return tf


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_make_config():
    spec = importlib.util.spec_from_file_location("anetcmp_make_config", MAKE_CONFIG)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MC = load_make_config()


@contextlib.contextmanager
def injected(c=None, p=None, sr=None, fr=None, r=None):
    """Punti di iniezione di pad_anet_comparison (c), pad_arch_comparison (p), nuaa_session_ablation_run (sr),
    nuaa_folds_run (fr) e pad_arch_run (r)."""
    mods = ((C, c), (PAC, p), (SR, sr), (FR, fr), (R, r))
    saved = [(m, k, getattr(m, k)) for m, d in mods for k in (d or {})]
    try:
        for m, d in mods:
            for k, v in (d or {}).items():
                setattr(m, k, v)
        yield
    finally:
        for m, k, v in saved:
            setattr(m, k, v)


def run_main(argv) -> tuple:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = C.main(argv)
    return rc, out.getvalue(), err.getvalue()


def ref_arch():
    return FR.load_module("kuznetsov_architectures_c5_test", REF_ARCH)


# ============================================================================ mondo sintetico

def epochs_of(h: dict, n: int) -> list:
    return [{"epoch": i + 1, **{k: h[k][i] for k in C.HISTORY_KEYS}} for i in range(n)]


class World(CPC.World):
    """Il mondo di C2 più la configurazione della PR #27, la prova C2 finta, la valutazione C2 finta e la configurazione C5."""

    def build(self):
        super().build()
        shutil.copy2(DELIVERED_BN, self.base / "bnrecal.json")
        self.c5_path = self.base / "anetcmp.json"
        self.c2_history = dict(self.history)
        self.write_c5()

    def write_parents(self, history=None, card_over=None):
        super().write_parents(history, card_over)
        if hasattr(self, "c5_path"):
            self.c2_history = dict(self.history)
            self.write_c5()

    def write_c5(self, doc_over=None, results_over=None, status_over=None, c2_over=None):
        b = self.base
        pr_sha = sha(self.cfg_path.read_bytes())
        t = b / "c2trial"
        t.mkdir(exist_ok=True)
        tdoc = {"schema": PAC.TRIAL_SCHEMA, "run_id": C2_TRIAL_ID, "status": "PASS", "commit": "b8f7cca", "config_sha256": pr_sha,
                "unit": dict(PAC.TRIAL_UNIT), "machine": dict(FIXED_M), "environment": dict(FIXED_E),
                "threads": {"requested": {"intra_op": 10, "inter_op": 10}, "effective": {"intra_op": 10, "inter_op": 10}, "set_by_this_runner": True,
                            "environment": {"TF_NUM_INTRAOP_THREADS": None, "TF_NUM_INTEROP_THREADS": None}, "logical_cpus": 4},
                "networks": {"LivenessNet": {"epochs": epochs_of(self.c2_history, 3)}, "ResNet-18": {"epochs": epochs_of(self.c2_history, 3)}},
                **(c2_over or {})}
        (t / "trial.json").write_text(json.dumps(tdoc, indent=1) + "\n", encoding="utf-8")
        (t / "status.json").write_text(json.dumps({"status": "PASS", "run_id": C2_TRIAL_ID, "trial_json_sha256": sha((t / "trial.json").read_bytes())}) + "\n",
                                       encoding="utf-8")
        e = b / "eval"
        (e / "public").mkdir(parents=True, exist_ok=True)
        models = [{"run": run, "position": 0, "network": n, "fold": f, "seed": s, "weights_sha256": sha(f"{run}{n}{f}{s}".encode())}
                  for run in C.C2_RUNS for f in (1, 2, 3, 4) for s in (42, 43, 44) for n in ("LivenessNet", "ResNet-18")]
        res = {"run_id": EVAL_ID, "status": "PASS", "models": models, **(results_over or {})}
        st = {"run_id": EVAL_ID, "status": "PASS", "official": True, "restricted_sha256sums_sha256": "d" * 64,
              "runs": {k: {"run_id": v} for k, v in C2_RUN_IDS.items()}, **(status_over or {})}
        (e / "public" / "results.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        (e / "status.json").write_text(json.dumps(st, indent=1) + "\n", encoding="utf-8")
        pr = json.loads(self.cfg_path.read_text(encoding="utf-8"))
        with injected(fr={"FRAME_SHAPE": SHAPE}):
            doc = MC.doc(pr=pr, ev_status=st, data_sha=lambda p: sha((b / p).read_bytes()), code_sha=lambda p: sha((ROOT / p).read_bytes()),
                         PREREG="archcmp.json", BNCFG="bnrecal.json", C2_TRIAL="c2trial", EVAL="eval")
        doc["trial"]["estimate_minutes"], doc["trial"]["upper_estimate_minutes"] = 1, 2
        for k, v in (doc_over or {}).items():
            doc[k] = v
        self.write_doc(doc)
        return doc

    def write_doc(self, doc: dict) -> None:
        self.c5_path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")

    def c5_doc(self) -> dict:
        return json.loads(self.c5_path.read_text(encoding="utf-8"))

    def argv(self, cmd: str, *extra, out=None, ck=None, attempt=1, cfg=None) -> list:
        a = [cmd, "--config", str(cfg or self.c5_path), "--input-root", str(self.base)]
        if cmd == "trial":
            a += ["--out-root", str(out or self.out), "--checkpoint-root", str(ck or (self.tmp / "ckpt")), "--attempt", str(attempt), "--i-am-authorized"]
        return a + list(extra)


def hooks(w: World, **c) -> dict:
    """Le iniezioni di base: stato git, forma dei frame 8×8, SHA-256 delle configurazioni dello schema e delle celle del
    mondo, il codice di riferimento al posto del file finto del mondo, memoria di picco fissa (un numero fra gli
    identificativi sintetici dei soggetti farebbe scattare la scansione delle uscite)."""
    base = CPC.hooks(w)
    return {"c": {"REPO_STATE": lambda root: dict(STATE), "ARCH_LOADER": lambda path: ref_arch(), **c},
            "p": base["p"], "sr": {**base["sr"], "peak_rss_mb": lambda: 2048.0}, "fr": base["fr"]}


WORLD = None


def world() -> World:
    global WORLD
    if WORLD is None:
        WORLD = World()
    return WORLD


def load(w: World = None):
    w = w or world()
    with injected(**hooks(w)):
        return C.load_config(w.c5_path, w.base)


# ============================================================================ dati sintetici delle definizioni

def frames(fold: int, subjects=4):
    """(soggetto, etichetta) dei frame di test di un fold: per ogni soggetto 3 bona fide e 2 attacchi (classi di numerosità
    diversa, così uno scambio fra i denominatori si vede)."""
    return [(f"s{fold}{i}", lab) for i in range(subjects) for lab, per in ((0, 3), (1, 2)) for _ in range(per)]


def unit_raw(fold, seed, scores: dict, thr=None, ids=None):
    fr = frames(fold)
    y = [l for _, l in fr]
    sub = [s for s, _ in fr]
    ids = ids or [f"f{fold}_{i}" for i in range(len(fr))]
    return {"unit": ("NUAA", fold, seed), "networks": {n: {"y": list(y), "s": list(s), "subject": list(sub), "threshold": (thr or {}).get(n, 0.5),
                                                          "sample_id": list(ids)} for n, s in scores.items()}}


def scores_for(fold, seed, state, nets=DEF.NETWORKS):
    """Punteggi sintetici deterministici, diversi per rete, unità e stato."""
    out = {}
    n = len(frames(fold))
    for j, net in enumerate(nets):
        rng = np.random.default_rng(1000 * fold + 10 * seed + 3 * j + (7 if state == "re-estimated" else 0))
        out[net] = list(rng.random(n))
    return out


def conditions(nets=DEF.NETWORKS, drop=None, thr=None):
    """Le due condizioni primarie con le reti date; drop: {(rete, fold, seed, stato)} da togliere."""
    out = {}
    for st in DEF.BN_STATES:
        units = []
        for f in DEF.FOLDS:
            for s in DEF.PLAN_SEEDS:
                sc = scores_for(f, s, st, nets)
                for (n, ff, ss, sst) in (drop or ()):
                    if (ff, ss, sst) == (f, s, st):
                        sc.pop(n, None)
                units.append(unit_raw(f, s, sc, thr))
        out[(4, st, "fixed_0.5")] = units
    return out


def exact_acer(y, s, t=0.5):
    y, s = np.asarray(y), np.asarray(s)
    d = s > t
    ap = Fraction(int(np.sum((y == 1) & ~d)), int(np.sum(y == 1)))
    bp = Fraction(int(np.sum((y == 0) & d)), int(np.sum(y == 0)))
    return (ap + bp) / 2


# ============================================================================ registro

class TestARegistry(unittest.TestCase):
    def test_counts_at_256_are_those_of_the_plan(self):
        self.assertEqual(AN.expected_counts("AttackNetV1"), {"trainable": 33_588_194, "non_trainable": 544, "batchnorm_layers": 5})
        self.assertEqual(AN.expected_counts("AttackNetV2_1"), {"trainable": 33_588_194, "non_trainable": 544, "batchnorm_layers": 5})
        self.assertEqual(AN.expected_counts("AttackNetV2_2"), {"trainable": 16_806_274, "non_trainable": 448, "batchnorm_layers": 5})

    def test_batchnorm_layers_channels_and_counts(self):
        for n, ch in (("AttackNetV1", (16, 32, 32, 64, 128)), ("AttackNetV2_1", (16, 32, 32, 64, 128)), ("AttackNetV2_2", (16, 16, 32, 32, 128))):
            bns = AN.batchnorm_layers(n)
            self.assertEqual([b["name"] for b in bns], ["bn1_1", "bn1_final", "bn2_1", "bn2_final", "bn_dense"])
            self.assertEqual(tuple(b["channels"] for b in bns), ch)
            self.assertEqual(AN.expected_recalibration_counts(n, 1400),
                             {"bn1_1": 91_750_400, "bn1_final": 91_750_400, "bn2_1": 22_937_600, "bn2_final": 22_937_600, "bn_dense": 1_400})
            self.assertEqual(AN.bn_order(n), [b["name"] for b in bns])

    def test_differences_between_the_three_networks(self):
        t1, t21, t22 = (AN.topology(n) for n in AN.NETWORKS)
        e1, e21, e22 = (AN.expected_extras(n) for n in AN.NETWORKS)
        self.assertEqual(t1["concat1"][0], "Concatenate")
        self.assertEqual(t1["concat1"][1], ("conv1_1", "leaky_relu1_3"))
        self.assertEqual(t22["add1"][0], "Add")
        self.assertNotIn("concat1", t22)
        self.assertEqual(t1["dense1"][2], (128, "tanh", "GlorotUniform"))
        self.assertEqual(t21["dense1_linear"][2], (128, "linear", "GlorotUniform"))
        self.assertEqual(t21["tanh_activation"], ("Activation", ("dense1_linear",), ("tanh",)))
        self.assertEqual((e1["dropout1"], e21["dropout1"], e22["dropout1"]), (("rate", 0.25), ("rate", 0.5), ("rate", 0.25)))
        self.assertEqual(e21["dropout_dense"], ("rate", 0.5))
        self.assertTrue(all(v == ("regularizers", (None, None, None)) for n in AN.NETWORKS for v in AN.expected_extras(n).values() if v[0] == "regularizers"))
        # V1 e V2_1: stessi parametri e stesse BN, tre differenze (dropout delle fasi conv, testa, nome)
        diff = sorted(set(t1) ^ set(t21))
        self.assertEqual(diff, ["dense1", "dense1_linear", "tanh_activation"])

    def test_refusals(self):
        with self.assertRaises(AN.ModelProblem):
            AN.expected_counts("LivenessNet")
        with self.assertRaises(AN.ModelProblem):
            AN.layer_spec("AttackNetV1", (10, 8, 3))
        with self.assertRaises(AN.ModelProblem):
            AN.build("AttackNetV1", (8, 8, 3), None)


class TestBModels(unittest.TestCase):
    def setUp(self):
        need_tf(self)
        self.arch = ref_arch()

    def test_built_models_match_the_registry(self):
        for shape in ((8, 8, 3), (256, 256, 3)):
            for n in AN.NETWORKS:
                m = AN.build(n, shape, self.arch)
                got = AN.check_model(m, n, shape, AN.expected_counts(n, shape))
                self.assertEqual(got["moving_values"], AN.expected_counts(n, shape)["non_trainable"])

    def test_check_model_refusals(self):
        shape = (8, 8, 3)
        exp = AN.expected_counts("AttackNetV1", shape)
        cases = {"l2 regularizer (BaseArchitecture default)": self.arch.AttackNetV1(shape, 0.5, 0.01).get_model(),
                 "other dropout": self.arch.create_model("AttackNetV1", input_shape=shape, dropout_rate=0.4).get_model(),
                 "another network": self.arch.create_model("AttackNetV2_1", input_shape=shape).get_model()}
        for what, m in cases.items():
            with self.subTest(what), self.assertRaises(AN.ModelProblem):
                AN.check_model(m, "AttackNetV1", shape, exp)
        with self.assertRaises(AN.ModelProblem):
            AN.check_model(AN.build("AttackNetV1", shape, self.arch), "AttackNetV1", shape, {**exp, "trainable": exp["trainable"] + 1})

    def test_topology_differs_with_the_same_names_and_counts(self):
        """concat1 con gli ingressi scambiati: nomi, conteggi e attributi uguali, solo la topologia diversa."""
        import tensorflow as tf
        shape = (8, 8, 3)
        m = AN.build("AttackNetV1", shape, self.arch)
        conf = m.get_config()
        for l in conf["layers"]:
            if l["name"] == "concat1":
                l["inbound_nodes"][0] = list(reversed(l["inbound_nodes"][0]))
        m2 = tf.keras.Model.from_config(conf)
        with self.assertRaisesRegex(AN.ModelProblem, "topology differs from the registry at concat1"):
            AN.check_model(m2, "AttackNetV1", shape, AN.expected_counts("AttackNetV1", shape))

    def test_v2_2_has_no_regularizer_even_with_l2(self):
        shape = (8, 8, 3)
        m = self.arch.AttackNetV2_2(shape, 0.5, 0.01).get_model()
        self.assertEqual(AN.check_model(m, "AttackNetV2_2", shape, AN.expected_counts("AttackNetV2_2", shape))["batchnorm_layers"], 5)


# ============================================================================ definizioni

class TestCNonFinite(unittest.TestCase):
    def test_finite_argmin(self):
        nan, inf = float("nan"), float("inf")
        self.assertEqual(DEF.finite_argmin([nan, 0.5, 0.4, 0.4]), 2)
        self.assertIsNone(DEF.finite_argmin([nan, nan]))
        self.assertEqual(DEF.finite_argmin([inf, 0.9, nan]), 1)
        self.assertEqual(int(np.argmin([nan, 0.5])), 0)          # perché np.argmin non basta

    def test_nan_loss_at_epoch_1(self):
        h = {"loss": [float("nan"), 0.5], "accuracy": [0.5, 0.6], "val_loss": [0.7, 0.6], "val_accuracy": [0.5, 0.6], "lr": [1e-3, 1e-3]}
        o = DEF.training_outcome(h, True)
        self.assertTrue(o["evaluable"])
        self.assertEqual(o["non_finite"], [["loss", 1]])
        self.assertEqual(o["restored_epoch"], 2)

    def test_nan_val_loss_after_a_finite_epoch(self):
        h = {"loss": [0.6, 0.5], "accuracy": [0.5, 0.6], "val_loss": [0.7, float("nan")], "val_accuracy": [0.5, float("nan")], "lr": [1e-3, 1e-3]}
        o = DEF.training_outcome(h, True)
        self.assertTrue(o["evaluable"])
        self.assertEqual(o["restored_epoch"], 1)
        self.assertEqual(o["non_finite"], [["val_loss", 2], ["val_accuracy", 2]])

    def test_no_finite_val_loss(self):
        h = {"loss": [float("nan")], "val_loss": [float("inf")]}
        o = DEF.training_outcome(h, None)
        self.assertFalse(o["evaluable"])
        self.assertEqual(o["reason"], "not evaluable: no finite val_loss")

    def test_restored_weights_not_finite(self):
        h = {"loss": [0.5], "val_loss": [0.5]}
        o = DEF.training_outcome(h, DEF.arrays_finite([np.array([1.0, np.nan])]))
        self.assertFalse(o["evaluable"])
        self.assertEqual(o["reason"], "not evaluable: restored weights not finite")
        with self.assertRaises(FR.Failure):
            DEF.training_outcome(h, None)

    def test_dev_not_finite(self):
        ok = {"p": np.full((2, 2), 0.5), "z": np.zeros((2, 2))}
        bad = {"p": np.array([[0.5, 0.5], [np.nan, 0.5]]), "z": np.zeros((2, 2))}
        self.assertTrue(DEF.dev_outcome({"ordinary": ok, "re-estimated": ok}, True)["evaluable"])
        self.assertEqual(DEF.dev_outcome({"ordinary": ok, "re-estimated": bad}, True)["reason"], "not evaluable: non-finite dev scores or logits (re-estimated BN)")
        self.assertEqual(DEF.dev_outcome({"ordinary": ok}, False)["reason"], "not evaluable: non-finite re-estimated statistics")


class TestDPrepare(unittest.TestCase):
    def base(self):
        return [unit_raw(1, 42, scores_for(1, 42, "ordinary")), unit_raw(1, 43, scores_for(1, 43, "ordinary"))]

    def test_valid(self):
        pus = DEF.prepare(self.base(), DEF.REUSED)
        self.assertEqual(pus[0]["networks"], DEF.NETWORKS)
        self.assertEqual(pus[0]["n_frames"], 20)

    def test_frame_ids_swapped_for_the_same_subject_and_class(self):
        u = self.base()
        ids = list(u[0]["networks"]["AttackNetV1"]["sample_id"])
        ids[0], ids[1] = ids[1], ids[0]                      # stessi soggetto e classe
        u[0]["networks"]["AttackNetV1"]["sample_id"] = ids
        with self.assertRaisesRegex(FR.Failure, "same frame identifiers in the same order"):
            DEF.prepare(u)

    def test_labels_differ(self):
        u = self.base()
        u[0]["networks"]["AttackNetV2_2"]["y"][0] = 1
        with self.assertRaisesRegex(FR.Failure, "not scored on the same test frames"):
            DEF.prepare(u)

    def test_units_of_a_fold_on_other_frames(self):
        u = self.base()
        for n in u[1]["networks"]:
            u[1]["networks"][n]["sample_id"] = [f"x{i}" for i in range(20)]
        with self.assertRaisesRegex(FR.Failure, "two units of"):
            DEF.prepare(u)

    def test_identifier_in_two_folds(self):
        a = unit_raw(1, 42, scores_for(1, 42, "ordinary"))
        b = unit_raw(2, 42, scores_for(2, 42, "ordinary"), ids=[f"f1_{i}" for i in range(20)])
        with self.assertRaisesRegex(FR.Failure, "also appears"):
            DEF.prepare([a, b])

    def test_other_refusals(self):
        u = self.base()
        with self.assertRaisesRegex(FR.Failure, "malformed or duplicated"):
            DEF.prepare([u[0], u[0]])
        bad = cp.deepcopy(u[0])
        bad["unit"] = ("NUAA", 1, 45)
        with self.assertRaisesRegex(FR.Failure, "seeds of the plan"):
            DEF.prepare([bad])
        bad = cp.deepcopy(u[0])
        del bad["networks"]["ResNet-18"]
        with self.assertRaisesRegex(FR.Failure, "lacks ResNet-18"):
            DEF.prepare([bad], DEF.REUSED)
        bad = cp.deepcopy(u[0])
        bad["networks"]["AttackNetV3"] = bad["networks"]["AttackNetV1"]
        with self.assertRaisesRegex(FR.Failure, "networks must be among"):
            DEF.prepare([bad])
        bad = cp.deepcopy(u[0])
        bad["networks"]["AttackNetV1"]["extra"] = 1
        with self.assertRaisesRegex(FR.Failure, "exactly the fields"):
            DEF.prepare([bad])

    def test_subject_in_two_folds(self):
        a = unit_raw(1, 42, scores_for(1, 42, "ordinary"))
        b = unit_raw(2, 42, scores_for(2, 42, "ordinary"))
        for n in b["networks"]:
            b["networks"][n]["subject"] = list(a["networks"][n]["subject"])
        with self.assertRaisesRegex(FR.Failure, "a test subject appears in two folds"):
            DEF.pooled(DEF.prepare([a, b]))

    def test_states_on_other_identifiers(self):
        a = DEF.prepare([unit_raw(1, 42, scores_for(1, 42, "ordinary"))])
        ids = [f"f1_{i}" for i in range(20)]
        ids[0], ids[1] = ids[1], ids[0]
        b = DEF.prepare([unit_raw(1, 42, scores_for(1, 42, "re-estimated"), ids=ids)])
        with self.assertRaisesRegex(FR.Failure, "same frame identifiers"):
            DEF.pair_states(a, b)


class TestEQuantities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conds = conditions()
        cls.res = DEF.primary_analysis(cls.conds)

    def test_variation_and_contrast_are_exact_fractions(self):
        rows = self.res["rows"]
        for i, (f, s) in enumerate((f, s) for f in DEF.FOLDS for s in DEF.PLAN_SEEDS):
            sc = {st: scores_for(f, s, st) for st in DEF.BN_STATES}
            y = [l for _, l in frames(f)]
            acer = {(n, st): exact_acer(y, sc[st][n]) for n in DEF.NETWORKS for st in DEF.BN_STATES}
            for n in DEF.NETWORKS:
                v = rows[i][f"variation/{n}"]["acer"]
                self.assertIsInstance(v, Fraction)
                self.assertEqual(v, acer[(n, "re-estimated")] - acer[(n, "ordinary")])
            for k in DEF.ATTACKNETS:
                self.assertEqual(rows[i][f"contrast/{k}"]["acer"],
                                 (acer[(k, "re-estimated")] - acer[(k, "ordinary")]) - (acer[("LivenessNet", "re-estimated")] - acer[("LivenessNet", "ordinary")]))
        for k in DEF.ATTACKNETS:
            want = sum((rows[i][f"contrast/{k}"]["acer"] for i in range(12)), Fraction(0)) / 12
            self.assertEqual(self.res["contrasts"][k]["mean"]["acer"], float(want))

    def test_cross_check_with_the_frozen_c2_code(self):
        """C_k, divari e V_n − V_R uguali esattamente a pad_arch_definitions.primary_analysis con la rete al posto di
        «ResNet-18» (confronto solo interno: lì l'etichetta sarebbe falsa)."""
        def as_c2(minuend, subtrahend):
            return {c: [{"unit": u["unit"], "networks": {"ResNet-18": u["networks"][minuend], "LivenessNet": u["networks"][subtrahend]}}
                        for u in sorted(units, key=lambda u: (u["unit"][1], u["unit"][2]))] for c, units in self.conds.items()}
        for k in DEF.ATTACKNETS:
            d = D.primary_analysis(as_c2(k, "LivenessNet"))
            mine = self.res["contrasts"][k]
            self.assertEqual(d["contrast"]["mean"], mine["mean"])
            self.assertEqual(d["contrast"]["interval"], mine["interval"])
            for seed in DEF.PLAN_SEEDS:
                self.assertEqual(d["per_seed"][seed]["contrast"]["mean"], self.res["per_seed"][str(seed)]["contrasts"][k]["mean"])
                self.assertEqual(d["per_seed"][seed]["contrast"]["interval"], self.res["per_seed"][str(seed)]["contrasts"][k]["interval"])
        for n in DEF.OURS:
            d = D.primary_analysis(as_c2(n, "ResNet-18")) if n != "LivenessNet" else None
            if d is None:
                continue
            mine = self.res["descriptive_vs_resnet18"][n]
            self.assertEqual(d["gap_ordinary"]["mean_delta"], mine["gap_ordinary"]["mean"])
            self.assertEqual(d["gap_ordinary"]["interval"], mine["gap_ordinary"]["interval"])
            self.assertEqual(d["gap_re_estimated"]["interval"], mine["gap_re_estimated"]["interval"])
            self.assertEqual(d["contrast"]["mean"], mine["variation_difference"]["mean"])
            self.assertEqual(d["contrast"]["interval"], mine["variation_difference"]["interval"])

    def test_livenessnet_against_resnet18_is_the_opposite_of_c2(self):
        units = {c: [{"unit": u["unit"], "networks": {n: u["networks"][n] for n in DEF.REUSED}} for u in v] for c, v in self.conds.items()}
        d = D.primary_analysis(units)
        mine = self.res["descriptive_vs_resnet18"]["LivenessNet"]
        for k in D.METRICS:
            self.assertEqual(mine["gap_ordinary"]["mean"][k], -d["gap_ordinary"]["mean_delta"][k])
            self.assertEqual(mine["variation_difference"]["mean"][k], -d["contrast"]["mean"][k])
            self.assertEqual(mine["variation_difference"]["interval"][k]["valid_resamples"], d["contrast"]["interval"][k]["valid_resamples"])
            self.assertEqual(mine["variation_difference"]["interval"][k]["low"], -d["contrast"]["interval"][k]["high"])
            self.assertEqual(mine["variation_difference"]["interval"][k]["high"], -d["contrast"]["interval"][k]["low"])
            for st, g in (("ordinary", "gap_ordinary"), ("re_estimated", "gap_re_estimated")):
                self.assertEqual(mine[f"gap_{st}"]["mean"][k], -d[g]["mean_delta"][k])
                self.assertEqual(mine[f"gap_{st}"]["interval"][k]["low"], -d[g]["interval"][k]["high"])
                self.assertEqual(mine[f"gap_{st}"]["interval"][k]["high"], -d[g]["interval"][k]["low"])
                self.assertEqual(mine[f"gap_{st}"]["interval"][k]["valid_resamples"], d[g]["interval"][k]["valid_resamples"])
        self.assertEqual((mine["gap_ordinary"]["counts"]["below_zero"], mine["gap_ordinary"]["counts"]["above_zero"]),
                         (d["gap_ordinary"]["acer_counts"]["livenessnet_lower"], d["gap_ordinary"]["acer_counts"]["resnet18_lower"]))

    def test_independent_bootstrap_of_one_variation(self):
        """Il bootstrap di V_n ricalcolato qui in modo indipendente: soggetti uniti, default_rng(0), integers(0, n, n)."""
        net = "AttackNetV2_1"
        subs = sorted({s for f in DEF.FOLDS for s, _ in frames(f)})
        rng = np.random.default_rng(0)
        vals = []
        for _ in range(1000):
            pick = rng.integers(0, len(subs), len(subs))
            w = {s: int(np.sum(pick == i)) for i, s in enumerate(subs)}
            per, ok = [], True
            for f in DEF.FOLDS:
                for s in DEF.PLAN_SEEDS:
                    acc = {}
                    for st in DEF.BN_STATES:
                        sc = np.asarray(scores_for(f, s, st)[net])
                        fr = frames(f)
                        na = nb = a = r = 0
                        for (sub, lab), x in zip(fr, sc):
                            m = w[sub]
                            if lab == 1:
                                na += m
                                a += m * (x <= 0.5)
                            else:
                                nb += m
                                r += m * (x > 0.5)
                        if na == 0 or nb == 0:
                            ok = False
                            break
                        acc[st] = (Fraction(int(a), na) + Fraction(int(r), nb)) / 2
                    if not ok:
                        break
                    per.append(acc["re-estimated"] - acc["ordinary"])
                if not ok:
                    break
            if ok:
                vals.append(float(np.mean([float(v) for v in per])))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        got = self.res["variations"][net]["interval"]["acer"]
        self.assertEqual((got["low"], got["high"], got["valid_resamples"]), (float(lo), float(hi), len(vals)))

    def test_reading_conclusion_and_individual_intervals(self):
        for k in DEF.ATTACKNETS:
            c = self.res["contrasts"][k]
            self.assertIn(c["reading"], DEF.reading_names(k))
            self.assertEqual(c["conclusion"], DEF.conclusion(k, c["reading"]))
        self.assertEqual(self.res["intervals"], DEF.INTERVALS_NOTE)
        self.assertIn("individual", DEF.INTERVALS_NOTE)
        self.assertEqual(DEF.contrast_reading("AttackNetV1", {"low": 0.0, "high": 0.1}), "unresolved")
        self.assertEqual(DEF.contrast_reading("AttackNetV1", {"low": -0.1, "high": 0.0}), "unresolved")
        self.assertEqual(DEF.contrast_reading("AttackNetV1", {"low": None, "high": None}), "unresolved")
        self.assertEqual(DEF.contrast_reading("AttackNetV1", {"low": 0.01, "high": 0.1}), DEF.reading_names("AttackNetV1")[1])
        self.assertEqual(DEF.contrast_reading("AttackNetV1", {"low": -0.1, "high": -0.01}), DEF.reading_names("AttackNetV1")[0])
        with self.assertRaises(FR.Failure):
            DEF.contrast_reading("AttackNetV1", {"low": float("nan"), "high": 0.1})
        with self.assertRaises(FR.Failure):
            DEF.contrast_reading("LivenessNet", {"low": 0.1, "high": 0.2})
        for w in PAC.BANNED:
            for k in DEF.ATTACKNETS:
                for rd in DEF.reading_names(k):
                    self.assertNotIn(w, DEF.conclusion(k, rd).lower())
        self.assertIn("does not identify a cause", DEF.conclusion("AttackNetV1", DEF.reading_names("AttackNetV1")[0]))
        self.assertIn("algebraically lower for AttackNetV1 than for LivenessNet", DEF.conclusion("AttackNetV1", DEF.reading_names("AttackNetV1")[0]))
        self.assertIn("algebraically higher for AttackNetV2_2 than for LivenessNet", DEF.conclusion("AttackNetV2_2", DEF.reading_names("AttackNetV2_2")[1]))
        with self.assertRaises(FR.Failure):
            DEF.conclusion("AttackNetV1", DEF.reading_names("AttackNetV2_1")[0])
        self.assertIn("does not exclude it", DEF.conclusion("AttackNetV1", "unresolved"))

    def test_per_seed_apart(self):
        self.assertEqual(sorted(self.res["per_seed"]), ["42", "43", "44"])
        for s in self.res["per_seed"].values():
            self.assertEqual(s["contrasts"]["AttackNetV1"]["counts"]["units"], 4)

    def test_denominators_and_rankings(self):
        den = self.res["denominators"]["ordinary"]
        self.assertEqual((den["units"], den["subjects"], den["test_frames_summed_over_units"]), (12, 16, 12 * 20))
        self.assertEqual((den["attack_frames_summed_over_units"], den["bona_fide_frames_summed_over_units"]), (12 * 8, 12 * 12))
        self.assertEqual(den["per_fold"]["1"], {"frames": 20, "attack": 8, "bona_fide": 12, "subjects": 4})
        rk = self.res["rankings"]["ordinary"]
        self.assertEqual(len(rk["order"]), 5)
        self.assertEqual(sorted(sum(v.values()) for v in rk["unit_rank_counts"].values()), [12] * 5)

    def test_ties_share_the_lowest_rank(self):
        units = [unit_raw(1, 42, {n: [0.9 if l else 0.1 for _, l in frames(1)] for n in DEF.NETWORKS})]
        rk = DEF.rankings(DEF.prepare(units))
        self.assertEqual({e["rank"] for e in rk["order"]}, {1})
        self.assertEqual(rk["unit_rank_counts"]["ResNet-18"]["1"], 1)

    def test_percentage_points(self):
        self.assertEqual(DEF.pp(0.0998332546851536), 9.98)
        self.assertEqual(DEF.pp(Fraction(1, 3)), 33.33)
        self.assertIsNone(DEF.pp(None))
        self.assertEqual(DEF.pp_text(-0.16057264797151244), "-16.06")
        with self.assertRaises(FR.Failure):
            DEF.pp(float("nan"))


class TestFGuardian(unittest.TestCase):
    def test_threshold_not_05_for_one_network_stops(self):
        conds = conditions(thr={"AttackNetV2_2": 0.4})
        with self.assertRaisesRegex(FR.Failure, "effective threshold of AttackNetV2_2"):
            DEF.primary_analysis(conds)

    def test_reused_network_incomplete_stops(self):
        conds = conditions(drop={("ResNet-18", 2, 43, "re-estimated")})
        with self.assertRaisesRegex(FR.Failure, "ResNet-18 \\(reused from C2\\) has 11 of 12"):
            DEF.primary_analysis(conds)

    def test_attacknet_incomplete_not_computed_others_go_on(self):
        conds = conditions(drop={("AttackNetV1", 3, 44, "ordinary")})
        res = DEF.primary_analysis(conds, reasons={"AttackNetV1": "not evaluable: no finite val_loss"})
        self.assertFalse(res["guardian"]["AttackNetV1"]["computed"])
        self.assertEqual(res["guardian"]["AttackNetV1"]["text"],
                         "not computed (11 of 12 pairs evaluable in both BN states): not evaluable: no finite val_loss")
        self.assertEqual(res["contrasts"]["AttackNetV1"]["conclusion"],
                         "Not computed for AttackNetV1 (11 of 12 pairs evaluable in both BN states): not evaluable: no finite val_loss.")
        self.assertFalse(res["variations"]["AttackNetV1"]["computed"])
        self.assertEqual(res["contrasts"]["AttackNetV1"]["reading"], "not computed")
        self.assertFalse(res["descriptive_vs_resnet18"]["AttackNetV1"]["computed"])
        self.assertTrue(res["contrasts"]["AttackNetV2_1"]["computed"])
        full = DEF.primary_analysis(conditions())
        self.assertEqual(res["contrasts"]["AttackNetV2_1"]["interval"], full["contrasts"]["AttackNetV2_1"]["interval"])

    def test_missing_condition_and_other_keys(self):
        with self.assertRaisesRegex(FR.Failure, "is missing"):
            DEF.primary_analysis({(4, "ordinary", "fixed_0.5"): conditions()[(4, "ordinary", "fixed_0.5")]})
        with self.assertRaisesRegex(FR.Failure, "keyed by"):
            DEF.primary_analysis({(5, "ordinary", "fixed_0.5"): []})


class TestGOneClass(unittest.TestCase):
    def test_counts_for_five_networks(self):
        recs = [{"network": "AttackNetV1", "patience": 4, "bn_state": "ordinary", "result": D.one_class_run([0, 1], [0.1, 0.2])},
                {"network": "AttackNetV1", "patience": 4, "bn_state": "ordinary", "result": D.one_class_run([0, 1], [0.1, 0.9])},
                {"network": "AttackNetV1", "patience": 4, "bn_state": "ordinary", "result": D.one_class_run([1, 1], [0.1, 0.9])}]
        c = DEF.one_class_counts(recs)[0]
        self.assertEqual((c["one_class"], c["evaluable"], c["not_evaluable"], c["all_bona_fide"]), (1, 2, 1, 1))
        with self.assertRaises(FR.Failure):
            DEF.one_class_counts([{"network": "X", "patience": 4, "bn_state": "ordinary", "result": recs[0]["result"]}])

    def test_test_one_class(self):
        self.assertEqual(DEF.test_one_class([0, 1], [0.6, 0.7], 0.5)["class"], "attack")
        self.assertEqual(DEF.test_one_class([0, 1], [0.5, 0.5], 0.5)["class"], "bona fide")
        self.assertFalse(DEF.test_one_class([0, 1], [0.1, 0.7], 0.5)["one_class"])


# ============================================================================ trainer

class TestHTrainerCopy(unittest.TestCase):
    def test_copy_has_only_the_declared_substitutions(self):
        src = inspect.getsource(PAC.trial_trainer)
        for old, new in C.ANET_TRAINER_SUBSTITUTIONS:
            self.assertEqual(src.count(old), 1, old[:80])
            src = src.replace(old, new)
        self.assertEqual(src, inspect.getsource(C.anet_trainer))

    def test_names_of_the_copy(self):
        for n in ("hide_gpu", "training_callbacks", "check_scheduler", "fail"):
            self.assertIs(getattr(C, n), getattr(FR, n))
        for n in ("finite_argmin", "finite_or_none", "non_finite_entries"):
            self.assertIs(getattr(C, n), getattr(DEF, n))

    def test_recipe_of_the_trial(self):
        cfg, _, _, pcfg, _ = load()
        self.assertEqual(C.trial_training(cfg, pcfg), {**FR.FROZEN_TRAINING, "max_epochs": 3})


# ============================================================================ configurazione

def refused(test, mutate, pattern):
    w = world()
    doc = w.c5_doc()
    mutate(doc)
    w.write_doc(doc)
    try:
        with test.assertRaisesRegex(FR.Failure, pattern):
            load(w)
    finally:
        w.write_c5()


class TestIConfig(unittest.TestCase):
    def test_valid(self):
        cfg, s, cfg_pr, pcfg, bn = load()
        self.assertEqual(cfg["schema"], C.CONFIG_SCHEMA)
        self.assertEqual(cfg_pr["schema"], PAC.CONFIG_SCHEMA)
        self.assertIn("tolerances", bn)

    def test_refusals(self):
        cases = [
            ("schema", lambda d: d.update(schema="x"), "schema must be"),
            ("unknown key", lambda d: d.update(extra=1), "fields must be exactly"),
            ("test statement", lambda d: d.update(test="x"), "test must state exactly"),
            ("declaration phrase", lambda d: d["declarations"].update(post_hoc="an extension"), "post hoc"),
            ("not independent", lambda d: d["declarations"].update(not_independent="x"), "not an independent validation"),
            ("pre-registration pin", lambda d: d["preregistration"].update(sha256="0" * 64), "pre-registration configuration"),
            ("PR 27 pin", lambda d: d["bn_recalibration_config"].update(sha256="0" * 64), "configuration of PR #27"),
            ("architecture file", lambda d: d["architectures"].update(sha256="0" * 64), "reference architecture file pinned"),
            ("counts", lambda d: d["architectures"]["networks"]["AttackNetV1"].update(trainable_parameters=1), "differ from the arithmetic"),
            ("BN names", lambda d: d["architectures"]["networks"]["AttackNetV2_2"].update(batchnorm_names=["bn1_1"]), "batchnorm_names must be"),
            ("constructor", lambda d: d["architectures"]["networks"]["AttackNetV2_1"].update(constructor="get_all_architectures"), "constructor must be"),
            ("threads", lambda d: d["training"]["threads"].update(intra_op=8), "training.threads must be those"),
            ("trial overrides", lambda d: d["training"].update(trial_overrides={"max_epochs": 4}), "trial_overrides"),
            ("non-finite rule", lambda d: d["training"].update(non_finite_rule="x"), "not evaluable"),
            ("trial fold", lambda d: d["trial"].update(fold=2), "trial unit"),
            ("test role", lambda d: d["trial"].update(roles=["fitting", "dev", "test"]), "never the test"),
            ("trial networks order", lambda d: d["trial"].update(networks=list(reversed(AN.NETWORKS))), "trial.networks"),
            ("control", lambda d: d["trial"].update(control="ResNet-18"), "trial.control"),
            ("epochs", lambda d: d["trial"].update(epochs=2), "trial.epochs"),
            ("patience", lambda d: d["trial"].update(patience=12), "trial.patience"),
            ("estimator batch", lambda d: d["trial"].update(estimator_batch=8), "estimator_batch"),
            ("cap above 60", lambda d: d["trial"].update(wall_time_cap_minutes=61), "from 1 to 60"),
            ("upper estimate at the cap", lambda d: d["trial"].update(upper_estimate_minutes=60), "upper_estimate_minutes < wall_time_cap_minutes"),
            ("checkpoints inside", lambda d: d["trial"].update(checkpoints="saved in results"), "outside the repository"),
            ("control rule", lambda d: d["trial"].update(control_rule="the machine is compared"), "CONTROL_FAILED"),
            ("control rule without the machine", lambda d: d["trial"].update(control_rule="otherwise CONTROL_FAILED"), "machine and versions"),
            ("after outcome", lambda d: d["trial"].update(after_outcome="PASS, FAIL, CONTROL_FAILED, NOT_EVALUABLE reported"), "after_outcome"),
            ("attempts text", lambda d: d["trial"].update(attempts=""), "trial.attempts"),
            ("descriptive conditions", lambda d: d["definitions"].pop("descriptive_conditions"), "definitions needs exactly"),
            ("runs", lambda d: d["plan"].update(runs=list(reversed(C.run_names()))), "plan.runs"),
            ("seeds", lambda d: d["plan"].update(seeds=[42, 43]), "networks, patiences, folds, seeds"),
            ("reference cells", lambda d: d["plan"]["reference"].update(cells=36), "plan.reference.cells"),
            ("reference rule", lambda d: d["plan"]["reference"].update(rule="reused"), "not re-evaluated"),
            ("caps multiplier", lambda d: d["plan"]["caps"].update(wall_multiplier=0.9), "at least 1"),
            ("anomaly factor", lambda d: d["plan"]["caps"].update(anomaly_factor=0.5), "at least 1"),
            ("caps unit", lambda d: d["plan"]["caps"].update(unit="CPU hours"), "plan.caps.unit"),
            ("machine rule", lambda d: d["plan"].update(machine_rule="the same machine"), "same threads"),
            ("evaluation once", lambda d: d["plan"].update(evaluation="read"), "read once"),
            ("bootstrap", lambda d: d["definitions"]["bootstrap"].update(resamples=999), "recipe of the project"),
            ("conditions", lambda d: d["definitions"].update(conditions=d["definitions"]["conditions"][:4]), "8 conditions"),
            ("intervals", lambda d: d["definitions"]["primary"].update(intervals="95% intervals"), "individual intervals"),
            ("reading names", lambda d: d["definitions"]["primary"]["reading_names"].update(AttackNetV1=["a", "b", "c"]), "reading_names"),
            ("presentation", lambda d: d["definitions"].update(presentation="percent"), "percentage points"),
            ("primary patience", lambda d: d["definitions"]["primary"].update(patience=12), "definitions.primary.patience"),
            ("descriptive reading", lambda d: d["definitions"].update(descriptive_vs_resnet18="gaps"), "no reading"),
            ("causal wording", lambda d: d["definitions"].update(guardian="stops because of a threshold"), "causal or conclusive"),
            ("outputs", lambda d: d["outputs"].update(plan=["run_plan.csv"]), "outputs must be"),
            ("environment", lambda d: d.update(environment={"tensorflow": "2.16.0", "keras": "2.16.0"}), "environment must be"),
            ("choice not declared", lambda d: d["choices_of_the_author"].update(six_runs="six runs"), "a choice of the author"),
            ("choice missing", lambda d: d["choices_of_the_author"].pop("caps"), "choices_of_the_author needs exactly"),
            ("C2 run of the wrong patience", lambda d: d["c2_results"]["runs"].update(nuaa_patience_4=C2_RUN_IDS["nuaa_patience_12"]), "patience 4"),
            ("C2 trial id", lambda d: d["c2_trial"].update(run_id="x"), "ARCHTRIAL run id"),
        ]
        for name, mutate, pattern in cases:
            with self.subTest(name):
                refused(self, mutate, pattern)


class TestJInputs(unittest.TestCase):
    def go(self, frames=False):
        w = world()
        with injected(**hooks(w)):
            cfg, _, cfg_pr, pcfg, _ = C.load_config(w.c5_path, w.base)
            return C.load_inputs(cfg, cfg_pr, pcfg, w.base, frames=frames)

    def test_valid_with_and_without_frames(self):
        inp = self.go()
        self.assertEqual(len(inp.reference_cells), 48)
        self.assertEqual(inp.c2_trial["run_id"], C2_TRIAL_ID)
        inp = self.go(frames=True)
        self.assertTrue(inp.reference_card)
        self.assertEqual(inp.folds, {})

    def test_refusals(self):
        w = world()
        models = json.loads((w.base / "eval" / "public" / "results.json").read_text())["models"]
        cases = [
            ("C2 trial of another configuration", {"c2_over": {"config_sha256": "e" * 64}}, "another configuration"),
            ("C2 trial not PASS", {"c2_over": {"status": "INCOMPLETE"}}, "C2 trial"),
            ("evaluation not PASS", {"status_over": {"status": "FAIL"}}, "official PASS"),
            ("evaluation not official", {"status_over": {"official": False}}, "official PASS"),
            ("47 models", {"results_over": {"models": models[:47]}}, "must list 48"),
            ("duplicated model", {"results_over": {"models": models[:47] + models[:1]}}, "listed twice"),
            ("model of fold 5", {"results_over": {"models": [{**models[0], "fold": 5}] + models[1:]}}, "outside the plan"),
            ("model of an AttackNet", {"results_over": {"models": [{**models[0], "network": "AttackNetV1"}] + models[1:]}}, "outside the plan"),
            ("model without checkpoint SHA-256", {"results_over": {"models": [{**models[0], "weights_sha256": "x"}] + models[1:]}}, "outside the plan"),
        ]
        for name, over, pattern in cases:
            with self.subTest(name):
                try:
                    w.write_c5(**over)
                    with self.assertRaisesRegex(FR.Failure, pattern):
                        self.go()
                finally:
                    w.write_c5()

    def test_restricted_anchor_differs_from_the_configuration(self):
        w = world()
        doc = w.c5_doc()
        doc["c2_results"]["restricted_sha256sums_sha256"] = "e" * 64
        w.write_doc(doc)
        try:
            with self.assertRaisesRegex(FR.Failure, "restricted_sha256sums_sha256 of status.json differs"):
                self.go()
        finally:
            w.write_c5()

    def test_modified_c2_trial_after_the_configuration(self):
        """trial.json della prova C2 cambiato dopo la configurazione, con lo status.json aggiornato (coerente con il nuovo
        file): deve scattare il controllo degli SHA-256 fissati nella configurazione C5."""
        w = world()
        t = w.base / "c2trial"
        saved = {n: (t / n).read_bytes() for n in ("trial.json", "status.json")}
        try:
            (t / "trial.json").write_bytes(saved["trial.json"] + b" ")
            st = json.loads(saved["status.json"])
            st["trial_json_sha256"] = sha((t / "trial.json").read_bytes())
            (t / "status.json").write_text(json.dumps(st) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(FR.Failure, "differ from the SHA-256 of the configuration"):
                self.go()
        finally:
            for n, b in saved.items():
                (t / n).write_bytes(b)

    def test_check_inputs_command(self):
        w = world()
        with CPC.tensorflow_unloaded(), injected(**hooks(w)):
            rc, out, err = run_main(w.argv("check-inputs"))
        self.assertEqual(rc, 0, err)
        self.assertIn("RESULT: PASS (check-inputs)", out)
        self.assertIn("48 reused checkpoints", out)
        self.assertIn("test configuration, test hooks set: pad_anet_comparison.ARCH_LOADER", out)


# ============================================================================ piano

def fake_trial(w: World, folder: Path, status="PASS", over=None) -> dict:
    """Un trial.json finto di questo codice, con le misure necessarie al piano."""
    cfg, cfg_sha, _, _, _ = load(w)
    code = {"architecture": cfg["architectures"]["sha256"], **C.check_code(cfg)}

    def net(spe):
        """30 s per epoca, 15 s fuori dalle epoche, 70 s di fase 2 (10 per passo), 10 s d'inferenza per 1.000 frame; CPU =
        4 × tempo reale ovunque."""
        return {"epochs": [{"epoch": i + 1, "seconds_total": spe, "seconds_validation": 1.0, "cpu_seconds": 4 * spe} for i in range(3)],
                "inference": {r: {"frames": 100, "batch": 16, "seconds": 1.0, "cpu_seconds": 4.0} for r in C.TRIAL_ROLES},
                "steps": {k: {"seconds": 10.0, "cpu_seconds": 40.0} for k in C.PHASE2_STEPS},
                "training_overhead": {"seconds": 15.0, "cpu_seconds": 60.0}}
    doc = {"schema": C.TRIAL_SCHEMA, "run_id": RUN, "status": status, "config_sha256": cfg_sha, "code_sha256": code,
           "machine": {"processor": "p", "os": "o", "logical_cpus": 4, "ram_gb": 8.0},
           "networks": {"AttackNetV1": net(30.0), "AttackNetV2_1": net(30.0), "AttackNetV2_2": net(30.0)}, **(over or {})}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "trial.json").write_text(json.dumps(doc) + "\n", encoding="utf-8")
    (folder / "status.json").write_text(json.dumps({"status": status, "run_id": RUN, "trial_json_sha256": sha((folder / "trial.json").read_bytes())}) + "\n",
                                        encoding="utf-8")
    return doc


class TestKPlan(unittest.TestCase):
    def setUp(self):
        self.w = world()
        self.out = self.w.tmp / "plan_out"
        shutil.rmtree(self.out, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.out, True)

    def run_cmd(self, *argv):
        with injected(**hooks(self.w)):
            return run_main(self.w.argv(*argv))

    def test_plan_then_check_plan_then_refused_again(self):
        rc, out, err = self.run_cmd("plan", "--out-dir", str(self.out))
        self.assertEqual(rc, 0, err)
        self.assertEqual(sorted(os.listdir(self.out)), sorted(C.PLAN_OUTPUTS))
        rows = FR.parse_csv((self.out / "run_matrix.csv").read_bytes(), C.MATRIX_COLUMNS, "m")
        self.assertEqual(len(rows), 72)
        self.assertEqual(len(FR.parse_csv((self.out / "reference_cells.csv").read_bytes(), C.CELL_COLUMNS, "c")), 48)
        self.assertIn(C.TRIAL_PENDING, (self.out / "run_caps.csv").read_text())
        rc, out, err = self.run_cmd("check-plan", "--plan-dir", str(self.out))
        self.assertEqual(rc, 0, err)
        rc, out, err = self.run_cmd("plan", "--out-dir", str(self.out))
        self.assertEqual(rc, 1)
        self.assertIn("already exist", err)

    def test_check_plan_detects_a_changed_file(self):
        self.run_cmd("plan", "--out-dir", str(self.out))
        for n in C.PLAN_OUTPUTS:
            with self.subTest(n):
                p = self.out / n
                orig = p.read_bytes()
                p.write_bytes(orig.replace(b"1", b"2", 1))
                try:
                    rc, out, err = self.run_cmd("check-plan", "--plan-dir", str(self.out))
                    self.assertEqual(rc, 1)
                    self.assertIn(n, err)
                finally:
                    p.write_bytes(orig)

    def test_plan_with_a_trial_and_the_cap_rule(self):
        t = self.w.tmp / "fake_trial"
        shutil.rmtree(t, ignore_errors=True)
        fake_trial(self.w, t)
        rc, out, err = self.run_cmd("plan", "--out-dir", str(self.out), "--trial-run", str(t))
        self.assertEqual(rc, 0, err)
        caps = FR.parse_csv((self.out / "run_caps.csv").read_bytes(), C.CAPS_COLUMNS, "caps")
        plan = FR.parse_csv((self.out / "run_plan.csv").read_bytes(), C.PLAN_COLUMNS, "plan")
        # ricalcolo a mano: un addestramento a patience 4 dura da 5 × 30 + 15 + 70 = 235 s a 12 × 30 + 85 = 445 s;
        # 12 addestramenti: 2.820 s = 0,7833 h e 5.340 s = 1,4833 h; a patience 12 sempre 1,4833 h; ore di CPU × 4.
        # valutazione per rete e patience: (3 × 12.614 frame di test × 2 stati + 12 × 400 × 2) = 85.284 punteggi × 10 s
        # per 1.000 = 852,84 s = 0,2369 h (frame del mondo: quelli della configurazione del mondo, ricalcolati sotto).
        # tetto = (1,4833 + (3 − 1) × 445 / 3600) × 1,5 = 2,5958 -> 3,0 per tutti i sei run
        p4 = next(r for r in plan if r["network"] == "AttackNetV1" and r["patience"] == "4")
        p12 = next(r for r in plan if r["network"] == "AttackNetV1" and r["patience"] == "12")
        self.assertEqual((p4["epochs_min"], p4["epochs_max"], p12["epochs_min"]), ("5", "12", "12"))
        self.assertEqual((p4["training_wall_hours_min"], p4["training_wall_hours_max"], p12["training_wall_hours_min"]), ("0.78", "1.48", "1.48"))
        self.assertEqual((p4["training_cpu_hours_min"], p4["training_cpu_hours_max"]), ("3.13", "5.93"))
        cfg, _, _, pcfg, _ = load()
        ev = (3 * sum(pcfg["schema_expected"]["test_per_fold"]) * 2 + 12 * pcfg["dev"]["total"] * 2) * 10 / 1000 / 3600
        self.assertEqual(p4["evaluation_wall_hours"], f"{ev:.2f}")
        self.assertEqual([c["cap_wall_hours"] for c in caps], ["3.00"] * 6)
        self.assertEqual([c["wall_hours_min"] for c in caps], ["0.78", "0.78", "0.78", "1.48", "1.48", "1.48"])
        md = (self.out / "run_plan.md").read_text()
        self.assertIn("Training, 72 trainings with phase 2: 6.80-8.90 hours of wall time; 27.20-35.60 CPU hours.", md)
        self.assertNotIn(C.TRIAL_PENDING, (self.out / "run_caps.csv").read_text())
        rc, out, err = self.run_cmd("check-plan", "--plan-dir", str(self.out), "--trial-run", str(t))
        self.assertEqual(rc, 0, err)
        rc, out, err = self.run_cmd("check-plan", "--plan-dir", str(self.out))
        self.assertEqual(rc, 1)

    def test_cap_rule(self):
        """(stima massima + (fattore − 1) × costo massimo di un addestramento) × moltiplicatore, per eccesso a 0,5 h."""
        cfg, *_ = load()
        self.assertEqual(C.cap_of(cfg, 1.0, 0.4), 3.0)          # (1,0 + 2 × 0,4) × 1,5 = 2,7 -> 3,0
        self.assertEqual(C.cap_of(cfg, 1.0, 0.0), 1.5)          # esatto: non si arrotonda oltre
        self.assertEqual(C.cap_of(cfg, 1.01, 0.0), 2.0)
        self.assertIsNone(C.cap_of(cfg, None, 0.4))

    def test_evaluation_counts(self):
        cfg, _, _, pcfg, _ = load()
        rows = C.plan_rows(cfg, pcfg, None)
        self.assertEqual(sum(r["test_scores"] for r in rows), 3 * sum(pcfg["schema_expected"]["test_per_fold"]) * 2 * 6)
        self.assertEqual(sum(r["dev_reproductions"] for r in rows), 72 * pcfg["dev"]["total"] * 2)

    def test_real_numbers_of_the_plan(self):
        """Con la configurazione accettata: 454.104 punteggi di test e 57.600 riproduzioni del dev (piano, §C)."""
        test = [3242, 3114, 3439, 2819]
        self.assertEqual(3 * sum(test) * 2 * 6, 454_104)
        self.assertEqual(72 * 400 * 2, 57_600)

    def test_load_trial_refusals(self):
        t = self.w.tmp / "fake_trial2"
        cfg, cfg_sha, _, _, _ = load()
        code = C.check_code(cfg)
        for name, status, over, pattern in (("not PASS", "CONTROL_FAILED", None, "not a PASS"), ("other configuration", "PASS", {"config_sha256": "0" * 64}, "another configuration"),
                                            ("other code", "PASS", {"code_sha256": {}}, "other code"),
                                            ("missing network", "PASS", {"networks": {}}, "networks must be")):
            with self.subTest(name):
                shutil.rmtree(t, ignore_errors=True)
                fake_trial(self.w, t, status=status, over=over)
                with injected(**hooks(self.w)), self.assertRaisesRegex(FR.Failure, pattern):
                    C.load_trial(t, cfg, cfg_sha, code)

    def test_check_matrix_refusals(self):
        cfg, _, _, pcfg, _ = load()
        rows = C.matrix_rows(cfg, pcfg)
        cases = [("a row removed", lambda r: r.pop()), ("a row duplicated", lambda r: r.__setitem__(-1, dict(r[0]))),
                 ("a reference", lambda r: r[0].update(reference="C2")), ("wrong run", lambda r: r[0].update(run="anet_v1_p12")),
                 ("seed 45", lambda r: r[0].update(seed=45)),
                 ("swap of two networks keeping the counts", lambda r: (r[0].update(network="AttackNetV2_1"), r[12].update(network="AttackNetV1")))]
        for name, mutate in cases:
            with self.subTest(name):
                x = cp.deepcopy(rows)
                mutate(x)
                with self.assertRaises(FR.Failure):
                    C.check_matrix(x)

    def test_check_reference_cells_refusals(self):
        with injected(**hooks(self.w)):
            cfg, *_ = load()
            cells = C.load_c2_results(cfg, self.w.base)
        cases = [("removed", lambda r: r.pop()), ("duplicated", lambda r: r.__setitem__(-1, dict(r[0]))),
                 ("run of the other patience", lambda r: r[0].update(run_id=C2_RUN_IDS["nuaa_patience_12"])),
                 ("checkpoint SHA-256", lambda r: r[0].update(checkpoint_sha256="x"))]
        for name, mutate in cases:
            with self.subTest(name):
                x = cp.deepcopy(cells)
                mutate(x)
                with self.assertRaises(FR.Failure):
                    C.check_reference_cells(x)
        self.assertEqual(cells[0]["test_scores_file"], "p4_LivenessNet_fold1_seed42_test_scores.csv")


# ============================================================================ prova: rifiuti senza TensorFlow

class TestLTrialRefusals(unittest.TestCase):
    def setUp(self):
        self.w = world()
        self.out = self.w.tmp / "trial_refusals"
        shutil.rmtree(self.out, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.out, True)

    def go(self, argv, **c):
        h = hooks(self.w, **c)
        with injected(**h):
            return run_main(argv)

    def test_hooks_must_be_none_with_the_delivered_configuration(self):
        with injected(c={"TRAINER": lambda *a: None, "DEFAULT_CONFIG": os.path.relpath(self.w.c5_path, ROOT)}):
            rc, out, err = run_main(["trial", "--attempt", "1", "--i-am-authorized", "--checkpoint-root", str(self.w.tmp / "ck")])
        self.assertEqual(rc, 1)
        self.assertIn("test hooks are set", err)
        self.assertEqual(C.check_hooks(True), {"official_configuration": True, "set": [], "changed": []})
        with injected(r={"MAKE_SCORER": lambda *a: None}), self.assertRaisesRegex(FR.Failure, "pad_arch_run.MAKE_SCORER"):
            C.check_hooks(True)

    def test_authorisation_and_checkpoint_root_required(self):
        rc, out, err = self.go([a for a in self.w.argv("trial", out=self.out) if a != "--i-am-authorized"])
        self.assertIn("--i-am-authorized", err)
        argv = self.w.argv("trial", out=self.out)
        i = argv.index("--checkpoint-root")
        rc, out, err = self.go(argv[:i] + argv[i + 2:])
        self.assertIn("--checkpoint-root", err)
        self.assertFalse(self.out.exists())

    def test_dirty_tree_refused_before_any_folder(self):
        rc, out, err = self.go(self.w.argv("trial", out=self.out), REPO_STATE=lambda root: {"commit": "abcdef0", "dirty": True})
        self.assertEqual(rc, 1)
        self.assertIn("not clean", err)
        self.assertFalse(self.out.exists())

    def test_checkpoint_root_inside_the_repository_refused(self):
        rc, out, err = self.go(self.w.argv("trial", out=self.out, ck=ROOT / "results" / "anetcmp_ck"))
        self.assertEqual(rc, 1)
        self.assertIn("outside the repository", err)
        self.assertFalse(self.out.exists())

    def test_checkpoint_root_that_is_a_link_refused(self):
        link = self.w.tmp / "ck_link"
        if not link.exists():
            os.symlink(self.w.tmp, link)
        rc, out, err = self.go(self.w.argv("trial", out=self.out, ck=link))
        self.assertIn("symbolic link", err)

    def test_existing_run_folder_is_an_earlier_attempt(self):
        (self.out / RUN).mkdir(parents=True)
        rc, out, err = self.go(self.w.argv("trial", out=self.out))
        self.assertIn("--attempt must be 2", err)
        self.assertEqual(os.listdir(self.out / RUN), [])

    def test_second_attempt_gets_the_r2_suffix(self):
        from unittest import mock
        (self.out / RUN).mkdir(parents=True)
        with mock.patch.object(C, "load_inputs", lambda *a, **k: FR.fail("stopped by the test after the folder")):
            rc, out, err = self.go(self.w.argv("trial", out=self.out, attempt=2))
        self.assertEqual(rc, 1)
        st = json.loads((self.out / f"{RUN}-r2" / "status.json").read_text())
        self.assertEqual((st["run_id"], st["attempt"], st["previous_attempts"]), (f"{RUN}-r2", 2, [{"run_id": RUN, "status": "no status.json"}]))

    def test_attempt_rule(self):
        out, ck = self.out, self.w.tmp / "ck_attempts"
        shutil.rmtree(ck, ignore_errors=True)
        self.addCleanup(shutil.rmtree, ck, True)
        self.assertEqual(C.attempt_rule(out, ck, 1), {"attempt": 1, "previous_attempts": []})
        with self.assertRaisesRegex(FR.Failure, "--attempt must be 1"):
            C.attempt_rule(out, ck, 2)
        (out / "20261120-ANETTRIAL-1234567").mkdir(parents=True)
        (out / "20261120-ANETTRIAL-1234567" / "status.json").write_text('{"status": "FAIL"}', encoding="utf-8")
        ck.mkdir()
        (ck / "20261121-ANETTRIAL-abcdef0-r2").mkdir()
        got = C.attempt_rule(out, ck, 3)
        self.assertEqual(got["previous_attempts"], [{"run_id": "20261120-ANETTRIAL-1234567", "status": "FAIL"},
                                                    {"run_id": "20261121-ANETTRIAL-abcdef0-r2", "status": "no status.json"}])
        for bad in (1, 2, 4, True, None):
            with self.subTest(bad), self.assertRaisesRegex(FR.Failure, "--attempt must be 3"):
                C.attempt_rule(out, ck, bad)
        (ck / "20261121-anettrial-ABC").mkdir()
        with self.assertRaisesRegex(FR.Failure, "looks like a trial run id"):
            C.attempt_rule(out, ck, 3)
        rc, o, err = self.go([a for a in self.w.argv("trial", out=self.out) if a not in ("--attempt", "1")])
        self.assertIn("needs --attempt", err)

    def test_output_root_in_the_repository_outside_results_anetcmp(self):
        rc, out, err = self.go(self.w.argv("trial", out=ROOT / "delivery" / "x"))
        self.assertEqual(rc, 1)
        self.assertIn("refusing the run folder", err)
        self.assertFalse((ROOT / "delivery" / "x").exists())

    def test_statuses_precedence(self):
        ok = {n: {"outcome": {"evaluable": True}} for n in AN.NETWORKS}
        self.assertEqual(C.outcome_status(ok, {"outcome": "reproduced"}, None), "PASS")
        self.assertEqual(C.outcome_status(ok, {"outcome": "not reproduced"}, None), "CONTROL_FAILED")
        bad = {**ok, "AttackNetV1": {"outcome": {"evaluable": False}}}
        self.assertEqual(C.outcome_status(bad, {"outcome": "not reproduced"}, None), "NOT_EVALUABLE")
        self.assertEqual(C.outcome_status(bad, {}, {"step": "x"}), "INCOMPLETE")


class TestMPublicTexts(unittest.TestCase):
    def test_no_identifier_or_path_in_the_texts(self):
        cfg, *_ = load()
        texts = [C.TEST_READ, json.dumps(cfg), *(DEF.conclusion(k, r) for k in AN.NETWORKS for r in DEF.reading_names(k))]
        for t in texts:
            self.assertEqual(PAC.output_problems(t, FR.Inputs()), [])
            self.assertEqual(FR.public_problems(t, FR.Inputs(), markdown=True), [])

    def test_delivered_configuration(self):
        """La configurazione consegnata, se c'è con i file che fissa: valida, uguale a quella che make_config rigenera, con il
        codice fissato uguale ai file (saltato solo se un file manca, mai se un file è diverso)."""
        p = ROOT / C.DEFAULT_CONFIG
        if not p.is_file():
            skip_missing(self, "delivered configuration")
        d = json.loads(p.read_text(encoding="utf-8"))
        pinned = [d["preregistration"]["path"], d["bn_recalibration_config"]["path"], f"{d['c2_trial']['folder']}/trial.json",
                  d["c2_results"]["status"]["path"], d["c2_results"]["results"]["path"], *C.CODE_FILES]
        if any(not (ROOT / x).is_file() for x in pinned):
            skip_missing(self, "delivered configuration")
        cfg, *_ = C.load_config(p, ROOT)
        self.assertEqual(cfg["c2_results"]["evaluation_run_id"], EVAL_ID)
        self.assertEqual(MC.doc(), d)
        C.check_code(cfg)


# ============================================================================ agganci, configurazione ufficiale, definizioni aggiunte

def project_modules() -> dict:
    """I moduli del progetto (cartella scripts/) caricati dopo l'import di pad_anet_comparison, diretti e indiretti, con
    i loro agganci: variabili maiuscole al livello del modulo assegnate a None nel sorgente (ast)."""
    import ast
    out = {}
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f or Path(f).resolve().parent != (ROOT / "scripts").resolve() or name.startswith("check_") or name.startswith("__"):
            continue
        tree = ast.parse(Path(f).read_text(encoding="utf-8"))
        names = [t.id for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and n.value.value is None
                 for t in n.targets if isinstance(t, ast.Name) and t.id.isupper()]
        if names:
            out[name] = tuple(names)
    return out


class TestPHooks(unittest.TestCase):
    def test_the_hook_registry_is_complete(self):
        found = project_modules()
        self.assertIn("nuaa_session_ablation_run", found)
        self.assertEqual({m: sorted(v) for m, v in found.items()}, {m: sorted(v) for m, v in C.HOOKS.items()})

    def test_every_hook_stops_the_official_commands(self):
        for m, names in C.HOOKS.items():
            mod = C if m == "pad_anet_comparison" else sys.modules[m]
            for k in names:
                with self.subTest(f"{m}.{k}"):
                    saved = getattr(mod, k)
                    setattr(mod, k, lambda *a: None)
                    try:
                        with self.assertRaisesRegex(FR.Failure, f"{m}.{k}"):
                            C.check_hooks(True)
                        self.assertEqual(C.check_hooks(False)["set"], [f"{m}.{k}"])
                    finally:
                        setattr(mod, k, saved)
        with injected(fr={"FRAME_SHAPE": SHAPE}), self.assertRaisesRegex(FR.Failure, "nuaa_folds_run.FRAME_SHAPE"):
            C.check_hooks(True)

    def test_sr_repo_state_stops_the_trial_with_the_delivered_configuration(self):
        """Con la configurazione «consegnata» (quella del mondo, DEFAULT_CONFIG relativo alla radice) e il solo
        SR.REPO_STATE impostato, il comando vero si ferma prima di ogni cartella."""
        w = world()
        with injected(sr={"REPO_STATE": lambda root: dict(STATE)}, c={"DEFAULT_CONFIG": os.path.relpath(w.c5_path, ROOT)}):
            rc, out, err = run_main(["trial", "--attempt", "1", "--i-am-authorized", "--checkpoint-root", str(w.tmp / "ck_official")])
        self.assertEqual(rc, 1)
        self.assertIn("nuaa_session_ablation_run.REPO_STATE", err)
        self.assertFalse((w.tmp / "ck_official").exists())

    def test_official_configuration_by_identity(self):
        """Riconosciuta per identità del file: una grafia con «..» e un collegamento sono la configurazione consegnata, una
        copia no (la configurazione «consegnata» qui è quella del mondo, DEFAULT_CONFIG relativo alla radice)."""
        w = world()
        (w.tmp / "sub").mkdir(exist_ok=True)
        with injected(c={"DEFAULT_CONFIG": os.path.relpath(w.c5_path, ROOT)}):
            self.assertTrue(C.is_official(w.c5_path))
            self.assertTrue(C.is_official(w.tmp / "sub" / ".." / w.c5_path.relative_to(w.tmp)))
            link = w.tmp / "link_to_config.json"
            if not os.path.lexists(link):
                os.symlink(w.c5_path, link)
            self.assertTrue(C.is_official(link))
            copy = w.tmp / "copy_of_config.json"
            shutil.copy2(w.c5_path, copy)
            self.assertFalse(C.is_official(copy))
            self.assertFalse(C.is_official(w.tmp / "missing.json"))

    def test_machine_comparison_field_by_field(self):
        self.enterContext(injected(c={"MACHINE": lambda: dict(FIXED_M)}, fr={"environment": lambda: dict(FIXED_E)}))
        c2 = {"run_id": "r", "machine": dict(FIXED_M), "environment": dict(FIXED_E)}
        self.assertEqual(len(R.MACHINE_FIELDS) + len(R.VERSION_FIELDS), 11)
        for part, fields in (("machine", R.MACHINE_FIELDS), ("environment", R.VERSION_FIELDS)):
            for f in fields:
                with self.subTest(f):
                    got = C.machine_comparison({**c2, part: {**c2[part], f: "another value"}})
                    self.assertEqual(got["differing"], [f])
                    got = C.machine_comparison({**c2, part: {k: v for k, v in c2[part].items() if k != f}})
                    self.assertEqual(got["differing"], [f])

    def test_every_pinned_value_stops_the_official_commands(self):
        for (m, k), v in C.PINNED_VALUES.items():
            with self.subTest(f"{m}.{k}"):
                mod = sys.modules[m]
                setattr(mod, k, (lambda: 1.0) if callable(v) else ("x" if isinstance(v, str) else (1, 1, 1)))
                try:
                    with self.assertRaisesRegex(FR.Failure, f"{m}.{k}"):
                        C.check_hooks(True)
                    self.assertEqual(C.check_hooks(False)["changed"], [f"{m}.{k}"])
                finally:
                    setattr(mod, k, v)

    def test_pinned_values_are_complete(self):
        """Ogni valore che i test sostituiscono nei moduli del progetto (le iniezioni di base di hooks e quelle di
        TestOTrial.go) è un aggancio di HOOKS o una voce di PINNED_VALUES, e ogni voce di PINNED_VALUES è sostituita dai
        test: togliere una voce dall'elenco la lascerebbe passare inosservata nei comandi ufficiali."""
        names = {"c": "pad_anet_comparison", "p": "pad_arch_comparison", "sr": "nuaa_session_ablation_run", "fr": "nuaa_folds_run",
                 "r": "pad_arch_run"}
        replaced = {(names[g], k) for g, d in hooks(world(), MACHINE=lambda: dict(FIXED_M)).items() for k in (d or {})}
        replaced |= {("nuaa_folds_run", "environment")}
        hooked = {(m, k) for m, ks in C.HOOKS.items() for k in ks}
        self.assertEqual(sorted(replaced - hooked), sorted(C.PINNED_VALUES))

    def test_the_caught_messages_occur_once_in_the_frozen_sources(self):
        """Le due frasi dei moduli congelati che la prova riconosce: una volta ciascuna nel sorgente congelato e nel codice
        della prova (un cambio di testo sarebbe visto qui)."""
        src = (ROOT / "scripts" / "pad_anet_comparison.py").read_text(encoding="utf-8")
        for frozen, text in (("pad_arch_comparison.py", "inference pass: probabilities non-finite"),
                             ("nuaa_bn_recalibration.py", "the re-estimated mean or variance is not finite")):
            self.assertEqual((ROOT / "scripts" / frozen).read_text(encoding="utf-8").count(text), 1, frozen)
            self.assertEqual(src.count(f'"{text}"'), 1, text)

    def test_check_machine(self):
        w = world()
        fixed = hooks(w, MACHINE=lambda: dict(FIXED_M))
        fixed["fr"]["environment"] = lambda: dict(FIXED_E)
        with CPC.tensorflow_unloaded(), injected(**fixed):
            rc, out, err = run_main(w.argv("check-machine"))
            self.assertEqual(rc, 0, err)
            self.assertIn(f"check-machine: machine and versions equal to the C2 trial {C2_TRIAL_ID} (11 fields); TensorFlow not imported; "
                          "nothing written", out)
            self.assertNotIn("tensorflow", sys.modules)
        other = hooks(w, MACHINE=lambda: {**FIXED_M, "os": "another os"})
        other["fr"]["environment"] = lambda: dict(FIXED_E)
        with CPC.tensorflow_unloaded(), injected(**other):
            rc, out, err = run_main(w.argv("check-machine"))
        self.assertEqual(rc, 1)
        self.assertIn("os differ from the C2 trial", out)
        self.assertIn("RESULT: FAIL (check-machine)", out)

    def test_machine_comparison(self):
        self.enterContext(injected(c={"MACHINE": lambda: dict(FIXED_M)}, fr={"environment": lambda: dict(FIXED_E)}))
        c2 = {"run_id": "r", "machine": dict(FIXED_M), "environment": dict(FIXED_E)}
        self.assertTrue(C.machine_comparison(c2)["equal"])
        got = C.machine_comparison({**c2, "machine": {**c2["machine"], "processor": "another"}})
        self.assertEqual((got["equal"], got["differing"]), (False, ["processor"]))
        got = C.machine_comparison({**c2, "environment": {**c2["environment"], "numpy": "0.0"}})
        self.assertEqual(got["differing"], ["numpy"])
        self.assertEqual(C.machine_comparison({"run_id": "r"})["equal"], False)


class TestQDefinitionsAdded(unittest.TestCase):
    def test_first_non_finite_epoch_only(self):
        h = {"loss": [0.5, float("nan"), float("nan")], "val_loss": [0.5, 0.4, float("inf")]}
        self.assertEqual(DEF.non_finite_entries(h), [["loss", 2], ["val_loss", 3]])

    def test_statistics_finiteness_must_be_true(self):
        ok = {"p": np.full((2, 2), 0.5), "z": np.zeros((2, 2))}
        for v in (None, False, 1):
            self.assertFalse(DEF.dev_outcome({"ordinary": ok}, v)["evaluable"])

    def test_percentage_points_half_away_from_zero(self):
        self.assertEqual(DEF.pp(0.00125), 0.13)                  # round(100 * 0.00125, 2) darebbe 0.12
        self.assertEqual(DEF.pp(-0.00125), -0.13)
        self.assertEqual(DEF.pp(Fraction(1, 800)), 0.13)
        self.assertEqual(DEF.pp_text(Fraction(-1, 800)), "-0.13")
        self.assertEqual(DEF.pp(Fraction(1, 3)), 33.33)
        self.assertEqual(DEF.pp_text(Fraction(2, 3)), "66.67")
        self.assertEqual(DEF.pp_text(0.0998332546851536), "9.98")
        self.assertEqual(DEF.pp_text(0), "0.00")

    def test_units_in_fold_seed_order(self):
        a = DEF.prepare([unit_raw(1, 43, scores_for(1, 43, "ordinary")), unit_raw(1, 42, scores_for(1, 42, "ordinary"))])
        b = DEF.prepare([unit_raw(1, 43, scores_for(1, 43, "re-estimated")), unit_raw(1, 42, scores_for(1, 42, "re-estimated"))])
        with self.assertRaisesRegex(FR.Failure, "ascending \\(fold, seed\\) order"):
            DEF.pair_states(a, b)

    def test_descriptive_conditions(self):
        conds = {}
        for p in DEF.PATIENCES:
            for t, thr in (("fixed_0.5", None), ("dev_eer", {n: 0.3 for n in DEF.NETWORKS})):
                c = conditions(thr=thr)
                conds[(p, "ordinary", t)] = c[(4, "ordinary", "fixed_0.5")]
                conds[(p, "re-estimated", t)] = c[(4, "re-estimated", "fixed_0.5")]
        res = DEF.descriptive_analysis(conds)
        self.assertEqual(sorted(res["pairs"]), ["patience_12_dev_eer", "patience_12_fixed_0.5", "patience_4_dev_eer"])
        self.assertEqual(len(res["rankings"]), 8)
        eer = res["pairs"]["patience_4_dev_eer"]
        self.assertTrue(eer["contrasts"]["AttackNetV1"]["computed"])
        self.assertNotIn("reading", eer["contrasts"]["AttackNetV1"])
        self.assertNotIn("conclusion", eer["contrasts"]["AttackNetV1"])
        with self.assertRaisesRegex(FR.Failure, "not 0.5"):
            DEF.primary_analysis({k: v for k, v in conds.items() if k[2] == "fixed_0.5"} | {
                (4, "ordinary", "fixed_0.5"): conds[(4, "ordinary", "dev_eer")], (4, "re-estimated", "fixed_0.5"): conds[(4, "re-estimated", "dev_eer")]})
        with self.assertRaisesRegex(FR.Failure, "are missing \\(all 8 are required\\)"):
            DEF.descriptive_analysis({k: v for k, v in conds.items() if k[0] == 4})

    def test_test_frames_of_each_fold_in_the_matrix(self):
        cfg, _, _, pcfg, _ = load()
        test = list(pcfg["schema_expected"]["test_per_fold"])
        test[1] += 1                                            # distinti, così uno scambio fra fold si vede
        for r in C.matrix_rows(cfg, {**pcfg, "schema_expected": {**pcfg["schema_expected"], "test_per_fold": test}}):
            self.assertEqual(r["test_frames_of_fold"], test[r["fold"] - 1])


# ============================================================================ controllo delle BN (TensorFlow, senza la prova)

class TestRBnCheck(unittest.TestCase):
    def setUp(self):
        need_tf(self)
        import tensorflow as tf
        net, shape = "AttackNetV2_2", (8, 8, 3)
        rng = np.random.default_rng(5)
        self.fit = {"X": rng.integers(0, 255, (20, 8, 8, 3), dtype=np.uint8), "y": np.array([0, 1] * 10)}
        self.dev = {"X": rng.integers(0, 255, (10, 8, 8, 3), dtype=np.uint8), "y": np.array([0, 1] * 5)}
        self.orig = AN.build(net, shape, ref_arch())
        self.order = AN.bn_order(net)
        self.snap = BR.snapshot_weights(self.orig)
        self.copy = tf.keras.models.clone_model(self.orig)
        self.copy.set_weights(self.orig.get_weights())
        self.stats = R.reestimate(self.copy, self.fit, self.order, 16, None, "t")
        _, final = BR.evaluate(self.copy, self.fit, 16, "fit", layers=self.order)
        self.cons = BR.consistency(self.stats, final, 1e-4, 1e-6)
        self.ev_ord, _ = BR.evaluate(self.orig, self.dev, 16, "dev")
        self.dev_after, _ = BR.evaluate(self.copy, self.dev, 16, "dev after")
        c = AN.expected_counts(net, shape)
        self.cfg = {"architectures": {"input_shape": list(shape), "networks": {net: {"batchnorm_names": self.order,
                                                                                    "non_trainable_parameters": c["non_trainable"]}}}}
        self.bn_cfg = {"batchnorm": {"epsilon": 1e-3}}
        self.net = net

    def go(self, n_fit=20, ev_ord=None, dev_after=None):
        return C.bn_check(self.net, n_fit, self.order, self.orig, self.copy, self.stats, self.cons, self.snap, ev_ord or self.ev_ord,
                          dev_after or self.dev_after, self.cfg, self.bn_cfg)

    def test_pass(self):
        got = self.go()
        self.assertTrue(got["passed"])
        self.assertEqual({k: v["count"] for k, v in got["layers"].items()}, AN.expected_recalibration_counts(self.net, 20, (8, 8, 3)))

    def test_states_that_do_not_differ_fail_check_5(self):
        with self.assertRaisesRegex(FR.Failure, "BN check failed: 5_states_differ$"):
            self.go(dev_after=self.ev_ord)

    def test_other_frame_count_fails_check_2(self):
        with self.assertRaisesRegex(FR.Failure, "BN check failed: 2_values_per_channel$"):
            self.go(n_fit=21)


# ============================================================================ prova vera (TensorFlow)

class TestNTrainerBehaviour(unittest.TestCase):
    def setUp(self):
        need_tf(self)

    def data(self):
        rng = np.random.default_rng(3)
        X = rng.integers(0, 255, (24, 8, 8, 3), dtype=np.uint8)
        y = np.array([0, 1] * 12)
        return {"X_fit": X[:16], "y_fit": y[:16], "X_dev": X[16:], "y_dev": y[16:]}

    def test_same_card_and_weights_as_the_original_on_a_finite_history(self):
        arch = ref_arch()
        training = {**FR.FROZEN_TRAINING, "max_epochs": 2}
        out = {}
        for name, fn in (("original", PAC.trial_trainer), ("copy", C.anet_trainer)):
            with contextlib.redirect_stdout(io.StringIO()):
                card, model = fn(self.data(), training, "AttackNetV2_2", 42, lambda: AN.build("AttackNetV2_2", (8, 8, 3), arch), lambda s: None, PAC.EpochTimer([]))
            out[name] = (card, [w.copy() for w in model.get_weights()])
        a, b = out["original"][0], out["copy"][0]
        for k in ("history", "restored_epoch", "best_val_loss", "val_acc_at_best", "epochs_run", "restored_by"):
            self.assertEqual(a[k], b[k], k)
        self.assertEqual(b["non_finite"], [])
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(out["original"][1], out["copy"][1])))

    def test_non_finite_weights_give_a_card_with_nulls(self):
        arch = ref_arch()

        def build():
            m = AN.build("AttackNetV1", (8, 8, 3), arch)
            w = m.get_layer("output").get_weights()
            m.get_layer("output").set_weights([np.full_like(w[0], np.nan), w[1]])
            return m
        with contextlib.redirect_stdout(io.StringIO()):
            card, _ = C.anet_trainer(self.data(), {**FR.FROZEN_TRAINING, "max_epochs": 2}, "AttackNetV1", 42, build, lambda s: None, PAC.EpochTimer([]))
        self.assertIsNone(card["restored_epoch"])
        self.assertIsNone(card["best_val_loss"])
        self.assertIn(["val_loss", 1], card["non_finite"])
        self.assertTrue(all(v is None for v in card["history"]["val_loss"]))
        self.assertFalse(DEF.training_outcome(card["history"], None)["evaluable"])
        PAC.json_text(card, "card")                              # la scheda si può scrivere


class TestOTrial(unittest.TestCase):
    """La prova vera su CPU nel mondo 8×8: prima con la storia sintetica della scheda (CONTROL_FAILED, uscite scritte),
    poi con la storia di LivenessNet prodotta dal trainer (PASS); un'AttackNet resa non finita (NOT_EVALUABLE); uno scorer
    discorde (FAIL, uscite tolte)."""

    @classmethod
    def setUpClass(cls):
        cls.ready = TF_AVAILABLE and REF_ARCH.is_file()
        if not cls.ready:
            return
        tf_threads()
        cls.w = world()

    def setUp(self):
        need_tf(self)
        self.out = self.w.tmp / "trial_out"
        self.ck = self.w.tmp / "ckpt"
        for p in (self.out, self.ck):
            shutil.rmtree(p, ignore_errors=True)
            self.addCleanup(shutil.rmtree, p, True)

    def go(self, **c):
        r = c.pop("r", None)
        h = hooks(self.w, **{"MACHINE": lambda: dict(FIXED_M), **c})
        h["fr"]["environment"] = lambda: dict(FIXED_E)
        h["r"] = r
        with injected(**h):
            rc, out, err = run_main(self.w.argv("trial", out=self.out, ck=self.ck))
        return rc, out, err, self.out / RUN

    def doc(self, run):
        return json.loads((run / "trial.json").read_text())

    def test_1_control_failed_then_pass_then_plan(self):
        rc, out, err, run = self.go()
        self.assertEqual(rc, 3, err[-3000:] + out[-2000:])
        st = json.loads((run / "status.json").read_text())
        self.assertEqual((st["status"], st["control"]), ("CONTROL_FAILED", "not reproduced"))
        doc = self.doc(run)
        hist = {k: [e[k] for e in doc["control"]["epochs"]] for k in C.HISTORY_KEYS}
        for n in AN.NETWORKS:
            self.assertTrue(doc["networks"][n]["bn_check"]["passed"])
        # la storia vera di LivenessNet nella scheda e nella prova C2 del mondo: ora la prova deve riprodurla
        self.w.write_parents(hist)
        shutil.rmtree(self.out)
        shutil.rmtree(self.ck)
        rc, out, err, run = self.go()
        self.assertEqual(rc, 0, err[-3000:] + out[-2000:])
        self.assertEqual(sorted(os.listdir(run)), ["run.log", "status.json", "trial.json", "trial.md"])
        doc = self.doc(run)
        self.assertEqual(doc["status"], "PASS")
        self.assertEqual(doc["control"]["outcome"], "reproduced")
        self.assertEqual(doc["control"]["c2_trial"]["outcome"], "reproduced")
        n_fit = doc["unit"]["frames"]["fitting"]["n"]
        for n in AN.NETWORKS:
            rec = doc["networks"][n]
            self.assertTrue(rec["outcome"]["evaluable"])
            self.assertEqual(rec["batchnorm_layers"], 5)
            layers = rec["bn_check"]["layers"]
            self.assertEqual(list(layers), AN.bn_order(n))
            self.assertEqual({k: v["count"] for k, v in layers.items()}, AN.expected_recalibration_counts(n, n_fit, SHAPE))
            self.assertTrue(all(v["moving_mean_changed"] and v["moving_variance_changed"] for v in layers.values()))
            self.assertGreater(rec["bn_check"]["checks"]["5_states_differ"]["dev_max_abs_score_difference"], 0)
            self.assertTrue(rec["reload"]["weights_equal_bytes"])
            self.assertTrue(rec["installed_copy"]["equal_bytes_to_re_estimated_copy"])
            self.assertEqual(sorted(rec["steps"]), sorted(C.PHASE2_STEPS + ("installed_copy",)))
            self.assertEqual(sha((self.ck / RUN / rec["checkpoint"]["file"]).read_bytes()), rec["checkpoint"]["sha256"])
        self.assertTrue(doc["projection"]["available"])
        self.assertEqual(len(doc["projection"]["caps"]), 6)
        self.assertEqual(doc["threads"]["effective"], {"intra_op": 10, "inter_op": 10})
        self.assertEqual(doc["hooks"]["official_configuration"], False)
        self.assertNotIn("test", {r for r in ("fitting", "dev")})
        plan = self.w.tmp / "plan_from_trial"
        shutil.rmtree(plan, ignore_errors=True)
        with injected(**hooks(self.w)):
            rc, o, e = run_main(self.w.argv("plan", "--out-dir", str(plan), "--trial-run", str(run)))
            self.assertEqual(rc, 0, e)
            rc, o, e = run_main(self.w.argv("check-plan", "--plan-dir", str(plan), "--trial-run", str(run)))
            self.assertEqual(rc, 0, e)
        shutil.rmtree(plan, ignore_errors=True)
        st = json.loads((run / "status.json").read_text())
        self.assertEqual((st["attempt"], st["previous_attempts"], doc["attempt"]), (1, [], 1))
        self.assertTrue(doc["control"]["machine_and_versions"]["equal"])
        log = (run / "run.log").read_text()
        self.assertEqual(log.count("RESULT:"), 1)
        self.assertTrue(log.rstrip().endswith("RESULT: PASS (trial)"))
        rc, out, err, run2 = self.go()
        self.assertIn("--attempt must be 2", err)

    def test_2_not_evaluable_attacknet(self):
        def trainer(data, training, network, seed, build_model, cap_check, timer):
            if network != "AttackNetV1":
                return C.anet_trainer(data, training, network, seed, build_model, cap_check, timer)

            def bad():
                m = build_model()
                w = m.get_layer("output").get_weights()
                m.get_layer("output").set_weights([np.full_like(w[0], np.nan), w[1]])
                return m
            return C.anet_trainer(data, training, network, seed, bad, cap_check, timer)
        rc, out, err, run = self.go(TRAINER=trainer)
        self.assertEqual(rc, 5, err[-3000:])
        doc = self.doc(run)
        self.assertEqual(doc["networks"]["AttackNetV1"]["outcome"]["reason"], "not evaluable: no finite val_loss")
        self.assertNotIn("bn_check", doc["networks"]["AttackNetV1"])
        self.assertTrue(doc["networks"]["AttackNetV2_2"]["bn_check"]["passed"])
        self.assertFalse(doc["projection"]["available"])
        st = json.loads((run / "status.json").read_text())
        self.assertEqual(st["findings"]["not_evaluable"], {"AttackNetV1": "not evaluable: no finite val_loss"})
        self.assertEqual((run / "run.log").read_text().count("FINDING: AttackNetV1 not evaluable: no finite val_loss"), 1)

    def test_3_discordant_scorer_fails_and_removes_the_outputs(self):
        def scorer(path, s):
            real = FR.tf_scorer(path, s)
            return lambda X: np.clip(real(X)[:, ::-1], 0, 1)
        rc, out, err, run = self.go(r={"MAKE_SCORER": scorer})
        self.assertEqual(rc, 1)
        self.assertEqual(sorted(os.listdir(run)), ["run.log", "status.json"])
        st = json.loads((run / "status.json").read_text())
        self.assertEqual(st["status"], "FAIL")
        self.assertIn("do not agree", st["reason"])


    def test_4_installed_copy_different_fails(self):
        from unittest import mock
        real = C.BT.install

        def install(copy, installed):
            return real(copy, {n: (m, v * np.float32(1.5)) for n, (m, v) in installed.items()})
        with mock.patch.object(C.BT, "install", install):
            rc, out, err, run = self.go()
        self.assertEqual(rc, 1)
        st = json.loads((run / "status.json").read_text())
        self.assertIn("installed statistics", st["reason"])
        self.assertEqual(sorted(os.listdir(run)), ["run.log", "status.json"])

    def test_5_bn_check_failure_fails(self):
        from unittest import mock
        real = C.AN.expected_recalibration_counts
        with mock.patch.object(C.AN, "expected_recalibration_counts", lambda n, f, s=AN.INPUT_SHAPE: {k: v + 1 for k, v in real(n, f, s).items()}):
            rc, out, err, run = self.go()
        self.assertEqual(rc, 1)
        st = json.loads((run / "status.json").read_text())
        self.assertIn("BN check failed: 2_values_per_channel", st["reason"])


    def test_6_c2_trial_history_differs_from_the_card(self):
        """Scheda accettata riprodotta ma prova C2 diversa: CONTROL_FAILED (le due condizioni sono richieste entrambe)."""
        saved = dict(self.w.c2_history)
        try:
            self.w.c2_history = {**saved, "val_loss": [v + 1e-3 for v in saved["val_loss"]]}
            self.w.write_c5()
            rc, out, err, run = self.go()
            self.assertEqual(rc, 3, err[-2000:])
            c = self.doc(run)["control"]
            self.assertEqual((c["accepted_card"]["outcome"], c["c2_trial"]["outcome"]), ("reproduced", "not reproduced"))
        finally:
            self.w.c2_history = saved
            self.w.write_c5()


    def test_7_non_finite_dev_scores_are_not_evaluable(self):
        from unittest import mock
        real = BR.forward
        n_dev = self.w.counts("dev")["n"]

        def forward(model, X, batch, layers=(), scores=True):
            f = real(model, X, batch, layers=layers, scores=scores)
            if scores and len(X) == n_dev:
                f["p"] = f["p"].copy()
                f["p"][0, 1] = np.nan
            return f
        with mock.patch.object(BR, "forward", forward):
            rc, out, err, run = self.go()
        self.assertEqual(rc, 5, err[-2000:])
        for n in AN.NETWORKS:
            self.assertEqual(self.doc(run)["networks"][n]["outcome"]["reason"],
                             "not evaluable: non-finite scores or logits on the in-memory dev (ordinary BN)")

    def test_8_non_finite_statistics_are_not_evaluable(self):
        from unittest import mock
        with mock.patch.object(R, "reestimate", lambda *a, **k: FR.fail("t: bn1_1: the re-estimated mean or variance is not finite")):
            rc, out, err, run = self.go()
        self.assertEqual(rc, 5, err[-2000:])
        self.assertEqual(self.doc(run)["networks"]["AttackNetV1"]["outcome"]["reason"], "not evaluable: non-finite re-estimated statistics")

    def test_9_non_finite_fitting_scores_are_not_evaluable(self):
        from unittest import mock
        real = PAC.inference_pass
        n_fit = self.w.counts("armB_fitting")["n"]

        def inference_pass(model, X, batch):
            if len(X) == n_fit:
                FR.fail("inference pass: probabilities non-finite, outside [0, 1] or of the wrong shape")
            return real(model, X, batch)
        with mock.patch.object(PAC, "inference_pass", inference_pass):
            rc, out, err, run = self.go()
        self.assertEqual(rc, 5, err[-2000:])
        self.assertEqual(self.doc(run)["networks"]["AttackNetV2_1"]["outcome"]["reason"],
                         "not evaluable: probabilities non-finite, outside [0, 1] or of the wrong shape on the fitting in the timed inference pass "
                         "(ordinary BN)")

    def test_9b_forbidden_text_in_the_log_never_after_a_pass(self):
        def trainer(*a, **k):
            print(f"working in {Path.home()}", flush=True)
            return C.anet_trainer(*a, **k)
        rc, out, err, run = self.go(TRAINER=trainer)
        self.assertEqual(rc, 1)
        log = (run / "run.log").read_text()
        self.assertNotIn("RESULT: PASS", log)
        self.assertNotIn("RESULT: CONTROL_FAILED", log)
        self.assertTrue(log.rstrip().endswith("RESULT: FAIL (trial)"))
        self.assertNotIn("RESULT: PASS", out)
        self.assertEqual(sorted(os.listdir(run)), ["run.log", "status.json"])

    def test_9d_findings_survive_a_later_fail(self):
        """Controllo non riprodotto (altra macchina) e AttackNetV1 non valutabile, poi un FAIL su AttackNetV2_1 (scorer
        discorde): status.json FAIL e run.log conservano i due esiti, scritti nel momento in cui sono noti."""
        def trainer(data, training, network, seed, build_model, cap_check, timer):
            if network != "AttackNetV1":
                return C.anet_trainer(data, training, network, seed, build_model, cap_check, timer)

            def bad():
                m = build_model()
                w = m.get_layer("output").get_weights()
                m.get_layer("output").set_weights([np.full_like(w[0], np.nan), w[1]])
                return m
            return C.anet_trainer(data, training, network, seed, bad, cap_check, timer)

        def scorer(path, s):
            real = FR.tf_scorer(path, s)
            return lambda X: np.clip(real(X)[:, ::-1], 0, 1)
        rc, out, err, run = self.go(TRAINER=trainer, MACHINE=lambda: {**FIXED_M, "processor": "another processor"}, r={"MAKE_SCORER": scorer})
        self.assertEqual(rc, 1)
        st = json.loads((run / "status.json").read_text())
        self.assertEqual(st["status"], "FAIL")
        self.assertIn("machine and versions differing: processor", st["findings"]["control_not_reproduced"])
        self.assertEqual(st["findings"]["not_evaluable"], {"AttackNetV1": "not evaluable: no finite val_loss"})
        log = (run / "run.log").read_text()
        self.assertEqual(log.count("FINDING: LivenessNet control not reproduced"), 1)
        self.assertEqual(log.count("FINDING: AttackNetV1 not evaluable: no finite val_loss"), 1)
        self.assertEqual(log.count("FINDING:"), 2)
        self.assertEqual(sorted(os.listdir(run)), ["run.log", "status.json"])

    def after_threads_fail(self, label: str, seen: list):
        """check_threads che fallisce al controllo «after <label>» (thread alterati), dopo aver letto lo status.json in
        corso."""
        from unittest import mock
        real = PAC.check_threads

        def check_threads(tf, cfg_pr, when):
            if when == f"after {label}":
                seen.append(json.loads((self.out / RUN / "status.json").read_text()))
                FR.fail(f"threads changed {when}")
            return real(tf, cfg_pr, when)
        return mock.patch.object(PAC, "check_threads", check_threads)

    def test_9e_control_finding_before_the_after_threads_check(self):
        """Controllo non riprodotto (altra macchina), poi thread alterati al controllo «after LivenessNet»: FAIL, ma il
        finding è già nello status.json in corso (RUNNING) e resta nello stato finale e una sola volta in run.log."""
        seen = []
        with self.after_threads_fail("LivenessNet", seen):
            rc, out, err, run = self.go(MACHINE=lambda: {**FIXED_M, "processor": "another processor"})
        self.assertEqual(rc, 1, err[-2000:])
        self.assertEqual(seen[0]["status"], "RUNNING")
        self.assertIn("machine and versions differing: processor", seen[0]["findings"]["control_not_reproduced"])
        st = json.loads((run / "status.json").read_text())
        self.assertEqual(st["status"], "FAIL")
        self.assertIn("threads changed after LivenessNet", st["reason"])
        self.assertIn("machine and versions differing: processor", st["findings"]["control_not_reproduced"])
        log = (run / "run.log").read_text()
        self.assertEqual(log.count("FINDING: LivenessNet control not reproduced"), 1)
        self.assertEqual(log.count("FINDING:"), 1)

    def test_9f_not_evaluable_finding_before_the_after_threads_check(self):
        """AttackNetV1 non valutabile, poi thread alterati al controllo «after AttackNetV1»: FAIL, ma il finding è già
        nello status.json in corso (RUNNING) e resta nello stato finale e una sola volta in run.log."""
        def trainer(data, training, network, seed, build_model, cap_check, timer):
            if network != "AttackNetV1":
                return C.anet_trainer(data, training, network, seed, build_model, cap_check, timer)

            def bad():
                m = build_model()
                w = m.get_layer("output").get_weights()
                m.get_layer("output").set_weights([np.full_like(w[0], np.nan), w[1]])
                return m
            return C.anet_trainer(data, training, network, seed, bad, cap_check, timer)
        seen = []
        with self.after_threads_fail("AttackNetV1", seen):
            rc, out, err, run = self.go(TRAINER=trainer)
        self.assertEqual(rc, 1, err[-2000:])
        self.assertEqual(seen[0]["status"], "RUNNING")
        self.assertEqual(seen[0]["findings"]["not_evaluable"], {"AttackNetV1": "not evaluable: no finite val_loss"})
        st = json.loads((run / "status.json").read_text())
        self.assertEqual(st["status"], "FAIL")
        self.assertIn("threads changed after AttackNetV1", st["reason"])
        self.assertEqual(st["findings"]["not_evaluable"], {"AttackNetV1": "not evaluable: no finite val_loss"})
        log = (run / "run.log").read_text()
        self.assertEqual(log.count("FINDING: AttackNetV1 not evaluable: no finite val_loss"), 1)
        self.assertEqual(log.count("FINDING: AttackNet"), 1)

    def test_9c_another_machine_fails_the_control(self):
        rc, out, err, run = self.go(MACHINE=lambda: {**FIXED_M, "processor": "another processor"})
        self.assertEqual(rc, 3, err[-2000:])
        c = self.doc(run)["control"]
        self.assertEqual(c["machine_and_versions"]["differing"], ["processor"])
        self.assertEqual(c["accepted_card"]["outcome"], "reproduced")


class ZZRepositoryUntouched(unittest.TestCase):
    def test_declared_skips_only(self):
        self.assertEqual(sorted(MISSING_INPUT), ["delivered configuration", "reference architecture", "tensorflow"])


# le cartelle che il codice provato può scrivere (o che i test provano a fargli scrivere): un cambiamento qui fa fallire la
# suite; un cambiamento altrove in results/, export/ o delivery/ (un altro lavoro nello stesso checkout) è solo un avviso
GUARDED = ("results/anetcmp", "results/anetcmp_ck", "delivery/anetcmp", "delivery/x", "delivery/pad-attacknet-comparison")


def guarded_state() -> dict:
    return {g: (os.path.lexists(ROOT / g), CFR.snapshot(ROOT / g)) for g in GUARDED}


def main() -> int:
    guarded = guarded_state()
    before = CFR.snapshot(ROOT / "results", ROOT / "export", ROOT / "delivery")
    argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a != "--verbose"] + (["-v"] if "--verbose" in sys.argv else [])
    try:
        result = unittest.main(argv=argv, exit=False, verbosity=1).result
    finally:
        if WORLD is not None:
            WORLD.close()
        if CPC.WORLD is not None:
            CPC.WORLD.close()
    after = CFR.snapshot(ROOT / "results", ROOT / "export", ROOT / "delivery")
    changed = guarded != guarded_state()
    elsewhere = before != after and not changed
    skipped = [f"{t.id()}: {r}" for t, r in result.skipped]
    print(f"tests run: {result.testsRun}, failures: {len(result.failures)}, errors: {len(result.errors)}, skipped: {len(skipped)}")
    for s in skipped:
        print(f"skipped {s}")
    names = ", ".join(sorted(Path(k).name for k in set(before) ^ set(after) | {k for k in before if k in after and before[k] != after[k]})[:5])
    if changed:
        print(f"ERROR: a folder the tested code can write changed during the tests ({', '.join(GUARDED)}): {names}")
    elif elsewhere:
        print(f"WARNING: other files under results/, export/ or delivery/ changed during the tests (not written by this code): {names}")
    ok = result.wasSuccessful() and not changed
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
