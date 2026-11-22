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

Senza TensorFlow i test che costruiscono o addestrano un modello Keras, o che avviano TensorFlow in un sottoprocesso,
sono saltati, con un motivo dichiarato in MISSING_INPUT; gli altri girano.

Le cartelle temporanee del mondo sintetico stanno in TMPDIR: la suite fallisce se TMPDIR sta dentro la radice del
repository o della copia su cui gira (rilievo della revisione indipendente del 19/11); TMPDIR va tenuto fuori dalla
radice (il controllo dei mutanti lo mette in una cartella temporanea fuori dalla copia).

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
import nuaa_patience_verify as PV  # noqa: E402
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


@contextlib.contextmanager
def tensorflow_unloaded():
    """check-inputs rifiuta un processo in cui TensorFlow è già importato (FR.tf_state guarda sys.modules): i test che lo
    eseguono tolgono per la sua durata le voci di TensorFlow da sys.modules e le rimettono dopo, così non dipendono
    dall'ordine dei test. Il controllo resta vero: se check-inputs importasse TensorFlow, tf_state lo troverebbe."""
    saved = {k: v for k, v in sys.modules.items() if k == "tensorflow" or k.startswith("tensorflow.")}
    for k in saved:
        del sys.modules[k]
    try:
        yield
    finally:
        sys.modules.update(saved)


def tf_with_threads():
    """TensorFlow con i thread della configurazione (10 e 10) impostati prima di qualunque operazione, come fa il runner:
    i test che costruiscono modelli inizializzano TensorFlow nello stesso processo della prova, che poi reimposta gli
    stessi valori (nessun cambiamento, nessun errore)."""
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(P.REQUIRED_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(P.REQUIRED_THREADS)
    return tf


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
            "training": {"recipe": "r", "trial_overrides": {"max_epochs": 3}, "trainer": "t", "timing": "t", "checkpoint": "none saved",
                         "threads": {"intra_op": 10, "inter_op": 10, "rule": "t"}},
            "trial": {**P.TRIAL_UNIT, "roles": ["fitting", "dev"], "frames": {"fitting": self.counts("armB_fitting"), "dev": self.counts("dev")},
                      "networks": list(AM.NETWORKS), "inference_batch": 16, "inference": "i", "process": "p", "wall_time_cap_minutes": 60,
                      "cap_unit": P.CAP_UNIT, "cap_checks": "c", "estimate_minutes": 1, "upper_estimate_minutes": 2, "estimate": "e"},
            "baseline_check": {"reference_run": P4, "model": REF_MODEL, "card_sha256": self.card_sha, "epochs": 3, "fields": list(P.HISTORY_KEYS),
                               "difference": 0, "rule": "r", "source": "s",
                               "consequence": "if not reproduced, reported and explained before the pre-registration is closed"},
            "definitions": {"one_class": "o", "comparison": "c", "sign": "s", "subject_sensitivity": "s", "fixed_threshold": 0.5,
                            "decision": FR.DECISION_RULE, "eer_threshold": FR.THRESHOLD_CRITERION, "bootstrap": dict(SR.BOOT),
                            "conditions": [list(c) for c in D.CONDITIONS], "frame_ids": {"field": "sample_id", "rule": "r"},
                            "primary_analysis": {"patience": 4, "threshold": "fixed_0.5", "ordinary": [4, "ordinary", "fixed_0.5"],
                                                 "re_estimated": [4, "re-estimated", "fixed_0.5"],
                                                 "contrast": "re-estimated minus ordinary", "reading": "a choice of the author",
                                                 "reading_names": [D.CONTRAST_LOWER, D.CONTRAST_HIGHER, D.CONTRAST_UNRESOLVED],
                                                 "per_seed": "p", "secondary": "s"},
                            "per_seed_reading": "p"},
            "plan": {"networks": list(AM.NETWORKS), "patiences": [4, 12], "bn_states": list(D.BN_STATES), "thresholds": list(D.THRESHOLD_RULES),
                     "max_epochs": 12, "datasets": [nuaa, missing("CelebA-Spoof"), missing("Replay-Attack")], "order": "o",
                     "runs": P.nuaa_runs(nuaa), "caps": {"rule": "r", "unit": P.RUN_CAP_UNIT, "wall_multiplier": 1.5, "round_up_hours": 0.5},
                     "machine_rule": "the same machine and the same threads",
                     "costs": {"rates": "r", "training": "t", "evaluation": "e", "totals": "t"}},
            "outputs": {"trial": ["status.json", "run.log", "trial.json", "trial.md"], "plan": list(P.PLAN_OUTPUTS),
                        "compare_trials": list(P.COMPARE_OUTPUTS)},
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

def frame_ids_of(frames, split) -> list:
    """ID sintetici dei frame: split, soggetto e posizione (diversi per frame e per split, uguali per gli stessi frame)."""
    return [f"f{split}-{s}-{i}" for i, (s, _) in enumerate(frames)]


def unit(split, seed, frames, s_res, s_liv, t_res=0.5, t_liv=0.5, dataset="D", ids=None):
    """frames: lista di (soggetto, etichetta); punteggi d'attacco delle due reti; ID dei frame (sample_id) uguali per le
    due reti, quelli di frame_ids_of se non dati."""
    subj = [f for f, _ in frames]
    y = [lab for _, lab in frames]
    ids = list(ids if ids is not None else frame_ids_of(frames, split))
    return {"unit": (dataset, split, seed), "networks": {
        "ResNet-18": {"y": y, "s": list(s_res), "subject": subj, "threshold": t_res, "sample_id": list(ids)},
        "LivenessNet": {"y": y, "s": list(s_liv), "subject": subj, "threshold": t_liv, "sample_id": list(ids)}}}


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

    def test_threads(self):
        th = lambda **k: (lambda d: d["training"]["threads"].update(**k))        # noqa: E731
        self.refused(lambda d: d["training"].update(threads="not set"), "training.threads needs exactly")
        self.refused(lambda d: d["training"]["threads"].pop("inter_op"), "training.threads needs exactly")
        self.refused(lambda d: d["training"]["threads"].update(extra=1), "training.threads needs exactly")
        for bad in ("10", 10.0, 0, -10, True, None):
            self.refused(th(intra_op=bad), "training.threads.intra_op must be a positive integer")
            self.refused(th(inter_op=bad), "training.threads.inter_op must be a positive integer")
        self.refused(th(intra_op=15), "must be 10")
        self.refused(th(inter_op=8), "must be 10")
        self.refused(th(rule=""), "training.threads.rule")

    def test_definitions_added(self):
        pa = lambda **k: (lambda d: d["definitions"]["primary_analysis"].update(**k))   # noqa: E731
        self.refused(lambda d: d["definitions"]["frame_ids"].update(field="key"), "definitions.frame_ids.field")
        self.refused(lambda d: d["definitions"].pop("frame_ids"), "definitions needs exactly")
        self.refused(lambda d: d["definitions"].pop("per_seed_reading"), "definitions needs exactly")
        self.refused(pa(patience=12), "must fix patience 4")
        self.refused(pa(patience=True), "must fix patience 4")
        self.refused(pa(threshold="dev_eer"), "must fix patience 4")
        self.refused(pa(ordinary=[12, "ordinary", "fixed_0.5"]), "conditions must be those")
        self.refused(pa(re_estimated=[4, "ordinary", "fixed_0.5"]), "conditions must be those")
        self.refused(pa(ordinary=[4, "re-estimated", "fixed_0.5"], re_estimated=[4, "ordinary", "fixed_0.5"]), "conditions must be those")
        self.refused(pa(reading_names=["lower", "higher", "unresolved"]), "reading_names")
        self.refused(pa(reading="the rule"), "choice of the author")
        self.refused(pa(contrast="ordinary minus nothing"), "re-estimated minus ordinary")
        self.refused(pa(per_seed=""), "primary_analysis.per_seed")
        self.refused(pa(secondary="the reading is caused by BN"), "causal")

    def test_plan_added(self):
        def runs_swapped(d):
            d["plan"]["runs"][0]["trainings"][0:2] = d["plan"]["runs"][0]["trainings"][1::-1]
        self.refused(runs_swapped, "plan.runs must list")
        self.refused(lambda d: d["plan"]["runs"].reverse(), "plan.runs must list")
        self.refused(lambda d: d["plan"]["runs"].pop(), "plan.runs must list")
        self.refused(lambda d: d["plan"].pop("order"), "plan needs exactly")
        self.refused(lambda d: d["plan"].update(order=""), "plan.order")
        self.refused(lambda d: d["plan"]["caps"].update(wall_multiplier=0.9), "at least 1")
        self.refused(lambda d: d["plan"]["caps"].update(wall_multiplier=True), "plan.caps.wall_multiplier")
        self.refused(lambda d: d["plan"]["caps"].update(round_up_hours=0), "plan.caps.round_up_hours")
        self.refused(lambda d: d["plan"]["caps"].update(unit="hours"), "plan.caps.unit")
        self.refused(lambda d: d["plan"]["caps"].pop("rule"), "plan.caps needs exactly")
        self.refused(lambda d: d["plan"].update(machine_rule="any machine"), "same machine")
        self.refused(lambda d: d["plan"].update(machine_rule="the same machine"), "same threads")
        self.refused(lambda d: d["outputs"].update(compare_trials=["x.json"]), "outputs must be")


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
        with tensorflow_unloaded():
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
            card.write_bytes(saved + b" ")              # non dipende dalla storia scritta (TestLTrial la riscrive)
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

        out = self.w.tmp / "assemble_out"
        running = []

        def short(store, rows):
            running.append(json.loads((out / RUN / "status.json").read_text()))       # lo stato RUNNING, durante il run
            return real(store, sorted(rows, key=lambda r: r["key"])[1:])
        self.addCleanup(shutil.rmtree, out, True)
        with injected(fr={"assemble": short}):
            rc, o, err = run_main(self.w.argv("trial", out=out))
        self.assertEqual(rc, 1)
        self.assertIn("trial fitting: the keys differ from fitting_keys_sha256 of the reference card", err)
        run = out / RUN
        st = json.loads((run / "status.json").read_text())
        self.assertEqual(st["status"], "FAIL")
        want = {"requested": {"intra_op": 10, "inter_op": 10}, "effective": None, "set_by_this_runner": False, "logical_cpus": os.cpu_count()}
        self.assertEqual(running[0]["status"], "RUNNING")
        for s in (running[0], st):                    # fermata prima dell'import di TensorFlow: thread richiesti, non impostati
            self.assertEqual({k: s["threads"][k] for k in want}, want)
            self.assertEqual(sorted(s["threads"]["environment"]), ["TF_NUM_INTEROP_THREADS", "TF_NUM_INTRAOP_THREADS"])
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
    want = {"intra_op": 10, "inter_op": 10}
    for v in nets.values():
        v.setdefault("training_overhead", {"seconds": 5.0, "cpu_seconds": 10.0} if v["epochs"] else None)
        if v["epochs"]:
            v.setdefault("threads", dict(want))
    code = {**cfg["code"], "scripts/pad_arch_comparison.py": sha((ROOT / "scripts/pad_arch_comparison.py").read_bytes())}
    doc = {"schema": P.TRIAL_SCHEMA, "run_id": RUN, "status": status, "unit": {**P.TRIAL_UNIT, "frames": cfg["trial"]["frames"]}, "networks": nets,
           "config_sha256": sha(w.cfg_path.read_bytes()), "code_sha256": code, "commit": STATE["commit"],
           "machine": {"processor": "Test CPU", "logical_cpus": 15, "ram_gb": 24.0, "os": "macOS 27.2.1",
                       "core_levels": [{"name": "Super", "physical_cores": 5}, {"name": "Performance", "physical_cores": 10}],
                       "architecture": "arm64"},
           "threads": {"requested": dict(want), "effective": dict(want), "set_by_this_runner": True,
                       "environment": {"TF_NUM_INTRAOP_THREADS": None, "TF_NUM_INTEROP_THREADS": None}, "logical_cpus": 15}, **(over or {})}
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
        self.assertEqual({k: cfg["training"]["threads"][k] for k in P.THREAD_KEYS}, {"intra_op": 10, "inter_op": 10})
        self.assertEqual(cfg["definitions"]["primary_analysis"]["ordinary"], [4, "ordinary", "fixed_0.5"])
        self.assertEqual(cfg["definitions"]["primary_analysis"]["re_estimated"], [4, "re-estimated", "fixed_0.5"])
        self.assertEqual(len(cfg["plan"]["runs"]), 2)
        self.assertEqual(sum(len(r["trainings"]) for r in cfg["plan"]["runs"]), 48)
        self.assertEqual(cfg["test"], P.TEST_READ)


# ============================================================================ TensorFlow: modelli

class TestKModels(unittest.TestCase):
    def setUp(self):
        need_tf(self)
        tf = tf_with_threads()
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
        tf_with_threads()
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
        """Prova completa; in più: i thread impostati con i valori della configurazione prima di ogni addestramento (le
        chiamate di impostazione e del trainer registrate nell'ordine)."""
        import tensorflow as tf
        from unittest import mock
        log = []
        real = {k: getattr(tf.config.threading, k) for k in ("set_intra_op_parallelism_threads", "set_inter_op_parallelism_threads")}

        def trainer(data, training, network, *a, **k):
            log.append(f"train {network}")
            return P.trial_trainer(data, training, network, *a, **k)
        with mock.patch.object(tf.config.threading, "set_intra_op_parallelism_threads",
                               lambda n: log.append(f"intra {n}") or real["set_intra_op_parallelism_threads"](n)), \
                mock.patch.object(tf.config.threading, "set_inter_op_parallelism_threads",
                                  lambda n: log.append(f"inter {n}") or real["set_inter_op_parallelism_threads"](n)):
            rc, out, err, run = self.go(TRAINER=trainer)
        self.assertEqual(rc, 0, err[-3000:])
        self.assertEqual(sorted(os.listdir(run)), ["run.log", "status.json", "trial.json", "trial.md"])
        st = self.assert_final_status(run, "PASS")
        self.assertEqual(st["trial_json_sha256"], sha((run / "trial.json").read_bytes()))
        once = json.loads((run / "trial.json").read_text())
        self.assertEqual({k: st[k] for k in ("finished", "wall_seconds", "process_cpu_seconds")},
                         {k: once[k] for k in ("finished", "wall_seconds", "process_cpu_seconds")})      # misurati una volta
        self.assertIn(f"process CPU time (all threads) {once['process_cpu_seconds']:.0f} s", (run / "trial.md").read_text())
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
        want = {"intra_op": 10, "inter_op": 10}
        self.assertEqual({k: doc["threads"][k] for k in ("requested", "effective", "set_by_this_runner")}, {"requested": want, "effective": want,
                                                                                                         "set_by_this_runner": True})
        self.assertEqual(doc["threads"]["logical_cpus"], os.cpu_count())
        self.assertEqual(sorted(doc["threads"]["environment"]), ["TF_NUM_INTEROP_THREADS", "TF_NUM_INTRAOP_THREADS"])
        self.assertEqual([doc["networks"][n]["threads"] for n in AM.NETWORKS], [want, want])
        self.assertEqual(st["threads"], doc["threads"])
        self.assertIn("TensorFlow threads set by this runner: requested intra-op 10 and inter-op 10", (run / "trial.md").read_text())
        self.assertEqual(log, ["intra 10", "inter 10", "train LivenessNet", "train ResNet-18"])
        self.assertEqual(doc["schema"], P.TRIAL_SCHEMA)
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
        self.assertEqual(doc["networks"]["ResNet-18"]["threads"], {"intra_op": 10, "inter_op": 10})   # riletti prima della rete
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
        self.assertEqual(rec["threads"], {"intra_op": 10, "inter_op": 10})
        # i thread riletti dopo la rete: un getter che cambia durante il training ferma la prova a fine rete
        patcher = mock.patch.object(tf.config.threading, "get_inter_op_parallelism_threads", lambda: 3)

        def changing(*a, **k):
            out = fake(*a, **k)
            patcher.start()
            return out
        try:
            with injected(p={"TRAINER": changing}), self.assertRaises(FR.Failure) as cm, contextlib.redirect_stdout(io.StringIO()):
                P.trial_network(self.cfg, self.pcfg, "LivenessNet", None, self.fit, self.dev, SR.Budget(60), {"epochs": [], "inference": {}})
        finally:
            patcher.stop()
        self.assertIn("after LivenessNet", str(cm.exception))

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
        self.assertEqual({k: st["threads"][k] for k in ("requested", "effective", "set_by_this_runner")},
                         {"requested": {"intra_op": 10, "inter_op": 10}, "effective": {"intra_op": 10, "inter_op": 10}, "set_by_this_runner": True})
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
        self.assertEqual(sorted(m), ["architecture", "core_levels", "logical_cpus", "os", "processor", "ram_gb"])
        self.assertEqual(P.output_problems(json.dumps(m), FR.Inputs()), [])

    def test_core_levels_are_named_by_the_system(self):
        """I gruppi di core portano il nome dato dal sistema, non un nome fisso (difetto della PR #29: «performance» ed
        «efficiency» anche sull'Apple M5 Pro, dove il sistema li chiama «Super» e «Performance»)."""
        m = P.machine()
        self.assertIsInstance(m["core_levels"], list)
        for lv in m["core_levels"]:
            self.assertEqual(sorted(lv), ["name", "physical_cores"])
        self.assertNotIn("performance_cores", m)
        self.assertNotIn("efficiency_cores", m)
        two = {"core_levels": [{"name": "Super", "physical_cores": 5}, {"name": "Performance", "physical_cores": 10}]}
        self.assertEqual(P.core_levels_text(two), "5 Super, 10 Performance")
        self.assertEqual(P.core_levels_text({"core_levels": []}), "not reported")
        self.assertEqual(P.core_levels_text({}), "not reported")

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


# ============================================================================ ID dei frame (richiesta del coordinatore, 19/11/2026)

def two_splits(seeds=(42,), liv=None):
    """Due split (soggetti a, b, c e d, e, f), ResNet-18 perfetta e LivenessNet costante (o liv(frames)), per i seed dati."""
    units = []
    for split, subs in ((1, list("abc")), (2, list("def"))):
        f = frames_of(subs)
        for seed in seeds:
            units.append(unit(split, seed, f, perfect(f), liv(f) if liv else [0.2] * len(f)))
    return units


class TestBFrameIds(unittest.TestCase):
    def refused(self, units, needle, fn=None):
        with self.assertRaises(FR.Failure) as cm:
            (fn or D.compare_condition)(units)
        msg = str(cm.exception)
        self.assertIn(needle, msg)
        self.assertNotIn("other-frame", msg)
        self.assertFalse(any(f"f{s}-" in msg for s in (1, 2)), msg)        # nessun ID di frame nel messaggio
        return msg

    def test_networks_with_the_same_subjects_and_labels_and_one_identifier_different(self):
        """(a) Fra le due reti: stessi soggetti, stesse etichette, stesso ordine, un solo ID diverso -> il confronto si ferma
        per il controllo degli ID (messaggio degli ID, non quello dei frame)."""
        units = two_splits()
        n = units[0]["networks"]
        n["LivenessNet"]["sample_id"][3] = "other-frame"
        self.assertEqual((n["ResNet-18"]["y"], n["ResNet-18"]["subject"]), (n["LivenessNet"]["y"], n["LivenessNet"]["subject"]))
        self.refused(units, "the two networks are not scored on the same frame identifiers in the same order")
        self.refused(units, "the two networks are not scored on the same frame identifiers in the same order", D.leave_one_subject_out)

    def test_same_identifiers_in_another_order(self):
        """(c) Stessi ID in ordine diverso, con soggetto ed etichetta coerenti (due frame dello stesso soggetto e classe
        scambiati): Failure."""
        units = two_splits()
        ids = units[0]["networks"]["LivenessNet"]["sample_id"]
        self.assertEqual(frames_of(list("abc"))[0], frames_of(list("abc"))[1])
        ids[0], ids[1] = ids[1], ids[0]
        self.refused(units, "same frame identifiers in the same order")

    def test_identifiers_missing_duplicated_empty_or_of_another_length(self):
        """(d) Per ciascuna rete: campo mancante, un ID in meno o in più, duplicato, vuoto, non testuale, non una lista,
        array a due dimensioni, campo in più."""
        cases = ((lambda n: n.pop("sample_id"), "must hold exactly the fields"),
                 (lambda n: n.update(sample_id=n["sample_id"][:-1]), "frame identifiers for"),
                 (lambda n: n.update(sample_id=n["sample_id"] + ["x"]), "frame identifiers for"),
                 (lambda n: n["sample_id"].__setitem__(2, n["sample_id"][1]), "duplicated frame identifiers"),
                 (lambda n: n["sample_id"].__setitem__(0, " "), "non-empty text"),
                 (lambda n: n["sample_id"].__setitem__(0, 7), "non-empty text"),
                 (lambda n: n.update(sample_id="abc"), "must be a list"),
                 (lambda n: n.update(sample_id=np.array([n["sample_id"]])), "must be a list"),
                 (lambda n: n.update(extra=1), "must hold exactly the fields"))
        for mutate, needle in cases:
            for net in AM.NETWORKS:
                units = two_splits()
                mutate(units[0]["networks"][net])
                self.refused(units, needle)

    def test_units_of_a_split_on_other_frames(self):
        """(e) Unità dello stesso split (seed diversi): un ID diverso; gli stessi frame in un altro ordine (ID, soggetti ed
        etichette permutati insieme, in modo coerente) -> Failure; gli stessi frame -> passa."""
        units = two_splits(seeds=(42, 43))
        for n in AM.NETWORKS:
            units[1]["networks"][n]["sample_id"][0] = "other-frame"
        self.refused(units, "two units of ('D', 1) are not scored on the same frames")
        units = two_splits(seeds=(42, 43))
        f = list(reversed(frames_of(list("abc"))))
        units[1] = unit(1, 43, f, perfect(f), [0.2] * len(f), ids=list(reversed(frame_ids_of(frames_of(list("abc")), 1))))
        self.refused(units, "two units of ('D', 1) are not scored on the same frames")
        self.assertEqual(D.compare_condition(two_splits(seeds=(42, 43)))["acer_counts"]["units"], 4)

    def test_identifier_in_two_splits(self):
        units = two_splits()
        for n in AM.NETWORKS:
            units[1]["networks"][n]["sample_id"][0] = units[0]["networks"]["ResNet-18"]["sample_id"][0]
        self.refused(units, "also appears in")

    def test_good_case_numbers_and_public_rows(self):
        """(f) Il caso buono passa con gli stessi numeri di prima; nelle righe il numero dei frame e l'impronta degli ID,
        nessun ID; ID dati come array di testi accettati."""
        units = two_splits(seeds=(42, 43))
        out = D.compare_condition(units)
        self.assertEqual(out["mean_delta"], {"acer": -0.5, "apcer": -1.0, "bpcer": 0.0})
        self.assertEqual(out["acer_counts"], {"resnet18_lower": 4, "livenessnet_lower": 0, "ties": 0, "undefined": 0, "units": 4})
        self.assertEqual((out["interval"]["acer"]["low"], out["interval"]["acer"]["high"]), (-0.5, -0.5))
        row = out["rows"][0]
        self.assertEqual(row["n_frames"], 12)
        self.assertEqual(row["frame_ids_sha256"], sha("\n".join(sorted(frame_ids_of(frames_of(list("abc")), 1))).encode()))
        self.assertEqual(D.ids_sha256(["b", "a"]), sha(b"a\nb"))
        public = repr({k: v for k, v in out.items() if k != "prepared"})
        self.assertNotIn("f1-", public)
        self.assertNotIn("f2-", public)
        u = two_splits()
        for n in AM.NETWORKS:
            for x in u:
                x["networks"][n]["sample_id"] = np.array(x["networks"][n]["sample_id"])
        self.assertEqual(D.compare_condition(u)["mean_delta"]["acer"], -0.5)


class TestCFrameIdsBetweenConditions(unittest.TestCase):
    def conditions(self):
        a = two_splits()
        b = two_splits(liv=perfect)
        return a, b

    def test_conditions_with_the_same_subjects_and_labels_and_one_identifier_different(self):
        """(b) Fra due condizioni: stessi soggetti, etichette e conteggi per classe in ogni unità, un ID diverso (lo stesso
        nelle due reti della seconda condizione) -> il confronto si ferma per il controllo degli ID."""
        ua, ub = self.conditions()
        for n in AM.NETWORKS:
            ub[0]["networks"][n]["sample_id"][5] = "other-frame"
        a, b = D.compare_condition(ua), D.compare_condition(ub)
        x, y = a["prepared"][0], b["prepared"][0]
        self.assertEqual(x["frames"][1:], y["frames"][1:])
        self.assertEqual([(c["subjects"], list(c["na"]), list(c["nb"])) for c in (x["counts"]["ResNet-18"], y["counts"]["ResNet-18"])][0],
                         [(c["subjects"], list(c["na"]), list(c["nb"])) for c in (x["counts"]["ResNet-18"], y["counts"]["ResNet-18"])][1])
        with self.assertRaises(FR.Failure) as cm:
            D.compare_conditions(a, b)
        self.assertIn("the two conditions are not scored on the same frame identifiers in the same order", str(cm.exception))
        self.assertNotIn("other-frame", str(cm.exception))

    def test_conditions_with_the_same_identifiers_in_another_order(self):
        ua, ub = self.conditions()
        for n in AM.NETWORKS:
            ids = ub[1]["networks"][n]["sample_id"]
            ids[0], ids[1] = ids[1], ids[0]                  # stesso soggetto e stessa etichetta
        with self.assertRaises(FR.Failure) as cm:
            D.compare_conditions(D.compare_condition(ua), D.compare_condition(ub))
        self.assertIn("same frame identifiers in the same order", str(cm.exception))

    def test_conditions_with_the_same_identifiers_and_other_subjects(self):
        """Stessi ID, un frame con un altro soggetto della stessa classe (conteggi per soggetto diversi): il controllo dei
        soggetti e delle etichette, a parte da quello degli ID."""
        ua, ub = self.conditions()
        for n in AM.NETWORKS:
            ub[0]["networks"][n]["subject"][0] = "b"
        with self.assertRaises(FR.Failure) as cm:
            D.compare_conditions(D.compare_condition(ua), D.compare_condition(ub))
        self.assertIn("is not scored on the same test frames (subjects and labels)", str(cm.exception))


# ============================================================================ (e) analisi primaria

def primary_conditions(seed43_like_ordinary=True):
    """Le 8 condizioni sintetiche: a (4, ordinary, 0,5) Δ = −1/2 in ogni unità (ResNet-18 perfetta, LivenessNet costante
    0,2: APCER 1); a (4, re-estimated, 0,5) Δ = −1/4 nelle unità del seed 42 (LivenessNet sbaglia un attacco su due per
    soggetto: APCER 1/2) e −1/2 in quelle del seed 43; le altre sei Δ = 0 (due reti perfette)."""
    def half(f):
        out, seen = [], {}
        for s, lab in f:
            if lab:
                seen[s] = seen.get(s, 0) + 1
                out.append(0.9 if seen[s] % 2 else 0.2)
            else:
                out.append(0.1)
        return out
    conds = {}
    for c in D.CONDITIONS:
        units = []
        for split, subs in ((1, list("abc")), (2, list("def"))):
            f = frames_of(subs)
            for seed in (42, 43):
                if c == D.PRIMARY_ORDINARY:
                    liv = [0.2] * len(f)
                elif c == D.PRIMARY_RE_ESTIMATED:
                    liv = half(f) if seed == 42 else [0.2] * len(f)
                else:
                    liv = perfect(f)
                units.append(unit(split, seed, f, perfect(f), liv))
        conds[c] = units
    return conds


class TestNPrimaryAnalysis(unittest.TestCase):
    def test_constants(self):
        self.assertEqual((D.PRIMARY_ORDINARY, D.PRIMARY_RE_ESTIMATED), ((4, "ordinary", "fixed_0.5"), (4, "re-estimated", "fixed_0.5")))
        self.assertTrue(D.PRIMARY_ORDINARY in D.CONDITIONS and D.PRIMARY_RE_ESTIMATED in D.CONDITIONS)

    def test_known_gaps_and_contrast(self):
        r = D.primary_analysis(primary_conditions())
        self.assertEqual(r["conditions"], {"ordinary": [4, "ordinary", "fixed_0.5"], "re-estimated": [4, "re-estimated", "fixed_0.5"]})
        self.assertEqual(r["gap_ordinary"]["mean_delta"], {"acer": -0.5, "apcer": -1.0, "bpcer": 0.0})
        self.assertEqual(r["gap_re_estimated"]["mean_delta"], {"acer": -0.375, "apcer": -0.75, "bpcer": 0.0})
        self.assertEqual(r["contrast"]["mean"], {"acer": 0.125, "apcer": 0.25, "bpcer": 0.0})
        self.assertEqual((r["contrast"]["interval"]["acer"]["low"], r["contrast"]["interval"]["acer"]["high"]), (0.125, 0.125))
        self.assertEqual(r["contrast"]["reading"], D.CONTRAST_HIGHER)
        self.assertEqual(r["contrast"]["units"], 4)
        by_unit = {row["unit"]: row for row in r["rows"]}
        self.assertEqual(by_unit[("D", 1, 42)]["contrast"]["acer"], Fraction(1, 4))
        self.assertEqual(by_unit[("D", 1, 42)]["delta_re_estimated"]["acer"], Fraction(-1, 4))
        self.assertEqual(by_unit[("D", 1, 42)]["delta_ordinary"]["acer"], Fraction(-1, 2))
        self.assertEqual(by_unit[("D", 2, 43)]["contrast"]["acer"], Fraction(0))
        self.assertEqual(by_unit[("D", 1, 42)]["n_frames"], 12)
        self.assertNotIn("f1-", repr(r))

    def test_per_seed_apart(self):
        r = D.primary_analysis(primary_conditions())
        self.assertEqual(sorted(r["per_seed"]), [42, 43])
        s42, s43 = r["per_seed"][42], r["per_seed"][43]
        self.assertEqual((s42["contrast"]["mean"]["acer"], s42["contrast"]["reading"]), (0.25, D.CONTRAST_HIGHER))
        self.assertEqual((s43["contrast"]["mean"]["acer"], s43["contrast"]["reading"]), (0.0, D.CONTRAST_UNRESOLVED))
        self.assertEqual((s42["gap_ordinary"]["mean_delta"]["acer"], s42["gap_re_estimated"]["mean_delta"]["acer"]), (-0.5, -0.25))
        self.assertEqual((s43["gap_ordinary"]["mean_delta"]["acer"], s43["gap_re_estimated"]["mean_delta"]["acer"]), (-0.5, -0.5))
        self.assertEqual(s42["contrast"]["units"], 2)
        self.assertEqual(r["contrast"]["reading"], D.CONTRAST_HIGHER)          # la lettura viene dai seed insieme

    def test_order_chosen_by_the_function(self):
        """L'ordine della sottrazione non dipende da chi chiama: stesse condizioni in ordine diverso danno lo stesso
        risultato; scambiare il contenuto delle due chiavi primarie inverte il segno del contrasto."""
        conds = primary_conditions()
        rev = dict(reversed(list(conds.items())))
        self.assertEqual(D.primary_analysis(rev)["contrast"]["mean"], D.primary_analysis(conds)["contrast"]["mean"])
        sw = dict(conds)
        sw[D.PRIMARY_ORDINARY], sw[D.PRIMARY_RE_ESTIMATED] = conds[D.PRIMARY_RE_ESTIMATED], conds[D.PRIMARY_ORDINARY]
        r = D.primary_analysis(sw)
        self.assertEqual(r["contrast"]["mean"]["acer"], -0.125)
        self.assertEqual(r["contrast"]["reading"], D.CONTRAST_LOWER)
        only = {k: conds[k] for k in (D.PRIMARY_ORDINARY, D.PRIMARY_RE_ESTIMATED)}
        self.assertEqual(D.primary_analysis(only)["contrast"]["mean"]["acer"], 0.125)

    def test_missing_or_unknown_condition(self):
        conds = primary_conditions()
        for k in (D.PRIMARY_ORDINARY, D.PRIMARY_RE_ESTIMATED):
            c = dict(conds)
            c.pop(k)
            with self.assertRaises(FR.Failure) as cm:
                D.primary_analysis(c)
            self.assertIn("is missing", str(cm.exception))
        for bad in ({**conds, (4, "ordinary", "youden"): conds[D.PRIMARY_ORDINARY]}, {**conds, (4, "ordinary"): []},
                    [conds[D.PRIMARY_ORDINARY]]):
            with self.assertRaises(FR.Failure):
                D.primary_analysis(bad)

    def test_frame_identifiers_differ_between_the_two_conditions(self):
        conds = primary_conditions()
        for u in conds[D.PRIMARY_RE_ESTIMATED][2:]:             # le due unità dello split 2, nelle due reti
            for n in AM.NETWORKS:
                u["networks"][n]["sample_id"][4] = "other-frame"
        with self.assertRaises(FR.Failure) as cm:
            D.primary_analysis(conds)
        self.assertIn("the two conditions are not scored on the same frame identifiers", str(cm.exception))

    def test_contrast_reading_bounds(self):
        self.assertEqual(D.contrast_reading({"low": -0.3, "high": -0.01}), D.CONTRAST_LOWER)
        self.assertEqual(D.contrast_reading({"low": 0.01, "high": 0.3}), D.CONTRAST_HIGHER)
        self.assertEqual(D.contrast_reading({"low": -0.3, "high": 0.0}), D.CONTRAST_UNRESOLVED)
        self.assertEqual(D.contrast_reading({"low": 0.0, "high": 0.3}), D.CONTRAST_UNRESOLVED)
        self.assertEqual(D.contrast_reading({"low": None, "high": None}), D.CONTRAST_UNRESOLVED)
        self.assertEqual(len({D.CONTRAST_LOWER, D.CONTRAST_HIGHER, D.CONTRAST_UNRESOLVED}), 3)
        for name in (D.CONTRAST_LOWER, D.CONTRAST_HIGHER):
            self.assertIn("ResNet-18 - LivenessNet", name)
            self.assertIn("re-estimated BN", name)


# ============================================================================ (f) lettura per seed

class TestOPerSeedReading(unittest.TestCase):
    @staticmethod
    def e(d, lo, hi):
        return {"difference": d, "low": lo, "high": hi}

    def test_rule(self):
        e = self.e
        r = D.per_seed_reading([e(-0.2, -0.3, -0.1)] * 3)
        self.assertEqual((r["verdict"], r["direction"]), ("consistent", D.SIGN_RESNET_LOWER))
        r = D.per_seed_reading([e(0.2, 0.1, 0.3), e(0.1, 0.05, 0.2), e(0.3, 0.01, 0.5)])
        self.assertEqual((r["verdict"], r["direction"]), ("consistent", D.SIGN_LIVENESSNET_LOWER))
        for seeds, reason in (([e(-0.2, -0.3, -0.1), e(-0.2, -0.3, 0.0), e(-0.2, -0.3, -0.1)], "does not exclude 0"),
                              ([e(0.2, 0.0, 0.3), e(0.2, 0.1, 0.3), e(0.2, 0.1, 0.3)], "does not exclude 0"),
                              ([e(-0.2, -0.3, -0.1), e(0.2, 0.1, 0.3), e(-0.2, -0.3, -0.1)], "sign differs"),
                              ([e(-0.2, -0.3, -0.1), e(0.0, -0.1, 0.1), e(-0.2, -0.3, -0.1)], "zero difference"),
                              ([e(-0.2, -0.3, -0.1), None, e(-0.2, -0.3, -0.1)], "no value"),
                              ([e(-0.2, -0.3, -0.1), e(None, None, None), e(-0.2, -0.3, -0.1)], "no difference"),
                              ([e(-0.2, -0.3, -0.1), e(-0.2, None, None), e(-0.2, -0.3, -0.1)], "no bootstrap interval")):
            r = D.per_seed_reading(seeds)
            self.assertEqual((r["verdict"], r["direction"]), ("inconclusive", None))
            self.assertIn(reason, r["reason"])
        for bad in ([], [e(-0.2, -0.3, -0.1)], [e(-0.2, -0.3, -0.1)] * 4, None):
            with self.assertRaises(FR.Failure):
                D.per_seed_reading(bad)
        self.assertEqual(D.PLAN_SEEDS, (42, 43, 44))

    def test_same_as_the_rule_of_pr21(self):
        """Stessi verdetti e stesse ragioni di nuaa_session_ablation_run.reading_verdict su casi casuali."""
        rng = np.random.default_rng(5)
        comp = SR.COMPARISONS[0]
        for _ in range(300):
            seeds = []
            for _ in range(3):
                kind = rng.integers(0, 6)
                d = float(rng.choice([-0.2, -0.1, 0.0, 0.1, 0.2]))
                lo, hi = sorted(float(x) for x in rng.choice([-0.3, -0.05, 0.0, 0.05, 0.3], 2))
                seeds.append(None if kind == 0 else self.e(None, lo, hi) if kind == 1 else self.e(d, None, None) if kind == 2 else self.e(d, lo, hi))
            mine, theirs = D.per_seed_reading(seeds), SR.reading_verdict(comp, "acer_05", seeds)
            # stessa regola e stesse ragioni; un seed assente qui è «no value» (non c'è un valore unito sui fold)
            reason = {"a seed has no value": "a seed has no pooled value (incomplete run)"}.get(mine["reason"], mine["reason"])
            self.assertEqual((mine["verdict"], reason), (theirs["verdict"], theirs["reason"]))
            if theirs["verdict"] == "consistent":
                self.assertEqual(mine["direction"], D.SIGN_RESNET_LOWER if "lower" in theirs["direction"] else D.SIGN_LIVENESSNET_LOWER)

    def test_reported_by_compare_condition(self):
        out = D.compare_condition(two_splits(seeds=(42, 43, 44)))
        self.assertEqual((out["per_seed_reading"]["verdict"], out["per_seed_reading"]["direction"]), ("consistent", D.SIGN_RESNET_LOWER))
        for seeds in ((42,), (42, 43), (43, 44)):          # seed mancanti: mai «consistent»
            r = D.compare_condition(two_splits(seeds=seeds))["per_seed_reading"]
            self.assertEqual((r["verdict"], r["reason"]), ("inconclusive", "a seed has no value"))
        units = two_splits(seeds=(42, 43, 44))
        units[4] = unit(2, 43, frames_of(list("def")), perfect(frames_of(list("def"))), perfect(frames_of(list("def"))))
        units[1] = unit(1, 43, frames_of(list("abc")), perfect(frames_of(list("abc"))), perfect(frames_of(list("abc"))))
        out = D.compare_condition(units)
        self.assertEqual(out["per_seed"][43]["mean_delta"]["acer"], 0.0)
        self.assertEqual((out["per_seed_reading"]["verdict"], out["per_seed_reading"]["reason"]),
                         ("inconclusive", "a seed has a zero difference"))
        src = inspect.getsource(D.sign_of) + inspect.getsource(D.compare_conditions)
        self.assertNotIn("per_seed_reading", src)


# ============================================================================ thread di TensorFlow (modulo finto)

class FakeThreading:
    def __init__(self, raise_on_set=False, offset=0):
        self.calls, self.vals, self.raise_on_set, self.offset = [], {"intra": 0, "inter": 0}, raise_on_set, offset

    def set_intra_op_parallelism_threads(self, n):
        self.calls.append(("intra", n))
        if self.raise_on_set:
            raise RuntimeError("Intra op parallelism cannot be modified after initialization.")
        self.vals["intra"] = n

    def set_inter_op_parallelism_threads(self, n):
        self.calls.append(("inter", n))
        if self.raise_on_set:
            raise RuntimeError("Inter op parallelism cannot be modified after initialization.")
        self.vals["inter"] = n

    def get_intra_op_parallelism_threads(self):
        return self.vals["intra"] + self.offset

    def get_inter_op_parallelism_threads(self):
        return self.vals["inter"]


def fake_tf(**k):
    return types.SimpleNamespace(config=types.SimpleNamespace(threading=FakeThreading(**k)))


THREAD_CFG = {"training": {"threads": {"intra_op": 10, "inter_op": 10, "rule": "r"}}}
NO_THREAD_ENV = {"TF_NUM_INTRAOP_THREADS": None, "TF_NUM_INTEROP_THREADS": None}


class TestHThreads(unittest.TestCase):
    """Prima delle classi che importano TensorFlow (K, L): check-inputs rifiuta un processo in cui TensorFlow è già importato."""
    def setUp(self):
        from unittest import mock
        self.env = mock.patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        for v in P.TF_THREAD_ENV.values():
            os.environ.pop(v, None)

    def test_set_with_the_configuration_values(self):
        tf = fake_tf()
        rec = P.set_threads(tf, THREAD_CFG)
        self.assertEqual(tf.config.threading.calls, [("intra", 10), ("inter", 10)])
        self.assertEqual(rec, {"requested": {"intra_op": 10, "inter_op": 10}, "effective": {"intra_op": 10, "inter_op": 10},
                               "set_by_this_runner": True, "environment": NO_THREAD_ENV, "logical_cpus": os.cpu_count()})
        self.assertEqual(P.check_threads(tf, THREAD_CFG, "x"), {"intra_op": 10, "inter_op": 10})

    def test_getter_discordant(self):
        with self.assertRaises(FR.Failure) as cm:
            P.set_threads(fake_tf(offset=1), THREAD_CFG)
        self.assertIn("differ from training.threads", str(cm.exception))
        tf = fake_tf()
        tf.config.threading.vals.update(intra=10, inter=4)
        with self.assertRaises(FR.Failure):
            P.check_threads(tf, THREAD_CFG, "after LivenessNet")

    def test_runtime_error_is_a_failure(self):
        with self.assertRaises(FR.Failure) as cm:
            P.set_threads(fake_tf(raise_on_set=True), THREAD_CFG)
        self.assertIn("cannot be set", str(cm.exception))

    def test_environment_variables(self):
        os.environ["TF_NUM_INTRAOP_THREADS"] = "10"
        os.environ["TF_NUM_INTEROP_THREADS"] = " 10 "
        rec = P.set_threads(fake_tf(), THREAD_CFG)
        self.assertEqual(rec["environment"], {"TF_NUM_INTRAOP_THREADS": "10", "TF_NUM_INTEROP_THREADS": " 10 "})
        for var in P.TF_THREAD_ENV.values():
            for bad in ("8", "", "ten", "10.0"):
                os.environ.pop("TF_NUM_INTRAOP_THREADS", None)
                os.environ.pop("TF_NUM_INTEROP_THREADS", None)
                os.environ[var] = bad
                tf = fake_tf()
                with self.assertRaises(FR.Failure) as cm:
                    P.set_threads(tf, THREAD_CFG)
                self.assertIn("contradicts training.threads", str(cm.exception))
                self.assertEqual(tf.config.threading.calls, [])          # rifiuto prima di impostare
        os.environ.pop("TF_NUM_INTRAOP_THREADS", None)
        os.environ.pop("TF_NUM_INTEROP_THREADS", None)
        self.assertEqual(P.thread_environment(THREAD_CFG), NO_THREAD_ENV)

    def test_trial_and_check_inputs_refuse_the_environment_before_any_work(self):
        w = world()
        out = w.tmp / "env_out"
        self.addCleanup(shutil.rmtree, out, True)
        os.environ["TF_NUM_INTEROP_THREADS"] = "15"
        with injected(**hooks(w)):
            rc, o, err = run_main(w.argv("trial", out=out))
            self.assertEqual(rc, 1)
            self.assertIn("TF_NUM_INTEROP_THREADS='15' contradicts training.threads.inter_op", err)
            self.assertFalse(out.exists())
            with tensorflow_unloaded():
                rc, o, err = run_main(w.argv("check-inputs"))
                self.assertEqual(rc, 1)
                self.assertIn("contradicts", err)
                os.environ["TF_NUM_INTEROP_THREADS"] = "10"
                rc, o, err = run_main(w.argv("check-inputs"))
        self.assertEqual(rc, 0, err)
        self.assertIn("TF_NUM_INTEROP_THREADS '10'", o)

    def test_set_threads_is_the_only_place_that_sets_them(self):
        """Controllo sul solo testo del codice (inspect.getsource), non sul comportamento: un'unica chiamata di
        impostazione e, in cmd_trial, l'import di TensorFlow prima di set_threads e set_threads prima di trial_network.
        L'ordine effettivo delle chiamate rispetto al training è provato con TensorFlow in
        TestLTrial.test_trial_pass_then_plan_then_refused_again."""
        src = inspect.getsource(P)
        self.assertEqual(src.count("set_intra_op_parallelism_threads("), 1)
        self.assertEqual(src.count("set_inter_op_parallelism_threads("), 1)
        trial = inspect.getsource(P.cmd_trial)
        self.assertLess(trial.index("import tensorflow as tf"), trial.index("set_threads(tf, cfg)"))
        self.assertLess(trial.index("set_threads(tf, cfg)"), trial.index("trial_network("))


# ============================================================================ prima epoca diversa

def baseline_cfg():
    return {"baseline_check": {"epochs": 3, "reference_run": P4, "model": REF_MODEL, "rule": "r", "consequence": "c"}}


def epochs_of(h: dict, n: int) -> list:
    return [{"epoch": i + 1, **{k: h[k][i] for k in P.HISTORY_KEYS}} for i in range(n)]


class TestQFirstDifferingEpoch(unittest.TestCase):
    def test_first_differing_epoch_is_not_the_epoch_of_the_largest_difference(self):
        r = P.history_differences([1.0, 2.0, 3.0], [1.5, 4.0, 3.0])
        self.assertEqual(r, {"epochs_compared": 3, "first_differing_epoch": 1, "max_abs_difference": 2.0, "epoch_of_max_abs_difference": 2})
        r = P.history_differences([1.0, 2.0, 3.0, 4.0], [1.0, 2.5, 3.0, 4.5])     # stesso massimo in due epoche: la prima
        self.assertEqual((r["first_differing_epoch"], r["epoch_of_max_abs_difference"]), (2, 2))
        self.assertEqual(P.history_differences([1.0, 2.0], [1.0, 2.0, 7.0])["epochs_compared"], 2)

    def test_equal_histories(self):
        self.assertEqual(P.history_differences([0.5, 0.25], [0.5, 0.25]),
                         {"epochs_compared": 2, "first_differing_epoch": None, "max_abs_difference": 0.0, "epoch_of_max_abs_difference": None})

    def test_baseline_check_field_by_field(self):
        """La forma della prova 20261118-ARCHTRIAL-0858ae7: loss diversa dall'epoca 1 con il massimo all'epoca 2,
        val_loss diversa solo all'epoca 3, accuracy uguale."""
        ref = {"loss": [0.7, 0.6, 0.5, 0.4], "accuracy": [0.5, 0.6, 0.7, 0.8], "val_loss": [0.71, 0.69, 0.65, 0.6],
               "val_accuracy": [0.5, 0.5, 0.75, 0.8], "lr": [0.001] * 4}
        new = json.loads(json.dumps(ref))
        new["loss"][0] += 0.001
        new["loss"][1] += 0.01
        new["val_loss"][2] += 1e-6
        new["lr"][3] = 0.5                                   # epoca 4 fuori dalle 3 confrontate
        b = P.baseline_check(baseline_cfg(), {"history": ref}, {"epochs": epochs_of(new, 3)})
        self.assertEqual(b["outcome"], "not reproduced")
        self.assertEqual(b["first_differing_epoch"], {"loss": 1, "accuracy": None, "val_loss": 3, "val_accuracy": None, "lr": None})
        self.assertEqual(b["epoch_of_max_abs_difference"], {"loss": 2, "accuracy": None, "val_loss": 3, "val_accuracy": None, "lr": None})
        self.assertAlmostEqual(b["max_abs_difference"]["loss"], 0.01, places=12)
        self.assertEqual(b["max_abs_difference"]["accuracy"], 0.0)
        b = P.baseline_check(baseline_cfg(), {"history": ref}, {"epochs": epochs_of(ref, 3)})
        self.assertEqual((b["outcome"], set(b["first_differing_epoch"].values()), set(b["max_abs_difference"].values())),
                         ("reproduced", {None}, {0.0}))
        b = P.baseline_check(baseline_cfg(), {"history": ref}, {"epochs": epochs_of(ref, 2)})
        self.assertEqual((b["outcome"], b["first_differing_epoch"], b["epoch_of_max_abs_difference"]), ("not checked", None, None))

    def test_trial_md_reports_both_epochs(self):
        ref = {k: [0.5, 0.4, 0.3] for k in P.HISTORY_KEYS}
        new = json.loads(json.dumps(ref))
        new["loss"] = [0.51, 0.6, 0.3]
        b = P.baseline_check(baseline_cfg(), {"history": ref}, {"epochs": epochs_of(new, 3)})
        md = P.trial_md(minimal_trial_doc(b))
        self.assertIn("| loss | 1 | 0.2 | 2 |", md)
        self.assertIn("| accuracy | none | 0 | none |", md)
        self.assertIn("TensorFlow threads set by this runner: requested intra-op 10 and inter-op 10", md)
        self.assertIn("environment variables: TF_NUM_INTRAOP_THREADS absent, TF_NUM_INTEROP_THREADS '10'", md)


def minimal_trial_doc(bcheck: dict) -> dict:
    """Un trial.json minimo del formato pad-arch-trial/2 per trial_md (senza TensorFlow)."""
    rec = {"epochs": [], "inference": {}, "rates": P.network_rates({"epochs": [], "inference": {}}, 32, 16)}
    return {"run_id": RUN, "status": "PASS", "unit": {**P.TRIAL_UNIT, "frames": {"fitting": {"n": 32}, "dev": {"n": 16}}}, "epochs_fixed": 3,
            "batch": 16, "device": "CPU (GPU hidden)", "cap_minutes": 60, "cap_unit": P.CAP_UNIT, "started": "s", "finished": "f",
            "wall_seconds": 1.0, "process_cpu_seconds": 2.0, "peak_rss_mb": 3.0,
            "machine": {"processor": "Test CPU", "core_levels": [{"name": "Super", "physical_cores": 5}, {"name": "Performance", "physical_cores": 10}],
                        "logical_cpus": 15, "ram_gb": 24.0, "os": "macOS"},
            "environment": {"tensorflow_installed": "2.15.0", "keras": "2.15.0", "python": "3.11", "numpy": "1.26"},
            "threads": {"requested": {"intra_op": 10, "inter_op": 10}, "effective": {"intra_op": 10, "inter_op": 10}, "set_by_this_runner": True,
                        "environment": {"TF_NUM_INTRAOP_THREADS": None, "TF_NUM_INTEROP_THREADS": "10"}, "logical_cpus": 15},
            "stopped": None, "networks": {n: rec for n in AM.NETWORKS}, "data_check": {"fitting_keys_equal_to_reference_card": True},
            "baseline_check": bcheck, "command": "c"}


# ============================================================================ piano: ordine, tetti, macchina

class TestSPlanOrderAndCaps(unittest.TestCase):
    def setUp(self):
        self.w = world()
        self.ctx = injected(**hooks(self.w))
        self.ctx.__enter__()
        self.addCleanup(self.ctx.__exit__, None, None, None)
        self.out = self.w.tmp / "plan_order_out"
        self.addCleanup(shutil.rmtree, self.out, True)
        self.addCleanup(shutil.rmtree, self.w.tmp / "trialfake2", True)

    def plan(self, *extra):
        return run_main(self.w.argv("plan", "--out-dir", str(self.out), *extra))

    def test_order_of_the_48_trainings(self):
        self.assertEqual(self.plan()[0], 0)
        rows = [l.split(",") for l in (self.out / "run_order.csv").read_text().splitlines()]
        self.assertEqual(rows[0], list(P.ORDER_COLUMNS))
        self.assertEqual(len(rows), 49)
        self.assertEqual(rows[1], ["nuaa_patience_4", "1", "LivenessNet", "1", "42", "4", P4, "arm_b_fold1_seed42", "1"])
        self.assertEqual(rows[2], ["nuaa_patience_4", "2", "ResNet-18", "1", "42", "4", "none", "none", "none"])
        self.assertEqual(rows[25][:7], ["nuaa_patience_12", "1", "LivenessNet", "1", "42", "12", P12])
        accepted = SR.training_order(4, SAR.SEEDS)
        for r in rows[1:]:
            if r[2] == "LivenessNet":
                self.assertEqual(int(r[8]), int(r[1]))                            # stessa posizione del run accettato
                self.assertEqual(accepted[int(r[1]) - 1], ("arm_b", int(r[3]), int(r[4])))
        self.assertEqual([(r[3], r[4]) for r in rows[1:25:2]], [(str(f), str(s)) for f in (1, 2, 3, 4) for s in SAR.SEEDS])
        caps = (self.out / "run_caps.csv").read_text().splitlines()
        self.assertEqual(caps[1], "nuaa_patience_4,NUAA,4,24,,,,,")
        self.assertIn("Machine: no trial given.", (self.out / "run_plan.md").read_text())

    def test_caps_from_the_trial(self):
        folder = self.w.tmp / "trialfake2"
        fake_trial(self.w, folder)
        rc, out, err = self.plan("--trial-run", str(folder))
        self.assertEqual(rc, 0, err)
        cfg, _, _ = P.load_config(self.w.cfg_path, self.w.base)
        rows = P.plan_rows(cfg, json.loads((folder / "trial.json").read_text()))
        caps = {l.split(",")[0]: l.split(",") for l in (self.out / "run_caps.csv").read_text().splitlines()[1:]}
        col = {k: i for i, k in enumerate(P.CAPS_COLUMNS)}
        for p in (4, 12):
            rs = [r for r in rows if r["dataset"] == "NUAA" and r["patience"] == p]
            wmax = rs[0]["wall_hours_max"] + rs[1]["wall_hours_max"]
            c = caps[f"nuaa_patience_{p}"]
            self.assertEqual(c[col["wall_hours_max"]], f"{wmax:.2f}")
            self.assertEqual(c[col["wall_hours_min"]], f"{rs[0]['wall_hours_min'] + rs[1]['wall_hours_min']:.2f}")
            self.assertEqual(c[col["cpu_hours_max"]], f"{rs[0]['cpu_hours_max'] + rs[1]['cpu_hours_max']:.2f}")
            cap = float(c[col["cap_wall_hours"]])
            self.assertEqual(cap % 0.5, 0.0)
            self.assertGreaterEqual(cap, 1.5 * wmax)
            self.assertLess(cap - 0.5, 1.5 * wmax)
        md = (self.out / "run_plan.md").read_text()
        self.assertIn(f"Machine of the trial {RUN}: Test CPU, 15 logical CPUs, 24.0 GB RAM, macOS 27.2.1.", md)
        self.assertIn("TensorFlow threads (training.threads): intra-op 10, inter-op 10, set by the runner; rule: t.", md)
        self.assertIn("Machine rule: the same machine and the same threads", md)

    def test_cap_rule(self):
        cfg = {"plan": {"caps": {"wall_multiplier": 1.5, "round_up_hours": 0.5},
                        "datasets": [{"name": "NUAA", "splits": [1, 2, 3, 4], "seeds": [42, 43, 44]}]}}
        rows = [{"dataset": "NUAA", "patience": p, "new_trainings": 12, "wall_hours_min": 1.0, "wall_hours_max": x, "cpu_hours_min": 8.0,
                 "cpu_hours_max": 9.0} for p in (4, 12) for x in (1.0, 2.0)]
        c = P.run_caps(cfg, rows)
        self.assertEqual([(x["wall_hours_max"], x["cap_wall_hours"], x["cpu_hours_max"]) for x in c], [(3.0, 4.5, 18.0), (3.0, 4.5, 18.0)])
        rows[0]["wall_hours_max"] = 1.1                       # 3,1 x 1,5 = 4,65 -> 5,0
        self.assertEqual(P.run_caps(cfg, rows)[0]["cap_wall_hours"], 5.0)
        rows[0]["new_trainings"] = 11
        with self.assertRaises(FR.Failure):
            P.run_caps(cfg, rows)

    def test_trial_of_the_previous_format_or_without_threads_refused(self):
        folder = self.w.tmp / "trialfake2"
        fake_trial(self.w, folder, over={"schema": P.TRIAL_SCHEMA_V1})
        rc, out, err = self.plan("--trial-run", str(folder))
        self.assertEqual(rc, 1)
        self.assertIn("schema pad-arch-trial/1 is the format before the runner set the TensorFlow threads", err)
        good = fake_trial(self.w, folder)
        for over, needle in (({"threads": {**good["threads"], "set_by_this_runner": False}}, "were not set by the runner"),
                             ({"threads": {**good["threads"], "requested": {"intra_op": 15, "inter_op": 10}}}, "were not set by the runner"),
                             ({"threads": {**good["threads"], "effective": {"intra_op": 0, "inter_op": 0}}}, "were not set by the runner"),
                             ({"threads": {k: v for k, v in good["threads"].items() if k != "environment"}}, "must hold exactly"),
                             ({"machine": {"processor": "x", "os": "y", "logical_cpus": 15}}, "machine record must hold"),
                             ({"machine": {**good["machine"], "logical_cpus": 0}}, "machine processor"),
                             ({"machine": {**good["machine"], "ram_gb": "24"}}, "ram_gb")):
            fake_trial(self.w, folder, over=over)
            rc, out, err = self.plan("--trial-run", str(folder))
            self.assertEqual(rc, 1, over)
            self.assertIn(needle, err)
        nets = json.loads(json.dumps(good["networks"]))
        nets["ResNet-18"]["threads"] = {"intra_op": 10, "inter_op": 15}
        fake_trial(self.w, folder, nets=nets)
        rc, out, err = self.plan("--trial-run", str(folder))
        self.assertIn("the threads recorded for ResNet-18 differ", err)
        self.assertFalse(self.out.exists())


# ============================================================================ confronto fra due prove

def history_epochs(h: dict) -> list:
    return [{"epoch": i + 1, **{k: h[k][i] for k in P.HISTORY_KEYS}, "seconds_total": 1.0} for i in range(len(h["loss"]))]


def comparison_trial(folder: Path, run_id: str, schema=None, hist=None, over=None, status="PASS") -> dict:
    """Una prova finta per compare-trials (status.json e trial.json coerenti) nel formato dato."""
    schema = schema or P.TRIAL_SCHEMA
    base = {k: [0.5, 0.4, 0.3] for k in P.HISTORY_KEYS}
    hist = hist or {n: base for n in AM.NETWORKS}
    threads = ({"logical_cpus": 10, "inter_op_parallelism_threads": 0, "intra_op_parallelism_threads": 0, "set_by_this_runner": False}
               if schema == P.TRIAL_SCHEMA_V1 else
               {"requested": {"intra_op": 10, "inter_op": 10}, "effective": {"intra_op": 10, "inter_op": 10}, "set_by_this_runner": True,
                "environment": {"TF_NUM_INTRAOP_THREADS": None, "TF_NUM_INTEROP_THREADS": None}, "logical_cpus": 15})
    doc = {"schema": schema, "run_id": run_id, "status": status, "commit": run_id.split("-")[-1], "config_sha256": "c" * 64,
           "unit": {**P.TRIAL_UNIT, "frames": {}}, "machine": {"processor": "Test CPU", "logical_cpus": 15, "ram_gb": 24.0, "os": "macOS 27.2.1"},
           "threads": threads, "networks": {n: {"epochs": history_epochs(h)} for n, h in hist.items()}, **(over or {})}
    folder.mkdir(parents=True, exist_ok=True)
    b = (json.dumps(doc, indent=1) + "\n").encode()
    (folder / "trial.json").write_bytes(b)
    (folder / "status.json").write_text(json.dumps({"run_id": run_id, "status": status, "trial_json_sha256": sha(b)}), encoding="utf-8")
    return doc


class TestRCompareTrials(unittest.TestCase):
    NEW, REF = "20261120-ARCHTRIAL-1111111", "20261119-ARCHTRIAL-2222222"

    def setUp(self):
        self.tmp = Path(world().tmp) / "compare"
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.new, self.ref, self.out = self.tmp / "new", self.tmp / "ref", self.tmp / "out"

    def run_cmp(self, out=None, new=None, ref=None):
        return run_main(["compare-trials", "--trial-run", str(new or self.new), "--reference-run", str(ref or self.ref),
                         "--compare-out-dir", str(out or self.out)])

    def test_identical_histories_reproduced_across_formats(self):
        comparison_trial(self.new, self.NEW)
        comparison_trial(self.ref, self.REF, schema=P.TRIAL_SCHEMA_V1)
        rc, out, err = self.run_cmp()
        self.assertEqual(rc, 0, err)
        self.assertEqual(sorted(os.listdir(self.out)), sorted(P.COMPARE_OUTPUTS))
        doc = json.loads((self.out / "trial_comparison.json").read_text())
        self.assertEqual(doc["outcome"], "reproduced")
        self.assertEqual((doc["new"]["run_id"], doc["reference"]["run_id"]), (self.NEW, self.REF))
        self.assertEqual(doc["reference"]["threads"]["recorded"]["set_by_this_runner"], False)
        self.assertIn("0 = not set through tf.config.threading); format /1 does not record the environment variables",
                      doc["reference"]["threads"]["note"])
        self.assertEqual(doc["new"]["threads"]["recorded"]["requested"], {"intra_op": 10, "inter_op": 10})
        self.assertEqual(doc["new"]["files_sha256"], {"status.json": sha((self.new / "status.json").read_bytes()),
                                                      "trial.json": sha((self.new / "trial.json").read_bytes())})
        f = doc["networks"]["LivenessNet"]["fields"]["loss"]
        self.assertEqual(f, {"epochs_compared": 3, "first_differing_epoch": None, "max_abs_difference": 0.0, "epoch_of_max_abs_difference": None})
        self.assertIn("Outcome: **reproduced**", (self.out / "trial_comparison.md").read_text())
        rc, out, err = self.run_cmp()
        self.assertEqual(rc, 1)
        self.assertIn("already exist", err)

    def test_first_differing_epoch_and_largest_difference(self):
        base = {k: [0.5, 0.4, 0.3] for k in P.HISTORY_KEYS}
        other = json.loads(json.dumps(base))
        other["val_loss"] = [0.6, 1.4, 0.3]
        comparison_trial(self.new, self.NEW, hist={"LivenessNet": other, "ResNet-18": base})
        comparison_trial(self.ref, self.REF)
        self.assertEqual(self.run_cmp()[0], 0)
        doc = json.loads((self.out / "trial_comparison.json").read_text())
        self.assertEqual(doc["outcome"], "not reproduced")
        f = doc["networks"]["LivenessNet"]["fields"]["val_loss"]
        self.assertEqual((f["first_differing_epoch"], f["epoch_of_max_abs_difference"]), (1, 2))
        self.assertAlmostEqual(f["max_abs_difference"], 1.0, places=12)
        self.assertTrue(doc["networks"]["ResNet-18"]["reproduced"])
        self.assertFalse(doc["networks"]["LivenessNet"]["reproduced"])

    def test_different_number_of_epochs(self):
        base = {k: [0.5, 0.4, 0.3] for k in P.HISTORY_KEYS}
        short = {k: v[:2] for k, v in base.items()}
        comparison_trial(self.new, self.NEW, hist={"LivenessNet": base, "ResNet-18": short})
        comparison_trial(self.ref, self.REF)
        self.assertEqual(self.run_cmp()[0], 0)
        doc = json.loads((self.out / "trial_comparison.json").read_text())
        r = doc["networks"]["ResNet-18"]
        self.assertEqual((r["epochs_new"], r["epochs_reference"], r["same_number_of_epochs"], r["reproduced"]), (2, 3, False, False))
        self.assertEqual(r["fields"]["loss"]["first_differing_epoch"], None)
        self.assertEqual(doc["outcome"], "not reproduced")
        shutil.rmtree(self.out)
        empty = {k: [] for k in P.HISTORY_KEYS}
        comparison_trial(self.new, self.NEW, hist={"LivenessNet": empty, "ResNet-18": empty})
        comparison_trial(self.ref, self.REF, hist={"LivenessNet": empty, "ResNet-18": empty}, status="INCOMPLETE")
        self.assertEqual(self.run_cmp()[0], 0)
        self.assertEqual(json.loads((self.out / "trial_comparison.json").read_text())["outcome"], "not reproduced")

    def test_failures_leave_no_output(self):
        base = {k: [0.5, 0.4, 0.3] for k in P.HISTORY_KEYS}
        cases = [
            (lambda: comparison_trial(self.new, self.NEW, hist={"LivenessNet": base}), "networks must be exactly"),
            (lambda: (self.new / "trial.json").unlink(), "trial.json"),
            (lambda: (self.new / "status.json").write_text("{broken", encoding="utf-8"), "status.json"),
            (lambda: (self.new / "trial.json").write_text("{}", encoding="utf-8"), "SHA-256 differs"),
            (lambda: comparison_trial(self.new, self.NEW, over={"schema": "pad-arch-trial/9"}), "schema must be"),
            (lambda: comparison_trial(self.new, self.NEW, over={"unit": {**P.TRIAL_UNIT, "fold": 2}}), "not a trial of the unit"),
            (lambda: comparison_trial(self.new, self.NEW, status="FAIL"), "not a PASS or INCOMPLETE"),
            (lambda: comparison_trial(self.new, self.NEW, over={"run_id": self.REF}), "run id or status differ"),
            (lambda: comparison_trial(self.new, self.NEW, over={"threads": {"logical_cpus": 15}}), "threads record"),
            (lambda: comparison_trial(self.new, self.NEW, over={"machine": {"os": "x"}}), "machine record"),
            (lambda: comparison_trial(self.new, self.NEW, over={"commit": "zz"}), "commit or config_sha256"),
            (lambda: comparison_trial(self.new, self.NEW, over={"machine": {"processor": str(Path.home()), "logical_cpus": 15, "ram_gb": 24.0,
                                                                            "os": "macOS"}}), "trial_comparison.json refused"),
            (lambda: comparison_trial(self.new, self.NEW, over={"networks": {n: {"epochs": [{"epoch": 2, **{k: 0.1 for k in P.HISTORY_KEYS}}]}
                                                                             for n in AM.NETWORKS}}), "numbered from 1"),
            (lambda: comparison_trial(self.new, self.NEW, over={"networks": {n: {"epochs": [{"epoch": 1, **{k: True for k in P.HISTORY_KEYS}}]}
                                                                             for n in AM.NETWORKS}}), "numeric"),
        ]
        for make, needle in cases:
            shutil.rmtree(self.tmp, ignore_errors=True)
            comparison_trial(self.new, self.NEW)
            comparison_trial(self.ref, self.REF)
            make()
            rc, out, err = self.run_cmp()
            self.assertEqual(rc, 1, needle)
            self.assertIn(needle, err)
            self.assertFalse(self.out.exists(), needle)

    def test_output_folder_rules(self):
        """Uscita dentro una delle cartelle lette; le cartelle del repository: TestWOutputFolderFakeRoot (radice finta)."""
        comparison_trial(self.new, self.NEW)
        comparison_trial(self.ref, self.REF)
        for out, needle in ((self.new / "cmp", "inputs and outputs must be separate"), (self.ref, "inputs and outputs must be separate")):
            rc, o, err = self.run_cmp(out=out)
            self.assertEqual(rc, 1, out)
            self.assertIn(needle, err)
            self.assertFalse(out.exists() and any(n in os.listdir(out) for n in P.COMPARE_OUTPUTS))
        rc, o, err = self.run_cmp(ref=self.new)
        self.assertEqual(rc, 1)
        self.assertIn("must be different", err)
        rc, o, err = run_main(["compare-trials", "--trial-run", str(self.new), "--compare-out-dir", str(self.out)])
        self.assertIn("needs --trial-run", err)
        self.assertFalse(self.out.exists())


# ============================================================================ rilievi della revisione indipendente (19/11)

class TestTReviewDefinitions(unittest.TestCase):
    def test_labels_are_part_of_the_frames(self):
        """Stessi ID e soggetti, un'etichetta diversa (nelle due reti): fra seed dello stesso split e fra condizioni."""
        units = two_splits(seeds=(42, 43))
        self.assertIs(units[1]["networks"]["ResNet-18"]["y"], units[1]["networks"]["LivenessNet"]["y"])   # una lista per le due reti
        units[1]["networks"]["ResNet-18"]["y"][0] = 1 - units[1]["networks"]["ResNet-18"]["y"][0]
        with self.assertRaises(FR.Failure) as cm:
            D.compare_condition(units)
        self.assertIn("two units of ('D', 1) are not scored on the same frames", str(cm.exception))
        ua, ub = two_splits(), two_splits(liv=perfect)
        ub[0]["networks"]["ResNet-18"]["y"][0] = 1 - ub[0]["networks"]["ResNet-18"]["y"][0]
        with self.assertRaises(FR.Failure) as cm:
            D.compare_conditions(D.compare_condition(ua), D.compare_condition(ub))
        self.assertIn("same test frames (subjects and labels)", str(cm.exception))

    def test_seed_outside_the_plan(self):
        f = frames_of(list("abc"))
        for seed in (45, True, "42"):
            with self.assertRaises(FR.Failure) as cm:
                D.compare_condition([unit(1, seed, f, perfect(f), perfect(f))])
            self.assertIn("seeds of the plan", str(cm.exception))

    def test_per_seed_reading_uses_the_interval_of_each_seed(self):
        """Seed 42 e 43: Δ = −1/2; seed 44: LivenessNet sbaglia solo gli attacchi del soggetto a (Δ medio < 0, ma
        l'intervallo del seed tocca lo zero); l'intervallo complessivo è tutto sotto zero."""
        def wrong_on_a(f):
            return [0.2 if (s == "a" and lab) else (0.9 if lab else 0.1) for s, lab in f]
        units = []
        for split, subs in ((1, list("abc")), (2, list("def"))):
            f = frames_of(subs)
            for seed in (42, 43, 44):
                units.append(unit(split, seed, f, perfect(f), wrong_on_a(f) if seed == 44 else [0.2] * len(f)))
        out = D.compare_condition(units)
        self.assertLess(out["interval"]["acer"]["high"], 0)
        self.assertEqual(out["per_seed"][44]["interval"]["acer"]["high"], 0.0)
        self.assertLess(out["per_seed"][44]["mean_delta"]["acer"], 0)
        r = out["per_seed_reading"]
        self.assertEqual((r["verdict"], r["reason"]), ("inconclusive", "the bootstrap interval of a seed does not exclude 0 on the side of the difference"))
        self.assertEqual((out["sign_reason"], out["per_seed"][44]["sign_reason"]), ("interval entirely below zero", "interval contains zero"))

    def test_contrast_reading_from_acer_only(self):
        iv = lambda lo, hi: {"low": lo, "high": hi, "valid_resamples": 1000}      # noqa: E731
        c = {"contrast_mean": {}, "units": 2, "units_changing_sign": 0,
             "contrast_interval": {"acer": iv(-0.2, -0.1), "apcer": iv(0.1, 0.2), "bpcer": iv(0.1, 0.2)}}
        s = D.contrast_summary(c)
        self.assertEqual((s["reading"], s["reason"]), (D.CONTRAST_LOWER, "interval entirely below zero"))

    def test_n_subjects_and_reasons(self):
        r = D.primary_analysis(primary_conditions())
        self.assertEqual((r["gap_ordinary"]["n_subjects"], r["gap_re_estimated"]["n_subjects"]), (6, 6))
        self.assertEqual(r["contrast"]["reason"], "interval entirely above zero")
        self.assertEqual(r["per_seed"][43]["contrast"]["reason"], "interval contains zero")
        self.assertEqual(r["gap_ordinary"]["sign_reason"], "interval entirely below zero")
        self.assertEqual([D.interval_reason(x) for x in ({"low": -1, "high": -0.5}, {"low": 0.5, "high": 1}, {"low": 0.0, "high": 1},
                                                          {"low": -1, "high": 0.0}, {"low": None, "high": None})],
                         [D.INTERVAL_REASONS[i] for i in (0, 1, 2, 2, 3)])
        f = frames_of(list("ab"))
        out = D.compare_condition([unit(1, 42, f, perfect(f), [0.2] * len(f))])          # 2 soggetti: nessun intervallo
        self.assertEqual((out["sign"], out["sign_reason"]), (D.SIGN_UNRESOLVED, D.INTERVAL_REASONS[3]))
        self.assertEqual(out["per_seed"][42]["sign_reason"], D.INTERVAL_REASONS[3])
        out = D.compare_condition(two_splits(seeds=(42,), liv=perfect))                     # reti uguali: [0, 0]
        self.assertEqual((out["sign"], out["sign_reason"]), (D.SIGN_UNRESOLVED, D.INTERVAL_REASONS[2]))

    def test_prepared_malformed(self):
        a, b = D.compare_condition(two_splits()), D.compare_condition(two_splits(liv=perfect))
        for bad in ({**b, "prepared": [{k: v for k, v in p.items() if k != "frames"} for p in b["prepared"]]}, {**b, "prepared": []},
                    {k: v for k, v in b.items() if k != "prepared"}, {k: v for k, v in b.items() if k != "sign"}, None):
            with self.assertRaises(FR.Failure) as cm:
                D.compare_conditions(a, bad)
            self.assertIn("field prepared malformed", str(cm.exception))


class TestUReviewRunner(unittest.TestCase):
    def test_non_finite_history(self):
        nan, inf = float("nan"), float("inf")
        for ref, new in (([1.0, 2.0, 3.0], [nan, 2.0, 5.0]), ([nan], [nan]), ([1.0], [inf]), ([1.0], [True]), ([-inf], [1.0])):
            with self.assertRaises(FR.Failure) as cm:
                P.history_differences(ref, new)
            self.assertIn("not a finite number", str(cm.exception))
        ref = {k: [0.5, 0.4, 0.3] for k in P.HISTORY_KEYS}
        new = json.loads(json.dumps(ref))
        new["val_loss"][1] = nan
        with self.assertRaises(FR.Failure):
            P.baseline_check(baseline_cfg(), {"history": ref}, {"epochs": epochs_of(new, 3)})

    def test_outcome_equals_the_shared_history_rule(self):
        """L'esito di baseline_check (una sola fonte, history_differences) è quello di PV.shared_history su storie casuali,
        anche con una scheda più corta delle epoche fissate."""
        rng = np.random.default_rng(3)
        for _ in range(200):
            ref = {k: [float(x) for x in rng.choice([0.1, 0.2], 4)] for k in P.HISTORY_KEYS}
            new = {k: [float(x) for x in rng.choice([0.1, 0.2], 3)] if rng.random() < 0.5 else ref[k][:3] for k in P.HISTORY_KEYS}
            b = P.baseline_check(baseline_cfg(), {"history": ref}, {"epochs": epochs_of(new, 3)})
            self.assertEqual(b["outcome"] == "reproduced", PV.shared_history(ref, new, 3)["identical"])
        short = {k: [0.1, 0.2] for k in P.HISTORY_KEYS}          # scheda più corta delle epoche fissate: mai riprodotta
        b = P.baseline_check(baseline_cfg(), {"history": short}, {"epochs": epochs_of({k: [0.1, 0.2, 0.3] for k in P.HISTORY_KEYS}, 3)})
        self.assertEqual(b["outcome"], "not reproduced")

    def test_boolean_or_non_finite_epoch_in_compare_trials(self):
        tmp = Path(world().tmp) / "compare_bool"
        shutil.rmtree(tmp, ignore_errors=True)
        self.addCleanup(shutil.rmtree, tmp, True)
        base = {k: [0.5, 0.4, 0.3] for k in P.HISTORY_KEYS}
        comparison_trial(tmp / "ref", TestRCompareTrials.REF)
        for over in ({"networks": {n: {"epochs": [{"epoch": True, **{k: 0.1 for k in P.HISTORY_KEYS}}]} for n in AM.NETWORKS}},
                     {"networks": {n: {"epochs": [{"epoch": 1, **{k: float("nan") for k in P.HISTORY_KEYS}}]} for n in AM.NETWORKS}},
                     {"networks": {n: {"epochs": [{"epoch": 1, **{k: float("inf") for k in P.HISTORY_KEYS}}]} for n in AM.NETWORKS}}):
            comparison_trial(tmp / "new", TestRCompareTrials.NEW, hist={n: base for n in AM.NETWORKS}, over=over)
            rc, o, err = run_main(["compare-trials", "--trial-run", str(tmp / "new"), "--reference-run", str(tmp / "ref"),
                                   "--compare-out-dir", str(tmp / "out")])
            self.assertEqual(rc, 1)
            self.assertIn("numbered from 1 with finite numeric", err)
            self.assertFalse((tmp / "out").exists())


class TestVReferenceConditions(unittest.TestCase):
    NEW, REF = TestRCompareTrials.NEW, TestRCompareTrials.REF

    def setUp(self):
        self.tmp = Path(world().tmp) / "conditions"
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.tmp, True)
        comparison_trial(self.tmp / "new", self.NEW)
        comparison_trial(self.tmp / "ref", self.REF, schema=P.TRIAL_SCHEMA_V1)
        self.out = self.tmp / "out"

    def run_cmp(self, cond=None):
        argv = ["compare-trials", "--trial-run", str(self.tmp / "new"), "--reference-run", str(self.tmp / "ref"), "--compare-out-dir", str(self.out)]
        return run_main(argv + (["--reference-conditions", str(cond)] if cond else []))

    def cond(self, text: str) -> Path:
        q = self.tmp / "conditions.txt"
        q.write_text(text, encoding="utf-8")
        return q

    def test_not_recorded_without_the_file(self):
        self.assertEqual(self.run_cmp()[0], 0)
        doc = json.loads((self.out / "trial_comparison.json").read_text())
        self.assertEqual(doc["reference"]["threads_as_imposed"], {"source": "not recorded"})
        self.assertNotIn("threads_as_imposed", doc["new"])
        self.assertIn(f"How the threads were imposed in the reference trial {self.REF}: not recorded.", (self.out / "trial_comparison.md").read_text())

    def test_command_line_reported(self):
        """Riga del comando riportata (spazi ai lati tolti), con i valori letti e la nota su che cosa è dichiarato e che
        cosa è controllato; il resto del file non è riportato."""
        q = self.cond("== conditions before\nload 1.1\n== command\n   TF_NUM_INTRAOP_THREADS=10 TF_NUM_INTEROP_THREADS=10 python -I -B "
                      "scripts/pad_arch_comparison.py trial --i-am-authorized --out-root <folder>   \nTF_NUM_INTRAOP_THREADS only\n== git\n2222222\n")
        rc, o, err = self.run_cmp(q)
        self.assertEqual(rc, 0, err)
        c = json.loads((self.out / "trial_comparison.json").read_text())["reference"]["threads_as_imposed"]
        self.assertEqual(c, {"source": "conditions file", "file": "conditions.txt", "sha256": sha(q.read_bytes()),
                             "command_line": "TF_NUM_INTRAOP_THREADS=10 TF_NUM_INTEROP_THREADS=10 python -I -B scripts/pad_arch_comparison.py "
                                             "trial --i-am-authorized --out-root <folder>",
                             "values": {"TF_NUM_INTRAOP_THREADS": 10, "TF_NUM_INTEROP_THREADS": 10}, "note": P.CONDITIONS_NOTE})
        md = (self.out / "trial_comparison.md").read_text()
        self.assertIn("in the conditions file conditions.txt (SHA-256 " + sha(q.read_bytes()), md)
        self.assertIn("declared by a conditions file written by the author around the reference trial, not by its record", md)
        self.assertIn("does not record the environment variables", md)
        self.assertNotIn("load 1.1", md)

    def test_conditions_refused(self):
        ok = "TF_NUM_INTRAOP_THREADS=10 TF_NUM_INTEROP_THREADS=10 run\n"
        for text, needle in (("2222222\nno variables here\n", "0 lines hold both"), ("2222222\nTF_NUM_INTRAOP_THREADS=10 only\n", "0 lines hold both"),
                             ("2222222\n" + ok + "TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 b\n", "2 lines hold both"),
                             (f"2222222\nTF_NUM_INTRAOP_THREADS=10 TF_NUM_INTEROP_THREADS=10 cd {Path.home()}\n", "trial_comparison.json refused"),
                             (ok, "does not hold the commit 2222222"),
                             ("22222223\n" + ok, "does not hold the commit 2222222"),
                             ("a2222222\n" + ok, "does not hold the commit 2222222"),
                             ("2222222\nTF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=10 run\n", "TF_NUM_INTRAOP_THREADS must be set once"),
                             ("2222222\nTF_NUM_INTRAOP_THREADS=10 TF_NUM_INTEROP_THREADS=15 run\n", "TF_NUM_INTEROP_THREADS must be set once"),
                             ("2222222\nTF_NUM_INTRAOP_THREADS=10.0 TF_NUM_INTEROP_THREADS=10 run\n", "TF_NUM_INTRAOP_THREADS must be set once"),
                             ("2222222\nTF_NUM_INTRAOP_THREADS= TF_NUM_INTEROP_THREADS=10 run\n", "TF_NUM_INTRAOP_THREADS must be set once"),
                             ("2222222\nTF_NUM_INTRAOP_THREADS=10 TF_NUM_INTEROP_THREADS=10 TF_NUM_INTRAOP_THREADS=10 run\n",
                              "TF_NUM_INTRAOP_THREADS must be set once")):
            shutil.rmtree(self.out, ignore_errors=True)
            rc, o, err = self.run_cmp(self.cond(text))
            self.assertEqual(rc, 1, text)
            self.assertIn(needle, err)
            self.assertFalse(self.out.exists())
        rc, o, err = self.run_cmp(self.tmp / "missing.txt")
        self.assertEqual(rc, 1)
        self.assertFalse(self.out.exists())
        q = self.tmp / "bin.txt"
        q.write_bytes(b"\xff\xfe TF_NUM_INTRAOP_THREADS TF_NUM_INTEROP_THREADS")
        self.assertIn("not UTF-8", self.run_cmp(q)[2])


class TestWOutputFolderFakeRoot(unittest.TestCase):
    def test_output_folder_inside_a_fake_repository(self):
        """Le regole della cartella d'uscita di compare-trials provate su una radice finta (nessun percorso del
        repository vero)."""
        tmp = Path(world().tmp) / "fakeroot_cmp"
        shutil.rmtree(tmp, ignore_errors=True)
        self.addCleanup(shutil.rmtree, tmp, True)
        root = tmp / "repo"
        for d in ("scripts", "results", "delivery/archcmp"):
            (root / d).mkdir(parents=True)
        comparison_trial(tmp / "new", TestRCompareTrials.NEW)
        comparison_trial(tmp / "ref", TestRCompareTrials.REF)
        with injected(p={"ROOT": root}):
            for out in (root / "scripts" / "cmp", root / "results" / "cmp", root / "cmp"):
                rc, o, err = run_main(["compare-trials", "--trial-run", str(tmp / "new"), "--reference-run", str(tmp / "ref"),
                                       "--compare-out-dir", str(out)])
                self.assertEqual(rc, 1, out)
                self.assertIn("refusing the comparison folder", err)
                self.assertFalse(out.exists())
            rc, o, err = run_main(["compare-trials", "--trial-run", str(tmp / "new"), "--reference-run", str(tmp / "ref")])
            self.assertEqual(rc, 0, err)
            self.assertEqual(sorted(os.listdir(root / P.DEFAULT_COMPARE_DIR)), sorted(P.COMPARE_OUTPUTS))


SUBPROCESS_THREADS = r"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
import nuaa_folds_run as FR
import pad_arch_comparison as P
cfg = {"training": {"threads": {"intra_op": 10, "inter_op": 10, "rule": "r"}}}
import tensorflow as tf
if sys.argv[2] == "after-operation":
    tf.constant(1.0)
try:
    rec = P.set_threads(tf, cfg)
    print(json.dumps({"ok": True, "effective": rec["effective"], "getters": P.read_threads(tf)}))
except FR.Failure as e:
    print(json.dumps({"ok": False, "error": str(e)}))
"""


class TestKThreadsInAProcess(unittest.TestCase):
    def test_set_threads_in_a_clean_process_and_after_an_operation(self):
        """Con TensorFlow vero, in processi nuovi (nessun addestramento): in un processo pulito i getter leggono 10 e 10;
        dopo un'operazione di TensorFlow l'impostazione è un Failure."""
        if not TF_AVAILABLE:
            skip_missing(self, "tensorflow")
        import subprocess
        env = {k: v for k, v in os.environ.items() if k not in P.TF_THREAD_ENV.values()}
        env.update(CUDA_VISIBLE_DEVICES="-1", TF_CPP_MIN_LOG_LEVEL="2")
        out = {}
        for mode in ("clean", "after-operation"):
            r = subprocess.run([sys.executable, "-I", "-B", "-c", SUBPROCESS_THREADS, str(Path(P.__file__).resolve().parent), mode],
                               capture_output=True, text=True, env=env, timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            out[mode] = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual(out["clean"], {"ok": True, "effective": {"intra_op": 10, "inter_op": 10}, "getters": {"intra_op": 10, "inter_op": 10}})
        self.assertFalse(out["after-operation"]["ok"])
        self.assertIn("the TensorFlow threads cannot be set", out["after-operation"]["error"])



# ============================================================================ rilievi della seconda revisione (A3)

class TestXReviewDefinitions(unittest.TestCase):
    def test_sign_reason_from_acer(self):
        """LivenessNet decide tutto attacco: ΔACER = −1/2 (sotto zero), ΔAPCER = 0 (intervallo che contiene lo zero)."""
        out = D.compare_condition(two_splits(liv=lambda f: [0.9] * len(f)))
        self.assertEqual((out["interval"]["apcer"]["low"], out["interval"]["apcer"]["high"]), (0.0, 0.0))
        self.assertEqual((out["sign"], out["sign_reason"]), (D.SIGN_RESNET_LOWER, "interval entirely below zero"))

    def test_units_changing_sign_in_the_contrast(self):
        r = D.primary_analysis(primary_conditions())
        self.assertEqual((r["contrast"]["units"], r["contrast"]["units_changing_sign"]), (4, 0))

    def test_seed_must_be_an_integer(self):
        f = frames_of(list("abc"))
        for seed in (42.0, np.float64(42.0), np.bool_(True)):
            with self.assertRaises(FR.Failure) as cm:
                D.compare_condition([unit(1, seed, f, perfect(f), perfect(f))])
            self.assertIn("integer among the seeds", str(cm.exception))
        self.assertEqual(D.compare_condition([unit(1, np.int64(42), f, perfect(f), [0.2] * len(f))])["mean_delta"]["acer"], -0.5)

    def test_one_dataset_per_call(self):
        f1, f2 = frames_of(list("abc")), frames_of(list("def"))
        units = [unit(1, 42, f1, perfect(f1), perfect(f1)), unit(2, 42, f2, perfect(f2), perfect(f2), dataset="E")]
        with self.assertRaises(FR.Failure) as cm:
            D.compare_condition(units)
        self.assertIn("one dataset", str(cm.exception))

    def test_non_finite_readings_refused(self):
        nan = float("nan")
        for iv in ({"low": nan, "high": 0.1}, {"low": -0.1, "high": float("inf")}, {"low": 0.2, "high": 0.1}, {"low": None, "high": 0.1},
                   {"low": True, "high": 0.1}):
            for fn in (D.interval_reason, D.sign_of, D.contrast_reading):
                with self.assertRaises(FR.Failure):
                    fn(iv)
        e = {"difference": -0.2, "low": -0.3, "high": -0.1}
        for bad in ({**e, "difference": nan}, {**e, "low": nan}, {**e, "high": float("-inf")}):
            with self.assertRaises(FR.Failure) as cm:
                D.per_seed_reading([e, bad, e])
            self.assertIn("not a finite number", str(cm.exception))

    def test_prepared_frames_malformed(self):
        a, b = D.compare_condition(two_splits()), D.compare_condition(two_splits(liv=perfect))
        for frames in ("x", (), ((), (), ()), (("i",), ("s",)), (("i", "j"), ("s",), (0,)), [("i",), ("s",), (0,)]):
            bad = {**b, "prepared": [{**b["prepared"][0], "frames": frames}] + b["prepared"][1:]}
            with self.assertRaises(FR.Failure) as cm:
                D.compare_conditions(a, bad)
            self.assertIn("field prepared malformed", str(cm.exception))


class TestYReviewRunner(unittest.TestCase):
    def test_history_values(self):
        r = P.history_differences([np.float32(0.5), 1], [np.float64(0.5), np.int64(2)])
        self.assertEqual((r["first_differing_epoch"], r["max_abs_difference"]), (2, 1.0))
        for bad in ([np.bool_(True)], [True], ["0.5"], [None], [Fraction(1, 2)]):
            with self.assertRaises(FR.Failure):
                P.history_differences([0.5], bad)
        with self.assertRaises(FR.Failure) as cm:
            P.history_differences([1e308], [-1e308])
        self.assertIn("difference of two values is not finite", str(cm.exception))

    def test_json_text_refuses_non_finite(self):
        self.assertEqual(P.json_text({"a": 1.5}, "x"), '{\n "a": 1.5\n}\n')
        for doc in ({"a": float("nan")}, {"a": float("inf")}, {"a": object()}):
            with self.assertRaises(FR.Failure) as cm:
                P.json_text(doc, "x.json")
            self.assertIn("x.json: not written", str(cm.exception))

    def test_machine_record(self):
        good = {"processor": "p", "os": "o", "logical_cpus": 15, "ram_gb": 24.0}
        self.assertEqual(P.check_machine({**good, "ram_gb": None}, "m")["ram_gb"], None)
        for bad in ({**good, "processor": 5}, {**good, "os": None}, {**good, "ram_gb": 0}, {**good, "ram_gb": -1.0}, {**good, "ram_gb": float("nan")},
                    {**good, "ram_gb": float("inf")}, {**good, "ram_gb": True}, {**good, "logical_cpus": 0}, {**good, "logical_cpus": True}):
            with self.assertRaises(FR.Failure):
                P.check_machine(bad, "m")

    def test_machine_with_a_fake_sysctl(self):
        gib = str(24 * 2 ** 30)
        cases = {
            "M5 Pro": ({"machdep.cpu.brand_string": "Apple M5 Pro", "hw.memsize": gib, "hw.nperflevels": "2", "hw.perflevel0.name": "Super",
                        "hw.perflevel0.physicalcpu": "5", "hw.perflevel1.name": "Performance", "hw.perflevel1.physicalcpu": "10"},
                       [{"name": "Super", "physical_cores": 5}, {"name": "Performance", "physical_cores": 10}], 24.0),
            "M4": ({"machdep.cpu.brand_string": "Apple M4", "hw.memsize": str(16 * 2 ** 30), "hw.nperflevels": "2", "hw.perflevel0.name": "Performance",
                    "hw.perflevel0.physicalcpu": "4", "hw.perflevel1.name": "Efficiency", "hw.perflevel1.physicalcpu": "6"},
                   [{"name": "Performance", "physical_cores": 4}, {"name": "Efficiency", "physical_cores": 6}], 16.0),
            "Intel": ({"machdep.cpu.brand_string": "Intel", "hw.memsize": gib}, [], 24.0),
            "Linux": ({}, [], None),
        }
        for name, (values, levels, ram) in cases.items():
            with injected(p={"SYSCTL": values.get}):
                m = P.machine()
            self.assertEqual((m["core_levels"], m["ram_gb"]), (levels, ram), name)
        self.assertEqual(P.core_levels_text({"core_levels": cases["M5 Pro"][1]}), "5 Super, 10 Performance")
        self.assertEqual(P.core_levels_text({"core_levels": [{"name": None, "physical_cores": None}]}), "not reported not reported")
        self.assertEqual(P.core_levels_text({"core_levels": []}), "not reported")
        self.assertEqual((P.ram_text({"ram_gb": None}), P.ram_text({"ram_gb": 24.0})), ("RAM not reported", "24.0 GB RAM"))
        doc = minimal_trial_doc(P.baseline_check(baseline_cfg(), {"history": {k: [0.5] * 3 for k in P.HISTORY_KEYS}},
                                                 {"epochs": epochs_of({k: [0.5] * 3 for k in P.HISTORY_KEYS}, 3)}))
        doc["machine"] = {**doc["machine"], "ram_gb": None, "core_levels": [], "logical_cpus": 4}
        md = P.trial_md(doc)
        self.assertIn("RAM not reported", md)
        self.assertNotIn("None", md.split("## Epochs")[0])

    def test_thread_records(self):
        v1 = {"logical_cpus": 10, "inter_op_parallelism_threads": 0, "intra_op_parallelism_threads": 0, "set_by_this_runner": False}
        w = {"intra_op": 10, "inter_op": 10}
        v2 = {"requested": dict(w), "effective": dict(w), "set_by_this_runner": True, "logical_cpus": 15,
              "environment": {"TF_NUM_INTRAOP_THREADS": None, "TF_NUM_INTEROP_THREADS": "10"}}
        self.assertIs(P.check_thread_record(v1, P.TRIAL_SCHEMA_V1, "x"), v1)
        self.assertIs(P.check_thread_record(v2, P.TRIAL_SCHEMA, "x"), v2)
        for bad in ({**v1, "intra_op_parallelism_threads": "0"}, {**v1, "inter_op_parallelism_threads": -1}, {**v1, "set_by_this_runner": 0},
                    {**v1, "logical_cpus": 0}, {**v1, "extra": 1}):
            with self.assertRaises(FR.Failure):
                P.check_thread_record(bad, P.TRIAL_SCHEMA_V1, "x")
        for bad in ({**v2, "requested": {**w, "extra": 1}}, {**v2, "effective": {"intra_op": 10}}, {**v2, "requested": {"intra_op": 0, "inter_op": 10}},
                    {**v2, "set_by_this_runner": 1}, {**v2, "environment": {"TF_NUM_INTRAOP_THREADS": None}},
                    {**v2, "environment": {"TF_NUM_INTRAOP_THREADS": None, "TF_NUM_INTEROP_THREADS": 10}}, {**v2, "extra": 1}):
            with self.assertRaises(FR.Failure):
                P.check_thread_record(bad, P.TRIAL_SCHEMA, "x")


class TestZReviewPlanAndCompare(unittest.TestCase):
    def setUp(self):
        self.w = world()
        self.ctx = injected(**hooks(self.w))
        self.ctx.__enter__()
        self.addCleanup(self.ctx.__exit__, None, None, None)
        self.tmp = self.w.tmp / "review3"
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def plan(self, folder):
        return run_main(self.w.argv("plan", "--out-dir", str(self.tmp / "plan"), "--trial-run", str(folder)))

    def test_plan_thread_record_exact(self):
        folder = self.tmp / "trial"
        good = fake_trial(self.w, folder)
        for over in ({"threads": {**good["threads"], "set_by_this_runner": 1}}, {"threads": {**good["threads"], "extra": 1}},
                     {"threads": {**good["threads"], "requested": {"intra_op": 10, "inter_op": 10, "x": 1}}}):
            fake_trial(self.w, folder, over=over)
            rc, o, err = self.plan(folder)
            self.assertEqual(rc, 1, over)
        fake_trial(self.w, folder, over={"machine": {**good["machine"], "ram_gb": None}})
        rc, o, err = self.plan(folder)
        self.assertEqual(rc, 0, err)
        self.assertIn("logical CPUs, RAM not reported, macOS", (self.tmp / "plan" / "run_plan.md").read_text())

    def cmp(self, new, ref, out=None):
        return run_main(["compare-trials", "--trial-run", str(new), "--reference-run", str(ref), "--compare-out-dir", str(out or self.tmp / "out")])

    def test_compare_record_checks(self):
        new, ref = self.tmp / "new", self.tmp / "ref"
        comparison_trial(ref, TestRCompareTrials.REF)
        for make, needle in ((lambda: comparison_trial(new, TestRCompareTrials.NEW, over={"config_sha256": "zz"}), "commit or config_sha256"),
                             (lambda: comparison_trial(new, TestRCompareTrials.NEW, over={"machine": {"processor": 5, "logical_cpus": 15, "ram_gb": 1.0,
                                                                                                    "os": "o"}}), "machine processor"),
                             (lambda: comparison_trial(new, TestRCompareTrials.NEW, over={"machine": {"processor": "p", "logical_cpus": 15,
                                                                                                    "ram_gb": float("nan"), "os": "o"}}), "ram_gb"),
                             (lambda: comparison_trial(new, TestRCompareTrials.NEW, schema=P.TRIAL_SCHEMA_V1,
                                                       over={"threads": {"logical_cpus": 10, "inter_op_parallelism_threads": "0",
                                                                         "intra_op_parallelism_threads": 0, "set_by_this_runner": False}}), "malformed values")):
            shutil.rmtree(new, ignore_errors=True)
            make()
            rc, o, err = self.cmp(new, ref)
            self.assertEqual(rc, 1)
            self.assertIn(needle, err)
            self.assertFalse((self.tmp / "out").exists())
        comparison_trial(new, TestRCompareTrials.NEW)
        (new / "status.json").write_text(json.dumps({**json.loads((new / "status.json").read_text()), "run_id": "x"}), encoding="utf-8")
        rc, o, err = self.cmp(new, ref)
        self.assertIn("ARCHTRIAL run id", err)
        big = {k: [1e308, 0.4, 0.3] for k in P.HISTORY_KEYS}
        small = {k: [-1e308, 0.4, 0.3] for k in P.HISTORY_KEYS}
        comparison_trial(new, TestRCompareTrials.NEW, hist={n: big for n in AM.NETWORKS})
        comparison_trial(ref, TestRCompareTrials.REF, hist={n: small for n in AM.NETWORKS})
        rc, o, err = self.cmp(new, ref)
        self.assertEqual(rc, 1)
        self.assertIn("RESULT: FAIL (compare-trials)", o)
        self.assertIn("not finite", err)
        self.assertFalse((self.tmp / "out").exists())

    def test_nested_trial_folders_both_directions(self):
        outer = self.tmp / "outer"
        comparison_trial(outer, TestRCompareTrials.REF)
        comparison_trial(outer / "inner", TestRCompareTrials.NEW)
        for new, ref in ((outer / "inner", outer), (outer, outer / "inner")):
            rc, o, err = self.cmp(new, ref)
            self.assertEqual(rc, 1)
            self.assertIn("different and not nested", err)

    def test_markdown_first_differing_epoch_column(self):
        """Il .md di compare-trials: nella colonna «first differing epoch» la prima epoca diversa (1), non quella dello
        scarto massimo (2): il difetto della PR #29 non torna nel nuovo testo."""
        base = {k: [0.5, 0.4, 0.3] for k in P.HISTORY_KEYS}
        other = json.loads(json.dumps(base))
        other["val_loss"] = [0.6, 1.4, 0.3]
        comparison_trial(self.tmp / "new", TestRCompareTrials.NEW, hist={"LivenessNet": other, "ResNet-18": base})
        comparison_trial(self.tmp / "ref", TestRCompareTrials.REF)
        self.assertEqual(self.cmp(self.tmp / "new", self.tmp / "ref")[0], 0)
        md = (self.tmp / "out" / "trial_comparison.md").read_text()
        self.assertIn("| LivenessNet | val_loss | 3 / 3 | 3 | 1 | 1 | 2 |", md)


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
