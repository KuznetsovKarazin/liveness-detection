"""
Test di scripts/n1_train.py e scripts/n1_trainer.py su dati SINTETICI, in cartelle temporanee: nessuna rete, nessuna
immagine, nessun dato reale, nessun training né inferenza veri.

Le fixture sono manifest NUAA finti con lo schema M2 (colonne, chiavi, campi derivati dal nome) e una cache .npz finta
con gli stessi membri (X_* uint8 N x 8 x 8 x 3 con l'indice codificato nei pixel, y_* int64, s_* <U4), generati a run
time con lo stesso campionamento `spread` e lo stesso GroupShuffleSplit del trainer. L'orchestrazione usa un trainer
finto iniettato (n1_train.TRAINER) e uno scorer finto deterministico (n1_train.MAKE_SCORER); n1_trainer.train è
provato con un modulo tensorflow FINTO (nessun TensorFlow serve né viene importato).

Coperti: configurazione (valori della scheda, tipi, chiavi, dati della scheda per un run non -trial), cartella dello
studio (nome, nuova, protetta, collegamento, ingressi separati), controlli prima del training (SHA-256 e schema della
cache, manifest, ordine di campionamento, collegamento y/s, split per SHA-256, C1 dentro il test, frame derivati),
trainer (seed di training e dello split distinti, iperparametri, checkpoint all'epoca di val_loss minimo, nessun test),
controllo dei pesi, punteggi su CPU (regola > 0,5, pareggio bona fide, NA per classe assente, ricalcolo di val_loss),
aggregato (tutti i seed, mediana e intervallo, scansione del pubblico, punteggi alterati), percorsi d'errore (nessun
PASS vecchio, uscite di train e score mai sovrascritte, log conservati) e un controllo finale che il repository vero
non sia stato toccato.

  python scripts/check_n1_train.py            # uscita 1 se un test fallisce; ultima riga RESULT: PASS / FAIL

Dipendenze: numpy e scikit-learn. Senza OpenCV (cv2) la suite registra un modulo cv2 vuoto, perché
src/tesi_app/evaluation.py lo importa solo per leggere immagini: lo dichiara con una riga NOTE.
"""
import atexit
import contextlib
import csv
import hashlib
import io
import json
import os
import shutil
import statistics
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
STUDY = "20261024-N1-LivenessNet-abcdef0-trial"

# cartelle dei dati, dei risultati e dei pesi in una cartella temporanea PRIMA di importare il codice
TMP_ROOT = Path(tempfile.mkdtemp(prefix="n1-train-tests-"))
atexit.register(shutil.rmtree, TMP_ROOT, True)          # tolta anche se la suite si interrompe (eccezione, Ctrl-C)
for _k, _s in (("LIVENESSLAB_DATA_DIR", "data"), ("LIVENESSLAB_RESULTS_DIR", "results"), ("LIVENESSLAB_WEIGHTS_DIR", "weights")):
    os.environ[_k] = str(TMP_ROOT / _s)
    (TMP_ROOT / _s).mkdir()
REPO_N1_BEFORE = (ROOT / "results" / "n1").exists()

# I soli motivi per cui un test si salta: ingressi mancanti dichiarati qui (nessuno di piattaforma)
MISSING_INPUT = {"delivered configuration": "n1_nuaa_stability.json is not next to the code (delivery/n1/, or ../ in the package)"}

CV2_NOTE = ""
try:
    import cv2  # noqa: F401
except ImportError:
    _cv2 = types.ModuleType("cv2")
    _cv2.__dict__["__getattr__"] = lambda n: (_ for _ in ()).throw(AttributeError(n)) if n.startswith("__") else 0
    sys.modules["cv2"] = _cv2
    CV2_NOTE = "NOTE: OpenCV not installed: an empty cv2 module is registered (evaluation.py uses it only to read images)"

sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402
import n1_train as N  # noqa: E402
import n1_trainer as T  # noqa: E402
import nuaa_manifests as nm  # noqa: E402

PER_CLASS = 12
SUBJECTS = tuple(f"{i:04d}" for i in range(1, 9))


def H(*parts) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()


def row(label: int, subject: str, session: str, pic: int) -> dict:
    name = f"{subject}_{subject}_0{pic % 2}_0{(pic % 3) if label else 0}_{session}_{pic}.jpg"
    return {"key": nm.make_key(label, name), "label": str(label), **nm.parse_name(name), "official_split": nm.SESSION_SPLIT[session],
            "sha256": H(label, name), "bytes": str(1000 + pic)}


def write_csv(p: Path, rows) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=nm.COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: r["key"]))
    p.write_text(buf.getvalue(), encoding="utf-8")


def frames(n: int, base: int) -> np.ndarray:
    X = np.zeros((n, 8, 8, 3), np.uint8)
    for i in range(n):
        v = base + i
        X[i, :, :, 0], X[i, :, :, 1], X[i, :, :, 2] = v // 65536, (v // 256) % 256, v % 256
    return X


def frame_id(X) -> np.ndarray:
    X = np.asarray(X)
    return X[:, 0, 0, 0].astype(int) * 65536 + X[:, 0, 0, 1].astype(int) * 256 + X[:, 0, 0, 2].astype(int)


class Fixture:
    """Manifest, cache e configurazione sintetici coerenti fra loro (stesso campionamento e stesso split del trainer)."""

    def __init__(self, base: Path):
        self.base = base
        self.man = base / "manifests"
        self.man.mkdir(parents=True)
        tr = [row(lab, s, sess, p) for s in SUBJECTS for lab in (0, 1) for sess in ("01", "02") for p in range(3)]
        te = [row(lab, s, "03", p) for s in SUBJECTS for lab in (0, 1) for p in range(4)]
        self.rows = {"official_train": tr, "official_test": te}
        by_tr, by_te = {r["key"]: r for r in tr}, {r["key"]: r for r in te}
        self.order_train = nm.spread_keys(tr, PER_CLASS)
        self.order_test = nm.spread_keys(te, PER_CLASS)
        y = np.array([int(by_tr[k]["label"]) for k in self.order_train], np.int64)
        s = np.array([by_tr[k]["subject"] for k in self.order_train])
        tr_i, va_i = T.split_indices(y, s, 42)
        self.tr_i, self.va_i = tr_i, va_i
        self.rows["cnn_train"] = [by_tr[self.order_train[i]] for i in tr_i]
        self.rows["cnn_es"] = [by_tr[self.order_train[i]] for i in va_i]
        self.rows["cnn_test"] = [by_te[k] for k in self.order_test]
        # C1: un frame su tre del test; il soggetto 0008 vi compare solo con attacchi (BPCER NA per soggetto)
        c1 = [by_te[k] for i, k in enumerate(self.order_test) if i % 3 == 0 and not (by_te[k]["subject"] == "0008" and by_te[k]["label"] == "0")]
        if not any(r["subject"] == "0008" for r in c1):
            c1.append(next(by_te[k] for k in self.order_test if by_te[k]["subject"] == "0008" and by_te[k]["label"] == "1"))
        self.rows["c1"] = c1
        self.files = {n: N.CARD_DATA["manifests"][n]["file"] for n in N.MANIFESTS}
        for n, rows in self.rows.items():
            write_csv(self.man / self.files[n], rows)
        self.cache = base / "nuaa_12.npz"
        self.arrays = {"X_train": frames(len(self.order_train), 0), "y_train": y, "s_train": s,
                       "X_test": frames(len(self.order_test), 10000),
                       "y_test": np.array([int(by_te[k]["label"]) for k in self.order_test], np.int64),
                       "s_test": np.array([by_te[k]["subject"] for k in self.order_test])}
        self.write_cache()
        self.config = base / "config.json"
        self.write_config()

    def write_cache(self, arrays=None):
        np.savez_compressed(self.cache, **(arrays or self.arrays))

    def data_block(self) -> dict:
        mem = N.npz_members(self.cache)
        return {"m2_run": "20261006-M2-seed42-test", "cache": {"file": self.cache.name, "sha256": N.sha256_file(self.cache), "per_class": PER_CLASS,
                "members": {k: {"dtype": mem[k]["dtype"], "shape": mem[k]["shape"]} for k in N.MEMBERS}},
                "manifests": {n: {"file": self.files[n], "sha256": N.sha256_file(self.man / self.files[n]), **N.counts_of(self.rows[n])}
                              for n in N.MANIFESTS},
                "previous_checkpoints_sha256": [H("old")]}

    def write_config(self, mutate=None, redeclare: bool = True):
        cfg = json.loads(json.dumps(N.FROZEN))
        cfg["data"] = self.data_block()
        if mutate:
            mutate(cfg)
        self.config.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        return cfg

    def rewrite(self, name: str, rows):
        """Manifest guasto con SHA-256 e conteggi ridichiarati nella configurazione: scatta la regola, non il checksum."""
        self.rows[name] = rows
        write_csv(self.man / self.files[name], rows)
        self.write_config()


# ============================================================================ trainer e scorer finti

def fake_prob(X, seed: int = 42) -> np.ndarray:
    """Probabilità finte deterministiche, diverse per seed (moltiplicatore 37, 41, 43 per 42, 43, 44)."""
    s = ((frame_id(X) * {42: 37, 43: 41, 44: 43}.get(seed, 37)) % 101) / 100.0
    s = np.where(frame_id(X) % 17 == 0, 0.5, s)          # alcuni pareggi esatti a 0,5
    return np.stack([1 - s, s], axis=1)


def fake_scorer(path, sha):
    import re
    m = re.search(r"seed(\d+)", str(path))          # cartella seed<N> del checkpoint
    seed = int(m.group(1)) if m else 42
    return lambda X: fake_prob(X, seed)


class FakeTrainer:
    """Trainer iniettato: registra gli argomenti, scrive pesi e scheda coerenti (val_loss minimo = val_loss dello scorer
    finto sulla validazione); `spoil` altera scheda o pesi dopo la scrittura."""

    def __init__(self, spoil=None, plan=None):
        self.calls, self.spoil, self.plan = [], spoil, plan

    def __call__(self, data, training, out_dir, train_seed, split_seed, split_check):
        self.calls.append({"keys": sorted(data), "train_seed": train_seed, "split_seed": split_seed, "training": training})
        tr_i, va_i = T.split_indices(data["y_train"], data["s_train"], split_seed)
        T.check_split(tr_i, va_i, split_check)
        best = N.val_loss(fake_prob(data["X_train"][va_i], train_seed), np.asarray(data["y_train"])[va_i])
        vl = self.plan(best) if self.plan else [best + 0.3, best + 0.1, best, best + 0.01, best + 0.02, best + 0.03, best + 0.04]
        out_dir = Path(out_dir)
        (out_dir / "LivenessNet__nuaa.h5").write_bytes(b"h5 seed %d" % train_seed)
        (out_dir / "LivenessNet__nuaa.keras").write_bytes(b"keras seed %d" % train_seed)
        b = int(np.argmin(vl))
        fired = len(vl) == b + 1 + training["early_stopping_patience"]
        card = {"arch": "LivenessNet", "dataset": "nuaa", "protocol": "official", "train_seed": train_seed, "split_seed": split_seed,
                "n_train": len(tr_i), "n_val": len(va_i), "val_split": "by subject", "max_epochs": training["max_epochs"],
                "epochs_run": len(vl), "restored_epoch": b + 1, "restored_by": "early stopping" if fired else "end of epochs (set_weights)",
                "best_val_loss": vl[b], "val_acc_at_best": 1.0, "history": {"loss": vl, "accuracy": [0.5] * len(vl), "val_loss": vl,
                "val_accuracy": [0.5] * len(vl), "lr": [0.001] * len(vl)}, "lr": training["initial_learning_rate"], "batch": training["batch"],
                "early_stopping_patience": training["early_stopping_patience"], "train_seconds": 1.0,
                "device": "CPU", "visible_devices": ["CPU"],
                "test_evaluation": "none (scores computed afterwards on CPU)",
                "split_check": "passed: fitting and early-stopping frames equal to the manifests by SHA-256",
                "weights_file": "LivenessNet__nuaa.h5", "weights_sha256": N.sha256_file(out_dir / "LivenessNet__nuaa.h5"),
                "keras_file": "LivenessNet__nuaa.keras", "keras_sha256": N.sha256_file(out_dir / "LivenessNet__nuaa.keras")}
        if self.spoil:
            self.spoil(card, out_dir)
        (out_dir / "LivenessNet__nuaa.json").write_text(json.dumps(card), encoding="utf-8")
        return card


# ============================================================================ tensorflow finto per n1_trainer.train

class _Any:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return _Any()

    def __getattr__(self, n):
        if n.startswith("__"):
            raise AttributeError(n)
        return _Any()


def fake_tf(log: dict, plan, gpu_stays: bool = False):
    tf = types.ModuleType("tensorflow")

    class ES:
        def __init__(self, monitor=None, patience=None, restore_best_weights=None):
            log["es"] = {"monitor": monitor, "patience": patience, "restore": restore_best_weights}
            self.patience, self.stopped_epoch, self.best_weights = patience, 0, None

    class RLR:
        def __init__(self, **k):
            log["rlr"] = k

    class DS:
        @staticmethod
        def from_tensor_slices(t):
            log.setdefault("slices", []).append(len(t[0]))
            return DS()

        def shuffle(self, n, seed=None):
            log["shuffle"] = (n, seed)
            return self

        def batch(self, b):
            log.setdefault("batch", []).append(b)
            return self

        def map(self, f):
            return self

        def prefetch(self, n):
            return self

    class Adam:
        def __init__(self, learning_rate=None, clipnorm=None):
            log["adam"] = (learning_rate, clipnorm)

    tf.random = types.SimpleNamespace(set_seed=lambda s: log.__setitem__("tf_seed", s))
    vis = {"devices": ["CPU", "GPU"]}

    def set_visible(devs, kind):
        log["hidden"] = (list(devs), kind)
        if gpu_stays:
            return
        vis["devices"] = ["CPU"]
    tf.config = types.SimpleNamespace(optimizer=types.SimpleNamespace(set_experimental_options=lambda o: log.__setitem__("opts", o)),
                                      set_visible_devices=set_visible,
                                      get_visible_devices=lambda: [types.SimpleNamespace(device_type=d) for d in vis["devices"]])
    tf.data = types.SimpleNamespace(Dataset=DS)
    tf.keras = types.SimpleNamespace(optimizers=types.SimpleNamespace(legacy=types.SimpleNamespace(Adam=Adam)), losses=_Any(), metrics=_Any(),
                                     Sequential=_Any, layers=_Any(), callbacks=types.SimpleNamespace(EarlyStopping=ES, ReduceLROnPlateau=RLR))

    class Model:
        def __init__(self):
            self.w = b"init"

        def compile(self, **k):
            pass

        def fit(self, ds, validation_data=None, epochs=None, class_weight=None, callbacks=None, verbose=None):
            log["fit"] = {"epochs": epochs, "class_weight": class_weight}
            vl = plan(epochs)
            es = callbacks[0]
            es.best_weights = b"best"
            b = int(np.argmin(vl))
            if len(vl) == b + 1 + es.patience:
                es.stopped_epoch = len(vl) - 1
                self.w = b"best"
            else:
                self.w = b"last"
            return types.SimpleNamespace(history={"loss": vl, "accuracy": [0.5] * len(vl), "val_loss": vl, "val_accuracy": [0.6] * len(vl),
                                                  "lr": [0.001] * len(vl)})

        def set_weights(self, w):
            log["set_weights"] = w
            self.w = w

        def predict(self, *a, **k):
            raise AssertionError("the trainer must not predict")

        def save(self, p):
            Path(p).write_bytes(b"keras:" + self.w)

        def save_weights(self, p, save_format=None):
            Path(p).write_bytes(b"h5:" + self.w)

    arch = types.SimpleNamespace(create_model=lambda name: (log.__setitem__("arch", name), types.SimpleNamespace(get_model=Model))[1])
    return tf, arch


@contextlib.contextmanager
def modules(mods: dict):
    saved = {k: sys.modules.get(k) for k in mods}
    try:
        for k, v in mods.items():
            sys.modules[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# ============================================================================ esecuzione dei comandi

def run(*args) -> tuple:
    """Esegue n1_train.main catturando stdout e stderr; restituisce (codice, testo)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = N.main(list(args))
    return code, out.getvalue() + err.getvalue()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="case-", dir=TMP_ROOT))
        self.fx = Fixture(self.tmp / "in")
        self.study = self.tmp / "out" / STUDY
        self.trainer = FakeTrainer()
        self.p = [mock.patch.object(N, "TRAINER", self.trainer), mock.patch.object(N, "MAKE_SCORER", fake_scorer),
                  mock.patch.object(N, "REPO_STATE", lambda d: {"commit": "abcdef0" + "0" * 33, "dirty": False, "dirty_paths": [], "dirty_excludes": None})]
        for x in self.p:
            x.start()

    def tearDown(self):
        for x in self.p:
            x.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def common(self, study=None):
        return ["--config", str(self.fx.config), "--manifests", str(self.fx.man), "--cache", str(self.fx.cache),
                "--study-dir", str(study or self.study)]

    def check(self, study=None):
        return run("check", *self.common(study))

    def train(self, seed=42, auth=True):
        return run("train", *self.common(), "--train-seed", str(seed), *(["--i-am-authorized"] if auth else []))

    def score(self, seed=42, auth=True):
        return run("score", *self.common(), "--train-seed", str(seed), *(["--i-am-authorized"] if auth else []))

    def aggregate(self, *extra):
        return run("aggregate", *self.common(), *extra)

    def status(self, d) -> dict:
        return json.loads((Path(d) / "status.json").read_text())

    def ok(self, res):
        self.assertEqual(res[0], 0, res[1])
        self.assertIn("RESULT: PASS", res[1])

    def ko(self, res, *needles):
        self.assertEqual(res[0], 1, res[1])
        self.assertIn("RESULT: FAIL", res[1])
        for n in needles:
            self.assertIn(n, res[1])

    def full(self):
        self.ok(self.check())
        for s in (42, 43, 44):
            self.ok(self.train(s))
            self.ok(self.score(s))


# ============================================================================ configurazione

class ConfigTests(Base):
    def test_valid_check_writes_only_its_files(self):
        res = self.check()
        self.ok(res)
        self.assertEqual(sorted(os.listdir(self.study)), ["check.json", "check.log", "status.json"])
        doc = json.loads((self.study / "check.json").read_text())
        self.assertTrue(doc["passed"])
        self.assertEqual(doc["checks"]["split"]["n_fitting"], len(self.fx.rows["cnn_train"]))
        self.assertEqual(doc["tensorflow"], "not imported")
        self.assertEqual(self.status(self.study)["result"], "PASS")
        self.assertIn("check PASS", (self.study / "check.log").read_text())

    def frozen_refusal(self, mutate, needle):
        self.fx.write_config(mutate)
        res = self.check()
        self.ko(res, needle)
        self.assertFalse(self.study.exists(), "nothing may be written when the configuration is refused")

    def test_epochs_13_refused(self):
        self.frozen_refusal(lambda c: c["training"].__setitem__("max_epochs", 13), "training differs from the N1 card")

    def test_float_epochs_refused(self):
        self.frozen_refusal(lambda c: c["training"].__setitem__("max_epochs", 12.0), "training differs from the N1 card")

    def test_other_seeds_refused(self):
        self.frozen_refusal(lambda c: c.__setitem__("train_seeds", [42, 43]), "train_seeds differs")

    def test_split_seed_refused(self):
        self.frozen_refusal(lambda c: c.__setitem__("split_seed", 43), "split_seed differs")

    def test_patience_lr_batch_refused(self):
        for k, v in (("early_stopping_patience", 5), ("initial_learning_rate", 0.01), ("batch", 32)):
            with self.subTest(k=k):
                self.frozen_refusal(lambda c: c["training"].__setitem__(k, v), "training differs")

    def test_extra_or_missing_key_refused(self):
        self.frozen_refusal(lambda c: c.__setitem__("best_seed", 42), "not allowed")
        self.frozen_refusal(lambda c: c.pop("scoring"), "not allowed")

    def test_bad_data_block_refused(self):
        self.frozen_refusal(lambda c: c["data"]["cache"].__setitem__("sha256", "ABC"), "data.cache.sha256")
        self.frozen_refusal(lambda c: c["data"]["manifests"]["cnn_es"].__setitem__("rows", True), "counts must be")

    def test_non_trial_requires_card_data(self):
        res = self.check(self.tmp / "out" / "20261024-N1-LivenessNet-abcdef0")
        self.ko(res, "non-trial run: data differs from the N1 card")
        self.assertFalse((self.tmp / "out").exists())

    def test_card_data_and_frozen_values(self):
        self.assertEqual(N.CARD_DATA["cache"]["sha256"], "dc286dc371e61d638da1842c062a38c5368a9ec4da07ccbc36b7fff6f3177920")
        self.assertEqual((N.CARD_DATA["manifests"]["cnn_train"]["rows"], N.CARD_DATA["manifests"]["cnn_es"]["rows"],
                          N.CARD_DATA["manifests"]["cnn_test"]["rows"], N.CARD_DATA["manifests"]["c1"]["rows"]), (2507, 493, 3000, 300))
        t = N.FROZEN["training"]
        self.assertEqual((t["max_epochs"], t["batch"], t["initial_learning_rate"], t["early_stopping_patience"]), (12, 16, 0.001, 4))
        self.assertEqual((N.FROZEN["train_seeds"], N.FROZEN["split_seed"]), ([42, 43, 44], 42))

    def test_delivered_configuration(self):
        for p in (ROOT / "delivery" / "n1" / "n1_nuaa_stability.json", ROOT.parent / "n1_nuaa_stability.json"):
            if p.is_file():
                cfg = json.loads(p.read_text())
                self.assertEqual(N.config_problems(cfg), [])
                self.assertEqual(cfg["data"], N.CARD_DATA)
                return
        self.skipTest("missing input: " + MISSING_INPUT["delivered configuration"])


# ============================================================================ cartella dello studio e ingressi

class StudyFolderTests(Base):
    def test_bad_name_refused(self):
        res = self.check(self.tmp / "out" / "20261024-N1-seed42-abcdef0-trial")
        self.ko(res, "is not YYYYMMDD-N1-LivenessNet")
        self.assertFalse((self.tmp / "out").exists())

    def test_non_empty_folder_refused_untouched(self):
        self.study.mkdir(parents=True)
        (self.study / ".DS_Store").write_text("x")
        self.ko(self.check(), "must be new")
        self.assertEqual(os.listdir(self.study), [".DS_Store"])

    def test_rerun_check_after_pass_refused(self):
        self.ok(self.check())
        before = (self.study / "check.json").read_bytes()
        self.ko(self.check(), "must be new")
        self.assertEqual((self.study / "check.json").read_bytes(), before)
        self.assertEqual(self.status(self.study)["result"], "PASS")

    def test_inside_repository_refused(self):
        for where in (ROOT / "scripts", ROOT / "results" / "m2", ROOT / "results"):
            with self.subTest(where=where.name):
                res = self.check(where / STUDY)
                self.ko(res, "study folder refused")
                self.assertFalse((where / STUDY).exists())

    def test_protected_moved_folder_refused(self):
        res = self.check(TMP_ROOT / "data" / STUDY)
        self.ko(res, "protected folder")
        self.assertFalse((TMP_ROOT / "data" / STUDY).exists())

    def test_symlink_refused(self):
        real = self.tmp / "real"
        real.mkdir()
        (self.tmp / "out").mkdir()
        self.study.symlink_to(real)
        self.ko(self.check(), "symbolic link")
        self.assertEqual(os.listdir(real), [])

    def test_inputs_inside_study_refused(self):
        self.study.mkdir(parents=True)
        man = self.study / "manifests"
        shutil.copytree(self.fx.man, man)
        res = run("check", "--config", str(self.fx.config), "--manifests", str(man), "--cache", str(self.fx.cache), "--study-dir", str(self.study))
        self.ko(res, "inputs and outputs must be separate")
        self.assertEqual(sorted(os.listdir(self.study)), ["manifests"])

    def test_outer_tree_refused(self):
        with mock.patch.object(N, "outer_roots", lambda: (self.tmp,)):
            self.ko(self.check(), "accepted packages")
        self.assertFalse(self.study.exists())


# ============================================================================ controlli prima del training

class PreRunCheckTests(Base):
    def failed_check(self, *needles):
        res = self.check()
        self.ko(res, *needles)
        self.assertFalse((self.study / "check.json").exists())
        self.assertEqual(self.status(self.study)["result"], "FAIL")
        self.assertIn("ERROR", (self.study / "check.log").read_text(), "the log of a failed check is kept")
        return res

    def test_cache_sha_mismatch(self):
        cfg = json.loads(self.fx.config.read_text())
        cfg["data"]["cache"]["sha256"] = H("other")
        self.fx.config.write_text(json.dumps(cfg))
        self.failed_check("SHA-256", "differs from the expected")

    def test_extra_member(self):
        self.fx.write_cache({**self.fx.arrays, "X_val": self.fx.arrays["X_train"][:2]})
        self.fx.write_config()
        self.failed_check("members")

    def test_member_dtype(self):
        self.fx.write_cache({**self.fx.arrays, "y_train": self.fx.arrays["y_train"].astype(np.int32)})
        cfg = json.loads(self.fx.config.read_text())
        cfg["data"]["cache"]["sha256"] = N.sha256_file(self.fx.cache)
        self.fx.config.write_text(json.dumps(cfg))
        self.failed_check("y_train has dtype <i4")

    def test_pickled_member_refused(self):
        arr = dict(self.fx.arrays)
        arr["s_train"] = np.array(list(arr["s_train"]), dtype=object)
        np.savez(self.fx.cache, **arr)
        cfg = json.loads(self.fx.config.read_text())
        cfg["data"]["cache"]["sha256"] = N.sha256_file(self.fx.cache)          # SHA ridichiarato: scatta lo schema
        self.fx.config.write_text(json.dumps(cfg))
        self.failed_check("pickle refused")

    def test_subject_array_swapped(self):
        s = self.fx.arrays["s_train"].copy()
        i, j = 0, len(s) - 1
        s[i], s[j] = s[j], s[i]
        self.fx.write_cache({**self.fx.arrays, "s_train": s})
        self.fx.write_config()
        self.failed_check("s_train: subjects differ")

    def test_label_flipped(self):
        y = self.fx.arrays["y_test"].copy()
        y[3] = 1 - y[3]
        self.fx.write_cache({**self.fx.arrays, "y_test": y})
        self.fx.write_config()
        self.failed_check("y_test: labels differ")

    def test_row_moved_between_fitting_and_validation(self):
        tr, es = list(self.fx.rows["cnn_train"]), list(self.fx.rows["cnn_es"])
        moved = next(r for r in tr if r["subject"] == es[0]["subject"]) if any(r["subject"] == es[0]["subject"] for r in tr) else tr[0]
        tr.remove(moved)
        es.append(moved)
        self.fx.rows["cnn_es"] = es
        write_csv(self.fx.man / self.fx.files["cnn_es"], es)
        self.fx.rewrite("cnn_train", tr)
        self.failed_check()

    def test_row_with_other_hash_than_official(self):
        te = [dict(r) for r in self.fx.rows["cnn_test"]]
        te[5]["sha256"] = H("swapped")
        self.fx.rewrite("cnn_test", te)
        self.failed_check("rows with the same key as official_test but different fields")

    def test_c1_outside_test(self):
        c1 = list(self.fx.rows["c1"])
        extra = next(r for r in self.fx.rows["official_test"] if r["key"] not in {x["key"] for x in self.fx.rows["cnn_test"]})
        self.fx.rewrite("c1", c1 + [extra])
        self.failed_check("keys are not in cnn_test")

    def test_test_manifest_not_the_sampling(self):
        te = list(self.fx.rows["cnn_test"])
        c1 = {r["key"] for r in self.fx.rows["c1"]}
        out = next(r for r in te if r["key"] not in c1)
        extra = next(r for r in self.fx.rows["official_test"] if r["key"] not in {x["key"] for x in te} and r["label"] == out["label"])
        self.fx.rewrite("cnn_test", [extra] + [r for r in te if r is not out])
        self.failed_check("the sampling order of the official test list does not give cnn_test")

    def test_duplicated_frame(self):
        X = self.fx.arrays["X_train"].copy()
        X[self.fx.va_i[0]] = X[self.fx.tr_i[0]]
        self.fx.write_cache({**self.fx.arrays, "X_train": X})
        self.fx.write_config()
        self.failed_check("derived frames")

    def test_train_test_shared_frame(self):
        X = self.fx.arrays["X_test"].copy()
        X[0] = self.fx.arrays["X_train"][0]
        self.fx.write_cache({**self.fx.arrays, "X_test": X})
        self.fx.write_config()
        self.failed_check("train_test_shared")

    def test_invalid_manifest_row(self):
        es = [dict(r) for r in self.fx.rows["cnn_es"]]
        es[0]["session"] = "99"
        self.fx.rewrite("cnn_es", es)
        self.failed_check("invalid input")

    def test_split_seed_matters(self):
        y, s = self.fx.arrays["y_train"], self.fx.arrays["s_train"]
        sha = [{r["key"]: r for r in self.fx.rows["official_train"]}[k]["sha256"] for k in self.fx.order_train]
        sc = {"sha256": sha, "train": [r["sha256"] for r in self.fx.rows["cnn_train"]], "val": [r["sha256"] for r in self.fx.rows["cnn_es"]]}
        T.check_split(*T.split_indices(y, s, 42), sc)
        with self.assertRaises(SystemExit):
            T.check_split(*T.split_indices(y, s, 43), sc)


# ============================================================================ trainer (tensorflow finto)

def plan_stop(epochs):
    return [0.9, 0.8, 0.7, 0.75, 0.76, 0.77, 0.78][:epochs]          # minimo all'epoca 3, arresto dopo l'epoca 7


def plan_decreasing(epochs):
    return [1.0 - 0.05 * i for i in range(epochs)]


class TrainerTests(Base):
    def call(self, plan, seed=43, split_seed=42, data=None, out=None, gpu_stays=False):
        log = {}
        tf, arch = fake_tf(log, plan, gpu_stays)
        out = out or (self.tmp / "ck")
        out.mkdir(exist_ok=True)
        sha = [{r["key"]: r for r in self.fx.rows["official_train"]}[k]["sha256"] for k in self.fx.order_train]
        sc = {"sha256": sha, "train": [r["sha256"] for r in self.fx.rows["cnn_train"]], "val": [r["sha256"] for r in self.fx.rows["cnn_es"]]}
        data = data or {k: self.fx.arrays[k] for k in ("X_train", "y_train", "s_train")}
        with modules({"tensorflow": tf}), contextlib.redirect_stdout(io.StringIO()):
            card = T.train(data, N.FROZEN["training"], out, seed, split_seed, sc, arch)
        return card, log, out

    def test_seeds_and_hyperparameters(self):
        card, log, out = self.call(plan_stop, seed=43)
        self.assertEqual(log["tf_seed"], 43)
        self.assertEqual(log["shuffle"], (4096, 43))
        self.assertEqual(log["es"], {"monitor": "val_loss", "patience": 4, "restore": True})
        self.assertEqual(log["rlr"], {"monitor": "val_loss", "factor": 0.5, "patience": 2, "min_lr": 1e-6})
        self.assertEqual(log["adam"], (0.001, 1.0))
        self.assertEqual(log["fit"]["epochs"], 12)
        self.assertEqual(log["batch"], [16, 16])
        self.assertEqual(log["arch"], "LivenessNet")
        self.assertEqual(log["slices"], [len(self.fx.tr_i), len(self.fx.va_i)])
        self.assertEqual((card["train_seed"], card["split_seed"], card["n_train"], card["n_val"]), (43, 42, len(self.fx.tr_i), len(self.fx.va_i)))
        self.assertEqual(N.weights_problems(out, json.loads(self.fx.config.read_text()), 43), [])

    def test_cpu_only(self):
        card, log, out = self.call(plan_stop)
        self.assertEqual(log["hidden"], ([], "GPU"))
        self.assertEqual((card["device"], card["visible_devices"]), ("CPU", ["CPU"]))

    def test_gpu_still_visible_refused(self):
        with self.assertRaises(SystemExit) as cm:
            self.call(plan_stop, gpu_stays=True)
        self.assertIn("CPU only expected", str(cm.exception))
        self.assertFalse((self.tmp / "ck" / "LivenessNet__nuaa.h5").exists())

    def test_python_and_numpy_seeds(self):
        import random
        self.call(plan_stop, seed=44)
        a = (random.random(), float(np.random.random()))
        random.seed(44); np.random.seed(44)
        self.assertEqual(a, (random.random(), float(np.random.random())))

    def test_early_stopping_restores_minimum(self):
        card, log, out = self.call(plan_stop)
        self.assertEqual((card["epochs_run"], card["restored_epoch"], card["restored_by"]), (7, 3, "early stopping"))
        self.assertNotIn("set_weights", log)
        self.assertEqual((out / "LivenessNet__nuaa.h5").read_bytes(), b"h5:best")

    def test_end_of_epochs_restores_best(self):
        card, log, out = self.call(plan_decreasing)
        self.assertEqual((card["epochs_run"], card["restored_epoch"], card["restored_by"]), (12, 12, "end of epochs (set_weights)"))
        self.assertEqual(log["set_weights"], b"best")
        self.assertEqual((out / "LivenessNet__nuaa.h5").read_bytes(), b"h5:best")
        self.assertEqual(card["weights_sha256"], N.sha256_file(out / "LivenessNet__nuaa.h5"))

    def test_other_split_seed_refused(self):
        with self.assertRaises(SystemExit) as cm:
            self.call(plan_stop, split_seed=43)
        self.assertIn("split check failed", str(cm.exception))

    def test_test_frames_never_given(self):
        data = {k: self.fx.arrays[k] for k in ("X_train", "y_train", "s_train", "X_test")}
        with self.assertRaises(SystemExit) as cm:
            self.call(plan_stop, data=data)
        self.assertIn("only X_train, y_train and s_train", str(cm.exception))

    def test_existing_outputs_never_overwritten(self):
        out = self.tmp / "ck"
        out.mkdir()
        (out / "LivenessNet__nuaa.h5").write_bytes(b"old")
        with self.assertRaises(SystemExit):
            self.call(plan_stop, out=out)
        self.assertEqual((out / "LivenessNet__nuaa.h5").read_bytes(), b"old")


# ============================================================================ train (orchestrazione)

class TrainTests(Base):
    def test_train_pass(self):
        self.ok(self.check())
        res = self.train(43)
        self.ok(res)
        sd = self.study / "seed43"
        self.assertEqual(sorted(os.listdir(sd)), ["checkpoints", "status.json", "train.json", "train.log"])
        self.assertEqual(self.trainer.calls[0]["keys"], ["X_train", "s_train", "y_train"])
        self.assertEqual((self.trainer.calls[0]["train_seed"], self.trainer.calls[0]["split_seed"]), (43, 42))
        doc = json.loads((sd / "train.json").read_text())
        self.assertEqual((doc["train_seed"], doc["run_id"], doc["check_sha256"]), (43, STUDY, N.sha256_file(self.study / "check.json")))
        self.assertEqual(doc["weights_sha256"], N.sha256_file(sd / "checkpoints" / "LivenessNet__nuaa.h5"))
        self.assertEqual(self.status(sd)["result"], "PASS")

    def test_needs_authorization(self):
        self.ok(self.check())
        self.ko(self.train(auth=False), "--i-am-authorized")
        self.assertFalse((self.study / "seed42").exists())

    def test_seed_outside_configuration(self):
        self.ok(self.check())
        self.ko(self.train(45), "not one of the configuration seeds")
        self.assertFalse((self.study / "seed45").exists())

    def test_without_check(self):
        self.study.mkdir(parents=True)
        self.ko(self.train(), "missing")
        self.assertEqual(os.listdir(self.study), [])

    def test_seed_folder_never_reused(self):
        self.ok(self.check())
        self.ok(self.train())
        before = {p.name: p.read_bytes() for p in (self.study / "seed42").iterdir() if p.is_file()}
        self.ko(self.train(), "never reused")
        self.assertEqual({p.name: p.read_bytes() for p in (self.study / "seed42").iterdir() if p.is_file()}, before)
        self.assertEqual(len(self.trainer.calls), 1)

    def test_code_changed_since_check(self):
        self.ok(self.check())
        with mock.patch.object(N, "code_hashes", lambda: {"scripts/n1_train.py": "0" * 64}):
            self.ko(self.train(), "the code changed since check")
        self.assertFalse((self.study / "seed42").exists())

    def test_inputs_changed_after_check(self):
        self.ok(self.check())
        cfg = json.loads(self.fx.config.read_text())
        self.fx.config.write_text(json.dumps(cfg, indent=2))          # stessa configurazione, byte diversi
        self.ko(self.train(), "belongs to another step, study or configuration")

    def spoiled(self, spoil, needle, plan=None):
        self.ok(self.check())
        with mock.patch.object(N, "TRAINER", FakeTrainer(spoil, plan)):
            res = self.train()
        self.ko(res, "weights check failed", needle)
        sd = self.study / "seed42"
        self.assertFalse((sd / "train.json").exists())
        self.assertEqual(self.status(sd)["result"], "FAIL")
        self.assertTrue((sd / "train.log").is_file() and (sd / "checkpoints" / "LivenessNet__nuaa.h5").is_file(), "failed outcomes are kept")

    def test_weights_hash_differs(self):
        self.spoiled(lambda c, d: (d / "LivenessNet__nuaa.h5").write_bytes(b"changed"), "SHA-256 differs from the card")

    def test_restored_epoch_not_minimum(self):
        self.spoiled(lambda c, d: c.__setitem__("restored_epoch", c["restored_epoch"] + 1), "not the epoch of minimum val_loss")

    def test_previous_checkpoint(self):
        def spoil(c, d):
            (d / "LivenessNet__nuaa.h5").write_bytes(b"old weights")
            c["weights_sha256"] = N.sha256_file(d / "LivenessNet__nuaa.h5")
        self.fx.write_config(lambda c: c["data"]["previous_checkpoints_sha256"].append(hashlib.sha256(b"old weights").hexdigest()))
        self.spoiled(spoil, "previous checkpoint")

    def test_stopped_without_patience(self):
        self.spoiled(lambda c, d: None, "not 4 epochs after the minimum", plan=lambda b: [b + 0.2, b, b + 0.1])

    def test_card_device_differs(self):
        self.spoiled(lambda c, d: c.__setitem__("device", "GPU"), "card field device")

    def test_card_seed_differs(self):
        self.spoiled(lambda c, d: c.__setitem__("train_seed", 43), "train_seed")

    def test_epochs_over_budget(self):
        self.spoiled(lambda c, d: None, "outside 1..12", plan=lambda b: [b + 0.01 * (13 - i) for i in range(13)])

    def test_trainer_crash_kept_as_failure(self):
        self.ok(self.check())

        def boom(*a, **k):
            raise RuntimeError("out of memory")
        with mock.patch.object(N, "TRAINER", boom):
            self.ko(self.train(), "unexpected RuntimeError")
        sd = self.study / "seed42"
        self.assertEqual(self.status(sd)["result"], "FAIL")
        self.assertIn("out of memory", (sd / "train.log").read_text())
        self.assertFalse((sd / "train.json").exists())


# ============================================================================ score e metriche

class ScoreTests(Base):
    def setUp(self):
        super().setUp()
        self.ok(self.check())
        self.ok(self.train())
        self.sd = self.study / "seed42"

    def test_score_pass(self):
        self.ok(self.score())
        self.assertEqual(sorted(os.listdir(self.sd)), sorted(["checkpoints", "status.json", "train.json", "train.log", "score.log", *N.SCORE_OUTPUTS]))
        rows = list(csv.DictReader((self.sd / "scores_test.csv").open()))
        self.assertEqual(len(rows), len(self.fx.rows["cnn_test"]))
        self.assertEqual(sum(r["in_c1_subset"] == "1" for r in rows), len(self.fx.rows["c1"]))
        ties = [r for r in rows if float(r["score_attack"]) == 0.5]
        self.assertTrue(ties and all(r["prediction"] == "0" for r in ties), "a tie at 0.5 is bona fide")
        self.assertTrue(all(r["prediction"] == ("1" if float(r["score_attack"]) > 0.5 else "0") for r in rows))
        # metriche indipendenti dal codice: conteggi a mano
        met = json.loads((self.sd / "metrics.json").read_text())["metrics"]
        for name, sel in (("test_3000", rows), ("c1_300", [r for r in rows if r["in_c1_subset"] == "1"])):
            att = [r for r in sel if r["label"] == "1"]
            bf = [r for r in sel if r["label"] == "0"]
            acc = sum(float(r["score_attack"]) <= 0.5 for r in att)
            rej = sum(float(r["score_attack"]) > 0.5 for r in bf)
            m = met[name]
            self.assertEqual((m["accepted_attacks"], m["n_attack"], m["rejected_bona_fide"], m["n_bona_fide"]), (acc, len(att), rej, len(bf)))
            self.assertAlmostEqual(m["apcer"], acc / len(att), places=15)
            self.assertAlmostEqual(m["acer"], (acc / len(att) + rej / len(bf)) / 2, places=15)
            pairs = [(float(a["score_attack"]), float(b["score_attack"])) for a in att for b in bf]
            auc = sum(1.0 if x > z else 0.5 if x == z else 0.0 for x, z in pairs) / len(pairs)
            self.assertAlmostEqual(m["auroc"], auc, places=12)
        subj = list(csv.DictReader((self.sd / "per_subject.csv").open()))
        s8 = [r for r in subj if r["set"] == "c1_300" and r["subject_id"] == "0008"]
        self.assertEqual((s8[0]["n_bona_fide"], s8[0]["bpcer"], s8[0]["acer"]), ("0", "NA", "NA"))
        self.assertNotEqual(s8[0]["apcer"], "NA")
        doc = json.loads((self.sd / "score.json").read_text())
        self.assertEqual(doc["device"], "CPU")
        self.assertLess(doc["val_loss_abs_diff"], 1e-12)

    def test_scores_never_recomputed(self):
        self.ok(self.score())
        before = (self.sd / "scores_test.csv").read_bytes()
        self.ko(self.score(), "never recomputed in place")
        self.assertEqual((self.sd / "scores_test.csv").read_bytes(), before)
        self.assertEqual(self.status(self.sd)["result"], "PASS")

    def no_outputs(self):
        self.assertFalse(any((self.sd / n).exists() for n in N.SCORE_OUTPUTS))
        self.assertEqual(self.status(self.sd)["result"], "FAIL")

    def test_weights_changed_after_train(self):
        (self.sd / "checkpoints" / "LivenessNet__nuaa.h5").write_bytes(b"other")
        self.ko(self.score(), "weights check failed")
        self.no_outputs()

    def test_val_loss_not_reproduced(self):
        with mock.patch.object(N, "MAKE_SCORER", lambda p, s: (lambda X: fake_prob(X)[:, ::-1])):
            self.ko(self.score(), "not the one of minimum val_loss")
        self.no_outputs()

    def test_nan_scores(self):
        with mock.patch.object(N, "MAKE_SCORER", lambda p, s: (lambda X: np.full((len(X), 2), np.nan))):
            self.ko(self.score(), "non-finite")
        self.no_outputs()

    def test_train_record_of_another_study(self):
        doc = json.loads((self.sd / "train.json").read_text())
        doc["run_id"] = "20261024-N1-LivenessNet-1234567-trial"
        (self.sd / "train.json").write_text(json.dumps(doc))
        self.ko(self.score(), "records of another run are refused")
        self.assertFalse(any((self.sd / n).exists() for n in N.SCORE_OUTPUTS))

    def test_needs_authorization(self):
        self.ko(self.score(auth=False), "--i-am-authorized")
        self.assertFalse(any((self.sd / n).exists() for n in N.SCORE_OUTPUTS))

    def test_set_metrics_na_and_ties(self):
        m = N.set_metrics([1, 1, 1], [0.2, 0.5, 0.9])
        self.assertEqual((m["bpcer"], m["acer"], m["auroc"], m["n_bona_fide"]), (None, None, None, 0))
        self.assertAlmostEqual(m["apcer"], 2 / 3)
        self.assertEqual(m["ties_at_threshold"], 1)

    def test_val_loss_formula(self):
        p = np.array([[0.9, 0.1], [0.2, 0.8], [1.0, 0.0]])
        y = np.array([0, 1, 1])
        want = -(np.log(0.9) + np.log(0.8) + np.log(1e-7)) / 3
        self.assertAlmostEqual(N.val_loss(p, y), want, places=9)


# ============================================================================ aggregato

class AggregateTests(Base):
    def test_full_pipeline(self):
        self.full()
        res = self.aggregate()
        self.ok(res)
        doc = json.loads((self.study / "aggregate.json").read_text())
        self.assertEqual(doc["train_seeds"], [42, 43, 44])
        for name in N.SETS:
            for met in ("apcer", "bpcer", "acer", "auroc"):
                vals = [json.loads((self.study / f"seed{s}" / "metrics.json").read_text())["metrics"][name][met] for s in (42, 43, 44)]
                s = doc["summary"][name][met]
                self.assertEqual(s["values"], vals)
                self.assertEqual((s["median"], s["min"], s["max"]), (statistics.median(vals), min(vals), max(vals)))
        # mediana, non media: almeno una metrica ha tre valori distinti con media diversa dalla mediana
        self.assertTrue(any(len(set(v)) == 3 and statistics.mean(v) != statistics.median(v)
                            for v in (doc["summary"][n][m]["values"] for n in N.SETS for m in ("apcer", "bpcer", "acer", "auroc"))))
        md = (self.study / "aggregate.md").read_text()
        self.assertNotIn("real/", md)
        self.assertNotIn("attack/", md)
        for subj in SUBJECTS:
            self.assertNotIn(subj, md + (self.study / "aggregate.json").read_text())
        # intervallo stampato come minimo-massimo, con i valori esatti della tabella
        apcer = doc["summary"]["test_3000"]["apcer"]
        row = next(l for l in md.splitlines() if l.startswith("| APCER |"))
        self.assertTrue(row.endswith(f"| {apcer['min']:.4f}-{apcer['max']:.4f} |"), row)
        # tabella per soggetto di tutti i seed: le righe dei tre per_subject.csv, nello stesso ordine
        allrows = list(csv.DictReader((self.study / "per_subject_all_seeds.csv").open()))
        per_seed = [r for s in (42, 43, 44) for r in csv.DictReader((self.study / f"seed{s}" / "per_subject.csv").open())]
        self.assertEqual(allrows, per_seed)
        self.assertEqual({r["train_seed"] for r in allrows}, {"42", "43", "44"})
        self.assertEqual(self.status(self.study)["result"], "PASS")

    def test_public_scan_hit_writes_nothing(self):
        self.full()
        with mock.patch.object(N, "public_problems", lambda text, inp: ["subject identifiers"]):
            self.ko(self.aggregate(), "aggregate.json would hold subject identifiers: not written")
        self.assertFalse(any((self.study / n).exists() for n in N.AGG_OUTPUTS))
        self.assertEqual(self.status(self.study)["result"], "FAIL")

    def test_no_seed_selection(self):
        self.full()
        self.ko(self.aggregate("--train-seed", "42"), "no seed selection")
        self.assertFalse((self.study / "aggregate.json").exists())

    def test_missing_seed(self):
        self.ok(self.check())
        for s in (42, 43):
            self.ok(self.train(s))
            self.ok(self.score(s))
        self.ko(self.aggregate(), "seed44")
        self.assertFalse((self.study / "aggregate.json").exists())

    def test_tampered_scores_after_pass_leave_no_pass(self):
        self.full()
        self.ok(self.aggregate())
        sd = self.study / "seed43"
        rows = list(csv.DictReader((sd / "scores_test.csv").open()))
        r = next(x for x in rows if x["label"] == "1" and x["prediction"] == "1")
        r["score_attack"], r["prediction"] = "0.1", "0"
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=N.SCORE_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        (sd / "scores_test.csv").write_text(buf.getvalue())
        sc = json.loads((sd / "score.json").read_text())
        sc["files"]["scores_test.csv"] = N.sha256_text(buf.getvalue())         # SHA ridichiarato: deve scattare il ricalcolo
        (sd / "score.json").write_text(json.dumps(sc))
        self.ko(self.aggregate(), "metrics recomputed from the scores differ")
        self.assertFalse(any((self.study / n).exists() for n in N.AGG_OUTPUTS), "no previous PASS output may remain")
        self.assertEqual(self.status(self.study)["result"], "FAIL")

    def test_scores_changed_after_scoring(self):
        self.full()
        p = self.study / "seed44" / "scores_test.csv"
        p.write_text(p.read_text().replace("\n", "\r\n"))
        self.ko(self.aggregate(), "changed after scoring")

    def rewrite_scores(self, sd, change):
        rows = list(csv.DictReader((sd / "scores_test.csv").open()))
        change(rows)
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=N.SCORE_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        (sd / "scores_test.csv").write_text(buf.getvalue())
        sc = json.loads((sd / "score.json").read_text())
        sc["files"]["scores_test.csv"] = N.sha256_text(buf.getvalue())         # SHA ridichiarato
        (sd / "score.json").write_text(json.dumps(sc))

    def test_score_row_with_wrong_c1_flag(self):
        self.full()
        self.rewrite_scores(self.study / "seed43", lambda rows: rows[0].__setitem__("in_c1_subset", "0" if rows[0]["in_c1_subset"] == "1" else "1"))
        self.ko(self.aggregate(), "rows inconsistent with the manifests")
        self.assertFalse(any((self.study / n).exists() for n in N.AGG_OUTPUTS))

    def test_train_record_of_another_seed(self):
        self.full()
        sd = self.study / "seed43"
        tr = json.loads((sd / "train.json").read_text())
        tr["train_seed"] = 42
        (sd / "train.json").write_text(json.dumps(tr))
        sc = json.loads((sd / "score.json").read_text())
        sc["train_sha256"] = N.sha256_file(sd / "train.json")                # legame score → train ridichiarato
        (sd / "score.json").write_text(json.dumps(sc))
        self.ko(self.aggregate(), "seed43: records of another seed")
        self.assertFalse(any((self.study / n).exists() for n in N.AGG_OUTPUTS))

    def test_score_record_of_other_weights(self):
        self.full()
        sd = self.study / "seed44"
        sc = json.loads((sd / "score.json").read_text())
        sc["weights_sha256"] = H("other weights")
        (sd / "score.json").write_text(json.dumps(sc))
        self.ko(self.aggregate(), "seed44: score.json does not belong to its train.json")
        self.assertFalse(any((self.study / n).exists() for n in N.AGG_OUTPUTS))

    def test_summary_with_na(self):
        self.assertEqual(N.summary([0.1, None, 0.3]), {"values": [0.1, None, 0.3], "median": None, "min": None, "max": None})
        self.assertEqual(N.summary([None, None, None])["median"], None)
        self.assertEqual(N.summary([0.3, 0.1, 0.2]), {"values": [0.3, 0.1, 0.2], "median": 0.2, "min": 0.1, "max": 0.3})

    def test_score_row_with_wrong_label(self):
        self.full()
        sd = self.study / "seed42"
        txt = (sd / "scores_test.csv").read_text().splitlines()
        h, first = txt[0].split(","), txt[1].split(",")
        first[h.index("label")] = "1" if first[h.index("label")] == "0" else "0"
        txt[1] = ",".join(first)
        new = "\n".join(txt) + "\n"
        (sd / "scores_test.csv").write_text(new)
        sc = json.loads((sd / "score.json").read_text())
        sc["files"]["scores_test.csv"] = N.sha256_text(new)
        (sd / "score.json").write_text(json.dumps(sc))
        self.ko(self.aggregate(), "rows inconsistent with the manifests")

    def test_same_weights_refused(self):
        self.ok(self.check())
        for s in (42, 43, 44):
            spoil = None if s == 42 else (lambda c, d, s=s: (d.joinpath("LivenessNet__nuaa.h5").write_bytes(b"h5 seed 42"),
                                                              c.__setitem__("weights_sha256", N.sha256_file(d / "LivenessNet__nuaa.h5"))))
            with mock.patch.object(N, "TRAINER", FakeTrainer(spoil)):
                self.ok(self.train(s))
            self.ok(self.score(s))
        self.ko(self.aggregate(), "same weights")

    def test_folder_named_as_output_refused_before_removal(self):
        self.full()
        self.ok(self.aggregate())
        (self.study / "aggregate.md").unlink()
        (self.study / "aggregate.md").mkdir()
        self.ko(self.aggregate(), "not a regular file")
        self.assertTrue((self.study / "aggregate.json").is_file(), "refused before any removal")

    def test_public_scan(self):
        self.ok(self.check())
        inp = N.load_inputs(json.loads(self.fx.config.read_text()), types.SimpleNamespace(cache=str(self.fx.cache), manifests=str(self.fx.man)), False)
        self.assertEqual(N.public_problems("| APCER | 0.0013 | 2,507 frames |", inp), [])
        self.assertIn("subject identifiers", N.public_problems("worst subject 0003", inp))
        self.assertIn("image keys", N.public_problems(self.fx.rows["cnn_test"][0]["key"], inp))
        self.assertIn("image SHA-256 values", N.public_problems(self.fx.rows["cnn_test"][0]["sha256"], inp))
        self.assertTrue(N.public_problems("/Us" + "ers/someone/file", inp))


# ============================================================================ ambiente

class EnvironmentTests(unittest.TestCase):
    def test_tensorflow_never_imported(self):
        self.assertFalse([m for m in sys.modules if m.split(".")[0] in ("tensorflow", "keras", "tf_keras")])

    def test_select_threshold_unchanged(self):
        # i moduli accettati non sono toccati: SHA-256 di select_threshold.py consegnato con M1 e M2
        self.assertEqual(N.sha256_file(ROOT / "scripts" / "select_threshold.py"),
                         "cacb709672c9c7be5cd68d61923e79bb640e1e8ee77a78864f6786577d07e93c")


N1_CODE = ("n1_train.py", "n1_trainer.py", "n1_metal_relu_repro.py", "n1_metal_diagnostic.py", "n1_metal_checks.py")
LIBRARIES = {"np", "numpy", "sklearn", "tf", "tensorflow", "keras", "npy_format"}

BLOCK_TF = r'''
import runpy, sys
class _B:
    def find_spec(self, name, path=None, target=None):
        if name == "tensorflow" or name.startswith("tensorflow."):
            raise ModuleNotFoundError("No module named 'tensorflow' (blocked by the test)", name=name)
        return None
sys.meta_path.insert(0, _B())
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
'''


def private_library_uses(src: str) -> list:
    """Attributi privati (un solo trattino basso iniziale) di numpy, scikit-learn, TensorFlow e Keras, e import da loro
    moduli privati: API non stabili fra le versioni (per esempio numpy.lib.format._read_array_header assente in NumPy 2)."""
    import ast
    tree = ast.parse(src)
    roots = ("numpy", "sklearn", "tensorflow", "keras")
    names = set(LIBRARIES)
    for n in ast.walk(tree):                     # nomi legati a un modulo di libreria da un import
        if isinstance(n, ast.Import):
            for x in n.names:
                if x.name.split(".")[0] in roots:
                    names.add((x.asname or x.name).split(".")[0])
        if isinstance(n, ast.ImportFrom) and n.module and n.module.split(".")[0] in roots:
            names.update(x.asname or x.name for x in n.names)

    def base(e):
        while isinstance(e, (ast.Attribute, ast.Call, ast.Subscript)):
            e = e.func if isinstance(e, ast.Call) else e.value
        return e.id if isinstance(e, ast.Name) else None
    changed = True
    while changed:                               # alias per assegnazione (fmt = npy_format, f = np.lib.format, ...)
        changed = False
        for n in ast.walk(tree):
            if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None and base(n.value) in names:
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    if isinstance(t, ast.Name) and t.id not in names:
                        names.add(t.id)
                        changed = True
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr.startswith("_") and not n.attr.startswith("__"):
            if base(n.value) in names:
                out.append(f"line {n.lineno}: {ast.unparse(n)}")
        if isinstance(n, ast.ImportFrom) and n.module and n.module.split(".")[0] in ("numpy", "sklearn", "tensorflow", "keras"):
            if any(p.startswith("_") for p in n.module.split(".")) or any(x.name.startswith("_") for x in n.names):
                out.append(f"line {n.lineno}: from {n.module} import ...")
    return out


class PortabilityTests(unittest.TestCase):
    def test_no_private_library_api(self):
        for f in N1_CODE:
            with self.subTest(file=f):
                self.assertEqual(private_library_uses((ROOT / "scripts" / f).read_text(encoding="utf-8")), [])

    def test_scan_detects_private_api(self):
        self.assertTrue(private_library_uses("import numpy as np\nnp.lib.format._read_array_header(f, v)\n"))
        self.assertTrue(private_library_uses("from sklearn.utils._param_validation import x\n"))
        self.assertTrue(private_library_uses("import numpy\nfmt = numpy.lib.format\nfmt._x(1)\n"))
        self.assertTrue(private_library_uses("import numpy.lib.format as npy_format\nfmt = npy_format\nfmt._read_array_header(f, v)\n"))
        self.assertTrue(private_library_uses("from numpy.lib import format as F\nF._read_array_header(f, v)\n"))
        self.assertEqual(private_library_uses("import numpy as np\nnp.lib.format.read_magic(f)\nx.__class__\n"), [])

    def npz_with(self, version):
        import zipfile
        import numpy.lib.format as fmt
        p = TMP_ROOT / f"v{version[0]}.npz"
        with zipfile.ZipFile(p, "w") as zf:
            with zf.open("y_train.npy", "w") as fh:
                fmt.write_array(fh, np.arange(4, dtype=np.int64), version=version)
        return p

    def test_npy_versions_1_and_2_read_with_public_functions(self):
        for v in ((1, 0), (2, 0)):
            with self.subTest(version=v):
                self.assertEqual(N.npz_members(self.npz_with(v)), {"y_train": {"dtype": "<i8", "shape": [4], "fortran": False}})

    def test_npy_version_3_refused(self):
        with self.assertRaises(N.N1Error) as cm:
            N.npz_members(self.npz_with((3, 0)))
        self.assertIn("format version 3.0 not supported", str(cm.exception))

    def test_tensorflow_scripts_without_tensorflow(self):
        import subprocess
        boot = TMP_ROOT / "block_tf.py"
        boot.write_text(BLOCK_TF, encoding="utf-8")
        for args in (["n1_metal_relu_repro.py"], ["n1_metal_diagnostic.py"], ["n1_metal_checks.py", "fit-steps"],
                     ["n1_metal_checks.py", "c1-linear", "--weights", str(TMP_ROOT / "absent.h5")]):
            with self.subTest(args=args):
                r = subprocess.run([sys.executable, str(boot), str(ROOT / "scripts" / args[0]), *args[1:]], capture_output=True,
                                   text=True, timeout=120)
                self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
                self.assertIn("SKIP: TensorFlow not installed (missing input)", r.stdout)
                self.assertNotIn("Traceback", r.stderr)


class C1RescoreTests(Base):
    """n1_metal_checks.py c1-rescore senza TensorFlow: ingressi controllati prima (ERROR, codice 1, nessun traceback),
    poi SKIP e codice 2; SHA-256 del checkpoint e dei punteggi C1 sostituiti con quelli di file sintetici."""

    def setUp(self):
        super().setUp()
        import n1_metal_checks as MC
        self.MC = MC
        self.w = self.tmp / "LivenessNet__nuaa.h5"
        self.w.write_bytes(b"synthetic C1 weights")
        self.sc = self.tmp / "scores.csv"
        self.sc.write_text("sample_id,label,score_attack,analyzer_id\n", encoding="utf-8")
        self.p += [mock.patch.object(MC, "C1_WEIGHTS_SHA256", N.sha256_file(self.w)),
                   mock.patch.object(MC, "C1_SCORES_SHA256", N.sha256_file(self.sc))]
        for x in self.p[-2:]:
            x.start()

    def call(self, **over):
        a = types.SimpleNamespace(config=str(self.fx.config), manifests=str(self.fx.man), cache=str(self.fx.cache),
                                  weights=str(self.w), c1_scores=str(self.sc))
        for k, v in over.items():
            setattr(a, k, v)
        out = io.StringIO()
        with modules({"tensorflow": None}), contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = self.MC.c1_rescore(a)
        return code, out.getvalue()

    def test_skip_without_tensorflow(self):
        code, text = self.call()
        self.assertEqual(code, 2, text)
        self.assertIn("SKIP: TensorFlow not installed (missing input)", text)

    def test_invalid_inputs_error_exit_1(self):
        bad = self.tmp / "bad.json"
        bad.write_text("{}", encoding="utf-8")
        for over, needle in (({"config": str(bad)}, "ERROR: configuration bad.json refused"),
                             ({"weights": str(self.tmp / "absent.h5")}, "ERROR: C1 weights"),
                             ({"manifests": str(self.tmp)}, "ERROR: manifest")):
            with self.subTest(over=over):
                code, text = self.call(**over)
                self.assertEqual(code, 1, text)
                self.assertIn(needle, text)
                self.assertNotIn("Traceback", text)


class MutationHarnessTests(unittest.TestCase):
    """classify di check_n1_mutations.py: una mutazione è rilevata solo da test che girano e falliscono."""

    def test_classify(self):
        import check_n1_mutations as M
        ok = "....\ntests run: 86, failures: 0, errors: 0, skipped: 0\nRESULT: PASS\n"
        self.assertEqual(M.classify(0, ok, 86)[0], "pass")
        self.assertEqual(M.classify(1, "F...\ntests run: 86, failures: 1, errors: 0, skipped: 0\nRESULT: FAIL\n", 86)[0], "fail")
        self.assertEqual(M.classify(1, "tests run: 86, failures: 0, errors: 2, skipped: 0\n", 86)[0], "fail")
        for rc, out in ((1, "Traceback (most recent call last):\nSyntaxError: invalid syntax\n"),
                        (1, "ModuleNotFoundError: No module named 'n1_train'\n"),
                        (0, "tests run: 86, failures: 1, errors: 0, skipped: 0\n"),
                        (1, "tests run: 86, failures: 0, errors: 0, skipped: 0\n"),
                        (1, "tests run: 3, failures: 1, errors: 0, skipped: 0\n"),
                        (1, ok + ok)):
            with self.subTest(out=out[:40], rc=rc):
                self.assertEqual(M.classify(rc, out, 86)[0], "error")


class ZZRepositoryUntouched(unittest.TestCase):
    def test_repository_results_untouched(self):
        self.assertEqual((ROOT / "results" / "n1").exists(), REPO_N1_BEFORE)
        for where in ("scripts", "results", "results/m2"):
            self.assertFalse((ROOT / where / STUDY).exists())


def main() -> int:
    if CV2_NOTE:
        print(CV2_NOTE)
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    res = unittest.TextTestRunner(verbosity=1, stream=sys.stdout).run(suite)
    shutil.rmtree(TMP_ROOT, ignore_errors=True)
    print(f"tests run: {res.testsRun}, failures: {len(res.failures)}, errors: {len(res.errors)}, skipped: {len(res.skipped)}")
    for t, why in res.skipped:
        print(f"SKIP {t.id()}: {why}")
    ok = res.wasSuccessful()
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
