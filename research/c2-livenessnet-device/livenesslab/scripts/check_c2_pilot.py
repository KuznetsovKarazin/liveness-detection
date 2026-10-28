"""
Test di scripts/c2_pilot.py e degli adattamenti di scripts/train_cnn.py su dati SINTETICI, in cartelle temporanee:
nessuna rete, nessuna immagine, nessun dato reale, nessun training né inferenza veri.

Le fixture sono manifest CASIA-FASD e NUAA finti con lo stesso schema di quelli M2 (colonne, chiavi, campi derivati,
forma canonica) e cache .npz finte con gli stessi membri (X_* uint8 N x H x W x 3, y_* int64, s_* stringhe), generate a
run time: nessun valore reale. Il training passa dal vero train_cnn.train con un modulo tensorflow FINTO (nessun
TensorFlow serve né viene importato); i punteggi da uno scorer finto deterministico iniettato (c2_pilot.MAKE_SCORER).

Coperti: check senza rilevatore né pesi (sottoprocesso con import bloccati e registro delle aperture di file), split
sovrapposti per soggetto e per SHA-256, manifest e cache alterati (anche con SHA-256 ridichiarati, così scatta la
regola e non il checksum), separazione fra dataset per file e per frame, collegamento cache ↔ manifest, configurazione
(checkpoint pooled, criterio, insiemi), cartelle del run protette e nuove o -trial, adattamenti di train_cnn.py
(comportamento predefinito invariato, --out-dir, --no-test-eval, controllo dello split), controllo dei pesi
(checkpoint e schede alterati, pesi esistenti o pooled), punteggi grezzi (threshold e prediction null, precisione di
andata e ritorno, CPU e training=False nello scorer vero simulato), selezione solo sul dev (rifiuto per chiave e per
SHA-256 delle righe del test e di NUAA, senza la regola di sessione di NUAA), casi del selettore M1 (pareggi, punteggi
costanti, classe singola), applicazione (threshold.json alterato, punteggi incompleti, NaN, duplicati), invalidazione
delle uscite e scritture atomiche, dispositivo del training (LivenessNet dentro tf.device("/CPU:0") senza nascondere la
GPU, AttackNet sul dispositivo predefinito, con e senza GPU simulata, in entrambi gli ordini; GPU nascosta rifiutata;
arresto prima del salvataggio se variabili o ottimizzatore sono altrove o se le uscite in grafo differiscono da quelle in
eager, anche con le variabili sulla CPU; registrazione in scheda, manifest e train.json; schede alterate rifiutate prima
di ogni punteggio), e un controllo finale che il repository vero non sia stato toccato.

  python scripts/check_c2_pilot.py                    # uscita 1 se un test fallisce
  python scripts/check_c2_pilot.py --block tensorflow,keras,h5py,cv2
                                                      # come in un ambiente senza questi pacchetti: un import bloccato
                                                      # solleva ModuleNotFoundError (anche nei sottoprocessi della suite)

Dipendenze: numpy e scikit-learn. Senza OpenCV (cv2) la suite registra un modulo cv2 vuoto, perché
src/tesi_app/evaluation.py lo importa solo per leggere immagini: lo dichiara con una riga NOTE.
"""
import argparse
import ast
import contextlib
import copy
import csv
import hashlib
import importlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "c2_pilot.py"
TRAIN_CNN = ROOT / "scripts" / "train_cnn.py"
# SHA-256 di scripts/select_threshold.py consegnato con M1 e M2 (deve restare invariato)
SELECT_THRESHOLD_SHA256 = "cacb709672c9c7be5cd68d61923e79bb640e1e8ee77a78864f6786577d07e93c"
RUN = "20261020-C2-seed42-abcdef0"
TRIAL = RUN + "-trial"

# cartelle dei dati, dei risultati e dei pesi in una cartella temporanea PRIMA di importare il codice: i percorsi
# predefiniti non puntano mai al repository vero
TMP_ROOT = Path(tempfile.mkdtemp(prefix="c2-pilot-tests-"))
for _k, _s in (("LIVENESSLAB_DATA_DIR", "data"), ("LIVENESSLAB_RESULTS_DIR", "results"), ("LIVENESSLAB_WEIGHTS_DIR", "weights")):
    os.environ[_k] = str(TMP_ROOT / _s)
    (TMP_ROOT / _s).mkdir()
DECOY = TMP_ROOT / "weights" / "LivenessNet__casia_fasd.h5"           # esca: il check non deve mai aprirla
DECOY.write_bytes(b"decoy weights")

# I soli motivi per cui un test di questa suite si salta: ingressi mancanti dichiarati qui (nessuno di piattaforma)
MISSING_INPUT = {
    "approved configuration": "the approved configuration c2_pilot_casia_to_nuaa.json (delivery/c2/pilot/, or "
                              "../../m2-c2-pilot-protocol/ in the package) or its amended version c2_pilot_casia_to_nuaa_v2.json "
                              "(delivery/c2/livenessnet_device/, or ../ in the package) is not next to the code: not compared",
    "git": "git not available or the code is not in a git work tree: the repository-state sentinel is not compared",
}


def skip_missing(test: unittest.TestCase, what: str, detail: str = ""):
    """Unico modo di saltare un test di questa suite: un ingresso mancante dichiarato in MISSING_INPUT."""
    test.skipTest(f"missing input: {MISSING_INPUT[what]}" + (f" ({detail})" if detail else ""))


# ============================================================================ import bloccati (ambiente senza pacchetti)

BLOCKED = []
IMPORT_LOG = []


class Blocker:
    """Finder che rende assenti i pacchetti elencati (e registra ogni tentativo di importarli)."""

    def find_spec(self, name, path=None, target=None):
        if any(name == b or name.startswith(b + ".") for b in BLOCKED):
            IMPORT_LOG.append(name)
            raise ModuleNotFoundError(f"No module named {name!r} (blocked by the test)", name=name)
        return None


# avvio dei sottoprocessi: stesso blocco degli import, registro degli import tentati e dei file di pesi aperti
BOOT = r'''
import os, runpy, sys
log = os.environ["C2_TEST_LOG"]
names = [n for n in os.environ.get("C2_TEST_BLOCK", "").split(",") if n]
class _B:
    def find_spec(self, name, path=None, target=None):
        if any(name == b or name.startswith(b + ".") for b in names):
            open(log, "a").write("IMPORT " + name + "\n")
            raise ModuleNotFoundError("No module named %r (blocked by the test)" % name, name=name)
        return None
sys.meta_path.insert(0, _B())
EXT = (".h5", ".keras", ".caffemodel", ".onnx", ".pth", ".pb", ".prototxt", ".tflite")
weights = os.environ.get("LIVENESSLAB_WEIGHTS_DIR", "\0")
def _hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0])
        if p.endswith(EXT) or p.startswith(weights):
            with open(log, "a") as f:
                f.write("OPEN " + os.path.basename(p) + "\n")
sys.addaudithook(_hook)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
'''

# cv2 vuoto per evaluation.py quando OpenCV manca (solo le funzioni sulle immagini lo usano)
CV2_STUB = types.ModuleType("cv2")


def _cv2_getattr(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return 0


CV2_STUB.__dict__["__getattr__"] = _cv2_getattr
CV2_NOTE = ""


def ensure_cv2():
    global CV2_NOTE
    try:
        import cv2  # noqa: F401
    except ImportError:
        sys.modules["cv2"] = CV2_STUB
        CV2_NOTE = "NOTE: OpenCV (cv2) not available: tesi_app.evaluation imported with an empty cv2 stand-in (its image functions are not used)"


C2 = None      # c2_pilot, importato in main() dopo il blocco degli import
TC = None      # train_cnn, importato con i sostituti di cv2 e del rilevatore se mancano
rm = nm = st = None


@contextlib.contextmanager
def modules(stubs: dict):
    """Moduli sostitutivi in sys.modules solo per le chiavi date, ripristinate all'uscita (mock.patch.dict toglierebbe
    anche i moduli veri importati nel frattempo, per esempio numpy o scikit-learn, che verrebbero poi importati due volte)."""
    old = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def load(name: str, path: Path, stubs: dict = None):
    with modules(stubs or {}):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


def load_train_cnn():
    stubs = {}
    try:
        importlib.import_module("tesi_app.face")
    except Exception:  # noqa: BLE001  (rilevatore non importabile senza OpenCV: un sostituto, mai usato)
        face = types.ModuleType("tesi_app.face")
        face.get_detector = lambda: None
        stubs["tesi_app.face"] = face
    return load("train_cnn_under_test", TRAIN_CNN, stubs)


# ============================================================================ tensorflow finto

class _Any:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return _Any()

    def __getattr__(self, n):
        if n.startswith("__"):
            raise AttributeError(n)
        return _Any()


class FakeES:
    def __init__(self, monitor=None, patience=None, restore_best_weights=None):
        self.patience, self.stopped_epoch, self.best_weights = patience, 0, None


def plan_early_stop(epochs):
    """val_loss con minimo all'epoca 3 e quattro epoche senza miglioramento: l'early stopping scatta dopo l'epoca 7."""
    return [0.9, 0.8, 0.7, 0.75, 0.76, 0.77, 0.78][:epochs]


def plan_decreasing(epochs):
    return [1.0 - 0.05 * i for i in range(epochs)]


class FakeDevices:
    """Dispositivi del tensorflow finto: GPU visibile o no (Mac con Metal / Linux), GPU presente ma nascosta, pila degli
    ambiti tf.device e guasti simulati: ambito ignorato nel posizionamento delle variabili (ignore_scope), calcolo sulla
    GPU con le variabili sulla CPU (compute_on), solo le funzioni di Keras (predict, evaluate) sulla GPU mentre una
    tf.function nuova resta nell'ambito (keras_on), variabili dell'ottimizzatore altrove (optimizer_on), scarto grafo/eager
    imposto (gap). Senza gap imposto, lo scarto grafo/eager imita tensorflow-metal: LIVENESS_GPU_GAP per LivenessNet
    quando il calcolo è sulla GPU, 2^-18 (circa 3,8e-6, come il 4,6e-6 reale) per le AttackNet sulla GPU, 0 sulla CPU."""
    LIVENESS_GPU_GAP = 2.0

    def __init__(self, gpu=True, hidden_gpu=False, ignore_scope=False, compute_on=None, keras_on=None, keras_paths=("predict", "evaluate"),
                 optimizer_on=None, gap=None):
        self.gpu, self.hidden_gpu, self.ignore_scope, self.compute_on, self.keras_on = gpu, hidden_gpu, ignore_scope, compute_on, keras_on
        self.keras_paths = keras_paths
        self.optimizer_on, self.gap, self.stack, self.in_graph = optimizer_on, gap, [], False

    def current(self) -> str:
        if self.stack and not self.ignore_scope:
            return self.stack[-1]
        return "GPU:0" if self.gpu else "CPU:0"

    def compute(self) -> str:
        return self.compute_on or self.current()

    def graph_gap(self, arch: str, keras: str = None) -> float:
        """Scarto di una chiamata in grafo: keras = "predict" o "evaluate" per le funzioni costruite da Keras (keras_on vale
        per quelle in keras_paths), dove lo scarto imposto non si applica (resta esatto solo sulle uscite degli strati)."""
        if self.gap is not None and not keras:
            return self.gap
        on = self.keras_on if keras in self.keras_paths and self.keras_on else self.compute()
        if on != "GPU:0":
            return 0.0
        return self.LIVENESS_GPU_GAP if arch == "LivenessNet" else 2.0 ** -18

    @staticmethod
    def full(short: str) -> str:
        return f"/job:localhost/replica:0/task:0/device:{short}"

    def device(self, name):
        devs = self

        class Scope:
            def __enter__(self):
                devs.stack.append(str(name).lstrip("/"))

            def __exit__(self, *a):
                devs.stack.pop()
                return False
        return Scope()

    def function(self, f):
        """tf.function finto: la chiamata avviene «in grafo» (lo scarto simulato si applica)."""
        def run(*a, **k):
            self.in_graph = True
            try:
                return f(*a, **k)
            finally:
                self.in_graph = False
        return run


class FakeInputLayer:
    name = "input"


class FakeSub:
    """Sotto-modello con le uscite di tutti gli strati (tf.keras.Model finto): in eager uscite fisse, in grafo con lo
    scarto simulato sull'ultimo strato."""
    def __init__(self, model):
        self.model = model

    def __call__(self, x, training=None):
        import numpy as np
        m = self.model
        m.note("graph_call" if m.devs.in_graph else "eager_call")
        last = np.zeros((len(x), 2))
        last[0, 0] = 1.0                             # IndexError con un lotto vuoto, come np.max su uscite vuote
        FakeSub.calls = getattr(FakeSub, "calls", 0) + 1
        out = [np.full((len(x), 3), 0.5) + (0.01 * FakeSub.calls if training else 0.0), last]    # training=True: rumore del Dropout, diverso a ogni chiamata
        if m.devs.in_graph:
            # scarto sull'ultimo strato (come bn_dense, non il primo), esatto: |g - e| / max|e| = gap
            out[1] = out[1] + np.where(out[1] == 0.0, m.devs.graph_gap(m.arch), 0.0)
        return out


class FakeModel:
    LOG = []
    DEVLOG = []          # (evento, architettura, ambito corrente: "CPU:0" o "" fuori da ogni tf.device)

    def __init__(self, arch, plan, devs=None):
        self.arch, self.plan, self.w, self.devs = arch, plan, b"init", devs or FakeDevices()
        self.variables = [types.SimpleNamespace(device=FakeDevices.full(self.devs.current())) for _ in range(2)]
        self.optimizer = types.SimpleNamespace(variables=lambda: self._opt_vars)
        self._opt_vars = []
        self.inputs = [self]
        self.layers = [FakeInputLayer(), types.SimpleNamespace(name="dense1", output="dense1"), types.SimpleNamespace(name="output", output="output")]
        self.note("create")

    def note(self, event):
        FakeModel.DEVLOG.append((event, self.arch, self.devs.stack[-1] if self.devs.stack else ""))

    def __call__(self, x, training=None):
        import numpy as np
        self.note("eager_call")
        return np.tile([0.25, 0.75], (len(x), 1)) + (0.1 if training else 0.0)

    def compute_loss(self, x=None, y=None, y_pred=None):
        self.note("compute_loss")
        return 0.5

    def evaluate(self, X, Y, batch_size=None, verbose=None, return_dict=None):
        self.note("evaluate")
        return {"loss": 0.5 + 0.5 * self.devs.graph_gap(self.arch, keras="evaluate"), "accuracy": 0.5}

    def compile(self, **k):
        self.note("compile")

    def fit(self, ds, validation_data=None, epochs=None, class_weight=None, callbacks=None, verbose=None):
        self.note("fit")
        self._opt_vars = [types.SimpleNamespace(device=FakeDevices.full(self.devs.optimizer_on or self.devs.current()))]
        losses = self.plan(epochs)
        es = callbacks[0]
        n = len(losses)
        es.best_weights = b"best"
        if n < epochs:
            es.stopped_epoch = n - 1
            self.w = b"best"                       # Keras ripristina i pesi migliori quando l'early stopping scatta
        else:
            self.w = b"last"
        FakeModel.LOG.append(("fit", self.arch, epochs))
        return types.SimpleNamespace(history={"loss": [x + 0.1 for x in losses], "val_loss": list(losses), "val_accuracy": [0.5] * n})

    def set_weights(self, w):
        self.note("set_weights")
        self.w = w

    def predict(self, X, batch_size=None, verbose=None):
        import numpy as np
        self.note("predict")
        FakeModel.LOG.append(("predict", self.arch, len(X)))
        return np.tile([0.25, 0.75], (len(X), 1)) + np.array([0.0, 0.75 * self.devs.graph_gap(self.arch, keras="predict")])

    def save(self, path):
        self.note("save")
        Path(path).write_bytes(b"keras:" + self.arch.encode() + self.w)

    def save_weights(self, path, save_format=None):
        self.note("save_weights")
        Path(path).write_bytes(b"h5:" + self.arch.encode() + self.w)


def fake_modules(plan=plan_early_stop, devs=None):
    devs = devs or FakeDevices()
    tf = types.ModuleType("tensorflow")
    tf.random, tf.data = _Any(), types.SimpleNamespace(Dataset=_Any())
    types_of = lambda gpu: [types.SimpleNamespace(device_type=t) for t in (("CPU", "GPU") if gpu else ("CPU",))]   # noqa: E731
    tf.config = types.SimpleNamespace(
        optimizer=_Any(), get_visible_devices=lambda: types_of(devs.gpu),
        list_physical_devices=lambda kind=None: [d for d in types_of(devs.gpu or devs.hidden_gpu) if kind in (None, d.device_type)],
        list_logical_devices=lambda kind=None: [types.SimpleNamespace(name="/device:GPU:0")] if devs.gpu and kind == "GPU" else [],
        set_visible_devices=lambda *a: FakeModel.DEVLOG.append(("set_visible_devices", "", "")))
    tf.device, tf.function, tf.constant = devs.device, devs.function, lambda x: x
    layers = _Any()
    layers.InputLayer = FakeInputLayer
    tf.keras = types.SimpleNamespace(optimizers=_Any(), losses=_Any(), metrics=_Any(), Sequential=_Any, layers=layers,
                                     Model=lambda inputs, outputs: FakeSub(inputs[0]),
                                     callbacks=types.SimpleNamespace(EarlyStopping=FakeES, ReduceLROnPlateau=_Any))
    arch = types.ModuleType("kuznetsov_architectures")
    arch.create_model = lambda name: types.SimpleNamespace(get_model=lambda: FakeModel(name, plan, devs))
    return {"tensorflow": tf, "kuznetsov_architectures": arch}


def fake_trainer(plan=plan_early_stop, devs=None):
    """train_cnn.train vero con tensorflow e architetture finti (nessun training); devs: dispositivi simulati (default
    una GPU visibile, come su un Mac con tensorflow-metal)."""
    def run(*a, **k):
        with modules(fake_modules(plan, devs)):
            return TC.train(*a, **k)
    return run


def frame_id(X):
    import numpy as np
    X = np.asarray(X)
    return X[:, 0, 0, 0].astype(int) * 65536 + X[:, 0, 0, 1].astype(int) * 256 + X[:, 0, 0, 2].astype(int)


def default_scorer(arch, path, sha):
    salt = 7 if arch == "LivenessNet" else 13
    return lambda X: ((frame_id(X) * salt) % 101) / 100.0


# ============================================================================ fixture sintetiche

BF_VIDEOS, ATT_VIDEOS, FRAMES = ("1", "HR_1"), ("3", "5", "7", "HR_4"), (0, 1)
NUAA_SUBJECTS = tuple(f"{i:04d}" for i in range(1, 5))
FILES = {"source_train": "casia_cnn_train.csv", "source_dev": "casia_cnn_early_stopping.csv", "source_test": "casia_cnn_test.csv",
         "target_test": "nuaa_cnn_test.csv"}
NAMES = {"source_train": "casia_fasd/cnn_train", "source_dev": "casia_fasd/cnn_es", "source_test": "casia_fasd/cnn_test",
         "target_test": "nuaa/cnn_test"}


def H(*parts) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()


def casia_rows(split: str, subjects) -> list:
    rows = []
    for s in subjects:
        for v in BF_VIDEOS + ATT_VIDEOS:
            for f in FRAMES:
                name = f"{s}_{v}.avi_{f}_{'real' if v in BF_VIDEOS else 'fake'}.jpg"
                rows.append(rm.casia_row(split, name, H("casia", split, name), 1000 + len(rows)))
    return rows


def nuaa_rows() -> list:
    rows = []
    for s in NUAA_SUBJECTS:
        for lab in (0, 1):
            for pic in range(3):
                name = f"{s}_{s}_00_01_03_{pic}.jpg"
                key = nm.make_key(lab, name)
                rows.append({"key": key, "label": str(lab), **nm.parse_name(name), "official_split": "test", "sha256": H("nuaa", key),
                             "bytes": str(2000 + len(rows))})
    return rows


def split_by_subject(train_rows):
    """Split di train_cnn.py (GroupShuffleSplit, seed 42) sull'ordine di campionamento: (fitting, early stopping)."""
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit
    keys = rm.spread_keys(train_rows, 1500)
    by = {r["key"]: r for r in train_rows}
    y = np.array([int(by[k]["label"]) for k in keys])
    g = np.array([by[k]["subject"] for k in keys])
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(np.zeros(len(y)), y, g))
    return [by[keys[i]] for i in tr], [by[keys[i]] for i in va]


def frames(code: int, n: int):
    import numpy as np
    X = np.zeros((n, 4, 4, 3), np.uint8)
    for i in range(n):
        X[i, 0, 0] = (code, i // 256, i % 256)
        X[i, 1:] = (i * 37 + code) % 251
    return X


class World:
    """Fixture completa: manifest, cache e configurazione in una cartella temporanea."""

    def __init__(self, base: Path, run_name: str = TRIAL):
        import numpy as np
        self.base = Path(base)
        self.src_dir, self.tgt_dir, self.cache_dir = self.base / "src_manifests", self.base / "tgt_manifests", self.base / "cache"
        for d in (self.src_dir, self.tgt_dir, self.cache_dir):
            d.mkdir(parents=True)
        self.run = self.base / run_name
        train_all = casia_rows("train", range(1, 11))
        fit, dev = split_by_subject(train_all)
        self.rows = {"source_train": fit, "source_dev": dev, "source_test": casia_rows("test", range(1, 7)), "target_test": nuaa_rows()}
        order = rm.spread_keys(fit + dev, 1500)
        by = {r["key"]: r for r in fit + dev}
        t_order = rm.spread_keys(self.rows["source_test"], 1500)
        tb = self.rows_by("source_test")
        n_order = nm.spread_keys(self.rows["target_test"], 1500)
        nb = self.rows_by("target_test")
        self.arrays = {
            "source": {"X_train": frames(1, len(order)), "y_train": np.array([int(by[k]["label"]) for k in order], np.int64),
                       "s_train": np.array([by[k]["subject"] for k in order]), "X_test": frames(2, len(t_order)),
                       "y_test": np.array([int(tb[k]["label"]) for k in t_order], np.int64), "s_test": np.array([tb[k]["subject"] for k in t_order])},
            "target": {"X_train": frames(4, 6), "y_train": np.array([0, 0, 0, 1, 1, 1], np.int64), "s_train": np.array(["0001"] * 6),
                       "X_test": frames(3, len(n_order)), "y_test": np.array([int(nb[k]["label"]) for k in n_order], np.int64),
                       "s_test": np.array([nb[k]["subject"] for k in n_order])}}
        self.cfg = self.base_config()
        self.write_all()

    def rows_by(self, role):
        return {r["key"]: r for r in self.rows[role]}

    @property
    def config(self) -> Path:
        return self.base / "config.json"

    def cache(self, side) -> Path:
        return self.cache_dir / ("casia_fasd_1500.npz" if side == "source" else "nuaa_1500.npz")

    def base_config(self) -> dict:
        return {
            "schema": "c2-pilot-config/draft-1", "status": "synthetic test fixture", "seed": 42,
            "direction": {"source": "casia_fasd", "target": "nuaa"},
            "models": [{"id": "livenessnet__casia_fasd_c2", "arch": "LivenessNet"}, {"id": "attacknet_v2_2__casia_fasd_c2", "arch": "AttackNetV2_2"}],
            "checkpoints": {"proposed": "new training of both models, seed 42",
                            "alternative_not_recommended": {"LivenessNet__casia_fasd.h5": H("w", 1), "AttackNetV2_2__casia_fasd.h5": H("w", 2)},
                            "excluded_pooled": {"LivenessNet__casia_fasd-pooled.h5": H("w", 3), "AttackNetV2_2__casia_fasd-pooled.h5": H("w", 4),
                                                "reason": "synthetic"}},
            "budget": {"max_epochs": 12, "batch": 16, "learning_rate": 0.001, "wall_time_cap_minutes": 120},
            "manifests": {}, "derived_frames": {"source": {"file": "casia_fasd_1500.npz"}, "target": {"file": "nuaa_1500.npz"}},
            "threshold": {"set": "source_dev only", "criterion": "eer"},
            "evaluations": [{"id": "E1", "model": "livenessnet__casia_fasd_c2", "set": "source_test"},
                            {"id": "E2", "model": "attacknet_v2_2__casia_fasd_c2", "set": "source_test"},
                            {"id": "E3", "model": "livenessnet__casia_fasd_c2", "set": "target_test"},
                            {"id": "E4", "model": "attacknet_v2_2__casia_fasd_c2", "set": "target_test"}],
            "scoring": {"device": "CPU"},
            "training": {"device": {"LivenessNet": "CPU", "AttackNetV2_2": "default", "note": "synthetic"}}}

    def write_manifest(self, role, rows=None, redeclare=True, raw_rows=False):
        if rows is not None:
            self.rows[role] = rows
        ds = "casia_fasd" if role != "target_test" else "nuaa"
        p = (self.src_dir if ds == "casia_fasd" else self.tgt_dir) / FILES[role]
        if raw_rows:
            (rm.write_rows(ds, p, self.rows[role]) if ds == "casia_fasd" else nm.write_rows(p, self.rows[role]))
        elif ds == "casia_fasd":
            rm.write_manifest(ds, p, self.rows[role])
        else:
            nm.write_manifest(p, self.rows[role])
        if redeclare:
            c = C2.counts_of(ds, self.rows[role])
            self.cfg["manifests"][role] = {"name": NAMES[role], "file": FILES[role], "sha256": C2.sha256_file(p),
                                           **{k: c[k] for k in ("rows", "n_bona_fide", "n_attack", "n_subjects")},
                                           **({"n_videos": c["n_videos"]} if ds == "casia_fasd" else {})}
        return p

    def write_cache(self, side, redeclare=True):
        import numpy as np
        np.savez_compressed(self.cache(side), **self.arrays[side])
        if redeclare:
            e = {"file": self.cache(side).name, "sha256": C2.sha256_file(self.cache(side)), "X_test": len(self.arrays[side]["X_test"])}
            if side == "source":
                e["X_train"] = len(self.arrays[side]["X_train"])
            self.cfg["derived_frames"][side] = e

    def write_config(self):
        self.config.write_text(json.dumps(self.cfg, indent=1) + "\n", encoding="utf-8")

    def write_all(self):
        for role in FILES:
            self.write_manifest(role)
        for side in ("source", "target"):
            self.write_cache(side)
        self.write_config()

    def args(self, cmd, *extra, run=None):
        a = [cmd, "--config", str(self.config), "--run-dir", str(run or self.run), "--source-manifests", str(self.src_dir),
             "--target-manifests", str(self.tgt_dir), "--source-cache", str(self.cache("source")), "--target-cache", str(self.cache("target"))]
        if cmd in ("train", "all"):
            a += ["--epochs", "12"]
        if cmd in ("train", "score", "all"):
            a += ["--i-am-authorized"]
        return a + list(extra)


def run_c2(argv) -> tuple:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = C2.main(list(argv))
    return rc, out.getvalue(), err.getvalue()


def pipeline(w: World, upto: str = "apply", run=None) -> None:
    steps = [("check",), ("train",), ("score", "--set", "dev"), ("select",), ("score", "--set", "source_test,target"), ("apply",)]
    names = ["check", "train", "score_dev", "select", "score_tests", "apply"]
    for name, st_args in zip(names, steps):
        rc, out, err = run_c2(w.args(st_args[0], *st_args[1:], run=run))
        if rc != 0:
            raise AssertionError(f"{name} failed: {err}{out}")
        if name == upto:
            return


def read_rows(p: Path) -> list:
    return C2.read_scores(p)


def rewrite_scores(w: World, name: str, fn, record: str = None, run=None) -> None:
    """Riscrive un file di punteggi con fn(righe) e, con record, ridichiara il suo SHA-256 nel record (fixture avversaria:
    deve scattare la regola, non il checksum)."""
    run = Path(run or w.run)
    rows = fn(read_rows(run / name))
    text = C2.csv_text(rows) if rows and isinstance(rows[0], dict) else rows
    (run / name).write_text(text, encoding="utf-8", newline="")
    if record:
        d = json.loads((run / record).read_text())
        d["files"][name]["sha256"] = C2.sha256_file(run / name)
        (run / record).write_text(json.dumps(d, indent=1) + "\n")


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="c2t-", dir=TMP_ROOT)
        self.tmp = Path(self._tmp.name)
        self._p = [mock.patch.object(C2, "TRAINER", fake_trainer()), mock.patch.object(C2, "MAKE_SCORER", default_scorer),
                   mock.patch.object(C2, "REPO_STATE", lambda run: {"commit": "abcdef0" + "0" * 33, "dirty": False, "dirty_excludes": None})]
        for p in self._p:
            p.start()

    def tearDown(self):
        for p in self._p:
            p.stop()
        self._tmp.cleanup()

    def world(self, run_name=TRIAL) -> World:
        return World(self.tmp / "w", run_name)

    def assertFail(self, res, text: str = None, step: str = None):
        rc, out, err = res
        self.assertEqual(rc, 1, out + err)
        self.assertIn("RESULT: FAIL", out)
        self.assertIn("ERROR: ", err)
        self.assertNotIn("Traceback", out + err)
        if text:
            self.assertIn(text, err)
        if step:
            self.assertIn(f"RESULT: FAIL ({step})", out)

    def assertPass(self, res):
        rc, out, err = res
        self.assertEqual(rc, 0, out + err)
        self.assertIn("RESULT: PASS", out)

    def assertNoOutputs(self, run: Path, *steps):
        for s in steps:
            for n in C2.OUTPUTS[s]:
                self.assertFalse((run / n).exists(), n)
                self.assertFalse((run / (n + ".tmp")).exists(), n)
        if "train" in steps:
            self.assertFalse((run / "checkpoints").exists())
            self.assertFalse((run / "checkpoints.tmp").exists())
            self.assertFalse([e for e in os.listdir(run) if e.startswith(".checkpoints.")] if run.is_dir() else [])
        if run.is_dir():                    # temporanei nuovi (nomi casuali) delle uscite dei passi controllati
            names = {n for s_ in steps for n in C2.OUTPUTS[s_]}
            self.assertFalse([e for e in os.listdir(run) if e.endswith(".tmp") and any(e.startswith(f".{n}.") for n in names)])


# ============================================================================ configurazione

class TestConfig(Base):
    def test_fixture_config_passes(self):
        w = self.world()
        self.assertEqual(C2.config_problems(w.cfg), [])

    def test_approved_config(self):
        """Configurazione emendata (PR #12) accanto al codice, con lo SHA-256 del runner; uguale come valori JSON
        all'originale approvata (PR #7, invariata) salvo training.device, che dà la politica di TRAINING_DEVICE e di
        DEVICE_POLICY di train_cnn.py; l'originale (training.device "Metal GPU") è rifiutata."""
        orig = next((c for c in (ROOT / "delivery" / "c2" / "pilot" / "c2_pilot_casia_to_nuaa.json",
                                 ROOT.parent.parent / "m2-c2-pilot-protocol" / "c2_pilot_casia_to_nuaa.json") if c.is_file()), None)
        amended = next((c for c in (ROOT / "delivery" / "c2" / "livenessnet_device" / "c2_pilot_casia_to_nuaa_v2.json",
                                    ROOT.parent / "c2_pilot_casia_to_nuaa_v2.json") if c.is_file()), None)
        if orig is None or amended is None:
            skip_missing(self, "approved configuration", "original" if orig is None else "amended")
        cfg, sha = C2.load_config(amended)
        self.assertEqual(sha, C2.APPROVED_CONFIG_SHA256)
        self.assertEqual([m["arch"] for m in cfg["models"]], ["LivenessNet", "AttackNetV2_2"])
        self.assertEqual({a: cfg["training"]["device"][a] for a in C2.ARCHS}, C2.TRAINING_DEVICE)
        self.assertEqual(C2.TRAINING_DEVICE, {a: TC.DEVICE_POLICY[a] for a in C2.ARCHS})
        raw = orig.read_bytes()
        base = json.loads(raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), "4065d8535d86b3743db3b58c399123afc57b3fa482c3964449f2d29f36c84f7a")
        self.assertEqual(cfg["amends"], {**cfg["amends"], "sha256": hashlib.sha256(raw).hexdigest(), "changed": ["training.device"],
                                         "file": "research/m2-c2-pilot-protocol/c2_pilot_casia_to_nuaa.json", "approved_in": "PR #7"})
        self.assertEqual(sorted(cfg["amends"]), ["approved_in", "changed", "file", "reason", "sha256", "unchanged"])
        self.assertTrue(str(base["training"]["device"]).startswith("Metal GPU"))
        rest = copy.deepcopy(cfg)
        del rest["amends"]
        rest["training"]["device"] = base["training"]["device"]
        self.assertEqual(rest, base)                                    # split, seed, modelli, budget, soglia, ... identici
        self.assertEqual(list(rest), list(base))
        self.assertTrue(any("training.device" in b for b in C2.config_problems(base)))
        with self.assertRaises(C2.C2Error) as cm:
            C2.load_config(orig)
        self.assertIn("refused: training.device must give exactly the device policy", str(cm.exception))

    def test_refusals(self):
        w = self.world()
        cases = {
            "schema": lambda c: c.update(schema="other"),
            "seed": lambda c: c.update(seed=7),
            "direction": lambda c: c.update(direction={"source": "nuaa", "target": "casia_fasd"}),
            "checkpoint in a model": lambda c: c["models"][0].update(checkpoint="LivenessNet__casia_fasd-pooled.h5"),
            "pooled model id": lambda c: c["models"][0].update(id="livenessnet__casia_fasd-pooled"),
            "pooled manifest name": lambda c: c["manifests"]["source_train"].update(name="casia_fasd/pooled_train"),
            "pooled manifest file": lambda c: c["manifests"]["source_dev"].update(file="casia_pooled_early_stopping.csv"),
            "mirror test as dev": lambda c: c["manifests"]["source_dev"].update(name="casia_fasd/mirror_test"),
            "existing checkpoints used": lambda c: c["checkpoints"].update(proposed="existing official weights"),
            "extra checkpoint field": lambda c: c["checkpoints"].update(use={"LivenessNet": "x"}),
            "pooled list missing": lambda c: c["checkpoints"].pop("excluded_pooled"),
            "criterion": lambda c: c["threshold"].update(criterion="apcer10"),
            "threshold set": lambda c: c["threshold"].update(set="target_test"),
            "evaluations": lambda c: c["evaluations"].reverse(),
            "device": lambda c: c["scoring"].update(device="GPU"),
            "rows": lambda c: c["manifests"]["source_test"].update(rows=c["manifests"]["source_test"]["rows"] + 1),
            "budget": lambda c: c["budget"].update(learning_rate=1),
            "architectures": lambda c: c["models"][1].update(arch="AttackNetV1"),
            "training device missing": lambda c: c.pop("training"),
            "training device as text": lambda c: c["training"].update(device="Metal GPU (not deterministic; evidence = saved weights SHA-256)"),
            "LivenessNet on the default device": lambda c: c["training"]["device"].update(LivenessNet="default"),
            "AttackNet on the CPU": lambda c: c["training"]["device"].update(AttackNetV2_2="CPU"),
            "architecture without a device": lambda c: c["training"]["device"].pop("AttackNetV2_2"),
            "extra training device field": lambda c: c["training"]["device"].update(AttackNetV1="default"),
            "training device note missing": lambda c: c["training"]["device"].pop("note"),
            "training device note empty": lambda c: c["training"]["device"].update(note=" "),
        }
        for name, fn in cases.items():
            c = copy.deepcopy(w.cfg)
            fn(c)
            self.assertTrue(C2.config_problems(c), name)

    def test_original_training_device_refused(self):
        """Il training.device della configurazione originale ("Metal GPU") è rifiutato dal check, senza uscite."""
        w = self.world()
        w.cfg["training"]["device"] = "Metal GPU (not deterministic; evidence = saved weights SHA-256)"
        w.write_config()
        self.assertFail(run_c2(w.args("check")), "training.device must give exactly the device policy", "check")
        self.assertNoOutputs(w.run, "check")

    def test_config_changed_after_check_refused(self):
        w = self.world()
        self.assertPass(run_c2(w.args("check")))
        w.cfg["status"] = "edited"
        w.write_config()
        self.assertFail(run_c2(w.args("train")), "differs from the one recorded at check", "train")

    def test_invalid_json_refused(self):
        w = self.world()
        w.config.write_text("{not json")
        self.assertFail(run_c2(w.args("check")), "not valid JSON")
        self.assertNoOutputs(w.run, "check")


# ============================================================================ cartella del run

class TestRunFolder(Base):
    def fake_root(self):
        r = self.tmp / "repo"
        for d in ("data/processed", "models/weights", "results/c1", "results/m1", "results/m2", "results/m2_rgb", "results/eval",
                  "export/drive/old-run", "export/research", "docs", "scripts", "src", "delivery", "config", ".git"):
            (r / d).mkdir(parents=True)
        return r

    def test_names(self):
        for bad in ("run", "20261020-C2-seed43-abcdef0", "20261020-M2-seed42-abcdef0", "20261020-C2-seed42-ABCDEF0", "20261020-C2-seed42-abc"):
            self.assertIn("is not YYYYMMDD", C2.run_dir_problem(self.tmp / bad), bad)
        self.assertEqual(C2.run_dir_problem(self.tmp / RUN), "")
        self.assertEqual(C2.run_dir_problem(self.tmp / TRIAL), "")

    def test_protected_places(self):
        r = self.fake_root()
        with mock.patch.object(C2, "ROOT", r):
            for where in ("models/weights", "models", "results/c1", "results/m1", "results/m2", "results/m2_rgb", "export/drive",
                          "export/drive/old-run", "export/research", "export", "data/processed", "data", "docs", "scripts", "src",
                          "delivery", "config", ".git", ".", "results"):
                self.assertTrue(C2.run_dir_problem(r / where / TRIAL), where)
            self.assertEqual(C2.run_dir_problem(r / "results" / "c2" / TRIAL), "")
            self.assertEqual(C2.run_dir_problem(self.tmp / "elsewhere" / TRIAL), "")
            # cartelle spostate con le variabili d'ambiente: protette anche fuori dal repository
            moved = self.tmp / "moved_weights"
            moved.mkdir()
            with mock.patch.object(C2, "WEIGHTS", moved):
                self.assertIn("protected", C2.run_dir_problem(moved / TRIAL))
            res = self.tmp / "moved_results"
            (res / "m2").mkdir(parents=True)
            (res / "m2_rgb").mkdir(parents=True)
            with mock.patch.object(C2, "RESULTS", res):
                self.assertTrue(C2.run_dir_problem(res / "m2" / TRIAL))
                self.assertIn("protected", C2.run_dir_problem(res / "m2_rgb" / TRIAL))
            # una cartella del run che contiene una cartella protetta (pesi spostati dentro di essa)
            holder = self.tmp / TRIAL
            (holder / "w").mkdir(parents=True)
            with mock.patch.object(C2, "WEIGHTS", holder / "w"):
                self.assertIn("protected", C2.run_dir_problem(holder))

    def test_protected_run_dir_refused_before_any_write(self):
        r = self.fake_root()
        w = self.world()
        target = r / "results" / "m1" / TRIAL
        with mock.patch.object(C2, "ROOT", r):
            self.assertFail(run_c2(w.args("check", run=target)), "run folder refused", "check")
        self.assertFalse(target.exists())

    def test_alias_of_protected_folder(self):
        # un secondo percorso verso la stessa cartella (link simbolico) è riconosciuto per identità dei file
        r = self.fake_root()
        alias = self.tmp / "alias_to_weights"
        alias.symlink_to(r / "models" / "weights", target_is_directory=True)
        with mock.patch.object(C2, "ROOT", r):
            self.assertTrue(C2.run_dir_problem(alias / TRIAL))

    def test_non_trial_must_be_new(self):
        w = self.world(RUN)
        w.run.mkdir()
        (w.run / "keep.txt").write_text("x")
        with mock.patch.object(C2, "APPROVED_CONFIG_SHA256", C2.sha256_file(w.config)):
            self.assertFail(run_c2(w.args("check")), "must be new", "check")
        self.assertEqual(os.listdir(w.run), ["keep.txt"])          # nemmeno status.json

    def test_non_trial_needs_approved_config_clean_tree_and_commit(self):
        w = self.world(RUN)
        self.assertFail(run_c2(w.args("check")), "is not the approved (amended) one")
        with mock.patch.object(C2, "APPROVED_CONFIG_SHA256", C2.sha256_file(w.config)):
            with mock.patch.object(C2, "REPO_STATE", lambda run: {"commit": "abcdef0" + "0" * 33, "dirty": True}):
                self.assertFail(run_c2(w.args("check")), "must be clean")
            with mock.patch.object(C2, "REPO_STATE", lambda run: {"commit": "1234567" + "0" * 33, "dirty": False}):
                self.assertFail(run_c2(w.args("check")), "must be clean")
            self.assertPass(run_c2(w.args("check")))
            doc = json.loads((w.run / "check.json").read_text())
            self.assertIs(doc["trial"], False)
            self.assertIs(doc["approved_config"], True)
            with mock.patch.object(C2, "REPO_STATE", lambda run: {"commit": "abcdef1" + "0" * 33, "dirty": False}):
                self.assertFail(run_c2(w.args("train")), "commit or state differs from check", "train")
            self.assertPass(run_c2(w.args("train")))

    def test_trial_folder_reusable(self):
        w = self.world()
        self.assertPass(run_c2(w.args("check")))
        self.assertPass(run_c2(w.args("check")))


# ============================================================================ check

class TestCheck(Base):
    def test_pass_and_record(self):
        w = self.world()
        self.assertPass(run_c2(w.args("check")))
        doc = json.loads((w.run / "check.json").read_text())
        self.assertEqual(doc["config_sha256"], C2.sha256_file(w.config))
        for role, rows in w.rows.items():
            subj = "subject" if role == "target_test" else "subject_key"
            self.assertEqual((doc["manifests"][role]["n_subjects_bona_fide"], doc["manifests"][role]["n_subjects_attack"]),
                             (len({r[subj] for r in rows if r["label"] == "0"}), len({r[subj] for r in rows if r["label"] == "1"})), role)
        self.assertEqual(sorted(doc["manifests"]), sorted(FILES))
        for role, e in doc["manifests"].items():
            self.assertEqual(e["sha256"], w.cfg["manifests"][role]["sha256"])
        self.assertEqual(doc["detector"], "not used")
        self.assertEqual(doc["weights"], "not loaded")
        self.assertEqual(doc["derived_frames"]["casia_nuaa_shared"], 0)
        self.assertEqual(sorted(os.listdir(w.run)), ["check.json", "status.json"])
        self.assertEqual(json.loads((w.run / "status.json").read_text())["result"], "PASS")
        self.assertIn("src/tesi_app/evaluation.py", doc["code_sha256"])

    def test_no_detector_no_weights_no_tensorflow(self):
        """In un sottoprocesso con tensorflow, keras, h5py, cv2 e il rilevatore bloccati: nessun tentativo di importarli,
        nessun file di pesi aperto (un'esca è nella cartella dei pesi)."""
        w = self.world()
        log = self.tmp / "log.txt"
        log.write_text("")
        env = {**os.environ, "C2_TEST_LOG": str(log), "PYTHONDONTWRITEBYTECODE": "1",
               "C2_TEST_BLOCK": "tensorflow,keras,h5py,cv2,tesi_app.face,tesi_app.evaluation,tesi_app.core,train_cnn,sklearn,torch"}
        r = subprocess.run([sys.executable, "-I", "-B", "-c", BOOT, str(SCRIPT), *w.args("check")], env=env, capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("RESULT: PASS (check)", r.stdout)
        self.assertEqual(log.read_text(), "", "imports or weight files touched by check")
        self.assertTrue(DECOY.exists())

    def test_tampered_manifest(self):
        w = self.world()
        p = w.src_dir / FILES["source_dev"]
        p.write_bytes(p.read_bytes() + b"\n")
        self.assertFail(run_c2(w.args("check")), "differs from the configuration", "check")
        self.assertNoOutputs(w.run, "check")

    def test_row_inconsistent_with_key_redeclared(self):
        w = self.world()
        rows = copy.deepcopy(w.rows["source_dev"])
        rows[0]["label"] = "1" if rows[0]["label"] == "0" else "0"
        w.write_manifest("source_dev", rows)
        w.write_config()
        self.assertFail(run_c2(w.args("check")), "label values not consistent with the key")

    def test_counts_against_config(self):
        w = self.world()
        w.cfg["manifests"]["source_test"]["n_subjects"] += 1
        w.write_config()
        self.assertFail(run_c2(w.args("check")), "n_subjects")
        w = World(self.tmp / "w2")
        w.cfg["manifests"]["source_train"]["n_videos"] -= 1
        w.write_config()
        self.assertFail(run_c2(w.args("check")), "n_videos")

    def test_overlap_by_subject(self):
        w = self.world()
        dev_subject = w.rows["source_dev"][0]["subject"]
        extra = [r for r in casia_rows("train", [int(dev_subject)]) if r["frame"] == "0"][:1]
        extra[0]["key"] = extra[0]["key"].replace("_0_", "_7_")
        extra = [rm.casia_row("train", extra[0]["key"].split("/", 1)[1], H("moved"), 1)]
        w.write_manifest("source_train", w.rows["source_train"] + extra)
        w.write_config()
        self.assertFail(run_c2(w.args("check")), "share 1 values of subject_key")

    def test_overlap_by_sha(self):
        w = self.world()
        rows = copy.deepcopy(w.rows["source_train"])
        rows[0]["sha256"] = w.rows["source_dev"][0]["sha256"]
        w.write_manifest("source_train", rows)
        w.write_config()
        self.assertFail(run_c2(w.args("check")), "source_train and source_dev share 1 values of sha256")

    def test_train_and_test_share_a_file(self):
        w = self.world()
        rows = copy.deepcopy(w.rows["source_test"])
        rows[0]["sha256"] = w.rows["source_train"][0]["sha256"]
        w.write_manifest("source_test", rows)
        w.write_config()
        self.assertFail(run_c2(w.args("check")), "source_train and source_test share 1 values of sha256")

    def test_test_row_in_training(self):
        w = self.world()
        w.write_manifest("source_train", w.rows["source_train"] + [w.rows["source_test"][0]])
        w.write_config()
        self.assertFail(run_c2(w.args("check")), "source_train: 1 rows outside the mirror split train_img")

    def test_cross_dataset_file(self):
        w = self.world()
        rows = copy.deepcopy(w.rows["target_test"])
        rows[0]["sha256"] = w.rows["source_test"][3]["sha256"]
        w.write_manifest("target_test", rows)
        w.write_config()
        self.assertFail(run_c2(w.args("check")), "cross-dataset overlap: 1 CASIA-FASD files")

    def test_cross_dataset_frame(self):
        w = self.world()
        w.arrays["target"]["X_test"][2] = w.arrays["source"]["X_test"][5]
        w.write_cache("target")
        w.write_config()
        self.assertFail(run_c2(w.args("check")), "1 frames shared between CASIA-FASD and NUAA")

    def test_fit_dev_frame_shared(self):
        w = self.world()
        order = rm.spread_keys(w.rows["source_train"] + w.rows["source_dev"], 1500)
        i_dev = order.index(w.rows["source_dev"][0]["key"])
        i_fit = order.index(w.rows["source_train"][0]["key"])
        w.arrays["source"]["X_train"][i_dev] = w.arrays["source"]["X_train"][i_fit]
        w.write_cache("source")
        w.write_config()
        self.assertFail(run_c2(w.args("check")), "between fitting and early-stopping")

    def test_cache_tampered_and_linkage(self):
        w = self.world()
        p = w.cache("target")
        p.write_bytes(p.read_bytes() + b"x")
        self.assertFail(run_c2(w.args("check")), "differs from the configuration")
        for member, fn in (("y_test", lambda a: a[::-1].copy()), ("s_train", lambda a: a[::-1].copy())):
            w = World(self.tmp / f"w_{member}")
            side = "target" if member == "y_test" else "source"
            w.arrays[side][member] = fn(w.arrays[side][member])
            w.write_cache(side)
            w.write_config()
            self.assertFail(run_c2(w.args("check")), "linkage refused")
        w = World(self.tmp / "w_missing")
        del w.arrays["source"]["s_test"]
        w.write_cache("source")
        w.write_config()
        self.assertFail(run_c2(w.args("check")), "missing members")
        w = World(self.tmp / "w_count")
        w.cfg["derived_frames"]["source"]["X_test"] += 1
        w.write_config()
        self.assertFail(run_c2(w.args("check")), "frames, configuration")

    def test_pooled_checkpoint_in_config(self):
        w = self.world()
        w.cfg["models"][0]["weights"] = "LivenessNet__casia_fasd-pooled.h5"
        w.write_config()
        self.assertFail(run_c2(w.args("check")), "no checkpoint")

    def test_pass_then_failure_leaves_nothing(self):
        w = self.world()
        pipeline(w, "apply")
        p = w.tgt_dir / FILES["target_test"]
        p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
        self.assertFail(run_c2(w.args("check")), step="check")
        self.assertNoOutputs(w.run, *C2.STEPS)
        self.assertEqual(os.listdir(w.run), ["status.json"])
        st_ = json.loads((w.run / "status.json").read_text())
        self.assertEqual((st_["result"], st_["step"]), ("FAIL", "check"))
        self.assertTrue(st_["outputs"].startswith("removed"))

    def test_non_canonical_manifest_redeclared(self):
        w = self.world()
        rows = list(reversed(w.rows["source_dev"]))
        w.write_manifest("source_dev", rows, raw_rows=True)
        w.write_config()
        self.assertFail(run_c2(w.args("check")), "not in canonical form")


# ============================================================================ train_cnn.py: adattamenti

ORIGINAL_CARD_KEYS = ["arch", "dataset", "protocol", "seed", "n_train", "n_val", "n_test", "val_split", "epochs_run", "best_epoch",
                      "train_seconds", "lr", "batch", "best_val_loss", "val_acc_at_best", "test", "weights_sha256", "saved"]
# scheda con la registrazione del dispositivo (27/10/2026): unico campo nuovo, prima del test
CARD_KEYS = ORIGINAL_CARD_KEYS[:ORIGINAL_CARD_KEYS.index("test")] + ["training_device"] + ORIGINAL_CARD_KEYS[ORIGINAL_CARD_KEYS.index("test"):]


class TestTrainCnn(Base):
    def data(self):
        w = self.world()
        a = w.arrays["source"]
        return w, {"X_train": a["X_train"], "y_train": a["y_train"], "s_train": a["s_train"], "X_test": a["X_test"], "y_test": a["y_test"]}

    def split_check(self, w):
        order = rm.spread_keys(w.rows["source_train"] + w.rows["source_dev"], 1500)
        by = {r["key"]: r for r in w.rows["source_train"] + w.rows["source_dev"]}
        return {"sha256": [by[k]["sha256"] for k in order], "train": [r["sha256"] for r in w.rows["source_train"]],
                "val": [r["sha256"] for r in w.rows["source_dev"]]}

    def test_default_behaviour_unchanged(self):
        w, data = self.data()
        weights = self.tmp / "weights_default"
        weights.mkdir()
        FakeModel.LOG.clear()
        with mock.patch.object(TC, "WEIGHTS", weights), contextlib.redirect_stdout(io.StringIO()):
            info = fake_trainer()("LivenessNet", "casia_fasd", data, 12, 16, 1e-3)
        self.assertEqual(sorted(os.listdir(weights)), ["LivenessNet__casia_fasd.h5", "LivenessNet__casia_fasd.json",
                                                       "LivenessNet__casia_fasd.keras", "manifest.json"])
        card_text = (weights / "LivenessNet__casia_fasd.json").read_text()
        card = json.loads(card_text)
        self.assertEqual(list(card), CARD_KEYS)
        self.assertEqual(card_text, json.dumps(info, indent=1))              # stessi byte del formato C1 (rientranza 1)
        man_text = (weights / "manifest.json").read_text()
        self.assertEqual(man_text, json.dumps(json.loads(man_text), indent=1))
        self.assertEqual(list(info), CARD_KEYS)
        self.assertEqual(card["n_test"], len(data["y_test"]))
        man = json.loads((weights / "manifest.json").read_text())
        self.assertEqual(list(man["LivenessNet__casia_fasd"]), [k for k in CARD_KEYS if k != "test"] + ["test_acer", "test_eer", "test_auc"])
        self.assertEqual(man["LivenessNet__casia_fasd"]["training_device"], card["training_device"])
        self.assertIn(("predict", "LivenessNet", len(data["y_test"])), FakeModel.LOG)

    def test_out_dir_no_test_eval_split_check(self):
        w, data = self.data()
        weights = self.tmp / "weights_sentinel"
        weights.mkdir()
        (weights / "manifest.json").write_text("{}")
        out = self.tmp / "out"
        del data["X_test"], data["y_test"]                 # nessun X_test: il test non è nemmeno disponibile
        FakeModel.LOG.clear()
        with mock.patch.object(TC, "WEIGHTS", weights), contextlib.redirect_stdout(io.StringIO()) as o:
            info = fake_trainer()("AttackNetV2_2", "casia_fasd", data, 12, 16, 1e-3, "official", out_dir=out,
                                  split_check=self.split_check(w), test_eval=False)
        self.assertEqual(sorted(os.listdir(weights)), ["manifest.json"])
        self.assertEqual((weights / "manifest.json").read_text(), "{}")
        self.assertEqual(sorted(os.listdir(out)), ["AttackNetV2_2__casia_fasd.h5", "AttackNetV2_2__casia_fasd.json", "AttackNetV2_2__casia_fasd.keras"])
        card = json.loads((out / "AttackNetV2_2__casia_fasd.json").read_text())
        self.assertNotIn("test", card)
        self.assertNotIn("n_test", card)
        self.assertEqual(card["test_evaluation"], "disabled")
        self.assertTrue(card["split_check"].startswith("passed"))
        self.assertEqual((card["restored_epoch"], card["best_epoch"], card["epochs_run"]), (3, 3, 7))
        self.assertEqual(card["restored_by"], "early stopping")
        self.assertEqual(card["n_val"], len(w.rows["source_dev"]))
        # nessuna valutazione del test: l'unico predict è quello del controllo del dispositivo, su 4 frame di validazione
        self.assertEqual([e for e in FakeModel.LOG if e[0] == "predict"], [("predict", "AttackNetV2_2", 4)])
        self.assertIn("TEST not evaluated", o.getvalue())
        self.assertEqual(info["weights_sha256"], C2.sha256_file(out / "AttackNetV2_2__casia_fasd.h5"))

    def test_restored_epoch_at_epoch_limit(self):
        w, data = self.data()
        out = self.tmp / "out"
        with contextlib.redirect_stdout(io.StringIO()):
            info = fake_trainer(plan_decreasing)("LivenessNet", "casia_fasd", data, 12, 16, 1e-3, out_dir=out, test_eval=False)
        self.assertEqual((info["restored_epoch"], info["epochs_run"], info["restored_by"]), (12, 12, "end of epochs (set_weights)"))
        self.assertTrue((out / "LivenessNet__casia_fasd.h5").read_bytes().endswith(b"best"))

    def test_out_dir_inside_weights_refused(self):
        w, data = self.data()
        weights = self.tmp / "weights_x"
        weights.mkdir()
        with mock.patch.object(TC, "WEIGHTS", weights):
            for d in (weights, weights / "sub"):
                with self.assertRaises(SystemExit) as cm:
                    fake_trainer()("LivenessNet", "casia_fasd", data, 12, 16, 1e-3, out_dir=d, test_eval=False)
                self.assertIn("inside the weights folder", str(cm.exception.code))
        self.assertEqual(os.listdir(weights), [])

    def test_split_check_failures(self):
        w, data = self.data()
        sc = self.split_check(w)
        bad = {**sc, "train": sc["train"][1:] + [sc["val"][0]], "val": [sc["train"][0]] + sc["val"][1:]}   # un frame scambiato
        out = self.tmp / "out"
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stdout(io.StringIO()):
            fake_trainer()("LivenessNet", "casia_fasd", data, 12, 16, 1e-3, out_dir=out, split_check=bad, test_eval=False)
        self.assertIn("does not reproduce", str(cm.exception.code))
        self.assertFalse(out.exists())
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stdout(io.StringIO()):
            fake_trainer()("LivenessNet", "casia_fasd", data, 12, 16, 1e-3, "pooled", out_dir=out, split_check=sc, test_eval=False)
        self.assertIn("official protocol only", str(cm.exception.code))
        # lati con lunghezze diverse dai manifest
        short = {**sc, "val": sc["val"][1:]}
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stdout(io.StringIO()):
            fake_trainer()("LivenessNet", "casia_fasd", data, 12, 16, 1e-3, out_dir=out, split_check={**short, "sha256": sc["sha256"]}, test_eval=False)
        self.assertIn("frames, its manifest", str(cm.exception.code))
        # stessi valori distinti, molteplicità diverse: confronto come multinsiemi
        x_, y_, z_ = sorted(set(sc["val"]))[:3]
        sha2 = [x_ if v == z_ else v for v in sc["sha256"]]                # cache: z diventa x -> {x: 2, y: 1}
        want_val = [y_ if v == z_ else v for v in sc["val"]]               # manifest: z diventa y -> {x: 1, y: 2}
        self.assertEqual({v for i, v in enumerate(sha2) if v in (x_, y_)}, set(want_val) & {x_, y_})
        multi = {"sha256": sha2, "train": sc["train"], "val": want_val}
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stdout(io.StringIO()):
            fake_trainer()("LivenessNet", "casia_fasd", data, 12, 16, 1e-3, out_dir=out, split_check=multi, test_eval=False)
        self.assertIn("does not reproduce", str(cm.exception.code))
        few = {**data, "s_train": data["s_train"].copy()}
        few["s_train"][:] = "1"                      # meno di 5 soggetti: split casuale, controllo rifiutato
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stdout(io.StringIO()):
            fake_trainer()("LivenessNet", "casia_fasd", few, 12, 16, 1e-3, out_dir=out, split_check=sc, test_eval=False)
        self.assertIn("not by subject", str(cm.exception.code))
        self.assertFalse(out.exists())

    def test_cli_options_and_defaults(self):
        tree = ast.parse(TRAIN_CNN.read_text(encoding="utf-8"))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        guard = next(n for n in tree.body if isinstance(n, ast.If) and "__main__" in ast.unparse(n.test))
        self.assertEqual(ast.unparse(guard.body[0]), "main()")
        opts = {}
        for n in ast.walk(main):
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "ap.add_argument":
                opts[n.args[0].value] = {k.arg: ast.unparse(k.value) for k in n.keywords if k.arg != "help"}
        self.assertEqual(opts, {
            "--dataset": {"required": "True", "choices": "list(DATASETS)"}, "--arch": {"nargs": "'*'", "default": "ARCHS"},
            "--per-class": {"type": "int", "default": "1500"}, "--epochs": {"type": "int", "default": "10"},
            "--batch": {"type": "int", "default": "16"}, "--lr": {"type": "float", "default": "0.001"},
            "--protocol": {"choices": "['official', 'pooled']", "default": "'official'"}, "--out-dir": {"default": "None"},
            "--no-test-eval": {"action": "'store_true'"}})
        call = next(n for n in ast.walk(main) if isinstance(n, ast.Call) and ast.unparse(n.func) == "train")
        self.assertEqual(ast.unparse(call), "train(arch, a.dataset, data, a.epochs, a.batch, a.lr, a.protocol, out_dir=a.out_dir, test_eval=not a.no_test_eval)")
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
        self.assertEqual([ast.unparse(d) for d in fn.args.defaults], ["'official'", "None", "None", "True"])


# ============================================================================ train e controllo dei pesi

class TestTrainStep(Base):
    def test_train_and_weights_check(self):
        w = self.world()
        pipeline(w, "train")
        rec = json.loads((w.run / "train.json").read_text())
        self.assertEqual(sorted(rec["models"]), sorted(m["id"] for m in w.cfg["models"]))
        self.assertEqual(sorted(os.listdir(w.run / "checkpoints")), sorted(f"{a}__casia_fasd.{x}" for a in ("LivenessNet", "AttackNetV2_2")
                                                                          for x in ("h5", "json", "keras")))
        self.assertEqual(rec["test_evaluation"], "disabled")
        for m in w.cfg["models"]:
            card = json.loads((w.run / "checkpoints" / rec["models"][m["id"]]["card_file"]).read_text())
            self.assertEqual(rec["models"][m["id"]]["train_seconds"], card["train_seconds"])
        self.assertEqual(os.listdir(TMP_ROOT / "weights"), [DECOY.name])

    def test_authorization_and_epochs(self):
        w = self.world()
        self.assertPass(run_c2(w.args("check")))
        a = [x for x in w.args("train") if x != "--i-am-authorized"]
        self.assertFail(run_c2(a), "--i-am-authorized", "train")
        a = w.args("train")
        a[a.index("--epochs") + 1] = "10"
        self.assertFail(run_c2(a), "differs from the configuration budget")
        self.assertNoOutputs(w.run, "train")

    def test_split_not_reproduced(self):
        """Manifest coerenti ma con un altro soggetto nel dev: il check passa (non usa scikit-learn), il training si
        ferma sul controllo dello split e non lascia pesi."""
        w = self.world()
        dev_s = {r["subject"] for r in w.rows["source_dev"]}
        fit_s = sorted({r["subject"] for r in w.rows["source_train"]}, key=int)
        swap_in, swap_out = fit_s[0], sorted(dev_s, key=int)[0]
        allrows = w.rows["source_train"] + w.rows["source_dev"]
        new_dev = [r for r in allrows if r["subject"] in (dev_s - {swap_out}) | {swap_in}]
        new_fit = [r for r in allrows if r not in new_dev]
        w.write_manifest("source_train", new_fit)
        w.write_manifest("source_dev", new_dev)
        w.write_config()
        self.assertPass(run_c2(w.args("check")))
        self.assertFail(run_c2(w.args("train")), "does not reproduce its manifest", "train")
        self.assertNoOutputs(w.run, "train")

    def test_existing_or_pooled_weights_refused(self):
        w = self.world()
        w.cfg["checkpoints"]["excluded_pooled"]["LivenessNet__casia_fasd-pooled.h5"] = hashlib.sha256(b"h5:LivenessNet" + b"best").hexdigest()
        w.write_config()
        self.assertPass(run_c2(w.args("check")))
        self.assertFail(run_c2(w.args("train")), "existing checkpoint refused by the configuration (excluded_pooled", "train")
        self.assertNoOutputs(w.run, "train")

    def test_wall_time_cap_boundary(self):
        import time as _time
        for secs, ok in ((59, True), (60, True), (61, False)):
            w = World(self.tmp / f"cap{secs}")
            w.cfg["budget"]["wall_time_cap_minutes"] = 1
            w.write_config()
            self.assertPass(run_c2(w.args("check")))
            clock = iter([1000.0, 1000.0 + secs])
            fake = types.SimpleNamespace(time=lambda: next(clock), strftime=_time.strftime)
            with mock.patch.object(C2, "time", fake):
                res = run_c2(w.args("train"))
            if ok:
                self.assertPass(res)
                self.assertEqual(json.loads((w.run / "train.json").read_text())["wall_seconds"], secs)
            else:
                self.assertFail(res, "above the cap of 1 minutes", "train")

    def test_wall_time_cap(self):
        w = self.world()
        w.cfg["budget"]["wall_time_cap_minutes"] = 0
        w.write_config()
        self.assertPass(run_c2(w.args("check")))
        clock = iter(range(0, 10 ** 6, 100))
        with mock.patch.object(C2.time, "time", lambda: next(clock)):
            self.assertFail(run_c2(w.args("train")), "above the cap", "train")
        self.assertNoOutputs(w.run, "train")

    def tampered(self, fn, text):
        w = self.world()
        pipeline(w, "train")
        fn(w)
        self.assertFail(run_c2(w.args("score", "--set", "dev")), text, "score")
        self.assertNoOutputs(w.run, "score_dev", "select", "score_tests", "apply")

    def test_tampered_checkpoint(self):
        def f(w):
            p = w.run / "checkpoints" / "LivenessNet__casia_fasd.h5"
            p.write_bytes(p.read_bytes() + b"x")
        self.tampered(f, "weights SHA-256 differs from the card")

    def test_tampered_card(self):
        def f(w):
            p = w.run / "checkpoints" / "AttackNetV2_2__casia_fasd.json"
            d = json.loads(p.read_text())
            d["n_val"] += 1
            p.write_text(json.dumps(d))
        self.tampered(f, "card SHA-256 differs from train.json")

    def redeclared_card(self, w, edit):
        p = w.run / "checkpoints" / "AttackNetV2_2__casia_fasd.json"
        d = json.loads(p.read_text())
        edit(d)
        p.write_text(json.dumps(d))
        rec = json.loads((w.run / "train.json").read_text())
        rec["models"]["attacknet_v2_2__casia_fasd_c2"]["card_sha256"] = C2.sha256_file(p)
        (w.run / "train.json").write_text(json.dumps(rec))

    def test_card_with_test_field_redeclared(self):
        self.tampered(lambda w: self.redeclared_card(w, lambda d: d.update(test={"acer": 0.1}, n_test=5)), "holds a test evaluation")

    def test_card_restored_epoch_redeclared(self):
        self.tampered(lambda w: self.redeclared_card(w, lambda d: d.update(restored_epoch=5)), "is not the epoch of minimum val_loss")

    def test_card_best_val_loss_redeclared(self):
        self.tampered(lambda w: self.redeclared_card(w, lambda d: d.update(best_val_loss=d["best_val_loss"] + 0.5)), "is not the epoch of minimum val_loss")

    def test_card_hyperparameters_redeclared(self):
        self.tampered(lambda w: self.redeclared_card(w, lambda d: d.update(lr=0.01)), "card field lr")

    def test_trainer_gets_no_test_frames(self):
        w = self.world()
        seen = []
        real = fake_trainer()

        def spy(arch, ds, data, *a, **k):
            seen.append((sorted(data), k.get("test_eval"), k.get("out_dir") is not None, k.get("split_check") is not None))
            return real(arch, ds, data, *a, **k)
        with mock.patch.object(C2, "TRAINER", spy):
            pipeline(w, "train")
        self.assertEqual(seen, [(["X_train", "s_train", "y_train"], False, True, True)] * 2)

    def test_code_changed_since_check(self):
        w = self.world()
        self.assertPass(run_c2(w.args("check")))
        real = C2.code_hashes()
        with mock.patch.object(C2, "code_hashes", lambda: {**real, "scripts/c2_pilot.py": "0" * 64}):
            self.assertFail(run_c2(w.args("train")), "the code changed since check", "train")

    def test_missing_train_record(self):
        w = self.world()
        pipeline(w, "train")
        (w.run / "train.json").unlink()
        self.assertFail(run_c2(w.args("score", "--set", "dev")), "missing train.json")


# ============================================================================ punteggi grezzi

class TestScore(Base):
    def test_dev_raw_records(self):
        w = self.world()
        pipeline(w, "score_dev")
        rows = read_rows(w.run / "raw_scores_dev.csv")
        self.assertEqual(len(rows), 2 * len(w.rows["source_dev"]))
        dev = w.rows_by("source_dev")
        rec = json.loads((w.run / "train.json").read_text())
        for r in rows:
            self.assertEqual((r["threshold"], r["prediction"]), ("null", "null"))
            self.assertEqual(r["split"], "dev")
            self.assertEqual(r["file_sha256"], dev[r["sample_id"]]["sha256"])
            self.assertEqual(r["checkpoint_sha256"], rec["models"][r["model_id"]]["weights_sha256"])
            self.assertEqual(r["config_sha256"], C2.sha256_file(w.config))
            self.assertEqual(repr(float(r["score_attack"])), r["score_attack"])
            self.assertEqual(r["attack_type"] == "null", r["label"] == "0")
        self.assertEqual(list(rows[0]), C2.SCORE_COLUMNS)
        self.assertFalse((w.run / "raw_scores_source_test.csv").exists())

    def test_record_fields_against_the_schema(self):
        """Campi di research/templates/score-schema.json (più model_id e file_sha256), valori fissati per dataset."""
        schema = ["run_id", "sample_id", "dataset_id", "split", "subject_id", "session_id", "video_id", "attack_type", "modality",
                  "label", "score_attack", "threshold", "prediction", "checkpoint_sha256", "config_sha256"]
        self.assertEqual(C2.SCORE_COLUMNS, schema + ["model_id", "file_sha256"])
        w = self.world()
        pipeline(w, "score_tests")
        tok = {"3": "warped", "5": "cut", "7": "video", "HR_4": "video"}
        for name, role in (("raw_scores_dev.csv", "source_dev"), ("raw_scores_source_test.csv", "source_test"),
                           ("raw_scores_target.csv", "target_test")):
            man = w.rows_by(role)
            for r in read_rows(w.run / name):
                m = man[r["sample_id"]]
                self.assertEqual((r["run_id"], r["modality"], r["label"]), (TRIAL, "RGB", m["label"]))
                self.assertEqual(r["split"], "dev" if role == "source_dev" else "test")
                if role == "target_test":       # NUAA: soggetto e sessione dal nome, nessun video, attacchi stampati
                    self.assertEqual((r["dataset_id"], r["subject_id"], r["session_id"], r["video_id"]), ("nuaa", m["subject"], m["session"], "null"))
                    self.assertEqual(r["attack_type"], "print" if m["label"] == "1" else "null")
                else:                           # CASIA-FASD: soggetto e video con lo split, nessuna sessione
                    self.assertEqual((r["dataset_id"], r["subject_id"], r["session_id"], r["video_id"]), ("casia_fasd", m["subject_key"], "null", m["video_key"]))
                    self.assertEqual(r["attack_type"], tok[m["video"]] if m["label"] == "1" else "null")

    def test_round_trip_precision(self):
        w = self.world()
        vals = [0.1 + 0.2, 1 / 3, 2 ** -40, 0.0, 1.0]
        scorer = lambda arch, p, s: (lambda X: [vals[i % len(vals)] for i in range(len(X))])   # noqa: E731
        with mock.patch.object(C2, "MAKE_SCORER", scorer):
            pipeline(w, "score_dev")
        got = {float(r["score_attack"]) for r in read_rows(w.run / "raw_scores_dev.csv")}
        self.assertEqual(got, set(vals))
        self.assertIn("0.30000000000000004", (w.run / "raw_scores_dev.csv").read_text())

    def test_tests_refused_without_threshold(self):
        w = self.world()
        pipeline(w, "score_dev")
        self.assertFail(run_c2(w.args("score", "--set", "source_test,target")), "threshold.json is missing", "score")
        self.assertNoOutputs(w.run, "score_tests")
        self.assertFail(run_c2(w.args("score", "--set", "target")), "use 'dev' or 'source_test,target'")

    def test_threshold_sha_recorded_before_test_scoring(self):
        w = self.world()
        seen = {}

        def scorer(arch, p, sha):
            f = default_scorer(arch, p, sha)

            def s(X):
                if (w.run / "threshold.json").exists() and "thr" not in seen:
                    seen["thr"] = C2.sha256_file(w.run / "threshold.json")
                return f(X)
            return s
        with mock.patch.object(C2, "MAKE_SCORER", scorer):
            pipeline(w, "score_tests")
        rec = json.loads((w.run / "score_tests.json").read_text())
        self.assertEqual(rec["threshold_sha256_before_scoring"], C2.sha256_file(w.run / "threshold.json"))
        self.assertEqual(seen["thr"], rec["threshold_sha256_before_scoring"])
        for name in ("raw_scores_source_test.csv", "raw_scores_target.csv"):
            self.assertTrue(all(r["threshold"] == "null" and r["prediction"] == "null" for r in read_rows(w.run / name)))

    def test_threshold_changed_during_scoring(self):
        w = self.world()
        pipeline(w, "select")

        def scorer(arch, p, sha):
            f = default_scorer(arch, p, sha)

            def s(X):
                thr = w.run / "threshold.json"
                thr.write_text(thr.read_text() + " ")
                return f(X)
            return s
        with mock.patch.object(C2, "MAKE_SCORER", scorer):
            self.assertFail(run_c2(w.args("score", "--set", "source_test,target")), "changed while the test sets were scored")
        self.assertNoOutputs(w.run, "score_tests")

    def test_edited_threshold_refused_before_test_scoring(self):
        w = self.world()
        pipeline(w, "select")
        p = w.run / "threshold.json"
        d = json.loads(p.read_text())
        mid = next(iter(d["models"]))
        d["models"][mid]["value"] = 0.5
        p.write_text(json.dumps(d, indent=1))
        self.assertFail(run_c2(w.args("score", "--set", "source_test,target")), "differs from the selection recomputed")

    def test_invalid_scores_refused(self):
        for bad in (float("nan"), 1.5, -0.1):
            w = World(self.tmp / f"w{bad}")
            scorer = lambda arch, p, s, b=bad: (lambda X: [b] * len(X))   # noqa: E731
            pipeline(w, "train")
            with mock.patch.object(C2, "MAKE_SCORER", scorer):
                self.assertFail(run_c2(w.args("score", "--set", "dev")), "non-finite or outside [0, 1]")
            self.assertNoOutputs(w.run, "score_dev")

    def test_tf_scorer_cpu_and_inference_mode(self):
        """Lo scorer vero con un tensorflow finto: GPU nascosta, CPU, training=False, pixel / 255, colonna dell'attacco,
        pesi verificati per SHA-256 prima del caricamento."""
        import numpy as np
        calls = []

        class Dev:
            def __init__(self, name):
                self.name = name

            def __enter__(self):
                calls.append(("device", self.name))

            def __exit__(self, *a):
                return False

        class M:
            def load_weights(self, p):
                calls.append(("load", Path(p).name))

            def __call__(self, x, training=None):
                calls.append(("call", training, float(x.max()) if len(x) else 0.0, len(x)))
                return np.stack([1 - x[:, 0, 0, 0], x[:, 0, 0, 0]], axis=1)
        tf = types.ModuleType("tensorflow")
        tf.config = types.SimpleNamespace(set_visible_devices=lambda d, kind: calls.append(("visible", list(d), kind)),
                                          optimizer=types.SimpleNamespace(set_experimental_options=lambda o: None))
        tf.device = Dev
        arch = types.ModuleType("kuznetsov_architectures")
        arch.create_model = lambda name: types.SimpleNamespace(get_model=lambda: M())
        wfile = self.tmp / "w.h5"
        wfile.write_bytes(b"weights")
        with modules({"tensorflow": tf, "kuznetsov_architectures": arch}), \
                mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            score = C2.tf_scorer("LivenessNet", wfile, C2.sha256_file(wfile))
            X = np.zeros((3, 2, 2, 3), np.uint8)
            X[:, 0, 0, 0] = (0, 51, 255)
            s = score(X)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertEqual(list(s), [0.0, float(np.float32(51) / np.float32(255)), 1.0])
        self.assertIn(("visible", [], "GPU"), calls)
        self.assertTrue(all(c[1] == "/CPU:0" for c in calls if c[0] == "device"))
        self.assertTrue(all(c[1] is False for c in calls if c[0] == "call"))
        self.assertTrue(all(c[2] <= 1.0 for c in calls if c[0] == "call"))
        self.assertIn(("load", "w.h5"), calls)
        with modules({"tensorflow": tf, "kuznetsov_architectures": arch}):
            self.assertEqual(len(C2.tf_scorer("LivenessNet", wfile, C2.sha256_file(wfile))(np.zeros((0, 2, 2, 3), np.uint8))), 0)
            # lotti: 2 x 16 + 1 frame, ogni frame con il proprio punteggio, lotti da 16, 16 e 1
            calls.clear()
            n = 2 * C2.SCORE_BATCH + 1
            X = np.zeros((n, 2, 2, 3), np.uint8)
            X[:, 0, 0, 0] = np.arange(n) * 7
            s = C2.tf_scorer("LivenessNet", wfile, C2.sha256_file(wfile))(X)
            self.assertEqual(list(s), [float(np.float32(v) / np.float32(255)) for v in np.arange(n) * 7])
            self.assertEqual(C2.SCORE_BATCH, 16)
            sizes = [c[3] for c in calls if c[0] == "call"]
            self.assertEqual(sizes, [16, 16, 1])
        with modules({"tensorflow": tf, "kuznetsov_architectures": arch}):
            with self.assertRaises(C2.C2Error):
                C2.tf_scorer("LivenessNet", wfile, "0" * 64)


# ============================================================================ select

class TestSelect(Base):
    def test_select_dev_only(self):
        import numpy as np
        w = self.world()
        pipeline(w, "score_dev")
        opened = []

        def hook(ev, args):
            if ev == "open" and args and isinstance(args[0], (str, os.PathLike)) and str(TMP_ROOT) in os.fsdecode(args[0]):
                opened.append(Path(os.fsdecode(args[0])).name)
        HOOKS.append(hook)
        try:
            self.assertPass(run_c2(w.args("select")))
        finally:
            HOOKS.remove(hook)
        self.assertFalse([n for n in opened if n.endswith(".npz")], opened)
        self.assertFalse([n for n in opened if n.startswith("raw_scores_") and n != "raw_scores_dev.csv"], opened)
        doc = json.loads((w.run / "threshold.json").read_text())
        rows = read_rows(w.run / "raw_scores_dev.csv")
        for mid, e in doc["models"].items():
            sel = sorted((r for r in rows if r["model_id"] == mid), key=lambda r: r["sample_id"])
            y = np.array([int(r["label"]) for r in sel]); s = np.array([float(r["score_attack"]) for r in sel])
            # ricalcolo indipendente: minimo di |fp*n_a - fn*n_b| sui candidati, pareggio verso il più alto
            n_a, n_b = int((y == 1).sum()), int((y == 0).sum())
            cands = sorted(set(s.tolist()), reverse=True) + [float(np.nextafter(s.min(), -np.inf))]
            best = min(cands, key=lambda t: (abs(int((s[y == 0] > t).sum()) * n_a - int((s[y == 1] <= t).sum()) * n_b), -t))
            self.assertEqual(e["value"], best)
            self.assertEqual(e["dev"]["accepted_attacks"], int((s[y == 1] <= best).sum()))
            self.assertEqual(e["dev"]["rejected_bona_fide"], int((s[y == 0] > best).sum()))
            self.assertEqual(e["dev"]["n_subjects_bona_fide"], len({r["subject_id"] for r in sel if r["label"] == "0"}))
            self.assertEqual(e["dev"]["n_subjects_attack"], len({r["subject_id"] for r in sel if r["label"] == "1"}))
            cm, _ = C2.metrics_api()
            self.assertEqual((e["dev"]["roc_eer_information_only"], e["dev"]["auc"]), (cm(y, s)["eer"], cm(y, s)["auc"]))
        text = (w.run / "threshold.json").read_text()
        for r in w.rows["source_dev"]:
            self.assertNotIn(r["key"].split("/", 1)[1], text)
            self.assertNotIn(r["subject_key"], text)
            self.assertNotIn(r["sha256"], text)

    def inject(self, w, row_fn, text):
        pipeline(w, "score_dev")
        rewrite_scores(w, "raw_scores_dev.csv", row_fn, record="score_dev.json")
        self.assertFail(run_c2(w.args("select")), text, "select")
        self.assertNoOutputs(w.run, "select", "score_tests", "apply")

    def extra_row(self, w, role, sha_from=None):
        """Riga in più nei punteggi del dev con chiave (o solo SHA-256) di una riga del test sorgente o di NUAA."""
        def fn(rows):
            r = dict(rows[0])
            src = w.rows[role][0]
            if sha_from:
                r["file_sha256"] = src["sha256"]
                r["sample_id"] = r["sample_id"].replace("_0_", "_9_")
            else:
                r.update(C2.expected_fields(role, src))
                r["split"] = "dev"
            return rows + [r]
        return fn

    def test_refuses_source_test_row_by_key(self):
        w = self.world()
        self.inject(w, self.extra_row(w, "source_test"), "(1 by key, 1 by SHA-256)")

    def test_refuses_nuaa_row_by_key(self):
        w = self.world()
        self.inject(w, self.extra_row(w, "target_test"), "rows of the source test or of NUAA among the dev scores (1 by key")

    def test_refuses_row_by_sha_only(self):
        for role in ("source_test", "target_test"):
            w = World(self.tmp / role)
            self.inject(w, self.extra_row(w, role, sha_from=True), "(0 by key, 1 by SHA-256)")

    def test_refuses_dev_row_relabelled_with_test_sha(self):
        # stessa chiave del dev, SHA-256 di un file NUAA: nessuna riga in più, rifiutata per SHA-256
        w = self.world()
        def fn(rows):
            rows[0]["file_sha256"] = w.rows["target_test"][0]["sha256"]
            return rows
        self.inject(w, fn, "(0 by key, 1 by SHA-256)")

    def test_no_nuaa_session_rule(self):
        w = self.world()
        pipeline(w, "score_dev")
        boom = mock.Mock(side_effect=AssertionError("NUAA session rule used"))
        with mock.patch.object(C2.st, "session_of", boom), mock.patch.object(C2.st, "check_not_test", boom):
            self.assertPass(run_c2(w.args("select")))
        src = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("session_of", src)
        self.assertNotIn("check_not_test", src)

    def test_incomplete_or_inconsistent_dev(self):
        cases = {
            "missing row": (lambda rows: rows[1:], "manifest rows without score"),
            "duplicate row": (lambda rows: rows + [rows[0]], "duplicated (model, sample) rows"),
            "NaN": (lambda rows: [dict(rows[0], score_attack="nan")] + rows[1:], "1 scores not finite"),
            "above 1": (lambda rows: [dict(rows[0], score_attack="1.5")] + rows[1:], "1 scores not finite, outside [0, 1]"),
            "below 0": (lambda rows: [dict(rows[0], score_attack="-0.5")] + rows[1:], "1 scores not finite, outside [0, 1]"),
            "not repr": (lambda rows: [dict(rows[0], score_attack="0.50")] + rows[1:], "not round-trip"),
            "threshold set": (lambda rows: [dict(rows[0], threshold="0.5")] + rows[1:], "threshold or prediction not null"),
            "label flipped": (lambda rows: [dict(rows[0], label="1" if rows[0]["label"] == "0" else "0")] + rows[1:], "1 rows with fields different from the manifest"),
            "other model": (lambda rows: [dict(rows[0], model_id="livenessnet__casia_fasd-pooled")] + rows[1:], "models outside the configuration"),
            "other checkpoint": (lambda rows: [dict(rows[0], checkpoint_sha256="0" * 64)] + rows[1:], "1 rows with another checkpoint SHA-256"),
        }
        for name, (fn, text) in cases.items():
            w = World(self.tmp / name.replace(" ", "_"))
            self.inject(w, fn, text)

    def test_scores_changed_without_record(self):
        w = self.world()
        pipeline(w, "score_dev")
        rewrite_scores(w, "raw_scores_dev.csv", lambda rows: [dict(rows[0], score_attack="0.25")] + rows[1:])
        self.assertFail(run_c2(w.args("select")), "SHA-256 differs from score_dev.json")

    def fake_ctx(self):
        w = self.world()
        cfg, sha = C2.load_config(w.config)
        return C2.Ctx(w.run, cfg, sha, {}, True)

    def rows_for(self, ctx, bf, att):
        rows = []
        for mid, _ in C2.model_list(ctx.cfg):
            for i, v in enumerate(bf):
                rows.append({"model_id": mid, "sample_id": f"b{i}", "label": "0", "score_attack": repr(float(v)), "subject_id": f"s{i}", "attack_type": "null"})
            for i, v in enumerate(att):
                rows.append({"model_id": mid, "sample_id": f"a{i}", "label": "1", "score_attack": repr(float(v)), "subject_id": f"s{i}", "attack_type": "cut"})
        return rows

    def test_m1_hand_derived_cases(self):
        """Casi del selettore M1 (delivery/m1/metrics.md, emendamento del 2/10), criterio eer."""
        ctx = self.fake_ctx()
        cases = [  # (bona fide, attacchi, soglia, attacchi accettati, bona fide rifiutati, pareggi alla soglia)
            ([0.1], [0.9], 0.1, 0, 0, 1),
            ([0.5], [0.5000001], 0.5, 0, 0, 1),
            ([0.05, 0.15, 0.25, 0.35], [i / 10 for i in range(1, 11)], 0.3, 3, 1, 1),
            ([0.2, 0.5, 0.5], [0.5, 0.5, 0.8], 0.5, 2, 0, 4),
            ([0.3, 0.3], [0.3, 0.3], 0.3, 2, 0, 4),
            ([0.0, 0.0], [0.0, 0.0], 0.0, 2, 0, 4),
            ([0.3, 0.7, 0.8], [0.2, 0.9], 0.7, 1, 1, 1),
        ]
        for bf, att, t, acc, rej, ties in cases:
            out = C2.selection(ctx, self.rows_for(ctx, bf, att))
            for e in out.values():
                self.assertEqual(e["value"], t, (bf, att))
                self.assertEqual((e["dev"]["accepted_attacks"], e["dev"]["rejected_bona_fide"], e["dev"]["ties_at_value"]), (acc, rej, ties), (bf, att))
        out = C2.selection(ctx, self.rows_for(ctx, [0.8, 0.9], [0.1, 0.2]))     # punteggi invertiti
        self.assertTrue(all(e["flag"].startswith("non-informative") for e in out.values()))
        out = C2.selection(ctx, self.rows_for(ctx, [0.3, 0.3], [0.3, 0.3]))     # costanti: AUC 0,5, nessuna segnalazione
        self.assertTrue(all(e["dev"]["auc"] == 0.5 and e["flag"] == "" for e in out.values()))

    def test_single_class_dev(self):
        ctx = self.fake_ctx()
        for bf, att in (([0.1, 0.2], []), ([], [0.3, 0.4])):
            with self.assertRaises(C2.C2Error) as cm:
                C2.selection(ctx, self.rows_for(ctx, bf, att))
            self.assertIn("single class", str(cm.exception))

    def test_uses_m1_selector_unchanged(self):
        self.assertEqual(C2.sha256_file(ROOT / "scripts" / "select_threshold.py"), SELECT_THRESHOLD_SHA256)
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        self.assertFalse(defined & {"threshold_counts", "candidate_thresholds", "select_threshold"})
        used = {ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Attribute) and ast.unparse(n.value) == "st"}
        self.assertTrue({"st.threshold_counts", "st.candidate_thresholds", "st.select_threshold"} <= used)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == "st.select_threshold"]
        self.assertEqual([ast.unparse(c.args[2]) for c in calls], ["'eer'"])


# ============================================================================ apply

class TestApply(Base):
    def test_apply_records_and_table(self):
        w = self.world()
        pipeline(w, "apply")
        thr = json.loads((w.run / "threshold.json").read_text())
        table = json.loads((w.run / "c2_table.json").read_text())
        values = {mid: e["value"] for mid, e in thr["models"].items()}
        for e, (i, role) in C2.EVALUATIONS.items():
            mid = w.cfg["models"][i]["id"]
            rows = read_rows(w.run / f"scores_{e}.csv")
            self.assertEqual(len(rows), len(w.rows[role]))
            self.assertEqual({r["sample_id"] for r in rows}, set(w.rows_by(role)))
            for r in rows:
                self.assertEqual(r["threshold"], repr(values[mid]))
                self.assertEqual(r["prediction"], "1" if float(r["score_attack"]) > values[mid] else "0")
            cell = next(x for x in table["evaluations"] if x["id"] == e)["at_dev_threshold"]
            att = [r for r in rows if r["label"] == "1"]
            bf = [r for r in rows if r["label"] == "0"]
            self.assertEqual(cell["accepted_attacks"], sum(r["prediction"] == "0" for r in att))
            self.assertEqual(cell["rejected_bona_fide"], sum(r["prediction"] == "1" for r in bf))
            self.assertEqual(cell["apcer"], cell["accepted_attacks"] / len(att))
            self.assertEqual(cell["bpcer"], cell["rejected_bona_fide"] / len(bf))
            self.assertEqual(cell["acer"], (cell["apcer"] + cell["bpcer"]) / 2)
            self.assertEqual(cell["ties_at_threshold"], sum(float(r["score_attack"]) == values[mid] for r in rows))
            self.assertEqual(cell["n_subjects_attack"], len({r["subject_id"] for r in att}))
            types_ = {"source_test": ["cut", "video", "warped"], "target_test": ["print"]}[role]
            self.assertEqual(sorted(cell["apcer_by_attack_type"]), types_)
            self.assertEqual(sum(v["n_attack"] for v in cell["apcer_by_attack_type"].values()), len(att))
            fixed = next(x for x in table["evaluations"] if x["id"] == e)["at_fixed_0_5"]
            self.assertEqual(fixed["threshold"], 0.5)
            self.assertIn("reference only", fixed["label"])
        dev = read_rows(w.run / "scores_dev.csv")
        self.assertEqual(len(dev), 2 * len(w.rows["source_dev"]))
        self.assertTrue(all(r["threshold"] != "null" for r in dev))
        md = (w.run / "c2_table.md").read_text()
        self.assertIn("0.5 (fixed, reference)", md)
        for r in w.rows["source_test"] + w.rows["target_test"]:
            self.assertNotIn(r["key"], md + json.dumps(table))
        for name in ("raw_scores_source_test.csv", "raw_scores_target.csv"):     # i grezzi restano grezzi
            self.assertTrue(all(r["threshold"] == "null" for r in read_rows(w.run / name)))

    def test_ties_at_threshold_are_bona_fide(self):
        w = self.world()
        with mock.patch.object(C2, "MAKE_SCORER", lambda arch, p, s: (lambda X: [0.5] * len(X))):
            pipeline(w, "apply")
        thr = json.loads((w.run / "threshold.json").read_text())
        self.assertTrue(all(e["value"] == 0.5 for e in thr["models"].values()))
        for e in C2.EVALUATIONS:
            rows = read_rows(w.run / f"scores_{e}.csv")
            self.assertTrue(all(r["prediction"] == "0" for r in rows))
        table = json.loads((w.run / "c2_table.json").read_text())
        for x in table["evaluations"]:
            self.assertEqual((x["at_dev_threshold"]["apcer"], x["at_dev_threshold"]["bpcer"]), (1.0, 0.0))
            self.assertEqual(x["at_dev_threshold"]["ties_at_threshold"], x["at_dev_threshold"]["n"])
            self.assertEqual(x["descriptive"]["auc"], 0.5)
        disp = json.loads((w.run / "subject_dispersion.json").read_text())
        for e in disp["evaluations"].values():         # pareggi alla soglia: bona fide accettati, attacchi accettati
            self.assertEqual(set(e["bpcer_per_subject"]["values_sorted"]), {0.0})
            self.assertEqual(set(e["apcer_per_subject"]["values_sorted"]), {1.0})

    def test_missing_class_is_null(self):
        import numpy as np
        cm, bt = C2.metrics_api()
        y, s = np.array([1, 1]), np.array([0.2, 0.9])
        c = C2.cell(cm, bt, y, s, ["a", "b"], ["cut", "video"], 0.5, "casia_fasd")
        self.assertIsNone(c["bpcer"])
        self.assertIsNone(c["acer"])
        self.assertEqual((c["n_bona_fide"], c["n_subjects_bona_fide"], c["apcer"]), (0, 0, 0.5))
        self.assertIsNone(c["apcer_by_attack_type"]["warped"]["apcer"])
        self.assertEqual(c["apcer_by_attack_type"]["warped"]["n_attack"], 0)

    def test_threshold_changed_after_recording(self):
        w = self.world()
        pipeline(w, "score_tests")
        p = w.run / "threshold.json"
        p.write_text(p.read_text().replace('"set": "source_dev"', '"set": "source_dev" '))
        self.assertFail(run_c2(w.args("apply")), "changed since its SHA-256 was recorded", "apply")
        self.assertNoOutputs(w.run, "apply")

    def test_threshold_edited_and_record_updated(self):
        w = self.world()
        pipeline(w, "score_tests")
        p = w.run / "threshold.json"
        d = json.loads(p.read_text())
        for e in d["models"].values():
            e["value"] = 0.99
        p.write_text(json.dumps(d, indent=1))
        rec = json.loads((w.run / "score_tests.json").read_text())
        rec["threshold_sha256_before_scoring"] = C2.sha256_file(p)
        (w.run / "score_tests.json").write_text(json.dumps(rec))
        self.assertFail(run_c2(w.args("apply")), "differs from the selection recomputed")
        self.assertNoOutputs(w.run, "apply")

    def test_raw_scores_tampered(self):
        cases = {
            "NaN": (lambda rows: [dict(rows[0], score_attack="nan")] + rows[1:], "scores not finite"),
            "missing": (lambda rows: rows[:-1], "manifest rows without score"),
            "duplicate": (lambda rows: rows + [rows[-1]], "duplicated (model, sample) rows"),
            "outside": (lambda rows: [dict(rows[0], sample_id=rows[0]["sample_id"].replace("_0.jpg", "_9.jpg"))] + rows[1:], "rows outside the manifest"),
            "prediction": (lambda rows: [dict(rows[0], prediction="1")] + rows[1:], "threshold or prediction not null"),
            "file hash": (lambda rows: [dict(rows[0], file_sha256="0" * 64)] + rows[1:], "fields different from the manifest"),
        }
        for name, (fn, text) in cases.items():
            w = World(self.tmp / name.replace(" ", "_"))
            pipeline(w, "score_tests")
            rewrite_scores(w, "raw_scores_target.csv", fn, record="score_tests.json")
            self.assertFail(run_c2(w.args("apply")), text, "apply")
            self.assertNoOutputs(w.run, "apply")

    def test_raw_scores_changed_without_record(self):
        w = self.world()
        pipeline(w, "score_tests")
        rewrite_scores(w, "raw_scores_source_test.csv", lambda rows: rows[:-1])
        self.assertFail(run_c2(w.args("apply")), "SHA-256 differs from score_tests.json")

    def test_pass_then_failure_leaves_no_old_pass(self):
        w = self.world()
        pipeline(w, "apply")
        rewrite_scores(w, "raw_scores_target.csv", lambda rows: rows[:-1], record="score_tests.json")
        self.assertFail(run_c2(w.args("apply")), step="apply")
        self.assertNoOutputs(w.run, "apply")
        self.assertTrue((w.run / "threshold.json").exists())

    def test_atomic_write(self):
        w = self.world()
        pipeline(w, "score_tests")
        real = os.replace
        for exc in (OSError("disk full (simulated)"), KeyboardInterrupt()):     # anche un'eccezione che non è OSError
            n = {"i": 0}

            def flaky(a, b):
                n["i"] += 1
                if n["i"] == 3:
                    raise exc
                return real(a, b)
            with mock.patch.object(C2.os, "replace", flaky), contextlib.redirect_stdout(io.StringIO()) as o, \
                    contextlib.redirect_stderr(io.StringIO()):
                if isinstance(exc, OSError):        # errore imprevisto: FAIL esplicito, codice 1
                    self.assertEqual(C2.main(w.args("apply")), 1)
                    self.assertIn("RESULT: FAIL (apply)", o.getvalue())
                else:                               # interruzione: codice 130, FAIL, nessun file parziale
                    try:
                        rc = C2.main(w.args("apply"))
                    except KeyboardInterrupt:       # un'interruzione non gestita fermerebbe la suite: qui è un fallimento
                        rc = "KeyboardInterrupt propagated"
                    self.assertEqual(rc, 130)
                    self.assertIn("RESULT: FAIL (apply)", o.getvalue())
                    st_ = json.loads((w.run / "status.json").read_text())
                    self.assertEqual((st_["result"], st_["reason"]), ("FAIL", "interrupted (KeyboardInterrupt)"))
            self.assertNoOutputs(w.run, "apply")
            self.assertFalse([x for x in os.listdir(w.run) if x.endswith(".tmp")])

    def test_public_scan(self):
        w = self.world()
        pipeline(w, "check")
        inp = C2.load_inputs(w.cfg, C2.parser().parse_args(w.args("select")), caches=False)
        C2.public_scan(json.dumps({"value": 0.5, "n": 3}), inp, "x")
        for leak in (w.rows["source_dev"][0]["key"].split("/", 1)[1], w.rows["source_dev"][0]["subject_key"],
                     w.rows["target_test"][0]["key"], w.rows["source_test"][0]["sha256"]):
            with self.assertRaises(C2.C2Error):
                C2.public_scan(f"note {leak} here", inp, "x")


# ============================================================================ invalidazione, pipeline completa

class TestInvalidation(Base):
    def test_rerun_of_a_step_removes_downstream(self):
        w = self.world()
        pipeline(w, "apply")
        self.assertPass(run_c2(w.args("score", "--set", "dev")))
        self.assertTrue((w.run / "raw_scores_dev.csv").exists())
        self.assertNoOutputs(w.run, "select", "score_tests", "apply")
        self.assertTrue((w.run / "checkpoints").is_dir())
        self.assertPass(run_c2(w.args("check")))
        self.assertNoOutputs(w.run, "train", "score_dev", "select", "score_tests", "apply")

    def test_deletions_go_through_path_guard(self):
        w = self.world()
        pipeline(w, "train")
        seen = []
        real = C2.pg.safe_rmtree

        def spy(path, *a, **k):
            seen.append(Path(path).name)
            return real(path, *a, **k)
        with mock.patch.object(C2.pg, "safe_rmtree", spy):
            self.assertPass(run_c2(w.args("train")))
        self.assertIn("checkpoints", seen)

    def test_all_command(self):
        w = self.world()
        self.assertPass(run_c2(w.args("all")))
        for s in C2.STEPS:
            for n in C2.OUTPUTS[s]:
                self.assertTrue((w.run / n).exists(), n)

    def test_all_stops_at_first_failure(self):
        w = self.world()
        with mock.patch.object(C2, "MAKE_SCORER", lambda arch, p, s: (lambda X: [float("nan")] * len(X))):
            self.assertFail(run_c2(w.args("all")), "non-finite", "all")
        self.assertTrue((w.run / "train.json").exists())
        self.assertNoOutputs(w.run, "score_dev", "select", "score_tests", "apply")


# ============================================================================ struttura della suite e repository vero

def snapshot() -> dict:
    """Stato del repository vero: git status (con i file non versionati) e, per le cartelle ignorate da git (dati, pesi,
    risultati, export), nome, dimensione e data di modifica di ogni file."""
    out = {"git": None, "files": {}}
    try:
        r = subprocess.run(["git", "--no-optional-locks", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
                           capture_output=True, text=True, timeout=60)
        # i file compilati di Python (__pycache__) creati importando il codice non contano
        out["git"] = "\n".join(l for l in r.stdout.splitlines() if "__pycache__" not in l and not l.endswith(".pyc")) if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        out["git"] = None
    for d in ("data/processed", "models/weights", "results", "export"):
        base = ROOT / d
        if base.is_dir():
            for dp, _, fs in os.walk(base):
                for f in fs:
                    p = Path(dp) / f
                    try:
                        s = p.lstat()
                    except OSError:
                        continue
                    out["files"][str(p.relative_to(ROOT))] = (s.st_size, s.st_mtime_ns)
    return out


START = None
HOOKS = []


def _audit(ev, args):
    for h in list(HOOKS):
        h(ev, args)


class TestRefusalsAndStatus(Base):
    def status(self, w):
        return json.loads((w.run / "status.json").read_text())

    def test_option_refusal_keeps_outputs_and_marks_fail(self):
        """Opzioni rifiutate (autorizzazione, epoche) prima di qualunque modifica: le uscite restano, status.json dice FAIL."""
        w = self.world()
        pipeline(w, "apply")
        self.assertEqual(self.status(w)["result"], "PASS")
        before = {n: (w.run / n).read_bytes() for n in os.listdir(w.run) if (w.run / n).is_file() and n != "status.json"}
        for args, text, cmd in (([x for x in w.args("train") if x != "--i-am-authorized"], "--i-am-authorized", "train"),
                                (w.args("train")[:-3] + ["--epochs", "11", "--i-am-authorized"], "differs from the configuration budget", "train"),
                                ([x for x in w.args("score", "--set", "dev") if x != "--i-am-authorized"], "--i-am-authorized", "score")):
            self.assertFail(run_c2(args), text, cmd)
            after = {n: (w.run / n).read_bytes() for n in os.listdir(w.run) if (w.run / n).is_file() and n != "status.json"}
            self.assertEqual(after, before, text)
            self.assertTrue((w.run / "checkpoints").is_dir())
            st_ = self.status(w)
            self.assertEqual(st_["result"], "FAIL")
            self.assertTrue(st_["outputs"].startswith("untouched"))

    def test_identity_refusal_leaves_outputs_and_marks_fail(self):
        w = self.world()
        pipeline(w, "apply")
        before = {n: (w.run / n).read_bytes() for n in os.listdir(w.run) if (w.run / n).is_file() and n != "status.json"}
        w.cfg["status"] = "edited"
        w.write_config()
        self.assertFail(run_c2(w.args("apply")), "differs from the one recorded at check", "apply")
        after = {n: (w.run / n).read_bytes() for n in os.listdir(w.run) if (w.run / n).is_file() and n != "status.json"}
        self.assertEqual(before, after)
        st_ = self.status(w)
        self.assertEqual((st_["result"], st_["step"]), ("FAIL", "apply"))
        self.assertTrue(st_["outputs"].startswith("untouched"))

    def snapshot_outputs(self, w):
        return {n: (w.run / n).read_bytes() for n in sorted(os.listdir(w.run)) if (w.run / n).is_file() and n != "status.json"}

    def test_every_identity_refusal_leaves_outputs(self):
        cases = {
            "code": (lambda w: mock.patch.object(C2, "code_hashes", lambda real=C2.code_hashes(): {**real, "scripts/path_guard.py": "0" * 64}),
                     "the code changed since check"),
            "commit": (None, "commit or state differs from check"),
            "check.json of another run": (None, "belongs to another run folder"),
            "check.json not passed": (None, "check.json does not record a passed step"),
        }
        for name, (patcher, text) in cases.items():
            w = World(self.tmp / name.replace(" ", "_").replace(".", "_"), RUN if name == "commit" else TRIAL)
            with mock.patch.object(C2, "APPROVED_CONFIG_SHA256", C2.sha256_file(w.config)):
                pipeline(w, "apply")
                if name == "check.json of another run":
                    d = json.loads((w.run / "check.json").read_text()); d["run_id"] = RUN + "-x"
                    (w.run / "check.json").write_text(json.dumps(d))
                if name == "check.json not passed":
                    d = json.loads((w.run / "check.json").read_text()); d["passed"] = "true"
                    (w.run / "check.json").write_text(json.dumps(d))
                before = self.snapshot_outputs(w)
                ctx = patcher(w) if patcher else contextlib.nullcontext()
                if name == "commit":
                    ctx = mock.patch.object(C2, "REPO_STATE", lambda run: {"commit": "abcdef1" + "0" * 33, "dirty": False})
                with ctx:
                    self.assertFail(run_c2(w.args("apply")), text, "apply")
            self.assertEqual(self.snapshot_outputs(w), before, name)
            st_ = self.status(w)
            self.assertEqual(st_["result"], "FAIL", name)
            self.assertTrue(st_["outputs"].startswith("untouched"), name)

    def test_no_status_in_a_refused_folder(self):
        bad = self.tmp / "not-a-run-name"
        bad.mkdir()
        w = self.world()
        for cmd in (("check",), ("train",), ("select",), ("apply",)):
            self.assertFail(run_c2(w.args(*cmd, run=bad)), "run folder refused")
        self.assertEqual(os.listdir(bad), [])
        r = self.tmp / "repo"
        for d in ("results/m1", "models/weights", "data", "docs", "scripts", "src", "delivery", "config", ".git", "export"):
            (r / d).mkdir(parents=True)
        target = r / "results" / "m1" / TRIAL
        target.mkdir()
        with mock.patch.object(C2, "ROOT", r):
            pipeline_args = w.args("train", run=target)
            self.assertFail(run_c2(pipeline_args), "run folder refused", "train")
            self.assertFail(run_c2(w.args("check", run=target)), "run folder refused", "check")
        self.assertEqual(os.listdir(target), [])
        # anche dopo un comando che ha accettato un'altra cartella, nella stessa sessione
        pipeline(w, "check")
        self.assertFail(run_c2(w.args("select", run=bad)), "run folder refused")
        self.assertEqual(os.listdir(bad), [])

    def test_new_folder_rule(self):
        for name, make in (("dir", lambda d: (d / "status.json").mkdir()),
                           ("symlink", lambda d: (d / "status.json").symlink_to(d.parent / "elsewhere.json")),
                           ("check", lambda d: (d / "check.json").write_text("{}")),
                           ("subdir", lambda d: (d / "checkpoints").mkdir()),
                           ("ds_store", lambda d: (d / ".DS_Store").write_bytes(b"x"))):
            w = World(self.tmp / name, RUN)
            w.run.mkdir()
            (w.run.parent / "elsewhere.json").write_text("{}")
            make(w.run)
            with mock.patch.object(C2, "APPROVED_CONFIG_SHA256", C2.sha256_file(w.config)):
                self.assertFail(run_c2(w.args("check")), "must be new", "check")
        w = World(self.tmp / "only_status", RUN)
        w.run.mkdir()
        (w.run / "status.json").write_text('{"result": "FAIL"}')
        with mock.patch.object(C2, "APPROVED_CONFIG_SHA256", C2.sha256_file(w.config)):
            self.assertPass(run_c2(w.args("check")))

    def test_status_running_and_pass_write_failure(self):
        w = self.world()
        pipeline(w, "train")
        seen, cmds = [], []

        def scorer(arch, p, sha):
            f = default_scorer(arch, p, sha)

            def s(X):
                d = json.loads((w.run / "status.json").read_text())
                seen.append(d["result"])
                cmds.append((d["command"], d["step"]))
                return f(X)
            return s
        with mock.patch.object(C2, "MAKE_SCORER", scorer):
            self.assertPass(run_c2(w.args("score", "--set", "dev")))
        self.assertEqual(set(seen), {"RUNNING"})
        self.assertEqual(set(cmds), {("score", "score_dev")})
        self.assertEqual(json.loads((w.run / "status.json").read_text())["command"], "score")
        real = C2.write_status

        def broken(result, cmd, reason=""):
            if result == "PASS":
                raise PermissionError("status.json read-only (simulated)")
            return real(result, cmd, reason)
        with mock.patch.object(C2, "write_status", broken):
            self.assertFail(run_c2(w.args("select")), "PermissionError", "select")
        (w.run / "status.json").unlink()
        (w.run / "status.json").mkdir()
        self.assertFail(run_c2(w.args("select")), "status.json in the run folder is not a regular file", "select")

    def test_failed_select_or_test_scoring_removes_downstream(self):
        w = self.world()
        pipeline(w, "apply")
        rewrite_scores(w, "raw_scores_dev.csv", lambda rows: rows[1:], record="score_dev.json")
        self.assertFail(run_c2(w.args("select")), "manifest rows without score", "select")
        self.assertNoOutputs(w.run, "select", "score_tests", "apply")
        w2 = World(self.tmp / "w2")
        pipeline(w2, "apply")
        with mock.patch.object(C2, "MAKE_SCORER", lambda arch, p, s: (lambda X: [float("nan")] * len(X))):
            self.assertFail(run_c2(w2.args("score", "--set", "source_test,target")), "non-finite", "score")
        self.assertNoOutputs(w2.run, "score_tests", "apply")
        self.assertTrue((w2.run / "threshold.json").exists())

    def test_threshold_raw_dev_hash_checked(self):
        w = self.world()
        pipeline(w, "score_tests")
        p = w.run / "threshold.json"
        d = json.loads(p.read_text())
        d["raw_scores_dev_sha256"] = "0" * 64
        p.write_text(json.dumps(d, indent=1))
        rec = json.loads((w.run / "score_tests.json").read_text())
        rec["threshold_sha256_before_scoring"] = C2.sha256_file(p)
        (w.run / "score_tests.json").write_text(json.dumps(rec))
        self.assertFail(run_c2(w.args("apply")), "differs from the selection recomputed", "apply")

    def test_reference_architectures_pinned(self):
        lv = self.tmp / "livedetection"
        (lv / "src").mkdir(parents=True)
        (lv / "src" / "architectures.py").write_text("# architectures v1\n")
        w = self.world()
        with mock.patch.object(C2, "LIVEDETECTION", lv):
            self.assertPass(run_c2(w.args("check")))
            doc = json.loads((w.run / "check.json").read_text())
            self.assertEqual(doc["code_sha256"][C2.ARCH_KEY], C2.sha256_file(lv / "src" / "architectures.py"))
            (lv / "src" / "architectures.py").write_text("# architectures v2\n")
            self.assertFail(run_c2(w.args("train")), "the code changed since check", "train")
        arch = Path(C2.LIVEDETECTION) / "src" / "architectures.py"
        self.assertEqual(C2.code_hashes()[C2.ARCH_KEY], C2.sha256_file(arch) if arch.is_file() else "absent")
        self.assertIn("src/tesi_app/paths.py", C2.CODE_FILES)
        self.assertIn("scripts/package_rules.py", C2.CODE_FILES)

    def test_check_json_of_another_run(self):
        w = self.world()
        pipeline(w, "check")
        d = json.loads((w.run / "check.json").read_text())
        d["run_id"] = RUN
        (w.run / "check.json").write_text(json.dumps(d))
        self.assertFail(run_c2(w.args("train")), "belongs to another run folder", "train")

    def test_unexpected_exceptions(self):
        w = self.world()
        pipeline(w, "train")

        def boom(arch, p, s):
            def f(X):
                raise RuntimeError("simulated backend failure")
            return f
        with mock.patch.object(C2, "MAKE_SCORER", boom):
            self.assertFail(run_c2(w.args("score", "--set", "dev")), "unexpected error RuntimeError", "score")
        self.assertNoOutputs(w.run, "score_dev", "select", "score_tests", "apply")
        self.assertEqual(self.status(w)["result"], "FAIL")

        def bad_trainer(*a, **k):
            Path(k["out_dir"], "partial.h5").write_bytes(b"x")
            raise KeyError("X_test")
        with mock.patch.object(C2, "TRAINER", bad_trainer):
            self.assertFail(run_c2(w.args("train")), "unexpected error KeyError", "train")
        self.assertNoOutputs(w.run, "train")

    def test_metrics_code_is_pinned(self):
        self.assertIn("src/tesi_app/evaluation.py", C2.CODE_FILES)
        self.assertIn("scripts/select_threshold.py", C2.CODE_FILES)
        self.assertIn("src/tesi_app/evaluation.py", C2.code_hashes())


class TestTrainCnnGuards(Base):
    def data(self):
        w = self.world()
        a = w.arrays["source"]
        return {"X_train": a["X_train"], "y_train": a["y_train"], "s_train": a["s_train"], "X_test": a["X_test"], "y_test": a["y_test"]}

    def test_no_test_eval_requires_out_dir(self):
        weights = self.tmp / "weights_guard"
        weights.mkdir()
        (weights / "manifest.json").write_text("{}")
        with mock.patch.object(TC, "WEIGHTS", weights), self.assertRaises(SystemExit) as cm:
            fake_trainer()("LivenessNet", "casia_fasd", self.data(), 12, 16, 1e-3, test_eval=False)
        self.assertIn("--no-test-eval requires --out-dir", str(cm.exception.code))
        self.assertEqual(os.listdir(weights), ["manifest.json"])
        self.assertEqual((weights / "manifest.json").read_text(), "{}")

    def test_out_dir_protected_folders(self):
        r = self.tmp / "repo"
        for d in ("data/processed", "models/weights", "results/c1", "results/m1", "results/m2", "results/m2_rgb", "export/drive",
                  "docs", "scripts", "src", "delivery", "config", ".git"):
            (r / d).mkdir(parents=True)
        with mock.patch.object(TC, "ROOT", r):
            for where in ("results/c1/x", "results/m1/x", "results/m2/x", "results/m2_rgb/x", "data/x", "models/x", "export/drive/x",
                          "docs/x", "results", "."):
                self.assertTrue(TC.out_dir_problem(r / where), where)
            self.assertEqual(TC.out_dir_problem(r / "results" / "c2" / TRIAL / "checkpoints.tmp"), "")
            self.assertEqual(TC.out_dir_problem(self.tmp / "outside" / "ck"), "")
            moved = self.tmp / "moved_data"
            moved.mkdir()
            with mock.patch.object(TC, "DATA", moved):
                self.assertIn("data folder", TC.out_dir_problem(moved / "x"))
                self.assertIn("data folder", TC.out_dir_problem(self.tmp))       # antenato della cartella dei dati
            with self.assertRaises(SystemExit) as cm:
                fake_trainer()("LivenessNet", "casia_fasd", self.data(), 12, 16, 1e-3, out_dir=r / "results" / "m1" / "ck", test_eval=False)
            self.assertIn("--out-dir refused", str(cm.exception.code))
        self.assertFalse((r / "results" / "m1" / "ck").exists())

    def test_out_dir_alias_by_identity(self):
        """Un altro percorso verso la cartella dei pesi: collegamento simbolico e alias simulato (una seconda cartella vera
        di cui os.path.samefile riporta l'identità della cartella dei pesi, come nel test M2 delle grafie)."""
        weights = self.tmp / "weights_alias"
        weights.mkdir()
        link = self.tmp / "link_to_weights"
        link.symlink_to(weights, target_is_directory=True)
        alias = self.tmp / "other_folder"
        alias.mkdir()
        real = os.path.samefile

        def same(a, b):
            ids = {Path(a).resolve(), Path(b).resolve()}
            return ids == {alias.resolve(), weights.resolve()} or real(a, b)
        with mock.patch.object(TC, "WEIGHTS", weights):
            self.assertIn("weights folder", TC.out_dir_problem(link / "ck"))
            with mock.patch("os.path.samefile", same):
                self.assertIn("weights folder", TC.out_dir_problem(alias / "ck"))
                self.assertIn("weights folder", TC.out_dir_problem(alias))
            self.assertEqual(TC.out_dir_problem(alias / "ck"), "")


class TestRecordsAndWeights(Base):
    def test_scores_come_from_the_frame_of_each_key(self):
        w = self.world()
        pipeline(w, "score_tests")
        arch = {m["id"]: m["arch"] for m in w.cfg["models"]}
        frames_of = {"source_dev": (w.arrays["source"]["X_train"], rm.spread_keys(w.rows["source_train"] + w.rows["source_dev"], 1500)),
                     "source_test": (w.arrays["source"]["X_test"], rm.spread_keys(w.rows["source_test"], 1500)),
                     "target_test": (w.arrays["target"]["X_test"], nm.spread_keys(w.rows["target_test"], 1500))}
        for role, name in (("source_dev", "raw_scores_dev.csv"), ("source_test", "raw_scores_source_test.csv"), ("target_test", "raw_scores_target.csv")):
            X, order = frames_of[role]
            pos = {k: i for i, k in enumerate(order)}
            rows = read_rows(w.run / name)
            self.assertTrue(rows)
            for r in rows:
                want = default_scorer(arch[r["model_id"]], None, None)(X[[pos[r["sample_id"]]]])[0]
                self.assertEqual(float(r["score_attack"]), float(want), (role, r["sample_id"]))
                self.assertEqual(r["config_sha256"], C2.sha256_file(w.config))

    def test_casia_attack_types(self):
        self.assertEqual(C2.CASIA_ATTACK_TYPE, {"3": "warped", "4": "warped", "HR_2": "warped", "5": "cut", "6": "cut", "HR_3": "cut",
                                                "7": "video", "8": "video", "HR_4": "video"})
        for v, t in (("1", "null"), ("2", "null"), ("HR_1", "null"), ("4", "warped"), ("HR_2", "warped"), ("6", "cut"), ("HR_3", "cut"), ("8", "video")):
            name = f"3_{v}.avi_0_{'real' if v in ('1', '2', 'HR_1') else 'fake'}.jpg"
            self.assertEqual(C2.expected_fields("source_test", rm.casia_row("test", name, H("t"), 1))["attack_type"], t, v)
        w = self.world()
        pipeline(w, "apply")
        table = json.loads((w.run / "c2_table.json").read_text())
        rows = read_rows(w.run / "scores_E1.csv")
        tok = {"3": "warped", "5": "cut", "7": "video", "HR_4": "video"}
        exp = {}
        for r in rows:
            if r["label"] == "1":
                t = tok[w.rows_by("source_test")[r["sample_id"]]["video"]]
                n, acc = exp.get(t, (0, 0))
                exp[t] = (n + 1, acc + (r["prediction"] == "0"))
        self.assertEqual({t: n for t, (n, _) in exp.items()}, {"warped": 12, "cut": 12, "video": 24})
        cell = next(x for x in table["evaluations"] if x["id"] == "E1")["at_dev_threshold"]["apcer_by_attack_type"]
        for t, (n, acc) in exp.items():
            self.assertEqual((cell[t]["n_attack"], cell[t]["accepted"], cell[t]["apcer"]), (n, acc, acc / n))
        nuaa = next(x for x in table["evaluations"] if x["id"] == "E3")["at_dev_threshold"]
        self.assertEqual(nuaa["apcer_by_attack_type"]["print"]["apcer"], nuaa["apcer"])

    def test_official_weights_refused(self):
        w = self.world()
        w.cfg["checkpoints"]["alternative_not_recommended"]["AttackNetV2_2__casia_fasd.h5"] = hashlib.sha256(b"h5:AttackNetV2_2" + b"best").hexdigest()
        w.write_config()
        self.assertPass(run_c2(w.args("check")))
        self.assertFail(run_c2(w.args("train")), "refused by the configuration (alternative_not_recommended", "train")
        self.assertNoOutputs(w.run, "train")

    def tampered(self, fn, text):
        w = self.world()
        pipeline(w, "train")
        fn(w)
        self.assertFail(run_c2(w.args("score", "--set", "dev")), text, "score")
        self.assertNoOutputs(w.run, "score_dev", "select", "score_tests", "apply")

    def card(self, w, arch, edit, redeclare=True, field=None):
        p = w.run / "checkpoints" / f"{arch}__casia_fasd.json"
        d = json.loads(p.read_text())
        edit(d)
        p.write_text(json.dumps(d))
        if redeclare:
            rec = json.loads((w.run / "train.json").read_text())
            mid = next(m["id"] for m in w.cfg["models"] if m["arch"] == arch)
            rec["models"][mid]["card_sha256"] = C2.sha256_file(p)
            if field:
                rec["models"][mid].update(field(d))
            (w.run / "train.json").write_text(json.dumps(rec))

    def test_keras_file_changed(self):
        def f(w):
            p = w.run / "checkpoints" / "LivenessNet__casia_fasd.keras"
            p.write_bytes(p.read_bytes() + b"x")
        self.tampered(f, "Keras file SHA-256 differs")

    def test_same_weights_for_two_models(self):
        def f(w):
            src = w.run / "checkpoints" / "LivenessNet__casia_fasd.h5"
            dst = w.run / "checkpoints" / "AttackNetV2_2__casia_fasd.h5"
            shutil.copy2(src, dst)
            sha = C2.sha256_file(dst)
            self.card(w, "AttackNetV2_2", lambda d: d.update(weights_sha256=sha), field=lambda d: {"weights_sha256": sha})
        self.tampered(f, "the two models have the same weights")

    def test_card_split_check_not_passed(self):
        self.tampered(lambda w: self.card(w, "LivenessNet", lambda d: d.update(split_check="failed")), "does not record a passed split check")

    def test_card_epochs_above_budget(self):
        def edit(d):
            d["epochs_run"] = 13
            d["val_loss_history"] = d["val_loss_history"] + [9.0] * (13 - len(d["val_loss_history"]))
        self.tampered(lambda w: self.card(w, "LivenessNet", edit, field=lambda d: {"epochs_run": 13}), "not consistent with the budget")

    def test_train_record_of_another_configuration(self):
        def f(w):
            rec = json.loads((w.run / "train.json").read_text())
            rec["config_sha256"] = "0" * 64
            (w.run / "train.json").write_text(json.dumps(rec))
        self.tampered(f, "train.json was written with another configuration")

    def test_score_row_of_another_configuration(self):
        w = self.world()
        pipeline(w, "score_dev")
        rewrite_scores(w, "raw_scores_dev.csv", lambda rows: [dict(rows[0], config_sha256="0" * 64)] + rows[1:], record="score_dev.json")
        self.assertFail(run_c2(w.args("select")), "1 rows with another configuration SHA-256", "select")

    def test_subject_dispersion(self):
        w = self.world()
        pipeline(w, "apply")
        disp = json.loads((w.run / "subject_dispersion.json").read_text())
        self.assertEqual(sorted(disp["evaluations"]), ["E1", "E2", "E3", "E4"])
        for e in ("E1", "E2", "E3", "E4"):           # ogni valutazione con il proprio modello (scorer diversi per architettura)
            rows = read_rows(w.run / f"scores_{e}.csv")
            self.assertEqual(disp["evaluations"][e]["model"], {r["model_id"] for r in rows}.pop())
            for lab, key in (("0", "bpcer_per_subject"), ("1", "apcer_per_subject")):
                per = {}
                for r in rows:
                    if r["label"] == lab:
                        n, err = per.get(r["subject_id"], (0, 0))
                        per[r["subject_id"]] = (n + 1, err + (r["prediction"] != lab))
                vals = sorted(err / n for n, err in per.values())
                got = disp["evaluations"][e][key]
                self.assertEqual(sorted(got), ["max", "median", "min", "n_subjects", "values_sorted"])
                import numpy as np
                self.assertEqual((got["values_sorted"], got["n_subjects"], got["min"], got["max"], got["median"]),
                                 (vals, len(vals), vals[0], vals[-1], float(np.median(vals))))
            self.assertEqual(sorted(disp["evaluations"][e]), ["apcer_per_subject", "bpcer_per_subject", "model", "set"])
        text = (w.run / "subject_dispersion.json").read_text()
        for r in w.rows["source_test"] + w.rows["target_test"]:
            self.assertNotIn(r["key"], text)


class TestThirdReview(Base):
    """Rilievi della terza revisione (20/10): ogni rifiuto dichiarato nel README ha un test."""

    def test_run_folder_alias_by_identity(self):
        """Cartelle protette spostate (pesi, dati, risultati consegnati) raggiunte da un altro percorso: alias simulato
        (una seconda cartella vera di cui os.path.samefile riporta l'identità della cartella protetta)."""
        real = os.path.samefile
        for name in ("weights", "data", "results/eval", "results/c1", "results/m1", "results/m2", "results/m2_rgb"):
            base = self.tmp / name.replace("/", "_")
            moved = base / "target"
            moved.mkdir(parents=True)
            alias = base / "alias"
            alias.mkdir()

            res = base / "res"
            if name.startswith("results/"):
                moved.rmdir()
                moved = res / name.split("/", 1)[1]
                moved.mkdir(parents=True)

            def same(a, b, alias=alias, moved=moved):
                return {Path(a).resolve(), Path(b).resolve()} == {alias.resolve(), moved.resolve()} or real(a, b)
            patches = {"weights": ("WEIGHTS", moved), "data": ("DATA", moved)}.get(name, ("RESULTS", res))
            with mock.patch.object(C2, patches[0], patches[1]):
                self.assertIn("protected", C2.run_dir_problem(moved / TRIAL), name)
                self.assertEqual(C2.run_dir_problem(alias / TRIAL), "", name)
                with mock.patch("os.path.samefile", same):
                    self.assertIn("protected", C2.run_dir_problem(alias / TRIAL), name)

    def test_run_id_must_end_after_the_commit(self):
        for bad in (RUN + "-foo", TRIAL + "-x", RUN + "x", "x" + RUN):
            self.assertIn("is not YYYYMMDD", C2.run_dir_problem(self.tmp / bad), bad)

    def test_stale_temporary_files_removed(self):
        w = self.world()
        pipeline(w, "apply")
        (w.run / "threshold.json.tmp").write_text("stale")
        (w.run / "scores_E1.csv.tmp").write_text("stale")
        (w.run / "checkpoints.tmp").mkdir()
        (w.run / "checkpoints.tmp" / "old.h5").write_bytes(b"stale")
        self.assertPass(run_c2(w.args("check")))
        left = sorted(os.listdir(w.run))
        self.assertEqual(left, ["check.json", "status.json"])

    def test_card_with_n_test_only(self):
        w = self.world()
        pipeline(w, "train")
        p = w.run / "checkpoints" / "LivenessNet__casia_fasd.json"
        d = json.loads(p.read_text())
        d["n_test"] = 7
        p.write_text(json.dumps(d))
        rec = json.loads((w.run / "train.json").read_text())
        rec["models"]["livenessnet__casia_fasd_c2"]["card_sha256"] = C2.sha256_file(p)
        (w.run / "train.json").write_text(json.dumps(rec))
        self.assertFail(run_c2(w.args("score", "--set", "dev")), "holds a test evaluation", "score")

    def test_identifier_in_threshold_json_refused(self):
        w = self.world()
        pipeline(w, "score_dev")
        leak = w.rows["source_dev"][0]["key"].split("/", 1)[1]
        real = C2.selection

        def leaky(ctx, rows):
            out = real(ctx, rows)
            for e in out.values():
                e["flag"] = f"see {leak}"
            return out
        with mock.patch.object(C2, "selection", leaky):
            self.assertFail(run_c2(w.args("select")), "threshold.json: identifiers in a public file", "select")
        self.assertNoOutputs(w.run, "select", "score_tests", "apply")

    def test_markdown_table_scanned_as_markdown(self):
        w = self.world()
        pipeline(w, "score_tests")
        marker = "T" + "BD"
        real = C2.table_md
        with mock.patch.object(C2, "table_md", lambda doc: real(doc) + f"\n{marker}\n"):
            self.assertFail(run_c2(w.args("apply")), "c2_table.md: identifiers in a public file", "apply")
        self.assertNoOutputs(w.run, "apply")
        leak = w.rows["target_test"][0]["key"]
        with mock.patch.object(C2, "table_md", lambda doc: real(doc) + f"\n{leak}\n"):
            self.assertFail(run_c2(w.args("apply")), "c2_table.md: identifiers in a public file", "apply")

    def test_dev_frame_in_source_test_cache(self):
        w = self.world()
        order = rm.spread_keys(w.rows["source_train"] + w.rows["source_dev"], 1500)
        i_dev = order.index(w.rows["source_dev"][0]["key"])
        w.arrays["source"]["X_test"][3] = w.arrays["source"]["X_train"][i_dev]
        w.write_cache("source")
        w.write_config()
        self.assertFail(run_c2(w.args("check")), "1 frames shared between training and source-test", "check")

    def test_cache_shapes(self):
        import numpy as np
        w = self.world()
        a = w.arrays["target"]
        a["X_test"] = np.zeros((len(a["X_test"]), 5, 4, 3), np.uint8) + frames(3, len(a["X_test"]))[:, :1, :1]
        w.write_cache("target")
        w.write_config()
        self.assertFail(run_c2(w.args("check")), "frame shapes differ between source and target caches", "check")
        w = World(self.tmp / "w_count")
        a = w.arrays["source"]
        n = len(a["X_test"])
        a["X_test"] = np.concatenate([a["X_test"], frames(9, 1)])            # un frame in più, y e s corretti
        w.write_cache("source")
        w.cfg["derived_frames"]["source"]["X_test"] = n                     # conteggio dichiarato invariato
        w.write_config()
        self.assertFail(run_c2(w.args("check")), "(uint8 frames N x H x W x 3 expected)", "check")

    def test_raw_scores_bom_cr_header(self):
        cases = {
            "BOM": (lambda t: "\ufeff" + t, "byte-order mark or CR line endings"),
            "CR": (lambda t: t.replace("\n", "\r\n"), "byte-order mark or CR line endings"),
            "header": (lambda t: t.replace("score_attack", "score", 1), "header differs from the score record fields"),
        }
        for name, (fn, text) in cases.items():
            w = World(self.tmp / name)
            pipeline(w, "score_dev")
            p = w.run / "raw_scores_dev.csv"
            p.write_bytes(fn(p.read_text(encoding="utf-8")).encode("utf-8"))
            d = json.loads((w.run / "score_dev.json").read_text())
            d["files"]["raw_scores_dev.csv"]["sha256"] = C2.sha256_file(p)
            (w.run / "score_dev.json").write_text(json.dumps(d))
            self.assertFail(run_c2(w.args("select")), text, "select")

    def test_worst_attack_type(self):
        w = self.world()
        pipeline(w, "apply")
        table = json.loads((w.run / "c2_table.json").read_text())
        for e in table["evaluations"]:
            for c in (e["at_dev_threshold"], e["at_fixed_0_5"]):
                vals = [v["apcer"] for v in c["apcer_by_attack_type"].values() if v["apcer"] is not None]
                self.assertEqual(c["apcer_worst_attack_type"], max(vals))
        cells = [x["at_fixed_0_5"]["apcer_by_attack_type"] for x in table["evaluations"] if x["dataset"] == "casia_fasd"]
        self.assertTrue(any(len({v["apcer"] for v in c.values()}) > 1 for c in cells), "fixture without different per-type APCER")

    def test_overlap_by_key(self):
        w = self.world()
        w.write_manifest("source_train", w.rows["source_train"] + [w.rows["source_dev"][0]])
        w.write_config()
        self.assertFail(run_c2(w.args("check")), "source_train and source_dev share 1 values of key", "check")

    def test_test_scoring_needs_authorisation(self):
        w = self.world()
        pipeline(w, "apply")
        self.assertFail(run_c2([x for x in w.args("score", "--set", "source_test,target") if x != "--i-am-authorized"]),
                        "score needs --i-am-authorized", "score")
        for n in C2.OUTPUTS["score_tests"] + C2.OUTPUTS["apply"]:      # rifiuto prima di qualunque modifica
            self.assertTrue((w.run / n).exists(), n)

    def test_threshold_read_back(self):
        w = self.world()
        pipeline(w, "score_dev")
        real = C2.write_all

        def corrupting(run, files):
            real(run, files)
            if "threshold.json" in files:
                p = Path(run) / "threshold.json"
                d = json.loads(p.read_text())
                for e in d["models"].values():
                    e["value"] = 0.123456
                p.write_text(json.dumps(d))
        with mock.patch.object(C2, "write_all", corrupting):
            self.assertFail(run_c2(w.args("select")), "threshold.json read back differs from the selection", "select")
        self.assertFalse((w.run / "threshold.json").exists())

    def test_missing_numpy_exit_code(self):
        log = self.tmp / "np.txt"
        log.write_text("")
        env = {**os.environ, "C2_TEST_LOG": str(log), "C2_TEST_BLOCK": "numpy"}
        r = subprocess.run([sys.executable, "-I", "-B", "-c", BOOT, str(SCRIPT), "check", "--config", "x", "--run-dir", str(self.tmp / TRIAL)],
                           env=env, capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("ERROR: missing Python package 'numpy'", r.stderr)
        self.assertIn("RESULT: FAIL (import)", r.stdout)
        self.assertFalse((self.tmp / TRIAL).exists())

    def test_mutation_runner_timeout(self):
        mut = load("check_c2_mutations_timeout", ROOT / "scripts" / "check_c2_mutations.py")

        def slow(*a, **k):
            raise subprocess.TimeoutExpired(cmd="suite", timeout=k.get("timeout"))
        with mock.patch.object(mut.subprocess, "run", slow):
            rc, line = mut.run_tests(self.tmp, self.tmp / "e")
        self.assertIsNone(rc)
        self.assertTrue(line.startswith("timeout after"), line)
        self.assertEqual(mut.detected(rc, line), "invalid")

    def test_mutation_verdicts(self):
        mut = load("check_c2_mutations_under_test", ROOT / "scripts" / "check_c2_mutations.py")
        self.assertEqual(mut.detected(0, "tests run: 5, failures: 0, errors: 0, skipped: 3"), "survived")    # un salto non rileva
        self.assertEqual(mut.detected(1, "tests run: 5, failures: 1, errors: 0, skipped: 0"), "killed")
        self.assertEqual(mut.detected(1, "tests run: 5, failures: 0, errors: 2, skipped: 0"), "killed")
        self.assertEqual(mut.detected(1, "no test summary"), "invalid")
        self.assertEqual(mut.detected(1, "tests run: 5, failures: 0, errors: 0, skipped: 0"), "invalid")

    def test_non_informative_reasons(self):
        w = self.world()
        cfg, sha = C2.load_config(w.config)
        ctx = C2.Ctx(w.run, cfg, sha, {}, True)
        rows_for = TestSelect.rows_for
        for bf, att, flag in (
                ([0.8, 0.9], [0.1, 0.2], "non-informative (dev AUC < 0.5, operational dev EER > 0.5)"),
                ([0.7, 0.4, 0.2, 0.8], [0.1, 0.7, 0.7, 0.1], "non-informative (dev AUC < 0.5)"),
                ([0.3, 0.6], [0.7, 0.4, 0.5], "non-informative (operational dev EER > 0.5)"),
                ([0.5, 0.2], [0.8], "")):
            out = C2.selection(ctx, rows_for(self, ctx, bf, att))
            for e in out.values():
                self.assertEqual(e["flag"], flag, (bf, att))


class TestFourthReview(Base):
    """Rilievi della quarta revisione (20/10)."""

    def h5(self, w):
        p = w.run / "checkpoints" / "AttackNetV2_2__casia_fasd.h5"
        p.write_bytes(p.read_bytes() + b"x")

    def test_weights_checked_before_select_and_apply(self):
        w = self.world()
        pipeline(w, "score_dev")
        self.h5(w)
        self.assertFail(run_c2(w.args("select")), "weights check failed", "select")
        w2 = World(self.tmp / "w2")
        pipeline(w2, "score_tests")
        self.h5(w2)
        self.assertFail(run_c2(w2.args("apply")), "weights check failed", "apply")
        self.assertNoOutputs(w2.run, "apply")

    def test_epochs_passed_to_the_trainer(self):
        w = self.world()
        FakeModel.LOG.clear()
        pipeline(w, "train")
        fits = [e for e in FakeModel.LOG if e[0] == "fit"]
        self.assertEqual(sorted(fits), [("fit", "AttackNetV2_2", 12), ("fit", "LivenessNet", 12)])

    def test_table_values_recomputed(self):
        import numpy as np
        w = self.world()
        pipeline(w, "apply")
        table = json.loads((w.run / "c2_table.json").read_text())
        cm, _ = C2.metrics_api()
        for e in table["evaluations"]:
            rows = read_rows(w.run / f"scores_{e['id']}.csv")
            y = np.array([int(r["label"]) for r in rows]); s = np.array([float(r["score_attack"]) for r in rows])
            c = e["at_fixed_0_5"]
            acc, rej = int(((s <= 0.5) & (y == 1)).sum()), int(((s > 0.5) & (y == 0)).sum())
            na, nb = int((y == 1).sum()), int((y == 0).sum())
            self.assertEqual((c["threshold"], c["accepted_attacks"], c["rejected_bona_fide"], c["n_attack"], c["n_bona_fide"]), (0.5, acc, rej, na, nb))
            self.assertEqual((c["apcer"], c["bpcer"], c["acer"]), (acc / na, rej / nb, (acc / na + rej / nb) / 2))
            self.assertEqual(c["ties_at_threshold"], int((s == 0.5).sum()))
            m = cm(y, s)
            self.assertEqual(e["descriptive"], {"auc": m["auc"], "eer": m["eer"], "bpcer_at_apcer10": m["bpcer_at_apcer10"]})

    def test_non_trial_dirty_tree_after_check(self):
        w = self.world(RUN)
        with mock.patch.object(C2, "APPROVED_CONFIG_SHA256", C2.sha256_file(w.config)):
            pipeline(w, "train")
            before = {n: (w.run / n).read_bytes() for n in os.listdir(w.run) if (w.run / n).is_file() and n != "status.json"}
            with mock.patch.object(C2, "REPO_STATE", lambda run: {"commit": "abcdef0" + "0" * 33, "dirty": True}):
                self.assertFail(run_c2(w.args("score", "--set", "dev")), "commit or state differs from check", "score")
            after = {n: (w.run / n).read_bytes() for n in os.listdir(w.run) if (w.run / n).is_file() and n != "status.json"}
        self.assertEqual(before, after)
        self.assertTrue((w.run / "checkpoints").is_dir())

    def test_markdown_null_never_zero(self):
        cell = {"threshold": 0.5, "n": 2, "n_bona_fide": 0, "n_attack": 2, "accepted_attacks": 1, "rejected_bona_fide": 0,
                "ties_at_threshold": 0, "apcer": 0.5, "bpcer": None, "acer": None, "n_subjects_bona_fide": 0, "n_subjects_attack": 1,
                "apcer_by_attack_type": {"cut": {"n_attack": 0, "accepted": 0, "apcer": None}}}
        doc = {"run_id": TRIAL, "config_sha256": "0" * 64, "threshold_sha256": "1" * 64,
               "evaluations": [{"id": "E1", "model": "m", "set": "source_test", "at_dev_threshold": cell, "at_fixed_0_5": cell,
                                "descriptive": {"auc": None, "eer": None, "bpcer_at_apcer10": None}}]}
        md = C2.table_md(doc)
        row = next(l for l in md.splitlines() if l.startswith("| E1 |"))
        self.assertIn("| 0.5000 | null | null |", row)
        self.assertIn("cut null (0/0)", row)
        lines = md.splitlines()
        head = next(l for l in lines if l.startswith("| id |"))
        sep = lines[lines.index(head) + 1]
        self.assertEqual(sep.count("---|"), head.count("|") - 1)
        self.assertIn("E1 AUC null, EER null, BPCER@APCER10 null", md)

    def test_threshold_json_selector_hash(self):
        w = self.world()
        pipeline(w, "select")
        doc = json.loads((w.run / "threshold.json").read_text())
        self.assertEqual(doc["select_threshold_py_sha256"], SELECT_THRESHOLD_SHA256)

    def test_public_scan_nuaa_hashes(self):
        w = self.world()
        pipeline(w, "check")
        inp = C2.load_inputs(w.cfg, C2.parser().parse_args(w.args("select")), caches=False)
        with self.assertRaises(C2.C2Error):
            C2.public_scan(f"x {w.rows['target_test'][2]['sha256']} y", inp, "x")

    def test_cache_dtypes(self):
        import numpy as np
        for side, member, conv, text in (("source", "y_train", lambda a: a.astype(np.float64), "linkage refused"),
                                         ("target", "y_test", lambda a: a.astype(np.float64), "linkage refused"),
                                         ("source", "X_test", lambda a: a.astype(np.uint16), "uint8 frames"),
                                         ("target", "X_test", lambda a: a.astype(np.float32), "uint8 frames")):
            w = World(self.tmp / f"{side}_{member}_{conv(np.zeros(1)).dtype}")
            w.arrays[side][member] = conv(w.arrays[side][member])
            w.write_cache(side)
            w.write_config()
            self.assertFail(run_c2(w.args("check")), text, "check")

    def test_nuaa_frame_equal_to_fitting_or_dev_frame(self):
        for role in ("source_train", "source_dev"):
            w = World(self.tmp / role)
            order = rm.spread_keys(w.rows["source_train"] + w.rows["source_dev"], 1500)
            i = order.index(w.rows[role][0]["key"])
            w.arrays["target"]["X_test"][1] = w.arrays["source"]["X_train"][i]
            w.write_cache("target")
            w.write_config()
            self.assertFail(run_c2(w.args("check")), "1 frames shared between CASIA-FASD and NUAA", "check")

    def test_every_code_file_pinned(self):
        """Ogni file di CODE_FILES modificato dopo il check (in una copia dell'albero) fa rifiutare il passo successivo."""
        self.assertEqual(C2.CODE_FILES, ("scripts/c2_pilot.py", "scripts/train_cnn.py", "scripts/select_threshold.py", "scripts/path_guard.py",
                                         "scripts/rgb_manifests.py", "scripts/nuaa_manifests.py", "scripts/package_rules.py",
                                         "src/tesi_app/evaluation.py", "src/tesi_app/paths.py"))
        root = self.tmp / "root_copy"
        for f in C2.CODE_FILES:
            (root / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, root / f)
        w = self.world()
        with mock.patch.object(C2, "ROOT", root):
            for f in C2.CODE_FILES:
                self.assertPass(run_c2(w.args("check")))
                orig = (root / f).read_bytes()
                (root / f).write_bytes(orig + b"\n# changed after check\n")
                self.assertFail(run_c2(w.args("train")), "the code changed since check", "train")
                (root / f).write_bytes(orig)

    def test_existing_file_as_run_folder(self):
        w = self.world()
        w.run.write_text("not a folder")
        self.assertFail(run_c2(w.args("check")), "run folder path exists and is not a folder", "check")
        self.assertFail(run_c2(w.args("train")), "run folder does not exist or is not a folder", "train")
        self.assertEqual(w.run.read_text(), "not a folder")


class TestSweep(Base):
    """Test aggiunti dopo lo sweep automatico di mutazioni del quinto giro (20/10)."""

    def problems(self, edit):
        w = World(self.tmp / f"cfg{len(os.listdir(self.tmp))}")
        c = copy.deepcopy(w.cfg)
        edit(c)
        return " | ".join(C2.config_problems(c))

    def test_config_cases(self):
        cases = [
            (lambda c: c["budget"].update(batch=True), "budget: max_epochs"),
            (lambda c: c["models"].pop(), "models must list exactly two models"),
            (lambda c: c["models"].append({"id": "x_three", "arch": "LivenessNet"}), "models must list exactly two models"),
            (lambda c: c["models"][0].update(id="Bad-Id"), "model id 'Bad-Id' not allowed"),
            (lambda c: c["checkpoints"]["excluded_pooled"].update({"LivenessNet__casia_fasd-pooled.h5": "abc"}), "checkpoints.excluded_pooled must list"),
            (lambda c: c["manifests"]["source_dev"].update(sha256="xyz"), "manifests.source_dev.sha256 is not a SHA-256"),
            (lambda c: c["derived_frames"]["source"].update(sha256="xyz"), "derived_frames.source.sha256 is not a SHA-256"),
            (lambda c: c["manifests"]["source_train"].update(file="sub/casia_cnn_train.csv"), "manifests.source_train.file 'sub/casia_cnn_train.csv' not allowed"),
            (lambda c: c["manifests"]["source_test"].update(n_subjects=-1), "manifests.source_test.n_subjects is not a non-negative integer"),
            (lambda c: c["manifests"]["source_test"].update(n_videos=-1), "manifests.source_test.n_videos is not a non-negative integer"),
            (lambda c: c["derived_frames"]["target"].update(X_test=0), "derived_frames.target.X_test is not a positive integer"),
            (lambda c: c["budget"].update(learning_rate=0.0), "budget: max_epochs"),
            (lambda c: c["budget"].update(max_epochs=0), "budget: max_epochs"),
        ]
        for edit, text in cases:
            self.assertIn(text, self.problems(edit))
        # valori limite ammessi
        for edit in (lambda c: c["budget"].update(max_epochs=1, batch=1), lambda c: c["derived_frames"]["target"].update(X_test=1),
                     lambda c: c["manifests"]["source_test"].update(n_subjects=0, n_videos=0)):
            self.assertNotIn("budget", self.problems(edit))
            self.assertNotIn("X_test is not", self.problems(edit))
            self.assertNotIn("not a non-negative", self.problems(edit))
        # strutture sbagliate: elenco di problemi, mai un'eccezione
        for edit in (lambda c: c.update(models=[1, 2]), lambda c: c.update(models=[{"id": "a", "arch": "LivenessNet"}]),
                     lambda c: c["derived_frames"].update(source="x")):
            self.assertTrue(self.problems(edit))

    def test_config_file_errors(self):
        w = self.world()
        self.assertFail(run_c2(w.args("check")[:2] + [str(self.tmp / "nope.json")] + w.args("check")[3:]), "configuration not found")
        w.config.write_text("[]")
        self.assertFail(run_c2(w.args("check")), "is not a JSON object")

    def test_input_files_missing_or_wrong(self):
        w = self.world()
        (w.src_dir / FILES["source_test"]).unlink()
        self.assertFail(run_c2(w.args("check")), "manifest casia_cnn_test.csv (source_test) not found")
        w = World(self.tmp / "nocache")
        w.cache("target").unlink()
        self.assertFail(run_c2(w.args("check")), "target cache nuaa_1500.npz not found")
        w = World(self.tmp / "othername")
        a = w.args("check")
        shutil.copy2(w.cache("source"), w.cache_dir / "other_1500.npz")
        a[a.index("--source-cache") + 1] = str(w.cache_dir / "other_1500.npz")
        self.assertFail(run_c2(a), "is not the file of the configuration")
        w = World(self.tmp / "corrupt")
        w.cache("source").write_bytes(b"not an npz")
        w.cfg["derived_frames"]["source"]["sha256"] = C2.sha256_file(w.cache("source"))
        w.write_config()
        self.assertFail(run_c2(w.args("check")), "source cache casia_fasd_1500.npz unreadable")

    def test_target_cache_needs_only_the_test_members(self):
        w = self.world()
        for k in ("X_train", "y_train", "s_train"):
            del w.arrays["target"][k]
        w.write_cache("target")
        w.write_config()
        self.assertPass(run_c2(w.args("check")))

    def test_duplicated_keys_in_a_manifest(self):
        w = self.world()
        w.write_manifest("target_test", w.rows["target_test"] + [w.rows["target_test"][0]])
        w.write_config()
        self.assertFail(run_c2(w.args("check")), "duplicated keys")

    def test_invalid_nuaa_manifest_reported(self):
        w = self.world()
        rows = copy.deepcopy(w.rows["target_test"])
        rows[0]["label"] = "1" if rows[0]["label"] == "0" else "0"
        w.write_manifest("target_test", rows)
        w.write_config()
        self.assertFail(run_c2(w.args("check")), "nuaa_cnn_test.csv: invalid input (see the ERROR line above)", "check")

    def test_per_class_limit_of_the_cache(self):
        w = self.world()
        p = w.cache("target")
        p.rename(w.cache_dir / "nuaa_2.npz")
        w.cfg["derived_frames"]["target"]["file"] = "nuaa_2.npz"
        w.write_config()
        a = w.args("check")
        a[a.index("--target-cache") + 1] = str(w.cache_dir / "nuaa_2.npz")
        self.assertFail(run_c2(a), "manifest rows are not in the sampling order (per-class limit)")

    def test_label_array_longer_than_the_manifest(self):
        import numpy as np
        w = self.world()
        a = w.arrays["target"]
        a["y_test"] = np.concatenate([a["y_test"], [0]])
        a["s_test"] = np.concatenate([a["s_test"], ["0001"]])
        a["X_test"] = np.concatenate([a["X_test"], frames(8, 1)])
        w.write_cache("target")
        n = len(w.rows["target_test"])
        w.cfg["derived_frames"]["target"]["X_test"] = n
        w.write_config()
        self.assertFail(run_c2(w.args("check")), f"target cache X_test: {n + 1} frames, configuration {n}, manifests {n}")

    def test_missing_packages(self):
        with modules({"train_cnn": None}), mock.patch.object(C2, "TRAINER", None):
            with self.assertRaises(C2.C2Error) as cm:
                C2.load_trainer()
        self.assertIn("train needs scripts/train_cnn.py", str(cm.exception))
        with modules({"tensorflow": None}), self.assertRaises(C2.C2Error) as cm:
            C2.tf_scorer("LivenessNet", self.tmp / "w.h5", "0" * 64)
        self.assertIn("score needs TensorFlow", str(cm.exception))
        with modules({"tesi_app.evaluation": None}), self.assertRaises(C2.C2Error) as cm:
            C2.metrics_api()
        self.assertIn("select and apply use src/tesi_app/evaluation.py", str(cm.exception))

    def train_rec_edit(self, w, edit):
        rec = json.loads((w.run / "train.json").read_text())
        edit(rec)
        (w.run / "train.json").write_text(json.dumps(rec))

    def refused_at_score(self, edit, text, card_edit=None):
        w = World(self.tmp / f"t{len(os.listdir(self.tmp))}")
        pipeline(w, "train")
        if card_edit:
            card_edit(w)
        if edit:
            self.train_rec_edit(w, edit)
        self.assertFail(run_c2(w.args("score", "--set", "dev")), text, "score")

    def test_train_record_problems(self):
        mid = "livenessnet__casia_fasd_c2"
        self.refused_at_score(lambda r: r["models"].pop(mid), "train.json lists models")
        self.refused_at_score(lambda r: r["models"][mid].update(keras_file="other.keras"), "file names in train.json differ")
        self.refused_at_score(lambda r: r["models"][mid].update(weights_sha256="0" * 64), "weights SHA-256 differs from the card or from train.json")
        self.refused_at_score(lambda r: r["models"][mid].update(epochs_run=11), "train.json and card disagree on epochs")
        self.refused_at_score(None, "missing weights, Keras file or card",
                              card_edit=lambda w: (w.run / "checkpoints" / "LivenessNet__casia_fasd.keras").unlink())

    def card_edit(self, w, arch, edit):
        p = w.run / "checkpoints" / f"{arch}__casia_fasd.json"
        d = json.loads(p.read_text())
        edit(d)
        p.write_text(json.dumps(d))
        mid = next(m["id"] for m in w.cfg["models"] if m["arch"] == arch)
        self.train_rec_edit(w, lambda r: r["models"][mid].update(card_sha256=C2.sha256_file(p), epochs_run=d["epochs_run"],
                                                                  restored_epoch=d["restored_epoch"]))

    def test_card_epoch_one_accepted_and_nan_history_refused(self):
        w = self.world()
        pipeline(w, "train")
        self.card_edit(w, "LivenessNet", lambda d: d.update(epochs_run=1, val_loss_history=[0.5], restored_epoch=1, best_epoch=1, best_val_loss=0.5))
        self.assertPass(run_c2(w.args("score", "--set", "dev")))
        w2 = World(self.tmp / "nan")
        pipeline(w2, "train")
        self.card_edit(w2, "LivenessNet", lambda d: d.update(val_loss_history=[float("nan")] + d["val_loss_history"][1:]))
        self.assertFail(run_c2(w2.args("score", "--set", "dev")), "val_loss history not consistent with the budget", "score")

    def test_raw_file_problems(self):
        w = self.world()
        pipeline(w, "score_dev")
        (w.run / "raw_scores_dev.csv").unlink()
        self.assertFail(run_c2(w.args("select")), "missing raw_scores_dev.csv", "select")
        for name, fn, text in (("utf8", lambda b: b + b"\xff\xfe\n", "unreadable CSV"),
                               ("fields", lambda b: b + b"a,b\n", "rows with a wrong number of fields")):
            w = World(self.tmp / name)
            pipeline(w, "score_dev")
            p = w.run / "raw_scores_dev.csv"
            p.write_bytes(fn(p.read_bytes()))
            d = json.loads((w.run / "score_dev.json").read_text())
            d["files"]["raw_scores_dev.csv"]["sha256"] = C2.sha256_file(p)
            (w.run / "score_dev.json").write_text(json.dumps(d))
            self.assertFail(run_c2(w.args("select")), text, "select")
        w = World(self.tmp / "rec")
        pipeline(w, "score_dev")
        d = json.loads((w.run / "score_dev.json").read_text())
        d["config_sha256"] = "0" * 64
        (w.run / "score_dev.json").write_text(json.dumps(d))
        self.assertFail(run_c2(w.args("select")), "score_dev.json was written with another configuration", "select")

    def test_test_records_count_rows(self):
        w = self.world()
        pipeline(w, "score_tests")
        rec = json.loads((w.run / "score_tests.json").read_text())
        self.assertEqual(rec["files"]["raw_scores_source_test.csv"]["rows"], 2 * len(w.rows["source_test"]))
        self.assertEqual(rec["files"]["raw_scores_target.csv"]["rows"], 2 * len(w.rows["target_test"]))

    def test_selection_internal_checks(self):
        w = self.world()
        cfg, sha = C2.load_config(w.config)
        ctx = C2.Ctx(w.run, cfg, sha, {}, True)
        rows = TestSelect.rows_for(self, ctx, [0.1, 0.4], [0.6, 0.9])
        real = C2.st.select_threshold
        with mock.patch.object(C2.st, "select_threshold", lambda y, s, c: {**real(y, s, c), "value": 0.123}):
            with self.assertRaises(C2.C2Error) as cm:
                C2.selection(ctx, rows)
        self.assertIn("selected value is not a candidate threshold", str(cm.exception))
        with mock.patch.object(C2.st, "select_threshold", lambda y, s, c: {**real(y, s, c), "fn": 5}):
            with self.assertRaises(C2.C2Error) as cm:
                C2.selection(ctx, rows)
        self.assertIn("counts at the selected value differ from the selection", str(cm.exception))

    def test_decision_check(self):
        w = self.world()
        pipeline(w, "apply")
        inp = C2.load_inputs(w.cfg, C2.parser().parse_args(w.args("apply")), caches=False)
        cfg, sha = C2.load_config(w.config)
        ctx = C2.Ctx(w.run, cfg, sha, {}, True)
        rec = json.loads((w.run / "train.json").read_text())
        thr = {m: e["value"] for m, e in json.loads((w.run / "threshold.json").read_text())["models"].items()}
        mid = w.cfg["models"][0]["id"]
        rows = read_rows(w.run / "scores_E1.csv")
        self.assertEqual(C2.score_problems(rows, "source_test", ctx, inp, rec, "decision", thr, only=mid), [])
        bad = [dict(rows[0], prediction="1" if rows[0]["prediction"] == "0" else "0")] + rows[1:]
        self.assertIn("1 rows with a wrong threshold or prediction", " ".join(C2.score_problems(bad, "source_test", ctx, inp, rec, "decision", thr, only=mid)))
        bad = [dict(rows[0], threshold="0.5")] + rows[1:]
        self.assertTrue(C2.score_problems(bad, "source_test", ctx, inp, rec, "decision", thr, only=mid))

    def test_threshold_file_problems(self):
        w = self.world()
        pipeline(w, "select")
        (w.run / "threshold.json").write_text("{not json")
        self.assertFail(run_c2(w.args("score", "--set", "source_test,target")), "threshold.json missing or not valid JSON", "score")
        w = World(self.tmp / "gone")
        pipeline(w, "score_tests")
        (w.run / "threshold.json").unlink()
        self.assertFail(run_c2(w.args("apply")), "threshold.json is missing", "apply")

    def test_identifier_in_table_json(self):
        w = self.world()
        pipeline(w, "score_tests")
        leak = w.rows["source_test"][0]["key"].split("/", 1)[1]
        real = C2.cell
        with mock.patch.object(C2, "cell", lambda *a, **k: {**real(*a, **k), "note": leak}):
            self.assertFail(run_c2(w.args("apply")), "c2_table.json: identifiers in a public file", "apply")

    def test_config_messages_name_the_value(self):
        for edit, text in ((lambda c: c.update(schema="other"), "schema 'other' is not"), (lambda c: c.update(seed=7), "seed 7 is not 42"),
                           (lambda c: c["models"][1].update(arch="AttackNetV1"), "model architectures ['LivenessNet', 'AttackNetV1'] are not"),
                           (lambda c: c["manifests"]["source_train"].update(name="casia_fasd/pooled_train"), "manifests.source_train.name 'casia_fasd/pooled_train' is not"),
                           (lambda c: c["derived_frames"]["source"].update(file="x.txt"), "derived_frames.source.file 'x.txt' is not")):
            self.assertIn(text, self.problems(edit))

    def test_restored_epoch_message(self):
        w = self.world()
        pipeline(w, "train")
        self.card_edit(w, "LivenessNet", lambda d: d.update(restored_epoch=5))
        self.assertFail(run_c2(w.args("score", "--set", "dev")), "restored epoch 5 is not the epoch of minimum val_loss (3)", "score")

    def test_subject_counts_per_class(self):
        import numpy as np
        rows = [{"label": "0", "subject_key": "a", "video_key": "a1"}, {"label": "1", "subject_key": "b", "video_key": "b1"},
                {"label": "1", "subject_key": "c", "video_key": "c1"}]
        c = C2.counts_of("casia_fasd", rows)
        self.assertEqual((c["n_subjects_bona_fide"], c["n_subjects_attack"], c["n_subjects"]), (1, 2, 3))
        self.assertEqual(C2.subjects(np.array([0, 1, 1]), ["a", "b", "c"]), {"n_subjects_bona_fide": 1, "n_subjects_attack": 2})

    def test_corrupt_record(self):
        w = self.world()
        pipeline(w, "train")
        (w.run / "train.json").write_text("{not json")
        self.assertFail(run_c2(w.args("score", "--set", "dev")), "train.json is not valid JSON", "score")

    def test_system_exit_inside_a_command(self):
        w = self.world()
        for code, want in ((1, 1), (2, 1)):
            with mock.patch.dict(C2.COMMANDS, {"check": lambda a, c=code: sys.exit(c)}):
                try:
                    rc = run_c2(w.args("check"))[0]
                except SystemExit:
                    rc = "SystemExit propagated"
            self.assertEqual(rc, want, code)
        with mock.patch.dict(C2.COMMANDS, {"check": lambda a: sys.exit(0)}):
            with self.assertRaises(SystemExit) as cm:
                run_c2(w.args("check"))
        self.assertEqual(cm.exception.code, 0)

    def test_apply_reads_no_cache(self):
        w = self.world()
        pipeline(w, "score_tests")
        opened = []

        def hook(ev, args):
            if ev == "open" and args and isinstance(args[0], (str, os.PathLike)) and str(TMP_ROOT) in os.fsdecode(args[0]):
                opened.append(Path(os.fsdecode(args[0])).name)
        HOOKS.append(hook)
        try:
            self.assertPass(run_c2(w.args("apply")))
        finally:
            HOOKS.remove(hook)
        self.assertFalse([n for n in opened if n.endswith(".npz")], opened)
        self.assertTrue({"raw_scores_dev.csv", "raw_scores_source_test.csv", "raw_scores_target.csv", "threshold.json"} <= set(opened))

    def test_trial_accepts_unapproved_config_and_dirty_tree(self):
        w = self.world()
        with mock.patch.object(C2, "REPO_STATE", lambda run: {"commit": "1234567" + "0" * 33, "dirty": True, "dirty_excludes": None}):
            self.assertPass(run_c2(w.args("check")))
            self.assertPass(run_c2(w.args("train")))
        doc = json.loads((w.run / "check.json").read_text())
        self.assertEqual((doc["trial"], doc["approved_config"], doc["dirty"]), (True, False, True))

    def test_cli_options_and_defaults(self):
        ap = C2.parser()
        sub = next(a for a in ap._actions if isinstance(a, argparse._SubParsersAction))
        self.assertEqual(sorted(sub.choices), ["all", "apply", "check", "score", "select", "train"])
        common = {"config": (True, None, None), "run_dir": (True, None, None),
                  "source_manifests": (False, str(C2.RESULTS / "m2_rgb" / "casia_fasd" / "manifests"), None),
                  "target_manifests": (False, str(C2.RESULTS / "m2" / "manifests"), None),
                  "source_cache": (False, str(C2.DATA / "processed" / "casia_fasd_1500.npz"), None),
                  "target_cache": (False, str(C2.DATA / "processed" / "nuaa_1500.npz"), None)}
        extra = {"train": {"epochs": (True, None, int), "i_am_authorized": (False, False, None)},
                 "all": {"epochs": (True, None, int), "i_am_authorized": (False, False, None)},
                 "score": {"i_am_authorized": (False, False, None), "set": (True, None, None)}}
        for name, p in sub.choices.items():
            got = {a.dest: (a.required, a.default, a.type) for a in p._actions if a.dest != "help"}
            self.assertEqual(got, {**common, **extra.get(name, {})}, name)
        for argv in (["train", "--config", "c", "--run-dir", "r"], ["score", "--config", "c", "--run-dir", "r"], ["check", "--config", "c"]):
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
                ap.parse_args(argv)
            self.assertEqual(cm.exception.code, 2, argv)

    def test_five_subjects_split_by_subject(self):
        import numpy as np
        w = self.world()
        a = w.arrays["source"]
        s5 = np.array([str((int(x) - 1) % 5 + 1) for x in a["s_train"]])
        data = {"X_train": a["X_train"], "y_train": a["y_train"], "s_train": s5}
        with contextlib.redirect_stdout(io.StringIO()):
            info = fake_trainer()("LivenessNet", "casia_fasd", data, 12, 16, 1e-3, out_dir=self.tmp / "o5", test_eval=False)
        self.assertEqual(info["val_split"], "per soggetto")


def tree_bytes(d: Path) -> dict:
    """Contenuto di un albero (percorso relativo -> byte; collegamenti come bersaglio), senza seguire i collegamenti."""
    out = {}
    for dp, dns, fns in os.walk(d):
        for n in fns + [x for x in dns if (Path(dp) / x).is_symlink()]:
            q = Path(dp) / n
            out[str(q.relative_to(d))] = ("link", os.readlink(q)) if q.is_symlink() else q.read_bytes()
    return out


class TestTeacherReview21(Base):
    """Revisione del docente del 21/10 sulla PR #8: ingressi e uscite separati, alberi con i pacchetti accettati,
    temporanei sicuri, opzioni prima del lavoro, provenienza legata al run corrente."""

    def inputs_snapshot(self, w):
        return {"src": tree_bytes(w.src_dir), "tgt": tree_bytes(w.tgt_dir), "cache": tree_bytes(w.cache_dir), "cfg": w.config.read_bytes()}

    def moved(self, w, where: Path, which="source"):
        """Copia della cartella dei manifest (source o target) in `where` e argomenti che la usano."""
        src = w.src_dir if which == "source" else w.tgt_dir
        where.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, where)
        return where

    def test_inputs_inside_or_around_the_run_refused(self):
        w = self.world()
        pipeline(w, "train")
        run_before = tree_bytes(w.run)
        cases = {"inside checkpoints": w.run / "checkpoints" / "m", "inside the run": w.run / "m", "equal to the run": None}
        for name, where in cases.items():
            if where is None:
                a = w.args("check")
                a[a.index("--source-manifests") + 1] = str(w.run)
            else:
                self.moved(w, where)
                a = w.args("check")
                a[a.index("--source-manifests") + 1] = str(where)
            before = tree_bytes(w.run)
            for cmd in (a, a[:1] and [a[0].replace("check", "train")] + a[1:] + ["--epochs", "12", "--i-am-authorized"]):
                self.assertFail(run_c2(cmd), "inputs refused", None)
                self.assertEqual(tree_bytes(w.run), before, name)          # nulla cancellato né scritto, nemmeno status.json
            if where is not None:
                shutil.rmtree(where)
        self.assertEqual(tree_bytes(w.run), run_before)

    def test_inputs_by_link_alias_config_cache_and_ancestor(self):
        w = self.world()
        pipeline(w, "check")
        snap = self.inputs_snapshot(w)
        run_before = tree_bytes(w.run)
        link = self.tmp / "link_to_run"
        link.symlink_to(w.run, target_is_directory=True)
        (w.run / "m").mkdir()
        for f in w.src_dir.iterdir():
            shutil.copy2(f, w.run / "m" / f.name)
        run_before = tree_bytes(w.run)
        for opt, val in (("--source-manifests", link / "m"), ("--target-manifests", link), ("--config", w.run / "check.json"),
                         ("--source-cache", w.run / "check.json"), ("--source-manifests", w.run.parent)):
            a = w.args("check")
            a[a.index(opt) + 1] = str(val)
            self.assertFail(run_c2(a), "inputs refused", "check")
            self.assertEqual(tree_bytes(w.run), run_before, opt)
        # alias simulato (stessa identità per os.path.samefile, percorso diverso)
        alias = self.tmp / "alias_dir"
        alias.mkdir()
        real = os.path.samefile

        def same(x, y):
            return {Path(x).resolve(), Path(y).resolve()} == {alias.resolve(), w.run.resolve()} or real(x, y)
        a = w.args("check")
        a[a.index("--target-manifests") + 1] = str(alias)
        with mock.patch("os.path.samefile", same):
            self.assertFail(run_c2(a), "inputs refused", "check")
        self.assertEqual(tree_bytes(w.run), run_before)
        self.assertEqual(self.inputs_snapshot(w), snap)

    def fork_layout(self):
        f = self.tmp / "fork"
        sentinel = f / "research" / "c1-livenesslab" / "livenesslab" / "models" / "weights" / "LivenessNet__casia_fasd.h5"
        sentinel.parent.mkdir(parents=True)
        sentinel.write_bytes(b"accepted C1 checkpoint")
        for pkg in ("m1-metrics/livenesslab/scripts", "m2-rgb-manifests/livenesslab/scripts", "m2-c2-pilot-protocol", "m2-c2-pilot-impl/livenesslab/scripts"):
            (f / "research" / pkg).mkdir(parents=True, exist_ok=True)
        (f / "research" / "m1-metrics" / "config.json").write_text("{}")
        return f, f / "research" / "m2-c2-pilot-impl" / "livenesslab", sentinel

    def test_group_repository_layout_protected(self):
        f, root, sentinel = self.fork_layout()
        before = tree_bytes(f)
        C2._outer_roots.cache_clear()
        with mock.patch.object(C2, "ROOT", root), mock.patch.object(TC, "ROOT", root):
            for where in ("research/c1-livenesslab/livenesslab/models/weights", "research/m1-metrics", "research/m2-rgb-manifests/livenesslab",
                          "research/m2-c2-pilot-protocol", "research/m2-c2-pilot-impl", "research/m2-c2-pilot-impl/livenesslab/results/c2", ".", "other"):
                self.assertIn("work tree", C2.run_dir_problem(f / where / TRIAL), where)
                self.assertIn("work tree", TC.out_dir_problem(f / where), where)
            self.assertEqual(C2.run_dir_problem(self.tmp / "outside" / TRIAL), "")
            self.assertEqual(TC.out_dir_problem(self.tmp / "outside" / "ck"), "")
            w = self.world()
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stdout(io.StringIO()):
                fake_trainer()("LivenessNet", "casia_fasd", {"X_train": w.arrays["source"]["X_train"], "y_train": w.arrays["source"]["y_train"],
                                                             "s_train": w.arrays["source"]["s_train"]}, 12, 16, 1e-3,
                               out_dir=sentinel.parent, test_eval=False)
            self.assertIn("work tree", str(cm.exception.code))
            self.assertFail(run_c2(w.args("check", run=f / "research" / "m1-metrics" / TRIAL)), "run folder refused", "check")
        C2._outer_roots.cache_clear()
        self.assertEqual(tree_bytes(f), before)
        self.assertEqual(sentinel.read_bytes(), b"accepted C1 checkpoint")

    def test_git_work_tree_protected(self):
        if shutil.which("git") is None:
            skip_missing(self, "git")
        g = self.tmp / "gitrepo"
        root = g / "pkg" / "livenesslab"
        (root / "scripts").mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(g)], check=True, capture_output=True)
        (g / "keep.txt").write_text("x")
        C2._outer_roots.cache_clear()
        with mock.patch.object(C2, "ROOT", root), mock.patch.object(TC, "ROOT", root):
            self.assertIn("work tree", C2.run_dir_problem(g / "anything" / TRIAL))
            self.assertIn("work tree", TC.out_dir_problem(g / "pkg" / "out"))
            self.assertEqual(C2.run_dir_problem(self.tmp / "free" / TRIAL), "")
        C2._outer_roots.cache_clear()
        self.assertEqual(C2.run_dir_problem(self.tmp / "free" / TRIAL), "")        # ROOT vero: la sua radice non è un albero esterno

    def test_run_folder_symlink_refused(self):
        w = self.world()
        real = self.tmp / "real_run"
        real.mkdir()
        w.run.symlink_to(real, target_is_directory=True)
        self.assertFail(run_c2(w.args("check")), "symbolic link", "check")
        self.assertEqual(os.listdir(real), [])

    def test_temporary_and_output_links_never_followed(self):
        ext = self.tmp / "external.txt"
        ext.write_text("external, must stay")
        ext_dir = self.tmp / "external_dir"
        ext_dir.mkdir()
        (ext_dir / "keep.h5").write_bytes(b"keep")
        w = self.world()
        w.run.mkdir()
        names = [n for s_ in C2.STEPS for n in C2.OUTPUTS[s_]] + [C2.STATUS]
        for n in names:
            (w.run / (n + ".tmp")).symlink_to(ext)
        (w.run / "status.json.tmp").unlink()
        (w.run / "status.json.tmp").symlink_to(self.tmp / "missing_target")      # collegamento senza bersaglio
        (w.run / "notes.tmp").write_text("a file of the user, not a temporary of this code")
        (w.run / ".check.json.old.tmp").write_text("stale new-style temp")
        (w.run / "checkpoints").symlink_to(ext_dir, target_is_directory=True)
        (w.run / "checkpoints.tmp").symlink_to(ext_dir, target_is_directory=True)
        pipeline(w, "apply")                                                 # PASS con i collegamenti presenti
        self.assertEqual(ext.read_text(), "external, must stay")
        self.assertEqual(tree_bytes(ext_dir), {"keep.h5": b"keep"})
        self.assertEqual([e for e in os.listdir(w.run) if e.endswith(".tmp") or e.endswith(".tmp/")], ["notes.tmp"], os.listdir(w.run))
        self.assertFalse((w.run / "status.json.tmp").is_symlink())
        self.assertEqual((w.run / "notes.tmp").read_text(), "a file of the user, not a temporary of this code")
        self.assertFalse((w.run / "checkpoints").is_symlink())
        # FAIL con un'uscita e status.json come collegamenti
        (w.run / "threshold.json.tmp").symlink_to(ext)
        rewrite_scores(w, "raw_scores_target.csv", lambda rows: rows[:-1], record="score_tests.json")
        (w.run / "scores_E1.csv").unlink()
        (w.run / "scores_E1.csv").symlink_to(ext)
        self.assertFail(run_c2(w.args("apply")), step="apply")
        self.assertEqual(ext.read_text(), "external, must stay")
        self.assertFalse((w.run / "scores_E1.csv").exists() or (w.run / "scores_E1.csv").is_symlink())
        (w.run / "status.json").unlink()
        (w.run / "status.json").symlink_to(ext)
        self.assertFail(run_c2(w.args("apply")), "status.json in the run folder is not a regular file", "apply")
        self.assertEqual(ext.read_text(), "external, must stay")

    def test_write_all_refuses_link_targets_and_uses_new_temporaries(self):
        ext = self.tmp / "ext.txt"
        ext.write_text("x")
        d = self.tmp / "out"
        d.mkdir()
        (d / "a.json").symlink_to(ext)
        with self.assertRaises(C2.C2Error):
            C2.write_all(d, {"a.json": "new"})
        self.assertEqual(ext.read_text(), "x")
        calls = []
        real = C2.tempfile.mkstemp

        def spy(**k):
            calls.append(k)
            return real(**k)
        with mock.patch.object(C2.tempfile, "mkstemp", spy):
            C2.write_all(d, {"b.json": "ok"})
        self.assertEqual((d / "b.json").read_text(), "ok")
        self.assertEqual([(c["dir"], c["suffix"]) for c in calls], [(d, ".tmp")])
        self.assertEqual(sorted(os.listdir(d)), ["a.json", "b.json"])

    def test_options_validated_before_reading_images(self):
        f, root, sentinel = self.fork_layout()
        calls = []
        bad = (["--dataset", "nuaa", "--no-test-eval"], ["--dataset", "nuaa", "--epochs", "0"], ["--dataset", "nuaa", "--arch", "Foo"],
               ["--dataset", "nuaa", "--batch", "0"], ["--dataset", "nuaa", "--lr", "0"], ["--dataset", "nuaa", "--per-class", "-1"],
               ["--dataset", "nuaa", "--out-dir", str(sentinel.parent), "--no-test-eval"],
               ["--dataset", "nuaa", "--out-dir", str(sentinel), "--no-test-eval"])
        C2._outer_roots.cache_clear()
        with mock.patch.object(TC, "build_arrays", lambda *a, **k: calls.append(a) or {}), mock.patch.object(TC, "ROOT", root), \
                mock.patch.object(TC, "train", lambda *a, **k: calls.append(("train", a))):
            for argv in bad:
                with self.assertRaises(SystemExit) as cm, contextlib.redirect_stdout(io.StringIO()):
                    TC.main(argv)
                self.assertTrue(str(cm.exception.code).startswith("ERROR: "), argv)
            self.assertEqual(calls, [])
            with contextlib.redirect_stdout(io.StringIO()):
                TC.main(["--dataset", "nuaa", "--out-dir", str(self.tmp / "ok"), "--no-test-eval", "--arch", "LivenessNet"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(sentinel.read_bytes(), b"accepted C1 checkpoint")

    def copy_from(self, a_run, b_run, *names):
        for n in names:
            src, dst = a_run / n, b_run / n
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            elif dst.exists() or dst.is_symlink():
                dst.unlink()
            (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)

    def two_runs(self, upto_b):
        wa = World(self.tmp / "A", "20261021-C2-seed42-abcdef0-trial")
        pipeline(wa, "apply")
        wb = World(self.tmp / "B", "20261021-C2-seed42-abcdef1-trial")
        wb.cfg, wb.config.parent.mkdir(exist_ok=True)
        shutil.copy2(wa.config, wb.config)                                   # stessa configurazione, stessi ingressi
        for d in ("src_dir", "tgt_dir", "cache_dir"):
            shutil.rmtree(getattr(wb, d)); shutil.copytree(getattr(wa, d), getattr(wb, d))
        pipeline(wb, upto_b)
        return wa, wb

    def test_records_of_another_run_refused(self):
        cases = [("check", ("train.json", "checkpoints"), ("score", "--set", "dev"), "train.json belongs to another run"),
                 ("train", ("score_dev.json", "raw_scores_dev.csv"), ("select",), "score_dev.json belongs to another run"),
                 ("score_dev", ("threshold.json",), ("score", "--set", "source_test,target"), "threshold.json belongs to another run"),
                 ("select", ("score_tests.json", "raw_scores_source_test.csv", "raw_scores_target.csv"), ("apply",),
                  "score_tests.json belongs to another run")]
        for upto, names, cmd, text in cases:
            for d in ("A", "B"):
                shutil.rmtree(self.tmp / d, ignore_errors=True)
            wa, wb = self.two_runs(upto)
            if upto == "select":                                              # B fino al punteggio dei test, poi i record di A
                pipeline(wb, "score_tests")
            self.copy_from(wa.run, wb.run, *names)
            self.assertFail(run_c2(wb.args(*cmd)), text, cmd[0])
            if "belongs" in text and names[0] != "threshold.json":
                self.assertIn(f"({wa.run.name!r})", run_c2(wb.args(*cmd))[2])
            self.assertNotEqual(json.loads((wb.run / names[0]).read_text())["run_id"], wb.run.name)

    def test_inputs_refused_before_any_change(self):
        """Ingressi mancanti o del tipo sbagliato: rifiuto prima di accettare la cartella, nessun file scritto o tolto."""
        w = self.world()
        pipeline(w, "apply")
        before = tree_bytes(w.run)
        for cmd, opt, val, text in (("select", "--source-manifests", self.tmp / "nope", "--source-manifests nope is not a folder"),
                                    ("apply", "--config", self.tmp / "nope.json", "configuration not found: nope.json"),
                                    ("train", "--target-cache", self.tmp / "nuaa_1500.npz", "target cache nuaa_1500.npz not found"),
                                    ("check", "--source-cache", self.tmp / "casia_fasd_1500.npz", "source cache casia_fasd_1500.npz not found"),
                                    ("select", "--target-manifests", w.config, "--target-manifests config.json is not a folder")):
            a = w.args(cmd)
            a[a.index(opt) + 1] = str(val)
            self.assertFail(run_c2(a), text, cmd)
            self.assertEqual(tree_bytes(w.run), before, cmd)          # status.json compreso: ancora PASS

    def test_check_configuration_refused_before_any_change(self):
        w = self.world()
        pipeline(w, "apply")
        before = {n: (w.run / n).read_bytes() for n in os.listdir(w.run) if (w.run / n).is_file() and n != "status.json"}
        w.config.write_text("{not json")
        self.assertFail(run_c2(w.args("check")), "not valid JSON", "check")
        after = {n: (w.run / n).read_bytes() for n in os.listdir(w.run) if (w.run / n).is_file() and n != "status.json"}
        self.assertEqual(after, before)
        self.assertTrue(json.loads((w.run / "status.json").read_text())["outputs"].startswith("untouched"))

    def test_output_link_to_a_folder_removed_not_followed(self):
        ext_dir = self.tmp / "ext_folder"
        ext_dir.mkdir()
        (ext_dir / "keep.txt").write_text("keep")
        w = self.world()
        w.run.mkdir()
        (w.run / "check.json").symlink_to(ext_dir, target_is_directory=True)
        self.assertPass(run_c2(w.args("check")))
        self.assertFalse((w.run / "check.json").is_symlink())
        self.assertEqual(tree_bytes(ext_dir), {"keep.txt": b"keep"})

    def test_out_dir_link_or_file_refused(self):
        target = self.tmp / "real_out"
        target.mkdir()
        link = self.tmp / "link_out"
        link.symlink_to(target, target_is_directory=True)
        f = self.tmp / "a_file"
        f.write_text("x")
        self.assertIn("is a symbolic link", TC.out_dir_problem(link))
        self.assertIn("exists and is not a folder", TC.out_dir_problem(f))
        self.assertEqual(TC.out_dir_problem(target), "")

    def test_output_folder_with_an_output_name_refused(self):
        w = self.world()
        pipeline(w, "score_dev")
        (w.run / "threshold.json").mkdir()
        self.assertFail(run_c2(w.args("select")), "threshold.json in the run folder is a folder, not an output file: refused", "select")
        self.assertTrue((w.run / "threshold.json").is_dir())

    def test_output_file_mode(self):
        w = self.world()
        pipeline(w, "check")
        self.assertEqual((w.run / "check.json").stat().st_mode & 0o777, 0o666 & ~C2._UMASK)

    def test_validate_args_limits_accepted(self):
        ns = types.SimpleNamespace(arch=["LivenessNet"], epochs=1, batch=1, lr=1e-3, per_class=0, no_test_eval=False, out_dir=None,
                                   dataset="casia_fasd", protocol="official")
        self.assertEqual(TC.validate_args(ns), "")
        for field, value in (("epochs", 0), ("batch", 0), ("per_class", -1), ("lr", 0.0), ("arch", [])):
            self.assertTrue(TC.validate_args(types.SimpleNamespace(**{**vars(ns), field: value})), field)

    def test_out_dir_with_existing_outputs_refused(self):
        """--out-dir con pesi, file Keras o scheda già presenti (collegamento a un file esterno, collegamento senza
        bersaglio, file regolare): rifiuto prima di qualunque training, file esterno identico, nessuna sovrascrittura."""
        import numpy as np
        ext = self.tmp / "external.h5"
        ext.write_bytes(b"external weights")
        w = self.world()
        data = {k: w.arrays["source"][k] for k in ("X_train", "y_train", "s_train")}
        for name, make in (("link", lambda d: (d / "LivenessNet__casia_fasd.h5").symlink_to(ext)),
                           ("dangling", lambda d: (d / "LivenessNet__casia_fasd.keras").symlink_to(self.tmp / "nothing_here")),
                           ("regular", lambda d: (d / "LivenessNet__casia_fasd.json").write_text("existing card"))):
            out = self.tmp / f"out_{name}"
            out.mkdir()
            make(out)
            before = tree_bytes(out)
            FakeModel.LOG.clear()
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stdout(io.StringIO()):
                fake_trainer()("LivenessNet", "casia_fasd", data, 12, 16, 1e-3, out_dir=out, test_eval=False)
            self.assertIn("already holds", str(cm.exception.code), name)
            self.assertEqual(FakeModel.LOG, [], name)
            self.assertEqual(tree_bytes(out), before, name)
            calls = []
            with mock.patch.object(TC, "build_arrays", lambda *a, **k: calls.append(a) or {}), self.assertRaises(SystemExit) as cm, \
                    contextlib.redirect_stdout(io.StringIO()):
                TC.main(["--dataset", "casia_fasd", "--arch", "LivenessNet", "--out-dir", str(out), "--no-test-eval"])
            self.assertIn("already holds", str(cm.exception.code), name)
            self.assertEqual(calls, [], name)
        self.assertEqual(ext.read_bytes(), b"external weights")
        # cartella con i file di un'altra architettura: ammessa
        out = self.tmp / "out_other"
        out.mkdir()
        (out / "AttackNetV2_2__casia_fasd.h5").write_bytes(b"other model")
        self.assertEqual(TC.existing_outputs_problem(out, "LivenessNet", "casia_fasd", "official"), "")

    def test_validate_args_refuses_non_finite_lr(self):
        ns = types.SimpleNamespace(arch=["LivenessNet"], epochs=1, batch=1, lr=1e-3, per_class=0, no_test_eval=False, out_dir=None,
                                   dataset="casia_fasd", protocol="official")
        self.assertEqual(TC.validate_args(ns), "")
        for v in (float("nan"), float("inf")):
            self.assertIn("finite", TC.validate_args(types.SimpleNamespace(**{**vars(ns), "lr": v})), v)

    def test_forged_records_refused(self):
        cases = (("train.json", "check_sha256", "was not written after the check of this run"), ("train.json", "step", "records the step 'check'"),
                 ("train.json", "commit", "was written at another commit than the check of this run"),
                 ("score_dev.json", "train_sha256", "score_dev.json was written with another train.json"),
                 ("score_tests.json", "score_dev_sha256", "score_tests.json was written with another score_dev.json"))
        for i, (rec, field, text) in enumerate(cases):
            w = World(self.tmp / f"forged{i}")
            pipeline(w, "score_tests")
            r = json.loads((w.run / rec).read_text())
            (w.run / rec).write_text(json.dumps(dict(r, **{field: "check" if field == "step" else "1234567" if field == "commit" else "0" * 64})))
            self.assertFail(run_c2(w.args("apply")), text, "apply")

    def test_linked_checkpoints_and_records_refused(self):
        w = self.world()
        pipeline(w, "train")
        other = self.tmp / "other_ck"
        shutil.copytree(w.run / "checkpoints", other)
        shutil.rmtree(w.run / "checkpoints")
        (w.run / "checkpoints").symlink_to(other, target_is_directory=True)
        self.assertFail(run_c2(w.args("score", "--set", "dev")), "is a symbolic link", "score")
        self.assertTrue((other / "LivenessNet__casia_fasd.h5").exists())
        w2 = World(self.tmp / "w2")
        pipeline(w2, "train")
        keep = self.tmp / "train_copy.json"
        shutil.copy2(w2.run / "train.json", keep)
        (w2.run / "train.json").unlink()
        (w2.run / "train.json").symlink_to(keep)
        self.assertFail(run_c2(w2.args("score", "--set", "dev")), "missing train.json", "score")


class TestTrainingDevice(Base):
    """Dispositivo del training (richiesta del 27/10/2026): LivenessNet solo su CPU dentro tf.device("/CPU:0"), senza
    nascondere la GPU; AttackNet sul dispositivo predefinito; verifica delle variabili e dello scarto grafo/eager con
    arresto prima di salvare; registrazione nella scheda, nel manifest e in train.json; controllo prima di ogni punteggio."""

    def fresh(self) -> World:
        """Un mondo sintetico nuovo per ogni uso (più usi nello stesso test)."""
        self._n = getattr(self, "_n", 0) + 1
        return World(self.tmp / f"w{self._n}", TRIAL)

    def data(self):
        a = self.fresh().arrays["source"]
        return {"X_train": a["X_train"], "y_train": a["y_train"], "s_train": a["s_train"], "X_test": a["X_test"], "y_test": a["y_test"]}

    def train_all(self, devs, archs=("LivenessNet", "AttackNetV2_2"), test_eval=False):
        data = self.data()
        if not test_eval:
            del data["X_test"], data["y_test"]
        out = self.tmp / f"out{getattr(self, '_n', 0)}"
        FakeModel.DEVLOG.clear()
        cards = {}
        with contextlib.redirect_stdout(io.StringIO()) as o:
            for arch in archs:
                cards[arch] = fake_trainer(devs=devs)(arch, "casia_fasd", data, 12, 16, 1e-3, out_dir=out, test_eval=test_eval)
        return out, cards, o.getvalue()

    def events(self, arch):
        return [(e, d) for e, a, d in FakeModel.DEVLOG if a == arch]

    @staticmethod
    def record(requested, dev, visible, gap, physical=None):
        return {"requested": requested, "expected": dev, "effective": dev, "visible": visible, "physical": physical or visible,
                "graph_eager_max_rel_diff": gap, "graph_eager_layer": "output" if gap else None}

    def test_livenessnet_on_cpu_attacknet_on_the_default_device(self):
        """Mac con GPU visibile: ogni operazione di LivenessNet (creazione, compile, fit con validazione, ripristino,
        chiamate in grafo ed eager del controllo, salvataggio) dentro l'ambito CPU; AttackNet fuori da ogni ambito, sulla
        GPU; nessuna disattivazione globale delle GPU e nessun ambito lasciato aperto."""
        devs = FakeDevices(gpu=True)
        out, cards, log = self.train_all(devs)
        ev = self.events("LivenessNet")
        self.assertEqual([e for e, _ in ev][:3], ["create", "compile", "fit"])
        self.assertTrue({"graph_call", "eager_call", "save", "save_weights"} <= {e for e, _ in ev})
        self.assertTrue(all(d == "CPU:0" for _, d in ev), ev)
        self.assertTrue(all(d == "" for _, d in self.events("AttackNetV2_2")), self.events("AttackNetV2_2"))
        self.assertFalse(any(e == "set_visible_devices" for e, _, _ in FakeModel.DEVLOG))
        self.assertEqual(devs.stack, [])
        self.assertEqual(cards["LivenessNet"]["training_device"], self.record("CPU", "CPU:0", ["CPU", "GPU"], 0.0))
        self.assertEqual(cards["AttackNetV2_2"]["training_device"], self.record("default", "GPU:0", ["CPU", "GPU"], 2.0 ** -18))
        for arch in ("LivenessNet", "AttackNetV2_2"):
            card = json.loads((out / f"{arch}__casia_fasd.json").read_text())
            self.assertEqual(card["training_device"], cards[arch]["training_device"])
        self.assertIn("richiesto CPU, effettivo CPU:0", log)
        self.assertIn("richiesto default, effettivo GPU:0", log)

    def test_attacknet_first_then_livenessnet(self):
        """Ordine inverso nello stesso processo: AttackNet resta sulla GPU senza ambito, LivenessNet sulla CPU."""
        devs = FakeDevices(gpu=True)
        _, cards, _ = self.train_all(devs, ("AttackNetV2_2", "LivenessNet"))
        self.assertEqual(cards["AttackNetV2_2"]["training_device"]["effective"], "GPU:0")
        self.assertEqual(cards["LivenessNet"]["training_device"]["effective"], "CPU:0")
        self.assertTrue(all(d == "" for _, d in self.events("AttackNetV2_2")))
        self.assertTrue(all(d == "CPU:0" for _, d in self.events("LivenessNet")))
        self.assertEqual(devs.stack, [])

    def test_test_evaluation_inside_the_cpu_scope(self):
        """Percorso predefinito con valutazione del test: anche predict di LivenessNet avviene dentro l'ambito CPU."""
        self.train_all(FakeDevices(gpu=True), test_eval=True)
        self.assertIn(("predict", "CPU:0"), self.events("LivenessNet"))
        self.assertIn(("predict", ""), self.events("AttackNetV2_2"))

    def test_without_gpu(self):
        """Senza GPU (Linux, contenitore): entrambe sulla CPU, AttackNet senza ambito, atteso CPU:0."""
        _, cards, _ = self.train_all(FakeDevices(gpu=False))
        self.assertEqual(cards["LivenessNet"]["training_device"], self.record("CPU", "CPU:0", ["CPU"], 0.0))
        self.assertEqual(cards["AttackNetV2_2"]["training_device"], self.record("default", "CPU:0", ["CPU"], 0.0))
        self.assertTrue(all(d == "" for _, d in self.events("AttackNetV2_2")))

    def assertStopped(self, devs, arch, text, fitted):
        data = self.data()
        out = self.tmp / f"stopped_{arch}"
        stack = list(devs.stack)
        FakeModel.DEVLOG.clear()
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stdout(io.StringIO()):
            fake_trainer(devs=devs)(arch, "casia_fasd", data, 12, 16, 1e-3, out_dir=out, test_eval=False)
        self.assertTrue(str(cm.exception.code).startswith("ERROR: "), cm.exception.code)
        self.assertIn(text, str(cm.exception.code))
        self.assertEqual("fit" in [e for e, _ in self.events(arch)], fitted)
        self.assertFalse(out.exists())                                  # nessun peso, file Keras o scheda
        self.assertEqual(devs.stack, stack)                             # ambito chiuso anche sull'errore
        return str(cm.exception.code)

    def test_cpu_not_guaranteed_stops_before_fit(self):
        """Ambito CPU ignorato nel posizionamento (variabili create sulla GPU): errore prima del fit, nessun file."""
        msg = self.assertStopped(FakeDevices(gpu=True, ignore_scope=True), "LivenessNet", "CPU execution not guaranteed", fitted=False)
        self.assertIn("requires every variable on CPU:0, found ['GPU:0']", msg)

    def test_compute_on_gpu_with_variables_on_cpu_stops(self):
        """Variabili sulla CPU ma calcolo sulla GPU (il caso che il solo posizionamento non vede): lo scarto grafo/eager
        del difetto ferma il training dopo il fit, prima di salvare."""
        msg = self.assertStopped(FakeDevices(gpu=True, compute_on="GPU:0"), "LivenessNet", "graph-mode outputs differ from eager by 2.000e+00",
                                 fitted=True)
        self.assertIn("CPU execution not guaranteed", msg)

    def test_keras_functions_on_gpu_stop(self):
        """Funzioni di Keras (predict, evaluate: test e validazione) sulla GPU mentre una tf.function nuova resta
        nell'ambito CPU: le uscite degli strati coincidono, predict ed evaluate no; il training si ferma."""
        msg = self.assertStopped(FakeDevices(gpu=True, keras_on="GPU:0"), "LivenessNet", "graph-mode outputs differ from eager by 2.000e+00",
                                 fitted=True)
        self.assertIn("CPU execution not guaranteed", msg)
        self.assertIn(("predict", "LivenessNet", 4), FakeModel.LOG)
        for path in ("predict", "evaluate"):            # ciascuna delle due funzioni da sola (test, validazione di fit)
            self.assertStopped(FakeDevices(gpu=True, keras_on="GPU:0", keras_paths=(path,)), "LivenessNet", "graph-mode outputs differ", fitted=True)

    def test_not_finite_comparison_stops(self):
        """Scarto grafo/eager NaN o infinito: training fermato prima di salvare per ogni architettura, anche per la
        politica "default" di AttackNet (la tolleranza resta solo di LivenessNet)."""
        for arch in ("LivenessNet", "AttackNetV2_2"):
            for gap in (float("nan"), float("inf")):
                with self.subTest(arch=arch, gap=gap):
                    msg = self.assertStopped(FakeDevices(gpu=True, gap=gap), arch, f"graph/eager comparison not finite ({gap})", fitted=True)
                    self.assertTrue(msg.startswith(f"ERROR: {arch}: "), msg)
                    self.assertNotIn("CPU execution not guaranteed", msg)

    def test_not_measured_comparison_stops(self):
        """Dopo il fit lo scarto grafo/eager deve essere un numero: None (non misurato) ferma il training prima di salvare."""
        for arch in ("LivenessNet", "AttackNetV2_2"):
            with self.subTest(arch=arch), mock.patch.object(TC, "graph_eager_gap", return_value=(None, None)):
                self.assertStopped(FakeDevices(gpu=True), arch, f"ERROR: {arch}: graph/eager comparison not measured after training (None)", fitted=True)

    def test_not_finite_validation_loss_stops(self):
        """val_loss NaN o infinito in una qualunque epoca: training fermato prima di salvare, per ogni architettura."""
        for arch in ("LivenessNet", "AttackNetV2_2"):
            for bad in (float("nan"), float("inf")):
                with self.subTest(arch=arch, bad=bad):
                    out = self.tmp / f"val_loss_{arch}_{bad}"
                    with self.assertRaises(SystemExit) as cm, contextlib.redirect_stdout(io.StringIO()):
                        fake_trainer(plan=lambda e, b=bad: [0.6, b, 0.4], devs=FakeDevices(gpu=True))(
                            arch, "casia_fasd", self.data(), 12, 16, 1e-3, out_dir=out, test_eval=False)
                    self.assertIn(f"ERROR: {arch}: validation loss not finite after training", str(cm.exception.code))
                    self.assertFalse(out.exists())                      # nessun peso, file Keras o scheda

    def test_gap_at_the_tolerance(self):
        """Scarto uguale alla tolleranza accettato; appena sopra rifiutato."""
        _, cards, _ = self.train_all(FakeDevices(gpu=True, gap=TC.GRAPH_EAGER_TOL), ("LivenessNet",))
        self.assertEqual(cards["LivenessNet"]["training_device"]["graph_eager_max_rel_diff"], TC.GRAPH_EAGER_TOL)
        self.assertStopped(FakeDevices(gpu=True, gap=TC.GRAPH_EAGER_TOL * 1.01), "LivenessNet", "above 1e-05", fitted=True)

    def test_attacknet_gap_recorded_not_enforced(self):
        """Per la politica "default" lo scarto è solo registrato (AttackNet non è colpita dal difetto)."""
        _, cards, _ = self.train_all(FakeDevices(gpu=True, gap=0.25), ("AttackNetV2_2",))
        self.assertEqual(cards["AttackNetV2_2"]["training_device"]["graph_eager_max_rel_diff"], 0.25)

    def test_optimizer_variables_off_cpu_stop_before_saving(self):
        msg = self.assertStopped(FakeDevices(gpu=True, optimizer_on="GPU:0"), "LivenessNet", "CPU execution not guaranteed", fitted=True)
        self.assertIn("found ['CPU:0', 'GPU:0']", msg)

    def test_attacknet_policy_verified(self):
        """Un ambito CPU rimasto aperto sposterebbe AttackNet dalla GPU: rilevato e fermato (politica "default" verificata)."""
        devs = FakeDevices(gpu=True)
        devs.stack.append("CPU:0")
        msg = self.assertStopped(devs, "AttackNetV2_2", "device policy 'default' requires every variable on GPU:0, found ['CPU:0']", fitted=False)
        self.assertNotIn("CPU execution not guaranteed", msg)
        self.assertStopped(FakeDevices(gpu=True, optimizer_on="CPU:0"), "AttackNetV2_2", "found ['CPU:0', 'GPU:0']", fitted=True)

    def test_hidden_gpu_refused(self):
        """Una GPU presente ma nascosta al processo (AttackNet passerebbe in silenzio sulla CPU) è rifiutata per ogni
        architettura, prima di creare il modello."""
        for arch in ("AttackNetV2_2", "LivenessNet"):
            msg = self.assertStopped(FakeDevices(gpu=False, hidden_gpu=True), arch, "a GPU is present but hidden from the process", fitted=False)
            self.assertNotIn(("create", arch, ""), FakeModel.DEVLOG, msg)

    def test_policy_table(self):
        self.assertEqual(TC.DEVICE_POLICY, {"LivenessNet": "CPU", "AttackNetV1": "default", "AttackNetV2_1": "default", "AttackNetV2_2": "default"})
        self.assertEqual(sorted(TC.DEVICE_POLICY), sorted(TC.ARCHS))
        self.assertEqual(C2.TRAINING_DEVICE, {a: TC.DEVICE_POLICY[a] for a in C2.ARCHS})
        self.assertEqual((C2.GRAPH_EAGER_TOL, TC.GRAPH_EAGER_TOL), (1e-5, 1e-5))
        with mock.patch.dict(TC.DEVICE_POLICY, {}, clear=True), self.assertRaises(SystemExit) as cm:
            fake_trainer()("LivenessNet", "casia_fasd", self.data(), 12, 16, 1e-3, out_dir=self.tmp / "o", test_eval=False)
        self.assertIn("no device policy for 'LivenessNet'", str(cm.exception.code))
        with mock.patch.dict(TC.DEVICE_POLICY, {"LivenessNet": "GPU"}), self.assertRaises(SystemExit) as cm:
            fake_trainer()("LivenessNet", "casia_fasd", self.data(), 12, 16, 1e-3, out_dir=self.tmp / "o", test_eval=False)
        self.assertIn("unknown device policy 'GPU'", str(cm.exception.code))
        self.assertFalse((self.tmp / "o").exists())
        self.assertEqual(TC.short_device("/job:localhost/replica:0/task:0/device:CPU:0"), "CPU:0")
        self.assertEqual(TC.short_device(""), "")

    def test_train_record_holds_the_device(self):
        w = self.fresh()
        pipeline(w, "train")
        rec = json.loads((w.run / "train.json").read_text())
        for m in w.cfg["models"]:
            card = json.loads((w.run / "checkpoints" / rec["models"][m["id"]]["card_file"]).read_text())
            self.assertEqual(rec["models"][m["id"]]["training_device"], card["training_device"])
            self.assertEqual(card["training_device"]["requested"], C2.TRAINING_DEVICE[m["arch"]])

    def test_train_step_stops_when_cpu_not_guaranteed(self):
        for devs in (FakeDevices(gpu=True, ignore_scope=True), FakeDevices(gpu=True, compute_on="GPU:0")):
            w = self.fresh()
            self.assertPass(run_c2(w.args("check")))
            with mock.patch.object(C2, "TRAINER", fake_trainer(devs=devs)):
                self.assertFail(run_c2(w.args("train")), "CPU execution not guaranteed", "train")
            self.assertNoOutputs(w.run, "train")

    def test_train_step_without_gpu(self):
        w = self.fresh()
        with mock.patch.object(C2, "TRAINER", fake_trainer(devs=FakeDevices(gpu=False))):
            pipeline(w, "score_dev")
        rec = json.loads((w.run / "train.json").read_text())
        self.assertEqual({m["training_device"]["effective"] for m in rec["models"].values()}, {"CPU:0"})

    def tampered(self, fn, text, passes=False):
        w = self.fresh()
        pipeline(w, "train")
        fn(w)
        if passes:
            self.assertPass(run_c2(w.args("score", "--set", "dev")))
            return
        self.assertFail(run_c2(w.args("score", "--set", "dev")), text, "score")
        self.assertNoOutputs(w.run, "score_dev", "select", "score_tests", "apply")

    def card(self, arch, edit, same_in_train_json=True):
        """Scheda modificata e ridichiarata (SHA-256 in train.json aggiornato, così scatta la regola del dispositivo e non
        il checksum); con same_in_train_json anche la voce training_device di train.json segue la scheda."""
        def f(w):
            p = w.run / "checkpoints" / f"{arch}__casia_fasd.json"
            d = json.loads(p.read_text())
            edit(d)
            p.write_text(json.dumps(d))
            rec = json.loads((w.run / "train.json").read_text())
            mid = next(m["id"] for m in w.cfg["models"] if m["arch"] == arch)
            rec["models"][mid]["card_sha256"] = C2.sha256_file(p)
            if same_in_train_json:
                rec["models"][mid]["training_device"] = d.get("training_device")
            (w.run / "train.json").write_text(json.dumps(rec))
        return f

    def test_livenessnet_gpu_training_refused_before_scoring(self):
        self.tampered(self.card("LivenessNet", lambda d: d.update(training_device=self.record("CPU", "GPU:0", ["CPU", "GPU"], 0.0))),
                      "training device expected 'GPU:0', expected 'CPU:0' (CPU execution not guaranteed)")
        self.tampered(self.card("LivenessNet", lambda d: d["training_device"].update(effective="GPU:0")),
                      "training device effective 'GPU:0', expected 'CPU:0' (CPU execution not guaranteed)")

    def test_livenessnet_graph_eager_gap_refused(self):
        self.tampered(self.card("LivenessNet", lambda d: d["training_device"].update(graph_eager_max_rel_diff=0.5)),
                      "graph-mode outputs differed from eager by 0.5 (relative), above 1e-05")
        self.tampered(self.card("LivenessNet", lambda d: d["training_device"].update(graph_eager_max_rel_diff=1e-5, graph_eager_layer="output")),
                      None, passes=True)

    def test_graph_eager_gap_malformed(self):
        for v in (None, "0.0", 0, -1e-9, float("nan"), float("inf"), True):
            for arch in ("LivenessNet", "AttackNetV2_2"):
                self.tampered(self.card(arch, lambda d, v=v: d["training_device"].update(graph_eager_max_rel_diff=v)), "graph/eager difference")

    def test_attacknet_gap_not_enforced_before_scoring(self):
        self.tampered(self.card("AttackNetV2_2", lambda d: d["training_device"].update(graph_eager_max_rel_diff=0.5)), None, passes=True)

    def test_livenessnet_requested_default_refused(self):
        self.tampered(self.card("LivenessNet", lambda d: d["training_device"].update(requested="default")),
                      "training device requested 'default', expected 'CPU'")

    def test_card_without_device_refused(self):
        self.tampered(self.card("AttackNetV2_2", lambda d: d.pop("training_device")), "the card does not record the training device")
        self.tampered(self.card("LivenessNet", lambda d: d.update(training_device="CPU")), "the card does not record the training device")

    def test_attacknet_device_against_visible_devices(self):
        """AttackNet: con una GPU visibile l'atteso è GPU:0 (CPU:0 rifiutato); senza GPU è CPU:0 (GPU:0 rifiutato)."""
        self.tampered(self.card("AttackNetV2_2", lambda d: d.update(training_device=self.record("default", "CPU:0", ["CPU", "GPU"], 0.0))),
                      "training device expected 'CPU:0', expected 'GPU:0'")
        self.tampered(self.card("AttackNetV2_2", lambda d: d.update(training_device=self.record("default", "GPU:0", ["CPU"], 0.0))),
                      "training device effective 'GPU:0', expected 'CPU:0'")
        self.tampered(self.card("AttackNetV2_2", lambda d: d["training_device"].update(requested="CPU")), "training device requested 'CPU', expected 'default'")

    def test_hidden_gpu_recorded_refused(self):
        rec = self.record("default", "CPU:0", ["CPU"], 0.0, physical=["CPU", "GPU"])
        self.tampered(self.card("AttackNetV2_2", lambda d: d.update(training_device=rec)), "a GPU was present but hidden from the training process")

    def test_visible_or_physical_devices_malformed(self):
        for key in ("visible", "physical"):
            for v in (None, [], ["GPU"], ["CPU", "TPU"], "CPU", ["GPU", "CPU"], ["CPU", "CPU"]):
                self.tampered(self.card("LivenessNet", lambda d, k=key, v=v: d["training_device"].update({k: v})), "not recorded correctly")

    def test_device_record_fields_exact(self):
        """Esattamente i campi scritti da train_cnn.py: un campo in più o uno mancante è rifiutato."""
        self.tampered(self.card("LivenessNet", lambda d: d["training_device"].update(note="cpu")), "training device fields")
        self.tampered(self.card("AttackNetV2_2", lambda d: d["training_device"].pop("graph_eager_layer")), "training device fields")

    def test_visible_devices_not_physical_refused(self):
        self.tampered(self.card("LivenessNet", lambda d: d["training_device"].update(visible=["CPU", "GPU"], physical=["CPU"])),
                      "visible devices ['CPU', 'GPU'] not among the physical devices ['CPU']")

    def test_graph_eager_layer_consistent_with_the_difference(self):
        """Strato registrato (nome non vuoto) solo se lo scarto non è 0, come lo scrive train_cnn.py."""
        for gap, layer in ((0.0, "foo"), (1e-6, None), (1e-6, ""), (1e-6, 3)):
            for arch in ("LivenessNet", "AttackNetV2_2"):
                self.tampered(self.card(arch, lambda d, g=gap, l=layer: d["training_device"].update(graph_eager_max_rel_diff=g,
                                                                                                    graph_eager_layer=l)),
                              "graph/eager layer")
        self.tampered(self.card("LivenessNet", lambda d: d["training_device"].update(graph_eager_max_rel_diff=1e-6,
                                                                                     graph_eager_layer="bn_dense")), None, passes=True)

    def test_train_record_and_card_disagree(self):
        def f(w):
            rec = json.loads((w.run / "train.json").read_text())
            rec["models"]["livenessnet__casia_fasd_c2"]["training_device"] = {**rec["models"]["livenessnet__casia_fasd_c2"]["training_device"],
                                                                              "effective": "GPU:0"}
            (w.run / "train.json").write_text(json.dumps(rec))
        self.tampered(f, "train.json and card disagree on the training device")

    def test_device_checked_before_every_scoring(self):
        """Il controllo vale anche per il secondo punteggio (test sorgente e NUAA): scheda alterata dopo select."""
        w = self.fresh()
        pipeline(w, "select")
        self.card("LivenessNet", lambda d: d["training_device"].update(effective="GPU:0"))(w)
        self.assertFail(run_c2(w.args("score", "--set", "source_test,target")), "training device effective 'GPU:0'", "score")
        self.assertNoOutputs(w.run, "score_tests", "apply")


class TestZZStructure(unittest.TestCase):
    def test_tensorflow_never_imported(self):
        # nessun modulo di TensorFlow o Keras resta caricato dopo tutta la suite (i moduli finti si tolgono all'uscita)
        loaded = sorted(m for m in sys.modules if m.split(".")[0] in ("tensorflow", "keras", "tf_keras"))
        self.assertEqual(loaded, [])

    def test_every_skip_is_a_missing_input(self):
        src = Path(__file__).read_text(encoding="utf-8")
        self.assertEqual(src.count(".skip" + "Test("), 1)
        used = re.findall(r'skip_missing\((?:self|test), "([^"]+)"', src)
        self.assertEqual(sorted(set(used)), sorted(MISSING_INPUT))
        for why in MISSING_INPUT.values():
            self.assertFalse(re.search(r"case|filesystem|platform|linux|macos|windows|tensorflow", why, re.I), why)

    def test_real_repository_untouched(self):
        now = snapshot()
        if START["git"] is None or now["git"] is None:
            skip_missing(self, "git")
        self.assertEqual(now["git"], START["git"], "git status of the real repository changed during the tests")
        self.assertEqual(now["files"], START["files"], "files under data/processed, models/weights, results or export changed")
        self.assertEqual(os.listdir(TMP_ROOT / "weights"), [DECOY.name])


def main() -> int:
    global C2, TC, rm, nm, st, START, BLOCKED
    ap = argparse.ArgumentParser(description="Test di c2_pilot.py e degli adattamenti di train_cnn.py su dati sintetici")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--block", default="", help="pacchetti da rendere assenti, separati da virgole (es. tensorflow,keras,h5py,cv2)")
    a = ap.parse_args()
    BLOCKED = [b for b in a.block.split(",") if b]
    if BLOCKED:
        sys.meta_path.insert(0, Blocker())
        for b in BLOCKED:
            for n in [m for m in sys.modules if m == b or m.startswith(b + ".")]:
                del sys.modules[n]
        print(f"NOTE: imports blocked for this run: {', '.join(BLOCKED)}")
    START = snapshot()
    sys.addaudithook(_audit)
    ensure_cv2()
    if CV2_NOTE:
        print(CV2_NOTE)
    C2 = load("c2_pilot", SCRIPT)
    sys.modules["c2_pilot"] = C2
    rm, nm, st = C2.rm, C2.nm, C2.st
    TC = load_train_cnn()
    classes = (TestConfig, TestRunFolder, TestCheck, TestTrainCnn, TestTrainCnnGuards, TestTrainStep, TestScore, TestSelect, TestApply,
               TestInvalidation, TestRefusalsAndStatus, TestRecordsAndWeights, TestThirdReview, TestFourthReview, TestSweep,
               TestTeacherReview21, TestTrainingDevice, TestZZStructure)
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in classes)
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    if a.verbose or not result.wasSuccessful():
        print(log.getvalue())
    for t, why in result.skipped:
        print(f"skipped {t.id().rsplit('.', 1)[-1]}: {why}")
    if BLOCKED:
        leaked = sorted({m.split(".")[0] for m, v in sys.modules.items() if v is not CV2_STUB} & set(BLOCKED))
        print(f"blocked imports attempted: {len(IMPORT_LOG)}; blocked packages loaded: {leaked or 'none'}")
    shutil.rmtree(TMP_ROOT, ignore_errors=True)
    print(f"tests run: {result.testsRun}, failures: {len(result.failures)}, errors: {len(result.errors)}, skipped: {len(result.skipped)}")
    print("RESULT: " + ("PASS" if result.wasSuccessful() else "FAIL"))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
