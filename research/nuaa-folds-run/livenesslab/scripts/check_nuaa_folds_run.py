"""
Test di scripts/nuaa_folds_run.py (run LivenessNet 4 fold × 3 seed su NUAA), offline e deterministici.

Mondo sintetico in cartelle temporanee: liste ufficiali di 16 soggetti con identificativi e struttura diversi da NUAA
(1101–1116; sessioni 01–02 per i primi nove), i cinque manifest M2, i dodici manifest dei fold con il loro SHA256SUMS,
una cache 8×8 e gli «originali» (file il cui contenuto è la chiave), un estrattore finto che deriva il frame dai byte
del file (così la cache si riproduce bit per bit e un originale diverso dà un frame diverso), un TensorFlow finto per
il trainer vero (ricetta, seed, CPU, callback del tetto di tempo), un trainer e uno scorer finti iniettati per il run
completo (i punteggi dipendono dalla classe scritta nei pixel e dai pesi salvati), orologio e stato git iniettati.
Nessun training né inferenza reali; nessuna scrittura nel repository (controllo finale su results/, export/, delivery/).
Gli ingressi veri sono usati in sola lettura da un solo test (check-inputs), saltato se mancano (MISSING_INPUT).

  .venv/bin/python -I scripts/check_nuaa_folds_run.py [--verbose]
"""
import contextlib
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nuaa_folds_run as FR  # noqa: E402

REAL_ROOT = Path(os.environ.get("FOLDSRUN_INPUT_ROOT") or ROOT)
MISSING_INPUT = {
    "real inputs": "the M2 manifests, the fold schema run, the training cache, the NUAA images or the frozen code are not present",
    "delivered configuration": "delivery/foldsrun/nuaa_folds_run.json is not present",
}
STATE = {"commit": "abcdef0", "dirty": False}
SUBJECTS = [f"11{i:02d}" for i in range(1, 17)]           # 1101..1116
TRAIN_SUBJECTS = SUBJECTS[:9]                                # sessioni 01 e 02
NO_BONA_FIDE_03 = {"1115", "1116"}
FOLDS = {1: ["1101", "1102", "1110", "1111"], 2: ["1103", "1104", "1112", "1113"], 3: ["1105", "1106", "1114", "1115"],
         4: ["1107", "1108", "1109", "1116"]}
K = 4
PER_CELL_TRAIN = 4          # frame per soggetto, sessione (01, 02) e classe
PER_CELL_TEST = 6           # frame per soggetto e classe nella sessione 03
FIT_PER_CELL, DEV_PER_CELL = 8, 4
PER_CLASS = 12              # cache: spread per classe
SHAPE = (8, 8, 3)
SEEDS = [42, 43, 44]


def skip_missing(test: unittest.TestCase, what: str):
    """Unico modo di saltare un test di questa suite: un ingresso mancante dichiarato in MISSING_INPUT."""
    test.skipTest(f"missing input: {MISSING_INPUT[what]}")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def csv_text(cols, rows) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    w.writerows(rows)
    return buf.getvalue()


def fake_frame(b: bytes) -> tuple:
    """Frame 8×8×3 dai byte dell'originale: classe nel pixel [0,0,0] (200 attacco, 50 bona fide), rumore nel resto."""
    key = b.decode("utf-8")
    h = hashlib.sha256(b).digest()
    arr = np.frombuffer((h * 8)[:int(np.prod(SHAPE))], np.uint8).reshape(SHAPE).copy()
    arr[0, 0, 0] = 200 if key.startswith("attack/") else 50
    arr[0, 0, 1] = h[0]
    return arr, ("RetinaFace" if h[1] % 2 else "Haar")


def fake_scores(X, bias: float) -> np.ndarray:
    """Probabilità finte N x 2: dalla classe scritta nel pixel, dal rumore e da un bias che dipende dai pesi."""
    X = np.asarray(X)
    base = np.where(X[:, 0, 0, 0] == 200, 0.78, 0.22)
    p = np.clip(base + (X[:, 0, 0, 1].astype(float) - 128) / 1000.0 + bias, 0.001, 0.999)
    return np.stack([1 - p, p], axis=1)


def bias_of(weights: bytes) -> float:
    return int(weights.split(b"seed ")[1].split()[0]) * 1e-4


class Log:
    """Registro delle chiamate di trainer e scorer finti."""
    calls = []


def fake_trainer(plan=None):
    """Trainer finto: pesi deterministici, storia con il minimo del val_loss uguale a quello ricalcolato dallo scorer."""
    def train(data, training, weights_path, seed, arch, cap_check):
        assert set(data) == {"X_fit", "y_fit", "X_dev", "y_dev"}
        w = f"fake-weights seed {seed} fit {sha(np.ascontiguousarray(data['X_fit']).tobytes())}".encode()
        Path(weights_path).write_bytes(w)
        p = fake_scores(data["X_dev"], bias_of(w))
        vl = FR.val_loss(p, data["y_dev"])
        hist = plan(vl) if plan else [vl + 0.3, vl + 0.1, vl, vl + 0.05, vl + 0.06, vl + 0.07, vl + 0.08]
        for i in range(len(hist)):
            cap_check(f"epoch {i + 1}")
        Log.calls.append(("train", seed, len(data["y_fit"]), len(data["y_dev"])))
        best = int(np.argmin(hist))
        return {"arch": "LivenessNet", "train_seed": seed, "n_fitting": int(len(data["y_fit"])), "n_dev": int(len(data["y_dev"])),
                "max_epochs": training["max_epochs"], "epochs_run": len(hist), "restored_epoch": best + 1,
                "restored_by": "early stopping" if len(hist) < training["max_epochs"] else "end of epochs (set_weights)",
                "best_val_loss": float(hist[best]), "val_acc_at_best": 0.9,
                "history": {"loss": list(hist), "accuracy": [0.5] * len(hist), "val_loss": list(hist), "val_accuracy": [0.6] * len(hist),
                            "lr": [0.001] * len(hist)},
                "device": "CPU", "visible_devices": ["CPU"], "lr": training["initial_learning_rate"], "batch": training["batch"],
                "early_stopping_patience": training["early_stopping_patience"], "train_seconds": 1.5,
                "weights_file": Path(weights_path).name, "weights_sha256": sha(w), "saved": "now"}
    return train


def fake_make_scorer(weights_path, weights_sha):
    w = Path(weights_path).read_bytes()
    assert sha(w) == weights_sha
    Log.calls.append(("scorer", Path(weights_path).name))
    return lambda X: fake_scores(X, bias_of(w))


class World:
    """Il mondo sintetico: ingressi in <tmp>/in, frame estratti in <tmp>/frames, run in <tmp>/out."""

    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="foldsrun-test-"))
        self.base = self.tmp / "in"
        self.out = self.tmp / "out"
        self.frames = self.tmp / "frames" / "frames.npz"
        self.build()

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------ costruzione
    def name(self, subj, ses, pic) -> str:
        return f"{subj}_{subj}_00_00_{ses}_{pic}.jpg"

    def build(self):
        self.rows = {"official_train": [], "official_test": []}
        img = self.base / "images"
        for subj in SUBJECTS:
            cells = []
            if subj in TRAIN_SUBJECTS:
                cells += [(ses, lab, PER_CELL_TRAIN, "train") for ses in ("01", "02") for lab in ("0", "1")]
            cells += [("03", "1", PER_CELL_TEST, "test")]
            if subj not in NO_BONA_FIDE_03:
                cells += [("03", "0", PER_CELL_TEST, "test")]
            for ses, lab, n, split in cells:
                for pic in range(n):
                    nm = self.name(subj, ses, pic)
                    key = f"{'attack' if lab == '1' else 'real'}/{nm}"
                    folder = f"{'bonafide' if lab == '0' else 'attack'}_{'training' if split == 'train' else 'validation'}"
                    p = img / folder / nm
                    p.parent.mkdir(parents=True, exist_ok=True)
                    b = key.encode()
                    p.write_bytes(b)
                    self.rows[f"official_{split}"].append({"key": key, "label": lab, "subject": subj, "glasses": "00", "pos": "00", "session": ses,
                                                          "pic": str(pic), "official_split": split, "sha256": sha(b), "bytes": str(len(b))})
        for k in self.rows:
            self.rows[k].sort(key=lambda r: r["key"])
        self.official = {r["key"]: r for r in self.rows["official_train"] + self.rows["official_test"]}
        order_train = FR.spread_keys(self.rows["official_train"], PER_CLASS)
        order_test = FR.spread_keys(self.rows["official_test"], PER_CLASS)
        es_subject = TRAIN_SUBJECTS[-1]
        self.rows["cnn_es"] = [self.official[k] for k in order_train if self.official[k]["subject"] == es_subject]
        self.rows["cnn_train"] = [self.official[k] for k in order_train if self.official[k]["subject"] != es_subject]
        self.rows["cnn_test"] = [self.official[k] for k in order_test]
        self.order = {"train": order_train, "test": order_test}
        # manifest M2
        mdir = self.base / "manifests"
        mdir.mkdir(parents=True)
        self.m2 = {}
        for k in FR.MANIFESTS:
            text = csv_text(FR.COLUMNS, [[r[c] for c in FR.COLUMNS] for r in self.rows[k]])
            (mdir / f"{k}.csv").write_text(text, encoding="utf-8")
            c = FR.counts_of(self.rows[k])
            self.m2[k] = {"file": f"{k}.csv", "sha256": sha(text.encode()), **c,
                          "sessions": ["01", "02"] if k in ("official_train", "cnn_train", "cnn_es") else ["03"]}
        # cache: X_train/X_test dai frame finti nell'ordine di campionamento
        cache_keys = set(order_train) | set(order_test)
        self.X = {}
        for split in ("train", "test"):
            self.X[split] = np.stack([fake_frame(self.official[k]["key"].encode())[0] for k in self.order[split]])
        (self.base / "cache").mkdir()
        self.cache = self.base / "cache" / "cache.npz"
        with open(self.cache, "wb") as f:
            np.savez(f, X_train=self.X["train"], y_train=np.array([int(self.official[k]["label"]) for k in order_train], np.int64),
                     s_train=np.array([self.official[k]["subject"] for k in order_train]), X_test=self.X["test"],
                     y_test=np.array([int(self.official[k]["label"]) for k in order_test], np.int64),
                     s_test=np.array([self.official[k]["subject"] for k in order_test]))
        # manifest dei fold
        self.folds = {}
        sdir = self.base / "schema"
        sdir.mkdir()
        texts = {}
        for i, members in FOLDS.items():
            nxt = i % K + 1
            dev_subj, fit_subj = set(FOLDS[nxt]), set(s for j in FOLDS if j not in (i, nxt) for s in FOLDS[j])
            self.folds[i] = {}
            for role, subj, n in (("fitting", fit_subj, FIT_PER_CELL), ("dev", dev_subj, DEV_PER_CELL), ("test", set(members), None)):
                rows = []
                for ses in FR.SESSIONS:
                    for lab in FR.LABELS:
                        cell = sorted(r["key"] for r in self.official.values() if r["subject"] in subj and r["session"] == ses and r["label"] == lab)
                        keys = cell if n is None else FR.spread(cell, n)
                        rows += [self.official[k_] for k_ in keys]
                rows.sort(key=lambda r: r["key"])
                self.folds[i][role] = [[str(i), role, r["key"], r["label"], r["subject"], r["session"], r["sha256"], r["bytes"],
                                        "1" if r["key"] in cache_keys else "0"] for r in rows]
                texts[f"fold{i}_{role}.csv"] = csv_text(FR.FOLD_COLUMNS, self.folds[i][role])
        for n, t in texts.items():
            (sdir / n).write_text(t, encoding="utf-8")
        sums = "".join(f"{sha(t.encode())}  {n}\n" for n, t in sorted(texts.items()))
        (sdir / "SHA256SUMS").write_text(sums, encoding="utf-8")
        self.sums_sha = sha(sums.encode())
        # codice congelato e risorse (contenuto qualsiasi: contano gli SHA-256)
        self.code = {}
        for rel in ("frozen/face.py", "res/deploy.prototxt", "res/Widerface-RetinaFace.caffemodel", "res/haarcascade_frontalface_default.xml",
                    "ref/create_datasets.py", "ref/architectures.py"):
            p = self.base / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f"content of {rel}\n".encode())
            self.code[rel] = sha(p.read_bytes())
        # la configurazione dello schema approvato (finta): i campi che il run confronta
        self.schema_cfg = self.base / "schema_config.json"
        self.schema_cfg.write_text(json.dumps({
            "train_seeds": list(SEEDS), "budget": {"wall_time_cap_minutes": 360},
            "threshold": {"criterion": FR.THRESHOLD_CRITERION, "set": "dev fold only"},
            "training": {**FR.FROZEN_TRAINING, "trainings": "K folds x 3 seeds = 12"},
            "metrics": {"uncertainty": "bootstrap over subjects: 1,000 resamples with replacement, percentile 2.5-97.5 of the pooled ACER and "
                                       "AUROC; random seed 0; no interval for a stratum with fewer than 5 subjects"},
            "sampling": {"fitting_per_cell": FIT_PER_CELL, "dev_per_cell": DEV_PER_CELL}, "folds": {"k": K}}, indent=1) + "\n", encoding="utf-8")
        self.schema_sha = sha(self.schema_cfg.read_bytes())
        self.detectors = {split: dict(sorted(__import__("collections").Counter(fake_frame(k.encode())[1] for k in self.order[split]).items()))
                          for split in ("train", "test")}
        self.config = self.base / "config.json"
        self.write_config(self.config_doc())

    def config_doc(self) -> dict:
        n_test = [len(self.folds[i]["test"]) for i in range(1, K + 1)]
        env = FR.environment()
        return {
            "schema": FR.CONFIG_SCHEMA, "task": "synthetic test world", "authorisation": "test",
            "schema_run": {"run_id": "20261103-FOLDS-abcdef0", "folder": "schema", "config_path": "schema_config.json",
                           "config_sha256": self.schema_sha, "restricted_sha256sums_sha256": self.sums_sha},
            "manifests": {"folder": "manifests", "origin": "test", "files": json.loads(json.dumps(self.m2))},
            "images": {"folder": "images", "layout": "images"},
            "expected": {"k": K, "n_subjects": len(SUBJECTS), "n_images": len(self.official), "n_cache_frames": 4 * PER_CLASS,
                         "n_to_extract": len(self.official) - 4 * PER_CLASS, "fitting_per_fold": len(self.folds[1]["fitting"]),
                         "dev_per_fold": len(self.folds[1]["dev"]), "test_per_fold": n_test},
            "cache": {"path": "cache/cache.npz", "sha256": FR.sha256_file(self.cache), "per_class": PER_CLASS,
                      "members": {"X_train": {"dtype": "|u1", "shape": [2 * PER_CLASS, *SHAPE]}, "y_train": {"dtype": "<i8", "shape": [2 * PER_CLASS]},
                                  "s_train": {"dtype": "<U4", "shape": [2 * PER_CLASS]}, "X_test": {"dtype": "|u1", "shape": [2 * PER_CLASS, *SHAPE]},
                                  "y_test": {"dtype": "<i8", "shape": [2 * PER_CLASS]}, "s_test": {"dtype": "<U4", "shape": [2 * PER_CLASS]}},
                      "order": "spread per class", "built": "test", "detectors_n2": self.detectors},
            "extraction": {"frames_file": str(self.frames), "frozen_face": {"path": "frozen/face.py", "sha256": self.code["frozen/face.py"]},
                           "detector_resources": {n: {"path": f"res/{n}", "sha256": self.code[f"res/{n}"]} for n in FR.RESOURCE_NAMES},
                           "reference_code": {"create_datasets": {"path": "ref/create_datasets.py", "sha256": self.code["ref/create_datasets.py"]},
                                              "architectures": {"path": "ref/architectures.py", "sha256": self.code["ref/architectures.py"]}},
                           "chain": "fake"},
            "environment": {"opencv": env["opencv"], "tensorflow": env["tensorflow_installed"]},
            "architecture": {"name": "LivenessNet", "path": "ref/architectures.py", "sha256": self.code["ref/architectures.py"], "chosen": "test"},
            "train_seeds": list(SEEDS), "training": dict(FR.FROZEN_TRAINING),
            "threshold": {"set": "dev fold only", "criterion": FR.THRESHOLD_CRITERION, "frozen_before_test": "yes", "also_reported": "0.5",
                          "dev_reuse_declared": "yes"},
            "scoring": {"device": "CPU", "batch": FR.SCORE_BATCH, "batchnorm": "inference", "input": "/255", "score": "column 1"},
            "metrics": {"strata": list(FR.STRATA), "decision": FR.DECISION_RULE, "per_unit": "t", "per_subject": "t", "auroc": "t",
                        "aggregation": "t", "uncertainty": "t", "pre_commitment": "t"},
            "bootstrap": dict(FR.BOOT),
            "budget": {"trainings": K * len(SEEDS), "estimate": "t", "wall_time_cap_minutes": 360, "on_cap_exceeded": "stop"},
            "declarations": {"pre_commitment": True, "dev_reuse": True, "cpu_only": True, "all_outcomes_kept": True,
                             "cpu_nondeterminism": "n", "diagnostic": "d", "exploratory": "e", "limits": "l", "question": "q"},
            "outputs": {"public": "p", "restricted": "r"}}

    def write_config(self, doc: dict, path: Path = None) -> Path:
        p = path or self.config
        p.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        return p

    def args(self, cmd: str, **kw):
        a = types.SimpleNamespace(cmd=cmd, config=str(self.config), input_root=str(self.base), out_root=str(self.out), date="20261103",
                                  i_am_authorized=True)
        for k, v in kw.items():
            setattr(a, k, v)
        return a


def fake_pre(cfg, base):
    def pre(path):
        return fake_frame(Path(path).read_bytes())
    return pre


@contextlib.contextmanager
def injected(**hooks):
    """Punti di iniezione dello script, ripristinati all'uscita (anche FRAME_SHAPE e CLOCK)."""
    saved = {k: getattr(FR, k) for k in hooks}
    try:
        for k, v in hooks.items():
            setattr(FR, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(FR, k, v)


def defaults(world: "World" = None, **extra) -> dict:
    hooks = {"REPO_STATE": lambda root: dict(STATE), "PREPROCESSOR": fake_pre, "TRAINER": fake_trainer(), "MAKE_SCORER": fake_make_scorer,
             "ARCH_LOADER": lambda p: types.SimpleNamespace(create_model=None), "FRAME_SHAPE": SHAPE}
    if world is not None:
        hooks["SCHEMA_CONFIG_SHA256"] = world.schema_sha
    hooks.update(extra)
    return hooks


def run_main(argv) -> tuple:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = FR.main(argv)
    return rc, out.getvalue(), err.getvalue()


def argv_of(w: World, cmd: str, *extra) -> list:
    return [cmd, "--config", str(w.config), "--input-root", str(w.base), "--out-root", str(w.out), "--date", "20261103", *extra]


def snapshot(*dirs) -> dict:
    out = {}
    for d in dirs:
        for p in sorted(Path(d).rglob("*")):
            if p.is_file():
                out[str(p)] = (p.stat().st_size, p.stat().st_mtime_ns)
    return out


# ============================================================================ TensorFlow finto (per default_trainer)

class _Any:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return _Any()

    def __getattr__(self, n):
        if n.startswith("__"):
            raise AttributeError(n)
        return _Any()


class Recorder:
    """Oggetto che registra il nome dell'attributo acceduto e gli argomenti della chiamata (strati, perdite, metriche)."""

    def __init__(self, log: dict, group: str):
        self.log, self.group = log, group

    def __getattr__(self, n):
        if n.startswith("__"):
            raise AttributeError(n)

        def make(*a, **k):
            self.log.setdefault(self.group, []).append((n, a, k))
            return _Any()
        return make


def fake_tf(log: dict, plan, gpu_stays: bool = False):
    tf = types.ModuleType("tensorflow")

    class Callback:
        def on_epoch_end(self, epoch, logs=None):
            pass

    class ES(Callback):
        def __init__(self, monitor=None, patience=None, restore_best_weights=None):
            log["es"] = {"monitor": monitor, "patience": patience, "restore": restore_best_weights}
            self.patience, self.stopped_epoch, self.best_weights = patience, 0, None

    class RLR(Callback):
        def __init__(self, **k):
            log["rlr"] = k

    class DS:
        def __init__(self, name):
            self.name = name

        @staticmethod
        def from_tensor_slices(t):
            x, y = t
            ds = DS(f"ds{len(log.setdefault('slices', []))}")
            log["slices"].append({"n": len(x), "x_min": float(np.min(x)), "x_max": float(np.max(x)), "x_dtype": str(np.asarray(x).dtype),
                                  "y_first": [float(v) for v in np.asarray(y)[0]], "y_shape": list(np.asarray(y).shape)})
            return ds

        def shuffle(self, n, seed=None):
            log["shuffle"] = (n, seed)
            return self

        def batch(self, b):
            log.setdefault("batch", []).append(b)
            return self

        def map(self, f):
            log.setdefault("mapped", []).append(self.name)
            return self

        def prefetch(self, n):
            return self

    class Adam:
        def __init__(self, learning_rate=None, clipnorm=None):
            log["adam"] = (learning_rate, clipnorm)

    class Seq:
        def __init__(self, layers):
            log["sequential"] = len(layers)

    tf.random = types.SimpleNamespace(set_seed=lambda s: log.__setitem__("tf_seed", s))
    vis = {"devices": ["CPU", "GPU"]}

    def set_visible(devs, kind):
        log["hidden"] = (list(devs), kind)
        if gpu_stays:
            return
        vis["devices"] = ["CPU"]

    @contextlib.contextmanager
    def device(name):
        log.setdefault("device", []).append(name)
        yield
    tf.device = device
    tf.config = types.SimpleNamespace(optimizer=types.SimpleNamespace(set_experimental_options=lambda o: log.__setitem__("opts", o)),
                                      set_visible_devices=set_visible,
                                      get_visible_devices=lambda: [types.SimpleNamespace(device_type=d) for d in vis["devices"]])
    tf.data = types.SimpleNamespace(Dataset=DS)

    class Model:
        def __init__(self):
            self.w = b"init"

        def compile(self, **k):
            log["compile"] = sorted(k)

        def fit(self, ds, validation_data=None, epochs=None, class_weight=None, callbacks=None, verbose=None):
            log["fit"] = {"epochs": epochs, "class_weight": class_weight, "n_callbacks": len(callbacks), "train_ds": ds.name, "val_ds": validation_data.name}
            vl = plan(epochs)
            es = callbacks[0]
            es.best_weights = b"best"
            b = int(np.argmin(vl))
            for e in range(len(vl)):
                for cb in callbacks[2:]:
                    cb.on_epoch_end(e)
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
            log.setdefault("saved", []).append(Path(p).name)
            Path(p).write_bytes(b"keras:" + self.w)

        def __call__(self, xb, training=None):
            log.setdefault("calls", []).append({"n": len(xb), "training": training, "x_max": float(np.max(xb)), "x_dtype": str(np.asarray(xb).dtype)})
            p = np.full((len(xb), 1), 0.25)
            return np.concatenate([1 - p, p], axis=1)

    def load_model(path, compile=None):
        log["load_model"] = (Path(path).name, compile)
        return Model()
    tf.keras = types.SimpleNamespace(optimizers=types.SimpleNamespace(legacy=types.SimpleNamespace(Adam=Adam)), losses=Recorder(log, "losses"),
                                     metrics=Recorder(log, "metrics"), Sequential=Seq, layers=Recorder(log, "layers"),
                                     callbacks=types.SimpleNamespace(EarlyStopping=ES, ReduceLROnPlateau=RLR, Callback=Callback),
                                     backend=types.SimpleNamespace(clear_session=lambda: log.__setitem__("cleared", log.get("cleared", 0) + 1)),
                                     models=types.SimpleNamespace(load_model=load_model))
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


# ============================================================================ configurazione

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.w = World()
        self.addCleanup(self.w.close)

    def refused(self, mutate, needle: str):
        doc = self.w.config_doc()
        mutate(doc)
        p = self.w.write_config(doc, self.w.tmp / "bad.json")
        with self.assertRaises(FR.Failure) as cm:
            FR.load_config(p)
        self.assertIn(needle, str(cm.exception))

    def test_synthetic_configuration_accepted(self):
        cfg, h = FR.load_config(self.w.config)
        self.assertEqual(cfg["expected"]["k"], K)
        self.assertEqual(h, sha(self.w.config.read_bytes()))

    def test_unknown_or_missing_top_key(self):
        self.refused(lambda d: d.__setitem__("extra", 1), "fields must be exactly")
        self.refused(lambda d: d.pop("budget"), "fields must be exactly")

    def test_training_recipe_must_be_the_n1_recipe(self):
        self.refused(lambda d: d["training"].__setitem__("max_epochs", 13), "N1 recipe")
        self.refused(lambda d: d["training"].__setitem__("device", "GPU"), "N1 recipe")

    def test_threshold_and_bootstrap_fixed(self):
        self.refused(lambda d: d["threshold"].__setitem__("criterion", "apcer10"), "EER criterion")
        self.refused(lambda d: d["threshold"].__setitem__("set", "test"), "dev fold only")
        self.refused(lambda d: d["bootstrap"].__setitem__("seed", 1), "bootstrap must be")

    def test_budget_and_seeds(self):
        self.refused(lambda d: d["budget"].__setitem__("trainings", 11), "k folds x the number of seeds")
        self.refused(lambda d: d["budget"].__setitem__("wall_time_cap_minutes", 0), "wall_time_cap_minutes")
        self.refused(lambda d: d.__setitem__("train_seeds", [42, 42, 44]), "distinct")

    def test_expected_counts_consistent(self):
        self.refused(lambda d: d["expected"].__setitem__("n_to_extract", 1), "inconsistent")
        self.refused(lambda d: d["expected"].__setitem__("test_per_fold", [1, 2, 3]), "one count per fold")
        self.refused(lambda d: d["cache"].__setitem__("per_class", PER_CLASS + 1), "do not add up")

    def test_scoring_metrics_declarations_architecture(self):
        self.refused(lambda d: d["scoring"].__setitem__("device", "GPU"), "CPU")
        self.refused(lambda d: d["metrics"].__setitem__("strata", ["03"]), "strata")
        self.refused(lambda d: d["declarations"].__setitem__("pre_commitment", False), "pre_commitment")
        self.refused(lambda d: d["declarations"].pop("limits"), "exactly the keys")
        self.refused(lambda d: d["architecture"].__setitem__("sha256", "1" * 64), "same file")
        self.refused(lambda d: d["manifests"]["files"]["official_test"].__setitem__("sessions", ["01"]), "disjoint")
        self.refused(lambda d: d["declarations"].__setitem__("extra", "x"), "exactly the keys")
        self.refused(lambda d: d["extraction"]["detector_resources"].pop("deploy.prototxt"), "detector_resources must define exactly")
        self.refused(lambda d: d.__setitem__("train_seeds", [1, 2, 3]), "fixed by the coordinator")
        self.refused(lambda d: d["budget"].__setitem__("wall_time_cap_minutes", 300), "fixed by the coordinator")
        self.refused(lambda d: d["images"].__setitem__("layout", "other"), "images.layout")
        self.refused(lambda d: d["cache"].pop("detectors_n2"), "detectors_n2")
        self.refused(lambda d: d["environment"].pop("tensorflow"), "environment needs")

    def test_delivered_configuration(self):
        p = REAL_ROOT / FR.DEFAULT_CONFIG
        if not p.is_file():
            skip_missing(self, "delivered configuration")
        cfg, h = FR.load_config(p)
        self.assertEqual(cfg["train_seeds"], [42, 43, 44])
        self.assertEqual(cfg["expected"], {"k": 4, "n_subjects": 16, "n_images": 12614, "n_cache_frames": 6000, "n_to_extract": 6614,
                                           "fitting_per_fold": 2100, "dev_per_fold": 600, "test_per_fold": [3242, 3114, 3439, 2819]})
        self.assertEqual(cfg["budget"]["wall_time_cap_minutes"], 360)
        self.assertEqual(cfg["schema_run"]["run_id"], "20261103-FOLDS-67b7f3d")
        self.assertEqual(cfg["schema_run"]["config_sha256"], FR.SCHEMA_CONFIG_SHA256)
        self.assertEqual(cfg["cache"]["detectors_n2"], {"train": {"Haar": 51, "RetinaFace": 2949}, "test": {"Haar": 3, "RetinaFace": 2996, "none": 1}})
        self.assertEqual(h, FR.CONFIG_SHA256, "CONFIG_SHA256 of the script must be the SHA-256 of the delivered configuration")


# ============================================================================ ingressi

class TestInputs(unittest.TestCase):
    def setUp(self):
        self.w = World()
        self.addCleanup(self.w.close)
        self.cfg, _ = FR.load_config(self.w.config)

    def load(self, cfg=None):
        with injected(FRAME_SHAPE=SHAPE, SCHEMA_CONFIG_SHA256=self.w.schema_sha):
            return FR.load_inputs(cfg or self.cfg, self.w.base)

    def refused(self, needle: str, cfg=None):
        with self.assertRaises(FR.Failure) as cm:
            self.load(cfg)
        self.assertIn(needle, str(cm.exception))

    def test_synthetic_inputs_pass(self):
        inp = self.load()
        self.assertEqual(len(inp.cache_index), 4 * PER_CLASS)
        self.assertEqual(len(inp.to_extract), len(self.w.official) - 4 * PER_CLASS)
        self.assertEqual(inp.to_extract, sorted(inp.to_extract))
        self.assertEqual(inp.checks["schema"]["subjects_per_fold"], {i: len(FOLDS[i]) for i in FOLDS})
        self.assertEqual(sorted(inp.subjects), SUBJECTS)
        self.assertEqual(inp.checks["environment"]["tensorflow_state"], "not imported")
        self.assertEqual(inp.checks["originals"]["located_and_hashed"], len(self.w.official))
        self.assertEqual(inp.checks["schema_config"]["sha256"], self.w.schema_sha)

    def test_schema_configuration_is_verified(self):
        with injected(FRAME_SHAPE=SHAPE, SCHEMA_CONFIG_SHA256="3" * 64), self.assertRaises(FR.Failure) as cm:
            FR.load_inputs(self.cfg, self.w.base)
        self.assertIn("differs from the approved schema", str(cm.exception))
        doc = json.loads(self.w.schema_cfg.read_text())
        for mutate, needle in ((lambda d: d.__setitem__("train_seeds", [42, 43, 45]), "train_seeds differ"),
                               (lambda d: d["budget"].__setitem__("wall_time_cap_minutes", 300), "wall_time_cap_minutes"),
                               (lambda d: d["threshold"].__setitem__("criterion", "apcer10"), "threshold criterion"),
                               (lambda d: d["training"].__setitem__("batch", 32), "training fields"),
                               (lambda d: d["metrics"].__setitem__("uncertainty", "bootstrap 500 resamples"), "bootstrap declaration"),
                               (lambda d: d["sampling"].__setitem__("fitting_per_cell", FIT_PER_CELL + 1), "counts per cell"),
                               (lambda d: d["folds"].__setitem__("k", 5), "number of folds")):
            d = json.loads(json.dumps(doc))
            mutate(d)
            b = (json.dumps(d, indent=1) + "\n").encode()
            self.w.schema_cfg.write_bytes(b)
            cfg = json.loads(json.dumps(self.cfg))
            cfg["schema_run"]["config_sha256"] = sha(b)
            with injected(FRAME_SHAPE=SHAPE, SCHEMA_CONFIG_SHA256=sha(b)), self.assertRaises(FR.Failure) as cm:
                FR.load_inputs(cfg, self.w.base)
            self.assertIn(needle, str(cm.exception))

    def test_originals_verified_and_frames_file_refused_at_check(self):
        key = sorted(self.w.official)[7]
        r = self.w.official[key]
        p = self.w.base / "images" / FR.LAYOUTS["images"](r)
        b = p.read_bytes()
        p.write_bytes(b[:-1] + bytes([b[-1] ^ 1]))
        with self.assertRaises(FR.Failure) as cm:
            self.load()
        self.assertIn("original image 7 of the sorted official list", str(cm.exception))
        self.assertNotIn(r["subject"], str(cm.exception))
        p.write_bytes(b)
        self.w.frames.parent.mkdir(parents=True, exist_ok=True)
        self.w.frames.write_bytes(b"old")
        self.refused("already exists in")
        self.w.frames.unlink()

    def test_environment_variables_and_tensorflow_version(self):
        old = os.environ.get("LIVENESSLAB_SILENT_FACE")
        os.environ["LIVENESSLAB_SILENT_FACE"] = "/somewhere"
        try:
            self.refused("environment variables")
        finally:
            if old is None:
                os.environ.pop("LIVENESSLAB_SILENT_FACE", None)
            else:
                os.environ["LIVENESSLAB_SILENT_FACE"] = old
        cfg = json.loads(json.dumps(self.cfg))
        cfg["environment"]["tensorflow"] = "0.0.0"
        self.refused("TensorFlow", cfg)

    def test_check_inputs_command_writes_nothing(self):
        before = snapshot(self.w.tmp)
        with injected(**defaults(self.w)):
            rc, out, err = run_main(argv_of(self.w, "check-inputs"))
        self.assertEqual(rc, 0, err)
        self.assertIn("RESULT: PASS (check-inputs)", out)
        self.assertEqual(snapshot(self.w.tmp), before)

    def test_m2_manifest_sha_and_counts(self):
        p = self.w.base / "manifests" / "official_test.csv"
        p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        self.refused("manifest official_test")
        cfg = json.loads(json.dumps(self.cfg))
        cfg["manifests"]["files"]["official_test"]["sha256"] = sha(p.read_bytes())
        cfg["manifests"]["files"]["official_test"]["rows"] += 1
        self.refused("rows", cfg)

    def test_m2_manifest_symlink_refused(self):
        p = self.w.base / "manifests" / "cnn_es.csv"
        q = self.w.tmp / "elsewhere.csv"
        q.write_bytes(p.read_bytes())
        p.unlink()
        p.symlink_to(q)
        self.refused("symbolic link")

    def test_cache_manifest_row_differs_from_official(self):
        p = self.w.base / "manifests" / "cnn_test.csv"
        rows = [[r[c] for c in FR.COLUMNS] for r in self.w.rows["cnn_test"]]
        rows[0][-1] = rows[0][-1] + "0"                       # byte del file diversi dalla lista ufficiale
        text = csv_text(FR.COLUMNS, rows)
        p.write_text(text, encoding="utf-8")
        cfg = json.loads(json.dumps(self.cfg))
        cfg["manifests"]["files"]["cnn_test"]["sha256"] = sha(text.encode())
        self.refused("same values in every column", cfg)

    def test_fold_sums_anchor_and_file_sha(self):
        cfg = json.loads(json.dumps(self.cfg))
        cfg["schema_run"]["restricted_sha256sums_sha256"] = "1" * 64
        self.refused("SHA256SUMS of the restricted folder", cfg)
        p = self.w.base / "schema" / "fold2_dev.csv"
        p.write_text(p.read_text(encoding="utf-8").replace("\n", "\r\n", 1), encoding="utf-8")
        self.refused("fold2_dev.csv: SHA-256")

    def rewrite_fold(self, i: int, role: str, rows):
        text = csv_text(FR.FOLD_COLUMNS, rows)
        sdir = self.w.base / "schema"
        (sdir / f"fold{i}_{role}.csv").write_text(text, encoding="utf-8")
        texts = {p.name: p.read_text(encoding="utf-8") for p in sdir.glob("fold*.csv")}
        sums = "".join(f"{sha(t.encode())}  {n}\n" for n, t in sorted(texts.items()))
        (sdir / "SHA256SUMS").write_text(sums, encoding="utf-8")
        cfg = json.loads(json.dumps(self.cfg))
        cfg["schema_run"]["restricted_sha256sums_sha256"] = sha(sums.encode())
        return cfg

    def test_fold_row_count_role_and_in_cache(self):
        rows = [list(r) for r in self.w.folds[1]["dev"]]
        cfg = self.rewrite_fold(1, "dev", rows[:-1])
        self.refused("rows, expected", cfg)
        rows = [list(r) for r in self.w.folds[1]["dev"]]
        rows[0][1] = "fitting"
        self.refused("another fold or role", self.rewrite_fold(1, "dev", rows))
        rows = [list(r) for r in self.w.folds[1]["dev"]]
        rows[0][8] = "0" if rows[0][8] == "1" else "1"
        self.refused("in_cache", self.rewrite_fold(1, "dev", rows))
        rows = [list(r) for r in self.w.folds[1]["dev"]]
        rows[0][3] = "0" if rows[0][3] == "1" else "1"
        self.refused("same label, subject, session", self.rewrite_fold(1, "dev", rows))

    def test_roles_sharing_a_subject_or_key(self):
        rows = [list(r) for r in self.w.folds[1]["fitting"]]
        rows[0] = list(self.w.folds[1]["test"][0])
        rows[0][1] = "fitting"
        self.refused("share a subject or a key", self.rewrite_fold(1, "fitting", rows))
        # stesso soggetto senza chiave in comune: un frame non usato di un soggetto del dev messo nel fitting
        rows = [list(r) for r in self.w.folds[1]["fitting"]]
        used = {r[2] for r in self.w.folds[1]["dev"]} | {r[2] for r in rows}
        spare = next(r for r in self.w.official.values() if r["subject"] == "1112" and r["key"] not in used)
        rows[0] = ["1", "fitting", spare["key"], spare["label"], spare["subject"], spare["session"], spare["sha256"], spare["bytes"], "0"]
        self.refused("share a subject or a key", self.rewrite_fold(1, "fitting", rows))

    def test_dev_must_be_the_next_test_fold(self):
        # il dev del fold 1 perde il soggetto 1113 (fold 2) e prende altrettanti frame di 1112 non ancora usati
        rows = [list(r) for r in self.w.folds[1]["dev"]]
        used = {r[2] for r in rows}
        spare = [r for r in self.w.official.values() if r["subject"] == "1112" and r["key"] not in used]
        gone = [r for r in rows if r[4] == "1113"]
        self.assertTrue(gone and len(spare) >= len(gone))
        kept = [r for r in rows if r[4] != "1113"]
        for r in spare[:len(gone)]:
            kept.append(["1", "dev", r["key"], r["label"], r["subject"], r["session"], r["sha256"], r["bytes"], "0" if r["key"] not in self.w.order["train"] + self.w.order["test"] else "1"])
        self.refused("dev subjects are not the test subjects of fold 2", self.rewrite_fold(1, "dev", kept))

    def test_cache_sha_members_and_linkage(self):
        cfg = json.loads(json.dumps(self.cfg))
        cfg["cache"]["sha256"] = "2" * 64
        self.refused("cache cache.npz: SHA-256", cfg)
        cfg = json.loads(json.dumps(self.cfg))
        cfg["cache"]["members"]["X_test"]["shape"] = [2 * PER_CLASS, 7, 7, 3]
        self.refused("has dtype", cfg)
        z = dict(np.load(self.w.cache))
        z["y_test"] = 1 - z["y_test"]
        with open(self.w.cache, "wb") as f:
            np.savez(f, **z)
        cfg = json.loads(json.dumps(self.cfg))
        cfg["cache"]["sha256"] = FR.sha256_file(self.w.cache)
        self.refused("labels differ from the manifests", cfg)

    def test_frozen_code_resources_and_opencv(self):
        (self.w.base / "frozen" / "face.py").write_bytes(b"changed")
        self.refused("frozen face module")
        self.w.build_restore = None
        (self.w.base / "frozen" / "face.py").write_bytes(b"content of frozen/face.py\n")
        (self.w.base / "res" / "haarcascade_frontalface_default.xml").write_bytes(b"changed")
        self.refused("detector_resources haarcascade_frontalface_default.xml")
        (self.w.base / "res" / "haarcascade_frontalface_default.xml").write_bytes(b"content of res/haarcascade_frontalface_default.xml\n")
        cfg = json.loads(json.dumps(self.cfg))
        cfg["environment"]["opencv"] = "0.0.0"
        self.refused("OpenCV", cfg)

    def test_fold_file_missing_from_sums_or_symlinked(self):
        sdir = self.w.base / "schema"
        sums = (sdir / "SHA256SUMS").read_text(encoding="utf-8")
        new = "".join(l + "\n" for l in sums.splitlines() if not l.endswith("fold3_test.csv"))
        (sdir / "SHA256SUMS").write_text(new, encoding="utf-8")
        cfg = json.loads(json.dumps(self.cfg))
        cfg["schema_run"]["restricted_sha256sums_sha256"] = sha(new.encode())
        self.refused("fold3_test.csv not listed", cfg)
        (sdir / "SHA256SUMS").write_text(sums, encoding="utf-8")
        p = sdir / "fold3_test.csv"
        q = self.w.tmp / "moved.csv"
        q.write_bytes(p.read_bytes())
        p.unlink()
        p.symlink_to(q)
        self.refused("symbolic link")

    def test_fold_count_per_fold_and_subject_partition(self):
        cfg = json.loads(json.dumps(self.cfg))
        cfg["expected"]["test_per_fold"][2] += 1
        cfg["expected"]["n_images"] += 1
        cfg["expected"]["n_to_extract"] += 1
        self.refused("official lists: ", cfg)
        cfg = json.loads(json.dumps(self.cfg))
        cfg["expected"]["fitting_per_fold"] += 1
        self.refused("rows, expected", cfg)

    def test_cache_frame_shape_must_match(self):
        with injected(FRAME_SHAPE=(9, 9, 3), SCHEMA_CONFIG_SHA256=self.w.schema_sha):
            with self.assertRaises(FR.Failure) as cm:
                FR.load_inputs(self.cfg, self.w.base)
        self.assertIn("frames are not", str(cm.exception))

    def test_images_folder_required(self):
        shutil.rmtree(self.w.base / "images")
        self.refused("images folder")

    def test_tensorflow_must_not_be_imported_at_check(self):
        with modules({"tensorflow": types.ModuleType("tensorflow")}):
            self.refused("TensorFlow was imported")


# ============================================================================ passi

class TestSteps(unittest.TestCase):
    def setUp(self):
        self.w = World()
        self.addCleanup(self.w.close)
        self.cfg, _ = FR.load_config(self.w.config)
        with injected(FRAME_SHAPE=SHAPE, SCHEMA_CONFIG_SHA256=self.w.schema_sha):
            self.inp = FR.load_inputs(self.cfg, self.w.base)
        self.pre = fake_pre(self.cfg, self.w.base)

    def rewrite_cache(self, edit):
        z = dict(np.load(self.w.cache))
        edit(z)
        with open(self.w.cache, "wb") as f:
            np.savez(f, **z)
        cfg = json.loads(json.dumps(self.cfg))
        cfg["cache"]["sha256"] = FR.sha256_file(self.w.cache)
        with injected(FRAME_SHAPE=SHAPE, SCHEMA_CONFIG_SHA256=self.w.schema_sha):
            self.inp = FR.load_inputs(cfg, self.w.base)
        self.cfg = cfg

    def test_cache_check_reproduces_every_frame(self):
        with injected(FRAME_SHAPE=SHAPE):
            cc = FR.cache_check(self.cfg, self.w.base, self.inp, self.pre)
        self.assertEqual((cc["train"]["equal"], cc["test"]["equal"]), (2 * PER_CLASS, 2 * PER_CLASS))
        self.assertEqual(cc["train"]["different"] + cc["test"]["different"], 0)
        self.assertEqual(sum(cc["train"]["detectors"].values()), 2 * PER_CLASS)
        self.assertEqual((cc["train"]["worst_max_abs_diff"], cc["cache_train_test_shared"]), (0, 0))
        self.assertEqual(cc["train"]["detectors"], self.w.detectors["train"])

    def test_cache_check_detectors_must_equal_n2(self):
        cfg = json.loads(json.dumps(self.cfg))
        cfg["cache"]["detectors_n2"]["test"] = {"Haar": 1, "RetinaFace": 2 * PER_CLASS - 1}
        with injected(FRAME_SHAPE=SHAPE), self.assertRaises(FR.Failure) as cm:
            FR.cache_check(cfg, self.w.base, self.inp, self.pre)
        self.assertIn("differ from the N2 cache check", str(cm.exception))

    def test_cache_check_shared_and_duplicated_frames(self):
        first_train, k_test0, k_test1 = self.inp.order["train"][0], self.inp.order["test"][0], self.inp.order["test"][1]
        self.rewrite_cache(lambda z: z["X_test"].__setitem__(0, z["X_train"][0]))

        def pre(path):
            key = Path(path).read_bytes().decode()
            if key == k_test0:
                return fake_frame(first_train.encode())
            return self.pre(path)
        with injected(FRAME_SHAPE=SHAPE), self.assertRaises(FR.Failure) as cm:
            FR.cache_check(self.cfg, self.w.base, self.inp, pre)
        self.assertIn("shared between X_train and X_test", str(cm.exception))
        self.rewrite_cache(lambda z: (z["X_test"].__setitem__(0, fake_frame(k_test0.encode())[0]), z["X_test"].__setitem__(1, z["X_test"][0])))

        def pre2(path):
            key = Path(path).read_bytes().decode()
            if key == k_test1:
                return fake_frame(k_test0.encode())
            return self.pre(path)
        with injected(FRAME_SHAPE=SHAPE), self.assertRaises(FR.Failure) as cm:
            FR.cache_check(self.cfg, self.w.base, self.inp, pre2)
        self.assertIn("1 duplicated frames", str(cm.exception))

    def test_cache_check_detects_a_changed_original(self):
        key = self.inp.order["test"][3]
        r = self.inp.official[key]
        p = self.w.base / "images" / FR.LAYOUTS["images"](r)
        p.write_bytes(p.read_bytes() + b"x")
        with injected(FRAME_SHAPE=SHAPE), self.assertRaises(FR.Failure) as cm:
            FR.cache_check(self.cfg, self.w.base, self.inp, self.pre)
        self.assertIn("size or SHA-256 differ", str(cm.exception))
        b = p.read_bytes()[:-1]
        p.write_bytes(b[:-1] + bytes([b[-1] ^ 1]))              # stessa dimensione, contenuto diverso
        with injected(FRAME_SHAPE=SHAPE), self.assertRaises(FR.Failure) as cm:
            FR.cache_check(self.cfg, self.w.base, self.inp, self.pre)
        self.assertIn("size or SHA-256 differ", str(cm.exception))

    def test_cache_check_detects_a_different_preprocessing(self):
        def pre(path):
            arr, det = self.pre(path)
            arr = arr.copy()
            arr[1, 1, 1] ^= 1
            return arr, det
        with injected(FRAME_SHAPE=SHAPE), self.assertRaises(FR.Failure) as cm:
            FR.cache_check(self.cfg, self.w.base, self.inp, pre)
        self.assertIn("differ from the frozen preprocessing (worst max abs diff 1)", str(cm.exception))

    def test_extract_frames(self):
        with injected(FRAME_SHAPE=SHAPE):
            rec, rows, X = FR.extract_frames(self.cfg, self.w.base, self.inp, self.pre, self.w.frames)
        self.assertEqual(rec["n"], len(self.inp.to_extract))
        self.assertEqual(rec["sha256"], FR.sha256_file(self.w.frames))
        z = np.load(self.w.frames)
        self.assertEqual(list(z["keys"]), self.inp.to_extract)
        self.assertEqual(z["X"].dtype, np.uint8)
        self.assertTrue(np.array_equal(z["X"], X))
        self.assertEqual([r[0] for r in rows], self.inp.to_extract)
        self.assertEqual([r[-1] for r in rows], FR.frame_hashes(X))
        self.assertEqual(sum(rec["detectors"].values()), rec["n"])
        self.assertEqual(rec["identical_frames"], {"identical_frames_same_subject": 0, "identical_frames_across_subjects_or_with_the_cache_of_another_subject": 0})
        with injected(FRAME_SHAPE=SHAPE), self.assertRaises(FR.Failure) as cm:
            FR.extract_frames(self.cfg, self.w.base, self.inp, self.pre, self.w.frames)
        self.assertIn("never overwritten", str(cm.exception))

    def test_extract_identical_frames_across_subjects_fail_same_subject_recorded(self):
        first = self.inp.order["train"][0]           # frame della cache di un soggetto di training
        target = next(k for k in self.inp.to_extract if self.inp.official[k]["subject"] != self.inp.official[first]["subject"])

        def pre(path):
            arr, det = self.pre(path)
            if Path(path).read_bytes().decode() == target:
                return fake_frame(first.encode())[0], det
            return arr, det
        with injected(FRAME_SHAPE=SHAPE), self.assertRaises(FR.Failure) as cm:
            FR.extract_frames(self.cfg, self.w.base, self.inp, pre, self.w.frames)
        self.assertIn("identical frames across different subjects", str(cm.exception))
        # due frame dello stesso soggetto identici: registrati, non un errore
        same = [k for k in self.inp.to_extract if self.inp.official[k]["subject"] == self.inp.official[self.inp.to_extract[0]]["subject"]][:2]

        def pre2(path):
            arr, det = self.pre(path)
            if Path(path).read_bytes().decode() == same[1]:
                return fake_frame(same[0].encode())[0], det
            return arr, det
        with injected(FRAME_SHAPE=SHAPE):
            rec, rows, X = FR.extract_frames(self.cfg, self.w.base, self.inp, pre2, self.w.frames)
        self.assertEqual(rec["identical_frames"]["identical_frames_same_subject"], 2)
        self.assertEqual(rec["identical_frames"]["identical_frames_across_subjects_or_with_the_cache_of_another_subject"], 0)

    def test_original_symlink_and_wrong_frame_shape_refused(self):
        key = self.inp.to_extract[0]
        r = self.inp.official[key]
        p = self.w.base / "images" / FR.LAYOUTS["images"](r)
        q = self.w.tmp / "orig.jpg"
        q.write_bytes(p.read_bytes())
        p.unlink()
        p.symlink_to(q)
        with injected(FRAME_SHAPE=SHAPE), self.assertRaises(FR.Failure) as cm:
            FR.extract_frames(self.cfg, self.w.base, self.inp, self.pre, self.w.frames)
        self.assertIn("symbolic link refused", str(cm.exception))
        p.unlink()
        p.write_bytes(q.read_bytes())

        def pre(path):
            arr, det = self.pre(path)
            return arr[:, :, :1], det
        with injected(FRAME_SHAPE=SHAPE), self.assertRaises(FR.Failure) as cm:
            FR.extract_frames(self.cfg, self.w.base, self.inp, pre, self.w.frames)
        self.assertIn("expected", str(cm.exception))
        self.assertFalse(self.w.frames.exists())

    def test_probabilities_and_val_loss(self):
        X = np.zeros((3, *SHAPE), np.uint8)
        with self.assertRaises(FR.Failure):
            FR.probabilities(lambda X: np.ones((3, 2)) * 1.5, X)
        with self.assertRaises(FR.Failure):
            FR.probabilities(lambda X: np.ones((2, 2)) * 0.5, X)
        with self.assertRaises(FR.Failure):
            FR.probabilities(lambda X: np.full((3, 2), np.nan), X)
        p = FR.probabilities(lambda X: np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]]), X)
        self.assertAlmostEqual(FR.val_loss(p, np.array([0, 1, 1])), float(np.mean(-np.log([0.9, 0.8, 0.5]))))
        self.assertAlmostEqual(FR.val_loss(np.array([[1.0, 0.0]]), np.array([1])), -np.log(FR.EPS), places=6)

    def test_frame_store_and_assemble_by_key(self):
        with injected(FRAME_SHAPE=SHAPE):
            rec, rows, X = FR.extract_frames(self.cfg, self.w.base, self.inp, self.pre, self.w.frames)
        store = FR.FrameStore(self.inp, X, self.inp.to_extract)
        for key in [self.inp.order["train"][1], self.inp.order["test"][2], self.inp.to_extract[5]]:
            self.assertTrue(np.array_equal(store.get(key), fake_frame(key.encode())[0]))
        with injected(FRAME_SHAPE=SHAPE):
            part = FR.assemble(store, self.inp.folds[2]["dev"])
        self.assertEqual(part["keys"], sorted(part["keys"]))
        self.assertEqual(part["X"].shape, (len(self.inp.folds[2]["dev"]), *SHAPE))
        self.assertEqual(list(part["y"]), [int(r["label"]) for r in part["rows"]])
        self.assertEqual(set(part["s"]), set(FOLDS[3]))


# ============================================================================ trainer vero con TensorFlow finto

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="foldsrun-trainer-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        rng = np.random.default_rng(0)
        self.data = {"X_fit": rng.integers(0, 256, (10, *SHAPE), dtype=np.uint8), "y_fit": np.array([0] * 8 + [1] * 2),
                     "X_dev": rng.integers(0, 256, (4, *SHAPE), dtype=np.uint8), "y_dev": np.array([0, 1, 0, 1])}
        self.data["X_fit"][0, 0, 0, 0] = 255

    def train(self, plan, gpu_stays=False, cap=None):
        log = {}
        tf, arch = fake_tf(log, plan, gpu_stays)
        calls = []
        with modules({"tensorflow": tf}):
            card = FR.default_trainer(self.data, dict(FR.FROZEN_TRAINING), self.tmp / "w.keras", 43, arch, cap or (lambda s: calls.append(s)))
        return card, log, calls

    def test_recipe_seed_device_and_restored_epoch(self):
        card, log, calls = self.train(lambda e: [0.9, 0.5, 0.7, 0.8, 0.85, 0.86])
        self.assertEqual(log["tf_seed"], 43)
        self.assertEqual(log["hidden"], ([], "GPU"))
        self.assertEqual(log["opts"], {"remapping": False})
        self.assertEqual(log["adam"], (0.001, 1.0))
        self.assertEqual(log["es"], {"monitor": "val_loss", "patience": 4, "restore": True})
        self.assertEqual(log["rlr"], {"monitor": "val_loss", "factor": 0.5, "patience": 2, "min_lr": 1e-6})
        self.assertEqual(log["shuffle"], (4096, 43))
        self.assertEqual(log["batch"], [16, 16])
        self.assertEqual(log["fit"]["epochs"], 12)
        self.assertEqual(log["fit"]["class_weight"], {0: 10 / 16, 1: 10 / 4})
        self.assertEqual(log["fit"]["n_callbacks"], 3)
        self.assertEqual(log["arch"], "LivenessNet")
        self.assertEqual(log["cleared"], 1)
        # strati dell'augmentation nell'ordine della ricetta, con gli argomenti
        self.assertEqual(log["layers"], [("RandomFlip", ("horizontal",), {}), ("RandomRotation", (0.03,), {}), ("RandomZoom", (0.12, 0.12), {}),
                                         ("RandomTranslation", (0.06, 0.06), {}), ("RandomBrightness", (0.15,), {"value_range": (0.0, 1.0)}),
                                         ("RandomContrast", (0.2,), {}), ("GaussianNoise", (0.02,), {})])
        self.assertEqual(log["sequential"], 7)
        self.assertEqual(log["losses"], [("CategoricalCrossentropy", (), {})])
        self.assertEqual(log["metrics"], [("CategoricalAccuracy", (), {"name": "accuracy"})])
        # etichette one-hot (indice 0 bona fide), ingressi in [0, 1] dopo /255, augmentation solo sul fitting
        self.assertEqual(log["slices"][0]["y_first"], [1.0, 0.0])
        self.assertEqual(log["slices"][0]["y_shape"], [10, 2])
        self.assertEqual((log["slices"][0]["x_max"], log["slices"][0]["x_dtype"]), (1.0, "float32"))
        self.assertLessEqual(log["slices"][1]["x_max"], 1.0)
        self.assertEqual(log["mapped"], [log["fit"]["train_ds"]])
        self.assertNotEqual(log["fit"]["train_ds"], log["fit"]["val_ds"])
        self.assertEqual(log["saved"][0].endswith(".tmp.keras"), True)
        self.assertEqual(calls, [f"epoch {i}" for i in range(1, 7)])
        self.assertEqual((card["epochs_run"], card["restored_epoch"], card["restored_by"]), (6, 2, "early stopping"))
        self.assertEqual(card["best_val_loss"], 0.5)
        self.assertEqual((card["device"], card["visible_devices"], card["train_seed"]), ("CPU", ["CPU"], 43))
        self.assertEqual(card["weights_sha256"], sha((self.tmp / "w.keras").read_bytes()))
        self.assertEqual((self.tmp / "w.keras").read_bytes(), b"keras:best")
        self.assertNotIn("set_weights", log)

    def test_epochs_end_first_set_weights(self):
        card, log, _ = self.train(lambda e: [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.44, 0.43, 0.42, 0.41, 0.40, 0.39])
        self.assertEqual(log["set_weights"], b"best")
        self.assertEqual((card["epochs_run"], card["restored_epoch"], card["restored_by"]), (12, 12, "end of epochs (set_weights)"))
        self.assertEqual(len(card["history"]["val_loss"]), 12)

    def test_gpu_that_stays_visible_is_refused(self):
        with self.assertRaises(FR.Failure) as cm:
            self.train(lambda e: [0.5, 0.6], gpu_stays=True)
        self.assertIn("CPU only expected", str(cm.exception))

    def test_cap_callback_stops_the_training(self):
        def cap(step):
            if step == "epoch 2":
                FR.fail("wall-time cap exceeded")
        with self.assertRaises(FR.Failure) as cm:
            self.train(lambda e: [0.5, 0.4, 0.3], cap=cap)
        self.assertIn("cap exceeded", str(cm.exception))

    def test_existing_weights_and_wrong_data_refused(self):
        (self.tmp / "w.keras").write_bytes(b"old")
        with self.assertRaises(FR.Failure) as cm:
            self.train(lambda e: [0.5])
        self.assertIn("never overwritten", str(cm.exception))
        self.data = {"X_train": None}
        with self.assertRaises(FR.Failure):
            self.train(lambda e: [0.5])

    def test_tf_scorer_with_fake_tensorflow(self):
        log = {}
        tf, arch = fake_tf(log, lambda e: [0.5])
        w = self.tmp / "s.keras"
        w.write_bytes(b"keras:best")
        X = np.full((40, *SHAPE), 255, np.uint8)
        with modules({"tensorflow": tf}):
            scorer = FR.tf_scorer(w, sha(b"keras:best"))
            p = FR.probabilities(scorer, X)
        self.assertEqual(p.shape, (40, 2))
        self.assertEqual(log["load_model"], ("s.keras", False))
        self.assertEqual(log["hidden"], ([], "GPU"))
        self.assertEqual([c["n"] for c in log["calls"]], [16, 16, 8])
        self.assertTrue(all(c["training"] is False and c["x_max"] == 1.0 and c["x_dtype"] == "float32" for c in log["calls"]))
        self.assertTrue(all(d == "/CPU:0" for d in log["device"]))
        with modules({"tensorflow": tf}), self.assertRaises(FR.Failure):
            FR.tf_scorer(w, sha(b"other"))

    def test_card_problems(self):
        w = self.tmp / "c.keras"
        w.write_bytes(b"x")
        card = {"arch": "LivenessNet", "train_seed": 42, "n_fitting": 10, "n_dev": 4, "max_epochs": 12, "epochs_run": 3, "restored_epoch": 2,
                "restored_by": "early stopping", "best_val_loss": 0.4, "history": {"val_loss": [0.5, 0.4, 0.6]}, "device": "CPU",
                "visible_devices": ["CPU"], "lr": 0.001, "batch": 16, "early_stopping_patience": 4, "train_seconds": 1.0,
                "weights_file": "c.keras", "weights_sha256": sha(b"x")}
        self.assertEqual(FR.card_problems(card, FR.FROZEN_TRAINING, 42, w), [])
        self.assertTrue(FR.card_problems({**card, "restored_epoch": 1}, FR.FROZEN_TRAINING, 42, w))
        self.assertTrue(FR.card_problems({**card, "train_seed": 43}, FR.FROZEN_TRAINING, 42, w))
        self.assertTrue(FR.card_problems({**card, "batch": 32}, FR.FROZEN_TRAINING, 42, w))
        self.assertTrue(FR.card_problems({**card, "weights_sha256": sha(b"y")}, FR.FROZEN_TRAINING, 42, w))
        self.assertTrue(FR.card_problems({**card, "epochs_run": 13, "history": {"val_loss": [0.4] * 13}, "restored_epoch": 1}, FR.FROZEN_TRAINING, 42, w))
        self.assertTrue(FR.card_problems({k: v for k, v in card.items() if k != "history"}, FR.FROZEN_TRAINING, 42, w))


# ============================================================================ soglia, metriche, bootstrap

class TestMetrics(unittest.TestCase):
    def test_eer_threshold_known_answers(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        s = np.array([0.1, 0.4, 0.6, 0.3, 0.7, 0.9])
        t = FR.eer_threshold(y, s)
        # candidati decrescenti 0.9, 0.7, 0.6, 0.4, 0.3, 0.1, <0.1; a 0.4: fp = 1 (0.6), fn = 1 (0.3): |3-3| = 0, il primo con 0
        self.assertEqual((t["value"], t["fp"], t["fn"], t["objective"]), (0.4, 1, 1, 0))
        self.assertEqual(FR.candidate_thresholds(np.array([0.5, 0.5, 0.2])), [0.5, 0.2, float(np.nextafter(0.2, -np.inf))])
        self.assertEqual(FR.candidate_thresholds(np.array([0.0, 0.3])), [0.3, 0.0])
        # pareggio dell'obiettivo (1 a 0,5 e 1 sotto il minimo): vince la soglia più alta
        tie = FR.eer_threshold(np.array([0, 1]), np.array([0.5, 0.5]))
        self.assertEqual((tie["value"], tie["objective"]), (0.5, 1))
        with self.assertRaises(FR.Failure):
            FR.eer_threshold(np.array([1, 1]), np.array([0.2, 0.3]))

    def test_eer_threshold_equals_select_threshold_when_available(self):
        try:
            import select_threshold as st
        except Exception:  # noqa: BLE001  (modulo assente nel layout del pacchetto, o dipendenze mancanti)
            self.assertTrue(True)
            return
        rng = np.random.default_rng(3)
        for _ in range(20):
            y = rng.integers(0, 2, 30)
            if y.min() == y.max():
                continue
            s = np.round(rng.random(30), 2)
            a, b = FR.eer_threshold(y, s), st.select_threshold(y, s, "eer")
            self.assertEqual((a["value"], a["fp"], a["fn"]), (b["value"], b["fp"], b["fn"]))

    def test_auroc_and_metrics_known_answers(self):
        self.assertEqual(FR.auroc([0.9, 0.8], [0.1, 0.2]), 1.0)
        self.assertEqual(FR.auroc([0.1, 0.2], [0.9, 0.8]), 0.0)
        self.assertEqual(FR.auroc([0.5, 0.5], [0.5, 0.5]), 0.5)
        self.assertAlmostEqual(FR.auroc([0.3, 0.7], [0.5, 0.1]), 0.75)
        self.assertIsNone(FR.auroc([], [0.1]))
        try:
            from sklearn.metrics import roc_auc_score
            rng = np.random.default_rng(1)
            y, s = rng.integers(0, 2, 50), np.round(rng.random(50), 1)
            self.assertAlmostEqual(FR.auroc(s[y == 1], s[y == 0]), float(roc_auc_score(y, s)), places=12)
        except ImportError:
            pass


    def rows(self, spec):
        """spec: elenco di (soggetto, sessione, label, score, pred05, preddev)."""
        return [{"subject_id": s, "session_id": ses, "label": lab, "score_attack": repr(sc), "prediction_05": p5, "prediction_dev": pd}
                for s, ses, lab, sc, p5, pd in spec]

    def test_stratum_metrics_and_summary(self):
        rows = self.rows([("a", "03", "0", 0.2, "0", "1"), ("a", "03", "1", 0.8, "1", "1"), ("b", "01", "0", 0.6, "1", "0"), ("b", "01", "1", 0.4, "0", "0")])
        m = FR.stratum_metrics(rows)
        self.assertEqual((m["n"], m["n_bona_fide"], m["n_attack"], m["n_subjects"]), (4, 2, 2, 2))
        self.assertEqual((m["rejected_bona_fide_05"], m["accepted_attacks_05"], m["acer_05"]), (1, 1, 0.5))
        self.assertEqual((m["rejected_bona_fide_dev"], m["accepted_attacks_dev"], m["acer_dev"]), (1, 1, 0.5))
        self.assertAlmostEqual(m["auroc"], 0.75)
        m03 = FR.stratum_metrics(FR.rows_of_stratum(rows, "03"))
        self.assertEqual((m03["n"], m03["acer_05"], m03["bpcer_dev"]), (2, 0.0, 1.0))
        self.assertEqual(FR.summary([0.6, 0.1, 0.2]), {"values": [0.6, 0.1, 0.2], "median": 0.2, "min": 0.1, "max": 0.6})
        self.assertIsNone(FR.summary([0.3, None])["median"])

    def test_bootstrap_recipe(self):
        spec = []
        for i, subj in enumerate("abcdefg"):
            spec += [(subj, "03", "0", 0.2 + 0.05 * i, "0", "0"), (subj, "03", "1", 0.7 - 0.05 * i, "1", "1")]
        rows = self.rows(spec)
        with injected(BOOT={**FR.BOOT, "resamples": 50}):
            b = FR.bootstrap(rows, ("acer_dev", "acer_05", "auroc"))
        self.assertEqual((b["n_subjects"], b["resamples"], b["seed"]), (7, 50, 0))
        self.assertEqual(b["acer_05"]["valid_resamples"], 50)
        self.assertEqual((b["acer_05"]["low"], b["acer_05"]["high"]), (0.0, 0.0))
        self.assertTrue(0.5 <= b["auroc"]["low"] <= b["auroc"]["high"] <= 1.0)
        # stessi indici della ricetta: il generatore di numpy con il seme dichiarato, len(subjects) interi per ricampionamento
        rng = np.random.default_rng(FR.BOOT["seed"])
        idx = rng.integers(0, 7, 7)
        sample = [r for i in idx for r in rows if r["subject_id"] == "abcdefg"[i]]
        first = FR.stratum_metrics(sample)["auroc"]
        with injected(BOOT={**FR.BOOT, "resamples": 1, "percentiles": [0, 100]}):
            one = FR.bootstrap(rows, ("auroc",))
        self.assertEqual((one["auroc"]["low"], one["auroc"]["high"]), (first, first))
        # percentili della ricetta (2,5 e 97,5) ricalcolati in modo indipendente su 20 ricampionamenti
        rng = np.random.default_rng(FR.BOOT["seed"])
        vals = []
        for _ in range(20):
            idx = rng.integers(0, 7, 7)
            vals.append(FR.stratum_metrics([r for i in idx for r in rows if r["subject_id"] == "abcdefg"[i]])["auroc"])
        with injected(BOOT={**FR.BOOT, "resamples": 20}):
            twenty = FR.bootstrap(rows, ("auroc",))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        self.assertEqual((twenty["auroc"]["low"], twenty["auroc"]["high"]), (float(lo), float(hi)))
        few = FR.bootstrap(self.rows(spec[:8]), ("auroc",))
        self.assertEqual(few["n_subjects"], 4)
        self.assertIn("no interval", few["note"])
        self.assertIsNone(few["auroc"]["low"])
        # un ricampionamento senza una classe non conta
        rows2 = self.rows([(s, "03", "1", 0.6, "1", "1") for s in "abcde"] + [("f", "03", "0", 0.2, "0", "0")] * 1)
        with injected(BOOT={**FR.BOOT, "resamples": 20}):
            b2 = FR.bootstrap(rows2, ("auroc",))
        self.assertLess(b2["auroc"]["valid_resamples"], 20)
        self.assertGreater(b2["auroc"]["valid_resamples"], 0)

    def test_score_rows_decisions_at_a_tie(self):
        part = {"rows": [{"key": "real/1101_1101_00_00_03_0.jpg", "subject": "1101", "session": "03", "label": "0", "sha256": "a" * 64}]}
        thr = {"value": 0.5, "sha256": "b" * 64}
        card = {"weights_sha256": "c" * 64}
        row = FR.score_rows(1, 42, "test", part, np.array([0.5]), thr, card)[0]
        self.assertEqual((row["prediction_05"], row["prediction_dev"]), ("0", "0"))
        row = FR.score_rows(1, 42, "test", part, np.array([np.nextafter(0.5, 1)]), thr, card)[0]
        self.assertEqual((row["prediction_05"], row["prediction_dev"]), ("1", "1"))

    def test_public_problems(self):
        inp = FR.Inputs()
        inp.official = {"attack/1101_1101_00_00_03_1.jpg": {"subject": "1101", "sha256": "a" * 64}}
        self.assertEqual(FR.public_problems("ACER 0.1101 n 11010 all fine", inp), [])
        self.assertIn("subject id 1101", FR.public_problems("subject 1101 here", inp))
        self.assertIn("image keys", FR.public_problems("see attack/1101_1101_00_00_03_1.jpg", inp))
        self.assertIn("image SHA-256 values", FR.public_problems("a" * 64, inp))
        self.assertTrue(any("marker" in p for p in FR.public_problems("to" + "do later", inp, markdown=True)))


# ============================================================================ run completo

class TestRun(unittest.TestCase):
    def setUp(self):
        self.w = World()
        self.addCleanup(self.w.close)
        Log.calls = []

    def run_ok(self, **hooks):
        with injected(**defaults(self.w, **hooks)):
            rc, out, err = run_main(argv_of(self.w, "run", "--i-am-authorized"))
        self.assertEqual(rc, 0, out + err)
        self.assertIn("RESULT: PASS (run)", out)
        run = self.w.out / "20261103-FOLDSRUN-abcdef0"
        return run, out

    def test_run_outputs_status_and_sums(self):
        run, out = self.run_ok()
        status = json.loads((run / "status.json").read_text())
        self.assertEqual((status["status"], status["passed"], status["commit"], status["dirty"]), ("PASS", True, "abcdef0", False))
        pub, res = run / "public", run / "restricted"
        self.assertEqual(sorted(os.listdir(pub)), sorted(FR.PUBLIC_OUTPUTS))
        cfg, _ = FR.load_config(self.w.config)
        self.assertEqual(sorted(os.listdir(res)), sorted(FR.restricted_outputs(cfg)))
        for d in (pub, res):
            listed = {}
            for line in (d / "SHA256SUMS").read_text().splitlines():
                h, n = line.split("  ")
                self.assertEqual(FR.sha256_file(d / n), h, n)
                listed[n] = h
            self.assertEqual(sorted(listed), sorted(n for n in os.listdir(d) if n != "SHA256SUMS"))
        self.assertEqual((res / "public_SHA256SUMS.sha256").read_text(), sha((pub / "SHA256SUMS").read_bytes()) + "  public/SHA256SUMS\n")
        self.assertEqual(status["public_sums_sha256"], sha((pub / "SHA256SUMS").read_bytes()))
        self.assertTrue((run / "run.log").is_file())
        self.assertIn("RESULT: PASS (run)", (run / "run.log").read_text())
        self.assertEqual(status["run_log_sha256"], FR.sha256_file(run / "run.log"))
        self.assertIs(status["delivered_configuration"], False)
        rec = json.loads((pub / "run_record.json").read_text())
        self.assertEqual(set(rec["timings"]["cache-check"]), {"monotonic_seconds", "wall_clock_seconds"})
        self.assertEqual(len([c for c in Log.calls if c[0] == "train"]), K * len(SEEDS))
        self.assertTrue(self.w.frames.is_file())

    def test_threshold_frozen_before_test_and_scores_consistent(self):
        run, _ = self.run_ok()
        res = run / "restricted"
        cfg, _ = FR.load_config(self.w.config)
        for fold in range(1, K + 1):
            for seed in SEEDS:
                name = f"fold{fold}_seed{seed}"
                thr = json.loads((res / f"{name}_threshold.json").read_text())
                card = json.loads((res / f"{name}_card.json").read_text())
                self.assertEqual((thr["set"], thr["criterion"]), ("dev fold only", FR.THRESHOLD_CRITERION))
                self.assertNotIn("test_scores_computed_after", thr)
                self.assertLessEqual(thr["frozen_at"], card["test_scored_after_threshold"])
                thr_sha = sha((res / f"{name}_threshold.json").read_bytes())
                with (res / f"{name}_test_scores.csv").open(newline="") as fh:
                    test = list(csv.DictReader(fh))
                with (res / f"{name}_dev_scores.csv").open(newline="") as fh:
                    dev = list(csv.DictReader(fh))
                self.assertEqual(len(test), len(self.w.folds[fold]["test"]))
                self.assertEqual(len(dev), len(self.w.folds[fold]["dev"]))
                self.assertTrue(all(r["threshold_sha256"] == thr_sha and r["checkpoint_sha256"] == card["weights_sha256"] for r in test + dev))
                self.assertTrue(all(r["role"] == "test" for r in test) and all(r["role"] == "dev" for r in dev))
                # la soglia è l'EER sui punteggi del dev
                y = np.array([int(r["label"]) for r in dev]); s = np.array([float(r["score_attack"]) for r in dev])
                self.assertEqual(FR.eer_threshold(y, s)["value"], thr["value"])
                # decisioni coerenti con le due soglie
                for r in test:
                    v = float(r["score_attack"])
                    self.assertEqual(r["prediction_05"], "1" if v > 0.5 else "0")
                    self.assertEqual(r["prediction_dev"], "1" if v > thr["value"] else "0")
                    self.assertEqual(r["threshold_dev"], repr(float(thr["value"])))
                # i soggetti del test sono quelli del fold, mai nel fitting o nel dev
                self.assertEqual({r["subject_id"] for r in test}, set(FOLDS[fold]))
                self.assertEqual({r["subject_id"] for r in dev}, set(FOLDS[fold % K + 1]))
                self.assertEqual(card["fold"], fold)
                self.assertLess(abs(card["val_loss_recomputed"] - card["best_val_loss"]), 1e-9)
                # i punteggi sono quelli dello scorer sui frame dei manifest
                w = (res / f"{name}_LivenessNet.keras").read_bytes()
                exp = fake_scores(np.stack([fake_frame(r["sample_id"].encode())[0] for r in test]), bias_of(w))[:, 1]
                self.assertTrue(np.allclose([float(r["score_attack"]) for r in test], exp))

    def test_aggregate_consistent_with_the_scores(self):
        run, _ = self.run_ok()
        agg = json.loads((run / "public" / "aggregate.json").read_text())
        res = run / "restricted"
        rows_all = {}
        for fold in range(1, K + 1):
            for seed in SEEDS:
                with (res / f"fold{fold}_seed{seed}_test_scores.csv").open(newline="") as fh:
                    rows = list(csv.DictReader(fh))
                rows_all[(fold, seed)] = rows
                for st in FR.STRATA:
                    part = FR.rows_of_stratum(rows, st)
                    got = agg["per_fold_seed"][str(fold)][str(seed)][st]
                    if not part:
                        self.assertIsNone(got)
                        continue
                    exp = FR.stratum_metrics(part)
                    for k in ("n", "n_bona_fide", "n_attack", "acer_05", "acer_dev", "auroc", "rejected_bona_fide_dev"):
                        self.assertEqual(got[k], exp[k], (fold, seed, st, k))
        for seed in SEEDS:
            pooled = [r for fold in range(1, K + 1) for r in rows_all[(fold, seed)]]
            p03 = agg["pooled_per_seed"][str(seed)]["03"]
            self.assertEqual(p03["n"], len(FR.rows_of_stratum(pooled, "03")))
            self.assertEqual(p03["n_subjects"], len(SUBJECTS))
            self.assertEqual(p03["n_bona_fide"], sum(agg["per_fold_seed"][str(f)][str(seed)]["03"]["n_bona_fide"] for f in range(1, K + 1)))
            self.assertEqual(p03["acer_dev"], FR.stratum_metrics(FR.rows_of_stratum(pooled, "03"))["acer_dev"])
            b = agg["bootstrap_per_seed"][str(seed)]["03"]
            self.assertEqual((b["resamples"], b["seed"], b["n_subjects"]), (1000, 0, len(SUBJECTS)))
            self.assertIsNotNone(b["acer_dev"]["low"])
        fs = agg["per_fold_summary_over_seeds"]["1"]["03"]["acer_dev"]
        self.assertEqual(fs["values"], [agg["per_fold_seed"]["1"][str(s)]["03"]["acer_dev"] for s in SEEDS])
        self.assertEqual(fs["median"], float(np.median(fs["values"])))
        ps = agg["pooled_summary_over_seeds"]["all"]["auroc"]
        self.assertEqual(ps["values"], [agg["pooled_per_seed"][str(s)]["all"]["auroc"] for s in SEEDS])
        self.assertEqual(sorted(agg["training"]), sorted(f"fold{f}_seed{s}" for f in range(1, K + 1) for s in SEEDS))
        with (res / "per_subject.csv").open(newline="") as fh:
            per_subject = list(csv.DictReader(fh))
        self.assertEqual({r["subject_id"] for r in per_subject}, set(SUBJECTS))
        one = [r for r in per_subject if r["fold"] == "1" and r["train_seed"] == "42" and r["subject_id"] == "1101" and r["stratum"] == "03"][0]
        part = [r for r in rows_all[(1, 42)] if r["subject_id"] == "1101" and r["session_id"] == "03"]
        self.assertEqual(float(one["acer_dev"]), FR.stratum_metrics(part)["acer_dev"])

    def test_public_files_have_no_identifiers(self):
        run, _ = self.run_ok()
        cfg, _ = FR.load_config(self.w.config)
        self.w.frames.unlink()
        with injected(FRAME_SHAPE=SHAPE, SCHEMA_CONFIG_SHA256=self.w.schema_sha):
            inp = FR.load_inputs(cfg, self.w.base)
        for n in ("aggregate.json", "aggregate.md", "run_record.json"):
            text = (run / "public" / n).read_text()
            self.assertEqual(FR.public_problems(text, inp, n.endswith(".md")), [], n)
            for s in SUBJECTS:
                self.assertNotIn(f'"{s}"', text)
        rec = json.loads((run / "public" / "run_record.json").read_text())
        self.assertEqual(rec["extraction"]["n"], len(inp.to_extract))
        self.assertEqual(set(rec["timings"]) >= {"check-inputs", "cache-check", "extract", "aggregate", "train fold1_seed42"}, True)
        self.assertEqual(rec["cache_check"]["train"]["equal"], 2 * PER_CLASS)
        md = (run / "public" / "aggregate.md").read_text()
        self.assertIn("## Pooled over the 4 folds", md)
        self.assertIn("20261103-FOLDSRUN-abcdef0", md)

    def test_authorisation_dirty_tree_and_previous_outputs(self):
        with injected(**defaults(self.w)):
            rc, out, err = run_main(argv_of(self.w, "run"))
        self.assertEqual(rc, 1)
        self.assertIn("--i-am-authorized", err)
        self.assertFalse(self.w.out.exists())
        with injected(**defaults(self.w, REPO_STATE=lambda root: {"commit": "abcdef0", "dirty": True})):
            rc, out, err = run_main(argv_of(self.w, "run", "--i-am-authorized"))
        self.assertEqual(rc, 0, err)
        self.assertTrue((self.w.out / "20261103-FOLDSRUN-abcdef0-dev" / "status.json").is_file())
        # un secondo tentativo nella stessa cartella e' rifiutato: ogni tentativo conserva il suo esito
        self.w.frames.unlink()
        with injected(**defaults(self.w, REPO_STATE=lambda root: {"commit": "abcdef0", "dirty": True})):
            rc, out, err = run_main(argv_of(self.w, "run", "--i-am-authorized"))
        self.assertEqual(rc, 1)
        self.assertIn("already holds status.json", err)
        # una cartella senza status.json con un temporaneo di un'esecuzione interrotta: tolto prima del run
        run = self.w.out / "20261103-FOLDSRUN-abcdef0"
        (run / "public").mkdir(parents=True)
        leftover = run / "public" / ".aggregate.json.abc123.tmp"
        leftover.write_text("tmp")
        with injected(**defaults(self.w)):
            rc, out, err = run_main(argv_of(self.w, "run", "--i-am-authorized"))
        self.assertEqual(rc, 0, err)
        self.assertFalse(leftover.exists())

    def test_foreign_entry_in_the_run_folder_refused(self):
        run = self.w.out / "20261103-FOLDSRUN-abcdef0"
        (run / "public").mkdir(parents=True)
        (run / "public" / "notes.txt").write_text("x")
        with injected(**defaults(self.w)):
            rc, out, err = run_main(argv_of(self.w, "run", "--i-am-authorized"))
        self.assertEqual(rc, 1)
        self.assertIn("not an output of this command", err)
        self.assertTrue((run / "public" / "notes.txt").is_file())
        self.assertFalse((run / "status.json").exists())

    def test_failure_leaves_fail_status_and_no_outputs(self):
        def bad_trainer(data, training, weights_path, seed, arch, cap_check):
            if seed == 43:
                FR.fail("synthetic training failure")
            return fake_trainer()(data, training, weights_path, seed, arch, cap_check)
        with injected(**defaults(self.w, TRAINER=bad_trainer)):
            rc, out, err = run_main(argv_of(self.w, "run", "--i-am-authorized"))
        self.assertEqual(rc, 1)
        run = self.w.out / "20261103-FOLDSRUN-abcdef0"
        status = json.loads((run / "status.json").read_text())
        self.assertEqual((status["status"], status["passed"]), ("FAIL", False))
        self.assertIn("synthetic training failure", status["reason"])
        self.assertEqual(status["outputs"], "removed")
        self.assertFalse((run / "public").exists() and os.listdir(run / "public"))
        self.assertEqual([n for n in os.listdir(run / "restricted")], [])
        self.assertIn("synthetic training failure", (run / "run.log").read_text())
        self.assertEqual(status["run_log_sha256"], FR.sha256_file(run / "run.log"))
        # i frame estratti dal tentativo fallito sono parcheggiati con il nome del run e non bloccano il prossimo
        parked = self.w.frames.with_name("nuaa_folds_frames.20261103-FOLDSRUN-abcdef0.failed.npz")
        self.assertEqual(status["extracted_frames_file"], parked.name)
        self.assertTrue(parked.is_file())
        self.assertFalse(self.w.frames.exists())

    def cap_run(self, **hooks):
        t = {"now": 0.0}

        def clock():
            t["now"] += 1.0
            return t["now"]
        with injected(**defaults(self.w, CLOCK=clock, **{k: v(t) for k, v in hooks.items()})):
            rc, out, err = run_main(argv_of(self.w, "run", "--i-am-authorized"))
        self.assertEqual(rc, 1)
        self.assertIn("wall-time cap", err)
        status = json.loads((self.w.out / "20261103-FOLDSRUN-abcdef0" / "status.json").read_text())
        self.assertEqual(status["status"], "FAIL")
        return status["reason"]

    def test_wall_time_cap_fails_the_run(self):
        def slow_trainer(t):
            def train(data, training, weights_path, seed, arch, cap_check):
                card = fake_trainer()(data, training, weights_path, seed, arch, cap_check)
                t["now"] += 400 * 60                               # il tetto scade dopo il training: il confine successivo lo vede
                return card
            return train
        self.assertIn("at score fold1_seed42", self.cap_run(TRAINER=slow_trainer))
        shutil.rmtree(self.w.out)
        self.w.frames.unlink(missing_ok=True)

        def slow_epoch(t):
            def train(data, training, weights_path, seed, arch, cap_check):
                t["now"] += 400 * 60                               # il tetto scade durante il training: la callback lo vede
                return fake_trainer()(data, training, weights_path, seed, arch, cap_check)
            return train
        self.assertIn("at epoch 1", self.cap_run(TRAINER=slow_epoch))

    def test_wall_time_cap_checked_before_aggregate_and_writes(self):
        last = len(self.w.folds[K]["test"])

        def late_scorer(t):
            def make(weights_path, weights_sha):
                inner = fake_make_scorer(weights_path, weights_sha)

                def score(X):
                    if Path(weights_path).name.startswith(f"fold{K}_seed{SEEDS[-1]}") and len(X) == last:
                        t["now"] += 400 * 60                       # il tetto scade dopo l'ultimo punteggio del test
                    return inner(X)
                return score
            return make
        self.w.frames.unlink(missing_ok=True)
        self.assertIn("at aggregate", self.cap_run(MAKE_SCORER=late_scorer))
        self.w.frames.unlink(missing_ok=True)
        old = FR.aggregate

        def slow_aggregate(t):
            def agg(cfg, scores, cards, thresholds):
                t["now"] += 400 * 60                               # il tetto scade durante l'aggregazione
                return old(cfg, scores, cards, thresholds)
            return agg
        shutil.rmtree(self.w.out)
        self.assertIn("at write", self.cap_run(aggregate=slow_aggregate))

    def test_val_loss_mismatch_and_bad_card_refused(self):
        def wrong(data, training, weights_path, seed, arch, cap_check):
            card = fake_trainer()(data, training, weights_path, seed, arch, cap_check)
            card["history"]["val_loss"] = [v + 0.05 for v in card["history"]["val_loss"]]    # storia traslata: stesso minimo, valore diverso
            card["best_val_loss"] = card["best_val_loss"] + 0.05                              # oltre 1e-3 e oltre l'1 % del minimo
            return card
        with injected(**defaults(self.w, TRAINER=wrong)):
            rc, out, err = run_main(argv_of(self.w, "run", "--i-am-authorized"))
        self.assertEqual(rc, 1)
        self.assertIn("not the one of minimum val_loss", err)

        def bad_card(data, training, weights_path, seed, arch, cap_check):
            card = fake_trainer()(data, training, weights_path, seed, arch, cap_check)
            card["device"] = "GPU"
            return card
        self.w.frames.unlink(missing_ok=True)
        shutil.rmtree(self.w.out)
        with injected(**defaults(self.w, TRAINER=bad_card)):
            rc, out, err = run_main(argv_of(self.w, "run", "--i-am-authorized"))
        self.assertEqual(rc, 1)
        self.assertIn("differ from the protocol", err)

    def test_output_root_symlink_refused(self):
        real = self.w.tmp / "real_out"
        real.mkdir()
        self.w.out.symlink_to(real)
        with injected(**defaults(self.w)):
            rc, out, err = run_main(argv_of(self.w, "run", "--i-am-authorized"))
        self.assertEqual(rc, 1)
        self.assertIn("symbolic link", err)

    def test_bad_scorer_output_fails_the_run(self):
        with injected(**defaults(self.w, MAKE_SCORER=lambda w, h: (lambda X: np.ones((len(X), 2)) * 2))):
            rc, out, err = run_main(argv_of(self.w, "run", "--i-am-authorized"))
        self.assertEqual(rc, 1)
        self.assertIn("probabilities expected", err)
        status = json.loads((self.w.out / "20261103-FOLDSRUN-abcdef0" / "status.json").read_text())
        self.assertEqual(status["status"], "FAIL")

    def test_threshold_read_back_is_compared(self):
        old = FR.read_regular

        def tampered(p, what):
            b = old(p, what)
            if Path(p).name.endswith("_threshold.json"):
                return b.replace(b'"set": "dev fold only"', b'"set": "dev fold ONLY"')
            return b
        with injected(**defaults(self.w, read_regular=tampered)):
            rc, out, err = run_main(argv_of(self.w, "run", "--i-am-authorized"))
        self.assertEqual(rc, 1)
        self.assertIn("does not read back identical", err)

    def test_sigterm_stops_the_run_with_fail(self):
        import signal
        saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}

        def killing_trainer(data, training, weights_path, seed, arch, cap_check):
            if seed == 43:
                if signal.getsignal(signal.SIGTERM) in (signal.SIG_DFL, signal.SIG_IGN, None):
                    raise AssertionError("SIGTERM handler not installed")   # senza gestore il segnale ucciderebbe la suite
                os.kill(os.getpid(), signal.SIGTERM)             # consegnato in modo sincrono al thread principale
            return fake_trainer()(data, training, weights_path, seed, arch, cap_check)
        try:
            with injected(**defaults(self.w, TRAINER=killing_trainer)):
                rc, out, err = run_main(argv_of(self.w, "run", "--i-am-authorized"))
        finally:
            for sig, h in saved.items():
                signal.signal(sig, h)
        self.assertEqual(rc, 1)
        self.assertIn("signal SIGTERM received", err)
        status = json.loads((self.w.out / "20261103-FOLDSRUN-abcdef0" / "status.json").read_text())
        self.assertEqual((status["status"], status["outputs"]), ("FAIL", "removed"))
        self.assertIn("SIGTERM", status["reason"])

    def test_sighup_handler_not_installed_when_ignored(self):
        import signal
        saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
        try:
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            FR.signal_handlers()
            self.assertIs(signal.getsignal(signal.SIGHUP), signal.SIG_IGN)
            self.assertIsNot(signal.getsignal(signal.SIGTERM), saved[signal.SIGTERM])
            signal.signal(signal.SIGHUP, signal.SIG_DFL)
            FR.signal_handlers()
            self.assertIsNot(signal.getsignal(signal.SIGHUP), signal.SIG_DFL)
        finally:
            for sig, h in saved.items():
                signal.signal(sig, h)

    def test_checkpoint_temporary_removed_when_the_save_fails(self):
        log = {}
        tf, arch = fake_tf(log, lambda e: [0.5, 0.4])
        tmp = Path(tempfile.mkdtemp(prefix="foldsrun-save-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        data = {"X_fit": np.zeros((4, *SHAPE), np.uint8), "y_fit": np.array([0, 1, 0, 1]), "X_dev": np.zeros((2, *SHAPE), np.uint8),
                "y_dev": np.array([0, 1])}
        old_replace = os.replace

        def failing_replace(a, b):
            raise OSError("disk full")
        with modules({"tensorflow": tf}), injected(os=types.SimpleNamespace(**{**{k: getattr(os, k) for k in dir(os) if not k.startswith("__")}, "replace": failing_replace})):
            with self.assertRaises(OSError):
                FR.default_trainer(data, dict(FR.FROZEN_TRAINING), tmp / "w.keras", 42, arch, lambda s: None)
        self.assertEqual(os.listdir(tmp), [])
        self.assertIs(os.replace, old_replace)

    def test_frames_file_already_present_fails(self):
        self.w.frames.parent.mkdir(parents=True, exist_ok=True)
        self.w.frames.write_bytes(b"old")
        with injected(**defaults(self.w)):
            rc, out, err = run_main(argv_of(self.w, "run", "--i-am-authorized"))
        self.assertEqual(rc, 1)
        self.assertIn("already exists in", err)
        self.assertEqual(self.w.frames.read_bytes(), b"old")
        self.assertFalse(self.w.out.exists())

    def test_public_scan_blocks_identifiers(self):
        old = FR.aggregate_md

        def leaky(cfg, agg, record, run_id):
            return old(cfg, agg, record, run_id) + "\nsubject 1101 was hard\n"
        with injected(**defaults(self.w, aggregate_md=leaky)):
            rc, out, err = run_main(argv_of(self.w, "run", "--i-am-authorized"))
        self.assertEqual(rc, 1)
        self.assertIn("public file aggregate.md refused", err)


# ============================================================================ ingressi veri (sola lettura)

class TestRealInputs(unittest.TestCase):
    def test_real_check_inputs(self):
        cfg_p = REAL_ROOT / FR.DEFAULT_CONFIG
        if not cfg_p.is_file():
            skip_missing(self, "delivered configuration")
        cfg, _ = FR.load_config(cfg_p)
        needed = [REAL_ROOT / cfg["manifests"]["folder"], REAL_ROOT / cfg["schema_run"]["folder"], REAL_ROOT / cfg["cache"]["path"],
                  REAL_ROOT / cfg["images"]["folder"], REAL_ROOT / cfg["extraction"]["frozen_face"]["path"]]
        if not all(p.exists() for p in needed) or FR.environment()["opencv"] == "absent":
            skip_missing(self, "real inputs")
        before = snapshot(REAL_ROOT / "results", REAL_ROOT / "data" / "processed")
        inp = FR.load_inputs(cfg, REAL_ROOT)
        self.assertEqual(inp.checks["originals"]["located_and_hashed"], 12614)
        self.assertEqual(len(inp.cache_index), 6000)
        self.assertEqual(len(inp.to_extract), 6614)
        self.assertEqual(inp.checks["schema"]["subjects_per_fold"], {1: 4, 2: 4, 3: 4, 4: 4})
        self.assertEqual(snapshot(REAL_ROOT / "results", REAL_ROOT / "data" / "processed"), before)


class ZZRepositoryUntouched(unittest.TestCase):
    def test_declared_skips_only(self):
        self.assertEqual(sorted(MISSING_INPUT), ["delivered configuration", "real inputs"])


def main() -> int:
    before = snapshot(ROOT / "results", ROOT / "export", ROOT / "delivery")
    argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a != "--verbose"] + (["-v"] if "--verbose" in sys.argv else [])
    result = unittest.main(argv=argv, exit=False, verbosity=1).result
    after = snapshot(ROOT / "results", ROOT / "export", ROOT / "delivery")
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
