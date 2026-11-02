"""
Test di scripts/c1crop_check.py su dati SINTETICI, in cartelle temporanee: nessuna rete, nessuna immagine vera, nessun
dato NUAA, nessun TensorFlow, nessun rilevatore (pre-elaborazione, scorer, figura e stato git sono iniettati).

Fixture: manifest NUAA finti con lo schema M2 (official_test, cnn_test, c1), file "originali" finti (byte con SHA-256 e
dimensione nel manifest), una cache npz finta con X_test costruito dai frame finti dell'estrattore della cache
nell'ordine di campionamento, un file dei pesi finto, due copie finte di face.py (contenuti diversi), le risorse del
rilevatore e il codice di riferimento finti, gli score C1 finti (schema del gruppo) coerenti con lo scorer finto e una
configurazione coerente. I run sono -trial, salvo i test del vincolo sulla configurazione consegnata.

  python scripts/check_c1crop.py        # uscita 1 se un test fallisce; ultima riga RESULT: PASS / FAIL

Dipendenze: numpy e scikit-learn (OpenCV facoltativo, come in check_n2_sessions.py).
"""
import atexit
import contextlib
import copy
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
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
STUDY = "20261103-C1CROP-abcdef0-trial"
STUDY_REAL = "20261103-C1CROP-abcdef0"
COMMIT = "abcdef0" + "0" * 33

TMP_ROOT = Path(tempfile.mkdtemp(prefix="c1crop-tests-"))
atexit.register(shutil.rmtree, TMP_ROOT, True)
for _k, _s in (("LIVENESSLAB_DATA_DIR", "data"), ("LIVENESSLAB_RESULTS_DIR", "results"), ("LIVENESSLAB_WEIGHTS_DIR", "weights"),
               ("LIVENESSLAB_LIVEDETECTION", "livedetection"), ("LIVENESSLAB_SILENT_FACE", "silent_face")):
    os.environ[_k] = str(TMP_ROOT / _s)
    (TMP_ROOT / _s).mkdir()
FAKE_ARCH = b'''# fake reference architectures: the "model" returns the mean of the input as the attack probability
import numpy as np


class _Model:
    def __init__(self):
        self.loaded = None
        self.seen = []

    def load_weights(self, p):
        self.loaded = p

    def __call__(self, xb, training=False):
        x = np.asarray(xb, dtype=np.float64)
        self.seen.append(x)
        m = x.reshape(len(x), -1).mean(axis=1)
        return np.stack([1 - m, m], axis=1)


class _Arch:
    def __init__(self):
        self.model = _Model()

    def get_model(self):
        return self.model


LAST = []


def create_model(name):
    assert name == "AttackNetV2_1", name
    a = _Arch()
    LAST.append(a.model)
    return a
'''
REF_CODE = {("scripts", "create_datasets.py"): b"# fake reference create_datasets\n", ("src", "architectures.py"): FAKE_ARCH}
for _rel, _data in REF_CODE.items():
    _p = TMP_ROOT / "livedetection" / Path(*_rel)
    _p.parent.mkdir(parents=True, exist_ok=True)
    _p.write_bytes(_data)
REPO_BEFORE = (ROOT / "results" / "c1crop").exists()
REPO_LISTING = sorted(p.name for p in (ROOT / "results" / "c1crop").iterdir()) if REPO_BEFORE else None
MISSING_INPUT = {"delivered configuration": "c1crop.json and the two face copies are not next to the code (delivery/c1crop/, or ../ in the package)"}

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
import c1crop_check as C  # noqa: E402
import n2_sessions as N2  # noqa: E402
import n1_train as N1  # noqa: E402
import nuaa_manifests as nm  # noqa: E402

PER_CLASS = 12
SUBJECTS = tuple(f"{i:04d}" for i in range(1, 9))
PNG = b"\x89PNG\r\n\x1a\n" + b"fake"
C1_RUN = "20260928-C1-seed42-a80f6f9"
C1_CFG_SHA = hashlib.sha256(b"fake c1 configuration").hexdigest()
C1_COLUMNS = ["run_id", "sample_id", "dataset_id", "split", "subject_id", "session_id", "video_id", "attack_type", "modality", "label", "score_attack",
              "threshold", "prediction", "checkpoint_sha256", "config_sha256", "analyzer_id"]
FACE_TEXT = {C.EXTRACTORS[0]: "# fake cache extractor (0ac7143)\nclass FaceDetector:\n    def detect(self, img, *a, **k):\n        return None\n",
             C.EXTRACTORS[1]: "# fake C1 extractor (a80f6f9)\nclass FaceDetector:\n    def detect(self, img, *a, **k):\n        return None\n"}
FACE_FILES = {C.EXTRACTORS[0]: "face_0ac7143.py", C.EXTRACTORS[1]: "face_a80f6f9.py"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_of(key: str) -> bytes:
    return f"image:{key}".encode()


def row(label: int, subject: str, session: str, pic: int) -> dict:
    name = f"{subject}_{subject}_0{pic % 2}_0{(pic % 3) if label else 0}_{session}_{pic}.jpg"
    key = nm.make_key(label, name)
    data = content_of(key)
    return {"key": key, "label": str(label), **nm.parse_name(name), "official_split": nm.SESSION_SPLIT[session], "sha256": sha(data), "bytes": str(len(data))}


def csv_text(rows, cols) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def write_csv(p: Path, rows) -> None:
    p.write_text(csv_text(sorted(rows, key=lambda r: r["key"]), nm.COLUMNS), encoding="utf-8")


def det_of_name(fname: str) -> str:
    """Rilevatore finto dal numero dell'immagine (ultimo campo del nome): 3 nessun volto, 2 Haar, altrimenti RetinaFace."""
    pic = int(fname.rsplit("_", 1)[1].split(".")[0])
    return "none" if pic == 3 else "Haar" if pic == 2 else "RetinaFace"


def frame_of(data: bytes, det: str, extractor: str):
    """Frame finto: diverso fra i due estrattori solo quando il volto è RetinaFace (come nella realtà: Haar e immagine
    intera non cambiano); riquadro 1 px più stretto e più basso con l'estrattore C1."""
    tag = extractor.encode() if det == "RetinaFace" else b"shared"
    h = hashlib.sha256(tag + b":" + data).digest()
    X = np.frombuffer((h * 6144)[:256 * 256 * 3], dtype=np.uint8).reshape(256, 256, 3).copy()
    if det == "none":
        return X, det, None
    d = 1 if (det == "RetinaFace" and extractor == C.EXTRACTORS[1]) else 0
    return X, det, {"x": 10, "y": 12, "w": 40 - d, "h": 44 - d, "confidence": 0.9 if det == "RetinaFace" else 0.5}


def fake_preprocessor(name, face_path, cfg):
    def pre(path):
        p = Path(path)
        return frame_of(p.read_bytes(), det_of_name(p.name), name)
    return pre


def fake_prob(X, weights_sha: str) -> np.ndarray:
    out = np.zeros((len(X), 2), dtype=np.float64)
    for i, x in enumerate(np.asarray(X)):
        v = int.from_bytes(hashlib.sha256(np.ascontiguousarray(x[:8, :8]).tobytes() + weights_sha.encode()).digest()[:4], "big") / 2 ** 32
        out[i] = (1 - v, v)
    return out


def fake_scorer(path, weights_sha):
    return lambda X: fake_prob(X, weights_sha)


def fake_plotter(doc, pts):
    assert set(pts) == {"bona_fide", "attack"} and all(len(v["score_c1"]) == len(v["score_cache"]) for v in pts.values())
    return {"figure_paired.png": PNG}


def fake_repo_state(d):
    return {"commit": COMMIT, "dirty": False, "dirty_paths": [], "dirty_excludes": None}


class Fixture:
    def __init__(self, base: Path):
        self.base = base
        self.man, self.images = base / "manifests", base / "images"
        for d in (self.man, self.images):
            d.mkdir(parents=True)
        te = [row(lab, s, "03", p) for s in SUBJECTS for lab in (0, 1) for p in range(4)]
        by_te = {r["key"]: r for r in te}
        self.order = nm.spread_keys(te, PER_CLASS)
        self.rows = {"official_test": te, "cnn_test": [by_te[k] for k in self.order], "c1": [by_te[k] for i, k in enumerate(self.order) if i % 3 == 0 or i % 7 == 1]}
        self.c1 = sorted(self.rows["c1"], key=lambda r: r["key"])
        self.files = {"official_test": "nuaa_official_test.csv", "cnn_test": "nuaa_cnn_test.csv", "c1": "nuaa_c1_v2.csv"}
        for n, rows in self.rows.items():
            write_csv(self.man / self.files[n], rows)
        for r in te:
            p = self.path_of(r)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content_of(r["key"]))
        self.weights = base / "AttackNetV2_1__nuaa.h5"
        self.weights.write_bytes(b"fake AttackNetV2_1 weights " * 50)
        self.faces = {}
        for n, fn in FACE_FILES.items():
            self.faces[n] = base / "extractors" / fn
            self.faces[n].parent.mkdir(exist_ok=True)
            self.faces[n].write_text(FACE_TEXT[n], encoding="utf-8")
        self.resources = {}
        for n in ("deploy.prototxt", "Widerface-RetinaFace.caffemodel", "haarcascade_frontalface_default.xml"):
            p = base / "resources" / n
            p.parent.mkdir(exist_ok=True)
            p.write_bytes(f"fake {n}".encode())
            self.resources[n] = p
        self.cache = base / "nuaa_1500.npz"
        self.write_cache()
        self.c1_scores = base / "c1" / "scores.csv"
        self.c1_scores.parent.mkdir()
        self.write_c1_scores()
        self.config = self.make_config()
        self.config_path = base / "c1crop.json"
        self.write_config()

    def path_of(self, r: dict) -> Path:
        return self.images / N2.LAYOUTS["images"](r)

    def frame(self, r: dict, extractor: str):
        return frame_of(content_of(r["key"]), det_of_name(nm.name_of(r["key"])), extractor)

    def write_cache(self, x_test=None, extra=None) -> None:
        by = {r["key"]: r for r in self.rows["official_test"]}
        if x_test is None:
            x_test = np.stack([self.frame(by[k], C.EXTRACTORS[0])[0] for k in self.order])
        n = len(self.order)
        arrays = {"X_train": np.zeros((n, 256, 256, 3), np.uint8), "y_train": np.zeros(n, np.int64), "s_train": np.array(["0001"] * n),
                  "X_test": x_test, "y_test": np.array([int(by[k]["label"]) for k in self.order], np.int64), "s_test": np.array([by[k]["subject"] for k in self.order])}
        arrays.update(extra or {})
        np.savez(self.cache, **arrays)

    def c1_score_rows(self) -> list:
        wsha = sha(self.weights.read_bytes())
        out = []
        for r in self.c1:
            v = float(fake_prob(self.frame(r, C.EXTRACTORS[1])[0][None], wsha)[0, 1])
            out.append({"run_id": C1_RUN, "sample_id": r["key"], "dataset_id": "nuaa", "split": "test", "subject_id": r["subject"], "session_id": "00",
                        "video_id": "", "attack_type": "print" if r["label"] == "1" else "", "modality": "RGB", "label": r["label"], "score_attack": repr(v),
                        "threshold": "0.5", "prediction": "1" if v > 0.5 else "0", "checkpoint_sha256": wsha, "config_sha256": C1_CFG_SHA,
                        "analyzer_id": "attacknet_v2_1__nuaa"})
            out.append({**out[-1], "score_attack": repr(1 - v), "prediction": "1" if 1 - v > 0.5 else "0", "checkpoint_sha256": sha(b"other"),
                        "analyzer_id": "livenessnet__nuaa"})
        return out

    def write_c1_scores(self, rows=None) -> None:
        self.c1_scores.write_text(csv_text(self.c1_score_rows() if rows is None else rows, C1_COLUMNS), encoding="utf-8")

    def make_config(self) -> dict:
        mans = {n: {"file": self.files[n], "sha256": sha((self.man / self.files[n]).read_bytes()), **N1.counts_of(self.rows[n])} for n in C.MANIFESTS}
        n = len(self.order)
        members = {"X_train": {"dtype": "|u1", "shape": [n, 256, 256, 3]}, "y_train": {"dtype": "<i8", "shape": [n]}, "s_train": {"dtype": "<U4", "shape": [n]},
                   "X_test": {"dtype": "|u1", "shape": [n, 256, 256, 3]}, "y_test": {"dtype": "<i8", "shape": [n]}, "s_test": {"dtype": "<U4", "shape": [n]}}
        return {"schema": C.CONFIG_SCHEMA, "study": "synthetic", "architecture": "AttackNetV2_1", "dataset": "nuaa", "protocol": "official",
                "weights": {"file": self.weights.name, "sha256": sha(self.weights.read_bytes()), "bytes": self.weights.stat().st_size, "source": "synthetic",
                            "loading": "synthetic", "classes": {"0": "bona fide", "1": "attack"}, "score": "column 1"},
                "extractors": {n: {"file": FACE_FILES[n], "commit": ("0ac7143" if i == 0 else "a80f6f9") + "f" * 33, "sha256": sha(self.faces[n].read_bytes()),
                                   "role": "synthetic", "retinaface_box": "synthetic"} for i, n in enumerate(C.EXTRACTORS)},
                "difference": "synthetic", "chain": "synthetic",
                "detector_resources": {n: sha(p.read_bytes()) for n, p in self.resources.items()},
                "reference_code_sha256": {k: sha(REF_CODE[tuple(k.split("/")[1:])]) for k in C.REFERENCE_CODE},
                "scoring": {"device": "CPU", "batch": 16, "batchnorm": "inference", "input": "/255", "score": "column 1"},
                "metrics": {"threshold": 0.5, "decision": C.DECISION_RULE, "threshold_origin": "fixed", "reported": ["APCER", "BPCER", "ACER", "AUROC"],
                            "set": "c1_300", "paired": "s", "public": "aggregate", "restricted": "rows"},
                "cache": {"file": self.cache.name, "sha256": sha(self.cache.read_bytes()), "per_class": PER_CLASS, "members": members, "rule": "bit for bit"},
                "c1_reference": {"run_id": C1_RUN, "file": "scores.csv", "sha256": sha(self.c1_scores.read_bytes()), "analyzer_id": "attacknet_v2_1__nuaa",
                                 "rows": len(self.c1), "config_sha256": C1_CFG_SHA, "columns": C1_COLUMNS, "device": "synthetic", "session_id": "not compared",
                                 "parity": {"tolerance_abs": 1e-5, "rule": "within tolerance, same decision"}},
                "data": {"m2_run": "synthetic", "manifests": mans, "originals": "synthetic"}}

    def write_config(self, cfg=None, path=None) -> str:
        cfg = self.config if cfg is None else cfg
        p = self.config_path if path is None else path
        raw = (json.dumps(cfg, indent=1) + "\n").encode()
        p.write_bytes(raw)
        return sha(raw)

    def refresh_config(self) -> None:
        self.config = self.make_config()
        self.write_config()

    def args(self, study=None, **over) -> list:
        base = {"config": self.config_path, "manifests": self.man, "images": self.images, "cache": self.cache, "weights": self.weights,
                "c1_scores": self.c1_scores, "face_cache": self.faces[C.EXTRACTORS[0]], "face_c1": self.faces[C.EXTRACTORS[1]],
                "study_dir": self.base / "out" / (study or STUDY)}
        base.update(over)
        out = []
        for k, v in base.items():
            out += [f"--{k.replace('_', '-')}", str(v)]
        return out


def run(*args) -> tuple:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = C.main(list(args))
    return code, out.getvalue()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="c1crop-case-", dir=TMP_ROOT))
        self.fx = Fixture(self.tmp)
        self.study = self.tmp / "out" / STUDY
        self.patches = [mock.patch.object(C, "PREPROCESSOR", fake_preprocessor), mock.patch.object(N2, "RESOURCE_PATHS", self.fx.resources),
                        mock.patch.object(C, "PLOTTER", fake_plotter), mock.patch.object(C, "MAKE_SCORER", fake_scorer),
                        mock.patch.object(C, "MAKE_SCORER_C1", fake_scorer), mock.patch.object(C, "MAKE_SCORER_CPU1", fake_scorer),
                        mock.patch.object(C, "default_device", lambda: "FAKE"),
                        mock.patch.object(C, "CURRENT_FACE", str(self.fx.faces[C.EXTRACTORS[1]])), mock.patch.object(N1, "REPO_STATE", fake_repo_state)]
        for p in self.patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self.patches])
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def go(self, cmd, *extra, study=None, expect=0, **over) -> str:
        code, text = run(cmd, *self.fx.args(study, **over), *extra)
        self.assertEqual(code, expect, text)
        self.assertIn(f"RESULT: {'PASS' if expect == 0 else 'FAIL'} ({cmd})", text)
        return text

    def pipeline(self, upto="aggregate") -> str:
        text = self.go("check")
        for step in ("frames", "parity", "score", "aggregate"):
            if upto == "check" or (step == "parity" and upto == "frames") or (step == "score" and upto == "parity") or (step == "aggregate" and upto == "score"):
                return text
            text = self.go(step, *(["--i-am-authorized"] if step == "score" else []))
        return text

    def listing(self, d=None) -> list:
        d = d or self.study
        return sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()) if d.exists() else []

    def status(self, d=None) -> dict:
        return json.loads(((d or self.study) / "status.json").read_text(encoding="utf-8"))

    def clear(self) -> None:
        shutil.rmtree(self.study, ignore_errors=True)

    def refuse_check(self, needle: str, **over) -> str:
        """check rifiutato: nessun check.json; la cartella (log e stato) viene tolta per la prova successiva."""
        text = self.go("check", expect=1, **over)
        self.assertIn(needle, text)
        self.assertFalse((self.study / "check.json").exists(), text)
        self.clear()
        return text

    def record(self, name: str) -> dict:
        return json.loads((self.study / name).read_text(encoding="utf-8"))

    def redeclare(self, record: str, name: str, text: str) -> None:
        """Riscrive un file di uscita e aggiorna il suo SHA-256 nel record del passo (manomissione coerente)."""
        (self.study / name).write_text(text, encoding="utf-8")
        doc = self.record(record)
        doc["files"][name] = sha(text.encode())
        (self.study / record).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


class FixtureTests(Base):
    def test_fixture_covers_every_detector(self):
        dets = {det_of_name(nm.name_of(r["key"])) for r in self.fx.c1}
        self.assertEqual(dets, {"RetinaFace", "Haar", "none"})
        self.assertGreaterEqual(len(self.fx.c1), 6)
        self.assertEqual({r["label"] for r in self.fx.c1}, {"0", "1"})


class ConfigTests(Base):
    def test_check_pass(self):
        text = self.go("check")
        self.assertEqual(self.listing(), ["check.json", "check.log", "status.json"])
        doc = self.record("check.json")
        self.assertEqual(doc["checks"]["originals"]["n_verified"], len(self.fx.c1))
        self.assertEqual(doc["checks"]["c1_reference"]["rows"], len(self.fx.c1))
        self.assertEqual(doc["checks"]["cache"]["sha256"], self.fx.config["cache"]["sha256"])
        self.assertEqual(doc["detector"], "not used")
        self.assertEqual(doc["weights"], "not loaded")
        self.assertEqual(doc["tensorflow"], "not imported")
        self.assertEqual({k for k in doc["code_sha256"] if k.startswith("extractor:")}, {f"extractor:{n}" for n in C.EXTRACTORS})
        self.assertEqual(self.status()["result"], "PASS")
        self.assertIn("check PASS", text)

    def refused(self, mutate, needle):
        cfg = copy.deepcopy(self.fx.config)
        mutate(cfg)
        self.fx.write_config(cfg)
        text = self.go("check", expect=1)
        self.assertIn(needle, text)
        self.assertFalse(self.study.exists(), text)

    def test_fixed_rules(self):
        self.refused(lambda c: c.pop("difference"), "missing or not allowed")
        self.refused(lambda c: c.update(schema="x"), "schema")
        self.refused(lambda c: c.update(architecture="LivenessNet"), "architecture AttackNetV2_1")
        self.refused(lambda c: c["weights"].update(classes={"0": "attack", "1": "bona fide"}), "classes 0 bona fide")
        self.refused(lambda c: c["extractors"].pop(C.EXTRACTORS[1]), "extractors must be exactly")
        self.refused(lambda c: c["extractors"][C.EXTRACTORS[0]].update(commit="0ac7143"), "commit (40 hex)")
        self.refused(lambda c: c["extractors"][C.EXTRACTORS[0]].update(sha256=c["extractors"][C.EXTRACTORS[1]]["sha256"]), "must be different files")
        self.refused(lambda c: c["metrics"].update(threshold=0.6), "threshold")
        self.refused(lambda c: c["metrics"].update(decision="attack if score_attack >= 0.5"), "decision rule")
        self.refused(lambda c: c["metrics"].update(set="test_3000"), "set c1_300")
        self.refused(lambda c: c["scoring"].update(device="default"), "CPU")
        self.refused(lambda c: c["cache"]["members"]["X_test"].update(dtype="<f4"), "X_test uint8")
        self.refused(lambda c: c["cache"].pop("rule"), "the rule")
        self.refused(lambda c: c["c1_reference"]["parity"].update(tolerance_abs=1e-4), "tolerance_abs in (0, 1e-5]")
        self.refused(lambda c: c["c1_reference"]["parity"].update(tolerance_abs=1), "tolerance_abs in (0, 1e-5]")
        self.refused(lambda c: c["c1_reference"].update(analyzer_id="livenessnet__nuaa"), "analyzer_id attacknet_v2_1__nuaa")
        self.refused(lambda c: c["c1_reference"].update(columns=[x for x in C1_COLUMNS if x != "label"]), "columns (including")
        self.refused(lambda c: c["c1_reference"].update(rows=len(self.fx.c1) + 1), "c1 rows must equal c1_reference.rows")
        self.refused(lambda c: c["data"]["manifests"]["cnn_test"].update(rows="24"), "integer counts")

    def test_extractor_order_fixed(self):
        cfg = copy.deepcopy(self.fx.config)
        cfg["extractors"] = {C.EXTRACTORS[1]: cfg["extractors"][C.EXTRACTORS[1]], C.EXTRACTORS[0]: cfg["extractors"][C.EXTRACTORS[0]]}
        self.fx.write_config(cfg)
        text = self.go("check", expect=1)
        self.assertIn("in this order", text)

    def test_malformed_blocks_refused_cleanly(self):
        for key, value in (("weights", 1), ("extractors", []), ("cache", "x"), ("c1_reference", 3), ("data", {"m2_run": "x", "manifests": 3, "originals": "s"})):
            cfg = copy.deepcopy(self.fx.config)
            cfg[key] = value
            self.fx.write_config(cfg)
            text = self.go("check", expect=1)
            self.assertIn("configuration c1crop.json refused", text)
            self.assertNotIn("unexpected", text)

    def test_non_trial_requires_delivered_configuration(self):
        text = self.go("check", study=STUDY_REAL, expect=1)
        self.assertIn("not the delivered one", text)
        with mock.patch.object(C, "CONFIG_SHA256", sha(self.fx.config_path.read_bytes())):
            self.go("check", study=STUDY_REAL)
            text = self.go("check", study="20261103-C1CROP-1234567", expect=1)
            self.assertIn("must carry the current commit", text)

    def test_delivered_configuration(self):
        cands = [ROOT / "delivery" / "c1crop", ROOT.parent]
        folder = next((c for c in cands if (c / "c1crop.json").is_file()), None)
        if folder is None:
            self.skipTest("missing input: " + MISSING_INPUT["delivered configuration"])
        raw = (folder / "c1crop.json").read_bytes()
        doc = json.loads(raw.decode("utf-8"))
        self.assertEqual(C.config_problems(doc), [])
        self.assertEqual(sha(raw), C.CONFIG_SHA256, "the delivered configuration is not the one bound in c1crop_check.py")
        for n in C.EXTRACTORS:
            p = folder / doc["extractors"][n]["file"]
            self.assertTrue(p.is_file(), f"frozen copy {p.name} missing next to the configuration")
            self.assertEqual(sha(p.read_bytes()), doc["extractors"][n]["sha256"], f"{p.name}: SHA-256 differs from the configuration")
            self.assertTrue(doc["extractors"][n]["commit"].startswith(n.split("_")[1]))
        self.assertEqual(doc["extractors"][C.EXTRACTORS[1]]["sha256"], N1.sha256_file(ROOT / "src" / "tesi_app" / "face.py"),
                         "the C1 extractor copy must be the current application face.py")
        self.assertEqual(doc["weights"]["bytes"], 134417864)
        self.assertEqual(doc["data"]["manifests"]["c1"]["rows"], 300)
        self.assertEqual(doc["data"]["manifests"]["cnn_test"]["rows"], 3000)
        self.assertEqual(doc["cache"]["members"]["X_test"]["shape"], [3000, 256, 256, 3])
        self.assertEqual(doc["c1_reference"]["parity"]["tolerance_abs"], 1e-5)


class StudyFolderTests(Base):
    def test_bad_name_and_inside_repository(self):
        for name in ("20261103-N2-combined-abcdef0-trial", "20261103-C1CROP-xyz-trial", "20261103-c1crop-abcdef0-trial"):
            text = self.go("check", study=name, expect=1)
            self.assertIn("study folder refused", text)
        for rel in ("results/n1", "results/n2", "results/m2", "results/c1", "scripts", "data"):
            text = self.go("check", study_dir=ROOT / rel / STUDY, expect=1)
            self.assertIn("study folder refused", text)
            self.assertFalse((ROOT / rel / STUDY).exists())

    def test_non_empty_symlink_and_inputs_inside(self):
        self.study.mkdir(parents=True)
        (self.study / "x").write_text("x")
        text = self.go("check", expect=1)
        self.assertIn("must be new", text)
        self.assertEqual(self.listing(), ["x"])
        link = self.tmp / "out" / "20261103-C1CROP-abcdef4-trial"
        link.parent.mkdir(exist_ok=True)
        os.symlink(self.tmp / "elsewhere", link)
        text = self.go("check", study="20261103-C1CROP-abcdef4-trial", expect=1)
        self.assertIn("is a symbolic link", text)
        asfile = self.tmp / "out" / "20261103-C1CROP-abcdef5-trial"
        asfile.write_text("x")
        self.assertIn("exists and is not a folder", self.go("check", study="20261103-C1CROP-abcdef5-trial", expect=1))
        self.assertIn("is a symbolic link or not a folder", self.go("check", expect=1, manifests=self.fx.config_path))
        dl = self.tmp / "images_link"
        os.symlink(self.fx.images, dl)
        self.assertIn("is a symbolic link or not a folder", self.go("check", expect=1, images=dl))
        inside = self.tmp / "out" / "20261103-C1CROP-abcdef3-trial"
        inside.mkdir(parents=True)
        cfg_inside = inside / "c1crop.json"
        shutil.copy2(self.fx.config_path, cfg_inside)
        text = self.go("check", study_dir=inside, config=cfg_inside, expect=1)      # un ingresso dentro la cartella dello studio
        self.assertIn("inputs refused", text)
        self.assertEqual(self.listing(inside), ["c1crop.json"])
        text = self.go("check", weights=self.tmp / "missing.h5", expect=1)
        self.assertIn("not found", text)


class InputTests(Base):
    def test_manifest_sha_and_rows(self):
        p = self.fx.man / self.fx.files["cnn_test"]
        p.write_text(p.read_text() + "\n")
        self.refuse_check("manifest nuaa_cnn_test.csv (cnn_test): SHA-256")

    def test_session_outside_test_and_duplicated_key(self):
        extra = row(0, "0009", "01", 0)                      # sessione 01: lista ufficiale di training
        write_csv(self.fx.man / self.fx.files["official_test"], self.fx.rows["official_test"] + [extra])
        cfg = self.fx.make_config()
        m = cfg["data"]["manifests"]["official_test"]
        m.update(sha256=sha((self.fx.man / self.fx.files["official_test"]).read_bytes()), rows=m["rows"] + 1, n_bona_fide=m["n_bona_fide"] + 1, n_subjects=m["n_subjects"] + 1)
        self.fx.write_config(cfg)
        self.refuse_check("rows outside the official test split")
        write_csv(self.fx.man / self.fx.files["official_test"], self.fx.rows["official_test"])
        dup = self.fx.rows["official_test"][0]
        write_csv(self.fx.man / self.fx.files["official_test"], self.fx.rows["official_test"] + [dup])
        cfg = self.fx.make_config()
        m = cfg["data"]["manifests"]["official_test"]
        m.update(sha256=sha((self.fx.man / self.fx.files["official_test"]).read_bytes()), rows=m["rows"] + 1)
        m["n_bona_fide" if dup["label"] == "0" else "n_attack"] += 1
        self.fx.write_config(cfg)
        self.refuse_check("duplicated key values")

    def test_symlinked_manifest_and_renamed_inputs(self):
        p = self.fx.man / self.fx.files["c1"]
        real = self.tmp / "c1_elsewhere.csv"
        shutil.move(p, real)
        os.symlink(real, p)
        self.refuse_check("not found or not a regular file")
        p.unlink()
        shutil.move(real, p)
        for attr, src, needle in (("weights", self.fx.weights, "is not AttackNetV2_1__nuaa.h5"), ("cache", self.fx.cache, "is not the file of the configuration"),
                                  ("c1_scores", self.fx.c1_scores, "is not scores.csv")):
            other = self.tmp / f"renamed_{src.name}"
            shutil.copy2(src, other)
            self.refuse_check(needle, **{attr: other})

    def test_cache_extra_member(self):
        self.fx.write_cache(extra={"X_extra": np.zeros(3, np.uint8)})
        self.fx.refresh_config()
        self.refuse_check("members")

    def test_dirty_tree_refused_for_non_trial(self):
        with mock.patch.object(C, "CONFIG_SHA256", sha(self.fx.config_path.read_bytes())), \
                mock.patch.object(N1, "REPO_STATE", lambda d: {"commit": COMMIT, "dirty": True, "dirty_paths": ["x.txt"], "dirty_excludes": None}):
            text = self.go("check", study=STUDY_REAL, expect=1)
            self.assertIn("work tree is not clean", text)
            self.assertFalse((self.tmp / "out" / STUDY_REAL).exists())
            self.go("check")        # un run -trial accetta l'albero sporco e lo registra
            self.assertEqual(self.record("check.json")["dirty_paths"], ["x.txt"])

    def test_c1_row_differs_from_cnn_test(self):
        rows = copy.deepcopy(self.fx.rows["c1"])
        rows[0]["sha256"] = sha(b"other")
        write_csv(self.fx.man / self.fx.files["c1"], rows)
        self.fx.refresh_config()
        self.refuse_check("different fields")

    def test_sampling_order(self):
        c1_keys = {r["key"] for r in self.fx.rows["c1"]}
        drop = next(r for r in self.fx.rows["cnn_test"] if r["key"] not in c1_keys)      # una riga fuori da c1: resta solo l'ordine di campionamento a fermare
        rows = [r for r in self.fx.rows["official_test"] if r["key"] not in set(self.fx.order)][:1] + [r for r in self.fx.rows["cnn_test"] if r is not drop]
        write_csv(self.fx.man / self.fx.files["cnn_test"], rows)
        self.fx.refresh_config()
        self.refuse_check("sampling order of the official test list does not give cnn_test")

    def test_original_altered_or_missing(self):
        p = self.fx.path_of(self.fx.c1[0])
        p.write_bytes(p.read_bytes()[:-1] + b"X")
        self.refuse_check("size or SHA-256 differ")
        p.unlink()
        self.refuse_check("not found under the images folder")

    def test_cache_sha_and_schema(self):
        self.fx.write_cache(extra={"y_test": np.zeros(len(self.fx.order), np.int64)})
        self.refuse_check("cache nuaa_1500.npz: SHA-256")
        self.fx.write_cache(x_test=np.zeros((len(self.fx.order), 256, 256, 3), np.float32))
        self.fx.refresh_config()
        self.refuse_check("X_test has dtype")

    def test_weights_extractors_resources_reference(self):
        w = self.fx.weights.read_bytes()
        self.fx.weights.write_bytes(w + b"x")
        self.refuse_check("weights file size")
        self.fx.weights.write_bytes(w[:-1] + b"x")
        self.refuse_check("weights file: SHA-256 differs")
        self.fx.weights.write_bytes(w)
        f0 = self.fx.faces[C.EXTRACTORS[0]]
        t0 = f0.read_text()
        f0.write_text(t0 + "# changed\n")
        self.refuse_check(f"extractor {C.EXTRACTORS[0]} (face_0ac7143.py): SHA-256 differs from the configuration")
        f0.write_text(t0)
        f1 = self.fx.faces[C.EXTRACTORS[1]]
        t = f1.read_text()
        f1.write_text(t + "# changed\n")
        self.refuse_check("SHA-256 differs from the configuration")
        f1.write_text(t)
        with mock.patch.object(C, "CURRENT_FACE", str(self.fx.faces[C.EXTRACTORS[0]])):
            self.refuse_check("current application face.py is not the C1 extractor copy")
        other = self.tmp / "face_a80f6f9.py"
        other.write_text(t)
        self.refuse_check("not found", face_c1=self.tmp / "nope.py")
        self.refuse_check("is not face_0ac7143.py", face_cache=other)
        r = self.fx.resources["deploy.prototxt"]
        r.write_bytes(b"changed")
        self.refuse_check("detector resource deploy.prototxt: SHA-256 differs")
        r.write_bytes(b"fake deploy.prototxt")
        ref = TMP_ROOT / "livedetection" / "src" / "architectures.py"
        data = ref.read_bytes()
        ref.write_bytes(data + b"\n# changed\n")
        try:
            self.refuse_check("reference code architectures.py: SHA-256 differs")
        finally:
            ref.write_bytes(data)
        self.assertEqual(self.go("check").count("RESULT: PASS"), 1)

    def test_c1_scores_bound(self):
        rows = self.fx.c1_score_rows()
        self.fx.write_c1_scores(rows + [rows[0]])
        self.refuse_check("C1 scores scores.csv: SHA-256")
        self.fx.refresh_config()
        self.refuse_check("instead of")
        v2 = [r for r in rows if r["analyzer_id"] != "attacknet_v2_1__nuaa"] + [dict(r, sample_id=self.fx.rows["cnn_test"][5]["key"]) if i == 0 else r
                                                                                for i, r in enumerate(r for r in rows if r["analyzer_id"] == "attacknet_v2_1__nuaa")]
        self.fx.write_c1_scores(v2)
        self.fx.refresh_config()
        self.refuse_check("do not hold exactly the c1 manifest keys")
        for field, value in (("subject_id", "0099"), ("label", "x"), ("checkpoint_sha256", sha(b"other")), ("config_sha256", sha(b"z")), ("threshold", "0.4"),
                             ("score_attack", "-0.5"), ("prediction", "x"), ("run_id", "20260101-C1-seed42-0000000")):
            v3 = [dict(r, **{field: value}) if r["analyzer_id"] == "attacknet_v2_1__nuaa" and r is rows[0] else r for r in rows]
            self.fx.write_c1_scores(v3)
            self.fx.refresh_config()
            self.refuse_check("rows inconsistent")
        self.fx.write_c1_scores()
        self.fx.refresh_config()
        self.go("check")


class FramesTests(Base):
    def test_pass(self):
        self.pipeline("frames")
        self.assertEqual(self.listing(), ["check.json", "check.log", "frames.csv", "frames.json", "frames.log", "status.json"])
        doc = self.record("frames.json")
        rows = N1.read_csv_rows(self.study / "frames.csv", C.FRAME_COLUMNS)
        self.assertEqual(len(rows), 2 * len(self.fx.c1))
        by = {(r["extractor"], r["sample_id"]): r for r in rows}
        for r in self.fx.c1:
            det = det_of_name(nm.name_of(r["key"]))
            a, b = by[(C.EXTRACTORS[0], r["key"])], by[(C.EXTRACTORS[1], r["key"])]
            self.assertEqual((a["detector"], b["detector"]), (det, det))
            self.assertEqual(a["equal_to_cache"], "1")
            self.assertEqual(b["equal_to_cache"], "0" if det == "RetinaFace" else "1")
            self.assertEqual(a["frame_sha256"] == b["frame_sha256"], det != "RetinaFace")
            if det == "none":
                self.assertEqual(a["bbox_w"], "")
            elif det == "RetinaFace":
                self.assertEqual(int(a["bbox_w"]) - int(b["bbox_w"]), 1)
            self.assertEqual(a["file_sha256"], r["sha256"])
        s = doc["summary"]
        n_rf = sum(1 for r in self.fx.c1 if det_of_name(nm.name_of(r["key"])) == "RetinaFace")
        self.assertEqual(s["equal_to_cache"][C.EXTRACTORS[0]], len(self.fx.c1))
        self.assertEqual(s["frames_different_between_extractors"], n_rf)
        self.assertEqual(s["box_delta_c1_minus_cache"].get("dx 0, dy 0, dw -1, dh -1"), n_rf)
        self.assertEqual(doc["tensorflow"], "not imported")
        self.assertEqual(doc["files"]["frames.csv"], sha((self.study / "frames.csv").read_bytes()))

    def test_cache_chain_not_reconstructed_writes_nothing(self):
        def pre(name, face_path, cfg):
            inner = fake_preprocessor(name, face_path, cfg)

            def f(path):
                X, det, box = inner(path)
                if name == C.EXTRACTORS[0] and det == "RetinaFace":
                    X = X.copy()
                    X[0, 0, 0] ^= 1
                return X, det, box
            return f
        self.go("check")
        with mock.patch.object(C, "PREPROCESSOR", pre):
            text = self.go("frames", expect=1)
        self.assertIn("cache chain is not reconstructed", text)
        self.assertIn("max abs difference 1", text)
        self.assertEqual(self.listing(), ["check.json", "check.log", "frames.log", "status.json"])
        self.assertEqual(self.status()["result"], "FAIL")

    def test_duplicated_and_bad_shape(self):
        def dup(name, face_path, cfg):
            inner = fake_preprocessor(name, face_path, cfg)
            first = []

            def f(path):
                X, det, box = inner(path)
                if name == C.EXTRACTORS[1]:
                    if not first:
                        first.append(X)
                    return first[0], det, box
                return X, det, box
            return f
        self.go("check")
        with mock.patch.object(C, "PREPROCESSOR", dup):
            self.assertIn("duplicated derived frames", self.go("frames", expect=1))

        def bad(name, face_path, cfg):
            inner = fake_preprocessor(name, face_path, cfg)
            return lambda path: (inner(path)[0][:, :255], inner(path)[1], inner(path)[2])
        with mock.patch.object(C, "PREPROCESSOR", bad):
            self.assertIn("preprocessed frame has shape", self.go("frames", expect=1))
        self.assertNotIn("frames.csv", self.listing())

    def test_never_in_place_requires_check_code_and_inputs(self):
        self.assertIn("run check first", self.go("frames", expect=1))
        self.pipeline("frames")
        before = (self.study / "frames.csv").read_bytes()
        self.assertIn("never recomputed in place", self.go("frames", expect=1))
        self.assertEqual((self.study / "frames.csv").read_bytes(), before)
        with mock.patch.object(C, "code_hashes", lambda a: {"x": "y"}):
            self.assertIn("code changed since check", self.go("parity", expect=1))
        study2 = "20261103-C1CROP-abcdef1-trial"
        self.go("check", study=study2)
        d2 = self.tmp / "out" / study2
        doc = json.loads((d2 / "check.json").read_text(encoding="utf-8"))
        doc["checks"]["manifests"]["c1"]["rows"] += 1
        (d2 / "check.json").write_text(json.dumps(doc), encoding="utf-8")
        self.assertIn("manifests differ from those recorded at check", self.go("frames", study=study2, expect=1))
        doc["passed"] = False
        (d2 / "check.json").write_text(json.dumps(doc), encoding="utf-8")
        self.assertIn("does not record a passed step", self.go("frames", study=study2, expect=1))

    def test_commit_changed_for_non_trial(self):
        with mock.patch.object(C, "CONFIG_SHA256", sha(self.fx.config_path.read_bytes())):
            self.go("check", study=STUDY_REAL)
            with mock.patch.object(N1, "REPO_STATE", lambda d: {"commit": "1" * 40, "dirty": False, "dirty_paths": [], "dirty_excludes": None}):
                self.assertIn("repository commit differs from check", self.go("frames", study=STUDY_REAL, expect=1))


class ParityTests(Base):
    def test_pass(self):
        self.pipeline("parity")
        doc = self.record("parity.json")
        rows = N1.read_csv_rows(self.study / "parity.csv", C.PARITY_COLUMNS)
        self.assertEqual([r["sample_id"] for r in rows], [r["key"] for r in self.fx.c1])
        self.assertTrue(all(float(r["abs_diff"]) == 0 and r["prediction_stored"] == r["prediction_recomputed"] for r in rows))
        self.assertEqual((doc["n"], doc["within_tolerance"], doc["same_decision"], doc["max_abs_diff"]), (len(rows), len(rows), len(rows), 0.0))
        self.assertEqual((doc["within_1e-6"], doc["within_1e-7"], doc["batch"]), (len(rows), len(rows), 16))
        self.assertEqual((doc["c1_conditions"]["device"], doc["c1_conditions"]["batch"], doc["c1_conditions"]["max_abs_diff"], doc["c1_conditions"]["same_decision"]),
                         ("FAKE", 1, 0.0, len(rows)))
        self.assertTrue(all(r["score_c1_conditions"] == r["score_recomputed"] for r in rows))
        self.assertEqual((doc["cpu_batch1"]["device"], doc["cpu_batch1"]["batch"], doc["cpu_batch1"]["max_abs_diff"], doc["cpu_batch1"]["within_1e-5"]), ("CPU", 1, 0.0, len(rows)))
        self.assertEqual(doc["frames_sha256"], sha((self.study / "frames.json").read_bytes()))
        stored = {r["sample_id"]: float(r["score_stored"]) for r in rows}
        ref = N1.set_metrics([int(r["label"]) for r in self.fx.c1], [stored[r["key"]] for r in self.fx.c1])
        self.assertEqual(doc["c1_reference_metrics"], ref)
        self.assertEqual(doc["device"], "CPU")

    def shifted(self, delta):
        return lambda path, wsha: (lambda X: np.clip(fake_prob(X, wsha) + np.array([[-delta, delta]]), 0, 1))

    def test_tolerance_from_configuration(self):
        self.pipeline("frames")
        with mock.patch.object(C, "MAKE_SCORER", self.shifted(5e-6)), mock.patch.object(C, "MAKE_SCORER_C1", self.shifted(3e-6)), \
                mock.patch.object(C, "MAKE_SCORER_CPU1", self.shifted(2e-5)):
            self.go("parity")        # le riproduzioni informative non fermano lo studio
        doc = self.record("parity.json")
        self.assertEqual((doc["within_tolerance"], doc["within_1e-6"], doc["c1_conditions"]["within_1e-6"], doc["cpu_batch1"]["within_1e-5"]), (doc["n"], 0, 0, 0))
        study2 = "20261103-C1CROP-abcdef1-trial"
        self.go("check", study=study2)
        self.go("frames", study=study2)
        with mock.patch.object(C, "MAKE_SCORER", self.shifted(2e-5)):
            text = self.go("parity", study=study2, expect=1)
        self.assertIn("C1 chain is not reconstructed", text)
        self.assertIn("outside the tolerance 1e-05", text)
        self.assertNotIn("parity.csv", self.listing(self.tmp / "out" / study2))
        cfg = copy.deepcopy(self.fx.config)
        cfg["c1_reference"]["parity"]["tolerance_abs"] = 1e-7
        self.fx.write_config(cfg)
        study3 = "20261103-C1CROP-abcdef2-trial"
        self.go("check", study=study3)
        self.go("frames", study=study3)
        with mock.patch.object(C, "MAKE_SCORER", self.shifted(5e-7)):
            self.assertIn("outside the tolerance 1e-07", self.go("parity", study=study3, expect=1))

    def test_requires_frames_and_never_in_place(self):
        self.go("check")
        self.assertIn("run the previous step first", self.go("parity", expect=1))
        self.go("frames")
        self.go("parity")
        before = (self.study / "parity.json").read_bytes()
        self.assertIn("never recomputed in place", self.go("parity", expect=1))
        self.assertEqual((self.study / "parity.json").read_bytes(), before)

    def test_frame_differs_from_frames_csv(self):
        self.pipeline("frames")

        def pre(name, face_path, cfg):
            inner = fake_preprocessor(name, face_path, cfg)

            def f(path):
                X, det, box = inner(path)
                X = X.copy()
                X[1, 1, 1] ^= 1
                return X, det, box
            return f
        with mock.patch.object(C, "PREPROCESSOR", pre):
            self.assertIn("differ from frames.csv", self.go("parity", expect=1))
        rows = N1.read_csv_rows(self.study / "frames.csv", C.FRAME_COLUMNS)
        i = next(i for i, r in enumerate(rows) if r["extractor"] == C.EXTRACTORS[1])      # la parità controlla i frame dell'estrattore C1
        rows[i]["detector"] = "Haar" if rows[i]["detector"] != "Haar" else "RetinaFace"
        self.redeclare("frames.json", "frames.csv", N1.csv_text(rows, C.FRAME_COLUMNS))
        self.assertIn("differ from frames.csv", self.go("parity", expect=1))

    def test_nan_scores_refused(self):
        self.pipeline("frames")
        with mock.patch.object(C, "MAKE_SCORER", lambda p, s: (lambda X: np.full((len(X), 2), np.nan))):
            self.assertIn("non-finite", self.go("parity", expect=1))
        with mock.patch.object(C, "MAKE_SCORER_C1", lambda p, s: (lambda X: np.full((len(X), 2), np.nan))):
            self.assertIn("non-finite", self.go("parity", expect=1))
        with mock.patch.object(C, "MAKE_SCORER_CPU1", lambda p, s: (lambda X: np.full((len(X), 2), np.nan))):
            self.assertIn("non-finite", self.go("parity", expect=1))

    def test_frames_csv_rows_and_extractor_checked(self):
        self.pipeline("frames")
        rows = N1.read_csv_rows(self.study / "frames.csv", C.FRAME_COLUMNS)
        self.redeclare("frames.json", "frames.csv", N1.csv_text(rows[1:], C.FRAME_COLUMNS))
        self.assertIn("does not hold exactly the c1 keys", self.go("parity", expect=1))
        bad = [dict(r, extractor="other") if i == 0 else r for i, r in enumerate(rows)]
        self.redeclare("frames.json", "frames.csv", N1.csv_text(bad, C.FRAME_COLUMNS))
        self.assertIn("unknown extractor", self.go("parity", expect=1))

    def test_same_decision_required(self):
        rows = [dict(r, score_attack="0.5", prediction="0") for r in self.fx.c1_score_rows()]
        self.fx.write_c1_scores(rows)
        self.fx.refresh_config()
        self.go("check")
        self.go("frames")
        with mock.patch.object(C, "MAKE_SCORER", lambda p, s: (lambda X: np.full((len(X), 2), 0.5 + 5e-7))):
            text = self.go("parity", expect=1)
        self.assertIn("decisions differ", text)
        self.assertIn("0 scores outside the tolerance", text)
        study2 = "20261103-C1CROP-abcdef1-trial"
        self.go("check", study=study2)
        self.go("frames", study=study2)
        with mock.patch.object(C, "MAKE_SCORER", lambda p, s: (lambda X: np.full((len(X), 2), 0.5 - 5e-7))):
            self.go("parity", study=study2)


def pixel_scorer(path, wsha):
    """Score = primo pixel / 255: il test ricalcola le attese da sé."""
    return lambda X: np.stack([1 - np.asarray(X)[:, 0, 0, 0] / 255.0, np.asarray(X)[:, 0, 0, 0] / 255.0], axis=1)


class ScoreTests(Base):
    def test_pass_paired_rows(self):
        self.pipeline("parity")
        wsha = self.fx.config["weights"]["sha256"]
        # parity con lo scorer di riferimento, score con lo scorer "primo pixel": gli score C1 salvati non entrano nel confronto
        with mock.patch.object(C, "MAKE_SCORER", pixel_scorer):
            text = self.go("score", "--i-am-authorized")
        rows = N1.read_csv_rows(self.study / "scores_paired.csv", C.SCORE_COLUMNS)
        self.assertEqual([r["sample_id"] for r in rows], [r["key"] for r in self.fx.c1])
        changed = set()
        for r, m in zip(rows, self.fx.c1):
            f0, f1 = self.fx.frame(m, C.EXTRACTORS[0]), self.fx.frame(m, C.EXTRACTORS[1])
            v0, v1 = f0[0][0, 0, 0] / 255.0, f1[0][0, 0, 0] / 255.0
            p0, p1 = ("1" if v0 > 0.5 else "0"), ("1" if v1 > 0.5 else "0")
            exp_change = "unchanged" if p0 == p1 else ("attack_to_bona_fide" if p1 == "1" else "bona_fide_to_attack")
            changed.add(exp_change)
            self.assertEqual((float(r["score_cache"]), float(r["score_c1"])), (v0, v1))
            self.assertEqual((r["prediction_cache"], r["prediction_c1"], r["decision_change"]), (p0, p1, exp_change))
            self.assertEqual(float(r["abs_diff"]), abs(v0 - v1))
            self.assertEqual(float(r["signed_diff"]), v0 - v1)
            self.assertEqual((r["detector_cache"], r["detector_c1"]), (f0[1], f1[1]))
            self.assertEqual(r["frame_equal"], "1" if f0[1] != "RetinaFace" else "0")
            self.assertEqual((r["subject_id"], r["label"], r["file_sha256"], r["weights_sha256"]), (m["subject"], m["label"], m["sha256"], wsha))
        self.assertIn("unchanged", changed)
        met = self.record("metrics.json")
        self.assertEqual(met["metrics"], C.metrics_from_rows(rows))
        self.assertEqual(met["paired"], C.paired_stats(rows))
        self.assertEqual(met["paired"]["frames_equal"], sum(1 for r in rows if r["frame_equal"] == "1"))
        subj = N1.read_csv_rows(self.study / "per_subject.csv", C.SUBJECT_COLUMNS)
        self.assertEqual(len(subj), 2 * len({r["subject"] for r in self.fx.c1}))
        self.assertEqual({r["extractor"] for r in subj}, set(C.EXTRACTORS))
        sc = self.record("score.json")
        self.assertEqual(sc["parity_sha256"], sha((self.study / "parity.json").read_bytes()))
        self.assertEqual(sc["n_images"], len(rows))
        self.assertIn("score PASS", text)

    def test_needs_authorization_parity_and_chain(self):
        self.pipeline("frames")
        self.assertIn("run the previous step first", self.go("score", "--i-am-authorized", expect=1))
        self.go("parity")
        self.assertIn("needs --i-am-authorized", self.go("score", expect=1))
        p = self.study / "frames.json"
        p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        self.assertIn("does not belong to this frames.json", self.go("score", "--i-am-authorized", expect=1))

    def test_never_in_place_and_frames_checked(self):
        self.pipeline("score")
        before = (self.study / "scores_paired.csv").read_bytes()
        self.assertIn("never recomputed in place", self.go("score", "--i-am-authorized", expect=1))
        self.assertEqual((self.study / "scores_paired.csv").read_bytes(), before)
        study2 = "20261103-C1CROP-abcdef1-trial"
        for step in ("check", "frames", "parity"):
            self.go(step, study=study2)

        def pre(name, face_path, cfg):
            inner = fake_preprocessor(name, face_path, cfg)

            def f(path):
                X, det, box = inner(path)
                if name == C.EXTRACTORS[0]:
                    X = X.copy()
                    X[2, 2, 2] ^= 1
                return X, det, box
            return f
        with mock.patch.object(C, "PREPROCESSOR", pre):
            self.assertIn("differ from frames.csv", self.go("score", "--i-am-authorized", study=study2, expect=1))

    def test_frames_csv_changed_after_parity(self):
        self.pipeline("parity")
        p = self.study / "frames.csv"
        p.write_text(p.read_text(encoding="utf-8") + "x", encoding="utf-8")
        self.assertIn("frames.csv changed after frames", self.go("score", "--i-am-authorized", expect=1))

    def test_tie_at_threshold_is_bona_fide(self):
        self.pipeline("parity")
        with mock.patch.object(C, "MAKE_SCORER", lambda p, s: (lambda X: np.full((len(X), 2), 0.5))):
            self.go("score", "--i-am-authorized")
        rows = N1.read_csv_rows(self.study / "scores_paired.csv", C.SCORE_COLUMNS)
        self.assertTrue(all(r["prediction_cache"] == "0" and r["prediction_c1"] == "0" and r["decision_change"] == "unchanged" for r in rows))
        m = self.record("metrics.json")["metrics"][C.EXTRACTORS[0]]
        self.assertEqual((m["apcer"], m["bpcer"], m["ties_at_threshold"]), (1.0, 0.0, len(rows)))

    def test_nan_scores_refused(self):
        self.pipeline("parity")
        with mock.patch.object(C, "MAKE_SCORER", lambda p, s: (lambda X: np.full((len(X), 2), np.nan))):
            self.assertIn("non-finite", self.go("score", "--i-am-authorized", expect=1))
        self.assertNotIn("scores_paired.csv", self.listing())


class AggregateTests(Base):
    def test_full_pipeline(self):
        text = self.pipeline()
        self.assertTrue({"aggregate.json", "aggregate.md", "figure_paired.png"} <= set(self.listing()))
        doc = self.record("aggregate.json")
        self.assertEqual(doc["n"], len(self.fx.c1))
        self.assertEqual(doc["metrics"], self.record("metrics.json")["metrics"])
        self.assertEqual(doc["paired"], self.record("metrics.json")["paired"])
        self.assertEqual(doc["c1_reference"]["metrics"], self.record("parity.json")["c1_reference_metrics"])
        self.assertEqual(doc["frames"], self.record("frames.json")["summary"])
        self.assertNotIn("figure_points", doc)
        self.assertNotIn("score_c1", json.dumps(doc))
        inp = types.SimpleNamespace(rows=self.fx.rows)
        for name in ("aggregate.json", "aggregate.md"):
            self.assertEqual(N1.public_problems((self.study / name).read_text(encoding="utf-8"), inp), [], name)
        md = (self.study / "aggregate.md").read_text(encoding="utf-8")
        for n in C.EXTRACTORS:
            self.assertIn(f"| {n} |", md)
        self.assertIn("C1 delivered", md)
        self.assertIn("decisions changed at 0.5", md)
        self.assertEqual((self.study / "figure_paired.png").read_bytes(), PNG)
        self.assertIn("aggregate PASS", text)
        text = self.go("aggregate")
        self.assertIn("removed previous aggregate.json", text)

    def test_records_chain_and_tampering(self):
        self.pipeline()
        p = self.study / "parity.json"
        p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        self.assertIn("do not form one chain", self.go("aggregate", expect=1))

    def test_metrics_and_subjects_recomputed(self):
        self.pipeline("score")
        met = self.record("metrics.json")
        met["metrics"][C.EXTRACTORS[0]]["acer"] = 0.123
        self.redeclare("score.json", "metrics.json", json.dumps(met, indent=1) + "\n")
        self.assertIn("differ from metrics.json", self.go("aggregate", expect=1))
        study2 = "20261103-C1CROP-abcdef1-trial"
        for step, extra in (("check", []), ("frames", []), ("parity", []), ("score", ["--i-am-authorized"])):
            self.go(step, *extra, study=study2)
        d2 = self.tmp / "out" / study2
        subj = N1.read_csv_rows(d2 / "per_subject.csv", C.SUBJECT_COLUMNS)
        subj[0]["rejected_bona_fide"] = str(int(subj[0]["rejected_bona_fide"]) + 1)
        text = N1.csv_text(subj, C.SUBJECT_COLUMNS)
        (d2 / "per_subject.csv").write_text(text, encoding="utf-8")
        doc = json.loads((d2 / "score.json").read_text(encoding="utf-8"))
        doc["files"]["per_subject.csv"] = sha(text.encode())
        (d2 / "score.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        self.assertIn("per_subject.csv differs", self.go("aggregate", study=study2, expect=1))

    def test_rows_detectors_and_paired_tampered(self):
        self.pipeline("score")
        rows = N1.read_csv_rows(self.study / "scores_paired.csv", C.SCORE_COLUMNS)
        for field, value in (("label", "1" if rows[0]["label"] == "0" else "0"), ("decision_change", "unchanged" if rows[0]["decision_change"] != "unchanged" else "bona_fide_to_attack"),
                             ("abs_diff", "0.25"), ("frame_equal", "1" if rows[0]["frame_equal"] == "0" else "0"), ("prediction_c1", "1" if rows[0]["prediction_c1"] == "0" else "0")):
            bad = [dict(r, **{field: value}) if i == 0 else r for i, r in enumerate(rows)]
            self.redeclare("score.json", "scores_paired.csv", N1.csv_text(bad, C.SCORE_COLUMNS))
            self.assertIn("rows inconsistent", self.go("aggregate", expect=1), field)
        self.redeclare("score.json", "scores_paired.csv", N1.csv_text(rows[1:] + rows[:1], C.SCORE_COLUMNS))
        self.assertIn("in key order", self.go("aggregate", expect=1))
        self.redeclare("score.json", "scores_paired.csv", N1.csv_text(rows, C.SCORE_COLUMNS))
        sc = self.record("score.json")
        sc["detectors"][C.EXTRACTORS[0]] = {"Haar": len(rows)}
        (self.study / "score.json").write_text(json.dumps(sc, indent=1) + "\n", encoding="utf-8")
        self.assertIn("detector counts of score.json differ", self.go("aggregate", expect=1))
        sc["detectors"] = {n: dict(sorted(__import__("collections").Counter(r[c] for r in rows).items())) for n, c in zip(C.EXTRACTORS, ("detector_cache", "detector_c1"))}
        sc["weights_sha256"] = sha(b"other")
        (self.study / "score.json").write_text(json.dumps(sc, indent=1) + "\n", encoding="utf-8")
        self.assertIn("record another checkpoint", self.go("aggregate", expect=1))
        sc["weights_sha256"] = self.fx.config["weights"]["sha256"]
        par = self.record("parity.json")
        par["weights_sha256"] = sha(b"other")
        par_txt = json.dumps(par, indent=1) + "\n"
        (self.study / "parity.json").write_text(par_txt, encoding="utf-8")
        sc["parity_sha256"] = sha(par_txt.encode())
        (self.study / "score.json").write_text(json.dumps(sc, indent=1) + "\n", encoding="utf-8")
        self.assertIn("record another checkpoint", self.go("aggregate", expect=1))

    def test_public_scan_plotter_and_stale_outputs(self):
        self.pipeline("score")
        key = self.fx.c1[0]["key"]
        with mock.patch.object(C, "table_md", lambda doc: f"# table with {key}\n"):
            self.assertIn("would hold image keys", self.go("aggregate", expect=1))
        self.assertNotIn("aggregate.json", self.listing())
        with mock.patch.object(C, "table_md", lambda doc: f"# table with subject {self.fx.c1[0]['subject']}\n"):
            self.assertIn("subject identifiers", self.go("aggregate", expect=1))
        with mock.patch.object(C, "PLOTTER", lambda doc, pts: {"figure_paired.png": b"not png"}):
            self.assertIn("did not return the PNG figure", self.go("aggregate", expect=1))
        self.assertNotIn("aggregate.json", self.listing())
        (self.study / "aggregate.md").write_text("stale", encoding="utf-8")
        text = self.go("aggregate")
        self.assertIn("removed previous aggregate.md", text)
        self.assertNotEqual((self.study / "aggregate.md").read_text(encoding="utf-8"), "stale")
        (self.study / "aggregate.json").unlink()
        (self.study / "aggregate.json").mkdir()
        self.assertIn("not a regular file", self.go("aggregate", expect=1))
        self.assertTrue((self.study / "aggregate.json").is_dir())

    def test_tampered_after_pass_leaves_no_pass(self):
        self.pipeline()
        p = self.study / "scores_paired.csv"
        p.write_text(p.read_text(encoding="utf-8").replace("\n", "\n", 1) + "x,", encoding="utf-8")
        self.assertIn("changed after score", self.go("aggregate", expect=1))
        self.assertEqual(self.status()["result"], "FAIL")
        self.assertNotIn("aggregate.json", self.listing())


class UnitTests(unittest.TestCase):
    def test_prediction_and_change(self):
        self.assertEqual((C.prediction(0.5), C.prediction(0.5000001), C.prediction(0.0)), ("0", "1", "0"))
        self.assertEqual(C.change_of("1", "1"), "unchanged")
        self.assertEqual(C.change_of("1", "0"), "attack_to_bona_fide")
        self.assertEqual(C.change_of("0", "1"), "bona_fide_to_attack")

    def test_paired_stats_bins_and_changes(self):
        def r(diff, sign, lab, change):
            return {"abs_diff": diff, "signed_diff": sign, "label": lab, "decision_change": change, "frame_equal": "1" if diff == "0.0" else "0",
                    "detector_cache": "RetinaFace", "detector_c1": "Haar"}
        rows = [r("0.0", "0.0", "0", "unchanged"), r("0.0001", "0.0001", "1", "unchanged"), r("0.001", "0.001", "0", "bona_fide_to_attack"),
                r("0.01", "0.01", "1", "attack_to_bona_fide"), r("0.1", "0.1", "0", "unchanged"), r("0.2", "-0.2", "1", "attack_to_bona_fide")]
        s = C.paired_stats(rows)
        self.assertEqual(s["abs_diff_bins"], {"zero": 1, "(0, 0.0001]": 1, "(0.0001, 0.001]": 1, "(0.001, 0.01]": 1, "(0.01, 0.1]": 1, "(0.1, 1]": 1})
        self.assertEqual(s["decision_changes_c1_to_cache"], {"unchanged": {"all": 3, "bona_fide": 2, "attack": 1},
                                                             "attack_to_bona_fide": {"all": 2, "bona_fide": 0, "attack": 2},
                                                             "bona_fide_to_attack": {"all": 1, "bona_fide": 1, "attack": 0}})
        self.assertEqual((s["n"], s["frames_equal"], s["max_abs_diff"], s["detector_pairs"]), (6, 1, 0.2, {"RetinaFace/Haar": 6}))
        self.assertAlmostEqual(s["min_signed_diff_cache_minus_c1"], -0.2)
        with self.assertRaises(C.CropError):
            C.paired_stats([dict(rows[0], decision_change="x")])

    def test_load_extractor_checks_sha(self):
        d = Path(tempfile.mkdtemp(dir=TMP_ROOT))
        p = d / "face_0ac7143.py"
        p.write_text(FACE_TEXT[C.EXTRACTORS[0]], encoding="utf-8")
        det = C.load_extractor("unit_a", p, sha(p.read_bytes()))
        self.assertIsNone(det.detect(np.zeros((4, 4, 3), np.uint8)))
        with self.assertRaises(C.CropError):
            C.load_extractor("unit_b", p, "0" * 64)
        self.assertNotIn("tesi_app.face_unit_b", sys.modules)

    def test_tf_scorer_scaling_and_sha(self):
        w = TMP_ROOT / "unit_weights.h5"
        w.write_bytes(b"w")
        fake_tf = types.SimpleNamespace(config=types.SimpleNamespace(set_visible_devices=lambda a, b: None,
                                                                     optimizer=types.SimpleNamespace(set_experimental_options=lambda o: None)),
                                        device=lambda name: contextlib.nullcontext())
        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            score = C.tf_scorer(w, sha(b"w"))
            X = np.zeros((3, 256, 256, 3), np.uint8)
            X[1] = 255
            X[2] = 51
            p = score(X)
        self.assertEqual(p.shape, (3, 2))
        self.assertEqual(float(p[0, 1]), 0.0)
        self.assertEqual(float(p[1, 1]), 1.0)
        self.assertAlmostEqual(float(p[2, 1]), 51 / 255, places=6)
        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}), self.assertRaises(C.CropError):
            C.tf_scorer(w, "0" * 64)
        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            score1 = C.tf_scorer(w, sha(b"w"), device=None, batch=1)
            q = score1(X)
        self.assertEqual(q.shape, (3, 2))
        self.assertAlmostEqual(float(q[2, 1]), 51 / 255, places=6)
        sys.modules.pop("tensorflow", None)

    def test_tensorflow_never_imported(self):
        self.assertNotIn("tensorflow", sys.modules)

    def test_code_files_exist(self):
        for f in C.CODE_FILES:
            self.assertTrue((ROOT / f).is_file(), f)


class MutationHarnessTests(unittest.TestCase):
    def test_classify(self):
        import check_c1crop_mutations as M
        ok = "....\ntests run: 30, failures: 0, errors: 0, skipped: 0\nRESULT: PASS\n"
        self.assertEqual(M.classify(0, ok, 30)[0], "pass")
        self.assertEqual(M.classify(1, "tests run: 30, failures: 1, errors: 0, skipped: 0\n", 30)[0], "fail")
        self.assertEqual(M.classify(1, "SyntaxError\n", 30)[0], "error")
        self.assertEqual(M.classify(0, "tests run: 30, failures: 1, errors: 0, skipped: 0\n", 30)[0], "error")
        self.assertEqual(M.classify(1, "tests run: 30, failures: 0, errors: 0, skipped: 0\n", 30)[0], "error")
        self.assertEqual(M.classify(1, "tests run: 3, failures: 1, errors: 0, skipped: 0\n", 30)[0], "error")
        self.assertEqual(M.classify(1, ok + ok, 30)[0], "error")


class ZZRepositoryUntouched(unittest.TestCase):
    def test_repository_results_untouched(self):
        self.assertEqual((ROOT / "results" / "c1crop").exists(), REPO_BEFORE)
        if REPO_BEFORE:
            self.assertEqual(sorted(p.name for p in (ROOT / "results" / "c1crop").iterdir()), REPO_LISTING)
        for where in ("scripts", "results", "results/n1", "results/n2", "results/c1"):
            self.assertFalse((ROOT / where / STUDY).exists())


def main() -> int:
    if CV2_NOTE:
        print(CV2_NOTE)
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    res = unittest.TextTestRunner(verbosity=1, stream=sys.stdout).run(suite)
    shutil.rmtree(TMP_ROOT, ignore_errors=True)
    print(f"tests run: {res.testsRun}, failures: {len(res.failures)}, errors: {len(res.errors)}, skipped: {len(res.skipped)}")
    ok = res.wasSuccessful()
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
