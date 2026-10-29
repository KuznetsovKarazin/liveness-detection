"""
Test di scripts/n2_combined.py su dati SINTETICI, in cartelle temporanee: nessuna rete, nessuna immagine vera, nessun
dato NUAA, nessun TensorFlow, nessun rilevatore (pre-elaborazione, scorer, figura e stato git sono iniettati).

Fixture: manifest NUAA finti con lo schema M2 (official_test, cnn_test, c1), file "originali" finti (byte con SHA-256 e
dimensione nel manifest), un file del modello finto (SHA-256 e dimensione nella configurazione), l'aggregate.json di uno
studio N1 finto e una configurazione coerente. I run sono -trial, salvo i test del vincolo sulla configurazione consegnata.

  python scripts/check_n2_combined.py        # uscita 1 se un test fallisce; ultima riga RESULT: PASS / FAIL

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
STUDY = "20261030-N2-combined-abcdef0-trial"
STUDY_REAL = "20261030-N2-combined-abcdef0"
COMMIT = "abcdef0" + "0" * 33

TMP_ROOT = Path(tempfile.mkdtemp(prefix="n2-combined-tests-"))
atexit.register(shutil.rmtree, TMP_ROOT, True)
for _k, _s in (("LIVENESSLAB_DATA_DIR", "data"), ("LIVENESSLAB_RESULTS_DIR", "results"), ("LIVENESSLAB_WEIGHTS_DIR", "weights"),
               ("LIVENESSLAB_LIVEDETECTION", "livedetection"), ("LIVENESSLAB_SILENT_FACE", "silent_face")):
    os.environ[_k] = str(TMP_ROOT / _s)
    (TMP_ROOT / _s).mkdir()
REF_CODE = {("scripts", "create_datasets.py"): b"# fake reference create_datasets\n", ("src", "dataset_loader.py"): b"# fake reference dataset_loader\n"}
for _rel, _data in REF_CODE.items():
    _p = TMP_ROOT / "livedetection" / Path(*_rel)
    _p.parent.mkdir(parents=True, exist_ok=True)
    _p.write_bytes(_data)
REPO_N2_BEFORE = (ROOT / "results" / "n2").exists()
REPO_N2_LISTING = sorted(p.name for p in (ROOT / "results" / "n2").iterdir()) if REPO_N2_BEFORE else None
MISSING_INPUT = {"delivered configuration": "n2_combined.json is not next to the code (delivery/n2/, or ../ in the package)"}

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
import n2_combined as C  # noqa: E402
import n2_sessions as N2  # noqa: E402
import n1_train as N1  # noqa: E402
import nuaa_manifests as nm  # noqa: E402

PER_CLASS = 12
SUBJECTS = tuple(f"{i:04d}" for i in range(1, 9))
PNG = b"\x89PNG\r\n\x1a\n" + b"fake"
N1_RUN = "20261024-N1-LivenessNet-fb93496"
FACE_SHA = N1.sha256_file(ROOT / "src" / "tesi_app" / "face.py")


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


def frame_of(data: bytes, variant: str):
    h = hashlib.sha256(variant.encode() + b":" + data).digest()
    X = np.frombuffer((h * 6144)[:256 * 256 * 3], dtype=np.uint8).reshape(256, 256, 3).copy()
    det = "n/a" if variant == "whole_image" else ("none" if h[0] % 11 == 0 else "Haar" if h[0] % 5 == 0 else "RetinaFace")
    return X, det


def fake_preprocessor(variant, cfg):
    def pre(path):
        return frame_of(Path(path).read_bytes(), variant)
    return pre


def fake_prob(X, model_sha: str) -> np.ndarray:
    out = np.zeros((len(X), 2), dtype=np.float64)
    for i, x in enumerate(np.asarray(X)):
        v = int.from_bytes(hashlib.sha256(np.ascontiguousarray(x[:8, :8]).tobytes() + model_sha.encode()).digest()[:4], "big") / 2 ** 32
        out[i] = (1 - v, v)
    return out


def fake_scorer(path, model_sha):
    return lambda X: fake_prob(X, model_sha)


def fake_plotter(doc):
    return {"figure_distributions.png": PNG}


def fake_repo_state(d):
    return {"commit": COMMIT, "dirty": False, "dirty_paths": [], "dirty_excludes": None}


class Fixture:
    def __init__(self, base: Path):
        self.base = base
        self.man, self.images, self.n1 = base / "manifests", base / "images", base / "n1" / N1_RUN
        for d in (self.man, self.images, self.n1):
            d.mkdir(parents=True)
        te = [row(lab, s, "03", p) for s in SUBJECTS for lab in (0, 1) for p in range(4)]
        by_te = {r["key"]: r for r in te}
        self.order_test = nm.spread_keys(te, PER_CLASS)
        self.rows = {"official_test": te, "cnn_test": [by_te[k] for k in self.order_test],
                     "c1": [by_te[k] for i, k in enumerate(self.order_test) if i % 3 == 0]}
        self.test = sorted(self.rows["cnn_test"], key=lambda r: r["key"])
        self.files = {"official_test": "nuaa_official_test.csv", "cnn_test": "nuaa_cnn_test.csv", "c1": "nuaa_c1_v2.csv"}
        for n, rows in self.rows.items():
            write_csv(self.man / self.files[n], rows)
        for r in te:
            p = self.images / N2.LAYOUTS["images"](r)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content_of(r["key"]))
        self.model = base / "AttackNetV2_2_combined_all_final.h5"
        self.model.write_bytes(b"fake historical model " * 100)
        self.resources = {}
        for n in ("deploy.prototxt", "Widerface-RetinaFace.caffemodel", "haarcascade_frontalface_default.xml"):
            p = base / "resources" / n
            p.parent.mkdir(exist_ok=True)
            p.write_bytes(f"fake {n}".encode())
            self.resources[n] = p
        seeds = {}
        for s in (42, 43, 44):
            seeds[str(s)] = {"metrics": {name: {"n": 10, "n_bona_fide": 5, "n_attack": 5, "accepted_attacks": s - 41, "rejected_bona_fide": 2,
                                                "ties_at_threshold": 0, "threshold": 0.5, "decision": C.DECISION_RULE, "apcer": (s - 41) / 5, "bpcer": 0.4,
                                                "acer": ((s - 41) / 5 + 0.4) / 2, "auroc": 0.3 + s / 1000} for name in C.SETS},
                             "training": {"weights_sha256": sha(f"w{s}".encode())}}
        self.n1_doc = {"passed": True, "step": "aggregate", "run_id": N1_RUN, "train_seeds": [42, 43, 44], "threshold": 0.5,
                       "decision": C.DECISION_RULE, "sets": dict(C.SETS), "seeds": seeds}
        (self.n1 / "aggregate.json").write_text(json.dumps(self.n1_doc, indent=1) + "\n", encoding="utf-8")
        self.config = self.make_config()
        self.config_path = base / "n2_combined.json"
        self.write_config()

    def make_config(self) -> dict:
        mans = {n: {"file": self.files[n], "sha256": sha((self.man / self.files[n]).read_bytes()), **N1.counts_of(self.rows[n])} for n in C.MANIFESTS}
        return {"schema": C.CONFIG_SCHEMA, "study": "synthetic", "architecture": "AttackNetV2_2", "dataset": "nuaa", "protocol": "official",
                "model": {"file": self.model.name, "sha256": sha(self.model.read_bytes()), "bytes": self.model.stat().st_size, "source": "synthetic",
                          "keras_version": "2.15.0", "input_shape": [256, 256, 3], "output": "Dense(2, softmax)", "parameters": 1, "loading": "synthetic",
                          "classes": {"0": "bona fide", "1": "attack"}, "score": "column 1"},
                "variants": {"whole_image": {"description": "s", "crop": "none", "chain": "synthetic whole"},
                             "face_crop": {"description": "s", "detector": {"file": "src/tesi_app/face.py", "sha256": FACE_SHA},
                                           "crop": "margin 0.15 square", "fallback": "whole image", "chain": "synthetic crop"}},
                "detector_resources": {n: sha(p.read_bytes()) for n, p in self.resources.items()},
                "reference_code_sha256": {k: sha(REF_CODE[tuple(k.split("/")[1:])]) for k in C.REFERENCE_CODE},
                "quality_filter": "none",
                "scoring": {"device": "CPU", "batch": 16, "batchnorm": "inference", "input": "[-1, 1]", "score": "column 1"},
                "metrics": {"threshold": 0.5, "decision": C.DECISION_RULE, "threshold_origin": "fixed", "reported": ["APCER", "BPCER", "ACER", "AUROC"],
                            "sets": list(C.SETS), "comparison": "N1 rows", "distributions": "10 bins", "public": "aggregate", "restricted": "rows"},
                "n1_reference": {"run_id": N1_RUN, "aggregate_file": "aggregate.json", "aggregate_sha256": sha((self.n1 / "aggregate.json").read_bytes()),
                                 "train_seeds": [42, 43, 44]},
                "data": {"m2_run": "synthetic", "manifests": mans, "originals": "synthetic"}}

    def write_config(self, cfg=None, path=None) -> str:
        cfg = self.config if cfg is None else cfg
        p = self.config_path if path is None else path
        raw = (json.dumps(cfg, indent=1) + "\n").encode()
        p.write_bytes(raw)
        return sha(raw)

    def args(self, study=None, **over) -> list:
        base = {"config": self.config_path, "manifests": self.man, "images": self.images, "model": self.model, "n1_study": self.n1,
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
        self.tmp = Path(tempfile.mkdtemp(prefix="n2c-case-", dir=TMP_ROOT))
        self.fx = Fixture(self.tmp)
        self.study = self.tmp / "out" / STUDY
        self.patches = [mock.patch.object(C, "PREPROCESSOR", fake_preprocessor), mock.patch.object(N2, "RESOURCE_PATHS", self.fx.resources),
                        mock.patch.object(C, "PLOTTER", fake_plotter), mock.patch.object(C, "MAKE_SCORER", fake_scorer),
                        mock.patch.object(N1, "REPO_STATE", fake_repo_state)]
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
        if upto == "check":
            return text
        for v in C.VARIANTS:
            text = self.go("score", "--variant", v, "--i-am-authorized")
        if upto == "score":
            return text
        return self.go("aggregate")

    def listing(self, d=None) -> list:
        d = d or self.study
        return sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()) if d.exists() else []

    def status(self, d=None) -> dict:
        return json.loads(((d or self.study) / "status.json").read_text(encoding="utf-8"))


class ConfigTests(Base):
    def test_check_pass(self):
        text = self.go("check")
        self.assertEqual(self.listing(), ["check.json", "check.log", "status.json"])
        doc = json.loads((self.study / "check.json").read_text())
        self.assertEqual(doc["checks"]["originals"]["n_verified"], len(self.fx.test))
        self.assertEqual(doc["checks"]["n1_reference"]["seeds"], ["42", "43", "44"])
        self.assertEqual(doc["model"], "not loaded")
        self.assertIn("check PASS", text)

    def refused(self, mutate, needle):
        cfg = copy.deepcopy(self.fx.config)
        mutate(cfg)
        self.fx.write_config(cfg)
        text = self.go("check", expect=1)
        self.assertIn(needle, text)
        self.assertFalse(self.study.exists(), text)

    def test_fixed_rules(self):
        self.refused(lambda c: c.pop("quality_filter"), "missing or not allowed")
        self.refused(lambda c: c["metrics"].update(threshold=0.6), "threshold")
        self.refused(lambda c: c["metrics"].update(decision="attack if score_attack >= 0.5"), "decision rule")
        self.refused(lambda c: c["metrics"].update(sets=["test_3000"]), "sets")
        self.refused(lambda c: c["scoring"].update(device="default"), "CPU")
        self.refused(lambda c: c["model"].update(classes={"0": "attack", "1": "bona fide"}), "classes 0 bona fide")
        self.refused(lambda c: c["model"].update(input_shape=[224, 224, 3]), "input_shape")
        self.refused(lambda c: c["variants"].pop("face_crop"), "variants must be exactly")
        self.refused(lambda c: c["variants"]["face_crop"].pop("fallback"), "fallback rule")
        self.refused(lambda c: c["n1_reference"].update(train_seeds=[42]), "n1_reference")
        self.refused(lambda c: c.update(schema="x"), "schema")

    def test_malformed_blocks_refused_cleanly(self):
        for key, value in (("model", 1), ("variants", []), ("data", {"m2_run": "x", "manifests": 3, "originals": "s"})):
            cfg = copy.deepcopy(self.fx.config)
            cfg[key] = value
            self.fx.write_config(cfg)
            text = self.go("check", expect=1)
            self.assertIn("configuration n2_combined.json refused", text)
            self.assertNotIn("unexpected", text)

    def test_variant_order_fixed(self):
        cfg = copy.deepcopy(self.fx.config)
        cfg["variants"] = {"face_crop": cfg["variants"]["face_crop"], "whole_image": cfg["variants"]["whole_image"]}
        self.fx.write_config(cfg)
        text = self.go("check", expect=1)
        self.assertIn("in this order", text)

    def test_non_trial_requires_delivered_configuration(self):
        text = self.go("check", study=STUDY_REAL, expect=1)
        self.assertIn("not the delivered one", text)
        with mock.patch.object(C, "CONFIG_SHA256", sha(self.fx.config_path.read_bytes())):
            self.go("check", study=STUDY_REAL)
            text = self.go("check", study="20261030-N2-combined-1234567", expect=1)
            self.assertIn("must carry the current commit", text)

    def test_delivered_configuration(self):
        cands = [ROOT / "delivery" / "n2" / "n2_combined.json", ROOT.parent / "n2_combined.json"]
        cfg = next((c for c in cands if c.is_file()), None)
        if cfg is None:
            self.skipTest("missing input: " + MISSING_INPUT["delivered configuration"])
        raw = cfg.read_bytes()
        doc = json.loads(raw.decode("utf-8"))
        self.assertEqual(C.config_problems(doc), [])
        self.assertEqual(sha(raw), C.CONFIG_SHA256, "the delivered configuration is not the one bound in n2_combined.py")
        self.assertEqual(doc["variants"]["face_crop"]["detector"]["sha256"], FACE_SHA, "face_crop detector SHA-256 must be the current face.py")
        self.assertEqual(doc["model"]["bytes"], 201809888)
        self.assertEqual(doc["data"]["manifests"]["cnn_test"]["rows"], 3000)


class StudyFolderTests(Base):
    def test_bad_name_and_inside_repository(self):
        for name in ("20261030-N2-sessions-abcdef0-trial", "20261030-N2-combined-xyz-trial"):
            text = self.go("check", study=name, expect=1)
            self.assertIn("study folder refused", text)
        for rel in ("results/n1", "results/m2", "scripts"):
            text = self.go("check", study_dir=ROOT / rel / STUDY, expect=1)
            self.assertIn("study folder refused", text)
            self.assertFalse((ROOT / rel / STUDY).exists())

    def test_non_empty_symlink_and_inputs_inside(self):
        self.study.mkdir(parents=True)
        (self.study / "x").write_text("x")
        self.assertIn("must be new", self.go("check", expect=1))
        shutil.rmtree(self.study)
        real = self.tmp / "real"
        real.mkdir()
        os.symlink(real, self.study)
        self.assertIn("symbolic link", self.go("check", expect=1))
        self.study.unlink()
        self.study.mkdir()
        shutil.copy(self.fx.model, self.study / self.fx.model.name)
        self.assertIn("inputs refused", self.go("check", model=self.study / self.fx.model.name, expect=1))
        self.assertIn("inputs refused", self.go("check", images=self.tmp / "absent", expect=1))


class CheckTests(Base):
    def test_manifest_and_originals(self):
        p = self.fx.man / self.fx.files["c1"]
        p.write_text(p.read_text() + "\n")
        self.assertIn("c1): SHA-256", self.go("check", expect=1))
        p.write_text(p.read_text()[:-1])
        r = self.fx.test[2]
        q = self.fx.images / N2.LAYOUTS["images"](r)
        q.write_bytes(q.read_bytes() + b"x")
        shutil.rmtree(self.tmp / "out", ignore_errors=True)
        self.assertIn("size or SHA-256 differ", self.go("check", expect=1))
        q.unlink()
        shutil.rmtree(self.tmp / "out", ignore_errors=True)
        self.assertIn("not found under the images folder", self.go("check", expect=1))

    def test_c1_row_differs_from_cnn_test(self):
        rows = copy.deepcopy(self.fx.rows["c1"])
        rows[0]["sha256"] = "2" * 64
        write_csv(self.fx.man / self.fx.files["c1"], rows)
        cfg = copy.deepcopy(self.fx.config)
        cfg["data"]["manifests"]["c1"]["sha256"] = sha((self.fx.man / self.fx.files["c1"]).read_bytes())
        self.fx.write_config(cfg)
        self.assertIn("same key as cnn_test but different fields", self.go("check", expect=1))

    def test_model_face_resources_reference(self):
        self.fx.model.write_bytes(b"other model")
        self.assertIn("model file size", self.go("check", expect=1))
        self.fx.model.write_bytes(b"x" * self.fx.config["model"]["bytes"])
        shutil.rmtree(self.tmp / "out", ignore_errors=True)
        self.assertIn("model file: SHA-256", self.go("check", expect=1))
        self.fx.model.write_bytes(b"fake historical model " * 100)
        cfg = copy.deepcopy(self.fx.config)
        cfg["variants"]["face_crop"]["detector"]["sha256"] = "3" * 64
        self.fx.write_config(cfg)
        shutil.rmtree(self.tmp / "out", ignore_errors=True)
        self.assertIn("face.py: SHA-256", self.go("check", expect=1))
        self.fx.write_config()
        self.fx.resources["Widerface-RetinaFace.caffemodel"].write_bytes(b"changed")
        shutil.rmtree(self.tmp / "out", ignore_errors=True)
        self.assertIn("detector resource Widerface-RetinaFace.caffemodel", self.go("check", expect=1))
        self.fx.resources["Widerface-RetinaFace.caffemodel"].write_bytes(b"fake Widerface-RetinaFace.caffemodel")
        ref = TMP_ROOT / "livedetection" / "src" / "dataset_loader.py"
        ref.write_bytes(b"changed")
        try:
            shutil.rmtree(self.tmp / "out", ignore_errors=True)
            self.assertIn("reference code dataset_loader.py", self.go("check", expect=1))
        finally:
            ref.write_bytes(REF_CODE[("src", "dataset_loader.py")])

    def test_n1_aggregate(self):
        p = self.fx.n1 / "aggregate.json"
        p.write_text(p.read_text() + "\n")
        self.assertIn("N1 aggregate.json missing or SHA-256", self.go("check", expect=1))
        doc = copy.deepcopy(self.fx.n1_doc)
        doc["threshold"] = 0.4
        p.write_text(json.dumps(doc, indent=1) + "\n")
        cfg = copy.deepcopy(self.fx.config)
        cfg["n1_reference"]["aggregate_sha256"] = sha(p.read_bytes())
        self.fx.write_config(cfg)
        shutil.rmtree(self.tmp / "out", ignore_errors=True)
        self.assertIn("same threshold", self.go("check", expect=1))
        doc = copy.deepcopy(self.fx.n1_doc)
        del doc["seeds"]["43"]["metrics"]["c1_300"]
        p.write_text(json.dumps(doc, indent=1) + "\n")
        cfg["n1_reference"]["aggregate_sha256"] = sha(p.read_bytes())
        self.fx.write_config(cfg)
        shutil.rmtree(self.tmp / "out", ignore_errors=True)
        self.assertIn("metrics of seed 43 missing", self.go("check", expect=1))


class ScoreTests(Base):
    def test_pass_both_variants(self):
        self.pipeline("score")
        for v in C.VARIANTS:
            vd = self.study / v
            rows = list(csv.DictReader(io.StringIO((vd / "scores_test.csv").read_text())))
            self.assertEqual([r["sample_id"] for r in rows], [r["key"] for r in self.fx.test])
            X = np.stack([frame_of(content_of(r["key"]), v)[0] for r in self.fx.test])
            exp = fake_prob(X, self.fx.config["model"]["sha256"])[:, 1]
            for r, e in zip(rows, exp):
                self.assertEqual(float(r["score_attack"]), float(e))
                self.assertEqual(r["prediction"], "1" if e > 0.5 else "0")
                self.assertEqual(r["model_sha256"], self.fx.config["model"]["sha256"])
            sc = json.loads((vd / "score.json").read_text())
            dets = {"n/a": len(rows)} if v == "whole_image" else {k: n for k, n in sc["detectors"].items()}
            self.assertEqual(sc["detectors"], dets)
            self.assertEqual(sc["fallback_whole_image"], None if v == "whole_image" else sc["detectors"].get("none", 0))
            met = json.loads((vd / "metrics.json").read_text())["metrics"]
            self.assertEqual(met["test_3000"]["n"], len(self.fx.test))
            self.assertEqual(met["c1_300"]["n"], len(self.fx.rows["c1"]))
        self.assertNotEqual((self.study / "whole_image" / "scores_test.csv").read_text(), (self.study / "face_crop" / "scores_test.csv").read_text())

    def test_refusals(self):
        self.go("check")
        self.assertIn("--i-am-authorized", self.go("score", "--variant", "whole_image", expect=1))
        self.assertIn("--variant must be one of", self.go("score", "--variant", "other", "--i-am-authorized", expect=1))
        self.assertIn("needs --variant", self.go("score", "--i-am-authorized", expect=1))
        self.go("score", "--variant", "whole_image", "--i-am-authorized")
        before = self.listing()
        self.assertIn("never reused", self.go("score", "--variant", "whole_image", "--i-am-authorized", expect=1))
        self.assertEqual(self.listing(), before)

    def test_requires_check_and_code_unchanged(self):
        self.assertIn("run check first", self.go("score", "--variant", "whole_image", "--i-am-authorized", expect=1))
        self.go("check")
        with mock.patch.object(C, "code_hashes", lambda: {"x": "0" * 64}):
            self.assertIn("code changed since check", self.go("score", "--variant", "whole_image", "--i-am-authorized", expect=1))

    def test_wrong_frame_shape_and_duplicates(self):
        self.go("check")
        with mock.patch.object(C, "PREPROCESSOR", lambda v, cfg: (lambda p: (np.zeros((8, 8, 3), np.uint8), "n/a"))):
            text = self.go("score", "--variant", "whole_image", "--i-am-authorized", expect=1)
        self.assertIn("preprocessed frame has shape", text)
        with mock.patch.object(C, "PREPROCESSOR", lambda v, cfg: (lambda p: (np.zeros((256, 256, 3), np.uint8), "n/a"))):
            text = self.go("score", "--variant", "face_crop", "--i-am-authorized", expect=1)
        self.assertIn("duplicated derived frames", text)
        self.assertNotIn("face_crop/scores_test.csv", self.listing())

    def test_nan_scores_refused(self):
        self.go("check")
        with mock.patch.object(C, "MAKE_SCORER", lambda p, s: (lambda X: np.full((len(X), 2), np.nan))):
            self.assertIn("non-finite", self.go("score", "--variant", "whole_image", "--i-am-authorized", expect=1))

    def test_inputs_changed_after_check(self):
        self.go("check")
        self.fx.model.write_bytes(b"changed" * 100)
        self.assertIn("frozen code check failed", self.go("score", "--variant", "whole_image", "--i-am-authorized", expect=1))

    def test_check_record_tampered(self):
        self.go("check")
        p = self.study / "check.json"
        doc = json.loads(p.read_text())
        doc["checks"]["manifests"]["cnn_test"]["rows"] = 999
        p.write_text(json.dumps(doc, indent=1) + "\n")
        self.assertIn("manifests differ from those recorded at check", self.go("score", "--variant", "whole_image", "--i-am-authorized", expect=1))
        self.assertFalse((self.study / "whole_image").exists())

    def test_tie_at_threshold_is_bona_fide(self):
        self.go("check")
        with mock.patch.object(C, "MAKE_SCORER", lambda p, s: (lambda X: np.full((len(X), 2), 0.5))):
            self.go("score", "--variant", "whole_image", "--i-am-authorized")
        rows = list(csv.DictReader(io.StringIO((self.study / "whole_image" / "scores_test.csv").read_text())))
        self.assertTrue(all(r["prediction"] == "0" for r in rows))
        met = json.loads((self.study / "whole_image" / "metrics.json").read_text())["metrics"]["test_3000"]
        self.assertEqual((met["ties_at_threshold"], met["rejected_bona_fide"], met["accepted_attacks"]), (len(rows), 0, met["n_attack"]))


class AggregateTests(Base):
    def test_full_pipeline(self):
        text = self.pipeline()
        for n in C.AGG_OUTPUTS:
            self.assertTrue((self.study / n).is_file(), n)
        doc = json.loads((self.study / "aggregate.json").read_text())
        self.assertEqual(list(doc["variants"]), list(C.VARIANTS))
        for v in C.VARIANTS:
            rows = list(csv.DictReader(io.StringIO((self.study / v / "scores_test.csv").read_text())))
            self.assertEqual(doc["variants"][v]["metrics"], C.metrics_from_rows(rows))
            h = doc["variants"][v]["distributions"]["test_3000"]
            self.assertEqual(sum(h["bona_fide"]) + sum(h["attack"]), len(self.fx.test))
        self.assertEqual(doc["n1_reference"]["seeds"]["43"]["test_3000"]["apcer"], 0.4)
        md = (self.study / "aggregate.md").read_text()
        self.assertIn("| combined AttackNetV2.2, whole image | test (3,000 frames of nuaa_cnn_test) |", md)
        self.assertIn("| N1 LivenessNet, seed 44 |", md)
        for n in ("aggregate.json", "aggregate.md"):
            t = (self.study / n).read_text()
            for s in SUBJECTS:
                self.assertNotRegex(t, rf"(?<![\w.]){s}(?![\w.])", n)
            self.assertNotRegex(t, r"\b(real|attack)/\d{4}_", n)
        subj = list(csv.DictReader(io.StringIO((self.study / "per_subject_all_variants.csv").read_text())))
        self.assertEqual(len(subj), 2 * (len(SUBJECTS) + len({r["subject"] for r in self.fx.rows["c1"]})))
        self.assertIn("aggregate PASS", text)

    def test_missing_variant_and_tampering(self):
        self.go("check")
        self.go("score", "--variant", "whole_image", "--i-am-authorized")
        self.assertIn("face_crop/score.json", self.go("aggregate", expect=1))
        self.go("score", "--variant", "face_crop", "--i-am-authorized")
        p = self.study / "face_crop" / "metrics.json"
        d = json.loads(p.read_text())
        d["metrics"]["test_3000"]["acer"] = 0.0
        p.write_text(json.dumps(d, indent=1) + "\n")
        self.assertIn("face_crop/metrics.json changed after scoring", self.go("aggregate", expect=1))
        self.assertNotIn("aggregate.json", self.listing())

    def test_scores_tampered_with_redeclared_sha(self):
        self.pipeline("score")
        vd = self.study / "whole_image"
        rows = list(csv.DictReader(io.StringIO((vd / "scores_test.csv").read_text())))
        rows[0]["label"] = "1" if rows[0]["label"] == "0" else "0"
        txt = csv_text(rows, C.SCORE_COLUMNS)
        (vd / "scores_test.csv").write_text(txt)
        sc = json.loads((vd / "score.json").read_text())
        sc["files"]["scores_test.csv"] = sha(txt.encode())
        (vd / "score.json").write_text(json.dumps(sc, indent=1) + "\n")
        self.assertIn("rows inconsistent", self.go("aggregate", expect=1))

    def test_metrics_recomputed(self):
        self.pipeline("score")
        vd = self.study / "whole_image"
        rows = list(csv.DictReader(io.StringIO((vd / "scores_test.csv").read_text())))
        v = float(rows[0]["score_attack"])
        rows[0]["score_attack"] = repr(0.99 if v <= 0.5 else 0.01)
        rows[0]["prediction"] = "1" if float(rows[0]["score_attack"]) > 0.5 else "0"
        txt = csv_text(rows, C.SCORE_COLUMNS)
        (vd / "scores_test.csv").write_text(txt)
        sc = json.loads((vd / "score.json").read_text())
        sc["files"]["scores_test.csv"] = sha(txt.encode())
        (vd / "score.json").write_text(json.dumps(sc, indent=1) + "\n")
        self.assertIn("differ from metrics.json", self.go("aggregate", expect=1))

    def test_per_subject_and_detectors_redeclared(self):
        self.pipeline("score")
        vd = self.study / "whole_image"
        rows = list(csv.DictReader(io.StringIO((vd / "per_subject.csv").read_text())))
        rows[0]["accepted_attacks"] = str(int(rows[0]["accepted_attacks"]) + 1)
        txt = csv_text(rows, C.SUBJECT_COLUMNS)
        (vd / "per_subject.csv").write_text(txt)
        sc = json.loads((vd / "score.json").read_text())
        sc["files"]["per_subject.csv"] = sha(txt.encode())
        (vd / "score.json").write_text(json.dumps(sc, indent=1) + "\n")
        self.assertIn("per_subject.csv differs from the table recomputed", self.go("aggregate", expect=1))
        shutil.rmtree(self.tmp / "out")
        self.pipeline("score")
        sc = json.loads((self.study / "face_crop" / "score.json").read_text())
        sc["detectors"] = {"RetinaFace": sum(sc["detectors"].values())}
        (self.study / "face_crop" / "score.json").write_text(json.dumps(sc, indent=1) + "\n")
        self.assertIn("detector counts", self.go("aggregate", expect=1))

    def test_public_scan_and_plotter(self):
        self.pipeline("score")
        with mock.patch.object(N1, "public_problems", lambda text, inp: ["image keys"]):
            self.assertIn("would hold image keys: not written", self.go("aggregate", expect=1))
        with mock.patch.object(C, "PLOTTER", lambda doc: {"figure_distributions.png": b"no"}):
            self.assertIn("did not return the PNG figure", self.go("aggregate", expect=1))
        for n in C.AGG_OUTPUTS:
            self.assertFalse((self.study / n).exists(), n)
        self.go("aggregate")
        (self.study / "aggregate.md").write_text("stale\n")
        self.assertIn("removed previous aggregate.md", self.go("aggregate"))

    def test_tampered_after_pass_leaves_no_pass(self):
        self.pipeline()
        (self.study / "whole_image" / "frames.csv").write_text("x\n")
        self.assertIn("changed after scoring", self.go("aggregate", expect=1))
        for n in C.AGG_OUTPUTS:
            self.assertFalse((self.study / n).exists(), n)
        self.assertEqual(self.status()["result"], "FAIL")


class EnvironmentTests(unittest.TestCase):
    def test_tensorflow_never_imported(self):
        self.assertNotIn("tensorflow", sys.modules)

    def test_code_files_exist(self):
        for f in C.CODE_FILES:
            self.assertTrue((ROOT / f).is_file(), f)

    def test_historical_chain_order(self):
        """La catena storica: ridimensionamento Lanczos PRIMA dell'enhancement, uint8 in uscita (con un enhancement finto)."""
        calls = []

        class Creator:
            def advanced_image_enhancement(self, img):
                calls.append(img.shape)
                return img
        fake_cv2 = types.SimpleNamespace(INTER_LANCZOS4=4, resize=lambda img, size, interpolation: np.zeros((size[1], size[0], 3), np.uint8))
        with mock.patch.dict(sys.modules, {"cv2": fake_cv2}):
            out = C.historical_chain(np.zeros((10, 20, 3), np.uint8), Creator())
        self.assertEqual(out.shape, (256, 256, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(calls, [(256, 256, 3)])

    def test_tf_scorer_input_scaling(self):
        """Lo scorer vero porta i frame in [-1, 1] con (x / 255 - 0,5) * 2 (modello finto che restituisce l'ingresso)."""
        tmp = Path(tempfile.mkdtemp(dir=TMP_ROOT))
        m = tmp / "m.h5"
        m.write_bytes(b"m")
        seen = []

        class Model:
            input_shape = (None, 256, 256, 3)
            output_shape = (None, 2)

            def __call__(self, xb, training=False):
                seen.append(np.asarray(xb))
                return np.tile([[0.25, 0.75]], (len(xb), 1))
        fake_tf = types.SimpleNamespace(config=types.SimpleNamespace(set_visible_devices=lambda a, b: None),
                                        device=lambda name: contextlib.nullcontext(), keras=types.SimpleNamespace(models=types.SimpleNamespace(load_model=lambda p, compile: Model())))
        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            score = C.tf_model_scorer(m, sha(b"m"))
            X = np.zeros((3, 256, 256, 3), np.uint8)
            X[1] = 255
            X[2] = 127
            p = score(X)
        self.assertEqual(p.shape, (3, 2))
        self.assertEqual(float(seen[0][0, 0, 0, 0]), -1.0)
        self.assertEqual(float(seen[0][1, 0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(seen[0][2, 0, 0, 0]), (127 / 255 - 0.5) * 2, places=6)
        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}), self.assertRaises(C.N2Error):
            C.tf_model_scorer(m, "0" * 64)
        sys.modules.pop("tensorflow", None)


class MutationHarnessTests(unittest.TestCase):
    def test_classify(self):
        import check_n2_combined_mutations as M
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
        self.assertEqual((ROOT / "results" / "n2").exists(), REPO_N2_BEFORE)
        if REPO_N2_BEFORE:
            self.assertEqual(sorted(p.name for p in (ROOT / "results" / "n2").iterdir()), REPO_N2_LISTING)
        for where in ("scripts", "results", "results/n1"):
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
