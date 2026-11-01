"""
Test di scripts/n2_sessions.py su dati SINTETICI, in cartelle temporanee: nessuna rete, nessuna immagine vera, nessun
dato NUAA, nessun TensorFlow, nessun rilevatore (pre-elaborazione, scorer, figure e stato git sono iniettati).

Le fixture sono manifest NUAA finti con lo schema M2, file "originali" finti (byte con SHA-256 e dimensione nel
manifest), una cache .npz finta costruita con la pre-elaborazione finta sugli stessi file (così cache-check e parity
hanno un vero confronto bit per bit), uno studio N1 finto (record, pesi finti, punteggi dello scorer finto) e una
configurazione coerente con tutto ciò. I run sono -trial, salvo i test del vincolo sulla configurazione consegnata.

Coperti: configurazione (schema chiuso, regole fisse, SHA-256 vincolato nei run non -trial), cartella dello studio,
controlli di check (cache, manifest, derivazione e conteggi dei 491, disgiunzione, originali, studio N1, codice
congelato), cache-check (frame diversi, duplicati, passo), parity (frame e tolleranza), score (autorizzazione, seed,
frame uguali a cache-check, regola di decisione), aggregate (ricalcolo, SHA dei file, strati, distribuzioni,
scansione del pubblico, figure, nessuna scelta del seed), percorsi d'errore (nulla scritto o nessun PASS vecchio) e un
controllo finale che il repository vero non sia stato toccato.

  python scripts/check_n2_sessions.py        # uscita 1 se un test fallisce; ultima riga RESULT: PASS / FAIL

Dipendenze: numpy e scikit-learn. Senza OpenCV (cv2) la suite registra un modulo cv2 vuoto (evaluation.py lo importa
solo per leggere immagini) e lo dichiara con una riga NOTE.
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
import statistics
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
STUDY = "20261030-N2-sessions-abcdef0-trial"
STUDY_REAL = "20261030-N2-sessions-abcdef0"
COMMIT = "abcdef0" + "0" * 33

TMP_ROOT = Path(tempfile.mkdtemp(prefix="n2-sessions-tests-"))
atexit.register(shutil.rmtree, TMP_ROOT, True)
for _k, _s in (("LIVENESSLAB_DATA_DIR", "data"), ("LIVENESSLAB_RESULTS_DIR", "results"), ("LIVENESSLAB_WEIGHTS_DIR", "weights"),
               ("LIVENESSLAB_LIVEDETECTION", "livedetection"), ("LIVENESSLAB_SILENT_FACE", "silent_face")):
    os.environ[_k] = str(TMP_ROOT / _s)
    (TMP_ROOT / _s).mkdir()
REF_CODE = {("scripts", "create_datasets.py"): b"# fake reference create_datasets\n", ("src", "architectures.py"): b"# fake reference architectures\n"}
for _rel, _data in REF_CODE.items():
    _p = TMP_ROOT / "livedetection" / Path(*_rel)
    _p.parent.mkdir(parents=True, exist_ok=True)
    _p.write_bytes(_data)
REPO_N2_BEFORE = (ROOT / "results" / "n2").exists()
REPO_N2_LISTING = sorted(p.name for p in (ROOT / "results" / "n2").iterdir()) if REPO_N2_BEFORE else None

MISSING_INPUT = {"delivered configuration": "n2_sessions.json and face_0ac7143.py are not next to the code (delivery/n2/, or ../ in the package)"}

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
import n2_sessions as N  # noqa: E402
import n1_train as N1  # noqa: E402
import nuaa_manifests as nm  # noqa: E402

PER_CLASS = 12
SUBJECTS = tuple(f"{i:04d}" for i in range(1, 9))
TEST_ONLY = tuple(f"{i:04d}" for i in range(9, 11))          # soggetti presenti solo nel test (come 7 dei 16 soggetti NUAA)
PNG = b"\x89PNG\r\n\x1a\n" + b"fake"
N1_RUN = "20261024-N1-LivenessNet-fb93496"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_of(key: str) -> bytes:
    return f"image:{key}".encode()


def row(label: int, subject: str, session: str, pic: int) -> dict:
    name = f"{subject}_{subject}_0{pic % 2}_0{(pic % 3) if label else 0}_{session}_{pic}.jpg"
    key = nm.make_key(label, name)
    data = content_of(key)
    return {"key": key, "label": str(label), **nm.parse_name(name), "official_split": nm.SESSION_SPLIT[session], "sha256": sha(data),
            "bytes": str(len(data))}


def csv_text(rows, cols) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def write_csv(p: Path, rows, sort: bool = True) -> None:
    p.write_text(csv_text(sorted(rows, key=lambda r: r["key"]) if sort else list(rows), nm.COLUMNS), encoding="utf-8")


def frame_of(data: bytes):
    """Pre-elaborazione finta: frame 8x8x3 deterministico dai byte del file e nome breve del rilevatore."""
    h = hashlib.sha256(b"frame:" + data).digest()
    X = np.frombuffer((h * 6)[:8 * 8 * 3], dtype=np.uint8).reshape(8, 8, 3).copy()
    return X, ("Haar" if h[0] % 5 == 0 else "RetinaFace")


def fake_preprocessor(frozen_face, cfg):
    def pre(path):
        return frame_of(Path(path).read_bytes())
    return pre


def fake_prob(X, weights_sha: str) -> np.ndarray:
    out = np.zeros((len(X), 2), dtype=np.float64)
    for i, x in enumerate(np.asarray(X)):
        v = int.from_bytes(hashlib.sha256(np.ascontiguousarray(x).tobytes() + weights_sha.encode()).digest()[:4], "big") / 2 ** 32
        out[i] = (1 - v, v)
    return out


def fake_scorer(path, weights_sha):
    return lambda X: fake_prob(X, weights_sha)


def fake_plotter(doc, strata):
    return {"figure_distributions.png": PNG, "figure_groups.png": PNG + b"g", "figure_strata.png": PNG + b"s"}


def fake_repo_state(d):
    return {"commit": COMMIT, "dirty": False, "dirty_paths": [], "dirty_excludes": None}


class Fixture:
    """Manifest, originali, cache, studio N1 e configurazione sintetici, coerenti fra loro."""

    def __init__(self, base: Path):
        self.base = base
        self.man, self.images, self.n1 = base / "manifests", base / "images", base / "n1" / N1_RUN
        for d in (self.man, self.images, self.n1):
            d.mkdir(parents=True)
        tr = [row(lab, s, sess, p) for s in SUBJECTS for lab in (0, 1) for sess in ("01", "02") for p in range(3)]
        te = [row(lab, s, "03", p) for s in SUBJECTS + TEST_ONLY for lab in (0, 1) for p in range(4)]
        self.rows = {"official_train": tr, "official_test": te}
        by_tr, by_te = {r["key"]: r for r in tr}, {r["key"]: r for r in te}
        self.order_train, self.order_test = nm.spread_keys(tr, PER_CLASS), nm.spread_keys(te, PER_CLASS)
        subj = sorted({r["subject"] for r in tr})
        self.val_subjects, self.fit_subjects = subj[-2:], subj[:-2]
        self.rows["cnn_train"] = [by_tr[k] for k in self.order_train if by_tr[k]["subject"] in self.fit_subjects]
        self.rows["cnn_es"] = [by_tr[k] for k in self.order_train if by_tr[k]["subject"] in self.val_subjects]
        self.rows["cnn_test"] = [by_te[k] for k in self.order_test]
        self.rows["c1"] = [by_te[k] for i, k in enumerate(self.order_test) if i % 3 == 0]
        self.n_test = len(self.order_test)
        used = set(self.order_train)
        self.target = sorted((r for r in tr if r["key"] not in used), key=lambda r: r["key"])
        # dev M1: la validazione più alcuni frame del target dei soggetti di validazione
        self.rows["m1_dev"] = self.rows["cnn_es"] + [r for r in self.target if r["subject"] in self.val_subjects][:5]
        self.files = {"official_train": "nuaa_official_train.csv", "official_test": "nuaa_official_test.csv", "cnn_train": "nuaa_cnn_train.csv",
                      "cnn_es": "nuaa_cnn_early_stopping.csv", "cnn_test": "nuaa_cnn_test.csv", "c1": "nuaa_c1_v2.csv", "m1_dev": "nuaa_m1_dev.csv"}
        for n, rows in self.rows.items():
            write_csv(self.man / self.files[n], rows if n != "cnn_test" else list(reversed(rows)), sort=(n != "cnn_test"))   # cnn_test in ordine inverso di campionamento
        for r in tr + te:
            p = self.images / N.LAYOUTS["images"](r)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content_of(r["key"]))
        self.X = {"train": np.stack([frame_of(content_of(k))[0] for k in self.order_train]),
                  "test": np.stack([frame_of(content_of(k))[0] for k in self.order_test])}
        self.cache = base / "nuaa_1500.npz"
        np.savez(self.cache, X_train=self.X["train"], y_train=np.array([int(by_tr[k]["label"]) for k in self.order_train], np.int64),
                 s_train=np.array([by_tr[k]["subject"] for k in self.order_train]), X_test=self.X["test"],
                 y_test=np.array([int(by_te[k]["label"]) for k in self.order_test], np.int64), s_test=np.array([by_te[k]["subject"] for k in self.order_test]))
        self.frozen = base / "face_0ac7143.py"
        self.frozen.write_text("# fake frozen face module\n", encoding="utf-8")
        self.resources = {}
        for n in ("deploy.prototxt", "Widerface-RetinaFace.caffemodel", "haarcascade_frontalface_default.xml"):
            p = base / "resources" / n
            p.parent.mkdir(exist_ok=True)
            p.write_bytes(f"fake {n}".encode())
            self.resources[n] = p
        self.n1_config_sha = sha(b"fake n1 configuration")
        self.n1_check = {"passed": True, "step": "check", "run_id": N1_RUN, "config_sha256": self.n1_config_sha, "commit": "fb93496"}
        (self.n1 / "check.json").write_text(json.dumps(self.n1_check, indent=1) + "\n", encoding="utf-8")
        self.n1_check_sha = sha((self.n1 / "check.json").read_bytes())
        self.seeds = {}
        for seed in (42, 43, 44):
            sd = self.n1 / f"seed{seed}"
            (sd / "checkpoints").mkdir(parents=True)
            w = sd / "checkpoints" / "LivenessNet__nuaa.h5"
            w.write_bytes(f"fake weights {seed}".encode())
            wsha = sha(w.read_bytes())
            card = sd / "checkpoints" / "LivenessNet__nuaa.json"
            card.write_text(json.dumps({"seed": seed}), encoding="utf-8")
            csha = sha(card.read_bytes())
            train = {"passed": True, "step": "train", "run_id": N1_RUN, "config_sha256": self.n1_config_sha, "check_sha256": self.n1_check_sha,
                     "train_seed": seed, "weights_sha256": wsha, "card_sha256": csha}
            (sd / "train.json").write_text(json.dumps(train, indent=1) + "\n", encoding="utf-8")
            p = fake_prob(self.X["test"], wsha)[:, 1]
            srows = []
            for i, k in enumerate(self.order_test):
                r = by_te[k]
                v = float(p[i])
                srows.append({"run_id": N1_RUN, "train_seed": str(seed), "sample_id": k, "subject_id": r["subject"], "session_id": r["session"],
                              "label": r["label"], "score_attack": repr(v), "prediction": "1" if v > 0.5 else "0",
                              "in_c1_subset": "1" if k in {x["key"] for x in self.rows["c1"]} else "0", "file_sha256": r["sha256"],
                              "checkpoint_sha256": wsha, "config_sha256": self.n1_config_sha})
            srows.sort(key=lambda r: r["sample_id"])
            txt = csv_text(srows, N1.SCORE_COLUMNS)
            (sd / "scores_test.csv").write_text(txt, encoding="utf-8")
            score = {"passed": True, "step": "score", "run_id": N1_RUN, "config_sha256": self.n1_config_sha, "check_sha256": self.n1_check_sha,
                     "train_seed": seed, "weights_sha256": wsha, "files": {"scores_test.csv": sha(txt.encode())}}
            (sd / "score.json").write_text(json.dumps(score, indent=1) + "\n", encoding="utf-8")
            self.seeds[str(seed)] = {"weights_sha256": wsha, "card_sha256": csha, "scores_test_sha256": sha(txt.encode())}
        self.config = self.make_config()
        self.config_path = base / "n2_sessions.json"
        self.write_config()

    def parity_keys(self) -> list:
        by = {}
        for r in sorted(self.rows["cnn_test"], key=lambda r: r["key"]):
            by.setdefault((r["subject"], r["label"]), r["key"])
        return [by[k] for k in sorted(by)]

    def make_config(self) -> dict:
        t = self.target
        m1 = {r["key"] for r in self.rows["m1_dev"]}
        mans = {n: {"file": self.files[n], "sha256": sha((self.man / self.files[n]).read_bytes()), **N1.counts_of(self.rows[n])} for n in N.MANIFESTS}
        return {
            "schema": N.CONFIG_SCHEMA, "study": "synthetic", "architecture": "LivenessNet", "dataset": "nuaa", "protocol": "official",
            "n1_study": {"run_id": N1_RUN, "config_sha256": self.n1_config_sha, "check_sha256": self.n1_check_sha,
                         "train_seeds": [42, 43, 44], "seeds": self.seeds, "weights_file": "checkpoints/LivenessNet__nuaa.h5",
                         "scores_file": "scores_test.csv", "scores_columns": list(N1.SCORE_COLUMNS)},
            "preprocessing": {"frozen_face": {"file": "face_0ac7143.py", "sha256": sha(self.frozen.read_bytes()), "source": "synthetic", "loaded_as": "n/a"},
                              "detector_resources": {n: sha(p.read_bytes()) for n, p in self.resources.items()},
                              "chain": "synthetic", "reference_code_sha256": {k: sha(REF_CODE[tuple(k.split("/")[1:])]) for k in N.REFERENCE_CODE},
                              "cache_environment": "synthetic"},
            "cache_check": {"rule": "bit for bit", "frames": 4 * PER_CLASS, "step": 1},
            "parity": {"rule": "synthetic", "selection": "first key per subject and class", "tolerance_abs": 1e-06, "n_keys": len(self.parity_keys()),
                       "keys_sha256": sha(("\n".join(self.parity_keys()) + "\n").encode())},
            "target": {"name": "sessions_01_02", "definition": "official_train - cnn_train - cnn_es", "manifest_file": "target_manifest.csv",
                       "manifest_sha256": sha(csv_text(t, nm.COLUMNS).encode()), "rows": len(t),
                       "n_bona_fide": sum(1 for r in t if r["label"] == "0"), "n_attack": sum(1 for r in t if r["label"] == "1"),
                       "n_subjects": len({r["subject"] for r in t}),
                       "by_session": {s: sum(1 for r in t if r["session"] == s) for s in sorted({r["session"] for r in t})},
                       "in_m1_dev": sum(1 for r in t if r["key"] in m1), "in_c1": 0, "in_cnn_test": 0,
                       "groups": {"fitting": "cnn_train subjects", "validation": "cnn_es subjects"}, "originals": "synthetic"},
            "reference": {"name": "session_03", "definition": "N1 scores", "strata": "subject x class"},
            "scoring": {"device": "CPU", "batch": 16, "batchnorm": "inference", "input": "/255", "score": "column 1", "scorer": "n1_train.tf_scorer"},
            "metrics": {"threshold": 0.5, "decision": N.DECISION_RULE, "reported": ["APCER", "BPCER", "ACER", "AUROC"], "sets": list(N.SETS),
                        "aggregation": "median and range", "distributions": "10 bins", "public": "aggregate", "restricted": "rows"},
            "data": {"m2_run": "synthetic", "cache": {"file": "nuaa_1500.npz", "sha256": sha(self.cache.read_bytes()), "per_class": PER_CLASS,
                                                      "members": {"X_train": {"dtype": "|u1", "shape": [2 * PER_CLASS, 8, 8, 3]},
                                                                  "y_train": {"dtype": "<i8", "shape": [2 * PER_CLASS]},
                                                                  "s_train": {"dtype": "<U4", "shape": [2 * PER_CLASS]},
                                                                  "X_test": {"dtype": "|u1", "shape": [2 * PER_CLASS, 8, 8, 3]},
                                                                  "y_test": {"dtype": "<i8", "shape": [2 * PER_CLASS]},
                                                                  "s_test": {"dtype": "<U4", "shape": [2 * PER_CLASS]}}},
                     "manifests": mans}}

    def write_config(self, cfg=None, path=None) -> str:
        cfg = self.config if cfg is None else cfg
        p = self.config_path if path is None else path
        raw = (json.dumps(cfg, indent=1) + "\n").encode()
        p.write_bytes(raw)
        return sha(raw)

    def args(self, study=None, **over) -> list:
        base = {"config": self.config_path, "manifests": self.man, "cache": self.cache, "images": self.images, "n1_study": self.n1,
                "frozen_face": self.frozen, "study_dir": self.base / "out" / (study or STUDY)}
        base.update(over)
        out = []
        for k, v in base.items():
            out += [f"--{k.replace('_', '-')}", str(v)]
        return out


def run(*args) -> tuple:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = N.main(list(args))
    return code, out.getvalue()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="n2-case-", dir=TMP_ROOT))
        self.fx = Fixture(self.tmp)
        self.study = self.tmp / "out" / STUDY
        self.patches = [mock.patch.object(N, "PREPROCESSOR", fake_preprocessor), mock.patch.object(N, "RESOURCE_PATHS", self.fx.resources),
                        mock.patch.object(N, "PLOTTER", fake_plotter), mock.patch.object(N1, "MAKE_SCORER", fake_scorer),
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

    def pipeline(self, upto="aggregate", study=None) -> str:
        text = self.go("check", study=study)
        if upto == "check":
            return text
        text = self.go("cache-check", study=study)
        if upto == "cache-check":
            return text
        text = self.go("parity", study=study)
        if upto == "parity":
            return text
        for s in (42, 43, 44):
            text = self.go("score", "--train-seed", str(s), "--i-am-authorized", study=study)
        if upto == "score":
            return text
        return self.go("aggregate", study=study)

    def status(self, d=None) -> dict:
        return json.loads(((d or self.study) / "status.json").read_text(encoding="utf-8"))

    def listing(self, d=None) -> list:
        d = d or self.study
        return sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()) if d.exists() else []


# ============================================================================ configurazione

class ConfigTests(Base):
    def test_check_pass_writes_only_its_files(self):
        text = self.go("check")
        self.assertEqual(self.listing(), ["check.json", "check.log", "parity_keys.csv", "status.json", "target_manifest.csv"])
        self.assertEqual(self.status()["result"], "PASS")
        doc = json.loads((self.study / "check.json").read_text())
        self.assertTrue(doc["trial"])
        self.assertFalse(doc["delivered_configuration"])
        self.assertEqual(doc["checks"]["target"]["rows"], len(self.fx.target))
        self.assertEqual(doc["detector"], "not used")
        self.assertEqual(doc["code_sha256"]["frozen_face"], self.fx.config["preprocessing"]["frozen_face"]["sha256"])
        self.assertEqual(sha((self.study / "target_manifest.csv").read_bytes()), self.fx.config["target"]["manifest_sha256"])
        self.assertIn("PASS target", text)

    def refused(self, mutate, needle: str):
        cfg = copy.deepcopy(self.fx.config)
        mutate(cfg)
        self.fx.write_config(cfg)
        text = self.go("check", expect=1)
        self.assertIn(needle, text)
        self.assertFalse(self.study.exists(), text)

    def test_missing_or_extra_key(self):
        self.refused(lambda c: c.pop("reference"), "missing or not allowed")
        self.refused(lambda c: c.update(extra=1), "missing or not allowed")

    def test_fixed_rules(self):
        self.refused(lambda c: c["metrics"].update(threshold=0.4), "threshold")
        self.refused(lambda c: c["metrics"].update(decision="attack if score_attack >= 0.5"), "decision rule")
        self.refused(lambda c: c["parity"].update(tolerance_abs=1e-05), "tolerance_abs")
        self.refused(lambda c: c["cache_check"].update(step=10), "step 1")
        self.refused(lambda c: c["cache_check"].update(frames=6000), "every cache frame")
        self.refused(lambda c: c["scoring"].update(device="GPU"), "CPU")
        self.refused(lambda c: c["n1_study"].update(train_seeds=[42, 43]), "[42, 43, 44]")
        self.refused(lambda c: c["metrics"].update(sets=list(N.SETS)[:-1]), "reported sets")
        self.refused(lambda c: c.update(schema="other/1"), "schema")

    def test_parity_keys_validated(self):
        self.refused(lambda c: c["parity"].update(keys=["real/x.jpg"]), "no key list")
        self.refused(lambda c: c["parity"].update(n_keys=0), "n_keys")
        self.refused(lambda c: c["parity"].update(keys_sha256="xyz"), "keys_sha256")

    def test_target_counts_consistent(self):
        self.refused(lambda c: c["target"].update(in_c1=1), "not consistent")
        self.refused(lambda c: c["target"].update(rows=c["target"]["rows"] + 1), "not consistent")

    def test_malformed_sha(self):
        self.refused(lambda c: c["preprocessing"]["frozen_face"].update(sha256="abc"), "frozen_face")
        self.refused(lambda c: c["data"]["manifests"]["c1"].update(sha256="zz"), "data.manifests.c1")

    def test_non_trial_requires_delivered_configuration(self):
        text = self.go("check", study=STUDY_REAL, expect=1)
        self.assertIn("not the delivered one", text)
        self.assertFalse((self.tmp / "out" / STUDY_REAL).exists())
        with mock.patch.object(N, "CONFIG_SHA256", sha(self.fx.config_path.read_bytes())):
            self.go("check", study=STUDY_REAL)
            doc = json.loads((self.tmp / "out" / STUDY_REAL / "check.json").read_text())
            self.assertTrue(doc["delivered_configuration"])
            self.assertFalse(doc["trial"])

    def test_non_trial_study_id_carries_the_commit(self):
        with mock.patch.object(N, "CONFIG_SHA256", sha(self.fx.config_path.read_bytes())):
            text = self.go("check", study="20261030-N2-sessions-1234567", expect=1)
            self.assertIn("must carry the current commit", text)

    def test_delivered_configuration(self):
        cands = [ROOT / "delivery" / "n2" / "n2_sessions.json", ROOT.parent / "n2_sessions.json"]
        cfg = next((c for c in cands if c.is_file() and (c.parent / "face_0ac7143.py").is_file()), None)
        if cfg is None:
            self.skipTest("missing input: " + MISSING_INPUT["delivered configuration"])
        raw = cfg.read_bytes()
        doc = json.loads(raw.decode("utf-8"))
        self.assertEqual(N.config_problems(doc), [])
        self.assertEqual(sha(raw), N.CONFIG_SHA256, "the delivered configuration is not the one bound in n2_sessions.py")
        self.assertEqual(sha((cfg.parent / "face_0ac7143.py").read_bytes()), doc["preprocessing"]["frozen_face"]["sha256"])
        self.assertEqual(doc["target"]["rows"], 491)
        self.assertEqual((doc["target"]["n_bona_fide"], doc["target"]["n_attack"], doc["target"]["in_m1_dev"]), (243, 248, 81))
        self.assertEqual(doc["parity"]["n_keys"], 24)
        self.assertNotIn("keys", doc["parity"])


# ============================================================================ cartella dello studio

class StudyFolderTests(Base):
    def test_bad_name_refused(self):
        for name in ("20261030-N1-LivenessNet-abcdef0-trial", "N2-sessions-abcdef0", "20261030-N2-sessions-xyz-trial"):
            text = self.go("check", study=name, expect=1)
            self.assertIn("study folder refused", text)
            self.assertFalse((self.tmp / "out" / name).exists())

    def test_non_empty_folder_refused_untouched(self):
        self.study.mkdir(parents=True)
        (self.study / "keep.txt").write_text("x")
        text = self.go("check", expect=1)
        self.assertIn("must be new", text)
        self.assertEqual(self.listing(), ["keep.txt"])

    def test_rerun_check_after_pass_refused(self):
        self.go("check")
        before = self.listing()
        text = self.go("check", expect=1)
        self.assertIn("must be new", text)
        self.assertEqual(self.listing(), before)

    def test_inside_repository_refused(self):
        for rel in ("results/m2", "results/n1", "scripts", "results"):
            text = self.go("check", study_dir=ROOT / rel / STUDY, expect=1)
            self.assertIn("study folder refused", text)
            self.assertFalse((ROOT / rel / STUDY).exists())

    def test_protected_moved_folder_refused(self):
        text = self.go("check", study_dir=Path(os.environ["LIVENESSLAB_RESULTS_DIR"]) / "n1" / STUDY, expect=1)
        self.assertIn("protected folder", text)
        text = self.go("check", study_dir=Path(os.environ["LIVENESSLAB_DATA_DIR"]) / STUDY, expect=1)
        self.assertIn("protected folder", text)

    def test_symlink_refused(self):
        real = self.tmp / "real"
        real.mkdir()
        (self.tmp / "out").mkdir()
        os.symlink(real, self.study)
        text = self.go("check", expect=1)
        self.assertIn("symbolic link", text)
        self.assertEqual(list(real.iterdir()), [])

    def test_inputs_inside_study_refused(self):
        self.study.mkdir(parents=True)
        cfg = self.study / "n2_sessions.json"
        shutil.copy(self.fx.config_path, cfg)
        text = self.go("check", config=cfg, expect=1)
        self.assertIn("inputs refused", text)
        self.assertEqual(self.listing(), ["n2_sessions.json"])

    def test_missing_inputs_refused(self):
        for over in ({"images": self.tmp / "absent"}, {"frozen_face": self.tmp / "absent.py"}, {"n1_study": self.tmp / "absent"}):
            text = self.go("check", expect=1, **over)
            self.assertIn("inputs refused", text)


# ============================================================================ check

class CheckTests(Base):
    def rewrite(self, name, rows):
        write_csv(self.fx.man / self.fx.files[name], rows)

    def test_cache_sha_mismatch(self):
        z = dict(np.load(self.fx.cache))
        z["X_train"] = z["X_train"].copy()
        z["X_train"][0, 0, 0, 0] ^= 1
        np.savez(self.fx.cache, **z)
        text = self.go("check", expect=1)
        self.assertIn("SHA-256", text)
        self.assertEqual(self.listing(), ["check.log", "status.json"])
        self.assertEqual(self.status()["result"], "FAIL")

    def test_manifest_count_mismatch(self):
        cfg = copy.deepcopy(self.fx.config)
        cfg["data"]["manifests"]["c1"]["rows"] += 1
        self.fx.write_config(cfg)
        text = self.go("check", expect=1)
        self.assertIn("(c1): rows", text)
        self.assertIn("differs from the configuration", text)

    def test_manifest_sha_mismatch(self):
        p = self.fx.man / self.fx.files["cnn_test"]
        p.write_text(p.read_text() + "\n")
        text = self.go("check", expect=1)
        self.assertIn("cnn_test): SHA-256", text)

    def test_target_count_mismatch(self):
        cfg = copy.deepcopy(self.fx.config)
        cfg["target"]["n_bona_fide"] += 1
        cfg["target"]["n_attack"] -= 1
        self.fx.write_config(cfg)
        text = self.go("check", expect=1)
        self.assertIn("target n_bona_fide", text)

    def test_target_manifest_sha_mismatch(self):
        cfg = copy.deepcopy(self.fx.config)
        cfg["target"]["manifest_sha256"] = "0" * 64
        self.fx.write_config(cfg)
        text = self.go("check", expect=1)
        self.assertIn("target manifest: SHA-256", text)
        self.assertNotIn("target_manifest.csv", self.listing())

    def test_row_moved_from_validation_to_target(self):
        rows = self.fx.rows["cnn_es"][1:]
        self.rewrite("cnn_es", rows)
        cfg = copy.deepcopy(self.fx.config)
        cfg["data"]["manifests"]["cnn_es"]["sha256"] = sha((self.fx.man / self.fx.files["cnn_es"]).read_bytes())
        cfg["data"]["manifests"]["cnn_es"]["rows"] -= 1
        cfg["data"]["manifests"]["cnn_es"]["n_" + ("bona_fide" if self.fx.rows["cnn_es"][0]["label"] == "0" else "attack")] -= 1
        self.fx.write_config(cfg)
        text = self.go("check", expect=1)
        self.assertIn("sampling order", text)

    def test_shared_key_with_different_fields(self):
        rows = copy.deepcopy(self.fx.rows["m1_dev"])
        rows[0]["sha256"] = "1" * 64
        self.rewrite("m1_dev", rows)
        cfg = copy.deepcopy(self.fx.config)
        cfg["data"]["manifests"]["m1_dev"]["sha256"] = sha((self.fx.man / self.fx.files["m1_dev"]).read_bytes())
        self.fx.write_config(cfg)
        text = self.go("check", expect=1)
        self.assertIn("same key as official_train but different fields", text)

    def test_original_missing_or_altered(self):
        r = self.fx.target[3]
        p = self.fx.images / N.LAYOUTS["images"](r)
        p.write_bytes(p.read_bytes() + b"x")
        text = self.go("check", expect=1)
        self.assertIn("size or SHA-256 differ", text)
        self.assertEqual(self.status()["result"], "FAIL")
        shutil.rmtree(self.tmp / "out")
        p.unlink()
        text = self.go("check", expect=1)
        self.assertIn("not found under the images folder", text)

    def test_raw_layout_accepted(self):
        raw = self.tmp / "raw"
        for r in self.fx.target:
            name = nm.name_of(r["key"])
            p = raw / ("ClientRaw" if r["label"] == "0" else "ImposterRaw") / r["subject"] / name.split("_", 1)[1]     # nome del mirror, senza il soggetto
            self.assertEqual(str(p.relative_to(raw)), N.LAYOUTS["raw"](r))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content_of(r["key"]))
        self.go("check", images=raw)
        doc = json.loads((self.study / "check.json").read_text())
        self.assertEqual(doc["checks"]["originals"]["by_layout"], {"raw": len(self.fx.target)})

    def test_symlinked_original_refused(self):
        r = self.fx.target[1]
        p = self.fx.images / N.LAYOUTS["images"](r)
        real = self.tmp / "elsewhere.jpg"
        real.write_bytes(p.read_bytes())
        p.unlink()
        os.symlink(real, p)
        text = self.go("check", expect=1)
        self.assertIn("symbolic link refused", text)

    def test_official_lists_sharing_a_file_refused(self):
        rows = copy.deepcopy(self.fx.rows["official_test"])
        sampled = {r["key"] for r in self.fx.rows["cnn_test"]}
        victim = next(r for r in rows if r["key"] not in sampled)      # una riga del test fuori dal campionamento: cnn_test resta coerente
        victim["sha256"] = self.fx.target[0]["sha256"]
        self.rewrite("official_test", rows)
        cfg = copy.deepcopy(self.fx.config)
        cfg["data"]["manifests"]["official_test"]["sha256"] = sha((self.fx.man / self.fx.files["official_test"]).read_bytes())
        self.fx.write_config(cfg)
        text = self.go("check", expect=1)
        self.assertIn("official train and test share values of sha256", text)

    def test_subject_in_both_fitting_and_validation_refused(self):
        moved = self.fx.rows["cnn_es"][0]
        tr = self.fx.rows["cnn_train"] + [moved]
        es = self.fx.rows["cnn_es"][1:]
        self.rewrite("cnn_train", tr)
        self.rewrite("cnn_es", es)
        cfg = copy.deepcopy(self.fx.config)
        for n, rows in (("cnn_train", tr), ("cnn_es", es)):
            cfg["data"]["manifests"][n].update(sha256=sha((self.fx.man / self.fx.files[n]).read_bytes()), **N1.counts_of(rows))
        self.fx.write_config(cfg)
        text = self.go("check", expect=1)
        self.assertIn("cnn_train and cnn_es share values of subject", text)

    def test_cache_member_dtype_refused(self):
        z = dict(np.load(self.fx.cache))
        z["y_train"] = z["y_train"].astype(np.float32)
        np.savez(self.fx.cache, **z)
        cfg = copy.deepcopy(self.fx.config)
        cfg["data"]["cache"]["sha256"] = sha(self.fx.cache.read_bytes())
        self.fx.write_config(cfg)
        text = self.go("check", expect=1)
        self.assertIn("y_train has dtype", text)

    def n1_rewrite_scores(self, seed, rows):
        txt = csv_text(rows, N1.SCORE_COLUMNS)
        (self.fx.n1 / f"seed{seed}" / "scores_test.csv").write_text(txt)
        sc = json.loads((self.fx.n1 / f"seed{seed}" / "score.json").read_text())
        sc["files"]["scores_test.csv"] = sha(txt.encode())
        (self.fx.n1 / f"seed{seed}" / "score.json").write_text(json.dumps(sc, indent=1) + "\n")
        cfg = copy.deepcopy(self.fx.config)
        cfg["n1_study"]["seeds"][str(seed)]["scores_test_sha256"] = sha(txt.encode())
        self.fx.write_config(cfg)

    def test_n1_scores_missing_row_or_wrong_c1_flag(self):
        rows = list(csv.DictReader(io.StringIO((self.fx.n1 / "seed42" / "scores_test.csv").read_text())))
        self.n1_rewrite_scores(42, rows[1:])
        text = self.go("check", expect=1)
        self.assertIn("do not hold exactly the cnn_test rows", text)
        rows[0]["in_c1_subset"] = "0" if rows[0]["in_c1_subset"] == "1" else "1"
        self.n1_rewrite_scores(42, rows)
        shutil.rmtree(self.tmp / "out", ignore_errors=True)
        text = self.go("check", expect=1)
        self.assertIn("rows inconsistent", text)

    def test_n1_folder_name_and_check_record(self):
        other = self.fx.n1.parent / "20261024-N1-LivenessNet-0000000"
        shutil.copytree(self.fx.n1, other)
        text = self.go("check", n1_study=other, expect=1)
        self.assertIn("is not the N1 study", text)
        shutil.rmtree(self.tmp / "out")
        p = self.fx.n1 / "check.json"
        doc = json.loads(p.read_text())
        doc["step"] = "train"
        p.write_text(json.dumps(doc, indent=1) + "\n")
        cfg = copy.deepcopy(self.fx.config)
        cfg["n1_study"]["check_sha256"] = sha(p.read_bytes())
        self.fx.write_config(cfg)
        text = self.go("check", expect=1)
        self.assertIn("belongs to another step", text)

    def test_frozen_face_other_name_refused(self):
        other = self.tmp / "face_other.py"
        shutil.copy(self.fx.frozen, other)
        text = self.go("check", frozen_face=other, expect=1)
        self.assertIn("is not face_0ac7143.py", text)

    def test_malformed_config_blocks_refused_cleanly(self):
        for key, value in (("preprocessing", 1), ("parity", []), ("data", {"m2_run": "x", "cache": 3, "manifests": {}}), ("target", "x")):
            cfg = copy.deepcopy(self.fx.config)
            cfg[key] = value
            self.fx.write_config(cfg)
            text = self.go("check", expect=1)
            self.assertIn("configuration n2_sessions.json refused", text)
            self.assertNotIn("Traceback", text)
            self.assertNotIn("unexpected", text)
        cfg = copy.deepcopy(self.fx.config)
        cfg["target"]["by_session"]["01"] = str(cfg["target"]["by_session"]["01"])
        self.fx.write_config(cfg)
        self.assertIn("not consistent", self.go("check", expect=1))

    def test_n1_check_sha_mismatch(self):
        p = self.fx.n1 / "check.json"
        p.write_text(p.read_text() + "\n")
        text = self.go("check", expect=1)
        self.assertIn("N1 check.json", text)

    def test_n1_weights_sha_mismatch(self):
        w = self.fx.n1 / "seed43" / "checkpoints" / "LivenessNet__nuaa.h5"
        w.write_bytes(b"other")
        text = self.go("check", expect=1)
        self.assertIn("seed43 weights", text)

    def test_n1_scores_tampered(self):
        p = self.fx.n1 / "seed42" / "scores_test.csv"
        rows = list(csv.DictReader(io.StringIO(p.read_text())))
        rows[0]["score_attack"] = repr(1 - float(rows[0]["score_attack"]))
        p.write_text(csv_text(rows, N1.SCORE_COLUMNS))
        text = self.go("check", expect=1)
        self.assertIn("seed42/scores_test.csv: missing or SHA-256 differs", text)
        # con gli SHA ridichiarati la regola di decisione o le righe del manifest devono fermare lo stesso
        cfg = copy.deepcopy(self.fx.config)
        rows[0]["prediction"] = "0" if float(rows[0]["score_attack"]) > 0.5 else "1"
        txt = csv_text(rows, N1.SCORE_COLUMNS)
        p.write_text(txt)
        cfg["n1_study"]["seeds"]["42"]["scores_test_sha256"] = sha(txt.encode())
        sc = json.loads((self.fx.n1 / "seed42" / "score.json").read_text())
        sc["files"]["scores_test.csv"] = sha(txt.encode())
        (self.fx.n1 / "seed42" / "score.json").write_text(json.dumps(sc, indent=1) + "\n")
        self.fx.write_config(cfg)
        shutil.rmtree(self.tmp / "out", ignore_errors=True)
        text = self.go("check", expect=1)
        self.assertIn("rows inconsistent", text)

    def test_n1_scores_of_another_subject(self):
        p = self.fx.n1 / "seed44" / "scores_test.csv"
        rows = list(csv.DictReader(io.StringIO(p.read_text())))
        j = next(i for i, r in enumerate(rows) if r["subject_id"] != rows[0]["subject_id"])
        rows[0]["subject_id"], rows[j]["subject_id"] = rows[j]["subject_id"], rows[0]["subject_id"]
        txt = csv_text(rows, N1.SCORE_COLUMNS)
        p.write_text(txt)
        cfg = copy.deepcopy(self.fx.config)
        cfg["n1_study"]["seeds"]["44"]["scores_test_sha256"] = sha(txt.encode())
        sc = json.loads((self.fx.n1 / "seed44" / "score.json").read_text())
        sc["files"]["scores_test.csv"] = sha(txt.encode())
        (self.fx.n1 / "seed44" / "score.json").write_text(json.dumps(sc, indent=1) + "\n")
        self.fx.write_config(cfg)
        text = self.go("check", expect=1)
        self.assertIn("rows inconsistent", text)

    def test_frozen_code_sha_mismatch(self):
        self.fx.frozen.write_text("# changed\n")
        text = self.go("check", expect=1)
        self.assertIn("frozen face module", text)
        self.fx.frozen.write_text("# fake frozen face module\n")
        self.fx.resources["deploy.prototxt"].write_bytes(b"changed")
        shutil.rmtree(self.tmp / "out", ignore_errors=True)
        text = self.go("check", expect=1)
        self.assertIn("detector resource deploy.prototxt", text)
        self.fx.resources["deploy.prototxt"].write_bytes(b"fake deploy.prototxt")
        ref = TMP_ROOT / "livedetection" / "src" / "architectures.py"
        ref.write_bytes(b"# changed\n")
        try:
            shutil.rmtree(self.tmp / "out", ignore_errors=True)
            text = self.go("check", expect=1)
            self.assertIn("reference code architectures.py", text)
        finally:
            ref.write_bytes(REF_CODE[("src", "architectures.py")])

    def test_parity_list_bound_by_sha(self):
        cfg = copy.deepcopy(self.fx.config)
        cfg["parity"]["keys_sha256"] = "0" * 64
        self.fx.write_config(cfg)
        text = self.go("check", expect=1)
        self.assertIn("parity list derived from cnn_test", text)
        self.assertNotIn("parity_keys.csv", self.listing())
        cfg["parity"].update(keys_sha256=self.fx.config["parity"]["keys_sha256"], n_keys=self.fx.config["parity"]["n_keys"] + 1)
        self.fx.write_config(cfg)
        shutil.rmtree(self.tmp / "out")
        text = self.go("check", expect=1)
        self.assertIn("parity list derived from cnn_test", text)
        self.fx.write_config()
        shutil.rmtree(self.tmp / "out")
        self.go("check")
        rows = list(csv.DictReader(io.StringIO((self.study / "parity_keys.csv").read_text())))
        self.assertEqual([r["sample_id"] for r in rows], self.fx.parity_keys())
        self.assertEqual(N.parity_keys(self.fx.rows["cnn_test"]), self.fx.parity_keys())

    def test_target_shares_subject_outside_groups(self):
        # un soggetto di fitting spostato in validazione nei manifest (cnn_train e cnn_es ridichiarati): i gruppi non sono più quelli dell'ordine di campionamento
        s = self.fx.fit_subjects[0]
        tr = [r for r in self.fx.rows["cnn_train"] if r["subject"] != s]
        es = self.fx.rows["cnn_es"] + [r for r in self.fx.rows["cnn_train"] if r["subject"] == s]
        self.rewrite("cnn_train", tr)
        self.rewrite("cnn_es", es)
        cfg = copy.deepcopy(self.fx.config)
        for n, rows in (("cnn_train", tr), ("cnn_es", es)):
            cfg["data"]["manifests"][n].update(sha256=sha((self.fx.man / self.fx.files[n]).read_bytes()), **N1.counts_of(rows))
        self.fx.write_config(cfg)
        self.go("check")       # la derivazione è ancora coerente: i gruppi seguono i manifest, non una lista fissa
        doc = json.loads((self.study / "check.json").read_text())
        self.assertEqual(doc["checks"]["target"]["n_validation_subjects"], 3)


# ============================================================================ cache-check

class CacheCheckTests(Base):
    def test_pass(self):
        text = self.pipeline("cache-check")
        doc = json.loads((self.study / "cache_check.json").read_text())
        self.assertEqual((doc["cache"]["train"]["equal"], doc["cache"]["test"]["equal"]), (2 * PER_CLASS, 2 * PER_CLASS))
        self.assertEqual(doc["cache"]["train"]["different"] + doc["cache"]["test"]["different"], 0)
        self.assertEqual(doc["target_frames"]["n"], len(self.fx.target))
        self.assertEqual(sum(doc["target_frames"]["detectors"].values()), len(self.fx.target))
        self.assertEqual(doc["tensorflow"], "not imported")
        rows = list(csv.DictReader(io.StringIO((self.study / "frames_target.csv").read_text())))
        self.assertEqual([r["sample_id"] for r in rows], [r["key"] for r in self.fx.target])
        self.assertIn("reproduced bit for bit", text)

    def test_different_frame_fails_and_writes_nothing(self):
        self.go("check")
        bad_key = self.fx.order_train[5]

        def pre_factory(frozen, cfg):
            base = fake_preprocessor(frozen, cfg)

            def pre(path):
                X, det = base(path)
                if Path(path).name == nm.name_of(bad_key):
                    X = X.copy()
                    X[0, 0, 0] ^= 1
                return X, det
            return pre
        with mock.patch.object(N, "PREPROCESSOR", pre_factory):
            text = self.go("cache-check", expect=1)
        self.assertIn("1 frames differ from the cache", text)
        self.assertIn("DIFF X_train[5]", text)
        self.assertNotIn("cache_check.json", self.listing())
        self.assertNotIn("frames_target.csv", self.listing())
        self.assertEqual(self.status()["result"], "FAIL")

    def test_duplicated_target_frame_refused(self):
        self.go("check")
        a, b = self.fx.target[0]["key"], self.fx.target[1]["key"]

        def pre_factory(frozen, cfg):
            base = fake_preprocessor(frozen, cfg)

            def pre(path):
                X, det = base(path)
                if Path(path).name in (nm.name_of(a), nm.name_of(b)):
                    return np.full((8, 8, 3), 7, np.uint8), det
                return X, det
            return pre
        with mock.patch.object(N, "PREPROCESSOR", pre_factory):
            text = self.go("cache-check", expect=1)
        self.assertIn("duplicated derived frames", text)
        self.assertIn("target_duplicates", text)

    def test_target_frame_equal_to_cache_refused(self):
        self.go("check")
        k, c = self.fx.target[0]["key"], self.fx.order_train[0]

        def pre_factory(frozen, cfg):
            base = fake_preprocessor(frozen, cfg)

            def pre(path):
                if Path(path).name == nm.name_of(k):
                    return base(self.fx.images / N.LAYOUTS["images"](self.fx.rows["official_train"][0]) if False else
                                next(p for p in self.fx.images.rglob(nm.name_of(c))))
                return base(path)
            return pre
        with mock.patch.object(N, "PREPROCESSOR", pre_factory):
            text = self.go("cache-check", expect=1)
        self.assertIn("target_shared_with_cache", text)

    def test_step_only_in_trial(self):
        with mock.patch.object(N, "CONFIG_SHA256", sha(self.fx.config_path.read_bytes())):
            self.go("check", study=STUDY_REAL)
            text = self.go("cache-check", "--step", "2", study=STUDY_REAL, expect=1)
            self.assertIn("only in -trial runs", text)
            self.go("cache-check", study=STUDY_REAL)
        self.go("check")
        self.go("cache-check", "--step", "3")
        doc = json.loads((self.study / "cache_check.json").read_text())
        self.assertEqual(doc["cache"]["train"]["compared"], 8)
        self.assertEqual(doc["step_size"], 3)

    def test_never_recomputed_in_place(self):
        self.pipeline("cache-check")
        before = sha((self.study / "cache_check.json").read_bytes())
        text = self.go("cache-check", expect=1)
        self.assertIn("never recomputed in place", text)
        self.assertEqual(sha((self.study / "cache_check.json").read_bytes()), before)

    def test_code_changed_since_check(self):
        self.go("check")
        with mock.patch.object(N, "code_hashes", lambda a: {"scripts/n2_sessions.py": "0" * 64}):
            text = self.go("cache-check", expect=1)
        self.assertIn("code changed since check", text)

    def test_commit_changed_for_non_trial(self):
        with mock.patch.object(N, "CONFIG_SHA256", sha(self.fx.config_path.read_bytes())):
            self.go("check", study=STUDY_REAL)
            with mock.patch.object(N1, "REPO_STATE", lambda d: {"commit": "1234567" + "0" * 33, "dirty": False, "dirty_paths": [], "dirty_excludes": None}):
                text = self.go("cache-check", study=STUDY_REAL, expect=1)
        self.assertIn("commit differs from check", text)

    def test_inputs_changed_after_check(self):
        self.go("check")
        p = self.fx.n1 / "seed42" / "scores_test.csv"
        rows = list(csv.DictReader(io.StringIO(p.read_text())))
        p.write_text(csv_text(rows, N1.SCORE_COLUMNS) + "\n")
        text = self.go("cache-check", expect=1)
        self.assertIn("SHA-256 differs from the configuration", text)

    def test_requires_check(self):
        text = self.go("cache-check", expect=1)
        self.assertIn("run check first", text)

    def test_target_manifest_changed_after_check(self):
        self.go("check")
        p = self.study / "target_manifest.csv"
        rows = list(csv.DictReader(io.StringIO(p.read_text())))
        p.write_text(csv_text(rows[1:], nm.COLUMNS))
        text = self.go("cache-check", expect=1)
        self.assertIn("target_manifest.csv changed after check", text)
        shutil.rmtree(self.tmp / "out")
        self.go("check")
        (self.study / "parity_keys.csv").write_text("sample_id,subject_id,label\n")
        text = self.go("cache-check", expect=1)
        self.assertIn("parity_keys.csv changed after check", text)

    def test_check_record_tampered(self):
        self.go("check")
        p = self.study / "check.json"
        doc = json.loads(p.read_text())
        doc["checks"]["manifests"]["c1"]["rows"] = 999
        p.write_text(json.dumps(doc, indent=1) + "\n")
        text = self.go("cache-check", expect=1)
        self.assertIn("manifests differ from those recorded at check", text)
        self.assertNotIn("cache_check.json", self.listing())


# ============================================================================ parity

class ParityTests(Base):
    def test_pass(self):
        text = self.pipeline("parity")
        doc = json.loads((self.study / "parity.json").read_text())
        self.assertEqual(doc["frames_equal"], self.fx.config["parity"]["n_keys"])
        for s in ("42", "43", "44"):
            self.assertEqual(doc["seeds"][s]["max_abs_diff"], 0.0)
            self.assertEqual(doc["seeds"][s]["within_tolerance"], doc["n_keys"])
        rows = list(csv.DictReader(io.StringIO((self.study / "parity.csv").read_text())))
        self.assertEqual(len(rows), 3 * doc["n_keys"])
        self.assertTrue(all(r["frame_equal"] == "1" and float(r["abs_diff"]) == 0 for r in rows))
        self.assertIn("parity PASS", text)

    def test_requires_cache_check(self):
        self.go("check")
        text = self.go("parity", expect=1)
        self.assertIn("run the previous step first", text)

    def test_frame_differs_from_cache(self):
        self.pipeline("cache-check")
        k = self.fx.parity_keys()[2]

        def pre_factory(frozen, cfg):
            base = fake_preprocessor(frozen, cfg)

            def pre(path):
                X, det = base(path)
                if Path(path).name == nm.name_of(k):
                    X = X.copy()
                    X[1, 1, 1] ^= 2
                return X, det
            return pre
        with mock.patch.object(N, "PREPROCESSOR", pre_factory):
            text = self.go("parity", expect=1)
        self.assertIn("1 of", text)
        self.assertIn("frames differ from the cache rows", text)
        self.assertNotIn("parity.json", self.listing())

    def test_tolerance(self):
        self.pipeline("cache-check")

        def shifted(eps):
            def make(path, wsha):
                def score(X):
                    p = fake_prob(X, wsha)
                    p[:, 1] = np.clip(p[:, 1] + eps, 0, 1)
                    p[:, 0] = 1 - p[:, 1]
                    return p
                return score
            return make
        with mock.patch.object(N1, "MAKE_SCORER", shifted(2e-6)):
            text = self.go("parity", expect=1)
        self.assertIn("outside the tolerance", text)
        self.assertNotIn("parity.json", self.listing())
        with mock.patch.object(N1, "MAKE_SCORER", shifted(5e-7)):
            self.go("parity")
        doc = json.loads((self.study / "parity.json").read_text())
        self.assertTrue(all(4e-7 < v["max_abs_diff"] <= 1e-6 for v in doc["seeds"].values()))

    def test_never_recomputed(self):
        self.pipeline("parity")
        text = self.go("parity", expect=1)
        self.assertIn("never recomputed in place", text)

    def test_frames_target_redeclared_in_cache_check_refused_at_score(self):
        self.pipeline("parity")
        p = self.study / "frames_target.csv"
        rows = list(csv.DictReader(io.StringIO(p.read_text())))
        txt = csv_text(rows[1:], N.FRAME_COLUMNS)
        p.write_text(txt)
        cc = json.loads((self.study / "cache_check.json").read_text())
        cc["files"]["frames_target.csv"] = sha(txt.encode())
        (self.study / "cache_check.json").write_text(json.dumps(cc, indent=1) + "\n")
        text = self.go("score", "--train-seed", "42", "--i-am-authorized", expect=1)
        self.assertIn("does not belong to this cache_check.json", text)
        # senza il legame di parity, le chiavi mancanti sarebbero comunque rifiutate
        par = json.loads((self.study / "parity.json").read_text())
        par["cache_check_sha256"] = sha((self.study / "cache_check.json").read_bytes())
        (self.study / "parity.json").write_text(json.dumps(par, indent=1) + "\n")
        text = self.go("score", "--train-seed", "42", "--i-am-authorized", expect=1)
        self.assertIn("does not hold exactly the target keys", text)

    def test_detector_differs_from_cache_check(self):
        self.pipeline("parity")
        k = self.fx.target[6]["key"]

        def pre_factory(frozen, cfg):
            base = fake_preprocessor(frozen, cfg)

            def pre(path):
                X, det = base(path)
                return (X, "Haar" if det == "RetinaFace" else "RetinaFace") if Path(path).name == nm.name_of(k) else (X, det)
            return pre
        with mock.patch.object(N, "PREPROCESSOR", pre_factory):
            text = self.go("score", "--train-seed", "42", "--i-am-authorized", expect=1)
        self.assertIn("differ from cache-check", text)

    def test_frames_target_changed_after_cache_check(self):
        self.pipeline("cache-check")
        p = self.study / "frames_target.csv"
        p.write_text(p.read_text() + "\n")
        text = self.go("parity", expect=1)
        self.assertIn("frames_target.csv changed after cache-check", text)


# ============================================================================ score

class ScoreTests(Base):
    def test_pass(self):
        self.pipeline("parity")
        text = self.go("score", "--train-seed", "42", "--i-am-authorized")
        sd = self.study / "seed42"
        self.assertEqual(sorted(p.name for p in sd.iterdir()), ["metrics.json", "per_subject.csv", "score.json", "score.log", "scores_sessions.csv", "status.json"])
        rows = list(csv.DictReader(io.StringIO((sd / "scores_sessions.csv").read_text())))
        self.assertEqual([r["sample_id"] for r in rows], [r["key"] for r in self.fx.target])
        wsha = self.fx.seeds["42"]["weights_sha256"]
        X = np.stack([frame_of(content_of(r["key"]))[0] for r in self.fx.target])
        exp = fake_prob(X, wsha)[:, 1]
        for r, v in zip(rows, exp):
            self.assertEqual(float(r["score_attack"]), float(v))
            self.assertEqual(r["prediction"], "1" if v > 0.5 else "0")
            self.assertEqual(r["checkpoint_sha256"], wsha)
            self.assertIn(r["group"], ("fitting", "validation"))
            self.assertEqual(r["group"], "validation" if r["subject_id"] in self.fx.val_subjects else "fitting")
        m1 = {r["key"] for r in self.fx.rows["m1_dev"]}
        self.assertEqual(sum(1 for r in rows if r["in_m1_dev"] == "1"), sum(1 for r in self.fx.target if r["key"] in m1))
        met = json.loads((sd / "metrics.json").read_text())["metrics"]
        self.assertEqual(set(met), set(N.TARGET_SETS))
        self.assertEqual(met["sessions_01_02_491"]["n"], len(self.fx.target))
        self.assertEqual(met["sessions_01_02_not_in_m1_dev"]["n"], len(self.fx.target) - met["sessions_01_02_491"]["n"] + sum(1 for r in rows if r["in_m1_dev"] == "0"))
        self.assertIn("score PASS", text)

    def test_needs_authorization_and_valid_seed(self):
        self.pipeline("parity")
        text = self.go("score", "--train-seed", "42", expect=1)
        self.assertIn("--i-am-authorized", text)
        text = self.go("score", "--train-seed", "45", "--i-am-authorized", expect=1)
        self.assertIn("not one of the N1 seeds", text)
        text = self.go("score", "--i-am-authorized", expect=1)
        self.assertIn("needs --train-seed", text)
        self.assertFalse((self.study / "seed42").exists())

    def test_requires_parity(self):
        self.pipeline("cache-check")
        text = self.go("score", "--train-seed", "42", "--i-am-authorized", expect=1)
        self.assertIn("parity.json", text)
        self.assertFalse((self.study / "seed42").exists())

    def test_seed_folder_never_reused(self):
        self.pipeline("parity")
        self.go("score", "--train-seed", "42", "--i-am-authorized")
        before = self.listing()
        text = self.go("score", "--train-seed", "42", "--i-am-authorized", expect=1)
        self.assertIn("never reused", text)
        self.assertEqual(self.listing(), before)

    def test_frame_differs_from_cache_check(self):
        self.pipeline("parity")
        k = self.fx.target[4]["key"]

        def pre_factory(frozen, cfg):
            base = fake_preprocessor(frozen, cfg)

            def pre(path):
                X, det = base(path)
                if Path(path).name == nm.name_of(k):
                    X = X.copy()
                    X[2, 2, 2] ^= 4
                return X, det
            return pre
        with mock.patch.object(N, "PREPROCESSOR", pre_factory):
            text = self.go("score", "--train-seed", "43", "--i-am-authorized", expect=1)
        self.assertIn("differ from cache-check", text)
        self.assertEqual(self.status(self.study / "seed43")["result"], "FAIL")
        self.assertNotIn("seed43/scores_sessions.csv", self.listing())

    def test_tie_at_threshold_is_bona_fide(self):
        self.pipeline("parity")
        with mock.patch.object(N1, "MAKE_SCORER", lambda p, s: (lambda X: np.full((len(X), 2), 0.5))):
            self.go("score", "--train-seed", "42", "--i-am-authorized")
        rows = list(csv.DictReader(io.StringIO((self.study / "seed42" / "scores_sessions.csv").read_text())))
        self.assertTrue(all(r["prediction"] == "0" for r in rows))
        met = json.loads((self.study / "seed42" / "metrics.json").read_text())["metrics"]["sessions_01_02_491"]
        self.assertEqual(met["ties_at_threshold"], len(rows))
        self.assertEqual(met["rejected_bona_fide"], 0)
        self.assertEqual(met["accepted_attacks"], met["n_attack"])

    def test_nan_scores_refused(self):
        self.pipeline("parity")
        with mock.patch.object(N1, "MAKE_SCORER", lambda p, s: (lambda X: np.full((len(X), 2), np.nan))):
            text = self.go("score", "--train-seed", "44", "--i-am-authorized", expect=1)
        self.assertIn("non-finite", text)

    def test_metrics_na_and_tie(self):
        rows = [{"label": "0", "score_attack": "0.5", "group": "fitting", "in_m1_dev": "0"},
                {"label": "0", "score_attack": "0.7", "group": "fitting", "in_m1_dev": "1"},
                {"label": "1", "score_attack": "0.9", "group": "fitting", "in_m1_dev": "0"}]
        met = N.metrics_from_rows(rows)
        self.assertEqual(met["sessions_01_02_491"]["rejected_bona_fide"], 1)      # il pareggio a 0,5 è bona fide
        self.assertEqual(met["sessions_01_02_491"]["ties_at_threshold"], 1)
        self.assertIsNone(met["sessions_01_02_validation_subjects"]["apcer"])
        self.assertIsNone(met["sessions_01_02_validation_subjects"]["auroc"])
        self.assertEqual(met["sessions_01_02_not_in_m1_dev"]["n"], 2)


# ============================================================================ aggregate

class AggregateTests(Base):
    def test_full_pipeline(self):
        text = self.pipeline()
        for n in N.AGG_OUTPUTS:
            self.assertTrue((self.study / n).is_file(), n)
        doc = json.loads((self.study / "aggregate.json").read_text())
        self.assertEqual(doc["train_seeds"], [42, 43, 44])
        self.assertEqual(set(doc["summary"]), set(N.SETS))
        for s in ("42", "43", "44"):
            m = doc["seeds"][s]["metrics"]
            self.assertEqual(m["session_03_3000"]["n"], self.fx.n_test)
            self.assertEqual(m["sessions_01_02_491"]["n"], len(self.fx.target))
            self.assertEqual(m["sessions_01_02_fitting_subjects"]["n"] + m["sessions_01_02_validation_subjects"]["n"], len(self.fx.target))
            d = doc["seeds"][s]["distributions"]["sessions_01_02_491"]
            self.assertEqual(sum(d["bona_fide"]) + sum(d["attack"]), len(self.fx.target))
            self.assertEqual(len(d["bin_edges"]), N.BINS + 1)
            ss = doc["seeds"][s]["strata_summary"]
            self.assertEqual(set(ss), {"fitting/bona_fide", "fitting/attack", "validation/bona_fide", "validation/attack"})
            self.assertEqual(ss["fitting/attack"]["strata"], len(self.fx.fit_subjects))
            for v in ss.values():
                self.assertEqual(v["strata"], v["lower_in_01_02"] + v["equal"] + v["higher_in_01_02"] + v["not_comparable"])
        # mediana e intervallo
        vals = [doc["seeds"][s]["metrics"]["sessions_01_02_491"]["acer"] for s in ("42", "43", "44")]
        self.assertEqual(doc["summary"]["sessions_01_02_491"]["acer"], {"values": vals, "median": statistics.median(vals), "min": min(vals), "max": max(vals)})
        self.assertTrue(len(set(vals)) >= 2, "the fixture should give different values per seed")
        # le metriche della sessione 03 coincidono con quelle ricalcolate dai punteggi N1
        rows = list(csv.DictReader(io.StringIO((self.fx.n1 / "seed42" / "scores_test.csv").read_text())))
        m = N1.set_metrics([int(r["label"]) for r in rows], [float(r["score_attack"]) for r in rows])
        self.assertEqual(doc["seeds"]["42"]["metrics"]["session_03_3000"], m)
        # strati riservati: 2 insiemi x 3 seed x soggetti x 2 classi
        st = list(csv.DictReader(io.StringIO((self.study / "strata_all_seeds.csv").read_text())))
        self.assertEqual(len(st), 2 * 3 * len(SUBJECTS) * 2)
        self.assertTrue(all(r["error_rate"] == "NA" for r in st if r["n"] == "0"))
        self.assertEqual({r["group"] for r in st}, {"fitting", "validation"})
        md = (self.study / "aggregate.md").read_text()
        self.assertIn(f"| sessions 01-02: the {len(self.fx.target)} target frames | 42 |", md)
        self.assertIn("median (range)", md)
        self.assertEqual((self.study / "figure_strata.png").read_bytes(), PNG + b"s")
        # pubblico senza identificativi: nessun soggetto, nessuna chiave, nessun SHA di immagini
        for n in ("aggregate.json", "aggregate.md"):
            t = (self.study / n).read_text()
            for s in SUBJECTS:
                self.assertNotRegex(t, rf"(?<![\w.]){s}(?![\w.])", n)
            self.assertNotRegex(t, r"\b(real|attack)/\d{4}_", n)
            for r in self.fx.target:
                self.assertNotIn(r["sha256"], t)
        self.assertIn("aggregate PASS", text)

    def test_no_seed_selection_and_missing_seed(self):
        self.pipeline("score")
        text = self.go("aggregate", "--train-seed", "42", expect=1)
        self.assertIn("no seed selection", text)
        shutil.rmtree(self.study / "seed44")
        text = self.go("aggregate", expect=1)
        self.assertIn("seed44/score.json", text)
        self.assertNotIn("aggregate.json", self.listing())

    def test_metrics_json_tampered(self):
        self.pipeline("score")
        p = self.study / "seed43" / "metrics.json"
        doc = json.loads(p.read_text())
        doc["metrics"]["sessions_01_02_491"]["acer"] = 0.123
        p.write_text(json.dumps(doc, indent=1) + "\n")
        text = self.go("aggregate", expect=1)
        self.assertIn("seed43/metrics.json changed after scoring", text)
        # con lo SHA ridichiarato in score.json deve fermarsi il ricalcolo delle metriche
        sc = json.loads((self.study / "seed43" / "score.json").read_text())
        sc["files"]["metrics.json"] = sha(p.read_bytes())
        (self.study / "seed43" / "score.json").write_text(json.dumps(sc, indent=1) + "\n")
        text = self.go("aggregate", expect=1)
        self.assertIn("seed43: metrics recomputed from the scores differ from metrics.json", text)
        self.assertNotIn("aggregate.json", self.listing())

    def test_scores_tampered_with_redeclared_sha(self):
        self.pipeline("score")
        sd = self.study / "seed42"
        rows = list(csv.DictReader(io.StringIO((sd / "scores_sessions.csv").read_text())))
        rows[0]["score_attack"] = repr(1 - float(rows[0]["score_attack"]))
        txt = csv_text(rows, N.SCORE_COLUMNS)
        (sd / "scores_sessions.csv").write_text(txt)
        sc = json.loads((sd / "score.json").read_text())
        sc["files"]["scores_sessions.csv"] = sha(txt.encode())
        (sd / "score.json").write_text(json.dumps(sc, indent=1) + "\n")
        text = self.go("aggregate", expect=1)
        self.assertTrue("inconsistent" in text or "differ from metrics.json" in text, text)

    def test_score_row_with_wrong_group_or_label(self):
        self.pipeline("score")
        sd = self.study / "seed42"
        rows = list(csv.DictReader(io.StringIO((sd / "scores_sessions.csv").read_text())))
        rows[0]["group"] = "validation" if rows[0]["group"] == "fitting" else "fitting"
        txt = csv_text(rows, N.SCORE_COLUMNS)
        (sd / "scores_sessions.csv").write_text(txt)
        sc = json.loads((sd / "score.json").read_text())
        sc["files"]["scores_sessions.csv"] = sha(txt.encode())
        (sd / "score.json").write_text(json.dumps(sc, indent=1) + "\n")
        text = self.go("aggregate", expect=1)
        self.assertIn("rows inconsistent", text)

    def redeclare_scores(self, seed, rows):
        sd = self.study / f"seed{seed}"
        txt = csv_text(rows, N.SCORE_COLUMNS)
        (sd / "scores_sessions.csv").write_text(txt)
        sc = json.loads((sd / "score.json").read_text())
        sc["files"]["scores_sessions.csv"] = sha(txt.encode())
        (sd / "score.json").write_text(json.dumps(sc, indent=1) + "\n")

    def test_rows_reordered_or_flag_changed(self):
        self.pipeline("score")
        rows = list(csv.DictReader(io.StringIO((self.study / "seed42" / "scores_sessions.csv").read_text())))
        self.redeclare_scores(42, list(reversed(rows)))
        self.assertIn("in key order", self.go("aggregate", expect=1))
        rows[0]["in_m1_dev"] = "0" if rows[0]["in_m1_dev"] == "1" else "1"
        self.redeclare_scores(42, rows)
        self.assertIn("rows inconsistent", self.go("aggregate", expect=1))

    def test_score_record_of_another_seed_or_detectors(self):
        self.pipeline("score")
        sd = self.study / "seed43"
        sc = json.loads((sd / "score.json").read_text())
        sc["train_seed"] = 42
        (sd / "score.json").write_text(json.dumps(sc, indent=1) + "\n")
        self.assertIn("seed43: score.json records another seed", self.go("aggregate", expect=1))
        sc["train_seed"] = 43
        sc["detectors"] = {"RetinaFace": sum(sc["detectors"].values())}
        (sd / "score.json").write_text(json.dumps(sc, indent=1) + "\n")
        self.assertIn("detector counts", self.go("aggregate", expect=1))

    def test_per_subject_redeclared_refused(self):
        self.pipeline("score")
        sd = self.study / "seed44"
        rows = list(csv.DictReader(io.StringIO((sd / "per_subject.csv").read_text())))
        rows[0]["rejected_bona_fide"] = str(int(rows[0]["rejected_bona_fide"]) + 1)
        txt = csv_text(rows, N.SUBJECT_COLUMNS)
        (sd / "per_subject.csv").write_text(txt)
        sc = json.loads((sd / "score.json").read_text())
        sc["files"]["per_subject.csv"] = sha(txt.encode())
        (sd / "score.json").write_text(json.dumps(sc, indent=1) + "\n")
        self.assertIn("per_subject.csv differs from the table recomputed", self.go("aggregate", expect=1))

    def test_public_scan_covers_markdown(self):
        self.pipeline("score")
        with mock.patch.object(N1, "public_problems", lambda text, inp: ["image keys"] if "## Single table" in text else []):
            text = self.go("aggregate", expect=1)
        self.assertIn("aggregate.md would hold image keys", text)
        self.assertNotIn("aggregate.json", self.listing())

    def test_reference_restricted_to_target_subjects(self):
        self.pipeline()
        doc = json.loads((self.study / "aggregate.json").read_text())
        n_target_subjects_test = sum(1 for r in self.fx.rows["cnn_test"] if r["subject"] in SUBJECTS)
        self.assertLess(n_target_subjects_test, self.fx.n_test)
        for s in ("42", "43", "44"):
            m = doc["seeds"][s]["metrics"]
            self.assertEqual(m["session_03_3000"]["n"], self.fx.n_test)
            self.assertEqual(m["session_03_9_subjects"]["n"], n_target_subjects_test)
            self.assertEqual(m["session_03_fitting_subjects"]["n"] + m["session_03_validation_subjects"]["n"], n_target_subjects_test)
            d = doc["seeds"][s]["distributions"]["session_03_9_subjects"]
            self.assertEqual(sum(d["bona_fide"]) + sum(d["attack"]), n_target_subjects_test)
        st = list(csv.DictReader(io.StringIO((self.study / "strata_all_seeds.csv").read_text())))
        self.assertEqual({r["subject_id"] for r in st}, set(SUBJECTS))
        ref_rows = [r for r in st if r["set"] == "session_03_9_subjects" and r["train_seed"] == "42"]
        n1 = list(csv.DictReader(io.StringIO((self.fx.n1 / "seed42" / "scores_test.csv").read_text())))
        for r in ref_rows:
            sel = [x for x in n1 if x["subject_id"] == r["subject_id"] and x["label"] == ("0" if r["class"] == "bona_fide" else "1")]
            err = sum(1 for x in sel if x["prediction"] == ("1" if r["class"] == "bona_fide" else "0"))
            self.assertEqual((int(r["n"]), int(r["errors"])), (len(sel), err), r)
        self.assertEqual(doc["sets"]["sessions_01_02_491"], f"sessions 01-02: the {len(self.fx.target)} target frames")
        self.assertEqual(doc["sets"]["session_03_9_subjects"], f"session 03: test frames of the {len(SUBJECTS)} target subjects (N1 scores)")
        self.assertEqual(set(doc["figures"]["public"]), set(N.PUBLIC_FIGURES))
        self.assertEqual(set(doc["figures"]["restricted"]), set(N.RESTRICTED_FIGURES))
        self.assertEqual((self.study / "figure_groups.png").read_bytes(), PNG + b"g")

    def test_tampered_after_pass_leaves_no_pass(self):
        self.pipeline()
        (self.study / "seed44" / "per_subject.csv").write_text("x\n")
        text = self.go("aggregate", expect=1)
        self.assertIn("changed after scoring", text)
        for n in N.AGG_OUTPUTS:
            self.assertFalse((self.study / n).exists(), n)
        self.assertEqual(self.status()["result"], "FAIL")

    def test_public_scan_hit_writes_nothing(self):
        self.pipeline("score")
        with mock.patch.object(N1, "public_problems", lambda text, inp: ["subject identifiers"]):
            text = self.go("aggregate", expect=1)
        self.assertIn("would hold subject identifiers: not written", text)
        for n in N.AGG_OUTPUTS:
            self.assertFalse((self.study / n).exists(), n)

    def test_plotter_output_validated(self):
        self.pipeline("score")
        with mock.patch.object(N, "PLOTTER", lambda doc, st: {"figure_distributions.png": b"not png", "figure_groups.png": PNG, "figure_strata.png": PNG}):
            text = self.go("aggregate", expect=1)
        self.assertIn("did not return the three PNG figures", text)
        with mock.patch.object(N, "PLOTTER", lambda doc, st: {"figure_distributions.png": PNG, "figure_strata.png": PNG}):
            text = self.go("aggregate", expect=1)
        self.assertIn("did not return the three PNG figures", text)
        for n in N.AGG_OUTPUTS:
            self.assertFalse((self.study / n).exists(), n)

    def test_aggregate_rerun_replaces_outputs(self):
        self.pipeline()
        (self.study / "aggregate.md").write_text("stale\n")
        text = self.go("aggregate")
        self.assertIn("removed previous aggregate.md", text)
        self.assertNotEqual((self.study / "aggregate.md").read_text(), "stale\n")

    def test_same_weights_refused(self):
        # studio N1 finto in cui i seed 42 e 43 hanno gli stessi pesi (record, punteggi e configurazione coerenti dall'inizio):
        # check, cache-check, parity e score passano, aggregate deve fermarsi
        w42, w43 = (self.fx.n1 / f"seed{s}" / "checkpoints" / "LivenessNet__nuaa.h5" for s in (42, 43))
        shutil.copy(w42, w43)
        wsha = sha(w43.read_bytes())
        tr = json.loads((self.fx.n1 / "seed43" / "train.json").read_text())
        tr["weights_sha256"] = wsha
        (self.fx.n1 / "seed43" / "train.json").write_text(json.dumps(tr, indent=1) + "\n")
        rows = list(csv.DictReader(io.StringIO((self.fx.n1 / "seed42" / "scores_test.csv").read_text())))
        for r in rows:
            r["train_seed"] = "43"
        txt = csv_text(rows, N1.SCORE_COLUMNS)
        (self.fx.n1 / "seed43" / "scores_test.csv").write_text(txt)
        sc = json.loads((self.fx.n1 / "seed43" / "score.json").read_text())
        sc["weights_sha256"], sc["files"]["scores_test.csv"] = wsha, sha(txt.encode())
        (self.fx.n1 / "seed43" / "score.json").write_text(json.dumps(sc, indent=1) + "\n")
        cfg = copy.deepcopy(self.fx.config)
        cfg["n1_study"]["seeds"]["43"].update(weights_sha256=wsha, scores_test_sha256=sha(txt.encode()))
        self.fx.write_config(cfg)
        self.pipeline("score")
        text = self.go("aggregate", expect=1)
        self.assertIn("two seeds have the same weights", text)
        self.assertNotIn("aggregate.json", self.listing())

    def test_histogram_and_strata_summary(self):
        rows = [{"label": "0", "score_attack": "0.0"}, {"label": "0", "score_attack": "1.0"}, {"label": "1", "score_attack": "0.55"},
                {"label": "1", "score_attack": "0.5"}]
        h = N.histogram(rows)
        self.assertEqual(sum(h["bona_fide"]), 2)
        self.assertEqual(h["bona_fide"][0], 1)
        self.assertEqual(h["bona_fide"][-1], 1)        # 1,0 nell'ultimo intervallo, chiuso a destra
        self.assertEqual(h["attack"][5], 2)
        self.assertEqual(h["bin_edges"][0], 0.0)
        a, b = SUBJECTS[0], SUBJECTS[1]
        grp = {a: "fitting", b: "validation"}
        t = [{"run_id": "r", "train_seed": "42", "set": "a", "subject_id": a, "group": "fitting", "class": "attack", "n": "2", "errors": "1", "error_rate": "0.5"},
             {"run_id": "r", "train_seed": "42", "set": "a", "subject_id": b, "group": "validation", "class": "attack", "n": "0", "errors": "0", "error_rate": "NA"},
             {"run_id": "r", "train_seed": "42", "set": "a", "subject_id": a, "group": "fitting", "class": "bona_fide", "n": "2", "errors": "2", "error_rate": "1.0"}]
        r = [{"subject_id": a, "class": "attack", "error_rate": "0.6"}, {"subject_id": b, "class": "attack", "error_rate": "0.1"},
             {"subject_id": a, "class": "bona_fide", "error_rate": "1.0"}]
        s = N.strata_summary(t, r)
        self.assertEqual(s["fitting/attack"], {"strata": 1, "lower_in_01_02": 1, "equal": 0, "higher_in_01_02": 0, "not_comparable": 0})
        self.assertEqual(s["validation/attack"], {"strata": 1, "lower_in_01_02": 0, "equal": 0, "higher_in_01_02": 0, "not_comparable": 1})
        self.assertEqual(s["fitting/bona_fide"]["equal"], 1)
        st = N.strata_rows("r", 42, "a", [{"subject_id": a, "label": "1", "prediction": "0"}, {"subject_id": a, "label": "1", "prediction": "1"}], grp)
        self.assertEqual([(x["class"], x["n"], x["errors"], x["error_rate"]) for x in st],
                         [("bona_fide", "0", "0", "NA"), ("attack", "2", "1", "0.5"), ("bona_fide", "0", "0", "NA"), ("attack", "0", "0", "NA")])
        # errore = attacco accettato (prediction 0) e bona fide rifiutato (prediction 1), mai il contrario
        st = N.strata_rows("r", 42, "a", [{"subject_id": a, "label": "1", "prediction": "0"}, {"subject_id": a, "label": "1", "prediction": "0"},
                                         {"subject_id": a, "label": "0", "prediction": "1"}, {"subject_id": a, "label": "0", "prediction": "0"},
                                         {"subject_id": a, "label": "0", "prediction": "0"}], grp)
        self.assertEqual([(x["class"], x["errors"], x["error_rate"]) for x in st[:2]], [("bona_fide", "1", repr(1 / 3)), ("attack", "2", "1.0")])
        with self.assertRaises(KeyError):
            N.strata_summary(r, t)       # gli argomenti non sono scambiabili (le righe di riferimento non hanno il gruppo)

    def test_summary_with_na(self):
        self.assertEqual(N1.summary([0.1, None, 0.3]), {"values": [0.1, None, 0.3], "median": None, "min": None, "max": None})
        self.assertEqual(N.range_text({"min": None, "max": None}), "NA")
        self.assertEqual(N.range_text({"min": 0.1, "max": 0.25}), "0.1000-0.2500")


# ============================================================================ ambiente e portabilità

class EnvironmentTests(unittest.TestCase):
    def test_unchanged_copies(self):
        """n1_train.py e train_cnn.py sono le copie accettate (SHA-256 del pacchetto N1 e del ramo research/pad-2026)."""
        self.assertEqual(N1.sha256_file(ROOT / "scripts" / "n1_train.py"), "78ae3e1da70c9214a3ef97f69c8e78eddbd0fc109fea08898877bd8c6c0c8b3c")
        self.assertEqual(N1.sha256_file(ROOT / "scripts" / "train_cnn.py"), "cbb010124d6a8b49e79a23cdc60b98c7962bc77006688dc5fdbe39c15e9e116c")

    def test_code_files_exist(self):
        for f in N.CODE_FILES:
            self.assertTrue((ROOT / f).is_file(), f)

    def test_no_private_library_api(self):
        import ast
        src = (ROOT / "scripts" / "n2_sessions.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        libs = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in ("numpy", "sklearn", "tensorflow", "keras", "matplotlib"):
                        libs[a.asname or a.name.split(".")[0]] = a.name
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in ("numpy", "sklearn", "tensorflow", "keras", "matplotlib"):
                for a in node.names:
                    libs[a.asname or a.name] = node.module
        bad = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr.startswith("_") and not node.attr.startswith("__"):
                base = node.value
                while isinstance(base, ast.Attribute):
                    base = base.value
                if isinstance(base, ast.Name) and base.id in libs:
                    bad.append(f"{base.id}.{node.attr}")
        self.assertEqual(bad, [])

    def test_frozen_module_loader(self):
        tmp = Path(tempfile.mkdtemp(dir=TMP_ROOT))
        p = tmp / "face_x.py"
        p.write_text("class FaceDetector:\n    def detect(self, img):\n        return None\n", encoding="utf-8")
        with self.assertRaises(N.N2Error):
            N.load_frozen_detector(p, "0" * 64)
        det = N.load_frozen_detector(p, sha(p.read_bytes()))
        self.assertIsNone(det.detect(None))
        self.assertIn("tesi_app.face_0ac7143", sys.modules)
        sys.modules.pop("tesi_app.face_0ac7143", None)
        rec = N.RecordingDetector(det)
        self.assertIsNone(rec.detect(None))
        self.assertEqual(N.detector_short(None), "none")
        self.assertEqual(N.detector_short(types.SimpleNamespace(detector="RetinaFace (Caffe, OpenCV DNN)")), "RetinaFace")
        self.assertEqual(N.detector_short(types.SimpleNamespace(detector="Haar cascade (Viola-Jones)")), "Haar")


class MutationHarnessTests(unittest.TestCase):
    def test_classify(self):
        import check_n2_mutations as M
        ok = "....\ntests run: 70, failures: 0, errors: 0, skipped: 0\nRESULT: PASS\n"
        self.assertEqual(M.classify(0, ok, 70)[0], "pass")
        self.assertEqual(M.classify(1, "F...\ntests run: 70, failures: 1, errors: 0, skipped: 0\nRESULT: FAIL\n", 70)[0], "fail")
        for rc, out in ((1, "Traceback (most recent call last):\nSyntaxError: invalid syntax\n"), (0, "tests run: 70, failures: 1, errors: 0, skipped: 0\n"),
                        (1, "tests run: 70, failures: 0, errors: 0, skipped: 0\n"), (1, "tests run: 3, failures: 1, errors: 0, skipped: 0\n"), (1, ok + ok)):
            with self.subTest(out=out[:40], rc=rc):
                self.assertEqual(M.classify(rc, out, 70)[0], "error")


class ZZRepositoryUntouched(unittest.TestCase):
    def test_tensorflow_never_imported(self):
        self.assertNotIn("tensorflow", sys.modules)

    def test_repository_results_untouched(self):
        self.assertEqual((ROOT / "results" / "n2").exists(), REPO_N2_BEFORE)
        if REPO_N2_BEFORE:
            self.assertEqual(sorted(p.name for p in (ROOT / "results" / "n2").iterdir()), REPO_N2_LISTING)
        for where in ("scripts", "results", "results/m2", "results/n1"):
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
