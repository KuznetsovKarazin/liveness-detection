"""
Test di scripts/nuaa_step_zero.py (step zero NUAA), offline e deterministici.

Mondo sintetico scritto in una cartella temporanea: immagini PNG minuscole con nomi NUAA (lette da cv2.imread per
contenuto), sei manifest coerenti (liste ufficiali, cnn_train/cnn_es dallo spread della lista di training, cnn_test
dallo spread della lista di test, c1 dentro cnn_test), una cache npz con lo stesso ordine, file di score con i layout
di N1, N2 (sessioni e combinato) e C1 (anche la colonna session_id sbagliata della v1, che non deve essere usata), i
riquadri del confronto del ritaglio, SHA256SUMS delle cartelle e la configurazione con gli hash veri. Statistiche
contro valori a mano (immagini costanti e a due toni); AUROC e Spearman su casi noti con pareggi (e contro scikit-learn
e scipy quando presenti); aggregati ricalcolati dalle righe riservate; percorsi d'errore (SHA diverso, file mancante,
collegamento simbolico, colonne mancanti, sessione del nome diversa dal manifest, score non finito, campione fuori
insieme, previsione incoerente, cache incoerente) -> FAIL senza uscite; nessun PASS vecchio dopo un FAIL; cartella di
uscita estranea o protetta rifiutata prima di qualunque modifica; nessun identificativo nel pubblico. Un test usa gli
ingressi veri in sola lettura (check-inputs) e si salta solo se mancano (MISSING_INPUT). Tutte le uscite vanno in
cartelle temporanee; al termine si controlla che results/, export/ e delivery/ del repository non siano cambiati.

  .venv/bin/python scripts/check_nuaa_step_zero.py        # uscita 1 se un test fallisce

Variabile facoltativa STEPZERO_INPUT_ROOT: radice degli ingressi veri (predefinita: la radice del repository; la usa il
controllo delle mutazioni, che esegue questa suite su una copia del codice).
"""
import contextlib
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nuaa_step_zero as SZ  # noqa: E402

REAL_ROOT = Path(os.environ.get("STEPZERO_INPUT_ROOT") or ROOT)
STATE = {"commit": "abcdef0", "dirty": False}
RUN_ID = "20261103-STEPZERO-abcdef0"
MISSING_INPUT = {
    "real inputs": "the NUAA images, the M2 manifests, the training cache or the accepted run folders are not present",
}
SUBJECTS_TRAIN = ("1101", "1102", "1103", "1104")       # sessioni 01 e 02
SUBJECTS_TEST = ("1105", "1106")                        # solo sessione 03; 1106 senza bona fide
PER_CELL = 3
PER_CLASS = 4
W, H = 8, 6


def skip_missing(test: unittest.TestCase, what: str):
    """Unico modo di saltare un test di questa suite: un ingresso mancante dichiarato in MISSING_INPUT."""
    test.skipTest(f"missing input: {MISSING_INPUT[what]}")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def png_bytes(img_bgr: np.ndarray) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".png", img_bgr)
    assert ok
    return buf.tobytes()


def image_for(index: int, label: int) -> np.ndarray:
    """Immagine BGR HxW deterministica: toni costanti che crescono con l'indice; gli attacchi hanno un bordo verticale
    (Laplaciano non nullo) e più saturazione."""
    v = 20 + (index * index) % 199                 # iniettiva sotto 100 immagini e non equispaziata: mediana per soggetto diversa dalla media
    img = np.zeros((H, W, 3), dtype=np.uint8)
    if label == 0:
        img[:, :, :] = v
    else:
        img[:, : W // 2, :] = v
        img[:, W // 2:, :] = (0, 0, min(255, v + 60))
    return img


# ============================================================================ mondo sintetico

class World:
    """Ingressi sintetici coerenti sotto una cartella temporanea `base`; `cfg` è la configurazione con gli hash."""

    def __init__(self, base: Path):
        self.base = base
        self.images = base / "data/NUAA/images"
        self.man_dir = base / "results/m2/manifests"
        self.rows = {}
        self.files = {}            # chiave -> byte PNG
        self.build()

    @staticmethod
    def key(label: int, subject: str, session: str, pic: int) -> str:
        return f"{'real' if label == 0 else 'attack'}/{subject}_{subject}_00_0{label}_{session}_{pic}.jpg"

    def build(self):
        rows_train, rows_test = [], []
        idx = 0
        for subj in SUBJECTS_TRAIN:
            for ses in ("01", "02"):
                for lab in (0, 1):
                    for pic in range(PER_CELL):
                        rows_train.append(self.row(lab, subj, ses, pic, idx)); idx += 1
        for subj in SUBJECTS_TEST:
            for lab in (0, 1):
                if subj == "1106" and lab == 0:
                    continue
                for pic in range(PER_CELL):
                    rows_test.append(self.row(lab, subj, "03", pic, idx)); idx += 1
        self.rows["official_train"], self.rows["official_test"] = rows_train, rows_test
        by = {r["key"]: r for r in rows_train + rows_test}
        order_train = SZ.spread_keys(rows_train, PER_CLASS)
        order_test = SZ.spread_keys(rows_test, PER_CLASS)
        self.order = {"train": order_train, "test": order_test}
        self.rows["cnn_es"] = [by[k] for k in order_train if by[k]["subject"] == "1104"]
        self.rows["cnn_train"] = [by[k] for k in order_train if by[k]["subject"] != "1104"]
        self.rows["cnn_test"] = [by[k] for k in order_test]
        c1_keys = SZ.spread_keys(self.rows["cnn_test"], 2)
        self.rows["c1"] = [by[k] for k in order_test if k in set(c1_keys)]
        used = {r["key"] for r in self.rows["cnn_train"]} | {r["key"] for r in self.rows["cnn_es"]}
        self.rows["target_491"] = sorted((r for r in rows_train if r["key"] not in used), key=lambda r: r["key"])
        self.by = by
        # file e manifest
        for r in rows_train + rows_test:
            p = self.images / SZ.LAYOUT[(r["label"], r["official_split"])] / r["key"].split("/", 1)[1]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(self.files[r["key"]])
        self.man_dir.mkdir(parents=True, exist_ok=True)
        self.man_files = {"official_train": "nuaa_official_train.csv", "official_test": "nuaa_official_test.csv", "cnn_train": "nuaa_cnn_train.csv",
                          "cnn_es": "nuaa_cnn_early_stopping.csv", "cnn_test": "nuaa_cnn_test.csv", "c1": "nuaa_c1_v2.csv"}
        mans = {}
        for name, f in self.man_files.items():
            text = self.csv(SZ.MANIFEST_COLUMNS, [[r[c] for c in SZ.MANIFEST_COLUMNS] for r in self.rows[name]])
            (self.man_dir / f).write_text(text, encoding="utf-8", newline="")
            rows = self.rows[name]
            nb = sum(1 for r in rows if r["label"] == "0")
            mans[name] = {"file": f, "sha256": sha(text.encode()), "rows": len(rows), "n_bona_fide": nb, "n_attack": len(rows) - nb,
                          "n_subjects": len({r["subject"] for r in rows}), "sessions": sorted({r["session"] for r in rows})}
        # cache: frame RGB sintetici (statistiche note dal solo indice)
        self.crop_img = {}
        X, Y, S = {}, {}, {}
        for member, order, src in (("train", order_train, rows_train), ("test", order_test, rows_test)):
            frames = []
            for i, k in enumerate(order):
                r = by[k]
                img = np.full((8, 8, 3), 10 + 3 * i, dtype=np.uint8)
                img[:, :, 0] = min(255, 10 + 3 * i + 40 * int(r["label"]))      # canale R più alto per gli attacchi
                frames.append(img)
                self.crop_img[k] = img
            X[member] = np.stack(frames)
            Y[member] = np.array([int(by[k]["label"]) for k in order], dtype=np.int64)
            S[member] = np.array([by[k]["subject"] for k in order], dtype="<U4")
        cache = self.base / "data/processed/nuaa_1500.npz"
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X_train=X["train"], y_train=Y["train"], s_train=S["train"], X_test=X["test"], y_test=Y["test"], s_test=S["test"])
        members = {"X_train": {"dtype": "|u1", "shape": [len(order_train), 8, 8, 3]}, "y_train": {"dtype": "<i8", "shape": [len(order_train)]},
                   "s_train": {"dtype": "<U4", "shape": [len(order_train)]}, "X_test": {"dtype": "|u1", "shape": [len(order_test), 8, 8, 3]},
                   "y_test": {"dtype": "<i8", "shape": [len(order_test)]}, "s_test": {"dtype": "<U4", "shape": [len(order_test)]}}
        # score: N1 (score = lum_mean / 300, monotono), N2 sessioni, combinato, C1 (con un analizzatore estraneo e session_id "00")
        self.raw_stats = {k: SZ.image_stats(self.decode(k), rgb=False) for k in by}
        scores = {}
        n1 = self.base / "results/n1/run-n1"
        files = {}
        for seed in (42, 43, 44):
            rows = [["run-n1", str(seed), r["key"], r["subject"], r["session"], r["label"], f"{self.score(r, seed):.6f}",
                     str(int(self.score(r, seed) > 0.5)), "1"] for r in self.rows["cnn_test"]]
            files[f"seed{seed}/scores_test.csv"] = self.csv(["run_id", "train_seed", "sample_id", "subject_id", "session_id", "label", "score_attack", "prediction", "in_c1_subset"], rows)
        self.write_folder(n1, files)
        for seed in (42, 43, 44):
            scores[f"n1_seed{seed}"] = self.hashed(n1, f"seed{seed}/scores_test.csv", {"set": "cnn_test", "rows": len(self.rows["cnn_test"]), "filter": {"train_seed": str(seed)},
                                                                                       "rejected_vs_accepted": True, "label": f"N1 seed {seed}"})
        n2 = self.base / "results/n2/run-n2"
        files = {}
        for seed in (42, 43, 44):
            rows = [["run-n2", str(seed), r["key"], r["subject"], r["session"], r["label"], "fitting", "0", f"{self.score(r, seed):.6f}", str(int(self.score(r, seed) > 0.5))]
                    for r in self.rows["target_491"]]
            files[f"seed{seed}/scores_sessions.csv"] = self.csv(["run_id", "train_seed", "sample_id", "subject_id", "session_id", "label", "group", "in_m1_dev", "score_attack", "prediction"], rows)
        self.write_folder(n2, files)
        for seed in (42, 43, 44):
            scores[f"n2_sessions_seed{seed}"] = self.hashed(n2, f"seed{seed}/scores_sessions.csv", {"set": "target_491", "rows": len(self.rows["target_491"]), "filter": {"train_seed": str(seed)},
                                                                                                    "rejected_vs_accepted": False, "label": f"N2 sessions seed {seed}"})
        n2c = self.base / "results/n2/run-n2c"
        files = {}
        for v in ("whole_image", "face_crop"):
            rows = [["run-n2c", v, r["key"], r["subject"], r["session"], r["label"], f"{self.score(r, 7):.6f}", str(int(self.score(r, 7) > 0.5))] for r in self.rows["cnn_test"]]
            if v == "whole_image":
                rows[0][6], rows[0][7] = "0.500000", "0"              # pareggio alla soglia: bona fide (accettato) per la regola
                self.tie_key = rows[0][2]
            files[f"{v}/scores_test.csv"] = self.csv(["run_id", "variant", "sample_id", "subject_id", "session_id", "label", "score_attack", "prediction"], rows)
        self.write_folder(n2c, files)
        for v in ("whole_image", "face_crop"):
            scores[f"n2_combined_{v}"] = self.hashed(n2c, f"{v}/scores_test.csv", {"set": "cnn_test", "rows": len(self.rows["cnn_test"]), "filter": {"variant": v},
                                                                                  "rejected_vs_accepted": False, "label": f"combined {v}"})
        c1 = self.base / "export/drive/02_Experiments/C1/run-c1"
        rows = []
        for an in ("livenessnet__nuaa", "attacknet_v1__nuaa", "minifasnet"):
            for r in self.rows["c1"]:
                s = self.score(r, 3 if an == "livenessnet__nuaa" else 4)
                rows.append([r["key"], "nuaa", "test", r["subject"], "00", "", "print", "RGB", r["label"], f"{s:.6f}", "0.5", str(int(s > 0.5)), an])
        self.write_folder(c1, {"scores.csv": self.csv(["sample_id", "dataset_id", "split", "subject_id", "session_id", "video_id", "attack_type", "modality", "label",
                                                        "score_attack", "threshold", "prediction", "analyzer_id"], rows)})
        for an in ("livenessnet__nuaa", "attacknet_v1__nuaa"):
            scores[f"c1_{an}"] = self.hashed(c1, "scores.csv", {"set": "c1", "rows": len(self.rows["c1"]), "filter": {"analyzer_id": an}, "rejected_vs_accepted": True, "label": f"C1 {an}"})
        # riquadri: bona fide 2x3, attacchi 4x3 (area relativa 6/48 = 0,125 e 12/48 = 0,25)
        crop = self.base / "results/c1crop/run-crop"
        rows = []
        for e in ("cache_0ac7143", "c1_a80f6f9"):
            for r in self.rows["c1"]:
                rows.append(["run-crop", e, r["key"], r["subject"], r["session"], r["label"], "RetinaFace", "0.9", "1", "1", "2" if r["label"] == "0" else "4", "3"])
        self.write_folder(crop, {"frames.csv": self.csv(["run_id", "extractor", "sample_id", "subject_id", "session_id", "label", "detector", "confidence",
                                                          "bbox_x", "bbox_y", "bbox_w", "bbox_h"], rows)})
        boxes = self.hashed(crop, "frames.csv", {"set": "c1", "extractors": ["cache_0ac7143", "c1_a80f6f9"], "rows": 2 * len(self.rows["c1"])})
        self.cfg = {
            "schema": SZ.CONFIG_SCHEMA, "task": "synthetic step zero", "nature": "test", "authorisation": "test",
            "declarations": {"no_training": True, "no_inference": True, "no_face_detection": True, "no_new_threshold": True,
                             "associations_not_causes": True, "unit": "frame"},
            "threshold": 0.5, "decision_rule": SZ.DECISION_RULE,
            "statistics": {n: f"definition of {n}" for n in SZ.RAW_STATS},
            "views": {"raw": "raw view text", "crop": "crop view text (chain)"},
            "measures": {k: f"text of {k}" for k in SZ.MEASURE_KEYS},
            "bootstrap": {"resamples": 1000, "seed": 0, "percentiles": [2.5, 97.5]},
            "manifests": {"folder": "results/m2/manifests", "origin": "synthetic", "files": mans},
            "images": {"folder": "data/NUAA/images", "layout": "<class>_<split>/<name>"},
            "cache": {"path": "data/processed/nuaa_1500.npz", "sha256": sha(cache.read_bytes()), "per_class": PER_CLASS, "members": members,
                      "order": "spread", "built": "synthetic"},
            "scores": scores, "face_boxes": {**boxes, "origin": "synthetic"},
            "sets": {s: f"set {s}" for s in SZ.RAW_SETS}, "outputs": {"public": "aggregates", "restricted": "rows"},
            "expected": {"n_images": len(by), "n_subjects": len(SUBJECTS_TRAIN) + len(SUBJECTS_TEST), "n_cache_frames": len(order_train) + len(order_test),
                         "n_target_491": len(self.rows["target_491"])},
        }
        self.config = self.base / "config/nuaa_step_zero.json"
        self.write_config()

    def by_cnn_test(self) -> set:
        return {r["key"] for r in self.rows["cnn_test"]}

    def row(self, lab: int, subj: str, ses: str, pic: int, idx: int) -> dict:
        key = self.key(lab, subj, ses, pic)
        b = png_bytes(image_for(idx, lab))
        self.files[key] = b
        return {"key": key, "label": str(lab), "subject": subj, "glasses": "00", "pos": f"0{lab}", "session": ses, "pic": str(pic),
                "official_split": SZ.SESSION_OF_SPLIT[ses], "sha256": sha(b), "bytes": str(len(b))}

    def decode(self, key: str) -> np.ndarray:
        import cv2
        return cv2.imdecode(np.frombuffer(self.files[key], dtype=np.uint8), cv2.IMREAD_COLOR)

    def score(self, r: dict, seed: int) -> float:
        """Score monotono nella luminanza media (Spearman 1 con lum_mean), spostato per seed."""
        return self.raw_stats[r["key"]]["lum_mean"] / 250 + seed * 1e-4

    @staticmethod
    def csv(header, rows) -> str:
        buf = io.StringIO(newline="")
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
        return buf.getvalue()

    @staticmethod
    def write_folder(d: Path, files: dict) -> None:
        for n, text in files.items():
            (d / n).parent.mkdir(parents=True, exist_ok=True)
            (d / n).write_text(text, encoding="utf-8", newline="")
        sums = "".join(f"{sha((d / n).read_bytes())}  {n}\n" for n in sorted(files))
        (d / "SHA256SUMS").write_text(sums, encoding="utf-8")

    def hashed(self, d: Path, rel: str, extra: dict) -> dict:
        return {"path": (d / rel).relative_to(self.base).as_posix(), "sha256": sha((d / rel).read_bytes()),
                "sums": (d / "SHA256SUMS").relative_to(self.base).as_posix(), "sums_sha256": sha((d / "SHA256SUMS").read_bytes()), **extra}

    def write_config(self) -> None:
        self.config.parent.mkdir(parents=True, exist_ok=True)
        self.config.write_text(json.dumps(self.cfg, indent=1) + "\n", encoding="utf-8")

    def snapshot(self) -> dict:
        out = {}
        for dp, dn, fn in os.walk(self.base):
            for f in fn:
                p = Path(dp) / f
                if self.base / "out" in p.parents:
                    continue
                out[str(p)] = sha(p.read_bytes()) if not p.is_symlink() else "link"
        return out


# ============================================================================ esecuzione

def run_cli(args, world: World, out: Path = None, state: dict = None) -> tuple:
    """(codice d'uscita, stdout, stderr) di main con ingressi e uscite del mondo sintetico e stato git finto."""
    out = out or world.base / "out"
    argv = [args[0], "--config", str(world.config), "--input-root", str(world.base), "--out-root", str(out), "--date", "20261103"] + list(args[1:])
    so, se = io.StringIO(), io.StringIO()
    old = SZ.REPO_STATE
    SZ.REPO_STATE = lambda root: dict(state or STATE)
    try:
        with contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
            rc = SZ.main(argv)
    finally:
        SZ.REPO_STATE = old
    return rc, so.getvalue(), se.getvalue()


def run_ok(world: World, out: Path = None) -> Path:
    rc, so, se = run_cli(["run", "--i-am-authorized"], world, out)
    assert rc == 0, se + so
    return (out or world.base / "out") / RUN_ID


def read_csv(p: Path) -> list:
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_sums(d: Path) -> bool:
    """Ogni riga di SHA256SUMS verificata e nomi elencati = file della cartella meno SHA256SUMS."""
    listed = []
    for line in (d / "SHA256SUMS").read_text().splitlines():
        h, n = line.split("  ", 1)
        listed.append(n)
        if sha((d / n).read_bytes()) != h:
            return False
    return sorted(listed) == sorted(set(os.listdir(d)) - {"SHA256SUMS"})


# ============================================================================ test

class TestKnownAnswers(unittest.TestCase):
    def test_constant_red_image(self):
        img = np.zeros((4, 5, 3), dtype=np.uint8)
        img[:, :, 2] = 255                                   # BGR: rosso
        s = SZ.image_stats(img, rgb=False)
        self.assertEqual((s["lum_mean"], s["lum_std"], s["sat_mean"], s["lap_var"]), (76.0, 0.0, 255.0, 0.0))   # round(0.299 * 255) = 76
        s2 = SZ.image_stats(img, rgb=True)                   # letto come RGB è blu: round(0.114 * 255) = 29
        self.assertEqual(s2["lum_mean"], 29.0)

    def test_two_tone_image(self):
        img = np.zeros((6, 8, 3), dtype=np.uint8)
        img[:, 4:, :] = 255
        s = SZ.image_stats(img, rgb=False)
        self.assertEqual((s["lum_mean"], s["lum_std"], s["sat_mean"]), (127.5, 127.5, 0.0))
        self.assertAlmostEqual(s["lap_var"], 2 * 255 ** 2 / 8, places=9)      # ±255 sulle due colonne del bordo, media 0

    def test_image_stats_refuses_wrong_shape(self):
        with self.assertRaises(SZ.Failure):
            SZ.image_stats(np.zeros((4, 4), dtype=np.uint8), rgb=False)

    def test_ranks_with_ties(self):
        self.assertEqual(SZ.ranks([3, 1, 1, 2]).tolist(), [4.0, 1.5, 1.5, 3.0])

    def test_auroc_known(self):
        self.assertEqual(SZ.auroc([0.9, 0.8], [0.1, 0.2]), 1.0)
        self.assertEqual(SZ.auroc([0.1, 0.2], [0.9, 0.8]), 0.0)
        self.assertEqual(SZ.auroc([0.5, 0.5], [0.5]), 0.5)                      # pareggi a 0,5
        self.assertEqual(SZ.auroc([0.7, 0.2], [0.5, 0.2]), 0.625)                # (1 + 1 + 0 + 0.5) / 4
        self.assertIsNone(SZ.auroc([], [0.1]))
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            return                                           # confronto facoltativo: senza scikit-learn restano i valori a mano
        rng = np.random.default_rng(0)
        pos, neg = rng.integers(0, 5, 30), rng.integers(0, 5, 20)
        self.assertAlmostEqual(SZ.auroc(pos, neg), roc_auc_score([1] * 30 + [0] * 20, np.concatenate([pos, neg])), places=12)

    def test_spearman_known(self):
        self.assertAlmostEqual(SZ.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(SZ.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(SZ.spearman([1, 1, 1], [1, 2, 3]))
        self.assertIsNone(SZ.spearman([1, 2], [1, 2]))
        self.assertAlmostEqual(SZ.spearman([1, 2, 2, 3], [1, 2, 3, 4]), 0.9486832980505138)   # ranghi medi: (1, 2.5, 2.5, 4) contro (1, 2, 3, 4)
        try:
            from scipy.stats import spearmanr
        except ImportError:
            return
        rng = np.random.default_rng(1)
        x, y = rng.integers(0, 6, 40), rng.integers(0, 6, 40)
        self.assertAlmostEqual(SZ.spearman(x, y), spearmanr(x, y).statistic, places=12)

    def test_quartiles(self):
        self.assertEqual(SZ.quartiles([4, 1, 3, 2]), {"n": 4, "median": 2.5, "q1": 1.75, "q3": 3.25})
        self.assertEqual(SZ.quartiles([]), {"n": 0, "median": None, "q1": None, "q3": None})

    def test_spread_copies_the_rule(self):
        self.assertEqual(SZ.spread([5, 1, 3, 2, 4], 2), [1, 3])
        self.assertEqual(SZ.spread([2, 1], 5), [1, 2])
        rows = [{"key": f"attack/b{i}", "label": "1"} for i in range(4)] + [{"key": f"real/a{i}", "label": "0"} for i in range(4)]
        self.assertEqual(SZ.spread_keys(rows, 2), ["real/a0", "real/a2", "attack/b0", "attack/b2"])

    def test_parse_name(self):
        f = SZ.parse_name("real/1104_1104_01_06_03_12.jpg", "x")
        self.assertEqual((f["subject"], f["session"], f["dir"]), ("1104", "03", "real"))
        for bad in ("real/1104_1105_01_06_03_12.jpg", "other/1104_1104_01_06_03_12.jpg", "real/1104.jpg"):
            with self.assertRaises(SZ.Failure):
                SZ.parse_name(bad, "x")

    def test_within_subject_auroc(self):
        subs = {"a": {"attack": {"x": np.array([3.0, 4.0])}, "bona_fide": {"x": np.array([1.0, 2.0])}},
                "b": {"attack": {"x": np.array([1.0])}, "bona_fide": {"x": np.array([2.0, 1.0])}},
                "c": {"attack": {"x": np.array([5.0])}}}
        got = SZ.subject_aurocs(subs, ("x",))
        self.assertEqual(sorted(got), ["a", "b"])                                   # c ha una sola classe
        self.assertEqual(got["a"], (2, 2, {"x": 1.0}))
        self.assertEqual(got["b"], (2, 1, {"x": 0.25}))                              # 1 contro (2, 1): 0 + 0.5 su 2 coppie

    def test_bootstrap_matches_the_declared_recipe(self):
        subs = {"a": {"attack": {"x": np.array([3.0, 4.0])}, "bona_fide": {"x": np.array([1.0, 2.0])}},
                "b": {"attack": {"x": np.array([1.0])}, "bona_fide": {"x": np.array([2.0, 5.0])}},
                "c": {"attack": {"x": np.array([5.0])}}}
        got = SZ.bootstrap_auroc(subs, ("x",))["x"]
        self.assertEqual(got, SZ.bootstrap_auroc(subs, ("x",))["x"])                 # deterministico
        # replica indipendente della ricetta dichiarata: generatore con seme BOOT.seed, indici fra i soggetti ordinati
        names = sorted(subs)
        rng = np.random.default_rng(SZ.BOOT["seed"])
        vals = []
        for _ in range(SZ.BOOT["resamples"]):
            idx = rng.integers(0, len(names), len(names))
            att = np.concatenate([subs[names[i]].get("attack", {}).get("x", np.empty(0)) for i in idx])
            bf = np.concatenate([subs[names[i]].get("bona_fide", {}).get("x", np.empty(0)) for i in idx])
            if len(att) and len(bf):
                vals.append(SZ.auroc(att, bf))
        lo, hi = np.percentile(vals, SZ.BOOT["percentiles"])
        self.assertEqual(got, {"low": float(lo), "high": float(hi), "valid_resamples": len(vals)})
        self.assertTrue(0 < got["valid_resamples"] < SZ.BOOT["resamples"])           # i ricampionamenti con il solo soggetto c non contano
        self.assertEqual(SZ.bootstrap_auroc({"c": subs["c"]}, ("x",))["x"], {"low": None, "high": None, "valid_resamples": 0})

    def test_bootstrap_recipe_literal(self):
        self.assertEqual(SZ.BOOT, {"resamples": 1000, "seed": 0, "percentiles": [2.5, 97.5]})

    def test_bootstrap_non_degenerate_fixture(self):
        """Dieci soggetti sbilanciati con valori continui: AUROC a forza bruta sulle coppie, stessi indici per ogni
        statistica (y = -x: low_x = 1 - high_y), e la deduplicazione dei soggetti ricampionati cambierebbe i percentili."""
        rng = np.random.default_rng(3)
        subs = {}
        for i in range(10):
            na, nb = int(rng.integers(1, 7)), int(rng.integers(1, 7))
            att = rng.normal(0.3 * i, 1.0, na)
            bf = rng.normal(0.0, 1.0, nb)
            subs[f"s{i:02d}"] = {"attack": {"x": att, "y": -att}, "bona_fide": {"x": bf, "y": -bf}}
        got = SZ.bootstrap_auroc(subs, ("x", "y"))
        self.assertEqual(got["x"]["valid_resamples"], SZ.BOOT["resamples"])             # ogni soggetto ha entrambe le classi

        def brute(att, bf):
            return float(np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a in att for b in bf]))
        names = sorted(subs)
        r = np.random.default_rng(SZ.BOOT["seed"])
        vals, dedup = [], []
        for _ in range(SZ.BOOT["resamples"]):
            idx = r.integers(0, len(names), len(names))
            vals.append(brute(np.concatenate([subs[names[i]]["attack"]["x"] for i in idx]), np.concatenate([subs[names[i]]["bona_fide"]["x"] for i in idx])))
            u = np.unique(idx)
            dedup.append(brute(np.concatenate([subs[names[i]]["attack"]["x"] for i in u]), np.concatenate([subs[names[i]]["bona_fide"]["x"] for i in u])))
        lo, hi = np.percentile(vals, SZ.BOOT["percentiles"])
        self.assertAlmostEqual(got["x"]["low"], lo, places=12)
        self.assertAlmostEqual(got["x"]["high"], hi, places=12)
        self.assertTrue(0 < lo < hi < 1)
        self.assertAlmostEqual(got["x"]["low"], 1 - got["y"]["high"], places=12)        # stessi indici di ricampionamento per x e y
        self.assertAlmostEqual(got["x"]["high"], 1 - got["y"]["low"], places=12)
        dlo, dhi = np.percentile(dedup, SZ.BOOT["percentiles"])
        self.assertTrue(abs(dlo - lo) > 1e-9 or abs(dhi - hi) > 1e-9)                   # la deduplicazione cambierebbe i percentili

    def test_within_subject_summary_median_min_max(self):
        rows = [{"key": k, "label": lab, "subject": s, "session": "01"} for s, lab, k in (
            ("a", "1", "a1"), ("a", "0", "a2"), ("a", "0", "a3"), ("a", "0", "a4"), ("a", "0", "a5"),     # 0,25
            ("b", "1", "b1"), ("b", "0", "b2"), ("b", "0", "b3"), ("b", "0", "b4"), ("b", "0", "b5"),     # 0,75
            ("c", "1", "c1"), ("c", "0", "c2"), ("c", "0", "c3"))]                                        # 1,0
        stats = {"a1": {"x": 1.5}, "a2": {"x": 1}, "a3": {"x": 2}, "a4": {"x": 3}, "a5": {"x": 4},
                 "b1": {"x": 3.5}, "b2": {"x": 1}, "b3": {"x": 2}, "b4": {"x": 3}, "b5": {"x": 4},
                 "c1": {"x": 5}, "c2": {"x": 1}, "c3": {"x": 2}}
        w = SZ.set_summary(rows, stats, ("x",), boot=False)["01"]["within_subject_auroc"]
        self.assertEqual(w, {"n_subjects": 3, "x": {"median": 0.75, "min": 0.25, "max": 1.0}})
        self.assertNotEqual(w["x"]["median"], (0.25 + 0.75 + 1.0) / 3)

    def test_public_problems(self):
        inp = SZ.Inputs()
        inp.all_rows = [{"sha256": "a" * 64}]
        inp.subjects = ["1101", "1112"]
        self.assertEqual(SZ.public_problems('{"n": 1, "v": 0.1112, "h": "b1112c"}', inp), [])
        self.assertIn("subject id 1112", SZ.public_problems("subject 1112 here", inp))
        self.assertIn("image keys", SZ.public_problems("real/1101_1101_00_00_01_0.jpg", inp))
        self.assertIn("image SHA-256 values", SZ.public_problems("a" * 64, inp))
        self.assertTrue(any("marker" in x for x in SZ.public_problems("T" + "BD", inp, markdown=True)))
        self.assertEqual(SZ.public_problems("T" + "BD", inp, markdown=False), [])


class TestWorld(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="stepzero-")
        cls.world = World(Path(cls.tmp))
        cls.run_dir = run_ok(cls.world)
        cls.doc = json.loads((cls.run_dir / "public/aggregate.json").read_text())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_check_inputs_pass_and_reads_only(self):
        before = self.world.snapshot()
        rc, so, se = run_cli(["check-inputs"], self.world)
        self.assertEqual(rc, 0, se)
        self.assertIn("RESULT: PASS (check-inputs)", so)
        self.assertEqual(self.world.snapshot(), before)
        self.assertFalse((self.world.base / "out2").exists())

    def test_run_outputs_and_sums(self):
        st = json.loads((self.run_dir / "status.json").read_text())
        self.assertEqual((st["status"], st["passed"], st["run_id"]), ("PASS", True, RUN_ID))
        try:
            import matplotlib  # noqa: F401
            expected = SZ.PUBLIC_OUTPUTS                                   # figura presente con matplotlib
        except ImportError:
            expected = tuple(n for n in SZ.PUBLIC_OUTPUTS if n != SZ.FIGURE)   # senza matplotlib: salto dichiarato
            self.assertIn("skipped", st["figure"])
        self.assertEqual(sorted(os.listdir(self.run_dir / "public")), sorted(expected))
        self.assertEqual(sorted(os.listdir(self.run_dir / "restricted")), sorted(SZ.RESTRICTED_OUTPUTS))
        self.assertTrue(check_sums(self.run_dir / "public") and check_sums(self.run_dir / "restricted"))
        anchor = (self.run_dir / "restricted/public_SHA256SUMS.sha256").read_text().split()[0]
        self.assertEqual(anchor, sha((self.run_dir / "public/SHA256SUMS").read_bytes()))
        self.assertEqual(st["public_sums_sha256"], anchor)

    def test_restricted_rows_and_hand_values(self):
        raw = read_csv(self.run_dir / "restricted/per_image_raw.csv")
        self.assertEqual(len(raw), len(self.world.by))
        by = {r["sample_id"]: r for r in raw}
        k = self.world.key(0, "1101", "01", 0)                 # indice 0: immagine costante a 20
        self.assertEqual((by[k]["lum_mean"], by[k]["lum_std"], by[k]["sat_mean"], by[k]["lap_var"]), ("20.000000", "0.000000", "0.000000", "0.000000"))
        self.assertEqual((by[k]["width"], by[k]["height"], by[k]["bytes"], by[k]["session_id"], by[k]["label"]), (str(W), str(H), self.world.by[k]["bytes"], "01", "0"))
        self.assertEqual(sum(int(r["in_target_491"]) for r in raw), len(self.world.rows["target_491"]))
        self.assertEqual(sum(int(r["in_c1"]) for r in raw), len(self.world.rows["c1"]))
        crop = read_csv(self.run_dir / "restricted/per_image_crop.csv")
        self.assertEqual(len(crop), self.world.cfg["expected"]["n_cache_frames"])
        c0 = crop[0]
        self.assertEqual((c0["cache_member"], c0["cache_index"], c0["sample_id"]), ("X_train", "0", self.world.order["train"][0]))
        img = self.world.crop_img[c0["sample_id"]]
        expected = SZ.image_stats(img, rgb=True)
        self.assertEqual(c0["lum_mean"], f"{expected['lum_mean']:.6f}")
        subj = read_csv(self.run_dir / "restricted/per_subject.csv")
        self.assertTrue(all(r["view"] in ("raw", "crop") for r in subj))
        raw01 = [r for r in subj if r["view"] == "raw" and r["subject_id"] == "1101" and r["session_id"] == "01" and r["label"] == "0"]
        self.assertEqual(len(raw01), 1)
        vals = sorted(float(by[self.world.key(0, "1101", "01", p)]["lum_mean"]) for p in range(PER_CELL))
        self.assertEqual(raw01[0]["median_lum_mean"], f"{vals[1]:.6f}")
        self.assertNotEqual(f"{vals[1]:.6f}", f"{sum(vals) / len(vals):.6f}")         # toni non equispaziati: la media sarebbe diversa

    def test_aggregate_counts_and_recomputation(self):
        d = self.doc
        self.assertEqual(d["counts"]["official_images"], len(self.world.by))
        s01 = d["sets_raw"]["official_train"]["01"]
        self.assertEqual((s01["n_bona_fide"], s01["n_attack"]), (len(SUBJECTS_TRAIN) * PER_CELL, len(SUBJECTS_TRAIN) * PER_CELL))
        self.assertEqual(d["sets_raw"]["official_test"]["03"]["classes"]["bona_fide"]["lum_mean"]["n"], PER_CELL)   # solo il soggetto 1105
        self.assertEqual(d["sets_raw"]["official_test"]["03"]["class_auroc"]["width"], 0.5)                     # larghezza costante: pareggi
        self.assertIn("all", d["sets_raw"]["official_train"])
        self.assertNotIn("all", d["sets_raw"]["official_test"])
        # ricalcolo dalle righe riservate
        raw = read_csv(self.run_dir / "restricted/per_image_raw.csv")
        att = [float(r["sat_mean"]) for r in raw if r["session_id"] == "02" and r["label"] == "1"]
        bf = [float(r["sat_mean"]) for r in raw if r["session_id"] == "02" and r["label"] == "0"]
        self.assertAlmostEqual(d["sets_raw"]["official_train"]["02"]["class_auroc"]["sat_mean"], SZ.auroc(att, bf), places=6)
        self.assertEqual(d["sets_raw"]["official_train"]["02"]["class_auroc"]["sat_mean"], 1.0)                 # gli attacchi sono più saturi
        self.assertAlmostEqual(d["sets_crop"]["cnn_test"]["03"]["class_auroc"]["lum_mean"], 1.0)                # canale R più alto per gli attacchi
        # AUROC entro soggetto e intervallo bootstrap sui soggetti
        w01 = d["sets_raw"]["official_train"]["01"]["within_subject_auroc"]
        self.assertEqual(w01["n_subjects"], len(SUBJECTS_TRAIN))
        self.assertEqual(w01["sat_mean"], {"median": 1.0, "min": 1.0, "max": 1.0})
        w03 = d["sets_raw"]["official_test"]["03"]["within_subject_auroc"]
        self.assertEqual((w03["n_subjects"], d["sets_raw"]["official_test"]["03"]["n_subjects"]), (1, 2))      # 1106 senza bona fide
        ci = d["sets_raw"]["official_train"]["02"]["class_auroc_ci"]["sat_mean"]
        self.assertEqual((ci["low"], ci["high"], ci["valid_resamples"]), (1.0, 1.0, SZ.BOOT["resamples"]))
        ci03 = d["sets_raw"]["official_test"]["03"]["class_auroc_ci"]["lum_mean"]
        self.assertTrue(0 < ci03["valid_resamples"] < SZ.BOOT["resamples"])                                   # ricampionamenti con il solo 1106
        self.assertTrue(ci03["low"] <= d["sets_raw"]["official_test"]["03"]["class_auroc"]["lum_mean"] <= ci03["high"] or ci03["low"] <= ci03["high"])
        self.assertIn("confound subject and class", " ".join(d["limits"]))
        rows = read_csv(self.run_dir / "restricted/per_subject_auroc.csv")
        r01 = [r for r in rows if r["view"] == "raw" and r["set"] == "official_train" and r["session"] == "01"]
        self.assertEqual(sorted(r["subject_id"] for r in r01), list(SUBJECTS_TRAIN))
        self.assertTrue(all(r["auroc_sat_mean"] == "1.000000" and r["n_bona_fide"] == str(PER_CELL) for r in r01))
        self.assertEqual([r["subject_id"] for r in rows if r["view"] == "raw" and r["set"] == "official_test" and r["session"] == "03"], ["1105"])
        self.assertTrue(all(r["auroc_width"] == "NA" for r in rows if r["view"] == "crop"))
        crop03 = [r for r in rows if r["view"] == "crop" and r["set"] == "cnn_test" and r["session"] == "03"]
        self.assertEqual([r["subject_id"] for r in crop03], ["1105"])
        crop_rows = read_csv(self.run_dir / "restricted/per_image_crop.csv")
        att = [float(r["lum_mean"]) for r in crop_rows if r["subject_id"] == "1105" and r["session_id"] == "03" and r["label"] == "1"]
        bf = [float(r["lum_mean"]) for r in crop_rows if r["subject_id"] == "1105" and r["session_id"] == "03" and r["label"] == "0"]
        self.assertEqual((crop03[0]["n_bona_fide"], crop03[0]["n_attack"]), (str(len(bf)), str(len(att))))
        self.assertEqual(crop03[0]["auroc_lum_mean"], f"{SZ.auroc(att, bf):.6f}")
        # conteggi dei soggetti della sessione 03 scritti dal run, non cablati
        self.assertEqual(d["counts"]["subjects_session_03"], {"bona_fide": 1, "attack": 2, "both_classes": 1})
        self.assertIn("come from 1 subjects and the attacks from 2 subjects, 1 subjects have both classes", " ".join(d["limits"]))
        self.assertEqual(d["measures"], self.world.cfg["measures"])
        self.assertEqual((d["views"], d["bootstrap"]), (self.world.cfg["views"], SZ.BOOT))
        self.assertIn("text of all_row", " ".join(d["limits"]))

    def test_score_associations(self):
        a = self.doc["score_associations"]
        self.assertEqual(sorted(a), sorted(self.world.cfg["scores"]))
        for c in ("bona_fide", "attack", "all"):
            self.assertAlmostEqual(a["n1_seed42"]["raw"]["03"][c]["spearman"]["lum_mean"], 1.0)              # score monotono in lum_mean
        self.assertIsNone(a["n1_seed42"]["raw"]["03"]["all"]["spearman"]["width"])                           # costante -> NA
        self.assertEqual(a["n1_seed42"]["raw"]["03"]["all"]["n"], len(self.world.rows["cnn_test"]))
        self.assertIsInstance(a["n1_seed42"]["crop"], dict)
        self.assertEqual(a["n2_sessions_seed42"]["crop"], "NA: frames not in the training cache")
        self.assertIn("all", a["n2_sessions_seed42"]["raw"])
        self.assertEqual(a["n2_sessions_seed42"]["raw"]["all"]["all"]["n"], len(self.world.rows["target_491"]))
        self.assertEqual(a["c1_livenessnet__nuaa"]["raw"]["03"]["all"]["n"], len(self.world.rows["c1"]))      # sessione dal nome, non dalla colonna "00"
        self.assertNotIn("00", a["c1_livenessnet__nuaa"]["raw"])

    def test_rejected_vs_accepted_and_boxes(self):
        r = self.doc["rejected_vs_accepted"]
        self.assertEqual(sorted(r), sorted(k for k, v in self.world.cfg["scores"].items() if v["rejected_vs_accepted"]))
        v = r["n1_seed42"]["raw"]
        bf = [x for x in self.world.rows["cnn_test"] if x["label"] == "0"]
        rejected = sum(1 for x in bf if self.world.score(x, 42) > 0.5)
        self.assertEqual((v["n_rejected"], v["n_accepted"]), (rejected, len(bf) - rejected))
        self.assertTrue(0 < rejected < len(bf))                                                 # entrambi i gruppi presenti
        self.assertEqual(v["auroc_rejected_vs_accepted"]["lum_mean"], 1.0)                      # rifiutati = luminanza più alta
        self.assertEqual(v["session"], "03")
        b = self.doc["face_box_annex"]["extractors"]
        for e in ("cache_0ac7143", "c1_a80f6f9"):
            self.assertAlmostEqual(b[e]["relative_area"]["bona_fide"]["median"], 6 / (W * H))
            self.assertAlmostEqual(b[e]["relative_area"]["attack"]["median"], 12 / (W * H))
            self.assertEqual(b[e]["class_auroc"], 1.0)

    def test_public_has_no_identifiers(self):
        inp = SZ.Inputs()
        inp.all_rows, inp.subjects = self.world.by.values(), sorted({r["subject"] for r in self.world.by.values()})
        for n, md in (("aggregate.json", False), ("aggregate.md", True)):
            self.assertEqual(SZ.public_problems((self.run_dir / "public" / n).read_text(), inp, md), [], n)
        md = (self.run_dir / "public/aggregate.md").read_text()
        self.assertIn("associations, not causes", md)
        self.assertIn("| 01 | bona fide |", md)

    def test_figure_present_with_plotter_and_skipped_without(self):
        old = SZ.PLOTTER
        try:
            SZ.PLOTTER = lambda doc, arrays: b"\x89PNG-test"
            run = run_ok(self.world, self.world.base / "out_fig")
            self.assertEqual((run / "public" / SZ.FIGURE).read_bytes(), b"\x89PNG-test")
            self.assertTrue(check_sums(run / "public"))

            def missing(doc, arrays):
                raise ImportError(name="matplotlib")
            SZ.PLOTTER = missing
            run = run_ok(self.world, self.world.base / "out_nofig")
            self.assertFalse((run / "public" / SZ.FIGURE).exists())
            doc = json.loads((run / "public/aggregate.json").read_text())
            self.assertIn("skipped", doc["figure"])
            self.assertNotIn(SZ.FIGURE, (run / "public/SHA256SUMS").read_text())
            self.assertIn("skipped", json.loads((run / "status.json").read_text())["figure"])
        finally:
            SZ.PLOTTER = old

    def test_forbidden_public_content_stops_the_run(self):
        old = SZ.aggregate_md
        try:
            SZ.aggregate_md = lambda doc: old(doc) + "\nreal/1101_1101_00_00_01_0.jpg\n"
            rc, so, se = run_cli(["run", "--i-am-authorized"], self.world, self.world.base / "out_forbidden")
            self.assertEqual(rc, 1)
            self.assertIn("public aggregate.md would contain forbidden content: image keys", se)
            run = self.world.base / "out_forbidden" / RUN_ID
            self.assertEqual(os.listdir(run / "public") if (run / "public").is_dir() else [], [])
            self.assertEqual(json.loads((run / "status.json").read_text())["status"], "FAIL")
        finally:
            SZ.aggregate_md = old

    def test_dirty_tree_gives_dev_run_id(self):
        rc, so, _ = run_cli(["run", "--i-am-authorized"], self.world, self.world.base / "out_dev", {"commit": "abcdef0", "dirty": True})
        self.assertEqual(rc, 0)
        self.assertTrue((self.world.base / "out_dev" / (RUN_ID + "-dev") / "status.json").exists())

    def test_run_requires_authorisation(self):
        rc, so, se = run_cli(["run"], self.world, self.world.base / "out_noauth")
        self.assertEqual(rc, 1)
        self.assertIn("--i-am-authorized", se)
        self.assertFalse((self.world.base / "out_noauth").exists())


class TestFailures(unittest.TestCase):
    """Ogni ingresso guasto: FAIL, status.json FAIL, nessuna uscita; ingressi intatti."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="stepzero-fail-")
        self.world = World(Path(self.tmp))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def expect_fail(self, text: str, cmd=("run", "--i-am-authorized")):
        rc, so, se = run_cli(list(cmd), self.world)
        self.assertEqual(rc, 1, so + se)
        self.assertIn(text, se)
        run = self.world.base / "out" / RUN_ID
        if cmd[0] == "run":
            st = json.loads((run / "status.json").read_text())
            self.assertEqual((st["status"], st["passed"]), ("FAIL", False))
            self.assertIn(text, st["reason"])
            for sub in ("public", "restricted"):
                self.assertEqual(os.listdir(run / sub) if (run / sub).is_dir() else [], [])
        return se

    def test_manifest_sha_mismatch(self):
        self.world.cfg["manifests"]["files"]["cnn_test"]["sha256"] = "0" * 64
        self.world.write_config()
        self.expect_fail("manifest cnn_test (nuaa_cnn_test.csv): SHA-256 mismatch", ("check-inputs",))
        self.expect_fail("manifest cnn_test (nuaa_cnn_test.csv): SHA-256 mismatch")

    def test_score_sha_and_sums(self):
        spec = self.world.cfg["scores"]["n1_seed42"]
        p = self.world.base / spec["path"]
        p.write_text(p.read_text().replace("0.", "0.0", 1), newline="")
        self.expect_fail("score file n1_seed42 (scores_test.csv): SHA-256 mismatch")
        self.world.cfg["scores"]["n1_seed42"]["sha256"] = sha(p.read_bytes())
        self.world.write_config()
        self.expect_fail("SHA-256 mismatch with its SHA256SUMS")
        sums = self.world.base / spec["sums"]
        sums.write_text("".join(l + "\n" for l in sums.read_text().splitlines() if "seed42" not in l))
        self.world.cfg["scores"]["n1_seed42"]["sums_sha256"] = sha(sums.read_bytes())
        self.world.write_config()
        self.expect_fail("not listed exactly once in its SHA256SUMS")
        self.world.cfg["scores"]["n1_seed42"]["sums_sha256"] = "1" * 64
        self.world.write_config()
        self.expect_fail("SHA256SUMS of score file n1_seed42: SHA-256 mismatch")

    def test_missing_and_altered_original(self):
        r = self.world.rows["official_test"][0]
        p = self.world.images / SZ.LAYOUT[(r["label"], r["official_split"])] / r["key"].split("/", 1)[1]
        data = p.read_bytes()
        flipped = bytearray(data)
        flipped[-1] ^= 0x01                                     # stessa dimensione, un bit diverso
        p.write_bytes(bytes(flipped))
        self.expect_fail("SHA-256 differs from the manifest")
        p.write_bytes(data + b"\0")
        self.expect_fail("size differs from the manifest")
        p.unlink()
        self.expect_fail("not found under the images folder")

    def test_tie_at_the_threshold_is_bona_fide(self):
        spec = self.world.cfg["scores"]["n2_combined_whole_image"]
        p = self.world.base / spec["path"]
        lines = p.read_text().splitlines()
        self.assertIn(",0.500000,0", lines[1])                   # la riga di pareggio con previsione 0 è accettata dal mondo
        lines[1] = lines[1].replace(",0.500000,0", ",0.500000,1")
        self.rehash(p, "\n".join(lines) + "\n", "n2_combined_whole_image")
        self.expect_fail("stored prediction '1' differs from the rule score > 0.5")

    def test_cache_changed_configuration_unchanged(self):
        cache = self.world.base / self.world.cfg["cache"]["path"]
        with np.load(cache) as z:
            d = {k: z[k] for k in z}
        d["X_train"] = d["X_train"].copy()
        d["X_train"][0, 0, 0, 0] ^= 1
        np.savez_compressed(cache, **d)
        self.expect_fail("SHA-256 " + sha(cache.read_bytes()) + " differs from the expected")

    def test_configuration_counts_altered(self):
        self.world.cfg["manifests"]["files"]["cnn_es"]["rows"] += 1
        self.world.write_config()
        self.expect_fail(f"manifest cnn_es: rows {len(self.world.rows['cnn_es'])} differs from the configuration")
        self.world.cfg["manifests"]["files"]["cnn_es"]["rows"] -= 1
        self.world.cfg["manifests"]["files"]["official_train"]["n_subjects"] += 1
        self.world.write_config()
        self.expect_fail(f"manifest official_train: n_subjects {len(SUBJECTS_TRAIN)} differs from the configuration")

    def test_expected_cache_frames(self):
        self.world.cfg["expected"]["n_cache_frames"] += 1
        self.world.write_config()
        self.expect_fail("frames do not match the expected count", ("check-inputs",))

    def test_box_with_zero_width(self):
        p = self.world.base / self.world.cfg["face_boxes"]["path"]
        lines = p.read_text().splitlines()
        parts = lines[1].split(",")
        parts[10] = "0"
        lines[1] = ",".join(parts)
        self.rehash(p, "\n".join(lines) + "\n", box=True)
        self.expect_fail("box width and height must be positive")

    def test_cache_fortran_and_pickle_members_refused(self):
        cache = self.world.base / self.world.cfg["cache"]["path"]
        with np.load(cache) as z:
            d = {k: z[k] for k in z}
        np.savez_compressed(cache, **{**d, "X_test": np.asfortranarray(d["X_test"])})
        self.world.cfg["cache"]["sha256"] = sha(cache.read_bytes())
        self.world.write_config()
        self.expect_fail("X_test has dtype |u1", ("check-inputs",))
        np.savez_compressed(cache, **{**d, "s_test": np.array(list(d["s_test"]) , dtype=object)})
        self.world.cfg["cache"]["sha256"] = sha(cache.read_bytes())
        self.world.write_config()
        self.expect_fail("holds Python objects (pickle refused)", ("check-inputs",))

    def test_c1_outside_cnn_test(self):
        outside = next(r for r in self.world.rows["official_test"] if r["key"] not in self.world.by_cnn_test())
        f = self.world.man_dir / "nuaa_c1_v2.csv"
        lines = f.read_text().splitlines()
        i = next(i for i, l in enumerate(lines[1:], 1) if l.split(",")[1] == outside["label"])   # stessa classe: i conteggi restano
        lines[i] = ",".join(outside[c] for c in SZ.MANIFEST_COLUMNS)
        f.write_text("\n".join(lines) + "\n", newline="")
        self.world.cfg["manifests"]["files"]["c1"]["sha256"] = sha(f.read_bytes())
        self.world.write_config()
        self.expect_fail("c1 is not inside cnn_test", ("check-inputs",))

    def test_official_split_differs_from_the_list(self):
        f = self.world.man_dir / "nuaa_official_train.csv"
        lines = f.read_text().splitlines()
        parts = lines[1].split(",")
        parts[7] = "test"
        lines[1] = ",".join(parts)
        f.write_text("\n".join(lines) + "\n", newline="")
        self.world.cfg["manifests"]["files"]["official_train"]["sha256"] = sha(f.read_bytes())
        self.world.write_config()
        self.expect_fail("official_split 'test' differs from the list official_train (train)", ("check-inputs",))

    def test_label_different_from_key_folder(self):
        f = self.world.man_dir / "nuaa_official_test.csv"
        lines = f.read_text().splitlines()
        i = next(i for i, l in enumerate(lines) if l.startswith("real/"))
        parts = lines[i].split(",")
        parts[1] = "1"
        lines[i] = ",".join(parts)
        f.write_text("\n".join(lines) + "\n", newline="")
        self.world.cfg["manifests"]["files"]["official_test"]["sha256"] = sha(f.read_bytes())
        self.world.write_config()
        self.expect_fail("label '1' does not match the key folder real", ("check-inputs",))

    def test_official_lists_sharing_a_file(self):
        src = self.world.rows["official_train"][0]
        dst = self.world.rows["official_test"][0]
        sp = self.world.images / SZ.LAYOUT[(src["label"], src["official_split"])] / src["key"].split("/", 1)[1]
        dp = self.world.images / SZ.LAYOUT[(dst["label"], dst["official_split"])] / dst["key"].split("/", 1)[1]
        dp.write_bytes(sp.read_bytes())
        f = self.world.man_dir / "nuaa_official_test.csv"
        lines = f.read_text().splitlines()
        parts = lines[1].split(",")
        self.assertEqual(parts[0], dst["key"])
        parts[8], parts[9] = src["sha256"], src["bytes"]
        lines[1] = ",".join(parts)
        f.write_text("\n".join(lines) + "\n", newline="")
        self.world.cfg["manifests"]["files"]["official_test"]["sha256"] = sha(f.read_bytes())
        self.world.write_config()
        self.expect_fail("official train and test lists share a file SHA-256", ("check-inputs",))

    def test_declarations_closed_set(self):
        for mutate, text in ((lambda c: c["declarations"].update(associations_not_causes="yes"), "declarations.associations_not_causes must be true"),
                             (lambda c: c["declarations"].pop("no_new_threshold"), "declarations.no_new_threshold must be true"),
                             (lambda c: c["declarations"].update(extra=3), "declarations.extra must be a text")):
            world = World(Path(tempfile.mkdtemp(prefix="stepzero-decl-")))
            try:
                mutate(world.cfg)
                world.write_config()
                rc, so, se = run_cli(["check-inputs"], world)
                self.assertEqual(rc, 1)
                self.assertIn(text, se)
            finally:
                shutil.rmtree(world.base, ignore_errors=True)

    def test_symbolic_link_input_refused(self):
        spec = self.world.cfg["face_boxes"]
        p = self.world.base / spec["path"]
        q = p.with_name("frames_real.csv")
        p.rename(q)
        os.symlink(q, p)
        self.expect_fail("is a symbolic link: inputs must be regular files")
        r = self.world.rows["official_train"][0]
        img = self.world.images / SZ.LAYOUT[(r["label"], r["official_split"])] / r["key"].split("/", 1)[1]
        img2 = img.with_name("copy.jpg")
        img.rename(img2)
        os.symlink(img2, img)
        p.unlink(); q.rename(p)
        self.expect_fail("symbolic link refused")

    def test_missing_column(self):
        spec = self.world.cfg["scores"]["n2_combined_face_crop"]
        p = self.world.base / spec["path"]
        text = p.read_text().replace("prediction", "pred")
        self.rehash(p, text, "n2_combined_face_crop")
        self.expect_fail("missing columns ['prediction']")

    def rehash(self, p: Path, text: str, score_name: str = None, box: bool = False) -> None:
        """Riscrive un file d'ingresso, il SHA256SUMS della sua cartella e ogni voce della configurazione che li cita
        (più file di score condividono la cartella o il file): così il controllo d'integrità passa e deve scattare la
        regola."""
        p.write_text(text, newline="")
        d = p.parent
        while not (d / "SHA256SUMS").exists():
            d = d.parent
        files = [l.split("  ", 1)[1] for l in (d / "SHA256SUMS").read_text().splitlines()]
        (d / "SHA256SUMS").write_text("".join(f"{sha((d / n).read_bytes())}  {n}\n" for n in files))
        rel_p, rel_s = p.relative_to(self.world.base).as_posix(), (d / "SHA256SUMS").relative_to(self.world.base).as_posix()
        for spec in list(self.world.cfg["scores"].values()) + [self.world.cfg["face_boxes"]]:
            if spec["path"] == rel_p:
                spec["sha256"] = sha(p.read_bytes())
            if spec["sums"] == rel_s:
                spec["sums_sha256"] = sha((d / "SHA256SUMS").read_bytes())
        self.world.write_config()

    def test_manifest_session_differs_from_name(self):
        f = self.world.man_dir / "nuaa_official_train.csv"                  # sessione 02 ammessa dalla lista, ma il nome dice 01
        text = f.read_text().replace(",01,0,train,", ",02,0,train,", 1)
        self.assertNotEqual(text, f.read_text())
        f.write_text(text, newline="")
        self.world.cfg["manifests"]["files"]["official_train"]["sha256"] = sha(f.read_bytes())
        self.world.write_config()
        self.expect_fail("session '02' differs from the name (01) or is outside")

    def test_non_finite_score(self):
        spec = self.world.cfg["scores"]["n1_seed43"]
        p = self.world.base / spec["path"]
        lines = p.read_text().splitlines()
        parts = lines[1].split(",")
        parts[6] = "nan"
        lines[1] = ",".join(parts)
        self.rehash(p, "\n".join(lines) + "\n", "n1_seed43")
        self.expect_fail("'nan' is not finite")

    def test_sample_outside_set_and_wrong_subject(self):
        spec = self.world.cfg["scores"]["n2_sessions_seed44"]
        p = self.world.base / spec["path"]
        lines = p.read_text().splitlines()
        parts = lines[1].split(",")
        parts[2] = self.world.rows["cnn_test"][0]["key"]
        lines[1] = ",".join(parts)
        self.rehash(p, "\n".join(lines) + "\n", "n2_sessions_seed44")
        self.expect_fail("not a key of the set target_491")
        lines = p.read_text().splitlines()
        parts = lines[1].split(",")
        parts[2], parts[3] = self.world.rows["target_491"][0]["key"], "9999"
        lines[1] = ",".join(parts)
        self.rehash(p, "\n".join(lines) + "\n", "n2_sessions_seed44")
        self.expect_fail("subject or label differ from the manifest")

    def test_inconsistent_prediction(self):
        spec = self.world.cfg["scores"]["c1_attacknet_v1__nuaa"]
        p = self.world.base / spec["path"]
        lines = p.read_text().splitlines()
        i = next(i for i, l in enumerate(lines) if l.endswith(",attacknet_v1__nuaa"))
        parts = lines[i].split(",")
        parts[11] = "1" if parts[11] == "0" else "0"
        lines[i] = ",".join(parts)
        self.rehash(p, "\n".join(lines) + "\n", "c1_attacknet_v1__nuaa")
        self.expect_fail("differs from the rule score > 0.5")

    def test_cache_label_mismatch(self):
        cache = self.world.base / self.world.cfg["cache"]["path"]
        with np.load(cache) as z:
            d = {k: z[k] for k in z}
        d["y_test"] = 1 - d["y_test"]
        np.savez_compressed(cache, **d)
        self.world.cfg["cache"]["sha256"] = sha(cache.read_bytes())
        self.world.write_config()
        self.expect_fail("label or subject differ from the manifest row")

    def test_cache_schema_mismatch(self):
        self.world.cfg["cache"]["members"]["X_test"]["shape"][1] = 9
        self.world.write_config()
        self.expect_fail("X_test has dtype", ("check-inputs",))

    def test_box_rows_mismatch(self):
        self.world.cfg["face_boxes"]["rows"] += 1
        self.world.write_config()
        self.expect_fail("face boxes: " + str(2 * len(self.world.rows["c1"])) + " rows instead of")

    def test_old_pass_removed_after_fail(self):
        run = run_ok(self.world)
        self.assertTrue((run / "public/aggregate.json").exists())
        self.world.cfg["expected"]["n_images"] += 1
        self.world.write_config()
        self.expect_fail("official lists: n_images")
        self.assertEqual(os.listdir(run / "public"), [])
        self.assertEqual(os.listdir(run / "restricted"), [])

    def test_foreign_entry_in_run_folder_refused(self):
        run = self.world.base / "out" / RUN_ID
        (run / "public").mkdir(parents=True)
        (run / "public/other.txt").write_text("x")
        rc, so, se = run_cli(["run", "--i-am-authorized"], self.world)
        self.assertEqual(rc, 1)
        self.assertIn("is not an output of this command: refused, nothing removed", se)
        self.assertEqual(os.listdir(run), ["public"])
        self.assertEqual(os.listdir(run / "public"), ["other.txt"])

    def test_protected_out_root_refused(self):
        for bad in (ROOT / "results" / "n1", ROOT / "data", ROOT):
            rc, so, se = run_cli(["run", "--i-am-authorized"], self.world, bad)
            self.assertEqual(rc, 1, bad)
            self.assertIn("refusing the run folder", se)
            self.assertFalse((bad / RUN_ID).exists())

    def test_run_folder_inside_input_package_refused(self):
        out = self.world.base / "results/n1/run-n1"
        rc, so, se = run_cli(["run", "--i-am-authorized"], self.world, out)
        self.assertEqual(rc, 1)
        self.assertIn("inputs and outputs must be separate", se)
        self.assertFalse((out / RUN_ID).exists())

    def test_config_errors(self):
        for mutate, text in ((lambda c: c["declarations"].update(no_inference=False), "declarations.no_inference must be true"),
                             (lambda c: c.update(threshold=0.4), "threshold must be 0.5"),
                             (lambda c: c["scores"]["n1_seed42"].update(set="elsewhere"), "needs set (one of"),
                             (lambda c: c["manifests"]["files"].pop("c1"), "needs exactly folder, origin and the files"),
                             (lambda c: c["cache"].update(path="/abs/cache.npz"), "must be a relative path"),
                             (lambda c: c.update(schema="x"), "schema must be"),
                             (lambda c: c["bootstrap"].update(resamples=999), "bootstrap must be"),
                             (lambda c: c["measures"].pop("all_row"), "measures must have exactly the fields"),
                             (lambda c: c["views"].update(crop=""), "views.crop must be a non-empty text"),
                             (lambda c: c.update(extra="x"), "fields must be exactly"),
                             (lambda c: c["cache"].pop("built"), "cache needs exactly"),
                             (lambda c: c["manifests"]["files"]["official_test"].update(sessions=["01", "03"]), "declared sessions of the official train and test lists overlap")):
            world = World(Path(tempfile.mkdtemp(prefix="stepzero-cfg-")))
            try:
                mutate(world.cfg)
                world.write_config()
                rc, so, se = run_cli(["check-inputs"], world)
                self.assertEqual(rc, 1)
                self.assertIn(text, se)
            finally:
                shutil.rmtree(world.base, ignore_errors=True)

    def test_inputs_untouched_by_a_run(self):
        before = self.world.snapshot()
        run_ok(self.world)
        self.assertEqual(self.world.snapshot(), before)


class TestRealInputs(unittest.TestCase):
    def test_real_check_inputs_read_only(self):
        cfg = REAL_ROOT / SZ.DEFAULT_CONFIG
        if not cfg.is_file():
            skip_missing(self, "real inputs")
        doc = json.loads(cfg.read_text())
        needed = [REAL_ROOT / doc["manifests"]["folder"], REAL_ROOT / doc["images"]["folder"], REAL_ROOT / doc["cache"]["path"],
                  REAL_ROOT / doc["face_boxes"]["path"]] + [REAL_ROOT / v["path"] for v in doc["scores"].values()]
        if not all(p.exists() for p in needed):
            skip_missing(self, "real inputs")
        so, se = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
            rc = SZ.main(["check-inputs", "--config", str(cfg), "--input-root", str(REAL_ROOT)])
        self.assertEqual(rc, 0, se.getvalue())
        self.assertIn("12614 originals", so.getvalue())


class TestSuite(unittest.TestCase):
    def test_every_skip_is_a_missing_input(self):
        src = Path(__file__).read_text(encoding="utf-8")
        self.assertEqual(src.count(".skip" + "Test("), 1)
        used = re.findall(r'skip_missing\((?:self|test), "([^"]+)"', src)
        self.assertEqual(sorted(set(used)), sorted(MISSING_INPUT))


def tree_state() -> dict:
    """File di results/, export/ e delivery/ del repository (dimensione e data di modifica): non devono cambiare."""
    out = {}
    for top in ("results", "export", "delivery"):
        for dp, dn, fn in os.walk(ROOT / top):
            for f in fn:
                p = os.path.join(dp, f)
                st = os.lstat(p)
                out[p] = (st.st_size, st.st_mtime_ns)
    return out


def main() -> int:
    before = tree_state()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c)
                               for c in (TestKnownAnswers, TestWorld, TestFailures, TestRealInputs, TestSuite))
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    changed = sorted(set(before.items()) ^ set(tree_state().items()))
    ok = result.wasSuccessful() and not changed
    if "--verbose" in sys.argv or not ok:
        print(log.getvalue())
    for p in changed[:10]:
        print(f"ERROR: repository file changed by the tests: {os.path.relpath(p[0], ROOT)}")
    for t, why in result.skipped:
        print(f"skipped {t.id().rsplit('.', 1)[-1]}: {why}")
    print(f"tests run: {result.testsRun}, failures: {len(result.failures) + (1 if changed else 0)}, errors: {len(result.errors)}, skipped: {len(result.skipped)}")
    print("RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
