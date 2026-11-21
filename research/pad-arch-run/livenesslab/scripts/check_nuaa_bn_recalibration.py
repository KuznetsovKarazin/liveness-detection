"""
Test di scripts/nuaa_bn_recalibration.py (ristima delle statistiche delle BatchNormalization dei 36 checkpoint del run
di pazienza 4), offline: nessun checkpoint vero caricato, nessun frame vero letto.

Mondo sintetico in cartelle temporanee, costruito sul mondo della suite del runner genitore
(check_nuaa_session_ablation_run.World: manifest M2, cache, run delle celle, run dei fold, frame estratti 8×8, schema
dei fold) con frame la cui classe è nella luminosità (attacco ~200, bona fide ~50, rumore ±20), più un run genitore finto
con 36 modelli Keras minuscoli che hanno gli stessi nomi e lo stesso ordine di strati di LivenessNet (ingresso 8×8×3,
due canali): "good" (separa le classi), "badstats" (stessi pesi, moving_mean sbagliata in bn_dense: dev saturato a 0,5,
recuperato dalla ristima), "bias" (bias d'uscita +50: saturato anche dopo), "inverted" (uscita invertita: fuori dal
criterio, non degenere). Schede, punteggi del dev e soglie del genitore sono calcolati con il valutatore dello script
sui file salvati; i *_test_scores.csv del genitore e i manifest del test dello schema hanno permessi 000 (una lettura
farebbe fallire il run). TensorFlow vero su CPU, modelli e dati minuscoli. Nessuna scrittura nel repository (controllo
finale su results/, export/, delivery/).

  .venv/bin/python -I -B scripts/check_nuaa_bn_recalibration.py [--verbose]
"""
import contextlib
import copy as copymod
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_nuaa_folds_run as CFR  # noqa: E402
import check_nuaa_session_ablation_run as SAR  # noqa: E402
import nuaa_bn_recalibration as RB  # noqa: E402
import nuaa_folds_run as FR  # noqa: E402
import nuaa_session_ablation_run as SR  # noqa: E402

MISSING_INPUT = {"delivered configuration": "delivery/bnrecal/nuaa_bn_recalibration.json is not present"}
STATE = {"commit": "abcdef0", "dirty": False}
SHAPE = (8, 8, 3)
BN = ["bn1_1", "bn1_2", "bn2_1", "bn2_2", "bn_dense"]
LAYERS = [["input_layer", "InputLayer"], ["conv1_1", "Conv2D"], ["relu1_1", "Activation"], ["bn1_1", "BatchNormalization"],
          ["conv1_2", "Conv2D"], ["relu1_2", "Activation"], ["bn1_2", "BatchNormalization"], ["maxpool1", "MaxPooling2D"],
          ["dropout1", "Dropout"], ["conv2_1", "Conv2D"], ["relu2_1", "Activation"], ["bn2_1", "BatchNormalization"],
          ["conv2_2", "Conv2D"], ["relu2_2", "Activation"], ["bn2_2", "BatchNormalization"], ["maxpool2", "MaxPooling2D"],
          ["dropout2", "Dropout"], ["flatten", "Flatten"], ["dense1", "Dense"], ["bn_dense", "BatchNormalization"],
          ["dropout_dense", "Dropout"], ["output", "Dense"]]
KINDS = {"arm_b_fold1_seed42": "badstats", "arm_a_star_fold2_seed43": "badstats", "arm_b_prime_fold1_seed44": "badstats",
         "arm_a_star_fold3_seed42": "bias", "arm_b_fold4_seed44": "bias", "arm_b_fold3_seed44": "inverted"}
EPOCH = {"good": 4, "badstats": 1, "bias": 2, "inverted": 3}
PARENT_RUN = "20261106-SESSABLRUN-abcdef0"
TEN = sorted(n for n, k in KINDS.items() if k in ("badstats", "bias"))
APART = ["arm_b_fold3_seed44"]
TF = RB.tf_module()


def skip_missing(test: unittest.TestCase, what: str):
    """Unico modo di saltare un test di questa suite: un ingresso mancante dichiarato in MISSING_INPUT."""
    test.skipTest(f"missing input: {MISSING_INPUT[what]}")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def bright_frame(b: bytes) -> tuple:
    """Frame 8×8×3 dai byte dell'originale: luminosità 200 (attacco) o 50 (bona fide) più rumore ±20 dai byte."""
    key = b.decode("utf-8")
    h = np.frombuffer((hashlib.sha256(b).digest() * 8)[:int(np.prod(SHAPE))], np.uint8).reshape(SHAPE)
    base = 200 if key.startswith("attack/") else 50
    arr = (base + (h.astype(np.int64) % 41) - 20).astype(np.uint8)
    return arr, ("RetinaFace" if h.flat[1] % 2 else "Haar")


# ============================================================================ modelli minuscoli

def tiny(kind: str = "good", dropout: bool = True, extra=None):
    """Modello con i nomi e l'ordine degli strati di LivenessNet (due canali, dense1 a due unità) e pesi fissati a mano:
    canale 0 = luminosità, canale 1 = 1 − luminosità, BN identità, uscita k·(u0 − u1)."""
    L = TF.keras.layers
    inp = L.Input(shape=SHAPE, name="input_layer")
    x = L.Conv2D(2, (3, 3), padding="same", name="conv1_1")(inp)
    if extra is not None:
        x = extra(x)
    x = L.Activation("relu", name="relu1_1")(x)
    x = L.BatchNormalization(name="bn1_1")(x)
    x = L.Conv2D(2, (3, 3), padding="same", name="conv1_2")(x)
    x = L.Activation("relu", name="relu1_2")(x)
    x = L.BatchNormalization(name="bn1_2")(x)
    x = L.MaxPooling2D((2, 2), name="maxpool1")(x)
    if dropout:
        x = L.Dropout(0.25, name="dropout1")(x)
    x = L.Conv2D(2, (3, 3), padding="same", name="conv2_1")(x)
    x = L.Activation("relu", name="relu2_1")(x)
    x = L.BatchNormalization(name="bn2_1")(x)
    x = L.Conv2D(2, (3, 3), padding="same", name="conv2_2")(x)
    x = L.Activation("relu", name="relu2_2")(x)
    x = L.BatchNormalization(name="bn2_2")(x)
    x = L.MaxPooling2D((2, 2), name="maxpool2")(x)
    if dropout:
        x = L.Dropout(0.25, name="dropout2")(x)
    x = L.Flatten(name="flatten")(x)
    x = L.Dense(2, activation="relu", name="dense1")(x)
    x = L.BatchNormalization(name="bn_dense")(x)
    if dropout:
        x = L.Dropout(0.5, name="dropout_dense")(x)
    out = L.Dense(2, activation="softmax", name="output")(x)
    m = TF.keras.Model(inp, out, name="LivenessNet")
    k1 = np.zeros((3, 3, 3, 2), np.float32)
    k1[1, 1, 0, 0], k1[1, 1, 0, 1] = 1.0, -1.0
    m.get_layer("conv1_1").set_weights([k1, np.array([0.0, 1.0], np.float32)])
    ident = np.zeros((3, 3, 2, 2), np.float32)
    ident[1, 1, 0, 0] = ident[1, 1, 1, 1] = 1.0
    for n in ("conv1_2", "conv2_1", "conv2_2"):
        m.get_layer(n).set_weights([ident, np.zeros(2, np.float32)])
    for n in BN:
        m.get_layer(n).set_weights([np.ones(2, np.float32), np.zeros(2, np.float32), np.zeros(2, np.float32),
                                    np.full(2, 1.0 - 0.001, np.float32)])
    w = np.zeros((8, 2), np.float32)
    w[0::2, 0], w[1::2, 1] = 0.25, 0.25            # flatten (h, w, c): canale 0 -> unità 0, canale 1 -> unità 1
    m.get_layer("dense1").set_weights([w, np.zeros(2, np.float32)])
    k = 5.0
    wo = np.array([[-k, k], [k, -k]], np.float32)
    bo = np.zeros(2, np.float32)
    if kind == "badstats":
        g, b, _, v = m.get_layer("bn_dense").get_weights()
        m.get_layer("bn_dense").set_weights([g, b, np.array([2.0, 0.0], np.float32), v])
    elif kind == "bias":
        bo = np.array([0.0, 50.0], np.float32)
    elif kind == "inverted":
        wo = -wo
    m.get_layer("output").set_weights([wo, bo])
    return m


def scrambled(seed: int):
    """Modello minuscolo con statistiche mobili, gamma e beta casuali in tutte le BN (ordine degli strati rilevante)."""
    m = tiny("good")
    rng = np.random.default_rng(seed)
    k1, _ = m.get_layer("conv1_1").get_weights()
    m.get_layer("conv1_1").set_weights([k1, np.array([-0.3, 0.6], np.float32)])    # relu1_1 taglia una parte dei valori
    for n in BN:
        m.get_layer(n).set_weights([rng.uniform(0.5, 1.5, 2).astype(np.float32), rng.uniform(-0.3, 0.3, 2).astype(np.float32),
                                    rng.uniform(-1, 1, 2).astype(np.float32), rng.uniform(0.2, 3, 2).astype(np.float32)])
    return m


def independent_eval(model, X, y) -> dict:
    """Valori del dev del genitore sintetico calcolati senza il valutatore dello script: model.predict (grafo, batch 16),
    logit dal sotto-modello fino all'ingresso di output con NumPy, cross-entropy con logsumexp, accuratezza all'argmax."""
    x = X.astype(np.float32) / 255.0
    p = np.asarray(model.predict(x, batch_size=16, verbose=0), np.float64)
    h = np.asarray(TF.keras.Model(model.input, model.get_layer("output").input).predict(x, batch_size=16, verbose=0), np.float64)
    W, b = (np.asarray(v, np.float64) for v in model.get_layer("output").get_weights())
    z = h @ W + b
    m = z.max(axis=1, keepdims=True)
    lse = (m + np.log(np.exp(z - m).sum(axis=1, keepdims=True)))[:, 0]
    vl = float(np.mean(lse - z[np.arange(len(y)), y]))
    correct = int((p.argmax(axis=1) == y).sum())
    return {"p": p, "val_loss": vl, "accuracy": float(np.float32(correct) / np.float32(len(y)))}


def frames_of(keys) -> np.ndarray:
    return np.stack([bright_frame(k.encode())[0] for k in keys])


# ============================================================================ mondo sintetico

def write_sums(folder: Path) -> str:
    texts = {p.name: (sha(p.read_bytes()) if os.access(p, os.R_OK) else "0" * 64) for p in sorted(folder.iterdir())
             if p.is_file() and p.name != "SHA256SUMS"}
    sums = "".join(f"{h}  {n}\n" for n, h in sorted(texts.items()))
    (folder / "SHA256SUMS").write_text(sums, encoding="utf-8")
    return sha(sums.encode())


class World(SAR.World):
    """Il mondo del runner genitore (frame per luminosità) più il run genitore finto, la nota e la configurazione."""

    def build(self):
        saved = CFR.fake_frame
        CFR.fake_frame = bright_frame
        try:
            super().build()
        finally:
            CFR.fake_frame = saved
        self.parent_cfg = self.config
        self.parent_cfg_sha = sha(self.parent_cfg.read_bytes())
        for i in range(1, CFR.K + 1):
            (self.base / "schema" / f"fold{i}_test.csv").chmod(0)       # mai aperti da questo runner
        for f in (self.base / "foldsrun" / "public").iterdir():
            f.chmod(0)                                                  # file pubblici del run dei fold: mai aperti
        self.models_dir = self.tmp / "models"
        self.models_dir.mkdir()
        self.kind_files = {}
        for kind in EPOCH:
            p = self.models_dir / f"{kind}.keras"
            tiny(kind).save(p)
            self.kind_files[kind] = p
        self.build_parent()
        self.note = self.tmp / "note" / "NOTE.md"
        self.note.parent.mkdir()
        self.note.write_text("# revised note (synthetic)\n", encoding="utf-8")
        self.bn_config = self.base / "bnrecal.json"
        self.write_bn_config(self.bn_doc())

    def keys(self, unit: str, fold: int) -> list:
        name = {"arm_b": "armB_fitting", "arm_a_star": "armAstar_fitting", "arm_b_prime": "armBprime_fitting", "dev": "dev"}[unit]
        return sorted(r[2] for r in self.cells[(name, fold)])

    def build_parent(self, over=None):
        """Il run genitore: per ogni modello pesi, scheda, punteggi del dev e soglia calcolati dal file salvato; over:
        nome -> funzione(scheda) per alterare una scheda; anchor in status.json e nella configurazione."""
        over = over or {}
        run = self.base / "parentrun"
        if run.exists():
            for p in run.rglob("*"):
                if p.is_file():
                    p.chmod(0o644)
            shutil.rmtree(run)
        (run / "restricted").mkdir(parents=True)
        (run / "public").mkdir()
        self.parent = run
        order = [SR.model_name(*x) for x in SR.training_order(CFR.K, SAR.SEEDS)]
        cache = self.__dict__.setdefault("ev_cache", {})       # valutazioni per (tipo, fold), riusate fra le ricostruzioni
        training = {}
        for arm, fold, seed in SR.training_order(CFR.K, SAR.SEEDS):
            name = SR.model_name(arm, fold, seed)
            kind = KINDS.get(name, "good")
            wp = run / "restricted" / f"{name}_LivenessNet.keras"
            shutil.copyfile(self.kind_files[kind], wp)
            w = sha(wp.read_bytes())
            dkeys, fkeys = self.keys("dev", fold), self.keys(arm, fold)
            if (kind, fold) not in cache:
                model = TF.keras.models.load_model(str(self.kind_files[kind]), compile=False)
                y = np.array([1 if k.startswith("attack/") else 0 for k in dkeys])
                cache[(kind, fold)] = independent_eval(model, frames_of(dkeys), y)
            ev = cache[(kind, fold)]
            e = EPOCH[kind]
            vl32 = float(np.float32(ev["val_loss"]))
            hist_vl = [vl32 + 0.5 * (i + 1) for i in range(e - 1)] + [vl32] + [vl32 + 0.25, vl32 + 0.3, vl32 + 0.35, vl32 + 0.4]
            hist_va = [0.25] * (e - 1) + [ev["accuracy"]] + [0.5] * 4
            card = {"arch": "LivenessNet", "train_seed": seed, "n_fitting": len(fkeys), "n_dev": len(dkeys), "max_epochs": 12,
                    "epochs_run": len(hist_vl), "restored_epoch": e, "restored_by": "early stopping", "best_val_loss": vl32,
                    "val_acc_at_best": ev["accuracy"], "history": {"loss": [0.1] * len(hist_vl), "accuracy": [0.9] * len(hist_vl),
                                                                    "val_loss": hist_vl, "val_accuracy": hist_va, "lr": [0.001] * len(hist_vl)},
                    "device": "CPU", "weights_file": wp.name, "weights_sha256": w, "arm": arm, "fold": fold,
                    "fitting_keys_sha256": sha("\n".join(fkeys).encode()), "dev_keys_sha256": sha("\n".join(dkeys).encode()),
                    "val_loss_recomputed": ev["val_loss"]}
            if name in over:
                over[name](card)
            thr = {"arm": arm, "fold": fold, "train_seed": seed, "value": 0.5000001, "checkpoint_sha256": w}
            tb = json.dumps(thr, indent=1) + "\n"
            (run / "restricted" / f"{name}_threshold.json").write_text(tb, encoding="utf-8")
            part = {"rows": [{"key": k, "subject": self.official[k]["subject"], "session": self.official[k]["session"],
                              "label": self.official[k]["label"], "sha256": self.official[k]["sha256"]} for k in dkeys]}
            rows = SR.score_rows(arm, fold, seed, "dev", part, ev["p"][:, 1], {"value": thr["value"], "sha256": sha(tb.encode())}, card)
            (run / "restricted" / f"{name}_dev_scores.csv").write_text(SR.rows_text(rows), encoding="utf-8")
            (run / "restricted" / f"{name}_card.json").write_text(json.dumps(card, indent=1) + "\n", encoding="utf-8")
            tp = run / "restricted" / f"{name}_test_scores.csv"
            tp.write_text("never read\n", encoding="utf-8")
            tp.chmod(0)
            training[name] = {"weights_sha256": card["weights_sha256"], "restored_epoch": card["restored_epoch"],
                              "best_val_loss": card["best_val_loss"], "n_fitting": card["n_fitting"], "n_dev": card["n_dev"],
                              "dev_threshold": thr["value"]}
        # non JSON di proposito: lo script deve solo calcolarne lo SHA-256, mai decodificarlo
        (run / "public" / "aggregate.json").write_bytes(b"{not JSON: the test metrics of the parent run would be here\n")
        self.parent_pub = write_sums(run / "public")
        self.parent_res = write_sums(run / "restricted")
        self.parent_status = {"run_id": PARENT_RUN, "status": "PASS", "config_sha256": self.parent_cfg_sha, "completed": order,
                              "missing": [], "public_sums_sha256": self.parent_pub, "restricted_sums_sha256": self.parent_res}
        (run / "status.json").write_text(json.dumps(self.parent_status, indent=1) + "\n", encoding="utf-8")
        if hasattr(self, "bn_config"):
            self.write_bn_config(self.bn_doc())

    def bn_doc(self) -> dict:
        n = len(SR.training_order(CFR.K, SAR.SEEDS))
        return {
            "schema": RB.CONFIG_SCHEMA, "task": "synthetic", "authorisation": "test",
            "parent_run": {"run_id": PARENT_RUN, "folder": "parentrun", "config_path": self.parent_cfg.name, "config_sha256": self.parent_cfg_sha,
                           "public_sha256sums_sha256": self.parent_pub, "restricted_sha256sums_sha256": self.parent_res, "models": n,
                           "used": "t"},
            "architecture": {"name": "LivenessNet", "path": "ref/architectures.py", "sha256": self.code["ref/architectures.py"],
                             "input_shape": list(SHAPE), "layers": [list(x) for x in LAYERS], "augmentation": "none"},
            "batchnorm": {"layers": list(BN), "kinds": {b: ("dense" if b == "bn_dense" else "conv") for b in BN}, "momentum": 0.99,
                          "epsilon": 0.001, "inference": "t", "running_variance_in_training": "t"},
            "estimator": {"statistics": "t", "order": "t", "mode": "t", "counts": "t", "variance": "t", "cast": "t",
                          "changed_tensors": ["moving_mean", "moving_variance"], "copy": "t", "consistency": "t"},
            "evaluation": {"device": "CPU", "batch": 16, "input": "t", "mode": "t", "probability": "t", "logits": "t", "val_loss": "t",
                           "accuracy": "t", "auroc": "t", "decision": "t"},
            "tolerances": {"dev_probability_max_abs_diff": {"value": 1e-05, "source": "t"},
                           "dev_decisions_discordant_at_0_5": {"value": 0, "source": "t"},
                           "val_loss_abs": {"value": 0.001, "source": "t"},
                           "dev_accuracy_abs": {"value": 1e-06, "source": "t"},
                           "softmax_of_logits_max_abs_diff": {"value": 0.0001, "source": "t"},
                           "consistency_rel": {"value": 0.0001, "source": "t"}, "consistency_abs": {"value": 1e-06, "source": "t"}},
            "stop_rule": "stop with BASELINE_FAILED",
            "recovery_criterion": {"accuracy_min": 0.75, "accuracy_min_inclusive": True, "val_loss_max": 1.0, "val_loss_max_inclusive": False,
                                   "text": "t", "source": "t"},
            "groups": {"degenerate": {"restored_epochs": [1, 2], "dev_accuracy": 0.5, "within": 1e-06, "source": "t"},
                       "ten": {"definition": "t", "expected": list(TEN)}, "apart": {"definition": "t", "expected": list(APART)},
                       "control": {"definition": "t", "expected_count": n - len(TEN) - len(APART)}},
            "reading_rule": {"primary": "p", "all": "all {k} of {n}", "none": "none {k} of {n}", "some": "some {k} of {n}", "controls": "c",
                             "apart": "a", "cells": {c: f"text {c}" for c in RB.CELLS}, "cells_header": "h", "worsening": "w", "limits": "l", "report": "r"},
            "outputs": {"public": list(RB.PUBLIC_OUTPUTS), "restricted": [f"<model>_{x}" for x in RB.PER_MODEL] + list(RB.RESTRICTED_FIXED)},
            "budget": {"wall_time_cap_minutes": 240, "estimate_minutes": 7, "upper_estimate_minutes": 9, "estimate": "t", "on_cap_exceeded": "INCOMPLETE", "voluntary_stop": "SIGUSR1: INCOMPLETE"},
            "environment": {"tensorflow": FR.environment()["tensorflow_installed"]},
            "note_path": str(self.note), "note_sha256": sha(self.note.read_bytes()), "test": RB.TEST_READ}

    def write_bn_config(self, doc: dict) -> Path:
        self.bn_config.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        return self.bn_config

    def argv(self, cmd: str, *extra, out=None, date="20261114") -> list:
        a = [cmd, "--config", str(self.bn_config), "--input-root", str(self.base), "--out-root", str(out or self.out), "--date", date]
        return a + (["--i-am-authorized"] if cmd == "run" else []) + list(extra)

    def close(self):
        for p in self.tmp.rglob("*"):
            with contextlib.suppress(OSError):
                if not p.is_symlink():
                    p.chmod(0o755 if p.is_dir() else 0o644)
        super().close()


@contextlib.contextmanager
def injected(rb=None, sr=None, fr=None):
    """Punti di iniezione (RB, SR, FR), ripristinati all'uscita."""
    saved = [(mod, k, getattr(mod, k)) for mod, d in ((RB, rb), (SR, sr), (FR, fr)) for k in (d or {})]
    try:
        for mod, d in ((RB, rb), (SR, sr), (FR, fr)):
            for k, v in (d or {}).items():
                setattr(mod, k, v)
        yield
    finally:
        for mod, k, v in saved:
            setattr(mod, k, v)


def hooks(world: World, rb=None, sr=None, fr=None) -> dict:
    return {"rb": {"REPO_STATE": lambda root: dict(STATE), **(rb or {})},
            "sr": {"SCHEMA_CONFIG_SHA256": world.schema_sha, "CELLS_CONFIG_SHA256": world.cells_cfg_sha, **(sr or {})},
            "fr": {"FRAME_SHAPE": SHAPE, **(fr or {})}}


def run_main(world: World, argv, **kw) -> tuple:
    out, err = io.StringIO(), io.StringIO()
    with injected(**hooks(world, **kw)), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = RB.main(argv)
    return rc, out.getvalue(), err.getvalue()


def read_csv(p: Path) -> list:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_folder(out: Path, date="20261114") -> Path:
    return out / f"{date}-BNRECAL-{STATE['commit']}"


def tree_state(folder: Path) -> dict:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns, sha(p.read_bytes()) if os.access(p, os.R_OK) else "unreadable")
            for p in sorted(folder.rglob("*")) if p.is_file()}


WORLD = None
PASS = {}


def world() -> World:
    global WORLD
    if WORLD is None:
        WORLD = World()
    return WORLD


def pass_run() -> dict:
    """Il run PASS condiviso (una volta per suite): codice, uscite, cartella, stato del genitore prima e dopo."""
    if not PASS:
        w = world()
        before = tree_state(w.parent)
        rc, out, err = run_main(w, w.argv("run"))
        PASS.update({"rc": rc, "out": out, "err": err, "run": run_folder(w.out), "before": before, "after": tree_state(w.parent)})
    return PASS


# ============================================================================ momenti

def batches(x: np.ndarray, b: int):
    return [x[i:i + b] for i in range(0, len(x), b)]


class TestMoments(unittest.TestCase):
    def data(self, conv: bool):
        """Valori con spostamenti di media fra i batch (frame ordinati per classe) e un ultimo batch corto."""
        rng = np.random.default_rng(1)
        n = 37                                            # 2 x 16 + 5
        shift = np.where(np.arange(n) < 32, 0.0, 3.0)        # l'ultimo batch (5 frame) ha un'altra media
        if conv:
            x = rng.normal(0, 1, (n, 4, 3, 2)) * np.array([1.0, 2.0]) + shift[:, None, None, None] + np.array([0.5, -1.0])
        else:
            x = rng.normal(0, 1, (n, 3)) * np.array([1.0, 0.5, 2.0]) + shift[:, None]
        return x.astype(np.float32)

    def accumulate(self, x, b):
        m = RB.Moments()
        for xb in batches(x, b):
            m.add(xb)
        return m

    def test_reference_equality_conv_and_dense(self):
        for conv in (True, False):
            x = self.data(conv)
            m = self.accumulate(x, 16)
            mean, var = m.result()
            flat = x.astype(np.float64).reshape(-1, x.shape[-1])
            np.testing.assert_allclose(mean, flat.mean(axis=0), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(var, flat.var(axis=0), rtol=1e-12, atol=1e-12)            # divisore N
            np.testing.assert_allclose(var, (flat ** 2).mean(axis=0) - flat.mean(axis=0) ** 2, rtol=1e-9)
            self.assertEqual(m.n, len(x) * (12 if conv else 1))                                     # frame x H x W per conv

    def test_population_not_sample_variance(self):
        x = self.data(False)
        _, var = self.accumulate(x, 16).result()
        flat = x.astype(np.float64)
        self.assertGreater(np.max(np.abs(var - flat.var(axis=0, ddof=1))), 1e-3)

    def test_batch_size_invariance(self):
        for conv in (True, False):
            x = self.data(conv)
            ref = self.accumulate(x, len(x)).result()
            for b in (16, 7, 1):
                got = self.accumulate(x, b).result()
                np.testing.assert_allclose(got[0], ref[0], rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(got[1], ref[1], rtol=1e-12, atol=1e-12)

    def test_naive_estimators_differ(self):
        """La media delle varianze dei batch e la media non pesata delle medie dei batch danno numeri diversi (ultimo
        batch corto, medie diverse fra batch): l'implementazione dà quelli globali."""
        for conv in (True, False):
            x = self.data(conv)
            bs = [xb.astype(np.float64).reshape(-1, x.shape[-1]) for xb in batches(x, 16)]
            self.assertEqual([len(xb) for xb in batches(x, 16)], [16, 16, 5])
            naive_var = np.mean([b.var(axis=0) for b in bs], axis=0)
            naive_mean = np.mean([b.mean(axis=0) for b in bs], axis=0)
            mean, var = self.accumulate(x, 16).result()
            self.assertGreater(np.min(np.abs(var - naive_var)), 0.3)
            self.assertGreater(np.min(np.abs(mean - naive_mean)), 0.3)
            flat = x.astype(np.float64).reshape(-1, x.shape[-1])
            np.testing.assert_allclose(var, flat.var(axis=0), rtol=1e-12)

    def test_short_last_batch_weight(self):
        """Un ultimo batch di un frame molto diverso: pesato per il suo conteggio, non come un batch intero."""
        x = np.concatenate([np.zeros((32, 2)), np.full((1, 2), 33.0)]).astype(np.float32)
        mean, var = self.accumulate(x, 16).result()
        np.testing.assert_allclose(mean, [1.0, 1.0], rtol=1e-12)
        np.testing.assert_allclose(var, [32.0, 32.0], rtol=1e-12)

    def test_empty(self):
        with self.assertRaisesRegex(FR.Failure, "moments of an empty set"):
            RB.Moments().result()
        m = RB.Moments()
        m.add(np.zeros((0, 3)))
        self.assertEqual(m.n, 0)


# ============================================================================ strati, tensori, inferenza

def part_of(keys) -> dict:
    X = frames_of(keys)
    return {"X": X, "y": np.array([1 if k.startswith("attack/") else 0 for k in keys]), "keys": list(keys)}


def fitting_part(n: int = 22) -> dict:
    w = world()
    keys = sorted(w.official)[::max(1, len(w.official) // n)][:n]
    return part_of(keys)


def numpy_layer_input(model, X, layer: str) -> np.ndarray:
    """Ingresso di uno strato su tutti i frame in un solo batch (riferimento indipendente dal codice dello script)."""
    sub = TF.keras.Model(model.input, model.get_layer(layer).input)
    return np.asarray(sub(X.astype(np.float32) / 255.0, training=False), np.float64)


class TestLayers(unittest.TestCase):
    def test_layer_by_layer_with_earlier_layers_re_estimated(self):
        m = scrambled(3)
        part = fitting_part(22)                                          # 16 + 6: ultimo batch corto
        stats = RB.recalibrate(m, part, BN, 16)
        for n in BN:                                                     # riferimento: momenti del modello con gli strati precedenti ristimati
            a = numpy_layer_input(m, part["X"], n).reshape(-1, 2)
            np.testing.assert_allclose(stats[n]["mean64"], a.mean(axis=0), rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(stats[n]["var64"], a.var(axis=0), rtol=1e-4, atol=1e-6)
        one_pass = RB.forward(scrambled(3), part["X"], 16, layers=BN, scores=False)["moments"]
        diff = {n: float(np.max(np.abs(one_pass[n].result()[1] - stats[n]["var64"]))) for n in BN}
        self.assertLess(diff["bn1_1"], 1e-9)                             # il primo strato non dipende da altre BN
        self.assertGreater(max(diff[n] for n in BN[1:]), 1e-3)           # gli altri sì: un solo passaggio dal modello originale è diverso

    def test_counts_conv_and_dense(self):
        m = scrambled(4)
        part = fitting_part(22)
        stats = RB.recalibrate(m, part, BN, 16)
        self.assertEqual({n: (stats[n]["count"], stats[n]["spatial"]) for n in BN},
                         {"bn1_1": (22 * 64, 64), "bn1_2": (22 * 64, 64), "bn2_1": (22 * 16, 16), "bn2_2": (22 * 16, 16), "bn_dense": (22, 1)})

    def test_only_the_ten_moving_tensors_change(self):
        orig = scrambled(5)
        cp = TF.keras.models.clone_model(orig)
        cp.set_weights(orig.get_weights())
        snap = RB.snapshot_weights(orig)
        self.assertEqual(RB.snapshot_weights(cp), snap)
        stats = RB.recalibrate(cp, fitting_part(22), BN, 16)
        after = RB.snapshot_weights(cp)
        moving = RB.moving_keys(cp, BN)
        self.assertEqual(len(moving), 10)
        self.assertEqual(RB.tensor_problems(snap, after, moving, stats), [])
        differ = sorted(k for k in snap if snap[k]["bytes"] != after[k]["bytes"])
        self.assertEqual(differ, sorted(moving))
        self.assertEqual(sorted(after[k]["name"] for k in differ), sorted(["moving_mean", "moving_variance"] * 5))
        self.assertEqual(RB.snapshot_weights(orig), snap)                # l'originale non cambia
        for n in BN:
            self.assertEqual(np.asarray(cp.get_layer(n).moving_mean.numpy()).tobytes(), stats[n]["mean64"].astype(np.float32).tobytes())

    def test_tensor_problems_detects_every_change(self):
        orig = scrambled(6)
        cp = TF.keras.models.clone_model(orig)
        cp.set_weights(orig.get_weights())
        snap = RB.snapshot_weights(orig)
        stats = RB.recalibrate(cp, fitting_part(22), BN, 16)
        moving = RB.moving_keys(cp, BN)
        g = cp.get_layer("bn2_1")
        g.gamma.assign(g.gamma.numpy() + np.float32(1e-3))
        self.assertIn("trained tensors changed: bn2_1/gamma", "; ".join(RB.tensor_problems(snap, RB.snapshot_weights(cp), moving, stats)))
        g.gamma.assign(orig.get_layer("bn2_1").gamma.numpy())
        k = cp.get_layer("conv1_1")
        k.kernel.assign(k.kernel.numpy() * np.float32(1.0000001) + np.float32(1e-9))
        self.assertIn("conv1_1/kernel", "; ".join(RB.tensor_problems(snap, RB.snapshot_weights(cp), moving, stats)))
        k.kernel.assign(orig.get_layer("conv1_1").kernel.numpy())
        self.assertEqual(RB.tensor_problems(snap, RB.snapshot_weights(cp), moving, stats), [])
        cp.get_layer("bn_dense").moving_variance.assign(orig.get_layer("bn_dense").moving_variance.numpy())
        probs = "; ".join(RB.tensor_problems(snap, RB.snapshot_weights(cp), moving, stats))
        self.assertIn("bn_dense/moving_variance: the copy does not hold the installed float32 statistic", probs)
        self.assertIn("moving tensors equal to the original after the recalibration: bn_dense/moving_variance", probs)
        wrong = {**stats, "bn1_1": {**stats["bn1_1"], "mean32": stats["bn1_1"]["mean32"] + np.float32(1)}}
        cp.get_layer("bn_dense").moving_variance.assign(stats["bn_dense"]["var32"])
        self.assertIn("bn1_1/moving_mean: the copy does not hold", "; ".join(RB.tensor_problems(snap, RB.snapshot_weights(cp), moving, wrong)))
        fewer = {k: v for k, v in snap.items() if k[0] != "dense1"}
        self.assertEqual(RB.tensor_problems(fewer, RB.snapshot_weights(cp), moving, stats),
                         ["the copy does not have the same weight tensors as the original"])

    def test_inference_mode_dropout_inactive(self):
        part = fitting_part(22)
        a = RB.forward(scrambled(7), part["X"], 16, layers=BN)
        nod = tiny("good", dropout=False)
        nod.set_weights(scrambled(7).get_weights())
        b = RB.forward(nod, part["X"], 16, layers=BN)
        for n in BN:
            self.assertEqual([x.tobytes() for x in a["moments"][n].result()], [x.tobytes() for x in b["moments"][n].result()])
        self.assertEqual(a["p"].tobytes(), b["p"].tobytes())
        m = scrambled(7)                                                  # in training il dropout cambierebbe gli ingressi
        sub = TF.keras.Model(m.input, m.get_layer("bn2_1").input)
        x = part["X"][:16].astype(np.float32) / 255.0
        self.assertFalse(np.array_equal(np.asarray(sub(x, training=True)), np.asarray(sub(x, training=False))))

    def test_logits_and_probabilities(self):
        part = fitting_part(22)
        m = tiny("good")
        rec, _ = RB.evaluate(m, part, 16, "t")
        ref = np.asarray(m(part["X"].astype(np.float32) / 255.0, training=False), np.float64)
        np.testing.assert_allclose(rec["p"], ref, atol=1e-6)
        self.assertLess(rec["softmax_gap"], 1e-6)
        self.assertEqual(rec["correct"], int((ref.argmax(1) == part["y"]).sum()))
        self.assertEqual(rec["accuracy"], float(np.float32(rec["correct"]) / np.float32(len(part["y"]))))
        self.assertAlmostEqual(rec["val_loss"], SR.val_loss_logits(rec["z"], part["y"]), places=12)

    def test_augmentation_or_extra_layer_refused(self):
        cfg = world().bn_doc()
        RB.check_architecture(tiny("good"), cfg, "m")
        for extra in (lambda x: TF.keras.layers.RandomFlip("horizontal", name="aug")(x),
                      lambda x: TF.keras.layers.Rescaling(1.0, name="rescale")(x)):
            with self.assertRaisesRegex(FR.Failure, "augmentation or preprocessing layers are refused"):
                RB.check_architecture(tiny("good", extra=extra), cfg, "m")
        with self.assertRaisesRegex(FR.Failure, "differ from the configuration"):
            RB.check_architecture(tiny("good", dropout=False), cfg, "m")
        bad = copymod.deepcopy(cfg)
        bad["batchnorm"]["epsilon"] = 0.01
        with self.assertRaisesRegex(FR.Failure, "differs from the recorded facts"):
            RB.check_architecture(tiny("good"), bad, "m")
        bad = copymod.deepcopy(cfg)
        bad["batchnorm"]["momentum"] = 0.9
        with self.assertRaisesRegex(FR.Failure, "differs from the recorded facts"):
            RB.check_architecture(tiny("good"), bad, "m")
        bad = copymod.deepcopy(cfg)
        bad["batchnorm"]["kinds"]["bn2_1"] = "dense"
        with self.assertRaisesRegex(FR.Failure, "differs from the recorded facts"):
            RB.check_architecture(tiny("good"), bad, "m")
        bad = copymod.deepcopy(cfg)
        bad["architecture"]["input_shape"] = [16, 16, 3]
        with self.assertRaisesRegex(FR.Failure, "input shape"):
            RB.check_architecture(tiny("good"), bad, "m")

    def test_consistency_check(self):
        m = scrambled(8)
        part = fitting_part(22)
        stats = RB.recalibrate(m, part, BN, 16)
        _, final = RB.evaluate(m, part, 16, "t", layers=BN)
        cons = RB.consistency(stats, final, 1e-4, 1e-6)
        self.assertEqual({n: c["channels_outside_tolerance"] for n, c in cons.items()}, {n: 0 for n in BN})
        self.assertLess(max(c["max_abs_diff_var"] for c in cons.values()), 1e-5)
        # una statistica sbagliata installata (tutti gli strati da un solo passaggio del modello originale)
        m2 = scrambled(8)
        one = RB.forward(scrambled(8), part["X"], 16, layers=BN, scores=False)["moments"]
        wrong = {}
        for n in BN:
            mean, var = one[n].result()
            m2.get_layer(n).moving_mean.assign(mean.astype(np.float32))
            m2.get_layer(n).moving_variance.assign(var.astype(np.float32))
            wrong[n] = {"mean32": mean.astype(np.float32), "var32": var.astype(np.float32)}
        _, final2 = RB.evaluate(m2, part, 16, "t", layers=BN)
        cons2 = RB.consistency(wrong, final2, 1e-4, 1e-6)
        self.assertEqual(cons2["bn1_1"]["channels_outside_tolerance"], 0)
        self.assertGreater(sum(c["channels_outside_tolerance"] for c in cons2.values()), 0)
        # tolleranze lette dal chiamante: una tolleranza larghissima accetta anche lo sbaglio
        self.assertEqual(sum(c["channels_outside_tolerance"] for c in RB.consistency(wrong, final2, 1e9, 1e9).values()), 0)
        shifted = {n: {**s, "var32": s["var32"] * np.float32(1.001)} for n, s in stats.items()}
        self.assertGreater(sum(c["channels_outside_tolerance"] for c in RB.consistency(shifted, final, 1e-4, 1e-6).values()), 0)
        self.assertEqual(sum(c["channels_outside_tolerance"] for c in RB.consistency(shifted, final, 1e-2, 1e-6).values()), 0)
        nan = {n: {**s, "var32": np.full_like(s["var32"], np.nan)} for n, s in stats.items()}
        self.assertEqual(sum(c["channels_outside_tolerance"] for c in RB.consistency(nan, final, 1e9, 1e9).values()), 2 * len(BN))
        m3 = scrambled(8)
        k1, b1 = m3.get_layer("conv1_1").get_weights()
        k1[0, 0, 0, 0] = np.nan
        m3.get_layer("conv1_1").set_weights([k1, b1])
        with self.assertRaisesRegex(FR.Failure, "bn1_1: the re-estimated mean or variance is not finite"):
            RB.recalibrate(m3, part, BN, 16, name="m3")
        absolute = {n: {**s, "mean32": s["mean32"] + np.float32(5e-6)} for n, s in stats.items()}
        self.assertGreater(sum(c["channels_outside_tolerance"] for c in RB.consistency(absolute, final, 0.0, 1e-6).values()), 0)
        self.assertEqual(sum(c["channels_outside_tolerance"] for c in RB.consistency(absolute, final, 0.0, 1e-5).values()), 0)

    def test_change_summary(self):
        s = {"old_mean": np.array([0.0, 1.0], np.float32), "old_var": np.array([1.0, 4.0], np.float32),
             "mean32": np.array([1.0, 1.0], np.float32), "var32": np.array([2.0, 1.0], np.float32)}
        c = RB.change_summary(s, 0.0)
        self.assertEqual((c["var_ratio_min"], c["var_ratio_median"], c["var_ratio_max"]), (0.25, 1.125, 2.0))
        self.assertEqual((c["mean_shift_min"], c["mean_shift_median"], c["mean_shift_max"]), (0.0, 0.5, 1.0))


# ============================================================================ baseline, criterio, gruppi, celle

def ev(p1, correct=None, n=None, vl=0.5, gap=0.0):
    p1 = np.asarray(p1, float)
    n = n or len(p1)
    correct = n if correct is None else correct
    return {"p": np.stack([1 - p1, p1], 1), "val_loss": vl, "softmax_gap": gap, "correct": correct,
            "accuracy": float(np.float32(correct) / np.float32(n)), "accuracy_05": 0.5}


CARD = {"best_val_loss": 0.5, "val_loss_recomputed": 0.5000004, "val_acc_at_best": 1.0, "n_dev": 4}


class TestBaseline(unittest.TestCase):
    def setUp(self):
        self.cfg = world().bn_doc()
        self.s = np.array([0.1, 0.2, 0.8, 0.9])

    def check(self, p_parent=None, e=None, card=None, cfg=None):
        return RB.baseline_checks(cfg or self.cfg, self.s if p_parent is None else p_parent, e or ev(self.s), self.s, card or CARD)

    def test_pass(self):
        c = self.check()
        self.assertTrue(c["baseline_pass"])
        self.assertAlmostEqual(c["val_loss_diff_recomputed"], 4e-7)
        self.assertEqual((c["correct"], c["correct_card"]), (4, 4))

    def test_probability_each_scorer(self):
        for kw in ({"p_parent": self.s + 2e-5}, {"e": ev(self.s + 2e-5)}):
            c = self.check(**kw)
            self.assertFalse(c["prob_pass"])
            self.assertFalse(c["baseline_pass"])
        self.assertTrue(self.check(p_parent=self.s + 9e-6)["prob_pass"])

    def test_discordant_decision(self):
        s = np.array([0.1, 0.5, 0.5000004, 0.9])
        c = RB.baseline_checks(self.cfg, np.array([0.1, 0.5000002, 0.5000004, 0.9]), ev(s), s, CARD)
        self.assertEqual((c["prob_parent_discordant_05"], c["prob_eval_discordant_05"]), (1, 0))
        self.assertLess(c["prob_parent_max_abs_diff"], 1e-5)
        self.assertFalse(c["prob_pass"])
        loose = copymod.deepcopy(self.cfg)
        loose["tolerances"]["dev_decisions_discordant_at_0_5"]["value"] = 1
        self.assertTrue(RB.baseline_checks(loose, np.array([0.1, 0.5000002, 0.5000004, 0.9]), ev(s), s, CARD)["prob_pass"])

    def test_val_loss_absolute_only(self):
        self.assertFalse(self.check(e=ev(self.s, vl=0.5 + 0.004))["val_loss_pass"])        # entro l'1 %, oltre 1e-3: rifiutato
        self.assertTrue(self.check(e=ev(self.s, vl=0.5 + 0.0009))["val_loss_pass"])
        big = {**CARD, "best_val_loss": 60.9868}
        self.assertFalse(self.check(e=ev(self.s, vl=61.5), card=big)["val_loss_pass"])     # nessun ramo relativo (0,61)
        self.assertFalse(self.check(e=ev(self.s, vl=60.9880), card=big)["val_loss_pass"])
        self.assertTrue(self.check(e=ev(self.s, vl=60.9877), card=big)["val_loss_pass"])
        self.assertEqual(self.check()["val_loss_tolerance"], "abs 0.001")
        tiny_ = {**CARD, "best_val_loss": 0.01}
        self.assertTrue(self.check(e=ev(self.s, vl=0.0109), card=tiny_)["val_loss_pass"])  # assoluto
        self.assertFalse(self.check(e=ev(self.s, vl=0.0111), card=tiny_)["val_loss_pass"])
        self.assertFalse(self.check(e=ev(self.s, vl=0.5 + 0.006))["baseline_pass"])

    def test_accuracy_one_frame(self):
        c = self.check(e=ev(self.s, correct=3))
        self.assertFalse(c["accuracy_pass"])
        self.assertFalse(c["baseline_pass"])
        self.assertEqual((c["correct"], c["correct_card"]), (3, 4))
        self.assertTrue(self.check(card={**CARD, "val_acc_at_best": 1.0 - 5e-7})["accuracy_pass"])

    def test_softmax_gap(self):
        c = self.check(e=ev(self.s, gap=2e-4))
        self.assertFalse(c["softmax_pass"])
        self.assertFalse(c["baseline_pass"])

    def test_tolerances_read_from_the_configuration(self):
        cfg = copymod.deepcopy(self.cfg)
        cfg["tolerances"]["dev_probability_max_abs_diff"]["value"] = 1e-8
        self.assertFalse(self.check(p_parent=self.s + 1e-7, cfg=cfg)["prob_pass"])
        cfg = copymod.deepcopy(self.cfg)
        cfg["tolerances"]["val_loss_abs"]["value"] = 1e-9
        self.assertFalse(self.check(e=ev(self.s, vl=0.5 + 1e-6), cfg=cfg)["val_loss_pass"])
        cfg = copymod.deepcopy(self.cfg)
        cfg["tolerances"]["dev_accuracy_abs"]["value"] = 0.3
        self.assertTrue(self.check(e=ev(self.s, correct=3), cfg=cfg)["accuracy_pass"])
        cfg = copymod.deepcopy(self.cfg)
        cfg["tolerances"]["softmax_of_logits_max_abs_diff"]["value"] = 1e-3
        self.assertTrue(self.check(e=ev(self.s, gap=2e-4), cfg=cfg)["softmax_pass"])

    def test_baseline_columns_complete(self):
        c = self.check()
        self.assertEqual(sorted(set(RB.BASELINE_COLUMNS) - {"arm", "fold", "seed", "n_dev", "fitting_accuracy", "fitting_loss",
                                                           "fitting_reference"}), sorted(c))


def rec(acc, vl, auroc=0.9, n=400):
    return {"accuracy": acc, "val_loss": vl, "auroc": auroc, "correct": int(round(acc * n)), "n": n}


class TestReading(unittest.TestCase):
    def setUp(self):
        self.cfg = world().bn_doc()

    def test_criterion_boundaries(self):
        rc = self.cfg["recovery_criterion"]
        self.assertTrue(RB.meets(0.75, 0.999999, rc))
        self.assertFalse(RB.meets(0.7499999, 0.1, rc))
        self.assertFalse(RB.meets(0.9, 1.0, rc))
        self.assertTrue(RB.meets(0.9, 0.9999999, rc))
        self.assertFalse(RB.meets(0.9, 1.0000001, rc))
        flipped = {**rc, "accuracy_min_inclusive": False, "val_loss_max_inclusive": True}
        self.assertFalse(RB.meets(0.75, 0.5, flipped))
        self.assertTrue(RB.meets(0.9, 1.0, flipped))
        other = {**rc, "accuracy_min": 0.8, "val_loss_max": 0.5}
        self.assertFalse(RB.meets(0.79, 0.4, other))
        self.assertFalse(RB.meets(0.81, 0.6, other))

    def test_groups(self):
        def card(e, acc, vl):
            return {"restored_epoch": e, "val_acc_at_best": acc, "best_val_loss": vl}
        self.assertEqual(RB.group_of(card(1, 0.5, 7.0), self.cfg), "ten")
        self.assertEqual(RB.group_of(card(2, 0.5000009, 2.0), self.cfg), "ten")
        self.assertEqual(RB.group_of(card(2, 0.500002, 2.0), self.cfg), "apart")
        self.assertEqual(RB.group_of(card(3, 0.5, 2.0), self.cfg), "apart")              # epoca 3 a 0,5: non degenere
        self.assertEqual(RB.group_of(card(3, 0.2375, 3.27), self.cfg), "apart")
        self.assertEqual(RB.group_of(card(4, 0.75, 0.99), self.cfg), "control")
        self.assertEqual(RB.group_of(card(4, 0.9, 1.0), self.cfg), "apart")
        self.assertEqual(RB.group_of(card(1, 0.9, 0.1), self.cfg), "control")
        cfg = copymod.deepcopy(self.cfg)
        cfg["groups"]["degenerate"]["restored_epochs"] = [1, 2, 3]
        self.assertEqual(RB.group_of(card(3, 0.5, 2.0), cfg), "ten")

    def row(self, db, da, fb, fa, group="ten"):
        changes = {n: {"var_ratio_min": 1.0, "var_ratio_median": 1.0, "var_ratio_max": 1.0, "mean_shift_min": 0.0, "mean_shift_median": 0.0,
                       "mean_shift_max": 0.0} for n in BN}
        m = {"arm": "arm_b", "fold": 1, "seed": 42, "card": {"restored_epoch": 1}}
        return RB.result_row(self.cfg, "arm_b_fold1_seed42", m, group, {"dev": db, "fit": fb}, {"dev": da, "fit": fa}, changes,
                             {"baseline": 1.0, "recalibration": 2.0, "evaluation": 3.0})

    def test_cells_and_flags(self):
        r = self.row(rec(0.5, 7.0, 0.5), rec(0.9, 0.3, 0.95), rec(0.5, 5.0, n=1400), rec(0.95, 0.2, n=1400))
        self.assertEqual(r["cell"], "fitting_not_met_dev_met")
        self.assertEqual({f: r[f] for f in RB.FLAGS}, {"gained_criterion": True, "lost_criterion": False, "dev_accuracy_lower": False,
                                                        "val_loss_higher": False, "dev_auroc_lower": False, "fitting_accuracy_lower": False,
                                                        "fitting_loss_higher": False})
        r = self.row(rec(0.9, 0.3, 0.95), rec(0.7, 1.2, 0.9), rec(0.95, 0.2, n=1400), rec(0.9, 0.4, n=1400), "control")
        self.assertEqual(r["cell"], "fitting_met_dev_not_met")
        self.assertEqual({f: r[f] for f in RB.FLAGS}, {"gained_criterion": False, "lost_criterion": True, "dev_accuracy_lower": True,
                                                        "val_loss_higher": True, "dev_auroc_lower": True, "fitting_accuracy_lower": True,
                                                        "fitting_loss_higher": True})
        for bad in (rec(0.5, 7.0, None), rec(0.5, float("nan")), rec(0.5, float("inf")), {**rec(0.5, 1.0), "accuracy": float("nan")}):
            with self.assertRaisesRegex(FR.Failure, "is not finite"):
                self.row(rec(0.5, 7.0), bad, rec(0.95, 0.2, n=1400), rec(0.95, 0.2, n=1400))
        with self.assertRaisesRegex(FR.Failure, "is not finite"):
            self.row(rec(0.5, 7.0), rec(0.5, 7.0), rec(0.95, float("nan"), n=1400), rec(0.95, 0.2, n=1400))
        r = self.row(rec(0.5, 7.0, 0.5), rec(0.5, 7.0, 0.5), rec(0.95, 0.2, n=1400), rec(0.95, 0.2, n=1400))
        self.assertEqual(r["cell"], "fitting_met_dev_not_met")
        self.assertFalse(any(r[f] for f in RB.FLAGS))
        r = self.row(rec(0.5, 7.0), rec(0.75, 0.999), rec(0.7, 5.0, n=1400), rec(0.5, 5.1, n=1400))
        self.assertEqual((r["cell"], r["dev_criterion_after"], r["fitting_criterion_before"]), ("fitting_not_met_dev_met", True, False))
        r = self.row(rec(0.5, 7.0), rec(0.75, 1.0), rec(0.75, 0.99, n=1400), rec(0.5, 5.1, n=1400))
        self.assertEqual((r["cell"], r["dev_criterion_after"], r["fitting_criterion_before"]), ("fitting_met_dev_not_met", False, True))
        self.assertEqual({r2 for r2 in RB.CELLS}, {RB.cell_of(a, b) for a in (True, False) for b in (True, False)})

    def test_aggregate_cases_and_lists(self):
        good, bad = (rec(0.9, 0.3), rec(0.9, 0.3, n=1400)), (rec(0.5, 7.0), rec(0.5, 5.0, n=1400))
        def rows(k_ten, n_ten=3, lose=0):
            out = []
            for i in range(n_ten):
                r = self.row(bad[0], good[0] if i < k_ten else bad[0], bad[1], good[1], "ten")
                out.append({**r, "seed": 40 + i})
            for i in range(3):
                r = self.row(good[0], bad[0] if i < lose else good[0], good[1], good[1], "control")
                out.append({**r, "fold": 2, "seed": 40 + i})
            out.append({**self.row(rec(0.2, 3.0), rec(0.3, 2.0), good[1], good[1], "apart"), "fold": 3})
            return out
        a = RB.aggregate(self.cfg, rows(3))
        self.assertEqual((a["primary"]["case"], a["primary"]["statement"]), ("all", "all 3 of 3"))
        a = RB.aggregate(self.cfg, rows(0))
        self.assertEqual((a["primary"]["case"], a["primary"]["statement"]), ("none", "none 0 of 3"))
        a = RB.aggregate(self.cfg, rows(2, lose=1))
        self.assertEqual((a["primary"]["case"], a["primary"]["positives"], a["primary"]["statement"]), ("some", 2, "some 2 of 3"))
        self.assertEqual(a["primary"]["not_recovered"], ["arm_b_fold1_seed42"])
        self.assertEqual((a["controls"]["keep"], a["controls"]["lose"], a["controls"]["lost"]), (2, 1, ["arm_b_fold2_seed40"]))
        self.assertEqual(a["cells"]["counts"]["ten"], {"fitting_met_dev_met": 0, "fitting_met_dev_not_met": 0, "fitting_not_met_dev_met": 2,
                                                       "fitting_not_met_dev_not_met": 1})
        self.assertEqual(sum(a["cells"]["counts"]["all"].values()), 7)
        self.assertEqual([w["model"] for w in a["worsening"]["models"]], ["arm_b_fold2_seed40"])
        self.assertEqual(a["worsening"]["counts"]["lost_criterion"], 1)
        self.assertEqual(a["worsening"]["models"][0]["sizes"]["lost_criterion"], {"before": 1, "after": 0, "difference": -1})
        self.assertEqual(a["worsening"]["models"][0]["sizes"]["dev_accuracy_lower"], {"before": 360, "after": 200, "difference": -160})
        self.assertEqual(a["apart"]["models"][0]["model"], "arm_b_fold3_seed42")
        self.assertEqual(a["cells"]["statements"], self.cfg["reading_rule"]["cells"])

    def test_public_tables_deterministic(self):
        r = [self.row(rec(0.5, 7.0), rec(0.9, 0.3), rec(0.5, 5.0, n=1400), rec(0.9, 0.3, n=1400))]
        self.assertEqual(RB.results_csv(self.cfg, r), RB.results_csv(self.cfg, copymod.deepcopy(r)))
        doc = {"run_id": "x", "status": "PASS", "parent_run_id": "p", "test_read": "t", "config_sha256": "0" * 64, "note_sha256": "1" * 64, "models": 1,
               "reading": RB.aggregate(self.cfg, r), "timings": {"wall_seconds": 1.0, "phases": {"a": 1.0}, "estimate_central_minutes": 32,
                                                                 "estimate_upper_minutes": 140, "cap_minutes": 240}}
        self.assertEqual(RB.aggregate_md(self.cfg, doc, r), RB.aggregate_md(self.cfg, copymod.deepcopy(doc), copymod.deepcopy(r)))
        self.assertEqual(len(RB.result_columns(self.cfg)), len(set(RB.result_columns(self.cfg))))


# ============================================================================ configurazione

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.w = world()

    def refused(self, mutate, needle: str):
        doc = self.w.bn_doc()
        mutate(doc)
        p = self.w.write_bn_config(doc)
        try:
            with self.assertRaises(FR.Failure) as cm:
                RB.load_config(p)
            self.assertIn(needle, str(cm.exception))
        finally:
            self.w.write_bn_config(self.w.bn_doc())

    def test_synthetic_configuration_accepted(self):
        cfg, h = RB.load_config(self.w.bn_config)
        self.assertEqual(h, sha(self.w.bn_config.read_bytes()))

    def test_closed_sets_and_texts(self):
        self.refused(lambda d: d.update(extra=1), "fields must be exactly")
        self.refused(lambda d: d.pop("test"), "fields must be exactly")
        self.refused(lambda d: d.update(schema="x"), "schema must be")
        self.refused(lambda d: d.update(test="not read, not computed"), "test must state exactly what is read of the test")
        self.refused(lambda d: d["tolerances"].update(val_loss_rel={"value": 0.01, "source": "t"}), "tolerances needs exactly")
        self.refused(lambda d: d["budget"].update(estimate_minutes=200), "estimate_minutes must not exceed")
        self.refused(lambda d: d["budget"].pop("upper_estimate_minutes"), "budget needs exactly")
        self.refused(lambda d: d.update(stop_rule="stop"), "stop_rule must declare the BASELINE_FAILED outcome")
        self.refused(lambda d: d.update(note_sha256="abc"), "note_sha256 must be null or a SHA-256")
        self.refused(lambda d: d["parent_run"].pop("used"), "parent_run needs exactly")
        self.refused(lambda d: d["parent_run"].update(config_sha256="x"), "parent_run SHA-256 values malformed")
        self.refused(lambda d: d["parent_run"].update(models=0), "parent_run.models must be a positive count")
        self.refused(lambda d: d["architecture"].update(name="AttackNet"), "architecture needs name LivenessNet")
        self.refused(lambda d: d["architecture"].update(input_shape=[8, 8]), "architecture.input_shape")
        self.refused(lambda d: d["architecture"]["layers"].append(["aug", "RandomFlip"]), "architecture.layers must list")
        self.refused(lambda d: d["batchnorm"].update(layers=["bn1_2", "bn1_1", "bn2_1", "bn2_2", "bn_dense"]), "network order")
        self.refused(lambda d: d["batchnorm"]["kinds"].update(bn1_1="other"), "batchnorm.kinds")
        self.refused(lambda d: d["batchnorm"].update(epsilon=0), "batchnorm.epsilon must be")
        self.refused(lambda d: d["estimator"].update(changed_tensors=["moving_mean", "moving_variance", "gamma"]), "estimator.changed_tensors")
        self.refused(lambda d: d["estimator"].update(variance=""), "estimator.variance must be a non-empty text")
        self.refused(lambda d: d["evaluation"].update(device="GPU"), "evaluation must be on the CPU")
        self.refused(lambda d: d["tolerances"].pop("consistency_abs"), "tolerances needs exactly")
        self.refused(lambda d: d["tolerances"]["val_loss_abs"].update(value=-1), "tolerances.val_loss_abs.value must be")
        self.refused(lambda d: d["tolerances"]["val_loss_abs"].update(value=True), "tolerances.val_loss_abs.value must be")
        self.refused(lambda d: d["tolerances"]["dev_decisions_discordant_at_0_5"].update(value=0.5), "must be a non-negative integer")
        self.refused(lambda d: d["tolerances"]["consistency_rel"].update(source=""), "tolerances.consistency_rel.source")
        self.refused(lambda d: d["recovery_criterion"].update(accuracy_min_inclusive="yes"), "inclusive flags")
        self.refused(lambda d: d["groups"]["ten"].update(expected=["0003"]), "groups.ten.expected must list")
        self.refused(lambda d: d["groups"]["apart"].update(expected=[TEN[0]]), "both among the ten and apart")
        self.refused(lambda d: d["groups"]["degenerate"].update(restored_epochs=[]), "groups.degenerate.restored_epochs")
        self.refused(lambda d: d["reading_rule"].update(some="some {k}"), "placeholders")
        self.refused(lambda d: d["reading_rule"]["cells"].pop("fitting_met_dev_met"), "reading_rule.cells needs exactly")
        self.refused(lambda d: d["reading_rule"].update(limits="the saturation is attributable to the statistics"), "causal wording")
        self.refused(lambda d: d["reading_rule"]["cells"].update(fitting_met_dev_met="fails even on its frames"), "causal wording")
        self.refused(lambda d: d["outputs"].update(public=["results.csv"]), "outputs.public must be")
        self.refused(lambda d: d["outputs"].update(restricted=["x"]), "outputs.restricted must list")
        self.refused(lambda d: d["budget"].update(wall_time_cap_minutes=0), "wall_time_cap_minutes must be a positive integer")
        self.refused(lambda d: d["budget"].update(on_cap_exceeded="stop"), "INCOMPLETE outcome")
        self.refused(lambda d: d["environment"].update(tensorflow=""), "environment.tensorflow")

    def test_delivered_configuration(self):
        p = ROOT / RB.DEFAULT_CONFIG
        if not p.is_file():
            skip_missing(self, "delivered configuration")
        cfg, _ = RB.load_config(p)
        t = {k: v["value"] for k, v in cfg["tolerances"].items()}
        self.assertEqual(t, {"dev_probability_max_abs_diff": 1e-05, "dev_decisions_discordant_at_0_5": 0, "val_loss_abs": 0.001,
                             "dev_accuracy_abs": 1e-06, "softmax_of_logits_max_abs_diff": 0.0001,
                             "consistency_rel": 0.0001, "consistency_abs": 1e-06})
        self.assertEqual(t["val_loss_abs"], FR.VAL_LOSS_TOL[0])
        self.assertEqual((cfg["budget"]["estimate_minutes"], cfg["budget"]["upper_estimate_minutes"]), (32, 140))
        self.assertEqual(cfg["test"], RB.TEST_READ)
        self.assertEqual(cfg["reading_rule"]["cells"], {"fitting_met_dev_met": "both", "fitting_met_dev_not_met": "fitting only",
                                                       "fitting_not_met_dev_met": "dev after only", "fitting_not_met_dev_not_met": "neither"})
        self.assertEqual(t["softmax_of_logits_max_abs_diff"], SR.PROB_TOL)
        self.assertEqual(cfg["evaluation"]["batch"], FR.SCORE_BATCH)
        self.assertEqual((len(cfg["groups"]["ten"]["expected"]), cfg["groups"]["apart"]["expected"], cfg["groups"]["control"]["expected_count"]),
                         (10, ["arm_b_fold3_seed44"], 25))
        self.assertEqual(cfg["budget"]["wall_time_cap_minutes"], 240)
        self.assertEqual(cfg["batchnorm"]["layers"], BN)
        self.assertEqual([n for n, _ in cfg["architecture"]["layers"]], [n for n, _ in LAYERS])
        self.assertEqual(cfg["architecture"]["layers"], LAYERS)
        self.assertEqual(cfg["parent_run"]["config_sha256"], SR.CONFIG_SHA256)
        self.assertEqual((cfg["recovery_criterion"]["accuracy_min"], cfg["recovery_criterion"]["val_loss_max"]), (0.75, 1.0))
        self.assertTrue(cfg["note_sha256"] is None or len(cfg["note_sha256"]) == 64)


# ============================================================================ ingressi (fase 0)

class TestInputs(unittest.TestCase):
    def setUp(self):
        self.w = world()

    def tearDown(self):
        self.w.write_bn_config(self.w.bn_doc())

    def load(self, doc=None):
        cfg, _ = RB.load_config(self.w.write_bn_config(doc or self.w.bn_doc()))
        with injected(**hooks(self.w)):
            return RB.load_inputs(cfg, self.w.base)

    def refused(self, needle: str, doc=None):
        with self.assertRaises(FR.Failure) as cm:
            self.load(doc)
        self.assertIn(needle, str(cm.exception))

    def test_synthetic_inputs_pass_without_tensorflow_model(self):
        inp = self.load()
        self.assertEqual(len(inp.order), 36)
        self.assertEqual(inp.folds, {})
        self.assertEqual(inp.checks["groups"], {"ten": [n for n in inp.order if n in TEN], "control": 30, "apart": APART})
        self.assertEqual(len(inp.cells_sha), 16)

    def test_check_inputs_command(self):
        before = tree_state(self.w.tmp)
        rc, out, err = run_main(self.w, self.w.argv("check-inputs"))
        self.assertEqual(rc, 0, err)
        self.assertIn("RESULT: PASS (check-inputs)", out)
        self.assertIn("fold schema and test manifests not opened", out)
        self.assertEqual(tree_state(self.w.tmp), before)

    def test_test_role_and_test_scores_refused(self):
        inp = self.load()
        with self.assertRaisesRegex(FR.Failure, "role 'test' refused: this runner assembles only the fitting and dev frames"):
            RB.role_rows(inp, "test", "arm_b", 1)
        with self.assertRaisesRegex(FR.Failure, "role 'test' refused"):
            RB.frames(None, inp, "test", "arm_b", 1)
        with self.assertRaisesRegex(FR.Failure, "never the test scores"):
            RB.parent_path(self.w.parent, "arm_b_fold1_seed42", "test_scores.csv")
        self.assertEqual(RB.role_rows(inp, "dev", "arm_b", 2), inp.cells[("dev", 2)])
        self.assertEqual(RB.role_rows(inp, "fitting", "arm_b_prime", 2), inp.cells[("arm_b_prime", 2)])

    def test_note_hash(self):
        doc = self.w.bn_doc()
        doc["note_sha256"] = None
        self.refused("note_sha256 is null", doc)
        cfg, _ = RB.load_config(self.w.write_bn_config(doc))
        with injected(**hooks(self.w)):
            self.assertIsNone(RB.load_inputs(cfg, self.w.base, need_note=False).note_sha)
        doc["note_sha256"] = "0" * 64
        self.refused("differs from note_sha256 of the configuration", doc)
        rc, out, err = run_main(self.w, self.w.argv("check-inputs"))
        self.assertEqual(rc, 1)
        self.assertIn("ERROR: revised note NOTE.md: SHA-256", err)

    def test_parent_config_and_status_anchors(self):
        doc = self.w.bn_doc()
        doc["parent_run"]["config_sha256"] = "1" * 64
        self.refused("parent configuration config.json: SHA-256 differs", doc)
        for k in ("public_sha256sums_sha256", "restricted_sha256sums_sha256"):
            doc = self.w.bn_doc()
            doc["parent_run"][k] = "1" * 64
            self.refused("status.json is not PASS with the run id, configuration and SHA256SUMS", doc)
        doc = self.w.bn_doc()
        doc["parent_run"]["run_id"] = "20261106-SESSABLRUN-0000000"
        self.refused("status.json is not PASS", doc)
        doc = self.w.bn_doc()
        doc["parent_run"]["models"] = 35
        self.refused("the completed models differ from the training order", doc)
        st = self.w.parent / "status.json"
        saved = st.read_text()
        try:
            st.write_text(json.dumps({**self.w.parent_status, "status": "FAIL"}), encoding="utf-8")
            self.refused("status.json is not PASS")
            st.write_text(json.dumps({**self.w.parent_status, "completed": self.w.parent_status["completed"][:-1]}), encoding="utf-8")
            self.refused("the completed models differ")
        finally:
            st.write_text(saved)

    def test_parent_sums_and_files(self):
        r = self.w.parent / "restricted" / "SHA256SUMS"
        saved = r.read_text()
        try:
            r.write_text(saved + "\n")
            self.refused("restricted SHA256SUMS has SHA-256")
        finally:
            r.write_text(saved)
        card = self.w.parent / "restricted" / "arm_b_fold2_seed43_card.json"
        saved = card.read_text()
        try:
            card.write_text(saved.replace('"train_seed": 43', '"train_seed": 42'))
            self.refused("arm_b_fold2_seed43_card.json: SHA-256")
        finally:
            card.write_text(saved)
        k = self.w.parent / "restricted" / "arm_b_fold2_seed43_LivenessNet.keras"
        b = k.read_bytes()
        try:
            k.write_bytes(b + b"x")
            self.refused("arm_b_fold2_seed43_LivenessNet.keras: SHA-256")
        finally:
            k.write_bytes(b)

    def rebuilt(self, over: dict, needle: str):
        try:
            self.w.build_parent(over)
            self.refused(needle)
        finally:
            self.w.build_parent()

    def test_card_checks(self):
        n = "arm_a_star_fold1_seed43"
        self.rebuilt({n: lambda c: c.update(weights_sha256="2" * 64)}, f"{n}: weights_sha256 of the card differs from the checkpoint")
        self.rebuilt({n: lambda c: c.update(fitting_keys_sha256="2" * 64)}, f"{n}: the fitting or dev keys")
        self.rebuilt({n: lambda c: c.update(dev_keys_sha256="2" * 64)}, f"{n}: the fitting or dev keys")
        self.rebuilt({n: lambda c: c.update(n_fitting=c["n_fitting"] + 1)}, "the card reports 33 fitting and 16 dev frames, the manifests hold 32 and 16")
        self.rebuilt({n: lambda c: c.update(fold=2)}, "arm, fold or seed differ from the file name")
        self.rebuilt({n: lambda c: c.pop("val_loss_recomputed")}, "fields ['val_loss_recomputed'] missing")
        self.rebuilt({n: lambda c: c["history"]["val_accuracy"].__setitem__(c["restored_epoch"] - 1, 0.1)},
                     "is not its history at the restored epoch")

    def test_parent_aggregate_hashed_never_parsed(self):
        """L'aggregato pubblico del genitore del mondo non è JSON: le fasi passano (nessun parser lo legge); i suoi byte
        alterati senza aggiornare il SHA256SUMS sono rifiutati."""
        a = self.w.parent / "public" / "aggregate.json"
        with self.assertRaises(json.JSONDecodeError):
            json.loads(a.read_bytes())
        self.load()
        saved = a.read_bytes()
        try:
            a.write_bytes(saved + b"x")
            self.refused("parent aggregate.json: SHA-256 differs from its SHA256SUMS")
        finally:
            a.write_bytes(saved)

    def test_dev_scores_and_threshold(self):
        n = "arm_b_fold2_seed42"
        p = self.w.parent / "restricted" / f"{n}_dev_scores.csv"
        rows = read_csv(p)
        try:
            for mut, needle in ((lambda rs: rs.reverse(), "keys different from the dev manifest"),
                                (lambda rs: rs[0].update(role="test"), "rows of another model or role"),
                                (lambda rs: rs[0].update(score_attack="1.5"), "score_attack outside [0, 1]"),
                                (lambda rs: rs[0].update(score_attack="x"), "score_attack not numeric")):
                rs = copymod.deepcopy(rows)
                mut(rs)
                p.write_text(CFR.csv_text(SR.SCORE_COLUMNS, [[r[c] for c in SR.SCORE_COLUMNS] for r in rs]), encoding="utf-8")
                doc = self.resum()
                self.refused(needle, doc)
            p.write_text(CFR.csv_text(SR.SCORE_COLUMNS, [[r[c] for c in SR.SCORE_COLUMNS] for r in rows]), encoding="utf-8")
            t = self.w.parent / "restricted" / f"{n}_threshold.json"
            saved = t.read_text()
            t.write_text(json.dumps({**json.loads(saved), "checkpoint_sha256": "4" * 64}))
            self.refused("the parent threshold does not belong to the checkpoint", self.resum())
            t.write_text(saved)
        finally:
            self.w.build_parent()

    def resum(self) -> dict:
        doc = self.w.bn_doc()
        res = write_sums(self.w.parent / "restricted")
        doc["parent_run"]["restricted_sha256sums_sha256"] = res
        (self.w.parent / "status.json").write_text(json.dumps({**self.w.parent_status, "restricted_sums_sha256": res}))
        return doc

    def test_groups_mismatch(self):
        doc = self.w.bn_doc()
        doc["groups"]["ten"]["expected"] = TEN[1:]
        self.refused("the degenerate models derived from the cards", doc)
        doc = self.w.bn_doc()
        doc["groups"]["apart"]["expected"] = []
        self.refused("the models outside the criterion derived from the cards", doc)
        doc = self.w.bn_doc()
        doc["groups"]["control"]["expected_count"] = 29
        self.refused("30 control models derived from the cards, the configuration expects 29", doc)

    def test_architecture_environment_and_parent_configuration(self):
        doc = self.w.bn_doc()
        doc["architecture"]["sha256"] = "5" * 64
        self.refused("the architecture differs from this configuration", doc)
        doc = self.w.bn_doc()
        doc["environment"]["tensorflow"] = "0.0.0"
        self.refused("TensorFlow", doc)
        doc = self.w.bn_doc()
        doc["evaluation"]["batch"] = 8
        self.refused("the scoring batch or device differ", doc)
        with mock_env("LIVENESSLAB_LIVEDETECTION", "x"):
            self.refused("environment variables")

    def test_folds_run_anchor_without_public_files(self):
        """Del run dei fold si leggono solo status, SHA256SUMS riservato ed extracted_frames.csv (i file pubblici hanno
        permessi 000 nel mondo); ognuno è controllato."""
        e = self.w.base / "foldsrun" / "restricted" / "extracted_frames.csv"
        saved = e.read_bytes()
        try:
            e.write_bytes(saved + b"\n")
            self.refused("folds run extracted_frames.csv: SHA-256 differs from its SHA256SUMS")
        finally:
            e.write_bytes(saved)
        st = self.w.base / "foldsrun" / "status.json"
        saved = st.read_bytes()
        try:
            st.write_text(json.dumps({**json.loads(saved), "status": "FAIL"}))
            self.refused("folds run 20261103-FOLDSRUN-abcdef0: status.json is not PASS")
        finally:
            st.write_bytes(saved)
        sums = self.w.base / "foldsrun" / "restricted" / "SHA256SUMS"
        saved = sums.read_bytes()
        try:
            sums.write_bytes(saved + b"\n")
            self.refused("folds run 20261103-FOLDSRUN-abcdef0: restricted SHA256SUMS has SHA-256")
        finally:
            sums.write_bytes(saved)
        self.assertIsNone(self.load().reference)

    def test_shared_loaders_still_check_the_frames(self):
        """I caricatori riusati del runner genitore restano attivi: un frame estratto alterato è rifiutato."""
        z = np.load(self.w.frames_file)
        X, keys = z["X"].copy(), z["keys"].copy()
        X[0, 0, 0, 0] ^= 1
        saved = self.w.frames_file.read_bytes()
        try:
            self.w.write_frames(X, list(keys))
            self.refused("extracted frames file frames.npz: SHA-256")
        finally:
            self.w.frames_file.write_bytes(saved)


@contextlib.contextmanager
def mock_env(k, v):
    old = os.environ.get(k)
    os.environ[k] = v
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = old


# ============================================================================ run

class TestRunPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = pass_run()
        cls.folder = cls.r["run"]

    def test_pass_and_outputs(self):
        self.assertEqual(self.r["rc"], 0, self.r["err"][-3000:])
        st = json.loads((self.folder / "status.json").read_text())
        self.assertEqual((st["status"], st["passed"], st["baseline_failed"]), ("PASS", True, []))
        self.assertEqual(sorted(os.listdir(self.folder / "public")), sorted(RB.PUBLIC_OUTPUTS))
        order = [SR.model_name(*x) for x in SR.training_order(CFR.K, SAR.SEEDS)]
        self.assertEqual(sorted(os.listdir(self.folder / "restricted")), sorted(RB.restricted_outputs(order)))
        for sub in ("public", "restricted"):
            listed = FR.read_sums((self.folder / sub / "SHA256SUMS").read_bytes(), "sums")
            self.assertEqual({n: sha((self.folder / sub / n).read_bytes()) for n in listed}, listed)
        self.assertEqual(st["public_sums_sha256"], sha((self.folder / "public" / "SHA256SUMS").read_bytes()))
        self.assertEqual(st["restricted_sums_sha256"], sha((self.folder / "restricted" / "SHA256SUMS").read_bytes()))
        self.assertIn("RESULT: PASS (run)", self.r["out"])

    def test_parent_untouched(self):
        self.assertEqual(self.r["after"], self.r["before"])
        rec = json.loads((self.folder / "public" / "run_record.json").read_text())
        self.assertEqual(rec["checks"]["final"], {"parent_checkpoints_unchanged": 36, "parent_restricted_sha256sums_unchanged": True})

    def test_results_all_models(self):
        rows = read_csv(self.folder / "public" / "results.csv")
        self.assertEqual(len(rows), 36)
        by = {f"{r['arm']}_fold{r['fold']}_seed{r['seed']}": r for r in rows}
        for n, r in by.items():
            kind = KINDS.get(n, "good")
            self.assertEqual(r["group"], {"good": "control", "badstats": "ten", "bias": "ten", "inverted": "apart"}[kind])
            self.assertEqual(int(r["restored_epoch"]), EPOCH[kind])
            if kind == "badstats":
                self.assertEqual((r["dev_criterion_before"], r["dev_criterion_after"], r["gained_criterion"], r["cell"]),
                                 ("0", "1", "1", "fitting_not_met_dev_met"))
                self.assertEqual(r["dev_accuracy_before"], "0.5")
            if kind == "bias":
                self.assertEqual((r["dev_criterion_after"], r["dev_accuracy_after"]), ("0", "0.5"))
            if kind == "inverted":
                self.assertEqual((r["dev_accuracy_before"], r["dev_accuracy_after"]), ("0.0", "0.0"))
            if kind == "good":
                self.assertEqual((r["dev_criterion_before"], r["dev_criterion_after"], r["fitting_criterion_before"]), ("1", "1", "1"))
            self.assertEqual((int(r["n_fitting"]), int(r["n_dev"])), (len(world().keys(r["arm"], int(r["fold"]))), len(world().keys("dev", int(r["fold"])))))
            for c in ("bn1_1_var_ratio_min", "bn_dense_mean_shift_max", "baseline_seconds", "recalibration_seconds", "evaluation_seconds"):
                float(r[c])
        a = json.loads((self.folder / "public" / "aggregate.json").read_text())["reading"]
        self.assertEqual((a["primary"]["case"], a["primary"]["positives"], a["primary"]["of"]), ("some", 3, 5))
        self.assertEqual(a["primary"]["statement"], "some 3 of 5")
        self.assertEqual(sorted(a["primary"]["not_recovered"]), sorted(n for n, k in KINDS.items() if k == "bias"))
        self.assertEqual((a["controls"]["n"], a["controls"]["keep"], a["controls"]["lose"]), (30, 30, 0))
        self.assertEqual(a["cells"]["counts"]["ten"]["fitting_not_met_dev_met"], 3)
        md = (self.folder / "public" / "aggregate.md").read_text()
        self.assertIn("**some 3 of 5**", md)
        self.assertRegex(md, r"arm_b_fold3_seed44 \(apart\): val_loss_higher [0-9.]+ -> [0-9.]+ \(difference [0-9.]+\)")
        doc = json.loads((self.folder / "public" / "aggregate.json").read_text())
        self.assertEqual((doc["timings"]["estimate_central_minutes"], doc["timings"]["estimate_upper_minutes"]), (7, 9))
        self.assertIn("estimate of about 7 minutes (upper estimate about 9)", md)
        self.assertEqual(doc["test_read"][1:], RB.TEST_READ[1:])
        self.assertIn("## All models", md)

    def test_baseline_table(self):
        rows = read_csv(self.folder / "public" / "baseline.csv")
        self.assertEqual(len(rows), 36)
        self.assertEqual({r["baseline_pass"] for r in rows}, {"1"})
        self.assertEqual(list(rows[0]), list(RB.BASELINE_COLUMNS))
        self.assertTrue(all(float(r["prob_parent_max_abs_diff"]) <= 1e-5 and r["correct"] == r["correct_card"] for r in rows))
        self.assertEqual({r["fitting_reference"] for r in rows}, {"none (no recorded value in inference mode)"})

    def test_restricted_files(self):
        n = "arm_b_prime_fold1_seed44"                                    # badstats, B': ultimo batch corto
        s = json.loads((self.folder / "restricted" / f"{n}_bn_statistics.json").read_text())
        nfit = len(world().keys("arm_b_prime", 1))
        self.assertEqual(nfit % 16, 6)
        self.assertEqual({k: v["count"] for k, v in s["layers"].items()}, {"bn1_1": nfit * 64, "bn1_2": nfit * 64, "bn2_1": nfit * 16,
                                                                             "bn2_2": nfit * 16, "bn_dense": nfit})
        self.assertEqual({k: v["consistency"]["channels_outside_tolerance"] for k, v in s["layers"].items()}, {b: 0 for b in BN})
        self.assertEqual(s["layers"]["bn_dense"]["old_moving_mean"], [2.0, 0.0])
        for v in s["layers"].values():
            self.assertEqual(v["installed_moving_mean_float32"], [float(np.float32(x)) for x in v["mean_float64"]])
        keys = world().keys("arm_b_prime", 1)
        X = frames_of(keys).astype(np.float64) / 255.0
        np.testing.assert_allclose(s["layers"]["bn1_1"]["mean_float64"][0], X[..., 0].mean(), rtol=1e-6)   # canale 0 = luminosità
        sc = read_csv(self.folder / "restricted" / f"{n}_scores.csv")
        self.assertEqual(list(sc[0]), list(RB.SCORE_COLUMNS))
        ndev = len(world().keys("dev", 1))
        self.assertEqual(len(sc), 2 * (ndev + nfit))
        self.assertEqual({(r["role"], r["statistics"]) for r in sc}, {(a, b) for a in ("dev", "fitting") for b in RB.STATISTICS})
        before = [float(r["score_attack"]) for r in sc if r["role"] == "dev" and r["statistics"] == "moving"]
        stored = [float(r["score_attack"]) for r in read_csv(world().parent / "restricted" / f"{n}_dev_scores.csv")]
        np.testing.assert_allclose(before, stored, atol=1e-6)

    def test_public_has_no_identifiers(self):
        for f in RB.PUBLIC_OUTPUTS:
            t = (self.folder / "public" / f).read_text()
            self.assertNotRegex(t, r"\b(real|attack)/\d{4}_")
            for s in CFR.SUBJECTS:
                self.assertNotRegex(t, rf"(?<![\w.]){s}(?![\w.])")

    def test_determinism_of_public_tables(self):
        w = world()
        out2 = w.tmp / "out2"
        rc, out, err = run_main(w, w.argv("run", out=out2))
        self.assertEqual(rc, 0, err[-2000:])
        r2 = run_folder(out2)
        self.assertEqual((r2 / "public" / "baseline.csv").read_bytes(), (self.folder / "public" / "baseline.csv").read_bytes())
        def strip(p):
            rows = read_csv(p)
            return [{k: v for k, v in r.items() if not k.endswith("_seconds")} for r in rows]
        self.assertEqual(strip(r2 / "public" / "results.csv"), strip(self.folder / "public" / "results.csv"))
        a1, a2 = (json.loads((x / "public" / "aggregate.json").read_text()) for x in (self.folder, r2))
        self.assertEqual(a1["reading"], a2["reading"])
        for n in os.listdir(self.folder / "restricted"):
            if n.endswith(("_scores.csv", "_bn_statistics.json")):
                self.assertEqual((r2 / "restricted" / n).read_bytes(), (self.folder / "restricted" / n).read_bytes(), n)
        shutil.rmtree(out2)

    def test_existing_run_folder_refused(self):
        w = world()
        rc, out, err = run_main(w, w.argv("run"))
        self.assertEqual(rc, 1)
        self.assertIn("already exists", err)
        self.assertEqual(json.loads((self.folder / "status.json").read_text())["status"], "PASS")


class TestRunStops(unittest.TestCase):
    def setUp(self):
        self.w = world()
        self.out = self.w.tmp / "stops"

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)
        self.w.write_bn_config(self.w.bn_doc())

    def go(self, *extra, **kw):
        rc, out, err = run_main(self.w, self.w.argv("run", *extra, out=self.out), **kw)
        return rc, out, err, run_folder(self.out)

    def test_baseline_failed_stops_before_any_recalibration(self):
        n = "arm_a_star_fold1_seed42"
        calls = []
        real = RB.recalibrate
        try:
            self.w.build_parent({n: lambda c: (c.update(best_val_loss=c["best_val_loss"] + 0.5),
                                               c["history"]["val_loss"].__setitem__(c["restored_epoch"] - 1, c["best_val_loss"]))})
            rc, out, err, run = self.go(rb={"recalibrate": lambda *a, **k: calls.append(1) or real(*a, **k)})
        finally:
            self.w.build_parent()
        self.assertEqual(rc, 3, err[-2000:])
        st = json.loads((run / "status.json").read_text())
        self.assertEqual((st["status"], st["baseline_failed"], st["completed"]), ("BASELINE_FAILED", [n], []))
        self.assertEqual(calls, [])
        self.assertEqual(sorted(os.listdir(run / "public")), ["SHA256SUMS", "baseline.csv", "run_record.json"])
        self.assertFalse((run / "restricted").exists())
        rows = {f"{r['arm']}_fold{r['fold']}_seed{r['seed']}": r for r in read_csv(run / "public" / "baseline.csv")}
        self.assertEqual(len(rows), 36)
        self.assertEqual((rows[n]["val_loss_pass"], rows[n]["baseline_pass"], rows[n]["prob_pass"], rows[n]["accuracy_pass"]), ("0", "0", "1", "1"))
        self.assertGreater(float(rows[n]["val_loss_diff"]), 0.49)
        self.assertEqual(sum(r["baseline_pass"] == "1" for r in rows.values()), 35)
        self.assertIn("BASELINE_FAILED", out)
        self.assertIn("RESULT: BASELINE_FAILED (run)", out)
        self.assertNotIn("dev_accuracy_after", (run / "public" / "run_record.json").read_text())

    def test_baseline_failed_by_the_parent_scorer(self):
        def shifted(path, w):
            s = FR.tf_scorer(path, w)
            def f(X):
                p = s(X)
                q = p[:, 1] + 2e-5 if "arm_b_prime_fold4_seed44" in str(path) else p[:, 1]
                return np.stack([1 - q, q], 1)
            return f
        rc, out, err, run = self.go(rb={"MAKE_SCORER": shifted})
        self.assertEqual(rc, 3, err[-2000:])
        rows = [r for r in read_csv(run / "public" / "baseline.csv") if r["baseline_pass"] == "0"]
        self.assertEqual([(r["arm"], r["fold"], r["seed"], r["prob_pass"], r["val_loss_pass"]) for r in rows], [("arm_b_prime", "4", "44", "0", "1")])
        self.assertGreater(float(rows[0]["prob_parent_max_abs_diff"]), 1e-5)

    def test_consistency_failure_is_fail(self):
        real = RB.recalibrate
        def wrong(copy, part, bn, batch, budget=None, name=""):
            st = real(copy, part, bn, batch, budget, name)
            s = st["bn2_1"]
            s["mean32"] = s["mean32"] + np.float32(0.5)
            copy.get_layer("bn2_1").moving_mean.assign(s["mean32"])
            return st
        rc, out, err, run = self.go(rb={"recalibrate": wrong})
        self.assertEqual(rc, 1)
        self.assertIn("consistency check failed: the inputs of ['bn2_1', 'bn2_2', 'bn_dense']", err)
        st = json.loads((run / "status.json").read_text())
        self.assertEqual((st["status"], st["outputs"]), ("FAIL", "removed"))
        self.assertEqual([p for p in run.rglob("*") if p.is_file() and p.name not in ("status.json", "run.log")], [])
        self.assertIn("RESULT: FAIL (run)", (run / "run.log").read_text())

    def test_trained_tensor_change_is_fail(self):
        real = RB.recalibrate
        def touch(copy, part, bn, batch, budget=None, name=""):
            st = real(copy, part, bn, batch, budget, name)
            g = copy.get_layer("bn1_2").beta
            g.assign(g.numpy() + np.float32(1e-6))
            return st
        rc, out, err, run = self.go(rb={"recalibrate": touch})
        self.assertEqual(rc, 1)
        self.assertIn("trained tensors changed: bn1_2/beta", err)

    def test_checkpoint_changed_during_the_run(self):
        real = RB.recalibrate
        first = self.w.parent / "restricted" / "arm_b_fold1_seed42_LivenessNet.keras"
        later = self.w.parent / "restricted" / "arm_b_fold2_seed42_LivenessNet.keras"
        for target, needle in ((later, "arm_b_fold2_seed42_LivenessNet.keras: SHA-256 changed just before loading"),
                               (first, "parent checkpoints changed during the run: ['arm_b_fold1_seed42']")):
            def touch(copy, part, bn, batch, budget=None, name="", target=target):
                if name == "arm_a_star_fold1_seed42":
                    with open(target, "ab") as fh:
                        fh.write(b"x")
                return real(copy, part, bn, batch, budget, name)
            try:
                rc, out, err, run = self.go(rb={"recalibrate": touch})
            finally:
                self.w.build_parent()
            self.assertEqual(rc, 1)
            self.assertIn(needle, err)
            self.assertEqual(json.loads((run / "status.json").read_text())["status"], "FAIL")
            shutil.rmtree(self.out)
        with self.assertRaisesRegex(FR.Failure, "SHA-256 changed just before loading"):
            RB.load_keras(first, "0" * 64)

    def test_public_scan_refuses_identifiers(self):
        doc = self.w.bn_doc()
        doc["reading_rule"]["primary"] = f"primary of subject {CFR.SUBJECTS[2]}"
        self.w.write_bn_config(doc)
        rc, out, err, run = self.go()
        self.assertEqual(rc, 1)
        self.assertIn(f"refused: subject id {CFR.SUBJECTS[2]}", err)
        self.assertFalse((run / "public").exists() and any((run / "public").iterdir()))

    def test_dirty_tree_and_authorisation(self):
        rc, out, err, run = self.go(rb={"REPO_STATE": lambda root: {"commit": "abcdef0", "dirty": True}})
        self.assertEqual(rc, 1)
        self.assertIn("the git work tree is not clean", err)
        self.assertFalse(self.out.exists())
        rc, out, err = run_main(self.w, [x for x in self.w.argv("run", out=self.out) if x != "--i-am-authorized"])
        self.assertEqual(rc, 1)
        self.assertIn("pass --i-am-authorized", err)
        doc = self.w.bn_doc()
        doc["note_sha256"] = None
        self.w.write_bn_config(doc)
        rc, out, err, run = self.go()
        self.assertEqual(rc, 1)
        self.assertIn("note_sha256 is null", err)
        self.assertFalse(self.out.exists())

    def test_run_folder_guard(self):
        rc, out, err, run = self.go("--date", "2026111")
        self.assertEqual(rc, 1)
        self.assertIn("--date must be YYYYMMDD", err)
        rc, out, err = run_main(self.w, self.w.argv("run", out=self.w.parent / "public"))
        self.assertEqual(rc, 1)
        self.assertIn("refusing the run folder", err)

    def test_cap_in_the_baseline_and_in_the_recalibration(self):
        doc = self.w.bn_doc()
        doc["budget"]["wall_time_cap_minutes"] = 1
        self.w.write_bn_config(doc)
        t = [0.0]
        rc, out, err, run = self.go(sr={"CLOCK": lambda: t.__setitem__(0, t[0] + 3.0) or t[0]})
        self.assertEqual(rc, 2, err[-2000:])
        st = json.loads((run / "status.json").read_text())
        self.assertEqual((st["status"], st["stopped"]["phase"], st["completed"]), ("INCOMPLETE", "baseline", []))
        self.assertEqual(sorted(os.listdir(run / "public")), ["SHA256SUMS", "baseline.csv", "run_record.json"])
        self.assertLess(len(read_csv(run / "public" / "baseline.csv")), 36)
        shutil.rmtree(self.out)
        # tetto durante la ricalibrazione: l'orologio salta dopo due modelli completi
        def clock():
            d = run_folder(self.out) / "restricted"
            return 1e6 if d.is_dir() and len(list(d.glob("*_scores.csv"))) >= 2 else 0.0
        doc["budget"]["wall_time_cap_minutes"] = 240
        self.w.write_bn_config(doc)
        rc, out, err, run = self.go(sr={"CLOCK": clock})
        self.assertEqual(rc, 2, err[-2000:])
        st = json.loads((run / "status.json").read_text())
        self.assertEqual((st["status"], st["stopped"]["phase"], len(st["completed"])), ("INCOMPLETE", "recalibration", 2))
        self.assertEqual(len(read_csv(run / "public" / "results.csv")), 2)
        agg = json.loads((run / "public" / "aggregate.json").read_text())
        self.assertIsNone(agg["reading"])
        self.assertIn("2 of 36 models complete", agg["reading_not_applied"])
        self.assertEqual(sorted(os.listdir(run / "restricted")), sorted([f"{n}_{x}" for n in st["completed"] for x in RB.PER_MODEL] +
                                                                         list(RB.RESTRICTED_FIXED)))

    def test_voluntary_stop(self):
        real = SR.Budget.check
        def check(self_, step):
            if step.startswith("recalibration arm_a_star_fold1_seed42"):
                self_.stop_requested = "SIGUSR1"
            return real(self_, step)
        with injected(sr={"Budget": type("B", (SR.Budget,), {"check": check})}):
            rc, out, err, run = self.go()
        self.assertEqual(rc, 2, err[-2000:])
        st = json.loads((run / "status.json").read_text())
        self.assertEqual((st["status"], st["stopped"]["cause"], st["completed"]), ("INCOMPLETE", "SIGUSR1", ["arm_b_fold1_seed42"]))


class ZZRepositoryUntouched(unittest.TestCase):
    def test_declared_skips_only(self):
        self.assertEqual(sorted(MISSING_INPUT), ["delivered configuration"])


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
