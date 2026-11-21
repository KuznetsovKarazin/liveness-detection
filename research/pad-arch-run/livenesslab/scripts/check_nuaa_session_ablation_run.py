"""
Test di scripts/nuaa_session_ablation_run.py (run di ablazione della sessione 03, bracci B / A* / B'), offline e
deterministici.

Mondo sintetico in cartelle temporanee, costruito sul mondo della suite del run dei fold (check_nuaa_folds_run.World:
16 soggetti 1101–1116 con identificativi e struttura diversi da NUAA, manifest M2, schema dei fold con SHA256SUMS, cache
8×8, originali il cui contenuto è la chiave): in più il file dei frame estratti, un run dei fold finto (status,
SHA256SUMS, extracted_frames.csv, aggregate.json), un run delle celle finto (manifest dei tre bracci e del dev comune
costruiti con le regole di annidamento, devmix presente e mai letto, SHA256SUMS, status, configurazione delle celle).
Trainer e scorer finti iniettati (i punteggi dipendono dalla classe scritta nei pixel, dal braccio e dal seed: A*
separa meglio di B'), orologio e stato git iniettati. Nessun training né inferenza reali; nessuna scrittura nel
repository (controllo finale su results/, export/, delivery/). Gli ingressi veri sono usati in sola lettura da un solo
test (check-inputs), saltato se mancano (MISSING_INPUT).

  .venv/bin/python -I scripts/check_nuaa_session_ablation_run.py [--verbose]
"""
import contextlib
import csv
import hashlib
import io
import json
import os
import shutil
import signal
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_nuaa_folds_run as CFR  # noqa: E402
import nuaa_folds_run as FR  # noqa: E402
import nuaa_session_ablation_run as SR  # noqa: E402

REAL_ROOT = Path(os.environ.get("SESSABLRUN_INPUT_ROOT") or ROOT)
MISSING_INPUT = {
    "real inputs": "the M2 manifests, the fold schema run, the cells run, the folds run, the training cache, the extracted frames, "
                   "the NUAA images or the reference code are not present",
    "delivered configuration": "delivery/sessionablationrun/nuaa_session_ablation_run.json is not present",
    "patience12 configuration": "delivery/patienceverify/nuaa_session_ablation_run_patience12.json or its parent configuration "
                                "is not present",
    "constant_lr configuration": "delivery/constlr/nuaa_session_ablation_run_constant_lr.json or its parent configurations are "
                                 "not present",
}
STATE = {"commit": "abcdef0", "dirty": False}
RUN = "20261105-SESSABLRUN-abcdef0"
SEEDS = [42, 43, 44]
K = CFR.K
SHAPE = CFR.SHAPE
SUBJECTS = CFR.SUBJECTS
B_CELLS = {c: CFR.FIT_PER_CELL for c in SR.CELLS_12}                       # 8 per cella
A_CELLS = {"01/bona_fide": 6, "01/attack": 5, "02/bona_fide": 5, "02/attack": 6, "03/bona_fide": 5, "03/attack": 5}
BP_CELLS = {c: A_CELLS[c] for c in SR.CELLS_12}
DEV_CELLS = {c: CFR.DEV_PER_CELL for c in SR.CELLS_12}
CELLS_RUN, FOLDS_RUN = "20261104-SESSABL-abcdef0", "20261103-FOLDSRUN-abcdef0"
N_MODELS = 3 * K * len(SEEDS)


def skip_missing(test: unittest.TestCase, what: str):
    """Unico modo di saltare un test di questa suite: un ingresso mancante dichiarato in MISSING_INPUT."""
    test.skipTest(f"missing input: {MISSING_INPUT[what]}")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


csv_text = CFR.csv_text


def cell_key(r: dict) -> str:
    return f"{r['session']}/{FR.CLASS_NAME[r['label']]}"


# ============================================================================ trainer e scorer finti

def fake_scores(X, arm: str, seed: int) -> np.ndarray:
    """Probabilità N x 2: classe nel pixel [0,0,0]; A* separa bene, B' meno, B con sovrapposizione delle classi."""
    X = np.asarray(X)
    att = X[:, 0, 0, 0] == 200
    noise = X[:, 0, 0, 1].astype(float) - 128
    if arm == "arm_a_star":
        p = np.where(att, 0.78, 0.22) + noise / 1000.0
    elif arm == "arm_b_prime":
        p = np.where(att, 0.62, 0.38) + noise / 600.0
    else:
        p = np.where(att, 0.56, 0.44) + noise / 400.0
    p = np.clip(p + seed * 1e-4, 0.001, 0.999)
    return np.stack([1 - p, p], axis=1)


def parse_weights(w: bytes) -> tuple:
    parts = w.decode().split()
    return parts[parts.index("arm") + 1], int(parts[parts.index("seed") + 1])


class Log:
    calls = []


def fake_trainer(plan=None, epoch_hook=None, loss=None):
    """Trainer finto: pesi con braccio e seed, storia il cui minimo è il val_loss che lo scorer ricalcola sul dev
    (loss: (X_dev, y_dev, braccio, seed) -> val_loss registrato, per default quello dei logit finti)."""
    def train(data, training, weights_path, seed, arch, cap_check):
        assert set(data) == {"X_fit", "y_fit", "X_dev", "y_dev"}
        name = Path(weights_path).name
        arm = next(a for a in sorted(SR.ARMS, key=len, reverse=True) if name.startswith(a + "_fold"))
        w = f"fake-weights arm {arm} seed {seed} fit {sha(np.ascontiguousarray(data['X_fit']).tobytes())}".encode()
        Path(weights_path).write_bytes(w)
        vl = (loss or (lambda X, y, a, s: SR.val_loss_logits(np.log(fake_scores(X, a, s)), y)))(data["X_dev"], data["y_dev"], arm, seed)
        hist = plan(vl) if plan else [vl + 0.3, vl + 0.1, vl, vl + 0.05, vl + 0.06, vl + 0.07, vl + 0.08]
        for i in range(len(hist)):
            if epoch_hook:
                epoch_hook(name, i + 1)
            cap_check(f"epoch {i + 1}")
        Log.calls.append(("train", name, len(data["y_fit"]), len(data["y_dev"]), sha(np.ascontiguousarray(data["X_dev"]).tobytes())))
        best = int(np.argmin(hist))
        return {"arch": "LivenessNet", "train_seed": seed, "n_fitting": int(len(data["y_fit"])), "n_dev": int(len(data["y_dev"])),
                "max_epochs": training["max_epochs"], "epochs_run": len(hist), "restored_epoch": best + 1,
                "restored_by": "early stopping" if len(hist) < training["max_epochs"] else "end of epochs (set_weights)",
                "best_val_loss": float(hist[best]), "val_acc_at_best": 0.9,
                "history": {"loss": list(hist), "accuracy": [0.5] * len(hist), "val_loss": list(hist), "val_accuracy": [0.6] * len(hist),
                            "lr": [0.001] * len(hist)},
                "device": "CPU", "visible_devices": ["CPU"], "lr": training["initial_learning_rate"], "batch": training["batch"],
                "early_stopping_patience": training["early_stopping_patience"], "scheduler": training["scheduler"], "train_seconds": 1.5,
                "weights_file": name, "weights_sha256": sha(w), "saved": "now"}
    return train


def fake_make_scorer(weights_path, weights_sha):
    w = Path(weights_path).read_bytes()
    assert sha(w) == weights_sha
    arm, seed = parse_weights(w)
    Log.calls.append(("scorer", Path(weights_path).name))
    return lambda X: fake_scores(X, arm, seed)


def fake_make_logits(weights_path, weights_sha):
    """Logit finti dello strato di uscita: log delle probabilità finte (la loro softmax è lo scorer finto)."""
    w = Path(weights_path).read_bytes()
    assert sha(w) == weights_sha
    arm, seed = parse_weights(w)
    Log.calls.append(("logits", Path(weights_path).name))
    return lambda X: np.log(fake_scores(X, arm, seed))


def saturated_logits(X, arm: str, seed: int) -> np.ndarray:
    """Logit finti con il primo frame di ogni chiamata classificato male con margine 40 (perdita 40 > -ln 1e-7 = 16,118):
    softmax cross-entropy dai logit e quella con le probabilità limitate divergono."""
    z = np.log(fake_scores(X, arm, seed))
    if len(X):
        y0 = int(np.asarray(X)[0, 0, 0, 0] == 200)
        z[0, y0], z[0, 1 - y0] = 0.0, 40.0
    return z


# ============================================================================ mondo sintetico

def write_sums(folder: Path) -> str:
    texts = {p.name: p.read_bytes() for p in sorted(folder.iterdir()) if p.is_file() and p.name != "SHA256SUMS"}
    sums = "".join(f"{sha(b)}  {n}\n" for n, b in sorted(texts.items()))
    (folder / "SHA256SUMS").write_text(sums, encoding="utf-8")
    return sha(sums.encode())


class World(CFR.World):
    """Il mondo del run dei fold più frame estratti, run dei fold, run delle celle e la configurazione di questo run."""

    def build(self):
        super().build()
        self.build_ablation()
        self.write_config(self.sr_doc())

    def build_ablation(self):
        b = self.base
        self.cache_keys = set(self.order["train"]) | set(self.order["test"])
        self.to_extract = sorted(k for k in self.official if k not in self.cache_keys)
        self.X_ext = np.stack([CFR.fake_frame(k.encode())[0] for k in self.to_extract])
        (b / "processed").mkdir()
        self.frames_file = b / "processed" / "frames.npz"
        self.write_frames(self.X_ext, self.to_extract)
        self.build_folds_run()
        self.build_cells_run()

    def write_frames(self, X, keys):
        with open(self.frames_file, "wb") as f:
            np.savez(f, X=X, keys=np.array(keys))

    def build_folds_run(self):
        fr = self.base / "foldsrun"
        (fr / "restricted").mkdir(parents=True)
        (fr / "public").mkdir()
        rows = []
        for i, k in enumerate(self.to_extract):
            o = self.official[k]
            rows.append([k, o["subject"], o["session"], o["label"], o["official_split"], "RetinaFace", o["sha256"], FR.frame_hashes(self.X_ext[i:i + 1])[0]])
        (fr / "restricted" / "extracted_frames.csv").write_text(csv_text(FR.FRAME_COLUMNS, rows), encoding="utf-8")
        (fr / "restricted" / "per_subject.csv").write_text("x\n", encoding="utf-8")
        # valori diversi per seed (0.90, 0.91, 0.92): il riferimento A deve essere copiato seed per seed
        metric = {s: {**{k: 0.1 for k in FR.METRIC_KEYS}, "auroc": round(0.9 + (s - 42) / 100, 2), "n": 10, "n_subjects": 9} for s in SEEDS}
        agg = {"run_id": FOLDS_RUN, "pooled_per_seed": {str(s): {st: dict(metric[s]) for st in FR.STRATA} for s in SEEDS},
               "bootstrap_per_seed": {str(s): {st: {"acer_05": {"low": 0.05, "high": 0.2}, "acer_dev": {"low": 0.05, "high": 0.2},
                                                    "auroc": {"low": round(0.8 + (s - 42) / 100, 2), "high": 0.95}} for st in FR.STRATA}
                                      for s in SEEDS}}
        (fr / "public" / "aggregate.json").write_text(json.dumps(agg, indent=1) + "\n", encoding="utf-8")
        self.folds_res = write_sums(fr / "restricted")
        self.folds_status = {"run_id": FOLDS_RUN, "status": "PASS", "config_sha256": "7" * 64, "restricted_sums_sha256": self.folds_res}
        self.write_folds_record()

    def write_folds_record(self, over=None, drop=()) -> dict:
        """run_record.json pubblico del run dei fold con code_sha256 del codice condiviso (quello di questa copia) e
        dell'architettura; aggiorna SHA256SUMS pubblico, status e configurazione e la restituisce."""
        fr = self.base / "foldsrun"
        code = {f: FR.sha256_file(SR.ROOT / f) for f in SR.SHARED_CODE}
        code[SR.FOLDS_RUN_ARCH_KEY] = self.code["ref/architectures.py"]
        code.update(over or {})
        for k in drop:
            code.pop(k)
        (fr / "public" / "run_record.json").write_text(json.dumps({"run_id": FOLDS_RUN, "code_sha256": code}, indent=1) + "\n", encoding="utf-8")
        self.folds_pub = write_sums(fr / "public")
        self.folds_status["public_sums_sha256"] = self.folds_pub
        (fr / "status.json").write_text(json.dumps(self.folds_status, indent=1) + "\n", encoding="utf-8")
        if not hasattr(self, "cells_cfg_sha"):              # durante la costruzione: la configurazione è scritta dopo
            return {}
        doc = self.sr_doc()
        self.write_config(doc)
        return doc

    def cells_rows(self, fold: int) -> dict:
        """Le righe dei manifest delle celle del fold, con le regole di annidamento della proposta."""
        cols = FR.FOLD_COLUMNS
        fit = [dict(zip(cols, r)) for r in self.folds[fold]["fitting"]]
        dev = [dict(zip(cols, r)) for r in self.folds[fold]["dev"]]
        b_rows = [r for r in fit if r["session"] != "03"]
        s12 = {r["subject"] for r in b_rows}
        a_rows = []
        for c in SR.CELLS:
            if c.startswith("03"):
                avail = sorted(k for k, o in self.official.items() if o["subject"] in s12 and cell_key(o) == c)
            else:
                avail = sorted(r["key"] for r in b_rows if cell_key(r) == c)
            a_rows += [{**{x: self.official[k][x] for x in ("key", "label", "subject", "session", "sha256", "bytes")},
                        "in_cache": "1" if k in self.cache_keys else "0"} for k in FR.spread(avail, A_CELLS[c])]
        bp_rows = [r for r in a_rows if r["session"] != "03"]
        dev_rows = [r for r in dev if r["session"] != "03"]
        mix = list(dev)
        out = {}
        for name, rows, role in (("armB_fitting", b_rows, "arm_b_fitting"), ("armAstar_fitting", a_rows, "arm_a_star_fitting"),
                                 ("armBprime_fitting", bp_rows, "arm_b_prime_fitting_optional"), ("dev", dev_rows, "dev_common"),
                                 ("devmix", mix, "dev_mix_alternative")):
            out[name] = [[str(fold), role] + [r[c] for c in cols[2:]] for r in sorted(rows, key=lambda r: r["key"])]
        return out

    def build_cells_run(self):
        cr = self.base / "cells"
        (cr / "restricted").mkdir(parents=True)
        (cr / "public").mkdir()
        self.cells = {}
        for fold in range(1, K + 1):
            for name, rows in self.cells_rows(fold).items():
                self.cells[(name, fold)] = rows
                (cr / "restricted" / f"{name}_fold{fold}.csv").write_text(csv_text(FR.FOLD_COLUMNS, rows), encoding="utf-8")
        (cr / "public" / "session_ablation_public.json").write_text("{}\n", encoding="utf-8")
        self.cells_pub = write_sums(cr / "public")
        self.cells_res = write_sums(cr / "restricted")
        self.cells_cfg = self.base / "cells_config.json"
        self.cells_cfg.write_text(json.dumps({"design": {
            "arm_b": {"cells": list(SR.CELLS_12), "per_cell": CFR.FIT_PER_CELL}, "arm_a_star": {"allocation": A_CELLS},
            "arm_b_prime": {"allocation": BP_CELLS}, "dev": {"cells": list(SR.CELLS_12), "per_cell": CFR.DEV_PER_CELL}}}, indent=1) + "\n",
            encoding="utf-8")
        self.cells_cfg_sha = sha(self.cells_cfg.read_bytes())
        self.write_cells_status()

    def write_cells_status(self, **over):
        st = {"status": "PASS", "run_id": CELLS_RUN, "config_sha256": self.cells_cfg_sha, "public_sha256sums": self.cells_pub,
              "restricted_sha256sums": self.cells_res, **over}
        (self.base / "cells" / "status.json").write_text(json.dumps(st, indent=1) + "\n", encoding="utf-8")

    def rewrite_cell(self, name: str, fold: int, rows) -> dict:
        """Riscrive un manifest delle celle, aggiorna SHA256SUMS e status e restituisce la configurazione coerente."""
        cr = self.base / "cells"
        (cr / "restricted" / f"{name}_fold{fold}.csv").write_text(csv_text(FR.FOLD_COLUMNS, rows), encoding="utf-8")
        self.cells_res = write_sums(cr / "restricted")
        self.write_cells_status()
        doc = self.sr_doc()
        self.write_config(doc)
        return doc

    def sr_doc(self) -> dict:
        d = self.config_doc()                    # la configurazione del run dei fold del mondo: sezioni riusate
        cache = {k: v for k, v in d["cache"].items() if k in ("path", "sha256", "per_class", "members")}
        cache["order"] = "spread per class"
        exp = d["expected"]
        return {
            "schema": SR.CONFIG_SCHEMA, "task": "synthetic test world", "authorisation": "test",
            "cells_run": {"run_id": CELLS_RUN, "folder": "cells", "config_path": "cells_config.json", "config_sha256": self.cells_cfg_sha,
                          "restricted_sha256sums_sha256": self.cells_res, "public_sha256sums_sha256": self.cells_pub},
            "schema_run": d["schema_run"],
            "folds_run": {"run_id": FOLDS_RUN, "folder": "foldsrun", "config_sha256": "7" * 64, "public_sha256sums_sha256": self.folds_pub,
                          "restricted_sha256sums_sha256": self.folds_res,
                          "frames_file": {"path": "processed/frames.npz", "sha256": FR.sha256_file(self.frames_file), "n": len(self.to_extract)}},
            "manifests": d["manifests"], "images": d["images"], "schema_expected": exp, "cache": cache,
            "arms": {"arm_b": {"label": "B", "file": SR.ARM_FILE["arm_b"], "role": SR.ARM_ROLE["arm_b"], "cells": dict(B_CELLS),
                               "total": sum(B_CELLS.values()), "description": "t"},
                     "arm_a_star": {"label": "A*", "file": SR.ARM_FILE["arm_a_star"], "role": SR.ARM_ROLE["arm_a_star"], "cells": dict(A_CELLS),
                                    "total": sum(A_CELLS.values()), "description": "t"},
                     "arm_b_prime": {"label": "B'", "file": SR.ARM_FILE["arm_b_prime"], "role": SR.ARM_ROLE["arm_b_prime"], "cells": dict(BP_CELLS),
                                     "total": sum(BP_CELLS.values()), "description": "t"}},
            "dev": {"file": SR.DEV_FILE, "role": SR.DEV_ROLE, "cells": dict(DEV_CELLS), "total": sum(DEV_CELLS.values()), "rule": "t"},
            "nesting": {"rule": "t"}, "excluded": {"arm_a_star_star": "t", "dev_mix": "t"},
            "test": {"source": "t", "per_fold": list(exp["test_per_fold"])},
            "training_order": [SR.model_name(*x) for x in SR.training_order(K, SEEDS)],
            "environment": d["environment"],
            "architecture": {"name": "LivenessNet", "path": "ref/architectures.py", "sha256": self.code["ref/architectures.py"], "chosen": "t"},
            "detector": "not used", "train_seeds": list(SEEDS), "training": dict(FR.FROZEN_TRAINING),
            "threshold": {"set": "common dev of the fold only", "criterion": FR.THRESHOLD_CRITERION, "frozen_before_test": "t",
                          "also_reported": "t", "dev_reuse_declared": "t", "val_loss_check": SR.VAL_LOSS_CHECK},
            "scoring": d["scoring"],
            "metrics": {"strata": list(FR.STRATA), "decision": FR.DECISION_RULE, "keys": list(FR.METRIC_KEYS), "per_unit": "t", "per_subject": "t",
                        "auroc": "t", "aggregation": "t", "pre_commitment": "t"},
            "comparisons": {"list": [{"name": n, "minuend": m, "subtrahend": s, "role": r} for n, m, s, r in SR.COMPARISONS],
                            "pairing": "t", "pooled": "t"},
            "bootstrap": {**SR.BOOT, "recipe": SR.BOOT_RECIPE},
            "reading_rule": {"primary_comparison": "a_star_minus_b", "primary_stratum": "03", "primary_metrics": ["auroc", "acer_dev"],
                             "rule": SR.READING_RULE, "verdicts": ["consistent", "inconclusive"], "statistical_test": "none"},
            "reference_a": {"source": "t", "label": SR.REFERENCE_LABEL, "use": "t"},
            "budget": {"trainings": N_MODELS, "estimate": "t", "wall_time_cap_minutes": 360,
                       "on_cap_exceeded": "INCOMPLETE with the complete models", "on_failure": "FAIL, complete models kept",
                       "voluntary_stop": "SIGUSR1 gives INCOMPLETE", "final_writes_after_cap": "t"},
            "declarations": {"pre_commitment": True, "dev_reuse": True, "cpu_only": True, "all_outcomes_kept": True, "cpu_nondeterminism": "t",
                             "nominal_pairing": "t", "diagnostic": "t", "limits": "l", "question": "q"},
            "previous_attempt": {k: "t" for k in SR.PREVIOUS_ATTEMPT_KEYS},
            "outputs": {"public": "p", "restricted": "r"}}


@contextlib.contextmanager
def injected(fr=None, **hooks):
    """Punti di iniezione dello script (e di nuaa_folds_run, fr=...), ripristinati all'uscita."""
    saved = {k: getattr(SR, k) for k in hooks}
    saved_fr = {k: getattr(FR, k) for k in (fr or {})}
    try:
        for k, v in hooks.items():
            setattr(SR, k, v)
        for k, v in (fr or {}).items():
            setattr(FR, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(SR, k, v)
        for k, v in saved_fr.items():
            setattr(FR, k, v)


def defaults(world: World, **extra) -> dict:
    hooks = {"REPO_STATE": lambda root: dict(STATE), "TRAINER": fake_trainer(), "MAKE_SCORER": fake_make_scorer,
             "MAKE_LOGITS": fake_make_logits,
             "ARCH_LOADER": lambda p: types.SimpleNamespace(create_model=None), "SCHEMA_CONFIG_SHA256": world.schema_sha,
             "CELLS_CONFIG_SHA256": world.cells_cfg_sha, "fr": {"FRAME_SHAPE": SHAPE}}
    hooks.update(extra)
    return hooks


def run_main(argv) -> tuple:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = SR.main(argv)
    return rc, out.getvalue(), err.getvalue()


def argv_of(w: World, cmd: str, *extra) -> list:
    return [cmd, "--config", str(w.config), "--input-root", str(w.base), "--out-root", str(w.out), "--date", "20261105", *extra]


snapshot = CFR.snapshot


def read_csv(p: Path) -> list:
    with p.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ============================================================================ configurazione

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.w = World()
        self.addCleanup(self.w.close)

    def refused(self, mutate, needle: str):
        doc = self.w.sr_doc()
        mutate(doc)
        p = self.w.write_config(doc, self.w.tmp / "bad.json")
        with self.assertRaises(FR.Failure) as cm:
            SR.load_config(p)
        self.assertIn(needle, str(cm.exception))

    def test_synthetic_configuration_accepted(self):
        cfg, h = SR.load_config(self.w.config)
        self.assertEqual(cfg["arms"]["arm_a_star"]["total"], cfg["arms"]["arm_b"]["total"])
        self.assertEqual(h, sha(self.w.config.read_bytes()))

    def test_unknown_or_missing_top_key_and_schema(self):
        self.refused(lambda d: d.__setitem__("extra", 1), "fields must be exactly")
        self.refused(lambda d: d.pop("reading_rule"), "fields must be exactly")
        self.refused(lambda d: d.__setitem__("schema", "nuaa-folds-run-config/1"), "schema must be")

    def test_arms_fixed(self):
        self.refused(lambda d: d["arms"]["arm_b"].__setitem__("file", "devmix_fold{k}.csv"), "must name the file")
        self.refused(lambda d: d["arms"]["arm_b_prime"].__setitem__("role", "arm_b_prime_fitting"), "must name the file")
        self.refused(lambda d: d["arms"]["arm_a_star"].__setitem__("label", "A"), "must name the file")
        self.refused(lambda d: d.__setitem__("arms", {k: d["arms"][k] for k in ("arm_a_star", "arm_b", "arm_b_prime")}), "in this order")
        self.refused(lambda d: d["arms"].pop("arm_b_prime"), "arms must define exactly")
        self.refused(lambda d: d["arms"]["arm_b"]["cells"].__setitem__("03/bona_fide", 5), "cells")

    def test_arm_totals_and_nesting_counts(self):
        def a_total(d):
            d["arms"]["arm_a_star"]["cells"]["03/attack"] += 1
            d["arms"]["arm_a_star"]["total"] += 1
        self.refused(a_total, "same total")
        self.refused(lambda d: d["arms"]["arm_b"].__setitem__("total", 1), "sum of its cells")

        def bp(d):
            d["arms"]["arm_b_prime"]["cells"]["01/attack"] += 1
            d["arms"]["arm_b_prime"]["total"] += 1
        self.refused(bp, "must equal the session 01-02 cells of arm A*")

        def big(d):
            for arm in ("arm_a_star", "arm_b_prime"):
                d["arms"][arm]["cells"]["01/bona_fide"] = 9
            d["arms"]["arm_a_star"]["cells"]["03/attack"] = 2
            d["arms"]["arm_b_prime"]["total"] = sum(d["arms"]["arm_b_prime"]["cells"].values())
        self.refused(big, "exceeds the cell of arm B")

    def test_dev_is_the_common_dev(self):
        self.refused(lambda d: d["dev"].__setitem__("file", "devmix_fold{k}.csv"), "never dev-mix")
        self.refused(lambda d: d["dev"].__setitem__("role", "dev_mix_alternative"), "never dev-mix")
        self.refused(lambda d: d["dev"]["cells"].__setitem__("03/bona_fide", 4), "dev.cells")
        self.refused(lambda d: d["threshold"].__setitem__("set", "dev fold only"), "common dev")

    def test_training_order_seeds_and_budget(self):
        self.refused(lambda d: d["training_order"].reverse(), "fixed order")
        self.refused(lambda d: d.__setitem__("train_seeds", [42, 43, 45]), "fixed by the coordinator")
        self.refused(lambda d: d["budget"].__setitem__("trainings", 24), "3 arms x k folds")
        self.refused(lambda d: d["budget"].__setitem__("wall_time_cap_minutes", 300), "fixed by the coordinator")
        self.refused(lambda d: d["budget"].__setitem__("on_cap_exceeded", "stop, no results"), "INCOMPLETE")
        self.refused(lambda d: d["budget"].__setitem__("voluntary_stop", "SIGUSR1 gives FAIL"), "budget.voluntary_stop")
        self.refused(lambda d: d["budget"].__setitem__("on_failure", "FAIL, everything removed"), "budget.on_failure")
        self.refused(lambda d: d["budget"].pop("on_failure"), "budget needs exactly")

    def test_comparisons_reading_rule_and_bootstrap_fixed(self):
        self.refused(lambda d: d["comparisons"]["list"].pop(), "comparisons.list")
        self.refused(lambda d: d["comparisons"]["list"][0].__setitem__("minuend", "arm_b_prime"), "comparisons.list")
        self.refused(lambda d: d["reading_rule"].__setitem__("primary_metrics", ["auroc"]), "primary reading")
        self.refused(lambda d: d["reading_rule"].__setitem__("primary_stratum", "all"), "primary reading")
        self.refused(lambda d: d["reading_rule"].__setitem__("rule", "any two seeds"), "rule of the script")
        self.refused(lambda d: d["reading_rule"].__setitem__("statistical_test", "t-test"), "rule of the script")
        self.refused(lambda d: d["bootstrap"].__setitem__("resamples", 500), "bootstrap must be")
        self.refused(lambda d: d["bootstrap"].__setitem__("recipe", "other"), "bootstrap must be")
        self.refused(lambda d: d["reference_a"].__setitem__("label", "reference"), "reference_a.label")

    def test_recipe_threshold_metrics_declarations(self):
        self.refused(lambda d: d["training"].__setitem__("max_epochs", 13), "recipe of the folds run")
        self.refused(lambda d: d["threshold"].__setitem__("criterion", "apcer10"), "EER criterion")
        self.refused(lambda d: d["metrics"].__setitem__("keys", ["auroc"]), "metrics needs")
        self.refused(lambda d: d["scoring"].__setitem__("device", "GPU"), "CPU")
        self.refused(lambda d: d["declarations"].__setitem__("pre_commitment", False), "pre_commitment")
        self.refused(lambda d: d["declarations"].pop("nominal_pairing"), "exactly the keys")
        self.refused(lambda d: d["threshold"].__setitem__("val_loss_check", "clipped probabilities"), "logits of the output layer")
        self.refused(lambda d: d["threshold"].pop("val_loss_check"), "threshold needs exactly")
        self.refused(lambda d: d["previous_attempt"].pop("cause"), "previous_attempt needs exactly")
        self.refused(lambda d: d["previous_attempt"].__setitem__("run_id", ""), "previous_attempt.run_id")
        self.refused(lambda d: d["excluded"].pop("dev_mix"), "excluded")

    def test_counts_tied_to_the_schema(self):
        self.refused(lambda d: d["test"].__setitem__("per_fold", [1, 2, 3, 4]), "test of the schema")
        self.refused(lambda d: d["folds_run"]["frames_file"].__setitem__("n", 3), "n_to_extract")
        self.refused(lambda d: d["schema_expected"].__setitem__("n_to_extract", 1), "inconsistent")

    def test_delivered_configuration(self):
        p = REAL_ROOT / SR.DEFAULT_CONFIG
        if not p.is_file():
            skip_missing(self, "delivered configuration")
        cfg, h = SR.load_config(p)
        self.assertEqual(cfg["arms"]["arm_b"]["cells"], {"01/bona_fide": 350, "01/attack": 350, "02/bona_fide": 350, "02/attack": 350})
        self.assertEqual(list(cfg["arms"]["arm_a_star"]["cells"].values()), [234, 233, 233, 234, 233, 233])
        self.assertEqual(list(cfg["arms"]["arm_b_prime"]["cells"].values()), [234, 233, 233, 234])
        self.assertEqual((cfg["arms"]["arm_b"]["total"], cfg["arms"]["arm_a_star"]["total"], cfg["arms"]["arm_b_prime"]["total"]), (1400, 1400, 934))
        self.assertEqual(cfg["dev"]["total"], 400)
        self.assertEqual(cfg["test"]["per_fold"], [3242, 3114, 3439, 2819])
        self.assertEqual(cfg["budget"]["wall_time_cap_minutes"], 360)
        self.assertEqual(cfg["budget"]["trainings"], 36)
        self.assertEqual(cfg["cells_run"]["restricted_sha256sums_sha256"], "0904ee72546160eadb493a94d8489a8fb21687006cd8fe1970b26007c567723c")
        self.assertEqual(cfg["schema_run"]["config_sha256"], SR.SCHEMA_CONFIG_SHA256)
        self.assertEqual(cfg["cells_run"]["config_sha256"], SR.CELLS_CONFIG_SHA256)
        self.assertEqual(cfg["folds_run"]["frames_file"]["sha256"], "c3fd9903589b7b074c46880870ba87924763f244846c744b657f9aaa3ebeee77")
        self.assertEqual(cfg["architecture"]["sha256"], "a83c02b0788bd6f98262abf00f5fe970abed501f231edba348820482aea64d30")
        self.assertIn("5 November 2026", cfg["authorisation"])
        self.assertEqual(h, SR.CONFIG_SHA256, "CONFIG_SHA256 of the script must be the SHA-256 of the delivered configuration")


# ============================================================================ ingressi

class TestInputs(unittest.TestCase):
    def setUp(self):
        self.w = World()
        self.addCleanup(self.w.close)
        self.cfg, _ = SR.load_config(self.w.config)

    def load(self, cfg=None, **hooks):
        h = {"SCHEMA_CONFIG_SHA256": self.w.schema_sha, "CELLS_CONFIG_SHA256": self.w.cells_cfg_sha, "fr": {"FRAME_SHAPE": SHAPE}}
        h.update(hooks)
        with injected(**h):
            return SR.load_inputs(cfg or self.cfg, self.w.base)

    def refused(self, needle: str, cfg=None, **hooks):
        with self.assertRaises(FR.Failure) as cm:
            self.load(cfg, **hooks)
        self.assertIn(needle, str(cm.exception))

    def cfg_now(self) -> dict:
        return json.loads(json.dumps(self.w.sr_doc()))

    def test_synthetic_inputs_pass(self):
        inp = self.load()
        self.assertEqual(len(inp.cells_sha), 4 * K)
        self.assertNotIn("devmix_fold1.csv", inp.cells_sha)
        self.assertEqual(inp.extracted_keys, self.w.to_extract)
        self.assertEqual(inp.X_extracted.shape, (len(self.w.to_extract), *SHAPE))
        n_rows = K * (sum(B_CELLS.values()) + sum(A_CELLS.values()) + sum(BP_CELLS.values()) + sum(DEV_CELLS.values())) + len(self.w.official)
        src = inp.checks["sources"]
        self.assertEqual(src["rows_from_cache"] + src["rows_from_extracted"], n_rows)
        self.assertTrue(src["rows_from_cache"] > 0 and src["rows_from_extracted"] > 0)
        self.assertEqual(inp.checks["environment"]["tensorflow_state"], "not imported")
        self.assertEqual(inp.checks["frozen_code"]["detector"], "not used (no extraction)")
        self.assertEqual(sorted(inp.code), sorted(("architecture",) + SR.CODE_FILES))
        self.assertEqual(inp.reference["run_id"], FOLDS_RUN)
        for fold in range(1, K + 1):
            self.assertEqual(len(inp.cells[("arm_a_star", fold)]), sum(A_CELLS.values()))

    def test_check_inputs_command_writes_nothing(self):
        before = snapshot(self.w.tmp)
        with injected(**defaults(self.w)):
            rc, out, err = run_main(argv_of(self.w, "check-inputs"))
        self.assertEqual(rc, 0, err)
        self.assertIn("RESULT: PASS (check-inputs)", out)
        self.assertIn("devmix not read", out)
        self.assertEqual(snapshot(self.w.tmp), before)

    def test_cells_run_anchors(self):
        cfg = self.cfg_now()
        cfg["cells_run"]["restricted_sha256sums_sha256"] = "1" * 64
        self.refused("status.json is not PASS", cfg)
        self.w.write_cells_status(status="FAIL")
        self.refused("status.json is not PASS")
        self.w.write_cells_status()
        (self.w.base / "cells" / "public" / "SHA256SUMS").write_text("", encoding="utf-8")
        self.refused("public SHA256SUMS has SHA-256")

    def test_cells_restricted_sums_anchor(self):
        p = self.w.base / "cells" / "restricted" / "SHA256SUMS"
        p.write_text(p.read_text() + "\n", encoding="utf-8")
        self.refused("restricted SHA256SUMS has SHA-256")

    def test_cell_manifest_missing_or_altered(self):
        p = self.w.base / "cells" / "restricted" / "armAstar_fitting_fold2.csv"
        b = p.read_bytes()
        p.write_bytes(b.replace(b"\n", b"\r\n", 1))
        self.refused("armAstar_fitting_fold2.csv: SHA-256")
        p.unlink()
        self.refused("armAstar_fitting_fold2.csv not found")
        p.write_bytes(b)
        sums = self.w.base / "cells" / "restricted" / "SHA256SUMS"
        text = "".join(l + "\n" for l in sums.read_text().splitlines() if not l.endswith("dev_fold3.csv"))
        sums.write_text(text, encoding="utf-8")
        self.w.cells_res = sha(text.encode())
        self.w.write_cells_status()
        self.refused("dev_fold3.csv not listed", self.cfg_now())

    def test_cell_counts_role_and_fold(self):
        rows = [list(r) for r in self.w.cells[("armB_fitting", 1)]]
        self.refused("counts per cell", self.w.rewrite_cell("armB_fitting", 1, rows[:-1]))
        rows = [list(r) for r in self.w.cells[("armB_fitting", 1)]]
        rows[0][1] = "arm_a_star_fitting"
        self.refused("a role other than arm_b_fitting", self.w.rewrite_cell("armB_fitting", 1, rows))
        rows = [list(r) for r in self.w.cells[("armB_fitting", 1)]]
        rows[0][0] = "2"
        self.refused("another fold", self.w.rewrite_cell("armB_fitting", 1, rows))
        self.w.rewrite_cell("armB_fitting", 1, self.w.cells[("armB_fitting", 1)])
        rows = [list(r) for r in self.w.cells[("dev", 2)]]
        self.refused("duplicated keys", self.w.rewrite_cell("dev", 2, rows[:-1] + [rows[0]]))

    def swap_row(self, name: str, fold: int, pick, replacement_key: str) -> dict:
        rows = [list(r) for r in self.w.cells[(name, fold)]]
        i = next(j for j, r in enumerate(rows) if pick(r))
        o = self.w.official[replacement_key]
        rows[i] = [str(fold), rows[i][1], o["key"], o["label"], o["subject"], o["session"], o["sha256"], o["bytes"],
                   "1" if o["key"] in self.w.cache_keys else "0"]
        rows.sort(key=lambda r: r[2])
        return self.w.rewrite_cell(name, fold, rows)

    def test_a_star_outside_b_refused(self):
        # una riga 01/bona fide di A* sostituita da un frame 01/bona fide di S12 non scelto per B
        b_keys = {r[2] for r in self.w.cells[("armB_fitting", 1)]}
        a_keys = {r[2] for r in self.w.cells[("armAstar_fitting", 1)]}
        s12 = {r[4] for r in self.w.cells[("armB_fitting", 1)]}
        spare = next(k for k, o in sorted(self.w.official.items()) if o["subject"] in s12 and o["session"] == "01" and o["label"] == "0"
                     and k not in b_keys and k not in a_keys)
        self.refused("not nested in arm B", self.swap_row("armAstar_fitting", 1, lambda r: r[5] == "01" and r[3] == "0", spare))

    def test_b_prime_differs_from_a_star_refused(self):
        a_keys = {r[2] for r in self.w.cells[("armAstar_fitting", 2)]}
        spare = next(r[2] for r in self.w.cells[("armB_fitting", 2)] if r[5] == "02" and r[3] == "1" and r[2] not in a_keys)
        self.refused("arm B' differs from the session 01-02 part of arm A*",
                     self.swap_row("armBprime_fitting", 2, lambda r: r[5] == "02" and r[3] == "1", spare))

    def test_a_star_session_03_from_outside_s12_refused(self):
        s12 = {r[4] for r in self.w.cells[("armB_fitting", 1)]}
        fit = {r[4] for r in self.w.folds[1]["fitting"]}
        a_keys = {r[2] for r in self.w.cells[("armAstar_fitting", 1)]}
        spare = next(k for k, o in sorted(self.w.official.items()) if o["subject"] in fit - s12 and o["session"] == "03"
                     and o["label"] == "1" and k not in a_keys)
        self.refused("uses subjects outside the S12 subjects", self.swap_row("armAstar_fitting", 1, lambda r: r[5] == "03" and r[3] == "1", spare))

    def test_b_must_be_the_schema_fitting(self):
        b_keys = {r[2] for r in self.w.cells[("armB_fitting", 3)]}
        s12 = {r[4] for r in self.w.cells[("armB_fitting", 3)]}
        spare = next(k for k, o in sorted(self.w.official.items()) if o["subject"] in s12 and o["session"] == "02" and o["label"] == "1"
                     and k not in b_keys)
        self.refused("arm B is not the session 01-02 part", self.swap_row("armB_fitting", 3, lambda r: r[5] == "02" and r[3] == "1", spare))

    def test_dev_with_session_03_or_other_frames_refused(self):
        dev_subj = {r[4] for r in self.w.folds[1]["dev"]}
        k03 = next(k for k, o in sorted(self.w.official.items()) if o["subject"] in dev_subj and o["session"] == "03" and o["label"] == "0")
        rows = [list(r) for r in self.w.cells[("dev", 1)]]
        o = self.w.official[k03]
        rows[0] = ["1", "dev_common", o["key"], o["label"], o["subject"], o["session"], o["sha256"], o["bytes"],
                   "1" if k03 in self.w.cache_keys else "0"]
        doc = self.w.rewrite_cell("dev", 1, sorted(rows, key=lambda r: r[2]))
        doc["dev"]["cells"] = dict(DEV_CELLS)
        self.refused("counts per cell", doc)        # la cella 03 non è fra quelle del dev: rifiutata già dai conteggi
        rows = [list(r) for r in self.w.cells[("dev", 1)]]
        dev_keys = {r[2] for r in rows}
        spare = next(k for k, o in sorted(self.w.official.items()) if o["subject"] in dev_subj and o["session"] == "01" and o["label"] == "0"
                     and k not in dev_keys)
        self.refused("the common dev is not the session 01-02 part", self.swap_row("dev", 1, lambda r: r[5] == "01" and r[3] == "0", spare))

    def test_dev_with_session_03_refused_directly(self):
        inp = self.load()
        cfg = self.cfg_now()
        rows = inp.cells[("dev", 2)]
        r03 = next(r for r in inp.folds[2]["dev"] if r["session"] == "03")
        inp.cells[("dev", 2)] = rows[:-1] + [{**r03, "role": "dev_common", "fold": "2"}]
        with self.assertRaises(FR.Failure) as cm:
            SR.check_cells(cfg, inp)
        self.assertIn("holds session-03 frames", str(cm.exception))

    def test_cell_row_differs_from_official_or_in_cache(self):
        rows = [list(r) for r in self.w.cells[("armBprime_fitting", 4)]]
        rows[0][7] = rows[0][7] + "0"
        self.refused("same label, subject, session, SHA-256 and size", self.w.rewrite_cell("armBprime_fitting", 4, rows))
        rows = [list(r) for r in self.w.cells[("armBprime_fitting", 4)]]
        rows[0][8] = "0" if rows[0][8] == "1" else "1"
        self.refused("in_cache does not match the cache manifests", self.w.rewrite_cell("armBprime_fitting", 4, rows))

    def test_devmix_is_never_read(self):
        p = self.w.base / "cells" / "restricted" / "devmix_fold1.csv"
        p.write_bytes(b"not even a csv")                     # SHA256SUMS non aggiornato: leggerlo darebbe un errore
        inp = self.load()
        self.assertEqual(inp.checks["cells_run"]["not_read"], "devmix_fold<k>.csv (dev-mix excluded by the coordinator)")
        p.unlink()
        self.load()

    def test_cells_configuration_pinned_and_compared(self):
        self.refused("differs from the configuration of the cells run", None, CELLS_CONFIG_SHA256="3" * 64)
        d = json.loads(self.w.cells_cfg.read_text())
        d["design"]["arm_a_star"]["allocation"]["03/attack"] = 4
        self.w.cells_cfg.write_text(json.dumps(d), encoding="utf-8")
        h = sha(self.w.cells_cfg.read_bytes())
        self.w.cells_cfg_sha = h
        self.w.write_cells_status()
        self.refused("the cells of arm_a_star", self.cfg_now(), CELLS_CONFIG_SHA256=h)
        d["design"]["arm_a_star"]["allocation"]["03/attack"] = 5
        d["design"]["dev"]["per_cell"] = 3
        self.w.cells_cfg.write_text(json.dumps(d), encoding="utf-8")
        h = sha(self.w.cells_cfg.read_bytes())
        self.w.cells_cfg_sha = h
        self.w.write_cells_status()
        self.refused("cells of the common dev", self.cfg_now(), CELLS_CONFIG_SHA256=h)

    def test_test_differs_from_schema_refused(self):
        p = self.w.base / "schema" / "fold2_test.csv"
        p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n", 1))
        self.refused("fold2_test.csv: SHA-256")

    def test_schema_configuration_pinned(self):
        self.refused("differs from the approved schema", None, SCHEMA_CONFIG_SHA256="3" * 64)
        doc = json.loads(self.w.schema_cfg.read_text())
        for mutate, needle in ((lambda d: d.__setitem__("train_seeds", [42, 43, 45]), "train_seeds differ"),
                               (lambda d: d["threshold"].__setitem__("criterion", "apcer10"), "threshold criterion"),
                               (lambda d: d["training"].__setitem__("batch", 32), "training fields"),
                               (lambda d: d["folds"].__setitem__("k", 5), "number of folds")):
            dd = json.loads(json.dumps(doc))
            mutate(dd)
            b = (json.dumps(dd, indent=1) + "\n").encode()
            self.w.schema_cfg.write_bytes(b)
            cfg = self.cfg_now()
            cfg["schema_run"]["config_sha256"] = sha(b)
            self.refused(needle, cfg, SCHEMA_CONFIG_SHA256=sha(b))

    def test_folds_run_anchors(self):
        cfg = self.cfg_now()
        cfg["folds_run"]["public_sha256sums_sha256"] = "4" * 64
        self.refused("folds run 20261103-FOLDSRUN-abcdef0: status.json is not PASS", cfg)
        st = self.w.base / "foldsrun" / "status.json"
        st.write_text(json.dumps({**self.w.folds_status, "status": "FAIL"}), encoding="utf-8")
        self.refused("status.json is not PASS")
        st.write_text(json.dumps(self.w.folds_status), encoding="utf-8")
        q = self.w.base / "foldsrun" / "public" / "SHA256SUMS"
        q.write_bytes(q.read_bytes() + b"\n")
        self.refused("folds run 20261103-FOLDSRUN-abcdef0: public SHA256SUMS has SHA-256")
        q.write_bytes(q.read_bytes()[:-1])
        p = self.w.base / "foldsrun" / "restricted" / "extracted_frames.csv"
        p.write_bytes(p.read_bytes() + b"\n")
        self.refused("extracted_frames.csv: SHA-256 differs")

    def test_shared_code_equal_to_the_folds_run(self):
        inp = self.load()
        self.assertEqual(inp.checks["folds_run"]["shared_code_equal_to_folds_run"], sorted(list(SR.SHARED_CODE) + ["architecture"]))
        self.assertEqual(inp.checks["folds_run"]["trainer"], "the trainer of the folds run")
        self.refused("code shared with the folds run differs from its run_record.json (code_sha256): scripts/nuaa_folds_run.py",
                     self.w.write_folds_record({"scripts/nuaa_folds_run.py": "0" * 64}), TRAINER_REVISION_SHA256="1" * 64)
        self.refused("(code_sha256): scripts/m2_writers.py", self.w.write_folds_record({"scripts/m2_writers.py": "0" * 64}))
        self.refused("(code_sha256): architecture", self.w.write_folds_record({SR.FOLDS_RUN_ARCH_KEY: "0" * 64}))
        self.refused("code_sha256 of scripts/nuaa_folds_run.py missing", self.w.write_folds_record(drop=["scripts/nuaa_folds_run.py"]))
        inp = self.load(self.w.write_folds_record(drop=["scripts/path_guard.py"]))      # un file non registrato non è confrontato
        self.assertNotIn("scripts/path_guard.py", inp.checks["folds_run"]["shared_code_equal_to_folds_run"])

    def test_trainer_revision_with_the_operative_scheduler(self):
        """Il trainer può essere quello registrato dal run dei fold oppure la revisione fissata TRAINER_REVISION_SHA256,
        e nessun'altra; gli altri file condivisi restano confrontati con il run dei fold."""
        here = sha((SR.ROOT / "scripts" / "nuaa_folds_run.py").read_bytes())
        cfg = self.w.write_folds_record({"scripts/nuaa_folds_run.py": "0" * 64})
        inp = self.load(cfg, TRAINER_REVISION_SHA256=here)
        self.assertEqual(inp.checks["folds_run"]["trainer"], "revision with the operative scheduler (TRAINER_REVISION_SHA256)")
        self.assertNotIn("scripts/nuaa_folds_run.py", inp.checks["folds_run"]["shared_code_equal_to_folds_run"])
        self.assertIn("scripts/m2_writers.py", inp.checks["folds_run"]["shared_code_equal_to_folds_run"])
        self.refused("(code_sha256): scripts/nuaa_folds_run.py", cfg, TRAINER_REVISION_SHA256="1" * 64)
        self.refused("(code_sha256): scripts/m2_writers.py", self.w.write_folds_record({"scripts/nuaa_folds_run.py": "0" * 64,
                                                                                        "scripts/m2_writers.py": "0" * 64}),
                     TRAINER_REVISION_SHA256=here)
        inp = self.load(self.w.write_folds_record(), TRAINER_REVISION_SHA256=here)          # registrato = revisione: run dei fold
        self.assertEqual(inp.checks["folds_run"]["trainer"], "the trainer of the folds run")
        self.assertIn("scripts/nuaa_folds_run.py", inp.checks["folds_run"]["shared_code_equal_to_folds_run"])

    def test_real_trainer_revision_pinned(self):
        self.assertEqual(SR.TRAINER_REVISION_SHA256, sha((SR.ROOT / "scripts" / "nuaa_folds_run.py").read_bytes()),
                         "TRAINER_REVISION_SHA256 must be the SHA-256 of scripts/nuaa_folds_run.py")

    def test_extracted_frames_csv_rows_checked(self):
        p = self.w.base / "foldsrun" / "restricted" / "extracted_frames.csv"
        rows = read_csv(p)
        rows[3]["frame_sha256"] = "0" * 64
        p.write_text(csv_text(FR.FRAME_COLUMNS, [[r[c] for c in FR.FRAME_COLUMNS] for r in rows]), encoding="utf-8")
        self.w.folds_res = write_sums(p.parent)
        self.w.folds_status["restricted_sums_sha256"] = self.w.folds_res
        (self.w.base / "foldsrun" / "status.json").write_text(json.dumps(self.w.folds_status), encoding="utf-8")
        self.refused("1 frames differ from the frame SHA-256", self.cfg_now())
        rows[3]["frame_sha256"] = FR.frame_hashes(self.w.X_ext[3:4])[0]
        rows[3]["session_id"] = "02" if rows[3]["session_id"] != "02" else "01"
        p.write_text(csv_text(FR.FRAME_COLUMNS, [[r[c] for c in FR.FRAME_COLUMNS] for r in rows]), encoding="utf-8")
        self.w.folds_res = write_sums(p.parent)
        self.w.folds_status["restricted_sums_sha256"] = self.w.folds_res
        (self.w.base / "foldsrun" / "status.json").write_text(json.dumps(self.w.folds_status), encoding="utf-8")
        self.refused("differs from the official lists", self.cfg_now())
        rows.pop(5)
        p.write_text(csv_text(FR.FRAME_COLUMNS, [[r[c] for c in FR.FRAME_COLUMNS] for r in rows]), encoding="utf-8")
        self.w.folds_res = write_sums(p.parent)
        self.w.folds_status["restricted_sums_sha256"] = self.w.folds_res
        (self.w.base / "foldsrun" / "status.json").write_text(json.dumps(self.w.folds_status), encoding="utf-8")
        self.refused("its keys are not the frames outside the cache", self.cfg_now())

    def test_frames_file_sha_members_keys_and_hashes(self):
        cfg = self.cfg_now()
        cfg["folds_run"]["frames_file"]["sha256"] = "5" * 64
        self.refused("extracted frames file frames.npz: SHA-256", cfg)
        X = self.w.X_ext.copy()
        X[2, 3, 3, 1] ^= 1
        self.w.write_frames(X, self.w.to_extract)
        self.refused("1 frames differ from the frame SHA-256 of extracted_frames.csv", self.cfg_now())
        self.w.write_frames(self.w.X_ext, list(reversed(self.w.to_extract)))
        self.refused("keys differ from the frames outside the cache", self.cfg_now())
        self.w.write_frames(self.w.X_ext[:, :, :, :1].copy(), self.w.to_extract)
        self.refused("differ from X", self.cfg_now())
        self.w.frames_file.unlink()
        self.refused("extracted frames file frames.npz not found")

    def test_check_sources_exactly_once(self):
        inp = self.load()
        cfg = self.cfg_now()
        key = next(r["key"] for r in inp.cells[("arm_b", 1)] if r["in_cache"] == "1")
        inp.extracted_keys = inp.extracted_keys + [key]               # trovato due volte
        with self.assertRaises(FR.Failure) as cm:
            SR.check_sources(cfg, inp)
        self.assertIn("found 2 times", str(cm.exception))
        inp = self.load()
        key = next(r["key"] for r in inp.folds[3]["test"] if r["in_cache"] == "0")
        inp.extracted_keys = [k for k in inp.extracted_keys if k != key]   # trovato zero volte
        with self.assertRaises(FR.Failure) as cm:
            SR.check_sources(cfg, inp)
        self.assertIn("found 0 times", str(cm.exception))
        inp = self.load()
        r = next(r for r in inp.cells[("dev", 4)] if r["in_cache"] == "1")
        r["in_cache"] = "0"
        with self.assertRaises(FR.Failure) as cm:
            SR.check_sources(cfg, inp)
        self.assertIn("in_cache does not match the source", str(cm.exception))

    def test_architecture_environment_and_originals(self):
        (self.w.base / "ref" / "architectures.py").write_bytes(b"changed")
        self.refused("architecture file architectures.py")
        (self.w.base / "ref" / "architectures.py").write_bytes(b"content of ref/architectures.py\n")
        cfg = self.cfg_now()
        cfg["environment"]["tensorflow"] = "0.0.0"
        self.refused("TensorFlow", cfg)
        old = os.environ.get("LIVENESSLAB_LIVEDETECTION")
        os.environ["LIVENESSLAB_LIVEDETECTION"] = "/somewhere"
        try:
            self.refused("environment variables")
        finally:
            if old is None:
                os.environ.pop("LIVENESSLAB_LIVEDETECTION", None)
            else:
                os.environ["LIVENESSLAB_LIVEDETECTION"] = old
        with CFR.modules({"tensorflow": types.ModuleType("tensorflow")}):
            self.refused("TensorFlow was imported")
        key = sorted(self.w.official)[9]
        p = self.w.base / "images" / FR.LAYOUTS["images"](self.w.official[key])
        p.write_bytes(p.read_bytes() + b"x")
        self.refused("original image 9 of the sorted official list")

    def test_cache_sha_checked(self):
        cfg = self.cfg_now()
        cfg["cache"]["sha256"] = "2" * 64
        self.refused("cache cache.npz: SHA-256", cfg)


# ============================================================================ ordine, metriche, bootstrap, regola

def rows_spec(spec):
    """spec: (soggetto, sessione, label, score, pred05, preddev) -> righe di punteggio."""
    return [{"sample_id": f"k{i}", "subject_id": s, "session_id": ses, "label": lab, "score_attack": repr(sc), "prediction_05": p5,
             "prediction_dev": pd} for i, (s, ses, lab, sc, p5, pd) in enumerate(spec)]


def random_rows(seed: int, n_subjects: int = 7, per: int = 6, ties: bool = True) -> list:
    rng = np.random.default_rng(seed)
    spec = []
    for i in range(n_subjects):
        for j in range(per):
            lab = "1" if (j + i) % 3 else "0"
            sc = float(np.round(rng.random(), 1 if ties else 6))
            spec.append((f"s{i:02d}", "03", lab, sc, "1" if sc > 0.5 else "0", "1" if sc > 0.35 else "0"))
    return rows_spec(spec)


AGG_CFG = {"train_seeds": SEEDS, "schema_expected": {"k": K}}
AGG_REF = {"run_id": FOLDS_RUN, "aggregate_sha256": "0" * 64, "pooled_per_seed": {}, "bootstrap_per_seed": {}}


def synth_scores(signs: dict, folds=range(1, K + 1), seeds=SEEDS) -> dict:
    """Righe del test per B e A*: per fold due soggetti, sessione 03, due bona fide e due attacchi ciascuno. Con segno +1
    per un seed A* separa perfettamente e B inverte (differenza dell'AUROC +1, intervallo [1, 1]); con -1 il contrario."""
    scores = {}
    for fold in folds:
        base = [(f"k{fold}_{j}_{i}", f"s{fold}{j}", lab) for j in range(2) for i, lab in enumerate("0011")]
        for seed in seeds:
            for arm in ("arm_b", "arm_a_star"):
                good = (arm == "arm_a_star") == (signs[seed] > 0)
                rows = []
                for key, subj, lab in base:
                    v = (0.9 if lab == "1" else 0.1) if good else (0.1 if lab == "1" else 0.9)
                    p = "1" if v > 0.5 else "0"
                    rows.append({"arm": arm, "fold": str(fold), "train_seed": str(seed), "role": "test", "sample_id": key, "subject_id": subj,
                                 "session_id": "03", "label": lab, "score_attack": repr(v), "prediction_05": p, "prediction_dev": p,
                                 "threshold_dev": "0.5", "file_sha256": "0" * 64, "checkpoint_sha256": "1" * 64, "threshold_sha256": "2" * 64})
                scores[(arm, fold, seed)] = rows
    return scores


class TestOrderMetricsRule(unittest.TestCase):
    def test_training_order(self):
        order = SR.training_order(4, [42, 43, 44])
        self.assertEqual(len(order), 36)
        self.assertEqual(order[:4], [("arm_b", 1, 42), ("arm_a_star", 1, 42), ("arm_b", 1, 43), ("arm_a_star", 1, 43)])
        self.assertEqual([x[0] for x in order[:24]], ["arm_b", "arm_a_star"] * 12)
        self.assertEqual(order[24:], [("arm_b_prime", f, s) for f in range(1, 5) for s in (42, 43, 44)])
        self.assertEqual(len(set(order)), 36)

    def test_weighted_metrics_equal_frame_metrics(self):
        for seed in range(4):
            rows = random_rows(seed)
            st = SR.subject_stats(rows)
            n = len(st["subjects"])
            self.assertEqual(SR.weighted_metrics(st, np.ones(n, np.int64)), {k: FR.stratum_metrics(rows)[k] for k in FR.METRIC_KEYS})
            rng = np.random.default_rng(seed + 10)
            for _ in range(15):
                pick = rng.integers(0, n, n)
                sample = [r for i in pick for r in rows if r["subject_id"] == st["subjects"][i]]
                exp = FR.stratum_metrics(sample)
                got = SR.weighted_metrics(st, np.bincount(pick, minlength=n))
                self.assertEqual(got, {k: exp[k] for k in FR.METRIC_KEYS})

    def test_arm_bootstrap_equals_the_folds_run_bootstrap(self):
        rows = random_rows(5, ties=False)
        with injected(BOOT={**SR.BOOT, "resamples": 200}), contextlib.ExitStack():
            FR_BOOT = FR.BOOT
            FR.BOOT = {**FR.BOOT, "resamples": 200}
            try:
                a = SR.arm_bootstrap(rows, ("acer_dev", "acer_05", "auroc"))
                b = FR.bootstrap(rows, ("acer_dev", "acer_05", "auroc"))
            finally:
                FR.BOOT = FR_BOOT
        self.assertEqual(a, b)

    def test_paired_bootstrap_known_answers(self):
        subj = [f"s{i}" for i in range(6)]
        good = rows_spec([(s, "03", lab, 0.9 if lab == "1" else 0.1, "1" if lab == "1" else "0", "1" if lab == "1" else "0")
                          for s in subj for lab in ("0", "1")])
        bad = rows_spec([(s, "03", lab, 0.1 if lab == "1" else 0.9, "0" if lab == "1" else "1", "0" if lab == "1" else "1")
                         for s in subj for lab in ("0", "1")])
        pb = SR.paired_bootstrap(good, bad)
        self.assertEqual((pb["n_subjects"], pb["resamples"], pb["seed"]), (6, 1000, 0))
        self.assertEqual((pb["auroc"]["low"], pb["auroc"]["high"], pb["auroc"]["valid_resamples"]), (1.0, 1.0, 1000))
        self.assertEqual((pb["acer_05"]["low"], pb["acer_05"]["high"]), (-1.0, -1.0))
        self.assertEqual((pb["apcer_dev"]["low"], pb["bpcer_dev"]["high"]), (-1.0, -1.0))
        same = SR.paired_bootstrap(good, good)
        self.assertEqual((same["auroc"]["low"], same["auroc"]["high"], same["acer_dev"]["low"]), (0.0, 0.0, 0.0))

    def test_paired_bootstrap_uses_the_same_indices(self):
        m, s = random_rows(1, n_subjects=6, ties=False), random_rows(2, n_subjects=6, ties=False)
        for r, q in zip(m, s):                            # stessi frame, punteggi diversi
            q.update({"subject_id": r["subject_id"], "label": r["label"], "sample_id": r["sample_id"]})
        with injected(BOOT={**SR.BOOT, "resamples": 40}):
            pb = SR.paired_bootstrap(m, s)
            picks = SR.boot_picks(6)
        subjects = sorted({r["subject_id"] for r in m})
        vals = []
        for pick in picks:
            a = FR.stratum_metrics([r for i in pick for r in m if r["subject_id"] == subjects[i]])["auroc"]
            b = FR.stratum_metrics([r for i in pick for r in s if r["subject_id"] == subjects[i]])["auroc"]
            vals.append(a - b)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        self.assertEqual((pb["auroc"]["low"], pb["auroc"]["high"]), (float(lo), float(hi)))
        rng = np.random.default_rng(0)
        self.assertTrue(all(np.array_equal(p, rng.integers(0, 6, 6)) for p in picks))

    def test_paired_bootstrap_refusals_few_subjects_and_missing_class(self):
        m = random_rows(1, n_subjects=6)
        with self.assertRaises(FR.Failure):
            SR.paired_bootstrap(m, m[:-1])
        few = SR.paired_bootstrap(random_rows(3, n_subjects=4), random_rows(3, n_subjects=4))
        self.assertIn("no interval", few["note"])
        self.assertIsNone(few["auroc"]["low"])
        few_arm = SR.arm_bootstrap(random_rows(3, n_subjects=4))
        self.assertIn("fewer than 5 subjects", few_arm["note"])
        self.assertEqual((few_arm["auroc"]["low"], few_arm["acer_dev"]["valid_resamples"]), (None, 0))
        one_bf = rows_spec([(f"s{i}", "03", "1", 0.6, "1", "1") for i in range(5)] + [("s5", "03", "0", 0.2, "0", "0")])
        with injected(BOOT={**SR.BOOT, "resamples": 30}):
            pb = SR.paired_bootstrap(one_bf, one_bf)
        self.assertLess(pb["auroc"]["valid_resamples"], 30)
        self.assertGreater(pb["auroc"]["valid_resamples"], 0)
        self.assertEqual(pb["apcer_05"]["valid_resamples"], 30)

    def test_same_frames_and_difference(self):
        a = random_rows(1, n_subjects=5)
        b = [dict(r) for r in a]
        b[0]["sample_id"] = "other"
        with self.assertRaises(FR.Failure):
            SR.same_frames(a, b)
        SR.same_frames(a, [dict(r) for r in a])
        self.assertIsNone(SR.difference(None, 0.1))
        self.assertAlmostEqual(SR.difference(0.3, 0.1), 0.2)

    def test_reading_rule_every_branch(self):
        comp = SR.COMPARISONS[0]
        e = lambda d, lo, hi: {"difference": d, "low": lo, "high": hi}   # noqa: E731
        v = SR.reading_verdict(comp, "auroc", [e(0.1, 0.02, 0.2), e(0.2, 0.05, 0.3), e(0.05, 0.01, 0.1)])
        self.assertEqual((v["verdict"], v["direction"]), ("consistent", "A* higher AUROC than B"))
        self.assertEqual(SR.verdict_text(v), "consistent: A* higher AUROC than B")
        v = SR.reading_verdict(comp, "acer_dev", [e(-0.1, -0.2, -0.02), e(-0.2, -0.3, -0.05), e(-0.05, -0.1, -0.01)])
        self.assertEqual((v["verdict"], v["direction"]), ("consistent", "A* lower ACER (dev threshold) than B"))
        v = SR.reading_verdict(SR.COMPARISONS[1], "auroc", [e(-0.1, -0.2, -0.02)] * 3)
        self.assertEqual(v["direction"], "B' lower AUROC than B")
        cases = [([e(0.1, 0.02, 0.2), None, e(0.1, 0.02, 0.2)], "no pooled value"),
                 ([e(0.1, 0.02, 0.2), e(None, None, None), e(0.1, 0.02, 0.2)], "no difference"),
                 ([e(0.1, 0.02, 0.2), e(0.1, None, None), e(0.1, 0.02, 0.2)], "no bootstrap interval"),
                 ([e(0.1, 0.02, 0.2), e(0.0, -0.1, 0.1), e(0.1, 0.02, 0.2)], "zero difference"),
                 ([e(0.1, 0.02, 0.2), e(-0.1, -0.2, -0.02), e(0.1, 0.02, 0.2)], "sign differs"),
                 ([e(0.1, 0.02, 0.2), e(0.1, 0.02, 0.2), e(-0.1, -0.2, -0.02)], "sign differs"),
                 ([e(-0.1, -0.2, -0.02), e(0.1, 0.02, 0.2), e(0.1, 0.02, 0.2)], "sign differs"),
                 ([e(0.1, 0.02, 0.2), e(0.1, -0.01, 0.2), e(0.1, 0.02, 0.2)], "does not exclude 0"),
                 ([e(0.1, 0.02, 0.2), e(0.1, -0.3, -0.01), e(0.1, 0.02, 0.2)], "does not exclude 0"),
                 ([e(-0.1, -0.2, -0.02), e(-0.1, 0.01, 0.3), e(-0.1, -0.2, -0.02)], "does not exclude 0"),
                 ([e(0.1, 0.0, 0.2)] * 3, "does not exclude 0")]
        for per_seed, reason in cases:
            v = SR.reading_verdict(comp, "auroc", per_seed)
            self.assertEqual((v["verdict"], v["direction"]), ("inconclusive", None), reason)
            self.assertIn(reason, v["reason"])
            self.assertEqual(SR.verdict_text(v), "inconclusive")

    def test_score_rows_carry_the_arm_and_tie_rule(self):
        part = {"rows": [{"key": "real/1101_1101_00_00_03_0.jpg", "subject": "1101", "session": "03", "label": "0", "sha256": "a" * 64}]}
        thr, card = {"value": 0.5, "sha256": "b" * 64}, {"weights_sha256": "c" * 64}
        row = SR.score_rows("arm_b", 1, 42, "test", part, np.array([0.5]), thr, card)[0]
        self.assertEqual((row["arm"], row["prediction_05"], row["prediction_dev"]), ("arm_b", "0", "0"))
        row = SR.score_rows("arm_b", 1, 42, "test", part, np.array([np.nextafter(0.5, 1)]), thr, card)[0]
        self.assertEqual((row["prediction_05"], row["prediction_dev"]), ("1", "1"))
        self.assertEqual(list(SR.SCORE_COLUMNS[:2]), ["arm", "fold"])

    def test_budget_raises_cap_exceeded(self):
        t = {"now": 0.0}
        with injected(CLOCK=lambda: t["now"]):
            b = SR.Budget(1)
            b.check("a")
            t["now"] = 61.0
            with self.assertRaises(SR.CapExceeded) as cm:
                b.check("train x")
        self.assertEqual(cm.exception.step, "train x")
        self.assertFalse(issubclass(SR.CapExceeded, FR.Failure))

    def test_budget_elapsed_uses_the_wall_clock_when_the_monotonic_stops(self):
        # macOS: il monotono non avanza durante lo stop della macchina, l'orologio di sistema sì (max dei due)
        t = {"wall": 1000.0}
        with injected(CLOCK=lambda: 5.0, time=types.SimpleNamespace(time=lambda: t["wall"])):
            b = SR.Budget(1)
            t["wall"] += 59.0
            b.check("a")
            self.assertEqual(b.elapsed(), 59.0)
            t["wall"] += 2.0
            with self.assertRaises(SR.CapExceeded) as cm:
                b.check("train y")
        self.assertEqual((cm.exception.step, cm.exception.cause), ("train y", "cap"))
        t = {"now": 0.0}
        with injected(CLOCK=lambda: t["now"], time=types.SimpleNamespace(time=lambda: 1000.0)):
            b = SR.Budget(1)
            t["now"] = 61.0
            with self.assertRaises(SR.CapExceeded):
                b.check("train z")

    def test_budget_stop_requested(self):
        with injected(CLOCK=lambda: 0.0):
            b = SR.Budget(1)
            b.check("a")
            b.stop_requested = "SIGUSR1"
            with self.assertRaises(SR.CapExceeded) as cm:
                b.check("epoch 3 of m")
        self.assertEqual((cm.exception.step, cm.exception.cause), ("epoch 3 of m", "SIGUSR1"))
        self.assertIn("stop requested by SIGUSR1 at epoch 3 of m", str(cm.exception))

    def test_reading_rule_on_the_three_seeds_in_the_aggregate(self):
        # seed 44 di segno opposto: la regola deve guardare tutti e tre i seed
        agg, _ = SR.aggregate(AGG_CFG, synth_scores({42: 1, 43: 1, 44: -1}), {}, {}, AGG_REF)
        c = agg["comparisons"]["a_star_minus_b"]
        self.assertEqual([c["pooled_per_seed"][str(s)]["03"]["auroc"]["difference"] for s in SEEDS], [1.0, 1.0, -1.0])
        self.assertEqual([c["pooled_per_seed"][str(s)]["03"]["auroc"]["low"] for s in SEEDS], [1.0, 1.0, -1.0])
        for st in ("03", "all"):
            v = c["verdicts"][st]["auroc"]
            self.assertEqual((v["verdict"], v["reason"]), ("inconclusive", "the sign differs across the seeds"), st)
        self.assertEqual(agg["primary_verdicts"]["auroc"]["verdict"], "inconclusive")
        agg, _ = SR.aggregate(AGG_CFG, synth_scores({42: 1, 43: 1, 44: 1}), {}, {}, AGG_REF)     # controllo: tre seed concordi
        self.assertEqual(agg["primary_verdicts"]["auroc"]["direction"], "A* higher AUROC than B")
        agg, _ = SR.aggregate(AGG_CFG, synth_scores({42: -1, 43: 1, 44: 1}), {}, {}, AGG_REF)    # il segno non è quello del primo seed
        self.assertEqual(agg["primary_verdicts"]["auroc"]["reason"], "the sign differs across the seeds")

    def test_pairing_checked_per_fold_and_seed(self):
        s = synth_scores({42: 1, 43: 1, 44: 1}, folds=(1,), seeds=(42,))
        agg, _ = SR.aggregate(AGG_CFG, s, {}, {}, AGG_REF)                          # controllo: stessi frame, nessun pooled
        self.assertEqual(sorted(agg["comparisons"]["a_star_minus_b"]["per_fold_seed"]["1"]), ["42"])
        self.assertEqual(agg["comparisons"]["a_star_minus_b"]["pooled_per_seed"], {})
        s[("arm_a_star", 1, 42)][0] = {**s[("arm_a_star", 1, 42)][0], "sample_id": "other"}
        with self.assertRaises(FR.Failure) as cm:
            SR.aggregate(AGG_CFG, s, {}, {}, AGG_REF)
        self.assertIn("not scored on the same test frames", str(cm.exception))

    def test_bootstrap_five_subjects_boundary(self):
        five = random_rows(3, n_subjects=5)
        pb = SR.paired_bootstrap(five, five)
        self.assertEqual((pb["n_subjects"], pb["auroc"]["low"], pb["auroc"]["valid_resamples"]), (5, 0.0, 1000))
        self.assertNotIn("note", pb)
        ab = SR.arm_bootstrap(five)
        self.assertEqual(ab["n_subjects"], 5)
        self.assertIsNotNone(ab["auroc"]["low"])
        self.assertNotIn("note", ab)
        four = random_rows(3, n_subjects=4)
        self.assertIsNone(SR.paired_bootstrap(four, four)["auroc"]["low"])
        self.assertIsNone(SR.arm_bootstrap(four)["auroc"]["low"])

    def test_real_trainer_stops_at_the_cap_without_weights(self):
        log = {}
        tf, arch = CFR.fake_tf(log, lambda e: [0.9, 0.8, 0.7])
        tmp = Path(tempfile.mkdtemp(prefix="sessablrun-trainer-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        rng = np.random.default_rng(0)
        data = {"X_fit": rng.integers(0, 256, (8, *SHAPE), dtype=np.uint8), "y_fit": np.array([0, 1] * 4),
                "X_dev": rng.integers(0, 256, (4, *SHAPE), dtype=np.uint8), "y_dev": np.array([0, 1, 0, 1])}
        t = {"now": 0.0}

        def clock():
            t["now"] += 61.0
            return t["now"]
        with CFR.modules({"tensorflow": tf}), injected(CLOCK=clock):
            b = SR.Budget(1)
            with self.assertRaises(SR.CapExceeded) as cm:
                FR.default_trainer(data, dict(FR.FROZEN_TRAINING), tmp / "w.keras", 42, arch, lambda step: b.check(f"{step} of m"))
        self.assertEqual(cm.exception.step, "epoch 1 of m")
        self.assertEqual(os.listdir(tmp), [])


# ============================================================================ run completo

class TestRun(unittest.TestCase):
    def setUp(self):
        self.w = World()
        self.addCleanup(self.w.close)
        Log.calls = []

    def run_cmd(self, *extra, **hooks):
        with injected(**defaults(self.w, **hooks)):
            return run_main(argv_of(self.w, "run", "--i-am-authorized", *extra))

    def run_ok(self, **hooks):
        rc, out, err = self.run_cmd(**hooks)
        self.assertEqual(rc, 0, out + err)
        self.assertIn("RESULT: PASS (run)", out)
        return self.w.out / RUN, out

    def test_run_outputs_status_and_sums(self):
        run, out = self.run_ok()
        status = json.loads((run / "status.json").read_text())
        self.assertEqual((status["status"], status["passed"], status["missing"], status["stopped"]), ("PASS", True, [], None))
        self.assertEqual(len(status["completed"]), N_MODELS)
        pub, res = run / "public", run / "restricted"
        self.assertEqual(sorted(os.listdir(pub)), sorted(SR.PUBLIC_OUTPUTS))
        cfg, _ = SR.load_config(self.w.config)
        self.assertEqual(sorted(os.listdir(res)), sorted(SR.restricted_outputs(cfg)))
        for d in (pub, res):
            listed = {}
            for line in (d / "SHA256SUMS").read_text().splitlines():
                h, n = line.split("  ")
                self.assertEqual(FR.sha256_file(d / n), h, n)
                listed[n] = h
            self.assertEqual(sorted(listed), sorted(n for n in os.listdir(d) if n != "SHA256SUMS"))
        self.assertEqual((res / "public_SHA256SUMS.sha256").read_text(), sha((pub / "SHA256SUMS").read_bytes()) + "  public/SHA256SUMS\n")
        self.assertEqual(status["public_sums_sha256"], sha((pub / "SHA256SUMS").read_bytes()))
        self.assertEqual(status["restricted_sums_sha256"], sha((res / "SHA256SUMS").read_bytes()))
        self.assertIn("RESULT: PASS (run)", (run / "run.log").read_text())
        self.assertRegex((run / "run.log").read_text(), r"== arm_b_fold1_seed42: test \d+/\d+, .*, peak RSS \d+ MB\n")
        self.assertEqual(status["run_log_sha256"], FR.sha256_file(run / "run.log"))
        self.assertIs(status["delivered_configuration"], False)
        self.assertIsNone(status["configuration_name"])
        self.assertNotIn("amendment", status)
        self.assertIs(status["final_writes_after_cap"], False)
        rec = json.loads((pub / "run_record.json").read_text())
        self.assertEqual(rec["completed"], [SR.model_name(*x) for x in SR.training_order(K, SEEDS)])
        self.assertIs(rec["final_writes_after_cap"], False)
        self.assertEqual(set(rec["timings"]["check-inputs"]), {"monotonic_seconds", "wall_clock_seconds"})

    def test_training_order_arms_and_common_dev(self):
        self.run_ok()
        trains = [c for c in Log.calls if c[0] == "train"]
        self.assertEqual([c[1] for c in trains], [f"{SR.model_name(*x)}_LivenessNet.keras" for x in SR.training_order(K, SEEDS)])
        n_fit = {"arm_b": sum(B_CELLS.values()), "arm_a_star": sum(A_CELLS.values()), "arm_b_prime": sum(BP_CELLS.values())}
        dev_by_fold = {}
        for _, name, nf, nd, dev_sha in trains:
            arm = next(a for a in sorted(SR.ARMS, key=len, reverse=True) if name.startswith(a + "_fold"))
            fold = int(name[len(arm) + 5])
            self.assertEqual((nf, nd), (n_fit[arm], sum(DEV_CELLS.values())), name)
            dev_by_fold.setdefault(fold, set()).add(dev_sha)
        self.assertTrue(all(len(v) == 1 for v in dev_by_fold.values()))         # lo stesso dev comune per i tre bracci

    def test_threshold_frozen_before_test_and_scores_consistent(self):
        run, _ = self.run_ok()
        res = run / "restricted"
        for arm, fold, seed in SR.training_order(K, SEEDS):
            name = SR.model_name(arm, fold, seed)
            thr = json.loads((res / f"{name}_threshold.json").read_text())
            card = json.loads((res / f"{name}_card.json").read_text())
            self.assertEqual((thr["arm"], thr["fold"], thr["train_seed"], thr["criterion"]), (arm, fold, seed, FR.THRESHOLD_CRITERION))
            self.assertIn("common dev", thr["set"])
            self.assertLessEqual(thr["frozen_at"], card["test_scored_after_threshold"])
            thr_sha = sha((res / f"{name}_threshold.json").read_bytes())
            test, dev = read_csv(res / f"{name}_test_scores.csv"), read_csv(res / f"{name}_dev_scores.csv")
            self.assertEqual(len(test), len(self.w.folds[fold]["test"]))
            self.assertEqual(sorted(r["sample_id"] for r in dev), sorted(r[2] for r in self.w.cells[("dev", fold)]))
            self.assertTrue(all(r["threshold_sha256"] == thr_sha and r["checkpoint_sha256"] == card["weights_sha256"] and r["arm"] == arm
                                for r in test + dev))
            y = np.array([int(r["label"]) for r in dev])
            s = np.array([float(r["score_attack"]) for r in dev])
            sel = FR.eer_threshold(y, s)
            self.assertEqual((thr["value"], thr["fp_rejected_bona_fide"], thr["fn_accepted_attacks"], thr["objective"], thr["n_attack"],
                              thr["n_bona_fide"]), (sel["value"], sel["fp"], sel["fn"], sel["objective"], sel["n_attack"], sel["n_bona_fide"]))
            for r in test:
                v = float(r["score_attack"])
                self.assertEqual((r["prediction_05"], r["prediction_dev"]), ("1" if v > 0.5 else "0", "1" if v > thr["value"] else "0"))
            self.assertEqual({r["subject_id"] for r in test}, set(CFR.FOLDS[fold]))
            self.assertEqual((card["arm"], card["fold"], card["n_fitting"]), (arm, fold, self.w.sr_doc()["arms"][arm]["total"]))
            exp = fake_scores(np.stack([CFR.fake_frame(r["sample_id"].encode())[0] for r in test]), arm, seed)[:, 1]
            self.assertTrue(np.allclose([float(r["score_attack"]) for r in test], exp))

    def test_aggregate_and_comparisons_consistent_with_the_scores(self):
        run, _ = self.run_ok()
        agg = json.loads((run / "public" / "aggregate.json").read_text())
        res = run / "restricted"
        rows = {(a, f, s): read_csv(res / f"{SR.model_name(a, f, s)}_test_scores.csv") for a, f, s in SR.training_order(K, SEEDS)}
        for (a, f, s), rr in rows.items():
            for st in FR.STRATA:
                part = FR.rows_of_stratum(rr, st)
                got = agg["per_arm_fold_seed"][a][str(f)][str(s)][st]
                self.assertEqual(got, FR.stratum_metrics(part) if part else None)
        for name, m, s_, role in SR.COMPARISONS:
            c = agg["comparisons"][name]
            self.assertEqual((c["minuend"], c["subtrahend"], c["role"]), (m, s_, role))
            for f in range(1, K + 1):
                for seed in SEEDS:
                    for st in FR.STRATA:
                        a, b = FR.rows_of_stratum(rows[(m, f, seed)], st), FR.rows_of_stratum(rows[(s_, f, seed)], st)
                        d = c["per_fold_seed"][str(f)][str(seed)][st]
                        if not a:
                            self.assertIsNone(d)
                            continue
                        ma, mb = FR.stratum_metrics(a), FR.stratum_metrics(b)
                        self.assertEqual(d, {k: SR.difference(ma[k], mb[k]) for k in FR.METRIC_KEYS})
            for seed in SEEDS:
                pm = [r for f in range(1, K + 1) for r in rows[(m, f, seed)]]
                ps = [r for f in range(1, K + 1) for r in rows[(s_, f, seed)]]
                for st in FR.STRATA:
                    e = c["pooled_per_seed"][str(seed)][st]
                    ma, mb = FR.stratum_metrics(FR.rows_of_stratum(pm, st)), FR.stratum_metrics(FR.rows_of_stratum(ps, st))
                    for k in FR.METRIC_KEYS:
                        self.assertEqual((e[k]["minuend"], e[k]["subtrahend"], e[k]["difference"]), (ma[k], mb[k], SR.difference(ma[k], mb[k])))
                    self.assertEqual(e["n_subjects"], 16 if st in ("03", "all") else 9)
            for st in FR.STRATA:
                for k in FR.METRIC_KEYS:
                    per_seed = [c["pooled_per_seed"][str(seed)][st][k] for seed in SEEDS]
                    v = c["verdicts"][st][k]
                    self.assertEqual({x: v[x] for x in ("verdict", "direction", "reason")}, SR.reading_verdict((name, m, s_, role), k, per_seed))
                    self.assertEqual(v["label"], "primary" if (name, st) == ("a_star_minus_b", "03") and k in ("auroc", "acer_dev") else "secondary")
        # bootstrap appaiato del primario ricalcolato sui frame (stessi indici per i due bracci), seed 43, strato 03
        pm = sorted((r for f in range(1, K + 1) for r in FR.rows_of_stratum(rows[("arm_a_star", f, 43)], "03")), key=lambda r: r["sample_id"])
        ps = sorted((r for f in range(1, K + 1) for r in FR.rows_of_stratum(rows[("arm_b", f, 43)], "03")), key=lambda r: r["sample_id"])
        subjects = sorted({r["subject_id"] for r in pm})
        by_m = {x: [r for r in pm if r["subject_id"] == x] for x in subjects}
        by_s = {x: [r for r in ps if r["subject_id"] == x] for x in subjects}
        vals = []
        for pick in SR.boot_picks(len(subjects)):
            a = FR.stratum_metrics([r for i in pick for r in by_m[subjects[i]]])["acer_dev"]
            b = FR.stratum_metrics([r for i in pick for r in by_s[subjects[i]]])["acer_dev"]
            vals.append(a - b)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        e = agg["comparisons"]["a_star_minus_b"]["pooled_per_seed"]["43"]["03"]["acer_dev"]
        self.assertEqual((e["low"], e["high"], e["valid_resamples"]), (float(lo), float(hi), 1000))
        # bootstrap per braccio uguale a quello del run dei fold
        b = FR.bootstrap(FR.rows_of_stratum(sorted((r for f in range(1, K + 1) for r in rows[("arm_b", f, 42)]), key=lambda r: r["sample_id"]), "all"),
                         ("acer_dev", "acer_05", "auroc"))
        got = agg["bootstrap_per_arm_seed"]["arm_b"]["42"]["all"]
        self.assertEqual({k: got[k] for k in ("acer_dev", "acer_05", "auroc", "n_subjects")}, {k: b[k] for k in ("acer_dev", "acer_05", "auroc", "n_subjects")})
        with (res / "per_subject.csv").open(newline="") as fh:
            ps_rows = list(csv.DictReader(fh))
        self.assertEqual({r["arm"] for r in ps_rows}, set(SR.ARMS))
        one = next(r for r in ps_rows if (r["arm"], r["fold"], r["train_seed"], r["subject_id"], r["stratum"]) == ("arm_a_star", "1", "42", "1101", "03"))
        self.assertEqual(float(one["auroc"]), FR.stratum_metrics([r for r in rows[("arm_a_star", 1, 42)]
                                                                  if r["subject_id"] == "1101" and r["session_id"] == "03"])["auroc"])

    def test_primary_verdict_and_reference(self):
        run, out = self.run_ok()
        agg = json.loads((run / "public" / "aggregate.json").read_text())
        v = agg["primary_verdicts"]["auroc"]
        self.assertEqual((v["verdict"], v["direction"], v["label"]), ("consistent", "A* higher AUROC than B", "primary"))
        self.assertIn("primary AUROC (A* - B, stratum 03): consistent: A* higher AUROC than B", out)
        ref = agg["reference_a"]
        self.assertEqual((ref["label"], ref["run_id"]), (SR.REFERENCE_LABEL, FOLDS_RUN))
        self.assertEqual([ref["pooled_per_seed"][str(s)]["03"]["auroc"] for s in SEEDS], [0.9, 0.91, 0.92])
        self.assertEqual([ref["bootstrap_per_seed"][str(s)]["all"]["auroc"]["low"] for s in SEEDS], [0.8, 0.81, 0.82])
        md = (run / "public" / "aggregate.md").read_text()
        self.assertIn("## Reference A (reference, different fitting frames and dev: indicative)", md)
        self.assertIn("## Primary verdicts (A* - B, stratum 03)", md)
        self.assertIn("For each comparison" + SR.READING_RULE[len("for each comparison"):] + ".", md)

    def test_public_files_have_no_identifiers(self):
        run, _ = self.run_ok()
        cfg, _ = SR.load_config(self.w.config)
        with injected(**defaults(self.w)):
            inp = SR.load_inputs(cfg, self.w.base)
        for n in ("aggregate.json", "aggregate.md", "run_record.json"):
            text = (run / "public" / n).read_text()
            self.assertEqual(FR.public_problems(text, inp, n.endswith(".md")), [], n)
            for s in SUBJECTS:
                self.assertNotIn(f'"{s}"', text, n)

    def test_public_scan_blocks_identifiers(self):
        old = SR.aggregate_md

        def leaky(cfg, agg, record, run_id):
            return old(cfg, agg, record, run_id) + "\nsubject 1101 was hard\n"
        rc, out, err = self.run_cmd(aggregate_md=leaky)
        self.assertEqual(rc, 1)
        self.assertIn("public file aggregate.md refused", err)
        run = self.w.out / RUN
        status = json.loads((run / "status.json").read_text())
        order = [SR.model_name(*x) for x in SR.training_order(K, SEEDS)]
        self.assert_kept(run, status, order)

    def cap_run(self, when, **hooks):
        """Run con l'orologio che salta di 400 minuti quando when(evento, nome) è vero (eventi: train, epoch, score)."""
        t = {"now": 0.0}

        def clock():
            t["now"] += 1.0
            return t["now"]

        def epoch_hook(name, e):
            if when("epoch", name, e):
                t["now"] += 400 * 60
        base_trainer = fake_trainer(epoch_hook=epoch_hook)

        def trainer(data, training, weights_path, seed, arch, cap_check):
            card = base_trainer(data, training, weights_path, seed, arch, cap_check)
            if when("train", Path(weights_path).name, None):
                t["now"] += 400 * 60
            return card

        def make(weights_path, weights_sha):
            inner = fake_make_scorer(weights_path, weights_sha)
            calls = {"n": 0}

            def score(X):
                calls["n"] += 1
                out = inner(X)
                if when("score", Path(weights_path).name, calls["n"]):
                    t["now"] += 400 * 60
                return out
            return score
        rc, out, err = self.run_cmd(CLOCK=clock, TRAINER=trainer, MAKE_SCORER=make, **hooks)
        return rc, out, err, self.w.out / RUN

    def test_cap_mid_run_gives_incomplete_with_complete_models_kept(self):
        target = "arm_a_star_fold2_seed43_LivenessNet.keras"
        rc, out, err, run = self.cap_run(lambda ev, name, x: ev == "train" and name == target)
        self.assertEqual(rc, 2, out + err)
        self.assertIn("RESULT: INCOMPLETE (run)", out)
        status = json.loads((run / "status.json").read_text())
        order = [SR.model_name(*x) for x in SR.training_order(K, SEEDS)]
        self.assertEqual((status["status"], status["passed"]), ("INCOMPLETE", False))
        self.assertEqual(status["completed"], order[:9])
        self.assertEqual(status["missing"], order[9:])
        self.assertEqual((status["stopped"]["model"], status["stopped"]["step"], status["stopped"]["cause"]),
                         ("arm_a_star_fold2_seed43", "threshold arm_a_star_fold2_seed43", "cap"))
        self.assertIs(status["final_writes_after_cap"], True)
        self.assertIn("arm_a_star_fold2_seed43_LivenessNet.keras", status["stopped"]["removed_files"])
        res = run / "restricted"
        self.assertEqual(sorted(os.listdir(res)), sorted([f"{n}_{x}" for n in order[:9] for x in SR.PER_MODEL] + list(SR.RESTRICTED_FIXED)))
        listed = [l.split("  ")[1] for l in (res / "SHA256SUMS").read_text().splitlines()]
        self.assertEqual(sorted(listed), sorted(n for n in os.listdir(res) if n != "SHA256SUMS"))
        agg = json.loads((run / "public" / "aggregate.json").read_text())
        self.assertEqual((agg["status"], agg["completed"], agg["trainings"]), ("INCOMPLETE", 9, N_MODELS))
        self.assertEqual(sorted(agg["training"]), sorted(order[:9]))
        self.assertEqual(agg["pooled_per_arm_seed"], {})
        self.assertEqual(sorted(agg["comparisons"]["a_star_minus_b"]["per_fold_seed"]), ["1", "2"])
        self.assertEqual(sorted(agg["comparisons"]["a_star_minus_b"]["per_fold_seed"]["2"]), ["42"])
        v = agg["primary_verdicts"]["auroc"]
        self.assertEqual((v["verdict"], v["reason"]), ("inconclusive", "a seed has no pooled value (incomplete run)"))
        rec = json.loads((run / "public" / "run_record.json").read_text())
        self.assertIs(rec["final_writes_after_cap"], True)
        md = (run / "public" / "aggregate.md").read_text()
        self.assertIn("Outcome: **INCOMPLETE**, 9 of 36 trainings complete; stopped at threshold arm_a_star_fold2_seed43", md)
        self.assertIn("RESULT: INCOMPLETE (run)", (run / "run.log").read_text())
        self.assertEqual(status["run_log_sha256"], FR.sha256_file(run / "run.log"))
        self.assertEqual((res / "public_SHA256SUMS.sha256").read_text(), sha((run / "public" / "SHA256SUMS").read_bytes()) + "  public/SHA256SUMS\n")

    def test_cap_at_the_end_of_an_epoch_interrupts_the_training(self):
        rc, out, err, run = self.cap_run(lambda ev, name, x: ev == "epoch" and name == "arm_b_fold1_seed43_LivenessNet.keras" and x == 2)
        self.assertEqual(rc, 2, out + err)
        status = json.loads((run / "status.json").read_text())
        self.assertEqual(status["stopped"]["step"], "epoch 2 of arm_b_fold1_seed43")
        self.assertEqual(status["completed"], ["arm_b_fold1_seed42", "arm_a_star_fold1_seed42"])
        self.assertFalse(any(e.startswith("arm_b_fold1_seed43") for e in os.listdir(run / "restricted")))
        self.assertNotIn(("train", "arm_b_fold1_seed43_LivenessNet.keras"), [c[:2] for c in Log.calls])

    def test_cap_before_the_test_scoring(self):
        rc, out, err, run = self.cap_run(lambda ev, name, x: ev == "score" and name == "arm_b_fold3_seed42_LivenessNet.keras" and x == 1)
        self.assertEqual(rc, 2, out + err)
        status = json.loads((run / "status.json").read_text())
        self.assertEqual(status["stopped"]["step"], "score arm_b_fold3_seed42")
        self.assertEqual(sorted(status["stopped"]["removed_files"]),
                         sorted(f"arm_b_fold3_seed42_{x}" for x in ("LivenessNet.keras", "threshold.json", "dev_scores.csv")))
        self.assertEqual(len(status["completed"]), 12)

    def test_cap_after_the_primary_pairs_keeps_the_primary_comparison(self):
        rc, out, err, run = self.cap_run(lambda ev, name, x: ev == "score" and name == "arm_a_star_fold4_seed44_LivenessNet.keras" and x == 2)
        self.assertEqual(rc, 2, out + err)
        status = json.loads((run / "status.json").read_text())
        self.assertEqual((len(status["completed"]), status["stopped"]["step"], status["stopped"]["removed_files"]),
                         (24, "train arm_b_prime_fold1_seed42", []))
        agg = json.loads((run / "public" / "aggregate.json").read_text())
        self.assertEqual(agg["primary_verdicts"]["auroc"]["verdict"], "consistent")
        self.assertEqual(sorted(agg["pooled_per_arm_seed"]), ["arm_a_star", "arm_b"])
        self.assertEqual(agg["comparisons"]["b_prime_minus_b"]["verdicts"]["03"]["auroc"]["reason"], "a seed has no pooled value (incomplete run)")
        self.assertEqual(agg["comparisons"]["b_prime_minus_b"]["pooled_per_seed"], {})

    def test_cap_mid_fold_4_pools_only_the_complete_seeds(self):
        rc, out, err, run = self.cap_run(lambda ev, name, x: ev == "score" and name == "arm_a_star_fold4_seed42_LivenessNet.keras" and x == 2)
        self.assertEqual(rc, 2, out + err)
        status = json.loads((run / "status.json").read_text())
        self.assertEqual((len(status["completed"]), status["stopped"]["step"], status["stopped"]["removed_files"]),
                         (20, "train arm_b_fold4_seed43", []))
        agg = json.loads((run / "public" / "aggregate.json").read_text())
        self.assertEqual({a: sorted(v) for a, v in agg["pooled_per_arm_seed"].items()}, {"arm_a_star": ["42"], "arm_b": ["42"]})
        self.assertEqual(agg["pooled_summary_over_seeds"], {})
        c = agg["comparisons"]["a_star_minus_b"]
        self.assertEqual((sorted(c["pooled_per_seed"]), c["pooled_summary_over_seeds"]), (["42"], {}))
        self.assertEqual(sorted(c["per_fold_seed"]["4"]), ["42"])
        for name, *_ in SR.COMPARISONS:
            for st in FR.STRATA:
                for k in FR.METRIC_KEYS:
                    v = agg["comparisons"][name]["verdicts"][st][k]
                    self.assertEqual((v["verdict"], v["reason"]), ("inconclusive", "a seed has no pooled value (incomplete run)"), (name, st, k))
        self.assertIn("Not available (incomplete run).", (run / "public" / "aggregate.md").read_text())

    def test_final_writes_after_cap_measured_before_the_aggregate(self):
        # il tetto è superato durante lo scoring dell'ultimo modello, dopo il suo controllo: 36/36, PASS, scritture oltre il tetto
        rc, out, err, run = self.cap_run(lambda ev, name, x: ev == "score" and name == "arm_b_prime_fold4_seed44_LivenessNet.keras" and x == 2)
        self.assertEqual(rc, 0, out + err)
        status = json.loads((run / "status.json").read_text())
        self.assertEqual((status["status"], status["stopped"], status["final_writes_after_cap"]), ("PASS", None, True))
        rec = json.loads((run / "public" / "run_record.json").read_text())
        self.assertIs(rec["final_writes_after_cap"], True)
        self.assertIn("cap already exceeded before them: yes", (run / "public" / "aggregate.md").read_text())

    def test_cap_removes_a_temporary_checkpoint_left_by_the_trainer(self):
        t = {"now": 0.0}
        base = fake_trainer()

        def trainer(data, training, weights_path, seed, arch, cap_check):
            if Path(weights_path).name == "arm_a_star_fold1_seed43_LivenessNet.keras":
                Path(weights_path).with_name(f".{Path(weights_path).name}.4242.tmp.keras").write_bytes(b"partial")
                t["now"] += 400 * 60
                cap_check("epoch 1")
            return base(data, training, weights_path, seed, arch, cap_check)
        rc, out, err = self.run_cmd(CLOCK=lambda: t["now"], TRAINER=trainer)
        self.assertEqual(rc, 2, out + err)
        status = json.loads((self.w.out / RUN / "status.json").read_text())
        self.assertEqual(status["stopped"]["removed_files"], [".arm_a_star_fold1_seed43_LivenessNet.keras.4242.tmp.keras"])
        self.assertFalse(any("arm_a_star_fold1_seed43" in e for e in os.listdir(self.w.out / RUN / "restricted")))

    def test_cap_before_the_first_training(self):
        rc, out, err, run = self.cap_run(lambda ev, name, x: False)
        self.assertEqual(rc, 0, out + err)            # controllo: con lo stesso orologio, senza salti, il run è completo
        shutil.rmtree(self.w.out)
        t = {"now": 0.0}

        def clock():
            t["now"] += 400 * 60
            return t["now"]
        rc, out, err = self.run_cmd(CLOCK=clock)
        self.assertEqual(rc, 2, out + err)
        status = json.loads((self.w.out / RUN / "status.json").read_text())
        self.assertEqual((status["completed"], status["stopped"]["step"]), ([], "train arm_b_fold1_seed42"))
        agg = json.loads((self.w.out / RUN / "public" / "aggregate.json").read_text())
        self.assertEqual((agg["training"], agg["per_arm_fold_seed"]), ({}, {}))
        self.assertIn("No complete model.", (self.w.out / RUN / "public" / "aggregate.md").read_text())

    def test_sigterm_stops_the_run_with_fail(self):
        saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
        base = fake_trainer()

        def killing_trainer(data, training, weights_path, seed, arch, cap_check):
            if Path(weights_path).name.startswith("arm_a_star_fold1_seed43"):
                if signal.getsignal(signal.SIGTERM) in (signal.SIG_DFL, signal.SIG_IGN, None):
                    raise AssertionError("SIGTERM handler not installed")
                os.kill(os.getpid(), signal.SIGTERM)
            return base(data, training, weights_path, seed, arch, cap_check)
        try:
            rc, out, err = self.run_cmd(TRAINER=killing_trainer)
        finally:
            for sig, h in saved.items():
                signal.signal(sig, h)
        self.assertEqual(rc, 1)
        self.assertIn("signal SIGTERM received", err)
        status = json.loads((self.w.out / RUN / "status.json").read_text())
        self.assert_kept(self.w.out / RUN, status, ["arm_b_fold1_seed42", "arm_a_star_fold1_seed42", "arm_b_fold1_seed43"])
        self.assertIn("signal SIGTERM received", (self.w.out / RUN / "run.log").read_text())

    def test_sigusr1_stops_like_the_cap(self):
        saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1)}
        base = fake_trainer()

        def stopping_trainer(data, training, weights_path, seed, arch, cap_check):
            if Path(weights_path).name.startswith("arm_b_fold2_seed43"):
                if signal.getsignal(signal.SIGUSR1) in (signal.SIG_DFL, signal.SIG_IGN, None):
                    raise AssertionError("SIGUSR1 handler not installed")
                os.kill(os.getpid(), signal.SIGUSR1)
            return base(data, training, weights_path, seed, arch, cap_check)
        try:
            rc, out, err = self.run_cmd(TRAINER=stopping_trainer)
        finally:
            for sig, h in saved.items():
                signal.signal(sig, h)
        self.assertEqual(rc, 2, out + err)
        self.assertIn("RESULT: INCOMPLETE (run)", out)
        run = self.w.out / RUN
        status = json.loads((run / "status.json").read_text())
        order = [SR.model_name(*x) for x in SR.training_order(K, SEEDS)]
        self.assertEqual((status["status"], status["completed"], status["missing"]), ("INCOMPLETE", order[:8], order[8:]))
        self.assertEqual((status["stopped"]["model"], status["stopped"]["step"], status["stopped"]["cause"]),
                         ("arm_b_fold2_seed43", "epoch 1 of arm_b_fold2_seed43", "SIGUSR1"))
        self.assertIs(status["final_writes_after_cap"], False)
        self.assertEqual(sorted(os.listdir(run / "restricted")), sorted([f"{n}_{x}" for n in order[:8] for x in SR.PER_MODEL] + list(SR.RESTRICTED_FIXED)))
        rec = json.loads((run / "public" / "run_record.json").read_text())
        self.assertIs(rec["final_writes_after_cap"], False)
        self.assertIn("(0.0 min, stop requested by SIGUSR1)", (run / "public" / "aggregate.md").read_text())
        self.assertIn("== STOP: stop requested by SIGUSR1 at epoch 1 of arm_b_fold2_seed43", (run / "run.log").read_text())

    def test_sighup_handler_not_installed_when_ignored(self):
        saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
        try:
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            SR.signal_handlers()
            self.assertIs(signal.getsignal(signal.SIGHUP), signal.SIG_IGN)
            self.assertIsNot(signal.getsignal(signal.SIGTERM), saved[signal.SIGTERM])
        finally:
            for sig, h in saved.items():
                signal.signal(sig, h)

    def test_existing_status_dirty_tree_authorisation_and_foreign_entries(self):
        with injected(**defaults(self.w)):
            rc, out, err = run_main(argv_of(self.w, "run"))
        self.assertEqual(rc, 1)
        self.assertIn("--i-am-authorized", err)
        self.assertFalse(self.w.out.exists())
        rc, out, err = self.run_cmd(REPO_STATE=lambda root: {"commit": "abcdef0", "dirty": True})
        self.assertEqual(rc, 0, err)
        self.assertTrue((self.w.out / (RUN + "-dev") / "status.json").is_file())
        rc, out, err = self.run_cmd(REPO_STATE=lambda root: {"commit": "abcdef0", "dirty": True})
        self.assertEqual(rc, 1)
        self.assertIn("already holds status.json", err)
        run = self.w.out / RUN
        (run / "restricted").mkdir(parents=True)
        (run / "restricted" / "devmix_fold1.csv").write_text("x")
        rc, out, err = self.run_cmd()
        self.assertEqual(rc, 1)
        self.assertIn("not an output of this command", err)
        self.assertFalse((run / "status.json").exists())
        os.unlink(run / "restricted" / "devmix_fold1.csv")
        leftover = run / "restricted" / ".per_subject.csv.abc.tmp"
        leftover.write_text("tmp")
        self.run_ok()
        self.assertFalse(leftover.exists())

    def test_output_root_symlink_refused(self):
        real = self.w.tmp / "real_out"
        real.mkdir()
        self.w.out.symlink_to(real)
        rc, out, err = self.run_cmd()
        self.assertEqual(rc, 1)
        self.assertIn("symbolic link", err)

    def assert_kept(self, run: Path, status: dict, kept: list):
        """FAIL tardivo: pubblico e per_subject.csv tolti, i cinque file dei modelli completi conservati con un SHA256SUMS
        riservato solo su quelli, nessun aggregato."""
        self.assertEqual((status["status"], status["passed"], status["kept_complete_models"]), ("FAIL", False, kept))
        self.assertEqual(status["outputs"], f"public removed; restricted outputs of {len(kept)} complete models kept, not aggregated; no verdict")
        self.assertEqual(os.listdir(run / "public") if (run / "public").is_dir() else [], [])
        res = run / "restricted"
        self.assertEqual(sorted(os.listdir(res)), sorted([f"{n}_{x}" for n in kept for x in SR.PER_MODEL] + ["SHA256SUMS"]))
        listed = {}
        for line in (res / "SHA256SUMS").read_text().splitlines():
            h, n = line.split("  ")
            self.assertEqual(FR.sha256_file(res / n), h, n)
            listed[n] = h
        self.assertEqual(sorted(listed), sorted(f"{n}_{x}" for n in kept for x in SR.PER_MODEL))
        self.assertEqual(status["restricted_sums_sha256"], sha((res / "SHA256SUMS").read_bytes()))
        self.assertIn("FAIL outputs: public removed; restricted outputs of", (run / "run.log").read_text())
        self.assertEqual(status["run_log_sha256"], FR.sha256_file(run / "run.log"))

    def test_failures_give_fail_and_keep_complete_models(self):
        base = fake_trainer()

        def bad(data, training, weights_path, seed, arch, cap_check):
            if Path(weights_path).name.startswith("arm_b_fold2"):
                FR.fail("synthetic training failure")
            return base(data, training, weights_path, seed, arch, cap_check)
        rc, out, err = self.run_cmd(TRAINER=bad)
        self.assertEqual(rc, 1)
        run = self.w.out / RUN
        status = json.loads((run / "status.json").read_text())
        self.assertIn("synthetic training failure", status["reason"])
        order = [SR.model_name(*x) for x in SR.training_order(K, SEEDS)]
        self.assert_kept(run, status, order[:6])
        self.assertIn("synthetic training failure", (run / "run.log").read_text())

    def test_failure_at_the_first_model_removes_everything(self):
        base = fake_trainer()

        def bad(data, training, weights_path, seed, arch, cap_check):
            card = base(data, training, weights_path, seed, arch, cap_check)      # pesi scritti, poi l'errore
            FR.fail("synthetic failure after the weights")
        rc, out, err = self.run_cmd(TRAINER=bad)
        self.assertEqual(rc, 1)
        run = self.w.out / RUN
        status = json.loads((run / "status.json").read_text())
        self.assertEqual((status["status"], status["outputs"], status["kept_complete_models"], status["restricted_sums_sha256"]),
                         ("FAIL", "removed", [], "none"))
        self.assertEqual(os.listdir(run / "restricted"), [])

    def test_failure_after_the_public_writes_keeps_only_the_models(self):
        old = FR.write_status

        def failing(run, doc):
            if doc["status"] == "PASS":
                raise OSError("synthetic: status.json not writable")
            return old(run, doc)
        rc, out, err = self.run_cmd(fr={"FRAME_SHAPE": SHAPE, "write_status": failing})
        self.assertEqual(rc, 1)
        run = self.w.out / RUN
        status = json.loads((run / "status.json").read_text())
        self.assertIn("status.json not writable", status["reason"])
        self.assert_kept(run, status, [SR.model_name(*x) for x in SR.training_order(K, SEEDS)])

    def test_val_loss_card_counts_and_bad_scorer_refused(self):
        base = fake_trainer()

        def wrong(data, training, weights_path, seed, arch, cap_check):
            card = base(data, training, weights_path, seed, arch, cap_check)
            card["history"]["val_loss"] = [v + 0.05 for v in card["history"]["val_loss"]]
            card["best_val_loss"] += 0.05
            return card
        rc, out, err = self.run_cmd(TRAINER=wrong)
        self.assertEqual(rc, 1)
        self.assertIn("not the one of minimum val_loss", err)
        self.assertEqual(os.listdir(self.w.out / RUN / "restricted"), [])          # i pesi del modello in errore tolti
        self.assertEqual(json.loads((self.w.out / RUN / "status.json").read_text())["outputs"], "removed")
        shutil.rmtree(self.w.out)

        def short(data, training, weights_path, seed, arch, cap_check):
            card = base(data, training, weights_path, seed, arch, cap_check)
            return {**card, "n_fitting": card["n_fitting"] - 1}
        rc, out, err = self.run_cmd(TRAINER=short)
        self.assertEqual(rc, 1)
        self.assertIn("fitting and", err)
        shutil.rmtree(self.w.out)
        def gpu(data, training, weights_path, seed, arch, cap_check):
            return {**base(data, training, weights_path, seed, arch, cap_check), "device": "GPU"}
        rc, out, err = self.run_cmd(TRAINER=gpu)
        self.assertEqual(rc, 1)
        self.assertIn("differ from the protocol", err)
        shutil.rmtree(self.w.out)
        rc, out, err = self.run_cmd(MAKE_SCORER=lambda w, h: (lambda X: np.ones((len(X), 2)) * 2))
        self.assertEqual(rc, 1)
        self.assertIn("probabilities expected", err)

    def test_val_loss_check_on_logits(self):
        """Un frame del dev con logit saturi (perdita 40): il controllo con le probabilità limitate avrebbe fallito,
        quello dai logit (come Keras) passa; un checkpoint diverso (logit e probabilità di un altro modello) fallisce."""
        def sat_scorer(weights_path, weights_sha):
            arm, seed = parse_weights(Path(weights_path).read_bytes())
            return lambda X: SR.softmax(saturated_logits(X, arm, seed))

        def sat_logits(weights_path, weights_sha):
            arm, seed = parse_weights(Path(weights_path).read_bytes())
            return lambda X: saturated_logits(X, arm, seed)
        trainer = fake_trainer(loss=lambda X, y, a, s: SR.val_loss_logits(saturated_logits(X, a, s), y))
        rc, out, err = self.run_cmd(TRAINER=trainer, MAKE_SCORER=sat_scorer, MAKE_LOGITS=sat_logits)
        self.assertEqual(rc, 0, out + err)
        run = self.w.out / RUN
        for arm, fold, seed in SR.training_order(K, SEEDS):
            card = json.loads((run / "restricted" / f"{SR.model_name(arm, fold, seed)}_card.json").read_text())
            self.assertAlmostEqual(card["val_loss_recomputed"], card["best_val_loss"], places=9)
            self.assertEqual(card["val_loss_check"], SR.VAL_LOSS_CHECK)
            gap = abs(card["val_loss_recomputed_clipped"] - card["best_val_loss"])
            # il vecchio controllo (probabilità limitate) qui avrebbe dato FAIL
            self.assertGreater(gap, max(FR.VAL_LOSS_TOL[0], FR.VAL_LOSS_TOL[1] * card["best_val_loss"]))
        shutil.rmtree(self.w.out)

        def other_scorer(weights_path, weights_sha):            # checkpoint sbagliato: un altro modello (braccio diverso)
            arm, seed = parse_weights(Path(weights_path).read_bytes())
            other = "arm_a_star" if arm != "arm_a_star" else "arm_b"
            return lambda X: SR.softmax(saturated_logits(X, other, seed))

        def other_logits(weights_path, weights_sha):
            arm, seed = parse_weights(Path(weights_path).read_bytes())
            other = "arm_a_star" if arm != "arm_a_star" else "arm_b"
            return lambda X: saturated_logits(X, other, seed)
        rc, out, err = self.run_cmd(TRAINER=trainer, MAKE_SCORER=other_scorer, MAKE_LOGITS=other_logits)
        self.assertEqual(rc, 1, out + err)
        self.assertIn("not the one of minimum val_loss", err)
        self.assertEqual(os.listdir(self.w.out / RUN / "restricted"), [])
        shutil.rmtree(self.w.out)
        # logit non coerenti con le probabilità dello scorer (strato sbagliato o altro file): rifiutati
        rc, out, err = self.run_cmd(MAKE_LOGITS=lambda w, h: (lambda X: np.zeros((len(X), 2))))
        self.assertEqual(rc, 1, out + err)
        self.assertIn("differs from the scorer probabilities", err)
        shutil.rmtree(self.w.out)
        rc, out, err = self.run_cmd(MAKE_LOGITS=lambda w, h: (lambda X: np.full((len(X), 2), np.nan)))
        self.assertEqual(rc, 1, out + err)
        self.assertIn("non-finite", err)

    def test_val_loss_logits_formula(self):
        z = np.array([[0.0, 40.0], [2.0, -1.0], [1000.0, 0.0]])
        y = np.array([0, 0, 1])
        want = np.mean([np.log(np.exp(0.0) + np.exp(40.0)) - 0.0, np.log(np.exp(2.0) + np.exp(-1.0)) - 2.0, 1000.0])
        self.assertAlmostEqual(SR.val_loss_logits(z, y), want, places=9)
        self.assertGreater(SR.val_loss_logits(z, y) - FR.val_loss(SR.softmax(z), y), 1.0)   # il clip la sottostima
        p = np.array([[0.3, 0.7], [0.9, 0.1]])
        self.assertAlmostEqual(SR.val_loss_logits(np.log(p), np.array([1, 0])), FR.val_loss(p, np.array([1, 0])), places=12)

    def test_threshold_read_back_is_compared(self):
        old = FR.read_regular

        def tampered(p, what):
            b = old(p, what)
            if Path(p).name.endswith("_threshold.json"):
                return b.replace(b'"set": "common dev', b'"set": "COMMON dev')
            return b
        rc, out, err = self.run_cmd(read_regular=tampered)
        self.assertEqual(rc, 1)
        self.assertIn("does not read back identical", err)

    def test_architecture_changed_before_loading(self):
        def loader(p):
            raise AssertionError("must not be loaded")
        old = SR.load_inputs

        def load_then_change(cfg, base):
            inp = old(cfg, base)
            (base / "ref" / "architectures.py").write_bytes(b"changed after the check")
            return inp
        rc, out, err = self.run_cmd(ARCH_LOADER=loader, load_inputs=load_then_change)
        self.assertEqual(rc, 1)
        self.assertIn("changed just before loading", err)

    def test_run_id_format(self):
        with self.assertRaises(FR.Failure):
            SR.run_id_for("2026115", STATE)
        self.assertEqual(SR.run_id_for("20261105", {"commit": "abcdef0", "dirty": True}), RUN + "-dev")
        self.assertTrue(SR.RUN_ID_RE.match(RUN))


# ============================================================================ configurazione emendata (pazienza 12)

AMENDMENT_TEXT = {"authorisation": "test agreement", "agreed_precisions": ["first precision", "second precision"], "note": "n",
                  "time_estimate": "t", "shared_history_check": "s"}


def amended_doc(parent: dict, parent_sha: str, patience: int = 12, **over) -> dict:
    """La configurazione emendata del mondo: il genitore con la sola pazienza cambiata e la chiave amendment in fondo."""
    d = json.loads(json.dumps(parent))
    d["training"]["early_stopping_patience"] = patience
    d["amendment"] = {"parent_config_sha256": parent_sha, "changed_field": "training.early_stopping_patience", "from": 4, "to": 12, **AMENDMENT_TEXT,
                      **over}
    return d


class TestAmendment(unittest.TestCase):
    """Configurazione emendata: SHA-256 fissato, genitore = configurazione consegnata, ogni altro campo uguale."""

    def setUp(self):
        self.w = World()
        self.addCleanup(self.w.close)
        self.parent = self.w.sr_doc()
        self.parent_p = self.w.write_config(self.parent, self.w.tmp / "parent.json")
        self.parent_sha = sha(self.parent_p.read_bytes())
        self.child_p = self.w.tmp / "patience12.json"
        self.write_child(amended_doc(self.parent, self.parent_sha))

    def write_child(self, doc: dict) -> str:
        self.w.write_config(doc, self.child_p)
        self.child_sha = sha(self.child_p.read_bytes())
        return self.child_sha

    def hooks(self, **over) -> dict:
        h = {"CONFIG_SHA256": self.parent_sha, "PATIENCE12_CONFIG_SHA256": self.child_sha, "PARENT_CONFIG": str(self.parent_p)}
        h.update(over)
        return defaults(self.w, **h)

    def argv(self, cmd: str, *extra) -> list:
        a = argv_of(self.w, cmd, *extra)
        a[a.index("--config") + 1] = str(self.child_p)
        return a

    def cmd(self, cmd: str, *extra, **over) -> tuple:
        with injected(**self.hooks(**over)):
            return run_main(self.argv(cmd, *extra))

    def refused_both(self, needle: str, **over):
        """check-inputs e run rifiutano con lo stesso messaggio; il run non crea la cartella."""
        rc, out, err = self.cmd("check-inputs", **over)
        self.assertEqual(rc, 1, out + err)
        self.assertIn(needle, err)
        self.assertIn("RESULT: FAIL (check-inputs)", out)
        rc, out, err = self.cmd("run", "--i-am-authorized", **over)
        self.assertEqual(rc, 1, out + err)
        self.assertIn(needle, err)
        self.assertFalse((self.w.out / RUN).exists())

    def load_refused(self, doc: dict, needle: str):
        p = self.w.write_config(doc, self.w.tmp / "bad_amended.json")
        with self.assertRaises(FR.Failure) as cm:
            SR.load_config(p)
        self.assertIn(needle, str(cm.exception))

    # ---------------------------------------------------------------- file reali di delivery/ (nessun dato)

    def real_pair(self) -> tuple:
        cp, pp = REAL_ROOT / SR.PATIENCE12_CONFIG, REAL_ROOT / SR.DEFAULT_CONFIG
        if not cp.is_file() or not pp.is_file():
            skip_missing(self, "patience12 configuration")
        return cp, pp

    def test_real_patience12_configuration_pinned(self):
        cp, pp = self.real_pair()
        cfg, h = SR.load_config(cp)
        self.assertEqual(h, sha(cp.read_bytes()))
        self.assertEqual(h, SR.PATIENCE12_CONFIG_SHA256, "PATIENCE12_CONFIG_SHA256 must be the SHA-256 of the amended configuration")
        self.assertEqual(cfg["amendment"]["parent_config_sha256"], SR.CONFIG_SHA256)
        self.assertEqual(sha(pp.read_bytes()), SR.CONFIG_SHA256)
        self.assertEqual((cfg["amendment"]["changed_field"], cfg["amendment"]["from"], cfg["amendment"]["to"]),
                         ("training.early_stopping_patience", 4, 12))
        self.assertEqual(cfg["training"]["early_stopping_patience"], 12)
        self.assertEqual(cfg["training"]["max_epochs"], 12)
        self.assertEqual(cfg["budget"]["wall_time_cap_minutes"], 360)
        self.assertIn("7 November 2026", cfg["amendment"]["authorisation"])
        self.assertIn("B fold 3 seed 44", cfg["amendment"]["agreed_precisions"][0])
        self.assertEqual((SR.configuration_name(h), SR.configuration_name(SR.CONFIG_SHA256)),
                         ("patience12_verification", "session_ablation_run"))

    def test_real_patience12_differs_only_in_the_allowed_field(self):
        cp, pp = self.real_pair()
        child, parent = json.loads(cp.read_bytes()), json.loads(pp.read_bytes())
        self.assertIsNone(SR.config_difference(parent, child, SR.amendment_allowed(SR.AMENDMENTS[0])))
        self.assertEqual(SR.amendment_allowed(SR.AMENDMENTS[0]), ("training.early_stopping_patience", "amendment"))
        self.assertEqual(SR.config_difference(parent, child, ()), "training.early_stopping_patience")
        self.assertEqual(SR.config_difference(parent, child, ("training.early_stopping_patience",)), "amendment")
        self.assertEqual([k for k in child if k != "amendment"], list(parent), "same order of the keys as the parent")
        self.assertEqual(list(child["training"]), list(parent["training"]))
        self.assertEqual((parent["training"]["early_stopping_patience"], child["training"]["early_stopping_patience"]), (4, 12))
        self.assertEqual((parent["task"], parent["authorisation"], parent["budget"]["estimate"]),
                         (child["task"], child["authorisation"], child["budget"]["estimate"]))
        cfg, h = SR.load_config(cp)
        got = SR.check_amendment(cfg, h)
        self.assertEqual((got["parent_config_sha256"], got["from"], got["to"]), (SR.CONFIG_SHA256, 4, 12))
        self.assertIsNone(SR.check_amendment(*SR.load_config(pp)))

    # ---------------------------------------------------------------- config_difference

    def test_config_difference(self):
        a = {"x": 1, "training": {"lr": 0.001, "patience": 4, "aug": ["flip", "zoom"]}, "flag": True}
        cd = SR.config_difference
        self.assertIsNone(cd(a, json.loads(json.dumps(a))))
        self.assertIsNone(cd(a, {"flag": True, "training": {"aug": ["flip", "zoom"], "patience": 4, "lr": 0.001}, "x": 1}), "order")
        b = json.loads(json.dumps(a))
        b["training"]["lr"] = 0.002
        self.assertEqual(cd(a, b), "training.lr")
        b = json.loads(json.dumps(a))
        b["training"].pop("lr")
        self.assertEqual(cd(a, b), "training.lr", "missing field")
        b = json.loads(json.dumps(a))
        b["training"]["extra"] = 0
        self.assertEqual(cd(a, b), "training.extra", "extra field")
        b = json.loads(json.dumps(a))
        b["training"]["aug"][1] = "rotate"
        self.assertEqual(cd(a, b), "training.aug.1", "nested value")
        b = json.loads(json.dumps(a))
        b["training"]["aug"].append("noise")
        self.assertEqual(cd(a, b), "training.aug.2", "longer list")
        self.assertEqual(cd(b, a), "training.aug.2", "shorter list")
        self.assertEqual(cd(a, {**a, "flag": 1}), "flag", "true is not 1")
        self.assertEqual(cd(a, {**a, "x": 1.0}), "x", "1 is not 1.0")
        self.assertEqual(cd(a, {**a, "training": [1]}), "training")
        self.assertEqual(cd(1, 2), "<root>")
        self.assertEqual(cd(a, {**a, "z": 0}), "z")
        b = json.loads(json.dumps(a))
        b["training"]["patience"] = 12
        b["amendment"] = {"to": 12}
        self.assertEqual(cd(a, b), "training.patience")
        self.assertEqual(cd(a, b, ("training.patience",)), "amendment")
        self.assertIsNone(cd(a, b, ("training.patience", "amendment")))
        b["training"]["lr"] = 0.01
        self.assertEqual(cd(a, b, ("training.patience", "amendment")), "training.lr", "an allowed field does not hide another")
        self.assertEqual(cd({"training": {"p": 4, "q": 1}}, {"training": {"p": 12, "q": 2}}, ("training.p",)), "training.q")
        self.assertEqual(cd({"training": {"p": 4, "pq": 1}}, {"training": {"p": 12, "pq": 2}}, ("training.p",)), "training.pq",
                         "allowed paths are exact, not prefixes")
        self.assertEqual(cd({"training": {"amendment": 1}}, {"training": {"amendment": 2}}, ("amendment",)), "training.amendment",
                         "allowed paths are full dotted paths, not key names")

    # ---------------------------------------------------------------- check-inputs e run

    def test_amended_configuration_accepted(self):
        rc, out, err = self.cmd("check-inputs")
        self.assertEqual(rc, 0, out + err)
        self.assertIn(f"AMENDMENT: configuration patience12_verification {self.child_sha[:12]}, parent {self.parent_sha[:12]} equal in "
                      "every field except training.early_stopping_patience (4 -> 12) and amendment", out)
        self.assertIn("RESULT: PASS (check-inputs)", out)
        with injected(**self.hooks()):                    # la configurazione genitore: nessuna riga AMENDMENT, come prima
            rc, out, err = run_main(argv_of(self.w, "check-inputs", "--config", str(self.parent_p)))
        self.assertEqual(rc, 0, out + err)
        self.assertNotIn("AMENDMENT", out)

    def test_amendment_hash_not_pinned_refused(self):
        self.refused_both("differs from the agreed configuration", PATIENCE12_CONFIG_SHA256="3" * 64)

    def test_parent_hash_declared_differs_refused(self):
        self.write_child(amended_doc(self.parent, "5" * 64))
        self.refused_both("parent_config_sha256 differs from the delivered configuration")

    def test_parent_file_differs_refused(self):
        self.write_child(amended_doc(self.parent, "6" * 64))
        self.refused_both("parent configuration parent.json: SHA-256", CONFIG_SHA256="6" * 64)

    def test_other_field_difference_refused(self):
        d = amended_doc(self.parent, self.parent_sha)
        d["task"] = "another task"
        self.write_child(d)
        self.refused_both("field task differs from the parent configuration (only training.early_stopping_patience 4 -> 12 and "
                          "amendment may differ)")
        d = amended_doc(self.parent, self.parent_sha)
        d["budget"]["estimate"] = "another estimate"
        self.write_child(d)
        self.refused_both("field budget.estimate differs from the parent configuration")

    def test_parent_patience_must_be_4(self):
        p = json.loads(json.dumps(self.parent))
        p["training"]["early_stopping_patience"] = 5
        self.parent_p = self.w.write_config(p, self.w.tmp / "parent.json")
        self.parent_sha = sha(self.parent_p.read_bytes())
        self.write_child(amended_doc(p, self.parent_sha))
        self.refused_both("must go from 4 (parent) to 12")

    def test_amendment_fields_checked(self):
        base = amended_doc(self.parent, self.parent_sha)
        SR.load_config(self.w.write_config(base, self.w.tmp / "ok_amended.json"))
        self.load_refused(amended_doc(self.parent, self.parent_sha, changed_field="training.max_epochs"), "must declare changed_field")
        self.load_refused(amended_doc(self.parent, self.parent_sha, **{"from": 5}), "must declare changed_field")
        self.load_refused(amended_doc(self.parent, self.parent_sha, to=13), "must declare changed_field")
        self.load_refused(amended_doc(self.parent, self.parent_sha, to=12.0), "must declare changed_field")
        self.load_refused(amended_doc(self.parent, self.parent_sha, **{"from": 4.0}), "must declare changed_field")
        self.load_refused(amended_doc(self.parent, self.parent_sha, parent_config_sha256="xyz"), "parent_config_sha256 malformed")
        self.load_refused(amended_doc(self.parent, self.parent_sha, note=" "), "amendment.note")
        self.load_refused(amended_doc(self.parent, self.parent_sha, shared_history_check=""), "amendment.shared_history_check")
        self.load_refused(amended_doc(self.parent, self.parent_sha, agreed_precisions=["one"]), "agreed_precisions must list the 2 agreed items")
        self.load_refused(amended_doc(self.parent, self.parent_sha, agreed_precisions=["one", ""]), "amendment.agreed_precisions[1]")
        d = amended_doc(self.parent, self.parent_sha)
        d["amendment"].pop("time_estimate")
        self.load_refused(d, "amendment needs exactly")
        self.load_refused(amended_doc(self.parent, self.parent_sha, patience=4), "training of an amended configuration")
        d = amended_doc(self.parent, self.parent_sha)
        d["training"]["initial_learning_rate"] = 0.0005
        self.load_refused(d, "training of an amended configuration")
        d = amended_doc(self.parent, self.parent_sha)
        d["extra"] = 1
        self.load_refused(d, "unknown ['extra']")
        d = json.loads(json.dumps(self.parent))
        d["training"]["early_stopping_patience"] = 12                # pazienza 12 senza emendamento: ricetta cambiata
        self.load_refused(d, "recipe of the folds run, unchanged")
        self.load_refused(amended_doc(self.parent, self.parent_sha, extra="x"), "amendment needs exactly")   # chiave in più
        self.load_refused(amended_doc(self.parent, self.parent_sha, parent_config_sha256=self.parent_sha.upper()),
                          "parent_config_sha256 malformed")
        self.load_refused(amended_doc(self.parent, self.parent_sha, agreed_precisions="ab"), "agreed_precisions must list the 2 agreed items")
        self.load_refused(amended_doc(self.parent, self.parent_sha, agreed_precisions=["one", "two", "three"]), "agreed_precisions must list the 2 agreed items")
        for bad in (None, [], "amendment"):                          # amendment presente ma non un oggetto
            d = amended_doc(self.parent, self.parent_sha)
            d["amendment"] = bad
            self.load_refused(d, "amendment needs exactly")
            d["training"]["early_stopping_patience"] = 4
            self.load_refused(d, "training of an amended configuration")

    def test_amended_patience_must_be_an_integer(self):
        d = amended_doc(self.parent, self.parent_sha)
        d["training"]["early_stopping_patience"] = 12.0
        self.write_child(d)
        self.refused_both("must go from 4 (parent) to 12")

    def test_parent_file_missing_refused(self):
        self.refused_both("parent configuration missing.json not found", PARENT_CONFIG=str(self.w.tmp / "missing.json"))

    def test_amendment_checked_before_any_input_is_read(self):
        def no_inputs(*a, **k):
            raise AssertionError("inputs read before the amendment check")
        self.refused_both("differs from the agreed configuration", PATIENCE12_CONFIG_SHA256="3" * 64, load_inputs=no_inputs)

    def test_run_records_the_configuration_name(self):
        seen = []
        trainer = fake_trainer()

        def capture(data, training, *a):
            seen.append(training["early_stopping_patience"])
            return trainer(data, training, *a)
        rc, out, err = self.cmd("run", "--i-am-authorized", TRAINER=capture)
        self.assertEqual(rc, 0, out + err)
        self.assertIn(f"(delivered: True, patience12_verification)", out)
        self.assertEqual(seen, [12] * N_MODELS, "the trainer receives the patience of the configuration")
        run = self.w.out / RUN
        status = json.loads((run / "status.json").read_text())
        self.assertEqual((status["status"], status["delivered_configuration"], status["configuration_name"]),
                         ("PASS", True, "patience12_verification"))
        self.assertEqual(status["config_sha256"], self.child_sha)
        self.assertEqual(status["amendment"], {"parent_config_sha256": self.parent_sha, "changed_field": "training.early_stopping_patience", "from": 4,
                                               "to": 12, "other_fields": "equal to the parent configuration"})
        for f in ("run_record.json", "aggregate.json"):
            doc = json.loads((run / "public" / f).read_text())
            self.assertEqual((doc["configuration_name"], doc["amendment"]["to"]), ("patience12_verification", 12), f)
        md = (run / "public" / "aggregate.md").read_text()
        self.assertIn(f"(patience12_verification: training.early_stopping_patience 4 -> 12, every other field equal to the parent "
                      f"configuration `{self.parent_sha[:12]}`)", md)
        card = json.loads((run / "restricted" / "arm_b_fold1_seed42_card.json").read_text())
        self.assertEqual(card["early_stopping_patience"], 12)

    def test_run_with_the_parent_configuration(self):
        with injected(**self.hooks()):
            rc, out, err = run_main(argv_of(self.w, "run", "--i-am-authorized", "--config", str(self.parent_p)))
        self.assertEqual(rc, 0, out + err)
        self.assertIn("(delivered: True, session_ablation_run)", out)
        run = self.w.out / RUN
        status = json.loads((run / "status.json").read_text())
        self.assertEqual((status["delivered_configuration"], status["configuration_name"]), (True, "session_ablation_run"))
        self.assertNotIn("amendment", status)
        rec = json.loads((run / "public" / "run_record.json").read_text())
        self.assertEqual(rec["configuration_name"], "session_ablation_run")
        self.assertNotIn("amendment", rec)
        self.assertNotIn("patience12_verification", (run / "public" / "aggregate.md").read_text())

    def test_configuration_name(self):
        with injected(CONFIG_SHA256="a" * 64, PATIENCE12_CONFIG_SHA256="b" * 64, CONSTANT_LR_CONFIG_SHA256="d" * 64):
            self.assertEqual([SR.configuration_name(h) for h in ("a" * 64, "b" * 64, "d" * 64, "c" * 64)],
                             ["session_ablation_run", "patience12_verification", "constant_lr_control", None])

    def test_authorisation_message_names_both_agreements(self):
        rc, out, err = run_main(self.argv("run"))
        self.assertEqual(rc, 1)
        self.assertIn("5 November 2026", err)
        self.assertIn("verification run with early_stopping_patience 12 on 7 November 2026", err)
        self.assertIn("control with a constant learning rate on 10 November 2026", err)

    def test_real_trainer_uses_the_patience_of_the_configuration(self):
        log = {}
        tf, arch = CFR.fake_tf(log, lambda e: [0.9, 0.8, 0.7] + [0.75] * 9)
        tmp = Path(tempfile.mkdtemp(prefix="sessablrun-patience-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        rng = np.random.default_rng(0)
        data = {"X_fit": rng.integers(0, 256, (8, *SHAPE), dtype=np.uint8), "y_fit": np.array([0, 1] * 4),
                "X_dev": rng.integers(0, 256, (4, *SHAPE), dtype=np.uint8), "y_dev": np.array([0, 1, 0, 1])}
        training = {**FR.FROZEN_TRAINING, "early_stopping_patience": 12}
        with CFR.modules({"tensorflow": tf}):
            card = FR.default_trainer(data, training, tmp / "w.keras", 42, arch, lambda step: None)
        self.assertEqual((log["es"]["patience"], log["fit"]["epochs"]), (12, 12))
        self.assertEqual((card["early_stopping_patience"], card["epochs_run"], card["restored_epoch"]), (12, 12, 3))
        self.assertEqual(card["restored_by"], "end of epochs (set_weights)")
        self.assertEqual(FR.card_problems(card, training, 42, tmp / "w.keras"), [])
        self.assertIn("hyperparameters differ from the recipe", FR.card_problems(card, dict(FR.FROZEN_TRAINING), 42, tmp / "w.keras"))


# ============================================================================ configurazione emendata (lr costante)

CONSTANT_TEXT = {"authorisation": "test agreement of 10 November", "agreed_conditions": [f"condition {i}" for i in range(1, 6)],
                 "note": "n", "time_estimate": "t", "shared_history_check": "s"}
CONSTANT_TRAINING = {**FR.FROZEN_TRAINING, "early_stopping_patience": 12, "scheduler": FR.SCHEDULER_CONSTANT}


def constant_doc(p12: dict, p12_sha: str, scheduler: str = FR.SCHEDULER_CONSTANT, **over) -> dict:
    """La configurazione del controllo a lr costante del mondo: la configurazione patience12 con il solo scheduler cambiato e
    il blocco amendment nuovo al posto del suo (stessa posizione, in fondo)."""
    d = json.loads(json.dumps(p12))
    d["training"]["scheduler"] = scheduler
    d["amendment"] = {"parent_config_sha256": p12_sha, "changed_field": "training.scheduler", "from": FR.SCHEDULER_REDUCE,
                      "to": FR.SCHEDULER_CONSTANT, **CONSTANT_TEXT, **over}
    return d


class TestConstantLR(unittest.TestCase):
    """Controllo a lr costante: SHA-256 fissato, genitore = la configurazione patience12 (a sua volta emendata dalla
    consegnata), ogni campo uguale tranne training.scheduler (ReduceLROnPlateau -> costante) e amendment."""

    def setUp(self):
        self.w = World()
        self.addCleanup(self.w.close)
        self.parent = self.w.sr_doc()
        self.parent_p = self.w.write_config(self.parent, self.w.tmp / "parent.json")
        self.parent_sha = sha(self.parent_p.read_bytes())
        self.p12 = amended_doc(self.parent, self.parent_sha)
        self.p12_p = self.w.write_config(self.p12, self.w.tmp / "patience12.json")
        self.p12_sha = sha(self.p12_p.read_bytes())
        self.child_p = self.w.tmp / "constant_lr.json"
        self.write_child(constant_doc(self.p12, self.p12_sha))

    def write_child(self, doc: dict) -> str:
        self.w.write_config(doc, self.child_p)
        self.child_sha = sha(self.child_p.read_bytes())
        return self.child_sha

    def hooks(self, **over) -> dict:
        h = {"CONFIG_SHA256": self.parent_sha, "PATIENCE12_CONFIG_SHA256": self.p12_sha, "CONSTANT_LR_CONFIG_SHA256": self.child_sha,
             "PARENT_CONFIG": str(self.parent_p), "CONSTANT_LR_PARENT_CONFIG": str(self.p12_p)}
        h.update(over)
        return defaults(self.w, **h)

    def argv(self, cmd: str, *extra, config=None) -> list:
        a = argv_of(self.w, cmd, *extra)
        a[a.index("--config") + 1] = str(config or self.child_p)
        return a

    def cmd(self, cmd: str, *extra, config=None, **over) -> tuple:
        with injected(**self.hooks(**over)):
            return run_main(self.argv(cmd, *extra, config=config))

    def refused_both(self, needle: str, **over):
        """check-inputs e run rifiutano con lo stesso messaggio; il run non crea la cartella."""
        rc, out, err = self.cmd("check-inputs", **over)
        self.assertEqual(rc, 1, out + err)
        self.assertIn(needle, err)
        self.assertIn("RESULT: FAIL (check-inputs)", out)
        rc, out, err = self.cmd("run", "--i-am-authorized", **over)
        self.assertEqual(rc, 1, out + err)
        self.assertIn(needle, err)
        self.assertFalse((self.w.out / RUN).exists())

    def load_refused(self, doc: dict, needle: str):
        p = self.w.write_config(doc, self.w.tmp / "bad_constant.json")
        with self.assertRaises(FR.Failure) as cm:
            SR.load_config(p)
        self.assertIn(needle, str(cm.exception))

    # ---------------------------------------------------------------- file reali di delivery/ (nessun dato)

    def real_files(self) -> tuple:
        cp, pp, gp = REAL_ROOT / SR.CONSTANT_LR_CONFIG, REAL_ROOT / SR.PATIENCE12_CONFIG, REAL_ROOT / SR.DEFAULT_CONFIG
        if not cp.is_file() or not pp.is_file() or not gp.is_file():
            skip_missing(self, "constant_lr configuration")
        return cp, pp, gp

    def test_real_constant_lr_configuration_pinned(self):
        cp, pp, gp = self.real_files()
        cfg, h = SR.load_config(cp)
        self.assertEqual(h, sha(cp.read_bytes()))
        self.assertEqual(h, SR.CONSTANT_LR_CONFIG_SHA256, "CONSTANT_LR_CONFIG_SHA256 must be the SHA-256 of the constant_lr configuration")
        self.assertEqual(sha(pp.read_bytes()), SR.PATIENCE12_CONFIG_SHA256)
        self.assertEqual(sha(gp.read_bytes()), SR.CONFIG_SHA256)
        am = cfg["amendment"]
        self.assertEqual(list(am), ["parent_config_sha256", "changed_field", "from", "to", "authorisation", "agreed_conditions", "note",
                                    "time_estimate", "shared_history_check"])
        self.assertEqual((am["parent_config_sha256"], am["changed_field"], am["from"], am["to"]),
                         (SR.PATIENCE12_CONFIG_SHA256, "training.scheduler", FR.SCHEDULER_REDUCE, FR.SCHEDULER_CONSTANT))
        self.assertEqual(cfg["training"], CONSTANT_TRAINING)
        self.assertEqual((cfg["training"]["early_stopping_patience"], cfg["training"]["max_epochs"], cfg["training"]["initial_learning_rate"]),
                         (12, 12, 0.001))
        self.assertEqual((cfg["budget"]["wall_time_cap_minutes"], cfg["budget"]["trainings"], cfg["train_seeds"]), (360, 36, [42, 43, 44]))
        self.assertIn("comment of 10 November 2026", am["authorisation"])
        self.assertIn("learning rate costante a 0,001 e limite di 360 minuti", am["authorisation"])
        self.assertIn("senza ulteriori tentativi o conclusioni sul run incompleto", am["authorisation"])
        self.assertEqual(len(am["agreed_conditions"]), 5)
        for k in ("17,940.9 s", "21,600 s", "3,659.1 s"):
            self.assertIn(k, am["time_estimate"])
        self.assertIn("20261109-SESSABLRUN-6a50265", am["shared_history_check"])
        self.assertIn("before the first reduction", am["shared_history_check"])
        self.assertEqual([SR.configuration_name(x) for x in (h, SR.PATIENCE12_CONFIG_SHA256, SR.CONFIG_SHA256)],
                         ["constant_lr_control", "patience12_verification", "session_ablation_run"])

    def test_real_constant_lr_differs_only_in_the_scheduler(self):
        cp, pp, gp = self.real_files()
        child, parent = json.loads(cp.read_bytes()), json.loads(pp.read_bytes())
        e = SR.AMENDMENTS[1]
        self.assertIsNone(SR.config_difference(parent, child, SR.amendment_allowed(e)))
        self.assertEqual(SR.config_difference(parent, child, ()), "training.scheduler")
        self.assertEqual(SR.config_difference(parent, child, ("training.scheduler",)), "amendment.parent_config_sha256")
        self.assertEqual(list(child), list(parent), "same order of the keys as the parent")
        self.assertEqual(list(child["training"]), list(parent["training"]))
        self.assertEqual((parent["training"]["scheduler"], child["training"]["scheduler"]), (FR.SCHEDULER_REDUCE, FR.SCHEDULER_CONSTANT))
        # stesso stile di serializzazione del genitore: rifatto il genitore con le sole due modifiche si ottengono gli stessi byte
        self.assertEqual((json.dumps(parent, indent=1, ensure_ascii=True) + "\n").encode(), pp.read_bytes())
        rebuilt = json.loads(pp.read_bytes())
        rebuilt["training"]["scheduler"] = FR.SCHEDULER_CONSTANT
        rebuilt["amendment"] = child["amendment"]
        self.assertEqual((json.dumps(rebuilt, indent=1, ensure_ascii=True) + "\n").encode(), cp.read_bytes())
        cfg, h = SR.load_config(cp)
        got = SR.check_amendment(cfg, h)
        self.assertEqual(got, {"parent_config_sha256": SR.PATIENCE12_CONFIG_SHA256, "changed_field": "training.scheduler",
                               "from": FR.SCHEDULER_REDUCE, "to": FR.SCHEDULER_CONSTANT, "other_fields": "equal to the parent configuration"})
        self.assertEqual(SR.check_amendment(*SR.load_config(pp))["parent_config_sha256"], SR.CONFIG_SHA256)
        self.assertIsNone(SR.check_amendment(*SR.load_config(gp)))

    # ---------------------------------------------------------------- tabella

    def test_amendment_table(self):
        self.assertEqual([(e["name"], e["field"], e["sha256"], e["parent_sha256"], e["config"], e["parent_config"]) for e in SR.AMENDMENTS],
                         [("patience12_verification", "training.early_stopping_patience", "PATIENCE12_CONFIG_SHA256", "CONFIG_SHA256",
                           "PATIENCE12_CONFIG", "PARENT_CONFIG"),
                          ("constant_lr_control", "training.scheduler", "CONSTANT_LR_CONFIG_SHA256", "PATIENCE12_CONFIG_SHA256",
                           "CONSTANT_LR_CONFIG", "CONSTANT_LR_PARENT_CONFIG")])
        self.assertEqual((SR.PARENT_CONFIG, SR.CONSTANT_LR_PARENT_CONFIG, SR.CONSTANT_LR_CONFIG),
                         (SR.DEFAULT_CONFIG, SR.PATIENCE12_CONFIG, "delivery/constlr/nuaa_session_ablation_run_constant_lr.json"))
        self.assertEqual([SR.amended_training(e) for e in SR.AMENDMENTS],
                         [{**FR.FROZEN_TRAINING, "early_stopping_patience": 12}, CONSTANT_TRAINING])
        self.assertEqual([(e["from"], e["to"]) for e in SR.AMENDMENTS], [(4, 12), (FR.SCHEDULER_REDUCE, FR.SCHEDULER_CONSTANT)])
        self.assertEqual([e["agreed"] for e in SR.AMENDMENTS], [("agreed_precisions", 2), ("agreed_conditions", 5)])
        self.assertEqual(SR.amendment_allowed(SR.AMENDMENTS[1]), ("training.scheduler", "amendment"))
        self.assertEqual(SR.field_value({"training": {"scheduler": "x"}}, "training.scheduler"), "x")
        self.assertIsNone(SR.field_value({"training": {}}, "training.scheduler"))
        self.assertIsNone(SR.field_value({"training": "t"}, "training.scheduler"))

    # ---------------------------------------------------------------- check-inputs e run

    def test_constant_lr_configuration_accepted(self):
        rc, out, err = self.cmd("check-inputs")
        self.assertEqual(rc, 0, out + err)
        self.assertIn(f"AMENDMENT: configuration constant_lr_control {self.child_sha[:12]}, parent {self.p12_sha[:12]} equal in "
                      "every field except training.scheduler (ReduceLROnPlateau -> constant) and amendment", out)
        self.assertIn("RESULT: PASS (check-inputs)", out)
        rc, out, err = self.cmd("check-inputs", config=self.p12_p)          # la configurazione patience12 resta accettata
        self.assertEqual(rc, 0, out + err)
        self.assertIn(f"AMENDMENT: configuration patience12_verification {self.p12_sha[:12]}, parent {self.parent_sha[:12]}", out)
        rc, out, err = self.cmd("check-inputs", config=self.parent_p)       # e la consegnata, senza riga AMENDMENT
        self.assertEqual(rc, 0, out + err)
        self.assertNotIn("AMENDMENT", out)

    def test_constant_lr_hash_not_pinned_refused(self):
        self.refused_both("differs from the agreed configuration constant_lr_control", CONSTANT_LR_CONFIG_SHA256="3" * 64)

    def test_constant_lr_checked_before_any_input_is_read(self):
        def no_inputs(*a, **k):
            raise AssertionError("inputs read before the amendment check")
        self.refused_both("differs from the agreed configuration constant_lr_control", CONSTANT_LR_CONFIG_SHA256="3" * 64,
                          load_inputs=no_inputs)
        self.write_child(constant_doc(self.p12, self.parent_sha))
        self.refused_both("parent_config_sha256 differs from the patience12_verification configuration", load_inputs=no_inputs)

    def test_parent_declared_must_be_the_patience12_configuration(self):
        self.write_child(constant_doc(self.p12, "5" * 64))
        self.refused_both("parent_config_sha256 differs from the patience12_verification configuration")
        self.write_child(constant_doc(self.p12, self.parent_sha))          # il nonno (la configurazione consegnata) non basta
        self.refused_both("parent_config_sha256 differs from the patience12_verification configuration")

    def test_parent_file_differs_or_missing_refused(self):
        self.write_child(constant_doc(self.p12, "6" * 64))
        self.refused_both("parent configuration patience12.json: SHA-256", PATIENCE12_CONFIG_SHA256="6" * 64)
        self.write_child(constant_doc(self.p12, self.p12_sha))
        self.refused_both("parent configuration missing.json not found", CONSTANT_LR_PARENT_CONFIG=str(self.w.tmp / "missing.json"))
        self.refused_both("parent configuration parent.json: SHA-256", CONSTANT_LR_PARENT_CONFIG=str(self.parent_p))

    def test_other_field_difference_refused(self):
        d = constant_doc(self.p12, self.p12_sha)
        d["task"] = "another task"
        self.write_child(d)
        self.refused_both("field task differs from the parent configuration (only training.scheduler ReduceLROnPlateau -> constant and "
                          "amendment may differ)")
        d = constant_doc(self.p12, self.p12_sha)
        d["budget"]["wall_time_cap_minutes"] = 360.0
        self.write_child(d)
        self.refused_both("field budget.wall_time_cap_minutes differs from the parent configuration")
        d = constant_doc(self.p12, self.p12_sha)
        d["training"]["early_stopping_patience"] = 12.0             # uguale per ==, diverso per tipo: rifiutato
        self.write_child(d)
        self.refused_both("field training.early_stopping_patience differs from the parent configuration")

    def test_patience_or_recipe_changed_with_the_scheduler_refused(self):
        need = "training of an amended configuration must be the recipe of the folds run with only the agreed changes"
        for k, v in (("early_stopping_patience", 4), ("early_stopping_patience", 13), ("max_epochs", 20), ("initial_learning_rate", 0.0005),
                     ("batch", 32)):
            d = constant_doc(self.p12, self.p12_sha)
            d["training"][k] = v
            self.load_refused(d, need)
            self.write_child(d)
            self.refused_both(need)
        d = constant_doc(self.p12, self.p12_sha)
        d["training"]["lr_schedule"] = "none"
        self.load_refused(d, need)
        d = constant_doc(self.p12, self.p12_sha)
        self.load_refused({**d, "amendment": {**d["amendment"], "changed_field": "training.early_stopping_patience", "from": 4, "to": 12}},
                          "early_stopping_patience = 12)")          # la voce patience12 non ammette lo scheduler costante

    def test_unknown_scheduler_refused(self):
        need = "training of an amended configuration"
        for bad in ("none", "constant", FR.SCHEDULER_CONSTANT + " ", FR.SCHEDULER_REDUCE):
            self.load_refused(constant_doc(self.p12, self.p12_sha, scheduler=bad), need)
        self.load_refused(constant_doc(self.p12, self.p12_sha, to="none"), "must declare changed_field training.scheduler")
        d = json.loads(json.dumps(self.p12))
        d["training"]["scheduler"] = FR.SCHEDULER_CONSTANT                 # lr costante senza emendamento: rifiutato
        d.pop("amendment")
        self.load_refused(d, "recipe of the folds run, unchanged")

    def test_from_and_to_checked(self):
        need = "amendment must declare changed_field training.scheduler, from ReduceLROnPlateau, to constant"
        for over in ({"from": FR.SCHEDULER_CONSTANT, "to": FR.SCHEDULER_REDUCE}, {"from": "ReduceLROnPlateau"}, {"to": "constant"},
                     {"to": FR.SCHEDULER_CONSTANT + " "}, {"from": [FR.SCHEDULER_REDUCE]}, {"to": None}, {"from": FR.SCHEDULER_CONSTANT}):
            self.load_refused(constant_doc(self.p12, self.p12_sha, **over), need)
        self.load_refused(constant_doc(self.p12, self.p12_sha, changed_field="training.lr"), "must declare changed_field")

    def test_parent_scheduler_must_be_the_recipe(self):
        p12 = json.loads(json.dumps(self.p12))
        p12["training"]["scheduler"] = FR.SCHEDULER_CONSTANT              # un genitore già a lr costante: nessun cambio
        self.p12_p = self.w.write_config(p12, self.w.tmp / "patience12.json")
        self.p12_sha = sha(self.p12_p.read_bytes())
        self.write_child(constant_doc(p12, self.p12_sha))
        self.refused_both("training.scheduler must go from ReduceLROnPlateau (parent) to constant")

    def test_amendment_keys_checked(self):
        SR.load_config(self.w.write_config(constant_doc(self.p12, self.p12_sha), self.w.tmp / "ok_constant.json"))
        d = constant_doc(self.p12, self.p12_sha)
        d["amendment"].pop("agreed_conditions")
        self.load_refused(d, "amendment needs exactly")
        d["amendment"]["agreed_precisions"] = ["one", "two"]             # le precisazioni della pazienza non valgono qui
        self.load_refused(d, "amendment needs exactly")
        d = constant_doc(self.p12, self.p12_sha)
        d["amendment"].pop("shared_history_check")
        self.load_refused(d, "amendment needs exactly")
        self.load_refused(constant_doc(self.p12, self.p12_sha, extra="x"), "amendment needs exactly")
        self.load_refused(constant_doc(self.p12, self.p12_sha, agreed_conditions=["a", "b", "c", "d"]),
                          "agreed_conditions must list the 5 agreed items")
        self.load_refused(constant_doc(self.p12, self.p12_sha, agreed_conditions=["a"] * 6), "agreed_conditions must list the 5 agreed items")
        self.load_refused(constant_doc(self.p12, self.p12_sha, agreed_conditions="abcde"), "agreed_conditions must list the 5 agreed items")
        self.load_refused(constant_doc(self.p12, self.p12_sha, agreed_conditions=["a", "b", " ", "d", "e"]), "amendment.agreed_conditions[2]")
        self.load_refused(constant_doc(self.p12, self.p12_sha, note=""), "amendment.note")
        self.load_refused(constant_doc(self.p12, self.p12_sha, time_estimate=" "), "amendment.time_estimate")
        self.load_refused(constant_doc(self.p12, self.p12_sha, authorisation=None), "amendment.authorisation")
        self.load_refused(constant_doc(self.p12, self.p12_sha, parent_config_sha256="xyz"), "parent_config_sha256 malformed")

    def test_run_records_the_constant_lr_control(self):
        seen = []
        trainer = fake_trainer()

        def capture(data, training, *a):
            seen.append(training)
            return trainer(data, training, *a)
        rc, out, err = self.cmd("run", "--i-am-authorized", TRAINER=capture)
        self.assertEqual(rc, 0, out + err)
        self.assertIn("(delivered: True, constant_lr_control)", out)
        self.assertEqual(seen, [CONSTANT_TRAINING] * N_MODELS, "the trainer receives the constant scheduler and patience 12")
        run = self.w.out / RUN
        status = json.loads((run / "status.json").read_text())
        self.assertEqual((status["status"], status["delivered_configuration"], status["configuration_name"], status["config_sha256"]),
                         ("PASS", True, "constant_lr_control", self.child_sha))
        want = {"parent_config_sha256": self.p12_sha, "changed_field": "training.scheduler", "from": FR.SCHEDULER_REDUCE,
                "to": FR.SCHEDULER_CONSTANT, "other_fields": "equal to the parent configuration"}
        self.assertEqual(status["amendment"], want)
        for f in ("run_record.json", "aggregate.json"):
            doc = json.loads((run / "public" / f).read_text())
            self.assertEqual((doc["configuration_name"], doc["amendment"]), ("constant_lr_control", want), f)
        md = (run / "public" / "aggregate.md").read_text()
        self.assertIn(f"(constant_lr_control: training.scheduler ReduceLROnPlateau -> constant, every other field equal to the parent "
                      f"configuration `{self.p12_sha[:12]}`)", md)
        self.assertIn("for early stopping, the restored epoch and the EER threshold (constant learning rate, no schedule).", md)
        self.assertNotIn("the learning-rate schedule", md)
        card = json.loads((run / "restricted" / "arm_b_fold1_seed42_card.json").read_text())
        self.assertEqual((card["scheduler"], card["early_stopping_patience"]), (FR.SCHEDULER_CONSTANT, 12))
        self.assertEqual(card["history"]["lr"], [0.001] * card["epochs_run"])

    def test_run_with_the_patience12_configuration_keeps_the_schedule(self):
        rc, out, err = self.cmd("run", "--i-am-authorized", config=self.p12_p)
        self.assertEqual(rc, 0, out + err)
        run = self.w.out / RUN
        self.assertEqual(json.loads((run / "status.json").read_text())["configuration_name"], "patience12_verification")
        md = (run / "public" / "aggregate.md").read_text()
        self.assertIn("for early stopping, the learning-rate schedule, the restored epoch and the EER threshold. Test:", md)
        card = json.loads((run / "restricted" / "arm_b_fold1_seed42_card.json").read_text())
        self.assertEqual(card["scheduler"], FR.SCHEDULER_REDUCE)

    def test_run_refuses_a_card_with_a_reduced_or_missing_learning_rate(self):
        base = fake_trainer()

        def reduced(data, training, *a):
            card = base(data, training, *a)
            card["history"]["lr"][-1] = 0.0005
            return card
        rc, out, err = self.cmd("run", "--i-am-authorized", TRAINER=reduced)
        self.assertEqual(rc, 1, out + err)
        self.assertIn("learning rate changed with the constant scheduler", err)
        shutil.rmtree(self.w.out)

        def missing(data, training, *a):
            card = base(data, training, *a)
            card["history"].pop("lr")
            return card
        rc, out, err = self.cmd("run", "--i-am-authorized", TRAINER=missing)
        self.assertEqual(rc, 1, out + err)
        self.assertIn("history without the learning rate of every epoch", err)
        shutil.rmtree(self.w.out)

        def other(data, training, *a):
            return base(data, {**training, "scheduler": FR.SCHEDULER_REDUCE}, *a)
        rc, out, err = self.cmd("run", "--i-am-authorized", TRAINER=other)
        self.assertEqual(rc, 1, out + err)
        self.assertIn("scheduler differs from the configuration", err)

    def test_real_trainer_with_the_constant_scheduler(self):
        log = {}
        tf, arch = CFR.fake_tf(log, lambda e: [0.9, 0.8, 0.7] + [0.75] * 9)
        tmp = Path(tempfile.mkdtemp(prefix="sessablrun-constlr-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        rng = np.random.default_rng(0)
        data = {"X_fit": rng.integers(0, 256, (8, *SHAPE), dtype=np.uint8), "y_fit": np.array([0, 1] * 4),
                "X_dev": rng.integers(0, 256, (4, *SHAPE), dtype=np.uint8), "y_dev": np.array([0, 1, 0, 1])}
        with CFR.modules({"tensorflow": tf}):
            card = FR.default_trainer(data, dict(CONSTANT_TRAINING), tmp / "w.keras", 42, arch, lambda step: None)
        self.assertNotIn("rlr", log)
        self.assertEqual(log["fit"]["callbacks"], ["ES", "LearningRateLog", "Cap"])
        self.assertEqual((log["es"]["patience"], log["fit"]["epochs"], log["adam"]), (12, 12, (0.001, 1.0)))
        self.assertEqual((card["scheduler"], card["epochs_run"], card["restored_epoch"]), (FR.SCHEDULER_CONSTANT, 12, 3))
        self.assertEqual(card["history"]["lr"], [float(np.float32(0.001))] * 12)
        self.assertEqual(FR.card_problems(card, CONSTANT_TRAINING, 42, tmp / "w.keras"), [])
        self.assertIn("scheduler differs from the configuration",
                      FR.card_problems(card, {**FR.FROZEN_TRAINING, "early_stopping_patience": 12}, 42, tmp / "w.keras"))


# ============================================================================ ingressi veri (sola lettura)

class TestRealInputs(unittest.TestCase):
    def test_real_check_inputs(self):
        cfg_p = REAL_ROOT / SR.DEFAULT_CONFIG
        if not cfg_p.is_file():
            skip_missing(self, "delivered configuration")
        cfg, _ = SR.load_config(cfg_p)
        needed = [REAL_ROOT / cfg["manifests"]["folder"], REAL_ROOT / cfg["schema_run"]["folder"], REAL_ROOT / cfg["cells_run"]["folder"],
                  REAL_ROOT / cfg["folds_run"]["folder"], REAL_ROOT / cfg["folds_run"]["frames_file"]["path"], REAL_ROOT / cfg["cache"]["path"],
                  REAL_ROOT / cfg["images"]["folder"], REAL_ROOT / cfg["architecture"]["path"]]
        if not all(p.exists() for p in needed) or FR.environment()["opencv"] == "absent":
            skip_missing(self, "real inputs")
        before = snapshot(REAL_ROOT / "results", REAL_ROOT / "data" / "processed")
        inp = SR.load_inputs(cfg, REAL_ROOT)
        self.assertEqual(inp.checks["originals"]["located_and_hashed"], 12614)
        self.assertEqual((len(inp.cache_index), len(inp.extracted_keys)), (6000, 6614))
        self.assertEqual(len(inp.cells_sha), 16)
        src = inp.checks["sources"]
        self.assertEqual(src["rows_from_cache"] + src["rows_from_extracted"], 4 * (1400 + 1400 + 934 + 400) + 12614)
        self.assertEqual({f: v["s12_subjects"] for f, v in inp.checks["cells"]["nesting"].items()}, {1: 4, 2: 5, 3: 5, 4: 4})
        self.assertEqual(snapshot(REAL_ROOT / "results", REAL_ROOT / "data" / "processed"), before)


class ZZRepositoryUntouched(unittest.TestCase):
    def test_declared_skips_only(self):
        self.assertEqual(sorted(MISSING_INPUT), ["constant_lr configuration", "delivered configuration", "patience12 configuration",
                                                 "real inputs"])


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
