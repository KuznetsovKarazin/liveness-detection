"""
Test di scripts/nuaa_bn_test.py (confronto prima/dopo sul test della ristima delle BatchNormalization della PR #27),
offline: nessun checkpoint vero caricato, nessun frame vero letto, nessun punteggio vero aperto.

Mondo sintetico in cartelle temporanee, costruito sul mondo della suite della PR #27 (check_nuaa_bn_recalibration.World:
manifest M2, cache, celle, run dei fold, frame 8×8 con la classe nella luminosità, schema dei fold, run genitore con 36
modelli Keras minuscoli «good», «badstats», «bias», «inverted»), più: i manifest del test dello schema leggibili, i
<modello>_test_scores.csv del genitore calcolati sul test di ogni fold con un valutatore indipendente (model.predict),
e un run BNRECAL vero prodotto da nuaa_bn_recalibration sullo stesso mondo. I file pubblici del run dei fold restano con
permessi 000 (mai aperti). TensorFlow vero su CPU, modelli e dati minuscoli. Nessuna scrittura nel repository (controllo
finale su results/, export/, delivery/).

  .venv/bin/python -I -B scripts/check_nuaa_bn_test.py [--verbose] [-f]
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
import unittest
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_nuaa_bn_recalibration as RBS  # noqa: E402
import check_nuaa_folds_run as CFR  # noqa: E402
import check_nuaa_session_ablation_run as SAR  # noqa: E402
import nuaa_bn_recalibration as BR  # noqa: E402
import nuaa_bn_test as T  # noqa: E402
import nuaa_folds_run as FR  # noqa: E402
import nuaa_session_ablation_run as SR  # noqa: E402

MISSING_INPUT = {"delivered configuration": "delivery/bntest/nuaa_bn_test.json is not present"}
STATE = {"commit": "abcdef0", "dirty": False}
DATE = "20261116"
BNRECAL_ID = f"20261114-BNRECAL-{STATE['commit']}"
TF = RBS.TF


def skip_missing(test: unittest.TestCase, what: str):
    """Unico modo di saltare un test di questa suite: un ingresso mancante dichiarato in MISSING_INPUT."""
    test.skipTest(f"missing input: {MISSING_INPUT[what]}")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_csv(p: Path) -> list:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(p: Path, cols, rows) -> None:
    p.write_text(CFR.csv_text(cols, [[r[c] for c in cols] for r in rows]), encoding="utf-8")


# ============================================================================ riferimenti indipendenti

def ref_metrics(y, s) -> dict:
    """APCER, BPCER, ACER alla soglia 0,5 (attacco se s > 0,5) e AUROC per coppie (pareggi 0,5), senza il codice del
    progetto; None se una classe manca."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    a, b = s[y == 1], s[y == 0]
    apcer = float(np.sum(a <= 0.5)) / len(a) if len(a) else None
    bpcer = float(np.sum(b > 0.5)) / len(b) if len(b) else None
    acer = (apcer + bpcer) / 2 if apcer is not None and bpcer is not None else None
    auroc = None
    if len(a) and len(b):
        auroc = float(sum((x > z) + 0.5 * (x == z) for x in a for z in b)) / (len(a) * len(b))
    return {"apcer_05": apcer, "bpcer_05": bpcer, "acer_05": acer, "auroc": auroc}


def rows_of(y, s, subj, session="03") -> list:
    return [{"label": str(int(a)), "score_attack": repr(float(b)), "subject_id": c, "session_id": session, "sample_id": f"k{i}",
             "prediction_05": "1" if b > 0.5 else "0", "prediction_dev": "0"} for i, (a, b, c) in enumerate(zip(y, s, subj))]


def read_csv_text(text: str) -> list:
    return list(csv.DictReader(io.StringIO(text)))


def num(v):
    return None if v == "NA" else float(v)


# ============================================================================ mondo sintetico

class World(RBS.World):
    """Il mondo della PR #27 con i punteggi del test del genitore e un run BNRECAL vero."""

    def build(self):
        self.test_cache = {}
        super().build()
        for i in range(1, CFR.K + 1):
            (self.base / "schema" / f"fold{i}_test.csv").chmod(0o644)          # questo runner li apre
        self.bt_config = self.base / "bntest.json"
        self.bnrecal_root = self.base / "bnrecal_runs"
        self.run_bnrecal()
        self.write_bt_config(self.bt_doc())

    def test_keys(self, fold: int) -> list:
        return sorted(r[2] for r in self.folds[fold]["test"])

    def build_parent(self, over=None, test_over=None):
        """Il run genitore della suite della PR #27 più i punteggi del test di ogni modello (valutatore indipendente sul
        test del fold, righe nel formato del genitore); test_over: nome -> funzione(righe) per alterarle."""
        super().build_parent(over)
        if not hasattr(self, "test_cache"):
            return
        test_over = test_over or {}
        run = self.parent
        for arm, fold, seed in SR.training_order(CFR.K, SAR.SEEDS):
            name = SR.model_name(arm, fold, seed)
            kind = RBS.KINDS.get(name, "good")
            keys = self.test_keys(fold)
            if (kind, fold) not in self.test_cache:
                model = TF.keras.models.load_model(str(self.kind_files[kind]), compile=False)
                y = np.array([1 if k.startswith("attack/") else 0 for k in keys])
                self.test_cache[(kind, fold)] = RBS.independent_eval(model, RBS.frames_of(keys), y)["p"]
            card = json.loads((run / "restricted" / f"{name}_card.json").read_text())
            tb = (run / "restricted" / f"{name}_threshold.json").read_bytes()
            thr = {"value": json.loads(tb)["value"], "sha256": sha(tb)}
            part = {"rows": [{"key": k, "subject": self.official[k]["subject"], "session": self.official[k]["session"],
                              "label": self.official[k]["label"], "sha256": self.official[k]["sha256"]} for k in keys]}
            rows = SR.score_rows(arm, fold, seed, "test", part, self.test_cache[(kind, fold)][:, 1], thr, card)
            if name in test_over:
                test_over[name](rows)
            tp = run / "restricted" / f"{name}_test_scores.csv"
            tp.chmod(0o644)
            tp.write_text(SR.rows_text(rows), encoding="utf-8")
        self.parent_res = RBS.write_sums(run / "restricted")
        self.parent_status = {**self.parent_status, "restricted_sums_sha256": self.parent_res}
        (run / "status.json").write_text(json.dumps(self.parent_status, indent=1) + "\n", encoding="utf-8")
        if hasattr(self, "bn_config"):
            self.write_bn_config(self.bn_doc())

    def run_bnrecal(self):
        if self.bnrecal_root.exists():
            shutil.rmtree(self.bnrecal_root)
        out, err = io.StringIO(), io.StringIO()
        with RBS.injected(**RBS.hooks(self)), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = BR.main(self.argv("run", out=self.bnrecal_root))
        if rc != 0:
            raise RuntimeError(f"the synthetic BNRECAL run did not pass: {err.getvalue()[-2000:]}")
        self.bnrecal = self.bnrecal_root / BNRECAL_ID
        self.bnrecal_config_bytes = self.bn_config.read_bytes()

    def dev_worsened(self) -> list:
        rows = read_csv(self.bnrecal / "public" / "results.csv")
        flags = [f for f in BR.FLAGS if f != "gained_criterion"]
        return [f"{r['arm']}_fold{r['fold']}_seed{r['seed']}" for r in rows if any(r[f] == "1" for f in flags)]

    def bt_doc(self) -> dict:
        bn = self.bn_doc()
        st = json.loads((self.bnrecal / "status.json").read_text())
        tol = {**bn["tolerances"], **{k: {"value": (0 if k in T.COUNT_TOLERANCES else {"recalibration_statistics_rel": 1e-4,
                                                                                         "recalibration_statistics_abs": 1e-6}.get(k, 1e-5)),
                                          "source": "t"} for k in T.NEW_TOLERANCES}}
        return {
            "schema": T.CONFIG_SCHEMA, "task": "synthetic", "authorisation": "test",
            "declarations": {"post_hoc": "post hoc, not an independent validation", "threshold": "fixed at 0.5", "procedure": "p",
                             "cause": "c", "worsening": "w"},
            "parent_run": {**bn["parent_run"], "used": "rewritten for this run"},
            "bnrecal_run": {"run_id": BNRECAL_ID, "folder": f"bnrecal_runs/{BNRECAL_ID}", "config_path": self.bn_config.name,
                            "config_sha256": sha(self.bnrecal_config_bytes), "public_sha256sums_sha256": st["public_sums_sha256"],
                            "restricted_sha256sums_sha256": st["restricted_sums_sha256"], "models": 36, "used": "t"},
            "code": {f: sha((ROOT / f).read_bytes()) for f in T.CODE_FILES},
            **{k: bn[k] for k in T.SHARED if k not in ("parent_run", "evaluation")},
            "evaluation": {**bn["evaluation"], "probability": "rewritten for this run"}, "dev_worsened": {"definition": "d", "source": "s", "expected": self.dev_worsened()},
            "tolerances": tol, "stop_rules": {"baseline": "stop with BASELINE_FAILED", "recalibration": "stop with RECALIBRATION_MISMATCH",
                                         "fail": "FAIL: status.json and run.log kept"},
            "comparison": {"test_frames_per_fold": [len(self.folds[i]["test"]) for i in range(1, CFR.K + 1)], "scopes": list(T.SCOPES), "statistics": list(T.STATISTICS), "metrics": list(T.METRICS), "threshold": 0.5,
                           "decision": FR.DECISION_RULE, "frames": "t", "auroc": "t", "undefined": "t", "worsening": "w", "per_subject": "t"},
            "bootstrap": {**SR.BOOT, "recipe": "r", "combination": "c", "summaries": {"groups": list(T.GROUP_SUMMARIES), "arms": list(SR.ARMS)}},
            "reading_rule": {"primary": "p", "below": "BELOW {who} {scope} {metric} {difference} [{low}, {high}]",
                             "contains": "CONTAINS {who} {scope} {metric} {difference} [{low}, {high}]",
                             "above": "ABOVE {who} {scope} {metric} {difference} [{low}, {high}]",
                             "not_available": "NOREADING {who} {scope} {metric} ({reason})",
                             "count": "COUNT {lower}/{equal}/{higher} of {n}", "secondary": "s", "limits": "l"},
            "outputs": {"public": list(T.PUBLIC_OUTPUTS), "restricted": [f"<model>_{x}" for x in T.PER_MODEL] + list(T.RESTRICTED_FIXED)},
            "budget": {"wall_time_cap_minutes": 240, "estimate_minutes": 42, "upper_estimate_minutes": 113, "estimate": "t",
                       "on_cap_exceeded": "INCOMPLETE", "voluntary_stop": "SIGUSR1: INCOMPLETE"},
            "environment": {"tensorflow": bn["environment"]["tensorflow"], "matplotlib": T.environment()["matplotlib"]},
            "note_path": str(self.note), "note_sha256": sha(self.note.read_bytes()), "test": T.TEST_READ}

    def write_bt_config(self, doc: dict) -> Path:
        self.bt_config.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        return self.bt_config

    def bt_argv(self, cmd: str, *extra, out=None, date=DATE) -> list:
        a = [cmd, "--config", str(self.bt_config), "--input-root", str(self.base), "--out-root", str(out or self.out), "--date", date]
        return a + (["--i-am-authorized"] if cmd == "run" else []) + list(extra)

    def resum_bnrecal(self, public_changed: bool = False) -> None:
        """Dopo un'alterazione del run BNRECAL: SHA256SUMS ricalcolati, status.json e configurazione allineati."""
        if public_changed:
            ps = RBS.write_sums(self.bnrecal / "public")
            (self.bnrecal / "restricted" / "public_SHA256SUMS.sha256").write_text(f"{ps}  public/SHA256SUMS\n", encoding="utf-8")
        st = json.loads((self.bnrecal / "status.json").read_text())
        st["public_sums_sha256"] = sha((self.bnrecal / "public" / "SHA256SUMS").read_bytes())
        st["restricted_sums_sha256"] = RBS.write_sums(self.bnrecal / "restricted")
        (self.bnrecal / "status.json").write_text(json.dumps(st, indent=1) + "\n", encoding="utf-8")
        self.write_bt_config(self.bt_doc())


@contextlib.contextmanager
def altered_bnrecal(w: World, mutate, public_changed: bool = False):
    """Una copia del run BNRECAL alterata e riancorata, ripristinata all'uscita."""
    saved = w.tmp / "bnrecal_saved"
    shutil.copytree(w.bnrecal, saved)
    try:
        mutate(w.bnrecal)
        w.resum_bnrecal(public_changed)
        yield
    finally:
        shutil.rmtree(w.bnrecal)
        shutil.move(str(saved), str(w.bnrecal))
        w.write_bt_config(w.bt_doc())


@contextlib.contextmanager
def altered_parent(w: World, over=None, test_over=None):
    """Il genitore ricostruito con punteggi del test alterati; la configurazione della PR #27 cambia con il suo
    SHA256SUMS riservato: status.json del run BNRECAL e configurazione riancorati, tutto ripristinato all'uscita."""
    st = w.bnrecal / "status.json"
    saved_status, saved_bytes = st.read_text(), w.bnrecal_config_bytes
    try:
        w.build_parent(over, test_over)
        w.bnrecal_config_bytes = w.bn_config.read_bytes()
        st.write_text(json.dumps({**json.loads(saved_status), "config_sha256": sha(w.bnrecal_config_bytes)}, indent=1) + "\n", encoding="utf-8")
        w.write_bt_config(w.bt_doc())
        yield
    finally:
        w.build_parent()
        st.write_text(saved_status, encoding="utf-8")
        w.bnrecal_config_bytes = saved_bytes
        w.write_bt_config(w.bt_doc())


@contextlib.contextmanager
def injected(t=None, rb=None, sr=None, fr=None):
    saved = [(mod, k, getattr(mod, k)) for mod, d in ((T, t), (BR, rb), (SR, sr), (FR, fr)) for k in (d or {})]
    try:
        for mod, d in ((T, t), (BR, rb), (SR, sr), (FR, fr)):
            for k, v in (d or {}).items():
                setattr(mod, k, v)
        yield
    finally:
        for mod, k, v in saved:
            setattr(mod, k, v)


def hooks(world: World, t=None, rb=None, sr=None, fr=None) -> dict:
    return {"t": {"REPO_STATE": lambda root: dict(STATE), **(t or {})}, "rb": dict(rb or {}),
            "sr": {"SCHEMA_CONFIG_SHA256": world.schema_sha, "CELLS_CONFIG_SHA256": world.cells_cfg_sha, **(sr or {})},
            "fr": {"FRAME_SHAPE": RBS.SHAPE, **(fr or {})}}


def run_main(world: World, argv, **kw) -> tuple:
    out, err = io.StringIO(), io.StringIO()
    with injected(**hooks(world, **kw)), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = T.main(argv)
    return rc, out.getvalue(), err.getvalue()


def run_folder(out: Path, date=DATE) -> Path:
    return out / f"{date}-BNTEST-{STATE['commit']}"


def tree_state(folder: Path) -> dict:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns, sha(p.read_bytes()) if os.access(p, os.R_OK) else "unreadable")
            for p in sorted(folder.rglob("*")) if p.is_file()}


WORLD = None
PASS = {}


def world() -> World:
    global WORLD
    if WORLD is None:
        WORLD = World()
        RBS.WORLD = WORLD          # la suite della PR #27 chiude lo stesso mondo, se usata
    return WORLD


def load(w: World, doc=None, need_note=True):
    cfg, _, bn = T.load_config(w.write_bt_config(doc or w.bt_doc()), w.base)
    with injected(**hooks(w)):
        return T.load_inputs(cfg, bn, w.base, need_note=need_note)


def pass_run() -> dict:
    if not PASS:
        w = world()
        before = (tree_state(w.parent), tree_state(w.bnrecal))
        rc, out, err = run_main(w, w.bt_argv("run"))
        PASS.update({"rc": rc, "out": out, "err": err, "run": run_folder(w.out), "before": before,
                     "after": (tree_state(w.parent), tree_state(w.bnrecal))})
    return PASS


ORDER = [SR.model_name(*x) for x in SR.training_order(CFR.K, SAR.SEEDS)]


# ============================================================================ configurazione

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.w = world()

    def tearDown(self):
        self.w.write_bt_config(self.w.bt_doc())

    def refused(self, mutate, needle: str):
        doc = self.w.bt_doc()
        mutate(doc)
        with self.assertRaises(FR.Failure) as cm:
            T.load_config(self.w.write_bt_config(doc), self.w.base)
        self.assertIn(needle, str(cm.exception))

    def test_synthetic_configuration_accepted(self):
        cfg, h, bn = T.load_config(self.w.write_bt_config(self.w.bt_doc()), self.w.base)
        self.assertEqual(h, sha(self.w.bt_config.read_bytes()))
        self.assertEqual(bn["estimator"], cfg["estimator"])

    def test_closed_sets_and_texts(self):
        R = self.refused
        R(lambda d: d.update(extra=1), "fields must be exactly")
        R(lambda d: d.pop("post_hoc", None) or d.pop("declarations"), "fields must be exactly")
        R(lambda d: d.update(schema="x"), "schema must be")
        R(lambda d: d.update(test="the test is not read"), "test must state exactly")
        R(lambda d: d.update(note_sha256="abc"), "note_sha256 must be null or a SHA-256")
        R(lambda d: d["declarations"].update(post_hoc="a diagnostic extension"), "declarations.post_hoc must say")
        R(lambda d: d["declarations"].update(threshold="a fixed threshold"), "declarations.threshold must state")
        R(lambda d: d["declarations"].pop("cause"), "declarations needs exactly")
        R(lambda d: d["bnrecal_run"].update(run_id="20261114-X-abc"), "bnrecal_run.run_id")
        R(lambda d: d["bnrecal_run"].update(public_sha256sums_sha256="x"), "bnrecal_run SHA-256 values malformed")
        R(lambda d: d["bnrecal_run"].pop("used"), "bnrecal_run needs exactly")
        R(lambda d: d["code"].pop("scripts/path_guard.py"), "code needs exactly")
        R(lambda d: d["code"].update({T.ESTIMATOR_FILE: "1" * 64}), "not the estimator of PR #27")
        R(lambda d: d["tolerances"].pop("test_probability_max_abs_diff"), "tolerances needs exactly")
        R(lambda d: d["tolerances"]["test_probability_max_abs_diff"].update(value=0), "must be a finite positive number")
        R(lambda d: d["tolerances"]["test_decisions_discordant_at_0_5"].update(value=0.5), "must be a non-negative integer")
        R(lambda d: d["tolerances"]["recalibration_statistics_rel"].update(source=""), "recalibration_statistics_rel.source")
        R(lambda d: d["stop_rules"].update(baseline="stop"), "BASELINE_FAILED")
        R(lambda d: d["stop_rules"].update(recalibration="stop"), "RECALIBRATION_MISMATCH")
        R(lambda d: d["stop_rules"].update(fail="FAIL"), "stop_rules.fail must state")
        R(lambda d: d["comparison"].update(test_frames_per_fold=[]), "comparison.test_frames_per_fold")
        R(lambda d: d["comparison"].update(test_frames_per_fold=[80, 0, 80, 80]), "comparison.test_frames_per_fold")
        R(lambda d: d["dev_worsened"].update(expected=["0003"]), "dev_worsened.expected must list")
        R(lambda d: d["comparison"].update(scopes=["test"]), "comparison must have scopes")
        R(lambda d: d["comparison"].update(metrics=["acer_05", "auroc"]), "comparison must have scopes")
        R(lambda d: d["comparison"].update(threshold=0.4), "comparison.threshold must be 0.5")
        R(lambda d: d["comparison"].update(decision="attack if score_attack >= threshold"), "comparison.threshold must be 0.5")
        R(lambda d: d["comparison"].update(undefined=""), "comparison.undefined")
        R(lambda d: d["bootstrap"].update(resamples=999), "bootstrap resamples, seed, percentiles and min_subjects")
        R(lambda d: d["bootstrap"].update(seed=1), "bootstrap resamples, seed, percentiles and min_subjects")
        R(lambda d: d["bootstrap"].update(min_subjects=3), "bootstrap resamples, seed, percentiles and min_subjects")
        R(lambda d: d["bootstrap"].update(summaries={"groups": ["ten"], "arms": list(SR.ARMS)}), "bootstrap.summaries")
        R(lambda d: d["bootstrap"].update(combination=""), "bootstrap.combination")
        R(lambda d: d["reading_rule"].update(below="below {who}"), "reading_rule.below must contain the placeholders")
        R(lambda d: d["reading_rule"].update(not_available="none {who}"), "reading_rule.not_available must contain")
        R(lambda d: d["reading_rule"].update(count="{lower} of {n}"), "reading_rule.count must contain")
        R(lambda d: d["reading_rule"].update(limits="the saturation is caused by the statistics"), "causal or conclusive wording")
        R(lambda d: d["declarations"].update(cause="this proves the cause"), "causal or conclusive wording")
        R(lambda d: d["outputs"].update(public=["results.csv"]), "outputs.public must be")
        R(lambda d: d["outputs"].update(restricted=["x"]), "outputs.restricted must list")
        R(lambda d: d["budget"].update(wall_time_cap_minutes=241), "at most 240")
        R(lambda d: d["budget"].update(estimate_minutes=120), "estimate_minutes <= upper_estimate_minutes < wall_time_cap_minutes")
        R(lambda d: d["budget"].update(upper_estimate_minutes=240), "estimate_minutes <= upper_estimate_minutes < wall_time_cap_minutes")
        R(lambda d: d["budget"].update(on_cap_exceeded="stop"), "INCOMPLETE outcome")
        R(lambda d: d["environment"].pop("matplotlib"), "environment needs exactly")

    def test_rewritten_descriptive_keys(self):
        doc = self.w.bt_doc()
        doc["parent_run"]["used"] = "another true sentence"
        doc["evaluation"]["probability"] = "another true sentence"
        T.load_config(self.w.write_bt_config(doc), self.w.base)                       # chiavi riscritte: ammesse
        self.assertEqual(T.SHARED_REWRITTEN, {"parent_run": ("used",), "evaluation": ("probability",)})
        self.refused(lambda d: d["parent_run"].update(used=""), "parent_run.used must be a non-empty text")
        self.refused(lambda d: d["parent_run"].update(extra="x"), "parent_run differ from the PR #27 configuration")
        self.refused(lambda d: d["evaluation"].pop("probability"), "evaluation differ from the PR #27 configuration")
        self.refused(lambda d: d["evaluation"].update(device="GPU"), "evaluation differ from the PR #27 configuration")
        self.refused(lambda d: d["parent_run"].update(config_sha256="9" * 64), "parent_run differ from the PR #27 configuration")

    def test_shared_with_pr27(self):
        R = self.refused
        R(lambda d: d["estimator"].update(variance="sample variance"), "estimator differ from the PR #27 configuration")
        R(lambda d: d["batchnorm"].update(momentum=0.9), "batchnorm differ from the PR #27 configuration")
        R(lambda d: d["evaluation"].update(batch=8), "evaluation differ from the PR #27 configuration")
        R(lambda d: d["groups"]["ten"].update(expected=RBS.TEN[1:]), "groups differ from the PR #27 configuration")
        R(lambda d: d["parent_run"].update(models=35), "parent_run differ from the PR #27 configuration")
        R(lambda d: d["tolerances"]["val_loss_abs"].update(value=0.01), "tolerances val_loss_abs differ from the PR #27")
        R(lambda d: d["tolerances"]["dev_probability_max_abs_diff"].update(source="other"), "tolerances dev_probability_max_abs_diff differ")
        R(lambda d: d["bnrecal_run"].update(config_sha256="1" * 64), "PR #27 configuration bnrecal.json: SHA-256")
        R(lambda d: d["environment"].update(tensorflow="2.16.0"), "environment.tensorflow differs from the PR #27")

    def test_delivered_configuration(self):
        p = ROOT / T.DEFAULT_CONFIG
        if not p.is_file():
            skip_missing(self, "delivered configuration")
        cfg, _, bn = T.load_config(p, ROOT)
        self.assertEqual({k: v["value"] for k, v in cfg["tolerances"].items() if k in T.NEW_TOLERANCES},
                         {"test_probability_max_abs_diff": 1e-05, "test_decisions_discordant_at_0_5": 0, "recalibration_statistics_rel": 0.0001,
                          "recalibration_statistics_abs": 1e-06, "recalibration_dev_probability_max_abs_diff": 1e-05,
                          "recalibration_dev_decisions_discordant_at_0_5": 0})
        self.assertEqual(cfg["tolerances"]["recalibration_statistics_rel"]["value"], bn["tolerances"]["consistency_rel"]["value"])
        self.assertEqual(cfg["tolerances"]["recalibration_statistics_abs"]["value"], bn["tolerances"]["consistency_abs"]["value"])
        self.assertEqual((cfg["budget"]["estimate_minutes"], cfg["budget"]["upper_estimate_minutes"], cfg["budget"]["wall_time_cap_minutes"]),
                         (42, 112, 240))
        self.assertEqual(cfg["comparison"]["test_frames_per_fold"], [3242, 3114, 3439, 2819])
        self.assertIn("rewritten for this run", cfg["parent_run"]["used"])
        self.assertIn("rewritten for this run", cfg["evaluation"]["probability"])
        self.assertNotIn("never opened", cfg["parent_run"]["used"])
        self.assertEqual(cfg["code"][T.ESTIMATOR_FILE], T.ESTIMATOR_SHA256)
        self.assertEqual(len(cfg["dev_worsened"]["expected"]), 17)
        self.assertEqual(cfg["bnrecal_run"]["run_id"], "20261114-BNRECAL-a10a082")
        self.assertEqual(cfg["parent_run"]["run_id"], "20261106-SESSABLRUN-d999458")
        self.assertEqual(cfg["test"], T.TEST_READ)
        self.assertEqual({k: cfg["bootstrap"][k] for k in SR.BOOT}, SR.BOOT)
        self.assertTrue(cfg["note_sha256"] is None or len(cfg["note_sha256"]) == 64)


# ============================================================================ ingressi (fase 0)

class TestInputs(unittest.TestCase):
    def setUp(self):
        self.w = world()

    def tearDown(self):
        self.w.write_bt_config(self.w.bt_doc())

    def refused(self, needle: str, doc=None, **kw):
        with self.assertRaises(FR.Failure) as cm:
            if kw:
                with injected(**kw):
                    load(self.w, doc)
            else:
                load(self.w, doc)
        self.assertIn(needle, str(cm.exception))

    def test_synthetic_inputs_pass(self):
        inp = load(self.w)
        self.assertEqual(inp.order, ORDER)
        self.assertEqual(sorted(inp.folds), [1, 2, 3, 4])
        self.assertEqual(sorted(inp.test_files), sorted(ORDER))
        self.assertEqual(sorted(inp.bnrecal), sorted(ORDER))
        self.assertEqual(inp.dev_worsened, [n for n in ORDER if n in self.w.dev_worsened()])
        self.assertEqual(inp.checks["groups"]["ten"], [n for n in ORDER if n in RBS.TEN])
        self.assertIn("parsed in phase 1", inp.checks["parent_run"]["not_opened"])

    def test_check_inputs_command(self):
        before = tree_state(self.w.tmp)
        rc, out, err = run_main(self.w, self.w.bt_argv("check-inputs"))
        self.assertEqual(rc, 0, err)
        self.assertIn("RESULT: PASS (check-inputs)", out)
        self.assertIn("test scores hashed only, not parsed", out)
        self.assertEqual(tree_state(self.w.tmp), before)
        doc = self.w.bt_doc()
        doc["note_sha256"] = None
        self.w.write_bt_config(doc)
        rc, out, err = run_main(self.w, self.w.bt_argv("check-inputs"))
        self.assertEqual(rc, 0, err)
        self.assertIn("note null (not yet frozen)", out)

    def test_note_hash(self):
        doc = self.w.bt_doc()
        doc["note_sha256"] = None
        self.refused("note_sha256 is null", doc)
        self.assertIsNone(load(self.w, doc, need_note=False).note_sha)
        doc["note_sha256"] = "0" * 64
        self.refused("differs from note_sha256 of the configuration", doc)

    def test_code_pins(self):
        doc = self.w.bt_doc()
        doc["code"]["scripts/m2_writers.py"] = "2" * 64
        self.refused("differ from the SHA-256 pinned in the configuration", doc)
        with injected(t={"ESTIMATOR_SHA256": "3" * 64}):         # file diverso dal pin: rifiutato prima del confronto con la costante
            doc = self.w.bt_doc()
            doc["code"][T.ESTIMATOR_FILE] = "3" * 64
            self.refused(f"code files ['{T.ESTIMATOR_FILE}'] differ from the SHA-256 pinned", doc)

    def test_test_manifests_opened_and_checked(self):
        p = self.w.base / "schema" / "fold2_test.csv"
        p.chmod(0)
        try:
            with self.assertRaises((FR.Failure, PermissionError)) as cm:
                load(self.w)
            self.assertIn("fold2_test.csv", str(cm.exception))
        finally:
            p.chmod(0o644)

    def test_test_frames_per_fold(self):
        doc = self.w.bt_doc()
        doc["comparison"]["test_frames_per_fold"] = [x + 1 for x in doc["comparison"]["test_frames_per_fold"]]
        self.refused("differ from comparison.test_frames_per_fold", doc)

    def test_folds_run_public_never_opened(self):
        self.assertTrue(all(not os.access(f, os.R_OK) for f in (self.w.base / "foldsrun" / "public").iterdir()))
        load(self.w)

    def test_parent_test_scores_hashed(self):
        n = "arm_b_prime_fold2_seed43"
        p = self.w.parent / "restricted" / f"{n}_test_scores.csv"
        saved = p.read_bytes()
        try:
            p.write_bytes(saved + b"\n")
            self.refused(f"parent {n}_test_scores.csv: SHA-256")
        finally:
            p.write_bytes(saved)

    def test_parent_test_scores_unlisted(self):
        n = "arm_a_star_fold3_seed44"
        inp_cfg = self.w.bt_doc()
        s = self.w.parent / "restricted" / "SHA256SUMS"
        saved = s.read_text()
        try:
            s.write_text("".join(x + "\n" for x in saved.splitlines() if not x.endswith(f"{n}_test_scores.csv")))
            cfg, _, bn = T.load_config(self.w.write_bt_config(inp_cfg), self.w.base)
            inp = T.Inputs()
            inp.order = [n]
            cfg = copymod.deepcopy(cfg)
            cfg["parent_run"]["restricted_sha256sums_sha256"] = sha(s.read_bytes())
            inp.checks["parent_run"] = {}
            with self.assertRaisesRegex(FR.Failure, f"{n}_test_scores.csv not listed"):
                T.load_parent_test_files(cfg, self.w.base, inp)
        finally:
            s.write_text(saved)

    def test_bnrecal_anchors(self):
        doc = self.w.bt_doc()
        doc["bnrecal_run"]["run_id"] = "20261114-BNRECAL-0000000"
        self.refused("status.json is not PASS", doc)
        for k in ("public_sha256sums_sha256", "restricted_sha256sums_sha256"):
            doc = self.w.bt_doc()
            doc["bnrecal_run"][k] = "1" * 64
            self.refused("status.json is not PASS with the run id", doc)
        st = self.w.bnrecal / "status.json"
        saved = st.read_text()
        try:
            st.write_text(json.dumps({**json.loads(saved), "status": "FAIL"}))
            self.refused("status.json is not PASS")
            st.write_text(json.dumps({**json.loads(saved), "completed": json.loads(saved)["completed"][:-1]}))
            self.refused("the completed models differ")
        finally:
            st.write_text(saved)
        r = self.w.bnrecal / "restricted" / "SHA256SUMS"
        saved = r.read_bytes()
        try:
            r.write_bytes(saved + b"\n")
            self.refused("restricted SHA256SUMS has SHA-256")
        finally:
            r.write_bytes(saved)

    def test_bnrecal_files(self):
        n = "arm_b_fold2_seed43"
        for fn, needle in ((f"{n}_scores.csv", f"BNRECAL {n}_scores.csv: SHA-256 differs"),
                           (f"{n}_bn_statistics.json", f"BNRECAL {n}_bn_statistics.json: SHA-256 differs")):
            p = self.w.bnrecal / "restricted" / fn
            saved = p.read_bytes()
            try:
                p.write_bytes(saved + b" ")
                self.refused(needle)
            finally:
                p.write_bytes(saved)

    def test_bnrecal_content_checks(self):
        n = "arm_a_star_fold1_seed43"

        def stats_weights(d):
            p = d / "restricted" / f"{n}_bn_statistics.json"
            p.write_text(json.dumps({**json.loads(p.read_text()), "weights_sha256": "4" * 64}))
        with altered_bnrecal(self.w, stats_weights):
            self.refused(f"BNRECAL {n}_bn_statistics.json: checkpoint, estimator or layers differ")

        def stats_nan(d):
            p = d / "restricted" / f"{n}_bn_statistics.json"
            js = json.loads(p.read_text())
            js["layers"]["bn2_1"]["installed_moving_variance_float32"][0] = float("nan")
            p.write_text(json.dumps(js))
        with altered_bnrecal(self.w, stats_nan):
            self.refused("installed statistics of bn2_1 malformed or not finite")

        def stats_missing(d):
            p = d / "restricted" / f"{n}_bn_statistics.json"
            js = json.loads(p.read_text())
            del js["layers"]["bn1_2"]["installed_moving_mean_float32"]
            p.write_text(json.dumps(js))
        with altered_bnrecal(self.w, stats_missing):
            self.refused("installed statistics of bn1_2 missing or not numeric")

        def scores_order(d):
            p = d / "restricted" / f"{n}_scores.csv"
            rows = read_csv(p)
            dev = [i for i, r in enumerate(rows) if r["role"] == "dev" and r["statistics"] == "re-estimated"]
            rows[dev[0]], rows[dev[1]] = rows[dev[1]], rows[dev[0]]
            write_rows(p, BR.SCORE_COLUMNS, rows)
        with altered_bnrecal(self.w, scores_order):
            self.refused("the re-estimated dev rows are not the dev of this model in key order")

        def group_changed(d):
            p = d / "public" / "results.csv"
            rows = read_csv(p)
            rows[0]["group"] = "control" if rows[0]["group"] != "control" else "ten"
            write_rows(p, list(rows[0]), rows)
        with altered_bnrecal(self.w, group_changed, public_changed=True):
            self.refused("BNRECAL results.csv: a group differs")

        def code_changed(d):
            p = d / "public" / "run_record.json"
            js = json.loads(p.read_text())
            js["code_sha256"]["scripts/nuaa_folds_run.py"] = "5" * 64
            p.write_text(json.dumps(js))
        with altered_bnrecal(self.w, code_changed, public_changed=True):
            self.refused("code_sha256 differs from the code pinned")

        def pubsums(d):
            (d / "restricted" / "public_SHA256SUMS.sha256").write_text("0" * 64 + "  public/SHA256SUMS\n")
        with altered_bnrecal(self.w, pubsums):
            self.refused("public_SHA256SUMS.sha256 does not name the public SHA256SUMS")

    def test_dev_worsened_list(self):
        doc = self.w.bt_doc()
        doc["dev_worsened"]["expected"] = doc["dev_worsened"]["expected"][1:] if doc["dev_worsened"]["expected"] else [ORDER[0]]
        self.refused("the models with a worsened dev indicator", doc)

        def flag(d):
            p = d / "public" / "results.csv"
            rows = read_csv(p)
            rows[0]["fitting_loss_higher"] = "2"
            write_rows(p, list(rows[0]), rows)
        with altered_bnrecal(self.w, flag, public_changed=True):
            self.refused("a worsening flag is not 0 or 1")

    def test_environment(self):
        doc = self.w.bt_doc()
        doc["environment"]["matplotlib"] = "0.0.0"
        with self.assertRaises(FR.Failure) as cm:
            load(self.w, doc)
        self.assertIn("matplotlib", str(cm.exception))
        with RBS.mock_env("LIVENESSLAB_LIVEDETECTION", "x"):
            self.refused("environment variables")


# ============================================================================ baseline del test, controllo della ristima

class TestBaselineUnits(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(json.dumps(world().bt_doc()))
        self.s = np.array([0.1, 0.4999, 0.5001, 0.9])

    def check(self, p, stored=None, gap=0.0, problem="", cfg=None):
        stored = self.s if stored is None else stored
        return T.test_checks(cfg or self.cfg, np.asarray(p, float), gap, stored, problem, len(self.s))

    def test_tolerance_boundaries(self):
        self.assertTrue(self.check(self.s)["test_pass"])
        p = self.s + np.array([0, 0, 0, 1e-5])
        d = float(np.max(np.abs(p - self.s)))
        for value, ok in ((d, True), (float(np.nextafter(d, 0)), False)):                         # = tolleranza: passa
            cfg = copymod.deepcopy(self.cfg)
            cfg["tolerances"]["test_probability_max_abs_diff"]["value"] = value
            self.assertEqual(self.check(p, cfg=cfg)["test_prob_pass"], ok)
        self.assertFalse(self.check(self.s + np.array([0, 0, 0, 1.0001e-5]))["test_prob_pass"])
        self.assertTrue(self.check(self.s + np.array([0, 0, 0, 0.9999e-5]))["test_prob_pass"])
        c = self.check(self.s + np.array([0, 0, 0, 2e-5]))
        self.assertEqual((c["test_pass"], c["test_discordant_05"]), (False, 0))
        cfg = copymod.deepcopy(self.cfg)
        cfg["tolerances"]["test_probability_max_abs_diff"]["value"] = 1e-4
        self.assertTrue(self.check(self.s + np.array([0, 0, 0, 2e-5]), cfg=cfg)["test_pass"])

    def test_discordant_with_the_tie_rule(self):
        stored = np.array([0.1, 0.5, 0.5000004, 0.9])
        c = self.check(np.array([0.1, 0.5000002, 0.5000004, 0.9]), stored=stored)                   # 0,5 è bona fide, 0,5000002 attacco
        self.assertEqual(c["test_discordant_05"], 1)
        self.assertLess(c["test_prob_max_abs_diff"], 1e-5)
        self.assertFalse(c["test_pass"])
        c = self.check(np.array([0.1, 0.4999998, 0.5000004, 0.9]), stored=stored)                   # sotto 0,5: stessa decisione
        self.assertEqual((c["test_discordant_05"], c["test_pass"]), (0, True))
        cfg = copymod.deepcopy(self.cfg)
        cfg["tolerances"]["test_decisions_discordant_at_0_5"]["value"] = 1
        self.assertTrue(self.check(np.array([0.1, 0.5000002, 0.5000004, 0.9]), stored=stored, cfg=cfg)["test_pass"])

    def test_softmax_and_frames(self):
        c = self.check(self.s, gap=2e-4)
        self.assertEqual((c["test_softmax_pass"], c["test_pass"]), (False, False))
        self.assertTrue(self.check(self.s, gap=1e-4)["test_softmax_pass"])
        c = T.test_checks(self.cfg, self.s, 0.0, None, "4 stored rows for 5 assembled frames", 5)
        self.assertEqual((c["test_frames_pass"], c["test_prob_max_abs_diff"], c["test_pass"]), (False, None, False))
        self.assertEqual(set(T.BASELINE_COLUMNS) - {"arm", "fold", "seed", "n_dev", "dev_pass", "baseline_pass"} - set(T.DEV_COLUMNS), set(c))

    def test_stored_rows_alignment(self):
        w = world()
        inp = load(w)
        name = "arm_b_fold2_seed44"
        fold = inp.parent[name]["fold"]
        store = FR.FrameStore(inp, inp.X_extracted, inp.extracted_keys)
        with injected(**hooks(w)):
            part = FR.assemble(store, inp.folds[fold]["test"])
        s, prob = T.stored_test(inp, name, part)
        self.assertEqual(prob, "")
        self.assertEqual(len(s), len(part["keys"]))
        p = inp.test_files[name][0]
        rows = read_csv(p)
        saved = p.read_bytes()
        cases = [(lambda rs: rs.reverse(), "sample ids or their order differ"),
                 (lambda rs: rs[3].update(label="1" if rs[3]["label"] == "0" else "0"), "label, subject, session or file SHA-256 differ"),
                 (lambda rs: rs[2].update(subject_id=CFR.SUBJECTS[-1]), "label, subject, session or file SHA-256 differ"),
                 (lambda rs: rs[2].update(session_id="02" if rs[2]["session_id"] != "02" else "01"), "label, subject, session"),
                 (lambda rs: rs[1].update(file_sha256="6" * 64), "label, subject, session or file SHA-256 differ"),
                 (lambda rs: rs.pop(), "stored rows for"),
                 (lambda rs: rs[0].update(role="dev"), "rows of another model, role, checkpoint or threshold"),
                 (lambda rs: rs[0].update(checkpoint_sha256="7" * 64), "rows of another model"),
                 (lambda rs: rs[0].update(threshold_sha256="7" * 64), "rows of another model"),
                 (lambda rs: rs[0].update(train_seed="43"), "rows of another model"),
                 (lambda rs: rs[0].update(prediction_05="1" if rs[0]["prediction_05"] == "0" else "0"), "stored prediction_05 differs")]
        try:
            for mut, needle in cases:
                rs = copymod.deepcopy(rows)
                mut(rs)
                write_rows(p, SR.SCORE_COLUMNS, rs)
                inp.test_files[name] = (p, sha(p.read_bytes()))
                s, prob = T.stored_test(inp, name, part)
                self.assertIsNone(s)
                self.assertIn(needle, prob)
                for subj in CFR.SUBJECTS:
                    self.assertNotIn(subj, prob)
            rs = copymod.deepcopy(rows)
            rs[0]["score_attack"] = "x"
            write_rows(p, SR.SCORE_COLUMNS, rs)
            inp.test_files[name] = (p, sha(p.read_bytes()))
            with self.assertRaisesRegex(FR.Failure, "score_attack not numeric"):
                T.stored_test(inp, name, part)
            rs[0]["score_attack"] = "1.5"
            write_rows(p, SR.SCORE_COLUMNS, rs)
            inp.test_files[name] = (p, sha(p.read_bytes()))
            with self.assertRaisesRegex(FR.Failure, r"score_attack outside \[0, 1\]"):
                T.stored_test(inp, name, part)
            inp.test_files[name] = (p, "8" * 64)
            with self.assertRaisesRegex(FR.Failure, "SHA-256 changed since phase 0"):
                T.stored_test(inp, name, part)
        finally:
            p.write_bytes(saved)


class TestRecalCheck(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(json.dumps(world().bt_doc()))
        m = np.array([0.5, -2.0, 0.0], np.float32)
        v = np.array([1.0, 3.0, 0.0], np.float32)
        self.ref = {"bn1_1": (m, v), "bn_dense": (m * 2, v * 2)}
        self.stats = {n: {"mean32": a.copy(), "var32": b.copy()} for n, (a, b) in self.ref.items()}
        self.p = np.array([0.2, 0.7, 0.4])

    def check(self, stats=None, p=None):
        return T.recal_check(self.cfg, stats or self.stats, self.ref, self.p if p is None else p, self.p)

    def test_identical(self):
        c = self.check()
        self.assertEqual((c["statistics_identical"], c["statistics_outside_tolerance"], c["recalibration_pass"]), (True, 0, True))

    def test_reported_differences(self):
        """Colonne riportate, non vincolanti: scarto massimo di medie e varianze su tutti gli strati, canali fuori dalla
        coerenza sommati sugli strati (valori noti)."""
        st = copymod.deepcopy(self.stats)
        st["bn1_1"]["mean32"][0] = np.float32(0.5 + 0.25)                    # scarto 0,25 in una media
        st["bn_dense"]["mean32"][1] = np.float32(-4.0 - 0.125)               # 0,125 in un altro strato: vale il massimo
        st["bn_dense"]["var32"][1] = np.float32(6.0 + 0.5)                   # 0,5 in una varianza
        c = self.check(st)
        self.assertEqual((c["statistics_max_abs_diff_mean"], c["statistics_max_abs_diff_variance"]), (0.25, 0.5))
        self.assertEqual(c["statistics_outside_tolerance"], 3)
        inp = T.Inputs()
        inp.order = ["arm_b_fold1_seed42"]
        inp.parent = {"arm_b_fold1_seed42": {"arm": "arm_b", "fold": 1, "seed": 42}}
        inp.groups = {"arm_b_fold1_seed42": "ten"}
        cons = {"bn1_1": {"channels_outside_tolerance": 2}, "bn_dense": {"channels_outside_tolerance": 3}}
        row = read_csv_text(T.recalibration_csv(inp, {"arm_b_fold1_seed42": {"check": c, "cons": cons}}))[0]
        self.assertEqual((row["consistency_channels_outside"], row["statistics_max_abs_diff_mean"], row["statistics_max_abs_diff_variance"]),
                         ("5", "0.25", "0.5"))

    def test_relative_and_absolute_tolerance(self):
        st = copymod.deepcopy(self.stats)
        st["bn1_1"]["mean32"][1] = np.float32(-2.0 * (1 + 5e-5))
        c = self.check(st)
        self.assertEqual((c["statistics_identical"], c["statistics_pass"]), (False, True))
        st["bn1_1"]["mean32"][1] = np.float32(-2.0 * (1 + 2e-4))
        self.assertEqual(self.check(st)["statistics_outside_tolerance"], 1)
        st = copymod.deepcopy(self.stats)
        st["bn_dense"]["var32"][2] = np.float32(9e-7)                         # riferimento 0: solo la parte assoluta
        self.assertTrue(self.check(st)["statistics_pass"])
        st["bn_dense"]["var32"][2] = np.float32(1.2e-6)
        self.assertFalse(self.check(st)["statistics_pass"])
        st = copymod.deepcopy(self.stats)
        st["bn_dense"]["mean32"][0] = np.float32(np.nan)
        self.assertEqual(self.check(st)["statistics_outside_tolerance"], 1)
        st = copymod.deepcopy(self.stats)
        st["bn1_1"]["var32"] = np.zeros(4, np.float32)
        self.assertFalse(self.check(st)["recalibration_pass"])
        cfg = copymod.deepcopy(self.cfg)
        cfg["tolerances"]["recalibration_statistics_rel"]["value"] = 1e-2
        st = copymod.deepcopy(self.stats)
        st["bn1_1"]["mean32"][1] = np.float32(-2.0 * (1 + 2e-4))
        self.assertTrue(T.recal_check(cfg, st, self.ref, self.p, self.p)["statistics_pass"])

    def test_dev_probabilities(self):
        p = self.p + np.array([0, 1e-5, 0])
        d = float(np.max(np.abs(p - self.p)))
        for value, ok in ((d, True), (float(np.nextafter(d, 0)), False)):
            cfg = copymod.deepcopy(self.cfg)
            cfg["tolerances"]["recalibration_dev_probability_max_abs_diff"]["value"] = value
            self.assertEqual(T.recal_check(cfg, self.stats, self.ref, p, self.p)["dev_pass"], ok)
        self.assertTrue(self.check(p=self.p + np.array([0, 0.9999e-5, 0]))["dev_pass"])
        self.assertFalse(self.check(p=self.p + np.array([0, 1.0001e-5, 0]))["dev_pass"])
        c = T.recal_check(self.cfg, self.stats, self.ref, np.array([0.5000001, 0.7]), np.array([0.5, 0.7]))
        self.assertEqual((c["dev_discordant_05"], c["dev_pass"]), (1, False))
        c = T.recal_check(self.cfg, self.stats, self.ref, np.array([0.5]), np.array([0.5, 0.7]))
        self.assertEqual((c["dev_prob_max_abs_diff"], c["dev_pass"]), (None, False))


# ============================================================================ metriche, indicatori, bootstrap, lettura

class TestMetrics(unittest.TestCase):
    def test_against_independent_reference(self):
        rng = np.random.default_rng(5)
        y = rng.integers(0, 2, 60)
        s = np.round(rng.uniform(0, 1, 60), 2)
        s[:5] = 0.5                                                           # pareggi alla soglia: bona fide
        subj = rng.choice(["1101", "1102", "1103"], 60)
        rows = rows_of(y, s, subj)
        got, ref = T.scope_metrics(rows), ref_metrics(y, s)
        for k in T.METRICS:
            self.assertAlmostEqual(got[k], ref[k], places=12)
        self.assertEqual((got["n_bona_fide"], got["n_attack"], got["n_subjects"]), (int((y == 0).sum()), int((y == 1).sum()), 3))

    def test_class_absent_and_empty_scope(self):
        rows = rows_of([1, 1, 1], [0.2, 0.7, 0.9], ["1101"] * 3)
        m = T.scope_metrics(rows)
        self.assertAlmostEqual(m["apcer_05"], 1 / 3)
        self.assertEqual((m["bpcer_05"], m["acer_05"], m["auroc"]), (None, None, None))
        e = T.scope_metrics([])
        self.assertEqual((e["n"], e["n_bona_fide"]) + tuple(e[k] for k in T.METRICS), (0, 0, None, None, None, None))

    def test_scopes(self):
        rows = rows_of([0, 1], [0.1, 0.9], ["1101"] * 2, "01") + rows_of([0, 1], [0.6, 0.2], ["1101"] * 2, "03")
        self.assertEqual(len(T.scope_rows(rows, "test")), 4)
        self.assertEqual({r["session_id"] for r in T.scope_rows(rows, "session_03")}, {"03"})

    def test_flags_and_rows(self):
        self.assertIs(T.flag_of("acer_05", 0.1, 0.1), False)
        self.assertIs(T.flag_of("acer_05", 0.1, 0.1000001), True)
        self.assertIs(T.flag_of("apcer_05", 0.2, 0.1), False)
        self.assertIs(T.flag_of("bpcer_05", 0.0, 0.01), True)
        self.assertIs(T.flag_of("auroc", 0.9, 0.8999), True)
        self.assertIs(T.flag_of("auroc", 0.9, 0.95), False)
        self.assertIsNone(T.flag_of("auroc", None, 0.9))
        b = {"n_bona_fide": 2, "n_attack": 3, "n_subjects": 2, "apcer_05": 0.5, "bpcer_05": 0.0, "acer_05": 0.25, "auroc": 0.9}
        a = {**b, "apcer_05": 0.0, "acer_05": 0.0, "auroc": None, "bpcer_05": 0.1}
        m = {"arm": "arm_b", "fold": 1, "seed": 42}
        r = T.result_row("x", m, "ten", True, "test", b, a)
        self.assertEqual((r["acer_05_difference"], r["bpcer_05_difference"], r["auroc_difference"]), (-0.25, 0.1, None))
        self.assertEqual((r["acer_higher"], r["bpcer_higher"], r["apcer_higher"], r["auroc_lower"]), (False, True, False, None))
        with self.assertRaisesRegex(FR.Failure, "not on the same frames"):
            T.result_row("x", m, "ten", True, "test", b, {**a, "n_attack": 4})
        self.assertEqual(T.results_csv([r]).splitlines()[1].split(",")[-4:], ["0", "0", "1", "NA"])


def subj_world(n_subj: int, fold_seed: int, shift: float, session="03") -> tuple:
    """Righe prima e dopo per n_subj soggetti, stessi frame; dopo = prima più uno spostamento dei punteggi d'attacco."""
    rng = np.random.default_rng(fold_seed)
    y, s, subj = [], [], []
    for i in range(n_subj):
        for lab in (0, 1):
            for _ in range(3):
                y.append(lab)
                s.append(float(np.clip(rng.normal(0.35 + 0.3 * lab, 0.2), 0, 1)))
                subj.append(f"{1100 + 10 * fold_seed + i:04d}")
    y, s = np.array(y), np.array(s)
    after = np.clip(s + shift * (y == 1), 0, 1)
    return rows_of(y, s, subj, session), rows_of(y, after, subj, session)


class TestBootstrap(unittest.TestCase):
    def test_one_model_equals_the_project_paired_bootstrap(self):
        rb, ra = subj_world(6, 1, 0.2)
        mine = T.paired_group_bootstrap([(1, SR.subject_stats(rb), SR.subject_stats(ra))])
        ref = SR.paired_bootstrap(ra, rb)
        for k in T.METRICS:
            self.assertEqual((mine[k]["low"], mine[k]["high"], mine[k]["valid_resamples"]),
                             (ref[k]["low"], ref[k]["high"], ref[k]["valid_resamples"]), k)
        self.assertEqual((mine["n_subjects"], mine["n_folds"], mine["n_subjects_bona_fide"], mine["n_subjects_attack"]), (6, 1, 6, 6))

    def test_two_folds_pooled_against_frame_resampling(self):
        """Riferimento indipendente: SR.boot_picks sui 7 soggetti uniti in ordine crescente, per ogni modello i frame dei
        soggetti estratti del suo fold, metriche ricalcolate sui frame, media delle differenze dei due modelli; un
        ricampionamento in cui un modello non ha una metrica non conta."""
        f1, f2 = subj_world(3, 1, 0.25), subj_world(4, 2, -0.1)
        mine = T.paired_group_bootstrap([(1, SR.subject_stats(f1[0]), SR.subject_stats(f1[1])),
                                         (2, SR.subject_stats(f2[0]), SR.subject_stats(f2[1]))])
        pooled = sorted({r["subject_id"] for r in f1[0] + f2[0]})
        vals = {k: [] for k in T.METRICS}
        for pick in SR.boot_picks(len(pooled)):
            chosen = [pooled[i] for i in pick]
            diffs = {k: [] for k in T.METRICS}
            for b, a in (f1, f2):
                def take(rows):
                    out = [r for c in chosen for r in rows if r["subject_id"] == c]
                    return np.array([int(r["label"]) for r in out]), np.array([float(r["score_attack"]) for r in out])
                mb, ma = ref_metrics(*take(b)), ref_metrics(*take(a))
                for k in T.METRICS:
                    diffs[k].append(None if mb[k] is None or ma[k] is None else ma[k] - mb[k])
            for k in T.METRICS:
                if all(d is not None for d in diffs[k]):
                    vals[k].append(np.mean(diffs[k]))
        for k in T.METRICS:
            self.assertEqual(mine[k]["valid_resamples"], len(vals[k]), k)
            lo, hi = np.percentile(vals[k], [2.5, 97.5])
            self.assertAlmostEqual(mine[k]["low"], lo, places=10)
            self.assertAlmostEqual(mine[k]["high"], hi, places=10)
        self.assertLess(mine["acer_05"]["valid_resamples"], SR.BOOT["resamples"])     # 3 soggetti su 7: un fold a volte vuoto
        self.assertEqual(mine["n_subjects"], 7)

    def test_pairing(self):
        rb, _ = subj_world(5, 3, 0.0)
        out = T.paired_group_bootstrap([(1, SR.subject_stats(rb), SR.subject_stats(copymod.deepcopy(rb)))])
        for k in T.METRICS:
            self.assertEqual((out[k]["low"], out[k]["high"]), (0.0, 0.0))                 # stessi soggetti prima e dopo
        rb, ra = subj_world(5, 3, -0.4)                                      # attacchi spostati sotto 0,5: ACER più alto
        out = T.paired_group_bootstrap([(1, SR.subject_stats(rb), SR.subject_stats(ra))])
        self.assertGreater(out["acer_05"]["low"], 0)
        self.assertEqual(T.reading_case(0.1, out["acer_05"]), "above")
        _, other = subj_world(5, 4, 0.3)
        with self.assertRaisesRegex(FR.Failure, "not on the same subjects"):
            T.paired_group_bootstrap([(1, SR.subject_stats(rb), SR.subject_stats(other))])
        r2, a2 = subj_world(5, 4, 0.1)
        with self.assertRaisesRegex(FR.Failure, "two models of fold 1"):
            T.paired_group_bootstrap([(1, SR.subject_stats(rb), SR.subject_stats(ra)), (1, SR.subject_stats(r2), SR.subject_stats(a2))])
        with self.assertRaisesRegex(FR.Failure, "a held-out subject appears in two folds"):
            T.paired_group_bootstrap([(1, SR.subject_stats(rb), SR.subject_stats(ra)), (2, SR.subject_stats(rb), SR.subject_stats(ra))])

    def test_few_subjects_and_subject_counts(self):
        rb, ra = subj_world(4, 5, 0.1)
        out = T.paired_group_bootstrap([(3, SR.subject_stats(rb), SR.subject_stats(ra))])     # un fold con quattro soggetti
        self.assertIn("fewer than 5 subjects", out["note"])
        self.assertEqual((out["acer_05"]["low"], out["acer_05"]["valid_resamples"], out["n_subjects"]), (None, 0, 4))
        y = [0, 1, 1, 1, 0, 0]
        rows = rows_of(y, [0.2, 0.7, 0.8, 0.9, 0.1, 0.3], ["1101", "1101", "1102", "1103", "1104", "1105"])
        out = T.paired_group_bootstrap([(1, SR.subject_stats(rows), SR.subject_stats(rows))])
        self.assertEqual((out["n_subjects"], out["n_subjects_bona_fide"], out["n_subjects_attack"]), (5, 3, 3))

    def test_undefined_in_one_model_drops_the_resample(self):
        """Due fold: nel fold 1 un solo soggetto ha bona fide. Un ricampionamento senza di lui non conta per BPCER, ACER e
        AUROC della media, anche se l'altro modello è definito; APCER conta se ogni modello ha attacchi."""
        y = [0, 1] + [1, 1] * 5
        subj = ["1101", "1101"] + [f"11{i:02d}" for i in range(2, 7) for _ in range(2)]
        s = [0.3, 0.8, 0.6, 0.7, 0.45, 0.9, 0.55, 0.65, 0.52, 0.4, 0.75, 0.85]
        rb1, ra1 = rows_of(y, s, subj), rows_of(y, [min(1.0, x + 0.05) for x in s], subj)
        rb2, ra2 = subj_world(5, 7, 0.1)
        s2 = sorted({r["subject_id"] for r in rb2})
        models = [(1, SR.subject_stats(rb1), SR.subject_stats(ra1)), (2, SR.subject_stats(rb2), SR.subject_stats(ra2))]
        out = T.paired_group_bootstrap(models)
        pooled = sorted(set(subj) | set(s2))
        picks = [set(pooled[i] for i in p) for p in SR.boot_picks(len(pooled))]
        f1 = set(subj)
        bf_ok = sum(1 for c in picks if "1101" in c and c & set(s2))
        at_ok = sum(1 for c in picks if c & f1 and c & set(s2))
        self.assertLess(bf_ok, at_ok)
        for k in ("bpcer_05", "acer_05", "auroc"):
            self.assertEqual(out[k]["valid_resamples"], bf_ok, k)
        self.assertEqual(out["apcer_05"]["valid_resamples"], at_ok)


class TestReading(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(json.dumps(world().bt_doc()))

    def test_three_cases_at_the_boundaries(self):
        self.assertEqual(T.reading_case(-0.1, {"low": -0.2, "high": -1e-12}), "below")
        self.assertEqual(T.reading_case(-0.1, {"low": -0.2, "high": 0.0}), "contains")
        self.assertEqual(T.reading_case(0.1, {"low": 0.0, "high": 0.2}), "contains")
        self.assertEqual(T.reading_case(0.1, {"low": 1e-12, "high": 0.2}), "above")
        self.assertEqual(T.reading_case(None, {"low": 0.1, "high": 0.2}), "not_available")
        self.assertEqual(T.reading_case(0.1, {"low": None, "high": None}), "not_available")

    def rows(self, d_ten, d_ctrl=0.0):
        """Righe di results.csv e statistiche per soggetto: per ogni gruppo un modello per fold con differenza fissata."""
        rows, stats = [], {}
        k = 0
        for g, d in (("ten", d_ten), ("control", d_ctrl), ("apart", d_ctrl)):
            for fold in (1, 2, 3, 4) if g != "apart" else (3,):
                arm = SR.ARMS[k % 3]
                k += 1
                for scope in T.SCOPES:
                    rb, ra = subj_world(3, fold, d if scope == "test" else -d)      # ambiti con differenze di segno opposto
                    mb, ma = T.scope_metrics(rb), T.scope_metrics(ra)
                    r = T.result_row("x", {"arm": arm, "fold": fold, "seed": 40 + k}, g, False, scope, mb, ma)
                    rows.append(r)
                    stats[(SR.model_name(arm, fold, 40 + k), scope)] = (SR.subject_stats(rb), SR.subject_stats(ra))
        return rows, stats

    def test_summaries_and_reading_applied(self):
        for d, case in ((0.45, "below"), (0.0, "contains"), (-0.45, "above")):
            rows, stats = self.rows(d)
            summ = T.summaries(self.cfg, rows, stats)
            self.assertEqual(len(summ), (len(T.GROUP_SUMMARIES) + len(SR.ARMS)) * len(T.SCOPES) * len(T.METRICS))
            prim = [s for s in summ if s["label"] == "primary"]
            self.assertEqual(len(prim), 1)
            p = prim[0]
            self.assertEqual((p["summary"], p["scope"], p["metric"]), ("ten", "test", "acer_05"))
            self.assertEqual(p["reading"], case, (d, p))
            self.assertTrue(p["statement"].startswith({"below": "BELOW", "contains": "CONTAINS", "above": "ABOVE"}[case]))
            ten = [r for r in rows if r["group"] == "ten" and r["scope"] == "test"]
            self.assertAlmostEqual(p["mean_difference"], np.mean([r["acer_05_difference"] for r in ten]), places=12)
            a = T.reading(self.cfg, rows, summ)
            c = a["primary"]["count"]
            self.assertEqual((c["lower"], c["equal"], c["higher"], c["n"]),
                             (sum(r["acer_05_difference"] < 0 for r in ten), sum(r["acer_05_difference"] == 0 for r in ten),
                              sum(r["acer_05_difference"] > 0 for r in ten), 4))
            self.assertEqual(a["primary"]["count_statement"], f"COUNT {c['lower']}/{c['equal']}/{c['higher']} of 4")
            apart = [s for s in summ if s["summary"] == "apart"]
            self.assertTrue(all(s["reading"] == "not_available" and s["statement"].startswith("NOREADING") for s in apart))
            n_flags = sum(1 for r in rows for k in T.METRICS if r[T.WORSE[k][0]])
            self.assertEqual(len(a["worsening"]["cases"]), n_flags)
            for w in a["worsening"]["cases"]:                                  # valori delle voci dai campi giusti
                k = next(m for m in T.METRICS if T.WORSE[m][0] == w["flag"])
                r = next(r for r in rows if T.name_of(r) == w["model"] and r["scope"] == w["scope"])
                self.assertEqual((w["before"], w["after"], w["difference"]), (r[f"{k}_before"], r[f"{k}_after"], r[f"{k}_difference"]))
            self.assertEqual(a["primary"]["models_higher"], [T.name_of(r) for r in ten if r["acer_05_difference"] > 0])
            self.assertAlmostEqual(p["mean_before"], np.mean([r["acer_05_before"] for r in ten]), places=12)
            self.assertAlmostEqual(p["mean_after"], np.mean([r["acer_05_after"] for r in ten]), places=12)
            if p["low"] is not None:                                           # segnaposto al loro posto
                self.assertIn(f"{T.METRIC_TEXT['acer_05']} {T.f4(p['mean_difference'])} [{T.f4(p['low'])}, {T.f4(p['high'])}]", p["statement"])
                self.assertIn("group ten (4 models) test", p["statement"])

    def test_figure_error_bar(self):
        self.assertEqual(T.error_bar(0.5, 0.25, 1.0), [[0.25], [0.5]])

    def test_undefined_model_blocks_the_mean(self):
        rows, stats = self.rows(0.3)
        rows[0] = {**rows[0], "auroc_before": None, "auroc_difference": None, "auroc_lower": None}
        summ = T.summaries(self.cfg, rows, stats)
        s = next(x for x in summ if x["summary"] == "ten" and x["scope"] == rows[0]["scope"] and x["metric"] == "auroc")
        self.assertEqual((s["mean_difference"], s["reading"], s["models_undefined"]), (None, "not_available", 1))
        self.assertIn("not defined for at least one model", s["statement"])
        a = T.reading(self.cfg, rows, summ)
        self.assertEqual(len(a["undefined"]), 1)


# ============================================================================ run

class TestRunPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = pass_run()
        cls.folder = cls.r["run"]

    def test_pass_and_outputs(self):
        self.assertEqual(self.r["rc"], 0, self.r["err"][-3000:])
        st = json.loads((self.folder / "status.json").read_text())
        self.assertEqual((st["status"], st["passed"], st["baseline_failed"], st["recalibration_mismatch"]), ("PASS", True, [], []))
        self.assertEqual(sorted(os.listdir(self.folder / "public")), sorted(T.PUBLIC_OUTPUTS))
        self.assertEqual(sorted(os.listdir(self.folder / "restricted")), sorted(T.restricted_outputs(ORDER)))
        for sub in ("public", "restricted"):
            listed = FR.read_sums((self.folder / sub / "SHA256SUMS").read_bytes(), "sums")
            self.assertEqual({n: sha((self.folder / sub / n).read_bytes()) for n in listed}, listed)
            self.assertEqual(sorted(listed), sorted(n for n in os.listdir(self.folder / sub) if n != "SHA256SUMS"))
        self.assertEqual(st["public_sums_sha256"], sha((self.folder / "public" / "SHA256SUMS").read_bytes()))
        self.assertEqual(st["restricted_sums_sha256"], sha((self.folder / "restricted" / "SHA256SUMS").read_bytes()))
        self.assertIn("RESULT: PASS (run)", self.r["out"])

    def test_inputs_untouched(self):
        self.assertEqual(self.r["after"], self.r["before"])
        rec = json.loads((self.folder / "public" / "run_record.json").read_text())
        self.assertEqual(rec["checks"]["final"], {"parent_checkpoints_unchanged": 36, "parent_restricted_sha256sums_unchanged": True,
                                                  "bnrecal_sha256sums_unchanged": True})
        self.assertEqual(rec["test_read"], T.TEST_READ)
        self.assertIsInstance(rec["process_cpu_seconds"], float)
        self.assertIn("not an independent validation", rec["post_hoc"])
        self.assertEqual(rec["code_sha256"][T.ESTIMATOR_FILE], T.ESTIMATOR_SHA256)
        self.assertEqual(rec["checks"]["before_metrics_against_stored_scores"]["metrics_05_equal"], 72)

    def test_baseline_and_recalibration_tables(self):
        b = read_csv(self.folder / "public" / "baseline.csv")
        self.assertEqual(list(b[0]), list(T.BASELINE_COLUMNS))
        self.assertEqual((len(b), {r["baseline_pass"] for r in b}, {r["test_frames_problem"] for r in b}), (36, {"1"}, {"none"}))
        self.assertTrue(all(float(r["test_prob_max_abs_diff"]) <= 1e-5 and r["test_discordant_05"] == "0" for r in b))
        rc = read_csv(self.folder / "public" / "recalibration_check.csv")
        self.assertEqual(list(rc[0]), list(T.RECAL_COLUMNS))
        self.assertEqual((len(rc), {r["recalibration_pass"] for r in rc}, {r["statistics_identical"] for r in rc}), (36, {"1"}, {"1"}))
        rec = json.loads((self.folder / "public" / "run_record.json").read_text())
        self.assertEqual(rec["statistics_identical_to_pr27"], 36)

    def test_restricted_scores_and_statistics(self):
        w = world()
        for n in ("arm_b_fold1_seed42", "arm_b_prime_fold1_seed44", "arm_b_fold3_seed44"):
            sc = read_csv(self.folder / "restricted" / f"{n}_test_scores.csv")
            self.assertEqual(list(sc[0]), list(BR.SCORE_COLUMNS))
            fold = int(n.split("fold")[1][0])
            keys = w.test_keys(fold)
            self.assertEqual([r["sample_id"] for r in sc], keys + keys)
            self.assertEqual([r["statistics"] for r in sc], ["original"] * len(keys) + ["re-estimated"] * len(keys))
            stored = [float(r["score_attack"]) for r in read_csv(w.parent / "restricted" / f"{n}_test_scores.csv")]
            np.testing.assert_allclose([float(r["score_attack"]) for r in sc[:len(keys)]], stored, atol=1e-5)
            mine = json.loads((self.folder / "restricted" / f"{n}_bn_statistics.json").read_text())
            theirs = json.loads((w.bnrecal / "restricted" / f"{n}_bn_statistics.json").read_text())
            for layer in RBS.BN:
                self.assertEqual(mine["layers"][layer]["installed_moving_mean_float32"], theirs["layers"][layer]["installed_moving_mean_float32"])
        ps = read_csv(self.folder / "restricted" / "per_subject.csv")
        self.assertEqual(list(ps[0]), list(T.SUBJECT_COLUMNS))
        self.assertEqual({(r["scope"], r["statistics"]) for r in ps}, {(a, b) for a in T.SCOPES for b in T.STATISTICS})
        self.assertEqual(len({(r["arm"], r["fold"], r["seed"]) for r in ps}), 36)
        n = "arm_b_fold1_seed42"                                               # prima saturato, dopo no: righe prima e dopo diverse
        sc = read_csv(self.folder / "restricted" / f"{n}_test_scores.csv")
        mine = [r for r in ps if (r["arm"], r["fold"], r["seed"]) == ("arm_b", "1", "42")]
        self.assertEqual(len(mine), 2 * 2 * 4)
        for r in mine:
            part = [x for x in sc if x["statistics"] == r["statistics"] and x["subject_id"] == r["subject_id"]
                    and (r["scope"] == "test" or x["session_id"] == "03")]
            ref = ref_metrics([int(x["label"]) for x in part], [float(x["score_attack"]) for x in part])
            self.assertEqual((int(r["n_bona_fide"]), int(r["n_attack"])), (sum(x["label"] == "0" for x in part), sum(x["label"] == "1" for x in part)))
            for k in T.METRICS:
                self.assertEqual(num(r[k]), ref[k], (r["subject_id"], r["scope"], r["statistics"], k))
        self.assertNotEqual([r["acer_05"] for r in mine if r["statistics"] == "original"],
                            [r["acer_05"] for r in mine if r["statistics"] == "re-estimated"])

    def test_results_against_independent_metrics(self):
        rows = read_csv(self.folder / "public" / "results.csv")
        self.assertEqual(list(rows[0]), list(T.RESULT_COLUMNS))
        self.assertEqual(len(rows), 72)
        worse = set(world().dev_worsened())
        for r in rows:
            n = f"{r['arm']}_fold{r['fold']}_seed{r['seed']}"
            sc = read_csv(self.folder / "restricted" / f"{n}_test_scores.csv")
            if r["scope"] == "session_03":
                sc = [x for x in sc if x["session_id"] == "03"]
            for st, side in (("original", "before"), ("re-estimated", "after")):
                part = [x for x in sc if x["statistics"] == st]
                ref = ref_metrics([int(x["label"]) for x in part], [float(x["score_attack"]) for x in part])
                for k in T.METRICS:
                    got = num(r[f"{k}_{side}"])
                    if ref[k] is None:
                        self.assertIsNone(got)
                    else:
                        self.assertAlmostEqual(got, ref[k], places=12, msg=(n, r["scope"], k))
            for k in T.METRICS:
                b, a, d = num(r[f"{k}_before"]), num(r[f"{k}_after"]), num(r[f"{k}_difference"])
                flag = r[T.WORSE[k][0]]
                if b is None or a is None:
                    self.assertEqual((d, flag), (None, "NA"))
                    continue
                self.assertAlmostEqual(d, a - b, places=15)
                self.assertEqual(flag, "1" if (a > b if T.WORSE[k][1] > 0 else a < b) else "0")
            self.assertEqual(r["dev_worsened"], "1" if n in worse else "0")
            self.assertEqual(r["group"], "ten" if n in RBS.TEN else "apart" if n in RBS.APART else "control")
        by = {(f"{r['arm']}_fold{r['fold']}_seed{r['seed']}", r["scope"]): r for r in rows}
        self.assertLess(float(by[("arm_b_fold1_seed42", "test")]["acer_05_after"]), float(by[("arm_b_fold1_seed42", "test")]["acer_05_before"]))

    def test_summaries_and_reading(self):
        summ = read_csv(self.folder / "public" / "summary.csv")
        self.assertEqual(list(summ[0]), list(T.SUMMARY_COLUMNS))
        self.assertEqual(len(summ), 56)
        rows = read_csv(self.folder / "public" / "results.csv")
        for s in summ:
            mem = [r for r in rows if r["scope"] == s["scope"] and (s["summary"] == "all" or r["group" if s["summary_kind"] == "group" else "arm"] == s["summary"])]
            self.assertEqual(int(s["n_models"]), len(mem))
            d = [num(r[f"{s['metric']}_difference"]) for r in mem]
            if None in d:
                self.assertEqual(s["mean_difference"], "NA")
            else:
                self.assertAlmostEqual(float(s["mean_difference"]), float(np.mean(d)), places=12)
            if s["low"] != "NA":
                lo, hi = float(s["low"]), float(s["high"])
                self.assertEqual(s["reading"], "below" if hi < 0 else "above" if lo > 0 else "contains")
        apart = [s for s in summ if s["summary"] == "apart"]
        self.assertTrue(all(s["low"] == "NA" and s["reading"] == "not_available" and s["n_subjects"] == "4" for s in apart))
        w = world()
        for scope in T.SCOPES:
            keys = [k for f in range(1, CFR.K + 1) for k in w.test_keys(f) if scope == "test" or w.official[k]["session"] == "03"]
            want = (len({w.official[k]["subject"] for k in keys}), len({w.official[k]["subject"] for k in keys if w.official[k]["label"] == "0"}),
                    len({w.official[k]["subject"] for k in keys if w.official[k]["label"] == "1"}))
            s = next(s for s in summ if s["summary"] == "all" and s["scope"] == scope)
            self.assertEqual((int(s["n_subjects"]), int(s["n_subjects_bona_fide"]), int(s["n_subjects_attack"])), want)
        self.assertTrue(all(int(s["valid_resamples"]) <= 1000 for s in summ))
        doc = json.loads((self.folder / "public" / "aggregate.json").read_text())
        p = doc["reading"]["primary"]
        ps = next(s for s in summ if s["label"] == "primary")
        self.assertEqual(p["case"], ps["reading"])
        self.assertTrue(p["statement"].startswith({"below": "BELOW", "contains": "CONTAINS", "above": "ABOVE", "not_available": "NOREADING"}[p["case"]]))
        ten = [r for r in rows if r["group"] == "ten" and r["scope"] == "test"]
        self.assertEqual(p["count"]["n"], len(ten))
        self.assertEqual(p["count"]["higher"], sum(1 for r in ten if float(r["acer_05_difference"]) > 0))
        md = (self.folder / "public" / "aggregate.md").read_text()
        self.assertIn(f"**{p['statement']}**", md)
        self.assertIn(p["count_statement"], md)
        self.assertIn("post hoc, not an independent validation", md)
        self.assertIn("## All models, before -> after", md)
        self.assertIn("## Paired summaries (mean over the models of after - before; subject bootstrap, held-out subjects pooled over the folds)", md)
        for stale in ("within each fold", "stratif"):
            self.assertNotIn(stale, md)
        n_flags = sum(1 for r in rows for f in T.FLAGS if r[f] == "1")
        self.assertEqual(len(doc["reading"]["worsening"]["cases"]), n_flags)
        self.assertEqual(md.count(" -> ") >= n_flags, True)

    def test_public_has_no_identifiers(self):
        for f in T.PUBLIC_OUTPUTS:
            if f == T.FIGURE:
                continue
            t = (self.folder / "public" / f).read_text()
            self.assertNotRegex(t, r"\b(real|attack)/\d{4}_")
            for s in CFR.SUBJECTS:
                self.assertNotRegex(t, rf"(?<![\w.]){s}(?![\w.])")
            for h in {r["sha256"] for r in world().official.values()}:
                self.assertNotIn(h, t)

    def test_figure_deterministic_and_regenerated(self):
        png = (self.folder / "public" / T.FIGURE).read_bytes()
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        out = world().tmp / "fig" / "again.png"
        out.parent.mkdir(exist_ok=True)
        rc, o, e = run_main(world(), ["figure", "--run-folder", str(self.folder), "--output", str(out)])
        self.assertEqual(rc, 0, e)
        self.assertEqual(out.read_bytes(), png)
        rc, o, e = run_main(world(), ["figure", "--run-folder", str(self.folder), "--output", str(out)])
        self.assertEqual(rc, 1)
        self.assertIn("already exists", e)
        res = (self.folder / "public" / "results.csv").read_text()
        summ = (self.folder / "public" / "summary.csv").read_text()
        self.assertEqual(T.figure_png(res, summ), png)
        rows = list(csv.DictReader(io.StringIO(res)))
        i = next(i for i, r in enumerate(rows) if r["acer_05_difference"] != "NA")
        rows[i]["acer_05_difference"] = repr(float(rows[i]["acer_05_difference"]) + 0.25)     # solo la colonna delle differenze
        changed = CFR.csv_text(T.RESULT_COLUMNS, [[r[c] for c in T.RESULT_COLUMNS] for r in rows])
        self.assertNotEqual(T.figure_png(changed, summ), png)

    def test_existing_run_folder_refused(self):
        w = world()
        rc, out, err = run_main(w, w.bt_argv("run"))
        self.assertEqual(rc, 1)
        self.assertIn("already exists", err)
        self.assertEqual(json.loads((self.folder / "status.json").read_text())["status"], "PASS")


def spies():
    """Chiamate registrate: ristime, valutazioni del test con le statistiche ristimate, fase 3."""
    log = {"recalibrate": 0, "test_after": 0, "comparison": 0}
    real_r, real_e, real_c = BR.recalibrate, BR.evaluate, T.score_phase

    def r(*a, **k):
        log["recalibrate"] += 1
        return real_r(*a, **k)

    def e(model, part, batch, what, layers=()):
        log["test_after"] += what.endswith("test after")
        return real_e(model, part, batch, what, layers)

    def c(*a, **k):
        log["comparison"] += 1
        return real_c(*a, **k)
    return log, {"rb": {"recalibrate": r, "evaluate": e}, "t": {"score_phase": c}}


class TestRunStops(unittest.TestCase):
    def setUp(self):
        self.w = world()
        self.out = self.w.tmp / "stops"

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)
        self.w.write_bt_config(self.w.bt_doc())

    def go(self, *extra, rb=None, t=None, sr=None):
        rc, out, err = run_main(self.w, self.w.bt_argv("run", *extra, out=self.out), rb=rb, t=t, sr=sr)
        return rc, out, err, run_folder(self.out)

    def test_baseline_failed_on_the_test(self):
        n = "arm_a_star_fold2_seed42"
        log, inj = spies()

        def shift(rows):
            v = float(rows[3]["score_attack"])
            rows[3]["score_attack"] = repr(v + 2e-5 if v < 0.5 else v - 2e-5)
        with altered_parent(self.w, test_over={n: shift}):
            rc, out, err, run = self.go(**inj)
        self.assertEqual(rc, 3, err[-2000:])
        self.assertEqual(log, {"recalibrate": 0, "test_after": 0, "comparison": 0})
        st = json.loads((run / "status.json").read_text())
        self.assertEqual((st["status"], st["baseline_failed"], st["completed"]), ("BASELINE_FAILED", [n], []))
        self.assertEqual(sorted(os.listdir(run / "public")), ["SHA256SUMS", "baseline.csv", "run_record.json"])
        self.assertFalse((run / "restricted").exists())
        rows = {f"{r['arm']}_fold{r['fold']}_seed{r['seed']}": r for r in read_csv(run / "public" / "baseline.csv")}
        self.assertEqual(len(rows), 36)
        self.assertEqual((rows[n]["test_prob_pass"], rows[n]["dev_pass"], rows[n]["baseline_pass"], rows[n]["test_discordant_05"]),
                         ("0", "1", "0", "0"))
        self.assertGreater(float(rows[n]["test_prob_max_abs_diff"]), 1e-5)
        self.assertEqual(sum(r["baseline_pass"] == "1" for r in rows.values()), 35)
        self.assertIn("RESULT: BASELINE_FAILED (run)", out)
        self.assertNotIn("acer_05_before", (run / "public" / "run_record.json").read_text())

    def test_baseline_failed_on_misaligned_frames(self):
        n = "arm_b_prime_fold3_seed42"
        with altered_parent(self.w, test_over={n: lambda rows: rows.__setitem__(slice(0, 2), [rows[1], rows[0]])}):
            rc, out, err, run = self.go()
        self.assertEqual(rc, 3, err[-2000:])
        r = [x for x in read_csv(run / "public" / "baseline.csv") if x["baseline_pass"] == "0"]
        self.assertEqual([(x["arm"], x["fold"], x["seed"], x["test_frames_pass"], x["test_prob_max_abs_diff"]) for x in r],
                         [("arm_b_prime", "3", "42", "0", "NA")])
        self.assertIn("sample ids or their order differ", r[0]["test_frames_problem"])

    def test_baseline_failed_on_the_dev(self):
        n = "arm_b_fold2_seed43"

        def shifted(path, wsha):
            s = FR.tf_scorer(path, wsha)
            def f(X):
                p = s(X)
                q = p[:, 1] + 2e-5 if n in str(path) else p[:, 1]
                return np.stack([1 - q, q], 1)
            return f
        log, inj = spies()
        rc, out, err, run = self.go(t={"MAKE_SCORER": shifted, **inj["t"]}, rb=inj["rb"])
        self.assertEqual(rc, 3, err[-2000:])
        self.assertEqual(log["recalibrate"] + log["comparison"], 0)
        r = [x for x in read_csv(run / "public" / "baseline.csv") if x["baseline_pass"] == "0"]
        self.assertEqual([(x["arm"], x["fold"], x["seed"], x["dev_pass"], x["test_pass"]) for x in r], [("arm_b", "2", "43", "0", "1")])

    def test_recalibration_mismatch_on_the_statistics(self):
        n = "arm_b_prime_fold2_seed44"

        def bump(d):
            p = d / "restricted" / f"{n}_bn_statistics.json"
            js = json.loads(p.read_text())
            js["layers"]["bn2_2"]["installed_moving_mean_float32"][1] += 1e-3
            p.write_text(json.dumps(js))
        log, inj = spies()
        with altered_bnrecal(self.w, bump):
            rc, out, err, run = self.go(**inj)
        self.assertEqual(rc, 4, err[-2000:])
        self.assertEqual((log["recalibrate"], log["test_after"], log["comparison"]), (36, 0, 0))
        st = json.loads((run / "status.json").read_text())
        self.assertEqual((st["status"], st["recalibration_mismatch"], st["completed"]), ("RECALIBRATION_MISMATCH", [n], []))
        self.assertEqual(sorted(os.listdir(run / "public")), ["SHA256SUMS", "baseline.csv", "recalibration_check.csv", "run_record.json"])
        self.assertFalse((run / "restricted").exists())
        rows = {f"{r['arm']}_fold{r['fold']}_seed{r['seed']}": r for r in read_csv(run / "public" / "recalibration_check.csv")}
        self.assertEqual((rows[n]["statistics_pass"], rows[n]["statistics_identical"], rows[n]["dev_pass"], rows[n]["recalibration_pass"]),
                         ("0", "0", "1", "0"))
        self.assertEqual(int(rows[n]["statistics_outside_tolerance"]), 1)
        self.assertEqual(sum(r["recalibration_pass"] == "1" for r in rows.values()), 35)
        self.assertIn("RESULT: RECALIBRATION_MISMATCH (run)", out)

    def test_recalibration_mismatch_on_the_dev(self):
        n = "arm_a_star_fold4_seed43"

        def bump(d):
            p = d / "restricted" / f"{n}_scores.csv"
            rows = read_csv(p)
            i = next(i for i, r in enumerate(rows) if r["role"] == "dev" and r["statistics"] == "re-estimated")
            v = float(rows[i]["score_attack"])
            rows[i]["score_attack"] = repr(v + 3e-5 if v < 0.5 else v - 3e-5)
            write_rows(p, BR.SCORE_COLUMNS, rows)
        with altered_bnrecal(self.w, bump):
            rc, out, err, run = self.go()
        self.assertEqual(rc, 4, err[-2000:])
        r = [x for x in read_csv(run / "public" / "recalibration_check.csv") if x["recalibration_pass"] == "0"]
        self.assertEqual([(x["arm"], x["fold"], x["seed"], x["statistics_pass"], x["dev_pass"]) for x in r], [("arm_a_star", "4", "43", "1", "0")])

    def test_consistency_failure_is_fail(self):
        real = BR.recalibrate

        def wrong(copy, part, bn, batch, budget=None, name=""):
            st = real(copy, part, bn, batch, budget, name)
            s = st["bn2_1"]
            s["mean32"] = s["mean32"] + np.float32(0.5)
            copy.get_layer("bn2_1").moving_mean.assign(s["mean32"])
            return st
        rc, out, err, run = self.go(rb={"recalibrate": wrong})
        self.assertEqual(rc, 1)
        self.assertIn("consistency check failed", err)
        st = json.loads((run / "status.json").read_text())
        self.assertEqual((st["status"], st["outputs"]), ("FAIL", "removed"))
        self.assertEqual([p for p in run.rglob("*") if p.is_file() and p.name not in ("status.json", "run.log")], [])

    def test_rebuilt_copy_checked_in_phase_3(self):
        """Il controllo della copia ricostruita fallisce all'ultimo modello: nessuna metrica del test nel log né nei file."""
        real = T.install
        calls = []

        def touch(copy, installed):
            out = real(copy, installed)
            calls.append(1)
            if len(calls) == 36:
                g = copy.get_layer("bn1_2").gamma
                g.assign(g.numpy() + np.float32(1e-6))
            return out
        rc, out, err, run = self.go(t={"install": touch})
        self.assertEqual(rc, 1)
        self.assertIn(f"{ORDER[-1]}: rebuilt copy: trained tensors changed: bn1_2/gamma", err)
        log = (run / "run.log").read_text()
        self.assertEqual(log.count("scored "), 35)
        self.assertNotIn("test ACER", log)
        self.assertNotIn("test ACER", out)
        self.assertEqual(json.loads((run / "status.json").read_text())["status"], "FAIL")
        self.assertEqual([p for p in run.rglob("*") if p.is_file() and p.name not in ("status.json", "run.log")], [])

    def test_trained_tensor_change_in_the_recalibration_is_fail(self):
        real = BR.recalibrate

        def touch(copy, part, bn, batch, budget=None, name=""):
            st = real(copy, part, bn, batch, budget, name)
            g = copy.get_layer("bn1_2").beta
            g.assign(g.numpy() + np.float32(1e-6))
            return st
        log, inj = spies()
        rc, out, err, run = self.go(rb={"recalibrate": touch}, t=inj["t"])
        self.assertEqual(rc, 1)
        self.assertIn("trained tensors changed: bn1_2/beta", err)
        self.assertEqual(log["comparison"], 0)

    def test_inputs_changed_during_the_run(self):
        first = self.w.parent / "restricted" / f"{ORDER[0]}_LivenessNet.keras"
        sums = self.w.bnrecal / "restricted" / "SHA256SUMS"
        real = T.install
        for target, needle in ((first, f"parent checkpoints changed during the run: ['{ORDER[0]}']"),
                               (sums, "the BNRECAL restricted SHA256SUMS changed during the run")):
            saved = target.read_bytes()

            calls = []

            def touch(copy, installed, target=target, calls=calls):
                calls.append(1)
                if len(calls) == 36:
                    with open(target, "ab") as fh:
                        fh.write(b"\n")
                return real(copy, installed)
            try:
                rc, out, err, run = self.go(t={"install": touch})
            finally:
                target.write_bytes(saved)
            self.assertEqual(rc, 1, err[-2000:])
            self.assertIn(needle, err)
            self.assertNotIn("test ACER", (run / "run.log").read_text())             # riverifica prima di ogni metrica
            self.assertEqual(json.loads((run / "status.json").read_text())["status"], "FAIL")
            shutil.rmtree(self.out)

    def test_public_scan_refuses_identifiers(self):
        doc = self.w.bt_doc()
        doc["reading_rule"]["count"] = f"COUNT {{lower}}/{{equal}}/{{higher}} of {{n}} for subject {CFR.SUBJECTS[3]}"
        self.w.write_bt_config(doc)
        rc, out, err, run = self.go()
        self.assertEqual(rc, 1)
        self.assertIn(f"refused: subject id {CFR.SUBJECTS[3]}", err)
        self.assertFalse((run / "public").exists() and any((run / "public").iterdir()))
        self.assertFalse((run / "restricted").exists() and any((run / "restricted").iterdir()))

    def test_dirty_tree_authorisation_note_date(self):
        rc, out, err, run = self.go(t={"REPO_STATE": lambda root: {"commit": "abcdef0", "dirty": True}})
        self.assertEqual(rc, 1)
        self.assertIn("the git work tree is not clean", err)
        self.assertFalse(self.out.exists())
        rc, out, err = run_main(self.w, [x for x in self.w.bt_argv("run", out=self.out) if x != "--i-am-authorized"])
        self.assertEqual(rc, 1)
        self.assertIn("pass --i-am-authorized", err)
        doc = self.w.bt_doc()
        doc["note_sha256"] = None
        self.w.write_bt_config(doc)
        rc, out, err, run = self.go()
        self.assertEqual(rc, 1)
        self.assertIn("note_sha256 is null", err)
        self.assertFalse(self.out.exists())
        self.w.write_bt_config(self.w.bt_doc())
        rc, out, err, run = self.go("--date", "2026111")
        self.assertEqual(rc, 1)
        self.assertIn("--date must be YYYYMMDD", err)
        rc, out, err = run_main(self.w, self.w.bt_argv("run", out=self.w.bnrecal / "public"))
        self.assertEqual(rc, 1)
        self.assertIn("refusing the run folder", err)

    def test_cap_in_the_baseline_and_in_the_comparison(self):
        doc = self.w.bt_doc()
        doc["budget"]["wall_time_cap_minutes"] = 1
        doc["budget"]["estimate_minutes"] = 0.1
        doc["budget"]["upper_estimate_minutes"] = 0.2
        self.w.write_bt_config(doc)
        t = [0.0]
        rc, out, err, run = self.go(sr={"CLOCK": lambda: t.__setitem__(0, t[0] + 3.0) or t[0]})
        self.assertEqual(rc, 2, err[-2000:])
        st = json.loads((run / "status.json").read_text())
        self.assertEqual((st["status"], st["stopped"]["phase"], st["completed"]), ("INCOMPLETE", "baseline", []))
        self.assertEqual(sorted(os.listdir(run / "public")), ["SHA256SUMS", "baseline.csv", "run_record.json"])
        shutil.rmtree(self.out)
        self.w.write_bt_config(self.w.bt_doc())

        scored = []
        real_e = BR.evaluate

        def counting(model, part, batch, what, layers=()):
            scored.append(what.endswith("test after"))
            return real_e(model, part, batch, what, layers)

        def clock():
            return 1e6 if sum(scored) >= 2 else 0.0
        rc, out, err, run = self.go(sr={"CLOCK": clock}, rb={"evaluate": counting})
        self.assertEqual(rc, 2, err[-2000:])
        self.assert_no_test_metric(run, out, "cap", 2)

    def assert_no_test_metric(self, run: Path, out: str, cause: str, n_scored: int):
        """INCOMPLETE nella fase 3a: pubblicati solo i controlli completati; nessuna metrica del test nei file, nello
        stdout o nel run.log; nessun file per frame del test con statistiche ristimate."""
        st = json.loads((run / "status.json").read_text())
        self.assertEqual((st["status"], st["stopped"]["phase"], st["stopped"]["cause"], st["completed"]), ("INCOMPLETE", "comparison", cause, []))
        self.assertEqual(sorted(os.listdir(run / "public")), ["SHA256SUMS", "baseline.csv", "recalibration_check.csv", "run_record.json"])
        self.assertFalse((run / "restricted").exists() and any((run / "restricted").iterdir()))
        log = (run / "run.log").read_text()
        self.assertEqual(log.count("   scored "), n_scored)
        for text in (log, out, (run / "public" / "run_record.json").read_text()):
            self.assertNotIn("test ACER", text)
            self.assertNotIn("acer_05_before", text)
            self.assertNotIn("primary:", text)

    def test_stop_in_phase_3a_at_the_first_and_the_last_model(self):
        for target, n_scored in ((ORDER[0], 0), (ORDER[-1], 35)):
            real = SR.Budget.check

            def check(self_, step, target=target):
                if step == f"comparison scoring {target}":
                    self_.stop_requested = "SIGUSR1"
                return real(self_, step)
            with injected(sr={"Budget": type("B", (SR.Budget,), {"check": check})}):
                rc, out, err, run = self.go()
            self.assertEqual(rc, 2, err[-2000:])
            self.assert_no_test_metric(run, out, "SIGUSR1", n_scored)
            shutil.rmtree(self.out)

    def test_voluntary_stop_in_the_recalibration(self):
        real = SR.Budget.check

        def check(self_, step):
            if step.startswith("recalibration arm_a_star_fold1_seed42"):
                self_.stop_requested = "SIGUSR1"
            return real(self_, step)
        with injected(sr={"Budget": type("B", (SR.Budget,), {"check": check})}):
            rc, out, err, run = self.go()
        self.assertEqual(rc, 2, err[-2000:])
        st = json.loads((run / "status.json").read_text())
        self.assertEqual((st["status"], st["stopped"]["cause"], st["stopped"]["phase"], st["completed"]), ("INCOMPLETE", "SIGUSR1", "recalibration", []))
        self.assertEqual(sorted(os.listdir(run / "public")), ["SHA256SUMS", "baseline.csv", "recalibration_check.csv", "run_record.json"])
        self.assertEqual(len(read_csv(run / "public" / "recalibration_check.csv")), 1)


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
