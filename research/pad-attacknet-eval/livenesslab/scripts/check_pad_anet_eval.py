"""
Test di scripts/pad_anet_eval.py (C5, la valutazione sul test NUAA): configurazione, file fissati, i 48 riferimenti di C2
e la riproduzione esatta di C2 (il cancello prima del test nuovo), i sei run (ultimo tentativo, PASS, modelli non
valutabili), il flusso di evaluate (riproduzione del dev, test, punteggi non finiti, analisi primaria e descrittive con
il guardiano, opposto esatto di C2, uscite) con un valutatore finto, senza TensorFlow; agganci; copie dichiarate;
pacchetto (righe attese del README, log, manifest).

Mondo sintetico in cartelle temporanee: il mondo della suite della PR B (check_pad_anet_run.World: preregistrazione C2,
configurazione della PR A, prova C5, piano, configurazione dei run) con una valutazione C2 finta coerente (48 file di
punteggi di test sui frame del test del mondo, results.json e units.csv calcolati con le funzioni congelate di C2), i sei
run scritti da pad_anet_run.py con i suoi finti e la configurazione di questo comando (delivery/aneteval/make_config.doc
sui file del mondo, i sei run registrati dalle cartelle). Nessuna scrittura nel repository.

  .venv/bin/python -I -B scripts/check_pad_anet_eval.py [--verbose] [-f] [--regenerating-logs]
"""
import ast
import contextlib
import csv
import difflib
import importlib.util
import inspect
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_pad_anet_comparison as CAC  # noqa: E402
import check_pad_anet_run as CNR  # noqa: E402
import check_pad_arch_run as CPR  # noqa: E402
import nuaa_folds_run as FR  # noqa: E402
import nuaa_session_ablation_run as SR  # noqa: E402
import pad_anet_comparison as C  # noqa: E402
import pad_anet_definitions as DEF  # noqa: E402
import pad_anet_eval as V  # noqa: E402
import pad_anet_run as NR  # noqa: E402
import pad_arch_comparison as PAC  # noqa: E402
import pad_arch_definitions as D  # noqa: E402
import pad_arch_eval as E  # noqa: E402
import pad_arch_run as R  # noqa: E402

MAKE_CONFIG = ROOT / "delivery" / "aneteval" / "make_config.py"
COPY_RECORDS = ROOT / "delivery" / "aneteval" / "copy_run_records.py"
LAUNCH = "delivery/aneteval/launch.zsh"
MISSING_INPUT = {
    "delivered configuration": "delivery/aneteval/pad_anet_eval.json or the files it pins are not present",
    "macOS tools": "zsh, caffeinate, pmset, sw_vers or git is not present (macOS): the test of delivery/aneteval/launch.zsh is skipped",
    "logs being regenerated": ("the suite was started with --regenerating-logs: the check that every log of test_logs/ cites the current "
                               "SHA-256 of its files is not made (first of the two passes that regenerate the logs)"),
}
REGENERATING_LOGS = "--regenerating-logs" in sys.argv
STATE = CNR.STATE
GUARDED = ("results/aneteval", "delivery/aneteval", "delivery/pad-attacknet-eval")
QUALITY = {"LivenessNet": 0.20, "ResNet-18": 0.25, "AttackNetV1": 0.15, "AttackNetV2_1": 0.18, "AttackNetV2_2": 0.22}
C2_DEV_EER = 0.45

if (os.path.realpath(tempfile.gettempdir()) + os.sep).startswith(os.path.realpath(ROOT) + os.sep):
    raise SystemExit("TMPDIR is inside the repository root: set it outside")


def skip_missing(test, what):
    test.skipTest(f"missing input: {MISSING_INPUT[what]}")


def sha(b: bytes) -> str:
    return CAC.sha(b)


def load_make_config():
    spec = importlib.util.spec_from_file_location("aneteval_make_config", MAKE_CONFIG)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MC = load_make_config()


def load_copy_records():
    spec = importlib.util.spec_from_file_location("aneteval_copy_run_records", COPY_RECORDS)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CR = load_copy_records()


@contextlib.contextmanager
def injected(v=None, **kw):
    saved = [(V, k, getattr(V, k)) for k in (v or {})]
    try:
        for k, x in (v or {}).items():
            setattr(V, k, x)
        with CNR.injected(**kw):
            yield
    finally:
        for m, k, x in saved:
            setattr(m, k, x)


def scores_for(net: str, fold: int, seed: int, patience: int, state: str, y) -> np.ndarray:
    """Punteggi d'attacco finti e deterministici di una rete su un test (separazione secondo la rete)."""
    rng = np.random.default_rng(seed * 1000 + fold * 10 + patience + (5 if state == "re-estimated" else 0) + len(net))
    y = np.asarray(y, np.float64)
    q = QUALITY[net] + (0.05 if state == "re-estimated" else 0.0)
    return np.clip(0.5 + (2 * y - 1) * q + rng.normal(0, 0.15, len(y)), 0.001, 0.999)


def test_lists(w) -> dict:
    """Le righe del test di ogni fold del mondo, come le carica pad_arch_eval.load_test_lists (in ordine di chiave)."""
    with injected(**CNR.base_hooks(w)):
        _, _, cfg_pr, pcfg, _ = C.load_config(w.c5_path, w.base)
        inp = PAC.load_inputs(cfg_pr, pcfg, w.base)
        ctx = V.Ctx()
        ctx.inp, ctx.pcfg, ctx.base, ctx.folds, ctx.test_per_fold = inp, pcfg, w.base, list(DEF.FOLDS), pcfg["schema_expected"]["test_per_fold"]
        lists = E.load_test_lists(ctx)
    return {f: sorted(rows, key=lambda r: r["key"]) for f, rows in lists.items()}


class World(CNR.World):
    """Il mondo della PR B con la valutazione C2 finta coerente, i sei run e la configurazione della valutazione."""

    def build(self):
        super().build()
        self.write_c2_eval()
        self.make_runs()
        self.launch_logs = self.tmp / "launchlogs"
        self.records = self.base / "pkgruns"
        self.write_records()
        self.eval_out = self.tmp / "aneteval"
        self.eval_cfg = self.base / "aneteval.json"
        (self.base / "PROTOCOL_amended.md").write_text("# Amended protocol (synthetic)\n", encoding="utf-8")
        self.write_eval_cfg()

    def write_c2_eval(self):
        """I 48 file di C2 sui frame del test del mondo, il SHA256SUMS riservato, results.json (modelli, analisi primaria e
        8 condizioni calcolate con le funzioni congelate di C2) e units.csv; poi la configurazione della PR A e quella dei
        run rifissate."""
        lists = test_lists(self)
        rdir = self.base / "eval" / "restricted"
        shutil.rmtree(rdir, ignore_errors=True)
        rdir.mkdir(parents=True)
        models, files = [], {}
        cells = [(p, n, f, s) for p in DEF.PATIENCES for n in DEF.REUSED for f in DEF.FOLDS for s in DEF.PLAN_SEEDS]
        for (p, n, f, s) in cells:
            run = "nuaa_patience_4" if p == 4 else "nuaa_patience_12"
            wsha = sha(f"{run}{n}{f}{s}".encode())
            models.append({"run": run, "position": 0, "network": n, "fold": f, "seed": s, "weights_sha256": wsha,
                           "dev_eer_threshold": {st: C2_DEV_EER for st in D.BN_STATES}})
            rows = lists[f]
            y = np.array([int(r["label"]) for r in rows])
            lines = []
            for st in D.BN_STATES:
                pr = scores_for(n, f, s, p, st, y)
                for i, r in enumerate(rows):
                    lines.append([run, n, str(f), str(s), str(p), st, "test", r["key"], r["subject"], r["session"], r["label"], repr(float(pr[i])),
                                  repr(float(np.log(1 - pr[i]))), repr(float(np.log(pr[i]))), "1" if pr[i] > 0.5 else "0", repr(C2_DEV_EER),
                                  "1" if pr[i] > C2_DEV_EER else "0", r["sha256"], wsha])
            fn = f"{V.C2_TAG[run]}_{PAC.training_name(n, f, s)}{V.TEST_SUFFIX}"
            files[fn] = FR.csv_text(E.TEST_SCORE_COLUMNS, lines)
        for fn, t in files.items():
            (rdir / fn).write_text(t, encoding="utf-8", newline="")
        sums = "".join(f"{sha(t.encode())}  {fn}\n" for fn, t in sorted(files.items()))
        (rdir / "SHA256SUMS").write_text(sums, encoding="utf-8")
        self.c2_sums_sha = sha(sums.encode())
        # i numeri di C2 calcolati dai file con le funzioni congelate (come la valutazione C2)
        pins = V.Pins()
        pins.c2_cells = [{"network": n, "fold": f, "seed": s, "patience": p, "checkpoint_sha256": sha(f"{'nuaa_patience_4' if p == 4 else 'nuaa_patience_12'}{n}{f}{s}".encode()),
                          "test_scores_file": f"{'p4' if p == 4 else 'p12'}_{PAC.training_name(n, f, s)}{V.TEST_SUFFIX}"} for (p, n, f, s) in cells]
        cfg = {"c2_evaluation": {"restricted_folder": "eval/restricted", "restricted_sha256sums_sha256": self.c2_sums_sha}}
        c2 = V.load_c2_scores(cfg, pins, self.base)
        conds = V.c2_conditions(c2)
        pa = E.jsonable(D.primary_analysis({c: conds[c] for c in E.PRIMARY}))
        results = {c: D.compare_condition(conds[c]) for c in D.CONDITIONS}
        primary = {k: pa[k] for k in ("conditions", "definition", "gap_ordinary", "gap_re_estimated", "contrast", "per_seed", "rows")}
        cond_list = [{"condition": list(c), "label": "x", **E.jsonable(E.public_condition(results[c]))} for c in D.CONDITIONS]
        units = FR.csv_text(E.UNIT_COLUMNS, [[E.cell(x[k]) for k in E.UNIT_COLUMNS] for c in D.CONDITIONS for x in E.unit_rows(c, results[c], conds)])
        (self.base / "eval" / "public").mkdir(parents=True, exist_ok=True)
        (self.base / "eval" / "public" / "units.csv").write_text(units, encoding="utf-8", newline="")
        self.write_c5(results_over={"models": models, "primary_analysis": primary, "conditions": cond_list},
                      status_over={"restricted_sha256sums_sha256": self.c2_sums_sha})
        self.write_anet()

    def make_runs(self, history_over=None):
        """I sei run scritti da pad_anet_run.py con i finti della sua suite (stessa radice, nomi del giorno)."""
        shutil.rmtree(self.anet_out, ignore_errors=True)
        for r in NR.RUNS:
            log = []
            rc, o, e = CNR.flow(self, run=r, argv=self.anet_argv("run", run=r), trainer=CNR.fake_trainer(log, history_over=history_over))
            if rc != 0:
                raise RuntimeError(f"the runner did not produce {r}: {(o + e)[-2000:]}")

    def write_records(self):
        """I file delle condizioni finti dello script di lancio e le copie dei record dei run (copy_run_records.copy_all)."""
        shutil.rmtree(self.launch_logs, ignore_errors=True)
        shutil.rmtree(self.records, ignore_errors=True)
        self.launch_logs.mkdir(parents=True)
        est = self.anet_out / self.est_anet.name                 # come in results/anetrun: il check-estimator nella radice dei run
        if not est.exists():
            shutil.copytree(self.est_anet, est)
        for f in V.record_folders(self.anet_out):
            lab = CR.label_of(f.name)
            (self.launch_logs / f"{lab}-conditions.txt").write_text(f"Conditions of {lab}, written by delivery/anetrun/launch.zsh (not by the runner).\n"
                                                                    "== nice of the launching shell\n0\n", encoding="utf-8")
        CR.copy_all(self.anet_out, self.launch_logs, self.records)

    def eval_doc(self, runs=None) -> dict:
        return MC.doc(data_sha=lambda p: sha((self.base / p).read_bytes()), code_sha=lambda p: sha((ROOT / p).read_bytes()),
                      runs=runs if runs is not None else MC.runs_record(self.anet_out), RECORDS="pkgruns", COMPARISON="anetcmp.json", RUNCFG="anetrun.json",
                      PROTOCOL="PROTOCOL_amended.md", C2_UNITS="eval/public/units.csv", C2_RESTRICTED="eval/restricted",
                      C2_RESTRICTED_SUMS=self.c2_sums_sha)

    def write_eval_cfg(self, d=None) -> None:
        self.eval_cfg.write_text(json.dumps(d or self.eval_doc(), indent=1) + "\n", encoding="utf-8")

    def eval_argv(self, cmd="evaluate", attempt=1, unofficial=True, out=None) -> list:
        a = [cmd, "--config", str(self.eval_cfg), "--input-root", str(self.base), "--runs-root", str(self.anet_out)]
        if cmd == "evaluate":
            a += ["--out-root", str(out or self.eval_out), "--i-am-authorized"] + (["--unofficial-out-root"] if unofficial else [])
            if attempt is not None:
                a += ["--attempt", str(attempt)]
        return a


WORLD = None


def world() -> World:
    global WORLD
    if WORLD is None:
        WORLD = World()
    return WORLD


def fake_score(dev_over=None, test_over=None, calls=None):
    """Il valutatore finto con la firma di score_model: sul dev i punteggi salvati dal run (riproduzione esatta, salvo
    dev_over), sul test i punteggi finti (salvo test_over)."""
    def score(ctx, rec, parts):
        out = {s: {} for s in D.BN_STATES}
        for role, part in parts.items():
            if calls is not None:
                calls.append((role, rec["key"], len(ctx.inp.folds)))
            for s in D.BN_STATES:
                if role == "dev":
                    p = np.array(rec["dev"][s]["s"], np.float64)
                    if dev_over:
                        p = dev_over(rec, s, p)
                else:
                    p = scores_for(rec["network"], rec["fold"], rec["seed"], rec["patience"], s, part["y"])
                    if test_over:
                        p = test_over(rec, s, p)
                out[s][role] = {"p": p, "z": CPR.lz(np.clip(p, 1e-6, 1 - 1e-6)), "softmax_gap": 0.0}
        return out
    return score


def eval_hooks(w, score=None, **v) -> dict:
    h = CNR.base_hooks(w)
    tfm = CPR.FakeTF()
    h["r"]["TF_MODULE"] = lambda: tfm
    h["v"] = {"REPO_STATE": lambda root: dict(STATE), "SCORE_MODEL": score or fake_score(), **v}
    return h


def run_eval(argv, hooks) -> tuple:
    out, err = io.StringIO(), io.StringIO()
    with injected(**hooks), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = V.main(argv)
    return rc, out.getvalue(), err.getvalue()


def eval_folder(w, attempt: int = 1) -> Path:
    """La cartella del tentativo, cercata nella radice d'uscita del test e non ricalcolata dalla data (un test a cavallo
    della mezzanotte non fallisce)."""
    found = sorted(w.eval_out.glob(f"[0-9]*-ANETEVAL-{STATE['commit']}" + ("" if attempt == 1 else f"-r{attempt}")))
    if len(found) != 1:
        raise AssertionError(f"evaluation folders of attempt {attempt}: {[f.name for f in found]}")
    return found[0]


# ============================================================================ A configurazione

class TestA0RecordFiles(unittest.TestCase):
    """Prima del mondo sintetico (il nome la fa eseguire per prima): i file pubblicabili di una cartella di comando dei run
    sono status.json, run.log e public/, mai restricted/."""

    def test_record_files_never_list_restricted(self):
        tmp = Path(tempfile.mkdtemp(prefix="aneteval-records-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        run = tmp / "20261122-ANETRUN-v1-p4-abcdef0"
        for rel in ("status.json", "run.log", "public/models.csv", "public/SHA256SUMS", "restricted/x.keras", "restricted/SHA256SUMS"):
            (run / rel).parent.mkdir(parents=True, exist_ok=True)
            (run / rel).write_text("x", encoding="utf-8")
        self.assertEqual(V.record_files(run), ["status.json", "run.log", "public/SHA256SUMS", "public/models.csv"])


class TestAConfig(unittest.TestCase):
    def setUp(self):
        self.w = world()
        self.addCleanup(self.w.write_eval_cfg)

    def refused(self, change, needle):
        d = self.w.eval_doc()
        change(d)
        self.w.write_eval_cfg(d)
        with self.assertRaisesRegex(FR.Failure, needle):
            V.load_config(self.w.eval_cfg)

    def test_world_configuration_loads_with_the_six_runs_recorded(self):
        cfg, _ = V.load_config(self.w.eval_cfg)
        self.assertEqual(sorted(cfg["runs"]), sorted(NR.RUNS))

    def test_runs_pending_accepted_by_the_configuration_refused_by_evaluate(self):
        self.w.write_eval_cfg(self.w.eval_doc(runs=V.RUNS_PENDING))
        cfg, _ = V.load_config(self.w.eval_cfg)
        self.assertEqual(cfg["runs"], V.RUNS_PENDING)
        with self.assertRaisesRegex(FR.Failure, "not recorded in the configuration yet"):
            V.load_runs(cfg, None, "unofficial", self.w.anet_out)

    def test_refusals(self):
        self.refused(lambda d: d.update(extra=1), "the top level")
        self.refused(lambda d: d.update(schema="x"), "schema")
        self.refused(lambda d: d["runs"].pop("anet_v1_p4"), "runs needs exactly")
        self.refused(lambda d: d["runs"]["anet_v1_p4"].update(run_id=d["runs"]["anet_v2_1_p4"]["run_id"]), "run id of anet_v1_p4")
        self.refused(lambda d: d["c2_evaluation"].update(run_id="x"), "20261121-ARCHEVAL-332fbdf")
        self.refused(lambda d: d["code"].pop(V.THIS_FILE.replace("pad_anet_eval", "pad_anet_run")), "code must pin")
        self.refused(lambda d: d["cap"].update(unit="CPU hours"), "cap.hours")
        self.refused(lambda d: d["outputs"]["public"].pop(), "outputs differ")
        self.refused(lambda d: d["choices_of_the_author"].update(cap="mine"), "choice of the author")
        self.refused(lambda d: d.update(test_read="x"), "test_read must state")
        for phrase in ("no automatic resume", "test_read_started", "second reading", "C5 card"):
            self.refused(lambda d: d.update(after_stop=d["after_stop"].replace(phrase, "x")), "after_stop must state")

    def test_delivered_configuration(self):
        p = ROOT / V.DEFAULT_CONFIG
        if not p.is_file():
            skip_missing(self, "delivered configuration")
        d = json.loads(p.read_text(encoding="utf-8"))
        pinned = [d["comparison"]["path"], d["run_configuration"]["path"], d["protocol"]["path"], d["c2_evaluation"]["units"]["path"], *V.CODE_FILES]
        if any(not (ROOT / x).is_file() for x in pinned):
            skip_missing(self, "delivered configuration")
        V.load_config(p)
        self.assertEqual(MC.doc(runs=d["runs"]), d)
        V.check_code(d)


# ============================================================================ B i 48 riferimenti di C2 e la riproduzione

class TestBC2(unittest.TestCase):
    def setUp(self):
        self.w = world()

    def pins(self):
        cfg, _ = V.load_config(self.w.eval_cfg)
        with injected(**CNR.base_hooks(self.w)):
            return cfg, V.load_pins(cfg, self.w.base)

    def test_c2_reproduced_exactly(self):
        cfg, pins = self.pins()
        c2 = V.load_c2_scores(cfg, pins, self.w.base)
        self.assertEqual(len(c2), 48)
        self.assertTrue(V.reproduce_c2(pins, c2)["exact"])

    def test_a_different_c2_number_fails(self):
        cfg, pins = self.pins()
        c2 = V.load_c2_scores(cfg, pins, self.w.base)
        k = (4, "ResNet-18", 2, 43)
        c2[k]["re-estimated"]["s"] = np.clip(c2[k]["re-estimated"]["s"] + 0.2, 0, 1)
        with self.assertRaisesRegex(FR.Failure, "C2 reproduction: the primary analysis differs"):
            V.reproduce_c2(pins, c2)
        c2 = V.load_c2_scores(cfg, pins, self.w.base)
        k = (12, "LivenessNet", 3, 44)
        c2[k]["ordinary"]["threshold_eer"] = 0.4
        with self.assertRaisesRegex(FR.Failure, "C2 reproduction: the 8 conditions differ"):
            V.reproduce_c2(pins, c2)
        pins.c2_units = pins.c2_units + b"x"
        with self.assertRaisesRegex(FR.Failure, "units.csv differs"):
            V.reproduce_c2(pins, V.load_c2_scores(cfg, pins, self.w.base))

    def test_c2_files_checked(self):
        cfg, pins = self.pins()
        rdir = self.w.base / "eval" / "restricted"
        fn = pins.c2_cells[5]["test_scores_file"]
        saved = (rdir / fn).read_bytes()
        self.addCleanup((rdir / fn).write_bytes, saved)
        (rdir / fn).write_bytes(saved.replace(b",test,", b",dev,", 1))
        with self.assertRaisesRegex(FR.Failure, "differs from the restricted SHA256SUMS"):
            V.load_c2_scores(cfg, pins, self.w.base)
        bad = dict(cfg)
        bad["c2_evaluation"] = {**cfg["c2_evaluation"], "restricted_sha256sums_sha256": "0" * 64}
        with self.assertRaisesRegex(FR.Failure, "C2 restricted SHA256SUMS: SHA-256 differs"):
            V.load_c2_scores(bad, pins, self.w.base)

    def test_c2_rows_checked(self):
        """Con il SHA256SUMS riscritto (un file alterato e rifissato), i controlli di contenuto: ruolo, rete e checkpoint di
        un'altra cella."""
        cfg, pins = self.pins()
        rdir = self.w.base / "eval" / "restricted"
        fn = pins.c2_cells[0]["test_scores_file"]
        saved_f, saved_s = (rdir / fn).read_bytes(), (rdir / "SHA256SUMS").read_bytes()
        self.addCleanup((rdir / "SHA256SUMS").write_bytes, saved_s)
        self.addCleanup((rdir / fn).write_bytes, saved_f)
        for change, needle in ((lambda t: t.replace(",test,", ",dev,", 1), "rows of another cell"),
                               (lambda t: t.replace(",LivenessNet,", ",ResNet-18,"), "rows of another cell"),          # rete di un'altra cella
                               (lambda t: t.replace(pins.c2_cells[0]["checkpoint_sha256"], "0" * 64), "rows of another cell"),   # altro checkpoint
                               (lambda t: re.sub(r",(\d),([0-9.]+),(\d),([0-9.]+),", lambda m: m.group(0), t, count=1), None)):
            if needle is None:
                continue
            t = change(saved_f.decode())
            (rdir / fn).write_text(t, encoding="utf-8", newline="")
            sums = saved_s.decode().replace(sha(saved_f), sha(t.encode()))
            (rdir / "SHA256SUMS").write_text(sums, encoding="utf-8")
            c = {**cfg, "c2_evaluation": {**cfg["c2_evaluation"], "restricted_sha256sums_sha256": sha(sums.encode())}}
            with self.assertRaisesRegex(FR.Failure, needle):
                V.load_c2_scores(c, pins, self.w.base)


# ============================================================================ C i sei run

class TestCRuns(unittest.TestCase):
    def setUp(self):
        self.w = world()

    def load(self, mode="unofficial"):
        cfg, _ = V.load_config(self.w.eval_cfg)
        with injected(**CNR.base_hooks(self.w)):
            pins = V.load_pins(cfg, self.w.base)
            return V.load_runs(cfg, pins, mode, self.w.anet_out)

    def test_six_runs_loaded(self):
        runs = self.load()
        self.assertEqual(sorted(runs), sorted(NR.RUNS))
        self.assertEqual(sum(len(r["models"]) for r in runs.values()), 72)
        self.assertEqual(sum(len(r["not_evaluable"]) for r in runs.values()), 0)

    def test_only_the_last_attempt_is_evaluated(self):
        root = self.w.anet_out
        rid = V.latest_attempt("anet_v1_p4", root)
        later = root / (rid + "-r2")
        shutil.copytree(root / rid, later)
        self.addCleanup(shutil.rmtree, later, True)
        with self.assertRaisesRegex(FR.Failure, "the last attempt in the runs root is"):
            self.load()

    def test_official_evaluation_needs_official_runs(self):
        with self.assertRaisesRegex(FR.Failure, "an official evaluation needs an official run"):
            self.load("official")

    def test_a_changed_run_file_stops(self):
        rid = V.latest_attempt("anet_v2_1_p4", self.w.anet_out)
        p = self.w.anet_out / rid / "restricted" / "AttackNetV2_1_fold2_seed43_dev_ordinary.csv"
        saved = p.read_bytes()
        self.addCleanup(p.write_bytes, saved)
        p.write_bytes(saved + b"\n")
        with self.assertRaisesRegex(FR.Failure, "differs from its SHA256SUMS line"):
            self.load()

    def test_status_recorded_in_the_configuration(self):
        d = self.w.eval_doc()
        d["runs"]["anet_v2_2_p12"]["status_sha256"] = "0" * 64
        self.w.write_eval_cfg(d)
        self.addCleanup(self.w.write_eval_cfg)
        with self.assertRaisesRegex(FR.Failure, "status.json differs from the SHA-256 recorded"):
            self.load()


# ============================================================================ D flusso di evaluate (valutatore finto)

class TestDEvaluate(unittest.TestCase):
    def setUp(self):
        self.w = world()
        shutil.rmtree(self.w.eval_out, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.w.eval_out, True)

    def test_pass_outputs_and_order(self):
        calls = []
        rc, out, err = run_eval(self.w.eval_argv(), eval_hooks(self.w, score=fake_score(calls=calls)))
        self.assertEqual(rc, 0, (err + out)[-4000:])
        run = eval_folder(self.w)
        self.assertEqual(sorted(os.listdir(run / "public")), sorted(V.PUBLIC_OUTPUTS))
        self.assertEqual(len(os.listdir(run / "restricted")), 73)
        # ogni riproduzione del dev prima di qualunque test; lo schema dei fold caricato solo dopo
        roles = [c[0] for c in calls]
        self.assertEqual(roles, ["dev"] * 72 + ["test"] * 72)
        self.assertTrue(all(c[2] == 0 for c in calls if c[0] == "dev"))
        doc = json.loads((run / "public" / "results.json").read_text())
        self.assertEqual(doc["status"], "PASS")
        self.assertTrue(doc["c2_reproduction"]["exact"])
        self.assertEqual(doc["c2_opposite"], {"gap_ordinary": True, "gap_re_estimated": True, "variation_difference": True})
        pa = doc["primary_analysis"]
        self.assertEqual(sorted(pa["contrasts"]), sorted(DEF.ATTACKNETS))
        self.assertTrue(all(c["computed"] for c in pa["contrasts"].values()))
        self.assertEqual(sorted(doc["descriptive"]["pairs"]), sorted(["patience_12_fixed_0.5", "patience_4_dev_eer", "patience_12_dev_eer"]))
        for f in os.listdir(run / "public"):
            self.assertEqual(PAC.output_problems((run / "public" / f).read_text(), FR.Inputs()), [], f)
        logt = (run / "run.log").read_text()
        self.assertEqual(logt.count("RESULT:"), 1)
        self.assertTrue(logt.rstrip().endswith("RESULT: PASS (evaluate)"))
        st = json.loads((run / "status.json").read_text())
        self.assertTrue(st["c2_reproduction"]["exact"])
        tab = FR.parse_csv((run / "public" / "table.csv").read_bytes(), V.TABLE_COLUMNS, "table.csv")
        self.assertEqual(len(tab), 10)
        self.assertIsNotNone(doc["c2_reproduction"]["primary_contrast"]["mean"])
        self.assertEqual(len(doc["test_one_class"]), len(DEF.NETWORKS) * 8)
        self.assertEqual({r["models"] for r in doc["test_one_class"]}, {12})

    def test_dev_mismatch_stops_before_the_test(self):
        calls = []

        def over(rec, s, p):
            return p + (0.01 if rec["key"].endswith("AttackNetV1_fold2_seed43") else 0.0)
        rc, out, err = run_eval(self.w.eval_argv(), eval_hooks(self.w, score=fake_score(dev_over=over, calls=calls)))
        self.assertEqual(rc, 4, (err + out)[-3000:])
        self.assertNotIn("test", [c[0] for c in calls])
        run = eval_folder(self.w)
        self.assertFalse((run / "public").exists())
        self.assertFalse((run / "restricted").exists())

    def test_c2_not_reproduced_fails_before_any_test_frame(self):
        calls = []
        with mock.patch.object(V, "reproduce_c2", lambda pins, c2: FR.fail("C2 reproduction: the primary analysis differs (test)")):
            rc, out, err = run_eval(self.w.eval_argv(), eval_hooks(self.w, score=fake_score(calls=calls)))
        self.assertEqual(rc, 1)
        self.assertEqual(calls, [])
        st = json.loads((eval_folder(self.w) / "status.json").read_text())
        self.assertIn("C2 reproduction", st["reason"])

    def test_non_finite_test_scores_stop_before_the_analysis(self):
        def over(rec, s, p):
            if rec["key"].endswith("AttackNetV2_2_fold3_seed44") and s == "re-estimated":
                p = p.copy()
                p[:3] = np.nan
            return p

        def score(ctx, rec, parts):
            out = fake_score(test_over=over)(ctx, rec, parts)
            for s in D.BN_STATES:
                for role, x in out[s].items():
                    x["non_finite"] = int(np.sum(~np.isfinite(x["p"]))) if role == "test" else 0
            return out
        rc, out, err = run_eval(self.w.eval_argv(), eval_hooks(self.w, score=score))
        self.assertEqual(rc, 6, (err + out)[-3000:])
        run = eval_folder(self.w)
        st = json.loads((run / "status.json").read_text())
        self.assertEqual(st["status"], "NON_FINITE_TEST")
        self.assertEqual(st["non_finite_test"], [{"model": "anet_v2_2_p4_AttackNetV2_2_fold3_seed44", "bn_state": "re-estimated",
                                                  "non_finite_frames": 3}, {"model": "anet_v2_2_p12_AttackNetV2_2_fold3_seed44",
                                                                            "bn_state": "re-estimated", "non_finite_frames": 3}])
        self.assertFalse((run / "public").exists())
        self.assertTrue((run / "restricted" / "SHA256SUMS").is_file())

    def test_refusals_before_any_folder(self):
        w = self.w
        cases = [([x for x in w.eval_argv() if x != "--i-am-authorized"], "pass --i-am-authorized"),
                 (w.eval_argv(unofficial=False), "needs --unofficial-out-root"),
                 (w.eval_argv(attempt=None), "--attempt N is required"),
                 (w.eval_argv(out=ROOT / "results" / "elsewhere"), "inside the repository")]
        for argv, needle in cases:
            with self.subTest(needle):
                rc, out, err = run_eval(argv, eval_hooks(w))
                self.assertEqual(rc, 1)
                self.assertIn(needle, err)
                self.assertFalse(w.eval_out.exists())
        h = eval_hooks(w)
        h["v"]["REPO_STATE"] = lambda root: {"commit": "abcdef0", "dirty": True}
        rc, out, err = run_eval(w.eval_argv(), h)
        self.assertIn("work tree is not clean", err)

    def test_check_inputs(self):
        before = CPR.tree(self.w.tmp)
        rc, out, err = run_eval(self.w.eval_argv("check-inputs"), eval_hooks(self.w))
        self.assertEqual(rc, 0, err)
        self.assertIn("C2 reproduced exactly", out)
        self.assertIn("six runs PASS, last attempts, 72 evaluable models, 0 not evaluable", out)
        self.assertIn("the records of 7 commands in the package equal to the runs root", out)
        self.assertEqual(CPR.tree(self.w.tmp), before)


class TestDDetails(unittest.TestCase):
    def setUp(self):
        self.w = world()

    def test_phase_1_must_be_complete_for_the_12_positions(self):
        """Un run il cui status.json non elenca tutte le 12 posizioni nella fase 1 (copia dei run, status.json alterato e
        rifissato nella configurazione): rifiutato."""
        w = self.w
        root = Path(tempfile.mkdtemp(dir=w.tmp, prefix="runs-"))
        self.addCleanup(shutil.rmtree, root, True)
        for d in os.listdir(w.anet_out):
            shutil.copytree(w.anet_out / d, root / d)
        rid = V.latest_attempt("anet_v1_p12", root)
        st = json.loads((root / rid / "status.json").read_text())
        st["complete_phase_1"] = st["complete_phase_1"][:-1]
        (root / rid / "status.json").write_text(json.dumps(st, indent=1) + "\n", encoding="utf-8")
        cfg = {**V.load_config(w.eval_cfg)[0], "runs": MC.runs_record(root)}
        with injected(**CNR.base_hooks(w)):
            pins = V.load_pins(cfg, w.base)
            with self.assertRaisesRegex(FR.Failure, "phase 1 not complete for the 12 positions"):
                V.load_runs(cfg, pins, "unofficial", root)

    def test_a_new_unit_without_its_c2_reference_refused(self):
        cfg, _ = V.load_config(self.w.eval_cfg)
        with injected(**CNR.base_hooks(self.w)):
            pins = V.load_pins(cfg, self.w.base)
        c2 = V.load_c2_scores(cfg, pins, self.w.base)
        c = E.PRIMARY[0]
        fake = {c: [{"unit": (V.DATASET, 9, 42), "networks": {"AttackNetV1": {}}}]}
        with mock.patch.object(E, "build_conditions", lambda plan, recs, meta: fake):
            with self.assertRaisesRegex(FR.Failure, "has no C2 reference"):
                V.build_conditions([], [], {}, c2)

    def test_the_opposite_of_c2_is_required_before_the_test(self):
        """B4: l'opposto di C2 controllato sulle sole unità di C2, prima di TensorFlow e di qualunque frame (nessuna chiamata
        al valutatore)."""
        shutil.rmtree(self.w.eval_out, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.w.eval_out, True)
        calls = []
        with mock.patch.object(V, "c2_opposite_check", lambda pa, pins: {"gap_ordinary": True, "gap_re_estimated": False, "variation_difference": True}):
            rc, out, err = run_eval(self.w.eval_argv(), eval_hooks(self.w, score=fake_score(calls=calls)))
        self.assertEqual(rc, 1)
        self.assertIn("on the C2 units is not the exact opposite of C2", err)
        self.assertEqual(calls, [])
        st = json.loads((eval_folder(self.w) / "status.json").read_text())
        self.assertFalse(st["test_read_started"])
        self.assertFalse((eval_folder(self.w) / "restricted").exists())

    def test_the_opposite_of_c2_confirmed_after_the_analysis(self):
        shutil.rmtree(self.w.eval_out, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.w.eval_out, True)
        real, seen = V.c2_opposite_check, []

        def check(pa, pins):
            seen.append(1)
            out = real(pa, pins)
            return out if len(seen) == 1 else {**out, "gap_ordinary": False}
        with mock.patch.object(V, "c2_opposite_check", check):
            rc, out, err = run_eval(self.w.eval_argv(), eval_hooks(self.w))
        self.assertEqual(rc, 1)
        self.assertIn("is not the exact opposite of C2", err)
        st = json.loads((eval_folder(self.w) / "status.json").read_text())
        self.assertTrue(st["test_read_started"])
        self.assertTrue((eval_folder(self.w) / "restricted" / "SHA256SUMS").is_file())
        self.assertEqual(len(os.listdir(eval_folder(self.w) / "restricted")), 73)
        self.assertFalse((eval_folder(self.w) / "public").exists())

    def test_c2_ids_and_thresholds_checked(self):
        cfg, _ = V.load_config(self.w.eval_cfg)
        with injected(**CNR.base_hooks(self.w)):
            pins = V.load_pins(cfg, self.w.base)
        c2 = V.load_c2_scores(cfg, pins, self.w.base)
        lists = test_lists(self.w)
        V.c2_ids_check(c2, lists)
        bad = {f: rows[1:] for f, rows in lists.items()}
        with self.assertRaisesRegex(FR.Failure, "its frame identifiers are not the test keys"):
            V.c2_ids_check(c2, bad)
        self.assertEqual(V.c2_thresholds_check(pins, c2), 48)
        pins.c2_results = json.loads(json.dumps(pins.c2_results))
        pins.c2_results["models"][7]["dev_eer_threshold"]["re-estimated"] = 0.4
        with self.assertRaisesRegex(FR.Failure, "differs from models\\[\\].dev_eer_threshold"):
            V.c2_thresholds_check(pins, c2)

    def test_second_reading_rule(self):
        """B1: dopo un tentativo che ha letto il test un nuovo tentativo è una seconda lettura: rifiutato senza
        --second-reading, ammesso con, e registrato; --second-reading senza una lettura precedente rifiutato."""
        w = self.w
        shutil.rmtree(w.eval_out, ignore_errors=True)
        self.addCleanup(shutil.rmtree, w.eval_out, True)
        rc, out, err = run_eval(w.eval_argv() + ["--second-reading"], eval_hooks(w))
        self.assertEqual(rc, 1)
        self.assertIn("no earlier attempt started reading the test", err)
        rc, out, err = run_eval(w.eval_argv(), eval_hooks(w))
        self.assertEqual(rc, 0, err[-2000:])
        rc, out, err = run_eval(w.eval_argv(attempt=2), eval_hooks(w))
        self.assertEqual(rc, 1)
        self.assertIn("a new attempt is a second reading of the test", err)
        rc, out, err = run_eval(w.eval_argv(attempt=2) + ["--second-reading"], eval_hooks(w))
        self.assertEqual(rc, 0, err[-2000:])
        st = json.loads((eval_folder(w, 2) / "status.json").read_text())
        self.assertTrue(st["second_reading"])
        self.assertEqual(len(st["attempt"]["earlier_readings"]), 1)

    def test_a_stop_after_the_reading_started_is_a_reading(self):
        """N1: un arresto brusco (nessun gestore) dopo l'inizio della lettura del test: la cartella com'è al momento della
        lettura (copiata dalla lettura finta delle liste) ha status.json RUNNING con test_read_started vero e restricted/
        ancora vuota; un nuovo tentativo su quella radice è una seconda lettura."""
        w = self.w
        shutil.rmtree(w.eval_out, ignore_errors=True)
        self.addCleanup(shutil.rmtree, w.eval_out, True)
        killed = Path(tempfile.mkdtemp(dir=w.tmp, prefix="killed-"))
        self.addCleanup(shutil.rmtree, killed, True)

        class Stop(Exception):
            pass

        def stop(ctx):
            run = eval_folder(w)
            shutil.copytree(run, killed / run.name)
            raise Stop("abrupt stop (test)")
        with mock.patch.object(E, "load_test_lists", stop):
            rc, out, err = run_eval(w.eval_argv(), eval_hooks(w))
        self.assertEqual(rc, 1)
        st = json.loads((killed / eval_folder(w).name / "status.json").read_text())
        self.assertEqual((st["status"], st["test_read_started"]), ("RUNNING", True))
        self.assertEqual(os.listdir(killed / eval_folder(w).name / "restricted"), [])          # nessun punteggio del test ancora
        self.assertTrue(V.previous_attempts(killed)[0]["test_read_started"])
        with self.assertRaisesRegex(FR.Failure, "a new attempt is a second reading of the test"):
            V.attempt_rule(killed, 2)
        self.assertEqual(V.attempt_rule(killed, 2, True)["earlier_readings"], [eval_folder(w).name])

    def test_running_or_restricted_attempts_count_as_readings(self):
        """N1, dalla parte sicura: un tentativo rimasto RUNNING con test_read_started falso, o con restricted/ non vuota,
        conta come lettura, e così uno stato illeggibile o senza test_read_started; uno FAIL con test_read_started falso e
        restricted/ vuota no."""
        root = Path(tempfile.mkdtemp(dir=self.w.tmp, prefix="attempts-"))
        self.addCleanup(shutil.rmtree, root, True)
        run = root / "20261123-ANETEVAL-abcdef0"
        run.mkdir()
        running, failed = json.dumps({"status": "RUNNING", "test_read_started": False}), json.dumps({"status": "FAIL", "test_read_started": False})
        for status, restricted, read in ((running, None, True), (failed, "x.csv", True), (failed, "", False), (failed, None, False),
                                         ("{not json", None, True), (json.dumps({"status": "FAIL"}), None, True)):
            with self.subTest(status=status, restricted=restricted):
                shutil.rmtree(run / "restricted", ignore_errors=True)
                (run / "status.json").write_text(status, encoding="utf-8")
                if restricted is not None:
                    (run / "restricted").mkdir()
                    if restricted:
                        (run / "restricted" / restricted).write_text("x", encoding="utf-8")
                self.assertEqual(V.previous_attempts(root)[0]["test_read_started"], read)

    def test_a_dev_mismatch_attempt_is_not_a_reading(self):
        w = self.w
        shutil.rmtree(w.eval_out, ignore_errors=True)
        self.addCleanup(shutil.rmtree, w.eval_out, True)
        rc, out, err = run_eval(w.eval_argv(), eval_hooks(w, score=fake_score(dev_over=lambda rec, s, p: p + 0.01)))
        self.assertEqual(rc, 4)
        rc, out, err = run_eval(w.eval_argv(attempt=2), eval_hooks(w))
        self.assertEqual(rc, 0, err[-2000:])

    def test_published_configuration_rule(self):
        """B2: dopo i run make_config cambia solo runs rispetto alla configurazione pubblicata."""
        d = self.w.eval_doc()
        self.assertEqual(MC.published_differences(d, {**d, "runs": V.RUNS_PENDING}), [])
        self.assertEqual(MC.published_differences(d, {**d, "cap": {**d["cap"], "hours": 3.0}}), ["cap"])
        self.assertEqual(MC.published_differences(d, {**d, "code": {}}), ["code"])
        ref = self.w.tmp / "published.json"
        ref.write_text(json.dumps({**d, "runs": V.RUNS_PENDING}), encoding="utf-8")
        self.assertEqual(MC.published_config(ref)["runs"], V.RUNS_PENDING)

    def test_opposite_check_sees_a_different_c2_value(self):
        cfg, _ = V.load_config(self.w.eval_cfg)
        with injected(**CNR.base_hooks(self.w)):
            pins = V.load_pins(cfg, self.w.base)
        q = {"mean": {m: 0.1 for m in DEF.METRICS}, "interval": {m: {"low": -0.2, "high": 0.3, "valid_resamples": 5} for m in DEF.METRICS}}
        pins.c2_results = {"primary_analysis": {
            "gap_ordinary": {"mean_delta": {m: -0.1 for m in DEF.METRICS}, "interval": {m: {"low": -0.3, "high": 0.2, "valid_resamples": 5} for m in DEF.METRICS}},
            "gap_re_estimated": {"mean_delta": {m: -0.1 for m in DEF.METRICS}, "interval": {m: {"low": -0.3, "high": 0.2, "valid_resamples": 6} for m in DEF.METRICS}},
            "contrast": {"mean": {m: -0.1 for m in DEF.METRICS}, "interval": {m: {"low": -0.3, "high": 0.2, "valid_resamples": 5} for m in DEF.METRICS}}}}
        pa = {"descriptive_vs_resnet18": {DEF.REFERENCE: {"gap_ordinary": q, "gap_re_estimated": q, "variation_difference": q}}}
        self.assertEqual(V.c2_opposite_check(pa, pins), {"gap_ordinary": True, "gap_re_estimated": False, "variation_difference": True})


class TestDOneClass(unittest.TestCase):
    def test_one_class_models_on_the_test_counted(self):
        """B3: un modello che decide attacco per ogni frame del test (punteggi tutti 0,9) è contato, a 0,5 e alla soglia EER,
        nel suo stato della BN."""
        w = world()
        shutil.rmtree(w.eval_out, ignore_errors=True)
        self.addCleanup(shutil.rmtree, w.eval_out, True)

        def over(rec, s, p):
            return np.full_like(p, 0.999) if rec["key"] == "anet_v1_p4_AttackNetV1_fold2_seed42" and s == "ordinary" else p
        rc, out, err = run_eval(w.eval_argv(), eval_hooks(w, score=fake_score(test_over=over)))
        self.assertEqual(rc, 0, (err + out)[-2000:])
        doc = json.loads((eval_folder(w) / "public" / "results.json").read_text())
        rows = {(r["network"], r["patience"], r["bn_state"], r["threshold"]): r for r in doc["test_one_class"]}
        for t in ("fixed_0.5", "dev_eer"):
            self.assertEqual((rows[("AttackNetV1", 4, "ordinary", t)]["one_class"], rows[("AttackNetV1", 4, "ordinary", t)]["all_attack"]), (1, 1))
        self.assertEqual(rows[("AttackNetV1", 4, "re-estimated", "fixed_0.5")]["one_class"], 0)


class TestEGuardian(unittest.TestCase):
    def test_not_evaluable_attacknet_is_not_computed(self):
        """Un run con modelli non valutabili (val_loss mai finita per il seed 43 di AttackNetV2_2): il guardiano non calcola
        variazione, contrasto né confronto per quella rete, con il motivo; le altre reti procedono."""
        w = world()

        def hist(net, seed, h):
            return {**h, "val_loss": [float("nan")] * len(h["val_loss"])} if net == "AttackNetV2_2" and seed == 43 else h
        try:
            w.make_runs(history_over=hist)
            w.write_records()
            w.write_eval_cfg()
            shutil.rmtree(w.eval_out, ignore_errors=True)
            rc, out, err = run_eval(w.eval_argv(), eval_hooks(w))
            self.assertEqual(rc, 0, (err + out)[-3000:])
            doc = json.loads((eval_folder(w) / "public" / "results.json").read_text())
            c = doc["primary_analysis"]["contrasts"]["AttackNetV2_2"]
            self.assertFalse(c["computed"])
            self.assertIn("not evaluable: no finite val_loss", c["text"])
            self.assertIn("anet_v2_2_p4/", c["text"])
            self.assertNotIn("anet_v2_2_p12/", c["text"])
            self.assertTrue(doc["primary_analysis"]["contrasts"]["AttackNetV1"]["computed"])
            self.assertEqual(len(doc["not_evaluable"]), 8)
            # table.csv: unità e denominatori della rete sulle sue sole unità (F2)
            rows = {(r["network"], r["bn_state"]): r for r in csv.DictReader(io.StringIO((eval_folder(w) / "public" / "table.csv").read_text()))}
            for s in D.BN_STATES:
                a, ref = rows[("AttackNetV2_2", s)], rows[("LivenessNet", s)]
                self.assertLess(int(a["units"]), int(ref["units"]))
                for k in ("n_attack", "n_bona_fide"):
                    self.assertEqual(int(a[k]) * int(ref["units"]), int(ref[k]) * int(a["units"]), k)
                    self.assertLess(int(a[k]), int(ref[k]), k)
        finally:
            w.make_runs()
            w.write_records()
            w.write_eval_cfg()
            shutil.rmtree(w.eval_out, ignore_errors=True)


# ============================================================================ E2 record dei run e script di lancio

class TestERunRecords(unittest.TestCase):
    def setUp(self):
        self.w = world()
        self.cfg, _ = V.load_config(self.w.eval_cfg)
        self.addCleanup(self.w.write_records)

    def check(self):
        return V.check_run_copies(self.cfg, self.w.base, self.w.anet_out)

    def test_copies_equal_to_the_runs_root(self):
        got = self.check()
        self.assertEqual(got["folders"], 7)                  # sei run e un check-estimator
        self.assertEqual(sorted(os.listdir(self.w.records / V.latest_attempt("anet_v1_p4", self.w.anet_out))),
                         ["conditions.txt", "public", "run.log", "status.json"])
        self.assertFalse(any("restricted" in str(p) for p in self.w.records.rglob("*")))

    def test_a_changed_byte_stops(self):
        rid = V.latest_attempt("anet_v2_1_p12", self.w.anet_out)
        p = self.w.records / rid / "public" / "models.csv"
        p.write_bytes(p.read_bytes() + b"\n")
        with self.assertRaisesRegex(FR.Failure, "public/models.csv differs from the file of the runs root"):
            self.check()

    def test_a_missing_or_extra_record_stops(self):
        rid = V.latest_attempt("anet_v2_2_p4", self.w.anet_out)
        shutil.rmtree(self.w.records / rid)
        with self.assertRaisesRegex(FR.Failure, "every attempt and every check-estimator"):
            self.check()
        self.w.write_records()
        (self.w.records / rid / "public" / "extra.txt").write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(FR.Failure, "the copy holds"):
            self.check()
        self.w.write_records()
        (self.w.records / rid / "conditions.txt").unlink()
        with self.assertRaisesRegex(FR.Failure, "the copy holds"):
            self.check()

    def test_a_different_copy_needs_force(self):
        rid = V.latest_attempt("anet_v1_p4", self.w.anet_out)
        p = self.w.records / rid / "status.json"
        p.write_bytes(p.read_bytes() + b" ")
        with self.assertRaisesRegex(SystemExit, "exists and differs: pass --force"):
            CR.copy_all(self.w.anet_out, self.w.launch_logs, self.w.records)
        self.assertEqual(CR.copy_all(self.w.anet_out, self.w.launch_logs, self.w.records, force=True),
                         sum(len(V.record_files(f)) + 1 for f in V.record_folders(self.w.anet_out)))
        self.check()

    def test_every_attempt_is_a_record(self):
        rid = V.latest_attempt("anet_v1_p12", self.w.anet_out)
        later = self.w.anet_out / (rid + "-r2")
        shutil.copytree(self.w.anet_out / rid, later)
        self.addCleanup(shutil.rmtree, later, True)
        self.assertIn(later, V.record_folders(self.w.anet_out))
        self.assertIn(self.w.anet_out / rid, V.record_folders(self.w.anet_out))       # il tentativo precedente resta un record
        with self.assertRaisesRegex(FR.Failure, "every attempt"):
            self.check()

    def test_evaluate_checks_the_copies(self):
        """N5: evaluate verifica le copie dei record come check-inputs (qui con la scansione sui manifest caricati), prima di
        qualsiasi lettura del test."""
        shutil.rmtree(self.w.eval_out, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.w.eval_out, True)
        rid = V.latest_attempt("anet_v2_1_p4", self.w.anet_out)
        p = self.w.records / rid / "run.log"
        p.write_bytes(p.read_bytes() + b"x")
        calls = []
        rc, out, err = run_eval(self.w.eval_argv(), eval_hooks(self.w, score=fake_score(calls=calls)))
        self.assertEqual(rc, 1)
        self.assertIn("run.log differs from the file of the runs root", err)
        self.assertEqual(calls, [])
        self.assertFalse(json.loads((eval_folder(self.w) / "status.json").read_text())["test_read_started"])

    def test_evaluate_scans_the_copies_with_the_manifests(self):
        """F3: l'identificativo di un soggetto in un conditions.txt copiato: check-inputs (senza i manifest, come dichiarato)
        passa, evaluate (con i manifest caricati) si ferma prima di qualsiasi lettura del test."""
        shutil.rmtree(self.w.eval_out, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.w.eval_out, True)
        rid = V.latest_attempt("anet_v2_2_p4", self.w.anet_out)
        p = self.w.records / rid / "conditions.txt"
        p.write_text(p.read_text(encoding="utf-8") + f"worst subject {CPR.CFR.SUBJECTS[3]}\n", encoding="utf-8")
        rc, out, err = run_eval(self.w.eval_argv("check-inputs"), eval_hooks(self.w))
        self.assertEqual(rc, 0, err)
        calls = []
        rc, out, err = run_eval(self.w.eval_argv(), eval_hooks(self.w, score=fake_score(calls=calls)))
        self.assertEqual(rc, 1)
        self.assertIn("subject id", err)
        self.assertEqual(calls, [])
        self.assertFalse(json.loads((eval_folder(self.w) / "status.json").read_text())["test_read_started"])

    def test_check_inputs_checks_the_copies(self):
        rid = V.latest_attempt("anet_v1_p12", self.w.anet_out)
        p = self.w.records / rid / "status.json"
        p.write_bytes(p.read_bytes() + b" ")
        rc, out, err = run_eval(self.w.eval_argv("check-inputs"), eval_hooks(self.w))
        self.assertEqual(rc, 1)
        self.assertIn("status.json differs from the file of the runs root", err)

    def test_forbidden_text_in_a_conditions_file_stops(self):
        rid = V.latest_attempt("anet_v1_p4", self.w.anet_out)
        (self.w.records / rid / "conditions.txt").write_text(f"launched in {Path.home()}\n", encoding="utf-8")
        with self.assertRaisesRegex(FR.Failure, "holds text that a publishable output must not hold"):
            self.check()
        (self.w.launch_logs / f"{CR.label_of(rid)}-conditions.txt").write_text(f"launched in {Path.home()}\n", encoding="utf-8")
        with self.assertRaises(SystemExit):
            CR.plan(self.w.anet_out, self.w.launch_logs)

    def test_missing_conditions_file_and_labels(self):
        rid = V.latest_attempt("anet_v2_2_p12", self.w.anet_out)
        (self.w.launch_logs / f"{CR.label_of(rid)}-conditions.txt").unlink()
        with self.assertRaisesRegex(SystemExit, "conditions file anet_v2_2_p12-attempt1-conditions.txt is missing"):
            CR.plan(self.w.anet_out, self.w.launch_logs)
        self.assertEqual(CR.label_of("20261122-ANETRUN-v2_1-p12-abcdef0-r3"), "anet_v2_1_p12-attempt3")
        self.assertEqual(CR.label_of("20261122-ANETEST-abcdef0"), "check-estimator-attempt1")


FAKE_EVAL = """import json, os, sys
with open(os.environ["FAKE_LOG"], "a") as fh:
    fh.write(json.dumps([os.path.basename(sys.argv[0])] + sys.argv[1:]) + "\\n")
print("fake", sys.argv[1])
sys.exit(int(os.environ.get("FAKE_RC_" + sys.argv[1].replace("-", "_"), "0")))
"""


class TestELaunch(unittest.TestCase):
    """delivery/aneteval/launch.zsh eseguito davvero in una cartella temporanea con pad_anet_run.py e pad_anet_eval.py
    finti (registrano i propri argomenti)."""

    def setUp(self):
        if any(shutil.which(t) is None for t in ("zsh", "caffeinate", "pmset", "sw_vers", "git")):
            skip_missing(self, "macOS tools")
        self.tmp = Path(tempfile.mkdtemp(prefix="aneteval-launch-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.repo = self.tmp / "repo"
        for d in ("scripts", "delivery/aneteval", ".venv/bin"):
            (self.repo / d).mkdir(parents=True)
        shutil.copy2(ROOT / LAUNCH, self.repo / LAUNCH)
        for n in ("pad_anet_run.py", "pad_anet_eval.py"):
            (self.repo / "scripts" / n).write_text(FAKE_EVAL, encoding="utf-8")
        os.symlink(sys.executable, self.repo / ".venv" / "bin" / "python")
        self.logs, self.calls = self.tmp / "logs", self.tmp / "calls.jsonl"

    def launch(self, *args, attempt="1", nice=0, battery=False, log_dir=None, **rc):
        env = {k: v for k, v in os.environ.items() if not k.startswith("TF_NUM_") and not k.startswith("ANETEVAL_")}
        # un pmset finto prima nel PATH, all'alimentatore o a batteria: i test non dipendono dalla macchina
        fake = self.tmp / "fakebin"
        fake.mkdir(exist_ok=True)
        source = "Battery Power" if battery else "AC Power"
        (fake / "pmset").write_text(f"#!/bin/sh\nif [ \"$2\" = batt ]; then echo \"Now drawing from '{source}'\"; "
                                    "else echo ' powermode 0'; fi\n", encoding="utf-8")
        # un ps finto per il nice di launch.zsh (ps -o nice= -p $$): il test non dipende dal nice di chi lancia la suite
        (fake / "ps").write_text(f"#!/bin/sh\necho '  {nice}'\n", encoding="utf-8")
        for tool in ("pmset", "ps"):
            (fake / tool).chmod(0o755)
        env["PATH"] = f"{fake}{os.pathsep}{env.get('PATH', '')}"
        env["ZDOTDIR"] = str(fake)                                           # nessun file d'avvio di zsh dell'utente
        env.update({"ANETEVAL_LOG_DIR": str(log_dir or self.logs), "FAKE_LOG": str(self.calls), **({"ANETEVAL_ATTEMPT": attempt} if attempt else {})})
        env.update({f"FAKE_RC_{k}": str(v) for k, v in rc.items()})
        r = subprocess.run(["zsh", LAUNCH, *args], cwd=self.repo, env=env, capture_output=True, text=True, timeout=120)
        calls = [json.loads(l) for l in self.calls.read_text().splitlines()] if self.calls.exists() else []
        return r, calls

    def test_evaluate_launched_after_both_checks_with_every_condition(self):
        r, calls = self.launch("evaluate", evaluate=4)
        self.assertEqual(r.returncode, 4, r.stderr)
        self.assertEqual(calls, [["pad_anet_run.py", "check-machine"], ["pad_anet_eval.py", "check-inputs"],
                                 ["pad_anet_eval.py", "evaluate", "--attempt", "1", "--i-am-authorized"]])
        c = (self.logs / "evaluate-attempt1-conditions.txt").read_text()
        for part in ("== conditions before", "== system", "== power", "== nice of the launching shell\n0\n", "== git",
                     "check-machine exit 0", "check-inputs exit 0", "== command", "evaluate exit 4", "== conditions after"):
            self.assertIn(part, c)
        self.assertEqual(PAC.output_problems(c, FR.Inputs()), [])

    def test_a_failed_check_stops_evaluate(self):
        for which in ("check_machine", "check_inputs"):
            with self.subTest(which):
                shutil.rmtree(self.logs, ignore_errors=True)
                self.calls.unlink(missing_ok=True)
                r, calls = self.launch("evaluate", **{which: 1})
                self.assertEqual(r.returncode, 1)
                self.assertNotIn("evaluate", [c[1] for c in calls])

    def test_nice_and_attempt_required(self):
        real = subprocess.run(["ps", "-o", "nice=", "-p", str(os.getpid())], capture_output=True, text=True)
        self.assertEqual(real.stdout.strip(), str(os.nice(0)))              # la forma usata da launch.zsh, qualunque nice
        r, calls = self.launch("evaluate", nice=5)
        self.assertEqual((r.returncode, calls), (1, []))
        self.assertIn("== not launched: the nice value is not 0", (self.logs / "evaluate-attempt1-conditions.txt").read_text())
        r, calls = self.launch("evaluate", attempt=None)
        self.assertEqual((r.returncode, calls), (9, []))

    def test_log_folder_inside_the_repository_refused(self):
        r, calls = self.launch("evaluate", log_dir=self.repo / "x")
        self.assertEqual((r.returncode, calls), (9, []))
        self.assertIn("ANETEVAL_LOG_DIR must be outside the repository", r.stderr)
        self.assertFalse((self.repo / "x").exists())

    def test_an_existing_conditions_file_is_not_overwritten(self):
        self.logs.mkdir()
        (self.logs / "evaluate-attempt1-conditions.txt").write_text("earlier\n", encoding="utf-8")
        r, calls = self.launch("evaluate")
        self.assertEqual((r.returncode, calls), (9, []))
        self.assertIn("evaluate-attempt1-conditions.txt already exists", r.stderr)
        self.assertEqual((self.logs / "evaluate-attempt1-conditions.txt").read_text(), "earlier\n")

    def test_evaluate_needs_the_power_adapter(self):
        """N6: a batteria evaluate non parte, con il motivo nel file delle condizioni; check-inputs sì."""
        r, calls = self.launch("evaluate", battery=True)
        self.assertEqual((r.returncode, calls), (1, []))
        c = (self.logs / "evaluate-attempt1-conditions.txt").read_text()
        self.assertIn("Battery Power", c)
        self.assertIn("== not launched: evaluate needs the power adapter", c)
        r, calls = self.launch("check-inputs", attempt=None, battery=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual([c[1] for c in calls], ["check-machine", "check-inputs", "check-inputs"])

    def test_check_inputs_launched(self):
        r, calls = self.launch("check-inputs", attempt=None)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual([c[1] for c in calls], ["check-machine", "check-inputs", "check-inputs"])


# ============================================================================ F agganci e copie

def module_hooks(path: Path) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sorted(t.id for node in tree.body if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and node.value.value is None
                  for t in node.targets if isinstance(t, ast.Name))


def code_without_docstring(fn) -> str:
    f = ast.parse(inspect.getsource(fn)).body[0]
    if f.body and isinstance(f.body[0], ast.Expr) and isinstance(f.body[0].value, ast.Constant):
        f.body = f.body[1:]
    f.name = "f"
    return ast.unparse(f)


def hunks_of(a: str, b: str) -> list:
    a, b = a.split("\n"), b.split("\n")
    return [("\n".join(a[i1:i2]), "\n".join(b[j1:j2])) for t, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
            if t != "equal"]


COPY_HUNKS = {
    "score_model": [("    R.check_loaded(orig, ctx.cfg_pr, rec['network'])", "    C.check_loaded(orig, ctx.cfg_c5, rec['network'])"),
                    ("        out['ordinary'][role] = forward_scores(orig, part['X'], ctx.batch, ctx.sg, f'{name} {role} ordinary')",
                     "        out['ordinary'][role] = E.forward_scores(orig, part['X'], ctx.batch, ctx.sg, f'{name} {role} ordinary')"),
                    ("        out['re-estimated'][role] = forward_scores(copy, part['X'], ctx.batch, ctx.sg, f'{name} {role} re-estimated')",
                     "        out['re-estimated'][role] = E.forward_scores(copy, part['X'], ctx.batch, ctx.sg, f'{name} {role} re-estimated')")],
    "reproduce_dev": [],
    # punteggi di test non finiti contati (test_scores) invece che rifiutati, restituiti con il piano, e scritti appena calcolati
    "score_test": [       ('    meta = {}', '    meta, non_finite = ({}, [])'),
        (       "            res = (SCORE_MODEL or score_model)(ctx, rec, {'test': test})",
                "            res = (SCORE_MODEL or test_scores)(ctx, rec, {'test': test})"),
        (       '',
                "                bad = rec['test'][s].get('non_finite')\n"
                '                if bad is None:\n'
                "                    bad = int(np.sum(~np.isfinite(np.asarray(rec['test'][s]['p'], np.float64))))\n"
                '                if bad:\n'
                "                    non_finite.append({'model': rec['key'], 'bn_state': s, 'non_finite_frames': bad})\n"
                '            mw.write_all(ctx.rdir, {f"{rec[\'key\']}{TEST_SUFFIX}": csv_text(E.TEST_SCORE_COLUMNS, E.test_score_rows(rec, meta))}, '
                "newline='')"),
        ('    return meta', '    return (meta, non_finite)')],
}


# la copia di pad_anet_definitions.descriptive_analysis: i motivi per patience al posto di un solo testo
DEF_COPY_HUNKS = [('def f(conditions: dict, reasons=None) -> dict:\n    _check_conditions(conditions)\n    missing = [c for c in CONDITIONS if c not in conditions]',
  'def f(conds: dict, runs: dict) -> dict:\n    DEF._check_conditions(conds)\n    missing = [c for c in DEF.CONDITIONS if c not in conds]'),
 ("    for p, t in ((12, PRIMARY_THRESHOLD), (4, 'dev_eer'), (12, 'dev_eer')):\n        r = pair_analysis(conditions, p, t, reasons, readings=False)",
  "    for p, t in ((12, DEF.PRIMARY_THRESHOLD), (4, 'dev_eer'), (12, 'dev_eer')):\n"
  '        r = DEF.pair_analysis(conds, p, t, reasons_of(runs, p), readings=False)'),
 ('    for c in CONDITIONS:\n'
  "        units = sorted(conditions[c], key=lambda u: (u['unit'][1], u['unit'][2]))\n"
  "        ranks['/'.join(map(str, c))] = rankings(prepare(units, REUSED))\n"
  "    return {'pairs': pairs, 'rankings': ranks, 'intervals': INTERVALS_NOTE, 'note': 'descriptive: no reading'}",
  '    for c in DEF.CONDITIONS:\n'
  "        units = sorted(conds[c], key=lambda u: (u['unit'][1], u['unit'][2]))\n"
  "        ranks['/'.join(map(str, c))] = DEF.rankings(DEF.prepare(units, DEF.REUSED))\n"
  "    return {'pairs': pairs, 'rankings': ranks, 'intervals': DEF.INTERVALS_NOTE, 'note': 'descriptive: no reading'}")]


class TestFHooksAndCopies(unittest.TestCase):
    def test_descriptive_analysis_differs_from_pr_a_only_as_declared(self):
        self.assertEqual(hunks_of(code_without_docstring(DEF.descriptive_analysis), code_without_docstring(V.descriptive_analysis)), DEF_COPY_HUNKS)

    def test_hook_registry_complete(self):
        self.assertEqual(module_hooks(ROOT / V.THIS_FILE), sorted(V.MY_HOOKS))
        self.assertEqual(V.HOOKS["pad_anet_eval"], V.MY_HOOKS)
        self.assertEqual(sorted(module_hooks(ROOT / E.THIS_FILE)), sorted(V.HOOKS["pad_arch_eval"]))

    def test_every_hook_stops_the_official_commands(self):
        for m, names in V.HOOKS.items():
            mod = V if m == "pad_anet_eval" else sys.modules[m]
            for k in names:
                with self.subTest(f"{m}.{k}"):
                    saved = getattr(mod, k)
                    setattr(mod, k, lambda *a: None)
                    try:
                        with self.assertRaisesRegex(FR.Failure, f"{m}.{k}"):
                            V.check_hooks(True)
                    finally:
                        setattr(mod, k, saved)

    def test_copies_differ_from_c2_only_as_declared(self):
        for name, declared in COPY_HUNKS.items():
            with self.subTest(name):
                self.assertEqual(hunks_of(code_without_docstring(getattr(E, name)), code_without_docstring(getattr(V, name))), declared)

    def test_official_out_root_checks_the_hooks(self):
        w = world()
        official = ROOT / V.DEFAULT_OUT_ROOT
        before = sorted(os.listdir(official)) if official.is_dir() else None
        rc, out, err = run_eval(w.eval_argv(unofficial=False, out=official), eval_hooks(w))
        self.assertEqual(rc, 1)
        self.assertIn("test hooks are set", err)
        self.assertEqual(sorted(os.listdir(official)) if official.is_dir() else None, before)     # nulla scritto, esista o no


# ============================================================================ G pacchetto

def package():
    return next((d for d in (ROOT / "delivery" / "pad-attacknet-eval", ROOT.parent) if (d / "test_logs").is_dir() and (d / "README.md").is_file()
                 and (d / "SHA256SUMS").is_file()), None)


# il check-inputs di sviluppo del 22/11 (codice e configurazione precedenti, run pending), dichiarato nel README e
# pubblicato com'era: cita gli SHA-256 dei file di allora, non quelli attuali
DEVELOPMENT_LOGS = {"check_inputs_20261122_development.txt": "bee19ce887321d18d2e7508ba3ad853e61220250e425e3ab84dc43ff3566d262"}


class TestGPackage(unittest.TestCase):
    def test_manifest_lists_every_file_but_itself(self):
        pkg = package()
        if pkg is None:
            skip_missing(self, "delivered configuration")
        listed = sorted(FR.read_sums((pkg / "SHA256SUMS").read_bytes(), "SHA256SUMS"))
        files = sorted(str(p.relative_to(pkg)) for p in pkg.rglob("*") if p.is_file() and p != pkg / "SHA256SUMS" and "__pycache__" not in p.parts)
        self.assertEqual(listed, files)

    def test_readme_lines(self):
        pkg = package()
        if pkg is None:
            skip_missing(self, "delivered configuration")
        readme = " ".join((pkg / "README.md").read_text(encoding="utf-8").split())
        n = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__]).countTestCases()
        spec = importlib.util.spec_from_file_location("aneteval_mutations", ROOT / "scripts" / "check_pad_anet_eval_mutations.py")
        M = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(M)
        lines = {"tests": f"tests run: {n}, failures: 0, errors: 0, skipped: 0",
                 "list": (f"{len(M.MUTANTS)} mutations ({sum(1 for m in M.MUTANTS if m[1] == M.OFF)} offline, "
                          f"{sum(1 for m in M.MUTANTS if m[1] == M.TF)} with TensorFlow), {len(M.CONTROL)} control mutation and "
                          f"{len(M.REDUNDANT)} declared redundant, every pattern found exactly once"),
                 "killed": f"mutations killed: {len(M.MUTANTS)}/{len(M.MUTANTS)}"}
        for k, line in lines.items():
            with self.subTest(k):
                self.assertIn(line, readme)
        launch = len([t for t in dir(TestELaunch) if t.startswith("test_")])         # saltati senza gli strumenti di macOS
        lines["linux"] = f"tests run: {n}, failures: 0, errors: 0, skipped: {launch}"
        self.assertIn(lines["linux"], readme)
        self.assertEqual(sorted(set(re.findall(r"tests run: \d+, failures: \d+, errors: \d+, skipped: \d+", readme))), sorted({lines["tests"], lines["linux"]}))

    def test_every_log_cites_the_current_sha256_of_its_files(self):
        pkg = package()
        if pkg is None:
            skip_missing(self, "delivered configuration")
        if REGENERATING_LOGS:
            skip_missing(self, "logs being regenerated")
        logs = sorted(x for x in (pkg / "test_logs").glob("*.txt") if x.name not in DEVELOPMENT_LOGS)
        self.assertGreaterEqual(len(logs), 4)
        for name, h in DEVELOPMENT_LOGS.items():          # il log di sviluppo dichiarato, com'era, con le impronte di allora
            self.assertEqual(sha((pkg / "test_logs" / name).read_bytes()), h, name)
        for log in sorted((pkg / "test_logs").glob("*.txt")):        # nessun percorso personale (per esempio da un avviso)
            self.assertEqual(PAC.output_problems(log.read_text(encoding="utf-8"), FR.Inputs()), [], log.name)
        for log in logs:
            with self.subTest(log.name):
                lines = log.read_text(encoding="utf-8").splitlines()
                self.assertIn("exit 0", lines)
                cited = [l.split() for l in lines if l.startswith("# SHA-256 ")]
                self.assertGreaterEqual(len(cited), 5)
                for _, _, h, f in cited:
                    self.assertEqual(h, sha((ROOT / f).read_bytes()), f)


class ZZRepositoryUntouched(unittest.TestCase):
    def test_declared_skips_only(self):
        self.assertEqual(sorted(MISSING_INPUT), ["delivered configuration", "logs being regenerated", "macOS tools"])


def guard_state() -> dict:
    out = {}
    for g in GUARDED:
        p = ROOT / g
        out[g] = sorted((str(x.relative_to(ROOT)), x.stat().st_mtime_ns, x.stat().st_size) for x in p.rglob("*") if x.is_file()) if p.exists() else None
    return out


def main() -> int:
    before = guard_state()
    argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a not in ("--verbose", "--regenerating-logs")]
    prog = unittest.main(argv=argv, exit=False, verbosity=2 if "--verbose" in sys.argv else 1)
    res = prog.result
    changed = [g for g in GUARDED if before[g] != guard_state()[g]]
    for t, why in res.skipped:
        print(f"skipped {t.id()}: {why}")
    print(f"tests run: {res.testsRun}, failures: {len(res.failures)}, errors: {len(res.errors)}, skipped: {len(res.skipped)}")
    if changed:
        print(f"ERROR: the suite changed {', '.join(changed)}")
    ok = res.wasSuccessful() and not changed
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
