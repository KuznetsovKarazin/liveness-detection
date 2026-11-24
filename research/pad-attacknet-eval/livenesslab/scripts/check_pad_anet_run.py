"""
Test di scripts/pad_anet_run.py (C5, i sei run delle AttackNet): configurazione, file fissati, regola della macchina,
check-machine, check-inputs, agganci, copie dichiarate (confronto del sorgente con pad_arch_run), flusso di un run con
la regola (n) (componenti finti, senza TensorFlow) e, con TensorFlow, un run vero su frame 8×8.

Mondo sintetico in cartelle temporanee: il mondo della suite della PR A (check_pad_anet_comparison.World: preregistrazione
C2 finta, configurazione della PR #27 consegnata, prova C2 e valutazione C2 finte, configurazione C5) unito a quello della
suite dei run C2 (check_pad_arch_run.World: le 24 schede di riferimento del braccio B), più: un record finto della prova
C5 accettato da pad_anet_comparison.load_trial, i tre file del piano rigenerati con pad_anet_comparison.plan_texts, il
SHA256SUMS finto del pacchetto della PR A, la configurazione di questo runner costruita con delivery/anetrun/make_config.doc
sui file del mondo e un record finto di check-estimator. Nessuna scrittura nel repository.

Senza TensorFlow i test che costruiscono o addestrano un modello Keras sono saltati, con un motivo dichiarato in
MISSING_INPUT; gli altri girano. TMPDIR deve stare fuori dalla radice del repository.

  .venv/bin/python -I -B scripts/check_pad_anet_run.py [--verbose] [-f]
"""
import ast
import contextlib
import importlib.util
import inspect
import io
import json
import math
import os
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
import check_pad_arch_run as CPR  # noqa: E402
import nuaa_bn_recalibration as BR  # noqa: E402
import nuaa_folds_run as FR  # noqa: E402
import nuaa_session_ablation_run as SR  # noqa: E402
import pad_anet_comparison as C  # noqa: E402
import pad_anet_definitions as DEF  # noqa: E402
import pad_anet_models as AN  # noqa: E402
import pad_anet_run as NR  # noqa: E402
import pad_arch_comparison as PAC  # noqa: E402
import pad_arch_definitions as D  # noqa: E402
import pad_arch_run as R  # noqa: E402

TF_AVAILABLE = CAC.TF_AVAILABLE
MAKE_CONFIG = ROOT / "delivery" / "anetrun" / "make_config.py"
MISSING_INPUT = {
    "tensorflow": "TensorFlow is not installed: the tests that build or train a Keras model are skipped",
    "reference architecture": "src/livedetection/src/architectures.py (the reference code) is not present",
    "delivered configuration": "delivery/anetrun/pad_anet_run.json or the files it pins are not present",
    "macOS tools": "zsh, caffeinate, pmset, sw_vers or git is not present (macOS): the test of delivery/anetrun/launch.zsh is skipped",
    "logs being regenerated": ("the suite was started with --regenerating-logs: the check that every log of test_logs/ cites the current "
                               "SHA-256 of its files is not made (first of the two passes that regenerate the logs)"),
}
REGENERATING_LOGS = "--regenerating-logs" in sys.argv
STATE = CAC.STATE
DATE = "20261123"
TRIAL_ID = f"20261122-ANETTRIAL-{STATE['commit']}"
FIXED_M, FIXED_E = CAC.FIXED_M, CAC.FIXED_E
EPOCHS = 4
GUARDED = ("results/anetrun", "delivery/anetrun", "delivery/pad-attacknet-run", "delivery/x")

if (os.path.realpath(tempfile.gettempdir()) + os.sep).startswith(os.path.realpath(ROOT) + os.sep):
    raise SystemExit("TMPDIR is inside the repository root: set it outside (the synthetic world must never be written in the repository)")


def skip_missing(test: unittest.TestCase, what: str):
    """Unico modo di saltare un test: un ingresso mancante dichiarato in MISSING_INPUT."""
    test.skipTest(f"missing input: {MISSING_INPUT[what]}")


def sha(b: bytes) -> str:
    return CAC.sha(b)


def load_make_config():
    spec = importlib.util.spec_from_file_location("anetrun_make_config", MAKE_CONFIG)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MC = load_make_config()


# ============================================================================ mondo sintetico

def fake_c5_trial(w, folder: Path, over=None) -> None:
    """Un record della prova C5 accettato da pad_anet_comparison.load_trial (stessa configurazione C5 e stesso codice del
    mondo), con le misure che il piano usa (epoche, passi della fase 2, inferenza, tempo fuori dalle epoche)."""
    with injected(**CAC.hooks(w)):
        cfg, cfg_sha, *_ = C.load_config(w.c5_path, w.base)
        code = {"architecture": cfg["architectures"]["sha256"], **C.check_code(cfg)}
    nets = {n: {"epochs": [{"epoch": i + 1, "seconds_total": 10.0 + i, "cpu_seconds": 70.0 + i} for i in range(3)],
                "steps": {k: {"seconds": 2.0, "cpu_seconds": 14.0} for k in C.PHASE2_STEPS},
                "inference": {r: {"seconds": 1.0, "cpu_seconds": 7.0, "frames": 400} for r in C.TRIAL_ROLES},
                "training_overhead": {"seconds": 1.0, "cpu_seconds": 7.0}, "outcome": {"evaluable": True}} for n in AN.NETWORKS}
    doc = {"schema": C.TRIAL_SCHEMA, "run_id": TRIAL_ID, "status": "PASS", "commit": STATE["commit"], "config_sha256": cfg_sha,
           "code_sha256": code, "hooks": {"official_configuration": True, "set": [], "changed": []}, "machine": dict(FIXED_M),
           "environment": dict(FIXED_E),
           "threads": {"requested": {"intra_op": 10, "inter_op": 10}, "effective": {"intra_op": 10, "inter_op": 10}, "set_by_this_runner": True,
                       "environment": {"TF_NUM_INTRAOP_THREADS": None, "TF_NUM_INTEROP_THREADS": None}, "logical_cpus": FIXED_M["logical_cpus"]},
           "networks": nets, **(over or {})}
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    b = (json.dumps(doc, indent=1) + "\n").encode()
    (folder / "trial.json").write_bytes(b)
    (folder / "status.json").write_text(json.dumps({"run_id": TRIAL_ID, "status": "PASS", "trial_json_sha256": sha(b)}, indent=1) + "\n",
                                        encoding="utf-8")


class World(CAC.World, CPR.World):
    """Il mondo della PR A unito a quello dei run C2 (le 24 schede), con la prova C5, il piano, il pacchetto della PR A,
    la configurazione di questo runner e il record di check-estimator."""

    def build(self):
        super().build()
        self.pkg = self.base / "anetpkg"
        self.anet_out = self.tmp / "anetruns"
        self.anet_cfg = self.base / "anetrun.json"
        self.write_anet()

    def write_anet(self, trial_over=None, plan_over=None, sums_over=None, doc_over=None):
        fake_c5_trial(self, self.pkg / "trial")
        with injected(**CAC.hooks(self)):
            cfg, cfg_sha, _, pcfg, _ = C.load_config(self.c5_path, self.base)
            code = C.check_code(cfg)
            cells = C.load_c2_results(cfg, self.base)
            trial, trial_sha = C.load_trial(self.pkg / "trial", cfg, cfg_sha, code)
        texts = C.plan_texts(cfg, cfg_sha, pcfg, cells, trial, trial_sha)
        (self.pkg / "plan").mkdir(exist_ok=True)
        for k in NR.PLAN_KEYS:
            t = (plan_over or {}).get(k, texts[f"{k}.csv"])
            (self.pkg / "plan" / f"{k}.csv").write_text(t, encoding="utf-8")
        if trial_over:                                   # il piano dalla prova buona, poi la prova alterata (ancorata comunque)
            fake_c5_trial(self, self.pkg / "trial", trial_over)
        lines = {f: sha((self.pkg / f).read_bytes()) for f in ("trial/status.json", "trial/trial.json", *(f"plan/{k}.csv" for k in NR.PLAN_KEYS))}
        lines.update(sums_over or {})
        (self.pkg / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sorted(lines.items())), encoding="utf-8")
        d = MC.doc(data_sha=lambda p: sha((self.base / p).read_bytes()), code_sha=lambda p: sha((ROOT / p).read_bytes()),
                   COMPARISON="anetcmp.json", PACKAGE="anetpkg", TRIAL_ID=TRIAL_ID, BNCFG="bnrecal.json", BTCFG="bntest.json")
        for k, v in (doc_over or {}).items():
            d[k] = v
        self.write_anet_cfg(d)
        self.est_anet = self.tmp / "anetest" / f"{DATE}-ANETEST-{STATE['commit']}"
        self.write_anet_estimator()

    def write_anet_cfg(self, d: dict) -> None:
        self.anet_cfg.write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8")

    def anet_doc(self) -> dict:
        return json.loads(self.anet_cfg.read_text(encoding="utf-8"))

    def write_anet_estimator(self, status="PASS", over=None, official=False):
        shutil.rmtree(self.est_anet, ignore_errors=True)
        (self.est_anet / "public").mkdir(parents=True)
        cfg, cfg_sha = NR.load_config(self.anet_cfg)
        code = NR.check_code(cfg)
        doc = {"schema": NR.ESTIMATOR_SCHEMA, "run_id": self.est_anet.name, "status": status, "config_sha256": cfg_sha, "code_sha256": code,
               "models": [{"recalibration_pass": True} for _ in range(36)], "expected_models": 36, "machine_rule": {"equal": True},
               "threads_rule": {"equal": True}, "official": official, **(over or {})}
        b = (json.dumps(doc, indent=1) + "\n").encode()
        (self.est_anet / "public" / "estimator_check.json").write_bytes(b)
        (self.est_anet / "status.json").write_text(json.dumps({"run_id": self.est_anet.name, "status": status,
                                                               "estimator_check_json_sha256": sha(b)}), encoding="utf-8")

    def anet_argv(self, cmd: str, *extra, run="anet_v1_p4", attempt=1, est=True, unofficial=True, out=None) -> list:
        a = [cmd, "--config", str(self.anet_cfg), "--input-root", str(self.base)]
        if cmd in ("run", "check-estimator"):
            a += ["--out-root", str(out or self.anet_out), "--date", DATE, "--i-am-authorized"] + (["--unofficial-out-root"] if unofficial else [])
            if attempt is not None:
                a += ["--attempt", str(attempt)]
        if cmd == "run":
            a += ["--run", run] + (["--estimator-check", str(self.est_anet)] if est else [])
        return a + list(extra)


WORLD = None


def world() -> World:
    global WORLD
    if WORLD is None:
        WORLD = World()
    return WORLD


@contextlib.contextmanager
def injected(n=None, c=None, p=None, sr=None, fr=None, r=None, br=None):
    """Punti di iniezione di pad_anet_run (n), pad_anet_comparison (c), pad_arch_comparison (p),
    nuaa_session_ablation_run (sr), nuaa_folds_run (fr), pad_arch_run (r) e nuaa_bn_recalibration (br)."""
    mods = ((NR, n), (C, c), (PAC, p), (SR, sr), (FR, fr), (R, r), (BR, br))
    saved = [(m, k, getattr(m, k)) for m, d in mods for k in (d or {})]
    try:
        for m, d in mods:
            for k, v in (d or {}).items():
                setattr(m, k, v)
        yield
    finally:
        for m, k, v in saved:
            setattr(m, k, v)


def base_hooks(w: World, **n) -> dict:
    """Le iniezioni della PR A (stato git, forma 8×8, SHA-256 dello schema e delle celle, architettura di riferimento,
    memoria fissa) più macchina e versioni uguali alla prova C5 finta e lo stato git per questo runner."""
    h = CAC.hooks(w)
    h["r"] = {"MACHINE": lambda: dict(FIXED_M), "ENVIRONMENT": lambda: dict(FIXED_E)}
    h["fr"] = {**h["fr"], "environment": lambda: dict(FIXED_E)}
    h["n"] = {"REPO_STATE": lambda root: dict(STATE), **n}
    return h


def run_main(argv, hooks: dict) -> tuple:
    out, err = io.StringIO(), io.StringIO()
    with injected(**hooks), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = NR.main(argv)
    return rc, out.getvalue(), err.getvalue()


# ============================================================================ componenti finti (flusso senza TensorFlow)

SCORE = {"AttackNetV1": 0.3, "AttackNetV2_1": 0.4, "AttackNetV2_2": 0.6}


def fake_val_loss(y, c: float) -> float:
    y = np.asarray(y)
    return float(np.mean(-np.log(np.where(y == 1, c, 1 - c))))


def fake_history(y_dev, network: str, epochs: int = EPOCHS) -> dict:
    """Storia finta: val_loss minimo all'ultima epoca, uguale al val_loss dello scorer finto sul dev."""
    vl = fake_val_loss(y_dev, SCORE[network])
    return {"loss": [0.7 - 0.01 * i for i in range(epochs)], "accuracy": [0.5 + 0.01 * i for i in range(epochs)],
            "val_loss": [vl + 0.1 * (epochs - 1 - i) for i in range(epochs)], "val_accuracy": [0.5] * epochs, "lr": [0.001] * epochs}


class FakeModel:
    def __init__(self, tag, weights_finite=True):
        self.tag = tag
        self.weights_finite = weights_finite

    def save(self, path):
        Path(path).write_bytes(f"fake model {self.tag}\n".encode())

    def get_weights(self):
        return [np.ones(3, np.float32) if self.weights_finite else np.array([1.0, np.nan], np.float32)]


def fake_trainer(log: list, history_over=None, weights_finite=None):
    """Un trainer con la firma di pad_anet_comparison.anet_trainer: storia finta (history_over(network, seed, h) la può
    alterare con NaN), valori non finiti scritti come None con l'elenco non_finite, epoca ripristinata = val_loss finita
    minima, come la regola (n)."""
    def trainer(data, training, network, seed, build_model, cap_check, timer):
        y = np.asarray(data["y_dev"])
        log.append(("train", network, seed, len(data["y_fit"]), len(y)))
        h = fake_history(y, network)
        if history_over:
            h = history_over(network, seed, h)
        for i in range(EPOCHS):
            timer.record.append({"epoch": i + 1, **{k: h[k][i] for k in R.HISTORY_KEYS}, "seconds_total": 0.1, "seconds_validation": 0.01,
                                 "cpu_seconds": 0.2})
            cap_check(f"epoch {i + 1}")
        best = DEF.finite_argmin(h["val_loss"])
        hj = {k: [DEF.finite_or_none(v) for v in h[k]] for k in R.HISTORY_KEYS}
        card = {"arch": network, "train_seed": seed, "n_fitting": len(data["y_fit"]), "n_dev": len(y), "max_epochs": training["max_epochs"],
                "epochs_run": EPOCHS, "restored_epoch": None if best is None else best + 1, "restored_by": "end of epochs (set_weights)",
                "best_val_loss": None if best is None else h["val_loss"][best], "history": hj, "non_finite": DEF.non_finite_entries(h),
                "device": "CPU", "visible_devices": ["CPU"], "lr": training["initial_learning_rate"], "batch": training["batch"],
                "early_stopping_patience": training["early_stopping_patience"], "scheduler": training["scheduler"],
                "class_weight": {"0": 1.0, "1": 1.0}, "train_seconds": 0.3, "checkpoint": "not saved by the trainer"}
        wf = True if weights_finite is None else weights_finite(network, seed)
        return card, FakeModel(f"{network} {seed}", wf)
    return trainer


def fake_probs(path, n: int) -> np.ndarray:
    c = SCORE[Path(path).name.split("_fold")[0]]
    return np.stack([np.full(n, 1 - c), np.full(n, c)], axis=1)


def fake_scorer(path, sha256):
    if FR.sha256_file(path) != sha256:
        raise FR.Failure("SHA-256 changed")
    return lambda X: fake_probs(path, len(X))


def fake_logits(path, sha256):
    return lambda X: np.log(fake_probs(path, len(X)))


def fake_forward_checked(bad=(), role=None):
    """pad_anet_comparison.forward_checked finto (passate in memoria della fase 1): NotEvaluable per le reti/seed in bad,
    solo sulla passata del ruolo dato (role: "dev" o "fitting"; None = tutte)."""
    def fwd(model, part, batch, what, state, layers=()):
        if any(t in model.tag for t in bad) and (role is None or what == f"in-memory {role}"):
            raise C.NotEvaluable(f"{DEF.NOT_EVALUABLE}: non-finite scores or logits on the {what} ({state} BN)")
        return {"p": np.zeros((len(part["y"]), 2))}, {}
    return fwd


def fake_reestimate(log: list = None, p_over=None, ordinary_over=None, raise_for=()):
    """La fase 2 finta (come check_pad_arch_run.fake_reestimate), con NotEvaluable per i modelli in raise_for."""
    def re(ctx, rec, fit, dev):
        if log is not None:
            log.append(("re-estimate", rec["name"]))
        if rec["name"] in raise_for:
            raise C.NotEvaluable(f"{DEF.NOT_EVALUABLE}: non-finite re-estimated statistics")
        n = len(dev["y"])
        c = SCORE[rec["network"]]
        p = np.clip(c + np.linspace(-0.25, 0.25, n), 0.01, 0.99)
        if p_over:
            p = p_over(rec, p)
        po = np.full(n, c)
        if ordinary_over:
            po = ordinary_over(rec, po)
        stats, cons = {}, {}
        for layer in ctx.cfg["estimator"]["orders"][rec["network"]]:
            z2 = np.zeros(2)
            stats[layer] = {"spatial": 4, "frames": len(fit["y"]), "count": 4 * len(fit["y"]), "mean64": z2, "var64": z2 + 1,
                            "old_mean": z2.astype(np.float32), "old_var": (z2 + 1).astype(np.float32), "mean32": z2.astype(np.float32),
                            "var32": (z2 + 1).astype(np.float32)}
            cons[layer] = {"max_abs_diff_mean": 0.0, "max_abs_diff_var": 0.0, "count": 4 * len(fit["y"]), "channels_outside_tolerance": 0}
        return {"stats": stats, "cons": cons, "changes": {k: {"var_ratio_min": 1.0} for k in stats}, "re_estimated": {"p": p, "z": CPR.lz(p)},
                "ordinary": {"p": po, "z": CPR.lz(po)}, "fitting_after": {"accuracy": 0.5, "val_loss": 0.7, "auroc": 0.5, "n": len(fit["y"])}}
    return re


def flow_hooks(w: World, log=None, trainer=None, reestimate=None, **n) -> dict:
    log = log if log is not None else []
    tfm = CPR.FakeTF(log=log)
    h = base_hooks(w, **{"ARCH_LOADER": lambda path: None, "TRAINER": trainer or fake_trainer(log), "MAKE_SCORER": fake_scorer,
                         "REESTIMATE": reestimate or fake_reestimate(log), **n})
    h["r"]["TF_MODULE"] = lambda: tfm
    h["sr"]["MAKE_LOGITS"] = fake_logits
    return h


def run_folder(w: World, run="anet_v1_p4", attempt=1) -> Path:
    return w.anet_out / (f"{DATE}-ANETRUN-{NR.RUN_TAG[run]}-{STATE['commit']}" + ("" if attempt == 1 else f"-r{attempt}"))


@contextlib.contextmanager
def fresh_out(w: World):
    shutil.rmtree(w.anet_out, ignore_errors=True)
    try:
        yield
    finally:
        shutil.rmtree(w.anet_out, ignore_errors=True)


def flow(w: World, fwd=None, **kw) -> tuple:
    """Un run del flusso finto (anet_v1_p4 salvo run=...), con pad_anet_comparison.forward_checked finto."""
    run = kw.pop("run", "anet_v1_p4")
    argv = kw.pop("argv", None) or w.anet_argv("run", run=run)
    with mock.patch.object(C, "forward_checked", fwd or fake_forward_checked()):
        return run_main(argv, flow_hooks(w, **kw))


def csv_rows(p: Path) -> list:
    return FR.parse_csv(p.read_bytes(), NR.MODEL_COLUMNS, p.name)


# ============================================================================ A configurazione

class TestAConfig(unittest.TestCase):
    def setUp(self):
        self.w = world()

    def refused(self, mutate, needle: str):
        d = self.w.anet_doc()
        mutate(d)
        p = self.w.tmp / "anetrun_mut.json"
        p.write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(FR.Failure, needle):
            NR.load_config(p)

    def test_synthetic_configuration_accepted(self):
        cfg, h = NR.load_config(self.w.anet_cfg)
        self.assertEqual(cfg["runs"], list(NR.RUNS))
        self.assertEqual(h, sha(self.w.anet_cfg.read_bytes()))

    def test_closed_sets(self):
        self.refused(lambda d: d.update(extra=1), "the top level")
        self.refused(lambda d: d.pop("non_evaluable"), "the top level")
        self.refused(lambda d: d["estimator_check"].update(extra=1), "estimator_check")
        self.refused(lambda d: d["non_evaluable"].pop("in_memory_dev"), "non_evaluable")
        self.refused(lambda d: d["choices_of_the_author"].pop("card_check"), "choices_of_the_author")

    def test_values_fixed(self):
        self.refused(lambda d: d.update(runs=list(reversed(d["runs"]))), "runs must be")
        self.refused(lambda d: d["estimator"]["orders"]["AttackNetV1"].reverse(), "estimator.orders")
        self.refused(lambda d: d["estimator_check"].update(models=12), "36 models")
        self.refused(lambda d: d["estimator_check"].update(bnrecal_run_id="x"), "BNRECAL")
        self.refused(lambda d: d["cap"].update(unit="CPU hours"), "cap.unit")
        self.refused(lambda d: d["outputs"]["public"].pop(), "outputs differ")
        self.refused(lambda d: d["machine_rule"]["fields"].pop(), "machine_rule must list")
        self.refused(lambda d: d["trial_record"].update(run_id="20261122-ARCHTRIAL-abcdef0"), "ANETTRIAL")
        self.refused(lambda d: d["code"].pop("scripts/pad_anet_comparison.py"), "code must pin")
        self.refused(lambda d: d.update(schema="x"), "schema")
        self.refused(lambda d: d["dev_evaluation"].update(states=["ordinary"]), "dev_evaluation.states")

    def test_texts_and_choices(self):
        self.refused(lambda d: d.update(sequence=""), "sequence must be a non-empty text")
        self.refused(lambda d: d.update(test="x"), "no test frame")
        self.refused(lambda d: d["choices_of_the_author"].update(in_memory_dev="mine"), "choice of the author")
        self.refused(lambda d: d["non_evaluable"].update(rule=""), "non_evaluable.rule")

    def test_choices_keys_are_those_of_make_config(self):
        self.assertEqual(sorted(MC.doc(data_sha=lambda p: "0" * 64, code_sha=lambda p: "0" * 64)["choices_of_the_author"]), sorted(NR.CHOICE_KEYS))

    def test_delivered_configuration(self):
        """La configurazione consegnata, se c'è con i file che fissa: valida e uguale a quella che make_config rigenera."""
        p = ROOT / NR.DEFAULT_CONFIG
        if not p.is_file():
            skip_missing(self, "delivered configuration")
        d = json.loads(p.read_text(encoding="utf-8"))
        pinned = [d["comparison"]["path"], d["trial_record"]["package_sums"]["path"], *(e["path"] for e in d["plan_files"].values()),
                  d["estimator_check"]["bn_test_config"]["path"], *NR.CODE_FILES]
        if any(not (ROOT / x).is_file() for x in pinned):
            skip_missing(self, "delivered configuration")
        NR.load_config(p)
        self.assertEqual(MC.doc(), d)
        NR.check_code(d)


# ============================================================================ B file fissati

class TestBPins(unittest.TestCase):
    def setUp(self):
        self.w = world()
        self.addCleanup(self.w.write_anet)

    def pins(self, est_base=None):
        cfg, _ = NR.load_config(self.w.anet_cfg)
        with injected(**base_hooks(self.w)):
            return NR.load_pins(cfg, self.w.base, est_base)

    def test_pins_pass(self):
        pins = self.pins()
        self.assertEqual(sorted(pins.positions), sorted(NR.RUNS))
        for r, (net, p) in NR.RUNS.items():
            pos = pins.positions[r]
            self.assertEqual([(x["position"], x["fold"], x["seed"]) for x in pos],
                             [(i + 1, f, s) for i, (f, s) in enumerate((f, s) for f in DEF.FOLDS for s in DEF.PLAN_SEEDS)])
            self.assertTrue(all(x["network"] == net and x["patience"] == p for x in pos))
            self.assertGreater(pins.caps[r], 0)
        self.assertEqual(pins.trial["run_id"], TRIAL_ID)

    def refused(self, needle, **kw):
        self.w.write_anet(**kw)
        with self.assertRaisesRegex(FR.Failure, needle):
            self.pins()

    def test_plan_file_different_from_the_regenerated_stops(self):
        cfg_text = (self.w.pkg / "plan" / "run_caps.csv").read_text()
        self.refused("run_caps.csv differs from the file regenerated", plan_over={"run_caps": cfg_text.replace("2.50", "9.50", 1)
                                                                                   if "2.50" in cfg_text else cfg_text + "x\n"})
        m = (self.w.pkg / "plan" / "run_matrix.csv").read_text()
        lines = m.splitlines(keepends=True)
        self.refused("run_matrix.csv differs", plan_over={"run_matrix": lines[0] + lines[2] + lines[1] + "".join(lines[3:])})

    def test_package_sums_anchor_the_trial_and_the_plan(self):
        self.refused("trial/trial.json of its SHA256SUMS", sums_over={"trial/trial.json": "0" * 64})
        self.refused("plan/run_order.csv of its SHA256SUMS", sums_over={"plan/run_order.csv": "0" * 64})

    def test_trial_of_other_code_or_configuration_refused(self):
        self.refused("produced with another configuration", trial_over={"config_sha256": "0" * 64})
        self.refused("produced with other code", trial_over={"code_sha256": {"architecture": "0" * 64}})

    def test_unofficial_trial_refused(self):
        self.refused("unofficial trial", trial_over={"hooks": {"official_configuration": False, "set": [], "changed": []}})

    def test_pinned_file_changed_stops(self):
        d = self.w.anet_doc()
        d["comparison"]["sha256"] = "0" * 64
        self.w.write_anet_cfg(d)
        with self.assertRaisesRegex(FR.Failure, "differs from the configuration"):
            self.pins()

    def test_pr27_configuration_must_be_that_of_pr_a(self):
        d = self.w.anet_doc()
        alt = self.w.base / "bnrecal_other.json"
        alt.write_text((self.w.base / "bnrecal.json").read_text() + " ", encoding="utf-8")
        d["estimator_check"]["bn_recalibration_config"] = {"path": alt.name, "sha256": sha(alt.read_bytes())}
        self.w.write_anet_cfg(d)
        with self.assertRaisesRegex(FR.Failure, "differs from the one of the PR A configuration"):
            self.pins()

    def test_run_positions_and_caps_read(self):
        m = (self.w.pkg / "plan" / "run_matrix.csv").read_bytes()
        bad = m.replace(b"anet_v1_p4,2,AttackNetV1,1,43", b"anet_v1_p4,2,AttackNetV1,1,44", 1)
        with self.assertRaisesRegex(FR.Failure, "position 2 of anet_v1_p4"):
            NR.run_positions(bad, "anet_v1_p4")
        caps = (self.w.pkg / "plan" / "run_caps.csv").read_bytes()
        self.assertEqual(NR.run_cap_hours(caps, "anet_v2_2_p4"),
                         float([r for r in FR.parse_csv(caps, C.CAPS_COLUMNS, "c") if r["run"] == "anet_v2_2_p4"][0]["cap_wall_hours"]))
        with self.assertRaisesRegex(FR.Failure, "0 rows"):
            NR.run_cap_hours(caps, "nuaa_patience_4")


# ============================================================================ C macchina, check-machine, check-inputs

class TestCMachineAndInputs(unittest.TestCase):
    def setUp(self):
        self.w = world()

    def test_check_machine_pass_nothing_written(self):
        before = CPR.tree(self.w.tmp)
        rc, out, err = run_main(self.w.anet_argv("check-machine"), base_hooks(self.w))
        self.assertEqual(rc, 0, err)
        self.assertIn(f"check-machine: machine and versions equal to the C5 trial {TRIAL_ID} (11 fields)", out)
        self.assertIn("TensorFlow not imported; nothing written", out)
        self.assertTrue(out.rstrip().endswith("RESULT: PASS (check-machine)"))
        self.assertEqual(CPR.tree(self.w.tmp), before)

    def test_different_machine_or_versions_stop(self):
        for part, f in (("MACHINE", "processor"), ("ENVIRONMENT", "keras")):
            with self.subTest(f):
                h = base_hooks(self.w)
                src = FIXED_M if part == "MACHINE" else FIXED_E
                h["r"][part] = lambda src=src, f=f: {**src, f: "another"}
                rc, out, err = run_main(self.w.anet_argv("check-machine"), h)
                self.assertEqual(rc, 1)
                self.assertIn(f"machine rule: {f} differ from the official trial {TRIAL_ID}", err)
                rc, out, err = run_main(self.w.anet_argv("run"), {**h, "n": {**h["n"]}})
                self.assertEqual(rc, 1)
                self.assertFalse(self.w.anet_out.exists())

    def test_thread_variable_set_stops(self):
        with CPR.thread_env(TF_NUM_INTRAOP_THREADS="10"):
            rc, out, err = run_main(self.w.anet_argv("check-machine"), base_hooks(self.w))
        self.assertEqual(rc, 1)
        self.assertIn("thread rule", err)

    def test_check_inputs_pass_nothing_written(self):
        before = CPR.tree(self.w.tmp)
        rc, out, err = run_main(self.w.anet_argv("check-inputs"), base_hooks(self.w))
        self.assertEqual(rc, 0, err)
        self.assertIn("24 reference cards with keys equal to the manifests (24 unit checks)", out)
        self.assertIn("anet_v2_2_p12: 12 positions", out)
        self.assertIn("no frame assembled; TensorFlow not imported; nothing written", out)
        self.assertEqual(CPR.tree(self.w.tmp), before)
        rc, out, err = run_main(self.w.anet_argv("check-inputs", "--estimator-check", str(self.w.est_anet)), base_hooks(self.w))
        self.assertEqual(rc, 0, err)
        self.assertIn(f"estimator check {self.w.est_anet.name} PASS (36 models, not official)", out)

    def test_estimator_record_checked(self):
        w = self.w
        self.addCleanup(w.write_anet_estimator)
        for kw, needle in (({"status": "FAIL"}, "not a PASS"), ({"over": {"config_sha256": "0" * 64}}, "another configuration"),
                           ({"over": {"code_sha256": {}}}, "other code"), ({"over": {"models": [{"recalibration_pass": False}] * 36}}, "not every model"),
                           ({"over": {"schema": "pad-arch-estimator-check/1"}}, "schema"), ({"over": {"machine_rule": {"equal": False}}}, "machine and thread")):
            with self.subTest(needle):
                w.write_anet_estimator(**kw)
                rc, out, err = run_main(w.anet_argv("check-inputs", "--estimator-check", str(w.est_anet)), base_hooks(w))
                self.assertEqual(rc, 1)
                self.assertIn(needle, err)


# ============================================================================ D agganci e configurazione ufficiale

def module_hooks(path: Path) -> list:
    """Le variabili di modulo assegnate a None nel sorgente (gli agganci), lette con ast."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sorted(t.id for node in tree.body if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and node.value.value is None
                  for t in node.targets if isinstance(t, ast.Name))


class TestDHooks(unittest.TestCase):
    def test_hook_registry_of_this_module_is_complete(self):
        self.assertEqual(module_hooks(ROOT / NR.THIS_FILE), sorted(NR.MY_HOOKS))
        self.assertEqual({k: v for k, v in NR.HOOKS.items() if k != "pad_anet_run"}, C.HOOKS)
        self.assertEqual(NR.HOOKS["pad_anet_run"], NR.MY_HOOKS)

    def test_every_hook_stops_the_official_commands(self):
        for m, names in NR.HOOKS.items():
            mod = NR if m == "pad_anet_run" else sys.modules[m]
            for k in names:
                with self.subTest(f"{m}.{k}"):
                    saved = getattr(mod, k)
                    setattr(mod, k, lambda *a: None)
                    try:
                        with self.assertRaisesRegex(FR.Failure, f"{m}.{k}"):
                            NR.check_hooks(True)
                        self.assertEqual(NR.check_hooks(False)["set"], [f"{m}.{k}"])
                    finally:
                        setattr(mod, k, saved)

    def test_official_out_root_checks_the_hooks(self):
        """Con la radice ufficiale (results/anetrun) un aggancio impostato ferma il comando prima di ogni cartella."""
        w = world()
        h = flow_hooks(w)
        argv = w.anet_argv("run", unofficial=False, out=ROOT / NR.DEFAULT_OUT_ROOT)
        i = argv.index("--date")
        rc, out, err = run_main(argv[:i] + argv[i + 2:], h)
        self.assertEqual(rc, 1)
        self.assertIn("test hooks are set", err)
        self.assertFalse((ROOT / NR.DEFAULT_OUT_ROOT).exists())

    def test_official_configuration_by_identity(self):
        w = world()
        with injected(n={"DEFAULT_CONFIG": os.path.relpath(w.anet_cfg, ROOT)}):
            self.assertTrue(NR.is_official_config(w.anet_cfg))
            copy = w.tmp / "anetrun_copy.json"
            shutil.copy2(w.anet_cfg, copy)
            self.assertFalse(NR.is_official_config(copy))


# ============================================================================ E copie dichiarate

def body(fn) -> str:
    return inspect.getsource(fn)


def code_without_docstring(src: str) -> str:
    """Il codice di una funzione senza la docstring e i commenti (le docstring delle copie sono diverse per
    costruzione), normalizzato da ast: il confronto è sulle istruzioni."""
    fn = ast.parse(src).body[0]
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant):
        fn.body = fn.body[1:]
    fn.name = "f"
    return ast.unparse(fn)


# Le differenze dichiarate fra le copie e le funzioni di pad_arch_run (C2), sul codice normalizzato da ast (senza
# docstring e commenti), nell'ordine: (righe dell'originale, righe della copia). Sostituzioni: trainer della prova e
# costruttore del registro, regola (n) (storia, pesi, passate in memoria, modello non valutabile), nessun controllo della
# storia di LivenessNet, configurazione della PR A, controllo dei frame per fold, uscite con i non valutabili, stato
# dei run, radice e agganci di questo runner. Una differenza non elencata qui fa fallire il test.
COPY_HUNKS = {'cmd_run': [('        fail("run trains 24 networks on the restricted NUAA frames: pass --i-am-authorized (only after the coordinator\'s verification, merge '
              'and go-ahead)")',
              "        fail('run trains 12 networks on the restricted NUAA frames: pass --i-am-authorized (only after the publication of this command and "
              "configuration and the link on the C5 card)')"),
             ("    base, cfg_p, cfg, cfg_sha, pins, env, mrule, state, rid, run, est_folder, official, attempt, command = prepare(a, 'run')\n"
              '    patience = RUNS[a.run]',
              "    base, cfg_p, cfg, cfg_sha, pins, env, mrule, state, rid, run, est_folder, official, attempt, command, hooks = prepare(a, 'run')\n"
              '    net, patience = RUNS[a.run]'),
             ("    head = {'run_id': rid, 'run': a.run, 'commit': state['commit'], 'dirty': False, 'config_sha256': cfg_sha, 'command': 'run', 'command_line': "
              "command, 'official': official, 'attempt': attempt}",
              "    head = {'run_id': rid, 'run': a.run, 'network': net, 'commit': state['commit'], 'dirty': False, 'config_sha256': cfg_sha, 'command': 'run', "
              "'command_line': command, 'official': official, 'attempt': attempt, 'hooks': hooks}"),
             ("        write_status(run, {**head, 'status': 'RUNNING', 'passed': False, 'started': started, 'pid': os.getpid(), **extra, 'note': 'RUNNING "
              "without a live process means a failed run; SIGUSR1 to this pid stops it as INCOMPLETE'}, scan)",
              "        R.write_status(run, {**head, 'status': 'RUNNING', 'passed': False, 'started': started, 'pid': os.getpid(), **extra, 'note': 'RUNNING "
              "without a live process means a failed run; SIGUSR1 to this pid stops it as INCOMPLETE'}, scan)"),
             ('        inp = PAC.load_inputs(pins.cfg_pr, pins.pcfg, base)',
              '        inp = C.load_inputs(pins.cfg_c5, pins.cfg_pr, pins.pcfg, base, frames=True)'),
             ('        refs = load_reference_cards(pins.cfg_pr, base, patience)\n        check_unit_keys(inp, refs, pins.pcfg)',
              '        refs = R.load_reference_cards(pins.cfg_pr, base, patience)\n        R.check_unit_keys(inp, refs, pins.pcfg)'),
             ('        tf = tf_module()', '        tf = R.tf_module()'),
             ("        extra['threads_rule'] = threads_rule(pins.trial, threads)\n        require_threads(extra['threads_rule'])",
              "        extra['threads_rule'] = R.threads_rule(pins.trial, threads)\n        R.require_threads(extra['threads_rule'])"),
             ('        ctx.cfg, ctx.cfg_pr, ctx.pcfg, ctx.bn_cfg, ctx.inp, ctx.store, ctx.budget, ctx.tf, ctx.refs = (cfg, pins.cfg_pr, pins.pcfg, '
              'pins.bn_cfg, inp, store, budget, tf, refs)',
              '        ctx.cfg, ctx.cfg_c5, ctx.cfg_pr, ctx.pcfg, ctx.bn_cfg, ctx.inp, ctx.store, ctx.budget, ctx.tf, ctx.refs = (cfg, pins.cfg_c5, '
              'pins.cfg_pr, pins.pcfg, pins.bn_cfg, inp, store, budget, tf, refs)'),
             ("        ctx.arch = (ARCH_LOADER or FR.default_arch_loader)(rel_path(base, pins.cfg_pr['architectures']['LivenessNet']['path']))",
              "        ctx.arch = (ARCH_LOADER or FR.default_arch_loader)(rel_path(base, pins.cfg_c5['architectures']['path']))"),
             ('                ctx.current = rec\n                phase_two(ctx, rec)',
              "                if rec.get('evaluable'):\n                    ctx.current = rec\n                    phase_two(ctx, rec)"),
             ("                discard_model(ctx.rdir, ctx.current['name'])\n                mark_discarded(ctx.current)",
              "                R.discard_model(ctx.rdir, ctx.current['name'])\n                R.mark_discarded(ctx.current)"),
             ('        res_sha = write_restricted_sums(ctx.rdir)',
              "        ne = [{'name': r['name'], 'reason': r['reason']} for r in recs if r.get('evaluable') is False]\n"
              '        res_sha = R.write_restricted_sums(ctx.rdir)'),
             ("            doc = {'schema': RUN_SCHEMA, **head, 'status': status, 'patience': patience, 'started': started, 'finished': finished, "
              "'wall_seconds': round(wall, 1), 'process_cpu_seconds': cpu_total, 'cap_hours': cap_h, 'cap_unit': CAP_UNIT, 'stopped': stopped, 'positions': "
              "len(recs), 'complete_phase_1': len(done1), 'complete_phase_2': len(done2), 'machine': (MACHINE or PAC.machine)(), 'environment': (ENVIRONMENT "
              "or FR.environment)(), 'machine_rule': mrule, 'threads': threads, 'threads_rule': extra['threads_rule'], 'estimator_check': pins.estimator, "
              "'training': ctx.training, 'sequence': cfg['sequence'], 'after_stop': cfg['after_stop'], 'models': [public_model(r) for r in recs], 'one_class': "
              "one_class_rows(done2, patience), 'dev_frames': frames, 'checks': checks, 'code_sha256': pins.code, 'inputs_sha256': {**pins.files, "
              "'configuration': cfg_sha}, 'peak_rss_mb': round(SR.peak_rss_mb(), 1), 'test_read': TEST_READ, 'command': command}\n"
              "            texts = {'models.csv': csv_text(MODEL_COLUMNS, [model_row(r, a.run, patience) for r in recs]), 'one_class.csv': "
              "csv_text(ONE_CLASS_COLUMNS, [[cell(o[k]) for k in ONE_CLASS_COLUMNS] for o in doc['one_class']]), 'summary.json': PAC.json_text(doc, "
              "'summary.json'), 'report.md': report_md(doc)}\n"
              '            pub_sha = write_public(run, inp, texts)\n'
              "        print(f'run folder: {run.name}, elapsed {wall:.0f} s ({wall / 3600:.2f} h of the {cap_h:.2f}-hour cap), process CPU {cpu_total:.0f} s')",
              "            doc = {'schema': RUN_SCHEMA, **head, 'status': status, 'patience': patience, 'started': started, 'finished': finished, "
              "'wall_seconds': round(wall, 1), 'process_cpu_seconds': cpu_total, 'cap_hours': cap_h, 'cap_unit': CAP_UNIT, 'stopped': stopped, 'positions': "
              "len(recs), 'complete_phase_1': len(done1), 'complete_phase_2': len(done2), 'not_evaluable': ne, 'machine': (R.MACHINE or PAC.machine)(), "
              "'environment': (R.ENVIRONMENT or FR.environment)(), 'machine_rule': mrule, 'threads': threads, 'threads_rule': extra['threads_rule'], "
              "'estimator_check': pins.estimator, 'training': ctx.training, 'sequence': cfg['sequence'], 'non_evaluable': cfg['non_evaluable'], 'after_stop': "
              "cfg['after_stop'], 'models': [public_model(r) for r in recs], 'one_class': one_class_rows(done2, patience), 'dev_frames': frames, 'checks': "
              "checks, 'code_sha256': pins.code, 'inputs_sha256': {**pins.files, 'configuration': cfg_sha}, 'peak_rss_mb': round(SR.peak_rss_mb(), 1), "
              "'test_read': TEST_READ, 'command': command}\n"
              "            texts = {'models.csv': csv_text(MODEL_COLUMNS, [model_row(r, a.run, patience) for r in recs]), 'one_class.csv': "
              "csv_text(ONE_CLASS_COLUMNS, [[R.cell(o[k]) for k in ONE_CLASS_COLUMNS] for o in doc['one_class']]), 'summary.json': PAC.json_text(doc, "
              "'summary.json'), 'report.md': report_md(doc)}\n"
              '            pub_sha = R.write_public(run, inp, texts)\n'
              "        print(f'run folder: {run.name}, wall {wall:.0f} s ({wall / 3600:.2f} h of the {cap_h:.2f}-hour cap), process CPU {cpu_total:.0f} s; not "
              "evaluable {len(ne)}')"),
             ('',
              '        sys.stdout.flush()\n'
              '        sys.stderr.flush()\n'
              '        log[0].flush()\n'
              "        if (probs := PAC.output_problems((run / LOG).read_text(encoding='utf-8', errors='replace'), inp)):\n"
              "            fail(f'{LOG} holds text that a publishable output must not hold: ' + '; '.join(probs))"),
             ("        if (probs := PAC.output_problems((run / LOG).read_text(encoding='utf-8', errors='replace'), inp)):\n"
              "            fail(f'{LOG} holds text that a publishable output must not hold: ' + '; '.join(probs))\n"
              "        write_status(run, {**head, 'status': status, 'passed': status == 'PASS', 'started': started, 'finished': finished, 'wall_seconds': "
              "round(wall, 1), 'process_cpu_seconds': cpu_total, **extra, 'stopped': stopped, 'complete_phase_1': [r['name'] for r in done1], "
              "'complete_phase_2': [r['name'] for r in done2], 'missing_phase_1': [r['name'] for r in recs if not r.get('phase_1')], 'evaluator_mismatches': "
              "mismatches, 'dev_frames': frames, 'machine': (MACHINE or PAC.machine)(), 'code_sha256': pins.code, 'inputs_sha256': {**pins.files, "
              "'configuration': cfg_sha}, 'public_sha256sums_sha256': pub_sha, 'restricted_sha256sums_sha256': res_sha, 'run_log_sha256': sha256_file(run / "
              'LOG)}, scan)',
              "        R.write_status(run, {**head, 'status': status, 'passed': status == 'PASS', 'started': started, 'finished': finished, 'wall_seconds': "
              "round(wall, 1), 'process_cpu_seconds': cpu_total, **extra, 'stopped': stopped, 'complete_phase_1': [r['name'] for r in done1], "
              "'complete_phase_2': [r['name'] for r in done2], 'missing_phase_1': [r['name'] for r in recs if not r.get('phase_1')], 'not_evaluable': ne, "
              "'evaluator_mismatches': mismatches, 'dev_frames': frames, 'machine': (R.MACHINE or PAC.machine)(), 'code_sha256': pins.code, 'inputs_sha256': "
              "{**pins.files, 'configuration': cfg_sha}, 'public_sha256sums_sha256': pub_sha, 'restricted_sha256sums_sha256': res_sha, 'run_log_sha256': "
              'sha256_file(run / LOG)}, scan)'),
             ("            discard_model(run / RESTRICTED, cur['name'])", "            R.discard_model(run / RESTRICTED, cur['name'])"),
             ("        extra['restricted'] = restricted_note(run / RESTRICTED)\n"
              '        fail_cleanup(run, list(PUBLIC_OUTPUTS), log, head, started, budget, c0, extra, e, scan)',
              "        extra['not_evaluable'] = [{'name': r['name'], 'reason': r['reason']} for r in recs if r.get('evaluable') is False]\n"
              "        extra['restricted'] = R.restricted_note(run / RESTRICTED)\n"
              '        R.fail_cleanup(run, list(PUBLIC_OUTPUTS), log, head, started, budget, c0, extra, e, scan)')],
 'phase_one': [("    fit, dev = (unit_frames(ctx.store, ctx.inp, 'fitting', fold), unit_frames(ctx.store, ctx.inp, 'dev', fold))",
                "    fit, dev = (R.unit_frames(ctx.store, ctx.inp, 'fitting', fold), R.unit_frames(ctx.store, ctx.inp, 'dev', fold))"),
               ("    card, model = (TRAINER or PAC.trial_trainer)({'X_fit': fit['X'], 'y_fit': fit['y'], 'X_dev': dev['X'], 'y_dev': dev['y']}, ctx.training, "
                "net, seed, lambda: PAC.build_checked(ctx.cfg_pr, net, ctx.arch), lambda step: ctx.budget.check(f'{step} of {name}'), timer)",
                "    card, model = (TRAINER or C.anet_trainer)({'X_fit': fit['X'], 'y_fit': fit['y'], 'X_dev': dev['X'], 'y_dev': dev['y']}, ctx.training, "
                "net, seed, lambda: C.build_checked(ctx.cfg_c5, net, ctx.arch), lambda step: ctx.budget.check(f'{step} of {name}'), timer)"),
               ("    if not same_history_records(epochs, card['history'], card['epochs_run']) or len(epochs) != card['epochs_run']:",
                "    if not history_matches(epochs, card['history'], card['epochs_run']):"),
               ("    if (bad := non_finite_history(card['history'])):\n"
                '        fail(f"{name}: {\'; \'.join(bad)}: the run ends as FAIL (a card with a non-finite value cannot be written)")',
                "    weights_finite = DEF.arrays_finite(model.get_weights()) if card['restored_epoch'] is not None else None\n"
                "    outcome = DEF.training_outcome(card['history'], weights_finite)\n"
                "    rec['outcome'] = outcome\n"
                "    rec['card'] = {k: card[k] for k in ('epochs_run', 'restored_epoch', 'restored_by', 'best_val_loss', 'non_finite')}\n"
                "    base_card = {**card, 'test_evaluation': 'none (no test frame is assembled in this run)', 'network': net, 'arm': ARM, 'fold': fold, 'run': "
                "ctx.run_name, 'position': pos['position'], 'fitting_keys_sha256': BR.keys_sha(fit['rows']), 'dev_keys_sha256': BR.keys_sha(dev['rows']), "
                "'epoch_timing': [{k: DEF.finite_or_none(v) if k in HISTORY_KEYS else v for k, v in e.items()} for e in epochs], 'training_overhead': "
                "{'seconds': rec['train_seconds'] - sum((e['seconds_total'] for e in epochs)), 'cpu_seconds': rec['train_cpu_seconds'] - sum((e['cpu_seconds'] "
                "for e in epochs))}, 'rule_n': outcome}\n"
                "    reason = None if outcome['evaluable'] else outcome['reason']\n"
                '    if reason is None:\n'
                '        try:\n'
                "            for role, part in (('dev', dev), ('fitting', fit)):\n"
                "                ctx.budget.check(f'in-memory {role} scoring {name}')\n"
                "                C.forward_checked(model, part, ctx.cfg['estimator']['batch'], f'in-memory {role}', 'ordinary')\n"
                '        except C.NotEvaluable as e:\n'
                '            reason = str(e)\n'
                '    if reason is not None:\n'
                '        del model\n'
                '        gc.collect()\n'
                "        card = {**base_card, 'rule_n': {**outcome, 'evaluable': False, 'reason': reason}, 'not_evaluable_reason': reason, 'checkpoint': "
                "NOT_EVALUABLE_CHECKPOINT, 'weights_file': None, 'weights_sha256': None, 'saved': None}\n"
                "        rec['weights_sha256'] = None\n"
                "        rec['threads_after'] = PAC.check_threads(ctx.tf, ctx.cfg_pr, f'after {name}')\n"
                "        mw.write_all(ctx.rdir, {f'{name}_card.json': PAC.json_text(card, f'{name}_card.json')})\n"
                '        not_evaluable(ctx, rec, reason)\n'
                "        rec['phase_1'] = True\n"
                "        rec['phase_1_seconds'] = time.monotonic() - t_phase\n"
                '        return'),
               ('    (SAVE_MODEL or save_checkpoint)(model, weights)', '    (SAVE_MODEL or R.save_checkpoint)(model, weights)'),
               ("    card = {**card, 'checkpoint': CHECKPOINT_TEXT, 'test_evaluation': 'none (no test frame is assembled in this run)', 'weights_file': "
                "weights.name, 'weights_sha256': sha256_file(weights), 'saved': now(), 'network': net, 'arm': ARM, 'fold': fold, 'run': ctx.run_name, "
                "'position': pos['position'], 'fitting_keys_sha256': BR.keys_sha(fit['rows']), 'dev_keys_sha256': BR.keys_sha(dev['rows']), 'epoch_timing': "
                "epochs, 'training_overhead': {'seconds': rec['train_seconds'] - sum((e['seconds_total'] for e in epochs)), 'cpu_seconds': "
                "rec['train_cpu_seconds'] - sum((e['cpu_seconds'] for e in epochs))}}",
                "    card = {**base_card, 'checkpoint': CHECKPOINT_TEXT, 'weights_file': weights.name, 'weights_sha256': sha256_file(weights), 'saved': "
                'now()}'),
               ("    rec['card'] = {k: card[k] for k in ('epochs_run', 'restored_epoch', 'restored_by', 'best_val_loss')}\n"
                "    if net == 'LivenessNet':\n"
                "        rec['history_check'] = history_check(ref, card)\n"
                "        card['history_check'] = rec['history_check']\n"
                '        print(f"   {name}: history against {ref[\'model\']} of the accepted run: {rec[\'history_check\'][\'outcome\']}" + (\'\' if '
                'rec[\'history_check\'][\'first_differing_epoch\'] is None else f" (first differing epoch '
                '{rec[\'history_check\'][\'first_differing_epoch\']})"), flush=True)',
                ''),
               ('        fail(f"{name}: val_loss recomputed on the dev {vl:.6f}, recorded minimum {card[\'best_val_loss\']:.6f}: the checkpoint is not the one '
                'of minimum val_loss")',
                '        fail(f"{name}: val_loss recomputed on the dev {vl:.6f}, recorded minimum {card[\'best_val_loss\']:.6f}: the checkpoint is not the one '
                'of minimum finite val_loss")'),
               ('', "    rec['evaluable'] = True"),
               ("    mw.write_all(ctx.rdir, {f'{name}_dev_scorer.csv': csv_text(SCORER_COLUMNS, scorer_rows(rec, dev, p[:, 1], z))}, newline='')",
                "    mw.write_all(ctx.rdir, {f'{name}_dev_scorer.csv': csv_text(R.SCORER_COLUMNS, R.scorer_rows(rec, dev, p[:, 1], z))}, newline='')"),
               ('    print(f"== {name}: {card[\'epochs_run\']} epochs, restored epoch {card[\'restored_epoch\']}, val_loss {vl:.6f} recomputed on the dev", '
                'flush=True)',
                "    nf = '; non-finite ' + ', '.join((f'{k} from epoch {e}' for k, e in card['non_finite'])) if card['non_finite'] else ''\n"
                '    print(f"== {name}: {card[\'epochs_run\']} epochs, restored epoch {card[\'restored_epoch\']}, val_loss {vl:.6f} recomputed on the '
                'dev{nf}", flush=True)')],
 'phase_two': [("    fit, dev = (unit_frames(ctx.store, ctx.inp, 'fitting', rec['fold']), unit_frames(ctx.store, ctx.inp, 'dev', rec['fold']))\n"
                '    res = (REESTIMATE or reestimate_model)(ctx, rec, fit, dev)\n'
                "    summ_o, unit_o = dev_state(dev, res['ordinary']['p'], res['ordinary']['z'], 'ordinary', name)\n"
                "    agreement = evaluator_agreement(rec['scorer_p'], res['ordinary']['p'], summ_o['threshold_eer'], ctx.bn_cfg)\n"
                "    summ_r, unit_r = dev_state(dev, res['re_estimated']['p'], res['re_estimated']['z'], 're-estimated', name)",
                "    fit, dev = (R.unit_frames(ctx.store, ctx.inp, 'fitting', rec['fold']), R.unit_frames(ctx.store, ctx.inp, 'dev', rec['fold']))\n"
                '    try:\n'
                '        res = (REESTIMATE or reestimate_model)(ctx, rec, fit, dev)\n'
                '    except C.NotEvaluable as e:\n'
                '        not_evaluable(ctx, rec, str(e))\n'
                "        rec['phase_2_seconds'] = time.monotonic() - t0\n"
                '        return\n'
                "    summ_o, unit_o = R.dev_state(dev, res['ordinary']['p'], res['ordinary']['z'], 'ordinary', name)\n"
                "    agreement = R.evaluator_agreement(rec['scorer_p'], res['ordinary']['p'], summ_o['threshold_eer'], ctx.bn_cfg)\n"
                "    summ_r, unit_r = R.dev_state(dev, res['re_estimated']['p'], res['re_estimated']['z'], 're-estimated', name)"),
               ('    doc = stats_doc(rec, res, summ_r, order, ctx.cfg)', '    doc = R.stats_doc(rec, res, summ_r, order, ctx.cfg)'),
               ("    mw.write_all(ctx.rdir, {f'{name}_dev_ordinary.csv': csv_text(SCORE_COLUMNS, score_rows(rec, 'ordinary', dev, res['ordinary']['p'], "
                "res['ordinary']['z'], summ_o)), f'{name}_dev_re-estimated.csv': csv_text(SCORE_COLUMNS, score_rows(rec, 're-estimated', dev, "
                "res['re_estimated']['p'], res['re_estimated']['z'], summ_r))}, newline='')",
                "    mw.write_all(ctx.rdir, {f'{name}_dev_ordinary.csv': csv_text(R.SCORE_COLUMNS, R.score_rows(rec, 'ordinary', dev, res['ordinary']['p'], "
                "res['ordinary']['z'], summ_o)), f'{name}_dev_re-estimated.csv': csv_text(R.SCORE_COLUMNS, R.score_rows(rec, 're-estimated', dev, "
                "res['re_estimated']['p'], res['re_estimated']['z'], summ_r))}, newline='')"),
               ("    rec['dev_units'] = {'ordinary': unit_o, 're-estimated': unit_r}", ''),
               ('    print(f"== {name}: dev EER threshold ordinary {fmt_thr(summ_o)}, re-estimated {fmt_thr(summ_r)}; evaluator agreement {(\'yes\' if '
                'agreement[\'pass\'] else \'NO\')}", flush=True)',
                '    print(f"== {name}: dev EER threshold ordinary {R.fmt_thr(summ_o)}, re-estimated {R.fmt_thr(summ_r)}; evaluator agreement {(\'yes\' if '
                'agreement[\'pass\'] else \'NO\')}", flush=True)')]}


def hunks_of(orig: str, mine: str) -> list:
    import difflib
    a, b = orig.split("\n"), mine.split("\n")
    return [("\n".join(a[i1:i2]), "\n".join(b[j1:j2])) for t, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
            if t != "equal"]


class TestECopies(unittest.TestCase):
    def test_phase_one_phase_two_cmd_run_differ_from_c2_only_as_declared(self):
        for name, declared in COPY_HUNKS.items():
            with self.subTest(name):
                self.assertEqual(hunks_of(code_without_docstring(body(getattr(R, name))), code_without_docstring(body(getattr(NR, name)))), declared)

    def test_the_caught_message_occurs_once_in_the_frozen_source(self):
        """Il messaggio di nuaa_bn_recalibration che la fase 2 traduce in «non valutabile»: una volta nel sorgente congelato e
        una volta in questo runner (un cambio di testo sarebbe visto qui)."""
        text = "the re-estimated mean or variance is not finite"
        self.assertEqual((ROOT / "scripts" / "nuaa_bn_recalibration.py").read_text(encoding="utf-8").count(text), 1)
        self.assertEqual((ROOT / NR.THIS_FILE).read_text(encoding="utf-8").count(text), 1)

    def check(self, mine, orig, subs, name_subs=()):
        src = body(orig)
        for a, b in subs:
            self.assertEqual(src.count(a), 1, a)
            src = src.replace(a, b)
        self.assertEqual(code_without_docstring(body(mine)), code_without_docstring(src))

    def test_check_estimator_is_the_c2_command_with_declared_substitutions(self):
        subs = [("base, cfg_p, cfg, cfg_sha, pins, env, mrule, state, rid, run, _, official, attempt, command = prepare(a, \"estimator\")",
                 "base, cfg_p, cfg, cfg_sha, pins, env, mrule, state, rid, run, _, official, attempt, command, hooks = prepare(a, \"estimator\")"),
                ('"command_line": command, "official": official, "attempt": attempt}', '"command_line": command, "official": official, "attempt": attempt, "hooks": hooks}'),
                ("        write_status(run, {**head, \"status\": \"RUNNING\"", "        R.write_status(run, {**head, \"status\": \"RUNNING\""),
                ("        tf = tf_module()", "        tf = R.tf_module()"),
                ("        extra[\"threads_rule\"] = threads_rule(", "        extra[\"threads_rule\"] = R.threads_rule("),
                ("        require_threads(", "        R.require_threads("),
                ("                stats = reestimate(", "                stats = R.reestimate("),
                ("                cons = pr27_checks(", "                cons = R.pr27_checks("),
                ("        status = estimator_status(", "        status = R.estimator_status("),
                ('"machine": (MACHINE or PAC.machine)(), "environment": (ENVIRONMENT or FR.environment)(),',
                 '"machine": (R.MACHINE or PAC.machine)(), "environment": (R.ENVIRONMENT or FR.environment)(),'),
                ("        write_status(run, {**head, \"status\": status,", "        R.write_status(run, {**head, \"status\": status,"),
                ("        fail_cleanup(", "        R.fail_cleanup(")]
        self.check(NR.cmd_check_estimator, R.cmd_check_estimator, subs)

    def test_reestimate_model_is_the_c2_function_with_declared_substitutions(self):
        subs = [("    check_loaded(orig, ctx.cfg_pr, net)", "    C.check_loaded(orig, ctx.cfg_c5, net)"),
                ('    ev_ord, _ = BR.evaluate(orig, dev, batch, f"{name} dev ordinary")', '    ev_ord, _ = C.forward_checked(orig, dev, batch, "dev", "ordinary")'),
                ("    stats = reestimate(copy, fit, order, batch, ctx.budget, name)\n",
                 "    try:\n        stats = R.reestimate(copy, fit, order, batch, ctx.budget, name)\n    except Failure as e:\n"
                 "        if str(e).endswith(\"the re-estimated mean or variance is not finite\"):\n"
                 "            raise C.NotEvaluable(f\"{NOT_EVALUABLE}: non-finite re-estimated statistics\") from None\n        raise\n"),
                ('    fit_after, final = BR.evaluate(copy, fit, batch, f"{name} fitting after", layers=order)',
                 '    fit_after, final = C.forward_checked(copy, fit, batch, "fitting", "re-estimated", layers=order)'),
                ('    dev_after, _ = BR.evaluate(copy, dev, batch, f"{name} dev after")', '    dev_after, _ = C.forward_checked(copy, dev, batch, "dev", "re-estimated")'),
                ("    cons = pr27_checks(", "    cons = R.pr27_checks(")]
        self.check(NR.reestimate_model, R.reestimate_model, subs)

    def test_card_problems_equal_to_c2_on_finite_histories(self):
        w = world()
        log = []
        tr = fake_trainer(log)
        tr_data = {"X_fit": np.zeros((4, 1)), "y_fit": np.array([0, 1, 0, 1]), "X_dev": np.zeros((2, 1)), "y_dev": np.array([0, 1])}
        training = {"max_epochs": 12, "initial_learning_rate": 0.001, "batch": 16, "early_stopping_patience": 4, "scheduler": None}
        card, _ = tr(tr_data, training, "AttackNetV1", 42, None, lambda s: None, PAC.EpochTimer([]))
        p = w.tmp / "AttackNetV1_fold1_seed42.keras"
        p.write_bytes(b"x")
        card = {**card, "weights_file": p.name, "weights_sha256": sha(b"x")}
        self.assertEqual(NR.card_problems(card, training, 42, p, "AttackNetV1"), R.card_problems(card, training, 42, p, "AttackNetV1"))
        # storia con NaN: la val_loss finita minima è attesa come epoca ripristinata; per C2 sarebbe un difetto
        h = dict(card["history"])
        h["val_loss"] = [None] + h["val_loss"][1:]
        bad = {**card, "history": h}
        self.assertEqual(NR.card_problems(bad, training, 42, p, "AttackNetV1"), [])
        h2 = dict(h)
        h2["val_loss"] = list(h["val_loss"])
        h2["val_loss"][-1] = None
        wrong = {**card, "history": h2}
        self.assertIn("restored epoch is not the epoch of minimum val_loss", NR.card_problems(wrong, training, 42, p, "AttackNetV1"))


# ============================================================================ F flusso di un run (senza TensorFlow)

class TestFRunFlow(unittest.TestCase):
    def setUp(self):
        self.w = world()
        shutil.rmtree(self.w.anet_out, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.w.anet_out, True)

    def test_pass_outputs(self):
        log = []
        rc, out, err = flow(self.w, log=log)
        self.assertEqual(rc, 0, err[-3000:])
        run = run_folder(self.w)
        self.assertEqual(sorted(os.listdir(run)), ["public", "restricted", "run.log", "status.json"])
        self.assertEqual(sorted(os.listdir(run / "public")), sorted(NR.PUBLIC_OUTPUTS))
        restricted = sorted(os.listdir(run / "restricted"))
        self.assertEqual(len(restricted), 12 * len(R.PER_MODEL) + 1)
        st = json.loads((run / "status.json").read_text())
        self.assertEqual((st["status"], st["network"], st["not_evaluable"], st["official"]), ("PASS", "AttackNetV1", [], False))
        self.assertEqual(st["hooks"]["official_configuration"], False)
        self.assertEqual(len(st["complete_phase_2"]), 12)
        trains = [x for x in log if x[0] == "train"]
        self.assertEqual([(x[1], x[2]) for x in trains], [("AttackNetV1", s) for f in DEF.FOLDS for s in DEF.PLAN_SEEDS])
        rows = csv_rows(run / "public" / "models.csv")
        self.assertEqual([r["model_status"] for r in rows], ["complete"] * 12)
        self.assertEqual({r["evaluable"] for r in rows}, {"1"})
        logt = (run / "run.log").read_text()
        self.assertEqual(logt.count("RESULT:"), 1)
        self.assertTrue(logt.rstrip().endswith("RESULT: PASS (run anet_v1_p4)"))
        summ = json.loads((run / "public" / "summary.json").read_text())
        self.assertEqual(summ["dev_frames"]["folds_compared"], 4)
        self.assertEqual(len(summ["one_class"]), 2)
        for f in os.listdir(run / "public"):
            self.assertEqual(PAC.output_problems((run / "public" / f).read_text(), FR.Inputs()), [])

    def test_no_finite_val_loss_is_not_evaluable_and_the_run_goes_on(self):
        def hist(net, seed, h):
            return {**h, "val_loss": [math.nan] * EPOCHS} if seed == 43 else h
        log = []
        rc, out, err = flow(self.w, log=log, trainer=fake_trainer(log, history_over=hist))
        self.assertEqual(rc, 0, err[-3000:])
        run = run_folder(self.w)
        st = json.loads((run / "status.json").read_text())
        self.assertEqual([x["name"] for x in st["not_evaluable"]], [f"AttackNetV1_fold{f}_seed43" for f in DEF.FOLDS])
        self.assertEqual({x["reason"] for x in st["not_evaluable"]}, {"not evaluable: no finite val_loss"})
        files = os.listdir(run / "restricted")
        self.assertIn("AttackNetV1_fold1_seed43_card.json", files)
        self.assertNotIn("AttackNetV1_fold1_seed43.keras", files)
        self.assertNotIn(("re-estimate", "AttackNetV1_fold1_seed43"), log)
        rows = {r["seed"] + r["fold"]: r for r in csv_rows(run / "public" / "models.csv")}
        self.assertEqual(rows["431"]["model_status"], "not evaluable")
        self.assertEqual(rows["431"]["not_evaluable_reason"], "not evaluable: no finite val_loss")
        card = json.loads((run / "restricted" / "AttackNetV1_fold1_seed43_card.json").read_text())
        self.assertIsNone(card["weights_file"])
        self.assertEqual(card["not_evaluable_reason"], "not evaluable: no finite val_loss")
        self.assertEqual(len(st["complete_phase_1"]), 12)
        self.assertEqual(rows["431"]["phase_1"], "1")
        summ = json.loads((run / "public" / "summary.json").read_text())
        ne = [m for m in summ["models"] if m["name"] == "AttackNetV1_fold1_seed43"][0]
        self.assertIsNotNone(ne["threads_after"])
        self.assertEqual(card["rule_n"]["evaluable"], False)

    def test_partly_non_finite_history_restores_the_finite_minimum(self):
        """Valori non finiti alla prima epoca (loss NaN, val_loss infinito): il run continua, l'epoca ripristinata è quella
        della val_loss finita minima e i valori non finiti sono registrati (grandezza, prima epoca)."""
        def first(net, seed, h):
            if seed != 44:
                return h
            return {**h, "val_loss": [math.inf] + h["val_loss"][1:], "loss": [math.nan] + h["loss"][1:]}
        log = []
        rc, out, err = flow(self.w, log=log, trainer=fake_trainer(log, history_over=first))
        self.assertEqual(rc, 0, err[-3000:])
        run = run_folder(self.w)
        rows = {r["seed"] + r["fold"]: r for r in csv_rows(run / "public" / "models.csv")}
        self.assertEqual((rows["441"]["model_status"], rows["441"]["restored_epoch"]), ("complete", str(EPOCHS)))
        self.assertEqual(rows["441"]["history_non_finite"], "loss from epoch 1; val_loss from epoch 1")
        self.assertEqual(rows["421"]["history_non_finite"], "NA")
        card = json.loads((run / "restricted" / "AttackNetV1_fold1_seed44_card.json").read_text())
        self.assertIsNone(card["history"]["val_loss"][0])
        shutil.rmtree(self.w.anet_out)

        def hist(net, seed, h):
            if seed != 44:
                return h
            vl = list(h["val_loss"])
            vl[-1] = math.inf
            return {**h, "val_loss": vl, "loss": [math.nan] + h["loss"][1:]}
        log = []
        rc, out, err = flow(self.w, log=log, trainer=fake_trainer(log, history_over=hist),
                            MAKE_SCORER=lambda path, s: (lambda X: fake_probs(path, len(X))))
        # lo scorer finto dà il val_loss dell'ultima epoca: con l'ultima infinita il checkpoint non è quello del minimo finito
        self.assertEqual(rc, 1)
        self.assertIn("is not the one of minimum finite val_loss", err + out)

    def test_non_finite_restored_weights_not_evaluable(self):
        log = []
        rc, out, err = flow(self.w, log=log, trainer=fake_trainer(log, weights_finite=lambda n, s: s != 42))
        self.assertEqual(rc, 0, err[-3000:])
        st = json.loads((run_folder(self.w) / "status.json").read_text())
        self.assertEqual({x["reason"] for x in st["not_evaluable"]}, {"not evaluable: restored weights not finite"})
        self.assertEqual(len(st["not_evaluable"]), 4)

    def test_in_memory_passes_on_the_dev_and_the_fitting(self):
        """Le due passate in memoria della fase 1, nello stato ordinario: il dev con i suoi frame, poi il fitting con i suoi."""
        seen = []

        def fwd(model, part, batch, what, state, layers=()):
            seen.append((what, state, len(part["y"])))
            return {"p": np.zeros((len(part["y"]), 2))}, {}
        rc, out, err = flow(self.w, fwd=fwd)
        self.assertEqual(rc, 0, err[-2000:])
        with injected(**base_hooks(self.w)):
            _, _, _, pcfg, _ = C.load_config(self.w.c5_path, self.w.base)
        n_fit, n_dev = pcfg["arms"]["arm_b"]["total"], pcfg["dev"]["total"]
        self.assertEqual(seen[:2], [("in-memory dev", "ordinary", n_dev), ("in-memory fitting", "ordinary", n_fit)])
        self.assertEqual(len(seen), 24)

    def test_non_finite_in_memory_fitting_not_evaluable(self):
        rc, out, err = flow(self.w, fwd=fake_forward_checked(bad=("AttackNetV1 43",), role="fitting"))
        self.assertEqual(rc, 0, err[-3000:])
        run = run_folder(self.w)
        st = json.loads((run / "status.json").read_text())
        self.assertEqual(len(st["not_evaluable"]), 4)
        self.assertEqual({x["reason"] for x in st["not_evaluable"]},
                         {"not evaluable: non-finite scores or logits on the in-memory fitting (ordinary BN)"})
        self.assertNotIn("AttackNetV1_fold1_seed43.keras", os.listdir(run / "restricted"))
        card = json.loads((run / "restricted" / "AttackNetV1_fold1_seed43_card.json").read_text())
        reason = "not evaluable: non-finite scores or logits on the in-memory fitting (ordinary BN)"
        self.assertEqual((card["rule_n"]["evaluable"], card["rule_n"]["reason"], card["not_evaluable_reason"]), (False, reason, reason))

    def test_non_finite_in_memory_dev_not_evaluable(self):
        rc, out, err = flow(self.w, fwd=fake_forward_checked(bad=("AttackNetV1 42",), role="dev"))
        self.assertEqual(rc, 0, err[-3000:])
        run = run_folder(self.w)
        st = json.loads((run / "status.json").read_text())
        self.assertEqual(len(st["not_evaluable"]), 4)
        self.assertIn("non-finite scores or logits on the in-memory dev", st["not_evaluable"][0]["reason"])
        self.assertNotIn("AttackNetV1_fold1_seed42.keras", os.listdir(run / "restricted"))

    def test_not_evaluable_in_phase_two(self):
        name = "AttackNetV1_fold2_seed42"
        rc, out, err = flow(self.w, reestimate=fake_reestimate(raise_for=(name,)))
        self.assertEqual(rc, 0, err[-3000:])
        run = run_folder(self.w)
        st = json.loads((run / "status.json").read_text())
        self.assertEqual(st["not_evaluable"], [{"name": name, "reason": "not evaluable: non-finite re-estimated statistics"}])
        files = os.listdir(run / "restricted")
        self.assertIn(f"{name}.keras", files)
        self.assertNotIn(f"{name}_dev_ordinary.csv", files)
        self.assertIn(f"   {name}: not evaluable: non-finite re-estimated statistics", (run / "run.log").read_text())

    def test_evaluator_mismatch(self):
        rc, out, err = flow(self.w, reestimate=fake_reestimate(ordinary_over=lambda rec, po: po + (0.01 if rec["seed"] == 42 else 0.0)))
        self.assertEqual(rc, 4, err[-3000:])
        run = run_folder(self.w)
        self.assertFalse((run / "public").exists())
        self.assertTrue((run / "restricted" / "SHA256SUMS").is_file())
        st = json.loads((run / "status.json").read_text())
        self.assertEqual(st["status"], "MISMATCH")
        self.assertEqual(st["restricted_sha256sums_sha256"], sha((run / "restricted" / "SHA256SUMS").read_bytes()))

    def test_cap_incomplete(self):
        log = []

        def hist(net, seed, h):
            return h
        h = flow_hooks(self.w, log=log)
        saved = NR.load_pins

        def pins_small_cap(*a, **k):
            p = saved(*a, **k)
            p.caps = {r: 1e-9 for r in p.caps}
            return p
        with mock.patch.object(NR, "load_pins", pins_small_cap), mock.patch.object(C, "forward_checked", fake_forward_checked()):
            rc, out, err = run_main(self.w.anet_argv("run"), h)
        self.assertEqual(rc, 2, err[-3000:])
        st = json.loads((run_folder(self.w) / "status.json").read_text())
        self.assertEqual(st["status"], "INCOMPLETE")
        self.assertEqual(st["stopped"]["cause"], "cap")

    def test_patience_12_run(self):
        """Un run a patience 12: la patience arriva al trainer, nella scheda e nel riepilogo; stessa ricetta di C2."""
        seen = []
        log = []
        real = fake_trainer(log)

        def trainer(data, training, network, seed, *a):
            seen.append(training["early_stopping_patience"])
            return real(data, training, network, seed, *a)
        rc, out, err = flow(self.w, trainer=trainer, run="anet_v2_1_p12", argv=self.w.anet_argv("run", run="anet_v2_1_p12"))
        self.assertEqual(rc, 0, err[-2000:])
        self.assertEqual(seen, [12] * 12)
        run = run_folder(self.w, "anet_v2_1_p12")
        summ = json.loads((run / "public" / "summary.json").read_text())
        self.assertEqual((summ["training"]["early_stopping_patience"], summ["patience"], summ["network"]), (12, 12, "AttackNetV2_1"))
        card = json.loads((run / "restricted" / "AttackNetV2_1_fold4_seed44_card.json").read_text())
        self.assertEqual(card["early_stopping_patience"], 12)
        self.assertEqual({r["patience"] for r in csv_rows(run / "public" / "models.csv")}, {"12"})

    def budget_class(self, stop_at=None, caps=None):
        """SR.Budget con il tetto registrato e, se stop_at, il superamento al passo dato (un tetto che scatta lì)."""
        class B(SR.Budget):
            def __init__(b, cap_minutes):
                super().__init__(cap_minutes)
                if caps is not None:
                    caps.append(cap_minutes)

            def check(b, step):
                if step == stop_at:
                    raise SR.CapExceeded(step, 1.0)
                return super().check(step)
        return B

    def test_cap_during_a_later_training_discards_only_that_model(self):
        name = "AttackNetV1_fold1_seed44"                       # posizione 3
        with mock.patch.object(SR, "Budget", self.budget_class(stop_at=f"epoch 2 of {name}")):
            rc, out, err = flow(self.w)
        self.assertEqual(rc, 2, err[-2000:])
        run = run_folder(self.w)
        st = json.loads((run / "status.json").read_text())
        self.assertEqual((st["status"], st["stopped"]["discarded"]), ("INCOMPLETE", name))
        self.assertEqual(st["complete_phase_1"], ["AttackNetV1_fold1_seed42", "AttackNetV1_fold1_seed43"])
        self.assertFalse(any(f.startswith(name) for f in os.listdir(run / "restricted")))
        self.assertTrue((run / "restricted" / "AttackNetV1_fold1_seed43.keras").is_file())
        rows = csv_rows(run / "public" / "models.csv")
        self.assertEqual([r["model_status"] for r in rows[:4]], ["phase 1 complete, BN states not evaluated"] * 2 + [R.DISCARDED, "not started"])
        self.assertEqual(rows[2]["epochs_run"], "NA")

    def test_cap_read_from_run_caps_in_hours_of_wall_time(self):
        caps = []
        with mock.patch.object(SR, "Budget", self.budget_class(caps=caps)):
            rc, out, err = flow(self.w, run="anet_v2_2_p4", argv=self.w.anet_argv("run", run="anet_v2_2_p4"))
        self.assertEqual(rc, 0, err[-2000:])
        row = [r for r in FR.parse_csv((self.w.pkg / "plan" / "run_caps.csv").read_bytes(), C.CAPS_COLUMNS, "c") if r["run"] == "anet_v2_2_p4"][0]
        self.assertEqual(caps, [float(row["cap_wall_hours"]) * 60])
        st = json.loads((run_folder(self.w, "anet_v2_2_p4") / "status.json").read_text())
        self.assertEqual((st["cap_hours"], st["cap_unit"]), (float(row["cap_wall_hours"]), PAC.RUN_CAP_UNIT))

    def test_card_counts_checked(self):
        log = []
        real = fake_trainer(log)

        def trainer(*a, **k):
            card, model = real(*a, **k)
            return {**card, "n_dev": card["n_dev"] + 1}, model
        rc, out, err = flow(self.w, trainer=trainer)
        self.assertEqual(rc, 1)
        self.assertIn("the card reports", json.loads((run_folder(self.w) / "status.json").read_text())["reason"])

    def test_date_only_for_an_unofficial_root(self):
        argv = self.w.anet_argv("run", unofficial=False, out=ROOT / NR.DEFAULT_OUT_ROOT)
        rc, out, err = flow(self.w, argv=argv)
        self.assertEqual(rc, 1)
        self.assertIn("--date is accepted only with --unofficial-out-root", err)
        self.assertFalse((ROOT / NR.DEFAULT_OUT_ROOT).exists())

    def test_delivered_configuration_checks_the_hooks_also_with_an_unofficial_root(self):
        with injected(n={"DEFAULT_CONFIG": os.path.relpath(self.w.anet_cfg, ROOT)}):
            rc, out, err = flow(self.w)
        self.assertEqual(rc, 1)
        self.assertIn("test hooks are set", err)
        self.assertFalse(self.w.anet_out.exists())

    def test_attempts_and_refusals(self):
        rc, out, err = flow(self.w)
        self.assertEqual(rc, 0, err[-2000:])
        rc, out, err = flow(self.w)
        self.assertEqual(rc, 1)
        self.assertIn("this attempt must be --attempt 2", err)
        rc, out, err = flow(self.w, argv=self.w.anet_argv("run", attempt=2))
        self.assertEqual(rc, 0, err[-2000:])
        self.assertTrue(run_folder(self.w, attempt=2).is_dir())
        rc, out, err = flow(self.w, argv=self.w.anet_argv("run", attempt=None))
        self.assertEqual(rc, 1)
        self.assertIn("--attempt N is required", err)

    def test_refusals_before_any_folder(self):
        w = self.w
        cases = [(w.anet_argv("run", est=False), "run needs --estimator-check"),
                 (w.anet_argv("run", unofficial=False), "needs --unofficial-out-root"),
                 ([x for x in w.anet_argv("run") if x != "--i-am-authorized"], "pass --i-am-authorized"),
                 (w.anet_argv("run", out=ROOT / "results" / "elsewhere"), "inside the repository")]
        for argv, needle in cases:
            with self.subTest(needle):
                rc, out, err = flow(w, argv=argv)
                self.assertEqual(rc, 1)
                self.assertIn(needle, err)
                self.assertFalse(w.anet_out.exists())
        h = flow_hooks(w)
        h["n"]["REPO_STATE"] = lambda root: {"commit": "abcdef0", "dirty": True}
        rc, out, err = run_main(w.anet_argv("run"), h)
        self.assertEqual(rc, 1)
        self.assertIn("work tree is not clean", err)
        self.assertFalse(w.anet_out.exists())

    def test_official_run_needs_an_official_estimator_check(self):
        cfg, cfg_sha = NR.load_config(self.w.anet_cfg)
        code = NR.check_code(cfg)
        with self.assertRaisesRegex(FR.Failure, "official check-estimator"):
            NR.load_estimator_record(self.w.est_anet, cfg_sha, code, True)

    def test_dev_frames_checked_per_fold(self):
        recs = [{"fold": 1, "states": {"ordinary": {"n_frame_ids": 2, "frame_ids_sha256": "a"}, "re-estimated": {"n_frame_ids": 2, "frame_ids_sha256": "a"}}},
                {"fold": 1, "states": {"ordinary": {"n_frame_ids": 2, "frame_ids_sha256": "b"}}}]
        with self.assertRaisesRegex(FR.Failure, "dev frames of fold 1"):
            NR.check_dev_frames(recs)
        self.assertEqual(NR.check_dev_frames(recs[:1])["folds_compared"], 1)

    def test_failure_writes_fail_status_and_removes_public(self):
        def scorer(path, s):
            return lambda X: np.full((len(X), 2), np.nan)
        rc, out, err = flow(self.w, MAKE_SCORER=scorer)
        self.assertEqual(rc, 1)
        run = run_folder(self.w)
        st = json.loads((run / "status.json").read_text())
        self.assertEqual(st["status"], "FAIL")
        self.assertIn("non-finite or outside [0, 1]", st["reason"])
        self.assertTrue((run / "run.log").read_text().rstrip().endswith("RESULT: FAIL (run)"))

    def test_forbidden_text_in_the_log_never_after_a_pass(self):
        log = []
        real = fake_trainer(log)

        def trainer(*a, **k):
            print(f"working in {Path.home()}", flush=True)
            return real(*a, **k)
        rc, out, err = flow(self.w, trainer=trainer)
        self.assertEqual(rc, 1)
        logt = (run_folder(self.w) / "run.log").read_text()
        self.assertNotIn("RESULT: PASS", logt)
        self.assertTrue(logt.rstrip().endswith("RESULT: FAIL (run)"))
        self.assertIn("holds text that a publishable output must not hold", err)

    def test_threads_read_back_differ_stop_before_training(self):
        log = []
        h = flow_hooks(self.w, log=log)
        tfm = CPR.FakeTF(log=log, offset=1)
        h["r"]["TF_MODULE"] = lambda: tfm
        with mock.patch.object(C, "forward_checked", fake_forward_checked()):
            rc, out, err = run_main(self.w.anet_argv("run"), h)
        self.assertEqual(rc, 1)
        self.assertNotIn("train", [x[0] for x in log if isinstance(x, tuple)])
        st = json.loads((run_folder(self.w) / "status.json").read_text())
        self.assertEqual(st["status"], "FAIL")
        self.assertIn("thread", st["reason"])
        shutil.rmtree(self.w.anet_out)
        # thread effettivi della prova diversi da quelli di adesso: la regola dei thread ferma il run prima di ogni addestramento
        th = {"requested": {"intra_op": 10, "inter_op": 10}, "effective": {"intra_op": 9, "inter_op": 10}, "set_by_this_runner": True,
              "environment": {"TF_NUM_INTRAOP_THREADS": None, "TF_NUM_INTEROP_THREADS": None}, "logical_cpus": FIXED_M["logical_cpus"]}
        self.w.write_anet(trial_over={"threads": th})
        self.addCleanup(self.w.write_anet)
        log = []
        rc, out, err = flow(self.w, log=log)
        self.assertEqual(rc, 1)
        self.assertNotIn("train", [x[0] for x in log if isinstance(x, tuple)])
        self.assertIn("thread rule: effective differ from the official trial", json.loads((run_folder(self.w) / "status.json").read_text())["reason"])

    def test_inputs_changed_during_the_run_fail(self):
        log = []
        real = fake_trainer(log)
        cfgp = self.w.anet_cfg
        saved = cfgp.read_bytes()
        self.addCleanup(cfgp.write_bytes, saved)

        def trainer(data, training, network, seed, *a):
            if seed == 44 and len(log) > 8:
                cfgp.write_bytes(saved + b" ")
            return real(data, training, network, seed, *a)
        rc, out, err = flow(self.w, trainer=trainer)
        self.assertEqual(rc, 1)
        st = json.loads((run_folder(self.w) / "status.json").read_text())
        self.assertIn("the configuration changed during the run", st["reason"])

    def test_every_run_name_and_run_id(self):
        for r in NR.RUNS:
            rid = NR.run_id_for("run", DATE, STATE, 1, r)
            self.assertTrue(NR.RUN_ID_RE.match(rid))
            self.assertEqual(NR.run_id_for("run", DATE, STATE, 3, r), rid + "-r3")
        self.assertTrue(NR.EST_ID_RE.match(NR.run_id_for("estimator", DATE, STATE, 1)))

    def test_entry_that_looks_like_a_run_id_stops(self):
        self.w.anet_out.mkdir(parents=True)
        (self.w.anet_out / f"{DATE}-ANETRUN-v1-p4-XYZ").mkdir()
        with self.assertRaisesRegex(FR.Failure, "looks like a run id"):
            NR.previous_attempts(self.w.anet_out, "run", "anet_v1_p4")
        self.assertEqual(NR.previous_attempts(self.w.anet_out, "run", "anet_v2_1_p4"), [])


# ============================================================================ G emendamento del pacchetto della PR A (PR #34)

PR_A = "delivery/pad-attacknet-comparison"
AMENDED = "delivery/pad-attacknet-run/pr34_amended"
AMEND_NOTE = ("**Amended copy.** This file is `PROTOCOL.md` of the PR A package (pull request #34, folder "
              "`research/pad-attacknet-comparison/`), which stays unchanged; the only differences from it are this heading and two paragraphs marked "
              "\"Amendment of 22 November 2026\": one after the list of §6 (k)/(n) (the repetition of a stopped run), one after "
              "\"What follows each outcome\" in §7 (the order of publication of the trial). Both were fixed before any of the six "
              "runs. The SHA-256 of the original is in the `SHA256SUMS` of that package.")
AMEND_P7 = ("**Amendment of 22 November 2026: deviation from the pre-registered order.** The order pre-registered above for the "
            "official trial (\"it is reported on the C5 card before anything else\") was not kept: pull request #34, with the PR A "
            "package and the record of the trial `20261122-ANETTRIAL-a951ed0`, was opened on 22 November 2026 at 09:23:36 (+0100), "
            "before the outcome of the trial was reported on the C5 card (comment of 09:34:29, followed at 09:36:00 by the comment "
            "with the link to the pull request). The deviation concerns the order of publication only: no code, configuration, record "
            "or number of the PR A package depends on it. The sentence of the PR A `README.md` \"its outcome is reported on the C5 card "
            "first\" states the pre-registered order, not the one that took place; its amended copy is next to this file.")
AMEND_N = ("**Amendment of 22 November 2026: a stopped run (fixed before any of the six runs).** This replaces the sentence of the "
           "list above \"a run stopped at its cap or by a FAIL is resumed only for the missing trainings, with the same seeds, code and "
           "configuration, before the evaluation\": a run stopped at its cap, by SIGUSR1, by a FAIL, a MISMATCH or an interruption is not "
           "resumed within its folder; after its outcome has been reported on the C5 card, it is repeated as a new attempt in a new "
           "folder from position 1, with the same seeds, code and configuration, before the evaluation, as in C2; the earlier attempt "
           "is kept; the evaluation reads, for each of the six runs, only the last attempt, which must be PASS, with phase 1 complete "
           "for the 12 positions (not evaluable models included). The repetition of a stopped run retrains every position, those not "
           "evaluable included, and their outcome in the last attempt is the one evaluated; a run is never repeated because of a not "
           "evaluable model (the sentence \"no new attempt with the same seed for a non-finite case\" above stays in force). Reason: "
           "it is the rule of C2; a run resumed halfway would mix models of two processes; no number changes if no run stops. "
           "Clarification: a model found not evaluable in phase 2 keeps its checkpoint and phase-1 dev scores in `restricted/`, marked "
           "not evaluable in `models.csv`; the evaluation does not score it on the test. The rest of the list is unchanged.")
N_ANCHOR = "\n\n(m) **Descriptive comparison with ResNet-18**"
README_NOTE = ("**Amended copy.** This file is `README.md` of the PR A package (pull request #34, folder "
               "`research/pad-attacknet-comparison/`), which stays unchanged; the only differences from it are this heading and the "
               "first sentence of \"What follows\". The SHA-256 of the original is in the `SHA256SUMS` of that package.")
README_OLD = "The official trial is done (PASS, `PROTOCOL.md` §7); its outcome is reported on the C5 card first."
README_NEW = ("The official trial is done (PASS, `PROTOCOL.md` §7); its outcome was reported on the C5 card after the pull "
              "request of this package was opened, a deviation from the pre-registered order recorded in the amended `PROTOCOL.md` §7.")
P7_ANCHOR = "\n\n**Official trial** (`trial/`:"


def wrap(text: str) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, 120, break_long_words=False, break_on_hyphens=False))


def amended(name: str, text: str) -> str:
    """La copia emendata di un file del pacchetto della PR A: una nota dopo il titolo e, per PROTOCOL.md, i due paragrafi
    degli emendamenti (dopo l'elenco di §6 (k)/(n); dopo «What follows each outcome» in §7, prima del record della prova);
    per README.md la prima frase di «What follows» sostituita. Nient'altro cambia (lo controlla il test)."""
    title, rest = text.split("\n", 1)
    if name == "PROTOCOL.md":
        if rest.count(P7_ANCHOR) != 1:
            raise ValueError("PROTOCOL.md: the anchor of the record of the trial in §7 is not found exactly once")
        if rest.count(N_ANCHOR) != 1:
            raise ValueError("PROTOCOL.md: the anchor after the list of §6 (k)/(n) is not found exactly once")
        rest = rest.replace(N_ANCHOR, "\n\n" + wrap(AMEND_N) + N_ANCHOR)
        rest = rest.replace(P7_ANCHOR, "\n\n" + wrap(AMEND_P7) + P7_ANCHOR)
        return f"{title}\n\n{wrap(AMEND_NOTE)}\n{rest}"
    if rest.count(README_OLD) != 1:
        raise ValueError("README.md: the first sentence of What follows is not found exactly once")
    i = rest.index(README_OLD)
    j = rest.find("\n\n", i)
    j = len(rest.rstrip("\n")) if j < 0 else j                # il paragrafo è l'ultimo del file
    para = " ".join(rest[i:j].replace(README_OLD, README_NEW).split())
    return f"{title}\n\n{wrap(README_NOTE)}\n{rest[:i]}{wrap(para)}{rest[j:]}"


class TestGAmendment(unittest.TestCase):
    def test_amended_copies_differ_only_where_declared(self):
        """I due file emendati sono le copie dei file della PR A con le sole modifiche dichiarate; gli originali hanno lo
        SHA-256 del SHA256SUMS del pacchetto della PR A (quello fissato dalla configurazione)."""
        amended_dir = next((d for d in (ROOT / AMENDED, ROOT.parent / "pr34_amended") if d.is_dir()), None)   # repository o pacchetto
        if not (ROOT / PR_A / "SHA256SUMS").is_file() or amended_dir is None or not (ROOT / NR.DEFAULT_CONFIG).is_file():
            skip_missing(self, "delivered configuration")
        sums = FR.read_sums((ROOT / PR_A / "SHA256SUMS").read_bytes(), "PR A SHA256SUMS")
        for name in ("PROTOCOL.md", "README.md"):
            with self.subTest(name):
                orig = (ROOT / PR_A / name).read_bytes()
                self.assertEqual(sha(orig), sums[name])
                self.assertEqual((amended_dir / name).read_text(encoding="utf-8"), amended(name, orig.decode("utf-8")))
        cfg = json.loads((ROOT / NR.DEFAULT_CONFIG).read_text(encoding="utf-8"))
        self.assertEqual(sha((ROOT / PR_A / "SHA256SUMS").read_bytes()), cfg["trial_record"]["package_sums"]["sha256"])

    def test_amendment_texts(self):
        for t in (AMEND_NOTE, AMEND_P7, AMEND_N, README_NOTE, README_NEW):
            self.assertEqual(PAC.output_problems(t, FR.Inputs()), [])
        self.assertIn("09:23:36", AMEND_P7)
        self.assertIn("09:34:29", AMEND_P7)
        pr_a = (ROOT / PR_A / "PROTOCOL.md").read_text(encoding="utf-8") if (ROOT / PR_A / "PROTOCOL.md").is_file() else None
        if pr_a is not None:
            # la frase sostituita è quella preregistrata, alla lettera (a capo a parte)
            for old in AMEND_N.split('"')[1::2][:2]:
                self.assertIn(old, " ".join(pr_a.split()))


# ============================================================================ G2 script di lancio e manifest del pacchetto

LAUNCH = "delivery/anetrun/launch.zsh"
FAKE_RUNNER = """import json, os, sys
with open(os.environ["FAKE_LOG"], "a") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\\n")
print(f"fake {sys.argv[1]}")
sys.exit(int(os.environ.get("FAKE_RC_" + sys.argv[1].replace("-", "_"), "0")))
"""


class TestG2Launch(unittest.TestCase):
    """delivery/anetrun/launch.zsh eseguito davvero, in una cartella temporanea con un pad_anet_run.py finto (registra i
    propri argomenti e esce con il codice dato) e un .venv/bin/python che punta all'interprete della suite."""

    def setUp(self):
        if any(shutil.which(t) is None for t in ("zsh", "caffeinate", "pmset", "sw_vers", "git")):
            skip_missing(self, "macOS tools")
        self.tmp = Path(tempfile.mkdtemp(prefix="anetrun-launch-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.repo = self.tmp / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        (self.repo / "delivery" / "anetrun").mkdir(parents=True)
        (self.repo / ".venv" / "bin").mkdir(parents=True)
        shutil.copy2(ROOT / LAUNCH, self.repo / LAUNCH)
        (self.repo / "scripts" / "pad_anet_run.py").write_text(FAKE_RUNNER, encoding="utf-8")
        os.symlink(sys.executable, self.repo / ".venv" / "bin" / "python")
        self.logs = self.tmp / "logs"
        self.calls = self.tmp / "calls.jsonl"

    def launch(self, *args, nice=0, est=None, **rc):
        env = {k: v for k, v in os.environ.items() if not k.startswith("TF_NUM_") and not k.startswith("ANETRUN_")}
        env.update({"ANETRUN_LOG_DIR": str(self.logs), "ANETRUN_ATTEMPT": "1", "FAKE_LOG": str(self.calls)})
        if est:
            env["ANETRUN_ESTIMATOR"] = est
        env.update({f"FAKE_RC_{k}": str(v) for k, v in rc.items()})
        cmd = (["nice", "-n", str(nice)] if nice else []) + ["zsh", LAUNCH, *args]
        r = subprocess.run(cmd, cwd=self.repo, env=env, capture_output=True, text=True, timeout=120)
        calls = [json.loads(l) for l in self.calls.read_text().splitlines()] if self.calls.exists() else []
        return r, calls

    def conditions(self, label: str) -> str:
        return (self.logs / f"{label}-conditions.txt").read_text(encoding="utf-8")

    def test_check_estimator_launched_with_every_condition(self):
        r, calls = self.launch("check-estimator", check_estimator=0)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(calls, [["check-machine"], ["check-inputs"], ["check-estimator", "--attempt", "1", "--i-am-authorized"]])
        c = self.conditions("check-estimator-attempt1")
        for part in ("== conditions before", "== system", "ProductVersion", "== power", "== nice of the launching shell\n0\n",
                     "== environment: number of TF_NUM_ variables set\n0\n", "== git", "== check-machine", "check-machine exit 0",
                     "== check-inputs", "check-inputs exit 0", "== command", "check-estimator exit 0", "== conditions after"):
            self.assertIn(part, c)
        self.assertEqual(PAC.output_problems(c, FR.Inputs()), [])

    def test_run_passes_the_estimator_check_as_two_words_and_propagates_the_exit_code(self):
        env_est = "results/anetrun/20261123-ANETEST-abcdef0"
        r, calls = self.launch("run", "anet_v1_p4", est=env_est, run=2)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertEqual(calls, [["check-machine"], ["check-inputs", "--estimator-check", env_est],
                                 ["run", "--run", "anet_v1_p4", "--estimator-check", env_est, "--attempt", "1", "--i-am-authorized"]])
        self.assertIn("run exit 2", self.conditions("anet_v1_p4-attempt1"))

    def test_a_failed_check_stops_before_the_command(self):
        for which in ("check_machine", "check_inputs"):
            with self.subTest(which):
                shutil.rmtree(self.logs, ignore_errors=True)
                self.calls.unlink(missing_ok=True)
                r, calls = self.launch("check-estimator", **{which: 1})
                self.assertEqual(r.returncode, 1)
                self.assertNotIn("check-estimator", [c[0] for c in calls])
                self.assertIn("== not launched: check-machine or check-inputs failed", self.conditions("check-estimator-attempt1"))

    def test_nice_not_zero_stops_before_any_command(self):
        r, calls = self.launch("check-estimator", nice=5)
        self.assertEqual(r.returncode, 1)
        self.assertEqual(calls, [])
        c = self.conditions("check-estimator-attempt1")
        self.assertIn("== nice of the launching shell\n5\n", c)
        self.assertIn("== not launched: the nice value is not 0", c)
        self.assertIn("setopt nobgnice", r.stderr)

    def test_log_folder_inside_the_repository_refused(self):
        env_dir = self.logs
        self.logs = self.repo / "logs"
        try:
            r, calls = self.launch("check-estimator")
        finally:
            self.logs = env_dir
        self.assertEqual(r.returncode, 9)
        self.assertEqual(calls, [])


class TestG3Manifest(unittest.TestCase):
    def test_package_manifest_lists_every_file_but_itself(self):
        """SHA256SUMS del pacchetto (dal repository o dalla cartella del pacchetto): ogni file del pacchetto tranne se stesso,
        e nient'altro; i SHA256SUMS dentro livenesslab/ compresi."""
        pkg = next((d for d in (ROOT / "delivery" / "pad-attacknet-run", ROOT.parent) if (d / "pr34_amended").is_dir() and (d / "SHA256SUMS").is_file()), None)
        if pkg is None:
            skip_missing(self, "delivered configuration")
        listed = sorted(FR.read_sums((pkg / "SHA256SUMS").read_bytes(), "package SHA256SUMS"))
        files = sorted(str(p.relative_to(pkg)) for p in pkg.rglob("*") if p.is_file() and p != pkg / "SHA256SUMS" and "__pycache__" not in p.parts)
        self.assertEqual(listed, files)


class TestG4ReadmeLines(unittest.TestCase):
    """Le righe attese del README del pacchetto: uguali ai numeri di adesso (test caricati da questa suite, mutazioni
    dichiarate) e, quando i log di test_logs/ portano lo SHA-256 attuale di questa suite (log non scaduti), uguali alle
    ultime righe dei log."""

    def package(self):
        pkg = next((d for d in (ROOT / "delivery" / "pad-attacknet-run", ROOT.parent) if (d / "pr34_amended").is_dir() and (d / "README.md").is_file()), None)
        if pkg is None:
            skip_missing(self, "delivered configuration")
        return pkg

    def test_readme_lines_equal_the_current_numbers_and_the_logs(self):
        pkg = self.package()
        readme = " ".join((pkg / "README.md").read_text(encoding="utf-8").split())
        n = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__]).countTestCases()   # tutti, anche con -k
        spec = importlib.util.spec_from_file_location("anetrun_mutations", ROOT / "scripts" / "check_pad_anet_run_mutations.py")
        M = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(M)
        off = sum(1 for m in M.MUTANTS if m[1] == M.OFF)
        mlist = (f"{len(M.MUTANTS)} mutations ({off} offline, {len(M.MUTANTS) - off} with TensorFlow), {len(M.CONTROL)} control mutation and "
                 f"{len(M.REDUNDANT)} declared redundant, every pattern found exactly once")
        lines = {"tf": f"tests run: {n}, failures: 0, errors: 0, skipped: 0", "notf": f"tests run: {n}, failures: 0, errors: 0, skipped: 1",
                 "list": mlist, "killed": f"mutations killed: {len(M.MUTANTS)}/{len(M.MUTANTS)}",
                 "control": f"control mutations survived: {len(M.CONTROL) + len(M.REDUNDANT)}/{len(M.CONTROL) + len(M.REDUNDANT)}"}
        launch_tests = len([t for t in dir(TestG2Launch) if t.startswith("test_")])      # saltati senza gli strumenti di macOS
        lines["linux_tf"] = f"tests run: {n}, failures: 0, errors: 0, skipped: {launch_tests}"
        lines["linux_notf"] = f"tests run: {n}, failures: 0, errors: 0, skipped: {launch_tests + 1}"
        lines["linux_killed"] = (f"mutations killed: {len(M.MUTANTS) - sum(1 for m in M.MUTANTS if m[2] == M.L)}/"
                                 f"{len(M.MUTANTS) - sum(1 for m in M.MUTANTS if m[2] == M.L)}")
        for k, line in lines.items():
            with self.subTest(k):
                self.assertIn(line, readme)
        import re as _re
        # nessuna riga vecchia accanto a quelle giuste: ogni «tests run: …» e ogni «mutations killed: …» del README è attesa
        self.assertEqual(sorted(set(_re.findall(r"tests run: \d+, failures: \d+, errors: \d+, skipped: \d+", readme))),
                         sorted({lines[k] for k in ("tf", "notf", "linux_tf", "linux_notf")}))
        self.assertEqual(sorted(set(_re.findall(r"mutations killed: \d+/\d+", readme))), sorted({lines["killed"], lines["linux_killed"]}))
        logs = pkg / "test_logs"
        current = sha((ROOT / "scripts" / "check_pad_anet_run.py").read_bytes())
        for log, k in (("check_pad_anet_run_from_this_folder.txt", "tf"), ("check_pad_anet_run_without_tensorflow_from_this_folder.txt", "notf"),
                       ("check_pad_anet_run_mutations_list.txt", "list"), ("check_pad_anet_run_mutations.txt", "killed"),
                       ("check_pad_anet_run_mutations.txt", "control")):
            text = (logs / log).read_text(encoding="utf-8") if (logs / log).is_file() else ""
            if f"# SHA-256 {current}  scripts/check_pad_anet_run.py" not in text or "--regenerating-logs" in text:
                # log scaduto, o della prima passata di rigenerazione (con un salto in più): non è un log pubblicato; quelli
                # pubblicati (seconda passata) li controlla ogni esecuzione successiva, e TestG5LogsCurrent ne vuole le impronte
                continue
            with self.subTest(log):
                self.assertIn(lines[k], text.splitlines())


class TestG5LogsCurrent(unittest.TestCase):
    def test_every_log_cites_the_current_sha256_of_its_files(self):
        """Nel pacchetto consegnato ogni log di test_logs/ finisce con «exit 0» e ogni riga «# SHA-256 <impronta>  <file>»
        della sua intestazione è lo SHA-256 attuale del file. Nessuna clausola di salto se il pacchetto c'è; solo
        l'opzione esplicita --regenerating-logs (prima delle due passate che rigenerano i log) lo esclude."""
        pkg = next((d for d in (ROOT / "delivery" / "pad-attacknet-run", ROOT.parent) if (d / "pr34_amended").is_dir() and (d / "test_logs").is_dir()), None)
        if pkg is None:
            skip_missing(self, "delivered configuration")
        if REGENERATING_LOGS:
            skip_missing(self, "logs being regenerated")
        logs = sorted((pkg / "test_logs").glob("*.txt"))
        self.assertEqual(len(logs), 10)
        for log in logs:
            with self.subTest(log.name):
                lines = log.read_text(encoding="utf-8").splitlines()
                self.assertIn("exit 0", lines)
                cited = [l.split() for l in lines if l.startswith("# SHA-256 ")]
                self.assertGreaterEqual(len(cited), 9)
                for _, _, h, f in cited:
                    self.assertEqual(h, sha((ROOT / f).read_bytes()), f)


# ============================================================================ H testi pubblici

class TestHPublicTexts(unittest.TestCase):
    def test_no_identifier_or_path_in_the_texts(self):
        cfg, _ = NR.load_config(world().anet_cfg)
        for t in (NR.TEST_READ, json.dumps(cfg)):
            self.assertEqual(PAC.output_problems(t, FR.Inputs()), [])
            self.assertEqual(FR.public_problems(t, FR.Inputs(), markdown=True), [])


# ============================================================================ I con TensorFlow: un run vero su 8×8

class TestIRealRun(unittest.TestCase):
    def test_real_run_on_8x8(self):
        """Un run vero (anet_v2_2_p4) con il trainer, il costruttore del registro, lo scorer e la ristima veri su frame 8×8:
        PASS, 12 modelli completi, checkpoint ricaricati e controllati, accordo fra scorer e valutatore."""
        CAC.need_tf(self)
        w = world()
        shutil.rmtree(w.anet_out, ignore_errors=True)
        self.addCleanup(shutil.rmtree, w.anet_out, True)
        h = base_hooks(w, ARCH_LOADER=lambda path: CAC.ref_arch())
        rc, out, err = run_main(w.anet_argv("run", run="anet_v2_2_p4"), h)
        self.assertEqual(rc, 0, err[-3000:] + out[-3000:])
        run = run_folder(w, "anet_v2_2_p4")
        rows = csv_rows(run / "public" / "models.csv")
        self.assertEqual(len(rows), 12)
        self.assertEqual([r["model_status"] for r in rows], ["complete"] * 12)
        self.assertEqual({r["evaluator_agreement"] for r in rows}, {"1"})
        st = json.loads((run / "status.json").read_text())
        self.assertEqual((st["status"], st["threads_rule"]["equal"], st["hooks"]["set"]), ("PASS", True, sorted(["pad_anet_comparison.ARCH_LOADER", "pad_anet_comparison.REPO_STATE", "pad_anet_run.ARCH_LOADER",
                                                                                     "pad_anet_run.REPO_STATE", "pad_arch_comparison.ARCH_LOADER",
                                                                                     "pad_arch_comparison.REPO_STATE", "pad_arch_run.ENVIRONMENT",
                                                                                     "pad_arch_run.MACHINE"])))
        summ = json.loads((run / "public" / "summary.json").read_text())
        self.assertEqual(summ["dev_frames"]["models_compared"], 12)
        for m in summ["models"]:
            self.assertEqual(m["card"]["non_finite"], [])
        stats = json.loads((run / "restricted" / "AttackNetV2_2_fold1_seed42_bn_statistics.json").read_text())
        self.assertEqual(stats["order"], AN.bn_order("AttackNetV2_2"))


class ZZRepositoryUntouched(unittest.TestCase):
    def test_declared_skips_only(self):
        self.assertEqual(sorted(MISSING_INPUT), ["delivered configuration", "logs being regenerated", "macOS tools", "reference architecture", "tensorflow"])


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
    after = guard_state()
    changed = [g for g in GUARDED if before[g] != after[g]]
    skipped = len(res.skipped)
    for t, why in res.skipped:
        print(f"skipped {t.id()}: {why}")
    print(f"tests run: {res.testsRun}, failures: {len(res.failures)}, errors: {len(res.errors)}, skipped: {skipped}")
    if changed:
        print(f"ERROR: the suite changed {', '.join(changed)}")
    ok = res.wasSuccessful() and not changed
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
