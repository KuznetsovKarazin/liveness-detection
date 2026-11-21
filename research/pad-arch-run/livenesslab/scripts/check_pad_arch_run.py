"""
Test di scripts/pad_arch_run.py (i due run NUAA del confronto LivenessNet / ResNet-18, il controllo dello stimatore della
ristima BN e il controllo degli ingressi), offline: nessun dato reale, nessun checkpoint reale.

Mondo sintetico in cartelle temporanee, costruito sul mondo della suite della preregistrazione
(check_pad_arch_comparison.World: manifest M2, cache 8×8, run delle celle, run dei fold, frame estratti, configurazioni
a patience 4 e 12, configurazione della preregistrazione) più: le 24 schede del braccio B dei due run «accettati» finti,
il record finto della prova ufficiale (status.json e trial.json coerenti con la configurazione e il codice), run_order.csv
e run_caps.csv generati dalle funzioni della preregistrazione, una configurazione della PR #27 valida (solo per le
tolleranze e l'ordine), un file fissato per la PR #28, la configurazione di questo runner e il record finto di un
controllo dello stimatore PASS.

Due livelli:
  - senza TensorFlow (classi A-L): configurazione, file fissati, ordine, tetti, regola della macchina e dei thread,
    storia, ordine della ristima sulla topologia, dev e ID dei frame, e il flusso completo di run con un TensorFlow
    finto (thread e dispositivi), un trainer finto, un salvataggio finto, uno scorer e una ristima finti: tetto, SIGUSR1,
    storia diversa o non finita, cartella esistente, nuovo tentativo con ingressi guasti, metadati dello stato finale; un
    test per ogni controllo della macchina e dei thread (processore, altri campi, variabili d'ambiente discordanti o
    diverse dalla prova, thread richiesti e riletti, TensorFlow già inizializzato); check-estimator fino al primo
    modello (classe L: thread, GPU, regola della macchina e delle variabili dei thread, configurazioni delle PR #27 e #28,
    esito);
  - con TensorFlow (classi Z*, ultime: con -f i mutanti uccisi prima non le eseguono): lo stimatore su LivenessNet del
    codice di riferimento uguale bit per bit a quello accettato (BR.recalibrate), indipendenza dall'ordine fra strati che
    non si alimentano a vicenda e confronto con un riferimento a forza bruta su un piccolo modello con rami, ResNet-18
    vera (ordine controllato sul grafo, ristima), un run vero su CPU con frame 8×8 e un trainer accorciato a 2 epoche
    (trial_trainer vero), e check-estimator sul mondo della suite della PR #28 (run BNRECAL vero): PASS e MISMATCH.
Senza TensorFlow i test che costruiscono un modello Keras sono saltati con il motivo dichiarato in MISSING_INPUT; gli
altri girano. Un salto con un motivo non dichiarato rende l'esito FAIL. Le cartelle temporanee stanno in TMPDIR, che deve stare fuori dalla radice del repository (controllato).
Nessuna scrittura nel repository (controllo finale su results/, export/, delivery/).

  .venv/bin/python -I -B scripts/check_pad_arch_run.py [--verbose] [-f]
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_nuaa_folds_run as CFR  # noqa: E402
import check_nuaa_session_ablation_run as SAR  # noqa: E402
import check_pad_arch_comparison as CPA  # noqa: E402
import nuaa_bn_recalibration as BR  # noqa: E402
import nuaa_folds_run as FR  # noqa: E402
import nuaa_session_ablation_run as SR  # noqa: E402
import pad_arch_comparison as PAC  # noqa: E402
import pad_arch_definitions as D  # noqa: E402
import pad_arch_models as AM  # noqa: E402
import pad_arch_run as R  # noqa: E402

TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
REF_ARCH = CPA.REF_ARCH
MISSING_INPUT = {
    "tensorflow": "TensorFlow is not installed: the tests that build, train or re-estimate a Keras model are skipped",
    "reference architecture": "src/livedetection/src/architectures.py (LivenessNet of the reference code) is not present",
    "delivered configuration": "delivery/archrun/pad_arch_run.json is not present",
}
STATE = {"commit": "abcdef0", "dirty": False}
DATE = "20261120"
MACH = {"processor": "Test CPU", "core_levels": [{"name": "Super", "physical_cores": 5}, {"name": "Performance", "physical_cores": 10}],
        "ram_gb": 24.0, "os": "macOS 27.2.1", "logical_cpus": 15, "architecture": "arm64"}
ENV = {"python": "3.11.16", "numpy": "1.26.4", "platform": "darwin", "tensorflow": "2.15.0", "tensorflow-metal": "1.1.0", "keras": "2.15.0",
       "opencv": "4.11.0", "tensorflow_installed": "2.15.0"}
SCORE = {"LivenessNet": 0.3, "ResNet-18": 0.6}         # punteggio d'attacco costante dello scorer finto, per rete
EPOCHS = 3


def skip_missing(test: unittest.TestCase, what: str):
    """Unico modo di saltare un test di questa suite: un ingresso mancante dichiarato in MISSING_INPUT."""
    test.skipTest(f"missing input: {MISSING_INPUT[what]}")


def need_tf(test: unittest.TestCase):
    if not TF_AVAILABLE:
        skip_missing(test, "tensorflow")
    if not REF_ARCH.is_file():
        skip_missing(test, "reference architecture")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tmp_outside_repo():
    t = Path(tempfile.gettempdir()).resolve()
    if t == ROOT.resolve() or ROOT.resolve() in t.parents:
        raise SystemExit("TMPDIR is inside the repository: set it outside the repository root")


# ============================================================================ mondo sintetico

def fake_val_loss(y, c: float) -> float:
    """Il val_loss di uno scorer che dà a ogni frame la probabilità d'attacco c (dai logit log p, come il runner)."""
    y = np.asarray(y, np.int64)
    z = np.log(np.stack([np.full(len(y), 1 - c), np.full(len(y), c)], axis=1))
    return SR.val_loss_logits(z, y)


def fake_history(y_dev, network: str, epochs: int = EPOCHS) -> dict:
    vl = fake_val_loss(y_dev, SCORE[network])
    return {"loss": [0.9 - 0.1 * i for i in range(epochs)], "accuracy": [0.5 + 0.05 * i for i in range(epochs)],
            "val_loss": [vl + 0.1 * (epochs - 1 - i) for i in range(epochs)], "val_accuracy": [0.5] * epochs, "lr": [0.0010000000474974513] * epochs}


def bn_doc() -> dict:
    """Una configurazione della PR #27 valida per BR.load_config (le sue tolleranze, l'ordine di LivenessNet e il batch)."""
    layers = [["input_layer", "InputLayer"], ["conv1_1", "Conv2D"], ["relu1_1", "Activation"], ["bn1_1", "BatchNormalization"], ["conv1_2", "Conv2D"],
              ["relu1_2", "Activation"], ["bn1_2", "BatchNormalization"], ["maxpool1", "MaxPooling2D"], ["dropout1", "Dropout"], ["conv2_1", "Conv2D"],
              ["relu2_1", "Activation"], ["bn2_1", "BatchNormalization"], ["conv2_2", "Conv2D"], ["relu2_2", "Activation"], ["bn2_2", "BatchNormalization"],
              ["maxpool2", "MaxPooling2D"], ["dropout2", "Dropout"], ["flatten", "Flatten"], ["dense1", "Dense"], ["bn_dense", "BatchNormalization"],
              ["dropout_dense", "Dropout"], ["output", "Dense"]]
    bn = list(AM.LIVENESSNET_BN)
    return {"schema": BR.CONFIG_SCHEMA, "task": "t", "authorisation": "a",
            "parent_run": {"run_id": "20261106-SESSABLRUN-abcdef0", "folder": "parentrun", "config_path": "c.json", "config_sha256": "1" * 64,
                           "public_sha256sums_sha256": "2" * 64, "restricted_sha256sums_sha256": "3" * 64, "models": 36, "used": "t"},
            "architecture": {"name": "LivenessNet", "path": "ref/architectures.py", "sha256": "4" * 64, "input_shape": [8, 8, 3], "layers": layers,
                             "augmentation": "none"},
            "batchnorm": {"layers": bn, "kinds": {b: ("dense" if b == "bn_dense" else "conv") for b in bn}, "momentum": 0.99, "epsilon": 0.001,
                          "inference": "t", "running_variance_in_training": "t"},
            "estimator": {"statistics": "t", "order": "t", "mode": "t", "counts": "t", "variance": "t", "cast": "t",
                          "changed_tensors": ["moving_mean", "moving_variance"], "copy": "t", "consistency": "t"},
            "evaluation": {"device": "CPU", "batch": 16, "input": "t", "mode": "t", "probability": "t", "logits": "t", "val_loss": "t", "accuracy": "t",
                           "auroc": "t", "decision": "t"},
            "tolerances": {k: {"value": v, "source": "t"} for k, v in (("dev_probability_max_abs_diff", 1e-5), ("dev_decisions_discordant_at_0_5", 0),
                                                                     ("val_loss_abs", 1e-3), ("dev_accuracy_abs", 1e-6),
                                                                     ("softmax_of_logits_max_abs_diff", 1e-4), ("consistency_rel", 1e-4),
                                                                     ("consistency_abs", 1e-6))},
            "stop_rule": "stop with BASELINE_FAILED",
            "recovery_criterion": {"accuracy_min": 0.75, "accuracy_min_inclusive": True, "val_loss_max": 1.0, "val_loss_max_inclusive": False, "text": "t",
                                   "source": "t"},
            "groups": {"degenerate": {"restored_epochs": [1, 2], "dev_accuracy": 0.5, "within": 1e-6, "source": "t"},
                       "ten": {"definition": "t", "expected": ["arm_b_fold1_seed42"]}, "apart": {"definition": "t", "expected": []},
                       "control": {"definition": "t", "expected_count": 35}},
            "reading_rule": {"primary": "p", "all": "all {k} of {n}", "none": "none {k} of {n}", "some": "some {k} of {n}", "controls": "c", "apart": "a",
                             "cells": {c: f"text {c}" for c in BR.CELLS}, "cells_header": "h", "worsening": "w", "limits": "l", "report": "r"},
            "outputs": {"public": list(BR.PUBLIC_OUTPUTS), "restricted": [f"<model>_{x}" for x in BR.PER_MODEL] + list(BR.RESTRICTED_FIXED)},
            "budget": {"wall_time_cap_minutes": 240, "estimate_minutes": 7, "upper_estimate_minutes": 9, "estimate": "t", "on_cap_exceeded": "INCOMPLETE",
                       "voluntary_stop": "SIGUSR1: INCOMPLETE"},
            "environment": {"tensorflow": "2.15.0"}, "note_path": "note.md", "note_sha256": None, "test": BR.TEST_READ}


@contextlib.contextmanager
def alias_of(alias, target):
    """Un secondo percorso per la stessa cartella su ogni filesystem: os.stat (usato da os.path.samefile e da Path.exists)
    riporta per alias e per ogni percorso sotto di esso l'identità del percorso corrispondente sotto target, come fa un
    filesystem che non distingue le maiuscole per una grafia diversa (la tecnica dei test delle grafie di M2)."""
    real, a, t = os.stat, str(alias), str(target)

    def fake_stat(p, *args, **kwargs):
        if isinstance(p, (str, os.PathLike)):
            q = os.fspath(p)
            if isinstance(q, str) and (q == a or q.startswith(a + os.sep)):
                p = t + q[len(a):]
        return real(p, *args, **kwargs)
    os.stat = fake_stat
    try:
        yield
    finally:
        os.stat = real


def author(t: str = "x") -> str:
    return f"a choice of the author: {t}"


class World(CPA.World):
    """Il mondo della preregistrazione con le 24 schede di riferimento, il record della prova, i file del piano, le
    configurazioni della PR #27 e #28, la configurazione di questo runner e un controllo dello stimatore PASS."""

    def build(self):
        self.alter = {}
        super().build()
        self.runs_out = self.tmp / "runs"
        self.br_cfg = self.base / "bnrecal.json"
        self.br_cfg.write_text(json.dumps(bn_doc(), indent=1) + "\n", encoding="utf-8")
        self.bt_cfg = self.base / "bntest.json"
        self.bt_cfg.write_text("{\"synthetic\": \"bytes pinned only\"}\n", encoding="utf-8")
        self.trial_dir = self.base / "trial"
        self.write_trial()
        self.run_cfg = self.base / "archrun.json"
        self.write_run_cfg(self.run_doc())
        self.est = self.tmp / "estimator" / f"{DATE}-ARCHEST-{STATE['commit']}"
        self.write_estimator_record()

    def dev_y(self, fold: int) -> list:
        return [int(r[3]) for r in sorted(self.cells[("dev", fold)], key=lambda r: r[2])]

    def card_keys(self, name: str, fold: int) -> str:
        return sha("\n".join(sorted(r[2] for r in self.cells[(name, fold)])).encode())

    def write_parents(self, history=None, card_over=None):
        """I due run accettati finti con le 12 schede del braccio B ciascuno (storia di LivenessNet uguale a quella del
        trainer finto: il controllo della storia dà «identical» salvo alterazioni in self.alter) e la configurazione
        della preregistrazione."""
        names = [SR.model_name(*x) for x in SR.training_order(SAR.K, SAR.SEEDS)]
        self.sums = {}
        for key, rid, cfgp, pat in (("patience_4", CPA.P4, self.config, 4), ("patience_12", CPA.P12, self.config12, 12)):
            run = self.base / key
            shutil.rmtree(run, ignore_errors=True)
            (run / "restricted").mkdir(parents=True)
            for f in range(1, SAR.K + 1):
                for s in SAR.SEEDS:
                    model = SR.model_name("arm_b", f, s)
                    card = {"arch": "LivenessNet", "arm": "arm_b", "fold": f, "train_seed": s, "early_stopping_patience": pat,
                            "history": fake_history(self.dev_y(f), "LivenessNet"), "fitting_keys_sha256": self.card_keys("armB_fitting", f),
                            "dev_keys_sha256": self.card_keys("dev", f)}
                    if (key, f, s) in self.alter:
                        self.alter[(key, f, s)](card)
                    p = run / "restricted" / f"{model}_card.json"
                    p.write_text(json.dumps(card, indent=1) + "\n", encoding="utf-8")
                    if key == "patience_4" and model == CPA.REF_MODEL:
                        self.card_sha = sha(p.read_bytes())
            self.sums[key] = CPA.write_sums(run / "restricted")
            if getattr(self, "after_sums", None):
                self.after_sums(key, run / "restricted")
                self.sums[key] = sha((run / "restricted" / "SHA256SUMS").read_bytes())
            st = {"run_id": rid, "status": "PASS", "config_sha256": sha(cfgp.read_bytes()), "restricted_sums_sha256": self.sums[key], "completed": names}
            (run / "status.json").write_text(json.dumps(st, indent=1) + "\n", encoding="utf-8")
        self.write_cfg(self.doc())

    def write_trial(self, status="PASS", over=None):
        shutil.rmtree(self.trial_dir, ignore_errors=True)
        CPA.fake_trial(self, self.trial_dir, status=status, over={"machine": dict(MACH), "environment": dict(ENV), **(over or {})})
        self.trial = json.loads((self.trial_dir / "trial.json").read_text())

    def plan_files(self, order_text=None, caps_text=None):
        cfg_pr = json.loads(self.cfg_path.read_text())
        (self.base / "run_order.csv").write_text(order_text if order_text is not None else R.order_text(cfg_pr), encoding="utf-8")
        (self.base / "run_caps.csv").write_text(caps_text if caps_text is not None else R.caps_text(cfg_pr, self.trial), encoding="utf-8")

    def package_sums(self, over=None):
        """Il SHA256SUMS finto del pacchetto consegnato, con le righe trial/status.json e trial/trial.json della prova."""
        lines = {"trial/status.json": sha((self.trial_dir / "status.json").read_bytes()), "trial/trial.json": sha((self.trial_dir / "trial.json").read_bytes()),
                 "PROTOCOL.md": "1" * 64, **(over or {})}
        (self.base / "package_SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sorted(lines.items())), encoding="utf-8")

    def run_doc(self) -> dict:
        self.plan_files()
        self.package_sums()
        pin = lambda p: {"path": p.name, "sha256": sha(p.read_bytes())}  # noqa: E731
        return {
            "schema": R.CONFIG_SCHEMA, "task": "t", "authorisation": "after the verification, the merge and the go-ahead",
            "operations": "commit between the commands; launch with nohup",
            "preregistration": pin(self.cfg_path),
            "trial_record": {"run_id": CPA.RUN, "folder": "trial", "status_sha256": sha((self.trial_dir / "status.json").read_bytes()),
                             "trial_json_sha256": sha((self.trial_dir / "trial.json").read_bytes()),
                             "package_sums": pin(self.base / "package_SHA256SUMS")},
            "plan_files": {"run_order": pin(self.base / "run_order.csv"), "run_caps": pin(self.base / "run_caps.csv")},
            "code": {f: sha((ROOT / f).read_bytes()) for f in R.CODE_FILES},
            "estimator_check": {"bn_recalibration_config": pin(self.br_cfg), "bn_test_config": pin(self.bt_cfg), "bnrecal_run_id": "20261114-BNRECAL-abcdef0",
                                "models": 36, "rule": "the work stops", "tolerances": "t", "where": author(), "cap_minutes": 240,
                                "cap_unit": R.EST_CAP_UNIT, "cap_rule": author()},
            "runs": list(R.RUNS),
            "machine_rule": {"fields": list(R.MACHINE_FIELDS), "versions": list(R.VERSION_FIELDS), "threads": list(R.THREAD_FIELDS),
                             "consequence": author("a difference stops the work before any training"),
                             "scope": author("the model, not the identity of the physical unit")},
            "sequence": {"phase_1": "p", "phase_2": "p", "reason": author("no threshold.json")},
            "checkpoint": {"format": "f", "when": R.CHECKPOINT_TEXT, "file": "f"},
            "history_check": {"fields": list(R.HISTORY_KEYS), "difference": 0, "reference": "r", "rule": "r", "first_differing_epoch": author(),
                              "non_finite": author(), "consequence": "the run continues and the model stays in the comparison"},
            "estimator": {"orders": R.expected_orders(), "batch": 16, "estimator": "e", "order_rule": author(), "consistency": "c"},
            "dev_evaluation": {"states": list(D.BN_STATES), "threshold": "t", "one_class": "o", "frame_ids": "public outputs and restricted files",
                               "evaluator": author(), "agreement": author("1e-5, MISMATCH")},
            "cap": {"source": "s", "unit": R.CAP_UNIT, "checks": "c", "on_cap": f"stops as INCOMPLETE, {R.DISCARDED}"},
            "after_stop": author("no automatic resume"), "attempts": author("--unofficial-out-root"),
            "outputs": {"run": ["status.json", "run.log"], "public": list(R.PUBLIC_OUTPUTS), "restricted": [f"<model>{x}" for x in R.PER_MODEL] + ["SHA256SUMS"]},
            "choices_of_the_author": {k: "c" for k in R.CHOICE_KEYS},
            "test": R.TEST_READ}

    def write_run_cfg(self, doc: dict) -> Path:
        self.run_cfg.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        return self.run_cfg

    def cfg_sha(self) -> str:
        return sha(self.run_cfg.read_bytes())

    def write_estimator_record(self, status="PASS", over=None, cfg_sha=None):
        """Il record finto di un controllo dello stimatore (stessa configurazione e stesso codice di adesso)."""
        shutil.rmtree(self.est, ignore_errors=True)
        (self.est / "public").mkdir(parents=True)
        cfg, _ = R.load_config(self.run_cfg)
        code = R.check_code(cfg)
        rows = [{"recalibration_pass": True} for _ in range(36)]
        doc = {"schema": R.ESTIMATOR_SCHEMA, "run_id": self.est.name, "status": status, "config_sha256": cfg_sha or self.cfg_sha(), "code_sha256": code,
               "models": rows, "expected_models": 36, "machine_rule": {"equal": True}, "threads_rule": {"equal": True}, "official": False, **(over or {})}
        b = (json.dumps(doc, indent=1) + "\n").encode()
        (self.est / "public" / "estimator_check.json").write_bytes(b)
        (self.est / "status.json").write_text(json.dumps({"run_id": self.est.name, "status": status, "estimator_check_json_sha256": sha(b)}),
                                              encoding="utf-8")

    def refresh(self):
        """Dopo un'alterazione delle schede o della prova: configurazione della preregistrazione, file del piano e
        configurazione di questo runner coerenti, e il record dello stimatore con il nuovo SHA-256."""
        self.write_parents()
        self.write_trial()
        self.write_run_cfg(self.run_doc())
        self.write_estimator_record()

    def argv(self, cmd: str, *extra, run="nuaa_patience_4", out=None, attempt=None, est=True, unofficial=True) -> list:
        a = [cmd, "--config", str(self.run_cfg), "--input-root", str(self.base)]
        if cmd in ("run", "check-estimator"):
            a += ["--out-root", str(out or self.runs_out), "--date", DATE, "--i-am-authorized"] + (["--unofficial-out-root"] if unofficial else [])
        if cmd == "run":
            a += ["--run", run] + (["--estimator-check", str(self.est)] if est else [])
        if attempt is not None:
            a += ["--attempt", str(attempt)]
        return a + list(extra)


WORLD = None


def world() -> World:
    global WORLD
    if WORLD is None:
        WORLD = World()
    return WORLD


@contextlib.contextmanager
def injected(r=None, p=None, sr=None, fr=None, br=None):
    """Punti di iniezione di pad_arch_run (r), pad_arch_comparison (p), nuaa_session_ablation_run (sr), nuaa_folds_run
    (fr) e nuaa_bn_recalibration (br), ripristinati all'uscita."""
    mods = ((R, r), (PAC, p), (SR, sr), (FR, fr), (BR, br))
    saved = [(m, k, getattr(m, k)) for m, d in mods for k in (d or {})]
    try:
        for m, d in mods:
            for k, v in (d or {}).items():
                setattr(m, k, v)
        yield
    finally:
        for m, k, v in saved:
            setattr(m, k, v)


def base_hooks(w: World, **r) -> dict:
    """Stato git, macchina e versioni uguali al record della prova, i SHA-256 dello schema e delle celle del mondo, frame 8×8."""
    return {"r": {"REPO_STATE": lambda root: dict(STATE), "MACHINE": lambda: dict(MACH), "ENVIRONMENT": lambda: dict(ENV), **r},
            "sr": {"SCHEMA_CONFIG_SHA256": w.schema_sha, "CELLS_CONFIG_SHA256": w.cells_cfg_sha, "peak_rss_mb": lambda: 2048.0},
            "fr": {"FRAME_SHAPE": CPA.SHAPE}}


# ============================================================================ TensorFlow e modelli finti (flusso senza TensorFlow)

class FakeTF:
    """Il TensorFlow finto del flusso: thread (CPA.FakeThreading; offset sposta il getter intra-op, inter_offset quello
    inter-op, raise_on_set simula TensorFlow già inizializzato) e dispositivi per FR.hide_gpu (registrati nel log)."""

    def __init__(self, offset=0, log=None, inter_offset=0, raise_on_set=False):
        self.threading = CPA.FakeThreading(offset=offset, raise_on_set=raise_on_set)
        self.log = log if log is not None else []
        th = self.threading
        orig_intra, orig_inter = th.set_intra_op_parallelism_threads, th.set_inter_op_parallelism_threads
        th.set_intra_op_parallelism_threads = lambda n: self.log.append(f"intra {n}") or orig_intra(n)
        th.set_inter_op_parallelism_threads = lambda n: self.log.append(f"inter {n}") or orig_inter(n)
        if inter_offset:
            th.get_inter_op_parallelism_threads = lambda: th.vals["inter"] + inter_offset
        self.config = types.SimpleNamespace(threading=th, set_visible_devices=lambda d, k: self.log.append(f"visible {k} {d}"),
                                            get_visible_devices=lambda: [types.SimpleNamespace(device_type="CPU")],
                                            optimizer=types.SimpleNamespace(set_experimental_options=lambda o: None))


class FakeModel:
    def __init__(self, network, fold, seed):
        self.tag = f"{network} {fold} {seed}"

    def save(self, path):
        Path(path).write_bytes(f"fake model {self.tag}\n".encode())


def fake_trainer(log: list, epochs: int = EPOCHS, on_epoch=None):
    """Un trainer con la firma di PAC.trial_trainer: storia fissata (val_loss minimo all'ultima epoca, uguale al val_loss
    dello scorer finto sul dev), epoche registrate nel timer, cap_check a ogni epoca."""
    def trainer(data, training, network, seed, build_model, cap_check, timer):
        y = np.asarray(data["y_dev"])
        log.append(("train", network, seed, len(data["y_fit"]), len(y)))
        h = fake_history(y, network, epochs)
        for i in range(epochs):
            timer.record.append({"epoch": i + 1, **{k: h[k][i] for k in R.HISTORY_KEYS}, "seconds_total": 0.1, "seconds_validation": 0.01, "cpu_seconds": 0.2})
            if on_epoch:
                on_epoch(network, seed, i + 1)
            cap_check(f"epoch {i + 1}")
        best = int(np.argmin(h["val_loss"]))
        card = {"arch": network, "train_seed": seed, "n_fitting": len(data["y_fit"]), "n_dev": len(y), "max_epochs": training["max_epochs"],
                "epochs_run": epochs, "restored_epoch": best + 1, "restored_by": "end of epochs (set_weights)", "best_val_loss": h["val_loss"][best],
                "val_acc_at_best": h["val_accuracy"][best], "history": h, "device": "CPU", "visible_devices": ["CPU"], "lr": training["initial_learning_rate"],
                "batch": training["batch"], "early_stopping_patience": training["early_stopping_patience"], "scheduler": training["scheduler"],
                "class_weight": {"0": 1.0, "1": 1.0}, "train_seconds": 0.3, "test_evaluation": "none", "cpu_nondeterminism": "declared",
                "checkpoint": "none saved (trial)"}
        return card, FakeModel(network, data["y_fit"].shape, seed)
    return trainer


def fake_probs(path, n: int) -> np.ndarray:
    c = SCORE["LivenessNet" if Path(path).name.startswith("LivenessNet") else "ResNet-18"]
    return np.stack([np.full(n, 1 - c), np.full(n, c)], axis=1)


def fake_scorer(path, sha256):
    if FR.sha256_file(path) != sha256:
        raise FR.Failure("SHA-256 changed")
    return lambda X: fake_probs(path, len(X))


def fake_logits(path, sha256):
    return lambda X: np.log(fake_probs(path, len(X)))


def lz(p) -> np.ndarray:
    p = np.asarray(p, np.float64)
    return np.log(np.stack([1 - p, p], axis=1))


def fake_reestimate(log: list = None, p_over=None, ordinary_over=None):
    """La fase 2 finta: stato ordinario uguale allo scorer finto (salvo ordinary_over), statistiche di forma corretta,
    punteggi del dev con la BN ristimata variabili per frame."""
    def re(ctx, rec, fit, dev):
        if log is not None:
            log.append(("re-estimate", rec["name"]))
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
            stats[layer] = {"spatial": 4, "frames": len(fit["y"]), "count": 4 * len(fit["y"]), "mean64": z2, "var64": z2 + 1, "old_mean": z2.astype(np.float32),
                            "old_var": (z2 + 1).astype(np.float32), "mean32": z2.astype(np.float32), "var32": (z2 + 1).astype(np.float32)}
            cons[layer] = {"max_abs_diff_mean": 0.0, "max_abs_diff_var": 0.0, "count": 4 * len(fit["y"]), "channels_outside_tolerance": 0}
        return {"stats": stats, "cons": cons, "changes": {k: {"var_ratio_min": 1.0} for k in stats}, "re_estimated": {"p": p, "z": lz(p)},
                "ordinary": {"p": po, "z": lz(po)}, "fitting_after": {"accuracy": 0.5, "val_loss": 0.7, "auroc": 0.5, "n": len(fit["y"])}}
    return re


def flow_hooks(w: World, log=None, tf=None, **r):
    log = log if log is not None else []
    tfm = tf or FakeTF(log=log)
    h = base_hooks(w, **{"TF_MODULE": lambda: tfm, "ARCH_LOADER": lambda path: None, "TRAINER": fake_trainer(log), "MAKE_SCORER": fake_scorer,
                         "REESTIMATE": fake_reestimate(log), **r})
    h["sr"]["MAKE_LOGITS"] = fake_logits
    return h


def run_main(argv, hooks: dict) -> tuple:
    out, err = io.StringIO(), io.StringIO()
    with injected(**hooks), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = R.main(argv)
    return rc, out.getvalue(), err.getvalue()


def run_folder(w: World, run="nuaa_patience_4", attempt=1) -> Path:
    return w.runs_out / (f"{DATE}-ARCHRUN-{R.RUN_TAG[run]}-{STATE['commit']}" + ("" if attempt == 1 else f"-r{attempt}"))


def tree(folder: Path) -> dict:
    return {str(p.relative_to(folder)): sha(p.read_bytes()) for p in sorted(folder.rglob("*")) if p.is_file()} if folder.exists() else {}


@contextlib.contextmanager
def altered_world(alter: dict = None, trial_over=None):
    """Un'alterazione del mondo (schede o prova) applicata e poi tolta, con tutte le impronte riallineate."""
    w = world()
    w.alter = alter or {}
    if trial_over:
        saved = World.write_trial
        World.write_trial = lambda self, status="PASS", over=None: saved(self, status, {**(over or {}), **trial_over})
    try:
        w.refresh()
        yield w
    finally:
        if trial_over:
            World.write_trial = saved
        w.alter = {}
        w.refresh()


THREAD_ENV_VARS = ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")


@contextlib.contextmanager
def thread_env(**values):
    """Le due variabili dei thread nell'ambiente: solo quelle date (le altre tolte), poi ripristinate."""
    saved = {k: os.environ.get(k) for k in THREAD_ENV_VARS}
    for k in THREAD_ENV_VARS:
        os.environ.pop(k, None)
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


@contextlib.contextmanager
def pre_registration_threads(w: World, **threads):
    """La configurazione della preregistrazione con training.threads alterato, fissata di nuovo nella configurazione di
    questo runner; tutto rimesso a posto all'uscita."""
    saved = w.cfg_path.read_bytes()
    d = w.run_doc()
    doc = json.loads(saved)
    doc["training"]["threads"].update(threads)
    try:
        w.cfg_path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        d["preregistration"]["sha256"] = sha(w.cfg_path.read_bytes())
        w.write_run_cfg(d)
        yield w
    finally:
        w.cfg_path.write_bytes(saved)
        w.write_run_cfg(w.run_doc())
        w.write_estimator_record()


# ============================================================================ A configurazione

class TestAConfig(unittest.TestCase):
    def setUp(self):
        self.w = world()
        self.doc = self.w.run_doc()
        self.addCleanup(self.w.write_run_cfg, self.w.run_doc())

    def refused(self, mutate, needle: str):
        d = json.loads(json.dumps(self.doc))
        mutate(d)
        with self.assertRaisesRegex(FR.Failure, needle):
            R.load_config(self.w.write_run_cfg(d))

    def test_synthetic_configuration_accepted(self):
        cfg, s = R.load_config(self.w.write_run_cfg(self.doc))
        self.assertEqual(s, sha(self.w.run_cfg.read_bytes()))
        self.assertEqual(cfg["estimator"]["orders"], R.expected_orders())

    def test_closed_sets_no_default(self):
        """Nessun valore predefinito: ogni campo mancante o in più, a ogni livello, è rifiutato."""
        self.refused(lambda d: d.pop("after_stop"), "fields must be exactly")
        self.refused(lambda d: d.update(extra=1), "fields must be exactly")
        for sec in ("preregistration", "trial_record", "plan_files", "estimator_check", "machine_rule", "sequence", "checkpoint", "history_check",
                    "estimator", "dev_evaluation", "cap", "outputs", "choices_of_the_author"):
            k = sorted(self.doc[sec])[0]
            self.refused(lambda d, s=sec, k=k: d[s].pop(k), f"{sec}")
            self.refused(lambda d, s=sec: d[s].update(unknown="x"), f"{sec}")
        self.refused(lambda d: d["code"].pop(R.CODE_FILES[0]), "code needs exactly")

    def test_values_fixed(self):
        self.refused(lambda d: d.update(schema="pad-arch-run-config/0"), "schema must be")
        self.refused(lambda d: d.update(test=d["test"] + " "), "test must state exactly")
        self.refused(lambda d: d.update(runs=["nuaa_patience_12", "nuaa_patience_4"]), "runs must be")
        self.refused(lambda d: d["machine_rule"].update(fields=list(R.MACHINE_FIELDS[:-1])), "machine_rule must compare")
        self.refused(lambda d: d["machine_rule"].update(versions=list(R.VERSION_FIELDS) + ["opencv"]), "machine_rule must compare")
        self.refused(lambda d: d["machine_rule"].update(threads=["effective"]), "machine_rule must compare")
        self.refused(lambda d: d["machine_rule"].update(consequence=author("reported")), "before any training")
        self.refused(lambda d: d["history_check"].update(difference=1e-9), "difference 0")
        self.refused(lambda d: d["history_check"].update(difference=False), "difference 0")
        self.refused(lambda d: d["history_check"].update(fields=["loss"]), "history_check must compare")
        self.refused(lambda d: d["history_check"].update(consequence="the run stops"), "stays in the comparison")
        self.refused(lambda d: d["estimator"]["orders"]["ResNet-18"].reverse(), "estimator.orders")
        self.refused(lambda d: d["estimator"]["orders"]["LivenessNet"].pop(), "estimator.orders")
        self.refused(lambda d: d["estimator"].update(batch=32), "estimator.batch")
        self.refused(lambda d: d["estimator"].update(batch=True), "estimator.batch")
        self.refused(lambda d: d["dev_evaluation"].update(states=["ordinary"]), "dev_evaluation.states")
        self.refused(lambda d: d["cap"].update(unit="hours of CPU"), "cap.unit")
        self.refused(lambda d: d["cap"].update(on_cap="stops"), "INCOMPLETE")
        self.refused(lambda d: d["checkpoint"].update(when="w"), "CHECKPOINT_TEXT")
        self.refused(lambda d: d["outputs"].update(public=list(R.PUBLIC_OUTPUTS[:-1])), "outputs must list")
        self.refused(lambda d: d["trial_record"].update(run_id="20261119-SESSABLRUN-b8f7cca"), "ARCHTRIAL")
        self.refused(lambda d: d["trial_record"].update(status_sha256="x"), "malformed")
        self.refused(lambda d: d["preregistration"].update(sha256="A" * 64), "malformed")
        self.refused(lambda d: d["estimator_check"].update(bnrecal_run_id="x"), "BNRECAL run id")
        self.refused(lambda d: d["estimator_check"].update(models=0), "positive integer")
        self.refused(lambda d: d["estimator_check"].update(rule="compared"), "stops")
        self.refused(lambda d: d["code"].update({R.CODE_FILES[0]: "z" * 64}), "code SHA-256 values malformed")
        self.refused(lambda d: d["choices_of_the_author"].update(sequence=""), "non-empty text")

    def test_choices_declared_as_the_author(self):
        """Ogni scelta che la preregistrazione non fissa è dichiarata «choice of the author» nel suo campo."""
        for path in (("sequence", "reason"), ("estimator", "order_rule"), ("estimator_check", "where"), ("estimator_check", "cap_rule"),
                     ("dev_evaluation", "evaluator"), ("dev_evaluation", "agreement"), ("machine_rule", "consequence"), ("machine_rule", "scope"),
                     ("history_check", "first_differing_epoch"), ("history_check", "non_finite")):
            self.refused(lambda d, p=path: d[p[0]].update({p[1]: "chosen, no threshold.json, before any training, 1e-5, MISMATCH"}),
                         "choice of the author")
        self.refused(lambda d: d.update(after_stop="no automatic resume"), "choice of the author")
        self.refused(lambda d: d.update(after_stop=author("resume")), "no automatic resume")
        self.refused(lambda d: d.update(attempts="--unofficial-out-root"), "choice of the author")
        self.refused(lambda d: d.update(attempts=author("attempts")), "unofficial output root")

    def test_texts_required(self):
        """Le frasi che la configurazione deve contenere (testi delle correzioni della revisione)."""
        self.refused(lambda d: d.update(authorisation="after the comment of 19 November"), "merge and the go-ahead")
        self.refused(lambda d: d.update(operations="commit"), "nohup")
        self.refused(lambda d: d["machine_rule"].update(scope=author("same machine")), "not the identity of the physical unit")
        self.refused(lambda d: d["sequence"].update(reason=author("same operations")), "threshold.json")
        self.refused(lambda d: d["dev_evaluation"].update(frame_ids="outputs hold only counts"), "public outputs")
        self.refused(lambda d: d["dev_evaluation"].update(agreement=author("binding")), "MISMATCH")
        self.refused(lambda d: d["cap"].update(on_cap="stops as INCOMPLETE"), "discarded at the cap")
        self.refused(lambda d: d["estimator_check"].update(cap_unit="hours"), "cap_unit")
        self.refused(lambda d: d["estimator_check"].update(cap_minutes=0), "cap_minutes")
        self.refused(lambda d: d["trial_record"]["package_sums"].update(sha256="x"), "package_sums.sha256 malformed")

    def test_delivered_configuration(self):
        p = ROOT / R.DEFAULT_CONFIG
        if not p.is_file():
            skip_missing(self, "delivered configuration")
        cfg, _ = R.load_config(p)
        self.assertEqual(cfg["code"], {f: sha((ROOT / f).read_bytes()) for f in R.CODE_FILES})
        self.assertTrue(R.TEST_READ.startswith(R.TEST_SENTENCE))
        self.assertNotIn("loaded as an image", json.dumps(cfg))
        text = json.dumps(cfg).lower()
        for banned in (r"\bclaude\b", r"\bai\b", r"\bassistant\b", r"\bserver\b", r"/users/", r"\bshe\b", r"\bhe\b", r"\bher\b",
                       r"\bhis\b", r"\bhim\b"):
            self.assertNotRegex(text, banned)
        for k, v in cfg["choices_of_the_author"].items():
            self.assertTrue(v.strip(), k)


# ============================================================================ B file fissati, ordine e tetti

class TestBPins(unittest.TestCase):
    def setUp(self):
        self.w = world()

    def pins(self):
        cfg, _ = R.load_config(self.w.run_cfg)
        with injected(**base_hooks(self.w)):
            return R.load_pins(cfg, self.w.base)

    def test_pins_pass(self):
        p = self.pins()
        self.assertEqual(list(p.positions), list(R.RUNS))
        for run, pat in R.RUNS.items():
            self.assertEqual([(x["network"], x["fold"], x["seed"]) for x in p.positions[run]], PAC.plan_order([1, 2, 3, 4], [42, 43, 44]))
            self.assertTrue(all(x["patience"] == pat for x in p.positions[run]))
        self.assertEqual(p.trial["run_id"], CPA.RUN)

    def refused(self, needle, plan=None, doc_mutate=None):
        """Un'alterazione dei file del piano (plan: nome -> testo nuovo, con l'impronta aggiornata nella configurazione) o
        della configurazione (doc_mutate) deve fermare load_pins con il messaggio atteso; tutto è poi rimesso a posto."""
        w = self.w
        files = [w.base / "run_order.csv", w.base / "run_caps.csv"]
        saved = {q: q.read_bytes() for q in files}
        try:
            d = w.run_doc()
            for name, text in (plan or {}).items():
                (w.base / f"{name}.csv").write_text(text, encoding="utf-8")
                d["plan_files"][name] = {"path": f"{name}.csv", "sha256": sha((w.base / f"{name}.csv").read_bytes())}
            if doc_mutate:
                doc_mutate(d)
            w.write_run_cfg(d)
            with self.assertRaisesRegex(FR.Failure, needle):
                self.pins()
        finally:
            for q, b in saved.items():
                q.write_bytes(b)
            w.write_run_cfg(w.run_doc())
            w.write_estimator_record()

    def test_order_different_from_the_plan_stops(self):
        """Ordine diverso dal piano: due posizioni scambiate in run_order.csv (con l'impronta aggiornata) -> stop."""
        text = (self.w.base / "run_order.csv").read_text()
        lines = text.splitlines(keepends=True)
        swapped = lines[:1] + [lines[2], lines[1]] + lines[3:]
        self.refused("differs from the order regenerated", plan={"run_order": "".join(swapped)})
        # anche senza passare dalla rigenerazione: le posizioni lette dal file sono controllate contro PAC.plan_order
        cfg_pr = json.loads(self.w.cfg_path.read_text())
        with self.assertRaisesRegex(FR.Failure, "position 1 of nuaa_patience_4 is"):
            R.run_positions("".join(swapped).encode(), cfg_pr, "nuaa_patience_4")
        renumbered = text.replace("nuaa_patience_4,1,LivenessNet", "nuaa_patience_4,0,LivenessNet")
        with self.assertRaisesRegex(FR.Failure, "with position 0"):
            R.run_positions(renumbered.encode(), cfg_pr, "nuaa_patience_4")
        with self.assertRaisesRegex(FR.Failure, "23 positions"):
            R.run_positions("".join(lines[:1] + lines[2:]).encode(), cfg_pr, "nuaa_patience_4")
        ref = text.replace("nuaa_patience_4,1,LivenessNet,1,42,4," + CPA.P4, "nuaa_patience_4,1,LivenessNet,1,42,4," + CPA.P12)
        with self.assertRaisesRegex(FR.Failure, "reference run"):
            R.run_positions(ref.encode(), cfg_pr, "nuaa_patience_4")

    def test_caps_read_not_rewritten(self):
        text = (self.w.base / "run_caps.csv").read_text()
        self.refused("caps regenerated", plan={"run_caps": text.replace("nuaa_patience_4,NUAA,4,24,", "nuaa_patience_4,NUAA,4,24,0")})
        self.assertGreater(R.run_cap_hours(text.encode(), "nuaa_patience_4"), 0)
        for bad, needle in (("x", "not a number"), ("nan", "finite positive"), ("0.00", "finite positive")):
            rows = text.splitlines()
            head, row = rows[0], rows[1].split(",")
            row[PAC.CAPS_COLUMNS.index("cap_wall_hours")] = bad
            with self.assertRaisesRegex(FR.Failure, needle):
                R.run_cap_hours((head + "\n" + ",".join(row) + "\n").encode(), "nuaa_patience_4")
        with self.assertRaisesRegex(FR.Failure, "0 rows"):
            R.run_cap_hours(text.encode(), "nuaa_patience_8")

    def test_pinned_files(self):
        self.refused("pre-registration configuration .*SHA-256", doc_mutate=lambda d: d["preregistration"].update(sha256="0" * 64))
        self.refused("run_order.csv .*SHA-256", doc_mutate=lambda d: d["plan_files"]["run_order"].update(sha256="0" * 64))
        self.refused("run_caps.csv .*SHA-256", doc_mutate=lambda d: d["plan_files"]["run_caps"].update(sha256="0" * 64))
        self.refused("trial/trial.json of the delivered package", doc_mutate=lambda d: d["trial_record"].update(trial_json_sha256="0" * 64))
        self.refused("trial/status.json of the delivered package", doc_mutate=lambda d: d["trial_record"].update(status_sha256="0" * 64))
        # il file della prova cambiato mentre configurazione e pacchetto lo nominano ancora: lo SHA-256 del file è controllato
        for fn in ("trial.json", "status.json"):
            q = self.w.trial_dir / fn
            saved = q.read_bytes()
            try:
                self.w.write_run_cfg(self.w.run_doc())
                q.write_bytes(saved + b" ")
                with self.assertRaisesRegex(FR.Failure, f"trial record {fn}: SHA-256"):
                    self.pins()
            finally:
                q.write_bytes(saved)
                self.w.write_run_cfg(self.w.run_doc())
                self.w.write_estimator_record()
        self.refused("PR #27 configuration", doc_mutate=lambda d: d["estimator_check"]["bn_recalibration_config"].update(sha256="0" * 64))
        self.refused("PR #28 configuration", doc_mutate=lambda d: d["estimator_check"]["bn_test_config"].update(sha256="0" * 64))
        self.refused("code files", doc_mutate=lambda d: d["code"].update({R.CODE_FILES[-1]: "0" * 64}))
        self.refused("requires PASS", doc_mutate=lambda d: d["trial_record"].update(run_id="20261119-ARCHTRIAL-1111111"))

    def test_trial_record_anchored_to_the_package(self):
        """Il record della prova deve coincidere con le righe trial/status.json e trial/trial.json del SHA256SUMS del
        pacchetto consegnato (fissato a sua volta)."""
        w = self.w
        try:
            d = w.run_doc()
            w.package_sums({"trial/trial.json": "2" * 64})
            d["trial_record"]["package_sums"]["sha256"] = sha((w.base / "package_SHA256SUMS").read_bytes())
            w.write_run_cfg(d)
            with self.assertRaisesRegex(FR.Failure, "trial/trial.json of the delivered package"):
                self.pins()
            d = w.run_doc()
            d["trial_record"]["package_sums"]["sha256"] = "0" * 64
            w.write_run_cfg(d)
            with self.assertRaisesRegex(FR.Failure, "package SHA256SUMS"):
                self.pins()
        finally:
            w.write_run_cfg(w.run_doc())
            w.write_estimator_record()

    def test_estimator_cap_is_the_cap_of_pr27(self):
        self.refused("cap of the PR #27 configuration", doc_mutate=lambda d: d["estimator_check"].update(cap_minutes=241))

    def test_trial_not_pass_refused(self):
        with altered_world(trial_over={"status": "INCOMPLETE"}) as w:
            (w.trial_dir / "status.json").write_text(json.dumps({"run_id": CPA.RUN, "status": "INCOMPLETE",
                                                                 "trial_json_sha256": sha((w.trial_dir / "trial.json").read_bytes())}))
            w.write_run_cfg(w.run_doc())
            with self.assertRaisesRegex(FR.Failure, "requires PASS"):
                self.pins()

    def test_order_against_pr27(self):
        """L'ordine di LivenessNet della configurazione deve essere quello della configurazione della PR #27."""
        d = bn_doc()
        d["architecture"]["layers"][3], d["architecture"]["layers"][6] = ["bn1_1x", "BatchNormalization"], ["bn1_2", "BatchNormalization"]
        d["batchnorm"]["layers"][0] = "bn1_1x"
        d["batchnorm"]["kinds"] = {("bn1_1x" if k == "bn1_1" else k): v for k, v in d["batchnorm"]["kinds"].items()}
        d["batchnorm"]["kinds"] = {k: d["batchnorm"]["kinds"][k] for k in d["batchnorm"]["layers"]}
        saved = self.w.br_cfg.read_bytes()
        try:
            self.w.br_cfg.write_text(json.dumps(d), encoding="utf-8")
            self.w.write_run_cfg(self.w.run_doc())
            with self.assertRaisesRegex(FR.Failure, "LivenessNet order or the batch"):
                self.pins()
        finally:
            self.w.br_cfg.write_bytes(saved)
            self.w.write_run_cfg(self.w.run_doc())
            self.w.write_estimator_record()


# ============================================================================ C regola della macchina e dei thread

class TestCMachineRule(unittest.TestCase):
    def setUp(self):
        self.trial = {"run_id": CPA.RUN, "machine": dict(MACH), "environment": dict(ENV),
                      "threads": {"requested": {"intra_op": 10, "inter_op": 10}, "effective": {"intra_op": 10, "inter_op": 10},
                                  "set_by_this_runner": True, "environment": dict(CPA.NO_THREAD_ENV), "logical_cpus": 15}}

    def rule(self, m=None, e=None):
        with injected(r={"MACHINE": lambda: m or dict(MACH), "ENVIRONMENT": lambda: e or dict(ENV)}):
            return R.machine_rule(self.trial)

    def test_same_machine_passes_and_is_recorded(self):
        rec = self.rule()
        self.assertTrue(rec["equal"])
        self.assertEqual(sorted(rec["machine"]), sorted(R.MACHINE_FIELDS))
        self.assertTrue(all(v["equal"] for v in rec["versions"].values()))

    def test_different_machine_stops(self):
        for f, v in (("processor", "Apple M4"), ("logical_cpus", 10), ("ram_gb", 16.0), ("os", "macOS 27.3"), ("architecture", "x86_64"),
                     ("core_levels", [{"name": "Performance", "physical_cores": 4}])):
            with self.assertRaisesRegex(FR.Failure, f"machine rule: {f}"):
                self.rule(m={**MACH, f: v})
        m = dict(MACH)
        del m["ram_gb"]
        with self.assertRaisesRegex(FR.Failure, "machine rule: ram_gb"):
            self.rule(m=m)

    def test_null_or_missing_on_both_sides_stops(self):
        """Un campo assente o nullo da entrambe le parti non vale come uguale."""
        for f in ("ram_gb", "core_levels"):
            m = {**MACH, f: None}
            t = dict(self.trial, machine={**MACH, f: None})
            with injected(r={"MACHINE": lambda: m, "ENVIRONMENT": lambda: dict(ENV)}):
                with self.assertRaisesRegex(FR.Failure, f"machine rule: {f}"):
                    R.machine_rule(t)
        m = {k: v for k, v in MACH.items() if k != "os"}
        t = dict(self.trial, machine=dict(m))
        with injected(r={"MACHINE": lambda: m, "ENVIRONMENT": lambda: dict(ENV)}):
            with self.assertRaisesRegex(FR.Failure, "machine rule: os"):
                R.machine_rule(t)
        e = {k: v for k, v in ENV.items() if k != "keras"}
        t = dict(self.trial, environment=dict(e))
        with injected(r={"MACHINE": lambda: dict(MACH), "ENVIRONMENT": lambda: e}):
            with self.assertRaisesRegex(FR.Failure, "machine rule: keras"):
                R.machine_rule(t)

    def test_different_versions_stop(self):
        for f in R.VERSION_FIELDS:
            with self.assertRaisesRegex(FR.Failure, f"machine rule: {f}"):
                self.rule(e={**ENV, f: "0.0"})

    def test_threads(self):
        th = {k: self.trial["threads"][k] for k in R.THREAD_FIELDS}
        rule = R.threads_rule(self.trial, th)
        self.assertTrue(rule["equal"])
        R.require_threads(rule)
        for f, v in (("effective", {"intra_op": 10, "inter_op": 9}), ("requested", {"intra_op": 8, "inter_op": 10}), ("set_by_this_runner", False),
                     ("environment", {"TF_NUM_INTRAOP_THREADS": "10", "TF_NUM_INTEROP_THREADS": None})):
            rule = R.threads_rule(self.trial, {**th, f: v})
            self.assertFalse(rule["equal"])
            self.assertFalse(rule["fields"][f]["equal"])
            with self.assertRaisesRegex(FR.Failure, f"thread rule: {f}"):
                R.require_threads(rule)

    def test_threads_missing_or_null_on_both_sides(self):
        th = {k: self.trial["threads"][k] for k in R.THREAD_FIELDS}
        for f in ("requested", "effective", "set_by_this_runner"):
            t = dict(self.trial, threads={**self.trial["threads"], f: None})
            self.assertFalse(R.threads_rule(t, {**th, f: None})["equal"], f)
        t = dict(self.trial, threads={k: v for k, v in self.trial["threads"].items() if k != "environment"})
        self.assertFalse(R.threads_rule(t, {k: v for k, v in th.items() if k != "environment"})["equal"])


# ============================================================================ D storia di LivenessNet

def ref_of(h: dict) -> dict:
    return {"card": {"history": h}, "model": "arm_b_fold1_seed42", "sha256": "0" * 64}


class TestDHistory(unittest.TestCase):
    def setUp(self):
        self.h = fake_history([0, 1, 0, 1], "LivenessNet", 5)

    def test_identical(self):
        r = R.history_check(ref_of(self.h), {"history": json.loads(json.dumps(self.h))})
        self.assertEqual((r["outcome"], r["first_differing_epoch"], r["largest_abs_difference"], r["epoch_of_largest_difference"]), ("identical", None, 0.0, None))
        self.assertEqual(r["epochs_compared"], 5)

    def test_different_first_epoch_and_largest(self):
        n = json.loads(json.dumps(self.h))
        n["val_loss"][3] += 0.5
        n["loss"][1] += 1e-9
        r = R.history_check(ref_of(self.h), {"history": n})
        self.assertEqual(r["outcome"], "different")
        self.assertEqual(r["first_differing_epoch"], 2)
        self.assertEqual(r["field_of_largest_difference"], "val_loss")
        self.assertEqual(r["epoch_of_largest_difference"], 4)
        self.assertAlmostEqual(r["largest_abs_difference"], 0.5, places=12)
        self.assertEqual(r["fields"]["loss"]["first_differing_epoch"], 2)

    def test_different_number_of_epochs(self):
        """Epoche in comune identiche, numeri diversi: «different», prima epoca diversa = min(epoche) + 1; dettaglio per
        grandezza e i due numeri di epoche riportati."""
        n = {k: v[:3] for k, v in self.h.items()}
        r = R.history_check(ref_of(self.h), {"history": n})
        self.assertEqual((r["outcome"], r["first_differing_epoch"], r["epochs_compared"], r["epochs_new"], r["epochs_reference"]),
                         ("different", 4, 3, 3, 5))
        self.assertTrue(all(r["fields"][k]["first_differing_epoch"] is None for k in R.HISTORY_KEYS))
        r = R.history_check(ref_of(n), {"history": self.h})
        self.assertEqual((r["first_differing_epoch"], r["epochs_new"], r["epochs_reference"]), (4, 5, 3))
        m = json.loads(json.dumps(n))
        m["loss"][1] += 0.1
        self.assertEqual(R.history_check(ref_of(self.h), {"history": m})["first_differing_epoch"], 2)

    def test_non_finite_does_not_stop(self):
        """Un valore non finito nella storia nuova: esito «different» con la ragione, nessun Failure."""
        n = json.loads(json.dumps(self.h))
        n["lr"][2] = float("nan")
        n["val_loss"][3] += 0.25
        r = R.history_check(ref_of(self.h), {"history": n})
        self.assertEqual(r["outcome"], "different")
        self.assertEqual(r["first_differing_epoch"], 3)
        self.assertEqual(r["fields"]["lr"]["non_finite"], ["non-finite value in lr, new history, epoch 3"])
        self.assertEqual(r["non_finite"], ["non-finite value in lr, new history, epoch 3"])
        self.assertEqual(r["field_of_largest_difference"], "val_loss")
        n = json.loads(json.dumps(self.h))
        n["loss"][0] = float("inf")
        r = R.history_check(ref_of(self.h), {"history": n})
        self.assertEqual((r["outcome"], r["first_differing_epoch"]), ("different", 1))
        ref = json.loads(json.dumps(self.h))
        ref["accuracy"][1] = float("nan")
        r = R.history_check(ref_of(ref), {"history": self.h})
        self.assertEqual(r["fields"]["accuracy"]["non_finite"], ["non-finite value in accuracy, reference history, epoch 2"])

    def test_non_finite_does_not_hide_an_earlier_difference(self):
        """loss [1, 2.5, 3, NaN] contro [1, 2, 3, 4]: prima epoca diversa 2 (non 4), scarto massimo 0,5 all'epoca 2, il valore
        non finito riportato a parte."""
        ref = {k: [1.0, 2.0, 3.0, 4.0] for k in R.HISTORY_KEYS}
        new = {**{k: list(v) for k, v in ref.items()}, "loss": [1.0, 2.5, 3.0, float("nan")]}
        r = R.history_check(ref_of(ref), {"history": new})
        self.assertEqual((r["outcome"], r["first_differing_epoch"]), ("different", 2))
        self.assertEqual((r["largest_abs_difference"], r["field_of_largest_difference"], r["epoch_of_largest_difference"]), (0.5, "loss", 2))
        self.assertEqual(r["fields"]["loss"]["epochs_compared"], 3)
        self.assertEqual(r["non_finite"], ["non-finite value in loss, new history, epoch 4"])

    def test_difference_not_representable(self):
        """Due valori finiti di segno opposto con differenza non finita (1e308 e -1e308): esito «different» con la ragione,
        nessun Failure; le epoche precedenti confrontate."""
        ref = {k: [1.0, 1e308, 3.0] for k in R.HISTORY_KEYS}
        new = {**{k: list(v) for k, v in ref.items()}, "val_loss": [1.5, -1e308, 3.0]}
        r = R.history_check(ref_of(ref), {"history": new})
        self.assertEqual((r["outcome"], r["first_differing_epoch"]), ("different", 1))
        self.assertEqual(r["non_finite"], ["difference not representable in val_loss at epoch 2"])
        self.assertEqual((r["largest_abs_difference"], r["field_of_largest_difference"]), (0.5, "val_loss"))
        new2 = {**{k: list(v) for k, v in ref.items()}, "loss": [1.0, -1e308, 3.0]}
        r = R.history_check(ref_of(ref), {"history": new2})
        self.assertEqual((r["outcome"], r["first_differing_epoch"], r["largest_abs_difference"]), ("different", 2, 0.0))
        self.assertEqual(r["fields"]["loss"]["epochs_compared"], 1)

    def test_largest_difference_tie(self):
        """A pari scarto il campo riportato è il primo in HISTORY_KEYS."""
        n = json.loads(json.dumps(self.h))
        n["accuracy"][1] += 0.5
        n["val_accuracy"][2] += 0.5
        r = R.history_check(ref_of(self.h), {"history": n})
        self.assertEqual((r["field_of_largest_difference"], r["epoch_of_largest_difference"]), ("accuracy", 2))
        n = json.loads(json.dumps(self.h))
        n["val_accuracy"][2] += 0.5
        n["accuracy"][1] += 0.25
        self.assertEqual(R.history_check(ref_of(self.h), {"history": n})["field_of_largest_difference"], "val_accuracy")

    def test_timer_history_comparison(self):
        h = fake_history([0, 1], "LivenessNet", 2)
        ep = [{**{k: h[k][i] for k in R.HISTORY_KEYS}} for i in range(2)]
        self.assertTrue(R.same_history_records(ep, h, 2))
        ep[1]["loss"] += 1e-12
        self.assertFalse(R.same_history_records(ep, h, 2))
        h2 = {**h, "loss": [float("nan"), h["loss"][1]]}
        ep2 = [{**{k: h2[k][i] for k in R.HISTORY_KEYS}} for i in range(2)]
        self.assertTrue(R.same_history_records(ep2, h2, 2))


# ============================================================================ E ordine della ristima

def chain_topology(names_bn) -> dict:
    t = {"input_layer": ("InputLayer", (), ())}
    prev = "input_layer"
    for i, n in enumerate(names_bn):
        t[f"conv{i}"] = ("Conv2D", (prev,), ())
        t[n] = ("BatchNormalization", (f"conv{i}",), ())
        prev = n
    t["output"] = ("Dense", (prev,), ())
    return t


class TestEBnOrder(unittest.TestCase):
    def test_resnet18_order_is_topological(self):
        topo = AM.resnet18_topology()
        order = R.expected_orders()["ResNet-18"]
        self.assertEqual(len(order), AM.RESNET18_BN_LAYERS)
        self.assertEqual(R.bn_order_problems(order, topo), [])
        self.assertEqual(R.bn_ancestors(topo, "bn1"), set())
        self.assertEqual(R.bn_ancestors(topo, "layer2_block1_downsample_bn"), R.bn_ancestors(topo, "layer2_block1_bn1"))
        self.assertIn("layer1_block2_bn2", R.bn_ancestors(topo, "layer2_block1_downsample_bn"))
        self.assertNotIn("layer2_block1_bn2", R.bn_ancestors(topo, "layer2_block1_downsample_bn"))
        self.assertNotIn("layer2_block1_downsample_bn", R.bn_ancestors(topo, "layer2_block1_bn2"))

    def test_independent_layers_in_any_order(self):
        topo = AM.resnet18_topology()
        order = R.expected_orders()["ResNet-18"]
        for s in (2, 3, 4):
            o = list(order)
            i, j = o.index(f"layer{s}_block1_bn2"), o.index(f"layer{s}_block1_downsample_bn")
            o[i], o[j] = o[j], o[i]
            self.assertEqual(R.bn_order_problems(o, topo), [])
            o2 = list(order)
            k = o2.index(f"layer{s}_block1_bn1")
            o2.insert(k, o2.pop(o2.index(f"layer{s}_block1_downsample_bn")))
            self.assertEqual(R.bn_order_problems(o2, topo), [])

    def test_dependent_layers_refused(self):
        topo = AM.resnet18_topology()
        order = R.expected_orders()["ResNet-18"]
        o = [order[1], order[0]] + order[2:]
        self.assertIn("layer1_block1_bn1 comes before bn1", "; ".join(R.bn_order_problems(o, topo)))
        o = list(order)
        o.insert(o.index("layer2_block1_bn1"), o.pop(o.index("layer2_block1_bn2")))
        self.assertTrue(R.bn_order_problems(o, topo))
        self.assertTrue(R.bn_order_problems(order[:-1], topo))
        self.assertTrue(R.bn_order_problems(order + [order[0]], topo))
        self.assertTrue(R.bn_order_problems(order[:-1] + ["other"], topo))

    def test_livenessnet_chain(self):
        topo = chain_topology(AM.LIVENESSNET_BN)
        self.assertEqual(R.bn_order_problems(list(AM.LIVENESSNET_BN), topo), [])
        self.assertTrue(R.bn_order_problems(list(reversed(AM.LIVENESSNET_BN)), topo))
        with self.assertRaisesRegex(FR.Failure, "not in the model"):
            R.bn_ancestors({"a": ("BatchNormalization", ("missing",), ())}, "a")


# ============================================================================ F dev, soglia, ID dei frame

def part_of(keys, y, subjects=None):
    subjects = subjects or ["s1"] * len(keys)
    return {"y": np.asarray(y), "rows": [{"key": k, "subject": s, "session": "01", "label": str(v), "sha256": "0" * 64} for k, v, s in zip(keys, y, subjects)]}


class TestFDev(unittest.TestCase):
    def test_dev_state_counts_and_threshold(self):
        y = [0, 0, 1, 1, 1]
        p = np.array([0.1, 0.6, 0.4, 0.7, 0.9])
        z = np.log(np.stack([1 - p, p], 1))
        try:
            s, unit = R.dev_state(part_of([f"k{i}" for i in range(5)], y), p, z, "ordinary", "m")
        except FR.Failure as e:
            self.fail(f"dev_state refused a valid dev: {e}")
        self.assertEqual(s["threshold_eer"], FR.eer_threshold(np.array(y), p)["value"])
        self.assertEqual(s["threshold_eer"], D.condition_threshold("dev_eer", y, p))
        self.assertEqual((s["decided_attack_05"], s["correct_05"]), (3, 3))
        self.assertEqual(s["accuracy_05"], 0.6)
        self.assertFalse(s["one_class_05"]["one_class"])
        self.assertEqual(s["frame_ids_sha256"], D.ids_sha256([f"k{i}" for i in range(5)]))
        self.assertEqual(unit[D.FRAME_ID_FIELD], [f"k{i}" for i in range(5)])
        self.assertEqual(unit["threshold"], s["threshold_eer"])
        self.assertEqual(s["decided_attack_eer"], int(np.sum(p > s["threshold_eer"])))
        self.assertEqual(s["correct_eer"], int(np.sum((p > s["threshold_eer"]) == (np.array(y) == 1))))

    def test_score_exactly_on_the_eer_threshold_is_bona_fide(self):
        """Un punteggio esattamente sulla soglia EER (e uno esattamente su 0,5) è deciso bona fide, nel riepilogo e nel CSV."""
        y = [0, 0, 1, 1]
        p = np.array([0.2, 0.5, 0.5, 0.9])
        z = np.log(np.stack([1 - p, p], 1))
        part = part_of(["a", "b", "c", "d"], y)
        s, _ = R.dev_state(part, p, z, "ordinary", "m")
        thr = s["threshold_eer"]
        self.assertIn(thr, p.tolist())
        self.assertEqual(s["decided_attack_eer"], int(np.sum(p > thr)))
        self.assertEqual(s["decided_attack_05"], 1)
        rec = {"network": "LivenessNet", "fold": 1, "seed": 42, "weights_sha256": "0" * 64}
        rows = R.score_rows(rec, "ordinary", part, p, z, s)
        col = {c: i for i, c in enumerate(R.SCORE_COLUMNS)}
        self.assertEqual([r[col["prediction_05"]] for r in rows], ["0", "0", "0", "1"])
        self.assertEqual([r[col["prediction_dev_eer"]] for r in rows], ["1" if v > thr else "0" for v in p])
        self.assertEqual([r[col["threshold_dev_eer"]] for r in rows], [repr(thr)] * 4)
        self.assertEqual([r[col["score_attack"]] for r in rows], [repr(float(v)) for v in p])
        self.assertEqual([r[col[D.FRAME_ID_FIELD]] for r in rows], ["a", "b", "c", "d"])
        s2, _ = R.dev_state(part_of(["a", "b", "c", "d"], y), np.array([0.2, 0.3, 0.3, 0.9]), z, "ordinary", "m")
        rows2 = R.score_rows(rec, "ordinary", part, np.array([0.2, 0.3, 0.3, 0.9]), z, s2)
        self.assertEqual([r[col["prediction_dev_eer"]] for r in rows2], ["1" if v > s2["threshold_eer"] else "0" for v in (0.2, 0.3, 0.3, 0.9)])
        self.assertIn("0", [r[col["prediction_dev_eer"]] for r in rows2 if r[col["score_attack"]] == repr(s2["threshold_eer"])])

    def test_eer_cross_check(self):
        """Le soglie EER di pad_arch_definitions e di nuaa_folds_run devono coincidere (controllo incrociato)."""
        real = D.condition_threshold
        with mock.patch.object(D, "condition_threshold", lambda rule, y, s: real(rule, y, s) / 2):
            with self.assertRaisesRegex(FR.Failure, "differs between pad_arch_definitions and nuaa_folds_run"):
                R.dev_state(part_of(["a", "b"], [0, 1]), np.array([0.2, 0.7]), np.zeros((2, 2)), "ordinary", "m")

    def test_tie_at_threshold_is_bona_fide(self):
        s, _ = R.dev_state(part_of(["a", "b", "c"], [0, 1, 1]), np.array([0.5, 0.5, 0.9]), np.zeros((3, 2)), "ordinary", "m")
        self.assertEqual((s["decided_attack_05"], s["correct_05"]), (1, 2))

    def test_one_class_and_missing_class(self):
        s, _ = R.dev_state(part_of(["a", "b"], [0, 1]), np.array([0.2, 0.3]), np.zeros((2, 2)), "ordinary", "m")
        self.assertEqual((s["one_class_05"]["one_class"], s["one_class_05"]["class"]), (True, "bona fide"))
        try:
            s, _ = R.dev_state(part_of(["a", "b"], [1, 1]), np.array([0.2, 0.9]), np.zeros((2, 2)), "ordinary", "m")
        except FR.Failure as e:
            self.fail(f"a dev without one class must give no threshold, not a failure: {e}")
        self.assertFalse(s["one_class_05"]["evaluable"])
        self.assertIsNone(s["threshold_eer"])
        self.assertIsNone(s["decided_attack_eer"])

    def test_domain(self):
        with self.assertRaisesRegex(FR.Failure, "duplicated frame identifiers"):
            R.dev_state(part_of(["a", "a"], [0, 1]), np.array([0.2, 0.3]), np.zeros((2, 2)), "ordinary", "m")
        with self.assertRaisesRegex(FR.Failure, "finite and in"):
            R.dev_state(part_of(["a", "b"], [0, 1]), np.array([0.2, 1.3]), np.zeros((2, 2)), "ordinary", "m")
        with self.assertRaisesRegex(FR.Failure, "logits"):
            R.dev_state(part_of(["a", "b"], [0, 1]), np.array([0.2, 0.3]), np.zeros((3, 2)), "ordinary", "m")

    def recs(self, ids_a, ids_b, subj_b=None, state_ids=None, b_states=("ordinary", "re-estimated"), fold=1, seed=42):
        def unit(ids, subj=None):
            return {"y": np.array([0, 1]), "s": np.array([0.2, 0.7]), "subject": np.array(subj or ["s1", "s2"]), "threshold": 0.5,
                    D.FRAME_ID_FIELD: list(ids)}
        a = {"name": f"LivenessNet_fold{fold}_seed{seed}", "network": "LivenessNet", "fold": fold, "seed": seed,
             "dev_units": {"ordinary": unit(ids_a), "re-estimated": unit(state_ids or ids_a)}}
        b = {"name": f"ResNet-18_fold{fold}_seed{seed}", "network": "ResNet-18", "fold": fold, "seed": seed,
             "dev_units": {st: unit(state_ids if (state_ids and st == "re-estimated") else ids_b, subj_b) for st in b_states}}
        return [a, b]

    def test_frame_ids_equal(self):
        out = R.check_dev_frames(self.recs(["x", "y"], ["x", "y"]))
        self.assertEqual(out["units"]["fold1_seed42"], {"n_frames": 2, "frame_ids_sha256": D.ids_sha256(["x", "y"])})
        self.assertEqual((out["units_two_networks_compared"], out["units_two_states_compared"]), ({"ordinary": 1, "re-estimated": 1}, 1))
        out = R.check_dev_frames(self.recs(["x", "y"], ["x", "y"], b_states=("ordinary",)))
        self.assertEqual((out["units_two_networks_compared"], out["units_two_states_compared"]), ({"ordinary": 1, "re-estimated": 0}, 0))
        out = R.check_dev_frames(self.recs(["x", "y"], ["x", "y"]) + self.recs(["x", "y"], ["x", "y"], seed=43))
        self.assertEqual(out["units_two_states_compared"], 2)

    def test_frame_ids_differ_between_networks(self):
        """Stessi soggetti e classi, un ID diverso fra le due reti della stessa unità: stop (D.prepare_units)."""
        with self.assertRaisesRegex(FR.Failure, "the two networks are not scored on the same frame identifiers"):
            R.check_dev_frames(self.recs(["x", "y"], ["x", "z"]))
        with self.assertRaisesRegex(FR.Failure, "the two networks are not scored on the same frame identifiers"):
            R.check_dev_frames(self.recs(["x", "y"], ["y", "x"]))
        with self.assertRaisesRegex(FR.Failure, "the two networks are not scored on the same test frames"):
            R.check_dev_frames(self.recs(["x", "y"], ["x", "y"], subj_b=["s1", "s3"]))

    def test_frame_ids_differ_between_units_of_a_fold(self):
        with self.assertRaisesRegex(FR.Failure, "two units of .* are not scored on the same frames"):
            R.check_dev_frames(self.recs(["x", "y"], ["x", "y"]) + self.recs(["x", "w"], ["x", "w"], seed=43))

    def test_frame_ids_differ_between_bn_states(self):
        """Stessi soggetti e classi, un ID diverso fra i due stati della BN dello stesso modello: stop (D.same_frames)."""
        with self.assertRaisesRegex(FR.Failure, "the two BN states: unit .* the two conditions are not scored on the same frame identifiers"):
            R.check_dev_frames(self.recs(["x", "y"], ["x", "y"], state_ids=["x", "w"]))

    def test_unit_frames_refuses_test(self):
        """Ruolo test (o qualunque ruolo diverso da fitting e dev) rifiutato prima di assemblare."""
        cells = {("arm_b", 1): [], ("dev", 1): []}
        inp = types.SimpleNamespace(assembled=[], folds={}, cells=cells)
        for role in ("test", "dev_mix", "fitting_all"):
            with self.assertRaisesRegex(FR.Failure, f"role '{role}' refused"):
                R.unit_frames(None, inp, role, 1)
        self.assertEqual(inp.assembled, [])
        inp.assembled.append("test")
        with self.assertRaisesRegex(FR.Failure, "only the fitting and dev frames"):
            R.unit_frames(None, inp, "dev", 1)
        inp2 = types.SimpleNamespace(assembled=[], folds={1: {"test": []}}, cells=cells)
        self.assertEqual(len(R.unit_frames(None, types.SimpleNamespace(assembled=[], folds={}, cells=cells), "dev", 1)["keys"]), 0)
        with self.assertRaisesRegex(FR.Failure, "fold schema was loaded"):
            R.unit_frames(None, inp2, "dev", 1)


# ============================================================================ G controllo degli ingressi

class TestGCheckInputs(unittest.TestCase):
    def test_check_inputs_pass_nothing_written(self):
        w = world()
        before = tree(w.tmp)
        with CPA.tensorflow_unloaded():
            rc, out, err = run_main(w.argv("check-inputs", "--estimator-check", str(w.est)), base_hooks(w))
        self.assertEqual(rc, 0, err[-2000:])
        self.assertIn("RESULT: PASS (check-inputs)", out)
        self.assertIn("24 reference cards", out)
        self.assertIn("TensorFlow not imported", out)
        self.assertEqual(tree(w.tmp), before)

    def test_thread_environment_at_10_refused(self):
        """check-inputs rifiuta, come check-estimator e run, le variabili dei thread presenti e uguali a 10 (la prova
        ufficiale le aveva assenti)."""
        w = world()
        with thread_env(TF_NUM_INTRAOP_THREADS="10", TF_NUM_INTEROP_THREADS="10"):
            rc, out, err = run_main(w.argv("check-inputs"), base_hooks(w))
        self.assertEqual(rc, 1)
        self.assertIn("thread rule: environment variables TF_NUM_INTRAOP_THREADS '10', TF_NUM_INTEROP_THREADS '10' differ from the official trial", err)
        self.assertIn("RESULT: FAIL (check-inputs)", out)

    def test_reference_cards_checked(self):
        for alter, needle in ((lambda c: c.update(dev_keys_sha256="0" * 64), "dev: the keys differ"),
                              (lambda c: c.update(early_stopping_patience=12), "patience differ"),
                              (lambda c: c["history"].update(lr=[0.001]), "same, non-zero number of epochs"),
                              (lambda c: c.pop("fitting_keys_sha256"), "fitting_keys_sha256 or dev_keys_sha256")):
            with altered_world({("patience_4", 2, 43): alter}) as w:
                rc, out, err = run_main(w.argv("check-inputs"), base_hooks(w))
                self.assertEqual(rc, 1)
                self.assertIn(needle, err)

    def test_reference_card_listed_and_hashed(self):
        """Una scheda di riferimento assente dal SHA256SUMS riservato del run accettato, o diversa dalla sua riga: stop."""
        w = world()
        target = "arm_b_fold2_seed43_card.json"

        def unlisted(key, rdir):
            if key == "patience_4":
                lines = [l for l in (rdir / "SHA256SUMS").read_text().splitlines(keepends=True) if not l.rstrip().endswith(target)]
                (rdir / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")

        def changed(key, rdir):
            if key == "patience_4":
                (rdir / target).write_text((rdir / target).read_text() + " ", encoding="utf-8")
        for hook, needle in ((unlisted, f"{target} not listed"), (changed, f"reference {target}: SHA-256 differs")):
            w.after_sums = hook
            try:
                w.refresh()
                rc, out, err = run_main(w.argv("check-inputs"), base_hooks(w))
                self.assertEqual(rc, 1)
                self.assertIn(needle, err)
            finally:
                w.after_sums = None
                w.refresh()

    def test_estimator_record_checked(self):
        w = world()
        for kw, needle in (({"status": "MISMATCH"}, "not a PASS of check-estimator"), ({"cfg_sha": "0" * 64}, "another configuration"),
                           ({"over": {"schema": "pad-arch-estimator-check/0"}}, "schema, status or run id"),
                           ({"over": {"status": "INCOMPLETE"}}, "schema, status or run id"),
                           ({"over": {"run_id": "20261120-ARCHEST-1234567"}}, "schema, status or run id"),
                           ({"over": {"code_sha256": {}}}, "other code"), ({"over": {"models": [{"recalibration_pass": False}] * 36}}, "not every model"),
                           ({"over": {"expected_models": 37}}, "not every model"), ({"over": {"machine_rule": {"equal": False}}}, "machine and thread rule")):
            try:
                w.write_estimator_record(**kw)
                rc, out, err = run_main(w.argv("check-inputs", "--estimator-check", str(w.est)), base_hooks(w))
                self.assertEqual(rc, 1, kw)
                self.assertIn(needle, err)
            finally:
                w.write_estimator_record()
        (w.est / "public" / "estimator_check.json").write_text("{}")
        try:
            rc, out, err = run_main(w.argv("check-inputs", "--estimator-check", str(w.est)), base_hooks(w))
            self.assertIn("differs from the SHA-256 in its status.json", err)
        finally:
            w.write_estimator_record()
        cfg, cfg_sha = R.load_config(w.run_cfg)
        code = R.check_code(cfg)
        self.assertFalse(R.load_estimator_record(w.est, cfg_sha, code, False)["official"])
        with self.assertRaisesRegex(FR.Failure, "an official run needs an official check-estimator"):
            R.load_estimator_record(w.est, cfg_sha, code, True)
        try:
            w.write_estimator_record(over={"official": True})
            self.assertTrue(R.load_estimator_record(w.est, cfg_sha, code, True)["official"])
        finally:
            w.write_estimator_record()


# ============================================================================ H flusso completo di run (TensorFlow finto)

FLOW = {}


def flow_pass() -> dict:
    """Il run nuaa_patience_4 completo nel mondo con il TensorFlow finto (una volta, condiviso dai test)."""
    if not FLOW:
        w = world()
        shutil.rmtree(w.runs_out, ignore_errors=True)
        log = []
        rc, out, err = run_main(w.argv("run"), flow_hooks(w, log))
        FLOW.update(rc=rc, out=out, err=err, log=log, run=run_folder(w))
    return FLOW


class TestHRunFlow(unittest.TestCase):
    def setUp(self):
        self.w = world()

    def test_pass_outputs(self):
        f = flow_pass()
        self.assertEqual(f["rc"], 0, f["err"][-3000:])
        run = f["run"]
        self.assertEqual(sorted(os.listdir(run)), ["public", "restricted", "run.log", "status.json"])
        self.assertEqual(sorted(os.listdir(run / "public")), sorted(R.PUBLIC_OUTPUTS))
        names = [PAC.training_name(n, fo, s) for n, fo, s in PAC.plan_order([1, 2, 3, 4], [42, 43, 44])]
        want = sorted([f"{n}{x}" for n in names for x in R.PER_MODEL] + ["SHA256SUMS"])
        self.assertEqual(sorted(os.listdir(run / "restricted")), want)
        sums = FR.read_sums((run / "restricted" / "SHA256SUMS").read_bytes(), "s")
        self.assertEqual(sorted(sums), sorted(n for n in want if n != "SHA256SUMS"))
        for n, h in sums.items():
            self.assertEqual(FR.sha256_file(run / "restricted" / n), h)

    def test_order_of_the_plan_and_threads_before_training(self):
        """Le 24 posizioni nell'ordine del piano; i thread impostati (10 e 10) e la GPU nascosta prima del primo
        addestramento; la fase 2 dopo tutti gli addestramenti, nello stesso ordine."""
        f = flow_pass()
        log = f["log"]
        trains = [(x[1], x[2]) for x in log if isinstance(x, tuple) and x[0] == "train"]
        self.assertEqual(trains, [(n, s) for n, fo, s in PAC.plan_order([1, 2, 3, 4], [42, 43, 44])])
        first_train = next(i for i, x in enumerate(log) if isinstance(x, tuple) and x[0] == "train")
        self.assertEqual(log[:3], ["intra 10", "inter 10", "visible GPU []"])
        self.assertLess(2, first_train)
        res = [i for i, x in enumerate(log) if isinstance(x, tuple) and x[0] == "re-estimate"]
        self.assertGreater(min(res), max(i for i, x in enumerate(log) if isinstance(x, tuple) and x[0] == "train"))
        self.assertEqual([log[i][1] for i in res], [PAC.training_name(n, fo, s) for n, fo, s in PAC.plan_order([1, 2, 3, 4], [42, 43, 44])])

    def test_final_status_metadata(self):
        """Pattern 37: lo stato finale ha started, finished, tempo reale e CPU del processo, macchina, thread, commit,
        stato dell'albero, SHA-256 di configurazione, codice e ingressi; gli stessi valori nel riepilogo pubblico."""
        f = flow_pass()
        st = json.loads((f["run"] / "status.json").read_text())
        for k in ("started", "finished", "wall_seconds", "process_cpu_seconds", "machine", "threads", "threads_rule", "machine_rule", "commit",
                  "dirty", "config_sha256", "code_sha256", "inputs_sha256", "cap_hours", "cap_unit", "estimator_check", "public_sha256sums_sha256",
                  "restricted_sha256sums_sha256", "run_log_sha256"):
            self.assertIn(k, st)
        self.assertEqual(st["status"], "PASS")
        self.assertIs(st["dirty"], False)
        self.assertIsInstance(st["process_cpu_seconds"], float)
        self.assertEqual(st["threads"]["effective"], {"intra_op": 10, "inter_op": 10})
        self.assertTrue(st["machine_rule"]["equal"])
        self.assertEqual(st["code_sha256"][R.THIS_FILE], sha((ROOT / R.THIS_FILE).read_bytes()))
        self.assertEqual(st["inputs_sha256"]["configuration"], self.w.cfg_sha())
        summ = json.loads((f["run"] / "public" / "summary.json").read_text())
        for k in ("started", "finished", "wall_seconds", "process_cpu_seconds"):
            self.assertEqual(summ[k], st[k])
        self.assertEqual(st["public_sha256sums_sha256"], sha((f["run"] / "public" / "SHA256SUMS").read_bytes()))
        self.assertEqual(st["run_log_sha256"], sha((f["run"] / "run.log").read_bytes()))

    def test_public_outputs_hold_no_identifier(self):
        f = flow_pass()
        for n in R.PUBLIC_OUTPUTS:
            text = (f["run"] / "public" / n).read_text()
            self.assertNotIn("real/", text)
            self.assertNotIn("attack/", text)
            self.assertNotIn(str(self.w.tmp), text)
            for s in CFR.SUBJECTS:
                self.assertNotRegex(text, rf"(?<![\w.]){s}(?![\w.])")
        import csv
        rows = list(csv.DictReader(io.StringIO((f["run"] / "public" / "models.csv").read_text())))
        self.assertEqual(len(rows), 24)
        self.assertEqual([r["phase_2"] for r in rows], ["1"] * 24)
        self.assertEqual({r["history_check"] for r in rows if r["network"] == "LivenessNet"}, {"identical"})
        self.assertEqual({r["history_check"] for r in rows if r["network"] == "ResNet-18"}, {"not applicable"})
        oc = list(csv.DictReader(io.StringIO((f["run"] / "public" / "one_class.csv").read_text())))
        self.assertEqual({r["patience"] for r in oc}, {"4"})
        self.assertEqual({(r["network"], r["bn_state"]): r["text"] for r in oc},
                         {("LivenessNet", "ordinary"): "12 of 12 evaluable", ("LivenessNet", "re-estimated"): "0 of 12 evaluable",
                          ("ResNet-18", "ordinary"): "12 of 12 evaluable", ("ResNet-18", "re-estimated"): "0 of 12 evaluable"})
        keys = {r[2] for fo in range(1, SAR.K + 1) for r in self.w.cells[("dev", fo)] + self.w.cells[("armB_fitting", fo)]}
        for n in R.PUBLIC_OUTPUTS:
            text = (f["run"] / "public" / n).read_text()
            self.assertFalse([k for k in keys if k in text or k.split("/", 1)[1] in text], n)

    def test_restricted_scores_and_card(self):
        f = flow_pass()
        import csv
        r = f["run"] / "restricted"
        o = list(csv.DictReader(io.StringIO((r / "LivenessNet_fold2_seed43_dev_ordinary.csv").read_text())))
        e = list(csv.DictReader(io.StringIO((r / "LivenessNet_fold2_seed43_dev_re-estimated.csv").read_text())))
        x = list(csv.DictReader(io.StringIO((r / "ResNet-18_fold2_seed43_dev_ordinary.csv").read_text())))
        keys = sorted(row[2] for row in self.w.cells[("dev", 2)])
        self.assertEqual([q["sample_id"] for q in o], keys)
        self.assertEqual([q["sample_id"] for q in e], keys)
        self.assertEqual([q["sample_id"] for q in x], keys)
        self.assertEqual({q["bn_state"] for q in o} | {q["bn_state"] for q in e}, {"ordinary", "re-estimated"})
        card = json.loads((r / "LivenessNet_fold2_seed43_card.json").read_text())
        self.assertEqual(card["weights_sha256"], FR.sha256_file(r / "LivenessNet_fold2_seed43.keras"))
        self.assertEqual(card["checkpoint"], R.CHECKPOINT_TEXT)
        self.assertEqual(card["history_check"]["outcome"], "identical")
        self.assertEqual(card["early_stopping_patience"], 4)
        self.assertEqual(card["position"], 9)
        st = json.loads((r / "ResNet-18_fold2_seed43_bn_statistics.json").read_text())
        self.assertEqual(st["order"], R.expected_orders()["ResNet-18"])
        self.assertEqual(list(st["layers"]), R.expected_orders()["ResNet-18"])
        self.assertEqual(st["dev_re_estimated"]["n"], len(keys))
        self.assertTrue(st["evaluator_agreement"]["pass"])
        sc = list(csv.DictReader(io.StringIO((r / "LivenessNet_fold2_seed43_dev_scorer.csv").read_text())))
        self.assertEqual([q["sample_id"] for q in sc], keys)
        self.assertEqual({q["score_attack"] for q in sc}, {repr(SCORE["LivenessNet"])})
        self.assertEqual([q["score_attack"] for q in o], [repr(SCORE["LivenessNet"])] * len(keys))
        thr = float(o[0]["threshold_dev_eer"])
        self.assertEqual(thr, json.loads((r / "LivenessNet_fold2_seed43_bn_statistics.json").read_text())["dev_ordinary"]["threshold_eer"])
        rows = list(csv.DictReader(io.StringIO((f["run"] / "public" / "models.csv").read_text())))
        liv = [q for q in rows if q["network"] == "LivenessNet"]
        self.assertEqual({q["ordinary_one_class_05"] for q in liv}, {"bona fide"})
        self.assertEqual({q["ordinary_one_class_05"] for q in rows if q["network"] == "ResNet-18"}, {"attack"})
        self.assertEqual({q["evaluator_max_abs_diff"] for q in rows}, {"0.0"})
        self.assertEqual({q["evaluator_agreement"] for q in rows}, {"1"})
        self.assertEqual({q["model_status"] for q in rows}, {"complete"})
        pub = FR.read_sums((f["run"] / "public" / "SHA256SUMS").read_bytes(), "s")
        self.assertEqual(sorted(pub), sorted(n for n in R.PUBLIC_OUTPUTS if n != "SHA256SUMS"))
        for n, h in pub.items():
            self.assertEqual(sha((f["run"] / "public" / n).read_bytes()), h)
        rep = (f["run"] / "public" / "report.md").read_text()
        self.assertIn(R.TEST_SENTENCE[1:], rep)
        self.assertNotIn("loaded or evaluated", rep)
        self.assertNotIn("loaded as an image", rep)
        self.assertIn("Not an official run", rep)

    def test_same_command_same_folder_refused(self):
        """Lo stesso comando nella stessa cartella dopo un PASS: rifiutato prima di scrivere, la cartella del PASS intatta."""
        f = flow_pass()
        before = tree(f["run"])
        rc, out, err = run_main(self.w.argv("run"), flow_hooks(self.w))
        self.assertEqual(rc, 1)
        self.assertIn(f"--attempt 1 refused: 1 earlier attempts of this run in the output root ({f['run'].name} PASS)", err)
        self.assertIn("must be --attempt 2", err)
        self.assertEqual(tree(f["run"]), before)
        rc, out, err = run_main(self.w.argv("run", attempt=3), flow_hooks(self.w))
        self.assertIn("must be --attempt 2", err)
        with injected(r={"REPO_STATE": lambda root: dict(STATE)}):
            self.assertEqual(R.run_dir_problem(f["run"], self.w.runs_out, {})[:40], f"the run folder {f['run'].name} already exists"[:40])

    def test_new_attempt_with_broken_inputs(self):
        """Dopo un PASS, un nuovo tentativo (--attempt 2) con un ingresso guasto: cartella nuova con FAIL, nessuna uscita
        pubblica, nessun PASS; la cartella del PASS intatta."""
        f = flow_pass()
        before = tree(f["run"])
        w = self.w
        frames = w.base / "processed" / "frames.npz"
        saved = frames.read_bytes()
        try:
            frames.write_bytes(saved[:-10] + b"0" * 10)
            rc, out, err = run_main(w.argv("run", attempt=2), flow_hooks(w))
        finally:
            frames.write_bytes(saved)
        self.assertEqual(rc, 1)
        run2 = run_folder(w, attempt=2)
        st = json.loads((run2 / "status.json").read_text())
        self.assertEqual(st["status"], "FAIL")
        self.assertEqual(st["attempt"], {"attempt": 2, "previous": [{"run_id": f["run"].name, "attempt": 1, "status": "PASS"}]})
        self.assertIs(st["official"], False)
        for part in ("run", "--attempt 2", "--date 20261120", "--unofficial-out-root", "--out-root '<outside the repository>/runs'",
                     "--run nuaa_patience_4", "--i-am-authorized"):
            self.assertIn(part, st["command_line"])
        self.assertNotIn(str(w.tmp), st["command_line"])
        self.assertFalse((run2 / "public").exists() and os.listdir(run2 / "public"))
        self.assertEqual(sorted(os.listdir(run2)), ["run.log", "status.json"])
        for k in ("started", "finished", "wall_seconds", "process_cpu_seconds", "reason"):
            self.assertIn(k, st)
        self.assertEqual(tree(f["run"]), before)
        shutil.rmtree(run2)

    def test_failure_after_training_removes_public_keeps_complete_models(self):
        """Un errore dopo alcuni modelli (scorer che fallisce alla posizione 3): FAIL, nessuna uscita pubblica, il modello
        in corso tolto, i modelli completi restano."""
        w = self.w
        calls = []

        def scorer(path, s):
            calls.append(path)
            if len(calls) == 3:
                raise FR.Failure("synthetic scorer failure")
            return fake_scorer(path, s)
        out_root = w.tmp / "runs_fail"
        rc, out, err = run_main(w.argv("run", out=out_root), flow_hooks(w, MAKE_SCORER=scorer))
        self.assertEqual(rc, 1)
        run = out_root / run_folder(w).name
        st = json.loads((run / "status.json").read_text())
        self.assertEqual(st["status"], "FAIL")
        self.assertIn("synthetic scorer failure", st["reason"])
        self.assertEqual(st["complete_phase_1"], ["LivenessNet_fold1_seed42", "ResNet-18_fold1_seed42"])
        self.assertFalse((run / "public").exists())
        self.assertEqual(sorted(os.listdir(run / "restricted")),
                         sorted(f"{n}{x}" for n in st["complete_phase_1"] for x in (".keras", "_card.json", "_dev_scorer.csv")))
        self.assertIn("not validated (no restricted SHA256SUMS)", st["restricted"])
        shutil.rmtree(out_root)

    def test_history_different_reported_run_continues(self):
        """Storia diversa in un modello di LivenessNet (scheda di riferimento alterata): riportata con la prima epoca
        diversa, il run continua fino al PASS e il modello resta nel confronto."""
        def alter(card):
            card["history"]["val_loss"][1] += 0.25
        with altered_world({("patience_4", 3, 44): alter}) as w:
            out_root = w.tmp / "runs_hist"
            rc, out, err = run_main(w.argv("run", out=out_root), flow_hooks(w))
            self.assertEqual(rc, 0, err[-2000:])
            import csv
            run = out_root / run_folder(w).name
            rows = {r["position"]: r for r in csv.DictReader(io.StringIO((run / "public" / "models.csv").read_text()))}
            self.assertEqual(rows["17"]["history_check"], "different")
            self.assertEqual(rows["17"]["first_differing_epoch"], "2")
            self.assertEqual(rows["17"]["field_of_largest_difference"], "val_loss")
            self.assertEqual(rows["17"]["phase_2"], "1")
            self.assertEqual(sum(r["history_check"] == "different" for r in rows.values()), 1)
            self.assertIn("first differing epoch 2", out)
            shutil.rmtree(out_root)

    def test_cap_incomplete_keeps_outcome(self):
        """Tetto superato durante l'addestramento della posizione 5: INCOMPLETE, i file del modello in corso tolti, i 4
        modelli completi conservati e riportati, nessuna fase 2."""
        w = self.w
        state = {"jump": False}

        def on_epoch(network, seed, epoch):
            if (network, seed) == ("LivenessNet", 44) and epoch == 2:
                state["jump"] = True
        import time

        def clock():                     # orologio monotono finto: un salto di 10^7 s dopo la seconda epoca della posizione 5
            return time.monotonic() + (10 ** 7 if state["jump"] else 0)
        out_root = w.tmp / "runs_cap"
        log = []
        h = flow_hooks(w, log, TRAINER=fake_trainer(log, on_epoch=on_epoch))
        h["sr"]["CLOCK"] = clock
        rc, out, err = run_main(w.argv("run", out=out_root), h)
        self.assertEqual(rc, 2, err[-2000:])
        run = out_root / run_folder(w).name
        st = json.loads((run / "status.json").read_text())
        self.assertEqual(st["status"], "INCOMPLETE")
        self.assertEqual(st["stopped"]["cause"], "cap")
        self.assertEqual(st["stopped"]["step"], "epoch 2 of LivenessNet_fold1_seed44")
        self.assertEqual(st["stopped"]["discarded"], "LivenessNet_fold1_seed44")
        self.assertEqual(st["complete_phase_1"], ["LivenessNet_fold1_seed42", "ResNet-18_fold1_seed42", "LivenessNet_fold1_seed43", "ResNet-18_fold1_seed43"])
        self.assertEqual(st["complete_phase_2"], [])
        self.assertEqual(len(st["missing_phase_1"]), 20)
        for k in ("started", "finished", "wall_seconds", "process_cpu_seconds"):
            self.assertIn(k, st)
        self.assertFalse(any(e.startswith("LivenessNet_fold1_seed44") for e in os.listdir(run / "restricted")))
        self.assertEqual(sorted(os.listdir(run / "public")), sorted(R.PUBLIC_OUTPUTS))
        summ = json.loads((run / "public" / "summary.json").read_text())
        self.assertEqual((summ["status"], summ["complete_phase_1"], summ["complete_phase_2"]), ("INCOMPLETE", 4, 0))
        self.assertNotIn(("re-estimate", "LivenessNet_fold1_seed42"), log)
        disc = summ["models"][4]
        self.assertEqual(disc, {"position": 5, "name": "LivenessNet_fold1_seed44", "network": "LivenessNet", "fold": 1, "seed": 44,
                                "model_status": R.DISCARDED})
        self.assertEqual([m["model_status"] for m in summ["models"][:4]], ["phase 1 complete, BN states not evaluated"] * 4)
        import csv
        rows = list(csv.DictReader(io.StringIO((run / "public" / "models.csv").read_text())))
        self.assertEqual(rows[4]["model_status"], R.DISCARDED)
        for k in ("epochs_run", "restored_epoch", "history_check", "train_seconds", "phase_1_seconds", "first_differing_epoch"):
            self.assertEqual(rows[4][k], "NA", k)
        self.assertEqual(rows[5]["model_status"], "not started")
        rep = (run / "public" / "report.md").read_text()
        self.assertIn(f"| 5 | LivenessNet | 1 | 44 | {R.DISCARDED} | NA | NA | NA | NA |", rep)
        self.assertIn("no model has its BN states evaluated", rep)
        self.assertIn("comparisons made between the two networks of a unit: 0 units", rep)
        self.assertEqual(summ["one_class"], [])
        shutil.rmtree(out_root)

    def test_sigusr1_incomplete_in_phase_two(self):
        """Arresto volontario (SIGUSR1, registrato nel Budget) durante la fase 2: INCOMPLETE con i 24 modelli della fase 1 e
        quelli completi della fase 2."""
        w = self.w
        log = []
        holder = {}
        orig_budget = SR.Budget

        class B(orig_budget):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                holder["b"] = self

        def re(ctx, rec, fit, dev):
            out = fake_reestimate(log)(ctx, rec, fit, dev)
            if rec["name"] == "ResNet-18_fold1_seed43":
                holder["b"].stop_requested = "SIGUSR1"
            return out
        out_root = w.tmp / "runs_stop"
        h = flow_hooks(w, log, REESTIMATE=re)
        h["sr"]["Budget"] = B
        rc, out, err = run_main(w.argv("run", out=out_root), h)
        self.assertEqual(rc, 2, err[-2000:])
        st = json.loads(((out_root / run_folder(w).name) / "status.json").read_text())
        self.assertEqual((st["status"], st["stopped"]["cause"]), ("INCOMPLETE", "SIGUSR1"))
        self.assertEqual(len(st["complete_phase_1"]), 24)
        self.assertEqual(st["complete_phase_2"], ["LivenessNet_fold1_seed42", "ResNet-18_fold1_seed42", "LivenessNet_fold1_seed43", "ResNet-18_fold1_seed43"])
        self.assertNotIn("discarded", st["stopped"])
        import csv
        rows = list(csv.DictReader(io.StringIO(((out_root / run_folder(w).name) / "public" / "models.csv").read_text())))
        self.assertEqual([r["phase_2"] for r in rows], ["1"] * 4 + ["0"] * 20)
        self.assertEqual([r["phase_1"] for r in rows], ["1"] * 24)
        self.assertEqual(st["dev_frames"]["units_two_states_compared"], 2)
        shutil.rmtree(out_root)

    def threads_fail_in_folder(self, needle: str, **tf_kw):
        """Un difetto dei thread visibile solo dopo l'import di TensorFlow: FAIL nella cartella del run (solo status.json e
        run.log), con il messaggio atteso, prima di ogni addestramento."""
        w = self.w
        log = []
        out_root = w.tmp / "runs_threads"
        try:
            rc, out, err = run_main(w.argv("run", out=out_root), flow_hooks(w, log, tf=FakeTF(log=log, **tf_kw)))
            self.assertEqual(rc, 1, err[-2000:])
            self.assertFalse(any(isinstance(x, tuple) and x[0] == "train" for x in log))
            st = json.loads(((out_root / run_folder(w).name) / "status.json").read_text())
            self.assertEqual(st["status"], "FAIL")
            self.assertIn(needle, st["reason"])
            self.assertEqual(sorted(os.listdir(out_root / run_folder(w).name)), ["run.log", "status.json"])
        finally:
            shutil.rmtree(out_root, ignore_errors=True)

    def test_effective_intra_op_threads_not_10_stop_before_training(self):
        """Thread intra-op effettivi (riletti dal getter) diversi da 10, inter-op 10: FAIL prima di ogni addestramento."""
        self.threads_fail_in_folder("after setting: TensorFlow threads {'intra_op': 11, 'inter_op': 10} differ from training.threads", offset=1)

    def test_effective_inter_op_threads_not_10_stop_before_training(self):
        """Thread inter-op effettivi (riletti dal getter) diversi da 10, intra-op 10: FAIL prima di ogni addestramento."""
        self.threads_fail_in_folder("after setting: TensorFlow threads {'intra_op': 10, 'inter_op': 11} differ from training.threads", inter_offset=1)

    def test_tensorflow_initialised_before_the_threads_stops_before_training(self):
        """TensorFlow già inizializzato quando il runner imposta i thread (il setter solleva RuntimeError): FAIL prima di
        ogni addestramento."""
        self.threads_fail_in_folder("the TensorFlow threads cannot be set (Intra op parallelism cannot be modified after initialization.): "
                                    "TensorFlow was initialised before the runner set them", raise_on_set=True)

    def test_threads_changed_during_a_training(self):
        """I thread effettivi cambiati durante il primo addestramento (getter finto): FAIL alla rilettura dopo quel modello,
        prima del secondo addestramento."""
        w = self.w
        log = []
        tfm = FakeTF(log=log)

        def trainer(*a, **k):
            out = fake_trainer(log)(*a, **k)
            tfm.threading.offset = 1
            return out
        out_root = w.tmp / "runs_threads3"
        rc, out, err = run_main(w.argv("run", out=out_root), flow_hooks(w, log, tf=tfm, TRAINER=trainer))
        self.assertEqual(rc, 1)
        st = json.loads(((out_root / run_folder(w).name) / "status.json").read_text())
        self.assertIn("after LivenessNet_fold1_seed42: TensorFlow threads", st["reason"])
        self.assertEqual(len([x for x in log if isinstance(x, tuple) and x[0] == "train"]), 1)
        shutil.rmtree(out_root)

    def refused_before_any_folder(self, needle: str, w=None, **r):
        """Il run rifiutato prima di creare la radice d'uscita, con il messaggio atteso, senza nessuna chiamata a
        TensorFlow né addestramento."""
        w = w or self.w
        log = []
        out_root = w.tmp / "runs_refused_early"
        rc, out, err = run_main(w.argv("run", out=out_root), flow_hooks(w, log, **r))
        self.assertEqual(rc, 1, needle)
        self.assertIn(needle, err)
        self.assertFalse(out_root.exists(), needle)
        self.assertEqual(log, [], needle)

    def test_intra_op_environment_variable_contradicting_refused_before_any_folder(self):
        with thread_env(TF_NUM_INTRAOP_THREADS="8"):
            self.refused_before_any_folder("environment variable TF_NUM_INTRAOP_THREADS='8' contradicts training.threads.intra_op = 10")

    def test_inter_op_environment_variable_contradicting_refused_before_any_folder(self):
        with thread_env(TF_NUM_INTEROP_THREADS="8"):
            self.refused_before_any_folder("environment variable TF_NUM_INTEROP_THREADS='8' contradicts training.threads.inter_op = 10")

    def test_environment_variables_at_10_refused_before_any_folder(self):
        """Variabili dei thread presenti e uguali a 10 (ammesse da training.threads.rule della preregistrazione): rifiutate
        prima di ogni cartella, perché la prova ufficiale le aveva assenti (scelta dell'autore)."""
        with thread_env(TF_NUM_INTRAOP_THREADS="10", TF_NUM_INTEROP_THREADS="10"):
            self.refused_before_any_folder("thread rule: environment variables TF_NUM_INTRAOP_THREADS '10', TF_NUM_INTEROP_THREADS '10' differ "
                                           f"from the official trial {CPA.RUN} (TF_NUM_INTRAOP_THREADS absent, TF_NUM_INTEROP_THREADS absent)")
        with thread_env(TF_NUM_INTEROP_THREADS="10"):
            self.refused_before_any_folder("thread rule: environment variables TF_NUM_INTRAOP_THREADS absent, TF_NUM_INTEROP_THREADS '10' differ")

    def test_thread_record_different_from_the_trial(self):
        """Il record dei thread della prova con le variabili a 10 e l'ambiente di adesso senza: rifiutato prima di ogni
        cartella."""
        trial_threads = {"requested": {"intra_op": 10, "inter_op": 10}, "effective": {"intra_op": 10, "inter_op": 10}, "set_by_this_runner": True,
                         "environment": {"TF_NUM_INTRAOP_THREADS": "10", "TF_NUM_INTEROP_THREADS": "10"}, "logical_cpus": 15}
        with altered_world(trial_over={"threads": trial_threads}) as w, thread_env():
            self.refused_before_any_folder("thread rule: environment variables TF_NUM_INTRAOP_THREADS absent, TF_NUM_INTEROP_THREADS absent differ "
                                           f"from the official trial {CPA.RUN} (TF_NUM_INTRAOP_THREADS '10', TF_NUM_INTEROP_THREADS '10')", w=w)

    def test_requested_intra_op_threads_not_10_refused_before_any_folder(self):
        """La preregistrazione (fissata nella configurazione) con 8 thread intra-op richiesti e 10 inter-op: rifiutata
        prima di ogni cartella."""
        with pre_registration_threads(self.w, intra_op=8):
            self.refused_before_any_folder("configuration: training.threads.intra_op must be 10 (fixed by the coordinator on 19 November 2026")

    def test_requested_inter_op_threads_not_10_refused_before_any_folder(self):
        with pre_registration_threads(self.w, inter_op=8):
            self.refused_before_any_folder("configuration: training.threads.inter_op must be 10 (fixed by the coordinator on 19 November 2026")

    def test_processor_different_from_the_trial_stops_before_any_folder(self):
        self.refused_before_any_folder(f"machine rule: processor differ from the official trial {CPA.RUN}",
                                       MACHINE=lambda: {**MACH, "processor": "Apple M4"})

    def test_other_machine_fields_stop_before_any_folder(self):
        """Ogni altro campo della macchina diverso dalla prova, uno alla volta: rifiuto prima di ogni cartella."""
        for f, v in (("core_levels", [{"name": "Performance", "physical_cores": 4}]), ("ram_gb", 16.0), ("os", "macOS 27.3"), ("logical_cpus", 10),
                     ("architecture", "x86_64")):
            self.refused_before_any_folder(f"machine rule: {f} differ from the official trial {CPA.RUN}", MACHINE=lambda f=f, v=v: {**MACH, f: v})

    def test_refusals_before_any_folder(self):
        w = self.w
        out_root = w.tmp / "runs_refused"
        cases = [(w.argv("run", out=out_root, est=False), {}, "needs --estimator-check"),
                 ([a for a in w.argv("run", out=out_root) if a != "--i-am-authorized"], {}, "--i-am-authorized"),
                 (w.argv("run", out=out_root), {"REPO_STATE": lambda root: {"commit": "abcdef0", "dirty": True}}, "not clean"),
                 (w.argv("run", out=out_root, attempt=0), {}, "this attempt must be --attempt 1"),
                 (w.argv("run", out=out_root) + ["--date", "2026-11-20"], {}, "YYYYMMDD")]
        for argv, r, needle in cases:
            rc, out, err = run_main(argv, flow_hooks(w, **r))
            self.assertEqual(rc, 1, needle)
            self.assertIn(needle, err)
            self.assertFalse(out_root.exists(), needle)
        w.write_estimator_record(status="MISMATCH")
        try:
            rc, out, err = run_main(w.argv("run", out=out_root), flow_hooks(w))
            self.assertIn("not a PASS of check-estimator", err)
            self.assertFalse(out_root.exists())
        finally:
            w.write_estimator_record()

    def test_output_root_inside_inputs_refused(self):
        w = self.w
        rc, out, err = run_main(w.argv("run", out=w.base / "patience_4"), flow_hooks(w))
        self.assertEqual(rc, 1)
        self.assertIn("refusing the run folder", err)

    def test_patience_12_uses_its_reference_and_patience(self):
        w = self.w
        out_root = w.tmp / "runs_p12"
        log = []
        rc, out, err = run_main(w.argv("run", run="nuaa_patience_12", out=out_root), flow_hooks(w, log))
        self.assertEqual(rc, 0, err[-2000:])
        run = out_root / run_folder(w, "nuaa_patience_12").name
        card = json.loads((run / "restricted" / "LivenessNet_fold4_seed44_card.json").read_text())
        self.assertEqual(card["early_stopping_patience"], 12)
        self.assertEqual(card["history_check"]["reference_card_sha256"], FR.read_sums((w.base / "patience_12" / "restricted" / "SHA256SUMS").read_bytes(), "s")
                         ["arm_b_fold4_seed44_card.json"])
        shutil.rmtree(out_root)

    def test_val_loss_check(self):
        """Checkpoint il cui val_loss ricalcolato sul dev non è il minimo della scheda: FAIL."""
        w = self.w
        out_root = w.tmp / "runs_vl"

        def scorer(path, s):
            return lambda X: np.tile([0.9, 0.1], (len(X), 1))
        h = flow_hooks(w, MAKE_SCORER=scorer)
        h["sr"]["MAKE_LOGITS"] = lambda path, s: (lambda X: np.log(np.tile([0.9, 0.1], (len(X), 1))))
        rc, out, err = run_main(w.argv("run", out=out_root), h)
        self.assertEqual(rc, 1)
        self.assertIn("not the one of minimum val_loss", err)
        shutil.rmtree(out_root)


class TestIRunStops(unittest.TestCase):
    """Le altre fermate di run: tetto prima di un addestramento, prima dei punteggi del dev (checkpoint già salvato e poi
    tolto), prima della fase 2; uscite pubbliche e run.log con un identificativo rifiutati; ingressi cambiati durante il
    run; ID dei frame del dev diversi fra i due stati della BN."""

    def setUp(self):
        self.w = world()
        self.out_root = self.w.tmp / "runs_stops"
        shutil.rmtree(self.out_root, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.out_root, True)

    def jump_run(self, when: str):
        """Un salto dell'orologio monotono finto in un punto del flusso: dopo i punteggi del dev della posizione 4
        (scorer), dopo il salvataggio del checkpoint della posizione 5, o dopo la fase 1 della posizione 24."""
        state = {"jump": False}
        import time
        calls = []

        def scorer(path, s):
            calls.append(Path(path).name)
            if when == "after position 4" and len(calls) == 4:
                state["jump"] = True
            return fake_scorer(path, s)

        def save(model, path):
            R.save_checkpoint(model, path)
            if when == "after saving position 5" and Path(path).name == "LivenessNet_fold1_seed44.keras":
                state["jump"] = True
            if when == "after phase 1" and Path(path).name == "ResNet-18_fold4_seed44.keras":
                state["after_last_save"] = True

        def logits(path, s):
            if state.get("after_last_save"):
                state["jump"] = True
            return fake_logits(path, s)
        log = []
        h = flow_hooks(self.w, log, MAKE_SCORER=scorer, SAVE_MODEL=save)
        h["sr"]["CLOCK"] = lambda: time.monotonic() + (10 ** 7 if state["jump"] else 0)
        h["sr"]["MAKE_LOGITS"] = logits
        rc, out, err = run_main(self.w.argv("run", out=self.out_root), h)
        run = self.out_root / run_folder(self.w).name
        return rc, err, json.loads((run / "status.json").read_text()), run, log

    def test_cap_before_a_training(self):
        rc, err, st, run, log = self.jump_run("after position 4")
        self.assertEqual(rc, 2, err[-2000:])
        self.assertEqual(st["stopped"]["step"], "train LivenessNet_fold1_seed44")
        self.assertEqual(len(st["complete_phase_1"]), 4)
        self.assertEqual(len([x for x in log if isinstance(x, tuple) and x[0] == "train"]), 4)

    def test_cap_before_dev_scoring_discards_the_checkpoint(self):
        rc, err, st, run, log = self.jump_run("after saving position 5")
        self.assertEqual(rc, 2, err[-2000:])
        self.assertEqual(st["stopped"]["step"], "dev scoring LivenessNet_fold1_seed44")
        self.assertEqual(st["stopped"]["discarded"], "LivenessNet_fold1_seed44")
        self.assertFalse([e for e in os.listdir(run / "restricted") if "fold1_seed44" in e])
        sums = FR.read_sums((run / "restricted" / "SHA256SUMS").read_bytes(), "s")
        self.assertEqual(len(sums), 4 * 3)

    def test_cap_before_phase_two(self):
        rc, err, st, run, log = self.jump_run("after phase 1")
        self.assertEqual(rc, 2, err[-2000:])
        self.assertEqual(st["stopped"]["step"], "re-estimation LivenessNet_fold1_seed42")
        self.assertEqual((len(st["complete_phase_1"]), st["complete_phase_2"]), (24, []))
        self.assertFalse([x for x in log if isinstance(x, tuple) and x[0] == "re-estimate"])

    def test_public_output_with_an_identifier_refused(self):
        key = sorted(r[2] for r in self.w.cells[("dev", 1)])[0]
        real = R.report_md
        with mock.patch.object(R, "report_md", lambda doc: real(doc) + f"\n{key}\n"):
            rc, out, err = run_main(self.w.argv("run", out=self.out_root), flow_hooks(self.w))
        self.assertEqual(rc, 1)
        run = self.out_root / run_folder(self.w).name
        st = json.loads((run / "status.json").read_text())
        self.assertIn("report.md refused", st["reason"])
        self.assertFalse((run / "public").exists() and os.listdir(run / "public"))

    def test_run_log_with_an_identifier_refused(self):
        def trainer(*a, **k):
            print(f"subject {CFR.SUBJECTS[0]}")
            return fake_trainer([])(*a, **k)
        rc, out, err = run_main(self.w.argv("run", out=self.out_root), flow_hooks(self.w, TRAINER=trainer))
        self.assertEqual(rc, 1)
        run = self.out_root / run_folder(self.w).name
        st = json.loads((run / "status.json").read_text())
        self.assertIn("run.log holds text", st["reason"])
        self.assertIn("restricted SHA256SUMS written before the error", st["restricted"])
        self.assertTrue((run / "restricted" / "SHA256SUMS").is_file())
        self.assertFalse((run / "public").exists() and os.listdir(run / "public"))

    def test_non_finite_new_history_fails_the_run(self):
        """Un NaN nella storia nuova (loss, epoca 2) del modello alla posizione 1: il run finisce FAIL con grandezza ed
        epoca nella ragione, subito dopo l'addestramento (nessun altro addestramento, nessun checkpoint né file del modello),
        nessuna uscita pubblica."""
        log = []
        base = fake_trainer(log)

        def trainer(data, training, network, seed, build_model, cap_check, timer):
            card, model = base(data, training, network, seed, build_model, cap_check, timer)
            card["history"]["loss"][1] = float("nan")
            timer.record[1]["loss"] = float("nan")
            return card, model
        rc, out, err = run_main(self.w.argv("run", out=self.out_root), flow_hooks(self.w, log, TRAINER=trainer))
        self.assertEqual(rc, 1)
        run = self.out_root / run_folder(self.w).name
        st = json.loads((run / "status.json").read_text())
        self.assertEqual(st["status"], "FAIL")
        self.assertEqual(st["reason"], "LivenessNet_fold1_seed42: non-finite value in loss, new history, epoch 2: the run ends as FAIL (a card "
                                       "with a non-finite value cannot be written)")
        self.assertEqual(len([x for x in log if isinstance(x, tuple) and x[0] == "train"]), 1)
        self.assertEqual(os.listdir(run / "restricted"), [])
        self.assertEqual(st["complete_phase_1"], [])
        self.assertFalse((run / "public").exists())
        self.assertEqual(sorted(os.listdir(run)), ["restricted", "run.log", "status.json"])

    def test_model_in_progress_discarded_on_fail(self):
        """Un errore dopo il salvataggio del checkpoint della posizione 1 (controllo della scheda): FAIL, e i file del
        modello in corso, checkpoint compreso, tolti."""
        def save(model, path):
            R.save_checkpoint(model, path)
            raise FR.Failure("synthetic failure after the checkpoint")
        rc, out, err = run_main(self.w.argv("run", out=self.out_root), flow_hooks(self.w, SAVE_MODEL=save))
        self.assertEqual(rc, 1)
        run = self.out_root / run_folder(self.w).name
        self.assertIn("synthetic failure after the checkpoint", json.loads((run / "status.json").read_text())["reason"])
        self.assertEqual(os.listdir(run / "restricted"), [])

    def test_test_role_found_at_the_end_fails_the_run(self):
        """Un ruolo «test» registrato fra quelli assemblati dopo l'ultima chiamata di unit_frames (nella fase 2 dell'ultimo
        modello): il controllo finale PAC.assert_no_test ferma il run come FAIL, nessuna uscita pubblica."""
        real = fake_reestimate()

        def re(ctx, rec, fit, dev):
            out = real(ctx, rec, fit, dev)
            if rec["name"] == "ResNet-18_fold4_seed44":
                ctx.inp.assembled.append("test")
            return out
        rc, out, err = run_main(self.w.argv("run", out=self.out_root), flow_hooks(self.w, REESTIMATE=re))
        self.assertEqual(rc, 1)
        run = self.out_root / run_folder(self.w).name
        st = json.loads((run / "status.json").read_text())
        self.assertEqual(st["reason"], "roles ['test'] were assembled: only the fitting and dev frames may be assembled")
        self.assertFalse((run / "public").exists())

    def changed_during_run(self, path: Path, needle: str):
        saved = path.read_bytes()
        n = {"k": 0}

        def trainer(*a, **k):
            n["k"] += 1
            if n["k"] == 3:
                path.write_bytes(saved + b" ")
            return fake_trainer([])(*a, **k)
        try:
            rc, out, err = run_main(self.w.argv("run", out=self.out_root), flow_hooks(self.w, TRAINER=trainer))
        finally:
            path.write_bytes(saved)
        self.assertEqual(rc, 1)
        st = json.loads(((self.out_root / run_folder(self.w).name) / "status.json").read_text())
        self.assertIn(needle, st["reason"])

    def test_configuration_changed_during_the_run(self):
        self.changed_during_run(self.w.run_cfg, "the configuration changed during the run")

    def test_reference_card_changed_during_the_run(self):
        self.changed_during_run(self.w.base / "patience_4" / "restricted" / "arm_b_fold4_seed44_card.json", "a reference card changed")

    def test_trial_record_changed_during_the_run(self):
        self.changed_during_run(self.w.trial_dir / "trial.json", "trial record trial.json changed")

    def test_dev_frame_ids_between_bn_states_stop_the_run(self):
        """Un ID dei frame diverso fra i due stati della BN di un modello (stessi soggetti ed etichette): FAIL."""
        real = R.dev_state

        def dev_state(part, p, z, state, what):
            summ, unit = real(part, p, z, state, what)
            if state == "re-estimated" and "_fold2_" in what:          # tutte le unità della fold 2, entrambe le reti
                unit[D.FRAME_ID_FIELD][0] = unit[D.FRAME_ID_FIELD][0] + "x"
            return summ, unit
        with mock.patch.object(R, "dev_state", dev_state):
            rc, out, err = run_main(self.w.argv("run", out=self.out_root), flow_hooks(self.w))
        self.assertEqual(rc, 1)
        st = json.loads(((self.out_root / run_folder(self.w).name) / "status.json").read_text())
        self.assertIn("the two BN states: unit ('NUAA', 2, 42): the two conditions are not scored on the same frame identifiers", st["reason"])

    def test_dev_frame_ids_between_networks_stop_the_run(self):
        real = R.dev_state

        def dev_state(part, p, z, state, what):
            summ, unit = real(part, p, z, state, what)
            if what == "ResNet-18_fold3_seed42" and state == "ordinary":
                unit[D.FRAME_ID_FIELD][-1] = unit[D.FRAME_ID_FIELD][-1] + "x"
            return summ, unit
        with mock.patch.object(R, "dev_state", dev_state):
            rc, out, err = run_main(self.w.argv("run", out=self.out_root), flow_hooks(self.w))
        self.assertEqual(rc, 1)
        st = json.loads(((self.out_root / run_folder(self.w).name) / "status.json").read_text())
        self.assertIn("unit ('NUAA', 3, 42): the two networks are not scored on the same frame identifiers", st["reason"])


class FakeVar:
    def __init__(self, name, a):
        self.name, self.a = f"x/{name}:0", np.asarray(a, np.float32)

    def numpy(self):
        return self.a


class FakeBN:
    """Uno strato BN finto per BR.snapshot_weights e BR.moving_keys (pesi gamma, beta, media e varianza mobili)."""

    def __init__(self, name, gamma, mm, mv):
        self.name = name
        self.gamma, self.beta = FakeVar("gamma", gamma), FakeVar("beta", np.zeros(len(gamma)))
        self.moving_mean, self.moving_variance = FakeVar("moving_mean", mm), FakeVar("moving_variance", mv)
        self.weights = [self.gamma, self.beta, self.moving_mean, self.moving_variance]


class FakeModelBN:
    def __init__(self, layers):
        self.layers = layers

    def get_layer(self, n):
        return next(l for l in self.layers if l.name == n)


def moments_of(a):
    m = BR.Moments()
    m.add(np.asarray(a, np.float64))
    return m


class TestJAgreementAndChecks(unittest.TestCase):
    """L'accordo fra i due valutatori (MISMATCH), i controlli della PR #27 in comune (pr27_checks) con oggetti finti, la
    regola dei tentativi e della radice d'uscita, le riverifiche finali e il gestore di SIGUSR1."""

    def setUp(self):
        self.w = world()
        self.bn = bn_doc()
        self.out_root = self.w.tmp / "runs_j"
        shutil.rmtree(self.out_root, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.out_root, True)

    def test_agreement_rule(self):
        ag = R.evaluator_agreement
        a = np.array([0.2, 0.5, 0.3, 0.9])
        r = ag(a, a.copy(), 0.3, self.bn)
        self.assertEqual((r["pass"], r["max_abs_diff"], r["discordant_05"], r["discordant_eer"]), (True, 0.0, 0, 0))
        r = ag(a, a + np.array([0, 0, 0, 2e-5]), 0.3, self.bn)
        self.assertEqual((r["pass"], r["discordant_05"], r["discordant_eer"]), (False, 0, 0))
        self.assertAlmostEqual(r["max_abs_diff"], 2e-5, places=12)
        r = ag(a, a + np.array([0, 1e-6, 0, 0]), 0.3, self.bn)            # 0,5 + 1e-6: discordante a 0,5, non alla soglia EER
        self.assertEqual((r["pass"], r["discordant_05"], r["discordant_eer"]), (False, 1, 0))
        r = ag(a, a + np.array([0, 0, 1e-6, 0]), 0.3, self.bn)            # 0,3 + 1e-6: discordante alla soglia EER 0,3
        self.assertEqual((r["pass"], r["discordant_05"], r["discordant_eer"]), (False, 0, 1))
        r = ag(a, a + np.array([0, 0, 1e-6, 0]), None, self.bn)
        self.assertEqual((r["pass"], r["discordant_eer"]), (True, None))
        try:
            r = ag(a, a[:3], 0.3, self.bn)
        except ValueError as e:
            self.fail(f"scores of different shapes must give a failed agreement, not an error: {e}")
        self.assertEqual((r["pass"], r["max_abs_diff"]), (False, None))
        a2 = np.array([0.2, 0.4, 0.3, 0.9])
        r = ag(a2, a2 + 5e-6, 0.35, self.bn)
        self.assertTrue(r["pass"])
        self.assertAlmostEqual(r["max_abs_diff"], 5e-6, places=12)

    def test_mismatch_outcome(self):
        """Scarto finto sopra la tolleranza per un modello: il run completa gli addestramenti, esito MISMATCH (codice 4),
        nessuna uscita pubblica, discrepanza nello stato, SHA256SUMS riservato scritto."""
        log = []
        ov = lambda rec, p: p + (2e-5 if rec["name"] == "LivenessNet_fold2_seed43" else 0)  # noqa: E731
        rc, out, err = run_main(self.w.argv("run", out=self.out_root), flow_hooks(self.w, log, REESTIMATE=fake_reestimate(log, ordinary_over=ov)))
        self.assertEqual(rc, 4, err[-2000:])
        run = self.out_root / run_folder(self.w).name
        st = json.loads((run / "status.json").read_text())
        self.assertEqual(st["status"], "MISMATCH")
        self.assertEqual([m["name"] for m in st["evaluator_mismatches"]], ["LivenessNet_fold2_seed43"])
        self.assertAlmostEqual(st["evaluator_mismatches"][0]["max_abs_diff"], 2e-5, places=10)
        self.assertFalse((run / "public").exists())
        self.assertIsNone(st["public_sha256sums_sha256"])
        self.assertEqual(st["restricted_sha256sums_sha256"], sha((run / "restricted" / "SHA256SUMS").read_bytes()))
        self.assertEqual(len([x for x in log if isinstance(x, tuple) and x[0] == "train"]), 24)
        self.assertIn("RESULT: MISMATCH", out)

    def test_small_gap_reported(self):
        ov = lambda rec, p: p + (5e-6 if rec["name"] == "ResNet-18_fold4_seed42" else 0)  # noqa: E731
        rc, out, err = run_main(self.w.argv("run", out=self.out_root), flow_hooks(self.w, REESTIMATE=fake_reestimate(ordinary_over=ov)))
        self.assertEqual(rc, 0, err[-2000:])
        import csv
        rows = {r["position"]: r for r in csv.DictReader(io.StringIO(((self.out_root / run_folder(self.w).name) / "public" / "models.csv").read_text()))}
        self.assertAlmostEqual(float(rows["20"]["evaluator_max_abs_diff"]), 5e-6, places=10)
        self.assertEqual((rows["20"]["evaluator_discordant_05"], rows["20"]["evaluator_discordant_eer"]), ("0", "0"))
        self.assertEqual(rows["19"]["evaluator_max_abs_diff"], "0.0")

    def pr27(self, alter=None, fit_gap=0.0, dev_gap=0.0, count=None):
        orig = FakeModelBN([FakeBN("bn", [1.0, 1.0], [0.0, 0.0], [1.0, 1.0])])
        snap = BR.snapshot_weights(orig)
        x = np.array([[1.0, 2.0], [3.0, 5.0], [2.0, 2.0]])
        m, v = x.mean(axis=0), x.var(axis=0)
        copy = FakeModelBN([FakeBN("bn", [1.0, 1.0], m, v)])
        stats = {"bn": {"mean32": m.astype(np.float32), "var32": v.astype(np.float32), "count": 3 if count is None else count}}
        if alter:
            alter(copy, stats)
        return R.pr27_checks("m", stats, snap, copy, ["bn"], {"softmax_gap": fit_gap}, {"bn": moments_of(x)}, {"softmax_gap": dev_gap}, self.bn)

    def test_pr27_checks(self):
        cons = self.pr27()
        self.assertEqual(cons["bn"]["channels_outside_tolerance"], 0)
        with self.assertRaisesRegex(FR.Failure, "trained tensors changed"):
            self.pr27(alter=lambda c, st: setattr(c.layers[0].gamma, "a", np.float32([2.0, 1.0])))
        with self.assertRaisesRegex(FR.Failure, "does not hold the installed"):
            self.pr27(alter=lambda c, st: setattr(c.layers[0].moving_mean, "a", np.float32([9.0, 9.0])))

        def inconsistent(c, st):
            st["bn"]["mean32"] = st["bn"]["mean32"] + np.float32(0.5)
            c.layers[0].moving_mean.a = st["bn"]["mean32"]
        with self.assertRaisesRegex(FR.Failure, "consistency check failed"):
            self.pr27(alter=inconsistent)
        with self.assertRaisesRegex(FR.Failure, "counted a different number of values per channel"):
            self.pr27(count=4)
        with self.assertRaisesRegex(FR.Failure, "differs from the fitting probabilities"):
            self.pr27(fit_gap=1e-3)
        with self.assertRaisesRegex(FR.Failure, "differs from the dev probabilities"):
            self.pr27(dev_gap=1e-3)

    def test_official_output_root(self):
        rc, out, err = run_main(self.w.argv("run", out=self.out_root, unofficial=False), flow_hooks(self.w))
        self.assertEqual(rc, 1)
        self.assertIn("needs --unofficial-out-root", err)
        self.assertFalse(self.out_root.exists())
        a = self.w.argv("run", unofficial=True)
        a[a.index("--out-root") + 1] = str(ROOT / R.DEFAULT_OUT_ROOT)
        before = tree(ROOT / "results")
        rc, out, err = run_main(a, flow_hooks(self.w))
        self.assertIn("with the official output root results/archrun: refused", err)
        self.assertEqual(tree(ROOT / "results"), before)

    def test_symlink_output_root_refused(self):
        real = self.w.tmp / "runs_real_dir"
        real.mkdir(exist_ok=True)
        link = self.w.tmp / "runs_link"
        if not link.is_symlink():
            os.symlink(real, link)
        try:
            rc, out, err = run_main(self.w.argv("run", out=link), flow_hooks(self.w))
            self.assertEqual(rc, 1)
            self.assertIn("the output root is a symbolic link", err)
            self.assertEqual(os.listdir(real), [])
        finally:
            link.unlink()

    def test_previous_attempts(self):
        d = self.out_root
        d.mkdir()
        (d / f"{DATE}-ARCHRUN-p4-abcdef0").mkdir()
        (d / f"{DATE}-ARCHRUN-p4-abcdef0" / "status.json").write_text('{"status": "INCOMPLETE"}')
        (d / f"20261121-ARCHRUN-p4-1234567-r2").mkdir()
        (d / f"{DATE}-ARCHRUN-p12-abcdef0-r5").mkdir()
        (d / f"{DATE}-ARCHEST-abcdef0").mkdir()
        prev = R.previous_attempts(d, "run", "nuaa_patience_4")
        self.assertEqual(prev, [{"run_id": f"{DATE}-ARCHRUN-p4-abcdef0", "attempt": 1, "status": "INCOMPLETE"},
                                {"run_id": "20261121-ARCHRUN-p4-1234567-r2", "attempt": 2, "status": "unreadable"}])
        self.assertEqual(R.attempt_rule(d, "run", "nuaa_patience_4", 3)["attempt"], 3)
        with self.assertRaisesRegex(FR.Failure, "must be --attempt 3"):
            R.attempt_rule(d, "run", "nuaa_patience_4", 2)
        with self.assertRaisesRegex(FR.Failure, "must be --attempt 6"):
            R.attempt_rule(d, "run", "nuaa_patience_12", 1)
        self.assertEqual(R.attempt_rule(d, "estimator", None, 2)["previous"][0]["attempt"], 1)
        self.assertEqual(R.attempt_rule(self.w.tmp / "none", "run", "nuaa_patience_4", 1), {"attempt": 1, "previous": []})

    def test_malformed_entries_stop(self):
        for bad in (f"{DATE}-ARCHRUN-p4-abcdef0-r02", f"{DATE}-ARCHRUN-p4-abcdef0-r1", f"{DATE}-ARCHRUN-p4-abcdef0-r3.old",
                    f"{DATE}-archrun-p4-abcdef0", f"{DATE}-ARCHRUN-p4-ABCDEF0"):
            d = self.out_root
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir()
            (d / bad).mkdir()
            (d / f"{DATE}-ARCHRUN-p12-abcdef0-r02").mkdir()                 # un altro run: non riguarda p4
            with self.assertRaisesRegex(FR.Failure, f"the entry '{re.escape(bad)}' looks like a run id"):
                R.previous_attempts(d, "run", "nuaa_patience_4")
            with self.assertRaisesRegex(FR.Failure, "looks like a run id"):
                R.previous_attempts(d, "run", "nuaa_patience_12")
        shutil.rmtree(self.out_root)
        self.out_root.mkdir()
        (self.out_root / f"{DATE}-ARCHEST-abcdef0-r1").mkdir()
        with self.assertRaisesRegex(FR.Failure, "looks like a run id"):
            R.previous_attempts(self.out_root, "estimator")

    def test_command_text(self):
        """Il comando registrato dai valori interpretati: --opt=valore e --opt valore, percorsi relativi, assoluti dentro e
        fuori dal repository; nessun percorso personale."""
        parse = lambda argv: R.build_parser().parse_args(argv)  # noqa: E731
        outside = self.w.tmp / "est" / "X"
        t = R.command_text(parse(["run", f"--estimator-check={outside}", "--out-root", str(self.w.tmp / "runs"), "--unofficial-out-root",
                                  "--run=nuaa_patience_4", "--attempt=2", "--date", DATE, "--i-am-authorized"]))
        self.assertEqual(t, "python -I -B scripts/pad_arch_run.py run --config delivery/archrun/pad_arch_run.json --out-root "
                            "'<outside the repository>/runs' --unofficial-out-root --run nuaa_patience_4 --estimator-check "
                            f"'<outside the repository>/X' --date {DATE} --attempt 2 --i-am-authorized")
        t = R.command_text(parse(["check-inputs", f"--config={ROOT / 'delivery' / 'a.json'}", "--estimator-check", "results/archrun/Y",
                                  f"--out-root={ROOT / 'results' / 'archrun'}"]))
        self.assertIn("--config delivery/a.json", t)
        self.assertIn("--estimator-check results/archrun/Y", t)
        self.assertIn("--out-root results/archrun", t)
        t = R.command_text(parse(["check-inputs", f"--input-root={self.w.base}"]))
        self.assertIn("--input-root '<outside the repository>/in'", t)
        for x in (str(self.w.tmp), str(Path.home()), str(ROOT)):
            self.assertNotIn(x, t)

    def test_personal_text_in_the_command_refused_before_the_folder(self):
        import getpass
        name = getpass.getuser()
        if len(name) < 3:
            name = "personalname"
        est = self.w.tmp / name
        a = [x if x != str(self.w.est) else str(est) for x in self.w.argv("run", out=self.out_root)]
        a[a.index("--estimator-check")] = f"--estimator-check={est}"
        del a[a.index(f"--estimator-check={est}") + 1]
        with mock.patch.object(PAC, "personal_tokens", lambda: [name]):
            rc, out, err = run_main(a, flow_hooks(self.w))
        self.assertEqual(rc, 1)
        self.assertIn("the recorded command holds text that a publishable output must not hold", err)
        self.assertFalse(self.out_root.exists())

    def test_status_scanned(self):
        """Un testo personale nel record della macchina: status.json RUNNING rifiutato prima di ogni lavoro; lo stato FAIL
        è scritto con i testi personali sostituiti."""
        token = str(self.w.tmp / "personal")
        real = R.machine_rule
        with mock.patch.object(R, "machine_rule", lambda trial: {**real(trial), "note": token}), \
                mock.patch.object(PAC, "personal_tokens", lambda: [token]):
            log = []
            rc, out, err = run_main(self.w.argv("run", out=self.out_root), flow_hooks(self.w, log))
        self.assertEqual(rc, 1)
        text = ((self.out_root / run_folder(self.w).name) / "status.json").read_text()
        self.assertNotIn(token, text)
        self.assertIn("<redacted>", text)
        st = json.loads(text)
        self.assertEqual(st["status"], "FAIL")
        self.assertIn("status.json refused", st["reason"])
        self.assertFalse([x for x in log if isinstance(x, tuple) and x[0] == "train"])

    def test_timer_history_diverges(self):
        """La storia registrata dalla callback dei tempi diversa da quella di Keras: FAIL."""
        def trainer(data, training, network, seed, build_model, cap_check, timer):
            card, model = fake_trainer([])(data, training, network, seed, build_model, cap_check, timer)
            timer.record[-1]["loss"] += 0.01
            return card, model
        rc, out, err = run_main(self.w.argv("run", out=self.out_root), flow_hooks(self.w, TRAINER=trainer))
        self.assertEqual(rc, 1)
        st = json.loads(((self.out_root / run_folder(self.w).name) / "status.json").read_text())
        self.assertIn("the history recorded by the timing callback differs from the Keras history", st["reason"])

    def test_package_sums_changed_during_the_run(self):
        q = self.w.base / "package_SHA256SUMS"
        saved = q.read_bytes()
        n = {"k": 0}

        def trainer(*a, **k):
            n["k"] += 1
            if n["k"] == 2:
                q.write_bytes(saved + b"\n")
            return fake_trainer([])(*a, **k)
        try:
            rc, out, err = run_main(self.w.argv("run", out=self.out_root), flow_hooks(self.w, TRAINER=trainer))
        finally:
            q.write_bytes(saved)
        self.assertEqual(rc, 1)
        st = json.loads(((self.out_root / run_folder(self.w).name) / "status.json").read_text())
        self.assertIn("pinned file package_SHA256SUMS: SHA-256", st["reason"])

    def test_root_kind(self):
        """La radice d'uscita per identità: la radice ufficiale anche con un'altra grafia (alias con lo stesso os.stat, come
        un filesystem che non distingue le maiuscole) o attraverso un collegamento simbolico intermedio; ogni altra radice
        dentro il repository, o che lo contiene, rifiutata; fuori solo con il flag."""
        top = self.w.tmp / "rootkind"
        shutil.rmtree(top, ignore_errors=True)
        repo = top / "repo"
        official = repo / "results" / "archrun"
        official.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, top, True)
        rk = R.root_kind

        def kind(*a):
            """L'esito di root_kind come valore (True, False o «refused: …»), così che un rifiuto sia un'asserzione fallita."""
            try:
                return rk(*a)
            except FR.Failure as e:
                return f"refused: {e}"
        self.assertIs(kind(official, repo, official, False), True)
        with self.assertRaisesRegex(FR.Failure, "with the official output root"):
            rk(official, repo, official, True)
        with alias_of(repo / "Results", repo / "results"):
            self.assertIs(kind(repo / "Results" / "archrun", repo, official, False), True)
            with self.assertRaisesRegex(FR.Failure, "with the official output root"):
                rk(repo / "Results" / "archrun", repo, official, True)
        link = top / "link"
        os.symlink(repo / "results", link)
        self.assertIs(kind(link / "archrun", repo, official, False), True)
        with self.assertRaisesRegex(FR.Failure, "with the official output root"):
            rk(link / "archrun", repo, official, True)
        for inside in (official / "x", repo / "results", repo / "other", link / "archrun" / "x"):
            with self.assertRaisesRegex(FR.Failure, "inside the repository .or containing it. is refused"):
                rk(inside, repo, official, True)
        with self.assertRaisesRegex(FR.Failure, "inside the repository .or containing it. is refused"):
            rk(top, repo, official, True)
        with self.assertRaisesRegex(FR.Failure, "needs --unofficial-out-root"):
            rk(top / "elsewhere", repo, official, False)
        self.assertFalse(rk(top / "elsewhere", repo, official, True))
        # radice ufficiale non ancora creata: lo stesso confronto con l'antenato esistente
        shutil.rmtree(official)
        self.assertIs(kind(official, repo, official, False), True)
        with alias_of(repo / "Results", repo / "results"):
            self.assertIs(kind(repo / "Results" / "archrun", repo, official, False), True)

    def test_code_changed_at_the_end(self):
        real = R.check_code
        calls = {"n": 0}

        def check_code(cfg):
            calls["n"] += 1
            out = real(cfg)
            return out if calls["n"] == 1 else {**out, R.THIS_FILE: "0" * 64}
        with mock.patch.object(R, "check_code", check_code):
            rc, out, err = run_main(self.w.argv("run", out=self.out_root), flow_hooks(self.w))
        self.assertEqual(rc, 1)
        st = json.loads(((self.out_root / run_folder(self.w).name) / "status.json").read_text())
        self.assertIn("the code changed during the run", st["reason"])

    def test_estimator_record_changed_during_the_run(self):
        q = self.w.est / "public" / "estimator_check.json"
        saved = q.read_bytes()
        n = {"k": 0}

        def trainer(*a, **k):
            n["k"] += 1
            if n["k"] == 2:
                q.write_bytes(saved + b" ")
            return fake_trainer([])(*a, **k)
        try:
            rc, out, err = run_main(self.w.argv("run", out=self.out_root), flow_hooks(self.w, TRAINER=trainer))
        finally:
            q.write_bytes(saved)
        self.assertEqual(rc, 1)
        st = json.loads(((self.out_root / run_folder(self.w).name) / "status.json").read_text())
        self.assertIn("the estimator check record changed during the run", st["reason"])

    def test_sigusr1_handler_installed(self):
        """Un SIGUSR1 vero al processo durante la posizione 3: il gestore del run lo registra e il run si ferma come
        INCOMPLETE al controllo successivo (il gestore no-op messo qui prima evita che, senza il gestore del run, il
        segnale termini il processo dei test)."""
        import signal
        prev = signal.signal(signal.SIGUSR1, lambda *a: None)
        n = {"k": 0}

        def trainer(*a, **k):
            n["k"] += 1
            if n["k"] == 3:
                os.kill(os.getpid(), signal.SIGUSR1)
            return fake_trainer([])(*a, **k)
        try:
            rc, out, err = run_main(self.w.argv("run", out=self.out_root), flow_hooks(self.w, TRAINER=trainer))
        finally:
            signal.signal(signal.SIGUSR1, prev)
        self.assertEqual(rc, 2, err[-2000:])
        st = json.loads(((self.out_root / run_folder(self.w).name) / "status.json").read_text())
        self.assertEqual((st["status"], st["stopped"]["cause"], st["stopped"]["step"]), ("INCOMPLETE", "SIGUSR1", "epoch 1 of LivenessNet_fold1_seed43"))


# ============================================================================ K componenti della fase 1

class TestKPhaseOne(unittest.TestCase):
    def test_card_problems_network(self):
        tr = dict(FR.FROZEN_TRAINING)
        p = Path(tempfile.mkdtemp()) / "m.keras"
        p.write_bytes(b"x")
        try:
            card = {**fake_trainer([])({"X_fit": np.zeros((2,)), "y_fit": np.array([0, 1]), "y_dev": np.array([0, 1]), "X_dev": None}, tr, "ResNet-18", 42,
                                       None, lambda s: None, types.SimpleNamespace(record=[]))[0], "weights_file": "m.keras", "weights_sha256": sha(b"x")}
            self.assertEqual(R.card_problems(card, tr, 42, p, "ResNet-18"), [])
            self.assertTrue(R.card_problems(card, tr, 42, p, "LivenessNet"))
            self.assertTrue(R.card_problems({**card, "restored_epoch": 1}, tr, 42, p, "ResNet-18"))
            self.assertTrue(R.card_problems(card, {**tr, "early_stopping_patience": 12}, 42, p, "ResNet-18"))
        finally:
            shutil.rmtree(p.parent)

    def test_save_checkpoint_atomic_never_overwrites(self):
        d = Path(tempfile.mkdtemp())
        try:
            R.save_checkpoint(FakeModel("n", 1, 1), d / "a.keras")
            self.assertEqual(os.listdir(d), ["a.keras"])
            with self.assertRaisesRegex(FR.Failure, "never overwritten"):
                R.save_checkpoint(FakeModel("n", 1, 1), d / "a.keras")

            class Broken:
                def save(self, path):
                    Path(path).write_bytes(b"partial")
                    raise OSError("disk full")
            with self.assertRaises(OSError):
                R.save_checkpoint(Broken(), d / "b.keras")
            self.assertEqual(os.listdir(d), ["a.keras"])
        finally:
            shutil.rmtree(d)

    def test_discard_model(self):
        d = Path(tempfile.mkdtemp())
        try:
            for n in ("M.keras", "M_card.json", "N.keras", f".M.keras.{os.getpid()}.tmp.keras"):
                (d / n).write_text("x")
            removed = R.discard_model(d, "M")
            self.assertEqual(sorted(os.listdir(d)), ["N.keras"])
            self.assertEqual(len(removed), 3)
        finally:
            shutil.rmtree(d)

    def test_run_ids(self):
        st = {"commit": "abcdef0"}
        self.assertEqual(R.run_id_for("run", DATE, st, 1, "nuaa_patience_12"), f"{DATE}-ARCHRUN-p12-abcdef0")
        self.assertEqual(R.run_id_for("run", DATE, st, 3, "nuaa_patience_4"), f"{DATE}-ARCHRUN-p4-abcdef0-r3")
        self.assertEqual(R.run_id_for("estimator", DATE, st, 1), f"{DATE}-ARCHEST-abcdef0")
        with self.assertRaisesRegex(FR.Failure, "--attempt"):
            R.run_id_for("run", DATE, st, True, "nuaa_patience_4")


# ============================================================================ L check-estimator senza TensorFlow

EST_TOLERANCES = (("recalibration_statistics_rel", 1e-4), ("recalibration_statistics_abs", 1e-6),
                  ("recalibration_dev_probability_max_abs_diff", 1e-5), ("recalibration_dev_decisions_discordant_at_0_5", 0))


class StopAtFirstCheck(SR.Budget):
    """Un Budget con l'arresto volontario già richiesto: check-estimator si ferma (INCOMPLETE) al primo controllo del
    tetto, prima di toccare un modello."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.stop_requested = "SIGUSR1"


@contextlib.contextmanager
def estimator_inputs(w: World, bt_over=None, bn_path: Path = None):
    """Gli ingressi di check-estimator senza il mondo della PR #28 (che richiede TensorFlow): BT.load_config restituisce
    una configurazione della PR #28 coerente con quella di questo runner (run BNRECAL, 36 modelli, SHA-256 della
    configurazione della PR #27 fissata, tolleranze) e la configurazione della PR #27 letta da bn_path (default: quella
    fissata); BR.load_inputs restituisce 36 nomi di modelli e nessun frame; BT.load_bnrecal e FR.FrameStore non leggono
    nulla. Basta per i controlli che precedono il primo modello."""
    bn_cfg, _ = BR.load_config(bn_path or w.br_cfg)
    bt = {"bnrecal_run": {"run_id": "20261114-BNRECAL-abcdef0", "models": 36, "config_sha256": sha(w.br_cfg.read_bytes()), "folder": "bnrecal"},
          "tolerances": {k: {"value": v, "source": "t"} for k, v in EST_TOLERANCES}}
    for k, v in (bt_over or {}).items():
        bt["bnrecal_run"][k] = v
    inp = BR.Inputs()
    inp.order = [f"model{i:02d}" for i in range(36)]
    inp.X_extracted, inp.extracted_keys = None, None
    with mock.patch.object(R.BT, "load_config", lambda p, base: (bt, sha(Path(p).read_bytes()), bn_cfg)), \
            mock.patch.object(R.BT, "load_bnrecal", lambda cfg, base, i: None), \
            mock.patch.object(R.BR, "load_inputs", lambda cfg, base: inp), \
            mock.patch.object(R.FR, "FrameStore", lambda *a, **k: None):
        yield


class TestLEstimatorFlow(unittest.TestCase):
    """check-estimator fino al primo modello con un TensorFlow finto: thread impostati e confrontati con la prova, GPU
    nascosta, confronto dei thread con la prova registrato, regola della macchina e delle variabili dei thread prima di
    ogni cartella; il confronto con le configurazioni delle PR #27 e #28 in load_pins; l'esito MISMATCH."""

    def setUp(self):
        self.w = world()
        self.root = self.w.tmp / "est_flow"
        shutil.rmtree(self.root, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.root, True)

    def go(self, w=None, **r):
        w = w or self.w
        log = []
        tfm = FakeTF(log=log)
        h = base_hooks(w, ESTIMATOR_BASE=w.base, TF_MODULE=lambda: tfm, **r)
        h["sr"]["Budget"] = StopAtFirstCheck
        with estimator_inputs(w):
            rc, out, err = run_main(w.argv("check-estimator", out=self.root), h)
        return rc, out, err, self.root / f"{DATE}-ARCHEST-{STATE['commit']}", log

    def test_threads_set_and_gpu_hidden(self):
        rc, out, err, est, log = self.go()
        self.assertEqual(rc, 2, err[-2000:])
        self.assertEqual(log, ["intra 10", "inter 10", "visible GPU []"])
        st = json.loads((est / "status.json").read_text())
        self.assertEqual((st["status"], st["stopped"]["step"]), ("INCOMPLETE", "estimator model00"))
        self.assertIs(st["threads_rule"]["equal"], True)
        self.assertEqual(st["threads"]["effective"], {"intra_op": 10, "inter_op": 10})
        self.assertIs(st["machine_rule"]["equal"], True)
        doc = json.loads((est / "public" / "estimator_check.json").read_text())
        self.assertEqual((doc["status"], doc["models"]), ("INCOMPLETE", []))

    def test_machine_different_refused_before_any_folder(self):
        """La regola della macchina nel flusso di check-estimator: processore diverso dalla prova, rifiuto prima di ogni
        cartella e prima di TensorFlow."""
        rc, out, err, est, log = self.go(MACHINE=lambda: {**MACH, "processor": "Apple M4"})
        self.assertEqual(rc, 1)
        self.assertIn(f"machine rule: processor differ from the official trial {CPA.RUN}", err)
        self.assertFalse(self.root.exists())
        self.assertEqual(log, [])

    def test_environment_variables_at_10_refused_before_any_folder(self):
        with thread_env(TF_NUM_INTRAOP_THREADS="10", TF_NUM_INTEROP_THREADS="10"):
            rc, out, err, est, log = self.go()
        self.assertEqual(rc, 1)
        self.assertIn("thread rule: environment variables TF_NUM_INTRAOP_THREADS '10', TF_NUM_INTEROP_THREADS '10' differ from the official trial", err)
        self.assertFalse(self.root.exists())
        self.assertEqual(log, [])

    def test_pins_against_pr28(self):
        """load_pins con la configurazione della PR #28: run BNRECAL, numero dei modelli e configurazione della PR #27
        (contenuto e SHA-256) confrontati con quelli di questo runner."""
        w = self.w
        cfg, _ = R.load_config(w.run_cfg)

        def pins(**k):
            with estimator_inputs(w, **k), injected(**base_hooks(w)):
                return R.load_pins(cfg, w.base, w.base)
        self.assertEqual(pins().bt_cfg["bnrecal_run"]["models"], 36)
        for over in ({"models": 35}, {"run_id": "20261114-BNRECAL-1234567"}):
            with self.assertRaisesRegex(FR.Failure, "the BNRECAL run or its number of models differ"):
                pins(bt_over=over)
        with self.assertRaisesRegex(FR.Failure, "the PR #27 configuration pinned here is not the one of the PR #28 configuration"):
            pins(bt_over={"config_sha256": "0" * 64})
        other = w.tmp / "bnrecal_other.json"
        doc = bn_doc()
        doc["task"] = "another synthetic configuration"
        other.write_text(json.dumps(doc), encoding="utf-8")
        self.addCleanup(other.unlink)
        with self.assertRaisesRegex(FR.Failure, "the PR #27 configuration pinned here is not the one of the PR #28 configuration"):
            pins(bn_path=other)

    def test_estimator_status(self):
        """L'esito di check-estimator: MISMATCH se un modello non riproduce il run BNRECAL; un arresto resta INCOMPLETE."""
        ok, bad = {"recalibration_pass": True}, {"recalibration_pass": False}
        self.assertEqual(R.estimator_status("PASS", [ok, ok]), "PASS")
        self.assertEqual(R.estimator_status("PASS", [ok, bad, ok]), "MISMATCH")
        self.assertEqual(R.estimator_status("INCOMPLETE", [bad]), "INCOMPLETE")
        self.assertEqual(R.estimator_status("PASS", []), "PASS")


# ============================================================================ Z con TensorFlow (ultime)

def tf_with_threads():
    """TensorFlow con i thread della configurazione (10 e 10) e la GPU nascosta prima di qualunque operazione, come fa il
    runner: i comandi provati dopo nello stesso processo reimpostano gli stessi valori senza errori."""
    tf = CPA.tf_with_threads()
    FR.hide_gpu(tf)
    return tf


def small_branched(tf):
    """Un piccolo modello con rami: due rami con BN che non si alimentano a vicenda, uniti da una somma, poi una BN."""
    L = tf.keras.layers
    inp = L.Input((8, 8, 3), name="input_layer")
    a = L.Conv2D(4, 3, padding="same", name="conv_a")(inp)
    a = L.BatchNormalization(name="bn_a")(a)
    a = L.Activation("relu", name="relu_a")(a)
    b = L.Conv2D(4, 1, padding="same", name="conv_b")(inp)
    b = L.BatchNormalization(name="bn_b")(b)
    x = L.Add(name="add")([a, b])
    x = L.Conv2D(4, 3, padding="same", name="conv_c")(x)
    x = L.BatchNormalization(name="bn_c")(x)
    x = L.GlobalAveragePooling2D(name="gap")(x)
    x = L.Dense(3, name="dense")(x)
    x = L.BatchNormalization(name="bn_d")(x)
    out = L.Dense(2, activation="softmax", name="output")(x)
    m = tf.keras.Model(inp, out)
    rng = np.random.default_rng(1)
    m.set_weights([rng.normal(0, 0.5, w.shape).astype(np.float32) if "variance" not in v.name else np.abs(rng.normal(1, 0.3, w.shape)).astype(np.float32)
                   for v, w in zip(m.weights, m.get_weights())])
    return m


def frames_part(n=37, seed=0):
    rng = np.random.default_rng(seed)
    return {"X": rng.integers(0, 256, (n, 8, 8, 3), dtype=np.uint8), "y": rng.integers(0, 2, n)}


def clone(tf, m):
    c = tf.keras.models.clone_model(m)
    c.set_weights(m.get_weights())
    return c


class TestZAEstimator(unittest.TestCase):
    def setUp(self):
        need_tf(self)
        self.tf = tf_with_threads()

    def test_livenessnet_same_statistics_as_the_accepted_estimator(self):
        """Su LivenessNet del codice di riferimento, stessi dati sintetici: le statistiche di reestimate sono quelle di
        BR.recalibrate bit per bit (float64 e float32), e la copia ha gli stessi tensori."""
        m = AM.build("LivenessNet", (8, 8, 3), CPA.ref_arch())
        rng = np.random.default_rng(3)
        m.set_weights([w + rng.normal(0, 0.1, w.shape).astype(np.float32) for w in m.get_weights()])
        part = frames_part(37)
        a, b = clone(self.tf, m), clone(self.tf, m)
        sa = R.reestimate(a, part, list(AM.LIVENESSNET_BN), 16)
        sb = BR.recalibrate(b, part, list(AM.LIVENESSNET_BN), 16)
        for n in AM.LIVENESSNET_BN:
            for k in ("mean64", "var64", "mean32", "var32"):
                self.assertEqual(sa[n][k].tobytes(), sb[n][k].tobytes(), (n, k))
            self.assertEqual(sa[n]["count"], sb[n]["count"])
        self.assertEqual(BR.snapshot_weights(a), BR.snapshot_weights(b))
        self.assertEqual(sa["bn1_1"]["count"], 37 * 64)
        with self.assertRaisesRegex(FR.Failure, "BN order refused"):
            R.reestimate(clone(self.tf, m), part, list(reversed(AM.LIVENESSNET_BN)), 16)

    def test_order_between_independent_layers_does_not_change_the_result(self):
        m = small_branched(self.tf)
        part = frames_part(41, 2)
        topo = AM.model_topology(m)
        r1 = R.reestimate(clone(self.tf, m), part, ["bn_a", "bn_b", "bn_c", "bn_d"], 16)
        r2 = R.reestimate(clone(self.tf, m), part, ["bn_b", "bn_a", "bn_c", "bn_d"], 16)
        for n in r1:
            for k in ("mean64", "var64", "mean32", "var32"):
                self.assertEqual(r1[n][k].tobytes(), r2[n][k].tobytes(), (n, k))
        self.assertTrue(R.bn_order_problems(["bn_c", "bn_a", "bn_b", "bn_d"], topo))
        with self.assertRaisesRegex(FR.Failure, "bn_c comes before"):
            R.reestimate(clone(self.tf, m), part, ["bn_a", "bn_c", "bn_b", "bn_d"], 16)

    def test_against_brute_force_reference(self):
        """Riferimento a forza bruta scritto qui: per ogni BN, in un ordine topologico, l'ingresso su tutti i frame in una
        sola chiamata, media e varianza di popolazione con NumPy in float64, assegnazione in float32."""
        m = small_branched(self.tf)
        part = frames_part(41, 4)
        ref = clone(self.tf, m)
        X = part["X"].astype(np.float32) / 255.0
        want = {}
        for n in ["bn_b", "bn_a", "bn_c", "bn_d"]:
            layer = ref.get_layer(n)
            a = np.asarray(self.tf.keras.Model(ref.input, layer.input)(X, training=False), np.float64)
            a = a.reshape(-1, a.shape[-1])
            want[n] = (a.mean(axis=0), a.var(axis=0))
            layer.moving_mean.assign(want[n][0].astype(np.float32))
            layer.moving_variance.assign(want[n][1].astype(np.float32))
        got = R.reestimate(clone(self.tf, m), part, ["bn_a", "bn_b", "bn_c", "bn_d"], 16)
        for n, (mu, var) in want.items():
            np.testing.assert_allclose(got[n]["mean64"], mu, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(got[n]["var64"], var, rtol=1e-5, atol=1e-6)
        self.assertEqual(got["bn_d"]["count"], 41)
        self.assertEqual(got["bn_a"]["count"], 41 * 64)

    def test_resnet18_graph_order_and_re_estimation(self):
        m = AM.build("ResNet-18", (16, 16, 3))
        order = R.expected_orders()["ResNet-18"]
        self.assertEqual(R.bn_order_problems(order, AM.model_topology(m)), [])
        self.assertEqual(sorted(AM.model_counts(m)["batchnorm_names"]), sorted(order))
        part = {"X": np.random.default_rng(5).integers(0, 256, (9, 16, 16, 3), dtype=np.uint8), "y": np.array([0, 1] * 4 + [0])}
        c = clone(self.tf, m)
        snap = BR.snapshot_weights(m)
        stats = R.reestimate(c, part, order, 16)
        self.assertEqual(BR.tensor_problems(snap, BR.snapshot_weights(c), BR.moving_keys(c, order), stats), [])
        _, final = BR.evaluate(c, part, 16, "fitting", layers=order)
        cons = BR.consistency(stats, final, 1e-4, 1e-6)
        self.assertEqual(sum(v["channels_outside_tolerance"] for v in cons.values()), 0)


class TestZBRealRun(unittest.TestCase):
    """Un run vero su CPU nel mondo 8×8: trial_trainer vero accorciato a 2 epoche, checkpoint .keras veri, scorer e logit
    veri, ristima vera (LivenessNet e ResNet-18), controlli della PR #27 veri; poi lo stato ristimato ricostruito dal
    checkpoint e dalle statistiche salvate dà gli stessi punteggi del dev del file riservato."""

    def setUp(self):
        need_tf(self)
        tf_with_threads()

    def test_real_run_and_reconstruction(self):
        import tensorflow as tf
        w = world()

        def short(data, training, network, seed, build_model, cap_check, timer):
            card, model = PAC.trial_trainer(data, {**training, "max_epochs": 2}, network, seed, build_model, cap_check, timer)
            return {**card, "max_epochs": training["max_epochs"]}, model
        out_root = w.tmp / "runs_real"
        h = base_hooks(w, ARCH_LOADER=lambda path: CPA.ref_arch(), TRAINER=short, TF_MODULE=lambda: tf)
        rc, out, err = run_main(w.argv("run", out=out_root), h)
        self.addCleanup(shutil.rmtree, out_root, True)
        self.assertEqual(rc, 0, (out + err)[-4000:])
        run = out_root / run_folder(w).name
        st = json.loads((run / "status.json").read_text())
        self.assertEqual((st["status"], len(st["complete_phase_2"]), st["evaluator_mismatches"]), ("PASS", 24, []))
        import csv
        rows = list(csv.DictReader(io.StringIO((run / "public" / "models.csv").read_text())))
        self.assertEqual({r["history_check"] for r in rows if r["network"] == "LivenessNet"}, {"different"})   # schede finte
        self.assertEqual({r["evaluator_agreement"] for r in rows}, {"1"})
        cfg, _ = R.load_config(w.run_cfg)
        with injected(**base_hooks(w)):
            pins = R.load_pins(cfg, w.base)
            inp = PAC.load_inputs(pins.cfg_pr, pins.pcfg, w.base)
        store = FR.FrameStore(inp, inp.X_extracted, inp.extracted_keys)
        for name, net in (("ResNet-18_fold1_seed42", "ResNet-18"), ("LivenessNet_fold2_seed43", "LivenessNet")):
            fold = int(name.split("fold")[1][0])
            dev = BR.frames(store, inp, "dev", "arm_b", fold)
            s = json.loads((run / "restricted" / f"{name}_bn_statistics.json").read_text())
            order = R.expected_orders()[net]
            self.assertEqual(s["order"], order)
            self.assertEqual(list(s["layers"]), order)
            m = tf.keras.models.load_model(str(run / "restricted" / f"{name}.keras"), compile=False)
            ordinary = list(csv.DictReader(io.StringIO((run / "restricted" / f"{name}_dev_ordinary.csv").read_text())))
            ev, _ = BR.evaluate(m, dev, 16, "ordinary")
            self.assertEqual([r["score_attack"] for r in ordinary], [repr(float(v)) for v in ev["p"][:, 1]])
            for n in order:
                for k, attr in (("installed_moving_mean_float32", "moving_mean"), ("installed_moving_variance_float32", "moving_variance")):
                    vals = s["layers"][n][k]
                    self.assertEqual([float(np.float32(v)) for v in vals], vals, (n, k))        # valori float32 esatti nel file
                    getattr(m.get_layer(n), attr).assign(np.array(vals, np.float32))
            ev, _ = BR.evaluate(m, dev, 16, "re-estimated")
            sc = list(csv.DictReader(io.StringIO((run / "restricted" / f"{name}_dev_re-estimated.csv").read_text())))
            self.assertEqual([r["sample_id"] for r in sc], dev["keys"])
            self.assertEqual([r["score_attack"] for r in sc], [repr(float(v)) for v in ev["p"][:, 1]])
            self.assertEqual(s["dev_re_estimated"]["n"], len(sc))


class TestZCCheckEstimator(unittest.TestCase):
    """check-estimator sul mondo della suite della PR #28 (36 modelli Keras minuscoli e un run BNRECAL vero): PASS;
    statistiche del run BNRECAL alterate per un modello: MISMATCH documentato e un run non parte con quel record; tensori
    addestrati cambiati o statistiche incoerenti: FAIL; tetto: INCOMPLETE. La regola dei thread e della macchina di
    check-estimator e il confronto delle configurazioni delle PR #27 e #28 sono provati senza TensorFlow (TestLEstimatorFlow)."""

    @classmethod
    def setUpClass(cls):
        if not TF_AVAILABLE or not REF_ARCH.is_file():
            return
        tf_with_threads()
        import check_nuaa_bn_test as BTS
        cls.BTS, cls.bw = BTS, BTS.world()

    def setUp(self):
        need_tf(self)
        tf_with_threads()
        self.w = world()
        self.config_for()
        self.addCleanup(self.w.write_estimator_record)
        self.addCleanup(lambda: self.w.write_run_cfg(self.w.run_doc()))
        self.roots = []
        self.addCleanup(lambda: [shutil.rmtree(r, ignore_errors=True) for r in self.roots])

    def config_for(self, mutate=None) -> dict:
        bw, w = self.bw, self.w
        d = w.run_doc()
        d["estimator_check"]["bn_recalibration_config"] = {"path": str(bw.bn_config), "sha256": sha(bw.bn_config.read_bytes())}
        d["estimator_check"]["bn_test_config"] = {"path": str(bw.bt_config), "sha256": sha(bw.bt_config.read_bytes())}
        d["estimator_check"]["bnrecal_run_id"] = json.loads(bw.bt_config.read_text())["bnrecal_run"]["run_id"]
        if mutate:
            mutate(d)
        w.write_run_cfg(d)
        return d

    def go(self, name: str, **r):
        root = self.w.tmp / name
        self.roots.append(root)
        h = base_hooks(self.w, ESTIMATOR_BASE=self.bw.base, **r)
        h["sr"].update(self.BTS.hooks(self.bw)["sr"])
        rc, out, err = run_main(self.w.argv("check-estimator", out=root), h)
        return rc, out, err, root / f"{DATE}-ARCHEST-{STATE['commit']}"

    def test_pass_and_mismatch(self):
        rc, out, err, est = self.go("est_pass")
        self.assertEqual(rc, 0, (out + err)[-4000:])
        doc = json.loads((est / "public" / "estimator_check.json").read_text())
        self.assertEqual((doc["status"], len(doc["models"]), doc["official"]), ("PASS", 36, False))
        self.assertTrue(all(r["statistics_pass"] and r["dev_pass"] for r in doc["models"]))
        self.assertTrue(doc["threads_rule"]["equal"])
        self.assertEqual((doc["cap_minutes"], doc["cap_unit"]), (240, R.EST_CAP_UNIT))
        cfg, cfg_sha = R.load_config(self.w.run_cfg)
        self.assertEqual(R.load_estimator_record(est, cfg_sha, R.check_code(cfg), False)["models"], 36)
        real = R.BT.load_bnrecal

        def altered(cfg, base, inp):
            real(cfg, base, inp)
            name = inp.order[5]
            a, v = inp.bnrecal[name]["stats"]["bn2_1"]
            inp.bnrecal[name]["stats"]["bn2_1"] = (a + np.float32(1e-2), v)
        with mock.patch.object(R.BT, "load_bnrecal", altered):
            rc, out, err, est2 = self.go("est_mismatch")
        self.assertEqual(rc, 4, (out + err)[-3000:])
        st = json.loads((est2 / "status.json").read_text())
        self.assertEqual((st["status"], st["models_reproduced"]), ("MISMATCH", 35))
        import csv
        rows = list(csv.DictReader(io.StringIO((est2 / "public" / "estimator_check.csv").read_text())))
        self.assertEqual(sum(r["statistics_pass"] == "0" for r in rows), 1)
        with self.assertRaisesRegex(FR.Failure, "not a PASS of check-estimator"):
            R.load_estimator_record(est2, cfg_sha, R.check_code(cfg), False)

    def wrapped(self, alter):
        real = R.reestimate

        def re(copy, part, order, batch, budget=None, name=""):
            stats = real(copy, part, order, batch, budget, name)
            alter(copy, stats)
            return stats
        return mock.patch.object(R, "reestimate", re)

    def test_tensors_and_consistency_checked(self):
        with self.wrapped(lambda c, s: c.get_layer("bn1_1").gamma.assign(c.get_layer("bn1_1").gamma + 1.0)):
            rc, out, err, est = self.go("est_tensors")
        self.assertEqual(rc, 1)
        self.assertIn("trained tensors changed", json.loads((est / "status.json").read_text())["reason"])

        def inconsistent(c, s):
            s["bn1_2"]["mean32"] = s["bn1_2"]["mean32"] + np.float32(0.5)
            c.get_layer("bn1_2").moving_mean.assign(s["bn1_2"]["mean32"])
        with self.wrapped(inconsistent):
            rc, out, err, est = self.go("est_consistency")
        self.assertEqual(rc, 1)
        self.assertIn("consistency check failed", json.loads((est / "status.json").read_text())["reason"])

    def test_cap(self):
        import time
        root = self.w.tmp / "est_cap"
        self.roots.append(root)
        hooks = base_hooks(self.w, ESTIMATOR_BASE=self.bw.base)
        hooks["sr"].update(self.BTS.hooks(self.bw)["sr"])
        calls = {"n": 0}

        def clock():                     # la prima lettura (avvio del Budget) è normale, poi un salto di 10^7 s
            calls["n"] += 1
            return time.monotonic() + (0 if calls["n"] == 1 else 10 ** 7)
        hooks["sr"]["CLOCK"] = clock
        rc, out, err = run_main(self.w.argv("check-estimator", out=root), hooks)
        self.assertEqual(rc, 2, (out + err)[-3000:])
        st = json.loads((root / f"{DATE}-ARCHEST-{STATE['commit']}" / "status.json").read_text())
        self.assertEqual(st["status"], "INCOMPLETE")
        self.assertTrue(st["stopped"]["step"].startswith("estimator arm_b_fold1_seed42"), st["stopped"])
        self.assertEqual(st["models_checked"], 0)

    def test_cap_checks_inside_a_model(self):
        """Arresto richiesto (SIGUSR1 nel Budget) in tre punti del primo modello: prima della ristima (il tetto è passato a
        reestimate), dopo la ristima (tetto prima del passaggio di coerenza), dopo il passaggio di coerenza (tetto prima del
        dev ristimato)."""
        holder = {}
        orig_budget = SR.Budget

        class B(orig_budget):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                holder["b"] = self

        def stop(*a, **k):
            holder["b"].stop_requested = "SIGUSR1"
        real_arch, real_re, real_eval = BR.check_architecture, R.reestimate, BR.evaluate

        def arch(*a, **k):
            real_arch(*a, **k)
            stop()

        def re(*a, **k):
            out = real_re(*a, **k)
            stop()
            return out

        def ev(*a, **k):
            out = real_eval(*a, **k)
            if k.get("layers"):
                stop()
            return out
        for i, (patch, step) in enumerate(((mock.patch.object(BR, "check_architecture", arch), "recalibration bn1_1 of arm_b_fold1_seed42"),
                                           (mock.patch.object(R, "reestimate", re), "consistency pass arm_b_fold1_seed42"),
                                           (mock.patch.object(BR, "evaluate", ev), "dev after arm_b_fold1_seed42"))):
            root = self.w.tmp / f"est_stop{i}"
            self.roots.append(root)
            hooks = base_hooks(self.w, ESTIMATOR_BASE=self.bw.base)
            hooks["sr"].update(self.BTS.hooks(self.bw)["sr"])
            hooks["sr"]["Budget"] = B
            with patch:
                rc, out, err = run_main(self.w.argv("check-estimator", out=root), hooks)
            self.assertEqual(rc, 2, (out + err)[-3000:])
            st = json.loads((root / f"{DATE}-ARCHEST-{STATE['commit']}" / "status.json").read_text())
            self.assertEqual((st["status"], st["stopped"]["step"], st["models_checked"]), ("INCOMPLETE", step, 0))


class TestZDPhaseTwoChecks(unittest.TestCase):
    """I controlli della fase 2 (reestimate_model) su un LivenessNet 8×8 salvato: passaggio normale; tensori addestrati
    cambiati, statistica installata diversa, statistiche incoerenti, checkpoint di un'altra architettura o cambiato, copia
    diversa prima della ristima, softmax dello stato ordinario: Failure; tetto prima di ogni passaggio: CapExceeded con il
    passo atteso."""

    def setUp(self):
        need_tf(self)
        self.tf = tf_with_threads()
        self.d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.d, True)
        m = AM.build("LivenessNet", (8, 8, 3), CPA.ref_arch())
        rng = np.random.default_rng(7)
        m.set_weights([w + rng.normal(0, 0.1, w.shape).astype(np.float32) for w in m.get_weights()])
        m.save(self.d / "LivenessNet_fold1_seed42.keras")
        c = AM.expected_counts("LivenessNet", (8, 8, 3))
        ctx = R.Ctx()
        ctx.bn_cfg, ctx.rdir, ctx.budget = bn_doc(), self.d, SR.Budget(60)
        ctx.cfg = {"estimator": {"batch": 16, "orders": R.expected_orders()}}
        ctx.cfg_pr = {"architectures": {"LivenessNet": {"input_shape": [8, 8, 3], "trainable_parameters": c["trainable"],
                                                        "non_trainable_parameters": c["non_trainable"], "batchnorm_layers": c["batchnorm_layers"]}}}
        self.ctx = ctx
        self.rec = {"name": "LivenessNet_fold1_seed42", "network": "LivenessNet", "weights_sha256": FR.sha256_file(self.d / "LivenessNet_fold1_seed42.keras")}
        self.fit, self.dev = frames_part(33, 8), frames_part(17, 9)

    def run_it(self):
        return R.reestimate_model(self.ctx, self.rec, self.fit, self.dev)

    def test_normal(self):
        res = self.run_it()
        self.assertEqual(sorted(res["stats"]), sorted(AM.LIVENESSNET_BN))
        self.assertEqual((res["re_estimated"]["p"].shape, res["ordinary"]["p"].shape), ((17,), (17,)))
        self.assertEqual(sum(c["channels_outside_tolerance"] for c in res["cons"].values()), 0)
        m = self.tf.keras.models.load_model(str(self.d / "LivenessNet_fold1_seed42.keras"), compile=False)
        ev, _ = BR.evaluate(m, self.dev, 16, "ordinary")
        self.assertEqual(res["ordinary"]["p"].tolist(), ev["p"][:, 1].tolist())

    def wrapped(self, alter):
        real = R.reestimate

        def re(copy, part, order, batch, budget=None, name=""):
            stats = real(copy, part, order, batch, budget, name)
            alter(copy, stats)
            return stats
        return mock.patch.object(R, "reestimate", re)

    def test_trained_tensor_changed(self):
        with self.wrapped(lambda c, s: c.get_layer("bn1_1").gamma.assign(c.get_layer("bn1_1").gamma + 1.0)):
            with self.assertRaisesRegex(FR.Failure, "trained tensors changed"):
                self.run_it()

    def test_installed_statistic_differs(self):
        with self.wrapped(lambda c, s: c.get_layer("bn2_1").moving_mean.assign(c.get_layer("bn2_1").moving_mean + 1.0)):
            with self.assertRaisesRegex(FR.Failure, "does not hold the installed"):
                self.run_it()

    def test_inconsistent_statistics(self):
        def alter(c, s):
            s["bn1_2"]["mean32"] = s["bn1_2"]["mean32"] + np.float32(0.5)
            c.get_layer("bn1_2").moving_mean.assign(s["bn1_2"]["mean32"])
        with self.wrapped(alter):
            with self.assertRaisesRegex(FR.Failure, "consistency check failed"):
                self.run_it()

    def test_count_checked(self):
        with self.wrapped(lambda c, s: s["bn_dense"].update(count=s["bn_dense"]["count"] + 1)):
            with self.assertRaisesRegex(FR.Failure, "counted a different number"):
                self.run_it()

    def test_wrong_architecture(self):
        self.ctx.cfg_pr["architectures"]["LivenessNet"]["trainable_parameters"] += 1
        with self.assertRaisesRegex(FR.Failure, "trainable"):
            self.run_it()

    def test_checkpoint_changed(self):
        self.rec["weights_sha256"] = "0" * 64
        with self.assertRaisesRegex(FR.Failure, "SHA-256 changed"):
            self.run_it()

    def test_copy_differs_before_the_re_estimation(self):
        real = self.tf.keras.models.clone_model

        def clone_model(m):
            c = real(m)
            c.set_weights = lambda w: None
            return c
        with mock.patch.object(self.tf.keras.models, "clone_model", clone_model):
            with self.assertRaisesRegex(FR.Failure, "the in-memory copy differs from the original before the re-estimation"):
                self.run_it()

    def test_ordinary_softmax_checked(self):
        self.ctx.bn_cfg["tolerances"]["softmax_of_logits_max_abs_diff"]["value"] = 1e-300
        with self.assertRaisesRegex(FR.Failure, "ordinary dev probabilities"):
            self.run_it()

    def stop_when(self, evaluate_call=None, after_reestimate=False):
        """Un arresto richiesto (SIGUSR1 nel Budget) dopo la n-esima chiamata di BR.evaluate o dopo la ristima."""
        budget, calls = self.ctx.budget, {"n": 0}
        real_eval, real_re = BR.evaluate, R.reestimate

        def evaluate(*a, **k):
            out = real_eval(*a, **k)
            calls["n"] += 1
            if calls["n"] == evaluate_call:
                budget.stop_requested = "SIGUSR1"
            return out

        def re(*a, **k):
            out = real_re(*a, **k)
            if after_reestimate:
                budget.stop_requested = "SIGUSR1"
            return out
        with mock.patch.object(BR, "evaluate", evaluate), mock.patch.object(R, "reestimate", re):
            with self.assertRaises(SR.CapExceeded) as cm:
                self.run_it()
        return cm.exception.step

    def test_cap_checks(self):
        self.ctx.budget.stop_requested = "SIGUSR1"
        with self.assertRaises(SR.CapExceeded) as cm:
            self.run_it()
        self.assertEqual(cm.exception.step, "ordinary dev scoring LivenessNet_fold1_seed42")
        self.ctx.budget.stop_requested = None
        self.assertEqual(self.stop_when(evaluate_call=1), "recalibration bn1_1 of LivenessNet_fold1_seed42")
        self.ctx.budget.stop_requested = None
        self.assertEqual(self.stop_when(after_reestimate=True), "consistency pass LivenessNet_fold1_seed42")
        self.ctx.budget.stop_requested = None
        self.assertEqual(self.stop_when(evaluate_call=2), "re-estimated dev scoring LivenessNet_fold1_seed42")


REPO_DIRS = ("scripts", "docs", "delivery", "results", "export", "src")
REPO_BEFORE = CFR.snapshot(*(ROOT / d for d in REPO_DIRS))


class ZZRepositoryUntouched(unittest.TestCase):
    """Ultima classe: il repository (scripts, docs, delivery, results, export, src) uguale, file per file (dimensione e
    data), a com'era all'import della suite; e gli unici salti possibili sono quelli dichiarati."""

    def test_repository_untouched(self):
        self.assertEqual(CFR.snapshot(*(ROOT / d for d in REPO_DIRS)), REPO_BEFORE)

    def test_declared_skips_only(self):
        """Nel sorgente di questa suite l'unico salto è quello di skip_missing (nessun decoratore di salto, nessun altro
        skipTest) e ogni sua chiamata nomina un ingresso dichiarato in MISSING_INPUT; main controlla inoltre che ogni
        salto dell'esecuzione abbia uno dei motivi dichiarati."""
        text = Path(__file__).read_text(encoding="utf-8")
        self.assertEqual(text.count("." + "skipTest("), 1)
        for word in ("unittest." + "skip", "@" + "skip", "Skip" + "Test", "expected" + "Failure"):
            self.assertNotIn(word, text)
        named = re.findall(r"skip_missing\(\w+, \"([^\"]+)\"\)", text)
        self.assertTrue(named)
        self.assertEqual(sorted(set(named)), sorted(MISSING_INPUT))


def main() -> int:
    tmp_outside_repo()
    argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a != "--verbose"] + (["-v"] if "--verbose" in sys.argv else [])
    try:
        result = unittest.main(argv=argv, exit=False, verbosity=1).result
    finally:
        if WORLD is not None:
            WORLD.close()
        bts = sys.modules.get("check_nuaa_bn_test")
        if bts is not None and bts.WORLD is not None:
            bts.WORLD.close()
    changed = CFR.snapshot(*(ROOT / d for d in REPO_DIRS)) != REPO_BEFORE
    skipped = [f"{t.id()}: {r}" for t, r in result.skipped]
    undeclared = [f"{t.id()}: {r}" for t, r in result.skipped if r not in {f"missing input: {v}" for v in MISSING_INPUT.values()}]
    print(f"tests run: {result.testsRun}, failures: {len(result.failures)}, errors: {len(result.errors)}, skipped: {len(skipped)}")
    for s in skipped:
        print(f"skipped {s}")
    if changed:
        print("ERROR: the repository (scripts, docs, delivery, results, export, src) changed during the tests")
    for s in undeclared:
        print(f"ERROR: skipped for a reason not declared in MISSING_INPUT: {s}")
    ok = result.wasSuccessful() and not changed and not undeclared
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
