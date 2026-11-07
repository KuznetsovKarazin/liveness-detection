"""
Test di scripts/nuaa_degenerate_trainings.py (training degeneri dalle curve train/dev), offline e deterministici.

Mondo sintetico in cartelle temporanee, con identificativi diversi da NUAA (soggetti 1101 e 1102): run di ablazione con
tre bracci, un fold e due seed (6 modelli, dev di 8 frame), run dei fold con un modello (dev di 12 frame), tentativo
fallito con una scheda. Storie costruite a mano, con l'esito dell'EarlyStopping calcolato a mano nei commenti (mai con
la funzione sotto test):
  arm_b f1 s42        degenere: val_loss 5,6,7,8,9 sempre a 0,5 -> migliore 1, arresto 5
  arm_a_star f1 s42   non degenere: val_loss strettamente decrescente, 12 epoche, ripristino 12, recupero all'epoca 4
  arm_b_prime f1 s42  al confine: migliore 3 con val_acc 0,25, arresto 7, nessun recupero
  arm_b f1 s43        ricade: recupero all'epoca 1, migliore 2 (0,875), poi a 0,5, arresto 6
  arm_a_star f1 s43   degenere con migliore 2: val_loss 9,3,4,5,6,7 a 0,5 -> arresto 6
  arm_b_prime f1 s43  non degenere fermato dalla regola proprio all'epoca 12 (migliore 8): restored_by «early stopping»
  folds f1 s42        non degenere, 12 epoche senza arresto
Ogni controllo ha un test che lo fa scattare con tutte le ancore ricalcolate (fallisce la regola, non il checksum).
Il test sugli ingressi veri gira solo con --real (sola lettura); altrimenti è saltato con il motivo dichiarato in
MISSING_INPUT. Alla fine si controlla che results/, export/ e delivery/ del repository non siano cambiati.

  .venv/bin/python -I scripts/check_nuaa_degenerate_trainings.py [--verbose] [--real]

Variabile facoltativa DEGEN_INPUT_ROOT: radice degli ingressi veri e della configurazione consegnata (predefinita: la
radice del repository; il controllo delle mutazioni la sposta su una cartella vuota, così i mutanti non leggono mai i
dati veri).
"""
import contextlib
import copy
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nuaa_degenerate_trainings as DG  # noqa: E402

REAL = "--real" in sys.argv
REAL_ROOT = Path(os.environ.get("DEGEN_INPUT_ROOT") or ROOT)
MISSING_INPUT = {
    "real not requested": "the read-only test on the real inputs runs only with --real",
    "real inputs": "the ablation run, the folds run or the failed attempt folders are not present",
    "delivered configuration": "delivery/degenerate/nuaa_degenerate_trainings.json is not present",
}
STATE = {"commit": "abcdef0", "dirty": False}
DATE = "20261107"
RUN_ID = f"{DATE}-DEGEN-abcdef0"
ABL_ID, FOLDS_ID, FAILED_ID = "20261106-SESSABLRUN-0000001", "20261103-FOLDSRUN-0000002", "20261106-SESSABLRUN-0000003"
ARMS = ["arm_b", "arm_a_star", "arm_b_prime"]
SUBJ_BF, SUBJ_AT = "1101", "1102"
TRAIN_SPE = {"arm_b": 10.0, "arm_a_star": 20.0, "arm_b_prime": 5.0, "folds": 30.0}   # secondi per epoca, scelti a mano


def skip_missing(test: unittest.TestCase, what: str):
    """Unico modo di saltare un test di questa suite: un motivo dichiarato in MISSING_INPUT."""
    test.skipTest(f"missing input: {MISSING_INPUT[what]}")


def sha(b) -> str:
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def jbytes(doc) -> bytes:
    return (json.dumps(doc, indent=1) + "\n").encode("utf-8")


def hist(val_loss, val_acc, acc=None, loss=None, lr=None) -> dict:
    n = len(val_loss)
    assert len(val_acc) == n
    return {"loss": loss or [round(0.2 / (i + 1), 4) for i in range(n)], "accuracy": acc or [min(0.99, 0.9 + 0.01 * i) for i in range(n)],
            "val_loss": list(map(float, val_loss)), "val_accuracy": list(map(float, val_acc)),
            "lr": lr or [0.0010000000474974513] * min(n, 3) + [0.0005000000237487257] * max(0, n - 3)}


# storie a mano: (val_loss, val_acc, epochs_run, restored_epoch, restored_by) con l'esito calcolato a mano
H = {
    # migliore 1 (5); epoche 2-5 peggiori: wait 1..4, all'epoca 5 wait = 4 = patience -> arresto 5, ripristino 1
    "arm_b_fold1_seed42": ([5, 6, 7, 8, 9], [0.5] * 5, 5, 1, "early stopping"),
    # strettamente decrescente: ogni epoca migliora, nessun arresto, ripristino 12 alla fine delle epoche
    "arm_a_star_fold1_seed42": ([5, 4, 3, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06],
                                [0.5, 0.5, 0.5, 0.875, 0.875, 0.875, 0.875, 1, 1, 1, 1, 1], 12, 12, "end of epochs (set_weights)"),
    # 4 < 5 all'epoca 3 azzera wait; epoche 4-7 peggiori -> arresto 7, ripristino 3 (val_acc 0,25: confine)
    # (5 all'epoca 2 è uguale, non un miglioramento: primo val_loss sotto l'epoca 1 all'epoca 3)
    "arm_b_prime_fold1_seed42": ([5, 5, 4, 7, 8, 9, 10], [0.5, 0.5, 0.25, 0.375, 0.5, 0.5, 0.5], 7, 3, "early stopping"),
    # migliore 2 (0,4); epoche 3-6 peggiori -> arresto 6, ripristino 2 (0,875); recupero all'epoca 1 (0,75 e 0,5 < 1)
    "arm_b_fold1_seed43": ([0.5, 0.4, 2, 3, 4, 5], [0.75, 0.875, 0.5, 0.5, 0.5, 0.5], 6, 2, "early stopping"),
    # migliore 2 (3 < 9); epoche 3-6 peggiori -> arresto 6, ripristino 2, val_acc 0,5: degenere
    "arm_a_star_fold1_seed43": ([9, 3, 4, 5, 6, 7], [0.5] * 6, 6, 2, "early stopping"),
    # migliore 8 (0,2); epoche 9-12 peggiori: wait 4 all'epoca 12 -> fermato dalla regola proprio all'ultima epoca
    "arm_b_prime_fold1_seed43": ([3, 2, 1, 0.5, 0.4, 0.3, 0.25, 0.2, 0.21, 0.22, 0.23, 0.24],
                                 [0.5, 0.625, 0.75, 0.875, 1, 1, 1, 1, 1, 1, 1, 1], 12, 8, "early stopping"),
    # run dei fold: strettamente decrescente, 12 epoche
    "fold1_seed42": ([40, 20, 10, 5, 2, 1, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1],
                     [0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 1], 12, 12, "end of epochs (set_weights)"),
}
DEGENERATE = {"arm_b_fold1_seed42", "arm_a_star_fold1_seed43"}
# soglie dev congelate, diverse per modello (uno scambio di righe fra modelli dello stesso fold non le riallinea)
THR_VALUE = {"arm_b_fold1_seed42": 0.95, "arm_a_star_fold1_seed42": 0.55, "arm_b_prime_fold1_seed42": 0.6, "arm_b_fold1_seed43": 0.85,
             "arm_a_star_fold1_seed43": 0.15, "arm_b_prime_fold1_seed43": 0.5, "fold1_seed42": 0.7}


def as_float(v: str) -> float:
    try:
        return float(v)
    except ValueError:
        return 0.0


def score_repr(v: str) -> str:
    """Come i runner: repr(float(score)) (un valore non numerico resta com'è: lo rifiuta il controllo dello score)."""
    try:
        return repr(float(v))
    except ValueError:
        return v


def dev_scores(val_acc: float, n: int, constant: bool) -> list:
    """(label, score) di un dev bilanciato di n frame con accuratezza alla soglia 0,5 uguale a val_acc."""
    half = n // 2
    k = round(val_acc * n)
    if constant:                                    # uscita costante «attacco»: metà giusta (solo per 0,5)
        assert k == half
        return [(0, 1.0)] * half + [(1, 1.0)] * half
    bf_ok = min(k, half) if k > half else k // 2 + k % 2
    at_ok = k - bf_ok
    return ([(0, 0.1)] * bf_ok + [(0, 0.9)] * (half - bf_ok) + [(1, 0.8)] * at_ok + [(1, 0.2)] * (half - at_ok))


def make_config(abl_folder: str) -> dict:
    """Configurazione del mondo sintetico (le ancore sono riempite da World.write)."""
    return {"schema": DG.CONFIG_SCHEMA, "task": "synthetic study", "request": {"date": "2026-11-06", "card": "C2", "text": "test"},
            "inputs": {"ablation_run": {"label": "sessablrun", "run_id": ABL_ID, "folder": abl_folder, "status_sha256": "",
                                        "public_sums_sha256": "", "restricted_sums_sha256": "", "run_log_sha256": "", "arms": list(ARMS),
                                        "folds": [1], "seeds": [42, 43], "models": 6, "dev_rows": 8},
                       "folds_run": {"label": "foldsrun", "run_id": FOLDS_ID, "folder": "runs/folds", "status_sha256": "",
                                     "public_sums_sha256": "", "restricted_sums_sha256": "", "run_log_sha256": "", "folds": [1],
                                     "seeds": [42], "models": 1, "dev_rows": 12},
                       "failed_attempt": {"label": "sessablrun_failed", "run_id": FAILED_ID, "folder": "runs/failed", "status_sha256": "",
                                          "restricted_sums_sha256": "", "cards": {}, "note": "synthetic"}},
            "recipe": {"max_epochs": 12, "early_stopping_patience": 4, "monitor": "val_loss", "min_delta": 0, "restore_best_weights": True,
                       "batch": 16, "lr": 0.001, "bn_momentum": 0.99, "source": "test"},
            "criteria": {"degenerate": {"restored_epochs": [1, 2], "val_acc_at_best": 0.5, "tolerance": 1e-06, "text": "degenerate rule"},
                         "saturation": {"val_accuracy": 0.5, "tolerance": 1e-06, "text": "saturation"},
                         "recovery": {"val_accuracy_min": 0.75, "val_loss_max": 1.0, "text": "recovery rule",
                                      "sensitivity": {"val_accuracy_min": [0.7, 0.75, 0.8], "val_loss_max": [0.7, 1.0, 1.5]}},
                         "checkpoint_agreement_tolerance": 1e-06, "below_chance_val_accuracy": 0.45, "boundary_restored_epoch_max": 3,
                         "train_accuracy_reference": 0.9, "late_recovery_after_epoch": 5, "defined_after_the_run": True,
                         "pair_train_epochs": 3, "train_side_epochs": 5, "bn_residual_epochs": 5},
            "alternative_rules": [{"label": "patience 4 (run)", "monitor": "val_loss", "patience": 4, "start_from_epoch": 0},
                                  {"label": "patience 2", "monitor": "val_loss", "patience": 2, "start_from_epoch": 0},
                                  {"label": "patience 8", "monitor": "val_loss", "patience": 8, "start_from_epoch": 0},
                                  {"label": "start_from_epoch 3", "monitor": "val_loss", "patience": 4, "start_from_epoch": 3},
                                  {"label": "monitor val_accuracy", "monitor": "val_accuracy", "patience": 4, "start_from_epoch": 0}],
            "examples": [{"run": "sessablrun", "arm": "arm_b", "fold": 1, "seed": 42, "why": "degenerate"},
                         {"run": "foldsrun", "arm": "folds", "fold": 1, "seed": 42, "why": "folds run"}],
            "time_scenarios": [{"label": "all x 12", "selection": "all", "epochs": 12, "with_overhead": True},
                               {"label": "b and a_star x 12", "selection": "arms: arm_b, arm_a_star", "epochs": 12, "with_overhead": True},
                               {"label": "degenerate x 12", "selection": "degenerate", "epochs": 12, "with_overhead": False},
                               {"label": "stopped x 12", "selection": "stopped before max_epochs", "epochs": 12, "with_overhead": True}],
            "declarations": {**{k: True for k in DG.DECLARATION_FLAGS}, **{k: f"text {k}" for k in DG.DECLARATION_TEXTS}},
            "outputs": {"public": "public files", "restricted": "none"}}


class World:
    """Ingressi sintetici coerenti; write() li scrive con tutte le ancore ricalcolate (SHA256SUMS, status, configurazione)."""

    def __init__(self, abl_folder: str = "runs/ablation"):
        self.tmp = Path(tempfile.mkdtemp(prefix="degen_"))
        self.base = self.tmp / "in"
        self.out = self.tmp / "out"
        self.cfg_path = self.tmp / "cfg" / "nuaa_degenerate_trainings.json"
        self.abl_folder = abl_folder
        self.cards = {n: self.card(n, "folds_run" if n.startswith("fold") else "ablation_run") for n in H}
        self.failed_cards = {"arm_b_fold1_seed42": self.card("arm_b_fold1_seed42", "failed_attempt")}
        self.dev = {n: self.dev_rows(n) for n in H}
        self.abl_status_extra = {}
        self.failed_status = {"run_id": FAILED_ID, "status": "FAIL", "passed": False}
        self.failed_status_extra = {}               # campi che sostituiscono quelli calcolati (kept_complete_models)
        self.listed_extra = {}                      # nome -> sha aggiunto al SHA256SUMS del run di ablazione
        self.sums_extra = b""                       # righe aggiunte al SHA256SUMS del run di ablazione
        self.thr_extra = {}                         # modello -> campi che sostituiscono quelli della soglia
        self.dev_raw = {}                           # modello -> funzione(byte) -> byte sul dev_scores scritto
        self.symlink = set()                        # file del riservato dell'ablazione scritti come collegamento
        self.run_log_line = "run folder: x (public/ and restricted/), 6/6 trainings, wall 1000 s, aggregate 1.0 s"
        self.cfg = make_config(abl_folder)
        self.write()

    def card(self, name: str, group: str) -> dict:
        vl, va, n, r, by = H[name]
        arm, fold, seed = DG.name_fields(name)
        h = hist(vl, va)
        c = {"arch": "LivenessNet", "train_seed": seed, "n_fitting": 150 if group != "folds_run" else 230,
             "n_dev": 12 if group == "folds_run" else 8, "max_epochs": 12, "epochs_run": n, "restored_epoch": r, "restored_by": by,
             "best_val_loss": h["val_loss"][r - 1], "val_acc_at_best": h["val_accuracy"][r - 1], "history": h, "device": "CPU",
             "visible_devices": ["CPU"], "lr": 0.001, "batch": 16, "early_stopping_patience": 4, "class_weight": {"0": 1.0, "1": 1.0},
             "train_seconds": round(TRAIN_SPE[arm] * n, 1), "test_evaluation": "none", "cpu_nondeterminism": "declared",
             "weights_file": f"{name}_LivenessNet.keras", "weights_sha256": sha(f"weights {name}"), "saved": "2026-11-06T08:00:00+0100",
             "fold": fold, "fitting_keys_sha256": sha("fit"), "dev_keys_sha256": "AUTO", "val_loss_recomputed": h["val_loss"][r - 1],
             "test_scored_after_threshold": "2026-11-06T08:00:20+0100"}
        if group != "folds_run":
            c["arm"] = arm
        if group == "ablation_run":
            c["val_loss_check"] = "logits"
            c["val_loss_recomputed_clipped"] = h["val_loss"][r - 1]
        return c

    def dev_rows(self, name: str) -> list:
        c = self.cards[name]
        arm, fold, seed = DG.name_fields(name)
        rows = []
        for i, (lab, s) in enumerate(dev_scores(c["val_acc_at_best"], c["n_dev"], name in DEGENERATE and name != "arm_a_star_fold1_seed43")):
            subj = SUBJ_AT if lab else SUBJ_BF
            r = {"fold": str(fold), "train_seed": str(seed), "role": "dev", "sample_id": f"{'attack' if lab else 'real'}/{subj}_{subj}_00_0{lab}_01_{i}.jpg",
                 "subject_id": subj, "session_id": "01", "label": str(lab), "score_attack": repr(float(s)), "prediction_05": str(int(s > 0.5)),
                 "prediction_dev": "AUTO", "threshold_dev": "AUTO", "file_sha256": sha(f"img {i}"), "checkpoint_sha256": c["weights_sha256"],
                 "threshold_sha256": "AUTO"}
            if not name.startswith("fold"):
                r = {"arm": arm, **r}
            rows.append(r)
        return rows

    def write(self) -> None:
        """Riscrive gli ingressi con tutte le ancore coerenti (dopo ogni modifica di un test)."""
        if self.base.exists():
            shutil.rmtree(self.base)
        cfg = self.cfg
        for g, folder, names in (("ablation_run", self.abl_folder, [n for n in H if not n.startswith("fold")]),
                                 ("folds_run", "runs/folds", [n for n in H if n.startswith("fold")])):
            d = self.base / folder
            (d / "restricted").mkdir(parents=True)
            (d / "public").mkdir()
            files = {}
            for n in names:
                c = dict(self.cards[n])
                rows = [dict(r) for r in self.dev[n]]
                if c["dev_keys_sha256"] == "AUTO":           # come i runner: SHA-256 dei sample_id uniti da «\n»
                    c["dev_keys_sha256"] = sha("\n".join(r["sample_id"] for r in rows))
                files[f"{n}_card.json"] = jbytes(c)
                arm, fold, seed = DG.name_fields(n)
                thr = {**({"arm": arm} if g == "ablation_run" else {}), "fold": fold, "train_seed": seed, "set": "dev",
                       "value": THR_VALUE[n], "dev_scores_sha256": sha("\n".join(score_repr(r["score_attack"]) for r in rows)),
                       "checkpoint_sha256": c["weights_sha256"], "frozen_at": "2026-11-06T08:00:04+0100", **self.thr_extra.get(n, {})}
                tb = jbytes(thr)
                files[f"{n}_threshold.json"] = tb
                for r in rows:                                   # campi derivati dalla soglia, salvo valori scelti dal test
                    if r["threshold_sha256"] == "AUTO":
                        r["threshold_sha256"] = sha(tb)
                    if r["threshold_dev"] == "AUTO":
                        r["threshold_dev"] = repr(float(THR_VALUE[n]))
                    if r["prediction_dev"] == "AUTO":
                        r["prediction_dev"] = "1" if as_float(r["score_attack"]) > THR_VALUE[n] else "0"
                buf = io.StringIO(newline="")
                w = csv.DictWriter(buf, fieldnames=list(rows[0]) if rows else list(DG.DEV_COLUMNS[g]), lineterminator="\n")
                w.writeheader()
                w.writerows(rows)
                files[f"{n}_dev_scores.csv"] = self.dev_raw.get(n, lambda b: b)(buf.getvalue().encode())
            for fn, b in files.items():
                if g == "ablation_run" and fn in self.symlink:  # stesso contenuto, ma come collegamento simbolico
                    (self.tmp / fn).write_bytes(b)
                    (d / "restricted" / fn).symlink_to(self.tmp / fn)
                else:
                    (d / "restricted" / fn).write_bytes(b)
            listed = {fn: sha(b) for fn, b in files.items()}
            listed.update({f"{n}_LivenessNet.keras": self.cards[n]["weights_sha256"] for n in names})
            listed.update(self.listed_extra if g == "ablation_run" else {})
            sums = "".join(f"{h}  {fn}\n" for fn, h in sorted(listed.items())).encode() + (self.sums_extra if g == "ablation_run" else b"")
            (d / "restricted" / "SHA256SUMS").write_bytes(sums)
            pub = b"0" * 64 + b"  aggregate.json\n"
            (d / "public" / "SHA256SUMS").write_bytes(pub)
            log = (self.run_log_line + "\n").encode()
            (d / "run.log").write_bytes(log)
            st = {"run_id": cfg["inputs"][g]["run_id"], "status": "PASS", "passed": True, "wall_seconds": 1000.2,
                  "public_sums_sha256": sha(pub), "restricted_sums_sha256": sha(sums), "run_log_sha256": sha(log),
                  **(self.abl_status_extra if g == "ablation_run" else {})}
            (d / "status.json").write_bytes(jbytes(st))
            spec = cfg["inputs"][g]
            spec.update({"status_sha256": sha(jbytes(st)), "public_sums_sha256": sha(pub), "restricted_sums_sha256": sha(sums), "run_log_sha256": sha(log)})
        d = self.base / "runs/failed"
        (d / "restricted").mkdir(parents=True)
        files = {f"{n}_card.json": jbytes(c) for n, c in self.failed_cards.items()}
        for fn, b in files.items():
            (d / "restricted" / fn).write_bytes(b)
        listed = {fn: sha(b) for fn, b in files.items()}
        listed.update({f"{n}_LivenessNet.keras": c["weights_sha256"] for n, c in self.failed_cards.items()})
        sums = "".join(f"{h}  {fn}\n" for fn, h in sorted(listed.items())).encode()
        (d / "restricted" / "SHA256SUMS").write_bytes(sums)
        st = {**self.failed_status, "restricted_sums_sha256": sha(sums), "kept_complete_models": sorted(self.failed_cards),
              **self.failed_status_extra}
        (d / "status.json").write_bytes(jbytes(st))
        cfg["inputs"]["failed_attempt"].update({"status_sha256": sha(jbytes(st)), "restricted_sums_sha256": sha(sums),
                                                "cards": {fn: sha(b) for fn, b in files.items()}})
        self.cfg_path.parent.mkdir(parents=True, exist_ok=True)
        self.cfg_path.write_bytes(jbytes(cfg))

    def argv(self, cmd: str, *extra, out=None) -> list:
        a = [cmd, "--config", str(self.cfg_path), "--input-root", str(self.base)]
        if cmd == "run":
            a += ["--out-root", str(out or self.out), "--date", DATE]
        return a + list(extra)

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


def fake_png(kind, data) -> bytes:
    return b"\x89PNG-" + kind.encode()


@contextlib.contextmanager
def patched(**attrs):
    """Stato git finto e figure finte (salvo dove un test le sostituisce) per la durata."""
    new = {"REPO_STATE": lambda r: dict(STATE), "PLOTTER": fake_png, **attrs}
    old = {k: getattr(DG, k) for k in new}
    for k, v in new.items():
        setattr(DG, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(DG, k, v)


def run_main(argv) -> tuple:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = DG.main(argv)
    return rc, out.getvalue(), err.getvalue()


def snapshot(*dirs) -> dict:
    out = {}
    for d in dirs:
        d = Path(d)
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    out[str(p)] = (p.stat().st_size, p.stat().st_mtime_ns)
    return out


class WorldCase(unittest.TestCase):
    def setUp(self):
        self.w = World()
        self.addCleanup(self.w.close)

    def check(self, **attrs) -> tuple:
        with patched(**attrs):
            return run_main(self.w.argv("check-inputs"))

    def run_cmd(self, *extra, out=None, **attrs) -> tuple:
        with patched(**attrs):
            return run_main(self.w.argv("run", "--i-am-authorized", *extra, out=out))

    def assertFails(self, needle: str):
        rc, out, err = self.check()
        self.assertEqual(rc, 1, out + err)
        self.assertIn(needle, err)
        self.assertIn("RESULT: FAIL (check-inputs)", out)

    def assertRunFailsClean(self, needle: str, out=None):
        rc, so, se = self.run_cmd(out=out)
        self.assertEqual(rc, 1, so + se)
        self.assertIn(needle, se)
        run = (out or self.w.out) / RUN_ID
        st = json.loads((run / "status.json").read_text())
        self.assertEqual((st["status"], st["passed"]), ("FAIL", False))
        self.assertEqual(sorted(os.listdir(run / "public")) if (run / "public").exists() else [], [])
        return st

    def doc(self, out=None) -> dict:
        return json.loads(((out or self.w.out) / RUN_ID / "public" / "aggregate.json").read_text())


# ============================================================================ regole (casi a mano)

class TestSimulation(unittest.TestCase):
    def h(self, vl, va=None):
        return {"val_loss": vl, "val_accuracy": va or [0.5] * len(vl)}

    def test_best_at_epoch_1_patience_4_stops_at_5(self):
        s = DG.simulate_es(self.h([1, 2, 3, 4, 5, 6, 7]), 12, patience=4)
        self.assertEqual((s["stop"], s["restored"], s["early"]), (5, 1, True))

    def test_improvement_at_epoch_2_resets_the_counter(self):
        # 3 < 5 all'epoca 2: wait torna a 0; epoche 3-6 peggiori -> arresto 6, ripristino 2
        s = DG.simulate_es(self.h([5, 3, 4, 5, 6, 7, 8]), 12, patience=4)
        self.assertEqual((s["stop"], s["restored"], s["early"]), (6, 2, True))

    def test_end_of_epochs_without_stop(self):
        s = DG.simulate_es(self.h([12 - i for i in range(12)]), 12, patience=4)
        self.assertEqual((s["stop"], s["restored"], s["early"]), (12, 12, False))
        self.assertEqual(DG.RESTORED_BY[s["early"]], "end of epochs (set_weights)")

    def test_stop_exactly_at_the_last_epoch_is_early(self):
        vl = [10, 9, 8, 7, 6, 5, 4, 3, 3.5, 3.6, 3.7, 3.8]      # migliore 8, wait 4 all'epoca 12
        s = DG.simulate_es(self.h(vl), 12, patience=4)
        self.assertEqual((s["stop"], s["restored"], s["early"]), (12, 8, True))
        s5 = DG.simulate_es(self.h(vl), 12, patience=5)        # con pazienza 5 nessun arresto: fine delle epoche
        self.assertEqual((s5["stop"], s5["restored"], s5["early"]), (12, 8, False))

    def test_equal_value_is_not_an_improvement(self):
        s = DG.simulate_es(self.h([5, 5, 5, 5, 5]), 12, patience=4)
        self.assertEqual((s["stop"], s["restored"]), (5, 1))

    def test_not_determinable_when_the_history_ends_first(self):
        self.assertIsNone(DG.simulate_es(self.h([5, 4, 3, 2, 1]), 12, patience=4))
        self.assertIsNone(DG.simulate_es(self.h([1, 2, 3, 4, 5]), 12, patience=6))

    def test_patience_2(self):
        s = DG.simulate_es(self.h([1, 2, 3, 4]), 12, patience=2)
        self.assertEqual((s["stop"], s["restored"]), (3, 1))

    def test_start_from_epoch(self):
        # epoche 1-3 ignorate: migliore 4 (9), epoche 5-8 uguali (non migliorano) -> arresto 8, ripristino 4
        s = DG.simulate_es(self.h([1, 9, 9, 9, 9, 9, 9, 9]), 12, patience=4, start=3)
        self.assertEqual((s["stop"], s["restored"]), (8, 4))

    def test_monitor_val_accuracy_maximises(self):
        h = {"val_loss": [9, 8, 7, 6, 5, 4], "val_accuracy": [0.5, 0.9, 0.8, 0.8, 0.8, 0.8]}
        s = DG.simulate_es(h, 12, monitor="val_accuracy", patience=4)
        self.assertEqual((s["stop"], s["restored"], s["val_acc"], s["val_loss"]), (6, 2, 0.9, 8))


class TestCriteria(unittest.TestCase):
    crit = make_config("runs/ablation")["criteria"]

    def test_degenerate(self):
        c = self.crit
        self.assertTrue(DG.is_degenerate(1, 0.5, c))
        self.assertTrue(DG.is_degenerate(2, 0.5000005, c))
        self.assertFalse(DG.is_degenerate(3, 0.5, c))
        self.assertFalse(DG.is_degenerate(2, 0.51, c))
        self.assertFalse(DG.is_degenerate(1, 0.502, c))
        self.assertFalse(DG.is_degenerate(0, 0.5, c))

    def test_saturation(self):
        self.assertTrue(DG.is_half(0.5, self.crit))
        self.assertTrue(DG.is_half(0.5000009, self.crit))
        self.assertFalse(DG.is_half(0.4975, self.crit))
        self.assertFalse(DG.is_half(0.4995, self.crit))         # oltre la tolleranza 1e-6 della configurazione
        self.assertFalse(DG.is_half(0.500002, self.crit))

    def test_recovery_needs_both_conditions(self):
        h = {"val_accuracy": [0.9, 0.75, 0.74, 0.75], "val_loss": [1.5, 1.0, 0.1, 0.99]}
        self.assertEqual(DG.first_recovery(h, 0.75, 1.0), 4)     # 0,9 con 1,5; 0,75 con 1,0 (non < 1); 0,74 sotto
        self.assertEqual(DG.first_recovery(h, 0.75, 1.6), 1)
        self.assertIsNone(DG.first_recovery(h, 0.95, 1.6))

    def test_auroc_and_dev_diag(self):
        self.assertEqual(DG.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(DG.auroc([0, 0, 1, 1], [1.0, 1.0, 1.0, 1.0]), 0.5)
        self.assertEqual(DG.auroc([0, 1, 0, 1], [0.1, 0.1, 0.9, 0.2]), 0.375)    # coppie: (0,1 vs 0,1) 0,5, (0,2 vs 0,1) 1 -> 1,5/4
        d = DG.dev_diag([0, 0, 1, 1], [0.5, 0.6, 0.5, 0.7], 2)
        self.assertEqual((d["acc_05"], d["pred_attack_05"], d["distinct"]), (0.5, 0.5, 3))   # 0,5 = bona fide (pareggio)


# ============================================================================ configurazione

class TestConfig(WorldCase):
    def test_check_inputs_passes_and_writes_nothing(self):
        before = snapshot(self.w.tmp)
        rc, out, err = self.check()
        self.assertEqual(rc, 0, out + err)
        self.assertIn("early stopping replay reproduced on 8/8 cards", out)
        self.assertEqual(snapshot(self.w.tmp), before)

    def test_schema_and_unknown_key(self):
        self.w.cfg["schema"] = "other/1"
        self.w.write()
        self.assertFails("schema must be")
        self.w.cfg["schema"] = DG.CONFIG_SCHEMA
        self.w.cfg["extra"] = 1
        self.w.write()
        self.assertFails("fields must be exactly")

    def test_models_count_must_match(self):
        self.w.cfg["inputs"]["ablation_run"]["models"] = 7
        self.w.write()
        self.assertFails("models must be 6")

    def test_first_rule_must_be_the_run(self):
        self.w.cfg["alternative_rules"][0]["patience"] = 5
        self.w.write()
        self.assertFails("alternative_rules[0] must be the rule of the run")

    def test_saturation_and_degenerate_share_one_definition(self):
        self.w.cfg["criteria"]["saturation"]["tolerance"] = 1e-4
        self.w.write()
        self.assertFails("one definition")

    def test_sensitivity_must_include_the_declared_value(self):
        self.w.cfg["criteria"]["recovery"]["sensitivity"]["val_loss_max"] = [0.7, 1.5]
        self.w.write()
        self.assertFails("including the declared value")

    def test_unknown_time_selection(self):
        self.w.cfg["time_scenarios"][0]["selection"] = "everything"
        self.w.write()
        self.assertFails("selection must be one of")

    def test_declaration_must_be_true(self):
        self.w.cfg["declarations"]["no_test_scores_read"] = False
        self.w.write()
        self.assertFails("declarations.no_test_scores_read must be true")

    def test_example_must_exist(self):
        self.w.cfg["examples"][0]["seed"] = 99
        self.w.write()
        self.assertFails("is not a model of the inputs")

    def test_recipe_of_the_cards(self):
        self.w.cfg["recipe"]["early_stopping_patience"] = 3
        self.w.cfg["alternative_rules"][0]["patience"] = 3
        self.w.write()
        self.assertFails("early_stopping_patience = 4 differs from the recipe")


# ============================================================================ ingressi

class TestInputs(WorldCase):
    def test_sha_mismatch_of_a_card(self):
        self.w.write()
        p = self.w.base / "runs/ablation/restricted/arm_b_fold1_seed42_card.json"
        p.write_bytes(p.read_bytes() + b" ")
        self.assertFails("differs from the restricted SHA256SUMS")

    def test_status_anchor(self):
        p = self.w.base / "runs/ablation/status.json"
        p.write_bytes(p.read_bytes() + b" ")
        self.assertFails("status.json: SHA-256")

    def test_restricted_sums_anchor(self):
        p = self.w.base / "runs/folds/restricted/SHA256SUMS"
        p.write_bytes(p.read_bytes() + b"\n")
        self.assertFails("restricted SHA256SUMS: SHA-256")

    def test_run_log_anchor(self):
        p = self.w.base / "runs/ablation/run.log"
        p.write_bytes(p.read_bytes() + b"x\n")
        self.assertFails("run.log: SHA-256")

    def test_status_must_be_pass(self):
        self.w.abl_status_extra = {"status": "FAIL", "passed": False}
        self.w.write()
        self.assertFails("not a PASS of run")

    def test_failed_attempt_status_must_be_fail(self):
        self.w.failed_status = {"run_id": FAILED_ID, "status": "PASS", "passed": True}
        self.w.write()
        self.assertFails("not a FAIL of run")

    def test_failed_attempt_card_anchor(self):
        p = self.w.base / "runs/failed/restricted/arm_b_fold1_seed42_card.json"
        self.w.cfg["inputs"]["failed_attempt"]["cards"]["arm_b_fold1_seed42_card.json"] = "0" * 64
        self.w.cfg_path.write_bytes(jbytes(self.w.cfg))
        self.assertTrue(p.is_file())
        self.assertFails("differs from the restricted SHA256SUMS")

    def test_extra_card_listed(self):
        self.w.listed_extra = {"arm_b_fold2_seed42_card.json": "1" * 64}
        self.w.write()
        self.assertFails("training cards listed")

    def test_history_of_wrong_length(self):
        self.w.cards["arm_b_fold1_seed42"]["history"]["lr"].append(0.0005)
        self.w.write()
        self.assertFails("history.lr has 6 values, epochs_run is 5")

    def test_history_non_finite(self):
        self.w.cards["arm_b_fold1_seed42"]["history"]["val_loss"][3] = float("nan")
        self.w.write()
        self.assertFails("not a finite number")

    def test_card_keys_closed(self):
        self.w.cards["fold1_seed42"]["arm"] = "folds"
        self.w.write()
        self.assertFails("keys differ from the expected set")

    def test_restored_epoch_not_argmin(self):
        c = self.w.cards["arm_b_prime_fold1_seed42"]
        c["restored_epoch"], c["best_val_loss"], c["val_acc_at_best"] = 1, 5.0, 0.5
        self.w.write()
        self.assertFails("is not the epoch of minimum val_loss")

    def test_simulation_does_not_reproduce_the_stop(self):
        # val_loss 5,6,7,8,9,10 con migliore 1: la regola si ferma all'epoca 5, la scheda dichiara 6 epoche
        c = self.w.cards["arm_b_fold1_seed42"]
        c["history"] = hist([5, 6, 7, 8, 9, 10], [0.5] * 6)
        c["epochs_run"] = 6
        self.w.write()
        self.assertFails("the replay of the Keras EarlyStopping gives stop 5, restored 1")

    def test_simulation_does_not_reproduce_the_restored_epoch(self):
        # minimo uguale all'epoca 1 e 3 (5 e 5): argmin = 1, la scheda dichiara 3 con lo stesso valore
        c = self.w.cards["arm_b_fold1_seed42"]
        c["history"] = hist([5, 6, 5, 8, 9], [0.5] * 5)
        c["restored_epoch"] = 3
        self.w.write()
        self.assertFails("is not the epoch of minimum val_loss (1)")

    def test_restored_by_must_follow_the_replay(self):
        self.w.cards["arm_b_prime_fold1_seed43"]["restored_by"] = "end of epochs (set_weights)"
        self.w.write()
        self.assertFails("restored_by 'end of epochs (set_weights)' differs from the replay")

    def test_best_val_loss_must_be_the_history(self):
        self.w.cards["arm_a_star_fold1_seed42"]["val_acc_at_best"] = 0.875
        self.w.write()
        self.assertFails("best_val_loss or val_acc_at_best differ from the history")

    def test_weights_listed(self):
        self.w.cards["arm_b_fold1_seed43"]["weights_file"] = "other.keras"
        self.w.write()
        self.assertFails("weights_file or weights_sha256 differ")

    def test_dev_rows_and_balance(self):
        self.w.dev["arm_b_fold1_seed42"] = self.w.dev["arm_b_fold1_seed42"][:-1]
        self.w.write()
        self.assertFails("7 rows; the configuration expects 8")

    def test_dev_unbalanced(self):
        rows = self.w.dev["arm_a_star_fold1_seed42"]
        rows[0].update(label="1", score_attack="0.8", prediction_05="1")     # 3 bona fide, 5 attacchi
        self.w.write()
        self.assertFails("the dev is not balanced")

    def test_dev_accuracy_against_the_card(self):
        rows = self.w.dev["arm_a_star_fold1_seed42"]
        rows[0].update(score_attack="0.9", prediction_05="1")      # un bona fide giusto diventa sbagliato: 7/8
        self.w.write()
        self.assertFails("differs from val_acc_at_best")

    def test_dev_tie_at_0_5_is_bona_fide(self):
        rows = self.w.dev["arm_a_star_fold1_seed42"]
        rows[0]["score_attack"] = "0.5"                            # pareggio: resta bona fide, accuratezza invariata
        self.w.write()
        rc, out, err = self.check()
        self.assertEqual(rc, 0, out + err)

    def test_dev_checkpoint_and_role(self):
        self.w.dev["arm_b_fold1_seed42"][0]["checkpoint_sha256"] = "3" * 64
        self.w.write()
        self.assertFails("checkpoint_sha256 differs from the weights of the card")
        self.w.dev["arm_b_fold1_seed42"][0]["checkpoint_sha256"] = self.w.cards["arm_b_fold1_seed42"]["weights_sha256"]
        self.w.dev["arm_b_fold1_seed42"][0]["role"] = "test"
        self.w.write()
        self.assertFails("a row carries another role")
        self.w.dev["arm_b_fold1_seed42"][0]["role"] = "dev"
        self.w.dev["arm_b_fold1_seed42"][1]["arm"] = "arm_a_star"
        self.w.write()
        self.assertFails("a row carries another arm")
        self.w.dev["arm_b_fold1_seed42"][1]["arm"] = "arm_b"
        self.w.dev["arm_b_fold1_seed42"][2]["train_seed"] = "43"
        self.w.write()
        self.assertFails("a row carries another fold or seed")

    def test_dev_score_range(self):
        self.w.dev["arm_b_fold1_seed42"][0]["score_attack"] = "1.5"
        self.w.write()
        self.assertFails("score_attack outside [0, 1]")

    def test_dev_shared_within_a_fold(self):
        rows = self.w.dev["arm_b_fold1_seed43"]
        rows[0]["subject_id"] = "1103"
        self.w.write()
        self.assertFails("do not share the same dev")

    def test_threshold_consistency(self):
        n = "arm_b_fold1_seed42"
        for extra, needle in (({"frozen_at": "2026-11-06T07:59:00+0100"}, "are not in order"),       # prima dei pesi
                              ({"arm": "arm_a_star"}, f"{n}_threshold.json: arm differs from the card"),
                              ({"fold": 2}, "fold or train_seed differ from the card"),
                              ({"checkpoint_sha256": "4" * 64}, f"{n}_threshold.json: checkpoint_sha256 differs"),
                              ({"value": "x"}, "value is not a finite number"),
                              ({"dev_scores_sha256": "0" * 64}, f"the scores differ from dev_scores_sha256 of {n}_threshold.json")):
            self.w.thr_extra = {n: extra}
            self.w.write()
            self.assertFails(needle)
        self.w.thr_extra = {"fold1_seed42": {"train_seed": 43}}                     # anche le soglie del run dei fold
        self.w.write()
        self.assertFails("fold or train_seed differ from the card")

    def test_wall_line(self):
        self.w.run_log_line = "run folder: x, wall 1005 s"
        self.w.write()
        self.assertFails("differs from wall_seconds of status.json")
        self.w.run_log_line = "nothing"
        self.w.write()
        self.assertFails("0 lines 'run folder")
        self.w.run_log_line = "run folder: x, wall 1000 s\nrun folder: y, wall 1000 s"
        self.w.write()
        self.assertFails("2 lines 'run folder")

    def test_failed_attempt_kept_models(self):
        self.w.failed_status_extra = {"kept_complete_models": ["arm_b_fold1_seed42", "arm_b_fold1_seed43"]}
        self.w.write()
        self.assertFails("kept_complete_models differ from the cards of the configuration")

    def test_dev_rows_compared_one_by_one(self):
        a, b = "arm_b_fold1_seed42", "arm_b_fold1_seed43"
        # scambio riga per riga di score, decisioni e soglia fra due modelli dello stesso fold, ancore riallineate
        for ra, rb in zip(self.w.dev[a], self.w.dev[b]):
            for k in ("score_attack", "prediction_05", "prediction_dev", "threshold_dev"):
                ra[k], rb[k] = rb[k], ra[k]
        # nei file veri threshold_dev e prediction_dev sono scritti: lo scambio porta con sé quelli dell'altro modello
        self.w.dev[a] = [dict(r, threshold_dev=repr(THR_VALUE[b]), prediction_dev="1" if float(r["score_attack"]) > THR_VALUE[b] else "0")
                         for r in self.w.dev[a]]
        self.w.write()
        self.assertFails(f"{a}_dev_scores.csv: threshold_dev or threshold_sha256 differ from the frozen threshold")

    def test_dev_decisions_follow_the_rule(self):
        n = "arm_a_star_fold1_seed42"
        r = self.w.dev[n][0]                                       # bona fide a 0,1: prediction_05 0, prediction_dev 0
        r["prediction_05"] = "1"
        self.w.write()
        self.assertFails("prediction_05 does not follow")
        r["prediction_05"] = "0"
        r["prediction_dev"] = "1"
        self.w.write()
        self.assertFails("prediction_dev does not follow")
        r["prediction_dev"] = "AUTO"
        r["threshold_sha256"] = "5" * 64
        self.w.write()
        self.assertFails("threshold_dev or threshold_sha256 differ")

    def test_dev_tie_at_the_dev_threshold_is_bona_fide(self):
        n = "arm_b_prime_fold1_seed43"                             # soglia 0,5: un attacco a 0,5 esatto è bona fide (sbagliato)
        r = next(x for x in self.w.dev[n] if x["label"] == "1")
        r.update(score_attack="0.5", prediction_05="0", prediction_dev="0")
        self.w.write()
        rc, out, err = self.check()
        self.assertEqual(rc, 1, out + err)                         # le decisioni sono giuste; l'accuratezza scende a 7/8
        self.assertIn("differs from val_acc_at_best", err)
        self.assertNotIn("does not follow", err)

    def test_duplicated_dev_row(self):
        n = "arm_a_star_fold1_seed42"
        self.w.dev[n][1] = dict(self.w.dev[n][0])
        self.w.write()
        self.assertFails("a sample_id or a file_sha256 appears in more than one row")
        self.w.dev[n] = self.w.dev_rows(n)
        self.w.dev[n][1]["file_sha256"] = self.w.dev[n][0]["file_sha256"]
        self.w.write()
        self.assertFails("a sample_id or a file_sha256 appears in more than one row")
        self.w.dev[n] = self.w.dev_rows(n)
        self.w.dev[n][1]["sample_id"] = self.w.dev[n][0]["sample_id"]   # solo la chiave ripetuta, file diversi
        self.w.write()
        self.assertFails("a sample_id or a file_sha256 appears in more than one row")

    def test_dev_keys_against_the_card(self):
        self.w.cards["arm_b_fold1_seed43"]["dev_keys_sha256"] = "1" * 64
        self.w.write()
        self.assertFails("the sample_id list differs from dev_keys_sha256 of the card")

    def test_other_subjects_with_the_same_counts(self):
        for r in self.w.dev["arm_b_fold1_seed43"]:
            if r["subject_id"] == SUBJ_BF:
                r["subject_id"] = "1103"                            # stessi conteggi per classe e di soggetti
        self.w.write()
        self.assertFails("fold 1: the models of the fold do not share the same dev")

    def test_other_keys_with_the_same_counts(self):
        for r in self.w.dev["arm_b_fold1_seed43"]:
            r["sample_id"] = r["sample_id"].replace(".jpg", "b.jpg")  # dev_keys_sha256 riallineato: diverso dagli altri modelli
        self.w.write()
        self.assertFails("fold 1: the models of the fold do not share the same dev")

    def test_card_fields_against_the_name(self):
        for field, value, needle in (("train_seed", 99, "train_seed differs from the file name"), ("fold", 2, "fold differs from the file name"),
                                     ("arm", "arm_b_prime", "arm differs from the file name"), ("arch", "Other", "arch is not LivenessNet"),
                                     ("n_fitting", 0, "n_fitting must be a positive integer"), ("n_dev", 1, "n_dev must be an integer >= 2")):
            self.w.cards = {n: self.w.card(n, "folds_run" if n.startswith("fold") else "ablation_run") for n in H}
            self.w.cards["arm_b_fold1_seed42"][field] = value
            self.w.write()
            self.assertFails(needle)

    def test_card_n_dev_against_the_configuration(self):
        self.w.cards["arm_b_fold1_seed42"]["n_dev"] = 10
        self.w.write()
        self.assertFails("n_dev of the card is 10; the configuration expects 8")

    def test_history_ranges(self):
        for key, i, value, needle in (("accuracy", 1, 1.2, "accuracy outside [0, 1]"), ("val_accuracy", 1, -0.1, "accuracy outside [0, 1]"),
                                      ("loss", 0, -0.5, "negative loss"), ("lr", 0, 0.0, "non-positive lr")):
            self.w.cards = {n: self.w.card(n, "folds_run" if n.startswith("fold") else "ablation_run") for n in H}
            self.w.cards["arm_b_fold1_seed42"]["history"][key][i] = value
            self.w.write()
            self.assertFails(needle)

    def test_dev_label_out_of_range(self):
        self.w.dev["arm_a_star_fold1_seed42"][0]["label"] = "2"
        self.w.write()
        self.assertFails("label out of range")

    def test_symbolic_link_input(self):
        self.w.symlink = {"arm_b_fold1_seed42_dev_scores.csv"}
        self.w.write()
        self.assertFails("arm_b_fold1_seed42_dev_scores.csv is a symbolic link")

    def test_irregular_csv_row(self):
        def extra_field(b):
            lines = b.decode().split("\n")
            lines[2] += ",x"
            return "\n".join(lines).encode()
        self.w.dev_raw = {"arm_b_fold1_seed42": extra_field}
        self.w.write()
        self.assertFails("rows with a wrong number of fields")

    def test_sums_listed_twice(self):
        self.w.sums_extra = b"0" * 64 + b"  arm_b_fold1_seed42_card.json\n"
        self.w.write()
        self.assertFails("arm_b_fold1_seed42_card.json listed twice")

    def test_failed_card_must_be_a_model_of_the_run(self):
        c = copy.deepcopy(self.w.cards["fold1_seed42"])
        self.w.failed_cards = {"fold1_seed42": c}
        c["arm"] = "folds"
        self.w.write()
        self.assertFails("not a model of the ablation run")


# ============================================================================ run e uscite

class TestRun(WorldCase):
    def test_pass_outputs_and_known_answers(self):
        before = snapshot(self.w.base)
        rc, out, err = self.run_cmd()
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(snapshot(self.w.base), before)
        run = self.w.out / RUN_ID
        st = json.loads((run / "status.json").read_text())
        self.assertEqual((st["status"], st["passed"], st["run_id"]), ("PASS", True, RUN_ID))
        self.assertIn("none", st["restricted"])
        self.assertEqual(sorted(os.listdir(run)), ["public", "run.log", "status.json"])
        self.assertEqual(sorted(os.listdir(run / "public")), sorted(DG.PUBLIC_OUTPUTS))
        sums = (run / "public" / "SHA256SUMS").read_bytes()
        self.assertEqual(st["public_sums_sha256"], sha(sums))
        self.assertEqual(DG.read_sums(sums, "x"), {n: sha((run / "public" / n).read_bytes()) for n in DG.PUBLIC_OUTPUTS if n != "SHA256SUMS"})
        self.assertEqual(st["run_log_sha256"], sha((run / "run.log").read_bytes()))
        T = self.doc()["tables"]
        # T1: i due degeneri, i casi al confine (ripristino <= 3 non degeneri)
        self.assertEqual([(r["arm"], r["seed"]) for r in T["T1"]["rows"]], [("arm_a_star", 43), ("arm_b", 42)])
        self.assertEqual(T["T1"]["per_arm"], {"arm_b": 1, "arm_a_star": 1, "arm_b_prime": 0})
        self.assertEqual(T["T1"]["per_seed"], {"42": 1, "43": 1})
        self.assertEqual(sorted((r["arm"], r["seed"], r["restored_epoch"]) for r in T["T1"]["boundary"]),
                         [("arm_b", 43, 2), ("arm_b_prime", 42, 3)])
        self.assertEqual(T["T2"]["rows"][0], {"arm": "arm_b", "seed": 42, "fold1": "D1"})
        # T3: saturazione all'epoca 1 (val_acc 0,5): degeneri 2/2, non degeneri 3/4 (la ricaduta parte da 0,75), fold 1/1
        t3 = {r["group"]: r for r in T["T3"]["rows"]}
        self.assertEqual((t3["degenerate"]["epoch1_saturated"], t3["non-degenerate"]["epoch1_saturated"], t3["folds run"]["epoch1_saturated"]), (2, 3, 1))
        # epoche a 0,5: degeneri 5 + 6 = 11; non degeneri 3 (a_star s42) + 5 (b_prime s42) + 4 (b s43) + 1 (b_prime s43) = 13
        self.assertEqual((t3["degenerate"]["saturated_epochs"], t3["non-degenerate"]["saturated_epochs"], t3["folds run"]["saturated_epochs"]), (11, 13, 2))
        self.assertEqual(t3["degenerate"]["val_loss_min"], 3.0)
        self.assertEqual([(r["arm"], r["epoch"]) for r in T["T3"]["below_chance_epochs"]], [("arm_b_prime", 3), ("arm_b_prime", 4)])
        # T4: recupero (0,75 e < 1): a_star s42 epoca 4; b s43 epoca 1; b_prime s43 epoca 4 (epoca 3: 0,75 con 1,0, non < 1); fold 4
        t4 = {r["group"]: r for r in T["T4"]["rows"]}
        self.assertEqual((t4["degenerate"]["with_recovery"], t4["non-degenerate"]["with_recovery"]), (0, 3))
        self.assertEqual((t4["non-degenerate"]["min"], t4["non-degenerate"]["median"], t4["non-degenerate"]["max"]), (1, 4, 4))
        self.assertEqual(t4["folds run"]["median"], 7)                         # epoca 7: 0,75 con 0,5 (prima val_loss >= 1)
        self.assertEqual(t4["folds run"]["median_steps"], 105)                 # 7 epoche x 15 passi (230 / 16, per eccesso)
        self.assertEqual(t4["arm_b"]["median_steps"], 10)                      # 1 epoca x 10 passi (150 / 16, per eccesso)
        # T5: con loss < 1,5 il b_prime s43 recupera all'epoca 3
        t5 = [r for r in T["T5"]["rows"] if (r["val_accuracy_min"], r["val_loss_max"]) == (0.75, 1.5)][0]
        self.assertEqual((t5["non-degenerate"]["with_recovery"], t5["non-degenerate"]["median"]), (3, 3))
        # T6: primo val_loss sotto quello dell'epoca 1: degenere s43 all'epoca 2 con dev a 0,5
        fi = {r["group"]: r for r in T["T6"]["first_improvement"]}
        self.assertEqual(fi["degenerate"]["first_val_loss_below_epoch1"], [2, None])
        self.assertEqual(fi["degenerate"]["with_val_accuracy_0.5_in_that_epoch"], 1)
        self.assertEqual([(r["last_val_loss_vs_previous"], r["last_val_accuracy_vs_previous"]) for r in T["T6"]["rows"]], [("up", "="), ("up", "=")])
        self.assertEqual([(r["arm"], r["seed"]) for r in T["T6"]["non_degenerate_without_recovery"]], [("arm_b_prime", 42)])
        # T7: regole alternative
        t7 = {r["rule"]: r for r in T["T7"]["rows"]}
        self.assertEqual(t7["patience 4 (run)"]["degenerate"], {"determinable": 2, "models": 2, "degenerate": 2, "restored_median": 1.5})
        # patience 2: b_prime s42 (5,6,4,...) si ferma all'epoca 3 con migliore 1 (wait 2 all'epoca 3? no: 4 < 5 migliora) ->
        # migliore 3, arresto 5; b s43 migliore 2, arresto 4; a_star s42 e fold: 12; b_prime s43 migliore 8, arresto 10
        self.assertEqual(t7["patience 2"]["non-degenerate"]["determinable"], 4)
        self.assertEqual(t7["patience 2"]["non-degenerate"]["degenerate"], 0)
        # patience 8: il degenere b s42 (5 epoche) non è determinabile; b_prime s43 (fermato a 12 con migliore 8): 12 -> fine, migliore 8
        self.assertEqual(t7["patience 8"]["degenerate"]["determinable"], 0)
        self.assertEqual(t7["patience 8"]["non-degenerate"]["determinable"], 2)
        # monitor val_accuracy: a_star s43 (degenere, tutto a 0,5) migliore 1, arresto 5 -> determinabile e degenere
        self.assertEqual(t7["monitor val_accuracy"]["degenerate"]["determinable"], 2)
        self.assertEqual(t7["monitor val_accuracy"]["degenerate"]["degenerate"], 2)
        ch = {c["rule"]: c["non_degenerate_changes"] for c in T["T7"]["changes"]}
        self.assertIn({"run": "sessablrun", "arm": "arm_b_prime", "fold": 1, "seed": 42, "restored_run": 3, "restored_rule": None}, ch["patience 8"])
        # T8: lr ridotto dall'epoca 4 nei modelli con più di 3 epoche
        self.assertEqual(T["T8"]["rows"][0]["first_reduced_epoch"], [4, 4])
        self.assertEqual(T["T8"]["rows"][0]["decided_with_dev_at_0.5"], 2)
        # non degeneri: riduzione all'epoca 4, decisa a fine epoca 3 (indice 2): a dev 0,5 a_star s42 e b s43, non b_prime
        self.assertEqual((T["T8"]["rows"][1]["with_reduction"], T["T8"]["rows"][1]["decided_with_dev_at_0.5"]), (4, 2))
        # T10: degenere s42 a uscita costante (attacco), s43 no
        t10 = {(r["arm"], r["seed"]): r for r in T["T10"]["rows"]}
        r = t10[("arm_b", 42)]
        self.assertEqual((r["acc_05"], r["pred_attack_05"], r["auroc"], r["distinct"], r["mean_bona_fide"]), (0.5, 1.0, 0.5, 1, 1.0))
        r = t10[("arm_b_prime", 42)]                                          # confine: 2/8 giusti
        self.assertEqual(r["acc_05"], 0.25)
        self.assertEqual(T["T10"]["dev_subjects_per_fold"], {"1": 2})
        self.assertEqual(T["T10"]["max_abs_difference_acc_05_val_acc_at_best"], 0.0)
        # T11: copia identica
        self.assertEqual(T["T11"]["rows"][0]["identical_history"], True)
        self.assertEqual(T["T11"]["identical_histories"], 1)
        # T12: tempi (secondi per epoca 10/20/5; train 565 s; muro 1000 s; resto 72,5 s per modello)
        t12 = T["T12"]
        self.assertEqual((t12["run_train_seconds"], t12["run_wall_seconds"], t12["overhead_seconds_per_model"]), (565.0, 1000.0, 72.5))
        sc = {s["selection"]: s for s in t12["scenarios"]}
        self.assertEqual((sc["all"]["training_seconds"], sc["all"]["with_overhead_seconds"]), (840.0, 1275.0))
        self.assertEqual((sc["arms: arm_b, arm_a_star"]["models"], sc["arms: arm_b, arm_a_star"]["training_seconds"]), (4, 720.0))
        self.assertEqual((sc["degenerate"]["training_seconds"], sc["degenerate"]["with_overhead_seconds"]), (360.0, None))
        self.assertEqual((sc["stopped before max_epochs"]["models"], sc["stopped before max_epochs"]["degenerate"]), (4, 2))
        self.assertEqual((t12["saved_to_threshold_seconds"]["median"], t12["threshold_to_test_scores_seconds"]["median"]), (4.0, 16.0))
        # T13: 0,99^(10 x epoca)
        self.assertEqual(T["T13"]["rows"][0]["residual"][0], 0.99 ** 10)
        # CSV: una riga per epoca e per modello (5+12+7+6+6+12 + 12 + 5 del tentativo)
        curves = list(csv.DictReader(io.StringIO((run / "public" / "curves.csv").read_text())))
        self.assertEqual(len(curves), 65)
        self.assertEqual(list(curves[0]), list(DG.EPOCH_COLUMNS))
        self.assertEqual(sum(1 for r in curves if r["is_best"] == "1"), 8)
        models = list(csv.DictReader(io.StringIO((run / "public" / "models.csv").read_text())))
        self.assertEqual([r["degenerate"] for r in models if r["run"] == "sessablrun"].count("1"), 2)
        self.assertEqual({r["first_recovery_epoch"] for r in models if r["arm"] == "arm_b" and r["seed"] == "43" and r["run"] == "sessablrun"}, {"1"})
        # primo val_loss sotto l'epoca 1: un valore uguale non conta (5, 5, 4 -> epoca 3)
        self.assertEqual({r["first_val_loss_below_epoch1"] for r in models if r["arm"] == "arm_b_prime" and r["seed"] == "42"
                          and r["run"] == "sessablrun"}, {"3"})

    def test_determinism_difference_is_a_flag_not_an_error(self):
        c = self.w.failed_cards["arm_b_fold1_seed42"]
        c["history"]["accuracy"][2] += 0.001                    # lato train diverso: storia non identica
        c["train_seconds"] = 51.0
        self.w.write()
        rc, out, err = self.run_cmd()
        self.assertEqual(rc, 0, out + err)
        r = self.doc()["tables"]["T11"]["rows"][0]
        self.assertFalse(r["identical_history"])
        self.assertAlmostEqual(r["max_abs_history_difference"], 0.001, places=12)
        self.assertEqual(self.doc()["tables"]["T11"]["identical_histories"], 0)

    def test_figures_with_fake_plotter_and_declared_skip(self):
        rc, out, err = self.run_cmd()
        self.assertEqual(rc, 0, out + err)
        pub = self.w.out / RUN_ID / "public"
        self.assertEqual((pub / "figure_examples.png").read_bytes(), b"\x89PNG-examples")
        self.assertEqual((pub / "figure_val_loss_all.png").read_bytes(), b"\x89PNG-val_loss_all")

        def missing(kind, data):
            raise ImportError(name="matplotlib")
        out2 = self.w.tmp / "out2"
        rc, so, se = self.run_cmd(out=out2, PLOTTER=missing)
        self.assertEqual(rc, 0, so + se)
        pub = out2 / RUN_ID / "public"
        self.assertFalse(any((pub / n).exists() for n in DG.FIGURES))
        self.assertIn("skipped: missing Python package 'matplotlib'", self.doc(out2)["figures"]["note"])
        self.assertNotIn("png", (pub / "SHA256SUMS").read_text())
        self.assertIn("skipped", json.loads((out2 / RUN_ID / "status.json").read_text())["figures"])

    def test_figure_data_examples_and_grid(self):
        captured = {}

        def plot(kind, data):
            captured[kind] = data
            return b"\x89PNG"
        rc, out, err = self.run_cmd(PLOTTER=plot)
        self.assertEqual(rc, 0, out + err)
        ex = captured["examples"]["examples"]
        self.assertEqual([(e["label"], e["status"], e["restored"], e["stopped"]) for e in ex],
                         [("arm_b fold 1 seed 42", "degenerate", 1, 5), ("folds run fold 1 seed 42", "not degenerate", 12, 12)])
        self.assertEqual([e["recovered"] for e in ex], [None, 7])
        grid = captured["val_accuracy_all"]["grid"]
        self.assertEqual([len(r) for r in grid], [1] * 7)                     # 3 bracci x 2 seed + 1 seed del run dei fold
        self.assertEqual(grid[3][0]["label"], "arm_a_star fold 1 seed 43")

    def test_real_default_plotter_when_matplotlib_is_present(self):
        rc, out, err = self.run_cmd(PLOTTER=None)
        self.assertEqual(rc, 0, out + err)
        pub = self.w.out / RUN_ID / "public"
        try:
            import matplotlib  # noqa: F401
            for n in DG.FIGURES:
                self.assertEqual((pub / n).read_bytes()[:8], b"\x89PNG\r\n\x1a\n", n)
        except ImportError:
            self.assertFalse((pub / DG.FIGURES[0]).exists())

    def test_previous_outputs_removed_at_the_start(self):
        self.assertEqual(self.run_cmd()[0], 0)
        pub = self.w.out / RUN_ID / "public"
        self.assertTrue((pub / "figure_examples.png").exists())

        def missing(kind, data):
            raise ImportError(name="matplotlib")
        rc, so, se = self.run_cmd(PLOTTER=missing)                            # stessa cartella, ora senza figure
        self.assertEqual(rc, 0, so + se)
        self.assertEqual(sorted(os.listdir(pub)), sorted(n for n in DG.PUBLIC_OUTPUTS if n not in DG.FIGURES))

    def test_two_runs_give_identical_public_files(self):
        out2 = self.w.tmp / "out2"
        self.assertEqual(self.run_cmd()[0], 0)
        self.assertEqual(self.run_cmd(out=out2)[0], 0)
        a, b = self.w.out / RUN_ID / "public", out2 / RUN_ID / "public"
        self.assertEqual(sorted(os.listdir(a)), sorted(os.listdir(b)))
        for n in os.listdir(a):
            self.assertEqual((a / n).read_bytes(), (b / n).read_bytes(), n)
        s1, s2 = (json.loads((d / RUN_ID / "status.json").read_text()) for d in (self.w.out, out2))
        self.assertEqual({k: v for k, v in s1.items() if k not in ("finished", "wall_seconds", "run_log_sha256")},
                         {k: v for k, v in s2.items() if k not in ("finished", "wall_seconds", "run_log_sha256")})

    def test_subject_id_in_a_folder_name_blocks_the_public(self):
        self.w.close()
        self.w = World(abl_folder=f"runs/{SUBJ_BF}/ablation")
        self.addCleanup(self.w.close)
        st = self.assertRunFailsClean("public file aggregate.json refused: subject id")
        self.assertIn("public file", st["reason"])
        run = self.w.out / RUN_ID
        for f in ("status.json", "run.log"):                     # il motivo nomina la categoria, mai il valore
            self.assertNotIn(SUBJ_BF, (run / f).read_text(), f)
            self.assertIn("subject id", (run / f).read_text(), f)

    def test_image_key_and_absolute_path_block_the_public(self):
        old = DG.aggregate_md
        try:
            DG.aggregate_md = lambda doc: old(doc) + "\nreal/1101_1101_00_00_01_0.jpg\n"
            self.assertRunFailsClean("image keys")
            DG.aggregate_md = lambda doc: old(doc) + "\n/Us" "ers/someone/x\n"
            self.assertRunFailsClean("absolute path")
        finally:
            DG.aggregate_md = old

    def test_subject_scan_ignores_digits_inside_numbers(self):
        inp = DG.Inputs()
        inp.subjects = {"0030", "1101"}
        self.assertEqual(DG.public_problems("val 0.0030 and 21101 and seed1101 and x_1101", inp), [])
        self.assertEqual(DG.public_problems("subject 1101 here", inp), ["subject id"])
        self.assertEqual(DG.public_problems("a,0030,b", inp), ["subject id"])

    def test_scan_of_addresses_and_markers(self):
        inp = DG.Inputs()
        self.assertEqual(DG.public_problems("host 10.1.2.3 here", inp), ["IP address"])
        self.assertEqual(DG.public_problems("local 127.0.0.1 and 0.0.0.0", inp), [])
        marker = "TO" + "DO"
        self.assertEqual(DG.public_problems(f"a {marker} b", inp, markdown=True), ["unfinished-work marker"])
        self.assertEqual(DG.public_problems(f"a {marker} b", inp, markdown=False), [])
        self.assertEqual(DG.public_problems("/Us" "ers/x/y", inp), ["absolute path"])

    def test_write_public_leaves_nothing_after_an_error(self):
        d = self.w.tmp / "pub"
        d.mkdir()
        old = DG.write_binary
        try:
            def broken(dd, files):
                raise OSError("disk full")
            DG.write_binary = broken
            with self.assertRaises(OSError):
                DG.write_public(d, {"aggregate.json": "{}\n", "curves.csv": "a\n", "figure_examples.png": b"\x89PNG", "SHA256SUMS": "x\n"})
        finally:
            DG.write_binary = old
        self.assertEqual(os.listdir(d), [])

    def test_output_root_symbolic_link(self):
        real = self.w.tmp / "real_out"
        real.mkdir()
        link = self.w.tmp / "link_out"
        link.symlink_to(real)
        rc, so, se = self.run_cmd(out=link)
        self.assertEqual(rc, 1, so + se)
        self.assertIn("the output root is a symbolic link", se)
        self.assertEqual(os.listdir(real), [])

    def test_rerun_after_pass_with_a_foreign_entry_touches_nothing(self):
        # comportamento comune ai runner (m2_writers.output_problem e run_dir_problem): una cartella con voci estranee non
        # è nostra; nulla è tolto né scritto, l'errore è stampato e il PASS precedente resta com'è
        self.assertEqual(self.run_cmd()[0], 0)
        run = self.w.out / RUN_ID
        before = snapshot(run)
        (run / "public" / "notes.txt").write_text("x")
        rc, so, se = self.run_cmd()
        self.assertEqual(rc, 1)
        self.assertIn("public/notes.txt in the run folder is not an output of this command: refused, nothing removed", se)
        after = snapshot(run)
        after.pop(str(run / "public" / "notes.txt"))
        self.assertEqual(after, before)
        self.assertEqual(json.loads((run / "status.json").read_text())["status"], "PASS")

    def test_figure_labels_are_scanned(self):
        old = DG.figure_data
        try:
            def leaky(cfg, inp):
                d = old(cfg, inp)
                d["examples"][0]["label"] = f"subject {SUBJ_AT}"
                return d
            DG.figure_data = leaky
            st = self.assertRunFailsClean("public file figure labels refused: subject id")
            self.assertNotIn(SUBJ_AT, st["reason"])
        finally:
            DG.figure_data = old

    def test_pass_then_failure_leaves_no_pass(self):
        self.assertEqual(self.run_cmd()[0], 0)
        run = self.w.out / RUN_ID
        self.assertTrue((run / "public" / "aggregate.json").exists())
        self.w.cards["arm_b_fold1_seed42"]["history"]["lr"].append(0.0005)
        self.w.write()
        st = self.assertRunFailsClean("history.lr has 6 values")
        self.assertEqual(st["outputs"], "removed")
        self.assertIn("RESULT: FAIL (run)", (run / "run.log").read_text())

    def test_failure_after_the_checks_leaves_no_public(self):
        def boom(kind, data):
            raise RuntimeError("plot failed")
        self.assertEqual(self.run_cmd()[0], 0)
        rc, so, se = self.run_cmd(PLOTTER=boom)
        self.assertEqual(rc, 1, so + se)
        self.assertIn("unexpected error RuntimeError: plot failed", se)
        st = json.loads((self.w.out / RUN_ID / "status.json").read_text())
        self.assertEqual((st["status"], st["outputs"]), ("FAIL", "removed"))
        self.assertEqual(os.listdir(self.w.out / RUN_ID / "public"), [])

    def test_failure_after_the_public_write_removes_it(self):
        calls = []
        real = DG.sha256_file

        def once(p):                                    # il primo hash di run.log: dopo la scrittura del pubblico
            if not calls and Path(p).name == "run.log":
                calls.append(p)
                raise OSError("disk error after the public write")
            return real(p)
        rc, so, se = self.run_cmd(sha256_file=once)
        self.assertEqual(rc, 1, so + se)
        self.assertEqual(len(calls), 1)
        st = json.loads((self.w.out / RUN_ID / "status.json").read_text())
        self.assertEqual((st["status"], st["outputs"]), ("FAIL", "removed"))
        self.assertIn("disk error after the public write", st["reason"])
        self.assertEqual(os.listdir(self.w.out / RUN_ID / "public"), [])

    def test_foreign_file_in_the_run_folder(self):
        run = self.w.out / RUN_ID
        (run / "public").mkdir(parents=True)
        (run / "public" / "notes.txt").write_text("x")
        rc, so, se = self.run_cmd()
        self.assertEqual(rc, 1)
        self.assertIn("is not an output of this command", se)
        self.assertTrue((run / "public" / "notes.txt").exists())
        self.assertFalse((run / "status.json").exists())

    def test_output_inside_an_input(self):
        rc, so, se = self.run_cmd(out=self.w.base / "runs" / "ablation" / "x")
        self.assertEqual(rc, 1)
        self.assertIn("refusing the run folder", se)

    def test_authorisation_and_dirty_tree(self):
        with patched():
            rc, so, se = run_main(self.w.argv("run"))
        self.assertEqual(rc, 1)
        self.assertIn("--i-am-authorized", se)
        rc, so, se = self.run_cmd(REPO_STATE=lambda r: {"commit": "abcdef0", "dirty": True})
        self.assertEqual(rc, 0, so + se)
        self.assertTrue((self.w.out / f"{RUN_ID}-dev" / "status.json").is_file())
        self.assertEqual(json.loads((self.w.out / f"{RUN_ID}-dev" / "public" / "aggregate.json").read_text())["dirty"], True)

    def test_run_id_format(self):
        self.assertTrue(DG.RUN_ID_RE.match("20261107-DEGEN-abcdef0-dev"))
        self.assertFalse(DG.RUN_ID_RE.match("20261107-FOLDSRUN-abcdef0"))
        with self.assertRaises(DG.Failure):
            DG.run_id_for("2026117", STATE)


# ============================================================================ configurazione consegnata e ingressi veri

class TestDelivered(unittest.TestCase):
    def test_delivered_configuration(self):
        p = ROOT / DG.DEFAULT_CONFIG                    # accanto al codice (la copia delle mutazioni la porta con sé)
        if not p.is_file():
            skip_missing(self, "delivered configuration")
        cfg, h = DG.load_config(p)
        self.assertEqual(h, DG.CONFIG_SHA256)
        self.assertEqual(cfg["inputs"]["ablation_run"]["run_id"], "20261106-SESSABLRUN-d999458")
        self.assertEqual(cfg["criteria"]["degenerate"]["restored_epochs"], [1, 2])
        self.assertEqual((cfg["criteria"]["recovery"]["val_accuracy_min"], cfg["criteria"]["recovery"]["val_loss_max"]), (0.75, 1.0))
        self.assertEqual(len(cfg["examples"]), 6)


class TestRealInputs(unittest.TestCase):
    def test_real_inputs_read_only(self):
        if not REAL:
            skip_missing(self, "real not requested")
        p = REAL_ROOT / DG.DEFAULT_CONFIG
        if not p.is_file():
            skip_missing(self, "delivered configuration")
        cfg, _ = DG.load_config(p)
        if not all((REAL_ROOT / cfg["inputs"][g]["folder"] / "status.json").is_file() for g in DG.GROUPS):
            skip_missing(self, "real inputs")
        before = snapshot(REAL_ROOT / "results")
        with patched(PLOTTER=fake_png):
            rc, out, err = run_main(["check-inputs", "--config", str(p), "--input-root", str(REAL_ROOT)])
        self.assertEqual(rc, 0, out + err)
        inp = DG.load_inputs(cfg, REAL_ROOT)
        DG.enrich(cfg, inp)
        T = DG.analyse(cfg, inp)
        # valori della scheda dei fatti (docs/tesi/degeneri-scheda-dei-fatti.md)
        self.assertEqual((T["T1"]["count"], T["T1"]["per_arm"], T["T1"]["per_seed"], T["T1"]["per_fold"]),
                         (10, {"arm_b": 2, "arm_a_star": 5, "arm_b_prime": 3}, {"42": 2, "43": 1, "44": 7}, {"1": 2, "2": 3, "3": 2, "4": 3}))
        self.assertEqual((T["T3"]["saturated_epochs"], T["T3"]["saturated_epochs_with_train_accuracy_above_reference"]), (164, 164))
        self.assertEqual(T["T6"]["non_degenerate_recovered_late"], 22)
        self.assertEqual(T["T11"]["identical_histories"], 6)
        self.assertEqual(len(DG.epoch_rows(inp)), 533)
        self.assertEqual(round([s for s in T["T12"]["scenarios"] if s["selection"] == "all"][0]["with_overhead_seconds"]), 13090)
        self.assertEqual(snapshot(REAL_ROOT / "results"), before)


class ZZRepositoryUntouched(unittest.TestCase):
    def test_declared_skips_only(self):
        self.assertEqual(sorted(MISSING_INPUT), ["delivered configuration", "real inputs", "real not requested"])


def main() -> int:
    before = snapshot(ROOT / "results", ROOT / "export", ROOT / "delivery")
    argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a not in ("--verbose", "--real")] + (["-v"] if "--verbose" in sys.argv else [])
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
