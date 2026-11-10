"""
Test di scripts/nuaa_patience_verify.py (confronto del run di verifica con pazienza 12 con il run genitore), offline e
deterministici.

Mondo sintetico in cartelle temporanee, con identificativi diversi da NUAA (soggetti 1101 e 1102): due run con gli
stessi 36 modelli (3 bracci x 4 fold x 3 seed, dev di 8 frame), schede, soglie, dev_scores e score di test finti (mai
letti: il test lo controlla), aggregati pubblici con la sezione del test. Storie costruite a mano, con l'esito
dell'EarlyStopping calcolato a mano nei commenti (mai con la funzione sotto test):
  genitore (pazienza 4)
    i dieci degeneri   val_loss 5,6,7,8,9 sempre a 0,5 -> migliore 1, arresto 5 (degenere, nessun recupero)
    B f3 s44           val_loss 5,5,4,7,8,9,10, val_acc 0,5 0,5 0,25 ... -> migliore 3, arresto 7 (non degenere, nessun recupero)
    B' f1 s42          val_loss 3,2,1,0,5,0,6,0,7,0,8,0,9 -> migliore 4, arresto 8 (recupero all'epoca 4)
    gli altri 24       val_loss strettamente decrescente, 12 epoche, ripristino 12 (recupero all'epoca 4)
  verifica (pazienza 12: nessun arresto, 12 epoche, ripristino all'argmin): le prime epochs_run epoche uguali al
  genitore, poi la continuazione dello scenario:
    base               i dieci recuperano all'epoca 7 (ripristino 12), B f3 s44 all'epoca 9  -> positive
    one_of_ten         A* f2 s43 resta a 0,5 con val_loss crescente (ripristino 1, degenere)   -> negative
    outside_ten        B f3 s44 resta a 0,5 (ripristino 3, non degenere, nessun recupero)      -> positive, conclusione estesa
    1e-9               val_loss di B f2 s42 all'epoca 2 aumentata di 1e-9                       -> not valid
    INCOMPLETE / FAIL  solo una parte dei modelli completi                                    -> no verdict
Ogni controllo ha un test che lo fa scattare con tutte le ancore ricalcolate (fallisce la regola, non il checksum).
Il test sugli ingressi veri gira solo con --real (sola lettura); altrimenti è saltato con il motivo dichiarato in
MISSING_INPUT. Alla fine si controlla che results/ (salvo results/sessionablationrun/, dove scrive solo il runner del
training), export/ e delivery/ del repository non siano cambiati.

  .venv/bin/python -I scripts/check_nuaa_patience_verify.py [--verbose] [--real]

Variabile facoltativa PATVER_INPUT_ROOT: radice degli ingressi veri e della configurazione consegnata (predefinita: la
radice del repository; il controllo delle mutazioni la sposta su una cartella vuota, così i mutanti non leggono mai i
dati veri).
"""
import contextlib
import copy
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
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nuaa_patience_verify as PV  # noqa: E402

REAL = "--real" in sys.argv
REAL_ROOT = Path(os.environ.get("PATVER_INPUT_ROOT") or ROOT)
MISSING_INPUT = {
    "real not requested": "the read-only test on the real inputs runs only with --real",
    "real inputs": "the parent run folder is not present",
    "delivered configuration": "delivery/patienceverify/nuaa_patience_verify.json is not present",
    "anchors null": "the anchors of the verification run in the delivered configuration are null (to be filled after the run)",
    "previous package configuration": "delivery/degenerate/nuaa_degenerate_trainings.json is not present",
}
STATE = {"commit": "abcdef0", "dirty": False}
DATE = "20261109"
RUN_ID = f"{DATE}-PATVER-abcdef0"
PAR_ID, NEW_ID = "20261106-SESSABLRUN-0000001", "20261109-SESSABLRUN-0000002"
PAR_CFG, NEW_CFG = "a" * 64, "b" * 64
ARMS = ["arm_b", "arm_a_star", "arm_b_prime"]
FOLDS, SEEDS = [1, 2, 3, 4], [42, 43, 44]
NAMES = [f"{a}_fold{f}_seed{s}" for a in ARMS for f in FOLDS for s in SEEDS]
TEN = sorted(["arm_a_star_fold2_seed43", "arm_a_star_fold2_seed44", "arm_a_star_fold3_seed42", "arm_a_star_fold3_seed44",
              "arm_a_star_fold4_seed44", "arm_b_fold1_seed42", "arm_b_fold4_seed44", "arm_b_prime_fold1_seed44",
              "arm_b_prime_fold2_seed44", "arm_b_prime_fold4_seed44"])
NOREC = "arm_b_fold3_seed44"
STOP8 = "arm_b_prime_fold1_seed42"
SUBJ_BF, SUBJ_AT = "1101", "1102"
N_DEV = 8
SPE = {"arm_b": 10.0, "arm_a_star": 20.0, "arm_b_prime": 5.0}    # secondi per epoca, scelti a mano
LR = [0.0010000000474974513] * 3 + [0.0005000000237487257] * 3 + [0.0002500000118743628] * 6

# genitore: (val_loss, val_acc), esito a mano nel docstring
PARENT = {n: ([5, 6, 7, 8, 9], [0.5] * 5) for n in TEN}
PARENT[NOREC] = ([5, 5, 4, 7, 8, 9, 10], [0.5, 0.5, 0.25, 0.375, 0.5, 0.5, 0.5])
PARENT[STOP8] = ([3, 2, 1, 0.5, 0.6, 0.7, 0.8, 0.9], [0.5, 0.625, 0.75, 0.875, 0.875, 0.875, 0.875, 0.875])
GOOD = ([5, 4, 3, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06], [0.5, 0.5, 0.5, 0.875, 0.875, 0.875, 0.875, 1, 1, 1, 1, 1])
# continuazioni del run di verifica (epoche dopo le epochs_run del genitore)
CONT = {n: ([4, 0.9, 0.5, 0.4, 0.3, 0.2, 0.1], [0.5, 0.75, 0.875, 1, 1, 1, 1]) for n in TEN}   # recupero all'epoca 7
CONT[NOREC] = ([3, 0.9, 0.5, 0.4, 0.3], [0.5, 0.75, 0.875, 1, 1])                            # recupero all'epoca 9
CONT[STOP8] = ([0.45, 0.4, 0.35, 0.3], [1, 1, 1, 1])
STAY = {"arm_a_star_fold2_seed43": ([10, 11, 12, 13, 14, 15, 16], [0.5] * 7),                # resta degenere
        NOREC: ([11, 12, 13, 14, 15], [0.5] * 5)}                                           # resta senza recupero, ripristino 3


def skip_missing(test: unittest.TestCase, what: str):
    """Unico modo di saltare un test di questa suite: un motivo dichiarato in MISSING_INPUT."""
    test.skipTest(f"missing input: {MISSING_INPUT[what]}")


def sha(b) -> str:
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def jbytes(doc) -> bytes:
    return (json.dumps(doc, indent=1) + "\n").encode("utf-8")


def history(vl, va) -> dict:
    """Storia completa: loss, accuracy e lr dipendono solo dall'indice (le epoche comuni restano uguali)."""
    n = len(vl)
    assert len(va) == n
    return {"loss": [round(0.2 / (i + 1), 4) for i in range(n)], "accuracy": [min(0.99, 0.9 + 0.01 * i) for i in range(n)],
            "val_loss": [float(v) for v in vl], "val_accuracy": [float(v) for v in va], "lr": LR[:n]}


def dev_scores(val_acc: float, n: int) -> list:
    """(label, score) di un dev bilanciato (prima i bona fide) con accuratezza alla soglia 0,5 uguale a val_acc; a 0,5
    uscita costante «attacco» (AUROC 0,5)."""
    half = n // 2
    k = round(val_acc * n)
    if k == half:
        return [(0, 1.0)] * half + [(1, 1.0)] * half
    bf_ok = min(k, half) if k > half else k // 2 + k % 2
    at_ok = k - bf_ok
    return [(0, 0.1)] * bf_ok + [(0, 0.9)] * (half - bf_ok) + [(1, 0.8)] * at_ok + [(1, 0.2)] * (half - at_ok)


def public_aggregate(run_id: str, status: str, cfg_sha: str, new: bool) -> dict:
    """Aggregato pubblico finto con la sezione del test (verdetti e mediane) e un campo non letto."""
    strata, metrics = ["01", "02", "03", "all"], list(PV.METRICS)

    def v(verdict="inconclusive", direction=None):
        return {"verdict": verdict, "direction": direction, "reason": "the sign differs across the seeds", "label": "secondary"}
    comps = {}
    for i, (c, (mi, su)) in enumerate(PV.COMPARISONS.items()):
        comps[c] = {"minuend": mi, "subtrahend": su, "role": "primary" if i == 0 else "secondary",
                    "verdicts": {s: {m: v() for m in metrics} for s in strata},
                    "pooled_summary_over_seeds": {s: {m: {"values": [0.1, 0.2, 0.3], "median": round(0.01 * (j + 1) * (2 if new else 1), 4),
                                                          "min": 0.1, "max": 0.3} for j, m in enumerate(metrics)} for s in strata}}
    pv = {"auroc": v(), "acer_dev": v()}
    if new:
        comps["a_star_minus_b"]["verdicts"]["03"]["auroc"] = v("consistent", "A* higher AUROC than B")
        comps["a_star_minus_b"]["pooled_summary_over_seeds"]["03"]["acer_dev"]["median"] = None
        pv["auroc"] = v("consistent", "A* higher AUROC than B")
    return {"run_id": run_id, "config_sha256": cfg_sha, "status": status, "strata": strata, "primary_verdicts": pv,
            "comparisons": comps, "per_arm_fold_seed": {"not": "read"}}


def make_config() -> dict:
    c = json.loads((ROOT / PV.DEFAULT_CONFIG).read_text()) if (ROOT / PV.DEFAULT_CONFIG).is_file() else None
    cfg = {"schema": PV.CONFIG_SCHEMA, "task": "synthetic comparison", "request": {"date": "2026-11-07", "card": "C2", "text": "test"},
           "inputs": {"parent_run": {"label": "parent", "run_id": PAR_ID, "folder": "runs/parent", "config_sha256": PAR_CFG,
                                     "commit": "0000001", "early_stopping_patience": 4, "status_sha256": "", "public_sums_sha256": "",
                                     "restricted_sums_sha256": ""},
                      "verification_run": {"label": "patience12", "run_id": NEW_ID, "folder": "runs/new", "config_sha256": NEW_CFG,
                                           "commit": "0000002", "configuration_name": "patience12_verification",
                                           "amendment": {"parent_config_sha256": PAR_CFG, "changed_field": "training.early_stopping_patience",
                                                         "from": 4, "to": 12},
                                           "early_stopping_patience": 12, "status_sha256": "", "public_sums_sha256": "",
                                           "restricted_sums_sha256": "", "anchors_note": "filled by the test world"},
                      "arms": list(ARMS), "folds": list(FOLDS), "seeds": list(SEEDS), "models": 36, "dev_rows": N_DEV},
           "recipe": {"max_epochs": 12, "monitor": "val_loss", "min_delta": 0, "restore_best_weights": True, "batch": 16, "lr": 0.001,
                      "source": "test"},
           "criteria": {"degenerate": {"restored_epochs": [1, 2], "val_acc_at_best": 0.5, "tolerance": 1e-06, "text": "degenerate rule"},
                        "saturation": {"val_accuracy": 0.5, "tolerance": 1e-06, "text": "saturation"},
                        "recovery": {"val_accuracy_min": 0.75, "val_loss_max": 1.0, "text": "recovery rule"},
                        "checkpoint_agreement_tolerance": 1e-06, "shared_history_difference": 0, "source": "test"},
           "original_degenerate": list(TEN), "models_without_recovery_parent": [NOREC],
           "reading_rule": {k: f"rule {k} (A* - B)" for k in "abcde"},
           "time": {"estimate_seconds": 13090, "cap_seconds": 21600, "source": "test"},
           "declarations": {**{k: True for k in PV.DECLARATION_FLAGS}, **{k: f"text {k}" for k in PV.DECLARATION_TEXTS}},
           "outputs": {"public": "public files", "restricted": "none"}}
    assert c is None or sorted(c) == sorted(cfg)            # stesso schema della configurazione consegnata
    return cfg


class World:
    """Ingressi sintetici coerenti; write() li scrive con tutte le ancore ricalcolate (SHA256SUMS, status, configurazione)."""

    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="patver_"))
        self.base = self.tmp / "in"
        self.out = self.tmp / "out"
        self.cfg_path = self.tmp / "cfg" / "nuaa_patience_verify.json"
        self.cfg = make_config()
        self.parent_hist = dict(PARENT)             # storie del genitore (scenario)
        self.cont = dict(CONT)                      # continuazioni del run di verifica (scenario)
        self.new_status = "PASS"
        self.completed = list(NAMES)                # modelli completi del run di verifica
        self.card_edit = {}                         # (run, nome) -> funzione(scheda) prima della scrittura
        self.dev_edit = {}                          # (run, nome) -> funzione(righe)
        self.thr_edit = {}                          # (run, nome) -> funzione(soglia) prima della scrittura
        self.status_edit = {"parent": {}, "new": {}}
        self.agg_edit = {"parent": {}, "new": {}}
        self.null_anchors = False
        self.anchor_override = {}                   # (run, chiave) -> valore nella configurazione
        self.write()

    def histories(self, run: str, name: str) -> dict:
        vl, va = self.parent_hist.get(name, GOOD)
        if run == "new" and len(vl) < 12:
            cv, ca = self.cont[name]
            vl, va = list(vl) + list(cv), list(va) + list(ca)
        return history(vl, va)

    def card(self, run: str, name: str) -> dict:
        arm, fold, seed = PV.name_fields(name)
        h = self.histories(run, name)
        n = len(h["val_loss"])
        r = h["val_loss"].index(min(h["val_loss"])) + 1          # argmin (primo minimo)
        by = "early stopping" if n < 12 else "end of epochs (set_weights)"   # nessuna storia del mondo ferma all'epoca 12
        return {"arch": "LivenessNet", "arm": arm, "fold": fold, "train_seed": seed, "n_fitting": 150, "n_dev": N_DEV, "max_epochs": 12,
                "epochs_run": n, "restored_epoch": r, "restored_by": by, "best_val_loss": h["val_loss"][r - 1],
                "val_acc_at_best": h["val_accuracy"][r - 1], "history": h, "device": "CPU", "visible_devices": ["CPU"], "lr": 0.001,
                "batch": 16, "early_stopping_patience": 4 if run == "parent" else 12, "class_weight": {"0": 1.0, "1": 1.0},
                "train_seconds": round(SPE[arm] * n, 1), "test_evaluation": "none", "cpu_nondeterminism": "declared",
                "weights_file": f"{name}_LivenessNet.keras", "weights_sha256": sha(f"weights {run} {name}"),
                "saved": "2026-11-09T08:00:00+0100", "fitting_keys_sha256": sha(f"fit {name}"), "dev_keys_sha256": "AUTO",
                "val_loss_recomputed": h["val_loss"][r - 1], "val_loss_check": "logits", "val_loss_recomputed_clipped": h["val_loss"][r - 1],
                "test_scored_after_threshold": "2026-11-09T08:00:20+0100"}

    def dev_rows(self, card: dict, name: str) -> list:
        arm, fold, seed = PV.name_fields(name)
        rows = []
        for i, (lab, s) in enumerate(dev_scores(card["val_acc_at_best"], N_DEV)):
            subj = SUBJ_AT if lab else SUBJ_BF
            rows.append({"arm": arm, "fold": str(fold), "train_seed": str(seed), "role": "dev",
                         "sample_id": f"{'attack' if lab else 'real'}/{subj}_{subj}_00_0{lab}_0{fold}_{i}.jpg", "subject_id": subj,
                         "session_id": "01", "label": str(lab), "score_attack": repr(float(s)), "prediction_05": str(int(s > 0.5)),
                         "prediction_dev": "AUTO", "threshold_dev": "AUTO", "file_sha256": sha(f"img {fold} {i}"),
                         "checkpoint_sha256": card["weights_sha256"], "threshold_sha256": "AUTO"})
        return rows

    def write_run(self, run: str, names: list, status: str) -> dict:
        g = "parent_run" if run == "parent" else "verification_run"
        spec = self.cfg["inputs"][g]
        d = self.base / spec["folder"]
        (d / "restricted").mkdir(parents=True)
        files = {}
        for n in names:
            c = self.card(run, n)
            rows = self.dev_rows(c, n)
            rows = self.dev_edit.get((run, n), lambda r: r)(rows)
            if c["dev_keys_sha256"] == "AUTO":
                c["dev_keys_sha256"] = sha("\n".join(r["sample_id"] for r in rows))
            c = self.card_edit.get((run, n), lambda x: x)(c)
            files[f"{n}_card.json"] = jbytes(c)
            arm, fold, seed = PV.name_fields(n)
            thr = {"arm": arm, "fold": fold, "train_seed": seed, "set": "dev", "value": 0.5,
                   "dev_scores_sha256": sha("\n".join(repr(float(r["score_attack"])) for r in rows)),
                   "checkpoint_sha256": c["weights_sha256"], "frozen_at": "2026-11-09T08:00:04+0100"}
            thr = self.thr_edit.get((run, n), lambda t: t)(thr)
            tb = jbytes(thr)
            files[f"{n}_threshold.json"] = tb
            for r in rows:
                r["threshold_sha256"] = sha(tb) if r["threshold_sha256"] == "AUTO" else r["threshold_sha256"]
                r["threshold_dev"] = repr(0.5) if r["threshold_dev"] == "AUTO" else r["threshold_dev"]
                if r["prediction_dev"] == "AUTO":
                    r["prediction_dev"] = "1" if float(r["score_attack"]) > 0.5 else "0"
            buf = io.StringIO(newline="")
            w = csv.DictWriter(buf, fieldnames=list(PV.DEV_COLUMNS), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
            files[f"{n}_dev_scores.csv"] = buf.getvalue().encode()
            files[f"{n}_test_scores.csv"] = b"SECRET TEST SCORES: never read by the comparison\n"
        for fn, b in files.items():
            (d / "restricted" / fn).write_bytes(b)
        for n in names:                                          # gli score di test non si possono nemmeno leggere
            os.chmod(d / "restricted" / f"{n}_test_scores.csv", 0)
        listed = {fn: sha(b) for fn, b in files.items()}
        listed.update({f"{n}_LivenessNet.keras": sha(f"weights {run} {n}") for n in names})
        sums = "".join(f"{h}  {fn}\n" for fn, h in sorted(listed.items())).encode()
        (d / "restricted" / "SHA256SUMS").write_bytes(sums)
        log = f"run log {run}\n".encode()
        (d / "run.log").write_bytes(log)
        st = {"run_id": spec["run_id"], "commit": spec["commit"], "dirty": False, "config_sha256": spec["config_sha256"],
              "delivered_configuration": True, "command": "run", "status": status, "passed": status == "PASS",
              "finished": "2026-11-09T12:00:00+0100", "wall_seconds": 1000.2 if run == "parent" else 13000.0}
        if run == "new":
            st.update({"configuration_name": "patience12_verification",
                       "amendment": {**spec["amendment"], "other_fields": "equal to the parent configuration"}})
        pub = None
        if status != "FAIL":
            (d / "public").mkdir()
            agg = {**public_aggregate(spec["run_id"], status, spec["config_sha256"], run == "new"), **self.agg_edit[run]}
            pfiles = {"aggregate.json": jbytes(agg), "aggregate.md": b"# aggregate\n", "run_record.json": b"{}\n"}
            for fn, b in pfiles.items():
                (d / "public" / fn).write_bytes(b)
            pub = "".join(f"{sha(b)}  {fn}\n" for fn, b in sorted(pfiles.items())).encode()
            (d / "public" / "SHA256SUMS").write_bytes(pub)
            st.update({"completed": list(names), "missing": [n for n in NAMES if n not in names], "stopped": None if status == "PASS" else
                       {"model": "x", "cause": "cap"}, "final_writes_after_cap": False, "public_sums_sha256": sha(pub)})
        else:
            st.update({"reason": "synthetic failure", "outputs": "public removed", "kept_complete_models": list(names)})
        st.update({"restricted_sums_sha256": sha(sums), "run_log_sha256": sha(log), **self.status_edit[run]})
        sb = jbytes(st)
        (d / "status.json").write_bytes(sb)
        return {"status_sha256": sha(sb), "public_sums_sha256": sha(pub) if pub is not None else PV.NO_PUBLIC,
                "restricted_sums_sha256": sha(sums)}

    def write(self) -> None:
        """Riscrive gli ingressi con tutte le ancore coerenti (dopo ogni modifica di un test)."""
        if self.base.exists():
            for p in self.base.rglob("*_test_scores.csv"):
                os.chmod(p, 0o600)
            shutil.rmtree(self.base)
        cfg = self.cfg
        cfg["inputs"]["parent_run"].update(self.write_run("parent", NAMES, "PASS"))
        anchors = self.write_run("new", self.completed, self.new_status)
        cfg["inputs"]["verification_run"].update({k: (None if self.null_anchors else v) for k, v in anchors.items()})
        for (g, k), v in self.anchor_override.items():
            cfg["inputs"][g][k] = v
        self.cfg_path.parent.mkdir(parents=True, exist_ok=True)
        self.cfg_path.write_bytes(jbytes(cfg))

    def patch_files(self, run: str, files: dict, where: str = "restricted", sums: bytes = None) -> None:
        """Riscrive file di un run già scritto e ricalcola le ancore (SHA256SUMS della cartella, status.json,
        configurazione); sums: byte del SHA256SUMS usati così come sono (SHA256SUMS malformati o incompleti)."""
        g = "parent_run" if run == "parent" else "verification_run"
        d = self.base / self.cfg["inputs"][g]["folder"]
        for fn, b in files.items():
            (d / where / fn).write_bytes(b)
        if sums is None:
            listed = PV.read_sums((d / where / "SHA256SUMS").read_bytes(), "sums")
            listed.update({fn: sha(b) for fn, b in files.items()})
            sums = "".join(f"{h}  {fn}\n" for fn, h in sorted(listed.items())).encode()
        (d / where / "SHA256SUMS").write_bytes(sums)
        st = json.loads((d / "status.json").read_text())
        st[f"{where}_sums_sha256"] = sha(sums)
        sb = jbytes(st)
        (d / "status.json").write_bytes(sb)
        self.cfg["inputs"][g].update({"status_sha256": sha(sb), f"{where}_sums_sha256": sha(sums)})
        self.cfg_path.write_bytes(jbytes(self.cfg))

    def argv(self, cmd: str, *extra, out=None) -> list:
        a = [cmd, "--config", str(self.cfg_path), "--input-root", str(self.base)]
        if cmd == "run":
            a += ["--out-root", str(out or self.out), "--date", DATE]
        return a + list(extra)

    def close(self):
        for p in self.tmp.rglob("*_test_scores.csv"):
            os.chmod(p, 0o600)
        shutil.rmtree(self.tmp, ignore_errors=True)


def fake_png(kind, data) -> bytes:
    return b"\x89PNG-" + kind.encode() + sha(json.dumps(data, sort_keys=True)).encode()


@contextlib.contextmanager
def patched(**attrs):
    """Stato git finto e figure finte (salvo dove un test le sostituisce) per la durata."""
    new = {"REPO_STATE": lambda r: dict(STATE), "PLOTTER": fake_png, **attrs}
    old = {k: getattr(PV, k) for k in new}
    for k, v in new.items():
        setattr(PV, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(PV, k, v)


def run_main(argv) -> tuple:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = PV.main(argv)
    return rc, out.getvalue(), err.getvalue()


def snapshot(*dirs, skip=()) -> dict:
    out = {}
    for d in dirs:
        d = Path(d)
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if any(p == s or s in p.parents for s in skip):
                    continue
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

    def run_ok(self, out=None, **attrs) -> dict:
        rc, so, se = self.run_cmd(out=out, **attrs)
        self.assertEqual(rc, 0, so + se)
        return json.loads(((out or self.w.out) / RUN_ID / "public" / "aggregate.json").read_text())

    def public(self, name: str, out=None) -> str:
        return ((out or self.w.out) / RUN_ID / "public" / name).read_text()

    def csv_rows(self, name: str) -> list:
        return list(csv.DictReader(io.StringIO(self.public(name))))


# ============================================================================ regole (casi a mano)

class TestRules(unittest.TestCase):
    CRIT = make_config()["criteria"]

    def h(self, vl, va=None):
        return {"val_loss": vl, "val_accuracy": va or [0.5] * len(vl)}

    def test_patience_4_best_at_epoch_1_stops_at_5(self):
        s = PV.simulate_es(self.h([5, 6, 7, 8, 9]), 12, patience=4)
        self.assertEqual((s["stop"], s["restored"], s["early"]), (5, 1, True))

    def test_patience_12_never_stops_within_12_epochs(self):
        # anche con il migliore all'epoca 1, wait arriva a 11 all'epoca 12: nessun arresto, fine delle epoche
        s = PV.simulate_es(self.h([5] + [6] * 11), 12, patience=12)
        self.assertEqual((s["stop"], s["restored"], s["early"]), (12, 1, False))
        self.assertIsNone(PV.simulate_es(self.h([5, 6, 7, 8, 9]), 12, patience=12))

    def test_equal_value_is_not_an_improvement(self):
        s = PV.simulate_es(self.h([5, 5, 5, 5, 5]), 12, patience=4)
        self.assertEqual((s["stop"], s["restored"]), (5, 1))

    def test_recovery_boundaries(self):
        self.assertEqual(PV.first_recovery(self.h([1.0, 0.99], [0.75, 0.75]), 0.75, 1.0), 2)     # val_loss 1,0 non è < 1
        self.assertIsNone(PV.first_recovery(self.h([0.5], [0.7499]), 0.75, 1.0))
        self.assertEqual(PV.first_recovery(self.h([0.5], [0.75]), 0.75, 1.0), 1)                  # 0,75 è >= 0,75

    def test_degenerate_criterion(self):
        self.assertTrue(PV.is_degenerate(2, 0.5 + 5e-7, self.CRIT))
        self.assertFalse(PV.is_degenerate(3, 0.5, self.CRIT))
        self.assertFalse(PV.is_degenerate(1, 0.5 + 2e-6, self.CRIT))

    def test_auroc_with_ties(self):
        self.assertEqual(PV.auroc([0, 0, 1, 1], [1.0, 1.0, 1.0, 1.0]), 0.5)
        self.assertEqual(PV.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(PV.auroc([0, 1], [0.9, 0.1]), 0.0)

    def test_shared_history_identical_and_1e9(self):
        hp = history([5, 6, 7, 8, 9], [0.5] * 5)
        hn = history([5, 6, 7, 8, 9, 4, 0.9], [0.5] * 5 + [0.5, 0.75])
        s = PV.shared_history(hp, hn, 5)
        self.assertTrue(s["identical"])
        self.assertEqual({k: v["epoch"] for k, v in s["fields"].items()}, {k: None for k in PV.HISTORY_KEYS})
        hn["val_loss"][1] += 1e-9                           # epoca 2
        hn["lr"][3] *= 1.0 + 1e-12                          # epoca 4, scarto più piccolo ma non zero
        s = PV.shared_history(hp, hn, 5)
        self.assertFalse(s["identical"])
        self.assertEqual(s["fields"]["val_loss"]["epoch"], 2)
        self.assertGreater(s["fields"]["val_loss"]["max_abs_difference"], 0)
        self.assertLess(s["fields"]["val_loss"]["max_abs_difference"], 2e-9)
        self.assertEqual(s["fields"]["lr"]["epoch"], 4)

    def test_shared_history_only_on_the_shared_epochs(self):
        hp = history([5, 6, 7, 8, 9], [0.5] * 5)
        hn = history([5, 6, 7, 8, 9, 1, 1], [0.5] * 7)
        hn["val_loss"][5] = 123.0                           # dopo le epoche comuni: non conta
        self.assertTrue(PV.shared_history(hp, hn, 5)["identical"])

    def test_shared_history_shorter_new_is_not_identical(self):
        hp = history([5, 6, 7, 8, 9], [0.5] * 5)
        hn = history([5, 6, 7, 8], [0.5] * 4)
        s = PV.shared_history(hp, hn, 5)
        self.assertFalse(s["complete"])
        self.assertFalse(s["identical"])


# ============================================================================ configurazione

class TestConfig(WorldCase):
    def edit(self, fn):
        cfg = json.loads(self.w.cfg_path.read_text())
        fn(cfg)
        self.w.cfg_path.write_text(json.dumps(cfg))

    def test_unknown_top_key(self):
        self.edit(lambda c: c.update({"extra": 1}))
        self.assertFails("configuration: fields must be exactly")

    def test_partial_null_anchors(self):
        self.edit(lambda c: c["inputs"]["verification_run"].update({"status_sha256": None}))
        self.assertFails("must be all null (to be filled after the run) or all filled")

    def test_malformed_anchor(self):
        self.edit(lambda c: c["inputs"]["verification_run"].update({"status_sha256": "ABC"}))
        self.assertFails("must be SHA-256 values")

    def test_none_only_for_public(self):
        self.edit(lambda c: c["inputs"]["verification_run"].update({"restricted_sums_sha256": "none"}))
        self.assertFails("must be SHA-256 values")

    def test_shared_history_difference_must_be_zero(self):
        self.edit(lambda c: c["criteria"].update({"shared_history_difference": 1e-9}))
        self.assertFails("criteria.shared_history_difference must be 0")

    def test_ten_must_be_ten_distinct_models(self):
        self.edit(lambda c: c.update({"original_degenerate": TEN[:9]}))
        self.assertFails("original_degenerate must list the ten")
        self.edit(lambda c: c.update({"original_degenerate": TEN[:9] + ["arm_c_fold1_seed42"]}))
        self.assertFails("original_degenerate must list the ten")

    def test_without_recovery_outside_the_ten(self):
        self.edit(lambda c: c.update({"models_without_recovery_parent": [TEN[0]]}))
        self.assertFails("models_without_recovery_parent must list models of the run outside the ten")

    def test_amendment_must_be_4_to_12(self):
        self.edit(lambda c: c["inputs"]["verification_run"]["amendment"].update({"to": 12.0}))
        self.assertFails("amendment must declare training.early_stopping_patience 4 -> 12")
        self.edit(lambda c: c["inputs"]["verification_run"]["amendment"].update({"to": 12, "parent_config_sha256": "c" * 64}))
        self.assertFails("parent_config_sha256 must be the config_sha256 of the parent run")

    def test_amendment_values(self):
        needle = "amendment must declare training.early_stopping_patience 4 -> 12"
        for k, v in (("changed_field", "training.batch"), ("from", 5), ("to", 8)):
            with self.subTest(field=k):
                cfg = copy.deepcopy(self.w.cfg)
                cfg["inputs"]["verification_run"]["amendment"][k] = v
                self.w.cfg_path.write_bytes(jbytes(cfg))
                self.assertFails(needle)

    def test_patience_of_the_runs_and_max_epochs(self):
        self.edit(lambda c: c["inputs"]["parent_run"].update({"early_stopping_patience": 5}))
        self.assertFails("the patience of the two runs must be the from and to values")

    def test_saturation_must_match_degenerate(self):
        self.edit(lambda c: c["criteria"]["saturation"].update({"tolerance": 1e-5}))
        self.assertFails("criteria.saturation must use the value and tolerance")

    def test_run_id_and_commit(self):
        self.edit(lambda c: c["inputs"]["verification_run"].update({"run_id": NEW_ID + "-dev"}))
        self.assertFails("run id of the session ablation runner without -dev")
        self.edit(lambda c: c["inputs"]["verification_run"].update({"run_id": NEW_ID, "commit": "0000003"}))
        self.assertFails("commit must be the commit of its run id")

    def test_declaration_flags_and_estimate(self):
        self.edit(lambda c: c["declarations"].update({"no_test_scores_read": False}))
        self.assertFails("declarations.no_test_scores_read must be true")
        self.edit(lambda c: (c["declarations"].update({"no_test_scores_read": True}), c["time"].update({"estimate_seconds": 30000})))
        self.assertFails("time.estimate_seconds must be positive and not above time.cap_seconds")


# ============================================================================ esiti del confronto

class TestOutcomes(WorldCase):
    def test_all_recover_positive(self):
        doc = self.run_ok()
        b = doc["sections"]["b"]
        self.assertEqual((doc["verdict"], b["verdict"]), ("positive", "positive"))
        self.assertEqual((b["ten_recovered"], b["ten_complete"]), (10, 10))
        self.assertEqual(set(b["ten_first_recovery"].values()), {7})
        self.assertEqual((b["ten_without_recovery"], b["degenerate_among_complete"], b["without_recovery_outside_ten"]), ([], [], []))
        self.assertEqual(len(b["without_recovery_parent"]), 11)          # i dieci e B f3 s44
        self.assertIn("B fold 3 seed 44", b["without_recovery_parent"])
        self.assertNotIn("not enough", b["conclusion"])
        self.assertNotIn("does not rule out", b["conclusion"])
        self.assertIn("does not separate the stop from the lr", b["conclusion"])
        self.assertIn("a positive outcome does not by itself show that every model is stable", b["conclusion"])
        a = doc["sections"]["a"]
        self.assertEqual((a["identical"], a["models_compared"], a["valid"], a["largest_difference"]), (36, 36, True, None))
        self.assertEqual(sorted(os.listdir(self.w.out / RUN_ID / "public")), sorted(PV.PUBLIC_OUTPUTS))
        self.assertEqual(len(self.csv_rows("comparison.csv")), 36)
        self.assertEqual(len(self.csv_rows("curves.csv")), 36 * 12)
        st = json.loads((self.w.out / RUN_ID / "status.json").read_text())
        self.assertEqual((st["status"], st["verdict"], st["verification_status"]), ("PASS", "positive", "PASS"))
        self.assertFalse((self.w.out / RUN_ID / "restricted").exists())

    def test_one_of_the_ten_does_not_recover_negative(self):
        self.w.cont["arm_a_star_fold2_seed43"] = STAY["arm_a_star_fold2_seed43"]
        self.w.write()
        doc = self.run_ok()
        b = doc["sections"]["b"]
        self.assertEqual(b["verdict"], "negative")
        self.assertEqual(b["ten_without_recovery"], ["A* fold 2 seed 43"])
        self.assertEqual(b["degenerate_among_complete"], ["A* fold 2 seed 43"])
        self.assertIn("this change is not enough", b["conclusion"])
        self.assertIn("does not rule out a role of early stopping", b["conclusion"])
        self.assertNotIn("every model stable", b["conclusion"])
        self.assertIn("**Verdict: negative.**", self.public("aggregate.md"))

    # A* f3 s42 (uno dei dieci) nel nuovo run: val_loss 4, 3, 1,5, 1,0, 1,2, 1,1, 1,05 dopo le 5 epoche comuni,
    # val_accuracy 0,5, 0,625, 0,75, 0,75, 0,625, 0,625, 0,625 -> minimo all'epoca 9 (1,0): ripristino 9,
    # val_acc 0,75, non degenere; nessun recupero (epoca 8: 0,75 ma val_loss 1,5; epoca 9: 0,75 ma val_loss 1,0,
    # non < 1)
    NOT_REC = "arm_a_star_fold3_seed42"
    NOT_REC_CONT = ([4, 3, 1.5, 1.0, 1.2, 1.1, 1.05], [0.5, 0.625, 0.75, 0.75, 0.625, 0.625, 0.625])

    def test_one_of_the_ten_without_recovery_not_degenerate_is_negative(self):
        self.w.cont[self.NOT_REC] = self.NOT_REC_CONT
        self.w.write()
        doc = self.run_ok()
        b = doc["sections"]["b"]
        self.assertEqual((doc["verdict"], b["verdict"]), ("negative", "negative"))
        self.assertEqual((b["ten_without_recovery"], b["ten_degenerate"], b["degenerate_among_complete"]),
                         (["A* fold 3 seed 42"], [], []))
        self.assertEqual((b["ten_recovered"], b["ten_complete"], doc["sections"]["a"]["valid"]), (9, 10, True))
        self.assertIn("Negative: this change is not enough (among the ten, without recovery: A* fold 3 seed 42; "
                      "degenerate among the 36: none). This does not rule out a role of early stopping",
                      b["conclusion"])
        st = json.loads((self.w.out / RUN_ID / "status.json").read_text())
        self.assertEqual((st["status"], st["verdict"]), ("PASS", "negative"))
        md = self.public("aggregate.md")
        sec = md[md.index("## (b) Primary outcome"):md.index("## (c) Secondary outcomes")]
        self.assertIn("**Verdict: negative.**", sec)
        self.assertIn("this change is not enough", sec)
        self.assertIn("does not rule out a role of early stopping", sec)
        self.assertIn("Without recovery among the ten: A\\* fold 3 seed 42.", sec)
        self.assertIn("Degenerate among the complete models of the verification run: none.", sec)

    def test_recovery_flag_needs_val_loss_below_1(self):
        self.w.cont[self.NOT_REC] = self.NOT_REC_CONT
        self.w.write()
        self.run_ok()
        rows = [r for r in self.csv_rows("curves.csv") if (r["arm"], r["fold"], r["seed"]) == ("arm_a_star", "3", "42")]
        self.assertEqual([(r["val_accuracy"], r["val_loss"]) for r in rows[7:9]], [("0.75", "1.5"), ("0.75", "1.0")])
        self.assertEqual([r["recovered_by_epoch"] for r in rows], ["0"] * 12)
        self.assertEqual([r["is_restored"] for r in rows], ["0"] * 8 + ["1"] + ["0"] * 3)
        c = [r for r in self.csv_rows("comparison.csv")
             if (r["arm"], r["fold"], r["seed"]) == ("arm_a_star", "3", "42")][0]
        self.assertEqual((c["new_recovered"], c["new_first_recovery"], c["new_restored_epoch"], c["new_degenerate"]),
                         ("0", "", "9", "0"))

    def test_recovered_but_still_degenerate_is_negative(self):
        # B' f4 s44: val_loss 0,5 all'epoca 1 con il dev a 0,5 (degenere nel genitore); nel nuovo run recupera all'epoca 7
        # (0,75 e 0,9 < 1) ma il minimo resta l'epoca 1 (0,5 < 0,9): ripristino 1, ancora degenere -> negativo
        n = "arm_b_prime_fold4_seed44"
        self.w.parent_hist[n] = ([0.5, 6, 7, 8, 9], [0.5] * 5)
        self.w.cont[n] = ([4, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99], [0.5, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75])
        self.w.write()
        doc = self.run_ok()
        b = doc["sections"]["b"]
        self.assertEqual(b["ten_without_recovery"], [])
        self.assertEqual(b["ten_first_recovery"]["B' fold 4 seed 44"], 7)
        self.assertEqual((b["ten_degenerate"], b["degenerate_among_complete"]), (["B' fold 4 seed 44"], ["B' fold 4 seed 44"]))
        self.assertEqual(b["verdict"], "negative")
        self.assertIn("degenerate among the 36: B' fold 4 seed 44", b["conclusion"])

    def test_outside_the_ten_without_recovery_extends_the_conclusion(self):
        self.w.cont[NOREC] = STAY[NOREC]
        self.w.write()
        doc = self.run_ok()
        b = doc["sections"]["b"]
        self.assertEqual(b["verdict"], "positive")
        self.assertEqual(b["without_recovery_outside_ten"], ["B fold 3 seed 44"])
        self.assertEqual(b["degenerate_among_complete"], [])                 # ripristino 3: non degenere
        self.assertIn("Outside the ten, without recovery within 12 epochs: B fold 3 seed 44; this change is not enough to make every "
                      "model stable.", b["conclusion"])
        self.assertIn("This does not rule out a role of early stopping", b["conclusion"])     # NOTE (b): «in either case»
        self.assertIn("Without recovery outside the ten (listed separately): B fold 3 seed 44.", self.public("aggregate.md"))

    def test_shared_history_differing_by_1e9_is_not_valid(self):
        def bump(c):
            c["history"]["val_loss"][1] += 1e-9
            return c
        self.w.card_edit[("new", "arm_b_fold2_seed42")] = bump
        self.w.write()
        doc = self.run_ok()
        a, b = doc["sections"]["a"], doc["sections"]["b"]
        self.assertEqual((a["valid"], a["identical"]), (False, 35))
        w = a["largest_difference"]
        self.assertEqual((w["model"], w["epoch"], w["field"]), ("B fold 2 seed 42", 2, "val_loss"))
        self.assertTrue(0 < w["max_abs_difference"] < 2e-9)
        self.assertEqual(b["verdict"], "not valid")
        self.assertIn("no verdict", b["conclusion"])
        row = [r for r in self.csv_rows("comparison.csv") if (r["arm"], r["fold"], r["seed"]) == ("arm_b", "2", "42")][0]
        self.assertEqual(row["shared_identical"], "0")
        self.assertNotEqual(row["max_abs_diff_shared_val_loss"], "0.0")
        self.assertEqual(row["max_abs_diff_shared_loss"], "0.0")

    def test_difference_at_the_last_shared_epoch(self):
        # B f1 s42 (uno dei dieci): 5 epoche comuni; train loss dell'epoca 5 aumentata di 1e-9 -> non valido
        def bump(c):
            c["history"]["loss"][4] += 1e-9
            return c
        self.w.card_edit[("new", "arm_b_fold1_seed42")] = bump
        self.w.write()
        doc = self.run_ok()
        w = doc["sections"]["a"]["largest_difference"]
        self.assertEqual((w["model"], w["epoch"], w["field"]), ("B fold 1 seed 42", 5, "loss"))
        self.assertEqual(doc["verdict"], "not valid")

    def test_tie_at_half_is_bona_fide(self):
        # B' f2 s44 del genitore (degenere, acc 0,5): bona fide a 0,5 esatto (giusti: pareggio = bona fide), attacchi a 0,2
        # (sbagliati) -> acc 0,5 alla soglia 0,5, AUROC 0 (gli attacchi sotto i bona fide)
        def edit(rows):
            for r in rows:
                r["score_attack"] = repr(0.5 if r["label"] == "0" else 0.2)
                r["prediction_05"] = "0"
            return rows
        self.w.dev_edit[("parent", "arm_b_prime_fold2_seed44")] = edit
        self.w.write()
        self.run_ok()
        r = [r for r in self.csv_rows("comparison.csv") if (r["arm"], r["fold"], r["seed"]) == ("arm_b_prime", "2", "44")][0]
        self.assertEqual((r["parent_dev_acc_at_checkpoint"], r["parent_dev_auroc_at_checkpoint"]), ("0.5", "0.0"))

    def test_incomplete_run_gives_no_verdict(self):
        self.w.new_status = "INCOMPLETE"
        self.w.completed = NAMES[:20]
        self.w.write()
        doc = self.run_ok()
        b = doc["sections"]["b"]
        self.assertEqual(b["verdict"], "no verdict")
        self.assertEqual(len(b["missing_models"]), 16)
        self.assertIn("INCOMPLETE", b["conclusion"])
        self.assertFalse(doc["sections"]["d"]["reported"])
        self.assertEqual(doc["sections"]["a"]["models_compared"], 20)
        rows = self.csv_rows("comparison.csv")
        self.assertEqual((len(rows), sum(r["new_status"] == "missing" for r in rows)), (36, 16))
        self.assertEqual(len(self.csv_rows("curves.csv")), 20 * 12)
        self.assertIn("not reported: the verification run is INCOMPLETE", self.public("aggregate.md"))

    def test_fail_run_with_kept_models_gives_no_verdict(self):
        self.w.new_status = "FAIL"
        self.w.completed = NAMES[:5]
        self.w.write()
        self.assertEqual(self.w.cfg["inputs"]["verification_run"]["public_sums_sha256"], "none")
        doc = self.run_ok()
        self.assertEqual(doc["verdict"], "no verdict")
        self.assertEqual(doc["inputs"]["verification_run"]["test_section"], "none (FAIL run: no public folder)")

    def test_no_verdict_takes_precedence_over_not_valid(self):
        self.w.new_status = "INCOMPLETE"
        self.w.completed = NAMES[:20]
        self.w.card_edit[("new", "arm_b_fold2_seed42")] = lambda c: (c["history"]["val_loss"].__setitem__(1, 4 + 1e-9), c)[1]
        self.w.write()
        doc = self.run_ok()
        self.assertEqual((doc["verdict"], doc["sections"]["a"]["valid"]), ("no verdict", False))

    def test_secondary_values_by_hand(self):
        self.run_ok()
        rows = {(r["arm"], r["fold"], r["seed"]): r for r in self.csv_rows("comparison.csv")}
        r = rows[("arm_b", "1", "42")]                       # uno dei dieci
        self.assertEqual((r["in_ten"], r["parent_epochs_run"], r["parent_restored_epoch"], r["parent_degenerate"], r["parent_recovered"],
                          r["parent_first_recovery"]), ("1", "5", "1", "1", "0", ""))
        self.assertEqual((r["parent_dev_acc_at_checkpoint"], r["parent_dev_auroc_at_checkpoint"], r["parent_epochs_at_half"]),
                         ("0.5", "0.5", "5"))
        self.assertEqual((r["new_epochs_run"], r["new_restored_epoch"], r["new_degenerate"], r["new_first_recovery"],
                          r["new_best_val_loss"], r["new_dev_acc_at_checkpoint"], r["new_dev_auroc_at_checkpoint"], r["new_epochs_at_half"]),
                         ("12", "12", "0", "7", "0.1", "1.0", "1.0", "6"))
        self.assertEqual((r["parent_lr_last"], r["new_lr_last"]), (repr(LR[4]), repr(LR[11])))
        self.assertEqual((r["shared_epochs"], r["shared_identical"]), ("5", "1"))
        s = rows[(STOP8.split("_fold")[0], "1", "42")]
        self.assertEqual((s["parent_epochs_run"], s["parent_restored_epoch"], s["parent_first_recovery"], s["new_restored_epoch"]),
                         ("8", "4", "4", "12"))
        n = rows[("arm_b", "3", "44")]                        # B f3 s44: dev 0,25 al checkpoint, 2 frame su 8
        self.assertEqual((n["parent_restored_epoch"], n["parent_dev_acc_at_checkpoint"], n["parent_recovered"], n["new_first_recovery"],
                          n["parent_epochs_at_half"]), ("3", "0.25", "0", "9", "5"))

    def test_curves_flags(self):
        self.run_ok()
        rows = [r for r in self.csv_rows("curves.csv") if (r["arm"], r["fold"], r["seed"]) == ("arm_b", "1", "42")]
        self.assertEqual([r["shared_with_parent"] for r in rows], ["1"] * 5 + ["0"] * 7)
        self.assertEqual([r["is_restored"] for r in rows], ["0"] * 11 + ["1"])
        self.assertEqual([r["recovered_by_epoch"] for r in rows], ["0"] * 6 + ["1"] * 6)
        self.assertEqual([r["saturated"] for r in rows], ["1"] * 6 + ["0"] * 6)

    def test_times_by_hand(self):
        e = self.run_ok()["sections"]["e"]
        par = {r["arm"]: r for r in e["per_arm"] if r["run"] == "parent"}
        new = {r["arm"]: r for r in e["per_arm"] if r["run"] == "new"}
        # B nel genitore: due dei dieci (5 epoche), B f3 s44 (7), nove a 12 -> 125 epoche, 10 s ciascuna
        self.assertEqual((par["B"]["epochs"], par["B"]["train_seconds"], par["B"]["seconds_per_epoch"]), (125, 1250.0, 10.0))
        self.assertEqual((new["A*"]["epochs"], new["A*"]["seconds_per_epoch"]), (144, 20.0))
        self.assertEqual(e["new_train_seconds"], 144 * (10 + 20 + 5))
        self.assertEqual((e["new_wall_seconds"], e["estimate_seconds"], e["wall_minus_estimate_seconds"], e["within_cap"]),
                         (13000.0, 13090, -90.0, True))


# ============================================================================ test dal pubblico, identificativi, determinismo

class TestPublic(WorldCase):
    def test_test_section_read_from_the_public_only(self):
        reads = []
        real = PV.read_regular

        def rec(p, what):
            reads.append(Path(p).name)
            return real(p, what)
        doc = self.run_ok(read_regular=rec)
        self.assertFalse([n for n in reads if "test_scores" in n])
        self.assertIn("aggregate.json", reads)
        d = doc["sections"]["d"]
        self.assertTrue(d["reported"])
        self.assertEqual(d["label"], "not used to choose or judge the recipe")
        prim = {p["metric"]: p for p in d["primary"]}
        self.assertEqual((prim["auroc"]["parent"]["verdict"], prim["auroc"]["new"]["verdict"], prim["auroc"]["new"]["direction"]),
                         ("inconclusive", "consistent", "A* higher AUROC than B"))
        rows = {(r["comparison"], r["stratum"], r["metric"]): r for r in d["rows"]}
        self.assertEqual(len(rows), 3 * 4 * 7)
        self.assertEqual(rows[("A* - B", "03", "auroc")]["new"]["verdict"], "consistent")
        self.assertEqual(d["verdicts_changed"], 1)
        self.assertEqual((rows[("B' - B", "01", "auroc")]["parent"]["median"], rows[("B' - B", "01", "auroc")]["new"]["median"]), (0.07, 0.14))
        self.assertIsNone(rows[("A* - B", "03", "acer_dev")]["new"]["median"])
        md = self.public("aggregate.md")
        self.assertIn("## (d) Test, not used to choose or judge the recipe", md)
        self.assertLess(md.index("## (b) Primary outcome"), md.index("## (d) Test"))
        self.assertLess(md.index("## (a) Shared histories"), md.index("## (b) Primary outcome"))
        self.assertLess(md.index("## (c) Secondary outcomes"), md.index("## (d) Test"))
        self.assertLess(md.index("## (d) Test"), md.index("## (e) Times"))
        self.assertLess(md.index("## (e) Times"), md.index("## Declarations"))

    def test_test_verdicts_do_not_change_the_primary_verdict(self):
        base = self.run_ok()["verdict"]
        w2 = World()
        self.addCleanup(w2.close)
        w2.agg_edit["new"] = {"primary_verdicts": {"auroc": {"verdict": "inconclusive", "direction": None, "reason": "r"},
                                                   "acer_dev": {"verdict": "consistent", "direction": "A* lower ACER", "reason": "r"}}}
        w2.write()
        with patched():
            rc, so, se = run_main(w2.argv("run", "--i-am-authorized"))
        self.assertEqual(rc, 0, so + se)
        doc = json.loads((w2.out / RUN_ID / "public" / "aggregate.json").read_text())
        self.assertEqual(doc["verdict"], base)
        self.assertEqual({p["metric"]: p["new"]["verdict"] for p in doc["sections"]["d"]["primary"]},
                         {"auroc": "inconclusive", "acer_dev": "consistent"})

    def test_no_identifier_in_the_public(self):
        self.run_ok()
        pub = self.w.out / RUN_ID / "public"
        for n in os.listdir(pub):
            text = (pub / n).read_bytes().decode("utf-8", errors="replace")
            for bad in ("real/", "attack/", ".jpg", "SECRET"):
                self.assertNotIn(bad, text, n)
            for s in (SUBJ_BF, SUBJ_AT):                    # parola intera, come la scansione (non dentro un hash o un numero)
                self.assertIsNone(re.search(rf"(?<![\w.]){s}(?![\w.])", text), n)

    def test_figure_labels_are_scanned(self):
        old = PV.figure_data
        try:
            def leaky(cfg, inp, T):
                d = old(cfg, inp, T)
                d["ten"][0]["label"] = f"subject {SUBJ_AT}"
                return d
            PV.figure_data = leaky
            st = self.assertRunFailsClean("public file figure labels refused: subject id")
            self.assertNotIn(SUBJ_AT, st["reason"])
        finally:
            PV.figure_data = old

    def test_markdown_markers_refused(self):
        self.w.cfg["reading_rule"]["b"] = "rule b, FIX" + "ME later"
        self.w.write()
        self.assertRunFailsClean("public file aggregate.md refused: unfinished-work marker")

    def test_ip_address_refused(self):
        self.w.cfg["reading_rule"]["c"] = "rule c from 10.1.2.3"
        self.w.write()
        self.assertRunFailsClean("refused: IP address")

    def test_determinism_two_runs_same_bytes(self):
        o1, o2 = self.w.tmp / "o1", self.w.tmp / "o2"
        self.run_ok(out=o1)
        self.run_ok(out=o2)
        p1, p2 = o1 / RUN_ID / "public", o2 / RUN_ID / "public"
        self.assertEqual(sorted(os.listdir(p1)), sorted(os.listdir(p2)))
        for n in os.listdir(p1):
            self.assertEqual((p1 / n).read_bytes(), (p2 / n).read_bytes(), n)

    def test_figures_data(self):
        seen = {}

        def rec(kind, data):
            seen[kind] = data
            return fake_png(kind, data)
        doc = self.run_ok(PLOTTER=rec)
        self.assertEqual(sorted(seen), ["no_recovery", "ten", "val_accuracy_all"])
        self.assertEqual(len(seen["ten"]["ten"]), 10)
        self.assertEqual(len(seen["ten"]["no_recovery"]), 11)              # i dieci e B f3 s44 (senza recupero nel genitore)
        self.assertEqual(len(seen["ten"]["all"]), 36)
        it = seen["ten"]["ten"][0]
        self.assertEqual((len(it["parent"]["val_loss"]), len(it["new"]["val_loss"])), (5, 12))
        self.assertEqual(doc["figures"]["files"], sorted(PV.FIGURES))

    def test_figures_skipped_without_matplotlib(self):
        def missing(kind, data):
            raise ImportError("no matplotlib", name="matplotlib")
        doc = self.run_ok(PLOTTER=missing)
        self.assertEqual(doc["figures"]["note"], "skipped: missing Python package 'matplotlib'")
        self.assertFalse(any(n.endswith(".png") for n in os.listdir(self.w.out / RUN_ID / "public")))

    def test_sha256sums_cover_every_public_file(self):
        self.run_ok()
        pub = self.w.out / RUN_ID / "public"
        listed = PV.read_sums((pub / "SHA256SUMS").read_bytes(), "sums")
        self.assertEqual(sorted(listed), sorted(n for n in os.listdir(pub) if n != "SHA256SUMS"))
        for n, h in listed.items():
            self.assertEqual(sha((pub / n).read_bytes()), h)


# ============================================================================ rifiuti

class TestRefusals(WorldCase):
    def test_null_anchors_refuse_run_and_check_inputs_is_structural(self):
        self.w.null_anchors = True
        self.w.write()
        reads = []
        real = PV.read_regular

        def rec(p, what):
            reads.append(str(p))
            return real(p, what)
        rc, so, se = self.run_cmd(read_regular=rec)
        self.assertEqual(rc, 1)
        self.assertIn("the anchors of the verification run are null (to be filled after the run): run refused", se)
        self.assertFalse((self.w.out / RUN_ID).exists())
        reads.clear()
        rc, so, se = self.check(read_regular=rec)
        self.assertEqual(rc, 0, so + se)
        self.assertIn("RESULT: PASS (check-inputs, structural only: the anchors of the verification run are null", so)
        self.assertIn("verification run 20261109-SESSABLRUN-0000002 not read: anchors null", so)
        self.assertFalse([p for p in reads if "/runs/new/" in p])

    def test_parent_not_pass(self):
        self.w.status_edit["parent"] = {"status": "INCOMPLETE", "passed": False}
        self.w.write()
        self.assertFails("parent_run status.json: status 'INCOMPLETE'")

    def test_new_status_hash_differs(self):
        self.w.anchor_override[("verification_run", "status_sha256")] = "0" * 64
        self.w.write()
        self.assertFails("verification_run status.json: SHA-256")

    def test_public_sums_hash_differs(self):
        self.w.anchor_override[("parent_run", "public_sums_sha256")] = "0" * 64
        self.w.write()
        self.assertFails("parent_run status.json: the public SHA256SUMS anchor differs")

    def test_restricted_sums_hash_differs(self):
        self.w.anchor_override[("verification_run", "restricted_sums_sha256")] = "0" * 64
        self.w.write()
        self.assertFails("verification_run status.json: the restricted SHA256SUMS anchor differs")

    def test_public_none_for_a_pass_run(self):
        self.w.anchor_override[("verification_run", "public_sums_sha256")] = "none"
        self.w.write()
        self.assertFails("must be 'none' exactly when the run is FAIL")

    def test_new_status_identity(self):
        for k, v, needle in (("config_sha256", "c" * 64, "config_sha256"), ("commit", "0000009", "commit"),
                             ("configuration_name", "session_ablation_run", "configuration_name"),
                             ("run_id", "20261109-SESSABLRUN-0000009", "run_id"), ("dirty", True, "not clean")):
            with self.subTest(field=k):
                self.w.status_edit["new"] = {k: v}
                self.w.write()
                self.assertFails(needle)

    def test_new_status_amendment(self):
        self.w.status_edit["new"] = {"amendment": {"parent_config_sha256": PAR_CFG, "changed_field": "training.early_stopping_patience",
                                                   "from": 4, "to": 8}}
        self.w.write()
        self.assertFails("amendment differs from the configuration")

    def test_new_status_models_split(self):
        self.w.new_status = "INCOMPLETE"
        self.w.completed = NAMES[:20]
        self.w.status_edit["new"] = {"missing": NAMES[21:]}
        self.w.write()
        self.assertFails("the complete and missing models must split the 36 models")

    def test_pass_with_missing_or_incomplete_without(self):
        self.w.status_edit["new"] = {"status": "INCOMPLETE", "passed": False}
        self.w.write()
        self.assertFails("INCOMPLETE without missing models")

    def test_pass_with_missing_models(self):
        self.w.completed = NAMES[1:]
        self.w.status_edit["new"] = {"completed": NAMES[1:], "missing": NAMES[:1]}
        self.w.write()
        self.assertFails("verification_run status.json: PASS with missing models")

    def test_public_sums_file_differs(self):
        p = self.w.base / "runs/new/public/SHA256SUMS"
        p.write_bytes(p.read_bytes() + b"0" * 64 + b"  extra.txt\n")
        self.assertFails("verification_run public SHA256SUMS: SHA-256")

    def test_restricted_sums_file_differs(self):
        p = self.w.base / "runs/parent/restricted/SHA256SUMS"
        p.write_bytes(p.read_bytes() + b"0" * 64 + b"  extra.txt\n")
        self.assertFails("parent_run restricted SHA256SUMS: SHA-256")

    def test_model_missing_in_a_pass_run(self):
        self.w.completed = NAMES[1:]
        self.w.status_edit["new"] = {"completed": NAMES, "missing": []}
        self.w.write()
        self.assertFails("training cards listed, status.json declares 36 complete models")

    def test_card_with_patience_4_in_the_new_run(self):
        self.w.card_edit[("new", "arm_b_fold1_seed43")] = lambda c: (c.update({"early_stopping_patience": 4}), c)[1]
        self.w.write()
        self.assertFails("early_stopping_patience = 4 differs from the recipe of its run (12)")

    def test_card_history_length(self):
        self.w.card_edit[("new", "arm_b_fold1_seed43")] = lambda c: (c["history"]["lr"].append(0.1), c)[1]
        self.w.write()
        self.assertFails("history.lr has 13 values")

    def test_card_restored_epoch_not_argmin(self):
        self.w.card_edit[("parent", "arm_b_fold1_seed43")] = lambda c: (c.update({"restored_epoch": 11}), c)[1]
        self.w.write()
        self.assertFails("restored_epoch 11 is not the epoch of minimum val_loss (12)")

    def test_parent_card_with_patience_12(self):
        self.w.card_edit[("parent", "arm_b_fold1_seed42")] = lambda c: (c.update({"early_stopping_patience": 12}), c)[1]
        self.w.write()
        self.assertFails("early_stopping_patience = 12 differs from the recipe of its run (4)")

    def test_card_replay_not_reproduced(self):
        # genitore B f1 s43 troncato a 6 epoche strettamente decrescenti: con pazienza 4 nessun arresto entro la sesta
        # epoca e storia più corta di 12 -> replay non determinabile, la scheda è rifiutata
        def cut(c):
            for k in c["history"]:
                c["history"][k] = c["history"][k][:6]
            c.update({"epochs_run": 6, "restored_epoch": 6, "best_val_loss": c["history"]["val_loss"][5],
                      "val_acc_at_best": c["history"]["val_accuracy"][5], "restored_by": "early stopping"})
            return c
        self.w.card_edit[("parent", "arm_b_fold1_seed43")] = cut
        self.w.write()
        self.assertFails("the replay of the Keras EarlyStopping (patience 4) gives not determinable")

    def test_card_restored_by(self):
        self.w.card_edit[("new", "arm_b_fold1_seed43")] = lambda c: (c.update({"restored_by": "early stopping"}), c)[1]
        self.w.write()
        self.assertFails("restored_by 'early stopping' differs from the replay")

    def test_dev_prediction_05_inconsistent(self):
        def edit(rows):
            rows[0]["prediction_05"] = "1" if rows[0]["prediction_05"] == "0" else "0"
            return rows
        self.w.dev_edit[("new", "arm_b_fold1_seed43")] = edit
        self.w.write()
        self.assertFails("prediction_05 does not follow")

    def test_dev_accuracy_differs_from_the_card(self):
        def edit(rows):                      # stesso dev, score diversi: l'accuratezza a 0,5 non è più val_acc_at_best
            rows[0]["score_attack"] = repr(0.95)
            rows[0]["prediction_05"] = "1"
            return rows
        self.w.dev_edit[("parent", "arm_b_fold1_seed43")] = edit
        self.w.write()
        self.assertFails("accuracy at 0.5 of the checkpoint")

    def test_dev_differs_from_the_parent(self):
        def edit(rows):
            for r in rows:
                r["file_sha256"] = sha("other " + r["file_sha256"])
            return rows
        for s in SEEDS:                       # tutto il fold 1 di B nel nuovo run: il dev resta comune nel run
            for a in ARMS:
                self.w.dev_edit[("new", f"{a}_fold1_seed{s}")] = edit
        self.w.write()
        self.assertFails("the dev differs from the parent run")

    def test_dev_differs_within_a_fold(self):
        def edit(rows):
            rows[0]["file_sha256"] = sha("other")
            return rows
        self.w.dev_edit[("new", "arm_b_fold1_seed43")] = edit
        self.w.write()
        self.assertFails("the models of the fold do not share the same dev")

    def test_card_unknown_key(self):
        self.w.card_edit[("new", "arm_b_fold1_seed43")] = lambda c: (c.update({"extra": 1}), c)[1]
        self.w.write()
        self.assertFails("keys differ from the expected set")

    def test_card_best_val_loss_differs_from_the_history(self):
        self.w.card_edit[("parent", "arm_b_fold1_seed43")] = lambda c: (c.update({"best_val_loss": c["best_val_loss"] + 1}), c)[1]
        self.w.write()
        self.assertFails("best_val_loss or val_acc_at_best differ from the history at the restored epoch")

    def test_card_weights_not_in_the_sums(self):
        self.w.card_edit[("new", "arm_b_fold1_seed43")] = lambda c: (c.update({"weights_sha256": sha("other weights")}), c)[1]
        self.w.write()
        self.assertFails("weights_file or weights_sha256 differ from the restricted SHA256SUMS")

    def test_card_dev_keys_differ_from_the_rows(self):
        self.w.card_edit[("new", "arm_b_fold1_seed43")] = lambda c: (c.update({"dev_keys_sha256": sha("other keys")}), c)[1]
        self.w.write()
        self.assertFails("the sample_id list differs from dev_keys_sha256 of the card")

    def test_threshold_checkpoint_differs(self):
        self.w.thr_edit[("new", "arm_b_fold1_seed43")] = lambda t: (t.update({"checkpoint_sha256": sha("x")}), t)[1]
        self.w.write()
        self.assertFails("_threshold.json: checkpoint_sha256 differs from the weights of the card")

    def test_threshold_frozen_before_the_weights(self):
        self.w.thr_edit[("parent", "arm_b_fold1_seed43")] = lambda t: (t.update({"frozen_at": "2026-11-09T07:59:59+0100"}), t)[1]
        self.w.write()
        self.assertFails("saved, frozen_at and test_scored_after_threshold are not in order")

    def test_threshold_dev_scores_sha256_differs(self):
        self.w.thr_edit[("new", "arm_b_fold1_seed43")] = lambda t: (t.update({"dev_scores_sha256": sha("x")}), t)[1]
        self.w.write()
        self.assertFails("the scores differ from dev_scores_sha256 of arm_b_fold1_seed43_threshold.json")

    def test_dev_row_checkpoint_differs(self):
        def edit(rows):
            rows[0]["checkpoint_sha256"] = sha("x")
            return rows
        self.w.dev_edit[("new", "arm_b_fold1_seed43")] = edit
        self.w.write()
        self.assertFails("_dev_scores.csv: checkpoint_sha256 differs from the weights of the card")

    def test_dev_row_threshold_differs(self):
        def edit(rows):
            rows[0]["threshold_dev"] = repr(0.4)
            return rows
        self.w.dev_edit[("new", "arm_b_fold1_seed43")] = edit
        self.w.write()
        self.assertFails("threshold_dev or threshold_sha256 differ from the frozen threshold")

    def test_dev_prediction_dev_inconsistent(self):
        def edit(rows):
            rows[0]["prediction_dev"] = "0" if float(rows[0]["score_attack"]) > 0.5 else "1"
            return rows
        self.w.dev_edit[("parent", "arm_b_fold1_seed43")] = edit
        self.w.write()
        self.assertFails("prediction_dev does not follow")

    def test_dev_not_balanced(self):
        def edit(rows):                      # un bona fide etichettato attacco: 3 contro 5
            rows[0]["label"] = "1"
            return rows
        self.w.dev_edit[("new", "arm_b_fold1_seed43")] = edit
        self.w.write()
        self.assertFails("the dev is not balanced (3 bona fide, 5 attacks)")

    def test_fitting_set_differs_from_the_parent(self):
        self.w.card_edit[("new", "arm_b_fold1_seed43")] = lambda c: (c.update({"fitting_keys_sha256": sha("x")}), c)[1]
        self.w.write()
        self.assertFails("the fitting set differs from the parent run")

    def test_listed_file_hash(self):
        self.w.write()
        p = self.w.base / "runs/new/restricted/arm_b_fold1_seed43_card.json"
        p.write_bytes(p.read_bytes() + b" ")
        self.assertFails("SHA-256")

    def test_symlink_input_refused(self):
        p = self.w.base / "runs/parent/restricted/arm_b_fold1_seed43_threshold.json"
        q = self.w.tmp / "thr.json"
        q.write_bytes(p.read_bytes())
        p.unlink()
        p.symlink_to(q)
        self.assertFails("is a symbolic link")

    def test_ten_differs_from_the_parent(self):
        self.w.cfg["original_degenerate"] = sorted(TEN[:9] + ["arm_b_fold1_seed43"])
        self.w.write()
        self.assertFails("original_degenerate differs from the degenerate models of the parent run")

    def test_without_recovery_differs_from_the_parent(self):
        self.w.cfg["models_without_recovery_parent"] = []
        self.w.write()
        self.assertFails("models_without_recovery_parent differs from the parent run")

    def test_public_aggregate_hash(self):
        self.w.write()
        p = self.w.base / "runs/new/public/aggregate.json"
        p.write_bytes(p.read_bytes() + b" ")
        self.assertFails("public aggregate.json: SHA-256 differs from the public SHA256SUMS")

    def test_public_aggregate_identity(self):
        self.w.agg_edit["new"] = {"status": "INCOMPLETE"}
        self.w.write()
        self.assertFails("public aggregate.json: status differs from status.json")

    def test_public_aggregate_structure(self):
        for edit, needle in (({"strata": []}, "strata must be a list"),
                             ({"primary_verdicts": {"auroc": {"verdict": "inconclusive", "direction": None, "reason": "r"}}},
                              "primary_verdicts must hold exactly"),
                             ({"primary_verdicts": {"auroc": {"verdict": "maybe", "direction": None, "reason": "r"},
                                                    "acer_dev": {"verdict": "inconclusive", "direction": None, "reason": "r"}}},
                              "is not a verdict"),
                             ({"primary_verdicts": {"auroc": {"verdict": "consistent", "direction": None, "reason": "r"},
                                                    "acer_dev": {"verdict": "inconclusive", "direction": None, "reason": "r"}}},
                              "has a direction only when consistent")):
            with self.subTest(needle=needle):
                self.w.agg_edit["new"] = edit
                self.w.write()
                self.assertFails(needle)

    def test_public_aggregate_comparisons(self):
        agg = public_aggregate(NEW_ID, "PASS", NEW_CFG, True)
        c = copy.deepcopy(agg["comparisons"])
        c["b_prime_minus_b"]["minuend"] = "arm_a_star"
        self.w.agg_edit["new"] = {"comparisons": c}
        self.w.write()
        self.assertFails("comparisons.b_prime_minus_b must be arm_b_prime - arm_b")
        c = copy.deepcopy(agg["comparisons"])
        del c["a_star_minus_b"]["verdicts"]["02"]["bpcer_05"]
        self.w.agg_edit["new"] = {"comparisons": c}
        self.w.write()
        self.assertFails("has no verdict or median for stratum 02, metric bpcer_05")
        c = copy.deepcopy(agg["comparisons"])
        c["a_star_minus_b"]["pooled_summary_over_seeds"]["02"]["auroc"]["median"] = "0.1"
        self.w.agg_edit["new"] = {"comparisons": c}
        self.w.write()
        self.assertFails("is not a number or null")

    def test_run_folder_with_status_refused(self):
        self.assertEqual(self.run_cmd()[0], 0)
        run = self.w.out / RUN_ID
        before = snapshot(run)
        rc, so, se = self.run_cmd()
        self.assertEqual(rc, 1)
        self.assertIn(f"the run folder {RUN_ID} already holds status.json: every attempt keeps its outcome", se)
        self.assertEqual(snapshot(run), before)

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
        rc, so, se = self.run_cmd(out=self.w.base / "runs" / "new" / "x")
        self.assertEqual(rc, 1)
        self.assertIn("refusing the run folder", se)

    def test_failure_after_the_checks_leaves_no_public(self):
        def boom(kind, data):
            raise RuntimeError("plot failed")
        rc, so, se = self.run_cmd(PLOTTER=boom)
        self.assertEqual(rc, 1, so + se)
        self.assertIn("unexpected error RuntimeError: plot failed", se)
        st = json.loads((self.w.out / RUN_ID / "status.json").read_text())
        self.assertEqual((st["status"], st["outputs"]), ("FAIL", "removed"))
        pub = self.w.out / RUN_ID / "public"
        self.assertEqual(os.listdir(pub) if pub.exists() else [], [])

    def test_failure_after_the_public_write_removes_it(self):
        calls = []
        real = PV.sha256_file

        def once(p):
            if not calls and Path(p).name == "run.log":
                calls.append(p)
                raise OSError("disk error after the public write")
            return real(p)
        rc, so, se = self.run_cmd(sha256_file=once)
        self.assertEqual(rc, 1, so + se)
        st = json.loads((self.w.out / RUN_ID / "status.json").read_text())
        self.assertEqual((st["status"], st["outputs"]), ("FAIL", "removed"))
        self.assertEqual(os.listdir(self.w.out / RUN_ID / "public"), [])

    def test_authorisation_and_dirty_tree(self):
        with patched():
            rc, so, se = run_main(self.w.argv("run"))
        self.assertEqual(rc, 1)
        self.assertIn("--i-am-authorized", se)
        rc, so, se = self.run_cmd(REPO_STATE=lambda r: {"commit": "abcdef0", "dirty": True})
        self.assertEqual(rc, 0, so + se)
        self.assertEqual(json.loads((self.w.out / f"{RUN_ID}-dev" / "public" / "aggregate.json").read_text())["dirty"], True)

    def test_run_id_format(self):
        self.assertTrue(PV.RUN_ID_RE.match("20261109-PATVER-abcdef0-dev"))
        self.assertFalse(PV.RUN_ID_RE.match("20261109-DEGEN-abcdef0"))
        with self.assertRaisesRegex(PV.Failure, "--date must be YYYYMMDD"):
            PV.run_id_for("2026119", STATE)
        with self.assertRaisesRegex(PV.Failure, "does not follow YYYYMMDD-PATVER-<commit>"):
            PV.run_id_for(DATE, {"commit": "XYZ", "dirty": False})


# ============================================================================ controlli uno per uno (audit dei rifiuti)

def put(path: str, value):
    """Modifica della configurazione: il campo con il percorso a punti prende il valore."""
    def edit(cfg):
        *head, last = path.split(".")
        d = cfg
        for k in head:
            d = d[k]
        d[last] = value
    return edit


def both(*edits):
    def edit(cfg):
        for e in edits:
            e(cfg)
    return edit


def setv(**kv):
    """Modifica di una scheda o di una soglia: i campi prendono i valori."""
    return lambda d: (d.update(kv), d)[1]


def hist(key: str, i: int, value):
    """Modifica della storia di una scheda: history[key][i] = value."""
    return lambda c: (c["history"][key].__setitem__(i, value), c)[1]


def row(i: int, **kv):
    """Modifica di una riga del dev: i campi della riga i prendono i valori."""
    return lambda rows: (rows[i].update(kv), rows)[1]


class TestConfigChecks(WorldCase):
    """Ogni controllo di load_config fatto scattare da solo, ogni volta dalla configurazione intatta del mondo."""

    def cases(self, cases):
        for i, (needle, edit) in enumerate(cases):
            with self.subTest(case=i, needle=needle):
                cfg = copy.deepcopy(self.w.cfg)
                edit(cfg)
                self.w.cfg_path.write_bytes(jbytes(cfg))
                self.assertFails(needle)

    def test_schema_texts_and_closed_sets(self):
        self.cases([("schema must be 'nuaa-patience-verify-config/1'", put("schema", "x")),
                    ("configuration: task must be a non-empty text", put("task", "   ")),
                    ("configuration: task must be a non-empty text", put("task", 1)),
                    ("configuration: request needs exactly card, date, text", put("request.extra", "x")),
                    ("configuration: request needs exactly card, date, text",
                     put("request", ["card", "date", "text"]))])

    def test_inputs_lists_and_counts(self):
        folds = "configuration: inputs.folds must be a non-empty list of distinct integers >= 1"
        rows = "configuration: inputs.dev_rows must be a positive even integer"
        self.cases([("configuration: inputs.arms must list", put("inputs.arms", ["arm_b", "arm_a_star", "arm_c"])),
                    ("configuration: inputs.arms must list", put("inputs.arms", {a: 1 for a in ARMS})),
                    (folds, put("inputs.folds", 5)),
                    (folds, put("inputs.folds", [1, 1, 2, 3])), (folds, put("inputs.folds", [0, 1, 2, 3])),
                    (folds, put("inputs.folds", [])),
                    ("inputs.seeds must be a non-empty list of distinct integers >= 0",
                     put("inputs.seeds", [42, 43, 44.0])),
                    ("configuration: inputs.models must be 36", put("inputs.models", 35)),
                    ("configuration: inputs.models must be 36", put("inputs.models", 36.0)),
                    (rows, put("inputs.dev_rows", 7)), (rows, put("inputs.dev_rows", 0)),
                    (rows, put("inputs.dev_rows", 8.0))])

    def test_runs_identity_and_anchors(self):
        v, p = "inputs.verification_run.", "inputs.parent_run."
        distinct = "configuration: the two runs must have distinct run ids, labels and folders"
        patience = "configuration: inputs.parent_run.early_stopping_patience must be a positive integer"
        self.cases([("inputs.verification_run.commit must be the commit of its run id", put(v + "commit", 1)),
                    ("inputs.parent_run.config_sha256 must be a SHA-256",
                     both(put(p + "config_sha256", "x"), put(v + "amendment.parent_config_sha256", "x"))),
                    (patience, put(p + "early_stopping_patience", 4.0)),
                    (patience, put(p + "early_stopping_patience", 0)),
                    ("configuration: inputs.parent_run.status_sha256 must be a SHA-256",
                     put(p + "status_sha256", "ABC")),
                    ("the anchors of inputs.verification_run must be SHA-256 values",
                     put(v + "public_sums_sha256", "ABC")),
                    ("inputs.verification_run.configuration_name must be 'patience12_verification'",
                     put(v + "configuration_name", "other")),
                    (distinct, put(v + "label", "parent")), (distinct, put(v + "folder", "runs/parent")),
                    (distinct, both(put(v + "run_id", PAR_ID), put(v + "commit", "0000001")))])

    def test_recipe_values(self):
        rule = "configuration: the recipe of the runs is monitor val_loss, min_delta 0, restore_best_weights true"
        self.cases([("configuration: recipe.batch must be a positive integer", put("recipe.batch", 0)),
                    ("configuration: recipe.batch must be a positive integer", put("recipe.batch", 16.0)),
                    ("configuration: recipe.lr must be positive", put("recipe.lr", 0)),
                    ("configuration: recipe.lr must be positive", put("recipe.lr", float("nan"))),
                    (rule, put("recipe.min_delta", 0.1)), (rule, put("recipe.monitor", "val_accuracy")),
                    (rule, put("recipe.restore_best_weights", False)),
                    ("the patience of the verification run must equal recipe.max_epochs",
                     put("recipe.max_epochs", 13))])

    def test_criteria_values(self):
        c = "criteria."
        dg = "configuration: criteria.degenerate needs val_acc_at_best in [0, 1] and a tolerance in (0, 1e-3)"
        self.cases([("criteria.degenerate.restored_epochs must be a non-empty list of distinct integers >= 1",
                     put(c + "degenerate.restored_epochs", [0, 1])),
                    (dg, both(put(c + "degenerate.val_acc_at_best", 1.5), put(c + "saturation.val_accuracy", 1.5))),
                    (dg, both(put(c + "degenerate.tolerance", 0.01), put(c + "saturation.tolerance", 0.01))),
                    ("configuration: criteria.saturation must use the value and tolerance of criteria.degenerate",
                     put(c + "saturation.val_accuracy", 0.4)),
                    ("configuration: criteria.recovery.val_accuracy_min must be a positive number",
                     put(c + "recovery.val_accuracy_min", 0)),
                    ("configuration: criteria.checkpoint_agreement_tolerance must be in (0, 1e-3)",
                     put(c + "checkpoint_agreement_tolerance", 0.1)),
                    ("configuration: criteria.shared_history_difference must be 0",
                     put(c + "shared_history_difference", 0.0))])

    def test_lists_of_the_parent(self):
        ten = "original_degenerate must list the ten degenerate models of the parent run, sorted, once each"
        nr = "models_without_recovery_parent must list models of the run outside the ten, sorted, once each"
        self.cases([(ten, put("original_degenerate", sorted(TEN[:9] + [TEN[0]]))),
                    (ten, put("original_degenerate", TEN[::-1])), (ten, put("original_degenerate", 10)),
                    (ten, put("original_degenerate", sorted(TEN + [TEN[0]]))),     # 11 voci, 10 distinte
                    (nr, put("models_without_recovery_parent", 1)),
                    (nr, put("models_without_recovery_parent", [NOREC, NOREC])),
                    (nr, put("models_without_recovery_parent", ["arm_c_fold1_seed42"])),
                    (nr, put("models_without_recovery_parent", [NOREC, "arm_b_fold1_seed43"]))])

    def test_time_values(self):
        tm = "configuration: time.estimate_seconds must be positive and not above time.cap_seconds"
        self.cases([(tm, put("time.estimate_seconds", 0)), (tm, put("time.cap_seconds", float("inf")))])

    def test_time_estimate_equal_to_the_cap_accepted(self):
        cfg = copy.deepcopy(self.w.cfg)
        cfg["time"].update({"estimate_seconds": 21600, "cap_seconds": 21600})
        self.w.cfg_path.write_bytes(jbytes(cfg))
        rc, out, err = self.check()
        self.assertEqual(rc, 0, out + err)


class TestInputChecks(WorldCase):
    """Ogni controllo di schede, soglie, dev, status e file fatto scattare da solo, con le ancore ricalcolate (fallisce
    la regola, non il checksum)."""
    M = "arm_b_fold1_seed43"            # 12 epoche in entrambi i run, ripristino 12, dev con accuratezza 1

    def cases(self, cases, run="new"):
        for i, (needle, kind, edit) in enumerate(cases):
            with self.subTest(case=i, needle=needle):
                self.w.card_edit, self.w.thr_edit, self.w.dev_edit = {}, {}, {}
                {"card": self.w.card_edit, "thr": self.w.thr_edit, "dev": self.w.dev_edit}[kind][(run, self.M)] = edit
                self.w.write()
                self.assertFails(needle)

    def restricted(self, run: str, fn: str) -> Path:
        return self.w.base / "runs" / run / "restricted" / fn

    def test_card_fields(self):
        m = f"card {self.M}"
        names = "arm, fold or train_seed differ from the file name"
        n_dev = "n_dev must be an integer >= 2"
        epochs = "epochs_run must be an integer between 1 and max_epochs"
        self.cases([(f"{m}: arch is not LivenessNet", "card", setv(arch="Other")),
                    (names, "card", setv(fold=2)), (names, "card", setv(train_seed=42)),
                    (names, "card", setv(arm="arm_a_star")),
                    (f"{m}: fold and train_seed must be integers (not booleans)", "card", setv(fold=True)),   # fold 1
                    (f"{m}: fold and train_seed must be integers (not booleans)", "card", setv(fold=1.0)),
                    ("batch = 32 differs from the recipe of its run (16)", "card", setv(batch=32)),
                    (n_dev, "card", setv(n_dev=8.0)), (n_dev, "card", setv(n_dev=1)),
                    (f"{m}: train_seconds must be a positive number", "card", setv(train_seconds=0)),
                    (epochs, "card", setv(epochs_run=12.0)), (epochs, "card", setv(epochs_run=13)),
                    (f"{m} saved: 'x' is not a time", "card", setv(saved="x")),
                    (f"{m}: dev_keys_sha256 and fitting_keys_sha256 must be SHA-256 values", "card",
                     setv(dev_keys_sha256="x")),
                    (f"{m} test_scored_after_threshold: 'x' is not a time", "card",
                     setv(test_scored_after_threshold="x"))])

    def test_card_history(self):
        acc, loss = "accuracy outside [0, 1] in the history", "negative loss in the history"
        self.cases([("history must have exactly the keys", "card",
                     lambda c: (c["history"].update({"x": [0.1] * 12}), c)[1]),
                    ("history.loss holds a value that is not a finite number", "card", hist("loss", 0, float("nan"))),
                    (acc, "card", hist("accuracy", 0, -0.1)), (acc, "card", hist("val_accuracy", 2, 1.5)),
                    (loss, "card", hist("loss", 0, -0.1)), (loss, "card", hist("val_loss", 0, -1.0)),
                    ("non-positive lr in the history", "card", hist("lr", 0, 0.0)),
                    ("restored_epoch 12.0 is not the epoch of minimum val_loss", "card", setv(restored_epoch=12.0)),
                    ("best_val_loss or val_acc_at_best differ from the history at the restored epoch", "card",
                     setv(val_acc_at_best=0.875)),
                    ("weights_file or weights_sha256 differ from the restricted SHA256SUMS", "card",
                     setv(weights_file="arm_b_fold1_seed42_LivenessNet.keras",
                          weights_sha256=sha("weights new arm_b_fold1_seed42")))])

    def test_parent_card_fitting(self):
        n_fit = "n_fitting must be a positive integer"
        self.cases([(n_fit, "card", setv(n_fitting=0)), (n_fit, "card", setv(n_fitting=150.0)),
                    ("dev_keys_sha256 and fitting_keys_sha256 must be SHA-256 values", "card",
                     setv(fitting_keys_sha256="x"))],
                   run="parent")

    def test_fitting_size_differs_from_the_parent(self):
        self.w.card_edit[("new", self.M)] = setv(n_fitting=151)
        self.w.write()
        self.assertFails(f"verification_run {self.M}: the fitting set differs from the parent run")

    def test_card_replay_stops_before_epochs_run(self):
        # genitore B f1 s43 con val_loss 5,6,7,8,9,10 (6 epoche): con pazienza 4 il replay si ferma alla quinta epoca
        # (migliore 1, wait 4 all'epoca 5), la scheda registra 6 epoche e ripristino 1 -> rifiutata
        self.w.parent_hist[self.M] = ([5, 6, 7, 8, 9, 10], [0.5] * 6)
        self.w.cont[self.M] = ([4, 3, 2, 1, 0.9, 0.8], [0.5] * 6)
        self.w.write()
        self.assertFails("the replay of the Keras EarlyStopping (patience 4) gives stop 5, restored 1, "
                         "the card records epochs_run 6")

    def test_card_recipe_value_true_is_not_1(self):
        # ricetta con batch 1: una scheda con batch true (uguale a 1 per Python) è rifiutata
        self.w.cfg["recipe"]["batch"] = 1
        for run in ("parent", "new"):
            for n in NAMES:
                self.w.card_edit[(run, n)] = setv(batch=1)
        self.w.card_edit[("new", self.M)] = setv(batch=True)
        self.w.write()
        self.assertFails("batch = True differs from the recipe of its run (1)")

    def test_threshold_fields(self):
        t = f"{self.M}_threshold.json"
        names = f"{t}: arm, fold or train_seed differ from the card"
        self.cases([(names, "thr", setv(arm="arm_a_star")), (names, "thr", setv(fold=2)),
                    (names, "thr", setv(train_seed=42)),
                    (f"{t}: fold and train_seed must be integers (not booleans)", "thr", setv(fold=True)),   # fold 1
                    (f"{t}: value is not a finite number", "thr", setv(value="0.5")),
                    (f"{t}: dev_scores_sha256 must be a SHA-256", "thr", setv(dev_scores_sha256="x")),
                    (f"{t} frozen_at: 'x' is not a time", "thr", setv(frozen_at="x")),
                    (f"{t}: saved, frozen_at and test_scored_after_threshold are not in order", "thr",
                     setv(frozen_at="2026-11-09T08:00:30+0100"))])

    def test_dev_rows(self):
        d = f"{self.M}_dev_scores.csv"
        other = f"{d}: a row carries another arm, fold or seed"
        dup = f"{d}: a sample_id or a file_sha256 appears in more than one row"
        self.cases([(f"{d}: 6 rows; the configuration expects 8", "dev", lambda r: r[1:4] + r[5:]),
                    (f"{d}: n_dev of the card is 6; the configuration expects 8", "card", setv(n_dev=6)),
                    (f"{d}: a row carries another role", "dev", row(0, role="test")),
                    (other, "dev", row(0, fold="2")), (other, "dev", row(0, train_seed="42")),
                    (other, "dev", row(0, arm="arm_a_star")),
                    (f"{d}: label out of range", "dev", row(0, label="2")),
                    (f"{d}: score_attack outside [0, 1]", "dev", row(0, score_attack="1.5", prediction_05="1")),
                    (f"{d}: threshold_dev or threshold_sha256 differ", "dev", row(0, threshold_sha256=sha("x"))),
                    (dup, "dev", lambda r: (r[1].update({"sample_id": r[0]["sample_id"]}), r)[1]),
                    (dup, "dev", lambda r: (r[1].update({"file_sha256": r[0]["file_sha256"]}), r)[1])])

    def test_dev_score_not_a_number(self):
        p = self.restricted("new", f"{self.M}_dev_scores.csv")
        rows = list(csv.DictReader(io.StringIO(p.read_text())))
        rows[0]["score_attack"] = "abc"
        buf = io.StringIO(newline="")
        w = csv.DictWriter(buf, fieldnames=list(PV.DEV_COLUMNS), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        self.w.patch_files("new", {p.name: buf.getvalue().encode()})
        self.assertFails(f"{p.name}: score_attack is not a number")

    def test_missing_and_irregular_files(self):
        p = self.restricted("parent", f"{self.M}_threshold.json")
        p.unlink()
        self.assertFails(f"parent_run {self.M}_threshold.json {self.M}_threshold.json not found")
        p.mkdir()
        self.assertFails(f"{self.M}_threshold.json is not a regular file")

    def test_malformed_json_and_csv(self):
        card, dev = f"{self.M}_card.json", f"{self.M}_dev_scores.csv"
        fields = f"{dev}: rows with a wrong number of fields"

        def lines(edit):
            def f():
                ls = self.restricted("new", dev).read_text().split("\n")
                edit(ls)
                return "\n".join(ls).encode()
            return f
        for i, (needle, fn, data) in enumerate([
                (f"{card} is not valid JSON", card, lambda: b"{"),
                (f"{card} is not a JSON object", card, lambda: b"[]"),
                (f"{dev} is not UTF-8", dev, lambda: b"\xff\xfe\x00"),
                (f"{dev}: columns", dev, lines(lambda ls: ls.__setitem__(0, ls[0].replace(",threshold_sha256", "")))),
                (fields, dev, lines(lambda ls: ls.__setitem__(1, ls[1] + ",x"))),
                (fields, dev, lines(lambda ls: ls.__setitem__(1, ls[1].rsplit(",", 1)[0])))]):
            with self.subTest(case=i, needle=needle):
                self.w.write()
                self.w.patch_files("new", {fn: data()})
                self.assertFails(needle)

    def test_malformed_sums(self):
        def sums(run, where, edit):
            p = self.w.base / "runs" / run / where / "SHA256SUMS"
            return lambda: edit(p.read_bytes().decode().splitlines(True))
        thr = f"{self.M}_threshold.json"
        for i, (needle, run, where, data) in enumerate([
                ("verification_run restricted SHA256SUMS: malformed line", "new", "restricted",
                 sums("new", "restricted", lambda ls: "".join(ls) + "not a line\n")),
                ("restricted SHA256SUMS: arm_a_star_fold1_seed42_LivenessNet.keras listed twice", "new", "restricted",
                 sums("new", "restricted", lambda ls: "".join(ls + ls[:1]))),
                (f"verification_run restricted SHA256SUMS: {thr} not listed", "new", "restricted",
                 sums("new", "restricted", lambda ls: "".join(x for x in ls if not x.endswith(f"  {thr}\n")))),
                ("parent_run public SHA256SUMS: aggregate.json not listed", "parent", "public",
                 sums("parent", "public", lambda ls: "".join(x for x in ls if not x.endswith("  aggregate.json\n"))))]):
            with self.subTest(case=i, needle=needle):
                self.w.write()
                self.w.patch_files(run, {}, where=where, sums=data().encode())
                self.assertFails(needle)

    def test_status_passed_and_wall_seconds(self):
        wall = "verification_run status.json: wall_seconds must be a positive number"
        passed = "verification_run status.json: status 'PASS' (passed False)"
        split = "verification_run status.json: the complete and missing models must split the 36 models"
        for i, (needle, edit) in enumerate([(passed, {"passed": False}), (wall, {"wall_seconds": 0}),
                                            (wall, {"wall_seconds": float("inf")}),
                                            ("status 'RUNNING' (passed False) is not one of PASS, INCOMPLETE, FAIL",
                                             {"status": "RUNNING", "passed": False}),
                                            ("verification_run status.json: amendment differs from the configuration",
                                             {"amendment": "4 -> 12"}),
                                            (split, {"completed": {n: 1 for n in NAMES}}), (split, {"missing": None})]):
            with self.subTest(case=i, needle=needle):
                self.w.status_edit["new"] = edit
                self.w.write()
                self.assertFails(needle)

    def test_incomplete_run_without_complete_models(self):
        self.w.new_status = "INCOMPLETE"
        self.w.completed = []
        self.w.write()
        self.assertFails("must split the 36 models of the configuration, with at least one complete")

    def test_public_aggregate_structure_details(self):
        agg = public_aggregate(NEW_ID, "PASS", NEW_CFG, True)
        strata = "public aggregate.json: strata must be a list of distinct names"
        reason = {"verdict": "inconclusive", "direction": None, "reason": None}
        for i, (needle, edit) in enumerate([
                ("comparisons must hold exactly", {"comparisons": {k: v for k, v in agg["comparisons"].items()
                                                                   if k != "a_star_minus_b_prime"}}),
                (strata, {"strata": ["01", "02", "03", 4]}), (strata, {"strata": ["01", "01", "02", "03", "all"]}),
                ("primary_verdicts.acer_dev is not a verdict",
                 {"primary_verdicts": {**agg["primary_verdicts"], "acer_dev": reason}}),
                ("primary_verdicts.auroc is not a verdict",
                 {"primary_verdicts": {**agg["primary_verdicts"], "auroc": "x"}}),
                (strata, {"strata": 5}),
                ("primary_verdicts must hold exactly auroc, acer_dev", {"primary_verdicts": ["acer_dev", "auroc"]}),
                ("comparisons must hold exactly", {"comparisons": list(PV.COMPARISONS)}),
                ("comparisons.a_star_minus_b must be arm_a_star - arm_b",
                 {"comparisons": {**agg["comparisons"], "a_star_minus_b": []}})]):
            with self.subTest(case=i, needle=needle):
                self.w.agg_edit["new"] = edit
                self.w.write()
                self.assertFails(needle)


class TestOutputChecks(WorldCase):
    """Rifiuti dal lato delle uscite: scansione del pubblico, figure, cartella del run, log, stato git."""

    def test_image_keys_refused(self):
        self.w.cfg["reading_rule"]["c"] = "rule c, e.g. real/0001_x"
        self.w.write()
        self.assertRunFailsClean("refused: image keys")

    def test_forbidden_text_refused(self):
        self.w.cfg["reading_rule"]["c"] = "rule c from /ho" "me/someone"
        self.w.write()
        self.assertRunFailsClean("refused: absolute path")

    def test_empty_figure_refused(self):
        for i, png in enumerate((b"", "png as text")):
            with self.subTest(png=png):
                out = self.w.tmp / f"o{i}"
                rc, so, se = self.run_cmd(out=out, PLOTTER=lambda kind, data: png)
                self.assertEqual(rc, 1, so + se)
                self.assertIn("figure figure_ten.png: the plotter returned no PNG bytes", se)
                st = json.loads((out / RUN_ID / "status.json").read_text())
                self.assertEqual((st["status"], st["outputs"]), ("FAIL", "removed"))

    def test_output_root_symlink_refused(self):
        real = self.w.tmp / "real_out"
        real.mkdir()
        link = self.w.tmp / "out_link"
        link.symlink_to(real)
        rc, so, se = self.run_cmd(out=link)
        self.assertEqual(rc, 1, so + se)
        self.assertIn("refusing the run folder: the output root is a symbolic link", se)
        self.assertEqual(os.listdir(real), [])

    def test_run_folder_is_a_file(self):
        self.w.out.mkdir(parents=True)
        (self.w.out / RUN_ID).write_text("x")
        rc, so, se = self.run_cmd()
        self.assertEqual(rc, 1, so + se)
        self.assertIn(f"refusing the run folder: {RUN_ID} exists and is not a folder", se)
        self.assertEqual((self.w.out / RUN_ID).read_text(), "x")

    def test_run_log_symlink_refused(self):
        run = self.w.out / RUN_ID
        run.mkdir(parents=True)
        target = self.w.tmp / "elsewhere.log"
        target.write_text("keep")
        (run / "run.log").symlink_to(target)
        rc, so, se = self.run_cmd()
        self.assertEqual(rc, 1, so + se)
        self.assertIn("run.log is a link or not a regular file: refused", se)
        self.assertEqual(target.read_text(), "keep")
        self.assertEqual(json.loads((run / "status.json").read_text())["status"], "FAIL")

    def test_input_inside_the_run_folder(self):
        run = self.w.out / RUN_ID
        run.mkdir(parents=True)
        p = run / self.w.cfg_path.name
        p.write_bytes(self.w.cfg_path.read_bytes())
        self.w.cfg_path = p
        rc, so, se = self.run_cmd()
        self.assertEqual(rc, 1, so + se)
        self.assertIn(f"the configuration {p.name} is inside the run folder, or contains it", se)
        self.assertFalse((run / "status.json").exists())

    def test_within_cap_at_the_cap(self):
        for i, (wall, within) in enumerate(((21600.0, True), (21600.1, False))):
            with self.subTest(wall=wall):
                self.w.status_edit["new"] = {"wall_seconds": wall}
                self.w.write()
                self.assertEqual(self.run_ok(out=self.w.tmp / f"o{i}")["sections"]["e"]["within_cap"], within)

    def test_git_state_not_readable(self):
        def no_repo(args, **kw):
            return PV.subprocess.CompletedProcess(args, 128, "", "fatal: not a git repository")

        def no_git(args, **kw):
            raise FileNotFoundError("git")
        with mock.patch.object(PV.subprocess, "run", no_repo):
            with self.assertRaisesRegex(PV.Failure, "git state not readable"):
                PV.repo_state(self.w.tmp)
        with mock.patch.object(PV.subprocess, "run", no_git):
            with self.assertRaisesRegex(PV.Failure, "git not available: FileNotFoundError"):
                PV.repo_state(self.w.tmp)


# ============================================================================ configurazione consegnata e ingressi veri

class TestDelivered(unittest.TestCase):
    def delivered(self):
        p = ROOT / PV.DEFAULT_CONFIG                    # accanto al codice (la copia delle mutazioni la porta con sé)
        if not p.is_file():
            skip_missing(self, "delivered configuration")
        return PV.load_config(p)

    def test_delivered_configuration_structure(self):
        cfg, _ = self.delivered()
        v, p = cfg["inputs"]["verification_run"], cfg["inputs"]["parent_run"]
        self.assertEqual((p["run_id"], v["run_id"], v["config_sha256"], v["commit"], v["configuration_name"]),
                         ("20261106-SESSABLRUN-d999458", "20261109-SESSABLRUN-6a50265",
                          "ca039df5d3114f9cb983c7256330aae671de54ae34d949ad4e9b9c80beb6aa43", "6a50265", "patience12_verification"))
        self.assertEqual(cfg["original_degenerate"], TEN)
        self.assertEqual(cfg["models_without_recovery_parent"], [NOREC])
        self.assertEqual(cfg["time"]["estimate_seconds"], 13090)

    def test_delivered_configuration(self):
        cfg, h = self.delivered()
        if not PV.anchors_filled(cfg):
            skip_missing(self, "anchors null")
        self.assertEqual(h, PV.CONFIG_SHA256)

    def test_criteria_equal_to_the_previous_package(self):
        cfg, _ = self.delivered()
        q = ROOT / "delivery/degenerate/nuaa_degenerate_trainings.json"
        if not q.is_file():
            skip_missing(self, "previous package configuration")
        prev = json.loads(q.read_text())
        for k in ("degenerate", "saturation", "checkpoint_agreement_tolerance"):
            self.assertEqual(cfg["criteria"][k], prev["criteria"][k], k)
        self.assertEqual({k: cfg["criteria"]["recovery"][k] for k in ("val_accuracy_min", "val_loss_max", "text")},
                         {k: prev["criteria"]["recovery"][k] for k in ("val_accuracy_min", "val_loss_max", "text")})
        self.assertEqual(cfg["inputs"]["parent_run"]["status_sha256"], prev["inputs"]["ablation_run"]["status_sha256"])
        self.assertEqual(cfg["inputs"]["parent_run"]["restricted_sums_sha256"], prev["inputs"]["ablation_run"]["restricted_sums_sha256"])
        self.assertEqual(cfg["inputs"]["parent_run"]["public_sums_sha256"], prev["inputs"]["ablation_run"]["public_sums_sha256"])


class TestRealInputs(unittest.TestCase):
    def test_real_inputs_read_only(self):
        if not REAL:
            skip_missing(self, "real not requested")
        p = REAL_ROOT / PV.DEFAULT_CONFIG
        if not p.is_file():
            skip_missing(self, "delivered configuration")
        cfg, _ = PV.load_config(p)
        if not (REAL_ROOT / cfg["inputs"]["parent_run"]["folder"] / "status.json").is_file():
            skip_missing(self, "real inputs")
        folders = [REAL_ROOT / cfg["inputs"][g]["folder"] for g in PV.RUNS]
        before = snapshot(*folders) if PV.anchors_filled(cfg) else snapshot(folders[0])
        rc, out, err = run_main(["check-inputs", "--config", str(p), "--input-root", str(REAL_ROOT)])
        self.assertEqual(rc, 0, out + err)
        inp = PV.load_inputs(cfg, REAL_ROOT, need_verification=False)
        self.assertEqual(len(inp.parent.models), 36)
        self.assertEqual((snapshot(*folders) if PV.anchors_filled(cfg) else snapshot(folders[0])), before)


class ZZRepositoryUntouched(unittest.TestCase):
    def test_declared_skips_only(self):
        self.assertEqual(sorted(MISSING_INPUT), ["anchors null", "delivered configuration", "previous package configuration",
                                                 "real inputs", "real not requested"])


# la cartella dei run di training non è nemmeno elencata (un training può essere in corso e scrivere lì)
SKIP = (ROOT / "results" / "sessionablationrun",)


def main() -> int:
    before = snapshot(ROOT / "results", ROOT / "export", ROOT / "delivery", skip=SKIP)
    argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a not in ("--verbose", "--real")] + (["-v"] if "--verbose" in sys.argv else [])
    result = unittest.main(argv=argv, exit=False, verbosity=1).result
    after = snapshot(ROOT / "results", ROOT / "export", ROOT / "delivery", skip=SKIP)
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
