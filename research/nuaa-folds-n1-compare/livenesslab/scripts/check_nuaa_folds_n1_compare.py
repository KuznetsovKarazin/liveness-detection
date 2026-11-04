"""
Test di scripts/nuaa_folds_n1_compare.py (confronto descrittivo run dei fold contro N1 sui 3.000 frame di cnn_test),
offline e deterministici.

Mondo sintetico in cartelle temporanee, con identificativi e struttura diversi da NUAA: 16 soggetti 1101–1116; un
cnn_test di 64 frame della sessione 03 (2 bona fide e 2 attacchi per soggetto, 4 bona fide per 1113 e 1114, nessun
bona fide per 1115 e 1116: 32 + 32); manifest di training di N1 con i soggetti 1101–1109 (sessioni 01–02); configurazione
dello schema dei fold con gli SHA-256 dei tre manifest; run dei fold con quattro fold di 4 soggetti, i 12 file di
punteggio del test (le righe di cnn_test più righe estranee al confronto, con pareggi esatti a 0,5 e alla soglia dev),
le 12 soglie, status.json, aggregate.json e i due SHA256SUMS; N1 con i tre file di punteggio, metrics.json,
aggregate.json, status.json e SHA256SUMS. Punteggi costruiti per dare risposte calcolate a mano (commenti accanto ai
valori attesi): ACER, AUROC con pareggi, differenze, mediana e intervallo sui seed, tabella 2×2 delle decisioni,
conteggi per soggetto, bootstrap (intervallo degenere noto e confronto con una reimplementazione indipendente con gli
stessi indici). Ogni controllo di check-inputs ha un test che lo fa fallire (il mondo è riscritto con tutte le ancore
coerenti, così fallisce solo il controllo colpito). Stato RUNNING → PASS/FAIL, invalidazione delle uscite, albero sporco
(-dev), autorizzazione, cartelle rifiutate, scansione del pubblico. Gli ingressi veri sono usati in sola lettura da un
solo test, saltato se mancano (MISSING_INPUT), che riproduce anche i valori di controllo del coordinatore e quelli
pubblici di N1. Alla fine si controlla che results/, export/ e delivery/ del repository non siano cambiati.

  .venv/bin/python -I scripts/check_nuaa_folds_n1_compare.py [--verbose]

Variabile facoltativa FOLDSCOMPARE_INPUT_ROOT: radice degli ingressi veri (predefinita: la radice del repository; la
usa il controllo delle mutazioni, che esegue questa suite su una copia del codice).
"""
import contextlib
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nuaa_folds_n1_compare as NC  # noqa: E402

REAL_ROOT = Path(os.environ.get("FOLDSCOMPARE_INPUT_ROOT") or ROOT)
MISSING_INPUT = {
    "real inputs": "the cnn_test manifest, the folds run 20261103-FOLDSRUN-e174812 or N1 20261024-N1-LivenessNet-fb93496 are not present",
    "delivered configuration": "delivery/foldscompare/nuaa_folds_n1_compare.json is not present",
}
STATE = {"commit": "abcdef0", "dirty": False}
DATE = "20261104"
RUN_ID = f"{DATE}-FOLDSCOMPARE-abcdef0"
SEEDS = [42, 43, 44]
SUBJECTS = [f"11{i:02d}" for i in range(1, 17)]           # 1101..1116
N1_TRAIN = SUBJECTS[:9]                                    # soggetti di training di N1 (sessioni 01-02)
FOLDS = {1: ["1101", "1102", "1110", "1111"], 2: ["1103", "1104", "1112", "1113"], 3: ["1105", "1106", "1114", "1115"],
         4: ["1107", "1108", "1109", "1116"]}
FOLD_OF = {s: f for f, ss in FOLDS.items() for s in ss}
N_BF = {**{s: 2 for s in SUBJECTS}, "1113": 4, "1114": 4, "1115": 0, "1116": 0}   # 32 bona fide
N_ATT = {s: 2 for s in SUBJECTS}                                                     # 32 attacchi
# soglie dev congelate per (fold, seed): fold 1 a 0,75 (i bona fide a 0,7 diventano corretti), fold 2 a 0,9 (seed 42 e 43:
# gli attacchi a 0,9 pareggiano la soglia e sono bona fide, quindi accettati), il resto a 0,5
THRESHOLDS = {(f, s): (0.75 if f == 1 else 0.9 if f == 2 and s in (42, 43) else 0.5) for f in FOLDS for s in SEEDS}
# N1: bona fide sbagliati (0,8) per 1101-1108 (seed 42 e 44) o 1101-1104 (seed 43), giusti a 0,3; attacchi sbagliati per
# 1101-1102 a 0,3 (pareggio con i bona fide giusti nell'AUROC) e 1103-1104 a 0,5 (pareggio con la soglia: bona fide),
# giusti a 0,6. Run dei fold: bona fide sbagliati a 0,5 per 1101-1102 (0,7), giusti a 0,1; attacchi a 0,9.
N1_BF_WRONG = {42: set(SUBJECTS[:8]), 43: set(SUBJECTS[:4]), 44: set(SUBJECTS[:8])}
# risposte a mano (32 bona fide, 32 attacchi):
#   fold 0,5: BPCER 4/32 = 0.125, APCER 0 -> ACER 0.0625 (ogni seed); AUROC 1 (0,9 > ogni bona fide)
#   fold dev: seed 42/43 APCER 8/32 (attacchi del fold 2 a 0,9 = soglia), BPCER 0 -> 0.125; seed 44 -> 0
#   N1: APCER 8/32 = 0.25; BPCER 16/32 (seed 42, 44) o 8/32 (seed 43) -> ACER 0.375 / 0.25 / 0.375
#   N1 AUROC seed 42/44: 24 att 0,6 > 16 bf 0,3 (384) + 4 att 0,3 = 16 bf 0,3 (4*16*0,5 = 32) + 4 att 0,5 > 16 bf 0,3 (64)
#     = 480/1024 = 0.46875; seed 43 (24 bf a 0,3): 576 + 48 + 96 = 720/1024 = 0.703125
EXPECTED = {42: {"folds_acer_05": 0.0625, "folds_acer_dev": 0.125, "folds_auroc": 1.0, "n1_acer_05": 0.375, "n1_auroc": 0.46875},
            43: {"folds_acer_05": 0.0625, "folds_acer_dev": 0.125, "folds_auroc": 1.0, "n1_acer_05": 0.25, "n1_auroc": 0.703125},
            44: {"folds_acer_05": 0.0625, "folds_acer_dev": 0.0, "folds_auroc": 1.0, "n1_acer_05": 0.375, "n1_auroc": 0.46875}}
CONTROL = {"acer_05": [0.0625, 0.0625, 0.0625], "acer_dev": [0.125, 0.125, 0.0], "auroc": [1.0, 1.0, 1.0]}
FOLDS_RUN_ID = "20261103-FOLDSRUN-0000001"
N1_RUN_ID = "20261024-N1-LivenessNet-0000002"


def skip_missing(test: unittest.TestCase, what: str):
    """Unico modo di saltare un test di questa suite: un ingresso mancante dichiarato in MISSING_INPUT."""
    test.skipTest(f"missing input: {MISSING_INPUT[what]}")


def sha(b) -> str:
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def csv_bytes(cols, rows) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(cols), lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def jbytes(doc) -> bytes:
    return (json.dumps(doc, indent=1) + "\n").encode("utf-8")


def manifest_row(subject: str, lab: str, session: str, pic: int) -> dict:
    name = f"{subject}_{subject}_00_0{lab}_{session}_{pic}.jpg"
    key = f"{'real' if lab == '0' else 'attack'}/{name}"
    return {"key": key, "label": lab, "subject": subject, "glasses": "00", "pos": f"0{lab}", "session": session, "pic": str(pic),
            "official_split": "test" if session == "03" else "train", "sha256": sha(key), "bytes": str(1000 + pic)}


def safe_ref_metrics(rows, label, score, preds: dict) -> dict:
    """ref_metrics su righe di testo; {} se un campo non è un numero (le ancore di un mondo volutamente guasto)."""
    try:
        return ref_metrics([int(r[label]) for r in rows], [float(r[score]) for r in rows],
                           {k: [int(r[c]) for r in rows] for k, c in preds.items()})
    except ValueError:
        return {}


def ref_metrics(y, s, preds: dict) -> dict:
    """Riferimento indipendente: conteggi diretti e AUROC per coppie (pareggi 0,5)."""
    att = [v for v, t in zip(s, y) if t == 1]
    bf = [v for v, t in zip(s, y) if t == 0]
    out = {"n": len(y), "n_bona_fide": len(bf), "n_attack": len(att)}
    for name, p in preds.items():
        acc = sum(1 for t, q in zip(y, p) if t == 1 and q == 0)
        rej = sum(1 for t, q in zip(y, p) if t == 0 and q == 1)
        ap = acc / len(att) if att else None
        bp = rej / len(bf) if bf else None
        out.update({f"accepted_attacks_{name}": acc, f"rejected_bona_fide_{name}": rej, f"apcer_{name}": ap, f"bpcer_{name}": bp,
                    f"acer_{name}": (ap + bp) / 2 if ap is not None and bp is not None else None})
    out["auroc"] = (sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in att for b in bf) / (len(att) * len(bf))) if att and bf else None
    return out


class World:
    """Ingressi sintetici coerenti; write() li scrive con tutte le ancore ricalcolate (SHA256SUMS, status, configurazione)."""

    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="foldscompare_"))
        self.base = self.tmp / "in"
        self.out = self.tmp / "out"
        self.cfg_path = self.tmp / "cfg" / "nuaa_folds_n1_compare.json"
        self.test = [manifest_row(s, lab, "03", i) for s in SUBJECTS for lab, n in (("0", N_BF[s]), ("1", N_ATT[s])) for i in range(n)]
        self.train = [manifest_row(s, lab, "01", 0) for s in N1_TRAIN[:7] for lab in ("0", "1")]
        self.es = [manifest_row(s, lab, "02", 0) for s in N1_TRAIN[7:] for lab in ("0", "1")]
        self.schema_extra = {}
        self.thr = {(f, s): {"fold": f, "train_seed": s, "set": "dev fold only", "criterion": "eer", "value": THRESHOLDS[(f, s)],
                             "checkpoint_sha256": "AUTO"} for f in FOLDS for s in SEEDS}
        self.ckpt = {(f, s): sha(f"checkpoint {f} {s}") for f in FOLDS for s in SEEDS}
        self.frows = {(f, s): self.fold_rows(f, s) for f in FOLDS for s in SEEDS}
        self.n1_config = sha("n1 configuration")
        self.n1_weights = {s: sha(f"n1 weights {s}") for s in SEEDS}
        self.n1rows = {s: self.n1_rows(s) for s in SEEDS}
        self.folds_status = {"run_id": FOLDS_RUN_ID, "status": "PASS", "passed": True}
        self.n1_status = {"result": "PASS", "step": "aggregate"}
        self.drop_listed = set()            # nomi tolti dai SHA256SUMS dei due run
        self.patch_folds_agg = None         # funzione(doc) che altera aggregate.json del run dei fold
        self.patch_n1_agg = None
        self.patch_n1_metrics = None        # funzione(seed, doc)
        self.cfg = self.make_config()
        self.write()

    # ------------------------------------------------------------------ righe
    def fold_rows(self, f: int, s: int) -> list:
        rows = []
        for subj in FOLDS[f]:
            for r in self.test:
                if r["subject"] != subj:
                    continue
                if r["label"] == "0":
                    score = 0.7 if subj in ("1101", "1102") else 0.1
                else:
                    score = 0.9
                rows.append(self.frow(f, s, r, score))
            # righe estranee al confronto: un attacco della sessione 03 e, per i soggetti con sessioni 01-02, un bona
            # fide della 01 a 0,5 esatto (pareggio con la soglia fissa e con le soglie dev a 0,5)
            rows.append(self.frow(f, s, manifest_row(subj, "1", "03", 9), 0.95))
            if subj in N1_TRAIN:
                rows.append(self.frow(f, s, manifest_row(subj, "0", "01", 8), 0.5))
        return rows

    def frow(self, f, s, r, score) -> dict:
        t = THRESHOLDS[(f, s)]
        return {"fold": str(f), "train_seed": str(s), "role": "test", "sample_id": r["key"], "subject_id": r["subject"],
                "session_id": r["session"], "label": r["label"], "score_attack": repr(float(score)),
                "prediction_05": "1" if score > 0.5 else "0", "prediction_dev": "1" if score > t else "0", "threshold_dev": repr(float(t)),
                "file_sha256": r["sha256"], "checkpoint_sha256": "AUTO", "threshold_sha256": "AUTO"}

    def n1_rows(self, s: int) -> list:
        rows = []
        for r in self.test:
            subj = r["subject"]
            if r["label"] == "0":
                score = 0.8 if subj in N1_BF_WRONG[s] else 0.3
            else:
                score = 0.3 if subj in ("1101", "1102") else 0.5 if subj in ("1103", "1104") else 0.6
            rows.append({"run_id": N1_RUN_ID, "train_seed": str(s), "sample_id": r["key"], "subject_id": subj, "session_id": "03",
                         "label": r["label"], "score_attack": repr(float(score)), "prediction": "1" if score > 0.5 else "0",
                         "in_c1_subset": "0", "file_sha256": r["sha256"], "checkpoint_sha256": "AUTO", "config_sha256": "AUTO"})
        return rows

    # ------------------------------------------------------------------ configurazione
    def make_config(self) -> dict:
        real = json.loads((ROOT / NC.DEFAULT_CONFIG).read_text(encoding="utf-8")) if (ROOT / NC.DEFAULT_CONFIG).is_file() else None
        dec = real["declarations"] if real else {**{k: True for k in NC.DECLARATION_FLAGS}, **{k: f"text {k}" for k in NC.DECLARATION_TEXTS}}
        met = real["metrics"] if real else {k: f"text {k}" for k in ("auroc", "differences", "paired_frames", "per_seed", "per_subject", "summary")}
        return {"schema": NC.CONFIG_SCHEMA, "task": "synthetic comparison", "request": {"date": "2026-11-04", "card": "C2", "text": "test"},
                "manifest": {"folder": "manifests", "file": "nuaa_cnn_test.csv", "sha256": "", "rows": 64, "n_bona_fide": 32, "n_attack": 32,
                             "n_subjects": 16, "session": "03"},
                "n1_training_manifests": {"files": {"cnn_train": {"file": "nuaa_cnn_train.csv", "sha256": ""},
                                                    "cnn_es": {"file": "nuaa_cnn_early_stopping.csv", "sha256": ""}},
                                          "sessions": ["01", "02"], "training_subjects_in_set": 9},
                "schema_config": {"path": "schema/nuaa_subject_folds.json", "sha256": ""},
                "folds_run": {"run_id": FOLDS_RUN_ID, "folder": f"foldsrun/{FOLDS_RUN_ID}", "status_sha256": "", "public_sums_sha256": "",
                              "restricted_sums_sha256": "", "k": 4},
                "n1_run": {"run_id": N1_RUN_ID, "folder": f"n1/{N1_RUN_ID}", "sha256sums_sha256": "", "threshold": 0.5},
                "train_seeds": list(SEEDS), "decision": NC.DECISION_RULE, "metrics": met, "bootstrap": dict(NC.BOOT),
                "control_values": {"source": "synthetic", "acer_05_percent": [6.25, 6.25, 6.25], "acer_dev_percent": [12.5, 12.5, 0.0],
                                   "auroc": [1.0, 1.0, 1.0], "comparison": NC.CONTROL_COMPARISON},
                "declarations": dec, "outputs": {"public": "p", "restricted": "r"}}

    # ------------------------------------------------------------------ scrittura con ancore
    def write(self, keep_config_anchors: bool = False):
        b = self.base
        mdir = b / "manifests"
        mdir.mkdir(parents=True, exist_ok=True)
        files = {"cnn_test": (self.cfg["manifest"]["file"], self.test),
                 "cnn_train": (self.cfg["n1_training_manifests"]["files"]["cnn_train"]["file"], self.train),
                 "cnn_es": (self.cfg["n1_training_manifests"]["files"]["cnn_es"]["file"], self.es)}
        man_sha = {}
        for k, (fn, rows) in files.items():
            data = csv_bytes(["key", "label", "subject", "glasses", "pos", "session", "pic", "official_split", "sha256", "bytes"], rows)
            (mdir / fn).write_bytes(data)
            man_sha[k] = sha(data)
        schema = {"schema": "nuaa-subject-folds-config/1", "manifests": {k: {"file": files[k][0], "sha256": man_sha[k]} for k in files}}
        for k, v in self.schema_extra.items():
            schema["manifests"][k].update(v)
        sp = b / self.cfg["schema_config"]["path"]
        sp.parent.mkdir(parents=True, exist_ok=True)
        sb = jbytes(schema)
        sp.write_bytes(sb)
        self.schema_sha = sha(sb)
        # run dei fold
        fdir = b / self.cfg["folds_run"]["folder"]
        (fdir / "restricted").mkdir(parents=True, exist_ok=True)
        (fdir / "public").mkdir(parents=True, exist_ok=True)
        listed = {}
        for (f, s), doc in self.thr.items():
            d = {k: (self.ckpt[(f, s)] if v == "AUTO" else v) for k, v in doc.items()}
            tb = jbytes(d)
            name = f"fold{f}_seed{s}"
            (fdir / "restricted" / f"{name}_threshold.json").write_bytes(tb)
            listed[f"{name}_threshold.json"] = sha(tb)
            rows = [{k: (sha(tb) if (v == "AUTO" and k == "threshold_sha256") else self.ckpt[(f, s)] if v == "AUTO" else v)
                     for k, v in r.items()} for r in self.frows[(f, s)]]
            cb = csv_bytes(NC.FOLD_SCORE_COLUMNS, rows)
            (fdir / "restricted" / f"{name}_test_scores.csv").write_bytes(cb)
            listed[f"{name}_test_scores.csv"] = sha(cb)
            listed[f"{name}_LivenessNet.keras"] = self.ckpt[(f, s)]
        listed["per_subject.csv"] = sha("other")
        rs = "".join(f"{h}  {n}\n" for n, h in sorted(listed.items()) if n not in self.drop_listed).encode()
        (fdir / "restricted" / "SHA256SUMS").write_bytes(rs)
        pooled = {}
        for s in SEEDS:
            rows = [r for f in FOLDS for r in self.frows[(f, s)]]
            pooled[str(s)] = {"all": safe_ref_metrics(rows, "label", "score_attack", {"05": "prediction_05", "dev": "prediction_dev"})}
        agg = {"run_id": FOLDS_RUN_ID, "pooled_per_seed": pooled}
        if self.patch_folds_agg:
            self.patch_folds_agg(agg)
        ab = jbytes(agg)
        (fdir / "public" / "aggregate.json").write_bytes(ab)
        ps = "".join(f"{h}  {n}\n" for n, h in sorted({"aggregate.json": sha(ab), "aggregate.md": sha("md")}.items())
                     if n not in self.drop_listed).encode()
        (fdir / "public" / "SHA256SUMS").write_bytes(ps)
        st = {**self.folds_status, "public_sums_sha256": sha(ps), "restricted_sums_sha256": sha(rs)}
        stb = jbytes(st)
        (fdir / "status.json").write_bytes(stb)
        # N1
        ndir = b / self.cfg["n1_run"]["folder"]
        nl = {}
        aggn = {"passed": True, "run_id": N1_RUN_ID, "config_sha256": self.n1_config, "threshold": 0.5, "seeds": {}}
        for s in SEEDS:
            (ndir / f"seed{s}").mkdir(parents=True, exist_ok=True)
            rows = [{k: (self.n1_weights[s] if (v == "AUTO" and k == "checkpoint_sha256") else self.n1_config if v == "AUTO" else v)
                     for k, v in r.items()} for r in self.n1rows[s]]
            cb = csv_bytes(NC.N1_SCORE_COLUMNS, rows)
            (ndir / f"seed{s}" / "scores_test.csv").write_bytes(cb)
            nl[f"seed{s}/scores_test.csv"] = sha(cb)
            m = safe_ref_metrics(rows, "label", "score_attack", {"05": "prediction"}) or {k: None for k in (
                "n", "n_bona_fide", "n_attack", "accepted_attacks_05", "rejected_bona_fide_05", "apcer_05", "bpcer_05", "acer_05", "auroc")}
            t3 = {"n": m["n"], "n_bona_fide": m["n_bona_fide"], "n_attack": m["n_attack"], "accepted_attacks": m["accepted_attacks_05"],
                  "rejected_bona_fide": m["rejected_bona_fide_05"], "apcer": m["apcer_05"], "bpcer": m["bpcer_05"], "acer": m["acer_05"],
                  "auroc": m["auroc"]}
            met = {"run_id": N1_RUN_ID, "train_seed": s, "metrics": {"test_3000": dict(t3)}, "weights_sha256": self.n1_weights[s]}
            if self.patch_n1_metrics:
                self.patch_n1_metrics(s, met)
            mb = jbytes(met)
            (ndir / f"seed{s}" / "metrics.json").write_bytes(mb)
            nl[f"seed{s}/metrics.json"] = sha(mb)
            aggn["seeds"][str(s)] = {"metrics": {"test_3000": dict(t3)}, "training": {"weights_sha256": self.n1_weights[s]},
                                     "scores_sha256": sha(cb)}
        if self.patch_n1_agg:
            self.patch_n1_agg(aggn)
        nab = jbytes(aggn)
        (ndir / "aggregate.json").write_bytes(nab)
        nl["aggregate.json"] = sha(nab)
        nsb = jbytes(self.n1_status)
        (ndir / "status.json").write_bytes(nsb)
        nl["status.json"] = sha(nsb)
        ns = "".join(f"{h}  {n}\n" for n, h in sorted(nl.items()) if n not in self.drop_listed).encode()
        (ndir / "SHA256SUMS").write_bytes(ns)
        if not keep_config_anchors:
            self.cfg["manifest"]["sha256"] = man_sha["cnn_test"]
            for k in ("cnn_train", "cnn_es"):
                self.cfg["n1_training_manifests"]["files"][k]["sha256"] = man_sha[k]
            self.cfg["schema_config"]["sha256"] = self.schema_sha
            self.cfg["folds_run"].update({"status_sha256": sha(stb), "public_sums_sha256": sha(ps), "restricted_sums_sha256": sha(rs)})
            self.cfg["n1_run"]["sha256sums_sha256"] = sha(ns)
        self.write_config()

    def write_config(self):
        self.cfg_path.parent.mkdir(parents=True, exist_ok=True)
        self.cfg_path.write_text(json.dumps(self.cfg, indent=1) + "\n", encoding="utf-8")

    def folds_file(self, f, s, suffix="test_scores.csv") -> Path:
        return self.base / self.cfg["folds_run"]["folder"] / "restricted" / f"fold{f}_seed{s}_{suffix}"

    def n1_file(self, rel) -> Path:
        return self.base / self.cfg["n1_run"]["folder"] / rel

    def argv(self, cmd, *extra, out=None) -> list:
        return [cmd, "--config", str(self.cfg_path), "--input-root", str(self.base), "--out-root", str(out or self.out), "--date", DATE, *extra]

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


@contextlib.contextmanager
def patched(world: World = None, **attrs):
    """Stato git finto, SHA-256 dello schema del mondo sintetico e altri attributi del modulo sostituiti per la durata."""
    new = {"REPO_STATE": lambda r: dict(STATE), **({"SCHEMA_CONFIG_SHA256": world.schema_sha} if world else {}), **attrs}
    old = {k: getattr(NC, k) for k in new}
    for k, v in new.items():
        setattr(NC, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(NC, k, v)


def run_main(argv) -> tuple:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = NC.main(argv)
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

    def check(self, *extra) -> tuple:
        with patched(self.w):
            return run_main(self.w.argv("check-inputs", *extra))

    def run_cmd(self, *extra, out=None, **attrs) -> tuple:
        with patched(self.w, **attrs):
            return run_main(self.w.argv("run", "--i-am-authorized", *extra, out=out))

    def assertFails(self, needle: str):
        rc, out, err = self.check()
        self.assertEqual(rc, 1, out + err)
        self.assertIn(needle, err)
        self.assertIn("RESULT: FAIL (check-inputs)", out)


# ============================================================================ metriche (risposte a mano)

class TestMetrics(unittest.TestCase):
    def test_auroc_ties_count_half(self):
        # att {0.3, 0.6}, bf {0.3, 0.1}: 0.3=0.3 -> 0.5, 0.3>0.1, 0.6>0.3, 0.6>0.1 -> 3.5/4
        self.assertEqual(NC.auroc([0.3, 0.6], [0.3, 0.1]), 0.875)
        self.assertEqual(NC.auroc([0.5, 0.5], [0.5]), 0.5)
        self.assertIsNone(NC.auroc([], [0.2]))
        self.assertIsNone(NC.auroc([0.2], []))

    def test_auroc_matches_pairwise_reference(self):
        rng = np.random.default_rng(7)
        att, bf = np.round(rng.random(40), 1), np.round(rng.random(30), 1)       # molti pareggi
        ref = ref_metrics([1] * 40 + [0] * 30, list(att) + list(bf), {})["auroc"]
        self.assertAlmostEqual(NC.auroc(att, bf), ref, places=12)

    def test_rates(self):
        y = np.array([0, 0, 0, 0, 1, 1])
        p = np.array([1, 0, 0, 0, 0, 1])
        r = NC.rates(y, p)
        self.assertEqual((r["rejected_bona_fide"], r["accepted_attacks"]), (1, 1))
        self.assertEqual((r["bpcer"], r["apcer"], r["acer"]), (0.25, 0.5, 0.375))
        self.assertIsNone(NC.rates(np.array([1, 1]), np.array([1, 0]))["acer"])

    def test_decision_table_by_hand(self):
        a = {"y": np.array([0, 0, 0, 1, 1, 1]), "n1_p": np.array([0, 1, 1, 1, 0, 0]), "f_p05": np.array([0, 0, 1, 1, 1, 0]),
             "f_pdev": np.array([1, 1, 1, 0, 0, 0])}
        t = NC.decision_table(a)
        # bf: frame 0 ok/ok, 1 N1 sbagliato / fold ok, 2 entrambi sbagliati; att: 3 ok/ok, 4 N1 sbagliato / fold ok, 5 entrambi
        self.assertEqual(t["05"]["bona_fide"], {"both_correct": 1, "n1_only_correct": 0, "folds_only_correct": 1, "both_wrong": 1})
        self.assertEqual(t["05"]["attack"], {"both_correct": 1, "n1_only_correct": 0, "folds_only_correct": 1, "both_wrong": 1})
        self.assertEqual(t["05"]["decisions_changed"], 2)
        # dev: bf 0 N1 ok / fold sbagliato, 1 e 2 entrambi sbagliati; att 3 N1 ok / fold sbagliato, 4 e 5 entrambi sbagliati
        self.assertEqual(t["dev"]["bona_fide"], {"both_correct": 0, "n1_only_correct": 1, "folds_only_correct": 0, "both_wrong": 2})
        self.assertEqual(t["dev"]["attack"], {"both_correct": 0, "n1_only_correct": 1, "folds_only_correct": 0, "both_wrong": 2})
        self.assertEqual(t["dev"]["decisions_changed"], 2)

    def test_bootstrap_too_few_subjects(self):
        a = {"subject": np.array(["a", "b", "c", "d"]), "y": np.array([0, 1, 0, 1])}
        b = NC.bootstrap(a)
        self.assertIn("fewer than 5", b["note"])
        self.assertEqual(b["diff_auroc"], {"low": None, "high": None, "valid_resamples": 0})

    def test_bootstrap_skips_resamples_without_a_class(self):
        # cinque soggetti, bona fide solo per il primo: i ricampionamenti senza di lui non contano per ACER e AUROC
        subj = np.array(["a", "a", "b", "c", "d", "e"])
        y = np.array([0, 1, 1, 1, 1, 1])
        z = np.zeros(6, int)
        a = {"subject": subj, "y": y, "n1_s": np.array([0.1, 0.9, 0.8, 0.7, 0.6, 0.9]), "n1_p": np.array([0, 1, 1, 1, 1, 1]),
             "f_s": np.array([0.2, 0.9, 0.9, 0.9, 0.9, 0.9]), "f_p05": np.array([0, 1, 1, 1, 1, 1]), "f_pdev": z + 1}
        rng = np.random.default_rng(0)
        valid = sum(1 for _ in range(NC.BOOT["resamples"]) if 0 in rng.integers(0, 5, 5))
        b = NC.bootstrap(a)
        self.assertLess(valid, NC.BOOT["resamples"])
        for k in NC.BOOT_KEYS:
            self.assertEqual(b[k]["valid_resamples"], valid, k)
        self.assertEqual((b["folds_acer_05"]["low"], b["folds_acer_05"]["high"]), (0.0, 0.0))
        self.assertEqual((b["folds_acer_dev"]["low"], b["folds_acer_dev"]["high"]), (0.5, 0.5))     # BPCER 1, APCER 0

    def test_bootstrap_restarts_for_every_call(self):
        rng = np.random.default_rng(1)
        subj = np.repeat(np.array([f"s{i:02d}" for i in range(8)]), 4)
        y = np.tile([0, 0, 1, 1], 8)
        a = {"subject": subj, "y": y, "n1_s": rng.random(32), "f_s": rng.random(32)}
        a.update(n1_p=(a["n1_s"] > 0.5).astype(int), f_p05=(a["f_s"] > 0.5).astype(int), f_pdev=(a["f_s"] > 0.4).astype(int))
        first, second = NC.bootstrap(a), NC.bootstrap(a)       # due seed di training con gli stessi dati
        self.assertEqual(first, second)
        # punteggi continui: i percentili 2,5 e 97,5 cadono fra due valori distinti, quindi l'interpolazione lineare conta
        subjects = sorted(set(subj.tolist()))
        gen = np.random.default_rng(0)
        vals = []
        for _ in range(1000):
            ix = np.concatenate([np.flatnonzero(subj == subjects[i]) for i in gen.integers(0, 8, 8)])
            vals.append(ref_metrics(list(y[ix]), list(a["f_s"][ix]), {})["auroc"] - ref_metrics(list(y[ix]), list(a["n1_s"][ix]), {})["auroc"])
        lo, hi = np.percentile(vals, [2.5, 97.5], method="linear")
        self.assertNotEqual(lo, np.percentile(vals, 2.5, method="nearest"))
        self.assertAlmostEqual(first["diff_auroc"]["low"], lo, places=12)
        self.assertAlmostEqual(first["diff_auroc"]["high"], hi, places=12)

    def test_n1_auroc_note(self):
        agg = {"summary_over_seeds": {"n1_auroc": NC.summary([0.3484928888888889, 0.3754402222222223, 0.45821866666666666])}}
        self.assertEqual(NC.n1_auroc_note(agg), "N1 AUROC is below 0.5 on these frames (0.3485-0.4582, as published by N1); "
                                                "the AUROC difference includes that.")
        agg = {"summary_over_seeds": {"n1_auroc": NC.summary([0.3, 0.5, 0.4])}}
        self.assertEqual(NC.n1_auroc_note(agg), "")

    def test_summary(self):
        self.assertEqual(NC.summary([0.3, 0.1, 0.2]), {"values": [0.3, 0.1, 0.2], "median": 0.2, "min": 0.1, "max": 0.3})
        self.assertIsNone(NC.summary([0.1, None, 0.2])["median"])


# ============================================================================ configurazione

class TestConfig(WorldCase):
    def edit(self, fn):
        fn(self.w.cfg)
        self.w.write_config()

    def test_world_passes(self):
        rc, out, err = self.check()
        self.assertEqual(rc, 0, out + err)
        self.assertIn("RESULT: PASS (check-inputs)", out)
        self.assertIn("acer_05 0.06250/0.06250/0.06250; acer_dev 0.12500/0.12500/0.00000; auroc 1.00000/1.00000/1.00000", out)
        self.assertIn("(delivered: False)", out)
        self.assertIn("12 test score files", out)
        self.assertIn("64/64/64 cnn_test rows per seed", out)

    def test_schema(self):
        self.edit(lambda c: c.update(schema="other/1"))
        self.assertFails("schema must be")

    def test_unknown_field(self):
        self.edit(lambda c: c.update(extra=1))
        self.assertFails("fields must be exactly")

    def test_missing_field(self):
        self.edit(lambda c: c.pop("outputs"))
        self.assertFails("fields must be exactly")

    def test_request(self):
        self.edit(lambda c: c["request"].pop("card"))
        self.assertFails("request needs date, card and text")

    def test_manifest_spec(self):
        self.edit(lambda c: c["manifest"].update(extra=1))
        self.assertFails("manifest needs exactly")

    def test_manifest_counts_add_up(self):
        self.edit(lambda c: c["manifest"].update(rows=65))
        self.assertFails("must equal manifest.rows")

    def test_seeds(self):
        self.edit(lambda c: c.update(train_seeds=[42, 43]))
        self.assertFails("train_seeds must be")

    def test_decision(self):
        self.edit(lambda c: c.update(decision="attack if score_attack >= threshold"))
        self.assertFails("decision must be")

    def test_bootstrap(self):
        self.edit(lambda c: c["bootstrap"].update(resamples=100))
        self.assertFails("bootstrap must be")

    def test_n1_threshold(self):
        self.edit(lambda c: c["n1_run"].update(threshold=0.4))
        self.assertFails("n1_run.threshold must be")

    def test_folds_k(self):
        self.edit(lambda c: c["folds_run"].update(k=1))
        self.assertFails("folds_run.k must be")

    def test_declaration_flag(self):
        self.edit(lambda c: c["declarations"].update(no_new_threshold=False))
        self.assertFails("declarations.no_new_threshold must be true")

    def test_declaration_keys(self):
        self.edit(lambda c: c["declarations"].pop("training_roles"))
        self.assertFails("declarations must have exactly")

    def test_control_values_shape(self):
        self.edit(lambda c: c["control_values"].update(auroc=[1.0, 1.0]))
        self.assertFails("control_values.auroc must list")

    def test_control_comparison(self):
        self.edit(lambda c: c["control_values"].update(comparison="within 0.01"))
        self.assertFails("control_values.comparison must be")

    def test_bootstrap_recipe(self):
        self.edit(lambda c: c["bootstrap"].update(recipe="numpy default generator"))
        self.assertFails("bootstrap must be")

    def test_sha_fields(self):
        self.edit(lambda c: c["folds_run"].update(status_sha256="abc"))
        self.assertFails("folds_run.status_sha256 must be a SHA-256")

    def test_config_symlink(self):
        link = self.w.tmp / "link.json"
        link.symlink_to(self.w.cfg_path)
        with patched(self.w):
            rc, out, err = run_main(["check-inputs", "--config", str(link), "--input-root", str(self.w.base)])
        self.assertEqual(rc, 1)
        self.assertIn("symbolic link", err)


# ogni ramo della configurazione a insieme chiuso: (modifica, messaggio atteso)
CONFIG_BRANCHES = [
    (lambda c: c["request"].update(text=" "), "request.text must be a non-empty text"),
    (lambda c: c.update(task=""), "task must be a non-empty text"),
    (lambda c: c["manifest"].update(file=""), "manifest.file must be a non-empty text"),
    (lambda c: c["manifest"].update(sha256="x"), "manifest.sha256 must be a SHA-256"),
    (lambda c: c["manifest"].update(n_attack=0, rows=32), "manifest.n_attack must be a positive integer"),
    (lambda c: c["manifest"].update(n_subjects=True), "manifest.n_subjects must be a positive integer"),
    (lambda c: c["n1_training_manifests"].pop("sessions"), "n1_training_manifests needs files, sessions"),
    (lambda c: c["n1_training_manifests"]["files"].pop("cnn_es"), "must define exactly cnn_train and cnn_es"),
    (lambda c: c["n1_training_manifests"]["files"]["cnn_train"].update(sha256="y"), "files.cnn_train needs file and sha256"),
    (lambda c: c["n1_training_manifests"]["files"]["cnn_es"].update(file=""), "files.cnn_es.file must be a non-empty text"),
    (lambda c: c["n1_training_manifests"].update(sessions=[]), "n1_training_manifests.sessions must list sessions"),
    (lambda c: c["n1_training_manifests"].update(training_subjects_in_set=-1), "training_subjects_in_set must be a non-negative"),
    (lambda c: c["schema_config"].update(sha256="z"), "schema_config needs path and sha256"),
    (lambda c: c["schema_config"].update(path=""), "schema_config.path must be a non-empty text"),
    (lambda c: c["folds_run"].pop("k"), "folds_run needs exactly"),
    (lambda c: c["folds_run"].update(folder=""), "folds_run.folder must be a non-empty text"),
    (lambda c: c["n1_run"].pop("threshold"), "n1_run needs folder, run_id"),
    (lambda c: c["n1_run"].update(run_id=""), "n1_run.run_id must be a non-empty text"),
    (lambda c: c["n1_run"].update(sha256sums_sha256="w"), "n1_run.sha256sums_sha256 must be a SHA-256"),
    (lambda c: c["metrics"].pop("summary"), "metrics needs exactly"),
    (lambda c: c["metrics"].update(summary=""), "metrics.summary must be a non-empty text"),
    (lambda c: c["control_values"].pop("source"), "control_values needs source"),
    (lambda c: c["control_values"].update(source=""), "control_values.source must be a non-empty text"),
    (lambda c: c["control_values"].update(acer_dev_percent=[10.0, 20.0, 150.0]), "control_values.acer_dev_percent must list"),
    (lambda c: c["control_values"].update(acer_05_percent=[0.1, 0.2, "0.3"]), "control_values.acer_05_percent must list"),
    (lambda c: c["control_values"].update(auroc=[0.9, 0.9, 1.2]), "control_values.auroc must list"),
    (lambda c: c["control_values"].pop("comparison"), "control_values needs source"),
    (lambda c: c["declarations"].update(limits=""), "declarations.limits must be a non-empty text"),
    (lambda c: c["outputs"].pop("public"), "outputs needs public and restricted"),
    (lambda c: c["outputs"].update(public=""), "outputs.public must be a non-empty text"),
]


class TestConfigBranches(WorldCase):
    def test_every_branch_fails(self):
        base = json.loads(self.w.cfg_path.read_text())
        for i, (fn, needle) in enumerate(CONFIG_BRANCHES):
            with self.subTest(needle=needle):
                cfg = json.loads(json.dumps(base))
                fn(cfg)
                self.w.cfg = cfg
                self.w.write_config()
                self.assertFails(needle)

    def test_not_json(self):
        self.w.cfg_path.write_text("{not json")
        self.assertFails("is not valid JSON")

    def test_not_an_object(self):
        self.w.cfg_path.write_text("[1, 2]\n")
        self.assertFails("is not a JSON object")


class TestFormats(WorldCase):
    def test_sums_malformed_line(self):
        p = self.w.n1_file("SHA256SUMS")
        p.write_bytes(p.read_bytes() + b"not a checksum line\n")
        self.w.cfg["n1_run"]["sha256sums_sha256"] = sha(p.read_bytes())
        self.w.write_config()
        self.assertFails("N1 SHA256SUMS: malformed line")

    def test_sums_listed_twice(self):
        p = self.w.n1_file("SHA256SUMS")
        first = p.read_bytes().splitlines(keepends=True)[0]
        p.write_bytes(p.read_bytes() + first)
        self.w.cfg["n1_run"]["sha256sums_sha256"] = sha(p.read_bytes())
        self.w.write_config()
        self.assertFails("listed twice")

    def test_ragged_csv(self):
        self.w.frows[(2, 43)][0]["extra"] = "x"
        cols = NC.FOLD_SCORE_COLUMNS
        self.w.write()
        p = self.w.folds_file(2, 43)
        lines = p.read_text().splitlines()
        lines[1] += ",surplus"
        p.write_text("\n".join(lines) + "\n")
        TestFoldsRun.resum_restricted(self)
        self.assertFails("rows with a wrong number of fields")
        self.assertEqual(cols, NC.FOLD_SCORE_COLUMNS)

    def test_not_utf8(self):
        self.w.write()
        p = self.w.folds_file(2, 43)
        p.write_bytes(p.read_bytes() + b"\xff\xfe")
        TestFoldsRun.resum_restricted(self)
        self.assertFails("is not UTF-8")

    def test_empty_score_file(self):
        self.w.frows[(3, 42)] = []
        self.w.write()
        self.assertFails("fold3_seed42_test_scores.csv: no rows")

    def test_threshold_value_not_a_number(self):
        self.w.thr[(2, 42)]["value"] = "0.9"
        self.w.write()
        self.assertFails("fold2_seed42_threshold.json: fold, seed, set, value or checkpoint differ")

    def test_threshold_not_an_object(self):
        self.w.write()
        p = self.w.folds_file(1, 42, "threshold.json")
        p.write_text("[]\n")
        TestFoldsRun.resum_restricted(self)
        self.assertFails("fold1_seed42_threshold.json is not a JSON object")

    def test_folds_public_aggregate_unlisted(self):
        # aggregate.json assente dal SHA256SUMS pubblico del run dei fold (l'N1 lo elenca ancora)
        self.w.write()
        d = self.w.base / self.w.cfg["folds_run"]["folder"]
        ps = b"".join(l + b"\n" for l in (d / "public" / "SHA256SUMS").read_bytes().splitlines() if not l.endswith(b"aggregate.json"))
        (d / "public" / "SHA256SUMS").write_bytes(ps)
        st = json.loads((d / "status.json").read_text())
        st["public_sums_sha256"] = sha(ps)
        (d / "status.json").write_bytes(jbytes(st))
        self.w.cfg["folds_run"].update(public_sums_sha256=sha(ps), status_sha256=sha(jbytes(st)))
        self.w.write_config()
        self.assertFails("folds run aggregate.json: SHA-256 differs from the public SHA256SUMS")

    def test_n1_aggregate_run_id(self):
        self.w.patch_n1_agg = lambda d: d.update(run_id="other")
        self.w.write()
        self.assertFails("N1 aggregate.json: not a passed aggregate")

    def test_n1_metrics_run_id(self):
        self.w.patch_n1_metrics = lambda s, d: d.update(run_id="other") if s == 42 else None
        self.w.write()
        self.assertFails("N1 seed42/metrics.json: run_id or train_seed differ")

    def test_n1_metrics_sha(self):
        p = self.w.n1_file("seed43/metrics.json")
        p.write_bytes(p.read_bytes() + b" ")
        self.assertFails("N1 seed43/metrics.json: SHA-256")

    def test_n1_weights_malformed(self):
        self.w.n1_weights[44] = "not-a-sha"
        self.w.write()
        self.assertFails("N1 seed44: weights_sha256 of metrics.json and of the aggregate differ")


# ============================================================================ ingressi

class TestManifests(WorldCase):
    def test_schema_sha_fixed_in_code(self):
        with patched(self.w, SCHEMA_CONFIG_SHA256="0" * 64):
            rc, out, err = run_main(self.w.argv("check-inputs"))
        self.assertEqual(rc, 1)
        self.assertIn("differs from the approved schema", err)

    def test_schema_sha_of_config(self):
        self.w.cfg["schema_config"]["sha256"] = "0" * 64
        self.w.write_config()
        self.assertFails("differs from the approved schema")

    def test_schema_registers_another_cnn_test(self):
        self.w.schema_extra = {"cnn_test": {"sha256": "1" * 64}}
        self.w.write()
        self.assertFails("differ from manifests.cnn_test of the schema configuration")

    def test_schema_registers_another_training_manifest(self):
        self.w.schema_extra = {"cnn_es": {"file": "other.csv"}}
        self.w.write()
        self.assertFails("differ from manifests.cnn_es of the schema configuration")

    def test_manifest_sha(self):
        p = self.w.base / "manifests" / "nuaa_cnn_test.csv"
        p.write_bytes(p.read_bytes() + b"\n")
        self.w.schema_extra = {}
        self.assertFails("manifest cnn_test (nuaa_cnn_test.csv): SHA-256")

    def test_manifest_invalid_row(self):
        self.w.test[0]["subject"] = "1199"              # il nome dice un altro soggetto
        self.w.write()
        self.assertFails("manifest cnn_test: invalid input")

    def test_manifest_counts(self):
        self.w.cfg["manifest"].update(n_subjects=15)
        self.w.write()
        self.assertFails("n_subjects = 16 differs from the configuration (15)")

    def test_manifest_session(self):
        self.w.cfg["manifest"].update(session="02")
        self.w.write()
        self.assertFails("a row has a session other than 02")

    def test_manifest_duplicated_content(self):
        self.w.test[1]["sha256"] = self.w.test[0]["sha256"]
        self.w.write()
        self.assertFails("two rows share a key or a file SHA-256")

    def test_manifest_duplicated_key(self):
        dup = dict(self.w.test[0], sha256=sha("another content"))      # stessa chiave, contenuto diverso
        self.w.test.append(dup)
        self.w.cfg["manifest"].update(rows=65, n_bona_fide=33)
        self.w.write()
        self.assertFails("two rows share a key or a file SHA-256")

    def test_training_manifest_sha(self):
        p = self.w.base / "manifests" / "nuaa_cnn_early_stopping.csv"
        p.write_bytes(p.read_bytes() + b"\n")
        self.assertFails("manifest cnn_es (nuaa_cnn_early_stopping.csv): SHA-256")

    def test_training_manifest_session(self):
        self.w.train.append(manifest_row("1101", "0", "03", 7))
        self.w.write()
        self.assertFails("manifest cnn_train: a row has a session outside")

    def test_training_manifest_shares_keys(self):
        self.w.cfg["n1_training_manifests"]["sessions"] = ["01", "02", "03"]
        self.w.train.append(dict(self.w.test[0]))
        self.w.write()
        self.assertFails("share keys with cnn_test")

    def test_training_subjects_in_set(self):
        self.w.cfg["n1_training_manifests"]["training_subjects_in_set"] = 8
        self.w.write()
        self.assertFails("N1 training subjects in cnn_test: 9, the configuration declares 8")

    def test_manifest_symlink(self):
        p = self.w.base / "manifests" / "nuaa_cnn_test.csv"
        real = self.w.tmp / "real.csv"
        shutil.move(p, real)
        p.symlink_to(real)
        self.assertFails("is a symbolic link")


class TestFoldsRun(WorldCase):
    def corrupt(self, p: Path):
        p.write_bytes(p.read_bytes() + b" ")

    def set_row(self, f, s, i, **kw):
        self.w.frows[(f, s)][i].update(kw)
        self.w.write()

    def test_status_sha(self):
        self.corrupt(self.w.base / self.w.cfg["folds_run"]["folder"] / "status.json")
        self.assertFails("folds run status.json: SHA-256")

    def test_status_not_pass(self):
        self.w.folds_status["status"] = "FAIL"
        self.w.write()
        self.assertFails("not a PASS of run")

    def test_status_run_id(self):
        self.w.folds_status["run_id"] = "20261103-FOLDSRUN-9999999"
        self.w.write()
        self.assertFails("not a PASS of run")

    def test_status_anchors_differ_from_config(self):
        self.w.cfg["folds_run"]["public_sums_sha256"] = "2" * 64
        self.w.write_config()
        self.assertFails("the SHA256SUMS anchors differ from the configuration")

    def test_public_sums(self):
        self.w.write()
        st = self.w.base / self.w.cfg["folds_run"]["folder"] / "public" / "SHA256SUMS"
        self.corrupt(st)
        # status e configurazione ancorati al vecchio valore: la differenza è del file
        self.assertFails("folds run public SHA256SUMS: SHA-256")

    def test_aggregate_not_as_listed(self):
        self.corrupt(self.w.base / self.w.cfg["folds_run"]["folder"] / "public" / "aggregate.json")
        self.assertFails("aggregate.json: SHA-256 differs from the public SHA256SUMS")

    def test_aggregate_run_id(self):
        self.w.patch_folds_agg = lambda d: d.update(run_id="other")
        self.w.write()
        self.assertFails("folds run aggregate.json: run_id differs")

    def test_restricted_sums(self):
        self.corrupt(self.w.base / self.w.cfg["folds_run"]["folder"] / "restricted" / "SHA256SUMS")
        self.assertFails("folds run restricted SHA256SUMS: SHA-256")

    def test_score_file_not_listed(self):
        self.w.drop_listed = {"fold3_seed43_test_scores.csv"}
        self.w.write()
        self.assertFails("fold3_seed43_test_scores.csv not listed")

    def test_threshold_not_listed(self):
        self.w.drop_listed = {"fold2_seed44_threshold.json"}
        self.w.write()
        self.assertFails("fold2_seed44_threshold.json not listed")

    def test_checkpoint_not_listed(self):
        self.w.drop_listed = {"fold4_seed42_LivenessNet.keras"}
        self.w.write()
        self.assertFails("fold4_seed42_LivenessNet.keras not listed")

    def test_score_file_sha(self):
        self.corrupt(self.w.folds_file(2, 43))
        self.assertFails("fold2_seed43_test_scores.csv: SHA-256")

    def test_threshold_file_sha(self):
        self.corrupt(self.w.folds_file(1, 44, "threshold.json"))
        self.assertFails("fold1_seed44_threshold.json: SHA-256")

    def test_threshold_of_another_fold(self):
        self.w.thr[(3, 42)]["fold"] = 2
        self.w.write()
        self.assertFails("fold3_seed42_threshold.json: fold, seed, set, value or checkpoint differ")

    def test_threshold_of_another_seed(self):
        self.w.thr[(3, 42)]["train_seed"] = 43
        self.w.write()
        self.assertFails("fold3_seed42_threshold.json: fold, seed, set, value or checkpoint differ")

    def test_threshold_set(self):
        self.w.thr[(1, 43)]["set"] = "test"
        self.w.write()
        self.assertFails("fold1_seed43_threshold.json: fold, seed, set, value or checkpoint differ")

    def test_threshold_checkpoint(self):
        self.w.thr[(4, 44)]["checkpoint_sha256"] = "3" * 64
        self.w.write()
        self.assertFails("fold4_seed44_threshold.json: fold, seed, set, value or checkpoint differ")

    def test_columns(self):
        self.w.write()
        p = self.w.folds_file(1, 42)
        p.write_text(p.read_text().replace("threshold_sha256", "thr_sha256", 1))
        self.resum_restricted()           # SHA256SUMS e status riscritti: solo le colonne sono sbagliate
        self.assertFails("fold1_seed42_test_scores.csv: columns")

    def resum_restricted(self):
        d = self.w.base / self.w.cfg["folds_run"]["folder"]
        rs = d / "restricted" / "SHA256SUMS"
        listed = NC.read_sums(rs.read_bytes(), "x")
        for n in listed:
            if (d / "restricted" / n).is_file():
                listed[n] = sha((d / "restricted" / n).read_bytes())
        rb = "".join(f"{h}  {n}\n" for n, h in sorted(listed.items())).encode()
        rs.write_bytes(rb)
        st = json.loads((d / "status.json").read_text())
        st["restricted_sums_sha256"] = sha(rb)
        sb = jbytes(st)
        (d / "status.json").write_bytes(sb)
        self.w.cfg["folds_run"].update(restricted_sums_sha256=sha(rb), status_sha256=sha(sb))
        self.w.write_config()

    def test_row_fold(self):
        self.set_row(1, 42, 0, fold="2")
        self.assertFails("a row carries another fold, seed or role")

    def test_row_seed(self):
        self.set_row(1, 42, 0, train_seed="43")
        self.assertFails("a row carries another fold, seed or role")

    def test_row_role(self):
        self.set_row(2, 44, 3, role="dev")
        self.assertFails("a row carries another fold, seed or role")

    def test_label_range(self):
        self.set_row(2, 42, 1, label="2")
        self.assertFails("label out of range")

    def test_score_not_a_number(self):
        self.set_row(3, 43, 0, score_attack="high")
        self.assertFails("is not a number")

    def test_score_outside_range(self):
        self.set_row(3, 43, 0, score_attack="1.5", prediction_05="1", prediction_dev="1")
        self.assertFails("outside [0, 1]")

    def test_score_nan(self):
        self.set_row(3, 43, 0, score_attack="nan")
        self.assertFails("outside [0, 1]")

    def test_threshold_dev_of_row(self):
        self.set_row(4, 42, 0, threshold_dev="0.6")
        self.assertFails("threshold_dev differs from the frozen threshold of fold4_seed42")

    def test_threshold_dev_text(self):
        self.set_row(4, 42, 0, threshold_dev="0.50")          # stesso valore, testo diverso da quello scritto dal run
        self.assertFails("threshold_dev differs from the frozen threshold of fold4_seed42")

    def test_threshold_sha_of_row(self):
        self.set_row(4, 43, 2, threshold_sha256="4" * 64)
        self.assertFails("threshold_sha256 differs from the SHA-256 of fold4_seed43_threshold.json")

    def test_checkpoint_of_row(self):
        self.set_row(4, 43, 2, checkpoint_sha256="5" * 64)
        self.assertFails("checkpoint_sha256 differs from the restricted SHA256SUMS")

    def test_prediction_05(self):
        self.set_row(1, 44, 0, prediction_05="0" if self.w.frows[(1, 44)][0]["prediction_05"] == "1" else "1")
        self.assertFails("a prediction does not follow")

    def test_prediction_dev(self):
        self.set_row(1, 44, 0, prediction_dev="0" if self.w.frows[(1, 44)][0]["prediction_dev"] == "1" else "1")
        self.assertFails("a prediction does not follow")

    def test_tie_at_05_is_bona_fide(self):
        i = next(i for i, r in enumerate(self.w.frows[(1, 42)]) if r["score_attack"] == "0.5")
        self.set_row(1, 42, i, prediction_05="1")
        self.assertFails("a prediction does not follow")

    def test_tie_at_dev_is_bona_fide(self):
        i = next(i for i, r in enumerate(self.w.frows[(2, 42)]) if r["score_attack"] == "0.9")
        self.set_row(2, 42, i, prediction_dev="1")
        self.assertFails("a prediction does not follow")

    def test_subject_in_two_folds(self):
        r = manifest_row("1101", "1", "03", 11)
        self.w.frows[(2, 43)].append(self.w.frow(2, 43, r, 0.9))
        self.w.write()
        self.assertFails("folds run seed 43: a subject is in the test of two folds")

    def test_row_differs_from_manifest(self):
        i = next(i for i, r in enumerate(self.w.frows[(3, 44)]) if r["sample_id"] in {x["key"] for x in self.w.test})
        self.set_row(3, 44, i, file_sha256="6" * 64)
        self.assertFails("a cnn_test row differs from the manifest")

    def test_row_session_differs_from_manifest(self):
        self.set_row(3, 44, 0, session_id="02")
        self.assertFails("a cnn_test row differs from the manifest")

    def test_key_missing(self):
        self.w.frows[(2, 42)].pop(0)
        self.w.write()
        self.assertFails("folds run seed 42: 63 cnn_test keys found, 1 missing")

    def test_key_twice(self):
        self.w.frows[(2, 42)].append(dict(self.w.frows[(2, 42)][0]))
        self.w.write()
        self.assertFails("1 in more than one row")

    def test_folds_differ_between_seeds(self):
        # seed 44: 1110 e 1112 scambiati fra i fold 1 e 2 (soglie a 0,75 e 0,5, punteggi coerenti)
        for f, s_out, s_in in ((1, "1110", "1112"), (2, "1112", "1110")):
            rows = [r for r in self.w.frows[(f, 44)] if r["subject_id"] != s_out]
            for r in self.w.frows[(3 - f, 44)]:
                if r["subject_id"] == s_in:
                    rows.append(self.w.frow(f, 44, {"key": r["sample_id"], "subject": s_in, "session": r["session_id"], "label": r["label"],
                                                    "sha256": r["file_sha256"]}, float(r["score_attack"])))
            self.w.frows[(f, 44, "new")] = rows
        self.w.frows[(1, 44)], self.w.frows[(2, 44)] = self.w.frows.pop((1, 44, "new")), self.w.frows.pop((2, 44, "new"))
        self.w.write()
        self.assertFails("the held-out subjects of a fold differ between seeds")


class TestN1Run(WorldCase):
    def corrupt(self, p: Path):
        p.write_bytes(p.read_bytes() + b" ")

    def set_row(self, s, i, **kw):
        self.w.n1rows[s][i].update(kw)
        self.w.write()

    def test_sums_sha(self):
        self.corrupt(self.w.n1_file("SHA256SUMS"))
        self.assertFails("N1 SHA256SUMS: SHA-256")

    def test_status_not_listed(self):
        self.w.drop_listed = {"status.json"}
        self.w.write()
        self.assertFails("N1 SHA256SUMS: status.json not listed")

    def test_status_not_pass(self):
        self.w.n1_status["result"] = "FAIL"
        self.w.write()
        self.assertFails("N1 status.json: not a PASS")

    def test_scores_sha(self):
        self.corrupt(self.w.n1_file("seed43/scores_test.csv"))
        self.assertFails("N1 seed43/scores_test.csv: SHA-256")

    def test_aggregate_not_passed(self):
        self.w.patch_n1_agg = lambda d: d.update(passed=False)
        self.w.write()
        self.assertFails("N1 aggregate.json: not a passed aggregate")

    def test_aggregate_threshold(self):
        self.w.patch_n1_agg = lambda d: d.update(threshold=0.4)
        self.w.write()
        self.assertFails("N1 aggregate.json: not a passed aggregate")

    def test_metrics_seed(self):
        self.w.patch_n1_metrics = lambda s, d: d.update(train_seed=99) if s == 44 else None
        self.w.write()
        self.assertFails("N1 seed44/metrics.json: run_id or train_seed differ")

    def test_scores_sha_in_aggregate(self):
        self.w.patch_n1_agg = lambda d: d["seeds"]["42"].update(scores_sha256="7" * 64)
        self.w.write()
        self.assertFails("SHA-256 differs from scores_sha256 of the N1 aggregate")

    def test_weights_differ(self):
        self.w.patch_n1_agg = lambda d: d["seeds"]["43"]["training"].update(weights_sha256="8" * 64)
        self.w.write()
        self.assertFails("N1 seed43: weights_sha256 of metrics.json and of the aggregate differ")

    def test_columns(self):
        old = NC.N1_SCORE_COLUMNS
        cols = tuple(c for c in old if c != "in_c1_subset")
        self.w.write()
        p = self.w.n1_file("seed42/scores_test.csv")
        rows = list(csv.DictReader(io.StringIO(p.read_text())))
        p.write_bytes(csv_bytes(cols, rows))
        self.resum()
        self.assertFails("N1 seed42/scores_test.csv: columns")

    def resum(self):
        d = self.w.base / self.w.cfg["n1_run"]["folder"]
        listed = NC.read_sums((d / "SHA256SUMS").read_bytes(), "x")
        listed = {n: sha((d / n).read_bytes()) for n in listed}
        agg = json.loads((d / "aggregate.json").read_text())
        for s in SEEDS:
            agg["seeds"][str(s)]["scores_sha256"] = listed[f"seed{s}/scores_test.csv"]
        (d / "aggregate.json").write_bytes(jbytes(agg))
        listed["aggregate.json"] = sha(jbytes(agg))
        ns = "".join(f"{h}  {n}\n" for n, h in sorted(listed.items())).encode()
        (d / "SHA256SUMS").write_bytes(ns)
        self.w.cfg["n1_run"]["sha256sums_sha256"] = sha(ns)
        self.w.write_config()

    def test_row_run_id(self):
        self.set_row(42, 0, run_id="other")
        self.assertFails("a row carries another run_id or train_seed")

    def test_row_seed(self):
        self.set_row(42, 0, train_seed="43")
        self.assertFails("a row carries another run_id or train_seed")

    def test_row_checkpoint(self):
        self.set_row(43, 5, checkpoint_sha256="9" * 64)
        self.assertFails("checkpoint_sha256 or config_sha256 differ")

    def test_row_config(self):
        self.set_row(43, 5, config_sha256="a" * 64)
        self.assertFails("checkpoint_sha256 or config_sha256 differ")

    def test_label_range(self):
        self.set_row(44, 1, label="x")
        self.assertFails("N1 seed44/scores_test.csv: label out of range")

    def test_score_range(self):
        self.set_row(44, 1, score_attack="-0.1", prediction="0")
        self.assertFails("outside [0, 1]")

    def test_prediction(self):
        self.set_row(44, 1, prediction="0" if self.w.n1rows[44][1]["prediction"] == "1" else "1")
        self.assertFails("a prediction does not follow")

    def test_tie_is_bona_fide(self):
        i = next(i for i, r in enumerate(self.w.n1rows[42]) if r["score_attack"] == "0.5")
        self.set_row(42, i, prediction="1")
        self.assertFails("a prediction does not follow")

    def test_key_not_in_set(self):
        r = manifest_row("1101", "1", "03", 30)
        self.set_row(42, 0, sample_id=r["key"])
        self.assertFails("a key is not in cnn_test")

    def test_key_twice(self):
        self.w.n1rows[42].append(dict(self.w.n1rows[42][0]))
        self.w.write()
        self.assertFails("a key appears twice")

    def test_row_differs_from_manifest(self):
        self.set_row(43, 2, subject_id="1102")
        self.assertFails("N1 seed43/scores_test.csv: a row differs from the manifest")

    def test_key_missing(self):
        self.w.n1rows[44].pop()
        self.w.write()
        self.assertFails("63 keys, 1 cnn_test keys missing")


class TestAnchors(WorldCase):
    def test_folds_anchor_rate(self):
        self.w.patch_folds_agg = lambda d: d["pooled_per_seed"]["43"]["all"].update(acer_dev=d["pooled_per_seed"]["43"]["all"]["acer_dev"] + 1e-9)
        self.w.write()
        self.assertFails("anchors not reproduced: folds run seed 43 (all held-out frames) acer_dev")

    def test_folds_anchor_count(self):
        self.w.patch_folds_agg = lambda d: d["pooled_per_seed"]["42"]["all"].update(n=d["pooled_per_seed"]["42"]["all"]["n"] + 1)
        self.w.write()
        self.assertFails("folds run seed 42 (all held-out frames) n:")

    def test_folds_anchor_auroc(self):
        self.w.patch_folds_agg = lambda d: d["pooled_per_seed"]["44"]["all"].update(auroc=0.5)
        self.w.write()
        self.assertFails("folds run seed 44 (all held-out frames) auroc")

    def test_folds_anchor_missing(self):
        self.w.patch_folds_agg = lambda d: d["pooled_per_seed"].pop("44")
        self.w.write()
        self.assertFails("anchors not reproduced")

    def test_n1_metrics_anchor(self):
        self.w.patch_n1_metrics = lambda s, d: d["metrics"]["test_3000"].update(auroc=0.47) if s == 42 else None
        self.w.write()
        self.assertFails("N1 seed 42 metrics.json auroc")

    def test_n1_aggregate_anchor(self):
        self.w.patch_n1_agg = lambda d: d["seeds"]["43"]["metrics"]["test_3000"].update(rejected_bona_fide=9)
        self.w.write()
        self.assertFails("N1 seed 43 aggregate.json rejected_bona_fide")

    def test_n1_anchor_apcer(self):
        self.w.patch_n1_agg = lambda d: d["seeds"]["44"]["metrics"]["test_3000"].update(apcer=0.26)
        self.w.write()
        self.assertFails("N1 seed 44 aggregate.json apcer")

    def test_control_values_not_reproduced(self):
        self.w.cfg["control_values"]["acer_dev_percent"][1] = 12.51
        self.w.write_config()
        self.assertFails("control values of the coordinator not reproduced: acer_dev seed 43")

    def test_control_auroc_not_reproduced(self):
        self.w.cfg["control_values"]["auroc"][2] = 0.99998
        self.w.write_config()
        self.assertFails("auroc seed 44")

    def test_control_values_agree_at_given_digits(self):
        # ACER: round(100 * g, 2); AUROC: round(g, 5) (i valori del coordinatore, seed 42)
        self.assertTrue(NC.control_agrees(0.09533333333333334, 9.53, "acer_dev"))
        self.assertFalse(NC.control_agrees(0.09533333333333334, 9.54, "acer_dev"))
        self.assertFalse(NC.control_agrees(0.0956, 9.53, "acer_dev"))
        self.assertTrue(NC.control_agrees(0.15, 15.00, "acer_05"))
        self.assertTrue(NC.control_agrees(0.9385149, 0.93851, "auroc"))
        self.assertFalse(NC.control_agrees(0.938516, 0.93851, "auroc"))
        self.assertFalse(NC.control_agrees(0.93852, 0.93851, "auroc"))
        self.assertFalse(NC.control_agrees(None, 0.93851, "auroc"))


# ============================================================================ run

class TestRun(WorldCase):
    def test_pass_outputs_and_known_answers(self):
        before = snapshot(self.w.base)
        rc, out, err = self.run_cmd()
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(snapshot(self.w.base), before)
        run = self.w.out / RUN_ID
        st = json.loads((run / "status.json").read_text())
        self.assertEqual((st["status"], st["passed"], st["run_id"]), ("PASS", True, RUN_ID))
        self.assertFalse(st["delivered_configuration"])
        self.assertEqual(sorted(os.listdir(run)), ["public", "restricted", "run.log", "status.json"])
        self.assertEqual(sorted(os.listdir(run / "public")), sorted(NC.PUBLIC_OUTPUTS))
        self.assertEqual(sorted(os.listdir(run / "restricted")), sorted(NC.RESTRICTED_OUTPUTS))
        pub_sums = (run / "public" / "SHA256SUMS").read_bytes()
        self.assertEqual(st["public_sums_sha256"], sha(pub_sums))
        self.assertEqual(NC.read_sums(pub_sums, "x"), {n: sha((run / "public" / n).read_bytes()) for n in ("aggregate.json", "aggregate.md")})
        rs = (run / "restricted" / "SHA256SUMS").read_bytes()
        self.assertEqual(st["restricted_sums_sha256"], sha(rs))
        self.assertEqual(NC.read_sums(rs, "x"), {n: sha((run / "restricted" / n).read_bytes())
                                                  for n in ("paired_frames.csv", "per_subject.csv", "public_SHA256SUMS.sha256")})
        self.assertEqual((run / "restricted" / "public_SHA256SUMS.sha256").read_text(), f"{sha(pub_sums)}  public/SHA256SUMS\n")
        self.assertEqual(st["run_log_sha256"], sha((run / "run.log").read_bytes()))
        self.assertIn("RESULT: PASS (run)", (run / "run.log").read_text())
        agg = json.loads((run / "public" / "aggregate.json").read_text())
        for s in SEEDS:
            p = agg["per_seed"][str(s)]
            e = EXPECTED[s]
            self.assertEqual((p["folds"]["n_bona_fide"], p["folds"]["n_attack"], p["n1"]["n_bona_fide"], p["n1"]["n_attack"]), (32, 32, 32, 32))
            self.assertEqual(p["folds"]["acer_05"], e["folds_acer_05"])
            self.assertEqual(p["folds"]["acer_dev"], e["folds_acer_dev"])
            self.assertEqual(p["folds"]["auroc"], e["folds_auroc"])
            self.assertEqual(p["n1"]["acer_05"], e["n1_acer_05"])
            self.assertEqual(p["n1"]["auroc"], e["n1_auroc"])
            d = p["difference_folds_minus_n1"]
            self.assertEqual(d["diff_acer_05"], e["folds_acer_05"] - e["n1_acer_05"])
            self.assertEqual(d["diff_acer_dev"], e["folds_acer_dev"] - e["n1_acer_05"])
            self.assertEqual(d["diff_auroc"], e["folds_auroc"] - e["n1_auroc"])
            self.assertEqual(p["test_subjects_per_fold"], {"1": 4, "2": 4, "3": 4, "4": 4})
        p = agg["per_seed"]["42"]
        # 0,5: bf N1 sbagliati 1101-1108 (16), fold sbagliati 1101-1102 (4); attacchi N1 sbagliati 1101-1104 (8), fold nessuno
        self.assertEqual((p["folds"]["rejected_bona_fide_05"], p["folds"]["accepted_attacks_05"]), (4, 0))
        self.assertEqual((p["folds"]["rejected_bona_fide_dev"], p["folds"]["accepted_attacks_dev"]), (0, 8))
        self.assertEqual((p["n1"]["rejected_bona_fide_05"], p["n1"]["accepted_attacks_05"]), (16, 8))
        t = p["paired_decisions"]
        self.assertEqual(t["05"]["bona_fide"], {"both_correct": 16, "n1_only_correct": 0, "folds_only_correct": 12, "both_wrong": 4})
        self.assertEqual(t["05"]["attack"], {"both_correct": 24, "n1_only_correct": 0, "folds_only_correct": 8, "both_wrong": 0})
        self.assertEqual(t["05"]["decisions_changed"], 20)
        # dev: attacchi del fold 2 (1103, 1104, 1112, 1113) accettati dal run dei fold; N1 sbagliato su 1101-1104
        self.assertEqual(t["dev"]["bona_fide"], {"both_correct": 16, "n1_only_correct": 0, "folds_only_correct": 16, "both_wrong": 0})
        self.assertEqual(t["dev"]["attack"], {"both_correct": 20, "n1_only_correct": 4, "folds_only_correct": 4, "both_wrong": 4})
        self.assertEqual(t["dev"]["decisions_changed"], 24)
        c = p["per_subject_counts"]
        self.assertEqual((c["n_subjects"], c["with_both_classes"]), (16, 14))
        self.assertEqual(c["comparison"]["05"]["error_rate"], {"folds_lower": 8, "equal": 8, "folds_higher": 0, "not_defined": 0})
        self.assertEqual(c["comparison"]["05"]["acer"], {"folds_lower": 8, "equal": 6, "folds_higher": 0, "not_defined": 2})
        self.assertEqual(c["comparison"]["dev"]["error_rate"], {"folds_lower": 8, "equal": 6, "folds_higher": 2, "not_defined": 0})
        self.assertEqual(c["comparison"]["dev"]["acer"], {"folds_lower": 8, "equal": 4, "folds_higher": 2, "not_defined": 2})
        so = agg["summary_over_seeds"]
        self.assertEqual(so["n1_acer_05"], {"values": [0.375, 0.25, 0.375], "median": 0.375, "min": 0.25, "max": 0.375})
        self.assertEqual(so["diff_acer_dev"], {"values": [-0.25, -0.125, -0.375], "median": -0.25, "min": -0.375, "max": -0.125})
        self.assertEqual(so["diff_auroc"]["values"], [0.53125, 0.296875, 0.53125])
        self.assertEqual(so["folds_acer_dev"]["median"], 0.125)
        b = agg["bootstrap_per_seed"]["42"]
        self.assertEqual((b["folds_auroc"]["low"], b["folds_auroc"]["high"]), (1.0, 1.0))      # ogni attacco sopra ogni bona fide
        # N1 identico nei seed 42 e 44: con il generatore ripartito da 0 per ogni seed gli intervalli coincidono
        b44, b43 = agg["bootstrap_per_seed"]["44"], agg["bootstrap_per_seed"]["43"]
        for k in ("n1_auroc", "n1_acer_05", "folds_acer_05"):
            self.assertEqual(b[k], b44[k], k)
        self.assertNotEqual(b["n1_acer_05"], b43["n1_acer_05"])
        self.assertEqual(agg["bootstrap"]["recipe"], NC.BOOT_RECIPE)
        self.assertEqual(b["n_subjects"], 16)
        self.assertEqual(agg["control_values"]["computed"], CONTROL)
        for k in ("descriptive", "post_hoc", "no_training", "no_new_threshold"):
            self.assertIs(agg["declarations"][k], True)
        md = (run / "public" / "aggregate.md").read_text()
        self.assertIn("| 42 | folds, dev | 0 / 8 | 0.0000 | 0.2500 | 0.1250", md)
        self.assertIn("| 42 | 0.5 | 16 / 0 / 12 / 4 | 24 / 0 / 8 / 0 | 20 |", md)
        self.assertIn("| 42 | dev | 16 / 0 / 16 / 0 | 20 / 4 / 4 / 4 | 24 |", md)
        self.assertIn("| acer_dev | 42 | 12.50 % | 0.12500 |", md)
        self.assertIn("| auroc | 44 | 1.00000 | 1.00000 |", md)
        self.assertIn("numpy.random.default_rng(0) (PCG64)", md)
        self.assertNotIn("N1 AUROC is below 0.5", md)          # seed 43: 0.703125
        self.assertTrue(all(len(line) <= 120 or line.startswith("|") for line in md.splitlines()))
        rows = list(csv.DictReader(io.StringIO((run / "restricted" / "paired_frames.csv").read_text())))
        self.assertEqual(len(rows), 3 * 64)
        r = next(x for x in rows if x["train_seed"] == "42" and x["subject_id"] == "1103" and x["label"] == "1")
        self.assertEqual((r["n1_score_attack"], r["n1_prediction_05"], r["fold"], r["fold_score_attack"], r["fold_threshold_dev"],
                          r["fold_prediction_05"], r["fold_prediction_dev"]), ("0.5", "0", "2", "0.9", "0.9", "1", "0"))
        subj = list(csv.DictReader(io.StringIO((run / "restricted" / "per_subject.csv").read_text())))
        self.assertEqual(len(subj), 3 * 16)
        r = next(x for x in subj if x["train_seed"] == "42" and x["subject_id"] == "1113")
        self.assertEqual((r["fold"], r["n_bona_fide"], r["n_attack"], r["folds_errors_dev"], r["folds_acer_dev"], r["n1_acer_05"]),
                         ("2", "4", "2", "2", "0.5", "0.0"))
        # 1101, seed 42: N1 bona fide a 0,8 e attacchi a 0,3 -> AUROC 0; run dei fold bona fide 0,7, attacchi 0,9 -> 1
        r = next(x for x in subj if x["train_seed"] == "42" and x["subject_id"] == "1101")
        self.assertEqual((r["n1_auroc"], r["folds_auroc"], r["n1_errors_05"], r["folds_errors_05"]), ("0.0", "1.0", "4", "2"))
        # 1103, seed 43: N1 bona fide a 0,8 e attacchi a 0,5 -> 0; fold bona fide 0,1, attacchi 0,9 -> 1
        r = next(x for x in subj if x["train_seed"] == "43" and x["subject_id"] == "1103")
        self.assertEqual((r["n1_auroc"], r["folds_auroc"]), ("0.0", "1.0"))           # N1 bf 0,8 e att 0,5
        r = next(x for x in subj if x["train_seed"] == "43" and x["subject_id"] == "1115")
        self.assertEqual((r["n1_acer_05"], r["folds_auroc"]), ("NA", "NA"))

    def test_bootstrap_matches_independent_resampling(self):
        rc, out, err = self.run_cmd()
        self.assertEqual(rc, 0, out + err)
        agg = json.loads((self.w.out / RUN_ID / "public" / "aggregate.json").read_text())
        by_subject = {s: [] for s in SUBJECTS}
        n1 = {r["sample_id"]: r for r in self.w.n1rows[43]}
        for f in FOLDS:
            for r in self.w.frows[(f, 43)]:
                if r["sample_id"] in n1:
                    by_subject[r["subject_id"]].append((int(r["label"]), float(r["score_attack"]), int(r["prediction_dev"]),
                                                        float(n1[r["sample_id"]]["score_attack"]), int(n1[r["sample_id"]]["prediction"])))
        rng = np.random.default_rng(0)
        d_dev, d_auc, n1_auc = [], [], []
        for _ in range(1000):
            sample = [x for i in rng.integers(0, 16, 16) for x in by_subject[SUBJECTS[i]]]
            y = [x[0] for x in sample]
            f = ref_metrics(y, [x[1] for x in sample], {"dev": [x[2] for x in sample]})
            n = ref_metrics(y, [x[3] for x in sample], {"05": [x[4] for x in sample]})
            if f["acer_dev"] is not None and n["acer_05"] is not None:
                d_dev.append(f["acer_dev"] - n["acer_05"])
            if f["auroc"] is not None and n["auroc"] is not None:
                d_auc.append(f["auroc"] - n["auroc"])
                n1_auc.append(n["auroc"])
        b = agg["bootstrap_per_seed"]["43"]
        for key, vals in (("diff_acer_dev", d_dev), ("diff_auroc", d_auc), ("n1_auroc", n1_auc)):
            lo, hi = np.percentile(vals, [2.5, 97.5])
            self.assertAlmostEqual(b[key]["low"], lo, places=12)
            self.assertAlmostEqual(b[key]["high"], hi, places=12)
            self.assertEqual(b[key]["valid_resamples"], len(vals))

    def test_dirty_tree_dev_suffix(self):
        rc, out, err = self.run_cmd(REPO_STATE=lambda r: {"commit": "abcdef0", "dirty": True})
        self.assertEqual(rc, 0, out + err)
        self.assertTrue((self.w.out / f"{RUN_ID}-dev" / "status.json").is_file())
        self.assertFalse((self.w.out / RUN_ID).exists())

    def test_needs_authorization(self):
        with patched(self.w):
            rc, out, err = run_main(self.w.argv("run"))
        self.assertEqual(rc, 1)
        self.assertIn("--i-am-authorized", err)
        self.assertFalse(self.w.out.exists())

    def test_check_inputs_writes_nothing(self):
        before = snapshot(self.w.tmp)
        rc, out, err = self.check()
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(snapshot(self.w.tmp), before)
        self.assertFalse(self.w.out.exists())

    def test_existing_status_refused(self):
        run = self.w.out / RUN_ID
        run.mkdir(parents=True)
        (run / "status.json").write_text('{"status": "FAIL"}\n')
        rc, out, err = self.run_cmd()
        self.assertEqual(rc, 1)
        self.assertIn("already holds status.json", err)
        self.assertEqual((run / "status.json").read_text(), '{"status": "FAIL"}\n')

    def test_failure_removes_previous_outputs(self):
        run = self.w.out / RUN_ID
        for sub, names in (("public", NC.PUBLIC_OUTPUTS), ("restricted", NC.RESTRICTED_OUTPUTS)):
            (run / sub).mkdir(parents=True)
            for n in names:
                (run / sub / n).write_text("old PASS\n")
        self.w.cfg["control_values"]["auroc"][0] = 0.9
        self.w.write_config()   # AUROC di controllo diverso: FAIL dopo l'avvio
        rc, out, err = self.run_cmd()
        self.assertEqual(rc, 1)
        st = json.loads((run / "status.json").read_text())
        self.assertEqual((st["status"], st["passed"], st["outputs"]), ("FAIL", False, "removed"))
        self.assertIn("control values of the coordinator not reproduced", st["reason"])
        self.assertEqual(os.listdir(run / "public"), [])
        self.assertEqual(os.listdir(run / "restricted"), [])
        self.assertIn("RESULT: FAIL (run)", (run / "run.log").read_text())
        self.assertEqual(st["run_log_sha256"], sha((run / "run.log").read_bytes()))

    def test_previous_outputs_invalidated_at_start(self):
        run = self.w.out / RUN_ID
        (run / "public").mkdir(parents=True)
        (run / "public" / ".aggregate.json.x1.tmp").write_text("stale temporary\n")
        rc, out, err = self.run_cmd()
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(sorted(os.listdir(run / "public")), sorted(NC.PUBLIC_OUTPUTS))

    def test_failure_after_writes_removes_outputs(self):
        def status_fails_on_pass(run, doc):
            if doc["status"] == "PASS":
                raise OSError("disk full")
            NC.mw.write_all(run, {NC.STATUS: json.dumps(doc, indent=1) + "\n"})
        rc, out, err = self.run_cmd(write_status=status_fails_on_pass)
        self.assertEqual(rc, 1)
        run = self.w.out / RUN_ID
        st = json.loads((run / "status.json").read_text())
        self.assertEqual((st["status"], st["outputs"]), ("FAIL", "removed"))
        self.assertEqual(os.listdir(run / "public"), [])
        self.assertEqual(os.listdir(run / "restricted"), [])

    def test_unexpected_error_fails_cleanly(self):
        def boom(cfg, inp):
            raise ZeroDivisionError("synthetic")
        rc, out, err = self.run_cmd(aggregate=boom)
        self.assertEqual(rc, 1)
        st = json.loads((self.w.out / RUN_ID / "status.json").read_text())
        self.assertEqual(st["status"], "FAIL")
        self.assertIn("unexpected error ZeroDivisionError", st["reason"])

    def test_foreign_entry_refused(self):
        run = self.w.out / RUN_ID / "restricted"
        run.mkdir(parents=True)
        (run / "notes.txt").write_text("mine\n")
        rc, out, err = self.run_cmd()
        self.assertEqual(rc, 1)
        self.assertIn("is not an output of this command", err)
        self.assertEqual((run / "notes.txt").read_text(), "mine\n")
        self.assertFalse((self.w.out / RUN_ID / "status.json").exists())

    def test_out_root_symlink_refused(self):
        real = self.w.tmp / "real_out"
        real.mkdir()
        link = self.w.tmp / "link_out"
        link.symlink_to(real)
        rc, out, err = self.run_cmd(out=link)
        self.assertEqual(rc, 1)
        self.assertIn("symbolic link", err)
        self.assertEqual(os.listdir(real), [])

    def test_out_root_inside_repository_refused(self):
        target = ROOT / "docs" / "foldscompare-test-never-created"
        rc, out, err = self.run_cmd(out=target)
        self.assertEqual(rc, 1)
        self.assertIn("refusing the run folder", err)
        self.assertFalse(target.exists())

    def test_input_inside_run_folder_refused(self):
        out = self.w.base / self.w.cfg["n1_run"]["folder"]
        rc, out_, err = self.run_cmd(out=out)
        self.assertEqual(rc, 1)
        self.assertIn(f"the N1 run {N1_RUN_ID} is inside the run folder, or contains it", err)
        self.assertFalse((out / RUN_ID).exists())

    def test_configuration_inside_run_folder_refused(self):
        run = self.w.out / RUN_ID
        run.mkdir(parents=True)
        cfg = run / "config.json"
        shutil.copy(self.w.cfg_path, cfg)
        argv = self.w.argv("run", "--i-am-authorized")
        argv[argv.index("--config") + 1] = str(cfg)
        with patched(self.w):
            rc, out, err = run_main(argv)
        self.assertEqual(rc, 1)
        self.assertIn("the configuration config.json is inside the run folder", err)
        self.assertEqual(os.listdir(run), ["config.json"])

    def test_public_scan_blocks_identifiers(self):
        old = NC.aggregate_md

        def leaky(cfg, agg, head):
            return old(cfg, agg, head) + "\nsubject 1107 was hard\n"
        rc, out, err = self.run_cmd(aggregate_md=leaky)
        self.assertEqual(rc, 1)
        self.assertIn("public file aggregate.md refused: subject id 1107", err)
        for sub in ("public", "restricted"):
            d = self.w.out / RUN_ID / sub
            self.assertTrue(not d.exists() or os.listdir(d) == [], sub)
        self.assertEqual(json.loads((self.w.out / RUN_ID / "status.json").read_text())["status"], "FAIL")

    def test_public_scan_rules(self):
        inp = NC.Inputs()
        inp.manifest = [manifest_row("1101", "0", "03", 0)]
        inp.subjects, inp.n1_training_subjects = {"1101"}, {"1109"}
        self.assertEqual(NC.public_problems("ACER 0.1101 and 11010 and 1.1109", inp), [])
        self.assertIn("image keys", NC.public_problems("real/1101_1101_00_00_03_0.jpg", inp))
        self.assertIn("image SHA-256 values", NC.public_problems(inp.manifest[0]["sha256"], inp))
        self.assertIn("subject id 1109", NC.public_problems("subject 1109", inp))
        self.assertTrue(any(p.startswith("absolute path") for p in NC.public_problems("/Us" "ers/x", inp)))
        self.assertTrue(any(p.startswith("IP address") for p in NC.public_problems("host 10.1.2.3", inp)))
        self.assertTrue(any(p.startswith("unfinished") for p in NC.public_problems("FIX" "ME later", inp, markdown=True)))

    def test_public_files_have_no_identifiers(self):
        rc, out, err = self.run_cmd()
        self.assertEqual(rc, 0, out + err)
        for n in ("aggregate.json", "aggregate.md"):
            text = (self.w.out / RUN_ID / "public" / n).read_text()
            for s in SUBJECTS:
                self.assertNotRegex(text, rf"(?<![\w.]){s}(?![\w.])")
            self.assertNotIn("real/", text)
            self.assertNotIn("attack/", text)

    def test_run_id_and_date(self):
        argv = self.w.argv("run", "--i-am-authorized")
        argv[argv.index("--date") + 1] = "2026-11-04"
        with patched(self.w):
            rc, out, err = run_main(argv)
        self.assertEqual(rc, 1)
        self.assertIn("--date must be YYYYMMDD", err)
        self.assertTrue(NC.RUN_ID_RE.match(RUN_ID))
        self.assertFalse(NC.RUN_ID_RE.match("20261104-FOLDSRUN-abcdef0"))


# ============================================================================ configurazione consegnata e ingressi veri

def real_values(seed_metrics: dict) -> dict:
    return {k: [seed_metrics[s][e][m] for s in SEEDS] for k, (e, m) in
            {"acer_05": ("folds", "acer_05"), "acer_dev": ("folds", "acer_dev"), "auroc": ("folds", "auroc"),
             "n1_acer": ("n1", "acer_05"), "n1_auroc": ("n1", "auroc")}.items()}


class TestDelivered(unittest.TestCase):
    def test_delivered_configuration(self):
        p = REAL_ROOT / NC.DEFAULT_CONFIG
        if not p.is_file():
            skip_missing(self, "delivered configuration")
        cfg, h = NC.load_config(p)
        self.assertEqual(h, NC.CONFIG_SHA256)
        self.assertEqual(cfg["manifest"]["sha256"], "ce9669dccae37041e778ff7bcba5f4cef08fa79f50d65be84806b084c5d95748")
        self.assertEqual(cfg["control_values"]["acer_05_percent"], [10.90, 15.00, 10.00])
        self.assertEqual(cfg["control_values"]["acer_dev_percent"], [9.53, 15.30, 11.70])
        self.assertEqual(cfg["control_values"]["auroc"], [0.93851, 0.95268, 0.96701])


class TestRealInputs(unittest.TestCase):
    def test_real_check_inputs_and_coordinator_values(self):
        p = REAL_ROOT / NC.DEFAULT_CONFIG
        if not p.is_file():
            skip_missing(self, "delivered configuration")
        cfg, _ = NC.load_config(p)
        needed = [REAL_ROOT / cfg["manifest"]["folder"] / cfg["manifest"]["file"], REAL_ROOT / cfg["folds_run"]["folder"] / "status.json",
                  REAL_ROOT / cfg["n1_run"]["folder"] / "SHA256SUMS", REAL_ROOT / cfg["schema_config"]["path"]]
        if not all(q.is_file() for q in needed):
            skip_missing(self, "real inputs")
        before = snapshot(REAL_ROOT / "results")
        rc, out, err = run_main(["check-inputs", "--config", str(p), "--input-root", str(REAL_ROOT)])
        self.assertEqual(rc, 0, out + err)
        inp = NC.load_inputs(cfg, REAL_ROOT)
        got = real_values(NC.seed_metrics(cfg, inp))
        # valori di controllo del coordinatore (4/11), indipendenti dalla configurazione
        # accordo alle cifre date dal coordinatore: ACER in percentuale a 2 decimali, AUROC a 5
        for k, want in (("acer_05", [10.90, 15.00, 10.00]), ("acer_dev", [9.53, 15.30, 11.70])):
            self.assertEqual([round(100 * g, 2) for g in got[k]], want, k)
        self.assertEqual([round(g, 5) for g in got["auroc"]], [0.93851, 0.95268, 0.96701])
        # valori pubblici di N1 (aggregate.json di N1: ACER 0,4010-0,5143, AUROC 0,3485-0,4582)
        for g, w in zip(got["n1_acer"], [0.43833333333333335, 0.5143333333333333, 0.401]):
            self.assertAlmostEqual(g, w, places=12)
        for g, w in zip(got["n1_auroc"], [0.3484928888888889, 0.3754402222222223, 0.45821866666666666]):
            self.assertAlmostEqual(g, w, places=12)
        agg, paired, subj = NC.aggregate(cfg, inp)
        self.assertEqual(len(paired), 9000)
        self.assertEqual(agg["per_seed"]["42"]["per_subject_counts"]["n_subjects"], 16)
        for s in SEEDS:
            self.assertEqual((agg["per_seed"][str(s)]["folds"]["n_bona_fide"], agg["per_seed"][str(s)]["folds"]["n_attack"]), (1500, 1500))
        head = {"run_id": "20261104-FOLDSCOMPARE-0000000", "commit": "0000000", "dirty": False, "config_sha256": "0" * 64, "finished": "now"}
        self.assertEqual(NC.public_problems(json.dumps(agg), inp), [])
        md = NC.aggregate_md(cfg, agg, head)
        self.assertEqual(NC.public_problems(md, inp, markdown=True), [])
        self.assertIn("N1 AUROC is below 0.5 on these frames (0.3485-0.4582, as published by N1); the AUROC difference includes that.",
                      " ".join(md.split()))
        self.assertEqual(snapshot(REAL_ROOT / "results"), before)
        print("\n   real inputs: " + "; ".join(f"{k} " + "/".join(f"{v:.5f}" for v in vals) for k, vals in got.items()), file=sys.stderr)


class ZZRepositoryUntouched(unittest.TestCase):
    def test_declared_skips_only(self):
        self.assertEqual(sorted(MISSING_INPUT), ["delivered configuration", "real inputs"])


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
