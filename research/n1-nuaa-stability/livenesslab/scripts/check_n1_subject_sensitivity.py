"""
Test di scripts/n1_subject_sensitivity.py (N1, parte 1), offline e deterministici.

Fixture sintetiche con risposte note calcolate a mano (commenti accanto ai valori attesi): pareggi alla soglia 0,5 e
alla soglia M1 (pareggio = bona fide), classe assente per soggetto e dopo l'esclusione (NA, mai 0), cambio di segno del
ΔACER e ΔACER nullo, riproduzione di una piccola tabella M1 (blocco principale e blocco senza i soggetti dev) e suo
rifiuto quando una cella differisce; SHA-256 d'ingresso diverso (file, SHA256SUMS, riga mancante), collegamenti
simbolici, score diversi fra C1 e M1, previsione o soglia salvate incoerenti → FAIL; nessun PASS vecchio dopo un FAIL;
cartella di uscita estranea, collegata o protetta → rifiuto prima di qualunque modifica; ingressi intatti byte per byte.
Due test usano gli ingressi veri (sola lettura, uscite in una cartella temporanea) e si saltano solo se mancano
(ingresso mancante dichiarato in MISSING_INPUT). Tutte le uscite vanno in cartelle temporanee; al termine si controlla
che results/, export/ e delivery/ del repository non siano cambiati.

  .venv/bin/python scripts/check_n1_subject_sensitivity.py        # uscita 1 se un test fallisce

Variabile facoltativa N1_INPUT_ROOT: radice degli ingressi veri (predefinita: la radice del repository; la usa il
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
from fractions import Fraction
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import n1_subject_sensitivity as NS  # noqa: E402

REAL_ROOT = Path(os.environ.get("N1_INPUT_ROOT") or ROOT)
STATE = {"commit": "abcdef0", "dirty": False}
RUN_ID = "20261024-N1-subjects-abcdef0"

MISSING_INPUT = {
    "real inputs": "the accepted C1 and M1 run folders are not present (export/drive/02_Experiments/C1 and M1)",
}


def skip_missing(test: unittest.TestCase, what: str):
    """Unico modo di saltare un test di questa suite: un ingresso mancante dichiarato in MISSING_INPUT."""
    test.skipTest(f"missing input: {MISSING_INPUT[what]}")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ============================================================================ fixture

TABLE_HEAD = ("| analyzer | n | n_bona_fide | n_attack | APCER@0.5 | BPCER@0.5 | ACER@0.5 | accepted attacks@0.5 | rejected bona fide@0.5 | "
              "dev threshold | APCER@dev | BPCER@dev | ACER@dev | accepted attacks@dev | rejected bona fide@dev | test EER | test AUC | "
              "operational dev EER | dev AUC | dev flag |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")


def table_row(a, cells) -> str:
    return f"| {a} | " + " | ".join(cells) + " | 0.5000 | 0.5000 | 0.0000 | 1.0000 |  |\n"


def table_text(main: dict, dev: dict) -> str:
    return ("# fixture table\n\nText.\n\n" + TABLE_HEAD + "".join(table_row(a, c) for a, c in main.items()) +
            "\n## Sensitivity (not the main result): test images of the dev subjects removed\n\nText.\n\n" + TABLE_HEAD +
            "".join(table_row(a, c) for a, c in dev.items()))


# Fixture A: una CNN (cnn_a), soglia M1 0,3, quattro soggetti, soggetto dev S2.
#   S1: bona fide 0.2, 0.4; attacco 0.6          S2: attacchi 0.5 (pareggio a 0,5), 0.25 (solo attacchi)
#   S3: bona fide 0.3 (pareggio a 0,3), 0.7 (solo bona fide)          S4: bona fide 0.45; attacco 0.35
A_SAMPLES = [("real/S1_a.jpg", "S1", 0, "0.2"), ("real/S1_b.jpg", "S1", 0, "0.4"), ("attack/S1_c.jpg", "S1", 1, "0.6"),
             ("attack/S2_a.jpg", "S2", 1, "0.5"), ("attack/S2_b.jpg", "S2", 1, "0.25"),
             ("real/S3_a.jpg", "S3", 0, "0.3"), ("real/S3_b.jpg", "S3", 0, "0.7"),
             ("real/S4_a.jpg", "S4", 0, "0.45"), ("attack/S4_b.jpg", "S4", 1, "0.35")]
# completo: n_b 5, n_a 4; a 0,5 FA 3 (0.5, 0.25, 0.35), FR 1 (0.7): APCER 3/4, BPCER 1/5, ACER 19/40 = 0.4750
#           a 0,3 FA 1 (0.25), FR 3 (0.4, 0.7, 0.45): APCER 1/4, BPCER 3/5, ACER 17/40 = 0.4250
# senza S2 (blocco dev): n_b 5, n_a 2; a 0,5 FA 1, FR 1: 0.5000, 0.2000, 0.3500; a 0,3 FA 0, FR 3: 0.0000, 0.6000, 0.3000
A_MAIN = ["9", "5", "4", "0.7500", "0.2000", "0.4750", "3", "1", "0.300000", "0.2500", "0.6000", "0.4250", "1", "3"]
A_DEV = ["7", "5", "2", "0.5000", "0.2000", "0.3500", "1", "1", "0.300000", "0.0000", "0.6000", "0.3000", "0", "3"]
A_EXP = {"n_images": 9, "n_bona_fide": 5, "n_attack": 4, "n_subjects": 4, "n_dev_subjects": 1, "dataset_id": "nuaa", "split": "test"}
# ΔACER completo 17/40 − 19/40 = −1/20; senza un soggetto:
#   S1: a 0,5 (3/3 + 1/3)/2 = 2/3, a 0,3 (1/3 + 2/3)/2 = 1/2 → −1/6
#   S2: 7/20 → 3/10 → −1/20 (uguale al completo)
#   S3: a 0,5 (3/4 + 0)/2 = 3/8, a 0,3 (1/4 + 2/3)/2 = 11/24 → +1/12 (cambio di segno)
#   S4: a 0,5 (2/3 + 1/4)/2 = 11/24, a 0,3 (1/3 + 1/2)/2 = 5/12 → −1/24
A_LOSO = {"S1": Fraction(-1, 6), "S2": Fraction(-1, 20), "S3": Fraction(1, 12), "S4": Fraction(-1, 24)}

# Fixture B: classe bona fide presente in un solo soggetto (T1); soglia M1 0,3; soggetto dev T3.
#   T1: bona fide 0.2, attacco 0.8          T2: attacco 0.4          T3: attacco 0.9
B_SAMPLES = [("real/T1_a.jpg", "T1", 0, "0.2"), ("attack/T1_b.jpg", "T1", 1, "0.8"), ("attack/T2_a.jpg", "T2", 1, "0.4"),
             ("attack/T3_a.jpg", "T3", 1, "0.9")]
# completo: n_b 1, n_a 3; a 0,5 FA 1 (0.4), FR 0: 1/3, 0, 1/6; a 0,3 FA 0, FR 0: ACER 0 → ΔACER −1/6
# senza T1: nessun bona fide → BPCER, ACER e ΔACER NA; senza T2: 0 → 0, Δ 0 (segno 0); senza T3: 1/4 → 0, Δ −1/4
B_MAIN = ["4", "1", "3", "0.3333", "0.0000", "0.1667", "1", "0", "0.300000", "0.0000", "0.0000", "0.0000", "0", "0"]
B_DEV = ["3", "1", "2", "0.5000", "0.0000", "0.2500", "1", "0", "0.300000", "0.0000", "0.0000", "0.0000", "0", "0"]
B_EXP = {"n_images": 4, "n_bona_fide": 1, "n_attack": 3, "n_subjects": 3, "n_dev_subjects": 1, "dataset_id": "nuaa", "split": "test"}

SCORE_HEADER = ["run_id", "sample_id", "dataset_id", "split", "subject_id", "session_id", "video_id", "attack_type", "modality", "label",
                "score_attack", "threshold", "prediction", "checkpoint_sha256", "config_sha256", "analyzer_id"]


def score_csv(samples, analyzer, threshold, run_id, extra_analyzer=True, override=None) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(SCORE_HEADER)
    t = float(threshold)
    for sid, subj, lab, s in samples:
        row = [run_id, sid, "nuaa", "test", subj, "00", "", "print" if lab else "", "RGB", str(lab), s, threshold,
               str(int(float(s) > t)), "x" * 64, "y" * 64, analyzer]
        if override:
            row = override(row)
        w.writerow(row)
        if extra_analyzer:              # un analizzatore non richiesto: ignorato
            w.writerow([run_id, sid, "nuaa", "test", subj, "00", "", "", "RGB", str(lab), "0.99", "0.5", "1", "z", "z", "other"])
    return buf.getvalue().encode("utf-8")


class Fixture:
    """Cartelle C1 e M1 finte (file, SHA256SUMS) e configurazione con gli SHA-256 calcolati qui."""

    def __init__(self, tmp: Path, samples, exp, main, dev, dev_subject, analyzer="cnn_a", tm1="0.3", c1_override=None, m1_override=None,
                 table=None, threshold_doc=None):
        self.tmp, self.base = tmp, tmp / "inputs"
        c1d, m1d = self.base / "C1run", self.base / "M1run"
        c1d.mkdir(parents=True)
        m1d.mkdir(parents=True)
        man = io.StringIO(newline="")
        mwr = csv.writer(man, lineterminator="\n")
        mwr.writerow(["file", "label", "subject", "session", "sha256", "bytes"])
        for sid, subj, lab, _ in samples:
            mwr.writerow([sid, lab, subj, "01", "0" * 64, 1])
        thr = threshold_doc or {"synthetic": False, "analyzers": {analyzer: {"value": float(tm1), "decision_rule": "attack if score > value (tie = bona fide)"}}}
        files = {
            "c1_scores": (c1d / "scores.csv", score_csv(samples, analyzer, "0.5", "C1", override=c1_override)),
            "c1_manifest": (c1d / "nuaa_manifest.csv", man.getvalue().encode("utf-8")),
            "m1_scores_test_dev_threshold": (m1d / "scores_test_dev_threshold.csv", score_csv(samples, analyzer, tm1, "M1", override=m1_override)),
            "m1_threshold": (m1d / "threshold.json", (json.dumps(thr, indent=1) + "\n").encode("utf-8")),
            "m1_table": (m1d / "nuaa_test_dev_threshold_table.md", (table or table_text({analyzer: main}, {analyzer: dev})).encode("utf-8")),
            "m1_dev_manifest": (m1d / "nuaa_dev_manifest.csv", f"file,label,subject,in_training_sample,sha256,bytes\nattack/d.jpg,1,{dev_subject},1,{'0' * 64},1\n".encode("utf-8")),
        }
        for p, b in files.values():
            p.write_bytes(b)
        sums = {}
        for d in (c1d, m1d):
            text = "".join(f"{sha(q.read_bytes())}  {q.name}\n" for q in sorted(d.iterdir()))
            (d / "SHA256SUMS").write_text(text, encoding="utf-8")
            sums[d] = sha((d / "SHA256SUMS").read_bytes())
        self.files = {k: p for k, (p, _) in files.items()}
        self.cfg = {"schema": NS.CONFIG_SCHEMA, "analyzers": [analyzer], "fixed_threshold": 0.5, "decision_rule": NS.DECISION_RULE,
                    "expected": dict(exp), "inputs": {}}
        for k, p in self.files.items():
            self.cfg["inputs"][k] = {"path": p.relative_to(self.base).as_posix(), "sha256": sha(p.read_bytes()),
                                     "sums": (p.parent / "SHA256SUMS").relative_to(self.base).as_posix(), "sums_sha256": sums[p.parent]}
        self.config = tmp / "config.json"
        self.write_config()
        self.out = tmp / "out"
        self.run = self.out / RUN_ID

    def write_config(self):
        self.config.write_text(json.dumps(self.cfg, indent=1) + "\n", encoding="utf-8")

    def args(self, cmd="run"):
        return [cmd, "--config", str(self.config), "--input-root", str(self.base), "--out-root", str(self.out), "--date", "20261024"]

    def snapshot(self) -> dict:
        return {str(p): sha(p.read_bytes()) for p in sorted(self.base.rglob("*")) if p.is_file()} | {str(self.config): sha(self.config.read_bytes())}


def call(argv, state=STATE) -> tuple:
    """Esegue il comando nel processo, con lo stato git fissato; (codice, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    old = NS.REPO_STATE
    NS.REPO_STATE = (lambda root: dict(state))
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = NS.main(argv)
    finally:
        NS.REPO_STATE = old
    return rc, out.getvalue(), err.getvalue()


def read_csv(p: Path) -> list:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="n1-subjects-test-")
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def fixture_a(self, **kw):
        return Fixture(self.tmp, A_SAMPLES, A_EXP, A_MAIN, A_DEV, "S2", **kw)

    def fixture_b(self, **kw):
        return Fixture(self.tmp, B_SAMPLES, B_EXP, B_MAIN, B_DEV, "T3", **kw)

    def assertPass(self, fx):
        before = fx.snapshot()
        rc, out, err = call(fx.args())
        self.assertEqual(rc, 0, out + err)
        self.assertIn("RESULT: PASS (run)", out)
        self.assertEqual(fx.snapshot(), before, "inputs changed")
        st = json.loads((fx.run / "status.json").read_text())
        self.assertEqual((st["status"], st["passed"]), ("PASS", True))
        return out

    def assertFailRun(self, fx, needle, admitted=True):
        before = fx.snapshot()
        rc, out, err = call(fx.args())
        self.assertEqual(rc, 1, out + err)
        self.assertIn("RESULT: FAIL (run)", out)
        self.assertIn(needle, err)
        self.assertEqual(fx.snapshot(), before, "inputs changed")
        if admitted:
            st = json.loads((fx.run / "status.json").read_text())
            self.assertEqual((st["status"], st["passed"], st["outputs"]), ("FAIL", False, "removed"))
            for sub in ("public", "restricted"):
                d = fx.run / sub
                self.assertEqual(sorted(os.listdir(d)) if d.is_dir() else [], [], f"outputs left in {sub}")
        return err


# ============================================================================ risposte note

class TestKnownAnswers(Base):
    def test_fixture_a_per_subject_counts_and_ties(self):
        fx = self.fixture_a()
        self.assertPass(fx)
        rows = {r["subject_id"]: r for r in read_csv(fx.run / "restricted" / "n1_per_subject.csv")}
        self.assertEqual(sorted(rows), ["S1", "S2", "S3", "S4"])
        got = {s: (int(r["n_bona_fide"]), int(r["n_attack"]), int(r["fa_at_0_5"]), int(r["fr_at_0_5"]), int(r["fa_at_m1"]), int(r["fr_at_m1"]))
               for s, r in rows.items()}
        self.assertEqual(got, {"S1": (2, 1, 0, 0, 0, 1),      # 0.4 > 0.3 rifiutato alla soglia M1
                               "S2": (0, 2, 2, 0, 1, 0),      # 0.5 a 0,5: pareggio = bona fide → attacco accettato
                               "S3": (2, 0, 0, 1, 0, 1),      # 0.3 a 0,3: pareggio = bona fide accettato; 0.7 rifiutato
                               "S4": (1, 1, 1, 0, 0, 1)})
        self.assertEqual((rows["S2"]["apcer_at_0_5"], rows["S2"]["bpcer_at_0_5"]), ("1.000000", "NA"))
        self.assertEqual((rows["S3"]["apcer_at_m1"], rows["S3"]["bpcer_at_m1"]), ("NA", "0.500000"))

    def test_fixture_a_loso_delta_and_sign_change(self):
        fx = self.fixture_a()
        self.assertPass(fx)
        loso = {r["excluded_subject"]: r for r in read_csv(fx.run / "restricted" / "n1_loso.csv")}
        for s, d in A_LOSO.items():
            self.assertEqual(loso[s]["delta_acer_exact"], f"{d.numerator}/{d.denominator}", s)
            self.assertEqual(loso[s]["full_delta_acer"], "-0.050000")
        self.assertEqual({s: (r["sign"], r["sign_change"]) for s, r in loso.items()},
                         {"S1": ("-", "false"), "S2": ("-", "false"), "S3": ("+", "true"), "S4": ("-", "false")})
        self.assertEqual((loso["S1"]["n_bona_fide"], loso["S1"]["n_attack"], loso["S1"]["acer_at_0_5"], loso["S1"]["acer_at_m1"]),
                         ("3", "3", "0.666667", "0.500000"))
        pub = json.loads((fx.run / "public" / "n1_subject_sensitivity_public.json").read_text())
        r = pub["analyzers"]["cnn_a"]
        self.assertEqual((r["full"]["delta_acer_exact"], r["full"]["at_0_5"]["acer"], r["full"]["at_m1"]["acer"]), ("-1/20", "0.475000", "0.425000"))
        s = r["leave_one_subject_out"]
        self.assertEqual((s["delta_min_exact"], s["delta_max_exact"], s["n_sign_changes"], s["n_opposite_sign"], s["n_zero"], s["n_na"], s["n_defined"]),
                         ("-1/6", "1/12", 1, 1, 0, 0, 4))
        self.assertEqual((s["n_bona_fide_after_exclusion"], s["n_attack_after_exclusion"]), ([3, 5], [2, 4]))
        self.assertEqual((s["subjects_both_classes"], s["subjects_attack_only"], s["subjects_bona_fide_only"]), (2, 1, 1))
        self.assertEqual(pub["accepted_m1_table_reproduced"]["cells_checked"], {"main": 14, "without the dev subjects": 14})

    def test_fixture_b_class_absent_gives_na_and_zero_delta(self):
        fx = self.fixture_b()
        self.assertPass(fx)
        loso = {r["excluded_subject"]: r for r in read_csv(fx.run / "restricted" / "n1_loso.csv")}
        t1 = loso["T1"]                                  # nessun bona fide dopo l'esclusione
        self.assertEqual((t1["n_bona_fide"], t1["apcer_at_0_5"], t1["bpcer_at_0_5"], t1["acer_at_0_5"], t1["acer_at_m1"],
                          t1["delta_acer"], t1["sign"], t1["sign_change"]), ("0", "0.500000", "NA", "NA", "NA", "NA", "NA", "NA"))
        self.assertEqual((loso["T2"]["delta_acer"], loso["T2"]["sign"], loso["T2"]["sign_change"]), ("0.000000", "0", "true"))
        self.assertEqual(loso["T3"]["delta_acer_exact"], "-1/4")
        s = json.loads((fx.run / "public" / "n1_subject_sensitivity_public.json").read_text())["analyzers"]["cnn_a"]["leave_one_subject_out"]
        self.assertEqual((s["n_na"], s["n_defined"], s["delta_min_exact"], s["delta_max_exact"], s["n_sign_changes"], s["n_opposite_sign"], s["n_zero"]),
                         (1, 2, "-1/4", "0/1", 1, 0, 1))
        self.assertEqual(s["n_bona_fide_after_exclusion"], [0, 1])

    def test_public_outputs_have_no_subject_ids_and_sums_hold(self):
        fx = self.fixture_a()
        self.assertPass(fx)
        for f in (fx.run / "public").iterdir():
            text = f.read_text()
            for s in ("S1", "S2", "S3", "S4"):
                self.assertNotIn(s, text, f.name)
        for sub in ("public", "restricted"):
            d = fx.run / sub
            listed = {}
            for line in (d / "SHA256SUMS").read_text().splitlines():
                h, n = line.split("  ", 1)
                listed[n] = h
            self.assertEqual(sorted(listed), sorted(x for x in os.listdir(d) if x != "SHA256SUMS"))
            for n, h in listed.items():
                self.assertEqual(sha((d / n).read_bytes()), h, n)
        anchor = (fx.run / "restricted" / "public_SHA256SUMS.sha256").read_text().split()[0]
        self.assertEqual(anchor, sha((fx.run / "public" / "SHA256SUMS").read_bytes()))

    def test_per_image_decisions(self):
        fx = self.fixture_a()
        self.assertPass(fx)
        rows = {r["sample_id"]: r for r in read_csv(fx.run / "restricted" / "n1_per_image.csv")}
        self.assertEqual(len(rows), 9)
        self.assertEqual((rows["attack/S2_a.jpg"]["prediction_at_0_5"], rows["real/S3_a.jpg"]["prediction_at_m1"]), ("0", "0"))
        self.assertEqual((rows["attack/S1_c.jpg"]["prediction_at_0_5"], rows["real/S1_b.jpg"]["prediction_at_m1"]), ("1", "1"))

    def test_rates_na_never_zero(self):
        r = NS.rates({"n_bona_fide": 0, "n_attack": 2, "fa": 0, "fr": 0})
        self.assertEqual((r["apcer"], r["bpcer"], r["acer"]), (Fraction(0), None, None))
        r = NS.rates({"n_bona_fide": 3, "n_attack": 0, "fa": 0, "fr": 0})
        self.assertEqual((r["apcer"], r["bpcer"], r["acer"]), (None, Fraction(0), None))
        self.assertEqual((NS.dec(None), NS.dec(Fraction(0)), NS.dec(Fraction(-1, 3))), ("NA", "0.000000", "-0.333333"))
        self.assertEqual((NS.sign(None), NS.sign(Fraction(0)), NS.sign(Fraction(-1, 9)), NS.sign(Fraction(1, 9))), ("NA", "0", "-", "+"))

    def test_run_id_dirty_suffix(self):
        fx = self.fixture_a()
        rc, out, err = call(fx.args(), {"commit": "abcdef0", "dirty": True})
        self.assertEqual(rc, 0, out + err)
        st = json.loads((fx.out / (RUN_ID + "-dev") / "status.json").read_text())
        self.assertEqual((st["run_id"], st["dirty"]), (RUN_ID + "-dev", True))
        self.assertFalse(fx.run.exists())


# ============================================================================ riproduzione della tabella M1

class TestTableReproduction(Base):
    def test_main_cell_differs(self):
        bad = list(A_MAIN)
        bad[11] = "0.4251"                               # ACER@dev
        fx = Fixture(self.tmp, A_SAMPLES, A_EXP, bad, A_DEV, "S2")
        self.assertIn("ACER@dev: recomputed 0.4250 differs from the accepted 0.4251", self.assertFailRun(fx, "M1 table (main block)"))

    def test_count_differs(self):
        bad = list(A_MAIN)
        bad[6] = "2"                                     # accepted attacks@0.5 (con la regola >= sarebbe 2)
        self.assertFailRun(Fixture(self.tmp, A_SAMPLES, A_EXP, bad, A_DEV, "S2"), "accepted attacks@0.5")

    def test_dev_block_differs(self):
        bad = list(A_DEV)
        bad[2] = "3"
        self.assertFailRun(Fixture(self.tmp, A_SAMPLES, A_EXP, A_MAIN, bad, "S2"), "M1 table (without the dev subjects block)")

    def test_threshold_cell_differs(self):
        bad = list(A_MAIN)
        bad[8] = "0.300001"
        self.assertFailRun(Fixture(self.tmp, A_SAMPLES, A_EXP, bad, A_DEV, "S2"), "dev threshold")

    def test_missing_dev_block(self):
        t = "# t\n\n" + TABLE_HEAD + table_row("cnn_a", A_MAIN)
        self.assertFailRun(Fixture(self.tmp, A_SAMPLES, A_EXP, A_MAIN, A_DEV, "S2", table=t), "1 tables found instead of 2")

    def test_missing_row(self):
        t = table_text({"cnn_z": A_MAIN}, {"cnn_a": A_DEV})
        self.assertFailRun(Fixture(self.tmp, A_SAMPLES, A_EXP, A_MAIN, A_DEV, "S2", table=t), "no row for cnn_a")


# ============================================================================ ingressi

class TestInputs(Base):
    def test_check_inputs_pass_and_no_writes(self):
        fx = self.fixture_a()
        before = fx.snapshot()
        rc, out, err = call(fx.args("check-inputs"))
        self.assertEqual(rc, 0, out + err)
        self.assertFalse(fx.out.exists())
        self.assertEqual(fx.snapshot(), before)

    def test_file_changed(self):
        fx = self.fixture_a()
        p = fx.files["c1_scores"]
        p.write_bytes(p.read_bytes().replace(b",0.6,", b",0.61,", 1))
        self.assertFailRun(fx, "input c1_scores (scores.csv): SHA-256 mismatch")
        rc, out, err = call(fx.args("check-inputs"))
        self.assertEqual(rc, 1)
        self.assertIn("SHA-256 mismatch", err)

    def test_file_and_sums_changed(self):
        fx = self.fixture_a()
        p = fx.files["m1_threshold"]
        p.write_bytes(p.read_bytes().replace(b"0.3", b"0.31"))
        fx.cfg["inputs"]["m1_threshold"]["sha256"] = sha(p.read_bytes())
        s = p.parent / "SHA256SUMS"
        s.write_text("".join(f"{sha(q.read_bytes())}  {q.name}\n" for q in sorted(p.parent.iterdir()) if q.name != "SHA256SUMS"))
        fx.write_config()
        self.assertFailRun(fx, "SHA256SUMS of m1_scores_test_dev_threshold: SHA-256 mismatch")   # stesso SHA256SUMS, il primo ingresso M1 lo rileva

    def test_sums_line_differs(self):
        fx = self.fixture_a()
        s = fx.files["m1_table"].parent / "SHA256SUMS"
        text = s.read_text().replace(fx.cfg["inputs"]["m1_table"]["sha256"], "0" * 64)
        s.write_text(text)
        for k in ("m1_scores_test_dev_threshold", "m1_threshold", "m1_table", "m1_dev_manifest"):
            fx.cfg["inputs"][k]["sums_sha256"] = sha(s.read_bytes())
        fx.write_config()
        self.assertFailRun(fx, "input m1_table: SHA-256 mismatch with its SHA256SUMS")

    def test_sums_line_missing(self):
        fx = self.fixture_a()
        s = fx.files["c1_manifest"].parent / "SHA256SUMS"
        s.write_text("".join(l + "\n" for l in s.read_text().splitlines() if "nuaa_manifest.csv" not in l))
        for k in ("c1_scores", "c1_manifest"):
            fx.cfg["inputs"][k]["sums_sha256"] = sha(s.read_bytes())
        fx.write_config()
        self.assertFailRun(fx, "is not listed exactly once in its SHA256SUMS")

    def test_symlink_input(self):
        fx = self.fixture_a()
        p = fx.files["m1_dev_manifest"]
        real = self.tmp / "elsewhere.csv"
        real.write_bytes(p.read_bytes())
        p.unlink()
        p.symlink_to(real)
        self.assertFailRun(fx, "is a symbolic link: inputs must be regular files")

    def test_scores_differ_between_c1_and_m1(self):
        fx = self.fixture_a(m1_override=lambda r: r[:10] + ["0.6000001"] + r[11:] if r[1] == "attack/S1_c.jpg" else r)
        self.assertFailRun(fx, "score differs between C1 (0.6) and M1 (0.6000001)")

    def test_stored_prediction_inconsistent(self):
        fx = self.fixture_a(c1_override=lambda r: r[:12] + ["1"] + r[13:] if r[1] == "attack/S2_a.jpg" else r)
        self.assertFailRun(fx, "stored prediction '1' differs from the rule score > threshold")

    def test_row_threshold_differs(self):
        fx = self.fixture_a(m1_override=lambda r: r[:11] + ["0.31"] + r[12:] if r[1] == "real/S1_a.jpg" else r)
        self.assertFailRun(fx, "threshold 0.31 differs from the expected 0.3")

    def test_subject_differs_from_manifest(self):
        fx = self.fixture_a(c1_override=lambda r: r[:4] + ["S9"] + r[5:] if r[1] == "real/S4_a.jpg" else r)
        self.assertFailRun(fx, "subject or label differ")

    def test_synthetic_threshold_refused(self):
        doc = {"synthetic": True, "analyzers": {"cnn_a": {"value": 0.3, "decision_rule": "attack if score > value (tie = bona fide)"}}}
        self.assertFailRun(self.fixture_a(threshold_doc=doc), "synthetic must be false")

    def test_expected_counts_differ(self):
        fx = self.fixture_a()
        fx.cfg["expected"]["n_subjects"] = 16
        fx.write_config()
        self.assertFailRun(fx, "4 subjects instead of 16")

    def test_bad_config_nothing_written(self):
        fx = self.fixture_a()
        fx.cfg["fixed_threshold"] = 0.4
        fx.write_config()
        self.assertFailRun(fx, "fixed_threshold must be 0.5", admitted=False)
        self.assertFalse(fx.out.exists())

    def test_manifest_differs_from_both_score_files(self):
        sub = (lambda r: r[:4] + ["S9"] + r[5:] if r[1] == "real/S4_a.jpg" else r)
        self.assertFailRun(self.fixture_a(c1_override=sub, m1_override=sub), "subject or label differ")

    def test_sample_sets_differ(self):
        ren = (lambda r: r[:1] + ["real/S4_x.jpg"] + r[2:] if r[1] == "real/S4_a.jpg" else r)
        self.assertFailRun(self.fixture_a(m1_override=ren), "the samples of the C1 scores, the M1 scores and the C1 manifest differ")

    def test_row_missing(self):
        fx = self.fixture_a()
        p = fx.files["m1_scores_test_dev_threshold"]
        lines = p.read_text().splitlines(keepends=True)
        p.write_text("".join(l for l in lines if not (l.split(",")[1] == "real/S4_a.jpg" and l.rstrip().endswith(",cnn_a"))))
        self.refresh(fx, "m1_scores_test_dev_threshold")
        self.assertFailRun(fx, "8 rows for cnn_a instead of 9")

    def test_duplicate_row(self):
        fx = self.fixture_a()
        p = fx.files["c1_scores"]
        lines = p.read_text().splitlines(keepends=True)
        p.write_text("".join(lines) + next(l for l in lines if l.split(",")[1] == "real/S4_a.jpg" and l.rstrip().endswith(",cnn_a")))
        self.refresh(fx, "c1_scores")
        self.assertFailRun(fx, "duplicated sample real/S4_a.jpg for cnn_a")

    def test_dataset_differs(self):
        ds = (lambda r: r[:2] + ["casia_fasd"] + r[3:] if r[1] == "real/S1_a.jpg" else r)
        self.assertFailRun(self.fixture_a(c1_override=ds, m1_override=ds), "dataset or split differs from the configuration")

    def test_class_counts_differ(self):
        fx = self.fixture_a()
        fx.cfg["expected"]["n_attack"] = 5
        fx.write_config()
        self.assertFailRun(fx, "5 bona fide and 4 attacks instead of 5 and 5")

    def test_dev_subject_count_differs(self):
        fx = self.fixture_a()
        fx.cfg["expected"]["n_dev_subjects"] = 2
        fx.write_config()
        self.assertFailRun(fx, "M1 dev manifest: 1 subjects instead of 2")

    @staticmethod
    def refresh(fx, key):
        """Dopo una modifica voluta di un ingresso: SHA256SUMS della cartella e configurazione riallineati (l'ingresso
        resta «accettato», così si prova il controllo successivo e non quello degli SHA-256)."""
        p = fx.files[key]
        s = p.parent / "SHA256SUMS"
        s.write_text("".join(f"{sha(q.read_bytes())}  {q.name}\n" for q in sorted(p.parent.iterdir()) if q.name != "SHA256SUMS"))
        fx.cfg["inputs"][key]["sha256"] = sha(p.read_bytes())
        for k, v in fx.cfg["inputs"].items():
            if fx.files[k].parent == p.parent:
                v["sums_sha256"] = sha(s.read_bytes())
        fx.write_config()


# ============================================================================ uscite

class TestOutputs(Base):
    def test_no_stale_pass_after_failure(self):
        fx = self.fixture_a()
        self.assertPass(fx)
        p = fx.files["c1_scores"]
        p.write_bytes(p.read_bytes() + b"\n")
        self.assertFailRun(fx, "SHA-256 mismatch")
        self.assertEqual(sorted(os.listdir(fx.run)), ["public", "restricted", "status.json"])

    def test_previous_outputs_removed_before_computation(self):
        fx = self.fixture_a()
        self.assertPass(fx)
        seen = {}

        def probe(a):
            seen["public"] = sorted(os.listdir(fx.run / "public"))
            seen["restricted"] = sorted(os.listdir(fx.run / "restricted"))
            seen["status"] = json.loads((fx.run / "status.json").read_text())["status"]
            NS.fail("probe stop")
        with mock.patch.object(NS, "prepare", probe):
            self.assertFailRun(fx, "probe stop")
        self.assertEqual(seen, {"public": [], "restricted": [], "status": "RUNNING"})

    def test_failure_after_writing_removes_outputs(self):
        fx = self.fixture_a()
        real = NS.write_status

        def status(run, doc):
            if doc["status"] == "PASS":
                raise OSError("disk full")
            return real(run, doc)
        with mock.patch.object(NS, "write_status", status):
            rc, out, err = call(fx.args())
        self.assertNotEqual(rc, 0)
        st = json.loads((fx.run / "status.json").read_text())
        self.assertEqual((st["status"], st["outputs"], st["reason"]), ("FAIL", "removed", "unexpected error OSError"))
        self.assertEqual(os.listdir(fx.run / "public") + os.listdir(fx.run / "restricted"), [])

    def test_rerun_identical_outputs(self):
        fx = self.fixture_a()
        self.assertPass(fx)
        first = {p.name: p.read_bytes() for p in (fx.run / "restricted").iterdir() if p.suffix == ".csv"}
        self.assertPass(fx)
        self.assertEqual({p.name: p.read_bytes() for p in (fx.run / "restricted").iterdir() if p.suffix == ".csv"}, first)

    def test_foreign_file_refused_nothing_removed(self):
        fx = self.fixture_a()
        self.assertPass(fx)
        (fx.run / "public" / "notes.txt").write_text("x")
        before = sorted(p.name for p in (fx.run / "public").iterdir())
        st = (fx.run / "status.json").read_bytes()
        rc, out, err = call(fx.args())
        self.assertEqual(rc, 1)
        self.assertIn("public/notes.txt in the run folder is not an output of this command", err)
        self.assertEqual(sorted(p.name for p in (fx.run / "public").iterdir()), before)
        self.assertEqual((fx.run / "status.json").read_bytes(), st)

    def test_output_symlink_refused(self):
        fx = self.fixture_a()
        target = self.tmp / "target"
        target.mkdir()
        fx.out.mkdir()
        fx.run.symlink_to(target, target_is_directory=True)
        rc, out, err = call(fx.args())
        self.assertEqual(rc, 1)
        self.assertIn("symbolic link", err)
        self.assertEqual(list(target.iterdir()), [])

    def test_output_file_link_refused(self):
        fx = self.fixture_a()
        self.assertPass(fx)
        ext = self.tmp / "external.json"
        ext.write_text("keep")
        q = fx.run / "public" / "n1_subject_sensitivity_public.json"
        q.unlink()
        q.symlink_to(ext)
        rc, out, err = call(fx.args())
        self.assertEqual(rc, 0, out + err)               # il collegamento è tolto come collegamento, il bersaglio mai
        self.assertEqual(ext.read_text(), "keep")
        self.assertFalse(q.is_symlink())

    def test_protected_repository_folder_refused(self):
        fx = self.fixture_a()
        target = ROOT / "results" / "m1" / "n1-test-must-not-exist"
        self.assertFalse(target.exists())
        argv = fx.args()
        argv[argv.index("--out-root") + 1] = str(target)
        try:
            rc, out, err = call(argv)
            self.assertEqual(rc, 1)
            self.assertIn("refusing the run folder", err)
            self.assertFalse(target.exists())
        finally:
            if target.exists():
                shutil.rmtree(target)

    def test_repository_folder_outside_results_n1_refused(self):
        fx = self.fixture_a()
        argv = fx.args()
        argv[argv.index("--out-root") + 1] = str(ROOT / "docs" / "n1-test-must-not-exist")
        rc, out, err = call(argv)
        self.assertEqual(rc, 1)
        self.assertFalse((ROOT / "docs" / "n1-test-must-not-exist").exists())

    def test_input_among_outputs_refused(self):
        fx = self.fixture_a()
        fx.run.mkdir(parents=True)
        cfg2 = fx.run / "status.json"
        shutil.copy(fx.config, cfg2)
        argv = fx.args()
        argv[argv.index("--config") + 1] = str(cfg2)
        before = cfg2.read_bytes()
        rc, out, err = call(argv)
        self.assertEqual(rc, 1)
        self.assertIn("inputs and outputs must be separate", err)
        self.assertEqual(cfg2.read_bytes(), before)

    def test_output_inside_inputs_refused(self):
        fx = self.fixture_a()
        argv = fx.args()
        argv[argv.index("--out-root") + 1] = str(fx.files["m1_table"].parent)
        before = fx.snapshot()
        rc, out, err = call(argv)
        self.assertEqual(rc, 1)
        self.assertEqual(fx.snapshot(), before)
        self.assertFalse((fx.files["m1_table"].parent / RUN_ID).exists())


# ============================================================================ ingressi veri (sola lettura)

class TestRealInputs(unittest.TestCase):
    def setUp(self):
        cfg = json.loads((ROOT / NS.DEFAULT_CONFIG).read_text())
        if not all((REAL_ROOT / v["path"]).is_file() for v in cfg["inputs"].values()):
            skip_missing(self, "real inputs")

    def test_check_inputs_real(self):
        rc, out, err = call(["check-inputs", "--input-root", str(REAL_ROOT)])
        self.assertEqual(rc, 0, out + err)

    def test_run_real_reproduces_accepted_table(self):
        with tempfile.TemporaryDirectory(prefix="n1-subjects-real-") as tmp:
            rc, out, err = call(["run", "--input-root", str(REAL_ROOT), "--out-root", tmp, "--date", "20261024"])
            self.assertEqual(rc, 0, out + err)
            pub = json.loads((Path(tmp) / RUN_ID / "public" / "n1_subject_sensitivity_public.json").read_text())
            self.assertEqual(pub["accepted_m1_table_reproduced"]["cells_checked"], {"main": 56, "without the dev subjects": 56})
            got = {a: r["full"]["delta_acer_exact"] for a, r in pub["analyzers"].items()}
            # dalla tabella M1 accettata: ACER@dev − ACER@0.5 (conteggi su 150 + 150)
            self.assertEqual(got, {"livenessnet__nuaa": "-1/30", "attacknet_v1__nuaa": "-1/300", "attacknet_v2_1__nuaa": "-4/75",
                                   "attacknet_v2_2__nuaa": "1/30"})
            for r in pub["analyzers"].values():
                self.assertEqual((r["leave_one_subject_out"]["n_exclusions"], r["leave_one_subject_out"]["n_na"]), (16, 0))


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
                               for c in (TestKnownAnswers, TestTableReproduction, TestInputs, TestOutputs, TestRealInputs, TestSuite))
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
