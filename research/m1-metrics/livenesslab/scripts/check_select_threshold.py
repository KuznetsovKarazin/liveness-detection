"""
Test di regressione di scripts/select_threshold.py su dati sintetici, senza rete, senza immagini e senza inferenza.

Crea in una cartella temporanea un piccolo mondo fittizio (manifest del dev, riepilogo, rapporto di separazione, cache
del dev sintetica, manifest e cache C1 fittizi) e una cartella dati vuota: le cartelle NUAA mancano, quindi select e
apply girano in modalità sintetica con la sola regola del nome e il manifest C1. Controlla che il flusso regolare esca
con 0 e che ogni input difettoso esca con 1 e un messaggio `ERROR:` (mai un traceback). Non legge né scrive results/.

  .venv/bin/python scripts/check_select_threshold.py            # uscita 1 se un test fallisce
  .venv/bin/python scripts/check_select_threshold.py --verbose  # anche il log di unittest

Libreria standard (unittest, tempfile, subprocess); il modulo in prova usa numpy e scikit-learn solo dove già li usa.
"""
import argparse
import csv
import hashlib
import importlib.util
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

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "select_threshold.py"
ELIGIBLE = ["livenessnet__nuaa", "attacknet_v1__nuaa", "attacknet_v2_1__nuaa", "attacknet_v2_2__nuaa"]
COLUMNS = ["file", "label", "subject", "in_training_sample", "sha256", "bytes"]


def load_module():
    spec = importlib.util.spec_from_file_location("select_threshold_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ST = load_module()


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fake_hash(i: int) -> str:
    return hashlib.sha256(f"image-{i}".encode()).hexdigest()


class World:
    """Mondo sintetico in una cartella temporanea: dati vuoti (niente NUAA), risultati fittizi, uscite fuori dai risultati."""

    def __init__(self, base: Path):
        self.base = base
        self.data = base / "data"; self.data.mkdir()
        self.results = base / "results"
        (self.results / "c1").mkdir(parents=True); (self.results / "eval").mkdir(); (self.results / "m1").mkdir()
        self.dev = base / "dev"; self.dev.mkdir()
        self.out = base / "out"; self.out.mkdir()
        # dev: due soggetti, sessioni 01/02; 12 bona fide e 20 attacchi
        self.rows = []
        for i in range(32):
            subj = "0901" if i % 2 else "0908"
            lab = 0 if i < 12 else 1
            name = f"{subj}_{subj}_01_00_0{1 + i % 2}_{i}.jpg"
            self.rows.append({"file": ("real/" if lab == 0 else "attack/") + name, "label": str(lab), "subject": subj,
                              "in_training_sample": str(i % 3 != 0 and 1 or 0), "sha256": fake_hash(i), "bytes": str(1000 + i)})
        # test C1 fittizio: 6 bona fide e 6 attacchi di altri soggetti, sessione 03, più 2 attacchi dei soggetti del dev
        self.c1_rows = []
        for i in range(14):
            subj = "0901" if i >= 12 else "0905"
            lab = 0 if i < 6 else 1
            name = f"{subj}_{subj}_00_01_03_{i}.jpg"
            self.c1_rows.append({"file": ("real/" if lab == 0 else "attack/") + name, "label": str(lab), "subject": subj,
                                 "session": "00", "sha256": fake_hash(100 + i), "bytes": "2000"})
        self.write_all()

    # --- scrittura
    def write_csv(self, p, rows, cols):
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n"); w.writeheader(); w.writerows(rows)

    def write_all(self, rows=None, summary_patch=None, separation_patch=None):
        rows = self.rows if rows is None else rows
        self.write_csv(self.man, rows, COLUMNS)
        h = sha(self.man)
        summ = {"manifest_sha256": h, "n": len(rows), "n_bona_fide": sum(r["label"] == "0" for r in rows),
                "n_attack": sum(r["label"] == "1" for r in rows),
                "n_in_training_sample": sum(r["in_training_sample"] == "1" for r in rows),
                "validation_subjects": sorted({r["subject"] for r in rows})}
        summ.update(summary_patch or {})
        self.summary.write_text(json.dumps(summ, indent=1), encoding="utf-8")
        sep = {"passed": True, "problems": [], "dev_manifest_sha256": h}
        sep.update(separation_patch or {})
        self.separation.write_text(json.dumps(sep, indent=1), encoding="utf-8")
        self.write_csv(self.c1_manifest, self.c1_rows, ["file", "label", "subject", "session", "sha256", "bytes"])
        # punteggi deterministici: attacchi alti, bona fide bassi, con sovrapposizione
        fps = {a: f"fp-{a}" for a in ELIGIBLE}
        dev_scores = {a: {r["file"]: {"y": int(r["label"]), "s": round(((j * 7 + k * 3) % 10) / 20 + (0.45 if r["label"] == "1" else 0.05), 6)}
                          for j, r in enumerate(self.rows)} for k, a in enumerate(ELIGIBLE)}
        self.cache.write_text(json.dumps({"synthetic": True, "scores": dev_scores, "fingerprints": fps}, indent=1), encoding="utf-8")
        c1_scores = {a: {r["file"]: {"y": int(r["label"]), "s": round(((j * 5 + k) % 10) / 10, 6)} for j, r in enumerate(self.c1_rows)}
                     for k, a in enumerate(ELIGIBLE)}
        c1_scores["minifasnet"] = {r["file"]: {"y": int(r["label"]), "s": 0.5} for r in self.c1_rows}
        self.c1_cache.write_text(json.dumps({"scores": c1_scores, "fingerprints": {**fps, "minifasnet": "fp-m"}}, indent=1), encoding="utf-8")

    @property
    def man(self): return self.dev / "nuaa_dev_manifest.csv"
    @property
    def summary(self): return self.dev / "nuaa_dev_summary.json"
    @property
    def separation(self): return self.dev / "nuaa_dev_separation.json"
    @property
    def cache(self): return self.out / "dev_cache.json"
    @property
    def c1_manifest(self): return self.results / "c1" / "nuaa_manifest.csv"
    @property
    def c1_cache(self): return self.results / "eval" / "nuaa.json"
    @property
    def threshold(self): return self.out / "threshold.json"

    # --- esecuzione
    def run(self, *args):
        env = dict(os.environ, LIVENESSLAB_DATA_DIR=str(self.data), LIVENESSLAB_RESULTS_DIR=str(self.results))
        r = subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
        return r.returncode, r.stdout, r.stderr

    def select(self, cache=None, criterion="eer"):
        return self.run("select", "--cache", str(cache or self.cache), "--manifest", str(self.man), "--summary", str(self.summary),
                        "--separation", str(self.separation), "--c1-manifest", str(self.c1_manifest), "--criterion", criterion,
                        "--out", str(self.threshold), "--synthetic")

    def apply(self, threshold=None, expected_c1=None):
        # la cache C1 fittizia non è quella di C1: il suo SHA-256 atteso si dichiara esplicitamente
        return self.run("apply", "--threshold", str(threshold or self.threshold), "--c1-cache", str(self.c1_cache),
                        "--c1-manifest", str(self.c1_manifest), "--manifest", str(self.man), "--summary", str(self.summary),
                        "--separation", str(self.separation), "--dev-cache", str(self.cache), "--out-dir", str(self.out),
                        "--expected-c1-cache-sha256", expected_c1 or sha(self.c1_cache))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="check_select_threshold_"))
        self.w = World(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def assert_ok(self, res):
        rc, out, err = res
        self.assertEqual(rc, 0, f"expected exit 0, got {rc}; stderr:\n{err[-4000:]}")

    def assert_refused(self, res, fragment=""):
        rc, out, err = res
        # conta solo un traceback non gestito dello script: i blocchi "Exception ignored in ..." che l'interprete può
        # stampare alla chiusura (garbage collection, sottoprocessi) non dipendono dallo script e non sono un errore
        own = re.sub(r"^Exception ignored [^\n]*\n(?:Traceback \(most recent call last\):\n(?:[ \t][^\n]*\n)*)?(?![ \t]|Traceback )[^\n]*\n?", "", err, flags=re.M)
        self.assertNotIn("Traceback (most recent call last)", own, f"unhandled traceback instead of a clean error; stderr:\n{err[-4000:]}")
        self.assertEqual(rc, 1, f"expected exit 1, got {rc}; stderr:\n{err[-4000:]}")
        self.assertIn("ERROR:", err, f"no ERROR: line; stderr:\n{err[-4000:]}")
        if fragment:
            self.assertIn(fragment, err, f"expected {fragment!r} in stderr:\n{err[-4000:]}")


class TestFlow(Base):
    """Flusso regolare: select e apply sintetici escono con 0, senza cartelle NUAA lo dicono, la tabella ha hash e sensibilità."""

    def test_select_apply_ok(self):
        rc, out, err = self.w.select()
        self.assertEqual(rc, 0, err)
        self.assertIn("name rule", out)
        thr = json.loads(self.w.threshold.read_text())
        self.assertIs(thr["synthetic"], True)
        self.assertEqual(set(thr["analyzers"]), set(ELIGIBLE))
        rc, out, err = self.w.apply()
        self.assertEqual(rc, 0, err)
        md = (self.w.out / "nuaa_test_dev_threshold_table.md").read_text()
        self.assertIn(f"threshold.json sha256 `{sha(self.w.threshold)}`", md)
        self.assertIn("## Sensitivity (not the main result)", md)
        self.assertIn("2 attacks and 0 bona fide removed", md)
        self.assertIn("No score inversion is applied", md)
        # nessun identificativo dei soggetti del dev nella tabella: solo "the two dev subjects"; gli SHA-256 e il commit
        # sono esadecimali e possono contenere per caso "0901" o "0908", quindi conta solo un identificativo isolato
        self.assertIn("the two dev subjects", md)
        for subj in ("0901", "0908"):
            self.assertIsNone(re.search(rf"(?<![0-9a-f]){subj}(?![0-9a-f])", md), f"dev subject {subj} in the table:\n{md}")
        # cache C1 diversa da quella attesa (default: il valore registrato alla consegna di C1): rifiutata
        self.assert_refused(self.w.run("apply", "--threshold", str(self.w.threshold), "--c1-cache", str(self.w.c1_cache),
                                       "--c1-manifest", str(self.w.c1_manifest), "--manifest", str(self.w.man), "--summary", str(self.w.summary),
                                       "--separation", str(self.w.separation), "--dev-cache", str(self.w.cache), "--out-dir", str(self.w.out)),
                            "differs from the expected C1 cache")
        self.assert_refused(self.w.apply(expected_c1="0" * 64), "differs from the expected C1 cache")

    def test_non_informative_flag(self):
        # funzione pura: AUC del dev < 0,5 oppure EER del dev > 0,5, con il motivo
        self.assertEqual(ST.dev_flag({"dev": {"auc": 0.8, "eer": 0.2}}), "")
        self.assertEqual(ST.dev_flag({"dev": {"auc": 0.45, "eer": 0.48}}), "non-informative (dev AUC < 0.5)")
        self.assertEqual(ST.dev_flag({"dev": {"auc": 0.55, "eer": 0.52}}), "non-informative (dev EER > 0.5)")
        self.assertEqual(ST.dev_flag({"dev": {"auc": 0.3, "eer": 0.7}}), "non-informative (dev AUC < 0.5, dev EER > 0.5)")
        self.assertEqual(ST.dev_flag({"dev": {"auc": 0.5, "eer": 0.5}}), "")
        # flusso: punteggi dev invertiti per un analizzatore -> riga segnalata nella tabella, nessuna inversione
        c = json.loads(self.w.cache.read_text())
        for v in c["scores"]["attacknet_v1__nuaa"].values():
            v["s"] = round(1.0 - v["s"], 6)
        self.w.cache.write_text(json.dumps(c, indent=1), encoding="utf-8")
        self.assert_ok(self.w.select())
        self.assert_ok(self.w.apply())
        md = (self.w.out / "nuaa_test_dev_threshold_table.md").read_text()
        row = next(l for l in md.splitlines() if l.startswith("| attacknet_v1__nuaa |"))
        self.assertIn("non-informative (dev AUC < 0.5", row)
        other = next(l for l in md.splitlines() if l.startswith("| livenessnet__nuaa |"))
        self.assertNotIn("non-informative", other)

    def test_apcer10(self):
        self.assert_ok(self.w.select(criterion="apcer10"))
        self.assert_ok(self.w.apply())


class TestSelectRefusals(Base):
    """select rifiuta manifest, riepiloghi, rapporti e cache difettosi con ERROR: e uscita 1."""

    def mutate_rows(self, fn, **kw):
        rows = [dict(r) for r in self.w.rows]
        fn(rows)
        self.w.write_all(rows, **kw)

    def test_row_duplicated(self):
        self.mutate_rows(lambda rs: rs.append(dict(rs[3])))
        self.assert_refused(self.w.select(), "duplicated keys")

    def test_row_removed_counts_stale(self):
        self.mutate_rows(lambda rs: rs.pop(), summary_patch={"n": 32})
        self.assert_refused(self.w.select(), "does not certify")

    def test_label_flipped(self):
        self.mutate_rows(lambda rs: rs[20].__setitem__("label", "0"))
        self.assert_refused(self.w.select(), "labels not consistent")

    def test_subject_changed(self):
        self.mutate_rows(lambda rs: rs[5].__setitem__("subject", "0001"))
        self.assert_refused(self.w.select(), "subjects different")

    def test_dotdot_key(self):
        self.mutate_rows(lambda rs: rs[0].__setitem__("file", "real/../../etc/x.jpg"))
        self.assert_refused(self.w.select(), "invalid keys")

    def test_bytes_malformed(self):
        self.mutate_rows(lambda rs: rs[0].__setitem__("bytes", "12a"))
        self.assert_refused(self.w.select(), "malformed bytes")

    def test_test_session_in_manifest(self):
        self.mutate_rows(lambda rs: rs[0].__setitem__("file", "real/0901_0901_01_00_03_0.jpg"))
        self.assert_refused(self.w.select())

    def test_separation_passed_with_problems(self):
        self.w.write_all(separation_patch={"problems": ["something failed"]})
        self.assert_refused(self.w.select(), "does not certify")

    def test_separation_not_passed(self):
        self.w.write_all(separation_patch={"passed": "true"})
        self.assert_refused(self.w.select(), "does not certify")

    def test_separation_duplicate_key(self):
        txt = self.w.separation.read_text().replace('"passed": true', '"passed": false, "passed": true')
        self.w.separation.write_text(txt)
        self.assert_refused(self.w.select(), "duplicated JSON key")

    def test_summary_invalid_json(self):
        self.w.summary.write_text("{oops")
        self.assert_refused(self.w.select(), "invalid JSON")

    def test_summary_missing(self):
        self.w.summary.unlink()
        self.assert_refused(self.w.select(), "missing")

    def cache_variant(self, fn):
        c = json.loads(self.w.cache.read_text()); fn(c)
        p = self.w.out / "bad_cache.json"; p.write_text(json.dumps(c))
        return self.w.select(cache=p)

    def test_cache_ineligible_analyzer(self):
        self.assert_refused(self.cache_variant(lambda c: c["scores"].__setitem__("lbp", c["scores"]["livenessnet__nuaa"])), "not eligible")

    def test_cache_y_string(self):
        k = self.w.rows[0]["file"]
        self.assert_refused(self.cache_variant(lambda c: c["scores"]["livenessnet__nuaa"][k].__setitem__("y", "0")), "labels that are not")

    def test_cache_s_out_of_range(self):
        k = self.w.rows[0]["file"]
        self.assert_refused(self.cache_variant(lambda c: c["scores"]["livenessnet__nuaa"][k].__setitem__("s", 1.5)), "valid score")

    def test_cache_entry_missing_s(self):
        k = self.w.rows[0]["file"]
        self.assert_refused(self.cache_variant(lambda c: c["scores"]["livenessnet__nuaa"][k].pop("s")), "valid score")

    def test_cache_nan(self):
        p = self.w.out / "nan_cache.json"
        p.write_text(self.w.cache.read_text().replace('"s": 0.05', '"s": NaN', 1))
        self.assert_refused(self.w.select(cache=p), "invalid JSON")

    def test_cache_test_key(self):
        k = self.w.c1_rows[0]["file"]
        self.assert_refused(self.cache_variant(lambda c: c["scores"]["livenessnet__nuaa"].__setitem__(k, {"y": 0, "s": 0.1})),
                            "official test split or of the C1 manifest")

    def test_cache_session_03_key(self):
        self.assert_refused(self.cache_variant(
            lambda c: c["scores"]["livenessnet__nuaa"].__setitem__("attack/0009_0009_00_00_03_7.jpg", {"y": 1, "s": 0.9})),
            "official test split or of the C1 manifest")


class TestApplyRefusals(Base):
    """apply rifiuta un threshold.json modificato o incompleto."""

    def setUp(self):
        super().setUp()
        rc, out, err = self.w.select()
        self.assertEqual(rc, 0, err)
        self.thr = json.loads(self.w.threshold.read_text())

    def variant(self, fn, raw=None):
        t = json.loads(json.dumps(self.thr)); fn(t)
        p = self.w.out / "threshold_variant.json"
        txt = json.dumps(t, indent=1)
        p.write_text(raw(txt) if raw else txt)
        return self.w.apply(p)

    def test_control(self):
        self.assert_ok(self.variant(lambda t: None))

    def test_value_edited(self):
        e = self.thr["analyzers"]["attacknet_v1__nuaa"]
        edited = round(e["value"] - 0.013, 6) if e["value"] > 0.5 else round(e["value"] + 0.013, 6)
        self.assert_refused(self.variant(lambda t: t["analyzers"]["attacknet_v1__nuaa"].__setitem__("value", edited)), "recomputation")

    def test_dev_figure_edited(self):
        self.assert_refused(self.variant(lambda t: t["analyzers"]["attacknet_v1__nuaa"]["dev"].__setitem__("auc", 0.99)), "recomputation")

    def test_synthetic_missing(self):
        self.assert_refused(self.variant(lambda t: t.pop("synthetic")), "synthetic")

    def test_synthetic_not_bool(self):
        self.assert_refused(self.variant(lambda t: t.__setitem__("synthetic", 1)), "synthetic")

    def test_synthetic_false_on_synthetic_cache(self):
        self.assert_refused(self.variant(lambda t: t.__setitem__("synthetic", False)))

    def test_criterion_unknown(self):
        self.assert_refused(self.variant(lambda t: t.__setitem__("criterion", "acer")), "criterion")

    def test_value_out_of_range(self):
        self.assert_refused(self.variant(lambda t: t["analyzers"]["livenessnet__nuaa"].__setitem__("value", -0.1)), "[0, 1]")

    def test_analyzer_missing(self):
        self.assert_refused(self.variant(lambda t: t["analyzers"].pop("attacknet_v2_2__nuaa")), "eligible")

    def test_analyzer_extra(self):
        self.assert_refused(self.variant(lambda t: t["analyzers"].__setitem__("lbp", t["analyzers"]["livenessnet__nuaa"])), "eligible")

    def test_duplicate_key(self):
        self.assert_refused(self.variant(lambda t: None, raw=lambda s: s.replace('"criterion": "eer"', '"criterion": "apcer10", "criterion": "eer"', 1)),
                            "duplicated JSON key")

    def test_dev_cache_changed(self):
        c = json.loads(self.w.cache.read_text())
        c["scores"]["livenessnet__nuaa"][self.w.rows[0]["file"]]["s"] = 0.33
        self.w.cache.write_text(json.dumps(c))
        self.assert_refused(self.w.apply(), "not found with sha256")


class TestFunctions(unittest.TestCase):
    """Funzioni pure del modulo, senza sottoprocessi."""

    def test_key_problem(self):
        self.assertEqual(ST.key_problem("real/0001_0001_00_00_01_0.jpg"), "")
        for bad in ("real/../x.jpg", "real/..", "attack/a/b.jpg", "../x.jpg", "fake/x.jpg", "real/.x.jpg", "real/x.txt", "real\\x.jpg", None):
            self.assertTrue(ST.key_problem(bad), bad)

    def test_session_of(self):
        self.assertEqual(ST.session_of("SSSS_SSSS_xx_xx_03_n.jpg"), "03")
        self.assertEqual(ST.session_of("short_name.jpg"), "")

    def test_repo_state_outside_git(self):
        saved = ST.ROOT
        with tempfile.TemporaryDirectory() as d:
            ST.ROOT = Path(d)
            try:
                self.assertEqual(ST.git("rev-parse", "HEAD"), "n/d")
                self.assertEqual(ST.repo_state(), {"commit": "n/d", "dirty": False})
            finally:
                ST.ROOT = saved

    def test_summary_problems_counts(self):
        rows = [{"file": "real/0901_0901_00_00_01_0.jpg", "label": "0", "subject": "0901", "in_training_sample": "1"}]
        ok = {"manifest_sha256": "h", "n": 1, "n_bona_fide": 1, "n_attack": 0, "n_in_training_sample": 1, "validation_subjects": ["0901"]}
        self.assertEqual(ST.summary_problems(rows, ok, "h"), [])
        self.assertTrue(ST.summary_problems(rows, {**ok, "n": 2}, "h"))
        self.assertTrue(ST.summary_problems(rows, {**ok, "n": True}, "h"))
        self.assertTrue(ST.summary_problems(rows, ok, "other"))

    def test_load_json_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            p.write_text('{"a": 1, "a": 2}')
            with self.assertRaises(SystemExit) as cm, redirect_stderr():
                ST.load_json(p)
            self.assertEqual(cm.exception.code, 1)


class redirect_stderr:
    """Silenzia lo stderr dei messaggi ERROR: attesi nei test in processo."""

    def __enter__(self):
        self.saved = sys.stderr; sys.stderr = io.StringIO(); return self

    def __exit__(self, *exc):
        sys.stderr = self.saved
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Test di regressione di select_threshold.py su dati sintetici")
    ap.add_argument("--verbose", action="store_true", help="stampa il log completo di unittest")
    args = ap.parse_args()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c)
                               for c in (TestFunctions, TestFlow, TestSelectRefusals, TestApplyRefusals))
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    if args.verbose or not result.wasSuccessful():
        print(log.getvalue())
    print(f"tests run: {result.testsRun}, failures: {len(result.failures)}, errors: {len(result.errors)}, skipped: {len(result.skipped)}")
    print("RESULT: " + ("PASS" if result.wasSuccessful() else "FAIL"))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
