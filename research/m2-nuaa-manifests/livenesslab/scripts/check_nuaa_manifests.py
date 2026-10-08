"""
Test di scripts/nuaa_manifests.py su dati sintetici, senza rete, senza immagini e senza inferenza.

Parte offline (sempre): parser dei nomi NUAA e delle righe delle liste ufficiali, correzione della sessione del
manifest C1 (versione 1 -> versione 2), conteggi pubblici e privati, scansione del contenuto pubblico, struttura del
file delle regole, e il controllo delle sovrapposizioni su manifest SINTETICI con sovrapposizioni note: un caso pulito
che deve passare (codice 0, sovrapposizione di soggetto solo riportata) e un caso guasto per ogni violazione (chiave
duplicata fra split, stesso hash con nome diverso, stesso soggetto fra train ed early stopping, stesso soggetto e
sessione fra train e test, duplicato interno per hash e per chiave, sottoinsieme rotto, intersezione dichiarata
diversa), ciascuno rilevato dalla regola attesa con codice 1 e un messaggio `ERROR:` (mai un traceback). Regole e
manifest malformati sono rifiutati con un errore pulito.

Parte sui dati reali (solo se results/m2 contiene i manifest scritti da build): check-overlap rieseguito in una
cartella temporanea con le regole versionate; si salta in modo pulito se i manifest mancano (con --require-real il
salto diventa un errore).

  .venv/bin/python scripts/check_nuaa_manifests.py            # uscita 1 se un test fallisce
  .venv/bin/python scripts/check_nuaa_manifests.py --verbose  # anche il log di unittest

Libreria standard (unittest, tempfile, subprocess).
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "nuaa_manifests.py"
RULES = ROOT / "delivery" / "m2" / "nuaa_overlap_rules.json"
REQUIRE_REAL = False
REAL_DIR = None          # cartella con manifests/ (e facoltativi riepilogo e aggregato): default results/m2
REAL_AGGREGATE = None    # aggregato pubblico (default <REAL_DIR>/nuaa_manifests_aggregate.json, oppure quello del pacchetto)
C1_V1 = None             # manifest C1 versione 1 (default results/c1/nuaa_manifest.csv)


def load_module():
    spec = importlib.util.spec_from_file_location("nuaa_manifests_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NM = load_module()


def fake_hash(tag) -> str:
    return hashlib.sha256(f"image-{tag}".encode()).hexdigest()


def row(label: int, subject: str, session: str, pic: int, glasses="00", pos="00", sha=None, split="train") -> dict:
    """Riga sintetica con nome NUAA valido; lo hash dipende dalla chiave salvo indicazione contraria."""
    pic += 600                       # numeri oltre 499: nessun nome coincide con un'immagine reale di NUAA
    name = f"{subject}_{subject}_{glasses}_{pos}_{session}_{pic}.jpg"
    key = f"{'real' if label == 0 else 'attack'}/{name}"
    return {"key": key, "label": label, "subject": subject, "glasses": glasses, "pos": pos, "session": session,
            "pic": str(pic), "official_split": split, "sha256": sha or fake_hash(key), "bytes": 100 + pic}


def clean_world() -> dict:
    """Manifest sintetici puliti: train (soggetti 0001, 0102, sessioni 01/02), early stopping (soggetto 0003),
    dev = early stopping + un'immagine in più dello stesso soggetto, test (sessione 03, soggetti 0001 e 0004:
    0001 è condiviso con il train, sovrapposizione di soggetto ammessa e solo riportata), c1 sottoinsieme del test."""
    train = [row(0, "0001", "01", 0), row(1, "0001", "01", 1), row(0, "0102", "02", 0), row(1, "0102", "02", 1)]
    es = [row(0, "0003", "01", 0), row(1, "0003", "02", 1)]
    dev = es + [row(0, "0003", "02", 7)]
    test = [row(0, "0001", "03", 0, split="test"), row(1, "0001", "03", 1, split="test"),
            row(0, "0004", "03", 0, split="test"), row(1, "0004", "03", 1, split="test")]
    c1 = [test[0], test[3]]
    return {"train": train, "es": es, "dev": dev, "test": test, "c1": c1}


SYN_RULES = {
    "manifests": {n: f"{n}.csv" for n in ("train", "es", "dev", "test", "c1")},
    "rules": [
        {"id": "S01-internal", "kind": "no_internal_duplicates", "manifests": "*", "by": ["key", "hash"], "severity": "must"},
        {"id": "S02-train-vs-es", "kind": "disjoint", "a": "train", "b": ["es", "dev"], "by": ["key", "hash", "subject"], "severity": "must"},
        {"id": "S03-train-vs-test", "kind": "disjoint", "a": ["train", "es", "dev"], "b": ["test", "c1"],
         "by": ["key", "hash", "subject_session"], "severity": "must"},
        {"id": "S04-subjects", "kind": "disjoint", "a": ["train", "es", "dev"], "b": ["test", "c1"], "by": ["subject"], "severity": "report"},
        {"id": "S05-c1-in-test", "kind": "subset", "a": "c1", "b": "test", "by": ["key", "hash"], "severity": "must"},
        {"id": "S06-es-in-dev", "kind": "subset", "a": "es", "b": "dev", "by": ["key"], "severity": "must"},
        {"id": "S07-reuse", "kind": "intersection", "a": "dev", "b": "es", "by": ["key"], "expected": 2, "severity": "must"},
    ],
}


def write_csv(p: Path, rows, encoding="utf-8"):
    with p.open("w", newline="", encoding=encoding) as f:
        w = csv.DictWriter(f, fieldnames=NM.COLUMNS, lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def declared(p: Path) -> dict:
    """Righe e SHA-256 attesi di un manifest, come li dichiara il file delle regole (valori finti se il file manca)."""
    if not p.is_file():
        return {"file": p.name, "expected_rows": 1, "expected_sha256": fake_hash("missing")}
    return {"file": p.name, "expected_rows": len(p.read_text(encoding="utf-8-sig").splitlines()) - 1, "expected_sha256": sha_file(p)}


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_world(d: Path, world: dict, rules: dict = None, expect: dict = None) -> Path:
    """Scrive i manifest sintetici e le regole; i manifest dichiarati come semplice nome di file ricevono righe e
    SHA-256 attesi calcolati sui file scritti, salvo i valori forzati in `expect` (casi di troncamento)."""
    man = d / "manifests"
    man.mkdir(parents=True, exist_ok=True)
    for n, rows in world.items():
        write_csv(man / f"{n}.csv", rows)
    doc = json.loads(json.dumps(rules or SYN_RULES))
    for n, m in doc["manifests"].items():
        if isinstance(m, str):
            doc["manifests"][n] = (expect or {}).get(n) or declared(man / m)
    rp = d / "rules.json"
    rp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return rp


def run_overlap(d: Path, world: dict, rules: dict = None, expect: dict = None):
    rp = write_world(d, world, rules, expect)
    out = d / "out"
    r = subprocess.run([sys.executable, str(SCRIPT), "check-overlap", "--rules", str(rp), "--manifests-dir", str(d / "manifests"),
                        "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    rep = json.loads((out / "overlap_report.json").read_text()) if (out / "overlap_report.json").exists() else None
    return r, rep


def verdict(rep, rid):
    return next(v for v in rep["rules"] if v["id"] == rid)


def failed_checks(rep, rid):
    return [c for c in verdict(rep, rid)["checks"] if c["outcome"] == "FAIL"]


class quiet:
    """Silenzia stdout e stderr dei messaggi attesi nei test in processo."""

    def __enter__(self):
        self.saved = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        return self

    def __exit__(self, *exc):
        sys.stdout, sys.stderr = self.saved
        return False


# ----------------------------------------------------------------------------- parser e correzione C1

class TestParser(unittest.TestCase):
    def test_parse_name(self):
        self.assertEqual(NM.parse_name("0004_0004_01_06_03_618.jpg"),
                         {"subject": "0004", "glasses": "01", "pos": "06", "session": "03", "pic": "618"})

    def test_parse_name_refuses(self):
        for bad in ("0004_0005_01_06_03_618.jpg",       # prefisso della cartella diverso dall'ID
                    "0004_01_06_03_618.jpg",            # nome originale senza prefisso
                    "0004_0004_01_06_03_118.png",       # estensione
                    "0004_0004_1_06_03_618.jpg",        # campo a una cifra
                    "real/0004_0004_01_06_03_618.jpg", "", None):
            with self.assertRaises(ValueError, msg=bad):
                NM.parse_name(bad)

    def test_parse_list_line(self):
        line = "/kaggle/input/nuaaaa/raw/ImposterRaw/0102/0102_01_00_01_600.jpg \n"
        self.assertEqual(NM.parse_list_line(line), ("ImposterRaw", "0102_0102_01_00_01_600.jpg"))
        # nome su disco = cartella + "_" + nome originale, e il parser dei nomi lo rilegge con il soggetto della cartella
        self.assertEqual(NM.parse_name(NM.parse_list_line(line)[1])["session"], "01")

    def test_parse_list_line_refuses(self):
        for bad in ("/kaggle/input/nuaaaa/raw/ClientRaw/0001/0102_00_00_01_0.jpg",   # cartella diversa dall'ID
                    "/other/raw/ClientRaw/0001/0001_00_00_01_700.jpg",
                    "/kaggle/input/nuaaaa/raw/ClientRaw/0001/0001_00_00_0.jpg"):
            with self.assertRaises(ValueError, msg=bad):
                NM.parse_list_line(bad)

    def test_read_official_list(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "client_train_raw.txt"
            p.write_text("/kaggle/input/nuaaaa/raw/ClientRaw/0001/0001_00_00_01_700.jpg\n\n"
                         "/kaggle/input/nuaaaa/raw/ClientRaw/0001/0001_00_00_01_705.jpg\n")
            self.assertEqual(NM.read_official_list(p, 0), ["0001_0001_00_00_01_700.jpg", "0001_0001_00_00_01_705.jpg"])
            for content in ("/kaggle/input/nuaaaa/raw/ImposterRaw/0001/0001_00_00_01_700.jpg\n",           # classe sbagliata
                            "/kaggle/input/nuaaaa/raw/ClientRaw/0001/0001_00_00_01_700.jpg\n" * 2):       # riga duplicata
                p.write_text(content)
                with self.assertRaises(SystemExit) as cm, quiet():
                    NM.read_official_list(p, 0)
                self.assertEqual(cm.exception.code, 1)

    def test_c1_session_correction(self):
        # versione 1: la colonna session era il terzo campo (glasses); la versione 2 rilegge il quinto campo
        v1 = [{"file": "real/0004_0004_01_06_03_800.jpg", "label": "0", "subject": "0004", "session": "01", "sha256": fake_hash(1), "bytes": "10"},
              {"file": "attack/0102_0102_00_01_03_7.jpg", "label": "1", "subject": "0102", "session": "00", "sha256": fake_hash(2), "bytes": "11"}]
        rows, stats = NM.c1_correction(v1)
        self.assertEqual([r["session"] for r in rows], ["03", "03"])
        self.assertEqual([r["glasses"] for r in rows], ["01", "00"])
        self.assertEqual([r["pos"] for r in rows], ["06", "01"])
        self.assertEqual(stats["v1_session_equals_glasses"], 2)
        self.assertEqual(stats["v1_session_equals_session"], 0)
        self.assertEqual(stats["session_values"], {"03": 2})
        self.assertEqual(stats["problems"], [])
        self.assertEqual([(r["key"], r["sha256"], r["bytes"]) for r in rows], [(x["file"], x["sha256"], x["bytes"]) for x in v1])

    def test_c1_correction_reports_inconsistencies(self):
        v1 = [{"file": "real/0004_0004_01_06_03_800.jpg", "label": "1", "subject": "0004", "session": "01", "sha256": fake_hash(1), "bytes": "1"},
              {"file": "real/0004_0004_01_06_03_801.jpg", "label": "0", "subject": "0005", "session": "01", "sha256": fake_hash(2), "bytes": "1"},
              {"file": "real/../x.jpg", "label": "0", "subject": "0004", "session": "01", "sha256": fake_hash(3), "bytes": "1"}]
        _, stats = NM.c1_correction(v1)
        self.assertEqual(len(stats["problems"]), 3)

    def test_counts_public_private(self):
        w = clean_world()
        pub, priv = NM.counts(w["train"], private=False), NM.counts(w["train"], private=True)
        self.assertEqual((pub["n"], pub["n_bona_fide"], pub["n_attack"], pub["n_subjects"]), (4, 2, 2, 2))
        self.assertEqual(pub["by_session"], {"01": {"bona_fide": 1, "attack": 1}, "02": {"bona_fide": 1, "attack": 1}})
        self.assertNotIn("by_subject", pub)
        self.assertEqual(priv["by_subject"]["0001"], {"bona_fide": 1, "attack": 1})

    def test_public_scan(self):
        self.assertEqual(NM.public_problems('{"n": 2, "sha": "ab0102cd", "date": "2026-10-06", "v": "01020"}', {"0102", "0108"}), [])
        self.assertEqual(NM.public_problems("subjects 0102 and 0108", {"0102", "0108"}), ["subject id 0102", "subject id 0108"])
        self.assertTrue(NM.public_problems('{"f": "real/0102_0102_00_00_01_600.jpg"}', {"0102"}))
        self.assertTrue(any("absolute path" in b for b in NM.public_problems("/" + "Users/someone/x", set())))
        self.assertTrue(any("IP address" in b for b in NM.public_problems("host " + ".".join(["10", "1", "2", "3"]), set())))
        self.assertEqual(NM.public_problems("host 127.0.0.1", set()), [])
        self.assertTrue(NM.public_problems("open point: T" + "BD", set(), markdown=True))
        self.assertEqual(NM.public_problems("open point: T" + "BD", set()), [])          # marker solo nei .md


# ----------------------------------------------------------------------------- regole

class TestRules(unittest.TestCase):
    def test_versioned_rules_are_valid(self):
        doc = json.loads(RULES.read_text(encoding="utf-8"))
        self.assertEqual(NM.rule_problems(doc, list(doc["manifests"])), [])
        # le regole versionate nominano esattamente i manifest scritti da build
        self.assertEqual({n: m["file"] for n, m in doc["manifests"].items()}, {n: f for n, (f, _) in NM.MANIFESTS.items()})
        self.assertEqual({n: m["expected_rows"] for n, m in doc["manifests"].items()},
                         {"official_train": 3491, "official_test": 9123, "cnn_train": 2507, "cnn_es": 493, "cnn_test": 3000,
                          "m1_dev": 574, "c1": 300})
        ids = {r["id"]: r for r in doc["rules"]}
        self.assertEqual(ids["R08-dev-es-reuse"]["expected"], 493)
        # R15: coerenza per chiave su tutti i manifest, obbligatoria
        self.assertEqual({k: ids["R15-key-consistency"][k] for k in ("kind", "manifests", "by", "severity")},
                         {"kind": "key_consistency", "manifests": "*", "by": ["key"], "severity": "must"})
        self.assertEqual(ids["R05-subjects-train-vs-test"]["severity"], "report")
        self.assertTrue(all(r["severity"] == "must" for i, r in ids.items() if i != "R05-subjects-train-vs-test"))

    def test_source_declarations_identical_in_documents(self):
        # dichiarazioni sulla fonte: stesso testo nel codice e nei documenti (a meno degli a capo); nel pacchetto pubblico
        # c'è solo il modello del rapporto sotto delivery/m2
        norm = lambda t: re.sub(r"\s+", " ", t)   # noqa: E731
        files = [p for p in (ROOT / "delivery" / "m2" / n for n in ("naming_schema.md", "README.md", "report.md.tmpl")) if p.is_file()]
        self.assertIn("report.md.tmpl", [p.name for p in files])
        for p in files:
            t = norm(p.read_text(encoding="utf-8"))
            for label, decl in NM.SOURCE_DECLARATIONS:
                self.assertIn(decl, t, f"{p.name}: {label}")
                if p.name != "report.md.tmpl":                    # nello schema e nel README come voce con etichetta
                    self.assertIn(f"- {label}: {decl}", t, f"{p.name}: {label}")
        # documentato e verificato restano distinti: il significato di glasses e pos non è mai detto verificato
        self.assertIn("not verified", NM.NOT_VERIFIED_SOURCE + " not verified")
        self.assertNotIn("glasses", NM.VERIFIED_SOURCE)

    def test_rule_problems(self):
        names = ["a", "b"]
        base = {"manifests": {"a": {"file": "a.csv", "expected_rows": 1, "expected_sha256": fake_hash("a")},
                              "b": {"file": "b.csv", "expected_rows": 1, "expected_sha256": fake_hash("b")}}}
        bad_rules = [
            {"id": "x", "kind": "disjoint", "a": "a", "b": "c", "by": ["key"], "severity": "must"},              # manifest ignoto
            {"id": "x", "kind": "overlap", "a": "a", "b": "b", "by": ["key"], "severity": "must"},               # tipo ignoto
            {"id": "x", "kind": "disjoint", "a": "a", "b": "b", "by": ["key"], "severity": "should"},            # severità ignota
            {"id": "x", "kind": "disjoint", "a": "a", "b": "b", "by": ["name"], "severity": "must"},             # dimensione ignota
            {"id": "x", "kind": "intersection", "a": "a", "b": "b", "by": ["key"], "severity": "must"},          # expected mancante
            {"id": "x", "kind": "disjoint", "a": "a", "b": "b", "by": ["key"], "severity": "must", "expected": 1},
            {"id": "x", "kind": "no_internal_duplicates", "manifests": ["z"], "by": ["hash"], "severity": "must"},
            {"id": "x", "kind": "disjoint", "a": "a", "b": "a", "by": ["key"], "severity": "must"},              # a == b
            {"id": "x", "kind": "subset", "a": ["a", "b"], "b": "b", "by": ["key"], "severity": "must"},         # b su entrambi i lati
        ]
        for r in bad_rules:
            self.assertTrue(NM.rule_problems({**base, "rules": [r]}, names), r)
        rule = {"id": "x", "kind": "disjoint", "a": "a", "b": "b", "by": ["key"], "severity": "must"}
        # manifest dichiarati senza righe o SHA-256 attesi, con zero righe attese, o con un percorso al posto del nome
        for m in ("a.csv", {"file": "a.csv", "expected_rows": 0, "expected_sha256": fake_hash("a")},
                  {"file": "a.csv", "expected_rows": 3}, {"file": "../a.csv", "expected_rows": 1, "expected_sha256": fake_hash("a")},
                  {"file": "a.csv", "expected_rows": 1, "expected_sha256": "ABC"}):
            self.assertTrue(NM.rule_problems({"manifests": {**base["manifests"], "a": m}, "rules": [rule]}, names), m)
        dup = rule
        self.assertTrue(NM.rule_problems({**base, "rules": [dup, dup]}, names))
        self.assertEqual(NM.rule_problems({**base, "rules": [dup]}, names), [])


# ----------------------------------------------------------------------------- casi sintetici a risultato noto

class TestOverlapCases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def assertFailsOn(self, world, rid, by, rules=None):
        r, rep = run_overlap(self.d, world, rules)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("ERROR:", r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertFalse(rep["passed"])
        self.assertEqual(verdict(rep, rid)["verdict"], "FAIL")
        self.assertIn(by, {c["by"] for c in failed_checks(rep, rid)})
        return rep

    def test_clean_case_passes(self):
        r, rep = run_overlap(self.d, clean_world())
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(rep["passed"])
        self.assertEqual(rep["n_must_failed"], 0)
        self.assertTrue(all(v["verdict"] == "pass" for v in rep["rules"] if v["severity"] == "must"))
        # soggetto 0001 nel train e nel test (sessioni diverse): ammesso, solo riportato
        s04 = verdict(rep, "S04-subjects")
        self.assertEqual(s04["verdict"], "reported")
        c = next(c for c in s04["checks"] if c["a"] == "train" and c["b"] == "test")
        self.assertEqual((c["n_shared"], c["rows_a"], c["rows_b"]), (1, 2, 2))
        self.assertTrue((self.d / "out" / "overlap_report.md").exists())
        # nessun identificativo di soggetto nel rapporto quando tutte le regole must passano
        txt = (self.d / "out" / "overlap_report.json").read_text() + (self.d / "out" / "overlap_report.md").read_text()
        self.assertEqual(NM.public_problems(txt, {"0001", "0102", "0003", "0004"}), [])
        # matrice: ogni coppia non ordinata, quattro dimensioni
        self.assertEqual(len(rep["matrix"]["pairs"]), 10)
        self.assertEqual(rep["matrix"]["pairs"]["train|test"]["subject"]["n_shared"], 1)
        self.assertEqual(rep["matrix"]["pairs"]["train|test"]["subject_session"]["n_shared"], 0)

    def test_duplicate_key_across_splits(self):
        w = clean_world()
        w["test"].append(dict(w["train"][0]))
        rep = self.assertFailsOn(w, "S03-train-vs-test", "key")
        self.assertIn("examples", failed_checks(rep, "S03-train-vs-test")[0])

    def test_same_hash_different_name(self):
        w = clean_world()
        w["test"].append(row(0, "0004", "03", 9, split="test", sha=w["train"][2]["sha256"]))
        rep = self.assertFailsOn(w, "S03-train-vs-test", "hash")
        by = {c["by"] for c in failed_checks(rep, "S03-train-vs-test")}
        self.assertEqual(by, {"hash"})                     # stessa immagine, nome e (soggetto, sessione) diversi

    def test_same_subject_train_vs_es(self):
        w = clean_world()
        w["es"].append(row(0, "0001", "02", 50))
        w["dev"].append(w["es"][-1])
        rep = self.assertFailsOn(w, "S02-train-vs-es", "subject",
                                 rules={**SYN_RULES, "rules": [r if r["id"] != "S07-reuse" else {**r, "expected": 3} for r in SYN_RULES["rules"]]})
        self.assertEqual({c["by"] for c in failed_checks(rep, "S02-train-vs-es")}, {"subject"})

    def test_same_subject_session_train_vs_test(self):
        w = clean_world()
        w["test"].append(row(1, "0102", "02", 99))     # immagine del train (sessione 02) finita nel test: stesso soggetto e sessione
        rep = self.assertFailsOn(w, "S03-train-vs-test", "subject_session")
        self.assertEqual({c["by"] for c in failed_checks(rep, "S03-train-vs-test")}, {"subject_session"})

    def test_internal_duplicate_hash(self):
        w = clean_world()
        w["test"].append(row(1, "0004", "03", 42, split="test", sha=w["test"][2]["sha256"]))
        rep = self.assertFailsOn(w, "S01-internal", "hash")
        c = failed_checks(rep, "S01-internal")
        self.assertEqual([(x["manifest"], x["by"], x["duplicated_values"], x["rows"]) for x in c], [("test", "hash", 1, 2)])

    def test_internal_duplicate_key(self):
        w = clean_world()
        w["train"].append(dict(w["train"][1]))
        rep = self.assertFailsOn(w, "S01-internal", "key")
        self.assertIn(("train", "key"), {(x["manifest"], x["by"]) for x in failed_checks(rep, "S01-internal")})

    def test_subset_broken(self):
        w = clean_world()
        w["c1"].append(row(0, "0005", "03", 3, split="test"))         # immagine di C1 che non è nel test
        rep = self.assertFailsOn(w, "S05-c1-in-test", "key")
        self.assertEqual({c["n_not_in_b"] for c in failed_checks(rep, "S05-c1-in-test")}, {1})

    def test_declared_intersection_differs(self):
        w = clean_world()
        w["dev"] = w["dev"][1:]                                        # un'immagine dell'early stopping non è più nel dev
        rep = self.assertFailsOn(w, "S07-reuse", "key")
        self.assertEqual(failed_checks(rep, "S07-reuse")[0]["n_shared"], 1)
        self.assertEqual(verdict(rep, "S06-es-in-dev")["verdict"], "FAIL")

    def test_report_rule_never_fails(self):
        # la stessa sovrapposizione di soggetto dichiarata "must" fa fallire, dichiarata "report" no
        w = clean_world()
        r, rep = run_overlap(self.d, w)
        self.assertEqual(r.returncode, 0)
        must = {**SYN_RULES, "rules": [r_ if r_["id"] != "S04-subjects" else {**r_, "severity": "must"} for r_ in SYN_RULES["rules"]]}
        self.assertFailsOn(w, "S04-subjects", "subject", rules=must)

    def test_in_process_matches_cli(self):
        w = clean_world()
        w["test"].append(dict(w["train"][0]))
        mans = {n: [{k: str(v) for k, v in r.items()} for r in rows] for n, rows in w.items()}
        v = {x["id"]: x["verdict"] for x in NM.evaluate_rules(mans, SYN_RULES["rules"])}
        self.assertEqual(v["S03-train-vs-test"], "FAIL")
        self.assertEqual(v["S01-internal"], "pass")

    def test_malformed_inputs_are_clean_errors(self):
        cases = []
        w = clean_world(); w["test"][0] = dict(w["test"][0], label=1)                        # etichetta contro la cartella
        cases.append((w, None))
        w = clean_world(); w["train"][0] = dict(w["train"][0], sha256="xyz")                  # hash malformato
        cases.append((w, None))
        w = clean_world(); w["train"][0] = dict(w["train"][0], key="real/../x.jpg")           # chiave non valida
        cases.append((w, None))
        cases.append((clean_world(), {**SYN_RULES, "manifests": {**SYN_RULES["manifests"], "ghost": "ghost.csv"}}))   # file mancante
        cases.append((clean_world(), {**SYN_RULES, "rules": SYN_RULES["rules"] + [
            {"id": "S99", "kind": "disjoint", "a": "train", "b": "nope", "by": ["key"], "severity": "must"}]}))       # manifest ignoto
        for world, rules in cases:
            with tempfile.TemporaryDirectory() as d:
                r, rep = run_overlap(Path(d), world, rules)
                self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
                self.assertIn("ERROR:", r.stderr)
                self.assertNotIn("Traceback", r.stderr)
                self.assertIsNone(rep)                                                         # nessun rapporto parziale

    def test_missing_column_is_clean_error(self):
        w = clean_world()
        rp = write_world(self.d, w)
        p = self.d / "manifests" / "es.csv"
        p.write_text("key,label,subject,sha256\nreal/0003_0003_00_00_01_600.jpg,0,0003," + fake_hash(0) + "\n")
        r = subprocess.run([sys.executable, str(SCRIPT), "check-overlap", "--rules", str(rp), "--manifests-dir", str(self.d / "manifests"),
                            "--out", str(self.d / "out")], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("missing columns", r.stderr)
        self.assertNotIn("Traceback", r.stderr)


class TestPackageRules(unittest.TestCase):
    def test_same_constants_as_export_m1(self):
        # package_rules.py duplica le costanti di export_m1.py (che resta invariato): devono restare identiche.
        # export_m1.py esiste solo nel repository di sviluppo (non nel pacchetto pubblico): senza il file il test si
        # salta, con --require-real diventa un errore
        p = ROOT / "scripts" / "export_m1.py"
        if not p.is_file():
            self.skipTest("scripts/export_m1.py not present (public package): constants not compared")
        sys.path.insert(0, str(ROOT / "scripts"))
        spec = importlib.util.spec_from_file_location("export_m1_reference", p)
        export_m1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(export_m1)
        import package_rules
        self.assertEqual(package_rules.C1_MANIFEST_SHA256, export_m1.C1_MANIFEST_SHA256)
        self.assertEqual([(lab, rx.pattern, rx.flags) for lab, rx in package_rules.FORBIDDEN_TEXT],
                         [(lab, rx.pattern, rx.flags) for lab, rx in export_m1.FORBIDDEN_TEXT])
        self.assertIs(NM.FORBIDDEN_TEXT, package_rules.FORBIDDEN_TEXT)
        self.assertEqual((package_rules.IPV4.pattern, package_rules.IP_ALLOW, package_rules.MARKERS.pattern, package_rules.MARKERS.flags),
                         (export_m1.IPV4.pattern, export_m1.IP_ALLOW, export_m1.MARKERS.pattern, export_m1.MARKERS.flags))

    def test_package_rules_source_passes_its_own_scan(self):
        import package_rules
        self.assertEqual(NM.public_problems((ROOT / "scripts" / "package_rules.py").read_text(encoding="utf-8"), set()), [])
        self.assertTrue(package_rules.FORBIDDEN_TEXT[1][1].search("vale" + "cass"))


class TestManifestValidation(unittest.TestCase):
    """Campi incoerenti con la chiave, sessione vuota o mancante, BOM, colonne duplicate, troncamenti, regole vuote."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def assertCleanError(self, r, text=None):
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("ERROR:", r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        if text:
            self.assertIn(text, r.stderr)

    def test_fields_inconsistent_with_key(self):
        base = clean_world()["train"][0]       # real/0001_0001_00_00_01_600.jpg
        cases = {"subject": dict(base, subject="1"),                  # zeri iniziali persi
                 "glasses": dict(base, glasses="01"), "pos": dict(base, pos="0"), "pic": dict(base, pic="00"),
                 "session": dict(base, session="02"), "official_split": dict(base, official_split="test"),
                 "label": dict(base, label=1), "subject ": dict(base, subject="0001 "),          # spazio
                 "NUAA file names": dict(base, key="real/photo.jpg"),
                 "NUAA file names ": dict(base, key="real/0001_0001_00_00_01_ 0.jpg"),
                 "malformed bytes": dict(base, bytes="12a"), "malformed sha256": dict(base, sha256=fake_hash(0).upper())}
        for what, bad in cases.items():
            w = clean_world(); w["train"][0] = bad
            problems = NM.manifest_problems([{k: str(v) for k, v in r.items()} for r in w["train"]])
            self.assertTrue(problems, what)
            self.assertIn(what.strip(), " ".join(problems), what)
        self.assertEqual(NM.manifest_problems([{k: str(v) for k, v in r.items()} for r in clean_world()["train"]]), [])

    def test_session_99_and_empty_split_are_refused(self):
        # regressione della revisione: prima il ripiego SESSION_SPLIT.get(..., "") accettava session=99 con split vuoto
        r99 = dict(row(0, "0001", "99", 1), official_split="")
        problems = " ".join(NM.manifest_problems([{k: str(v) for k, v in r99.items()}]))
        self.assertIn("session values outside", problems)
        empty = dict(row(0, "0001", "01", 1), official_split="")
        problems = " ".join(NM.manifest_problems([{k: str(v) for k, v in empty.items()}]))
        self.assertIn("official_split values other than train/test", problems)
        for split in ("validation", "Train", " train"):
            bad = dict(row(0, "0001", "01", 1), official_split=split)
            self.assertTrue(NM.manifest_problems([{k: str(v) for k, v in bad.items()}]), split)
        w = clean_world(); w["test"][0] = r99
        r, rep = run_overlap(self.d, w)
        self.assertCleanError(r, "session values outside")
        self.assertIsNone(rep)

    def test_wrong_rows_stop_report_rules_too(self):
        # una riga incoerente ferma check-overlap prima di qualsiasi regola, anche se l'unica regola è "report"
        w = clean_world(); w["test"][0] = dict(w["test"][0], subject="0004")
        rules = {**SYN_RULES, "rules": [r for r in SYN_RULES["rules"] if r["severity"] == "report"]}
        r, rep = run_overlap(self.d, w, rules)
        self.assertCleanError(r, "subject values not consistent with the key")
        self.assertIsNone(rep)

    def test_session_empty_or_missing(self):
        w = clean_world(); w["es"][0] = dict(w["es"][0], session="")
        r, _ = run_overlap(self.d, w)
        self.assertCleanError(r, "session values not consistent with the key")
        with tempfile.TemporaryDirectory() as d:
            rp = write_world(Path(d), clean_world())
            p = Path(d) / "manifests" / "es.csv"
            lines = p.read_text().splitlines()
            idx = lines[0].split(",").index("session")
            p.write_text("\n".join(",".join(x for i, x in enumerate(line.split(",")) if i != idx) for line in lines) + "\n")
            r = subprocess.run([sys.executable, str(SCRIPT), "check-overlap", "--rules", str(rp), "--manifests-dir", str(Path(d) / "manifests"),
                                "--out", str(Path(d) / "out")], cwd=ROOT, capture_output=True, text=True)
            self.assertCleanError(r, "missing columns ['session']")

    def test_truncated_manifest_is_refused(self):
        w = clean_world()
        expect = {n: declared(p) for n in w for p in [write_world(self.d / "full", w).parent / "manifests" / f"{n}.csv"]}
        t = clean_world(); t["c1"] = t["c1"][:1]                       # C1 troncato: sottoinsieme ancora vero
        r, rep = run_overlap(self.d / "trunc", t, expect=expect)
        self.assertCleanError(r, "c1: 1 rows, expected 2")
        self.assertIsNone(rep)
        t = clean_world(); t["c1"] = [t["test"][1], t["test"][3]]       # stesse righe attese, contenuto diverso
        r, _ = run_overlap(self.d / "swap", t, expect=expect)
        self.assertCleanError(r, "differs from the declared")

    def test_declared_intersection_larger(self):
        # 3 immagini in comune fra dev ed early stopping con 2 dichiarate: deve fallire (l'uguaglianza non è un >=)
        w = clean_world()
        extra = row(1, "0003", "01", 5)
        w["es"].append(extra); w["dev"].append(extra)
        r, rep = run_overlap(self.d, w)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        c = failed_checks(rep, "S07-reuse")
        self.assertEqual((c[0]["n_shared"], c[0]["expected"]), (3, 2))
        self.assertEqual(verdict(rep, "S06-es-in-dev")["verdict"], "pass")

    def test_bom_is_accepted(self):
        w = clean_world()
        rp = write_world(self.d, w)
        p = self.d / "manifests" / "train.csv"
        write_csv(p, w["train"], encoding="utf-8-sig")
        doc = json.loads(rp.read_text()); doc["manifests"]["train"] = declared(p); rp.write_text(json.dumps(doc))
        self.assertTrue(p.read_bytes().startswith(b"\xef\xbb\xbf"))
        r = subprocess.run([sys.executable, str(SCRIPT), "check-overlap", "--rules", str(rp), "--manifests-dir", str(self.d / "manifests"),
                            "--out", str(self.d / "out")], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(json.loads((self.d / "out" / "overlap_report.json").read_text())["manifests"]["train"]["rows"], 4)

    def test_duplicated_column_is_refused(self):
        rp = write_world(self.d, clean_world())
        p = self.d / "manifests" / "es.csv"
        lines = p.read_text().splitlines()
        p.write_text("\n".join([lines[0] + ",key"] + [line + "," + line.split(",")[0] for line in lines[1:]]) + "\n")
        r = subprocess.run([sys.executable, str(SCRIPT), "check-overlap", "--rules", str(rp), "--manifests-dir", str(self.d / "manifests"),
                            "--out", str(self.d / "out")], cwd=ROOT, capture_output=True, text=True)
        self.assertCleanError(r, "duplicated columns in the header ['key']")

    def test_vacuous_rules_fail(self):
        w = {n: [{k: str(v) for k, v in r.items()} for r in rows] for n, rows in clean_world().items()}
        w["c1"] = []
        v = {x["id"]: x["verdict"] for x in NM.evaluate_rules(w, SYN_RULES["rules"])}
        self.assertEqual(v["S05-c1-in-test"], "FAIL")          # sottoinsieme vuoto: non passa a vuoto
        self.assertEqual(v["S04-subjects"], "FAIL")            # anche una regola "report"
        self.assertEqual(v["S01-internal"], "FAIL")

    def test_public_overlap_report_has_no_keys(self):
        w = clean_world()
        w["test"].append(dict(w["train"][0]))                  # immagine del train nel test: regola must fallita
        r, rep = run_overlap(self.d, w)
        self.assertEqual(r.returncode, 1)
        out = self.d / "out"
        self.assertIn("examples", failed_checks(rep, "S03-train-vs-test")[0])
        self.assertIn("real/0001_0001_00_00_01_600.jpg", (out / "overlap_report.md").read_text())
        pub = (out / "overlap_report_public.json").read_text() + (out / "overlap_report_public.md").read_text()
        self.assertNotIn("examples", pub)
        self.assertEqual(NM.public_problems(pub, {"0001", "0102", "0003", "0004"}), [])


class TestDerivedFrames(unittest.TestCase):
    def test_frame_duplicates(self):
        import numpy as np
        rng = np.random.default_rng(0)
        xtr = rng.integers(0, 256, (6, 4, 4, 3), dtype=np.uint8)
        xte = rng.integers(0, 256, (3, 4, 4, 3), dtype=np.uint8)
        xtr[5] = xtr[1]                     # duplicato interno del train, fra stima (1) e validazione (5)
        xte[2] = xtr[0]                     # frame del train ripetuto nel test
        rec = {"train_idx": np.array([0, 1, 2]), "val_idx": np.array([3, 4, 5])}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cache.npz"
            np.savez_compressed(p, X_train=xtr, X_test=xte)
            out = NM.derived_frame_check(p, rec)
        self.assertEqual((out["train"]["duplicated_values"], out["train"]["duplicated_rows"], out["train"]["distinct"]), (1, 2, 5))
        self.assertEqual(out["test"]["duplicated_values"], 0)
        self.assertEqual(out["train_test_shared"], 1)
        self.assertEqual(out["fit_vs_validation_shared"], 1)
        # stessi byte con forma diversa: frame diversi
        self.assertNotEqual(NM.frame_hashes(xtr[:1])[0], NM.frame_hashes(xtr[:1].reshape(1, 2, 8, 3))[0])


class TestBuildWithoutData(unittest.TestCase):
    def test_missing_numpy_is_clean_error(self):
        # senza numpy: messaggio chiaro e codice 1, mai un traceback
        code = ("import sys, runpy; sys.modules['numpy'] = None; sys.argv = ['nuaa_manifests.py', 'check-overlap']; "
                f"runpy.run_path({str(SCRIPT)!r}, run_name='__main__')")
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("missing Python package 'numpy'", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_build_without_data_is_clean_error(self):
        # cartella dati vuota: build si ferma con un errore pulito, senza traceback e senza scrivere manifest
        with tempfile.TemporaryDirectory() as d:
            env = {**os.environ, "LIVENESSLAB_DATA_DIR": str(Path(d) / "data"), "LIVENESSLAB_RESULTS_DIR": str(Path(d) / "results")}
            r = subprocess.run([sys.executable, str(SCRIPT), "build", "--out", str(Path(d) / "out")], cwd=ROOT,
                               capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            self.assertIn("ERROR:", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            self.assertFalse((Path(d) / "out" / "manifests").exists())

    def test_build_refuses_read_only_output(self):
        r = subprocess.run([sys.executable, str(SCRIPT), "build", "--out", str(NM.RESULTS / "c1" / "m2")], cwd=ROOT,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("read-only area", r.stderr)


# ----------------------------------------------------------------------------- regole di derivazione

def derivation_world() -> dict:
    """Mondo sintetico per le derivazioni: train ufficiale con 4 soggetti e 6 immagini per classe ciascuno, campione
    `spread(..., 8)` per classe diviso per soggetto in stima e validazione, dev = tutte le immagini dei soggetti di
    validazione; test ufficiale e campioni `spread(..., 6)` e `spread(..., 2)`."""
    train = [row(lab, s, "01" if i % 2 else "02", i) for s in ("0001", "0003", "0005", "0007") for lab in (0, 1) for i in range(6)]
    test = [row(lab, s, "03", i, split="test") for s in ("0004", "0006") for lab in (0, 1) for i in range(5)]
    sample = NM.spread_keys([{k: str(v) for k, v in r.items()} for r in train], 8)
    by_key = {r["key"]: r for r in train}
    val_subj = {"0005", "0007"}
    fit = [by_key[k] for k in sample if by_key[k]["subject"] not in val_subj]
    val = [by_key[k] for k in sample if by_key[k]["subject"] in val_subj]
    dev = [r for r in train if r["subject"] in val_subj]
    tkeys = {r["key"]: r for r in test}
    cnn_test = [tkeys[k] for k in NM.spread_keys([{k: str(v) for k, v in r.items()} for r in test], 6)]
    c1 = [tkeys[k] for k in NM.spread_keys([{k: str(v) for k, v in r.items()} for r in test], 2)]
    return {"official_train": train, "official_test": test, "cnn_train": fit, "cnn_es": val, "cnn_test": cnn_test, "m1_dev": dev, "c1": c1}


DERIV_RULES = {"manifests": {n: f"{n}.csv" for n in ("official_train", "official_test", "cnn_train", "cnn_es", "cnn_test", "m1_dev", "c1")},
               "rules": [
                   {"id": "D1-partition", "kind": "spread_partition", "source": "official_train", "parts": ["cnn_train", "cnn_es"], "per_class": 8,
                    "by": ["key"], "severity": "must"},
                   {"id": "D2-closure", "kind": "subject_closure", "a": "m1_dev", "source": "official_train", "subjects_of": "cnn_es",
                    "by": ["key"], "severity": "must"},
                   {"id": "D3-test-spread", "kind": "equals_spread", "a": "cnn_test", "source": "official_test", "per_class": 6, "by": ["key"], "severity": "must"},
                   {"id": "D4-c1-spread", "kind": "equals_spread", "a": "c1", "source": "official_test", "per_class": 2, "by": ["key"], "severity": "must"}]}


def strs(world):
    return {n: [{k: str(v) for k, v in r.items()} for r in rows] for n, rows in world.items()}


class TestDerivations(unittest.TestCase):
    def verdicts(self, world):
        return {v["id"]: v["verdict"] for v in NM.evaluate_rules(strs(world), DERIV_RULES["rules"])}

    def test_spread_keys_order(self):
        rows = strs({"x": [row(1, "0001", "01", i) for i in range(4)] + [row(0, "0001", "01", i) for i in range(4)]})["x"]
        keys = NM.spread_keys(rows, 2)
        self.assertEqual([k.split("/")[0] for k in keys], ["real", "real", "attack", "attack"])      # bona fide prima
        self.assertEqual(keys[:2], ["real/0001_0001_00_00_01_600.jpg", "real/0001_0001_00_00_01_602.jpg"])

    def test_clean_world_passes(self):
        self.assertEqual(set(self.verdicts(derivation_world()).values()), {"pass"})
        with tempfile.TemporaryDirectory() as d:
            r, rep = run_overlap(Path(d), derivation_world(), DERIV_RULES)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_each_violation(self):
        cases = {
            "train row removed": (lambda w: w.update(cnn_train=w["cnn_train"][1:]), {"D1-partition"}),
            "row in both parts": (lambda w: w.update(cnn_train=w["cnn_train"] + [w["cnn_es"][0]]), {"D1-partition"}),
            "row outside the sample": (lambda w: w.update(cnn_es=w["cnn_es"] + [next(r for r in w["m1_dev"] if r not in w["cnn_es"])]), {"D1-partition"}),
            "dev row removed": (lambda w: w.update(m1_dev=w["m1_dev"][1:]), {"D2-closure"}),
            "dev row of another subject": (lambda w: w.update(m1_dev=w["m1_dev"] + [w["cnn_train"][0]]), {"D2-closure"}),
            "test row duplicated": (lambda w: w.update(cnn_test=w["cnn_test"] + [w["cnn_test"][0]]), {"D3-test-spread"}),
            "c1 row replaced": (lambda w: w.update(c1=w["c1"][1:] + [next(r for r in w["official_test"] if r not in w["c1"])]), {"D4-c1-spread"}),
            "source row removed": (lambda w: w.update(official_test=w["official_test"][1:]), {"D3-test-spread", "D4-c1-spread"}),
        }
        for what, (mutate, expected) in cases.items():
            w = derivation_world(); mutate(w)
            failing = {k for k, v in self.verdicts(w).items() if v == "FAIL"}
            self.assertEqual(failing, expected, what)

    def test_partition_reports_counts(self):
        w = derivation_world(); w["cnn_train"] = w["cnn_train"][2:]
        v = next(x for x in NM.evaluate_rules(strs(w), DERIV_RULES["rules"]) if x["id"] == "D1-partition")
        c = v["checks"][0]
        self.assertEqual((c["n_missing"], c["n_extra"], c["n_expected"] - c["n_rows"]), (2, 0, 2))
        self.assertIn("examples", c)

    def test_derivation_rule_problems(self):
        names = list(DERIV_RULES["manifests"])
        base = {"manifests": {n: {"file": f"{n}.csv", "expected_rows": 1, "expected_sha256": fake_hash(n)} for n in names}}
        ok = DERIV_RULES["rules"][0]
        self.assertEqual(NM.rule_problems({**base, "rules": DERIV_RULES["rules"]}, names), [])
        for bad in ({**ok, "by": ["hash"]}, {**ok, "per_class": 0}, {**ok, "parts": ["cnn_train", "cnn_train"]},
                    {**ok, "parts": ["official_train", "cnn_es"]}, {**ok, "source": "nope"}, {**ok, "expected": 3},
                    {**DERIV_RULES["rules"][1], "subjects_of": "m1_dev"}, {**DERIV_RULES["rules"][2], "a": ["cnn_test", "c1"]}):
            self.assertTrue(NM.rule_problems({**base, "rules": [bad]}, names), bad)


class TestKeyConsistency(unittest.TestCase):
    """Scenario della revisione: due SHA scambiati fra righe di C1, test ufficiale e test delle CNN invariati, righe e
    SHA-256 dichiarati aggiornati. Le regole per insiemi di chiavi e di hash passano tutte; solo la coerenza per chiave
    lo rileva."""

    RULES = {"manifests": DERIV_RULES["manifests"],
             "rules": DERIV_RULES["rules"] + [
                 {"id": "S-c1-in-test", "kind": "subset", "a": ["c1", "cnn_test"], "b": "official_test", "by": ["key", "hash"], "severity": "must"},
                 {"id": "S-c1-in-cnn-test", "kind": "subset", "a": "c1", "b": "cnn_test", "by": ["key", "hash"], "severity": "must"},
                 {"id": "S-internal", "kind": "no_internal_duplicates", "manifests": "*", "by": ["key", "hash"], "severity": "must"}]}
    CONSISTENCY = {"id": "S-key-consistency", "kind": "key_consistency", "manifests": "*", "by": ["key"], "severity": "must"}

    def swapped(self):
        w = derivation_world()
        a, b = w["c1"][0], w["c1"][1]
        w["c1"] = [dict(a, sha256=b["sha256"]), dict(b, sha256=a["sha256"])] + w["c1"][2:]
        return w

    def test_swap_passes_set_rules_and_fails_key_consistency(self):
        with tempfile.TemporaryDirectory() as d:
            r, rep = run_overlap(Path(d) / "sets", self.swapped(), self.RULES)        # checksum della fixture aggiornati
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)                     # il punto della revisione
            rules = {**self.RULES, "rules": self.RULES["rules"] + [self.CONSISTENCY]}
            r, rep = run_overlap(Path(d) / "key", self.swapped(), rules)
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            self.assertEqual({v["id"] for v in rep["rules"] if v["verdict"] == "FAIL"}, {"S-key-consistency"})
            c = [x for x in verdict(rep, "S-key-consistency")["checks"] if x["outcome"] == "FAIL"]
            self.assertEqual([(x["by"], x["duplicated_values"]) for x in c], [("key->sha256", 2)])

    def test_every_field_is_compared(self):
        for field, value in (("bytes", "1"), ("glasses", "01"), ("pic", "1"), ("label", "1")):
            w = derivation_world()
            mans = strs(w)
            mans["c1"][0] = dict(mans["c1"][0], **{field: value})            # in process: senza la validazione delle righe
            v = NM.evaluate_rules(mans, [self.CONSISTENCY])[0]
            self.assertEqual(v["verdict"], "FAIL", field)
            self.assertIn(f"key->{field}", {x["by"] for x in v["checks"] if x["outcome"] == "FAIL"})

    def test_key_in_exactly_two_manifests(self):
        # una chiave presente in due soli manifest (train ufficiale e train delle CNN) con SHA-256 diverso: R15 fallisce
        mans = strs(derivation_world())
        k = mans["cnn_train"][0]["key"]
        self.assertEqual(sum(any(r["key"] == k for r in rows) for rows in mans.values()), 2)
        mans["cnn_train"][0] = dict(mans["cnn_train"][0], sha256=fake_hash("other"))
        v = NM.evaluate_rules(mans, [self.CONSISTENCY])[0]
        self.assertEqual(v["verdict"], "FAIL")
        self.assertEqual([(x["by"], x["duplicated_values"]) for x in v["checks"] if x["outcome"] == "FAIL"], [("key->sha256", 1)])

    def test_clean_world_and_rule_problems(self):
        v = NM.evaluate_rules(strs(derivation_world()), [self.CONSISTENCY])[0]
        self.assertEqual(v["verdict"], "pass")
        self.assertGreater(v["checks"][0]["keys_in_several_manifests"], 0)
        names = list(DERIV_RULES["manifests"])
        base = {"manifests": {n: {"file": f"{n}.csv", "expected_rows": 1, "expected_sha256": fake_hash(n)} for n in names}}
        self.assertEqual(NM.rule_problems({**base, "rules": [self.CONSISTENCY]}, names), [])
        self.assertTrue(NM.rule_problems({**base, "rules": [{**self.CONSISTENCY, "by": ["hash"]}]}, names))
        self.assertTrue(NM.rule_problems({**base, "rules": [{**self.CONSISTENCY, "manifests": ["c1", "nope"]}]}, names))


class TestRecountCompare(unittest.TestCase):
    def test_recount_block_detects_changes(self):
        w = strs(derivation_world())
        b = NM.recount_block(w)
        self.assertEqual(b["declaration_counts"]["m1_dev_rows_that_are_validation_images"], len(w["cnn_es"]))
        self.assertEqual(b["test_from_subjects_absent_from_official_train"]["c1"]["bona_fide"],
                         {"n": 2, "from_subjects_absent_from_official_train": 2})
        w2 = strs(derivation_world()); w2["cnn_es"] = w2["cnn_es"][1:]
        self.assertNotEqual(NM.recount_block(w2), b)

    def test_compare_c1(self):
        with tempfile.TemporaryDirectory() as d:
            v2 = [row(0, "0004", "03", 1, glasses="01", split="test"), row(1, "0004", "03", 2, glasses="00", split="test")]
            write_csv(Path(d) / "v2.csv", v2)
            with (Path(d) / "v1.csv").open("w") as f:
                f.write("file,label,subject,session,sha256,bytes\n")
                for r in v2:
                    f.write(f"{r['key']},{r['label']},{r['subject']},{r['glasses']},{r['sha256']},{r['bytes']}\n")
            r = subprocess.run([sys.executable, str(SCRIPT), "compare-c1", "--v1", str(Path(d) / "v1.csv"), "--v2", str(Path(d) / "v2.csv")],
                               cwd=ROOT, capture_output=True, text=True)
            # righe uguali, ma il file v1 non è quello consegnato con C1: rifiutato con un errore pulito
            self.assertIn("2/2", r.stdout)
            self.assertEqual(r.returncode, 1)
            self.assertIn("not the manifest delivered with C1", r.stderr)
            self.assertNotIn("Traceback", r.stderr)


# ----------------------------------------------------------------------------- dati reali (facoltativo)

def real_dir() -> Path:
    return Path(REAL_DIR) if REAL_DIR else NM.M2_DIR


def need_real(test):
    man_dir = real_dir() / "manifests"
    doc = json.loads(RULES.read_text(encoding="utf-8"))
    missing = [m["file"] for m in doc["manifests"].values() if not (man_dir / m["file"]).is_file()]
    if missing:
        if REQUIRE_REAL:
            test.fail(f"real manifests missing in {man_dir}: {missing}")
        test.skipTest("real manifests not available (run nuaa_manifests.py build, or pass --real-dir)")
    return man_dir


class TestRealData(unittest.TestCase):
    def run_nm(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], cwd=ROOT, capture_output=True, text=True)

    def test_real_self_test(self):
        man_dir = need_real(self)
        with tempfile.TemporaryDirectory() as d:
            r = self.run_nm("check-overlap", "--self-test", "--manifests-dir", man_dir, "--out", d)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            st_rep = json.loads((Path(d) / "overlap_self_test.json").read_text())
        self.assertEqual((st_rep["n_faults"], st_rep["n_detected_as_expected"]), (8, 8))

    def test_real_reconstruct(self):
        man_dir = need_real(self)
        r = self.run_nm("reconstruct", "--manifests-dir", man_dir)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stdout.count("identical: yes"), 3)

    def test_real_recount(self):
        man_dir = need_real(self)
        agg = Path(REAL_AGGREGATE) if REAL_AGGREGATE else next((p for p in (real_dir() / "nuaa_manifests_aggregate.json",
                                                                              ROOT.parent / "nuaa_manifests_aggregate.json") if p.is_file()), None)
        if agg is None:
            if REQUIRE_REAL:
                self.fail("public aggregate not found (pass --real-aggregate)")
            self.skipTest("public aggregate not available")
        r = self.run_nm("recount", "--manifests-dir", man_dir, "--aggregate", agg)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("1,987/3,362", r.stdout)

    def test_real_compare_c1(self):
        man_dir = need_real(self)
        v1 = Path(C1_V1) if C1_V1 else NM.C1_MANIFEST
        if not v1.is_file():
            if REQUIRE_REAL:
                self.fail("C1 manifest version 1 not found (pass --c1-v1)")
            self.skipTest("C1 manifest version 1 not available")
        r = self.run_nm("compare-c1", "--v1", v1, "--v2", man_dir / "nuaa_c1_v2.csv")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("300/300", r.stdout)

    def test_real_manifests_overlap(self):
        man_dir = need_real(self)
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, str(SCRIPT), "check-overlap", "--manifests-dir", str(man_dir), "--out", d],
                               cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            rep = json.loads((Path(d) / "overlap_report.json").read_text())
        summ_p = real_dir() / "nuaa_manifests_summary.json"
        if summ_p.is_file():
            summ = json.loads(summ_p.read_text())
            self.assertTrue(summ["passed"])
            by_file = {v["file"].split("/")[-1]: v["sha256"] for v in summ["manifests"].values()}
            self.assertEqual({v["file"]: v["sha256"] for v in rep["manifests"].values()}, by_file)
        rows = {n: v["rows"] for n, v in rep["manifests"].items()}
        self.assertEqual(rows, {"official_train": 3491, "official_test": 9123, "cnn_train": 2507, "cnn_es": 493,
                                "cnn_test": 3000, "m1_dev": 574, "c1": 300})


def main() -> int:
    global REQUIRE_REAL, REAL_DIR, REAL_AGGREGATE, C1_V1
    ap = argparse.ArgumentParser(description="Test di nuaa_manifests.py su dati sintetici (e sui manifest reali, se presenti)")
    ap.add_argument("--verbose", action="store_true", help="stampa il log completo di unittest")
    ap.add_argument("--require-real", action="store_true", help="i manifest reali devono esserci (niente salto dei test sui dati reali)")
    ap.add_argument("--real-dir", default=None, help="cartella con manifests/ (es. la cartella riservata del run); default results/m2")
    ap.add_argument("--real-aggregate", default=None, help="aggregato pubblico da ricontare (default nella cartella reale o accanto al codice)")
    ap.add_argument("--c1-v1", default=None, help="manifest C1 versione 1 (default results/c1/nuaa_manifest.csv)")
    args = ap.parse_args()
    REQUIRE_REAL, REAL_DIR, REAL_AGGREGATE, C1_V1 = args.require_real, args.real_dir, args.real_aggregate, args.c1_v1
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c)
                               for c in (TestParser, TestRules, TestPackageRules, TestOverlapCases, TestManifestValidation, TestDerivedFrames,
                                         TestDerivations, TestKeyConsistency, TestRecountCompare, TestBuildWithoutData, TestRealData))
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    if args.verbose or not result.wasSuccessful():
        print(log.getvalue())
    print(f"tests run: {result.testsRun}, failures: {len(result.failures)}, errors: {len(result.errors)}, skipped: {len(result.skipped)}")
    print("RESULT: " + ("PASS" if result.wasSuccessful() else "FAIL"))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
