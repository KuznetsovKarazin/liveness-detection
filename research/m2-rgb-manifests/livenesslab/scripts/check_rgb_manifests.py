"""
Test di scripts/rgb_manifests.py (CASIA-FASD e CelebA-Spoof) su dati sintetici, senza rete, senza immagini e senza
inferenza.

Parte offline (sempre): campi dalle chiavi e insiemi chiusi (soggetti locali allo split, token video, classe del
token, righe dello shard), validazione stretta delle righe (ogni campo, valori vuoti, spazi, zeri iniziali, colonne in
più), struttura dei file delle regole versionati e rifiuto di regole malformate, ogni tipo di regola su manifest
SINTETICI con sovrapposizioni note (caso pulito che passa e un caso guasto per ogni violazione, rilevato dalla regola
attesa, con codice 1 e messaggio `ERROR:` senza traceback), sensibilità alle mutazioni (conteggi dichiarati più alti e
più bassi, scambio di SHA-256, ogni campo della coerenza per chiave), dichiarazioni non verificabili, scansione del
contenuto pubblico, logica degli split del commit 0ac7143 (ramo per gruppi o stratificato, ordine di campionamento),
regola di conversione di CelebA-Spoof, testo delle dichiarazioni identico nei documenti, e un controllo di deriva di
scikit-learn su un ingresso fisso (errore esplicito se la libreria dà uno split diverso da quello della versione con
cui sono stati scritti i manifest dichiarati).

Parte sui dati reali (solo se <real-dir>/<dataset>/manifests contiene i manifest dichiarati; default results/m2_rgb):
check-overlap con le regole versionate, self-test, recount e reconstruct; si salta in modo pulito se mancano (con
--require-real il salto diventa un errore).

  .venv/bin/python scripts/check_rgb_manifests.py                  # uscita 1 se un test fallisce
  .venv/bin/python scripts/check_rgb_manifests.py --require-real --real-dir <cartella del run>

Libreria standard più numpy e scikit-learn (gli stessi di rgb_manifests.py); nessuna API privata.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "rgb_manifests.py"
REQUIRE_REAL = False
REAL_DIR = None


def load_module():
    spec = importlib.util.spec_from_file_location("rgb_manifests_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RM = load_module()


# chiavi costruite a run time: nessun identificativo letterale (anche fittizio) nel sorgente pubblico, che la
# scansione dell'export confronta con i manifest reali
def SK(*parts) -> str:
    return ":".join(map(str, parts))


def CK(i: int) -> str:
    return "celeba_" + f"{i:05d}"


def fake_hash(tag) -> str:
    return hashlib.sha256(f"synthetic-{tag}".encode()).hexdigest()


# frame non multipli di 25 e righe oltre lo shard reale non esistono nella copia: nessun nome reale nei test
def c_row(split, subject, video, frame, sha=None):
    cls = "real" if video in RM.CASIA_REAL_VIDEOS else "fake"
    name = f"{subject}_{video}.avi_{frame}_{cls}.jpg"
    return RM.casia_row(split, name, sha or fake_hash(f"{split}/{name}"), 1000 + frame)


def casia_world() -> dict:
    """Mondo sintetico pulito: train soggetti 1-3 (fit 1-2, early stopping 3), test soggetti 1-2, eval 2 righe."""
    tr = [c_row("train", s, v, f) for s in (1, 2, 3) for v in ("1", "3") for f in (7, 13)]
    te = [c_row("test", s, v, f) for s in (1, 2) for v in ("1", "3") for f in (7, 13)]
    return {"mirror_train": tr, "mirror_test": te, "cnn_train": [r for r in tr if r["subject"] != "3"],
            "cnn_es": [r for r in tr if r["subject"] == "3"], "cnn_test": list(te), "eval": [te[0], te[-1]]}


SYN_RULES = {
    "dataset": "casia_fasd",
    "rules": [
        {"id": "R01-dup", "kind": "no_internal_duplicates", "manifests": "*", "by": ["key", "hash"], "severity": "must", "description": "d"},
        {"id": "R02-fit-es", "kind": "disjoint", "a": "cnn_train", "b": "cnn_es", "by": ["key", "hash", "subject", "video"], "severity": "must", "description": "d"},
        {"id": "R03-sides", "kind": "disjoint", "a": ["mirror_train", "cnn_train", "cnn_es"], "b": ["mirror_test", "cnn_test", "eval"],
         "by": ["key", "hash"], "severity": "must", "description": "d"},
        {"id": "R04-local", "kind": "disjoint", "a": "mirror_train", "b": "mirror_test", "by": ["local_subject"], "severity": "report", "description": "d"},
        {"id": "R05-split", "kind": "spread_partition", "source": "mirror_train", "parts": ["cnn_train", "cnn_es"], "per_class": 1500,
         "by": ["key"], "severity": "must", "description": "d"},
        {"id": "R06-test", "kind": "equals_spread", "a": "cnn_test", "source": "mirror_test", "per_class": 1500, "by": ["key"], "severity": "must", "description": "d"},
        {"id": "R07-eval-in-test", "kind": "subset", "a": "eval", "b": "cnn_test", "by": ["key", "hash"], "severity": "must", "description": "d"},
        {"id": "R08-eval-count", "kind": "intersection", "a": "eval", "b": ["cnn_test", "mirror_train"], "by": ["key", "video"],
         "expected": {"key": 2, "video": 2}, "severity": "must", "description": "d"},
        {"id": "R09-identity", "kind": "declared_not_checkable", "severity": "report", "reason": "no identity field", "description": "d"},
        {"id": "R10-consistency", "kind": "key_consistency", "manifests": "*", "by": ["key"], "severity": "must", "description": "d"},
        {"id": "R11-test-equals", "kind": "partition", "parts": ["cnn_test"], "sources": ["mirror_test"], "by": ["key"], "severity": "must", "description": "d"},
        {"id": "R12-dups-declared", "kind": "internal_duplicates", "manifests": ["mirror_test"], "by": ["hash"], "expected": {"mirror_test": 0},
         "severity": "must", "description": "d"},
    ],
}


def declare(ds, man_dir: Path, world: dict) -> dict:
    return {n: {"file": RM.SPEC[ds]["manifests"][n][0], "expected_rows": len(rows),
                "expected_sha256": RM.sha256(man_dir / RM.SPEC[ds]["manifests"][n][0])} for n, rows in world.items()}


def write_world(d: Path, world: dict, rules: dict = None, ds="casia_fasd") -> Path:
    man = d / "manifests"
    man.mkdir(parents=True, exist_ok=True)
    for n, rows in world.items():
        RM.write_manifest(ds, man / RM.SPEC[ds]["manifests"][n][0], rows)
    doc = json.loads(json.dumps(rules or SYN_RULES))
    doc["manifests"] = declare(ds, man, world)
    rp = d / "rules.json"
    rp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return rp


def run_cli(d: Path, world: dict, rules: dict = None, ds="casia_fasd", *extra):
    rp = write_world(d, world, rules, ds)
    r = subprocess.run([sys.executable, str(SCRIPT), "check-overlap", "--dataset", ds, "--rules", str(rp), "--manifests-dir",
                        str(d / "manifests"), "--out", str(d / "out"), *extra], cwd=ROOT, capture_output=True, text=True)
    p = d / "out" / "overlap_report.json"
    return r, (json.loads(p.read_text()) if p.exists() else None)


def evaluate(world: dict, rules=None, ds="casia_fasd") -> dict:
    vs = RM.evaluate_rules(ds, world, (rules or SYN_RULES)["rules"])
    return {v["id"]: v for v in vs}


def failing(world, rules=None, ds="casia_fasd") -> set:
    return {i for i, v in evaluate(world, rules, ds).items() if v["verdict"] == "FAIL"}


class quiet:
    def __enter__(self):
        self.saved = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        return self

    def __exit__(self, *exc):
        sys.stdout, sys.stderr = self.saved
        return False


# ----------------------------------------------------------------------------- campi e validazione delle righe

class TestFields(unittest.TestCase):
    def test_casia_fields(self):
        f = RM.casia_fields("test/28_HR_2.avi_130_fake.jpg")
        self.assertEqual(f, {"label": "1", "mirror_split": "test", "subject": "28", "subject_key": SK("test", 28), "video": "HR_2",
                             "video_key": SK("test", 28, "HR_2"), "frame": "130"})
        self.assertEqual(RM.casia_fields("train/3_1.avi_7_real.jpg")["label"], "0")

    def test_casia_closed_sets(self):
        for key in ("train/28_1.avi_7_real.jpg",          # soggetto 28 fuori dalla numerazione del train (1-20)
                    "test/31_1.avi_7_real.jpg", "test/0_1.avi_7_real.jpg", "test/07_1.avi_7_real.jpg",
                    "test/7_9.avi_7_fake.jpg", "test/7_HR_5.avi_7_fake.jpg",          # token video fuori dall'insieme
                    "test/7_3.avi_7_real.jpg", "test/7_HR_1.avi_7_fake.jpg",          # classe incoerente con il token
                    "test/7_1.avi_07_real.jpg", "dev/7_1.avi_7_real.jpg", "test/7_1.avi_7_real.png",
                    "7_1.avi_7_real.jpg", "test/x/7_1.avi_7_real.jpg", "", None):
            with self.assertRaises(ValueError, msg=key):
                RM.casia_fields(key)

    def test_celeba_fields(self):
        self.assertEqual(RM.celeba_fields(CK(6716)), {"row": "6716", "official_split": "test"})
        self.assertEqual(RM.celeba_fields(CK(0))["row"], "0")
        for key in (CK(6717), "celeba_" + "6716", "celeba_" + "006716", CK(1) + ".jpg", "row/00001", "", None):
            with self.assertRaises(ValueError, msg=key):
                RM.celeba_fields(key)


class TestRowValidation(unittest.TestCase):
    def test_clean_rows(self):
        self.assertEqual(RM.row_problems("casia_fasd", casia_world()["mirror_train"]), [])
        self.assertEqual(RM.row_problems("celeba_spoof", [RM.celeba_row(9000 % 6717, 1, fake_hash(1), 10)]), [])

    def test_every_casia_field_is_checked(self):
        base = c_row("test", 4, "HR_3", 7)
        for col, bad in (("label", "0"), ("label", " 1"), ("mirror_split", "train"), ("mirror_split", "dev"), ("mirror_split", ""),
                         ("subject", "04"), ("subject", "5"), ("subject_key", SK("train", 4)), ("subject_key", "4"),
                         ("video", "HR_2"), ("video_key", SK("test", 4, "HR_2")), ("frame", "07"), ("frame", ""),
                         ("sha256", "A" * 64), ("sha256", ""), ("bytes", "0"), ("bytes", "012"), ("bytes", "-5"), ("bytes", "")):
            row = dict(base, **{col: bad})
            self.assertTrue(RM.row_problems("casia_fasd", [row]), f"{col}={bad!r} accepted")
        missing = {k: v for k, v in base.items() if k != "frame"}
        self.assertTrue(RM.row_problems("casia_fasd", [missing]))
        self.assertTrue(RM.row_problems("casia_fasd", [dict(base, extra="x")]))

    def test_every_celeba_field_is_checked(self):
        base = RM.celeba_row(42, 0, fake_hash(42), 99)
        for col, bad in (("label", "2"), ("label", ""), ("label_name", "spoof"), ("label_name", "unknown"), ("label_name", "Live"),
                         ("row", "042"), ("row", "43"), ("official_split", "train"), ("official_split", ""), ("sha256", "x"), ("bytes", "")):
            self.assertTrue(RM.row_problems("celeba_spoof", [dict(base, **{col: bad})]), f"{col}={bad!r} accepted")
        flipped = dict(base, label="1", label_name="spoof")             # coerente nella riga: lo ferma la coerenza per chiave
        self.assertEqual(RM.row_problems("celeba_spoof", [flipped]), [])


# ----------------------------------------------------------------------------- regole

class TestRules(unittest.TestCase):
    def test_versioned_rules_are_valid(self):
        for ds in RM.DATASETS:
            doc = json.loads(RM.rules_path(ds).read_text(encoding="utf-8"))
            self.assertEqual(RM.rule_problems(ds, doc), [], ds)
            self.assertEqual(set(doc["manifests"]), set(RM.SPEC[ds]["manifests"]), ds)
            self.assertTrue(all(m["expected_sha256"] != "0" * 64 for m in doc["manifests"].values()), ds)
            kinds = {r["kind"] for r in doc["rules"]}
            self.assertIn("key_consistency", kinds, ds)
            self.assertIn("declared_not_checkable", kinds, ds)
            # i duplicati per chiave sono sempre "must" su tutti i manifest
            self.assertTrue(any(r["kind"] == "no_internal_duplicates" and r["manifests"] == "*" and r["severity"] == "must"
                                and "key" in r["by"] for r in doc["rules"]), ds)

    def test_versioned_rules_are_locked(self):
        """Le regole versionate sono bloccate offline: righe attese per manifest, tabella id/tipo/severità/dimensioni e
        SHA-256 canonico dell'intero contenuto normativo (manifest dichiarati e regole senza testi descrittivi). Ogni
        modifica a una regola (rimossa, indebolita, severità, dimensioni, manifest, conteggi dichiarati) fa fallire il test.
        I testi descrittivi (description, reason) sono esclusi dal digest: una modifica solo della loro formulazione non
        viene rilevata da questo test."""
        for ds, (rows, table, digest) in LOCKED_RULES.items():
            doc = json.loads(RM.rules_path(ds).read_text(encoding="utf-8"))
            self.assertEqual({n: m["expected_rows"] for n, m in doc["manifests"].items()}, rows, ds)
            self.assertEqual([[r["id"], r["kind"], r["severity"], r.get("by")] for r in doc["rules"]], table, ds)
            self.assertEqual(rules_digest(doc), digest, f"{ds}: the normative content of the rules changed")

    def test_declared_leaks_and_verdicts_in_versioned_rules(self):
        c = {r["id"]: r for r in json.loads(RM.rules_path("casia_fasd").read_text())["rules"]}
        self.assertEqual(c["C15-pooled-eval-leak"]["expected"], 235)
        self.assertEqual(c["C16-pooled-subject-video-leak"]["expected"], {"subject": 50, "video": 444})
        self.assertEqual(c["C08-fit-vs-es"]["severity"], "must")
        self.assertEqual(c["C03-identity-train-vs-test"]["kind"], "declared_not_checkable")
        s = {r["id"]: r for r in json.loads(RM.rules_path("celeba_spoof").read_text())["rules"]}
        self.assertEqual(s["S12-pooled-eval-leak"]["expected"], 244)
        self.assertEqual(s["S02-exact-duplicate-images"]["expected"]["parquet"], 1)
        self.assertEqual(s["S08-identity-not-checkable"]["kind"], "declared_not_checkable")

    def test_malformed_rules_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            write_world(Path(d), casia_world())
            good = json.loads((Path(d) / "rules.json").read_text())
        self.assertEqual(RM.rule_problems("casia_fasd", good), [])
        mutations = [
            lambda x: x.update(dataset="celeba_spoof"),
            lambda x: x["manifests"]["cnn_es"].update(file="other.csv"),
            lambda x: x["manifests"]["cnn_es"].update(expected_rows=0),
            lambda x: x["manifests"]["cnn_es"].update(expected_sha256="abc"),
            lambda x: x["manifests"].update(unknown={"file": "u.csv", "expected_rows": 1, "expected_sha256": "a" * 64}),
            lambda x: x["rules"][0].update(kind="nearly_disjoint"),
            lambda x: x["rules"][0].update(severity="should"),
            lambda x: x["rules"][1].update(by=["identity"]),
            lambda x: x["rules"][1].update(by=["key", "key"]),
            lambda x: x["rules"][1].update(b="cnn_train"),
            lambda x: x["rules"][1].update(b="nope"),
            lambda x: x["rules"][1].update(expected=3),
            lambda x: x["rules"][4].update(by=["key", "hash"]),
            lambda x: x["rules"][4].update(per_class=0),
            lambda x: x["rules"][7].update(expected={"key": 2}),
            lambda x: x["rules"][7].update(expected=-1),
            lambda x: x["rules"][8].update(severity="must"),
            lambda x: x["rules"][8].update(by=["key"]),
            lambda x: x["rules"][8].pop("reason"),
            lambda x: x["rules"][10].update(sources=["cnn_test"]),
            lambda x: x["rules"][11].update(expected={"cnn_test": 0}),
            lambda x: x["rules"][2].update(description=""),
            lambda x: x["rules"].append(dict(x["rules"][0])),
            lambda x: x["rules"].append({"id": "Z", "kind": "celeba_conversion", "source": "mirror_train", "training": "cnn_train",
                                         "validation": "cnn_test", "eval": "eval", "by": ["key"], "severity": "must", "description": "d"}),
        ]
        for i, mut in enumerate(mutations):
            doc = json.loads(json.dumps(good))
            mut(doc)
            self.assertTrue(RM.rule_problems("casia_fasd", doc), f"mutation {i} accepted")
        doc = json.loads(json.dumps(good)); doc["rules"][1]["by"] = ["subject"]
        self.assertTrue(RM.rule_problems("celeba_spoof", dict(doc, dataset="celeba_spoof")))   # CelebA non ha soggetti

    def test_declarations_identical_in_documents(self):
        norm = lambda t: re.sub(r"\s+", " ", t)   # noqa: E731
        # repository di sviluppo: delivery/m2/rgb; pacchetto pubblico: la cartella che contiene livenesslab/
        docs = {}
        for n in ("naming_schema_rgb.md", "README.md", "report.md.tmpl"):
            p = next((q for q in (RM.RULES_DIR / n, ROOT.parent / n) if q.is_file()), None)
            if p is not None:
                docs[n] = norm(p.read_text(encoding="utf-8"))
        self.assertIn("report.md.tmpl", docs)
        if not {"naming_schema_rgb.md", "README.md"} & set(docs):
            self.skipTest("naming_schema_rgb.md and README.md not found next to the code: declarations not compared")
        self.assertEqual({"naming_schema_rgb.md", "README.md"} - set(docs), set())
        for ds in RM.DATASETS:
            for label, text in RM.SOURCE_DECLARATIONS[ds]:
                for n in ("naming_schema_rgb.md", "README.md"):
                    if n in docs:
                        self.assertIn(f"- {label}: {text}", docs[n], f"{n}: {ds} {label}")
            if "README.md" in docs:
                for text in RM.LIMITS[ds] + RM.SUPPORTED[ds]:
                    self.assertIn(text, docs["README.md"], f"README.md: {ds}: {text[:40]}")
        # documentato e verificato restano distinti: lo schema del tipo di attacco non è mai detto verificato
        self.assertNotIn("warped", dict(RM.SOURCE_DECLARATIONS["casia_fasd"])["Verified on the copy"])
        self.assertIn("[DOC-EXT]", dict(RM.SOURCE_DECLARATIONS["casia_fasd"])["Documented [DOC-EXT]"] + " [DOC-EXT]")


def rules_digest(doc: dict) -> str:
    """SHA-256 canonico del contenuto normativo di un file di regole: dataset, manifest dichiarati e regole senza i
    testi descrittivi (description, reason), JSON con chiavi ordinate e separatori fissi."""
    norm = {"dataset": doc.get("dataset"), "manifests": doc["manifests"],
            "rules": [{k: v for k, v in r.items() if k not in ("description", "reason")} for r in doc["rules"]]}
    return hashlib.sha256(json.dumps(norm, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# formulazioni corrette il 9/10/2026: la corrispondenza con il mirror è verificata, quella dello split `test` del mirror
# con lo split di test ufficiale di CelebA-Spoof no
ALLOWED_OFFICIAL_TEST = ("correspondence with the official test split not verified", "with the official test split is not verified",
                         "relates to the official test split", "(official test split by protocol [DOC-EXT])")


def wording_problems(text: str) -> list:
    """Formulazioni superate nei testi pubblici: «not verified (no network» e «official test split / shard» fuori dalle
    frasi corrette (lo split di CelebA-Spoof è lo split `test` del mirror)."""
    t = re.sub(r"\s+", " ", text)
    bad = [p for p in ("not verified (no network", "no network access") if p in t]
    for ok in ALLOWED_OFFICIAL_TEST:
        t = t.replace(ok, "")
    bad += [p for p in ("official test split", "official test shard") if p in t]
    return bad


class TestWording(unittest.TestCase):
    def test_wording_of_public_sources(self):
        texts = {p.name: p.read_text(encoding="utf-8") for p in sorted(RM.RULES_DIR.glob("*")) if p.suffix in (".md", ".json", ".tmpl")}
        texts["rgb_manifests.py (declarations)"] = json.dumps([RM.SOURCE_DECLARATIONS, RM.LIMITS, RM.SUPPORTED,
                                                               {ds: RM.SPEC[ds]["manifests"] for ds in RM.DATASETS}])
        for name, text in texts.items():
            self.assertEqual(wording_problems(text), [], name)
        self.assertTrue(wording_problems("shard 0 of the official test split"))
        self.assertTrue(wording_problems("is not verified (no network access)"))
        self.assertEqual(wording_problems("the mirror's `test` split (correspondence with the official test split not verified)"), [])

    def test_official_and_authorisation_notes(self):
        """Note del 9/10: per CelebA-Spoof "official" è solo il nome del protocollo nelle schede; per CASIA-FASD il protocollo
        viene da ICB 2012; documenti di licenza distinti dalle autorizzazioni gestite dal coordinatore."""
        norm = lambda t: re.sub(r"\s+", " ", t)   # noqa: E731
        texts = {p.name: norm(p.read_text(encoding="utf-8")) for p in sorted(RM.RULES_DIR.glob("*")) if p.suffix in (".md", ".json", ".tmpl")}
        texts["rgb_manifests.py (declarations)"] = norm(json.dumps([RM.SOURCE_DECLARATIONS, RM.LIMITS, RM.SUPPORTED]))
        for name, t in texts.items():
            self.assertNotIn("registered authorisation", t, name)
            self.assertNotIn("none received", t, name)
        self.assertIn("{dataset_note}", texts["report.md.tmpl"])
        self.assertIn("{authorisation_note}", texts.get("run-report.md.tmpl", "{authorisation_note}"))
        readme = next((norm(p.read_text(encoding="utf-8")) for p in (RM.RULES_DIR / "README.md", ROOT.parent / "README.md") if p.is_file()), None)
        if readme is None:
            self.skipTest("README.md not found next to the code: notes not compared")
        for note in (RM.CELEBA_OFFICIAL_NOTE, RM.CASIA_PROTOCOL_NOTE, RM.AUTHORISATION_NOTE):
            self.assertIn(note, readme)
        self.assertGreaterEqual(readme.count(RM.CELEBA_OFFICIAL_NOTE), 3)              # in testa alle sezioni CelebA-Spoof
        self.assertIn("ICB 2012", RM.CASIA_PROTOCOL_NOTE)
        self.assertIn("not the original CelebA-Spoof protocol", RM.CELEBA_OFFICIAL_NOTE)


class TestRuleEngine(unittest.TestCase):
    def test_clean_world(self):
        v = evaluate(casia_world())
        self.assertEqual({i for i, x in v.items() if x["verdict"] == "FAIL"}, set())
        self.assertEqual(v["R04-local"]["verdict"], "reported")              # numeri locali condivisi, solo riportati
        self.assertEqual(v["R09-identity"]["verdict"], RM.NOT_CHECKABLE)
        self.assertEqual(v["R09-identity"]["checks"][0]["reason"], "no identity field")

    def test_each_violation(self):
        w = casia_world()
        es0, tr0, te0 = w["cnn_es"][0], w["cnn_train"][0], w["cnn_test"][0]
        cases = {
            "es row in fit": ({"cnn_train": w["cnn_train"] + [es0]}, {"R02-fit-es", "R05-split"}),
            "test row in fit": ({"cnn_train": w["cnn_train"] + [te0]}, {"R03-sides", "R05-split"}),
            "train hash under test key": ({"cnn_test": [dict(te0, sha256=tr0["sha256"])] + w["cnn_test"][1:]},
                                          {"R03-sides", "R07-eval-in-test", "R10-consistency"}),   # la riga è anche in eval
            "duplicated row": ({"eval": w["eval"] + [w["eval"][0]]}, {"R01-dup"}),
            "eval outside test": ({"eval": w["eval"][:1] + [w["mirror_train"][0]]}, {"R03-sides", "R07-eval-in-test"}),
            "test row removed": ({"cnn_test": w["cnn_test"][1:]}, {"R06-test", "R07-eval-in-test", "R08-eval-count", "R11-test-equals"}),
            "es subject leaks": ({"cnn_es": w["cnn_es"] + [w["cnn_train"][0]], "cnn_train": w["cnn_train"][1:]}, {"R02-fit-es"}),
            "duplicated content in test": ({"mirror_test": [dict(w["mirror_test"][1], sha256=w["mirror_test"][2]["sha256"]) if i == 1 else r
                                                            for i, r in enumerate(w["mirror_test"])]},
                                           {"R01-dup", "R10-consistency", "R12-dups-declared"}),
        }
        for name, (change, expected) in cases.items():
            self.assertEqual(failing({**w, **change}), expected, name)

    def test_intersection_counts_both_directions(self):
        w = casia_world()
        for exp, ok in ((2, True), (1, False), (3, False)):
            rules = json.loads(json.dumps(SYN_RULES))
            rules["rules"][7]["expected"] = {"key": exp, "video": 2}
            self.assertEqual("R08-eval-count" not in failing(w, rules), ok, exp)

    def test_internal_duplicates_declared(self):
        w = casia_world()
        dup = [dict(r, sha256=w["mirror_test"][0]["sha256"]) if i == 1 else r for i, r in enumerate(w["mirror_test"])]
        w2 = {**w, "mirror_test": dup, "cnn_test": list(dup), "eval": [dup[0], dup[-1]]}
        rules = json.loads(json.dumps(SYN_RULES))
        rules["rules"][0]["by"] = ["key"]
        rules["rules"][11]["expected"] = {"mirror_test": 1}
        self.assertNotIn("R12-dups-declared", failing(w2, rules))
        self.assertIn("R12-dups-declared", failing(w, rules))           # dichiarato 1, trovato 0: fallisce anche così
        rules["rules"][11]["expected"] = {"mirror_test": 2}
        self.assertIn("R12-dups-declared", failing(w2, rules))

    def test_partition_counts_duplicates_across_parts(self):
        w = casia_world()
        v = evaluate({**w, "cnn_es": w["cnn_es"] + [w["cnn_train"][0]]})["R05-split"]["checks"][0]
        self.assertEqual((v["n_extra"], v["n_missing"], v["outcome"]), (1, 0, "FAIL"))

    def test_key_consistency_every_field(self):
        w = casia_world()
        r0 = w["cnn_test"][0]
        for col in RM.CASIA_COLUMNS[1:]:
            changed = dict(r0, **{col: "x"})
            v = evaluate({**w, "cnn_test": [changed] + w["cnn_test"][1:]})["R10-consistency"]
            bad = [c["by"] for c in v["checks"] if c["outcome"] == "FAIL"]
            self.assertEqual(bad, [f"key->{col}"], col)

    def test_sha_swap_passes_set_rules_and_fails_consistency(self):
        w = casia_world()
        a, b = w["cnn_test"][0], w["cnn_test"][1]
        swapped = [dict(a, sha256=b["sha256"]), dict(b, sha256=a["sha256"])] + w["cnn_test"][2:]
        self.assertEqual(failing({**w, "cnn_test": swapped}), {"R10-consistency"})

    def test_report_rule_never_fails_and_vacuous_rule_fails(self):
        w = casia_world()
        rules = json.loads(json.dumps(SYN_RULES))
        rules["rules"][1]["severity"] = "report"
        v = evaluate({**w, "cnn_train": w["cnn_train"] + [w["cnn_es"][0]]}, rules)
        self.assertEqual(v["R02-fit-es"]["verdict"], "reported")
        v = evaluate({**w, "cnn_es": []})
        self.assertEqual(v["R02-fit-es"]["verdict"], "FAIL")
        self.assertTrue(any(c.get("vacuous") for c in v["R02-fit-es"]["checks"]))
        v = evaluate({**w, "mirror_test": []})                       # anche una regola "report" vuota fallisce
        self.assertEqual(v["R04-local"]["verdict"], "FAIL")

    def test_celeba_conversion_rule(self):
        rows = []
        for i in range(4200):
            rows.append(RM.celeba_row(i, i % 2, fake_hash(f"c{i}"), 10))
        sets = RM.celeba_conversion_sets(rows)
        self.assertEqual((len(sets["eval"]), len(sets["training"]), len(sets["validation"])), (300, 3000, 900))
        self.assertTrue(set(sets["eval"]) <= set(sets["validation"]))
        self.assertFalse(set(sets["training"]) & set(sets["validation"]))
        world = {"parquet": rows, "prepared_training": [r for r in rows if r["key"] in set(sets["training"])],
                 "prepared_validation": [r for r in rows if r["key"] in set(sets["validation"])],
                 "eval": [r for r in rows if r["key"] in set(sets["eval"])]}
        rule = {"rules": [{"id": "K", "kind": "celeba_conversion", "source": "parquet", "training": "prepared_training",
                           "validation": "prepared_validation", "eval": "eval", "by": ["key"], "severity": "must", "description": "d"}]}
        self.assertEqual(failing(world, rule, "celeba_spoof"), set())
        swapped = world["eval"][1:] + [world["prepared_validation"][-1]]
        self.assertEqual(failing({**world, "eval": swapped}, rule, "celeba_spoof"), {"K"})


class TestCli(unittest.TestCase):
    def test_clean_case_passes(self):
        with tempfile.TemporaryDirectory() as d:
            r, rep = run_cli(Path(d), casia_world())
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertTrue(rep["passed"])
            pub = (Path(d) / "out" / "overlap_report_public.json").read_text()
            self.assertNotIn("examples", pub)

    def test_violation_exit_code_and_no_traceback(self):
        w = casia_world()
        with tempfile.TemporaryDirectory() as d:
            r, rep = run_cli(Path(d), {**w, "cnn_train": w["cnn_train"] + [w["cnn_es"][0]]})
            self.assertEqual(r.returncode, 1)
            self.assertIn("ERROR:", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            self.assertIn("R02-fit-es", r.stderr)
            self.assertTrue(any(c.get("examples") for v in rep["rules"] for c in v["checks"]))

    def test_row_validation_stops_every_rule(self):
        w = casia_world()
        bad = [dict(w["cnn_test"][0], mirror_split="dev")] + w["cnn_test"][1:]
        with tempfile.TemporaryDirectory() as d:
            r, rep = run_cli(Path(d), {**w, "cnn_test": bad})
            self.assertEqual(r.returncode, 1)
            self.assertIn("row validation failed", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            self.assertTrue(rep["row_validation_failed"])
            self.assertEqual(rep["rules"], [])

    def test_truncated_or_altered_manifest_is_refused(self):
        w = casia_world()
        with tempfile.TemporaryDirectory() as d:
            rp = write_world(Path(d), w)
            f = Path(d) / "manifests" / RM.SPEC["casia_fasd"]["manifests"]["eval"][0]
            f.write_text("\n".join(f.read_text().splitlines()[:-1]) + "\n")
            r = subprocess.run([sys.executable, str(SCRIPT), "check-overlap", "--dataset", "casia_fasd", "--rules", str(rp),
                                "--manifests-dir", str(Path(d) / "manifests"), "--out", str(Path(d) / "out")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)
            self.assertIn("differ from those declared", r.stderr)
            self.assertNotIn("Traceback", r.stderr)

    def run_declared(self, d: Path, rp: Path):
        return subprocess.run([sys.executable, str(SCRIPT), "check-overlap", "--dataset", "casia_fasd", "--rules", str(rp),
                               "--manifests-dir", str(d / "manifests"), "--out", str(d / "out")], capture_output=True, text=True)

    def test_expected_rows_and_sha_checked_separately(self):
        w = casia_world()
        for what in ("rows", "sha"):
            with tempfile.TemporaryDirectory() as d:
                rp = write_world(Path(d), w)
                doc = json.loads(rp.read_text())
                if what == "rows":
                    doc["manifests"]["eval"]["expected_rows"] += 1           # SHA-256 corretto, righe sbagliate
                else:
                    doc["manifests"]["eval"]["expected_sha256"] = "0" * 64     # righe corrette, SHA-256 sbagliato
                rp.write_text(json.dumps(doc))
                r = self.run_declared(Path(d), rp)
                self.assertEqual(r.returncode, 1, what)
                self.assertIn("rows, expected" if what == "rows" else "differs from the declared", r.stderr, what)
                self.assertNotIn("Traceback", r.stderr)

    def test_non_canonical_files_are_refused(self):
        w = casia_world()
        f_name = RM.SPEC["casia_fasd"]["manifests"]["eval"][0]
        for what in ("crlf", "bom", "reordered", "blank line"):
            with tempfile.TemporaryDirectory() as d:
                rp = write_world(Path(d), w)
                f = Path(d) / "manifests" / f_name
                raw = f.read_bytes()
                lines = raw.decode().splitlines()
                new = {"crlf": raw.replace(b"\n", b"\r\n"), "bom": b"\xef\xbb\xbf" + raw,
                       "reordered": ("\n".join([lines[0]] + lines[1:][::-1]) + "\n").encode(), "blank line": raw + b"\n"}[what]
                f.write_bytes(new)
                doc = json.loads(rp.read_text()); doc["manifests"]["eval"]["expected_sha256"] = RM.sha256(f)
                rp.write_text(json.dumps(doc))
                r = self.run_declared(Path(d), rp)
                self.assertEqual(r.returncode, 1, what)
                self.assertIn("canonical form" if what != "blank line" else "casia_eval.csv", r.stderr, what)
                self.assertNotIn("Traceback", r.stderr)

    def test_header_must_be_exact(self):
        w = casia_world()
        with tempfile.TemporaryDirectory() as d:
            rp = write_world(Path(d), w)
            f = Path(d) / "manifests" / RM.SPEC["casia_fasd"]["manifests"]["eval"][0]
            lines = [ln.split(",") for ln in f.read_text().splitlines()]
            for ln in lines:                                          # colonne e valori scambiati in modo coerente
                ln[1], ln[2] = ln[2], ln[1]
            f.write_text("\n".join(",".join(ln) for ln in lines) + "\n")
            doc = json.loads(rp.read_text()); doc["manifests"]["eval"]["expected_sha256"] = RM.sha256(f)
            rp.write_text(json.dumps(doc))
            r = subprocess.run([sys.executable, str(SCRIPT), "check-overlap", "--dataset", "casia_fasd", "--rules", str(rp),
                                "--manifests-dir", str(Path(d) / "manifests"), "--out", str(Path(d) / "out")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)
            self.assertIn("header", r.stderr)
            self.assertNotIn("Traceback", r.stderr)

    def test_spread_with_small_per_class(self):
        w = casia_world()                                             # 6 righe per classe nel train: per_class 3 < 6
        rules = {"rules": [{"id": "K", "kind": "equals_spread", "a": "cnn_train", "source": "mirror_train", "per_class": 3, "by": ["key"],
                            "severity": "must", "description": "d"}]}
        keys = [sorted(r["key"] for r in w["mirror_train"] if r["label"] == lab) for lab in ("0", "1")]
        want = {k for ks in keys for k in (ks[0], ks[2], ks[4])}      # spread(6, 3): passo 2, indici 0, 2, 4
        self.assertEqual(set(RM.spread_keys(w["mirror_train"], 3)), want)
        exp = [r for r in w["mirror_train"] if r["key"] in want]
        self.assertEqual(failing({**w, "cnn_train": exp}, rules), set())
        self.assertEqual(failing({**w, "cnn_train": exp[:-1]}, rules), {"K"})

    def test_self_test_requires_exactly_the_expected_rules(self):
        w = casia_world()
        with tempfile.TemporaryDirectory() as d:
            rp = write_world(Path(d), w)
            doc = json.loads(rp.read_text())
            fault = ("X", "es row in fit", {"cnn_train": w["cnn_train"] + [w["cnn_es"][0]]}, {"R02-fit-es", "R05-split"})
            wrong = (fault[0], fault[1], fault[2], {"R02-fit-es", "R10-consistency"})              # regole dichiarate diverse
            saved = RM.self_test_faults
            res = []
            try:
                with quiet():
                    for faults in ([fault] * 8, [fault] * 7 + [wrong], [fault] * 7):
                        faults = [(f"X{i}",) + f[1:] for i, f in enumerate(faults)]
                        RM.self_test_faults = lambda ds, m, f=faults: f
                        res.append(RM.run_self_test("casia_fasd", doc, rp, Path(d) / "manifests", w, Path(d)))
            finally:
                RM.self_test_faults = saved
        self.assertEqual(res, [True, False, False])                          # esatto, regola diversa, meno di 8 guasti

    def test_counts_rule(self):
        w = casia_world()
        exp = {n: {k: RM.counts("casia_fasd", rows, False)[k] for k in RM.count_fields("casia_fasd")} for n, rows in w.items()}
        rule = {"rules": [{"id": "N", "kind": "counts", "by": ["key"], "severity": "must", "expected": exp, "description": "d"}]}
        self.assertEqual(failing(w, rule), set())
        moved = {**w, "cnn_train": [r for r in w["cnn_train"] if r["subject"] != "2"],
                 "cnn_es": w["cnn_es"] + [r for r in w["cnn_train"] if r["subject"] == "2"]}
        self.assertEqual(failing(moved), set())                              # partizione e disgiunzione intatte...
        self.assertEqual(failing(moved, rule), {"N"})                        # ...i conteggi dichiarati no
        doc = {"dataset": "casia_fasd", "manifests": {n: {"file": RM.SPEC["casia_fasd"]["manifests"][n][0], "expected_rows": 1,
                                                          "expected_sha256": "a" * 64} for n in w}, "rules": json.loads(json.dumps(rule["rules"]))}
        self.assertEqual(RM.rule_problems("casia_fasd", doc), [])
        doc["rules"][0]["expected"].pop("eval")
        self.assertTrue(RM.rule_problems("casia_fasd", doc))


class TestPublicScan(unittest.TestCase):
    def test_record_problems(self):
        w = casia_world()
        ids = RM.identifiers("casia_fasd", w)
        name = w["mirror_test"][0]["key"].split("/", 1)[1]
        clean = "subjects 30, videos 360, frames 2,408; example 9_HR_4.avi_620_fake.jpg; hash " + "b" * 64 + "; " + SK("train", 99)
        self.assertEqual(RM.record_problems(clean, ids), [])
        for text in ("subject " + SK("test", 1) + " here", "video " + SK("train", 3, 3), SK("train", 2, "HR_4") + " (subject part)", f"file {name}", "sha " + w["eval"][0]["sha256"]):
            self.assertTrue(RM.record_problems(text, ids), text)
        cids = RM.identifiers("celeba_spoof", {"parquet": [RM.celeba_row(42, 0, fake_hash("q"), 3)]})
        self.assertTrue(RM.record_problems("row " + CK(42), cids))
        self.assertEqual(RM.record_problems("row " + CK(43), cids), [])
        self.assertTrue(RM.public_text_problems("path " + "/Us" + "ers/x", ids))


# ----------------------------------------------------------------------------- split (logica del commit 0ac7143)

class TestSplits(unittest.TestCase):
    def test_branch_by_number_of_groups(self):
        import numpy as np
        y = np.array([0, 1] * 20)
        _, va, branch = RM.official_split(y, np.array([str(i % 10) for i in range(40)]))
        self.assertIn("GroupShuffleSplit", branch)
        _, va, branch = RM.official_split(y, np.array([str(i % 4) for i in range(40)]))
        self.assertIn("test_size=0.1", branch)
        self.assertEqual(len(va), 4)

    def test_sampling_order_and_groups(self):
        w = casia_world()
        smp = RM.sampling("casia_fasd", w["mirror_train"], w["mirror_test"])
        self.assertEqual(smp["train"]["y"].tolist(), sorted(smp["train"]["y"].tolist()))      # bona fide e poi attacchi
        self.assertEqual(smp["train"]["keys"][:2], sorted(r["key"] for r in w["mirror_train"] if r["label"] == "0")[:2])
        self.assertEqual(set(smp["train"]["s"].tolist()), {"1", "2", "3"})
        c = RM.sampling("celeba_spoof", [RM.celeba_row(7, 0, fake_hash(7), 1)], [RM.celeba_row(8, 1, fake_hash(8), 1)])
        self.assertEqual(c["train"]["s"].tolist(), [CK(7) + ".jpg"])                     # 0ac7143: gruppo = nome del file

    def test_pooled_sizes(self):
        import numpy as np
        f, v, t = RM.pooled_split(np.array([0] * 404 + [1] * 1251), np.array([0] * 591 + [1] * 1500))
        self.assertEqual((len(f), len(v), len(t)), (2696, 300, 750))
        self.assertFalse(set(f) & set(v) or set(f) & set(t) or set(v) & set(t))

    def test_library_drift_canary(self):
        """Ingresso fisso: se scikit-learn dà uno split diverso da quello della versione dichiarata, reconstruct non
        riprodurrebbe i manifest dichiarati. Il test fallisce con un messaggio esplicito (deriva della libreria)."""
        import numpy as np
        sizes = [3, 5, 7, 11, 13, 4, 6, 8, 10, 12, 9, 2, 15, 14, 1, 16, 5, 7, 3, 9]
        groups = np.array([str(i + 1) for i, n in enumerate(sizes) for _ in range(n)])
        y = np.array([i % 2 for i in range(len(groups))])
        _, va, _ = RM.official_split(y, groups)
        f, v, t = RM.pooled_split(y[:100], y[100:])
        got = {"val_groups": sorted(set(groups[va].tolist()), key=int), "pooled_val_head": sorted(v.tolist())[:5],
               "pooled_test_head": sorted(t.tolist())[:5]}
        expected = CANARY
        self.assertEqual(got, expected, f"scikit-learn {RM.library_versions()['scikit-learn']} gives a different split than "
                         f"scikit-learn {RM.BUILT_WITH['scikit-learn']} on a fixed input: `reconstruct` would not reproduce the "
                         "declared manifests (drift of the library, not of the data); the declared manifests remain the reference")

    def test_reconstruct_drift_message(self):
        w = casia_world()
        with tempfile.TemporaryDirectory() as d:
            rp = write_world(Path(d), {**w, "cnn_es": w["cnn_es"][:1], "cnn_train": w["cnn_train"] + w["cnn_es"][1:],
                                       "pooled_train": w["cnn_train"], "pooled_es": w["cnn_es"], "pooled_test": w["cnn_test"]})
            r = subprocess.run([sys.executable, str(SCRIPT), "reconstruct", "--dataset", "casia_fasd", "--rules", str(rp),
                                "--manifests-dir", str(Path(d) / "manifests")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)
            self.assertIn("the splits rebuilt from the manifests differ", r.stderr)
            self.assertNotIn("Traceback", r.stderr)


# regole versionate bloccate: righe attese, [id, tipo, severità, dimensioni] e SHA-256 canonico (rules_digest)
LOCKED_RULES = {
    "casia_fasd": ({"mirror_train": 1655, "mirror_test": 2408, "prepared_training": 1655, "prepared_validation": 2408, "eval": 300, "cnn_train": 1339, "cnn_es": 316, "cnn_test": 2091, "pooled_train": 2696, "pooled_es": 300, "pooled_test": 750},
        [["C01-internal-duplicates", "no_internal_duplicates", "must", ["key", "hash"]],
         ["C02-mirror-train-vs-test", "disjoint", "must", ["key", "hash"]],
         ["C03-identity-train-vs-test", "declared_not_checkable", "report", None],
         ["C04-local-numbers-train-vs-test", "disjoint", "report", ["local_subject"]],
         ["C05-prepared-training-equals-mirror", "partition", "must", ["key"]],
         ["C06-prepared-validation-equals-mirror", "partition", "must", ["key"]],
         ["C07-cnn-split-partition", "spread_partition", "must", ["key"]],
         ["C08-fit-vs-es", "disjoint", "must", ["key", "hash", "subject", "video"]],
         ["C09-train-side-vs-test-side", "disjoint", "must", ["key", "hash", "subject", "video"]],
         ["C10-cnn-test-spread", "equals_spread", "must", ["key"]],
         ["C11-eval-spread", "equals_spread", "must", ["key"]],
         ["C12-eval-in-cnn-test", "subset", "must", ["key", "hash"]],
         ["C13-pooled-partition", "partition", "must", ["key"]],
         ["C14-pooled-sides-by-hash", "disjoint", "must", ["key", "hash"]],
         ["C15-pooled-eval-leak", "intersection", "must", ["key", "hash"]],
         ["C16-pooled-subject-video-leak", "intersection", "must", ["subject", "video"]],
         ["C17-key-consistency", "key_consistency", "must", ["key"]],
         ["C18-declared-counts", "counts", "must", ["key"]]],
        "cb12a7bc2c359559c20088976b57147bdb28604c3e9dd8fc700d40113ba513a6"),
    "celeba_spoof": ({"parquet": 6678, "prepared_training": 3000, "prepared_validation": 900, "eval": 300, "cnn_train": 2400, "cnn_es": 600, "cnn_test": 900, "pooled_train": 2808, "pooled_es": 312, "pooled_test": 780},
        [["S01-internal-duplicate-keys", "no_internal_duplicates", "must", ["key"]],
         ["S02-exact-duplicate-images", "internal_duplicates", "must", ["hash"]],
         ["S03-conversion-rule", "celeba_conversion", "must", ["key"]],
         ["S04-eval-in-prepared-validation", "subset", "must", ["key", "hash"]],
         ["S05-official-split", "partition", "must", ["key"]],
         ["S06-fit-vs-es", "disjoint", "must", ["key", "hash"]],
         ["S07-train-side-vs-test-side", "disjoint", "must", ["key", "hash"]],
         ["S08-identity-not-checkable", "declared_not_checkable", "report", None],
         ["S09-all-from-official-test", "subset", "must", ["key", "hash"]],
         ["S10-pooled-partition", "partition", "must", ["key"]],
         ["S11-cnn-test-equals-prepared", "partition", "must", ["key"]],
         ["S12-pooled-eval-leak", "intersection", "must", ["key", "hash"]],
         ["S13-key-consistency", "key_consistency", "must", ["key"]],
         ["S14-declared-counts", "counts", "must", ["key"]]],
        "4e29e2ad1461993d4ef945afcdb0c840baa928f50ae33faef47444afed5337a7"),
}


# valori registrati con scikit-learn 1.3.2 e numpy 1.26.4 (versioni con cui sono stati scritti i manifest dichiarati)
CANARY = {"val_groups": ["1", "5", "7", "10"], "pooled_val_head": [12, 32, 34, 47, 50], "pooled_test_head": [1, 2, 4, 15, 17]}


# ----------------------------------------------------------------------------- dati reali (facoltativo)

def real_dir() -> Path:
    return Path(REAL_DIR) if REAL_DIR else RM.OUT_DIR


def need_real(test, ds):
    man_dir = real_dir() / ds / "manifests"
    doc = json.loads(RM.rules_path(ds).read_text(encoding="utf-8"))
    missing = [m["file"] for m in doc["manifests"].values() if not (man_dir / m["file"]).is_file()]
    if missing:
        if REQUIRE_REAL:
            test.fail(f"real manifests missing in {man_dir.name}: {missing}")
        test.skipTest(f"real {ds} manifests not available (run rgb_manifests.py build, or pass --real-dir)")
    return man_dir


class TestRealData(unittest.TestCase):
    def run_rm(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], cwd=ROOT, capture_output=True, text=True)

    def check_dataset(self, ds, rows, n_faults):
        man_dir = need_real(self, ds)
        with tempfile.TemporaryDirectory() as d:
            r = self.run_rm("check-overlap", "--dataset", ds, "--self-test", "--manifests-dir", man_dir, "--out", d)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            rep = json.loads((Path(d) / "overlap_report.json").read_text())
            st_rep = json.loads((Path(d) / "overlap_self_test.json").read_text())
        self.assertEqual({n: v["rows"] for n, v in rep["manifests"].items()}, rows)
        self.assertEqual((st_rep["n_faults"], st_rep["n_detected_as_expected"]), (n_faults, n_faults))
        self.assertFalse(any(f["stopped_by_checksum"] for f in st_rep["faults"]))
        r = self.run_rm("reconstruct", "--dataset", ds, "--manifests-dir", man_dir)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stdout.count("identical: yes"), 6)
        # aggregato pubblico: accanto ai manifest (risultati di build) oppure nella cartella pubblica che contiene livenesslab/
        agg = next((p for p in (real_dir() / ds / f"{ds}_manifests_aggregate.json", ROOT.parent / f"{ds}_manifests_aggregate.json")
                    if p.is_file()), real_dir() / ds / f"{ds}_manifests_aggregate.json")
        if not agg.is_file():
            if REQUIRE_REAL:
                self.fail(f"{agg.name} not found")
            self.skipTest("public aggregate not available")
        r = self.run_rm("recount", "--dataset", ds, "--manifests-dir", man_dir, "--aggregate", agg)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout

    def test_real_casia(self):
        out = self.check_dataset("casia_fasd", {"mirror_train": 1655, "mirror_test": 2408, "prepared_training": 1655, "prepared_validation": 2408,
                                                "eval": 300, "cnn_train": 1339, "cnn_es": 316, "cnn_test": 2091, "pooled_train": 2696,
                                                "pooled_es": 300, "pooled_test": 750}, 10)
        self.assertIn("eval images in pooled fit or validation: 235; subjects on both sides: 50", out)
        self.assertIn("videos on both sides: 444", out)

    def test_real_celeba(self):
        out = self.check_dataset("celeba_spoof", {"parquet": 6678, "prepared_training": 3000, "prepared_validation": 900, "eval": 300,
                                                  "cnn_train": 2400, "cnn_es": 600, "cnn_test": 900, "pooled_train": 2808, "pooled_es": 312,
                                                  "pooled_test": 780}, 10)
        self.assertIn("eval images in pooled fit or validation: 244", out)


def main() -> int:
    global REQUIRE_REAL, REAL_DIR
    ap = argparse.ArgumentParser(description="Test di rgb_manifests.py su dati sintetici (e sui manifest reali, se presenti)")
    ap.add_argument("--verbose", action="store_true", help="stampa il log completo di unittest")
    ap.add_argument("--require-real", action="store_true", help="i manifest reali devono esserci (niente salto dei test sui dati reali)")
    ap.add_argument("--real-dir", default=None, help="cartella con <dataset>/manifests e gli aggregati (default results/m2_rgb)")
    args = ap.parse_args()
    REQUIRE_REAL, REAL_DIR = args.require_real, args.real_dir
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c)
                               for c in (TestFields, TestRowValidation, TestRules, TestWording, TestRuleEngine, TestCli, TestPublicScan, TestSplits,
                                         TestRealData))
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    if args.verbose or not result.wasSuccessful():
        print(log.getvalue())
    print(f"tests run: {result.testsRun}, failures: {len(result.failures)}, errors: {len(result.errors)}, skipped: {len(result.skipped)}")
    print("RESULT: " + ("PASS" if result.wasSuccessful() else "FAIL"))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
