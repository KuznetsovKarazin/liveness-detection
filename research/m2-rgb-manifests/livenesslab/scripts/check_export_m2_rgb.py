"""
Test offline di scripts/export_m2_rgb.py (seconda tranche M2), senza dati, senza rete e senza eseguire l'export:
elenchi del pacchetto (ogni file esiste, ogni modulo di scripts/ importato è nell'elenco, nessun manifest né riepilogo
privato nella parte pubblica, registri dei dati solo nel riservato), run-id distinto da quello della tranche NUAA,
segnaposto del README e del run report (ogni campo del modello ha un valore anche senza risultati), stato di consegna,
ricerca di record per soggetto, video o immagine nella parte pubblica e file condivisi con la tranche NUAA.

  .venv/bin/python scripts/check_export_m2_rgb.py            # uscita 1 se un test fallisce
"""
import argparse
import contextlib
import importlib.util
import io
import os
import re
import shutil
import string
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_m2_rgb.py"


def load_module():
    spec = importlib.util.spec_from_file_location("export_m2_rgb_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EX = load_module()


DEV_ONLY = "the dataset register drafts are restricted: they are in the development repository only, not in the public package"


def has_registers() -> bool:
    return all((EX.DELIV / n).is_file() for n in EX.RESTRICTED_DOCS)


def public_doc(n: str):
    """Documento pubblico: in delivery/m2/rgb (repository di sviluppo) o nella cartella che contiene livenesslab/."""
    return next((p for p in (EX.DELIV / n, ROOT.parent / n) if p.is_file()), None)


class TestLists(unittest.TestCase):
    def need_registers(self):
        if not has_registers():
            self.skipTest(DEV_ONLY)

    def test_code_and_documents_exist(self):
        self.assertEqual([c for c in EX.CODE if not (ROOT / c).is_file()], [])
        for t in ("README.md", "run-report.md.tmpl"):                 # modelli dei testi pubblici: anche nel pacchetto
            self.assertTrue((EX.DELIV / t).is_file(), t)
        missing = [n for n in EX.PUBLIC_DOCS if public_doc(n) is None]
        if missing and not (ROOT.parent / "README.md").is_file():
            self.skipTest("the public documents are next to livenesslab/ only in the delivered folder, not in a bare copy of the code")
        self.assertEqual(missing, [])
        if has_registers():
            self.assertEqual([n for n in EX.RESTRICTED_DOCS if not (EX.DELIV / n).is_file()], [])

    def test_code_imports_are_listed(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(EX.copy_code(Path(d)), [])
            self.assertEqual(EX.code_import_problems(Path(d)), [])
            # un modulo di scripts/ fuori dall'elenco viene rifiutato (export_m2.py esiste solo nel repository di sviluppo)
            if (ROOT / "scripts" / "export_m2.py").is_file():
                (Path(d) / "scripts" / "extra.py").write_text("import export_m2\n")
                self.assertTrue(EX.code_import_problems(Path(d)))

    def test_public_and_restricted_lists(self):
        public = EX.PUBLIC_DOCS + list(EX.PUBLIC_RESULTS.values()) + EX.CODE
        self.assertFalse([f for f in public if f.endswith(".csv") or "summary" in f or "register" in f or "self_test" in f])
        self.assertNotIn("overlap_report.json", EX.PUBLIC_RESULTS)
        self.assertFalse(set(EX.RESTRICTED_DOCS) & set(EX.PUBLIC_DOCS))

    def test_run_id_distinct_from_nuaa(self):
        self.assertTrue(EX.RUN_ID_RX.match("20261009-M2-rgb-seed42-abcdef0"))
        self.assertFalse(EX.RUN_ID_RX.match("20261009-M2-seed42-abcdef0"))     # forma della tranche NUAA: rifiutata
        self.assertIn("results/m2", EX.READ_ONLY)                             # i risultati NUAA non vengono mai toccati

    def test_registers_have_run_id_and_authorisation(self):
        self.need_registers()
        for n in EX.RESTRICTED_DOCS:
            text = (EX.DELIV / n).read_text(encoding="utf-8")
            out = EX.fill_run_id(text, "20261009-M2-rgb-seed42-abcdef0")
            self.assertIn("02_Experiments/M2/20261009-M2-rgb-seed42-abcdef0/", out)
            self.assertIn("authorisation: handled by the coordinator (task D1)", out)
            for field in ("Source URL", "Download date", "Version / archive", "Licence / README / authorisation", "Location"):
                self.assertIn(f"| {field} |", out, n)
            self.assertNotIn("Terms of use", out)
            self.assertNotIn("not re-verified", out)
        self.assertIn("personal-data", (EX.DELIV / "dataset_register_celeba_spoof.md").read_text(encoding="utf-8"))

    def test_templates(self):
        readme = (EX.DELIV / "README.md").read_text(encoding="utf-8")
        for ph in ("{commit}", "{run_id}", "{delivery_status}"):
            self.assertIn(ph, readme)
        txt = EX.delivery_status("20261009-M2-rgb-seed42-abcdef0")[0]
        self.assertIn("kept locally and not uploaded", txt)
        self.assertIn("as a supplementary document outside the NUAA run", txt)
        self.assertIn("`M2_provenienza_RGB_20261009.pdf` was placed on 9 October 2026", txt)
        self.assertIn("the restricted RGB run folder stays local", txt)
        self.assertNotIn("note is uploaded", txt)
        tmpl = (EX.DELIV / "run-report.md.tmpl").read_text(encoding="utf-8")
        keys = {f for _, f, _, _ in string.Formatter().parse(tmpl) if f}

        class A:
            run_id, strict, out = None, False, str(ROOT / "export")
        info = {k: "x" for k in ("verification", "build", "tests", "package", "nuaa", "export")}
        with tempfile.TemporaryDirectory() as d:
            none = {ds: None for ds in EX.DATASETS}
            fields = EX.report_fields(none, none, none, none, "20261009-M2-rgb-seed42-abcdef0", "abcdef0", A, Path(d), (1, 1), info, [], [])
        fields["tree_line"] = "working tree clean: yes"
        self.assertEqual(keys - set(fields), set())
        tmpl.format(**fields)
        self.assertEqual(re.findall(r"^## (.+)$", tmpl, re.M), ["Identity", "Data and protocol", "Evidence", "Interpretation"])

    def test_delivery_status_with_upload(self):
        with tempfile.TemporaryDirectory() as d:
            m2 = Path(d) / "m2"
            for ds in EX.DATASETS:
                (m2 / ds / "manifests").mkdir(parents=True)
                (m2 / ds / "manifests" / "a.csv").write_text(ds)
            sums = Path(d) / "SHA256SUMS"
            sums.write_text("".join(f"{EX.sha256(p)}  {p.relative_to(m2).as_posix()}\n" for p in sorted(m2.rglob("*.csv")))
                            + f"{'0' * 64}  README.md\n")
            txt, problems = EX.delivery_status("20261010-M2-rgb-seed42-abcdef0", "20261009-M2-rgb-seed42-1234567", "2026-10-09", sums, m2)
            self.assertEqual(problems, [])
            self.assertIn("uploaded to Drive on 2026-10-09", txt)
            self.assertIn("2 manifests are byte-identical to that upload", txt)
            self.assertIn("other files are not compared", txt)
            self.assertNotIn("changes only code", txt)
            (m2 / "celeba_spoof" / "manifests" / "a.csv").write_text("changed")
            _, problems = EX.delivery_status("20261010-M2-rgb-seed42-abcdef0", "20261009-M2-rgb-seed42-1234567", "2026-10-09", sums, m2)
            self.assertTrue(problems)

    def test_dirty_tree_and_strict_gates(self):
        self.assertTrue(EX.tree_gate(True, False))            # albero modificato senza --trial: rifiutato
        self.assertEqual(EX.tree_gate(True, True), "")         # prova esplicita
        self.assertEqual(EX.tree_gate(False, False), "")
        rid = "20261008-M2-rgb-seed42-abcdef0"
        self.assertEqual(EX.effective_run_id(rid, True, True), rid + "-trial")      # prova da albero modificato
        self.assertEqual(EX.effective_run_id(rid, False, True), rid)                 # albero pulito: run-id normale
        self.assertEqual(EX.effective_run_id(rid + "-trial", True, True), rid + "-trial")
        self.assertTrue(EX.RUN_ID_RX.match(rid + "-trial"))
        self.assertIn("not for delivery", EX.TRIAL_NOTE)
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(EX.nuaa_package_problem(Path(d)))  # pacchetto NUAA assente: con --strict l'export si ferma
        if (EX.DELIV / "run-report.md.tmpl").is_file():
            self.assertIn("{tree_line}", (EX.DELIV / "run-report.md.tmpl").read_text(encoding="utf-8"))
        self.assertTrue(re.fullmatch(r"\d{8}", EX.today()))
        self.assertEqual([c for c in EX.CODE if c.startswith("requirements")], ["requirements.txt"])
        # gli script dell'inventario pubblicato sulla scheda sono nel pacchetto (promessi per questa PR)
        self.assertEqual({c for c in EX.CODE if c.startswith("scripts/m2_inventory/")},
                         {f"scripts/m2_inventory/{p.name}" for p in (ROOT / "scripts" / "m2_inventory").glob("*.py")})

    def test_local_patterns_file_is_never_exported(self):
        import subprocess
        self.assertFalse([c for c in EX.CODE + EX.PUBLIC_DOCS + EX.RESTRICTED_DOCS if c.startswith("config/") or "local_scan_patterns" in c])
        with tempfile.TemporaryDirectory() as d:
            env = {**os.environ, "LIVENESSLAB_LOCAL_SCAN_PATTERNS": str(Path(d) / "absent.json")}
            r = subprocess.run([sys.executable, str(SCRIPT), "--strict", "--dry-run", "--out", str(Path(d) / "out")], cwd=ROOT,
                               capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 1)                  # export --strict senza il file locale: rifiutato
            self.assertIn("local scan patterns file", r.stderr)
            self.assertFalse((Path(d) / "out").exists())

    def test_stale_or_invalid_artefacts_are_refused(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            f = Path(d)
            man = {"eval": {"file": "manifests/casia_eval.csv", "sha256": "a" * 64}}
            good = {"casia_fasd_manifests_summary.json": {"passed": True, "commit": "c" * 40, "manifests": man},
                    "casia_fasd_manifests_aggregate.json": {"passed": True, "commit": "c" * 40, "manifests": man},
                    "overlap_report.json": {"passed": True, "commit": "c" * 40, "rules_sha256": "r", "manifests": {"eval": {"file": "casia_eval.csv", "sha256": "a" * 64}}},
                    "overlap_report_public.json": {"passed": True, "commit": "c" * 40, "manifests": {"eval": {"file": "casia_eval.csv", "sha256": "a" * 64}}},
                    "overlap_self_test.json": {"passed": True, "commit": "c" * 40, "rules_sha256": "r"}}
            def write(docs):
                for n, v in docs.items():
                    (f / n).write_text(json.dumps(v))
                for n in ("casia_fasd_m2_report.md", "casia_fasd_m2_report_public.md", "overlap_report.md", "overlap_report_public.md"):
                    (f / n).write_text("x")
            write(good)
            self.assertEqual(EX.artefact_problems(f, "casia_fasd", "c" * 40), [])
            self.assertTrue(EX.artefact_problems(f, "casia_fasd", "d" * 40))                     # artefatti di un altro commit
            for name, change in (("overlap_report_public.json", {"passed": False}), ("casia_fasd_manifests_aggregate.json", {"manifests": {}}),
                                 ("overlap_self_test.json", {"rules_sha256": "other"})):
                write(good); write({name: {**good[name], **change}})
                self.assertTrue(EX.artefact_problems(f, "casia_fasd", "c" * 40), name)
            write(good); (f / "overlap_report.md").unlink()
            self.assertTrue(EX.artefact_problems(f, "casia_fasd", "c" * 40))
            write(good); (f / "x.json.tmp").write_text("x")
            self.assertTrue(EX.artefact_problems(f, "casia_fasd", "c" * 40))

    def test_main_refuses_failed_artefacts(self):
        """main controlla gli artefatti di ogni dataset con artefact_problems: con un run finto che lascia un rapporto
        pubblico delle sovrapposizioni fallito, l'export rifiuta (codice 1) e lo dice."""
        import contextlib
        import json
        import subprocess
        import shutil
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=EX.ROOT, capture_output=True, text=True).stdout.strip() or "n/d"
        with tempfile.TemporaryDirectory() as d:
            m2 = Path(d) / "m2"
            # documenti di consegna fittizi: il test gira anche dal pacchetto pubblico, che non ha i modelli del repository
            deliv = Path(d) / "deliv"
            deliv.mkdir()
            (deliv / "README.md").write_text("{commit} {run_id}\n\n{delivery_status}\n")
            (deliv / "run-report.md.tmpl").write_text("run {run_id}\n")
            for n in EX.RESTRICTED_DOCS:
                (deliv / n).write_text("register draft {run_id}\n")
            for n in EX.PUBLIC_DOCS:
                src = ROOT / "delivery" / "m2" / "rgb" / n
                (deliv / n).write_bytes(src.read_bytes() if src.is_file() else b"document\n")
            man, ovm = {}, {}                      # nessun manifest: artefatti coerenti fra loro, solo il rapporto pubblico CelebA fallito

            def fake_run(*cmd, env=None, cwd=None):
                for ds in EX.DATASETS:
                    f = m2 / ds
                    (f / "manifests").mkdir(parents=True, exist_ok=True)
                    docs = {f"{ds}_manifests_summary.json": {"passed": True, "commit": commit, "manifests": man},
                            f"{ds}_manifests_aggregate.json": {"passed": True, "commit": commit, "manifests": man},
                            "overlap_report.json": {"passed": True, "commit": commit, "rules_sha256": "r", "manifests": ovm, "rules": [], "n_must_failed": 0},
                            "overlap_report_public.json": {"passed": ds != "celeba_spoof", "commit": commit, "manifests": ovm, "rules": [], "n_must_failed": 0},
                            "overlap_self_test.json": {"passed": True, "commit": commit, "rules_sha256": "r"}}
                    for n, v in docs.items():
                        (f / n).write_text(json.dumps(v))
                    for n in (f"{ds}_m2_report.md", f"{ds}_m2_report_public.md", "overlap_report.md", "overlap_report_public.md", "overlap_self_test.md"):
                        (f / n).write_text("x\n")
                return 0, "tests run: 1, failures: 0, errors: 0, skipped: 0\nbuild checks passed (1 checks)\n"
            saved = (EX.run, sys.argv, EX.DELIV)
            EX.run, EX.DELIV = fake_run, deliv
            sys.argv = ["export_m2_rgb.py", "--trial", "--run-id", "20261013-M2-rgb-seed42-test000", "--out", str(Path(d) / "out"), "--m2", str(m2)]
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                    rc = EX.main()
            finally:
                EX.run, sys.argv, EX.DELIV = saved
        self.assertEqual(rc, 1)
        self.assertIn("celeba_spoof: overlap_report_public.json is not a passed artefact", buf.getvalue())
        self.assertNotIn("casia_fasd: overlap_report_public.json is not a passed artefact", buf.getvalue())

    def test_overlap_manifests_differ_from_summary(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            f = Path(d)
            man = {"eval": {"file": "manifests/casia_eval.csv", "sha256": "a" * 64}}
            for n, v in (("casia_fasd_manifests_summary.json", {"passed": True, "commit": "c", "manifests": man}),
                         ("casia_fasd_manifests_aggregate.json", {"passed": True, "commit": "c", "manifests": man}),
                         ("overlap_report.json", {"passed": True, "commit": "c", "rules_sha256": "r", "manifests": {"eval": {"file": "casia_eval.csv", "sha256": "b" * 64}}}),
                         ("overlap_report_public.json", {"passed": True, "commit": "c", "manifests": {"eval": {"file": "casia_eval.csv", "sha256": "a" * 64}}}),
                         ("overlap_self_test.json", {"passed": True, "commit": "c", "rules_sha256": "r"})):
                (f / n).write_text(json.dumps(v))
            for n in ("casia_fasd_m2_report.md", "casia_fasd_m2_report_public.md", "overlap_report.md", "overlap_report_public.md"):
                (f / n).write_text("x")
            problems = EX.artefact_problems(f, "casia_fasd", "c")
        self.assertEqual(problems, ["casia_fasd: overlap_report.json refers to other manifests than the summary"])

    def test_helpers_identical_to_the_original_exports(self):
        """export_helpers.py è una copia testuale delle funzioni di export_m1.py ed export_m2.py (solo nel repository di
        sviluppo, non pubblicabili): sorgente di ogni funzione e FORBIDDEN_EXT identici."""
        import inspect
        m1, m2 = ROOT / "scripts" / "export_m1.py", ROOT / "scripts" / "export_m2.py"
        if not (m1.is_file() and m2.is_file()):
            self.skipTest("scripts/export_m1.py and scripts/export_m2.py are development-only: the copy cannot be compared here")
        sys.path.insert(0, str(ROOT / "scripts"))
        import export_helpers as h
        mods = {}
        for name, path in (("m1", m1), ("m2", m2)):
            spec = importlib.util.spec_from_file_location(f"export_{name}_reference", path)
            mods[name] = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mods[name])
        import ast
        import textwrap

        def code(f):
            """AST della funzione senza docstring (quelle della copia sono riformulate in modo neutro)."""
            node = ast.parse(textwrap.dedent(inspect.getsource(f))).body[0]
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant) \
                    and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
            return ast.dump(node)
        for mod, names in ((mods["m2"], ("sha256", "run", "tests_line", "scan", "completeness_problems", "fill_run_id", "hardware_line")),
                           (mods["m1"], ("write_sums", "redact", "local_imports"))):
            for n in names:
                self.assertEqual(code(getattr(h, n)), code(getattr(mod, n)), n)
        # unica differenza ammessa (16/10/2026): env_versions legge i file dell'ambiente con un glob invece di elencarli per
        # nome; il resto della funzione è identico all'originale, che con quella sola sostituzione deve dare la copia
        orig = textwrap.dedent(inspect.getsource(mods["m1"].env_versions))
        loop = re.search(r"    for f in \(.*?\):\n        p = ROOT / f\n        if p.exists\(\):\n            (vs \|= .*?)\n", orig)
        self.assertIsNotNone(loop, "env_versions of export_m1.py changed: update the allowed difference")
        expected = orig.replace(loop.group(0), '    for p in sorted(ROOT.glob("requirements*.txt")):\n        ' + loop.group(1) + "\n")
        expected_node = ast.parse(expected).body[0]
        expected_node.body = expected_node.body[1:] if isinstance(expected_node.body[0], ast.Expr) else expected_node.body
        self.assertEqual(code(h.env_versions), ast.dump(expected_node), "env_versions")
        self.assertEqual(h.env_versions() >= mods["m1"].env_versions(), True)        # stesse versioni (e nessuna in meno)
        self.assertEqual(h.FORBIDDEN_EXT, mods["m1"].FORBIDDEN_EXT)

    def test_public_records(self):
        # identificativi costruiti a run time (nessun letterale nel sorgente pubblico, che la scansione dell'export confronta)
        sk, vk, fn, ck = ":".join(("test", "7")), ":".join(("test", "7", "HR_2")), "7_HR_2.avi_" + "126_fake.jpg", "celeba_" + "00042"
        ids = {"subject_keys": {sk}, "video_keys": {vk}, "names": {fn, ck},
               "hashes": {"a" * 63 + "1"}}
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "ok.md").write_text("example 9_HR_4.avi_620_fake.jpg, celeba_09999, test:99, hash " + "b" * 64 + "\n")
            self.assertEqual(EX.public_record_problems(Path(d), ids), [])
        for text in ("subject " + sk, "video " + vk, "frame " + fn, "row " + ck, "sha " + "a" * 63 + "1"):
            with tempfile.TemporaryDirectory() as d:
                (Path(d) / "f.json").write_text(text)
                self.assertTrue(EX.public_record_problems(Path(d), ids), text)

    def test_shared_files_identical_to_nuaa_package(self):
        """I moduli condivisi sono identici a quelli del pacchetto NUAA: export/ nel repository di sviluppo, oppure la
        cartella m2-nuaa-manifests accanto a questa (come in research/ del repository del gruppo)."""
        nuaa = next((p for p in (ROOT / "export" / "research" / "m2-nuaa-manifests", ROOT.parent.parent / "m2-nuaa-manifests")
                     if (p / "SHA256SUMS").is_file()), None)
        if nuaa is None:
            self.skipTest("NUAA public package not found (neither export/research/m2-nuaa-manifests nor ../m2-nuaa-manifests)")
        listed = dict(reversed(line.split("  ", 1)) for line in (nuaa / "SHA256SUMS").read_text(encoding="utf-8").splitlines() if "  " in line)
        for f, published in EX.REFERENCE_FILES.items():
            sub = published.split("research/m2-nuaa-manifests/", 1)[1]
            self.assertEqual(EX.sha256(ROOT / f), listed.get(sub), f)


class TestPathGuard(unittest.TestCase):
    """Nessuna cartella protetta, né un suo antenato, viene mai cancellata (difetto trovato dal coordinatore il 15/10:
    --m2 accettava un antenato, per esempio results/ o la radice del repository). Solo cartelle temporanee: le
    cancellazioni vere avvengono in un albero finto, e shutil.rmtree è intercettato nei test di main."""
    RUN = "20261016-M2-rgb-seed42-abcdef0"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name).resolve()
        self.home, self.root = t / "home", t / "work" / "repo"            # repository fuori dalla home: controlli distinti
        self.sentinels = []
        for d in ("data", "models", "results/c1", "results/m1", "results/m2", "results/eval", "results/m2_rgb/casia_fasd", "docs",
                  "scripts", ".git", "export/research/m2-nuaa-manifests", "export/research/m2-rgb-manifests",
                  "export/drive/02_Experiments/M2/20261006-M2-seed42-36c677a", "export/drive/02_Experiments/M2/20261008-M2-rgb-seed42-532731e"):
            (self.root / d).mkdir(parents=True)
            if "m2_rgb" not in d and "m2-rgb-manifests" not in d:
                (self.root / d / "SENTINEL").write_text("keep")
                self.sentinels.append(self.root / d / "SENTINEL")
        self.home.mkdir()
        self.outside = t / "scratch"
        self.outside.mkdir()
        (t / "link_to_c1").symlink_to(self.root / "results" / "c1")
        self.link = t / "link_to_c1"

    def tearDown(self):
        self.tmp.cleanup()

    def problem(self, p, run=None):
        return EX.deletion_problem(p, run or self.RUN, root=self.root, home=self.home)

    def test_refused_and_allowed(self):
        r = self.root
        refused = {"repository root": r, "results (ancestor)": r / "results", "export (ancestor)": r / "export",
                   "filesystem root": Path("/"), "home": self.home, "ancestor of home": self.home.parent,
                   "baseline equal": r / "results" / "c1", "baseline descendant": r / "results" / "m1" / "x",
                   "NUAA results": r / "results" / "m2", "data": r / "data", "models": r / "models", "docs": r / "docs",
                   "scripts": r / "scripts", "git": r / ".git", "other tranche": r / "export" / "research" / "m2-nuaa-manifests",
                   "accepted NUAA run": r / "export" / "drive" / "02_Experiments" / "M2" / "20261006-M2-seed42-36c677a",
                   "delivered RGB run": r / "export" / "drive" / "02_Experiments" / "M2" / "20261008-M2-rgb-seed42-532731e",
                   "relative with ..": r / "results" / "m2_rgb" / ".." / "c1", "symlink to a baseline": self.link,
                   "other folder of the repository": r / "notes"}
        for what, p in refused.items():
            self.assertTrue(self.problem(p), what)
        # il motivo dichiarato distingue i controlli (antenato del repository, antenato di una cartella protetta, home)
        for p, why in ((r.parent, "repository root or one of its ancestors"), (r / "results", "contains the protected folder"),
                       (r / "export", "contains the protected folder"), (self.home, "home folder"), (Path("/"), "filesystem root")):
            self.assertIn(why, self.problem(p), str(p))
        self.assertTrue(self.problem(r / "export" / "drive" / "02_Experiments" / "M2" / "20261008-M2-rgb-seed42-532731e",
                                     run="20261008-M2-rgb-seed42-532731e"))        # un run consegnato non si riscrive
        allowed = {"results/m2_rgb": r / "results" / "m2_rgb", "inside m2_rgb": r / "results" / "m2_rgb" / "casia_fasd",
                   "public folder of this tranche": r / "export" / "research" / "m2-rgb-manifests",
                   "new run folder": r / "export" / "drive" / "02_Experiments" / "M2" / self.RUN,
                   "scratch outside the repository": self.outside / "m2"}
        for what, p in allowed.items():
            self.assertEqual(self.problem(p), "", what)
        for p in list(refused.values()) + list(allowed.values()):         # cancellazione vera solo se ammessa, nell'albero finto
            if not self.problem(p) and p.exists():
                shutil.rmtree(p)
        self.assertEqual([s for s in self.sentinels if not s.is_file()], [])
        self.assertFalse((r / "results" / "m2_rgb").exists())

    def test_safe_rmtree_deletes_only_allowed_folders(self):
        # path_guard.safe_rmtree sull'albero finto: rifiuta cartelle protette e antenati senza toccarli, cancella solo
        # le cartelle ammesse (qui: results/m2_rgb e una cartella di scratch, entrambe temporanee)
        sys.path.insert(0, str(ROOT / "scripts"))
        import path_guard as G
        r = self.root
        prot, allowed = EX.PROTECTED, EX.allowed_folders(self.RUN, r)
        for p in (r, r / "results", r / "results" / "c1", r / "export", self.home,
                  r / "export" / "drive" / "02_Experiments" / "M2" / "20261008-M2-rgb-seed42-532731e"):
            with self.assertRaises(SystemExit, msg=str(p)):
                G.safe_rmtree(p, r, prot, allowed, self.home)
            self.assertTrue(p.exists(), str(p))
        self.assertEqual([x for x in self.sentinels if not x.is_file()], [])
        (r / "results" / "m2_rgb" / "casia_fasd" / "old.json").write_text("{}")
        (self.outside / "m2" / "x").mkdir(parents=True)
        for p in (r / "results" / "m2_rgb", self.outside / "m2"):
            G.safe_rmtree(p, r, prot, allowed, self.home)
            self.assertFalse(p.exists(), str(p))
        self.assertEqual([x for x in self.sentinels if not x.is_file()], [])

    @staticmethod
    def case_insensitive(p: Path) -> bool:
        return str(p).swapcase() != str(p) and os.path.exists(str(p).swapcase())

    def test_other_spellings_of_the_same_folder(self):
        # filesystem che non distingue le maiuscole (macOS): una grafia diversa è la stessa cartella e va rifiutata
        r = self.root
        if not self.case_insensitive(r):
            self.skipTest("case-sensitive filesystem: other spellings are different folders")
        sw = lambda p: Path(str(p).swapcase())                              # noqa: E731
        low = lambda p: Path(str(p).lower())                                # noqa: E731
        for p in (sw(r), low(r), sw(r / "results"), low(r / "results"), sw(r / "results" / "c1"), sw(self.home),
                  sw(r / "export" / "drive" / "02_Experiments" / "M2" / "20261006-M2-seed42-36c677a"),
                  sw(r / "results" / "c1" / "not-yet-there"), sw(r / "notes" / "new")):
            self.assertTrue(self.problem(p), str(p))
        for p in (sw(r / "results" / "m2_rgb"), sw(r / "results" / "m2_rgb" / "new"),
                  sw(r / "export" / "drive" / "02_Experiments" / "M2" / self.RUN), sw(self.outside / "m2")):
            self.assertEqual(self.problem(p), "", str(p))                   # cartelle ammesse anche con un'altra grafia
        self.assertIn("repository root", self.problem(sw(r)))
        self.assertIn("contains the protected folder", self.problem(sw(r / "results")))

    def test_main_never_deletes_a_protected_folder(self):
        calls = []
        saved = (EX.shutil.rmtree, EX.run, sys.argv)
        EX.shutil.rmtree = lambda *a, **k: calls.append(a)

        def no_run(*a, **k):                                                 # nessun passo dell'export deve partire
            raise AssertionError("an export step was started: " + " ".join(str(x) for x in a[1:3]))
        EX.run = no_run
        real_link = self.outside / "link_to_real_c1"                          # collegamento a una cartella protetta vera
        real_link.symlink_to(ROOT / "results" / "c1")
        try:
            bad = [str(ROOT / "results"), str(ROOT), "/", str(Path.home()), str(ROOT / "results" / "m2_rgb" / ".." / "c1"),
                   str(real_link), str(ROOT / "results" / "c1"), str(ROOT / "docs")]
            if self.case_insensitive(ROOT):                                   # altre grafie della radice e di results/
                bad += [str(ROOT).lower(), str(ROOT).swapcase(), str(ROOT / "results").lower(), str(ROOT / "results").upper()]
            for m2 in bad:
                sys.argv = ["export_m2_rgb.py", "--trial", "--run-id", self.RUN, "--m2", m2, "--out", str(self.outside / "out")]
                err = io.StringIO()
                with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                    rc = EX.main()
                self.assertEqual(rc, 1, m2)
                self.assertTrue("cannot be" in err.getvalue() or "read-only area" in err.getvalue(), (m2, err.getvalue()))
            for out in (str(ROOT / "results"), str(ROOT / "scripts")):        # --out che porterebbe a scrivere in aree protette
                sys.argv = ["export_m2_rgb.py", "--trial", "--run-id", self.RUN, "--m2", str(self.outside / "m2"), "--out", out]
                err = io.StringIO()
                with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(EX.main(), 1, out)
                self.assertIn("cannot be written", err.getvalue())
            sys.argv = ["export_m2_rgb.py", "--trial", "--run-id", self.RUN, "--dry-run", "--m2", str(self.outside / "m2"), "--out", str(self.outside / "out")]
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                EX.main()
            self.assertNotIn("cannot be", err.getvalue())                    # cartelle di scratch: ammesse
            with self.assertRaises(SystemExit):
                EX.safe_rmtree(ROOT / "results", self.RUN)                   # controllo anche subito prima di cancellare
            with self.assertRaises(SystemExit):
                EX.safe_rmtree(ROOT, self.RUN)
            self.assertEqual(calls, [])
            EX.safe_rmtree(self.outside / "m2", self.RUN)
            self.assertEqual(len(calls), 1)
        finally:
            EX.shutil.rmtree, EX.run, sys.argv = saved


class TestHelpers(unittest.TestCase):
    """Comportamento delle funzioni di export_helpers.py su cartelle sintetiche (gira anche dal pacchetto pubblico). I
    testi vietati sono costruiti a run time, così il sorgente pubblico resta pulito."""

    def setUp(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import export_helpers
        self.H = export_helpers
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def scan_one(self, name: str, content: bytes) -> list:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / name).write_bytes(content)
            return [r for _, r in self.H.scan(Path(d), set())]

    def test_scan(self):
        self.assertEqual(self.scan_one("a.md", b"# Title\n\nplain text, host 127.0.0.1\n"), [])
        cases = {"absolute path": ("p.md", ("see /" + "Users/someone/file").encode()),
                 "forbidden file type": ("x" + ".npz", b"plain text"),         # contenuto testuale: lo ferma solo l'estensione
                 "IP address": ("ip.txt", ("host " + ".".join(["10", "1", "2", "3"])).encode()),
                 "unfinished-work marker": ("m.md", ("open: T" + "BD").encode())}
        import package_rules
        if package_rules.TOOL_WORDS:                            # nome di uno strumento, dal file locale (assente nel pacchetto)
            cases["tool reference"] = ("t.md", ("made with " + package_rules.TOOL_WORDS[0].title()).encode())
        for what, (name, content) in cases.items():
            found = self.scan_one(name, content)
            self.assertTrue(found, what)
            if what in ("forbidden file type", "IP address", "unfinished-work marker"):
                self.assertTrue(any(f.startswith(what) for f in found), (what, found))
        self.assertEqual(self.scan_one("m.txt", ("open: T" + "BD").encode()), [])            # i segni solo nei .md
        (self.d / "a.txt").write_text("x")
        os.symlink(self.d / "a.txt", self.d / "b.txt")
        self.assertIn("symbolic link", " ".join(r for _, r in self.H.scan(self.d, set())))

    def test_completeness_and_write_sums(self):
        f = self.d / "pkg"
        (f / "sub").mkdir(parents=True)
        (f / "a.txt").write_text("a"); (f / "sub" / "b.txt").write_text("b")
        sha, n, problems = self.H.write_sums(f)
        self.assertEqual((n, problems), (2, []))
        lines = (f / "SHA256SUMS").read_text().splitlines()
        self.assertEqual(sorted(l.split("  ", 1)[1] for l in lines), ["a.txt", "sub/b.txt"])          # non elenca se stesso
        self.assertEqual(sha, self.H.sha256(f / "SHA256SUMS"))
        self.assertEqual(self.H.write_sums(f)[1], 2)                                                  # riscritto: stesso conteggio
        self.assertEqual(self.H.completeness_problems(f), [])
        (f / "c.txt").write_text("c")
        self.assertTrue(self.H.completeness_problems(f)); (f / "c.txt").unlink()                     # file in più
        (f / "a.txt").write_text("A")
        self.assertIn("SHA-256 differs", " ".join(self.H.completeness_problems(f))); (f / "a.txt").write_text("a")
        (f / "sub" / "b.txt").unlink()
        self.assertTrue(self.H.completeness_problems(f)); (f / "sub" / "b.txt").write_text("b")       # file mancante
        os.symlink(f / "a.txt", f / "l.txt")
        self.assertIn("symbolic links", " ".join(self.H.completeness_problems(f))); (f / "l.txt").unlink()
        self.assertEqual(self.H.completeness_problems(f), [])

    def test_redact(self):
        root, home = str(self.H.ROOT), str(Path.home())
        self.assertEqual(self.H.redact(root + "/scripts/x.py"), "<repo>/scripts/x.py")
        self.assertEqual(self.H.redact("cache in " + home + "/.cache"), "cache in ~/.cache")
        tmp = "/priv" + "ate/tmp/abc/run.log"
        self.assertEqual(self.H.redact("log " + tmp + " done"), "log <tmp> done")
        self.assertEqual(self.H.redact("no path here"), "no path here")

    def test_fill_run_id(self):
        self.assertEqual(self.H.fill_run_id("folder {run_id}/", "20261013-M2-rgb-seed42-abcdef0"), "folder 20261013-M2-rgb-seed42-abcdef0/")
        with self.assertRaises(ValueError):
            self.H.fill_run_id("no placeholder", "x")
        with self.assertRaises(ValueError):
            self.H.fill_run_id("{run_id} and <run-id>", "x")


def main() -> int:
    ap = argparse.ArgumentParser(description="Test offline di export_m2_rgb.py")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (TestLists, TestPathGuard, TestHelpers))
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    if args.verbose or not result.wasSuccessful():
        print(log.getvalue())
    print(f"tests run: {result.testsRun}, failures: {len(result.failures)}, errors: {len(result.errors)}, skipped: {len(result.skipped)}")
    print("RESULT: " + ("PASS" if result.wasSuccessful() else "FAIL"))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
