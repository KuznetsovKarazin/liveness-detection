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
# I soli motivi per cui un test di questa suite si salta: tutti ingressi mancanti (file del repository di sviluppo o del
# pacchetto accanto al codice), nessuno di piattaforma (revisione del 17/10/2026). L'export ufficiale, che ha tutti gli
# ingressi, conta ogni salto come controllo fallito; test_every_skip_is_a_missing_input lo verifica sul sorgente.
MISSING_INPUT = {
    "registers": DEV_ONLY,
    "public documents": "the public documents are next to livenesslab/ only in the delivered folder, not in a bare copy of the code",
    "original exports": "scripts/export_m1.py and scripts/export_m2.py are development-only: the copy cannot be compared here",
    "NUAA package": "NUAA public package not found (neither export/research/m2-nuaa-manifests nor ../m2-nuaa-manifests)",
    "git": "git not available: the repository state cannot be computed",
}


def skip_missing(test: unittest.TestCase, what: str):
    """Unico modo di saltare un test di questa suite: un ingresso mancante dichiarato in MISSING_INPUT."""
    test.skipTest(f"missing input: {MISSING_INPUT[what]}")


def has_registers() -> bool:
    return all((EX.DELIV / n).is_file() for n in EX.RESTRICTED_DOCS)


def public_doc(n: str):
    """Documento pubblico: in delivery/m2/rgb (repository di sviluppo) o nella cartella che contiene livenesslab/."""
    return next((p for p in (EX.DELIV / n, ROOT.parent / n) if p.is_file()), None)


@contextlib.contextmanager
def alias_of(alias, target):
    """Simula un secondo percorso per la stessa cartella su ogni filesystem: os.stat (usato da os.path.samefile e da
    Path.exists) riporta per alias e per ogni percorso sotto di esso l'identità del percorso corrispondente sotto target,
    come fa un filesystem che non distingue le maiuscole per una grafia diversa. Un confronto fra stringhe dei percorsi
    risolti non lo riconosce; il confronto per identità sì."""
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


class TestLists(unittest.TestCase):
    def need_registers(self):
        if not has_registers():
            skip_missing(self, "registers")

    def test_code_and_documents_exist(self):
        self.assertEqual([c for c in EX.CODE if not (ROOT / c).is_file()], [])
        for t in ("README.md", "run-report.md.tmpl"):                 # modelli dei testi pubblici: anche nel pacchetto
            self.assertTrue((EX.DELIV / t).is_file(), t)
        missing = [n for n in EX.PUBLIC_DOCS if public_doc(n) is None]
        if missing and not (ROOT.parent / "README.md").is_file():
            skip_missing(self, "public documents")
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

    def fake_export(self, d: Path, celeba_public_passed=True, json_dirty=False, outside=("", ""), start_dirty=True) -> tuple:
        """main con i passi dell'export sostituiti da un run finto che scrive artefatti coerenti fra loro (nessun manifest)
        in una cartella temporanea; lo stato del repository fuori da --m2 è dato dal test (prima e dopo la rigenerazione),
        così l'esito non dipende dallo stato vero del repository. Restituisce (codice, uscita, run report o None)."""
        import json
        import subprocess
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=EX.ROOT, capture_output=True, text=True).stdout.strip() or "n/d"
        m2 = Path(d) / "m2"
        # documenti di consegna fittizi: il test gira anche dal pacchetto pubblico, che non ha i modelli del repository
        deliv = Path(d) / "deliv"
        deliv.mkdir()
        (deliv / "README.md").write_text("{commit} {run_id}\n\n{delivery_status}\n")
        (deliv / "run-report.md.tmpl").write_text("run {run_id}\ntree: {tree_line}\n")
        for n in EX.RESTRICTED_DOCS:
            (deliv / n).write_text("register draft {run_id}\n")
        for n in EX.PUBLIC_DOCS:
            src = ROOT / "delivery" / "m2" / "rgb" / n
            (deliv / n).write_bytes(src.read_bytes() if src.is_file() else b"document\n")
        man, ovm = {}, {}                      # nessun manifest: artefatti coerenti fra loro
        self.export_calls = []                 # comandi che l'export avrebbe eseguito (letti da check_m2_writers.py)
        state = {"commit": commit, "dirty": json_dirty, "dirty_excludes": None}

        def fake_run(*cmd, env=None, cwd=None):
            self.export_calls.append([str(c) for c in cmd])
            for ds in EX.DATASETS:
                f = m2 / ds
                (f / "manifests").mkdir(parents=True, exist_ok=True)
                docs = {f"{ds}_manifests_summary.json": {"passed": True, "manifests": man},
                        f"{ds}_manifests_aggregate.json": {"passed": True, "manifests": man},
                        "overlap_report.json": {"passed": True, "rules_sha256": "r", "manifests": ovm, "rules": [], "n_must_failed": 0},
                        "overlap_report_public.json": {"passed": celeba_public_passed or ds != "celeba_spoof", "manifests": ovm, "rules": [],
                                                       "n_must_failed": 0},
                        "overlap_self_test.json": {"passed": True, "rules_sha256": "r"}}
                for n, v in docs.items():
                    (f / n).write_text(json.dumps({**v, **state}))
                for n in (f"{ds}_m2_report.md", f"{ds}_m2_report_public.md", "overlap_report.md", "overlap_report_public.md", "overlap_self_test.md"):
                    (f / n).write_text("x\n")
            return 0, "tests run: 1, failures: 0, errors: 0, skipped: 0\nbuild checks passed (1 checks)\n"
        trees = iter([(None, commit, outside[0]), (None, commit, outside[1])])
        saved = (EX.run, sys.argv, EX.DELIV, EX.tree_outside, EX.repo_dirty)
        # albero intero (soglia iniziale) dato dal test: con --trial può essere modificato anche se fuori da --m2 è pulito
        EX.run, EX.DELIV, EX.tree_outside, EX.repo_dirty = fake_run, deliv, lambda p: next(trees), lambda root=None: start_dirty
        sys.argv = ["export_m2_rgb.py", "--trial", "--run-id", "20261013-M2-rgb-seed42-test000", "--out", str(Path(d) / "out"), "--m2", str(m2)]
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                rc = EX.main()
        finally:
            EX.run, sys.argv, EX.DELIV, EX.tree_outside, EX.repo_dirty = saved
        rr = next((Path(d) / "out" / "research").glob("*/run-report.md"), None)
        return rc, buf.getvalue(), rr.read_text() if rr else None

    def test_main_refuses_failed_artefacts(self):
        """main controlla gli artefatti di ogni dataset con artefact_problems: con un run finto che lascia un rapporto
        pubblico delle sovrapposizioni fallito, l'export rifiuta (codice 1) e lo dice."""
        with tempfile.TemporaryDirectory() as d:
            rc, out, _ = self.fake_export(Path(d), celeba_public_passed=False)
        self.assertEqual(rc, 1)
        self.assertIn("celeba_spoof: overlap_report_public.json is not a passed artefact", out)
        self.assertNotIn("casia_fasd: overlap_report_public.json is not a passed artefact", out)
        self.assertNotIn("repository state", out)                         # stato dei JSON coerente con il run report

    def test_main_compares_json_state_with_the_run_report(self):
        """Revisione del 17/10/2026: i JSON dicevano dirty true e il run report working tree clean. main confronta commit,
        dirty e cartella esclusa di ogni JSON con lo stato registrato e controlla che durante la rigenerazione nulla cambi
        fuori da --m2; il run report dichiara la semantica del campo."""
        with tempfile.TemporaryDirectory() as d:
            rc, out, rr = self.fake_export(Path(d))
        self.assertNotIn("repository state", out)
        self.assertNotIn("working tree changed", out)
        self.assertIn("the results folder is outside the repository", rr)
        self.assertIn("plain repository state, `false`", rr)
        with tempfile.TemporaryDirectory() as d:
            rc, out, rr = self.fake_export(Path(d), json_dirty=True)          # JSON dirty, albero pulito fuori da --m2
        self.assertEqual(rc, 1)
        self.assertEqual(out.count("differs from the run report"), 2 * len(EX.STATE_JSON), out)
        with tempfile.TemporaryDirectory() as d:
            rc, out, _ = self.fake_export(Path(d), outside=("", " M scripts/x.py"))   # modifica comparsa durante il run
        self.assertEqual(rc, 1)
        self.assertIn("the working tree changed outside the results folder during the regeneration", out)
        self.assertIn("M scripts/x.py", out)
        # albero modificato fuori da --m2 e invariato durante il run: JSON dirty true, coerente con il run report
        with tempfile.TemporaryDirectory() as d:
            rc, out, rr = self.fake_export(Path(d), json_dirty=True, outside=(" M scripts/x.py", " M scripts/x.py"))
        self.assertNotIn("differs from the run report", out)          # (codice 1 solo per i manifest assenti del run finto)
        self.assertNotIn("working tree changed", out)
        self.assertIn("plain repository state, `true`", rr)
        # albero intero modificato ma pulito fuori da --m2: il valore atteso è quello fuori da --m2, non quello intero
        with tempfile.TemporaryDirectory() as d:
            rc, out, rr = self.fake_export(Path(d), json_dirty=False, outside=("", ""), start_dirty=True)
        self.assertNotIn("differs from the run report", out)
        self.assertIn("plain repository state, `false`", rr)
        # git status fallito all'inizio (commit leggibile): export rifiutato prima di qualsiasi passo
        with tempfile.TemporaryDirectory() as d:
            rc, out, rr = self.fake_export(Path(d), outside=(None, None))
            self.assertFalse((Path(d) / "m2").exists())
        self.assertEqual(rc, 1)
        self.assertIn("git status failed", out)

    def test_repository_state_helpers(self):
        import json
        commit = "c" * 40
        with tempfile.TemporaryDirectory() as d:
            m2 = Path(d)
            good = {"commit": commit, "dirty": False, "dirty_excludes": "results/m2_rgb"}
            for ds in EX.DATASETS:
                (m2 / ds).mkdir()
                for n in EX.STATE_JSON:
                    (m2 / ds / n.format(ds=ds)).write_text(json.dumps({"passed": True, **good}))
            self.assertEqual(EX.json_state_problems(m2, commit, False, "results/m2_rgb"), [])
            self.assertEqual(len(EX.json_state_problems(m2, commit, True, "results/m2_rgb")), 10)      # dirty diverso
            self.assertEqual(len(EX.json_state_problems(m2, "d" * 40, False, "results/m2_rgb")), 10)   # altro commit
            self.assertEqual(len(EX.json_state_problems(m2, commit, False, None)), 10)                 # altra esclusione
            f = m2 / "celeba_spoof" / "overlap_self_test.json"
            f.write_text(json.dumps({"passed": True, "commit": commit, "dirty": False}))               # campo assente
            self.assertEqual(len(EX.json_state_problems(m2, commit, False, "results/m2_rgb")), 1)
        self.assertEqual(EX.tree_change_problem(("results/m2_rgb", commit, ""), ("results/m2_rgb", commit, "")), "")
        self.assertIn("results/m2_rgb", EX.tree_change_problem(("results/m2_rgb", commit, ""), ("results/m2_rgb", commit, "?? notes.txt")))
        self.assertTrue(EX.tree_change_problem(("results/m2_rgb", commit, ""), ("results/m2_rgb", "d" * 40, "")))
        line = EX.tree_line(False, "results/m2_rgb", False)
        self.assertIn("working tree clean at the start of the export: yes", line)
        self.assertIn("only `results/m2_rgb` was deleted and regenerated", line)
        self.assertIn("(`dirty_excludes`) and is `false`", line)
        self.assertIn("NO (explicit trial export, `--trial`)", EX.tree_line(True, "results/m2_rgb", True))
        self.assertIn("is `true`", EX.tree_line(True, "results/m2_rgb", True))

    def test_start_gate_uses_the_same_status_as_the_json(self):
        """Revisione del 17/10/2026: con status.showUntrackedFiles=no un file non versionato sfuggiva alla soglia iniziale
        (git status --porcelain vuoto) ma non ai JSON; ora soglia, controllo dopo la rigenerazione e JSON usano lo stesso
        comando (rgb_manifests.STATUS_ARGS). Su un repository git temporaneo."""
        import subprocess
        if shutil.which("git") is None:
            skip_missing(self, "git")
        self.assertIn("--untracked-files=all", EX.RM.STATUS_ARGS)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            g = lambda *a: subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test", "-c", "commit.gpgsign=false", *a],   # noqa: E731
                                          cwd=root, capture_output=True, text=True, check=True).stdout
            (root / "a.txt").write_text("a\n")
            g("init", "-q"); g("add", "-A"); g("commit", "-q", "-m", "init"); g("config", "status.showUntrackedFiles", "no")
            self.assertFalse(EX.repo_dirty(root))
            (root / "b.txt").write_text("b\n")
            self.assertEqual(g("status", "--porcelain").strip(), "")         # nascosto dalla configurazione
            self.assertTrue(EX.repo_dirty(root))
            self.assertEqual(EX.repo_dirty(root), EX.RM.repo_state_excluding(Path(d + "-elsewhere"), root)["dirty"])

    def test_failed_git_status_is_never_clean(self):
        saved = EX.RM.tree_state
        try:
            EX.RM.tree_state = lambda out, root=None: (None, "c" * 40, None)       # rev-parse risponde, status no
            self.assertTrue(EX.repo_dirty(Path("/")))
            self.assertTrue(EX.RM.repo_state_excluding(Path("/"))["dirty"])
            EX.RM.tree_state = lambda out, root=None: (None, None, None)           # nessun repository
            self.assertFalse(EX.repo_dirty(Path("/")))
        finally:
            EX.RM.tree_state = saved
        ok = ("results/m2_rgb", "c" * 40, "")
        self.assertIn("git status failed", EX.tree_change_problem(ok, ("results/m2_rgb", "c" * 40, None)))
        self.assertIn("git status failed", EX.tree_change_problem(("results/m2_rgb", "c" * 40, None), ok))
        self.assertTrue(EX.status_failed((None, "c", None)))
        self.assertFalse(EX.status_failed((None, None, None)))
        self.assertIn("--no-optional-locks", EX.RM.STATUS_ARGS)

    def test_skips_fail_the_export_with_their_reason(self):
        self.assertEqual(EX.skip_problems("check_export_m2_rgb.py", "tests run: 3, failures: 0, errors: 0, skipped: 0\n", 0), [])
        self.assertTrue(EX.skip_problems("check_export_m2_rgb.py", "no summary", None))
        txt = "skipped test_x: missing input: NUAA public package not found\ntests run: 3, failures: 0, errors: 0, skipped: 1\n"
        p = EX.skip_problems("check_export_m2_rgb.py", txt, 1)
        self.assertEqual(len(p), 1)
        self.assertIn("1 skipped tests (missing inputs: test_x: missing input: NUAA public package not found)", p[0])

    def test_every_skip_is_a_missing_input(self):
        """Nessun salto di piattaforma: ogni salto passa da skip_missing con un motivo di MISSING_INPUT (revisione del
        17/10/2026: su un filesystem che distingue le maiuscole un test si saltava e l'export non poteva mai passare)."""
        src = Path(__file__).read_text(encoding="utf-8")
        self.assertEqual(src.count(".skip" + "Test("), 1)                      # solo dentro skip_missing
        used = re.findall(r'skip_missing\(self, "([^"]+)"\)', src)
        self.assertEqual(sorted(set(used)), sorted(MISSING_INPUT))
        for why in MISSING_INPUT.values():
            self.assertFalse(re.search(r"case|filesystem|platform|linux|macos|windows", why, re.I), why)

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
            skip_missing(self, "original exports")
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
            skip_missing(self, "NUAA package")
        listed = dict(reversed(line.split("  ", 1)) for line in (nuaa / "SHA256SUMS").read_text(encoding="utf-8").splitlines() if "  " in line)
        for f, published in EX.REFERENCE_FILES.items():
            sub = published.split("research/m2-nuaa-manifests/", 1)[1]
            if f not in EX.mw.MAINTENANCE:
                self.assertEqual(EX.sha256(ROOT / f), listed.get(sub), f)
                continue
            # manutenzione del 22/10/2026: la copia consegnata resta com'è; il file corrente è quella copia oppure il file
            # corretto con lo SHA-256 dichiarato e l'AST identico alla copia consegnata fuori dalle funzioni dichiarate
            self.assertEqual(listed.get(sub), EX.mw.MAINTENANCE[f]["accepted"], f)
            accepted = (nuaa / sub).read_text(encoding="utf-8")
            self.assertIn(EX.reference_state(f, listed.get(sub), accepted=accepted), ("=", EX.mw.MAINT_LABEL), f)


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
        # due percorsi diversi per la stessa cartella vanno riconosciuti per identità. Gira su ogni piattaforma, senza
        # salti (revisione del 17/10/2026: su Linux il test si saltava e la mutazione del confronto per stringhe
        # sopravviveva): un alias simulato sempre (una seconda cartella vera che riporta l'identità della radice finta,
        # alias_of) e, su un filesystem che non distingue le maiuscole (macOS), anche le grafie diverse vere
        r, base = self.root, Path(self.tmp.name).resolve()
        with tempfile.TemporaryDirectory() as other:
            alias_base = Path(other).resolve() / "alias"
            alias_base.mkdir()
            spellings = {"alias": lambda p: alias_base / Path(p).relative_to(base)}
            if self.case_insensitive(r):
                spellings["swapcase"] = lambda p: Path(str(p).swapcase())
                spellings["lower"] = lambda p: Path(str(p).lower())
            for kind, sw in spellings.items():
                with alias_of(alias_base, base):
                    for p in (r, r / "results", r / "results" / "c1", self.home,
                              r / "export" / "drive" / "02_Experiments" / "M2" / "20261006-M2-seed42-36c677a",
                              r / "results" / "c1" / "not-yet-there", r / "notes" / "new"):
                        self.assertTrue(self.problem(sw(p)), (kind, str(p)))
                    for p in (r / "results" / "m2_rgb", r / "results" / "m2_rgb" / "new",
                              r / "export" / "drive" / "02_Experiments" / "M2" / self.RUN, self.outside / "m2"):
                        self.assertEqual(self.problem(sw(p)), "", (kind, str(p)))    # cartelle ammesse anche con un altro percorso
                    self.assertIn("repository root", self.problem(sw(r)), kind)
                    self.assertIn("contains the protected folder", self.problem(sw(r / "results")), kind)
            # controllo della simulazione: senza alias_of lo stesso percorso è un'altra cartella, fuori dal repository
            self.assertEqual(self.problem(alias_base / "work" / "repo"), "")

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
            alias = self.outside / "alias_of_root"                            # su ogni piattaforma: alias simulato della radice vera
            alias.mkdir()
            bad += [str(alias), str(alias / "results"), str(alias / "results" / "c1")]
            for m2 in bad:
                sys.argv = ["export_m2_rgb.py", "--trial", "--run-id", self.RUN, "--m2", m2, "--out", str(self.outside / "out")]
                err = io.StringIO()
                with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()), alias_of(alias, ROOT):
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
    for t, why in result.skipped:                       # motivo di ogni test saltato (letto dall'export)
        print(f"skipped {t.id().rsplit('.', 1)[-1]}: {why}")
    print(f"tests run: {result.testsRun}, failures: {len(result.failures)}, errors: {len(result.errors)}, skipped: {len(result.skipped)}")
    print("RESULT: " + ("PASS" if result.wasSuccessful() else "FAIL"))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
