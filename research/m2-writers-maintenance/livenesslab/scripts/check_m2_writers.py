"""
Regressioni della manutenzione dei writer M2 (22/10/2026): scripts/m2_writers.py, write_all, invalidate e
write_manifest di scripts/rgb_manifests.py e scripts/nuaa_manifests.py, validazione della cartella di uscita prima di
ogni modifica e radici protette di scripts/export_m2_rgb.py. Solo cartelle temporanee e dati sintetici: nessuna
immagine, nessun peso, nessuna rete; il repository vero non è mai una cartella di uscita.

Casi: temporaneo a nome fisso (quello delle copie accettate) come collegamento simbolico, fisso o senza bersaglio a un
file esterno, nei percorsi PASS e FAIL; uscita come collegamento simbolico (anche senza bersaglio) o fisso; cartella di
uscita come collegamento o come file; <out>/manifests come collegamento; manifest come collegamento; cartella di uscita
uguale o interna alla cartella dei manifest; regole o modello con il nome di un'uscita; cartella di uscita che contiene
i dati; cartelle protette di un repository finto; pacchetti fratelli in una copia sintetica della disposizione del
repository del gruppo (research/<pacchetto>/livenesslab), anche dalla riga di comando e con un albero git; radici
dell'export; ingressi e file esterni identici byte per byte dopo ogni rifiuto; dichiarazione della manutenzione
(SHA-256 esatti, AST identico alle copie accettate fuori dalle funzioni dichiarate).

  .venv/bin/python scripts/check_m2_writers.py        # uscita 1 se un test fallisce

Libreria standard più numpy (importato da rgb_manifests.py e nuaa_manifests.py).
"""
import contextlib
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
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import m2_writers as MW  # noqa: E402
import rgb_manifests as RM  # noqa: E402
import nuaa_manifests as NM  # noqa: E402
import export_m2_rgb as EX  # noqa: E402

RGB, NUAA = ROOT / "scripts" / "rgb_manifests.py", ROOT / "scripts" / "nuaa_manifests.py"

# I soli motivi per cui un test di questa suite si salta: ingressi mancanti dichiarati, nessuno di piattaforma
MISSING_INPUT = {
    "accepted copies": "accepted copies not found (neither ../<package> nor export/research/<package> nor the development commit)",
    "git": "git not available: the git work tree for the test cannot be built",
}


def skip_missing(test: unittest.TestCase, what: str):
    """Unico modo di saltare un test di questa suite: un ingresso mancante dichiarato in MISSING_INPUT."""
    test.skipTest(f"missing input: {MISSING_INPUT[what]}")


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_for_writers", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CR, CN = load("check_rgb_manifests"), load("check_nuaa_manifests")
EXT = b"external file, must never change\n"


def tree_bytes(d: Path) -> dict:
    """Contenuto di un albero (percorso relativo -> byte; collegamenti come bersaglio), senza seguire i collegamenti."""
    out = {}
    for dp, dns, fns in os.walk(d):
        for n in fns + [x for x in dns if (Path(dp) / x).is_symlink()]:
            q = Path(dp) / n
            out[str(q.relative_to(d))] = ("link", os.readlink(q)) if q.is_symlink() else q.read_bytes()
    return out


def link(kind: str, at: Path, target: Path):
    if kind == "symlink":
        at.symlink_to(target)
    elif kind == "dangling":
        at.symlink_to(target.parent / "does-not-exist")
    else:
        os.link(target, at)


def env_for(t: Path) -> dict:
    return {**os.environ, "LIVENESSLAB_DATA_DIR": str(t / "data"), "LIVENESSLAB_WEIGHTS_DIR": str(t / "weights"),
            "LIVENESSLAB_RESULTS_DIR": str(t / "results")}


def run(script: Path, *args, t: Path = None, env=None):
    return subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True, text=True, cwd=str(t or tempfile.gettempdir()),
                          env=env)


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.t = Path(self._tmp.name).resolve()
        self.ext = self.t / "external.txt"
        self.ext.write_bytes(EXT)
        self.ext_ino = os.stat(self.ext).st_ino

    def tearDown(self):
        self._tmp.cleanup()

    def assert_external_intact(self, why=""):
        self.assertEqual(self.ext.read_bytes(), EXT, why)
        self.assertEqual(os.stat(self.ext).st_ino, self.ext_ino, why)

    def no_temporaries(self, out: Path):
        return [p.name for p in Path(out).iterdir() if p.name.endswith(".tmp")] if Path(out).is_dir() else []


# ----------------------------------------------------------------------------- write_all, invalidate, write_manifest

class TestWriters(Base):
    MODS = (("rgb", RM), ("nuaa", NM))

    def test_fixed_temporary_link_never_followed(self):
        # difetto riprodotto dal coordinatore: <nome>.tmp preparato come collegamento faceva sovrascrivere il file esterno
        for tag, mod in self.MODS:
            for kind in ("symlink", "hardlink", "dangling"):
                out = self.t / f"{tag}-{kind}"
                out.mkdir()
                link(kind, out / "a.json.tmp", self.ext)
                mod.write_all(out, {"a.json": "new a\n", "b.md": "new b\n"})
                self.assert_external_intact((tag, kind))
                self.assertFalse((out / "a.json").is_symlink(), (tag, kind))
                self.assertEqual((out / "a.json").read_text(), "new a\n", (tag, kind))
                self.assertNotEqual(os.stat(out / "a.json").st_ino, self.ext_ino, (tag, kind))
                self.assertEqual(sorted(mod.invalidate(out, ["a.json", "b.md"])), ["a.json", "a.json.tmp", "b.md"], (tag, kind))
                self.assertEqual(list(out.iterdir()), [], (tag, kind))
                self.assert_external_intact((tag, kind))

    def test_fixed_temporary_link_on_the_failure_path(self):
        for tag, mod in self.MODS:
            for kind in ("symlink", "hardlink"):
                for exc in (OSError, KeyboardInterrupt):
                    out = self.t / f"{tag}-{kind}-{exc.__name__}"
                    out.mkdir()
                    link(kind, out / "a.json.tmp", self.ext)
                    calls = {"n": 0}
                    real = os.replace

                    def boom(a, b, exc=exc):
                        calls["n"] += 1
                        if calls["n"] == 2:
                            raise exc("simulated failure on the second rename")
                        return real(a, b)
                    with mock.patch.object(MW.os, "replace", boom), self.assertRaises(exc):
                        mod.write_all(out, {"a.json": "1", "b.json": "2"})
                    self.assert_external_intact((tag, kind, exc))
                    self.assertFalse((out / "a.json").exists() or (out / "b.json").exists(), (tag, kind, exc))
                    self.assertEqual([p.name for p in out.iterdir()], ["a.json.tmp"], (tag, kind, exc))   # solo il collegamento preparato

    def test_output_link_is_refused_and_removed_as_link(self):
        for tag, mod in self.MODS:
            for kind in ("symlink", "dangling"):
                out = self.t / f"{tag}-out-{kind}"
                out.mkdir()
                link(kind, out / "a.json", self.ext)
                err = io.StringIO()
                with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
                    mod.write_all(out, {"a.json": "new\n", "b.json": "new\n"})
                self.assertIn("a link or not a regular file", err.getvalue())
                self.assert_external_intact((tag, kind))
                self.assertEqual(list(out.iterdir()), [], (tag, kind))          # collegamento tolto, nessun temporaneo
            out = self.t / f"{tag}-out-dir"
            (out / "a.json").mkdir(parents=True)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                mod.write_all(out, {"a.json": "x"})
            self.assertTrue((out / "a.json").is_dir())

    def test_hardlink_output_is_replaced_by_name(self):
        for tag, mod in self.MODS:
            out = self.t / f"{tag}-hard"
            out.mkdir()
            os.link(self.ext, out / "a.json")
            mod.write_all(out, {"a.json": "new\n"})
            self.assert_external_intact(tag)
            self.assertEqual((out / "a.json").read_text(), "new\n")
            self.assertEqual(os.stat(self.ext).st_nlink, 1)

    def test_output_folder_link_or_file_refused(self):
        ext_dir = self.t / "ext_dir"
        ext_dir.mkdir()
        for tag, mod in self.MODS:
            out = self.t / f"{tag}-linkdir"
            out.symlink_to(ext_dir)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                mod.write_all(out, {"a.json": "x"})
            self.assertEqual(list(ext_dir.iterdir()), [], tag)

    def test_invalidate_never_follows_links(self):
        target_dir = self.t / "target_dir"
        target_dir.mkdir()
        (target_dir / "keep").write_text("keep")
        for tag, mod in self.MODS:
            out = self.t / f"{tag}-inv"
            out.mkdir()
            (out / "a.json").symlink_to(self.ext)
            (out / "b.json").symlink_to(target_dir)
            (out / "c.json").symlink_to(self.t / "missing")
            (out / ".a.json.x1y2.tmp").write_text("stale")
            (out / "d.json.tmp").symlink_to(self.ext)
            (out / "other.txt").write_text("other")
            self.assertEqual(sorted(mod.invalidate(out, ["a.json", "b.json", "c.json", "d.json"])),
                             [".a.json.x1y2.tmp", "a.json", "b.json", "c.json", "d.json.tmp"], tag)
            self.assertEqual([p.name for p in out.iterdir()], ["other.txt"], tag)
            self.assert_external_intact(tag)
            self.assertEqual((target_dir / "keep").read_text(), "keep")
            (out / "e.json").mkdir()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                mod.invalidate(out, ["e.json"])
            self.assertTrue((out / "e.json").is_dir())

    def test_write_manifest_same_bytes_and_atomic(self):
        import csv
        w = CR.casia_world()["mirror_train"]
        rows_n = CN.clean_world()["train"]
        for tag, write, cols, rows in (("rgb", lambda p, r: RM.write_manifest("casia_fasd", p, r), RM.SPEC["casia_fasd"]["columns"], w),
                                       ("nuaa", NM.write_manifest, NM.COLUMNS, rows_n)):
            buf = io.StringIO(newline="")
            dw = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
            dw.writeheader()
            dw.writerows(sorted(rows, key=lambda r: r["key"]))
            for kind in ("none", "symlink", "hardlink"):
                d = self.t / f"{tag}-man-{kind}"
                d.mkdir()
                p = d / "m.csv"
                if kind != "none":
                    link(kind, p, self.ext)
                if kind == "symlink":                                   # manifest come collegamento simbolico: rifiutato e tolto
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        write(p, rows)
                    self.assertFalse(os.path.lexists(p))
                    self.assert_external_intact((tag, kind))
                    continue
                h = write(p, rows)
                self.assertEqual(p.read_bytes(), buf.getvalue().encode("utf-8"), (tag, kind))
                self.assertEqual(h, hashlib.sha256(p.read_bytes()).hexdigest())
                self.assertFalse(p.is_symlink())
                self.assert_external_intact((tag, kind))
                self.assertEqual(self.no_temporaries(d), [])

    def test_temporary_files_are_unique_and_exclusive(self):
        seen = []
        real = MW.tempfile.mkstemp

        def spy(*a, **k):
            fd, p = real(*a, **k)
            seen.append(p)
            return fd, p
        with mock.patch.object(MW.tempfile, "mkstemp", spy):
            for tag, mod in self.MODS:
                out = self.t / f"{tag}-uniq"
                mod.write_all(out, {"a.json": "1\n"})
                mod.write_all(out, {"a.json": "2\n"})
        self.assertEqual(len(seen), 4)
        self.assertEqual(len(set(seen)), 4)
        for p in seen:
            self.assertRegex(Path(p).name, r"^\.a\.json\..+\.tmp$")
        self.assertEqual(oct(os.stat(self.t / "rgb-uniq" / "a.json").st_mode & 0o777), oct(0o666 & ~MW._UMASK))


# ----------------------------------------------------------------------------- comandi: percorsi PASS e FAIL

class TestCommands(Base):
    REPORTS = ("overlap_report.json", "overlap_report.md", "overlap_report_public.json", "overlap_report_public.md")

    def rgb_world(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        return CR.write_world(d, CR.casia_world())

    def nuaa_world(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        return CN.write_world(d, CN.clean_world())

    def overlap(self, tag, rp, man, out, *extra):
        if tag == "rgb":
            return run(RGB, "check-overlap", "--dataset", "casia_fasd", "--rules", rp, "--manifests-dir", man, "--out", out, *extra, t=self.t)
        return run(NUAA, "check-overlap", "--rules", rp, "--manifests-dir", man, "--out", out, *extra, t=self.t)

    def test_links_at_output_names_pass_and_fail(self):
        # collegamenti preparati con i nomi delle uscite e dei temporanei a nome fisso: file esterno intatto dopo PASS e FAIL
        for tag in ("rgb", "nuaa"):
            for kind in ("symlink", "hardlink", "dangling"):
                d = self.t / f"{tag}-{kind}"
                rp = (self.rgb_world if tag == "rgb" else self.nuaa_world)(d)
                out = d / "out"
                out.mkdir()
                for n in self.REPORTS:
                    link(kind, out / (n + ".tmp"), self.ext)
                link(kind, out / "overlap_report.json", self.ext)
                inputs = tree_bytes(d / "manifests"), rp.read_bytes()
                r = self.overlap(tag, rp, d / "manifests", out)
                self.assertEqual(r.returncode, 0, (tag, kind, r.stderr))
                self.assert_external_intact((tag, kind, "PASS"))
                for n in self.REPORTS:
                    self.assertFalse((out / n).is_symlink(), (tag, kind, n))
                    self.assertTrue((out / n).is_file(), (tag, kind, n))
                self.assertEqual(self.no_temporaries(out), [], (tag, kind))
                self.assertEqual((tree_bytes(d / "manifests"), rp.read_bytes()), inputs)
                # FAIL preliminare nella stessa cartella, con i collegamenti preparati di nuovo
                for n in self.REPORTS:
                    link(kind, out / (n + ".tmp"), self.ext)
                doc = json.loads(rp.read_text())
                key = "eval" if tag == "rgb" else "c1"
                doc["manifests"][key]["expected_sha256"] = "0" * 64
                rp.write_text(json.dumps(doc))
                r = self.overlap(tag, rp, d / "manifests", out)
                self.assertEqual(r.returncode, 1, (tag, kind))
                self.assertNotIn("Traceback", r.stderr)
                self.assert_external_intact((tag, kind, "FAIL"))
                self.assertFalse(json.loads((out / "overlap_report_public.json").read_text())["passed"])
                self.assertFalse((out / "overlap_report.md").exists())
                self.assertEqual(self.no_temporaries(out), [], (tag, kind))

    def assert_refused(self, r, what):
        self.assertEqual(r.returncode, 1, (what, r.stdout + r.stderr))
        self.assertIn("ERROR:", r.stderr, what)
        self.assertNotIn("Traceback", r.stderr, what)

    def test_output_folder_link_file_and_inputs(self):
        for tag in ("rgb", "nuaa"):
            d = self.t / f"{tag}-io"
            rp = (self.rgb_world if tag == "rgb" else self.nuaa_world)(d)
            man = d / "manifests"
            ext_dir = self.t / f"{tag}-ext-dir"
            ext_dir.mkdir()
            (d / "out_link").symlink_to(ext_dir)
            (d / "out_file").write_text("a file\n")
            (man / "sub").mkdir()
            prepared = d / "prepared"
            prepared.mkdir()
            (prepared / "overlap_report.json").write_bytes(rp.read_bytes())       # regole con il nome di un'uscita
            (prepared / "overlap_report.json.tmp").write_bytes(rp.read_bytes())   # ... o del suo temporaneo a nome fisso
            (prepared / "keep.txt").write_text("keep\n")
            before = tree_bytes(d)
            cases = {"--out symbolic link": (rp, man, d / "out_link"), "--out is a file": (rp, man, d / "out_file"),
                     "--out equal to --manifests-dir": (rp, man, man), "--out inside --manifests-dir": (rp, man, man / "sub"),
                     "--rules is an output of --out": (prepared / "overlap_report.json", man, prepared),
                     "--rules is the fixed temporary of an output": (prepared / "overlap_report.json.tmp", man, prepared)}
            for what, (rules, m, out) in cases.items():
                r = self.overlap(tag, rules, m, out)
                self.assert_refused(r, (tag, what))
                self.assertIn("refused", r.stderr, (tag, what))
                self.assertEqual(tree_bytes(d), before, (tag, what))           # nessun file scritto o tolto, ingressi identici
                self.assertEqual(list(ext_dir.iterdir()), [], (tag, what))
            # report: modello con il nome di un rapporto breve
            names = RM.report_outputs("casia_fasd") if tag == "rgb" else NM.REPORT_OUTPUTS
            tmpl = prepared / names[0]
            tmpl.write_text("{x}\n")
            before = tree_bytes(d)
            args = (["report", "--dataset", "casia_fasd"] if tag == "rgb" else ["report"]) + ["--out", prepared, "--template", tmpl]
            r = run(RGB if tag == "rgb" else NUAA, *args, t=self.t)
            self.assert_refused(r, (tag, "template"))
            self.assertIn("--template", r.stderr)
            self.assertEqual(tree_bytes(d), before, tag)

    def test_build_output_folders(self):
        # build: cartella che contiene i dati, <out>/manifests collegamento o file; manifest come collegamento
        for tag, script in (("rgb", RGB), ("nuaa", NUAA)):
            t = self.t / f"{tag}-build"
            (t / "data").mkdir(parents=True)
            if tag == "rgb":
                CR.make_casia_tree(t)
            ext_dir = self.t / f"{tag}-ext-man"
            ext_dir.mkdir()
            base = ["build", "--dataset", "casia_fasd"] if tag == "rgb" else ["build"]
            snap = tree_bytes(t / "data")
            r = run(script, *base, "--out", t, t=self.t, env=env_for(t))            # cartella antenata dei dati
            self.assert_refused(r, (tag, "ancestor of data"))
            self.assertIn("data", r.stderr)
            o1 = t / "o1"
            o1.mkdir()
            (o1 / "manifests").symlink_to(ext_dir)
            summary = o1 / ("casia_fasd_manifests_summary.json" if tag == "rgb" else "nuaa_manifests_summary.json")
            summary.write_text("summary of an earlier run\n")
            r = run(script, *base, "--out", o1, t=self.t, env=env_for(t))
            self.assert_refused(r, (tag, "manifests link"))
            self.assertIn("symbolic link", r.stderr)
            self.assertEqual(summary.read_text(), "summary of an earlier run\n", tag)     # rifiuto prima di qualunque modifica
            o2 = t / "o2"
            o2.mkdir()
            (o2 / "manifests").write_text("x")
            r = run(script, *base, "--out", o2, t=self.t, env=env_for(t))
            self.assert_refused(r, (tag, "manifests file"))
            self.assertEqual(list(ext_dir.iterdir()), [], tag)
            self.assertEqual(tree_bytes(t / "data"), snap, tag)
        # rgb: un manifest preparato come collegamento (simbolico o fisso) a un file esterno: il file esterno resta intatto
        t = self.t / "rgb-build"
        for kind in ("symlink", "hardlink"):
            out = t / f"o-{kind}"
            (out / "manifests").mkdir(parents=True)
            link(kind, out / "manifests" / RM.SPEC["casia_fasd"]["manifests"]["eval"][0], self.ext)
            r = run(RGB, "build", "--dataset", "casia_fasd", "--out", out, t=self.t, env=env_for(t))
            self.assertEqual(r.returncode, 1)                                    # build fallito (copia sintetica guasta)
            self.assertNotIn("Traceback", r.stderr)
            self.assert_external_intact(kind)
            self.assertFalse((out / "manifests" / RM.SPEC["casia_fasd"]["manifests"]["eval"][0]).is_symlink())

    def test_nuaa_refuses_the_outputs_of_the_rgb_tranche(self):
        d = self.t / "nuaa-foreign"
        rp = self.nuaa_world(d)
        out = d / "out"
        out.mkdir()
        kept = {n: f"{n} of the RGB tranche\n" for n in ("casia_fasd_manifests_summary.json", "overlap_report.json", "overlap_self_test.json")}
        for n, txt in kept.items():
            (out / n).write_text(txt)
        for args in (["check-overlap", "--rules", rp, "--manifests-dir", d / "manifests"], ["report"], ["build"]):
            r = run(NUAA, *args, "--out", out, t=self.t, env=env_for(d))
            self.assert_refused(r, args[0])
            self.assertIn("CASIA-FASD/CelebA-Spoof tranche", r.stderr)
        self.assertEqual({n: (out / n).read_text() for n in kept}, kept)


# ----------------------------------------------------------------------------- self-test ed errori imprevisti

# processo separato con un errore iniettato (argv: cartella scripts, modo, nome, argomenti del comando):
#   write <nome> <eccezione>  write_all scrive davvero il gruppo, poi solleva l'eccezione se il gruppo contiene <nome>
#   selftest - <eccezione>    run_self_test solleva subito l'eccezione
#   validate - <eccezione>    outputs_problem solleva l'eccezione (errore durante la validazione)
#   refill <nomi,...> <ecc.>  la prima invalidate toglie, poi riscrive le uscite indicate con passed true e solleva
DRIVER = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import {mod} as M
MODE, NAME, EXC = sys.argv[2], sys.argv[3], {{"RuntimeError": RuntimeError, "KeyboardInterrupt": KeyboardInterrupt}}[sys.argv[4]]
state = {{"done": False}}
def boom(*a, **k):
    raise EXC("injected failure")
if MODE == "write":
    real = M.write_all
    def write_then_fail(out, files):
        real(out, files)
        if NAME in files:
            raise EXC("injected failure")
    M.write_all = write_then_fail
elif MODE == "cleanupfail":
    real = M.write_all
    def eacces(*a, **k):
        raise PermissionError(13, "injected EACCES")
    def write_then_fail_and_lock(out, files):
        real(out, files)
        if NAME in files:
            M.invalidate = eacces                       # da qui nessuna uscita si può togliere
            raise EXC("injected failure")
    M.write_all = write_then_fail_and_lock
elif MODE == "selftest":
    M.run_self_test = boom
elif MODE == "validate":
    M.outputs_problem = boom
elif MODE == "refill":
    real_inv = M.invalidate
    def refill(out, names):
        r = real_inv(out, names)
        if not state["done"]:
            state["done"] = True
            for n in NAME.split(","):
                (Path(out) / n).write_text(json.dumps({{"passed": True}}) + "\\n")
            raise EXC("injected failure")
        return r
    M.invalidate = refill
sys.argv = [M.__file__] + sys.argv[5:]
M.main()
"""


def run_driver(t: Path, mod: str, mode: str, name: str, exc: str, *args, env=None):
    drv = t / f"driver_{mod}.py"
    drv.write_text(DRIVER.format(mod=mod))
    return subprocess.run([sys.executable, str(drv), str(ROOT / "scripts"), mode, name, exc, *map(str, args)], capture_output=True,
                          text=True, cwd=t, env=env)


class TestFailuresNeverLeaveAPass(Base):
    """Revisione indipendente del 22/10: `check-overlap --self-test` su manifest senza `pooled_train` usciva con un
    KeyError (traceback) lasciando i rapporti a passed true; ora ogni fallimento del self-test e ogni errore imprevisto,
    anche dopo la scrittura delle uscite, danno una riga FAIL esplicita, codice 1 e nessuna uscita con passed true."""

    def assert_no_pass(self, out: Path, stage: str, self_test: bool = True):
        for n in ("overlap_report.json", "overlap_report_public.json"):
            rep = json.loads((out / n).read_text())
            self.assertIs(rep["passed"], False, n)
            self.assertEqual(rep["stage"], stage, n)
        self.assertNotIn("error", json.loads((out / "overlap_report_public.json").read_text()))
        self.assertIn("error", json.loads((out / "overlap_report.json").read_text()))
        for n in ("overlap_report.md", "overlap_report_public.md"):
            self.assertFalse((out / n).exists(), n)
        if self_test:
            self.assertIs(json.loads((out / "overlap_self_test.json").read_text())["passed"], False)
        for p in out.glob("*.json"):
            self.assertNotEqual(json.loads(p.read_text()).get("passed"), True, p.name)
        self.assertEqual(self.no_temporaries(out), [])

    def assert_fail_line(self, r, *texts):
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        for t in texts:
            self.assertIn(t, r.stderr)

    def test_self_test_without_the_needed_manifests(self):
        # mondi sintetici senza i manifest richiesti dal self-test (RGB: pooled_*, NUAA: official_*, m1_dev, ...)
        for tag, script, rp_of, args in (
                ("rgb", RGB, lambda d: CR.write_world(d, CR.casia_world()), ["--dataset", "casia_fasd"]),
                ("nuaa", NUAA, lambda d: CN.write_world(d, CN.clean_world()), [])):
            d = self.t / tag
            d.mkdir()
            rp = rp_of(d)
            before = tree_bytes(d / "manifests"), rp.read_bytes()
            r = run(script, "check-overlap", *args, "--rules", rp, "--manifests-dir", d / "manifests", "--out", d / "out", "--self-test", t=self.t)
            self.assert_fail_line(r, "self-test FAIL", "--self-test needs", "missing")
            self.assert_no_pass(d / "out", "self-test")
            self.assertEqual((tree_bytes(d / "manifests"), rp.read_bytes()), before, tag)

    def in_process_overlap(self, tag, d, patch, value):
        mod = RM if tag == "rgb" else NM
        rp = CR.write_world(d, CR.casia_world()) if tag == "rgb" else CN.write_world(d, CN.clean_world())
        argv = (["x", "check-overlap", "--dataset", "casia_fasd"] if tag == "rgb" else ["x", "check-overlap"]) + [
            "--rules", str(rp), "--manifests-dir", str(d / "manifests"), "--out", str(d / "out"), "--self-test"]
        err = io.StringIO()
        with mock.patch.object(sys, "argv", argv), mock.patch.object(mod, patch, value), contextlib.redirect_stderr(err), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as cm:
            mod.main()
        self.assertEqual(cm.exception.code, 1)
        return err.getvalue()

    def test_self_test_exception_or_failure_after_the_reports(self):
        for tag in ("rgb", "nuaa"):
            # errore imprevisto dentro il self-test, dopo i quattro rapporti PASS
            d = self.t / f"{tag}-exc"
            d.mkdir()
            out = self.in_process_overlap(tag, d, "run_self_test", mock.Mock(side_effect=KeyError("pooled_train")))
            self.assertIn("self-test FAIL: KeyError", out)
            self.assert_no_pass(d / "out", "self-test")
            # fail() (SystemExit) dentro il self-test: rapporti marcati FAIL anche così
            d = self.t / f"{tag}-exit"
            d.mkdir()
            mod = RM if tag == "rgb" else NM
            out = self.in_process_overlap(tag, d, "run_self_test", mock.Mock(side_effect=lambda *a, **k: mod.fail("refused inside the self-test")))
            self.assertIn("self-test FAIL: SystemExit", out)
            self.assert_no_pass(d / "out", "self-test")
            # self-test che non rileva un guasto come atteso (scrive il suo rapporto FAIL): rapporti marcati FAIL
            d = self.t / f"{tag}-notexp"
            d.mkdir()

            def fake(*a, **k):
                o = a[5] if tag == "rgb" else a[4]
                (Path(o) / "overlap_self_test.json").write_text('{"passed": false}\n')
                (Path(o) / "overlap_self_test.md").write_text("FAIL\n")
                return False
            out = self.in_process_overlap(tag, d, "run_self_test", fake)
            self.assertIn("self-test FAIL: a faulty copy was not rejected", out)
            self.assert_no_pass(d / "out", "self-test")
            self.assertTrue((d / "out" / "overlap_self_test.md").exists())

    def overlap_args(self, tag, d, rp, *extra):
        return ["check-overlap"] + (["--dataset", "casia_fasd"] if tag == "rgb" else []) + [
            "--rules", rp, "--manifests-dir", d / "manifests", "--out", d / "out", *extra]

    def world(self, tag, d):
        d.mkdir(parents=True, exist_ok=True)
        return CR.write_world(d, CR.casia_world()) if tag == "rgb" else CN.write_world(d, CN.clean_world())

    def test_unexpected_error_after_writing_the_outputs(self):
        # errore imprevisto subito dopo la scrittura dei quattro rapporti PASS di check-overlap
        for tag, mod in (("rgb", "rgb_manifests"), ("nuaa", "nuaa_manifests")):
            d = self.t / f"{tag}-after"
            rp = self.world(tag, d)
            r = run_driver(self.t, mod, "write", "overlap_report_public.md", "RuntimeError", *self.overlap_args(tag, d, rp))
            self.assert_fail_line(r, "check-overlap FAIL: unexpected error RuntimeError: injected failure")
            self.assert_no_pass(d / "out", "error", self_test=False)
        # build: errore subito dopo la scrittura del riepilogo (copia sintetica di CASIA-FASD); un aggregato di un run precedente
        t = self.t / "rgb-build"
        CR.make_casia_tree(t)
        out = t / "out"
        out.mkdir()
        (out / "casia_fasd_manifests_aggregate.json").write_text('{"passed": true}\\n')
        r = run_driver(self.t, "rgb_manifests", "write", "casia_fasd_manifests_summary.json", "RuntimeError", "build", "--dataset", "casia_fasd",
                       "--out", out, env=env_for(t))
        self.assert_fail_line(r, "build FAIL: unexpected error RuntimeError: injected failure")
        self.assertEqual([n for n in RM.build_outputs("casia_fasd") if (out / n).exists()], [])
        self.assertEqual(self.no_temporaries(out), [])

    def test_unexpected_error_removes_own_and_downstream_outputs(self):
        # dopo la validazione, per ogni comando: tutte le uscite del comando e quelle a valle tolte, le altre restano
        for tag, mod in (("rgb", "rgb_manifests"), ("nuaa", "nuaa_manifests")):
            if tag == "rgb":
                own = {"build": RM.build_outputs("casia_fasd"), "check-overlap": RM.overlap_outputs() + RM.report_outputs("casia_fasd"),
                       "report": RM.report_outputs("casia_fasd")}
            else:
                own = {"build": list(NM.BUILD_OUTPUTS + NM.OVERLAP_OUTPUTS + NM.REPORT_OUTPUTS),
                       "check-overlap": list(NM.OVERLAP_OUTPUTS + NM.REPORT_OUTPUTS), "report": list(NM.REPORT_OUTPUTS)}
            every = own["build"]
            for cmd, names in own.items():
                d = self.t / f"{tag}-{cmd}"
                rp = self.world(tag, d)
                out = d / "out"
                out.mkdir()
                for n in every:                                   # uscite PASS di un run precedente
                    (out / n).write_text('{"passed": true}\\n')
                args = {"build": ["build"], "report": ["report"], "check-overlap": []}[cmd]
                if tag == "rgb" and cmd != "check-overlap":
                    args = args + ["--dataset", "casia_fasd"]
                args = self.overlap_args(tag, d, rp) if cmd == "check-overlap" else args + ["--out", out]
                r = run_driver(self.t, mod, "refill", ",".join(every), "RuntimeError", *args, env=env_for(d))
                self.assert_fail_line(r, f"{cmd} FAIL: unexpected error RuntimeError: injected failure")
                left = sorted(p.name for p in out.iterdir())
                expected = sorted(set(every) - set(names)) + (["overlap_report.json", "overlap_report_public.json"] if cmd == "check-overlap" else [])
                self.assertEqual(left, sorted(expected), (tag, cmd))
                if cmd == "check-overlap":
                    for n in ("overlap_report.json", "overlap_report_public.json"):
                        self.assertEqual(json.loads((out / n).read_text())["stage"], "error", (tag, n))

    def test_clean_up_that_cannot_remove_an_output(self):
        # doppio guasto: errore dopo i rapporti PASS e poi nessuna uscita si può togliere (EACCES): ogni JSON diventa FAIL,
        # ogni file rimasto è nominato, codice 1
        for tag, mod in (("rgb", "rgb_manifests"), ("nuaa", "nuaa_manifests")):
            d = self.t / f"{tag}-locked"
            rp = self.world(tag, d)
            r = run_driver(self.t, mod, "cleanupfail", "overlap_report_public.md", "RuntimeError", *self.overlap_args(tag, d, rp))
            self.assert_fail_line(r, "check-overlap FAIL: unexpected error RuntimeError: injected failure")
            for n in ("overlap_report.md", "overlap_report_public.md"):
                self.assertIn(f"could not invalidate {n} (PermissionError", r.stderr, (tag, n))
                self.assertIn(f"could not invalidate {n} (PermissionError: [Errno 13] injected EACCES): left behind", r.stderr, (tag, n))
            self.assertIn("could not invalidate overlap_report_public.json (PermissionError: [Errno 13] injected EACCES): overwritten with a FAIL document", r.stderr)
            for p in (d / "out").glob("*.json"):
                self.assertIs(json.loads(p.read_text())["passed"], False, (tag, p.name))
            self.assertNotIn("error", json.loads((d / "out" / "overlap_report_public.json").read_text()))

    def test_error_during_validation_removes_nothing(self):
        # errore imprevisto dentro la validazione della cartella di uscita: nulla si toglie né si scrive, riga FAIL esplicita
        for tag, mod in (("rgb", "rgb_manifests"), ("nuaa", "nuaa_manifests")):
            d = self.t / f"{tag}-val"
            rp = self.world(tag, d)
            (d / "out").mkdir()
            for n in ("overlap_report.json", "overlap_report_public.json", "overlap_report.md"):
                (d / "out" / n).write_text('{"passed": true}\\n')
            before = tree_bytes(d)
            r = run_driver(self.t, mod, "validate", "-", "RuntimeError", *self.overlap_args(tag, d, rp))
            self.assert_fail_line(r, "check-overlap FAIL: unexpected error RuntimeError: injected failure")
            self.assertEqual(tree_bytes(d), before, tag)

    def test_keyboard_interrupt_never_leaves_a_pass(self):
        # Ctrl-C nel self-test e dopo la scrittura dei rapporti: rapporti FAIL, riga FAIL, poi l'interruzione (non il codice 1)
        for tag, mod in (("rgb", "rgb_manifests"), ("nuaa", "nuaa_manifests")):
            for mode, name in (("selftest", "-"), ("write", "overlap_report_public.md")):
                d = self.t / f"{tag}-int-{mode}"
                rp = self.world(tag, d)
                extra = ["--self-test"] if mode == "selftest" else []
                r = run_driver(self.t, mod, mode, name, "KeyboardInterrupt", *self.overlap_args(tag, d, rp, *extra))
                self.assertIn(r.returncode, (130, -2), (tag, mode, r.stderr[-500:]))
                self.assertIn("FAIL", r.stderr)
                for p in (d / "out").glob("*.json"):
                    self.assertIs(json.loads(p.read_text()).get("passed"), False, (tag, mode, p.name))
                self.assertFalse((d / "out" / "overlap_report_public.md").exists(), (tag, mode))
                self.assertTrue((d / "out" / "overlap_report_public.json").exists(), (tag, mode))
                self.assertEqual(self.no_temporaries(d / "out"), [])


# ----------------------------------------------------------------------------- radici protette

def fake_fork(t: Path, pkg: str = "m2-writers-maintenance") -> tuple:
    """Copia sintetica della disposizione del repository del gruppo con una sentinella in ogni pacchetto accettato."""
    f = t / "fork"
    sentinels = []
    for p in ("c1-livenesslab/livenesslab/models/weights", "m1-metrics", "m2-nuaa-manifests/livenesslab/scripts",
              "m2-rgb-manifests/livenesslab/results/m2_rgb", "m2-c2-pilot-impl/livenesslab/scripts", f"{pkg}/livenesslab/results/m2_rgb"):
        (f / "research" / p).mkdir(parents=True, exist_ok=True)
        s = f / "research" / p / "SENTINEL"
        s.write_text("accepted\n")
        sentinels.append(s)
    return f, f / "research" / pkg / "livenesslab", sentinels


class TestProtectedRoots(Base):
    def setUp(self):
        super().setUp()
        MW._outer_roots.cache_clear()

    def tearDown(self):
        MW._outer_roots.cache_clear()
        super().tearDown()

    def test_repository_folders_protected(self):
        r = self.t / "repo"
        for p in ("data", "models/weights", "results/c1", "results/m1", "results/m2", "results/m2_rgb/casia_fasd", "results/eval",
                  "docs", "scripts", "delivery/m2/rgb", "export/research", ".git"):
            (r / p).mkdir(parents=True)
        for mod, own, other in ((RM, "results/m2_rgb", "results/m2"), (NM, "results/m2", "results/m2_rgb")):
            with mock.patch.object(mod, "ROOT", r), mock.patch.object(mod, "DATA", r / "data"), mock.patch.object(mod, "WEIGHTS", r / "models" / "weights"), \
                    mock.patch.object(mod, "RESULTS", r / "results"):
                ok_dir = r / own / "casia_fasd" if mod is RM else r / own
                out_dir = r / "results" / "m2_rgb" if mod is RM else r / "results" / "m2"
                with mock.patch.object(mod, "OUT_DIR" if mod is RM else "M2_DIR", out_dir):
                    problem = (lambda o: RM.outputs_problem(o, "casia_fasd", ["a.json"], {})) if mod is RM else (lambda o: NM.outputs_problem(o, ["a.json"], {}))
                    self.assertEqual(problem(ok_dir), "", mod.__name__)
                    self.assertEqual(problem(self.t / "scratch"), "", mod.__name__)
                    for bad in ("", "results", "results/c1", "results/m1", "results/eval", other, "data", "data/x", "models/weights", "docs",
                                "scripts", "delivery/m2/rgb", "export/research", ".git", "notes"):
                        self.assertIn("refused", problem(r / bad), (mod.__name__, bad))
                    self.assertIn("refused", problem(self.t), mod.__name__)        # antenato del repository
        # dati spostati con le variabili d'ambiente: rifiutati in entrambe le direzioni anche fuori dal repository
        moved = self.t / "elsewhere" / "data"
        moved.mkdir(parents=True)
        with mock.patch.object(RM, "DATA", moved):
            self.assertIn("protected folder", RM.outputs_problem(moved / "x", "casia_fasd", ["a.json"], {}))
            self.assertIn("protected folder", RM.outputs_problem(moved.parent, "casia_fasd", ["a.json"], {}))
        with mock.patch.object(NM, "WEIGHTS", moved):
            self.assertIn("protected folder", NM.outputs_problem(moved.parent, ["a.json"], {}))

    def test_sibling_packages_in_the_group_layout(self):
        f, root, sentinels = fake_fork(self.t)
        for mod in (RM, NM):
            with mock.patch.object(mod, "ROOT", root):
                problem = (lambda o: RM.outputs_problem(o, "casia_fasd", ["a.json"], {})) if mod is RM else (lambda o: NM.outputs_problem(o, ["a.json"], {}))
                for where in ("research/c1-livenesslab/livenesslab/models/weights", "research/m1-metrics", "research/m2-nuaa-manifests",
                              "research/m2-rgb-manifests/livenesslab/results/m2_rgb", "research/m2-c2-pilot-impl",
                              "research/m2-writers-maintenance/livenesslab/results/m2_rgb", "research", ".", "other"):
                    self.assertIn("work tree", problem(f / where), (mod.__name__, where))
                self.assertIn("work tree", problem(self.t), mod.__name__)            # cartella che contiene l'albero
                self.assertEqual(problem(self.t / "outside"), "", mod.__name__)
        with mock.patch.object(EX, "ROOT", root):
            for where in ("research/m1-metrics", "research/m2-nuaa-manifests", "research/m2-rgb-manifests", "research/m2-rgb-manifests/livenesslab/results/m2_rgb",
                          "other/out"):
                self.assertIn("work tree", EX.deletion_problem(f / where, "20261022-M2-rgb-seed42-test000", root=root), where)
            self.assertIn("repository root or one of its ancestors", EX.deletion_problem(f, "20261022-M2-rgb-seed42-test000", root=root))
            # dentro livenesslab/ vale già path_guard (cartelle dell'export dentro la radice)
            self.assertIn("not a folder of this export", EX.deletion_problem(root / "export", "20261022-M2-rgb-seed42-test000", root=root))
            self.assertEqual(EX.deletion_problem(self.t / "outside", "20261022-M2-rgb-seed42-test000", root=root), "")
        for s in sentinels:
            self.assertEqual(s.read_text(), "accepted\n")

    def copy_code(self, root: Path):
        for item in ("scripts", "src/tesi_app", "delivery/m2"):
            shutil.copytree(ROOT / item, root / item, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    def test_commands_from_the_group_layout(self):
        # i comandi veri, dalla copia del codice nella disposizione del gruppo: ogni uscita nell'albero è rifiutata
        f, root, sentinels = fake_fork(self.t)
        self.copy_code(root)
        d = self.t / "w"
        d.mkdir()
        rp = CR.write_world(d, CR.casia_world())
        rpn = CN.write_world(d / "n", CN.clean_world())
        before = tree_bytes(f / "research" / "m2-rgb-manifests"), tree_bytes(f / "research" / "m2-nuaa-manifests"), tree_bytes(d)
        env = env_for(self.t / "empty")
        for out in (f / "research" / "m2-rgb-manifests" / "livenesslab" / "results" / "m2_rgb", f / "research" / "m2-nuaa-manifests",
                    root / "results" / "m2_rgb" / "casia_fasd"):
            r = run(root / "scripts" / "rgb_manifests.py", "check-overlap", "--dataset", "casia_fasd", "--rules", rp, "--manifests-dir",
                    d / "manifests", "--out", out, t=self.t, env=env)
            self.assert_refused_tree(r, out)
            r = run(root / "scripts" / "nuaa_manifests.py", "check-overlap", "--rules", rpn, "--manifests-dir", d / "n" / "manifests",
                    "--out", out, t=self.t, env=env)
            self.assert_refused_tree(r, out)
        rid = ("--run-id", "20261022-M2-rgb-seed42-test000")
        r = run(root / "scripts" / "export_m2_rgb.py", "--trial", "--dry-run", *rid, "--m2", f / "research" / "m1-metrics", "--out", self.t / "o", t=self.t, env=env)
        self.assert_refused_tree(r, "export --m2")
        r = run(root / "scripts" / "export_m2_rgb.py", "--trial", "--dry-run", *rid, "--m2", self.t / "m2", "--out", f, t=self.t, env=env)
        self.assert_refused_tree(r, "export --out")
        self.assertEqual((tree_bytes(f / "research" / "m2-rgb-manifests"), tree_bytes(f / "research" / "m2-nuaa-manifests"), tree_bytes(d)), before)
        for s in sentinels:
            self.assertEqual(s.read_text(), "accepted\n")
        # fuori dall'albero: ammesso
        r = run(root / "scripts" / "rgb_manifests.py", "check-overlap", "--dataset", "casia_fasd", "--rules", rp, "--manifests-dir",
                d / "manifests", "--out", self.t / "ok", t=self.t, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)

    def assert_refused_tree(self, r, what):
        self.assertEqual(r.returncode, 1, (what, r.stdout + r.stderr))
        self.assertIn("work tree", r.stderr, what)
        self.assertNotIn("Traceback", r.stderr, what)

    def test_git_work_tree_protected(self):
        if shutil.which("git") is None:
            skip_missing(self, "git")
        g = self.t / "gitrepo"
        root = g / "pkg" / "livenesslab"                       # nessun research/: vale l'albero di lavoro git
        root.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(g)], check=True, capture_output=True)
        with mock.patch.object(RM, "ROOT", root):
            self.assertIn("work tree", RM.outputs_problem(g / "other", "casia_fasd", ["a.json"], {}))
            self.assertIn("work tree", RM.outputs_problem(g, "casia_fasd", ["a.json"], {}))
            self.assertEqual(RM.outputs_problem(self.t / "outside", "casia_fasd", ["a.json"], {}), "")
        with mock.patch.object(EX, "ROOT", root):
            self.assertIn("work tree", EX.deletion_problem(g / "other", "20261022-M2-rgb-seed42-test000", root=root))
        self.assertEqual(MW.outer_roots(ROOT), MW.outer_roots(ROOT))

    def test_export_protects_moved_data_folders(self):
        # ogni cartella protetta spostata con le variabili d'ambiente: uguale, interna e antenata rifiutate; la cartella dei
        # risultati di questa tranche (m2_rgb) resta quella che l'export rigenera
        run_id = "20261022-M2-rgb-seed42-test000"
        for name in ("DATA", "WEIGHTS"):
            moved = self.t / f"moved_{name}" / "x"
            moved.mkdir(parents=True)
            with mock.patch.object(EX.RM, name, moved):
                for p in (moved, moved / "sub", moved.parent):
                    self.assertIn("protected folder", EX.deletion_problem(p, run_id), (name, str(p)))
                self.assertEqual(EX.deletion_problem(self.t / f"other_{name}", run_id), "", name)
        results = self.t / "moved_results"
        for sub in ("c1", "eval", "m1", "m2", "m2_rgb"):
            (results / sub).mkdir(parents=True)
        with mock.patch.object(EX.RM, "RESULTS", results):
            for sub in ("c1", "eval", "m1", "m2"):
                for p in (results / sub, results / sub / "run"):
                    self.assertIn(f"protected folder ({sub})", EX.deletion_problem(p, run_id), str(p))
            self.assertIn("protected folder", EX.deletion_problem(results, run_id))            # antenata di tutte
            self.assertEqual(EX.deletion_problem(results / "m2_rgb", run_id), "")
            self.assertEqual(EX.deletion_problem(results / "m2_rgb" / "casia_fasd", run_id), "")


class TestExportMain(Base):
    """Il controllo dell'identità dei file condivisi con il pacchetto NUAA (reference_report) e i passi dell'export, con
    il main vero e un run finto (check_export_m2_rgb.TestLists.fake_export): nessun dato, solo cartelle temporanee."""

    def accepted(self):
        a = MW.accepted_text("scripts/nuaa_manifests.py", ROOT)
        if a is None:
            skip_missing(self, "accepted copies")
        return a

    def published(self, nuaa_sha):
        """SHA-256 pubblicati: quello dato per nuaa_manifests.py, quelli dei file correnti per gli altri."""
        return lambda pub: nuaa_sha if pub.endswith("nuaa_manifests.py") else MW.sha256_bytes((ROOT / pub.split("livenesslab/", 1)[1]).read_bytes())

    def test_reference_report(self):
        self.accepted()
        f = "scripts/nuaa_manifests.py"
        acc = MW.MAINTENANCE[f]["accepted"]
        with mock.patch.object(EX, "published_sha", self.published(acc)):
            lines, failed = EX.reference_report()
            self.assertEqual(failed, [])
            self.assertEqual(sum(MW.MAINT_LABEL in x for x in lines), 1)
            self.assertEqual(len(lines), len(EX.REFERENCE_FILES))
            # copia dell'albero con nuaa_manifests.py cambiato fuori dalla manutenzione dichiarata: controllo fallito
            root = self.t / "root"
            for g in EX.REFERENCE_FILES:
                (root / g).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / g, root / g)
            (root / f).write_bytes((ROOT / f).read_bytes() + b"# changed\n")
            lines, failed = EX.reference_report(root)
            self.assertEqual(failed, [f"{f} differs from the copy delivered in the NUAA package"])
            (root / f).write_bytes(MW.accepted_text(f, ROOT).encode("utf-8"))              # copia accettata: "="
            self.assertEqual(EX.reference_report(root)[1], [])
            (root / "scripts" / "package_rules.py").write_bytes((ROOT / "scripts" / "package_rules.py").read_bytes() + b"\n")
            self.assertEqual(EX.reference_report(root)[1], ["scripts/package_rules.py differs from the copy delivered in the NUAA package"])
        with mock.patch.object(EX, "published_sha", self.published("0" * 64)):
            self.assertEqual(EX.reference_report()[1], [f"{f} differs from the copy delivered in the NUAA package"])
        with mock.patch.object(EX, "published_sha", lambda pub: None):                          # pacchetto NUAA non disponibile
            lines, failed = EX.reference_report()
            self.assertEqual(failed, [])
            self.assertTrue(all("NOT VERIFIED AGAINST" in x for x in lines))

    def test_main_reports_the_nuaa_identity_and_runs_the_writer_suite(self):
        self.accepted()
        CE = load("check_export_m2_rgb")
        helper = CE.TestLists("test_main_refuses_failed_artefacts")
        msg = "scripts/nuaa_manifests.py differs from the copy delivered in the NUAA package"
        for nuaa_sha, expected in ((MW.MAINTENANCE["scripts/nuaa_manifests.py"]["accepted"], False), ("0" * 64, True)):
            d = self.t / f"export-{expected}"
            d.mkdir()
            with mock.patch.object(CE.EX, "published_sha", self.published(nuaa_sha)), mock.patch.object(CE.EX.mw, "_outer_roots", lambda r: ()):
                rc, out, _ = helper.fake_export(d)
            self.assertEqual(msg in out, expected, out)
            self.assertEqual(rc, 1)                                   # il run finto non ha manifest: l'export fallisce comunque
            self.assertIn(["scripts/check_m2_writers.py"], [c[1:] for c in helper.export_calls])


# ----------------------------------------------------------------------------- dichiarazione della manutenzione

class TestMaintenanceDeclaration(unittest.TestCase):
    def test_declared_functions_are_fixed(self):
        # allargare le funzioni dichiarate (per esempio a evaluate_rules o load_manifest) è rifiutato da questo test
        common = ("invalidate", "write_all", "write_manifest", "write_rows", "outputs_problem", "cmd_build")
        self.assertEqual(MW.MAINTENANCE["scripts/rgb_manifests.py"]["changed"],
                         common + ("VALIDATED", "self_test_faults", "run_self_test", "cmd_check_overlap", "self_test_failure",
                                   "cmd_report", "main", "unexpected_failure"))
        self.assertEqual(MW.MAINTENANCE["scripts/nuaa_manifests.py"]["changed"],
                         common + ("VALIDATED", "run_self_test", "cmd_check_overlap", "overlap_run", "self_test_failure",
                                   "cmd_report", "main", "unexpected_failure", "RGB_OUTPUTS"))
        self.assertEqual(MW.MAINTENANCE["scripts/export_m2_rgb.py"]["changed"], ("CODE", "deletion_problem", "reference_state", "reference_report", "main"))
        for m in MW.MAINTENANCE.values():
            self.assertEqual(m["added"], ("import m2_writers as mw",))
        self.assertEqual({f: m["accepted"][:12] for f, m in MW.MAINTENANCE.items()},
                         {"scripts/rgb_manifests.py": "61e8d1a668b2", "scripts/nuaa_manifests.py": "bc1ce7b1c50e", "scripts/export_m2_rgb.py": "fa5b1eea8f52"})

    def test_current_files_have_the_pinned_sha256(self):
        for f, m in MW.MAINTENANCE.items():
            self.assertEqual(MW.sha256_bytes((ROOT / f).read_bytes()), m["current"], f"{f}: update MAINTENANCE['current'] after review")

    def accepted(self, f):
        a = MW.accepted_text(f, ROOT)
        if a is None:
            skip_missing(self, "accepted copies")
        return a

    def test_changes_limited_to_the_declared_functions(self):
        for f in MW.MAINTENANCE:
            a = self.accepted(f)
            self.assertEqual(MW.maintenance_state(f, (ROOT / f).read_bytes(), a), MW.MAINT_LABEL, f)
            self.assertEqual(MW.maintenance_state(f, a.encode("utf-8"), a), "=", f)
            self.assertEqual(MW.maintenance_state(f, (ROOT / f).read_bytes(), None), "", f)             # senza copia accettata
            self.assertEqual(MW.maintenance_state(f, (ROOT / f).read_bytes(), a + "\n"), "", f)         # copia non verificata

    def test_other_changes_are_refused(self):
        cases = {"scripts/rgb_manifests.py": [("SEED = 42", "SEED = 43"),
                                              ("def rules_path(ds: str) -> Path:\n    return RULES_DIR", "def rules_path(ds: str) -> Path:\n    return ROOT")],
                 "scripts/nuaa_manifests.py": [("MAX_EPOCHS = 12", "MAX_EPOCHS = 13"), ('SESSION_SPLIT = {"01": "train"', 'SESSION_SPLIT = {"01": "test"')],
                 "scripts/export_m2_rgb.py": [('PUBLIC_NAME = "m2-rgb-manifests"', 'PUBLIC_NAME = "m2-rgb"'),
                                              ("def tree_gate(dirty: bool, trial: bool) -> str:\n    \"\"\"", "def tree_gate(dirty: bool, trial: bool) -> str:\n    return ''\n    \"\"\"")]}
        for f, changes in cases.items():
            a = self.accepted(f)
            cur = (ROOT / f).read_text(encoding="utf-8")
            for old, new in changes:
                self.assertEqual(cur.count(old), 1, (f, old))
                changed = cur.replace(old, new).encode("utf-8")
                self.assertEqual(MW.maintenance_state(f, changed, a), "", (f, old))                      # SHA-256 diverso
                with mock.patch.dict(MW.MAINTENANCE[f], {"current": MW.sha256_bytes(changed)}):
                    self.assertEqual(MW.maintenance_state(f, changed, a), "", (f, old))                  # AST diverso fuori
            # una modifica dentro una funzione dichiarata: senza lo SHA-256 dichiarato è rifiutata; con lo SHA-256 dichiarato
            # l'AST fuori dalle funzioni dichiarate resta quello della copia accettata
            head = "def main() -> int:\n" if f.endswith("export_m2_rgb.py") else "def cmd_report(a) -> None:\n"
            self.assertEqual(cur.count(head), 1, f)
            body = cur.replace(head, head + "    pass\n").encode("utf-8")
            self.assertEqual(MW.maintenance_state(f, body, a), "", f)
            with mock.patch.dict(MW.MAINTENANCE[f], {"current": MW.sha256_bytes(body)}):
                self.assertEqual(MW.maintenance_state(f, body, a), MW.MAINT_LABEL, f)
            self.assertEqual(MW.maintenance_state(f, cur.replace("\n", "\r\n").encode("utf-8"), a), "", f)      # copia CRLF
            self.assertEqual(MW.maintenance_state(f, b"def broken(:", a), "", f)

    def test_export_reference_state(self):
        f = "scripts/nuaa_manifests.py"
        a = self.accepted(f)
        h = MW.MAINTENANCE[f]["accepted"]
        self.assertEqual(EX.reference_state(f, h, accepted=a), MW.MAINT_LABEL)
        self.assertEqual(EX.reference_state(f, h, a.encode("utf-8"), accepted=a), "=")
        self.assertEqual(EX.reference_state(f, h, (ROOT / f).read_bytes() + b"# x\n", accepted=a), "")
        self.assertEqual(EX.reference_state(f, "0" * 64, accepted=a), "")                                # SHA pubblicato diverso
        for g in ("scripts/select_threshold.py", "scripts/package_rules.py"):
            self.assertEqual(EX.reference_state(g, MW.sha256_bytes((ROOT / g).read_bytes())), "=")
            self.assertEqual(EX.reference_state(g, "0" * 64), "")
        self.assertIn("scripts/m2_writers.py", EX.CODE)

    def test_every_skip_is_a_missing_input(self):
        src = Path(__file__).read_text(encoding="utf-8")
        self.assertEqual(src.count(".skip" + "Test("), 1)
        used = re.findall(r'skip_missing\((?:self|test), "([^"]+)"', src)
        self.assertEqual(sorted(set(used)), sorted(MISSING_INPUT))
        for why in MISSING_INPUT.values():
            self.assertFalse(re.search(r"case|filesystem|platform|linux|macos|windows", why, re.I), why)


def main() -> int:
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c)
                               for c in (TestWriters, TestCommands, TestFailuresNeverLeaveAPass, TestProtectedRoots, TestExportMain, TestMaintenanceDeclaration))
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    if "--verbose" in sys.argv or not result.wasSuccessful():
        print(log.getvalue())
    for t, why in result.skipped:
        print(f"skipped {t.id().rsplit('.', 1)[-1]}: {why}")
    print(f"tests run: {result.testsRun}, failures: {len(result.failures)}, errors: {len(result.errors)}, skipped: {len(result.skipped)}")
    print("RESULT: " + ("PASS" if result.wasSuccessful() else "FAIL"))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
