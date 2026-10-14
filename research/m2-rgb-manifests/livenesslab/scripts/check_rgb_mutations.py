"""
Controllo delle mutazioni della seconda tranche M2 (CASIA-FASD e CelebA-Spoof), offline e deterministico.

Copia il codice (scripts/, src/tesi_app/, delivery/m2/rgb/, requirements.txt e, se presenti accanto al codice, i
documenti del pacchetto pubblico) in una cartella temporanea, applica una mutazione dichiarata alla volta a
scripts/rgb_manifests.py, scripts/export_m2_rgb.py o scripts/export_helpers.py (coppie cerca/sostituisci che devono comparire esattamente una
volta: se un modello non si trova più, il controllo si ferma con un errore), esegue i test offline sulla copia mutata
(scripts/check_rgb_manifests.py o scripts/check_export_m2_rgb.py, con cartelle dati, risultati e pesi vuote) e conta le
mutazioni rilevate: una mutazione è rilevata solo se la suite gira fino al riepilogo con almeno un fallimento o errore;
una suite che non parte (import o sintassi rotti) rende la mutazione non valida. Codice 1 se una mutazione sopravvive o
non è valida, o se un modello non si trova.

Nessuna rete, nessun dato, nessun percorso assoluto nell'uscita. Funziona dalla radice del repository di sviluppo e da
livenesslab/ del pacchetto pubblico.
  python scripts/check_rgb_mutations.py            # tutte le mutazioni (alcuni minuti)
  python scripts/check_rgb_mutations.py --list     # solo l'elenco, con la verifica dei modelli
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (nome, file mutato, test che deve rilevarla, testo cercato, testo sostitutivo)
RGB = "scripts/rgb_manifests.py"
EXPORT = "scripts/export_m2_rgb.py"
RGB_TESTS = "scripts/check_rgb_manifests.py"
EXPORT_TESTS = "scripts/check_export_m2_rgb.py"
RGB_MUTANTS = [
    ('check-overlap: no invalidate', '    invalidate(out, overlap_outputs() + report_outputs(ds))\n', '    pass\n'),
    ('build: no invalidate', '    invalidate(out, build_outputs(ds))\n', '    pass\n'),
    ('build: aggregate written on failure', '    if not problems:\n        files[f"{ds}_manifests_aggregate.json"] = agg_text', '    if True:\n        files[f"{ds}_manifests_aggregate.json"] = agg_text'),
    ('build: scan skipped', '    bad = public_text_problems(agg_text, ids) if not problems else []', '    bad = []'),
    ('preliminary: no FAIL reports', '        write_all(out, {"overlap_report.json": json.dumps({**base, "error": str(e)}, indent=1) + "\\n",\n                        "overlap_report_public.json": json.dumps(base, indent=1) + "\\n"})\n', '        pass\n'),
    ('preliminary: error text in public', '"overlap_report_public.json": json.dumps(base, indent=1)', '"overlap_report_public.json": json.dumps({**base, "error": str(e)}, indent=1)'),
    ('report: accepts failed overlap', '    if ov.get("passed") is not True:\n        fail(', '    if False:\n        fail('),
    ('report: no invalidate', '    invalidate(out, report_outputs(ds))                  #', '    pass  #'),
    ('invalidate: keeps .tmp', 'for q in (Path(out) / n, Path(out) / (n + ".tmp")):', 'for q in (Path(out) / n,):'),
    ('write_all: no cleanup', '        for n in files:\n            (out / (n + ".tmp")).unlink(missing_ok=True)\n            (out / n).unlink(missing_ok=True)\n        raise', '        raise'),
    ('write_all: cleanup only on OSError', '    except BaseException:\n', '    except OSError:\n'),
    ('write_all: direct write', '            t = out / (n + ".tmp")', '            t = out / n'),
    ('rules error not ManifestError', '        raise ManifestError(f"{relpath(rules_p)}: missing or invalid rules file", "rules") from None', '        raise'),
    ('integrity: fail instead of report', '        raise ManifestError(f"manifests differ from those declared in {relpath(rules_p)}: " + "; ".join(mismatch), "integrity")', '        fail("manifests differ")'),
    ('public-scan: old public kept', '        write_all(out, {"overlap_report.json": json.dumps({**rep, "passed": False, "stage": "public-scan",', '        write_all(out, {"overlap_report_public.json": pub_json, "overlap_report.json": json.dumps({**rep, "passed": False, "stage": "public-scan",'),
]
EXPORT_MUTANTS = [
    ("export: artefact check not called", "        failed += artefact_problems(m2 / ds, ds, commit)\n", "        pass\n"),
    ("export: dirty tree accepted", "    if dirty and not trial:\n        return (", "    if False:\n        return ("),
    ("export: failed or other-commit artefacts accepted",
     '        if d and (d.get("passed") is not True or d.get("commit") != commit):', "        if False:"),
]
HELPERS = "scripts/export_helpers.py"
HELPER_MUTANTS = [
    ("helpers: scan without text rules", "        for label, rx in FORBIDDEN_TEXT:\n", "        for label, rx in []:\n"),
    ("helpers: scan without forbidden extensions", "        if p.suffix.lower() in FORBIDDEN_EXT:\n", "        if False:\n"),
    ("helpers: scan without IP check", "            if ip not in IP_ALLOW and ip not in versions:\n", "            if False:\n"),
    ("helpers: scan without markers", "                if m := MARKERS.search(line):\n", "                if False:\n"),
    ("helpers: scan ignores symbolic links", "        if p.is_symlink():\n            bad.append((r, \"symbolic link\")); continue\n", ""),
    ("helpers: completeness ignores different SHA-256", "    if wrong:\n", "    if False:\n"),
    ("helpers: write_sums lists itself", ' and p.relative_to(folder).as_posix() != "SHA256SUMS")', ")"),
    ("helpers: redact is a no-op", '    txt = txt.replace(str(ROOT), "<repo>")', '    return txt\n    txt = txt.replace(str(ROOT), "<repo>")'),
    ("helpers: fill_run_id without the leftover check", '    if "{run_id}" in out or "<run-id>" in out:\n', "    if False:\n"),
    ("helpers: fill_run_id without the missing check", '    if "{run_id}" not in text:\n', "    if False:\n"),
]
MUTANTS = ([(n, RGB, RGB_TESTS, a, b) for n, a, b in RGB_MUTANTS] + [(n, EXPORT, EXPORT_TESTS, a, b) for n, a, b in EXPORT_MUTANTS]
           + [(n, HELPERS, EXPORT_TESTS, a, b) for n, a, b in HELPER_MUTANTS])
DOCS = ("README.md", "naming_schema_rgb.md")        # documenti letti da alcuni test se si trovano accanto al codice


def copy_tree(dst: Path) -> Path:
    """Copia minima del codice in dst/livenesslab (e dei documenti accanto, come nel pacchetto pubblico)."""
    code = dst / "livenesslab"
    for item in ("scripts", "src/tesi_app", "delivery/m2/rgb"):
        src = ROOT / item
        if src.is_dir():
            shutil.copytree(src, code / item, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in ("requirements.txt",):
        if (ROOT / f).is_file():
            shutil.copy2(ROOT / f, code / f)
    for n in DOCS:                                   # pacchetto pubblico: i documenti stanno nella cartella sopra livenesslab/
        if (ROOT.parent / n).is_file() and not (ROOT / "delivery" / "m2" / "rgb" / n).is_file():
            shutil.copy2(ROOT.parent / n, dst / n)
    return code


def pattern_problems() -> list:
    problems = []
    for name, target, _, old, _ in MUTANTS:
        n = (ROOT / target).read_text(encoding="utf-8").count(old)
        if n != 1:
            problems.append(f"{name}: the pattern occurs {n} times in {target} (expected exactly once)")
    return problems


def run_tests(code: Path, tests: str, empty: Path) -> tuple:
    env = {**os.environ, "LIVENESSLAB_DATA_DIR": str(empty / "data"), "LIVENESSLAB_RESULTS_DIR": str(empty / "results"),
           "LIVENESSLAB_WEIGHTS_DIR": str(empty / "weights"), "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-I", tests], cwd=code, env=env, capture_output=True, text=True, timeout=1800)
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("tests run")), "no test summary")
    return r.returncode, line


def detected(rc: int, line: str) -> str:
    """killed: la suite è girata (riga «tests run») e ha almeno un fallimento o errore; invalid: la suite non è partita
    (import o sintassi rotti) o è uscita con un codice senza fallimenti; survived: tutti i test passano."""
    import re
    m = re.match(r"tests run: (\d+), failures: (\d+), errors: (\d+)", line)
    if not m:
        return "invalid"
    if int(m.group(2)) + int(m.group(3)) > 0:
        return "killed"
    return "survived" if rc == 0 else "invalid"


def main() -> int:
    ap = argparse.ArgumentParser(description="Controllo delle mutazioni di rgb_manifests.py ed export_m2_rgb.py (offline)")
    ap.add_argument("--list", action="store_true", help="elenca le mutazioni e verifica i modelli, senza eseguire i test")
    a = ap.parse_args()
    problems = pattern_problems()
    for p in problems:
        print(f"ERROR: {p}", file=sys.stderr)
    if problems:
        print("RESULT: FAIL (a declared mutation no longer matches the code: update the list after review)")
        return 1
    if a.list:
        for name, target, tests, _, _ in MUTANTS:
            print(f"{name}  [{target} -> {tests}]")
        print(f"{len(MUTANTS)} mutations, every pattern found exactly once")
        return 0
    killed = invalid = 0
    with tempfile.TemporaryDirectory(prefix="m2-rgb-mutations-") as tmp:
        tmp = Path(tmp)
        empty = tmp / "empty"
        base = copy_tree(tmp / "base")
        rc, line = run_tests(base, RGB_TESTS, empty)
        rc2, line2 = run_tests(base, EXPORT_TESTS, empty)
        print(f"unmutated copy: {RGB_TESTS} exit {rc} ({line}); {EXPORT_TESTS} exit {rc2} ({line2})", flush=True)
        if rc or rc2:
            print("RESULT: FAIL (the offline tests do not pass on the unmutated copy)")
            return 1
        for name, target, tests, old, new in MUTANTS:
            work = tmp / "work"
            shutil.rmtree(work, ignore_errors=True)
            code = copy_tree(work)
            f = code / target
            f.write_text(f.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")
            rc, line = run_tests(code, tests, empty)
            verdict = detected(rc, line)
            killed += verdict == "killed"
            invalid += verdict == "invalid"
            print(f"{verdict.upper():<9} {name} ({line})", flush=True)
            if verdict == "invalid":
                print(f"ERROR: invalid mutant {name!r}: the test suite did not run to a summary with failures (exit {rc})", file=sys.stderr)
    print(f"mutations killed: {killed}/{len(MUTANTS)}" + (f" ({invalid} invalid)" if invalid else ""))
    print("RESULT: " + ("PASS" if killed == len(MUTANTS) else "FAIL"))
    return 0 if killed == len(MUTANTS) else 1


if __name__ == "__main__":
    sys.exit(main())
