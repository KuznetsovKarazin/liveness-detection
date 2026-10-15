"""
Controllo delle mutazioni dei percorsi d'errore di scripts/nuaa_manifests.py (M2, tranche NUAA), offline e
deterministico.

Copia il codice (scripts/, src/tesi_app/, delivery/m2/, requirements.txt) in una cartella temporanea, applica una
mutazione dichiarata alla volta a scripts/nuaa_manifests.py o al caricamento dei modelli locali in
scripts/package_rules.py (coppie cerca/sostituisci che devono comparire esattamente una volta: se un modello non si trova più, il controllo si ferma con un errore), esegue scripts/check_nuaa_manifests.py
sulla copia mutata con cartelle dati, risultati e pesi vuote e conta le mutazioni rilevate. Una mutazione è rilevata
solo se i test arrivano al riepilogo («tests run») con almeno un fallimento o errore; una suite che non parte (import o
sintassi rotti) rende la mutazione non valida. Codice 1 se una mutazione sopravvive o non è valida, o se un modello non
si trova.

Nessuna rete, nessun dato, nessun percorso assoluto nell'uscita. Funziona dalla radice del repository di sviluppo e da
livenesslab/ del pacchetto pubblico.
  python scripts/check_nuaa_mutations.py            # tutte le mutazioni (alcuni minuti)
  python scripts/check_nuaa_mutations.py --list     # solo l'elenco, con la verifica dei modelli
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = "scripts/nuaa_manifests.py"
TESTS = "scripts/check_nuaa_manifests.py"
RULES = "scripts/package_rules.py"      # quarto elemento facoltativo di una mutazione: il file mutato (default TARGET)
# (nome, testo cercato, testo sostitutivo): le garanzie sui percorsi d'errore introdotte il 14/10/2026
MUTANTS = [
    ("build: no invalidation", "    invalidate(out, BUILD_OUTPUTS + OVERLAP_OUTPUTS + REPORT_OUTPUTS)\n", "    pass\n"),
    ("build: aggregate written on failure", '    if not problems:\n        files["nuaa_manifests_aggregate.json"] = agg_text',
     '    if True:\n        files["nuaa_manifests_aggregate.json"] = agg_text'),
    ("build: scan skipped", "    bad = public_problems(agg_text, subject_ids) if not problems else []", "    bad = []"),
    ("check-overlap: no invalidation", "    invalidate(out, OVERLAP_OUTPUTS + REPORT_OUTPUTS)\n", "    pass\n"),
    ("check-overlap: short reports not invalidated", "    invalidate(out, OVERLAP_OUTPUTS + REPORT_OUTPUTS)\n",
     "    invalidate(out, OVERLAP_OUTPUTS)\n"),
    ("preliminary: no FAIL reports",
     '        write_all(out, {"overlap_report.json": json.dumps({**base, "error": str(e)}, indent=1) + "\\n",\n'
     '                        "overlap_report_public.json": json.dumps(base, indent=1) + "\\n"})\n', "        pass\n"),
    ("preliminary: error text in public", '"overlap_report_public.json": json.dumps(base, indent=1)',
     '"overlap_report_public.json": json.dumps({**base, "error": str(e)}, indent=1)'),
    ("rules error not reported", '    doc = guarded("rules", load_json, rules_p)', "    doc = load_json(rules_p)"),
    ("unreadable manifest not reported", '        rows = guarded("integrity", read_csv, p, COLUMNS)', "        rows = read_csv(p, COLUMNS)"),
    ("unreadable manifest: wrong stage", '        rows = guarded("integrity", read_csv, p, COLUMNS)',
     '        rows = guarded("row-validation", read_csv, p, COLUMNS)'),
    ("integrity: exit without report",
     '        raise Preliminary(f"manifests differ from those declared in {relpath(rules_p)}: " + "; ".join(mismatch), "integrity")',
     '        fail("manifests differ from those declared")'),
    ("row validation: exit without report", '            raise Preliminary(f"{relpath(p)}: " + "; ".join(bad_rows), "row-validation")',
     '            fail("rows differ")'),
    ("public scan: public report kept",
     '        write_all(out, {"overlap_report.json": json.dumps({**rep, "passed": False, "stage": "public-scan",',
     '        write_all(out, {"overlap_report_public.json": pub_json, "overlap_report.json": json.dumps({**rep, "passed": False, "stage": "public-scan",'),
    ("report: no invalidation", "    invalidate(out, REPORT_OUTPUTS)                      #", "    pass  #"),
    ("report: accepts a failed check", '    if summ.get("passed") is not True or ov.get("passed") is not True:\n        fail(', "    if False:\n        fail("),
    ("invalidate: keeps temporary files", 'for q in (Path(out) / n, Path(out) / (n + ".tmp")):', "for q in (Path(out) / n,):"),
    ("write_all: no cleanup", '        for n in files:\n            (out / (n + ".tmp")).unlink(missing_ok=True)\n            (out / n).unlink(missing_ok=True)\n        raise',
     "        raise"),
    ("write_all: cleanup only on OSError", "    except BaseException:\n        for n in files:", "    except OSError:\n        for n in files:"),
    ("write_all: direct write", '            (out / (n + ".tmp")).write_text(text, encoding="utf-8")', '            (out / n).write_text(text, encoding="utf-8")'),
    # modelli locali della scansione (scripts/package_rules.py, 14/10/2026)
    ("local patterns: file ignored", "    if not p.is_file():\n        return []\n", "    return []\n", RULES),
    ("local patterns: not added to the rules", "    if local:\n        rules.insert(1,", "    if False:\n        rules.insert(1,", RULES),
    ("local patterns: tool patterns ignored", "    if tool:\n        rules.append(", "    if False:\n        rules.append(", RULES),
    ("local patterns: malformed file accepted", '        raise ValueError(f"{p.name}: \'{key}\' must',
     '        return []\n        raise ValueError(f"{p.name}: \'{key}\' must', RULES),
    ("local patterns: strict export not refused when missing", '    if not p.is_file():\n        return (f"the local',
     '    if False:\n        return (f"the local', RULES),
]


def copy_tree(dst: Path) -> Path:
    """Copia minima del codice in dst/livenesslab (e dei documenti accanto, come nel pacchetto pubblico)."""
    code = dst / "livenesslab"
    for item in ("scripts", "src/tesi_app", "delivery/m2"):
        if (ROOT / item).is_dir():
            shutil.copytree(ROOT / item, code / item, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if (ROOT / "requirements.txt").is_file():
        shutil.copy2(ROOT / "requirements.txt", code / "requirements.txt")
    for n in ("README.md", "naming_schema.md"):          # pacchetto pubblico: documenti nella cartella sopra livenesslab/
        if (ROOT.parent / n).is_file() and not (ROOT / "delivery" / "m2" / n).is_file():
            shutil.copy2(ROOT.parent / n, dst / n)
    return code


def run_tests(code: Path, empty: Path) -> tuple:
    env = {**os.environ, "LIVENESSLAB_DATA_DIR": str(empty / "data"), "LIVENESSLAB_RESULTS_DIR": str(empty / "results"),
           "LIVENESSLAB_WEIGHTS_DIR": str(empty / "weights"), "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("LIVENESSLAB_LOCAL_SCAN_PATTERNS", None)     # la copia non ha il file locale: solo regole generiche e file di prova
    r = subprocess.run([sys.executable, "-I", TESTS], cwd=code, env=env, capture_output=True, text=True, timeout=1800)
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("tests run")), "no test summary")
    return r.returncode, line


def detected(rc: int, line: str) -> str:
    m = re.match(r"tests run: (\d+), failures: (\d+), errors: (\d+)", line)
    if not m:
        return "invalid"
    if int(m.group(2)) + int(m.group(3)) > 0:
        return "killed"
    return "survived" if rc == 0 else "invalid"


def main() -> int:
    ap = argparse.ArgumentParser(description="Controllo delle mutazioni dei percorsi d'errore di nuaa_manifests.py (offline)")
    ap.add_argument("--list", action="store_true", help="elenca le mutazioni e verifica i modelli, senza eseguire i test")
    a = ap.parse_args()
    problems = []
    for n, old, _, target in (m if len(m) == 4 else (*m, TARGET) for m in MUTANTS):
        k = (ROOT / target).read_text(encoding="utf-8").count(old)
        if k != 1:
            problems.append(f"{n}: the pattern occurs {k} times in {target} (expected exactly once)")
    for p in problems:
        print(f"ERROR: {p}", file=sys.stderr)
    if problems:
        print("RESULT: FAIL (a declared mutation no longer matches the code: update the list after review)")
        return 1
    if a.list:
        for m in MUTANTS:
            print(m[0])
        print(f"{len(MUTANTS)} mutations, every pattern found exactly once")
        return 0
    killed = invalid = 0
    with tempfile.TemporaryDirectory(prefix="m2-nuaa-mutations-") as tmp:
        tmp = Path(tmp)
        empty = tmp / "empty"
        rc, line = run_tests(copy_tree(tmp / "base"), empty)
        print(f"unmutated copy: {TESTS} exit {rc} ({line})", flush=True)
        if rc:
            print("RESULT: FAIL (the offline tests do not pass on the unmutated copy)")
            return 1
        for name, old, new, target in (m if len(m) == 4 else (*m, TARGET) for m in MUTANTS):
            work = tmp / "work"
            shutil.rmtree(work, ignore_errors=True)
            code = copy_tree(work)
            f = code / target
            f.write_text(f.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")
            rc, line = run_tests(code, empty)
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
