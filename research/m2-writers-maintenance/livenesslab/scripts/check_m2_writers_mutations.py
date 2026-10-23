"""
Controllo delle mutazioni della manutenzione dei writer M2 (22/10/2026), offline e deterministico.

Copia il codice (scripts/, src/tesi_app/, delivery/m2/, requirements.txt) in una cartella temporanea con la disposizione
del repository del gruppo (<tmp>/research/m2-writers-maintenance/livenesslab) e, accanto, le sole copie accettate dei
tre file dichiarati in m2_writers.MAINTENANCE (<tmp>/research/<pacchetto>/livenesslab/scripts/...), lette e verificate
per SHA-256 con m2_writers.accepted_text; applica una mutazione dichiarata alla volta (coppie cerca/sostituisci che
devono comparire esattamente una volta: se un modello non si trova più, il controllo si ferma con un errore) e esegue
scripts/check_m2_writers.py sulla copia mutata con cartelle dati, risultati e pesi vuote. Una mutazione è rilevata solo
se la suite arriva al riepilogo («tests run») con almeno un fallimento o errore; una suite che non parte la rende non
valida. Nella copia mutata lo SHA-256 dichiarato del file mutato (m2_writers.MAINTENANCE["current"]) è rifissato su quello
del file mutato, così una mutazione è rilevata solo dal comportamento; una mutazione di controllo (solo un commento) deve
sopravvivere ed è riportata a parte. La copia non mutata deve passare senza test saltati (altrimenti alcune mutazioni non sarebbero provate). Codice 1
se una mutazione sopravvive o non è valida, se un modello non si trova o se mancano le copie accettate.

Nessuna rete, nessun dato, nessun percorso assoluto nell'uscita. Funziona dalla radice del repository di sviluppo e da
livenesslab/ del pacchetto.
  python scripts/check_m2_writers_mutations.py            # tutte le mutazioni (alcuni minuti)
  python scripts/check_m2_writers_mutations.py --list     # solo l'elenco, con la verifica dei modelli
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import m2_writers as MW  # noqa: E402

TESTS = "scripts/check_m2_writers.py"
W, R, N, E = "scripts/m2_writers.py", "scripts/rgb_manifests.py", "scripts/nuaa_manifests.py", "scripts/export_m2_rgb.py"
# (nome, file mutato, testo cercato, testo sostitutivo)
MUTANTS = [
    # validazione della cartella di uscita (m2_writers.output_problem)
    ("output folder: symbolic link accepted", W, "        if w.is_symlink():\n            return f\"{w.name} is a symbolic link", "        if False:\n            return f\"{w.name} is a symbolic link"),
    ("output folder: file accepted", W, "        if w.exists() and not w.is_dir():\n", "        if False:\n"),
    ("output folder: work tree with the accepted packages not checked", W,
     "        if pg.within(out, g) or pg.contains(out, g):\n            return f\"{out} is inside, or contains", "        if False:\n            return f\"{out} is inside, or contains"),
    ("output folder: work tree containing the output accepted", W, "        if pg.within(out, g) or pg.contains(out, g):\n", "        if pg.within(out, g):\n"),
    ("outer roots: git work tree ignored", W, "        if top is not None and not pg.same(top, root):\n            out.append(top)", "        if False:\n            out.append(top)"),
    ("outer roots: research/ layout ignored", W, '        if (x / "research").is_dir() and', '        if False and (x / "research").is_dir() and'),
    ("output folder: path_guard skipped", W, "    msg = pg.deletion_problem(out, root, protected, allowed)\n", '    msg = ""\n'),
    ("output folder: protected ancestors accepted", W, "        if pg.within(out, q) or (Path(q).exists() and pg.contains(out, q)):\n", "        if pg.within(out, q):\n"),
    ("inputs: output folder inside an input accepted", W, "        if pg.within(out, p):\n", "        if False:\n"),
    ("inputs: input among the outputs accepted", W, "            if hit := touched_by(q, out, outputs, dirs):\n", '            if hit := "":\n'),
    ("inputs: fixed temporaries not among the outputs", W, 'return entry == n or entry == n + ".tmp" or (', "return entry == n or ("),
    ("other tranche: outputs not refused", W, "    if found:\n        return f\"output folder {Path(out).name} holds", "    if False:\n        return f\"output folder {Path(out).name} holds"),
    # invalidazione e scrittura
    ("invalidate: link to a folder followed", W, "            if q.is_symlink():\n                q.unlink()\n                removed.append(q.name)\n            elif q.is_dir():",
     "            if q.is_symlink() and not q.is_dir():\n                q.unlink()\n                removed.append(q.name)\n            elif q.is_dir():"),
    ("invalidate: folder with an output name not refused", W, "            elif q.is_dir():\n                _fail(", "            elif False:\n                _fail("),
    ("invalidate: unique temporaries kept", W, ' or (entry.startswith(f".{n}.") and entry.endswith(".tmp"))', ""),
    ("write_all: output link not refused", W, "            if q.is_symlink() or (q.exists() and not q.is_file()):\n                _fail(f\"{n} in the output",
     "            if False:\n                _fail(f\"{n} in the output"),
    ("write_all: dangling output link left on failure", W, "            if q.is_symlink() or q.is_file():           #", "            if q.is_file():           #"),
    ("write_all: output folder link not refused", W, "    if out.is_symlink():\n        _fail(f\"{out.name} is a symbolic link", "    if False:\n        _fail(f\"{out.name} is a symbolic link"),
    ("write_all: permissions of write_text not kept", W, "                os.fchmod(fd, 0o666 & ~_UMASK)", "                pass"),
    ("write_all: fixed temporary name", W, '            fd, tp = tempfile.mkstemp(prefix=f".{n}.", suffix=".tmp", dir=out)\n',
     '            fd, tp = os.open(out / (n + ".tmp"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666), str(out / (n + ".tmp"))\n'),
    # dichiarazione della manutenzione
    ("maintenance: AST not compared", W, '    return MAINT_LABEL if same else ""', "    return MAINT_LABEL"),
    ("maintenance: exact SHA-256 not required", W, '    if h != m["current"] or accepted is None', "    if accepted is None"),
    ("maintenance: accepted copy not verified", W, ' or sha256_bytes(accepted.encode("utf-8")) != m["accepted"]:', ":"),
    ("maintenance: every node skipped", W, "        if ast.unparse(n) in added:\n            continue", "        if True:\n            continue"),
    # chiamate nei moduli
    ("rgb build: output folder not validated", R, "    if msg := outputs_problem(out, ds, build_outputs(ds),", "    if False and outputs_problem(out, ds, build_outputs(ds),"),
    ("rgb build: manifests folder not validated", R, '"the rules file": rules_path(ds)}, [out / "manifests"]):', '"the rules file": rules_path(ds)}, []):'),
    ("rgb check-overlap: inputs not validated", R, '{"--rules": rules_p, "--manifests-dir": man_dir}', "{}"),
    ("rgb report: template not validated", R, '{"--template": Path(a.template), **{', "{**{"),
    ("nuaa build: output folder not validated", N, "    if msg := outputs_problem(out, BUILD_OUTPUTS", "    if False and outputs_problem(out, BUILD_OUTPUTS"),
    ("nuaa build: manifests folder not validated", N, '"the rules file": RULES}, [out / "manifests"]):', '"the rules file": RULES}, []):'),
    ("nuaa check-overlap: inputs not validated", N, '{"--rules": rules_p, "--manifests-dir": Path(a.manifests_dir)}', "{}"),
    ("nuaa report: template not validated", N, '{"--template": Path(a.template), **{', "{**{"),
    ("nuaa: outputs of the RGB tranche not checked", N, '    msg = mw.foreign_outputs_problem(out, RGB_OUTPUTS, "the CASIA-FASD/CelebA-Spoof tranche")', '    msg = ""'),
    # export_m2_rgb.py
    ("export: work tree with the accepted packages not protected", E,
     "        if path_guard.within(path, g) or path_guard.contains(path, g):\n", "        if False:\n"),
    ("export: moved data folders not protected", E, "        if path_guard.within(path, q) or (Path(q).exists() and path_guard.contains(path, q)):\n", "        if False:\n"),
    ("export: maintenance state of nuaa_manifests.py ignored", E,
     "    return mw.maintenance_state(f, data, accepted if accepted is not None else mw.accepted_text(f, ROOT))", '    return ""'),
    ("export: any published SHA-256 accepted for the maintenance", E, '    if f not in mw.MAINTENANCE or h != mw.MAINTENANCE[f]["accepted"]:', "    if f not in mw.MAINTENANCE:"),
    # revisione indipendente del 22/10: ogni cartella protetta spostata, controllo dell'identità in main, passo della suite
    ("export: moved data folder not protected", E, "    for q in (RM.DATA, RM.WEIGHTS, ", "    for q in (RM.WEIGHTS, "),
    ("export: moved weights folder not protected", E, "(RM.DATA, RM.WEIGHTS, RM.RESULTS", "(RM.DATA, RM.RESULTS"),
    ("export: moved results/c1 not protected", E, ' RM.RESULTS / "c1", RM.RESULTS / "eval",', ' RM.RESULTS / "eval",'),
    ("export: moved results/eval not protected", E, ' RM.RESULTS / "eval", RM.RESULTS / "m1",', ' RM.RESULTS / "m1",'),
    ("export: moved results/m1 not protected", E, ' RM.RESULTS / "m1", RM.RESULTS / "m2"):', ' RM.RESULTS / "m2"):'),
    ("export: moved results/m2 not protected", E, ', RM.RESULTS / "m2"):\n', '):\n'),
    ("export: differences of the shared files not reported", E,
     '        if h is not None and not state:\n            failed.append(f"{f} differs', '        if False:\n            failed.append(f"{f} differs'),
    ("export: reference failures ignored by main", E, "    failed += ref_failed\n", "    pass\n"),
    ("export: writer suite not run", E, ',\n              ("check_m2_writers.txt", ["scripts/check_m2_writers.py"])]', "]"),
    # revisione del 22/10 (seconda): fallimenti del self-test ed errori imprevisti, anche dopo la scrittura delle uscite
    ('rgb self-test: fail() and interruptions not handled', R, '        except BaseException as e:                    # anche fail()', '        except Exception as e:                    # anche fail()'),
    ('rgb self-test: reports not marked as failed', R, '            self_test_failure(out, rep, pub, err, state, ds)\n', ''),
    ('rgb self-test failure: markdown reports kept', R, '    mw.invalidate_each(out, ["overlap_report.md"', '    [] and mw.invalidate_each(out, ["overlap_report.md"'),
    ('rgb self-test failure: public report left passing', R, '"overlap_report_public.json": json.dumps({**pub, "passed": False, "stage": "self-test"}, indent=1)', '"overlap_report_public.json": json.dumps(pub, indent=1)'),
    ('rgb self-test failure: no self-test report on errors', R, '    if error:\n        files["overlap_self_test.json"]', '    if False:\n        files["overlap_self_test.json"]'),
    ('rgb self-test: missing manifests not refused', R, '    if missing:                                         # rifiuto esplicito (22/10/2026', '    if False:  #'),
    ('rgb main: interruptions not handled', R, '    except BaseException as e:                          # errore imprevisto', '    except Exception as e:                          # errore imprevisto'),
    ('rgb unexpected error: outputs not removed', R, '            if admitted:                                # un nome alla volta', '            if False:                                # un nome alla volta'),
    ('rgb unexpected error: no FAIL reports', R, '            if a.cmd == "check-overlap" and admitted:', '            if False:'),
    ('nuaa self-test: fail() and interruptions not handled', N, '        except BaseException as e:                    # anche fail()', '        except Exception as e:                    # anche fail()'),
    ('nuaa self-test: reports not marked as failed', N, '            self_test_failure(out, rep, pub, err)\n', ''),
    ('nuaa self-test failure: markdown reports kept', N, '    mw.invalidate_each(out, ["overlap_report.md"', '    [] and mw.invalidate_each(out, ["overlap_report.md"'),
    ('nuaa self-test failure: public report left passing', N, '"overlap_report_public.json": json.dumps({**pub, "passed": False, "stage": "self-test"}, indent=1)', '"overlap_report_public.json": json.dumps(pub, indent=1)'),
    ('nuaa self-test failure: no self-test report on errors', N, '    if error:\n        files["overlap_self_test.json"]', '    if False:\n        files["overlap_self_test.json"]'),
    ('nuaa self-test: missing manifests not refused', N, '    if missing:                                         # rifiuto esplicito, gestito', '    if False:  #'),
    ('nuaa main: interruptions not handled', N, '    except BaseException as e:                          # errore imprevisto', '    except Exception as e:                          # errore imprevisto'),
    ('nuaa unexpected error: outputs not removed', N, '            if admitted:                                # un nome alla volta', '            if False:                                # un nome alla volta'),
    ('nuaa unexpected error: no FAIL reports', N, '            if a.cmd == "check-overlap" and admitted:', '            if False:'),
    # revisione del 22/10 (terza): fail() e Ctrl-C nel self-test, validazione prima dell'azione, tutte le uscite a valle
    ('rgb self-test: interruption not re-raised', R, '                raise interrupted\n', '                pass\n'),
    ('rgb unexpected error: interruption not re-raised', R, '        raise e\n    fail(f"{a.cmd} FAIL: unexpected error {err}")', '        pass\n    fail(f"{a.cmd} FAIL: unexpected error {err}")'),
    ('rgb main: fail() treated as unexpected', R, '        if isinstance(e, SystemExit):                   # fail()', '        if False:                   # fail()'),
    ('rgb unexpected error: acts before validation', R, '            admitted = bool(VALIDATED) and VALIDATED[0] == out', '            admitted = True'),
    ('rgb validated folder never recorded', R, '        VALIDATED[:] = [Path(out)]', '        pass'),
    ('nuaa self-test: interruption not re-raised', N, '                raise interrupted\n', '                pass\n'),
    ('nuaa unexpected error: interruption not re-raised', N, '        raise e\n    fail(f"{a.cmd} FAIL: unexpected error {err}")', '        pass\n    fail(f"{a.cmd} FAIL: unexpected error {err}")'),
    ('nuaa main: fail() treated as unexpected', N, '        if isinstance(e, SystemExit):                   # fail()', '        if False:                   # fail()'),
    ('nuaa unexpected error: acts before validation', N, '            admitted = bool(VALIDATED) and VALIDATED[0] == out', '            admitted = True'),
    ('nuaa validated folder never recorded', N, '        VALIDATED[:] = [Path(out)]', '        pass'),
    ('rgb unexpected error: build leaves downstream outputs', R, '"build": lambda ds: build_outputs(ds)', '"build": lambda ds: [f"{ds}_manifests_summary.json", f"{ds}_manifests_aggregate.json"]'),
    ('rgb unexpected error: check-overlap leaves short reports', R, '"check-overlap": lambda ds: overlap_outputs() + report_outputs(ds)', '"check-overlap": lambda ds: overlap_outputs()'),
    ('rgb unexpected error: report outputs kept', R, '"report": lambda ds: report_outputs(ds)}', '"report": lambda ds: []}'),
    ('nuaa unexpected error: build leaves downstream outputs', N, '"build": BUILD_OUTPUTS + OVERLAP_OUTPUTS + REPORT_OUTPUTS, "check-overlap"', '"build": BUILD_OUTPUTS, "check-overlap"'),
    ('nuaa unexpected error: check-overlap leaves short reports', N, '"check-overlap": OVERLAP_OUTPUTS + REPORT_OUTPUTS,\n', '"check-overlap": OVERLAP_OUTPUTS,\n'),
    ('nuaa unexpected error: report outputs kept', N, '             "report": REPORT_OUTPUTS}.get(a.cmd)', '             "report": ()}.get(a.cmd)'),
    # revisione del 22/10 (quarta): doppio guasto nella pulizia (un'uscita che non si può togliere)
    ("clean-up: JSON that cannot be removed not overwritten with a FAIL", W, '        if n.endswith(".json"):\n            try:\n                write_all(out, {n: json.dumps({**failed_doc',
     '        if False:\n            try:\n                write_all(out, {n: json.dumps({**failed_doc'),
    ("clean-up: output left behind not named", W, '        print(f"ERROR: could not invalidate {n} ({why}): left behind", file=sys.stderr, flush=True)\n', ""),
    ("rgb unexpected error: plain invalidate (double fault silent)", R, "                mw.invalidate_each(out, names(a.dataset), None, mw.CLEANUP_DOC)", "                mw.invalidate(out, names(a.dataset))"),
    ("nuaa unexpected error: plain invalidate (double fault silent)", N, "                mw.invalidate_each(out, names, None, mw.CLEANUP_DOC)", "                mw.invalidate(out, names)"),
    # revisione del docente del 23/10: ingressi con i nomi della pulizia, verifica preliminare, invalidazione completata
    ("inputs: resolved path not checked", W, "    for cand in {Path(os.path.abspath(p)), Path(p).resolve()}:", "    for cand in {Path(os.path.abspath(p))}:"),
    ("preflight: non-regular objects accepted", W, "        if not (stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode)):", "        if False:"),
    ("preflight: FIFO accepted", W, "        if not (stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode)):", "        if not (stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode) or stat.S_ISFIFO(st.st_mode)):"),
    ("preflight: entries of the folders written by build not checked", W, " + [e for d in map(Path, dirs) if d.is_dir() for e in sorted(d.iterdir())]", ""),
    ("invalidation stopped midway: not completed", W, "        left = invalidate_each(out, names, None, failed_doc or CLEANUP_DOC)\n", "        left = []\n"),
    ("invalidation stopped midway: no explicit refusal", W, '        _fail(f"the invalidation of the previous outputs stopped', '        raise SystemExit(1)\n        _fail(f"the invalidation of the previous outputs stopped'),
    ("rgb invalidate: stopped invalidation not completed", R, "    return mw.invalidate_all(out, names)", "    return mw.invalidate(out, names)"),
    ("nuaa invalidate: stopped invalidation not completed", N, "    return mw.invalidate_all(out, names)", "    return mw.invalidate(out, names)"),
    ("export: symbolic-link folders accepted", E, "        if Path(p).is_symlink():\n            return f\"{what} {Path(p).name} is a symbolic link", "        if False:\n            return f\"{what} {Path(p).name} is a symbolic link"),
    ("export: --uploaded-sums inside a deleted folder accepted", E, "            if path_guard.within(uploaded_sums, p) or path_guard.within(p, uploaded_sums):", "            if False:"),
    ("export: paths check not called", E, "    if msg := export_paths_problem(", '    if msg := "" and export_paths_problem('),
    # revisione avversaria del 23/10: grafie e alias, voci delle cartelle d'ingresso, uscite che non si possono togliere
    ("inputs: identity with the entries the clean-up removes not checked", W, "            if i in removed:", "            if False:"),
    ("inputs: names compared with case", W, "owned_name(cand.name.casefold(), n.casefold())", "owned_name(cand.name, n)"),
    ("inputs: entries of input folders not checked", W, '[(label, p)] + [(f"{label} entry", e) for e in entries_of(p)]', "[(label, p)]"),
    ("preflight: folder that does not accept writes accepted", W, "        if d.is_dir() and not os.access(d, os.W_OK | os.X_OK):", "        if False:"),
    ("preflight: immutable or append-only outputs accepted", W, 'getattr(st, "st_flags", 0) & LOCK_FLAGS or (', "False or ("),
    ("preflight: append-only flags ignored", W, ' | getattr(stat, "UF_APPEND", 0) | getattr(stat, "SF_APPEND", 0)', ""),
    ("invalidation stopped midway: inaccurate message", W, '"the other outputs were removed or marked FAIL" if not left else', '"the other outputs were removed or marked FAIL" if True else'),
    ("rgb report: files read in the output folder not checked", R, '"overlap_self_test.json")},', '"overlap_self_test.json") if False},'),
    ("nuaa report: files read in the output folder not checked", N, '"overlap_self_test.json")}}):', '"overlap_self_test.json") if False}}):'),
]
# mutazione di controllo: solo un commento in un file con lo SHA-256 fissato; con lo SHA-256 rifissato nella copia deve
# SOPRAVVIVERE (prova che le mutazioni sono rilevate dal comportamento e non dallo SHA-256 dichiarato)
CONTROL = [("control: comment only in rgb_manifests.py", R, "    mw.write_all(out, files)\n", "    mw.write_all(out, files)  # control mutant: comment only\n")]


def copy_tree(dst: Path) -> Path:
    """Codice in dst/research/m2-writers-maintenance/livenesslab e copie accettate verificate nei pacchetti accanto."""
    code = dst / "research" / "m2-writers-maintenance" / "livenesslab"
    for item in ("scripts", "src/tesi_app", "delivery/m2"):
        if (ROOT / item).is_dir():
            shutil.copytree(ROOT / item, code / item, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if (ROOT / "requirements.txt").is_file():
        shutil.copy2(ROOT / "requirements.txt", code / "requirements.txt")
    for rel, m in MW.MAINTENANCE.items():
        p = dst / "research" / m["package"] / "livenesslab" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(ACCEPTED[rel], encoding="utf-8")
    return code


def run_tests(code: Path, empty: Path) -> tuple:
    env = {**os.environ, "LIVENESSLAB_DATA_DIR": str(empty / "data"), "LIVENESSLAB_RESULTS_DIR": str(empty / "results"),
           "LIVENESSLAB_WEIGHTS_DIR": str(empty / "weights"), "PYTHONDONTWRITEBYTECODE": "1"}
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


ACCEPTED = {}


def repin(code: Path, target: str) -> None:
    """Nella copia mutata lo SHA-256 dichiarato del file mutato (MAINTENANCE["current"]) diventa quello del file mutato:
    una mutazione è rilevata solo dai test di comportamento, mai dal solo confronto dello SHA-256."""
    if target not in MW.MAINTENANCE:
        return
    w = code / "scripts" / "m2_writers.py"
    old = MW.MAINTENANCE[target]["current"]
    text = w.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"ERROR: pinned SHA-256 of {target} not found exactly once in the copy of m2_writers.py")
    w.write_text(text.replace(old, MW.sha256_bytes((code / target).read_bytes())), encoding="utf-8")


def one(tmp: Path, i: int, name: str, target: str, old: str, new: str) -> tuple:
    work = tmp / f"work{i}"
    code = copy_tree(work)
    f = code / target
    f.write_text(f.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")
    repin(code, target)
    rc, line = run_tests(code, tmp / f"empty{i}")
    shutil.rmtree(work, ignore_errors=True)
    return name, detected(rc, line), line, rc


def main() -> int:
    ap = argparse.ArgumentParser(description="Controllo delle mutazioni della manutenzione dei writer M2 (offline)")
    ap.add_argument("--list", action="store_true", help="elenca le mutazioni e verifica i modelli, senza eseguire i test")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1), help="mutazioni eseguite in parallelo")
    a = ap.parse_args()
    problems = [f"{n}: the pattern occurs {k} times in {t} (expected exactly once)"
                for n, t, old, _ in MUTANTS + CONTROL if (k := (ROOT / t).read_text(encoding="utf-8").count(old)) != 1]
    for rel in MW.MAINTENANCE:
        ACCEPTED[rel] = MW.accepted_text(rel, ROOT)
        if ACCEPTED[rel] is None:
            problems.append(f"accepted copy of {rel} not found (../{MW.MAINTENANCE[rel]['package']}, export/research or the development commit)")
    for p in problems:
        print(f"ERROR: {p}", file=sys.stderr)
    if problems:
        print("RESULT: FAIL (a declared mutation no longer matches the code, or an accepted copy is missing)")
        return 1
    if a.list:
        for n, t, _, _ in MUTANTS:
            print(f"{n}  [{t}]")
        for n, t, _, _ in CONTROL:
            print(f"{n}  [{t}] (control, must survive)")
        print(f"{len(MUTANTS)} mutations and {len(CONTROL)} control mutation, every pattern found exactly once")
        return 0
    killed = invalid = 0
    with tempfile.TemporaryDirectory(prefix="m2-writers-mutations-") as tmp:
        tmp = Path(tmp)
        rc, line = run_tests(copy_tree(tmp / "base"), tmp / "empty")
        print(f"unmutated copy: {TESTS} exit {rc} ({line})", flush=True)
        if rc or not line.endswith("skipped: 0"):
            print("RESULT: FAIL (the tests do not pass, or skip, on the unmutated copy)")
            return 1
        with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
            futs = [ex.submit(one, tmp, i, n, t, old, new) for i, (n, t, old, new) in enumerate(MUTANTS)]
            cfuts = [ex.submit(one, tmp, len(MUTANTS) + i, n, t, old, new) for i, (n, t, old, new) in enumerate(CONTROL)]
            survived_controls = 0
            for fu in cfuts:
                name, verdict, line, rc = fu.result()
                survived_controls += verdict == "survived"
                print(f"CONTROL   {verdict.upper():<9} {name} ({line})", flush=True)
            for fu in futs:
                name, verdict, line, rc = fu.result()
                killed += verdict == "killed"
                invalid += verdict == "invalid"
                print(f"{verdict.upper():<9} {name} ({line})", flush=True)
                if verdict == "invalid":
                    print(f"ERROR: invalid mutant {name!r}: the test suite did not run to a summary with failures (exit {rc})", file=sys.stderr)
    print(f"mutations killed: {killed}/{len(MUTANTS)}" + (f" ({invalid} invalid)" if invalid else ""))
    print(f"control mutations survived: {survived_controls}/{len(CONTROL)}")
    ok = killed == len(MUTANTS) and survived_controls == len(CONTROL)
    print("RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
