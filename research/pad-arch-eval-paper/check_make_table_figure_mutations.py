"""
Mutation check di make_table_figure.py: ogni mutante toglie o indebolisce un controllo dichiarato (README, docstring)
in una copia temporanea del pacchetto e del pacchetto dei risultati, poi esegue check_make_table_figure.py su quella
copia; il mutante è ucciso se la suite fallisce. Prima di tutto la copia non mutata deve passare (controllo).

I mutanti R01–R23, N04, N07, N10, N13, Z2, Z6 e V7 sono quelli della revisione indipendente del 21/11/2026,
riportati sul codice attuale (R10: con la creazione esclusiva O_CREAT | O_EXCL il vecchio O_NOFOLLOW era ridondante ed
è stato tolto; il mutante toglie O_EXCL. R20: il doppione del controllo è stato tolto; il mutante toglie l'unico
controllo rimasto); X01–X24, Y01–Y04, W1–W5 e V1 sono dell'autore; N17, V3 e V4 sono equivalenti e sono solo elencati,
con il motivo.
Senza matplotlib i mutanti marcati «needs matplotlib» non sono verificabili e non sono mai contati come uccisi.

  python3 -I -B check_make_table_figure_mutations.py [--jobs N]
"""
import concurrent.futures
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
RESULTS = HERE.parent / "pad-arch-eval-results"


def _inside(p, root):
    p, root = os.path.realpath(p), os.path.realpath(root)
    return p == root or p.startswith(root + os.sep)


if _inside(tempfile.gettempdir(), HERE) or _inside(tempfile.gettempdir(), RESULTS):
    print(f"ERROR: TMPDIR ({tempfile.gettempdir()}) is inside the package or the results package; set it elsewhere")
    print("RESULT: FAIL")
    sys.exit(1)
HAS_MPL = importlib.util.find_spec("matplotlib") is not None

MUTANTS = [
    ("R01", "zero gap counted as change of sign", False,
     "return (o < 0 < e) or (o > 0 > e)", "return (o < 0 <= e) or (o > 0 > e)"),
    ("R02", "ties never counted", False, '"ties": sum(v == 0 for v in vals)}', '"ties": 0}'),
    ("R03", "interval low <= high not checked", False, "    if not low <= high:\n", "    if False:\n"),
    ("R04", "upper bound of valid resamples not checked", False,
     "if not 1 <= valid <= 1000:", "if not 1 <= valid:"),
    ("R05", "value of primary_analysis.rows not compared", False,
     "if type(value) is not float or value != float(want):", "if type(value) is not float:"),
    ("R06", "models deciding every frame bona fide ignored", False, "elif pair == (1, 0):", "elif False:"),
    ("R07", "positive frame counts not checked", False,
     "all(type(n) is int and n > 0 for n in frames)", "all(type(n) is int for n in frames)"),
    ("R08", "conditions.csv: low and high of APCER and BPCER not compared", False,
     'exp[f"low_{m}"] = repr(get(blk, f"interval.{m}.low"))',
     'exp[f"low_{m}"] = r[f"low_{m}"] if m != "acer" else repr(get(blk, f"interval.{m}.low"))'),
    ("R09", "inode re-check in read_regular dropped", False,
     "if not stat.S_ISREG(st2.st_mode) or (st2.st_dev, st2.st_ino) != (st.st_dev, st.st_ino):",
     "if not stat.S_ISREG(st2.st_mode):"),
    ("R10", "output files not created exclusively", False,
     "os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644", "os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644"),
    ("R11", "final newline of SHA256SUMS not checked", False,
     '    if not text.endswith("\\n"):\n', '    if False:\n'),
    ("R12", "plan of checks_before_primary_analysis not checked", False,
     'if not same(cb["plan"][key], {"combinations": 12, "threshold": 0.5}):', "if False:"),
    ("R13", "n_subjects not checked per seed", False,
     '    if not same(get(block, "n_subjects"), N_SUBJECTS):',
     '    if "per_seed" not in path and not same(get(block, "n_subjects"), N_SUBJECTS):'),
    ("R14", "frame identifier hash not checked", False, 'if r["frame_ids_sha256"] != ids[str(fold)]:', "if False:"),
    ("R15", "sign rule not checked per seed", False,
     '    if (get(block, "sign"), get(block, "sign_reason")) != sign_rule(low, high):',
     '    if "per_seed" not in path and (get(block, "sign"), get(block, "sign_reason")) != sign_rule(low, high):'),
    ("R16", "reading rule not checked per seed", False,
     '    if (get(block, "reading"), get(block, "reason")) != contrast_rule(low, high):',
     '    if "per_seed" not in path and (get(block, "reading"), get(block, "reason")) != contrast_rule(low, high):'),
    ("R17", "APCER and BPCER mean gaps not compared", False,
     '        if got != float(exact):\n            raise Refusal(f"results.json: {path}.mean_delta.{m}',
     '        if m == "acer" and got != float(exact):\n            raise Refusal(f"results.json: {path}.mean_delta.{m}'),
    ("R18", "temporary directory left after a failure", False,
     "        shutil.rmtree(tmp, ignore_errors=True)", "        pass"),
    ("R19", "matplotlib defaults not restored (a local matplotlibrc changes the figure)", True,
     "        matplotlib.rcdefaults()\n", ""),
    ("R20", "--against equal to the inputs accepted", False,
     "            if same_file(against, inputs):\n                raise Refusal(f\"the directory to check is the input",
     "            if False:\n                raise Refusal(f\"the directory to check is the input"),
    ("R21", "duplicate line of SHA256SUMS accepted", False,
     "        if m.group(2) in listed:\n", "        if False:\n"),
    ("R22", "field count of CSV rows not checked", False, "        if len(r) != len(header):\n", "        if False:\n"),
    ("R23", "units of the contrast not checked", False,
     '    if not same(get(block, "units"), len(keys)):', "    if False:"),
    ("X01", "a bound at zero excludes zero in the sign rule", False,
     "    if high < 0:\n        return SIGN_BELOW", "    if high <= 0:\n        return SIGN_BELOW"),
    ("X02", "rounding half down", False, "if (q - n) * 2 >= 1:", "if (q - n) * 2 > 1:"),
    ("X03", "pin of SHA256SUMS not checked", False, "if sha256(sums) != PUBLIC_SUMS_SHA256:", "if False:"),
    ("X04", "valid resamples not uniform across blocks accepted", False, "        if len(seen) != 1:", "        if False:"),
    ("X05", "change of sign between states without resolved labels", False,
     "return label_o in resolved and label_e in resolved and label_o != label_e", "return label_o != label_e"),
    ("X06", "values rounding to zero not explained", False, "if x != 0 and points(x) == 0:", "if False:"),
    ("X07", "output inside the results package accepted", False,
     "if root is not None and inside(parent, root):", "if False:"),
    ("X08", "output inside this package accepted", False,
     "if inside(parent, str(HERE)) and not", "if False and not"),
    ("X09", "signals not recorded", False, '        state["received"].append(signum)', "        pass"),
    ("X10", "PNG structure not checked under --text-only", False,
     "ok = png_structure_ok(committed) if", "ok = True if"),
    ("X11", "PDF structure not checked under --text-only", False,
     'else pdf_structure_ok(committed)', "else True"),
    ("X12", "loser of two runs gets the generic message", False,
     "            os.rename(tmp, out)\n        except OSError as e:\n            if e.errno in RACE_ERRNOS:",
     "            os.rename(tmp, out)\n        except OSError as e:\n            if False:"),
    ("X13", "repeated options accepted", False, "        if n > 1:\n", "        if False:\n"),
    ("X14", "only the primary rows of units.csv validated", False,
     "        u = check_unit_row(r, frames, ids)\n",
     "        u = check_unit_row(r, frames, ids) if (r[\"patience\"], r[\"threshold\"]) == (str(PATIENCE), THRESHOLD)"
     " else None\n"),
    ("X15", "row count of units.csv not checked", False, "    if len(rows) != n_all:", "    if False:"),
    ("X16", "one_class counts not checked", False,
     "if type(k) is not int or type(n) is not int or not 0 <= k <= n:", "if False:"),
    ("X17", "dev EER thresholds not checked", False,
     "if not (math.isfinite(v) and 0 <= v <= 1 and t == repr(v)):", "if False:"),
    ("X18", "malformed structures end in a traceback", False,
     "except (AttributeError, KeyError, TypeError, ValueError, IndexError) as e:", "except ZeroDivisionError as e:"),
    ("X19", "unit points never stacked", False,
     "while lvl in last and x - last[lvl] < LEVEL_GAP_POINTS:", "while False:"),
    ("X20", "sign of values rounding to zero dropped", False,
     "    if x < 0:\n        return minus + s", "    if x < 0 and s != \"0.00\":\n        return minus + s"),
    ("X21", "top level of results.json not checked", False, "    if not isinstance(res, dict):\n", "    if False:\n"),
    ("X22", "extra seeds in per_seed accepted", False,
     "if not isinstance(per_seed, dict) or sorted(per_seed) != [str(s) for s in SEEDS]:",
     "if not isinstance(per_seed, dict):"),
    ("X23", "SIGINT outside the recorders not decided from the state", False,
     "        if state is not None:\n            return finish(out, state, None, extra=int(signal.SIGINT))",
     "        if False:\n            return finish(out, state, None, extra=int(signal.SIGINT))"),
    ("X24", "existing output directory accepted", False,
     '    if os.path.lexists(out):\n        raise Refusal(f"the output directory already exists',
     '    if False:\n        raise Refusal(f"the output directory already exists'),
    ("N04", "output directory left in mode 0700", False, "        os.chmod(tmp, 0o755)\n", "        pass\n"),
    ("N07", "signal handlers not restored", False,
     "    for s, h in old.items():\n        signal.signal(s, h)", "    for s, h in []:\n        signal.signal(s, h)"),
    ("N10", "bytes after the PNG IEND chunk accepted", False, "return end == len(b)", "return True"),
    ("N13", "temporary directory in the system temporary directory, not next to the output", False,
     'tempfile.mkdtemp(prefix=".mtf-", dir=parent)', 'tempfile.mkdtemp(prefix=".mtf-")'),
    ("Y01", "output name not reserved before the rename (an empty directory is replaced)", False,
     "            os.mkdir(out, 0o755)\n", "            pass\n"),
    ("Y02", "no check of the signals between the reservation and the rename", False,
     '        state["reserved_id"] = _identity(out)\n        _poll(state)\n',
     '        state["reserved_id"] = _identity(out)\n'),
    ("Y03", "mean outside its copied interval accepted", False, "if not low <= mean_value <= high:", "if False:"),
    ("Y04", "ENOTDIR and EISDIR not treated as the race", False,
     "RACE_ERRNOS = (errno.EEXIST, errno.ENOTEMPTY, errno.ENOTDIR, errno.EISDIR)",
     "RACE_ERRNOS = (errno.EEXIST, errno.ENOTEMPTY)"),
    ("Z2", "own empty reservation not removed after an error", False, "            os.rmdir(out)\n",
     "            pass\n"),
    ("Z6", "SIGINT not recorded", False, "SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)",
     "SIGNALS = (signal.SIGTERM, signal.SIGHUP)"),
    ("W1", "reservation not recognised by inode before the rename", False,
     'if _identity(out) != state["reserved_id"] or os.listdir(out):', "if False:"),
    ("W2", "reservation removed by name, not by identity", False,
     'if state.get("reserved_id") is not None and _identity(out) == state["reserved_id"]:',
     'if state.get("reserved_id") is not None:'),
    ("W3", "completion decided without the inode of the output", False,
     'return state.get("tmp_id") is not None and _identity(out) == state["tmp_id"]',
     'return state.get("tmp_id") is not None'),
    ("W4", "signals not checked before the reservation", False,
     "        os.chmod(tmp, 0o755)\n        _poll(state)\n", "        os.chmod(tmp, 0o755)\n"),
    ("W5", "signal received after completion not reported", False,
     'note = f"; signal {sigs[0]} received, handled after the outputs were complete" if sigs else ""', 'note = ""'),
    ("V1", "no-signal failure reported as an interruption", False, "    elif failure is None:\n",
     "    elif failure is None and False:\n"),
    ("V7", "temporary directory removed by path, not by identity", False,
     'if tmp is not None and state.get("tmp_id") is not None and _identity(tmp) == state["tmp_id"]:',
     "if tmp is not None:"),
]

# mutanti equivalenti: elencati, non eseguiti né contati
EQUIVALENT = [
    ("N17", "results package recognised without the 'runs' level", 'p.parent.parent.name == "runs"', "True",
     "equivalent: the package root is then the parent of the run folder's parent; the refusal only widens and no "
     "input layout accepted by the plan can tell the two apart"),
    ("V3", "emptiness of the reservation not checked before the rename", " or os.listdir(out):", ":",
     "equivalent: a rename onto a non-empty reservation fails with ENOTEMPTY and gives the same message, exit code and "
     "cleanup"),
    ("V4", "signals not checked after each file", "            _poll(state)\n        os.chmod", "        os.chmod",
     "equivalent: a signal recorded during the writing is seen at the check before the reservation, with the same "
     "message, exit code and cleanup, only later"),
]


def copy_layout(dst):
    shutil.copytree(HERE, dst / "pad-arch-eval-paper", ignore=shutil.ignore_patterns("test_logs"))
    shutil.copytree(RESULTS, dst / "pad-arch-eval-results")


def run_suite(root):
    r = subprocess.run([sys.executable, "-I", "-B", str(root / "pad-arch-eval-paper" / "check_make_table_figure.py")],
                       capture_output=True, text=True, timeout=900)
    last = [ln for ln in r.stdout.splitlines() if ln.startswith("tests run")]
    return r.returncode, (last[-1] if last else r.stderr.strip()[-200:])


def one(base, mutant):
    mid, desc, mpl, old, new = mutant
    root = base / mid
    copy_layout(root)
    f = root / "pad-arch-eval-paper" / "make_table_figure.py"
    src = f.read_text()
    if src.count(old) != 1:
        return mid, desc, "PATTERN NOT FOUND", ""
    f.write_text(src.replace(old, new))
    if mpl and not HAS_MPL:
        return mid, desc, "not verifiable without matplotlib", ""
    code, line = run_suite(root)
    shutil.rmtree(root, ignore_errors=True)
    return mid, desc, "killed" if code != 0 else "SURVIVED", line


def main():
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
    base = Path(tempfile.mkdtemp(prefix="mtf-mut-"))
    try:
        copy_layout(base / "control")
        code, line = run_suite(base / "control")
        print(f"control (no mutation): {'PASS' if code == 0 else 'FAIL'} ({line})")
        if code != 0:
            print("RESULT: FAIL (the unmutated copy does not pass)")
            return 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(lambda m: one(base, m), MUTANTS))
        src = (HERE / "make_table_figure.py").read_text()
        for mid, desc, old, new, why in EQUIVALENT:
            assert src.count(old) == 1, mid
            print(f"{mid} equivalent (not run, not counted): {desc}; {why}")
        for mid, desc, verdict, line in results:
            print(f"{mid} {verdict}: {desc}" + (f" ({line})" if line else ""))
        killed = sum(v == "killed" for _, _, v, _ in results)
        nv = sum(v.startswith("not verifiable") for _, _, v, _ in results)
        print(f"mutations killed: {killed}/{len(MUTANTS)}")
        if nv:
            print(f"mutations not verifiable without matplotlib (never counted as killed): {nv}")
        ok = killed == len(MUTANTS)
        print("RESULT: PASS" if ok else ("RESULT: INCOMPLETE" if killed + nv == len(MUTANTS) else "RESULT: FAIL"))
        return 0 if ok else (2 if killed + nv == len(MUTANTS) else 1)
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
