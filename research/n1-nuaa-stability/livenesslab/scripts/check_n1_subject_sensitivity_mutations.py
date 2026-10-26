"""
Controllo delle mutazioni di scripts/n1_subject_sensitivity.py (N1, parte 1), offline e deterministico.

Copia il codice (scripts/n1_subject_sensitivity.py, scripts/check_n1_subject_sensitivity.py, scripts/m2_writers.py,
scripts/path_guard.py e la configurazione delivery/n1/subject_sensitivity/) in una cartella temporanea, applica una
mutazione dichiarata alla volta (coppia cerca/sostituisci che deve comparire esattamente una volta: se un modello non si
trova più, il controllo si ferma con un errore) ed esegue scripts/check_n1_subject_sensitivity.py sulla copia mutata,
con gli ingressi veri in sola lettura (N1_INPUT_ROOT = radice di questo repository). Una mutazione è rilevata solo se la
suite arriva al riepilogo («tests run») con almeno un fallimento o errore, cioè dal comportamento; una suite che non
parte la rende non valida. Una mutazione di controllo (solo un commento) deve sopravvivere.

Test saltati: la copia non mutata deve passare e può saltare test solo per un ingresso mancante dichiarato
(MISSING_INPUT di scripts/check_n1_subject_sensitivity.py, cioè le cartelle riservate dei run C1 e M1, assenti nella
disposizione del pacchetto); i salti sono riportati. Una mutazione conta come rilevata solo per fallimenti o errori,
mai per un salto: se una copia mutata salta test diversi da quelli della copia non mutata e non fallisce, è non valida.
Codice 1 se una mutazione sopravvive o non è valida, se un salto non è un ingresso mancante dichiarato o se un modello
non si trova.

  .venv/bin/python scripts/check_n1_subject_sensitivity_mutations.py            # tutte le mutazioni
  .venv/bin/python scripts/check_n1_subject_sensitivity_mutations.py --list     # solo l'elenco, con la verifica dei modelli
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
from check_n1_subject_sensitivity import MISSING_INPUT  # noqa: E402

TESTS = "scripts/check_n1_subject_sensitivity.py"
S = "scripts/n1_subject_sensitivity.py"
COPY = ("scripts/n1_subject_sensitivity.py", "scripts/check_n1_subject_sensitivity.py", "scripts/m2_writers.py", "scripts/path_guard.py",
        "delivery/n1/subject_sensitivity/n1_subject_sensitivity.json")
# (nome, file mutato, testo cercato, testo sostitutivo)
MUTANTS = [
    # regola di decisione e metriche
    ("tie at the threshold counted as attack (FA)", S, 'fa = sum(1 for r in rows if r[2] == 1 and not r[3] > t)', 'fa = sum(1 for r in rows if r[2] == 1 and r[3] < t)'),
    ("tie at the threshold counted as attack (FR)", S, 'fr = sum(1 for r in rows if r[2] == 0 and r[3] > t)', 'fr = sum(1 for r in rows if r[2] == 0 and r[3] >= t)'),
    ("absent attacks give APCER 0", S, 'apcer = Fraction(c["fa"], c["n_attack"]) if c["n_attack"] else None', 'apcer = Fraction(c["fa"], c["n_attack"]) if c["n_attack"] else Fraction(0)'),
    ("absent bona fide give BPCER 0", S, 'bpcer = Fraction(c["fr"], c["n_bona_fide"]) if c["n_bona_fide"] else None', 'bpcer = Fraction(c["fr"], c["n_bona_fide"]) if c["n_bona_fide"] else Fraction(0)'),
    ("ACER not halved", S, "    acer = (apcer + bpcer) / 2 if apcer", "    acer = (apcer + bpcer) if apcer"),
    ("undefined delta gives 0", S, 'return rm1["acer"] - r05["acer"] if rm1["acer"] is not None and r05["acer"] is not None else None',
     'return rm1["acer"] - r05["acer"] if rm1["acer"] is not None and r05["acer"] is not None else Fraction(0)'),
    ("delta with the opposite orientation", S, 'return rm1["acer"] - r05["acer"] if', 'return r05["acer"] - rm1["acer"] if'),
    ("M1 threshold not used", S, "results = {x: analyze(rows[x], 0.5, thr[x]) for x in an}", "results = {x: analyze(rows[x], 0.5, 0.5) for x in an}"),
    ("leave-one-out excludes nothing", S, 'c05, cm1 = minus(full["t05"], per[s]["t05"]), minus(full["tm1"], per[s]["tm1"])', 'c05, cm1 = full["t05"], full["tm1"]'),
    ("sign change inverted", S, '"sign_change": NA if d is None or d_full is None else sign(d) != sign(d_full)}', '"sign_change": NA if d is None or d_full is None else sign(d) == sign(d_full)}'),
    ("zero counted as opposite sign", S, 'if d0 is not None and v["sign"] in "+-" and sign(d0) in "+-" and v["sign"] != sign(d0)', 'if d0 is not None and v["sign"] != sign(d0)'),
    ("NA exclusions not counted", S, '"n_na": len(res["loso"]) - len(ds),', '"n_na": 0,'),
    ("subject ids in the public aggregate", S, '"n_images": ctx["exp"]["n_images"], "n_subjects"', '"n_images": ctx["exp"]["n_images"], "ids": ctx["results"][ctx["analyzers"][0]]["subjects"], "n_subjects"'),
    # ingressi
    ("input SHA-256 not checked", S, '        if h != spec["sha256"]:', "        if False:"),
    ("SHA256SUMS SHA-256 not checked", S, '        if sh != spec["sums_sha256"]:', "        if False:"),
    ("SHA256SUMS line not compared", S, "        if listed != h:", "        if False:"),
    ("SHA256SUMS line not required", S, "        if listed is None:", "        if False:"),
    ("symbolic-link input accepted", S, "    if stat.S_ISLNK(st.st_mode):\n        fail(", "    if False:\n        fail("),
    ("C1 and M1 scores not compared", S, "            if tc != tm:", "            if False:"),
    ("manifest not compared", S, "if (sc, lc) != (sm, lm) or (sc, lc) != man[sid]:", "if (sc, lc) != (sm, lm):"),
    ("sample sets not compared", S, "        if set(c1[a]) != set(m1[a]) or set(c1[a]) != set(man):", "        if False:"),
    ("row count not checked", S, '        if len(d) != exp["n_images"]:', "        if False:"),
    ("duplicate rows accepted", S, "        if sid in out[a]:", "        if False:"),
    ("dataset and split not checked", S, '        if r["dataset_id"] != exp.get("dataset_id", r["dataset_id"])', "        if False and r[\"dataset_id\"] != exp.get(\"dataset_id\", r[\"dataset_id\"])"),
    ("row threshold not checked", S, "        if t != thresholds[a]:", "        if False:"),
    ("stored prediction not checked", S, '        if r["prediction"] not in ("0", "1") or int(r["prediction"]) != int(s > thresholds[a]):', "        if False:"),
    ("synthetic threshold accepted", S, '    if doc.get("synthetic") is not False:', "    if False:"),
    ("class counts not checked", S, '        if (nb, len(rows) - nb) != (exp["n_bona_fide"], exp["n_attack"]):', "        if False:"),
    ("subject count not checked", S, '        if len(subj) != exp["n_subjects"]:', "        if False:"),
    ("dev subject count not checked", S, '    if len(dev_subjects) != exp["n_dev_subjects"]:', "    if False:"),
    # riproduzione della tabella M1
    ("accepted table not compared", S, "                if by[a].get(col) != v:", "                if False:"),
    ("dev block without the dev subjects", S, '("without the dev subjects", tabs[1], dev_subjects)', '("without the dev subjects", tabs[1], set())'),
    ("number of tables not checked", S, "    if len(tabs) != 2:", "    if False:"),
    # uscite
    ("run folder not validated", S, "    if msg := run_dir_problem(run, ROOT, inputs):", '    if msg := "":'),
    ("foreign entries accepted", S, "                if not any(mw.owned_name(e, n) for n in names):", "                if False:"),
    ("run folder inside an input package accepted", S, 'if label.startswith("SHA256SUMS of") and pg.within(run, Path(p).parent):', "if False:"),
    ("previous outputs kept until the end", S, "        left = invalidate_outputs(run)\n        if left:", "        left = []\n        if left:"),
    ("outputs kept after a failure", S, "        left = invalidate_outputs(run)\n        try:", "        left = []\n        try:"),
    ("failure recorded as PASS", S, '"status": "FAIL", "passed": False, "reason": reason', '"status": "PASS", "passed": False, "reason": reason'),
    ("dirty tree not declared", S, '("-dev" if state["dirty"] else "")', '""'),
]
# mutazione di controllo: solo un commento; deve SOPRAVVIVERE (prova che la suite rileva il comportamento, non il testo)
CONTROL = [("control: comment only", S, "def frac(x) -> str:\n", "def frac(x) -> str:  # control mutant: comment only\n")]


def copy_tree(dst: Path) -> Path:
    code = dst / "code"
    for rel in COPY:
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, code / rel)
    return code


def run_tests(code: Path, tmp: Path) -> tuple:
    tmp.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "N1_INPUT_ROOT": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp)}
    r = subprocess.run([sys.executable, "-I", TESTS], cwd=code, env=env, capture_output=True, text=True, timeout=900)
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("tests run")), "no test summary")
    skips = tuple(sorted(l for l in r.stdout.splitlines() if l.startswith("skipped ")))
    return r.returncode, line, skips


def undeclared_skips(skips) -> list:
    """Salti il cui motivo non è un ingresso mancante dichiarato (riga «skipped <test>: missing input: <motivo>»)."""
    allowed = {f"missing input: {why}" for why in MISSING_INPUT.values()}
    return [s for s in skips if s.split(": ", 1)[-1] not in allowed]


def detected(rc: int, line: str, skips=(), base_skips=()) -> str:
    """killed solo per fallimenti o errori; survived solo con exit 0 e gli stessi salti della copia non mutata."""
    m = re.match(r"tests run: (\d+), failures: (\d+), errors: (\d+), skipped: (\d+)", line)
    if not m:
        return "invalid"
    if int(m.group(2)) + int(m.group(3)) > 0:
        return "killed"
    return "survived" if rc == 0 and tuple(skips) == tuple(base_skips) else "invalid"


def one(tmp: Path, i: int, name: str, target: str, old: str, new: str, base_skips=()) -> tuple:
    work = tmp / f"work{i}"
    code = copy_tree(work)
    f = code / target
    f.write_text(f.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")
    rc, line, skips = run_tests(code, work / "tmp")
    shutil.rmtree(work, ignore_errors=True)
    return name, detected(rc, line, skips, base_skips), line, rc


def main() -> int:
    ap = argparse.ArgumentParser(description="Controllo delle mutazioni di n1_subject_sensitivity.py (offline)")
    ap.add_argument("--list", action="store_true", help="elenca le mutazioni e verifica i modelli, senza eseguire i test")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1), help="mutazioni eseguite in parallelo")
    a = ap.parse_args()
    problems = [f"{n}: the pattern occurs {k} times in {t} (expected exactly once)"
                for n, t, old, _ in MUTANTS + CONTROL if (k := (ROOT / t).read_text(encoding="utf-8").count(old)) != 1]
    for p in problems:
        print(f"ERROR: {p}", file=sys.stderr)
    if problems:
        print("RESULT: FAIL (a declared mutation no longer matches the code)")
        return 1
    if a.list:
        for n, t, _, _ in MUTANTS:
            print(f"{n}  [{t}]")
        for n, t, _, _ in CONTROL:
            print(f"{n}  [{t}] (control, must survive)")
        print(f"{len(MUTANTS)} mutations and {len(CONTROL)} control mutation, every pattern found exactly once")
        return 0
    killed = invalid = survived_controls = 0
    with tempfile.TemporaryDirectory(prefix="n1-subjects-mutations-") as tmp:
        tmp = Path(tmp)
        rc, line, base_skips = run_tests(copy_tree(tmp / "base"), tmp / "base" / "tmp")
        print(f"unmutated copy: {TESTS} exit {rc} ({line})", flush=True)
        for s in base_skips:
            print(f"unmutated copy: {s}", flush=True)
        if rc or not re.fullmatch(r"tests run: \d+, failures: 0, errors: 0, skipped: \d+", line):
            print("RESULT: FAIL (the tests do not pass on the unmutated copy)")
            return 1
        if bad := undeclared_skips(base_skips):
            for s in bad:
                print(f"ERROR: skip that is not a declared missing input: {s}", file=sys.stderr)
            print("RESULT: FAIL (the unmutated copy skips tests for a reason that is not a declared missing input)")
            return 1
        if base_skips:
            print(f"unmutated copy: {len(base_skips)} tests skipped, all for a declared missing input; mutations are counted "
                  "killed only by failures or errors of the tests that run", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
            futs = [ex.submit(one, tmp, i, n, t, old, new, base_skips) for i, (n, t, old, new) in enumerate(MUTANTS)]
            cfuts = [ex.submit(one, tmp, len(MUTANTS) + i, n, t, old, new, base_skips) for i, (n, t, old, new) in enumerate(CONTROL)]
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
                    print(f"ERROR: invalid mutant {name!r}: the test suite did not run to a summary with failures, or skipped other tests (exit {rc})", file=sys.stderr)
    print(f"mutations killed: {killed}/{len(MUTANTS)}" + (f" ({invalid} invalid)" if invalid else ""))
    print(f"control mutations survived: {survived_controls}/{len(CONTROL)}")
    ok = killed == len(MUTANTS) and survived_controls == len(CONTROL)
    print("RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
