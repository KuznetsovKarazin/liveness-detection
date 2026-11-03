"""
Controllo delle mutazioni di scripts/nuaa_step_zero.py (step zero NUAA), offline e deterministico.

Copia il codice (scripts/nuaa_step_zero.py, scripts/check_nuaa_step_zero.py, scripts/m2_writers.py, scripts/path_guard.py,
scripts/package_rules.py e la configurazione delivery/stepzero/) in una cartella temporanea, applica una mutazione
dichiarata alla volta (coppia cerca/sostituisci che deve comparire esattamente una volta: se un modello non si trova
più, il controllo si ferma con un errore) ed esegue scripts/check_nuaa_step_zero.py sulla copia mutata, con gli ingressi
veri in sola lettura (STEPZERO_INPUT_ROOT = radice di questo repository). Una mutazione è rilevata solo se la suite
arriva al riepilogo («tests run») con almeno un fallimento o errore, cioè dal comportamento; una suite che non parte
la rende non valida. Una mutazione di controllo (solo un commento) deve sopravvivere.

Test saltati: la copia non mutata deve passare e può saltare test solo per un ingresso mancante dichiarato
(MISSING_INPUT di scripts/check_nuaa_step_zero.py); i salti sono riportati. Una mutazione conta come rilevata solo per
fallimenti o errori, mai per un salto. Codice 1 se una mutazione sopravvive o non è valida, se un salto non è un
ingresso mancante dichiarato o se un modello non si trova.

  .venv/bin/python scripts/check_nuaa_step_zero_mutations.py            # tutte le mutazioni
  .venv/bin/python scripts/check_nuaa_step_zero_mutations.py --list     # solo l'elenco, con la verifica dei modelli
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
from check_nuaa_step_zero import MISSING_INPUT  # noqa: E402

TESTS = "scripts/check_nuaa_step_zero.py"
S = "scripts/nuaa_step_zero.py"
COPY = ("scripts/nuaa_step_zero.py", "scripts/check_nuaa_step_zero.py", "scripts/m2_writers.py", "scripts/path_guard.py",
        "scripts/package_rules.py")
CONFIG = "nuaa_step_zero.json"
# (nome, file mutato, testo cercato, testo sostitutivo)
MUTANTS = [
    # statistiche e misure
    ("RGB frames read as BGR", S, "gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY if rgb else cv2.COLOR_BGR2GRAY)", "gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)"),
    ("Laplacian kernel changed", S, "lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=1)", "lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)"),
    ("ties not given the average rank", S, "    avg = (ends - counts + 1 + ends) / 2", "    avg = ends.astype(np.float64)"),
    ("Mann-Whitney U with the wrong correction", S, "u = r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2", "u = r[:len(pos)].sum() - len(pos) * len(pos) / 2"),
    ("constant variable gives Spearman 0", S, "    if d == 0:\n        return None", "    if d == 0:\n        return 0.0"),
    ("quartiles swapped", S, '"q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75))', '"q1": float(np.percentile(v, 75)), "q3": float(np.percentile(v, 25))'),
    ("class AUROC with the classes swapped", S, '"class_auroc": {n: auroc(cls["attack"][n], cls["bona_fide"][n]) for n in names}', '"class_auroc": {n: auroc(cls["bona_fide"][n], cls["attack"][n]) for n in names}'),
    ("Spearman over all classes drops the bona fide", S, '        part["all"] = part["bona_fide"] + part["attack"]', '        part["all"] = part["attack"]'),
    ("rejected and accepted swapped", S, 'and scores[r["key"]][1] == 1]\n    acc', 'and scores[r["key"]][1] == 0]\n    acc'),
    ("relative box area not divided by the height", S, 'rel["attack" if keys[k]["label"] == "1" else "bona_fide"].append(w * h / (r["width"] * r["height"]))',
     'rel["attack" if keys[k]["label"] == "1" else "bona_fide"].append(w * h / r["width"])'),
    ("crop view claimed for frames outside the cache", S, '        in_cache = all(r["key"] in crop for r in rows)', "        in_cache = True"),
    # ingressi
    ("input SHA-256 not checked", S, '    if h != spec["sha256"]:\n        fail(f"{what} ({p.name}): SHA-256 mismatch', '    if False:\n        fail(f"{what} ({p.name}): SHA-256 mismatch'),
    ("SHA256SUMS SHA-256 not checked", S, '    if sh != spec["sums_sha256"]:', "    if False:"),
    ("SHA256SUMS line not compared", S, "    if listed != h:", "    if False:"),
    ("SHA256SUMS line not required", S, "    if listed is None:", "    if False:"),
    ("symbolic-link input accepted", S, "    if stat.S_ISLNK(st.st_mode):\n        fail(", "    if False:\n        fail("),
    ("manifest SHA-256 not checked", S, '        if h != spec["sha256"]:\n            fail(f"manifest {name}', '        if False:\n            fail(f"manifest {name}'),
    ("manifest session not compared with the name", S, 'if r["session"] != f["session"] or r["session"] not in spec["sessions"]:', 'if r["session"] not in spec["sessions"]:'),
    ("original size not checked", S, '    if p.stat().st_size != int(r["bytes"]):', "    if False:"),
    ("original SHA-256 not checked (size still checked)", S, '    if sha256_file(p) != r["sha256"]:', "    if False:"),
    ("tie at the threshold counted as attack", S, 'int(r["prediction"]) != int(s > t):', 'int(r["prediction"]) != int(s >= t):'),
    ("cache SHA-256 not checked", S, '    if inp.cache_sha != c["sha256"]:', "    if False:"),
    ("configuration counts of a manifest not checked", S, '            if spec[k] != v:\n                fail(f"manifest {name}: {k} {v}', '            if False:\n                fail(f"manifest {name}: {k} {v}'),
    ("expected cache frames not checked", S, 'if n != cfg["expected"]["n_cache_frames"] or mem["X_train"]', 'if False or mem["X_train"]'),
    ("box with zero width or height accepted", S, "        if w <= 0 or h <= 0:", "        if False:"),
    ("Fortran-ordered cache member accepted", S, '        if got != want or mem[k]["fortran"]:', "        if got != want:"),
    ("cache member with Python objects accepted", S, "                if dtype.hasobject:", "                if False:"),
    ("c1 outside cnn_test accepted", S, '    if not set(inp.by_key["c1"]) <= set(inp.by_key["cnn_test"]):', "    if False:"),
    ("label not compared with the key folder", S, 'if r["label"] not in ("0", "1") or KEY_DIR_OF_LABEL[int(r["label"])] != f["dir"]:', 'if r["label"] not in ("0", "1"):'),
    ("official lists sharing a file SHA-256 accepted", S, '    if {r["sha256"] for r in tr.values()} & {r["sha256"] for r in te.values()}:', "    if False:"),
    ("per-subject median replaced by the mean", S, "[f6(float(np.median([v[n] for v in vals]))) for n in names]", "[f6(float(np.mean([v[n] for v in vals]))) for n in names]"),
    ("within-subject AUROC counts every subject", S, '"within_subject_auroc": {"n_subjects": len(within),', '"within_subject_auroc": {"n_subjects": len(per_subject[ses]),'),
    ("bootstrap percentiles [25, 75]", S, '            lo, hi = np.percentile(vals, BOOT["percentiles"])', "            lo, hi = np.percentile(vals, [25, 75])"),
    ("bootstrap with 100 resamples", S, '        for _ in range(BOOT["resamples"]):', "        for _ in range(100):"),
    ("bootstrap seed 1", S, '        rng = np.random.default_rng(BOOT["seed"])\n        vals = []', "        rng = np.random.default_rng(1)\n        vals = []"),
    ("bootstrap generator outside the statistic loop (different resamples per statistic)", S,
     '    subjects = sorted(subs)\n    out = {}\n    for n in names:\n        att = [subs[s]["attack"][n] if "attack" in subs[s] else np.empty(0) for s in subjects]\n'
     '        bf = [subs[s]["bona_fide"][n] if "bona_fide" in subs[s] else np.empty(0) for s in subjects]\n        rng = np.random.default_rng(BOOT["seed"])\n',
     '    subjects = sorted(subs)\n    out = {}\n    rng = np.random.default_rng(BOOT["seed"])\n    for n in names:\n        att = [subs[s]["attack"][n] if "attack" in subs[s] else np.empty(0) for s in subjects]\n'
     '        bf = [subs[s]["bona_fide"][n] if "bona_fide" in subs[s] else np.empty(0) for s in subjects]\n'),
    ("bootstrap resampled subjects deduplicated", S, "            idx = rng.integers(0, len(subjects), len(subjects))", "            idx = np.unique(rng.integers(0, len(subjects), len(subjects)))"),
    ("within-subject median replaced by the mean", S, '{"median": float(np.median([v[2][n] for v in within.values()])),', '{"median": float(np.mean([v[2][n] for v in within.values()])),'),
    ("within-subject min and max swapped", S, '"min": float(min(v[2][n] for v in within.values())),\n                                                     "max": float(max(v[2][n] for v in within.values()))}',
     '"min": float(max(v[2][n] for v in within.values())),\n                                                     "max": float(min(v[2][n] for v in within.values()))}'),
    ("bootstrap recipe of the configuration not compared", S, '    if cfg["bootstrap"] != BOOT:', "    if False:"),
    ("declared sessions of the official lists may overlap", S, '    if set(man["files"]["official_train"]["sessions"]) & set(man["files"]["official_test"]["sessions"]):', "    if False:"),
    ("subject counts of session 03 not computed from the rows", S, '    s03 = {"bona_fide": len(bf03), "attack": len(at03), "both_classes": len(bf03 & at03)}', '    s03 = {"bona_fide": len(bf03), "attack": len(bf03), "both_classes": len(bf03 & at03)}'),
    ("official_split not compared with the list", S, '            if r["official_split"] != LIST_SPLIT[name]:', "            if False:"),
    ("crop rows of the per-subject AUROC labelled raw", S, '("crop", crop, STATS, CROP_SETS)):\n        for s in sets:\n            per = by_subject', '("raw", crop, STATS, CROP_SETS)):\n        for s in sets:\n            per = by_subject'),
    ("bootstrap resample without a class counted as 0.5", S, "            if a is not None:\n                vals.append(a)", "            vals.append(a if a is not None else 0.5)"),
    ("public SHA256SUMS computed before the figure", S, '    if png is not None:\n        public[FIGURE] = png\n    public["SHA256SUMS"] = sums_text(public)', '    public["SHA256SUMS"] = sums_text(public)\n    if png is not None:\n        public[FIGURE] = png'),
    ("declared text declaration not validated", S, '        if k not in DECLARATIONS and not (isinstance(v, str) and v):', "        if False:"),
    ("stored prediction not checked", S, '            if r["prediction"] not in ("0", "1") or int(r["prediction"]) != int(s > t):', "            if False:"),
    ("score subject and label not compared", S, '            if r["subject_id"] != m["subject"] or r["label"] != m["label"]:', "            if False:"),
    ("cache labels and subjects not compared", S, '                if str(int(y[i])) != r["label"] or str(s[i]) != r["subject"]:', "                if False:"),
    ("target not excluded from the sampled frames", S, 'inp.rows["target_491"] = sorted((r for r in inp.rows["official_train"] if r["key"] not in used)',
     'inp.rows["target_491"] = sorted((r for r in inp.rows["official_train"] if True)'),
    ("expected counts of the official lists not checked", S, '        if exp[k] != v:\n            fail(f"official lists:', '        if False:\n            fail(f"official lists:'),
    ("box row count not checked", S, '    if sum(len(d) for d in out.values()) != spec["rows"] or any(len(d) != len(keys) for d in out.values()):', "    if False:"),
    ("authorisation flag not required", S, "    if not a.i_am_authorized:", "    if False:"),
    # uscite
    ("public scan skipped", S, "        if probs := public_problems(text, inp, is_md):", '        if probs := []:'),
    ("foreign entries accepted", S, "                if not any(mw.owned_name(e, n) for n in names):", "                if False:"),
    ("run folder not validated", S, "    if msg := run_dir_problem(run, ROOT, inputs):", '    if msg := "":'),
    ("failure recorded as PASS", S, '"status": "FAIL", "passed": False, "reason": reason', '"status": "PASS", "passed": False, "reason": reason'),
    ("dirty tree not declared", S, '("-dev" if state["dirty"] else "")', '""'),
    ("figure note not recorded in status.json", S, '"figure": doc["figure"],\n                           "public_sums_sha256"', '"figure": "none",\n                           "public_sums_sha256"'),
]
# mutazione di controllo: solo un commento; deve SOPRAVVIVERE (prova che la suite rileva il comportamento, non il testo)
CONTROL = [("control: comment only", S, "def f6(v) -> str:\n", "def f6(v) -> str:  # control mutant: comment only\n")]
# controllo dichiarato ridondante, non raggiungibile da nessun ingresso: le sessioni dichiarate delle due liste sono disgiunte
# (load_config), ogni riga ha la sessione del nome dentro quelle della sua lista e official_split uguale alla lista, quindi
# una chiave (stesso nome, stessa sessione) non può stare in entrambe le liste. Deve SOPRAVVIVERE.
REDUNDANT = [("declared redundant (must survive): official lists sharing a key", S,
              '    if set(tr) & set(te):\n        fail("official train and test lists share a key")', '    if False:\n        fail("official train and test lists share a key")')]


def copy_tree(dst: Path) -> Path:
    """Copia del codice e, se c'è, della configurazione consegnata (delivery/stepzero/ nel repository di sviluppo e nel
    pacchetto livenesslab/, oppure accanto alla radice del pacchetto), al percorso predefinito dello script."""
    code = dst / "code"
    for rel in COPY:
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, code / rel)
    for c in (ROOT / "delivery" / "stepzero", ROOT.parent):
        if (c / CONFIG).is_file():
            (code / "delivery" / "stepzero").mkdir(parents=True, exist_ok=True)
            shutil.copy2(c / CONFIG, code / "delivery" / "stepzero" / CONFIG)
            break
    return code


def run_tests(code: Path, tmp: Path) -> tuple:
    tmp.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "STEPZERO_INPUT_ROOT": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp)}
    r = subprocess.run([sys.executable, "-I", TESTS], cwd=code, env=env, capture_output=True, text=True, timeout=1800)
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
    ap = argparse.ArgumentParser(description="Controllo delle mutazioni di nuaa_step_zero.py (offline)")
    ap.add_argument("--list", action="store_true", help="elenca le mutazioni e verifica i modelli, senza eseguire i test")
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1), help="mutazioni eseguite in parallelo")
    a = ap.parse_args()
    problems = [f"{n}: the pattern occurs {k} times in {t} (expected exactly once)"
                for n, t, old, _ in MUTANTS + CONTROL + REDUNDANT if (k := (ROOT / t).read_text(encoding="utf-8").count(old)) != 1]
    for p in problems:
        print(f"ERROR: {p}", file=sys.stderr)
    if problems:
        print("RESULT: FAIL (a declared mutation no longer matches the code)")
        return 1
    if a.list:
        for n, t, _, _ in MUTANTS:
            print(f"{n}  [{t}]")
        for n, t, _, _ in CONTROL + REDUNDANT:
            print(f"{n}  [{t}] (must survive)")
        print(f"{len(MUTANTS)} mutations, {len(CONTROL)} control mutation and {len(REDUNDANT)} declared redundant, every pattern found exactly once")
        return 0
    killed = invalid = survived_controls = 0
    with tempfile.TemporaryDirectory(prefix="stepzero-mutations-") as tmp:
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
            cfuts = [ex.submit(one, tmp, len(MUTANTS) + i, n, t, old, new, base_skips) for i, (n, t, old, new) in enumerate(CONTROL + REDUNDANT)]
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
    print(f"control mutations survived: {survived_controls}/{len(CONTROL) + len(REDUNDANT)} ({len(CONTROL)} control, {len(REDUNDANT)} declared redundant)")
    ok = killed == len(MUTANTS) and survived_controls == len(CONTROL) + len(REDUNDANT)
    print("RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
