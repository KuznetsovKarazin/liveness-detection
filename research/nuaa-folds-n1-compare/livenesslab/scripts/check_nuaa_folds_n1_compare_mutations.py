"""
Controllo delle mutazioni di scripts/nuaa_folds_n1_compare.py (confronto run dei fold contro N1), offline e
deterministico.

Copia il codice (scripts/nuaa_folds_n1_compare.py, scripts/check_nuaa_folds_n1_compare.py, i moduli condivisi
importati e, se c'è, la configurazione delivery/foldscompare/) in una cartella temporanea, applica una mutazione
dichiarata alla volta (coppia cerca/sostituisci che deve comparire esattamente una volta: se un modello non si trova
più, il controllo si ferma con un errore) ed esegue scripts/check_nuaa_folds_n1_compare.py sulla copia mutata, con gli
ingressi veri in sola lettura (FOLDSCOMPARE_INPUT_ROOT = radice di questo repository). Una mutazione è rilevata solo
se la suite arriva al riepilogo («tests run») con almeno un fallimento o errore, cioè dal comportamento; una suite che
non parte la rende non valida. Una mutazione di controllo (solo un commento) deve sopravvivere.

Test saltati: la copia non mutata deve passare e può saltare test solo per un ingresso mancante dichiarato
(MISSING_INPUT di scripts/check_nuaa_folds_n1_compare.py); i salti sono riportati. Una mutazione conta come rilevata
solo per fallimenti o errori, mai per un salto. Codice 1 se una mutazione sopravvive o non è valida, se un salto non è
un ingresso mancante dichiarato o se un modello non si trova.

  .venv/bin/python -I scripts/check_nuaa_folds_n1_compare_mutations.py            # tutte le mutazioni
  .venv/bin/python -I scripts/check_nuaa_folds_n1_compare_mutations.py --list     # solo l'elenco, con la verifica dei modelli
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
from check_nuaa_folds_n1_compare import MISSING_INPUT  # noqa: E402

TESTS = "scripts/check_nuaa_folds_n1_compare.py"
S = "scripts/nuaa_folds_n1_compare.py"
COPY = (S, TESTS, "scripts/m2_writers.py", "scripts/path_guard.py", "scripts/package_rules.py", "scripts/nuaa_manifests.py",
        "scripts/select_threshold.py", "src/tesi_app/__init__.py", "src/tesi_app/paths.py")
CONFIG = "delivery/foldscompare/nuaa_folds_n1_compare.json"
# (nome, file mutato, testo cercato, testo sostitutivo)
MUTANTS = [
    # regola di decisione e metriche
    ("decision: tie at 0.5 counted as attack in the folds rows", S,
     'r["prediction_05"] != ("1" if s > FIXED_THRESHOLD else "0")', 'r["prediction_05"] != ("1" if s >= FIXED_THRESHOLD else "0")'),
    ("decision: tie at the dev threshold counted as attack", S,
     'r["prediction_dev"] != ("1" if s > float(value) else "0")', 'r["prediction_dev"] != ("1" if s >= float(value) else "0")'),
    ("decision: tie at 0.5 counted as attack in the N1 rows", S,
     'if r["prediction"] != ("1" if s > FIXED_THRESHOLD else "0"):', 'if r["prediction"] != ("1" if s >= FIXED_THRESHOLD else "0"):'),
    ("auroc: ties get the highest rank (counted 1, not 0.5)", S,
     "avg = np.cumsum(cnt) - (cnt - 1) / 2.0", "avg = np.cumsum(cnt) * 1.0"),
    ("acer: maximum of APCER and BPCER instead of the mean", S,
     "acer = (apcer + bpcer) / 2 if", "acer = max(apcer, bpcer) if"),
    ("apcer: divided by all frames", S, "apcer = accepted / n_a if n_a else None", "apcer = accepted / len(y) if n_a else None"),
    ("folds AUROC computed on the N1 scores", S,
     'f = metrics(y, sel(a["f_s"]), {"05": sel(a["f_p05"]), "dev": sel(a["f_pdev"])})',
     'f = metrics(y, sel(a["n1_s"]), {"05": sel(a["f_p05"]), "dev": sel(a["f_pdev"])})'),
    ("difference at dev uses the folds ACER at 0.5", S,
     'flat["diff_acer_dev"] = d(f["acer_dev"], n["acer_05"])', 'flat["diff_acer_dev"] = d(f["acer_05"], n["acer_05"])'),
    ("difference of AUROC with the sign reversed", S,
     'flat["diff_auroc"] = d(f["auroc"], n["auroc"])', 'flat["diff_auroc"] = d(n["auroc"], f["auroc"])'),
    ("summary: mean instead of median", S,
     '"median": float(statistics.median(values))', '"median": float(statistics.mean(values))'),
    # bootstrap
    ("bootstrap: another random seed", S, 'rng = np.random.default_rng(BOOT["seed"])', 'rng = np.random.default_rng(BOOT["seed"] + 1)'),
    ("bootstrap: frames resampled instead of subjects", S,
     "ix = np.concatenate([index[i] for i in pick])", 'ix = rng.integers(0, len(a["y"]), len(a["y"]))'),
    ("bootstrap: percentiles 5-95", S, 'np.percentile(vals[k], BOOT["percentiles"], method="linear")',
     'np.percentile(vals[k], [5, 95], method="linear")'),
    ("bootstrap: percentiles by nearest rank", S, 'np.percentile(vals[k], BOOT["percentiles"], method="linear")',
     'np.percentile(vals[k], BOOT["percentiles"], method="nearest")'),
    ("bootstrap: generator shared between the training seeds", S, 'rng = np.random.default_rng(BOOT["seed"])',
     'rng = bootstrap.__dict__.setdefault("rng", np.random.default_rng(BOOT["seed"]))'),
    ("bootstrap: resamples without a class counted", S,
     "            if m[k] is not None:\n                vals[k].append(m[k])", "            if True:\n                vals[k].append(m[k] or 0.0)"),
    ("bootstrap: minimum of subjects not enforced", S, 'if len(subjects) < BOOT["min_subjects"]:', "if False:"),
    # confronti appaiati e per soggetto
    ("2x2: N1-only cell counts the folds-only frames", S,
     '"n1_only_correct": int((m & n1_ok & ~f_ok).sum())', '"n1_only_correct": int((m & ~n1_ok & f_ok).sum())'),
    ("2x2: changed decisions at dev counted at 0.5", S,
     't["decisions_changed"] = int((a["n1_p"] != fp).sum())', 't["decisions_changed"] = int((a["n1_p"] != a["f_p05"]).sum())'),
    ("subjects: lower or equal counted as equal", S, 'c["equal"] += fv == nv', 'c["equal"] += fv <= nv'),
    ("subjects: both classes counted with one class", S, "if r[3] and r[4]", "if r[3] or r[4]"),
    ("paired frames: decisions at 0.5 and dev swapped", S,
     'int(a["fold"][i]), fmt(a["f_s"][i]), fmt(a["thr"][i]), int(a["f_p05"][i]), int(a["f_pdev"][i])])',
     'int(a["fold"][i]), fmt(a["f_s"][i]), fmt(a["thr"][i]), int(a["f_pdev"][i]), int(a["f_p05"][i])])'),
    # per soggetto
    ("subjects: folds AUROC computed on the N1 scores", S, 'fmt(auroc(a["f_s"][ix][y == 1], a["f_s"][ix][y == 0]))])',
     'fmt(auroc(a["n1_s"][ix][y == 1], a["n1_s"][ix][y == 0]))])'),
    ("subjects: N1 AUROC computed on the folds scores", S, 'fmt(auroc(a["n1_s"][ix][y == 1], a["n1_s"][ix][y == 0])), fmt(',
     'fmt(auroc(a["f_s"][ix][y == 1], a["f_s"][ix][y == 0])), fmt('),
    ("aggregate.md: changed decisions at dev printed from 0.5", S, 'cell(pd["attack"]), pd["decisions_changed"]])',
     'cell(pd["attack"]), agg["per_seed"][str(s)]["paired_decisions"]["05"]["decisions_changed"]])'),
    ("aggregate.md: N1 AUROC note also when one seed is at or above 0.5", S,
     'if v["max"] is None or v["max"] >= 0.5:', 'if v["max"] is None:'),
    # ingressi: manifest
    ("inputs: manifest duplicated keys accepted", S,
     'if len({r["key"] for r in rows}) != len(rows) or len({r["sha256"] for r in rows}) != len(rows):',
     'if len({r["sha256"] for r in rows}) != len(rows):'),
    ("inputs: schema SHA-256 fixed in the code not compared", S,
     'if h != sc["sha256"] or h != SCHEMA_CONFIG_SHA256:', 'if h != sc["sha256"]:'),
    ("inputs: cnn_test SHA-256 of the schema not compared", S,
     'if not isinstance(spec, dict) or spec.get("sha256") != m["sha256"] or spec.get("file") != m["file"]:',
     'if not isinstance(spec, dict) or spec.get("file") != m["file"]:'),
    ("inputs: manifest SHA-256 not compared", S, '    if hb != m["sha256"]:', "    if False:"),
    ("inputs: manifest counts not compared", S, "        if m[k] != v:\n", "        if False:\n"),
    ("inputs: manifest file SHA-256 duplicates accepted", S,
     'if len({r["key"] for r in rows}) != len(rows) or len({r["sha256"] for r in rows}) != len(rows):',
     'if len({r["key"] for r in rows}) != len(rows):'),
    ("inputs: N1 training subjects from the last manifest only", S,
     '        train_subjects |= {r["subject"] for r in trows}', '        train_subjects = {r["subject"] for r in trows}'),
    ("inputs: training subjects in the set not compared", S, '    if overlap != tm["training_subjects_in_set"]:', "    if False:"),
    # ingressi: run dei fold
    ("folds run: status PASS not required", S,
     'if st_doc.get("status") != "PASS" or st_doc.get("passed") is not True or', 'if st_doc.get("passed") is not True or'),
    ("folds run: restricted SHA256SUMS not anchored", S,
     '    if sha256_bytes(rb) != fr["restricted_sums_sha256"]:', "    if False:"),
    ("folds run: score files not compared with the SHA256SUMS", S, "                if sha256_bytes(b) != listed[fn]:", "                if False:"),
    ("folds run: role of the rows not checked", S,
     'if r["fold"] != str(fold) or r["train_seed"] != str(seed) or r["role"] != "test":',
     'if r["fold"] != str(fold) or r["train_seed"] != str(seed):'),
    ("folds run: scores above 1 accepted", S,
     "if not math.isfinite(x) or not 0.0 <= x <= 1.0:", "if not math.isfinite(x):"),
    ("folds run: threshold_dev of the rows not compared", S,
     'if r["threshold_dev"] != repr(float(value)) or float(r["threshold_dev"]) != float(value):', "if False:"),
    ("folds run: threshold_sha256 of the rows not compared", S, '                if r["threshold_sha256"] != thr_sha:', "                if False:"),
    ("folds run: checkpoint of the rows not compared", S, '                if r["checkpoint_sha256"] != ckpt:', "                if False:"),
    ("folds run: checkpoint of the threshold file not compared", S,
     'or not _number(value) or thr.get("checkpoint_sha256") != ckpt:', "or not _number(value):"),
    ("folds run: subject in two folds accepted", S, "                if prev != fold:", "                if False:"),
    ("folds run: rows compared with the manifest without the file SHA-256", S,
     '            and r["file_sha256"] == o["sha256"])', "            )"),
    ("folds run: a key in two rows accepted", S, "if missing or twice or len(inp.folds[seed]) != len(inp.by_key):",
     "if missing or len(inp.folds[seed]) != len(inp.by_key):"),
    ("folds run: folds of different seeds accepted", S,
     'if any(parts[s] != parts[cfg["train_seeds"][0]] for s in cfg["train_seeds"]):', "if False:"),
    # ingressi: N1
    ("N1: SHA256SUMS not anchored", S, '    if sha256_bytes(sb) != n1["sha256sums_sha256"]:', "    if False:"),
    ("N1: checkpoint and configuration of the rows not compared", S,
     'if r["checkpoint_sha256"] != weights or r["config_sha256"] != agg.get("config_sha256"):', "if False:"),
    ("N1: scores_sha256 of the aggregate not compared", S, '        if sa.get("scores_sha256") != sha256_bytes(b):', "        if False:"),
    ("N1: weights_sha256 of metrics.json and of the aggregate not compared", S,
     'if not _hex(weights) or (sa.get("training") or {}).get("weights_sha256") != weights:', "if not _hex(weights):"),
    ("N1: a key in two rows accepted", S, '            if r["sample_id"] in out:', "            if False:"),
    # ancore e valori di controllo
    ("anchors: tolerance 1e-6", S, "ok = g is not None and _number(w) and abs(g - w) <= ANCHOR_TOL",
     "ok = g is not None and _number(w) and abs(g - w) <= 1e-6"),
    ("anchors: not enforced", S, '    if problems:\n        fail("anchors not reproduced', '    if False:\n        fail("anchors not reproduced'),
    ("control values: ACER agreement at one decimal of the percentage", S,
     'return round(100 * g, CONTROL_DIGITS["acer"]) == round(w, CONTROL_DIGITS["acer"])', 'return round(100 * g, 1) == round(w, 1)'),
    ("control values: ACER compared as a fraction", S,
     'return round(100 * g, CONTROL_DIGITS["acer"]) == round(w, CONTROL_DIGITS["acer"])',
     'return round(g, CONTROL_DIGITS["acer"]) == round(w, CONTROL_DIGITS["acer"])'),
    ("control values: AUROC agreement at four decimals", S,
     'return round(g, CONTROL_DIGITS["auroc"]) == round(w, CONTROL_DIGITS["auroc"])', 'return round(g, 4) == round(w, 4)'),
    ("control values: ACER at dev read from the ACER at 0.5", S,
     '"acer_dev": [per[s]["folds"]["acer_dev"] for s in cfg["train_seeds"]]', '"acer_dev": [per[s]["folds"]["acer_05"] for s in cfg["train_seeds"]]'),
    ("control values: not compared", S, "            if not control_agrees(g, w, k):", "            if False:"),
    # uscite e run
    ("public: the Markdown not scanned", S,
     'if probs := public_problems(text, inp, name.endswith(".md")):', 'if probs := (public_problems(text, inp) if name.endswith(".json") else []):'),
    ("public: N1 training subjects not scanned", S,
     "for s in sorted(inp.subjects | inp.n1_training_subjects):", "for s in sorted(inp.subjects):"),
    ("restricted: SHA256SUMS without per_subject.csv", S, '        restricted["SHA256SUMS"] = sums_text(restricted)',
     '        restricted["SHA256SUMS"] = sums_text({k: v for k, v in restricted.items() if k != "per_subject.csv"})'),
    ("restricted: public anchor of another file", S,
     '"public_SHA256SUMS.sha256": sha256_bytes(public["SHA256SUMS"].encode("utf-8"))',
     '"public_SHA256SUMS.sha256": sha256_bytes(public["aggregate.json"].encode("utf-8"))'),
    ("run id: dirty tree without -dev", S, '+ ("-dev" if state["dirty"] else "")', '+ ""'),
    ("run: authorisation not required", S, "    if not a.i_am_authorized:", "    if False:"),
    ("run: existing status.json overwritten", S, "    if (run / STATUS).exists() or (run / STATUS).is_symlink():", "    if False:"),
    ("run: previous outputs not invalidated at the start", S,
     "        left = invalidate_outputs(run)\n        if left:", "        left = []\n        if left:"),
    ("run: failure leaves the outputs", S, "        left = invalidate_outputs(run)\n        if log is not None:",
     "        left = []\n        if log is not None:"),
    ("run: foreign entries in the run folder accepted", S, "                if not any(mw.owned_name(e, n) for n in names):", "                if False:"),
    ("run: symbolic output root accepted", S, "    if (out_root if out_root is not None else run.parent).is_symlink():", "    if False:"),
    ("run: input containing the run folder accepted", S,
     "        if pg.within(p, run) or (Path(p).is_dir() and pg.within(run, p)):", "        if pg.within(p, run):"),
    ("run: run log SHA-256 not recorded", S, '"run_log_sha256": sha256_file(run / LOG)})', '"run_log_sha256": "none"})'),
]
# mutazione di controllo: solo un commento; deve SOPRAVVIVERE (prova che la suite rileva il comportamento, non il testo)
CONTROL = [("control: comment only", S, "def f4(v) -> str:\n", "def f4(v) -> str:  # control mutant: comment only\n")]


def copy_tree(dst: Path) -> Path:
    code = dst / "code"
    for rel in COPY + ((CONFIG,) if (ROOT / CONFIG).is_file() else ()):
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, code / rel)
    return code


def run_tests(code: Path, tmp: Path) -> tuple:
    tmp.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "FOLDSCOMPARE_INPUT_ROOT": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp)}
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
    ap = argparse.ArgumentParser(description="Controllo delle mutazioni di nuaa_folds_n1_compare.py (offline)")
    ap.add_argument("--list", action="store_true", help="elenca le mutazioni e verifica i modelli, senza eseguire i test")
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1), help="mutazioni eseguite in parallelo")
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
            print(f"{n}  [{t}] (must survive)")
        print(f"{len(MUTANTS)} mutations and {len(CONTROL)} control mutation, every pattern found exactly once")
        return 0
    killed = invalid = survived_controls = 0
    with tempfile.TemporaryDirectory(prefix="foldscompare-mutations-") as tmp:
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
