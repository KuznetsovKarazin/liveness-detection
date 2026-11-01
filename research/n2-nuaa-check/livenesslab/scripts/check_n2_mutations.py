"""
Mutazioni dichiarate di scripts/n2_sessions.py: ogni mutazione toglie o altera un controllo o una regola, e la suite
scripts/check_n2_sessions.py deve fallire (mutazione rilevata dal comportamento, non da un confronto del sorgente).
Stesso programma di check_n1_mutations.py: ogni mutazione in una copia NUOVA del codice in una cartella temporanea,
tolta dopo; la suite non mutata deve passare in una copia nuova prima e dopo; una mutazione di controllo (solo un
commento) deve sopravvivere; una mutazione è rilevata solo se la suite gira davvero e riporta fallimenti o errori con lo
stesso numero di test della suite non mutata (funzione classify); un import fallito, un SyntaxError o una riga di
riepilogo mancante sono errori del programma, mai mutazioni rilevate. Il testo da sostituire deve comparire esattamente
una volta nel file vero, che non cambia.

  python scripts/check_n2_mutations.py        # righe finali: mutations killed: N/N, control mutations survived: 1/1, RESULT: PASS / FAIL

Dipendenze: quelle della suite (numpy, scikit-learn); nessun dato, nessun TensorFlow.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["scripts/n2_sessions.py", "scripts/n1_train.py", "scripts/train_cnn.py", "scripts/check_n2_sessions.py", "scripts/check_n2_mutations.py",
         "scripts/nuaa_manifests.py", "scripts/select_threshold.py", "scripts/path_guard.py", "scripts/m2_writers.py", "scripts/package_rules.py",
         "src/tesi_app/__init__.py", "src/tesi_app/paths.py", "src/tesi_app/evaluation.py", "src/tesi_app/core.py", "src/tesi_app/face.py"]
S = "scripts/n2_sessions.py"

MUTATIONS = [
    # (file, descrizione, testo originale, testo mutato)
    (S, "delivered configuration SHA-256 not compared", "    if cfg_sha != CONFIG_SHA256:", "    if False:"),
    (S, "config key set not compared", "    if set(cfg) != want:", "    if False:"),
    (S, "threshold and decision rule not fixed in the configuration",
     'if not isinstance(m, dict) or m.get("threshold") != THRESHOLD or m.get("decision") != DECISION_RULE or m.get("sets") != list(SETS):',
     'if not isinstance(m, dict) or m.get("sets") != list(SETS):'),
    (S, "parity tolerance not fixed in the configuration", '    if not isinstance(par, dict) or par.get("tolerance_abs") != TOLERANCE:', "    if False:"),
    (S, "cache-check step not fixed in the configuration", '    if not isinstance(cc, dict) or n_cache is None or cc.get("frames") != n_cache or cc.get("step") != 1:', "    if False:"),
    (S, "parity rule fields not validated", '    if not isinstance(par, dict) or not _int(par.get("n_keys")) or par.get("n_keys") < 1 or not _hex(par.get("keys_sha256")) or "keys" in par:', "    if False:"),
    (S, "target counts accepted when inconsistent", '            or not all(_int(v) for v in t["by_session"].values()) or t["rows"] != sum(t["by_session"].values()) or t["in_c1"] != 0 or t["in_cnn_test"] != 0:',
     '            or not all(_int(v) for v in t["by_session"].values()) or t["rows"] != sum(t["by_session"].values()):'),
    (S, "study folder name not checked", "    if not RUN_ID_RE.match(d.name):", "    if False:"),
    (S, "protected repository folders accepted", 'if msg := pg.deletion_problem(d, ROOT, pg.BASE_PROTECTED, [ROOT / "results" / "n2"]):', "if False:"),
    (S, "moved protected folders accepted", "        if pg.within(d, q) or (Path(q).exists() and pg.contains(d, q)):", "        if False:"),
    (S, "symbolic link study folder accepted", "    if d.is_symlink():\n        return f\"{d.name} is a symbolic link", "    if False:\n        return f\"{d.name} is a symbolic link"),
    (S, "inputs inside the study accepted", "        if pg.within(p, d) or pg.contains(p, d):", "        if False:"),
    (S, "study folder need not be new", "    if d.exists() and any(d.iterdir()):", "    if False:"),
    (S, "non-trial study id without the commit", "    if not is_trial(d) and not str(state.get(\"commit\", \"\")).startswith(m.group(2)):", "    if False:"),
    (S, "cache SHA-256 not compared", '    if inp.cache_sha != d["cache"]["sha256"]:', "    if False:"),
    (S, "manifest SHA-256 not compared", '        if got != e["sha256"]:\n            fail(f"manifest', '        if False:\n            fail(f"manifest'),
    (S, "manifest counts not compared", "            if c[f] != e[f]:", "            if False:"),
    (S, "rows with the same key not compared field by field", "            if inp.by_key[sup][k] != r:", "            if False:"),
    (S, "sampling order not compared", '    if sorted(inp.order["train"]) != sorted(list(inp.by_key["cnn_train"]) + list(inp.by_key["cnn_es"])):', "    if False:"),
    (S, "target counts not compared", "        if t[k] != v:\n            fail(f\"target {k}", "        if False:\n            fail(f\"target {k}"),
    (S, "target manifest SHA-256 not compared", '    if inp.target_sha != t["manifest_sha256"]:', "    if False:"),
    (S, "original SHA-256 and size not compared", '            if p.stat().st_size != int(r["bytes"]) or sha256_file(p) != r["sha256"]:', "            if False:"),
    (S, "symbolic link accepted as an original", "        if p.is_symlink():\n            fail(f\"original of {r['key']}: symbolic link refused\")", "        if False:\n            pass"),
    (S, "official train and test lists not compared", '        if {r[dim] for r in inp.rows["official_train"]} & {r[dim] for r in inp.rows["official_test"]}:', "        if False:"),
    (S, "subject shared by fitting and validation accepted", '    for dim in ("key", "sha256", "subject"):\n        if {r[dim] for r in inp.rows["cnn_train"]}',
     '    for dim in ("key", "sha256"):\n        if {r[dim] for r in inp.rows["cnn_train"]}'),
    (S, "cache member dtype and shape not compared", '        if got != want or mem[k]["fortran"]:', "        if False:"),
    (S, "N1 study folder name not checked", '    if d.name != n1["run_id"]:\n        fail(f"--n1-study', '    if False:\n        fail(f"--n1-study'),
    (S, "N1 check.json not read as a record", '    N1.load_record(ck, "check", n1["run_id"], n1["config_sha256"])\n', "    pass\n"),
    (S, "N1 score keys not compared with cnn_test", '        if sorted(keys) != sorted(inp.by_key["cnn_test"]) or len(set(keys)) != len(keys):\n            fail(f"N1 seed{seed} scores',
     '        if False:\n            fail(f"N1 seed{seed} scores'),
    (S, "frozen module name not checked", '    if ff.name != pre["frozen_face"]["file"]:', "    if False:"),
    (S, "parity not bound to cache_check.json at score", '    if par.get("cache_check_sha256") != sha256_file(d / "cache_check.json"):', "    if False:"),
    (S, "frames_target keys not compared at score", '    if sorted(expected) != [r["key"] for r in inp.target]:', "    if False:"),
    (S, "detector name not compared at score", '        if h != expected[r["key"]]["frame_sha256"] or det != expected[r["key"]]["detector"]:', '        if h != expected[r["key"]]["frame_sha256"]:'),
    (S, "score row order not compared in aggregate", '        if keys != [r["key"] for r in inp.target]:', "        if False:"),
    (S, "score.json seed not compared in aggregate", '        if sc.get("train_seed") != seed or sc.get("weights_sha256") != inp.n1[seed]["weights_sha256"] \\',
     '        if sc.get("weights_sha256") != inp.n1[seed]["weights_sha256"] \\'),
    (S, "M1 dev flag taken from the record", '"in_m1_dev": "1" if r["sample_id"] in inp.m1_dev_keys else "0", "file_sha256": m["sha256"],',
     '"in_m1_dev": r["in_m1_dev"], "file_sha256": m["sha256"],'),
    (S, "per-subject table not recomputed in aggregate", "        if N1.csv_text(subj_rows, SUBJECT_COLUMNS) != N1.csv_text(recomputed, SUBJECT_COLUMNS):", "        if False:"),
    (S, "detector counts not compared in aggregate", '        if sc.get("detectors") != dict(Counter(r["detector"] for r in rows)) or sc.get("detectors") != cache_rec["target_frames"]["detectors"]:',
     "        if False:"),
    (S, "public scan skipped for aggregate.md", '    for name, text in (("aggregate.json", js), ("aggregate.md", md)):\n        if probs := N1.public_problems(text, inp):',
     '    for name, text in (("aggregate.json", js),):\n        if probs := N1.public_problems(text, inp):'),
    (S, "reference not restricted to the target subjects", '    nine = [r for r in rows if r["group"] != "other"]', "    nine = rows"),
    (S, "stratum error definition inverted", '            err = sum(1 for r in sel if r["prediction"] == ("1" if lab == "0" else "0"))',
     '            err = sum(1 for r in sel if r["prediction"] == ("0" if lab == "0" else "1"))'),
    (S, "set titles with a literal count", "    return {k: tpl.format(**v) for k, tpl in SETS.items()}", '    return {k: tpl.format(**{**v, "n_target": 491}) for k, tpl in SETS.items()}'),
    (S, "N1 check.json SHA-256 not compared", '    if ck.is_symlink() or not ck.is_file() or sha256_file(ck) != n1["check_sha256"]:', "    if ck.is_symlink() or not ck.is_file():"),
    (S, "N1 weights SHA-256 not compared", '        if sha256_file(w) != b["weights_sha256"]:', "        if False:"),
    (S, "N1 scores SHA-256 not compared", '        if sp.is_symlink() or not sp.is_file() or sha256_file(sp) != b["scores_test_sha256"]:', "        if sp.is_symlink() or not sp.is_file():"),
    (S, "N1 score rows not compared with cnn_test",
     '                    or not (0 <= v <= 1) or r["prediction"] != ("1" if v > THRESHOLD else "0"):\n                fail(f"N1 seed{seed} scores',
     '                    or False:\n                fail(f"N1 seed{seed} scores'),
    (S, "N1 C1 flag not compared", 'or r["in_c1_subset"] != ("1" if r["sample_id"] in inp.by_key["c1"] else "0")', "or False"),
    (S, "frozen face SHA-256 not compared", '    if sha256_file(ff) != pre["frozen_face"]["sha256"]:', "    if False:"),
    (S, "detector resource SHA-256 not compared", "        elif sha256_file(p) != want:", "        elif False:"),
    (S, "reference code SHA-256 not compared", '        elif sha256_file(p) != pre["reference_code_sha256"][key]:', "        elif False:"),
    (S, "parity list SHA-256 and count not compared", '    if len(inp.parity_keys) != par["n_keys"] or inp.parity_sha != par["keys_sha256"]:', "    if False:"),
    (S, "parity list derived without the key order", '    for r in sorted(rows, key=lambda r: r["key"]):\n        by.setdefault', '    for r in rows:\n        by.setdefault'),
    (S, "code change since check accepted", "    if check.get(\"code_sha256\") != code_hashes(a):", "    if False:"),
    (S, "commit change for a non-trial run accepted", "    if not is_trial(d) and N1.repo_state(d).get(\"commit\") != check.get(\"commit\"):", "    if False:"),
    (S, "inputs changed after check accepted", "        if inp.checks.get(k) != check[\"checks\"].get(k):", "        if False:"),
    (S, "outputs recomputed in place", "    if found:\n        fail(f\"{d.name} already holds", "    if False:\n        fail(f\"{d.name} already holds"),
    (S, "files of a previous step (or of check) not compared", "        if q.is_symlink() or not q.is_file() or sha256_file(q) != h:", "        if False:"),
    (S, "check files not verified after check", '    verify_files(d, check, "check", "check.json")\n', '    pass\n'),
    (S, "step allowed outside trial runs", "    if a.step != 1 and not is_trial(d):", "    if False:"),
    (S, "different cache frames accepted", "    if total_diff or (a.step == 1 and compared != n_frames):", "    if False:"),
    (S, "duplicated derived frames accepted", "    if dup:\n        fail(f\"duplicated derived frames", "    if False:\n        fail(f\"duplicated derived frames"),
    (S, "parity frame inequality accepted", "    if not all(equal):", "    if False:"),
    (S, "parity tolerance ignored", '    bad = {s: v for s, v in seeds.items() if v["within_tolerance"] != v["n_keys"]}', "    bad = {}"),
    (S, "parity tolerance applied with <", '"within_tolerance": sum(1 for x in diffs if x <= tol)', '"within_tolerance": sum(1 for x in diffs if x <= tol * 10)'),
    (S, "score without authorization", '    if not a.i_am_authorized:\n        fail("score needs', '    if False:\n        fail("score needs'),
    (S, "seed outside the N1 seeds accepted", '    if a.train_seed not in cfg["n1_study"]["train_seeds"]:', "    if False:"),
    (S, "seed folder reused", "    if os.path.lexists(sd):\n        fail(f\"{sd.name} already exists", "    if False:\n        fail(f\"{sd.name} already exists"),
    (S, "score frames not compared with cache-check", '        if h != expected[r["key"]]["frame_sha256"] or det != expected[r["key"]]["detector"]:', "        if False:"),
    (S, "tie at 0.5 as attack in the records", '"score_attack": repr(v), "prediction": "1" if v > THRESHOLD else "0", "detector": det,',
     '"score_attack": repr(v), "prediction": "1" if v >= THRESHOLD else "0", "detector": det,'),
    (S, "aggregate accepts a seed selection", "    if a.train_seed is not None:\n        fail(\"aggregate takes every seed", "    if False:\n        fail(\"aggregate takes every seed"),
    (S, "aggregate does not compare the score files", "            if (sd / n).is_symlink() or not (sd / n).is_file() or sha256_file(sd / n) != h:", "            if False:"),
    (S, "aggregate does not compare rows with the manifests and frames",
     '            if any(r[k] != x for k, x in exp.items()) or not (0 <= v <= 1) or r["prediction"] != ("1" if v > THRESHOLD else "0"):', "            if False:"),
    (S, "aggregate does not recompute the metrics", "        if json.dumps(met, sort_keys=True) != json.dumps(stored, sort_keys=True):", "        if False:"),
    (S, "aggregate keeps previous outputs", '            q.unlink()\n            print(f"   removed previous {n}")', "            pass"),
    (S, "public file written despite the scan", '            fail(f"{name} would hold', '            print(f"{name} would hold'),
    (S, "figures not validated", '    if set(figs) != set(PUBLIC_FIGURES) | set(RESTRICTED_FIGURES) or not all(isinstance(v, bytes) and v[:8] == b"\\x89PNG\\r\\n\\x1a\\n" for v in figs.values()):',
     "    if False:"),
    (S, "absent class counted as 0 in the strata", '"n": str(len(sel)), "errors": str(err), "error_rate": N1.fmt_na(err / len(sel) if sel else None)})',
     '"n": str(len(sel)), "errors": str(err), "error_rate": N1.fmt_na(err / len(sel) if sel else 0.0)})'),
    (S, "strata summary ignores NA", '                if r["error_rate"] == NA or q is None or q["error_rate"] == NA:', "                if q is None:"),
    (S, "histogram bins changed", "counts, _ = np.histogram(s, bins=BINS, range=(0.0, 1.0))", "counts, _ = np.histogram(s, bins=BINS, range=(0.0, 0.9))"),
    (S, "group of a score row taken from the record instead of the manifests", '                   "group": inp.group[m["subject"]], "in_m1_dev":', '                   "group": r["group"], "in_m1_dev":'),
    (S, "validation subjects set not separated", '            "sessions_01_02_validation_subjects": [r for r in rows if r["group"] == "validation"],',
     '            "sessions_01_02_validation_subjects": [r for r in rows if r["group"] in ("validation", "fitting")],'),
    (S, "two seeds with the same weights accepted", '    if len({v["weights_sha256"] for v in seeds.values()}) != len(seeds):', "    if False:"),
    (S, "non-finite probabilities accepted (N1 check bypassed)", "    p = N1.probabilities(scorer_for(inp, seed), frames)[:, 1]\n    rows, values = [], []",
     "    p = np.asarray(scorer_for(inp, seed)(frames), dtype=np.float64)[:, 1]\n    rows, values = [], []"),
    ("scripts/check_n2_mutations.py", "harness counts a crash as killed", "    if f + e > 0 and rc != 0:\n        return \"fail\", line",
     "    if rc != 0:\n        return \"fail\", line"),
    ("scripts/check_n2_mutations.py", "harness accepts a missing summary line", '    if len(m) != 1:\n        return "error", f"no single summary line (exit code {rc})"',
     '    if len(m) != 1:\n        return ("fail" if rc else "pass"), "no summary line"'),
]


def copy_tree(dst: Path) -> None:
    """Copia del codice e, se c'è, della configurazione consegnata con il face.py congelato (delivery/n2/ nel repository
    di sviluppo, ../ alla radice del pacchetto nel repository del gruppo), per test_delivered_configuration."""
    for f in FILES:
        (dst / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dst / f)
    for c in (ROOT / "delivery" / "n2", ROOT.parent):
        if (c / "n2_sessions.json").is_file() and (c / "face_0ac7143.py").is_file():
            (dst / "delivery" / "n2").mkdir(parents=True, exist_ok=True)
            for n in ("n2_sessions.json", "face_0ac7143.py"):
                shutil.copy2(c / n, dst / "delivery" / "n2" / n)
            break


SUMMARY = re.compile(r"^tests run: (\d+), failures: (\d+), errors: (\d+), skipped: (\d+)$", re.M)


def classify(rc: int, out: str, expected_tests: int = None) -> tuple:
    """("pass", riga) con riepilogo a 0 fallimenti, 0 errori e codice 0; ("fail", riga) con fallimenti o errori e codice
    diverso da 0; ("error", motivo) in ogni altro caso (mai contato come mutazione rilevata)."""
    m = SUMMARY.findall(out)
    if len(m) != 1:
        return "error", f"no single summary line (exit code {rc})"
    n, f, e, _ = map(int, m[0])
    line = f"tests run: {n}, failures: {f}, errors: {e}"
    if expected_tests is not None and n != expected_tests:
        return "error", f"{line}: {n} tests instead of {expected_tests}"
    if f + e == 0 and rc == 0:
        return "pass", line
    if f + e > 0 and rc != 0:
        return "fail", line
    return "error", f"{line} with exit code {rc}"


def run_suite(tree: Path) -> tuple:
    env = {k: v for k, v in os.environ.items() if not k.startswith("LIVENESSLAB_")}
    r = subprocess.run([sys.executable, str(tree / "scripts" / "check_n2_sessions.py")], cwd=tree, env=env, capture_output=True, text=True, timeout=1800)
    return r.returncode, r.stdout + r.stderr


CONTROL = [(S, "control: comment only in n2_sessions.py", "def frame_hash(arr) -> str:\n", "def frame_hash(arr) -> str:  # control mutant: comment only\n")]


def judge(base: Path, tag: str, mutation=None, expected_tests: int = None) -> tuple:
    tree = base / tag
    copy_tree(tree)
    if mutation is not None:
        f, old, new = mutation
        src = (ROOT / f).read_text(encoding="utf-8")
        (tree / f).write_text(src.replace(old, new), encoding="utf-8")
    try:
        return classify(*run_suite(tree), expected_tests)
    finally:
        shutil.rmtree(tree, ignore_errors=True)


def main() -> int:
    base = Path(tempfile.mkdtemp(prefix="n2-mutations-"))
    try:
        bad_decl = [f"{what}: the original text occurs {(ROOT / f).read_text(encoding='utf-8').count(old)} times in {f}"
                    for f, what, old, _ in MUTATIONS + CONTROL if (ROOT / f).read_text(encoding="utf-8").count(old) != 1]
        if bad_decl:
            for b in bad_decl:
                print(f"DECLARATION ERROR: {b}")
            print("RESULT: FAIL")
            return 1
        state, line = judge(base, "unmutated-before")
        if state != "pass":
            print(f"ERROR: the unmutated suite does not pass in a fresh copy ({line}): no mutation can be judged")
            print("RESULT: FAIL")
            return 1
        n_tests = int(re.match(r"tests run: (\d+)", line).group(1))
        print(f"unmutated copy (before): PASS ({line})")
        killed, bad, errors = 0, [], []
        for i, (f, what, old, new) in enumerate(MUTATIONS, 1):
            state, line = judge(base, f"m{i:02d}", (f, old, new), n_tests)
            if state == "fail":
                killed += 1
                print(f"  {i:2d} killed    {f.split('/')[-1]}: {what} ({line})")
            elif state == "pass":
                bad.append(f"{i} {what}")
                print(f"  {i:2d} SURVIVED  {f.split('/')[-1]}: {what}")
            else:
                errors.append(f"{i} {what}: {line}")
                print(f"  {i:2d} ERROR     {f.split('/')[-1]}: {what} ({line})")
        survived_control = 0
        for f, what, old, new in CONTROL:
            state, line = judge(base, "control", (f, old, new), n_tests)
            survived_control += state == "pass"
            print(f"  {what}: {'survived (as required)' if state == 'pass' else 'NOT SURVIVED (' + line + ')'}")
        state, line = judge(base, "unmutated-after", None, n_tests)
        after = state == "pass"
        print(f"unmutated copy (after): {'PASS' if after else 'FAIL'} ({line})")
        print(f"mutations killed: {killed}/{len(MUTATIONS)}")
        print(f"control mutations survived: {survived_control}/{len(CONTROL)}")
        for b in bad:
            print(f"NOT KILLED: {b}")
        for b in errors:
            print(f"HARNESS ERROR (not counted as killed): {b}")
        ok = killed == len(MUTATIONS) and survived_control == len(CONTROL) and after and not errors
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
