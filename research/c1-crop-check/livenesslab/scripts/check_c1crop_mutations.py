"""
Mutazioni dichiarate di scripts/c1crop_check.py: la suite scripts/check_c1crop.py deve fallire per ciascuna
(stesso programma di check_n1_mutations.py, check_n2_mutations.py e check_n2_combined_mutations.py: copia nuova per mutazione, suite non mutata prima
e dopo, mutazione di controllo che sopravvive, classify che conta solo i fallimenti reali dei test).

  python scripts/check_c1crop_mutations.py   # righe finali: mutations killed: N/N, control mutations survived: 1/1, RESULT: PASS / FAIL

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
FILES = ["scripts/c1crop_check.py", "scripts/n2_sessions.py", "scripts/n1_train.py", "scripts/train_cnn.py", "scripts/check_c1crop.py", "scripts/check_c1crop_mutations.py",
         "scripts/nuaa_manifests.py", "scripts/select_threshold.py", "scripts/path_guard.py", "scripts/m2_writers.py", "scripts/package_rules.py",
         "src/tesi_app/__init__.py", "src/tesi_app/paths.py", "src/tesi_app/evaluation.py", "src/tesi_app/core.py", "src/tesi_app/face.py"]
S = "scripts/c1crop_check.py"

MUTATIONS = [
    (S, "delivered configuration SHA-256 not compared", "    if cfg_sha != CONFIG_SHA256:", "    if False:"),
    (S, "config key set not compared", "    if set(cfg) != want:", "    if False:"),
    (S, "architecture not fixed", '    if cfg["architecture"] != ARCH or cfg["dataset"] != "nuaa" or cfg["protocol"] != "official":', "    if False:"),
    (S, "classes not fixed", '            or w.get("classes") != {"0": "bona fide", "1": "attack"}:', "            or False:"),
    (S, "extractor order not fixed", "    if not isinstance(ex, dict) or tuple(ex) != EXTRACTORS:", "    if not isinstance(ex, dict) or set(ex) != set(EXTRACTORS):"),
    (S, "identical extractors accepted", '        if not out and ex[EXTRACTORS[0]]["sha256"] == ex[EXTRACTORS[1]]["sha256"]:', "        if False:"),
    (S, "threshold and decision rule not fixed", '    if not isinstance(m, dict) or m.get("threshold") != THRESHOLD or m.get("decision") != DECISION_RULE or m.get("set") != SET:',
     '    if not isinstance(m, dict) or m.get("set") != SET:'),
    (S, "parity tolerance not bounded", '            or not isinstance(par.get("tolerance_abs"), float) or not (0 < par["tolerance_abs"] <= 1e-5) or "rule" not in par:',
     '            or not isinstance(par.get("tolerance_abs"), float) or "rule" not in par:'),
    (S, "X_test schema not fixed", '            or c["members"]["X_test"] != {"dtype": "|u1", "shape": [c["per_class"] * 2] + list(TARGET_SIZE) + [3]} or "rule" not in c:',
     '            or "rule" not in c:'),
    (S, "study folder name not checked", "    if not RUN_ID_RE.match(d.name):", "    if False:"),
    (S, "protected repository folders accepted", 'if msg := pg.deletion_problem(d, ROOT, pg.BASE_PROTECTED, [ROOT / "results" / "c1crop"]):', "if False:"),
    (S, "symbolic link study folder accepted", "    if d.is_symlink():\n        return f\"{d.name} is a symbolic link", "    if False:\n        return f\"{d.name} is a symbolic link"),
    (S, "inputs inside the study accepted", "        if pg.within(p, d) or pg.contains(p, d):", "        if False:"),
    (S, "study folder need not be new", "    if d.exists() and any(d.iterdir()):", "    if False:"),
    (S, "non-trial study id without the commit", "    if not is_trial(d) and not str(state.get(\"commit\", \"\")).startswith(m.group(2)):", "    if False:"),
    (S, "manifest SHA-256 not compared", '        if got != e["sha256"]:', "        if False:"),
    (S, "rows with the same key not compared field by field", "            if inp.by_key[sup][k] != r:", "            if False:"),
    (S, "sampling order not compared with cnn_test", "    if inp.order != nm.spread_keys(inp.rows[\"cnn_test\"], k):", "    if False:"),
    (S, "original SHA-256 not compared (locate bypassed)", "            N2.locate(Path(a.images), r, layouts)", "            layouts[\"images\"] += 1"),
    (S, "cache SHA-256 not compared", '    if got != c["sha256"]:', "    if False:"),
    (S, "cache member schema not compared", '        if {"dtype": mem[k]["dtype"], "shape": mem[k]["shape"]} != want or mem[k]["fortran"]:', "        if False:"),
    (S, "weights size not compared", '    elif wp.stat().st_size != cfg["weights"]["bytes"]:', "    elif False:"),
    (S, "weights SHA-256 not compared", '    elif sha256_file(wp) != cfg["weights"]["sha256"]:', "    elif False:"),
    (S, "extractor copy SHA-256 not compared", '        elif sha256_file(p) != e["sha256"]:', "        elif False:"),
    (S, "current face.py not compared with the C1 copy", '    if not cur.is_file() or sha256_file(cur) != cfg["extractors"][EXTRACTORS[1]]["sha256"]:', "    if False:"),
    (S, "detector resource SHA-256 not compared", "        elif sha256_file(p) != want:", "        elif False:"),
    (S, "reference code SHA-256 not compared", '        elif sha256_file(p) != cfg["reference_code_sha256"][key]:', "        elif False:"),
    (S, "C1 scores SHA-256 not compared", '    if got != r["sha256"]:\n        fail(f"C1 scores', '    if False:\n        fail(f"C1 scores'),
    (S, "C1 score rows not bound to the manifest", '        if x["run_id"] != r["run_id"] or x["dataset_id"] != "nuaa" or x["split"] != "test" or x["subject_id"] != m["subject"] \\',
     '        if False or x["subject_id"] != m["subject"] \\'),
    (S, "C1 score keys not compared with the c1 manifest", "    if sorted(keys) != inp.c1_keys:", "    if False:"),
    (S, "code change since check accepted", "    if check.get(\"code_sha256\") != code_hashes(a):", "    if False:"),
    (S, "inputs changed after check accepted", "        if inp.checks.get(k) != check[\"checks\"].get(k):", "        if False:"),
    (S, "extractor SHA-256 not compared before loading", "    if sha256_file(p) != sha:\n        fail(f\"extractor {name}", "    if False:\n        fail(f\"extractor {name}"),
    (S, "frame shape not checked", "        if arr.shape != TARGET_SIZE + (3,) or arr.dtype != np.uint8:", "        if False:"),
    (S, "cache chain not reconstructed accepted", '    if not all(c0["equal"]):', "    if False:"),
    (S, "duplicated derived frames accepted", '    if len(set(c0["hashes"])) != len(c0["hashes"]) or len(set(c1["hashes"])) != len(c1["hashes"]):', "    if False:"),
    (S, "frames outputs recomputed in place", '    N2.refuse_existing(d, FRAMES_OUTPUTS, "frames")', "    pass"),
    (S, "frames of parity and score not compared with frames.csv", '        if h != expected[k]["frame_sha256"] or det != expected[k]["detector"]:', "        if False:"),
    (S, "parity outside the tolerance accepted", "    if within != len(diffs) or same_dec != len(diffs):", "    if False:"),
    (S, "parity tolerance taken from the code", '    tol = cfg["c1_reference"]["parity"]["tolerance_abs"]', "    tol = 1.0"),
    (S, "score without authorization", '    if not a.i_am_authorized:\n        fail("score needs', '    if False:\n        fail("score needs'),
    (S, "score outputs recomputed in place", '    N2.refuse_existing(d, SCORE_OUTPUTS, "score")', "    pass"),
    (S, "parity not bound to frames in score", '    if par.get("frames_sha256") != sha256_file(d / "frames.json"):\n        fail("parity.json does not belong',
     '    if False:\n        fail("parity.json does not belong'),
    (S, "tie at 0.5 as attack", '    return "1" if v > THRESHOLD else "0"', '    return "1" if v >= THRESHOLD else "0"'),
    (S, "decision change direction swapped", '    return CHANGES[1] if pred_c1 == "1" else CHANGES[2]', '    return CHANGES[2] if pred_c1 == "1" else CHANGES[1]'),
    (S, "input not scaled by 255", "            xb = X[i:i + batch].astype(np.float32) / 255.0", "            xb = X[i:i + batch].astype(np.float32)"),
    (S, "weights SHA-256 not compared before loading", "    if sha256_file(weights_path) != weights_sha:", "    if False:"),
    (S, "non-finite probabilities accepted (N1 check bypassed)", '    p = N1.probabilities((MAKE_SCORER or tf_scorer)(a.weights, cfg["weights"]["sha256"]), frames)[:, 1]',
     '    p = np.asarray((MAKE_SCORER or tf_scorer)(a.weights, cfg["weights"]["sha256"])(frames), dtype=np.float64)[:, 1]'),
    (S, "abs diff bins miscounted", '        bins[f"({lo:g}, {hi:g}]"] = sum(1 for x in ad if lo < x <= hi)', '        bins[f"({lo:g}, {hi:g}]"] = sum(1 for x in ad if lo <= x <= hi)'),
    (S, "aggregate does not verify the chain of records", '    if par.get("frames_sha256") != sha256_file(d / "frames.json") or sc.get("parity_sha256") != sha256_file(d / "parity.json"):',
     "    if False:"),
    (S, "aggregate does not compare rows with the manifests and frames",
     '        if any(r[k] != x for k, x in exp.items()) or not (0 <= v0 <= 1 and 0 <= v1 <= 1) or float(r["abs_diff"]) != abs(v0 - v1) or float(r["signed_diff"]) != v0 - v1:',
     "        if False:"),
    (S, "aggregate does not recompute the metrics", '    if json.dumps(met, sort_keys=True) != json.dumps(stored.get("metrics"), sort_keys=True) or json.dumps(paired, sort_keys=True) != json.dumps(stored.get("paired"), sort_keys=True):',
     "    if False:"),
    (S, "per-subject table not recomputed in aggregate", "    if N1.csv_text(subj, SUBJECT_COLUMNS) != N1.csv_text(subject_rows(d.name, rows), SUBJECT_COLUMNS):", "    if False:"),
    (S, "detector counts not compared in aggregate", '    if sc.get("detectors") != {n: dict(Counter(r[c] for r in rows)) for n, c in zip(EXTRACTORS, ("detector_cache", "detector_c1"))}:', "    if False:"),
    (S, "aggregate keeps previous outputs", '            q.unlink()\n            print(f"   removed previous {n}")', "            pass"),
    (S, "public file written despite the scan", '            fail(f"{name} would hold', '            print(f"{name} would hold'),
    (S, "figure not validated", '    if set(figs) != {"figure_paired.png"} or not all(isinstance(v, bytes) and v[:8] == b"\\x89PNG\\r\\n\\x1a\\n" for v in figs.values()):', "    if False:"),
    (S, "rows outside the official test split accepted", '        if any(r["official_split"] != "test" for r in rows):', "        if False:"),
    (S, "duplicated keys in a manifest accepted", '            if len({r[dim] for r in rows}) != len(rows):', "            if False:"),
    (S, "cache member set not compared", "    if set(mem) != set(CACHE_MEMBERS):", "    if False:"),
    (S, "frames.csv key set not compared", "        if sorted(out[n]) != inp.c1_keys:", "        if False:"),
    (S, "unknown extractor in frames.csv accepted", '        if r["extractor"] not in out:', "        if False:"),
    (S, "parity decisions not compared", "    if within != len(diffs) or same_dec != len(diffs):", "    if within != len(diffs):"),
    (S, "frames.csv not verified at score (frames.json record not reloaded)", '    N2.load_step(d, "frames.json", "frames", d.name, cfg_sha, check_sha)\n    par = N2.load_step(d, "parity.json"',
     '    par = N2.load_step(d, "parity.json"'),
    (S, "study path that is a file accepted", "    if d.exists() and not d.is_dir():", "    if False:"),
    (S, "parity.json checkpoint not compared in aggregate", '    if sc.get("weights_sha256") != cfg["weights"]["sha256"] or par.get("weights_sha256") != cfg["weights"]["sha256"]:',
     '    if sc.get("weights_sha256") != cfg["weights"]["sha256"]:'),
    (S, "input folders not checked (links or files accepted)", '        if kind == "dir" and (p.is_symlink() or not p.is_dir()):', "        if False:"),
    (S, "symbolic link manifest accepted", "        if p.is_symlink() or not p.is_file():\n            fail(f\"manifest", "        if False:\n            fail(f\"manifest"),
    (S, "weights file name not compared", '    if wp.name != cfg["weights"]["file"]:', "    if False:"),
    (S, "cache file name not compared", '    if cp.name != c["file"]:', "    if False:"),
    (S, "C1 scores file name not compared", '    if p.name != r["file"]:', "    if False:"),
    (S, "dirty work tree accepted for a non-trial run", '    if not is_trial(d) and state.get("dirty"):', "    if False:"),
    (S, "required C1 columns not checked", '            or not isinstance(r.get("columns"), list) or not set(C1_REQUIRED_COLUMNS) <= set(r["columns"]) or not isinstance(par, dict) \\',
     '            or not isinstance(r.get("columns"), list) or not isinstance(par, dict) \\'),
    ("scripts/check_c1crop_mutations.py", "harness counts a crash as killed", "    if f + e > 0 and rc != 0:\n        return \"fail\", line", "    if rc != 0:\n        return \"fail\", line"),
]


def copy_tree(dst: Path) -> None:
    for f in FILES:
        (dst / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dst / f)
    for c in (ROOT / "delivery" / "c1crop", ROOT.parent):
        if (c / "c1crop.json").is_file():
            (dst / "delivery" / "c1crop").mkdir(parents=True, exist_ok=True)
            for n in ("c1crop.json", "face_0ac7143.py", "face_a80f6f9.py"):
                if (c / n).is_file():
                    shutil.copy2(c / n, dst / "delivery" / "c1crop" / n)
            break


SUMMARY = re.compile(r"^tests run: (\d+), failures: (\d+), errors: (\d+), skipped: (\d+)$", re.M)


def classify(rc: int, out: str, expected_tests: int = None) -> tuple:
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
    r = subprocess.run([sys.executable, str(tree / "scripts" / "check_c1crop.py")], cwd=tree, env=env, capture_output=True, text=True, timeout=1800)
    return r.returncode, r.stdout + r.stderr


CONTROL = [(S, "control: comment only in c1crop_check.py", "def metrics_from_rows(rows) -> dict:\n", "def metrics_from_rows(rows) -> dict:  # control mutant: comment only\n")]


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
    base = Path(tempfile.mkdtemp(prefix="c1crop-mutations-"))
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
