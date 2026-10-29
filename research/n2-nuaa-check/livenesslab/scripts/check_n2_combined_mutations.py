"""
Mutazioni dichiarate di scripts/n2_combined.py: la suite scripts/check_n2_combined.py deve fallire per ciascuna
(stesso programma di check_n1_mutations.py e check_n2_mutations.py: copia nuova per mutazione, suite non mutata prima
e dopo, mutazione di controllo che sopravvive, classify che conta solo i fallimenti reali dei test).

  python scripts/check_n2_combined_mutations.py   # righe finali: mutations killed: N/N, control mutations survived: 1/1, RESULT: PASS / FAIL

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
FILES = ["scripts/n2_combined.py", "scripts/n2_sessions.py", "scripts/n1_train.py", "scripts/check_n2_combined.py", "scripts/check_n2_combined_mutations.py",
         "scripts/nuaa_manifests.py", "scripts/select_threshold.py", "scripts/path_guard.py", "scripts/m2_writers.py", "scripts/package_rules.py",
         "src/tesi_app/__init__.py", "src/tesi_app/paths.py", "src/tesi_app/evaluation.py", "src/tesi_app/core.py", "src/tesi_app/face.py"]
S = "scripts/n2_combined.py"

MUTATIONS = [
    (S, "delivered configuration SHA-256 not compared", "    if cfg_sha != CONFIG_SHA256:", "    if False:"),
    (S, "config key set not compared", "    if set(cfg) != want:", "    if False:"),
    (S, "classes and input shape not fixed", '            or mdl.get("input_shape") != list(TARGET_SIZE) + [3] or mdl.get("classes") != {"0": "bona fide", "1": "attack"}:', "            or False:"),
    (S, "variant order not fixed", "    if not isinstance(v, dict) or tuple(v) != VARIANTS:", "    if not isinstance(v, dict) or set(v) != set(VARIANTS):"),
    (S, "fallback rule not required", '        if not isinstance(fc.get("detector"), dict) or not N2._hex(fc["detector"].get("sha256")) or "fallback" not in fc:',
     '        if not isinstance(fc.get("detector"), dict) or not N2._hex(fc["detector"].get("sha256")):'),
    (S, "threshold and decision rule not fixed", '    if not isinstance(m, dict) or m.get("threshold") != THRESHOLD or m.get("decision") != DECISION_RULE or m.get("sets") != list(SETS):',
     '    if not isinstance(m, dict) or m.get("sets") != list(SETS):'),
    (S, "study folder name not checked", "    if not RUN_ID_RE.match(d.name):", "    if False:"),
    (S, "protected repository folders accepted", 'if msg := pg.deletion_problem(d, ROOT, pg.BASE_PROTECTED, [ROOT / "results" / "n2"]):', "if False:"),
    (S, "symbolic link study folder accepted", "    if d.is_symlink():\n        return f\"{d.name} is a symbolic link", "    if False:\n        return f\"{d.name} is a symbolic link"),
    (S, "inputs inside the study accepted", "        if pg.within(p, d) or pg.contains(p, d):", "        if False:"),
    (S, "study folder need not be new", "    if d.exists() and any(d.iterdir()):", "    if False:"),
    (S, "non-trial study id without the commit", "    if not is_trial(d) and not str(state.get(\"commit\", \"\")).startswith(m.group(2)):", "    if False:"),
    (S, "manifest SHA-256 not compared", '        if got != e["sha256"]:', "        if False:"),
    (S, "rows with the same key not compared field by field", "            if inp.by_key[sup][k] != r:", "            if False:"),
    (S, "original SHA-256 not compared (locate bypassed)", "            N2.locate(Path(a.images), r, layouts)", "            layouts[\"images\"] += 1"),
    (S, "model size not compared", '        if mp.stat().st_size != cfg["model"]["bytes"]:', "        if False:"),
    (S, "model SHA-256 not compared", '        if sha256_file(mp) != cfg["model"]["sha256"]:', "        if False:"),
    (S, "current face.py SHA-256 not compared", '    if sha256_file(ROOT / "src" / "tesi_app" / "face.py") != cfg["variants"]["face_crop"]["detector"]["sha256"]:', "    if False:"),
    (S, "detector resource SHA-256 not compared", "        elif sha256_file(p) != want:", "        elif False:"),
    (S, "reference code SHA-256 not compared", '        elif sha256_file(p) != cfg["reference_code_sha256"][key]:', "        elif False:"),
    (S, "N1 aggregate SHA-256 not compared", '    if p.is_symlink() or not p.is_file() or sha256_file(p) != n1["aggregate_sha256"]:', "    if p.is_symlink() or not p.is_file():"),
    (S, "N1 aggregate threshold and sets not compared", "            or doc.get(\"threshold\") != THRESHOLD or doc.get(\"decision\") != DECISION_RULE or set(doc.get(\"sets\", {})) != set(SETS):",
     "            or False:"),
    (S, "code change since check accepted", "    if check.get(\"code_sha256\") != code_hashes():", "    if False:"),
    (S, "inputs changed after check accepted", "        if inp.checks.get(k) != check[\"checks\"].get(k):", "        if False:"),
    (S, "score without authorization", '    if not a.i_am_authorized:\n        fail("score needs', '    if False:\n        fail("score needs'),
    (S, "variant outside the two accepted", "    if a.variant not in VARIANTS:", "    if False:"),
    (S, "variant folder reused", "    if os.path.lexists(vd):", "    if False:"),
    (S, "frame shape not checked", "        if arr.shape != TARGET_SIZE + (3,) or arr.dtype != np.uint8:", "        if False:"),
    (S, "duplicated derived frames accepted", "    if len(set(hashes)) != len(hashes):", "    if False:"),
    (S, "tie at 0.5 as attack in the records", '"score_attack": repr(v), "prediction": "1" if v > THRESHOLD else "0", "in_c1_subset":',
     '"score_attack": repr(v), "prediction": "1" if v >= THRESHOLD else "0", "in_c1_subset":'),
    (S, "input not scaled to [-1, 1]", "xb = (X[i:i + N1.SCORE_BATCH].astype(np.float32) / 255.0 - 0.5) * 2.0", "xb = X[i:i + N1.SCORE_BATCH].astype(np.float32) / 255.0"),
    (S, "model SHA-256 not compared before loading", "    if sha256_file(model_path) != model_sha:", "    if False:"),
    (S, "enhancement before the resize", "    resized = cv2.resize(rgb, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)\n    return np.asarray(creator.advanced_image_enhancement(resized), dtype=np.uint8)",
     "    return np.asarray(cv2.resize(creator.advanced_image_enhancement(rgb), TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4), dtype=np.uint8)"),
    (S, "aggregate does not compare the score files", "            if (vd / n).is_symlink() or not (vd / n).is_file() or sha256_file(vd / n) != h:", "            if False:"),
    (S, "aggregate does not compare rows with the manifests",
     '            if any(r[k] != x for k, x in exp.items()) or not (0 <= v <= 1) or r["prediction"] != ("1" if v > THRESHOLD else "0"):', "            if False:"),
    (S, "aggregate does not recompute the metrics", "        if json.dumps(met, sort_keys=True) != json.dumps(stored, sort_keys=True):", "        if False:"),
    (S, "per-subject table not recomputed in aggregate", "        if N1.csv_text(subj_rows, SUBJECT_COLUMNS) != N1.csv_text(recomputed, SUBJECT_COLUMNS):", "        if False:"),
    (S, "detector counts not compared in aggregate", '        if sc.get("detectors") != dict(Counter(r["detector"] for r in rows)):', "        if False:"),
    (S, "aggregate keeps previous outputs", '            q.unlink()\n            print(f"   removed previous {n}")', "            pass"),
    (S, "public file written despite the scan", '            fail(f"{name} would hold', '            print(f"{name} would hold'),
    (S, "figure not validated", '    if set(figs) != {"figure_distributions.png"} or not all(isinstance(v, bytes) and v[:8] == b"\\x89PNG\\r\\n\\x1a\\n" for v in figs.values()):', "    if False:"),
    (S, "non-finite probabilities accepted (N1 check bypassed)", '    p = N1.probabilities((MAKE_SCORER or tf_model_scorer)(a.model, cfg["model"]["sha256"]), frames)[:, 1]',
     '    p = np.asarray((MAKE_SCORER or tf_model_scorer)(a.model, cfg["model"]["sha256"])(frames), dtype=np.float64)[:, 1]'),
    ("scripts/check_n2_combined_mutations.py", "harness counts a crash as killed", "    if f + e > 0 and rc != 0:\n        return \"fail\", line", "    if rc != 0:\n        return \"fail\", line"),
]


def copy_tree(dst: Path) -> None:
    for f in FILES:
        (dst / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dst / f)
    for c in (ROOT / "delivery" / "n2", ROOT.parent):
        if (c / "n2_combined.json").is_file():
            (dst / "delivery" / "n2").mkdir(parents=True, exist_ok=True)
            shutil.copy2(c / "n2_combined.json", dst / "delivery" / "n2" / "n2_combined.json")
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
    r = subprocess.run([sys.executable, str(tree / "scripts" / "check_n2_combined.py")], cwd=tree, env=env, capture_output=True, text=True, timeout=1800)
    return r.returncode, r.stdout + r.stderr


CONTROL = [(S, "control: comment only in n2_combined.py", "def metrics_from_rows(rows) -> dict:\n", "def metrics_from_rows(rows) -> dict:  # control mutant: comment only\n")]


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
    base = Path(tempfile.mkdtemp(prefix="n2c-mutations-"))
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
