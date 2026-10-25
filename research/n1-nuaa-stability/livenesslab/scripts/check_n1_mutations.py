"""
Mutazioni dichiarate di scripts/n1_train.py, scripts/n1_trainer.py e degli script sul difetto della GPU: ogni
mutazione toglie o altera un controllo o una regola, e la suite scripts/check_n1_train.py deve fallire (mutazione
rilevata dal comportamento, non da un confronto del sorgente). Ogni mutazione si applica a una copia NUOVA del codice
in una cartella temporanea, tolta dopo l'esecuzione (nessun file lasciato da una mutazione può far fallire le
successive); la suite non mutata deve passare in una copia nuova prima e dopo le mutazioni, e una mutazione di
controllo (solo un commento) deve sopravvivere. Una mutazione è rilevata solo se la suite gira davvero e riporta
fallimenti o errori nella riga «tests run: N, failures: F, errors: E, skipped: S» (stesso N della suite non mutata);
un import fallito, un SyntaxError o una riga di riepilogo mancante sono errori del programma, mai mutazioni rilevate
(funzione classify). Il codice vero non cambia:
il testo da sostituire deve comparire esattamente una volta, altrimenti la mutazione è un errore della dichiarazione.

  python scripts/check_n1_mutations.py        # righe finali: mutations killed: N/N, control mutations survived: 1/1, RESULT: PASS / FAIL

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
FILES = ["scripts/n1_train.py", "scripts/n1_trainer.py", "scripts/check_n1_train.py", "scripts/check_n1_mutations.py", "scripts/n1_metal_relu_repro.py",
         "scripts/n1_metal_diagnostic.py", "scripts/n1_metal_checks.py", "scripts/nuaa_manifests.py",
         "scripts/select_threshold.py", "scripts/path_guard.py", "scripts/m2_writers.py", "scripts/package_rules.py",
         "src/tesi_app/__init__.py", "src/tesi_app/paths.py", "src/tesi_app/evaluation.py", "src/tesi_app/core.py"]
T, R = "scripts/n1_train.py", "scripts/n1_trainer.py"

MUTATIONS = [
    # (file, descrizione, testo originale, testo mutato)
    (T, "cache SHA-256 not compared", 'if inp.cache_sha != d["cache"]["sha256"]:', "if False:"),
    (T, "cache dtype/shape not compared", 'if got != want or mem[k]["fortran"]:', "if False:"),
    (T, "pickled members accepted", "if dtype.hasobject:", "if False:"),
    (T, "label array not linked", "not np.array_equal(y, exp_y)", "False"),
    (T, "subject array not linked", "not np.array_equal(s, exp_s)", "False"),
    (T, "split seed + 1 in check", 'n1_trainer.split_indices(inp.arrays["y_train"], inp.arrays["s_train"], cfg["split_seed"])',
     'n1_trainer.split_indices(inp.arrays["y_train"], inp.arrays["s_train"], cfg["split_seed"] + 1)'),
    (T, "rows with the same key not compared field by field", "if inp.by_key[sup][k] != r:", "if False:"),
    (T, "test sampling order not compared", 'if order_test != nm.spread_keys(inp.rows["cnn_test"], k):', "if False:"),
    (T, "derived frames not refused", "        if bad:\n", "        if False:\n"),
    (T, "frozen card values not compared", "if k in cfg and json.dumps(cfg[k], sort_keys=True) != json.dumps(v, sort_keys=True):", "if False:"),
    (T, "config key set not compared", "    if set(cfg) != want:", "    if False:"),
    (T, "non-trial accepts any data", "    if trial:\n        return []", "    if True:\n        return []"),
    (T, "study folder need not be new", "if d.exists() and any(d.iterdir()):", "if False:"),
    (T, "study folder name not checked", "    if not RUN_ID_RE.match(d.name):", "    if False:"),
    (T, "protected repository folders accepted", 'if msg := pg.deletion_problem(d, ROOT, pg.BASE_PROTECTED, [ROOT / "results" / "n1"]):',
     "if False:"),
    (T, "moved protected folders accepted", "        if pg.within(d, q) or (Path(q).exists() and pg.contains(d, q)):", "        if False:"),
    (T, "symbolic link study folder accepted", "    if d.is_symlink():\n        return f\"{d.name} is a symbolic link", "    if False:\n        return f\"{d.name} is a symbolic link"),
    (T, "inputs inside the study accepted", "if pg.within(p, d) or pg.contains(p, d):", "if False:"),
    (T, "seed folder reused", "    if os.path.lexists(sd):", "    if False:"),
    (T, "train without authorization", '    if not a.i_am_authorized:\n        fail("train', '    if False:\n        fail("train'),
    (T, "seed outside the configuration", "if a.train_seed not in cfg[\"train_seeds\"]:", "if False:"),
    (T, "code change since check accepted", 'if check.get("code_sha256") != code_hashes():', "if False:"),
    (T, "record of another study accepted", 'doc.get("run_id") != run_id or ', ""),
    (T, "weights SHA-256 not compared", "if sha256_file(p) != card.get(key):", "if False:"),
    (T, "previous checkpoints accepted", 'if card.get(key) in cfg["data"]["previous_checkpoints_sha256"]:', "if False:"),
    (T, "card fields not compared", "        if card.get(k) != v or type(card.get(k)) is not type(v):", "        if False:"),
    (T, "restored epoch not compared", 'if card.get("restored_epoch") != best or card.get("best_val_loss") != vl[best - 1]:',
     'if card.get("best_val_loss") != vl[best - 1]:'),
    (T, "patience rule not checked", 'if n < t["max_epochs"] and not fired:', "if False:"),
    (T, "epoch budget not checked", '    if not 1 <= n <= t["max_epochs"]:', "    if False:"),
    (T, "tie at 0.5 as attack in the records", '"prediction": "1" if v > THRESHOLD else "0"', '"prediction": "1" if v >= THRESHOLD else "0"'),
    (T, "tie at 0.5 as attack in the metrics", "pred = s > THRESHOLD", "pred = s >= THRESHOLD"),
    (T, "val_loss of the checkpoint not compared", 'if not (diff <= VAL_LOSS_TOL[0] or diff <= VAL_LOSS_TOL[1] * train["best_val_loss"]):', "if False:"),
    (T, "scores out of [0, 1] accepted", "if p.shape != (len(X), 2) or not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):", "if False:"),
    (T, "scores recomputed in place", '    if found:\n        fail(f"{sd.name} already holds', '    if False:\n        fail(f"{sd.name} already holds'),
    (T, "score without authorization", '    if not a.i_am_authorized:\n        fail("score', '    if False:\n        fail("score'),
    (T, "absent class counted as 0 per subject", "bpcer = rej / len(bf) if bf else None", "bpcer = rej / max(len(bf), 1)"),
    (T, "aggregate does not recompute the metrics", "if json.dumps(met, sort_keys=True) != json.dumps(stored, sort_keys=True):", "if False:"),
    (T, "aggregate does not compare the score files", "if (sd / n).is_symlink() or not (sd / n).is_file() or sha256_file(sd / n) != h:", "if False:"),
    (T, "aggregate keeps previous outputs", '            q.unlink()\n            print(f"   removed previous {n}")', "            pass"),
    (T, "aggregate accepts a seed selection", "    if a.train_seed is not None:", "    if False:"),
    (T, "aggregate accepts identical weights", 'if len(weights) != len(cfg["train_seeds"]):', "if False:"),
    (T, "aggregate does not compare rows with the manifests", 'if any(r[k] != x for k, x in exp.items()) or not (0 <= v <= 1) or r["prediction"] != ("1" if v > THRESHOLD else "0"):',
     "if False:"),
    (T, "mean instead of median", '"median": statistics.median(values)', '"median": statistics.mean(values)'),
    (T, "public scan ignores subject identifiers", 'out.append("subject identifiers")', "pass"),
    (T, "public scan ignores image keys", 'out.append("image keys")', "pass"),
    (T, "public file written despite the scan", '            fail(f"{name} would hold', '            print(f"{name} would hold'),
    (T, "private NumPy API for the .npy headers", "return getattr(fmt, NPY_HEADERS[ver])(fh)", "return fmt._read_array_header(fh, ver)"),
    (T, ".npy format version not checked", "    if ver not in NPY_HEADERS:", "    if False:"),
    ("scripts/n1_metal_relu_repro.py", "reproducer without TensorFlow exits 0",
     'print("SKIP: TensorFlow not installed (missing input)")\n        return 2', 'print("SKIP: TensorFlow not installed (missing input)")\n        return 0'),
    ("scripts/n1_metal_checks.py", "checks without TensorFlow raise", "    try:\n        import tensorflow as tf\n    except ModuleNotFoundError:\n        print(\"SKIP: TensorFlow not installed (missing input)\")\n        return 2\n    if not tf.config.list_physical_devices(\"GPU\"):\n        print(\"SKIP: no GPU visible (missing input)\")\n        return 2\n    if a.disable_meta_optimizer:",
     "    import tensorflow as tf\n    if a.disable_meta_optimizer:"),
    ("scripts/n1_metal_checks.py", "c1-rescore without TensorFlow exits 0",
     '    except ImportError:\n        print("SKIP: TensorFlow not installed (missing input)")\n        return 2',
     '    except ImportError:\n        print("SKIP: TensorFlow not installed (missing input)")\n        return 0'),
    ("scripts/n1_metal_checks.py", "c1-rescore input errors not caught",
     "    except N.N1Error as e:\n        print(f\"ERROR: {e}\")\n        return 1\n    try:\n        import tensorflow",
     "    except ValueError as e:\n        print(f\"ERROR: {e}\")\n        return 1\n    try:\n        import tensorflow"),
    (T, "aggregate accepts records of another seed", '        if train.get("train_seed") != seed or sc.get("train_seed") != seed:', "        if False:"),
    (T, "aggregate accepts a score.json of another train.json",
     '        if sc.get("train_sha256") != sha256_file(sd / "train.json") or sc.get("weights_sha256") != train["weights_sha256"]:', "        if False:"),
    (T, "C1 flag not compared with the manifests", '                   "in_c1_subset": "1" if r["sample_id"] in inp.c1_keys else "0", "file_sha256": m["sha256"],',
     '                   "file_sha256": m["sha256"],'),
    (T, "per-subject table of the last seed only", "        subj_all += read_csv_rows(", "        subj_all = read_csv_rows("),
    (T, "range printed as maximum-minimum", "f\"{s['min']:.4f}-{s['max']:.4f}\"", "f\"{s['max']:.4f}-{s['min']:.4f}\""),
    (T, "summary with NA only when every seed is NA", "    if any(v is None for v in values):", "    if all(v is None for v in values):"),
    ("scripts/check_n1_mutations.py", "harness counts a crash as killed", "    if f + e > 0 and rc != 0:\n        return \"fail\", line",
     "    if rc != 0:\n        return \"fail\", line"),
    ("scripts/check_n1_mutations.py", "harness accepts a missing summary line", '    if len(m) != 1:\n        return "error", f"no single summary line (exit code {rc})"',
     '    if len(m) != 1:\n        return ("fail" if rc else "pass"), "no summary line"'),
    (R, "training seed not given to TensorFlow", "tf.random.set_seed(train_seed)", "tf.random.set_seed(42)"),
    (R, "training seed not given to the shuffle", ".shuffle(4096, seed=train_seed)", ".shuffle(4096, seed=42)"),
    (R, "split seed ignored by the trainer", "random_state=split_seed).split(", "random_state=42).split("),
    (R, "best weights not restored at the end of the epochs", "model.set_weights(es.best_weights)", "pass"),
    (R, "early stopping without restore", "restore_best_weights=True", "restore_best_weights=False"),
    (R, "patience not from the configuration", "patience=patience, restore", "patience=5, restore"),
    (R, "epochs not from the configuration", 'training["max_epochs"]', "10"),
    (R, "test frames accepted by the trainer", 'if set(data) != {"X_train", "y_train", "s_train"}:', "if False:"),
    (R, "GPU not hidden", '        tf.config.set_visible_devices([], "GPU")', "        pass"),
    (R, "visible devices not checked", '    if devices != ["CPU"]:', "    if False:"),
    (R, "split not checked by the trainer", "    check_split(tr_i, va_i, split_check)\n", "    pass\n"),
    (R, "existing checkpoint overwritten", "    if found:\n        raise SystemExit", "    if False:\n        raise SystemExit"),
]


def copy_tree(dst: Path) -> None:
    """Copia del codice e della configurazione consegnata. La configurazione sta in delivery/n1/ nel repository di
    sviluppo e alla radice del pacchetto (../n1_nuaa_stability.json rispetto a livenesslab/) nel repository del gruppo:
    serve a test_delivered_configuration, che senza di essa si salterebbe; si legge da quei due soli percorsi."""
    for f in FILES:
        (dst / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dst / f)
    for c in (ROOT / "delivery" / "n1" / "n1_nuaa_stability.json", ROOT.parent / "n1_nuaa_stability.json"):
        if c.is_file():
            (dst / "delivery" / "n1").mkdir(parents=True, exist_ok=True)
            shutil.copy2(c, dst / "delivery" / "n1" / c.name)
            break


SUMMARY = re.compile(r"^tests run: (\d+), failures: (\d+), errors: (\d+), skipped: (\d+)$", re.M)


def classify(rc: int, out: str, expected_tests: int = None) -> tuple:
    """Esito di un'esecuzione della suite: ("pass", riga) se la riga di riepilogo c'è con 0 fallimenti, 0 errori e
    codice 0; ("fail", riga) se c'è con fallimenti o errori e codice diverso da 0 (mutazione rilevata dai test);
    ("error", motivo) in ogni altro caso (import fallito, SyntaxError, riga mancante, codice incoerente, numero di test
    diverso da quello della suite non mutata): mai contato come mutazione rilevata."""
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
    r = subprocess.run([sys.executable, str(tree / "scripts" / "check_n1_train.py")], cwd=tree, env=env,
                       capture_output=True, text=True, timeout=900)
    return r.returncode, r.stdout + r.stderr


# mutazione di controllo: solo un commento; deve sopravvivere (la suite non dipende dal testo del sorgente)
CONTROL = [(T, "control: comment only in n1_train.py", "def frames_of(inp: Inputs, split: str) -> np.ndarray:\n",
            "def frames_of(inp: Inputs, split: str) -> np.ndarray:  # control mutant: comment only\n")]


def judge(base: Path, tag: str, mutation=None, expected_tests: int = None) -> tuple:
    """Esito (classify) della suite in una copia NUOVA del codice (nessun file lasciato da un'esecuzione precedente),
    con la mutazione applicata, se c'è."""
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
    base = Path(tempfile.mkdtemp(prefix="n1-mutations-"))
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
        n_tests = int(re.match(r"tests run: (\d+)", line).group(1))      # ogni copia mutata deve eseguire gli stessi test
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
