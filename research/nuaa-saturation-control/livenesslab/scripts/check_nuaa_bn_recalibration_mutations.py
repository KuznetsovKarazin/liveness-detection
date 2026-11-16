"""
Controllo delle mutazioni di scripts/nuaa_bn_recalibration.py (ristima delle statistiche delle BatchNormalization),
offline e deterministico, con la stessa meccanica dei controlli precedenti (check_nuaa_session_ablation_run_mutations.py).

Copia il codice (lo script e la sua suite, il runner genitore e la sua suite, nuaa_folds_run.py e la sua suite, che
forniscono il mondo sintetico di base, m2_writers.py, path_guard.py, package_rules.py e, se c'è, la configurazione
consegnata delivery/bnrecal/nuaa_bn_recalibration.json) in una cartella temporanea, applica una mutazione dichiarata
alla volta (coppia cerca/sostituisci che deve comparire esattamente una volta: se un modello non si trova più, il
controllo si ferma con un errore) ed esegue scripts/check_nuaa_bn_recalibration.py sulla copia mutata. Una mutazione è
rilevata solo se la suite arriva al riepilogo («tests run») con almeno un fallimento o errore; una suite che non parte la
rende non valida. La mutazione di controllo (solo un commento) e le mutazioni dichiarate ridondanti devono sopravvivere.

Nessun dato reale: la suite usa solo il mondo sintetico (modelli Keras minuscoli e frame 8×8 in una cartella
temporanea). Le mutazioni girano una alla volta (macchina da 16 GB: un solo processo TensorFlow per volta), con la suite in modalità
failfast (-f): si ferma al primo fallimento; la copia non mutata, il mutante di controllo e i ridondanti girano completi.
Una mutazione può toccare più punti (tuple di modelli e sostituti, ognuno presente esattamente una volta).

  .venv/bin/python -I -B scripts/check_nuaa_bn_recalibration_mutations.py            # tutte le mutazioni, una alla volta
  .venv/bin/python -I -B scripts/check_nuaa_bn_recalibration_mutations.py --list     # solo l'elenco, con la verifica dei modelli
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
TESTS = "scripts/check_nuaa_bn_recalibration.py"
S = "scripts/nuaa_bn_recalibration.py"
COPY = ("scripts/nuaa_bn_recalibration.py", "scripts/check_nuaa_bn_recalibration.py", "scripts/nuaa_session_ablation_run.py",
        "scripts/check_nuaa_session_ablation_run.py", "scripts/nuaa_folds_run.py", "scripts/check_nuaa_folds_run.py",
        "scripts/m2_writers.py", "scripts/path_guard.py", "scripts/package_rules.py")
CONFIG = ("delivery", "bnrecal", "nuaa_bn_recalibration.json")
SKIP_OK = "missing input: delivery/bnrecal/nuaa_bn_recalibration.json is not present"
# (nome, file mutato, testo cercato, testo sostitutivo)
MUTANTS = [
    # momenti: unione dei batch, conteggi, varianza, precisione
    ("batch means merged unweighted", S, "        self.mean = self.mean + d * (nb / n)", "        self.mean = self.mean + d * 0.5"),
    ("between-batch term dropped (mean of the batch variances)", S, "        self.m2 = self.m2 + m2b + d * d * (self.n * nb / n)",
     "        self.m2 = self.m2 + m2b"),
    ("last batch counted as a full batch in the accumulator (dense rows)", S, "        n = self.n + nb\n", "        n = self.n + max(nb, 16 * (nb // x.shape[0] if x.shape[0] else 1))\n"),
    ("sample variance (divisor N - 1)", S, "        return self.mean, self.m2 / self.n", "        return self.mean, self.m2 / (self.n - 1)"),
    ("spatial dimensions not counted", S, "        x = x.reshape(-1, x.shape[-1])",
     "        x = x.mean(axis=tuple(range(1, x.ndim - 1))) if x.ndim > 2 else x"),
    ("float32 accumulation", S, "        x = np.asarray(a, dtype=np.float64)", "        x = np.asarray(a, dtype=np.float32)"),
    ("batch sum of squares not centred", S, "        m2b = ((x - mb) ** 2).sum(axis=0)", "        m2b = (x ** 2).sum(axis=0)"),
    # passaggio in inferenza, ingresso, logit
    ("training mode in the forward pass", S, "            res = sub(xb, training=False)", "            res = sub(xb, training=True)"),
    ("frames not divided by 255", S, "        xb = X[i:i + batch].astype(np.float32) / 255.0", "        xb = X[i:i + batch].astype(np.float32)"),
    ("logit bias dropped", S, '        out["z"] = np.concatenate(hs) @ W + b if hs else np.zeros((0, 2))',
     '        out["z"] = np.concatenate(hs) @ W if hs else np.zeros((0, 2))'),
    # ristima: ordine, tensori assegnati, cast
    ("layers re-estimated in reverse order", S, "    for n in bn_layers:\n        if budget is not None:",
     "    for n in reversed(bn_layers):\n        if budget is not None:"),
    ("moving_mean not assigned", S, "        layer.moving_mean.assign(m32)", "        pass"),
    ("gamma assigned instead of moving_variance", S, "        layer.moving_variance.assign(v32)", "        layer.gamma.assign(v32)"),
    ("cast through float16", S, "        m32, v32 = mean.astype(np.float32), var.astype(np.float32)",
     "        m32, v32 = mean.astype(np.float16).astype(np.float32), var.astype(np.float32)"),
    # confronto dei tensori
    ("changed tensors never recorded", S, '        if a["bytes"] != b["bytes"]:\n            differ.append(k)', '        if False:\n            differ.append(k)'),
    ("trained tensor changes ignored", S, "    extra = [f\"{k[0]}/{orig[k]['name']}\" for k in differ if k not in moving]", "    extra = []"),
    ("installed value not compared", S, '            if b["bytes"] != np.asarray(want, np.float32).tobytes():', "            if False:"),
    ("unchanged moving tensors accepted", S, "    if same:\n", "    if False:\n"),
    # coerenza
    ("consistency of the variances not checked", S,
     "        bad = int(np.sum(~(dm <= rel * np.abs(m) + abs_)) + np.sum(~(dv <= rel * np.abs(v) + abs_)))",
     "        bad = int(np.sum(~(dm <= rel * np.abs(m) + abs_)))"),
    ("consistency blind to NaN", S,
     "        bad = int(np.sum(~(dm <= rel * np.abs(m) + abs_)) + np.sum(~(dv <= rel * np.abs(v) + abs_)))",
     "        bad = int(np.sum(dm > rel * np.abs(m) + abs_) + np.sum(dv > rel * np.abs(v) + abs_))"),
    ("non-finite re-estimated statistics accepted", S, "        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(var))):", "        if False:"),
    ("non-finite values accepted in the flags", S, "    if any(v is None or not math.isfinite(v) for v in vals):", "    if False:"),
    ("consistency failure ignored by the run", S, '        if bad := {n: c["channels_outside_tolerance"] for n, c in cons.items() if c["channels_outside_tolerance"]}:',
     "        if bad := {}:"),
    ("consistency tolerances not read from the configuration", S, '    rel, abs_ = tol(cfg, "consistency_rel"), tol(cfg, "consistency_abs")',
     "    rel, abs_ = 1e9, 1e9"),
    # baseline: ogni tolleranza e la regola di arresto
    ("parent scorer not compared", S, '         "prob_pass": d_par <= pt and d_ev <= pt and dis_par <= dt and dis_ev <= dt,',
     '         "prob_pass": d_ev <= pt and dis_par <= dt and dis_ev <= dt,'),
    ("evaluator probabilities not compared", S, '         "prob_pass": d_par <= pt and d_ev <= pt and dis_par <= dt and dis_ev <= dt,',
     '         "prob_pass": d_par <= pt and dis_par <= dt and dis_ev <= dt,'),
    ("discordant decisions not counted", S, '         "prob_pass": d_par <= pt and d_ev <= pt and dis_par <= dt and dis_ev <= dt,',
     '         "prob_pass": d_par <= pt and d_ev <= pt,'),
    ("discordant decisions compared with themselves", S,
     "    dis_par = int(np.sum((p_parent > FR.FIXED_THRESHOLD) != (stored > FR.FIXED_THRESHOLD)))",
     "    dis_par = int(np.sum((p_parent > FR.FIXED_THRESHOLD) != (p_parent > FR.FIXED_THRESHOLD)))"),
    ("probability tolerance fixed in the code", S, '    pt, dt = tol(cfg, "dev_probability_max_abs_diff"), tol(cfg, "dev_decisions_discordant_at_0_5")',
     "    pt, dt = 1e-5, 0"),
    ("relative val_loss branch reintroduced", S, '"val_loss_pass": dvl <= va,', '"val_loss_pass": dvl <= va or dvl <= 0.01 * card["best_val_loss"],'),
    ("val_loss compared with val_loss_recomputed", S, '    dvl = abs(ev["val_loss"] - card["best_val_loss"])',
     '    dvl = abs(ev["val_loss"] - card["val_loss_recomputed"])'),
    ("val_loss tolerance fixed in the code", S, '    va = tol(cfg, "val_loss_abs")', "    va = 1e-3"),
    ("accuracy tolerance fixed in the code", S, '    acc_t = tol(cfg, "dev_accuracy_abs")', "    acc_t = 1e-6"),
    ("accuracy at 0.5 instead of argmax", S, '    dacc = abs(ev["accuracy"] - card["val_acc_at_best"])', '    dacc = abs(ev["accuracy_05"] - card["val_acc_at_best"])'),
    ("softmax check outside the baseline verdict", S,
     '    c["baseline_pass"] = bool(c["prob_pass"] and c["softmax_pass"] and c["val_loss_pass"] and c["accuracy_pass"])',
     '    c["baseline_pass"] = bool(c["prob_pass"] and c["val_loss_pass"] and c["accuracy_pass"])'),
    ("stop rule not applied", S, '            if failed:\n                status = "BASELINE_FAILED"', '            if False:\n                status = "BASELINE_FAILED"'),
    # criterio, gruppi, celle, indicatori, lettura
    ("accuracy threshold exclusive", S, '    a_ok = acc >= rc["accuracy_min"] if rc["accuracy_min_inclusive"] else acc > rc["accuracy_min"]',
     '    a_ok = acc > rc["accuracy_min"]'),
    ("val_loss threshold inclusive", S, '    l_ok = loss <= rc["val_loss_max"] if rc["val_loss_max_inclusive"] else loss < rc["val_loss_max"]',
     '    l_ok = loss <= rc["val_loss_max"]'),
    ("degenerate accuracy tolerance widened", S, 'abs(card["val_acc_at_best"] - dg["dev_accuracy"]) <= dg["within"]',
     'abs(card["val_acc_at_best"] - dg["dev_accuracy"]) <= 0.01'),
    ("degenerate epochs not read", S, '    if card["restored_epoch"] in dg["restored_epochs"] and', '    if card["restored_epoch"] <= 3 and'),
    ("apart group never assigned", S, '    if not meets(card["val_acc_at_best"], card["best_val_loss"], cfg["recovery_criterion"]):\n        return "apart"',
     '    if False:\n        return "apart"'),
    ("degenerate list not compared", S, '    if sorted(got["ten"]) != sorted(gr["ten"]["expected"]):', "    if False:"),
    ("control count not compared", S, '    if len(got["control"]) != gr["control"]["expected_count"]:', "    if False:"),
    ("2x2 cell with the dev before", S, '         "cell": cell_of(fit_b, dev_a),', '         "cell": cell_of(fit_b, dev_b),'),
    ("2x2 cell with the fitting after", S, '         "cell": cell_of(fit_b, dev_a),', '         "cell": cell_of(fit_a, dev_a),'),
    ("lost criterion never flagged", S, '"lost_criterion": dev_b and not dev_a,', '"lost_criterion": False,'),
    ("equal accuracy flagged as lower", S, '"dev_accuracy_lower": da["correct"] < db["correct"]', '"dev_accuracy_lower": da["correct"] <= db["correct"]'),
    ("val_loss flag reversed", S, '"val_loss_higher": da["val_loss"] > db["val_loss"]', '"val_loss_higher": da["val_loss"] < db["val_loss"]'),
    ("primary counts the criterion before", S, '    k, n = sum(r["dev_criterion_after"] for r in ten), len(ten)', '    k, n = sum(r["dev_criterion_before"] for r in ten), len(ten)'),
    ("case 'some' read as 'all'", S, '    case = "all" if n and k == n else "none" if k == 0 else "some"', '    case = "all" if k > 0 else "none"'),
    ("losses among the controls not listed", S, '    lose = [name_of(r) for r in ctrl if r["dev_criterion_before"] and not r["dev_criterion_after"]]', "    lose = []"),
    ("after-scores labelled as before", S, '+ score_rows(m, "dev", dev, dev_after, STATISTICS[1]) +', '+ score_rows(m, "dev", dev, dev_after, STATISTICS[0]) +'),
    ("dev after evaluated on the original", S, '        dev_after, _ = evaluate(copy, dev, batch, f"{name} dev after")', '        dev_after, _ = evaluate(orig, dev, batch, f"{name} dev after")'),
    ("fitting after evaluated on the original", S, '        fit_after, final = evaluate(copy, fit, batch, f"{name} fitting after", layers=bn)',
     '        fit_after, final = evaluate(orig, fit, batch, f"{name} fitting after", layers=bn)'),
    ("fitting before evaluated on the dev", S, '        ev_fit, _ = evaluate(model, fit, batch, f"{name} fitting")', '        ev_fit, _ = evaluate(model, dev, batch, f"{name} fitting")'),
    # ingressi, test, cartella, pubblico
    ("test role accepted", S, "    if role not in ROLES:", "    if False:"),
    ("test scores of the parent accepted", S, "    if suffix not in PARENT_SUFFIXES:", "    if False:"),
    ("fitting and dev keys not compared with the card", S,
     '        if card["fitting_keys_sha256"] != keys_sha(fit_rows) or card["dev_keys_sha256"] != keys_sha(dev_rows):', "        if False:"),
    ("weights SHA-256 of the card not compared", S, '        if card["weights_sha256"] != w:', "        if False:"),
    ("parent aggregate decoded", S, '    if sha256_bytes(ab) != sums[PUBLIC]["aggregate.json"]:',
     '    if SR.read_json(ab, "parent aggregate.json") is None or sha256_bytes(ab) != sums[PUBLIC]["aggregate.json"]:'),
    ("parent aggregate hash not compared", S, '    if sha256_bytes(ab) != sums[PUBLIC]["aggregate.json"]:', "    if False:"),
    ("folds run loaded with its public files", S, "    load_folds_frames_rows(pcfg, base, inp)", "    SR.load_folds_run(pcfg, base, inp)"),
    ("folds run extracted_frames.csv not compared", S, '    if sha256_bytes(eb) != listed["extracted_frames.csv"]:', "    if False:"),
    ("test statement of the configuration not compared", S, '    if cfg["test"] != TEST_READ:', "    if False:"),
    ("estimate fixed in the code", S, '"estimate_central_minutes": cfg["budget"]["estimate_minutes"],', '"estimate_central_minutes": 32,'),
    ("worsening size not reported", S, '    return {"before": b, "after": a, "difference": a - b}', '    return {"before": b, "after": a, "difference": 0}'),
    # punto 12 della revisione
    ("BN output instead of its input", S, "    outs = [model.get_layer(n).input for n in layers]", "    outs = [model.get_layer(n).output for n in layers]"),
    ("bn1_1 statistics before the ReLU", S, "    outs = [model.get_layer(n).input for n in layers]",
     '    outs = [(model.get_layer("relu1_1").input if n == "bn1_1" else model.get_layer(n).input) for n in layers]'),
    ("re-estimated on the dev", S, "        stats = recalibrate(copy, fit, bn, batch, budget, name)", "        stats = recalibrate(copy, dev, bn, batch, budget, name)"),
    ("criterion with or", S, "    return bool(a_ok and l_ok)", "    return bool(a_ok or l_ok)"),
    ("before taken from the recalibrated copy", S, '        before = {"dev": base[name]["dev"], "fit": base[name]["fit"]}',
     '        before = {"dev": dev_after, "fit": fit_after}'),
    ("baseline and recalibration in the same loop per model", S,
     ("            baseline_phase(cfg, inp, store, budget, base)\n",
      '            failed = [n for n in inp.order if not base[n]["checks"]["baseline_pass"]]',
      '        budget.check(f"recalibration {name}")\n'),
     ("            pass\n", "            failed = []",
      '        budget.check(f"recalibration {name}")\n        baseline_phase(cfg, __import__("types").SimpleNamespace(order=[name], '
      'parent=inp.parent, cells=inp.cells), store, budget, base)\n        if not base[name]["checks"]["baseline_pass"]:\n'
      '            raise SR.CapExceeded("baseline", 0.0)\n')),
    ("reading applied to INCOMPLETE", S, '        if status == "PASS":\n            reading = aggregate(cfg, rows)',
     '        if status in ("PASS", "INCOMPLETE"):\n            reading = aggregate(cfg, rows)'),
    ("results written with BASELINE_FAILED", S, '        if status in ("PASS", "INCOMPLETE") and phase == "recalibration":', "        if True:"),
    ("copy sharing the variables of the original", S, "        copy = tf.keras.models.clone_model(orig)\n        copy.set_weights(orig.get_weights())",
     "        copy = orig"),
    ("last batch weighted as a full batch in every layer", S, "            mom[n].add(np.asarray(a))",
     "            mom[n].add(np.asarray(a) if len(a) == batch else np.concatenate([np.asarray(a)] * (batch // len(a)) + [np.asarray(a)[:batch % len(a)]]))"),
    ("parent status not required PASS", S, 'if st.get("status") != "PASS" or st.get("run_id") != pr["run_id"]', 'if st.get("run_id") != pr["run_id"]'),
    ("parent SHA256SUMS not anchored", S, "        if sha256_bytes(b) != pr[key]:", "        if False:"),
    ("parent file SHA-256 not compared", S, "            if h != sums[RESTRICTED][fn]:", "            if False:"),
    ("note SHA-256 not compared", S, '        if h != cfg["note_sha256"]:', "        if False:"),
    ("null note accepted by check-inputs", S, "        if need_note:\n            fail(", "        if False:\n            fail("),
    ("dirty tree accepted", S, '    if state["dirty"]:', "    if False:"),
    ("existing run folder reused", S, "    if os.path.lexists(run):", "    if False:"),
    ("saved layers not compared", S, '    if got != cfg["architecture"]["layers"]:', "    if False:"),
    ("BatchNormalization facts not compared", S, '                or c["momentum"] != bn["momentum"] or c["epsilon"] != bn["epsilon"]:', "                or False:"),
    ("checkpoint not verified before loading", S, '    if sha256_file(path) != sha:\n        fail(f"{Path(path).name}: SHA-256 changed just before loading")',
     '    if False:\n        fail(f"{Path(path).name}: SHA-256 changed just before loading")'),
    ("checkpoints not verified at the end", S,
     '    changed = [n for n in inp.order if sha256_file(inp.parent[n]["weights"]) != inp.parent[n]["weights_sha256"]]', "    changed = []"),
    ("public files not scanned", S, '        if probs := FR.public_problems(text, inp, pname.endswith(".md")):', "        if False:"),
    ("authorisation flag not required", S, "    if not a.i_am_authorized:", "    if False:"),
    ("causal wording accepted", S, "    if banned:\n", "    if False:\n"),
]
# mutazione di controllo: solo un commento; deve SOPRAVVIVERE (prova che la suite rileva il comportamento, non il testo)
CONTROL = [("control: comment only", S, "def cell_of(fit_met: bool, dev_met: bool) -> str:\n",
            "def cell_of(fit_met: bool, dev_met: bool) -> str:  # control mutant: comment only\n")]
# controlli dichiarati ridondanti, che devono SOPRAVVIVERE:
REDUNDANT = [
    # il conteggio di Moments è verificato dalla suite (frame × H × W per conv, frame per il denso) e la guardia confronta
    # lo stesso numero derivato dagli stessi array: scatta solo se Moments è sbagliato, e le mutazioni di Moments sono
    # uccise dai test dei momenti
    ("declared redundant (must survive): count guard of the recalibration", S,
     '        if mom.n != len(part["X"]) * spatial:', "        if False:"),
    # set_weights copia ogni tensore: la copia appena fatta è sempre uguale all'originale
    ("declared redundant (must survive): copy equal to the original before the recalibration", S,
     "        if snapshot_weights(copy) != snap:", "        if False:"),
    # con una copia vera (clone_model) l'originale non cambia: la guardia non scatta; è lei a uccidere il mutante
    # «copy sharing the variables of the original» (copy = orig), quindi non è un controllo morto
    ("declared redundant (must survive): original unchanged after the recalibration", S,
     "        if snapshot_weights(orig) != snap:", "        if False:"),
    # lo schema dei fold non è mai caricato (nessuna chiamata a load_fold_manifests): inp.folds resta vuoto per costruzione
    ("declared redundant (must survive): fold schema never loaded", S, "    if inp.folds:", "    if False:"),
    # dopo la ristima logit e probabilità vengono dallo stesso passaggio e dalla stessa formula già controllata nella baseline
    ("declared redundant (must survive): softmax check after the recalibration", S, '            if rec["softmax_gap"] > sg:', "            if False:"),
]


def pairs(x) -> tuple:
    """Un modello o, per una mutazione in più punti, una tupla di modelli (applicati tutti)."""
    return x if isinstance(x, tuple) else (x,)


def copy_tree(dst: Path) -> Path:
    code = dst / "code"
    for rel in COPY:
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, code / rel)
    if ROOT.joinpath(*CONFIG).is_file():
        code.joinpath(*CONFIG[:-1]).mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT.joinpath(*CONFIG), code.joinpath(*CONFIG))
    return code


def run_tests(code: Path, tmp: Path, failfast: bool = False) -> tuple:
    """La suite sulla copia; con failfast si ferma al primo fallimento (solo per i mutanti: la copia non mutata, il
    mutante di controllo e i ridondanti girano completi)."""
    tmp.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp), "CUDA_VISIBLE_DEVICES": "-1"}
    r = subprocess.run([sys.executable, "-I", "-B", TESTS] + (["-f"] if failfast else []), cwd=code, env=env, capture_output=True,
                       text=True, timeout=3600)
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("tests run")), "no test summary")
    skips = tuple(sorted(l for l in r.stdout.splitlines() if l.startswith("skipped ")))
    return r.returncode, line, skips


def detected(rc: int, line: str, skips=(), base_skips=()) -> str:
    """killed solo per fallimenti o errori; survived solo con exit 0 e gli stessi salti della copia non mutata (con
    failfast una suite che passa esegue comunque tutti i test, quindi i salti sono confrontabili)."""
    m = re.match(r"tests run: (\d+), failures: (\d+), errors: (\d+), skipped: (\d+)", line)
    if not m:
        return "invalid"
    if int(m.group(2)) + int(m.group(3)) > 0:
        return "killed"
    return "survived" if rc == 0 and tuple(skips) == tuple(base_skips) else "invalid"


def one(tmp: Path, i: int, name: str, target: str, old, new, base_skips=(), failfast: bool = False) -> tuple:
    work = tmp / f"work{i}"
    code = copy_tree(work)
    f = code / target
    text = f.read_text(encoding="utf-8")
    for o, n in zip(pairs(old), pairs(new)):
        text = text.replace(o, n, 1)
    f.write_text(text, encoding="utf-8")
    rc, line, skips = run_tests(code, work / "tmp", failfast)
    shutil.rmtree(work, ignore_errors=True)
    return name, detected(rc, line, skips, base_skips), line, rc


def main() -> int:
    ap = argparse.ArgumentParser(description="Controllo delle mutazioni di nuaa_bn_recalibration.py (offline, sintetico)")
    ap.add_argument("--list", action="store_true", help="elenca le mutazioni e verifica i modelli, senza eseguire i test")
    a = ap.parse_args()
    problems = [f"{n}: the pattern occurs {k} times in {t} (expected exactly once)"
                for n, t, old, _ in MUTANTS + CONTROL + REDUNDANT for o in pairs(old)
                if (k := (ROOT / t).read_text(encoding="utf-8").count(o)) != 1]
    names = [n for n, *_ in MUTANTS + CONTROL + REDUNDANT]
    problems += [f"duplicated mutation name: {n}" for n in sorted({n for n in names if names.count(n) > 1})]
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
    with tempfile.TemporaryDirectory(prefix="bnrecal-mutations-") as tmp:
        tmp = Path(tmp)
        rc, line, base_skips = run_tests(copy_tree(tmp / "base"), tmp / "base" / "tmp")
        print(f"unmutated copy: {TESTS} exit {rc} ({line})", flush=True)
        for s in base_skips:
            print(f"unmutated copy: {s}", flush=True)
        if rc or not re.fullmatch(r"tests run: \d+, failures: 0, errors: 0, skipped: \d+", line):
            print("RESULT: FAIL (the tests do not pass on the unmutated copy)")
            return 1
        if any(s.split(": ", 1)[-1] != SKIP_OK for s in base_skips):
            print("RESULT: FAIL (the unmutated copy skips tests for a reason that is not a declared missing input)")
            return 1
        for i, (n, t, old, new) in enumerate(CONTROL + REDUNDANT):
            name, verdict, line, rc = one(tmp, len(MUTANTS) + i, n, t, old, new, base_skips)
            survived_controls += verdict == "survived"
            print(f"CONTROL   {verdict.upper():<9} {name} ({line})", flush=True)
        for i, (n, t, old, new) in enumerate(MUTANTS):
            name, verdict, line, rc = one(tmp, i, n, t, old, new, base_skips, failfast=True)
            killed += verdict == "killed"
            invalid += verdict == "invalid"
            print(f"{verdict.upper():<9} {name} ({line})", flush=True)
            if verdict == "invalid":
                print(f"ERROR: invalid mutant {name!r}: the test suite did not run to a summary with failures, or skipped other tests (exit {rc})",
                      file=sys.stderr)
    print(f"mutations killed: {killed}/{len(MUTANTS)}" + (f" ({invalid} invalid)" if invalid else ""))
    print(f"control mutations survived: {survived_controls}/{len(CONTROL) + len(REDUNDANT)} ({len(CONTROL)} control, {len(REDUNDANT)} declared redundant)")
    ok = killed == len(MUTANTS) and survived_controls == len(CONTROL) + len(REDUNDANT)
    print("RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
