"""
Controllo delle mutazioni di scripts/pad_anet_run.py (i sei run C5: configurazione, file fissati, regola della macchina,
agganci, copie dichiarate, fasi 1 e 2 con la regola (n), uscite), offline e deterministico, con la meccanica di
check_pad_anet_comparison_mutations.py.

Copia il codice (il modulo, la suite scripts/check_pad_anet_run.py, i moduli della PR A e di C2 importati, le suite che
forniscono il mondo sintetico, delivery/anetrun/make_config.py e delivery/anetcmp/make_config.py, la configurazione della
PR #27, src/livedetection/src/architectures.py e lo strumento tools/run_without_tensorflow.py) in una cartella
temporanea, applica una mutazione dichiarata alla volta (coppia cerca/sostituisci che deve comparire esattamente una
volta) ed esegue la suite sulla copia mutata. La configurazione consegnata NON è copiata: il suo test è saltato con il
motivo dichiarato (SKIP_OK). Una mutazione è rilevata solo se la suite arriva al riepilogo con almeno un fallimento o
errore. La mutazione di controllo e quelle dichiarate ridondanti devono sopravvivere.

Modi: «offline» (TensorFlow nascosto: il flusso dei run gira con componenti finti) oppure «tf» (la suite completa, con
il run vero su 8×8).

  .venv/bin/python -I -B scripts/check_pad_anet_run_mutations.py                 # tutte, una alla volta
  .venv/bin/python -I -B scripts/check_pad_anet_run_mutations.py --list          # elenco e verifica dei modelli
  .venv/bin/python -I -B scripts/check_pad_anet_run_mutations.py --only 3,7      # solo queste (da 1); c = controllo e ridondanti
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
TESTS = "scripts/check_pad_anet_run.py"
TOOL = "tools/run_without_tensorflow.py"
TOOL_SOURCES = ("tools/run_without_tensorflow.py", "delivery/pad-attacknet-run/livenesslab/tools/run_without_tensorflow.py",
                "delivery/pad-attacknet-comparison/livenesslab/tools/run_without_tensorflow.py",
                "delivery/pad-arch-comparison/livenesslab/tools/run_without_tensorflow.py")
S = "scripts/pad_anet_run.py"
L = "delivery/anetrun/launch.zsh"
COPY = (S, TESTS, "scripts/pad_anet_comparison.py", "scripts/pad_anet_definitions.py", "scripts/pad_anet_models.py",
        "scripts/pad_arch_comparison.py", "scripts/pad_arch_definitions.py", "scripts/pad_arch_models.py", "scripts/pad_arch_run.py",
        "scripts/nuaa_folds_run.py", "scripts/nuaa_session_ablation_run.py", "scripts/nuaa_bn_recalibration.py", "scripts/nuaa_bn_test.py",
        "scripts/nuaa_patience_verify.py", "scripts/m2_writers.py", "scripts/path_guard.py", "scripts/package_rules.py",
        "scripts/check_nuaa_folds_run.py", "scripts/check_nuaa_session_ablation_run.py", "scripts/check_pad_arch_comparison.py",
        "scripts/check_pad_arch_run.py", "scripts/check_pad_anet_comparison.py", "delivery/anetrun/make_config.py", "delivery/anetrun/launch.zsh",
        "delivery/anetcmp/make_config.py", "delivery/bnrecal/nuaa_bn_recalibration.json", "src/livedetection/src/architectures.py")
SKIP_OK = ("missing input: delivery/anetrun/pad_anet_run.json or the files it pins are not present",
           "missing input: TensorFlow is not installed: the tests that build or train a Keras model are skipped",
           "missing input: zsh, caffeinate, pmset, sw_vers or git is not present (macOS): the test of delivery/anetrun/launch.zsh is skipped")
OFF, TF = "offline", "tf"
MACOS_TOOL_NAMES = ("zsh", "caffeinate", "pmset", "sw_vers", "git")       # quelli di TestG2Launch della suite
MACOS_TOOLS = all(shutil.which(t) for t in MACOS_TOOL_NAMES)
# (nome, modo, file mutato, testo cercato, testo sostitutivo)
MUTANTS = [
    ("schema not checked", OFF, S, '    if cfg["schema"] != CONFIG_SCHEMA:', "    if False:"),
    ("runs in another order accepted", OFF, S, '    if cfg["runs"] != list(RUNS):', "    if False:"),
    ("estimator orders not checked", OFF, S, '    if es["orders"] != expected_orders():', "    if False:"),
    ("BNRECAL models not checked", OFF, S, ' or ec["models"] != 36:', ":"),
    ("choice of the author not required", OFF, S, '    if "choice of the author" not in v:', "    if False:"),
    ("test sentence not required", OFF, S, '    if TEST_SENTENCE not in cfg["test"]:', "    if False:"),
    ("public outputs not checked", OFF, S, ' or out["public"] != list(PUBLIC_OUTPUTS) or ', " or "),
    ("PR A package not anchored", OFF, S, '    if bad:\n        fail(f"PR A package:', '    if False:\n        fail(f"PR A package:'),
    ("plan files not compared with the regenerated ones", OFF, S, '        if b != want[f"{k}.csv"].encode("utf-8"):', "        if False:"),
    ("unofficial trial accepted", OFF, S, ' or pins.trial.get("hooks", {}).get("official_configuration") is not True:', ":"),
    ("PR #27 configuration not that of PR A", OFF, S, '    if sha256_bytes(brb) != pins.cfg_c5["bn_recalibration_config"]["sha256"]:', "    if False:"),
    ("positions checked only by number", OFF, S, '        if got != (str(i), net, str(f), str(s), str(patience), ARM, "none"):', "        if got[0] != str(i):"),
    ("cap from another column", OFF, S, '        cap = float(rows[0]["cap_wall_hours"])', '        cap = float(rows[0]["wall_hours_max"])'),
    ("machine rule skipped", OFF, S, "    return env, R.machine_rule(pins.trial)", '    return env, {"equal": True, "trial_run": pins.trial["run_id"]}'),
    ("thread environment of the trial not required", OFF, S, "    env = R.thread_environment_rule(pins.cfg_pr, pins.trial)",
     "    env = PAC.thread_environment(pins.cfg_pr)"),
    ("hooks not checked on an official root", OFF, S, "    hooks = check_hooks(official or is_official_config(cfg_p))", "    hooks = check_hooks(False)"),
    ("hooks of this module not registered", OFF, S, '    HOOKS = {**C.HOOKS, "pad_anet_run": MY_HOOKS}'.lstrip(), "HOOKS = dict(C.HOOKS)"),
    ("rule (n) not applied", OFF, S, '    outcome = DEF.training_outcome(card["history"], weights_finite)', '    outcome = {"evaluable": True, "reason": None}'),
    ("restored weights not checked", OFF, S, "    weights_finite = DEF.arrays_finite(model.get_weights()) if", "    weights_finite = True if"),
    ("in-memory passes not made", OFF, S, '                C.forward_checked(model, part, ctx.cfg["estimator"]["batch"], f"in-memory {role}", "ordinary")', "                pass"),
    ("not evaluable model saved", OFF, S, "    if reason is not None:\n        del model", "    if False:\n        del model"),
    ("phase 2 for a not evaluable model", OFF, S, '                if rec.get("evaluable"):\n                    ctx.current = rec', '                if True:\n                    ctx.current = rec'),
    ("not evaluable in phase 2 not caught", OFF, S, "    except C.NotEvaluable as e:\n        not_evaluable(ctx, rec, str(e))",
     "    except ZeroDivisionError as e:\n        not_evaluable(ctx, rec, str(e))"),
    ("card check without the finite minimum", OFF, S, "    if isinstance(h, dict) and any(v is None", "    if False and isinstance(h, dict) and any(v is None"),
    ("val_loss not checked", OFF, S, '    if not (diff <= FR.VAL_LOSS_TOL[0] or diff <= FR.VAL_LOSS_TOL[1] * card["best_val_loss"]):', "    if False:"),
    ("evaluator mismatch ignored", OFF, S, '        if mismatches:\n            status = "MISMATCH"', '        if False:\n            status = "MISMATCH"'),
    ("dev frames not compared", OFF, S, "        if len(v) != 1:", "        if False:"),
    ("attempt rule off", OFF, S, "    if attempt != need:", "    if False:"),
    ("attempt not required", OFF, S, '    if attempt is None:\n        fail("--attempt N is required', '    if False:\n        fail("--attempt N is required'),
    ("dirty tree accepted", OFF, S, '    if state["dirty"]:', "    if False:"),
    ("estimator check not required", OFF, S, "        if not a.estimator_check:", "        if False:"),
    ("official run with an unofficial estimator check", OFF, S, '    if official and doc.get("official") is not True:', "    if False:"),
    ("estimator check of other code accepted", OFF, S, '    if doc.get("code_sha256") != code:', "    if False:"),
    ("other output root without the flag", OFF, S, '    if not unofficial:\n        fail(f"--out-root other than', '    if False:\n        fail(f"--out-root other than'),
    ("run.log not checked before the RESULT line", OFF, S,
     '        if probs := PAC.output_problems((run / LOG).read_text(encoding="utf-8", errors="replace"), inp):\n            fail(f"{LOG} holds text',
     '        if False:\n            fail(f"{LOG} holds text'),
    ("not evaluable models not in the final status", OFF, S, '[r["name"] for r in recs if not r.get("phase_1")], "not_evaluable": ne,',
     '[r["name"] for r in recs if not r.get("phase_1")],'),
    ("not evaluable status not given", OFF, S, '    if rec.get("evaluable") is False:\n        return NOT_EVALUABLE', '    if False:\n        return NOT_EVALUABLE'),
    ("threads read back not required", OFF, S, '        R.require_threads(extra["threads_rule"])\n        print(f"   {PAC.threads_text(threads)}", flush=True)',
     '        print(f"   {PAC.threads_text(threads)}", flush=True)'),
    ("inputs not checked at the end", OFF, S, "        checks = final_checks(cfg, cfg_p, cfg_sha, base, pins, refs, est_folder)", "        checks = {}"),
    ("estimator outcome not computed", OFF, S, "        status = R.estimator_status(status, rows)", "        status = status"),
    ("estimator statistics tolerance changed", OFF, S, "                check = BT.recal_check(bt_cfg, stats,", "                check = BT.recal_check(bt_cfg, dict(stats),"),
    ("re-estimated statistics not translated", OFF, S, '            raise C.NotEvaluable(f"{NOT_EVALUABLE}: non-finite re-estimated statistics") from None',
     '            raise C.NotEvaluable("not evaluable") from None'),
    ("reloaded model not checked", OFF, S, "    C.check_loaded(orig, ctx.cfg_c5, net)", "    pass"),
    ("phase 2 order of another network", OFF, S, '    order = ctx.cfg["estimator"]["orders"][net]\n    tf.keras.backend.clear_session()',
     '    order = list(reversed(ctx.cfg["estimator"]["orders"][net]))\n    tf.keras.backend.clear_session()'),
    ("fitting not scored in memory", OFF, S, '            for role, part in (("dev", dev), ("fitting", fit)):', '            for role, part in (("dev", dev),):'),
    ("dev pass on the fitting frames", OFF, S, '            for role, part in (("dev", dev), ("fitting", fit)):', '            for role, part in (("dev", fit), ("fitting", fit)):'),
    ("patience of the run not passed", OFF, S, '        ctx.training = {**pins.pcfg["training"], "early_stopping_patience": patience}',
     '        ctx.training = {**pins.pcfg["training"], "early_stopping_patience": 4}'),
    ("cap in the wrong unit", OFF, S, "    budget = SR.Budget(cap_h * 60)", "    budget = SR.Budget(cap_h * 3600)"),
    ("model in progress not discarded at the cap", OFF, S, '                R.discard_model(ctx.rdir, ctx.current["name"])\n                R.mark_discarded(ctx.current)',
     "                pass"),
    ("date accepted for an official root", OFF, S, "    if a.date is not None and official:", "    if False:"),
    ("hooks checked only for the official root", OFF, S, "    hooks = check_hooks(official or is_official_config(cfg_p))", "    hooks = check_hooks(official)"),
    ("card counts not checked", OFF, S, '    if card["n_fitting"] != ctx.pcfg["arms"][ARM]["total"] or card["n_dev"] != ctx.pcfg["dev"]["total"]:', "    if False:"),
    ("not evaluable model without phase 1 recorded", OFF, S, '        not_evaluable(ctx, rec, reason)\n        rec["phase_1"] = True', "        not_evaluable(ctx, rec, reason)"),
    ("not evaluable model without the threads after", OFF, S,
     '        rec["threads_after"] = PAC.check_threads(ctx.tf, ctx.cfg_pr, f"after {name}")\n        mw.write_all(ctx.rdir, {f"{name}_card.json"',
     '        mw.write_all(ctx.rdir, {f"{name}_card.json"'),
    ("restricted SHA256SUMS not written", OFF, S, "        res_sha = R.write_restricted_sums(ctx.rdir)", "        res_sha = None"),
    ("keys not checked again before each training (seen by the copy comparison)", OFF, S, '        PAC.keys_check(ref["card"], role, part["rows"])', "        pass"),
    ("final check that no test frame was assembled removed (seen by the copy comparison)", OFF, S,
     '        PAC.assert_no_test(inp)\n        frames = check_dev_frames(', "        frames = check_dev_frames("),
    ("keys of every unit not checked at the start (seen by the copy comparison)", OFF, S,
     "        R.check_unit_keys(inp, refs, pins.pcfg)\n        store = FR.FrameStore(", "        store = FR.FrameStore("),
    ("reason not in the card of a not evaluable model", OFF, S, ' "not_evaluable_reason": reason,\n', "\n"),
    ("rule (n) record of the card not marked not evaluable", OFF, S, ' "rule_n": {**outcome, "evaluable": False, "reason": reason},', ""),
    ("launch: the estimator check passed as one word", OFF, L, "  [[ -n $est ]] && ci+=(--estimator-check $est)", '  [[ -n $est ]] && ci+=("--estimator-check $est")'),
    ("launch: nice not checked", OFF, L, "if [[ $nice_now != 0 ]]; then", "if false; then"),
    ("launch: command launched after a failed check", OFF, L, "if ! grep -q '^check-machine exit 0$' $C || ! grep -q '^check-inputs exit 0$' $C; then",
     "if false; then"),
    ("launch: exit code of the command not propagated", OFF, L, "exit $rc\n", "exit 0\n"),
    ("phase 1 trainer of another network", TF, S, "lambda: C.build_checked(ctx.cfg_c5, net, ctx.arch)", 'lambda: C.build_checked(ctx.cfg_c5, "AttackNetV1", ctx.arch)'),
]
# mutazione di controllo, che deve SOPRAVVIVERE
CONTROL = [("control (must survive): a comment changed", OFF, S, "# ============================================================================ uscite",
            "# ============================================================================ uscite pubbliche")]
# controlli dichiarati ridondanti, che devono SOPRAVVIVERE: nessuno. I tre controlli che non possono fallire da soli (le
# chiavi ricontrollate prima di ogni addestramento, gemelle di pad_arch_run.check_unit_keys all'avvio; il controllo finale
# che nessun frame di test sia stato assemblato, già fatto da pad_arch_run.unit_frames a ogni assemblaggio) sono comunque
# visti dal confronto delle copie con il sorgente di pad_arch_run (TestECopies): toglierli cambia le differenze dichiarate.
REDUNDANT = []


def pairs(x) -> tuple:
    return x if isinstance(x, tuple) else (x,)


def tool_source() -> Path:
    for t in TOOL_SOURCES:
        if (ROOT / t).is_file():
            return ROOT / t
    sys.exit("run_without_tensorflow.py not found")


def copy_tree(dst: Path) -> Path:
    code = dst / "code"
    for rel in COPY:
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, code / rel)
    (code / TOOL).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tool_source(), code / TOOL)
    return code


def run_tests(code: Path, tmp: Path, mode: str, failfast: bool = False) -> tuple:
    tmp.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp), "CUDA_VISIBLE_DEVICES": "-1", "TF_CPP_MIN_LOG_LEVEL": "2"}
    cmd = [sys.executable, "-I", "-B"] + ([TOOL] if mode == OFF else []) + [TESTS] + (["-f"] if failfast else [])
    r = subprocess.run(cmd, cwd=code, env=env, capture_output=True, text=True, timeout=3600)
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("tests run")), "no test summary")
    skips = tuple(sorted(l for l in r.stdout.splitlines() if l.startswith("skipped ")))
    return r.returncode, line, skips


def detected(rc: int, line: str, skips=(), base_skips=()) -> str:
    m = re.match(r"tests run: (\d+), failures: (\d+), errors: (\d+), skipped: (\d+)", line)
    if not m:
        return "invalid"
    if int(m.group(2)) + int(m.group(3)) > 0:
        return "killed"
    return "survived" if rc == 0 and tuple(skips) == tuple(base_skips) else "invalid"


def one(tmp: Path, i: int, name: str, mode: str, target: str, old, new, base_skips=(), failfast: bool = False) -> tuple:
    work = tmp / f"work{i}"
    code = copy_tree(work)
    f = code / target
    text = f.read_text(encoding="utf-8")
    for o, n in zip(pairs(old), pairs(new)):
        text = text.replace(o, n, 1)
    f.write_text(text, encoding="utf-8")
    rc, line, skips = run_tests(code, work / "tmp", mode, failfast)
    shutil.rmtree(work, ignore_errors=True)
    return name, detected(rc, line, skips, base_skips), line, rc


def main() -> int:
    ap = argparse.ArgumentParser(description="Controllo delle mutazioni dei run C5 (offline, sintetico)")
    ap.add_argument("--list", action="store_true", help="elenca le mutazioni e verifica i modelli, senza eseguire i test")
    ap.add_argument("--mode", choices=[OFF, TF], default=None, help="solo le mutazioni di questo modo")
    ap.add_argument("--only", default=None, help="solo le mutazioni con questi numeri (da 1, separati da virgole; c = controllo e ridondanti)")
    a = ap.parse_args()
    problems = [f"{n}: the pattern occurs {k} times in {t} (expected exactly once)"
                for n, _, t, old, _ in MUTANTS + CONTROL + REDUNDANT for o in pairs(old)
                if (k := (ROOT / t).read_text(encoding="utf-8").count(o)) != 1]
    names = [n for n, *_ in MUTANTS + CONTROL + REDUNDANT]
    problems += [f"duplicated mutation name: {n}" for n in sorted({n for n in names if names.count(n) > 1})]
    for p in problems:
        print(f"ERROR: {p}", file=sys.stderr)
    if problems:
        print("RESULT: FAIL (a declared mutation no longer matches the code)")
        return 1
    if a.list:
        for i, (n, mode, t, _, _) in enumerate(MUTANTS, start=1):
            print(f"{i:3d} {n}  [{t}, {mode}]")
        for n, mode, t, _, _ in CONTROL + REDUNDANT:
            print(f"    {n}  [{t}, {mode}] (must survive)")
        print(f"{len(MUTANTS)} mutations ({sum(1 for m in MUTANTS if m[1] == OFF)} offline, {sum(1 for m in MUTANTS if m[1] == TF)} with TensorFlow), "
              f"{len(CONTROL)} control mutation and {len(REDUNDANT)} declared redundant, every pattern found exactly once")
        return 0
    tokens = None if a.only is None else [x.strip() for x in a.only.split(",") if x.strip()]
    only = None if tokens is None else {int(x) for x in tokens if x != "c"}
    chosen = [(i, m) for i, m in enumerate(MUTANTS, start=1) if (a.mode is None or m[1] == a.mode) and (only is None or i in only)]
    controls = [m for m in CONTROL + REDUNDANT if (a.mode is None or m[1] == a.mode) and (tokens is None or "c" in tokens)]
    # le mutazioni di launch.zsh hanno test solo con gli strumenti di macOS (saltati altrove, con il motivo dichiarato
    # della suite): senza quegli strumenti sono «not applicable», elencate e tolte dal conteggio, non contate come uccise
    na = [(i, m) for i, m in chosen if m[2] == L and not MACOS_TOOLS]
    chosen = [(i, m) for i, m in chosen if (i, m) not in na]
    for i, (n, mode, t, _, _) in na:
        print(f"NOT APPLICABLE {i:3d} {n} [{mode}] (its tests need {', '.join(MACOS_TOOL_NAMES)}, not all present)", flush=True)
    modes = sorted({m[1] for _, m in chosen} | {m[1] for m in controls})
    killed = invalid = survived_controls = 0
    with tempfile.TemporaryDirectory(prefix="anetrun-mutations-") as tmp:
        tmp = Path(tmp)
        base = {}
        for mode in modes:
            rc, line, skips = run_tests(copy_tree(tmp / f"base_{mode}"), tmp / f"base_{mode}" / "tmp", mode)
            print(f"unmutated copy ({mode}): {TESTS} exit {rc} ({line})", flush=True)
            for s in skips:
                print(f"unmutated copy ({mode}): {s}", flush=True)
            if rc or not re.fullmatch(r"tests run: \d+, failures: 0, errors: 0, skipped: \d+", line):
                print(f"RESULT: FAIL (the tests do not pass on the unmutated copy, mode {mode})")
                return 1
            if any(s.split(": ", 1)[-1] not in SKIP_OK for s in skips):
                print("RESULT: FAIL (the unmutated copy skips tests for a reason that is not a declared missing input)")
                return 1
            base[mode] = skips
        for j, (n, mode, t, old, new) in enumerate(controls):
            name, verdict, line, rc = one(tmp, 1000 + j, n, mode, t, old, new, base[mode])
            survived_controls += verdict == "survived"
            print(f"CONTROL   {verdict.upper():<9} {name} [{mode}] ({line})", flush=True)
        for i, (n, mode, t, old, new) in chosen:
            name, verdict, line, rc = one(tmp, i, n, mode, t, old, new, base[mode], failfast=True)
            killed += verdict == "killed"
            invalid += verdict == "invalid"
            print(f"{verdict.upper():<9} {i:3d} {name} [{mode}] ({line})", flush=True)
            if verdict == "invalid":
                print(f"ERROR: invalid mutant {name!r}: the test suite did not run to a summary with failures, or skipped other tests (exit {rc})",
                      file=sys.stderr)
    print(f"mutations killed: {killed}/{len(chosen)}" + (f" ({invalid} invalid)" if invalid else "")
          + ("" if len(chosen) + len(na) == len(MUTANTS) else f" (selection of {len(chosen) + len(na)} of {len(MUTANTS)})")
          + (f" ({len(na)} of launch.zsh not applicable: their tests need the tools of macOS)" if na else ""))
    if controls:
        print(f"control mutations survived: {survived_controls}/{len(controls)}")
    ok = killed == len(chosen) and survived_controls == len(controls)
    print("RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
