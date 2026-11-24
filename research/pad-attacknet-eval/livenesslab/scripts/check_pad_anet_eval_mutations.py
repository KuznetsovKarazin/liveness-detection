"""
Controllo delle mutazioni di scripts/pad_anet_eval.py (la valutazione C5: configurazione, file fissati, i 48 riferimenti
di C2 e la loro riproduzione esatta, i sei run, riproduzione del dev, test, punteggi non finiti, analisi, uscite),
offline e deterministico, con la meccanica di check_pad_anet_run_mutations.py: una mutazione dichiarata alla volta su
una copia del codice in una cartella temporanea, la suite scripts/check_pad_anet_eval.py sulla copia (TensorFlow
nascosto: il flusso usa un valutatore finto), uccisa se la suite arriva al riepilogo con almeno un fallimento. La
configurazione consegnata e il pacchetto non sono copiati: i loro test sono saltati con il motivo dichiarato.

  .venv/bin/python -I -B scripts/check_pad_anet_eval_mutations.py                 # tutte
  .venv/bin/python -I -B scripts/check_pad_anet_eval_mutations.py --list          # elenco e verifica dei modelli
  .venv/bin/python -I -B scripts/check_pad_anet_eval_mutations.py --only 3,7      # solo queste (da 1); c = controllo
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
TESTS = "scripts/check_pad_anet_eval.py"
TOOL = "tools/run_without_tensorflow.py"
TOOL_SOURCES = ("tools/run_without_tensorflow.py", "delivery/pad-attacknet-eval/livenesslab/tools/run_without_tensorflow.py",
                "delivery/pad-attacknet-run/livenesslab/tools/run_without_tensorflow.py",
                "delivery/pad-arch-comparison/livenesslab/tools/run_without_tensorflow.py")
S = "scripts/pad_anet_eval.py"
L = "delivery/aneteval/launch.zsh"                    # le sue mutazioni: «not applicable» senza gli strumenti di macOS
CR = "delivery/aneteval/copy_run_records.py"
MC = "delivery/aneteval/make_config.py"
COPY = (S, TESTS, "scripts/pad_anet_run.py", "scripts/pad_anet_comparison.py", "scripts/pad_anet_definitions.py", "scripts/pad_anet_models.py",
        "scripts/pad_arch_eval.py", "scripts/pad_arch_comparison.py", "scripts/pad_arch_definitions.py", "scripts/pad_arch_models.py",
        "scripts/pad_arch_run.py", "scripts/nuaa_folds_run.py", "scripts/nuaa_session_ablation_run.py", "scripts/nuaa_bn_recalibration.py",
        "scripts/nuaa_bn_test.py", "scripts/nuaa_patience_verify.py", "scripts/m2_writers.py", "scripts/path_guard.py", "scripts/package_rules.py",
        "scripts/check_nuaa_folds_run.py", "scripts/check_nuaa_session_ablation_run.py", "scripts/check_pad_arch_comparison.py",
        "scripts/check_pad_arch_run.py", "scripts/check_pad_anet_comparison.py", "scripts/check_pad_anet_run.py",
        "delivery/aneteval/make_config.py", "delivery/aneteval/copy_run_records.py", "delivery/aneteval/launch.zsh", "delivery/anetrun/make_config.py", "delivery/anetrun/launch.zsh", "delivery/anetcmp/make_config.py",
        "delivery/bnrecal/nuaa_bn_recalibration.json", "src/livedetection/src/architectures.py")
SKIP_OK = ("missing input: delivery/aneteval/pad_anet_eval.json or the files it pins are not present",
           "missing input: zsh, caffeinate, pmset, sw_vers or git is not present (macOS): the test of delivery/aneteval/launch.zsh is skipped")
OFF, TF = "offline", "tf"
MACOS_TOOL_NAMES = ("zsh", "caffeinate", "pmset", "sw_vers", "git")
MACOS_TOOLS = all(shutil.which(t) for t in MACOS_TOOL_NAMES)
# (nome, modo, file mutato, testo cercato, testo sostitutivo)
MUTANTS = [
    ("schema not checked", OFF, S, '    if cfg["schema"] != CONFIG_SCHEMA:', "    if False:"),
    ("run ids not checked against their run", OFF, S, '            if not m or f"{m.group(1)}-p{m.group(2)}" != NR.RUN_TAG[r]:', "            if not m:"),
    ("C2 evaluation run id not checked", OFF, S, '    if ce["run_id"] != "20261121-ARCHEVAL-332fbdf" or not SR._hex(ce["restricted_sha256sums_sha256"]):',
     '    if not SR._hex(ce["restricted_sha256sums_sha256"]):'),
    ("public outputs not checked", OFF, S, ' or out["public"] != list(PUBLIC_OUTPUTS) or ', " or "),
    ("choices not required of the author", OFF, S, '        NR._author(v, f"choices_of_the_author.{k}")', "        pass"),
    ("C2 restricted SHA256SUMS not anchored", OFF, S, '    if sha256_bytes(sb) != cfg["c2_evaluation"]["restricted_sha256sums_sha256"]:', "    if False:"),
    ("C2 file not checked against its line", OFF, S, '        if sha256_bytes(b) != sums[fn]:\n            fail(f"C2 {fn}: SHA-256',
     '        if False:\n            fail(f"C2 {fn}: SHA-256'),
    ("C2 rows of another cell accepted", OFF, S, 'any((g["network"], g["fold"], g["train_seed"], g["patience"], g["role"], g["checkpoint_sha256"]) !=',
     'any((c["network"], str(c["fold"]), str(c["seed"]), str(c["patience"]), g["role"], g["checkpoint_sha256"]) !='),
    ("C2 primary analysis not compared", OFF, S, "    if bad:\n        fail(f\"C2 reproduction: the primary analysis", "    if False:\n        fail(f\"C2 reproduction: the primary analysis"),
    ("C2 conditions not compared", OFF, S, '    if pub != theirs:', "    if False:"),
    ("C2 units.csv not compared", OFF, S, '    if units.encode("utf-8") != pins.c2_units:', "    if False:"),
    ("C2 reproduction after the test", OFF, S, "        c2_rep = reproduce_c2(pins, c2)\n        c2_rep[\"thresholds_checked\"]",
     '        c2_rep = {"exact": True, "conditions": 8}\n        c2_rep["thresholds_checked"]'),
    ("C2 thresholds at 0.5 replaced by the EER", OFF, S, '"threshold": float(FR.FIXED_THRESHOLD) if rule == "fixed_0.5" else r["threshold_eer"]',
     '"threshold": r["threshold_eer"]'),
    ("not the last attempt evaluated", OFF, S, "        if last != rec[\"run_id\"]:", "        if False:"),
    ("status of a run not checked against the configuration", OFF, S, '    if sha256_bytes(sb) != record["status_sha256"]:', "    if False:"),
    ("run file not checked against its line", OFF, S, '            if sha256_file(FR.regular_path(folder / sub / fn, fn)) != h:\n                fail(f"{what}: {sub}/{fn}',
     '            if False:\n                fail(f"{what}: {sub}/{fn}'),
    ("official evaluation with unofficial runs", OFF, S, '    if mode == "official" and (st.get("official") is not True or not inside):', "    if False:"),
    ("not evaluable models scored", OFF, S, "        if name in ne:\n            not_eval.append(", "        if False:\n            not_eval.append("),
    ("phase 1 not required complete", OFF, S, '            or st.get("complete_phase_1") != names:', "            or False:"),
    ("dev mismatch ignored", OFF, S, '            if any(not r["pass"] for r in repro):\n                status = "MISMATCH"',
     '            if False:\n                status = "MISMATCH"'),
    ("non-finite test scores ignored", OFF, S, '                if non_finite:\n                    status = "NON_FINITE_TEST"', "                pass"),
    ("non-finite count dropped", OFF, S, '                    non_finite.append({"model": rec["key"], "bn_state": s, "non_finite_frames": bad})', "                    pass"),
    ("reference unit missing accepted", OFF, S, '            if key not in by:\n                fail(f"condition {c}: unit {key}', '            if False:\n                fail(f"condition {c}: unit {key}'),
    ("reasons of the not evaluable models not passed", OFF, S, "    primary = DEF.primary_analysis({c: conds[c] for c in E.PRIMARY}, reasons_of(runs, DEF.PRIMARY_PATIENCE))",
     "    primary = DEF.primary_analysis({c: conds[c] for c in E.PRIMARY})"),
    ("descriptive analysis not computed", OFF, S, "    descriptive = descriptive_analysis(conds, runs)", '    descriptive = {"pairs": {}}'),
    ("opposite of C2 not required", OFF, S, "            if not all(opposite.values()):", "            if False:"),
    ("opposite check with the bounds not swapped", OFF, S, 'ia["low"] == -ib["high"] and ia["high"] == -ib["low"]', 'ia["low"] == -ib["low"]'),
    ("hooks not checked on an official root", OFF, S, "    hooks = check_hooks(official or is_official_config(cfg_p))", "    hooks = check_hooks(False)"),
    ("attempt not required", OFF, S, '    if attempt is None:\n        fail("--attempt N is required', '    if False:\n        fail("--attempt N is required'),
    ("dirty tree accepted", OFF, S, '    if state["dirty"]:', "    if False:"),
    ("other output root without the flag", OFF, S, '    if not a.unofficial_out_root:\n        fail(f"--out-root other than', '    if False:\n        fail(f"--out-root other than'),
    ("authorisation not required", OFF, S, "    if not a.i_am_authorized:", "    if False:"),
    ("after_stop not checked", OFF, S, '    if "no automatic resume" not in cfg["after_stop"] or "test_read_started" not in cfg["after_stop"] or "second reading" not in cfg["after_stop"] \\',
     '    if False and "no automatic resume" not in cfg["after_stop"] or "test_read_started" not in cfg["after_stop"] or "second reading" not in cfg["after_stop"] \\'),
    ("second reading accepted without the option", OFF, S, "    if read and not second_reading:", "    if False:"),
    ("second reading option without a reading accepted", OFF, S, "    if second_reading and not read:", "    if False:"),
    ("unreadable attempt not counted as a reading", OFF, S, '"test_read_started": read is not False or st == "RUNNING"', '"test_read_started": read is True or st == "RUNNING"'),
    ("attempt left RUNNING not counted as a reading", OFF, S, 'read is not False or st == "RUNNING" or restricted})', 'read is not False or restricted})'),
    ("attempt with restricted scores not counted as a reading", OFF, S, 'read is not False or st == "RUNNING" or restricted})', 'read is not False or st == "RUNNING"})'),
    ("RUNNING status not rewritten when the reading starts", OFF, S, "                running(inp)\n                rdir.mkdir()", "                rdir.mkdir()"),
    ("second reading not recorded", OFF, S, '"second_reading": attempt["second_reading"], "hooks": hooks,', '"hooks": hooks,'),
    ("restricted scores removed after the reading", OFF, S, '            if extra.get("test_read_started"):\n                # dopo la lettura', '            if False:\n                # dopo la lettura'),
    ("opposite of C2 after the test only", OFF, S, '        c2_rep["opposite_before_test"] = c2_opposite_before_test(pins, c2)', '        c2_rep["opposite_before_test"] = None'),
    ("C2 thresholds not compared with results.json", OFF, S, "        if not isinstance(t, dict) or any(t.get(st) != rec[st][\"threshold_eer\"] for st in D.BN_STATES):", "        if False:"),
    ("C2 identifiers not compared with the test keys", OFF, S, '        if rec["ordinary"]["ids"] != keys:', "        if False:"),
    ("contrast mean of C2 read from the wrong key", OFF, S, '"primary_contrast": {"mean": ct.get("mean"),', '"primary_contrast": {"mean": ct.get("mean_delta"),'),
    ("test one-class not counted", OFF, S, '                if r["one_class"]:\n                    c["one_class"] += 1', '                if False:\n                    c["one_class"] += 1'),
    ("reasons of the other patience", OFF, S, '            if x["patience"] == patience:', "            if True:"),
    ("table denominators of every unit", OFF, S, "            den = DEF.denominators(mine) if mine else None", "            den = DEF.denominators(pus)"),
    ("run records: folders not compared", OFF, S, "    if have != [f.name for f in want]:", "    if False:"),
    ("run records: files of a copy not compared", OFF, S, '        if got != sorted(names + ["conditions.txt"]):', "        if False:"),
    ("run records: bytes not compared", OFF, S, "            if read_regular(pkg / f.name / nm, nm) != read_regular(f / nm, nm):", "            if False:"),
    ("run records: copies not scanned", OFF, S, '            if probs := PAC.output_problems(read_regular(pkg / f.name / nm, nm).decode("utf-8", "replace"), inp or FR.Inputs()):',
     "            if probs := []:"),
    ("run records: earlier attempts left out", OFF, S, '    out = [runs_root / p["run_id"] for r in NR.RUNS for p in NR.previous_attempts(runs_root, "run", r)]',
     '    out = [runs_root / V for r in NR.RUNS for V in [latest_attempt(r, runs_root)]]'),
    ("run records: conditions.txt not required", OFF, S, 'if got != sorted(names + ["conditions.txt"]):',
     'if sorted(x for x in got if x != "conditions.txt") != sorted(names):'),
    ("run records: check-estimator left out", OFF, S, '    out += [runs_root / p["run_id"] for p in NR.previous_attempts(runs_root, "estimator")]', "    pass"),
    ("run records: restricted/ published", OFF, S, '[f"{PUBLIC}/{n}" for n in sorted(os.listdir(folder / PUBLIC))]',
     '[f"{PUBLIC}/{n}" for n in sorted(os.listdir(folder / PUBLIC))] + [f"{RESTRICTED}/{n}" for n in sorted(os.listdir(folder / RESTRICTED))]'),
    ("run records: evaluate without the copies", OFF, S, '        extra["run_records"] = check_run_copies(cfg, base, NR.folder_arg(a.runs_root), inp)',
     '        extra["run_records"] = None'),
    ("copy of the records: a missing conditions file accepted", OFF, CR, "        if not cond.is_file():", "        if False:"),
    ("copy of the records: a different copy rewritten without --force", OFF, CR,
     "            if dst.exists() and dst.read_bytes() != src.read_bytes() and not force:", "            if False:"),
    ("published configuration: a field other than runs admitted", OFF, MC, '    return [k for k in keys if k != "runs" and new.get(k) != published.get(k)]',
     '    return [k for k in keys if k not in ("runs", "cap") and new.get(k) != published.get(k)]'),
    ("test one-class counted at 0.5 only", OFF, S, "    for (p, st, rule) in DEF.CONDITIONS:\n        for u in conds[(p, st, rule)]:",
     '    for (p, st, rule) in [c for c in DEF.CONDITIONS if c[2] == "fixed_0.5"]:\n        for u in conds[(p, st, rule)]:'),
    ("run records: evaluate scans the copies without the manifests", OFF, S, "inp or FR.Inputs()):", "FR.Inputs()):"),
    ("run records: check-inputs without the copies", OFF, S, "        copies = check_run_copies(cfg, base, NR.folder_arg(a.runs_root))", '        copies = {"folders": 7, "files_compared": 0}'),
    ("launch: evaluate after a failed check", OFF, L, "if ! grep -q '^check-machine exit 0$' $C || ! grep -q '^check-inputs exit 0$' $C; then", "if false; then"),
    ("launch: nice not checked", OFF, L, "if [[ $nice_now != 0 ]]; then", "if false; then"),
    ("launch: exit code not propagated", OFF, L, "exit $rc\n", "exit 0\n"),
    ("launch: log folder inside the repository accepted", OFF, L, "if [[ $log_dir == $repo || $log_dir == $repo/* ]]; then", "if false; then"),
    ("launch: an existing conditions file overwritten", OFF, L, "if [[ -e $C ]]; then", "if false; then"),
    ("launch: attempt not required for evaluate", OFF, L, '  evaluate) if [[ -z $attempt ]]; then', "  evaluate) if false; then"),
    ("launch: evaluate on battery", OFF, L, """  if [[ $power != *"'AC Power'"* ]]; then""", "  if false; then"),
    ("registry not used for the reloaded model", OFF, S, '    C.check_loaded(orig, ctx.cfg_c5, rec["network"])\n    out = {s: {} for s in D.BN_STATES}\n    for role, part in parts.items():\n        out["ordinary"][role] = E.forward_scores',
     '    out = {s: {} for s in D.BN_STATES}\n    for role, part in parts.items():\n        out["ordinary"][role] = E.forward_scores'),
    ("C2 rows of another checkpoint accepted", OFF, S, 'g["role"], g["checkpoint_sha256"]) !=', 'g["role"], c["checkpoint_sha256"]) !='),
]
# mutazione di controllo, che deve SOPRAVVIVERE
CONTROL = [("control (must survive): a comment changed", OFF, S, "# ============================================================================ uscite",
            "# ============================================================================ uscite pubbliche")]
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
    # il primo test fallito (unittest lo scrive su stderr): solo il nome del test e della classe, nessun percorso
    first = re.search(r"^(FAIL|ERROR): (\w+) \((?:__main__\.)?([\w.]+)\)", r.stderr, re.M)
    if first:
        line += f"; first {first.group(1)}: {first.group(3).rsplit('.', 1)[0].rsplit('.', 1)[-1]}.{first.group(2)}"
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
    ap = argparse.ArgumentParser(description="Controllo delle mutazioni della valutazione C5 (offline, sintetico)")
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
    with tempfile.TemporaryDirectory(prefix="aneteval-mutations-") as tmp:
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
