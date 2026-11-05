"""
Controllo delle mutazioni di scripts/nuaa_session_ablation.py, offline e deterministico.

Copia il codice (scripts/nuaa_session_ablation.py e la sua suite, scripts/nuaa_subject_folds.py con la sua suite che
fornisce la fixture dello schema, scripts/m2_writers.py, scripts/path_guard.py, scripts/package_rules.py e
delivery/sessionablation/nuaa_session_ablation.json) in una cartella temporanea, applica una mutazione dichiarata alla
volta (coppia cerca/sostituisci che deve comparire esattamente una volta: se un modello non si trova più, il controllo
si ferma con un errore) ed esegue scripts/check_nuaa_session_ablation.py sulla copia mutata, con gli ingressi veri in
sola lettura (SESSABL_INPUT_ROOT = radice di questo repository). Una mutazione è rilevata solo se la suite arriva al
riepilogo («tests run») con almeno un fallimento o errore; una suite che non parte la rende non valida. Una mutazione
di controllo (solo un commento) deve sopravvivere. Codice 1 se una mutazione sopravvive o non è valida, se un salto
non è un ingresso mancante dichiarato o se un modello non si trova.

  .venv/bin/python scripts/check_nuaa_session_ablation_mutations.py            # tutte le mutazioni
  .venv/bin/python scripts/check_nuaa_session_ablation_mutations.py --list     # solo l'elenco, con la verifica dei modelli
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
from check_nuaa_session_ablation import MISSING_INPUT  # noqa: E402

TESTS = "scripts/check_nuaa_session_ablation.py"
S = "scripts/nuaa_session_ablation.py"
COPY = (S, TESTS, "scripts/nuaa_subject_folds.py", "scripts/check_nuaa_subject_folds.py", "scripts/m2_writers.py",
        "scripts/path_guard.py", "scripts/package_rules.py", "delivery/sessionablation/nuaa_session_ablation.json")
# (nome, file mutato, testo cercato, testo sostitutivo)
MUTATIONS = [
    ("control: comment only (must survive)", S, "# ============================================================================ disegno",
     "# ============================================================================ disegno (control mutation)"),
    ("roles: dev fold = held-out fold", S, "        dev = i % k + 1\n", "        dev = i\n"),
    ("roles: fitting includes the dev fold", S, "fit = [j for j in range(1, k + 1) if j not in (i, dev)]", "fit = [j for j in range(1, k + 1) if j != i]"),
    ("S12: every fitting subject, also those without sessions 01-02", S, "s12 = fit_subjects & has12", "s12 = set(fit_subjects)"),
    ("arm B: no rounding to the step", S, "return (v // step) * step", "return v"),
    ("arm B: minimum taken over the session-03 cells too", S,
     'min_b = min(len(folds[i]["cells_s12"][c]) for i in folds for c in CELLS_12)', 'min_b = min(len(folds[i]["cells_s12"][c]) for i in folds for c in CELLS)'),
    ("allocation: remainder to the first cells, not in remainder_order", S, "for c in [x for x in order if x in out][:rest]:", "for c in list(out)[:rest]:"),
    ("allocation: short cells never capped", S, "        if not short:", "        if True:"),
    ("allocation: caps from the largest availability", S, "caps_a = {c: min(", "caps_a = {c: max("),
    ("dev mix: caps from the largest availability", S, "caps_m = {c: min(", "caps_m = {c: max("),
    ("nesting: A* session 01-02 cells sampled from the available frames", S,
     '"armAstar_fitting": {**{c: sample(b_cells[c], alloc_a[c],', '"armAstar_fitting": {**{c: sample(f["cells_s12"][c], alloc_a[c],'),
    ("nesting: B' sampled from the available frames", S, '"armBprime_fitting": {c: sample(b_cells[c], alloc_bp[c],',
     '"armBprime_fitting": {c: sample(f["cells_s12"][c], alloc_bp[c],'),
    ("config: B' allocation not compared with arm A*", S, 'if d["arm_b_prime"]["allocation"] != alloc_bp:', "if False:"),
    ("config: nesting rule not checked", S, '    if d.get("nesting") != NESTING_RULE:', "    if False:"),
    ("budget: optional arm B' added to the base total", S,
     '"total_minutes": [round((total_mid + score_mid) / 60, 1), round((total_hi + score_hi) / 60, 1)],',
     '"total_minutes": [round((total_mid + score_mid + bp_train[0]) / 60, 1), round((total_hi + score_hi + bp_train[1]) / 60, 1)],'),
    ("sampling: shortfall silently truncated", S, "    if len(rows) < n:", "    if False:"),
    ("sampling: first n names instead of spread", S, "return [by_name[x] for x in NF.spread(list(by_name), n)]", "return [by_name[x] for x in sorted(by_name)[:n]]"),
    ("config: arm B count not compared with the rule", S, 'if d["arm_b"]["per_cell"] != n:', "if False:"),
    ("config: arm A* allocation not compared with the rule", S,
     'if d["arm_a_star"]["allocation"] != alloc_a or d["arm_a_star"]["variant"] != var_a:', "if False:"),
    ("config: dev count not compared with the rule", S, 'if d["dev"]["per_cell"] != dn:', "if False:"),
    ("config: arm A* total not tied to arm B", S, 'if d["arm_a_star"]["total"] != len(CELLS_12) * d["arm_b"]["per_cell"]:', "if False:"),
    ("disclosure: single-subject cells published", S, "        if value and len(with_frames) < 2:", "        if False:"),
    ("disclosure: rank rule not applied", S, "        elif dis.would_reveal(v):", "        elif False:"),
    ("disclosure: schema baseline not loaded", S, "    for _, _, subs, cells in baseline:\n        dis.add(dis.vector(subs, cells))",
     "    for _, _, subs, cells in []:\n        dis.add(dis.vector(subs, cells))"),
    ("disclosure: R1s never recognised", S, "known = not dis.full.reduce(v)", "known = False"),
    ("rank: elimination without the pivot scaling", S, "v = {c: b * v.get(c, 0) - a * r.get(c, 0) for c in set(v) | set(r)}",
     "v = {c: v.get(c, 0) - r.get(c, 0) for c in set(v) | set(r)}"),
    ("rank: dependent vectors also checked per subject", S, "        if not self.full.reduce(v):\n            return []", "        if False:\n            return []"),
    ("disclosure: final verification ignores differencing", S, "    if worse:", "    if False:"),
    ("inputs: schema public availability not compared", S, "                if av.get(c) != v:", "                if False:"),
    ("inputs: schema public roles not compared", S, 'if pf.get("dev_fold") != f["dev_fold"] or pf.get("fitting_folds") != f["fitting_folds"]:', "if False:"),
    ("inputs: schema public manifest SHA-256 not compared", S, 'if r.get("manifest_sha256") != inp["file_sha"][(i, role)]:', "if False:"),
    ("inputs: fold manifest rows not compared with the official lists", S,
     'if o is None or any(o[c] != x[c] for c in ("label", "subject", "session", "sha256", "bytes")):', "if o is None:"),
    ("inputs: schema manifests not checked against the subjects of their role", S,
     'if not {x["subject"] for x in inp["schema_rows"][(i, role)]} <= subs:', "if False:"),
    ("inputs: fold manifest fold and role not checked", S, 'if x["fold"] != str(k) or x["role"] != role:', "if False:"),
    ("inputs: test manifests not required to cover every frame", S, "if len(tested) != len(set(tested)) or set(tested) != set(official):", "if False:"),
    ("inputs: fold schema status not checked", S, 'if not isinstance(st, dict) or st.get("status") != "PASS" or st.get("run_id") != fs["run_id"]:',
     "if not isinstance(st, dict):"),
    ("inputs: SHA256SUMS pins not compared", S, "        if sha256_bytes(b) != pin:", "        if False:"),
    ("inputs: listed files not compared with SHA256SUMS", S, "    if sha256_bytes(b) != sums[name]:", "    if False:"),
    ("inputs: fold schema configuration SHA-256 not compared", S, 'if sha256_bytes(fb) != fc["sha256"]:', "if False:"),
    ("inputs: M2 manifest SHA-256 not compared", S, 'if sha256_bytes(b) != spec.get("sha256"):', "if False:"),
    ("inputs: expected test frames not compared", S, 'if len(f["test_rows"]) != test_frames[i - 1]:', "if False:"),
    ("budget: cap not enforced", S, 'if out["total_minutes"][1] > b["wall_time_cap_minutes"]:', "if False:"),
    ("budget: scaled by fitting frames only", S, "ratio = (fit[arm] + dev_frames) / ref_frames", "ratio = fit[arm] / ref_frames"),
    ("outputs: restricted frames written in the public JSON", S, '"frames": dd["value"] if dd["status"] == "published" else None}', '"frames": dd["value"]}'),
    ("outputs: restricted values printed in the Markdown", S, '    if dd["status"] == "published":', "    if True:"),
    ("outputs: public identifiers not refused", S, "        if problems:\n            fail(\"public outputs refused: \"",
     "        if False:\n            fail(\"public outputs refused: \""),
    ("outputs: failure leaves the outputs", S, "        left = invalidate_outputs(run, k)\n        reason =", "        left = []\n        reason ="),
    ("outputs: foreign entries in the run folder accepted", S, "                if not any(mw.owned_name(e, n) for n in names):", "                if False:"),
    ("outputs: symbolic output root accepted", S, "if (out_root if out_root is not None else run.parent).is_symlink():", "if False:"),
    ("outputs: run inside an input folder accepted", S, "if pg.within(run, p if Path(p).is_dir() else Path(p).parent):", "if False:"),
    ("run: --i-am-authorized not required", S, "    if not a.i_am_authorized:", "    if False:"),
    ("run id: dirty tree without -dev", S, '+ ("-dev" if state["dirty"] else "")', '+ ""'),
]


def check_patterns() -> list:
    out = []
    for name, f, old, new in MUTATIONS:
        n = (ROOT / f).read_text(encoding="utf-8").count(old)
        if n != 1:
            out.append(f"{name}: pattern found {n} times in {f}")
    return out


def run_one(name: str, f: str, old: str, new: str) -> tuple:
    tmp = Path(tempfile.mkdtemp(prefix="sessabl_mut_"))
    try:
        for rel in COPY:
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, tmp / rel)
        p = tmp / f
        p.write_text(p.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
        env = dict(os.environ, SESSABL_INPUT_ROOT=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([sys.executable, "-I", str(tmp / TESTS)], cwd=tmp, capture_output=True, text=True, timeout=900, env=env)
        text = r.stdout + r.stderr
        m = re.search(r"tests run: (\d+), failures: (\d+), errors: (\d+), skipped: (\d+)", text)
        if not m:
            return name, "invalid (suite did not report)", text[-800:]
        tests, failures, errors, skipped = map(int, m.groups())
        skips = re.findall(r"^SKIPPED (.*)$", text, re.M)
        return name, {"tests": tests, "failures": failures, "errors": errors, "skipped": skipped, "skips": skips}, ""
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    problems = check_patterns()
    if problems:
        for p in problems:
            print(f"ERROR: {p}")
        return 1
    if a.list:
        for i, (name, f, _, _) in enumerate(MUTATIONS, 1):
            print(f"{i:2d}. {name} ({f})")
        print(f"{len(MUTATIONS)} mutations, patterns found exactly once")
        return 0
    base = run_one("baseline (unmutated copy)", S, "", "")
    if not isinstance(base[1], dict) or base[1]["failures"] or base[1]["errors"]:
        print("ERROR: the unmutated copy does not pass:", base[2] or base[1])
        return 1
    for s in base[1]["skips"]:
        if not any(v in s for v in MISSING_INPUT.values()):
            print(f"ERROR: baseline skip is not a declared missing input: {s}")
            return 1
    print(f"baseline: {base[1]['tests']} tests, skipped {base[1]['skipped']}")
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as ex:
        results = list(ex.map(lambda m: run_one(*m), MUTATIONS))
    killed, survived, invalid = 0, [], []
    for (name, f, _, _), (_, res, tail) in zip(MUTATIONS, results):
        control = name.startswith("control")
        if not isinstance(res, dict):
            invalid.append(name)
            print(f"INVALID  {name}: {res}")
            continue
        detected = res["failures"] + res["errors"] > 0
        if res["skips"] != base[1]["skips"] and not detected:
            invalid.append(name)
            print(f"INVALID  {name}: different skips without failure")
            continue
        if control:
            print(f"{'SURVIVED' if not detected else 'KILLED  '} {name}")
            if detected:
                invalid.append(name)
            continue
        if detected:
            killed += 1
            print(f"KILLED   {name} ({res['failures']} failures, {res['errors']} errors)")
        else:
            survived.append(name)
            print(f"SURVIVED {name}")
    n = len([m for m in MUTATIONS if not m[0].startswith("control")])
    print(f"mutations killed: {killed}/{n}; survived: {len(survived)}; invalid: {len(invalid)}; "
          f"control survived: {'yes' if not any(x.startswith('control') for x in invalid) else 'no'}")
    ok = killed == n and not invalid
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
