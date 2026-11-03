"""
Controllo delle mutazioni di scripts/nuaa_subject_folds.py, offline e deterministico.

Copia il codice (scripts/nuaa_subject_folds.py, scripts/check_nuaa_subject_folds.py, scripts/m2_writers.py,
scripts/path_guard.py, scripts/package_rules.py e delivery/folds/nuaa_subject_folds.json) in una cartella temporanea,
applica una mutazione dichiarata alla volta (coppia cerca/sostituisci che deve comparire esattamente una volta: se un
modello non si trova più, il controllo si ferma con un errore) ed esegue scripts/check_nuaa_subject_folds.py sulla
copia mutata, con i manifest veri in sola lettura (FOLDS_INPUT_ROOT = radice di questo repository). Una mutazione è
rilevata solo se la suite arriva al riepilogo («tests run») con almeno un fallimento o errore; una suite che non parte
la rende non valida. Una mutazione di controllo (solo un commento) deve sopravvivere. Codice 1 se una mutazione
sopravvive o non è valida, se un salto non è un ingresso mancante dichiarato o se un modello non si trova.

  .venv/bin/python scripts/check_nuaa_subject_folds_mutations.py            # tutte le mutazioni
  .venv/bin/python scripts/check_nuaa_subject_folds_mutations.py --list     # solo l'elenco, con la verifica dei modelli
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
from check_nuaa_subject_folds import MISSING_INPUT  # noqa: E402

TESTS = "scripts/check_nuaa_subject_folds.py"
S = "scripts/nuaa_subject_folds.py"
COPY = (S, TESTS, "scripts/m2_writers.py", "scripts/path_guard.py", "scripts/package_rules.py", "delivery/folds/nuaa_subject_folds.json")
# (nome, file mutato, testo cercato, testo sostitutivo)
MUTATIONS = [
    ("control: comment only (must survive)", S, "# ============================================================================ schema",
     "# ============================================================================ schema (control mutation)"),
    ("strata: bona fide in session 03 ignored", S, '(bf_03 if r["label"] == "0" else att_03).add(r["subject"])', '(bf_03 if r["label"] == "9" else att_03).add(r["subject"])'),
    ("strata: subjects without 01-02 frames treated as having them", S,
     'st = "A" if s in has_0102 and s in bf_03 else "B" if s in has_0102 else "C" if s in bf_03 else "D"',
     'st = "A" if s in bf_03 else "B"'),
    ("dealing: fold size ignored (property count first)", S,
     "key=lambda i: (len(folds[i]), shared(folds[i], s), i))", "key=lambda i: (shared(folds[i], s), i))"),
    ("dealing: shared property ignored (size, then index)", S,
     "key=lambda i: (len(folds[i]), shared(folds[i], s), i))", "key=lambda i: (len(folds[i]), i))"),
    ("dealing: strata order reversed", S, "    for st in STRATA:\n        for s in sorted(x for x, v in props.items()",
     "    for st in reversed(STRATA):\n        for s in sorted(x for x, v in props.items()"),
    ("dealing: bona fide 03 not a shared property", S,
     '        if props[s]["bona_fide_03"]:\n            n += sum(1 for x in fold if props[x]["bona_fide_03"])',
     '        if False:\n            n += sum(1 for x in fold if props[x]["bona_fide_03"])'),
    ("constraint: minimum subjects per property not checked", S, "            if n < MIN_SUBJECTS:", "            if False:"),
    ("constraint: minimum lowered to one", S, "MIN_SUBJECTS = 2 ", "MIN_SUBJECTS = 1 "),
    ("constraint: sessions 01-02 property not checked", S,
     '("sessions_01_02", "frames in sessions 01-02")):', '):'),
    ("roles: dev fold = held-out fold", S, "        dev = i % k + 1\n", "        dev = i\n"),
    ("roles: fitting includes the dev fold", S, "fit = [j for j in range(1, k + 1) if j not in (i, dev)]", "fit = [j for j in range(1, k + 1) if j != i]"),
    ("sampling: shortfall silently truncated", S, "                        if len(rows) < n:\n                            fail(",
     "                        if False:\n                            fail("),
    ("sampling: spread replaced by the first n names", S, 'names = spread([r["key"].split("/", 1)[1] for r in rows], n)',
     'names = sorted(r["key"].split("/", 1)[1] for r in rows)[:n]'),
    ("sampling: test role sampled like dev", S, '                    if role == "test":\n                        chosen += rows',
     '                    if False:\n                        chosen += rows'),
    ("spread: unsorted input", S, "    items = sorted(items)\n    if k >= len(items):", "    items = list(items)\n    if k >= len(items):"),
    ("inputs: manifest SHA-256 not compared", S, 'if h != spec["sha256"]:\n            fail(f"manifest {k}', 'if False:\n            fail(f"manifest {k}'),
    ("inputs: counts not compared", S, "            if spec[x] != v:\n                fail(", "            if False:\n                fail("),
    ("inputs: columns not checked", S, "if r.fieldnames != list(COLUMNS):", "if False:"),
    ("inputs: name/session agreement not checked", S, 'if not m or m.group(1) != x["subject"] or m.group(5) != x["session"]:', "if not m:"),
    ("inputs: duplicated keys accepted", S, "if any(n > 1 for n in keys.values()):", "if False:"),
    ("inputs: official lists overlap accepted", S,
     'if {r["key"] for r in tr} & {r["key"] for r in te} or {r["sha256"] for r in tr} & {r["sha256"] for r in te}:', "if False:"),
    ("inputs: duplicated content inside a list accepted", S, 'if len({r["sha256"] for r in out[k]}) != len(out[k]):', "if False:"),
    ("inputs: cache key outside the official lists accepted", S, 'if o is None or any(o[c] != r[c] for c in COLUMNS):', "if False:"),
    ("inputs: cache row compared on the SHA-256 only", S, 'if o is None or any(o[c] != r[c] for c in COLUMNS):',
     'if o is None or o["sha256"] != r["sha256"]:'),
    ("inputs: cache manifests sharing keys accepted", S, "    if len(cache) != len(set(cache)):", "    if False:"),
    ("inputs: expected n_images not compared", S, '    if len(official) != cfg["expected"]["n_images"]:', "    if False:"),
    ("inputs: test list with sessions 01-02 accepted", S,
     'if any(r["session"] == "03" for r in tr) or any(r["session"] != "03" for r in te):', 'if any(r["session"] == "03" for r in tr):'),
    ("inputs: subject of the name not compared", S, 'if not m or m.group(1) != x["subject"] or m.group(5) != x["session"]:',
     'if not m or m.group(5) != x["session"]:'),
    ("inputs: malformed sha256 accepted", S, 'if not re.fullmatch(r"[0-9a-f]{64}", x["sha256"]) or not x["bytes"].isdigit():',
     'if not x["bytes"].isdigit():'),
    ("config: schema not checked", S, 'if not isinstance(cfg, dict) or cfg.get("schema") != CONFIG_SCHEMA:', 'if not isinstance(cfg, dict):'),
    ("config: duplicated seeds accepted", S, "or len(set(seeds)) != len(seeds) or", "or"),
    ("inputs: symbolic links accepted", S, "    if stat.S_ISLNK(st.st_mode):\n        fail(", "    if False:\n        fail("),
    ("config: missing top-level field accepted", S, '        if key not in cfg:\n            fail(f"configuration: {key} is missing")',
     '        if False:\n            fail(f"configuration: {key} is missing")'),
    ("outputs: availability of another role in the public JSON", S, '"availability": pf["availability"][role]}',
     '"availability": pf["availability"]["test"]}'),
    ("outputs: dirty flag inverted in the public JSON", S, '"dirty": state["dirty"],\n        "manifests"', '"dirty": not state["dirty"],\n        "manifests"'),
    ("outputs: public SHA256SUMS listing one file", S, '        public["SHA256SUMS"] = sums_text(public)',
     '        public["SHA256SUMS"] = sums_text({"fold_schema_public.md": public["fold_schema_public.md"]})'),
    ("outputs: public JSON not scanned for identifiers", S, "    for name, text in public.items():", "    for name, text in public.items():\n        if name.endswith('.json'):\n            continue"),
    ("outputs: run inside the manifests folder accepted", S,
     '        if label.startswith("manifest") and pg.within(run, Path(p).parent):', '        if False:'),
    ("outputs: symbolic output root accepted", S, "    if (out_root if out_root is not None else run.parent).is_symlink():", "    if False:"),
    ("outputs: manifests read before the invalidation", S, "    state = repo_state(root)\n    state[\"code_sha256\"]",
     "    mans = load_manifests(cfg, mdir)\n    state = repo_state(root)\n    state[\"code_sha256\"]"),
    ("outputs: in_cache flag inverted", S, "1 if r[\"key\"] in cache else 0] for r in rows])", "0 if r[\"key\"] in cache else 1] for r in rows])"),
    ("outputs: public manifest SHA-256 of another text", S, '"manifest_sha256": sha256_bytes(text.encode("utf-8"))', '"manifest_sha256": sha256_bytes(b"")'),
    ("declared redundant (must survive): outputs of a previous run not invalidated at the start; the failure path removes "
     "them and write_all is atomic, so no test can observe it; kept for symmetry with the other packages after the m2_writers review of 23 October", S,
     "    left = invalidate_outputs(run, k)\n    if left:", "    left = []\n    if left:"),
    ("outputs: failure leaves the outputs", S, "        left = invalidate_outputs(run, k)\n        reason =", "        left = []\n        reason ="),
    ("outputs: foreign entries in the run folder accepted", S, "                if not any(mw.owned_name(e, n) for n in names):", "                if False:"),
    ("outputs: public identifiers not scanned", S, "        if problems:\n            fail(\"public outputs refused: \"", "        if False:\n            fail(\"public outputs refused: \""),
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
    tmp = Path(tempfile.mkdtemp(prefix="folds_mut_"))
    try:
        for rel in COPY:
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, tmp / rel)
        (tmp / "results" / "folds").mkdir(parents=True)
        p = tmp / f
        p.write_text(p.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
        env = dict(os.environ, FOLDS_INPUT_ROOT=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([sys.executable, "-I", str(tmp / TESTS)], cwd=tmp, capture_output=True, text=True, timeout=600, env=env)
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
        control = name.startswith("control") or name.startswith("declared redundant")
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
    n = len([m for m in MUTATIONS if not (m[0].startswith("control") or m[0].startswith("declared redundant"))])
    print(f"mutations killed: {killed}/{n}; survived: {len(survived)}; invalid: {len(invalid)}; control survived: {'yes' if not invalid or 'control' not in ' '.join(invalid) else 'no'}")
    ok = killed == n and not invalid
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
