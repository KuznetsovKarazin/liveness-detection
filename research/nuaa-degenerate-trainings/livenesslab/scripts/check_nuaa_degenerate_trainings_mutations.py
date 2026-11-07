"""
Controllo delle mutazioni di scripts/nuaa_degenerate_trainings.py (training degeneri dalle curve train/dev), offline e
deterministico.

Copia il codice (scripts/nuaa_degenerate_trainings.py, scripts/check_nuaa_degenerate_trainings.py, i moduli condivisi
importati e, se c'è, la configurazione delivery/degenerate/) in una cartella temporanea, applica una mutazione
dichiarata alla volta (coppia cerca/sostituisci che deve comparire esattamente una volta: se un modello non si trova
più, il controllo si ferma con un errore) ed esegue scripts/check_nuaa_degenerate_trainings.py sulla copia mutata
SENZA --real e con DEGEN_INPUT_ROOT su una cartella vuota: i mutanti non leggono mai i dati veri (regola della macchina
da 16 GB: un processo pesante alla volta). Una mutazione è rilevata solo se la suite arriva al riepilogo («tests run»)
con almeno un fallimento o errore, cioè dal comportamento; una suite che non parte la rende non valida. Una mutazione
di controllo (solo un commento) deve sopravvivere.

Test saltati: la copia non mutata deve passare e può saltare test solo per un motivo dichiarato (MISSING_INPUT di
scripts/check_nuaa_degenerate_trainings.py: qui il test sugli ingressi veri, mai richiesto); i salti sono riportati.
Una mutazione conta come rilevata solo per fallimenti o errori, mai per un salto. Codice 1 se una mutazione
sopravvive o non è valida, se un salto non è dichiarato o se un modello non si trova.

  .venv/bin/python -I scripts/check_nuaa_degenerate_trainings_mutations.py              # tutte, una alla volta
  .venv/bin/python -I scripts/check_nuaa_degenerate_trainings_mutations.py --jobs 2     # in parallelo (al massimo 3)
  .venv/bin/python -I scripts/check_nuaa_degenerate_trainings_mutations.py --list       # solo l'elenco, con la verifica dei modelli
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
from check_nuaa_degenerate_trainings import MISSING_INPUT  # noqa: E402

TESTS = "scripts/check_nuaa_degenerate_trainings.py"
S = "scripts/nuaa_degenerate_trainings.py"
COPY = (S, TESTS, "scripts/m2_writers.py", "scripts/path_guard.py", "scripts/package_rules.py")
CONFIG = "delivery/degenerate/nuaa_degenerate_trainings.json"
MAX_JOBS = 3
# (nome, file mutato, testo cercato, testo sostitutivo)
MUTANTS = [
    # simulazione dell'EarlyStopping
    ("early stopping: patience + 1", S, "if wait >= patience and e > 0:", "if wait >= patience + 1 and e > 0:"),
    ("early stopping: > instead of >= on the patience", S, "if wait >= patience and e > 0:", "if wait > patience and e > 0:"),
    ("early stopping: an equal value counts as an improvement", S,
     'better = (lambda a, b: a < b) if monitor == "val_loss"', 'better = (lambda a, b: a <= b) if monitor == "val_loss"'),
    ("early stopping: start_from_epoch ignored", S, "        if e < start:\n            continue", "        if False:\n            continue"),
    ("early stopping: short histories treated as complete", S,
     "if len(vals) == n_max and best_ep is not None:", "if best_ep is not None:"),
    ("early stopping: a stop at the last epoch not marked as early", S,
     'return {"stop": e + 1, "restored": best_ep + 1, "early": True,', 'return {"stop": e + 1, "restored": best_ep + 1, "early": e + 1 < n_max,'),
    # criteri
    ("degenerate: restored epoch up to 3", S, 'return restored in d["restored_epochs"] and abs(', "return restored <= 3 and abs("),
    ("degenerate: val_acc_at_best not required", S,
     'and abs(val_acc - d["val_acc_at_best"]) < d["tolerance"]', "and True"),
    ("saturation: tolerance 0.01 fixed in the code", S,
     'return abs(v - crit["saturation"]["val_accuracy"]) < crit["saturation"]["tolerance"]', "return abs(v - 0.5) < 0.01"),
    ("recovery: val_loss condition dropped", S, "if va >= acc and vl < loss:", "if va >= acc:"),
    ("recovery: > instead of >= on val_accuracy", S, "if va >= acc and vl < loss:", "if va > acc and vl < loss:"),
    ("recovery: <= instead of < on val_loss", S, "if va >= acc and vl < loss:", "if va >= acc and vl <= loss:"),
    ("dev: a tie at 0.5 counted as attack", S,
     '"acc_05": sum(1 for s, y in zip(scores, labels) if (s > DECISION) == (y == 1))',
     '"acc_05": sum(1 for s, y in zip(scores, labels) if (s >= DECISION) == (y == 1))'),
    ("dev: predicted share with ties as attack", S,
     '"pred_attack_05": sum(1 for s in scores if s > DECISION) / len(scores),', '"pred_attack_05": sum(1 for s in scores if s >= DECISION) / len(scores),'),
    # controlli della scheda
    ("card: closed key set not checked", S, "    if sorted(c) != list(CARD_KEYS[group]):", "    if False:"),
    ("card: recipe not compared", S, "        if c[k] != rc[k] or isinstance(c[k], bool):", "        if isinstance(c[k], bool):"),
    ("card: history length not checked", S, "if not isinstance(h[k], list) or len(h[k]) != n:", "if not isinstance(h[k], list):"),
    ("card: non-finite history values accepted", S, "        if not all(_number(v) for v in h[k]):", "        if False:"),
    ("card: restored epoch not compared with the argmin", S, "    if not is_int(r) or r != argmin:", "    if not is_int(r):"),
    ("card: best values not read at the restored epoch", S,
     'if c["best_val_loss"] != h["val_loss"][r - 1] or c["val_acc_at_best"] != h["val_accuracy"][r - 1]:', "if False:"),
    ("card: replay of the stop not compared", S,
     'if sim is None or sim["stop"] != n or sim["restored"] != r:', "if sim is None:"),
    ("card: restored_by not compared with the replay", S, '    if c["restored_by"] != RESTORED_BY[sim["early"]]:', "    if False:"),
    ("card: weights not compared with the SHA256SUMS", S,
     'if c["weights_file"] != f"{name}_LivenessNet.keras" or not _hex(c["weights_sha256"]) or listed.get(c["weights_file"]) != c["weights_sha256"]:',
     'if not _hex(c["weights_sha256"]):'),
    # dev_scores e soglie
    ("dev: row count not checked", S, '    if len(rows) != spec["dev_rows"]:', "    if False:"),
    ("dev: n_dev of the card not compared", S, '    if card["n_dev"] != spec["dev_rows"]:', "    if False:"),
    ("dev: balance not checked", S, "    if labels.count(0) != labels.count(1):", "    if False:"),
    ("dev: accuracy at 0.5 not compared with val_acc_at_best", S,
     'if abs(d["acc_05"] - card["val_acc_at_best"]) > cfg["criteria"]["checkpoint_agreement_tolerance"]:', "if False:"),
    ("dev: checkpoint of the rows not compared", S, '        if r["checkpoint_sha256"] != card["weights_sha256"]:', "        if False:"),
    ("dev: role of the rows not checked", S, '        if r["role"] != "dev":', "        if False:"),
    ("dev: fold and seed of the rows not checked", S, '        if r["fold"] != str(fold) or r["train_seed"] != str(seed):', "        if False:"),
    ("dev: arm of the rows not checked", S, '        if group == "ablation_run" and r["arm"] != arm:', "        if False:"),
    ("dev: label outside {0, 1} accepted", S, '        if r["label"] not in ("0", "1"):', "        if False:"),
    ("dev: threshold_dev and threshold_sha256 of the rows not compared", S,
     '        if r["threshold_dev"] != repr(value) or r["threshold_sha256"] != thr["sha256"]:', "        if False:"),
    ("dev: threshold_sha256 of the rows not compared", S,
     '        if r["threshold_dev"] != repr(value) or r["threshold_sha256"] != thr["sha256"]:', '        if r["threshold_dev"] != repr(value):'),
    ("dev: prediction_05 not compared with the rule", S, '        if r["prediction_05"] != ("1" if s > DECISION else "0"):', "        if False:"),
    ("dev: prediction_05 with ties as attack", S,
     '        if r["prediction_05"] != ("1" if s > DECISION else "0"):', '        if r["prediction_05"] != ("1" if s >= DECISION else "0"):'),
    ("dev: prediction_dev not compared with the rule", S, '        if r["prediction_dev"] != ("1" if s > value else "0"):', "        if False:"),
    ("dev: prediction_dev with ties as attack", S,
     '        if r["prediction_dev"] != ("1" if s > value else "0"):', '        if r["prediction_dev"] != ("1" if s >= value else "0"):'),
    ("dev: duplicated sample_id accepted", S,
     'if len({k[0] for k in keys}) != len(keys) or len({k[1] for k in keys}) != len(keys):', 'if len({k[1] for k in keys}) != len(keys):'),
    ("dev: duplicated file_sha256 accepted", S,
     'if len({k[0] for k in keys}) != len(keys) or len({k[1] for k in keys}) != len(keys):', 'if len({k[0] for k in keys}) != len(keys):'),
    ("dev: dev_keys_sha256 of the card not compared", S,
     '    if sha256_bytes("\\n".join(k[0] for k in keys).encode()) != card["dev_keys_sha256"]:', "    if False:"),
    ("dev: dev_scores_sha256 of the threshold not compared", S,
     '    if sha256_bytes("\\n".join(repr(float(r["score_attack"])) for r in rows).encode()) != thr["dev_scores_sha256"]:', "    if False:"),
    ("dev: scores outside [0, 1] accepted", S, "if not math.isfinite(s) or not 0.0 <= s <= 1.0:", "if not math.isfinite(s):"),
    ("dev: models of a fold with different devs accepted", S, "        if dev_of_fold.setdefault(fold, keys) != keys:", "        if False:"),
    ("dev: models of a fold compared by counts only", S, "        if dev_of_fold.setdefault(fold, keys) != keys:",
     "        if len(dev_of_fold.setdefault(fold, keys)) != len(keys):"),
    ("thresholds: order of the times not checked", S, "    if not saved <= frozen <= tested:", "    if False:"),
    ("thresholds: arm not compared", S, '    if group == "ablation_run" and thr.get("arm") != arm:', "    if False:"),
    ("thresholds: fold and seed not compared", S, '    if thr.get("fold") != fold or thr.get("train_seed") != seed:', "    if False:"),
    ("thresholds: checkpoint not compared", S, '    if thr.get("checkpoint_sha256") != card["weights_sha256"]:', "    if False:"),
    ("thresholds: value not checked", S, '    if not _number(thr.get("value")):', "    if False:"),
    # schede: un controllo per condizione
    ("card: arch not checked", S, '    if c["arch"] != "LivenessNet":', "    if False:"),
    ("card: fold not compared with the file name", S, '    if c["fold"] != fold:', "    if False:"),
    ("card: seed not compared with the file name", S, '    if c["train_seed"] != seed:', "    if False:"),
    ("card: arm not compared with the file name", S, '    if "arm" in c and c["arm"] != arm:', "    if False:"),
    ("card: n_fitting < 1 accepted", S, '    if not is_int(c["n_fitting"]) or c["n_fitting"] < 1:', '    if not is_int(c["n_fitting"]):'),
    ("card: n_dev < 2 accepted", S, '    if not is_int(c["n_dev"]) or c["n_dev"] < 2:', '    if not is_int(c["n_dev"]):'),
    ("card: accuracy outside [0, 1] accepted", S, '    if not all(0 <= v <= 1 for k in ("accuracy", "val_accuracy") for v in h[k]):', "    if False:"),
    ("card: accuracy above 1 accepted", S,
     '    if not all(0 <= v <= 1 for k in ("accuracy", "val_accuracy") for v in h[k]):', '    if not all(0 <= v for k in ("accuracy", "val_accuracy") for v in h[k]):'),
    ("card: negative loss accepted", S, '    if not all(v >= 0 for k in ("loss", "val_loss") for v in h[k]):', "    if False:"),
    ("card: non-positive lr accepted", S, '    if not all(v > 0 for v in h["lr"]):', "    if False:"),
    # ingressi e formati
    ("inputs: symbolic link accepted", S, "    if stat.S_ISLNK(st.st_mode):\n        fail(", "    if False:\n        fail("),
    ("inputs: CSV rows with a wrong number of fields accepted", S, "    if any(None in x or None in x.values() for x in rows):", "    if False:"),
    ("inputs: a name listed twice in a SHA256SUMS accepted", S, "        if m.group(2) in out:", "        if False:"),
    ("inputs: two wall-time lines accepted", S, "        if len(walls) != 1:", "        if len(walls) < 1:"),
    ("inputs: kept_complete_models of the failed attempt not compared", S,
     '    if sorted(st.get("kept_complete_models") or []) != names:', "    if False:"),
    ("rules: saturation tolerance widened", S,
     'return abs(v - crit["saturation"]["val_accuracy"]) < crit["saturation"]["tolerance"]',
     'return abs(v - crit["saturation"]["val_accuracy"]) <= crit["saturation"]["tolerance"] + 1e-3'),
    ("rules: first improvement counts an equal val_loss", S,
     'if h["val_loss"][i] < h["val_loss"][0]), None)', 'if h["val_loss"][i] <= h["val_loss"][0]), None)'),
    ("public: IPv4 addresses not scanned", S,
     '    if any(ip not in pr.IP_ALLOW for ip in (m.group(0) for m in pr.IPV4.finditer(text))):', "    if False:"),
    ("public: markers not scanned in the Markdown", S, "    if markdown and pr.MARKERS.search(text):", "    if False:"),
    ("public: the subject id value printed in the reason", S, '            bad.append("subject id")', '            bad.append(f"subject id {s}")'),
    ("public: write_public leaves the texts after an error", S,
     "        for n in list(texts) + list(blobs):\n            (d / n).unlink(missing_ok=True)", "        for n in []:\n            (d / n).unlink(missing_ok=True)"),
    ("run: symbolic output root accepted", S, "    if out_root.is_symlink():", "    if False:"),
    # ancore
    ("anchors: listed files not compared with the SHA256SUMS", S, "    if sha256_bytes(b) != listed[fn]:", "    if False:"),
    ("anchors: status.json not compared with the configuration", S, '    if sha256_bytes(sb) != spec["status_sha256"]:', "    if False:"),
    ("anchors: restricted SHA256SUMS of a run not compared", S,
     '    if sha256_bytes(rb) != spec["restricted_sums_sha256"]:\n        fail(f"{group} restricted',
     '    if False:\n        fail(f"{group} restricted'),
    ("anchors: run.log not compared", S, '    if sha256_bytes(lb) != spec["run_log_sha256"]:', "    if False:"),
    ("anchors: status PASS or FAIL not required", S,
     '    if st.get("status") != want or st.get("passed") is not (want == "PASS") or st.get("run_id") != spec["run_id"]:',
     '    if st.get("run_id") != spec["run_id"]:'),
    ("anchors: extra cards in the SHA256SUMS accepted", S, "    if cards != names:", "    if not set(names) <= set(cards):"),
    ("anchors: failed-attempt cards not compared with the configuration", S, '        if listed.get(fn) != spec["cards"][fn]:', "        if False:"),
    ("anchors: failed-attempt card outside the run accepted", S, "        if name not in main:", "        if False:"),
    ("anchors: wall time of run.log not compared with status.json", S,
     'if not _number(st.get("wall_seconds")) or abs(walls[0] - st["wall_seconds"]) > WALL_TOL:', "if False:"),
    # tabelle
    ("tables: steps per epoch rounded down", S, 'm["steps"] = math.ceil(c["n_fitting"] / rc["batch"])', 'm["steps"] = c["n_fitting"] // rc["batch"]'),
    ("tables: boundary cases with restored epoch < 3", S,
     'if m["card"]["restored_epoch"] <= crit["boundary_restored_epoch_max"]]}', 'if m["card"]["restored_epoch"] < crit["boundary_restored_epoch_max"]]}'),
    ("tables: lr decision read at the epoch after", S,
     'fired_sat = sum(1 for m, e in zip(grp, firsts) if e and half(m["h"]["val_accuracy"][e - 2]))',
     'fired_sat = sum(1 for m, e in zip(grp, firsts) if e and half(m["h"]["val_accuracy"][e - 1]))'),
    ("tables: overhead divided by all models", S, "over = (inp.wall - tot) / len(main)", "over = (inp.wall - tot) / len(main + fr)"),
    ("tables: identical history without comparing the values", S,
     '"identical_history": same_len and mx == 0.0,', '"identical_history": same_len,'),
    ("tables: recovery steps with a fixed batch count", S,
     '"median_steps": median([m["rec"] * m["steps"] for m in grp if m["rec"]]),', '"median_steps": median([m["rec"] * 10 for m in grp if m["rec"]]),'),
    # pubblico e run
    ("public: subject identifiers not scanned", S, "    for s in sorted(inp.subjects):", "    for s in []:"),
    ("public: subject scan without word boundaries", S,
     'if re.search(rf"(?<![\\w.]){re.escape(s)}(?![\\w.])", text):', "if re.search(re.escape(s), text):"),
    ("public: figure labels not scanned", S, '    scan["figure labels"] = json.dumps(fig_data)', "    pass"),
    ("public: written without the scan", S, "    for name, text in scan.items():", "    for name, text in {}.items():"),
    ("public: the Markdown not scanned", S,
     'if probs := public_problems(text, inp, name.endswith(".md")):',
     'if probs := (public_problems(text, inp) if not name.endswith(".md") else []):'),
    ("public: figure skip not declared (figures required)", S,
     '        return {}, f"skipped: missing Python package {e.name!r}", data', "        raise"),
    ("run: previous outputs not invalidated at the start", S,
     "        left = invalidate_outputs(run)\n        if left:", "        left = []\n        if left:"),
    ("run: a failure leaves the outputs", S, "        left = invalidate_outputs(run)\n        if log is not None:",
     "        left = []\n        if log is not None:"),
    ("run: foreign entries in the run folder accepted", S, "                if not any(mw.owned_name(e, n) for n in names):", "                if False:"),
    ("run id: dirty tree without -dev", S, '+ ("-dev" if state["dirty"] else "")', '+ ""'),
    ("run: authorisation not required", S, "    if not a.i_am_authorized:", "    if False:"),
    ("run: SHA256SUMS of the public without the figures", S,
     '    public["SHA256SUMS"] = sums_text(public)', '    public["SHA256SUMS"] = sums_text({k: v for k, v in public.items() if isinstance(v, str)})'),
]
# mutazione di controllo: solo un commento; deve SOPRAVVIVERE (prova che la suite rileva il comportamento, non il testo)
CONTROL = [("control: comment only", S, "def median(v):\n", "def median(v):  # control mutant: comment only\n")]
# controlli dichiarati ridondanti: devono SOPRAVVIVERE, perché un altro controllo li copre sempre
# (una cartella del run dentro un ingresso è rifiutata anche da m2_writers.output_problem, «the output folder is ... or
# inside it»; il controllo locale resta come prima difesa, con un messaggio più chiaro)
REDUNDANT = [("declared redundant (must survive): run folder inside an input", S,
              "        if pg.within(p, run) or (Path(p).is_dir() and pg.within(run, p)):", "        if pg.within(p, run):")]


def copy_tree(dst: Path) -> Path:
    code = dst / "code"
    for rel in COPY + ((CONFIG,) if (ROOT / CONFIG).is_file() else ()):
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, code / rel)
    return code


def run_tests(code: Path, tmp: Path) -> tuple:
    tmp.mkdir(parents=True, exist_ok=True)
    empty = tmp / "no_real_inputs"
    empty.mkdir(exist_ok=True)
    # mai --real; radice degli ingressi veri su una cartella vuota: i mutanti non leggono dati veri
    env = {**os.environ, "DEGEN_INPUT_ROOT": str(empty), "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp)}
    r = subprocess.run([sys.executable, "-I", TESTS], cwd=code, env=env, capture_output=True, text=True, timeout=900)
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("tests run")), "no test summary")
    skips = tuple(sorted(l for l in r.stdout.splitlines() if l.startswith("skipped ")))
    return r.returncode, line, skips


def undeclared_skips(skips) -> list:
    """Salti il cui motivo non è dichiarato (riga «skipped <test>: missing input: <motivo>»)."""
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
    ap = argparse.ArgumentParser(description="Controllo delle mutazioni di nuaa_degenerate_trainings.py (offline, senza dati veri)")
    ap.add_argument("--list", action="store_true", help="elenca le mutazioni e verifica i modelli, senza eseguire i test")
    ap.add_argument("--jobs", type=int, default=1, help=f"mutazioni eseguite in parallelo (predefinito 1, al massimo {MAX_JOBS})")
    a = ap.parse_args()
    if not 1 <= a.jobs <= MAX_JOBS:
        print(f"ERROR: --jobs must be between 1 and {MAX_JOBS}", file=sys.stderr)
        return 2
    problems = [f"{n}: the pattern occurs {k} times in {t} (expected exactly once)"
                for n, t, old, _ in MUTANTS + CONTROL + REDUNDANT if (k := (ROOT / t).read_text(encoding="utf-8").count(old)) != 1]
    if len({n for n, *_ in MUTANTS}) != len(MUTANTS):
        problems.append("two mutations share a name")
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
    with tempfile.TemporaryDirectory(prefix="degen-mutations-") as tmp:
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
                print(f"ERROR: skip that is not declared: {s}", file=sys.stderr)
            print("RESULT: FAIL (the unmutated copy skips tests for a reason that is not declared)")
            return 1
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            cfuts = [ex.submit(one, tmp, len(MUTANTS) + i, n, t, old, new, base_skips) for i, (n, t, old, new) in enumerate(CONTROL + REDUNDANT)]
            futs = [ex.submit(one, tmp, i, n, t, old, new, base_skips) for i, (n, t, old, new) in enumerate(MUTANTS)]
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
    print(f"control mutations survived: {survived_controls}/{len(CONTROL) + len(REDUNDANT)} ({len(CONTROL)} control, {len(REDUNDANT)} declared redundant)")
    ok = killed == len(MUTANTS) and survived_controls == len(CONTROL) + len(REDUNDANT)
    print("RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
