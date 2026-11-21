"""
Controllo delle mutazioni di scripts/pad_arch_run.py (i due run NUAA del confronto LivenessNet / ResNet-18, il controllo
dello stimatore e il controllo degli ingressi), offline e deterministico, in due stadi.

Copia il codice (lo script, la suite scripts/check_pad_arch_run.py, i moduli importati pad_arch_comparison.py,
pad_arch_definitions.py, pad_arch_models.py, nuaa_bn_recalibration.py, nuaa_bn_test.py, nuaa_folds_run.py,
nuaa_session_ablation_run.py, nuaa_patience_verify.py, m2_writers.py, path_guard.py, package_rules.py, le suite che
forniscono i mondi sintetici check_pad_arch_comparison.py, check_nuaa_folds_run.py, check_nuaa_session_ablation_run.py,
check_nuaa_bn_recalibration.py, check_nuaa_bn_test.py, e il file di LivenessNet del codice di riferimento
src/livedetection/src/architectures.py) in una cartella temporanea, applica una mutazione dichiarata alla volta (coppia
cerca/sostituisci che deve comparire esattamente una volta: se un modello non si trova più, il controllo si ferma con un
errore) ed esegue la suite sulla copia mutata. Quasi tutte le mutazioni toccano lo script; quelle dei controlli dei
thread che lo script prende da pad_arch_comparison.py (variabili d'ambiente, valori richiesti, rilettura, TensorFlow già
inizializzato) toccano quel modulo, solo nella copia. La configurazione consegnata NON è copiata: il suo test confronta gli
SHA-256 fissati con i file; nella copia quel test è saltato con il motivo dichiarato (SKIP_OK).

Due stadi, per ogni mutazione (con la suite in modalità failfast, -f):
  1. senza TensorFlow: la suite gira in un processo figlio in cui TensorFlow è nascosto (un finder di sys.meta_path che
     per tensorflow e i suoi sottomoduli dà None, scritto nella cartella temporanea: la stessa tecnica di
     livenesslab/tools/run_without_tensorflow.py, senza dipendere da quel percorso); i test che richiedono TensorFlow
     sono saltati con il loro motivo dichiarato. Una mutazione con un fallimento o un errore è uccisa qui.
  2. con TensorFlow, solo per le mutazioni sopravvissute allo stadio 1.
Il rapporto elenca le mutazioni uccise senza TensorFlow e quelle che lo richiedono, e per ciascuna se è stata uccisa da
un'asserzione fallita (almeno un fallimento) o solo da un errore (un'eccezione nel test). In un ambiente senza TensorFlow lo
stadio 2 non si può eseguire: lo strumento stampa le mutazioni che restano «not verifiable without TensorFlow», non le
conta mai come uccise ed esce con RESULT: INCOMPLETE (codice 2). La copia non mutata, la mutazione di controllo (solo un
commento) e le mutazioni dichiarate ridondanti (REDUNDANT, con il motivo) girano complete nei due stadi e devono
sopravvivere in entrambi; una suite che non arriva al riepilogo, o che salta test per un motivo diverso da quelli della
copia non mutata nello stesso stadio, rende la mutazione non valida.

Nessun dato reale: la suite usa solo i mondi sintetici (frame 8×8, modelli Keras minuscoli su CPU in cartelle
temporanee). Le mutazioni girano una alla volta (un solo processo TensorFlow per volta).

  .venv/bin/python -I -B scripts/check_pad_arch_run_mutations.py            # tutte le mutazioni, in due stadi
  .venv/bin/python -I -B scripts/check_pad_arch_run_mutations.py --list     # solo l'elenco, con la verifica dei modelli
"""
import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = "scripts/check_pad_arch_run.py"
S = "scripts/pad_arch_run.py"
COPY = (S, TESTS, "scripts/pad_arch_comparison.py", "scripts/pad_arch_definitions.py", "scripts/pad_arch_models.py",
        "scripts/nuaa_bn_recalibration.py", "scripts/nuaa_bn_test.py", "scripts/nuaa_folds_run.py", "scripts/nuaa_session_ablation_run.py",
        "scripts/nuaa_patience_verify.py", "scripts/m2_writers.py", "scripts/path_guard.py", "scripts/package_rules.py",
        "scripts/check_pad_arch_comparison.py", "scripts/check_nuaa_folds_run.py", "scripts/check_nuaa_session_ablation_run.py",
        "scripts/check_nuaa_bn_recalibration.py", "scripts/check_nuaa_bn_test.py", "src/livedetection/src/architectures.py")
SKIP_OK = ("missing input: delivery/archrun/pad_arch_run.json is not present",
           "missing input: TensorFlow is not installed: the tests that build, train or re-estimate a Keras model are skipped")
HIDE_TF = '''"""Esegue uno script con TensorFlow nascosto (find_spec None per tensorflow e i suoi sottomoduli)."""
import runpy
import sys
from pathlib import Path


class Hide:
    def __init__(self, inner):
        self.inner = inner

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "tensorflow":
            return None
        f = getattr(self.inner, "find_spec", None)
        return f(name, path, target) if f else None

    def __getattr__(self, k):
        return getattr(self.inner, k)


sys.meta_path[:] = [Hide(f) for f in sys.meta_path]
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(script).resolve().parent))
runpy.run_path(script, run_name="__main__")
'''
# (nome, file mutato, testo cercato, testo sostitutivo)
MUTANTS = [
    # configurazione
    ("runs in another order accepted", S, '    if cfg["runs"] != list(RUNS):', "    if False:"),
    ("machine rule fields not fixed", S, '    if mr["fields"] != list(MACHINE_FIELDS) or mr["versions"]', '    if mr["versions"]'),
    ("history difference as a boolean accepted", S, 'if hc["fields"] != list(HISTORY_KEYS) or hc["difference"] != 0 or isinstance(hc["difference"], bool):',
     'if hc["fields"] != list(HISTORY_KEYS) or hc["difference"] != 0:'),
    ("history consequence not required", S, '    if "stays in the comparison" not in hc["consequence"] or "continues" not in hc["consequence"]:', "    if False:"),
    ("estimator orders not fixed", S, '    if es["orders"] != expected_orders():', "    if False:"),
    ("cap unit not fixed", S, '    if cp["unit"] != CAP_UNIT:', "    if False:"),
    ("choices not declared as the author", S, '    if "choice of the author" not in v:', "    if False:"),
    ("test text not fixed", S, '    if cfg["test"] != TEST_READ:\n        fail("configuration: test must state exactly', '    if False:\n        fail("configuration: test must state exactly'),
    ("authorisation text not required", S, '    if "merge" not in cfg["authorisation"] or "go-ahead" not in cfg["authorisation"]:', "    if False:"),
    ("operations text not required", S, '    if "nohup" not in cfg["operations"] or "commit" not in cfg["operations"]:', "    if False:"),
    ("machine scope text not required", S, '    if "not the identity of the physical unit" not in mr["scope"]:', "    if False:"),
    ("frame_ids text not required", S, '    if "public outputs" not in de["frame_ids"] or "restricted" not in de["frame_ids"]:', "    if False:"),
    ("discarded state not required in cap.on_cap", S, '    if "INCOMPLETE" not in cp["on_cap"] or DISCARDED not in cp["on_cap"]:', '    if "INCOMPLETE" not in cp["on_cap"]:'),
    ("estimator cap unit not fixed", S, '    if ec["cap_unit"] != EST_CAP_UNIT:', "    if False:"),
    # file fissati, ordine, tetti
    ("pinned file SHA-256 not compared", S, '    if sha256_bytes(b) != e["sha256"]:', "    if False:"),
    ("code pins not compared", S, "    bad = [f for f in CODE_FILES if got[f] != cfg[\"code\"][f]]", "    bad = []"),
    ("run_order.csv not compared with the regenerated order", S, '    if ob.decode("utf-8", "replace") != order_text(pins.cfg_pr):', "    if False:"),
    ("positions not compared with the plan", S, "        if got != (str(i), net, str(f), str(s), str(patience)):", "        if False:"),
    ("reference run of a position not compared", S, '        if r["reference_run"] != (ref if net == "LivenessNet" else "none"):', "        if False:"),
    ("run_caps.csv not compared with the regenerated caps", S, '    if cb.decode("utf-8", "replace") != caps_text(pins.cfg_pr, pins.trial):', "    if False:"),
    ("cap of the plan not used", S, "    budget = SR.Budget(cap_h * 60)", "    budget = SR.Budget(10 ** 6)"),
    ("trial not required to be PASS", S, '    if pins.trial["status"] != "PASS" or pins.trial["run_id"] != tr["run_id"]:', '    if pins.trial["run_id"] != tr["run_id"]:'),
    ("trial record files not hashed", S, '        if h != tr[key]:\n            fail(f"trial record {fn}', '        if False:\n            fail(f"trial record {fn}'),
    ("trial record not anchored to the package", S, '    bad = [entry for key, entry in PACKAGE_ENTRIES.items() if sums.get(entry) != tr[key]]', "    bad = []"),
    ("LivenessNet order not compared with PR #27", S,
     '    if pins.bn_cfg["batchnorm"]["layers"] != cfg["estimator"]["orders"]["LivenessNet"] or pins.bn_cfg["evaluation"]["batch"] != cfg["estimator"]["batch"]:',
     "    if False:"),
    ("estimator cap not compared with PR #27", S, '    if pins.bn_cfg["budget"]["wall_time_cap_minutes"] != ec["cap_minutes"]:', "    if False:"),
    ("PR #27 configuration not compared with PR #28", S,
     '        if bn_cfg != pins.bn_cfg or pins.bt_cfg["bnrecal_run"]["config_sha256"] != ec["bn_recalibration_config"]["sha256"]:', "        if False:"),
    ("BNRECAL run and models not compared with PR #28", S,
     '        if pins.bt_cfg["bnrecal_run"]["run_id"] != ec["bnrecal_run_id"] or pins.bt_cfg["bnrecal_run"]["models"] != ec["models"]:', "        if False:"),
    # regola della macchina e dei thread
    ("machine rule not applied", S, '    bad = [f for part in ("machine", "versions") for f, v in rec[part].items() if not v["equal"]]', "    bad = []"),
    ("versions not compared", S, '    bad = [f for part in ("machine", "versions") for f, v in rec[part].items() if not v["equal"]]',
     '    bad = [f for part in ("machine",) for f, v in rec[part].items() if not v["equal"]]'),
    ("absent or null machine field equal", S, "    return f in a and f in b and a[f] is not None and b[f] is not None and a[f] == b[f]",
     "    return a.get(f) == b.get(f)"),
    ("thread rule not applied", S, '    if not rule["equal"]:\n        bad = [f for f, v in rule["fields"].items()', '    if False:\n        bad = [f for f, v in rule["fields"].items()'),
    ("thread rule on the effective threads only", S, '    return {"fields": rec, "equal": all(v["equal"] for v in rec.values()),',
     '    return {"fields": rec, "equal": rec["effective"]["equal"],'),
    ("absent thread field equal", S, '"equal": f in tt and f in threads and tt[f] == threads[f]} for f in THREAD_FIELDS}',
     '"equal": tt.get(f) == threads.get(f)} for f in THREAD_FIELDS}'),
    ("null thread values equal", S, '        if rec[f]["trial"] is None or rec[f]["now"] is None:', "        if False:"),
    ("threads not set by the runner", S, "        threads = PAC.set_threads(tf, pins.cfg_pr)            # subito dopo l'import, prima di qualunque operazione",
     '        threads = {"requested": dict(pins.trial["threads"]["requested"]), "effective": PAC.read_threads(tf), "set_by_this_runner": True, '
     '"environment": env, "logical_cpus": os.cpu_count()}'),
    ("thread environment variables not checked before the folder", S,
     "    env = thread_environment_rule(pins.cfg_pr, pins.trial)   # variabili dei thread diverse dalla prova: rifiuto prima di ogni cartella",
     '    env = {"TF_NUM_INTRAOP_THREADS": None, "TF_NUM_INTEROP_THREADS": None}'),
    ("thread environment compared with the trial only by check-inputs", S,
     "    env = thread_environment_rule(pins.cfg_pr, pins.trial)   # variabili dei thread diverse dalla prova: rifiuto prima di ogni cartella",
     "    env = PAC.thread_environment(pins.cfg_pr)"),
    ("thread environment not compared with the trial by check-inputs", S,
     "    env = thread_environment_rule(pins.cfg_pr, pins.trial)\n    machine_rule(pins.trial)",
     "    env = PAC.thread_environment(pins.cfg_pr)\n    machine_rule(pins.trial)"),
    ("thread environment different from the trial accepted", S, "    if env != want:", "    if False:"),
    ("TF_NUM_INTRAOP_THREADS not read", "scripts/pad_arch_comparison.py", "    for k in THREAD_KEYS:\n        var = TF_THREAD_ENV[k]",
     "    for k in THREAD_KEYS[1:]:\n        var = TF_THREAD_ENV[k]"),
    ("TF_NUM_INTEROP_THREADS not read", "scripts/pad_arch_comparison.py", "    for k in THREAD_KEYS:\n        var = TF_THREAD_ENV[k]",
     "    for k in THREAD_KEYS[:1]:\n        var = TF_THREAD_ENV[k]"),
    ("requested intra-op threads not required to be 10", "scripts/pad_arch_comparison.py", "    for k in THREAD_KEYS:\n        v = th[k]",
     "    for k in THREAD_KEYS[1:]:\n        v = th[k]"),
    ("requested inter-op threads not required to be 10", "scripts/pad_arch_comparison.py", "    for k in THREAD_KEYS:\n        v = th[k]",
     "    for k in THREAD_KEYS[:1]:\n        v = th[k]"),
    ("effective intra-op threads not read back", "scripts/pad_arch_comparison.py", '"intra_op": tf.config.threading.get_intra_op_parallelism_threads()',
     '"intra_op": 10'),
    ("effective inter-op threads not read back", "scripts/pad_arch_comparison.py", '"inter_op": tf.config.threading.get_inter_op_parallelism_threads()',
     '"inter_op": 10'),
    ("TensorFlow initialised before the threads accepted", "scripts/pad_arch_comparison.py",
     '        fail(f"the TensorFlow threads cannot be set ({e}): TensorFlow was initialised before the runner set them")', "        pass"),
    ("processor not compared", S, 'MACHINE_FIELDS = ("processor", "core_levels", "ram_gb", "os", "logical_cpus", "architecture")',
     'MACHINE_FIELDS = ("core_levels", "ram_gb", "os", "logical_cpus", "architecture")'),
    ("RAM not compared", S, 'MACHINE_FIELDS = ("processor", "core_levels", "ram_gb", "os", "logical_cpus", "architecture")',
     'MACHINE_FIELDS = ("processor", "core_levels", "os", "logical_cpus", "architecture")'),
    ("machine rule not applied in check-estimator", S, "    mrule = machine_rule(pins.trial)\n",
     '    mrule = machine_rule(pins.trial) if kind == "run" else {"equal": True}\n'),
    ("GPU not hidden in run", S, "        FR.hide_gpu(tf)\n        ctx.cfg, ctx.cfg_pr", "        ctx.cfg, ctx.cfg_pr"),
    ("GPU not hidden in check-estimator", S, "        FR.hide_gpu(tf)\n        order, batch", "        order, batch"),
    ("threads not read back after a model", S, '    rec["threads_after"] = PAC.check_threads(ctx.tf, ctx.cfg_pr, f"after {name}")', '    rec["threads_after"] = None'),
    # controllo dello stimatore richiesto dal run
    ("estimator record of another configuration accepted", S, '    if doc.get("config_sha256") != cfg_sha:', "    if False:"),
    ("estimator record of other code accepted", S, '    if doc.get("code_sha256") != code:', "    if False:"),
    ("estimator record not PASS accepted", S, '    if st.get("status") != "PASS" or not isinstance(st.get("run_id"), str) or not EST_ID_RE.match(st["run_id"]):',
     '    if not isinstance(st.get("run_id"), str) or not EST_ID_RE.match(st["run_id"]):'),
    ("schema and status of estimator_check.json not checked", S,
     '    if doc.get("schema") != ESTIMATOR_SCHEMA or doc.get("status") != "PASS" or doc.get("run_id") != st["run_id"]:', "    if False:"),
    ("estimator record with a failed model accepted", S, ' or any(r.get("recalibration_pass") is not True for r in rows):', ":"),
    ("estimator record without the machine rule accepted", S, '    if (doc.get("machine_rule") or {}).get("equal") is not True or (doc.get("threads_rule") or {}).get("equal") is not True:',
     "    if False:"),
    ("unofficial estimator record accepted by an official run", S, '    if official and doc.get("official") is not True:', "    if False:"),
    ("MISMATCH of check-estimator reported as PASS", S, '    if status == "PASS" and any(not r["recalibration_pass"] for r in rows):', "    if False:"),
    ("estimator status not used by check-estimator", S, "        status = estimator_status(status, rows)", "        pass"),
    # schede di riferimento e chiavi
    ("patience of the reference card not checked", S,
     '    if (card.get("arm"), card.get("fold"), card.get("train_seed"), card.get("early_stopping_patience")) != (ARM, f, s, patience):',
     '    if (card.get("arm"), card.get("fold"), card.get("train_seed")) != (ARM, f, s):'),
    ("reference card not required in the SHA256SUMS", S, '            if fn not in sums:\n                fail(f"accepted run', '            if False:\n                fail(f"accepted run'),
    ("reference card SHA-256 not compared", S, '            if sha256_bytes(b) != sums[fn]:', "            if False:"),
    ("reference card keys not checked before the runs", S, "            PAC.keys_check(ref[\"card\"], role, rows)", "            pass"),
    # ruoli e frame
    ("test role accepted by unit_frames", S, "    if role not in ROLES:\n        fail(f\"role {role!r} refused", "    if False:\n        fail(f\"role {role!r} refused"),
    ("assembled roles not checked", S, '    inp.assembled.append(role)\n    PAC.assert_no_test(inp)', '    inp.assembled.append(role)'),
    ("BN states not compared", S, '        D.same_frames(prepared[D.BN_STATES[0]][u], prepared[D.BN_STATES[1]][u], "dev frames, the two BN states")', "        pass"),
    ("networks of a unit not compared", S, '        prepared[st] = {pu["unit"]: pu for pu in D.prepare_units(complete)} if complete else {}',
     '        prepared[st] = {pu["unit"]: pu for pu in D.prepare_units([{**c, "networks": {n: c["networks"]["LivenessNet"] for n in c["networks"]}} for c in complete])} if complete else {}'),
    # storia
    ("history always identical", S, '"outcome": "different" if firsts else "identical"', '"outcome": "identical"'),
    ("number of epochs ignored by the history check", S, "    if n_ref != n_new:\n        firsts.append(min(n_ref, n_new) + 1)", "    pass"),
    ("last differing epoch reported as the first", S, '"first_differing_epoch": min(firsts) if firsts else None, "largest_abs_difference"',
     '"first_differing_epoch": max(firsts) if firsts else None, "largest_abs_difference"'),
    ("last differing epoch of a field reported as the first", S, '    return {**d, "first_differing_epoch": min(firsts) if firsts else None,',
     '    return {**d, "first_differing_epoch": max(firsts) if firsts else None,'),
    ("different history stops the run", S, '        card["history_check"] = rec["history_check"]',
     '        card["history_check"] = rec["history_check"]\n        if rec["history_check"]["outcome"] == "different":\n            fail("history")'),
    ("non-finite epoch not counted as differing", S, '    firsts = [x for x in (d["first_differing_epoch"], cut, over) if x is not None]',
     '    firsts = [x for x in (d["first_differing_epoch"], over) if x is not None]'),
    ("unrepresentable difference not counted as differing", S, '    firsts = [x for x in (d["first_differing_epoch"], cut, over) if x is not None]',
     '    firsts = [x for x in (d["first_differing_epoch"], cut) if x is not None]'),
    ("earlier difference hidden by a non-finite value", S, "    d = PAC.history_differences(list(ref[:m]), list(new[:m]))", "    d = PAC.history_differences([], [])"),
    ("tie of the largest difference to the last field", S, 'key=lambda k: (fields[k]["max_abs_difference"], -HISTORY_KEYS.index(k))',
     'key=lambda k: (fields[k]["max_abs_difference"], HISTORY_KEYS.index(k))'),
    ("timing history not compared with Keras", S, "    return a == b\n", "    return True\n"),
    ("non-finite new history not stopped", S, '    if bad := non_finite_history(card["history"]):', "    if False:"),
    # scheda, checkpoint, punteggi
    ("network of the card not checked", S, '    out = [] if card.get("arch") == network else', '    out = [] if True else'),
    ("val_loss of the checkpoint not checked", S, "    if not (diff <= FR.VAL_LOSS_TOL[0] or diff <= FR.VAL_LOSS_TOL[1] * card[\"best_val_loss\"]):", "    if False:"),
    ("checkpoint overwritten", S, "    if os.path.lexists(path):\n        fail(f\"{path.name} already exists", "    if False:\n        fail(f\"{path.name} already exists"),
    ("EER threshold replaced by 0.5", S, '    thr = D.condition_threshold("dev_eer", y, s) if oc["evaluable"] else None',
     '    thr = float(FR.FIXED_THRESHOLD) if oc["evaluable"] else None'),
    ("threshold computed without both classes", S, '    thr = D.condition_threshold("dev_eer", y, s) if oc["evaluable"] else None', '    thr = D.condition_threshold("dev_eer", y, s)'),
    ("EER cross-check removed", S, '    if eer is not None and float(eer["value"]) != thr:', "    if False:"),
    ("tie at 0.5 decided attack", S, "    d05 = D.attack_decisions(s, FR.FIXED_THRESHOLD)\n    deer = D.attack_decisions(s, thr)",
     "    d05 = s >= FR.FIXED_THRESHOLD\n    deer = D.attack_decisions(s, thr)"),
    ("tie at the EER threshold decided attack", S, "    deer = D.attack_decisions(s, thr) if thr is not None else None\n    n = len(y)",
     "    deer = (s >= thr) if thr is not None else None\n    n = len(y)"),
    ("CSV prediction at 0.5 with >=", S, "    d05 = D.attack_decisions(p, FR.FIXED_THRESHOLD)", "    d05 = p >= FR.FIXED_THRESHOLD"),
    ("CSV prediction at the EER threshold with >=", S, "    deer = D.attack_decisions(p, thr) if thr is not None else None", "    deer = (p >= thr) if thr is not None else None"),
    # accordo dei valutatori
    ("evaluator agreement ignored", S, '    ok = diff <= tol_p and d05 <= tol_d and (deer is None or deer <= tol_d)', "    ok = True"),
    ("agreement without the EER threshold", S, '    ok = diff <= tol_p and d05 <= tol_d and (deer is None or deer <= tol_d)', "    ok = diff <= tol_p and d05 <= tol_d"),
    ("agreement without the 0.5 decisions", S, '    ok = diff <= tol_p and d05 <= tol_d and (deer is None or deer <= tol_d)', "    ok = diff <= tol_p and (deer is None or deer <= tol_d)"),
    ("discordances counted as agreements", S, "    d05 = int(np.sum(D.attack_decisions(a, FR.FIXED_THRESHOLD) != D.attack_decisions(b, FR.FIXED_THRESHOLD)))",
     "    d05 = int(np.sum(D.attack_decisions(a, FR.FIXED_THRESHOLD) == D.attack_decisions(b, FR.FIXED_THRESHOLD)))"),
    ("evaluator gap always 0", S, "    diff = float(np.max(np.abs(a - b))) if len(a) else 0.0\n    d05", "    diff = 0.0\n    d05"),
    ("different shapes accepted", S, "    if a.shape != b.shape:\n        return", "    if False:\n        return"),
    ("phase-1 scores from the bona fide column", S, '    rec["scorer_p"] = p[:, 1].copy()', '    rec["scorer_p"] = p[:, 0].copy()'),
    ("MISMATCH not raised", S, "        if mismatches:\n            status = \"MISMATCH\"", "        if False:\n            status = \"MISMATCH\""),
    ("public outputs written with a MISMATCH", S, '        if status != "MISMATCH":\n            doc =', '        if True:\n            doc ='),
    # tetto, arresto, fasi
    ("no cap check before a training", S, '    ctx.budget.check(f"train {name}")', "    pass"),
    ("no cap check at the end of an epoch", S, '                                                 lambda step: ctx.budget.check(f"{step} of {name}"), timer)',
     '                                                 lambda step: None, timer)'),
    ("no cap check before the dev scoring", S, '    ctx.budget.check(f"dev scoring {name}")', "    pass"),
    ("no cap check before the re-estimation", S, '    ctx.budget.check(f"re-estimation {name}")', "    pass"),
    ("no cap check before the ordinary dev", S, '    ctx.budget.check(f"ordinary dev scoring {name}")', "    pass"),
    ("re-estimation without cap checks", S, "    stats = reestimate(copy, fit, order, batch, ctx.budget, name)", "    stats = reestimate(copy, fit, order, batch, None, name)"),
    ("no cap check before the consistency pass", S, '    ctx.budget.check(f"consistency pass {name}")', "    pass"),
    ("no cap check before the re-estimated dev", S, '    ctx.budget.check(f"re-estimated dev scoring {name}")', "    pass"),
    ("no cap check in check-estimator", S, '                budget.check(f"estimator {name}")', "                pass"),
    ("check-estimator re-estimation without cap checks", S, "                stats = reestimate(copy, fit, order, batch, budget, name)",
     "                stats = reestimate(copy, fit, order, batch, None, name)"),
    ("no cap check before the consistency pass of check-estimator", S, '                budget.check(f"consistency pass {name}")', "                pass"),
    ("no cap check before the dev of check-estimator", S, '                budget.check(f"dev after {name}")', "                pass"),
    ("SIGUSR1 handler not installed in run", S, "    budget = SR.Budget(cap_h * 60)\n    SR.signal_handlers(budget)", "    budget = SR.Budget(cap_h * 60)"),
    ("model in progress kept on a FAIL", S, '            discard_model(run / RESTRICTED, cur["name"])', "            pass"),
    ("no test role check at the end of the run", S, "        PAC.assert_no_test(inp)\n        frames = check_dev_frames(", "        frames = check_dev_frames("),
    ("model in progress kept at the cap", S, '                discard_model(ctx.rdir, ctx.current["name"])', "                pass"),
    ("discarded model reported as complete", S, "                mark_discarded(ctx.current)", "                pass"),
    ("complete model discarded at a stop in phase 2", S, '            if ctx.current is not None and not ctx.current.get("phase_1"):\n                stopped["discarded"]',
     '            if ctx.current is not None:\n                stopped["discarded"]'),
    ("INCOMPLETE reported with exit code 0", S, 'STATUS_CODE = {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2, "MISMATCH": 4}',
     'STATUS_CODE = {"PASS": 0, "FAIL": 1, "INCOMPLETE": 0, "MISMATCH": 4}'),
    # ristima
    ("BN order not checked on the graph", S, "    if probs := bn_order_problems(list(order), AM.model_topology(copy)):", "    if False:"),
    ("ancestor order reversed", S, "if pos[a] > pos[n]]", "if pos[a] < pos[n]]"),
    ("only direct inputs as ancestors", S, "        stack.extend(topology[n][1])", "        pass"),
    ("trained tensors not checked", S, '    if probs := BR.tensor_problems(snap, BR.snapshot_weights(copy), BR.moving_keys(copy, order), stats):', "    if False:"),
    ("consistency not checked", S, '    if bad := {n: c["channels_outside_tolerance"] for n, c in cons.items() if c["channels_outside_tolerance"]}:\n        fail(f"{name}: consistency check failed: {bad}',
     '    if False:\n        fail(f"{name}: consistency check failed: {bad}'),
    ("consistency counts not checked", S, '    if any(c["count"] != stats[n]["count"] for n, c in cons.items()):', "    if False:"),
    ("softmax after the re-estimation not checked", S, '        if r["softmax_gap"] > sg:', "        if False:"),
    ("softmax of the ordinary state not checked", S, '    if ev_ord["softmax_gap"] > BR.tol(bn_cfg, "softmax_of_logits_max_abs_diff"):', "    if False:"),
    ("copy different before the re-estimation accepted (run)", S,
     '    if BR.snapshot_weights(copy) != snap:\n        fail(f"{name}: the in-memory copy differs from the original before the re-estimation")\n    stats = reestimate(copy, fit, order, batch, ctx.budget',
     '    if False:\n        fail(f"{name}: the in-memory copy differs from the original before the re-estimation")\n    stats = reestimate(copy, fit, order, batch, ctx.budget'),
    ("checks of PR #27 not applied in check-estimator", S, "                cons = pr27_checks(name, stats, snap, copy, order, fit_after, final, dev_after, bn_cfg)",
     '                cons = BR.consistency(stats, final, 1.0, 1.0)'),
    ("loaded checkpoint not checked", S, "    check_loaded(orig, ctx.cfg_pr, net)", "    pass"),
    # uscite, stato, riverifiche
    ("statistics in float64 in the file", S, '"installed_moving_mean_float32": [float(x) for x in s["mean32"]],', '"installed_moving_mean_float32": [float(x) for x in s["mean64"]],'),
    ("layer order rearranged in the statistics file", S, "    lay = {}\n    for n in order:\n        s, c = res[\"stats\"][n], res[\"cons\"][n]",
     "    lay = {}\n    for n in reversed(order):\n        s, c = res[\"stats\"][n], res[\"cons\"][n]"),
    ("phase_2 column equal to phase_1", S, '"phase_1": bool(rec.get("phase_1")), "phase_2": bool(rec.get("phase_2")),\n         "epochs_run"',
     '"phase_1": bool(rec.get("phase_1")), "phase_2": bool(rec.get("phase_1")),\n         "epochs_run"'),
    ("one-class class hidden", S, '    return oc["class"] if oc["one_class"] else "no"', '    return "yes" if oc["one_class"] else "no"'),
    ("public SHA256SUMS empty", S, '    files = {**texts, "SHA256SUMS": FR.sums_text(texts)}', '    files = {**texts, "SHA256SUMS": ""}'),
    ("public outputs not scanned", S, "    for n, text in texts.items():\n        if probs := PAC.output_problems(text, inp):\n            fail(f\"{n} refused: \" + \"; \".join(probs))\n    files =",
     "    files ="),
    ("run.log not scanned", S, '        if probs := PAC.output_problems((run / LOG).read_text(encoding="utf-8", errors="replace"), inp):', "        if False:"),
    ("configuration not re-verified at the end", S, '    if sha256_bytes(read_regular(cfg_p, "configuration")) != cfg_sha:\n        fail("the configuration changed during the run")',
     '    if False:\n        fail("the configuration changed during the run")'),
    ("code not re-verified at the end", S, '    if check_code(cfg) != pins.code:\n        fail("the code changed during the run")', '    if False:\n        fail("the code changed during the run")'),
    ("reference cards not re-verified at the end", S, '            if sha256_bytes(read_regular(BR.parent_path(run, ref["model"], "card.json"), "reference card")) != ref["sha256"]:',
     "            if False:"),
    ("trial record not re-verified at the end", S, '        if sha256_bytes(read_regular(folder / fn, f"trial record {fn}")) != cfg["trial_record"][key]:', "        if False:"),
    ("estimator record not re-verified at the end", S, '        if sha256_bytes(b) != pins.estimator["estimator_check_json_sha256"]:', "        if False:"),
    ("started missing from the final status", S, '        write_status(run, {**head, "status": status, "passed": status == "PASS", "started": started, "finished": finished,\n                              "wall_seconds": round(wall, 1), "process_cpu_seconds": cpu_total, **extra, "stopped": stopped,\n                              "complete_phase_1"',
     '        write_status(run, {**head, "status": status, "passed": status == "PASS", "finished": finished,\n                              "wall_seconds": round(wall, 1), "process_cpu_seconds": cpu_total, **extra, "stopped": stopped,\n                              "complete_phase_1"'),
    ("restricted note ignores the SHA256SUMS written", S, '    if (rdir / "SHA256SUMS").is_file():\n        return', '    if False:\n        return'),
    ("existing run folder not refused", S, "    if os.path.lexists(run):\n        return f\"the run folder", "    if False:\n        return f\"the run folder"),
    ("symbolic link output root accepted", S, '    if out_root.is_symlink():\n        return "the output root is a symbolic link"', '    if False:\n        return "the output root is a symbolic link"'),
    ("dirty tree accepted", S, '    if state["dirty"]:\n        fail("the git work tree is not clean: the runs', '    if False:\n        fail("the git work tree is not clean: the runs'),
    ("estimator record not required by run", S, '        if not a.estimator_check:\n            fail("run needs --estimator-check', '        if False:\n            fail("run needs --estimator-check'),
    ("one-class counts with the wrong patience", S, '    records = [{"network": r["network"], "patience": patience,', '    records = [{"network": r["network"], "patience": 12,'),
    # tentativi e radice d'uscita
    ("attempt rule not applied", S, "    if attempt != need:", "    if False:"),
    ("earlier attempts of other runs counted", S, '        if not loose or (kind == "run" and loose.group(1) != str(RUNS[run])):', "        if not loose:"),
    ("attempt number of a suffix ignored", S, "        k = int(g) if g else 1", "        k = 1"),
    ("unofficial output root without the flag", S, '    if not unofficial_flag:\n        fail("--out-root other', '    if False:\n        fail("--out-root other'),
    ("flag accepted on the official root", S, '        if unofficial_flag:\n            fail("--unofficial-out-root with', '        if False:\n            fail("--unofficial-out-root with'),
    ("official root compared by string", S, "    if pg.within(out_root, official) and pg.within(official, out_root):",
     "    if os.path.abspath(out_root) == os.path.abspath(official):"),
    ("non-official root inside the repository accepted", S, "    if pg.within(out_root, repo) or pg.within(repo, out_root):", "    if False:"),
    ("root containing the repository accepted", S, "    if pg.within(out_root, repo) or pg.within(repo, out_root):", "    if pg.within(out_root, repo):"),
    ("malformed entries of the output root ignored", S, '            fail(f"output root: the entry {e!r} looks like',
     '            continue\n            fail(f"output root: the entry {e!r} looks like'),
    ("suffix -r1 accepted", S, r'RUN_ID_RE = re.compile(r"^\d{8}-ARCHRUN-p(4|12)-[0-9a-f]{7,40}(?:-r([2-9]|[1-9]\d+))?$")',
     r'RUN_ID_RE = re.compile(r"^\d{8}-ARCHRUN-p(4|12)-[0-9a-f]{7,40}(?:-r([1-9]\d*))?$")'),
    ("personal paths in the command", S, '    return f"<outside the repository>/{Path(p).name}"', "    return str(p)"),
    ("command built from the raw command line", S, '    parts = [a.cmd, "--config", path_text(folder_arg(a.config))]', '    parts = [a.cmd, "--config", str(a.config)]'),
    ("command not scanned before the folder", S, "    if probs := PAC.output_problems(command, FR.Inputs()):", "    if False:"),
    ("status.json not scanned", S, "    if probs := PAC.output_problems(text, inp):\n        if not redact:", "    if False:\n        if not redact:"),
    ("personal text not redacted in the FAIL status", S, '            text = text.replace(t, "<redacted>")', "            pass"),
    ("package SHA256SUMS not re-verified at the end", S, '              cfg["estimator_check"]["bn_test_config"], cfg["trial_record"]["package_sums"]):',
     '              cfg["estimator_check"]["bn_test_config"]):'),
]
# solo un commento: deve sopravvivere
CONTROL = [("comment only (control)", S, "# ============================================================================ uscite\n",
            "# ============================================================================ uscite (mutante di controllo)\n")]
# controlli ridondanti, dichiarati: la mutazione deve sopravvivere perché un altro controllo dice la stessa cosa
REDUNDANT = [
    ("thread rule not recorded in the run status (redundant: after thread_environment_rule before the folder, PAC.set_threads after the import "
     "and PAC.load_trial on the trial record, the comparison with the trial cannot differ, and it is recorded on every path that reaches it)", S,
     '        extra["threads_rule"] = threads_rule(pins.trial, threads)\n        require_threads(extra["threads_rule"])\n        print(',
     '        require_threads(threads_rule(pins.trial, threads))\n        extra["threads_rule"] = threads_rule(pins.trial, threads)\n        print('),
    ("thread rule not applied in check-estimator (redundant: environment variables refused before the folder by thread_environment_rule, "
     "requested and effective threads by PAC.set_threads, set_by_this_runner and the trial values by PAC.load_trial; require_threads itself is "
     "tested on the rule)", S,
     '        extra["threads_rule"] = threads_rule(pins.trial, threads)\n        require_threads(extra["threads_rule"])\n        FR.hide_gpu(tf)\n        order,',
     '        extra["threads_rule"] = threads_rule(pins.trial, threads)\n        FR.hide_gpu(tf)\n        order,'),
    ("threads read back before a training (redundant: read back after set_threads, compared with the trial, and after the previous model)", S,
     '    rec["threads_before"] = PAC.check_threads(ctx.tf, ctx.cfg_pr, f"before {name}")', '    rec["threads_before"] = None'),
    ("keys of a position re-checked in phase 1 (redundant: check_unit_keys checks the same cards and rows before the first training)", S,
     '        PAC.keys_check(ref["card"], role, part["rows"])', "        pass"),
    ("frame counts of the card (redundant: n_fitting and n_dev are the lengths of the assembled rows, whose number check_unit_keys compares with the configuration)", S,
     '    if card["n_fitting"] != ctx.pcfg["arms"][ARM]["total"] or card["n_dev"] != ctx.pcfg["dev"]["total"]:', "    if False:"),
    ("frame counts of the manifests (redundant: the keys of the same rows are compared with the reference card)", S,
     "            if len(rows) != n:", "            if False:"),
    ("original model unchanged after the re-estimation (redundant: the re-estimation works on a clone; tensor_problems checks the copy)", S,
     '    if BR.snapshot_weights(orig) != snap:\n        fail(f"{name}: the original model changed during the re-estimation")',
     '    if False:\n        fail(f"{name}: the original model changed during the re-estimation")'),
    ("number of timing epochs (redundant: the repr lists of the two histories differ in length when the numbers differ)", S,
     ' or len(epochs) != card["epochs_run"]:', ":"),
    ("number of models in check-estimator (redundant: load_pins compares the number with the PR #28 configuration and BT.load_bnrecal the parent order with it)", S,
     '        if len(inp.order) != ec["models"]:', "        if False:"),
    ("fold schema in check-estimator (redundant: nuaa_bn_recalibration.load_inputs refuses a loaded fold schema)", S,
     "        if inp.folds:\n            fail(\"the fold schema was loaded: this command", "        if False:\n            fail(\"the fold schema was loaded: this command"),
    ("copy different before the re-estimation in check-estimator (redundant: clone_model plus set_weights; the same check in the run is tested and the tensors are checked after)", S,
     '                if BR.snapshot_weights(copy) != snap:\n                    fail(f"{name}: the in-memory copy differs',
     '                if False:\n                    fail(f"{name}: the in-memory copy differs'),
]


def pairs(x):
    return x if isinstance(x, tuple) else (x,)


def copy_tree(dest: Path) -> Path:
    code = dest / "code"
    for rel in COPY:
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, code / rel)
    (dest / "hide_tensorflow.py").write_text(HIDE_TF, encoding="utf-8")
    return code


def run_tests(code: Path, tmp: Path, with_tf: bool, failfast: bool = False) -> tuple:
    tmp.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp), "CUDA_VISIBLE_DEVICES": "-1", "TF_CPP_MIN_LOG_LEVEL": "2"}
    env.pop("TF_NUM_INTRAOP_THREADS", None)
    env.pop("TF_NUM_INTEROP_THREADS", None)
    cmd = [sys.executable, "-I", "-B"] + ([] if with_tf else [str(code.parent / "hide_tensorflow.py")]) + [TESTS] + (["-f"] if failfast else [])
    r = subprocess.run(cmd, cwd=code, env=env, capture_output=True, text=True, timeout=7200)
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("tests run")), "no test summary")
    skips = tuple(sorted(l for l in r.stdout.splitlines() if l.startswith("skipped ")))
    return r.returncode, line, skips


def detected(rc: int, line: str, skips=(), base_skips=()) -> str:
    """«killed» (con il modo: «by a failed assertion» se almeno un fallimento, «by an error» se solo errori),
    «survived» o «invalid»."""
    m = re.match(r"tests run: (\d+), failures: (\d+), errors: (\d+), skipped: (\d+)", line)
    if not m:
        return "invalid"
    if int(m.group(2)) > 0:
        return "killed by a failed assertion"
    if int(m.group(3)) > 0:
        return "killed by an error"
    return "survived" if rc == 0 and tuple(skips) == tuple(base_skips) else "invalid"


def one(tmp: Path, i: int, target: str, old, new, with_tf: bool, base_skips=(), failfast: bool = False) -> tuple:
    work = tmp / f"work{i}_{'tf' if with_tf else 'notf'}"
    code = copy_tree(work)
    f = code / target
    text = f.read_text(encoding="utf-8")
    for o, n in zip(pairs(old), pairs(new)):
        text = text.replace(o, n, 1)
    f.write_text(text, encoding="utf-8")
    rc, line, skips = run_tests(code, work / "tmp", with_tf, failfast)
    shutil.rmtree(work, ignore_errors=True)
    return detected(rc, line, skips, base_skips), line, rc


def baseline(tmp: Path, with_tf: bool) -> tuple:
    rc, line, skips = run_tests(copy_tree(tmp / f"base_{'tf' if with_tf else 'notf'}"), tmp / "base" / "tmp", with_tf)
    label = "with TensorFlow" if with_tf else "without TensorFlow"
    print(f"unmutated copy {label}: {TESTS} exit {rc} ({line})", flush=True)
    for s in skips:
        print(f"unmutated copy {label}: {s}", flush=True)
    ok = not rc and re.fullmatch(r"tests run: \d+, failures: 0, errors: 0, skipped: \d+", line) and \
        all(s.split(": ", 1)[-1] in SKIP_OK for s in skips)
    return bool(ok), skips


def main() -> int:
    ap = argparse.ArgumentParser(description="Controllo delle mutazioni dei run NUAA del confronto tra architetture (offline, sintetico, due stadi)")
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
    tf_here = importlib.util.find_spec("tensorflow") is not None
    killed_notf, killed_tf, survived, invalid, not_verifiable, by_assert, by_error = [], [], [], [], [], [], []
    controls_ok = 0
    with tempfile.TemporaryDirectory(prefix="archrun-mutations-") as tmp:
        tmp = Path(tmp)
        ok0, skips0 = baseline(tmp, False)
        ok1, skips1 = baseline(tmp, True) if tf_here else (True, ())
        if not ok0 or not ok1:
            print("RESULT: FAIL (the tests do not pass on the unmutated copy, or skip for an undeclared reason)")
            return 1
        for i, (n, t, old, new) in enumerate(CONTROL + REDUNDANT):
            v0, line0, _ = one(tmp, 1000 + i, t, old, new, False, skips0)
            v1, line1 = ("not run", "TensorFlow not installed")
            if tf_here:
                v1, line1, _ = one(tmp, 1000 + i, t, old, new, True, skips1)
            good = v0 == "survived" and (v1 == "survived" if tf_here else True)
            controls_ok += good
            print(f"CONTROL   {'SURVIVED' if good else 'NOT SURVIVED'} {n} (without TensorFlow: {v0}, {line0}; with TensorFlow: {v1}, {line1})", flush=True)
        for i, (n, t, old, new) in enumerate(MUTANTS):
            v, line, rc = one(tmp, i, t, old, new, False, skips0, failfast=True)
            if v.startswith("killed"):
                killed_notf.append(n)
                (by_error if v.endswith("error") else by_assert).append(n)
                print(f"KILLED without TensorFlow, {v[7:]}  {n} ({line})", flush=True)
                continue
            if v == "invalid":
                invalid.append(n)
                print(f"INVALID without TensorFlow {n} ({line}, exit {rc})", flush=True)
                continue
            if not tf_here:
                not_verifiable.append(n)
                print(f"NOT VERIFIABLE without TensorFlow {n} (survives without TensorFlow; TensorFlow is not installed)", flush=True)
                continue
            v, line, rc = one(tmp, i, t, old, new, True, skips1, failfast=True)
            if v.startswith("killed"):
                killed_tf.append(n)
                (by_error if v.endswith("error") else by_assert).append(n)
                print(f"KILLED with TensorFlow, {v[7:]}  {n} ({line})", flush=True)
            else:
                {"survived": survived, "invalid": invalid}[v].append(n)
                print(f"{v.upper()} with TensorFlow  {n} ({line})", flush=True)
    print(f"mutations killed without TensorFlow: {len(killed_notf)}/{len(MUTANTS)}")
    print(f"mutations killed only with TensorFlow: {len(killed_tf)}/{len(MUTANTS)}" + ("" if tf_here else " (stage not run: TensorFlow not installed)"))
    for n in killed_tf:
        print(f"  needs TensorFlow: {n}")
    print(f"killed by a failed assertion: {len(by_assert)}; killed by an error: {len(by_error)}")
    for n in by_error:
        print(f"  killed by an error: {n}")
    if not_verifiable:
        print(f"mutations not verifiable without TensorFlow (never counted as killed): {len(not_verifiable)}")
        for n in not_verifiable:
            print(f"  not verifiable: {n}")
    if survived or invalid:
        print(f"survived: {survived}; invalid: {invalid}")
    print(f"control mutations survived: {controls_ok}/{len(CONTROL) + len(REDUNDANT)} ({len(CONTROL)} control, {len(REDUNDANT)} declared redundant)")
    if not_verifiable and not survived and not invalid and controls_ok == len(CONTROL) + len(REDUNDANT):
        print("RESULT: INCOMPLETE (some mutations need TensorFlow to be verified)")
        return 2
    ok = len(killed_notf) + len(killed_tf) == len(MUTANTS) and controls_ok == len(CONTROL) + len(REDUNDANT)
    print("RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
