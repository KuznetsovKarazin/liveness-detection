"""
Controllo delle mutazioni di scripts/nuaa_bn_test.py (confronto prima/dopo sul test della ristima delle BatchNormalization
della PR #27), offline e deterministico, con la stessa meccanica di check_nuaa_bn_recalibration_mutations.py.

Copia il codice (lo script e la sua suite, nuaa_bn_recalibration.py e la sua suite, il runner genitore e la sua suite,
nuaa_folds_run.py e la sua suite, che forniscono il mondo sintetico, m2_writers.py, path_guard.py, package_rules.py e,
se ci sono, le configurazioni consegnate delivery/bntest/nuaa_bn_test.json e delivery/bnrecal/nuaa_bn_recalibration.json)
in una cartella temporanea, applica una mutazione dichiarata alla volta (coppia cerca/sostituisci che deve comparire
esattamente una volta: se un modello non si trova più, il controllo si ferma con un errore) ed esegue
scripts/check_nuaa_bn_test.py sulla copia mutata. Una mutazione è rilevata solo se la suite arriva al riepilogo («tests
run») con almeno un fallimento o errore; una suite che non parte la rende non valida. La mutazione di controllo (solo un
commento) e le mutazioni dichiarate ridondanti devono sopravvivere. Solo nuaa_bn_test.py è mutato: lo stimatore della PR
#27 resta il file fissato per SHA-256.

Nessun dato reale: la suite usa solo il mondo sintetico (modelli Keras minuscoli e frame 8×8 in una cartella
temporanea). Le mutazioni girano una alla volta (macchina da 16 GB: un solo processo TensorFlow per volta), con la suite
in modalità failfast (-f); la copia non mutata, il mutante di controllo e i ridondanti girano completi.

  .venv/bin/python -I -B scripts/check_nuaa_bn_test_mutations.py            # tutte le mutazioni, una alla volta
  .venv/bin/python -I -B scripts/check_nuaa_bn_test_mutations.py --list     # solo l'elenco, con la verifica dei modelli
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
TESTS = "scripts/check_nuaa_bn_test.py"
S = "scripts/nuaa_bn_test.py"
COPY = ("scripts/nuaa_bn_test.py", "scripts/check_nuaa_bn_test.py", "scripts/nuaa_bn_recalibration.py", "scripts/check_nuaa_bn_recalibration.py",
        "scripts/nuaa_session_ablation_run.py", "scripts/check_nuaa_session_ablation_run.py", "scripts/nuaa_folds_run.py",
        "scripts/check_nuaa_folds_run.py", "scripts/m2_writers.py", "scripts/path_guard.py", "scripts/package_rules.py")
CONFIGS = (("delivery", "bntest", "nuaa_bn_test.json"), ("delivery", "bnrecal", "nuaa_bn_recalibration.json"))
SKIP_OK = "missing input: delivery/bntest/nuaa_bn_test.json is not present"
# (nome, file mutato, testo cercato, testo sostitutivo)
MUTANTS = [
    # configurazione
    ("unknown configuration key accepted", S, "    if sorted(cfg) != sorted(TOP_KEYS):", "    if False:"),
    ("test statement not compared", S, '    if cfg["test"] != TEST_READ:', "    if False:"),
    ("post hoc declaration not checked", S, '    if "post hoc" not in de["post_hoc"] or "not an independent validation" not in de["post_hoc"]:', "    if False:"),
    ("threshold declaration not checked", S, '    if "0.5" not in de["threshold"]:', "    if False:"),
    ("estimator pin of the configuration not compared with the code", S, "    if co[ESTIMATOR_FILE] != ESTIMATOR_SHA256:", "    if False:"),
    ("shared sections not compared with PR #27", S, "    diff = [k for k in SHARED if shared_part(cfg, k) != shared_part(bn_cfg, k) or not isinstance(cfg[k], dict)",
     "    diff = [k for k in SHARED if False"),
    ("key sets of the shared sections not compared", S, "            or sorted(cfg[k]) != sorted(bn_cfg[k])]", "            or False]"),
    ("rewritten keys extended to numeric keys", S, 'SHARED_REWRITTEN = {"parent_run": ("used",), "evaluation": ("probability",)}',
     'SHARED_REWRITTEN = {"parent_run": ("used", "models", "config_sha256"), "evaluation": ("probability", "batch", "device")}'),
    ("rewritten keys not required as texts", S, '            _text(cfg[s].get(k), f"{s}.{k}")', "            pass"),
    ("FAIL statement not checked", S,
     '    if not isinstance(sr_["fail"], str) or "FAIL" not in sr_["fail"] or "status.json" not in sr_["fail"] or "run.log" not in sr_["fail"]:',
     "    if False:"),
    ("test frames per fold not compared", S, '    if got != cfg["comparison"]["test_frames_per_fold"] or any(n == 0 for n in got):', "    if False:"),
    ("estimator left out of the shared sections", S, 'SHARED = ("parent_run", "architecture", "batchnorm", "estimator", ',
     'SHARED = ("parent_run", "architecture", "batchnorm", '),
    ("PR #27 tolerances not compared", S, '    tdiff = [k for k in BR.TOLERANCE_KEYS if tol[k] != bn_cfg["tolerances"][k]]', "    tdiff = []"),
    ("PR #27 configuration SHA-256 not compared", S, '    if sha256_bytes(bb) != br["config_sha256"]:', "    if False:"),
    ("bootstrap constants not compared", S, "    if {k: bo[k] for k in SR.BOOT} != SR.BOOT:", "    if False:"),
    ("threshold of the comparison not compared", S,
     '    if cp["threshold"] != FR.FIXED_THRESHOLD or isinstance(cp["threshold"], bool) or cp["decision"] != FR.DECISION_RULE:',
     '    if cp["decision"] != FR.DECISION_RULE:'),
    ("causal wording accepted", S, "    if banned:\n", "    if False:\n"),
    ("cap above the coordinator's limit accepted", S, 'not 1 <= bu["wall_time_cap_minutes"] <= 240', 'not 1 <= bu["wall_time_cap_minutes"]'),
    ("upper estimate not below the cap", S, '    if not bu["estimate_minutes"] <= bu["upper_estimate_minutes"] < bu["wall_time_cap_minutes"]:',
     '    if not bu["estimate_minutes"] <= bu["upper_estimate_minutes"]:'),
    ("reading placeholders not checked", S, "        if any(ph not in rr[k] for ph in READING_PLACEHOLDERS):", "        if False:"),
    ("count placeholders not checked", S, "    if any(ph not in rr[\"count\"] for ph in COUNT_PLACEHOLDERS):", "    if False:"),
    ("environment of PR #27 not compared", S, '    if bn_cfg["environment"]["tensorflow"] != cfg["environment"]["tensorflow"]:', "    if False:"),
    # ingressi
    ("code pins not compared", S, '    bad = [f for f in CODE_FILES if got[f] != cfg["code"][f]]', "    bad = []"),
    ("note SHA-256 not compared", S, '        if h != cfg["note_sha256"]:', "        if False:"),
    ("null note accepted by load_inputs", S, "        if need_note:\n            fail(", "        if False:\n            fail("),
    ("parent test scores not hashed", S, "        if h != sums[fn]:\n", "        if False:\n"),
    ("BNRECAL status not required PASS", S, '    if st.get("status") != "PASS" or st.get("run_id") != br["run_id"]', '    if st.get("run_id") != br["run_id"]'),
    ("BNRECAL completed models not compared", S, '    if st.get("completed") != inp.order or st.get("baseline_failed") != [] or len(inp.order) != br["models"]:',
     '    if len(inp.order) != br["models"]:'),
    ("BNRECAL SHA256SUMS not anchored", S, "        if sha256_bytes(b) != br[key]:", "        if False:"),
    ("BNRECAL file SHA-256 not compared", S, "        if sha256_bytes(data) != sums[sub][fn]:", "        if False:"),
    ("BNRECAL public_SHA256SUMS.sha256 not compared", S,
     '    if read_listed(RESTRICTED, "public_SHA256SUMS.sha256").decode("utf-8", "replace") != ',
     '    if False and read_listed(RESTRICTED, "public_SHA256SUMS.sha256").decode("utf-8", "replace") != '),
    ("BNRECAL code_sha256 not compared", S, 'any(code.get(f) != cfg["code"][f] for f in CODE_FILES)', "False"),
    ("BNRECAL groups not compared", S, '    if any(by[n]["group"] != inp.groups[n] for n in inp.order):', "    if False:"),
    ("dev worsened list not compared", S, '    if sorted(worsened) != sorted(cfg["dev_worsened"]["expected"]):', "    if False:"),
    ("dev worsened flags not checked as 0 or 1", S, '    if any(by[n][f] not in ("0", "1") for n in inp.order for f in flags):', "    if False:"),
    ("gained criterion counted as a dev worsening", S, '    flags = [f for f in BR.FLAGS if f != "gained_criterion"]', "    flags = list(BR.FLAGS)"),
    ("BNRECAL dev rows order not compared", S, '        if [r["sample_id"] for r in dev] != want or any(', "        if any("),
    ("BNRECAL statistics of another checkpoint accepted", S, '        if js.get("weights_sha256") != m["weights_sha256"] or ', "        if "),
    ("non-finite BNRECAL statistics accepted", S,
     "            if a.ndim != 1 or a.shape != v.shape or not (np.all(np.isfinite(a)) and np.all(np.isfinite(v))):",
     "            if a.ndim != 1 or a.shape != v.shape:"),
    # baseline del test (fase 1)
    ("test probabilities not compared", S, '"test_prob_pass": d <= pt and dis <= dt})', '"test_prob_pass": dis <= dt})'),
    ("test discordant decisions not counted", S, '"test_prob_pass": d <= pt and dis <= dt})', '"test_prob_pass": d <= pt})'),
    ("tie rule >= in the test discordance", S, "        dis = int(np.sum((p_eval > FR.FIXED_THRESHOLD) != (stored > FR.FIXED_THRESHOLD)))",
     "        dis = int(np.sum((p_eval >= FR.FIXED_THRESHOLD) != (stored >= FR.FIXED_THRESHOLD)))"),
    ("test tolerances fixed in the code", S,
     '    pt, dt, sg = tol(cfg, "test_probability_max_abs_diff"), tol(cfg, "test_decisions_discordant_at_0_5"), tol(cfg, "softmax_of_logits_max_abs_diff")',
     "    pt, dt, sg = 1e-5, 0, 1e-4"),
    ("test softmax outside the verdict", S, '    c["test_pass"] = bool(c["test_frames_pass"] and c["test_prob_pass"] and c["test_softmax_pass"])',
     '    c["test_pass"] = bool(c["test_frames_pass"] and c["test_prob_pass"])'),
    ("test checks outside the baseline verdict", S, '"baseline_pass": bool(dc["dev_pass"] and tc["test_pass"])', '"baseline_pass": bool(dc["dev_pass"])'),
    ("dev checks outside the baseline verdict", S, '"baseline_pass": bool(dc["dev_pass"] and tc["test_pass"])', '"baseline_pass": bool(tc["test_pass"])'),
    ("stored sample ids and order not compared", S, '    if [r["sample_id"] for r in rows] != part["keys"]:', "    if False:"),
    ("stored session not compared", S,
     '    elif any((r["label"], r["subject_id"], r["session_id"], r["file_sha256"]) != (q["label"], q["subject"], q["session"], q["sha256"])',
     '    elif any((r["label"], r["subject_id"], r["file_sha256"]) != (q["label"], q["subject"], q["sha256"])'),
    ("stored threshold of another file accepted", S, 'm["weights_sha256"], m["threshold"]["sha256"]) for r in rows):',
     'm["weights_sha256"], r["threshold_sha256"]) for r in rows):'),
    ("stored prediction_05 not compared", S, '    if any(r["prediction_05"] != ("1" if v > FR.FIXED_THRESHOLD else "0") for r, v in zip(rows, s)):', "    if False:"),
    ("stored test scores not re-hashed in phase 1", S, "    if sha256_bytes(data) != h:\n        fail(f\"parent {p.name}: SHA-256 changed since phase 0\")",
     "    if False:\n        fail(f\"parent {p.name}: SHA-256 changed since phase 0\")"),
    ("stored scores outside [0, 1] accepted", S, "    if not np.all(np.isfinite(s)) or np.any(s < 0) or np.any(s > 1):\n        fail(f\"parent {p.name}",
     "    if False:\n        fail(f\"parent {p.name}"),
    ("baseline stop rule not applied", S, '            if failed:\n                status = "BASELINE_FAILED"', '            if False:\n                status = "BASELINE_FAILED"'),
    # ristima e controllo contro la PR #27 (fase 2)
    ("recalibration stop rule not applied", S, '                if mismatch:\n                    status = "RECALIBRATION_MISMATCH"',
     '                if False:\n                    status = "RECALIBRATION_MISMATCH"'),
    ("statistics not compared with PR #27", S, "            out_n += int(np.sum(~(d <= rel * np.abs(theirs.astype(np.float64)) + abs_)))", "            out_n += 0"),
    ("statistics comparison blind to NaN", S, "            out_n += int(np.sum(~(d <= rel * np.abs(theirs.astype(np.float64)) + abs_)))",
     "            out_n += int(np.sum(d > rel * np.abs(theirs.astype(np.float64)) + abs_))"),
    ("absolute statistics tolerance dropped", S, "            out_n += int(np.sum(~(d <= rel * np.abs(theirs.astype(np.float64)) + abs_)))",
     "            out_n += int(np.sum(~(d <= rel * np.abs(theirs.astype(np.float64)))))"),
    ("bit-identity always reported", S, "            identical = identical and mine.astype(np.float32).tobytes() == theirs.astype(np.float32).tobytes()",
     "            identical = True"),
    ("largest mean difference not reported", S,
     '        dm = max(dm, float(np.max(np.abs(s["mean32"].astype(np.float64) - rm.astype(np.float64)))))',
     '        dm = max(dm, float(np.min(np.abs(s["mean32"].astype(np.float64) - rm.astype(np.float64)))))'),
    ("largest variance difference not reported", S,
     '        dv = max(dv, float(np.max(np.abs(s["var32"].astype(np.float64) - rv.astype(np.float64)))))',
     '        dv = max(dv, float(np.min(np.abs(s["var32"].astype(np.float64) - rv.astype(np.float64)))))'),
    ("consistency channels not summed over the layers", S,
     '"consistency_channels_outside": sum(x["channels_outside_tolerance"] for x in recal[n]["cons"].values())}',
     '"consistency_channels_outside": max(x["channels_outside_tolerance"] for x in recal[n]["cons"].values())}'),
    ("dev after not compared with PR #27", S, '"dev_pass": dp is not None and dp <= pt and dis <= dt}', '"dev_pass": True}'),
    ("dev after discordance not counted", S, '"dev_pass": dp is not None and dp <= pt and dis <= dt}', '"dev_pass": dp is not None and dp <= pt}'),
    ("statistics outside the recalibration verdict", S, '    c["recalibration_pass"] = bool(c["statistics_pass"] and c["dev_pass"])', '    c["recalibration_pass"] = bool(c["dev_pass"])'),
    ("recalibration tolerances fixed in the code", S, '    rel, abs_ = tol(cfg, "recalibration_statistics_rel"), tol(cfg, "recalibration_statistics_abs")',
     "    rel, abs_ = 1e-4, 1e-6"),
    ("dev after compared with itself", S, '        check = recal_check(cfg, stats, inp.bnrecal[name]["stats"], dev_after["p"][:, 1], inp.bnrecal[name]["dev_after"])',
     '        check = recal_check(cfg, stats, inp.bnrecal[name]["stats"], dev_after["p"][:, 1], dev_after["p"][:, 1])'),
    ("re-estimated on the dev", S, "        stats = BR.recalibrate(copy, fit, bn, batch, budget, name)", "        stats = BR.recalibrate(copy, dev, bn, batch, budget, name)"),
    ("consistency failure ignored", S, '        if bad := {n: c["channels_outside_tolerance"] for n, c in cons.items() if c["channels_outside_tolerance"]}:',
     "        if bad := {}:"),
    ("trained tensor change ignored in phase 2", S, '            fail(f"{name}: " + "; ".join(probs))', "            pass"),
    # confronto (fase 3)
    ("rebuilt copy not checked", S, '            fail(f"{name}: rebuilt copy: " + "; ".join(probs))', "            pass"),
    ("statistics not installed in the rebuilt copy", S, "        layer.moving_mean.assign(mean32)", "        pass"),
    ("test after evaluated on the original", S, '        rec, _ = BR.evaluate(copy, test, batch, f"{name} test after")',
     '        rec, _ = BR.evaluate(orig, test, batch, f"{name} test after")'),
    ("before rows labelled re-estimated", S, '{"p": base[name]["p"], "z": base[name]["z"]}, STATISTICS[0])', '{"p": base[name]["p"], "z": base[name]["z"]}, STATISTICS[1])'),
    ("per-subject before and after swapped", S, "subject_rows(m, scope, STATISTICS[0], xb) + subject_rows(m, scope, STATISTICS[1], xa)",
     "subject_rows(m, scope, STATISTICS[0], xa) + subject_rows(m, scope, STATISTICS[1], xb)"),
    ("metrics before the input re-verification", S,
     '        checks = {**inp.checks, "final": final_checks(cfg, inp, base_dir)}\n        phases["input re-verification"] = time.time() - t\n        t = time.time()\n        if phase == "comparison" and status == "PASS":       # 3b tutto o niente: mai con INCOMPLETE, mai su un sottoinsieme\n            metrics_phase(inp, base, recal, after, run, out)\n',
     '        if phase == "comparison" and status == "PASS":       # 3b tutto o niente: mai con INCOMPLETE, mai su un sottoinsieme\n            metrics_phase(inp, base, recal, after, run, out)\n        checks = {**inp.checks, "final": final_checks(cfg, inp, base_dir)}\n        phases["input re-verification"] = time.time() - t\n        t = time.time()\n'),
    ("difference reversed", S, '        r[f"{k}_difference"] = difference(after[k], before[k])', '        r[f"{k}_difference"] = difference(before[k], after[k])'),
    ("equal value flagged as worse", S, "    return bool(after > before) if sign > 0 else bool(after < before)",
     "    return bool(after >= before) if sign > 0 else bool(after < before)"),
    ("AUROC flag reversed", S, '"auroc": ("auroc_lower", -1)}', '"auroc": ("auroc_lower", 1)}'),
    ("undefined metric flagged as no worsening", S, "    if before is None or after is None:\n        return None", "    if before is None or after is None:\n        return False"),
    ("session scope mapped to session 02", S, 'STRATUM = {"test": "all", "session_03": "03"}', 'STRATUM = {"test": "all", "session_03": "02"}'),
    ("dev worsened flag not reported", S, "name in inp.dev_worsened, scope, mb, ma))", "False, scope, mb, ma))"),
    ("frames of before and after not compared in a row", S,
     '    if (before["n_bona_fide"], before["n_attack"], before["n_subjects"]) != (after["n_bona_fide"], after["n_attack"], after["n_subjects"]):',
     "    if False:"),
    # bootstrap e lettura
    ("bootstrap not paired", S, "            a, b = SR.weighted_metrics(sa, wf), SR.weighted_metrics(sb, wf)",
     "            a, b = SR.weighted_metrics(sa, wf), SR.weighted_metrics(sb, np.ones_like(wf))"),
    ("resample with an undefined metric counted", S,
     "            if all(d is not None for d in diffs[k]):\n                vals[k].append(float(np.mean(diffs[k])))",
     "            if any(d is not None for d in diffs[k]):\n                vals[k].append(float(np.mean([d for d in diffs[k] if d is not None])))"),
    ("minimum number of subjects ignored", S, '    if n < SR.BOOT["min_subjects"]:', "    if False:"),
    ("weights of a fold taken in the wrong order", S, "            wf = w[index[fold]]", "            wf = w[index[fold][::-1]]"),
    ("pooled subjects not in ascending order", S, "    pooled = sorted({s for v in subj.values() for s in v})",
     "    pooled = sorted({s for v in subj.values() for s in v}, reverse=True)"),
    ("subject in two folds accepted", S, "    if len(pooled) != sum(len(v) for v in subj.values()):", "    if False:"),
    ("bona fide subjects miscounted", S, 'bf = {s for fold, sb, _ in models for s, nb in zip(sb["subjects"], sb["nb"]) if nb > 0}',
     'bf = {s for fold, sb, _ in models for s, nb in zip(sb["subjects"], sb["nb"]) if nb >= 0}'),
    ("subjects of two models of a fold not compared", S, "        if subj.setdefault(fold, sb[\"subjects\"]) != sb[\"subjects\"]:", "        if False:"),
    ("reading below with a zero upper end", S, '    if e["high"] < 0:\n        return "below"', '    if e["high"] <= 0:\n        return "below"'),
    ("reading above with a zero lower end", S, '    if e["low"] > 0:\n        return "above"', '    if e["low"] >= 0:\n        return "above"'),
    ("reading without an interval", S, '    if diff is None or e.get("low") is None or e.get("high") is None:\n        return "not_available"',
     '    if diff is None:\n        return "not_available"'),
    ("statement bounds swapped", S, 'difference=f4(md), low=f4(e["low"]), high=f4(e["high"]))', 'difference=f4(md), low=f4(e["high"]), high=f4(e["low"]))'),
    ("mean before from the after column", S, '                    mb = float(np.mean([r[f"{k}_before"] for r in mem])) if full else None',
     '                    mb = float(np.mean([r[f"{k}_after"] for r in mem])) if full else None'),
    ("worsening entry before from the after column", S, '"before": r[f"{k}_before"], "after": r[f"{k}_after"], "difference": r[f"{k}_difference"]})',
     '"before": r[f"{k}_after"], "after": r[f"{k}_after"], "difference": r[f"{k}_difference"]})'),
    ("models_higher counts equal ACER", S, 'r["acer_05_difference"] is not None and r["acer_05_difference"] > 0]}', 'r["acer_05_difference"] is not None and r["acer_05_difference"] >= 0]}'),
    ("figure error bars mirrored", S, "    return [[mean - low], [high - mean]]", "    return [[high - mean], [mean - low]]"),
    ("primary on AUROC", S, 'PRIMARY = {"summary": "ten", "scope": "test", "metric": "acer_05"}', 'PRIMARY = {"summary": "ten", "scope": "test", "metric": "auroc"}'),
    ("primary count on session 03", S, '    ten = [r for r in rows if r["group"] == "ten" and r["scope"] == "test"]\n    d = [r["acer_05_difference"] for r in ten]',
     '    ten = [r for r in rows if r["group"] == "ten" and r["scope"] == "session_03"]\n    d = [r["acer_05_difference"] for r in ten]'),
    ("equal ACER counted as lower", S, '"lower": sum(1 for x in d if x is not None and x < 0)', '"lower": sum(1 for x in d if x is not None and x <= 0)'),
    ("mean over the defined models only", S, "                    full = bool(mem) and len(defined) == len(mem)", "                    full = bool(defined)"),
    ("arms summarised by group", S, '    key = "group" if kind == "group" else "arm"', '    key = "group"'),
    ("worsening not listed", S, "            elif r[f]:\n                worse.append(", "            elif False:\n                worse.append("),
    ("undefined metric not listed", S, "            if r[f] is None:\n                undefined.append(", "            if False:\n                undefined.append("),
    # uscite, cartella, run
    ("public files not scanned", S, '        if isinstance(text, str) and (probs := FR.public_problems(text, inp, pname.endswith(".md"))):', "        if False:"),
    ("figure left out of the public SHA256SUMS", S, "for n, v in sorted(public.items()))", "for n, v in sorted(public.items()) if isinstance(v, str))"),
    ("results written after a baseline failure", S, '        if phase == "comparison" and status == "PASS":\n            doc = ',
     '        if status in ("PASS", "INCOMPLETE", "BASELINE_FAILED", "RECALIBRATION_MISMATCH"):\n            doc = '),
    ("metrics computed after an INCOMPLETE in phase 3a", S, '        if phase == "comparison" and status == "PASS":       # 3b tutto o niente',
     '        if phase == "comparison" and status in ("PASS", "INCOMPLETE"):       # 3b tutto o niente'),
    ("results published after an INCOMPLETE in phase 3a", S, '        if phase == "comparison" and status == "PASS":\n            doc = ',
     '        if phase == "comparison" and status in ("PASS", "INCOMPLETE"):\n            doc = '),
    ("reading applied to INCOMPLETE", S, '        if status == "PASS":\n            summ = summaries(cfg, rows, out["stats"])',
     '        if status in ("PASS", "INCOMPLETE"):\n            summ = summaries(cfg, rows, out["stats"])'),
    ("dirty tree accepted", S, '    if state["dirty"]:', "    if False:"),
    ("existing run folder reused", S, "    if os.path.lexists(run):\n        fail(f\"the run folder", "    if False:\n        fail(f\"the run folder"),
    ("authorisation flag not required", S, "    if not a.i_am_authorized:", "    if False:"),
    ("null note accepted by the run before writing", S,
     '    if cfg["note_sha256"] is None:\n        fail("configuration: note_sha256 is null: the NOTE.md must be frozen and its SHA-256 written before the run")\n    state',
     "    if False:\n        fail(\"configuration: note_sha256 is null: the NOTE.md must be frozen and its SHA-256 written before the run\")\n    state"),
    ("checkpoints not verified at the end", S,
     '    changed = [n for n in inp.order if sha256_file(inp.parent[n]["weights"]) != inp.parent[n]["weights_sha256"]]', "    changed = []"),
    ("BNRECAL SHA256SUMS not verified at the end", S,
     '        if sha256_bytes(read_regular(br / sub / "SHA256SUMS", f"BNRECAL {sub} SHA256SUMS")) != inp.bnrecal_sums[sub]:', "        if False:"),
    ("existing figure overwritten", S, "    if os.path.lexists(out):\n        fail(f\"{out.name} already exists", "    if False:\n        fail(f\"{out.name} already exists"),
    ("figure from a shifted results column", S, '                    v = r[f"{k}_difference"]', '                    v = r[f"{k}_after"]'),
]
# mutazione di controllo: solo un commento; deve SOPRAVVIVERE
CONTROL = [("control: comment only", S, "def difference(a, b):\n", "def difference(a, b):  # control mutant: comment only\n")]
# controlli dichiarati ridondanti, che devono SOPRAVVIVERE:
REDUNDANT = [
    # load_config impone code[ESTIMATOR_FILE] == ESTIMATOR_SHA256 e check_code impone file == code: il secondo confronto
    # del file con la costante non può scattare da solo (resta come difesa esplicita)
    ("declared redundant (must survive): estimator file against the code constant", S,
     "    if got[ESTIMATOR_FILE] != ESTIMATOR_SHA256:", "    if False:"),
    # con frame non allineati stored_test restituisce None e test_prob_pass è già False
    ("declared redundant (must survive): frames in the test verdict", S,
     '    c["test_pass"] = bool(c["test_frames_pass"] and c["test_prob_pass"] and c["test_softmax_pass"])',
     '    c["test_pass"] = bool(c["test_prob_pass"] and c["test_softmax_pass"])'),
    # i frame del test delle fasi 1 e 3 vengono dalle stesse righe dello schema caricate una volta sola
    ("declared redundant (must survive): test keys of phase 3 against phase 1", S,
     '        if sha256_bytes("\\n".join(test["keys"]).encode()) != base[name]["keys_sha256"]:', "        if False:"),
    # prima e dopo sono righe degli stessi frame (stesse chiavi, etichette e soggetti per costruzione)
    ("declared redundant (must survive): subjects of before and after in phase 3", S,
     '            if sb["subjects"] != sa["subjects"] or any(not np.array_equal(sb[k], sa[k]) for k in ("na", "nb")):\n                fail(f"{name} {scope}',
     '            if False:\n                fail(f"{name} {scope}'),
    # set_weights copia ogni tensore: la copia appena fatta è sempre uguale all'originale (come nella PR #27)
    ("declared redundant (must survive): copy equal to the original before the recalibration", S,
     "        if BR.snapshot_weights(copy) != snap:", "        if False:"),
    # un fold senza frame del test è già rifiutato nella fase 0 (comparison.test_frames_per_fold, nessun fold vuoto)
    ("declared redundant (must survive): empty test set in phase 1", S, '        if not test["keys"]:\n            fail(f"{name}: the test set of fold',
     '        if False:\n            fail(f"{name}: the test set of fold'),
]


def pairs(x) -> tuple:
    return x if isinstance(x, tuple) else (x,)


def copy_tree(dst: Path) -> Path:
    code = dst / "code"
    for rel in COPY:
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, code / rel)
    for c in CONFIGS:
        if ROOT.joinpath(*c).is_file():
            code.joinpath(*c[:-1]).mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT.joinpath(*c), code.joinpath(*c))
    return code


def run_tests(code: Path, tmp: Path, failfast: bool = False) -> tuple:
    tmp.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp), "CUDA_VISIBLE_DEVICES": "-1"}
    r = subprocess.run([sys.executable, "-I", "-B", TESTS] + (["-f"] if failfast else []), cwd=code, env=env, capture_output=True,
                       text=True, timeout=3600)
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
    ap = argparse.ArgumentParser(description="Controllo delle mutazioni di nuaa_bn_test.py (offline, sintetico)")
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
    with tempfile.TemporaryDirectory(prefix="bntest-mutations-") as tmp:
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
