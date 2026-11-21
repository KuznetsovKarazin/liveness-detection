"""
Controllo delle mutazioni di scripts/pad_arch_eval.py (la valutazione sul test NUAA dei 48 modelli del confronto
LivenessNet / ResNet-18), offline e deterministico, in due stadi: la forma di scripts/check_pad_arch_run_mutations.py.

Copia il codice (lo script, la suite scripts/check_pad_arch_eval.py, il comando dei run scripts/pad_arch_run.py e la sua
suite, che fornisce il mondo sintetico e produce le cartelle dei run, i moduli importati e le suite dei mondi sintetici:
l'elenco COPY di check_pad_arch_run_mutations più i file di questo comando) in una cartella temporanea, applica una
mutazione dichiarata alla volta (coppia cerca/sostituisci che deve comparire esattamente una volta: se un modello non si
trova più, il controllo si ferma con un errore) ed esegue la suite sulla copia mutata. La configurazione consegnata NON è
copiata: il suo test è saltato nella copia con il motivo dichiarato (SKIP_OK).

Due stadi, per ogni mutazione (suite con -f):
  1. senza TensorFlow: la suite gira in un processo figlio in cui TensorFlow è nascosto (lo stesso finder di
     check_pad_arch_run_mutations.HIDE_TF); i test che richiedono TensorFlow sono saltati con il loro motivo dichiarato.
     Una mutazione con un fallimento o un errore è uccisa qui.
  2. con TensorFlow, solo per le mutazioni sopravvissute allo stadio 1.
Con --without-tensorflow (o in un ambiente senza TensorFlow) lo stadio 2 non si esegue: le mutazioni sopravvissute sono
elencate «not verifiable without TensorFlow», mai contate come uccise, e l'esito è RESULT: INCOMPLETE (codice 2). La copia
non mutata, la mutazione di controllo (solo un commento) e le mutazioni dichiarate ridondanti (REDUNDANT, con il motivo)
girano complete negli stadi eseguiti e devono sopravvivere; una suite che non arriva al riepilogo, o che salta test per
un motivo diverso da quelli della copia non mutata nello stesso stadio, rende la mutazione non valida.

Nessun dato reale: la suite usa solo il mondo sintetico (frame 8×8, modelli Keras minuscoli su CPU in cartelle
temporanee). Le mutazioni girano una alla volta (un solo processo TensorFlow per volta).

  .venv/bin/python -I -B scripts/check_pad_arch_eval_mutations.py                         # tutte, in due stadi
  .venv/bin/python -I -B scripts/check_pad_arch_eval_mutations.py --without-tensorflow    # solo lo stadio 1
  .venv/bin/python -I -B scripts/check_pad_arch_eval_mutations.py --list                  # elenco e verifica dei modelli
"""
import argparse
import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_pad_arch_run_mutations as RM  # noqa: E402

TESTS = "scripts/check_pad_arch_eval.py"
S = "scripts/pad_arch_eval.py"
COPY = tuple(dict.fromkeys(RM.COPY + (S, TESTS, "scripts/check_pad_arch_run.py")))
SKIP_OK = ("missing input: delivery/archeval/pad_arch_eval.json or the files it pins are not present",
           "missing input: TensorFlow is not installed: the tests that build or score a Keras model are skipped")
# (nome, file mutato, testo cercato, testo sostitutivo)
MUTANTS = [
    # configurazione
    ("test text not fixed", S, '    if cfg["test"] != TEST_READ:\n        fail("configuration: test must state exactly', '    if False:\n        fail("configuration: test must state exactly'),
    ("authorisation text not required", S, '    if "published" not in cfg["authorisation"] or "before the test" not in cfg["authorisation"]:', "    if False:"),
    ("runs in another order accepted", S, '    if cfg["runs"] != list(R.RUNS):', "    if False:"),
    ("requested check not quoted", S, '    if "requested by the coordinator" not in pc["requested"]:', "    if False:"),
    ("two places of the check not stated", S, '    if "before any test frame" not in pc["where"] or "primary_analysis" not in pc["where"]:', "    if False:"),
    ("re-installation not stated", S, '    if "re-installed" not in bs["re_estimated"]:', "    if False:"),
    ("reproduction tolerance not stated", S, '    if "1e-5" not in cfg["dev_reproduction"] or "MISMATCH" not in cfg["dev_reproduction"]:', "    if False:"),
    ("pairs not fixed", S, '    if co["pairs"] != _pairs_json():', "    if False:"),
    ("cap unit not fixed", S, '    if cp["unit"] != CAP_UNIT:', "    if False:"),
    ("cap as a boolean accepted", S, '    if isinstance(cp["hours"], bool) or not isinstance(cp["hours"], (int, float))', '    if not isinstance(cp["hours"], (int, float))'),
    ("outputs not fixed", S, '    if ou["public"] != list(PUBLIC_OUTPUTS) or ou["restricted"]', '    if ou["restricted"]'),
    ("choices not declared as the author (primary check)", S, '    R._author(pc["where"], "primary_check.where")', "    pass"),
    # file fissati, tetto, conclusioni, coppie
    ("code pins not compared", S, "    bad = [f for f in CODE_FILES if got[f] != cfg[\"code\"][f]]", "    bad = []"),
    ("code pins not compared with the run configuration", S, '    bad = [f for f in R.CODE_FILES if cfg["code"][f] != pins.rcfg["code"][f]] + \\',
     '    bad = [] + \\'),
    ("pre-registration not anchored to the package", S, '        if sums.get(entry) != cfg["preregistration"][k]["sha256"]:', "        if False:"),
    ("run_plan.csv not regenerated", S, '    if texts["run_plan"] != plan_text(pins.cfg_pr, pins.trial):', "    if False:"),
    ("cap not compared with the plan", S, '    if cfg["cap"]["hours"] != pins.cap["cap_hours"]:', "    if False:"),
    ("cap not rounded up", S, '    cap = math.ceil(round(est * ca["wall_multiplier"] / ca["round_up_hours"], 9)) * ca["round_up_hours"]',
     '    cap = round(est * ca["wall_multiplier"] / ca["round_up_hours"]) * ca["round_up_hours"]'),
    ("cap without the multiplier", S, '    cap = math.ceil(round(est * ca["wall_multiplier"] / ca["round_up_hours"], 9)) * ca["round_up_hours"]',
     '    cap = math.ceil(round(est / ca["round_up_hours"], 9)) * ca["round_up_hours"]'),
    ("conclusion alternative not required once", S, '    if not m or m.group(1).count(CONCLUSION_ALTERNATIVE) != 1:', "    if not m:"),
    ("higher conclusion written as lower", S, '        return conclusions["excludes_zero"].replace(CONCLUSION_ALTERNATIVE, "higher")',
     '        return conclusions["excludes_zero"].replace(CONCLUSION_ALTERNATIVE, "lower")'),
    ("unresolved reading with the first conclusion", S, '    if reading == D.CONTRAST_UNRESOLVED:\n        return conclusions["otherwise"]',
     '    if reading == D.CONTRAST_UNRESOLVED:\n        return conclusions["excludes_zero"]'),
    ("pair contrast reversed", S, "                out.append((a, c))", "                out.append((c, a))"),
    # cartelle dei run
    ("run of the other patience accepted", S, '    if not m or m.group(1) != str(R.RUNS[run]):', "    if not m:"),
    ("run status not required PASS", S, '    if st.get("status") != "PASS" or st.get("passed") is not True or st.get("run_id") != folder.name or st.get("run") != run:',
     '    if st.get("run_id") != folder.name or st.get("run") != run:'),
    ("run of another configuration accepted", S, '    if st.get("config_sha256") != pins.rcfg_sha:', "    if False:"),
    ("run of other code accepted", S, '    if st.get("code_sha256") != pins.run_code:', "    if False:"),
    ("dirty run accepted", S, '    if st.get("dirty") is not False:', "    if False:"),
    ("stopped run accepted", S, '    if st.get("stopped") is not None or st.get("missing_phase_1") != [] or st.get("evaluator_mismatches") != []:', "    if False:"),
    ("incomplete run accepted", S, '    if st.get("complete_phase_1") != names or st.get("complete_phase_2") != names:', "    if False:"),
    ("run without the thread rule accepted", S, '    if (st.get("machine_rule") or {}).get("equal") is not True or (st.get("threads_rule") or {}).get("equal") is not True:',
     "    if False:"),
    ("run with other threads accepted", S, "    if th != want:\n        fail(f\"{what}: effective threads", "    if False:\n        fail(f\"{what}: effective threads"),
    ("run.log not checked", S, '    if sha256_file(FR.regular_path(folder / LOG, f"{folder.name} run.log")) != st.get("run_log_sha256"):', "    if False:"),
    ("SHA256SUMS of the status not compared", S, '        if sha256_bytes(b) != st.get(key):', "        if False:"),
    ("listed files not exactly those of the runner", S, '        if sorted(sums[sub]) != sorted(want_files):', "        if False:"),
    ("files not compared with their SHA256SUMS", S, '            if sha256_file(FR.regular_path(folder / sub / fn, f"{folder.name} {fn}")) != h:', "            if False:"),
    ("summary.json not checked", S, "            or (summ.get(\"inputs_sha256\") or {}).get(\"configuration\") != pins.rcfg_sha:\n        fail(",
     "            or True:\n        pass\n    if False:\n        fail("),
    ("card not checked", S, '                card.get("run"), card.get("position")) != (wsha, p["network"], p["fold"], p["seed"], R.RUNS[run], run, p["position"]):',
     '                card.get("run"), card.get("position")) == ():'),
    ("statistics order not checked", S, "                or list(sd.get(\"layers\") or {}) != order:\n        ", "                or False:\n        "),
    ("statistics agreement not checked", S, '        if (sd.get("evaluator_agreement") or {}).get("pass") is not True:', "        if False:"),
    ("non-float32 statistics accepted", S, "not np.array_equal(a.astype(np.float32).astype(np.float64), a)", "False"),
    ("negative variance accepted", S, "            if mean.shape != var.shape or not np.all(var >= 0):", "            if mean.shape != var.shape:"),
    ("run not re-verified at the end", S, '            if sha256_file(FR.regular_path(folder / sub / fn, fn)) != h:\n                fail(f"run {r[\'run_id\']}: {sub}/{fn} changed',
     '            if False:\n                fail(f"run {r[\'run_id\']}: {sub}/{fn} changed'),
    # revisione indipendente (correzioni 1-8 e mutanti del revisore)
    ("unofficial run accepted by an official evaluation", S, '    if mode == "official" and (st.get("official") is not True or not inside):', "    if False:"),
    ("official run accepted by an unofficial evaluation", S, '    if mode == "unofficial" and st.get("official") is not False:', "    if False:"),
    ("run folder in the official root accepted by an unofficial evaluation", S, '    if mode == "unofficial" and inside:', "    if False:"),
    ("subject sensitivity values in the public outputs", S, '    return {k: lo[k] for k in ("full_mean_delta_acer", "full_sign", "min", "max", "n_sign_differs", "n_not_available", "n_excluded")} | \\',
     '    return {k: lo[k] for k in ("full_mean_delta_acer", "full_sign", "min", "max", "n_sign_differs", "n_not_available", "n_excluded", "values")} | \\'),
    ("denominator ranges from the first exclusion only", S, '        {"n_attack_after_exclusion": [min(a for a, _ in tot), max(a for a, _ in tot)] if tot else None,',
     '        {"n_attack_after_exclusion": [tot[0][0], tot[0][0]] if tot else None,'),
    ("one-class counts not cross-checked with the runs", S, "    check_one_class(oc, runs)\n", ""),
    ("one-class on the ordinary state only (reviewer 2)", S, '"result": D.one_class_run(r["dev"][s]["y"], r["dev"][s]["s"])',
     '"result": D.one_class_run(r["dev"]["ordinary"]["y"], r["dev"]["ordinary"]["s"])'),
    ("degenerate count of another group", S, 'sum(1 for r in rs if BR.group_of(r["card"], bn_cfg) == "ten")', 'sum(1 for r in rs if BR.group_of(r["card"], bn_cfg) == "apart")'),
    ("card fields of the degenerate count not checked", S,
     '        if not FR.is_count(card.get("restored_epoch")) or not all(D.finite_real(card.get(k)) for k in ("val_acc_at_best", "best_val_loss")):',
     "        if False:"),
    ("cap in tenths of an hour (reviewer 5)", S, "    budget = SR.Budget(cap_h * 60)", "    budget = SR.Budget(cap_h * 600)"),
    ("configuration not re-verified (reviewer 6)", S, '    if sha256_bytes(read_regular(cfg_p, "configuration")) != cfg_sha:\n        fail("the configuration changed',
     '    if False:\n        fail("the configuration changed'),
    ("run status.json not re-verified (reviewer 7)", S, '    if sha256_bytes(read_regular(folder / STATUS, "status.json")) != r["status_sha256"]:', "    if False:"),
    ("schema configuration not checked (reviewer 8)", S, "    SR.check_schema_config(pcfg, ctx.base, inp)\n    SR.check_sources(pcfg, inp)\n",
     "    SR.check_sources(pcfg, inp)\n"),
    ("sources of the test rows not checked", S, "    SR.check_schema_config(pcfg, ctx.base, inp)\n    SR.check_sources(pcfg, inp)\n",
     "    SR.check_schema_config(pcfg, ctx.base, inp)\n"),
    ("every fold not required", S, '    if sorted(inp.folds) != list(range(1, pcfg["schema_expected"]["k"] + 1)):', "    if False:"),
    ("string threshold converted at point (b) (reviewer 11)", S, '            pairs.append((n, (nets[n] or {}).get("threshold") if isinstance(nets[n], dict) else None))',
     '            pairs.append((n, float((nets[n] or {}).get("threshold")) if isinstance(nets[n], dict) else None))'),
    ("fixed threshold of the plan from a constant (reviewer 13)", S, '"rule": t, "threshold": r["thresholds"][(s, t)]}',
     '"rule": t, "threshold": r["thresholds"][(s, "dev_eer")] if t == "dev_eer" else 0.5}'),
    ("installed mean and variance lengths not compared (reviewer 15)", S, "            if mean.shape != var.shape or not np.all(var >= 0):", "            if not np.all(var >= 0):"),
    ("conclusions searched without normalising the spaces (reviewer 16)", S, '    m = CONCLUSIONS_RE.search(" ".join(text.split()))', "    m = CONCLUSIONS_RE.search(text)"),
    ("no re-verification before the test", S, '                extra["checks_before_test"] = final_checks(ctx, cfg_p, cfg_sha, runs)     # un FAIL che non ha bisogno del test viene prima',
     '                extra["checks_before_test"] = {}'),
    ("no re-verification after the test", S, "            checks = final_checks(ctx, cfg_p, cfg_sha, runs)\n            scan_log(run, log, inp)",
     "            checks = {}\n            scan_log(run, log, inp)"),
    ("log not scanned before the numbers", S, "            scan_log(run, log, inp)\n", ""),
    ("number of the analysis in the log", S, '            print("   results written to public/ and restricted/ (no number of the analysis in this log); the final scan and status follow", flush=True)',
     '            print(f"   results written; ACER gap ordinary {pa[\'gap_ordinary\'][\'mean_delta\'][\'acer\']!r}", flush=True)'),
    ("analysis not announced after the final status", S, '    if status == "PASS":\n        print(f"analysis computed; report: {run.name}/public/report.md")',
     '    if False:\n        print(f"analysis computed; report: {run.name}/public/report.md")'),
    ("restricted files written and removed reported as never written", S,
     '        extra["restricted"] = ("no restricted result file was written" if not written else',
     '        extra["restricted"] = ("no restricted result file was written" if True else'),
    # seconda revisione indipendente (mutanti del revisore A-P, salvo L, già dichiarato come «run.log not checked»)
    ("one-class cross-check on the first run only (reviewer A)", S, "    for r in runs.values():\n        rows = parse_csv(r[\"one_class_csv\"]",
     "    for r in list(runs.values())[:1]:\n        rows = parse_csv(r[\"one_class_csv\"]"),
    ("official-root identity by string prefix (reviewer B)", S, "    inside = pg.within(folder, OFFICIAL_RUNS_DIR)",
     "    inside = str(folder).startswith(str(OFFICIAL_RUNS_DIR))"),
    ("bona fide range from the first exclusion only (reviewer C)", S,
     '         "n_bona_fide_after_exclusion": [min(b for _, b in tot), max(b for _, b in tot)] if tot else None}',
     '         "n_bona_fide_after_exclusion": [tot[0][1], tot[0][1]] if tot else None}'),
    ("final scan of run.log removed (reviewer D)", S,
     '        if probs := PAC.output_problems((run / LOG).read_text(encoding="utf-8", errors="replace"), inp):\n            fail(f"{LOG} holds text that a publishable output must not hold: " + "; ".join(probs))\n        R.write_status(',
     '        if False:\n            fail(f"{LOG} holds text that a publishable output must not hold: ")\n        R.write_status('),
    ("test_read_started set after the test lists (reviewer E)", S, '                extra["test_read_started"] = True\n                lists = load_test_lists(ctx)\n',
     '                lists = load_test_lists(ctx)\n                extra["test_read_started"] = True\n'),
    ("re-verification after the test lists (reviewer F)", S,
     '                extra["checks_before_test"] = final_checks(ctx, cfg_p, cfg_sha, runs)     # un FAIL che non ha bisogno del test viene prima\n                extra["test_read_started"] = True\n                lists = load_test_lists(ctx)\n',
     '                lists = load_test_lists(ctx)\n                extra["checks_before_test"] = final_checks(ctx, cfg_p, cfg_sha, runs)\n                extra["test_read_started"] = True\n'),
    ("report: re-estimated row shows the ordinary gap (reviewer G)", S, "f\"| ACER gap, re-estimated BN | {fmt(gr['mean_delta']['acer'])}",
     "f\"| ACER gap, re-estimated BN | {fmt(go['mean_delta']['acer'])}"),
    ("report: contrast interval of the ordinary gap (reviewer H)", S, "f\"| contrast | {fmt(c['mean']['acer'])} | {ci(c['interval']['acer'])}",
     "f\"| contrast | {fmt(c['mean']['acer'])} | {ci(go['interval']['acer'])}"),
    ("report: per-seed re-estimated from the ordinary (reviewer I)", S,
     "f\"{fmt(v['gap_re_estimated']['mean_delta']['acer'])} {ci(v['gap_re_estimated']['interval']['acer'])} | \"",
     "f\"{fmt(v['gap_ordinary']['mean_delta']['acer'])} {ci(v['gap_re_estimated']['interval']['acer'])} | \""),
    ("cap check before the re-verification removed (reviewer J)", S, '                budget.check("re-verification before the test")\n', ""),
    ("unofficial evaluation accepts a status without the official field (reviewer K)", S, '    if mode == "unofficial" and st.get("official") is not False:',
     '    if mode == "unofficial" and st.get("official") is True:'),
    ("report: per-seed contrast reading from the overall (reviewer N)", S,
     "f\"{fmt(v['contrast']['mean']['acer'])} {ci(v['contrast']['interval']['acer'])} | {v['contrast']['reading']} |\")",
     "f\"{fmt(v['contrast']['mean']['acer'])} {ci(v['contrast']['interval']['acer'])} | {c['reading']} |\")"),
    ("one-class comparison without the text column (reviewer O)", S,
     "        theirs = sorted(tuple(x[k] for k in R.ONE_CLASS_COLUMNS) for x in rows)\n        mine = sorted(tuple(R.cell(o[k]) for k in R.ONE_CLASS_COLUMNS) for o in oc if o[\"patience\"] == R.RUNS[r[\"run\"]])",
     "        theirs = sorted(tuple(x[k] for k in R.ONE_CLASS_COLUMNS[:4]) for x in rows)\n        mine = sorted(tuple(R.cell(o[k]) for k in R.ONE_CLASS_COLUMNS[:4]) for o in oc if o[\"patience\"] == R.RUNS[r[\"run\"]])"),
    ("re-verification skips the restricted folder (reviewer P)", S, "    for sub in (PUBLIC, RESTRICTED):\n        if FR.read_sums(read_regular(folder / sub / \"SHA256SUMS\"",
     "    for sub in (PUBLIC,):\n        if FR.read_sums(read_regular(folder / sub / \"SHA256SUMS\""),
    ("unofficial evaluation on the inputs of the repository", S, "    if not official and pg.within(base, ROOT) and pg.within(ROOT, base):", "    if False:"),
    ("run.log of a run not re-verified", S, '    if sha256_file(FR.regular_path(folder / LOG, "run.log")) != r["run_log_sha256"]:', "    if False:"),
    # score_model e forward_scores (uccisi dallo stadio con TensorFlow)
    ("softmax tolerance of the scores ignored", S, '    if gap > sg:\n        fail(f"{what}: the softmax of the logits differs from the probabilities',
     '    if False:\n        fail(f"{what}: the softmax of the logits differs from the probabilities'),
    ("bona fide column used as the attack score", S, '    return {"p": p[:, 1].copy(), "z": z, "softmax_gap": gap}', '    return {"p": p[:, 0].copy(), "z": z, "softmax_gap": gap}'),
    ("copy without the weights of the checkpoint", S, "    copy.set_weights(orig.get_weights())\n    if BR.snapshot_weights(copy) != snap:\n        fail(f\"{name}: the in-memory copy differs from the checkpoint",
     "    if BR.snapshot_weights(copy) != snap:\n        fail(f\"{name}: the in-memory copy differs from the checkpoint"),
    ("saved statistics not installed", S, '    stats = BT.install(copy, rec["installed"])',
     '    stats = {n: {"mean32": m32, "var32": v32} for n, (m32, v32) in rec["installed"].items()}'),
    ("installed statistics not checked against the checkpoint", S, "    if probs := BR.tensor_problems(snap, BR.snapshot_weights(copy), BR.moving_keys(copy, order), stats):\n        fail(f\"{name}: re-estimated copy",
     "    if False:\n        fail(f\"{name}: re-estimated copy"),
    # file del dev
    ("dev identifiers not compared", S, '        if [g[D.FRAME_ID_FIELD] for g in got] != [r["key"] for r in rows]:', "        if False:"),
    ("dev subjects and labels not compared", S, '        if any((g["subject_id"], g["session_id"], g["label"], g["file_sha256"]) != (r["subject"], r["session"], r["label"], r["sha256"])',
     '        if any(()'),
    ("dev rows of another state accepted", S, '               (rec["network"], str(rec["fold"]), str(rec["seed"]), state, "dev", rec["weights_sha256"]) for g in got):',
     '               (rec["network"], str(rec["fold"]), str(rec["seed"]), g["bn_state"], "dev", rec["weights_sha256"]) for g in got):'),
    ("saved prediction at 0.5 not checked", S, '        if any(g["prediction_05"] != ("1" if d else "0") for g, d in zip(got, D.attack_decisions(s, FR.FIXED_THRESHOLD))):', "        if False:"),
    ("dev without a class accepted", S, '        if len(set(y.tolist())) != 2:', "        if False:"),
    ("saved EER threshold not checked", S, '        if any(g["threshold_dev_eer"] != repr(thr["dev_eer"]) for g in got) or \\', "        if False and \\"),
    ("EER threshold from the other BN state", S, '            rec["thresholds"][(state, rule)] = thr[rule]', '            rec["thresholds"][("ordinary" if rec["thresholds"] else state, rule)] = thr[rule]'),
    # controllo (a) e (b)
    ("plan check removed", S, "    checked = check_plan(plan, ctx.folds, ctx.seeds)", "    checked = {}"),
    ("completeness per unit, not per network (third review)", S, "        miss = sorted((f, s) for (m, f, s) in want - set(got) if m == n)",
     "        miss = sorted({(f, s) for (_, f, s) in want} - {(f, s) for (_, f, s) in got})"),
    ("missing combinations not reported", S, '        for label, v in (("missing", miss), ("duplicated", dup), ("not in the plan", extra)):',
     '        for label, v in (("duplicated", dup), ("not in the plan", extra)):'),
    ("duplicated combinations not reported", S, '        for label, v in (("missing", miss), ("duplicated", dup), ("not in the plan", extra)):',
     '        for label, v in (("missing", miss), ("not in the plan", extra)):'),
    ("extra combinations not reported", S, '        for label, v in (("missing", miss), ("duplicated", dup), ("not in the plan", extra)):',
     '        for label, v in (("missing", miss), ("duplicated", dup)):'),
    ("networks outside the plan accepted", S, "    if other:\n        out.append(f\"networks outside the plan", "    if False:\n        out.append(f\"networks outside the plan"),
    ("threshold check removed", S, '    if cond[2] != "fixed_0.5":\n        return []', '    if True:\n        return []'),
    ("threshold check for LivenessNet only", S, "    for n in AM.NETWORKS:\n        bad = [t for m, t in pairs", "    for n in AM.NETWORKS[:1]:\n        bad = [t for m, t in pairs"),
    ("threshold check for ResNet-18 only", S, "    for n in AM.NETWORKS:\n        bad = [t for m, t in pairs", "    for n in AM.NETWORKS[1:]:\n        bad = [t for m, t in pairs"),
    ("threshold compared with a tolerance", S, "or not t == REQUIRED_THRESHOLD)]", "or not abs(t - REQUIRED_THRESHOLD) < 1e-6)]"),
    ("primary conditions not checked first", S, "    for cond in list(PRIMARY) + [c for c in D.CONDITIONS if c not in PRIMARY]:", "    for cond in reversed(D.CONDITIONS):"),
    ("unit check before the primary analysis removed", S, "        checked[str(list(cond))] = check_units(cond, conds[cond], ctx.folds, ctx.seeds)\n    primary =",
     "        pass\n    primary ="),
    ("unit check on the secondary conditions removed", S, "            checked[str(list(cond))] = check_units(cond, conds[cond], ctx.folds, ctx.seeds)\n    results =",
     "            pass\n    results ="),
    ("unit threshold read from the plan constant", S, '            pairs.append((n, (nets[n] or {}).get("threshold") if isinstance(nets[n], dict) else None))',
     "            pairs.append((n, REQUIRED_THRESHOLD))"),
    ("unit dataset not checked", S, '        if len(key) != 3 or key[0] != DATASET:', "        if len(key) != 3:"),
    ("primary analysis on other units than those checked", S, "    primary = D.primary_analysis({cond: conds[cond] for cond in PRIMARY})",
     "    primary = D.primary_analysis({cond: build_conditions(plan, recs, meta)[cond] for cond in PRIMARY})"),
    ("unit threshold not from the plan", S, '        net = {"y": m["y"], "s": r["test"][e["bn_state"]]["p"], "subject": m["subject"], "threshold": e["threshold"],',
     '        net = {"y": m["y"], "s": r["test"][e["bn_state"]]["p"], "subject": m["subject"], "threshold": r["thresholds"][(e["bn_state"], "fixed_0.5")],'),
    ("unit scores of the other BN state", S, '        net = {"y": m["y"], "s": r["test"][e["bn_state"]]["p"],', '        net = {"y": m["y"], "s": r["test"]["ordinary"]["p"],'),
    ("duplicated plan entry merged into one unit", S, '        if u is None or r["network"] in u["networks"]:', "        if u is None:"),
    # ordine, riproduzione, test
    ("dev reproduction ignored", S, '            if any(not r["pass"] for r in repro):\n                status = "MISMATCH"', '            if False:\n                status = "MISMATCH"'),
    ("dev agreement in one BN state only", S, "        for state in D.BN_STATES:\n            got = res[state][\"dev\"]", "        for state in D.BN_STATES[:1]:\n            got = res[state][\"dev\"]"),
    ("dev agreement against the wrong threshold", S, '            ag = R.evaluator_agreement(rec["dev"][state]["s"], got["p"], rec["thresholds"][(state, "dev_eer")], ctx.bn_cfg)',
     '            ag = R.evaluator_agreement(rec["dev"][state]["s"], got["p"], None, ctx.bn_cfg)'),
    ("assembled dev not compared with the saved identifiers", S, '        if dev["keys"] != rec["dev"]["ordinary"]["ids"]:', "        if False:"),
    ("test lists loaded before the dev reproduction", S, "        status, stopped, analysis, meta, repro = \"PASS\", None, None, None, []\n",
     "        status, stopped, analysis, meta, repro = \"PASS\", None, None, None, []\n        load_test_lists(ctx)\n"),
    ("test counts not compared with the plan", S, "        if len(rows) != ctx.test_per_fold[i]:", "        if False:"),
    ("test not checked disjoint from the fitting and the dev", S, '        if {r["key"] for r in rows} & {r["key"] for r in other} or {r["subject"] for r in rows} & {r["subject"] for r in other}:',
     "        if False:"),
    ("test assembled per model", S, '        for rec in [r for r in recs if r["fold"] == f]:\n            ctx.budget.check(f"test scoring {rec[\'key\']}")\n            res = (SCORE_MODEL or score_model)(ctx, rec, {"test": test})',
     '        for rec in [r for r in recs if r["fold"] == f]:\n            ctx.budget.check(f"test scoring {rec[\'key\']}")\n            res = (SCORE_MODEL or score_model)(ctx, rec, {"test": FR.assemble(ctx.store, lists[f])})'),
    ("final checks removed", S, "            checks = final_checks(ctx, cfg_p, cfg_sha, runs)\n", "            checks = {}\n"),
    ("no cap check before a dev reproduction", S, '        ctx.budget.check(f"dev reproduction {rec[\'key\']}")', "        pass"),
    ("no cap check before a test assembly", S, '        ctx.budget.check(f"test assembly fold {f}")', "        pass"),
    ("no cap check before a test scoring", S, '            ctx.budget.check(f"test scoring {rec[\'key\']}")', "            pass"),
    ("no cap check before the analysis", S, '                budget.check("analysis")', "                pass"),
    ("INCOMPLETE reported with exit code 0", S, 'STATUS_CODE = {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2, "MISMATCH": 4}',
     'STATUS_CODE = {"PASS": 0, "FAIL": 1, "INCOMPLETE": 0, "MISMATCH": 4}'),
    ("threads not set by the command", S, "        threads = PAC.set_threads(tf, pins.cfg_pr)            # subito dopo l'import, prima di qualunque operazione",
     '        threads = {"requested": dict(pins.trial["threads"]["requested"]), "effective": PAC.read_threads(tf), "set_by_this_runner": True, '
     '"environment": env, "logical_cpus": os.cpu_count()}'),
    ("thread rule not applied", S, '        R.require_threads(extra["threads_rule"])', "        pass"),
    ("machine rule not applied", S, "    mrule = R.machine_rule(pins.trial)\n    runs = load_runs(", '    mrule = {"equal": True}\n    runs = load_runs('),
    ("thread environment not checked", S, "    env = R.thread_environment_rule(pins.cfg_pr, pins.trial)     # variabili dei thread diverse dalla prova: rifiuto",
     '    env = {"TF_NUM_INTRAOP_THREADS": None, "TF_NUM_INTEROP_THREADS": None}'),
    # uscite
    ("restricted results kept after a failure", S, "        left = mw.invalidate_each(rdir, written, None, FR.FAIL_DOC) if rdir.is_dir() and written else []", "        left = []"),
    ("public outputs not scanned", S, "            for n, t in texts.items():\n                if probs := PAC.output_problems(t, inp):", "            for n, t in {}.items():\n                if probs := PAC.output_problems(t, inp):"),
    ("conclusion from the wrong reading", S, '"conclusion": conclusion_for(pins.conclusions, pa["contrast"]["reading"]),', '"conclusion": conclusion_for(pins.conclusions, D.CONTRAST_UNRESOLVED),'),
    ("ties written from the APCER difference", S, '"delta_acer_value": None if r["delta"]["acer"] is None else float(r["delta"]["acer"]), "tie": r["tie"]}',
     '"delta_acer_value": None if r["delta"]["acer"] is None else float(r["delta"]["acer"]), "tie": r["delta"]["apcer"] == 0}'),
    ("test prediction at 0.5 with >=", S, "        d05, deer = D.attack_decisions(p, t05), D.attack_decisions(p, teer)", "        d05, deer = p >= t05, D.attack_decisions(p, teer)"),
    ("public SHA256SUMS empty", S, '            pfiles = {**texts, "SHA256SUMS": FR.sums_text(texts)}', '            pfiles = {**texts, "SHA256SUMS": ""}'),
    ("dirty tree accepted", S, '    if state["dirty"]:\n        fail("the git work tree is not clean: the evaluation', '    if False:\n        fail("the git work tree is not clean: the evaluation'),
    ("attempt rule not applied", S, "    if attempt != need:", "    if False:"),
    ("malformed entries of the output root ignored", S, '            fail(f"output root: the entry {e!r} looks like an evaluation',
     '            continue\n            fail(f"output root: the entry {e!r} looks like an evaluation'),
    ("authorisation flag not required", S, "    if not a.i_am_authorized:\n        fail(\"evaluate reads", "    if False:\n        fail(\"evaluate reads"),
    ("personal paths in the command", S, '    parts = [a.cmd, "--config", R.path_text(R.folder_arg(a.config))]', '    parts = [a.cmd, "--config", str(R.folder_arg(a.config))]'),
    ("report without the primary analysis first", S, '          "## Primary analysis (requested by the coordinator: patience 4, threshold 0.5)", "",',
     '          "## Primary analysis", "",'),
]
# solo un commento: deve sopravvivere
CONTROL = [("comment only (control)", S, "# ============================================================================ uscite\n",
            "# ============================================================================ uscite (mutante di controllo)\n")]
# controlli ridondanti, dichiarati: la mutazione deve sopravvivere perché un altro controllo dice la stessa cosa
REDUNDANT = [
    ("existing evaluation folder (redundant: the folder name is an evaluation run id, so an existing folder is an earlier attempt and the attempt rule refuses the same attempt number first)", S, "    if os.path.lexists(run):\n        return f\"the evaluation folder", "    if False:\n        return f\"the evaluation folder"),
    ("prepared units counted (redundant: the completeness check has already required exactly the 12 combinations, each unit with both networks, and prepare_units refuses duplicated units)", S, '    if sorted(pu["unit"] for pu in prepared) != sorted((DATASET, f, s) for f in folds for s in seeds):', "    if False:"),
    ("boolean threshold (redundant: a boolean equals 0 or 1, never 0.5, so the comparison with 0.5 refuses it)", S, "        bad = [t for m, t in pairs if m == n and (isinstance(t, (bool, np.bool_)) or not isinstance",
     "        bad = [t for m, t in pairs if m == n and (not isinstance"),
    ("same folder for the two runs (redundant: load_run requires the run id of nuaa_patience_4 in the name of one folder and of nuaa_patience_12 in the other, and refuses a symbolic link)",
     S, '    if out["nuaa_patience_4"]["folder"].resolve() == out["nuaa_patience_12"]["folder"].resolve():', "    if False:"),
    ("checkpoint model unchanged after the copy was scored (redundant: the copy is a clone; tensor_problems checks the copy against the snapshot of the checkpoint)",
     S, '    if BR.snapshot_weights(orig) != snap:\n        fail(f"{name}: the checkpoint model changed', '    if False:\n        fail(f"{name}: the checkpoint model changed'),
    ("number of test scores per model (redundant: the scores of the test come from the assembled frames, and D.prepare_units refuses a unit with a different number of identifiers and scores)",
     S, "                if len(rec[\"test\"][s][\"p\"]) != len(test[\"keys\"]):", "                if False:"),
    ("assembled test empty or not in key order (redundant: FR.assemble sorts by key and the counts are compared with the plan before)",
     S, '        if not test["keys"] or test["keys"] != sorted(r["key"] for r in lists[f]):', "        if False:"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Controllo delle mutazioni della valutazione sul test NUAA (offline, sintetico, due stadi)")
    ap.add_argument("--list", action="store_true", help="elenca le mutazioni e verifica i modelli, senza eseguire i test")
    ap.add_argument("--without-tensorflow", action="store_true", help="solo lo stadio senza TensorFlow")
    a = ap.parse_args()
    problems = [f"{n}: the pattern occurs {k} times in {t} (expected exactly once)"
                for n, t, old, _ in MUTANTS + CONTROL + REDUNDANT for o in RM.pairs(old)
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
    # gli strumenti dello strumento dei run, con la copia, la suite e i salti di questo comando
    RM.COPY, RM.TESTS, RM.SKIP_OK = COPY, TESTS, SKIP_OK
    tf_here = importlib.util.find_spec("tensorflow") is not None and not a.without_tensorflow
    killed_notf, killed_tf, survived, invalid, not_verifiable, by_assert, by_error = [], [], [], [], [], [], []
    controls_ok = 0
    with tempfile.TemporaryDirectory(prefix="archeval-mutations-") as tmp:
        tmp = Path(tmp)
        ok0, skips0 = RM.baseline(tmp, False)
        ok1, skips1 = RM.baseline(tmp, True) if tf_here else (True, ())
        if not ok0 or not ok1:
            print("RESULT: FAIL (the tests do not pass on the unmutated copy, or skip for an undeclared reason)")
            return 1
        for i, (n, t, old, new) in enumerate(CONTROL + REDUNDANT):
            v0, line0, _ = RM.one(tmp, 1000 + i, t, old, new, False, skips0)
            v1, line1 = ("not run", "TensorFlow stage not run")
            if tf_here:
                v1, line1, _ = RM.one(tmp, 1000 + i, t, old, new, True, skips1)
            good = v0 == "survived" and (v1 == "survived" if tf_here else True)
            controls_ok += good
            print(f"CONTROL   {'SURVIVED' if good else 'NOT SURVIVED'} {n} (without TensorFlow: {v0}, {line0}; with TensorFlow: {v1}, {line1})", flush=True)
        for i, (n, t, old, new) in enumerate(MUTANTS):
            v, line, rc = RM.one(tmp, i, t, old, new, False, skips0, failfast=True)
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
                print(f"NOT VERIFIABLE without TensorFlow {n} (survives without TensorFlow; the TensorFlow stage is not run)", flush=True)
                continue
            v, line, rc = RM.one(tmp, i, t, old, new, True, skips1, failfast=True)
            if v.startswith("killed"):
                killed_tf.append(n)
                (by_error if v.endswith("error") else by_assert).append(n)
                print(f"KILLED with TensorFlow, {v[7:]}  {n} ({line})", flush=True)
            else:
                {"survived": survived, "invalid": invalid}[v].append(n)
                print(f"{v.upper()} with TensorFlow  {n} ({line})", flush=True)
    print(f"mutations killed without TensorFlow: {len(killed_notf)}/{len(MUTANTS)}")
    print(f"mutations killed only with TensorFlow: {len(killed_tf)}/{len(MUTANTS)}" + ("" if tf_here else " (stage not run)"))
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
    ok = len(killed_notf) + len(killed_tf) == len(MUTANTS) and controls_ok == len(CONTROL) + len(REDUNDANT) and not survived and not invalid
    print("RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
