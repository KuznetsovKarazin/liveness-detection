"""
Controllo delle mutazioni di scripts/pad_arch_definitions.py (definizioni operative), scripts/pad_arch_models.py (registro
delle reti) e scripts/pad_arch_comparison.py (configurazione, protezioni, prova tecnica, piano), offline e deterministico,
con la stessa meccanica di check_nuaa_bn_test_mutations.py.

Copia il codice (i tre script e la suite scripts/check_pad_arch_comparison.py, i moduli importati nuaa_folds_run.py,
nuaa_session_ablation_run.py, nuaa_bn_recalibration.py, nuaa_patience_verify.py, m2_writers.py, path_guard.py,
package_rules.py, le suite check_nuaa_folds_run.py e check_nuaa_session_ablation_run.py che forniscono il mondo
sintetico, e il file di LivenessNet del codice di riferimento src/livedetection/src/architectures.py) in una cartella
temporanea, applica una mutazione dichiarata alla volta (coppia cerca/sostituisci che deve comparire esattamente una
volta: se un modello non si trova più, il controllo si ferma con un errore) ed esegue la suite sulla copia mutata. La
configurazione consegnata NON è copiata: il suo test confronta gli SHA-256 fissati con i file e ucciderebbe ogni mutante
dei file fissati senza provare nulla; nella copia quel test è saltato con il motivo dichiarato (SKIP_OK). Una mutazione è
rilevata solo se la suite arriva al riepilogo («tests run») con almeno un fallimento o errore; una suite che non parte la
rende non valida. La mutazione di controllo (solo un commento) e le mutazioni dichiarate ridondanti devono sopravvivere.

Nessun dato reale: la suite usa solo il mondo sintetico (frame 8×8, modelli Keras addestrati su CPU in una cartella
temporanea). Le mutazioni girano una alla volta (macchina da 16 GB: un solo processo TensorFlow per volta), con la suite
in modalità failfast (-f); la copia non mutata, il mutante di controllo e i ridondanti girano completi.

  .venv/bin/python -I -B scripts/check_pad_arch_comparison_mutations.py            # tutte le mutazioni, una alla volta
  .venv/bin/python -I -B scripts/check_pad_arch_comparison_mutations.py --list     # solo l'elenco, con la verifica dei modelli
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
TESTS = "scripts/check_pad_arch_comparison.py"
D = "scripts/pad_arch_definitions.py"
M = "scripts/pad_arch_models.py"
S = "scripts/pad_arch_comparison.py"
COPY = (D, M, S, TESTS, "scripts/nuaa_folds_run.py", "scripts/nuaa_session_ablation_run.py", "scripts/nuaa_bn_recalibration.py",
        "scripts/nuaa_patience_verify.py", "scripts/m2_writers.py", "scripts/path_guard.py", "scripts/package_rules.py",
        "scripts/check_nuaa_folds_run.py", "scripts/check_nuaa_session_ablation_run.py", "src/livedetection/src/architectures.py")
SKIP_OK = ("missing input: delivery/archcmp/pad_arch_comparison.json or the accepted configurations it pins are not present",)
# (nome, file mutato, testo cercato, testo sostitutivo)
MUTANTS = [
    # (a) run a una sola classe
    ("decision rule with ties as attack", D, "    return s > threshold\n", "    return s >= threshold\n"),
    ("one-class classes swapped in the counts", D, 'c["all_bona_fide" if res["class"] == "bona fide" else "all_attack"] += 1',
     'c["all_attack" if res["class"] == "bona fide" else "all_bona_fide"] += 1'),
    ("boolean threshold accepted", D, "    if isinstance(t, bool) or not isinstance(t, (int, float, np.floating))", "    if not isinstance(t, (int, float, np.floating))"),
    ("absent class counted as evaluable", D, '    if n_att == 0 or n_bf == 0:\n        return {**out, "evaluable": False',
     '    if False:\n        return {**out, "evaluable": False'),
    ("all-attack run not one class", D, 'cls = "bona fide" if k == 0 else "attack" if k == n else None', 'cls = "bona fide" if k == 0 else None'),
    ("not evaluable run in the denominator", D, '            c["not_evaluable"] += 1\n            continue',
     '            c["not_evaluable"] += 1\n            c["evaluable"] += 1\n            continue'),
    ("scores outside [0, 1] accepted", D, "    if not np.all(np.isfinite(s)) or np.any(s < 0) or np.any(s > 1):", "    if not np.all(np.isfinite(s)):"),
    ("labels outside 0/1 accepted", D, "    if len(y) and not np.all(np.isin(y, (0, 1))):", "    if False:"),
    ("threshold outside [0, 1] accepted", D, "not math.isfinite(t) or not 0 <= t <= 1:", "not math.isfinite(t):"),
    ("EER threshold replaced by 0.5", D, '        return float(FR.eer_threshold(y, s)["value"])', "        return float(FR.FIXED_THRESHOLD)"),
    # (b) confronto
    ("accepted attacks counted as rejected", D, 'out["acc"][i] = int(np.sum(m & (y == 1) & ~d))', 'out["acc"][i] = int(np.sum(m & (y == 1) & d))'),
    ("APCER as a float, not an exact fraction", D, "    apcer = Fraction(int(acc), int(na)) if na else None", "    apcer = int(acc) / int(na) if na else None"),
    ("difference reversed", D, "    d = {k: diff(m[MINUEND][k], m[SUBTRAHEND][k]) for k in METRICS}", "    d = {k: diff(m[SUBTRAHEND][k], m[MINUEND][k]) for k in METRICS}"),
    ("bootstrap not paired between the networks", D,
     '    a, b = weighted_rates(pu["counts"][MINUEND], wf), weighted_rates(pu["counts"][SUBTRAHEND], wf)',
     '    a, b = weighted_rates(pu["counts"][MINUEND], wf), weighted_rates(pu["counts"][SUBTRAHEND], np.ones_like(wf))'),
    ("resample with an undefined unit counted", D,
     "            if all(v[k] is not None for v in per):\n                vals[k].append(float(np.mean([float(v[k]) for v in per])))",
     "            if any(v[k] is not None for v in per):\n                vals[k].append(float(np.mean([float(v[k]) for v in per if v[k] is not None])))"),
    ("minimum number of subjects ignored", D, '    if n < SR.BOOT["min_subjects"]:', "    if False:"),
    ("pooled subjects not in ascending order", D, "    pooled = sorted({s for v in subj.values() for s in v})",
     "    pooled = sorted({s for v in subj.values() for s in v}, reverse=True)"),
    ("subject in two splits accepted", D, "    if len(pooled) != sum(len(v) for v in subj.values()):", "    if False:"),
    ("networks on different frames accepted", D, '            fail(f"comparison: unit {key}: the two networks are not scored on the same test frames")', "            pass"),
    ("duplicated unit accepted", D, "        if len(key) != 3 or key in seen:", "        if len(key) != 3:"),
    ("units of a split on different subjects accepted", D,
     '        if split_subjects.setdefault(split, counts[MINUEND]["subjects"]) != counts[MINUEND]["subjects"]:', "        if False:"),
    ("mean over the defined units only", D, "        mean[k] = float(sum(d) / len(d)) if all(x is not None for x in d) else None",
     "        mean[k] = float(np.mean([float(x) for x in d if x is not None])) if any(x is not None for x in d) else None"),
    ("tie counted as ResNet-18 lower", D, '"resnet18_lower": sum(1 for x in da if x is not None and x < 0)', '"resnet18_lower": sum(1 for x in da if x is not None and x <= 0)'),
    # (c) segno
    ("resolved sign with a zero upper bound", D, "    if hi < 0:\n        return SIGN_RESNET_LOWER", "    if hi <= 0:\n        return SIGN_RESNET_LOWER"),
    ("resolved sign with a zero lower bound", D, "    if lo > 0:\n        return SIGN_LIVENESSNET_LOWER", "    if lo >= 0:\n        return SIGN_LIVENESSNET_LOWER"),
    ("missing interval not unresolved", D, "    if lo is None or hi is None:\n        return SIGN_UNRESOLVED", "    if False:\n        return SIGN_UNRESOLVED"),
    ("sign change without both conditions resolved", D, '"sign_change": bool(resolved and sa != sb)', '"sign_change": bool(sa != sb)'),
    ("inversion also for resolved conditions", D, '"point_estimate_inversion": bool(opposite_means and not resolved)', '"point_estimate_inversion": bool(opposite_means)'),
    ("contrast resampled with other indices", D, "        da, db = delta_at(x, wf), delta_at(y, wf)\n        return",
     "        da, db = delta_at(x, wf), delta_at(y, wf[::-1])\n        return"),
    ("unit with a zero difference counted as changing sign", D, 'r["delta_a"] * r["delta_b"] < 0)', 'r["delta_a"] * r["delta_b"] <= 0)'),
    ("conditions on different units accepted", D, '    if [p["unit"] for p in pa] != [p["unit"] for p in pb]:', "    if False:"),
    # (d) sensibilità ai soggetti
    ("absent class after the exclusion counted as zero", D,
     "        return (None if any(v is None for v in vals) else sum(vals) / len(vals)), dens",
     "        return (sum(v or 0 for v in vals) / len(vals)), dens"),
    ("subject not excluded", D, '            keep = np.array([s != excluded for s in pu["counts"][MINUEND]["subjects"]], np.int64)',
     '            keep = np.array([True for s in pu["counts"][MINUEND]["subjects"]], np.int64)'),
    ("every defined exclusion counted as a sign difference", D, '"n_sign_differs": sum(1 for x in values if x["sign_differs"])',
     '"n_sign_differs": sum(1 for x in values if x["sign_differs"] is not None)'),
    ("zero merged with the positive sign", D, "    return (v > 0) - (v < 0)", "    return 1 if v >= 0 else -1"),
    ("minimum of the exclusions from the maximum", D, '"min": min(defined) if defined else None', '"min": max(defined) if defined else None'),
    ("denominators swapped", D, 'dens.append({"unit": pu["unit"], "n_attack": na, "n_bona_fide": nb})',
     'dens.append({"unit": pu["unit"], "n_attack": nb, "n_bona_fide": na})'),
    # registro delle reti
    ("projection shortcut also in stage 1", M, "    return stage > 1 and block == 1\n", "    return block == 1\n"),
    ("convolutions with bias", M, "use_bias=False, kernel_initializer", "use_bias=True, kernel_initializer"),
    ("BN gamma/beta counted once", M, "            trunk += 2 * shape[0]", "            trunk += shape[0]"),
    ("LivenessNet flattened width wrong", M, "    flat = (h // 4) * (w // 4) * 32", "    flat = (h // 4) * (w // 4) * 16"),
    ("BN momentum and epsilon not checked", M,
     "        if isinstance(l, tf.keras.layers.BatchNormalization) and (l.momentum != BN_MOMENTUM or l.epsilon != BN_EPSILON):",
     "        if False:"),
    ("nested model accepted", M, "    if any(isinstance(l, tf.keras.Model) for l in model.layers):", "    if False:"),
    ("counts not compared with the configuration", M, "        if got[k] != arith[k] or got[k] != expected[k]:", "        if got[k] != arith[k]:"),
    ("ResNet-18 without the ReLU after the sum", M, '            x = L.Activation("relu", name=f"{p}_relu2")(x)\n', ""),
    ("convolutions initialised with glorot_uniform", M, 'kernel_initializer="he_normal"', 'kernel_initializer="glorot_uniform"'),
    ("topology not compared", M, "        if topo != want:", "        if False:"),
    ("initialiser not read from the model", M, 'c["use_bias"], c["kernel_initializer"]["class_name"])', 'c["use_bias"], "HeNormal")'),
    # configurazione
    ("unknown configuration key accepted", S, "    if sorted(cfg) != sorted(TOP_KEYS):", "    if False:"),
    ("test statement not compared", S, '    if cfg["test"] != TEST_READ:', "    if False:"),
    ("no-test declaration not checked", S, '    if "no test frame is assembled or evaluated" not in de["no_test"]:', "    if False:"),
    ("parent patience not checked", S, '        if e["early_stopping_patience"] != PARENT_PATIENCE[k] or isinstance(', "        if isinstance("),
    ("parameter counts not compared with the arithmetic", S, "        if any(got[k] != exp[k] or not FR.is_count(got[k]) for k in got):", "        if False:"),
    ("input shape not compared", S, '    if ar["ResNet-18"]["input_shape"] != shape or shape != list(FR.FRAME_SHAPE):', "    if False:"),
    ("trial overrides not fixed", S, '    if tr["trial_overrides"] != {"max_epochs": TRIAL_EPOCHS} or', "    if False or"),
    ("trial unit not fixed", S, '    if {k: t[k] for k in TRIAL_UNIT} != TRIAL_UNIT or any(isinstance(t[k], bool) for k in TRIAL_UNIT):', "    if False:"),
    ("test role accepted in the configuration", S, '    if t["roles"] != list(TRIAL_ROLES):', "    if False:"),
    ("cap above 60 minutes accepted", S, 'not 1 <= t["wall_time_cap_minutes"] <= CAP_LIMIT_MINUTES', 'not 1 <= t["wall_time_cap_minutes"]'),
    ("cap unit not fixed", S, '    if t["cap_unit"] != CAP_UNIT:', "    if False:"),
    ("upper estimate not below the cap", S, '    if not t["estimate_minutes"] <= t["upper_estimate_minutes"] < t["wall_time_cap_minutes"]:',
     '    if not t["estimate_minutes"] <= t["upper_estimate_minutes"]:'),
    ("history tolerance accepted", S, '    if bc["difference"] != 0 or isinstance(bc["difference"], bool):', "    if False:"),
    ("history fields not fixed", S, '    if bc["fields"] != list(HISTORY_KEYS):', "    if False:"),
    ("consequence not checked", S, '    if "not reproduced" not in bc["consequence"] or "pre-registration" not in bc["consequence"]:', "    if False:"),
    ("bootstrap recipe not compared", S, '    if df["bootstrap"] != SR.BOOT:', "    if False:"),
    ("conditions not compared", S, '    if df["conditions"] != [list(c) for c in D.CONDITIONS]:', "    if False:"),
    ("causal wording accepted", S, "    if banned:\n", "    if False:\n"),
    ("existing trainings accepted", S, '        if e["existing"] != 0 or isinstance(e["existing"], bool):', "        if False:"),
    ("new trainings not equal to the units", S, '        if units is not None and e["new"] is not None and e["new"] != units:', "        if False:"),
    ("null value of an available dataset accepted", S, '        if nulls or ds["missing_access"] is not None:', "        if False:"),
    ("other seeds accepted", S, '    if ds["seeds"] is not None and ds["seeds"] != SR.TRAIN_SEEDS:', "    if False:"),
    ("NUAA frames not compared with the accepted configuration", S, 'nuaa["frames_per_unit"] != want_fp \\', "False \\"),
    ("reference runs not compared", S, "    if ref != want_ref:", "    if False:"),
    ("amendment parent not compared", S, 'am.get("parent_config_sha256") != cfg["parent_runs"]["patience_4"]["config_sha256"] or', "False or"),
    ("trial frames not compared with the accepted configuration", S,
     '    if cfg["trial"]["frames"]["fitting"]["n"] != want_fp["fitting"] or cfg["trial"]["frames"]["dev"]["n"] != want_fp["dev"]:', "    if False:"),
    ("LivenessNet architecture not compared with the accepted run", S,
     '    if cfg["architectures"]["LivenessNet"]["path"] != p4["architecture"]["path"] or cfg["architectures"]["LivenessNet"]["sha256"] != p4["architecture"]["sha256"]:',
     "    if False:"),
    # ingressi
    ("code pins not compared", S, '    bad = [f for f in CODE_FILES if got[f] != cfg["code"][f]]', "    bad = []"),
    ("architecture file not hashed", S, '    if h != cfg["architectures"]["LivenessNet"]["sha256"]:', "    if False:"),
    ("accepted run status not required PASS", S, '    if st.get("status") != "PASS" or st.get("run_id") != e["run_id"]', '    if st.get("run_id") != e["run_id"]'),
    ("arm B models not required complete", S, '    if not isinstance(st.get("completed"), list) or any(n not in st["completed"] for n in arm_b):', "    if False:"),
    ("restricted SHA256SUMS not anchored", S, '    if sha256_bytes(b) != e["restricted_sha256sums_sha256"]:', "    if False:"),
    ("card not required in the SHA256SUMS", S, '    if sums.get(fn) != bc["card_sha256"]:', "    if False:"),
    ("card SHA-256 not compared", S, '    if sha256_bytes(b) != bc["card_sha256"]:', "    if False:"),
    ("card without the key fingerprints accepted", S,
     '    if not SR._hex(card.get("fitting_keys_sha256")) or not SR._hex(card.get("dev_keys_sha256")):', "    if False:"),
    ("trial keys not compared with the reference card", S, '    if BR.keys_sha(rows) != card[f"{role}_keys_sha256"]:', "    if False:"),
    ("card of another model accepted", S, '    if (card.get("arm"), card.get("fold"), card.get("train_seed")) != (TRIAL_UNIT["arm"], TRIAL_UNIT["fold"], TRIAL_UNIT["seed"]):',
     "    if False:"),
    ("short card history accepted", S,
     '    if not isinstance(h, dict) or any(not isinstance(h.get(k), list) or len(h[k]) < bc["epochs"] for k in HISTORY_KEYS):',
     "    if not isinstance(h, dict):"),
    ("fold schema accepted", S, '    if inp.folds:\n        fail("the fold schema was loaded', '    if False:\n        fail("the fold schema was loaded'),
    ("assembled test role accepted", S, "    other = sorted(set(inp.assembled) - set(TRIAL_ROLES))", "    other = []"),
    ("trial frame counts not compared", S, '        if got != cfg["trial"]["frames"][role]:', "        if False:"),
    ("environment not compared", S, '        if got != cfg["environment"][k]:', "        if False:"),
    # prova tecnica
    ("authorisation flag not required", S, "    if not a.i_am_authorized:", "    if False:"),
    ("dirty tree accepted", S, '    if state["dirty"]:', "    if False:"),
    ("existing run folder reused", S, "    if os.path.lexists(run):\n        fail(f\"the run folder", "    if False:\n        fail(f\"the run folder"),
    ("run folder anywhere in the repository", S, "[ROOT / DEFAULT_OUT_ROOT], [ROOT / x for x in EXTRA_PROTECTED], inputs,",
     "[ROOT], [ROOT / x for x in EXTRA_PROTECTED], inputs,"),
    ("output root that is a symbolic link accepted", S, '    if out_root.is_symlink():\n        return "the output root', '    if False:\n        return "the output root'),
    ("history compared on one epoch less", S, '    r = PV.shared_history(card["history"], hn, n)', '    r = PV.shared_history(card["history"], hn, n - 1)'),
    ("no cap check between the inference passes", S, '        budget.check(f"inference on the {role} of {net}")\n', ""),
    ("CPU time without the system time", S, "    return t.user + t.system", "    return t.user"),
    ("personal tokens of at least 30 characters", S, "        if v and len(v) >= 3:", "        if v and len(v) >= 30:"),
    ("no cap check before a network", S, '    budget.check(f"train {net}")\n', ""),
    ("cap reported as PASS", S, '            status = "INCOMPLETE"\n', '            status = "PASS"\n'),
    ("history check always reproduced", S, '    return {**base, "outcome": "reproduced" if r["identical"] else "not reproduced",',
     '    return {**base, "outcome": "reproduced",'),
    ("incomplete history checked", S, "    if len(ep) < n:\n", "    if False:\n"),
    ("started missing from the final status", S,
     '        FR.write_status(run, {**head, "status": status, "passed": status == "PASS", "started": started, "finished": now(),',
     '        FR.write_status(run, {**head, "status": status, "passed": status == "PASS", "finished": now(),'),
    ("started missing from the FAIL status", S,
     '            FR.write_status(run, {**head, "status": "FAIL", "passed": False, "started": started, "finished": now(),',
     '            FR.write_status(run, {**head, "status": "FAIL", "passed": False, "finished": now(),'),
    ("CPU time missing from the final status", S,
     '"process_cpu_seconds": round(cpu_seconds() - c0, 1),\n                              "cap_minutes": cap, "cap_unit": CAP_UNIT, "stopped": stopped,',
     '\n                              "cap_minutes": cap, "cap_unit": CAP_UNIT, "stopped": stopped,'),
    ("outputs kept after a FAIL", S, "        left = mw.invalidate_each(run, list(TRIAL_OUTPUTS), None, FR.FAIL_DOC)", "        left = []"),
    ("trial outputs not scanned", S, '            if probs := output_problems(text, inp):', "            if probs := []:"),
    ("run.log not scanned", S, '        if probs := output_problems((run / LOG).read_text(encoding="utf-8", errors="replace"), inp):', "        if probs := []:"),
    ("home folder not a personal text", S, "    out = [str(Path.home()), str(ROOT)]", "    out = [str(ROOT)]"),
    ("project rules not applied to the outputs", S, "    bad = FR.public_problems(text, inp)", "    bad = []"),
    ("final configuration check dropped", S, '    if sha256_bytes(read_regular(cfg_p, "configuration")) != cfg_sha:', "    if False:"),
    ("final card check dropped", S,
     '    if sha256_bytes(read_regular(BR.parent_path(run, bc["model"], "card.json"), "reference card")) != bc["card_sha256"]:', "    if False:"),
    ("trainer copy with an undeclared change", S, '    ("    return card\\n", "    return card, model\\n"),\n)', "\n)"),
    # tempi e piano
    ("inference rate from the fitting pass only", S,
     'out["inference_seconds_per_1000_frames"] = sum(inf[r]["seconds"] for r in TRIAL_ROLES) / sum(inf[r]["frames"] for r in TRIAL_ROLES) * 1000',
     'out["inference_seconds_per_1000_frames"] = inf["fitting"]["seconds"] / inf["fitting"]["frames"] * 1000'),
    ("training cost outside the epochs left out of the plan", S, 'e["new"] * (ep_min * per_epoch + ov)', 'e["new"] * ep_min * per_epoch'),
    ("trial of another run id accepted", S, '    if doc.get("schema") != TRIAL_SCHEMA or doc.get("run_id") != st["run_id"] or', '    if doc.get("schema") != TRIAL_SCHEMA or'),
    ("trial of another configuration accepted", S, '    if doc.get("config_sha256") != cfg_sha:', "    if False:"),
    ("plan without the check of the pinned code", S, "    check_code(cfg)                                    # i file fissati di ora contro la configurazione, come in trial\n", ""),
    ("training cost outside the epochs counting the epochs twice", S,
     '    rec["training_overhead"] = {"seconds": time.monotonic() - t0 - sum(e["seconds_total"] for e in rec["epochs"]),',
     '    rec["training_overhead"] = {"seconds": time.monotonic() - t0,'),
    ("assembled keys not compared with the reference card", S, 'keys_check(inp.reference_card, r, part["rows"])', "True"),
    ("trial of other code accepted", S, '    if not isinstance(code, dict) or any(code.get(f) != h for f, h in now_code.items()):', "    if False:"),
    ("validation included in the training rate", S, '(e["seconds_total"] - e["seconds_validation"]) / n_fit * 1000', 'e["seconds_total"] / n_fit * 1000'),
    ("existing plan files overwritten", S, "    if existing:\n        fail(", "    if False:\n        fail("),
    ("plan folder anywhere in the repository", S, "[ROOT / DEFAULT_PLAN_DIR]", "[ROOT]"),
    ("minimum epochs without the best epoch", S, 'ep_min, ep_max = min(p + 1, pl["max_epochs"]), pl["max_epochs"]', 'ep_min, ep_max = min(p, pl["max_epochs"]), pl["max_epochs"]'),
    ("re-estimation without the consistency pass", S, '(nbn + 1) * fp["fitting"]', 'nbn * fp["fitting"]'),
    ("test scored in one BN state only", S, '2 * fp["dev"] + 2 * t) / 1000', '2 * fp["dev"] + t) / 1000'),
    ("CPU hours without the measured ratio", S, '"cpu_hours_max": None if None in (w_max, ratio) else hours(w_max * ratio)}',
     '"cpu_hours_max": None if None in (w_max, ratio) else hours(w_max)}'),
    ("trial.json not anchored by its status", S, '    if sha256_bytes(b) != st.get("trial_json_sha256"):', "    if False:"),
    ("FAIL trial accepted by the plan", S, '    if st.get("status") not in ("PASS", "INCOMPLETE")', '    if st.get("status") not in ("PASS", "INCOMPLETE", "FAIL")'),
    ("trial of another unit accepted", S, '    if doc.get("unit") != {**TRIAL_UNIT, "frames": cfg["trial"]["frames"]} or', "    if False or"),
    ("value not fixed shown as an empty cell", S, "        if k in CONFIG_VALUES:\n            return TBF", "        if False:\n            return TBF"),
    ("totals over rows with missing numbers", S, "    full = [r for r in rows if all(r[k] is not None for k in HOUR_KEYS)]",
     "    full = [r for r in rows if any(r[k] is not None for k in HOUR_KEYS)]"),
]
# mutazione di controllo: solo un commento; deve SOPRAVVIVERE
CONTROL = [("control: comment only", D, "def diff(a, b):\n", "def diff(a, b):  # control mutant: comment only\n")]
# controlli dichiarati ridondanti, che devono SOPRAVVIVERE:
REDUNDANT = [
    # SR.load_config rifiuta già una configurazione emendata il cui training non è la ricetta con la sola pazienza 12
    # (amended_training), e la ricetta di patience 4 è controllata qui contro FROZEN_TRAINING
    ("declared redundant (must survive): patience-12 training against patience 4", S,
     '    if p12["training"] != {**p4["training"], "early_stopping_patience": 12}:', "    if False:"),
    # la storia registrata dalla callback e quella di Keras vengono dagli stessi log a fine epoca: il controllo resta come
    # difesa contro una callback che li alterasse
    ("declared redundant (must survive): timer history against the Keras history", S,
     "    if [{k: e[k] for k in HISTORY_KEYS} for e in rec[\"epochs\"]] != [", "    if False and [{k: e[k] for k in HISTORY_KEYS} for e in rec[\"epochs\"]] != ["),
    # assert_no_test è chiamata anche prima di ogni assemblaggio (trial_frames) e alla fine di load_inputs: la chiamata
    # dopo le due reti non può scattare da sola
    ("declared redundant (must survive): no-test check after the networks", S,
     "        assert_no_test(inp)\n        checks = final_checks(", "        checks = final_checks("),
]


def pairs(x) -> tuple:
    return x if isinstance(x, tuple) else (x,)


def copy_tree(dst: Path) -> Path:
    code = dst / "code"
    for rel in COPY:
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, code / rel)
    return code


def run_tests(code: Path, tmp: Path, failfast: bool = False) -> tuple:
    tmp.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp), "CUDA_VISIBLE_DEVICES": "-1", "TF_CPP_MIN_LOG_LEVEL": "2"}
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
    ap = argparse.ArgumentParser(description="Controllo delle mutazioni del confronto tra architetture (offline, sintetico)")
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
    with tempfile.TemporaryDirectory(prefix="archcmp-mutations-") as tmp:
        tmp = Path(tmp)
        rc, line, base_skips = run_tests(copy_tree(tmp / "base"), tmp / "base" / "tmp")
        print(f"unmutated copy: {TESTS} exit {rc} ({line})", flush=True)
        for s in base_skips:
            print(f"unmutated copy: {s}", flush=True)
        if rc or not re.fullmatch(r"tests run: \d+, failures: 0, errors: 0, skipped: \d+", line):
            print("RESULT: FAIL (the tests do not pass on the unmutated copy)")
            return 1
        if any(s.split(": ", 1)[-1] not in SKIP_OK for s in base_skips):
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
