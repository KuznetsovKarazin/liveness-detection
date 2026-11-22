"""
Controllo delle mutazioni di scripts/pad_anet_definitions.py (definizioni operative C5), scripts/pad_anet_models.py
(registro delle tre AttackNet) e scripts/pad_anet_comparison.py (configurazione, ingressi, piano, prova tecnica),
offline e deterministico, con la meccanica di check_pad_arch_comparison_mutations.py.

Copia il codice (i tre moduli, la suite scripts/check_pad_anet_comparison.py, i moduli C2 importati, le suite C2 che
forniscono il mondo sintetico, delivery/anetcmp/make_config.py, la configurazione della PR #27 copiata nel mondo,
src/livedetection/src/architectures.py e lo strumento tools/run_without_tensorflow.py del pacchetto) in una cartella
temporanea, applica una mutazione dichiarata alla volta (coppia cerca/sostituisci che deve comparire esattamente una
volta: se un modello non si trova più, il controllo si ferma con un errore) ed esegue la suite sulla copia mutata. La
configurazione consegnata NON è copiata: il suo test è saltato con il motivo dichiarato (SKIP_OK). Una mutazione è
rilevata solo se la suite arriva al riepilogo («tests run») con almeno un fallimento o errore; una suite che non parte
la rende non valida. La mutazione di controllo (solo un commento) e le mutazioni dichiarate ridondanti devono
sopravvivere.

Ogni mutazione ha un modo: «offline» (la suite gira con TensorFlow nascosto, tools/run_without_tensorflow.py: i test che
addestrano sono saltati, la mutazione deve essere rilevata dai test senza TensorFlow) oppure «tf» (la suite completa, con
la prova vera su CPU nel mondo 8×8). Le copie non mutate girano nei due modi; ogni mutante è confrontato con la copia non
mutata dello stesso modo (stessi salti).

  .venv/bin/python -I -B scripts/check_pad_anet_comparison_mutations.py                 # tutte, una alla volta
  .venv/bin/python -I -B scripts/check_pad_anet_comparison_mutations.py --list          # elenco e verifica dei modelli
  .venv/bin/python -I -B scripts/check_pad_anet_comparison_mutations.py --mode offline  # solo le mutazioni offline
  .venv/bin/python -I -B scripts/check_pad_anet_comparison_mutations.py --only 3,7      # solo le mutazioni con questi numeri (da 1)
  .venv/bin/python -I -B scripts/check_pad_anet_comparison_mutations.py --only c        # solo la mutazione di controllo e le ridondanti
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
TESTS = "scripts/check_pad_anet_comparison.py"
TOOL = "tools/run_without_tensorflow.py"
TOOL_SOURCES = ("tools/run_without_tensorflow.py", "delivery/pad-attacknet-comparison/livenesslab/tools/run_without_tensorflow.py",
                "delivery/pad-arch-comparison/livenesslab/tools/run_without_tensorflow.py")
E = "scripts/pad_anet_definitions.py"
M = "scripts/pad_anet_models.py"
S = "scripts/pad_anet_comparison.py"
COPY = (E, M, S, TESTS, "scripts/pad_arch_comparison.py", "scripts/pad_arch_definitions.py", "scripts/pad_arch_models.py", "scripts/pad_arch_run.py",
        "scripts/nuaa_folds_run.py", "scripts/nuaa_session_ablation_run.py", "scripts/nuaa_bn_recalibration.py", "scripts/nuaa_bn_test.py",
        "scripts/nuaa_patience_verify.py", "scripts/m2_writers.py", "scripts/path_guard.py", "scripts/package_rules.py",
        "scripts/check_nuaa_folds_run.py", "scripts/check_nuaa_session_ablation_run.py", "scripts/check_pad_arch_comparison.py",
        "delivery/anetcmp/make_config.py", "delivery/bnrecal/nuaa_bn_recalibration.json", "src/livedetection/src/architectures.py")
SKIP_OK = ("missing input: delivery/anetcmp/pad_anet_comparison.json or the files it pins are not present",
           "missing input: TensorFlow is not installed: the tests that build or train a Keras model are skipped")
OFF, TF = "offline", "tf"
# (nome, modo, file mutato, testo cercato, testo sostitutivo)
MUTANTS = [
    # registro
    ("Add merge counted as a concatenation", OFF, M, '        merged = 2 * f if merge == "Concatenate" else f\n', "        merged = 2 * f\n"),
    ("AttackNetV2_1 conv dropout at 0.25", OFF, M, '"AttackNetV2_1": 1.0, "AttackNetV2_2": 0.5}', '"AttackNetV2_1": 0.5, "AttackNetV2_2": 0.5}'),
    ("BN gamma/beta counted once", OFF, M, "shape, 2 * ch, 2 * ch)", "shape, ch, 2 * ch)"),
    ("dense BN with spatial values", OFF, M, '"spatial": int(np.prod(s[:-1])) if len(s) > 1 else 1,', '"spatial": int(np.prod(s[:-1])) if len(s) > 1 else 128,'),
    ("recalibration count without the frames", OFF, M, '    return {b["name"]: int(frames) * b["spatial"]', '    return {b["name"]: b["spatial"]'),
    ("concatenation inputs swapped in the specification", OFF, M, '(f"conv{p}_1", f"leaky_relu{p}_3"), shape=', '(f"leaky_relu{p}_3", f"conv{p}_1"), shape='),
    ("topology not compared", TF, M, "    if topo != want:\n", "    if False:\n"),
    ("dropout, alpha and regularizers not compared", TF, M, "    if ex != want_ex:\n", "    if False:\n"),
    # definizioni: regola (n)
    ("last minimum restored on ties", OFF, E, "float(v) < float(values[best])", "float(v) <= float(values[best])"),
    ("infinite val_loss restorable", OFF, E, "        if D.finite_real(v) and (best is None", "        if not math.isnan(v) and (best is None"),
    ("no finite val_loss evaluable", OFF, E, '"evaluable": best is not None, "reason"', '"evaluable": True, "reason"'),
    ("non-finite restored weights accepted", OFF, E, "    if not weights_finite:\n", "    if False:\n"),
    ("non-finite statistics accepted", OFF, E, "    if statistics_finite is not True:\n", "    if False:\n"),
    ("re-estimated dev state not checked", OFF, E, "    for s in BN_STATES:\n        st = states.get(s)", "    for s in BN_STATES[:1]:\n        st = states.get(s)"),
    # definizioni: frame e unità
    ("frame identifiers between networks not compared", OFF, E, "            if ids[n] != ids[present[0]]:\n", "            if False:\n"),
    ("networks on other frames accepted", OFF, E, '                fail(f"comparison: unit {key}: {present[0]} and {n} are not scored on the same test frames")', "                pass"),
    ("duplicated unit accepted", OFF, E, "        if len(key) != 3 or key in seen:", "        if len(key) != 3:"),
    ("seed outside the plan accepted", OFF, E, "or int(key[2]) not in PLAN_SEEDS:", "or False:"),
    ("required network not required", OFF, E, "        missing = [n for n in required if n not in nets]", "        missing = []"),
    ("units of a fold on other frames accepted", OFF, E, "        if split_frames.setdefault(split, frames) != frames:", "        if False:"),
    ("identifier in two folds accepted", OFF, E, "                if id_split.setdefault(i, split) != split:", "                if False:"),
    ("BN states on other frames accepted", OFF, E, '        D.same_frames(x, y, "the two BN states")', "        pass"),
    ("subject in two folds accepted", OFF, E, "    if len(allsub) != sum(len(v) for v in subj.values()):", "    if False:"),
    # definizioni: quantità
    ("contrast with the LivenessNet term of the wrong sign", OFF, E, '(-1, k, "ordinary"), (-1, REFERENCE, "re-estimated"), (1, REFERENCE, "ordinary"))',
     '(-1, k, "ordinary"), (-1, REFERENCE, "re-estimated"), (-1, REFERENCE, "ordinary"))'),
    ("variation reversed", OFF, E, '    return ((1, net, "re-estimated"), (-1, net, "ordinary"))', '    return ((-1, net, "re-estimated"), (1, net, "ordinary"))'),
    ("gap with ResNet-18 added", OFF, E, "    return ((1, net, state), (-1, EXTERNAL, state))", "    return ((1, net, state), (1, EXTERNAL, state))"),
    ("unit values as floats", OFF, E, "sum((Fraction(int(c)) * v for (c, _, _), v in zip(terms, vals)), Fraction(0))",
     "sum((float(c) * float(v) for (c, _, _), v in zip(terms, vals)), 0.0)"),
    ("bootstrap not paired across the units", OFF, E, '                per = [term_value(e, terms, w[index[e["ordinary"]["split"]]], cache) for e in entries]',
     '                per = [term_value(e, terms, np.ones_like(w[index[e["ordinary"]["split"]]]), cache) for e in entries]'),
    ("resample with an undefined unit counted", OFF, E, "                    if all(v[k] is not None for v in per):", "                    if any(v[k] is not None for v in per):"),
    ("pooled subjects not in ascending order", OFF, E, "    allsub = sorted({s for v in subj.values() for s in v})", "    allsub = sorted({s for v in subj.values() for s in v}, reverse=True)"),
    # definizioni: lettura, guardiano, descrittivi
    ("contrast resolved with a zero upper bound", OFF, E, "    if hi < 0:\n        return names[0]", "    if hi <= 0:\n        return names[0]"),
    ("contrast resolved with a zero lower bound", OFF, E, "    if lo > 0:\n        return names[1]", "    if lo >= 0:\n        return names[1]"),
    ("conclusion sides swapped", OFF, E, '        side = "lower" if reading == names[0] else "higher"', '        side = "higher" if reading == names[0] else "lower"'),
    ("threshold other than 0.5 accepted", OFF, E, "                if threshold_rule == PRIMARY_THRESHOLD and t != float(FR.FIXED_THRESHOLD):", "                if False:"),
    ("reused network incomplete accepted", OFF, E, "                if have[st] != want:", "                if False:"),
    ("contrast on 11 pairs", OFF, E, "        computed = pairs == want", "        computed = len(pairs) >= 11"),
    ("per-seed results on every unit", OFF, E, '        sub = [e for e in entries if e["ordinary"]["seed"] == seed]', "        sub = entries"),
    ("ties ranked apart", OFF, E, "        return {n: 1 + sum(1 for m in vals if vals[m] < vals[n]) for n in vals}",
     "        return {n: 1 + sum(1 for m in vals if vals[m] <= vals[n] and m != n) for n in vals}"),
    ("attack and bona fide denominators swapped", OFF, E, '        tot["attack"] += na\n        tot["bona_fide"] += nb', '        tot["attack"] += nb\n        tot["bona_fide"] += na'),
    ("percentage points with one decimal", OFF, E, '(d * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)', '(d * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)'),
    ("not evaluable run in the one-class denominator", OFF, E, '            c["not_evaluable"] += 1\n            continue',
     '            c["not_evaluable"] += 1\n            c["evaluable"] += 1\n            continue'),
    # configurazione
    ("unknown configuration key accepted", OFF, S, "    if sorted(cfg) != sorted(TOP_KEYS):", "    if False:"),
    ("declaration phrases not checked", OFF, S, "        if phrase not in de[k]:", "        if False:"),
    ("pinned file SHA-256 not compared", OFF, S, '    if sha256_bytes(b) != e["sha256"]:\n        fail(f"{what} {p.name}', '    if False:\n        fail(f"{what} {p.name}'),
    ("architecture file other than C2 accepted", OFF, S,
     '    if ar["path"] != cfg_pr["architectures"]["LivenessNet"]["path"] or ar["sha256"] != cfg_pr["architectures"]["LivenessNet"]["sha256"]:', "    if False:"),
    ("counts not compared with the registry", OFF, S, "        if any(got[k] != exp[k] or not FR.is_count(got[k]) for k in got):", "        if False:"),
    ("BN names not compared with the registry", OFF, S, '        if e["batchnorm_names"] != AN.bn_order(n):', "        if False:"),
    ("constructor not fixed", OFF, S, '        if e["constructor"] != AN.CONSTRUCTOR:', "        if False:"),
    ("threads other than C2 accepted", OFF, S, '    if tr["threads"] != cfg_pr["training"]["threads"]:', "    if False:"),
    ("trial unit not fixed", OFF, S, "    if {k: t[k] for k in TRIAL_UNIT} != TRIAL_UNIT or any(isinstance(t[k], bool) for k in TRIAL_UNIT):", "    if False:"),
    ("test role accepted", OFF, S, '    if t["roles"] != list(TRIAL_ROLES):', "    if False:"),
    ("cap above 60 minutes accepted", OFF, S, 'not 1 <= t["wall_time_cap_minutes"] <= CAP_LIMIT_MINUTES', 'not 1 <= t["wall_time_cap_minutes"]'),
    ("checkpoints inside the repository accepted", OFF, S, '    if "outside the repository" not in t["checkpoints"]:', "    if False:"),
    ("runs in another order accepted", OFF, S, '    if pl["runs"] != run_names():', "    if False:"),
    ("reference cells not fixed", OFF, S, '    _int_eq(pl["reference"]["cells"], REFERENCE_CELLS, "plan.reference.cells")', "    pass"),
    ("caps multipliers below 1 accepted", OFF, S, '"plan.caps.wall_multiplier") < 1 or _number(ca["anomaly_factor"], "plan.caps.anomaly_factor") < 1:',
     '"plan.caps.wall_multiplier") < 0 or _number(ca["anomaly_factor"], "plan.caps.anomaly_factor") < 0:'),
    ("individual intervals not fixed", OFF, S, '    if pr["intervals"] != DEF.INTERVALS_NOTE:', "    if False:"),
    ("causal wording accepted", OFF, S, "    if banned:\n", "    if False:\n"),
    ("choice not declared accepted", OFF, S, '    if "a choice of the author" not in v:', "    if False:"),
    # ingressi
    ("C2 trial SHA-256 not compared", OFF, S, '    if files[STATUS] != ct["status_sha256"] or files[PAC.TRIAL_JSON] != ct["trial_json_sha256"]:', "    if False:"),
    ("C2 trial of another configuration accepted", OFF, S, '    if doc["config_sha256"] != cfg_sha_pr:', "    if False:"),
    ("C2 evaluation not required PASS and official", OFF, S,
     '    if status.get("status") != "PASS" or status.get("run_id") != cr["evaluation_run_id"] or status.get("official") is not True:', "    if False:"),
    ("48 models not required in results.json", OFF, S, '    if not isinstance(models, list) or len(models) != REFERENCE_CELLS:', "    if not isinstance(models, list):"),
    ("restricted anchor not compared", OFF, S, '    if status.get("restricted_sha256sums_sha256") != cr["restricted_sha256sums_sha256"]:', "    if False:"),
    # piano
    ("matrix row outside the plan accepted", OFF, S, '                or r["reference"] != "none" or r["run"] != run_name(r["network"], r["patience"]):', "                or False:"),
    ("reference cell of the other patience accepted", OFF, S, 'or f"-ARCHRUN-{tag}-" not in r["run_id"]:', "or False:"),
    ("check-plan without the byte comparison", OFF, S, '        if got != text.encode("utf-8"):', "        if False:"),
    ("cap without the anomaly factor", OFF, S, '    v = (wall_max_hours + (ca["anomaly_factor"] - 1) * per_training_max_hours) * ca["wall_multiplier"]',
     '    v = wall_max_hours * ca["wall_multiplier"]'),
    ("trial of another configuration used by plan", OFF, S, '    if doc.get("config_sha256") != cfg_sha:', "    if False:"),
    ("trial of other code used by plan", OFF, S, '    if doc.get("code_sha256") != {"architecture": cfg["architectures"]["sha256"], **code}:', "    if False:"),
    # prova tecnica
    ("hooks accepted with the delivered configuration", OFF, S, "    if official and (setv or changed):", "    if False:"),
    ("dirty tree accepted", OFF, S, '    if state["dirty"]:', "    if False:"),
    ("checkpoint root inside the repository accepted", OFF, S, "    if pg.within(root, ROOT) or pg.within(ROOT, root):", "    if False:"),
    ("control reproduced without the C2 trial", TF, S, 'rec["outcome"] = "reproduced" if accepted["outcome"] == "reproduced" and c2_same and',
     'rec["outcome"] = "reproduced" if accepted["outcome"] == "reproduced" and'),
    ("not evaluable model not recorded", TF, S, '        if not out["evaluable"]:\n            raise NotEvaluable(out["reason"])', '        if False:\n            raise NotEvaluable(out["reason"])'),
    ("scorer and evaluator disagreement accepted", TF, S,
     '        if not rec["reload"]["agreement_in_memory"]["pass"] or not rec["scorer_agreement"]["pass"]:', "        if False:"),
    ("installed copy not compared", TF, S, '        if probs or not same or not rec["installed_copy"]["agreement"]["pass"]:', "        if False:"),
    ("BN check result not enforced", TF, S, "    if not passed:\n        fail(f\"{net}: BN check failed", "    if False:\n        fail(f\"{net}: BN check failed"),
    # aggiunte dopo la revisione indipendente del codice
    ("SR hooks missing from the registry", OFF, S, '"nuaa_session_ablation_run": ("REPO_STATE", "TRAINER", "MAKE_SCORER", "MAKE_LOGITS", "CLOCK", "ARCH_LOADER"),',
     '"nuaa_session_ablation_run": ("TRAINER", "MAKE_SCORER", "MAKE_LOGITS", "CLOCK", "ARCH_LOADER"),'),
    ("test values changed accepted", OFF, S, "    if official and (setv or changed):", "    if official and setv:"),
    ("official configuration by path string", OFF, S, "        return os.path.samefile(cfg_p, ROOT / DEFAULT_CONFIG)", "        return cfg_p == ROOT / DEFAULT_CONFIG"),
    ("attempt number not compared", OFF, S, "    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt != len(prev) + 1:", "    if False:"),
    ("run id look-alike accepted", OFF, S, "                    if not RUN_ID_RE.match(e):\n                        fail(", "                    if False:\n                        fail("),
    ("machine comparison always equal", OFF, S, '    rec["equal"] = not rec["differing"]', '    rec["equal"] = True'),
    ("control without the machine", TF, S, 'and c2_same and rec["machine_and_versions"]["equal"] else', "and c2_same else"),
    ("run.log not checked before the RESULT line", TF, S,
     '        if probs := PAC.output_problems((run / LOG).read_text(encoding="utf-8", errors="replace"), inp):', "        if False:"),
    ("non-finite fitting scores as FAIL", TF, S,
     '                    raise NotEvaluable(f"{DEF.NOT_EVALUABLE}: probabilities non-finite, outside [0, 1] or of the wrong shape on the {role} "', "                    raise Failure("),
    ("non-finite dev scores not checked", TF, S, '    if not DEF.arrays_finite([f["p"], f["z"]]):\n        raise NotEvaluable', "    if False:\n        raise NotEvaluable"),
    ("non-finite statistics as FAIL", TF, S, '            if str(e).endswith("the re-estimated mean or variance is not finite"):', "            if False:"),
    ("equal dev scores in the two states accepted", TF, S, '    c5 = all(v["statistics_finite"] for v in layers.values()) and dev_diff > 0',
     '    c5 = all(v["statistics_finite"] for v in layers.values()) and dev_diff >= 0'),
    ("epochs at least the patience", OFF, S, '            ep_min, ep_max = min(p + 1, pl["max_epochs"]), pl["max_epochs"]', '            ep_min, ep_max = min(p, pl["max_epochs"]), pl["max_epochs"]'),
    ("cap with the whole anomaly factor", OFF, S, '(wall_max_hours + (ca["anomaly_factor"] - 1) * per_training_max_hours)', '(wall_max_hours + ca["anomaly_factor"] * per_training_max_hours)'),
    ("test frames of another fold", OFF, S, '"test_frames_of_fold": test[f - 1],', '"test_frames_of_fold": test[f % 4],'),
    ("after outcome not checked", OFF, S, '    if any(x not in t["after_outcome"] for x in', '    if False and any(x not in t["after_outcome"] for x in'),
    ("control rule without the machine accepted", OFF, S, '    if "machine" not in t["control_rule"]:', "    if False:"),
    ("every non-finite epoch recorded", OFF, E, "                out.append([k, i + 1])\n                break", "                out.append([k, i + 1])"),
    ("statistics of unknown finiteness accepted", OFF, E, "    if statistics_finite is not True:", "    if statistics_finite is False:"),
    ("percentage points rounded half to even", OFF, E, '(d * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)', '(d * 100).quantize(Decimal("0.01"))'),
    ("negated interval without the swap", OFF, E, '"interval": {k: {"low": neg(v["high"]), "high": neg(v["low"]),', '"interval": {k: {"low": neg(v["low"]), "high": neg(v["high"]),'),
    ("units out of order accepted", OFF, E, "    if keys != sorted(keys):", "    if False:"),
    ("EER conditions required at 0.5", OFF, E, "                if threshold_rule == PRIMARY_THRESHOLD and t != float(FR.FIXED_THRESHOLD):",
     "                if t != float(FR.FIXED_THRESHOLD):"),
    ("descriptive conditions with readings", OFF, E, "        r = pair_analysis(conditions, p, t, reasons, readings=False)", "        r = pair_analysis(conditions, p, t, reasons, readings=True)"),
    ("missing descriptive condition accepted", OFF, E, "    if missing:\n        fail(f\"descriptive analysis", "    if False:\n        fail(f\"descriptive analysis"),
    ("control finding not recorded", TF, S, '    if rec["outcome"] != "reproduced":\n        ctx.record_finding(', '    if False:\n        ctx.record_finding('),
    ("not evaluable finding not recorded", TF, S, '        ctx.record_finding(net, rec["outcome"]["reason"])', '        pass'),
    ("control finding after the after-threads check", TF, S, '    print(f"   {CONTROL} control: {rec[\'outcome\']}", flush=True)\n',
     '    PAC.check_threads(tf, ctx.cfg_pr, f"after {CONTROL}")\n    print(f"   {CONTROL} control: {rec[\'outcome\']}", flush=True)\n'),
    ("not evaluable finding after the after-threads check", TF, S, '        ctx.record_finding(net, rec["outcome"]["reason"])',
     '        PAC.check_threads(tf, ctx.cfg_pr, f"after {net}")\n        ctx.record_finding(net, rec["outcome"]["reason"])'),
    ("findings not in the running status", TF, S, '"threads": threads,\n                              "findings": findings,\n', '"threads": threads,\n'),
    ("finding without the running status", TF, S, '            print(f"FINDING: {what} {reason}", flush=True)\n        running_status()',
     '            print(f"FINDING: {what} {reason}", flush=True)'),
    ("peak RSS not pinned", OFF, S, '("nuaa_session_ablation_run", "peak_rss_mb"): SR.peak_rss_mb, ', ''),
    ("findings not in the FAIL status", TF, S, '"reason": reason, "outputs": outputs,\n                                  "findings": findings,', '"reason": reason, "outputs": outputs,'),
    ("check-machine passes on a difference", OFF, S, '    if r["equal"]:\n        print(f"check-machine:', '    if True:\n        print(f"check-machine:'),
    ("values per channel not compared with the registry", TF, S,
     '    c2 = all(v["count"] == v["count_expected"] and v["channels"] == v["channels_registry"] for v in layers.values())', "    c2 = True"),
]
# mutazione di controllo, che deve SOPRAVVIVERE
CONTROL = [("control (must survive): a comment changed", OFF, E, "# ============================================================================ (a) quantità lineari",
            "# ============================================================================ (a) quantita lineari")]
# controlli dichiarati ridondanti, che devono SOPRAVVIVERE:
REDUNDANT = [
    # con 72 chiavi diverse nel dominio del piano (3 reti x 4 fold x 3 seed x 2 patience = 72) i conteggi per patience, rete e
    # cella sono per forza quelli attesi
    ("declared redundant (must survive): counts per patience, network and cell of the matrix", OFF, S,
     '    if set(per_p.values()) != {36} or set(per_n.values()) != {24} or set(per_cell.values()) != {12}:', "    if False:"),
    # una cartella con il run id di questo tentativo è contata fra i tentativi precedenti, quindi attempt_rule la rifiuta prima
    ("declared redundant (must survive): existing run folder", OFF, S, "    if os.path.lexists(run):\n        fail(f\"the run folder", "    if False:\n        fail(f\"the run folder"),
    # guardiano alla soglia EER: una soglia non finita o fuori da [0, 1] è già rifiutata da prepare (D.check_threshold)
    ("declared redundant (must survive): threshold domain in the guardian", OFF, E, '                t = D.check_threshold(t, f"guardian {n}")', "                t = float(t)"),
    # bn_check, controllo 1: order è la lista della configurazione, confrontata nello stesso controllo con i nomi del grafo
    # e del registro; pad_arch_run.reestimate ha già rifiutato un ordine non ammesso sul grafo prima della ristima
    ("declared redundant (must survive): BN order against the graph in bn_check", TF, S,
     '"order_problems": R.bn_order_problems(list(order), AM.model_topology(orig)),', '"order_problems": [],'),
    # bn_check, controllo 3: l'originale invariato è già controllato subito dopo la ristima (trial_attacknet)
    ("declared redundant (must survive): original unchanged in bn_check", TF, S, " and BR.snapshot_weights(orig) == snap\n", "\n"),
    # bn_check, controllo 1: la somma delle variabili mobili è già richiesta da AN.check_model sul modello ricaricato
    ("declared redundant (must survive): moving variables in bn_check", TF, S,
     '        and c1["moving_values"] == c1["non_trainable"] == cfg["architectures"]["networks"][net]["non_trainable_parameters"]', "        and True"),
    # bn_check, controllo 3: BR.tensor_problems, nello stesso controllo e in pr27_checks, vuole ogni tensore mobile cambiato
    ("declared redundant (must survive): variance changed in bn_check", TF, S,
     'all(v["moving_mean_changed"] and v["moving_variance_changed"] for v in layers.values())', 'all(v["moving_mean_changed"] for v in layers.values())'),
]


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
    ap = argparse.ArgumentParser(description="Controllo delle mutazioni del confronto C5 (offline, sintetico)")
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
    modes = sorted({m[1] for _, m in chosen} | {m[1] for m in controls})
    killed = invalid = survived_controls = 0
    with tempfile.TemporaryDirectory(prefix="anetcmp-mutations-") as tmp:
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
          + ("" if len(chosen) == len(MUTANTS) else f" (selection of {len(chosen)} of {len(MUTANTS)})"))
    if controls:
        print(f"control mutations survived: {survived_controls}/{len(controls)}")
    ok = killed == len(chosen) and survived_controls == len(controls)
    print("RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
