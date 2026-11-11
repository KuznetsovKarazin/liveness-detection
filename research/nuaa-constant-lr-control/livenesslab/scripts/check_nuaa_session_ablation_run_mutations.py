"""
Controllo delle mutazioni di scripts/nuaa_session_ablation_run.py (run di ablazione della sessione 03), offline e
deterministico, con la stessa meccanica dei controlli precedenti (check_nuaa_folds_run_mutations.py).

Copia il codice (scripts/nuaa_session_ablation_run.py e la sua suite, scripts/nuaa_folds_run.py e la sua suite, che
fornisce il mondo sintetico di base, scripts/m2_writers.py, scripts/path_guard.py, scripts/package_rules.py e, se c'è,
la configurazione delivery/sessionablationrun/ e le configurazioni emendate delivery/patienceverify/ e delivery/constlr/) in una cartella temporanea, applica una mutazione dichiarata alla volta
(coppia cerca/sostituisci che deve comparire esattamente una volta: se un modello non si trova più, il controllo si
ferma con un errore) ed esegue scripts/check_nuaa_session_ablation_run.py sulla copia mutata. Una mutazione è rilevata
solo se la suite arriva al riepilogo («tests run») con almeno un fallimento o errore, cioè dal comportamento; una suite
che non parte la rende non valida. Una mutazione di controllo (solo un commento) e le mutazioni dichiarate ridondanti
devono sopravvivere.

Nessun dato reale: SESSABLRUN_INPUT_ROOT è la radice della copia, che contiene solo il codice e la configurazione
consegnata, quindi il test «real inputs» (check-inputs sui dati veri, 2,5 GB di frame in memoria) è saltato nella copia
non mutata e in ogni mutante (salto dichiarato, MISSING_INPUT); resta, una volta sola, nella suite eseguita dalla radice
del repository. Le mutazioni girano una alla volta per default (--jobs 1, al massimo 2): la macchina ha 16 GB.

Test saltati: la copia non mutata deve passare e può saltare test solo per un ingresso mancante dichiarato
(MISSING_INPUT della suite); una mutazione conta come rilevata solo per fallimenti o errori, mai per un salto.

  .venv/bin/python scripts/check_nuaa_session_ablation_run_mutations.py            # tutte le mutazioni, una alla volta
  .venv/bin/python scripts/check_nuaa_session_ablation_run_mutations.py --list     # solo l'elenco, con la verifica dei modelli
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
from check_nuaa_session_ablation_run import MISSING_INPUT  # noqa: E402

TESTS = "scripts/check_nuaa_session_ablation_run.py"
S = "scripts/nuaa_session_ablation_run.py"
COPY = ("scripts/nuaa_session_ablation_run.py", "scripts/check_nuaa_session_ablation_run.py", "scripts/nuaa_folds_run.py",
        "scripts/check_nuaa_folds_run.py", "scripts/m2_writers.py", "scripts/path_guard.py", "scripts/package_rules.py")
CONFIG = "nuaa_session_ablation_run.json"
CONFIG_DIR = ("delivery", "sessionablationrun")
PATIENCE12 = ("delivery", "patienceverify", "nuaa_session_ablation_run_patience12.json")   # configurazione emendata, se c'è
CONSTANT_LR = ("delivery", "constlr", "nuaa_session_ablation_run_constant_lr.json")       # controllo a lr costante, se c'è
F = "False"
MAX_JOBS = 2                   # mai più di due suite in parallelo (16 GB)
# (nome, file mutato, testo cercato, testo sostitutivo)
MUTANTS = [
    # configurazione: bracci, dev, ordine, confronti, regola, bootstrap
    ("arm file, role or label not fixed", S, '        if a["file"] != ARM_FILE[arm] or a["role"] != ARM_ROLE[arm] or a["label"] != ARM_LABEL[arm]:', "        if False:"),
    ("arm order not checked", S, "    if not isinstance(arms, dict) or list(arms) != list(ARMS):", "    if not isinstance(arms, dict) or sorted(arms) != sorted(ARMS):"),
    ("A* and B totals not compared", S, '    if arms["arm_a_star"]["total"] != arms["arm_b"]["total"]:', "    if False:"),
    ("B' cells not tied to A*", S, '    if [arms["arm_b_prime"]["cells"][x] for x in CELLS_12] != [arms["arm_a_star"]["cells"][x] for x in CELLS_12]:', "    if False:"),
    ("A* cell larger than B accepted", S, '    if any(arms["arm_a_star"]["cells"][x] > arms["arm_b"]["cells"][x] for x in CELLS_12):', "    if False:"),
    ("dev file and role not fixed", S, '    if dv["file"] != DEV_FILE or dv["role"] != DEV_ROLE:', "    if False:"),
    ("dev cells with session 03 accepted", S, '    _cells(dv["cells"], CELLS_12, "dev.cells")', '    _cells(dv["cells"], list(dv["cells"]), "dev.cells")'),
    ("training order not compared", S, '    if cfg["training_order"] != order:', "    if False:"),
    ("train seeds not fixed", S, "    if seeds != TRAIN_SEEDS:", "    if False:"),
    ("cap not fixed", S, '    if bu["wall_time_cap_minutes"] != CAP_MINUTES:', "    if False:"),
    ("number of trainings not compared", S, '    if bu["trainings"] != len(ARMS) * exp["k"] * len(seeds):', "    if False:"),
    ("INCOMPLETE outcome not declared", S, '    if "INCOMPLETE" not in bu["on_cap_exceeded"]:', "    if False:"),
    ("comparisons not fixed", S, '    if co["list"] != want:', "    if False:"),
    ("primary reading not fixed", S, '    if (rr["primary_comparison"], rr["primary_stratum"], rr["primary_metrics"]) != (PRIMARY["comparison"], PRIMARY["stratum"], PRIMARY["metrics"]):', "    if False:"),
    ("reading rule text not compared", S, '    if rr["rule"] != READING_RULE or rr["verdicts"] != list(VERDICTS) or rr["statistical_test"] != "none":', "    if False:"),
    ("bootstrap recipe not compared", S, '    if cfg["bootstrap"] != {**BOOT, "recipe": BOOT_RECIPE}:', "    if False:"),
    ("reference label not fixed", S, '    if ra["label"] != REFERENCE_LABEL:', "    if False:"),
    ("training recipe not compared", S, '    if not amended and cfg["training"] != FR.FROZEN_TRAINING:', "    if False:"),
    ("threshold criterion not compared", S, '    if th["criterion"] != FR.THRESHOLD_CRITERION:', "    if False:"),
    ("common dev not declared in the threshold set", S, '    if "common dev" not in th["set"]:', "    if False:"),
    ("test counts not tied to the schema", S, '    if te["per_fold"] != exp["test_per_fold"]:', "    if False:"),
    ("frames count not tied to the schema", S, '    if ff["n"] != exp["n_to_extract"]:', "    if False:"),
    ("declaration flags not required", S, "        if dec[k] is not True:", "        if False:"),
    ("metric keys not compared", S, '    if me["strata"] != list(STRATA) or me["decision"] != FR.DECISION_RULE or me["keys"] != list(METRIC_KEYS):', "    if False:"),
    ("unknown top key accepted", S, "    if sorted(k for k in cfg if k != AMENDMENT_KEY) != sorted(TOP_KEYS):", "    if not set(TOP_KEYS) <= set(cfg):"),
    ("comparison arms swapped", S, '("a_star_minus_b", "arm_a_star", "arm_b", "primary")', '("a_star_minus_b", "arm_b", "arm_a_star", "primary")'),
    # ingressi: schema, run delle celle, manifest delle celle, annidamento
    ("schema configuration SHA-256 not checked", S, '    if h != sr["config_sha256"] or h != SCHEMA_CONFIG_SHA256:', "    if False:"),
    ("schema training fields not compared", S, '    if diff or not str(tr.get("device", "")).startswith("CPU"):', "    if False:"),
    ("schema seeds not compared", S, '    if sc.get("train_seeds") != cfg["train_seeds"]:', "    if False:"),
    ("cells run status not checked", S, 'st.get("status") != "PASS" or st.get("run_id") != cr["run_id"]', 'st.get("run_id") != cr["run_id"]'),
    ("cells public SHA256SUMS not anchored", S, '    if sha256_bytes(pub) != cr["public_sha256sums_sha256"]:', "    if False:"),
    ("cells restricted SHA256SUMS not anchored", S, '    if h != cr["restricted_sha256sums_sha256"]:', "    if False:"),
    ("cells configuration SHA-256 not pinned", S, '    if ch != cr["config_sha256"] or ch != CELLS_CONFIG_SHA256:', "    if False:"),
    ("cells allocations not compared", S, '        if want[arm] != arms[arm]["cells"]:', "        if False:"),
    ("cells dev allocation not compared", S, '    if dev_want != cfg["dev"]["cells"]:', "    if False:"),
    ("cell file not required in SHA256SUMS", S, '            if name not in listed:\n                fail(f"cells run SHA256SUMS: {name} not listed")',
     '            if name not in listed:\n                listed[name] = sha256_bytes(read_regular(folder / RESTRICTED / name, name))'),
    ("cell file SHA-256 not compared", S, '            if hh != listed[name]:\n                fail(f"cell manifest', '            if False:\n                fail(f"cell manifest'),
    ("cell fold and role not checked", S, '            if any(r["fold"] != str(fold) or r["role"] != role for r in rows):', "            if False:"),
    ("cell duplicated keys accepted", S, '            if len({r["key"] for r in rows}) != len(rows):\n                fail(f"cell manifest {name}: duplicated keys")',
     '            if False:\n                fail(f"cell manifest {name}: duplicated keys")'),
    ("cell counts not checked", S, '            if dict(got) != {x: n for x, n in cells.items()} or len(rows) != sum(cells.values()):', "            if False:"),
    ("cell row not compared with the official lists", S, '            if o is None or any(o[c] != r[c] for c in ("label", "subject", "session", "sha256", "bytes")):\n                fail(f"cell manifest {unit}',
     '            if o is None:\n                fail(f"cell manifest {unit}'),
    ("cell in_cache not compared with the cache", S, '            if r["in_cache"] not in ("0", "1") or (r["in_cache"] == "1") != (r["key"] in cache_keys):\n                fail(f"cell manifest {unit}',
     '            if r["in_cache"] not in ("0", "1"):\n                fail(f"cell manifest {unit}'),
    ("dev with session 03 accepted", S, '        if any(r["session"] == "03" for r in inp.cells[("dev", fold)]):', "        if False:"),
    ("dev not compared with the schema dev", S, '        if keys["dev"] != schema_dev12:', "        if False:"),
    ("arm B not compared with the schema fitting", S, '        if keys["arm_b"] != schema_fit12:', "        if False:"),
    ("A* and B' not nested in B", S, '        if not a12 <= keys["arm_b"] or not keys["arm_b_prime"] <= keys["arm_b"]:', "        if False:"),
    ("B' not compared with A* 01-02", S, '        if keys["arm_b_prime"] != a12:', "        if False:"),
    ("S12 subjects not checked", S, '        if not subj["arm_a_star"] <= subj["arm_b"] or not subj["arm_b_prime"] <= subj["arm_b"]:', "        if False:"),
    # ingressi: run dei fold, frame estratti, sorgenti, codice, ambiente
    ("folds run status not checked", S, 'st.get("status") != "PASS" or st.get("run_id") != fr_["run_id"]', 'st.get("run_id") != fr_["run_id"]'),
    ("folds run SHA256SUMS not anchored", S, '        if sha256_bytes(b) != fr_[key]:', "        if False:"),
    ("folds run file SHA-256 not compared", S, '        if sha256_bytes(b) != sums[sub][name]:', "        if False:"),
    ("extracted_frames.csv rows not compared", S, '        if (r["subject_id"], r["session_id"], r["label"], r["file_sha256"]) != (o["subject"], o["session"], o["label"], o["sha256"]) \\',
     "        if False \\"),
    ("extracted_frames.csv keys not compared", S, "    if keys != sorted(keys) or keys != inp.to_extract:", "    if False:"),
    ("frames file SHA-256 not checked", S, '    if h != ff["sha256"]:', "    if False:"),
    ("frame hashes not compared", S, '    bad = sum(1 for a, r in zip(hashes, inp.extracted_rows) if a != r["frame_sha256"])', "    bad = 0"),
    ("frames file keys not compared", S, "    if keys != inp.to_extract:", "    if False:"),
    ("frames file members not checked", S, '    if sorted(mem) != ["X", "keys"] or mem["X"] != want_x or mem["keys"]["shape"] != [ff["n"]] or mem["keys"]["dtype"][1] != "U":', "    if False:"),
    ("source found twice or never accepted", S, "            if n != 1:", "            if False:"),
    ("source and in_cache not compared", S, '            if (r["in_cache"] == "1") != (r["key"] in inp.cache_index):', "            if False:"),
    ("architecture SHA-256 not checked", S, '    if h != ar["sha256"]:', "    if False:"),
    ("architecture not verified before loading", S, '        if sha256_file(ap) != cfg["architecture"]["sha256"]:', "        if False:"),
    ("environment variables not refused", S, '    if refused:\n        fail(f"environment variables', '    if False:\n        fail(f"environment variables'),
    ("originals not checked", S, "    FR.check_originals(v, base, inp)\n    return inp", "    return inp"),
    ("TensorFlow import at check not detected", S, '"tensorflow_state": FR.tf_state()}', '"tensorflow_state": "not imported"}'),
    ("environment not compared with the configuration", S, "    inp.environment = FR.environment(v)", "    inp.environment = FR.environment()"),
    # training, tetto, soglia, punteggi
    ("cap not checked before training", S, '    budget.check(f"train {name}")', "    pass"),
    ("cap not checked before the threshold", S, '    budget.check(f"threshold {name}")', "    pass"),
    ("cap not checked before the test scoring", S, '    budget.check(f"score {name}")', "    pass"),
    ("epoch callback not tied to the budget", S, 'lambda step: budget.check(f"{step} of {name}"))', "lambda step: None)"),
    ("cap never exceeded", S, "        if self.elapsed() > self.cap:", "        if False:"),
    ("elapsed time from the wall clock only", S, "        return max(clock() - self.start[0], time.time() - self.start[1])", "        return time.time() - self.start[1]"),
    ("interrupted model files kept", S, "                removed = discard_model(run, name)", "                removed = []"),
    ("temporary checkpoint of the interrupted model kept", S, "    for e in checkpoint_temps(rdir, name):\n        (rdir / e).unlink()\n        removed.append(e)",
     "    for e in []:\n        (rdir / e).unlink()\n        removed.append(e)"),
    ("trainings go on after the cap", S, "                break\n            cards[(arm, fold, seed)]", "                continue\n            cards[(arm, fold, seed)]"),
    ("elapsed time from the monotonic clock only", S, "        return max(clock() - self.start[0], time.time() - self.start[1])", "        return clock() - self.start[0]"),
    ("final_writes_after_cap from the stop instead of the clock", S, '"final_writes_after_cap": over_cap, "checks"', '"final_writes_after_cap": bool(stopped), "checks"'),
    # arresto volontario (SIGUSR1)
    ("voluntary stop ignored", S, "        if self.stop_requested:\n", "        if False:\n"),
    ("SIGUSR1 handler not installed", S, "            signal.signal(signal.SIGUSR1, stop)", "            pass"),
    ("voluntary stop not declared in the configuration", S, '    if "SIGUSR1" not in bu["voluntary_stop"] or "INCOMPLETE" not in bu["voluntary_stop"]:', "    if False:"),
    # FAIL tardivo: modelli completi conservati senza verdetto
    ("complete models not kept on a late FAIL", S, "        kept, kept_sums = list(completed), None", "        kept, kept_sums = [], None"),
    ("public not removed on a late FAIL", S, '    if (run / PUBLIC).is_dir():\n        left += [f"{PUBLIC}/{n}"', '    if False:\n        left += [f"{PUBLIC}/{n}"'),
    ("per_subject.csv kept on a late FAIL", S, "[n for n in restricted_outputs(cfg) if n not in keep]",
     '[n for n in restricted_outputs(cfg) if n not in keep and n != "per_subject.csv"]'),
    ("late FAIL not declared in the configuration", S, '    if "FAIL" not in bu["on_failure"] or "kept" not in bu["on_failure"]:', "    if False:"),
    # codice condiviso con il run dei fold
    ("shared code not compared with the folds run", S, '    if bad:\n        fail(f"code shared with the folds run', '    if False:\n        fail(f"code shared with the folds run'),
    ("peak memory not in the model line", S, "AUROC {f4(m['auroc'])}, peak RSS {peak_rss_mb():.0f} MB\"", "AUROC {f4(m['auroc'])}\""),
    ("INCOMPLETE recorded as PASS", S, '        status = "PASS" if not missing else "INCOMPLETE"', '        status = "PASS"'),
    ("INCOMPLETE exits with 0", S, '    return 0 if status == "PASS" else 2', "    return 0"),
    ("cap treated as a failure", S, "            except CapExceeded as e:", "            except Failure as e:"),
    ("threshold with inverted dev labels", S, '    sel = FR.eer_threshold(dev["y"], p_dev[:, 1])', '    sel = FR.eer_threshold(1 - dev["y"], p_dev[:, 1])'),
    ("threshold.json not read back", S, '    if json.loads(back_bytes.decode("utf-8")) != thr:', "    if False:"),
    ("threshold SHA-256 not in the score rows", S, '                     "threshold_sha256": thr["sha256"]})', '                     "threshold_sha256": ""})'),
    ("val_loss of the checkpoint not recomputed", S, '    if not (diff <= FR.VAL_LOSS_TOL[0] or diff <= FR.VAL_LOSS_TOL[1] * card["best_val_loss"]):', "    if False:"),
    ("val_loss check on the clipped probabilities", S, '    vl = val_loss_logits(dev_logits(weights, card, dev["X"], p_dev, name), dev["y"])', '    vl = FR.val_loss(p_dev, dev["y"])'),
    ("val_loss from the logits with clipping", S, "    return float(np.mean(lse - z[np.arange(len(y)), y]))", "    return float(np.mean(np.minimum(lse - z[np.arange(len(y)), y], -np.log(FR.EPS))))"),
    ("logits not compared with the scorer probabilities", S, "    if gap > PROB_TOL:", "    if False:"),
    ("non-finite logits accepted", S, "    if z.shape != (len(X), 2) or not np.all(np.isfinite(z)):", "    if z.shape != (len(X), 2):"),
    ("val_loss_check not required in the configuration", S, '    if "logits" not in th["val_loss_check"]:', "    if False:"),
    ("card not checked against the recipe", S, '    if problems := FR.card_problems(card, cfg["training"], seed, weights):', "    if problems := []:"),
    ("card counts not checked", S, '    if card["n_fitting"] != cfg["arms"][arm]["total"] or card["n_dev"] != cfg["dev"]["total"]:', "    if False:"),
    ("fitting frames of arm B for every arm", S, "    fit, dev = FR.assemble(store, inp.cells[(arm, fold)]),", '    fit, dev = FR.assemble(store, inp.cells[("arm_b", fold)]),'),
    ("dev of the schema (with session 03) instead of the common dev", S, 'FR.assemble(store, inp.cells[("dev", fold)])', 'FR.assemble(store, inp.folds[fold]["dev"])'),
    ("test of another fold", S, '    test = FR.assemble(store, inp.folds[fold]["test"])', '    test = FR.assemble(store, inp.folds[fold % 4 + 1]["test"])'),
    ("tie at the dev threshold counted as attack", S, '"prediction_dev": "1" if v > thr["value"] else "0",', '"prediction_dev": "1" if v >= thr["value"] else "0",'),
    ("tie at 0.5 counted as attack", S, '"prediction_05": "1" if v > FR.FIXED_THRESHOLD else "0",', '"prediction_05": "1" if v >= FR.FIXED_THRESHOLD else "0",'),
    ("arm column wrong in the score rows", S, 'rows.append({"arm": arm, "fold": str(fold)', 'rows.append({"arm": "arm_b", "fold": str(fold)'),
    # metriche, bootstrap, confronti, regola di lettura
    ("Mann-Whitney ties not halved", S, "                U[i, j] = (int(lo) + int(hi)) / 2.0", "                U[i, j] = float(hi)"),
    ("AUROC ignores the resampling weights", S, '    out["auroc"] = float((w @ st["U"] @ w) / (na * nb)) if na and nb else None', '    out["auroc"] = float(st["U"].sum() / (na * nb)) if na and nb else None'),
    ("counts ignore the resampling weights", S, '    na, nb = int(w @ st["na"]), int(w @ st["nb"])', '    na, nb = int(st["na"].sum()), int(st["nb"].sum())'),
    ("accepted attacks counted on the attacks accepted as attacks", S, '            st["acc05"][i] += r["prediction_05"] == "0"', '            st["acc05"][i] += r["prediction_05"] == "1"'),
    ("dev decisions taken at 0.5", S, '            st["rejdev"][i] += r["prediction_dev"] == "1"', '            st["rejdev"][i] += r["prediction_05"] == "1"'),
    ("bootstrap seed 1", S, '    rng = np.random.default_rng(BOOT["seed"])\n    return [rng.integers', "    rng = np.random.default_rng(1)\n    return [rng.integers"),
    ("bootstrap with 100 resamples", S, '    return [rng.integers(0, n, n) for _ in range(BOOT["resamples"])]', "    return [rng.integers(0, n, n) for _ in range(100)]"),
    ("bootstrap percentiles [5, 95]", S, '    lo, hi = np.percentile(vals, BOOT["percentiles"])', "    lo, hi = np.percentile(vals, [5, 95])"),
    ("paired bootstrap with independent indices", S, "        a, b = weighted_metrics(sm, w), weighted_metrics(ss, w)",
     "        a, b = weighted_metrics(sm, w), weighted_metrics(ss, np.bincount(np.random.default_rng(7).integers(0, n, n), minlength=n))"),
    ("paired interval with fewer than 5 subjects", S, '        return {**out, **{k: {"low": None, "high": None, "valid_resamples": 0} for k in METRIC_KEYS}}', "        pass"),
    ("arm interval with fewer than 5 subjects", S, '        return {**out, **{k: {"low": None, "high": None, "valid_resamples": 0} for k in keys}}', "        pass"),
    ("resample without a class counted", S, "            if a[k] is not None and b[k] is not None:", "            if True:"),
    ("paired arms on different frames accepted", S, '    if sm["subjects"] != ss["subjects"] or any(not np.array_equal(sm[k], ss[k]) for k in ("na", "nb")):', "    if False:"),
    ("same test frames not checked", S, '        fail("comparison: the two arms are not scored on the same test frames")', "        pass"),
    ("difference reversed", S, "    return None if a is None or b is None else a - b", "    return None if a is None or b is None else b - a"),
    ("pooled from the first fold only", S, '    return sorted((r for f in range(1, k + 1) for r in scores[(arm, f, seed)]), key=lambda r: r["sample_id"])',
     '    return sorted((r for f in range(1, 2) for r in scores[(arm, f, seed)]), key=lambda r: r["sample_id"])'),
    ("pooled summary with a seed missing", S, "        if all(str(s) in pooled[arm] for s in seeds):", "        if any(str(s) in pooled[arm] for s in seeds):"),
    ("comparison summary with a seed missing", S, "            if all(e is not None for es in per_seed.values() for e in es):",
     "            if any(e is not None for es in per_seed.values() for e in es):"),
    ("pairing not checked per fold and seed", S, "                    same_frames(scores[(m, fold, seed)], scores[(s, fold, seed)])", "                    pass"),
    ("rule on the first two seeds only", S,
     '            per_seed = {key: [c["pooled_per_seed"].get(str(seed), {}).get(st, {}).get(key) for seed in seeds] for key in METRIC_KEYS}',
     '            per_seed = {key: [c["pooled_per_seed"].get(str(seed), {}).get(st, {}).get(key) for seed in seeds[:2]] for key in METRIC_KEYS}'),
    ("rule: sign of the first seed only", S, "    signs = {int(np.sign(x)) for x in d}", "    signs = {int(np.sign(d[0]))}"),
    ("paired interval refused with exactly 5 subjects", S, '    n = len(sm["subjects"])\n    out = {"n_subjects": n, "resamples": BOOT["resamples"], "seed": BOOT["seed"], "percentiles": BOOT["percentiles"]}\n    if n < BOOT["min_subjects"]:',
     '    n = len(sm["subjects"])\n    out = {"n_subjects": n, "resamples": BOOT["resamples"], "seed": BOOT["seed"], "percentiles": BOOT["percentiles"]}\n    if n <= BOOT["min_subjects"]:'),
    ("arm interval refused with exactly 5 subjects", S, '    n = len(st["subjects"])\n    out = {"n_subjects": n, "resamples": BOOT["resamples"], "seed": BOOT["seed"], "percentiles": BOOT["percentiles"]}\n    if n < BOOT["min_subjects"]:',
     '    n = len(st["subjects"])\n    out = {"n_subjects": n, "resamples": BOOT["resamples"], "seed": BOOT["seed"], "percentiles": BOOT["percentiles"]}\n    if n <= BOOT["min_subjects"]:'),
    ("reference A of the first seed for every seed", S, 'reference["pooled_per_seed"].get(str(s), {})', 'reference["pooled_per_seed"].get(str(seeds[0]), {})'),
    ("reference A intervals of the first seed for every seed", S, 'reference["bootstrap_per_seed"].get(str(s), {})', 'reference["bootstrap_per_seed"].get(str(seeds[0]), {})'),
    ("pooled with a missing fold", S, "    if any((arm, f, seed) not in scores for f in range(1, k + 1)):\n        return None", "    if False:\n        return None"),
    ("rule: zero difference accepted", S, "    if 0 in signs:", "    if False:"),
    ("rule: mixed signs accepted", S, "    if len(signs) != 1:", "    if False:"),
    ("rule: interval not required to exclude 0", S, '    if not all((e["low"] > 0) if sign > 0 else (e["high"] < 0) for e in per_seed):', "    if False:"),
    ("rule: interval on the other side of 0 accepted", S, '(e["low"] > 0) if sign > 0 else (e["high"] < 0) for e in per_seed', '(e["low"] > 0 or e["high"] < 0) for e in per_seed'),
    ("rule: interval touching 0 accepted", S, '(e["low"] > 0) if sign > 0 else', '(e["low"] >= 0) if sign > 0 else'),
    ("rule: missing seed ignored", S, '    if any(e is None for e in per_seed):\n        return {**out, "reason": "a seed has no pooled value (incomplete run)"}',
     '    if False:\n        return {**out, "reason": "a seed has no pooled value (incomplete run)"}'),
    ("rule: missing interval ignored", S, '    if any(e["low"] is None or e["high"] is None for e in per_seed):', "    if False:"),
    ("rule: direction reversed", S, '    return "higher" if sign > 0 else "lower"', '    return "lower" if sign > 0 else "higher"'),
    ("primary label on every stratum", S, '                primary = name == PRIMARY["comparison"] and st == PRIMARY["stratum"] and key in PRIMARY["metrics"]',
     '                primary = name == PRIMARY["comparison"] and key in PRIMARY["metrics"]'),
    ("primary verdicts from stratum all", S, '    primary = {key: comps[PRIMARY["comparison"]]["verdicts"][PRIMARY["stratum"]][key] for key in PRIMARY["metrics"]}',
     '    primary = {key: comps[PRIMARY["comparison"]]["verdicts"]["all"][key] for key in PRIMARY["metrics"]}'),
    ("per-subject rows without the arm", S, "        subject_rows += [[arm] + r for r in FR.per_subject_rows(fold, seed, rows)]", '        subject_rows += [["arm_b"] + r for r in FR.per_subject_rows(fold, seed, rows)]'),
    ("reference label not written", S, '    ref = {"label": REFERENCE_LABEL,', '    ref = {"label": "reference",'),
    # uscite, stato, protezioni
    ("public scan skipped", S, '            if probs := FR.public_problems(text, inp, pname.endswith(".md")):', "            if probs := []:"),
    ("restricted SHA256SUMS without the weights", S, '        mw.write_all(rdir, {"SHA256SUMS": "".join(f"{sha256_file(rdir / n)}  {n}\\n" for n in sorted(names))})',
     '        mw.write_all(rdir, {"SHA256SUMS": "".join(f"{sha256_file(rdir / n)}  {n}\\n" for n in sorted(names) if not n.endswith(".keras"))})'),
    ("public anchor not written in the restricted folder", S, '"public_SHA256SUMS.sha256": sha256_bytes(public["SHA256SUMS"].encode("utf-8")) + "  public/SHA256SUMS\\n"},',
     '"public_SHA256SUMS.sha256": "none  public/SHA256SUMS\\n"},'),
    ("existing status.json not refused", S, "    if (run / STATUS).exists() or (run / STATUS).is_symlink():", "    if False:"),
    ("run folder not validated", S, "    if msg := run_dir_problem(run, ROOT, inputs, cfg, out_root):", '    if msg := "":'),
    ("foreign entries accepted", S, "                if not any(mw.owned_name(e, n) for n in names):", "                if False:"),
    ("authorisation flag not required", S, "    if not a.i_am_authorized:", "    if False:"),
    ("dirty tree not declared", S, '("-dev" if state["dirty"] else "")', '""'),
    ("failure recorded as PASS", S, '"status": "FAIL", "passed": False, "reason": reason', '"status": "PASS", "passed": False, "reason": reason'),
    ("outputs not removed after a failure", S, "        left = [] if kept else invalidate_outputs(run, cfg)", "        left = []"),
    ("previous outputs not invalidated before the run", S, "        left = invalidate_outputs(run, cfg)\n        if left:", "        left = []\n        if left:"),
    ("run log SHA-256 not recorded", S, '                              "run_log_sha256": sha256_file(run / LOG)})', '                              "run_log_sha256": "none"})'),
    ("SIGTERM handler not installed", S, "    sigs = [signal.SIGTERM]\n", "    sigs = []\n"),
    ("SIGHUP handler installed even when ignored", S, "    if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:\n        sigs.append(signal.SIGHUP)", "    sigs.append(signal.SIGHUP)"),
    # configurazioni emendate (tabella AMENDMENTS: pazienza 12 del 7/11/2026, lr costante del 10/11/2026): costanti, voci,
    # campi, confronto con il genitore, nome, messaggi; revisione del trainer con lo scheduler operativo
    ("patience12 SHA-256 constant changed", S, 'PATIENCE12_CONFIG_SHA256 = "ca039df5', 'PATIENCE12_CONFIG_SHA256 = "0a039df5'),
    ("constant_lr SHA-256 constant changed", S, 'CONSTANT_LR_CONFIG_SHA256 = "60b38e09', 'CONSTANT_LR_CONFIG_SHA256 = "00b38e09'),
    ("constant_lr configuration path changed", S, 'CONSTANT_LR_CONFIG = "delivery/constlr/nuaa_session_ablation_run_constant_lr.json"',
     'CONSTANT_LR_CONFIG = "delivery/constlr/constant_lr.json"'),
    ("trainer revision SHA-256 changed", S, 'TRAINER_REVISION_SHA256 = "82fed566', 'TRAINER_REVISION_SHA256 = "02fed566'),
    ("parent read from the amended configuration", S, "PARENT_CONFIG = DEFAULT_CONFIG", "PARENT_CONFIG = PATIENCE12_CONFIG"),
    ("constant_lr parent read from the delivered configuration", S, "CONSTANT_LR_PARENT_CONFIG = PATIENCE12_CONFIG",
     "CONSTANT_LR_PARENT_CONFIG = DEFAULT_CONFIG"),
    ("patience12 target value changed", S, '"field": "training.early_stopping_patience", "from": 4, "to": 12,',
     '"field": "training.early_stopping_patience", "from": 4, "to": 13,'),
    ("patience12 field changed", S, '"field": "training.early_stopping_patience", "from": 4,', '"field": "training.max_epochs", "from": 4,'),
    ("constant_lr field changed", S, '"field": "training.scheduler", "from"', '"field": "training.optimizer", "from"'),
    ("constant_lr from and to swapped", S, '"from": FR.SCHEDULER_REDUCE,\n     "to": FR.SCHEDULER_CONSTANT,',
     '"from": FR.SCHEDULER_CONSTANT,\n     "to": FR.SCHEDULER_REDUCE,'),
    ("constant_lr training without the constant scheduler", S, '"training": {"early_stopping_patience": 12, "scheduler": FR.SCHEDULER_CONSTANT}}',
     '"training": {"early_stopping_patience": 12}}'),
    ("constant_lr training without patience 12", S, '"training": {"early_stopping_patience": 12, "scheduler": FR.SCHEDULER_CONSTANT}}',
     '"training": {"scheduler": FR.SCHEDULER_CONSTANT}}'),
    ("patience12 training without patience 12", S, '"training": {"early_stopping_patience": 12}},', '"training": {}},'),
    ("constant_lr parent hash is the delivered configuration", S, '"parent_config": "CONSTANT_LR_PARENT_CONFIG", "parent_sha256": "PATIENCE12_CONFIG_SHA256",',
     '"parent_config": "CONSTANT_LR_PARENT_CONFIG", "parent_sha256": "CONFIG_SHA256",'),
    ("patience12 agreed precisions count changed", S, '("agreed_precisions", 2)', '("agreed_precisions", 3)'),
    ("constant_lr agreed conditions count changed", S, '("agreed_conditions", 5)', '("agreed_conditions", 4)'),
    ("constant_lr agreed key replaced by the precisions", S, '("agreed_conditions", 5)', '("agreed_precisions", 5)'),
    ("configuration names tied to the parent hashes", S, 'tuple((e["sha256"], e["name"]) for e in AMENDMENTS)',
     'tuple((e["parent_sha256"], e["name"]) for e in AMENDMENTS)'),
    ("constant_lr not a known configuration name", S, 'tuple((e["sha256"], e["name"]) for e in AMENDMENTS)',
     'tuple((e["sha256"], e["name"]) for e in AMENDMENTS[:1])'),
    ("amendment never detected", S, "    amended = AMENDMENT_KEY in cfg ", "    amended = False "),
    ("null amendment treated as no amendment", S, "    amended = AMENDMENT_KEY in cfg ", "    amended = cfg.get(AMENDMENT_KEY) is not None "),
    ("amended training not compared", S, '    if amended and cfg["training"] not in [amended_training(e) for e in ([entry] if entry else AMENDMENTS)]:',
     "    if False:"),
    ("amended training compared with every entry", S, '[amended_training(e) for e in ([entry] if entry else AMENDMENTS)]',
     "[amended_training(e) for e in AMENDMENTS]"),
    ("amended training without the frozen recipe", S, '    return {**FR.FROZEN_TRAINING, **entry["training"]}', '    return dict(entry["training"])'),
    ("amended training without the changes", S, '    return {**FR.FROZEN_TRAINING, **entry["training"]}', "    return dict(FR.FROZEN_TRAINING)"),
    ("amendment entry not chosen by changed_field", S, '    return next((e for e in AMENDMENTS if am.get("changed_field") == e["field"]), None)',
     "    return AMENDMENTS[0]"),
    ("amendment entry for a non-object", S, "    if not isinstance(am, dict):\n        return None\n    return next(", "    if False:\n        return None\n    return next("),
    ("amendment block not checked", S, "    if amended:\n        am = cfg[AMENDMENT_KEY]", "    if False:\n        am = cfg[AMENDMENT_KEY]"),
    ("non-object amendment accepted", S, '        if not isinstance(am, dict):\n            fail(f"configuration: amendment needs exactly',
     '        if False:\n            fail(f"configuration: amendment needs exactly'),
    ("unknown changed_field accepted", S, "        if entry is None:\n            fail(", "        if False:\n            fail("),
    ("amendment keys not closed", S, "        _keys(am, list(AMENDMENT_COMMON_KEYS) + [agreed], AMENDMENT_KEY)", "        pass"),
    ("extra key inside amendment accepted", S, "        _keys(am, list(AMENDMENT_COMMON_KEYS) + [agreed], AMENDMENT_KEY)",
     "        if not set(AMENDMENT_COMMON_KEYS) | {agreed} <= set(am):\n            fail(\"configuration: amendment needs exactly\")"),
    ("amendment values not compared", S, '        if (am["from"], am["to"]) != (entry["from"], entry["to"]) or type(am["from"]) is not type(entry["from"]) \\',
     "        if False \\"),
    ("amendment from type not strict", S, ' or type(am["from"]) is not type(entry["from"]) \\\n                or type(am["to"]) is not type(entry["to"]):',
     " \\\n                or type(am[\"to\"]) is not type(entry[\"to\"]):"),
    ("amendment to type not strict", S, ' \\\n                or type(am["to"]) is not type(entry["to"]):', ":"),
    ("amendment parent hash format not checked", S, '        if not _hex(am["parent_config_sha256"]):', "        if False:"),
    ("amendment texts not checked", S, '        for k in ("authorisation", "note", "time_estimate", "shared_history_check"):', '        for k in ("authorisation",):'),
    ("agreed count not checked", S, '        if not isinstance(am[agreed], list) or len(am[agreed]) != n_agreed:',
     '        if not isinstance(am[agreed], list) or len(am[agreed]) < 1:'),
    ("agreed items not required to be a list", S, "if not isinstance(am[agreed], list) or len(", "if len("),
    ("one more agreed item accepted", S, "len(am[agreed]) != n_agreed:", "len(am[agreed]) not in (n_agreed, n_agreed + 1):"),
    ("agreed texts not checked", S, '            _text(v, f"amendment.{agreed}[{i}]")', "            pass"),
    ("configuration name for every hash", S, "    return next((name for const, name in CONFIGURATION_NAMES if cfg_sha == globals()[const]), None)",
     '    return next((name for const, name in CONFIGURATION_NAMES), None)'),
    ("field value without the missing-key guard", S, "        if not isinstance(v, dict) or k not in v:\n            return None", "        if False:\n            return None"),
    ("difference: types not strict", S, '    if type(parent) is not type(child):\n        return path or "<root>"', '    if False:\n        return path or "<root>"'),
    ("difference: extra child keys ignored", S, "        for k in list(parent) + [k for k in child if k not in parent]:", "        for k in list(parent):"),
    ("difference: allowed paths not skipped", S, "            if sub in allowed:\n                continue", "            if False:\n                continue"),
    ("difference: missing keys not reported", S, "            if k not in parent or k not in child:\n                return sub", "            if k not in parent:\n                return sub"),
    ("difference: nested dict difference ignored", S, "            if (d := config_difference(parent[k], child[k], allowed, sub)) is not None:\n                return d",
     "            if (d := config_difference(parent[k], child[k], allowed, sub)) is not None:\n                continue"),
    ("difference: list length not compared", S, "        for i in range(max(len(parent), len(child))):", "        for i in range(min(len(parent), len(child))):"),
    ("difference: nested list difference ignored", S, "            if (d := config_difference(parent[i], child[i], allowed, sub)) is not None:\n                return d",
     "            if (d := config_difference(parent[i], child[i], allowed, sub)) is not None:\n                continue"),
    ("difference: scalar values not compared", S, '    return None if parent == child else (path or "<root>")', "    return None"),
    ("difference: path without the parent", S, '            sub = f"{path}.{k}" if path else str(k)', "            sub = str(k)"),
    ("difference: allowed paths matched as prefixes", S, "            if sub in allowed:\n", "            if any(sub.startswith(x) for x in allowed):\n"),
    ("difference: allowed paths matched by key name", S, "            if sub in allowed:\n", "            if sub in allowed or k in allowed:\n"),
    ("amendment check run without an amendment", S, "    if am is None:\n        return None", "    if False:\n        return None"),
    ("amended hash not pinned", S, "    if cfg_sha != pinned:", "    if False:"),
    ("parent pinned hash taken from the configuration itself", S, 'pinned, parent_pinned = globals()[e["sha256"]], globals()[e["parent_sha256"]]',
     'pinned, parent_pinned = globals()[e["sha256"]], globals()[e["sha256"]]'),
    ("declared parent hash not compared", S, '    if am["parent_config_sha256"] != parent_pinned:', "    if False:"),
    ("parent file hash not compared", S, "    if ph != parent_pinned:", "    if False:"),
    ("parent read from the configuration path of the entry", S, '    pp = ROOT / globals()[e["parent_config"]]', '    pp = ROOT / globals()[e["config"]]'),
    ("parent fields not compared", S, "    if (d := config_difference(parent, cfg, amendment_allowed(e))) is not None:", "    if False:"),
    ("amendment key not allowed to differ", S, '    return (entry["field"], AMENDMENT_KEY)', '    return (entry["field"],)'),
    ("every field allowed to differ", S, '    return (entry["field"], AMENDMENT_KEY)', '    return (entry["field"], AMENDMENT_KEY, "task", "budget")'),
    ("changed field not allowed to differ", S, '    return (entry["field"], AMENDMENT_KEY)', "    return (AMENDMENT_KEY,)"),
    ("parent value not required", S, '    if vals != (e["from"], e["to"]) or any(type(v) is not type(e["to"]) for v in vals):', "    if False:"),
    ("changed value type not strict", S, ' or any(type(v) is not type(e["to"]) for v in vals):', ":"),
    ("check-inputs reads the inputs before the amendment check", S, "    amendment = check_amendment(cfg, cfg_sha)\n    inp = load_inputs(cfg, base)",
     "    inp = load_inputs(cfg, base)\n    amendment = check_amendment(cfg, cfg_sha)"),
    ("amendment not checked by check-inputs", S, "    amendment = check_amendment(cfg, cfg_sha)\n    inp = load_inputs(cfg, base)", "    amendment = None\n    inp = load_inputs(cfg, base)"),
    ("amendment line not printed", S, "    if amendment:\n        e = amendment_entry(amendment)\n        print(f\"AMENDMENT:",
     "    if False:\n        e = amendment_entry(amendment)\n        print(f\"AMENDMENT:"),
    ("amendment not checked by run", S, "    amendment = check_amendment(cfg, cfg_sha)\n    state = repo_state(ROOT)", "    amendment = None\n    state = repo_state(ROOT)"),
    ("delivered only for the parent configuration", S, '"delivered_configuration": configuration_name(cfg_sha) is not None,', '"delivered_configuration": cfg_sha == CONFIG_SHA256,'),
    ("configuration name not recorded", S, '"configuration_name": configuration_name(cfg_sha),', '"configuration_name": None,'),
    ("amendment not recorded", S, '    if amendment:\n        head["amendment"] = amendment', '    if False:\n        head["amendment"] = amendment'),
    ("amendment not described in aggregate.md", S, 'if record.get("amendment") else "")', 'if False else "")'),
    ("amendment text from the first entry", S, '    e = amendment_entry(am)\n    return f"{e', '    e = AMENDMENTS[0]\n    return f"{e'),
    ("constant learning rate not described in aggregate.md", S, 'if cfg["training"]["scheduler"] == FR.SCHEDULER_CONSTANT else', "if False else"),
    ("authorisation message without the agreement of 7 November", S, "and the verification run with early_stopping_patience 12 on 7 November 2026", "and later"),
    ("authorisation message without the agreement of 10 November", S, '"constant learning rate on 10 November 2026)"', '"later)"'),
    ("configuration name not printed", S, "              f\"{head['configuration_name']}), cap \"", "              f\"), cap \""),
    ("trainer revision never accepted", S, "    revision = shared[SHARED_CODE[0]][0] == TRAINER_REVISION_SHA256 != shared[SHARED_CODE[0]][1]", "    revision = False"),
    ("any trainer accepted", S, "    revision = shared[SHARED_CODE[0]][0] == TRAINER_REVISION_SHA256 != shared[SHARED_CODE[0]][1]", "    revision = True"),
    ("other shared files accepted with the trainer revision", S, "if here != there and not (f == SHARED_CODE[0] and revision)]", "if here != there and not revision]"),
    ("trainer revision not recorded", S, '"trainer": "revision with the operative scheduler (TRAINER_REVISION_SHA256)" if revision',
     '"trainer": "the trainer of the folds run" if revision'),
    ("error not written in the run log", S, '        tail = f"ERROR: {reason}\\nFAIL outputs: {outputs}\\nRESULT: FAIL (run)\\n"', '        tail = ""'),
]
# mutazione di controllo: solo un commento; deve SOPRAVVIVERE (prova che la suite rileva il comportamento, non il testo)
CONTROL = [("control: comment only", S, "def verdict_text(v: dict) -> str:\n", "def verdict_text(v: dict) -> str:  # control mutant: comment only\n")]
# controlli dichiarati ridondanti, che devono SOPRAVVIVERE:
REDUNDANT = [
    # i soggetti dei bracci stanno in S12 (soggetti di B), e B = la parte 01-02 del fitting dello schema, i cui soggetti sono
    # disgiunti dal dev e dal test (controllato da nuaa_folds_run.load_fold_manifests): la disgiunzione segue dagli altri controlli
    ("declared redundant (must survive): arm subjects disjoint from the dev and the test", S,
     '            if not subj[arm] <= fit_subj or subj[arm] & (test_subj | dev_subj) or subj[arm] & subj["dev"]:', "            if False:"),
    # dev comune = parte 01-02 del dev dello schema, disgiunto dal test per costruzione dello schema
    ("declared redundant (must survive): common dev disjoint from the test", S, '        if subj["dev"] & test_subj:', "        if False:"),
    # guardia difensiva: dopo discard_model nessun file di un modello interrotto resta in restricted/, quindi la guardia
    # non scatta mai quando discard_model funziona (la sua mutazione è uccisa proprio da questa guardia)
    ("declared redundant (must survive): extra restricted files refused", S,
     '        if extra:\n            fail(f"restricted folder holds files', '        if False:\n            fail(f"restricted folder holds files'),
]


def copy_tree(dst: Path) -> Path:
    """Copia del codice e, se c'è, della configurazione consegnata (delivery/sessionablationrun/ nel repository di
    sviluppo e nel pacchetto livenesslab/, oppure accanto alla radice del pacchetto), al percorso predefinito dello script."""
    code = dst / "code"
    for rel in COPY:
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, code / rel)
    for c in (ROOT.joinpath(*CONFIG_DIR), ROOT.parent):
        if (c / CONFIG).is_file():
            code.joinpath(*CONFIG_DIR).mkdir(parents=True, exist_ok=True)
            shutil.copy2(c / CONFIG, code.joinpath(*CONFIG_DIR, CONFIG))
            break
    for amended in (PATIENCE12, CONSTANT_LR):
        if ROOT.joinpath(*amended).is_file():
            code.joinpath(*amended[:-1]).mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT.joinpath(*amended), code.joinpath(*amended))
    return code


def run_tests(code: Path, tmp: Path) -> tuple:
    """La suite sulla copia; SESSABLRUN_INPUT_ROOT = la copia (solo codice e configurazione): nessun dato reale letto."""
    tmp.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "SESSABLRUN_INPUT_ROOT": str(code), "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp)}
    r = subprocess.run([sys.executable, "-I", TESTS], cwd=code, env=env, capture_output=True, text=True, timeout=3600)
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("tests run")), "no test summary")
    skips = tuple(sorted(l for l in r.stdout.splitlines() if l.startswith("skipped ")))
    return r.returncode, line, skips


def undeclared_skips(skips) -> list:
    """Salti il cui motivo non è un ingresso mancante dichiarato (riga «skipped <test>: missing input: <motivo>»)."""
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
    ap = argparse.ArgumentParser(description="Controllo delle mutazioni di nuaa_session_ablation_run.py (offline)")
    ap.add_argument("--list", action="store_true", help="elenca le mutazioni e verifica i modelli, senza eseguire i test")
    ap.add_argument("--jobs", type=int, default=1, help=f"mutazioni eseguite in parallelo (default 1, al massimo {MAX_JOBS})")
    a = ap.parse_args()
    if not 1 <= a.jobs <= MAX_JOBS:
        print(f"ERROR: --jobs must be between 1 and {MAX_JOBS} (16 GB machine: one heavy process at a time)", file=sys.stderr)
        print("RESULT: FAIL (refused)")
        return 1
    problems = [f"{n}: the pattern occurs {k} times in {t} (expected exactly once)"
                for n, t, old, _ in MUTANTS + CONTROL + REDUNDANT if (k := (ROOT / t).read_text(encoding="utf-8").count(old)) != 1]
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
    with tempfile.TemporaryDirectory(prefix="sessablrun-mutations-") as tmp:
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
                print(f"ERROR: skip that is not a declared missing input: {s}", file=sys.stderr)
            print("RESULT: FAIL (the unmutated copy skips tests for a reason that is not a declared missing input)")
            return 1
        if base_skips:
            print(f"unmutated copy: {len(base_skips)} tests skipped, all for a declared missing input; mutations are counted "
                  "killed only by failures or errors of the tests that run", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
            futs = [ex.submit(one, tmp, i, n, t, old, new, base_skips) for i, (n, t, old, new) in enumerate(MUTANTS)]
            cfuts = [ex.submit(one, tmp, len(MUTANTS) + i, n, t, old, new, base_skips) for i, (n, t, old, new) in enumerate(CONTROL + REDUNDANT)]
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
