"""
Controllo delle mutazioni di scripts/nuaa_patience_verify.py (confronto del run di verifica con pazienza 12 con il run
genitore), offline e deterministico.

Copia il codice (scripts/nuaa_patience_verify.py, scripts/check_nuaa_patience_verify.py, i moduli condivisi importati
e, se ci sono, la configurazione delivery/patienceverify/nuaa_patience_verify.json e quella del pacchetto precedente
delivery/degenerate/nuaa_degenerate_trainings.json, letta da un test sui criteri, e quella dello schema 2
delivery/constlr/nuaa_constant_lr_verify.json) in una cartella temporanea, applica una
mutazione dichiarata alla volta (coppia cerca/sostituisci che deve comparire esattamente una volta: se un modello non si
trova più, il controllo si ferma con un errore) ed esegue scripts/check_nuaa_patience_verify.py sulla copia mutata
SENZA --real e con PATVER_INPUT_ROOT su una cartella vuota: i mutanti non leggono mai i dati veri (regola della macchina
da 16 GB: un processo pesante alla volta). Una mutazione è rilevata solo se la suite arriva al riepilogo («tests run»)
con almeno un fallimento o errore, cioè dal comportamento; una suite che non parte la rende non valida. Una mutazione
di controllo (solo un commento) deve sopravvivere, e così i controlli dichiarati ridondanti.

Test saltati: la copia non mutata deve passare e può saltare test solo per un motivo dichiarato (MISSING_INPUT di
scripts/check_nuaa_patience_verify.py); i salti sono riportati. Una mutazione conta come rilevata solo per fallimenti o
errori, mai per un salto. Codice 1 se una mutazione sopravvive o non è valida, se un salto non è dichiarato o se un
modello non si trova.

  .venv/bin/python -I scripts/check_nuaa_patience_verify_mutations.py                 # tutte, una alla volta
  .venv/bin/python -I scripts/check_nuaa_patience_verify_mutations.py --list          # solo l'elenco, con la verifica dei modelli
  .venv/bin/python -I scripts/check_nuaa_patience_verify_mutations.py --only "<nome>" [--only "<nome>"]   # solo quelle
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
from check_nuaa_patience_verify import MISSING_INPUT  # noqa: E402

TESTS = "scripts/check_nuaa_patience_verify.py"
S = "scripts/nuaa_patience_verify.py"
COPY = (S, TESTS, "scripts/m2_writers.py", "scripts/path_guard.py", "scripts/package_rules.py")
CONFIGS = ("delivery/patienceverify/nuaa_patience_verify.json", "delivery/degenerate/nuaa_degenerate_trainings.json",
           "delivery/constlr/nuaa_constant_lr_verify.json")
MAX_JOBS = 3
# (nome, file mutato, testo cercato, testo sostitutivo)
MUTANTS = [
    # regole
    ("early stopping: patience + 1", S, "if wait >= patience and e > 0:", "if wait >= patience + 1 and e > 0:"),
    ("early stopping: an equal value counts as an improvement", S,
     'better = (lambda a, b: a < b) if monitor == "val_loss"', 'better = (lambda a, b: a <= b) if monitor == "val_loss"'),
    ("early stopping: short histories treated as complete", S,
     "if len(vals) == n_max and best_ep is not None:", "if best_ep is not None:"),
    ("degenerate: restored epoch up to 3", S, 'return restored in d["restored_epochs"] and abs(', "return restored <= 3 and abs("),
    ("degenerate: val_acc_at_best not required", S, 'and abs(val_acc - d["val_acc_at_best"]) < d["tolerance"]', "and True"),
    ("saturation: tolerance widened", S,
     'return abs(v - crit["saturation"]["val_accuracy"]) < crit["saturation"]["tolerance"]',
     'return abs(v - crit["saturation"]["val_accuracy"]) <= crit["saturation"]["tolerance"] + 0.2'),
    ("recovery: val_loss condition dropped", S, "if va >= acc and vl < loss:", "if va >= acc:"),
    ("recovery: > instead of >= on val_accuracy", S, "if va >= acc and vl < loss:", "if va > acc and vl < loss:"),
    ("recovery: <= instead of < on val_loss", S, "if va >= acc and vl < loss:", "if va >= acc and vl <= loss:"),
    ("dev: a tie at 0.5 counted as attack", S,
     '"acc_05": sum(1 for s, y in zip(scores, labels) if (s > DECISION) == (y == 1))',
     '"acc_05": sum(1 for s, y in zip(scores, labels) if (s >= DECISION) == (y == 1))'),
    # (a) storie comuni
    ("shared: a tolerance of 1e-6", S, "            if hn[k][i] != hp[k][i] and (at is None or d > mx):",
     "            if d > 1e-6 and (at is None or d > mx):"),
    ("shared: only val_loss compared", S, "    for k in HISTORY_KEYS:\n        mx, at = 0.0, None",
     '    for k in ("val_loss",):\n        mx, at = 0.0, None'),
    ("shared: lr not compared", S, "    for k in HISTORY_KEYS:\n        mx, at = 0.0, None",
     '    for k in [x for x in HISTORY_KEYS if x != "lr"]:\n        mx, at = 0.0, None'),
    ("shared: a shorter new history counted as identical", S,
     '    out["identical"] = out["complete"] and all(v["epoch"] is None for v in out["fields"].values())',
     '    out["identical"] = all(v["epoch"] is None for v in out["fields"].values())'),
    ("shared: compared on the first epoch only", S, "        for i in range(min(n_shared, len(hn[k]))):", "        for i in range(1):"),
    ("shared: comparison on the parent epochs minus one", S,
     'shared = {n: shared_history(par.models[n]["h"], new.models[n]["h"], span[n]) for n in N}',
     'shared = {n: shared_history(par.models[n]["h"], new.models[n]["h"], span[n] - 1) for n in N}'),
    ("shared: validity not required for a verdict", S, "    elif not valid:\n        verdict = VERDICT_NOT_VALID",
     "    elif False:\n        verdict = VERDICT_NOT_VALID"),
    ("shared: largest difference epoch off by one", S, "                mx, at = d, i + 1", "                mx, at = d, i"),
    # (b) verdetto
    ("verdict: incomplete run gets a verdict", S, '    if new.status != "PASS":\n        verdict = VERDICT_NONE',
     '    if new.status == "FAIL":\n        verdict = VERDICT_NONE'),
    ("verdict: recovery of the ten not required", S, "    ten_ok = not ten_not_rec and all(n in N for n in ten)",
     "    ten_ok = all(n in N for n in ten)"),
    ("verdict: degenerate models not required", S, "    elif ten_ok and not deg_new:", "    elif ten_ok:"),
    ("verdict: extended conclusion dropped", S, "    if verdict in (VERDICT_POSITIVE, VERDICT_NEGATIVE) and out_not_rec:",
     "    if False:"),
    ("verdict: models outside the ten without recovery not listed", S,
     'out_not_rec = [label(n) for n in names if n in N and n not in ten and not N[n]["recovered"]]', "out_not_rec = []"),
    ("verdict: the ten read from the parent run", S,
     'ten_not_rec = [label(n) for n in ten if n in N and not N[n]["recovered"]]', 'ten_not_rec = [label(n) for n in ten if not P[n]["recovered"]]'),
    ("verdict: negative conclusion without the declared confounder", S,
     '"degenerate among the 36: {\', \'.join(deg_new) or \'none\'}). This does not rule out a role of early stopping "',
     '"degenerate among the 36: {\', \'.join(deg_new) or \'none\'}). "'),
    # (c) secondari
    ("secondary: AUROC replaced by the accuracy", S, '"dev_auroc": m["dev"]["auroc"],', '"dev_auroc": m["dev"]["acc_05"],'),
    ("secondary: epochs at 0.5 counted on val_loss", S,
     '"epochs_at_half": sum(1 for v in h["val_accuracy"] if is_half(v, crit))', '"epochs_at_half": sum(1 for v in h["val_loss"] if is_half(v, crit))'),
    ("secondary: lr of the first epoch", S, '"lr_last": h["lr"][-1],', '"lr_last": h["lr"][0],'),
    ("curves: shared flag off by one", S, 'int(i + 1 <= T["rows_by_name"][n]["shared_epochs"])',
     'int(i + 1 < T["rows_by_name"][n]["shared_epochs"])'),
    ("curves: recovery flag without the val_loss", S,
     'int(h["val_accuracy"][i] >= rv["val_accuracy_min"] and h["val_loss"][i] < rv["val_loss_max"])',
     'int(h["val_accuracy"][i] >= rv["val_accuracy_min"])'),
    # (d) test dal pubblico
    ("test: reported for an incomplete run", S, '    if new.status == "PASS" and new.test is not None:', "    if new.test is not None:"),
    ("test: parent and new swapped", S, '"primary": [{"metric": k, "parent": par.test["primary"][k], "new": new.test["primary"][k]}',
     '"primary": [{"metric": k, "parent": new.test["primary"][k], "new": par.test["primary"][k]}'),
    ("test: median not read", S, 'out["rows"][(c, s, m)] = {**verdict(v, f"comparisons.{c} stratum {s} metric {m}"), "median": med}',
     'out["rows"][(c, s, m)] = {**verdict(v, f"comparisons.{c} stratum {s} metric {m}"), "median": None}'),
    ("test: verdict values not checked", S,
     '        if not isinstance(v, dict) or v.get("verdict") not in TEST_VERDICTS or not isinstance(v.get("reason"), str):',
     '        if not isinstance(v, dict):'),
    ("test: direction not checked", S, '        if (v["verdict"] == "consistent") != isinstance(v.get("direction"), str):', "        if False:"),
    ("test: strata not checked", S,
     "    if not isinstance(strata, list) or not strata or not all(isinstance(s, str) for s in strata) or len(set(strata)) != len(strata):",
     "    if not isinstance(strata, list):"),
    ("test: primary verdict keys not checked", S, "    if not isinstance(pv, dict) or sorted(pv) != sorted(PRIMARY_METRICS):",
     "    if not isinstance(pv, dict):"),
    ("test: minuend and subtrahend not checked", S,
     '        if not isinstance(cc, dict) or cc.get("minuend") != mi or cc.get("subtrahend") != su:', "        if not isinstance(cc, dict):"),
    ("test: a non-numeric median accepted", S, "                if med is not None and not _number(med):", "                if False:"),
    ("test: aggregate not compared with the SUMS", S, '        if sha256_bytes(ab) != plisted["aggregate.json"]:', "        if False:"),
    ("test: aggregate identity not compared", S, '            if agg.get(k) != st.get(k):', "            if False:"),
    ("test: changed verdicts counted wrongly", S,
     'D["verdicts_changed"] = sum(1 for r in D["rows"] if r["parent"] is None or r["parent"]["verdict"] != r["new"]["verdict"])',
     'D["verdicts_changed"] = 0'),
    # ancore e status
    ("anchors: status.json not compared", S, '    if sha256_bytes(sb) != spec["status_sha256"]:', "    if False:"),
    ("anchors: restricted SHA256SUMS not compared", S,
     '    if sha256_bytes(rb) != spec["restricted_sums_sha256"]:\n        fail(f"{g} restricted', '    if False:\n        fail(f"{g} restricted'),
    ("anchors: restricted anchor of the status not compared", S,
     '    if st.get("restricted_sums_sha256") != spec["restricted_sums_sha256"]:', "    if False:"),
    ("anchors: public anchor of the status not compared", S,
     '    if has_public and st.get("public_sums_sha256") != spec["public_sums_sha256"]:', "    if False:"),
    ("anchors: public SHA256SUMS file not compared", S,
     '        if sha256_bytes(pb) != spec["public_sums_sha256"]:', "        if False:"),
    ("anchors: 'none' accepted for a run with a public folder", S,
     '    if has_public != (spec["public_sums_sha256"] != NO_PUBLIC):', "    if False:"),
    ("anchors: listed files not compared with the SHA256SUMS", S, "    if sha256_bytes(b) != listed[fn]:", "    if False:"),
    ("anchors: null anchors accepted by run", S,
     '    if not anchors_filled(cfg):\n        fail("the anchors of the verification run are null (to be filled after the run): run refused")',
     '    if False:\n        fail("the anchors of the verification run are null (to be filled after the run): run refused")'),
    ("anchors: partial null anchors accepted", S, "    if any(filled) and not all(filled):", "    if False:"),
    ("anchors: malformed anchors accepted", S,
     '        if not _hex(ver["status_sha256"]) or not _hex(ver["restricted_sums_sha256"]) or \\',
     '        if False and not _hex(ver["restricted_sums_sha256"]) or \\'),
    ("status: any status of the parent accepted", S, '    allowed = ("PASS",) if g == "parent_run" else RUN_STATUSES',
     "    allowed = RUN_STATUSES"),
    ("status: identity not compared", S, '    for k in ("run_id", "commit", "config_sha256"):\n        if st.get(k) != spec[k]:',
     '    for k in ("run_id", "commit", "config_sha256"):\n        if False:'),
    ("status: dirty tree accepted", S, '    if st.get("dirty") not in (None, False):', "    if False:"),
    ("status: configuration_name not compared", S, '        if st.get("configuration_name") != spec["configuration_name"]:', "        if False:"),
    ("status: amendment not compared", S,
     '        if not isinstance(am, dict) or any(am.get(k) != v for k, v in spec["amendment"].items()):', "        if False:"),
    ("status: models split not checked", S, "            or sorted(completed + missing) != sorted(names):", "            or False:"),
    ("status: INCOMPLETE without missing accepted", S, '    if status == "INCOMPLETE" and not missing:', "    if False:"),
    ("status: PASS with missing models accepted", S, '    if status == "PASS" and missing:', "    if False:"),
    ("status: cards not compared with the complete models", S, "    if cards != sorted(completed):", "    if not set(cards) <= set(completed):"),
    # schede, soglie, dev
    ("card: patience of the run not compared", S,
     '    want = {"max_epochs": rc["max_epochs"], "early_stopping_patience": patience, "batch": rc["batch"], "lr": rc["lr"]}',
     '    want = {"max_epochs": rc["max_epochs"], "batch": rc["batch"], "lr": rc["lr"]}'),
    ("card: history length not checked", S, "if not isinstance(h[k], list) or len(h[k]) != n:", "if not isinstance(h[k], list):"),
    ("card: restored epoch not compared with the argmin", S, "    if not is_int(r) or r != argmin:", "    if not is_int(r):"),
    ("card: replay of the stop not compared", S, 'if sim is None or sim["stop"] != n or sim["restored"] != r:', "if False:"),
    ("card: restored_by not compared with the replay", S, '    if c["restored_by"] != RESTORED_BY[sim["early"]]:', "    if False:"),
    ("dev: accuracy at 0.5 not compared with val_acc_at_best", S,
     'if abs(d["acc_05"] - card["val_acc_at_best"]) > cfg["criteria"]["checkpoint_agreement_tolerance"]:', "if False:"),
    ("dev: prediction_05 not compared with the rule", S, '        if r["prediction_05"] != ("1" if s > DECISION else "0"):', "        if False:"),
    ("dev: models of a fold with different devs accepted", S, "        if dev_of_fold.setdefault(fold, keys) != keys:", "        if False:"),
    ("dev: the dev of the parent not compared", S, '        if m["keys"] != p["keys"]:', "        if False:"),
    ("fitting: fitting set of the parent not compared", S,
     '        if m["card"]["fitting_keys_sha256"] != p["card"]["fitting_keys_sha256"] or m["card"]["n_fitting"] != p["card"]["n_fitting"]:',
     "        if False:"),
    ("parent lists: the ten not compared with the parent", S, '    elif deg != cfg["original_degenerate"]:', "    elif False:"),
    ("parent lists: models without recovery not compared", S, '    if nr != cfg["models_without_recovery_parent"]:', "    if False:"),
    ("inputs: symbolic link accepted", S, "    if stat.S_ISLNK(st.st_mode):\n        fail(", "    if False:\n        fail("),
    # schede, soglie e dev: controlli riga per riga
    ('card: closed key set not checked', S, '    if sorted(c) != keys:', '    if False:'),
    ('card: best_val_loss not compared with the history', S, '    if c["best_val_loss"] != h["val_loss"][r - 1] or c["val_acc_at_best"] != h["val_accuracy"][r - 1]:', '    if False:'),
    ('card: weights not compared with the SHA256SUMS', S, '    if c["weights_file"] != f"{name}_LivenessNet.keras" or not _hex(c["weights_sha256"]) or listed.get(c["weights_file"]) != c["weights_sha256"]:', '    if False:'),
    ('dev: dev_keys_sha256 of the card not compared', S, '    if sha256_bytes("\\n".join(k[0] for k in keys).encode()) != card["dev_keys_sha256"]:', '    if False:'),
    ('threshold: checkpoint not compared with the card', S, '    if thr.get("checkpoint_sha256") != card["weights_sha256"]:', '    if False:'),
    ('threshold: frozen_at order not checked', S, '    if not saved <= frozen <= tested:', '    if False:'),
    ('dev: dev_scores_sha256 of the threshold not compared', S, '    if sha256_bytes("\\n".join(repr(float(r["score_attack"])) for r in rows).encode()) != thr["dev_scores_sha256"]:', '    if False:'),
    ('dev: checkpoint of the rows not compared', S, '        if r["checkpoint_sha256"] != card["weights_sha256"]:', '        if False:'),
    ('dev: threshold of the rows not compared', S, '        if r["threshold_dev"] != repr(value) or r["threshold_sha256"] != thr["sha256"]:', '        if False:'),
    ('dev: prediction_dev not compared with the rule', S, '        if r["prediction_dev"] != ("1" if s > value else "0"):', '        if False:'),
    ('dev: balance not checked', S, '    if labels.count(0) != labels.count(1):', '    if False:'),
    ('verdict: positive conclusion with a model outside the ten without the declared confounder', S, '        if verdict == VERDICT_POSITIVE:     # NOTE', '        if False:     # NOTE'),
    ('verdict: positive conclusion without the stability caveat', S, '"positive outcome does not by itself show that every model is stable.")', '"positive outcome.")'),
    # configurazione
    ("config: shared difference other than 0 accepted", S,
     '    if cr["shared_history_difference"] != 0 or not is_int(cr["shared_history_difference"]):', "    if False:"),
    ("config: ten not checked", S,
     "    if not isinstance(ten, list) or len(ten) != 10 or len(set(ten)) != 10 or not set(ten) <= set(names) or ten != sorted(ten):",
     "    if not isinstance(ten, list):"),
    ("config: amendment values not checked", S,
     '    if (am["changed_field"], am["from"], am["to"]) != (AMENDMENT["changed_field"], AMENDMENT["from"], AMENDMENT["to"]) or \\',
     "    if False or \\"),
    ("config: amendment types not checked", S, '            not is_int(am["from"]) or not is_int(am["to"]):', "            False:"),
    ("config: parent of the amendment not checked", S,
     '    if am["parent_config_sha256"] != par["config_sha256"]:\n        fail("configuration: inputs.verification_run.amendment',
     '    if False:\n        fail("configuration: inputs.verification_run.amendment'),
    ("config: declaration flags not checked", S, "        if dec[k] is not True:", "        if False:"),
    # pubblico e run
    ("public: subject identifiers not scanned", S, "    for s in sorted(inp.subjects):", "    for s in []:"),
    ("public: figure labels not scanned", S, '    scan["figure labels"] = json.dumps(fig_data)', "    pass"),
    ("public: markers not scanned in the Markdown", S, "    if markdown and pr.MARKERS.search(text):", "    if False:"),
    ("public: IPv4 addresses not scanned", S,
     '    if any(ip not in pr.IP_ALLOW for ip in (m.group(0) for m in pr.IPV4.finditer(text))):', "    if False:"),
    ("public: the subject id value printed in the reason", S, '            bad.append("subject id")', '            bad.append(f"subject id {s}")'),
    ("public: figure skip not declared (figures required)", S,
     '        return {}, f"skipped: missing Python package {e.name!r}"', "        raise"),
    ("public: SHA256SUMS without the figures", S,
     '    public["SHA256SUMS"] = sums_text(public)', '    public["SHA256SUMS"] = sums_text({k: v for k, v in public.items() if isinstance(v, str)})'),
    ("public: a date in the aggregate", S, '           "environment": environment(), "code_sha256": code_sha256()}',
     '           "environment": environment(), "code_sha256": code_sha256(), "written": time.time()}'),
    ("run: a failure leaves the outputs", S, "        left = invalidate_outputs(run)\n        if log is not None:",
     "        left = []\n        if log is not None:"),
    ("run: a run folder with status.json accepted", S, "    if (run / STATUS).exists() or (run / STATUS).is_symlink():", "    if False:"),
    ("run: foreign entries in the run folder accepted", S, "                if not any(mw.owned_name(e, n) for n in names):", "                if False:"),
    ("run id: dirty tree without -dev", S, '+ ("-dev" if state["dirty"] else "")', '+ ""'),
    ("run: authorisation not required", S, "    if not a.i_am_authorized:", "    if False:"),
    # ingressi: lettura dei file, JSON, CSV, SHA256SUMS, tempi
    ("inputs: a missing file not reported", S,
     "    except OSError:\n        fail(f\"{what} {Path(p).name} not found\")",
     "    except OSError:\n        raise"),
    ("inputs: a folder accepted as a file", S, "    if not stat.S_ISREG(st.st_mode):", "    if False:"),
    ("inputs: a file not listed in the SHA256SUMS not reported", S, "    if fn not in listed:", "    if False:"),
    ("json: invalid JSON not reported", S,
     "    except (UnicodeDecodeError, json.JSONDecodeError) as e:\n"
     "        fail(f\"{what} is not valid JSON: {e}\")",
     "    except (UnicodeDecodeError, json.JSONDecodeError) as e:\n        raise"),
    ("json: a non-object accepted", S,
     "    if not isinstance(doc, dict):\n        fail(f\"{what} is not a JSON object\")",
     "    if False:\n        fail(f\"{what} is not a JSON object\")"),
    ("csv: a file that is not UTF-8 not reported", S,
     "    except UnicodeDecodeError:\n        fail(f\"{what} is not UTF-8\")",
     "    except UnicodeDecodeError:\n        raise"),
    ("csv: columns not compared", S, "    if r.fieldnames != list(cols):", "    if False:"),
    ("csv: extra fields accepted", S, "None in x or None in x.values()", "None in x.values()"),
    ("csv: missing fields accepted", S, "None in x or None in x.values()", "None in x"),
    ("sums: a malformed line accepted", S,
     "        if not m:\n            fail(f\"{what}: malformed line",
     "        if False:\n            fail(f\"{what}: malformed line"),
    ("sums: a file listed twice accepted", S, "        if m.group(2) in out:", "        if False:"),
    ("time: a malformed time not reported", S,
     "    except (TypeError, ValueError):\n        fail(f\"{what}: {v!r} is not a time",
     "    except ZeroDivisionError:\n        fail(f\"{what}: {v!r} is not a time"),
    # configurazione: un controllo alla volta
    ("config: schema not checked", S, "    if cfg.get(\"schema\") not in SCHEMAS:", "    if False:"),
    ("config: unknown or missing top keys accepted", S, "    if sorted(cfg) != sorted(top):", "    if False:"),
    ("config: blank texts accepted", S,
     "    if not isinstance(v, str) or not v.strip():",
     "    if not isinstance(v, str):"),
    ("config: non-text values accepted", S, "    if not isinstance(v, str) or not v.strip():", "    if not v.strip():"),
    ("config: closed key sets not checked", S,
     "    if not isinstance(d, dict) or sorted(d) != sorted(keys):",
     "    if not isinstance(d, dict):"),
    ("config: a list of the keys accepted as an object", S,
     "    if not isinstance(d, dict) or sorted(d) != sorted(keys):",
     "    if sorted(d) != sorted(keys):"),
    ("config: integer lists that are not lists accepted", S,
     "    if not isinstance(v, list) or not v or",
     "    if not v or"),
    ("config: empty integer lists accepted", S,
     "isinstance(v, list) or not v or not all",
     "isinstance(v, list) or not all"),
    ("config: integer lists with non-integers accepted", S, " or not all(is_int(x) for x in v)", ""),
    ("config: integer lists with duplicates accepted", S, " or len(set(v)) != len(v) or \\", " or \\"),
    ("config: integer lists below the minimum accepted", S,
     "            (lo is not None and min(v) < lo):",
     "            False:"),
    ("config: arms not checked", S,
     "    if not isinstance(inputs[\"arms\"], list) or sorted(inputs[\"arms\"]) != sorted(ARM_LABEL)",
     "    if False and sorted(inputs[\"arms\"]) != sorted(ARM_LABEL)"),
    ("config: arms that are not a list accepted", S, "if not isinstance(inputs[\"arms\"], list) or ", "if "),
    ("config: models not compared with arms x folds x seeds", S,
     "    if not is_int(inputs[\"models\"]) or inputs[\"models\"] != n:",
     "    if False:"),
    ("config: a non-integer models accepted", S,
     "    if not is_int(inputs[\"models\"]) or inputs[\"models\"] != n:",
     "    if inputs[\"models\"] != n:"),
    ("config: dev_rows not checked", S,
     "    if not is_int(inputs[\"dev_rows\"]) or inputs[\"dev_rows\"] < 2 or inputs[\"dev_rows\"] % 2:",
     "    if False:"),
    ("config: a non-integer dev_rows accepted", S, "not is_int(inputs[\"dev_rows\"]) or ", ""),
    ("config: a dev_rows below 2 accepted", S, " or inputs[\"dev_rows\"] < 2", ""),
    ("config: an odd dev_rows accepted", S, " or inputs[\"dev_rows\"] % 2:", ":"),
    ("config: run id format not checked", S,
     "        if not re.fullmatch(r\"\\d{8}-SESSABLRUN-[0-9a-f]{7,40}\", spec[\"run_id\"]):",
     "        if False:"),
    ("config: commit not compared with the run id", S,
     "        if not isinstance(spec[\"commit\"], str) or not spec[\"run_id\"].endswith(\"-\" + spec[\"commit\"]):",
     "        if False:"),
    ("config: a non-text commit accepted", S, "if not isinstance(spec[\"commit\"], str) or not spec", "if not spec"),
    ("config: config_sha256 of a run not checked", S,
     "        if not _hex(spec[\"config_sha256\"]):",
     "        if False:"),
    ("config: patience of a run not checked", S,
     "        if not is_int(spec[\"early_stopping_patience\"]) or spec[\"early_stopping_patience\"] < 1:",
     "        if False:"),
    ("config: a patience below 1 accepted", S, " or spec[\"early_stopping_patience\"] < 1:", ":"),
    ("config: anchors of the parent not checked", S, "        if not _hex(par[k]):", "        if False:"),
    ("config: a malformed restricted anchor accepted", S,
     " or not _hex(ver[\"restricted_sums_sha256\"]) or \\",
     " or \\"),
    ("config: a malformed public anchor accepted", S,
     "not (_hex(ver[\"public_sums_sha256\"]) or ver[\"public_sums_sha256\"] == NO_PUBLIC):",
     "False:"),
    ("config: configuration_name not checked", S,
     "    if ver[\"configuration_name\"] != CONFIGURATION_NAME:",
     "    if False:"),
    ("config: patience of the runs not compared with the amendment", S,
     "    if (par[\"early_stopping_patience\"], ver[\"early_stopping_patience\"]) != (am[\"from\"], am[\"to\"]):",
     "    if False:"),
    ("config: two runs with the same run id accepted", S, "par[\"run_id\"] == ver[\"run_id\"] or ", ""),
    ("config: two runs with the same label accepted", S, " or par[\"label\"] == ver[\"label\"]", ""),
    ("config: two runs with the same folder accepted", S, " or par[\"folder\"] == ver[\"folder\"]", ""),
    ("config: recipe batch and max_epochs not checked", S,
     "        if not is_int(rc[k]) or rc[k] < 1:",
     "        if False:"),
    ("config: a non-integer recipe batch accepted", S,
     "        if not is_int(rc[k]) or rc[k] < 1:",
     "        if rc[k] < 1:"),
    ("config: recipe lr not checked", S, "    if not _number(rc[\"lr\"]) or rc[\"lr\"] <= 0:", "    if False:"),
    ("config: a non-finite recipe lr accepted", S,
     "    if not _number(rc[\"lr\"]) or rc[\"lr\"] <= 0:",
     "    if rc[\"lr\"] <= 0:"),
    ("config: min_delta not checked", S, "if rc[\"min_delta\"] != 0 or ", "if "),
    ("config: monitor not checked", S, " or rc[\"monitor\"] != \"val_loss\"", ""),
    ("config: restore_best_weights not checked", S, " or rc[\"restore_best_weights\"] is not True:", ":"),
    ("config: verification patience not compared with max_epochs", S,
     "    if ver[\"early_stopping_patience\"] != rc[\"max_epochs\"]:",
     "    if False:"),
    ("config: degenerate value not checked", S,
     "    if not _number(dg[\"val_acc_at_best\"]) or not 0 <= dg[\"val_acc_at_best\"] <= 1 or not _number(dg[\"tole"
     "rance\"]) or \\",
     "    if False or \\"),
    ("config: degenerate tolerance not checked", S,
     "            not 0 < dg[\"tolerance\"] < 1e-3:",
     "            False:"),
    ("config: saturation not compared with the degenerate criterion", S,
     "    if sa[\"val_accuracy\"] != dg[\"val_acc_at_best\"] or sa[\"tolerance\"] != dg[\"tolerance\"]:",
     "    if False:"),
    ("config: saturation value not compared", S, "if sa[\"val_accuracy\"] != dg[\"val_acc_at_best\"] or ", "if "),
    ("config: recovery thresholds not checked", S, "        if not _number(rv[k]) or rv[k] <= 0:", "        if False:"),
    ("config: checkpoint agreement tolerance not checked", S,
     "    if not _number(cr[\"checkpoint_agreement_tolerance\"]) or not 0 < cr[\"checkpoint_agreement_tolerance\"] "
     "< 1e-3:",
     "    if False:"),
    ("config: a shared difference 0.0 accepted", S,
     "    if cr[\"shared_history_difference\"] != 0 or not is_int(cr[\"shared_history_difference\"]):",
     "    if cr[\"shared_history_difference\"] != 0:"),
    ("config: ten that is not a list accepted", S, "    if not isinstance(ten, list) or len(ten)", "    if len(ten)"),
    ("config: ten of another size accepted", S, " or len(ten) != 10", ""),
    ("config: ten with duplicates accepted", S, " or len(set(ten)) != 10", ""),
    ("config: ten with unknown models accepted", S, " or not set(ten) <= set(names)", ""),
    ("config: ten not sorted accepted", S, " or ten != sorted(ten):", ":"),
    ("config: models without recovery not checked", S,
     "    if not isinstance(nr, list) or len(set(nr)) != len(nr) or not set(nr) <= set(names) or set(nr) & set(ten)"
     " or nr != sorted(nr):",
     "    if not isinstance(nr, list):"),
    ("config: models without recovery that are not a list accepted", S,
     "if not isinstance(nr, list) or len(set(nr))",
     "if len(set(nr))"),
    ("config: models without recovery with duplicates accepted", S, " or len(set(nr)) != len(nr)", ""),
    ("config: unknown models without recovery accepted", S, " or not set(nr) <= set(names)", ""),
    ("config: models without recovery inside the ten accepted", S, " or set(nr) & set(ten)", ""),
    ("config: models without recovery not sorted accepted", S, " or nr != sorted(nr):", ":"),
    ("config: time not checked", S,
     "    if not _number(tm[\"estimate_seconds\"]) or not _number(tm[\"cap_seconds\"]) or not 0 < tm[\"estimate_sec"
     "onds\"] <= tm[\"cap_seconds\"]:",
     "    if False:"),
    ("config: an infinite cap accepted", S, " or not _number(tm[\"cap_seconds\"])", ""),
    ("config: a zero estimate accepted", S,
     "or not 0 < tm[\"estimate_seconds\"]",
     "or not 0 <= tm[\"estimate_seconds\"]"),
    ("config: an estimate equal to the cap refused", S,
     "0 < tm[\"estimate_seconds\"] <= tm[\"cap_seconds\"]:",
     "0 < tm[\"estimate_seconds\"] < tm[\"cap_seconds\"]:"),
    # schede: un controllo alla volta
    ("card: arch not checked", S, "    if c[\"arch\"] != \"LivenessNet\":", "    if False:"),
    ("card: fold not compared with the file name", S, "c[\"fold\"] != fold or ", ""),
    ("card: train_seed not compared with the file name", S, "c[\"train_seed\"] != seed or ", ""),
    ("card: arm not compared with the file name", S, " or c[\"arm\"] != arm:", ":"),
    ("card: a boolean fold or train_seed accepted", S,
     "    if not is_int(c[\"fold\"]) or not is_int(c[\"train_seed\"]):",
     "    if False:"),
    ("card: recipe values not compared", S,
     "        if c[k] != v or isinstance(c[k], bool):",
     "        if isinstance(c[k], bool):"),
    ("card: a boolean equal to the recipe value accepted", S,
     "        if c[k] != v or isinstance(c[k], bool):",
     "        if c[k] != v:"),
    ("card: n_fitting not checked", S, "    if not is_int(c[\"n_fitting\"]) or c[\"n_fitting\"] < 1:", "    if False:"),
    ("card: a non-integer n_fitting accepted", S,
     "    if not is_int(c[\"n_fitting\"]) or c[\"n_fitting\"] < 1:",
     "    if c[\"n_fitting\"] < 1:"),
    ("card: n_dev not checked", S, "    if not is_int(c[\"n_dev\"]) or c[\"n_dev\"] < 2:", "    if False:"),
    ("card: an n_dev below 2 accepted", S,
     "    if not is_int(c[\"n_dev\"]) or c[\"n_dev\"] < 2:",
     "    if not is_int(c[\"n_dev\"]):"),
    ("card: train_seconds not checked", S,
     "    if not _number(c[\"train_seconds\"]) or c[\"train_seconds\"] <= 0:",
     "    if False:"),
    ("card: epochs_run not checked", S, "    if not is_int(n) or not 1 <= n <= rc[\"max_epochs\"]:", "    if False:"),
    ("card: epochs_run above max_epochs accepted", S, " or not 1 <= n <= rc[\"max_epochs\"]:", ":"),
    ("card: history keys not checked", S,
     "    if not isinstance(h, dict) or sorted(h) != sorted(HISTORY_KEYS):",
     "    if not isinstance(h, dict):"),
    ("card: non-finite history values accepted", S,
     "        if not all(_number(v) for v in h[k]):",
     "        if False:"),
    ("card: accuracy range not checked", S,
     "    if not all(0 <= v <= 1 for k in (\"accuracy\", \"val_accuracy\") for v in h[k]):",
     "    if False:"),
    ("card: val_accuracy range not checked", S,
     "for k in (\"accuracy\", \"val_accuracy\") for v in h[k]",
     "for k in (\"accuracy\",) for v in h[k]"),
    ("card: negative losses accepted", S,
     "    if not all(v >= 0 for k in (\"loss\", \"val_loss\") for v in h[k]):",
     "    if False:"),
    ("card: negative val_loss accepted", S,
     "for k in (\"loss\", \"val_loss\") for v in h[k]",
     "for k in (\"loss\",) for v in h[k]"),
    ("card: non-positive lr accepted", S, "    if not all(v > 0 for v in h[\"lr\"]):", "    if False:"),
    ("card: a non-integer restored_epoch accepted", S, "    if not is_int(r) or r != argmin:", "    if r != argmin:"),
    ("card: val_acc_at_best not compared with the history", S,
     " or c[\"val_acc_at_best\"] != h[\"val_accuracy\"][r - 1]:",
     ":"),
    ("card: stop of the replay not compared", S,
     "if sim is None or sim[\"stop\"] != n or sim[\"restored\"] != r:",
     "if sim is None or sim[\"restored\"] != r:"),
    ("card: weights_file not compared with the model name", S,
     "c[\"weights_file\"] != f\"{name}_LivenessNet.keras\" or ",
     ""),
    ("card: key hashes not checked", S,
     "    if not _hex(c[\"dev_keys_sha256\"]) or not _hex(c[\"fitting_keys_sha256\"]):",
     "    if False:"),
    ("card: dev_keys_sha256 format not checked", S, "not _hex(c[\"dev_keys_sha256\"]) or ", ""),
    ("card: saved not parsed", S, "    parse_time(c[\"saved\"], f\"{what} saved\")\n", ""),
    ("card: test_scored_after_threshold not parsed", S,
     "    parse_time(c[\"test_scored_after_threshold\"], f\"{what} test_scored_after_threshold\")\n",
     ""),
    ("fitting: size of the fitting set not compared with the parent", S,
     " or m[\"card\"][\"n_fitting\"] != p[\"card\"][\"n_fitting\"]:",
     ":"),
    # soglie e dev: un controllo alla volta
    ("threshold: arm not compared", S, "thr.get(\"arm\") != arm or ", ""),
    ("threshold: fold not compared", S, "thr.get(\"fold\") != fold or ", ""),
    ("threshold: train_seed not compared", S, " or thr.get(\"train_seed\") != seed:", ":"),
    ("threshold: a boolean fold or train_seed accepted", S,
     "    if not is_int(thr.get(\"fold\")) or not is_int(thr.get(\"train_seed\")):",
     "    if False:"),
    ("threshold: value not checked", S, "    if not _number(thr.get(\"value\")):", "    if False:"),
    ("threshold: dev_scores_sha256 format not checked", S,
     "    if not _hex(thr.get(\"dev_scores_sha256\")):",
     "    if False:"),
    ("threshold: frozen_at after the test score accepted", S,
     "    if not saved <= frozen <= tested:",
     "    if not saved <= frozen:"),
    ("dev: number of rows not compared", S, "    if len(rows) != n_dev:", "    if False:"),
    ("dev: n_dev of the card not compared", S, "    if card[\"n_dev\"] != n_dev:", "    if False:"),
    ("dev: role not checked", S, "        if r[\"role\"] != \"dev\":", "        if False:"),
    ("dev: fold of the rows not compared", S, "r[\"fold\"] != str(fold) or ", ""),
    ("dev: train_seed of the rows not compared", S, "r[\"train_seed\"] != str(seed) or ", ""),
    ("dev: arm of the rows not compared", S, " or r[\"arm\"] != arm:", ":"),
    ("dev: label not checked", S, "        if r[\"label\"] not in (\"0\", \"1\"):", "        if False:"),
    ("dev: a non-numeric score not reported", S,
     "        except ValueError:\n            fail(f\"{what}: score_attack is not a number\")",
     "        except ValueError:\n            raise"),
    ("dev: score range not checked", S, "        if not math.isfinite(s) or not 0.0 <= s <= 1.0:", "        if False:"),
    ("dev: threshold_sha256 of the rows not compared", S, " or r[\"threshold_sha256\"] != thr[\"sha256\"]:", ":"),
    ("dev: duplicate sample_id accepted", S, "len({k[0] for k in keys}) != len(keys) or ", ""),
    ("dev: duplicate file_sha256 accepted", S, " or len({k[1] for k in keys}) != len(keys):", ":"),
    # status e test dal pubblico: un controllo alla volta
    ("status: any status of the verification run accepted", S,
     "    if status not in allowed or st.get(\"passed\")",
     "    if st.get(\"passed\")"),
    ("status: passed not compared with the status", S, " or st.get(\"passed\") is not (status == \"PASS\"):", ":"),
    ("status: an amendment that is not an object accepted", S, "if not isinstance(am, dict) or any(", "if any("),
    ("status: complete models that are not a list accepted", S,
     "if not isinstance(completed, list) or not isinstance(missing, list)",
     "if not isinstance(missing, list)"),
    ("status: missing models that are not a list accepted", S, " or not isinstance(missing, list)", ""),
    ("status: no complete model accepted", S, " or not completed or ", " or "),
    ("status: wall_seconds not checked", S, "    if not _number(wall) or wall <= 0:", "    if False:"),
    ("status: an infinite wall_seconds accepted", S, "    if not _number(wall) or wall <= 0:", "    if wall <= 0:"),
    ("public: aggregate.json not required in the public SHA256SUMS", S,
     "        if \"aggregate.json\" not in plisted:",
     "        if False:"),
    ("test: strata that are not a list accepted", S,
     "    if not isinstance(strata, list) or not strata",
     "    if not strata"),
    ("test: empty strata accepted", S, "isinstance(strata, list) or not strata or", "isinstance(strata, list) or"),
    ("test: non-text strata accepted", S, " or not all(isinstance(s, str) for s in strata)", ""),
    ("test: duplicate strata accepted", S, " or len(set(strata)) != len(strata):", ":"),
    ("test: a verdict that is not an object accepted", S,
     "if not isinstance(v, dict) or v.get(\"verdict\")",
     "if v.get(\"verdict\")"),
    ("test: a verdict without reason accepted", S, " or not isinstance(v.get(\"reason\"), str):", ":"),
    ("test: primary verdicts that are not an object accepted", S,
     "if not isinstance(pv, dict) or sorted(pv)",
     "if sorted(pv)"),
    ("test: comparisons keys not checked", S,
     "    if not isinstance(comps, dict) or sorted(comps) != sorted(COMPARISONS):",
     "    if not isinstance(comps, dict):"),
    ("test: comparisons that are not an object accepted", S,
     "if not isinstance(comps, dict) or sorted(comps)",
     "if sorted(comps)"),
    ("test: a comparison that is not an object accepted", S,
     "if not isinstance(cc, dict) or cc.get(\"minuend\")",
     "if cc.get(\"minuend\")"),
    ("test: a missing verdict or median skipped", S,
     "                    fail(f\"{what}: comparisons.{c} has no verdict or median for stratum {s}, metric {m}\")",
     "                    continue"),
    # uscite, figure, cartella del run, log, git
    ("public: image keys not scanned", S, "    if re.search(r\"\\b(real|attack)/\\d{4}_\", text):", "    if False:"),
    ("public: package_rules not applied", S, "    for lab, rx in pr.FORBIDDEN_TEXT:", "    for lab, rx in []:"),
    ("figures: empty PNG bytes accepted", S, "        if not isinstance(b, bytes) or not b:", "        if False:"),
    ("figures: a text accepted as PNG bytes", S, "        if not isinstance(b, bytes) or not b:", "        if not b:"),
    ("run: a symbolic link as output root accepted", S, "    if out_root.is_symlink():", "    if False:"),
    ("run: output_problem of the run folder ignored", S,
     "    if msg:\n        return msg\n    if msg := mw.output_problem(run / PUBLIC",
     "    if False:\n        return msg\n    if msg := mw.output_problem(run / PUBLIC"),
    ("run: an input inside the run folder accepted", S,
     "        if pg.within(p, run) or (Path(p).is_dir() and pg.within(run, p)):",
     "        if Path(p).is_dir() and pg.within(run, p):"),
    ("log: a symbolic link as run.log accepted", S,
     "    if p.is_symlink() or (p.exists() and not p.is_file()):\n        fail(f\"{LOG}",
     "    if False:\n        fail(f\"{LOG}"),
    ("times: within_cap with < instead of <=", S,
     "\"within_cap\": new.st[\"wall_seconds\"] <= tm[\"cap_seconds\"],",
     "\"within_cap\": new.st[\"wall_seconds\"] < tm[\"cap_seconds\"],"),
    ("run id: date format not checked", S, "    if not re.fullmatch(r\"\\d{8}\", date or \"\"):", "    if False:"),
    ("run id: format not checked", S, "    if not (RUN_ID_RE_V2 if schema == CONFIG_SCHEMA_V2 else RUN_ID_RE).match(rid):", "    if False:"),
    ("git: unreadable state accepted", S,
     "    if c.returncode or s.returncode or not c.stdout.strip():",
     "    if False:"),
    ("git: a missing git not reported", S,
     "    except (OSError, subprocess.SubprocessError) as e:",
     "    except ZeroDivisionError as e:"),
    # schema 2 (controllo a learning rate costante): configurazione
    ("v2 config: top keys of schema 1 used", S, "    top = TOP_KEYS_V2 if v2 else TOP_KEYS", "    top = TOP_KEYS"),
    ("v2 config: configuration_name of the parent not required", S, 'common + (["configuration_name"] if v2 else [])', "common"),
    ("v2 config: pending state never recognised", S, "    pending = v2 and all(ver[k] is None for k in ANCHORS)", "    pending = False"),
    ("v2 config: identity null without the anchors accepted", S,
     "    if v2 and any((ver[k] is None) != pending for k in LATE_IDENTITY):", "    if False:"),
    ("v2 config: pending config_sha256 not checked", S,
     '        if ver["config_sha256"] is not None and not _hex(ver["config_sha256"]):', "        if False:"),
    ("v2 config: pending patience not checked", S,
     '        if not is_int(ver["early_stopping_patience"]) or ver["early_stopping_patience"] < 1:', "        if False:"),
    ("v2 config: pending label not checked", S, '        _text(ver["label"], "inputs.verification_run.label")', "        pass"),
    ("v2 config: pending identity checked as filled", S,
     '((("verification_run", ver),) if not pending else ())', '(("verification_run", ver),)'),
    ("v2 config: schema 1 checks of the runs used", S, "    (_runs_v2 if v2 else _runs_v1)(par, ver)", "    _runs_v1(par, ver)"),
    ("v2 config: configuration_name of the parent not checked", S,
     '    if par["configuration_name"] != PARENT_CONFIGURATION_NAME_V2:', "    if False:"),
    ("v2 config: configuration_name of the control not checked", S,
     '    if ver["configuration_name"] != CONFIGURATION_NAME_V2:', "    if False:"),
    ("v2 config: amendment field not checked", S, 'if am["changed_field"] != AMENDMENT_V2["changed_field"] or ', "if "),
    ("v2 config: amendment from not checked", S, ' or am["from"] != AMENDMENT_V2["from"]', ""),
    ("v2 config: amendment to not a text accepted", S, ' or not isinstance(am["to"], str) \\', " \\"),
    ("v2 config: amendment to prefix not checked", S,
     '            or not am["to"].startswith(AMENDMENT_V2["to_prefix"]):', "            or False:"),
    ("v2 config: parent of the amendment not checked", S,
     '    if am["parent_config_sha256"] != par["config_sha256"]:\n        fail("configuration: in schema 2',
     '    if False:\n        fail("configuration: in schema 2'),
    ("v2 config: different patience of the two runs accepted", S,
     '    if par["early_stopping_patience"] != ver["early_stopping_patience"]:', "    if False:"),
    ("v2 config: criteria of schema 2 not required", S, '"source"] + (list(CRITERIA_V2) if v2 else []), "criteria")',
     '"source"], "criteria")'),
    ("v2 config: window rule not checked", S, '        if cr["shared_history_window"] != WINDOW_RULE:', "        if False:"),
    ("v2 config: recorded lr not compared with the float32 recipe lr", S,
     ' or cr["recorded_recipe_lr"] != float32(rc["lr"]):', ":"),
    ("v2 config: float32 rounding dropped", S, '    return struct.unpack("f", struct.pack("f", v))[0]', "    return v"),
    ("v2 config: models reported apart not checked", S,
     "        if not isinstance(ap, list) or len(set(ap)) != len(ap) or not set(ap) <= set(names) or ap != sorted(ap):",
     "        if not isinstance(ap, list):"),
    ("v2 config: duplicate models reported apart accepted", S, " or len(set(ap)) != len(ap) or not set(ap)", " or not set(ap)"),
    ("v2 config: unknown models reported apart accepted", S, " or not set(ap) <= set(names) or ap != sorted(ap):", " or ap != sorted(ap):"),
    ("v2 config: unsorted models reported apart accepted", S, " <= set(names) or ap != sorted(ap):", " <= set(names):"),
    ("v2 config: delivered constant of schema 1 used", S,
     "    return CONFIG_SHA256_V2 if schema_v2(cfg) else CONFIG_SHA256", "    return CONFIG_SHA256"),
    # schema 2: ingressi e finestra
    ("v2 inputs: configuration_name of the parent status not compared", S,
     '    if g == "parent_run" and "configuration_name" in spec and st.get("configuration_name") != spec["configuration_name"]:',
     "    if False:"),
    ("v2 window: a tolerance on the recorded rate", S, '    while w < len(h["lr"]) and h["lr"][w] == lr_rec:',
     '    while w < len(h["lr"]) and abs(h["lr"][w] - lr_rec) < 1e-6:'),
    ("v2 window: first reduced epoch off by one", S, '    return w, (w + 1 if w < len(h["lr"]) else None)',
     '    return w, (w if w < len(h["lr"]) else None)'),
    ("v2 window: a parent without a window accepted", S, "        if w == 0:", "        if False:"),
    ("v2 window: a rising rate accepted", S, '        if first is not None and h["lr"][w] > lr_rec:', "        if False:"),
    ("v2 window: never reduced counted wrongly", S, '"never_reduced": len(wins) - len(reduced)', '"never_reduced": len(reduced)'),
    ("v2 window: minimum window wrong", S, '"min_shared_epochs": min(wins)', '"min_shared_epochs": max(wins)'),
    ("v2 window: first reduction minimum wrong", S, '"first_reduction_epoch_min": min(reduced) if reduced',
     '"first_reduction_epoch_min": max(reduced) if reduced'),
    ("v2 window: not computed by check-inputs", S,
     "    if schema_v2(cfg):\n        inp.checks[\"parent_run\"][\"shared_history_windows\"]",
     "    if False:\n        inp.checks[\"parent_run\"][\"shared_history_windows\"]"),
    ("v2 parent lists: degenerate reference models excluded", S, "    if schema_v2(cfg):\n        deg = []", "    if schema_v2(cfg):\n        pass"),
    ("v2 check-inputs: windows not printed", S, '    if "shared_history_windows" in p:', "    if False:"),
    # schema 2: analisi
    ("v2 (a): window replaced by the epochs run", S,
     '    span = {n: P[n]["shared_epochs"] if v2 else P[n]["epochs_run"] for n in names}',
     '    span = {n: P[n]["epochs_run"] for n in names}'),
    ("v2 (a): constant lr compared with a tolerance", S, '"lr_constant": all(v == lr_rec for v in lrs)',
     '"lr_constant": all(abs(v - lr_rec) < 1e-6 for v in lrs)'),
    ("v2 (a): models with another lr not listed", S, '    lr_bad = [label(n) for n in N if v2 and not N[n]["lr_constant"]]',
     "    lr_bad = []"),
    ("v2 (a): constant lr not required for validity", S, "    valid = identical == len(N) and not lr_bad",
     "    valid = identical == len(N)"),
    ("v2 (b): reading of schema 1 used", S, "    elif v2:\n        verdict, concl, lost = primary_v2(cfg, names, P, N, W)",
     "    elif False:\n        verdict, concl, lost = primary_v2(cfg, names, P, N, W)"),
    ("v2 (b): one of the ten without recovery not enough", S, "    if ten_not_rec:\n        concl = (f\"At least one",
     "    if len(ten_not_rec) > 1:\n        concl = (f\"At least one"),
    ("v2 (b): degenerate models not required", S, "    if not deg and not lost:", "    if not lost:"),
    ("v2 (b): lost recoveries not required", S, "    if not deg and not lost:", "    if not deg:"),
    ("v2 (b): lost recoveries not computed", S,
     '    lost = [label(n) for n in names if n not in ten and P[n]["recovered"] and not N[n]["recovered"]]', "    lost = []"),
    ("v2 (b): degenerate models of the control not computed", S, '    deg = [label(n) for n in names if N[n]["degenerate"]]', "    deg = []"),
    ("v2 (b): the other outcomes not added to a contribution", S, "        if deg or lost:\n            concl += (f\" Also:",
     "        if False:\n            concl += (f\" Also:"),
    ("v2 (b): first recovery of the reference from the control", S, '"first_recovery_parent": P[n]["first_recovery"],',
     '"first_recovery_parent": N[n]["first_recovery"],'),
    # schema 2: secondari, tabelle e testi
    ("v2 (c): recovery at the last epoch without the val_loss", S,
     '    return h["val_loss"][-1] < rv["val_loss_max"] and h["val_accuracy"][-1] >= rv["val_accuracy_min"]',
     '    return h["val_accuracy"][-1] >= rv["val_accuracy_min"]'),
    ("v2 (c): restored epoch change not detected", S,
     '"restored_epoch_changed": q["restored_epoch"] != p["restored_epoch"] if q else None',
     '"restored_epoch_changed": False if q else None'),
    ("v2 (c): models reported apart not marked", S, '"reported_apart": n in cfg["models_reported_apart"]}', '"reported_apart": False}'),
    ("v2 (c): extra columns not written", S, "        if v2:          # schema 2: colonne in più", "        if False:          # schema 2: colonne in più"),
    ("v2 curves: the ten read from no list", S, "int(n in ten_of(cfg)), i + 1]", "int(False), i + 1]"),
    ("v2 texts: fixed words of schema 1", S, '    if not schema_v2(cfg):\n        return {"parent": "parent"', '    if True:\n        return {"parent": "parent"'),
    ("v2 texts: words not passed to the Markdown", S, 'aggregate_md(doc, words(cfg), v2)', "aggregate_md(doc, None, v2)"),
    ("v2 texts: legend of the figures not labelled", S, '        out["legend"] = {"parent": W["parent_run"], "new": W["new_run"]}', "        pass"),
    ("v2 texts: runs of the times not labelled", S, 'per_arm(par, W["parent"]) + per_arm(new, W["new"])', 'per_arm(par, "parent") + per_arm(new, "new")'),
    ("v2 texts: no-verdict text with the fixed name", S, 'concl = (f"No verdict: the {W[\'new_run\']} is', 'concl = (f"No verdict: the verification run is'),
    # schema 2: run id e radice d'uscita
    ("v2 run: run id of schema 1", S, '    rid = run_id_for(a.date or time.strftime("%Y%m%d"), state, cfg["schema"])',
     '    rid = run_id_for(a.date or time.strftime("%Y%m%d"), state)'),
    ("v2 run: output root of schema 1 by default", S, "    o = a.out_root or OUT_ROOTS[cfg[\"schema\"]]", "    o = a.out_root or DEFAULT_OUT_ROOT"),
    ("v2 run: schema not passed to the folder check", S, "    if msg := run_dir_problem(run, ROOT, inputs, out_root, cfg[\"schema\"]):",
     "    if msg := run_dir_problem(run, ROOT, inputs, out_root):"),
    ("v2 card: scheduler key not required in the control cards", S,
     '    keys = sorted(CARD_KEYS + ("scheduler",)) if control else list(CARD_KEYS)', "    keys = list(CARD_KEYS)"),
    ("v2 card: scheduler value not compared with the amendment", S,
     '    if control and c["scheduler"] != cfg["inputs"]["verification_run"]["amendment"]["to"]:', "    if False:"),
    ("v2 card: scheduler key accepted in the reference cards", S,
     '    control = schema_v2(cfg) and what == "verification_run"', "    control = schema_v2(cfg)"),
    ("v2 run: allowed root of schema 1", S, "    allowed = [root / OUT_ROOTS[schema]]", "    allowed = [root / DEFAULT_OUT_ROOT]"),
]
# mutazione di controllo: solo un commento; deve SOPRAVVIVERE (prova che la suite rileva il comportamento, non il testo)
CONTROL = [("control: comment only", S, "def label(name: str) -> str:\n", "def label(name: str) -> str:  # control mutant: comment only\n")]
# controlli dichiarati ridondanti: devono SOPRAVVIVERE, perché un altro controllo li copre sempre
REDUNDANT = [
    # una cartella del run dentro un ingresso è rifiutata anche da m2_writers.output_problem
    ("declared redundant (must survive): run folder inside an input", S,
     "        if pg.within(p, run) or (Path(p).is_dir() and pg.within(run, p)):", "        if pg.within(p, run):"),
    # i nomi dei modelli vengono solo da model_names: bracci fra quelli di ARM_LABEL, fold e seed interi >= 0
    # (load_config)
    ("declared redundant (must survive): model name format", S,
     "    if not m:\n        fail(f\"model name {name!r}",
     "    if False:\n        fail(f\"model name {name!r}"),
    # sorted(arms) == sorted(ARM_LABEL) implica già la stessa lunghezza
    ("declared redundant (must survive): number of arms", S, " or len(inputs[\"arms\"]) != len(ARM_LABEL):", ":"),
    # completed + missing uguale ai 36 nomi distinti (controllo successivo della stessa riga) esclude già i doppioni
    ("declared redundant (must survive): duplicate complete models", S,
     " or len(set(completed)) != len(completed)",
     ""),
    # con stop == epochs_run il replay ha visto tutte le epoche: il ripristinato è il primo minimo, cioè
    # restored_epoch (già uguale all'argmin, controllo precedente)
    ("declared redundant (must survive): restored epoch of the replay", S, " or sim[\"restored\"] != r:", ":"),
    # i valori del SHA256SUMS sono esadecimali (read_sums): l'uguaglianza con listed nella stessa condizione lo implica
    ("declared redundant (must survive): weights_sha256 format", S, " or not _hex(c[\"weights_sha256\"])", ""),
    # nan e ±inf non stanno in [0, 1]: rifiutati dalla stessa condizione, con lo stesso messaggio
    ("declared redundant (must survive): score finiteness", S, "not math.isfinite(s) or ", ""),
    # genitore PASS, nessun mancante e schede uguali ai completi (load_run): ogni modello del genitore è caricato
    ("declared redundant (must survive): parent with incomplete models", S,
     "    if len(inp.parent.models) != cfg[\"inputs\"][\"models\"]:",
     "    if False:"),
    # solo cmd_run chiede il run di verifica, e rifiuta le ancore null prima di leggere gli ingressi
    ("declared redundant (must survive): null anchors in load_inputs", S,
     "        if need_verification:",
     "        if False:"),
    # output_problem della cartella del run copre già public/ (dirs=[run / PUBLIC]): collegamento, file al posto della
    # cartella, voci non regolari o bloccate, cartella senza scrittura
    ("declared redundant (must survive): output_problem of the public folder", S,
     "    if msg := mw.output_problem(run / PUBLIC, root",
     "    if False and mw.output_problem(run / PUBLIC, root"),
    # run_dir_problem rifiuta prima ogni uscita pubblica che invalidate non saprebbe togliere: resta solo una corsa
    ("declared redundant (must survive): outputs left before the run", S,
     "        if left:\n            fail(f\"previous outputs could not be removed",
     "        if False:\n            fail(f\"previous outputs could not be removed"),
    # invalidate_outputs toglie ogni figura (anche un collegamento) prima della scrittura: resta solo una corsa
    # nel repository path_guard.deletion_problem ammette solo la radice d'uscita dello schema (allowed): la radice
    # dell'altro schema è già rifiutata; la protezione in più resta come difesa se allowed cambiasse
    ("declared redundant (must survive): root of the other schema protected", S,
     "    return [root / x for x in EXTRA_PROTECTED] + [root / OUT_ROOTS[k] for k in SCHEMAS if k != schema]",
     "    return [root / x for x in EXTRA_PROTECTED]"),
    ("declared redundant (must survive): link among the figures before the write", S,
     "            if q.is_symlink() or (q.exists() and not q.is_file()):\n                fail(f\"{n} is a link",
     "            if False:\n                fail(f\"{n} is a link"),
]


def copy_tree(dst: Path) -> Path:
    code = dst / "code"
    for rel in COPY + tuple(c for c in CONFIGS if (ROOT / c).is_file()):
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, code / rel)
    return code


def run_tests(code: Path, tmp: Path) -> tuple:
    tmp.mkdir(parents=True, exist_ok=True)
    empty = tmp / "no_real_inputs"
    empty.mkdir(exist_ok=True)
    # mai --real; radice degli ingressi veri su una cartella vuota: i mutanti non leggono dati veri
    env = {**os.environ, "PATVER_INPUT_ROOT": str(empty), "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp)}
    r = subprocess.run([sys.executable, "-I", TESTS], cwd=code, env=env, capture_output=True, text=True, timeout=900)
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("tests run")), "no test summary")
    skips = tuple(sorted(l.split(": ", 1)[-1] for l in r.stdout.splitlines() if l.startswith("skipped ")))
    return r.returncode, line, skips


def undeclared_skips(skips) -> list:
    """Salti il cui motivo non è dichiarato («missing input: <motivo>» di MISSING_INPUT)."""
    allowed = {f"missing input: {why}" for why in MISSING_INPUT.values()}
    return [s for s in skips if s not in allowed]


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
    ap = argparse.ArgumentParser(description="Controllo delle mutazioni di nuaa_patience_verify.py (offline, senza dati veri)")
    ap.add_argument("--list", action="store_true", help="elenca le mutazioni e verifica i modelli, senza eseguire i test")
    ap.add_argument("--only", action="append", default=[], help="esegue solo la mutazione con questo nome (ripetibile)")
    ap.add_argument("--jobs", type=int, default=1, help=f"mutazioni eseguite in parallelo (predefinito 1, al massimo {MAX_JOBS})")
    a = ap.parse_args()
    if not 1 <= a.jobs <= MAX_JOBS:
        print(f"ERROR: --jobs must be between 1 and {MAX_JOBS}", file=sys.stderr)
        return 2
    problems = [f"{n}: the pattern occurs {k} times in {t} (expected exactly once)"
                for n, t, old, _ in MUTANTS + CONTROL + REDUNDANT if (k := (ROOT / t).read_text(encoding="utf-8").count(old)) != 1]
    if len({n for n, *_ in MUTANTS + CONTROL + REDUNDANT}) != len(MUTANTS) + len(CONTROL) + len(REDUNDANT):
        problems.append("two mutations share a name")
    problems += [f"--only {n!r}: no such mutation" for n in a.only if n not in {m[0] for m in MUTANTS + CONTROL + REDUNDANT}]
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
    mutants = [m for m in MUTANTS if not a.only or m[0] in a.only]
    controls = [m for m in CONTROL + REDUNDANT if not a.only or m[0] in a.only]
    killed = invalid = survived_controls = 0
    with tempfile.TemporaryDirectory(prefix="patver-mutations-") as tmp:
        tmp = Path(tmp)
        rc, line, base_skips = run_tests(copy_tree(tmp / "base"), tmp / "base" / "tmp")
        print(f"unmutated copy: {TESTS} exit {rc} ({line})", flush=True)
        for s in base_skips:
            print(f"unmutated copy: skipped: {s}", flush=True)
        if rc or not re.fullmatch(r"tests run: \d+, failures: 0, errors: 0, skipped: \d+", line):
            print("RESULT: FAIL (the tests do not pass on the unmutated copy)")
            return 1
        if bad := undeclared_skips(base_skips):
            for s in bad:
                print(f"ERROR: skip that is not declared: {s}", file=sys.stderr)
            print("RESULT: FAIL (the unmutated copy skips tests for a reason that is not declared)")
            return 1
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            cfuts = [ex.submit(one, tmp, len(MUTANTS) + i, n, t, old, new, base_skips) for i, (n, t, old, new) in enumerate(controls)]
            futs = [ex.submit(one, tmp, i, n, t, old, new, base_skips) for i, (n, t, old, new) in enumerate(mutants)]
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
    print(f"mutations killed: {killed}/{len(mutants)}" + (f" ({invalid} invalid)" if invalid else "") + (" (selection: --only)" if a.only else ""))
    print(f"control mutations survived: {survived_controls}/{len(controls)}")
    ok = killed == len(mutants) and survived_controls == len(controls)
    print("RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
