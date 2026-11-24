"""
Controllo delle mutazioni di make_table_figure.py: una mutazione dichiarata alla volta su una copia temporanea del
pacchetto (senza il record reale: i test del record reale sono saltati con il motivo dichiarato), poi la suite
check_make_table_figure.py su quella copia, per intero. Ogni mutazione toglie o indebolisce un controllo, oppure cambia
una regola di calcolo o di presentazione, e dichiara il test (classe.metodo) che deve ucciderla: è uccisa solo se la
suite fallisce E fra i test falliti c'è quello dichiarato; se fallisce soltanto un altro test la mutazione è «invalid»
(non contata come uccisa). La riga del verdetto dà il primo test fallito e quello dichiarato. Prima di tutto la copia non
mutata deve passare con i soli salti dichiarati; la mutazione di controllo (un commento cambiato) deve sopravvivere.

  python3 -I -B check_make_table_figure_mutations.py --list        # elenco; ogni ancora una sola volta; test dichiarati
  python3 -I -B check_make_table_figure_mutations.py [--jobs N]     # tutte (predefinito: una alla volta)
  python3 -I -B check_make_table_figure_mutations.py --only 3,7     # solo queste (da 1); c = controllo
"""
import argparse
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
TARGET = "make_table_figure.py"
SUITE = "check_make_table_figure.py"
SKIP_OK = ("missing input: matplotlib is not installed: the tests that need it (figure images, real build in the signal "
           "tests, machine state) are skipped",
           "missing input: the records of the evaluation and of its runs are not present: the tests against the real "
           "records are skipped")


def _inside(p, root):
    p, root = os.path.realpath(p), os.path.realpath(root)
    return p == root or p.startswith(root + os.sep)


# (nome, testo cercato, testo sostitutivo, test che deve ucciderla)
MUTANTS = [
    # lettura e integrità
    ("pin of public/SHA256SUMS not checked", "    if sha256(sums) != PUBLIC_SUMS_SHA256:", "    if False:",
     "B_Integrity.test_pin"),
    ("listed file not checked against its line", "        if sha256(b) != listed[name]:", "        if False:",
     "B_Integrity.test_one_byte_in_each_listed_file"),
    ("final newline of SHA256SUMS not checked", '    if not text.endswith("\\n"):', "    if False:",
     "B_Integrity.test_sums_rules"),
    ("duplicate line of SHA256SUMS accepted", "        if m.group(2) in listed:", "        if False:",
     "B_Integrity.test_sums_rules"),
    ("files listed in SHA256SUMS not checked", "    if sorted(listed) != sorted(names):", "    if False:",
     "B_Integrity.test_sums_rules"),
    ("inode re-check in read_regular dropped",
     "if not stat.S_ISREG(st2.st_mode) or (st2.st_dev, st2.st_ino) != (st.st_dev, st.st_ino):",
     "if not stat.S_ISREG(st2.st_mode):", "B_Integrity.test_input_changed_while_opened"),
    ("the restricted folder of the record looked up",
     '    out = {"status.json": read_regular(os.path.join(record, "status.json"), "status.json of the record")}',
     '    out = {"status.json": read_regular(os.path.join(record, "status.json"), "status.json of the record"),\n'
     '           "_r": os.lstat(os.path.join(record, "restricted"))}', "B_Integrity.test_restricted_never_opened"),
    # status.json e intestazione di results.json
    ("public_sha256sums_sha256 of status.json not compared",
     '("second_reading", False), ("public_sha256sums_sha256", sums_sha)):', '("second_reading", False)):',
     "C_Status.test_status_fields"),
    ("attempt of status.json not checked",
     '    expect(st, "attempt", {"attempt": 1, "previous": [], "second_reading": False, "earlier_readings": []},\n'
     '           "status.json")', "    pass", "C_Status.test_status_fields"),
    ("not_evaluable of status.json not compared",
     '    if not same(get(st, "not_evaluable", "status.json"), get(res, "not_evaluable")):', "    if False:",
     "C_Status.test_status_fields"),
    ("schema and official of results.json not checked",
     '    for path, value in (("schema", SCHEMA), ("run_id", RUN_ID), ("status", "PASS"), ("official", True),',
     '    for path, value in (("run_id", RUN_ID), ("status", "PASS"),', "C_Header.test_results_fields"),
    ("second reading accepted in results.json", '                        ("second_reading", False)):\n'
     '        expect(res, path, value)', '                        ):\n        expect(res, path, value)',
     "C_Header.test_results_fields"),
    ("hooks not checked", '    expect(res, "hooks|set", [])', "    pass", "C_Header.test_results_fields"),
    ("C2 reproduction not required exact", '    expect(res, "c2_reproduction|exact", True)', "    pass",
     "C_Header.test_results_fields"),
    ("C2 opposite flags not checked", '    expect(res, "c2_opposite",', '    (lambda *a: None)(res, "c2_opposite",',
     "C_Header.test_results_fields"),
    ("test frames per fold not required positive", "all(type(n) is int and n > 0 for n in frames)",
     "all(type(n) is int for n in frames)", "C_Header.test_frames_and_ids"),
    ("form of the frame identifier hashes not checked",
     "and all(isinstance(v, str) and HEX64_RE.fullmatch(v) for v in ids.values())):",
     "and all(isinstance(v, str) for v in ids.values())):", "C_Header.test_frame_ids_are_sha256"),
    ("C2 opposite: bounds not compared",
     '        if not (mine["mean"][m] == -get(c2, f"mean|{m}") and a["low"] == -b["high"] and a["high"] == -b["low"]',
     '        if not (mine["mean"][m] == -get(c2, f"mean|{m}")', "C_Header.test_c2_opposite_values"),
    ("C2 opposite: bounds compared without swapping", 'a["low"] == -b["high"] and a["high"] == -b["low"]',
     'a["low"] == -b["low"] and a["high"] == -b["high"]', "C_Header.test_valid_record_accepted"),
    ("C2 opposite: mean compared without the sign", 'mine["mean"][m] == -get(c2, f"mean|{m}")',
     'mine["mean"][m] == get(c2, f"mean|{m}")', "C_Header.test_valid_record_accepted"),
    # denominatori
    ("attack + bona fide = frames not checked",
     'if e["frames"] != frames[f - 1] or e["attack"] + e["bona_fide"] != e["frames"]:',
     'if e["frames"] != frames[f - 1]:', "C_Denominators.test_attack_plus_bona"),
    ("frames of the fold not compared with test_frames_per_fold",
     'if e["frames"] != frames[f - 1] or e["attack"] + e["bona_fide"]', 'if e["attack"] + e["bona_fide"]',
     "C_Denominators.test_frames_of_the_fold"),
    ("subjects of the folds not required to add up to 16",
     "            if sum(v[3] for v in folds.values()) != N_SUBJECTS:", "            if False:",
     "C_Denominators.test_subjects_add_up"),
    ("totals over the units not checked",
     'if not same({k: d.get(k) for k in want}, want) or sorted(d) != sorted(list(want) + ["per_fold"]):',
     'if sorted(d) != sorted(list(want) + ["per_fold"]):', "C_Denominators.test_totals"),
    ("denominators not required equal in every condition", "            elif folds != found:",
     "            elif False:", "C_Denominators.test_same_in_every_condition"),
    # units.csv
    ("effective threshold 0.5 not checked", '        if t != "0.5":', "        if False:",
     "C_Units.test_threshold_half"),
    ("EER threshold form not checked", "if not (math.isfinite(v) and 0 <= v <= 1 and t == repr(v)):",
     "if not math.isfinite(v):", "C_Units.test_dev_eer_threshold_form"),
    ("n_frames not compared with the fold",
     '    if not INT_RE.fullmatch(r["n_frames"]) or int(r["n_frames"]) != frames[fold - 1]:',
     '    if not INT_RE.fullmatch(r["n_frames"]):', "C_Units.test_n_frames"),
    ("frame identifier hash not compared", '    if r["frame_ids_sha256"] != ids[str(fold)]:', "    if False:",
     "C_Units.test_frame_ids"),
    ("fraction not written reduced accepted", '    if text != f"{f.numerator}/{f.denominator}":', "    if False:",
     "C_Units.test_fractions"),
    ("rate outside [0, 1] accepted", "        if not 0 <= u[m] <= 1:", "        if False:", "C_Units.test_fractions"),
    ("ACER not checked as the mean of APCER and BPCER", '    if u["acer"] != (u["apcer"] + u["bpcer"]) / 2:',
     "    if False:", "C_Units.test_acer_is_the_mean"),
    ("APCER not checked as a count over the attack frames",
     '    if (u["apcer"] * n_att).denominator != 1 or (u["bpcer"] * n_bf).denominator != 1:',
     '    if (u["bpcer"] * n_bf).denominator != 1:', "C_Units.test_rate_is_a_count_of_the_fold"),
    ("BPCER not checked as a count over the bona fide frames",
     '    if (u["apcer"] * n_att).denominator != 1 or (u["bpcer"] * n_bf).denominator != 1:',
     '    if (u["apcer"] * n_att).denominator != 1:', "C_Units.test_bpcer_is_a_count_of_the_fold"),
    ("LivenessNet or ResNet-18 missing from a unit accepted", "                if n not in out[c][k]:",
     "                if False:", "C_Units.test_reused_network_missing"),
    ("a network twice in a unit accepted", '        if r["network"] in out[c][k]:', "        if False:",
     "C_Units.test_network_twice"),
    ("presence of an AttackNet not required equal in the four conditions",
     "            if len(set(sets.values())) != 1:", "            if False:",
     "C_Units.test_presence_same_in_the_four_conditions"),
    ("network outside the plan accepted", '    if r["network"] not in NETWORKS:', "    if False:",
     "C_Units.test_condition_fold_network"),
    ("patience outside the plan accepted",
     'if r["patience"] not in [str(p) for p in PATIENCES] or r["bn_state"] not in STATES \\',
     'if r["bn_state"] not in STATES \\', "C_Units.test_condition_fold_network"),
    ("seed outside the plan accepted",
     '    if r["fold"] not in [str(f) for f in FOLDS] or r["seed"] not in [str(s) for s in SEEDS]:',
     '    if r["fold"] not in [str(f) for f in FOLDS]:', "C_Units.test_condition_fold_network"),
    # non valutabili, guardiano, modelli
    ("not_evaluable not compared with the models missing from units.csv", "            if missing != listed:",
     "            if False:", "C_NotEvaluable.test_list_matches_units"),
    ("run of a not evaluable entry not checked",
     """                or e["run"] != f"anet_{RUN_TAG[e['network']]}_p{e['patience']}":""", "                or False:",
     "C_NotEvaluable.test_entry_form"),
    ("not evaluable entry listed twice accepted",
     '        if key in seen:\n            raise Refusal(f"results.json: not_evaluable lists',
     '        if False:\n            raise Refusal(f"results.json: not_evaluable lists',
     "C_NotEvaluable.test_entry_twice"),
    ("guardian not compared", '        if not same(blk["guardian"], status):', "        if False:",
     "C_NotEvaluable.test_guardian"),
    ("reason of the not evaluable models dropped from the guardian",
     '(reasons.get((n, p)) or "fewer than 12 fold-seed pairs evaluable in both BN "',
     '("fewer than 12 fold-seed pairs evaluable in both BN "', "C_Header.test_valid_record_accepted"),
    ("dev EER thresholds of the models not compared with units.csv",
     'if type(v) is not float or repr(v) != U[(p, st, "dev_eer")][(m["fold"], m["seed"])][m["network"]]["threshold"]:',
     "if type(v) is not float:", "C_NotEvaluable.test_models_thresholds"),
    ("list of the models not compared", "    if seen != want:", "    if False:", "C_NotEvaluable.test_models_list"),
    ("model absent from units.csv accepted", '        if (m["fold"], m["seed"]) not in pres[(m["network"], p)]:',
     "        if False:", "C_NotEvaluable.test_model_not_in_units"),
    # quantità
    ("means not compared", "        if type(got) is not float or got != float(exact):",
     "        if type(got) is not float:", "C_Quantities.test_mean_each_metric"),
    ("means compared for ACER only", "        if type(got) is not float or got != float(exact):",
     '        if type(got) is not float or (m == "acer" and got != float(exact)):', "C_Quantities.test_mean_each_metric"),
    ("unit counts not compared", '    if not same(blk["counts"], counts):', "    if False:", "C_Quantities.test_counts"),
    ("a zero unit value counted above zero", '"above_zero": sum(x > 0 for x in acer),',
     '"above_zero": sum(x >= 0 for x in acer),', "C_Header.test_valid_record_accepted"),
    ("interval low <= high not checked", "    if not low <= high:", "    if False:",
     "C_Quantities.test_interval_low_high"),
    ("mean outside its interval accepted", "    if not low <= mean_value <= high:", "    if False:",
     "C_Quantities.test_interval_contains_mean"),
    ("valid resamples not bounded", "    if type(valid) is not int or not 0 <= valid <= 1000:",
     "    if type(valid) is not int:", "C_Quantities.test_interval_form"),
    ("bounds without valid resamples accepted", "        if low is not None or high is not None:", "        if False:",
     "C_Quantities.test_no_interval_without_resamples"),
    ("a bound at zero excludes zero in the reading",
     "        return names[0], INTERVAL_REASONS[0]\n    if low > 0:",
     "        return names[0], INTERVAL_REASONS[0]\n    if low >= 0:", "A3_BoundAtZero.test_valid_record_accepted"),
    ("readings lower and higher swapped", "    if high < 0:\n        return names[0], INTERVAL_REASONS[0]",
     "    if high < 0:\n        return names[1], INTERVAL_REASONS[0]", "C_Header.test_valid_record_accepted"),
    ("conclusion not checked", '        if blk["conclusion"] != conclusion_rule(net, rd):', "        if False:",
     "C_Quantities.test_conclusion"),
    ("reading not checked per seed", '        if (blk["reading"], blk["reason"]) != (rd, reason):',
     '        if "per_seed" not in path and (blk["reading"], blk["reason"]) != (rd, reason):',
     "C_Quantities.test_reading_rule_per_seed"),
    ("extra seeds accepted in per_seed",
     'if not isinstance(blk["per_seed"], dict) or sorted(blk["per_seed"]) != [str(s) for s in SEEDS]:',
     'if not isinstance(blk["per_seed"], dict):', "C_Quantities.test_per_seed_keys"),
    ("n_subjects not checked per seed", '            if not same(sb["n_subjects"], N_SUBJECTS):',
     '            if scope == "all seeds" and not same(sb["n_subjects"], N_SUBJECTS):', "C_Quantities.test_n_subjects"),
    ("order of the networks not checked", "if not isinstance(group, dict) or list(group) != list(nets):",
     "if not isinstance(group, dict) or sorted(group) != sorted(nets):", "C_Quantities.test_network_order"),
    ("'not computed' block not checked", "    if not same(blk, want):", "    if False:",
     "C_Quantities.test_not_computed_block"),
    ("extra fields (a reading) accepted in a block",
     "    if not isinstance(blk, dict) or sorted(blk) != sorted(want_keys):",
     "    if not isinstance(blk, dict) or not set(want_keys) <= set(blk):",
     "C_Quantities.test_reading_in_descriptive_refused"),
    ("note on the individual intervals not checked", '        expect(blk, "intervals", INTERVALS_NOTE, path)',
     "        pass", "C_Quantities.test_notes"),
    ("valid resamples not required uniform", "        if len(seen) != 1:", "        if False:",
     "C_Quantities.test_valid_resamples_uniform"),
    # righe per unità di results.json
    ("float value of a unit row not compared",
     'or type(e.get("value")) is not float or e["value"] != float(want[m]):', 'or type(e.get("value")) is not float:',
     "C_Rows.test_value_and_exact"),
    ("order of the unit rows not checked",
     '        if not isinstance(row, dict) or not same(row.get("unit"), ["NUAA", u[0], u[1]]):',
     "        if not isinstance(row, dict):", "C_Rows.test_order_and_keys"),
    ("quantities of a unit row not checked",
     '        if sorted(row) != sorted(["unit", "n_frames", "frame_ids_sha256"] + list(specs)):', "        if False:",
     "C_Rows.test_order_and_keys"),
    # table.csv, conditions.csv, classe unica, graduatorie
    ("table.csv not compared", '        if g != w:\n            raise Refusal(f"table.csv',
     '        if False:\n            raise Refusal(f"table.csv', "C_OtherFiles.test_table_csv_each_field"),
    ("row count of table.csv not checked", '    if len(got) != len(want):\n        raise Refusal(f"table.csv:',
     '    if False:\n        raise Refusal(f"table.csv:', "C_OtherFiles.test_table_csv_rows"),
    ("denominators of a network summed over units it does not have",
     '            "n_attack": sum(folds[u[0]][1] for u in mine), "n_bona_fide": sum(folds[u[0]][2] for u in mine),',
     '            "n_attack": sum(folds[u[0]][1] for u in keys), "n_bona_fide": sum(folds[u[0]][2] for u in keys),',
     "A_Numbers.test_not_evaluable_and_not_computed"),
    ("subjects summed per unit, not per fold", "    subjects = sum(folds[f][3] for f in sorted({u[0] for u in mine}))",
     "    subjects = sum(folds[u[0]][3] for u in mine)", "C_Header.test_valid_record_accepted"),
    ("conditions.csv not compared", '        if g != w:\n            raise Refusal(f"conditions.csv',
     '        if False:\n            raise Refusal(f"conditions.csv', "C_OtherFiles.test_conditions_csv_each_field"),
    ("test_one_class not compared", '    if not same(get(res, "test_one_class"), counts):', "    if False:",
     "C_OtherFiles.test_test_one_class"),
    ("models deciding every frame bona fide ignored", "                elif pair == (1, 0):",
     "                elif False:", "C_Header.test_valid_record_accepted"),
    ("rankings not compared", "        if not same(get(res, path), mine):", "        if False:",
     "C_OtherFiles.test_rankings"),
    ("ties ranked apart by the network order",
     "        return {n: 1 + sum(1 for m in vals if vals[m] < vals[n]) for n in vals}",
     "        return {n: 1 + sum(1 for m in vals if vals[m] < vals[n] or (vals[m] == vals[n] and NETWORKS.index(m) < "
     "NETWORKS.index(n))) for n in vals}", "C_Header.test_valid_record_accepted"),
    # regole di calcolo
    ("sign of the variations reversed",
     '        return ((1, net, "re-estimated"), (-1, net, "ordinary"))\n    if quantity == "contrast":',
     '        return ((-1, net, "re-estimated"), (1, net, "ordinary"))\n    if quantity == "contrast":',
     "C_Header.test_valid_record_accepted"),
    ("contrast computed against ResNet-18",
     '(-1, REFERENCE, "re-estimated"),\n                (1, REFERENCE, "ordinary"))',
     '(-1, EXTERNAL, "re-estimated"),\n                (1, EXTERNAL, "ordinary"))',
     "C_Header.test_valid_record_accepted"),
    # presentazione
    ("BN states swapped in the rate tables", '            r = agg.row(p, t, "rate", n, st, scope)',
     '            r = agg.row(p, t, "rate", n, STATES[1 - STATES.index(st)], scope)', "A_Numbers.test_rates_table"),
    ("Table 1 from patience 12", '    p, t = PRIMARY\n    rows = [(f"variation of {n}"',
     '    p, t = (12, "fixed_0.5")\n    rows = [(f"variation of {n}"', "A_Numbers.test_table_is_the_primary_condition"),
    ("attack and bona fide denominators swapped",
     '                    row.update(n_attack=str(r["n_attack"]), n_bona_fide=str(r["n_bona_fide"]),',
     '                    row.update(n_attack=str(r["n_bona_fide"]), n_bona_fide=str(r["n_attack"]),',
     "A_Numbers.test_rates_table"),
    ("rounding half down", "        if (q - n) * 2 >= 1:", "        if (q - n) * 2 > 1:", "A_Numbers.test_rounding"),
    ("rounding from the binary value of a float, not from the exact fraction",
     "    if isinstance(x, float):\n        d = Decimal(repr(x))",
     "    if isinstance(x, (float, Fraction)):\n        d = Decimal(float(x))", "A_Numbers.test_rounding_from_exact"),
    ("rows of the rate tables not in the order of the protocol",
     '    for n in NETWORKS:\n        for st in STATES:\n            r = agg.row(p, t, "rate", n, st, scope)',
     '    for n in sorted(NETWORKS):\n        for st in STATES:\n            r = agg.row(p, t, "rate", n, st, scope)',
     "A_Numbers.test_rows_order"),
    ("reading and its reason swapped in the table", """            line += f" \\"{r['reading']}\\" ({r['reading_reason']}) |\"""",
     """            line += f" \\"{r['reading_reason']}\\" ({r['reading']}) |\"""", "A_Numbers.test_contrasts_and_readings"),
    ("figure: mean drawn at the lower bound", "None if mu is None else float(points(mu))",
     "None if mu is None else float(points(lo))", "A_Numbers.test_figure_values"),
    ("figure: interval bounds swapped",
     'items.append((quantity, n, val(r, "acer"), bound(r, "acer", "low"), bound(r, "acer", "high"), units))',
     'items.append((quantity, n, val(r, "acer"), bound(r, "acer", "high"), bound(r, "acer", "low"), units))',
     "A_Numbers.test_figure_values"),
    ("tables: interval bounds swapped", '    low, high = bound(r, m, "low"), bound(r, m, "high")\n    iv = "[no interval]"',
     '    high, low = bound(r, m, "low"), bound(r, m, "high")\n    iv = "[no interval]"', "A_Numbers.test_variations"),
    ("aggregate: interval bounds swapped", '                            row[f"{m}_low"] = "" if low is None else repr(low)',
     '                            row[f"{m}_low"] = "" if low is None else repr(high)', "A_Numbers.test_intervals_copied"),
    ("a cell not computed written as zero", '                            row.update(computed="0", note=status["text"])',
     '                            row.update(computed="1", **{k: "0" for k in ("acer_exact", "apcer_exact", '
     '"bpcer_exact")}, **{k: "0.0" for k in METRICS})', "A2_NotComputedPrimary.test_not_computed_cells"),
    ("sign dropped on values that round to 0.00", "    if x < 0:\n        return minus + s",
     '    if x < 0 and s != "0.00":\n        return minus + s', "A_Numbers.test_rounding"),
    ("values that round to zero not explained", "                if x is not None and x != 0 and points(x) == 0:",
     "                if False:", "A_Numbers.test_near_zero"),
    ("unit points never stacked", "        while lvl in last and x - last[lvl] < LEVEL_GAP_POINTS:",
     "        while False:", "A_Numbers.test_unit_levels"),
    ("valid resamples of ACER and APCER swapped in the text",
     """    return f"{v['acer']:,} for ACER, {v['apcer']:,} for APCER""",
     """    return f"{v['apcer']:,} for ACER, {v['acer']:,} for APCER""", "A_Numbers.test_individual_intervals_declared"),
    ("intervals not declared individual",
     'in every block (checked). They are individual 95% intervals: no correction "',
     'in every block (checked). They are 95% intervals: no correction "',
     "A_Numbers.test_individual_intervals_declared"),
    ("attack and bona fide frames of a fold swapped in the text",
     'f"fold {k}: {f[k][0]:,} frames ({f[k][1]:,} attack, {f[k][2]:,} bona fide)',
     'f"fold {k}: {f[k][0]:,} frames ({f[k][2]:,} attack, {f[k][1]:,} bona fide)', "A_Numbers.test_denominators_text"),
    ("per-seed rates over every unit", "                *rates_table(agg, p, t, scope), \"\",",
     "                *rates_table(agg, p, t), \"\",", "A_Numbers.test_per_seed"),
    ("readings printed per seed",
     '                *quantity_table(agg, p, t, "contrast", ATTACKNETS, scope, "contrast of"), ""]',
     '                *quantity_table(agg, p, t, "contrast", ATTACKNETS, scope, "contrast of", reading=True), ""]',
     "A_Numbers.test_per_seed"),
    ("descriptive conditions labelled primary",
     '    return "primary" if (p, t) == PRIMARY else "descriptive"', '    return "primary"',
     "A_Numbers.test_descriptive_marked"),
    ("one-class counts written n of k", """            cells.append(f"{e['one_class']} of {e['models']}\"""",
     """            cells.append(f"{e['models']} of {e['one_class']}\"""", "A_Numbers.test_single_class"),
    # aggiunte della revisione 1: presentazione
    ("plus sign dropped from positive values", '    return ("+" + s) if signed and x > 0 else s', "    return s",
     "A_Numbers.test_variations"),
    ("ASCII hyphen as the minus sign in Markdown", 'MINUS = "−"', 'MINUS = "-"', "A_Numbers.test_near_zero"),
    ("U+2212 in the .tex", '    s = fmt(x, signed)\n    return f"${s}$"', '    s = fmt_u(x, signed)\n    return f"${s}$"',
     "A_Numbers.test_tex"),
    ("per-unit table: ordinary columns from the re-estimated BN",
     '            o = agg.row(p, t, "rate", n, "ordinary", "unit", u)',
     '            o = agg.row(p, t, "rate", n, "re-estimated", "unit", u)', "A_Numbers.test_rates_by_hand"),
    ("per-unit table: contrast column filled with the variation",
     '            c = agg.row(p, t, "contrast", n, VARIATION_STATE, "unit", u) if n in ATTACKNETS else None',
     '            c = agg.row(p, t, "variation", n, VARIATION_STATE, "unit", u) if n in ATTACKNETS else None',
     "A_Numbers.test_per_unit_contrast_column"),
    ("figure: unit points from APCER", '                units.append((float(points(val(ur, "acer"))), u))',
     '                units.append((float(points(val(ur, "apcer"))), u))', "A_Numbers.test_figure_values"),
    ("figure from patience 12", "    p, t = PRIMARY\n    items = []", '    p, t = (12, "fixed_0.5")\n    items = []',
     "A_Numbers.test_figure_values"),
    (".tex from another condition", '    p, t = PRIMARY\n    out = [f"% Table 1', '    p, t = (4, "dev_eer")\n    out = [f"% Table 1',
     "A_Numbers.test_tex"),
    ("per-seed intervals taken from all seeds", '                        q = an["scopes"][scope][(quantity, n, st)]',
     '                        q = an["scopes"]["all seeds"][(quantity, n, st)]', "A_Numbers.test_per_seed"),
    ("unit counts above and below swapped in the tables",
     """        line = f"| {n} | {cells} | {r['above_zero']} / {r['below_zero']} / {r['zero']} |\"""",
     """        line = f"| {n} | {cells} | {r['below_zero']} / {r['above_zero']} / {r['zero']} |\"""",
     "A_Numbers.test_unit_counts_and_tie"),
    ("attack and bona fide swapped in the test one-class table", """                extra.append(f"{e['all_attack']} attack")""",
     """                extra.append(f"{e['all_attack']} bona fide")""", "A_Numbers.test_single_class"),
    ("unit rows of the contrasts computed as variations",
     "                        v = unit_value(U, p, t, u, terms_of(quantity, n, st if st in STATES else None))",
     '                        v = unit_value(U, p, t, u, terms_of("variation" if quantity == "contrast" else quantity, n, '
     "st if st in STATES else None))", "A_Numbers.test_contrast_reference_is_livenessnet"),
    ("gaps: BN states swapped in Appendix C",
     '            for quantity, st, label in (("gap_vs_resnet18", "ordinary", "gap, ordinary BN"),',
     '            for quantity, st, label in (("gap_vs_resnet18", "re-estimated", "gap, ordinary BN"),',
     "A_Numbers.test_descriptive_values"),
    ("descriptive variations taken from the primary analysis",
     '                *quantity_table(agg, p, t, "variation", NETWORKS, "all seeds", "variation of"), "",',
     '                *quantity_table(agg, *PRIMARY, "variation", NETWORKS, "all seeds", "variation of"), "",',
     "A_Numbers.test_descriptive_values"),
    ("Appendix A from patience 12", '    p, t = PRIMARY\n    out = [\n        "# Appendix A.',
     '    p, t = (12, "fixed_0.5")\n    out = [\n        "# Appendix A.', "A_Numbers.test_per_seed"),
    ("figure labels with the bounds swapped",
     'ax.text(0.0, yy + LABEL_OFFSET, f"{n}: {fmt_u(mu_x, True)} [{fmt_u(low_x, True)}, {fmt_u(high_x, True)}]",',
     'ax.text(0.0, yy + LABEL_OFFSET, f"{n}: {fmt_u(mu_x, True)} [{fmt_u(high_x, True)}, {fmt_u(low_x, True)}]",',
     "A_Numbers.test_figure_labels"),
    ("thousands separator dropped in the tables", '    return f"{int(x):,}" if x else x', "    return x",
     "A_Numbers.test_thousands"),
    ("distinct test frames counted once per unit", "({sum(f[k][0] for k in FOLDS):,} distinct test frames",
     "({sum(f[u[0]][0] for u in UNITS):,} distinct test frames", "A_Numbers.test_denominators_text"),
    # aggiunte della revisione 1: conteggi sul dev dai record dei run
    ("pin of a run status not checked (every run)", '    if h != pin:\n        raise Refusal(f"status.json of {rid}',
     '    if False:\n        raise Refusal(f"status.json of {rid}', "C_Runs.test_run_status_pin"),
    ("pin of a C2 run status not checked", "        pin = C2_RUNS[p][1]", "        pin = h", "C_Runs.test_c2_run_pin"),
    ("runs entry of results.json not compared with status.json",
     '        if not same(want, get(st, f"runs|{run_key(key)}", "status.json")) or not isinstance(want, dict) \\',
     "        if not isinstance(want, dict) \\", "C_Runs.test_runs_entry"),
    ("public SHA256SUMS of a run not compared with its status",
     '("run", run_key(key)), ("public_sha256sums_sha256", sha256(r["SHA256SUMS"]))):', '("run", run_key(key))):',
     "C_Runs.test_run_sums_against_status"),
    ("status of a run not required PASS",
     '    for path, value in (("run_id", rid), ("status", "PASS"), ("passed", True), ("official", True),',
     '    for path, value in (("run_id", rid), ("passed", True), ("official", True),', "C_Runs.test_run_status_fields"),
    ("a CSV of a run not checked against its line", "            if sha256(b) != rl[name]:",
     "            if False:", "C_Runs.test_run_file_against_its_line"),
    ("one_class = attack + bona fide not checked",
     '        if vals["one_class"] != vals["all_attack"] + vals["all_bona_fide"]:', "        if False:",
     "C_Runs.test_one_class_rules"),
    ("one-class counts not cross-checked with models.csv", "        if any(vals[k] != v for k, v in got.items()):",
     "        if False:", "C_Runs.test_one_class_against_models"),
    ("evaluable models not compared with units.csv", "        if units != set(pres[(n, p)]):", "        if False:",
     "C_Runs.test_evaluable_against_units"),
    ("the restricted folder of a run looked up",
     '        check_dir(os.path.join(folder, "public"), f"the public directory of {rid}")',
     '        check_dir(os.path.join(folder, "public"), f"the public directory of {rid}")\n'
     '        os.path.lexists(os.path.join(folder, "restricted"))', "B_Integrity.test_restricted_never_opened"),
    ("output inside a directory of the runs accepted",
     "        if inside(parent, rr) or (os.path.lexists(out) and inside(out, rr)):", "        if False:",
     "D_Paths.test_output_inside_runs"),
    ("text of one_class.csv not checked",
     """        if x["text"] != f"{vals['one_class']} of {vals['evaluable']} evaluable":""", "        if False:",
     "C_Runs.test_one_class_rules"),
    ("evaluable + not_evaluable = 12 not checked", '        if vals["evaluable"] + vals["not_evaluable"] != N_UNITS:',
     "        if False:", "C_Runs.test_one_class_rules"),
    ("rows of one_class.csv not checked",
     '    if [(x["network"], x["patience"], x["bn_state"]) for x in rows] != want_rows:', "    if False:",
     "C_Runs.test_one_class_rules"),
    ("network of the status.json of an AttackNet run not checked",
     '    if net != "C2":\n        expect(rs, "network", net, f"status.json of {rid}")',
     '    if False:\n        expect(rs, "network", net, f"status.json of {rid}")', "C_Runs.test_run_status_fields"),
    ("dev and test one-class models not compared", "    if test and test == dev:", "    if test:",
     "A_Numbers.test_dev_same_text_only_when_equal"),
    ("dev counts: attack and bona fide swapped in the appendix",
     """            cells.append(f"{e['one_class']} of {e['models']} evaluable ({e['all_attack']} attack, \"""",
     """            cells.append(f"{e['one_class']} of {e['models']} evaluable ({e['all_bona_fide']} attack, \"""",
     "A_Numbers.test_single_class_dev"),
    ("dev counts: patience 4 and 12 swapped", "            e = counts[(n, p, st)]", "            e = counts[(n, 16 - p, st)]",
     "A_Numbers.test_single_class_dev"),
    # aggiunte della revisione 1: arrotondamenti, unità tolta, per fold
    ("rounding note computed from the exact values", "    def rp(r, m):\n        return points(val(r, m))",
     "    def rp(r, m):\n        return val(r, m)", "A_Numbers.test_rounding_note"),
    ("leaving one unit out: the unit not left out", "        keep = [u for u in UNITS if u != left]",
     "        keep = list(UNITS)", "A_Numbers.test_leave_one_out"),
    ("leaving one unit out: mean divided by 12", "    return sum(vals, Fraction(0)) / len(vals), vals",
     "    return sum(vals, Fraction(0)) / N_UNITS, vals", "A_Numbers.test_leave_one_out"),
    ("per fold: the units of another fold", "        units = [u for u in UNITS if u[0] == f]",
     "        units = [u for u in UNITS if u[0] == f % 4 + 1]", "A_Numbers.test_per_fold"),
    # percorsi, scrittura, opzioni
    ("output inside the results package accepted", "    if root is not None and inside(parent, root):",
     "    if False:", "D_Paths.test_output_inside_record_or_results_package"),
    ("output inside the record accepted", "    if inside(parent, record):", "    if False:",
     "D_Paths.test_output_inside_record_or_results_package"),
    ("output inside this package accepted", "    if inside(parent, str(HERE)) and not", "    if False and not",
     "D_Paths.test_output_inside_this_package"),
    ("existing output accepted", '    if os.path.lexists(out):\n        raise Refusal(f"the output directory already exists',
     '    if False:\n        raise Refusal(f"the output directory already exists', "D_Paths.test_output_exists_or_symlink"),
    ("repeated options accepted", "        if n > 1:\n", "        if False:\n", "D_Paths.test_options"),
    ("directory to check inside the record accepted", "            if inside(against, record):", "            if False:",
     "E_Check.test_against_refusals"),
    ("malformed structures not turned into a clean refusal",
     "    except (AttributeError, KeyError, TypeError, ValueError, IndexError) as e:", "    except ZeroDivisionError as e:",
     "C_Header.test_malformed_structure_fails_cleanly"),
    ("top level of a JSON file not checked",
     '    if not isinstance(d, dict):\n        raise Refusal(f"{name}: the top level',
     '    if False:\n        raise Refusal(f"{name}: the top level', "C_Header.test_json_parsing"),
    ("duplicate JSON keys accepted", '        if k in d:\n            raise Refusal(f"duplicate key',
     '        if False:\n            raise Refusal(f"duplicate key', "C_Header.test_json_parsing"),
    ("output files not created exclusively", "os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644",
     "os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644", "D_Paths.test_output_created_without_following_links"),
    ("temporary directory left after a failure", "        shutil.rmtree(tmp, ignore_errors=True)", "        pass",
     "D_Paths.test_failure_while_writing_leaves_nothing"),
    ("output directory left in mode 0700", "        os.chmod(tmp, 0o755)\n", "        pass\n",
     "D_Paths.test_write_next_to_output_mode_and_handlers"),
    ("signal handlers not restored", "    for s, h in old.items():\n        signal.signal(s, h)",
     "    for s, h in []:\n        signal.signal(s, h)", "D_Paths.test_write_next_to_output_mode_and_handlers"),
    ("temporary directory not next to the output", 'tempfile.mkdtemp(prefix=".mtf-", dir=parent)',
     'tempfile.mkdtemp(prefix=".mtf-")', "D_Paths.test_write_next_to_output_mode_and_handlers"),
    ("signals not recorded", '        state["received"].append(signum)', "        pass",
     "S_Signals.test_fake_before_reservation"),
    ("output name not reserved before the rename", "            os.mkdir(out, 0o755)\n", "            pass\n",
     "D_Paths.test_two_runs_on_the_same_folder"),
    ("ENOTDIR and EISDIR not treated as the race",
     "RACE_ERRNOS = (errno.EEXIST, errno.ENOTEMPTY, errno.ENOTDIR, errno.EISDIR)",
     "RACE_ERRNOS = (errno.EEXIST, errno.ENOTEMPTY)", "D_Paths.test_reserved_name_replaced_by_a_file"),
    ("reservation not recognised by inode before the rename",
     'if _identity(out) != state["reserved_id"] or os.listdir(out):', "if False:",
     "D_Paths.test_reservation_replaced_by_another_empty_directory"),
    ("completion decided without the inode of the output",
     'return state.get("tmp_id") is not None and _identity(out) == state["tmp_id"]',
     'return state.get("tmp_id") is not None', "D_Paths.test_temporary_directory_replaced_before_the_rename"),
    ("temporary directory removed by path, not by identity",
     'if tmp is not None and state.get("tmp_id") is not None and _identity(tmp) == state["tmp_id"]:',
     "if tmp is not None:", "D_Paths.test_temporary_directory_of_another_process_never_removed"),
    ("no-signal failure reported without its message", "    elif failure is None:\n",
     "    elif failure is None and False:\n", "D_Paths.test_output_moved_away_after_the_rename"),
    ("signal received after completion not reported",
     'note = f"; signal {sigs[0]} received, handled after the outputs were complete" if sigs else ""', 'note = ""',
     "S_Signals.test_fake_after_rename"),
    ("own empty reservation not removed after an error", "            os.rmdir(out)\n", "            pass\n",
     "D_Paths.test_cleanup_of_the_reservation_after_an_error"),
    ("no check of the signals between the reservation and the rename",
     '        state["reserved_id"] = _identity(out)\n        _poll(state)\n',
     '        state["reserved_id"] = _identity(out)\n', "S_Signals.test_fake_after_mkdir"),
    ("PNG structure not checked under --text-only", "ok = png_structure_ok(delivered) if", "ok = True if",
     "E_Check.test_text_only_checks_image_structure"),
    ("PDF structure not checked under --text-only", "else pdf_structure_ok(delivered)", "else True",
     "E_Check.test_text_only_checks_image_structure"),
    ("bytes after the PNG IEND chunk accepted", "            return end == len(b)", "            return True",
     "E_Check.test_text_only_checks_image_structure"),
    ("matplotlib defaults not restored (a local matplotlibrc changes the figure)", "        matplotlib.rcdefaults()\n",
     "", "G_Determinism.test_local_matplotlibrc_ignored"),
    ("creation date written in the PDF", 'metadata={"Creator": None, "Producer": None, "CreationDate": None}',
     'metadata={"Creator": None, "Producer": None}', "G_Determinism.test_images_without_dates_or_software"),
]
# mutazione di controllo: deve SOPRAVVIVERE
CONTROL = [("control (must survive): a comment changed",
            "# regole (PROTOCOL.md §6 (c): un estremo a zero contiene lo zero)",
            "# regole (PROTOCOL.md §6 (c): un estremo esattamente a zero contiene lo zero)", None)]
NEEDS_MPL = ("matplotlib defaults not restored (a local matplotlibrc changes the figure)",
             "creation date written in the PDF", "figure labels with the bounds swapped")
# mutazioni equivalenti: elencate con il motivo, non eseguite né contate
EQUIVALENT = [("SHA256SUMS decoded as UTF-8, not ASCII", '        text = sums.decode("ascii")',
               "equivalent: a byte that is not UTF-8 raises the same UnicodeDecodeError, refused with the same message, "
               "and a non-ASCII UTF-8 character can only be in a line, which must match the ASCII pattern of a "
               "SHA-256 and a file name; the same SHA256SUMS is refused either way")]


def suite_tests():
    """Le coppie classe.metodo dei test della suite, lette dal sorgente."""
    out, cls = set(), None
    for line in (HERE / SUITE).read_text(encoding="utf-8").splitlines():
        m = re.match(r"class (\w+)\(", line)
        if m:
            cls = m.group(1)
        m = re.match(r"    def (test_\w+)\(", line)
        if m and cls:
            out.add(f"{cls}.{m.group(1)}")
    return out


def copy_package(dst):
    shutil.copytree(HERE, dst / "pad-attacknet-eval-paper", ignore=shutil.ignore_patterns("test_logs", ".DS_Store"))
    return dst / "pad-attacknet-eval-paper"


def run_suite(pkg, tmp):
    tmp.mkdir(parents=True, exist_ok=True)
    env = {"PATH": "/usr/bin:/bin", "TMPDIR": str(tmp), "LC_ALL": "C.UTF-8", "HOME": str(tmp)}
    r = subprocess.run([sys.executable, "-I", "-B", str(pkg / SUITE)], cwd=str(tmp), env=env, capture_output=True,
                       text=True, timeout=3600)
    line = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith("tests run")), "no test summary")
    skips = tuple(sorted(ln for ln in r.stdout.splitlines() if ln.startswith("skipped ")))
    # i test falliti (unittest li scrive su stderr): solo classe e metodo, nessun percorso
    failed = []
    for m in re.finditer(r"^(?:FAIL|ERROR): (\w+) \((?:__main__\.)?([\w.]+)\)", r.stderr, re.M):
        name = f"{m.group(2).rsplit('.', 1)[0].rsplit('.', 1)[-1]}.{m.group(1)}"
        if name not in failed:
            failed.append(name)
    return r.returncode, line, skips, failed


def verdict(rc, line, skips, base_skips, failed, expected):
    m = re.match(r"tests run: (\d+), failures: (\d+), errors: (\d+), skipped: (\d+)", line)
    if not m:
        return "invalid"
    if int(m.group(2)) + int(m.group(3)) > 0:
        return "killed" if expected is None or expected in failed else "invalid"
    return "survived" if rc == 0 and skips == base_skips else "invalid"


def one(base, i, mutant, base_skips):
    name, old, new, expected = mutant
    work = base / f"m{i}"
    pkg = copy_package(work)
    f = pkg / TARGET
    text = f.read_text(encoding="utf-8")
    f.write_text(text.replace(old, new, 1), encoding="utf-8")
    rc, line, skips, failed = run_suite(pkg, work / "tmp")
    shutil.rmtree(work, ignore_errors=True)
    v = verdict(rc, line, skips, base_skips, failed, expected)
    note = f"{line}; first failing: {failed[0] if failed else 'none'}"
    if expected:
        note += f"; declared: {expected}" + ("" if expected in failed else " (not among the failing tests)")
    return name, v, note


def main():
    ap = argparse.ArgumentParser(description="Mutation check of make_table_figure.py (synthetic, offline)")
    ap.add_argument("--list", action="store_true", help="list the mutations and check the patterns and declared tests")
    ap.add_argument("--only", default=None, help="only these mutations (numbers from 1, comma separated; c = control)")
    ap.add_argument("--jobs", type=int, default=1, help="mutations run at the same time (default 1)")
    a = ap.parse_args()
    if _inside(tempfile.gettempdir(), HERE):
        print("ERROR: TMPDIR is inside this package; set it elsewhere")
        print("RESULT: FAIL")
        return 1
    src = (HERE / TARGET).read_text(encoding="utf-8")
    tests = suite_tests()
    problems = [f"{n}: the pattern occurs {src.count(o)} times in {TARGET} (expected exactly once)"
                for n, o, _, _ in MUTANTS + CONTROL if src.count(o) != 1]
    problems += [f"{n}: the declared test {e} is not in {SUITE}" for n, _, _, e in MUTANTS if e not in tests]
    problems += [f"{n} (equivalent): the pattern occurs {src.count(o)} times in {TARGET}" for n, o, _ in EQUIVALENT
                 if src.count(o) != 1]
    problems += [f"{n}: the replacement equals the pattern" for n, o, w, _ in MUTANTS + CONTROL if o == w]
    names = [m[0] for m in MUTANTS + CONTROL]
    problems += [f"duplicated mutation name: {n}" for n in sorted({n for n in names if names.count(n) > 1})]
    for p in problems:
        print(f"ERROR: {p}", file=sys.stderr)
    if problems:
        print("RESULT: FAIL (a declared mutation no longer matches the code or the suite)")
        return 1
    if a.list:
        for i, (n, _, _, e) in enumerate(MUTANTS, start=1):
            print(f"{i:3d} {n}  [killed by {e}]")
        for n, *_ in CONTROL:
            print(f"    {n}")
        for n, _, why in EQUIVALENT:
            print(f"    {n} (not run, not counted): {why}")
        print(f"{len(MUTANTS)} mutations, {len(CONTROL)} control mutation and {len(EQUIVALENT)} declared equivalent, "
              f"every pattern found exactly once, every declared test present in the suite")
        return 0
    tokens = None if a.only is None else [x.strip() for x in a.only.split(",") if x.strip()]
    chosen = [(i, m) for i, m in enumerate(MUTANTS, start=1)
              if tokens is None or str(i) in tokens]
    controls = CONTROL if tokens is None or "c" in tokens else []
    try:
        import matplotlib  # noqa: F401
        has_mpl = True
    except ImportError:
        has_mpl = False
    na = [(i, m) for i, m in chosen if m[0] in NEEDS_MPL and not has_mpl]
    chosen = [x for x in chosen if x not in na]
    for i, m in na:
        print(f"NOT VERIFIABLE {i:3d} {m[0]} (its test needs matplotlib, not installed)", flush=True)
    base = Path(tempfile.mkdtemp(prefix="mtf-mutations-"))
    try:
        rc, line, skips, failed = run_suite(copy_package(base / "control_copy"), base / "control_copy" / "tmp")
        print(f"unmutated copy: exit {rc} ({line})", flush=True)
        for s in skips:
            print(f"unmutated copy: {s}", flush=True)
        if rc or not re.fullmatch(r"tests run: \d+, failures: 0, errors: 0, skipped: \d+", line):
            print("RESULT: FAIL (the suite does not pass on the unmutated copy)")
            return 1
        if any(s.split(": ", 1)[-1] not in SKIP_OK for s in skips):
            print("RESULT: FAIL (the unmutated copy skips tests for a reason that is not a declared missing input)")
            return 1
        for n, _, why in EQUIVALENT:
            print(f"EQUIVALENT {n} (not run, not counted): {why}", flush=True)
        survived_controls = 0
        for j, m in enumerate(controls):
            name, v, note = one(base, 1000 + j, m, skips)
            survived_controls += v == "survived"
            print(f"CONTROL   {v.upper():<9} {name} ({note})", flush=True)
        killed = invalid = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
            futures = [(i, ex.submit(one, base, i, m, skips)) for i, m in chosen]
            for i, fut in futures:
                name, v, note = fut.result()
                killed += v == "killed"
                invalid += v == "invalid"
                print(f"{v.upper():<9} {i:3d} {name} ({note})", flush=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    total = len(chosen)
    print(f"mutations killed: {killed}/{total}" + (f" ({invalid} invalid)" if invalid else "")
          + ("" if total + len(na) == len(MUTANTS) else f" (selection of {total + len(na)} of {len(MUTANTS)})")
          + (f" ({len(na)} not verifiable without matplotlib, never counted as killed)" if na else ""))
    if controls:
        print(f"control mutations survived: {survived_controls}/{len(controls)}")
    ok = killed == total and survived_controls == len(controls)
    print("RESULT: " + ("PASS" if ok and not na else "INCOMPLETE" if ok else "FAIL"))
    return 0 if ok and not na else (2 if ok else 1)


if __name__ == "__main__":
    sys.exit(main())
