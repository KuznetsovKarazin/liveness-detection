"""
Controllo delle mutazioni del pilota C2, offline e deterministico.

Copia il codice necessario (scripts/ del pilota e dei moduli M2 che importa, src/tesi_app/ con paths, evaluation, core
e face, e la configurazione approvata se presente) in una cartella temporanea, applica una mutazione dichiarata alla
volta a scripts/c2_pilot.py o agli adattamenti di scripts/train_cnn.py (coppie cerca/sostituisci che devono comparire
esattamente una volta: se un modello non si trova più, il controllo si ferma con un errore), esegue
scripts/check_c2_pilot.py sulla copia mutata e conta le mutazioni rilevate. Una mutazione è rilevata (killed) solo se
la suite gira fino al riepilogo con almeno un fallimento o errore; un test saltato non è mai una rilevazione; una suite
che non parte (import o sintassi rotti) rende la mutazione non valida. Codice 1 se una mutazione sopravvive o non è
valida, o se un modello non si trova.

Nessuna rete, nessun dato, nessun TensorFlow (la suite usa un modulo finto). Funziona dalla radice del repository di
sviluppo e da livenesslab/ del pacchetto pubblico.
  python scripts/check_c2_mutations.py            # tutte le mutazioni (qualche minuto)
  python scripts/check_c2_mutations.py --list     # solo l'elenco, con la verifica dei modelli
  python scripts/check_c2_mutations.py --block tensorflow,keras,h5py,cv2   # suite con questi pacchetti resi assenti
  python scripts/check_c2_mutations.py --jobs 1     # una mutazione alla volta (default: fino a 4 in parallelo)
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
PILOT = "scripts/c2_pilot.py"
TRAIN = "scripts/train_cnn.py"
TESTS = "scripts/check_c2_pilot.py"
CODE = ["scripts/c2_pilot.py", "scripts/check_c2_pilot.py", "scripts/check_c2_mutations.py", "scripts/train_cnn.py", "scripts/select_threshold.py", "scripts/path_guard.py",
        "scripts/rgb_manifests.py", "scripts/nuaa_manifests.py", "scripts/package_rules.py", "src/tesi_app/__init__.py",
        "src/tesi_app/paths.py", "src/tesi_app/evaluation.py", "src/tesi_app/core.py", "src/tesi_app/face.py"]
# m2_writers.py: importato da rgb_manifests.py e nuaa_manifests.py del repository di sviluppo dalla manutenzione dei writer
# del 22/10/2026; le copie del pacchetto C2 accettato non lo importano, quindi è copiato solo se presente
OPTIONAL = ["delivery/c2/pilot/c2_pilot_casia_to_nuaa.json", "scripts/m2_writers.py"]

PILOT_MUTANTS = [
    # check: manifest, conteggi, disgiunzioni, cache
    ("check: manifest SHA-256 not compared", '        if got != e["sha256"]:\n            fail(f"manifest', '        if False:\n            fail(f"manifest'),
    ("check: counts not compared with the configuration", "            if c[f] != e[f]:\n", "            if False:\n"),
    ("check: overlap without subject", 'for dim in ("key", "sha256", "subject_key", "video_key"):', 'for dim in ("key", "sha256", "video_key"):'),
    ("check: overlap without SHA-256", 'for dim in ("key", "sha256", "subject_key", "video_key"):', 'for dim in ("key", "subject_key", "video_key"):'),
    ("check: mirror split not checked", "        if wrong:\n            fail(f\"{role}: {wrong} rows outside", "        if False:\n            fail(f\"{role}: {wrong} rows outside"),
    ("check: cross-dataset files not compared", "    if casia & nuaa:\n", "    if False:\n"),
    ("check: derived frames not compared", "        if out[k]:\n", "        if False:\n"),
    ("check: label array not linked", "not np.array_equal(y, exp_y) or ", ""),
    ("check: group array not linked", ' or s.dtype.kind != "U" or not np.array_equal(s, exp_s)', ""),
    ("check: cache SHA-256 not compared", '        if got != df[side]["sha256"]:\n', "        if False:\n"),
    # configurazione
    ("config: manifest name not checked", '            if e.get("name") != f"{ds}/{name}":\n', "            if False:\n"),
    ("config: checkpoint field in a model accepted", 'if not isinstance(m, dict) or set(m) != {"id", "arch"}:', "if not isinstance(m, dict):"),
    ("config: criterion not checked", 't.get("criterion") != "eer" or ', ""),
    ("config: pooled list not required", '            if len(hashes) != 2 or not all(', '            if False and not all('),
    # cartella del run e stato del repository
    ("run folder: path guard skipped", "    msg = pg.deletion_problem(run, ROOT, pg.BASE_PROTECTED, [ROOT / \"results\" / \"c2\"])\n",
     '    msg = ""\n'),
    ("run folder: moved protected folders accepted", "        if pg.within(run, q) or (Path(q).exists() and pg.contains(run, q)):\n", "        if False:\n"),
    ("run folder: non-trial reused", "    if run.is_dir() and not trial and any(not (e.name == STATUS and e.is_file() and not e.is_symlink()) for e in run.iterdir()):\n",
     "    if False:\n"),
    ("run folder: status.json directory or link counts as new", "(e.name == STATUS and e.is_file() and not e.is_symlink())", "(e.name == STATUS)"),
    ("non-trial: any configuration accepted", "        if cfg_sha != APPROVED_CONFIG_SHA256:\n", "        if False:\n"),
    ("non-trial: dirty tree accepted at check", '        if state.get("dirty") or not str(state.get("commit", "")).startswith(m.group(2)):\n', "        if False:\n"),
    ("steps: configuration SHA-256 not compared with check", '    if check.get("config_sha256") != cfg_sha:\n', "    if False:\n"),
    ("steps: code SHA-256 not compared with check", '    if check.get("code_sha256") != code_hashes():\n', "    if False:\n"),
    ("steps: commit not compared with check", '        if state.get("dirty") or state.get("commit") != check.get("commit"):\n', "        if False:\n"),
    # uscite
    ("outputs: downstream not invalidated", "    for s in STEPS[STEPS.index(step):]:\n", "    for s in STEPS[STEPS.index(step):STEPS.index(step) + 1]:\n"),
    ("outputs: deletion without path guard",
     '                    pg.safe_rmtree(d, ROOT, pg.BASE_PROTECTED, [ROOT / "results" / "c2"], ignore_errors=False)\n                    removed.append(d.name)',
     '                    __import__("shutil").rmtree(d)\n                    removed.append(d.name)'),
    # train
    ("train: authorisation not required", '    if not a.i_am_authorized:\n        fail("train needs', '    if False:\n        fail("train needs'),
    ("train: epochs not compared with the budget", '    if a.epochs != b["max_epochs"]:\n', "    if False:\n"),
    ("train: test frames passed to the trainer", 'data = {"X_train": z["X_train"], "y_train": z["y_train"], "s_train": z["s_train"]}',
     "data = {k: z[k] for k in z.files}"),
    ("train: split check not passed", "out_dir=tmp, split_check=split_check, test_eval=False)", "out_dir=tmp, split_check=None, test_eval=False)"),
    ("train: weights check after training skipped", '        if problems:\n            fail("weights check after training failed', '        if False:\n            fail("weights check after training failed'),
    ("train: temporary checkpoints kept on failure",
     '                pg.safe_rmtree(d, ROOT, pg.BASE_PROTECTED, [ROOT / "results" / "c2"], ignore_errors=False)\n        raise', "                pass\n        raise"),
    ("train: wall-time cap not applied", '        if rec["wall_seconds"] > 60 * b["wall_time_cap_minutes"]:\n', "        if False:\n"),
    # controllo dei pesi
    ("weights: file SHA-256 not compared", '        if w != card.get("weights_sha256") or w != e.get("weights_sha256"):\n', "        if False:\n"),
    ("weights: card SHA-256 not compared", '        if sha256_file(card_p) != e.get("card_sha256"):\n', "        if False:\n"),
    ("weights: existing or pooled weights accepted", "        if w in listed:\n", "        if False:\n"),
    ("weights: test evaluation in the card accepted", '        if "test" in card or "n_test" in card:\n', "        if False:\n"),
    ("weights: restored epoch not checked", 'card.get("restored_epoch") != best or ', ""),
    ("weights: card fields not compared", "            if card.get(k) != v:\n", "            if False:\n"),
    # score
    ("score: weights not checked before scoring", "    train_rec = checked_weights(ctx, inp)\n    files = {}", '    train_rec = load_record(ctx.run, "train.json")\n    files = {}'),
    ("score: threshold.json not recomputed before test scoring", "        verify_threshold(ctx, inp, train_rec)\n        scorers = make_scorers",
     "        scorers = make_scorers"),
    ("score: threshold.json change during scoring accepted", "        if sha256_file(thr_p) != thr_sha:\n", "        if False:\n"),
    ("score: raw threshold not null", '"threshold": NULL, "prediction": NULL, "checkpoint_sha256"', '"threshold": "0.5", "prediction": NULL, "checkpoint_sha256"'),
    ("score: scores not round-trip", '"score_attack": repr(float(v))', '"score_attack": f"{float(v):.6f}"'),
    ("score: invalid scorer output accepted", "        if s.shape != (len(keys),) or not np.all(np.isfinite(s)) or np.any(s < 0) or np.any(s > 1):\n", "        if False:\n"),
    # select e validazione dei punteggi
    ("select: test and NUAA rows not refused", '    if role == "source_dev" and (msg := test_rows_in_selection(rows, inp)):\n        fail(msg)',
     "    if False:\n        fail('')"),
    ("select: refusal by key only", '    by_sha = sum(1 for r in rows if r.get("file_sha256") in hashes)', "    by_sha = 0"),
    ("select: refusal by SHA-256 only", '    by_key = sum(1 for r in rows if r.get("sample_id") in keys)', "    by_key = 0"),
    ("scores: missing rows accepted", "        if got != set(want):\n", "        if False:\n"),
    ("scores: duplicates accepted", "    if dup:\n        bad.append", "    if False:\n        bad.append"),
    ("scores: NaN and out-of-range accepted", "    if not math.isfinite(v) or not 0.0 <= v <= 1.0 or repr(v) != text:", "    if repr(v) != text:"),
    ("scores: fields not compared with the manifest", "        if any(r.get(c) != v for c, v in exp.items()):\n", "        if False:\n"),
    ("scores: checkpoint SHA-256 not compared", '        if r.get("checkpoint_sha256") != train_rec["models"][mid]["weights_sha256"]:\n', "        if False:\n"),
    ("scores: file SHA-256 not compared with the record", '    if (rec.get("files") or {}).get(name, {}).get("sha256") != sha256_file(p):\n', "    if False:\n"),
    ("select: single-class dev not refused", "        if not (np.any(y == 0) and np.any(y == 1)):\n", "        if False:\n"),
    ("select: criterion apcer10", 'sel = st.select_threshold(y, s, "eer")', 'sel = st.select_threshold(y, s, "apcer10")'),
    ("public scan: identifiers accepted", '    if bad:\n        fail(f"{what}: identifiers', '    if False:\n        fail(f"{what}: identifiers'),
    # apply
    ("apply: tie counted as attack", '"prediction": str(int(float(r["score_attack"]) > t))', '"prediction": str(int(float(r["score_attack"]) >= t))'),
    ("apply: decision check with >=", 'r.get("prediction") != str(int(v > t))', 'r.get("prediction") != str(int(v >= t))'),
    ("apply: recorded threshold SHA-256 not compared", '    if st_rec.get("threshold_sha256_before_scoring") != thr_sha:\n', "    if False:\n"),
    ("apply: threshold.json not recomputed", "    values = verify_threshold(ctx, inp, train_rec)\n    raw = ",
     '    values = {m: e["value"] for m, e in json.loads(thr_p.read_text())["models"].items()}\n    raw = '),
    # revisione del 20/10: stato dopo un rifiuto, eccezioni impreviste, codice delle metriche, record, pesi, frame, tipi
    ("status: no FAIL marker after a refusal", '        write_status("FAIL", a.cmd, reason)\n', "        pass\n"),
    ("main: unexpected exceptions not caught", "    except Exception as e:      # noqa: BLE001", "    except ZeroDivisionError as e:      # noqa: BLE001"),
    ("code: evaluation.py not pinned", '"scripts/package_rules.py", "src/tesi_app/evaluation.py",', '"scripts/package_rules.py",'),
    ("code: paths.py not pinned", '              "src/tesi_app/paths.py")', "              )"),
    ("code: reference architectures not pinned", '    out[ARCH_KEY] = sha256_file(arch) if arch.is_file() else "absent"', '    out[ARCH_KEY] = "absent"'),
    ("status: RUNNING not written when the folder is accepted", '    write_status("RUNNING", RUN_STATE.get("cmd") or step)', "    pass"),
    ("main: KeyboardInterrupt not caught", "    except KeyboardInterrupt:   # interruzione", "    except ZeroDivisionError:   # interruzione"),
    ("main: PASS status write errors ignored", '        COMMANDS[a.cmd](a)\n        write_status("PASS", a.cmd)',
     '        COMMANDS[a.cmd](a)\n        try:\n            write_status("PASS", a.cmd)\n        except Exception:\n            pass'),
    ("status: written in a folder before the path check", "    RUN_STATE.update(step=step, removed=False)\n    if msg := run_dir_problem(run):",
     "    RUN_STATE.update(step=step, removed=False, run=run)\n    if msg := run_dir_problem(run):"),
    ("threshold: raw dev SHA-256 not compared", '    got = {k: v for k, v in doc.items() if k != "created"} if isinstance(doc, dict) else {}',
     '    got = {k: v for k, v in doc.items() if k != "created"} if isinstance(doc, dict) else {}\n    got["raw_scores_dev_sha256"] = want["raw_scores_dev_sha256"]'),
    ("dispersion: subject identifiers in the output", '        out[name] = {"n_subjects": len(vals),', '        out[name] = {"subjects": sorted(per), "n_subjects": len(vals),'),
    ("dispersion: wrong median", '"median": float(np.median(vals)) if vals else None', '"median": vals[len(vals) // 2] if vals else None'),
    ("run folder: moved m2_rgb results accepted", 'RESULTS / "m2", RESULTS / "m2_rgb", RESULTS / "eval"', 'RESULTS / "m2", RESULTS / "eval"'),
    ("run folder: folder containing a protected one accepted", "        if pg.within(run, q) or (Path(q).exists() and pg.contains(run, q)):\n            return f\"{run} is, contains",
     "        if pg.within(run, q):\n            return f\"{run} is, contains"),
    ("weights: best_val_loss not compared", ' or card.get("best_val_loss") != hist[best - 1]', ""),
    ("records: passed not required to be true", 'd.get("passed") is not True', 'not d.get("passed")'),
    ("steps: check.json of another run accepted", '    if check.get("run_id") != run.name:\n', "    if False:\n"),
    ("score: frames not selected by key", "    sel = np.array([idx[k] for k in keys], dtype=np.int64)", "    sel = np.arange(len(keys))"),
    ("attack type: wrong token mapping", '"5": "cut", "6": "cut"', '"5": "video", "6": "cut"'),
    ("weights: official non-pooled weights accepted", '    out = {}\n    for blk in ("excluded_pooled", "alternative_not_recommended"):',
     '    out = {}\n    for blk in ("excluded_pooled",):'),
    ("weights: Keras SHA-256 not compared", '        if sha256_file(ker) != card.get("keras_sha256") or card.get("keras_sha256") != e.get("keras_sha256"):\n',
     "        if False:\n"),
    ("weights: same weights for two models accepted", "        if w in seen:\n", "        if False:\n"),
    ("weights: split check of the card not required", '        if not str(card.get("split_check", "")).startswith("passed"):\n', "        if False:\n"),
    ("weights: epochs above the budget accepted", 'not 1 <= n_ep <= b["max_epochs"]', "not 1 <= n_ep"),
    ("scores: configuration SHA-256 of the rows not compared", '        if r.get("config_sha256") != ctx.cfg_sha:\n            cfgs += 1', "        if False:\n            cfgs += 1"),
    ("dispersion: attack errors counted as rejections", "int((v > t) if lab == 0 else (v <= t))", "int(v > t)"),
    # terza revisione (20/10)
    ("run folder: moved protected folders compared as strings", "        if pg.within(run, q) or (Path(q).exists() and pg.contains(run, q)):\n            return f\"{run} is, contains",
     "        if str(Path(run).resolve()).startswith(str(Path(q).resolve())):\n            return f\"{run} is, contains"),
    ("run folder: moved weights not protected", "    return [WEIGHTS, DATA, ", "    return [DATA, "),
    ("run folder: moved data not protected", "    return [WEIGHTS, DATA, ", "    return [WEIGHTS, "),
    ("run folder: moved results/c1 not protected", 'RESULTS / "c1", RESULTS / "m1"', 'RESULTS / "m1"'),
    ("run folder: moved results/m1 not protected", 'RESULTS / "m1", RESULTS / "m2", ', 'RESULTS / "m2", '),
    ("run folder: moved results/eval not protected", 'RESULTS / "m2_rgb", RESULTS / "eval"]', 'RESULTS / "m2_rgb"]'),
    ("run id: text after the commit accepted", '([0-9a-f]{7,40})(-trial)?$")', '([0-9a-f]{7,40})(-trial)?")'),
    ("weights: n_test in the card accepted", '        if "test" in card or "n_test" in card:\n', '        if "test" in card:\n'),
    ("select: threshold.json not scanned", '    public_scan(text, inp, "threshold.json")\n', ""),
    ("apply: Markdown table not scanned as Markdown", 'public_scan(mtext, inp, "c2_table.md", markdown=True)', 'public_scan(mtext, inp, "c2_table.md")'),
    ("check: early-stopping frames not compared with the source test", '"train_test_shared": len((fit | dev) & test)', '"train_test_shared": len(fit & test)'),
    ("check: overlap without key", 'for dim in ("key", "sha256", "subject_key", "video_key"):', 'for dim in ("sha256", "subject_key", "video_key"):'),
    ("apply: best attack type reported as the worst", '"apcer_worst_attack_type": max((v["apcer"]', '"apcer_worst_attack_type": min((v["apcer"]'),
    ("select: AUC reason missing", '("dev AUC < 0.5", m.get("auc") is not None and m["auc"] < 0.5)', '("dev AUC < 0.5", False)'),
    ("select: EER reason missing", '("operational dev EER > 0.5", op["acer"] is not None and op["acer"] > 0.5)', '("operational dev EER > 0.5", False)'),
    ("select: EER reason with >=", 'op["acer"] is not None and op["acer"] > 0.5)', 'op["acer"] is not None and op["acer"] >= 0.5)'),
    ("check: source/target frame shapes not compared", "    if len(shapes) != 1:\n", "    if False:\n"),
    ("check: number of frames not compared with the keys", " or X.shape[0] != len(keys) or X.shape[3] != 3:", " or X.shape[3] != 3:"),
    ("scores: BOM and CR accepted", '    if raw.startswith(b"\\xef\\xbb\\xbf") or b"\\r" in raw:\n', "    if False:\n"),
    ("scores: header not compared", "    if header != SCORE_COLUMNS:\n", "    if False:\n"),
    ("score: authorisation not required", '    if not a.i_am_authorized:\n        fail("score needs', '    if False:\n        fail("score needs'),
    ("select: threshold.json not read back", '                fail(f"{mid}: threshold.json read back differs from the selection")', "                pass"),
    # quarta revisione (20/10)
    ("select: weights not checked", '    train_rec = checked_weights(ctx, inp)\n    rows = check_raw(', '    train_rec = load_record(ctx.run, "train.json")\n    rows = check_raw('),
    ("apply: weights not checked", '    train_rec = checked_weights(ctx, inp)\n    st_rec = load_record(', '    train_rec = load_record(ctx.run, "train.json")\n    st_rec = load_record('),
    ("train: epochs not passed to the trainer", 'trainer(arch, "casia_fasd", data, a.epochs,', 'trainer(arch, "casia_fasd", data, 10,'),
    ("apply: descriptive EER wrong", '"eer": m.get("eer"), "bpcer_at_apcer10"', '"eer": m.get("auc"), "bpcer_at_apcer10"'),
    ("apply: descriptive BPCER@APCER10 dropped", '"bpcer_at_apcer10": m.get("bpcer_at_apcer10")}})', '"bpcer_at_apcer10": None}})'),
    ("apply: fixed 0.5 cell at the dev threshold", 'cell(compute_metrics, by_type, y, s, subj, types, 0.5, ds), "label"', 'cell(compute_metrics, by_type, y, s, subj, types, values[mid], ds), "threshold": 0.5, "label"'),
    ("dispersion: first model for every evaluation", "        mid = model_list(ctx.cfg)[i][0]", "        mid = model_list(ctx.cfg)[0][0]"),
    ("steps: dirty tree accepted after check", '        if state.get("dirty") or state.get("commit") != check.get("commit"):', '        if state.get("commit") != check.get("commit"):'),
    ("markdown: null written as 0", '    return "null" if v is None else', '    return "0" if v is None else'),
    ("threshold: wrong selector hash", 'sha256_file(ROOT / "scripts" / "select_threshold.py")', 'sha256_file(ROOT / "scripts" / "c2_pilot.py")'),
    ("public scan: NUAA hashes not scanned", '    ids["hashes"] |= {r["sha256"] for r in inp.rows["target_test"]}\n', ""),
    ("check: label array of any dtype", 'if y.dtype.kind not in "iu" or not np.array_equal(y, exp_y)', 'if not np.array_equal(y, exp_y)'),
    ("check: frames of any dtype", "        if X.dtype != np.uint8 or X.ndim != 4", "        if X.ndim != 4"),
    ("check: existing file as run folder not refused", '    if run.exists() and not run.is_dir():\n        fail("run folder path exists and is not a folder")\n', ""),
    # quinto giro (20/10)
    ('code: scripts/c2_pilot.py dropped from CODE_FILES', 'CODE_FILES = ("scripts/c2_pilot.py", "scripts/train_cnn.py", "scripts/select_threshold.py", "scripts/path_guard.py",\n              "scripts/rgb_manifests.py", "scripts/nuaa_manifests.py", "scripts/package_rules.py", "src/tesi_app/evaluation.py",\n              "src/tesi_app/paths.py")', 'CODE_FILES = ("scripts/train_cnn.py", "scripts/select_threshold.py", "scripts/path_guard.py", "scripts/rgb_manifests.py", "scripts/nuaa_manifests.py", "scripts/package_rules.py", "src/tesi_app/evaluation.py", "src/tesi_app/paths.py")'),
    ('code: scripts/train_cnn.py dropped from CODE_FILES', 'CODE_FILES = ("scripts/c2_pilot.py", "scripts/train_cnn.py", "scripts/select_threshold.py", "scripts/path_guard.py",\n              "scripts/rgb_manifests.py", "scripts/nuaa_manifests.py", "scripts/package_rules.py", "src/tesi_app/evaluation.py",\n              "src/tesi_app/paths.py")', 'CODE_FILES = ("scripts/c2_pilot.py", "scripts/select_threshold.py", "scripts/path_guard.py", "scripts/rgb_manifests.py", "scripts/nuaa_manifests.py", "scripts/package_rules.py", "src/tesi_app/evaluation.py", "src/tesi_app/paths.py")'),
    ('code: scripts/select_threshold.py dropped from CODE_FILES', 'CODE_FILES = ("scripts/c2_pilot.py", "scripts/train_cnn.py", "scripts/select_threshold.py", "scripts/path_guard.py",\n              "scripts/rgb_manifests.py", "scripts/nuaa_manifests.py", "scripts/package_rules.py", "src/tesi_app/evaluation.py",\n              "src/tesi_app/paths.py")', 'CODE_FILES = ("scripts/c2_pilot.py", "scripts/train_cnn.py", "scripts/path_guard.py", "scripts/rgb_manifests.py", "scripts/nuaa_manifests.py", "scripts/package_rules.py", "src/tesi_app/evaluation.py", "src/tesi_app/paths.py")'),
    ('code: scripts/path_guard.py dropped from CODE_FILES', 'CODE_FILES = ("scripts/c2_pilot.py", "scripts/train_cnn.py", "scripts/select_threshold.py", "scripts/path_guard.py",\n              "scripts/rgb_manifests.py", "scripts/nuaa_manifests.py", "scripts/package_rules.py", "src/tesi_app/evaluation.py",\n              "src/tesi_app/paths.py")', 'CODE_FILES = ("scripts/c2_pilot.py", "scripts/train_cnn.py", "scripts/select_threshold.py", "scripts/rgb_manifests.py", "scripts/nuaa_manifests.py", "scripts/package_rules.py", "src/tesi_app/evaluation.py", "src/tesi_app/paths.py")'),
    ('code: scripts/rgb_manifests.py dropped from CODE_FILES', 'CODE_FILES = ("scripts/c2_pilot.py", "scripts/train_cnn.py", "scripts/select_threshold.py", "scripts/path_guard.py",\n              "scripts/rgb_manifests.py", "scripts/nuaa_manifests.py", "scripts/package_rules.py", "src/tesi_app/evaluation.py",\n              "src/tesi_app/paths.py")', 'CODE_FILES = ("scripts/c2_pilot.py", "scripts/train_cnn.py", "scripts/select_threshold.py", "scripts/path_guard.py", "scripts/nuaa_manifests.py", "scripts/package_rules.py", "src/tesi_app/evaluation.py", "src/tesi_app/paths.py")'),
    ('code: scripts/nuaa_manifests.py dropped from CODE_FILES', 'CODE_FILES = ("scripts/c2_pilot.py", "scripts/train_cnn.py", "scripts/select_threshold.py", "scripts/path_guard.py",\n              "scripts/rgb_manifests.py", "scripts/nuaa_manifests.py", "scripts/package_rules.py", "src/tesi_app/evaluation.py",\n              "src/tesi_app/paths.py")', 'CODE_FILES = ("scripts/c2_pilot.py", "scripts/train_cnn.py", "scripts/select_threshold.py", "scripts/path_guard.py", "scripts/rgb_manifests.py", "scripts/package_rules.py", "src/tesi_app/evaluation.py", "src/tesi_app/paths.py")'),
    ('code: scripts/package_rules.py dropped from CODE_FILES', 'CODE_FILES = ("scripts/c2_pilot.py", "scripts/train_cnn.py", "scripts/select_threshold.py", "scripts/path_guard.py",\n              "scripts/rgb_manifests.py", "scripts/nuaa_manifests.py", "scripts/package_rules.py", "src/tesi_app/evaluation.py",\n              "src/tesi_app/paths.py")', 'CODE_FILES = ("scripts/c2_pilot.py", "scripts/train_cnn.py", "scripts/select_threshold.py", "scripts/path_guard.py", "scripts/rgb_manifests.py", "scripts/nuaa_manifests.py", "src/tesi_app/evaluation.py", "src/tesi_app/paths.py")'),
    ('code: src/tesi_app/evaluation.py dropped from CODE_FILES', 'CODE_FILES = ("scripts/c2_pilot.py", "scripts/train_cnn.py", "scripts/select_threshold.py", "scripts/path_guard.py",\n              "scripts/rgb_manifests.py", "scripts/nuaa_manifests.py", "scripts/package_rules.py", "src/tesi_app/evaluation.py",\n              "src/tesi_app/paths.py")', 'CODE_FILES = ("scripts/c2_pilot.py", "scripts/train_cnn.py", "scripts/select_threshold.py", "scripts/path_guard.py", "scripts/rgb_manifests.py", "scripts/nuaa_manifests.py", "scripts/package_rules.py", "src/tesi_app/paths.py")'),
    ('code: src/tesi_app/paths.py dropped from CODE_FILES', 'CODE_FILES = ("scripts/c2_pilot.py", "scripts/train_cnn.py", "scripts/select_threshold.py", "scripts/path_guard.py",\n              "scripts/rgb_manifests.py", "scripts/nuaa_manifests.py", "scripts/package_rules.py", "src/tesi_app/evaluation.py",\n              "src/tesi_app/paths.py")', 'CODE_FILES = ("scripts/c2_pilot.py", "scripts/train_cnn.py", "scripts/select_threshold.py", "scripts/path_guard.py", "scripts/rgb_manifests.py", "scripts/nuaa_manifests.py", "scripts/package_rules.py", "src/tesi_app/evaluation.py")'),
    ("check: NUAA frames not compared with the fitting frames", "(fit | dev | test) & set(h_tgt)", "(dev | test) & set(h_tgt)"),
    ("check: NUAA frames not compared with the early-stopping frames", "(fit | dev | test) & set(h_tgt)", "(fit | test) & set(h_tgt)"),
    ("train: wall-time cap in hours", 'rec["wall_seconds"] > 60 * b["wall_time_cap_minutes"]', 'rec["wall_seconds"] > 3600 * b["wall_time_cap_minutes"]'),
    ("train: wall-time cap with >=", 'rec["wall_seconds"] > 60 * b["wall_time_cap_minutes"]', 'rec["wall_seconds"] >= 60 * b["wall_time_cap_minutes"]'),
    ("records: CASIA session not null", '"subject_id": row["subject_key"], "session_id": NULL,', '"subject_id": row["subject_key"], "session_id": row["frame"],'),
    ("records: NUAA video not null", '"video_id": NULL, "attack_type": "print" if att else NULL', '"video_id": row["pic"], "attack_type": "print" if att else NULL'),
    ("records: NUAA subject from the key", '"subject_id": row["subject"], "session_id": row["session"],', '"subject_id": row["key"], "session_id": row["session"],'),
    ("apply: caches read", '    inp = load_inputs(ctx.cfg, a, caches=False)\n    train_rec = checked_weights(ctx, inp)\n    st_rec',
     '    inp = load_inputs(ctx.cfg, a, caches=True)\n    train_rec = checked_weights(ctx, inp)\n    st_rec'),
    ("check: trial folders held to the non-trial rules", "    if not trial:\n        if cfg_sha != APPROVED_CONFIG_SHA256:", "    if True:\n        if cfg_sha != APPROVED_CONFIG_SHA256:"),
    ("steps: trial folders held to the commit rule", "    if not trial:\n        state = repo_state(run)", "    if True:\n        state = repo_state(run)"),
    # sesto giro (21/10)
    ("score: last frames of a batch dropped", "            xb = X[i:i + SCORE_BATCH].astype(np.float32) / 255.0", "            xb = X[i:i + SCORE_BATCH - 1].astype(np.float32) / 255.0"),
    ("score: batches skip frames", "        for i in range(0, len(X), SCORE_BATCH):", "        for i in range(0, len(X), SCORE_BATCH + 1):"),
    ("score: batch size changed", "SCORE_BATCH = 16", "SCORE_BATCH = 17"),
    ("import: exit code 2 without numpy", '    print("RESULT: FAIL (import)")\n    sys.exit(1)', '    print("RESULT: FAIL (import)")\n    sys.exit(2)'),
    # settimo giro (21/10): opzioni della riga di comando
    ("cli: --epochs not required", '            p.add_argument("--epochs", type=int, required=True)', '            p.add_argument("--epochs", type=int, default=12)'),
    ("cli: --epochs as float", '            p.add_argument("--epochs", type=int, required=True)', '            p.add_argument("--epochs", type=float, required=True)'),
    ("cli: --set not required", '            p.add_argument("--set", required=True,', '            p.add_argument("--set", default="dev",'),
    ("cli: no authorisation flag for all", '        if name in ("train", "score", "all"):', '        if name in ("train", "score"):'),
    ("cli: source manifests default", 'default=str(RESULTS / "m2_rgb" / "casia_fasd" / "manifests")', 'default=str(RESULTS / "m2" / "casia_fasd" / "manifests")'),
    ("cli: target cache default", 'default=str(DATA / "processed" / "nuaa_1500.npz")', 'default=str(DATA / "processed" / "nuaa_all.npz")'),
    ("cli: --run-dir not required", '        p.add_argument("--run-dir", required=True,', '        p.add_argument("--run-dir", default=".",'),
    # revisione del docente del 21/10: ingressi e uscite separati, alberi esterni, temporanei sicuri, provenienza
    ("write_all: no cleanup", "    except BaseException:\n        for tp in tmps.values():\n            tp.unlink(missing_ok=True)\n        for n in done:\n            (run / n).unlink(missing_ok=True)\n        raise",
     "    except BaseException:\n        raise"),
    ("write_all: cleanup only on OSError", "    except BaseException:\n        for tp in tmps.values():", "    except OSError:\n        for tp in tmps.values():"),
    ("write_all: link outputs written through", "            if q.is_symlink() or (q.exists() and not q.is_file()):\n                raise C2Error(f\"{n} in the run folder is a link",
     "            if False:\n                raise C2Error(f\"{n} in the run folder is a link"),
    ("write_all: temporary with a fixed name", '            fd, tp = tempfile.mkstemp(prefix=f".{n}.", suffix=".tmp", dir=run)',
     '            tp = str(run / (n + ".tmp")); fd = os.open(tp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)'),
    ("score: tests scored without threshold.json", '    if step == "score_tests" and (thr_p.is_symlink() or not thr_p.is_file()):\n', "    if False:\n"),
    ("train: outputs removed before the option checks", '    ctx = open_run(a, "train", caches=True)\n    if not a.i_am_authorized:',
     '    ctx = open_run(a, "train", caches=True)\n    remove_outputs(ctx.run, "train")\n    if not a.i_am_authorized:'),
    ("score: outputs removed before the option checks", "    ctx = open_run(a, step, caches=True)\n    if not a.i_am_authorized:",
     "    ctx = open_run(a, step, caches=True)\n    remove_outputs(ctx.run, step)\n    if not a.i_am_authorized:"),
    ("check: outputs removed before the configuration checks", '    accept_run(run, "check")\n    cfg, cfg_sha = load_config(a.config)',
     '    accept_run(run, "check")\n    remove_outputs(run, "check")\n    cfg, cfg_sha = load_config(a.config)'),
    ("records: configuration SHA-256 not compared", '        if d.get("config_sha256") != ctx.cfg_sha:\n            fail(f"{name} was written', '        if False:\n            fail(f"{name} was written'),
    ("records: run_id not compared", '        if d.get("run_id") != ctx.run.name:\n', "        if False:\n"),
    ("records: step not compared", '        if d.get("step") != step:\n', "        if False:\n"),
    ("records: check.json of this run not compared", '        if d.get("check_sha256") != sha256_file(ctx.run / "check.json"):\n', "        if False:\n"),
    ("records: linked record accepted", '    if p.is_symlink() or not p.is_file():\n        fail(f"missing {name}', '    if not p.is_file():\n        fail(f"missing {name}'),
    ("weights: train.json read without provenance", '    rec = load_record(ctx.run, "train.json", ctx, "train")', '    rec = load_record(ctx.run, "train.json")'),
    ("weights: linked checkpoints accepted", "    if (ctx.run / CHECKPOINTS).is_symlink():\n", "    if False:\n"),
    ("scores: record read without provenance", '    rec = load_record(ctx.run, record, ctx, record.removesuffix(".json"))', "    rec = load_record(ctx.run, record)"),
    ("scores: train.json of the record not compared", '    if rec.get("train_sha256") != sha256_file(ctx.run / "train.json"):\n', "    if False:\n"),
    ("scores: score_dev.json of the test record not compared", '    if record == "score_tests.json" and rec.get("score_dev_sha256") != sha256_file(ctx.run / "score_dev.json"):\n', "    if False:\n"),
    ("apply: score_tests.json read without provenance", '    st_rec = load_record(ctx.run, "score_tests.json", ctx, "score_tests")', '    st_rec = load_record(ctx.run, "score_tests.json")'),
    ("threshold: threshold.json of another run accepted", '    if p.is_symlink() or not isinstance(doc, dict) or doc.get("run_id") != ctx.run.name:\n', "    if False:\n"),
    ("inputs: not checked before the steps", '    if msg := inputs_problem(a, run, caches):\n        fail(f"inputs refused: {msg}")\n    accept_run(run, step)',
     "    accept_run(run, step)"),
    ("inputs: not checked before check", '    if msg := inputs_problem(a, run, caches=True):\n        fail(f"inputs refused: {msg}")\n    if run.is_dir() and not trial',
     "    if run.is_dir() and not trial"),
    ("inputs: inside the run accepted", "        if pg.within(p, run) or pg.contains(p, run):\n            return f\"{opt} {p.name} is, contains",
     "        if pg.contains(p, run):\n            return f\"{opt} {p.name} is, contains"),
    ("inputs: ancestor of the run accepted", "        if pg.within(p, run) or pg.contains(p, run):\n            return f\"{opt} {p.name} is, contains",
     "        if pg.within(p, run):\n            return f\"{opt} {p.name} is, contains"),
    ("inputs: compared as strings", "        if pg.within(p, run) or pg.contains(p, run):\n            return f\"{opt} {p.name} is, contains",
     "        if str(Path(p).resolve()).startswith(str(Path(run).resolve())):\n            return f\"{opt} {p.name} is, contains"),
    ("inputs: missing configuration accepted", '        if opt == "--config" and not p.is_file():\n', "        if False:\n"),
    ("inputs: manifest folder not checked", '        if opt.endswith("manifests") and not p.is_dir():\n', "        if False:\n"),
    ("inputs: missing cache accepted", '        if opt.endswith("cache") and not p.is_file():\n', "        if False:\n"),
    ("inputs: caches not checked", '        out |= {"--source-cache": a.source_cache, "--target-cache": a.target_cache}', "        pass"),
    ("run folder: outer work tree accepted", "    for g in outer_roots():\n        if pg.within(run, g) or pg.contains(run, g):\n            return f\"{run} is inside",
     "    for g in []:\n        if pg.within(run, g) or pg.contains(run, g):\n            return f\"{run} is inside"),
    ("run folder: git work tree not detected", "        if top is not None and not pg.same(top, root):\n            out.append(top)\n    except (OSError, subprocess.SubprocessError):\n        pass\n    for x in root.parents:\n        if (x / \"research\").is_dir() and any(pg.same(x / \"research\", y) for y in root.parents) and not any(pg.same(x, o) for o in out):\n            out.append(x)\n    return tuple(out)",
     "        if False:\n            out.append(top)\n    except (OSError, subprocess.SubprocessError):\n        pass\n    for x in root.parents:\n        if (x / \"research\").is_dir() and any(pg.same(x / \"research\", y) for y in root.parents) and not any(pg.same(x, o) for o in out):\n            out.append(x)\n    return tuple(out)"),
    ("run folder: research layout not detected", "        if (x / \"research\").is_dir() and any(pg.same(x / \"research\", y) for y in root.parents) and not any(pg.same(x, o) for o in out):\n            out.append(x)\n    return tuple(out)",
     "        if False:\n            out.append(x)\n    return tuple(out)"),
    ("run folder: symbolic link accepted", "    if run.is_symlink():\n        return f\"{run.name} is a symbolic link", "    if False:\n        return f\"{run.name} is a symbolic link"),
    ("outputs: link removed through its target", "                if q.is_symlink():              # il collegamento si toglie, il suo bersaglio mai\n                    q.unlink(); removed.append(q.name)\n                elif q.is_dir():",
     "                if False:\n                    pass\n                elif q.is_dir():"),
    ("status: stale temporaries kept", '        if e == STATUS + ".tmp" or (e.startswith(f".{STATUS}.") and e.endswith(".tmp")):', "        if False:"),
    ("records: commit not compared", '        if d.get("commit") != ctx.check.get("commit"):\n', "        if False:\n"),
    ("records: commit of the record not written", '"check_sha256": sha256_file(ctx.run / "check.json"), "commit": ctx.check.get("commit")}', '"check_sha256": sha256_file(ctx.run / "check.json"), "commit_x": ctx.check.get("commit")}'),
    ("records: message without the other run id", "fail(f\"{name} belongs to another run ({d.get('run_id')!r})", "fail(f\"{name} belongs to another run ({d.get('step')!r})"),
    ("write_all: outputs written with another mode", "            os.chmod(tp, 0o666 & ~_UMASK)", "            os.chmod(tp, 0o600)"),
    ("status: any .tmp removed", '        if e == STATUS + ".tmp" or (e.startswith(f".{STATUS}.") and e.endswith(".tmp")):', '        if e == STATUS + ".tmp" or (e.startswith(f".{STATUS}.") or e.endswith(".tmp")):'),
    ("status: dangling temporary link kept", "            if q.is_symlink() or q.is_file():\n                q.unlink()\n    doc", "            if q.is_symlink() and q.is_file():\n                q.unlink()\n    doc"),
    ("outputs: folder with an output name not refused", '                    fail(f"{q.name} in the run folder is a folder, not an output file: refused")', "                    pass"),
    ("outputs: stale temporary files kept", '            temps = [e for e in entries if e == n + ".tmp" or (e.startswith(f".{n}.") and e.endswith(".tmp"))]', "            temps = []"),
    ("outputs: stale temporary checkpoints kept", '            dirs = [CHECKPOINTS, CHECKPOINTS + ".tmp"] + [e for e in entries if e.startswith(f".{CHECKPOINTS}.")]', "            dirs = [CHECKPOINTS]"),
    ("apply: raw test scores not validated", '        raw[role] = check_raw(ctx, inp, train_rec, role, "score_tests.json")',
     "        raw[role] = read_scores(ctx.run / RAW[role])"),
]
TRAIN_MUTANTS = [
    ("train_cnn: --no-test-eval accepted without --out-dir", "    if not test_eval and out_dir is None:\n", "    if False:\n"),
    ("train_cnn: weights and data folders compared as strings", "        if pg.within(d, q) or (Path(q).exists() and pg.contains(d, q)):\n",
     "        if str(Path(d).resolve()).startswith(str(Path(q).resolve())):\n"),
    ("train_cnn: protected folders of the repository accepted", '    return pg.deletion_problem(d, ROOT, pg.BASE_PROTECTED, [ROOT / "results" / "c2"])', '    return ""'),
    ("train_cnn: split sides of different length accepted", "        if len(idx) != len(split_check[side]):\n", "        if False:\n"),
    ("train_cnn: split compared as sets", "got, want = sorted(sha[int(i)] for i in idx), sorted(split_check[side])",
     "got, want = sorted(set(sha[int(i)] for i in idx)), sorted(set(split_check[side]))"),
    ("train_cnn: card written with another indentation", '    (dest / f"{key}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")',
     '    (dest / f"{key}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")'),
    ("train_cnn: outer work tree accepted", "    for g in outer_roots():\n        if pg.within(d, g) or pg.contains(d, g):\n", "    for g in []:\n        if pg.within(d, g) or pg.contains(d, g):\n"),
    ("train_cnn: git work tree not detected", "        if top is not None and not pg.same(top, root):\n            out.append(top)\n    except (OSError, subprocess.SubprocessError):\n        pass\n    for x in root.parents:",
     "        if False:\n            out.append(top)\n    except (OSError, subprocess.SubprocessError):\n        pass\n    for x in root.parents:"),
    ("train_cnn: research layout not detected", "        if (x / \"research\").is_dir() and any(pg.same(x / \"research\", y) for y in root.parents) and not any(pg.same(x, o) for o in out):\n            out.append(x)\n    return out",
     "        if False:\n            out.append(x)\n    return out"),
    ("train_cnn: --out-dir link accepted", "    if Path(d).is_symlink():\n        return f\"{Path(d)} is a symbolic link\"", "    if False:\n        return f\"{Path(d)} is a symbolic link\""),
    ("train_cnn: --out-dir file accepted", "    if Path(d).exists() and not Path(d).is_dir():\n", "    if False:\n"),
    ("train_cnn: options validated after reading images", "    if msg := validate_args(a):                       # prima di qualunque lettura di immagini o cache\n        raise SystemExit(f\"ERROR: {msg}\")\n",
     ""),
    ("train_cnn: unknown architecture accepted", "    if not a.arch or any(x not in ARCHS for x in a.arch):\n", "    if not a.arch:\n"),
    ("train_cnn: epochs 1 refused", "    if a.epochs < 1 or a.batch < 1", "    if a.epochs <= 1 or a.batch < 1"),
    ("train_cnn: batch 1 refused", "    if a.epochs < 1 or a.batch < 1", "    if a.epochs < 1 or a.batch <= 1"),
    ("train_cnn: per class 0 refused", "(a.per_class is not None and a.per_class < 0)", "(a.per_class is not None and a.per_class <= 0)"),
    ("train_cnn: lr 0 accepted", " or not a.lr > 0 or ", " or not a.lr >= 0 or "),
    ("train_cnn: empty architecture list accepted", "    if not a.arch or any(x not in ARCHS for x in a.arch):\n", "    if any(x not in ARCHS for x in a.arch):\n"),
    ("train_cnn: existing outputs overwritten by train", "    if out_dir is not None and (msg := existing_outputs_problem(dest, arch_name, ds, protocol)):\n", "    if False:\n"),
    ("train_cnn: dangling links ignored", 'if os.path.lexists(Path(d) / f"{key}{x}")]', 'if os.path.exists(Path(d) / f"{key}{x}")]'),
    ("train_cnn: existing outputs not checked by the CLI", "    for arch in a.arch if a.out_dir is not None else []:\n", "    for arch in []:\n"),
    ("train_cnn: non-finite lr accepted", " or not math.isfinite(a.lr) or ", " or "),
    ("train_cnn: epochs 0 accepted", "    if a.epochs < 1 or a.batch < 1", "    if a.epochs < 0 or a.batch < 1"),
    ("train_cnn: --no-test-eval without --out-dir accepted by the CLI", "    if a.no_test_eval and a.out_dir is None:\n        return", "    if False:\n        return"),
    ("train_cnn: --out-dir not checked by the CLI", "    if a.out_dir is not None and (msg := out_dir_problem(a.out_dir)):\n        return", "    if False:\n        return"),
    ("train_cnn: --out-dir ignored", "    dest = WEIGHTS if out_dir is None else Path(out_dir)\n", "    dest = WEIGHTS\n"),
    ("train_cnn: --out-dir not checked", "    if out_dir is not None and (msg := out_dir_problem(dest)):\n", "    if False and (msg := ''):\n"),
    ("train_cnn: split check not called", "        check_split(tr_i, va_i, split_check)\n", "        pass\n"),
    ("train_cnn: split compared by size only", "        if got != want:\n", "        if len(got) != len(want):\n"),
    ("train_cnn: split check without subject split", '        if not by_subject:\n            raise SystemExit("ERROR: split check requested',
     '        if False:\n            raise SystemExit("ERROR: split check requested'),
    ("train_cnn: test always evaluated", "    if test_eval:\n        # test: split", "    if True:\n        # test: split"),
    ("train_cnn: manifest.json written with --out-dir", "    if out_dir is None:\n        # manifest riassuntivo", "    if True:\n        # manifest riassuntivo"),
    ("train_cnn: default card changed", "    if out_dir is not None:\n        # scheda del pilota C2", "    if True:\n        # scheda del pilota C2"),
    ("train_cnn: --no-test-eval ignored by the CLI", "test_eval=not a.no_test_eval)", "test_eval=True)"),
]
# mutazioni dello sweep automatico del quinto giro (20/10): sopravvissute alla prima passata, ora rilevate dai test
# aggiunti (operatori di confronto e booleani, not, costanti, return di problemi, chiavi .get, chiamate ai controlli)
SWEEP_MUTANTS = [
    ('sweep 01: call public_scan removed (public_scan(jtext, inp, "c2_table.json"))', 'scripts/c2_pilot.py',
     '    public_scan(jtext, inp, "c2_table.json")',
     '    pass'),
    ('sweep 03: call fail removed (fail(f"configuration {p.name} is not a JSON object"))', 'scripts/c2_pilot.py',
     '        fail(f"configuration {p.name} is not a JSON object")',
     '        pass'),
    ('sweep 04: boolop (isinstance(v, int) and not isinstance(v, bool))', 'scripts/c2_pilot.py',
     '    return isinstance(v, int) and not isinstance(v, bool)',
     '    return isinstance(v, int) or not isinstance(v, bool)'),
    ('sweep 05: boolop (not isinstance(models, list) or len(models) != 2)', 'scripts/c2_pilot.py',
     '    if not isinstance(models, list) or len(models) != 2:',
     '    if not isinstance(models, list) and len(models) != 2:'),
    ('sweep 06: boolop (not isinstance(df, dict) or not isinstance(df.get("source"),)', 'scripts/c2_pilot.py',
     '    if not isinstance(df, dict) or not isinstance(df.get("source"), dict) or not isinstance(df.get("target"), dict):',
     "    if not isinstance(df, dict) and (not isinstance(df.get('source'), dict)) and (not isinstance(df.get('target'), dict)):"),
    ('sweep 07: problem return emptied ([f"train.json lists models {sorted(models)}, configuration {)', 'scripts/c2_pilot.py',
     '        return [f"train.json lists models {sorted(models)}, configuration {sorted(i for i, _ in model_list(cfg))}"]',
     '        return []'),
    ('sweep 08: call fail removed (fail(f"missing {p.name}"))', 'scripts/c2_pilot.py',
     '        fail(f"missing {p.name}")',
     '        pass'),
    ('sweep 09: call fail removed (fail(f"{p.name}: rows with a wrong number of fields"))', 'scripts/c2_pilot.py',
     '        fail(f"{p.name}: rows with a wrong number of fields")',
     '        pass'),
    ('sweep 11: call fail removed (fail("threshold.json is missing"))', 'scripts/c2_pilot.py',
     '        fail("threshold.json is missing")',
     '        pass'),
    ('sweep 12: call fail removed (fail(f"{what}: invalid input (see the ERROR line above)" if )', 'scripts/c2_pilot.py',
     '        fail(f"{what}: invalid input (see the ERROR line above)" if e.code in (None, 0, 1) else f"{what}: {e.code}")',
     '        pass'),
    ('sweep 13: compare Lt (b["max_epochs"] < 1)', 'scripts/c2_pilot.py',
     '    if not isinstance(b, dict) or not _int(b.get("max_epochs")) or b["max_epochs"] < 1 or not _int(b.get("batch")) or b["batch"] < 1 \\',
     '    if not isinstance(b, dict) or not _int(b.get("max_epochs")) or b[\'max_epochs\'] <= 1 or not _int(b.get("batch")) or b["batch"] < 1 \\'),
    ('sweep 14: compare Lt (b["batch"] < 1)', 'scripts/c2_pilot.py',
     '    if not isinstance(b, dict) or not _int(b.get("max_epochs")) or b["max_epochs"] < 1 or not _int(b.get("batch")) or b["batch"] < 1 \\',
     '    if not isinstance(b, dict) or not _int(b.get("max_epochs")) or b["max_epochs"] < 1 or not _int(b.get("batch")) or b[\'batch\'] <= 1 \\'),
    ('sweep 15: boolop (isinstance(models, list) and all(isinstance(m, dict) for m i)', 'scripts/c2_pilot.py',
     '    ids = [m.get("id") for m in models] if isinstance(models, list) and all(isinstance(m, dict) for m in models) else []',
     '    ids = [m.get("id") for m in models] if isinstance(models, list) or all((isinstance(m, dict) for m in models)) else []'),
    ('sweep 16: call fail removed (fail(f"{name} is not valid JSON"))', 'scripts/c2_pilot.py',
     '        fail(f"{name} is not valid JSON")',
     '        pass'),
    ('sweep 17: call fail removed (fail(f"manifest {e[\'file\']} ({role}) not found"))', 'scripts/c2_pilot.py',
     '            fail(f"manifest {e[\'file\']} ({role}) not found")',
     '            pass'),
    ('sweep 18: call fail removed (fail(f"manifest {e[\'file\']} ({role}): duplicated keys"))', 'scripts/c2_pilot.py',
     '            fail(f"manifest {e[\'file\']} ({role}): duplicated keys")',
     '            pass'),
    ('sweep 19: call fail removed (fail(f"{side} cache {p.name} is not the file of the configur)', 'scripts/c2_pilot.py',
     '            fail(f"{side} cache {p.name} is not the file of the configuration ({df[side][\'file\']})")',
     '            pass'),
    ('sweep 21: call fail removed (fail(f"{side} cache X_{split}: {len(rows) - len(keys)} manif)', 'scripts/c2_pilot.py',
     '            fail(f"{side} cache X_{split}: {len(rows) - len(keys)} manifest rows are not in the sampling order (per-class limit)")',
     '            pass'),
    ('sweep 22: boolop (len(y) != want_n[(side, split)] or len(keys) != want_n[(side)', 'scripts/c2_pilot.py',
     '        if len(y) != want_n[(side, split)] or len(keys) != want_n[(side, split)]:',
     '        if len(y) != want_n[side, split] and len(keys) != want_n[side, split]:'),
    ('sweep 23: call fail removed (fail(f"missing Python package {e.name!r}: train needs script)', 'scripts/c2_pilot.py',
     '        fail(f"missing Python package {e.name!r}: train needs scripts/train_cnn.py and its dependencies (TensorFlow, OpenCV, scikit-learn)")',
     '        pass'),
    ('sweep 24: boolop (e.get("weights_file") != h5.name or e.get("card_file") != ca)', 'scripts/c2_pilot.py',
     '        if e.get("weights_file") != h5.name or e.get("card_file") != card_p.name or e.get("keras_file") != ker.name:',
     "        if e.get('weights_file') != h5.name and e.get('card_file') != card_p.name and (e.get('keras_file') != ker.name):"),
    ('sweep 25: boolop (w != card.get("weights_sha256") or w != e.get("weights_sha25)', 'scripts/c2_pilot.py',
     '        if w != card.get("weights_sha256") or w != e.get("weights_sha256"):',
     "        if w != card.get('weights_sha256') and w != e.get('weights_sha256'):"),
    ('sweep 26: boolop (e.get("restored_epoch") != card.get("restored_epoch") or e.g)', 'scripts/c2_pilot.py',
     '        if e.get("restored_epoch") != card.get("restored_epoch") or e.get("epochs_run") != n_ep:',
     "        if e.get('restored_epoch') != card.get('restored_epoch') and e.get('epochs_run') != n_ep:"),
    ('sweep 27: call fail removed (fail(f"missing Python package {e.name!r}: score needs Tensor)', 'scripts/c2_pilot.py',
     '        fail(f"missing Python package {e.name!r}: score needs TensorFlow")',
     '        pass'),
    ('sweep 28: call fail removed (fail(f"{p.name}: unreadable CSV ({e})"))', 'scripts/c2_pilot.py',
     '        fail(f"{p.name}: unreadable CSV ({e})")',
     '        pass'),
    ('sweep 29: call fail removed (fail(f"missing Python package {e.name!r}: select and apply u)', 'scripts/c2_pilot.py',
     '        fail(f"missing Python package {e.name!r}: select and apply use src/tesi_app/evaluation.py (numpy, scikit-learn, OpenCV)")',
     '        pass'),
    ('sweep 30: call fail removed (fail(f"{mid}: selected value is not a candidate threshold"))', 'scripts/c2_pilot.py',
     '            fail(f"{mid}: selected value is not a candidate threshold")',
     '            pass'),
    ('sweep 31: call fail removed (fail(f"{mid}: counts at the selected value differ from the s)', 'scripts/c2_pilot.py',
     '            fail(f"{mid}: counts at the selected value differ from the selection")',
     '            pass'),
    ('sweep 32: call fail removed (fail("threshold.json missing or not valid JSON"))', 'scripts/c2_pilot.py',
     '        fail("threshold.json missing or not valid JSON")',
     '        pass'),
    ('sweep 33: boolop (RUN_STATE.get("cmd") or step)', 'scripts/c2_pilot.py',
     '    write_status("RUNNING", RUN_STATE.get("cmd") or step)',
     '    write_status("RUNNING", RUN_STATE.get(\'cmd\') and step)'),
    ('sweep 34: boolop (not isinstance(e.get("sha256"), str) or not HEX64.match(e["s)', 'scripts/c2_pilot.py',
     '                bad.append(f"manifests.{role}.file {e.get(\'file\')!r} not allowed")\n            if not isinstance(e.get("sha256"), str) or not HEX64.match(e["sha256"]):',
     '                bad.append(f"manifests.{role}.file {e.get(\'file\')!r} not allowed")\n            if not isinstance(e.get(\'sha256\'), str) and (not HEX64.match(e[\'sha256\'])):'),
    ('sweep 35: boolop (not isinstance(e.get("sha256"), str) or not HEX64.match(e["s)', 'scripts/c2_pilot.py',
     '                bad.append(f"derived_frames.{side}.file {e.get(\'file\')!r} is not <dataset>_<per class>.npz")\n            if not isinstance(e.get("sha256"), str) or not HEX64.match(e["sha256"]):',
     '                bad.append(f"derived_frames.{side}.file {e.get(\'file\')!r} is not <dataset>_<per class>.npz")\n            if not isinstance(e.get(\'sha256\'), str) and (not HEX64.match(e[\'sha256\'])):'),
    ('sweep 36: constant +1 (1)', 'scripts/c2_pilot.py',
     '    if not isinstance(b, dict) or not _int(b.get("max_epochs")) or b["max_epochs"] < 1 or not _int(b.get("batch")) or b["batch"] < 1 \\',
     '    if not isinstance(b, dict) or not _int(b.get("max_epochs")) or b["max_epochs"] < 1 or not _int(b.get("batch")) or b["batch"] < 2 \\'),
    ('sweep 37: constant +1 (1)', 'scripts/c2_pilot.py',
     '    if not isinstance(b, dict) or not _int(b.get("max_epochs")) or b["max_epochs"] < 1 or not _int(b.get("batch")) or b["batch"] < 1 \\',
     '    if not isinstance(b, dict) or not _int(b.get("max_epochs")) or b["max_epochs"] < 1 or not _int(b.get("batch")) or b["batch"] < 2 \\'),
    ('sweep 38: compare Gt (b["learning_rate"] > 0)', 'scripts/c2_pilot.py',
     '            or not isinstance(b.get("learning_rate"), float) or not b["learning_rate"] > 0 or not _int(b.get("wall_time_cap_minutes")):',
     '            or not isinstance(b.get("learning_rate"), float) or not b[\'learning_rate\'] >= 0 or not _int(b.get("wall_time_cap_minutes")):'),
    ('sweep 39: call fail removed (fail(f"{side} cache {p.name} unreadable ({e})"))', 'scripts/c2_pilot.py',
     '            fail(f"{side} cache {p.name} unreadable ({e})")',
     '            pass'),
    ('sweep 40: compare Eq (side == "source")', 'scripts/c2_pilot.py',
     '        need = {"X_train", "y_train", "s_train", "X_test", "y_test", "s_test"} if side == "source" else {"X_test", "y_test", "s_test"}',
     '        need = {"X_train", "y_train", "s_train", "X_test", "y_test", "s_test"} if side != \'source\' else {"X_test", "y_test", "s_test"}'),
    ('sweep 41: boolop (card_p.is_file() and h5.is_file() and ker.is_file())', 'scripts/c2_pilot.py',
     '        if not (card_p.is_file() and h5.is_file() and ker.is_file()):',
     '        if not (card_p.is_file() or h5.is_file() or ker.is_file()):'),
    ('sweep 42: constant +1 (1)', 'scripts/c2_pilot.py',
     '            wrong += 1',
     '            wrong += 2'),
    ('sweep 43: constant +1 (1)', 'scripts/c2_pilot.py',
     '            invalid += 1',
     '            invalid += 2'),
    ('sweep 44: constant +1 (1)', 'scripts/c2_pilot.py',
     '            ckpt += 1',
     '            ckpt += 2'),
    ('sweep 45: constant +1 (1)', 'scripts/c2_pilot.py',
     '            cfgs += 1',
     '            cfgs += 2'),
    ('sweep 46: boolop (r.get("threshold") != repr(t) or v is None or r.get("predict)', 'scripts/c2_pilot.py',
     '            phase_bad += r.get("threshold") != repr(t) or v is None or r.get("prediction") != str(int(v > t))',
     "            phase_bad += r.get('threshold') != repr(t) and v is None and (r.get('prediction') != str(int(v > t)))"),
    ('sweep 47: get key ("cmd")', 'scripts/c2_pilot.py',
     '    write_status("RUNNING", RUN_STATE.get("cmd") or step)',
     '    write_status("RUNNING", RUN_STATE.get(\'cmd_x\') or step)'),
    ('sweep 48: compare In (e.code in (0, None))', 'scripts/c2_pilot.py',
     '        if e.code in (0, None):',
     '        if e.code not in (0, None):'),
    ('sweep 49: boolop (not re.fullmatch(r"[a-z0-9_]+", str(m["id"])) or "pooled" in)', 'scripts/c2_pilot.py',
     '            elif not re.fullmatch(r"[a-z0-9_]+", str(m["id"])) or "pooled" in str(m["id"]):',
     "            elif not re.fullmatch('[a-z0-9_]+', str(m['id'])) and 'pooled' in str(m['id']):"),
    ('sweep 50: boolop (not _int(e.get(f)) or e[f] < 0)', 'scripts/c2_pilot.py',
     '                if not _int(e.get(f)) or e[f] < 0:',
     '                if not _int(e.get(f)) and e[f] < 0:'),
    ('sweep 51: boolop (not _int(e.get("n_videos")) or e["n_videos"] < 0)', 'scripts/c2_pilot.py',
     '            if ds == "casia_fasd" and (not _int(e.get("n_videos")) or e["n_videos"] < 0):',
     '            if ds == "casia_fasd" and (not _int(e.get(\'n_videos\')) and e[\'n_videos\'] < 0):'),
    ('sweep 52: boolop (not _int(e.get(mbr)) or e[mbr] <= 0)', 'scripts/c2_pilot.py',
     '                if not _int(e.get(mbr)) or e[mbr] <= 0:',
     '                if not _int(e.get(mbr)) and e[mbr] <= 0:'),
    ('sweep 53: compare Gt (len(ids) > i)', 'scripts/c2_pilot.py',
     '    want = [{"id": e, "model": ids[i] if len(ids) > i else None, "set": s} for e, (i, s) in EVALUATIONS.items()]',
     '    want = [{"id": e, "model": ids[i] if len(ids) >= i else None, "set": s} for e, (i, s) in EVALUATIONS.items()]'),
    ('sweep 54: constant +1 (1)', 'scripts/c2_pilot.py',
     '        wrong = sum(1 for r in inp.rows[role] if r["mirror_split"] != split)',
     '        wrong = sum(2 for r in inp.rows[role] if r["mirror_split"] != split)'),
    ('sweep 55: get key ("train_seconds")', 'scripts/c2_pilot.py',
     '                                  "train_seconds": info.get("train_seconds"),',
     '                                  "train_seconds": info.get(\'train_seconds_x\'),'),
    ('sweep 56: constant +1 (0)', 'scripts/c2_pilot.py',
     '        return np.concatenate(out) if out else np.zeros(0)',
     '        return np.concatenate(out) if out else np.zeros(1)'),
    ('sweep 57: constant +1 (1.0)', 'scripts/c2_pilot.py',
     '    if not math.isfinite(v) or not 0.0 <= v <= 1.0 or repr(v) != text:',
     '    if not math.isfinite(v) or not 0.0 <= v <= 2.0 or repr(v) != text:'),
    ('sweep 58: get key ("eer")', 'scripts/c2_pilot.py',
     '                            "roc_eer_information_only": m.get("eer")},',
     '                            "roc_eer_information_only": m.get(\'eer_x\')},'),
    ('sweep 59: constant +1 (15)', 'scripts/c2_pilot.py',
     '         "subjects bf / attack | APCER by attack type (accepted/n) |", "|" + "---|" * 15]',
     '         "subjects bf / attack | APCER by attack type (accepted/n) |", "|" + "---|" * 16]'),
    ('sweep 60: compare In (e.code in (None, 0, 1))', 'scripts/c2_pilot.py',
     '        fail(f"{what}: invalid input (see the ERROR line above)" if e.code in (None, 0, 1) else f"{what}: {e.code}")',
     '        fail(f"{what}: invalid input (see the ERROR line above)" if e.code not in (None, 0, 1) else f"{what}: {e.code}")'),
    ("sweep 61: get key ('schema')", 'scripts/c2_pilot.py',
     '        bad.append(f"schema {cfg.get(\'schema\')!r} is not {CONFIG_SCHEMA!r}")',
     '        bad.append(f"schema {cfg.get(\'schema_x\')!r} is not {CONFIG_SCHEMA!r}")'),
    ("sweep 62: get key ('seed')", 'scripts/c2_pilot.py',
     '        bad.append(f"seed {cfg.get(\'seed\')!r} is not {SEED}")',
     '        bad.append(f"seed {cfg.get(\'seed_x\')!r} is not {SEED}")'),
    ('sweep 63: compare Lt (e[f] < 0)', 'scripts/c2_pilot.py',
     '                if not _int(e.get(f)) or e[f] < 0:',
     '                if not _int(e.get(f)) or e[f] <= 0:'),
    ('sweep 64: compare Lt (e["n_videos"] < 0)', 'scripts/c2_pilot.py',
     '            if ds == "casia_fasd" and (not _int(e.get("n_videos")) or e["n_videos"] < 0):',
     '            if ds == "casia_fasd" and (not _int(e.get("n_videos")) or e[\'n_videos\'] <= 0):'),
    ('sweep 65: compare LtE (e[mbr] <= 0)', 'scripts/c2_pilot.py',
     '                if not _int(e.get(mbr)) or e[mbr] <= 0:',
     '                if not _int(e.get(mbr)) or e[mbr] < 0:'),
    ('sweep 66: compare Eq (r["label"] == "0")', 'scripts/c2_pilot.py',
     '           "n_subjects_bona_fide": len({r[subj] for r in rows if r["label"] == "0"}),',
     '           "n_subjects_bona_fide": len({r[subj] for r in rows if r[\'label\'] != \'0\'}),'),
    ('sweep 67: compare Eq (r["label"] == "1")', 'scripts/c2_pilot.py',
     '           "n_subjects_attack": len({r[subj] for r in rows if r["label"] == "1"})}',
     '           "n_subjects_attack": len({r[subj] for r in rows if r[\'label\'] != \'1\'})}'),
    ('sweep 68: constant +1 (1)', 'scripts/c2_pilot.py',
     '        if not _int(n_ep) or not 1 <= n_ep <= b["max_epochs"] or not isinstance(hist, list) or len(hist) != n_ep \\',
     '        if not _int(n_ep) or not 2 <= n_ep <= b["max_epochs"] or not isinstance(hist, list) or len(hist) != n_ep \\'),
    ('sweep 69: compare Eq (l == 1)', 'scripts/c2_pilot.py',
     '            "n_subjects_attack": len({s for s, l in zip(subj, y) if l == 1})}',
     '            "n_subjects_attack": len({s for s, l in zip(subj, y) if l != 1})}'),
    ('sweep 70: constant +1 (0)', 'scripts/c2_pilot.py',
     '        if e.code in (0, None):',
     '        if e.code in (1, None):'),
    ('sweep 71: get key ("file")', 'scripts/c2_pilot.py',
     '            if "pooled" in str(e.get("file", "")) or not str(e.get("file", "")).endswith(".csv") or "/" in str(e.get("file", "")):',
     '            if "pooled" in str(e.get("file", "")) or not str(e.get(\'file_x\', "")).endswith(".csv") or "/" in str(e.get("file", "")):'),
    ('sweep 72: constant +1 (0)', 'scripts/c2_pilot.py',
     '                if not _int(e.get(f)) or e[f] < 0:',
     '                if not _int(e.get(f)) or e[f] < 1:'),
    ('sweep 73: constant +1 (0)', 'scripts/c2_pilot.py',
     '            if ds == "casia_fasd" and (not _int(e.get("n_videos")) or e["n_videos"] < 0):',
     '            if ds == "casia_fasd" and (not _int(e.get("n_videos")) or e["n_videos"] < 1):'),
    ('sweep 74: constant +1 (0)', 'scripts/c2_pilot.py',
     '                if not _int(e.get(mbr)) or e[mbr] <= 0:',
     '                if not _int(e.get(mbr)) or e[mbr] <= 1:'),
    ('sweep 75: boolop (isinstance(v, float) and math.isfinite(v))', 'scripts/c2_pilot.py',
     '                or not all(isinstance(v, float) and math.isfinite(v) for v in hist):',
     '                or not all(isinstance(v, float) or math.isfinite(v) for v in hist):'),
    ('sweep 76: constant +1 (1)', 'scripts/c2_pilot.py',
     '               "files": {n: {"sha256": sha256_text(t), "rows": t.count("\\n") - 1} for n, t in texts.items()}, "created": now()}',
     '               "files": {n: {"sha256": sha256_text(t), "rows": t.count("\\n") - 2} for n, t in texts.items()}, "created": now()}'),
    ('sweep 77: constant +1 (1)', 'scripts/c2_pilot.py',
     '        fail(f"{what}: invalid input (see the ERROR line above)" if e.code in (None, 0, 1) else f"{what}: {e.code}")',
     '        fail(f"{what}: invalid input (see the ERROR line above)" if e.code in (None, 0, 2) else f"{what}: {e.code}")'),
    ('sweep 78: boolop (isinstance(h, str) and HEX64.match(h))', 'scripts/c2_pilot.py',
     '            if len(hashes) != 2 or not all(isinstance(h, str) and HEX64.match(h) for h in hashes):',
     '            if len(hashes) != 2 or not all(isinstance(h, str) or HEX64.match(h) for h in hashes):'),
    ("sweep 79: get key ('name')", 'scripts/c2_pilot.py',
     '                bad.append(f"manifests.{role}.name {e.get(\'name\')!r} is not {ds}/{name}")',
     '                bad.append(f"manifests.{role}.name {e.get(\'name_x\')!r} is not {ds}/{name}")'),
    ("sweep 80: get key ('file')", 'scripts/c2_pilot.py',
     '                bad.append(f"manifests.{role}.file {e.get(\'file\')!r} not allowed")',
     '                bad.append(f"manifests.{role}.file {e.get(\'file_x\')!r} not allowed")'),
    ("sweep 81: get key ('file')", 'scripts/c2_pilot.py',
     '                bad.append(f"derived_frames.{side}.file {e.get(\'file\')!r} is not <dataset>_<per class>.npz")',
     '                bad.append(f"derived_frames.{side}.file {e.get(\'file_x\')!r} is not <dataset>_<per class>.npz")'),
    ("sweep 82: get key ('restored_epoch')", 'scripts/c2_pilot.py',
     '                bad.append(f"{mid}: restored epoch {card.get(\'restored_epoch\')!r} is not the epoch of minimum val_loss ({best})")',
     '                bad.append(f"{mid}: restored epoch {card.get(\'restored_epoch_x\')!r} is not the epoch of minimum val_loss ({best})")'),
    ("sweep 83: get key ('arch')", 'scripts/c2_pilot.py',
     '                bad.append(f"model architectures {[m.get(\'arch\') for m in models]} are not {list(ARCHS)}")',
     '                bad.append(f"model architectures {[m.get(\'arch_x\') for m in models]} are not {list(ARCHS)}")'),
    ('sweep 84: compare Gt (v > t)', 'scripts/c2_pilot.py',
     '                per[k] = (n + 1, e + int((v > t) if lab == 0 else (v <= t)))',
     '                per[k] = (n + 1, e + int((v >= t) if lab == 0 else (v <= t)))'),
    ('sweep 85: compare GtE (len(set(groups.tolist())) >= 5)', 'scripts/train_cnn.py',
     '    by_subject = groups is not None and all(groups) and len(set(groups.tolist())) >= 5',
     '    by_subject = groups is not None and all(groups) and len(set(groups.tolist())) > 5'),
    ('sweep 86: constant +1 (5)', 'scripts/train_cnn.py',
     '    by_subject = groups is not None and all(groups) and len(set(groups.tolist())) >= 5',
     '    by_subject = groups is not None and all(groups) and len(set(groups.tolist())) >= 6'),
]
# dispositivo del training (richiesta del 27/10/2026): politica per architettura in train_cnn.py, verifica del
# dispositivo effettivo, registrazione e controllo prima dei punteggi in c2_pilot.py
DEVICE_MUTANTS = [
    ("device: LivenessNet on the default device", TRAIN, 'DEVICE_POLICY = {"LivenessNet": "CPU",', 'DEVICE_POLICY = {"LivenessNet": "default",'),
    ("device: AttackNet V2.2 forced on the CPU", TRAIN, '"AttackNetV2_1": "default", "AttackNetV2_2": "default"}', '"AttackNetV2_1": "default", "AttackNetV2_2": "CPU"}'),
    ("device: CPU scope not applied", TRAIN, '        return tf.device("/CPU:0")', "        return contextlib.nullcontext()"),
    ("device: GPU hidden for the whole process", TRAIN, "    with device_scope(tf, requested):\n",
     '    tf.config.set_visible_devices([], "GPU")\n    with device_scope(tf, requested):\n'),
    ("device: training outside the scope", TRAIN, "    with device_scope(tf, requested):\n        return _train_in_scope(",
     "    with device_scope(tf, requested):\n        pass\n    return _train_in_scope("),
    ("device: unknown architecture without policy accepted", TRAIN, '    if arch_name not in DEVICE_POLICY:\n        raise SystemExit',
     '    if False:\n        raise SystemExit'),
    ("device: unknown policy value accepted", TRAIN, '    raise SystemExit(f"ERROR: unknown device policy {requested!r}")', "    return contextlib.nullcontext()"),
    ("device: hidden GPU accepted", TRAIN, '    if "GPU" in physical and "GPU" not in visible:\n', "    if False:\n"),
    ("device: check before fit removed", TRAIN,
     "    if msg := device_problem(arch_name, requested, expected, sorted({short_device(v.device) for v in model.variables})):\n",
     "    if False:\n"),
    ("device: check after fit removed", TRAIN, "    if msg := device_problem(arch_name, requested, expected, effective, gap):\n", "    if False:\n"),
    ("device: graph/eager gap not checked", TRAIN, "    if msg := device_problem(arch_name, requested, expected, effective, gap):\n",
     "    if msg := device_problem(arch_name, requested, expected, effective):\n"),
    ("device: graph/eager tolerance loosened", TRAIN, "GRAPH_EAGER_TOL = 1e-5\n", "GRAPH_EAGER_TOL = 1e-2\n"),
    ("device: graph/eager gap with < instead of <=", TRAIN, " and not gap <= GRAPH_EAGER_TOL:", " and not gap < GRAPH_EAGER_TOL:"),
    ("device: graph/eager gap enforced for every policy", TRAIN, '    if requested == "CPU" and gap is not None and not gap <= GRAPH_EAGER_TOL:',
     "    if gap is not None and not gap <= GRAPH_EAGER_TOL:"),
    ("device: graph/eager gap measured on the first layer only", TRAIN, "    worst = int(np.argmax(rel))\n", "    worst = 0\n"),
    ("device: graph call replaced by eager", TRAIN, "    graph = [np.asarray(t) for t in tf.function(lambda t: sub(t, training=False))(xb)]",
     "    graph = [np.asarray(t) for t in sub(xb, training=False)]"),
    ("device: Keras predict not compared", TRAIN, '    graph += [np.asarray(model.predict(X, batch_size=len(X), verbose=0)),\n              np.asarray(model.evaluate(',
     "    graph += [probs,\n              np.asarray(model.evaluate("),
    ("device: Keras evaluate not compared", TRAIN, 'np.asarray(model.evaluate(X, Y, batch_size=len(X), verbose=0, return_dict=True)["loss"])]', "np.asarray(loss)]"),
    ("device: comparison in training mode", TRAIN, "    eager = [np.asarray(t) for t in sub(xb, training=False)]", "    eager = [np.asarray(t) for t in sub(xb, training=True)]"),
    ("device: comparison on no frame", TRAIN, "graph_eager_gap(tf, model, Xva[:4], Yva[:4])", "graph_eager_gap(tf, model, Xva[:0], Yva[:0])"),
    ("device: non-finite comparison reported as a device failure", TRAIN, '    if gap is not None and not np.isfinite(gap):\n', "    if False:\n"),
    ("device: non-finite comparison refused only for the CPU policy", TRAIN, '    if gap is not None and not np.isfinite(gap):\n',
     '    if requested == "CPU" and gap is not None and not np.isfinite(gap):\n'),
    ("device: layer named when every difference is 0", TRAIN, "(names[worst] if rel[worst] > 0 else None)", "names[worst]"),
    ("device: optimizer variables not checked", TRAIN, "list(model.variables) + list(model.optimizer.variables())", "list(model.variables)"),
    ("device: one variable on the right device is enough", TRAIN, "    if effective != [expected]:\n", "    if expected not in effective:\n"),
    ("device: default policy expected on the CPU", TRAIN,
     '    expected = "CPU:0" if requested == "CPU" or not tf.config.list_logical_devices("GPU") else "GPU:0"', '    expected = "CPU:0"'),
    ("device: not recorded in the card", TRAIN, '"training_device": device, **({"test": m} if test_eval else {})', '**({"test": m} if test_eval else {})'),
    ("device: visible devices not recorded", TRAIN, '"visible": visible, "physical": physical,', '"visible": ["CPU"], "physical": physical,'),
    ("device: check before scoring not called", PILOT, "        bad += device_problems(mid, arch, card, e)\n", ""),
    ("device: LivenessNet expected on the default device", PILOT, 'TRAINING_DEVICE = {"LivenessNet": "CPU",', 'TRAINING_DEVICE = {"LivenessNet": "default",'),
    ("device: card without device accepted", PILOT, '    if not isinstance(d, dict):\n        return [f"{mid}: the card does not record', '    if False:\n        return [f"{mid}: the card does not record'),
    ("device: requested device not compared", PILOT, '    if d.get("requested") != want:\n', "    if False:\n"),
    ("device: visible devices not validated", PILOT, "    elif not devs_ok(vis) or not devs_ok(phys):\n", "    elif not devs_ok(phys):\n"),
    ("device: physical devices not validated", PILOT, "    elif not devs_ok(vis) or not devs_ok(phys):\n", "    elif not devs_ok(vis):\n"),
    ("device: hidden GPU accepted before scoring", PILOT, '    elif "GPU" in phys and "GPU" not in vis:\n', "    elif False:\n"),
    ("device: default expected on the GPU without a GPU", PILOT, '        exp = "CPU:0" if want == "CPU" or "GPU" not in vis else "GPU:0"', '        exp = "CPU:0" if want == "CPU" else "GPU:0"'),
    ("device: effective device not compared", PILOT, 'for k in ("expected", "effective"):', 'for k in ("expected",):'),
    ("device: expected device not compared", PILOT, 'for k in ("expected", "effective"):', 'for k in ("effective",):'),
    ("device: graph/eager gap not validated", PILOT, "    if not isinstance(gap, float) or not math.isfinite(gap) or gap < 0:\n", "    if False:\n"),
    ("device: graph/eager gap not compared before scoring", PILOT, '    elif want == "CPU" and gap > GRAPH_EAGER_TOL:\n', "    elif False:\n"),
    ("device: graph/eager gap enforced for AttackNet before scoring", PILOT, '    elif want == "CPU" and gap > GRAPH_EAGER_TOL:\n', "    elif gap > GRAPH_EAGER_TOL:\n"),
    ("device: record fields not compared", PILOT, "    if sorted(d) != sorted(DEVICE_KEYS):\n", "    if False:\n"),
    ("device: a recorded field dropped", PILOT, ', "graph_eager_max_rel_diff", "graph_eager_layer")', ', "graph_eager_max_rel_diff")'),
    ("device: device lists not sorted or with repeats accepted", PILOT, " and v == sorted(set(v)) and ", " and "),
    ("device: visible devices not among the physical ones accepted", PILOT, "    elif not set(vis) <= set(phys):\n", "    elif False:\n"),
    ("device: graph/eager layer not matched with the difference", PILOT, "(layer is None) != (gap == 0)", "False"),
    ("device: graph/eager layer type not checked", PILOT, " or layer is not None and not (isinstance(layer, str) and layer):", ":"),
    ("device: train.json and card not compared", PILOT, '    if entry.get("training_device") != d:\n', "    if False:\n"),
    ("device: not recorded in train.json", PILOT, ', "training_device": info.get("training_device")}', "}"),
    ("train_cnn: non-finite validation loss accepted", TRAIN, '    if not all(np.isfinite(v) for v in hist.history["val_loss"]):\n', "    if False:\n"),
    ("device: unmeasured graph/eager comparison accepted after fit", TRAIN, "    if not isinstance(gap, float):\n", "    if False:\n"),
    ("config: training device not checked", PILOT,
     '    tdev = cfg.get("training", {}).get("device") if isinstance(cfg.get("training"), dict) else None\n',
     '    tdev = {**TRAINING_DEVICE, "note": "accepted"}\n'),
    ("config: training device policy not compared", PILOT, " or {a: tdev[a] for a in ARCHS} != TRAINING_DEVICE", ""),
    ("config: extra training device fields accepted", PILOT, 'sorted(tdev) != sorted([*ARCHS, "note"])', 'not {*ARCHS, "note"} <= set(tdev)'),
    ("config: training device note not checked", PILOT, ' \\\n            or not isinstance(tdev["note"], str) or not tdev["note"].strip():', ":"),
]
MUTANTS = [(n, PILOT, a, b) for n, a, b in PILOT_MUTANTS] + [(n, TRAIN, a, b) for n, a, b in TRAIN_MUTANTS]
MUTANTS += [tuple(m) for m in SWEEP_MUTANTS]
MUTANTS += DEVICE_MUTANTS


def copy_tree(dst: Path) -> Path:
    code = dst / "livenesslab"
    for f in CODE + [o for o in OPTIONAL if (ROOT / o).is_file()]:
        (code / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, code / f)
    return code


def pattern_problems() -> list:
    problems = []
    for name, target, old, _ in MUTANTS:
        n = (ROOT / target).read_text(encoding="utf-8").count(old)
        if n != 1:
            problems.append(f"{name}: the pattern occurs {n} times in {target} (expected exactly once)")
    return problems


BLOCK = ""
JOBS = 1
TIMEOUT = 1800                     # secondi per una suite su una copia mutata


def run_tests(code: Path, empty: Path) -> tuple:
    env = {**os.environ, "LIVENESSLAB_DATA_DIR": str(empty / "data"), "LIVENESSLAB_RESULTS_DIR": str(empty / "results"),
           "LIVENESSLAB_WEIGHTS_DIR": str(empty / "weights"), "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        r = subprocess.run([sys.executable, "-I", "-B", TESTS] + (["--block", BLOCK] if BLOCK else []), cwd=code, env=env, capture_output=True,
                           text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:      # una suite che non finisce: mutazione non valida, mai una rilevazione
        return None, f"timeout after {TIMEOUT} s"
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("tests run")), "no test summary")
    return r.returncode, line


def detected(rc: int, line: str) -> str:
    """killed: la suite è girata (riga «tests run») con almeno un fallimento o errore; invalid: la suite non è partita o
    è uscita con un codice senza fallimenti; survived: tutti i test passano (i test saltati non contano)."""
    m = re.match(r"tests run: (\d+), failures: (\d+), errors: (\d+)", line)
    if not m:
        return "invalid"
    if int(m.group(2)) + int(m.group(3)) > 0:
        return "killed"
    return "survived" if rc == 0 else "invalid"


def main() -> int:
    ap = argparse.ArgumentParser(description="Controllo delle mutazioni di c2_pilot.py e degli adattamenti di train_cnn.py (offline)")
    ap.add_argument("--list", action="store_true", help="elenca le mutazioni e verifica i modelli, senza eseguire i test")
    ap.add_argument("--block", default="", help="passato alla suite: pacchetti resi assenti (es. tensorflow,keras,h5py,cv2)")
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1), help="mutazioni eseguite in parallelo (default: fino a 4)")
    a = ap.parse_args()
    global BLOCK, JOBS
    BLOCK, JOBS = a.block, a.jobs
    problems = pattern_problems()
    for p in problems:
        print(f"ERROR: {p}", file=sys.stderr)
    if problems:
        print("RESULT: FAIL (a declared mutation no longer matches the code: update the list after review)")
        return 1
    if a.list:
        for name, target, _, _ in MUTANTS:
            print(f"{name}  [{target}]")
        print(f"{len(MUTANTS)} mutations, every pattern found exactly once")
        return 0
    killed = invalid = 0
    with tempfile.TemporaryDirectory(prefix="c2-mutations-") as tmp:
        tmp = Path(tmp)
        empty = tmp / "empty"
        base = copy_tree(tmp / "base")
        rc, line = run_tests(base, empty)
        print(f"unmutated copy: {TESTS} exit {rc} ({line})", flush=True)
        if rc:
            print("RESULT: FAIL (the offline tests do not pass on the unmutated copy)")
            return 1
        def one(i_m):
            i, (name, target, old, new) = i_m
            work = tmp / f"work{i}"
            code = copy_tree(work)
            f = code / target
            f.write_text(f.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")
            try:
                return run_tests(code, empty)
            finally:
                shutil.rmtree(work, ignore_errors=True)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, JOBS)) as ex:      # ogni mutazione in una copia propria
            outcomes = list(ex.map(one, enumerate(MUTANTS)))
        for (name, target, old, new), (rc, line) in zip(MUTANTS, outcomes):
            verdict = detected(rc, line)
            killed += verdict == "killed"
            invalid += verdict == "invalid"
            print(f"{verdict.upper():<9} {name} ({line})", flush=True)
            if verdict == "invalid":
                print(f"ERROR: invalid mutant {name!r}: the test suite did not run to a summary with failures (exit {rc})", file=sys.stderr)
    print(f"mutations killed: {killed}/{len(MUTANTS)}" + (f" ({invalid} invalid)" if invalid else ""))
    print("RESULT: " + ("PASS" if killed == len(MUTANTS) else "FAIL"))
    return 0 if killed == len(MUTANTS) else 1


if __name__ == "__main__":
    sys.exit(main())
