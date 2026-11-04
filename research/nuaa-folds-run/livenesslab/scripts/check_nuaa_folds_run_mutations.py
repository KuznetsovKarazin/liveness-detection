"""
Controllo delle mutazioni di scripts/nuaa_folds_run.py (run LivenessNet 4 fold × 3 seed), offline e deterministico.

Copia il codice (scripts/nuaa_folds_run.py, scripts/check_nuaa_folds_run.py, scripts/m2_writers.py, scripts/path_guard.py,
scripts/package_rules.py, scripts/train_cnn.py e, se c'è, la configurazione delivery/foldsrun/) in una cartella temporanea, applica una
mutazione dichiarata alla volta (coppia cerca/sostituisci che deve comparire esattamente una volta: se un modello non si
trova più, il controllo si ferma con un errore) ed esegue scripts/check_nuaa_folds_run.py sulla copia mutata, con gli
ingressi veri in sola lettura (FOLDSRUN_INPUT_ROOT = radice di questo repository). Una mutazione è rilevata solo se la
suite arriva al riepilogo («tests run») con almeno un fallimento o errore, cioè dal comportamento; una suite che non
parte la rende non valida. Una mutazione di controllo (solo un commento) deve sopravvivere.

Test saltati: la copia non mutata deve passare e può saltare test solo per un ingresso mancante dichiarato
(MISSING_INPUT di scripts/check_nuaa_folds_run.py); i salti sono riportati. Una mutazione conta come rilevata solo per
fallimenti o errori, mai per un salto. Codice 1 se una mutazione sopravvive o non è valida, se un salto non è un
ingresso mancante dichiarato o se un modello non si trova.

  .venv/bin/python scripts/check_nuaa_folds_run_mutations.py            # tutte le mutazioni
  .venv/bin/python scripts/check_nuaa_folds_run_mutations.py --list     # solo l'elenco, con la verifica dei modelli
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
from check_nuaa_folds_run import MISSING_INPUT  # noqa: E402

TESTS = "scripts/check_nuaa_folds_run.py"
S = "scripts/nuaa_folds_run.py"
COPY = ("scripts/nuaa_folds_run.py", "scripts/check_nuaa_folds_run.py", "scripts/m2_writers.py", "scripts/path_guard.py",
        "scripts/package_rules.py", "scripts/train_cnn.py")
CONFIG = "nuaa_folds_run.json"
# (nome, file mutato, testo cercato, testo sostitutivo)
MUTANTS = [
    # configurazione
    ("training recipe not compared", S, '    if cfg["training"] != FROZEN_TRAINING:', "    if False:"),
    ("bootstrap recipe not compared", S, '    if cfg["bootstrap"] != BOOT:', "    if False:"),
    ("threshold criterion not compared", S, 'or th["criterion"] != THRESHOLD_CRITERION or th["set"] != "dev fold only":', 'or th["set"] != "dev fold only":'),
    ("budget trainings not compared", S, '    if bu["trainings"] != exp["k"] * len(seeds):', "    if False:"),
    ("duplicated seeds accepted", S, "or not all(is_count(s) for s in seeds) or len(set(seeds)) != len(seeds):", "or not all(is_count(s) for s in seeds):"),
    ("expected counts not checked", S, '    if sum(exp["test_per_fold"]) != exp["n_images"] or exp["n_images"] != exp["n_cache_frames"] + exp["n_to_extract"]:', "    if False:"),
    ("declaration flags not required", S, '        if dec.get(k) is not True:\n            fail(f"configuration: declarations.{k} must be true")', '        if False:\n            fail(f"configuration: declarations.{k} must be true")'),
    ("architecture file not tied to the reference code", S, '    if ar["sha256"] != ex["reference_code"]["architectures"]["sha256"] or ar["path"] != ex["reference_code"]["architectures"]["path"]:', "    if False:"),
    # manifest M2 e fold
    ("M2 manifest SHA-256 not checked", S, '        if h != spec["sha256"]:\n            fail(f"manifest {k} ({spec[\'file\']}): SHA-256', '        if False:\n            fail(f"manifest {k} ({spec[\'file\']}): SHA-256'),
    ("M2 manifest counts not checked", S, '            if spec[x] != v:\n                fail(f"manifest {k}: {x} = {v}', '            if False:\n                fail(f"manifest {k}: {x} = {v}'),
    ("cache manifest rows not compared with the official lists", S, '            if o is None or any(o[c] != r[c] for c in COLUMNS):', "            if o is None:"),
    ("fold SHA256SUMS anchor not checked", S, '    if h != sr["restricted_sha256sums_sha256"]:', "    if False:"),
    ("fold file SHA-256 not compared with SHA256SUMS", S, '            if hh != listed[name]:', "            if False:"),
    ("fold file not required in SHA256SUMS", S, '            if name not in listed:\n                fail(f"schema SHA256SUMS: {name} not listed")', '            if name not in listed:\n                listed[name] = sha256_bytes(read_regular(folder / name, name))'),
    ("fold row count not checked", S, '            if len(rows) != want:', "            if False:"),
    ("fold and role of a row not checked", S, '                if r["fold"] != str(i) or r["role"] != role:', "                if False:"),
    ("fold row not compared with the official list", S, '                if o is None or any(o[c] != r[c] for c in ("label", "subject", "session", "sha256", "bytes")):', "                if o is None:"),
    ("in_cache flag not checked", S, '                if r["in_cache"] not in ("0", "1") or (r["in_cache"] == "1") != (r["key"] in cache_keys):', '                if r["in_cache"] not in ("0", "1"):'),
    ("roles sharing a subject accepted", S, '            if subj[a] & subj[b_] or keys[a] & keys[b_]:', "            if keys[a] & keys[b_]:"),
    ("dev not compared with the next test fold", S, '        if {r["subject"] for r in inp.folds[i]["dev"]} != test_subjects[nxt]:', "        if False:"),
    ("symbolic-link input accepted", S, "    if stat.S_ISLNK(st.st_mode):\n        fail(f\"{what} {p.name} is a symbolic link: inputs must be regular files\")\n    if not stat.S_ISREG(st.st_mode):\n        fail(f\"{what} {p.name} is not a regular file\")\n    with open(p, \"rb\") as f:",
     "    if False:\n        fail(f\"{what} {p.name} is a symbolic link: inputs must be regular files\")\n    if not stat.S_ISREG(st.st_mode):\n        fail(f\"{what} {p.name} is not a regular file\")\n    with open(p, \"rb\") as f:"),
    # cache, codice congelato, ambiente
    ("cache SHA-256 not checked", S, '    if inp.cache_sha != c["sha256"]:', "    if False:"),
    ("cache member schema not checked", S, '        if got != want or mem[k]["fortran"]:', "        if False:"),
    ("cache labels not linked to the manifests", S, '        if len(y) != len(keys) or y.dtype.kind not in "iu" or not np.array_equal(y, exp_y):', '        if len(y) != len(keys) or y.dtype.kind not in "iu":'),
    ("frozen face SHA-256 not checked", S, '    if h != ex["frozen_face"]["sha256"]:', "    if False:"),
    ("detector resource SHA-256 not checked", S, '            if hh != v["sha256"]:', "            if False:"),
    ("OpenCV version not compared", S, '        if out["opencv"] != cfg["environment"]["opencv"]:', "        if False:"),
    ("TensorFlow import at check not detected", S, '    if "tensorflow" in sys.modules:\n        fail("TensorFlow was imported', '    if False:\n        fail("TensorFlow was imported'),
    ("cache frame shape not checked", S, '    if tuple(mem["X_train"]["shape"][1:]) != FRAME_SHAPE or tuple(mem["X_test"]["shape"][1:]) != FRAME_SHAPE:', "    if False:"),
    # cache-check ed estrazione
    ("cache frame difference ignored", S, "            diff = int(np.abs(arr.astype(np.int16) - X[i].astype(np.int16)).max())", "            diff = 0"),
    ("original SHA-256 not checked", S, '    if p.stat().st_size != int(r["bytes"]) or sha256_file(p) != r["sha256"]:', '    if p.stat().st_size != int(r["bytes"]):'),
    ("original symbolic link accepted", S, '    if p.is_symlink():\n        fail(f"{what}: symbolic link refused")', '    if False:\n        fail(f"{what}: symbolic link refused")'),
    ("extracted frame shape not checked", S, "    if arr.shape != FRAME_SHAPE or arr.dtype != np.uint8:", "    if False:"),
    ("existing frames file overwritten by extract", S, '    if frames_path.exists() or frames_path.is_symlink():\n        fail(f"{frames_path.name} already exists: extracted frames are never overwritten (move it away first)")', '    if False:\n        fail("")'),
    ("identical frames across subjects accepted", S, "    if across:\n        fail(", "    if False:\n        fail("),
    ("extracted keys unsorted", S, "    keys = list(inp.to_extract)\n    X = np.zeros", "    keys = list(reversed(inp.to_extract))\n    X = np.zeros"),
    # training e scheda
    ("GPU left visible", S, '    if devices != ["CPU"]:\n        fail(f"visible devices {devices}, CPU only expected")', '    if False:\n        fail(f"visible devices {devices}, CPU only expected")'),
    ("seed not given to TensorFlow", S, "    random.seed(train_seed); np.random.seed(train_seed); tf.random.set_seed(train_seed)", "    random.seed(train_seed); np.random.seed(train_seed)"),
    ("early stopping without restore_best_weights", S, 'EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)', 'EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=False)'),
    ("shuffle without the training seed", S, ".shuffle(4096, seed=train_seed).batch(batch)", ".shuffle(4096).batch(batch)"),
    ("cap callback not called", S, "        def on_epoch_end(self, epoch, logs=None):\n            cap_check(f\"epoch {epoch + 1}\")", "        def on_epoch_end(self, epoch, logs=None):\n            pass"),
    ("best weights not restored when the epochs end first", S, "    if es.stopped_epoch == 0 and es.best_weights is not None:\n        model.set_weights(es.best_weights)", "    if False:\n        model.set_weights(es.best_weights)"),
    ("restored epoch not the minimum of val_loss", S, '    best = int(np.argmin(h["val_loss"]))\n    if es.stopped_epoch', '    best = int(np.argmax(h["val_loss"]))\n    if es.stopped_epoch'),
    ("card not checked against the recipe", S, '    if problems := card_problems(card, cfg["training"], seed, weights):', '    if problems := []:'),
    ("card restored epoch not checked", S, '    elif card["restored_epoch"] != int(np.argmin(vl)) + 1 or card["best_val_loss"] != float(vl[card["restored_epoch"] - 1]):', "    elif False:"),
    ("val_loss of the checkpoint not recomputed", S, "    if not (diff <= VAL_LOSS_TOL[0] or diff <= VAL_LOSS_TOL[1] * card[\"best_val_loss\"]):", "    if False:"),
    # soglia e punteggi
    ("threshold selected on the test instead of the dev", S, '    sel = eer_threshold(dev["y"], p_dev[:, 1])', '    sel = eer_threshold(test["y"], probabilities(scorer, test["X"])[:, 1])'),
    ("EER objective replaced by the sum", S, "        key = abs(fp * n_a - fn * n_b)", "        key = fp * n_a + fn * n_b"),
    ("EER ties to the lowest value", S, "        if best is None or key < best[0]:", "        if best is None or key <= best[0]:"),
    ("candidate below the minimum dropped", S, "    if low >= 0.0:\n        vals.append(low)", "    if False:\n        vals.append(low)"),
    ("tie at the threshold counted as attack", S, '"prediction_dev": "1" if v > thr["value"] else "0",', '"prediction_dev": "1" if v >= thr["value"] else "0",'),
    ("threshold.json not read back", S, "    if back != thr:\n        fail", "    if False:\n        fail"),
    ("threshold SHA-256 not written in the score rows", S, '                     "threshold_sha256": thr["sha256"]})', '                     "threshold_sha256": ""})'),
    ("scorer output not validated", S, "    if p.shape != (len(X), 2) or not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):", "    if False:"),
    ("dev frames scored with the test frames", S, '    p_dev = probabilities(scorer, dev["X"])', '    p_dev = probabilities(scorer, test["X"][:len(dev["X"])])'),
    # metriche e aggregato
    ("AUROC ties not averaged", S, "        ranks[order[i:j + 1]] = (i + j + 2) / 2", "        ranks[order[i:j + 1]] = i + 1"),
    ("AUROC with the classes swapped", S, '    out["auroc"] = auroc(s[y == 1], s[y == 0])', '    out["auroc"] = auroc(s[y == 0], s[y == 1])'),
    ("rejected and accepted swapped", S, "        accepted, rejected = int(np.sum((y == 1) & (pred == 0))), int(np.sum((y == 0) & (pred == 1)))", "        accepted, rejected = int(np.sum((y == 0) & (pred == 1))), int(np.sum((y == 1) & (pred == 0)))"),
    ("ACER with an absent class", S, '        out[f"acer_{name}"] = (apcer + bpcer) / 2 if apcer is not None and bpcer is not None else None', '        out[f"acer_{name}"] = (apcer or 0.0 + (bpcer or 0.0)) / 2'),
    ("stratum rows not filtered by session", S, '    return list(rows) if stratum == "all" else [r for r in rows if r["session_id"] == stratum]', "    return list(rows)"),
    ("median replaced by the mean", S, '    return {"values": values, "median": float(statistics.median(values)), "min": float(min(values)), "max": float(max(values))}', '    return {"values": values, "median": float(statistics.mean(values)), "min": float(min(values)), "max": float(max(values))}'),
    ("bootstrap seed 1", S, '    rng = np.random.default_rng(BOOT["seed"])\n    vals = {k: [] for k in keys}', '    rng = np.random.default_rng(1)\n    vals = {k: [] for k in keys}'),
    ("bootstrap with 100 resamples", S, '    for _ in range(BOOT["resamples"]):', "    for _ in range(100):"),
    ("bootstrap resampled subjects deduplicated", S, "        idx = rng.integers(0, len(subjects), len(subjects))\n        sample", "        idx = np.unique(rng.integers(0, len(subjects), len(subjects)))\n        sample"),
    ("bootstrap interval with few subjects", S, '    if len(subjects) < BOOT["min_subjects"]:', "    if False:"),
    ("pooled rows from one fold only", S, "        rows = [r for fold in range(1, k + 1) for r in scores[(fold, seed)]]\n        pooled[seed]", "        rows = list(scores[(1, seed)])\n        pooled[seed]"),
    ("per-subject rows from every subject", S, "        sel = [r for r in rows if r[\"subject_id\"] == subj]\n        for stratum in STRATA:", "        sel = list(rows)\n        for stratum in STRATA:"),
    # uscite, stato, protezioni
    ("public scan skipped", S, '            if probs := public_problems(text, inp, name.endswith(".md")):', '            if probs := []:'),
    ("subject identifiers not scanned", S, '            bad.append(f"subject id {s}")\n            break', "            break"),
    ("foreign entries accepted", S, "                if not any(mw.owned_name(e, n) for n in names):", "                if False:"),
    ("run folder not validated", S, "    if msg := run_dir_problem(run, ROOT, inputs, cfg, out_root):", '    if msg := "":'),
    ("authorisation flag not required", S, "    if not a.i_am_authorized:", "    if False:"),
    ("failure recorded as PASS", S, '"status": "FAIL", "passed": False, "reason": reason', '"status": "PASS", "passed": False, "reason": reason'),
    ("outputs not removed after a failure", S, "        left = invalidate_outputs(run, cfg)\n        try:\n            parked", "        left = []\n        try:\n            parked"),
    ("dirty tree not declared", S, '("-dev" if state["dirty"] else "")', '""'),
    ("wall-time cap never exceeded", S, "        if self.elapsed() > self.cap:", "        if False:"),
    ("previous outputs not invalidated before the run", S, "        left = invalidate_outputs(run, cfg)\n        if left:\n            fail(f\"previous outputs could not be removed", "        left = []\n        if left:\n            fail(f\"previous outputs could not be removed"),
    ("restricted SHA256SUMS without the weights", S, '        names = [n for n in restricted_outputs(cfg) if n != "SHA256SUMS"]', '        names = [n for n in restricted_outputs(cfg) if n != "SHA256SUMS" and not n.endswith(".keras")]'),
    ("public anchor not written in the restricted folder", S, '"public_SHA256SUMS.sha256": sha256_bytes(public["SHA256SUMS"].encode("utf-8")) + "  public/SHA256SUMS\\n"}', '"public_SHA256SUMS.sha256": "none  public/SHA256SUMS\\n"}'),
    ("error not written in the run log", S, '            log[0].write(f"ERROR: {reason}\\nRESULT: FAIL (run)\\n")', '            pass'),
    # ricetta e scoring (revisione del 3/11: mutazioni non dichiarate)
    ("class weights swapped", S, "    cw = {0: len(ytr) / (2 * (ytr == 0).sum()), 1: len(ytr) / (2 * (ytr == 1).sum())}", "    cw = {1: len(ytr) / (2 * (ytr == 0).sum()), 0: len(ytr) / (2 * (ytr == 1).sum())}"),
    ("bootstrap percentiles [5, 95]", S, '            lo, hi = np.percentile(vals[k], BOOT["percentiles"])', "            lo, hi = np.percentile(vals[k], [5, 95])"),
    ("tie at 0.5 counted as attack", S, '"prediction_05": "1" if v > FIXED_THRESHOLD else "0",', '"prediction_05": "1" if v >= FIXED_THRESHOLD else "0",'),
    ("RandomRotation(0.3)", S, "tf.keras.layers.RandomRotation(0.03)", "tf.keras.layers.RandomRotation(0.3)"),
    ("GaussianNoise removed", S, "tf.keras.layers.RandomContrast(0.2), tf.keras.layers.GaussianNoise(0.02)])", "tf.keras.layers.RandomContrast(0.2)])"),
    ("augmentation order changed", S, 'aug = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.03),', 'aug = tf.keras.Sequential([tf.keras.layers.RandomRotation(0.03), tf.keras.layers.RandomFlip("horizontal"),'),
    ("one-hot labels inverted", S, "    Ytr, Yva = np.eye(2)[ytr], np.eye(2)[yva]", "    Ytr, Yva = np.eye(2)[1 - ytr], np.eye(2)[1 - yva]"),
    ("training input not divided by 255", S, "    Xtr /= 255.0\n", "    Xtr /= 1.0\n"),
    ("scoring input not divided by 255", S, "xb = X[i:i + SCORE_BATCH].astype(np.float32) / 255.0", "xb = X[i:i + SCORE_BATCH].astype(np.float32) / 1.0"),
    ("scoring in training mode", S, "out.append(np.asarray(model(xb, training=False), dtype=np.float64))", "out.append(np.asarray(model(xb, training=True), dtype=np.float64))"),
    ("augmentation also on the dev", S, "    ds_va = tf.data.Dataset.from_tensor_slices((Xva, Yva)).batch(batch)", "    ds_va = tf.data.Dataset.from_tensor_slices((Xva, Yva)).batch(batch).map(lambda x, y: (aug(x, training=True), y))"),
    ("loss changed", S, "loss=tf.keras.losses.CategoricalCrossentropy()", "loss=tf.keras.losses.BinaryCrossentropy()"),
    ("budget not checked before aggregate", S, '        budget.check("aggregate")\n', "        pass\n"),
    ("val_loss tolerance loosened", S, "VAL_LOSS_TOL = (1e-3, 1e-2) ", "VAL_LOSS_TOL = (1e-1, 1e-1) "),
    ("duplicated cache frames accepted", S, '        if different or result[split]["duplicates_within"]:', "        if different:"),
    ("scoring batch 64", S, "        for i in range(0, len(X), SCORE_BATCH):", "        for i in range(0, len(X), 64):"),
    ("clear_session removed", S, "    tf.keras.backend.clear_session()", "    pass"),
    ("checkpoint not saved through a temporary name", S, "        model.save(tmp)\n        os.replace(tmp, weights_path)", "        model.save(weights_path)"),
    # controlli aggiunti dalla revisione del 3/11
    ("detectors not compared with the N2 cache check", S, '        if result[split]["detectors"] != cfg["cache"]["detectors_n2"][split]:', "        if False:"),
    ("frames shared between X_train and X_test accepted", S, "    if shared:\n        fail(", "    if False:\n        fail("),
    ("schema configuration SHA-256 not checked", S, '    if h != sr["config_sha256"] or h != SCHEMA_CONFIG_SHA256:', "    if False:"),
    ("schema training fields not compared", S, '    if diff or not str(tr.get("device", "")).startswith("CPU"):', "    if False:"),
    ("frames file not refused at check-inputs", S, '    if frames_path.exists() or frames_path.is_symlink():\n        fail(f"{frames_path.name} already exists in {frames_path.parent.name}: a previous extraction is in the way; move it away "\n             "(the run never overwrites it)")\n    return inp',
     '    return inp'),
    ("existing status.json not refused", S, "    if (run / STATUS).exists() or (run / STATUS).is_symlink():", "    if False:"),
    ("environment variables not refused", S, '    if refused:\n        fail(f"environment variables', '    if False:\n        fail(f"environment variables'),
    ("originals not located at check-inputs", S, "    check_originals(cfg, base, inp)\n", "    inp.checks['originals'] = {'located_and_hashed': len(inp.official), 'layout': cfg['images']['layout']}\n"),
    ("TensorFlow version not compared", S, '        if out["tensorflow_installed"] != cfg["environment"]["tensorflow"]:', "        if False:"),
    ("run log SHA-256 not recorded", S, '                           "run_log_sha256": sha256_file(run / LOG)})', '                           "run_log_sha256": "none"})'),
    ("failed extraction not parked", S, "            parked = park_failed_frames(frames_path, rid)", '            parked = "none"'),
    ("SIGHUP handler installed even when ignored", S, "    if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:\n        sigs.append(signal.SIGHUP)", "    sigs.append(signal.SIGHUP)"),
    ("SIGTERM handler not installed", S, "    sigs = [signal.SIGTERM]\n", "    sigs = []\n"),
    ("checkpoint temporary left behind when the save fails", S, "        tmp.unlink(missing_ok=True)                   # nessun temporaneo resta in restricted/ dopo un errore\n        raise", "        raise"),
]
# mutazione di controllo: solo un commento; deve SOPRAVVIVERE (prova che la suite rileva il comportamento, non il testo)
CONTROL = [("control: comment only", S, "def f4(v) -> str:\n", "def f4(v) -> str:  # control mutant: comment only\n")]
# controllo dichiarato ridondante: in un giro di K fold ogni altro fold sta nel fitting o nel dev del fold k, quindi una
# chiave estranea nel test è già rifiutata come condivisa e i test partizionano le immagini per costruzione. Deve SOPRAVVIVERE.
REDUNDANT = [("declared redundant (must survive): test folds partition the images", S,
              '    if set(all_test) != set(inp.official) or any(v != 1 for v in all_test.values()):', "    if False:"),
             # il conteggio dei frame da estrarre e' fissato dagli altri controlli: n_images = n_cache_frames + n_to_extract
             # (configurazione), le liste ufficiali hanno n_images righe e i manifest della cache n_cache_frames chiavi
             # dentro le liste, quindi official - cache = n_to_extract per costruzione. Deve SOPRAVVIVERE.
             ("declared redundant (must survive): frames to extract not counted", S,
              '    if len(inp.to_extract) != exp["n_to_extract"]:', "    if False:"),
             # il tetto e' controllato tre volte fra l'aggregazione e le scritture finali (prima del record, prima delle
             # scritture riservate, prima dei checksum): togliere il primo controllo lascia gli altri due, che la suite vede
             # («at write»). Deve SOPRAVVIVERE.
             ("declared redundant (must survive): first of the three cap checks before the final writes removed", S,
              '        budget.check("write")\n        record = {', "        record = {")]


def copy_tree(dst: Path) -> Path:
    """Copia del codice e, se c'è, della configurazione consegnata (delivery/foldsrun/ nel repository di sviluppo e nel
    pacchetto livenesslab/, oppure accanto alla radice del pacchetto), al percorso predefinito dello script."""
    code = dst / "code"
    for rel in COPY:
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, code / rel)
    for c in (ROOT / "delivery" / "foldsrun", ROOT.parent):
        if (c / CONFIG).is_file():
            (code / "delivery" / "foldsrun").mkdir(parents=True, exist_ok=True)
            shutil.copy2(c / CONFIG, code / "delivery" / "foldsrun" / CONFIG)
            break
    return code


def run_tests(code: Path, tmp: Path) -> tuple:
    tmp.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "FOLDSRUN_INPUT_ROOT": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp)}
    r = subprocess.run([sys.executable, "-I", TESTS], cwd=code, env=env, capture_output=True, text=True, timeout=1800)
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
    ap = argparse.ArgumentParser(description="Controllo delle mutazioni di nuaa_folds_run.py (offline)")
    ap.add_argument("--list", action="store_true", help="elenca le mutazioni e verifica i modelli, senza eseguire i test")
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1), help="mutazioni eseguite in parallelo")
    a = ap.parse_args()
    problems = [f"{n}: the pattern occurs {k} times in {t} (expected exactly once)"
                for n, t, old, _ in MUTANTS + CONTROL + REDUNDANT if (k := (ROOT / t).read_text(encoding="utf-8").count(old)) != 1]
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
    with tempfile.TemporaryDirectory(prefix="foldsrun-mutations-") as tmp:
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
