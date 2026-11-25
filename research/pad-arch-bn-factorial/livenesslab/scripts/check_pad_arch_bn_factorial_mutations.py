"""
Controllo delle mutazioni di scripts/pad_arch_bn_factorial.py e di delivery/archbn2x2/launch.zsh (la prova BN 2x2 di
PROTOCOL.md §4), offline e deterministico, con il meccanismo del test dichiarato di
delivery/pad-attacknet-eval-paper/check_make_table_figure_mutations.py.

Copia il codice (ogni scripts/*.py del repository, che dà i moduli riusati e i mondi sintetici delle loro suite, questo
comando, la sua suite, delivery/archbn2x2/make_config.py, delivery/archbn2x2/launch.zsh e l'architettura di riferimento
src/livedetection/src/architectures.py) in una cartella temporanea e applica una mutazione alla volta (coppia
cerca/sostituisci che deve comparire esattamente una volta: se un modello non si trova più, il controllo si ferma con un
errore). Ogni mutazione dichiara in DECLARED il test (Classe.metodo) che deve ucciderla: sulla copia mutata si esegue solo
quel test, e la mutazione è uccisa solo se quel test fallisce; se passa, o se la suite non arriva al riepilogo, la
mutazione è «invalid». La riga del verdetto dice il primo test fallito, il test dichiarato e l'ultima riga del suo errore
(solo il tipo e il messaggio, con i percorsi tolti). La mutazione di controllo (solo un commento) e le mutazioni
dichiarate ridondanti girano con la suite intera e devono sopravvivere. --list controlla anche che ogni test dichiarato
esista nella suite.

Nessun dato reale: la suite usa solo array sintetici e modelli minuscoli. Le mutazioni girano una alla volta (un solo
processo TensorFlow per volta). TMPDIR deve stare fuori dal repository e non contenere la cartella home.

  .venv/bin/python -I -B scripts/check_pad_arch_bn_factorial_mutations.py              # tutte le mutazioni, una alla volta
  .venv/bin/python -I -B scripts/check_pad_arch_bn_factorial_mutations.py --list       # l'elenco, con la verifica dei modelli e dei test
  .venv/bin/python -I -B scripts/check_pad_arch_bn_factorial_mutations.py --only "<nome>" [--only ...]   # una selezione
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
_spec = importlib.util.find_spec("pad_arch_run")          # i moduli riusati (nel repository, la stessa cartella scripts/)
REPO_SCRIPTS = Path(_spec.origin).parent if _spec is not None else ROOT / "scripts"
TESTS = "scripts/check_pad_arch_bn_factorial.py"
S = "scripts/pad_arch_bn_factorial.py"
L = "delivery/archbn2x2/launch.zsh"
OWN = (S, TESTS, "scripts/check_pad_arch_bn_factorial_mutations.py", "delivery/archbn2x2/make_config.py", L)
ARCH = ("src", "livedetection", "src", "architectures.py")
# (nome, file mutato, testo cercato, testo sostitutivo)
MUTANTS = [
    # augmentation (rilievo 1 della revisione del piano, I5 del codice)
    ("GaussianNoise omitted from the augmentation", S, "    for op in OPS:\n        x = apply_op(", "    for op in OPS[:-1]:\n        x = apply_op("),
    ("contrast clip removed", S, "        return tf.clip_by_value(out, 0, 255)", "        return out"),
    ("brightness not clipped", S, "        return tf.clip_by_value(x, lo, hi)", "        return x"),
    ("rotation filled with constant zeros", S, 'tf.float32), hd, wd), fill_mode="reflect", fill_value=0.0,', 'tf.float32), hd, wd), fill_mode="constant", fill_value=0.0,'),
    ("zoom with nearest interpolation", S, 'get_zoom_matrix(zooms, hd, wd), fill_mode="reflect", fill_value=0.0, interpolation="bilinear")',
     'get_zoom_matrix(zooms, hd, wd), fill_mode="reflect", fill_value=0.0, interpolation="nearest")'),
    ("zoom width and height swapped", S, 'zooms = tf.cast(tf.concat([tf.constant(p["zoom_w"][:, None]), tf.constant(p["zoom_h"][:, None])], axis=1), tf.float32)',
     'zooms = tf.cast(tf.concat([tf.constant(p["zoom_h"][:, None]), tf.constant(p["zoom_w"][:, None])], axis=1), tf.float32)'),
    ("translation width and height swapped", S, "        tr = tf.cast(tf.concat([wt, ht], axis=1), tf.float32)", "        tr = tf.cast(tf.concat([ht, wt], axis=1), tf.float32)"),
    ("flip never drawn", S, 'np.float32(np.rint(philox("augmentation", view_seed, d, "flip").random()))',
     'np.float32(np.floor(philox("augmentation", view_seed, d, "flip").random()))'),
    ("rotation range not multiplied by 2 pi", S, '"angle": (-rot * 2.0 * np.pi, rot * 2.0 * np.pi)', '"angle": (-rot, rot)'),
    ("zoom range 0.12 -> 0.08 in the recipe", S, '("RandomZoom", (0.12, 0.12), {})', '("RandomZoom", (0.08, 0.08), {})'),
    ("zoom range narrowed in the derivation", S, '"zoom_h": (1.0 - zh, 1.0 + zh)', '"zoom_h": (1.0 - zh * 2 / 3, 1.0 + zh * 2 / 3)'),
    ("brightness range 0.15 -> 0.10 in the derivation", S, '"brightness": (-br, br)', '"brightness": (-br * 2 / 3, br * 2 / 3)'),
    ("noise std 0.02 -> 0.03 in the recipe", S, '("GaussianNoise", (0.02,), {})', '("GaussianNoise", (0.03,), {})'),
    ("noise std scaled in the draw", S, '    sd = np.float32(RANGES["noise_stddev"])', '    sd = np.float32(1.5 * RANGES["noise_stddev"])'),
    ("frames divided by 255 twice", S, "    return X.astype(np.float32) / 255.0", "    return X.astype(np.float32) / 255.0 / 255.0"),
    ("frames not divided by 255", S, "    return X.astype(np.float32) / 255.0", "    return X.astype(np.float32)"),
    ("contrast factor drawn per batch", S, 'uniform(philox("augmentation", view_seed, d, "contrast")',
     'uniform(philox("augmentation", view_seed, digests[0], "contrast")'),
    # semi (rilievo 13 della revisione del piano)
    ("seeds by position instead of the frame key", S, "    return [frame_digest(k) for k in keys]", "    return [frame_digest(str(i)) for i, k in enumerate(keys)]"),
    ("view seeds 100-107 instead of 101-108", S, "VIEW_SEEDS = tuple(range(101, 109))", "VIEW_SEEDS = tuple(range(100, 108))"),
    ("Monte Carlo halves 1-4 and 1-4", S, "HALVES = ((101, 102, 103, 104), (105, 106, 107, 108))", "HALVES = ((101, 102, 103, 104), (101, 102, 103, 104))"),
    ("key without the view", S, 'f"{SEED_NAMESPACE}|{domain}|{int(view_seed)}|{digest}|{tag}"', 'f"{SEED_NAMESPACE}|{domain}|{digest}|{tag}"'),
    ("augmentation and dropout streams not separated", S, 'f"{SEED_NAMESPACE}|{domain}|{int(view_seed)}|{digest}|{tag}"',
     'f"{SEED_NAMESPACE}|{int(view_seed)}|{digest}|{tag}"'),
    # dropout e modalità delle BN (rilievi 12 e 14 della revisione del piano, M3 del codice)
    ("dropout_dense active", S, "    active = [d for d in drops if any(names.index(b) > names.index(d) for b in bns)]", "    active = list(drops)"),
    ("BatchNormalization in training mode in the chain", S, "                        x = layer(x, training=False)",
     "                        x = layer(x, training=True)"),
    ("whole model called with training=True", S, "                    res = sub(xb, training=False)", "                    res = sub(xb, training=True)"),
    ("dropout without the 1 / (1 - rate) scale", S, "    scale = tf.convert_to_tensor(1 / (1 - rate), dtype=x.dtype)",
     "    scale = tf.convert_to_tensor(1.0, dtype=x.dtype)"),
    ("mask keeps u < rate", S, "    return u >= np.float32(rate)", "    return u < np.float32(rate)"),
    ("masks per channel (noise_shape)", S, "mask_of(d, v, layer.name, tuple(x.shape[1:]), rate)", "mask_of(d, v, layer.name, (1, 1, int(x.shape[-1])), rate)"),
    ("noise_shape of the Dropout not checked", S, '    if any(l.noise_shape is not None for l in layers if type(l).__name__ == "Dropout"):', "    if False:"),
    ("dropout rate of the checkpoint not compared", S, '    if drops != {k: float(v) for k, v in dc["layers"].items()}:', "    if False:"),
    ("dropout active in E10", S, '"E10": (True, False)', '"E10": (True, True)'),
    ("GaussianNoise as the dropout factor (augmented input in E01)", S, '    sources = {"E10": aug8, "E01": clean8, "E11": aug8}',
     '    sources = {"E10": aug8, "E01": aug8, "E11": aug8}'),
    ("augmentation omitted in E11", S, '    sources = {"E10": aug8, "E01": clean8, "E11": aug8}', '    sources = {"E10": aug8, "E01": clean8, "E11": clean8}'),
    ("augmented frames used for the dev scoring", S, '    dev_ev, _ = BR.evaluate(copy, dev, batch, f"{label} dev")',
     '    dev_ev, _ = BR.evaluate(copy, {**dev, "X": np.clip(dev["X"].astype(np.int16) + 20, 0, 255).astype(np.uint8)}, batch, f"{label} dev")'),
    ("gamma assigned instead of moving_variance", S, "        layer.moving_variance.assign(v32)\n        want =", "        layer.gamma.assign(v32)\n        want ="),
    # riuso e metà (rilievi 7 e 17 della revisione del piano; revisore del codice)
    ("every layer reused", S, "    return [n for n in order if stochastic_upstream(st, cell, n) == stochastic_upstream(st, ref, n)]", "    return list(order)"),
    ("E11 reuses E00", S, 'REFERENCE = {"E01": "E00", "E11": "E10"}', 'REFERENCE = {"E01": "E00", "E11": "E00"}'),
    ("halves use all eight views", S, 'src = Source.augmented(views, fit["digests"], h) if FACTORS[cell][0] else Source.clean(fit["X"], fit["digests"], h)',
     'src = Source.augmented(views, fit["digests"], VIEW_SEEDS) if FACTORS[cell][0] else Source.clean(fit["X"], fit["digests"], VIEW_SEEDS)'),
    ("E11 halves reuse E10 on all views", S, '                    rcell = rec["cells"][(ref, tag if ref != "E00" else ALL_VIEWS)]',
     '                    rcell = rec["cells"][(ref, ALL_VIEWS)]'),
    ("EMA coefficient 0.99^e", S, '"initialization_coefficient": cfg["ema"]["momentum"] ** (upe * e)}', '"initialization_coefficient": cfg["ema"]["momentum"] ** e}'),
    # controlli e arresti (I4 del codice; rilievi 9, 10, 18 della revisione del piano)
    ("E00 tolerance on the dev scores widened", S, '    ok = diff <= tol["dev_probability_max_abs_diff"] and d05', "    ok = diff <= 1.0 and d05"),
    ("EER threshold value not compared", S, '(dthr is None or dthr <= tol["dev_probability_max_abs_diff"])', "True"),
    ("E00 control not binding", S, '"pass": bool(sc["outside"] == 0 and old["identical"] and dc["pass"])}', '"pass": True}'),
    ("stored moments not compared", S, '"pass": bool(sc["outside"] == 0 and old["identical"] and dc["pass"])}', '"pass": bool(sc["outside"] == 0 and dc["pass"])}'),
    ("E00 chain not binding", S, '        if not rec["controls"]["E00_chain"]["pass"]:', "        if False:"),
    ("baseline control not binding", S, '    if not c["pass"]:\n        raise ControlFailure("baseline"', '    if False:\n        raise ControlFailure("baseline"'),
    ("tensor check removed from the cells", S, "    if probs := BR.tensor_problems(orig_snap, BR.snapshot_weights(copy), BR.moving_keys(copy, order), stats):",
     "    if probs := []:"),
    ("consistency not binding", S, '    if bad := {n: c["channels_outside_tolerance"] for n, c in cons.items() if c["channels_outside_tolerance"]}:\n        raise ControlFailure("consistency"',
     '    if bad := {}:\n        raise ControlFailure("consistency"'),
    ("softmax not binding", S, '        if r is not None and r["softmax_gap"] > tol["softmax_of_logits_max_abs_diff"]:', "        if False:"),
    ("memory limit not checked", S, "        if gb > lim:", "        if False:"),
    ("mask frequency not binding", S, '    if any(not f["pass"] for f in freq.values()):', "    if False:"),
    ("determinism not binding", S, '        if not rec["controls"]["determinism"]["pass"]:', "        if False:"),
    ("views not regenerated in the determinism control", S, '"pass": bool(sc2["identical"] and same_p and all(regen.values()))}',
     '"pass": bool(sc2["identical"] and same_p)}'),
    ("duplicates not binding", S, '            if not entry["pass"]:', "            if False:"),
    ("non-finite views accepted", S, "                    if not np.all(np.isfinite(out)):", "                    if False:"),
    ("view bounds not checked", S, "                    if mn < lo or mx > hi:", "                    if False:"),
    ("EMA facts not checked", S, '    if math.ceil(card["n_fitting"] / card["batch"]) != upe or card["n_fitting"] != len(fit["X"]):', "    if False:"),
    ("original model not checked at the end", S, '    if BR.snapshot_weights(orig) != snap or trainable_sha256(orig) != rec["trainable_sha256"]:', "    if False:"),
    ("checkpoint file not checked at the end", S, '    if sha256_file(parent["weights"]) != parent["weights_sha256"]:', "    if False:"),
    ("tie at 0.5 decided attack", S, "    d = D.attack_decisions(np.asarray(p, np.float64), FR.FIXED_THRESHOLD)",
     "    d = np.asarray(p, np.float64) >= FR.FIXED_THRESHOLD"),
    # viste fuori dal repository (I6 del codice)
    ("free space not checked", S, "            if free < need:", "            if False:"),
    ("views folder accepted inside a forbidden root", S, "                if pg.within(parent, r):", "                if False:"),
    ("views folder kept after a stop", S, "        except BaseException:\n            self.close()\n            raise", "        except BaseException:\n            raise"),
    ("stale views folder not checked", S, "    if left := stale_views(views_parent):", "    if left := []:"),
    # protezione del test (rilievi 5 e 6 della revisione del piano, I1-I3 del codice) e ingressi
    ("read guard disabled", S, '    if event != "open" or not GUARD["active"] or not args:', "    if True:"),
    ("names compared with their case", S, "    names = {os.path.basename(full).lower(), os.path.basename(real).lower()}",
     "    names = {os.path.basename(full), os.path.basename(real)}"),
    ("resolved path not followed", S, "    chain = _chain(full) | _chain(real)", "    chain = _chain(full)"),
    ("relative os.open accepted", S, '    if relative and GUARD["inside"]:', "    if False:"),
    ("TMPDIR beside the repository accepted", S, '(REPO_PARENT or ROOT.parent, "the folder that contains the repository")', '(Path("/nonexistent-parent"), "the folder that contains the repository")'),
    ("zero Monte Carlo scale reported as not within", S, "            return None, (True if x == 0 else None)", "            return None, x == 0"),
    ("allowed folders outside the repository not checked at the start", S, "    if why := outside_problem(base, [cfg_p.parent] + list(outside_extra)):",
     '    if why := "":'),
    ("TMPDIR inside a root accepted at the start", S, "        if pg.within(tmp, r):", "        if False:"),
    ("allowed folder containing a root accepted", S, "            if pg.within(r, q):", "            if False:"),
    ("refused opens not checked at every step", S, '        if GUARD["hits"]:                              # un\'apertura', '        if False:                              # un\'apertura'),
    ("guard hits not checked at the end of the trial", S, '        if GUARD["hits"]:\n            fail(f"read guard: {len(GUARD[\'hits\'])} refused opens during the trial")',
     '        if False:\n            fail(f"read guard: {len(GUARD[\'hits\'])} refused opens during the trial")'),
    ("attempts counted across scopes", S, "        if m.group(1) == scope:", "        if True:"),
    ("views folder allowed inside the repository", S, "            for r in refuse_inside:", "            for r in ():"),
    ("translation range 0.06 -> 0.04", S, '"trans_h": (-th, th), "trans_w": (-tw, tw)', '"trans_h": (-0.04, 0.04), "trans_w": (-0.04, 0.04)'),
    ("zoom range fixed at 0.92-1.08", S, '"zoom_h": (1.0 - zh, 1.0 + zh), "zoom_w": (1.0 - zw, 1.0 + zw)', '"zoom_h": (0.92, 1.08), "zoom_w": (0.92, 1.08)'),
    ("Monte Carlo scale reduced by the number of units", S, '    s = math.sqrt(sum(r["h"] ** 2 for r in rs) / len(rs)) / 2',
     '    s = math.sqrt(sum(r["h"] ** 2 for r in rs) / len(rs)) / 2 / math.sqrt(len(rs))'),
    ("Monte Carlo half difference of the mean not halved", S, '"mean", len(rs), mean, hbar, abs(hbar) / 2, s', '"mean", len(rs), mean, hbar, abs(hbar), s'),
    ("within the view variability at 1 x the scale", S, "        return abs(x) / s, abs(x) <= k_within * s", "        return abs(x) / s, abs(x) <= s"),
    ("forbidden folders not compared", S, '    if chain & set(GUARD["forbidden"]):', "    if False:"),
    ("hard links not checked", S, 'and st.st_nlink > 1 and not chain & set(GUARD["link_exempt"]):', "and False:"),
    ("zones not allowlisted", S, "        if zone in chain and not chain & set(allowed):", "        if False:"),
    ("outside rule not applied", S, '    if GUARD["inside"] and not chain & set(GUARD["inside"]) and not chain & set(GUARD["outside"]):', "    if False:"),
    ("input root accepted with the official root", S, "    if official and (a.input_root is not None and Path(a.input_root).resolve() != ROOT.resolve()):",
     "    if False:"),
    ("configuration outside the repository accepted with the official root", S, "    if official and not pg.within(cfg_p, ROOT):", "    if False:"),
    ("unit keys not checked", S, '            if BR.keys_sha(rows) != p["card"].get(f"{role}_keys_sha256"):', "            if False:"),
    ("parent files outside the list accepted", S, "    if suffix not in PARENT_SUFFIXES:", "    if False:"),
    ("parent file SHA-256 not compared", S, '            if h != p["sums"][fn]:', "            if False:"),
    ("card network, fold, seed or patience not compared", S, '!= (u["network"], u["fold"], u["seed"], PARENT_RUNS[u["run"]], w):',
     '!= (card.get("network"), card.get("fold"), card.get("train_seed"), card.get("early_stopping_patience"), w):'),
    ("null note accepted by the trial", S, '    if cfg["analysis"]["sha256"] is None:', "    if False:"),
    ("dirty tree accepted", S, '    if state["dirty"]:', "    if False:"),
    ("sealed cell metrics published", S, '[] if sealed else texts["cells"]', 'texts["cells"]'),
    # controlli protettivi (C1 della quarta revisione del codice)
    ("cap not checked during the views", S, 'cfg["views"]["value_bounds"], cfg["views"]["free_space_margin_gb"], check=ctx.step,',
     'cfg["views"]["value_bounds"], cfg["views"]["free_space_margin_gb"], check=None,'),
    ("frozen note not compared", S, '        if h != an["sha256"]:', "        if False:"),
    ("Keras module not compared", S, '    if k != cfg["keras_functions"]["sha256"]:', "    if False:"),
    ("parent status not compared with its pin", S, '        if sha256_bytes(b) != e["status_sha256"]:', "        if False:"),
    ("configuration and code not rechecked at the end", S,
     '        if sha256_bytes(read_regular(cfg_p, "configuration")) != cfg_sha or check_code(cfg) != code:', "        if False:"),
    ("authorisation flag not required", S, "    if not a.i_am_authorized:", "    if False:"),
    ("saved dev rows of another role accepted", S, '!= (u["network"], str(u["fold"]), str(u["seed"]), state, "dev", w) for x in rr):',
     '!= (x["network"], x["fold"], x["train_seed"], x["bn_state"], x["role"], x["checkpoint_sha256"]) for x in rr):'),
    ("saved dev order not compared", S, '    if [r["sample_id"] for r in rows] != dev["keys"]:', "    if False:"),
    ("a required name pattern dropped", S, 'REQUIRED_PATTERNS = ("fold[0-9]*_test.csv", "*_test_scores.csv", ', 'REQUIRED_PATTERNS = ("fold[0-9]*_test.csv", '),
    ("budget not checked at the steps", S, "        self.budget.check(what)\n", "        pass\n"),
    ("threshold other than 0.5 accepted", S, '    if cfg["threshold"] != FR.FIXED_THRESHOLD:', "    if False:"),
    ("threads other than 10 accepted", S, '    if (th["intra_op"], th["inter_op"], th["device"]) != (PAC.REQUIRED_THREADS, PAC.REQUIRED_THREADS, "CPU"):',
     "    if False:"),
    ("test statement of the configuration not compared", S, '    if cfg["test"] != TEST_READ:', "    if False:"),
    ("results/archeval not required among the forbidden paths", S, '    if "results/archeval" not in rg["forbidden_paths"] or ', "    if "),
    ("attempt number not compared", S, "    if a.attempt != need:", "    if False:"),
    ("cap stop reported as FAIL", S, '            status = "INCOMPLETE"\n', '            status = "FAIL"\n'),
    # lanciatore (B1 della quarta revisione del codice)
    ("launch: nice not checked", L, "[[ $nice_now == 0 ]] || refuse", "true || refuse"),
    ("launch: trial on battery", L, "[[ $power == *\"'AC Power'\"* ]] || refuse", "true || refuse"),
    ("launch: low power mode accepted", L, "[[ ${lpm:-0} == 0 ]] || refuse", "true || refuse"),
    ("launch: powermode other than 0 accepted", L, "[[ ${pm:-0} == 0 ]] || refuse", "true || refuse"),
    ("launch: a refusal takes the attempt's name", L, '"== not launched: $2"; } > $N', '"== not launched: $2"; } > $C'),
    ("launch: a failed check-inputs takes the attempt's name", L, "  mv $C $N\n", "  :\n"),
    ("launch: views folder left accepted", L, "[[ $left == 0 ]] || refuse", "true || refuse"),
    ("launch: failed check-inputs ignored", L, "if ! grep -q '^check-inputs exit 0$' $C; then", "if false; then"),
    ("launch: log folder inside the repository accepted", L, "if [[ $log_dir == $repo || $log_dir == $repo/* ]]; then", "if false; then"),
    ("launch: existing conditions file overwritten", L, 'if [[ -e $C ]]; then print -u2 "$label-conditions.txt already exists"; exit 9; fi', ":"),
    # lanciatore e rifiuti interni della prova (R4 della quarta revisione del codice)
    ("launch: dirty work tree accepted", L, "[[ $dirty == 0 ]] || refuse", "true || refuse"),
    ("launch: folder that is not the root of a git work tree accepted", L, "[[ -n $top && ${top:A} == $repo ]] || refuse", "true || refuse"),
    ("launch: a trial refused before its run folder takes the attempt's name", L,
     "  mv $C $N; [[ -e $console ]] && mv $console ${N%-conditions.txt}.console.log\n", "  :\n"),
    ("launch: failed git status taken as a clean work tree", L, "st=$(git status --porcelain 2>/dev/null) || refuse",
     "st=$(git status --porcelain 2>/dev/null) || true"),
    ("launch: start line condition inverted", L, "if (( rc != 0 )) && ! grep -q '^== 2x2 trial ' $console; then",
     "if (( rc != 0 )) && grep -q '^== 2x2 trial ' $console; then"),
    # uscita della suite (C4 della quarta revisione del codice)
    ("suite: the Keras randrange warning not filtered in the runner", TESTS,
     '            warnings.filterwarnings("ignore", message=RANDRANGE_WARNING, category=DeprecationWarning)\n            self.warnings = None',
     "            self.warnings = None"),
]
# il test che deve uccidere ciascuna mutazione (Classe.metodo); --list controlla che esista nella suite
A, B, C, D = "TestAConfig.", "TestBRecipe.", "TestCDraws.", "TestDGuardAndComparisons."
E, Fd, G, ZA, ZB, H = "TestEAugmentation.", "TestFDropoutStructure.", "TestGMoments.", "TestZATrialPass.", "TestZBTrialStops.", "TestHLaunch."
DECLARED = {
    "GaussianNoise omitted from the augmentation": E + "test_augment_is_the_recipe_in_order",
    "contrast clip removed": E + "test_clips",
    "brightness not clipped": E + "test_clips",
    "rotation filled with constant zeros": E + "test_each_operation_equals_the_keras_layer",
    "zoom with nearest interpolation": E + "test_each_operation_equals_the_keras_layer",
    "zoom width and height swapped": E + "test_each_operation_equals_the_keras_layer",
    "translation width and height swapped": E + "test_each_operation_equals_the_keras_layer",
    "flip never drawn": C + "test_flip_and_independent_sides",
    "rotation range not multiplied by 2 pi": E + "test_draw_ranges_against_the_keras_layers_of_the_source",
    "zoom range 0.12 -> 0.08 in the recipe": B + "test_real_source_equals_the_recipe",
    "zoom range narrowed in the derivation": B + "test_ranges_follow_the_recipe",
    "brightness range 0.15 -> 0.10 in the derivation": E + "test_draw_ranges_against_the_keras_layers_of_the_source",
    "noise std 0.02 -> 0.03 in the recipe": B + "test_real_source_equals_the_recipe",
    "noise std scaled in the draw": E + "test_draw_ranges_against_the_keras_layers_of_the_source",
    "frames divided by 255 twice": E + "test_identity_parameters_give_the_scaled_frames",
    "frames not divided by 255": E + "test_identity_parameters_give_the_scaled_frames",
    "contrast factor drawn per batch": C + "test_pure_function_of_its_arguments",
    "seeds by position instead of the frame key": C + "test_digests_from_the_keys",
    "view seeds 100-107 instead of 101-108": C + "test_view_seeds",
    "Monte Carlo halves 1-4 and 1-4": C + "test_view_seeds",
    "key without the view": C + "test_pure_function_of_its_arguments",
    "augmentation and dropout streams not separated": C + "test_domains_and_keys",
    "dropout_dense active": Fd + "test_livenessnet_chain_accepted",
    "BatchNormalization in training mode in the chain": G + "test_T3_bn_in_inference_and_only_upstream_dropout",
    "whole model called with training=True": G + "test_T1_e00_by_the_new_path_is_the_estimator",
    "dropout without the 1 / (1 - rate) scale": Fd + "test_dropout_equals_tf_stateless_dropout",
    "mask keeps u < rate": C + "test_masks_by_frame_view_layer",
    "masks per channel (noise_shape)": G + "test_T7_same_masks_for_clean_and_augmented_inputs",
    "noise_shape of the Dropout not checked": Fd + "test_structure_refusals",
    "dropout rate of the checkpoint not compared": Fd + "test_structure_refusals",
    "dropout active in E10": D + "test_reuse_rule",
    # l'E01 su viste aumentate (o l'E11 su frame puliti) mescola strati riusati e strati ristimati su sorgenti diverse: il
    # controllo di coerenza della cella va in FAIL e il setUp di TestZATrialPass fallisce per ogni suo test
    "GaussianNoise as the dropout factor (augmented input in E01)": ZA + "test_cells_effects_layers",
    # stessa via del precedente (controllo di coerenza della cella)
    "augmentation omitted in E11": ZA + "test_controls_pass",
    # via indiretta: il duplicato esplicito di ResNet-18 non torna, la prova sintetica va in FAIL; il test dichiarato è
    # quello che ricostruisce gli stati
    "augmented frames used for the dev scoring": ZA + "test_states_rebuilt_from_the_restricted_files",
    "gamma assigned instead of moving_variance": G + "test_T8_only_moving_tensors_change",
    "every layer reused": D + "test_reuse_rule",
    "E11 reuses E00": D + "test_reuse_rule",
    "halves use all eight views": ZA + "test_halves_use_four_views",
    # le statistiche a 8 viste riusate sulle metà non passano la coerenza sulle 4 viste: la prova sintetica va in FAIL
    "E11 halves reuse E10 on all views": ZA + "test_controls_pass",
    "EMA coefficient 0.99^e": ZA + "test_cells_effects_layers",
    "E00 tolerance on the dev scores widened": D + "test_score_compare",
    "EER threshold value not compared": D + "test_score_compare",
    "E00 control not binding": ZB + "test_e00_not_reproduced",
    "stored moments not compared": ZB + "test_stored_moments_differ",
    "E00 chain not binding": ZB + "test_e00_chain_not_reproduced",
    "baseline control not binding": ZB + "test_baseline_not_reproduced",
    "tensor check removed from the cells": ZB + "test_trained_tensor_changed_in_a_cell",
    "consistency not binding": ZB + "test_consistency",
    "softmax not binding": ZB + "test_softmax",
    "memory limit not checked": ZB + "test_memory",
    "mask frequency not binding": ZB + "test_mask_frequency",
    "determinism not binding": ZB + "test_determinism",
    "views not regenerated in the determinism control": ZB + "test_views_not_regenerated_identically",
    "duplicates not binding": ZB + "test_duplicates",
    "non-finite views accepted": ZB + "test_non_finite_views",
    "view bounds not checked": E + "test_views_store_and_checks",
    "EMA facts not checked": ZB + "test_ema_facts",
    "original model not checked at the end": ZB + "test_model_or_checkpoint_changed_at_the_end",
    "checkpoint file not checked at the end": ZB + "test_model_or_checkpoint_changed_at_the_end",
    "tie at 0.5 decided attack": D + "test_rates_at_05_and_ties",
    "free space not checked": ZB + "test_disk_space",
    "views folder accepted inside a forbidden root": E + "test_views_store_and_checks",
    "views folder kept after a stop": E + "test_views_store_and_checks",
    "stale views folder not checked": ZB + "test_stale_views_folder",
    "read guard disabled": D + "test_guard_by_identity_and_lower_case",
    "names compared with their case": D + "test_guard_by_identity_and_lower_case",
    "resolved path not followed": D + "test_guard_by_identity_and_lower_case",
    "relative os.open accepted": D + "test_guard_by_identity_and_lower_case",
    "TMPDIR beside the repository accepted": ZB + "test_temporary_folder_beside_the_repository",
    "zero Monte Carlo scale reported as not within": D + "test_montecarlo_scale_without_reduction",
    "allowed folders outside the repository not checked at the start": ZB + "test_temporary_folder_checked_at_the_start",
    "TMPDIR inside a root accepted at the start": ZB + "test_temporary_folder_checked_at_the_start",
    "allowed folder containing a root accepted": ZB + "test_temporary_folder_checked_at_the_start",
    "refused opens not checked at every step": ZB + "test_swallowed_refusal_stops_at_the_next_step",
    "guard hits not checked at the end of the trial": ZB + "test_swallowed_refusal_after_the_last_step",
    "attempts counted across scopes": ZB + "test_attempts_counted_per_scope",
    "views folder allowed inside the repository": E + "test_views_store_and_checks",
    "translation range 0.06 -> 0.04": E + "test_draw_ranges_against_the_keras_layers_of_the_source",
    "zoom range fixed at 0.92-1.08": B + "test_ranges_follow_the_recipe",
    "Monte Carlo scale reduced by the number of units": D + "test_montecarlo_scale_without_reduction",
    "Monte Carlo half difference of the mean not halved": D + "test_montecarlo_scale_without_reduction",
    "within the view variability at 1 x the scale": D + "test_montecarlo_scale_without_reduction",
    "forbidden folders not compared": D + "test_guard_by_identity_and_lower_case",
    "hard links not checked": D + "test_guard_by_identity_and_lower_case",
    "zones not allowlisted": D + "test_zones_and_outside",
    "outside rule not applied": D + "test_zones_and_outside",
    "input root accepted with the official root": ZB + "test_official_root_refuses_outside_inputs",
    "configuration outside the repository accepted with the official root": ZB + "test_official_root_refuses_outside_inputs",
    "unit keys not checked": ZB + "test_check_inputs",
    "parent files outside the list accepted": D + "test_parent_files_listed",
    "parent file SHA-256 not compared": ZB + "test_check_inputs",
    "card network, fold, seed or patience not compared": ZB + "test_check_inputs",
    "null note accepted by the trial": ZB + "test_refusals_before_any_folder",
    "dirty tree accepted": ZB + "test_refusals_before_any_folder",
    "sealed cell metrics published": ZB + "test_sealed_metrics",
    "cap not checked during the views": ZB + "test_cap_during_the_views",
    "frozen note not compared": ZB + "test_frozen_note_changed",
    "Keras module not compared": ZB + "test_keras_module_pin",
    "parent status not compared with its pin": ZB + "test_parent_status_pin",
    "configuration and code not rechecked at the end": ZB + "test_configuration_changed_during_the_trial",
    "authorisation flag not required": ZB + "test_authorisation_flag_required",
    "saved dev rows of another role accepted": ZB + "test_saved_dev_rows_of_the_test_role",
    "saved dev order not compared": ZB + "test_saved_dev_key_order",
    "a required name pattern dropped": ZB + "test_required_name_patterns",
    "budget not checked at the steps": ZB + "test_cap_incomplete_keeps_complete_units",
    "threshold other than 0.5 accepted": A + "test_closed_sets_and_fixed_values",
    "threads other than 10 accepted": A + "test_closed_sets_and_fixed_values",
    "test statement of the configuration not compared": A + "test_closed_sets_and_fixed_values",
    "results/archeval not required among the forbidden paths": A + "test_closed_sets_and_fixed_values",
    "attempt number not compared": ZB + "test_refusals_before_any_folder",
    "cap stop reported as FAIL": ZB + "test_cap_incomplete_keeps_complete_units",
    "launch: nice not checked": H + "test_nice_not_zero",
    "launch: trial on battery": H + "test_battery",
    "launch: low power mode accepted": H + "test_low_power_mode",
    "launch: powermode other than 0 accepted": H + "test_powermode_not_zero",
    "launch: a refusal takes the attempt's name": H + "test_refusal_then_launch_with_the_same_attempt",
    "launch: a failed check-inputs takes the attempt's name": H + "test_failed_check_inputs_stops_the_trial",
    "launch: views folder left accepted": H + "test_views_folder_left",
    "launch: failed check-inputs ignored": H + "test_failed_check_inputs_stops_the_trial",
    "launch: log folder inside the repository accepted": H + "test_refusals_of_the_arguments",
    "launch: existing conditions file overwritten": H + "test_refusals_of_the_arguments",
    "launch: dirty work tree accepted": H + "test_dirty_work_tree",
    "launch: folder that is not the root of a git work tree accepted": H + "test_inside_a_clean_git_work_tree_but_not_its_root",
    "launch: a trial refused before its run folder takes the attempt's name": H + "test_trial_refused_before_its_run_folder",
    "launch: failed git status taken as a clean work tree": H + "test_git_status_fails",
    "launch: start line condition inverted": H + "test_started_trial_that_fails_uses_the_attempt",
    "suite: the Keras randrange warning not filtered in the runner": "ZYSuiteOutput.test_no_warning_or_personal_path_in_the_output",
}
# mutazione di controllo: solo un commento; deve SOPRAVVIVERE (la suite rileva il comportamento, non il testo)
CONTROL = [("control: comment only", S, "def fit_digests(keys) -> list:\n", "def fit_digests(keys) -> list:  # control mutant: comment only\n")]
# dichiarate ridondanti, devono SOPRAVVIVERE:
REDUNDANT = [
    # la cartella delle viste ha un solo livello di file: shutil.rmtree aprirebbe solo la cartella stessa, con il suo
    # percorso assoluto (i file li toglie con unlink e dir_fd, senza aprirli), quindi oggi non urta la regola sui
    # percorsi relativi; remove_own_folder la evita comunque per costruzione
    ("declared redundant (must survive): views folder removed with shutil.rmtree", S, "    for e in os.listdir(p):\n        q = p / e",
     "    shutil.rmtree(p)\n    return\n    for e in os.listdir(p):\n        q = p / e"),
    # i conteggi di Moments sono provati a parte (viste x frame x H x W); la guardia confronta il numero ricavato dagli
    # stessi array, quindi scatta solo se Moments è sbagliato
    ("declared redundant (must survive): count guard of a re-estimation", S, "        if mom.n != want:", "        if False:"),
    # con ogni strato alimentato solo dal precedente (il controllo prima di questo), uno strato può essere consumato solo
    # dal successivo
    ("declared redundant (must survive): consumers check of the chain", S, "    for n, c in consumers.items():\n        if len(c) != 1:",
     "    for n, c in consumers.items():\n        if False:"),
]


def pairs(x) -> tuple:
    return x if isinstance(x, tuple) else (x,)


def source_of(rel: str) -> Path:
    """I file di questo comando da ROOT; ogni altro script dalla cartella dei moduli riusati."""
    return ROOT / rel if rel in OWN else REPO_SCRIPTS / Path(rel).name


def copy_tree(dst: Path) -> Path:
    code = dst / "code"
    (code / "scripts").mkdir(parents=True, exist_ok=True)
    for p in sorted(REPO_SCRIPTS.glob("*.py")):
        shutil.copy2(p, code / "scripts" / p.name)
    for rel in OWN:
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, code / rel)
    arch = REPO_SCRIPTS.parent.joinpath(*ARCH)
    if arch.is_file():
        code.joinpath(*ARCH[:-1]).mkdir(parents=True, exist_ok=True)
        shutil.copy2(arch, code.joinpath(*ARCH))
    return code


def suite_tests() -> set:
    """Le coppie Classe.metodo dei test della suite, lette dal sorgente."""
    out, cls = set(), None
    for line in (ROOT / TESTS).read_text(encoding="utf-8").splitlines():
        m = re.match(r"class (\w+)\(", line)
        if m:
            cls = m.group(1)
        m = re.match(r"    def (test_\w+)\(", line)
        if m and cls:
            out.add(f"{cls}.{m.group(1)}")
    return out


def last_error(stderr: str, test: str) -> str:
    """La riga dell'errore del test dall'uscita di unittest, con i percorsi tolti: la prima che inizia con il nome di
    un'eccezione (tipo e messaggio); in mancanza, l'ultima riga non indentata (con assertEqual su liste sarebbe una riga
    del diff, quindi solo come ripiego)."""
    cls, meth = test.split(".")
    # dopo l'intestazione unittest stampa la prima riga della docstring del test, se c'è
    m = re.search(rf"^(?:FAIL|ERROR): {meth} \((?:__main__\.)?{cls}[^)]*\)\n(?:[^\n]*\n)?-+\n(.*?)(?=^=+$|^-{{20,}}$|\Z)", stderr, re.M | re.S)
    if not m:
        return "no traceback"
    lines = [ln for ln in m.group(1).splitlines() if ln.strip() and not ln.startswith(" ")]
    named = [ln for ln in lines if re.match(r"^[\w.]*(Error|Exception|Failure|Exceeded)\b", ln)]
    text = named[0] if named else (lines[-1] if lines else "no message")
    return re.sub(r"(/[^\s'\"]+)+", "<path>", text)[:160]


def run_tests(code: Path, tmp: Path, test: str = None) -> tuple:
    """La suite sulla copia (intera, oppure solo il test dato): codice d'uscita, riepilogo, salti, test falliti in ordine
    (Classe.metodo) e stderr."""
    tmp.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp), "CUDA_VISIBLE_DEVICES": "-1"}
    r = subprocess.run([sys.executable, "-I", "-B", TESTS] + ([test] if test else []), cwd=code, env=env, capture_output=True, text=True,
                       timeout=3600)
    line = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith("tests run")), "no test summary")
    skips = tuple(sorted(ln for ln in r.stdout.splitlines() if ln.startswith("skipped ")))
    failed = []
    for m in re.finditer(r"^(?:FAIL|ERROR): (\w+) \((?:__main__\.)?([\w.]+)\)", r.stderr, re.M):
        parts = m.group(2).split(".")                  # «Classe» con la suite intera, «Classe.metodo» con un test solo
        cls = parts[-2] if len(parts) > 1 and parts[-1] == m.group(1) else parts[-1]
        name = f"{cls}.{m.group(1)}"
        if name not in failed:
            failed.append(name)
    return r.returncode, line, skips, failed, r.stderr


def verdict(rc: int, line: str, failed: list, expected) -> str:
    m = re.match(r"tests run: (\d+), failures: (\d+), errors: (\d+), skipped: (\d+)", line)
    if not m:
        return "invalid"
    if int(m.group(2)) + int(m.group(3)) > 0:
        return "killed" if expected is None or expected in failed else "invalid"
    if expected is not None and int(m.group(4)) > 0:
        return "invalid"                               # il test dichiarato è stato saltato
    return "survived" if rc == 0 else "invalid"


def one(tmp: Path, i: int, name: str, target: str, old, new, expected=None) -> tuple:
    work = tmp / f"work{i}"
    code = copy_tree(work)
    f = code / target
    text = f.read_text(encoding="utf-8")
    for o, n in zip(pairs(old), pairs(new)):
        text = text.replace(o, n, 1)
    f.write_text(text, encoding="utf-8")
    rc, line, skips, failed, err = run_tests(code, work / "tmp", expected)
    shutil.rmtree(work, ignore_errors=True)
    v = verdict(rc, line, failed, expected)
    note = f"{line}; first failing: {failed[0] if failed else 'none'}"
    if expected:
        note += f"; declared: {expected}" + ("" if expected in failed else " (not among the failing tests)")
        if expected in failed:
            note += f"; error: {last_error(err, expected)}"
    return name, v, note


def main() -> int:
    ap = argparse.ArgumentParser(description="Mutation check of pad_arch_bn_factorial.py and launch.zsh (offline, synthetic)")
    ap.add_argument("--list", action="store_true", help="list the mutations and check the patterns and the declared tests")
    ap.add_argument("--only", action="append", default=None, help="run only the named mutations (with the unmutated copy and the control)")
    a = ap.parse_args()
    tmpdir = Path(tempfile.gettempdir()).resolve()
    if tmpdir == ROOT.resolve() or ROOT.resolve() in tmpdir.parents or tmpdir in Path.home().resolve().parents or tmpdir == Path.home().resolve():
        print("ERROR: TMPDIR must be outside the repository and must not contain the home folder")
        print("RESULT: FAIL")
        return 1
    allm = MUTANTS + CONTROL + REDUNDANT
    problems = [f"{n}: the pattern occurs {k} times in {t} (expected exactly once)"
                for n, t, old, _ in allm for o in pairs(old) if (k := source_of(t).read_text(encoding="utf-8").count(o)) != 1]
    names = [n for n, *_ in allm]
    problems += [f"duplicated mutation name: {n}" for n in sorted({n for n in names if names.count(n) > 1})]
    problems += [f"--only {n!r}: no such mutation" for n in (a.only or []) if n not in names]
    tests = suite_tests()
    problems += [f"{n}: no declared test" for n, *_ in MUTANTS if n not in DECLARED]
    problems += [f"declared test of an unknown mutation: {n}" for n in DECLARED if n not in [m[0] for m in MUTANTS]]
    problems += [f"{n}: declared test {t} is not in the suite" for n, t in DECLARED.items() if t not in tests]
    for p in problems:
        print(f"ERROR: {p}", file=sys.stderr)
    if problems:
        print("RESULT: FAIL (a declared mutation no longer matches the code or the suite)")
        return 1
    if a.list:
        for n, t, _, _ in MUTANTS:
            print(f"{n}  [{t}] -> {DECLARED[n]}")
        for n, t, _, _ in CONTROL + REDUNDANT:
            print(f"{n}  [{t}] (must survive the whole suite)")
        print(f"{len(MUTANTS)} mutations, {len(CONTROL)} control mutation and {len(REDUNDANT)} declared redundant, every pattern found exactly "
              "once, every declared test in the suite")
        return 0
    mutants = [m for m in MUTANTS if a.only is None or m[0] in a.only]
    controls = [m for m in CONTROL + REDUNDANT if a.only is None or m[0] in a.only or m in CONTROL]
    killed = invalid = survived_controls = 0
    with tempfile.TemporaryDirectory(prefix="archbn2x2-mutations-") as tmp:
        tmp = Path(tmp)
        rc, line, base_skips, failed, _ = run_tests(copy_tree(tmp / "base"), tmp / "base" / "tmp")
        print(f"unmutated copy: {TESTS} exit {rc} ({line})", flush=True)
        if rc or not re.fullmatch(r"tests run: \d+, failures: 0, errors: 0, skipped: 0", line):
            print("RESULT: FAIL (the tests do not pass on the unmutated copy, or skip tests)")
            return 1
        for i, (n, t, old, new) in enumerate(controls):
            name, v, note = one(tmp, len(MUTANTS) + i, n, t, old, new)
            survived_controls += v == "survived"
            print(f"CONTROL   {v.upper():<9} {name} ({note})", flush=True)
        for i, (n, t, old, new) in enumerate(mutants):
            name, v, note = one(tmp, i, n, t, old, new, DECLARED[n])
            killed += v == "killed"
            invalid += v == "invalid"
            print(f"{v.upper():<9} {name} ({note})", flush=True)
            if v == "invalid":
                print(f"ERROR: invalid mutant {name!r}: the declared test did not fail ({note})", file=sys.stderr)
    print(f"mutations killed: {killed}/{len(mutants)}" + (f" ({invalid} invalid)" if invalid else "") + ("" if a.only is None else " (selection)"))
    print(f"control mutations survived: {survived_controls}/{len(controls)}")
    ok = killed == len(mutants) and survived_controls == len(controls)
    print("RESULT: " + ("PASS" if ok else "FAIL") + ("" if a.only is None else " (selection only: run without --only for the campaign)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
