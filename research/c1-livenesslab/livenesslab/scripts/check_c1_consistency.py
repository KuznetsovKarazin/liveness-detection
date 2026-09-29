"""
Controllo di coerenza degli artefatti C1, per insiemi: la cache dei punteggi (results/eval/<ds>.json), il CSV esportato
da eval_dataset.py (<ds>_scores.csv), il file per campione nello schema del gruppo (<run-id>/scores.csv) e la tabella
(<ds>_table.md) devono contenere esattamente gli stessi analizzatori e gli stessi campioni, senza righe mancanti, in più
o duplicate, con gli stessi punteggi (uguaglianza esatta dei float), le stesse decisioni alla soglia, le stesse metriche
a quattro decimali e gli stessi conteggi di errori e di immagini senza volto.

Insiemi attesi: campioni = manifest (<ds>_manifest.csv); analizzatori con punteggio = quelli registrati nell'app, non
esclusi sul dataset e non descrittivi/non addestrati; le righe senza punteggio (esclusi e descrittivi) devono comparire
nella tabella con n = 0 e una nota. Per la consegna C1 su NUAA: 30 righe numeriche × 300 campioni = 9.000 coppie
uniche e 5 righe senza punteggio.

Il rapporto va in <run-id>/consistency_check.txt; codice di uscita 1 se c'è anche un solo problema.
Politica sulle righe del CSV esportato e di scores.csv: ogni riga deve essere una coppia attesa (analizzatore con
punteggio × campione del manifest), presente una sola volta, con un punteggio finito in [0, 1]; le righe di analizzatori
esclusi o descrittivi non devono comparire (vengono rifiutate come "extra"); un punteggio vuoto o non finito (anche il
letterale "nan") è un problema, mai un motivo per ignorare la riga. Insiemi e duplicati sono verificati su tutte le righe,
prima di qualsiasi lettura dei punteggi.
--self-test: ripete il controllo su nove copie guaste (riga della tabella tolta, riga duplicata, denominatore alterato,
campione tolto da scores.csv, riga duplicata in scores.csv, analizzatore in più nel CSV, analizzatore in più nel CSV con
punteggio vuoto, coppia duplicata nel CSV con punteggio vuoto, coppia duplicata nel CSV con "nan") e pretende che
ciascuna venga rifiutata. Manifest e configurazione vengono cercati in --c1 e, se assenti, nella cartella del run (--run-dir), dove la
consegna li contiene.

Uso:  python scripts/check_c1_consistency.py [--dataset nuaa] --c1 results/c1 --run-dir <cartella del run> [--self-test]
(--c1 senza valore vale LIVENESSLAB_RESULTS_DIR/c1; --run-dir senza valore sceglie la cartella *-C1-* piu' recente in --c1).
Un'eccezione durante il controllo non interrompe lo script: diventa un problema nel rapporto, con la fase in cui e' avvenuta.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")      # un terminale non UTF-8 non deve far fallire la stampa del rapporto
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import tesi_app.analyzers  # noqa: E402,F401
from tesi_app import evaluation as ev  # noqa: E402
from tesi_app.core import registry  # noqa: E402
from tesi_app.paths import RESULTS  # noqa: E402

TABLE_COLS = ["n", "n_bona_fide", "n_attack", "apcer", "bpcer", "acer", "eer", "bpcer_at_apcer10", "auc", "accuracy", "n_errors", "n_noface"]
METRIC_COLS = {"apcer": 3, "bpcer": 4, "acer": 5, "eer": 6, "bpcer_at_apcer10": 7, "auc": 8, "accuracy": 9}   # indice in TABLE_COLS


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def finite01(txt):
    """Il punteggio come float se è un numero finito in [0, 1], altrimenti None (vuoto, testo, nan, inf, fuori intervallo)."""
    try:
        v = float(txt)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and 0.0 <= v <= 1.0 else None


def expected_sets(dataset: str, manifest: Path):
    """(campioni attesi {file: label}, analizzatori con punteggio, analizzatori senza punteggio) dal manifest e dal registro."""
    samples, dup_manifest = {}, []
    for r in csv.DictReader(manifest.open(encoding="utf-8")):
        if r["file"] in samples:
            dup_manifest.append(r["file"])
        samples[r["file"]] = int(r["label"])
    scored, unscored = set(), set()
    for a in registry.all():
        if dataset in a.excluded_datasets() or a.reliability() in ("untrained", "descriptive"):
            unscored.add(a.id)
        else:
            scored.add(a.id)
    return samples, scored, unscored, dup_manifest


def parse_table(path: Path):
    """(righe numeriche {aid: [valori]}, righe senza punteggio {aid: nota}, duplicati)."""
    numeric, unscored, dups = {}, {}, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("| analyzer") or line.startswith("| analizzatore") or line.startswith("|---"):
            continue
        parts = [x.strip() for x in line.strip().strip("|").split("|")]
        aid = parts[0]
        if aid in numeric or aid in unscored:
            dups.append(aid)
        if len(parts) == len(TABLE_COLS) + 1 and parts[1].isdigit() and int(parts[1]) > 0:
            numeric[aid] = parts[1:]
        elif len(parts) >= 3 and parts[1] == "0":
            unscored[aid] = parts[2]
        else:
            numeric[aid] = parts[1:]      # riga malformata: contata come numerica e poi rifiutata
    return numeric, unscored, dups


def run_checks(dataset: str, cache_path: Path, exported_csv: Path, scores_csv: Path, table_path: Path, manifest: Path, config_path: Path, state=None):
    """Esegue tutti i controlli; restituisce (problemi, righe del rapporto). `state` (dizionario) riceve la fase corrente e le
    righe già prodotte, così un'eccezione può essere riferita alla fase e al file che l'hanno causata."""
    state = state if state is not None else {}
    P, L = state.setdefault("problems", []), state.setdefault("lines", [])
    def stage(s): state["stage"] = s
    stage(f"reading the manifest ({manifest.name})")
    samples, scored, unscored, dup_manifest = expected_sets(dataset, manifest)
    if not samples:
        P.append(f"manifest: 0 samples ({manifest.name})")
    if dup_manifest:
        P.append(f"manifest: {len(dup_manifest)} duplicated sample files ({dup_manifest[0]} ...)")
    stage(f"reading the configuration ({config_path.name})")
    cfg = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    cfg_thr = float(cfg.get("conventions", {}).get("threshold", 0.5))
    if cfg and (n_declared := cfg.get("dataset", {})) and (n_declared.get("n_bona_fide") is not None):
        if (n_declared.get("n_bona_fide"), n_declared.get("n_attack")) != (sum(1 for y in samples.values() if y == 0), sum(1 for y in samples.values() if y == 1)):
            P.append("config: n_bona_fide / n_attack differ from the manifest")
    n_bf = sum(1 for y in samples.values() if y == 0); n_at = len(samples) - n_bf
    L.append(f"expected: {len(samples)} samples from the manifest ({n_bf} bona fide, {n_at} attacks); {len(scored)} analyzers with a score; {len(unscored)} rows without score ({', '.join(sorted(unscored))})")
    L.append(f"expected pairs (analyzer, sample): {len(scored)} x {len(samples)} = {len(scored) * len(samples)}")

    # 1) cache: insiemi esatti, punteggi validi
    stage(f"reading the score cache ({cache_path.name})")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    cache_scored = {a for a, per in cache.get("scores", {}).items() if any(v.get("s") is not None for v in per.values())}
    if cache_scored != scored:
        P.append(f"cache: analyzers with scores differ from expected: missing {sorted(scored - cache_scored)}, extra {sorted(cache_scored - scored)}")
    for aid in sorted(cache_scored & scored):
        keys = set(cache["scores"][aid]); exp = set(samples)
        if keys != exp:
            P.append(f"cache/{aid}: samples differ: missing {len(exp - keys)}, extra {len(keys - exp)}")
        bad = [k for k, v in cache["scores"][aid].items() if v.get("s") is None or not (0.0 <= float(v["s"]) <= 1.0)]
        if bad:
            P.append(f"cache/{aid}: {len(bad)} samples without a valid score in [0, 1]")
        wrong_label = [k for k, v in cache["scores"][aid].items() if k in samples and int(v.get("y", -1)) != samples[k]]
        if wrong_label:
            P.append(f"cache/{aid}: {len(wrong_label)} labels differ from the manifest")
    L.append(f"1. score cache ({cache_path.name}): {len(cache_scored)} analyzers with scores (expected {len(scored)}), each checked for exactly the {len(samples)} manifest samples, scores in [0, 1] and labels as in the manifest")

    # 2) CSV esportato: coppie uniche, insieme esatto, punteggi = cache, flag di errore / nessun volto = cache
    stage(f"reading the exported CSV ({exported_csv.name})")
    exp_rows = list(csv.DictReader(exported_csv.open(encoding="utf-8")))
    # insiemi e duplicati su TUTTE le righe, prima di guardare i punteggi: una riga con punteggio vuoto o "nan" conta
    # come riga a tutti gli effetti (in più, duplicata o mancante) e il suo punteggio viene rifiutato subito dopo
    pairs = Counter((r.get("analyzer") or "", r.get("file") or "") for r in exp_rows)
    dup = [k for k, c in pairs.items() if c > 1]
    if dup:
        P.append(f"exported CSV: {len(dup)} duplicated (analyzer, sample) pairs")
    expected_pairs = {(a, f) for a in scored for f in samples}
    got_pairs = set(pairs)
    if got_pairs != expected_pairs:
        P.append(f"exported CSV: pairs differ: missing {len(expected_pairs - got_pairs)}, extra {len(got_pairs - expected_pairs)}")
    bad_value = sum(1 for r in exp_rows if finite01(r.get("attack_score")) is None)
    if bad_value:
        P.append(f"exported CSV: {bad_value} rows whose attack_score is empty or not a finite number in [0, 1]")
    exp_map = {(r["analyzer"], r["file"]): r for r in exp_rows if finite01(r.get("attack_score")) is not None}
    def cached(k):
        v = cache.get("scores", {}).get(k[0], {}).get(k[1])
        return None if v is None else v.get("s")
    diff = sum(1 for k, r in exp_map.items() if k in expected_pairs and (cached(k) is None or float(r["attack_score"]) != float(cached(k)) or int(r["label"]) != samples.get(k[1], -1)))
    if diff:
        P.append(f"exported CSV: {diff} rows with score or label different from the cache/manifest")
    def flags(aid, key):
        """Conteggio dei campioni con il flag `key` ("e" errore, "nf" nessun volto) nella cache per l'analizzatore."""
        return sum(1 for v in cache.get("scores", {}).get(aid, {}).values() if v.get(key))
    def cached_flag(k, key):
        return "1" if (cache.get("scores", {}).get(k[0], {}).get(k[1]) or {}).get(key) else "0"
    bad_flags = sum(1 for k, r in exp_map.items() if k in expected_pairs and (r.get("error", "0") != cached_flag(k, "e") or r.get("no_face", "0") != cached_flag(k, "nf")))
    if bad_flags:
        P.append(f"exported CSV: {bad_flags} rows whose error / no_face flags differ from the cache")
    L.append(f"2. exported CSV ({exported_csv.name}): {len(exp_rows)} rows, {len(got_pairs)} unique (analyzer, sample) pairs (expected {len(expected_pairs)}, checked on every row before any score filter), {len(exp_rows) - bad_value} rows with a finite attack_score in [0, 1], {diff} rows differing from the cache/manifest, {len(dup)} duplicated pairs, {bad_flags} rows with wrong error / no_face flags")

    # 3) scores.csv (schema del gruppo): coppie uniche, insieme esatto, round-trip, decisione alla soglia, hash della configurazione
    stage(f"reading scores.csv ({scores_csv.name})")
    sc_rows = list(csv.DictReader(scores_csv.open(encoding="utf-8")))
    sc_pairs = Counter((r.get("analyzer_id") or "", r.get("sample_id") or "") for r in sc_rows)      # tutte le righe
    dup = [k for k, c in sc_pairs.items() if c > 1]
    if dup:
        P.append(f"scores.csv: {len(dup)} duplicated (analyzer, sample) pairs")
    if set(sc_pairs) != expected_pairs:
        P.append(f"scores.csv: pairs differ: missing {len(expected_pairs - set(sc_pairs))}, extra {len(set(sc_pairs) - expected_pairs)}")
    thr = cfg_thr; bad_pred = bad_score = bad_cfg = bad_thr = 0
    cfg_sha = sha256(config_path) if config_path.exists() else None
    for r in sc_rows:
        k = (r["analyzer_id"], r["sample_id"])
        if k not in expected_pairs:
            continue
        s = finite01(r.get("score_attack"))
        if s is None:
            bad_score += 1                                         # vuoto, non numerico, nan/inf o fuori [0, 1]
            continue
        if float(r["threshold"]) != cfg_thr:
            bad_thr += 1
        if cached(k) is None or r["score_attack"] != repr(cached(k)) or int(r["label"]) != samples.get(k[1], -1):
            bad_score += 1
        if int(r["prediction"]) != (1 if s > cfg_thr else 0):
            bad_pred += 1
        if cfg_sha and r["config_sha256"] != cfg_sha:
            bad_cfg += 1
    if bad_score:
        P.append(f"scores.csv: {bad_score} rows whose score_attack is empty, not a finite number in [0, 1], not the round-trip text of the cache float, or whose label differs")
    if bad_thr:
        P.append(f"scores.csv: {bad_thr} rows whose threshold differs from the configuration ({cfg_thr})")
    if bad_pred:
        P.append(f"scores.csv: {bad_pred} predictions inconsistent with score > {cfg_thr} (configuration threshold)")
    if bad_cfg:
        P.append(f"scores.csv: {bad_cfg} rows whose config_sha256 differs from {config_path.name}")
    L.append(f"3. scores.csv (group schema): {len(sc_rows)} rows, {len(sc_pairs)} unique pairs (expected {len(expected_pairs)}, checked on every row), {bad_score} rows with score_attack empty, non-finite, not the repr of the cache float or with a wrong label, {bad_pred} predictions inconsistent with score > {thr} (tie = bona fide), {bad_thr} rows with a threshold different from the configuration, {bad_cfg} rows with config_sha256 different from {config_path.name}")

    # 4) tabella: insiemi esatti di righe numeriche e senza punteggio, n e denominatori, metriche ricalcolate, conteggi di errori
    stage(f"reading the table ({table_path.name})")
    numeric, tab_unscored, dups = parse_table(table_path)
    if dups:
        P.append(f"table: duplicated rows {sorted(set(dups))}")
    if set(numeric) != scored:
        P.append(f"table: numeric rows differ from expected: missing {sorted(scored - set(numeric))}, extra {sorted(set(numeric) - scored)}")
    if set(tab_unscored) != unscored:
        P.append(f"table: rows without score differ from expected: missing {sorted(unscored - set(tab_unscored))}, extra {sorted(set(tab_unscored) - unscored)}")
    by_a = {}
    for (a, f), r in exp_map.items():
        if f in samples:
            by_a.setdefault(a, []).append((samples[f], float(r["attack_score"])))
    n_metric_rows = 0
    for aid in sorted(set(numeric) & scored):
        vals = numeric[aid]
        if len(vals) != len(TABLE_COLS):
            P.append(f"table/{aid}: {len(vals)} columns instead of {len(TABLE_COLS)}"); continue
        rows = by_a.get(aid, [])
        if not all(v.isdigit() for v in vals[:3]):
            P.append(f"table/{aid}: n / n_bona_fide / n_attack not numeric: {vals[0]} / {vals[1]} / {vals[2]}"); continue
        if int(vals[0]) != len(samples) or int(vals[1]) != n_bf or int(vals[2]) != n_at:
            P.append(f"table/{aid}: n / n_bona_fide / n_attack = {vals[0]} / {vals[1]} / {vals[2]}, expected {len(samples)} / {n_bf} / {n_at}")
        if not all(v.isdigit() for v in vals[10:12]):
            P.append(f"table/{aid}: n_errors / n_noface not numeric: {vals[10]} / {vals[11]}")
        elif (int(vals[10]), int(vals[11])) != (flags(aid, "e"), flags(aid, "nf")):
            P.append(f"table/{aid}: n_errors / n_noface = {vals[10]} / {vals[11]}, cache flags {flags(aid, 'e')} / {flags(aid, 'nf')}")
        if len(rows) != len(samples):
            continue
        stage(f"recomputing the metrics of {aid}")
        m = ev.compute_metrics([y for y, _ in rows], [s for _, s in rows])
        for key, col in METRIC_COLS.items():
            t = vals[col]
            if t in ("—", "", "None"):
                if m.get(key) is not None:
                    P.append(f"table/{aid}: {key} missing but computable")
            elif re.fullmatch(r"-?\d+(\.\d+)?", t) is None:
                P.append(f"table/{aid}: {key} not numeric: {t}")
            elif m.get(key) is None or f"{float(t):.4f}" != f"{float(m[key]):.4f}":
                P.append(f"table/{aid}: {key} = {t}, recomputed {m.get(key) if m.get(key) is None else f'{float(m[key]):.4f}'}")
        n_metric_rows += 1
    L.append(f"4. table ({table_path.name}): {len(numeric)} numeric rows (expected {len(scored)}), n = {len(samples)} and denominators {n_bf} / {n_at} on every row, n_errors / n_noface as in the cache, metrics recomputed from the scores and compared at 4 decimals on {n_metric_rows} rows; {len(tab_unscored)} rows without score (expected {len(unscored)})")
    return P, L


def safe_checks(**kw):
    """run_checks che non può interrompersi: un'eccezione diventa un problema nel rapporto, con la fase e il file in cui
    è avvenuta, e le righe già prodotte restano nel rapporto (che viene comunque scritto)."""
    state = {}
    try:
        return run_checks(state=state, **kw)
    except Exception as exc:  # noqa: BLE001
        return (state.get("problems", []) + [f"check aborted while {state.get('stage', 'starting')}: {type(exc).__name__}: {exc}"],
                state.get("lines", []) + ["(partial report: the check stopped before completing)"])


def self_test(dataset, cache_path, exported_csv, scores_csv, table_path, manifest, config_path):
    """Copie guaste degli artefatti: ciascuna deve produrre almeno un problema."""
    tmp = Path(tempfile.mkdtemp(prefix="c1-selftest-"))
    try:
        return _self_test(tmp, dataset, cache_path, exported_csv, scores_csv, table_path, manifest, config_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _self_test(tmp, dataset, cache_path, exported_csv, scores_csv, table_path, manifest, config_path):
    table = table_path.read_text(encoding="utf-8").splitlines(keepends=True)
    numeric_lines = [l for l in table if l.startswith("| ") and not l.startswith("| analy") and not l.startswith("|---") and l.split("|")[2].strip().isdigit() and int(l.split("|")[2]) > 0]
    if not numeric_lines:
        raise ValueError(f"no numeric row in {table_path.name}: nothing to remove")
    victim = numeric_lines[0]
    cells = [c.strip() for c in victim.strip().strip("|").split("|")]
    n_, nb_, na_ = int(cells[1]), int(cells[2]), int(cells[3])
    faults = {}
    t1 = tmp / "table_missing_row.md"; t1.write_text("".join(l for l in table if l != victim), encoding="utf-8")
    faults["table row removed (" + victim.split("|")[1].strip() + ")"] = dict(table_path=t1)
    t2 = tmp / "table_duplicate_row.md"; t2.write_text("".join(table) + victim, encoding="utf-8")
    faults["table row duplicated"] = dict(table_path=t2)
    t3 = tmp / "table_bad_denominator.md"; t3.write_text("".join(l.replace(f"| {n_} | {nb_} | {na_} |", f"| {n_} | {nb_ - 1} | {na_ + 1} |", 1) if l == victim else l for l in table), encoding="utf-8")
    faults["table denominator altered"] = dict(table_path=t3)
    sc = scores_csv.read_text(encoding="utf-8").splitlines(keepends=True)
    ex = exported_csv.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(sc) < 2 or len(ex) < 2:
        raise ValueError("scores.csv or the exported CSV has no data row: nothing to remove or duplicate")
    s1 = tmp / "scores_missing_sample.csv"; s1.write_text("".join(sc[:1] + sc[2:]), encoding="utf-8")
    faults["scores.csv sample removed"] = dict(scores_csv=s1)
    s2 = tmp / "scores_duplicate.csv"; s2.write_text("".join(sc) + sc[1], encoding="utf-8")
    faults["scores.csv row duplicated"] = dict(scores_csv=s2)
    e1 = tmp / "export_extra_analyzer.csv"; e1.write_text("".join(ex) + ex[1].replace(ex[1].split(",")[0], "ghost_analyzer", 1), encoding="utf-8")
    faults["exported CSV extra analyzer"] = dict(exported_csv=e1)
    def with_score(line, value):
        """La riga del CSV esportato con la colonna attack_score (la quarta) sostituita."""
        cells = line.rstrip("\n").split(","); cells[3] = value
        return ",".join(cells) + "\n"
    e2 = tmp / "export_extra_analyzer_empty_score.csv"; e2.write_text("".join(ex) + with_score(ex[1].replace(ex[1].split(",")[0], "ghost_analyzer", 1), ""), encoding="utf-8")
    faults["exported CSV extra analyzer with empty score"] = dict(exported_csv=e2)
    e3 = tmp / "export_duplicate_empty_score.csv"; e3.write_text("".join(ex) + with_score(ex[1], ""), encoding="utf-8")
    faults["exported CSV duplicated pair with empty score"] = dict(exported_csv=e3)
    e4 = tmp / "export_duplicate_nan_score.csv"; e4.write_text("".join(ex) + with_score(ex[1], "nan"), encoding="utf-8")
    faults["exported CSV duplicated pair with nan score"] = dict(exported_csv=e4)
    results = []
    for name, override in faults.items():
        kw = dict(dataset=dataset, cache_path=cache_path, exported_csv=exported_csv, scores_csv=scores_csv, table_path=table_path, manifest=manifest, config_path=config_path)
        kw.update(override)
        problems, _ = safe_checks(**kw)
        results.append((name, bool(problems), problems[:1]))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="nuaa"); ap.add_argument("--run-dir", default=None)
    ap.add_argument("--c1", default=str(RESULTS / "c1")); ap.add_argument("--self-test", action="store_true", help="verifica che il controllo rifiuti copie guaste degli artefatti")
    a = ap.parse_args()
    c1 = Path(a.c1)
    if a.run_dir:
        run_dir = Path(a.run_dir)
    else:
        candidates = [p for p in c1.glob("*-C1-*") if p.is_dir()]
        if not candidates:
            raise SystemExit(f"--run-dir not given and no run folder (*-C1-*) found in {c1}: pass --run-dir <restricted run folder>")
        run_dir = max(candidates, key=lambda p: p.stat().st_mtime)
    # manifest e configurazione: nella cartella dei risultati (--c1) se rigenerati, altrimenti nella cartella del run consegnata
    manifest = c1 / f"{a.dataset}_manifest.csv" if (c1 / f"{a.dataset}_manifest.csv").exists() else run_dir / f"{a.dataset}_manifest.csv"
    config = c1 / f"{a.dataset}_config.json" if (c1 / f"{a.dataset}_config.json").exists() else run_dir / f"{a.dataset}_config.json"
    paths = dict(dataset=a.dataset, cache_path=RESULTS / "eval" / f"{a.dataset}.json", exported_csv=c1 / f"{a.dataset}_scores.csv",
                 scores_csv=run_dir / "scores.csv", table_path=c1 / f"{a.dataset}_table.md", manifest=manifest, config_path=config)
    missing = [str(p) for k, p in paths.items() if k != "dataset" and not p.exists()]
    head = [f"Consistency check of the C1 artefacts, run {run_dir.name}, {time.strftime('%Y-%m-%d %H:%M')}"]
    if missing:
        out = head + ["  ! missing file(s): " + ", ".join(missing), "result: 1 problems (check not performed)"]
        if run_dir.is_dir():
            (run_dir / "consistency_check.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
        print("\n".join(out))
        return 1
    problems, lines = safe_checks(**paths)
    out = head + [f"score cache: results/eval/{a.dataset}.json sha256 {sha256(paths['cache_path'])}",
                  f"scores.csv sha256 {sha256(paths['scores_csv'])}", ""] + lines
    for p in problems:
        out.append("  ! " + p)
    out.append(f"result: {'OK, no inconsistency' if not problems else str(len(problems)) + ' problems'}")
    if a.self_test and problems:
        out.append("")
        out.append("self-test skipped: the main check already reports problems (the self-test is meaningful only on a consistent package)")
    elif a.self_test:
        out.append("")
        out.append("self-test (each faulty copy must be rejected):")
        try:
            st = self_test(**paths)
        except Exception as exc:  # noqa: BLE001
            st = [(f"faulty copies could not be built ({type(exc).__name__}: {exc})", False, [])]
        for name, rejected, first in st:
            out.append(f"  {'rejected' if rejected else 'NOT DETECTED'}: {name}" + (f" -> {first[0]}" if first else ""))
        if not all(r for _, r, _ in st):
            problems.append("self-test: a faulty copy was not detected")
        out.append(f"self-test result: {'all faults detected' if all(r for _, r, _ in st) else 'FAILED'}")
    (run_dir / "consistency_check.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
