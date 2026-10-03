"""
Controllo di coerenza degli artefatti C1 (e M1, che ne ripete lo schema), per insiemi: la cache dei punteggi
(results/eval/<ds>.json), il CSV esportato da eval_dataset.py (<ds>_scores.csv), il file per campione nello schema del
gruppo (<run-id>/scores.csv), la tabella (<ds>_table.md) e, se presente, la tabella per tipo di attacco
(<ds>_table_by_attack_type.md) devono contenere esattamente gli stessi analizzatori e gli stessi campioni, senza righe
mancanti, in più o duplicate, con gli stessi punteggi, le stesse decisioni alla soglia, le stesse metriche e gli stessi
conteggi di errori e di immagini senza volto.

Insiemi attesi: campioni = manifest (<ds>_manifest.csv); analizzatori con punteggio = quelli registrati nell'app, non
esclusi sul dataset e non descrittivi/non addestrati; le righe senza punteggio (esclusi e descrittivi) devono comparire
nella tabella esattamente come [analizzatore, 0, nota], con la nota che eval_dataset.py scrive (TXT/translate_note).
Per la consegna C1 su NUAA: 30 righe numeriche × 300 campioni = 9.000 coppie uniche e 5 righe senza punteggio.

Il rapporto va in <run-id>/consistency_check.txt; codice di uscita 1 se c'è anche un solo problema.

Politica di formato, uniforme: ogni valore scritto da un programma si confronta come TESTO con il testo che quel
programma scrive, mai come numero riletto. Celle intere della tabella = str(valore) ("0300", "00" o cifre a larghezza
piena sono rifiutati); metriche = f"{valore:.4f}" oppure "—" se non calcolabili; attack_score del CSV esportato = repr del
float della cache (come lo scrive csv.writer); score_attack di scores.csv = repr(float); threshold di scores.csv =
repr(float) della soglia di configurazione; label, decisione, error e no_face = "0" o "1" esatti (niente spazi).

Tabelle Markdown, lettura rigorosa: la prima riga che (tolti gli spazi) inizia con "|" deve essere l'intestazione
attesa costruita dalla lista delle colonne, seguita dal separatore esatto; da lì all'ultima riga con "|" ogni riga è una
riga della tabella e deve essere in forma canonica ("| " + " | ".join(celle) + " |"); righe vuote o di testo nel mezzo,
un'intestazione o un separatore ripetuti (seconda tabella), una riga non canonica sono problemi.

Altri controlli: impronte (cache "fingerprints" = colonna fingerprint del CSV esportato = analyzers[].fingerprint della
configurazione dove c'è; checkpoint_sha256 di scores.csv = SHA-256 del checkpoint della configurazione se l'impronta ne
contiene il prefisso di 16 caratteri esadecimali, altrimenti l'impronta testuale, come in make_c1_scores.py ed
export_m1.py); colonne dello schema del gruppo (research/templates/score-schema.json) tutte presenti, con run_id = nome
della cartella del run, dataset_id, split "test", modality "RGB", subject_id / session_id = colonne subject / session del
manifest, video_id vuoto, attack_type del manifest (NUAA senza colonna: "print") per gli attacchi e vuoto per i bona fide;
SHA-256 del manifest = dataset.manifest_sha256 della configurazione; cache["dataset"] = dataset e, con
--expected-cache-sha256, SHA-256 della cache uguale a quello dato; con --expected-config-sha256 e
--expected-manifest-sha256, SHA-256 della configurazione e del manifest uguali a quelli dati (la configurazione e il
manifest non sono più ancorati solo a sé stessi). Il valore di ciascuna opzione --expected-*-sha256 si normalizza
(spazi iniziali e finali tolti, maiuscole portate a minuscole: lo SHA-256 esadecimale non distingue maiuscole e minuscole)
e poi deve essere di esattamente 64 caratteri esadecimali; un valore vuoto, di soli spazi, troncato, troppo lungo o con
caratteri non esadecimali è un errore: messaggio ERROR, codice di uscita 1 e nessun controllo eseguito (il rapporto
consistency_check.txt non viene scritto). Chiavi JSON duplicate in cache e configurazione. Configurazione:
analyzers[].id senza duplicati e uguali alle righe attese della tabella (con punteggio e senza), campo "excluded"
esattamente sugli analizzatori esclusi dal dataset; per ogni analizzatore con punteggio la cui impronta termina con uno
SHA-256 completo (64 caratteri esadecimali) deve esistere un checkpoint con weights_file e sha256 identico, e con un
prefisso di 16 caratteri esadecimali un checkpoint il cui sha256 inizia con quel prefisso. Manifest NUAA: subject = prefisso
a 4 cifre del nome del file.
Tabella per tipo di attacco: stesse regole di lettura, righe = analizzatori con punteggio × tipi del manifest, valori
ricalcolati con evaluation.metrics_by_attack_type e confrontati come testo; la tabella principale che contiene deve
essere identica, riga per riga, a <ds>_table.md.

--self-test: ripete il controllo su copie guaste di ogni artefatto (tabella, CSV esportato, scores.csv, manifest,
configurazione, cache, tabella per tipo di attacco; se quest'ultima manca viene sintetizzata dai valori attesi e prima
verificata come valida) e pretende per ciascuna il messaggio strutturale atteso, non un problema qualsiasi; verifica
inoltre la normalizzazione dei valori delle opzioni --expected-*-sha256 (valori malformati rifiutati, valori validi con
spazi o maiuscole accettati), su una riga a parte che non cambia il numero delle copie guaste.
Manifest e configurazione vengono cercati in --c1 e, se assenti, nella cartella del run (--run-dir).

Uso:  python scripts/check_c1_consistency.py [--dataset nuaa] --c1 results/c1 --run-dir <cartella del run> [--self-test]
      [--by-type-table <file>] [--expected-cache-sha256 <sha>] [--expected-config-sha256 <sha>] [--expected-manifest-sha256 <sha>]
(--c1 senza valore vale LIVENESSLAB_RESULTS_DIR/c1; --run-dir senza valore sceglie la cartella *-C1-* o *-M1-* piu'
recente in --c1). Un'eccezione durante il controllo non interrompe lo script: diventa un problema nel rapporto, con la
fase in cui e' avvenuta.
"""
import argparse
import csv
import hashlib
import importlib.util
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
# testi della tabella come li scrive eval_dataset.py (intestazione, note delle righe senza punteggio, titolo della tabella
# per tipo di attacco): il modulo si importa per file, prima di tesi_app (importa torch prima di TensorFlow)
_spec = importlib.util.spec_from_file_location("eval_dataset", ROOT / "scripts" / "eval_dataset.py")
ed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ed)
import tesi_app.analyzers  # noqa: E402,F401
from tesi_app import evaluation as ev  # noqa: E402
from tesi_app.core import registry  # noqa: E402
from tesi_app.paths import RESULTS  # noqa: E402

TABLE_COLS = ["n", "n_bona_fide", "n_attack", "apcer", "bpcer", "acer", "eer", "bpcer_at_apcer10", "auc", "accuracy", "n_errors", "n_noface"]
METRIC_COLS = {"apcer": 3, "bpcer": 4, "acer": 5, "eer": 6, "bpcer_at_apcer10": 7, "auc": 8, "accuracy": 9}   # indice in TABLE_COLS
BY_TYPE_COLS = ["attack_type", "n_attack", "accepted", "apcer", "n_bona_fide", "rejected_bona_fide", "bpcer"]
# colonne di research/templates/score-schema.json (schema_version 1.0); scores.csv aggiunge analyzer_id
SCHEMA_COLS = ["run_id", "sample_id", "dataset_id", "split", "subject_id", "session_id", "video_id", "attack_type", "modality", "label",
               "score_attack", "threshold", "prediction", "checkpoint_sha256", "config_sha256"]
EXPORT_COLS = ["analyzer", "file", "label", "attack_score", "error", "no_face", "fingerprint"]
SPLIT, MODALITY = "test", "RGB"                 # valori dei file consegnati (insieme di test, immagini RGB)
LANGS = ("en", "it")
FILE_KEYS = ("cache_path", "exported_csv", "scores_csv", "table_path", "manifest", "config_path", "by_type_path")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def normalize_sha256(value):
    """Valore di un'opzione --expected-*-sha256 normalizzato (spazi esterni tolti, minuscole) se è uno SHA-256 di 64
    caratteri esadecimali, altrimenti None (vuoto, solo spazi, troncato, troppo lungo, caratteri non esadecimali)."""
    v = value.strip().lower() if isinstance(value, str) else ""
    return v if re.fullmatch(r"[0-9a-f]{64}", v) else None


# valori di prova per la normalizzazione delle opzioni (--self-test): (descrizione, valore, atteso)
SHA_OPTION_CASES = [("empty", "", None), ("whitespace only", "   ", None), ("truncated (63 characters)", "a" * 63, None),
                    ("too long (65 characters)", "a" * 65, None), ("non-hexadecimal character", "g" + "a" * 63, None),
                    ("inner space", "a" * 32 + " " + "a" * 31, None),
                    ("upper case", "A" * 64, "a" * 64), ("surrounding spaces and newline", " " + "a" * 64 + "\n", "a" * 64)]


def option_self_test():
    """Verifica della normalizzazione delle opzioni --expected-*-sha256: (tutto giusto, riga del rapporto)."""
    wrong = [name for name, val, want in SHA_OPTION_CASES if normalize_sha256(val) != want]
    n_bad = sum(1 for _, _, want in SHA_OPTION_CASES if want is None)
    line = (f"option values (--expected-*-sha256): {n_bad} malformed values rejected, {len(SHA_OPTION_CASES) - n_bad} valid values "
            "with upper case or surrounding spaces accepted and lowered" if not wrong else
            f"option values (--expected-*-sha256): NOT handled as expected: {wrong}")
    return not wrong, line


def finite01(txt):
    """Il punteggio come float se è un numero finito in [0, 1], altrimenti None (vuoto, testo, nan, inf, fuori intervallo)."""
    try:
        v = float(txt)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and 0.0 <= v <= 1.0 else None


def label01(txt):
    """La label (o la decisione) come int se il testo è esattamente "0" o "1", altrimenti None (vuota, assente, spazi,
    testo, altro numero). Un int 0/1 (label della cache) vale come il suo testo."""
    t = str(txt) if txt is not None and not isinstance(txt, bool) else ""
    return int(t) if t in ("0", "1") else None


def finite(txt):
    """Il valore come float se è un numero finito, altrimenti None (vuoto, assente, testo, nan, inf)."""
    try:
        v = float(txt)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def cell(v):
    """Testo di una cella come lo scrive eval_dataset.py: "—" se non calcolabile, 4 decimali per un float, str per un intero."""
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def md_row(cells) -> str:
    """Riga Markdown canonica."""
    return "| " + " | ".join(cells) + " |"


def load_json(path: Path, what: str, P: list):
    """JSON con controllo delle chiavi duplicate (json.loads terrebbe in silenzio l'ultimo valore)."""
    dups = []
    def hook(pairs):
        dups.extend(k for k, c in Counter(k for k, _ in pairs).items() if c > 1)
        return dict(pairs)
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=hook)
    if dups:
        P.append(f"{what}: {len(dups)} duplicated JSON keys ({dups[0]} ...): the last value would silently win")
    return data


def missing_columns(path: Path, required):
    """Colonne richieste assenti dall'intestazione del CSV (lista vuota se ci sono tutte)."""
    with path.open(encoding="utf-8") as f:
        header = next(csv.reader(f), [])
    return [c for c in required if c not in header]


def malformed_rows(rows) -> int:
    """Righe con celle in più (DictReader le mette sotto la chiave None) o in meno (valori None)."""
    return sum(1 for r in rows if None in r or any(v is None for v in r.values()))


def expected_sets(dataset: str, manifest: Path):
    """(campioni attesi {file: label}, righe del manifest {file: riga}, tipi di attacco {file: tipo}, analizzatori con
    punteggio, analizzatori senza punteggio, duplicati, righe malformate) dal manifest e dal registro. Una riga con file
    vuoto o label diversa da "0"/"1" non entra nei campioni ed è contata. Tipi come in eval_dataset.read_attack_types:
    colonna attack_type se c'è; senza colonna, per NUAA "print" (solo foto stampate), altrove "unknown"; "" per i bona fide."""
    samples, meta, dup_manifest, bad_manifest = {}, {}, [], 0
    with manifest.open(encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        has_type = "attack_type" in (rd.fieldnames or [])
        for r in rd:
            f, y = r.get("file") or "", label01(r.get("label"))
            if not f or y is None:
                bad_manifest += 1
                continue
            if f in samples:
                dup_manifest.append(f)
            samples[f] = y; meta[f] = r
    types = {}
    for f, y in samples.items():
        t = (meta[f].get("attack_type") or "").strip() if has_type else ""
        if y == 1 and not t:
            t = "print" if not has_type and dataset == "nuaa" else "unknown"
        types[f] = t if y == 1 else ""
    scored, unscored = set(), set()
    for a in registry.all():
        if dataset in a.excluded_datasets() or a.reliability() in ("untrained", "descriptive"):
            unscored.add(a.id)
        else:
            scored.add(a.id)
    return samples, meta, types, scored, unscored, dup_manifest, bad_manifest


def expected_note(aid: str, dataset: str, lang: str) -> str:
    """Nota della riga senza punteggio come la scrive eval_dataset.py (nota italiana del modulo di valutazione, tradotta)."""
    a = registry.get(aid)
    it = a.exclusion_note(dataset) if dataset in a.excluded_datasets() else ed.TXT["it"]["note_noscore"]
    return ed.translate_note(it, lang)


def checkpoint_sha(fp, cfg: dict) -> str:
    """Valore atteso di checkpoint_sha256, con la regola di make_c1_scores.py ed export_m1.py: se l'ultimo campo
    dell'impronta inizia con 16 caratteri esadecimali che sono il prefisso dello SHA-256 di un checkpoint della
    configurazione (con weights_file), quello SHA-256 completo; altrimenti l'impronta testuale stessa."""
    prefix = (fp or "").split(":")[-1][:16]
    if not re.fullmatch(r"[0-9a-f]{16}", prefix):
        return fp or ""
    return next((c["sha256"] for c in cfg.get("checkpoints", []) if isinstance(c, dict) and c.get("weights_file")
                 and str(c.get("sha256", "")).startswith(prefix)), fp or "")


def parse_md_table(lines, cols, what: str, P: list, offset: int = 0):
    """Lettura rigorosa di una tabella Markdown in `lines`: (lingua dell'intestazione, [(numero di riga, celle)]).
    Intestazione e separatore solo per uguaglianza esatta con il testo atteso; ogni riga fra il separatore e l'ultima
    riga con "|" è una riga della tabella e deve essere canonica; il resto diventa un problema."""
    headers = {md_row([ed.TXT[lg]["analyzer"]] + cols): lg for lg in LANGS}
    header_cells = {tuple(c.lower() for c in [ed.TXT[lg]["analyzer"]] + cols) for lg in LANGS}
    sep = "|" + "---|" * (len(cols) + 1)
    idx = [i for i, l in enumerate(lines) if l.strip().startswith("|")]
    if not idx:
        P.append(f"{what}: no table found")
        return "en", []
    h = idx[0]
    lang = headers.get(lines[h])
    if lang is None:
        P.append(f"{what}: header at line {h + 1 + offset} differs from the expected one: {lines[h]!r} instead of {md_row(['analyzer'] + cols)!r}")
        lang = "en"
    start = h + 1
    if start < len(lines) and lines[start] == sep:
        start += 1
    else:
        P.append(f"{what}: missing separator line after the header (line {start + 1 + offset}, expected {sep!r})")
    rows = []
    for i in range(start, idx[-1] + 1):
        line, s = lines[i], lines[i].strip()
        where = f"line {i + 1 + offset}"
        if not s.startswith("|"):
            P.append(f"{what}: {where} inside the table is not a table row (blank line or text): {line!r}")
            continue
        cells = [c.strip() for c in s[1:-1].split("|")] if len(s) > 1 and s.endswith("|") else None
        if re.fullmatch(r"\|(\s*:?-+:?\s*\|)+", s) or (cells is not None and tuple(c.lower() for c in cells) in header_cells):
            P.append(f"{what}: second table at {where} (header or separator repeated): {line!r}")
            continue
        if cells is None or line != md_row(cells):
            P.append(f"{what}: row at {where} is not in the canonical form '| cell | ... |': {line!r}")
            continue
        rows.append((i + 1 + offset, cells))
    return lang, rows


def split_by_type_file(text: str, what: str, P: list):
    """(righe della tabella principale inclusa, righe della sezione per tipo, riga d'inizio della sezione) di un file
    <ds>_table_by_attack_type.md: le due parti sono separate dall'unico titolo "## APCER by attack type"."""
    lines = text.splitlines()
    titles = {"## " + ed.TXT[lg]["by_type_title"] for lg in LANGS}
    hs = [i for i, l in enumerate(lines) if l in titles]
    if len(hs) != 1:
        P.append(f"{what}: {len(hs)} headings {sorted(titles)} instead of exactly one")
        return lines, [], len(lines)
    return lines[:hs[0]], lines[hs[0] + 1:], hs[0] + 1


def expected_by_type(cache: dict, samples: dict, types: dict, scored, thr: float):
    """({(analizzatore, tipo): celle attese dopo l'analizzatore}, tipi attesi) ricalcolati dalla cache con
    evaluation.metrics_by_attack_type; un analizzatore senza tutti i punteggi validi viene saltato (già segnalato)."""
    exp_types = sorted({t for f, t in types.items() if samples.get(f) == 1})
    out = {}
    for aid in sorted(scored):
        per = cache.get("scores", {}).get(aid, {})
        recs = [(samples[f], finite01((per.get(f) or {}).get("s")), types[f]) for f in samples]
        if any(s is None for _, s, _ in recs):
            continue
        r = ev.metrics_by_attack_type([y for y, _, _ in recs], [s for _, s, _ in recs], [t for _, _, t in recs], thr, exp_types)
        for t, d in r["by_type"].items():
            out[(aid, t)] = [t, str(d["n_attack"]), str(d["accepted"]), cell(d["apcer"]), str(r["n_bona_fide"]),
                             str(r["rejected_bona_fide"]), cell(r["bpcer"])]
    return out, exp_types


def run_checks(dataset: str, cache_path: Path, exported_csv: Path, scores_csv: Path, table_path: Path, manifest: Path, config_path: Path,
               run_id=None, by_type_path=None, expected_cache_sha256=None, expected_config_sha256=None, expected_manifest_sha256=None,
               state=None):
    """Esegue tutti i controlli; restituisce (problemi, righe del rapporto). `state` (dizionario) riceve la fase corrente e le
    righe già prodotte, così un'eccezione può essere riferita alla fase e al file che l'hanno causata."""
    state = state if state is not None else {}
    P, L = state.setdefault("problems", []), state.setdefault("lines", [])
    def stage(s): state["stage"] = s
    if ed.COLS != TABLE_COLS or ed.BY_TYPE_COLS != BY_TYPE_COLS:
        P.append("eval_dataset.py writes table columns different from the ones this check expects")
    stage(f"reading the manifest ({manifest.name})")
    if miss := missing_columns(manifest, ("file", "label", "subject", "session")):
        P.append(f"manifest: missing column(s) {miss} ({manifest.name})")
    samples, meta, types, scored, unscored, dup_manifest, bad_manifest = expected_sets(dataset, manifest)
    if n_mal := malformed_rows(csv.DictReader(manifest.open(encoding="utf-8"))):
        P.append(f"manifest: {n_mal} malformed rows (more or fewer cells than the header)")
    if not samples:
        P.append(f"manifest: 0 samples ({manifest.name})")
    if dup_manifest:
        P.append(f"manifest: {len(dup_manifest)} duplicated sample files ({dup_manifest[0]} ...)")
    if bad_manifest:
        P.append(f"manifest: {bad_manifest} rows with an empty file or a label that is not 0 or 1 (left out of the expected samples)")
    if dataset == "nuaa":
        # soggetto NUAA = prefisso a 4 cifre del nome del file (SSSS_SSSS_...): una colonna subject riscritta viene rifiutata
        bad_subj = sorted(f for f, r in meta.items() if not re.fullmatch(r"\d{4}", r.get("subject") or "")
                          or not Path(f).name.startswith((r.get("subject") or "") + "_") or not re.match(r"\d{4}_", Path(f).name))
        if bad_subj:
            P.append(f"manifest: {len(bad_subj)} rows whose subject differs from the 4-digit prefix of the file name ({bad_subj[0]} ...)")
    if expected_manifest_sha256 and sha256(manifest) != expected_manifest_sha256.strip().lower():
        P.append(f"manifest: SHA-256 {sha256(manifest)} differs from --expected-manifest-sha256 {expected_manifest_sha256}")
    stage(f"reading the configuration ({config_path.name})")
    cfg = load_json(config_path, "config", P) if config_path.exists() else {}
    if expected_config_sha256 and (not config_path.exists() or sha256(config_path) != expected_config_sha256.strip().lower()):
        P.append(f"config: SHA-256 {sha256(config_path) if config_path.exists() else '(missing file)'} differs from --expected-config-sha256 {expected_config_sha256}")
    # analizzatori della configurazione = righe attese della tabella (con punteggio e senza), senza duplicati; "excluded"
    # esattamente sugli analizzatori esclusi dal dataset
    cfg_an = [an for an in cfg.get("analyzers", []) if isinstance(an, dict)] if isinstance(cfg.get("analyzers", []), list) else []
    cfg_ids = [an.get("id") for an in cfg_an]
    if cfg:
        if dup_ids := sorted({str(i) for i in cfg_ids if cfg_ids.count(i) > 1}):
            P.append(f"config: duplicated analyzers[].id {dup_ids}")
        if set(cfg_ids) != scored | unscored:
            P.append(f"config: analyzers differ from the table rows (analyzers with a score and rows without score): "
                     f"missing {sorted(map(str, (scored | unscored) - set(cfg_ids)))}, extra {sorted(map(str, set(cfg_ids) - (scored | unscored)))}")
        excl_exp = {a.id for a in registry.all() if dataset in a.excluded_datasets()}
        excl_cfg = {an.get("id") for an in cfg_an if an.get("excluded")}
        if excl_cfg != excl_exp:
            P.append(f"config: analyzers marked 'excluded' differ from the analyzers excluded on {dataset}: "
                     f"missing {sorted(map(str, excl_exp - excl_cfg))}, extra {sorted(map(str, excl_cfg - excl_exp))}")
    cfg_thr = finite(cfg.get("conventions", {}).get("threshold", 0.5))
    if cfg_thr is None:
        # soglia assente o non numerica nella configurazione: problema, e i controlli proseguono con 0,5
        P.append(f"config: conventions.threshold is not a finite number ({cfg.get('conventions', {}).get('threshold')!r}); 0.5 used for the checks")
        cfg_thr = 0.5
    cfg_ds = cfg.get("dataset", {}) if isinstance(cfg.get("dataset", {}), dict) else {}
    if cfg_ds.get("id") is not None and cfg_ds.get("id") != dataset:
        P.append(f"config: dataset.id is {cfg_ds.get('id')!r}, expected {dataset!r}")
    if cfg and (n_declared := cfg_ds) and (n_declared.get("n_bona_fide") is not None):
        if (n_declared.get("n_bona_fide"), n_declared.get("n_attack")) != (sum(1 for y in samples.values() if y == 0), sum(1 for y in samples.values() if y == 1)):
            P.append("config: n_bona_fide / n_attack differ from the manifest")
    man_sha, man_decl = sha256(manifest), cfg_ds.get("manifest_sha256")
    if not man_decl:
        P.append(f"config: dataset.manifest_sha256 missing ({config_path.name}): the manifest cannot be verified")
    elif man_sha != man_decl:
        P.append(f"manifest: SHA-256 {man_sha} differs from config dataset.manifest_sha256 {man_decl}")
    n_bf = sum(1 for y in samples.values() if y == 0); n_at = len(samples) - n_bf
    L.append(f"expected: {len(samples)} samples from the manifest ({n_bf} bona fide, {n_at} attacks); {len(scored)} analyzers with a score; {len(unscored)} rows without score ({', '.join(sorted(unscored))})")
    L.append(f"expected pairs (analyzer, sample): {len(scored)} x {len(samples)} = {len(scored) * len(samples)}")
    L.append(f"0. manifest ({manifest.name}): SHA-256 {'equal to' if man_sha == man_decl else 'NOT equal to'} config dataset.manifest_sha256"
             + (f", {'equal to' if man_sha == expected_manifest_sha256.strip().lower() else 'NOT equal to'} --expected-manifest-sha256" if expected_manifest_sha256 else "")
             + ("; subject = 4-digit prefix of the file name on every row" if dataset == "nuaa" else ""))
    L.append(f"0. configuration ({config_path.name}): " + (f"SHA-256 {'equal to' if config_path.exists() and sha256(config_path) == expected_config_sha256.strip().lower() else 'NOT equal to'} --expected-config-sha256; " if expected_config_sha256 else "")
             + f"{len(cfg_ids)} analyzers (expected {len(scored | unscored)} = table rows with and without score), 'excluded' on the analyzers excluded on {dataset}")

    # 1) cache: dataset, SHA-256 atteso, insiemi esatti, punteggi validi, impronte
    stage(f"reading the score cache ({cache_path.name})")
    cache = load_json(cache_path, "cache", P)
    if "dataset" in cache and cache["dataset"] != dataset:
        P.append(f"cache: dataset is {cache['dataset']!r}, expected {dataset!r}")
    if expected_cache_sha256 and sha256(cache_path) != expected_cache_sha256.strip().lower():
        P.append(f"cache: SHA-256 {sha256(cache_path)} differs from --expected-cache-sha256 {expected_cache_sha256}")
    cache_scored = {a for a, per in cache.get("scores", {}).items() if any(v.get("s") is not None for v in per.values())}
    if cache_scored != scored:
        P.append(f"cache: analyzers with scores differ from expected: missing {sorted(scored - cache_scored)}, extra {sorted(cache_scored - scored)}")
    for aid in sorted(cache_scored & scored):
        keys = set(cache["scores"][aid]); exp = set(samples)
        if keys != exp:
            P.append(f"cache/{aid}: samples differ: missing {len(exp - keys)}, extra {len(keys - exp)}")
        bad = [k for k, v in cache["scores"][aid].items() if finite01(v.get("s")) is None]
        if bad:
            P.append(f"cache/{aid}: {len(bad)} samples without a valid score in [0, 1]")
        wrong_label = [k for k, v in cache["scores"][aid].items() if k in samples and label01(v.get("y")) != samples[k]]   # y assente o vuota = diversa
        if wrong_label:
            P.append(f"cache/{aid}: {len(wrong_label)} labels differ from the manifest")
    cache_fp = cache.get("fingerprints", {}) if isinstance(cache.get("fingerprints", {}), dict) else {}
    no_fp = sorted(a for a in scored if not isinstance(cache_fp.get(a), str) or not cache_fp.get(a))
    if no_fp:
        P.append(f"cache: analyzers without a fingerprint: {no_fp}")
    cfg_fp = {an.get("id"): an.get("fingerprint") for an in cfg.get("analyzers", []) if isinstance(an, dict)}
    bad_cfg_fp = sorted(a for a in scored if a in cfg_fp and cfg_fp[a] != cache_fp.get(a))
    if bad_cfg_fp:
        P.append(f"config: analyzers[].fingerprint differs from the cache for {bad_cfg_fp}")
    # impronta con SHA-256 completo (64 esadecimali) o prefisso di 16 esadecimali: il checkpoint della configurazione deve
    # esistere con quello SHA-256 (identico o con quel prefisso); un checkpoint riscritto nella configurazione viene rifiutato
    ck_sha = [str(c.get("sha256", "")) for c in cfg.get("checkpoints", []) if isinstance(c, dict) and c.get("weights_file")]
    bad_ck = []
    for a in sorted(scored):
        last = (cache_fp.get(a) or "").split(":")[-1]
        if re.fullmatch(r"[0-9a-f]{64}", last):
            if last not in ck_sha:
                bad_ck.append(a)
        elif re.fullmatch(r"[0-9a-f]{16}", last):
            if not any(s.startswith(last) for s in ck_sha):
                bad_ck.append(a)
    if cfg and bad_ck:
        P.append(f"config: checkpoints[].sha256 does not match the fingerprint for {bad_ck}")
    exp_ck = {a: checkpoint_sha(cache_fp.get(a), cfg) for a in scored}
    L.append(f"1. score cache ({cache_path.name}): dataset {cache.get('dataset')!r}, {len(cache_scored)} analyzers with scores (expected {len(scored)}), each checked for exactly the {len(samples)} manifest samples, scores in [0, 1] and labels as in the manifest; fingerprints present and equal to the configuration for {sum(1 for a in scored if a in cfg_fp)} analyzers listed there; checkpoint SHA-256 of the configuration equal to the fingerprint (full SHA-256 or 16-character prefix) for {sum(1 for a in scored if re.fullmatch(r'[0-9a-f]{64}|[0-9a-f]{16}', (cache_fp.get(a) or '').split(':')[-1]))} analyzers"
             + ("; SHA-256 equal to --expected-cache-sha256" if expected_cache_sha256 and not any(p.startswith("cache: SHA-256") for p in P) else ""))

    # 2) CSV esportato: coppie uniche, insieme esatto, punteggi = cache (testo), flag e impronte = cache
    stage(f"reading the exported CSV ({exported_csv.name})")
    if miss := missing_columns(exported_csv, EXPORT_COLS):
        P.append(f"exported CSV: missing column(s) {miss}")
    exp_rows = list(csv.DictReader(exported_csv.open(encoding="utf-8")))
    if n_mal := malformed_rows(exp_rows):
        P.append(f"exported CSV: {n_mal} malformed rows (more or fewer cells than the header)")
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
    exp_map = {(r.get("analyzer") or "", r.get("file") or ""): r for r in exp_rows if finite01(r.get("attack_score")) is not None}
    def cached(k):
        v = cache.get("scores", {}).get(k[0], {}).get(k[1])
        return None if v is None else v.get("s")
    # label vuota o diversa da "0"/"1": problema a sé, non confusa con una label diversa dal manifest
    bad_label = sum(1 for r in exp_rows if label01(r.get("label")) is None)
    if bad_label:
        P.append(f"exported CSV: {bad_label} rows whose label is empty or not 0 or 1")
    # punteggio confrontato come testo: csv.writer di eval_dataset.py scrive repr del float della cache
    diff = sum(1 for k, r in exp_map.items() if k in expected_pairs and (cached(k) is None or finite(cached(k)) is None or r.get("attack_score") != repr(cached(k))
                                                                        or label01(r.get("label")) not in (None, samples.get(k[1], -1))))
    if diff:
        P.append(f"exported CSV: {diff} rows with score or label different from the cache/manifest (attack_score compared as text with the repr of the cache float)")
    def flags(aid, key):
        """Conteggio dei campioni con il flag `key` ("e" errore, "nf" nessun volto) nella cache per l'analizzatore."""
        return sum(1 for v in cache.get("scores", {}).get(aid, {}).values() if v.get(key))
    def cached_flag(k, key):
        return "1" if (cache.get("scores", {}).get(k[0], {}).get(k[1]) or {}).get(key) else "0"
    bad_flags = sum(1 for k, r in exp_map.items() if k in expected_pairs and (r.get("error") != cached_flag(k, "e") or r.get("no_face") != cached_flag(k, "nf")))
    if bad_flags:
        P.append(f"exported CSV: {bad_flags} rows whose error / no_face flags differ from the cache")
    bad_fp = sum(1 for r in exp_rows if (r.get("analyzer") or "", r.get("file") or "") in expected_pairs
                 and r.get("fingerprint") is not None and r.get("fingerprint") != cache_fp.get(r.get("analyzer")))
    if bad_fp:
        P.append(f"exported CSV: {bad_fp} rows whose fingerprint differs from the cache fingerprint of their analyzer")
    L.append(f"2. exported CSV ({exported_csv.name}): {len(exp_rows)} rows, {len(got_pairs)} unique (analyzer, sample) pairs (expected {len(expected_pairs)}, checked on every row before any score filter), {len(exp_rows) - bad_value} rows with a finite attack_score in [0, 1], {diff} rows differing from the cache/manifest (score as text), {len(dup)} duplicated pairs, {bad_flags} rows with wrong error / no_face flags, {bad_fp} rows with a fingerprint different from the cache")

    # 3) scores.csv (schema del gruppo): colonne dello schema, coppie uniche, valori per riga, round-trip, decisione, hash
    stage(f"reading scores.csv ({scores_csv.name})")
    if miss := missing_columns(scores_csv, SCHEMA_COLS + ["analyzer_id"]):
        P.append(f"scores.csv: missing column(s) {miss}")
    sc_rows = list(csv.DictReader(scores_csv.open(encoding="utf-8")))
    if n_mal := malformed_rows(sc_rows):
        P.append(f"scores.csv: {n_mal} malformed rows (more or fewer cells than the header)")
    sc_pairs = Counter((r.get("analyzer_id") or "", r.get("sample_id") or "") for r in sc_rows)      # tutte le righe
    dup = [k for k, c in sc_pairs.items() if c > 1]
    if dup:
        P.append(f"scores.csv: {len(dup)} duplicated (analyzer, sample) pairs")
    if set(sc_pairs) != expected_pairs:
        P.append(f"scores.csv: pairs differ: missing {len(expected_pairs - set(sc_pairs))}, extra {len(set(sc_pairs) - expected_pairs)}")
    thr = cfg_thr; thr_text = repr(float(cfg_thr)); bad_pred = bad_score = bad_cfg = bad_thr = 0
    malformed = Counter()                      # colonna -> righe con valore vuoto o non valido
    wrong, first_wrong = Counter(), {}         # colonna dello schema -> righe con un valore diverso dall'atteso
    cfg_sha = sha256(config_path) if config_path.exists() else None
    for r in sc_rows:
        k = (r.get("analyzer_id") or "", r.get("sample_id") or "")
        if k not in expected_pairs:
            continue
        m_row = meta.get(k[1], {})
        want = {"run_id": run_id, "dataset_id": dataset, "split": SPLIT, "modality": MODALITY,
                "subject_id": m_row.get("subject"), "session_id": m_row.get("session"),
                "video_id": m_row.get("video") or m_row.get("video_id") or "", "attack_type": types.get(k[1]),
                "checkpoint_sha256": exp_ck.get(k[0])}
        for col, w in want.items():
            got = r.get(col)
            if w is None or got is None:
                continue                         # colonna assente (già segnalata) o valore atteso non disponibile
            if col == "checkpoint_sha256" and not got.strip():
                malformed[col] += 1
            elif got != w:
                wrong[col] += 1; first_wrong.setdefault(col, (got, w))
        s = finite01(r.get("score_attack"))
        if s is None:
            bad_score += 1                                         # vuoto, non numerico, nan/inf o fuori [0, 1]
            continue
        lab, t_row, pred = label01(r.get("label")), finite(r.get("threshold")), label01(r.get("prediction"))
        if lab is None:
            malformed["label"] += 1
        if t_row is None:
            malformed["threshold"] += 1
        elif r.get("threshold") != thr_text:
            bad_thr += 1                                           # testo diverso da quello scritto dal generatore
        if cached(k) is None or r.get("score_attack") != repr(cached(k)) or (lab is not None and lab != samples.get(k[1], -1)):
            bad_score += 1
        if pred is None:
            malformed["prediction"] += 1
        elif pred != (1 if s > cfg_thr else 0):
            bad_pred += 1
        if cfg_sha and not (r.get("config_sha256") or "").strip():
            malformed["config_sha256"] += 1
        elif cfg_sha and r.get("config_sha256") != cfg_sha:
            bad_cfg += 1
    for col, n_bad in sorted(malformed.items()):
        P.append(f"scores.csv: {n_bad} rows whose {col} is empty or not valid" + (" (expected 0 or 1)" if col in ("label", "prediction") else ""))
    for col in [c for c in SCHEMA_COLS if wrong[c]]:
        got, w = first_wrong[col]
        P.append(f"scores.csv: {wrong[col]} rows whose {col} differs from the expected value (first: {got!r}, expected {w!r})")
    if bad_score:
        P.append(f"scores.csv: {bad_score} rows whose score_attack is empty, not a finite number in [0, 1], not the round-trip text of the cache float, or whose label differs")
    if bad_thr:
        P.append(f"scores.csv: {bad_thr} rows whose threshold differs from the configuration ({cfg_thr}; text {thr_text!r} expected)")
    if bad_pred:
        P.append(f"scores.csv: {bad_pred} predictions inconsistent with score > {cfg_thr} (configuration threshold)")
    if bad_cfg:
        P.append(f"scores.csv: {bad_cfg} rows whose config_sha256 differs from {config_path.name}")
    L.append(f"3. scores.csv (group schema): {len(sc_rows)} rows, {len(sc_pairs)} unique pairs (expected {len(expected_pairs)}, checked on every row), all {len(SCHEMA_COLS)} columns of score-schema.json required; run_id {run_id!r}, dataset_id, split {SPLIT!r}, modality {MODALITY!r}, subject_id / session_id / video_id / attack_type from the manifest and checkpoint_sha256 from the fingerprints: {sum(wrong.values())} values differing; {bad_score} rows with score_attack empty, non-finite, not the repr of the cache float or with a wrong label, {bad_pred} predictions inconsistent with score > {thr} (tie = bona fide), {bad_thr} rows with a threshold text different from {thr_text!r}, {bad_cfg} rows with config_sha256 different from {config_path.name}")

    # 4) tabella: lettura rigorosa, insiemi esatti di righe numeriche e senza punteggio, celle confrontate come testo
    stage(f"reading the table ({table_path.name})")
    lang, t_rows = parse_md_table(table_path.read_text(encoding="utf-8").splitlines(), TABLE_COLS, "table", P)
    numeric, tab_unscored, dups = {}, {}, []
    for _, cells in t_rows:
        aid, vals = cells[0], cells[1:]
        if aid in numeric or aid in tab_unscored:
            dups.append(aid)
        if aid in unscored or (aid not in scored and len(vals) == 2 and vals[0] == "0"):
            tab_unscored[aid] = vals
        else:
            numeric[aid] = vals
    if dups:
        P.append(f"table: duplicated rows {sorted(set(dups))}")
    if set(numeric) != scored:
        P.append(f"table: numeric rows differ from expected: missing {sorted(scored - set(numeric))}, extra {sorted(set(numeric) - scored)}")
    if set(tab_unscored) != unscored:
        P.append(f"table: rows without score differ from expected: missing {sorted(unscored - set(tab_unscored))}, extra {sorted(set(tab_unscored) - unscored)}")
    for aid in sorted(set(tab_unscored) & unscored):
        want_row = ["0", expected_note(aid, dataset, lang)]
        if tab_unscored[aid] != want_row:
            P.append(f"table/{aid}: row without score must be [{aid}, 0, note] with note {want_row[1]!r}; found {tab_unscored[aid]}")
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
        # celle intere confrontate come testo: "0300", "00" o cifre a larghezza piena non sono ciò che scrive eval_dataset.py
        if vals[:3] != [str(len(samples)), str(n_bf), str(n_at)]:
            P.append(f"table/{aid}: n / n_bona_fide / n_attack = {vals[0]} / {vals[1]} / {vals[2]}, expected {len(samples)} / {n_bf} / {n_at}")
        if vals[10:12] != [str(flags(aid, "e")), str(flags(aid, "nf"))]:
            P.append(f"table/{aid}: n_errors / n_noface = {vals[10]} / {vals[11]}, cache flags {flags(aid, 'e')} / {flags(aid, 'nf')}")
        if len(rows) != len(samples):
            continue
        stage(f"recomputing the metrics of {aid}")
        m = ev.compute_metrics([y for y, _ in rows], [s for _, s in rows])
        for key, col in METRIC_COLS.items():
            t, want_t = vals[col], cell(None if m.get(key) is None else float(m[key]))
            if t != want_t:
                # confronto sul TESTO della cella: "0.00004" o "0.12340" (stesso valore a 4 decimali, altro testo) non è la
                # cella che eval_dataset.py scrive e viene rifiutata; "—" solo se il valore non è calcolabile
                P.append(f"table/{aid}: {key} = {t}, recomputed {want_t}")
        n_metric_rows += 1
    L.append(f"4. table ({table_path.name}): strict Markdown table ({len(t_rows)} canonical rows), {len(numeric)} numeric rows (expected {len(scored)}), n = {len(samples)} and denominators {n_bf} / {n_at} on every row, n_errors / n_noface as in the cache, integer cells compared as text, metrics recomputed from the scores and compared with the 4-decimal cell text on {n_metric_rows} rows; {len(tab_unscored)} rows without score (expected {len(unscored)}), each [analyzer, 0, note] with the note of eval_dataset.py")

    # 5) tabella per tipo di attacco (facoltativa): tabella principale inclusa identica, righe = analizzatori × tipi, valori come testo
    if by_type_path is None:
        L.append("5. by-attack-type table: not present (not checked)")
        return P, L
    stage(f"reading the by-attack-type table ({by_type_path.name})")
    main_part, by_part, off = split_by_type_file(by_type_path.read_text(encoding="utf-8"), "by-type table", P)
    _, emb_rows = parse_md_table(main_part, TABLE_COLS, "by-type table (embedded main table)", P)
    emb, main = [c for _, c in emb_rows], [c for _, c in t_rows]
    if emb != main:
        first = next((i for i, (x, y) in enumerate(zip(emb, main)) if x != y), min(len(emb), len(main)))
        P.append(f"by-type table: the embedded main table differs from {table_path.name} ({len(emb)} rows vs {len(main)}; first different row: {first + 1})")
    _, b_rows = parse_md_table(by_part, BY_TYPE_COLS, "by-type table", P, offset=off) if by_part else ("en", [])
    stage(f"recomputing the metrics by attack type ({by_type_path.name})")
    exp_bt, exp_types = expected_by_type(cache, samples, types, scored, cfg_thr)
    got_bt, b_dups = {}, []
    for ln, cells in b_rows:
        if len(cells) != len(BY_TYPE_COLS) + 1:
            P.append(f"by-type table/{cells[0]}: {len(cells)} cells instead of {len(BY_TYPE_COLS) + 1} (line {ln})"); continue
        k = (cells[0], cells[1])
        if k in got_bt:
            b_dups.append(f"{k[0]}/{k[1]}")
        got_bt[k] = cells[1:]
    want_keys = {(a, t) for a in scored for t in exp_types}
    if b_dups:
        P.append(f"by-type table: duplicated rows {sorted(set(b_dups))}")
    if set(got_bt) != want_keys:
        P.append(f"by-type table: rows differ from expected: missing {sorted(f'{a}/{t}' for a, t in want_keys - set(got_bt))}, extra {sorted(f'{a}/{t}' for a, t in set(got_bt) - want_keys)}")
    n_bt = 0
    for k in sorted(set(got_bt) & want_keys):
        if k not in exp_bt:
            continue                                   # punteggi non validi: già segnalato nella cache
        for col, g, w in zip(BY_TYPE_COLS, got_bt[k], exp_bt[k]):
            if g != w:
                P.append(f"by-type table/{k[0]}/{k[1]}: {col} = {g}, recomputed {w}")
        n_bt += 1
    L.append(f"5. by-attack-type table ({by_type_path.name}): strict Markdown tables; embedded main table {'identical to' if emb == main else 'DIFFERENT from'} {table_path.name}; {len(got_bt)} rows (expected {len(want_keys)} = {len(scored)} analyzers x {len(exp_types)} types {exp_types}), n_attack, accepted, apcer, n_bona_fide, rejected_bona_fide, bpcer recomputed with metrics_by_attack_type and compared as text on {n_bt} rows")
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


def self_test(**paths):
    """Copie guaste degli artefatti: ciascuna deve produrre il suo messaggio strutturale atteso."""
    tmp = Path(tempfile.mkdtemp(prefix="c1-selftest-"))
    try:
        return _self_test(tmp, **paths)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _read_csv(p: Path):
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.reader(fh))


def _write_csv(p: Path, rows):
    """Stessa scrittura dei generatori (csv.writer, LF): su righe non toccate il file resta identico byte per byte."""
    with p.open("w", newline="", encoding="utf-8") as g:
        csv.writer(g, lineterminator="\n").writerows(rows)
    return p


def _self_test(tmp, dataset, cache_path, exported_csv, scores_csv, table_path, manifest, config_path, run_id=None, by_type_path=None,
               expected_cache_sha256=None, expected_config_sha256=None, expected_manifest_sha256=None):
    base = dict(dataset=dataset, cache_path=cache_path, exported_csv=exported_csv, scores_csv=scores_csv, table_path=table_path, manifest=manifest,
                config_path=config_path, run_id=run_id, by_type_path=by_type_path, expected_cache_sha256=expected_cache_sha256,
                expected_config_sha256=expected_config_sha256, expected_manifest_sha256=expected_manifest_sha256)
    faults, expect = {}, {}
    def add(name, override, msgs):
        faults[name] = override; expect[name] = msgs
    def wtext(fname, text):
        p = tmp / fname; p.write_text(text, encoding="utf-8"); return p

    # --- tabella principale
    table = table_path.read_text(encoding="utf-8").splitlines(keepends=True)
    hi = next((i for i, l in enumerate(table) if l.startswith("| analy")), None)
    if hi is None or hi + 1 >= len(table):
        raise ValueError(f"no header in {table_path.name}: faulty copies not buildable")
    header, sep_line = table[hi], table[hi + 1]
    body = [l for l in table[hi + 2:] if l.startswith("| ")]
    def cells_of(l):
        return [c.strip() for c in l.strip().strip("|").split("|")]
    numeric_lines = [l for l in body if cells_of(l)[1].isdigit() and int(cells_of(l)[1]) > 0]
    note_lines = [l for l in body if len(cells_of(l)) == 3 and cells_of(l)[1] == "0"]
    if not numeric_lines or not note_lines:
        raise ValueError(f"{table_path.name} has no numeric row or no row without score: faulty copies not buildable")
    victim, uvictim = numeric_lines[0], note_lines[-1]
    vc, uc = cells_of(victim), cells_of(uvictim)
    victim_aid, uaid = vc[0], uc[0]
    n_, nb_, na_ = int(vc[1]), int(vc[2]), int(vc[3])
    def tab_with(fname, repl):
        """Copia della tabella con la riga vittima sostituita (repl: testo, anche più righe o vuoto)."""
        return wtext(fname, "".join(repl if l == victim else l for l in table))
    def row_with(**kv):
        c = list(vc)
        for col, v in kv.items():
            c[1 + TABLE_COLS.index(col)] = v
        return md_row(c) + "\n"
    add(f"table row removed ({victim_aid})", dict(table_path=tab_with("t_removed.md", "")), [f"table: numeric rows differ from expected: missing ['{victim_aid}']"])
    add("table row duplicated", dict(table_path=wtext("t_dup.md", "".join(table) + victim)), [f"table: duplicated rows ['{victim_aid}']"])
    add("table denominator altered", dict(table_path=tab_with("t_denom.md", row_with(n_bona_fide=str(nb_ - 1), n_attack=str(na_ + 1)))),
        [f"table/{victim_aid}: n / n_bona_fide / n_attack = {n_} / {nb_ - 1} / {na_ + 1}"])
    apcer_txt = vc[1 + TABLE_COLS.index("apcer")]
    if re.fullmatch(r"\d+\.\d{4}", apcer_txt) is None:
        raise ValueError(f"apcer cell of {victim_aid} is not a 4-decimal number ({apcer_txt!r}): faulty copy not buildable")
    # cella APCER con una cifra in più ("0.1600" -> "0.16004"): stesso valore a 4 decimali, testo diverso
    add("table cell with an extra digit", dict(table_path=tab_with("t_digit.md", row_with(apcer=apcer_txt + "4"))),
        [f"table/{victim_aid}: apcer = {apcer_txt}4, recomputed {apcer_txt}"])
    add("table row without the '| ' prefix", dict(table_path=tab_with("t_noprefix.md", "|" + "|".join(vc) + "|\n")),
        ["is not in the canonical form", f"missing ['{victim_aid}']"])
    add("table row with a leading space", dict(table_path=tab_with("t_lead.md", " " + victim)), ["is not in the canonical form", f"missing ['{victim_aid}']"])
    add("table row whose analyzer starts with 'analyzer'", dict(table_path=wtext("t_prefix.md", "".join(table) + md_row(["analyzer_ghost"] + vc[1:]) + "\n")),
        ["table: numeric rows differ from expected: missing [], extra ['analyzer_ghost']"])
    add("table blank line inside the table", dict(table_path=tab_with("t_blank.md", "\n" + victim)), ["inside the table is not a table row"])
    add("table with a second table appended", dict(table_path=wtext("t_second.md", "".join(table) + "\n" + header + sep_line + victim)),
        ["second table at line"])
    add("table without separator", dict(table_path=wtext("t_nosep.md", "".join(l for i, l in enumerate(table) if i != hi + 1))),
        ["table: missing separator line"])
    add("table header with apcer / bpcer swapped", dict(table_path=wtext("t_swap.md", "".join(header.replace("| apcer | bpcer |", "| bpcer | apcer |") if l == header else l for l in table))),
        ["table: header at line"])
    add("table header with an extra column", dict(table_path=wtext("t_hextra.md", "".join(header.replace("| n_noface |", "| n_noface | extra |") if l == header else l for l in table))),
        ["table: header at line"])
    add("table header with a missing column", dict(table_path=wtext("t_hmiss.md", "".join(header.replace(" n_noface |", "") if l == header else l for l in table))),
        ["table: header at line"])
    add("table row without score with invented metrics", dict(table_path=wtext("t_note_metrics.md", "".join(md_row(uc + ["0.5000"]) + "\n" if l == uvictim else l for l in table))),
        [f"table/{uaid}: row without score must be"])
    add("table row without score with an empty note", dict(table_path=wtext("t_note_empty.md", "".join(md_row([uaid, "0", ""]) + "\n" if l == uvictim else l for l in table))),
        [f"table/{uaid}: row without score must be"])
    add("table row without score removed", dict(table_path=wtext("t_note_removed.md", "".join(l for l in table if l != uvictim))),
        [f"table: rows without score differ from expected: missing ['{uaid}']"])
    add("table n written as 0" + vc[1], dict(table_path=tab_with("t_n0.md", row_with(n="0" + vc[1]))), [f"table/{victim_aid}: n / n_bona_fide / n_attack = 0{vc[1]} / "])
    add("table n_errors written as 00", dict(table_path=tab_with("t_err00.md", row_with(n_errors="00"))), [f"table/{victim_aid}: n_errors / n_noface = 00 / "])
    fw = vc[1].translate(str.maketrans("0123456789", "０１２３４５６７８９"))
    add("table n in full-width digits", dict(table_path=tab_with("t_fullwidth.md", row_with(n=fw))), [f"table/{victim_aid}: n / n_bona_fide / n_attack = {fw} / "])

    # --- CSV esportato
    ex = _read_csv(exported_csv)
    if len(ex) < 2:
        raise ValueError("the exported CSV has no data row: faulty copies not buildable")
    eh = ex[0]
    def ex_row(i=1, **kv):
        r = list(ex[i])
        for col, v in kv.items():
            r[eh.index(col)] = v
        return r
    ghost = ex_row(analyzer="ghost_analyzer")
    add("exported CSV extra analyzer", dict(exported_csv=_write_csv(tmp / "e_extra.csv", ex + [ghost])), ["exported CSV: pairs differ: missing 0, extra 1"])
    add("exported CSV extra analyzer with empty score", dict(exported_csv=_write_csv(tmp / "e_extra_empty.csv", ex + [ex_row(analyzer="ghost_analyzer", attack_score="")])),
        ["exported CSV: pairs differ: missing 0, extra 1", "exported CSV: 1 rows whose attack_score is empty"])
    add("exported CSV duplicated pair with empty score", dict(exported_csv=_write_csv(tmp / "e_dup_empty.csv", ex + [ex_row(attack_score="")])),
        ["exported CSV: 1 duplicated (analyzer, sample) pairs", "exported CSV: 1 rows whose attack_score is empty"])
    add("exported CSV duplicated pair with nan score", dict(exported_csv=_write_csv(tmp / "e_dup_nan.csv", ex + [ex_row(attack_score="nan")])),
        ["exported CSV: 1 duplicated (analyzer, sample) pairs", "exported CSV: 1 rows whose attack_score is empty"])
    add("exported CSV inf score", dict(exported_csv=_write_csv(tmp / "e_inf.csv", [eh, ex_row(attack_score="inf")] + ex[2:])),
        ["exported CSV: 1 rows whose attack_score is empty"])
    add("exported CSV empty label", dict(exported_csv=_write_csv(tmp / "e_label.csv", [eh, ex_row(label="")] + ex[2:])),
        ["exported CSV: 1 rows whose label is empty or not 0 or 1"])
    keep = [i for i, c in enumerate(eh) if c not in ("error", "no_face")]
    add("exported CSV without error / no_face columns", dict(exported_csv=_write_csv(tmp / "e_noflags.csv", [[r[i] for i in keep] for r in ex])),
        ["exported CSV: missing column(s) ['error', 'no_face']"])
    add("exported CSV fingerprint altered", dict(exported_csv=_write_csv(tmp / "e_fp.csv", [eh, ex_row(fingerprint="ghost-fingerprint")] + ex[2:])),
        ["exported CSV: 1 rows whose fingerprint differs"])
    keep = [i for i, c in enumerate(eh) if c != "fingerprint"]
    add("exported CSV without fingerprint column", dict(exported_csv=_write_csv(tmp / "e_nofp.csv", [[r[i] for i in keep] for r in ex])),
        ["exported CSV: missing column(s) ['fingerprint']"])
    add("exported CSV score with a trailing zero (same float, other text)", dict(exported_csv=_write_csv(tmp / "e_text.csv", [eh, ex_row(attack_score=ex[1][eh.index("attack_score")] + "0")] + ex[2:])),
        ["exported CSV: 1 rows with score or label different"])

    # --- scores.csv
    sc = _read_csv(scores_csv)
    if len(sc) < 2:
        raise ValueError("scores.csv has no data row: faulty copies not buildable")
    sh = sc[0]
    for col in SCHEMA_COLS + ["analyzer_id"]:
        if col not in sh:
            raise ValueError(f"scores.csv has no column {col}: faulty copy not buildable")
    i_bf = next(i for i in range(1, len(sc)) if sc[i][sh.index("label")] == "0")
    i_at = next(i for i in range(1, len(sc)) if sc[i][sh.index("label")] == "1")
    def sc_with(i=1, **kv):
        rows = [list(r) for r in sc]
        for col, v in kv.items():
            rows[i][sh.index(col)] = v
        return rows
    def sc_row(i=1, **kv):
        return sc_with(i, **kv)[i]
    add("scores.csv sample removed", dict(scores_csv=_write_csv(tmp / "s_removed.csv", sc[:1] + sc[2:])), ["scores.csv: pairs differ: missing 1, extra 0"])
    add("scores.csv row duplicated", dict(scores_csv=_write_csv(tmp / "s_dup.csv", sc + [sc[1]])), ["scores.csv: 1 duplicated (analyzer, sample) pairs"])
    for col, msg in (("score_attack", "scores.csv: 1 rows whose score_attack is empty"),
                     ("label", "scores.csv: 1 rows whose label is empty or not valid"),
                     ("threshold", "scores.csv: 1 rows whose threshold is empty or not valid"),
                     ("prediction", "scores.csv: 1 rows whose prediction is empty or not valid"),
                     ("config_sha256", "scores.csv: 1 rows whose config_sha256 is empty or not valid"),
                     ("checkpoint_sha256", "scores.csv: 1 rows whose checkpoint_sha256 is empty or not valid")):
        add(f"scores.csv empty {col}", dict(scores_csv=_write_csv(tmp / f"s_empty_{col}.csv", sc_with(**{col: ""}))), [msg])
    add("scores.csv row with an extra cell", dict(scores_csv=_write_csv(tmp / "s_long.csv", [sh, sc[1] + ["extra"]] + sc[2:])), ["scores.csv: 1 malformed rows"])
    add("scores.csv duplicated pair with nan score", dict(scores_csv=_write_csv(tmp / "s_dup_nan.csv", sc + [sc_row(score_attack="nan")])),
        ["scores.csv: 1 duplicated (analyzer, sample) pairs", "scores.csv: 1 rows whose score_attack is empty"])
    add("scores.csv inf score", dict(scores_csv=_write_csv(tmp / "s_inf.csv", sc_with(score_attack="inf"))), ["scores.csv: 1 rows whose score_attack is empty"])
    add("scores.csv threshold written as 0.50", dict(scores_csv=_write_csv(tmp / "s_thr.csv", sc_with(threshold=sc[1][sh.index("threshold")] + "0"))),
        ["scores.csv: 1 rows whose threshold differs from the configuration"])
    for col in ("checkpoint_sha256", "video_id", "subject_id"):
        keep = [i for i, c in enumerate(sh) if c != col]
        add(f"scores.csv without {col} column", dict(scores_csv=_write_csv(tmp / f"s_no_{col}.csv", [[r[i] for i in keep] for r in sc])),
            [f"scores.csv: missing column(s) ['{col}']"])
    for col, value, i in (("checkpoint_sha256", "0" * 64, 1), ("run_id", "20260101-C1-seed42-0000000", 1), ("dataset_id", "casia_fasd", 1),
                          ("split", "dev", 1), ("modality", "NIR", 1), ("subject_id", "9999", 1), ("session_id", "99", 1),
                          ("video_id", "v1", 1), ("attack_type", "print", i_bf), ("attack_type", "", i_at)):
        add(f"scores.csv {col} altered to {value!r} (row {i})", dict(scores_csv=_write_csv(tmp / f"s_{col}_{i}.csv", sc_with(i, **{col: value}))),
            [f"scores.csv: 1 rows whose {col} differs from the expected value"])

    # --- manifest
    mf = _read_csv(manifest); mh = mf[0]
    def mf_with(**kv):
        rows = [list(r) for r in mf]
        for col, v in kv.items():
            rows[1][mh.index(col)] = v
        return rows
    if "bytes" in mh:
        add("manifest altered (bytes of one row)", dict(manifest=_write_csv(tmp / "m_bytes.csv", mf_with(bytes=mf[1][mh.index("bytes")] + "0"))),
            ["manifest: SHA-256"])
    add("manifest row removed", dict(manifest=_write_csv(tmp / "m_removed.csv", mf[:1] + mf[2:])),
        ["manifest: SHA-256", f"exported CSV: pairs differ: missing 0, extra {len(set(r[eh.index('analyzer')] for r in ex[1:]))}"])
    add("manifest subject altered", dict(manifest=_write_csv(tmp / "m_subject.csv", mf_with(subject="9999"))),
        ["manifest: SHA-256", "rows whose subject_id differs from the expected value"])
    if dataset == "nuaa":
        add("manifest subject different from the file-name prefix", dict(manifest=_write_csv(tmp / "m_subject_prefix.csv", mf_with(subject="9999"))),
            ["manifest: 1 rows whose subject differs from the 4-digit prefix of the file name"])
    add("manifest SHA-256 different from --expected-manifest-sha256", dict(expected_manifest_sha256="0" * 64),
        ["manifest: SHA-256", "differs from --expected-manifest-sha256"])

    # --- configurazione
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    scored_ids = {a.id for a in registry.all() if dataset not in a.excluded_datasets() and a.reliability() not in ("untrained", "descriptive")}
    fp_aid = next((an["id"] for an in cfg.get("analyzers", []) if an.get("id") in scored_ids and an.get("fingerprint")), None)
    if fp_aid:
        c2 = json.loads(json.dumps(cfg))
        next(an for an in c2["analyzers"] if an["id"] == fp_aid)["fingerprint"] = "ghost-fingerprint"
        add("config analyzer fingerprint altered", dict(config_path=wtext("config_fp.json", json.dumps(c2, indent=1))),
            [f"config: analyzers[].fingerprint differs from the cache for ['{fp_aid}']"])
    c3 = json.loads(json.dumps(cfg)); c3.get("dataset", {}).pop("manifest_sha256", None)
    add("config without dataset.manifest_sha256", dict(config_path=wtext("config_nomsha.json", json.dumps(c3, indent=1))), ["config: dataset.manifest_sha256 missing"])
    ck_aid = next((an["id"] for an in cfg.get("analyzers", []) if an.get("id") in scored_ids
                   and re.fullmatch(r"[0-9a-f]{64}", str(an.get("fingerprint", "")).split(":")[-1])), None)
    if ck_aid:
        c4 = json.loads(json.dumps(cfg)); fp64 = next(an["fingerprint"] for an in c4["analyzers"] if an["id"] == ck_aid).split(":")[-1]
        for c in c4.get("checkpoints", []):
            if c.get("sha256") == fp64:
                c["sha256"] = "f" * 64
        add("config checkpoint SHA-256 altered", dict(config_path=wtext("config_ck.json", json.dumps(c4, indent=1))),
            [f"config: checkpoints[].sha256 does not match the fingerprint for ['{ck_aid}']"])
    rm_aid = next((an["id"] for an in cfg.get("analyzers", []) if an.get("id") in scored_ids), None)
    if rm_aid:
        c5 = json.loads(json.dumps(cfg)); c5["analyzers"] = [an for an in c5["analyzers"] if an.get("id") != rm_aid]
        add("config analyzer removed from analyzers[]", dict(config_path=wtext("config_rm.json", json.dumps(c5, indent=1))),
            [f"config: analyzers differ from the table rows (analyzers with a score and rows without score): missing ['{rm_aid}']"])
    add("config SHA-256 different from --expected-config-sha256", dict(expected_config_sha256="0" * 64), ["config: SHA-256", "differs from --expected-config-sha256"])

    # --- cache
    cache_text = cache_path.read_text(encoding="utf-8")
    cache = json.loads(cache_text)
    k_aid = sorted(a for a in cache.get("scores", {}) if a in scored_ids)[0]
    n_k = len(cache["scores"][k_aid])
    k1 = json.loads(cache_text); k1["fingerprints"][k_aid] = "ghost-fingerprint"
    add("cache fingerprint altered", dict(cache_path=wtext("cache_fp.json", json.dumps(k1))), [f"exported CSV: {n_k} rows whose fingerprint differs"])
    k2 = json.loads(cache_text); k2["dataset"] = "casia_fasd"
    add("cache dataset altered", dict(cache_path=wtext("cache_ds.json", json.dumps(k2))), ["cache: dataset is 'casia_fasd'"])
    k3 = json.loads(cache_text); k3["scores"][k_aid].pop(sorted(k3["scores"][k_aid])[0])
    add("cache sample removed", dict(cache_path=wtext("cache_removed.json", json.dumps(k3))), [f"cache/{k_aid}: samples differ: missing 1, extra 0"])
    dup_text = cache_text.replace('"scores": {', '"scores": {"' + k_aid + '": {}, ', 1)
    add("cache with a duplicated JSON key", dict(cache_path=wtext("cache_dupkey.json", dup_text)), ["cache: 1 duplicated JSON keys"])
    add("cache SHA-256 different from --expected-cache-sha256", dict(expected_cache_sha256="0" * 64), ["cache: SHA-256"])

    # --- tabella per tipo di attacco: quella data o, se manca, una sintetizzata dai valori attesi e verificata come valida
    results = []
    bt_path = by_type_path
    if bt_path is None:
        samples, _, types, scored, _, _, _ = expected_sets(dataset, manifest)
        thr = finite(cfg.get("conventions", {}).get("threshold", 0.5)) or 0.5
        exp_bt, exp_types = expected_by_type(cache, samples, types, scored, thr)
        order = [cells_of(l)[0] for l in numeric_lines]
        bt_lines = ["## " + ed.TXT["en"]["by_type_title"], "", md_row(["analyzer"] + BY_TYPE_COLS), "|" + "---|" * (len(BY_TYPE_COLS) + 1)]
        bt_lines += [md_row([a] + exp_bt[(a, t)]) for a in order for t in exp_types if (a, t) in exp_bt]
        bt_path = wtext("by_type_valid.md", "".join(table) + "\n" + "\n".join(bt_lines) + "\n\n" + ed.TXT["en"]["by_type_note"].format(thr=thr) + "\n")
        problems, _ = safe_checks(**{**base, "by_type_path": bt_path})
        results.append(("synthesized valid by-type table accepted", not problems, [problems[0]] if problems else ["no problem"]))
    bt = bt_path.read_text(encoding="utf-8").splitlines(keepends=True)
    bh = next(i for i, l in enumerate(bt) if l.strip() == "## " + ed.TXT["en"]["by_type_title"] or l.strip() == "## " + ed.TXT["it"]["by_type_title"])
    b_body = [i for i in range(bh + 1, len(bt)) if bt[i].startswith("| ") and not bt[i].startswith("| analy")]
    b_head = next(i for i in range(bh + 1, len(bt)) if bt[i].startswith("| analy"))
    bv = b_body[0]; bvc = cells_of(bt[bv]); b_aid, b_t = bvc[0], bvc[1]
    def bt_with(fname, idx, repl):
        return wtext(fname, "".join(repl if i == idx else l for i, l in enumerate(bt)))
    def bt_row(**kv):
        c = list(bvc)
        for col, v in kv.items():
            c[1 + BY_TYPE_COLS.index(col)] = v
        return md_row(c) + "\n"
    vpos = next(i for i, l in enumerate(bt[:bh]) if l == victim)
    add("by-type table row removed", dict(by_type_path=bt_with("b_removed.md", bv, "")), [f"by-type table: rows differ from expected: missing ['{b_aid}/{b_t}']"])
    add("by-type table row duplicated", dict(by_type_path=wtext("b_dup.md", "".join(bt[:bv + 1]) + bt[bv] + "".join(bt[bv + 1:]))), [f"by-type table: duplicated rows ['{b_aid}/{b_t}']"])
    add("by-type table accepted altered", dict(by_type_path=bt_with("b_acc.md", bv, bt_row(accepted=str(int(bvc[3]) + 1)))), [f"by-type table/{b_aid}/{b_t}: accepted = {int(bvc[3]) + 1}, recomputed {bvc[3]}"])
    add("by-type table apcer with an extra digit", dict(by_type_path=bt_with("b_apcer.md", bv, bt_row(apcer=bvc[4] + "4"))), [f"by-type table/{b_aid}/{b_t}: apcer = {bvc[4]}4"])
    add("by-type table n_bona_fide written as 0" + bvc[5], dict(by_type_path=bt_with("b_nbf.md", bv, bt_row(n_bona_fide="0" + bvc[5]))), [f"by-type table/{b_aid}/{b_t}: n_bona_fide = 0{bvc[5]}"])
    add("by-type table analyzer swapped between two rows", dict(by_type_path=wtext("b_swap.md", "".join(
        md_row([cells_of(bt[b_body[1]])[0]] + bvc[1:]) + "\n" if i == bv else md_row([b_aid] + cells_of(bt[b_body[1]])[1:]) + "\n" if i == b_body[1] else l for i, l in enumerate(bt)))),
        [f"by-type table/{b_aid}/{b_t}: "])
    add("by-type table row not canonical", dict(by_type_path=bt_with("b_canon.md", bv, "|" + "|".join(bvc) + "|\n")), ["by-type table: row at line"])
    add("by-type table header with apcer / bpcer swapped", dict(by_type_path=bt_with("b_head.md", b_head, bt[b_head].replace("| apcer |", "| bpcer_x |").replace("| bpcer |", "| apcer |").replace("bpcer_x", "bpcer"))),
        ["by-type table: header at line"])
    add("by-type table heading removed", dict(by_type_path=bt_with("b_heading.md", bh, "\n")), ["by-type table: 0 headings"])
    add("by-type table embedded main table altered", dict(by_type_path=bt_with("b_embedded.md", vpos, row_with(apcer=apcer_txt + "4"))),
        ["by-type table: the embedded main table differs"])
    add("by-type table blank line inside the table", dict(by_type_path=bt_with("b_blank.md", bv, "\n" + bt[bv])), ["by-type table: line"])

    for name, override in faults.items():
        kw = dict(base)
        if name.startswith("by-type"):
            kw["by_type_path"] = bt_path
        kw.update(override)
        problems, _ = safe_checks(**kw)
        found = [next((p for p in problems if want in p), None) for want in expect[name]]
        if all(found):
            results.append((name, True, found[:1]))
        else:
            missing = [w for w, f in zip(expect[name], found) if f is None]
            results.append((name, False, [f"expected message not found: {missing[0]!r}; first problem: {problems[0] if problems else 'none'}"]))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Controllo di coerenza degli artefatti C1 / M1 (cache, CSV esportato, scores.csv, tabelle)")
    ap.add_argument("--dataset", default="nuaa", help="identificativo del dataset (default: nuaa)")
    ap.add_argument("--run-dir", default=None, help="cartella del run con scores.csv; se omessa, la cartella *-C1-* o *-M1-* più recente in --c1")
    ap.add_argument("--c1", default=str(RESULTS / "c1"), help="cartella con CSV esportato, tabella, manifest e configurazione (default: LIVENESSLAB_RESULTS_DIR/c1)")
    ap.add_argument("--by-type-table", default=None, help="tabella per tipo di attacco da controllare (default: <dataset>_table_by_attack_type.md in --c1, se presente)")
    sha_note = "; 64 caratteri esadecimali (spazi esterni tolti, maiuscole accettate), un valore vuoto o malformato è un errore"
    ap.add_argument("--expected-cache-sha256", default=None, help="SHA-256 atteso della cache dei punteggi; se dato, deve coincidere" + sha_note)
    ap.add_argument("--expected-config-sha256", default=None, help="SHA-256 atteso della configurazione (<dataset>_config.json); se dato, deve coincidere" + sha_note)
    ap.add_argument("--expected-manifest-sha256", default=None, help="SHA-256 atteso del manifest (<dataset>_manifest.csv); se dato, deve coincidere" + sha_note)
    ap.add_argument("--self-test", action="store_true", help="verifica che il controllo rifiuti copie guaste degli artefatti")
    a = ap.parse_args()
    # valori attesi: normalizzati prima di qualsiasi controllo; uno malformato ferma lo script senza scrivere il rapporto
    for what in ("cache", "config", "manifest"):
        attr = f"expected_{what}_sha256"
        value = getattr(a, attr)
        if value is None:
            continue
        norm = normalize_sha256(value)
        if norm is None:
            print(f"ERROR: --expected-{what}-sha256 {value!r} is not a SHA-256 (64 hexadecimal characters after removing surrounding "
                  "whitespace; upper case is accepted and lowered): check not run", file=sys.stderr)
            return 1
        setattr(a, attr, norm)
    c1 = Path(a.c1)
    if a.run_dir:
        run_dir = Path(a.run_dir)
    else:
        candidates = [p for p in list(c1.glob("*-C1-*")) + list(c1.glob("*-M1-*")) if p.is_dir()]
        if not candidates:
            raise SystemExit(f"--run-dir not given and no run folder (*-C1-* or *-M1-*) found in {c1}: pass --run-dir <restricted run folder>")
        run_dir = max(candidates, key=lambda p: p.stat().st_mtime)
    # manifest e configurazione: nella cartella dei risultati (--c1) se rigenerati, altrimenti nella cartella del run consegnata
    manifest = c1 / f"{a.dataset}_manifest.csv" if (c1 / f"{a.dataset}_manifest.csv").exists() else run_dir / f"{a.dataset}_manifest.csv"
    config = c1 / f"{a.dataset}_config.json" if (c1 / f"{a.dataset}_config.json").exists() else run_dir / f"{a.dataset}_config.json"
    by_type = Path(a.by_type_table) if a.by_type_table else (c1 / f"{a.dataset}_table_by_attack_type.md" if (c1 / f"{a.dataset}_table_by_attack_type.md").exists() else None)
    paths = dict(dataset=a.dataset, cache_path=RESULTS / "eval" / f"{a.dataset}.json", exported_csv=c1 / f"{a.dataset}_scores.csv",
                 scores_csv=run_dir / "scores.csv", table_path=c1 / f"{a.dataset}_table.md", manifest=manifest, config_path=config,
                 run_id=run_dir.name, by_type_path=by_type, expected_cache_sha256=a.expected_cache_sha256,
                 expected_config_sha256=a.expected_config_sha256, expected_manifest_sha256=a.expected_manifest_sha256)
    missing = [str(paths[k]) for k in FILE_KEYS if paths[k] is not None and not paths[k].exists()]
    kind = "M1" if "-M1-" in run_dir.name else "C1"
    head = [f"Consistency check of the {kind} artefacts, run {run_dir.name}, {time.strftime('%Y-%m-%d %H:%M')}"]
    if missing:
        out = head + ["  ! missing file(s): " + ", ".join(missing), "result: 1 problem(s) (check not performed)"]
        if run_dir.is_dir():
            (run_dir / "consistency_check.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
        print("\n".join(out))
        return 1
    problems, lines = safe_checks(**paths)
    out = head + [f"score cache: results/eval/{a.dataset}.json sha256 {sha256(paths['cache_path'])}",
                  f"scores.csv sha256 {sha256(paths['scores_csv'])}", ""] + lines
    for p in problems:
        out.append("  ! " + p)
    out.append(f"result: {'OK, no inconsistency' if not problems else str(len(problems)) + ' problem(s)'}")
    if a.self_test and problems:
        out.append("")
        out.append("self-test skipped: the main check already reports problems (the self-test is meaningful only on a consistent package)")
    elif a.self_test:
        out.append("")
        out.append("self-test (each faulty copy must be rejected with its expected message):")
        try:
            st = self_test(**paths)
        except Exception as exc:  # noqa: BLE001
            st = [(f"faulty copies could not be built ({type(exc).__name__}: {exc})", False, [])]
        for name, rejected, first in st:
            label = "accepted" if name.startswith("synthesized") and rejected else "rejected" if rejected else "NOT DETECTED"
            out.append(f"  {label}: {name}" + (f" -> {first[0]}" if first else ""))
        if not all(r for _, r, _ in st):
            problems.append("self-test: a faulty copy was not detected")
        opt_ok, opt_line = option_self_test()
        out.append("  " + opt_line)
        if not opt_ok:
            problems.append("self-test: an --expected-*-sha256 value was not handled as expected")
        n_copies = sum(1 for n, _, _ in st if not n.startswith("synthesized"))
        ok = all(r for _, r, _ in st) and opt_ok
        out.append(f"self-test result: {'all faults detected' if ok else 'FAILED'} ({n_copies} faulty copies)")
    (run_dir / "consistency_check.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
