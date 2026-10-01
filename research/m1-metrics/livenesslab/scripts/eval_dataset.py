"""
Valutazione da terminale (senza server): esegue gli analizzatori su un dataset di data/eval/<id>, aggiorna la cache
results/eval/<id>.json e stampa la tabella delle metriche. Serve per riprodurre una tabella in modo verificabile.

  python scripts/eval_dataset.py --dataset nuaa                      # usa la cache, calcola solo ciò che manca
  python scripts/eval_dataset.py --dataset nuaa --force              # nuova inferenza su tutte le immagini
  python scripts/eval_dataset.py --dataset nuaa --from-cache         # solo ricalcolo delle metriche dai punteggi salvati
  python scripts/eval_dataset.py --dataset nuaa --analyzers minifasnet livenessnet__nuaa --limit 50
  python scripts/eval_dataset.py --dataset nuaa --export results/eval/nuaa_scores.csv --table results/eval/nuaa_table.md
  python scripts/eval_dataset.py --dataset nuaa --from-cache --lang en --table ...   # intestazione e note in inglese (consegne al gruppo)
  python scripts/eval_dataset.py --dataset nuaa --from-cache --by-attack-type --manifest results/c1/nuaa_manifest.csv   # + APCER per tipo di attacco

L'export CSV contiene una riga per (immagine, analizzatore): file, etichetta (0 bona fide / 1 attacco), punteggio di
attacco, tempo, flag di errore / nessun volto, impronta del modello. L'intestazione del rapporto riporta commit del
repository e dei submodule, versioni delle librerie e backend di calcolo: sono i dati richiesti dal run report.

Con --by-attack-type il manifest deve coprire esattamente le immagini con punteggio di ogni analizzatore (righe in più o
in meno → uscita 1), i tipi di attacco devono appartenere al vocabolario (quello di score-schema.json del gruppo se ne
dichiara uno, altrimenti ATTACK_TYPES qui sotto; niente varianti come "Print" o "prnit") e una riga bona fide non può
avere un tipo. Senza l'opzione l'uscita non cambia.
"""
import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch  # noqa: E402,F401  (prima di TensorFlow: vedi run_app.py)
except Exception:  # noqa: BLE001
    pass

from tesi_app.core import registry  # noqa: E402
from tesi_app import analyzers  # noqa: E402,F401  (registra gli analizzatori)
from tesi_app import evaluation  # noqa: E402

COLS = ["n", "n_bona_fide", "n_attack", "apcer", "bpcer", "acer", "eer", "bpcer_at_apcer10", "auc", "accuracy", "n_errors", "n_noface"]
# testi della tabella nelle due lingue: italiano per l'app e i documenti, inglese per le consegne al gruppo di ricerca
TXT = {
    "it": {"analyzer": "analizzatore", "title": "Valutazione", "command": "Comando", "commit": "Commit", "dirty": " (modifiche non committate)",
           "devices": "dispositivi TF",
           "conventions": "Soglia 0,5 sul punteggio di attacco (pareggio = bona fide); etichette 0 = bona fide, 1 = attacco; unità = immagine; "
                          "APCER = attacchi accettati / n_attack, BPCER = bona fide rifiutati / n_bona_fide, ACER = media; "
                          "BPCER@APCER10 con soglia al quantile 0,1 degli attacchi; EER sulla curva ROC completa, senza interpolazione; "
                          "valori non calcolabili = —. Impronte dei modelli (SHA-256 dei pesi o id versione): vedi CSV.",
           "cache_mode": "Modalità cache: metriche ricalcolate dai punteggi salvati; impronte lette dalla cache, non verificate contro i file dei modelli.",
           "note_pooled": "non valutabile qui: con lo split 80/20 del codice di riferimento le immagini di questo dataset fanno parte del training del modello (il test del 20 % tenuto fuori è riportato nella scheda JSON dei pesi)",
           "note_noscore": "nessun punteggio (modello non addestrato o descrittivo)",
           "note_stale": "punteggi di una versione precedente del modello: da rivalutare",
           "by_type_title": "APCER per tipo di attacco",
           "by_type_note": "Soglia {thr} (pareggio = bona fide); APCER = attacchi accettati del tipo / n_attack del tipo; BPCER globale "
                           "= bona fide rifiutati / n_bona_fide (i bona fide non hanno tipo di attacco).",
           "by_type_source": "Tipi letti da `{manifest}`.",
           "by_type_nuaa": "Tipo `print` assegnato a tutti gli attacchi NUAA (il manifest non ha la colonna attack_type).",
           "by_type_unknown": "Il manifest non indica il tipo per {n} attacchi: tipo `unknown`."},
    "en": {"analyzer": "analyzer", "title": "Evaluation", "command": "Command", "commit": "Commit", "dirty": " (uncommitted changes)",
           "devices": "TF devices",
           "conventions": "Threshold 0.5 on the attack score (tie = bona fide); labels 0 = bona fide, 1 = attack; unit = image; "
                          "APCER = accepted attacks / n_attack, BPCER = rejected bona fide / n_bona_fide, ACER = their mean; "
                          "BPCER@APCER10 at the 0.1 quantile of the attack scores; EER on the full ROC curve, no interpolation; "
                          "non-computable values = —. Model fingerprints (SHA-256 of the weights or pinned revision): see the CSV.",
           "cache_mode": "Cache mode: metrics recomputed from the saved scores; fingerprints read from the cache, not verified against local model files.",
           "note_pooled": "not evaluated here: with the 80/20 pooled split of the reference code the images of this dataset are part of the model's training set (its held-out 20 % test is reported in the checkpoint JSON card)",
           "note_noscore": "no score (untrained or descriptive model)",
           "note_stale": "scores from a previous version of the model: re-evaluate",
           "by_type_title": "APCER by attack type",
           "by_type_note": "Threshold {thr} (tie = bona fide); APCER = accepted attacks of the type / n_attack of the type; global BPCER "
                           "= rejected bona fide / n_bona_fide (bona fide samples have no attack type).",
           "by_type_source": "Types read from `{manifest}`.",
           "by_type_nuaa": "Type `print` assigned to all NUAA attacks (the manifest has no attack_type column).",
           "by_type_unknown": "The manifest gives no type for {n} attacks: type `unknown`."},
}
BY_TYPE_COLS = ["attack_type", "n_attack", "accepted", "apcer", "n_bona_fide", "rejected_bona_fide", "bpcer"]
# schema dei punteggi del gruppo (sola lettura): se il campo attack_type dichiara un elenco di valori, quello è il vocabolario
SCORE_SCHEMA = Path(os.environ.get("LIVENESSLAB_SCORE_SCHEMA", "") or ROOT.parent / "liveness-detection" / "research" / "templates" / "score-schema.json")
# vocabolario di riserva: lo schema del gruppo (versione 1.0) dichiara attack_type come "string or null", senza elenco.
# Tipi dei dataset valutati (NUAA print; CASIA-FASD print, cut photo, replay; CelebA-Spoof print, replay, mask;
# SynthASpoof print, replay); `unknown` è il valore assegnato quando il manifest non dà il tipo.
ATTACK_TYPES = ("print", "cut_photo", "replay", "mask", "unknown")


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def attack_type_vocabulary():
    """(vocabolario, origine): l'elenco dichiarato da score-schema.json per attack_type (lista, oppure "a|b|c" come per
    split), se il file esiste e lo dichiara; altrimenti ATTACK_TYPES. Il file dello schema non viene mai modificato."""
    try:
        spec = json.loads(SCORE_SCHEMA.read_text(encoding="utf-8")).get("fields", {}).get("attack_type")
    except (OSError, ValueError, AttributeError):
        spec = None
    if isinstance(spec, list) and spec and all(isinstance(v, str) for v in spec):
        vocab = tuple(spec)
    elif isinstance(spec, str) and "|" in spec and " " not in spec.strip():
        vocab = tuple(v for v in spec.strip().split("|") if v)
    else:
        return ATTACK_TYPES, "built-in list (score-schema.json declares no list of values for attack_type)"
    return tuple(dict.fromkeys(vocab + ("unknown",))), "score-schema.json"


def translate_note(note: str, lang: str) -> str:
    """Le note delle righe senza punteggio sono prodotte in italiano dal modulo di valutazione: in inglese si sostituiscono
    per tipo (esclusione, senza punteggio, impronta cambiata); una nota sconosciuta resta com'è."""
    it, tr = TXT["it"], TXT[lang]
    for key in ("note_pooled", "note_noscore", "note_stale"):
        if note.startswith(it[key][:30]):
            return tr[key]
    return note


def git(*args, cwd=ROOT):
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "n/d"


def submodule_commit(path) -> str:
    """Commit di una cartella di codice esterno solo se è la radice di un proprio repository (.git file o cartella e
    `--show-toplevel` uguale alla cartella); altrimenti "n/d", mai il commit del repository che la contiene."""
    p = Path(path)
    if not (p / ".git").exists():
        return "n/d"
    top = git("rev-parse", "--show-toplevel", cwd=p)
    if top == "n/d" or Path(top).resolve() != p.resolve():
        return "n/d"
    return git("rev-parse", "HEAD", cwd=p)


def verify_checkpoints(ids) -> None:
    """Prima di una nuova inferenza: per ogni CNN e classificatore con scheda JSON, confronta l'hash SHA-256 del file dei
    pesi con quello dichiarato nella scheda. Un file mancante o diverso ferma l'esecuzione."""
    from tesi_app.analyzers import kuznetsov, classic
    from tesi_app.paths import WEIGHTS, file_sha256
    rows, bad = [], []
    for aid in ids:
        a = registry.get(aid)
        key = getattr(a, "weights_key", None)
        if key:
            p = kuznetsov.weights_file(key); declared = kuznetsov.card_sha256(key)
        else:
            p, pj = classic.clf_paths(aid); p = p if p.exists() else None; declared = classic.card_sha256(aid)
        if not declared:
            continue
        actual = file_sha256(p) if p is not None else None
        ok = actual == declared
        rows.append((aid, p.name if p is not None else "ASSENTE", declared[:16], "ok" if ok else "DIVERSO" if actual else "mancante"))
        if not ok:
            bad.append(aid)
    if rows:
        print("== checkpoint verificati (SHA-256 del file = scheda JSON):", flush=True)
        for r in rows:
            print(f"   {r[0]:34s} {r[1]:36s} {r[2]}…  {r[3]}", flush=True)
    if bad:
        sys.exit(f"pesi non verificati per: {', '.join(bad)} (cartella {WEIGHTS})")


def environment() -> dict:
    """Commit, submodule, librerie e backend: l'intestazione del run report."""
    from tesi_app.paths import LIVEDETECTION, SILENT_FACE
    commit = git("rev-parse", "HEAD")
    # fuori da un repository git (es. il pacchetto esportato) commit e stato non sono disponibili: "n/d", non "modificato"
    info = {"commit": commit, "dirty": bool(git("status", "--porcelain")) if commit != "n/d" else False,
            "livedetection": submodule_commit(LIVEDETECTION),
            "silent_face": submodule_commit(SILENT_FACE),
            "python": platform.python_version(), "platform": platform.platform()}
    for mod in ("tensorflow", "keras", "torch", "transformers", "numpy", "sklearn", "cv2", "mediapipe", "skimage", "scipy"):
        try:
            m = __import__(mod); info[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            info[mod] = "assente"
    try:
        import tensorflow as tf
        info["tf_devices"] = [d.device_type for d in tf.config.list_physical_devices()]   # "GPU" = Metal su Apple Silicon
    except Exception:  # noqa: BLE001
        info["tf_devices"] = "n/d"
    return info


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def table(summary: dict, names: dict, lang: str = "it") -> str:
    """Tabella Markdown delle metriche (righe con punteggio prima, poi le note), con i denominatori per riga."""
    rows = sorted(summary["analyzers"].items(), key=lambda kv: ("note" in kv[1], names.get(kv[0], (99, ""))))
    out = [f"| {TXT[lang]['analyzer']} | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 1)]
    for aid, m in rows:
        if "note" in m:
            out.append(f"| {aid} | {m['n']} | {translate_note(m['note'], lang)} |")
        else:
            vals = dict(m); vals["n_bona_fide"] = m.get("n_real")
            out.append(f"| {aid} | " + " | ".join(fmt(vals.get(c)) for c in COLS) + " |")
    return "\n".join(out)


def read_attack_types(manifest: Path, dataset: str):
    """{file: (label, tipo)} dal manifest (colonne `file`, `label`, opzionale `attack_type`), più il numero di attacchi
    senza tipo. `file` è la chiave della cache ("real/x.jpg", "attack/x.jpg"), come nel manifest di C1.
    Senza colonna: per NUAA tutti gli attacchi sono foto stampate (`print`, come in scores.csv di C1); altrove `unknown`."""
    with manifest.open(encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        if "file" not in cols or "label" not in cols:
            fail(f"manifest {manifest}: columns 'file' and 'label' are required")
        has_type = "attack_type" in cols
        vocab, _ = attack_type_vocabulary()
        out, n_missing = {}, 0
        for r in rd:
            if None in r or r.get("file") is None or r.get("label") is None:
                fail(f"manifest {manifest}: row with a wrong number of fields ({r.get('file')!r})")
            if r["file"] in out:
                fail(f"manifest {manifest}: duplicated file {r['file']}")
            if r["label"] not in ("0", "1"):
                fail(f"manifest {manifest}: label {r['label']!r} of {r['file']} is not 0 or 1")
            y = int(r["label"])
            t = (r.get("attack_type") or "").strip() if has_type else ""
            if y == 0 and t:
                fail(f"manifest {manifest}: bona fide row {r['file']} has attack type {t!r} (bona fide samples have no attack type)")
            if t and t not in vocab:
                fail(f"manifest {manifest}: attack type {t!r} of {r['file']} is not in the vocabulary {list(vocab)}")
            if y == 1 and not t:
                if not has_type and dataset == "nuaa":
                    t = "print"                                # NUAA: solo attacchi con foto stampata
                else:
                    t = "unknown"; n_missing += 1
            out[r["file"]] = (y, t if y == 1 else "")
    return out, has_type, n_missing


def by_type_table(summary: dict, names: dict, dataset: str, manifest: Path, lang: str) -> str:
    """Tabella Markdown analizzatore × tipo di attacco (APCER con denominatore) e BPCER globale, alla soglia fissa.
    Stesse righe della tabella principale (solo quelle con punteggio) e stessi punteggi di `summarize`."""
    T = TXT[lang]
    types, has_type, n_missing = read_attack_types(manifest, dataset)
    expected = sorted({t for y, t in types.values() if y == 1})
    cache = evaluation._load_cache(dataset)
    rows = sorted(summary["analyzers"].items(), key=lambda kv: ("note" in kv[1], names.get(kv[0], (99, ""))))
    out = [f"## {T['by_type_title']}", "", f"| {T['analyzer']} | " + " | ".join(BY_TYPE_COLS) + " |", "|" + "---|" * (len(BY_TYPE_COLS) + 1)]
    for aid, m in rows:
        if "note" in m:
            continue
        # stessi filtri di summarize: solo punteggi finiti
        per = cache.get("scores", {}).get(aid)
        if not isinstance(per, dict):
            fail(f"{aid}: no scores in the cache")
        recs = [(k, v) for k, v in per.items() if isinstance(v, dict) and isinstance(v.get("s"), (int, float))
                and not isinstance(v.get("s"), bool) and math.isfinite(v["s"])]
        # le immagini con punteggio devono essere esattamente quelle del manifest: righe in più o in meno fermano la tabella
        extra = sorted({k for k, _ in recs} - set(types))
        missing = sorted(set(types) - {k for k, _ in recs})
        if extra or missing:
            fail(f"{aid}: scored images and manifest rows differ: {len(extra)} scored samples not in the manifest "
                 f"(e.g. {extra[:2]}), {len(missing)} manifest rows without a score (e.g. {missing[:2]})")
        wrong = [k for k, v in recs if v.get("y") not in (0, 1) or isinstance(v.get("y"), bool) or int(v["y"]) != types[k][0]]
        if wrong:
            fail(f"{aid}: {len(wrong)} labels differ from the manifest ({wrong[0]} ...)")
        r = evaluation.metrics_by_attack_type([v["y"] for _, v in recs], [v["s"] for _, v in recs],
                                              [types[k][1] for k, _ in recs], m["threshold"], expected)
        if r["n_attack"] != m["n_attack"] or r["n_bona_fide"] != m["n_real"]:
            fail(f"{aid}: counts differ from the main table")
        for t, d in r["by_type"].items():
            out.append(f"| {aid} | " + " | ".join(fmt(v) for v in (t, d["n_attack"], d["accepted"], d["apcer"],
                                                                    r["n_bona_fide"], r["rejected_bona_fide"], r["bpcer"])) + " |")
    thr = str(evaluation.THRESHOLD) if lang == "en" else str(evaluation.THRESHOLD).replace(".", ",")
    # una sola nota sull'origine dei tipi: letti dal manifest oppure assegnati (NUAA senza colonna attack_type)
    source = T["by_type_nuaa"] if not has_type and dataset == "nuaa" else T["by_type_source"].format(manifest=manifest.name)
    notes = [T["by_type_note"].format(thr=thr) + " " + source]
    if n_missing:
        notes.append(T["by_type_unknown"].format(n=n_missing))
    return "\n".join(out) + "\n\n" + "\n\n".join(notes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="id del dataset (cartella di data/eval, oppure 'samples')")
    ap.add_argument("--analyzers", nargs="*", default=None, help="id degli analizzatori (default: tutti)")
    ap.add_argument("--limit", type=int, default=None, help="immagini per classe (sottoinsieme distribuito)")
    ap.add_argument("--force", action="store_true", help="ricalcola anche le immagini già in cache (nuova inferenza)")
    ap.add_argument("--from-cache", action="store_true", help="niente inferenza: solo metriche dai punteggi salvati")
    ap.add_argument("--export", help="CSV dei punteggi per immagine")
    ap.add_argument("--table", help="file Markdown in cui salvare tabella e intestazione")
    ap.add_argument("--lang", choices=("it", "en"), default="it", help="lingua di intestazione e note della tabella (en per le consegne al gruppo)")
    ap.add_argument("--by-attack-type", action="store_true", help="aggiunge la tabella dell'APCER per tipo di attacco (richiede --manifest)")
    ap.add_argument("--manifest", help="manifest CSV con colonne file, label e, opzionale, attack_type (file = chiave della cache, es. attack/x.jpg); solo con --by-attack-type")
    a = ap.parse_args()
    if a.by_attack_type and not a.manifest:
        ap.error("--by-attack-type requires --manifest")
    if a.manifest and not a.by_attack_type:
        ap.error("--manifest is used only with --by-attack-type")
    if a.manifest and not Path(a.manifest).is_file():
        ap.error(f"manifest not found: {a.manifest}")
    T = TXT[a.lang]

    ids = [i for i in (a.analyzers or registry.ids()) if i in registry.ids()]
    names = {x.id: (x.order, x.name) for x in registry.all()}
    env = environment()
    print("== ambiente:", json.dumps(env, ensure_ascii=False), flush=True)
    from tesi_app.paths import DATA, LIVEDETECTION, RESULTS, SILENT_FACE, WEIGHTS
    print("== percorsi:", json.dumps({"data": str(DATA), "results": str(RESULTS), "weights": str(WEIGHTS), "livedetection": str(LIVEDETECTION), "silent_face": str(SILENT_FACE)}), flush=True)
    cache0 = evaluation._load_cache(a.dataset)
    n_cached = sum(len(v) for v in cache0.get("scores", {}).values())
    ds_items = next((len(d["items"]) for d in evaluation.list_datasets() if d["id"] == a.dataset), 0)
    print(f"== dataset {a.dataset}: {ds_items} immagini in {DATA / 'eval' / a.dataset}, {n_cached} punteggi in cache ({RESULTS / 'eval' / (a.dataset + '.json')}), {len(ids)} analizzatori registrati", flush=True)
    if a.from_cache:
        # analizzatori con punteggi in cache ma non registrati (tipicamente una CNN senza scheda JSON in WEIGHTS): solo un
        # avviso, la tabella e il codice di uscita non cambiano
        orphans = sorted(k for k, v in cache0.get("scores", {}).items() if v and k not in registry.ids())
        for k in orphans:
            print(f"warning: '{k}' has cached scores but is not registered (missing checkpoint JSON card in {WEIGHTS}?): left out of the table", file=sys.stderr, flush=True)
    if a.from_cache and n_cached == 0:
        sys.exit(f"nessun punteggio in cache per '{a.dataset}': copiare la cache in {RESULTS / 'eval'} (o impostare LIVENESSLAB_RESULTS_DIR)")
    if not a.from_cache and ds_items == 0:
        sys.exit(f"nessuna immagine in {DATA / 'eval' / a.dataset} (o impostare LIVENESSLAB_DATA_DIR)")
    t0 = time.time()
    if a.from_cache:
        # ricalcolo autonomo: metriche dai punteggi salvati; le impronte dei modelli sono quelle registrate nella cache
        # (riportate nel CSV) e NON vengono ricalcolate, così non servono pesi, risorse di Silent-Face né modelli scaricati
        print("== modalità cache: impronte lette dalla cache, non verificate contro i file dei modelli", flush=True)
        summary = evaluation.summarize(a.dataset, ids, verify_fingerprints=False)
    else:
        verify_checkpoints(ids)
        def emit(ev):
            if ev["type"] == "eval_progress" and (ev["done"] % 25 == 0 or ev["done"] == ev["todo"]):
                print(f"  {ev['done']}/{ev['todo']} ({time.time() - t0:.0f} s)", flush=True)
            elif ev["type"] == "eval_error":
                sys.exit("errore: " + ev["message"])
            elif ev["type"] == "eval_start":
                print(f"== {ev['name']}: {ev['todo']} immagini da analizzare su {ev['total']}", flush=True)
        evaluation.run_evaluation(a.dataset, ids, emit, threading.Event(), a.limit, a.force)
        summary = evaluation.summarize(a.dataset, ids)
    md = table(summary, names, a.lang)
    if a.by_attack_type:
        md += "\n\n" + by_type_table(summary, names, a.dataset, Path(a.manifest), a.lang)
    print(md)
    head = (f"# {T['title']} `{a.dataset}` · {time.strftime('%Y-%m-%d %H:%M')}\n\n"
            f"{T['command']}: `{' '.join(sys.argv)}`  \n{T['commit']}: `{env['commit']}`{T['dirty'] if env['dirty'] else ''} · "
            f"livedetection `{env['livedetection'][:8]}` · Silent-Face `{env['silent_face'][:8]}`  \n"
            f"Python {env['python']} · TensorFlow {env['tensorflow']} · Keras {env['keras']} · torch {env['torch']} · transformers {env['transformers']} · "
            f"numpy {env['numpy']} · scikit-learn {env['sklearn']} · OpenCV {env['cv2']} · {T['devices']} {env['tf_devices']}  \n"
            f"{T['conventions']}\n" + (f"{T['cache_mode']}\n" if a.from_cache else "") + "\n")
    if a.table:
        Path(a.table).parent.mkdir(parents=True, exist_ok=True)
        Path(a.table).write_text(head + md + "\n", encoding="utf-8"); print("tabella salvata in", a.table)
    if a.export:
        cache = evaluation._load_cache(a.dataset)
        Path(a.export).parent.mkdir(parents=True, exist_ok=True)
        with open(a.export, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n"); w.writerow(["analyzer", "file", "label", "attack_score", "elapsed_ms", "error", "no_face", "fingerprint"])
            for aid, per_img in cache["scores"].items():
                if aid not in ids:
                    continue
                for name, r in sorted(per_img.items()):
                    w.writerow([aid, name, r["y"], "" if r["s"] is None else r["s"], round(r.get("t", 0), 1), int(bool(r.get("e"))), int(bool(r.get("nf"))), cache["fingerprints"].get(aid, "")])
        print("punteggi esportati in", a.export)


if __name__ == "__main__":
    main()
