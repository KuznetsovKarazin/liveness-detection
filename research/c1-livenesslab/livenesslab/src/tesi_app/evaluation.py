"""
Valutazione su dataset: esegue gli analizzatori su cartelle di immagini etichettate e calcola le metriche
biometriche ISO/IEC 30107-3 richieste dal docente (APCER, BPCER, ACER, EER, BPCER@APCER) più AUC e accuracy.

Convenzioni (le stesse del gruppo di ricerca e del repo del docente):
  - etichetta y = 0 bona fide, y = 1 attacco;
  - punteggio s = probabilità di ATTACCO (gli analizzatori dell'app restituiscono `score_real`, qui si usa 1 - score_real);
  - decisione a soglia fissa 0,5: attacco se s > 0,5. Il pareggio s = 0,5 conta come bona fide, in modo identico alla
    regola `p_real >= 0.5 -> reale` usata da ogni analizzatore per il verdetto sulla singola immagine. Per le metriche a
    soglia derivata (EER, BPCER@APCER) si usa invece s >= soglia, come la curva ROC di scikit-learn: la differenza conta
    solo per punteggi esattamente uguali alla soglia;
  - i punteggi delle CNN sono deterministici a parità di backend; tra GPU Metal e CPU differiscono nell'ordine di 1e-6
    (aritmetica diversa), quindi un'immagine con punteggio a cavallo di 0,5 può cambiare decisione tra le due macchine;
  - una metrica non calcolabile (classe assente, punteggi costanti) vale None, mai 0.

Le formule di `BiometricMetrics` derivano da `src/livedetection/src/evaluation_utils.py` (repo del docente, MIT), che non
è importabile direttamente perché dipende dal pacchetto `src` (collisione di nomi). Due differenze rispetto all'originale,
entrambe emerse in revisione e documentate nel diario:
  1. `calculate_bpcer_at_apcer` nell'originale conta i bona fide con punteggio <= soglia, cioè quelli ACCETTATI, e
     restituisce quindi 1 - BPCER (un classificatore perfetto ottiene 100 %). Qui si contano quelli con punteggio >= soglia.
  2. l'originale restituisce 0.0 quando una classe manca o i punteggi sono costanti; qui si restituisce None.
  3. l'EER nell'originale è cercato sui punti di `roc_curve` con `drop_intermediate=True` (default di scikit-learn), che
     scarta i punti collineari: il punto in cui APCER e BPCER coincidono può essere tra quelli scartati e l'EER risulta
     la media di due errori diversi. Qui la curva è completa (`drop_intermediate=False`).
HTER non viene calcolato: coinciderebbe con ACER finché la soglia non è scelta su un set di sviluppo separato (attività M1).

Struttura attesa dei dataset:  data/eval/<id>/real/*.jpg  e  data/eval/<id>/attack/*.jpg  (+ info.json opzionale).
I punteggi per immagine vengono salvati in results/eval/<id>.json e riusati; la cache di un analizzatore viene azzerata
quando cambia la sua "impronta" (hash dei pesi), così un modello riaddestrato non eredita i punteggi del precedente.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .core import RunCancelled, RunContext, registry
from .paths import DATA, RESULTS, ROOT

EVAL_DIR = DATA / "eval"
RESULTS_DIR = RESULTS / "eval"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
THRESHOLD = 0.5                      # soglia operativa fissa (probabilità di attacco)


# ----------------------------------------------------------------------------- metriche

class BiometricMetrics:
    """Metriche ISO/IEC 30107-3 su etichette (0 bona fide, 1 attacco) e punteggi di attacco."""

    @staticmethod
    def _is_single_class(y_true: np.ndarray) -> bool:
        return len(np.unique(y_true)) < 2

    @staticmethod
    def _is_constant_scores(y_scores: np.ndarray) -> bool:
        return y_scores.size == 0 or np.allclose(y_scores, y_scores[0])

    @staticmethod
    def calculate_apcer(y_true, y_pred) -> Optional[float]:
        """APCER = attacchi accettati come bona fide / attacchi. None se non ci sono attacchi."""
        attack_mask = (y_true == 1)
        if not np.any(attack_mask):
            return None
        return float(np.sum(y_pred[attack_mask] == 0) / attack_mask.sum())

    @staticmethod
    def calculate_bpcer(y_true, y_pred) -> Optional[float]:
        """BPCER = bona fide rifiutati come attacco / bona fide. None se non ci sono bona fide."""
        bonafide_mask = (y_true == 0)
        if not np.any(bonafide_mask):
            return None
        return float(np.sum(y_pred[bonafide_mask] == 1) / bonafide_mask.sum())

    @staticmethod
    def calculate_acer(apcer: Optional[float], bpcer: Optional[float]) -> Optional[float]:
        """ACER = media di APCER e BPCER, definita solo quando lo sono entrambe."""
        if apcer is None or bpcer is None:
            return None
        return (apcer + bpcer) / 2

    @staticmethod
    def calculate_eer(y_true, y_scores):
        """EER: punto della curva ROC in cui APCER e BPCER coincidono (senza interpolazione, come nel codice del docente,
        ma sulla curva completa: vedi differenza 3 nel docstring del modulo). Restituisce (eer, soglia); (None, None) se non calcolabile."""
        if y_scores.size == 0 or BiometricMetrics._is_single_class(y_true) or BiometricMetrics._is_constant_scores(y_scores):
            return None, None
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, drop_intermediate=False)   # fpr = bona fide rifiutati (BPCER), tpr = attacchi rilevati
        frr = 1 - tpr                                          # frr = attacchi accettati (APCER)
        idx = int(np.argmin(np.abs(fpr - frr)))
        thr = float(thresholds[idx])
        return float((fpr[idx] + frr[idx]) / 2), (None if not np.isfinite(thr) else thr)

    @staticmethod
    def calculate_bpcer_at_apcer(y_true, y_scores, target_apcer: float = 0.1) -> Optional[float]:
        """BPCER alla soglia che lascia passare circa `target_apcer` degli attacchi.
        La soglia è il quantile `target_apcer` dei punteggi di attacco: gli attacchi sotto la soglia sono accettati.
        I bona fide RIFIUTATI sono quelli con punteggio >= soglia (nell'originale del docente il confronto è <=, che
        conta gli accettati: è il difetto corretto qui)."""
        if y_scores.size == 0 or BiometricMetrics._is_constant_scores(y_scores):
            return None
        attack_scores = y_scores[y_true == 1]
        bonafide_scores = y_scores[y_true == 0]
        if attack_scores.size == 0 or bonafide_scores.size == 0:
            return None
        if attack_scores.size * target_apcer < 1:
            return None                              # con meno di 1/target attacchi il quantile non esiste (es. 5 attacchi per il 10 %)
        sorted_scores = np.sort(attack_scores)
        idx = int(np.clip(len(sorted_scores) * target_apcer, 0, len(sorted_scores) - 1))
        threshold = sorted_scores[idx]
        return float(np.sum(bonafide_scores >= threshold) / len(bonafide_scores))


def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = THRESHOLD) -> Dict[str, Any]:
    """Tutte le metriche: a soglia fissa (APCER, BPCER, ACER, accuracy), indipendenti dalla soglia (EER, AUC,
    BPCER@APCER10) e i punti della curva ROC per il grafico. Valori non calcolabili = None."""
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores, dtype=float)
    y_pred = (y_scores > threshold).astype(int)            # attacco solo se il punteggio supera la soglia (pareggio = bona fide)
    apcer = BiometricMetrics.calculate_apcer(y_true, y_pred)
    bpcer = BiometricMetrics.calculate_bpcer(y_true, y_pred)
    eer, eer_thr = BiometricMetrics.calculate_eer(y_true, y_scores)
    computable = not BiometricMetrics._is_single_class(y_true) and not BiometricMetrics._is_constant_scores(y_scores)
    out: Dict[str, Any] = {
        "n": int(len(y_true)), "n_real": int((y_true == 0).sum()), "n_attack": int((y_true == 1).sum()),
        "threshold": threshold,
        "apcer": apcer, "bpcer": bpcer, "acer": BiometricMetrics.calculate_acer(apcer, bpcer),
        "accuracy": float((y_pred == y_true).mean()) if len(y_true) else None,
        "eer": eer, "eer_threshold": eer_thr,
        "bpcer_at_apcer10": BiometricMetrics.calculate_bpcer_at_apcer(y_true, y_scores, 0.10),
        "auc": float(roc_auc_score(y_true, y_scores)) if computable else None,
    }
    if eer is not None and eer_thr is not None:
        y_pred_eer = (y_scores >= eer_thr).astype(int)
        out["apcer_at_eer"] = BiometricMetrics.calculate_apcer(y_true, y_pred_eer)
        out["bpcer_at_eer"] = BiometricMetrics.calculate_bpcer(y_true, y_pred_eer)
    if computable:
        # curva ROC compatta (max ~60 punti): x = BPCER (bona fide rifiutati), y = 1 - APCER (attacchi rilevati)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        step = max(1, len(fpr) // 60)
        out["roc"] = [[round(float(a), 4), round(float(b), 4)] for a, b in zip(fpr[::step], tpr[::step])] + [[1.0, 1.0]]
    return out


# ----------------------------------------------------------------------------- dataset

def imread(path: str) -> Optional[np.ndarray]:
    """Lettura immagine portabile: `cv2.imread` su Windows non apre percorsi con caratteri non ASCII."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def spread(items: List[Any], k: Optional[int]) -> List[Any]:
    """k elementi presi a passo costante dall'elenco ordinato: coprono tutti i soggetti invece dei soli primi."""
    if not k or k >= len(items):
        return list(items)
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def _samples_dataset() -> Dict[str, Any]:
    """Le immagini di prova dell'app come mini-dataset (solo etichette 'reale' e 'attacco*')."""
    sdir = ROOT / "samples"
    man = json.loads((sdir / "manifest.json").read_text(encoding="utf-8")) if (sdir / "manifest.json").exists() else {}
    items = []
    for p in sorted(sdir.glob("*.jpg")) + sorted(sdir.glob("*.png")):
        lab = man.get(p.name, {}).get("label", "")
        if lab == "reale":
            items.append((str(p), 0))
        elif lab.startswith("attacco"):
            items.append((str(p), 1))
    return {"id": "samples", "name": "Immagini di prova dell'app", "source": "Silent-Face + Wikimedia Commons + attacchi simulati",
            "note": "Mini-dataset dimostrativo: dipinto e volto IA sono esclusi (non sono né bona fide né attacchi fisici). Numeri NON significativi statisticamente.",
            "items": items}


def list_datasets() -> List[Dict[str, Any]]:
    """Le immagini di prova più ogni cartella data/eval/<id> con sottocartelle real/ e attack/."""
    out = [_samples_dataset()]
    if EVAL_DIR.exists():
        for d in sorted(EVAL_DIR.iterdir()):
            if not (d / "real").is_dir() or not (d / "attack").is_dir():
                continue
            info = json.loads((d / "info.json").read_text(encoding="utf-8")) if (d / "info.json").exists() else {}
            items = [(str(p), 0) for p in sorted((d / "real").iterdir()) if p.suffix.lower() in IMG_EXT] + \
                    [(str(p), 1) for p in sorted((d / "attack").iterdir()) if p.suffix.lower() in IMG_EXT]
            out.append({"id": d.name, "name": info.get("name", d.name), "source": info.get("source", ""), "note": info.get("note", ""),
                        "license": info.get("license", ""), "items": items})
    for ds in out:
        ds["n_real"] = sum(1 for _, y in ds["items"] if y == 0)
        ds["n_attack"] = sum(1 for _, y in ds["items"] if y == 1)
        ds["cached"] = (RESULTS_DIR / f"{ds['id']}.json").exists()
    return out


def describe_datasets() -> List[Dict[str, Any]]:
    """Come `list_datasets` ma senza l'elenco dei file (per l'API)."""
    return [{k: v for k, v in d.items() if k != "items"} for d in list_datasets()]


# ----------------------------------------------------------------------------- cache dei punteggi

_cache_lock = threading.RLock()     # lettura e scrittura della cache nello stesso processo: mai un rename mentre un altro thread legge


def key_of(path) -> str:
    """Chiave di un'immagine nella cache: cartella e nome ("real/x.jpg", "attack/x.jpg"), così due file omonimi nelle
    due classi non si sovrascrivono. Le cache scritte con il solo nome vengono migrate da `_migrate_keys`."""
    p = Path(path)
    return f"{p.parent.name}/{p.name}"


def _migrate_keys(data: Dict[str, Any], ds_id: str) -> None:
    """Porta le chiavi "nome.jpg" (formato precedente) al formato "cartella/nome.jpg" usando le cartelle del dataset;
    una chiave che non corrisponde più a nessun file viene scartata."""
    scores = data.get("scores", {})
    if not any("/" not in k for per in scores.values() for k in per):
        return
    lookup: Dict[str, str] = {}
    for d in list_datasets():
        if d["id"] == ds_id:
            for p, _ in d["items"]:
                lookup[Path(p).name] = key_of(p)
    for aid, per in list(scores.items()):
        new: Dict[str, Any] = {}
        for k, v in per.items():
            k2 = k if "/" in k else lookup.get(k)
            if k2:
                new[k2] = v
        scores[aid] = new


def _load_cache(ds_id: str) -> Dict[str, Any]:
    """Cache su disco: {"dataset", "scores": {aid: {"cartella/file": {"y","s","t"}}}, "fingerprints": {aid}, "notes": {aid}, "updated"}.
    Un file corrotto (scrittura interrotta) o con una struttura diversa viene ignorato invece di bloccare ogni valutazione successiva."""
    p = RESULTS_DIR / f"{ds_id}.json"
    data: Dict[str, Any] = {"dataset": ds_id, "scores": {}}
    with _cache_lock:
        if p.exists():
            try:
                loaded = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("scores"), dict):
                    data = loaded
                else:
                    print(f"[eval] cache {p.name} con struttura inattesa, la ignoro", flush=True)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"[eval] cache {p.name} illeggibile, la ignoro: {exc}", flush=True)
    data.setdefault("scores", {}); data.setdefault("fingerprints", {}); data.setdefault("notes", {})
    _migrate_keys(data, ds_id)
    return data


def _save_cache(ds_id: str, data: Dict[str, Any]) -> None:
    """Scrittura atomica (file temporaneo + rename): chi legge non trova mai un JSON a metà. Su Windows il rename
    fallisce se un altro processo tiene il file aperto: si ritenta qualche volta prima di arrendersi."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    final = RESULTS_DIR / f"{ds_id}.json"
    tmp = final.with_suffix(".json.tmp")
    with _cache_lock:
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        for attempt in range(5):
            try:
                os.replace(tmp, final)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    _summaries_cache.clear()


def summarize(ds_id: str, analyzer_ids: Optional[List[str]] = None, only_files: Optional[set] = None) -> Dict[str, Any]:
    """Metriche per analizzatore a partire dai punteggi in cache.
    `only_files` limita il calcolo alle immagini di una valutazione in corso (così `n` coincide con ciò che si sta valutando)."""
    cache = _load_cache(ds_id)
    out: Dict[str, Any] = {"dataset": ds_id, "analyzers": {}}
    for aid, per_img in cache.get("scores", {}).items():
        if analyzer_ids and aid not in analyzer_ids:
            continue
        try:
            a = registry.get(aid)
        except KeyError:
            continue                                   # punteggi di un analizzatore che non esiste più
        if ds_id in a.excluded_datasets():
            # vale anche per le cache scritte prima dell'introduzione dell'esclusione (es. pesi "pooled" sul proprio dataset)
            out["analyzers"][aid] = {"n": 0, "note": a.exclusion_note(ds_id)}
            continue
        if a.reliability() in ("untrained", "descriptive"):
            # pesi rimossi dopo l'ultima valutazione: i vecchi punteggi non vanno mostrati
            out["analyzers"][aid] = {"n": 0, "note": "nessun punteggio (modello non addestrato o descrittivo)"}
            continue
        if per_img and cache["fingerprints"].get(aid) != a.fingerprint():
            # punteggi calcolati con pesi, classificatore o prompt diversi da quelli attuali (o cache senza impronta):
            # non vanno mostrati come risultati validi; una nuova valutazione li ricalcola
            out["analyzers"][aid] = {"n": 0, "note": "punteggi di una versione precedente del modello: da rivalutare"}
            continue
        recs = [v for name, v in per_img.items() if only_files is None or name in only_files]
        rows = [(v["y"], v["s"]) for v in recs if v.get("s") is not None and np.isfinite(v["s"])]
        n_err = sum(1 for v in recs if v.get("e"))
        n_noface = sum(1 for v in recs if v.get("nf"))
        note = cache["notes"].get(aid)
        if len(rows) < 2:
            if note is None:
                if n_err:
                    note = f"{n_err} errori su {len(recs)} immagini"
                elif rows:
                    note = f"solo {len(rows)} immagine con punteggio: valutazione interrotta, da completare"
                else:
                    note = "nessun punteggio (modello non addestrato o descrittivo)"
            out["analyzers"][aid] = {"n": len(rows), "note": note}
            continue
        y = np.array([r[0] for r in rows]); s = np.array([r[1] for r in rows], dtype=float)
        m = compute_metrics(y, s)
        m["elapsed_ms_mean"] = float(np.mean([v.get("t", 0) for v in recs if v.get("s") is not None]))
        m["n_errors"] = n_err
        m["n_noface"] = n_noface
        m["fingerprint"] = cache["fingerprints"].get(aid)
        out["analyzers"][aid] = m
    out["updated"] = cache.get("updated")
    return out


_summaries_cache: Dict[str, Any] = {}


def all_summaries() -> Dict[str, Any]:
    """Riepiloghi di tutti i dataset che hanno una cache (per la tabella di confronto tra dataset). Il calcolo (metriche e
    curve ROC di tutti gli analizzatori) costa circa mezzo secondo di CPU: viene memorizzato e rifatto solo quando cambia
    un file di cache o l'impronta di un analizzatore, così richieste ripetute a /api/datasets non saturano il server."""
    key = repr(sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in RESULTS_DIR.glob("*.json"))) if RESULTS_DIR.exists() else "" 
    key += "|" + repr(sorted((a.id, a.fingerprint()) for a in registry.all()))
    with _cache_lock:
        if _summaries_cache.get("key") == key:
            return _summaries_cache["value"]
        value = {ds["id"]: summarize(ds["id"]) for ds in list_datasets() if ds["cached"]}
        _summaries_cache.clear(); _summaries_cache.update(key=key, value=value)
        return value


# ----------------------------------------------------------------------------- esecuzione

def run_evaluation(ds_id: str, analyzer_ids: List[str], emit: Callable[[Dict[str, Any]], None],
                   cancel: threading.Event, limit_per_class: Optional[int] = None, force: bool = False) -> None:
    """Esegue gli analizzatori sulle immagini del dataset (saltando quelle già in cache) ed emette
    eval_start / eval_progress / eval_result. Ogni immagine viene analizzata in modalità `quiet` (niente immagini
    intermedie codificate) per andare veloce."""
    ds = next((d for d in list_datasets() if d["id"] == ds_id), None)
    if ds is None:
        emit({"type": "eval_error", "message": f"Dataset {ds_id} non trovato"}); return
    items = ds["items"]
    if limit_per_class:
        # sottoinsieme distribuito su tutto l'elenco (e quindi su tutti i soggetti), non i primi N in ordine alfabetico
        real = spread([it for it in items if it[1] == 0], limit_per_class)
        att = spread([it for it in items if it[1] == 1], limit_per_class)
        items = real + att
    names = {key_of(p) for p, _ in items}
    cache = _load_cache(ds_id)
    scores = cache["scores"]
    known = set(registry.ids())
    for stale in [k for k in scores if k not in known]:
        scores.pop(stale); cache["fingerprints"].pop(stale, None); cache["notes"].pop(stale, None)   # analizzatori che non esistono più
    analyzers = []
    for a in (registry.get(aid) for aid in analyzer_ids):
        if a.reliability() in ("untrained", "descriptive"):
            # niente punteggio (rete non addestrata, Face Mesh descrittivo): resta in tabella con la nota, senza vecchi punteggi
            scores[a.id] = {}; cache["notes"].pop(a.id, None)
            continue
        if ds_id in a.excluded_datasets():
            # es. pesi "pooled": le immagini di questo dataset sono nel loro training, valutarle qui sarebbe barare
            scores[a.id] = {}; cache["notes"][a.id] = a.exclusion_note(ds_id)
            continue
        fp = a.fingerprint()
        if cache["fingerprints"].get(a.id) != fp:
            # pesi o algoritmo cambiati rispetto all'ultima valutazione (o cache scritta prima delle impronte, quindi di
            # provenienza non verificabile): i punteggi precedenti non valgono più e vengono ricalcolati
            scores[a.id] = {}
        cache["fingerprints"][a.id] = fp
        cache["notes"].pop(a.id, None)
        analyzers.append(a)
    def pending(a, p):
        """Da calcolare se manca o se l'ultima volta è finita in errore (gli errori transitori si ritentano)."""
        rec = scores.get(a.id, {}).get(key_of(p))
        return rec is None or rec.get("e")
    todo = [(p, y) for p, y in items if analyzers and (force or any(pending(a, p) for a in analyzers))]
    total = len(items)
    emit({"type": "eval_start", "dataset": ds_id, "name": ds["name"], "total": total, "todo": len(todo), "analyzers": analyzer_ids})
    t_start = time.time()
    for i, (path, y) in enumerate(todo):
        if cancel.is_set():
            break
        img = imread(path)
        name = key_of(path)
        if img is None:
            print(f"[eval] immagine illeggibile, saltata: {path}", flush=True)
            continue
        ctx = RunContext(img, lambda e: None, cancel, quiet=True)
        for a in analyzers:
            if not force and not pending(a, path):
                continue
            t0 = time.perf_counter()
            try:
                r = a.safe_run(ctx)
            except RunCancelled:
                break
            s = None if r.score_real is None or not np.isfinite(r.score_real) else float(1.0 - r.score_real)   # punteggio di ATTACCO
            rec: Dict[str, Any] = {"y": y, "s": s, "t": (time.perf_counter() - t0) * 1000}
            if r.reliability == "error":
                rec["e"] = True                        # errore dell'analizzatore su questa immagine (conteggiato in tabella)
            if r.details.get("no_face"):
                rec["nf"] = True                       # nessun volto rilevato: analisi sull'intera immagine
            scores.setdefault(a.id, {})[name] = rec
        if (i + 1) % 5 == 0 or i == len(todo) - 1:
            cache["updated"] = time.strftime("%Y-%m-%d %H:%M"); _save_cache(ds_id, cache)
            emit({"type": "eval_progress", "dataset": ds_id, "done": i + 1, "todo": len(todo), "total": total,
                  "elapsed_s": time.time() - t_start, "current": name, "partial": summarize(ds_id, analyzer_ids, names)["analyzers"]})
    cache["updated"] = time.strftime("%Y-%m-%d %H:%M"); _save_cache(ds_id, cache)
    emit({"type": "eval_result", "dataset": ds_id, "name": ds["name"], "cancelled": cancel.is_set(), "elapsed_s": time.time() - t_start,
          "summary": summarize(ds_id, analyzer_ids, names), "all": all_summaries(), "datasets": describe_datasets()})
