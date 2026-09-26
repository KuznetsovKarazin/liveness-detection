"""
Classificatori dei metodi classici (LBP, DoG, IQA, IDA) e delle sonde lineari sui foundation model (CLIP, DINOv2).

Ogni analizzatore di questo tipo espone nel proprio modulo:
  - `CROP`: margine e forma del ritaglio del volto (gli stessi usati in `run`);
  - `features(crop_bgr) -> np.ndarray`: funzione PURA che calcola il vettore di feature (la stessa usata in `run`);
  - `CLASSIFIER`: "svm" (RBF, per le feature fatte a mano) oppure "logreg" (regressione logistica, per gli embedding).
`scripts/train_classic.py` le usa per estrarre le feature dal training set ufficiale di un dataset, addestrare il
classificatore e salvare `models/weights/<id>_clf.joblib` con la scheda `<id>_clf.json` (dataset, numero di immagini,
metriche sul test ufficiale, SHA-256 del joblib). L'app carica il classificatore una volta per versione del file:
se il file cambia, cambia l'impronta dell'analizzatore e la cache dei punteggi su dataset viene azzerata.
Convenzione: classe 0 = bona fide, classe 1 = attacco (come nelle metriche).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..paths import WEIGHTS, file_sha256

# etichette dei dataset di addestramento (copia locale per non importare kuznetsov.py da qui)
DATASET_LABELS = {"nuaa": "NUAA", "casia_fasd": "CASIA-FASD", "celeba_spoof": "CelebA-Spoof", "synthaspoof": "SynthASpoof"}
CLASSIFIER_LABELS = {"svm": "SVM (kernel RBF)", "logreg": "regressione logistica"}
_cache: Dict[str, Any] = {}
_lock = threading.Lock()


def clf_paths(aid: str) -> Tuple[Path, Path]:
    """Percorsi del classificatore e della sua scheda per l'analizzatore `aid`."""
    return WEIGHTS / f"{aid}_clf.joblib", WEIGHTS / f"{aid}_clf.json"


def card_sha256(aid: str) -> Optional[str]:
    """Hash del classificatore dichiarato dalla scheda JSON (`clf_sha256`), se la scheda esiste: basta a riconoscere
    il classificatore e a ricalcolare le metriche dalla cache anche senza il file .joblib."""
    pj = clf_paths(aid)[1]
    if not pj.exists():
        return None
    try:
        return json.loads(pj.read_text(encoding="utf-8")).get("clf_sha256") or None
    except (OSError, ValueError):
        return None


def is_trained(aid: str) -> bool:
    """True se esiste un classificatore addestrato: il file .joblib oppure la sua scheda con l'hash."""
    return clf_paths(aid)[0].exists() or card_sha256(aid) is not None


def load_clf(aid: str) -> Tuple[Optional[Any], Dict[str, Any]]:
    """(classificatore, scheda) oppure (None, {}) se non addestrato. Ricaricato solo se il file cambia.
    Scheda presente ma file assente: errore esplicito, mai il ripiego silenzioso sull'euristica."""
    p, pj = clf_paths(aid)
    if not p.exists():
        if card_sha256(aid):
            raise RuntimeError(f"classificatore {aid} dichiarato dalla scheda JSON ma file .joblib assente in {WEIGHTS}: copiare i pesi (SHA-256 nella scheda)")
        return None, {}
    key = aid + ":" + file_sha256(p)
    with _lock:
        if key not in _cache:
            import joblib
            for old in [k for k in _cache if k.startswith(aid + ":")]:
                _cache.pop(old, None)
            meta = json.loads(pj.read_text(encoding="utf-8")) if pj.exists() else {}
            _cache[key] = (joblib.load(p), meta)
        return _cache[key]


def clf_fingerprint(aid: str, fallback: str, feature_version: str = "1") -> str:
    """Impronta per la cache dei punteggi: versione delle feature più hash del joblib se addestrato (altrimenti una
    versione dell'euristica). Se cambia il calcolo delle feature o il classificatore, la cache si azzera."""
    p, _ = clf_paths(aid)
    if p.exists():
        return f"{aid}-clf:v{feature_version}:" + file_sha256(p)[:16]
    card = card_sha256(aid)
    return f"{aid}-clf:v{feature_version}:" + (card[:16] if card else fallback)


def p_real_of(clf, vec: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> float:
    """Probabilità della classe 0 (bona fide) per un vettore di feature. I valori non finiti diventano 0 come nello
    script di addestramento (np.nan_to_num), così le due vie coincidono; una dimensione diversa da quella del
    training è un errore esplicito, non un risultato sbagliato in silenzio."""
    v = np.nan_to_num(np.asarray(vec, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if meta and meta.get("feature_dim") and int(meta["feature_dim"]) != int(v.size):
        raise ValueError(f"vettore di {v.size} valori, classificatore addestrato su {meta['feature_dim']}: riaddestrare con scripts/train_classic.py")
    proba = clf.predict_proba(v[None])[0]
    return float(proba[list(clf.classes_).index(0)])


def clf_note(meta: Dict[str, Any]) -> str:
    """Frase in italiano che descrive il classificatore caricato (per i passi e le spiegazioni)."""
    if not meta:
        return "Classificatore addestrato caricato."
    ds = DATASET_LABELS.get(meta.get("dataset", ""), meta.get("dataset", "?"))
    kind = CLASSIFIER_LABELS.get(meta.get("classifier", ""), meta.get("classifier", "classificatore"))
    t = meta.get("test", {}) or {}
    perf = ""
    if t.get("acer") is not None and t.get("auc") is not None:
        perf = f"; sul test ufficiale di {ds}: ACER {t['acer'] * 100:.1f} %, AUC {t['auc']:.3f}"
    return f"{kind[0].upper() + kind[1:]} addestrata sullo split ufficiale di training di {ds} ({meta.get('n_train', '?')} immagini{perf})."


def trained_on(meta: Dict[str, Any]) -> str:
    """Etichetta del dataset di addestramento ("" se sconosciuto)."""
    return DATASET_LABELS.get(meta.get("dataset", ""), meta.get("dataset", "")) if meta else ""
