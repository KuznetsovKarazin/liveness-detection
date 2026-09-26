"""
Analisi micro-texture con Local Binary Pattern (Määttä, Hadid, Pietikäinen 2011) estesa agli spazi colore
HSV e YCbCr (Boulkenafet, Komulainen, Hadid 2016). Classificatore SVM addestrabile (models/weights/lbp_clf.joblib).

Variante usata: LBP "uniform" di scikit-image (invariante alla rotazione, P+2 bin per istogramma), più compatta del
LBP^u2 a 59 bin dell'articolo originale. Vettore finale: 10·9 (LBP 8,1 su griglia 3×3) + 10 + 18 + 6·10 = 178 valori.
Classificatore: SVM addestrata con scripts/train_classic.py (models/weights/lbp_clf.joblib + .json).
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from ..core import Analyzer, NodeSpec, Result, RunContext, colorize, registry, tile
from .classic import clf_fingerprint, clf_note, is_trained, load_clf, p_real_of
from .common import COMMON_EDGES, COMMON_NODES, no_face, step_crop, step_face, step_input

CROP = dict(margin=0.0, square=True)     # stesso ritaglio in `run` e in `features` (scripts/train_classic.py)
CLASSIFIER = "svm"
FEATURE_VERSION = "1"           # da incrementare quando cambia il calcolo di `features()`


def lbp_hist(gray: np.ndarray, P: int, R: int, grid: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Istogrammi LBP uniformi (P vicini, raggio R) su una griglia grid×grid di celle, concatenati e normalizzati.
    Restituisce (vettore, immagine LBP)."""
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    n_bins = P + 2
    feats = []
    H, W = lbp.shape
    for i in range(grid):
        for j in range(grid):
            cell = lbp[i * H // grid:(i + 1) * H // grid, j * W // grid:(j + 1) * W // grid]
            h, _ = np.histogram(cell, bins=n_bins, range=(0, n_bins), density=True)
            feats.append(h)
    return np.concatenate(feats), lbp


def features(crop_bgr: np.ndarray) -> np.ndarray:
    """Vettore LBP (178 valori) di un ritaglio del volto: la STESSA sequenza di `run`, senza i passi della UI.
    Ordine: LBP(8,1) su griglia 3×3, LBP(8,2), LBP(16,2) in grigi; poi LBP(8,1) su H, S, V, Y, Cr, Cb."""
    small = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    feats = [lbp_hist(gray, 8, 1, 3)[0], lbp_hist(gray, 8, 2, 1)[0], lbp_hist(gray, 16, 2, 1)[0]]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV); ycc = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    for ch in [hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2], ycc[:, :, 0], ycc[:, :, 1], ycc[:, :, 2]]:
        feats.append(lbp_hist(ch, 8, 1, 1)[0])
    return np.concatenate(feats).astype(np.float32)


class LBPAnalyzer(Analyzer):
    id = "lbp"; name = "LBP micro-texture"; family = "classico"; order = 30; color = "#f0a35e"
    short = "Descrittore di texture locale: una foto stampata o uno schermo alterano la micro-texture della pelle (grana di stampa, pixel, riflessi)."
    reference = "Määttä et al., IJCB 2011 · Boulkenafet et al., IEEE TIFS 2016 (LBP in HSV/YCbCr)"
    reference_url = "https://ieeexplore.ieee.org/document/6117510"

    def reliability(self):
        return "trained" if is_trained(self.id) else "untrained"

    def fingerprint(self):
        return clf_fingerprint(self.id, "none", FEATURE_VERSION)

    def graph(self):
        nodes = COMMON_NODES + [NodeSpec("resize", "Resize 64×64"), NodeSpec("gray", "Scala di grigi"),
                                NodeSpec("lbp81", "LBP(8,1) 3×3"), NodeSpec("lbp82", "LBP(8,2)"), NodeSpec("lbp162", "LBP(16,2)"),
                                NodeSpec("color", "LBP in HSV/YCbCr"), NodeSpec("hist", "Istogrammi"),
                                NodeSpec("svm", "SVM", "model"), NodeSpec("verdict", "Verdetto", "output")]
        edges = COMMON_EDGES + [["crop", "resize"], ["resize", "gray"], ["gray", "lbp81"], ["gray", "lbp82"], ["gray", "lbp162"],
                                ["resize", "color"], ["lbp81", "hist"], ["lbp82", "hist"], ["lbp162", "hist"], ["color", "hist"],
                                ["hist", "svm"], ["svm", "verdict"]]
        return nodes, edges

    def run(self, ctx: RunContext) -> Result:
        aid = self.id
        step_input(ctx, aid); face = step_face(ctx, aid)
        crop = step_crop(ctx, aid, face, **CROP)
        with ctx.step(aid, "resize", "Normalizzazione a 64×64",
                      "Il metodo originale lavora su volti normalizzati a 64×64: rende i pattern confrontabili "
                      "indipendentemente dalla distanza dalla camera.") as s:
            small = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
            s.image(small)
        with ctx.step(aid, "gray", "Conversione in scala di grigi",
                      "Y = 0.299·R + 0.587·G + 0.114·B. L'LBP classico usa solo la luminanza: la micro-texture è "
                      "una proprietà di intensità, non di colore.") as s:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            s.image(gray); s.metric("media", float(gray.mean())); s.metric("dev. std", float(gray.std()))
        feats = []
        for sid, P, R, grid, desc in [
            ("lbp81", 8, 1, 3, "8 vicini a raggio 1, immagine divisa in 3×3 regioni (istogrammi concatenati): cattura la grana finissima."),
            ("lbp82", 8, 2, 1, "8 vicini a raggio 2: texture a scala media."),
            ("lbp162", 16, 2, 1, "16 vicini a raggio 2: pattern più ampi (bordi curvi, riflessi)."),
        ]:
            with ctx.step(aid, sid, f"LBP uniforme P={P}, R={R}",
                          f"Ogni pixel è codificato confrontandolo con i {P} vicini sul cerchio di raggio {R}: 1 se il vicino "
                          f"è più chiaro, 0 altrimenti. Si tengono solo i pattern 'uniformi' (≤2 transizioni), {P + 2} bin. {desc}") as s:
                h, lbp_img = lbp_hist(gray, P, R, grid)
                feats.append(h)
                s.image(colorize(lbp_img, cv2.COLORMAP_MAGMA))
                s.metric("bin istogramma", len(h)); s.metric("pattern uniformi (%)", float(100 * (lbp_img < P + 1).mean()))
        with ctx.step(aid, "color", "LBP negli spazi colore HSV e YCbCr",
                      "Boulkenafet et al. hanno mostrato che l'informazione cromatica aiuta molto: gli attacchi "
                      "alterano la saturazione (S) e la crominanza (Cb, Cr). Si calcola LBP(8,1) su ciascun canale.") as s:
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV); ycc = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
            maps = []
            for ch in [hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2], ycc[:, :, 0], ycc[:, :, 1], ycc[:, :, 2]]:
                h, m = lbp_hist(ch, 8, 1, 1); feats.append(h); maps.append(colorize(m, cv2.COLORMAP_MAGMA))
            s.image(tile([cv2.resize(m, (96, 96), interpolation=cv2.INTER_NEAREST) for m in maps], 6), max_side=700)
            s.note("Ordine: H, S, V, Y, Cr, Cb.")
        with ctx.step(aid, "hist", "Vettore di feature",
                      "Tutti gli istogrammi vengono concatenati in un unico vettore: è la 'firma di texture' del volto.") as s:
            vec = np.concatenate(feats).astype(np.float32)
            s.metric("dimensione vettore", int(vec.size))
            s.image(hist_image(vec))
        clf, meta = load_clf(aid)
        with ctx.step(aid, "svm", "Classificatore SVM",
                      "Una Support Vector Machine (kernel RBF) separa firme di volti reali da firme di attacchi. "
                      + (clf_note(meta) if clf is not None else
                         "Nessun modello addestrato per questo descrittore: il vettore è pronto per il training, non c'è verdetto.")) as s:
            if clf is not None:
                p_real = p_real_of(clf, vec, meta)   # classe 0 = bona fide
                s.metric("P(reale)", p_real)
                label = "real" if p_real >= 0.5 else "attack"
                rel = "trained"
            else:
                p_real, label, rel = None, "unknown", "untrained"
                s.note("Classificatore non addestrato.")
        with ctx.step(aid, "verdict", "Verdetto", "Decisione basata sulla firma di texture.") as s:
            s.metric("decisione", {"real": "VOLTO REALE", "attack": "ATTACCO", "unknown": "NON DETERMINABILE"}[label])
            if no_face(face):
                s.note("Nessun volto rilevato: texture calcolata sull'intera immagine.")
        expl = (f"Estratta la firma di micro-texture del volto: {int(vec.size)} valori (istogrammi LBP a tre scale in scala di grigi più i sei canali H, S, V, Y, Cr, Cb). "
                + (f"La SVM (addestrata su {meta.get('dataset', '?').upper() if meta else '?'}) ha assegnato P(reale) = {p_real:.2f}." if rel == "trained" else
                   "La SVM non è addestrata, quindi la firma è pronta ma non c'è verdetto."))
        return Result(aid, label, p_real, {"real": "Reale", "attack": "Attacco", "unknown": "SVM non addestrata"}[label], rel,
                      {"feature_dim": int(vec.size), "no_face": no_face(face)}, explanation=expl)


def hist_image(vec: np.ndarray, w: int = 640, h: int = 160) -> np.ndarray:
    """Istogramma a barre del vettore di feature."""
    img = np.full((h, w, 3), 24, np.uint8)
    n = len(vec); m = float(vec.max()) + 1e-9
    bw = max(1, w // n)
    for i, v in enumerate(vec):
        x = int(i * w / n); bh = int(v / m * (h - 10))
        cv2.rectangle(img, (x, h - bh), (x + bw, h), (240, 163, 94), -1)
    return img


registry.register(LBPAnalyzer())
