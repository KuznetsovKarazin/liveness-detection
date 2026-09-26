"""
Qualità dell'immagine (Image Quality Assessment) per il face anti-spoofing (Galbally, Marcel, Fierrez, IEEE TIP 2014):
una ri-cattura (stampa fotografata, schermo ripreso) degrada l'immagine in modo misurabile, anche quando l'occhio non lo
nota. Galbally et al. usano 25 misure di qualità: quelle "full-reference" confrontano l'immagine con una sua versione
leggermente sfocata (gaussiana 3×3, σ = 0,5: un'immagine già degradata cambia poco quando la si sfoca ancora), quelle
"no-reference" stimano la qualità da sole (blocchi JPEG, nitidezza, rumore, contrasto). Qui 18 misure, poi una SVM
addestrata con scripts/train_classic.py. BRISQUE (Mittal 2012) non è incluso: richiede i file del modello di
opencv_contrib, non presenti nel repository.
"""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
from skimage.measure import shannon_entropy
from skimage.metrics import structural_similarity

from ..core import Analyzer, NodeSpec, Result, RunContext, colorize, registry
from .classic import clf_fingerprint, clf_note, is_trained, load_clf, p_real_of
from .common import COMMON_EDGES, COMMON_NODES, no_face, step_crop, step_face, step_input

CROP = dict(margin=0.1, square=True)
CLASSIFIER = "svm"
FEATURE_VERSION = "1"           # da incrementare quando cambia il calcolo di `features()`
SIZE = 256
FULL_REF = ["MSE", "PSNR", "SNR", "SC", "MD", "AD", "NAE", "RAMD", "LMSE", "NXC", "SSIM", "TED"]
NO_REF = ["JQI", "HLFI", "blur_lap", "blur_crete", "noise", "entropy"]
NAMES = FULL_REF + NO_REF


def _jpeg_quality_index(g: np.ndarray) -> float:
    """Indice di qualità JPEG no-reference di Wang, Sheikh e Bovik (ICIP 2002): blocchi 8×8 (B), attività (A) e
    passaggi per lo zero (Z) lungo righe e colonne; formula empirica dell'articolo."""
    def one_dir(x):
        d = np.diff(x, axis=1).astype(np.float64)
        B = np.abs(d[:, 7::8]).mean() if d.shape[1] >= 8 else 0.0
        A = (8.0 * np.abs(d).mean() - B) / 7.0
        z = (d[:, :-1] * d[:, 1:]) < 0
        Z = z.mean()
        return B, A, Z
    Bh, Ah, Zh = one_dir(g); Bv, Av, Zv = one_dir(g.T)
    B, A, Z = (Bh + Bv) / 2, (Ah + Av) / 2, (Zh + Zv) / 2
    alpha, beta, g1, g2, g3 = -245.9, 261.9, -0.0240, 0.0160, 0.0064
    return float(alpha + beta * (B ** g1) * (A ** g2) * (Z ** g3)) if B > 0 and A > 0 and Z > 0 else 0.0


def _crete_blur(g: np.ndarray) -> float:
    """Metrica di sfocatura di Crété-Roffet et al. (SPIE 2007): quanto cambia la variazione tra pixel vicini se si sfoca
    ancora (0 = nitida, 1 = già sfocata)."""
    def one(x, k):
        b = cv2.blur(x, k)
        dv = np.abs(np.diff(x, axis=1)); db = np.abs(np.diff(b, axis=1))
        v = np.maximum(0, dv - db).sum()
        return 1.0 - v / (dv.sum() + 1e-9)
    return float(max(one(g, (1, 9)), one(g.T, (1, 9))))


def measures(gray: np.ndarray) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Le 18 misure di qualità (dizionario ordinato come NAMES) e alcune immagini intermedie."""
    I = gray.astype(np.float64)
    R = cv2.GaussianBlur(I, (3, 3), 0.5)                         # riferimento di Galbally: versione appena sfocata
    diff = I - R
    mse = float((diff ** 2).mean())
    m: Dict[str, float] = {}
    m["MSE"] = mse
    m["PSNR"] = float(10 * np.log10(255 ** 2 / (mse + 1e-9)))
    m["SNR"] = float(10 * np.log10(((I ** 2).sum() + 1e-9) / ((diff ** 2).sum() + 1e-9)))   # +1e-9 al numeratore: niente log(0) su un'immagine nera
    m["SC"] = float((I ** 2).sum() / ((R ** 2).sum() + 1e-9))
    m["MD"] = float(np.abs(diff).max())
    m["AD"] = float(diff.mean())
    m["NAE"] = float(np.abs(diff).sum() / (np.abs(I).sum() + 1e-9))
    m["RAMD"] = float(np.sort(np.abs(diff).ravel())[-10:].mean())
    lapI, lapR = cv2.Laplacian(I, cv2.CV_64F), cv2.Laplacian(R, cv2.CV_64F)
    m["LMSE"] = float(((lapI - lapR) ** 2).sum() / ((lapI ** 2).sum() + 1e-9))
    m["NXC"] = float((I * R).sum() / ((I ** 2).sum() + 1e-9))
    m["SSIM"] = float(structural_similarity(I, R, data_range=255.0))
    eI = np.hypot(cv2.Sobel(I, cv2.CV_64F, 1, 0), cv2.Sobel(I, cv2.CV_64F, 0, 1))
    eR = np.hypot(cv2.Sobel(R, cv2.CV_64F, 1, 0), cv2.Sobel(R, cv2.CV_64F, 0, 1))
    m["TED"] = float(np.abs(eI - eR).mean())
    m["JQI"] = _jpeg_quality_index(I)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(I)))
    yy, xx = np.indices(mag.shape); r = np.hypot(yy - SIZE / 2, xx - SIZE / 2)
    low, high = mag[r < SIZE / 8].sum(), mag[r >= SIZE / 8].sum()
    m["HLFI"] = float((low - high) / (low + high + 1e-9))
    m["blur_lap"] = float(lapI.var())
    m["blur_crete"] = _crete_blur(I)
    m["noise"] = float((I - cv2.medianBlur(gray, 3).astype(np.float64)).std())
    m["entropy"] = float(shannon_entropy(gray))
    return m, {"ref": R, "diff": diff, "lap": lapI, "edges": eI}


def features(crop_bgr: np.ndarray) -> np.ndarray:
    """Vettore delle 18 misure nell'ordine di NAMES."""
    gray = cv2.cvtColor(cv2.resize(crop_bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    m, _ = measures(gray)
    return np.array([m[k] for k in NAMES], np.float32)


class IQAAnalyzer(Analyzer):
    id = "iqa"; name = "Qualità d'immagine (IQA)"; family = "classico"; order = 34; color = "#e0c060"
    short = "Diciotto misure di qualità (nitidezza, rumore, blocchi JPEG, differenza da una versione sfocata): una ri-cattura è un'immagine degradata, e la degradazione si misura."
    reference = "Galbally, Marcel, Fierrez, Image Quality Assessment for Fake Biometric Detection, IEEE TIP 2014"
    reference_url = "https://doi.org/10.1109/TIP.2013.2292332"

    def reliability(self):
        return "trained" if is_trained(self.id) else "untrained"

    def fingerprint(self):
        return clf_fingerprint(self.id, "untrained-v1", FEATURE_VERSION)

    def graph(self):
        nodes = COMMON_NODES + [NodeSpec("gray", "Grigi 256×256"), NodeSpec("reference", "Riferimento sfocato"),
                                NodeSpec("fullref", "12 misure full-reference"), NodeSpec("noref", "6 misure no-reference"),
                                NodeSpec("features", "Vettore (18)"), NodeSpec("clf", "SVM", "model"),
                                NodeSpec("verdict", "Verdetto", "output")]
        edges = COMMON_EDGES + [["crop", "gray"], ["gray", "reference"], ["gray", "fullref"], ["reference", "fullref"],
                                ["gray", "noref"], ["fullref", "features"], ["noref", "features"], ["features", "clf"], ["clf", "verdict"]]
        return nodes, edges

    def run(self, ctx: RunContext) -> Result:
        aid = self.id
        step_input(ctx, aid); face = step_face(ctx, aid)
        crop = step_crop(ctx, aid, face, **CROP)
        with ctx.step(aid, "gray", "Scala di grigi 256×256",
                      "Le misure di qualità si calcolano sulla luminanza a dimensione fissa, così sono confrontabili tra immagini.") as s:
            gray = cv2.cvtColor(cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            s.image(gray)
        m, im = measures(gray)
        with ctx.step(aid, "reference", "Immagine di riferimento (sfocatura gaussiana 3×3, σ = 0,5)",
                      "Idea di Galbally et al.: non abbiamo l'originale con cui confrontare la foto, ma possiamo confrontarla "
                      "con una sua copia appena sfocata. Un'immagine nitida cambia molto, una già degradata (ri-cattura) cambia poco.") as s:
            s.image(im["ref"].astype(np.uint8))
        with ctx.step(aid, "fullref", "Misure full-reference (immagine vs riferimento)",
                      "MSE, PSNR, SNR, contenuto strutturale (SC), differenza massima (MD) e media (AD), errore assoluto normalizzato (NAE), "
                      "media delle 10 differenze maggiori (RAMD), errore sul Laplaciano (LMSE), cross-correlazione (NXC), SSIM e differenza dei bordi (TED).") as s:
            s.image(colorize(np.abs(im["diff"]), cv2.COLORMAP_VIRIDIS))
            for k in FULL_REF:
                s.metric(k, round(m[k], 4))
        with ctx.step(aid, "noref", "Misure no-reference",
                      "Indice di qualità JPEG (blocchi 8×8), indice alte/basse frequenze (HLFI), nitidezza (varianza del Laplaciano e "
                      "metrica di Crété), rumore (residuo dal filtro mediano) ed entropia.") as s:
            s.image(colorize(np.abs(im["lap"]), cv2.COLORMAP_MAGMA))
            for k in NO_REF:
                s.metric(k, round(m[k], 4))
        with ctx.step(aid, "features", "Vettore di feature", "Le 18 misure nell'ordine sopra, standardizzate dal classificatore.") as s:
            vec = np.array([m[k] for k in NAMES], np.float32)
            s.metric("dimensione vettore", int(vec.size))
        clf, meta = load_clf(aid)
        with ctx.step(aid, "clf", "Classificatore SVM",
                      "Galbally et al. usano LDA/QDA; qui una SVM con kernel RBF. "
                      + (clf_note(meta) if clf is not None else "Nessun classificatore addestrato: le misure ci sono, il verdetto no.")) as s:
            if clf is not None:
                p_real = p_real_of(clf, vec, meta); label = "real" if p_real >= 0.5 else "attack"; rel = "trained"
                s.metric("P(reale)", p_real)
            else:
                p_real, label, rel = None, "unknown", "untrained"
                s.note("Classificatore non addestrato.")
        with ctx.step(aid, "verdict", "Verdetto", "Decisione basata sulla qualità misurata dell'immagine.") as s:
            s.metric("decisione", {"real": "VOLTO REALE", "attack": "ATTACCO", "unknown": "NON DETERMINABILE"}[label])
            if no_face(face):
                s.note("Nessun volto rilevato: misure calcolate sull'intera immagine.")
        expl = (f"Nitidezza (varianza del Laplaciano) {m['blur_lap']:.0f}, sfocatura di Crété {m['blur_crete']:.2f}, SSIM con la copia sfocata {m['SSIM']:.3f} "
                f"(vicino a 1 = immagine già degradata), rumore {m['noise']:.2f}. "
                + (f"La SVM (addestrata su {meta.get('dataset', '?').upper() if meta else '?'}) ha assegnato P(reale) = {p_real:.2f}." if rel == "trained"
                   else "Classificatore non addestrato: nessun verdetto."))
        return Result(aid, label, p_real, {"real": "Reale", "attack": "Attacco", "unknown": "SVM non addestrata"}[label]
                      + (f" ({p_real:.0%})" if p_real is not None else ""), rel,
                      {"measures": {k: float(v) for k, v in m.items()}, "no_face": no_face(face)}, explanation=expl)


registry.register(IQAAnalyzer())
