"""
Difference of Gaussians (Tan, Li, Liu, Jiang, ECCV 2010): la baseline storica del dataset NUAA. Il volto in scala di
grigi a 64×64 viene filtrato con la differenza di due gaussiane (σ = 0,5 e 1,0): resta una banda di frequenze medie in
cui le stampe (che perdono i dettagli fini e aggiungono il retino) si distinguono dai volti dal vivo. Tan et al. usano
poi un classificatore sparso; Peixoto, Michelassi e Rocha (ICIP 2011) aggiungono statistiche dello spettro di Fourier
dell'immagine filtrata, più robuste alla cattiva illuminazione. Qui il vettore (52 valori) è: istogramma della
risposta DoG normalizzata (32 bin), profilo radiale dello spettro della DoG (16 bande) e 4 statistiche.
Come in Tan et al., che classificano i pixel dell'immagine filtrata con una regressione logistica sparsa, il vettore
include anche l'immagine DoG ridotta a 32×32 (1.024 valori, normalizzati): in tutto 1.076 valori. Classificatore:
regressione logistica addestrata con scripts/train_classic.py (senza, il metodo estrae le feature ma non dà verdetto).
"""
from __future__ import annotations

from typing import Dict

import cv2
import numpy as np
from scipy.stats import kurtosis

from ..core import Analyzer, NodeSpec, Result, RunContext, colorize, registry
from .classic import clf_fingerprint, clf_note, is_trained, load_clf, p_real_of
from .common import COMMON_EDGES, COMMON_NODES, no_face, step_crop, step_face, step_input

CROP = dict(margin=0.0, square=True)
CLASSIFIER = "logreg"          # come la regressione logistica (sparsa) di Tan et al.
FEATURE_VERSION = "2"          # 2: aggiunta l'immagine DoG 32×32 al vettore
RAW = 32                       # lato dell'immagine DoG usata come feature "grezza"
SIZE = 64                       # come in Tan et al.: volti normalizzati a 64×64
SIGMA_IN, SIGMA_OUT = 0.5, 1.0  # le due gaussiane dell'articolo
N_HIST, N_BANDS = 32, 16


def _parts(crop_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """Tutti i prodotti intermedi (usati sia da `features` sia da `run`, così i due percorsi coincidono)."""
    gray = cv2.cvtColor(cv2.resize(crop_bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    dog = cv2.GaussianBlur(gray, (0, 0), SIGMA_IN) - cv2.GaussianBlur(gray, (0, 0), SIGMA_OUT)
    std = float(dog.std())
    # ritaglio piatto: la DoG è un residuo di arrotondamento (≈1e-5) con deviazione zero; normalizzarlo darebbe valori
    # enormi e un istogramma vuoto (NaN): si usa una risposta nulla, come per un'immagine davvero senza struttura
    dogn = dog / std if std > 1e-6 else np.zeros_like(dog)
    sd = std + 1e-6
    hist, _ = np.histogram(dogn, bins=N_HIST, range=(-4, 4), density=True)
    hist = np.nan_to_num(hist)
    win = np.outer(np.hanning(SIZE), np.hanning(SIZE)).astype(np.float32)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(dog * win)))
    yy, xx = np.indices(mag.shape)
    r = np.sqrt((yy - SIZE / 2) ** 2 + (xx - SIZE / 2) ** 2)
    edges = np.linspace(0, SIZE / 2, N_BANDS + 1)
    prof = np.array([mag[(r >= edges[i]) & (r < edges[i + 1])].sum() for i in range(N_BANDS)], np.float64)
    prof = prof / (prof.sum() + 1e-9)
    hf = float(mag[r >= SIZE / 6].sum() / (mag.sum() - mag[SIZE // 2, SIZE // 2] + 1e-9))
    # su un ritaglio piatto (dev. std ≈ 0) la kurtosis è NaN: si usa 0, lo stesso valore che lo script di addestramento dà ai non finiti
    kurt = float(kurtosis(dogn.ravel())) if sd > 1e-5 else 0.0
    stats = np.array([sd, 0.0 if not np.isfinite(kurt) else kurt, float(np.abs(dogn).mean()), hf], np.float64)
    raw = cv2.resize(dogn, (RAW, RAW), interpolation=cv2.INTER_AREA).ravel()
    return {"gray": gray, "dog": dog, "dogn": dogn, "hist": hist, "mag": mag, "prof": prof, "stats": stats, "raw": raw}


def _vector(p: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([p["hist"], p["prof"], p["stats"], p["raw"]]).astype(np.float32)


def features(crop_bgr: np.ndarray) -> np.ndarray:
    """Vettore di feature DoG (1.076 valori): istogramma (32) + profilo radiale dello spettro (16) + statistiche (4)
    + immagine DoG normalizzata 32×32 (1.024)."""
    return _vector(_parts(crop_bgr))


class DoGAnalyzer(Analyzer):
    id = "dog"; name = "DoG (Tan et al. 2010)"; family = "classico"; order = 33; color = "#9fd66b"
    short = "Filtro Difference of Gaussians a 64×64: la banda di frequenze medie in cui le stampe si distinguono dai volti dal vivo (baseline storica del dataset NUAA), più le statistiche dello spettro."
    reference = "Tan, Li, Liu, Jiang, ECCV 2010 (NUAA) · Peixoto, Michelassi, Rocha, ICIP 2011"
    reference_url = "https://doi.org/10.1007/978-3-642-15567-3_37"

    def reliability(self):
        return "trained" if is_trained(self.id) else "untrained"

    def fingerprint(self):
        return clf_fingerprint(self.id, "untrained-v1", FEATURE_VERSION)

    def graph(self):
        nodes = COMMON_NODES + [NodeSpec("gray", "Grigi 64×64"), NodeSpec("dog", "DoG σ 0,5 − 1,0"),
                                NodeSpec("hist", "Istogramma DoG"), NodeSpec("fft", "Spettro della DoG"),
                                NodeSpec("features", "Vettore (1.076)"), NodeSpec("clf", "Regressione logistica", "model"),
                                NodeSpec("verdict", "Verdetto", "output")]
        edges = COMMON_EDGES + [["crop", "gray"], ["gray", "dog"], ["dog", "hist"], ["dog", "fft"], ["hist", "features"],
                                ["fft", "features"], ["features", "clf"], ["clf", "verdict"]]
        return nodes, edges

    def run(self, ctx: RunContext) -> Result:
        aid = self.id
        step_input(ctx, aid); face = step_face(ctx, aid)
        crop = step_crop(ctx, aid, face, **CROP)
        p = _parts(crop)
        with ctx.step(aid, "gray", "Scala di grigi 64×64",
                      "Come nell'articolo originale il volto viene normalizzato a 64×64 in luminanza: a questa scala la "
                      "grana di una stampa e la sfocatura di una ricattura sono ancora visibili, il rumore del sensore meno.") as s:
            s.image(p["gray"].astype(np.uint8)); s.metric("media", float(p["gray"].mean())); s.metric("dev. std", float(p["gray"].std()))
        with ctx.step(aid, "dog", f"Difference of Gaussians (σ = {SIGMA_IN} e {SIGMA_OUT})",
                      "L'immagine viene sfocata con due gaussiane e si prende la differenza: sparisce l'illuminazione lenta "
                      "(basse frequenze) e sparisce anche il rumore finissimo; resta una banda media in cui Tan et al. "
                      "osservano che i volti dal vivo hanno più energia e più struttura delle loro foto stampate.") as s:
            s.image(colorize(p["dogn"], cv2.COLORMAP_BONE))
            s.metric("dev. std DoG", float(p["stats"][0])); s.metric("kurtosis", float(p["stats"][1])); s.metric("media |DoG|", float(p["stats"][2]))
        with ctx.step(aid, "hist", "Istogramma della risposta DoG",
                      "Distribuzione dei valori filtrati, normalizzati alla loro deviazione standard, in 32 bin tra −4 e +4. "
                      "Una stampa concentra i valori vicino a zero (poca struttura), un volto dal vivo ha code più lunghe.") as s:
            s.image(_bars(p["hist"]))
            s.metric("bin", N_HIST); s.metric("quota |DoG| < 0,5σ", float(p["hist"][N_HIST // 2 - 2:N_HIST // 2 + 2].sum() * 8 / N_HIST))
        with ctx.step(aid, "fft", "Spettro di Fourier della DoG",
                      "Peixoto et al. calcolano la trasformata dell'immagine filtrata: il profilo radiale dell'energia (16 bande) "
                      "descrive come si distribuiscono le frequenze residue; il rapporto alte/totale dice quanto dettaglio è rimasto.") as s:
            s.image(colorize(np.log1p(p["mag"]), cv2.COLORMAP_INFERNO))
            s.metric("energia alte frequenze / totale", float(p["stats"][3]))
        with ctx.step(aid, "features", "Vettore di feature",
                      "Istogramma (32) + profilo radiale (16) + statistiche (dev. std, kurtosis, media |DoG|, quota alte frequenze) "
                      "+ l'immagine DoG ridotta a 32×32 (1.024 valori, come i pixel filtrati che Tan et al. danno al classificatore) = 1.076 valori.") as s:
            vec = _vector(p)
            s.image(colorize(p["raw"].reshape(RAW, RAW), cv2.COLORMAP_BONE)); s.metric("dimensione vettore", int(vec.size))
        clf, meta = load_clf(aid)
        with ctx.step(aid, "clf", "Regressione logistica",
                      "Come in Tan et al. (regressione logistica sparsa sui pixel filtrati; qui con regolarizzazione L2 e standardizzazione). "
                      + (clf_note(meta) if clf is not None else
                         "Nessun classificatore addestrato: il vettore è pronto per il training, non c'è verdetto.")) as s:
            if clf is not None:
                p_real = p_real_of(clf, vec, meta); label = "real" if p_real >= 0.5 else "attack"; rel = "trained"
                s.metric("P(reale)", p_real)
            else:
                p_real, label, rel = None, "unknown", "untrained"
                s.note("Classificatore non addestrato.")
        with ctx.step(aid, "verdict", "Verdetto", "Decisione basata sulla banda di frequenze medie del volto.") as s:
            s.metric("decisione", {"real": "VOLTO REALE", "attack": "ATTACCO", "unknown": "NON DETERMINABILE"}[label])
            if no_face(face):
                s.note("Nessun volto rilevato: filtro applicato all'intera immagine.")
        expl = (f"Risposta DoG con deviazione standard {p['stats'][0]:.1f}, kurtosis {p['stats'][1]:.2f} e {p['stats'][3] * 100:.0f} % di energia alle alte frequenze. "
                + (f"La regressione logistica (addestrata su {meta.get('dataset', '?').upper() if meta else '?'}) ha assegnato P(reale) = {p_real:.2f}." if rel == "trained"
                   else "Classificatore non addestrato: nessun verdetto."))
        return Result(aid, label, p_real, {"real": "Reale", "attack": "Attacco", "unknown": "Classificatore non addestrato"}[label]
                      + (f" ({p_real:.0%})" if p_real is not None else ""), rel,
                      {"feature_dim": int(vec.size), "no_face": no_face(face)}, explanation=expl)


def _bars(h: np.ndarray, w: int = 640, hh: int = 160) -> np.ndarray:
    """Istogramma a barre, con lo zero al centro."""
    img = np.full((hh, w, 3), 24, np.uint8)
    n = len(h); mx = float(np.nanmax(h)) if len(h) else 0.0; m = mx if np.isfinite(mx) and mx > 0 else 1.0; bw = w // n
    for i, v in enumerate(h):
        x0 = i * bw; y0 = int(hh - 8 - (v / m) * (hh - 16))
        cv2.rectangle(img, (x0 + 1, y0), (x0 + bw - 2, hh - 8), (94, 200, 240), -1)
    cv2.line(img, (w // 2, 0), (w // 2, hh), (120, 120, 120), 1)
    return img


registry.register(DoGAnalyzer())
