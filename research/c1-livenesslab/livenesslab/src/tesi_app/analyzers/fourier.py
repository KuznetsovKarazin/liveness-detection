"""
Analisi dello spettro di Fourier (Li, Wang, Tan, Jain 2004) e ricerca di pattern moiré (Garcia & de Queiroz 2015).
Euristica senza training: una foto stampata o ri-fotografata da uno schermo perde le alte frequenze; uno schermo
introduce picchi periodici (moiré) nello spettro. Le soglie sono state fissate su poche immagini campione, non sui
dataset della tesi: il verdetto è orientativo.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..core import Analyzer, NodeSpec, Result, RunContext, colorize, registry
from .common import COMMON_EDGES, COMMON_NODES, no_face, step_crop, step_face, step_input

HFD_LOW, HFD_HIGH = 0.24, 0.40     # HFD sotto/sopra cui il volto è considerato ricattura/reale (sui campioni)
MOIRE_PEAKS_MAX = 30.0             # numero di picchi periodici oltre cui la componente moiré vale 0


class FourierAnalyzer(Analyzer):
    id = "fourier"; name = "Spettro di Fourier"; family = "classico"; order = 31; color = "#5ec8f0"
    short = "Confronta l'energia alle alte frequenze e cerca picchi periodici (moiré): indizi fisici di ristampa o di schermo."
    reference = "Li et al., SPIE 2004 (High Frequency Descriptor) · Garcia & de Queiroz, IEEE TIFS 2015 (moiré)"
    reference_url = "https://doi.org/10.1117/12.541955"

    def fingerprint(self):
        return f"fourier:{HFD_LOW}-{HFD_HIGH}:{MOIRE_PEAKS_MAX}"

    def graph(self):
        nodes = COMMON_NODES + [NodeSpec("gray", "Scala di grigi"), NodeSpec("window", "Finestra di Hann"),
                                NodeSpec("fft", "FFT 2D"), NodeSpec("radial", "Profilo radiale"),
                                NodeSpec("hfd", "Descrittore HF"), NodeSpec("moire", "Picchi moiré"),
                                NodeSpec("score", "Score euristico", "decision"), NodeSpec("verdict", "Verdetto", "output")]
        edges = COMMON_EDGES + [["crop", "gray"], ["gray", "window"], ["window", "fft"], ["fft", "radial"], ["radial", "hfd"],
                                ["fft", "moire"], ["hfd", "score"], ["moire", "score"], ["score", "verdict"]]
        return nodes, edges

    def run(self, ctx: RunContext) -> Result:
        aid = self.id
        step_input(ctx, aid); face = step_face(ctx, aid)
        crop = step_crop(ctx, aid, face, margin=0.1, square=True)
        with ctx.step(aid, "gray", "Scala di grigi 256×256",
                      "La trasformata si calcola sulla luminanza; il volto è portato a 256×256 così le frequenze "
                      "sono confrontabili tra immagini diverse.") as s:
            gray = cv2.cvtColor(cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            s.image(gray)
        with ctx.step(aid, "window", "Finestra di Hann",
                      "Si moltiplica l'immagine per una finestra che si annulla ai bordi: evita che il taglio "
                      "brusco del ritaglio produca frequenze spurie (leakage).") as s:
            win = np.outer(np.hanning(256), np.hanning(256))
            g = gray.astype(np.float32) * win
            s.image(g)
        with ctx.step(aid, "fft", "Trasformata di Fourier 2D",
                      "FFT bidimensionale, spettro centrato (fftshift) e in scala logaritmica. Il centro sono le basse "
                      "frequenze (forme grandi), la periferia le alte (dettagli fini, rumore, grana).") as s:
            F = np.fft.fftshift(np.fft.fft2(g))
            mag = np.abs(F)
            logmag = np.log1p(mag)
            s.image(colorize(logmag, cv2.COLORMAP_INFERNO))
            s.metric("energia totale (log)", float(np.log10(mag.sum() + 1e-9)))
        with ctx.step(aid, "radial", "Profilo radiale dell'energia",
                      "Si somma l'energia lungo anelli concentrici: si ottiene una curva energia(frequenza). "
                      "Un volto reale ripreso dal vivo ha una coda alle alte frequenze più ricca di una ristampa.") as s:
            cy, cx = 128, 128
            yy, xx = np.indices(mag.shape)
            r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
            prof = np.bincount(r.ravel(), mag.ravel())[:128]
            prof_n = prof / (prof.sum() + 1e-9)
            s.image(curve_image(np.log1p(prof)))
            s.metric("frequenza mediana (px⁻¹·256)", int(np.searchsorted(np.cumsum(prof_n), 0.5)))
        with ctx.step(aid, "hfd", "High Frequency Descriptor (HFD)",
                      "HFD = energia oltre 1/3 della frequenza massima / energia totale (esclusa la componente continua). "
                      "Li et al. osservano che nelle foto stampate HFD crolla perché la stampa e la ri-cattura agiscono "
                      "da filtro passa-basso.") as s:
            total = mag.sum() - mag[cy, cx]
            hf = mag[r >= 128 // 3].sum()
            hfd = float(hf / (total + 1e-9))
            s.metric("HFD", hfd)
            s.image(colorize(np.where(r >= 128 // 3, logmag, 0), cv2.COLORMAP_INFERNO))
        with ctx.step(aid, "moire", "Ricerca di picchi periodici (moiré)",
                      "Un pattern periodico (pixel di uno schermo, retino di stampa) appare come picchi isolati e "
                      "simmetrici nello spettro, lontani dal centro. Si confronta ogni punto con la media locale: "
                      "i punti che la superano di molto sono candidati moiré.") as s:
            band = (r > 20) & (r < 120)
            local = cv2.GaussianBlur(logmag, (0, 0), 6)
            peaks = (logmag - local) * band
            thr = peaks[band].mean() + 4 * peaks[band].std()
            npk = int((peaks > thr).sum())
            strength = float(peaks[band].max())
            vis = colorize(logmag, cv2.COLORMAP_INFERNO)
            ys, xs = np.where(peaks > thr)
            for y, x in zip(ys[:200], xs[:200]):
                cv2.circle(vis, (int(x), int(y)), 3, (255, 255, 255), 1)
            s.image(vis)
            s.metric("picchi rilevati", npk); s.metric("intensità max picco", strength)
        with ctx.step(aid, "score", "Score euristico",
                      "Combinazione indicativa: HFD alto → più probabilmente reale; molti picchi periodici → più "
                      f"probabilmente schermo/stampa. Le soglie (HFD tra {HFD_LOW} e {HFD_HIGH}, picchi≈{int(MOIRE_PEAKS_MAX)}) sono state fissate su poche "
                      "immagini campione, NON sui dataset della tesi: il verdetto è solo orientativo e andrà ricalibrato.") as s:
            s_hfd = float(np.clip((hfd - HFD_LOW) / (HFD_HIGH - HFD_LOW), 0, 1))
            s_moire = float(np.clip(1 - npk / MOIRE_PEAKS_MAX, 0, 1))
            p_real = 0.6 * s_hfd + 0.4 * s_moire
            s.metric("componente HFD", s_hfd); s.metric("componente moiré", s_moire); s.metric("P(reale) euristica", p_real)
        label = "real" if p_real >= 0.5 else "attack"
        with ctx.step(aid, "verdict", "Verdetto", "Esito orientativo dell'analisi in frequenza.") as s:
            s.metric("decisione", "VOLTO REALE" if label == "real" else "ATTACCO (sospetto)")
            if no_face(face):
                s.note("Nessun volto rilevato: spettro calcolato sull'intera immagine.")
        expl = (f"Energia alle alte frequenze HFD = {hfd:.3f} (sui campioni: volto reale ≈ 0,38, stampe ≈ 0,27–0,29) e {npk} picchi periodici (moiré) nello spettro. "
                + ("Dettagli fini ben presenti: coerente con una ripresa dal vivo." if s_hfd > 0.6 else "Dettagli fini scarsi: tipico di una ristampa o di una ricattura, ma anche di una foto sfocata.")
                + (" I picchi periodici suggeriscono uno schermo o un retino di stampa." if npk > 10 else " Nessun pattern periodico evidente.")
                + " Soglie indicative, non calibrate sui dataset.")
        return Result(aid, label, p_real, ("Reale" if label == "real" else "Attacco") + f" ({p_real:.0%}, euristico)", "heuristic",
                      {"hfd": hfd, "moire_peaks": npk, "no_face": no_face(face)}, explanation=expl)


def curve_image(y: np.ndarray, w: int = 640, h: int = 200) -> np.ndarray:
    """Grafico a linea del profilo radiale, con la soglia di 1/3 della frequenza massima segnata."""
    img = np.full((h, w, 3), 24, np.uint8)
    y = y - y.min(); y = y / (y.max() + 1e-9)
    pts = [(int(i * (w - 1) / (len(y) - 1)), int(h - 10 - v * (h - 20))) for i, v in enumerate(y)]
    cv2.polylines(img, [np.array(pts, np.int32)], False, (240, 200, 94), 2)
    cv2.line(img, (w // 3, 0), (w // 3, h), (120, 120, 120), 1)
    cv2.putText(img, "alte frequenze ->", (w // 3 + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return img


registry.register(FourierAnalyzer())
