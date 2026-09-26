"""
Image Distortion Analysis (Wen, Han, Jain, IEEE TIFS 2015): riflessi speculari, sfocatura, momenti cromatici,
diversità di colore. Con SVM addestrata (scripts/train_classic.py → models/weights/ida_clf.joblib) è un classificatore;
senza, uno score euristico.

Il vettore qui è una versione ridotta (16 valori) di quello originale (121 valori: l'articolo usa istogrammi
completi per la diversità di colore). Le soglie dell'euristica sono indicative, non calibrate sui dataset.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.stats import skew

from ..core import Analyzer, NodeSpec, Result, RunContext, colorize, registry
from .classic import clf_fingerprint, clf_note, is_trained, load_clf, p_real_of
from .common import COMMON_EDGES, COMMON_NODES, no_face, step_crop, step_face, step_input

CROP = dict(margin=0.1, square=True)     # stesso ritaglio in `run` e in `features` (scripts/train_classic.py)
CLASSIFIER = "svm"
FEATURE_VERSION = "1"           # da incrementare quando cambia il calcolo di `features()`


def _skew(ch: np.ndarray) -> float:
    """Asimmetria di un canale; 0 (non NaN) se il canale è costante, come in un'immagine a toni di grigio."""
    return 0.0 if ch.std() < 1e-6 else float(skew(ch))


def features(crop_bgr: np.ndarray) -> np.ndarray:
    """Le 16 misure IDA di un ritaglio del volto: la STESSA sequenza di `run`, senza i passi della UI."""
    crop = cv2.resize(crop_bgr, (256, 256), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    V, S = hsv[:, :, 2].astype(np.float32), hsv[:, :, 1].astype(np.float32)
    spec_mask = (V > 200) & (S < 60)
    f = [float(spec_mask.mean() * 100), float(V[spec_mask].mean()) if spec_mask.any() else 0.0, float(V[spec_mask].var()) if spec_mask.any() else 0.0]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0)).mean(); gb = np.abs(cv2.Sobel(blurred, cv2.CV_64F, 1, 0)).mean()
    f += [lap, float(gb / (gx + 1e-9))]
    for i in range(3):
        ch = hsv[:, :, i].astype(np.float32).ravel()
        f += [float(ch.mean()), float(ch.std()), _skew(ch)]
    q = (crop // 8).reshape(-1, 3)
    keys = q[:, 0].astype(np.int32) * 1024 + q[:, 1].astype(np.int32) * 32 + q[:, 2].astype(np.int32)
    uniq, counts = np.unique(keys, return_counts=True)
    f += [float(len(uniq)), float(np.sort(counts)[-100:].sum() / counts.sum())]
    return np.array(f, np.float32)


class IDAAnalyzer(Analyzer):
    id = "ida"; name = "Distorsioni IDA"; family = "classico"; order = 32; color = "#f06e8e"
    short = "Misura le distorsioni introdotte da un attacco: riflessi speculari della carta/schermo, sfocatura da ricattura, colori impoveriti."
    reference = "Wen, Han, Jain, Face Spoof Detection with Image Distortion Analysis, IEEE TIFS 2015"
    reference_url = "https://doi.org/10.1109/TIFS.2015.2400395"

    def reliability(self):
        return "trained" if is_trained(self.id) else "heuristic"

    def fingerprint(self):
        return clf_fingerprint(self.id, "heuristic-v1", FEATURE_VERSION)

    def graph(self):
        nodes = COMMON_NODES + [NodeSpec("specular", "Riflessi speculari"), NodeSpec("blur", "Sfocatura"),
                                NodeSpec("chroma", "Momenti cromatici"), NodeSpec("diversity", "Diversità colore"),
                                NodeSpec("vector", "Vettore IDA"), NodeSpec("clf", "SVM / euristica", "model"),
                                NodeSpec("verdict", "Verdetto", "output")]
        edges = COMMON_EDGES + [["crop", "specular"], ["crop", "blur"], ["crop", "chroma"], ["crop", "diversity"],
                                ["specular", "vector"], ["blur", "vector"], ["chroma", "vector"], ["diversity", "vector"],
                                ["vector", "clf"], ["clf", "verdict"]]
        return nodes, edges

    def run(self, ctx: RunContext) -> Result:
        aid = self.id
        step_input(ctx, aid); face = step_face(ctx, aid)
        crop = cv2.resize(step_crop(ctx, aid, face, **CROP), (256, 256), interpolation=cv2.INTER_AREA)
        f = []
        with ctx.step(aid, "specular", "Componente di riflessione speculare",
                      "Carta lucida e schermi riflettono la luce in modo diverso dalla pelle. Si stimano i pixel "
                      "speculari (alta intensità, bassa saturazione nel modello dicromatico) e se ne calcolano "
                      "percentuale, intensità media e varianza.") as s:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            V, S = hsv[:, :, 2].astype(np.float32), hsv[:, :, 1].astype(np.float32)
            spec_mask = (V > 200) & (S < 60)
            pct = float(spec_mask.mean() * 100)
            mean_i = float(V[spec_mask].mean()) if spec_mask.any() else 0.0
            var_i = float(V[spec_mask].var()) if spec_mask.any() else 0.0
            f += [pct, mean_i, var_i]
            vis = crop.copy(); vis[spec_mask] = (60, 220, 255)
            s.image(vis); s.metric("pixel speculari (%)", pct); s.metric("intensità media", mean_i); s.metric("varianza", var_i)
        with ctx.step(aid, "blur", "Sfocatura",
                      "Un attacco è una ri-cattura: perde nitidezza. Due misure: varianza del Laplaciano e "
                      "rapporto di nitidezza tra immagine e sua versione sfocata (Gaussiana σ=3).") as s:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            blurred = cv2.GaussianBlur(gray, (0, 0), 3)
            gx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0)).mean(); gb = np.abs(cv2.Sobel(blurred, cv2.CV_64F, 1, 0)).mean()
            ratio = float(gb / (gx + 1e-9))
            f += [lap, ratio]
            s.image(colorize(np.abs(cv2.Laplacian(gray, cv2.CV_64F)), cv2.COLORMAP_VIRIDIS))
            s.metric("varianza Laplaciano", lap); s.metric("rapporto sfocato/nitido", ratio)
        with ctx.step(aid, "chroma", "Momenti cromatici (HSV)",
                      "Media, deviazione standard e asimmetria (skewness) di H, S e V. Stampa e schermi comprimono la "
                      "gamma cromatica: il volto appare più 'piatto' o con dominanti.") as s:
            mom = []
            for i, name in enumerate("HSV"):
                ch = hsv[:, :, i].astype(np.float32).ravel()
                m, sd, sk = float(ch.mean()), float(ch.std()), _skew(ch)
                mom += [m, sd, sk]
                s.metric(f"{name}: media / std / skew", [round(m, 2), round(sd, 2), round(sk, 3)])
            f += mom
            s.image(np.hstack([colorize(hsv[:, :, 0], cv2.COLORMAP_HSV), cv2.cvtColor(hsv[:, :, 1], cv2.COLOR_GRAY2BGR),
                               cv2.cvtColor(hsv[:, :, 2], cv2.COLOR_GRAY2BGR)]), max_side=760)
            s.note("Da sinistra: tonalità H, saturazione S, valore V.")
        with ctx.step(aid, "diversity", "Diversità di colore",
                      "Si quantizza l'immagine a 32 livelli per canale e si contano i colori distinti e la quota "
                      "coperta dai 100 colori più frequenti. Un volto reale ha più varietà cromatica.") as s:
            q = (crop // 8).reshape(-1, 3)
            keys = q[:, 0].astype(np.int32) * 1024 + q[:, 1].astype(np.int32) * 32 + q[:, 2].astype(np.int32)
            uniq, counts = np.unique(keys, return_counts=True)
            top100 = float(np.sort(counts)[-100:].sum() / counts.sum())
            f += [len(uniq), top100]
            s.image((q.reshape(256, 256, 3) * 8).astype(np.uint8))
            s.metric("colori distinti (32³)", int(len(uniq))); s.metric("quota top-100 colori", top100)
        with ctx.step(aid, "vector", "Vettore di feature IDA",
                      "Le 16 misure vengono concatenate in un vettore (versione ridotta di quello a 121 valori di Wen et al.).") as s:
            vec = np.array(f, np.float32)
            s.metric("dimensione", int(vec.size)); s.metric("valori", [round(float(v), 3) for v in vec])
        clf, meta = load_clf(aid)
        with ctx.step(aid, "clf", "Classificazione",
                      ("Con la SVM addestrata il vettore viene classificato. " + clf_note(meta)) if clf is not None else
                      "Nessuna SVM addestrata: si usa un'euristica: nitidezza alta, pochi riflessi speculari e molta diversità "
                      "cromatica spingono verso 'reale'. Soglie indicative.") as s:
            if clf is not None:
                p_real = p_real_of(clf, vec, meta); rel = "trained"   # classe 0 = bona fide
                s_blur = s_spec = s_div = None
            else:
                s_blur = float(np.clip((lap - 50) / 400, 0, 1))
                s_spec = float(np.clip(1 - pct / 8, 0, 1))
                s_div = float(np.clip((len(uniq) - 600) / 2400, 0, 1))
                p_real = 0.45 * s_blur + 0.25 * s_spec + 0.30 * s_div; rel = "heuristic"
                s.metric("comp. nitidezza", s_blur); s.metric("comp. speculare", s_spec); s.metric("comp. diversità", s_div)
            s.metric("P(reale)", p_real)
        label = "real" if p_real >= 0.5 else "attack"
        with ctx.step(aid, "verdict", "Verdetto", "Esito dell'analisi delle distorsioni.") as s:
            s.metric("decisione", "VOLTO REALE" if label == "real" else "ATTACCO (sospetto)")
            if no_face(face):
                s.note("Nessun volto rilevato: distorsioni misurate sull'intera immagine.")
        expl = (f"Pixel speculari {pct:.1f}% (riflessi di carta o schermo se alti), varianza del Laplaciano {lap:.0f} (nitidezza: bassa = ricattura), "
                f"{len(uniq)} colori distinti (un volto dal vivo ne ha di più). "
                + (f"La SVM (addestrata su {meta.get('dataset', '?').upper() if meta else '?'}) ha assegnato P(reale) = {p_real:.2f}." if rel == "trained" else
                   f"Euristica: componenti nitidezza {s_blur:.2f}, riflessi {s_spec:.2f}, diversità {s_div:.2f} → P(reale) = {p_real:.2f}. Soglie indicative."))
        return Result(aid, label, p_real, ("Reale" if label == "real" else "Attacco") + f" ({p_real:.0%}" + (")" if rel == "trained" else ", euristico)"),
                      rel, {"vector": [float(v) for v in vec], "no_face": no_face(face)}, explanation=expl)


registry.register(IDAAnalyzer())
