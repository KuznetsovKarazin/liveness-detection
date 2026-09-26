"""Passi condivisi tra analizzatori: ingresso immagine, rilevamento del volto, ritaglio."""
from __future__ import annotations

import numpy as np

from ..core import NodeSpec, RunContext
from ..face import Face, draw_face, get_detector


def step_input(ctx: RunContext, aid: str) -> np.ndarray:
    """Passo 'input': mostra l'immagine così com'è stata decodificata (BGR a 8 bit)."""
    img = ctx.image
    with ctx.step(aid, "input", "Immagine in ingresso",
                  "L'immagine viene decodificata in una matrice di pixel BGR a 8 bit (formato nativo di OpenCV). "
                  "Da qui in poi ogni passo trasforma questa matrice.") as s:
        h, w = img.shape[:2]
        s.image(img)
        s.metric("larghezza", w); s.metric("altezza", h); s.metric("canali", img.shape[2] if img.ndim == 3 else 1)
        s.metric("megapixel", round(w * h / 1e6, 2))
    return img


def step_face(ctx: RunContext, aid: str) -> Face:
    """Passo 'face': rilevamento del volto, calcolato una volta sola per analisi e condiviso tra gli analizzatori.
    Se nessun volto viene trovato si prosegue sull'intera immagine con un Face fittizio a confidenza 0:
    gli analizzatori lo segnalano nel verdetto (`no_face`) e la valutazione su dataset lo conteggia."""
    def _detect():
        return get_detector().detect(ctx.image)

    with ctx.step(aid, "face", "Rilevamento del volto",
                  "Un rilevatore RetinaFace (rete convoluzionale addestrata su WIDER FACE) individua il volto più "
                  "probabile e ne restituisce il riquadro (bounding box). Tutti gli algoritmi lavorano solo sulla "
                  "regione del volto: lo sfondo non contiene indizi di liveness e introdurrebbe rumore.") as s:
        face = ctx.cached("face", _detect)
        if face is None:
            s.note("Nessun volto trovato: l'analisi prosegue sull'intera immagine e il verdetto va preso con cautela.")
            H, W = ctx.image.shape[:2]
            face = Face(0, 0, W, H, 0.0, "nessuno (intera immagine)")
            ctx.cache["face"] = face
        s.image(draw_face(ctx.image, face))
        s.metric("rilevatore", face.detector)
        s.metric("confidenza", round(face.confidence, 3))
        s.metric("bbox [x,y,w,h]", face.bbox)
    return face


def no_face(face: Face) -> bool:
    """True se il volto è quello fittizio di `step_face` (nessun rilevamento)."""
    return face.confidence == 0.0


def step_crop(ctx: RunContext, aid: str, face: Face, margin: float, square: bool, note: str = "") -> np.ndarray:
    """Passo 'crop': ritaglio del volto con margine relativo; con `square` il riquadro è reso quadrato
    (al netto del taglio ai bordi dell'immagine, che può renderlo rettangolare)."""
    with ctx.step(aid, "crop", "Ritaglio della regione del volto",
                  f"Il riquadro viene allargato del {int(margin*100)}% per lato" + (" e reso quadrato" if square else "") +
                  ", così da includere il contorno del volto (capelli, bordo di una eventuale foto o schermo), "
                  "dove spesso si concentrano gli artefatti di un attacco. " + note) as s:
        crop = face.crop(ctx.image, margin=margin, square=square)
        s.image(crop)
        s.metric("dimensione ritaglio", f"{crop.shape[1]}×{crop.shape[0]}")
    return crop


# nodi e archi iniziali comuni a quasi tutte le pipeline
COMMON_NODES = [NodeSpec("input", "Immagine", "input"), NodeSpec("face", "Rilevamento volto"), NodeSpec("crop", "Ritaglio volto")]
COMMON_EDGES = [["input", "face"], ["face", "crop"]]
