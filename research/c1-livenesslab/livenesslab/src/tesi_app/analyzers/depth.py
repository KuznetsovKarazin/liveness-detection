"""
Stima della profondità monoculare con Depth Anything V2 (Yang et al., NeurIPS 2024) come indizio ausiliario di
liveness (Liu, Jourabloo, Liu, CVPR 2018: un volto reale ha rilievo, una foto o uno schermo sono piatti).

È una profondità STIMATA da una sola foto a colori, non misurata da un sensore: il modello "sa" che i volti hanno
rilievo e tende a ricostruirlo anche da una stampa. Per questo è un'euristica, distinta dagli esperimenti con la
profondità misurata (3DMAD) previsti nella tesi.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

from ..core import Analyzer, NodeSpec, Result, RunContext, colorize, registry
from .common import COMMON_EDGES, COMMON_NODES, no_face, step_crop, step_face, step_input

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
MODEL_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"   # commit del modello su Hugging Face: fissato per riproducibilità
CROP_MARGIN = 0.6
_lock = threading.Lock()
_m = {}


def _load():
    """Carica processor e modello (una volta sola)."""
    with _lock:
        if "m" not in _m:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            _m["p"] = AutoImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
            _m["m"] = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, revision=MODEL_REVISION).eval()
        return _m["m"], _m["p"]


class DepthAnalyzer(Analyzer):
    id = "depth"; name = "Profondità (Depth Anything)"; family = "pretrained"; order = 22; color = "#ffd166"
    short = "Ricostruisce una mappa di profondità 3D da una sola foto: un volto vero ha naso e guance in rilievo, carta e schermi sono superfici piatte."
    reference = "Yang et al., Depth Anything V2, NeurIPS 2024 · Liu, Jourabloo, Liu, CVPR 2018 (depth come supervisione ausiliaria)"
    reference_url = "https://arxiv.org/abs/2406.09414"

    def fingerprint(self):
        return "depth:" + MODEL_ID + "@" + MODEL_REVISION[:12] + ":v2"

    def graph(self):
        nodes = COMMON_NODES + [NodeSpec("prep", "Preprocessing 518 px"), NodeSpec("dpt", "Depth Anything V2 (ViT-S)", "model"),
                                NodeSpec("depthmap", "Mappa di profondità"), NodeSpec("plane", "Fit del piano"),
                                NodeSpec("relief", "Rilievo del volto"), NodeSpec("score", "Score euristico", "decision"),
                                NodeSpec("verdict", "Verdetto", "output")]
        edges = COMMON_EDGES + [["crop", "prep"], ["prep", "dpt"], ["dpt", "depthmap"], ["depthmap", "plane"], ["depthmap", "relief"],
                                ["plane", "score"], ["relief", "score"], ["score", "verdict"]]
        return nodes, edges

    def run(self, ctx: RunContext) -> Result:
        import torch
        aid = self.id
        step_input(ctx, aid); face = step_face(ctx, aid)
        crop = step_crop(ctx, aid, face, margin=CROP_MARGIN, square=True,
                         note="Margine ampio: la profondità del contorno (spalle, sfondo, eventuale cornice di carta o schermo) è parte dell'indizio.")
        # posizione reale del ritaglio nell'immagine: serve per sapere dove sta il volto DENTRO il ritaglio,
        # perché ai bordi dell'immagine il ritaglio viene tagliato e non è più centrato sul volto
        x0, y0, _, _ = face.crop_box(ctx.image.shape, margin=CROP_MARGIN, square=True)
        model, proc = _load()
        with ctx.step(aid, "prep", "Preprocessing",
                      "Il ritaglio viene portato a 518 pixel (multiplo di 14, la dimensione delle patch del ViT) e normalizzato "
                      "con media/deviazione standard di ImageNet.") as s:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            inputs = proc(images=rgb, return_tensors="pt")
            s.image(crop); s.metric("tensore", list(inputs["pixel_values"].shape))
        with ctx.step(aid, "dpt", "Inferenza Depth Anything V2",
                      "Un Vision Transformer (DINOv2 small, 25 M parametri) con testa DPT predice per ogni pixel una profondità "
                      "relativa (valori alti = più vicino). Il modello è addestrato su 62 M immagini con pseudo-etichette.") as s:
            with torch.no_grad():
                out = model(**inputs).predicted_depth[0].numpy()
            depth = cv2.resize(out, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)
            s.metric("min", float(depth.min())); s.metric("max", float(depth.max()))
            s.metric("parametri", f"{sum(p.numel() for p in model.parameters()):,}".replace(",", "."))
            s.image(colorize(depth, cv2.COLORMAP_INFERNO))
        with ctx.step(aid, "depthmap", "Mappa di profondità del volto",
                      "Profondità normalizzata nel riquadro del volto (0 = più lontano, 1 = più vicino). Nei volti reali il naso è "
                      "il punto più vicino, gli occhi e i bordi del viso i più lontani.") as s:
            H, W = crop.shape[:2]
            fx0, fy0 = max(0, face.x - x0), max(0, face.y - y0)
            fx1, fy1 = min(W, face.x - x0 + face.w), min(H, face.y - y0 + face.h)
            fd = depth[fy0:fy1, fx0:fx1]
            if fd.size < 4:                              # riquadro degenere: si usa tutto il ritaglio
                fd = depth; fx0, fy0, fx1, fy1 = 0, 0, W, H
            fdn = (fd - fd.min()) / (fd.max() - fd.min() + 1e-9)
            s.image(colorize(fdn, cv2.COLORMAP_VIRIDIS)); s.metric("riquadro volto nel ritaglio", [fx0, fy0, fx1, fy1])
        with ctx.step(aid, "plane", "Fit di un piano ai minimi quadrati",
                      "Si adatta un piano z = a·x + b·y + c alla profondità del volto. Una superficie piatta (foto, schermo) "
                      "è descritta quasi perfettamente dal piano: residuo basso. Un volto reale lascia un residuo strutturato.") as s:
            h, w = fdn.shape
            yy, xx = np.mgrid[0:h, 0:w]
            A = np.c_[xx.ravel(), yy.ravel(), np.ones(h * w)]
            coef, *_ = np.linalg.lstsq(A, fdn.ravel(), rcond=None)
            plane = (A @ coef).reshape(h, w)
            resid = fdn - plane
            resid_std = float(resid.std())
            s.image(colorize(np.abs(resid), cv2.COLORMAP_MAGMA))
            s.metric("residuo (dev. std)", resid_std); s.metric("inclinazione piano (a, b)", [round(float(coef[0]) * w, 3), round(float(coef[1]) * h, 3)])
        with ctx.step(aid, "relief", "Rilievo naso-contorno",
                      "Differenza tra la profondità nella zona centrale (naso) e quella ai bordi del volto (guance/orecchie), "
                      "in unità normalizzate. Il rilievo di un volto reale è tipicamente marcato; quello di una foto quasi nullo.") as s:
            cy, cx = h // 2, w // 2
            dh, dw = max(1, h // 10), max(1, w // 10)      # finestre mai vuote, anche su ritagli minuscoli
            bw = max(1, w // 8)
            nose = float(fdn[cy - dh:cy + dh, cx - dw:cx + dw].mean())
            border = float(np.concatenate([fdn[:, :bw].ravel(), fdn[:, -bw:].ravel()]).mean())
            relief = nose - border
            vis = cv2.cvtColor((fdn * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            cv2.rectangle(vis, (cx - dw, cy - dh), (cx + dw, cy + dh), (80, 220, 120), 2)
            cv2.rectangle(vis, (0, 0), (bw, h), (90, 90, 235), 2); cv2.rectangle(vis, (w - bw, 0), (w - 1, h), (90, 90, 235), 2)
            s.image(vis); s.metric("profondità naso", nose); s.metric("profondità bordi", border); s.metric("rilievo", relief)
        with ctx.step(aid, "score", "Score euristico",
                      "Combina residuo del piano e rilievo. LIMITE IMPORTANTE: i modelli di profondità monoculare 'sanno' che "
                      "un volto ha rilievo e tendono a ricostruirlo anche da una foto stampata; l'indizio funziona meglio quando "
                      "nel ritaglio compaiono la cornice della carta, lo schermo o le mani. Soglie indicative, da calibrare.") as s:
            s_res = float(np.clip((resid_std - 0.06) / (0.16 - 0.06), 0, 1))
            s_rel = float(np.clip((relief - 0.10) / (0.45 - 0.10), 0, 1))
            p_real = 0.5 * s_res + 0.5 * s_rel
            s.metric("componente residuo", s_res); s.metric("componente rilievo", s_rel); s.metric("P(reale) euristica", p_real)
        label = "real" if p_real >= 0.5 else "attack"
        with ctx.step(aid, "verdict", "Verdetto", "Esito orientativo basato sulla geometria 3D ricostruita.") as s:
            s.metric("decisione", "VOLTO REALE" if label == "real" else "ATTACCO (sospetto)")
            if no_face(face):
                s.note("Nessun volto rilevato: geometria calcolata sull'intera immagine.")
        expl = (f"Residuo del piano {resid_std:.3f} (piatto < 0,06, volto reale > 0,16) e rilievo naso-contorno {relief:.2f} (piatto < 0,10, reale > 0,45): "
                + ("la superficie ricostruita ha il rilievo di un volto." if p_real >= 0.5 else "la superficie ricostruita è quasi piatta, come carta o schermo.")
                + " Attenzione: il modello tende a 'inventare' il rilievo anche da una stampa, quindi l'indizio è debole se nel ritaglio non si vede il supporto.")
        return Result(aid, label, p_real, ("Reale" if label == "real" else "Attacco") + f" ({p_real:.0%}, euristico)", "heuristic",
                      {"resid_std": resid_std, "relief": relief, "no_face": no_face(face)}, explanation=expl)


registry.register(DepthAnalyzer())
