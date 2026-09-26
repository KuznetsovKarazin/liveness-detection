"""
MiniFASNet (Silent-Face-Anti-Spoofing, MiniVision, Apache-2.0): modelli PyTorch pre-addestrati su ~360k immagini
proprietarie, fusione di due reti a scala diversa (2.7× MiniFASNetV2 e 4.0× MiniFASNetV1SE), input 80×80 BGR.
Il preprocessing riproduce `test.py` / `generate_patches.py` del repo originale: patch centrata sul volto, allargata
di `scale` volte, ridotta a 80×80, tensore BGR senza normalizzazione (come nel training originale).
"""
from __future__ import annotations

import threading
from collections import OrderedDict

import cv2
import numpy as np

from ..core import Analyzer, NodeSpec, Result, RunContext, registry
from ..paths import SILENT_FACE, load_module
from .common import COMMON_NODES, no_face, step_face, step_input

_lock = threading.Lock()
_models = {}          # nome file -> (modello, h, w, scala)

# i due modelli dell'ensemble; il nome codifica scala e dimensione di input (es. "2.7_80x80_MiniFASNetV2.pth")
MODELS = ["2.7_80x80_MiniFASNetV2.pth", "4_0_0_80x80_MiniFASNetV1SE.pth"]


def _load(name: str):
    """Costruisce la rete giusta per il nome del file, carica i pesi e la mette in cache."""
    with _lock:
        if name in _models:
            return _models[name]
        import torch
        mf = load_module("minifasnet_models", SILENT_FACE / "src" / "model_lib" / "MiniFASNet.py")
        info = name.split("_")[0:-1]                       # es. ["2.7", "80x80"] oppure ["4", "0", "0", "80x80"]
        h, w = (int(v) for v in info[-1].split("x"))
        mtype = name.split(".pth")[0].split("_")[-1]        # MiniFASNetV2 | MiniFASNetV1SE
        scale = float(info[0])                             # "4_0_0" è il modo del repo di scrivere 4.0
        cls = {"MiniFASNetV2": mf.MiniFASNetV2, "MiniFASNetV1SE": mf.MiniFASNetV1SE}[mtype]
        model = cls(conv6_kernel=((h + 15) // 16, (w + 15) // 16))
        sd = torch.load(str(SILENT_FACE / "resources" / "anti_spoof_models" / name), map_location="cpu")
        if next(iter(sd)).startswith("module."):           # pesi salvati da DataParallel: si toglie il prefisso
            sd = OrderedDict((k[7:], v) for k, v in sd.items())
        model.load_state_dict(sd)
        model.eval()
        _models[name] = (model, h, w, scale)
        return _models[name]


def _crop_scaled(img, bbox, scale, out_w, out_h):
    """Patch attorno al volto allargata di `scale` volte e ridotta a (out_w, out_h): stessa geometria di
    `CropImage._get_new_box` del repo originale (la patch viene spostata, non tagliata, se esce dall'immagine)."""
    src_h, src_w = img.shape[:2]
    x, y, bw, bh = bbox
    scale = min((src_h - 1) / bh, min((src_w - 1) / bw, scale))
    nw, nh = bw * scale, bh * scale
    cx, cy = bw / 2 + x, bh / 2 + y
    l, t, r, b = cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2
    if l < 0: r -= l; l = 0
    if t < 0: b -= t; t = 0
    if r > src_w - 1: l -= r - src_w + 1; r = src_w - 1
    if b > src_h - 1: t -= b - src_h + 1; b = src_h - 1
    l, t, r, b = int(l), int(t), int(r), int(b)
    return cv2.resize(img[t:b + 1, l:r + 1], (out_w, out_h)), (l, t, r, b)


class MiniFASNetAnalyzer(Analyzer):
    id = "minifasnet"; name = "MiniFASNet (pre-addestrato)"; family = "pretrained"; order = 20; color = "#22c9a8"
    short = "Due reti leggere (MobileNet-like) addestrate su ~360k immagini, fuse a scala 2.7× e 4×: guardano il volto e il suo contorno."
    reference = "MiniVision, Silent-Face-Anti-Spoofing (2020), Apache-2.0"
    reference_url = "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing"

    def reliability(self):
        return "trained"

    def fingerprint(self):
        from ..paths import file_sha256
        return "minifasnet:" + "+".join(file_sha256(SILENT_FACE / "resources" / "anti_spoof_models" / m)[:16] for m in MODELS)

    def graph(self):
        nodes = COMMON_NODES[:2] + [
            NodeSpec("crop27", "Patch 2.7× → 80×80"), NodeSpec("crop40", "Patch 4.0× → 80×80"),
            NodeSpec("net27", "MiniFASNetV2", "model"), NodeSpec("net40", "MiniFASNetV1SE", "model"),
            NodeSpec("fuse", "Fusione softmax", "decision"), NodeSpec("verdict", "Verdetto", "output")]
        edges = [["input", "face"], ["face", "crop27"], ["face", "crop40"], ["crop27", "net27"], ["crop40", "net40"],
                 ["net27", "fuse"], ["net40", "fuse"], ["fuse", "verdict"]]
        return nodes, edges

    def run(self, ctx: RunContext) -> Result:
        import torch
        aid = self.id
        step_input(ctx, aid)
        face = step_face(ctx, aid)
        total = np.zeros(3); per_net = []
        for name, sid, nid in zip(MODELS, ["crop27", "crop40"], ["net27", "net40"]):
            model, h, w, scale = _load(name)
            with ctx.step(aid, sid, f"Patch a scala {scale}× ridotta a {w}×{h}",
                          f"Il riquadro del volto viene allargato di {scale}× attorno al centro e ridotto a {w}×{h} pixel. "
                          "A scala maggiore la rete vede anche cornice della foto, bordo dello schermo o mani che reggono "
                          "il dispositivo: indizi di contesto decisivi per un attacco.") as s:
                patch, (l, t, r, b) = _crop_scaled(ctx.image, face.bbox, scale, w, h)
                s.image(patch)
                s.metric("regione [l,t,r,b]", [l, t, r, b]); s.metric("input rete", f"{w}×{h}×3")
            with ctx.step(aid, nid, f"Inferenza {name.split('_')[-1].replace('.pth','')}",
                          "La patch diventa un tensore (1,3,80,80) e attraversa una rete depthwise-separable con "
                          "attenzione (SE) e uscita a 3 classi. Softmax sulle 3 classi: l'indice 1 è 'volto reale'.") as s:
                x = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0)   # BGR, 0-255, come nel training originale
                with torch.no_grad():
                    p = torch.softmax(model(x), dim=1).numpy()[0]
                total += p; per_net.append(float(p[1]))
                s.metric("P(classe 0: attacco)", float(p[0])); s.metric("P(classe 1: reale)", float(p[1])); s.metric("P(classe 2: attacco)", float(p[2]))
                s.metric("parametri", f"{sum(t.numel() for t in model.parameters()):,}".replace(",", "."))
        with ctx.step(aid, "fuse", "Fusione delle due reti",
                      "Le probabilità delle due reti vengono sommate (ensemble a scala multipla) e divise per 2. "
                      "Si decide con la soglia 0,5 su P(reale); la confidenza è la probabilità della classe scelta.") as s:
            fused = total / len(MODELS)
            p_real = float(fused[1])
            s.metric("P(reale) fusa", p_real); s.metric("P(attacco) fusa", 1.0 - p_real)
        # decisione con la stessa regola di tutti gli analizzatori (reale se P(reale) >= 0,5); l'originale Silent-Face usa
        # l'argmax sulle 3 classi, che con P(reale) massima ma sotto 0,5 direbbe "reale" in disaccordo con la valutazione
        label = "real" if p_real >= 0.5 else "attack"
        with ctx.step(aid, "verdict", "Verdetto",
                      "Decisione finale dell'ensemble pre-addestrato. Il modello è stato addestrato su dati "
                      "proprietari di MiniVision: buon riferimento pratico, non confrontabile 1:1 con le CNN del docente.") as s:
            s.metric("decisione", "VOLTO REALE" if label == "real" else "ATTACCO DI PRESENTAZIONE")
            s.metric("confidenza", max(p_real, 1.0 - p_real))
            if no_face(face):
                s.note("Nessun volto rilevato: le patch sono state prese dall'intera immagine.")
        agree = (per_net[0] >= 0.5) == (per_net[1] >= 0.5)
        expl = (f"La rete a scala 2,7× (volto e poco contesto) dà P(reale) = {per_net[0]:.2f}; quella a 4× (con cornice, mani, sfondo) dà {per_net[1]:.2f}. "
                + ("Le due scale concordano" if agree else "Le due scale sono in disaccordo: il contesto e il volto raccontano cose diverse")
                + f"; la fusione dà P(reale) = {p_real:.2f} → {'volto reale' if label == 'real' else 'attacco di presentazione'}. "
                "Modello pre-addestrato su dati proprietari: attendibile ma non esente da bias.")
        return Result(aid, label, p_real, ("Reale" if label == "real" else "Attacco") + f" ({max(p_real, 1.0 - p_real):.0%})",
                      "trained", {"probs": [float(v) for v in fused], "no_face": no_face(face)}, explanation=expl)


registry.register(MiniFASNetAnalyzer())
