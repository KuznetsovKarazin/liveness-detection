"""
Sonde lineari (linear probe) su foundation model: l'encoder resta congelato e produce un embedding del volto; sopra
c'è solo una regressione logistica addestrata con scripts/train_classic.py sullo split ufficiale di training di un
dataset. Due varianti:
  - CLIP linear probe: lo stesso encoder immagine ViT-B/32 del CLIP zero-shot (embedding a 512 dimensioni), così si
    confronta direttamente "zero-shot con prompt" e "un classificatore lineare sugli stessi vettori";
  - DINOv2 linear probe: DINOv2-small (Oquab et al., 2023), auto-supervisionato senza testo, token [CLS] a 384
    dimensioni. Il benchmark di Fang et al. (IJCB 2026) mostra che le feature congelate vanno bene intra-dataset e
    trasferiscono poco cross-dataset: è esattamente la domanda della tesi.
DINOv2 viene registrato solo se i pesi sono già nella cache di Hugging Face (scripts/download_models.py): l'app non
scarica nulla da sola.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

from ..core import Analyzer, NodeSpec, Result, RunContext, colorize, registry
from . import clip_zeroshot
from .classic import clf_fingerprint, clf_note, is_trained, load_clf, p_real_of
from .common import COMMON_EDGES, COMMON_NODES, no_face, step_crop, step_face, step_input

CLASSIFIER = "logreg"
FEATURE_VERSION = "1"
CROP = dict(margin=0.35, square=True)            # come il CLIP zero-shot: il contesto (bordi, mani) è informativo

DINO_MODEL_ID = "facebook/dinov2-small"
DINO_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"   # commit su Hugging Face, fissato per riproducibilità
_dino = {}
_lock = threading.Lock()


# ------------------------------------------------------------------ CLIP
def clip_embedding(crop_bgr: np.ndarray) -> np.ndarray:
    """Embedding normalizzato (512) dell'encoder immagine di CLIP ViT-B/32."""
    import torch
    model, proc = clip_zeroshot._load()
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        e = clip_zeroshot._feat(model.get_image_features(**proc(images=rgb, return_tensors="pt")))
        e = e / e.norm(dim=-1, keepdim=True)
    return e[0].numpy().astype(np.float32)


def features(crop_bgr: np.ndarray) -> np.ndarray:   # per scripts/train_classic.py (analizzatore "clip_probe")
    return clip_embedding(crop_bgr)


class CLIPProbe(Analyzer):
    id = "clip_probe"; name = "CLIP linear probe"; family = "pretrained"; order = 22; color = "#38b6e0"
    short = "Lo stesso encoder di CLIP, ma senza prompt: sull'embedding a 512 dimensioni lavora una regressione logistica addestrata sui dataset della tesi. Misura quanto sanno già le feature di CLIP, senza fine-tuning."
    reference = "Radford et al., CLIP, ICML 2021 · Fang et al., linear-probing benchmark per il face PAD, IJCB 2026"
    reference_url = "https://arxiv.org/abs/2607.26993"

    def reliability(self):
        return "trained" if is_trained(self.id) else "untrained"

    def fingerprint(self):
        return "clip-probe:" + clip_zeroshot.MODEL_REVISION[:12] + ":" + clf_fingerprint(self.id, "untrained-v1", FEATURE_VERSION)

    def graph(self):
        nodes = COMMON_NODES + [NodeSpec("prep", "Preprocessing CLIP 224²"), NodeSpec("img_enc", "Encoder immagine ViT-B/32", "model"),
                                NodeSpec("embed", "Embedding (512)"), NodeSpec("clf", "Regressione logistica", "model"),
                                NodeSpec("verdict", "Verdetto", "output")]
        edges = COMMON_EDGES + [["crop", "prep"], ["prep", "img_enc"], ["img_enc", "embed"], ["embed", "clf"], ["clf", "verdict"]]
        return nodes, edges

    def run(self, ctx: RunContext) -> Result:
        aid = self.id
        step_input(ctx, aid); face = step_face(ctx, aid)
        crop = step_crop(ctx, aid, face, **CROP, note="Stesso ritaglio del CLIP zero-shot, così i due metodi vedono la stessa immagine.")
        with ctx.step(aid, "prep", "Preprocessing CLIP", "Lato corto a 224, ritaglio centrale, normalizzazione con media e deviazione di CLIP.") as s:
            s.image(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)); s.metric("ingresso", "224×224")
        with ctx.step(aid, "img_enc", "Encoder immagine (congelato)",
                      "Il Vision Transformer di CLIP produce il vettore del volto: nessun peso viene modificato, è lo stesso "
                      "modello del CLIP zero-shot.") as s:
            v = clip_embedding(crop)
            s.image(clip_zeroshot._embedding_image(v)); s.metric("dimensione", int(v.size)); s.metric("norma", float(np.linalg.norm(v)))
        with ctx.step(aid, "embed", "Embedding", "Il vettore a 512 dimensioni, normalizzato a norma 1, è l'ingresso del classificatore (che al suo interno lo standardizza con media e deviazione del training set).") as s:
            s.metric("min / max", [round(float(v.min()), 3), round(float(v.max()), 3)])
        return _classify(self, ctx, aid, face, v, "CLIP")


# ------------------------------------------------------------------ DINOv2
def dino_available() -> bool:
    """True se lo snapshot di DINOv2 alla revisione fissata è già nella cache locale di Hugging Face."""
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(DINO_MODEL_ID, "config.json", revision=DINO_REVISION), str)
    except Exception:  # noqa: BLE001
        return False


def _load_dino():
    with _lock:
        if "m" not in _dino:
            from transformers import AutoImageProcessor, AutoModel
            _dino["p"] = AutoImageProcessor.from_pretrained(DINO_MODEL_ID, revision=DINO_REVISION)
            _dino["m"] = AutoModel.from_pretrained(DINO_MODEL_ID, revision=DINO_REVISION).eval()
        return _dino["m"], _dino["p"]


def dino_embedding(crop_bgr: np.ndarray):
    """(embedding [CLS] normalizzato a 384, mappa delle norme dei token di patch 16×16)."""
    import torch
    model, proc = _load_dino()
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        out = model(**proc(images=rgb, return_tensors="pt"))
    h = out.last_hidden_state[0]                       # [1 + n_patch, 384]
    cls = h[0]; cls = cls / cls.norm()
    patches = h[1:]
    n = int(round(patches.shape[0] ** 0.5))
    pmap = patches.norm(dim=-1)[: n * n].reshape(n, n).numpy()
    return cls.numpy().astype(np.float32), pmap


def dino_features(crop_bgr: np.ndarray) -> np.ndarray:  # per scripts/train_classic.py (analizzatore "dinov2_probe")
    return dino_embedding(crop_bgr)[0]


class DINOv2Probe(Analyzer):
    id = "dinov2_probe"; name = "DINOv2 linear probe"; family = "pretrained"; order = 23; color = "#7fb3ff"
    short = "Feature auto-supervisionate di DINOv2-small (senza testo, senza etichette) più una regressione logistica addestrata sui dataset della tesi: quanto 'vede' un foundation model visivo generico?"
    reference = "Oquab et al., DINOv2, TMLR 2024 · Fang et al., linear-probing benchmark per il face PAD, IJCB 2026"
    reference_url = "https://arxiv.org/abs/2304.07193"

    def reliability(self):
        return "trained" if is_trained(self.id) else "untrained"

    def fingerprint(self):
        return "dinov2-probe:" + DINO_REVISION[:12] + ":" + clf_fingerprint(self.id, "untrained-v1", FEATURE_VERSION)

    def graph(self):
        nodes = COMMON_NODES + [NodeSpec("prep", "Preprocessing 224²"), NodeSpec("img_enc", "DINOv2-small ViT-S/14", "model"),
                                NodeSpec("embed", "Token [CLS] (384)"), NodeSpec("clf", "Regressione logistica", "model"),
                                NodeSpec("verdict", "Verdetto", "output")]
        edges = COMMON_EDGES + [["crop", "prep"], ["prep", "img_enc"], ["img_enc", "embed"], ["embed", "clf"], ["clf", "verdict"]]
        return nodes, edges

    def run(self, ctx: RunContext) -> Result:
        aid = self.id
        step_input(ctx, aid); face = step_face(ctx, aid)
        crop = step_crop(ctx, aid, face, **CROP)
        with ctx.step(aid, "prep", "Preprocessing DINOv2", "Ridimensionamento a 256, ritaglio centrale 224×224, normalizzazione ImageNet; patch da 14 pixel (16×16 token).") as s:
            s.image(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)); s.metric("ingresso", "224×224"); s.metric("patch", "16×16 da 14 px")
        with ctx.step(aid, "img_enc", "Encoder DINOv2-small (congelato)",
                      "Vision Transformer ViT-S/14 addestrato senza etichette (distillazione auto-supervisionata su 142 M immagini). "
                      "La mappa mostra la norma dei token di patch: dove il modello 'vede' più struttura.") as s:
            v, pmap = dino_embedding(crop)
            s.image(colorize(cv2.resize(pmap, (224, 224), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_VIRIDIS))
            s.metric("token", int(pmap.size) + 1); s.metric("dimensione [CLS]", int(v.size))
        with ctx.step(aid, "embed", "Token [CLS]", "Il token globale, normalizzato, è l'embedding del volto (384 valori).") as s:
            s.image(clip_zeroshot._embedding_image(v)); s.metric("min / max", [round(float(v.min()), 3), round(float(v.max()), 3)])
        return _classify(self, ctx, aid, face, v, "DINOv2")


# ------------------------------------------------------------------ classificazione comune
def _classify(an: Analyzer, ctx: RunContext, aid: str, face, v: np.ndarray, enc: str) -> Result:
    clf, meta = load_clf(aid)
    with ctx.step(aid, "clf", "Regressione logistica",
                  "Un classificatore lineare sull'embedding: nessun fine-tuning dell'encoder. "
                  + (clf_note(meta) if clf is not None else "Nessun classificatore addestrato: l'embedding c'è, il verdetto no.")) as s:
        if clf is not None:
            p_real = p_real_of(clf, v, meta); label = "real" if p_real >= 0.5 else "attack"; rel = "trained"
            s.metric("P(reale)", p_real)
        else:
            p_real, label, rel = None, "unknown", "untrained"
            s.note("Classificatore non addestrato.")
    with ctx.step(aid, "verdict", "Verdetto", f"Decisione della sonda lineare sulle feature di {enc}.") as s:
        s.metric("decisione", {"real": "VOLTO REALE", "attack": "ATTACCO", "unknown": "NON DETERMINABILE"}[label])
        if no_face(face):
            s.note("Nessun volto rilevato: embedding dell'intera immagine.")
    expl = (f"Embedding {enc} a {int(v.size)} dimensioni con encoder congelato. "
            + (f"La regressione logistica (addestrata su {meta.get('dataset', '?').upper() if meta else '?'}) ha assegnato P(reale) = {p_real:.2f}." if rel == "trained"
               else "Classificatore non addestrato: nessun verdetto."))
    return Result(aid, label, p_real, {"real": "Reale", "attack": "Attacco", "unknown": "Sonda non addestrata"}[label]
                  + (f" ({p_real:.0%})" if p_real is not None else ""), rel,
                  {"embedding_dim": int(v.size), "no_face": no_face(face)}, explanation=expl)


registry.register(CLIPProbe())
if dino_available():
    registry.register(DINOv2Probe())
else:
    print("DINOv2 linear probe non registrato: pesi non in cache (eseguire scripts/download_models.py)")
