"""
CLIP zero-shot in stile FLIP (Srivatsan, Naseer, Nandakumar, ICCV 2023): un modello visione-linguaggio
confronta l'immagine del volto con descrizioni testuali di "volto reale" e di "attacco" senza alcun training.

Prompt ensemble (come in FLIP e nel CLIP originale): ogni descrizione viene inserita in 8 formulazioni ("a photo of…",
"a close-up photo of…", "a blurry photo of…", …) e gli 8 embedding testuali vengono mediati e rinormalizzati; il
risultato è un embedding per descrizione, meno sensibile alla singola frase.
Aggregazione: le similarità vengono raggruppate per classe con una media in scala logaritmica (logsumexp − log n),
così il numero diverso di frasi per classe (4 "reale", 6 "attacco") non sposta il risultato. Con una softmax su tutte
le frasi, come nella prima versione, sei frasi contro quattro avrebbero introdotto un prior 60/40 verso l'attacco.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np
from scipy.special import logsumexp

from ..core import Analyzer, NodeSpec, Result, RunContext, registry
from .common import COMMON_EDGES, COMMON_NODES, no_face, step_crop, step_face, step_input

MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"   # commit del modello su Hugging Face: fissato per riproducibilità
# (frase inglese usata dal modello, traduzione mostrata nella UI)
REAL_PROMPTS = [
    ("a real human face", "un volto umano reale"),
    ("a live person looking at the camera", "una persona dal vivo che guarda la camera"),
    ("a real person taking a selfie", "una persona reale che si fa un selfie"),
    ("a person with natural skin in a close-up portrait", "un primo piano di una persona con pelle naturale"),
]
ATTACK_PROMPTS = [
    ("a printed photo of a face", "una foto stampata di un volto"),
    ("a photograph of a photograph of a face", "la fotografia di una fotografia di un volto"),
    ("a face displayed on a phone screen", "un volto mostrato sullo schermo di un telefono"),
    ("a face on a computer monitor", "un volto su un monitor"),
    ("a person wearing a realistic mask", "una persona con una maschera realistica"),
    ("a paper cutout of a face", "un ritaglio di carta di un volto"),
]
# formulazioni dell'ensemble (sottoinsieme dei template di CLIP usati anche da FLIP): {} = descrizione
TEMPLATES = ["a photo of {}.", "a close-up photo of {}.", "a cropped photo of {}.", "a bright photo of {}.",
             "a dark photo of {}.", "a blurry photo of {}.", "a low resolution photo of {}.", "a bad photo of {}."]
_lock = threading.Lock()
_model = {}


def _feat(out):
    """transformers ≥5 restituisce un oggetto con pooler_output; le versioni precedenti un tensore."""
    return out.pooler_output if hasattr(out, "pooler_output") else out


def _load():
    """Carica CLIP e pre-calcola gli embedding dei prompt (una volta sola)."""
    with _lock:
        if "m" in _model:
            return _model["m"], _model["p"]
        import torch
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).eval()
        proc = CLIPProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        with torch.no_grad():
            prompts = REAL_PROMPTS + ATTACK_PROMPTS
            texts = [tpl.format(p[0]) for p in prompts for tpl in TEMPLATES]      # n_prompt × n_template frasi
            t = proc(text=texts, return_tensors="pt", padding=True)
            temb = _feat(model.get_text_features(**t))
            temb = temb / temb.norm(dim=-1, keepdim=True)
            temb = temb.reshape(len(prompts), len(TEMPLATES), -1).mean(dim=1)   # ensemble: media delle formulazioni
            temb = temb / temb.norm(dim=-1, keepdim=True)
        _model.update(m=model, p=proc, temb=temb)
        return model, proc


class CLIPZeroShot(Analyzer):
    id = "clip_zeroshot"; name = "CLIP zero-shot (FLIP)"; family = "pretrained"; order = 21; color = "#4fd1ff"
    short = "Un modello visione-linguaggio (400 M coppie immagine-testo) giudica se la foto somiglia più a 'un volto reale' o a 'una foto stampata / uno schermo', senza training sul nostro problema."
    reference = "Radford et al., CLIP, ICML 2021 · Srivatsan et al., FLIP, ICCV 2023"
    reference_url = "https://arxiv.org/abs/2309.16649"

    def reliability(self):
        return "zeroshot"

    def fingerprint(self):
        # cambia se cambiano modello o prompt: i punteggi in cache dipendono da entrambi
        return "clip:" + MODEL_ID + "@" + MODEL_REVISION[:12] + ":" + "|".join(p[0] for p in REAL_PROMPTS + ATTACK_PROMPTS) + ":v3-ensemble" + str(len(TEMPLATES))

    def graph(self):
        nodes = COMMON_NODES + [NodeSpec("prep", "Preprocessing CLIP 224²"), NodeSpec("img_enc", "Encoder immagine ViT-B/32", "model"),
                                NodeSpec("prompts", "Prompt testuali"), NodeSpec("txt_enc", "Encoder testo", "model"),
                                NodeSpec("sim", "Similarità coseno"), NodeSpec("softmax", "Softmax e aggregazione", "decision"),
                                NodeSpec("verdict", "Verdetto", "output")]
        edges = COMMON_EDGES + [["crop", "prep"], ["prep", "img_enc"], ["input", "prompts"], ["prompts", "txt_enc"],
                                ["img_enc", "sim"], ["txt_enc", "sim"], ["sim", "softmax"], ["softmax", "verdict"]]
        return nodes, edges

    def run(self, ctx: RunContext) -> Result:
        import torch
        aid = self.id
        step_input(ctx, aid); face = step_face(ctx, aid)
        crop = step_crop(ctx, aid, face, margin=0.35, square=True,
                         note="Per CLIP conviene includere il contesto (bordi di carta o schermo, mani): sono proprio gli indizi descritti dai prompt di attacco.")
        model, proc = _load()
        with ctx.step(aid, "prep", "Preprocessing CLIP",
                      "Ridimensionamento del lato corto a 224, ritaglio centrale 224×224, normalizzazione con media e "
                      "deviazione standard di CLIP. L'immagine diventa una griglia di 7×7 patch da 32 pixel.") as s:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            inputs = proc(images=rgb, return_tensors="pt")
            px = inputs["pixel_values"][0].numpy()
            # ricostruzione dell'immagine normalizzata solo per mostrarla (de-normalizzazione con i valori di CLIP)
            vis = (px.transpose(1, 2, 0) * np.array([0.2686, 0.2613, 0.2758]) + np.array([0.4815, 0.4578, 0.4082]))
            vis = (np.clip(vis, 0, 1) * 255).astype(np.uint8)
            vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            for k in range(1, 7):
                cv2.line(vis, (k * 32, 0), (k * 32, 224), (80, 80, 80), 1); cv2.line(vis, (0, k * 32), (224, k * 32), (80, 80, 80), 1)
            s.image(vis); s.metric("tensore", list(px.shape)); s.metric("patch", "7×7 da 32 px")
        with ctx.step(aid, "img_enc", "Encoder immagine (Vision Transformer)",
                      "Le 49 patch più un token [CLS] attraversano 12 blocchi di attenzione; il token finale viene "
                      "proiettato in uno spazio a 512 dimensioni condiviso con il testo e normalizzato a norma 1.") as s:
            with torch.no_grad():
                iemb = _feat(model.get_image_features(pixel_values=inputs["pixel_values"]))
                iemb = iemb / iemb.norm(dim=-1, keepdim=True)
            v = iemb[0].numpy()
            s.image(_embedding_image(v)); s.metric("dimensione embedding", int(v.size)); s.metric("norma", float(np.linalg.norm(v)))
        with ctx.step(aid, "prompts", "Prompt testuali",
                      "Descriviamo in linguaggio naturale le due ipotesi. Prompt 'reale': " + "; ".join(p[1] for p in REAL_PROMPTS) +
                      ". Prompt 'attacco': " + "; ".join(p[1] for p in ATTACK_PROMPTS) + ". Sono in inglese perché CLIP è addestrato su testo inglese. "
                      f"Ogni descrizione è inserita in {len(TEMPLATES)} formulazioni (\"a photo of…\", \"a blurry photo of…\", …) i cui embedding vengono mediati: è il prompt ensemble di CLIP e FLIP.") as s:
            s.metric("prompt reale", len(REAL_PROMPTS)); s.metric("prompt attacco", len(ATTACK_PROMPTS)); s.metric("formulazioni per prompt", len(TEMPLATES))
        with ctx.step(aid, "txt_enc", "Encoder testo (Transformer)",
                      "Ogni frase viene tokenizzata (BPE) e codificata da un Transformer a 12 layer nello stesso spazio "
                      "a 512 dimensioni dell'immagine. Gli embedding dei prompt sono pre-calcolati una volta sola.") as s:
            temb = _model["temb"]
            s.metric("frasi codificate", int(temb.shape[0] * len(TEMPLATES))); s.metric("embedding per descrizione (dopo la media)", int(temb.shape[0])); s.metric("dimensione", int(temb.shape[1]))
        with ctx.step(aid, "sim", "Similarità coseno immagine-testo",
                      "Prodotto scalare tra l'embedding dell'immagine e quello di ogni frase, moltiplicato per la "
                      "temperatura di CLIP (100). Più la frase descrive l'immagine, più il valore è alto.") as s:
            with torch.no_grad():
                logits = (100.0 * iemb @ temb.T)[0].numpy()
            for (en, it), l in zip(REAL_PROMPTS + ATTACK_PROMPTS, logits):
                s.metric(it, float(l))
            s.image(_bars_image(REAL_PROMPTS + ATTACK_PROMPTS, logits, len(REAL_PROMPTS)))
        with ctx.step(aid, "softmax", "Aggregazione per classe e softmax",
                      "Le similarità di ogni classe vengono riassunte con una media in scala logaritmica (logsumexp − log n), "
                      "così il numero diverso di frasi per classe non pesa; poi una softmax a due vie dà P(reale) e P(attacco). "
                      "È un giudizio zero-shot: nessun esempio di attacco è mai stato mostrato al modello per questo compito.") as s:
            n_r = len(REAL_PROMPTS)
            z_real = logsumexp(logits[:n_r]) - np.log(n_r)
            z_att = logsumexp(logits[n_r:]) - np.log(len(ATTACK_PROMPTS))
            p_real = float(1.0 / (1.0 + np.exp(z_att - z_real)))
            best = int(np.argmax(logits))
            s.metric("P(reale)", p_real); s.metric("P(attacco)", 1 - p_real)
            s.metric("frase più vicina", (REAL_PROMPTS + ATTACK_PROMPTS)[best][1])
        label = "real" if p_real >= 0.5 else "attack"
        with ctx.step(aid, "verdict", "Verdetto",
                      "Decisione zero-shot. FLIP mostra che, con un fine-tuning leggero, questa impostazione supera le CNN "
                      "nel cross-dataset: qui vediamo il punto di partenza senza alcun addestramento.") as s:
            s.metric("decisione", "VOLTO REALE" if label == "real" else "ATTACCO DI PRESENTAZIONE")
            if no_face(face):
                s.note("Nessun volto rilevato: il giudizio riguarda l'intera immagine.")
        order = np.argsort(-logits)[:2]; allp = REAL_PROMPTS + ATTACK_PROMPTS
        expl = (f"La descrizione più vicina all'immagine è «{allp[best][1]}» (similarità {logits[best]:.1f}), seguita da «{allp[order[1]][1]}» ({logits[order[1]]:.1f}). "
                f"Aggregando per classe: reale {p_real:.2f}, attacco {1 - p_real:.2f}. "
                + ("Margine ampio: il modello è abbastanza sicuro." if abs(p_real - 0.5) > 0.25 else "Margine stretto: giudizio incerto, da leggere con cautela.")
                + " Nessun addestramento sul nostro problema (zero-shot).")
        return Result(aid, label, p_real, ("Reale" if label == "real" else "Attacco") + f" ({max(p_real, 1 - p_real):.0%}, zero-shot)", "zeroshot",
                      {"p_real": p_real, "best_prompt": allp[best][0], "no_face": no_face(face)}, explanation=expl)


def _embedding_image(v: np.ndarray, rows: int = 16) -> np.ndarray:
    """I valori dell'embedding disposti in una griglia colorata a 16 righe (512 → 16×32, 384 → 16×24), solo per mostrarli."""
    v = np.asarray(v, np.float32).ravel()
    cols = int(np.ceil(v.size / rows))
    m = np.pad(v, (0, rows * cols - v.size)).reshape(rows, cols)
    img = cv2.applyColorMap(cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_TWILIGHT)
    return cv2.resize(img, (cols * 16, rows * 16), interpolation=cv2.INTER_NEAREST)


def _bars_image(prompts, logits, n_real, w=760, h=330) -> np.ndarray:
    """Barre orizzontali delle similarità: verdi i prompt 'reale', rosse quelli 'attacco'."""
    img = np.full((h, w, 3), 24, np.uint8)
    lo, hi = float(logits.min()) - 1, float(logits.max()) + 1
    rowh = h // len(prompts)
    for i, ((en, it), l) in enumerate(zip(prompts, logits)):
        y = i * rowh + 6
        color = (120, 220, 90) if i < n_real else (90, 90, 235)
        bw = int((l - lo) / (hi - lo) * (w - 330))
        cv2.rectangle(img, (320, y), (320 + bw, y + rowh - 12), color, -1)
        cv2.putText(img, it[:44], (8, y + rowh - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
        cv2.putText(img, f"{l:.1f}", (326 + bw, y + rowh - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    return img


registry.register(CLIPZeroShot())
