"""
Le quattro architetture del prof. Kuznetsov (LivenessNet, AttackNetV1, AttackNetV2.1, AttackNetV2.2), costruite con
il codice originale del suo repository (src/livedetection/src/architectures.py, importato per file) e precedute dal
suo preprocessing (`EnhancedDatasetCreator.advanced_image_enhancement` di scripts/create_datasets.py) riprodotto
passo per passo, così che ogni trasformazione sia visibile nella UI.

Pesi: models/weights/<Arch>__<dataset>.h5 (uno per ogni dataset di addestramento; vedi scripts/train_cnn.py).
Per ogni file di pesi viene registrata una variante "<id>__<dataset>" (es. livenessnet__nuaa); senza alcun file la rete
è registrata una volta sola con pesi casuali e verdetto marcato "non addestrato" (la pipeline resta dimostrabile).
Le varianti "-pooled" sono addestrate con lo split casuale 80/20 del docente, che mescola train e test ufficiali:
per questo NON vengono valutate sul proprio dataset nella modalità Valutazione (le immagini sarebbero nel training).
"""
from __future__ import annotations

import os
import json
import threading
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..core import Analyzer, NodeSpec, Result, RunContext, colorize, registry, tile
from ..paths import LIVEDETECTION, WEIGHTS, file_sha256, load_module
from .common import COMMON_EDGES, COMMON_NODES, no_face, step_crop, step_face, step_input

WEIGHT_EXTS = (".h5", ".weights.h5", ".keras")   # ordine di preferenza: l'HDF5 classico è l'unico che carica ovunque

_tf = None
_models: Dict[str, object] = {}       # chiave pesi -> (modello Keras, pesi_caricati: bool)
_submodels: Dict[str, object] = {}    # sotto-modelli per feature map e Grad-CAM
_lock = threading.Lock()


def tf():
    """Import pigro di TensorFlow (pesante e lento): avviene solo alla prima analisi con una CNN."""
    global _tf
    if _tf is None:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as _t
        # Il pass "remapper" di Grappler fallisce sulle AttackNet (concat + BiasAdd) sia su Metal sia su CPU Windows
        # ("Mutation::Apply error ... concat2/concat"): lo disattiviamo per evitare errori e possibili crash del processo.
        try:
            _t.config.optimizer.set_experimental_options({"remapping": False})
        except Exception:  # noqa: BLE001
            pass
        _tf = _t
    return _tf


def weights_file(key: str):
    """Il file di pesi per la chiave (es. "LivenessNet__nuaa"), nel primo formato disponibile, oppure None.
    ".h5" (senza .weights) è l'HDF5 classico, indipendente dalla piattaforma; ".weights.h5" e ".keras" sono il nuovo
    formato (saving_lib) che su Windows fallisce con "expected N variables, received 0"."""
    for ext in WEIGHT_EXTS:
        p = WEIGHTS / f"{key}{ext}"
        if p.exists():
            return p
    return None


def card_sha256(key: str) -> Optional[str]:
    """Hash dei pesi dichiarato dalla scheda models/weights/<key>.json (scritta dall'addestramento). Permette di
    riconoscere la variante e di ricalcolare le metriche dalla cache anche quando il file dei pesi non c'è
    (repository pubblico: schede sì, pesi no)."""
    pj = WEIGHTS / f"{key}.json"
    if not pj.exists():
        return None
    try:
        return json.loads(pj.read_text(encoding="utf-8")).get("weights_sha256") or None
    except (OSError, ValueError):
        return None


def load_model(arch_name: str, weights_key: Optional[str] = None):
    """Costruisce (o recupera dalla cache) il modello Keras dell'architettura e ne carica i pesi se presenti.
    Restituisce (modello, pesi_caricati)."""
    key = weights_key or arch_name
    with _lock:
        if key in _models:
            return _models[key]
        tf()
        arch = load_module("kuznetsov_architectures", LIVEDETECTION / "src" / "architectures.py")  # repo del docente
        model = arch.create_model(arch_name).get_model()
        p = weights_file(key)
        if p is not None:
            model.load_weights(str(p))
        elif card_sha256(key):
            raise RuntimeError(f"pesi {key} dichiarati dalla scheda JSON ma file assente in {WEIGHTS}: copiare i pesi (SHA-256 nella scheda)")
        _models[key] = (model, p is not None)
        return _models[key]


def _feature_model(key: str, model):
    """Sotto-modello che restituisce l'uscita del primo blocco convoluzionale (layer 1 = conv1_1 in tutte e quattro le reti)."""
    with _lock:
        if key not in _submodels:
            _submodels[key] = tf().keras.Model(model.input, model.get_layer(index=1).output)
        return _submodels[key]


def _gradcam_model(key: str, model):
    """Sotto-modello per Grad-CAM: restituisce (ultima mappa convoluzionale, ingresso dell'ultimo Dense) più i pesi
    dell'ultimo Dense, così il gradiente si calcola sui logit e non sulla softmax (che satura a 0/1 e annulla i gradienti)."""
    with _lock:
        if key not in _submodels:
            T = tf()
            conv_layer = next(layer for layer in reversed(model.layers) if isinstance(layer, T.keras.layers.Conv2D))
            last = model.layers[-1]                       # Dense(2, softmax) in tutte e quattro le architetture
            sub = T.keras.Model(model.inputs, [conv_layer.output, last.input])
            _submodels[key] = (sub, last.kernel, last.bias)
        return _submodels[key]


def trained_variants(arch_name: str) -> List[str]:
    """Elenco dei dataset per cui esistono pesi: models/weights/<Arch>__<dataset>.{h5,weights.h5,keras}."""
    if not WEIGHTS.exists():
        return []
    keys = set()
    for p in WEIGHTS.glob(f"{arch_name}__*"):
        # si toglie l'estensione più lunga che combacia: "A__nuaa.weights.h5" -> "A__nuaa", non "A__nuaa.weights"
        ext = next((e for e in sorted(WEIGHT_EXTS + (".json",), key=len, reverse=True) if p.name.endswith(e)), None)
        if ext == ".json" and not card_sha256(p.stem):
            continue                                   # scheda senza hash: non basta a identificare i pesi
        if ext:
            keys.add(p.name[: -len(ext)])
    return sorted(k.split("__", 1)[1] for k in keys)


DATASET_ORDER = ["nuaa", "casia_fasd", "celeba_spoof", "synthaspoof", "msspoof", "3dmad"]   # ordine dei sotto-gruppi nel catalogo
DATASET_LABELS = {"nuaa": "NUAA", "casia_fasd": "CASIA-FASD", "celeba_spoof": "CelebA-Spoof", "msspoof": "MSSpoof", "3dmad": "3DMAD",
                  "nuaa-pooled": "NUAA (split 80/20 del docente)", "casia_fasd-pooled": "CASIA-FASD (split 80/20 del docente)",
                  "celeba_spoof-pooled": "CelebA-Spoof (split 80/20 del docente)",
                  "synthaspoof": "SynthASpoof", "synthaspoof-pooled": "SynthASpoof (split 80/20 del docente)",
                  "msspoof-pooled": "MSSpoof (split 80/20 del docente)", "3dmad-pooled": "3DMAD (split 80/20 del docente)"}


def enhance_kuznetsov(ctx: RunContext, aid: str, img_rgb: np.ndarray) -> np.ndarray:
    """Riproduce `EnhancedDatasetCreator.advanced_image_enhancement` (repo del docente) come passi separati,
    con gli stessi parametri: bilaterale 9/75/75, CLAHE 3.0 su L (griglia 8×8), unsharp mask 1.5/−0.5 (σ=2),
    gamma 1.2 e scala lineare α=1.1 β=5. Ingresso e uscita in RGB, uint8."""
    with ctx.step(aid, "bilateral", "Filtro bilaterale (riduzione rumore)",
                  "Filtro bilaterale 9×9 (σcolor=75, σspace=75): attenua il rumore del sensore preservando i bordi. "
                  "È il primo passo dell'enhancement definito nel repo del docente.") as s:
        out = cv2.bilateralFilter(img_rgb, 9, 75, 75)
        s.image(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        s.metric("differenza media |in-out|", float(np.abs(out.astype(int) - img_rgb.astype(int)).mean()))
    with ctx.step(aid, "clahe", "CLAHE sul canale L (spazio LAB)",
                  "L'immagine passa in spazio LAB; sul solo canale di luminanza L si applica l'equalizzazione "
                  "adattiva dell'istogramma (clipLimit=3, griglia 8×8). Esalta il contrasto locale e quindi la "
                  "micro-texture della pelle, senza alterare i colori.") as s:
        lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0]
        L2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(L)
        lab[:, :, 0] = L2
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        s.image(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        s.metric("contrasto RMS prima", float(L.std())); s.metric("contrasto RMS dopo", float(L2.std()))
    with ctx.step(aid, "usm", "Unsharp masking (nitidezza)",
                  "Si sottrae una versione sfocata (Gaussiana σ=2) dall'immagine: out = 1.5·img − 0.5·blur. "
                  "Amplifica i dettagli ad alta frequenza, dove foto stampate e schermi perdono informazione.") as s:
        g = cv2.GaussianBlur(out, (0, 0), 2.0)
        out = cv2.addWeighted(out, 1.5, g, -0.5, 0)
        s.image(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        s.metric("varianza Laplaciano", float(cv2.Laplacian(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()))
    with ctx.step(aid, "gamma", "Correzione gamma e scala finale",
                  "Correzione gamma 1.2 (schiarisce i mezzitoni) seguita da una scala lineare (α=1.1, β=+5). "
                  "Chiude il preprocessing del docente: l'immagine è ora 'enhanced' come nel suo dataset.") as s:
        gc = (np.power(out.astype(np.float32) / 255.0, 1.0 / 1.2) * 255).astype(np.uint8)
        out = cv2.convertScaleAbs(gc, alpha=1.1, beta=5)
        s.image(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        s.metric("luminosità media", float(out.mean()))
    return out


def quality_check(ctx: RunContext, aid: str, img_rgb: np.ndarray) -> Dict[str, float]:
    """Misure di qualità analoghe a quelle con cui il docente filtra il dataset (nitidezza, contrasto, luminosità).
    Qui non si ricostruisce il suo punteggio composito con soglia 0.65: le misure servono solo a spiegare
    un'eventuale incertezza del verdetto."""
    with ctx.step(aid, "quality", "Controllo qualità",
                  "Il docente scarta dal dataset le immagini di qualità insufficiente. Qui si calcolano misure analoghe: "
                  "nitidezza (Laplaciano + Sobel), contrasto RMS e luminosità. Un valore basso spiega un'eventuale "
                  "incertezza del modello; nessuna immagine viene scartata.") as s:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, 3); sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, 3)
        sob = float(np.sqrt(sx ** 2 + sy ** 2).mean())
        sharp = lap * 0.7 + sob * 0.3
        contrast = float(gray.std())
        bright = float(gray.mean())
        s.image(colorize(np.abs(cv2.Laplacian(gray, cv2.CV_64F))))
        s.metric("nitidezza (Laplaciano·0.7 + Sobel·0.3)", float(sharp))
        s.metric("contrasto RMS", contrast); s.metric("luminosità media", bright)
        s.note("La mappa mostra il modulo del Laplaciano: le zone chiare sono i bordi netti.")
    return {"sharpness": float(sharp), "contrast": contrast, "brightness": bright}


class KuznetsovCNN(Analyzer):
    """Base delle quattro CNN del docente. Le sottoclassi fissano `arch`; `_register_all` crea una variante per dataset."""
    family = "docente"
    arch = "LivenessNet"
    color = "#8b7cff"
    trained_on: Optional[str] = None     # id del dataset di addestramento (variante); None = nessun peso

    @property
    def weights_key(self) -> str:
        return f"{self.arch}__{self.trained_on}" if self.trained_on else self.arch

    def reliability(self) -> str:
        """Solo un controllo di esistenza del file: il modello viene costruito e caricato alla prima analisi."""
        return "trained" if weights_file(self.weights_key) is not None or card_sha256(self.weights_key) else "untrained"

    def fingerprint(self) -> str:
        """SHA-256 del file dei pesi; se il file manca ma la scheda JSON lo dichiara, quello della scheda (stesso valore:
        così la cache dei punteggi resta valida e le metriche si ricalcolano senza i pesi)."""
        p = weights_file(self.weights_key)
        if p is not None:
            return file_sha256(p)
        return card_sha256(self.weights_key) or "untrained"

    def excluded_datasets(self) -> List[str]:
        """I pesi "-pooled" hanno visto anche il test ufficiale del proprio dataset: niente valutazione lì."""
        if self.trained_on and self.trained_on.endswith("-pooled"):
            return [self.trained_on[: -len("-pooled")]]
        return []

    def exclusion_note(self, ds_id: str) -> str:
        return ("non valutabile qui: con lo split 80/20 del docente le immagini di questo dataset fanno parte "
                "del training del modello (il test del 20 % tenuto fuori è riportato in models/weights/*.json)")

    def graph(self):
        nodes = COMMON_NODES + [
            NodeSpec("resize", "Resize 256×256"), NodeSpec("bilateral", "Filtro bilaterale"),
            NodeSpec("clahe", "CLAHE (LAB)"), NodeSpec("usm", "Unsharp mask"), NodeSpec("gamma", "Gamma + scala"),
            NodeSpec("quality", "Controllo qualità"), NodeSpec("normalize", "Normalizzazione [0,1]"),
            NodeSpec("cnn", self.arch, "model"), NodeSpec("features", "Feature map conv1"),
            NodeSpec("gradcam", "Grad-CAM"), NodeSpec("softmax", "Softmax", "decision"), NodeSpec("verdict", "Verdetto", "output"),
        ]
        edges = COMMON_EDGES + [["crop", "resize"], ["resize", "bilateral"], ["bilateral", "clahe"], ["clahe", "usm"],
                                ["usm", "gamma"], ["gamma", "quality"], ["quality", "normalize"], ["normalize", "cnn"],
                                ["cnn", "features"], ["cnn", "gradcam"], ["cnn", "softmax"], ["softmax", "verdict"]]
        return nodes, edges

    def run(self, ctx: RunContext) -> Result:
        aid = self.id
        step_input(ctx, aid)
        face = step_face(ctx, aid)
        crop = step_crop(ctx, aid, face, margin=0.15, square=True,
                         note="Nel dataset del docente le immagini sono già ritagli del volto.")
        with ctx.step(aid, "resize", "Ridimensionamento a 256×256",
                      "Le reti del docente lavorano a 256×256 pixel. Qui si usa interpolazione INTER_AREA, la stessa con cui "
                      "sono stati addestrati i pesi della tesi (il repo del docente usa Lanczos: differenza minima, ma va "
                      "detta). Da qui l'immagine è in ordine RGB come nel training.") as s:
            rgb = cv2.cvtColor(cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            s.image(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)); s.metric("dimensione", "256×256×3")
        enhanced = enhance_kuznetsov(ctx, aid, rgb)
        q = quality_check(ctx, aid, enhanced)
        with ctx.step(aid, "normalize", "Normalizzazione in [0,1]",
                      "I valori 0–255 vengono divisi per 255 e convertiti in float32, esattamente come in "
                      "`_finalize_dataset` del repo. Il tensore diventa (1, 256, 256, 3).") as s:
            x = enhanced.astype(np.float32) / 255.0
            s.image(cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
            s.metric("min", float(x.min())); s.metric("max", float(x.max())); s.metric("media", float(x.mean()))
            s.metric("forma tensore", [1, 256, 256, 3])
        model, trained = load_model(self.arch, self.weights_key)
        with ctx.step(aid, "cnn", f"Inferenza {self.arch}",
                      self.short + " " + (f"Pesi addestrati su {DATASET_LABELS.get(self.trained_on, self.trained_on)} caricati da models/weights." if trained else
                      "ATTENZIONE: nessun peso addestrato trovato per questa rete: è inizializzata a caso e il risultato non è significativo.")) as s:
            xb = x[None]
            probs = model(xb, training=False).numpy()[0]
            s.metric("parametri", f"{model.count_params():,}".replace(",", "."))
            s.metric("layer", len(model.layers))
            s.metric("stato pesi", "addestrati" if trained else "casuali (non addestrati)")
            s.metric("uscita grezza", [round(float(p), 4) for p in probs])
        with ctx.step(aid, "features", "Feature map del primo blocco convoluzionale",
                      "Attivazioni dei 16 filtri 3×3 del primo layer: mostrano cosa 'vede' la rete all'inizio "
                      "(bordi, gradienti, texture). Nei modelli addestrati alcuni filtri rispondono ai pattern "
                      "di stampa o ai pixel di uno schermo.") as s:
            sub = _feature_model("fm:" + self.weights_key, model)
            fm = sub(xb, training=False).numpy()[0]
            maps = [colorize(fm[:, :, i], cv2.COLORMAP_VIRIDIS) for i in range(min(16, fm.shape[-1]))]
            s.image(tile([cv2.resize(m, (96, 96)) for m in maps], 8), max_side=800)
            s.metric("filtri", fm.shape[-1]); s.metric("risoluzione mappa", f"{fm.shape[0]}×{fm.shape[1]}")
        with ctx.step(aid, "gradcam", "Grad-CAM: dove guarda la rete",
                      "Gradiente del logit della classe predetta rispetto all'ultima mappa convoluzionale, pesato e "
                      "sovrapposto al volto. Le zone calde sono quelle che hanno spinto la decisione. "
                      "Serve a verificare che la rete guardi la pelle e non lo sfondo.") as s:
            heat, flat = grad_cam(_gradcam_model("gc:" + self.weights_key, model), xb, int(np.argmax(probs)))
            heat = cv2.resize(heat, (256, 256))
            overlay = cv2.addWeighted(cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR), 0.55, colorize(heat, cv2.COLORMAP_JET), 0.45, 0)
            s.image(overlay)
            ys, xs = np.unravel_index(np.argmax(heat), heat.shape)
            s.metric("picco attivazione (x,y)", None if flat else [int(xs), int(ys)])
            if flat:
                s.note("Mappa piatta: gradienti nulli (rete saturata o non addestrata), nessuna zona prevale.")
        with ctx.step(aid, "softmax", "Softmax e soglia",
                      "L'uscita a 2 neuroni è normalizzata con softmax: indice 0 = bona fide (reale), "
                      "indice 1 = attacco, come nel repo (bonafide_label=0, attack_label=1). Soglia 0.5.") as s:
            p_real, p_attack = float(probs[0]), float(probs[1])
            s.metric("P(reale)", p_real); s.metric("P(attacco)", p_attack)
            s.image(prob_bar(p_real))
        label = "real" if p_real >= 0.5 else "attack"
        with ctx.step(aid, "verdict", "Verdetto",
                      "Interpretazione finale. Le metriche ISO/IEC 30107-3 (APCER: attacchi accettati come reali; "
                      "BPCER: reali rifiutati) si misurano su un intero dataset, non su una singola immagine: qui "
                      "riportiamo la decisione e la sua confidenza.") as s:
            s.metric("decisione", "VOLTO REALE" if label == "real" else "ATTACCO DI PRESENTAZIONE")
            s.metric("confidenza", round(max(p_real, p_attack), 4))
            if not trained:
                s.note("Verdetto non affidabile: pesi non addestrati.")
            if no_face(face):
                s.note("Nessun volto rilevato: la rete ha analizzato l'intera immagine.")
        peak = "senza un picco definito" if flat else f"con il picco in ({int(xs)}, {int(ys)}) sul volto"
        if trained:
            expl = (f"La rete ha assegnato P(reale) = {p_real:.2f} e P(attacco) = {p_attack:.2f} al volto ritagliato e pre-elaborato "
                    f"(256×256, filtro bilaterale, CLAHE, unsharp mask, gamma). Nitidezza misurata {q['sharpness']:.0f}. "
                    f"Mappa Grad-CAM {peak}: la decisione si basa su quella zona.")
        else:
            expl = (f"Pesi NON addestrati: l'uscita [{p_real:.2f}, {p_attack:.2f}] è casuale e non va interpretata come verdetto. "
                    f"La pipeline del docente è stata eseguita per intero (nitidezza {q['sharpness']:.0f}, contrasto {q['contrast']:.0f}).")
        if no_face(face):
            expl += " Nessun volto rilevato: analisi sull'intera immagine, verdetto da prendere con cautela."
        return Result(aid, label if trained else "unknown", p_real if trained else None,
                      ("Reale" if label == "real" else "Attacco") + (f" ({max(p_real, p_attack):.0%})" if trained else " · rete non addestrata"),
                      "trained" if trained else "untrained",
                      {"p_real": p_real, "p_attack": p_attack, "arch": self.arch, "no_face": no_face(face)}, explanation=expl)


def grad_cam(gm, x, class_idx: int):
    """Grad-CAM (Selvaraju et al., 2017) sui logit: mappa uint8 256 livelli e flag "piatta" se i gradienti sono nulli."""
    T = tf()
    sub, kernel, bias = gm
    with T.GradientTape() as tape:
        conv_out, pre = sub(x)
        logits = T.matmul(pre, kernel) + bias         # uscita dell'ultimo Dense PRIMA della softmax
        target = logits[:, class_idx]
    grads = tape.gradient(target, conv_out)
    pooled = T.reduce_mean(grads, axis=(0, 1, 2))          # peso di ogni canale = media dei suoi gradienti
    cam = T.reduce_sum(conv_out[0] * pooled, axis=-1).numpy()
    if not np.isfinite(cam).all() or cam.std() <= 1e-12:
        return np.zeros(cam.shape, np.uint8), True          # gradienti nulli o non finiti: nessuna informazione
    relu = np.maximum(cam, 0)                               # Grad-CAM classico: solo i contributi a favore della classe
    if relu.max() <= 1e-12:
        # tutti i contributi sono negativi (caso degenere noto): si mostra la mappa relativa, le zone "meno contro" sono le più calde
        relu = cam - cam.min()
    return (relu / relu.max() * 255).astype(np.uint8), False


def prob_bar(p_real: float, w: int = 420, h: int = 90) -> np.ndarray:
    """Barra orizzontale reale/attacco come immagine."""
    img = np.full((h, w, 3), 24, np.uint8)
    split = int(w * p_real)
    cv2.rectangle(img, (0, 20), (split, 70), (120, 220, 90), -1)
    cv2.rectangle(img, (split, 20), (w, 70), (90, 90, 235), -1)
    cv2.putText(img, f"reale {p_real:.0%}", (8, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
    cv2.putText(img, f"attacco {1 - p_real:.0%}", (w - 150, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


class LivenessNetA(KuznetsovCNN):
    id = "livenessnet"; arch = "LivenessNet"; name = "LivenessNet"; order = 10
    short = "CNN baseline del docente: 2 blocchi convoluzionali (16 e 32 filtri), BatchNorm, dropout, dense 64. 8,4 M parametri."
    reference = "Kuznetsov et al., Deep Learning Models for Robust Facial Liveness Detection, MTAP 2026"
    reference_url = "https://arxiv.org/abs/2508.09094"


class AttackNetV1A(KuznetsovCNN):
    id = "attacknet_v1"; arch = "AttackNetV1"; name = "AttackNet V1"; order = 11; color = "#a07cff"
    short = "Connessioni residue per concatenazione, LeakyReLU(0.2), tanh nel dense. 33,6 M parametri."
    reference = "Kuznetsov et al., AttackNet, Computers & Security 2024"
    reference_url = "https://arxiv.org/abs/2402.03769"


class AttackNetV21A(KuznetsovCNN):
    id = "attacknet_v2_1"; arch = "AttackNetV2_1"; name = "AttackNet V2.1"; order = 12; color = "#b57cff"
    short = "Come V1 ma con dropout pieno anche nei blocchi convoluzionali e attivazione tanh separata. 33,6 M parametri."
    reference = "Nurpeisova, ..., Kuznetsov, Deep Residual Learning for Face Anti-Spoofing, Technologies 2025"
    reference_url = "https://www.mdpi.com/2227-7080/13/9/413"


class AttackNetV22A(KuznetsovCNN):
    id = "attacknet_v2_2"; arch = "AttackNetV2_2"; name = "AttackNet V2.2"; order = 13; color = "#c97cff"
    short = "Skip connection per somma invece che concatenazione: metà parametri (16,8 M), miglior generalizzazione cross-dataset nel paper."
    reference = "Nurpeisova, ..., Kuznetsov, Technologies 2025 · Kuznetsov et al., MTAP 2026"
    reference_url = "https://www.mdpi.com/2227-7080/13/9/413"


def _register_all():
    """Per ogni architettura: una voce per ciascun dataset di addestramento disponibile; se non ce n'è nessuno,
    la voce non addestrata (a meno di LIVENESSLAB_SKIP_UNTRAINED=1, usato sul server per risparmiare memoria)."""
    skip_untrained = os.environ.get("LIVENESSLAB_SKIP_UNTRAINED", "0") == "1"
    for cls in (LivenessNetA, AttackNetV1A, AttackNetV21A, AttackNetV22A):
        variants = trained_variants(cls.arch)
        if not variants:
            if not skip_untrained:
                registry.register(cls())
            continue
        for ds in variants:
            inst = cls()
            inst.trained_on = ds
            inst.id = f"{cls.id}__{ds}"
            inst.name = f"{cls.name} · {DATASET_LABELS.get(ds, ds)}"
            base = ds[:-len("-pooled")] if ds.endswith("-pooled") else ds
            # nel catalogo le varianti sono raggruppate per dataset di addestramento (poi architettura, poi split):
            # ordine = 10 + posizione del dataset + architettura/10 + 0.01 se "pooled"
            inst.subgroup = DATASET_LABELS.get(base, base)
            ds_pos = DATASET_ORDER.index(base) if base in DATASET_ORDER else len(DATASET_ORDER)
            inst.order = 10 + ds_pos + (cls.order - 10) / 10 + (0.01 if ds.endswith("-pooled") else 0)
            inst.short = cls.short + f" Addestrata sul dataset {DATASET_LABELS.get(ds, ds)} con lo script della tesi (scripts/train_cnn.py)."
            registry.register(inst)


_register_all()
