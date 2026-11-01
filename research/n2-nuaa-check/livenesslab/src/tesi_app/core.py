"""
Nucleo del motore di analisi.

Ogni *Analyzer* è una pipeline di passi. Ogni passo emette eventi (running / done / error) con descrizione testuale,
immagine intermedia, metriche e tempo impiegato. La UI usa questi eventi per animare il diagramma a blocchi e le card.
Alla fine ogni analizzatore restituisce un `Result` con il verdetto sulla singola immagine.

Convenzione dei punteggi: `Result.score_real` è la probabilità (o uno score in [0,1]) che il volto sia bona fide.
Il verdetto sulla singola immagine è "reale" se score_real >= 0.5. La valutazione su dataset (evaluation.py) usa
1 - score_real come punteggio di attacco, con la stessa regola di pareggio.
"""
from __future__ import annotations

import base64
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np


# ----------------------------------------------------------------------------- immagini

def to_data_url(img: np.ndarray, max_side: int = 420, fmt: str = ".jpg", quality: int = 82) -> str:
    """Converte un array (BGR, gray o float) in data URL JPEG/PNG ridimensionata per la UI.
    Gli array float vengono riscalati in 0–255 sul loro min/max; i crop molto piccoli vengono ingranditi con
    interpolazione nearest così i singoli pixel restano visibili."""
    if img is None:
        return ""
    arr = img
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        arr = np.zeros_like(arr, dtype=np.uint8) if hi - lo < 1e-9 else ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
    h, w = arr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        arr = cv2.resize(arr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    elif max(h, w) < 160:
        k = int(160 / max(h, w)) + 1
        arr = cv2.resize(arr, (w * k, h * k), interpolation=cv2.INTER_NEAREST)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == ".jpg" else []
    ok, buf = cv2.imencode(fmt, arr, params)
    if not ok:
        return ""
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode("ascii")


MAX_UPLOAD_BYTES = 12 * 1024 * 1024      # (frame WebSocket 17 MB in base64) il client invia JPEG ridotti a 1600 px (< 1 MB): 12 MB è già molto generoso
MAX_PIXELS = 40_000_000                  # 40 megapixel: oltre, la decodifica costerebbe gigabyte di memoria
MAX_SIDE = 2000                          # lato massimo dopo la decodifica (gli analizzatori lavorano sul volto ritagliato)


def decode_data_url(data_url: str) -> np.ndarray:
    """Data URL (o base64 puro) -> immagine BGR, con limiti di dimensione perché l'immagine arriva da utenti anonimi:
    payload massimo 12 MB, dimensioni lette dall'intestazione PRIMA di decodificare (contro le "decompression bomb"),
    riduzione a 2000 px di lato. Solleva ValueError se non decodificabile o fuori limite."""
    payload = data_url.split(",", 1)[1] if "," in data_url else data_url
    if len(payload) > MAX_UPLOAD_BYTES * 4 // 3:
        raise ValueError("immagine troppo grande (max 12 MB)")
    try:
        data = base64.b64decode(payload, validate=False)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("codifica non valida") from exc
    try:
        from io import BytesIO
        from PIL import Image
        with Image.open(BytesIO(data)) as im:       # legge solo l'intestazione: dimensioni senza decodificare i pixel
            w, h = im.size
    except Exception as exc:  # noqa: BLE001
        raise ValueError("formato immagine non riconosciuto") from exc
    if w * h > MAX_PIXELS:
        raise ValueError(f"immagine troppo grande ({w}×{h} pixel, max {MAX_PIXELS // 1_000_000} megapixel)")
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Immagine non decodificabile")
    k = MAX_SIDE / max(img.shape[:2])
    if k < 1:
        img = cv2.resize(img, (max(1, int(img.shape[1] * k)), max(1, int(img.shape[0] * k))), interpolation=cv2.INTER_AREA)
    return img


def colorize(gray: np.ndarray, cmap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Mappa a colori di un'immagine a un canale (uint8 o float, riscalata sul min/max)."""
    g = gray
    if g.dtype != np.uint8:
        g = cv2.normalize(g.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(g, cmap)


def tile(images: List[np.ndarray], cols: int, pad: int = 2) -> np.ndarray:
    """Griglia di immagini della stessa dimensione (es. feature map), separate da un bordo scuro."""
    if not images:
        return np.zeros((10, 10, 3), np.uint8)
    imgs = [im if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) for im in images]
    h, w = imgs[0].shape[:2]
    rows = int(np.ceil(len(imgs) / cols))
    canvas = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), 18, np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        canvas[y:y + h, x:x + w] = cv2.resize(im, (w, h))
    return canvas


# ----------------------------------------------------------------------------- eventi

@dataclass
class StepEvent:
    """Un evento di un passo della pipeline, inviato alla UI via WebSocket."""
    analyzer: str
    step: str
    status: str                      # running | done | error
    title: str = ""
    description: str = ""
    image: str = ""                  # data URL dell'immagine intermedia (vuota in modalità quiet)
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "step", **self.__dict__}


@dataclass
class Result:
    """Verdetto finale di un analizzatore su una singola immagine."""
    analyzer: str
    label: str                       # real | attack | unknown
    score_real: Optional[float]      # probabilità/score che il volto sia reale in [0,1]; None se non stimabile
    verdict: str                     # testo breve mostrato nella UI
    reliability: str                 # trained | zeroshot | heuristic | descriptive | untrained | error
    details: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    explanation: str = ""            # spiegazione in italiano del risultato su QUESTA immagine (per il Riepilogo)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "result", **self.__dict__}


EmitFn = Callable[[Dict[str, Any]], None]


class RunCancelled(Exception):
    """Sollevata all'inizio di un passo se l'utente ha annullato l'analisi."""


class Step:
    """Context manager di un passo: emette 'running' all'ingresso e 'done' (o 'error') all'uscita.
    Dentro il blocco l'analizzatore aggiunge immagine, metriche e note con i metodi omonimi."""

    def __init__(self, ctx: "RunContext", analyzer_id: str, step_id: str, title: str, description: str):
        self.ctx, self.analyzer_id, self.step_id, self.title, self.description = ctx, analyzer_id, step_id, title, description
        self._image = ""
        self.metrics: Dict[str, Any] = {}
        self.notes: List[str] = []
        self._t0 = 0.0

    def image(self, img: np.ndarray, **kw) -> None:
        """Immagine intermedia del passo (ignorata in modalità quiet: la valutazione batch non la userebbe)."""
        if self.ctx.quiet:
            return
        self._image = to_data_url(img, **kw)

    def metric(self, key: str, value: Any) -> None:
        """Una misura da mostrare nella card; i float vengono arrotondati a 4 decimali, i NaN diventano None."""
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, np.generic):        # float32, int64, bool_ ...
            value = value.item()
        if isinstance(value, float):
            value = None if not np.isfinite(value) else round(value, 4)
        self.metrics[key] = value

    def note(self, text: str) -> None:
        self.notes.append(text)

    def __enter__(self) -> "Step":
        if self.ctx.cancel_event is not None and self.ctx.cancel_event.is_set():
            raise RunCancelled()
        self._t0 = time.perf_counter()
        self.ctx.emit(StepEvent(self.analyzer_id, self.step_id, "running", self.title, self.description).to_dict())
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = (time.perf_counter() - self._t0) * 1000
        if isinstance(exc, RunCancelled):
            return False                    # l'annullamento risale fino a safe_run senza evento di errore
        if exc is not None:
            self.notes.append(f"Errore: {exc}")
            self.ctx.emit(StepEvent(self.analyzer_id, self.step_id, "error", self.title, self.description,
                                    self._image, self.metrics, self.notes, elapsed).to_dict())
            return False                    # l'eccezione prosegue: sarà safe_run a trasformarla in un Result di errore
        self.ctx.emit(StepEvent(self.analyzer_id, self.step_id, "done", self.title, self.description,
                                self._image, self.metrics, self.notes, elapsed).to_dict())
        return False


class RunContext:
    """Stato condiviso di una singola analisi: immagine, funzione di emissione eventi, evento di annullamento,
    e una cache per risultati riusati da più analizzatori (es. il rilevamento del volto)."""

    def __init__(self, image_bgr: np.ndarray, emit: EmitFn, cancel_event: Optional[threading.Event] = None, quiet: bool = False):
        self.image = image_bgr
        self.emit = emit
        self.cancel_event = cancel_event
        self.quiet = quiet                  # True nella valutazione batch: niente immagini codificate
        self.cache: Dict[str, Any] = {}

    def step(self, analyzer_id: str, step_id: str, title: str, description: str) -> Step:
        return Step(self, analyzer_id, step_id, title, description)

    def cached(self, key: str, fn: Callable[[], Any]) -> Any:
        """Calcola `fn()` una sola volta per analisi (gli analizzatori girano in sequenza su un unico thread)."""
        if key not in self.cache:
            self.cache[key] = fn()
        return self.cache[key]


# ----------------------------------------------------------------------------- analyzer

@dataclass
class NodeSpec:
    """Un nodo del diagramma a blocchi della pipeline."""
    id: str
    label: str
    kind: str = "process"            # input | process | model | decision | output


class Analyzer:
    """Classe base di un analizzatore ("plugin"). Le sottoclassi definiscono `graph()` (nodi e archi del diagramma)
    e `run()` (i passi, con gli stessi id dei nodi), poi si registrano con `registry.register(...)`."""

    id: str = "base"
    name: str = "Analyzer"
    family: str = "classico"         # docente | pretrained | classico
    short: str = ""
    reference: str = ""
    reference_url: str = ""
    color: str = "#7c8cff"
    order: int = 100
    subgroup: str = ""               # sotto-gruppo nel catalogo (per le CNN: il dataset di addestramento)

    def describe(self) -> Dict[str, Any]:
        """Scheda completa per l'API /api/analyzers (catalogo, diagramma, documentazione)."""
        nodes, edges = self.graph()
        return {
            "id": self.id, "name": self.name, "family": self.family, "short": self.short,
            "reference": self.reference, "reference_url": self.reference_url, "color": self.color, "order": self.order,
            "subgroup": self.subgroup,
            "reliability": self.reliability(),
            "graph": {"nodes": [n.__dict__ for n in nodes], "edges": edges},
            "doc": self.doc(),
        }

    def doc(self) -> Dict[str, Any]:
        """Scheda informativa dal modulo docs_it; le varianti "<id>__<dataset>" ereditano quella dell'id base."""
        from .analyzers.docs_it import DOCS
        return DOCS.get(self.id) or DOCS.get(self.id.split("__", 1)[0], {})

    def reliability(self) -> str:
        """Quanto fidarsi del verdetto: trained | zeroshot | heuristic | descriptive | untrained.
        Deve essere economica: viene chiamata per ogni analizzatore a ogni richiesta del catalogo."""
        return "heuristic"

    def fingerprint(self) -> str:
        """Impronta della versione del modello (es. hash dei pesi). Se cambia, la cache dei punteggi su dataset
        viene azzerata per questo analizzatore. Per i metodi senza pesi basta l'id."""
        return self.id

    def excluded_datasets(self) -> List[str]:
        """Dataset su cui NON è lecito valutare questo analizzatore (es. immagini presenti nel suo training)."""
        return []

    def exclusion_note(self, ds_id: str) -> str:
        return "escluso dalla valutazione su questo dataset"

    def graph(self):
        raise NotImplementedError

    def run(self, ctx: RunContext) -> Result:
        raise NotImplementedError

    def safe_run(self, ctx: RunContext) -> Result:
        """Esegue `run` misurando il tempo; un'eccezione diventa un Result con reliability "error"
        (distinto da "untrained": il modello c'è, ma su questa immagine è fallito)."""
        t0 = time.perf_counter()
        try:
            res = self.run(ctx)
            res.elapsed_ms = (time.perf_counter() - t0) * 1000
            return res
        except RunCancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            # al client va solo il tipo dell'errore: il messaggio completo (che può contenere percorsi locali) resta nel log
            return Result(self.id, "unknown", None, f"Analisi interrotta ({type(exc).__name__})", "error",
                          {"error": type(exc).__name__}, (time.perf_counter() - t0) * 1000,
                          explanation=f"L'analisi si è interrotta con un errore ({type(exc).__name__}); i dettagli sono nel log del server.")


class Registry:
    """Elenco degli analizzatori disponibili, ordinati per `order` e nome."""

    def __init__(self):
        self._items: Dict[str, Analyzer] = {}

    def register(self, analyzer: Analyzer) -> None:
        self._items[analyzer.id] = analyzer

    def get(self, aid: str) -> Analyzer:
        return self._items[aid]

    def all(self) -> List[Analyzer]:
        return sorted(self._items.values(), key=lambda a: (a.order, a.name))

    def ids(self) -> List[str]:
        return [a.id for a in self.all()]


registry = Registry()
