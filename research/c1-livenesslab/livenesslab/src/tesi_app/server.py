"""
Server FastAPI: serve la UI statica, espone il catalogo degli analizzatori, le immagini di prova e i dataset,
e un WebSocket che riceve un'immagine (o una richiesta di valutazione) e ritrasmette in streaming gli eventi.

Protocollo WebSocket (messaggi JSON):
  client -> server:  {"type":"analyze","image":<data URL>,"analyzers":[ids]}
                     {"type":"evaluate","dataset":<id>,"analyzers":[ids],"limit":N|null,"force":bool}
                     {"type":"cancel"}
  server -> client:  run_start, analyzer_start, step, result, run_end, error        (analisi di una immagine)
                     eval_start, eval_progress, eval_result, eval_error             (valutazione su dataset)
Ogni evento di un'analisi porta il `run_id` della corsa, così il client scarta gli eventi di una corsa annullata.

Gli analizzatori girano in un unico thread di lavoro (TensorFlow e PyTorch non gradiscono l'esecuzione parallela
da più thread); l'event loop resta libero di servire le richieste HTTP e di inviare gli eventi.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows: TensorFlow e PyTorch portano ciascuno una copia di OpenMP
try:
    import torch  # noqa: F401  (vedi run_app.py: va importato prima di TensorFlow e h5py su Windows)
except Exception:  # noqa: BLE001
    pass

import asyncio
import hmac
import json
import math
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

import numpy as np

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from urllib.parse import urlparse

from .core import RunCancelled, RunContext, decode_data_url, registry, to_data_url
from .paths import ROOT, STATIC
from .version import VERSION
from . import analyzers  # noqa: F401  (l'import registra tutti gli analizzatori)
from . import evaluation
from .usage import log_event
from . import usage_report
from . import export

app = FastAPI(title="LivenessLab", version="0.2.0", docs_url=None, redoc_url=None, openapi_url=None)   # niente console API pubbliche
_executor = ThreadPoolExecutor(max_workers=1)   # un'analisi alla volta
# Limiti contro l'abuso da parte di client anonimi: lavori accodati sul thread unico, connessioni per indirizzo,
# valutazioni pesanti. Modificabili con variabili d'ambiente (vedi deploy/windows/livenesslab-task.ps1).
MAX_PENDING_JOBS = int(os.environ.get("LIVENESSLAB_MAX_PENDING", "4"))
MAX_CONN_PER_IP = int(os.environ.get("LIVENESSLAB_MAX_CONN_PER_IP", "4"))
ALLOW_EVAL = os.environ.get("LIVENESSLAB_ALLOW_EVAL", "1") == "1"       # valutazione su dataset dal browser
ALLOW_FORCE = os.environ.get("LIVENESSLAB_ALLOW_FORCE", "1") == "1"     # ricalcolo forzato (sovrascrive la cache): da spegnere sul server pubblico
MAX_LIMIT = 5000
_pending = {"n": 0}                       # lavori in coda o in esecuzione sul thread di lavoro
_conn_by_ip: dict = {}                    # connessioni WebSocket aperte per indirizzo
SAMPLE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")   # id delle immagini di prova: solo nome file semplice, mai percorsi

# Politica dei contenuti: solo risorse proprie, font Google, immagini in data URL (le immagini intermedie), WebSocket.
# 'unsafe-inline' per gli stili è necessario agli attributi style generati dal frontend; gli script restano solo esterni.
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
       "font-src https://fonts.gstatic.com; img-src 'self' data: blob:; media-src 'self' blob:; "
       "connect-src 'self' ws: wss:; worker-src 'self'; manifest-src 'self'; "
       "frame-ancestors 'none'; base-uri 'self'; form-action 'self'")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Intestazioni di sicurezza su ogni risposta: niente inclusione in frame di altri siti (la pagina usa la webcam),
    niente sniffing dei tipi, referrer non inviato, camera consentita solo alla pagina stessa."""
    resp = await call_next(request)
    resp.headers.setdefault("Content-Security-Policy", CSP)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault("Permissions-Policy", "camera=(self), microphone=(), geolocation=()")
    return resp


def _same_origin(ws: WebSocket) -> bool:
    """Accetta il WebSocket solo dalla pagina stessa (o da client senza Origin, es. script): evita che un altro sito
    apra connessioni e faccia lavorare il server dal browser di un visitatore (cross-site WebSocket hijacking)."""
    origin = ws.headers.get("origin")
    if not origin:
        return True
    host = (ws.headers.get("x-forwarded-host") or ws.headers.get("host") or "").split(",")[0].strip().split(":")[0].lower()
    ohost = (urlparse(origin).hostname or "").lower()
    return ohost == host or ohost in ("localhost", "127.0.0.1")


def _json_safe(obj: Any) -> Any:
    """Rende serializzabile un evento: NaN e infiniti diventano None (`json.dumps` li scriverebbe come token non
    validi che `JSON.parse` nel browser rifiuta), gli scalari e gli array numpy diventano tipi Python."""
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def dumps(obj: Any) -> str:
    return json.dumps(_json_safe(obj), ensure_ascii=False)


def _select_ids(requested) -> List[str]:
    """Filtra gli id richiesti dal client su quelli registrati, mantenendo l'ordine del catalogo.
    `None` (campo assente) = tutti; una lista vuota = nessuno; valori non stringa vengono ignorati."""
    known = registry.ids()
    if requested is None:
        return list(known)
    if not isinstance(requested, list):
        return []
    wanted = {i for i in requested if isinstance(i, str)}
    return [i for i in known if i in wanted]


# ----------------------------------------------------------------------------- HTTP
# Gli handler sono funzioni sincrone: FastAPI le esegue in un thread pool, così il lavoro (lettura file, calcolo
# delle metriche) non blocca l'event loop che sta inviando gli eventi WebSocket.

@app.get("/")
def index():
    """Pagina principale, con cache-busting: la versione di CSS e JS è l'mtime del file; {{VERSION}} è la versione dell'app."""
    html = (STATIC / "index.html").read_text(encoding="utf-8").replace("{{VERSION}}", VERSION)
    for name in ("style.css", "app.js"):
        html = html.replace(f"/static/{name}", f"/static/{name}?v={int((STATIC / name).stat().st_mtime)}")
    return HTMLResponse(html)


# File della PWA serviti dalla radice: il service worker deve stare alla radice per governare tutto il sito,
# e va sempre richiesto al server (mai dalla cache del browser) perché è lui a decidere quando aggiornare.
@app.get("/sw.js")
def service_worker():
    js = (STATIC / "sw.js").read_text(encoding="utf-8").replace("{{VERSION}}", VERSION)
    return Response(js, media_type="application/javascript",
                    headers={"Cache-Control": "no-cache", "Service-Worker-Allowed": "/"})


@app.get("/manifest.webmanifest")
def manifest():
    return Response((STATIC / "manifest.webmanifest").read_bytes(), media_type="application/manifest+json",
                    headers={"Cache-Control": "no-cache"})


@app.get("/favicon.ico")
def favicon():
    return Response((STATIC / "icons" / "icon-192.png").read_bytes(), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


# Endpoint riservato al gestore: rapporto degli utilizzi (log IIS + logs/usage.jsonl). Attivo solo se è impostato
# LIVENESSLAB_ADMIN_TOKEN; il token viaggia nell'intestazione X-Admin-Token (mai nell'URL, che finisce nei log).
ADMIN_TOKEN = os.environ.get("LIVENESSLAB_ADMIN_TOKEN", "")


@app.get("/api/usage/report")
def usage_report_endpoint(request: Request, days: int = 1, all: int = 0, me: str = ""):
    token = request.headers.get("x-admin-token", "")
    if not ADMIN_TOKEN or not hmac.compare_digest(token, ADMIN_TOKEN):
        return Response("Not found", status_code=404)
    mine = usage_report.MY_IPS | {x.strip() for x in me.split(",") if x.strip()}
    try:
        text = usage_report.report(days=max(0, min(int(days), 3650)), show_all=bool(all), mine=mine)
    except FileNotFoundError as exc:
        return Response(f"Rapporto non disponibile: {exc}", status_code=503, media_type="text/plain; charset=utf-8")
    return Response(text, media_type="text/plain; charset=utf-8", headers={"Cache-Control": "no-store"})


@app.get("/api/version")
def version():
    return {"version": VERSION}


@app.get("/api/analyzers")
def list_analyzers():
    """Catalogo degli analizzatori con diagramma e scheda informativa."""
    return [a.describe() for a in registry.all()]


_heavy = threading.BoundedSemaphore(2)     # al massimo due richieste pesanti (riepiloghi, export) in parallelo: oltre, 503


def _busy_response():
    return JSONResponse({"error": "Server occupato, riprova tra qualche secondo."}, status_code=503, headers={"Retry-After": "3"})


@app.get("/api/datasets")
def datasets():
    """Dataset disponibili per la valutazione e riepiloghi delle valutazioni già in cache (memorizzati: si ricalcolano
    solo quando cambia una cache o un modello)."""
    if not _heavy.acquire(blocking=False):
        return _busy_response()
    try:
        return JSONResponse(_json_safe({"datasets": evaluation.describe_datasets(), "results": evaluation.all_summaries()}),
                            headers={"Cache-Control": "no-cache"})   # memorizzato lato server; il browser deve sempre richiederlo
    finally:
        _heavy.release()


@app.get("/api/eval/export.xlsx")
def export_xlsx():
    """Tutte le valutazioni in cache in un file Excel (un foglio per dataset, confronto, punti ROC, info)."""
    if not _heavy.acquire(blocking=False):
        return _busy_response()
    try:
        data = export.build_workbook()
    finally:
        _heavy.release()
    name = f"livenesslab-valutazione-{time.strftime('%Y%m%d-%H%M')}.xlsx"
    return Response(content=data, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="{name}"', "Cache-Control": "no-store"})


@app.get("/api/glossary")
def glossary():
    from .analyzers.docs_it import GLOSSARIO
    return [{"termine": k, "definizione": v} for k, v in GLOSSARIO]


_samples_cache: dict = {}


@app.get("/api/samples")
def list_samples():
    """Immagini di prova: cartella samples/ con manifest.json (etichetta, fonte, licenza). L'elenco con le miniature è
    memorizzato e rifatto solo se cambiano i file (ricodificare le miniature a ogni richiesta costava 0,2 s di CPU)."""
    sdir = ROOT / "samples"
    mf = sdir / "manifest.json"
    files = sorted(sdir.glob("*.jpg")) + sorted(sdir.glob("*.png"))
    key = repr([(p.name, p.stat().st_mtime_ns) for p in files + ([mf] if mf.exists() else [])])
    if _samples_cache.get("key") == key:
        return _samples_cache["out"]
    manifest = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}
    out = []
    for p in sorted(sdir.glob("*.jpg")) + sorted(sdir.glob("*.png")):
        img = evaluation.imread(str(p))
        if img is None:
            continue
        m = manifest.get(p.name, {})
        out.append({"id": p.stem, "label": m.get("label", "reale" if "reale" in p.stem else "attacco"),
                    "thumb": to_data_url(img, max_side=160), "source": m.get("source", ""), "license": m.get("license", ""),
                    "note": m.get("note", "")})
    _samples_cache.clear(); _samples_cache.update(key=key, out=out)
    return out


@app.get("/api/samples/{sid}")
def get_sample(sid: str):
    """Immagine di prova a piena risoluzione (max 1600 px) da caricare nell'app."""
    if not SAMPLE_ID.match(sid):
        return JSONResponse({"error": "not found"}, 404)
    for ext in (".jpg", ".png"):
        p = ROOT / "samples" / f"{sid}{ext}"
        img = evaluation.imread(str(p)) if p.exists() else None
        if img is not None:
            return {"id": sid, "image": to_data_url(img, max_side=1600, quality=92)}
    return JSONResponse({"error": "not found"}, 404)


# ----------------------------------------------------------------------------- analisi (thread di lavoro)

def _run_all(image, analyzer_ids: List[str], emit, cancel: threading.Event) -> None:
    """Esegue in sequenza gli analizzatori richiesti sulla stessa immagine (contesto condiviso)."""
    ctx = RunContext(image, emit, cancel)
    for aid in analyzer_ids:
        if cancel.is_set():
            break
        a = registry.get(aid)
        emit({"type": "analyzer_start", "analyzer": aid, "ts": time.time()})
        try:
            res = a.safe_run(ctx)
        except RunCancelled:
            break
        emit(res.to_dict())
        try:   # traccia di memoria nel log: utile per capire eventuali crash del processo sul server
            import psutil
            print(f"[mem] dopo {aid}: RSS {psutil.Process().memory_info().rss / 1e6:.0f} MB, libera {psutil.virtual_memory().available / 1e6:.0f} MB", flush=True)
        except Exception:  # noqa: BLE001
            pass


# ----------------------------------------------------------------------------- WebSocket

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    if not _same_origin(ws):
        await ws.close(code=1008)
        return
    ip = ws.client.host if ws.client else "?"
    if _conn_by_ip.get(ip, 0) >= MAX_CONN_PER_IP:
        await ws.close(code=1013)                 # "try again later": troppe connessioni dallo stesso indirizzo
        return
    _conn_by_ip[ip] = _conn_by_ip.get(ip, 0) + 1
    try:
        await ws.accept()
    except Exception:
        _conn_by_ip[ip] = max(0, _conn_by_ip.get(ip, 1) - 1)      # l'handshake è fallito: il posto va restituito subito
        raise
    loop = asyncio.get_running_loop()
    current = {"task": None, "cancel": None}     # la corsa in corso su questa connessione (analisi o valutazione)
    user_agent = ws.headers.get("user-agent", "")
    log_event("connect", ip, user_agent)

    async def stream(worker, *args, on_error, on_end=None) -> None:
        """Lancia `worker(*args, emit, cancel)` nel thread di lavoro e inoltra al client gli eventi che emette.
        `emit` è chiamata dal thread di lavoro: mette l'evento in una coda asyncio in modo thread-safe.
        Un'eccezione del worker produce l'evento `on_error` invece di lasciare il client in attesa."""
        cancel = current["cancel"]
        queue: asyncio.Queue = asyncio.Queue()

        def emit(ev):
            loop.call_soon_threadsafe(queue.put_nowait, ev)

        TERMINAL = ("eval_result", "eval_error")     # arrivano sempre, anche dopo uno Stop: chiudono lo stato della UI

        async def drain():
            while True:
                ev = await queue.get()
                if ev is None:
                    break
                if cancel.is_set() and ev.get("type") not in TERMINAL:
                    continue                          # eventi di una corsa annullata: il client non li vuole più
                try:
                    await ws.send_text(dumps(ev))
                except Exception as exc:  # noqa: BLE001  (evento non serializzabile o socket chiuso: non si ferma il resto)
                    print(f"[ws] evento non inviato ({ev.get('type')}): {exc}", flush=True)

        drainer = asyncio.create_task(drain())
        error = None
        try:
            await loop.run_in_executor(_executor, worker, *args, emit, cancel)
        except Exception as exc:  # noqa: BLE001
            import traceback; traceback.print_exc()
            error = exc
        finally:
            emit(None)
            try:
                await drainer
            except Exception:  # noqa: BLE001  (connessione chiusa mentre si inviava)
                pass
        try:
            if error is not None:
                await ws.send_text(dumps(on_error(error)))
            elif on_end is not None:
                await ws.send_text(dumps(on_end(cancel.is_set())))
        except Exception:  # noqa: BLE001
            pass

    async def run(image, ids: List[str]) -> None:
        run_id = uuid.uuid4().hex[:8]

        def emit_tagged(ev, _emit):
            if ev is not None:
                ev["run_id"] = run_id
            _emit(ev)

        preview = await loop.run_in_executor(None, to_data_url, image, 900)   # codifica JPEG fuori dall'event loop
        await ws.send_text(dumps({"type": "run_start", "run_id": run_id, "analyzers": ids, "image": preview}))
        await stream(lambda img, ids_, emit, cancel: _run_all(img, ids_, lambda ev: emit_tagged(ev, emit), cancel), image, ids,
                     on_error=lambda exc: {"type": "error", "run_id": run_id, "message": "Errore interno del server durante l'analisi (dettagli nel log)."},
                     on_end=lambda cancelled: {"type": "run_end", "run_id": run_id, "cancelled": cancelled})

    async def run_eval(ds_id: str, ids: List[str], limit, force: bool) -> None:
        def worker(ds, ids_, limit_, force_, emit, cancel):   # riordina gli argomenti per run_evaluation(ds, ids, emit, cancel, limit, force)
            evaluation.run_evaluation(ds, ids_, emit, cancel, limit_, force_)
        await stream(worker, ds_id, ids, limit, force,
                     on_error=lambda exc: {"type": "eval_error", "message": "Errore interno del server durante la valutazione (dettagli nel log)."})

    async def start(coro_fn, *args) -> bool:
        """Annulla e attende l'eventuale corsa precedente, poi avvia la nuova con il proprio evento di annullamento.
        Rifiuta (False) se la coda del thread di lavoro è già piena: il client riceve un errore invece di attendere a vuoto."""
        if current["task"] is not None and not current["task"].done():
            current["cancel"].set()
            try:
                await current["task"]
            except Exception:  # noqa: BLE001
                pass
        if _pending["n"] >= MAX_PENDING_JOBS:
            return False
        _pending["n"] += 1                     # contato qui, prima che altre connessioni passino il controllo
        current["cancel"] = threading.Event()
        current["task"] = asyncio.create_task(coro_fn(*args))
        # rilasciato quando il task finisce, comunque finisca: anche se il client chiude prima che il lavoro entri in
        # `stream` (l'anteprima o l'invio di run_start sollevano), il posto in coda torna libero
        def _done(t):
            _pending["n"] = max(0, _pending["n"] - 1)
            if not t.cancelled():
                t.exception()                  # consumata: una chiusura del client prima di run_start non deve riempire il log
        current["task"].add_done_callback(_done)
        return True

    async def busy(kind: str) -> None:
        await ws.send_text(dumps({"type": kind, "message": "Server occupato: altre analisi sono in coda, riprova tra qualche istante."}))

    try:
        while True:
            frame = await ws.receive()
            if frame.get("type") == "websocket.disconnect":
                break
            raw = frame.get("text")
            try:
                if raw is None:
                    raise ValueError("frame non testuale")
                msg = json.loads(raw)
                if not isinstance(msg, dict):
                    raise ValueError("messaggio non è un oggetto")
            except Exception:  # noqa: BLE001
                await ws.send_text(dumps({"type": "error", "message": "Messaggio non valido."}))
                continue
            mtype = msg.get("type")
            if mtype == "cancel":
                if current["cancel"] is not None:
                    current["cancel"].set()
            elif mtype == "evaluate":
                if not ALLOW_EVAL:
                    await ws.send_text(dumps({"type": "eval_error", "message": "La valutazione su dataset non è abilitata su questo server."}))
                    continue
                limit = msg.get("limit")
                try:
                    limit = max(1, min(int(limit), MAX_LIMIT)) if limit else None
                except (TypeError, ValueError, OverflowError):
                    limit = None
                force = bool(msg.get("force")) and ALLOW_FORCE
                ids = _select_ids(msg.get("analyzers"))
                if not ids:
                    await ws.send_text(dumps({"type": "eval_error", "message": "Nessun analizzatore valido selezionato."}))
                    continue
                dataset = str(msg.get("dataset", "samples"))[:64]
                log_event("evaluate", ip, user_agent, dataset=dataset, analyzers=len(ids), limit=limit, force=force)
                if not await start(run_eval, dataset, ids, limit, force):
                    await busy("eval_error")
            elif mtype == "analyze":
                try:
                    # decodifica (immagine fino a 12 MB; il frame WebSocket in base64 arriva a 17 MB) nel pool di thread di default
                    image = await loop.run_in_executor(None, decode_data_url, str(msg.get("image", "")))
                except Exception as exc:  # noqa: BLE001
                    await ws.send_text(dumps({"type": "error", "message": f"Immagine non valida: {exc}"}))   # messaggi di decode_data_url, senza percorsi
                    continue
                ids = _select_ids(msg.get("analyzers"))
                if not ids:
                    await ws.send_text(dumps({"type": "error", "message": "Nessun analizzatore valido selezionato."}))
                    continue
                log_event("analyze", ip, user_agent, analyzers=len(ids), width=int(image.shape[1]), height=int(image.shape[0]))
                if not await start(run, image, ids):
                    await busy("error")
    except WebSocketDisconnect:
        pass
    finally:
        # il client se n'è andato: si ferma la corsa al primo passo utile, senza attendere
        if current["cancel"] is not None:
            current["cancel"].set()
        _conn_by_ip[ip] = _conn_by_ip.get(ip, 1) - 1
        if _conn_by_ip[ip] <= 0:
            _conn_by_ip.pop(ip, None)


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
