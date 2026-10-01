"""Registro degli utilizzi: una riga JSON per evento in logs/usage.jsonl (connessione WebSocket, analisi, valutazione),
con data e ora, indirizzo del client, browser e pochi dati sull'operazione. Serve a capire chi usa il servizio
(scripts/usage_report.py lo incrocia con il log di IIS). Nessun contenuto: né immagini né punteggi."""
import json
import threading
import time
from pathlib import Path

from .paths import ROOT

USAGE_LOG = ROOT / "logs" / "usage.jsonl"
_lock = threading.Lock()


def log_event(kind: str, ip: str, user_agent: str = "", **extra) -> None:
    """Aggiunge un evento; un errore di scrittura non deve mai interrompere il servizio."""
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "ip": ip, "ua": (user_agent or "")[:200], **extra}
    try:
        with _lock:
            USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
            with USAGE_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass
