"""Percorsi del progetto e caricamento di moduli esterni per file.

Tutto è relativo alla cartella del repository (ROOT), così l'app gira sia su macOS sia su Windows senza
configurazione. I dataset di addestramento e i pesi restano fuori dal controllo di versione. Ogni cartella si può
spostare con una variabile d'ambiente (come richiesto dal flusso di lavoro del gruppo di ricerca: i percorsi dei dati
non stanno nel codice), utile quando il codice vive in un altro repository:
  LIVENESSLAB_DATA_DIR      dataset preparati (default ROOT/data; contiene eval/<dataset>/...)
  LIVENESSLAB_RESULTS_DIR   cache dei punteggi e tabelle (default ROOT/results)
  LIVENESSLAB_WEIGHTS_DIR   pesi e classificatori addestrati (default ROOT/models/weights)
  LIVENESSLAB_LIVEDETECTION cartella del codice di riferimento delle CNN, con src/architectures.py (default ROOT/src/livedetection)
  LIVENESSLAB_SILENT_FACE   cartella di Silent-Face-Anti-Spoofing (default ROOT/src/third_party/Silent-Face-Anti-Spoofing)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]            # cartella del repository (Tesi/)
SRC = ROOT / "src"


def _env_dir(name: str, default: Path) -> Path:
    v = os.environ.get(name, "").strip()
    return Path(v).expanduser().resolve() if v else default


LIVEDETECTION = _env_dir("LIVENESSLAB_LIVEDETECTION", SRC / "livedetection")                 # codice di riferimento delle CNN (submodule, non modificato)
SILENT_FACE = _env_dir("LIVENESSLAB_SILENT_FACE", SRC / "third_party" / "Silent-Face-Anti-Spoofing")   # MiniFASNet + RetinaFace (submodule, Apache-2.0)
WEIGHTS = _env_dir("LIVENESSLAB_WEIGHTS_DIR", ROOT / "models" / "weights")                 # pesi addestrati (fuori da git)
DATA = _env_dir("LIVENESSLAB_DATA_DIR", ROOT / "data")                                     # dataset preparati (fuori da git)
RESULTS = _env_dir("LIVENESSLAB_RESULTS_DIR", ROOT / "results")                            # cache dei punteggi, tabelle (fuori da git)
STATIC = SRC / "tesi_app" / "static"                  # frontend


_sha_cache = {}


def file_sha256(path) -> str:
    """SHA-256 di un file (es. un checkpoint), calcolato una volta per (percorso, dimensione, mtime).
    È l'impronta usata per invalidare la cache dei punteggi e da riportare nei rapporti di riproducibilità."""
    import hashlib
    p = Path(path)
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns)
    if key not in _sha_cache:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _sha_cache[key] = h.hexdigest()
    return _sha_cache[key]


def load_module(name: str, path):
    """Importa un singolo file .py con un nome univoco scelto da noi.
    Serve perché sia il repo del docente sia Silent-Face usano un pacchetto chiamato `src`: importandoli nel modo
    normale i due nomi entrerebbero in conflitto tra loro e con il nostro `src/`."""
    import importlib.util
    import sys
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
