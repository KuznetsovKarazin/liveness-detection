"""
Funzioni comuni dell'inventario M2 di CASIA-FASD e CelebA-Spoof (sola lettura: nessun file viene scritto o modificato).
Percorsi relativi alla radice del repository, ricavata dalla posizione di questo file.
"""
import hashlib
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
WEIGHTS = ROOT / "models" / "weights"
# fuso orario fisso per date di download e ore di modifica: i numeri non dipendono dal TZ della macchina
ROME = ZoneInfo("Europe/Rome")
SEED = 42
PER_CLASS = 1500
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

CASIA_ARCHIVE = DATA / "raw" / "casiafasd.tar.gz"
CASIA_HF_META = DATA / "raw" / ".cache" / "huggingface" / "download" / "casiafasd.tar.gz.metadata"
CASIA_RAW = DATA / "raw" / "casia"
CASIA_IMAGES = DATA / "CASIA_FASD" / "images"
CASIA_EVAL = DATA / "eval" / "casia_fasd"
CASIA_CACHE = DATA / "processed" / "casia_fasd_1500.npz"
# cartelle del mirror: split del mirror → sottocartella e tag delle cartelle preparate
CASIA_SPLITS = (("train", "train_img/train_img", "training"), ("test", "test_img/test_img", "validation"))

CELEBA_PARQUET = DATA / "raw" / "celeba_spoof" / "data" / "test-00000-of-00010.parquet"
CELEBA_HF_META = DATA / "raw" / "celeba_spoof" / ".cache" / "huggingface" / "download" / "data" / "test-00000-of-00010.parquet.metadata"
CELEBA_IMAGES = DATA / "CelebA_Spoof" / "images"
CELEBA_EVAL = DATA / "eval" / "celeba_spoof"
CELEBA_CACHE = DATA / "processed" / "celeba_spoof_1500.npz"

FOLDERS = ("bonafide_training", "attack_training", "bonafide_validation", "attack_validation")


def rel(p: Path) -> str:
    """Percorso relativo alla radice del repository (mai assoluto nelle stampe)."""
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def spread(items, k):
    """Copia di scripts/train_cnn.py e scripts/prepare_datasets.py: k elementi a passo costante sull'elenco ordinato."""
    items = sorted(items)
    if k is None or k >= len(items):
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def hf_metadata(p: Path) -> dict:
    """File .metadata della cache huggingface_hub (local_dir): revisione, etag, timestamp del download."""
    lines = p.read_text().splitlines()
    return {"revision": lines[0].strip(), "etag": lines[1].strip(), "timestamp": float(lines[2])}


def list_images(d: Path):
    return [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT]


def strip_private(d):
    """Toglie le chiavi che iniziano con '_' (identificativi per soggetto o per riga)."""
    if isinstance(d, dict):
        return {k: strip_private(v) for k, v in d.items() if not str(k).startswith("_")}
    return d
