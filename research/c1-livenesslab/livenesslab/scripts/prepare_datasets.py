"""
Prepara i dataset pubblici scaricati in data/raw/ (NUAA e CASIA-FASD dai mirror Hugging Face, SynthASpoof dal Google
Drive ufficiale degli autori: data/raw/synthaspoof/SynthASpoof/{BonaFide,PAs/...}) in due forme:

1. data/<Nome>/images/{bonafide_training,attack_training,bonafide_validation,attack_validation}/
   = la struttura attesa da scripts/create_datasets.py del docente (per il training delle sue CNN).
   "training" = split ufficiale di train, "validation" = split ufficiale di TEST (per NUAA sessioni diverse ma soggetti
   in parte in comune; per CASIA-FASD e SynthASpoof soggetti/identità diversi).
2. data/eval/<id>/{real,attack}/  = sottoinsieme bilanciato del TEST set per la tab "Valutazione" dell'app,
   preso a passo costante sull'elenco ordinato (copre tutti i soggetti).

CelebA-Spoof non è gestito qui: è stato convertito una volta dallo shard 0 del mirror Hugging Face (parquet, via
pyarrow) come descritto in docs/ragionamenti/2026-09-12-09-valutazione-dataset.md.

I file vengono collegati con hard link (o copiati se il file system non lo permette): nessuna duplicazione dei dati.
Uso:  python scripts/prepare_datasets.py [--per-class 150] [--only nuaa casia synthaspoof]
"""
import argparse
import sys
import json
import re
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tesi_app.paths import DATA  # noqa: E402
RAW = DATA / "raw"


def extract(tgz: Path, dest: Path) -> Path:
    """Estrae l'archivio se la cartella di destinazione non esiste (cancellarla se un'estrazione è stata interrotta)."""
    if not dest.exists():
        print("estraggo", tgz.name, "→", dest)
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tgz) as tf:
            tf.extractall(dest)
    return dest


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        try:
            dst.hardlink_to(src)
        except Exception:  # noqa: BLE001  (file system diverso o senza hard link)
            shutil.copy2(src, dst)


def spread(items, k):
    """k elementi distribuiti uniformemente sull'elenco ordinato (così coprono più soggetti/video)."""
    items = sorted(items)
    if k >= len(items):
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def prepare_nuaa(per_class: int):
    """NUAA Imposter: le liste ufficiali client/imposter × train/test definiscono gli split."""
    base = extract(RAW / "nuaaaa.tar.gz", RAW / "nuaa") / "raw"

    def read_list(name):
        p = base / name
        # i file di split del mirror contengono percorsi assoluti di Kaggle: teniamo solo la parte dopo ClientRaw/ o ImposterRaw/
        rels = []
        for l in p.read_text(errors="ignore").splitlines():
            if not l.strip():
                continue
            rel = re.split(r"(?:ClientRaw|ImposterRaw)[\\/]", l.split()[0].replace("\\", "/"))[-1]
            rels.append(rel)
        return rels

    splits = {"train": (read_list("client_train_raw.txt"), read_list("imposter_train_raw.txt")),
              "test": (read_list("client_test_raw.txt"), read_list("imposter_test_raw.txt"))}
    out = DATA / "NUAA" / "images"
    for split, (clients, imposters) in splits.items():
        tag = "training" if split == "train" else "validation"
        for rel in clients:
            link_or_copy(base / "ClientRaw" / rel, out / f"bonafide_{tag}" / rel.replace("/", "_"))
        for rel in imposters:
            link_or_copy(base / "ImposterRaw" / rel, out / f"attack_{tag}" / rel.replace("/", "_"))
    ev = DATA / "eval" / "nuaa"
    for cls, rels, sub in (("real", splits["test"][0], "ClientRaw"), ("attack", splits["test"][1], "ImposterRaw")):
        for rel in spread(rels, per_class):
            link_or_copy(base / sub / rel, ev / cls / rel.replace("/", "_"))
    (ev / "info.json").write_text(json.dumps({
        "name": "NUAA Imposter (test)", "source": "Tan, Li, Liu, Jiang — NUAA Photograph Imposter Database, ECCV 2010; mirror HF akahana/anti-spoofing-nuaaaa",
        "license": "uso accademico (mirror non ufficiale)", "note": f"Attacchi con foto stampata (A4) ripresi da webcam. Sottoinsieme del test set ufficiale (3.362 bona fide, 5.761 attacchi): {per_class} reali + {per_class} attacchi presi a passo costante sugli elenchi ufficiali ordinati; i reali coprono 9 soggetti e gli attacchi 15 (la composizione del test set ufficiale)."}, ensure_ascii=False, indent=1), encoding="utf-8")
    n = {k: len(list((out / k).glob("*"))) for k in ("bonafide_training", "attack_training", "bonafide_validation", "attack_validation")}
    print("NUAA:", n, "| eval:", len(list((ev / "real").glob("*"))), "reali,", len(list((ev / "attack").glob("*"))), "attacchi")


def prepare_casia(per_class: int):
    """CASIA-FASD (frame estratti dai video dal mirror): cartelle train_img/test_img, suffisso _real/_fake nel nome."""
    base = extract(RAW / "casiafasd.tar.gz", RAW / "casia")
    out = DATA / "CASIA_FASD" / "images"
    ev = DATA / "eval" / "casia_fasd"
    test_pool = {"real": [], "attack": []}
    for split, sub in (("training", "train_img/train_img/color"), ("validation", "test_img/test_img/color")):
        for p in sorted((base / sub).glob("*.jpg")):
            m = re.search(r"_(real|fake)\.jpg$", p.name)
            if not m:
                continue
            cls = "bonafide" if m.group(1) == "real" else "attack"
            link_or_copy(p, out / f"{cls}_{split}" / p.name)
            if split == "validation":
                test_pool["real" if cls == "bonafide" else "attack"].append(p)
    for cls, pool in test_pool.items():
        for p in spread(pool, per_class):
            link_or_copy(p, ev / cls / p.name)
    (ev / "info.json").write_text(json.dumps({
        "name": "CASIA-FASD (test, frame)", "source": "Zhang et al. — A face antispoofing database with diverse attacks, ICB 2012; mirror HF akahana/anti-spoofing-casiafasd",
        "license": "uso accademico (mirror non ufficiale)", "note": f"Frame estratti dai video: attacchi con foto stampata, foto ritagliata (cut) e replay video; 3 qualità di camera; qui {per_class} reali + {per_class} attacchi dal test set. I frame di uno stesso video non sono campioni indipendenti."}, ensure_ascii=False, indent=1), encoding="utf-8")
    n = {k: len(list((out / k).glob("*"))) for k in ("bonafide_training", "attack_training", "bonafide_validation", "attack_validation")}
    print("CASIA-FASD:", n, "| eval:", len(list((ev / "real").glob("*"))), "reali,", len(list((ev / "attack").glob("*"))), "attacchi")


def prepare_synthaspoof(per_class: int):
    """SynthASpoof (Fang, Huber, Damer — CVPR Workshops 2023): 25.000 bona fide sintetici (StyleGAN2-ADA) e 78.800 attacchi
    ottenuti stampando o mostrando su schermo quegli stessi volti e rifotografandoli (stampa; replay da webcam, iPad, Samsung).
    Lo stesso nome di file nelle cinque cartelle indica la stessa identità sintetica. Non esiste uno split ufficiale (gli
    autori lo usano solo come training): qui split deterministico PER IDENTITÀ, una su dieci in "validation" (= test), così
    gli attacchi restano dalla stessa parte del bona fide da cui derivano. Gli attacchi prendono il prefisso del tipo."""
    base = RAW / "synthaspoof" / "SynthASpoof"
    bona = sorted((base / "BonaFide").glob("img*.png"))
    if not bona:
        print("SynthASpoof: cartella data/raw/synthaspoof/SynthASpoof/BonaFide non trovata, salto"); return
    test_ids = {p.stem for p in bona[9::10]}
    attack_dirs = {"print": "PAs/PrintAttack", "replay_webcam": "PAs/Webcam_ReplayAttack",
                   "replay_ipad": "PAs/iPad_ReplayAttack", "replay_samsung": "PAs/Samsung_ReplayAttack"}
    out = DATA / "SynthASpoof" / "images"
    ev = DATA / "eval" / "synthaspoof"
    for p in bona:
        link_or_copy(p, out / f"bonafide_{'validation' if p.stem in test_ids else 'training'}" / p.name)
    pools = {}
    for tag, sub in attack_dirs.items():
        pools[tag] = []
        for p in sorted((base / sub).glob("img*.png")):
            split = "validation" if p.stem in test_ids else "training"
            link_or_copy(p, out / f"attack_{split}" / f"{tag}_{p.name}")
            if split == "validation":
                pools[tag].append(p)
    # scheda Valutazione: bona fide dalla parte di validazione, attacchi bilanciati tra i quattro tipi
    for p in spread([p for p in bona if p.stem in test_ids], per_class):
        link_or_copy(p, ev / "real" / p.name)
    k, extra = divmod(per_class, len(pools))
    for i, (tag, pool) in enumerate(pools.items()):
        for p in spread(pool, k + (1 if i < extra else 0)):
            link_or_copy(p, ev / "attack" / f"{tag}_{p.name}")
    (ev / "info.json").write_text(json.dumps({
        "name": "SynthASpoof (sintetico)",
        "source": "Fang, Huber, Damer — SynthASpoof: Developing Face Presentation Attack Detection Based on Privacy-friendly Synthetic Data, CVPR Workshops 2023; Google Drive ufficiale degli autori (github.com/meilfang/SynthASpoof)",
        "license": "CC BY-NC-SA 4.0 (solo ricerca)",
        "note": f"Bona fide sintetici (StyleGAN2-ADA, nessun volto reale) e attacchi che li rifotografano: stampa (3.800) e replay da webcam, iPad e Samsung (25.000 ciascuno). Gli autori lo indicano come dati di addestramento, non di test. Split per identità sintetica (una su dieci in test); qui {per_class} bona fide + {per_class} attacchi (circa {k} per tipo) dalla parte di test."},
        ensure_ascii=False, indent=1), encoding="utf-8")
    n = {k2: len(list((out / k2).glob("*"))) for k2 in ("bonafide_training", "attack_training", "bonafide_validation", "attack_validation")}
    print("SynthASpoof:", n, "| eval:", len(list((ev / "real").glob("*"))), "reali,", len(list((ev / "attack").glob("*"))), "attacchi")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--only", nargs="+", choices=["nuaa", "casia", "synthaspoof"], default=["nuaa", "casia", "synthaspoof"])
    a = ap.parse_args()
    if "nuaa" in a.only: prepare_nuaa(a.per_class)
    if "casia" in a.only: prepare_casia(a.per_class)
    if "synthaspoof" in a.only: prepare_synthaspoof(a.per_class)
