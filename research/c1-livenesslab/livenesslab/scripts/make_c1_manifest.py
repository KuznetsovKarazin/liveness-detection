"""
Manifest e configurazione della tabella di riferimento C1 (NUAA, 150 bona fide + 150 attacchi):
  results/c1/nuaa_manifest.csv   una riga per immagine: file, label (0 bona fide / 1 attacco), subject e session (primo e terzo
                                 campo del nome file originale NUAA), sha256
  results/c1/nuaa_config.json    soglia e convenzioni, dataset e regola di campionamento, analizzatori con impronta,
                                 checkpoint con SHA-256 e protocollo, commit del repository, hash del manifest
Le immagini restano fuori dal repository pubblico: il manifest con gli hash permette a chi ha il dataset di verificare
di avere esattamente le stesse 300 immagini. Uso:  python scripts/make_c1_manifest.py [--dataset nuaa]
"""
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    import torch  # noqa: F401
except Exception:  # noqa: BLE001
    pass
import tesi_app.analyzers  # noqa: E402,F401
from tesi_app.core import registry  # noqa: E402
from tesi_app import evaluation as ev  # noqa: E402
from tesi_app.paths import DATA, RESULTS, WEIGHTS  # noqa: E402

# NUAA: <soggetto>_<soggetto>_<sessione>_<...>.jpg (es. 0004_0004_01_06_03_118.jpg)
NUAA_NAME = re.compile(r"^(\d{4})_\d{4}_(\d{2})_")
NAME_EN = {"(split 80/20 del docente)": "(80/20 pooled split of the reference code)", "MiniFASNet (pre-addestrato)": "MiniFASNet (pre-trained)",
           "Profondità (Depth Anything)": "Depth (Depth Anything, estimated from RGB)", "Spettro di Fourier": "Fourier spectrum", "Distorsioni IDA": "IDA distortions",
           "Qualità d'immagine (IQA)": "Image quality (IQA)", "CLIP zero-shot (FLIP)": "CLIP zero-shot", "LBP micro-texture": "LBP micro-texture"}


FAMILY_EN = {"docente": "reference_cnn", "pretrained": "pretrained", "classico": "classic"}   # id delle famiglie dell'app -> etichette del pacchetto


def name_en(name: str) -> str:
    """Etichetta inglese dell'analizzatore per il pacchetto (le etichette dell'app sono in italiano)."""
    for it, en in NAME_EN.items():
        name = name.replace(it, en)
    return name


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="nuaa"); ap.add_argument("--out", default=str(RESULTS / "c1"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ds = next(d for d in ev.list_datasets() if d["id"] == a.dataset)
    root = DATA / "eval" / a.dataset
    info = json.loads((root / "info.json").read_text(encoding="utf-8")) if (root / "info.json").exists() else {}
    if a.dataset == "nuaa":   # testi in inglese per il pacchetto (info.json resta in italiano per l'app)
        info = {"name": "NUAA Imposter (official test set)",
                "source": "Tan, Li, Liu, Jiang, NUAA Photograph Imposter Database, ECCV 2010; public Hugging Face mirror akahana/anti-spoofing-nuaaaa",
                "license": "academic use; unofficial mirror of a public dataset",
                "note": "printed A4 photographs re-captured by webcam; subset of the official test set (3,362 bona fide, 5,761 attacks): 150 bona fide and 150 attacks taken at a constant step over the sorted official lists; the bona fide cover 9 subjects and the attacks 15 (the composition of the official test set)"}
    # 1) manifest
    rows = []
    for path, label in sorted(ds["items"], key=lambda x: (x[1], str(x[0]))):
        p = Path(path); m = NUAA_NAME.match(p.name)
        rows.append({"file": ev.key_of(path), "label": int(label), "subject": m.group(1) if m else "", "session": m.group(2) if m else "",
                     "sha256": sha256(p), "bytes": p.stat().st_size})
    man = out / f"{a.dataset}_manifest.csv"
    with man.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "subject", "session", "sha256", "bytes"], lineterminator="\n"); w.writeheader(); w.writerows(rows)
    n_real = sum(1 for r in rows if r["label"] == 0); n_attack = len(rows) - n_real
    # 2) configurazione
    analyzers = []
    for an in sorted(registry.all(), key=lambda x: (x.order, x.id)):
        entry = {"id": an.id, "name": name_en(an.name), "family": FAMILY_EN.get(an.family, an.family), "reliability": an.reliability(), "fingerprint": an.fingerprint()}
        if a.dataset in an.excluded_datasets():
            entry["excluded"] = "not evaluated on its own dataset: with the 80/20 pooled split of the reference code the images of this dataset are part of the model's training set (its held-out 20 % test is reported in the checkpoint JSON card)"
        analyzers.append(entry)
    checkpoints = []
    for card in sorted(WEIGHTS.glob("*.json")):
        if card.name in ("manifest.json",):
            continue
        d = json.loads(card.read_text(encoding="utf-8"))
        # stesso ordine di preferenza dell'app (kuznetsov.WEIGHT_EXTS): il file caricato è l'.h5, di cui la scheda riporta l'hash
        weights = next((card.with_suffix(ext) for ext in (".h5", ".weights.h5", ".keras", ".joblib") if card.with_suffix(ext).exists()), None)
        alt = {q.name: sha256(q) for ext in (".keras",) for q in [card.with_suffix(ext)] if q.exists() and weights is not None and q != weights}
        checkpoints.append({"card": card.name, "weights_file": weights.name if weights else None, "sha256": d.get("weights_sha256") or d.get("clf_sha256") or d.get("sha256"),
                            "loaded_file_sha256_verified": (sha256(weights) == (d.get("weights_sha256") or d.get("clf_sha256"))) if weights else None,
                            "other_formats": alt or None,
                            "arch": d.get("arch") or d.get("analyzer"), "dataset": d.get("dataset"), "protocol": d.get("protocol"),
                            "seed": d.get("seed"), "saved": d.get("saved") or d.get("created"), "test": {k: d.get("test", {}).get(k) for k in ("n", "acer", "eer", "auc")} if d.get("test") else None})
    cfg = {
        "task": "C1", "table": f"results/c1/{a.dataset}_table.md", "generated_by": "scripts/make_c1_manifest.py",
        "repository": {"commit": git("rev-parse", "HEAD"), "livedetection": git("-C", "src/livedetection", "rev-parse", "HEAD"),
                       "silent_face": git("-C", "src/third_party/Silent-Face-Anti-Spoofing", "rev-parse", "HEAD")},
        "dataset": {"id": a.dataset, "name": ds["name"], "n_bona_fide": n_real, "n_attack": n_attack, "info": info,
                    "manifest": man.name, "manifest_sha256": sha256(man)},
        "conventions": {"label": "0 = bona fide, 1 = attack", "score": "probability of attack in [0, 1] (higher = more likely an attack)",
                        "threshold": 0.5, "decision": "attack if score > 0.5; a tie counts as bona fide",
                        "threshold_selection": "fixed a priori, not tuned on a development set (EER and BPCER@APCER10 are descriptive)",
                        "metrics": "ISO/IEC 30107-3: APCER, BPCER, ACER, accuracy at the threshold; EER on the full ROC without interpolation; BPCER@APCER=10%; AUC",
                        "differences_vs_reference_code": ["BPCER@APCER10 counts rejected bona fide (the reference evaluation_utils.py counted accepted ones)",
                                                          "non-computable values are null instead of 0",
                                                          "EER searched on the full ROC (drop_intermediate=False)"],
                        "no_face": "images where no face is detected get the analyzer's fallback (whole image) and are flagged in the CSV",
                        "frames": "NUAA images are still frames; consecutive frames of one session are not independent samples"},
        "commands": {"recompute_from_saved_scores": f"python scripts/eval_dataset.py --dataset {a.dataset} --from-cache --lang en --export results/c1/{a.dataset}_scores.csv --table results/c1/{a.dataset}_table.md",
                     "new_inference": f"python scripts/eval_dataset.py --dataset {a.dataset} --force --lang en --export results/c1/{a.dataset}_scores.csv --table results/c1/{a.dataset}_table.md",
                     "requires": {"recompute": "results/eval/<dataset>.json (saved scores) and the checkpoint JSON cards in the weights folder (they enumerate the CNN variants and classifiers); no weight files, images or downloaded models", "inference": "data/eval/<dataset>/ (images per manifest), models/weights/ (checkpoint files and cards, hashes verified before loading), the reference CNN code (src/architectures.py), Silent-Face-Anti-Spoofing (MiniFASNet weights and the RetinaFace detector), the Hugging Face cache for CLIP, Depth Anything and DINOv2, the MediaPipe face landmarker in <weights folder>/../third_party/face_landmarker.task (descriptive analyzer, no score)"}},
        "analyzers": analyzers, "checkpoints": checkpoints,
    }
    cfgp = out / f"{a.dataset}_config.json"
    cfgp.write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"manifest: {man} ({len(rows)} immagini: {n_real} bona fide, {n_attack} attacchi, sha256 {cfg['dataset']['manifest_sha256'][:16]}…)")
    print(f"config:   {cfgp} ({len(analyzers)} analizzatori, {len(checkpoints)} checkpoint)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
