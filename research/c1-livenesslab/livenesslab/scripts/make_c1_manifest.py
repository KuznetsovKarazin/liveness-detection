"""
Manifest e configurazione della tabella di riferimento C1 (NUAA, 150 bona fide + 150 attacchi):
  results/c1/nuaa_manifest.csv   una riga per immagine: file, label (0 bona fide / 1 attacco), subject, session, sha256
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
from tesi_app.paths import DATA, WEIGHTS  # noqa: E402

# NUAA: <soggetto>_<soggetto>_<sessione>_<...>.jpg (es. 0004_0004_01_06_03_118.jpg)
NUAA_NAME = re.compile(r"^(\d{4})_\d{4}_(\d{2})_")


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
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="nuaa"); ap.add_argument("--out", default=str(ROOT / "results" / "c1"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ds = next(d for d in ev.list_datasets() if d["id"] == a.dataset)
    root = DATA / "eval" / a.dataset
    info = json.loads((root / "info.json").read_text(encoding="utf-8")) if (root / "info.json").exists() else {}
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
        entry = {"id": an.id, "name": an.name, "family": an.family, "reliability": an.reliability(), "fingerprint": an.fingerprint()}
        if a.dataset in an.excluded_datasets():
            entry["excluded"] = an.exclusion_note(a.dataset)
        analyzers.append(entry)
    checkpoints = []
    for card in sorted(WEIGHTS.glob("*.json")):
        if card.name in ("manifest.json",):
            continue
        d = json.loads(card.read_text(encoding="utf-8"))
        weights = next((card.with_suffix(ext) for ext in (".keras", ".h5", ".joblib") if card.with_suffix(ext).exists()), None)
        checkpoints.append({"card": card.name, "weights_file": weights.name if weights else None, "sha256": d.get("weights_sha256") or d.get("clf_sha256") or d.get("sha256"),
                            "arch": d.get("arch") or d.get("analyzer"), "dataset": d.get("dataset"), "protocol": d.get("protocol"),
                            "seed": d.get("seed"), "saved": d.get("saved"), "test": {k: d.get("test", {}).get(k) for k in ("n", "acer", "eer", "auc")} if d.get("test") else None})
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
        "commands": {"recompute_from_saved_scores": f"python scripts/eval_dataset.py --dataset {a.dataset} --from-cache --export results/c1/{a.dataset}_scores.csv --table results/c1/{a.dataset}_table.md",
                     "new_inference": f"python scripts/eval_dataset.py --dataset {a.dataset} --force --export results/c1/{a.dataset}_scores.csv --table results/c1/{a.dataset}_table.md",
                     "requires": {"recompute": "results/eval/<dataset>.json (saved scores) only", "inference": "data/eval/<dataset>/ (images per manifest), models/weights/ (checkpoints), Hugging Face cache for CLIP, Depth Anything and DINOv2"}},
        "analyzers": analyzers, "checkpoints": checkpoints,
    }
    cfgp = out / f"{a.dataset}_config.json"
    cfgp.write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"manifest: {man} ({len(rows)} immagini: {n_real} bona fide, {n_attack} attacchi, sha256 {cfg['dataset']['manifest_sha256'][:16]}…)")
    print(f"config:   {cfgp} ({len(analyzers)} analizzatori, {len(checkpoints)} checkpoint)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
