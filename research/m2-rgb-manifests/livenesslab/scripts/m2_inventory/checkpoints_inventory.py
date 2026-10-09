"""
Schede dei checkpoint CASIA-FASD e CelebA-Spoof: SHA-256 dei .h5 confrontato con la scheda JSON e con
models/weights/manifest.json, conteggi n_train/n_val/n_test, storia del campo val_split nelle versioni git
(0ac7143 e 4d11e67), ora di scrittura delle cache rispetto al primo addestramento. Sola lettura (git show).
Uso:  python scripts/m2_inventory/checkpoints_inventory.py
"""
import datetime
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CASIA_CACHE, CELEBA_CACHE, ROME, ROOT, WEIGHTS, sha256_file  # noqa: E402

COMMITS = ("0ac7143", "4d11e67")


def git_show(rev: str, path: str):
    p = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT, capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else None


def run():
    manifest = json.loads((WEIGHTS / "manifest.json").read_text())
    cards = sorted(list(WEIGHTS.glob("*__casia_fasd*.json")) + list(WEIGHTS.glob("*__celeba_spoof*.json")))
    out = {}
    for c in cards:
        key = c.stem
        card = json.loads(c.read_text())
        h5 = WEIGHTS / f"{key}.h5"
        sha = sha256_file(h5)
        hist = {}
        for rev in COMMITS:
            txt = git_show(rev, f"models/weights/{c.name}")
            hist[rev] = json.loads(txt).get("val_split") if txt else None
        out[key] = {"protocol": card["protocol"], "n_train": card["n_train"], "n_val": card["n_val"], "n_test": card["n_test"],
                    "test_real": card["test"]["n_real"], "test_attack": card["test"]["n_attack"], "val_split": card["val_split"],
                    "saved": card["saved"], "h5_sha256": sha, "card_sha_ok": sha == card["weights_sha256"],
                    "manifest_sha_ok": sha == manifest.get(key, {}).get("weights_sha256"),
                    "keras_exists": (WEIGHTS / f"{key}.keras").exists(), "val_split_history": hist}
    tracked = subprocess.run(["git", "ls-files", "models/weights"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    cache_mtime = {p.name: datetime.datetime.fromtimestamp(p.stat().st_mtime, ROME).strftime("%Y-%m-%d %H:%M:%S") for p in (CASIA_CACHE, CELEBA_CACHE)}
    first_saved = min(v["saved"] for v in out.values())
    return {"cards": out, "tracked_other": [t for t in tracked if t.split("/")[-1] in ("manifest.json", "PROVENANCE.md")],
            "tracked_cards": sum(1 for t in tracked if t.split("/")[-1][:-5] in out),
            "cache_mtime": cache_mtime, "first_saved": first_saved}


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
