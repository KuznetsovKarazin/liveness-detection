"""
Inventario della copia CASIA-FASD (mirror HF): archivio, campi dai nomi dei frame, cartelle preparate, valutazione.
Sola lettura. L'archivio viene solo elencato (tarfile, nessuna estrazione).
Uso:  python scripts/m2_inventory/casia_inventory.py [--private]
      --private stampa anche gli identificativi per soggetto (solo per il riepilogo riservato).
"""
import argparse
import collections
import datetime
import os
import re
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CASIA_ARCHIVE, CASIA_EVAL, CASIA_HF_META, CASIA_IMAGES, CASIA_RAW, CASIA_SPLITS,  # noqa: E402
                    ROME, hf_metadata, sha256_file, spread, strip_private)

NAME = re.compile(r"^(\d+)_((?:HR_)?\d+)\.avi_(\d+)_(real|fake)\.jpg$")
VIDEOS = ["1", "2", "3", "4", "5", "6", "7", "8", "HR_1", "HR_2", "HR_3", "HR_4"]
REAL_VIDEOS = {"1", "2", "HR_1"}   # schema del rilascio originale (fonte esterna): qui si controlla solo la coerenza con le etichette


def archive_info():
    meta = hf_metadata(CASIA_HF_META)
    sha = sha256_file(CASIA_ARCHIVE)
    entries = files = 0
    per_dir = collections.Counter()
    with tarfile.open(CASIA_ARCHIVE) as tf:
        for m in tf:
            entries += 1
            if m.isfile():
                files += 1
                parts = m.name.split("/")       # ./train_img/train_img/color/x.jpg
                per_dir[f"{parts[1]}/{parts[3]}"] += 1
    return {"revision": meta["revision"], "etag": meta["etag"],
            "downloaded": datetime.datetime.fromtimestamp(meta["timestamp"], ROME).strftime("%Y-%m-%d %H:%M:%S %Z"),
            "bytes": CASIA_ARCHIVE.stat().st_size, "sha256": sha, "sha256_equals_etag": sha == meta["etag"],
            "tar_entries": entries, "tar_files": files, "tar_files_per_dir": dict(sorted(per_dir.items()))}


def names_info():
    """Campi dai nomi: soggetto (locale allo split), video, frame, etichetta; color e depth."""
    out = {}
    for split, sub, _ in CASIA_SPLITS:
        color = sorted(p.name for p in (CASIA_RAW / sub / "color").iterdir())
        depth = sorted(p.name for p in (CASIA_RAW / sub / "depth").iterdir())
        ms = [NAME.match(n) for n in color]
        bad = sum(m is None for m in ms)
        g = [m.groups() for m in ms if m]
        units = collections.Counter((s, v) for s, v, _, _ in g)
        lab_ok = all((lab == "real") == (v in REAL_VIDEOS) for _, v, _, lab in g)
        subjects = sorted({int(s) for s, _, _, _ in g})
        missing = [(s, v) for s in subjects for v in VIDEOS if (str(s), v) not in units]
        frames = sorted({int(f) for _, _, f, _ in g})
        out[split] = {"color_files": len(color), "depth_files": len(depth), "color_equals_depth_names": color == depth,
                      "nonmatching_names": bad, "real": sum(l == "real" for *_, l in g), "fake": sum(l == "fake" for *_, l in g),
                      "subjects": len(subjects), "videos": len(units), "missing_subject_video": len(missing),
                      "frames_per_video_min": min(units.values()), "frames_per_video_max": max(units.values()),
                      "frame_index_min": frames[0], "frame_index_max": frames[-1],
                      "frame_index_multiple_of_25": all(f % 25 == 0 for f in frames),
                      "labels_consistent_with_video_token": lab_ok,
                      "_subject_ids": subjects}
    return out


def image_info():
    """Dimensioni e modi dei frame (color e depth) e massimo dei pixel delle mappe depth per classe."""
    from PIL import Image
    import numpy as np
    sizes, modes = collections.Counter(), collections.Counter()
    depth_max_lt32 = collections.Counter()
    depth_n = collections.Counter()
    for _, sub, _ in CASIA_SPLITS:
        for kind in ("color", "depth"):
            for p in (CASIA_RAW / sub / kind).iterdir():
                with Image.open(p) as im:
                    sizes[(kind, im.size)] += 1
                    modes[(kind, im.mode, "real" if p.name.endswith("_real.jpg") else "fake")] += 1
                    if kind == "depth":
                        lab = "real" if p.name.endswith("_real.jpg") else "fake"
                        depth_n[lab] += 1
                        depth_max_lt32[lab] += int(np.asarray(im).max()) < 32
    return {"sizes": {f"{k}:{s[0]}x{s[1]}": v for (k, s), v in sizes.items()},
            "modes": {f"{k}:{m}:{l}": v for (k, m, l), v in sorted(modes.items())},
            "depth_files": dict(depth_n), "depth_max_below_32": dict(depth_max_lt32)}


def prepared_info():
    """Cartelle preparate = frame color del mirror con suffisso di classe, hard link; valutazione = spread(test, 150)."""
    out = {}
    eval_info = {}
    for split, sub, tag in CASIA_SPLITS:
        src = {p.name: p for p in (CASIA_RAW / sub / "color").glob("*.jpg")}
        for cls, suf in (("bonafide", "real"), ("attack", "fake")):
            d = CASIA_IMAGES / f"{cls}_{tag}"
            names = {p.name for p in d.iterdir()}
            expected = {n for n in src if n.endswith(f"_{suf}.jpg")}
            linked = sum(os.stat(d / n).st_ino == os.stat(src[n]).st_ino for n in names if n in src)
            out[f"{cls}_{tag}"] = {"files": len(names), "equals_mirror_color": names == expected, "hardlinked": linked}
        if split == "test":
            for cls, suf in (("real", "real"), ("attack", "fake")):
                pool = [src[n] for n in sorted(src) if n.endswith(f"_{suf}.jpg")]
                e = {p.name for p in (CASIA_EVAL / cls).glob("*.jpg")}
                eval_info[cls] = {"files": len(e), "equals_spread_test_150": e == {p.name for p in spread(pool, 150)},
                                  "hardlinked": sum(os.stat(CASIA_EVAL / cls / n).st_ino == os.stat(src[n]).st_ino for n in e),
                                  "subjects": len({n.split("_")[0] for n in e}), "videos": len({n.split(".avi")[0] for n in e})}
    return {"folders": out, "eval": eval_info}


def run(images: bool = True):
    res = {"archive": archive_info(), "names": names_info(), "prepared": prepared_info()}
    if images:
        res["images"] = image_info()
    return res


if __name__ == "__main__":
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    r = run()
    print(json.dumps(r if a.private else strip_private(r), indent=1, default=str))
