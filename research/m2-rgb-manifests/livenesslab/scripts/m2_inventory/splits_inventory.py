"""
Cache di addestramento (*_1500.npz) e ricostruzione numerica degli split dei checkpoint CASIA-FASD e CelebA-Spoof:
official (GroupShuffleSplit per soggetto o per immagine), pooled (80/20 stratificato + 10 % di validazione come nel
commit 0ac7143), immagini di data/eval finite nel training pooled. Nessuna rete viene caricata: solo indici ed etichette.
Uso:  python scripts/m2_inventory/splits_inventory.py [--private]
"""
import argparse
import collections
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CASIA_CACHE, CASIA_EVAL, CASIA_IMAGES, CELEBA_CACHE, CELEBA_EVAL, CELEBA_IMAGES, PER_CLASS, SEED,  # noqa: E402
                    list_images, sha256_file, spread, strip_private)

DATASETS = {"casia_fasd": (CASIA_IMAGES, CASIA_EVAL, CASIA_CACHE), "celeba_spoof": (CELEBA_IMAGES, CELEBA_EVAL, CELEBA_CACHE)}


def npy_shape(npz: Path, key: str):
    """Forma e tipo di un array senza caricarlo (intestazione .npy dentro lo zip)."""
    import numpy.lib.format as fmt
    with zipfile.ZipFile(npz) as zf, zf.open(key + ".npy") as fh:
        version = fmt.read_magic(fh)
        # API pubblica di numpy per versione del formato (niente funzioni private)
        reader = fmt.read_array_header_1_0 if version == (1, 0) else fmt.read_array_header_2_0
        shape, _, dtype = reader(fh)
    return list(shape), str(dtype)


def folder_lists(base: Path, tag: str):
    """Stesso ordine di build_arrays in scripts/train_cnn.py: bona fide poi attacchi, spread(iterdir, 1500)."""
    files, y = [], []
    for cls, lab in (("bonafide", 0), ("attack", 1)):
        fs = spread(list_images(base / f"{cls}_{tag}"), PER_CLASS)
        files += fs
        y += [lab] * len(fs)
    return files, np.array(y, np.int64)


def run_one(ds: str):
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    base, ev, cache = DATASETS[ds]
    z = np.load(cache)
    r = {"cache": {"bytes": cache.stat().st_size, "sha256": sha256_file(cache), "keys": z.files,
                   "X_train": npy_shape(cache, "X_train"), "X_test": npy_shape(cache, "X_test"),
                   "y_train": dict(sorted(collections.Counter(z["y_train"].tolist()).items())),
                   "y_test": dict(sorted(collections.Counter(z["y_test"].tolist()).items())),
                   "s_train_distinct": len(set(z["s_train"].tolist())), "s_test_distinct": len(set(z["s_test"].tolist()))}}
    ftr, ytr = folder_lists(base, "training")
    fte, yte = folder_lists(base, "validation")
    if ds == "casia_fasd":
        s_tr = np.array([p.name.split("_")[0] for p in ftr])
        s_te = np.array([p.name.split("_")[0] for p in fte])
    else:   # prima versione versionata di train_cnn.py: subject_of = nome del file per CelebA-Spoof
        s_tr = np.array([p.name for p in ftr])
        s_te = np.array([p.name for p in fte])
    r["cache_matches_folders"] = {"y_train": bool(np.array_equal(ytr, z["y_train"])), "y_test": bool(np.array_equal(yte, z["y_test"])),
                                  "s_train": bool(np.array_equal(s_tr, z["s_train"])), "s_test": bool(np.array_equal(s_te, z["s_test"]))}
    y, g = z["y_train"], z["s_train"]
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(np.zeros(len(y)), y, g))
    r["official_group_split"] = {"n_train": len(tr), "n_val": len(va), "val_by_class": dict(sorted(collections.Counter(y[va].tolist()).items())),
                                 "n_test": len(z["y_test"]), "test_by_class": r["cache"]["y_test"]}
    tr2, va2 = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    r["official_stratified_alt"] = {"n_train": len(tr2), "n_val": len(va2), "val_by_class": dict(sorted(collections.Counter(y[va2].tolist()).items()))}
    if ds == "casia_fasd":
        r["official_group_split"]["val_subjects"] = len(set(g[va].tolist()))
        r["official_group_split"]["fit_subjects"] = len(set(g[tr].tolist()))
        r["official_group_split"]["val_videos"] = len({ftr[i].name.split(".avi")[0] for i in va})
        r["official_group_split"]["fit_val_shared_subjects"] = len(set(g[tr].tolist()) & set(g[va].tolist()))
        r["official_group_split"]["_val_subject_ids"] = sorted(set(g[va].tolist()), key=int)
    # pooled
    yp = np.concatenate([z["y_train"], z["y_test"]])
    ptr, pte = train_test_split(np.arange(len(yp)), train_size=0.8, stratify=yp, random_state=SEED)
    fit, val = train_test_split(np.arange(len(ptr)), test_size=0.1, stratify=yp[ptr], random_state=SEED)
    r["pooled"] = {"n_train": len(fit), "n_val": len(val), "n_test": len(pte), "test_by_class": dict(sorted(collections.Counter(yp[pte].tolist()).items()))}
    allf = ftr + fte
    in_trainval = set(ptr.tolist())
    evn = {p.name for d in ("real", "attack") for p in (ev / d).iterdir()}
    pos = [len(ftr) + i for i, p in enumerate(fte) if p.name in evn]
    r["pooled"]["eval_in_cache_test"] = len(pos)
    r["pooled"]["eval_files"] = len(evn)
    r["pooled"]["eval_in_pooled_trainval"] = sum(p in in_trainval for p in pos)
    if ds == "casia_fasd":
        key = lambda i: ("train:" if i < len(ftr) else "test:") + allf[i].name.split(".avi")[0]     # noqa: E731
        skey = lambda i: ("train:" if i < len(ftr) else "test:") + allf[i].name.split("_")[0]     # noqa: E731
        vt, ve = {key(i) for i in ptr}, {key(i) for i in pte}
        st, se = {skey(i) for i in ptr}, {skey(i) for i in pte}
        r["pooled"]["videos_both_sides"] = len(vt & ve)
        r["pooled"]["videos_total"] = len(vt | ve)
        r["pooled"]["subjects_both_sides"] = len(st & se)
        r["pooled"]["subjects_total"] = len(st | se)
    return r


def run():
    return {ds: run_one(ds) for ds in DATASETS}


if __name__ == "__main__":
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    r = run()
    print(json.dumps(r if a.private else strip_private(r), indent=1, default=str))
