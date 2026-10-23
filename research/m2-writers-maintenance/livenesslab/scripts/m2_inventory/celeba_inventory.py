"""
Inventario della copia CelebA-Spoof (shard 0 del mirror HF): schema del parquet, etichette, righe senza immagine,
duplicati, regola che riproduce gli insiemi di righe delle cartelle preparate, identità byte a byte dei file.
Sola lettura (pyarrow, Pillow).
Uso:  python scripts/m2_inventory/celeba_inventory.py [--private]
      --private stampa anche gli indici di riga (solo per il riepilogo riservato).
"""
import argparse
import collections
import datetime
import hashlib
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CELEBA_EVAL, CELEBA_HF_META, CELEBA_IMAGES, CELEBA_PARQUET, ROME, hf_metadata, sha256_file,  # noqa: E402
                    spread, strip_private)

ROW = re.compile(r"^celeba_(\d+)\.jpg$")


def load():
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(CELEBA_PARQUET)
    t = pf.read()
    img = t.column("cropped_image").combine_chunks()
    return pf, t, t.column("labels").to_pylist(), t.column("labelNames").to_pylist(), img.field("bytes").to_pylist(), img.field("path").to_pylist()


def rows_of(d: Path):
    return {int(ROW.match(p.name).group(1)) for p in d.iterdir() if p.is_file()}


def run():
    from PIL import Image
    meta = hf_metadata(CELEBA_HF_META)
    sha = sha256_file(CELEBA_PARQUET)
    pf, t, lab, names, data, paths = load()
    res = {"revision": meta["revision"], "etag": meta["etag"],
           "downloaded": datetime.datetime.fromtimestamp(meta["timestamp"], ROME).strftime("%Y-%m-%d %H:%M:%S %Z"),
           "bytes": CELEBA_PARQUET.stat().st_size, "sha256": sha, "sha256_equals_etag": sha == meta["etag"],
           "columns": t.schema.names, "rows": pf.metadata.num_rows, "row_groups": pf.metadata.num_row_groups,
           "labels": dict(sorted(collections.Counter(lab).items())),
           "label_names_consistent": all((l == 0 and n == "live") or (l == 1 and n == "spoof") for l, n in zip(lab, names)),
           "path_null": sum(p is None for p in paths)}
    null = [i for i, b in enumerate(data) if b is None]
    res["rows_without_image"] = len(null)
    res["rows_without_image_by_label"] = dict(sorted(collections.Counter(lab[i] for i in null).items()))
    valid = {c: [i for i in range(len(lab)) if lab[i] == c and data[i] is not None] for c in (0, 1)}
    res["valid_rows_by_label"] = {c: len(v) for c, v in valid.items()}
    res["png_rows"] = sum(b[:4] == b"\x89PNG" for b in data if b is not None)
    h = collections.defaultdict(list)
    for i, b in enumerate(data):
        if b is not None:
            h[hashlib.sha256(b).hexdigest()].append(i)
    dups = [v for v in h.values() if len(v) > 1]
    res["distinct_images"] = len(h)
    res["exact_duplicate_groups"] = len(dups)
    res["_duplicate_rows"] = dups

    # cartelle preparate e valutazione
    sets = {f: rows_of(CELEBA_IMAGES / f) for f in ("bonafide_training", "attack_training", "bonafide_validation", "attack_validation")}
    ev = {c: rows_of(CELEBA_EVAL / c) for c in ("real", "attack")}
    where = {i: f for f, s in sets.items() for i in s}
    res["duplicates_folders"] = [sorted({where.get(i, "unused") for i in g}) for g in dups]
    res["folders"] = {}
    for f, s in sets.items():
        exp = 0 if f.startswith("bonafide") else 1
        size_mismatch = 0
        byte_identical = 0
        for i in s:
            p = CELEBA_IMAGES / f / f"celeba_{i:05d}.jpg"
            raw = p.read_bytes()
            byte_identical += raw == data[i]
            with Image.open(io.BytesIO(data[i])) as a, Image.open(p) as b:
                size_mismatch += a.size != b.size
        res["folders"][f] = {"files": len(s), "label_mismatch": sum(lab[i] != exp for i in s),
                             "null_rows": sum(data[i] is None for i in s), "size_mismatch": size_mismatch,
                             "byte_identical_to_parquet": byte_identical}
    keys = list(sets)
    res["folder_overlaps"] = sum(len(sets[a] & sets[b]) for k, a in enumerate(keys) for b in keys[k + 1:])
    # regola che riproduce gli insiemi di righe (la procedura originale non è versionata)
    rule = {}
    for c, cls, evname in ((0, "bonafide", "real"), (1, "attack", "attack")):
        e = set(spread(valid[c], 150))
        rest = [i for i in valid[c] if i not in e]
        rule[cls] = {"eval_equals_spread_valid_150": e == ev[evname],
                     "training_equals_first_1500_rest": set(rest[:1500]) == sets[f"{cls}_training"],
                     "validation_equals_eval_plus_next_300": (set(rest[1500:1800]) | e) == sets[f"{cls}_validation"],
                     "unused_rows": len(rest[1800:]),
                     "eval_byte_identical": sum((CELEBA_EVAL / evname / f"celeba_{i:05d}.jpg").read_bytes() == data[i] for i in ev[evname]),
                     "eval_files": len(ev[evname])}
        tr = sorted(sets[f"{cls}_training"])
        va = sorted(sets[f"{cls}_validation"])
        # intercalazione: righe di test dentro l'intervallo coperto dalle righe di training
        rule[cls]["validation_rows_inside_training_range"] = sum(tr[0] < i < tr[-1] for i in va)
        rule[cls]["_training_row_range"] = (tr[0], tr[-1])
        rule[cls]["_validation_row_range"] = (va[0], va[-1])
    res["rule"] = rule
    sizes = collections.Counter()
    for f in sets:
        for p in (CELEBA_IMAGES / f).iterdir():
            with Image.open(p) as im:
                sizes[im.size] += 1
    res["distinct_image_sizes"] = len(sizes)
    return res


if __name__ == "__main__":
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    r = run()
    print(json.dumps(r if a.private else strip_private(r), indent=1, default=str))
