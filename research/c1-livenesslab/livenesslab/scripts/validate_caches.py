"""
Controlla che le cache delle valutazioni (results/eval/<dataset>.json) siano valide rispetto allo stato attuale del codice:
per ogni dataset e analizzatore verifica che l'impronta salvata coincida con quella del modello attuale, che ci sia un
punteggio finito in [0,1] per ogni immagine del dataset (o una nota di esclusione), che non ci siano immagini estranee
e conta errori e immagini senza volto. Stampa una tabella e termina con codice 1 se qualcosa non torna.

Uso:  python scripts/validate_caches.py [--datasets nuaa casia_fasd ...]
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    import torch  # noqa: F401
except Exception:  # noqa: BLE001
    pass
import numpy as np  # noqa: E402

import tesi_app.analyzers  # noqa: E402,F401
from tesi_app.core import registry  # noqa: E402
from tesi_app import evaluation as ev  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="*", default=None)
    a = ap.parse_args()
    current = {an.id: an for an in registry.all()}
    problems = 0
    print(f"{'dataset':14s} {'analizzatore':40s} {'n':>4s} {'err':>3s} {'nf':>3s}  esito")
    for d in ev.list_datasets():
        if a.datasets and d["id"] not in a.datasets:
            continue
        cache = ev._load_cache(d["id"])
        expected = {ev.key_of(p) for p, _ in d["items"]}
        for aid, per in sorted(cache.get("scores", {}).items()):
            notes = []
            an = current.get(aid)
            if an is None:
                notes.append("analizzatore inesistente")
            else:
                if d["id"] in an.excluded_datasets():
                    print(f"{d['id']:14s} {aid:40s} {0:>4d} {'-':>3s} {'-':>3s}  ok (escluso: {an.exclusion_note(d['id'])[:40]})"); continue
                if an.reliability() in ("untrained", "descriptive"):
                    print(f"{d['id']:14s} {aid:40s} {0:>4d} {'-':>3s} {'-':>3s}  ok (senza punteggio: {an.reliability()})"); continue
                if cache["fingerprints"].get(aid) != an.fingerprint():
                    notes.append("impronta diversa dal modello attuale")
            scored = {k for k, v in per.items() if v.get("s") is not None and np.isfinite(v["s"])}
            bad = [k for k, v in per.items() if v.get("s") is not None and not (0 <= v["s"] <= 1)]
            missing = expected - scored; extra = set(per) - expected
            n_err = sum(1 for v in per.values() if v.get("e")); n_nf = sum(1 for v in per.values() if v.get("nf"))
            if bad: notes.append(f"{len(bad)} punteggi fuori [0,1]")
            if missing: notes.append(f"{len(missing)} immagini senza punteggio")
            if extra: notes.append(f"{len(extra)} chiavi non nel dataset")
            ok = not notes
            problems += 0 if ok else 1
            print(f"{d['id']:14s} {aid:40s} {len(scored):>4d} {n_err:>3d} {n_nf:>3d}  {'ok' if ok else 'NO: ' + '; '.join(notes)}")
    print("dataset:", [(d["id"], len(d["items"])) for d in ev.list_datasets() if not a.datasets or d["id"] in a.datasets], "| analizzatori:", len(current), "| problemi:", problems)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
