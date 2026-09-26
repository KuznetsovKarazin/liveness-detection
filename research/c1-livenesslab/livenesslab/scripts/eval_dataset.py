"""
Valutazione da terminale (senza server): esegue gli analizzatori su un dataset di data/eval/<id>, aggiorna la cache
results/eval/<id>.json e stampa la tabella delle metriche. Serve per riprodurre una tabella in modo verificabile.

  python scripts/eval_dataset.py --dataset nuaa                      # usa la cache, calcola solo ciò che manca
  python scripts/eval_dataset.py --dataset nuaa --force              # nuova inferenza su tutte le immagini
  python scripts/eval_dataset.py --dataset nuaa --from-cache         # solo ricalcolo delle metriche dai punteggi salvati
  python scripts/eval_dataset.py --dataset nuaa --analyzers minifasnet livenessnet__nuaa --limit 50
  python scripts/eval_dataset.py --dataset nuaa --export results/eval/nuaa_scores.csv --table results/eval/nuaa_table.md

L'export CSV contiene una riga per (immagine, analizzatore): file, etichetta (0 bona fide / 1 attacco), punteggio di
attacco, tempo, flag di errore / nessun volto, impronta del modello. L'intestazione del rapporto riporta commit del
repository e dei submodule, versioni delle librerie e backend di calcolo: sono i dati richiesti dal run report.
"""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch  # noqa: E402,F401  (prima di TensorFlow: vedi run_app.py)
except Exception:  # noqa: BLE001
    pass

from tesi_app.core import registry  # noqa: E402
from tesi_app import analyzers  # noqa: E402,F401  (registra gli analizzatori)
from tesi_app import evaluation  # noqa: E402

COLS = ["n", "apcer", "bpcer", "acer", "eer", "bpcer_at_apcer10", "auc", "accuracy", "n_errors", "n_noface"]


def git(*args, cwd=ROOT):
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "n/d"


def environment() -> dict:
    """Commit, submodule, librerie e backend: l'intestazione del run report."""
    from tesi_app.paths import LIVEDETECTION, SILENT_FACE
    commit = git("rev-parse", "HEAD")
    # fuori da un repository git (es. il pacchetto esportato) commit e stato non sono disponibili: "n/d", non "modificato"
    info = {"commit": commit, "dirty": bool(git("status", "--porcelain")) if commit != "n/d" else False,
            "livedetection": git("rev-parse", "HEAD", cwd=LIVEDETECTION),
            "silent_face": git("rev-parse", "HEAD", cwd=SILENT_FACE),
            "python": platform.python_version(), "platform": platform.platform()}
    for mod in ("tensorflow", "keras", "torch", "transformers", "numpy", "sklearn", "cv2", "mediapipe", "skimage", "scipy"):
        try:
            m = __import__(mod); info[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            info[mod] = "assente"
    try:
        import tensorflow as tf
        info["tf_devices"] = [d.device_type for d in tf.config.list_physical_devices()]   # "GPU" = Metal su Apple Silicon
    except Exception:  # noqa: BLE001
        info["tf_devices"] = "n/d"
    return info


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def table(summary: dict, names: dict) -> str:
    """Tabella Markdown delle metriche (righe con punteggio prima, poi le note)."""
    rows = sorted(summary["analyzers"].items(), key=lambda kv: ("note" in kv[1], names.get(kv[0], (99, ""))))
    out = ["| analizzatore | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 1)]
    for aid, m in rows:
        if "note" in m:
            out.append(f"| {aid} | {m['n']} | {m['note']} |")
        else:
            out.append(f"| {aid} | " + " | ".join(fmt(m.get(c)) for c in COLS) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="id del dataset (cartella di data/eval, oppure 'samples')")
    ap.add_argument("--analyzers", nargs="*", default=None, help="id degli analizzatori (default: tutti)")
    ap.add_argument("--limit", type=int, default=None, help="immagini per classe (sottoinsieme distribuito)")
    ap.add_argument("--force", action="store_true", help="ricalcola anche le immagini già in cache (nuova inferenza)")
    ap.add_argument("--from-cache", action="store_true", help="niente inferenza: solo metriche dai punteggi salvati")
    ap.add_argument("--export", help="CSV dei punteggi per immagine")
    ap.add_argument("--table", help="file Markdown in cui salvare tabella e intestazione")
    a = ap.parse_args()

    ids = [i for i in (a.analyzers or registry.ids()) if i in registry.ids()]
    names = {x.id: (x.order, x.name) for x in registry.all()}
    env = environment()
    print("== ambiente:", json.dumps(env, ensure_ascii=False), flush=True)
    from tesi_app.paths import DATA, LIVEDETECTION, RESULTS, SILENT_FACE, WEIGHTS
    print("== percorsi:", json.dumps({"data": str(DATA), "results": str(RESULTS), "weights": str(WEIGHTS), "livedetection": str(LIVEDETECTION), "silent_face": str(SILENT_FACE)}), flush=True)
    cache0 = evaluation._load_cache(a.dataset)
    n_cached = sum(len(v) for v in cache0.get("scores", {}).values())
    ds_items = next((len(d["items"]) for d in evaluation.list_datasets() if d["id"] == a.dataset), 0)
    print(f"== dataset {a.dataset}: {ds_items} immagini in {DATA / 'eval' / a.dataset}, {n_cached} punteggi in cache ({RESULTS / 'eval' / (a.dataset + '.json')}), {len(ids)} analizzatori registrati", flush=True)
    if a.from_cache and n_cached == 0:
        sys.exit(f"nessun punteggio in cache per '{a.dataset}': copiare la cache in {RESULTS / 'eval'} (o impostare LIVENESSLAB_RESULTS_DIR)")
    if not a.from_cache and ds_items == 0:
        sys.exit(f"nessuna immagine in {DATA / 'eval' / a.dataset} (o impostare LIVENESSLAB_DATA_DIR)")
    t0 = time.time()
    if a.from_cache:
        summary = evaluation.summarize(a.dataset, ids)
    else:
        def emit(ev):
            if ev["type"] == "eval_progress" and (ev["done"] % 25 == 0 or ev["done"] == ev["todo"]):
                print(f"  {ev['done']}/{ev['todo']} ({time.time() - t0:.0f} s)", flush=True)
            elif ev["type"] == "eval_error":
                sys.exit("errore: " + ev["message"])
            elif ev["type"] == "eval_start":
                print(f"== {ev['name']}: {ev['todo']} immagini da analizzare su {ev['total']}", flush=True)
        evaluation.run_evaluation(a.dataset, ids, emit, threading.Event(), a.limit, a.force)
        summary = evaluation.summarize(a.dataset, ids)
    md = table(summary, names)
    print(md)
    head = (f"# Valutazione `{a.dataset}` · {time.strftime('%Y-%m-%d %H:%M')}\n\n"
            f"Comando: `{' '.join(sys.argv)}`  \nCommit: `{env['commit']}`{' (modifiche non committate)' if env['dirty'] else ''} · "
            f"livedetection `{env['livedetection'][:8]}` · Silent-Face `{env['silent_face'][:8]}`  \n"
            f"Python {env['python']} · TensorFlow {env['tensorflow']} · Keras {env['keras']} · torch {env['torch']} · transformers {env['transformers']} · "
            f"numpy {env['numpy']} · scikit-learn {env['sklearn']} · OpenCV {env['cv2']} · dispositivi TF {env['tf_devices']}  \n"
            f"Soglia 0,5 sul punteggio di attacco (pareggio = bona fide); BPCER@APCER10 con soglia al quantile 0,1 degli attacchi; "
            f"EER senza interpolazione. Impronte dei modelli (SHA-256 dei pesi o id versione): vedi CSV.\n\n")
    if a.table:
        Path(a.table).parent.mkdir(parents=True, exist_ok=True)
        Path(a.table).write_text(head + md + "\n", encoding="utf-8"); print("tabella salvata in", a.table)
    if a.export:
        cache = evaluation._load_cache(a.dataset)
        Path(a.export).parent.mkdir(parents=True, exist_ok=True)
        with open(a.export, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n"); w.writerow(["analyzer", "file", "label", "attack_score", "elapsed_ms", "error", "no_face", "fingerprint"])
            for aid, per_img in cache["scores"].items():
                if aid not in ids:
                    continue
                for name, r in sorted(per_img.items()):
                    w.writerow([aid, name, r["y"], "" if r["s"] is None else r["s"], round(r.get("t", 0), 1), int(bool(r.get("e"))), int(bool(r.get("nf"))), cache["fingerprints"].get(aid, "")])
        print("punteggi esportati in", a.export)


if __name__ == "__main__":
    main()
