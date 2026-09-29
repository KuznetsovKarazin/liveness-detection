"""
Addestra i classificatori dei metodi classici (LBP, DoG, IQA, IDA) e delle sonde lineari (CLIP, DINOv2) sullo split
UFFICIALE di training di un dataset preparato da scripts/prepare_datasets.py, e li valuta sullo split ufficiale di
test (cartelle *_validation). Attenzione: lo split ufficiale di NUAA NON è a soggetti disgiunti (i soggetti 1–9 stanno
sia in train sia in test, in sessioni di ripresa diverse); CASIA-FASD e SynthASpoof sì. Stesso protocollo "official"
delle CNN di scripts/train_cnn.py.

Per ogni immagine: rilevamento del volto (RetinaFace, lo stesso dell'app), ritaglio con il `CROP` dell'analizzatore,
`features()` del suo modulo. Le feature vengono messe in cache (data/processed/feat_<analizzatore>_<dataset>_<n>.npz).
Classificatori: SVM RBF (feature fatte a mano) o regressione logistica (embedding), sempre con standardizzazione,
`class_weight="balanced"`, seed fisso. Uscita: models/weights/<id>_clf.joblib + <id>_clf.json (dataset, numeri,
metriche sul test, SHA-256 del joblib). Con il joblib presente l'analizzatore passa a "addestrato" e la cache dei
punteggi dell'app viene azzerata automaticamente (impronta = hash del file).

Uso:  python scripts/train_classic.py --dataset nuaa [--analyzers lbp dog iqa ida clip_probe dinov2_probe] [--per-class 1500] [--grid]
"""
import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    import torch  # noqa: F401  (su Windows va importato prima di TensorFlow)
except Exception:  # noqa: BLE001
    pass

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tesi_app.evaluation import compute_metrics, imread  # noqa: E402
from tesi_app.face import get_detector  # noqa: E402
from tesi_app.paths import DATA, WEIGHTS, file_sha256, load_module  # noqa: E402

SEED = 42
tc = load_module("train_cnn", ROOT / "scripts" / "train_cnn.py")      # DATASETS e spread, per non duplicarli
DATASETS, spread = tc.DATASETS, tc.spread
MODULES = {"lbp": "lbp", "dog": "dog", "iqa": "iqa", "ida": "ida", "clip_probe": "probe", "dinov2_probe": "probe"}
FEATURE_FN = {"dinov2_probe": "dino_features"}      # nome della funzione nel modulo, se non è `features`
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def analyzer_spec(aid: str):
    """(funzione feature, CROP, tipo di classificatore) dal modulo dell'analizzatore."""
    mod = importlib.import_module(f"tesi_app.analyzers.{MODULES[aid]}")
    fn = getattr(mod, FEATURE_FN.get(aid, "features"))
    return fn, mod.CROP, mod.CLASSIFIER


def feature_version(aid: str) -> str:
    """Versione delle feature dichiarata dal modulo (FEATURE_VERSION): cambia quando cambia il calcolo delle feature."""
    mod = importlib.import_module(f"tesi_app.analyzers.{MODULES[aid]}")
    return getattr(mod, "FEATURE_VERSION", "1")


def extract(aids, ds: str, per_class, cache_dir: Path):
    """Feature di tutti gli analizzatori richiesti, con un solo rilevamento del volto per immagine. Cache per analizzatore."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    # il nome della cache include la versione delle feature: se cambia il calcolo, la cache vecchia non viene riusata
    caches = {a: cache_dir / f"feat_{a}_v{feature_version(a)}_{ds}_{per_class or 'all'}.npz" for a in aids}
    todo = [a for a in aids if not caches[a].exists()]
    out = {a: {k: np.load(caches[a])[k] for k in np.load(caches[a]).files} for a in aids if a not in todo}
    for a in aids:
        if a not in todo:
            print(f"  {a}: uso la cache {caches[a].name}", flush=True)
    for a in [a for a in aids if a not in todo and "s_train" not in out[a]]:
        base = DATASETS[ds]
        for split, tag in (("train", "training"), ("test", "validation")):
            names = []
            for cls in ("bonafide", "attack"):
                files = spread([p for p in (base / f"{cls}_{tag}").iterdir() if p.suffix.lower() in IMG_EXT], per_class)
                names += [tc.subject_of(p.name, ds) for p in files if imread(p) is not None]
            if len(names) == len(out[a][f"y_{split}"]):
                out[a][f"s_{split}"] = np.array(names)
        np.savez_compressed(caches[a], **out[a])
    if not todo:
        return out
    specs = {a: analyzer_spec(a) for a in todo}
    det = get_detector()
    base = DATASETS[ds]
    acc = {a: {} for a in todo}
    for split, tag in (("train", "training"), ("test", "validation")):
        X = {a: [] for a in todo}; y = []; subj = []
        for cls, lab in (("bonafide", 0), ("attack", 1)):
            files = spread([p for p in (base / f"{cls}_{tag}").iterdir() if p.suffix.lower() in IMG_EXT], per_class)
            t0 = time.time()
            for i, p in enumerate(files, 1):
                img = imread(p)
                if img is None:
                    continue
                subj.append(tc.subject_of(p.name, ds))
                face = det.detect(img)
                crops = {}
                for a in todo:
                    fn, crop_cfg, _ = specs[a]
                    key = (crop_cfg["margin"], crop_cfg["square"])
                    if key not in crops:
                        crops[key] = face.crop(img, **crop_cfg) if face is not None else img
                    X[a].append(fn(crops[key]))
                y.append(lab)
                if i % 250 == 0:
                    print(f"  {ds} {split} {cls}: {i}/{len(files)} ({time.time() - t0:.0f} s)", flush=True)
            print(f"  {ds} {split} {cls}: {len(files)} immagini", flush=True)
        for a in todo:
            acc[a][f"X_{split}"] = np.stack(X[a]).astype(np.float32); acc[a][f"y_{split}"] = np.array(y, np.int64)
            acc[a][f"s_{split}"] = np.array(subj)
    for a in todo:
        np.savez_compressed(caches[a], **acc[a]); out[a] = acc[a]
    return out


def fit(aid: str, kind: str, data, ds: str, per_class, grid: bool = False):
    """Addestra, valuta sul test ufficiale, salva joblib e scheda JSON. Con `grid` gli iperparametri vengono scelti con
    validazione incrociata stratificata a 5 pieghe SUL SOLO TRAINING (AUC): il test ufficiale non entra mai nella scelta."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    Xtr, ytr, Xte, yte = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0); Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)
    if kind == "svm":
        clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=SEED))
        space = {"svc__C": [0.3, 1, 3, 10, 30, 100], "svc__gamma": ["scale", 0.003, 0.01, 0.03, 0.1]}
    else:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"))
        space = {"logisticregression__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]}
    chosen = {}
    t0 = time.time()
    if grid:
        groups = data.get("s_train")
        by_subject = groups is not None and all(groups) and len(set(groups.tolist())) >= 5
        if by_subject:
            # pieghe per soggetto: la stessa persona non sta mai da entrambe le parti (con pieghe casuali l'AUC in
            # validazione era 1,0: memorizzazione di soggetto e sessione, non generalizzazione)
            from sklearn.model_selection import StratifiedGroupKFold
            cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED); cv_iter = list(cv.split(Xtr, ytr, groups))
            print(f"  {aid}: validazione incrociata per soggetto ({len(set(groups.tolist()))} soggetti)", flush=True)
        else:
            cv_iter = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr))
            print(f"  {aid}: validazione incrociata casuale per immagine (nessun identificatore di soggetto)", flush=True)
        gs = GridSearchCV(clf, space, cv=cv_iter, scoring="roc_auc", n_jobs=-1, refit=True)
        gs.fit(Xtr, ytr); clf = gs.best_estimator_
        chosen = {k.split("__", 1)[1]: (v if isinstance(v, str) else float(v)) for k, v in gs.best_params_.items()}
        chosen["cv_auc"] = round(float(gs.best_score_), 4); chosen["cv"] = "per soggetto" if by_subject else "per immagine"
        print(f"  {aid}: iperparametri scelti in validazione incrociata {chosen}", flush=True)
    else:
        clf.fit(Xtr, ytr)
    train_s = time.time() - t0
    p_att = clf.predict_proba(Xte)[:, list(clf.classes_).index(1)]
    m = compute_metrics(yte, p_att)
    m.pop("roc", None)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    wpath, jpath = WEIGHTS / f"{aid}_clf.joblib", WEIGHTS / f"{aid}_clf.json"
    joblib.dump(clf, wpath)
    info = {"analyzer": aid, "dataset": ds, "protocol": "official", "classifier": kind, "per_class": per_class,
            "n_train": int(len(ytr)), "n_test": int(len(yte)), "feature_dim": int(Xtr.shape[1]), "seed": SEED,
            "train_seconds": round(train_s, 1), "hyperparameters": chosen or ("default (C=1, gamma=scale)" if kind == "svm" else "default (C=1)"), "created": datetime.now().isoformat(timespec="seconds"),
            "script_sha256": file_sha256(Path(__file__)), "clf_sha256": file_sha256(wpath), "test": m,
            "feature_version": feature_version(aid),
            "note": "Feature estratte sul ritaglio RetinaFace con il CROP dell'analizzatore; split ufficiale train/test del dataset "
                    "(per NUAA train e test condividono i soggetti 1–9 in sessioni diverse). Con --grid la validazione incrociata per "
                    "soggetto satura vicino ad AUC 1 e discrimina poco tra gli iperparametri: la scelta va letta come indicativa."}
    jpath.write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"== {aid} su {ds} [{kind}]: {len(ytr)} train, {len(yte)} test, {train_s:.0f} s | TEST acer {m['acer']} eer {m['eer']} auc {m['auc']} → {wpath.name}", flush=True)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--analyzers", nargs="*", default=list(MODULES))
    ap.add_argument("--per-class", type=int, default=1500)
    ap.add_argument("--grid", action="store_true", help="scelta di C/gamma con validazione incrociata sul training")
    a = ap.parse_args()
    aids = [x for x in a.analyzers if x in MODULES]
    if "dinov2_probe" in aids:
        from tesi_app.analyzers.probe import dino_available
        if not dino_available():
            print("dinov2_probe saltato: pesi non in cache (scripts/download_models.py)"); aids.remove("dinov2_probe")
    print(f"Feature di {aids} su {a.dataset} (max {a.per_class} per classe)…", flush=True)
    feats = extract(aids, a.dataset, a.per_class or None, DATA / "processed")
    for aid in aids:
        fit(aid, analyzer_spec(aid)[2], feats[aid], a.dataset, a.per_class, grid=a.grid)
