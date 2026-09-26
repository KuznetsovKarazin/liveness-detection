"""
Addestra le 4 CNN del docente (LivenessNet, AttackNetV1, AttackNetV2_1, AttackNetV2_2) su un dataset preparato da
scripts/prepare_datasets.py, con lo STESSO percorso di inferenza dell'app:
  volto (RetinaFace, margine 15 %, quadrato) → 256×256 (INTER_AREA) → enhancement del docente (bilateral, CLAHE, USM, gamma) → /255.

Protocolli:
  official  cartelle *_training = train (20 % tenuto per validazione, per soggetto quando il nome file lo permette),
            cartelle *_validation = TEST (split ufficiale del dataset; per NUAA i soggetti 1–9 sono in entrambi, in sessioni
            diverse, quindi non è a soggetti disgiunti; lo sono CASIA-FASD e SynthASpoof).
  pooled    come `_finalize_dataset` del repo del docente: train e test messi insieme, mescolati, split stratificato 80/20
            per immagine (stessi soggetti da entrambe le parti). Riproduce i numeri del docente; NON misura la generalizzazione.

Uscita in models/weights/: <Arch>__<dataset>[-pooled].h5 (HDF5 classico, letto dall'app), .keras, .json (metriche sul test,
iperparametri, seed) e manifest.json riassuntivo. Con i pesi presenti l'app registra le varianti "addestrata su <dataset>".

Riproducibilità: seed 42 per split, mescolamento, inizializzazione dei pesi e augmentation. Su GPU (Metal) alcune
operazioni restano non deterministiche: due run identici possono differire di poco.

Uso:  python scripts/train_cnn.py --dataset nuaa --per-class 1500 --epochs 10 [--arch LivenessNet AttackNetV2_2] [--protocol pooled]
"""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tesi_app.face import get_detector  # noqa: E402
from tesi_app.paths import DATA, LIVEDETECTION, WEIGHTS, file_sha256, load_module  # noqa: E402

SEED = 42
DATASETS = {"nuaa": DATA / "NUAA" / "images", "casia_fasd": DATA / "CASIA_FASD" / "images",
            "celeba_spoof": DATA / "CelebA_Spoof" / "images",
            "synthaspoof": DATA / "SynthASpoof" / "images"}
ARCHS = ["LivenessNet", "AttackNetV1", "AttackNetV2_1", "AttackNetV2_2"]


def spread(items, k):
    """k elementi presi a passo costante dall'elenco ordinato (coprono tutti i soggetti, non solo i primi)."""
    items = sorted(items)
    if k is None or k >= len(items):
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def preprocess(path: Path, det, creator):
    """Stessa catena dell'app: ritaglio del volto, 256×256, enhancement del docente. None se l'immagine non si legge."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    f = det.detect(img)
    crop = f.crop(img, margin=0.15, square=True) if f is not None else img
    if crop.size == 0:
        return None
    rgb = cv2.cvtColor(cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    enhanced = creator.advanced_image_enhancement(rgb)          # funzione del docente, invariata
    return enhanced.astype(np.uint8)


def subject_of(name: str, ds: str) -> str:
    """Identificatore del soggetto dal nome del file, per una validazione con persone NON viste in training.
    Stringa vuota quando il nome non lo contiene (CelebA-Spoof): in quel caso la validazione è casuale per immagine."""
    if ds == "nuaa":            # 0004_0004_01_06_03_193.jpg → 0004
        return name.split("_")[0]
    if ds == "casia_fasd":      # 28_HR_2.avi_125_fake.jpg → 28
        return name.split("_")[0]
    if ds == "synthaspoof":     # img000012.png / replay_ipad_img000012.png → img000012 (identità sintetica)
        m = re.search(r"(img\d+)", name)
        return m.group(1) if m else ""
    return ""


def build_arrays(ds: str, per_class, cache_dir: Path):
    """Carica e pre-elabora le immagini (con cache .npz per dataset e numero per classe).
    La cache non rileva modifiche alle immagini o al preprocessing: cancellarla in data/processed/ se cambiano."""
    cache = cache_dir / f"{ds}_{per_class or 'all'}.npz"
    if cache.exists():
        print(f"  uso la cache {cache.name} (cancellarla se le immagini o il preprocessing sono cambiati)", flush=True)
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    base = DATASETS[ds]
    det = get_detector()
    cd = load_module("kuznetsov_create_datasets", LIVEDETECTION / "scripts" / "create_datasets.py")
    creator = cd.EnhancedDatasetCreator.__new__(cd.EnhancedDatasetCreator)   # istanza "vuota": serve solo advanced_image_enhancement
    out = {}
    for split, tag in (("train", "training"), ("test", "validation")):
        X, y, subj = [], [], []
        for cls, lab in (("bonafide", 0), ("attack", 1)):
            files = spread([p for p in (base / f"{cls}_{tag}").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")], per_class)
            ok = 0
            for p in files:
                arr = preprocess(p, det, creator)
                if arr is not None:
                    X.append(arr); y.append(lab); subj.append(subject_of(p.name, ds)); ok += 1
            print(f"  {ds} {split} {cls}: {ok}/{len(files)} immagini", flush=True)
        out[f"X_{split}"] = np.stack(X); out[f"y_{split}"] = np.array(y, np.int64); out[f"s_{split}"] = np.array(subj)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def train(arch_name: str, ds: str, data, epochs: int, batch: int, lr: float, protocol: str = "official"):
    """Addestra un'architettura e salva pesi, metriche sul test e voce del manifest."""
    import tensorflow as tf
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    # bug noto di tensorflow-metal con le AttackNet (Mutation::Apply error nel remapper): disattiviamo quell'ottimizzazione
    tf.config.optimizer.set_experimental_options({"remapping": False})
    arch = load_module("kuznetsov_architectures", LIVEDETECTION / "src" / "architectures.py")
    model = arch.create_model(arch_name).get_model()
    if protocol == "pooled":
        # come `_finalize_dataset` del docente: tutte le immagini insieme, mescolate, split stratificato 80/20 per immagine
        Xp = np.concatenate([data["X_train"], data["X_test"]]); yp = np.concatenate([data["y_train"], data["y_test"]])
        tr_i, te_i = train_test_split(np.arange(len(yp)), train_size=0.8, stratify=yp, random_state=SEED)
        data = {"X_train": Xp[tr_i], "y_train": yp[tr_i], "X_test": Xp[te_i], "y_test": yp[te_i]}
    Xall, yall = data["X_train"].astype(np.float32) / 255.0, data["y_train"]
    groups = data.get("s_train")
    by_subject = groups is not None and all(groups) and len(set(groups.tolist())) >= 5
    if by_subject:
        # validazione con soggetti mai visti in training (altrimenti l'early stopping sarebbe ottimista)
        tr_i, va_i = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(Xall, yall, groups))
        print(f"  validazione per soggetto: {len(set(groups[va_i].tolist()))} soggetti su {len(set(groups.tolist()))} tenuti fuori dal training", flush=True)
    else:
        tr_i, va_i = train_test_split(np.arange(len(yall)), test_size=0.2, stratify=yall, random_state=SEED)
        print("  validazione casuale per immagine (nessun identificatore di soggetto nei nomi dei file, o cache .npz senza s_train)", flush=True)
    Xtr, ytr, Xva, yva = Xall[tr_i], yall[tr_i], Xall[va_i], yall[va_i]
    Ytr, Yva = np.eye(2)[ytr], np.eye(2)[yva]           # one-hot: indice 0 bona fide, 1 attacco (come nel repo del docente)
    # configurazione base del docente (config/model_configs.py: adam, categorical crossentropy, early stopping su val_loss)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr, clipnorm=1.0),
                  loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")])
    cw = {0: len(ytr) / (2 * (ytr == 0).sum()), 1: len(ytr) / (2 * (ytr == 1).sum())}   # bilanciamento delle classi
    # augmentation solo sul training; RandomBrightness lavora sull'intervallo [0,1] perché i pixel sono già divisi per 255
    aug = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.03), tf.keras.layers.RandomZoom(0.12, 0.12),
                               tf.keras.layers.RandomTranslation(0.06, 0.06), tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
                               tf.keras.layers.RandomContrast(0.2), tf.keras.layers.GaussianNoise(0.02)])
    ds_tr = tf.data.Dataset.from_tensor_slices((Xtr, Ytr)).shuffle(4096, seed=SEED).batch(batch).map(lambda x, y: (aug(x, training=True), y)).prefetch(2)
    ds_va = tf.data.Dataset.from_tensor_slices((Xva, Yva)).batch(batch)
    cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)]
    t0 = time.time()
    hist = model.fit(ds_tr, validation_data=ds_va, epochs=epochs, class_weight=cw, callbacks=cbs, verbose=2)
    train_s = time.time() - t0
    es = cbs[0]
    best_epoch = int(np.argmin(hist.history["val_loss"]))
    if es.stopped_epoch == 0 and es.best_weights is not None:
        # Keras ripristina i pesi migliori solo quando l'early stopping scatta: se le epoche finiscono prima, lo si fa qui
        model.set_weights(es.best_weights)
    # test: split ufficiale (official) oppure il 20 % tenuto fuori (pooled)
    Xte, yte = data["X_test"].astype(np.float32) / 255.0, data["y_test"]
    p = model.predict(Xte, batch_size=batch, verbose=0)
    s_attack = p[:, 1]                                              # probabilità di attacco = convenzione delle metriche
    from tesi_app.evaluation import compute_metrics
    m = compute_metrics(yte, s_attack)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    key = f"{arch_name}__{ds}" + ("-pooled" if protocol == "pooled" else "")
    wpath = WEIGHTS / f"{key}.keras"
    model.save(wpath)
    h5 = WEIGHTS / f"{key}.h5"
    model.save_weights(str(h5), save_format="h5")   # HDF5 classico (NON .weights.h5): è quello che l'app carica per primo, funziona anche su Windows
    info = {"arch": arch_name, "dataset": ds, "protocol": protocol, "seed": SEED,
            "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)), "val_split": "per soggetto" if by_subject else "casuale",
            "epochs_run": len(hist.history["loss"]), "best_epoch": best_epoch + 1, "train_seconds": round(train_s), "lr": lr, "batch": batch,
            "best_val_loss": float(hist.history["val_loss"][best_epoch]), "val_acc_at_best": float(hist.history["val_accuracy"][best_epoch]),
            "test": m, "weights_sha256": file_sha256(h5), "saved": time.strftime("%Y-%m-%d %H:%M")}
    (WEIGHTS / f"{key}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    # manifest riassuntivo (lettura-modifica-scrittura: non lanciare due training in parallelo che lo aggiornano)
    man_p = WEIGHTS / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    man[key] = {k: v for k, v in info.items() if k != "test"} | {"test_acer": m.get("acer"), "test_eer": m.get("eer"), "test_auc": m.get("auc")}
    man_p.write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(f"== {arch_name} su {ds} [{protocol}]: {info['epochs_run']} epoche in {train_s:.0f}s | val acc {info['val_acc_at_best']:.3f} | TEST acer {m['acer']} eer {m['eer']} auc {m['auc']} → {wpath.name}", flush=True)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--arch", nargs="*", default=ARCHS)
    ap.add_argument("--per-class", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--protocol", choices=["official", "pooled"], default="official")
    a = ap.parse_args()
    print(f"Preparo {a.dataset} (max {a.per_class} per classe)…", flush=True)
    data = build_arrays(a.dataset, a.per_class or None, DATA / "processed")   # 0 = tutte le immagini
    print({k: v.shape for k, v in data.items()}, flush=True)
    for arch in a.arch:
        train(arch, a.dataset, data, a.epochs, a.batch, a.lr, a.protocol)
