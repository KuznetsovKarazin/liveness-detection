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

Adattamenti per il pilota C2 (scripts/c2_pilot.py), tutti opzionali: senza di essi il comportamento è quello di prima.
  --out-dir DIR    pesi e scheda in DIR invece che in models/weights/, senza toccare manifest.json; DIR è controllata con
                   scripts/path_guard.py (rifiutati un collegamento simbolico, un file, l'albero di lavoro che contiene il
                   codice quando non ne è la radice, cioè i pacchetti accettati del repository del gruppo, le cartelle dei
                   pesi e dei dati, anche spostate, e ogni cartella protetta del repository: dentro il repository è
                   ammessa solo results/c2/) e non deve contenere già pesi, file Keras o scheda dello stesso modello,
                   nemmeno come collegamenti: nulla viene sovrascritto né seguito; la scheda aggiunge
                   restored_epoch, restored_by, val_loss_history e keras_sha256.
  --no-test-eval   nessuna valutazione finale su X_test (richiede --out-dir): la scheda non ha né il campo test né n_test
                   e aggiunge test_evaluation = "disabled"; con split_check aggiunge split_check = "passed: ...".
  split_check      (solo da codice, train(..., split_check=...)) SHA-256 del file di ogni frame di X_train: lo split per
                   soggetto deve riprodurre esattamente i manifest di fitting e di early stopping, altrimenti errore.

Tutte le opzioni sono controllate (validate_args) prima di leggere immagini o costruire cache.

Dispositivo (DEVICE_POLICY): LivenessNet solo su CPU. Con tensorflow-metal, sulla GPU in modalità grafo (fit, evaluate,
predict) la ReLU del Dense dopo il Flatten di LivenessNet viene saltata (pilota N1; riproduzione minima
scripts/n1_metal_relu_repro.py del pacchetto N1).
Creazione del modello, compile, training, validazione, eventuale valutazione del test e salvataggio avvengono dentro
tf.device("/CPU:0"), senza nascondere la GPU al processo; le AttackNet restano sul dispositivo predefinito di
TensorFlow (la GPU se visibile), come prima, e una GPU presente ma nascosta al processo è rifiutata. Verifiche, con
errore e nessun peso salvato se falliscono: prima del fit le variabili del modello sono sul dispositivo atteso; dopo il
fit lo sono anche quelle dell'ottimizzatore, e per LivenessNet la modalità grafo (dove la ReLU viene saltata) coincide con
l'eager su 4 frame di validazione: uscite di tutti gli strati con una tf.function, probabilità di model.predict e loss di
model.evaluate (le funzioni di Keras del test e della validazione), scarto relativo massimo entro GRAPH_EAGER_TOL; lo
scarto è registrato per ogni architettura. Scheda e
manifest registrano training_device (richiesto, atteso, effettivo, dispositivi visibili e fisici, scarto grafo/eager).

Riproducibilità: seed 42 per split, mescolamento, inizializzazione dei pesi e augmentation. Su GPU (Metal) alcune
operazioni restano non deterministiche: due run identici possono differire di poco.

Uso:  python scripts/train_cnn.py --dataset nuaa --per-class 1500 --epochs 10 [--arch LivenessNet AttackNetV2_2] [--protocol pooled]
      [--out-dir DIR] [--no-test-eval]
"""
import argparse
import contextlib
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
# dispositivo richiesto per architettura: "CPU" forzata (tf.device) oppure "default" (scelta di TensorFlow, invariata)
DEVICE_POLICY = {"LivenessNet": "CPU", "AttackNetV1": "default", "AttackNetV2_1": "default", "AttackNetV2_2": "default"}
# scarto relativo massimo ammesso fra le uscite degli strati in grafo e in eager per la politica CPU (su CPU è 0; con
# tensorflow-metal sulla GPU LivenessNet supera 1, AttackNetV2_2 è dell'ordine di 1e-6: 4,6e-6 su un Mac M4)
GRAPH_EAGER_TOL = 1e-5


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


def outer_roots() -> list:
    """Alberi che contengono il codice senza esserne la radice (pacchetti accettati accanto): l'albero di lavoro git che
    contiene ROOT, se diverso da ROOT, e ogni antenato X di ROOT con ROOT dentro X/research (disposizione del repository
    del gruppo, anche senza git). Nessuna uscita è ammessa dentro di essi."""
    import subprocess
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    import path_guard as pg
    root, out = Path(ROOT).resolve(), []
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root, capture_output=True, text=True, timeout=30)
        top = Path(r.stdout.strip()).resolve() if r.returncode == 0 and r.stdout.strip() else None
        if top is not None and not pg.same(top, root):
            out.append(top)
    except (OSError, subprocess.SubprocessError):
        pass
    for x in root.parents:
        if (x / "research").is_dir() and any(pg.same(x / "research", y) for y in root.parents) and not any(pg.same(x, o) for o in out):
            out.append(x)
    return out


def out_dir_problem(d) -> str:
    """Motivo per cui d non può essere la cartella di uscita di --out-dir ("" se può), con scripts/path_guard.py (identità
    dei file, in entrambe le direzioni): collegamento simbolico, file esistente, alberi esterni con i pacchetti accettati
    (outer_roots), cartelle dei pesi e dei dati anche spostate con le variabili d'ambiente, poi le regole di
    deletion_problem (radice, home, repository, cartelle protette; dentro il repository solo results/c2/)."""
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    import path_guard as pg
    if Path(d).is_symlink():
        return f"{Path(d)} is a symbolic link"
    if Path(d).exists() and not Path(d).is_dir():
        return f"{Path(d)} exists and is not a folder"
    for g in outer_roots():
        if pg.within(d, g) or pg.contains(d, g):
            return f"{Path(d)} is inside, or contains, the work tree {g.name} with the accepted packages"
    for q in (WEIGHTS, DATA):
        if pg.within(d, q) or (Path(q).exists() and pg.contains(d, q)):
            return f"{Path(d)} is, contains or is inside the {'weights' if q == WEIGHTS else 'data'} folder"
    return pg.deletion_problem(d, ROOT, pg.BASE_PROTECTED, [ROOT / "results" / "c2"])


def existing_outputs_problem(d, arch_name: str, ds: str, protocol: str) -> str:
    """Con --out-dir nessun file di uscita può esistere già (nemmeno come collegamento, anche senza bersaglio): pesi,
    file Keras e scheda non si sovrascrivono mai e nessun collegamento viene seguito."""
    key = f"{arch_name}__{ds}" + ("-pooled" if protocol == "pooled" else "")
    found = [f"{key}{x}" for x in (".keras", ".h5", ".json") if os.path.lexists(Path(d) / f"{key}{x}")]
    return f"{Path(d)} already holds {', '.join(found)} (files or links are never overwritten)" if found else ""


def check_split(tr_i, va_i, split_check: dict) -> None:
    """Controllo dello split per SHA-256 (pilota C2). split_check["sha256"]: SHA-256 del file di ogni frame di X_train,
    nell'ordine della cache; "train" e "val": SHA-256 delle righe dei manifest di fitting e di early stopping. I due lati
    dello split devono coincidere con i manifest (come multinsiemi), altrimenti SystemExit con un errore esplicito."""
    sha = list(split_check["sha256"])
    if len(sha) != len(tr_i) + len(va_i):
        raise SystemExit(f"ERROR: split check: {len(sha)} file hashes for {len(tr_i) + len(va_i)} training frames")
    for side, idx in (("train", tr_i), ("val", va_i)):
        if len(idx) != len(split_check[side]):
            raise SystemExit(f"ERROR: split check failed: the {side} side of GroupShuffleSplit has {len(idx)} frames, its manifest "
                             f"{len(split_check[side])} rows")
        got, want = sorted(sha[int(i)] for i in idx), sorted(split_check[side])
        if got != want:
            n_diff = len(set(got) ^ set(want))
            raise SystemExit(f"ERROR: split check failed: the {side} side of GroupShuffleSplit ({len(got)} frames) does not reproduce "
                             f"its manifest ({len(want)} rows, {n_diff} SHA-256 values differ)")


def short_device(name: str) -> str:
    """"/job:localhost/replica:0/task:0/device:CPU:0" → "CPU:0"; stringa vuota se il nome non indica un dispositivo."""
    return str(name).rsplit("device:", 1)[-1] if "device:" in str(name) else ""


def device_scope(tf, requested: str):
    """Ambito del training: tf.device("/CPU:0") per la politica CPU, nessun ambito per "default"."""
    if requested == "CPU":
        return tf.device("/CPU:0")
    if requested == "default":
        return contextlib.nullcontext()
    raise SystemExit(f"ERROR: unknown device policy {requested!r}")


def device_problem(arch_name: str, requested: str, expected: str, effective: list, gap: float = None) -> str:
    """Motivo per cui il training non rispetta la politica del dispositivo ("" se la rispetta): un solo dispositivo per
    le variabili, uguale a quello atteso; per la politica CPU, quando è stato misurato, scarto grafo/eager entro
    GRAPH_EAGER_TOL."""
    stop = " (CPU execution not guaranteed: training stopped, no weights saved)" if requested == "CPU" else " (training stopped, no weights saved)"
    if effective != [expected]:
        return f"{arch_name}: device policy {requested!r} requires every variable on {expected}, found {effective or 'none'}{stop}"
    if requested == "CPU" and gap is not None and not np.isfinite(gap):
        return f"{arch_name}: graph/eager comparison not finite ({gap}): outputs not finite after training (training stopped, no weights saved)"
    if requested == "CPU" and gap is not None and not gap <= GRAPH_EAGER_TOL:
        return (f"{arch_name}: graph-mode outputs differ from eager by {gap:.3e} (relative), above {GRAPH_EAGER_TOL:g}{stop}")
    return ""


def graph_eager_gap(tf, model, X, Y) -> tuple:
    """Scarto relativo massimo fra modalità grafo ed eager su X (etichette one-hot Y), e dove è massimo: uscite di tutti
    gli strati con una tf.function nuova; probabilità con model.predict e loss con model.evaluate, le funzioni di Keras
    del test e della validazione di fit (la loss eager è model.compute_loss sulle stesse uscite). Inferenza
    (training=False): nessun effetto su pesi e statistiche. Il punto è None se tutti gli scarti sono 0."""
    layers = [l for l in model.layers if not isinstance(l, tf.keras.layers.InputLayer)]
    sub = tf.keras.Model(model.inputs, [l.output for l in layers])
    xb = tf.constant(X)
    eager = [np.asarray(t) for t in sub(xb, training=False)]
    graph = [np.asarray(t) for t in tf.function(lambda t: sub(t, training=False))(xb)]
    probs = np.asarray(model(xb, training=False))
    loss = float(model.compute_loss(x=xb, y=tf.constant(Y), y_pred=tf.constant(probs)))
    graph += [np.asarray(model.predict(X, batch_size=len(X), verbose=0)),
              np.asarray(model.evaluate(X, Y, batch_size=len(X), verbose=0, return_dict=True)["loss"])]
    eager += [probs, np.asarray(loss)]
    names = [l.name for l in layers] + ["predict", "evaluate"]
    rel = [float(np.max(np.abs(g - e))) / max(float(np.max(np.abs(e))), 1e-12) for g, e in zip(graph, eager)]
    worst = int(np.argmax(rel))
    return rel[worst], (names[worst] if rel[worst] > 0 else None)


def train(arch_name: str, ds: str, data, epochs: int, batch: int, lr: float, protocol: str = "official",
          out_dir=None, split_check: dict = None, test_eval: bool = True):
    """Addestra un'architettura e salva pesi, metriche sul test e voce del manifest. Con out_dir pesi e scheda vanno lì
    (mai dentro models/weights/) e manifest.json non si tocca; con test_eval=False nessuna valutazione su X_test; con
    split_check lo split per soggetto è confrontato per SHA-256 con i manifest (vedi check_split). Il dispositivo segue
    DEVICE_POLICY ed è verificato prima e dopo il fit (vedi device_problem)."""
    if not test_eval and out_dir is None:
        # senza --out-dir la scheda e manifest.json di models/weights/ si aspettano le metriche del test
        raise SystemExit("ERROR: --no-test-eval requires --out-dir (models/weights/ and its manifest.json are never written without a test evaluation)")
    dest = WEIGHTS if out_dir is None else Path(out_dir)
    if out_dir is not None and (msg := out_dir_problem(dest)):
        raise SystemExit(f"ERROR: --out-dir refused: {msg}")
    if out_dir is not None and (msg := existing_outputs_problem(dest, arch_name, ds, protocol)):
        raise SystemExit(f"ERROR: --out-dir refused: {msg}")
    if split_check is not None and protocol != "official":
        raise SystemExit("ERROR: the split check applies to the official protocol only")
    if arch_name not in DEVICE_POLICY:
        raise SystemExit(f"ERROR: no device policy for {arch_name!r}")
    requested = DEVICE_POLICY[arch_name]
    import tensorflow as tf
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    # bug noto di tensorflow-metal con le AttackNet (Mutation::Apply error nel remapper): disattiviamo quell'ottimizzazione
    tf.config.optimizer.set_experimental_options({"remapping": False})
    visible = sorted({d.device_type for d in tf.config.get_visible_devices()})
    physical = sorted({d.device_type for d in tf.config.list_physical_devices()})
    if "GPU" in physical and "GPU" not in visible:
        # la politica "default" (AttackNet) userebbe la CPU in silenzio: nessuna GPU va nascosta al processo di training
        raise SystemExit(f"ERROR: a GPU is present but hidden from the process (visible devices {visible}): the training "
                         "never hides the GPUs (LivenessNet uses a CPU scope)")
    # dispositivo atteso: CPU:0 per la politica CPU; per "default" quello che TensorFlow sceglie (la GPU se visibile)
    expected = "CPU:0" if requested == "CPU" or not tf.config.list_logical_devices("GPU") else "GPU:0"
    arch = load_module("kuznetsov_architectures", LIVEDETECTION / "src" / "architectures.py")
    with device_scope(tf, requested):
        return _train_in_scope(tf, arch, arch_name, ds, data, epochs, batch, lr, protocol, out_dir, split_check, test_eval,
                               dest, requested, expected, visible, physical)


def _train_in_scope(tf, arch, arch_name, ds, data, epochs, batch, lr, protocol, out_dir, split_check, test_eval, dest,
                    requested, expected, visible, physical):
    """Corpo di train dentro l'ambito del dispositivo: creazione del modello, split, compile, fit, verifica del
    dispositivo, eventuale test, salvataggio di pesi, scheda e manifest."""
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    model = arch.create_model(arch_name).get_model()
    if msg := device_problem(arch_name, requested, expected, sorted({short_device(v.device) for v in model.variables})):
        raise SystemExit(f"ERROR: {msg}")
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
    if split_check is not None:
        if not by_subject:
            raise SystemExit("ERROR: split check requested but the validation split is not by subject")
        check_split(tr_i, va_i, split_check)
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
    # dopo il fit: variabili del modello e dell'ottimizzatore, e scarto grafo/eager (strati, predict, evaluate) su 4 frame
    # di validazione (il difetto di tensorflow-metal compare solo in grafo); nessun effetto su pesi, seed e storia
    effective = sorted({short_device(v.device) for v in list(model.variables) + list(model.optimizer.variables())})
    gap, gap_layer = graph_eager_gap(tf, model, Xva[:4], Yva[:4])
    if msg := device_problem(arch_name, requested, expected, effective, gap):
        raise SystemExit(f"ERROR: {msg}")
    device = {"requested": requested, "expected": expected, "effective": effective[0], "visible": visible, "physical": physical,
              "graph_eager_max_rel_diff": gap, "graph_eager_layer": gap_layer}
    m = None
    if test_eval:
        # test: split ufficiale (official) oppure il 20 % tenuto fuori (pooled)
        Xte, yte = data["X_test"].astype(np.float32) / 255.0, data["y_test"]
        p = model.predict(Xte, batch_size=batch, verbose=0)
        s_attack = p[:, 1]                                              # probabilità di attacco = convenzione delle metriche
        from tesi_app.evaluation import compute_metrics
        m = compute_metrics(yte, s_attack)
    dest.mkdir(parents=True, exist_ok=True)
    key = f"{arch_name}__{ds}" + ("-pooled" if protocol == "pooled" else "")
    wpath = dest / f"{key}.keras"
    model.save(wpath)
    h5 = dest / f"{key}.h5"
    model.save_weights(str(h5), save_format="h5")   # HDF5 classico (NON .weights.h5): è quello che l'app carica per primo, funziona anche su Windows
    info = {"arch": arch_name, "dataset": ds, "protocol": protocol, "seed": SEED,
            "n_train": int(len(ytr)), "n_val": int(len(yva)), **({"n_test": int(len(yte))} if test_eval else {}),
            "val_split": "per soggetto" if by_subject else "casuale",
            "epochs_run": len(hist.history["loss"]), "best_epoch": best_epoch + 1, "train_seconds": round(train_s), "lr": lr, "batch": batch,
            "best_val_loss": float(hist.history["val_loss"][best_epoch]), "val_acc_at_best": float(hist.history["val_accuracy"][best_epoch]),
            "training_device": device, **({"test": m} if test_eval else {}), "weights_sha256": file_sha256(h5), "saved": time.strftime("%Y-%m-%d %H:%M")}
    if out_dir is not None:
        # scheda del pilota C2: epoca ripristinata (quella di val_loss minimo, ripristinata dall'early stopping o qui sopra)
        # e storia di val_loss, per il controllo dei pesi prima di qualunque punteggio
        info |= {"restored_epoch": best_epoch + 1, "restored_by": "early stopping" if es.stopped_epoch else "end of epochs (set_weights)",
                 "val_loss_history": [float(v) for v in hist.history["val_loss"]], "keras_sha256": file_sha256(wpath)}
    if not test_eval:
        info["test_evaluation"] = "disabled"
    if split_check is not None:
        info["split_check"] = "passed: fitting and early-stopping frames equal to the manifests by SHA-256"
    (dest / f"{key}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    if out_dir is None:
        # manifest riassuntivo (lettura-modifica-scrittura: non lanciare due training in parallelo che lo aggiornano)
        man_p = WEIGHTS / "manifest.json"
        man = json.loads(man_p.read_text()) if man_p.exists() else {}
        man[key] = {k: v for k, v in info.items() if k != "test"} | {"test_acer": m.get("acer"), "test_eer": m.get("eer"), "test_auc": m.get("auc")}
        man_p.write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(f"  dispositivo: richiesto {requested}, effettivo {device['effective']} (visibili {visible}, scarto grafo/eager {gap:.3e} in {gap_layer})", flush=True)
    test_txt = f"TEST acer {m['acer']} eer {m['eer']} auc {m['auc']}" if test_eval else "TEST not evaluated (--no-test-eval)"
    print(f"== {arch_name} su {ds} [{protocol}]: {info['epochs_run']} epoche in {train_s:.0f}s | val acc {info['val_acc_at_best']:.3f} | {test_txt} → {wpath.name}", flush=True)
    return info


def validate_args(a) -> str:
    """Opzioni controllate prima di leggere immagini o costruire cache ("" se valide): architetture, epoche, lotto,
    learning rate, immagini per classe, --no-test-eval solo con --out-dir, --out-dir ammessa (out_dir_problem)."""
    if not a.arch or any(x not in ARCHS for x in a.arch):
        return f"--arch {a.arch} must name architectures among {ARCHS}"
    import math
    if a.epochs < 1 or a.batch < 1 or not a.lr > 0 or not math.isfinite(a.lr) or (a.per_class is not None and a.per_class < 0):
        return "--epochs and --batch must be at least 1, --lr positive and finite, --per-class not negative"
    if a.no_test_eval and a.out_dir is None:
        return "--no-test-eval requires --out-dir (models/weights/ and its manifest.json are never written without a test evaluation)"
    if a.out_dir is not None and (msg := out_dir_problem(a.out_dir)):
        return f"--out-dir refused: {msg}"
    for arch in a.arch if a.out_dir is not None else []:
        if msg := existing_outputs_problem(a.out_dir, arch, a.dataset, a.protocol):
            return f"--out-dir refused: {msg}"
    return ""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--arch", nargs="*", default=ARCHS)
    ap.add_argument("--per-class", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--protocol", choices=["official", "pooled"], default="official")
    ap.add_argument("--out-dir", default=None, help="cartella di uscita di pesi e schede (default models/weights/ con manifest.json)")
    ap.add_argument("--no-test-eval", action="store_true", help="nessuna valutazione finale su X_test (scheda senza test)")
    a = ap.parse_args(argv)
    if msg := validate_args(a):                       # prima di qualunque lettura di immagini o cache
        raise SystemExit(f"ERROR: {msg}")
    print(f"Preparo {a.dataset} (max {a.per_class} per classe)…", flush=True)
    data = build_arrays(a.dataset, a.per_class or None, DATA / "processed")   # 0 = tutte le immagini
    print({k: v.shape for k, v in data.items()}, flush=True)
    for arch in a.arch:
        train(arch, a.dataset, data, a.epochs, a.batch, a.lr, a.protocol, out_dir=a.out_dir, test_eval=not a.no_test_eval)


if __name__ == "__main__":
    main()
