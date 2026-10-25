"""
Trainer del pilota N1 (stabilità di LivenessNet su NUAA al variare del seed di training): copia della funzione `train`
di scripts/train_cnn.py, che resta invariata (il suo SHA-256 è fissato da export_m2.py e dal pilota C2), con le sole
differenze chieste dalla scheda N1:

  - seed di training (`train_seed`: random, numpy, TensorFlow, quindi inizializzazione dei pesi e augmentation, e
    mescolamento del tf.data) separato dal seed dello split (`split_seed`: GroupShuffleSplit per soggetto, sempre 42);
  - iperparametri presi dalla configurazione N1 già validata (massimo 12 epoche, lotto 16, learning rate iniziale
    0,001, patience 4): ottimizzatore, perdita, bilanciamento delle classi, augmentation e scheduler sono quelli del
    trainer attuale, riscritti qui riga per riga;
  - nessuna valutazione sul test: i dati passati al trainer contengono solo X_train, y_train e s_train (il test non
    entra nemmeno nel processo di training); i punteggi si calcolano dopo, su CPU, in scripts/n1_train.py;
  - split per soggetto obbligatorio (nessun ripiego casuale) e controllo dello split per SHA-256 contro i manifest M2
    di fitting e di early stopping, con la stessa regola di train_cnn.check_split;
  - scheda con la storia per epoca (loss, accuracy, val_loss, val_accuracy, learning rate), epoca ripristinata e SHA-256
    dei file .h5 e .keras.

Checkpoint: i pesi dell'epoca con val_loss minimo (EarlyStopping con restore_best_weights; se le epoche finiscono prima
che l'early stopping scatti, set_weights con i pesi migliori conservati dal callback, come in train_cnn.py). A parità di
val_loss vale la prima epoca (miglioramento stretto in Keras, argmin qui).

Dispositivo: solo CPU (GPU nascosta prima di qualunque operazione e controllata). Con TensorFlow 2.15 e
tensorflow-metal la ReLU dopo un Dense (MatMul + BiasAdd + ReLU) viene saltata sulla GPU in modalità grafo (fit,
evaluate, predict): l'uscita è la pre-attivazione; sulla GPU in eager e su CPU il calcolo è corretto e lo corregge solo
disable_meta_optimizer. In LivenessNet riguarda il Dense dopo il Flatten: su GPU il val_loss dell'early stopping non
sarebbe quello dei pesi valutati su CPU. Riproduzione: scripts/n1_metal_relu_repro.py.

Riproducibilità: i seed fissano inizializzazione, augmentation e ordine dei lotti; su CPU le riduzioni in parallelo
possono comunque dare differenze minime fra due run con lo stesso seed. Non si attiva enable_op_determinism (non
cambia la matematica ma può rifiutare operazioni dell'augmentation).
"""
import json
import random
import time
from pathlib import Path

import numpy as np

KEY = "LivenessNet__nuaa"           # nome dei file del checkpoint (stesso schema di train_cnn.py, protocollo official)


def check_split(tr_i, va_i, split_check: dict) -> None:
    """Stessa regola di train_cnn.check_split: SHA-256 del file di ogni frame di X_train (ordine della cache) e righe dei
    manifest di fitting ("train") e di early stopping ("val"), confrontati come multinsiemi. SystemExit se differiscono."""
    sha = list(split_check["sha256"])
    if len(sha) != len(tr_i) + len(va_i):
        raise SystemExit(f"ERROR: split check: {len(sha)} file hashes for {len(tr_i) + len(va_i)} training frames")
    for side, idx in (("train", tr_i), ("val", va_i)):
        if len(idx) != len(split_check[side]):
            raise SystemExit(f"ERROR: split check failed: the {side} side of GroupShuffleSplit has {len(idx)} frames, its manifest "
                             f"{len(split_check[side])} rows")
        got, want = sorted(sha[int(i)] for i in idx), sorted(split_check[side])
        if got != want:
            raise SystemExit(f"ERROR: split check failed: the {side} side of GroupShuffleSplit ({len(got)} frames) does not reproduce "
                             f"its manifest ({len(want)} rows, {len(set(got) ^ set(want))} SHA-256 values differ)")


def split_indices(y, groups, split_seed: int):
    """Split per soggetto del trainer: GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=split_seed)."""
    from sklearn.model_selection import GroupShuffleSplit
    return next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=split_seed).split(np.zeros(len(y)), y, groups))


def file_sha256(p) -> str:
    import hashlib
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train(data: dict, training: dict, out_dir, train_seed: int, split_seed: int, split_check: dict, arch_module) -> dict:
    """Addestra LivenessNet e salva in out_dir (cartella nuova, creata dal chiamante) LivenessNet__nuaa.h5, .keras e
    .json (scheda). `data`: solo X_train (uint8), y_train, s_train; `training`: blocco "training" della configurazione
    validata; `arch_module`: architetture del codice di riferimento (create_model). Restituisce la scheda."""
    if set(data) != {"X_train", "y_train", "s_train"}:
        raise SystemExit(f"ERROR: the trainer receives only X_train, y_train and s_train, got {sorted(data)}")
    out_dir = Path(out_dir)
    found = [f"{KEY}{x}" for x in (".keras", ".h5", ".json") if (out_dir / f"{KEY}{x}").exists() or (out_dir / f"{KEY}{x}").is_symlink()]
    if found:
        raise SystemExit(f"ERROR: {out_dir.name} already holds {', '.join(found)} (never overwritten)")
    import tensorflow as tf
    if training["device"] != "CPU":
        raise SystemExit(f"ERROR: unknown training device {training['device']!r}")
    # solo CPU: su GPU Metal, in grafo, la ReLU del Dense dopo il Flatten di LivenessNet viene saltata (vedi docstring)
    try:
        tf.config.set_visible_devices([], "GPU")
    except (RuntimeError, ValueError) as e:
        raise SystemExit(f"ERROR: the GPU cannot be hidden ({e}): TensorFlow was initialised before the trainer") from None
    devices = sorted({d.device_type for d in tf.config.get_visible_devices()})
    if devices != ["CPU"]:
        raise SystemExit(f"ERROR: visible devices {devices}, CPU only expected")
    random.seed(train_seed); np.random.seed(train_seed); tf.random.set_seed(train_seed)
    # bug noto di tensorflow-metal (Mutation::Apply error nel remapper): come in train_cnn.py
    tf.config.optimizer.set_experimental_options({"remapping": False})
    model = arch_module.create_model("LivenessNet").get_model()
    yall, groups = np.asarray(data["y_train"]), np.asarray(data["s_train"])
    if groups.dtype.kind != "U" or not all(groups) or len(set(groups.tolist())) < 5:
        raise SystemExit("ERROR: the validation split must be by subject (s_train with at least 5 subjects)")
    tr_i, va_i = split_indices(yall, groups, split_seed)
    check_split(tr_i, va_i, split_check)
    print(f"  validation by subject: {len(set(groups[va_i].tolist()))} of {len(set(groups.tolist()))} subjects held out "
          f"(split seed {split_seed}, train seed {train_seed})", flush=True)
    Xall = data["X_train"].astype(np.float32) / 255.0
    Xtr, ytr, Xva, yva = Xall[tr_i], yall[tr_i], Xall[va_i], yall[va_i]
    Ytr, Yva = np.eye(2)[ytr], np.eye(2)[yva]           # one-hot: indice 0 bona fide, 1 attacco
    lr, batch, epochs, patience = training["initial_learning_rate"], training["batch"], training["max_epochs"], training["early_stopping_patience"]
    # come train_cnn.py: Adam (legacy, clipnorm 1,0), categorical crossentropy, accuracy, classi bilanciate
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr, clipnorm=1.0),
                  loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")])
    cw = {0: len(ytr) / (2 * (ytr == 0).sum()), 1: len(ytr) / (2 * (ytr == 1).sum())}
    aug = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.03), tf.keras.layers.RandomZoom(0.12, 0.12),
                               tf.keras.layers.RandomTranslation(0.06, 0.06), tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
                               tf.keras.layers.RandomContrast(0.2), tf.keras.layers.GaussianNoise(0.02)])
    ds_tr = tf.data.Dataset.from_tensor_slices((Xtr, Ytr)).shuffle(4096, seed=train_seed).batch(batch).map(lambda x, y: (aug(x, training=True), y)).prefetch(2)
    ds_va = tf.data.Dataset.from_tensor_slices((Xva, Yva)).batch(batch)
    cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)]
    t0 = time.time()
    hist = model.fit(ds_tr, validation_data=ds_va, epochs=epochs, class_weight=cw, callbacks=cbs, verbose=2)
    train_s = time.time() - t0
    es = cbs[0]
    h = hist.history
    best = int(np.argmin(h["val_loss"]))
    if es.stopped_epoch == 0 and es.best_weights is not None:
        model.set_weights(es.best_weights)           # epoche finite prima dell'early stopping: pesi migliori anche qui
    wpath, h5 = out_dir / f"{KEY}.keras", out_dir / f"{KEY}.h5"
    model.save(wpath)
    model.save_weights(str(h5), save_format="h5")   # HDF5 classico (NON .weights.h5): è quello che l'app carica per primo, funziona anche su Windows
    card = {"arch": "LivenessNet", "dataset": "nuaa", "protocol": "official", "train_seed": int(train_seed), "split_seed": int(split_seed),
            "n_train": int(len(ytr)), "n_val": int(len(yva)), "val_split": "by subject",
            "max_epochs": int(epochs), "epochs_run": len(h["loss"]), "restored_epoch": best + 1,
            "restored_by": "early stopping" if es.stopped_epoch else "end of epochs (set_weights)",
            "best_val_loss": float(h["val_loss"][best]), "val_acc_at_best": float(h["val_accuracy"][best]),
            "history": {k: [float(v) for v in h[k]] for k in ("loss", "accuracy", "val_loss", "val_accuracy", "lr") if k in h},
            "device": "CPU", "visible_devices": devices, "lr": lr, "batch": batch, "early_stopping_patience": patience, "train_seconds": round(train_s, 1),
            "test_evaluation": "none (scores computed afterwards on CPU)",
            "split_check": "passed: fitting and early-stopping frames equal to the manifests by SHA-256",
            "weights_file": h5.name, "weights_sha256": file_sha256(h5), "keras_file": wpath.name, "keras_sha256": file_sha256(wpath),
            "saved": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out_dir / f"{KEY}.json").write_text(json.dumps(card, indent=1) + "\n", encoding="utf-8")
    print(f"== LivenessNet train seed {train_seed}: {card['epochs_run']} epochs in {train_s:.0f}s, restored epoch {card['restored_epoch']} "
          f"(val_loss {card['best_val_loss']:.6f})", flush=True)
    return card
