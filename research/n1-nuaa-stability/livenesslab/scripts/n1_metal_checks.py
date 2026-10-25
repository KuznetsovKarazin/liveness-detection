"""
Tre controlli del pilota N1 sul difetto di tensorflow-metal (ReLU dopo MatMul + BiasAdd saltata sulla GPU in modalità
grafo), stampati solo come aggregati.

  python scripts/n1_metal_checks.py fit-steps [--disable-meta-optimizer]
      Solo dati SINTETICI: una piccola rete Dense(relu) + BatchNormalization + Dense(softmax), stessi pesi iniziali,
      5 passi di fit (SGD 0,1, un lotto da 32) su CPU e su GPU; stampa la differenza massima dei pesi dopo i 5 passi.
      Ultima riga: WEIGHTS DIFFER (codice 1) oppure WEIGHTS AGREE (codice 0); senza GPU o senza TensorFlow una riga
      SKIP con il motivo e codice 2.

  python scripts/n1_metal_checks.py c1-rescore --config <n1_nuaa_stability.json> --manifests <manifest M2 NUAA>
      --cache <nuaa_1500.npz> --weights <LivenessNet__nuaa.h5 del C1> --c1-scores <scores.csv del run C1>
      Dati NUAA riservati, sola lettura: punteggi del checkpoint LivenessNet C1 (SHA-256 fissato sotto) sui 3.000 frame
      del test con lo stesso percorso su CPU di n1_train.py score (controlli della cache e dei manifest compresi);
      confronto con i punteggi C1 accettati delle 300 immagini (SHA-256 fissato sotto): correlazione di Pearson, scarto
      massimo, AUROC sui 300 (nuovi e accettati) e sui 3.000. Nessun punteggio per immagine viene scritto o stampato.
      Ingressi non validi: riga ERROR e codice 1 (nessun traceback); senza TensorFlow, dopo i controlli degli ingressi,
      SKIP e codice 2.

  python scripts/n1_metal_checks.py c1-linear --weights <LivenessNet__nuaa.h5 del C1>
      Pesi del checkpoint C1 (SHA-256 fissato sotto), ingressi CASUALI (nessuna immagine): probabilità con Keras predict
      sulla GPU contro la stessa rete su CPU, con dense1 normale e con dense1 lineare (ReLU tolta), su ingressi casuali lisci. Ultima riga
      PREDICT EQUALS LINEAR DENSE1 (codice 1) oppure PREDICT EQUALS LIVENESSNET (codice 0); SKIP e codice 2 senza GPU
      o senza TensorFlow.

Dipendenze: TensorFlow, numpy; scikit-learn per c1-rescore.
"""
import argparse
import csv
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

C1_WEIGHTS_SHA256 = "d861c5969f24376aff5aaac30c2b2b8793e0cd142dd016756dc82f188ccd99a5"   # LivenessNet__nuaa.h5 (C1)
C1_SCORES_SHA256 = "0a52d875eaeabe61173162dd645ec415349822e9855572b0c5c22bf3db594ed0"    # scores.csv del run C1 accettato
C1_ANALYZER = "livenessnet__nuaa"


def fit_steps(a) -> int:
    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        print("SKIP: TensorFlow not installed (missing input)")
        return 2
    if not tf.config.list_physical_devices("GPU"):
        print("SKIP: no GPU visible (missing input)")
        return 2
    if a.disable_meta_optimizer:
        tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})
    L = tf.keras.layers
    tf.keras.utils.set_random_seed(3)
    i = L.Input((256,))
    h = L.BatchNormalization()(L.Dense(64, activation="relu")(i))
    m = tf.keras.Model(i, L.Dense(2, activation="softmax")(h))
    x = np.random.default_rng(0).standard_normal((32, 256)).astype("float32")
    y = np.random.default_rng(1).integers(0, 2, 32)
    w0, res = m.get_weights(), {}
    for dev in ("/CPU:0", "/GPU:0"):
        m.set_weights(w0)
        m.compile(optimizer=tf.keras.optimizers.SGD(0.1), loss=tf.keras.losses.SparseCategoricalCrossentropy())
        with tf.device(dev):
            m.fit(x, y, epochs=5, batch_size=32, verbose=0, shuffle=False)
        res[dev] = m.get_weights()
    d = max(float(np.max(np.abs(p - q))) for p, q in zip(res["/CPU:0"], res["/GPU:0"]))
    print(f"TensorFlow {tf.__version__}, disable_meta_optimizer {a.disable_meta_optimizer}")
    print(f"  fit 5 steps (synthetic), max |w_cpu - w_gpu| = {d:.2e}")
    print("WEIGHTS DIFFER" if d > 1e-4 else "WEIGHTS AGREE")
    return 1 if d > 1e-4 else 0


def c1_rescore(a) -> int:
    import n1_train as N
    from sklearn.metrics import roc_auc_score
    for p, want, what in ((a.weights, C1_WEIGHTS_SHA256, "C1 weights"), (a.c1_scores, C1_SCORES_SHA256, "C1 scores")):
        if Path(p).is_symlink() or not Path(p).is_file() or N.sha256_file(p) != want:
            print(f"ERROR: {what}: missing or SHA-256 different from {want[:12]}…")
            return 1
    try:
        cfg, cfg_sha = N.load_config(a.config)
        inp = N.load_inputs(cfg, types.SimpleNamespace(cache=a.cache, manifests=a.manifests), frames=False)
    except N.N1Error as e:
        print(f"ERROR: {e}")
        return 1
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        print("SKIP: TensorFlow not installed (missing input)")
        return 2
    try:
        scorer = N.tf_scorer(Path(a.weights), C1_WEIGHTS_SHA256)
        s = N.probabilities(scorer, N.frames_of(inp, "test"))[:, 1]
    except N.N1Error as e:
        print(f"ERROR: {e}")
        return 1
    by_key = dict(zip(inp.order["test"], s))
    y = np.array([int(inp.by_key["cnn_test"][k]["label"]) for k in inp.order["test"]])
    with open(a.c1_scores, newline="", encoding="utf-8") as f:
        c1 = {r["sample_id"]: (float(r["score_attack"]), int(r["label"])) for r in csv.DictReader(f) if r["analyzer_id"] == C1_ANALYZER}
    if set(c1) != inp.c1_keys:
        print("ERROR: the C1 scores do not cover exactly the C1 subset of the test")
        return 1
    keys = sorted(c1)
    new = np.array([by_key[k] for k in keys])
    old = np.array([c1[k][0] for k in keys])
    yc = np.array([c1[k][1] for k in keys])
    if not np.array_equal(yc, np.array([int(inp.by_key["cnn_test"][k]["label"]) for k in keys])):
        print("ERROR: C1 labels differ from the manifest")
        return 1
    print(f"C1 LivenessNet checkpoint {C1_WEIGHTS_SHA256[:12]}… scored on the CPU (n1_train.py path), configuration {cfg_sha[:12]}…")
    print(f"  C1 300: Pearson correlation with the accepted C1 scores {np.corrcoef(new, old)[0, 1]:.4f}, max |difference| {np.max(np.abs(new - old)):.4f}")
    print(f"  C1 300: AUROC new {roc_auc_score(yc, new):.4f}, accepted C1 scores {roc_auc_score(yc, old):.4f}")
    print(f"  test 3000: AUROC {roc_auc_score(y, s):.4f}")
    return 0


def c1_linear(a) -> int:
    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        print("SKIP: TensorFlow not installed (missing input)")
        return 2
    if not tf.config.list_physical_devices("GPU"):
        print("SKIP: no GPU visible (missing input)")
        return 2
    import n1_train as N
    from tesi_app.paths import LIVEDETECTION, load_module
    if Path(a.weights).is_symlink() or not Path(a.weights).is_file() or N.sha256_file(a.weights) != C1_WEIGHTS_SHA256:
        print(f"ERROR: C1 weights: missing or SHA-256 different from {C1_WEIGHTS_SHA256[:12]}…")
        return 1
    tf.config.optimizer.set_experimental_options({"remapping": False})
    arch = load_module("kuznetsov_architectures", Path(LIVEDETECTION) / "src" / "architectures.py")
    m = arch.create_model("LivenessNet").get_model()
    m.load_weights(str(a.weights))
    lin = arch.create_model("LivenessNet").get_model()
    lin.get_layer("dense1").activation = tf.keras.activations.linear
    lin.set_weights(m.get_weights())
    # ingressi casuali lisci (rumore 16 x 16 ingrandito): con il rumore uniforme le probabilità saturano
    noise = np.random.default_rng(0).standard_normal((32, 16, 16, 3)).astype(np.float32)
    x = np.clip(0.5 + 0.15 * tf.image.resize(noise, (256, 256)).numpy(), 0, 1).astype(np.float32)
    with tf.device("/CPU:0"):
        p_cpu, p_lin = m(x, training=False).numpy()[:, 1], lin(x, training=False).numpy()[:, 1]
    with tf.device("/GPU:0"):
        p_gpu = m.predict(x, batch_size=32, verbose=0)[:, 1]
    d_net, d_lin = float(np.max(np.abs(p_gpu - p_cpu))), float(np.max(np.abs(p_gpu - p_lin)))
    print(f"C1 LivenessNet checkpoint {C1_WEIGHTS_SHA256[:12]}…, 32 smooth random inputs, TensorFlow {tf.__version__}; "
          f"p_attack on the CPU in [{p_cpu.min():.4f}, {p_cpu.max():.4f}]")
    print(f"  max |p_gpu_predict - p_cpu| = {d_net:.2e}; max |p_gpu_predict - p_cpu_with_linear_dense1| = {d_lin:.2e}")
    linear = d_lin < 1e-5 < d_net
    print("PREDICT EQUALS LINEAR DENSE1" if linear else "PREDICT EQUALS LIVENESSNET" if d_net < 1e-5 else "NEITHER")
    return 1 if linear else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit-steps")
    f.add_argument("--disable-meta-optimizer", action="store_true")
    c = sub.add_parser("c1-rescore")
    for k in ("--config", "--manifests", "--cache", "--weights", "--c1-scores"):
        c.add_argument(k, required=True)
    w = sub.add_parser("c1-linear")
    w.add_argument("--weights", required=True)
    a = ap.parse_args()
    return {"fit-steps": fit_steps, "c1-rescore": c1_rescore, "c1-linear": c1_linear}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
