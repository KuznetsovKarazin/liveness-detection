"""
Diagnostica del pilota N1 (solo dati SINTETICI casuali, nessuna immagine, nessun peso addestrato): confronta l'uscita
di LivenessNet (architetture del codice di riferimento, pesi di inizializzazione con seed fisso) calcolata in eager e in
modalità grafo (tf.function, come fit, evaluate e predict) sul dispositivo predefinito e su CPU, strato per strato, e
contro numpy per il Dense dopo il Flatten. Serve a documentare perché il trainer N1 addestra solo su CPU.

  python scripts/n1_metal_diagnostic.py [--disable-meta-optimizer]

Stampa per ogni modalità la differenza massima delle probabilità rispetto a CPU eager e, per lo strato dense1, rispetto
a numpy; ultima riga GRAPH-MODE MISMATCH oppure NO MISMATCH (codice 0 in entrambi i casi: è una misura, non un test);
senza TensorFlow «SKIP: TensorFlow not installed (missing input)» e codice 2.
Dipendenze: TensorFlow (con o senza tensorflow-metal), numpy.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--disable-meta-optimizer", action="store_true", help="grappler meta-optimizer disattivato")
    a = ap.parse_args()
    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        print("SKIP: TensorFlow not installed (missing input)")
        return 2
    from tesi_app.paths import LIVEDETECTION, load_module
    opts = {"remapping": False}                       # come train_cnn.py e n1_trainer.py
    if a.disable_meta_optimizer:
        opts = {"disable_meta_optimizer": True}
    tf.config.optimizer.set_experimental_options(opts)
    tf.keras.utils.set_random_seed(0)            # python, numpy e TensorFlow: pesi iniziali ripetibili
    arch = load_module("kuznetsov_architectures", Path(LIVEDETECTION) / "src" / "architectures.py")
    model = arch.create_model("LivenessNet").get_model()
    X = np.random.default_rng(0).integers(0, 256, (8, 256, 256, 3), dtype=np.uint8).astype(np.float32) / 255.0
    names = [layer.name for layer in model.layers[1:]]
    sub = tf.keras.Model(model.input, [layer.output for layer in model.layers[1:]])
    with tf.device("/CPU:0"):
        ref = [np.asarray(t, dtype=np.float64) for t in sub(X, training=False)]
    graph = tf.function(lambda x: sub(x, training=False))
    devices = [d.device_type for d in tf.config.get_visible_devices()]
    print(f"TensorFlow {tf.__version__}, visible devices {devices}, grappler options {opts}")
    out = {}
    for label, dev, fn in (("default device, eager", None, lambda: sub(X, training=False)), ("default device, graph", None, lambda: graph(X)),
                           ("CPU, graph", "/CPU:0", lambda: graph(X))):
        if dev:
            with tf.device(dev):
                res = [np.asarray(t, dtype=np.float64) for t in fn()]
        else:
            res = [np.asarray(t, dtype=np.float64) for t in fn()]
        out[label] = res
        worst = max(names, key=lambda n: float(np.max(np.abs(res[names.index(n)] - ref[names.index(n)]))))
        d_out = float(np.max(np.abs(res[-1] - ref[-1])))
        d_dense = float(np.max(np.abs(res[names.index("dense1")] - ref[names.index("dense1")])))
        print(f"  {label:24s} max |p - p_cpu_eager| = {d_out:.3e}; dense1 {d_dense:.3e}; largest difference at layer {worst}")
    flat = ref[names.index("flatten")]
    W, b = [w.numpy().astype(np.float64) for w in model.get_layer("dense1").weights]
    np_dense = np.maximum(flat @ W + b, 0)
    print(f"  numpy dense1 vs CPU eager: {float(np.max(np.abs(np_dense - ref[names.index('dense1')]))):.3e}")
    mism = float(np.max(np.abs(out["default device, graph"][-1] - ref[-1]))) > 1e-3
    print("GRAPH-MODE MISMATCH" if mism else "NO MISMATCH")
    return 0


if __name__ == "__main__":
    sys.exit(main())
