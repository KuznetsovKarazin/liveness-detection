"""
Riproduzione minima del difetto di tensorflow-metal trovato nel pilota N1 (solo dati SINTETICI, nessuna immagine,
nessun peso, nessun modello): x @ W + b seguito da ReLU (lo schema di un Dense con attivazione relu) calcolato su
/GPU:0 in eager e in modalità grafo (tf.function, come fit, evaluate e predict di Keras), e su CPU in grafo, contro
numpy in float64. Controlli: lo stesso calcolo senza ReLU e con tanh.

  python scripts/n1_metal_relu_repro.py [--disable-meta-optimizer]

Uscita: una riga per caso con l'errore massimo e se l'uscita del grafo GPU coincide con la pre-attivazione (ReLU
saltata), poi una riga finale:
  MISMATCH: ReLU after MatMul+BiasAdd dropped on GPU in graph mode     (codice 1)
  NO MISMATCH                                                          (codice 0)
  NO GPU: nothing to compare                                           (codice 2: nessuna GPU visibile)
  SKIP: TensorFlow not installed (missing input)                       (codice 2: TensorFlow assente)
Dipendenze: TensorFlow (con tensorflow-metal per la GPU), numpy.
"""
import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402

TOL = 1e-4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--disable-meta-optimizer", action="store_true", help="grappler meta-optimizer disattivato")
    a = ap.parse_args()
    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        print("SKIP: TensorFlow not installed (missing input)")
        return 2
    if a.disable_meta_optimizer:
        tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__}, GPUs {len(gpus)}, disable_meta_optimizer {a.disable_meta_optimizer}")
    if not gpus:
        print("NO GPU: nothing to compare")
        return 2
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 256)).astype("float32")
    W = (rng.standard_normal((256, 16)) / 16).astype("float32")
    b = (rng.standard_normal(16) * 0.5).astype("float32")
    pre = x.astype(np.float64) @ W.astype(np.float64) + b
    cases = {"relu": (tf.nn.relu, np.maximum(pre, 0)), "none": (lambda t: t, pre), "tanh": (tf.nn.tanh, np.tanh(pre))}
    dropped = False
    for name, (act, ref) in cases.items():
        def f(p, q, r, act=act):
            return act(tf.nn.bias_add(tf.matmul(p, q), r))
        with tf.device("/GPU:0"):
            eager = f(x, W, b).numpy()
            graph = tf.function(f)(x, W, b).numpy()
        with tf.device("/CPU:0"):
            cpu = tf.function(f)(x, W, b).numpy()
        err = {k: float(np.max(np.abs(v.astype(np.float64) - ref))) for k, v in (("gpu_eager", eager), ("gpu_graph", graph), ("cpu_graph", cpu))}
        as_pre = name != "none" and float(np.max(np.abs(graph.astype(np.float64) - pre))) < TOL
        bad = err["gpu_graph"] > TOL
        dropped |= bad and name == "relu"
        print(f"  {name:4s} gpu_eager {err['gpu_eager']:.2e}  gpu_graph {err['gpu_graph']:.2e}  cpu_graph {err['cpu_graph']:.2e}"
              f"  gpu_graph == pre-activation: {'n/a' if name == 'none' else 'yes' if as_pre else 'no'}  {'WRONG' if bad else 'ok'}")
    print("MISMATCH: ReLU after MatMul+BiasAdd dropped on GPU in graph mode" if dropped else "NO MISMATCH")
    return 1 if dropped else 0


if __name__ == "__main__":
    sys.exit(main())
