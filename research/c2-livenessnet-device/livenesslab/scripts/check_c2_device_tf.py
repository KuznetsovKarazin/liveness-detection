"""
Controllo del dispositivo del training con il TensorFlow VERO, solo su dati SINTETICI (nessuna immagine, nessun dato
reale, nessun peso esistente): scripts/train_cnn.py train, come lo chiama scripts/c2_pilot.py (out_dir, test_eval=False),
per LivenessNet e poi AttackNetV2_2 nello stesso processo, 1 epoca su 40 frame casuali 256×256 di 8 soggetti finti, in
una cartella temporanea cancellata alla fine.

  1. stesso processo, GPU visibile se c'è: training_device della scheda di LivenessNet (atteso CPU:0 per variabili e
     ottimizzatore, scarto grafo/eager entro GRAPH_EAGER_TOL) e di AttackNetV2_2 (atteso il dispositivo predefinito:
     GPU:0 se visibile);
  2. solo con una GPU visibile: in un sottoprocesso con la GPU nascosta (tf.config.set_visible_devices([], "GPU"), il
     percorso del trainer N1) train deve rifiutarsi (GPU presente ma nascosta); poi, aggirato quel solo controllo in
     quel sottoprocesso, lo stesso training di LivenessNet e confronto dei pesi salvati: con l'ambito tf.device("/CPU:0")
     devono coincidere;
  3. solo con una GPU visibile, controllo negativo: lo stesso training di LivenessNet in un sottoprocesso con la politica
     "default" (sulla GPU, come prima di questa modifica): lo scarto grafo/eager registrato deve superare
     GRAPH_EAGER_TOL (il difetto) e i pesi devono differire da quelli su CPU.

  python scripts/check_c2_device_tf.py          # ultima riga RESULT: PASS (codice 0) o RESULT: FAIL (codice 1)
Senza TensorFlow: SKIP e codice 2. Senza GPU i punti 2 e 3 sono saltati (dichiarato) e il punto 1 attende CPU:0 per
entrambe. Il nome casia_fasd delle schede è solo un'etichetta: i dati sono casuali.

Dipendenze: TensorFlow, h5py, numpy, scikit-learn, OpenCV (importato da train_cnn.py); il codice di riferimento delle architetture (LIVENESSLAB_LIVEDETECTION).
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

N, SUBJECTS, SIDE = 40, 8, 256


def synthetic():
    """40 frame uint8 casuali (seed 0), classi alternate, 8 soggetti finti: lo split per soggetto di train_cnn è attivo."""
    rng = np.random.default_rng(0)
    return {"X_train": rng.integers(0, 256, (N, SIDE, SIDE, 3), dtype=np.uint8), "y_train": np.arange(N, dtype=np.int64) % 2,
            "s_train": np.array([f"{i % SUBJECTS + 1}" for i in range(N)])}


def train_one(arch: str, out: Path) -> dict:
    import train_cnn
    train_cnn.train(arch, "casia_fasd", synthetic(), 1, 8, 1e-3, "official", out_dir=out, test_eval=False)
    return json.loads((out / f"{arch}__casia_fasd.json").read_text(encoding="utf-8"))


def weights(h5: Path) -> list:
    import h5py
    out = []
    with h5py.File(h5, "r") as f:
        f.visititems(lambda name, obj: out.append((name, np.asarray(obj))) if isinstance(obj, h5py.Dataset) else None)
    return sorted(out, key=lambda kv: kv[0])


def child(mode: str, out: str) -> int:
    """Sottoprocesso: --hidden (GPU nascosta prima di qualunque operazione: train deve rifiutarsi, poi il solo controllo
    della GPU nascosta è aggirato) o --gpu (politica "default" per LivenessNet), poi lo stesso training di LivenessNet."""
    import tensorflow as tf
    import train_cnn
    if mode == "--hidden":
        tf.config.set_visible_devices([], "GPU")
        try:
            train_one("LivenessNet", Path(out) / "refused")
            print("ERROR: train accepted a hidden GPU", file=sys.stderr)
            return 1
        except SystemExit as e:
            if "hidden from the process" not in str(e.code):
                print(f"ERROR: unexpected refusal {e.code}", file=sys.stderr)
                return 1
        visible = tf.config.get_visible_devices
        tf.config.list_physical_devices = lambda kind=None: [d for d in visible() if kind in (None, d.device_type)]
    else:
        train_cnn.DEVICE_POLICY["LivenessNet"] = "default"
    card = train_one("LivenessNet", Path(out))
    print(json.dumps(card["training_device"]))
    return 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] in ("--hidden", "--gpu"):
        return child(sys.argv[1], sys.argv[2])
    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        print("SKIP: TensorFlow not installed (missing input)")
        return 2
    gpu = bool(tf.config.list_physical_devices("GPU"))
    bad = []
    with tempfile.TemporaryDirectory(prefix="c2-device-tf-") as tmp:
        tmp = Path(tmp)
        cards = {a: train_one(a, tmp / "same_process") for a in ("LivenessNet", "AttackNetV2_2")}
        print(f"TensorFlow {tf.__version__}, GPU visible: {gpu}")
        import train_cnn
        for arch, card in cards.items():
            d = card["training_device"]
            print(f"  {arch}: requested {d['requested']}, effective {d['effective']}, visible {d['visible']}, "
                  f"graph/eager max relative difference {d['graph_eager_max_rel_diff']:.3e} ({d['graph_eager_layer']})")
        want = {"LivenessNet": "CPU:0", "AttackNetV2_2": "GPU:0" if gpu else "CPU:0"}
        for arch, exp in want.items():
            d = cards[arch]["training_device"]
            if d["effective"] != exp:
                bad.append(f"{arch} trained on {d['effective']}, expected {exp}")
        if not cards["LivenessNet"]["training_device"]["graph_eager_max_rel_diff"] <= train_cnn.GRAPH_EAGER_TOL:
            bad.append("LivenessNet graph/eager difference above the tolerance")
        if gpu:
            ref = weights(tmp / "same_process" / "LivenessNet__casia_fasd.h5")
            for mode, label in (("--hidden", "GPU hidden"), ("--gpu", "GPU, policy default (negative control)")):
                r = subprocess.run([sys.executable, str(Path(__file__).resolve()), mode, str(tmp / mode[2:])],
                                   capture_output=True, text=True, timeout=1800)
                if r.returncode:
                    bad.append(f"{label} run failed: {r.stderr.strip().splitlines()[-1:]}")
                    continue
                other = weights(tmp / mode[2:] / "LivenessNet__casia_fasd.h5")
                rec = json.loads(r.stdout.strip().splitlines()[-1])
                same_names = [k for k, _ in ref] == [k for k, _ in other]
                diff = max(float(np.max(np.abs(x - y))) for (_, x), (_, y) in zip(ref, other)) if same_names else float("inf")
                print(f"  LivenessNet weights, CPU scope with GPU visible vs {label}: {len(ref)} arrays, max |diff| = {diff:.3e}; "
                      f"that run: effective {rec['effective']}, graph/eager {rec['graph_eager_max_rel_diff']:.3e} ({rec['graph_eager_layer']})")
                if mode == "--hidden" and diff != 0.0:
                    bad.append(f"LivenessNet weights differ from the GPU-hidden run (max |diff| {diff:.3e})")
                if mode == "--gpu" and not diff > 0.0:
                    bad.append("LivenessNet weights on the GPU equal those on the CPU: the comparison does not separate the two paths")
                if mode == "--gpu" and not rec["graph_eager_max_rel_diff"] > train_cnn.GRAPH_EAGER_TOL:
                    bad.append("LivenessNet on the GPU: graph/eager difference within the tolerance, the defect is not detected")
        else:
            print("  no GPU visible: comparison with the GPU-hidden run and negative control on the GPU skipped")
    for b in bad:
        print(f"ERROR: {b}", file=sys.stderr)
    print("RESULT: " + ("FAIL" if bad else "PASS"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
