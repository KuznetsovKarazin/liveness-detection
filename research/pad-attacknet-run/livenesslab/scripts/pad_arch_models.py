"""
Registro delle due architetture del confronto tra reti PAD (incarico del coordinatore sulla scheda C2, 17/11/2026):
LivenessNet, quella del codice di riferimento del coordinatore (src/livedetection/src/architectures.py, caricata come fa
nuaa_folds_run.default_arch_loader, SHA-256 fissato nella configurazione), e ResNet-18, scritta qui perché
keras.applications di Keras 2.15 non la contiene.

ResNet-18 di He, Zhang, Ren, Sun, «Deep Residual Learning for Image Recognition» (CVPR 2016), versione a
post-attivazione (v1): layout della colonna 18-layer della Tabella 1 con scorciatoie a proiezione (opzione B, §3.3), come
nell'implementazione di riferimento torchvision; i risultati a 18 strati dell'articolo (Tabella 2) usano invece l'opzione
A (scorciatoie con zero-padding, §4.1); l'opzione B è quella delle reti a 50/101/152 strati. Che la proiezione 1x1 sia
seguita da una BatchNormalization è dedotto da §3.4 («BN right after each convolution»), non scritto nella Tabella 1.
  stem      ZeroPadding2D(3) -> Conv2D 7x7, 64 filtri, stride 2 -> BN -> ReLU -> ZeroPadding2D(1) -> MaxPooling2D 3x3
            stride 2 (lo zero di riempimento equivale al -inf del max-pool perché segue una ReLU: valori >= 0)
  stadi     quattro stadi di due blocchi base, 64, 128, 256 e 512 filtri; blocco base: conv 3x3 -> BN -> ReLU -> conv
            3x3 -> BN, somma con la scorciatoia, ReLU; nel primo blocco degli stadi 2-4 la prima conv ha stride 2
            (ZeroPadding2D(1) e padding "valid") e la scorciatoia è una proiezione conv 1x1 stride 2 + BN (opzione B);
            le conv 3x3 a stride 1 usano padding "same" (simmetrico: kernel dispari, stride 1)
  testa     GlobalAveragePooling2D -> Dense(2, softmax, name="output"), colonna 1 = attacco, come LivenessNet
Dimensioni spaziali con ingresso 256x256: 256 -> 128 (stem) -> 64 (max-pool) -> 64, 32, 16, 8 (stadi 1-4).
Convoluzioni senza bias, inizializzate con "he_normal" di Keras (normale troncata, deviazione sqrt(2 / fan_in); non
l'inizializzazione fan_out non troncata di torchvision); lo strato Dense finale usa l'inizializzatore predefinito di Keras
(Glorot uniforme, bias a zero), come lo strato output di LivenessNet. BatchNormalization con i default di Keras 2.15 (momentum 0,99, epsilon 1e-3),
gli stessi di LivenessNet, che non passa argomenti ai suoi BatchNormalization (verificato da check_model sul modello
costruito). Modello funzionale piatto: nessun sotto-modello annidato, nomi stabili per ogni strato, 20 strati
BatchNormalization al livello superiore (lo stimatore della ristima della PR #27 lavora per nome di strato).
Parametri: 11.176.512 nel tronco (convoluzioni e gamma/beta delle BN, come torchvision resnet18 senza il
classificatore) + 1.026 del Dense = 11.177.538 addestrabili, più 9.600 statistiche mobili (non addestrabili).

Il conteggio aritmetico (resnet18_spec, spec_counts, livenessnet_counts) e la topologia attesa (resnet18_topology) non
usano TensorFlow; TensorFlow è importato solo da resnet18, build, model_counts e check_model. check_model confronta la
topologia del modello costruito (per ogni strato: classe, strati in ingresso e attributi: kernel, stride, padding, bias e
inizializzatore delle convoluzioni, attivazioni, unità del Dense) con resnet18_topology, scritta qui a parte: ogni somma
seguita da una ReLU, ogni BN preceduta dalla sua convoluzione, l'ordine conv-BN-ReLU-conv-BN nei blocchi, le scorciatoie
collegate all'ingresso del blocco.
"""
from collections import Counter

NETWORKS = ("LivenessNet", "ResNet-18")
INPUT_SHAPE = (256, 256, 3)
BN_MOMENTUM, BN_EPSILON = 0.99, 1e-3             # default di tf.keras.layers.BatchNormalization in Keras 2.15
STAGES = ((1, 64), (2, 128), (3, 256), (4, 512))
BLOCKS = (1, 2)
RESNET18_TRUNK = 11_176_512
RESNET18_DENSE = 1_026
RESNET18_MOVING = 9_600
RESNET18_BN_LAYERS = 20
LIVENESSNET_BN = ("bn1_1", "bn1_2", "bn2_1", "bn2_2", "bn_dense")


class ModelProblem(Exception):
    """Modello costruito diverso dalla specifica dichiarata."""


def _problem(msg: str):
    raise ModelProblem(msg)


# ============================================================================ specifica aritmetica (senza TensorFlow)

def _block_prefix(stage: int, block: int) -> str:
    return f"layer{stage}_block{block}"


def _strided(stage: int, block: int) -> bool:
    """Riduzione con stride 2 e scorciatoia a proiezione: primo blocco degli stadi 2-4."""
    return stage > 1 and block == 1


def resnet18_spec() -> list:
    """Gli strati con parametri di ResNet-18, nell'ordine della rete: (nome, tipo, forma). conv: (k, k, ingresso,
    uscita), senza bias; bn: (canali,); dense: (ingresso, uscita), con bias."""
    out = [("conv1", "conv", (7, 7, 3, 64)), ("bn1", "bn", (64,))]
    cin = 64
    for stage, f in STAGES:
        for block in BLOCKS:
            p = _block_prefix(stage, block)
            out += [(f"{p}_conv1", "conv", (3, 3, cin, f)), (f"{p}_bn1", "bn", (f,)), (f"{p}_conv2", "conv", (3, 3, f, f)),
                    (f"{p}_bn2", "bn", (f,))]
            if _strided(stage, block):
                out += [(f"{p}_downsample_conv", "conv", (1, 1, cin, f)), (f"{p}_downsample_bn", "bn", (f,))]
            cin = f
    out.append(("output", "dense", (cin, 2)))
    return out


def spec_counts(spec: list) -> dict:
    """Conteggi dalla specifica: tronco (conv e gamma/beta), Dense, addestrabili, statistiche mobili, strati BN."""
    trunk = dense = moving = bn = 0
    for _, kind, shape in spec:
        if kind == "conv":
            k1, k2, cin, cout = shape
            trunk += k1 * k2 * cin * cout
        elif kind == "bn":
            trunk += 2 * shape[0]
            moving += 2 * shape[0]
            bn += 1
        elif kind == "dense":
            dense += shape[0] * shape[1] + shape[1]
        else:
            _problem(f"unknown layer kind {kind!r} in the specification")
    return {"trunk": trunk, "dense": dense, "trainable": trunk + dense, "non_trainable": moving, "batchnorm_layers": bn}


def resnet18_spatial(size: int) -> list:
    """Lato spaziale dopo stem, max-pool e i quattro stadi, con le formule di convoluzione e pooling usate qui."""
    s = (size + 2 * 3 - 7) // 2 + 1                 # conv 7x7 stride 2, padding 3
    out = [s]
    s = (s + 2 * 1 - 3) // 2 + 1                    # max-pool 3x3 stride 2, padding 1
    out.append(s)
    for stage, _ in STAGES:
        if stage > 1:
            s = (s + 2 * 1 - 3) // 2 + 1            # conv 3x3 stride 2, padding 1
        out.append(s)
    return out


def _conv(k: int, stride: int, padding: str) -> tuple:
    return ((k, k), (stride, stride), padding, False, "HeNormal")


def resnet18_topology() -> dict:
    """La topologia attesa di ResNet-18, senza TensorFlow: nome -> (classe, strati in ingresso nell'ordine della chiamata,
    attributi). Attributi: Conv2D (kernel, stride, padding, bias, inizializzatore), ZeroPadding2D (padding), MaxPooling2D
    (finestra, stride, padding), Activation (attivazione), Dense (unità, attivazione, inizializzatore); () altrimenti."""
    t = {"input_layer": ("InputLayer", (), ()), "conv1_pad": ("ZeroPadding2D", ("input_layer",), (((3, 3), (3, 3)),)),
         "conv1": ("Conv2D", ("conv1_pad",), _conv(7, 2, "valid")), "bn1": ("BatchNormalization", ("conv1",), ()),
         "relu1": ("Activation", ("bn1",), ("relu",)), "pool1_pad": ("ZeroPadding2D", ("relu1",), (((1, 1), (1, 1)),)),
         "pool1": ("MaxPooling2D", ("pool1_pad",), ((3, 3), (2, 2), "valid"))}
    x = "pool1"
    for stage, _ in STAGES:
        for block in BLOCKS:
            p = _block_prefix(stage, block)
            if _strided(stage, block):
                t[f"{p}_pad"] = ("ZeroPadding2D", (x,), (((1, 1), (1, 1)),))
                t[f"{p}_conv1"] = ("Conv2D", (f"{p}_pad",), _conv(3, 2, "valid"))
            else:
                t[f"{p}_conv1"] = ("Conv2D", (x,), _conv(3, 1, "same"))
            t[f"{p}_bn1"] = ("BatchNormalization", (f"{p}_conv1",), ())
            t[f"{p}_relu1"] = ("Activation", (f"{p}_bn1",), ("relu",))
            t[f"{p}_conv2"] = ("Conv2D", (f"{p}_relu1",), _conv(3, 1, "same"))
            t[f"{p}_bn2"] = ("BatchNormalization", (f"{p}_conv2",), ())
            short = x
            if _strided(stage, block):
                t[f"{p}_downsample_conv"] = ("Conv2D", (x,), _conv(1, 2, "valid"))
                t[f"{p}_downsample_bn"] = ("BatchNormalization", (f"{p}_downsample_conv",), ())
                short = f"{p}_downsample_bn"
            t[f"{p}_add"] = ("Add", (f"{p}_bn2", short), ())
            t[f"{p}_relu2"] = ("Activation", (f"{p}_add",), ("relu",))
            x = f"{p}_relu2"
    t["avgpool"] = ("GlobalAveragePooling2D", (x,), ())
    t["output"] = ("Dense", ("avgpool",), (2, "softmax", "GlorotUniform"))
    return t


def _tuples(v):
    return tuple(_tuples(x) for x in v) if isinstance(v, (list, tuple)) else v


def model_topology(model) -> dict:
    """La topologia di un modello funzionale costruito, dalla sua get_config() (stessa forma di resnet18_topology)."""
    out = {}
    for l in model.get_config()["layers"]:
        c, k = l["config"], l["class_name"]
        nodes = l.get("inbound_nodes") or []
        ins = tuple(n[0] for n in nodes[0]) if nodes else ()
        if k == "Conv2D":
            a = (_tuples(c["kernel_size"]), _tuples(c["strides"]), c["padding"], c["use_bias"], c["kernel_initializer"]["class_name"])
        elif k == "ZeroPadding2D":
            a = (_tuples(c["padding"]),)
        elif k == "MaxPooling2D":
            a = (_tuples(c["pool_size"]), _tuples(c["strides"]), c["padding"])
        elif k == "Activation":
            a = (c["activation"],)
        elif k == "Dense":
            a = (c["units"], c["activation"], c["kernel_initializer"]["class_name"])
        else:
            a = ()
        out[l["name"]] = (k, ins, a)
    return out


def livenessnet_counts(input_shape) -> dict:
    """Conteggi aritmetici di LivenessNet (architectures.py: conv 3x3 con bias, due max-pool 2x2, Dense 64, cinque BN)
    per un ingresso (H, W, 3) con H e W multipli di 4."""
    h, w, c = input_shape
    if h % 4 or w % 4:
        _problem("LivenessNet needs an input side divisible by 4 (two 2x2 max-pools)")
    convs = [(c, 16), (16, 16), (16, 32), (32, 32)]
    trainable = sum(9 * a * b + b for a, b in convs)
    flat = (h // 4) * (w // 4) * 32
    trainable += flat * 64 + 64 + 64 * 2 + 2
    bn_channels = (16, 16, 32, 32, 64)
    trainable += 2 * sum(bn_channels)
    return {"trainable": trainable, "non_trainable": 2 * sum(bn_channels), "batchnorm_layers": len(bn_channels)}


def expected_counts(network: str, input_shape) -> dict:
    """Parametri addestrabili, statistiche mobili e strati BN attesi per una rete del registro."""
    if network == "LivenessNet":
        return livenessnet_counts(input_shape)
    if network == "ResNet-18":
        c = spec_counts(resnet18_spec())
        return {"trainable": c["trainable"], "non_trainable": c["non_trainable"], "batchnorm_layers": c["batchnorm_layers"]}
    _problem(f"unknown network {network!r}: only {', '.join(NETWORKS)}")


# ============================================================================ costruzione (TensorFlow)

def resnet18(input_shape=INPUT_SHAPE):
    """ResNet-18 v1 come descritta nel docstring del modulo; ingresso float32 in [0, 1] (frame / 255, come LivenessNet)."""
    import tensorflow as tf
    L = tf.keras.layers

    def conv(x, filters, k, stride, padding, name):
        return L.Conv2D(filters, k, strides=stride, padding=padding, use_bias=False, kernel_initializer="he_normal", name=name)(x)

    inputs = L.Input(shape=tuple(input_shape), name="input_layer")
    x = L.ZeroPadding2D(3, name="conv1_pad")(inputs)
    x = conv(x, 64, 7, 2, "valid", "conv1")
    x = L.BatchNormalization(name="bn1")(x)
    x = L.Activation("relu", name="relu1")(x)
    x = L.ZeroPadding2D(1, name="pool1_pad")(x)
    x = L.MaxPooling2D(3, strides=2, name="pool1")(x)
    for stage, f in STAGES:
        for block in BLOCKS:
            p = _block_prefix(stage, block)
            if _strided(stage, block):
                y = L.ZeroPadding2D(1, name=f"{p}_pad")(x)
                y = conv(y, f, 3, 2, "valid", f"{p}_conv1")
            else:
                y = conv(x, f, 3, 1, "same", f"{p}_conv1")
            y = L.BatchNormalization(name=f"{p}_bn1")(y)
            y = L.Activation("relu", name=f"{p}_relu1")(y)
            y = conv(y, f, 3, 1, "same", f"{p}_conv2")
            y = L.BatchNormalization(name=f"{p}_bn2")(y)
            shortcut = x
            if _strided(stage, block):
                shortcut = conv(x, f, 1, 2, "valid", f"{p}_downsample_conv")
                shortcut = L.BatchNormalization(name=f"{p}_downsample_bn")(shortcut)
            x = L.Add(name=f"{p}_add")([y, shortcut])
            x = L.Activation("relu", name=f"{p}_relu2")(x)
    x = L.GlobalAveragePooling2D(name="avgpool")(x)
    outputs = L.Dense(2, activation="softmax", name="output")(x)
    return tf.keras.Model(inputs, outputs, name="ResNet18")


def build(network: str, input_shape, arch_module=None):
    """Il modello di una rete del registro. LivenessNet: create_model("LivenessNet", input_shape=...).get_model() del
    codice di riferimento (con input_shape = (256, 256, 3) è la chiamata create_model("LivenessNet") del trainer dei run
    accettati, che usa lo stesso valore come default); ResNet-18: resnet18."""
    if network == "LivenessNet":
        if arch_module is None:
            _problem("LivenessNet needs the reference architecture module")
        return arch_module.create_model("LivenessNet", input_shape=tuple(input_shape)).get_model()
    if network == "ResNet-18":
        return resnet18(input_shape)
    _problem(f"unknown network {network!r}: only {', '.join(NETWORKS)}")


def model_counts(model) -> dict:
    """Parametri addestrabili e non addestrabili e strati BatchNormalization di un modello costruito."""
    import tensorflow as tf
    tr = sum(int(tf.keras.backend.count_params(w)) for w in model.trainable_weights)
    nt = sum(int(tf.keras.backend.count_params(w)) for w in model.non_trainable_weights)
    bn = [l.name for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    return {"trainable": tr, "non_trainable": nt, "batchnorm_layers": len(bn), "batchnorm_names": bn}


def check_model(model, network: str, input_shape, expected: dict) -> dict:
    """Un modello costruito contro il registro e la configurazione: piatto (nessun sotto-modello), nomi unici, ingresso,
    uscita Dense(2, softmax) di nome output, BatchNormalization con momentum ed epsilon di Keras, conteggi uguali
    all'aritmetica e alla configurazione (expected: trainable, non_trainable, batchnorm_layers); per ResNet-18 la topologia
    uguale a resnet18_topology; per LivenessNet i cinque strati BN con i nomi del codice di riferimento."""
    import tensorflow as tf
    names = [l.name for l in model.layers]
    if any(isinstance(l, tf.keras.Model) for l in model.layers):
        _problem(f"{network}: nested models are refused (the BN re-estimation works by layer name on the top level)")
    if len(set(names)) != len(names):
        _problem(f"{network}: duplicated layer names {sorted(n for n, k in Counter(names).items() if k > 1)}")
    if tuple(model.input_shape[1:]) != tuple(input_shape):
        _problem(f"{network}: input shape {model.input_shape[1:]} differs from {tuple(input_shape)}")
    out = model.layers[-1]
    if out.name != "output" or not isinstance(out, tf.keras.layers.Dense) or out.units != 2 or out.activation.__name__ != "softmax":
        _problem(f"{network}: the last layer must be Dense(2, softmax) named output")
    for l in model.layers:
        if isinstance(l, tf.keras.layers.BatchNormalization) and (l.momentum != BN_MOMENTUM or l.epsilon != BN_EPSILON):
            _problem(f"{network}: {l.name} has momentum {l.momentum} and epsilon {l.epsilon}, not the Keras defaults "
                     f"{BN_MOMENTUM} and {BN_EPSILON}")
    got = model_counts(model)
    arith = expected_counts(network, input_shape)
    for k in ("trainable", "non_trainable", "batchnorm_layers"):
        if got[k] != arith[k] or got[k] != expected[k]:
            _problem(f"{network}: {k} {got[k]} differs from the arithmetic {arith[k]} or the configuration {expected[k]}")
    if network == "ResNet-18":
        topo, want = model_topology(model), resnet18_topology()
        if topo != want:
            bad = sorted(n for n in set(topo) | set(want) if topo.get(n) != want.get(n))
            _problem(f"{network}: the topology differs from resnet18_topology at {', '.join(bad[:5])}")
    if network == "LivenessNet" and tuple(got["batchnorm_names"]) != LIVENESSNET_BN:
        _problem(f"{network}: BatchNormalization layers {got['batchnorm_names']} differ from {list(LIVENESSNET_BN)}")
    return got
