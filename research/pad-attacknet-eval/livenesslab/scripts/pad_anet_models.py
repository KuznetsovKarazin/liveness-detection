"""
Registro delle tre AttackNet del confronto C5 (scheda C5 del coordinatore, 21/11/2026: «Usi da zero le implementazioni
del progetto AttackNetV1, AttackNetV2_1 e AttackNetV2_2»): AttackNetV1, AttackNetV2_1 e AttackNetV2_2 del codice di
riferimento del coordinatore (src/livedetection/src/architectures.py, lo stesso file fissato da C2 per LivenessNet,
SHA-256 nella configurazione), costruite solo con create_model(nome, input_shape=...) e i suoi valori predefiniti
(dropout_rate 0,5, l2_reg 0,0), come C2 ha costruito LivenessNet (pad_arch_models.build): nessun peso pre-addestrato.

Perché solo create_model: BaseArchitecture ha l2_reg=0.01 come valore predefinito e get_all_architectures lo eredita,
mentre create_model passa l2_reg=0.0; con 0,0 AttackNetV1 e AttackNetV2_1 hanno reg = None (nessun regolarizzatore) e
AttackNetV2_2 non ne ha in nessun caso. check_model verifica sul modello costruito che non ci sia alcun regolarizzatore
(kernel, bias, attività) e nessuna perdita aggiuntiva (model.losses vuoto).

La specifica di ciascuna rete (layer_spec) è scritta qui a parte, strato per strato, dalle tre build_model del codice
di riferimento: nome, classe, strati in ingresso nell'ordine della chiamata, attributi nella forma di
pad_arch_models.model_topology (importata così com'è), attributi in più (tasso del Dropout, alpha della LeakyReLU,
regolarizzatori, asse e parametri delle BatchNormalization), forma d'uscita e parametri. Da questa sola specifica si
ricavano, senza TensorFlow: la topologia attesa (topology), i conteggi aritmetici (expected_counts), gli strati
BatchNormalization nell'ordine della rete con canali e lato spaziale del loro ingresso (batchnorm_layers), quindi il
numero atteso di valori per canale della ristima (frame × H × W) e l'ordine della ristima (bn_order). check_model
confronta il modello costruito con tutto questo.

  rete           fusione        testa                                                  dropout delle fasi conv
  AttackNetV1    Concatenate    Dense(128, tanh) "dense1"                              0,25 (0,5 × 0,5)
  AttackNetV2_1  Concatenate    Dense(128) "dense1_linear" + Activation("tanh")        0,5
  AttackNetV2_2  Add            Dense(128, tanh) "dense1"                              0,25

Con ingresso 256×256×3: AttackNetV1 e AttackNetV2_1 33.588.194 parametri addestrabili e 544 statistiche mobili;
AttackNetV2_2 16.806.274 e 448; cinque BatchNormalization ciascuna (bn1_1, bn1_final, bn2_1, bn2_final, bn_dense).

TensorFlow è importato solo da build, check_model e layer_extras (sul modello costruito); il resto è aritmetica.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pad_arch_models as AM  # noqa: E402

NETWORKS = ("AttackNetV1", "AttackNetV2_1", "AttackNetV2_2")
INPUT_SHAPE = (256, 256, 3)
BN_MOMENTUM, BN_EPSILON = AM.BN_MOMENTUM, AM.BN_EPSILON     # default di Keras 2.15; le AttackNet non passano argomenti
DROPOUT_RATE, L2_REG = 0.5, 0.0                            # i valori predefiniti di create_model
LEAKY_ALPHA = 0.2
MERGE = {"AttackNetV1": "Concatenate", "AttackNetV2_1": "Concatenate", "AttackNetV2_2": "Add"}
CONV_DROPOUT_FACTOR = {"AttackNetV1": 0.5, "AttackNetV2_1": 1.0, "AttackNetV2_2": 0.5}   # dropout1, dropout2 = fattore × dropout_rate
LINEAR_HEAD = {"AttackNetV1": False, "AttackNetV2_1": True, "AttackNetV2_2": False}
CONSTRUCTOR = "create_model(name, input_shape=(256, 256, 3)) with its defaults dropout_rate=0.5 and l2_reg=0.0"
NO_REGULARIZER = (None, None, None)


class ModelProblem(Exception):
    """Modello costruito diverso dalla specifica dichiarata."""


def _problem(msg: str):
    raise ModelProblem(msg)


def _known(network: str) -> None:
    if network not in NETWORKS:
        _problem(f"unknown network {network!r}: only {', '.join(NETWORKS)}")


# ============================================================================ specifica (senza TensorFlow)

def _conv_attrs() -> tuple:
    """Conv2D(f, (3, 3), padding="same") del codice di riferimento: stride 1, bias, inizializzatore predefinito."""
    return ((3, 3), (1, 1), "same", True, "GlorotUniform")


def layer_spec(network: str, input_shape=INPUT_SHAPE) -> list:
    """Gli strati di una AttackNet nell'ordine della build_model del codice di riferimento: dizionari con name, cls,
    inbound (strati in ingresso nell'ordine della chiamata), attrs (forma di AM.model_topology), extras (layer_extras),
    shape (forma d'uscita senza il batch), trainable e non_trainable (parametri). Ingresso (H, W, 3) con H e W multipli
    di 4 (due max-pool 2x2)."""
    _known(network)
    h, w, c = (int(x) for x in input_shape)
    if h % 4 or w % 4 or h <= 0 or w <= 0:
        _problem(f"{network} needs input sides divisible by 4 (two 2x2 max-pools), got {tuple(input_shape)}")
    merge = MERGE[network]
    drop = CONV_DROPOUT_FACTOR[network] * DROPOUT_RATE
    out = []

    def add(name, cls, inbound, attrs=(), extras=(), shape=None, trainable=0, non_trainable=0):
        out.append({"name": name, "cls": cls, "inbound": tuple(inbound), "attrs": attrs, "extras": extras, "shape": shape,
                    "trainable": trainable, "non_trainable": non_trainable})

    def conv(name, x, cin, cout, hw):
        add(name, "Conv2D", (x,), _conv_attrs(), ("regularizers", NO_REGULARIZER), (hw[0], hw[1], cout), 9 * cin * cout + cout)

    def leaky(name, x, shape):
        add(name, "LeakyReLU", (x,), (), ("alpha", float(np.float32(LEAKY_ALPHA))), shape)

    def bn(name, x, shape):
        ch = shape[-1]
        add(name, "BatchNormalization", (x,), (), ("batchnorm", "last", BN_MOMENTUM, BN_EPSILON, True, True), shape, 2 * ch, 2 * ch)

    add("input_layer", "InputLayer", (), shape=(h, w, c))
    x, cin, hw = "input_layer", c, (h, w)
    for stage, f in ((1, 16), (2, 32)):
        p = f"{stage}"
        conv(f"conv{p}_1", x, cin, f, hw)
        leaky(f"leaky_relu{p}_1", f"conv{p}_1", (hw[0], hw[1], f))
        bn(f"bn{p}_1", f"leaky_relu{p}_1", (hw[0], hw[1], f))
        conv(f"conv{p}_2", f"bn{p}_1", f, f, hw)
        leaky(f"leaky_relu{p}_2", f"conv{p}_2", (hw[0], hw[1], f))
        conv(f"conv{p}_3", f"leaky_relu{p}_2", f, f, hw)
        leaky(f"leaky_relu{p}_3", f"conv{p}_3", (hw[0], hw[1], f))
        merged = 2 * f if merge == "Concatenate" else f
        mname = f"concat{p}" if merge == "Concatenate" else f"add{p}"
        add(mname, merge, (f"conv{p}_1", f"leaky_relu{p}_3"), shape=(hw[0], hw[1], merged))
        bn(f"bn{p}_final", mname, (hw[0], hw[1], merged))
        hw = (hw[0] // 2, hw[1] // 2)
        add(f"maxpool{p}", "MaxPooling2D", (f"bn{p}_final",), ((2, 2), (2, 2), "valid"), shape=(hw[0], hw[1], merged))
        add(f"dropout{p}", "Dropout", (f"maxpool{p}",), (), ("rate", drop), (hw[0], hw[1], merged))
        x, cin = f"dropout{p}", merged
    flat = hw[0] * hw[1] * cin
    add("flatten", "Flatten", (x,), shape=(flat,))
    if LINEAR_HEAD[network]:
        add("dense1_linear", "Dense", ("flatten",), (128, "linear", "GlorotUniform"), ("regularizers", NO_REGULARIZER), (128,), flat * 128 + 128)
        add("tanh_activation", "Activation", ("dense1_linear",), ("tanh",), (), (128,))
        head = "tanh_activation"
    else:
        add("dense1", "Dense", ("flatten",), (128, "tanh", "GlorotUniform"), ("regularizers", NO_REGULARIZER), (128,), flat * 128 + 128)
        head = "dense1"
    bn("bn_dense", head, (128,))
    add("dropout_dense", "Dropout", ("bn_dense",), (), ("rate", DROPOUT_RATE), (128,))
    add("output", "Dense", ("dropout_dense",), (2, "softmax", "GlorotUniform"), ("regularizers", NO_REGULARIZER), (2,), 128 * 2 + 2)
    return out


def topology(network: str, input_shape=INPUT_SHAPE) -> dict:
    """La topologia attesa nella forma di AM.model_topology: nome -> (classe, strati in ingresso, attributi)."""
    return {l["name"]: (l["cls"], l["inbound"], l["attrs"]) for l in layer_spec(network, input_shape)}


def expected_extras(network: str, input_shape=INPUT_SHAPE) -> dict:
    """Gli attributi in più attesi (layer_extras) per gli strati che ne hanno."""
    return {l["name"]: l["extras"] for l in layer_spec(network, input_shape) if l["extras"]}


def expected_counts(network: str, input_shape=INPUT_SHAPE) -> dict:
    """Parametri addestrabili, statistiche mobili e strati BN dalla specifica (aritmetica, senza TensorFlow)."""
    spec = layer_spec(network, input_shape)
    return {"trainable": sum(l["trainable"] for l in spec), "non_trainable": sum(l["non_trainable"] for l in spec),
            "batchnorm_layers": sum(1 for l in spec if l["cls"] == "BatchNormalization")}


def batchnorm_layers(network: str, input_shape=INPUT_SHAPE) -> list:
    """Gli strati BatchNormalization nell'ordine della rete: nome, canali, valori per frame e per canale dell'ingresso
    (H × W per uno strato convoluzionale, 1 per il denso) e tipo (conv o dense)."""
    out = []
    for l in layer_spec(network, input_shape):
        if l["cls"] == "BatchNormalization":
            s = l["shape"]
            out.append({"name": l["name"], "channels": int(s[-1]), "spatial": int(np.prod(s[:-1])) if len(s) > 1 else 1,
                        "kind": "conv" if len(s) > 1 else "dense"})
    return out


def bn_order(network: str) -> list:
    """L'ordine della ristima: gli strati BN nell'ordine della rete (la catena è lineare fra le BN: ogni BN dipende da
    tutte le precedenti, quindi è l'unico ordine topologico; il controllo sul grafo è pad_arch_run.bn_order_problems)."""
    return [b["name"] for b in batchnorm_layers(network)]


def expected_recalibration_counts(network: str, frames: int, input_shape=INPUT_SHAPE) -> dict:
    """Valori per canale attesi da nuaa_bn_recalibration.recalibrate per ogni strato BN: frame × H × W (frame per il
    denso)."""
    return {b["name"]: int(frames) * b["spatial"] for b in batchnorm_layers(network, input_shape)}


# ============================================================================ costruzione e controllo (TensorFlow)

def build(network: str, input_shape, arch_module):
    """Il modello di una AttackNet: create_model(network, input_shape=...).get_model() del codice di riferimento, con i
    valori predefiniti di create_model (dropout_rate 0,5, l2_reg 0,0). Nessun altro costruttore."""
    _known(network)
    if arch_module is None:
        _problem(f"{network} needs the reference architecture module")
    return arch_module.create_model(network, input_shape=tuple(input_shape)).get_model()


def _axis_last(axis, rank: int) -> bool:
    a = axis if isinstance(axis, (list, tuple)) else [axis]
    return list(a) in ([rank - 1], [-1])


def layer_extras(model) -> dict:
    """Gli attributi in più di un modello costruito, dalla get_config() di ciascuno strato: Dropout ("rate", tasso);
    LeakyReLU ("alpha", alpha); Conv2D e Dense ("regularizers", (kernel, bias, attività)); BatchNormalization
    ("batchnorm", "last" se l'asse è l'ultimo, momentum, epsilon, center, scale)."""
    out = {}
    for l in model.layers:
        c, k = l.get_config(), type(l).__name__
        if k == "Dropout":
            out[l.name] = ("rate", c["rate"])
        elif k == "LeakyReLU":
            out[l.name] = ("alpha", c["alpha"])
        elif k in ("Conv2D", "Dense"):
            out[l.name] = ("regularizers", tuple(c.get(x) for x in ("kernel_regularizer", "bias_regularizer", "activity_regularizer")))
        elif k == "BatchNormalization":
            out[l.name] = ("batchnorm", "last" if _axis_last(c["axis"], len(l.input_shape)) else c["axis"], c["momentum"], c["epsilon"],
                           c["center"], c["scale"])
    return out


def moving_sizes(model) -> dict:
    """Per ogni BatchNormalization del livello superiore il numero di valori di moving_mean e moving_variance."""
    import tensorflow as tf
    return {l.name: int(np.prod(l.moving_mean.shape)) + int(np.prod(l.moving_variance.shape))
            for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)}


def check_model(model, network: str, input_shape, expected: dict) -> dict:
    """Un modello costruito (o ricaricato) contro il registro e la configurazione: piatto (nessun sotto-modello), nomi
    unici, nome del modello, ingresso, uscita Dense(2, softmax) di nome output; conteggi (AM.model_counts) uguali
    all'aritmetica della specifica e alla configurazione (expected: trainable, non_trainable, batchnorm_layers);
    topologia (AM.model_topology) uguale a topology; attributi in più uguali a expected_extras (tassi di dropout, alpha,
    nessun regolarizzatore, BN sull'ultimo asse con momentum ed epsilon di Keras, gamma e beta); nessuna perdita
    aggiuntiva; strati BN con i nomi e nell'ordine del registro; le statistiche mobili delle BN del livello superiore
    sono tutte le variabili non addestrabili (nessuna BN fuori dal livello superiore) e ogni ingresso di BN ha la forma
    della specifica. Restituisce i conteggi."""
    import tensorflow as tf
    _known(network)
    names = [l.name for l in model.layers]
    if any(isinstance(l, tf.keras.Model) for l in model.layers):
        _problem(f"{network}: nested models are refused (the BN re-estimation works by layer name on the top level)")
    if len(set(names)) != len(names):
        _problem(f"{network}: duplicated layer names {sorted(n for n, k in Counter(names).items() if k > 1)}")
    if model.name != network:
        _problem(f"{network}: the model is named {model.name!r}")
    if tuple(model.input_shape[1:]) != tuple(input_shape):
        _problem(f"{network}: input shape {model.input_shape[1:]} differs from {tuple(input_shape)}")
    out = model.layers[-1]
    if out.name != "output" or not isinstance(out, tf.keras.layers.Dense) or out.units != 2 or out.activation.__name__ != "softmax":
        _problem(f"{network}: the last layer must be Dense(2, softmax) named output")
    got = AM.model_counts(model)
    arith = expected_counts(network, input_shape)
    for k in ("trainable", "non_trainable", "batchnorm_layers"):
        if got[k] != arith[k] or got[k] != expected[k]:
            _problem(f"{network}: {k} {got[k]} differs from the arithmetic {arith[k]} or the configuration {expected[k]}")
    topo, want = AM.model_topology(model), topology(network, input_shape)
    if topo != want:
        bad = sorted(n for n in set(topo) | set(want) if topo.get(n) != want.get(n))
        _problem(f"{network}: the topology differs from the registry at {', '.join(bad[:5])}")
    ex, want_ex = layer_extras(model), expected_extras(network, input_shape)
    if ex != want_ex:
        bad = sorted(n for n in set(ex) | set(want_ex) if ex.get(n) != want_ex.get(n))
        _problem(f"{network}: dropout rates, alpha, regularizers or BatchNormalization settings differ from the registry at {', '.join(bad[:5])}")
    if model.losses:
        _problem(f"{network}: {len(model.losses)} additional losses (a regularizer): none expected with l2_reg 0.0")
    bns = batchnorm_layers(network, input_shape)
    if got["batchnorm_names"] != [b["name"] for b in bns]:
        _problem(f"{network}: BatchNormalization layers {got['batchnorm_names']} differ from {[b['name'] for b in bns]}")
    sizes = moving_sizes(model)
    if sum(sizes.values()) != got["non_trainable"] or any(sizes[b["name"]] != 2 * b["channels"] for b in bns):
        _problem(f"{network}: the moving statistics of the top-level BatchNormalization layers ({sum(sizes.values())}) are not all the "
                 f"non-trainable variables ({got['non_trainable']}) or differ from the channels of the registry")
    spec_shapes = {l["name"]: l["shape"] for l in layer_spec(network, input_shape)}
    for b in bns:
        layer = model.get_layer(b["name"])
        if tuple(layer.input_shape[1:]) != tuple(spec_shapes[b["name"]]):
            _problem(f"{network}: {b['name']} has input shape {layer.input_shape[1:]}, the registry {spec_shapes[b['name']]}")
    return {**got, "moving_values": sum(sizes.values())}
