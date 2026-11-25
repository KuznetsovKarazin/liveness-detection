"""
La ristima BN 2x2 del follow-up registrato della C2 (delivery/pad-arch-score-followup/PROTOCOL.md §4), solo su fitting e
development: le statistiche mobili delle BatchNormalization dei checkpoint salvati di LivenessNet e ResNet-18 (run
20261120-ARCHRUN-p4 e -p12) sono ristimate sui frame di fitting della loro unità in quattro celle, augmentation sì/no x
dropout sì/no durante la sola ristima, e ogni cella è valutata sui frame di development con augmentation e dropout
spenti. Nessun passo di gradiente, nessuno stato dell'ottimizzatore, nessun peso addestrato modificato, nessun .keras
riscritto; nessun frame di test assemblato, nessun punteggio o metrica del test letto. Il comando che applica i nuovi
stati ai dati tenuti da parte non è in questo file: si scrive e si registra dopo la prova (NOTE.md §5).

Celle (PROTOCOL.md §4): E00 ingresso pulito / dropout spento (lo stimatore esistente, nuaa_bn_recalibration.recalibrate,
invariato, una passata pulita); E10 ingresso aumentato / dropout spento; E01 ingresso pulito / dropout a monte acceso;
E11 ingresso aumentato / dropout a monte acceso. Le celle stocastiche usano otto viste per frame di fitting (semi delle
viste 101-108), unite con l'unione esatta in float64 di conteggi, medie e scarti quadratici dello stimatore esistente
(nuaa_bn_recalibration.Moments), viste all'esterno e batch all'interno.

Augmentation: una riscrittura esplicita e senza stato della ricetta di addestramento (pad_arch_comparison.trial_trainer,
la Sequential RandomFlip("horizontal"), RandomRotation(0.03), RandomZoom(0.12, 0.12), RandomTranslation(0.06, 0.06),
RandomBrightness(0.15, value_range=(0.0, 1.0)), RandomContrast(0.2), GaussianNoise(0.02), applicata dopo la divisione per
255): stesse operazioni, stesso ordine, stesse clip, con le stesse funzioni di Keras (transform, get_rotation_matrix,
get_zoom_matrix, get_translation_matrix, tf.image.adjust_contrast, la formula del flip a batch di
tf.image.stateless_random_flip_left_right); gli estremi degli intervalli sono ricavati da RECIPE (recipe_ranges), che a sua
volta è confrontata con il sorgente fissato letto con l'AST prima di ogni calcolo. Cambiano solo le estrazioni casuali:
ogni parametro viene da un generatore Philox di NumPy con chiave (dominio, seme della vista, SHA-256 della chiave del
frame, operazione), quindi la vista di un frame non dipende dal batch, dalla posizione né dagli altri frame. Differenza
dichiarata: il fattore di contrasto è estratto per frame (Keras 2.15 ne estrae uno per batch di 16); la distribuzione
marginale è la stessa.

Dropout: LivenessNet deve essere una catena (ogni strato alimentato solo dal precedente, nessun ramo, gli strati Dropout
esattamente quelli dichiarati, con i rate del checkpoint e noise_shape None); nelle celle con dropout la catena gira strato
per strato con training=False per ogni strato, BatchNormalization compresa, e i Dropout a monte (quelli prima di almeno una
BatchNormalization: dropout1, dropout2) sono sostituiti da un dropout invertito con maschere senza stato con chiave (seme
della vista, frame, strato): si tiene se u >= rate e i valori tenuti sono scalati per 1 / (1 - rate), come tf.nn.dropout.
dropout_dense (dopo l'ultima BN) non è mai attivo. ResNet-18 non deve contenere strati Dropout o di rumore (altrimenti
FAIL).

Riuso (PROTOCOL.md §4, rilievo 7 della revisione del piano): uno strato con a monte le stesse operazioni stocastiche attive
della cella di riferimento (E01 -> E00, E11 -> E10) ne prende le statistiche; vale per bn1_1 e bn1_2 di LivenessNet in E01
ed E11 e per tutti gli strati di ResNet-18 in E01 ed E11. Con duplicate_check gli stessi strati sono calcolati anche
esplicitamente (otto copie pulite identiche, o le stesse otto viste aumentate) e confrontati con il riferimento entro la
tolleranza di riferimento; un duplicato fuori tolleranza ferma il comando (FAIL); la cella riportata usa sempre il riuso.

Controlli, in quest'ordine per ogni checkpoint (ogni fallimento ferma il comando come FAIL con failed_check, il controllo e
i suoi valori, in status.json; nessuna cella stocastica è riportata): SHA-256 e architettura del checkpoint; topologia;
dev ordinario contro _dev_ordinary.csv salvato; E00 contro _bn_statistics.json e _dev_re-estimated.csv salvati
(statistiche entro rel + abs, punteggi del dev entro 1e-5, nessuna decisione discordante a 0,5 né alla soglia EER salvata,
valore della soglia ricalcolata entro 1e-5); per LivenessNet E00 lungo la catena; poi le celle stocastiche, ciascuna con i
controlli della PR #27 (solo i tensori mobili cambiati, coerenza di un passaggio a più uscite sulle stesse viste e
maschere, conteggi, softmax), viste finite ed entro i limiti dichiarati, frequenza degli zeri delle maschere entro la banda
binomiale dichiarata, duplicati espliciti, metà Monte Carlo (viste 1-4 e 5-8, due ristime separate), una ripetizione
identica bit per bit (con il primo batch di ogni vista rigenerato e confrontato), e alla fine il modello originale e il
file del checkpoint invariati.

Protezione del test: i file letti sono in TEST_READ; un hook di audit su ogni open di Python confronta il file aperto per
identità (os.stat: dispositivo e inode del file e di ogni cartella che lo contiene; nomi senza distinzione fra maiuscole e
minuscole) e rifiuta: le cartelle vietate (results/archeval della radice del repository e della radice degli ingressi, più
ogni --forbid-outside); i nomi dei manifest di test dei fold e dei file di punteggi di test; ogni file con più di un
collegamento fisico fuori dall'installazione di Python e dalle cartelle di sistema; sotto results/ e data/ ogni file che
non sta in un ingresso dichiarato o nella cartella del run; fuori dal repository e dalla radice degli ingressi ogni file
che non sta nell'installazione di Python, nelle cartelle di sistema, nella cartella temporanea, nella cartella di
configurazione di Keras o nella radice d'uscita. Memoria: la memoria residente di picco è controllata a ogni passo contro
memory.peak_rss_limit_gb (FAIL sopra). Le viste di un fold stanno in un numpy.memmap in una cartella nuova dentro la
cartella temporanea del processo (fuori dal repository, controllato), dopo un controllo dello spazio libero; la cartella è
tolta alla fine del fold e a ogni arresto (segnali compresi); una cartella lasciata da una prova precedente ferma la
partenza.

Esiti: PASS (0), FAIL (1: errore, controllo fallito, SIGTERM, SIGHUP, memoria, spazio su disco; uscite pubbliche tolte,
status.json con la ragione e, per un controllo, failed_check), INCOMPLETE (2: tetto di tempo o SIGUSR1; unità complete
conservate).

  .venv/bin/python -I -B scripts/pad_arch_bn_factorial.py check-inputs [--config delivery/archbn2x2/pad_arch_bn_factorial.json]
  nohup caffeinate -dims .venv/bin/python -I -B scripts/pad_arch_bn_factorial.py trial --i-am-authorized > <log fuori dal repository> 2>&1 &
  .venv/bin/python -I -B scripts/pad_arch_bn_factorial.py measure     # tempi sintetici: nessun dato letto

Uscite (results/archbn2x2/<YYYYMMDD>-ARCHBN2X2-<ambito>-<commit>[-r<k>], ambito pilot, p4-24, p12-24 o custom; la regola dei
tentativi conta le cartelle dello stesso ambito): status.json, run.log; public/ controls.csv, cells.csv,
effects.csv, montecarlo.csv, layers.csv, timing.csv, summary.json, report.md, SHA256SUMS (solo aggregati; con scope.report_cell_metrics
false le metriche delle celle vanno solo in restricted/); restricted/ <unità>_<cella>_<viste>_dev.csv (punteggi del dev
per frame), <unità>_<cella>_<viste>_bn.json (statistiche per strato), sealed_cells.csv, sealed_effects.csv e sealed_montecarlo.csv (solo con le
metriche non pubbliche), SHA256SUMS. Viste e maschere non sono mai scritte fuori dal memmap temporaneo.
Dipendenze: NumPy; TensorFlow e Keras 2.15 (solo trial e measure, solo CPU). Formati: configurazione
pad-arch-bn-factorial-config/1, run pad-arch-bn-factorial/1.
"""
import argparse
import ast
import fnmatch
import gc
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import m2_writers as mw  # noqa: E402
import nuaa_bn_recalibration as BR  # noqa: E402
import nuaa_folds_run as FR  # noqa: E402
import nuaa_session_ablation_run as SR  # noqa: E402
import pad_arch_comparison as PAC  # noqa: E402
import pad_arch_definitions as D  # noqa: E402
import pad_arch_models as AM  # noqa: E402
import pad_arch_run as PR  # noqa: E402
import path_guard as pg  # noqa: E402

ROOT = PR.ROOT                      # la radice del repository dei moduli riusati (questo file sta nella stessa scripts/)
Failure, fail = FR.Failure, FR.fail
sha256_bytes, sha256_file, read_regular, rel_path, csv_text, parse_csv = (FR.sha256_bytes, FR.sha256_file, FR.read_regular,
                                                                          FR.rel_path, FR.csv_text, FR.parse_csv)
now = FR.now

DEFAULT_CONFIG = "delivery/archbn2x2/pad_arch_bn_factorial.json"
DEFAULT_OUT_ROOT = "results/archbn2x2"
CONFIG_SCHEMA = "pad-arch-bn-factorial-config/1"
RUN_SCHEMA = "pad-arch-bn-factorial/1"
SCOPES = ("pilot", "p4-24", "p12-24", "custom")
RUN_ID_RE = re.compile(r"^\d{8}-ARCHBN2X2-(pilot|p4-24|p12-24|custom)-[0-9a-f]{7,40}(?:-r([2-9]|[1-9]\d+))?$")
LOOKS_LIKE_RUN = re.compile(r"^\d{8}-ARCHBN2X2-", re.IGNORECASE)
STATUS_CODE = {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}
STATUS, LOG, PUBLIC, RESTRICTED = FR.STATUS, FR.LOG, "public", "restricted"
NETWORKS = AM.NETWORKS
PARENT_RUNS = {"nuaa_patience_4": 4, "nuaa_patience_12": 12}
CELLS = ("E00", "E10", "E01", "E11")
FACTORS = {"E00": (False, False), "E10": (True, False), "E01": (False, True), "E11": (True, True)}   # (augmentation, dropout)
REFERENCE = {"E01": "E00", "E11": "E10"}        # la cella da cui uno strato riusabile prende le statistiche
VIEW_SEEDS = tuple(range(101, 109))
HALVES = ((101, 102, 103, 104), (105, 106, 107, 108))
ALL_VIEWS = "all"
SEED_NAMESPACE = "pad-arch-bn-factorial/1"      # separa questi flussi da ogni altro uso delle stesse chiavi
SEED_DOMAINS = ("augmentation", "dropout")
VIEWS_PREFIX = "archbn2x2-views-"               # prefisso della cartella temporanea delle viste
# la ricetta di addestramento (pad_arch_comparison.trial_trainer): classe, argomenti posizionali, argomenti con nome
RECIPE = (("RandomFlip", ("horizontal",), {}), ("RandomRotation", (0.03,), {}), ("RandomZoom", (0.12, 0.12), {}),
          ("RandomTranslation", (0.06, 0.06), {}), ("RandomBrightness", (0.15,), {"value_range": (0.0, 1.0)}),
          ("RandomContrast", (0.2,), {}), ("GaussianNoise", (0.02,), {}))
OPS = ("flip", "rotation", "zoom", "translation", "brightness", "contrast", "noise")   # indice dell'operazione = posizione
KERAS_PREPROCESSING = ("src", "layers", "preprocessing", "image_preprocessing.py")
CHAIN_CLASSES = ("InputLayer", "Conv2D", "Activation", "BatchNormalization", "MaxPooling2D", "Dropout", "Flatten", "Dense")
RESNET_CLASSES = ("InputLayer", "ZeroPadding2D", "Conv2D", "BatchNormalization", "Activation", "MaxPooling2D", "Add",
                  "GlobalAveragePooling2D", "Dense")
PARENT_SUFFIXES = (".keras", "_card.json", "_bn_statistics.json", "_dev_ordinary.csv", "_dev_re-estimated.csv")
TOLERANCE_KEYS = ("statistics_rel", "statistics_abs", "dev_probability_max_abs_diff", "dev_decisions_discordant", "consistency_rel",
                  "consistency_abs", "softmax_of_logits_max_abs_diff")
TOP_KEYS = ("schema", "task", "authorisation", "operations", "scope", "parent_config", "parent_runs", "code", "keras_functions", "recipe",
            "views", "seeds", "dropout", "cells", "reuse", "monte_carlo", "determinism", "tolerances", "threshold", "estimator",
            "evaluation", "threads", "cap", "memory", "ema", "read_guard", "stop_rules", "outputs", "analysis", "test")
PUBLIC_OUTPUTS = ("controls.csv", "cells.csv", "effects.csv", "montecarlo.csv", "layers.csv", "timing.csv", "summary.json", "report.md",
                  "SHA256SUMS")
EXTRA_PROTECTED = PR.EXTRA_PROTECTED + ("results/archrun", "results/archeval")
REQUIRED_PATTERNS = ("fold[0-9]*_test.csv", "*_test_scores.csv", "*scores_test*", "*test*_scores*")
GUARD_ZONES = ("results", "data")
TEST_READ = (
    "no test frame is assembled or evaluated and no test score or metric is read. check-inputs and trial read: this configuration; "
    "the configuration of the two ARCHRUN runs (delivery/archrun/pad_arch_run.json, pinned) and, through pad_arch_run.load_pins, "
    "the files it pins (the pre-registration configuration, the official trial record and the package SHA256SUMS, run_order.csv, "
    "run_caps.csv, the configurations of PRs #27 and #28, the code files); the source of pad_arch_comparison.py (the "
    "augmentation recipe, read with the AST) and of the Keras 2.15 preprocessing module (SHA-256); what "
    "pad_arch_comparison.load_inputs reads (the M2 manifests, the cache hashed whole, the extracted frames file loaded whole: both "
    "frame files also hold frames of the held-out subjects; the cells run, the folds run status, restricted SHA256SUMS and "
    "extracted_frames.csv, the LivenessNet architecture file, the status and restricted SHA256SUMS of the two accepted runs and "
    "one accepted card); of each ARCHRUN run its status.json and restricted SHA256SUMS and, per unit of the scope, only "
    "<unit>.keras, <unit>_card.json, <unit>_bn_statistics.json, <unit>_dev_ordinary.csv and <unit>_dev_re-estimated.csv "
    "(development rows only). trial, in addition: the image arrays of the cache, loaded whole into the frame store, of which only "
    "the fitting and development frames of arm B are assembled (an explicit check refuses any other role). No command loads the "
    "fold schema (fold<k>_test.csv), opens a test score file or anything under results/archeval: an audit hook on every Python "
    "open compares the opened file by identity and refuses the forbidden folders and names, any undeclared file under results/ "
    "and data/, and any file outside the repository other than the Python installation, the system folders, the temporary "
    "folder and the Keras configuration folder")
STOP_KEYS = ("control_failure", "cap", "signals", "memory", "disk_space", "views", "masks", "determinism", "duplicates", "topology",
             "after_stop")

# punti di iniezione (solo i test)
REPO_STATE = None       # (root) -> {"commit", "dirty"} (default: SR.repo_state)
TF_MODULE = None        # () -> modulo TensorFlow (default: BR.tf_module)
CHECK_LOADED = None     # (model, cfg_pr, network) -> None (default: pad_arch_run.check_loaded)
MEMORY_PROBE = None     # () -> memoria residente di picco in GB (default: SR.peak_rss_mb / 1024)
DISK_FREE = None        # (folder) -> byte liberi (default: shutil.disk_usage)
REPO_PARENT = None      # cartella che contiene il repository (default: ROOT.parent; i test la sostituiscono)
UNIFORM = None          # (rng, shape) -> uniformi float32 in [0, 1) per le maschere (default: rng.random)
VIEW_HOOK = None        # (array di un batch di una vista) -> array (default: identità)
REPEAT_HOOK = None      # (statistiche della cella ripetuta) -> statistiche (default: identità)


# ============================================================================ guardia delle letture (rilievi 5 della revisione del piano, I1-I3 del codice)

GUARD = {"installed": False, "active": False, "forbidden": (), "patterns": (), "zones": (), "inside": (), "outside": (),
         "link_exempt": (), "hits": []}


def remove_own_folder(p: Path) -> None:
    """Toglie una cartella creata da questo comando (le viste, un solo livello di file): ogni file con os.remove, poi la
    cartella vuota con os.rmdir; nessuna apertura di file (shutil.rmtree aprirebbe nomi relativi con dir_fd)."""
    p = Path(p)
    if not p.is_dir() or p.is_symlink():
        return
    for e in os.listdir(p):
        q = p / e
        if q.is_dir() and not q.is_symlink():
            remove_own_folder(q)
        else:
            os.remove(q)
    os.rmdir(p)


def _ident(p):
    """(dispositivo, inode) di un percorso (seguendo i collegamenti simbolici), o None se non esiste."""
    try:
        st = os.stat(p)
    except (OSError, ValueError):
        return None
    return (st.st_dev, st.st_ino)


def _chain(full: str) -> set:
    """Le identità del percorso e di tutte le cartelle che lo contengono (quelle che esistono)."""
    out, q = set(), full
    while True:
        i = _ident(q)
        if i is not None:
            out.add(i)
        parent = os.path.dirname(q)
        if parent == q:
            return out
        q = parent


def _ids(paths) -> tuple:
    return tuple(i for i in (_ident(os.path.abspath(os.fspath(p))) for p in paths) if i is not None)


def guard_problem(p) -> str:
    """Il motivo per cui l'apertura di p è rifiutata ("" se è ammessa). Confronti per identità dei file (maiuscole e
    minuscole, collegamenti simbolici e altre grafie dello stesso percorso non aggirano il controllo) e nomi in minuscolo."""
    full = os.path.abspath(os.fspath(p))
    real = os.path.realpath(os.fspath(p))               # il percorso risolto dal sistema sul testo originale (collegamenti, '..' dopo un collegamento)
    names = {os.path.basename(full).lower(), os.path.basename(real).lower()}
    if any(fnmatch.fnmatch(n, pat) for n in names for pat in GUARD["patterns"]):
        return "a forbidden file name"
    chain = _chain(full) | _chain(real)
    if chain & set(GUARD["forbidden"]):
        return "a forbidden folder"
    try:
        st = os.stat(full)
        if os.path.isfile(full) and st.st_nlink > 1 and not chain & set(GUARD["link_exempt"]):
            return "a file with more than one hard link"
    except (OSError, ValueError):
        pass
    for zone, allowed in GUARD["zones"]:
        if zone in chain and not chain & set(allowed):
            return "an undeclared file under results/ or data/"
    if GUARD["inside"] and not chain & set(GUARD["inside"]) and not chain & set(GUARD["outside"]):
        return "an undeclared file outside the repository"
    return ""


def _guard_hook(event, args):
    if event != "open" or not GUARD["active"] or not args:
        return
    p = args[0]
    if isinstance(p, int) or p is None:
        return
    if isinstance(p, bytes):
        p = p.decode("utf-8", "replace")
    # l'evento di os.open (mode None) non porta dir_fd: un percorso relativo potrebbe essere risolto altrove che nella
    # cartella corrente, quindi con la guardia di un comando (radici dichiarate) è sempre rifiutato
    relative = len(args) > 1 and args[1] is None and not os.path.isabs(os.fspath(p))
    if relative and GUARD["inside"]:
        why = "a relative path given to os.open"
    else:
        why = guard_problem(p)
    if why:
        GUARD["hits"].append(os.path.basename(os.fspath(p)))
        raise Failure(f"read guard: an open of {os.path.basename(os.fspath(p))} ({why}) was refused (test data must never be opened)")


def install_guard(forbidden, patterns, zones=None, inside=None, outside=None, link_exempt=None) -> dict:
    """Attiva l'hook di audit (installato una sola volta per processo: un hook di audit non si toglie; resta inerte quando
    non è attivo). forbidden: cartelle vietate; patterns: nomi vietati (confrontati in minuscolo); zones: coppie (cartella,
    percorsi ammessi in essa); inside: le radici dentro cui vale solo la regola delle zone; outside: i percorsi ammessi fuori
    dalle radici; link_exempt: dove un file con più collegamenti fisici è ammesso."""
    if not GUARD["installed"]:
        sys.addaudithook(_guard_hook)
        GUARD["installed"] = True
    GUARD.update(active=False)
    GUARD.update(forbidden=_ids(forbidden), patterns=tuple(p.lower() for p in patterns),
                 zones=tuple((z, _ids(a)) for z, a in ((_ident(os.path.abspath(zz)), aa) for zz, aa in (zones or ())) if z is not None),
                 inside=_ids(inside or ()), outside=_ids(outside or ()), link_exempt=_ids(link_exempt or ()), hits=[])
    GUARD["active"] = True
    return {"forbidden_folders": len(GUARD["forbidden"]), "forbidden_name_patterns": list(GUARD["patterns"]),
            "allowlisted_zones": len(GUARD["zones"]), "outside_allowed": len(GUARD["outside"])}


def system_paths() -> list:
    """Le cartelle che si leggono fuori dal repository senza dati dello studio: installazione di Python, sistema, cartella
    temporanea, configurazione di Keras."""
    return [sys.prefix, sys.base_prefix, sys.exec_prefix, tempfile.gettempdir(), str(Path.home() / ".keras"), "/System", "/usr",
            "/Library", "/dev", "/etc", "/private/etc", "/private/var/db/timezone"]


def outside_problem(base: Path, outside_extra=()) -> str:
    """Il motivo per cui le cartelle ammesse fuori dal repository non vanno bene ("" se vanno bene): la cartella temporanea
    non deve stare dentro il repository o la radice degli ingressi, e nessuna cartella ammessa fuori (cartella temporanea,
    cartella della configurazione, radice d'uscita, cartella del comando) deve contenere il repository, la radice degli
    ingressi o la cartella home (tutto ciò che contiene sarebbe ammesso). La cartella temporanea non deve nemmeno stare
    nella cartella che contiene il repository: copre le cartelle accanto al repository (per esempio quella delle analisi)
    senza nominarle."""
    tmp = Path(tempfile.gettempdir())
    for r, what in ((ROOT, "the repository"), (base, "the input root"), (REPO_PARENT or ROOT.parent, "the folder that contains the repository")):
        if pg.within(tmp, r):
            return f"the temporary folder (TMPDIR) is inside {what}: set TMPDIR outside it"
    for q in [tmp, Path(__file__).resolve().parent] + [Path(x) for x in outside_extra]:
        for r, what in ((ROOT, "the repository"), (base, "the input root"), (Path.home(), "the home folder")):
            if pg.within(r, q):
                return f"an allowed folder outside the repository ({'TMPDIR' if q == tmp else q.name}) contains {what}"
    return ""


def guard_setup(cfg: dict, base: Path, allowed: list, extra_forbidden=(), outside_extra=()) -> dict:
    """La guardia di check-inputs e trial: cartelle vietate ancorate alla radice del repository e a quella degli ingressi;
    nomi vietati; zone results/ e data/ di entrambe le radici con i soli ingressi dichiarati (allowed); fuori dalle radici
    le sole cartelle di system_paths e outside_extra."""
    rg = cfg["read_guard"]
    roots = [ROOT] + ([base] if Path(base).resolve() != ROOT.resolve() else [])
    forbidden = [r / p for r in roots for p in rg["forbidden_paths"]] + [Path(p) for p in extra_forbidden]
    zones = [(r / z, allowed) for r in roots for z in rg["allowlisted_zones"]]
    sysp = system_paths()
    here = [Path(__file__).resolve().parent]            # la cartella di questo comando (nel repository è ROOT/scripts)
    return install_guard(forbidden, rg["forbidden_name_patterns"], zones, roots, sysp + here + list(outside_extra), link_exempt_paths())


def link_exempt_paths() -> list:
    """Dove i file con più collegamenti fisici sono normali: installazione di Python e cartelle di sistema."""
    return [sys.prefix, sys.base_prefix, sys.exec_prefix, "/System", "/usr", "/dev", "/private/var/db/timezone"]


# ============================================================================ configurazione

def _text(v, what: str) -> None:
    if not isinstance(v, str) or not v.strip():
        fail(f"configuration: {what} must be a non-empty text")


def _keys(d, keys, what: str) -> None:
    if not isinstance(d, dict) or sorted(d) != sorted(keys):
        fail(f"configuration: {what} needs exactly {', '.join(sorted(keys))}")


def _pin(e, what: str) -> None:
    _keys(e, ["path", "sha256"], what)
    _text(e["path"], f"{what}.path")
    if not SR._hex(e["sha256"]):
        fail(f"configuration: {what}.sha256 malformed")


def _number(v, what: str, positive=True) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or (v <= 0 if positive else v < 0):
        fail(f"configuration: {what} must be a finite {'positive' if positive else 'non-negative'} number")
    return float(v)


def _bool(v, what: str) -> None:
    if not isinstance(v, bool):
        fail(f"configuration: {what} must be true or false")


def unit_name(u: dict) -> str:
    return PAC.training_name(u["network"], u["fold"], u["seed"])


def unit_label(u: dict) -> str:
    return f"{PR.RUN_TAG[u['run']]}_{unit_name(u)}"


def check_units(units) -> list:
    if not isinstance(units, list) or not units:
        fail("configuration: scope.units must be a non-empty list")
    seen = set()
    for u in units:
        _keys(u, ["fold", "network", "run", "seed"], "scope.units[]")
        if u["run"] not in PARENT_RUNS or u["network"] not in NETWORKS or not FR.is_count(u["fold"]) or not FR.is_count(u["seed"]) \
                or u["fold"] < 1:
            fail(f"configuration: scope unit {u} malformed (run of {list(PARENT_RUNS)}, network of {list(NETWORKS)}, fold, seed)")
        k = (u["run"], u["network"], u["fold"], u["seed"])
        if k in seen:
            fail(f"configuration: scope unit {unit_label(u)} listed twice")
        seen.add(k)
    folds = [u["fold"] for u in units]
    if folds != sorted(folds):
        fail("configuration: scope.units must be grouped by fold, in increasing fold order (the views of a fold are generated once)")
    return units


def load_config(p: Path) -> tuple:
    """(configurazione, SHA-256 dei byte letti). Insiemi chiusi a ogni livello e valori fissati controllati qui, prima di
    leggere qualunque altro file; nessun valore predefinito."""
    data = read_regular(p, "configuration")
    cfg = SR.read_json(data, f"configuration {p.name}")
    if cfg.get("schema") != CONFIG_SCHEMA:
        fail(f"configuration {p.name}: schema must be {CONFIG_SCHEMA!r}")
    if sorted(cfg) != sorted(TOP_KEYS):
        fail(f"configuration: fields must be exactly {', '.join(TOP_KEYS)} (missing {sorted(set(TOP_KEYS) - set(cfg))}, "
             f"unknown {sorted(set(cfg) - set(TOP_KEYS))})")
    for k in ("task", "authorisation", "operations"):
        _text(cfg[k], k)
    if "--i-am-authorized" not in cfg["authorisation"]:
        fail("configuration: authorisation must say when --i-am-authorized is passed")
    if cfg["test"] != TEST_READ:
        fail("configuration: test must state exactly what is read (TEST_READ of the script)")
    sc = cfg["scope"]
    _keys(sc, ["name", "report_cell_metrics", "rule", "units"], "scope")
    if sc["name"] not in SCOPES:
        fail(f"configuration: scope.name must be one of {', '.join(SCOPES)}")
    _text(sc["rule"], "scope.rule")
    _bool(sc["report_cell_metrics"], "scope.report_cell_metrics")
    check_units(sc["units"])
    _pin(cfg["parent_config"], "parent_config")
    pr = cfg["parent_runs"]
    _keys(pr, list(PARENT_RUNS), "parent_runs")
    for r, e in pr.items():
        _keys(e, ["folder", "restricted_sha256sums_sha256", "run_id", "status_sha256"], f"parent_runs.{r}")
        if not isinstance(e["run_id"], str) or not PR.RUN_ID_RE.match(e["run_id"]) or f"-{PR.RUN_TAG[r]}-" not in e["run_id"]:
            fail(f"configuration: parent_runs.{r}.run_id must be an ARCHRUN run id of {r}")
        _text(e["folder"], f"parent_runs.{r}.folder")
        if not SR._hex(e["restricted_sha256sums_sha256"]) or not SR._hex(e["status_sha256"]):
            fail(f"configuration: parent_runs.{r} SHA-256 values malformed")
    _keys(cfg["code"], ["scripts/pad_arch_run.py"], "code")
    if not SR._hex(cfg["code"]["scripts/pad_arch_run.py"]):
        fail("configuration: code SHA-256 malformed")
    kf = cfg["keras_functions"]
    _keys(kf, ["functions", "module", "sha256"], "keras_functions")
    if kf["module"] != "keras.src.layers.preprocessing.image_preprocessing" or not SR._hex(kf["sha256"]) \
            or kf["functions"] != ["transform", "get_rotation_matrix", "get_zoom_matrix", "get_translation_matrix"]:
        fail("configuration: keras_functions must pin the Keras 2.15 preprocessing module and its four transform functions")
    rc = cfg["recipe"]
    _keys(rc, ["applied_to", "differences", "function", "operations", "source", "variable"], "recipe")
    if (rc["source"], rc["function"], rc["variable"]) != ("scripts/pad_arch_comparison.py", "trial_trainer", "aug"):
        fail("configuration: recipe must be the Sequential aug of pad_arch_comparison.trial_trainer")
    if rc["operations"] != recipe_json(RECIPE):
        fail("configuration: recipe.operations must be RECIPE of the script")
    _text(rc["applied_to"], "recipe.applied_to")
    _text(rc["differences"], "recipe.differences")
    if "per frame" not in rc["differences"]:
        fail("configuration: recipe.differences must declare the contrast factor drawn per frame")
    vw = cfg["views"]
    _keys(vw, ["folder", "free_space_margin_gb", "halves", "order", "seeds", "storage", "value_bounds"], "views")
    if vw["seeds"] != list(VIEW_SEEDS) or vw["halves"] != [list(h) for h in HALVES]:
        fail(f"configuration: views.seeds must be {list(VIEW_SEEDS)} and views.halves {[list(h) for h in HALVES]} (PROTOCOL.md §4)")
    if vw["storage"] not in ("memory", "memmap"):
        fail("configuration: views.storage must be memory or memmap")
    b = vw["value_bounds"]
    if not isinstance(b, list) or len(b) != 2 or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in b) or not b[0] < 0 < 1 < b[1]:
        fail("configuration: views.value_bounds must be [lower < 0, upper > 1]")
    _number(vw["free_space_margin_gb"], "views.free_space_margin_gb")
    _text(vw["order"], "views.order")
    _text(vw["folder"], "views.folder")
    sd = cfg["seeds"]
    _keys(sd, ["derivation", "domains", "namespace", "publication"], "seeds")
    if sd["namespace"] != SEED_NAMESPACE or sd["domains"] != list(SEED_DOMAINS):
        fail(f"configuration: seeds.namespace must be {SEED_NAMESPACE!r} and seeds.domains {list(SEED_DOMAINS)}")
    _text(sd["derivation"], "seeds.derivation")
    _text(sd["publication"], "seeds.publication")
    dr = cfg["dropout"]
    _keys(dr, ["mask_frequency_sigma", "networks", "rule"], "dropout")
    _keys(dr["networks"], list(NETWORKS), "dropout.networks")
    for net, e in dr["networks"].items():
        _keys(e, ["active", "layers"], f"dropout.networks.{net}")
        if not isinstance(e["layers"], dict) or any(not isinstance(r, float) or not 0 < r < 1 for r in e["layers"].values()) \
                or not isinstance(e["active"], list) or any(a not in e["layers"] for a in e["active"]):
            fail(f"configuration: dropout.networks.{net} must give the Dropout layers with their rates and the active ones among them")
    if dr["networks"]["ResNet-18"] != {"active": [], "layers": {}}:
        fail("configuration: ResNet-18 has no Dropout layer")
    _number(dr["mask_frequency_sigma"], "dropout.mask_frequency_sigma")
    _text(dr["rule"], "dropout.rule")
    if cfg["cells"] != {c: cell_text(c) for c in CELLS}:
        fail("configuration: cells must be the four cells of PROTOCOL.md §4 (cell_text of the script)")
    ru = cfg["reuse"]
    _keys(ru, ["duplicate_check", "rule"], "reuse")
    if ru["duplicate_check"] not in ("all", "first", "none"):
        fail("configuration: reuse.duplicate_check must be all, first or none")
    _text(ru["rule"], "reuse.rule")
    mc = cfg["monte_carlo"]
    _keys(mc, ["comparison", "runs"], "monte_carlo")
    if not isinstance(mc["runs"], list) or any(r not in PARENT_RUNS for r in mc["runs"]):
        fail("configuration: monte_carlo.runs must list runs of the parent runs")
    _text(mc["comparison"], "monte_carlo.comparison")
    de = cfg["determinism"]
    _keys(de, ["cell", "networks", "units"], "determinism")
    if de["cell"] not in ("E10", "E01", "E11") or not isinstance(de["networks"], list) or any(n not in NETWORKS for n in de["networks"]) \
            or de["units"] not in ("all", "first"):
        fail("configuration: determinism needs a stochastic cell, networks and units all or first")
    tol = cfg["tolerances"]
    _keys(tol, list(TOLERANCE_KEYS), "tolerances")
    for k in TOLERANCE_KEYS:
        if k == "dev_decisions_discordant":
            if not FR.is_count(tol[k]):
                fail("configuration: tolerances.dev_decisions_discordant must be a non-negative integer")
        else:
            _number(tol[k], f"tolerances.{k}")
    if cfg["threshold"] != FR.FIXED_THRESHOLD:
        fail("configuration: threshold must be 0.5 (the only scoring threshold of the primary contrasts, PROTOCOL.md §4)")
    es = cfg["estimator"]
    _keys(es, ["accumulation", "batch", "orders"], "estimator")
    if es["batch"] != FR.SCORE_BATCH or es["orders"] != PR.expected_orders():
        fail("configuration: estimator.batch must be 16 and estimator.orders the orders of pad_arch_run")
    _text(es["accumulation"], "estimator.accumulation")
    ev = cfg["evaluation"]
    _keys(ev, ["dev", "fitting_clean"], "evaluation")
    _text(ev["dev"], "evaluation.dev")
    _bool(ev["fitting_clean"], "evaluation.fitting_clean")
    th = cfg["threads"]
    _keys(th, ["device", "inter_op", "intra_op", "rule"], "threads")
    if (th["intra_op"], th["inter_op"], th["device"]) != (PAC.REQUIRED_THREADS, PAC.REQUIRED_THREADS, "CPU"):
        fail(f"configuration: threads must be {PAC.REQUIRED_THREADS} intra-op and {PAC.REQUIRED_THREADS} inter-op on the CPU")
    _text(th["rule"], "threads.rule")
    cp = cfg["cap"]
    _keys(cp, ["rule", "unit", "wall_minutes"], "cap")
    if not FR.is_count(cp["wall_minutes"]) or cp["wall_minutes"] < 1 or cp["unit"] != PAC.CAP_UNIT:
        fail(f"configuration: cap.wall_minutes must be a positive integer in {PAC.CAP_UNIT!r}")
    _text(cp["rule"], "cap.rule")
    me = cfg["memory"]
    _keys(me, ["peak_rss_limit_gb", "rule"], "memory")
    _number(me["peak_rss_limit_gb"], "memory.peak_rss_limit_gb")
    _text(me["rule"], "memory.rule")
    em = cfg["ema"]
    _keys(em, ["fitting_frames", "momentum", "train_batch", "updates_per_epoch"], "ema")
    if em["updates_per_epoch"] != math.ceil(em["fitting_frames"] / em["train_batch"]) or em["momentum"] != AM.BN_MOMENTUM:
        fail("configuration: ema.updates_per_epoch must be ceil(fitting_frames / train_batch) and ema.momentum the BN momentum")
    rg = cfg["read_guard"]
    _keys(rg, ["allowlisted_zones", "forbidden_name_patterns", "forbidden_paths", "outside_repository", "rule"], "read_guard")
    if "results/archeval" not in rg["forbidden_paths"] or not set(REQUIRED_PATTERNS) <= set(rg["forbidden_name_patterns"]) \
            or rg["allowlisted_zones"] != list(GUARD_ZONES):
        fail("configuration: read_guard must forbid results/archeval and the test names of REQUIRED_PATTERNS and allowlist results/ and data/")
    _text(rg["rule"], "read_guard.rule")
    _text(rg["outside_repository"], "read_guard.outside_repository")
    st = cfg["stop_rules"]
    _keys(st, list(STOP_KEYS), "stop_rules")
    for k in STOP_KEYS:
        _text(st[k], f"stop_rules.{k}")
    ou = cfg["outputs"]
    _keys(ou, ["public", "restricted", "run"], "outputs")
    if ou["public"] != list(PUBLIC_OUTPUTS) or ou["run"] != [STATUS, LOG]:
        fail("configuration: outputs must list the public and run files of the script")
    _text(ou["restricted"], "outputs.restricted")
    an = cfg["analysis"]
    _keys(an, ["frozen_at", "path", "sha256"], "analysis")
    _text(an["frozen_at"], "analysis.frozen_at")
    _text(an["path"], "analysis.path")
    if an["sha256"] is not None and not SR._hex(an["sha256"]):
        fail("configuration: analysis.sha256 must be null or a SHA-256")
    return cfg, sha256_bytes(data)


def recipe_json(recipe) -> list:
    return [{"class": c, "args": list(a), "kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()}} for c, a, kw in recipe]


def cell_text(c: str) -> str:
    aug, drop = FACTORS[c]
    return (f"{'augmented' if aug else 'unaugmented'} fitting frames, dropout {'on for the upstream Dropout layers only' if drop else 'off'}"
            + ("; the existing estimator (nuaa_bn_recalibration.recalibrate), one clean pass" if c == "E00" else "; eight views, seeds 101-108"))


# ============================================================================ ricetta e codice fissato

def literal(node):
    v = ast.literal_eval(node)
    return tuple(v) if isinstance(v, list) else v


def recipe_from_source(src: str, function: str = "trial_trainer", variable: str = "aug") -> dict:
    """L'augmentation della ricetta di addestramento letta dal sorgente con l'AST: la Sequential assegnata a `variable` in
    `function` (classi, argomenti posizionali e con nome, valori letterali), se i frame di addestramento sono divisi per 255
    una volta e se la Sequential è chiamata con training=True (che quella chiamata sia proprio sul dataset del fitting non
    è controllato: dichiarato in NOTE.md)."""
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function), None)
    if fn is None:
        fail(f"recipe: function {function} not found in the source")
    seq = None
    divs = 0
    applied = False
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == variable for t in n.targets):
            seq = n.value
        if isinstance(n, ast.AugAssign) and isinstance(n.op, ast.Div) and isinstance(n.target, ast.Name) and n.target.id == "Xtr" \
                and isinstance(n.value, ast.Constant) and n.value.value == 255.0:
            divs += 1
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == variable \
                and any(k.arg == "training" and isinstance(k.value, ast.Constant) and k.value.value is True for k in n.keywords):
            applied = True
    if seq is None or not isinstance(seq, ast.Call) or getattr(seq.func, "attr", None) != "Sequential" or len(seq.args) != 1 \
            or not isinstance(seq.args[0], ast.List):
        fail(f"recipe: {variable} = tf.keras.Sequential([...]) not found in {function}")
    ops = []
    for c in seq.args[0].elts:
        if not isinstance(c, ast.Call) or not hasattr(c.func, "attr"):
            fail("recipe: an element of the Sequential is not a layer call")
        ops.append((c.func.attr, tuple(literal(a) for a in c.args), {k.arg: literal(k.value) for k in c.keywords}))
    return {"operations": tuple(ops), "divisions_by_255": divs, "applied_with_training_true": applied}


def check_recipe(src: str) -> dict:
    r = recipe_from_source(src)
    if r["operations"] != RECIPE:
        fail(f"recipe: the Sequential of the training recipe {r['operations']} differs from RECIPE of this script")
    if r["divisions_by_255"] != 1 or not r["applied_with_training_true"]:
        fail("recipe: the training frames must be divided by 255 exactly once and the Sequential applied with training=True")
    return {"operations": [c for c, _, _ in RECIPE], "divisions_by_255": 1, "applied_with_training_true": True}


def recipe_ranges(recipe=RECIPE) -> dict:
    """Gli estremi delle estrazioni ricavati dalla ricetta con la semantica degli strati di Keras 2.15 per un fattore
    scalare: rotazione [-f, f] x 2 pi; zoom e traslazione [1 - h, 1 + h], [1 - w, 1 + w] e [-h, h], [-w, w]; luminosità
    [-f, f] x (massimo - minimo di value_range), poi clip a value_range; contrasto [1 - f, 1 + f]; rumore deviazione standard
    f. Un fattore non scalare o un flip non orizzontale sono rifiutati (la riscrittura non li copre)."""
    ops = {c: (a, kw) for c, a, kw in recipe}
    if ops["RandomFlip"][0] != ("horizontal",):
        fail("recipe: only the horizontal flip is implemented")
    vals = [ops["RandomRotation"][0][0], *ops["RandomZoom"][0], *ops["RandomTranslation"][0], ops["RandomBrightness"][0][0],
            ops["RandomContrast"][0][0], ops["GaussianNoise"][0][0]]
    if len(vals) != 8 or not all(isinstance(v, float) and v >= 0 for v in vals):
        fail("recipe: every factor of the recipe must be one non-negative float")
    rot, zh, zw, th, tw, br, ct, nz = vals
    vr = tuple(sorted(ops["RandomBrightness"][1]["value_range"]))
    return {"angle": (-rot * 2.0 * np.pi, rot * 2.0 * np.pi), "zoom_h": (1.0 - zh, 1.0 + zh), "zoom_w": (1.0 - zw, 1.0 + zw),
            "trans_h": (-th, th), "trans_w": (-tw, tw), "brightness": (-br, br), "value_range": vr, "contrast": (1.0 - ct, 1.0 + ct),
            "noise_stddev": nz}


RANGES = recipe_ranges()


def keras_module_path() -> Path:
    """Il file del modulo di pre-elaborazione di Keras, trovato senza importare Keras né TensorFlow."""
    import importlib.util
    spec = importlib.util.find_spec("keras")
    if spec is None or not spec.submodule_search_locations:
        fail("keras is not installed")
    return Path(list(spec.submodule_search_locations)[0]).joinpath(*KERAS_PREPROCESSING)


def check_code(cfg: dict) -> dict:
    """I file che questo comando fissa oltre a pad_arch_run.json: pad_arch_run.py, il modulo di pre-elaborazione di Keras e
    questo file (registrato, non fissato)."""
    got = {"scripts/pad_arch_run.py": sha256_bytes(read_regular(Path(PR.__file__), "code file"))}
    if got["scripts/pad_arch_run.py"] != cfg["code"]["scripts/pad_arch_run.py"]:
        fail("code: scripts/pad_arch_run.py differs from the SHA-256 pinned in the configuration")
    k = sha256_bytes(read_regular(keras_module_path(), "Keras preprocessing module"))
    if k != cfg["keras_functions"]["sha256"]:
        fail("code: the Keras preprocessing module differs from the SHA-256 pinned in the configuration (another Keras version?)")
    return {**got, "keras.src.layers.preprocessing.image_preprocessing": k,
            "scripts/pad_arch_bn_factorial.py": sha256_bytes(read_regular(Path(__file__), "code file"))}


# ============================================================================ estrazioni senza stato

def frame_digest(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def fit_digests(keys) -> list:
    """L'impronta di ogni frame di fitting dalla sua chiave (non dalla posizione): lo stesso frame ha le stesse estrazioni in
    ogni fold, ordine e batch."""
    return [frame_digest(k) for k in keys]


def philox(domain: str, view_seed: int, digest: str, tag: str) -> np.random.Generator:
    """Un generatore Philox a contatore (Philox4x64 di NumPy) con chiave i primi 128 bit di
    SHA-256("pad-arch-bn-factorial/1|<dominio>|<seme della vista>|<SHA-256 della chiave del frame>|<operazione o strato>"):
    funzione pura dei suoi argomenti, indipendente da batch, posizione, ordine e da ogni altra estrazione."""
    if domain not in SEED_DOMAINS:
        fail(f"unknown seed domain {domain!r}")
    h = hashlib.sha256(f"{SEED_NAMESPACE}|{domain}|{int(view_seed)}|{digest}|{tag}".encode("utf-8")).digest()
    return np.random.Generator(np.random.Philox(key=int.from_bytes(h[:16], "big")))


def scale_to(bounds, u) -> np.float32:
    """lo + (hi - lo) * u in float64, poi float32."""
    lo, hi = bounds
    return np.float32(lo + (hi - lo) * u)


def uniform(rng, bounds):
    """Un'estrazione uniforme negli estremi dal generatore (u in [0, 1), float64), poi float32."""
    return scale_to(bounds, rng.random())


def draw_params(digests, view_seed: int) -> dict:
    """I parametri delle operazioni per ogni frame di un batch (array sul batch, float32), ognuno dal suo generatore
    (dominio augmentation, nome dell'operazione), negli intervalli di RANGES: flip (1.0 se round(u) = 1, come il flip di
    Keras a batch), angolo, zoom e traslazione di altezza e larghezza, delta di luminosità, fattore di contrasto. Il rumore
    è estratto in noise_of."""
    p = {k: [] for k in ("flip", "angle", "zoom_h", "zoom_w", "trans_h", "trans_w", "brightness", "contrast")}
    for d in digests:
        p["flip"].append(np.float32(np.rint(philox("augmentation", view_seed, d, "flip").random())))
        p["angle"].append(uniform(philox("augmentation", view_seed, d, "rotation"), RANGES["angle"]))
        zh, zw = philox("augmentation", view_seed, d, "zoom").random(2)                 # altezza, poi larghezza (ordine di Keras)
        p["zoom_h"].append(scale_to(RANGES["zoom_h"], zh))
        p["zoom_w"].append(scale_to(RANGES["zoom_w"], zw))
        th, tw = philox("augmentation", view_seed, d, "translation").random(2)
        p["trans_h"].append(scale_to(RANGES["trans_h"], th))
        p["trans_w"].append(scale_to(RANGES["trans_w"], tw))
        p["brightness"].append(uniform(philox("augmentation", view_seed, d, "brightness"), RANGES["brightness"]))
        p["contrast"].append(uniform(philox("augmentation", view_seed, d, "contrast"), RANGES["contrast"]))
    return {k: np.asarray(v, np.float32) for k, v in p.items()}


def noise_of(digests, view_seed: int, shape) -> np.ndarray:
    """GaussianNoise per frame: z x deviazione standard + 0 in float32 da una normale standard (dominio augmentation, nome
    noise), come random_normal(mean=0, stddev) di Keras."""
    sd = np.float32(RANGES["noise_stddev"])
    return np.stack([philox("augmentation", view_seed, d, "noise").standard_normal(shape, dtype=np.float32) * sd + np.float32(0.0)
                     for d in digests])


# ============================================================================ augmentation (rilievo 1 della revisione del piano)

def keras_ip():
    from keras.src.layers.preprocessing import image_preprocessing as IP
    return IP


def apply_op(tf, op: str, x, p: dict, noise=None):
    """Un'operazione della ricetta su un batch (float32, NHWC) con parametri per frame, con le funzioni di Keras e
    l'aritmetica dello strato corrispondente in modalità training."""
    IP = keras_ip()
    b = int(x.shape[0])
    hd, wd = tf.cast(tf.shape(x)[1], tf.float32), tf.cast(tf.shape(x)[2], tf.float32)
    if op == "flip":                                   # tf.image.stateless_random_flip_left_right su un batch (rango 4)
        f = tf.reshape(tf.constant(p["flip"], tf.float32), [b, 1, 1, 1])
        return f * tf.reverse(x, [2]) + (1 - f) * x
    if op == "rotation":
        return IP.transform(x, IP.get_rotation_matrix(tf.constant(p["angle"], tf.float32), hd, wd), fill_mode="reflect", fill_value=0.0,
                            interpolation="bilinear")
    if op == "zoom":
        zooms = tf.cast(tf.concat([tf.constant(p["zoom_w"][:, None]), tf.constant(p["zoom_h"][:, None])], axis=1), tf.float32)
        return IP.transform(x, IP.get_zoom_matrix(zooms, hd, wd), fill_mode="reflect", fill_value=0.0, interpolation="bilinear")
    if op == "translation":
        ht = tf.constant(p["trans_h"][:, None]) * hd
        wt = tf.constant(p["trans_w"][:, None]) * wd
        tr = tf.cast(tf.concat([wt, ht], axis=1), tf.float32)
        return IP.transform(x, IP.get_translation_matrix(tr), interpolation="bilinear", fill_mode="reflect", fill_value=0.0)
    if op == "brightness":
        lo, hi = RANGES["value_range"]
        delta = tf.cast(tf.reshape(tf.constant(p["brightness"], tf.float32), [b, 1, 1, 1]) * (hi - lo), x.dtype)
        x = x + delta
        return tf.clip_by_value(x, lo, hi)
    if op == "contrast":                               # adjust_contrast per frame, poi la clip di RandomContrast
        out = tf.concat([tf.image.adjust_contrast(x[i:i + 1], p["contrast"][i]) for i in range(b)], axis=0)
        return tf.clip_by_value(out, 0, 255)
    if op == "noise":
        return x + tf.constant(noise, tf.float32)
    fail(f"unknown augmentation operation {op!r}")


def augment(tf, X01: np.ndarray, digests, view_seed: int) -> np.ndarray:
    """Una vista di un batch di frame già divisi per 255 (float32): le sette operazioni nell'ordine della ricetta."""
    p = draw_params(digests, view_seed)
    x = tf.convert_to_tensor(X01, tf.float32)
    for op in OPS:
        x = apply_op(tf, op, x, p, noise_of(digests, view_seed, X01.shape[1:]) if op == "noise" else None)
    return np.asarray(x.numpy(), dtype=np.float32)


def scaled(X: np.ndarray) -> np.ndarray:
    """La convenzione d'ingresso dello stimatore e della ricetta: frame uint8 in float32, divisi per 255 una volta."""
    return X.astype(np.float32) / 255.0


def stale_views(parent: Path) -> list:
    """Le cartelle delle viste lasciate da una prova precedente (uccisa senza pulizia) nella cartella temporanea."""
    return sorted(e for e in os.listdir(parent) if e.startswith(VIEWS_PREFIX)) if Path(parent).is_dir() else []


class Views:
    """Le otto viste aumentate dei frame di fitting di un fold, generate una volta (viste all'esterno, batch dello stimatore
    all'interno), controllate finite ed entro views.value_bounds; in memoria o in un numpy.memmap in una cartella nuova
    dentro la cartella temporanea del processo (mai dentro il repository o la radice degli ingressi), dopo il controllo
    dello spazio libero; close() toglie la cartella."""

    def __init__(self, tf, X: np.ndarray, digests, seeds, batch: int, storage: str, parent: Path, bounds, margin_gb: float = 2.0,
                 check=None, refuse_inside=()):
        self.seeds = tuple(seeds)
        self.n = len(X)
        self.X, self.digests, self.batch = X, list(digests), batch
        shape = (len(self.seeds),) + X.shape
        nbytes = int(np.prod(shape)) * 4
        self.folder = None
        self.stats = {"min": None, "max": None, "values": int(np.prod(shape)), "bytes": nbytes, "storage": storage}
        if storage == "memmap":
            parent = Path(parent)
            for r in refuse_inside:
                if pg.within(parent, r):
                    raise ControlFailure("disk_space", {"reason": "the temporary folder is inside the repository or the input root"},
                                         "the views folder must be outside the repository and the input root (set TMPDIR elsewhere)")
            free = DISK_FREE(parent) if DISK_FREE is not None else shutil.disk_usage(parent).free
            need = nbytes + int(margin_gb * 2 ** 30)
            if free < need:
                raise ControlFailure("disk_space", {"free_bytes": int(free), "needed_bytes": need},
                                     f"{free} bytes free in the temporary folder, {need} needed for the views and the margin")
            self.folder = Path(tempfile.mkdtemp(prefix=VIEWS_PREFIX, dir=parent))
            self.data = np.memmap(self.folder / "views.f32", dtype=np.float32, mode="w+", shape=shape)
        else:
            self.data = np.empty(shape, np.float32)
        lo, hi = bounds
        try:
            for vi, v in enumerate(self.seeds):
                for i in range(0, self.n, batch):
                    if check is not None:
                        check(f"views {v} batch {i // batch + 1}")
                    out = augment(tf, scaled(X[i:i + batch]), self.digests[i:i + batch], v)
                    if VIEW_HOOK is not None:
                        out = VIEW_HOOK(out)
                    if not np.all(np.isfinite(out)):
                        raise ControlFailure("views", {"view_seed": v, "non_finite_values": int(np.sum(~np.isfinite(out)))},
                                             f"view {v}: non-finite values in the augmented frames")
                    mn, mx = float(out.min()), float(out.max())
                    if mn < lo or mx > hi:
                        raise ControlFailure("views", {"view_seed": v, "min": mn, "max": mx, "bounds": [lo, hi]},
                                             f"view {v}: values outside the declared bounds [{lo}, {hi}]")
                    self.stats["min"] = mn if self.stats["min"] is None else min(self.stats["min"], mn)
                    self.stats["max"] = mx if self.stats["max"] is None else max(self.stats["max"], mx)
                    self.data[vi, i:i + batch] = out
            if self.folder is not None:
                self.data.flush()
        except BaseException:
            self.close()
            raise

    def view(self, seed: int) -> np.ndarray:
        return self.data[self.seeds.index(seed)]

    def regenerated_first_batches(self, tf) -> dict:
        """Il primo batch di ogni vista rigenerato e confrontato byte per byte con quello conservato (determinismo
        dell'augmentation)."""
        b = min(self.batch, self.n)
        return {v: bool(np.asarray(self.view(v)[:b]).tobytes() == augment(tf, scaled(self.X[:b]), self.digests[:b], v).tobytes())
                for v in self.seeds}

    def close(self) -> None:
        if self.folder is not None:
            self.data = None
            gc.collect()
            remove_own_folder(self.folder)
            self.folder = None


# ============================================================================ sorgenti di una ristima

class Source:
    """I frame di una passata di ristima: per seme della vista una funzione che dà batch float32 (già divisi per 255) e le
    impronte dei frame (per le maschere del dropout). clean(): n copie identiche dei frame di fitting puliti;
    augmented(): le viste."""

    def __init__(self, seeds, n: int, get, digests, kind: str):
        self.seeds, self.n, self.get, self.digests, self.kind = tuple(seeds), n, get, list(digests), kind

    @staticmethod
    def clean(X: np.ndarray, digests, seeds) -> "Source":
        return Source(seeds, len(X), lambda v, i, j: scaled(X[i:j]), digests, "clean")

    @staticmethod
    def augmented(views: Views, digests, seeds) -> "Source":
        if any(s not in views.seeds for s in seeds):
            fail("augmented source: a view seed was not generated")
        return Source(seeds, views.n, lambda v, i, j: np.asarray(views.view(v)[i:j]), digests, "augmented")


# ============================================================================ struttura della rete (rilievo 12 della revisione del piano)

def structure(model, network: str, cfg: dict) -> dict:
    """La struttura su cui contano le celle con dropout, controllata sul checkpoint caricato. LivenessNet: classi fra
    CHAIN_CLASSES; una sola catena (ogni strato dopo l'ingresso ha un solo strato in ingresso, il precedente in
    model.layers); gli strati Dropout esattamente quelli di dropout.networks, con i rate del checkpoint uguali alla
    configurazione e noise_shape None (le maschere sono per elemento); gli attivi = i Dropout prima di almeno una
    BatchNormalization, uguali a dropout.networks.<rete>.active. ResNet-18: classi fra RESNET_CLASSES (nessun Dropout,
    rumore o augmentation). Restituisce la catena (nomi degli strati, risolti sul modello che gira: una copia ha i suoi
    strati), i Dropout attivi con i rate e, per ogni BN, i Dropout attivi a monte. Ogni difetto: ControlFailure topology."""
    topo = AM.model_topology(model)
    classes = {l.name: type(l).__name__ for l in model.layers}
    dc = cfg["dropout"]["networks"][network]

    def refuse(msg, **v):
        raise ControlFailure("topology", {"network": network, **v}, f"{network}: {msg}")
    if network == "ResNet-18":
        bad = sorted(n for n, c in classes.items() if c not in RESNET_CLASSES)
        if bad:
            refuse("layers of classes outside the expected ones are refused: the comparator must contain no dropout, noise or augmentation layer",
                   classes=sorted({classes[n] for n in bad}))
        return {"chain": None, "active": {}, "upstream": {}}
    bad = sorted(n for n, c in classes.items() if c not in CHAIN_CLASSES)
    if bad:
        refuse("layers of classes not expected in the chain", classes=sorted({classes[n] for n in bad}))
    layers = model.layers
    consumers = {}
    for n, (_, ins, _) in topo.items():
        for i in ins:
            consumers.setdefault(i, []).append(n)
    for i, l in enumerate(layers[1:], start=1):
        if topo[l.name][1] != (layers[i - 1].name,):
            refuse(f"topology: {l.name} is not fed by the previous layer {layers[i - 1].name} only (the chain is required)")
    for n, c in consumers.items():
        if len(c) != 1:
            refuse(f"topology: {n} feeds {len(c)} layers (a branch): the chain is required")
    drops = {l.name: float(l.rate) for l in layers if type(l).__name__ == "Dropout"}
    if drops != {k: float(v) for k, v in dc["layers"].items()}:
        refuse(f"Dropout layers and rates of the checkpoint {drops} differ from the configuration {dc['layers']}")
    if any(l.noise_shape is not None for l in layers if type(l).__name__ == "Dropout"):
        refuse("a Dropout layer has a noise_shape: the masks of this command are element-wise")
    names = [l.name for l in layers]
    bns = [n for n in names if classes[n] == "BatchNormalization"]
    active = [d for d in drops if any(names.index(b) > names.index(d) for b in bns)]
    if active != dc["active"]:
        refuse(f"the Dropout layers before a BatchNormalization {active} differ from dropout.networks.{network}.active {dc['active']}")
    upstream = {b: [d for d in active if names.index(d) < names.index(b)] for b in bns}
    return {"chain": [l.name for l in layers[1:]], "active": {d: drops[d] for d in active}, "upstream": upstream}


def stochastic_upstream(st: dict, cell: str, layer: str) -> tuple:
    """Le operazioni stocastiche attive a monte dell'ingresso di una BN in una cella."""
    aug, drop = FACTORS[cell]
    return (("augmentation",) if aug else ()) + (tuple(st["upstream"].get(layer, ())) if drop else ())


def reuse_eligible(st: dict, cell: str, order) -> list:
    """Gli strati di una cella che prendono le statistiche della cella di riferimento (stesse operazioni stocastiche a monte)."""
    ref = REFERENCE.get(cell)
    if ref is None:
        return []
    return [n for n in order if stochastic_upstream(st, cell, n) == stochastic_upstream(st, ref, n)]


# ============================================================================ maschere del dropout

class MaskCounter:
    """Zeri e valori delle maschere di ogni Dropout, contati una volta per ristima (alla prima passata che lo attraversa)."""

    def __init__(self):
        self.counts = {}
        self.done = set()

    def add(self, layer: str, keep: np.ndarray) -> None:
        z, n = self.counts.get(layer, (0, 0))
        self.counts[layer] = (z + int(keep.size - np.count_nonzero(keep)), n + int(keep.size))


def mask_of(digest: str, view_seed: int, layer: str, shape, rate: float) -> np.ndarray:
    """Maschera di un frame: u >= rate con u uniforme float32 in [0, 1) (dominio dropout, nome = strato), come tf.nn.dropout."""
    rng = philox("dropout", view_seed, digest, layer)
    u = UNIFORM(rng, shape) if UNIFORM is not None else rng.random(shape, dtype=np.float32)
    return u >= np.float32(rate)


def apply_dropout(tf, x, keep: np.ndarray, rate: float):
    """tf.nn.dropout con una maschera data: where(keep, x * (1 / (1 - rate)), 0) nel dtype di x."""
    scale = tf.convert_to_tensor(1 / (1 - rate), dtype=x.dtype)
    return tf.where(tf.constant(keep), x * scale, tf.zeros_like(x))


# ============================================================================ momenti sulle viste

def moments_pass(tf, model, source: Source, targets, batch: int, st: dict, dropout_on: bool, counter: MaskCounter = None,
                 force_chain: bool = False) -> dict:
    """I momenti per canale (BR.Moments, unione esatta in float64) degli ingressi degli strati richiesti su tutte le viste
    della sorgente, viste all'esterno e batch all'interno, in inferenza. Senza dropout attivo: un sotto-modello con gli
    ingressi richiesti come uscite (come BR.forward, su ingressi già divisi per 255). Con dropout attivo (LivenessNet), o
    force_chain: la catena strato per strato, ogni strato chiamato con training=False e, con dropout_on, i Dropout attivi
    sostituiti dalle maschere senza stato di (vista, frame, strato); senza dropout_on i Dropout sono chiamati con
    training=False (identità). Le maschere di un Dropout sono contate alla prima passata che lo attraversa (counter)."""
    targets = list(targets)
    mom = {n: BR.Moments() for n in targets}
    use_chain = force_chain or (dropout_on and bool(st["active"]))
    if use_chain and st["chain"] is None:
        fail("the layer-by-layer pass needs the chain of LivenessNet")
    if not use_chain:
        sub = tf.keras.Model(model.input, [model.get_layer(n).input for n in targets])
    else:
        names = list(st["chain"])
        chain = [model.get_layer(n) for n in names]            # gli strati del modello che gira (una copia), per nome
        last = max(names.index(n) for n in targets)
        count = {d for d in st["active"] if names.index(d) < last} - (counter.done if counter is not None else set())
    for v in source.seeds:
        for i in range(0, source.n, batch):
            xb = source.get(v, i, i + batch)
            with tf.device("/CPU:0"):
                if not use_chain:
                    res = sub(xb, training=False)
                    res = list(res) if isinstance(res, (list, tuple)) else [res]
                    for n, a in zip(targets, res):
                        mom[n].add(np.asarray(a))
                    continue
                x = tf.convert_to_tensor(xb, tf.float32)
                digests = source.digests[i:i + batch]
                for k, layer in enumerate(chain[:last + 1]):
                    if layer.name in mom:
                        mom[layer.name].add(np.asarray(x))
                    if k == last:
                        break
                    if dropout_on and layer.name in st["active"]:
                        rate = st["active"][layer.name]
                        keep = np.stack([mask_of(d, v, layer.name, tuple(x.shape[1:]), rate) for d in digests])
                        if counter is not None and layer.name in count:
                            counter.add(layer.name, keep)
                        x = apply_dropout(tf, x, keep, rate)
                    else:
                        x = layer(x, training=False)
    if counter is not None and use_chain and dropout_on:
        counter.done |= count
    return mom


def reestimate_cell(tf, copy, source: Source, order, batch: int, st: dict, dropout_on: bool, reuse: dict, ctx, label: str,
                    counter: MaskCounter = None, force_chain: bool = False) -> dict:
    """La ristima di una cella su una copia in memoria, strato per strato nell'ordine dello stimatore: uno strato riusabile
    prende le statistiche di riferimento (valori installati in float32); ogni altro strato riceve i momenti del suo ingresso
    sulla sorgente con gli strati precedenti già assegnati, in float32, assegnati ai soli moving_mean e moving_variance. Il
    conteggio per canale deve essere viste x frame x H x W (frame per uno strato denso). ctx.step prima di ogni strato."""
    out = {}
    for n in order:
        ctx.step(f"{label} {n}")
        layer = copy.get_layer(n)
        old_mean, old_var = (np.asarray(getattr(layer, a).numpy()).copy() for a in BR.MOVING)
        spatial = int(np.prod(layer.input_shape[1:-1])) if len(layer.input_shape) > 2 else 1
        t0 = ctx.span_start()
        if n in reuse:
            r = reuse[n]
            layer.moving_mean.assign(r["mean32"])
            layer.moving_variance.assign(r["var32"])
            out[n] = {**{k: r[k] for k in ("count", "frames", "spatial", "mean64", "var64", "mean32", "var32")}, "old_mean": old_mean,
                      "old_var": old_var, "views": r.get("views", 1), "reused": True}
            ctx.span_end(t0, label, n)
            continue
        mom = moments_pass(tf, copy, source, (n,), batch, st, dropout_on, counter, force_chain)[n]
        mean, var = mom.result()
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(var))):
            raise ControlFailure("counts", {"cell": label, "layer": n}, f"{label}: {n}: the re-estimated mean or variance is not finite")
        m32, v32 = mean.astype(np.float32), var.astype(np.float32)
        layer.moving_mean.assign(m32)
        layer.moving_variance.assign(v32)
        want = len(source.seeds) * source.n * spatial
        if mom.n != want:
            raise ControlFailure("counts", {"cell": label, "layer": n, "count": mom.n, "expected": want},
                                 f"{label}: {n} accumulated {mom.n} values per channel, expected views x frames x H x W")
        out[n] = {"count": mom.n, "frames": source.n, "spatial": spatial, "old_mean": old_mean, "old_var": old_var, "mean64": mean,
                  "var64": var, "mean32": m32, "var32": v32, "views": len(source.seeds), "reused": False}
        ctx.span_end(t0, label, n)
    return out


# ============================================================================ confronti

def stat_compare(a: dict, b: dict, rel: float, abs_: float, keys=("mean32", "var32")) -> dict:
    """Statistiche di due ristime strato per strato: identiche (byte nel dtype originale), scarto assoluto massimo e canali
    fuori da |a - b| <= rel |b| + abs (un NaN conta come fuori)."""
    out = {"identical": True, "max_abs_diff": 0.0, "outside": 0, "layers": {}}
    for n in b:
        ident, mx, bad = True, 0.0, 0
        for k in keys:
            x, y = np.asarray(a[n][k], np.float64), np.asarray(b[n][k], np.float64)
            if x.shape != y.shape:
                ident, mx, bad = False, float("inf"), bad + max(x.size, y.size)
                continue
            x0, y0 = np.asarray(a[n][k]), np.asarray(b[n][k])
            ident &= x0.dtype == y0.dtype and x0.tobytes() == y0.tobytes()             # byte nel dtype originale
            d = np.abs(x - y)
            mx = max(mx, float(d.max()) if d.size else 0.0)
            bad += int(np.sum(~(d <= rel * np.abs(y) + abs_)))
        out["layers"][n] = {"identical": bool(ident), "max_abs_diff": mx, "outside": bad}
        out["identical"] &= bool(ident)
        out["max_abs_diff"] = max(out["max_abs_diff"], mx)
        out["outside"] += bad
    out["identical"] = bool(out["identical"])
    return out


def score_compare(p, q, y, threshold_q, cfg: dict) -> dict:
    """Punteggi d'attacco p contro punteggi di riferimento q (stessi frame): scarto assoluto massimo, decisioni discordanti
    a 0,5 e alla soglia EER di riferimento (attacco se punteggio > soglia), soglia EER di p contro threshold_q."""
    p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
    tol = cfg["tolerances"]
    if p.shape != q.shape:
        return {"pass": False, "reason": f"{p.shape} scores against {q.shape}"}
    diff = float(np.max(np.abs(p - q))) if len(p) else 0.0
    d05 = int(np.sum(D.attack_decisions(p, FR.FIXED_THRESHOLD) != D.attack_decisions(q, FR.FIXED_THRESHOLD)))
    deer = None if threshold_q is None else int(np.sum(D.attack_decisions(p, threshold_q) != D.attack_decisions(q, threshold_q)))
    thr = eer_or_none(y, p)
    dthr = None if (thr is None or threshold_q is None) else abs(thr - threshold_q)
    ok = diff <= tol["dev_probability_max_abs_diff"] and d05 <= tol["dev_decisions_discordant"] \
        and (deer is None or deer <= tol["dev_decisions_discordant"]) and (dthr is None or dthr <= tol["dev_probability_max_abs_diff"]) \
        and (thr is None) == (threshold_q is None)
    return {"pass": bool(ok), "max_abs_diff": diff, "identical": bool(np.array_equal(p, q)), "discordant_05": d05, "discordant_eer": deer,
            "threshold_eer": thr, "threshold_eer_reference": threshold_q, "threshold_abs_diff": dthr, "reason": None}


def eer_or_none(y, p):
    y = np.asarray(y)
    if not (np.any(y == 1) and np.any(y == 0)):
        return None
    return float(FR.eer_threshold(y, np.asarray(p, np.float64))["value"])


def rates_05(y, p) -> dict:
    """APCER, BPCER e ACER a 0,5 (attacco se punteggio > 0,5, un pareggio è bona fide); None per una classe assente dal dev."""
    y = np.asarray(y, np.int64)
    d = D.attack_decisions(np.asarray(p, np.float64), FR.FIXED_THRESHOLD)
    na, nb = int(np.sum(y == 1)), int(np.sum(y == 0))
    apcer = None if na == 0 else float(np.sum(~d[y == 1]) / na)
    bpcer = None if nb == 0 else float(np.sum(d[y == 0]) / nb)
    acer = None if apcer is None or bpcer is None else (apcer + bpcer) / 2
    return {"n": int(len(y)), "n_attack": na, "n_bona_fide": nb, "apcer_05": apcer, "bpcer_05": bpcer, "acer_05": acer,
            "decided_attack_05": int(np.sum(d))}


def mask_frequency(counter: MaskCounter, active: dict, sigma: float) -> dict:
    """Frequenza degli zeri delle maschere di ogni Dropout attivo contro il suo rate: |zeri - n rate| <= sigma sqrt(n rate (1 - rate))."""
    out = {}
    for layer, rate in active.items():
        z, n = counter.counts.get(layer, (0, 0))
        band = sigma * math.sqrt(n * rate * (1 - rate)) if n else 0.0
        out[layer] = {"zeros": z, "values": n, "rate": rate, "frequency": (z / n) if n else None, "band": band,
                      "pass": bool(n > 0 and abs(z - n * rate) <= band)}
    return out


class ControlFailure(Failure):
    """Un controllo non superato: il comando si ferma come FAIL, status.json nomina il controllo e i suoi valori."""

    def __init__(self, check: str, values: dict, message: str):
        super().__init__(f"control {check} failed: {message}")
        self.check, self.values = check, values


# ============================================================================ run genitori (ARCHRUN)

def parent_file(folder: Path, name: str, suffix: str) -> Path:
    """Un file per unità di un run genitore: solo i suffissi di PARENT_SUFFIXES (nessun file dello scorer, nessun altro)."""
    if suffix not in PARENT_SUFFIXES:
        fail(f"parent file {name}{suffix} refused: this command opens only {', '.join(PARENT_SUFFIXES)} of the parent runs")
    return folder / RESTRICTED / f"{name}{suffix}"


def load_parents(cfg: dict, base: Path, run_cfg_sha: str) -> dict:
    """I due run ARCHRUN: status.json con lo SHA-256 fissato, PASS, stesso run id e stessa configurazione di
    pad_arch_run.json, SHA256SUMS riservato ancorato; per ogni unità dell'ambito i cinque file di PARENT_SUFFIXES elencati e
    uguali alla loro riga; rete, fold, seed, patience e SHA-256 dei pesi della scheda; statistiche salvate e punteggi del dev
    letti (solo righe del dev)."""
    out, status = {}, {}
    for r, e in cfg["parent_runs"].items():
        folder = rel_path(base, e["folder"])
        b = read_regular(folder / STATUS, f"parent {r} status.json")
        if sha256_bytes(b) != e["status_sha256"]:
            fail(f"parent run {e['run_id']}: status.json differs from the pinned SHA-256")
        st = SR.read_json(b, f"parent {r} status.json")
        if st.get("status") != "PASS" or st.get("run_id") != e["run_id"] or st.get("run") != r or st.get("config_sha256") != run_cfg_sha \
                or st.get("restricted_sha256sums_sha256") != e["restricted_sha256sums_sha256"]:
            fail(f"parent run {e['run_id']}: status.json is not a PASS of {r} with the run id, the ARCHRUN configuration and the restricted "
                 "SHA256SUMS of this configuration")
        sb = read_regular(folder / RESTRICTED / "SHA256SUMS", f"parent {r} restricted SHA256SUMS")
        if sha256_bytes(sb) != e["restricted_sha256sums_sha256"]:
            fail(f"parent run {e['run_id']}: restricted SHA256SUMS differs from the pinned SHA-256")
        status[r] = {"folder": folder, "sums": FR.read_sums(sb, f"parent {r} restricted SHA256SUMS"), "status": st}
    for u in cfg["scope"]["units"]:
        name, p = unit_name(u), status[u["run"]]
        files = {}
        for suf in PARENT_SUFFIXES:
            fn = f"{name}{suf}"
            if fn not in p["sums"]:
                fail(f"parent run {u['run']}: {fn} not listed in its restricted SHA256SUMS")
            path = parent_file(p["folder"], name, suf)
            if suf == ".keras":
                h = sha256_file(FR.regular_path(path, f"parent {fn}"))
            else:
                files[suf] = read_regular(path, f"parent {fn}")
                h = sha256_bytes(files[suf])
            if h != p["sums"][fn]:
                fail(f"parent {fn}: SHA-256 differs from the restricted SHA256SUMS of its run")
        card = SR.read_json(files["_card.json"], f"{name}_card.json")
        w = p["sums"][f"{name}.keras"]
        if (card.get("network"), card.get("fold"), card.get("train_seed"), card.get("early_stopping_patience"), card.get("weights_sha256")) \
                != (u["network"], u["fold"], u["seed"], PARENT_RUNS[u["run"]], w):
            fail(f"parent {name}_card.json: network, fold, seed, patience or weights SHA-256 differ from the unit and the checkpoint")
        if not isinstance(card.get("restored_epoch"), int) or card["restored_epoch"] < 1 or not FR.is_count(card.get("n_fitting")) \
                or not FR.is_count(card.get("batch")):
            fail(f"parent {name}_card.json: restored_epoch, n_fitting or batch missing")
        stats = SR.read_json(files["_bn_statistics.json"], f"{name}_bn_statistics.json")
        order = cfg["estimator"]["orders"][u["network"]]
        if stats.get("weights_sha256") != w or stats.get("order") != order or list(stats.get("layers") or {}) != order:
            fail(f"parent {name}_bn_statistics.json: weights or order differ")
        rows = {}
        for suf, state in (("_dev_ordinary.csv", "ordinary"), ("_dev_re-estimated.csv", "re-estimated")):
            rr = parse_csv(files[suf], PR.SCORE_COLUMNS, f"{name}{suf}")
            if any((x["network"], x["fold"], x["train_seed"], x["bn_state"], x["role"], x["checkpoint_sha256"])
                   != (u["network"], str(u["fold"]), str(u["seed"]), state, "dev", w) for x in rr):
                fail(f"parent {name}{suf}: rows of another unit, state or role")
            rows[state] = rr
        out[unit_label(u)] = {"unit": u, "name": name, "weights": parent_file(p["folder"], name, ".keras"), "weights_sha256": w, "card": card,
                              "stats": stats, "rows": rows}
    return {"units": out, "runs": {r: {"run_id": cfg["parent_runs"][r]["run_id"], "status_sha256": cfg["parent_runs"][r]["status_sha256"],
                                       "restricted_sha256sums_sha256": cfg["parent_runs"][r]["restricted_sha256sums_sha256"]} for r in status}}


def saved_scores(rows, dev: dict, what: str) -> tuple:
    """(punteggi, soglia EER) di un file per frame salvato, con i frame nell'ordine della parte del dev."""
    if [r["sample_id"] for r in rows] != dev["keys"]:
        fail(f"{what}: the frames differ from the dev manifest in key order")
    try:
        s = np.array([float(r["score_attack"]) for r in rows], np.float64)
        thr = {r["threshold_dev_eer"] for r in rows}
    except ValueError:
        fail(f"{what}: score_attack not numeric")
    if len(thr) != 1:
        fail(f"{what}: more than one EER threshold")
    t = thr.pop()
    return s, (None if t == "NA" else float(t))


def saved_stats(doc: dict, order) -> dict:
    return {n: {"mean32": np.array(doc["layers"][n]["installed_moving_mean_float32"], np.float32),
                "var32": np.array(doc["layers"][n]["installed_moving_variance_float32"], np.float32),
                "old_mean": np.array(doc["layers"][n]["old_moving_mean"], np.float32),
                "old_var": np.array(doc["layers"][n]["old_moving_variance"], np.float32)} for n in order}


def unit_keys_check(inp, parents: dict) -> dict:
    """I frame di fitting e di dev di ogni unità uguali alle chiavi della sua scheda (BR.keys_sha sui manifest)."""
    for p in parents["units"].values():
        for role in ("fitting", "dev"):
            rows = BR.role_rows(inp, role, PR.ARM, p["unit"]["fold"])
            if BR.keys_sha(rows) != p["card"].get(f"{role}_keys_sha256"):
                fail(f"{p['name']}: the {role} keys of the manifests differ from {role}_keys_sha256 of its parent card")
    return {"units": len(parents["units"])}


# ============================================================================ contesto di una prova

class Ctx:
    """Lo stato di una prova: configurazione, tetto, limite di memoria, tempi, TensorFlow."""

    def __init__(self, cfg: dict, budget, tf=None):
        self.cfg, self.budget, self.tf = cfg, budget, tf
        self.timing = []
        self.unit = None
        self.peak_gb = 0.0

    def memory_gb(self) -> float:
        return MEMORY_PROBE() if MEMORY_PROBE is not None else SR.peak_rss_mb() / 1024

    def step(self, what: str) -> None:
        """I controlli prima di ogni passo: tetto di tempo e SIGUSR1 (INCOMPLETE), memoria residente di picco (FAIL)."""
        self.budget.check(what)
        if GUARD["hits"]:                              # un'apertura rifiutata che una libreria avesse ingoiato
            raise ControlFailure("read_guard", {"refused_opens": len(GUARD["hits"]), "step": what},
                                 f"{len(GUARD['hits'])} refused opens before {what}")
        gb = self.memory_gb()
        self.peak_gb = max(self.peak_gb, gb)
        lim = self.cfg["memory"]["peak_rss_limit_gb"]
        if gb > lim:
            raise ControlFailure("memory", {"peak_rss_gb": round(gb, 3), "limit_gb": lim, "step": what},
                                 f"peak resident memory {gb:.2f} GB above the declared limit {lim} GB at {what}")

    def span_start(self) -> tuple:
        return (now(), time.monotonic(), PAC.cpu_seconds())

    def span_end(self, t0: tuple, cell: str, layer: str) -> dict:
        if self.unit and cell.startswith(f"{self.unit} "):
            cell = cell[len(self.unit) + 1:]
        rec = {"unit": self.unit, "cell": cell, "layer": layer, "started": t0[0], "finished": now(),
               "wall_seconds": round(time.monotonic() - t0[1], 3), "process_cpu_seconds": round(PAC.cpu_seconds() - t0[2], 3)}
        self.timing.append(rec)
        return rec


# ============================================================================ una cella

def run_cell(ctx, copy, orig_snap: dict, cell: str, label: str, source: Source, fit: dict, dev: dict, st: dict, reuse: dict) -> dict:
    """Una cella stocastica su una copia nuova: ristima, controlli della PR #27 (solo i tensori mobili cambiati e uguali ai
    valori installati; coerenza di un passaggio a più uscite sulla stessa sorgente, viste e maschere; conteggi; softmax),
    il dev valutato con augmentation e dropout spenti (BR.evaluate, invariato) e, se configurato, il fitting pulito."""
    tf, cfg = ctx.tf, ctx.cfg
    order, batch = cfg["estimator"]["orders"][ctx.network], cfg["estimator"]["batch"]
    t0 = ctx.span_start()
    counter = MaskCounter()
    drop = FACTORS[cell][1]
    stats = reestimate_cell(tf, copy, source, order, batch, st, drop, reuse, ctx, label, counter)
    if probs := BR.tensor_problems(orig_snap, BR.snapshot_weights(copy), BR.moving_keys(copy, order), stats):
        raise ControlFailure("tensors", {"cell": label, "problems": probs}, f"{label}: " + "; ".join(probs))
    ctx.step(f"{label} consistency")
    tol = cfg["tolerances"]
    final = moments_pass(tf, copy, source, order, batch, st, drop)
    cons = BR.consistency(stats, final, tol["consistency_rel"], tol["consistency_abs"])
    if bad := {n: c["channels_outside_tolerance"] for n, c in cons.items() if c["channels_outside_tolerance"]}:
        raise ControlFailure("consistency", {"cell": label, "channels_outside": bad},
                             f"{label}: channel statistics outside rel {tol['consistency_rel']} + abs {tol['consistency_abs']}")
    want = {n: len(source.seeds) * source.n * stats[n]["spatial"] for n in order}
    if any(cons[n]["count"] != want[n] for n in order):
        raise ControlFailure("counts", {"cell": label}, f"{label}: the consistency pass counted a different number of values per channel")
    ctx.step(f"{label} dev")
    dev_ev, _ = BR.evaluate(copy, dev, batch, f"{label} dev")
    fit_ev = None
    if cfg["evaluation"]["fitting_clean"]:
        ctx.step(f"{label} clean fitting")
        fit_ev, _ = BR.evaluate(copy, fit, batch, f"{label} clean fitting")
    for r, what in ((dev_ev, "dev"), (fit_ev, "fitting")):
        if r is not None and r["softmax_gap"] > tol["softmax_of_logits_max_abs_diff"]:
            raise ControlFailure("softmax", {"cell": label, "part": what, "softmax_gap": r["softmax_gap"]},
                                 f"{label}: the softmax of the logits differs from the {what} probabilities by {r['softmax_gap']:.2e}")
    freq = mask_frequency(counter, st["active"], cfg["dropout"]["mask_frequency_sigma"]) if drop else {}
    if any(not f["pass"] for f in freq.values()):
        raise ControlFailure("masks", {"cell": label, "layers": freq}, f"{label}: mask zero frequency outside the declared band")
    sp = ctx.span_end(t0, label, "(cell)")
    progress(f"{label}: cell done in {sp['wall_seconds']:.0f} s")
    return {"cell": cell, "label": label, "stats": stats, "cons": cons, "dev": dev_ev, "fit": fit_ev, "masks": freq,
            "views": list(source.seeds), "source": source.kind}


def progress(msg: str) -> None:
    """Una riga di avanzamento nel log della prova (unità, cella, controllo, secondi; nessun percorso, nessun identificativo)."""
    print(f"   {msg}", flush=True)


def fresh_copy(tf, orig, snap: dict, what: str):
    copy = tf.keras.models.clone_model(orig)
    copy.set_weights(orig.get_weights())
    if BR.snapshot_weights(copy) != snap:
        raise ControlFailure("weights", {"unit": what}, f"{what}: the in-memory copy differs from the original before the re-estimation")
    return copy


def trainable_sha256(model) -> str:
    h = hashlib.sha256()
    for w in model.trainable_weights:
        a = np.asarray(w.numpy())
        h.update(w.name.encode() + a.dtype.str.encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


# ============================================================================ un'unità

def flags_for(cfg: dict, u: dict, first: dict) -> dict:
    """Che cosa gira per un'unità oltre alle quattro celle: duplicati espliciti, metà Monte Carlo, ripetizione."""
    net = u["network"]
    dup = cfg["reuse"]["duplicate_check"]
    de = cfg["determinism"]
    return {"duplicates": dup == "all" or (dup == "first" and first.get(("dup", net)) is None),
            "halves": u["run"] in cfg["monte_carlo"]["runs"],
            "repeat": net in de["networks"] and (de["units"] == "all" or first.get(("rep", net)) is None)}


class StepBudget:
    """L'interfaccia del tetto di BR.recalibrate (check(step), chiamato prima di ogni strato) sopra ctx.step: tetto,
    SIGUSR1 e memoria prima di ogni strato; il tempo di ogni strato è quello fra due controlli (finish() chiude l'ultimo)."""

    def __init__(self, ctx, label: str, order):
        self.ctx, self.label, self.order, self.i, self.t0 = ctx, label, list(order), 0, None

    def check(self, step: str) -> None:
        self.finish()
        self.ctx.step(step)
        self.t0 = self.ctx.span_start()

    def finish(self) -> None:
        if self.t0 is not None:
            self.ctx.span_end(self.t0, self.label, self.order[self.i])
            self.i += 1
            self.t0 = None


def process_unit(ctx, parent: dict, fit: dict, dev: dict, views, flags: dict) -> dict:
    """I controlli e le celle di un checkpoint (l'ordine è nella docstring del modulo)."""
    tf, cfg = ctx.tf, ctx.cfg
    u = parent["unit"]
    net, label = u["network"], unit_label(u)
    ctx.unit, ctx.network = label, net
    tol = cfg["tolerances"]
    order, batch = cfg["estimator"]["orders"][net], cfg["estimator"]["batch"]
    t_unit = ctx.span_start()
    rec = {"label": label, "unit": u, "cells": {}, "controls": {}, "complete": False}
    ctx.step(f"load {label}")
    tf.keras.backend.clear_session()
    orig = BR.load_model(parent["weights"], parent["weights_sha256"])
    (CHECK_LOADED or PR.check_loaded)(orig, ctx.cfg_pr, net)
    st = structure(orig, net, cfg)
    rec["controls"]["structure"] = {"pass": True, "active_dropout": st["active"], "chain": st["chain"] is not None}
    snap = BR.snapshot_weights(orig)
    rec["trainable_sha256"] = trainable_sha256(orig)
    card = parent["card"]
    e = card["restored_epoch"]
    upe = cfg["ema"]["updates_per_epoch"]
    if math.ceil(card["n_fitting"] / card["batch"]) != upe or card["n_fitting"] != len(fit["X"]):
        raise ControlFailure("ema", {"unit": label, "n_fitting": card["n_fitting"], "batch": card["batch"], "updates_per_epoch": upe},
                             f"{label}: the card's fitting frames and batch do not give {upe} updates per epoch, or differ from the fitting part")
    rec["ema"] = {"restored_epoch": e, "updates_per_epoch": upe, "initialization_coefficient": cfg["ema"]["momentum"] ** (upe * e)}
    # baseline: il dev ordinario contro il file salvato
    ctx.step(f"{label} ordinary dev")
    ev_ord, _ = BR.evaluate(orig, dev, batch, f"{label} dev ordinary")
    s_ord, thr_ord = saved_scores(parent["rows"]["ordinary"], dev, f"{parent['name']}_dev_ordinary.csv")
    c = score_compare(ev_ord["p"][:, 1], s_ord, dev["y"], thr_ord, cfg)
    rec["controls"]["baseline"] = c
    if not c["pass"]:
        raise ControlFailure("baseline", {"unit": label, **c}, f"{label}: the ordinary dev differs from the saved _dev_ordinary.csv")
    progress(f"{label}: baseline reproduced (max |dp| {c['max_abs_diff']:.1e})")
    # E00: lo stimatore esistente, invariato
    t0 = ctx.span_start()
    copy = fresh_copy(tf, orig, snap, label)
    ctx.step(f"{label} E00")
    sb = StepBudget(ctx, f"{label} E00", order)
    stats00 = PR.reestimate(copy, fit, order, batch, sb, label)
    sb.finish()
    ctx.span_end(t0, "E00", "(re-estimation)")
    ctx.step(f"{label} E00 consistency")
    fit00, final00 = BR.evaluate(copy, fit, batch, f"{label} E00 fitting", layers=order)
    ctx.step(f"{label} E00 dev")
    dev00, _ = BR.evaluate(copy, dev, batch, f"{label} E00 dev")
    PR.pr27_checks(label, stats00, snap, copy, order, fit00, final00, dev00, ctx.bn_cfg)
    saved = saved_stats(parent["stats"], order)
    sc = stat_compare(stats00, saved, tol["statistics_rel"], tol["statistics_abs"])
    old = stat_compare({n: {"mean32": stats00[n]["old_mean"], "var32": stats00[n]["old_var"]} for n in order},
                       {n: {"mean32": saved[n]["old_mean"], "var32": saved[n]["old_var"]} for n in order}, 0.0, 0.0)
    s_re, thr_re = saved_scores(parent["rows"]["re-estimated"], dev, f"{parent['name']}_dev_re-estimated.csv")
    dc = score_compare(dev00["p"][:, 1], s_re, dev["y"], thr_re, cfg)
    rec["controls"]["E00"] = {"statistics": {k: sc[k] for k in ("identical", "max_abs_diff", "outside")}, "stored_statistics_identical": old["identical"],
                              "dev": dc, "pass": bool(sc["outside"] == 0 and old["identical"] and dc["pass"])}
    if not rec["controls"]["E00"]["pass"]:
        raise ControlFailure("E00", {"unit": label, **rec["controls"]["E00"]},
                             f"{label}: E00 does not reproduce the saved statistics or dev scores within the tolerances")
    e00 = {"cell": "E00", "label": f"{label} E00", "stats": {n: {**stats00[n], "views": 1, "reused": False} for n in order},
           "cons": BR.consistency(stats00, final00, tol["consistency_rel"], tol["consistency_abs"]), "dev": dev00,
           "fit": fit00 if cfg["evaluation"]["fitting_clean"] else None, "masks": {}, "views": [], "source": "clean"}
    rec["cells"][("E00", ALL_VIEWS)] = e00
    progress(f"{label}: E00 reproduced (statistics identical: {'yes' if sc['identical'] else 'no'}, max |dp| {dc['max_abs_diff']:.1e})")
    del copy, final00
    clean8 = Source.clean(fit["X"], fit["digests"], VIEW_SEEDS)
    if st["chain"] is not None:                       # E00 lungo la catena (una passata pulita, dropout spento): un controllo
        t0 = ctx.span_start()
        copy = fresh_copy(tf, orig, snap, label)
        chain_stats = reestimate_cell(tf, copy, Source.clean(fit["X"], fit["digests"], (0,)), order, batch, st, False, {}, ctx,
                                      f"{label} E00 chain", force_chain=True)
        dev_c, _ = BR.evaluate(copy, dev, batch, f"{label} E00 chain dev")
        cc = stat_compare(chain_stats, stats00, tol["statistics_rel"], tol["statistics_abs"])
        pc = score_compare(dev_c["p"][:, 1], dev00["p"][:, 1], dev["y"], eer_or_none(dev["y"], dev00["p"][:, 1]), cfg)
        rec["controls"]["E00_chain"] = {"statistics": {k: cc[k] for k in ("identical", "max_abs_diff", "outside")}, "dev": pc,
                                        "pass": bool(cc["outside"] == 0 and pc["pass"])}
        ctx.span_end(t0, "E00 chain", "(control)")
        del copy
        if not rec["controls"]["E00_chain"]["pass"]:
            raise ControlFailure("E00_chain", {"unit": label, **rec["controls"]["E00_chain"]}, f"{label}: the chain does not reproduce E00")
        progress(f"{label}: E00 through the chain reproduced")
    aug8 = Source.augmented(views, fit["digests"], VIEW_SEEDS)
    sources = {"E10": aug8, "E01": clean8, "E11": aug8}
    for cell in ("E10", "E01", "E11"):
        ref = REFERENCE.get(cell)
        elig = reuse_eligible(st, cell, order)
        if ref is not None and elig == list(order):      # nessuna operazione stocastica diversa in nessuno strato: la cella di riferimento
            rec["cells"][(cell, ALL_VIEWS)] = {**rec["cells"][(ref, ALL_VIEWS)], "cell": cell, "label": f"{label} {cell}", "reused_cell": ref,
                                               "stats": {n: {**s, "reused": True} for n, s in rec["cells"][(ref, ALL_VIEWS)]["stats"].items()}}
            continue
        reuse = {n: rec["cells"][(ref, ALL_VIEWS)]["stats"][n] for n in elig} if ref else {}
        copy = fresh_copy(tf, orig, snap, label)
        rec["cells"][(cell, ALL_VIEWS)] = run_cell(ctx, copy, snap, cell, f"{label} {cell}", sources[cell], fit, dev, st, reuse)
        del copy
        gc.collect()
    # duplicati espliciti degli strati riusabili (rilievo 7 della revisione del piano)
    if flags["duplicates"]:
        rec["controls"]["duplicates"] = {}
        for cell in ("E01", "E11"):
            ref = REFERENCE[cell]
            elig = reuse_eligible(st, cell, order)
            if not elig:
                continue
            copy = fresh_copy(tf, orig, snap, label)
            t0 = ctx.span_start()
            shadow = reestimate_cell(tf, copy, sources[cell], elig, batch, st, FACTORS[cell][1], {}, ctx, f"{label} {cell} explicit")
            refst = {n: rec["cells"][(ref, ALL_VIEWS)]["stats"][n] for n in elig}
            cmpd = stat_compare(shadow, refst, tol["statistics_rel"], tol["statistics_abs"])
            entry = {"layers": elig, "statistics": {k: cmpd[k] for k in ("identical", "max_abs_diff", "outside")}}
            if elig == list(order):
                d, _ = BR.evaluate(copy, dev, batch, f"{label} {cell} explicit dev")
                rp = rec["cells"][(ref, ALL_VIEWS)]["dev"]["p"][:, 1]
                entry["dev"] = score_compare(d["p"][:, 1], rp, dev["y"], eer_or_none(dev["y"], rp), cfg)
            entry["pass"] = bool(cmpd["outside"] == 0 and entry.get("dev", {"pass": True})["pass"])
            rec["controls"]["duplicates"][cell] = entry
            progress(f"{label}: explicit {cell} {'within' if entry['pass'] else 'OUTSIDE'} the reference tolerance")
            ctx.span_end(t0, f"{cell} explicit", "(control)")
            del copy
            if not entry["pass"]:
                raise ControlFailure("duplicates", {"unit": label, "cell": cell, **entry},
                                     f"{label}: the explicit {cell} differs from {ref} on the reusable layers beyond the reference tolerance")
    # metà Monte Carlo: due ristime separate, viste 1-4 e 5-8
    if flags["halves"]:
        for cell in ("E10", "E01", "E11"):
            for h in HALVES:
                tag = f"{h[0]}-{h[-1]}"
                ref = REFERENCE.get(cell)
                elig = reuse_eligible(st, cell, order)
                if ref is not None and elig == list(order):
                    rc = rec["cells"][(ref, tag if ref != "E00" else ALL_VIEWS)]
                    rec["cells"][(cell, tag)] = {**rc, "cell": cell, "label": f"{label} {cell} {tag}", "reused_cell": ref,
                                                 "stats": {n: {**s, "reused": True} for n, s in rc["stats"].items()}}
                    continue
                reuse = {}
                if ref is not None:
                    rcell = rec["cells"][(ref, tag if ref != "E00" else ALL_VIEWS)]
                    reuse = {n: rcell["stats"][n] for n in elig}
                src = Source.augmented(views, fit["digests"], h) if FACTORS[cell][0] else Source.clean(fit["X"], fit["digests"], h)
                copy = fresh_copy(tf, orig, snap, label)
                rec["cells"][(cell, tag)] = run_cell(ctx, copy, snap, cell, f"{label} {cell} {tag}", src, fit, dev, st, reuse)
                del copy
                gc.collect()
    # ripetizione: deve essere identica bit per bit (S10), con il primo batch di ogni vista rigenerato
    if flags["repeat"]:
        cell = cfg["determinism"]["cell"]
        ref = REFERENCE.get(cell)
        reuse = {n: rec["cells"][(ref, ALL_VIEWS)]["stats"][n] for n in reuse_eligible(st, cell, order)} if ref else {}
        copy = fresh_copy(tf, orig, snap, label)
        r2 = run_cell(ctx, copy, snap, cell, f"{label} {cell} repeat", sources[cell], fit, dev, st, reuse)
        del copy
        s2 = REPEAT_HOOK(r2["stats"]) if REPEAT_HOOK is not None else r2["stats"]
        first = rec["cells"][(cell, ALL_VIEWS)]
        sc2 = stat_compare(s2, first["stats"], 0.0, 0.0, keys=("mean64", "var64", "mean32", "var32"))
        same_p = bool(np.array_equal(r2["dev"]["p"], first["dev"]["p"]))
        regen = views.regenerated_first_batches(tf) if FACTORS[cell][0] else {}
        rec["controls"]["determinism"] = {"cell": cell, "statistics_identical": sc2["identical"], "dev_identical": same_p,
                                          "max_abs_diff": sc2["max_abs_diff"], "views_regenerated_identical": all(regen.values()),
                                          "pass": bool(sc2["identical"] and same_p and all(regen.values()))}
        progress(f"{label}: repetition of {cell} {'bit-identical' if rec['controls']['determinism']['pass'] else 'NOT identical'}")
        if not rec["controls"]["determinism"]["pass"]:
            raise ControlFailure("determinism", {"unit": label, **rec["controls"]["determinism"]},
                                 f"{label}: the repetition of {cell} is not bit-identical")
    if BR.snapshot_weights(orig) != snap or trainable_sha256(orig) != rec["trainable_sha256"]:
        raise ControlFailure("weights", {"unit": label, "part": "model"}, f"{label}: the original model changed during the trial")
    if sha256_file(parent["weights"]) != parent["weights_sha256"]:
        raise ControlFailure("weights", {"unit": label, "part": "checkpoint file"}, f"{label}: the checkpoint file changed during the trial")
    rec["controls"]["weights"] = {"trainable_sha256": rec["trainable_sha256"], "unchanged": True, "checkpoint_sha256": parent["weights_sha256"]}
    del orig
    tf.keras.backend.clear_session()
    gc.collect()
    rec["complete"] = True
    ctx.span_end(t_unit, "(unit)", "(unit)")
    return rec


# ============================================================================ record e uscite

def cell_metrics(c: dict, dev: dict, e00: dict) -> dict:
    p = c["dev"]["p"][:, 1]
    r = rates_05(dev["y"], p)
    q = e00["dev"]["p"][:, 1]
    d = np.abs(p - q)
    return {**r, "auroc": c["dev"]["auroc"], "val_loss": c["dev"]["val_loss"], "accuracy": c["dev"]["accuracy"],
            "eer_threshold": eer_or_none(dev["y"], p), "max_abs_dp_vs_e00": float(d.max()) if len(d) else 0.0,
            "mean_abs_dp_vs_e00": float(d.mean()) if len(d) else 0.0,
            "decisions_changed_05_vs_e00": int(np.sum(D.attack_decisions(p, 0.5) != D.attack_decisions(q, 0.5))),
            "fitting_accuracy": None if c["fit"] is None else c["fit"]["accuracy"], "fitting_loss": None if c["fit"] is None else c["fit"]["val_loss"],
            "layers_reused": sum(1 for s in c["stats"].values() if s.get("reused"))}


CELL_COLUMNS = ("run", "network", "fold", "seed", "cell", "views", "n", "n_attack", "n_bona_fide", "apcer_05", "bpcer_05", "acer_05",
                "decided_attack_05", "auroc", "val_loss", "accuracy", "eer_threshold", "max_abs_dp_vs_e00", "mean_abs_dp_vs_e00",
                "decisions_changed_05_vs_e00", "fitting_accuracy", "fitting_loss", "layers_reused", "restored_epoch", "ema_coefficient")
EFFECT_COLUMNS = ("run", "network", "fold", "seed", "views", "metric", "augmentation_E10_minus_E00", "dropout_E01_minus_E00",
                  "interaction_E11_minus_E10_minus_E01_plus_E00")
EFFECTS = ("augmentation", "dropout", "interaction")
MC_COLUMNS = ("run", "network", "metric", "effect", "level", "units", "effect_value", "h_halves", "abs_h_bar_half", "s_mc",
              "ratio_to_s_mc", "within_view_variability")
LAYER_COLUMNS = ("run", "network", "fold", "seed", "cell", "views", "layer", "reused", "count", "views_used", "var_ratio_min",
                 "var_ratio_median", "var_ratio_max", "mean_shift_min", "mean_shift_median", "mean_shift_max", "vs_e00_var_ratio_median",
                 "vs_e00_mean_shift_median", "consistency_max_abs_diff_mean", "consistency_max_abs_diff_var")
CONTROL_COLUMNS = ("run", "network", "fold", "seed", "control", "pass", "detail")
TIMING_COLUMNS = ("unit", "cell", "layer", "started", "finished", "wall_seconds", "process_cpu_seconds")


def cell(v) -> str:
    return PR.cell(v)


def rows_for(rec: dict, dev: dict, eps: float) -> tuple:
    u = rec["unit"]
    head = [u["run"], u["network"], u["fold"], u["seed"]]
    e00 = rec["cells"][("E00", ALL_VIEWS)]
    cells, layers, metrics = [], [], {}
    for (c, views), r in rec["cells"].items():
        m = cell_metrics(r, dev, e00)
        metrics[(c, views)] = m
        cells.append([cell(x) for x in head + [c, views] + [m[k] for k in CELL_COLUMNS[6:23]]
                      + [rec["ema"]["restored_epoch"], rec["ema"]["initialization_coefficient"]]])
        for n, s in r["stats"].items():
            ch = BR.change_summary(s, eps)
            ve = BR.change_summary({**s, "old_mean": e00["stats"][n]["mean32"], "old_var": e00["stats"][n]["var32"]}, eps)
            cn = r["cons"].get(n, {})
            layers.append([cell(x) for x in head + [c, views, n, bool(s.get("reused")), s["count"], s.get("views", 1), ch["var_ratio_min"],
                                                     ch["var_ratio_median"], ch["var_ratio_max"], ch["mean_shift_min"], ch["mean_shift_median"],
                                                     ch["mean_shift_max"], ve["var_ratio_median"], ve["mean_shift_median"],
                                                     cn.get("max_abs_diff_mean"), cn.get("max_abs_diff_var")]])
    effects, numeric = [], {}
    for views in [ALL_VIEWS] + [f"{h[0]}-{h[-1]}" for h in HALVES]:
        def get(c, views=views):
            return metrics.get((c, views if c != "E00" else ALL_VIEWS))
        if any(get(c) is None for c in CELLS):
            continue
        for k in ("acer_05", "apcer_05", "bpcer_05"):
            v = {c: get(c)[k] for c in CELLS}
            if any(x is None for x in v.values()):
                eff = [None, None, None]
            else:
                eff = [v["E10"] - v["E00"], v["E01"] - v["E00"], v["E11"] - v["E10"] - v["E01"] + v["E00"]]
            effects.append([cell(x) for x in head + [views, k] + eff])
            numeric[(views, k)] = eff
    return cells, effects, layers, effect_record(rec, numeric)


def effect_record(rec: dict, numeric: dict) -> list:
    """Per metrica ed effetto di un'unità con le metà: l'effetto a 8 viste e H_u = effetto(101-104) - effetto(105-108)."""
    u = rec["unit"]
    out = []
    for k in ("acer_05", "apcer_05", "bpcer_05"):
        a, h1, h2 = (numeric.get((v, k)) for v in (ALL_VIEWS, "101-104", "105-108"))
        if a is None or h1 is None or h2 is None:
            continue
        for i, e in enumerate(EFFECTS):
            if None in (a[i], h1[i], h2[i]):
                continue
            out.append({"run": u["run"], "network": u["network"], "label": rec["label"], "metric": k, "effect": e, "value": a[i],
                        "h": h1[i] - h2[i]})
    return out


def montecarlo_rows(records: list, k_within: float = 2.0) -> list:
    """Il confronto Monte Carlo di NOTE.md §6.4, sul dev. Per run, rete, metrica ed effetto: s_MC = sqrt(media sulle unità
    di H_u²) / 2, la deviazione standard di un effetto a 8 viste di una singola unità (al primo ordine); per ogni unità
    l'effetto, H_u, il rapporto |effetto| / s_MC e se è entro k_within x s_MC; per la media sulle unità la media
    dell'effetto, H̄ (media di H_u), |H̄| / 2 e il confronto con s_MC non ridotta (le unità condividono viste e maschere
    attraverso i frame di fitting comuni: nessuna divisione per la radice del numero di unità)."""
    groups = {}
    for r in records:
        groups.setdefault((r["run"], r["network"], r["metric"], r["effect"]), []).append(r)
    rows = []

    def judge(x, s):
        if s == 0:                                     # nessuna variabilità osservata: con un effetto non nullo, NA
            return None, (True if x == 0 else None)
        return abs(x) / s, abs(x) <= k_within * s
    for (run, net, k, e), rs in sorted(groups.items()):
        s = math.sqrt(sum(r["h"] ** 2 for r in rs) / len(rs)) / 2
        for r in rs:
            ratio, within = judge(r["value"], s)
            rows.append([cell(x) for x in (run, net, k, e, r["label"], 1, r["value"], r["h"], None, s, ratio, within)])
        mean = sum(r["value"] for r in rs) / len(rs)
        hbar = sum(r["h"] for r in rs) / len(rs)
        ratio, within = judge(mean, s)
        rows.append([cell(x) for x in (run, net, k, e, "mean", len(rs), mean, hbar, abs(hbar) / 2, s, ratio, within)])
    return rows


def control_rows(rec: dict) -> list:
    u = rec["unit"]
    out = []
    for k, v in rec["controls"].items():
        if k == "duplicates":
            for c, e in v.items():
                out.append([cell(x) for x in [u["run"], u["network"], u["fold"], u["seed"], f"duplicates_{c}", e["pass"], json.dumps(strip(e))]])
            continue
        ok = v.get("pass", True) if isinstance(v, dict) else True
        out.append([cell(x) for x in [u["run"], u["network"], u["fold"], u["seed"], k, ok, json.dumps(strip(v))]])
    for (c, views), r in rec["cells"].items():
        if r.get("masks") and not r.get("reused_cell"):
            out.append([cell(x) for x in [u["run"], u["network"], u["fold"], u["seed"], f"masks_{c}_{views}",
                                          all(m["pass"] for m in r["masks"].values()), json.dumps(strip(r["masks"]))]])
    return out


def strip(v):
    """Una copia di un record di controllo serializzabile in JSON (float, interi, testi, liste, dizionari; scalari di NumPy convertiti)."""
    if isinstance(v, dict):
        return {str(k): strip(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [strip(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def restricted_files(rec: dict, dev: dict, parent: dict) -> dict:
    """Per cella e insieme di viste: i punteggi del dev per frame (con gli identificativi dei frame) e le statistiche per
    strato (valori installati in float32, valori in float64, valori conservati), quanto basta a ricostruire lo stato dal
    checkpoint."""
    out = {}
    u = rec["unit"]
    for (c, views), r in rec["cells"].items():
        p = r["dev"]["p"][:, 1]
        z = r["dev"]["z"]
        rows = [[u["network"], str(u["fold"]), str(u["seed"]), c, views, "dev", row["key"], row["label"], repr(float(p[i])), repr(float(z[i][0])),
                 repr(float(z[i][1])), "1" if p[i] > FR.FIXED_THRESHOLD else "0", parent["weights_sha256"]] for i, row in enumerate(dev["rows"])]
        out[f"{rec['label']}_{c}_{views}_dev.csv"] = csv_text(("network", "fold", "train_seed", "cell", "views", "role", D.FRAME_ID_FIELD, "label",
                                                               "score_attack", "logit_bona_fide", "logit_attack", "prediction_05",
                                                               "checkpoint_sha256"), rows)
        lay = {n: {"reused": bool(s.get("reused")), "count": int(s["count"]), "views": int(s.get("views", 1)),
                   "installed_moving_mean_float32": [float(x) for x in s["mean32"]], "installed_moving_variance_float32": [float(x) for x in s["var32"]],
                   "mean_float64": [float(x) for x in s["mean64"]], "variance_float64": [float(x) for x in s["var64"]],
                   "stored_moving_mean": [float(x) for x in s["old_mean"]], "stored_moving_variance": [float(x) for x in s["old_var"]]}
               for n, s in r["stats"].items()}
        out[f"{rec['label']}_{c}_{views}_bn.json"] = PAC.json_text({"unit": rec["label"], "cell": c, "views": views, "weights_sha256": parent["weights_sha256"],
                                                                    "layers": lay, "reproduce": "load the checkpoint and assign the installed "
                                                                    "float32 values to moving_mean and moving_variance"}, "bn json")
    return out


def report_md(doc: dict) -> str:
    L = [f"# 2x2 BN re-estimation on fitting/dev ({doc['run_id']})", "",
         f"Status: **{doc['status']}**. Scope {doc['scope']['name']}: {len(doc['scope']['units'])} units, {doc['complete_units']} complete. "
         "No test frame is assembled or evaluated, no test score or metric is read; every number is a fitting or development number "
         "(the development frames were also the early-stopping set of the checkpoints: in-sample).", "",
         f"Started {doc['started']}, finished {doc['finished']}; wall time {doc['wall_seconds']:.0f} s of a cap of {doc['cap_minutes']} minutes; "
         f"process CPU time (all threads) {doc['process_cpu_seconds']:.0f} s; peak resident memory {doc['peak_rss_gb']:.2f} GB.", ""]
    if doc["stopped"]:
        L += [f"Stopped: {doc['stopped']['cause']} at {doc['stopped']['step']} ({doc['stopped']['elapsed_minutes']} min).", ""]
    L += ["## Controls", "", "| run | network | fold | seed | control | pass |", "|---|---|---|---|---|---|"]
    for r in doc["controls"]:
        L.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |")
    L += ["", "## Time per unit", "", "| unit | wall s | CPU s |", "|---|---|---|"]
    for t in doc["timing_units"]:
        L.append(f"| {t['unit']} | {t['wall_seconds']:.1f} | {t['process_cpu_seconds']:.1f} |")
    L += ["", "Cell metrics: " + ("public (cells.csv, effects.csv, montecarlo.csv)." if doc["scope"]["report_cell_metrics"] else
                                  "sealed in restricted/ (scope.report_cell_metrics false); cells.csv, effects.csv and montecarlo.csv hold the header only."), ""]
    return "\n".join(L)


# ============================================================================ comandi

def repo_state(root: Path) -> dict:
    return REPO_STATE(root) if REPO_STATE is not None else SR.repo_state(root)


def resolve(a) -> tuple:
    base = Path(a.input_root).resolve() if a.input_root else ROOT
    cfg_p = Path(a.config) if Path(a.config).is_absolute() else ROOT / a.config
    return base, cfg_p


def declared_inputs(cfg: dict, base: Path, run_cfg: dict) -> list:
    """Gli ingressi dichiarati sotto results/ e data/: quelli di pad_arch_comparison.input_paths (configurazioni, manifest,
    cache, run delle celle e dei fold, frame estratti, run accettati), il record della prova ufficiale, i due run ARCHRUN."""
    pp = rel_path(base, run_cfg["preregistration"]["path"])
    cfg_pr, _, pcfg = PAC.load_config(pp, base)
    out = list(PAC.input_paths(cfg_pr, pcfg, base, pp).values())
    out.append(rel_path(base, run_cfg["trial_record"]["folder"]))
    out += [rel_path(base, e["folder"]) for e in cfg["parent_runs"].values()]
    return out


def load_all(a, base: Path, cfg_p: Path, outside_extra=()) -> tuple:
    """Configurazione, guardia delle letture, file fissati (pad_arch_run.load_pins), codice, ricetta, variabili dei thread,
    regola della macchina."""
    if why := outside_problem(base, [cfg_p.parent] + list(outside_extra)):
        fail(f"read guard: {why}")
    cfg, cfg_sha = load_config(cfg_p)
    install_guard([ROOT / p for p in cfg["read_guard"]["forbidden_paths"]] + [base / p for p in cfg["read_guard"]["forbidden_paths"]]
                  + [Path(p) for p in a.forbid_outside or ()], cfg["read_guard"]["forbidden_name_patterns"], link_exempt=link_exempt_paths())
    run_cfg_p, run_cfg_b = PR.read_pinned(base, cfg["parent_config"], "ARCHRUN configuration")
    run_cfg, run_cfg_sha = PR.load_config(run_cfg_p)
    allowed = declared_inputs(cfg, base, run_cfg)
    guard = guard_setup(cfg, base, allowed, a.forbid_outside or (), [cfg_p.parent] + list(outside_extra))
    pins = PR.load_pins(run_cfg, base)
    code = check_code(cfg)
    recipe = check_recipe(read_regular(Path(PAC.__file__), "pad_arch_comparison.py").decode("utf-8"))
    th = pins.cfg_pr["training"]["threads"]
    if (th["intra_op"], th["inter_op"]) != (cfg["threads"]["intra_op"], cfg["threads"]["inter_op"]):
        fail("threads: the pre-registration threads differ from this configuration")
    an = cfg["analysis"]
    if an["sha256"] is not None:
        h = sha256_bytes(read_regular(rel_path(ROOT, an["path"]), "frozen note"))
        if h != an["sha256"]:
            fail(f"frozen note {Path(an['path']).name}: SHA-256 {h} differs from analysis.sha256 of the configuration")
    env = PR.thread_environment_rule(pins.cfg_pr, pins.trial)
    mrule = PR.machine_rule(pins.trial)
    return cfg, cfg_sha, guard, run_cfg, run_cfg_sha, pins, code, recipe, env, mrule, allowed


def cmd_check_inputs(a) -> int:
    base, cfg_p = resolve(a)
    cfg, cfg_sha, guard, run_cfg, run_cfg_sha, pins, code, recipe, env, mrule, _ = load_all(a, base, cfg_p)
    parents = load_parents(cfg, base, run_cfg_sha)
    inp = PAC.load_inputs(pins.cfg_pr, pins.pcfg, base)
    keys = unit_keys_check(inp, parents)
    PAC.assert_no_test(inp)
    if GUARD["hits"]:
        fail(f"read guard: {len(GUARD['hits'])} refused opens")
    print(f"CHECK-INPUTS: configuration {cfg_sha[:12]}; ARCHRUN configuration {run_cfg_sha[:12]} and its pins; code pinned; recipe "
          f"{', '.join(recipe['operations'])} equal to the training source; machine and versions equal to the trial; thread variables "
          + ", ".join(f"{k} {'absent' if v is None else repr(v)}" for k, v in env.items())
          + f"; parent runs {', '.join(p['run_id'] for p in parents['runs'].values())} PASS with anchored SHA256SUMS; {keys['units']} units "
          f"({cfg['scope']['name']}) with their five files and keys; read guard on ({guard['forbidden_folders']} forbidden folders, "
          f"{guard['allowlisted_zones']} allowlisted zones); no frame assembled; TensorFlow {FR.tf_state()}; nothing written")
    print("RESULT: PASS (check-inputs)")
    return 0


def run_dir_problem(run: Path, out_root: Path, inputs: dict) -> str:
    if os.path.lexists(run):
        return f"the run folder {run.name} already exists: every attempt keeps its outcome"
    if out_root.is_symlink():
        return "the output root is a symbolic link"
    for label, p in inputs.items():
        if pg.within(p, run) or (Path(p).is_dir() and pg.within(run, p)):
            return f"{label} {Path(p).name} is inside the run folder, or contains it"
    return mw.output_problem(run, ROOT, pg.BASE_PROTECTED, [ROOT / DEFAULT_OUT_ROOT], [ROOT / x for x in EXTRA_PROTECTED], inputs,
                             [STATUS, LOG], [run / PUBLIC, run / RESTRICTED])


def previous_attempts(out_root: Path, scope: str) -> list:
    """Le cartelle dello stesso ambito nella radice d'uscita (qualunque data e commit), con il numero del tentativo."""
    out = []
    if not out_root.is_dir():
        return out
    for e in sorted(os.listdir(out_root)):
        if not LOOKS_LIKE_RUN.match(e):
            continue
        m = RUN_ID_RE.match(e)
        if not m:
            fail(f"output root: the entry {e!r} looks like a run id but does not have the form YYYYMMDD-ARCHBN2X2-<scope>-<commit>[-r<k>]")
        if m.group(1) == scope:
            out.append({"run_id": e, "attempt": int(m.group(2)) if m.group(2) else 1})
    return out


def command_text(a) -> str:
    parts = [a.cmd, "--config", PR.path_text(PR.folder_arg(a.config))]
    if a.input_root is not None:
        parts += ["--input-root", PR.path_text(Path(a.input_root).resolve())]
    parts += ["--out-root", PR.path_text(PR.folder_arg(a.out_root))]
    if a.unofficial_out_root:
        parts.append("--unofficial-out-root")
    for _ in a.forbid_outside or ():
        parts += ["--forbid-outside", "<a folder outside the repository>"]
    if a.date is not None:
        parts += ["--date", a.date]
    parts += ["--attempt", str(a.attempt)]
    if a.i_am_authorized:
        parts.append("--i-am-authorized")
    return "python -I -B scripts/pad_arch_bn_factorial.py " + " ".join(shlex.quote(x) for x in parts)


def cmd_trial(a) -> int:
    if not a.i_am_authorized:
        fail("trial loads saved checkpoints and the restricted NUAA fitting and dev frames: pass --i-am-authorized")
    base, cfg_p = resolve(a)
    out_root = PR.folder_arg(a.out_root)
    official = PR.root_kind(out_root, ROOT, ROOT / DEFAULT_OUT_ROOT, a.unofficial_out_root)
    if official and (a.input_root is not None and Path(a.input_root).resolve() != ROOT.resolve()):
        fail("the official output root needs the inputs of the repository: --input-root is refused")
    if official and not pg.within(cfg_p, ROOT):
        fail("the official output root needs a configuration inside the repository")
    views_parent = Path(tempfile.gettempdir())
    if left := stale_views(views_parent):
        fail(f"{len(left)} views folder(s) of an earlier trial left in the temporary folder ({VIEWS_PREFIX}*): remove them first")
    cfg, cfg_sha, guard, run_cfg, run_cfg_sha, pins, code, recipe, env, mrule, allowed = load_all(a, base, cfg_p, [] if official else [out_root])
    if cfg["analysis"]["sha256"] is None:
        fail("analysis.sha256 is null: NOTE.md (the choices and the analysis) must be frozen and its SHA-256 written before the trial")
    command = command_text(a)
    if probs := PAC.output_problems(command, FR.Inputs()):
        fail("the recorded command holds text that a publishable output must not hold: " + "; ".join(probs))
    state = repo_state(ROOT)
    if state["dirty"]:
        fail("the git work tree is not clean: the trial needs a committed state")
    prev = previous_attempts(out_root, cfg["scope"]["name"])
    need = max((p["attempt"] for p in prev), default=0) + 1
    if a.attempt != need:
        fail(f"--attempt {a.attempt} refused: this attempt must be --attempt {need}")
    date = a.date or time.strftime("%Y%m%d")
    if not re.fullmatch(r"\d{8}", date):
        fail("--date must be YYYYMMDD")
    rid = f"{date}-ARCHBN2X2-{cfg['scope']['name']}-{state['commit']}" + ("" if a.attempt == 1 else f"-r{a.attempt}")
    if not RUN_ID_RE.match(rid):
        fail(f"run id {rid} malformed")
    run = out_root / rid
    inputs = {"the configuration": cfg_p, **{f"the parent run {r}": rel_path(base, e["folder"]) for r, e in cfg["parent_runs"].items()}}
    if msg := run_dir_problem(run, out_root, inputs):
        fail(f"refusing the run folder: {msg}")
    run.mkdir(parents=True)
    guard = guard_setup(cfg, base, allowed + [run], a.forbid_outside or (), [cfg_p.parent] + ([] if official else [out_root]))
    head = {"schema": RUN_SCHEMA, "run_id": rid, "commit": state["commit"], "dirty": False, "config_sha256": cfg_sha, "command": "trial",
            "command_line": command, "official": official, "attempt": {"attempt": a.attempt, "previous": prev}}
    budget = SR.Budget(cfg["cap"]["wall_minutes"])
    SR.signal_handlers(budget)
    started, c0 = now(), PAC.cpu_seconds()
    scan = FR.Inputs()
    log, saved_warning = None, warnings.showwarning
    extra = {"machine_rule": mrule, "threads": {"environment": env}, "cap_minutes": cfg["cap"]["wall_minutes"], "cap_unit": cfg["cap"]["unit"],
             "read_guard": guard, "scope": {"name": cfg["scope"]["name"], "units": [unit_label(u) for u in cfg["scope"]["units"]]}}
    ctx = Ctx(cfg, budget)
    views = None
    recs = []
    try:
        PR.write_status(run, {**head, "status": "RUNNING", "passed": False, "started": started, "pid": os.getpid(), **extra,
                              "note": "RUNNING without a live process means a failed run; SIGUSR1 to this pid stops it as INCOMPLETE"}, scan)
        log = FR.open_log(run)
        warnings.showwarning = PAC.show_warning
        print(f"== 2x2 trial {rid}: configuration {cfg_sha[:12]}, scope {cfg['scope']['name']} ({len(cfg['scope']['units'])} units), cap "
              f"{cfg['cap']['wall_minutes']} min, pid {os.getpid()}", flush=True)
        parents = load_parents(cfg, base, run_cfg_sha)
        inp = PAC.load_inputs(pins.cfg_pr, pins.pcfg, base)
        scan = inp
        unit_keys_check(inp, parents)
        store = FR.FrameStore(inp, inp.X_extracted, inp.extracted_keys)
        tf = TF_MODULE() if TF_MODULE is not None else BR.tf_module()
        threads = PAC.set_threads(tf, pins.cfg_pr)
        extra["threads"] = threads
        extra["threads_rule"] = PR.threads_rule(pins.trial, threads)
        PR.require_threads(extra["threads_rule"])
        FR.hide_gpu(tf)
        ctx.tf, ctx.cfg_pr, ctx.bn_cfg = tf, pins.cfg_pr, pins.bn_cfg
        rdir = run / RESTRICTED
        rdir.mkdir()
        status, stopped = "PASS", None
        first = {}
        eps = pins.bn_cfg["batchnorm"]["epsilon"]
        texts = {"cells": [], "effects": [], "layers": [], "controls": [], "mc": []}
        try:
            fold = None
            for u in cfg["scope"]["units"]:
                if u["fold"] != fold:
                    if views is not None:
                        views.close()
                        views = None
                    fold = u["fold"]
                    ctx.step(f"frames of fold {fold}")
                    fit = PR.unit_frames(store, inp, "fitting", fold)
                    dev = PR.unit_frames(store, inp, "dev", fold)
                    fit["digests"] = fit_digests(fit["keys"])
                    ctx.unit = f"fold{fold}"
                    t0 = ctx.span_start()
                    views = Views(tf, fit["X"], fit["digests"], VIEW_SEEDS, cfg["estimator"]["batch"], cfg["views"]["storage"], views_parent,
                                  cfg["views"]["value_bounds"], cfg["views"]["free_space_margin_gb"], check=ctx.step,
                                  refuse_inside=(ROOT, base, run))
                    sp = ctx.span_end(t0, "views", "(views)")
                    progress(f"views of fold {fold} generated: {len(VIEW_SEEDS)} x {len(fit['X'])} frames in {sp['wall_seconds']:.0f} s")
                    extra.setdefault("views", {})[f"fold{fold}"] = {k: views.stats[k] for k in ("min", "max", "values", "bytes", "storage")}
                parent = parents["units"][unit_label(u)]
                flags = flags_for(cfg, u, first)
                rec = process_unit(ctx, parent, fit, dev, views, flags)
                if flags["duplicates"]:
                    first[("dup", u["network"])] = rec["label"]
                if flags["repeat"]:
                    first[("rep", u["network"])] = rec["label"]
                mw.write_all(rdir, restricted_files(rec, dev, parent), newline="")
                c_rows, e_rows, l_rows, mc_recs = rows_for(rec, dev, eps)
                texts["mc"] += mc_recs
                texts["cells"] += c_rows
                texts["effects"] += e_rows
                texts["layers"] += l_rows
                texts["controls"] += control_rows(rec)
                recs.append({"label": rec["label"], "complete": True})
                for k in list(rec["cells"]):
                    rec["cells"][k] = None
                print(f"== {rec['label']}: controls pass; cells E00, E10, E01, E11" + (" with halves" if flags["halves"] else "")
                      + (", duplicates checked" if flags["duplicates"] else "") + (", repetition identical" if flags["repeat"] else ""), flush=True)
        except SR.CapExceeded as e:
            status = "INCOMPLETE"
            stopped = {"step": e.step, "cause": e.cause, "elapsed_minutes": round(e.minutes, 1)}
            print(f"== {'CAP' if e.cause == 'cap' else 'STOP'}: {e}; the complete units are kept and reported", flush=True)
        finally:
            if views is not None:
                views.close()
                views = None
        PAC.assert_no_test(inp)
        if GUARD["hits"]:
            fail(f"read guard: {len(GUARD['hits'])} refused opens during the trial")
        if sha256_bytes(read_regular(cfg_p, "configuration")) != cfg_sha or check_code(cfg) != code:
            fail("the configuration or the code changed during the trial")
        wall, cpu_total, finished = budget.elapsed(), round(PAC.cpu_seconds() - c0, 1), now()
        res_sha = PR.write_restricted_sums(rdir)
        sealed = not cfg["scope"]["report_cell_metrics"]
        if sealed:
            mw.write_all(rdir, {"sealed_cells.csv": csv_text(CELL_COLUMNS, texts["cells"]), "sealed_effects.csv": csv_text(EFFECT_COLUMNS, texts["effects"]),
                                "sealed_montecarlo.csv": csv_text(MC_COLUMNS, montecarlo_rows(texts["mc"]))}, newline="")
            res_sha = PR.write_restricted_sums(rdir)
        units_t = [t for t in ctx.timing if t["cell"] == "(unit)"]
        doc = {**head, "status": status, "started": started, "finished": finished, "wall_seconds": round(wall, 1), "process_cpu_seconds": cpu_total,
               "cap_minutes": cfg["cap"]["wall_minutes"], "stopped": stopped, "scope": {**cfg["scope"], "units": [unit_label(u) for u in cfg["scope"]["units"]]},
               "complete_units": len(recs), "peak_rss_gb": round(ctx.peak_gb, 3), "controls": texts["controls"], "timing_units": units_t,
               "machine": (PR.MACHINE or PAC.machine)(), "environment": (PR.ENVIRONMENT or FR.environment)(), "machine_rule": mrule,
               "threads": extra["threads"], "threads_rule": extra["threads_rule"], "code_sha256": code, "recipe": recipe, "views": extra.get("views"),
               "parent_runs": parents["runs"], "inputs_sha256": {**pins.files, "configuration": cfg_sha, "archrun_configuration": run_cfg_sha},
               "test_read": TEST_READ}
        public = {"controls.csv": csv_text(CONTROL_COLUMNS, texts["controls"]),
                  "cells.csv": csv_text(CELL_COLUMNS, [] if sealed else texts["cells"]),
                  "effects.csv": csv_text(EFFECT_COLUMNS, [] if sealed else texts["effects"]),
                  "montecarlo.csv": csv_text(MC_COLUMNS, [] if sealed else montecarlo_rows(texts["mc"])),
                  "layers.csv": csv_text(LAYER_COLUMNS, texts["layers"]),
                  "timing.csv": csv_text(TIMING_COLUMNS, [[cell(t[k]) for k in TIMING_COLUMNS] for t in ctx.timing]),
                  "summary.json": PAC.json_text(strip(doc), "summary.json"), "report.md": report_md(doc)}
        pub_sha = PR.write_public(run, inp, public)
        print(f"run folder: {run.name}, wall {wall:.0f} s, process CPU {cpu_total:.0f} s, peak memory {ctx.peak_gb:.2f} GB")
        print(f"RESULT: {status} (trial)")
        warnings.showwarning = saved_warning
        FR.close_log(log)
        log = None
        if probs := PAC.output_problems((run / LOG).read_text(encoding="utf-8", errors="replace"), inp):
            fail(f"{LOG} holds text that a publishable output must not hold: " + "; ".join(probs))
        PR.write_status(run, {**head, "status": status, "passed": status == "PASS", "started": started, "finished": finished,
                              "wall_seconds": round(wall, 1), "process_cpu_seconds": cpu_total, **extra, "stopped": stopped,
                              "complete_units": [r["label"] for r in recs], "peak_rss_gb": round(ctx.peak_gb, 3),
                              "public_sha256sums_sha256": pub_sha, "restricted_sha256sums_sha256": res_sha, "run_log_sha256": sha256_file(run / LOG)}, scan)
    except BaseException as e:
        warnings.showwarning = saved_warning
        if views is not None:
            views.close()
        extra["complete_units"] = [r["label"] for r in recs]
        extra["peak_rss_gb"] = round(ctx.peak_gb, 3)
        if isinstance(e, ControlFailure):
            extra["failed_check"] = {"check": e.check, "values": strip(e.values)}
        if (run / RESTRICTED).is_dir():
            extra["restricted_sha256sums_sha256"] = PR.write_restricted_sums(run / RESTRICTED)
        PR.fail_cleanup(run, list(PUBLIC_OUTPUTS), log, head, started, budget, c0, extra, e, scan)
        raise
    return STATUS_CODE[status]


# ============================================================================ tempi sintetici (rilievo 8 della revisione del piano)

def timed(f, repeat: int = 2) -> float:
    """Tempo reale minimo di `repeat` chiamate dopo una chiamata di riscaldamento."""
    f()
    best = None
    for _ in range(repeat):
        t0 = time.monotonic()
        f()
        dt = time.monotonic() - t0
        best = dt if best is None else min(best, dt)
    return best


def measure_costs(tf, arch, shape=(256, 256, 3), batch: int = 16) -> dict:
    """Costi per batch delle parti nuove su array casuali della dimensione vera (nessun dato, pesi casuali): un batch di una
    vista (sette operazioni), le maschere di un batch per Dropout, per ogni BN un batch dello stimatore con il sotto-modello
    (dropout spento) e con la catena e le maschere (LivenessNet), il batch del passaggio di coerenza a più uscite, un batch
    di valutazione e la copia di un modello (clone_model + set_weights)."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 256, (batch,) + tuple(shape), dtype=np.uint8)
    d = [frame_digest(f"synthetic-{i}") for i in range(batch)]
    out = {"batch": batch, "view_batch": timed(lambda: augment(tf, scaled(X), d, 101))}
    cfg = {"dropout": {"networks": {"LivenessNet": {"layers": {"dropout1": 0.25, "dropout2": 0.25, "dropout_dense": 0.5},
                                                    "active": ["dropout1", "dropout2"]}, "ResNet-18": {"layers": {}, "active": []}}}}
    for net in NETWORKS:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(1)
        model = AM.build(net, shape, arch)
        st = structure(model, net, cfg)
        order = PR.expected_orders()[net]
        src = Source.clean(X, d, (101,))
        r = {"layers": {}, "copy": timed(lambda: fresh_copy(tf, model, BR.snapshot_weights(model), "measure"), 1)}
        for n in order:
            r["layers"][n] = {"sub": timed(lambda: moments_pass(tf, model, src, (n,), batch, st, False))}
            if st["chain"] is not None:
                r["layers"][n]["chain_dropout"] = timed(lambda: moments_pass(tf, model, src, (n,), batch, st, True))
        r["consistency_sub"] = timed(lambda: moments_pass(tf, model, src, order, batch, st, False))
        if st["chain"] is not None:
            r["consistency_chain"] = timed(lambda: moments_pass(tf, model, src, order, batch, st, True))
            r["masks"] = {k: timed(lambda: [mask_of(x, 101, k, s, 0.25) for x in d])
                          for k, s in (("dropout1", (shape[0] // 2, shape[1] // 2, 16)), ("dropout2", (shape[0] // 4, shape[1] // 4, 32)))}
        r["evaluate"] = timed(lambda: BR.evaluate(model, {"X": X, "y": np.arange(batch) % 2}, batch, "measure"))
        out[net] = r
        del model
        gc.collect()
    return out


def estimate_unit(c: dict, net: str, flags: dict, n_fit: int = 1400, n_dev: int = 400, views: int = 8) -> dict:
    """Secondi reali di un checkpoint dai costi per batch, passaggio per passaggio come in process_unit (costo per frame =
    costo per batch / dimensione del batch). Restituisce il totale e i secondi di ogni passo."""
    b = c["batch"]
    k = c[net]
    order = PR.expected_orders()[net]

    def sub(layers, v):
        return sum(v * n_fit * k["layers"][n]["sub"] / b for n in layers)

    def chain(layers, v):
        return sum(v * n_fit * k["layers"][n]["chain_dropout"] / b for n in layers)

    def ev(n):
        return n * k["evaluate"] / b

    def cons(v, key="consistency_sub"):
        return v * n_fit * k[key] / b
    s = {"baseline": ev(n_dev), "E00": sub(order, 1) + cons(1) + ev(n_dev)}
    copies = 1
    h = views // 2
    if net == "LivenessNet":
        drop = [n for n in order if n not in ("bn1_1", "bn1_2")]
        s["E00_chain"] = sub(order, 1) + ev(n_dev)
        s["E10"] = sub(order, views) + cons(views) + ev(n_dev) + ev(n_fit)
        s["E01"] = chain(drop, views) + cons(views, "consistency_chain") + ev(n_dev) + ev(n_fit)
        s["E11"] = s["E01"]
        copies += 4
        if flags["duplicates"]:
            s["duplicates"] = 2 * chain(("bn1_1", "bn1_2"), views)
            copies += 2
        if flags["halves"]:
            s["halves"] = 2 * (sub(order, h) + cons(h) + ev(n_dev) + ev(n_fit)) \
                + 4 * (chain(drop, h) + cons(h, "consistency_chain") + ev(n_dev) + ev(n_fit))
            copies += 6
        if flags["repeat"]:
            s["repeat"] = s["E11"]
            copies += 1
    else:
        s["E10"] = sub(order, views) + cons(views) + ev(n_dev) + ev(n_fit)
        copies += 1
        if flags["duplicates"]:
            s["duplicates"] = 2 * (sub(order, views) + ev(n_dev))
            copies += 2
        if flags["halves"]:
            s["halves"] = 2 * (sub(order, h) + cons(h) + ev(n_dev) + ev(n_fit))
            copies += 2
    s["copies"] = copies * k["copy"]
    return {"seconds": sum(s.values()), "steps": s}


def estimate_scope(c: dict, units: list, cfg_like: dict, n_fit: int = 1400, views: int = 8) -> dict:
    first, per = {}, {}
    folds = {u["fold"] for u in units}
    vs = len(folds) * views * n_fit * c["view_batch"] / c["batch"]
    total = vs
    for u in units:
        f = flags_for(cfg_like, u, first)
        if f["duplicates"]:
            first[("dup", u["network"])] = 1
        if f["repeat"]:
            first[("rep", u["network"])] = 1
        e = estimate_unit(c, u["network"], f, n_fit=n_fit, views=views)
        per[unit_label(u)] = {"seconds": round(e["seconds"], 1), "steps": {k: round(v, 1) for k, v in e["steps"].items()}}
        total += e["seconds"]
    return {"seconds": round(total, 1), "hours": round(total / 3600, 2), "views_seconds": round(vs, 1), "per_unit": per}


def cmd_measure(a) -> int:
    """Tempi sintetici: solo array e pesi casuali (nessun frame, checkpoint o punteggio letto), CPU, 10 + 10 thread; stampa
    i costi per batch e le stime degli ambiti in JSON; non scrive nulla."""
    tf = TF_MODULE() if TF_MODULE is not None else BR.tf_module()
    try:
        tf.config.threading.set_intra_op_parallelism_threads(PAC.REQUIRED_THREADS)
        tf.config.threading.set_inter_op_parallelism_threads(PAC.REQUIRED_THREADS)
    except RuntimeError:
        pass
    FR.hide_gpu(tf)
    arch = FR.default_arch_loader(rel_path(ROOT, a.arch))
    t0, c0 = time.monotonic(), PAC.cpu_seconds()
    costs = measure_costs(tf, arch)
    pilot = [{"run": "nuaa_patience_4", "network": n, "fold": 1, "seed": 42} for n in NETWORKS]
    p4 = [{"run": "nuaa_patience_4", "network": n, "fold": f, "seed": s} for f in (1, 2, 3, 4) for s in (42, 43, 44) for n in NETWORKS]
    p12 = [{**u, "run": "nuaa_patience_12"} for u in p4]

    def like(dup, mc):
        return {"reuse": {"duplicate_check": dup}, "monte_carlo": {"runs": mc}, "determinism": {"cell": "E11", "networks": ["LivenessNet"], "units": "first"}}
    doc = {"what": "synthetic timing of the new parts: random uint8 arrays 256x256x3, LivenessNet and ResNet-18 with random weights, batch 16, "
                   "CPU, 10 intra-op and 10 inter-op threads; per-batch wall seconds (minimum of two calls after a warm-up); no data read",
           "threads": {"intra_op": tf.config.threading.get_intra_op_parallelism_threads(), "inter_op": tf.config.threading.get_inter_op_parallelism_threads()},
           "costs": costs,
           "estimates": {"pilot: 2 checkpoints, every check, Monte Carlo": estimate_scope(costs, pilot, like("all", ["nuaa_patience_4"])),
                         "p4-24: duplicates on the first unit of each network, Monte Carlo": estimate_scope(costs, p4, like("first", ["nuaa_patience_4"])),
                         "p12-24: duplicates on the first unit of each network, no Monte Carlo": estimate_scope(costs, p12, like("first", []))},
           "memory": {"peak_rss_gb_of_this_measure": round(SR.peak_rss_mb() / 1024, 2),
                      "views_bytes_per_fold": 8 * 1400 * 256 * 256 * 3 * 4, "frame_store_bytes": 12614 * 256 * 256 * 3,
                      "fitting_part_bytes": 1400 * 256 * 256 * 3, "moments_batch_bytes_largest": 16 * 256 * 256 * 16 * 8,
                      "note": "the trial adds to the measured process the frame store (loaded whole), the fitting and dev parts and the views "
                              "of a fold (in memory, or as memmap pages that count as resident when read)"},
           "measure_wall_seconds": round(time.monotonic() - t0, 1), "measure_cpu_seconds": round(PAC.cpu_seconds() - c0, 1)}
    print(json.dumps(strip(doc), indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="The 2x2 BN re-estimation of PROTOCOL.md §4 on fitting/dev (augmentation and dropout during the "
                                             "re-estimation only; no test data)")
    ap.add_argument("cmd", choices=["check-inputs", "trial", "measure"])
    ap.add_argument("--arch", default="src/livedetection/src/architectures.py", help="measure: the LivenessNet architecture file")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="configuration JSON (relative to the repository root)")
    ap.add_argument("--input-root", default=None, help="base folder of the input paths of the configuration (default: repository root; "
                                                       "refused with the official output root)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="folder that receives <run-id>/ (official: results/archbn2x2)")
    ap.add_argument("--unofficial-out-root", action="store_true", help="accept an --out-root outside the repository (tests of the command)")
    ap.add_argument("--forbid-outside", action="append", default=None, help="an additional folder that must never be opened (repeatable)")
    ap.add_argument("--date", default=None, help="YYYYMMDD of the run id (default: today)")
    ap.add_argument("--attempt", type=int, default=1, help="attempt number: the last attempt in the output root + 1 (1 if none)")
    ap.add_argument("--i-am-authorized", action="store_true", help="required by trial")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return {"check-inputs": cmd_check_inputs, "trial": cmd_trial, "measure": cmd_measure}[a.cmd](a)
    except Failure as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(f"RESULT: FAIL ({a.cmd})")
        return 1
    except KeyboardInterrupt:
        print(f"RESULT: FAIL ({a.cmd}, interrupted)")
        return 130
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: unexpected error {type(e).__name__}: {e}", file=sys.stderr)
        print(f"RESULT: FAIL ({a.cmd})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
