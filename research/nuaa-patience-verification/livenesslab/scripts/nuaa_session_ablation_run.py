"""
Run di ablazione della sessione 03 su NUAA: LivenessNet addestrato con e senza la sessione 03 negli stessi soggetti del
fitting, tre bracci (B senza 03, A* con 03, B' controllo della numerosità per cella) × 4 fold × 3 seed = 36 training su
CPU. Autorizzato dal coordinatore il 5/11/2026 sulla scheda C2 dopo la proposta delivery/sessionablation/PROPOSAL.md:
«Procediamo con il confronto principale A* contro B, includendo B′ come controllo secondario. Per ora non includiamo A**
né dev-mix. Prima del run, fissi nella configurazione i tre bracci, i confronti e la regola di lettura dei risultati.
[...] 36 training, con tetto complessivo di 360 minuti. Se il tempo non basta, si fermi e segnali il run incompleto,
conservando log e risultati disponibili».

Passi del run, in un solo processo e nell'ordine (ognuno registrato con il tempo impiegato):
  1. check-inputs   configurazione a insieme chiuso; i cinque manifest M2 e i dodici manifest dello schema dei fold
                    (riusati da nuaa_folds_run: SHA-256, conteggi, ruoli disgiunti, test = ogni frame dei soggetti tenuti
                    fuori); la configurazione dello schema approvato; la cache nuaa_1500.npz (SHA-256, schema .npy,
                    posizione di ogni frame nell'ordine di campionamento M2); il run delle celle 20261104-SESSABL-5d89888
                    (status PASS, SHA256SUMS pubblico e riservato ancorati, configurazione con le allocazioni) e i suoi
                    16 manifest usati (armB/armAstar/armBprime_fitting_fold<k>.csv, dev_fold<k>.csv: SHA-256, ruoli,
                    conteggi per cella, annidamento, soggetti disgiunti dal dev e dal test, devmix mai letto); il run
                    dei fold 20261103-FOLDSRUN-e174812 (status PASS, SHA256SUMS ancorati, extracted_frames.csv e
                    aggregate.json del riferimento A); il file dei frame estratti (SHA-256, membri, chiavi, hash di ogni
                    frame uguale a extracted_frames.csv); ogni riga di ogni manifest trovata esattamente una volta (cache
                    o file estratto); i 12.614 originali (dimensione e SHA-256); architettura congelata e codice per
                    SHA-256; ambiente (OpenCV e TensorFlow registrati senza importare TensorFlow). Nulla scritto.
  2. train          per ogni modello, nell'ordine fissato in configurazione (per fold 1..4 e seed 42, 43, 44: B poi A*;
                    poi B' per fold e seed): fitting del braccio e dev comune dai manifest, ricetta del run dei fold
                    invariata su CPU (nuaa_folds_run.default_trainer); pesi .keras riservati con SHA-256.
  3. threshold      punteggi del dev comune con il checkpoint, val_loss ricalcolato dai logit dello strato di uscita (come
                    Keras: softmax cross-entropy dai logit, senza clip) contro la scheda, soglia EER (criterio M1)
                    scritta, riletta e ancorata per SHA-256 PRIMA di qualunque punteggio del test.
  4. score          ogni frame del test del fold (soggetti tenuti fuori), decisione a 0,5 e alla soglia dev congelata.
  5. aggregate      per braccio, fold, seed e strato (01, 02, 03, all): metriche; pooled sui quattro fold per braccio e
                    seed con bootstrap sui soggetti; confronti appaiati A*−B (primario), B'−B e A*−B' (secondari) per
                    fold, seed e strato e pooled con bootstrap appaiato (stessi indici per i due bracci); verdetti della
                    regola di lettura pre-dichiarata; riferimento A (indicativo) copiato dal run dei fold.
Tetto di tempo (budget.wall_time_cap_minutes, 360) controllato prima di ogni training, a fine di ogni epoca, prima della
soglia e prima dello scoring del test. DIVERSO dal run dei fold: al superamento il modello in corso è interrotto e i suoi
file tolti (né salvato né valutato), nessun altro training parte, e il run prosegue con l'aggregato dei modelli completi
e con le scritture finali (pochi secondi, dichiarate): stato INCOMPLETE con l'elenco dei training completi e mancanti.
L'arresto volontario (SIGUSR1) è trattato come il tetto: la richiesta è registrata e il run si ferma al controllo
successivo (al più a fine epoca), con lo stesso esito INCOMPLETE.
Esiti: PASS (36/36, codice 0), INCOMPLETE (tetto o SIGUSR1, codice 2), FAIL (errore, SIGTERM o SIGHUP, codice 1;
Ctrl-C, codice 130). Su FAIL il pubblico e per_subject.csv sono tolti; se ci sono modelli completi i loro cinque file
riservati restano, con un SHA256SUMS riservato solo su quelli, senza aggregato né verdetto (status.json:
kept_complete_models); senza modelli completi tutte le uscite sono tolte, come nel run dei fold.

Uscite (cartella del run <out-root>/YYYYMMDD-SESSABLRUN-<commit>[-dev], -dev se l'albero git non è pulito; --out-root
diverso dal predefinito results/sessionablationrun serve solo ai test e alla prova a secco):
  status.json, run.log
  public/      aggregate.json, aggregate.md, run_record.json, SHA256SUMS   (nessun identificativo, nessuna chiave)
  restricted/  <braccio>_fold<k>_seed<s>_card.json, _threshold.json, _dev_scores.csv, _test_scores.csv,
               _LivenessNet.keras; per_subject.csv (colonna arm), public_SHA256SUMS.sha256, SHA256SUMS
Dentro il repository la cartella del run è ammessa solo sotto results/sessionablationrun/.

  .venv/bin/python -I scripts/nuaa_session_ablation_run.py check-inputs                  # solo verifica, nulla scritto
  .venv/bin/python -I scripts/nuaa_session_ablation_run.py run --i-am-authorized         # tutti i passi

Configurazione emendata (verifica concordata dal coordinatore il 7/11/2026 sulla scheda C2: early_stopping_patience da 4
a 12, massimo 12 epoche, stessi 36 modelli, ogni altra impostazione invariata): delivery/patienceverify/
nuaa_session_ablation_run_patience12.json, con la chiave in più amendment. check-inputs e run verificano che il suo
SHA-256 sia PATIENCE12_CONFIG_SHA256, che amendment.parent_config_sha256 sia CONFIG_SHA256 e che la configurazione
genitore (letta dal repository, SHA-256 verificato) sia uguale in ogni campo tranne training.early_stopping_patience e
amendment (config_difference); status.json, run_record.json e aggregate.json registrano configuration_name
(session_ablation_run o patience12_verification; null per una configurazione sconosciuta) e, se emendata, amendment.
Stesso run id YYYYMMDD-SESSABLRUN-<commit> e stessa cartella results/sessionablationrun.

  .venv/bin/python -I scripts/nuaa_session_ablation_run.py check-inputs --config delivery/patienceverify/nuaa_session_ablation_run_patience12.json
  .venv/bin/python -I scripts/nuaa_session_ablation_run.py run --i-am-authorized --config delivery/patienceverify/nuaa_session_ablation_run_patience12.json

Riuso: le funzioni invariate di scripts/nuaa_folds_run.py (importato come modulo, senza effetti collaterali): manifest
M2 e dello schema, cache, originali, ambiente, FrameStore/assemble, trainer (ricetta), scorer, val_loss, soglia EER,
metriche, AUROC, righe per soggetto, controllo del pubblico, log. Nuovi qui: manifest delle celle, file dei frame
estratti, ordine dei 36 training, tetto con esito INCOMPLETE, bootstrap per statistiche sufficienti dei soggetti
(identico al ricalcolo sui frame ricampionati, verificato dalla suite), confronti appaiati e regola di lettura.
Dipendenze: NumPy; OpenCV (solo la versione, registrata); TensorFlow 2.15 (training e scoring, solo CPU).
"""
import argparse
import csv
import gc
import io
import json
import os
import re
import resource
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import m2_writers as mw  # noqa: E402
import nuaa_folds_run as FR  # noqa: E402
import path_guard as pg  # noqa: E402

Failure, fail = FR.Failure, FR.fail
sha256_bytes, sha256_file, read_regular, regular_path = FR.sha256_bytes, FR.sha256_file, FR.read_regular, FR.regular_path
rel_path, parse_csv, csv_text, read_sums, now, f4, fmt = FR.rel_path, FR.parse_csv, FR.csv_text, FR.read_sums, FR.now, FR.f4, FR.fmt

DEFAULT_CONFIG = "delivery/sessionablationrun/nuaa_session_ablation_run.json"
DEFAULT_OUT_ROOT = "results/sessionablationrun"
CONFIG_SCHEMA = "nuaa-session-ablation-run-config/1"
RUN_ID_RE = re.compile(r"^\d{8}-SESSABLRUN-[0-9a-f]{7,40}(-dev)?$")
TOP_KEYS = ("schema", "task", "authorisation", "cells_run", "schema_run", "folds_run", "manifests", "images", "schema_expected",
            "cache", "arms", "dev", "nesting", "excluded", "test", "training_order", "environment", "architecture", "detector",
            "train_seeds", "training", "threshold", "scoring", "metrics", "comparisons", "bootstrap", "reading_rule",
            "reference_a", "budget", "declarations", "previous_attempt", "outputs")
CONFIG_SHA256 = "4c2d29d13dede8a48c3a8e74e1ca48379c5c7e7721eb0a1f55fececbc684e23b"   # SHA-256 della configurazione consegnata, fissato prima del run
SCHEMA_CONFIG_SHA256 = "d1d43c148f762965c5ae367a0de1c59770324f7840034fd5a6c7fad0a14e73e8"   # delivery/folds/nuaa_subject_folds.json
CELLS_CONFIG_SHA256 = "2b86ef8e8cab3e9fac4ae9ca34471977c773bba34cc4990347f03b0ab38db015"    # delivery/sessionablation/nuaa_session_ablation.json
# configurazione emendata della verifica concordata il 7/11/2026 (sola modifica: early_stopping_patience 4 -> 12),
# SHA-256 fissato prima del run; la configurazione genitore è quella consegnata (CONFIG_SHA256), letta dal repository
PATIENCE12_CONFIG = "delivery/patienceverify/nuaa_session_ablation_run_patience12.json"
PATIENCE12_CONFIG_SHA256 = "ca039df5d3114f9cb983c7256330aae671de54ae34d949ad4e9b9c80beb6aa43"
PARENT_CONFIG = DEFAULT_CONFIG   # relativo a ROOT (un percorso assoluto solo nei test)
AMENDMENT_KEY = "amendment"
AMENDMENT_FIELD = "training.early_stopping_patience"
AMENDMENT_FROM, AMENDMENT_TO = 4, 12
AMENDMENT_ALLOWED = (AMENDMENT_FIELD, AMENDMENT_KEY)      # i soli percorsi in cui la configurazione emendata può differire
AMENDMENT_KEYS = ("parent_config_sha256", "changed_field", "from", "to", "authorisation", "agreed_precisions", "note",
                  "time_estimate", "shared_history_check")
CONFIGURATION_NAMES = (("CONFIG_SHA256", "session_ablation_run"), ("PATIENCE12_CONFIG_SHA256", "patience12_verification"))
TRAIN_SEEDS = [42, 43, 44]       # fissati dal coordinatore
CAP_MINUTES = 360                # tetto complessivo fissato dal coordinatore (5 novembre 2026)
SESSIONS, LABELS, STRATA = FR.SESSIONS, FR.LABELS, FR.STRATA
CLASS_NAME = FR.CLASS_NAME
CELLS = [f"{s}/{CLASS_NAME[lab]}" for s in SESSIONS for lab in LABELS]      # 01/bona_fide, 01/attack, ..., 03/attack
CELLS_12 = CELLS[:4]
ARMS = ("arm_b", "arm_a_star", "arm_b_prime")
ARM_LABEL = {"arm_b": "B", "arm_a_star": "A*", "arm_b_prime": "B'"}
ARM_FILE = {"arm_b": "armB_fitting_fold{k}.csv", "arm_a_star": "armAstar_fitting_fold{k}.csv", "arm_b_prime": "armBprime_fitting_fold{k}.csv"}
ARM_ROLE = {"arm_b": "arm_b_fitting", "arm_a_star": "arm_a_star_fitting", "arm_b_prime": "arm_b_prime_fitting_optional"}
ARM_CELLS = {"arm_b": CELLS_12, "arm_a_star": CELLS, "arm_b_prime": CELLS_12}
CELLS_DESIGN_KEY = {"arm_b": "arm_b", "arm_a_star": "arm_a_star", "arm_b_prime": "arm_b_prime"}
DEV_FILE, DEV_ROLE = "dev_fold{k}.csv", "dev_common"
EXCLUDED = ("arm_a_star_star", "dev_mix")          # esclusi dal coordinatore: devmix_fold<k>.csv non è mai letto
FOLD_COLUMNS = FR.FOLD_COLUMNS
# confronti appaiati: (nome, minuendo, sottraendo, ruolo)
COMPARISONS = (("a_star_minus_b", "arm_a_star", "arm_b", "primary"), ("b_prime_minus_b", "arm_b_prime", "arm_b", "secondary"),
               ("a_star_minus_b_prime", "arm_a_star", "arm_b_prime", "secondary"))
PRIMARY = {"comparison": "a_star_minus_b", "stratum": "03", "metrics": ["auroc", "acer_dev"]}
METRIC_KEYS = FR.METRIC_KEYS
METRIC_LABEL = {"apcer_05": "APCER (0.5)", "bpcer_05": "BPCER (0.5)", "acer_05": "ACER (0.5)", "apcer_dev": "APCER (dev threshold)",
                "bpcer_dev": "BPCER (dev threshold)", "acer_dev": "ACER (dev threshold)", "auroc": "AUROC"}
VERDICTS = ("consistent", "inconclusive")
READING_RULE = ("for each comparison, stratum and metric, the pooled difference (minuend - subtrahend) is 'consistent' only if it "
                "has the same non-zero sign for all three seeds AND the paired bootstrap interval of every seed excludes 0 on "
                "the side of that sign; otherwise 'inconclusive' (also when a seed has no pooled value, no difference or no "
                "interval); a consistent verdict states the direction (for example 'consistent: A* higher AUROC than B'); "
                "primary: AUROC and ACER at the dev threshold in stratum 03 of a_star_minus_b; the same rule, labelled "
                "secondary, for the other strata, the other metrics and the two secondary comparisons; no statistical test; "
                "applied mechanically by the script; every verdict reported")
REFERENCE_LABEL = "reference, different fitting frames and dev: indicative"
BOOT = {"resamples": 1000, "seed": 0, "percentiles": [2.5, 97.5], "min_subjects": 5}
BOOT_RECIPE = ("generator numpy.random.default_rng(0) (PCG64), restarted for every seed, comparison (or arm) and stratum; the "
               "held-out subjects of the stratum (pooled over the four folds) in ascending identifier order; at every resample "
               "integers(0, n, n) with n = the number of subjects; every metric computed on the pooled frames of the resampled "
               "subjects (through per-subject counts and pairwise Mann-Whitney counts, identical to recomputing it on the "
               "frames); the same subject indices for both arms of a comparison and for every metric; percentiles 2.5 and 97.5 "
               "with np.percentile, linear interpolation; a resample without a class does not count for the metrics that need "
               "it; no interval with fewer than 5 subjects")
CODE_FILES = ("scripts/nuaa_session_ablation_run.py", "scripts/nuaa_folds_run.py", "scripts/m2_writers.py", "scripts/path_guard.py",
              "scripts/package_rules.py")
# codice condiviso con il run dei fold (il trainer, lo scorer e i writer importati): SHA-256 uguale a code_sha256 del suo
# run_record.json pubblico; nuaa_folds_run.py è obbligatorio, gli altri sono confrontati se il run dei fold li registra
SHARED_CODE = ("scripts/nuaa_folds_run.py", "scripts/m2_writers.py", "scripts/path_guard.py", "scripts/package_rules.py")
FOLDS_RUN_ARCH_KEY = "reference_code.architectures"
DECLARATION_FLAGS = ("pre_commitment", "dev_reuse", "cpu_only", "all_outcomes_kept")
DECLARATION_KEYS = DECLARATION_FLAGS + ("cpu_nondeterminism", "nominal_pairing", "diagnostic", "limits", "question")
PREVIOUS_ATTEMPT_KEYS = ("run_id", "commit", "started", "outcome", "cause", "complete_models", "results_read", "relaunch")
VAL_LOSS_CHECK = "logits (as Keras: softmax cross-entropy from logits, no clipping)"
PROB_TOL = 1e-4                  # |softmax dei logit - probabilità dello scorer| ammesso (float32 contro float64)
STATUS, LOG = FR.STATUS, FR.LOG
PUBLIC, RESTRICTED = "public", "restricted"
PUBLIC_OUTPUTS = ("aggregate.json", "aggregate.md", "run_record.json", "SHA256SUMS")
PER_MODEL = FR.PER_MODEL
RESTRICTED_FIXED = ("per_subject.csv", "public_SHA256SUMS.sha256", "SHA256SUMS")
SCORE_COLUMNS = ("arm",) + FR.SCORE_COLUMNS
SUBJECT_COLUMNS = ("arm",) + FR.SUBJECT_COLUMNS
FAIL_DOC = FR.FAIL_DOC
EXTRA_PROTECTED = FR.EXTRA_PROTECTED + ("data", "results/foldsrun", "results/foldscompare", "results/sessionablation")

# punti di iniezione (solo i test): stato git, trainer, scorer, orologio, architettura
REPO_STATE = None       # (root) -> {"commit", "dirty"}
TRAINER = None          # (data, training, weights_path, train_seed, arch_module, cap_check) -> scheda (default: FR.default_trainer)
MAKE_SCORER = None      # (weights_path, sha256) -> funzione(X uint8) -> probabilità N x 2 (default: FR.tf_scorer)
MAKE_LOGITS = None      # (weights_path, sha256) -> funzione(X uint8) -> logit N x 2 dello strato di uscita (default: tf_logits)
CLOCK = None            # () -> secondi (default time.monotonic)
ARCH_LOADER = None      # (path) -> modulo con create_model (default: FR.default_arch_loader)


class CapExceeded(Exception):
    """Tetto di tempo superato, o arresto volontario richiesto (SIGUSR1), a un confine di passo o a fine epoca: non è un
    errore del run (esito INCOMPLETE). cause = "cap" oppure il nome del segnale."""

    def __init__(self, step: str, minutes: float, cause: str = "cap"):
        super().__init__(f"wall-time cap exceeded at {step} ({minutes:.1f} min)" if cause == "cap" else
                         f"stop requested by {cause} at {step} ({minutes:.1f} min)")
        self.step, self.minutes, self.cause = step, minutes, cause


def clock() -> float:
    return (CLOCK or time.monotonic)()


def peak_rss_mb() -> float:
    """Memoria residente di picco del processo (ru_maxrss: byte su macOS, KiB su Linux), in MB."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 20 if sys.platform == "darwin" else r / 1024


def model_name(arm: str, fold: int, seed: int) -> str:
    return f"{arm}_fold{fold}_seed{seed}"


def training_order(k: int, seeds) -> list:
    """L'ordine fissato dei training: per fold e seed B poi A* (le coppie del confronto primario complete insieme), poi B'."""
    first = [(arm, fold, seed) for fold in range(1, k + 1) for seed in seeds for arm in ("arm_b", "arm_a_star")]
    return first + [("arm_b_prime", fold, seed) for fold in range(1, k + 1) for seed in seeds]


# ============================================================================ configurazione

def _text(v, what: str) -> None:
    if not isinstance(v, str) or not v.strip():
        fail(f"configuration: {what} must be a non-empty text")


def _hex(v) -> bool:
    return isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) is not None


def _keys(d, keys, what: str) -> None:
    if not isinstance(d, dict) or sorted(d) != sorted(keys):
        fail(f"configuration: {what} needs exactly {', '.join(keys)}")


def _cells(spec, cells, what: str) -> None:
    if not isinstance(spec, dict) or list(spec) != list(cells) or not all(FR.is_count(v) and v > 0 for v in spec.values()):
        fail(f"configuration: {what} must give a positive count for exactly the cells {', '.join(cells)}, in this order")


def load_config(p: Path) -> tuple:
    """(configurazione, SHA-256 dei byte letti). Insiemi chiusi e valori fissati controllati qui, prima di tutto."""
    data = read_regular(p, "configuration")
    try:
        cfg = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        fail(f"configuration {p.name} is not valid JSON: {e}")
    if not isinstance(cfg, dict) or cfg.get("schema") != CONFIG_SCHEMA:
        fail(f"configuration {p.name}: schema must be {CONFIG_SCHEMA!r}")
    amended = AMENDMENT_KEY in cfg             # configurazione emendata: una sola chiave in più, controllata sotto
    if sorted(k for k in cfg if k != AMENDMENT_KEY) != sorted(TOP_KEYS):
        fail(f"configuration: fields must be exactly {', '.join(TOP_KEYS)} (missing {sorted(set(TOP_KEYS) - set(cfg))}, "
             f"unknown {sorted(set(cfg) - set(TOP_KEYS) - {AMENDMENT_KEY})})")
    for k in ("task", "authorisation", "detector"):
        _text(cfg[k], k)
    cr = cfg["cells_run"]
    _keys(cr, ["config_path", "config_sha256", "folder", "public_sha256sums_sha256", "restricted_sha256sums_sha256", "run_id"], "cells_run")
    for k in ("run_id", "folder", "config_path"):
        _text(cr[k], f"cells_run.{k}")
    if not all(_hex(cr[k]) for k in ("config_sha256", "public_sha256sums_sha256", "restricted_sha256sums_sha256")):
        fail("configuration: cells_run SHA-256 values malformed")
    sr = cfg["schema_run"]
    _keys(sr, ["config_path", "config_sha256", "folder", "restricted_sha256sums_sha256", "run_id"], "schema_run")
    for k in ("run_id", "folder", "config_path"):
        _text(sr[k], f"schema_run.{k}")
    if not _hex(sr["config_sha256"]) or not _hex(sr["restricted_sha256sums_sha256"]):
        fail("configuration: schema_run SHA-256 values malformed")
    fr_ = cfg["folds_run"]
    _keys(fr_, ["config_sha256", "folder", "frames_file", "public_sha256sums_sha256", "restricted_sha256sums_sha256", "run_id"], "folds_run")
    for k in ("run_id", "folder"):
        _text(fr_[k], f"folds_run.{k}")
    if not all(_hex(fr_[k]) for k in ("config_sha256", "public_sha256sums_sha256", "restricted_sha256sums_sha256")):
        fail("configuration: folds_run SHA-256 values malformed")
    ff = fr_["frames_file"]
    _keys(ff, ["n", "path", "sha256"], "folds_run.frames_file")
    _text(ff["path"], "folds_run.frames_file.path")
    if not _hex(ff["sha256"]) or not FR.is_count(ff["n"]):
        fail("configuration: folds_run.frames_file needs a SHA-256 and a count")
    man = cfg["manifests"]
    if not isinstance(man, dict) or sorted(man) != ["files", "folder", "origin"] or not isinstance(man["files"], dict) \
            or sorted(man["files"]) != sorted(FR.MANIFESTS):
        fail(f"configuration: manifests needs folder, origin and the files {', '.join(FR.MANIFESTS)}")
    _text(man["folder"], "manifests.folder")
    for k, v in man["files"].items():
        FR._check_manifest_spec(v, k)
    img = cfg["images"]
    _keys(img, ["folder", "layout"], "images")
    _text(img["folder"], "images.folder")
    if img["layout"] not in FR.LAYOUTS:
        fail(f"configuration: images.layout must be one of {sorted(FR.LAYOUTS)}")
    exp = cfg["schema_expected"]
    exp_keys = ["dev_per_fold", "fitting_per_fold", "k", "n_cache_frames", "n_images", "n_subjects", "n_to_extract", "test_per_fold"]
    _keys(exp, exp_keys, "schema_expected")
    for k in exp_keys:
        if k != "test_per_fold" and not FR.is_count(exp[k]):
            fail(f"configuration: schema_expected.{k} must be a non-negative integer")
    if exp["k"] < 2 or not isinstance(exp["test_per_fold"], list) or len(exp["test_per_fold"]) != exp["k"] \
            or not all(FR.is_count(v) for v in exp["test_per_fold"]):
        fail("configuration: schema_expected.test_per_fold must list one count per fold (k >= 2)")
    if sum(exp["test_per_fold"]) != exp["n_images"] or exp["n_images"] != exp["n_cache_frames"] + exp["n_to_extract"]:
        fail("configuration: schema_expected counts are inconsistent (test folds partition the images; images = cache + extracted)")
    if ff["n"] != exp["n_to_extract"]:
        fail("configuration: folds_run.frames_file.n must equal schema_expected.n_to_extract (the frames outside the cache)")
    c = cfg["cache"]
    _keys(c, ["members", "order", "path", "per_class", "sha256"], "cache")
    _text(c["path"], "cache.path")
    _text(c["order"], "cache.order")
    if not _hex(c["sha256"]) or not FR.is_count(c["per_class"]) or c["per_class"] < 1:
        fail("configuration: cache.sha256 or cache.per_class malformed")
    if not isinstance(c["members"], dict) or sorted(c["members"]) != sorted(FR.MEMBERS):
        fail(f"configuration: cache.members must define exactly {', '.join(FR.MEMBERS)}")
    for k, v in c["members"].items():
        if not isinstance(v, dict) or sorted(v) != ["dtype", "shape"] or not isinstance(v["dtype"], str) \
                or not isinstance(v["shape"], list) or not all(FR.is_count(x) for x in v["shape"]):
            fail(f"configuration: cache.members.{k} needs dtype and shape")
    if 4 * c["per_class"] != exp["n_cache_frames"]:
        fail("configuration: cache.per_class (two classes, two splits) does not give schema_expected.n_cache_frames")
    arms = cfg["arms"]
    if not isinstance(arms, dict) or list(arms) != list(ARMS):
        fail(f"configuration: arms must define exactly {', '.join(ARMS)}, in this order")
    for arm in ARMS:
        a = arms[arm]
        _keys(a, ["cells", "description", "file", "label", "role", "total"], f"arms.{arm}")
        if a["file"] != ARM_FILE[arm] or a["role"] != ARM_ROLE[arm] or a["label"] != ARM_LABEL[arm]:
            fail(f"configuration: arms.{arm} must name the file {ARM_FILE[arm]}, the role {ARM_ROLE[arm]} and the label {ARM_LABEL[arm]}")
        _cells(a["cells"], ARM_CELLS[arm], f"arms.{arm}.cells")
        if a["total"] != sum(a["cells"].values()):
            fail(f"configuration: arms.{arm}.total must be the sum of its cells")
        _text(a["description"], f"arms.{arm}.description")
    if arms["arm_a_star"]["total"] != arms["arm_b"]["total"]:
        fail("configuration: arm A* and arm B must have the same total (fixed sample size)")
    if [arms["arm_b_prime"]["cells"][x] for x in CELLS_12] != [arms["arm_a_star"]["cells"][x] for x in CELLS_12]:
        fail("configuration: the cells of arm B' must equal the session 01-02 cells of arm A*")
    if any(arms["arm_a_star"]["cells"][x] > arms["arm_b"]["cells"][x] for x in CELLS_12):
        fail("configuration: a session 01-02 cell of arm A* exceeds the cell of arm B (nesting impossible)")
    dv = cfg["dev"]
    _keys(dv, ["cells", "file", "role", "rule", "total"], "dev")
    if dv["file"] != DEV_FILE or dv["role"] != DEV_ROLE:
        fail(f"configuration: dev must name the file {DEV_FILE} and the role {DEV_ROLE} (the common dev, never dev-mix)")
    _cells(dv["cells"], CELLS_12, "dev.cells")
    if dv["total"] != sum(dv["cells"].values()):
        fail("configuration: dev.total must be the sum of its cells")
    _text(dv["rule"], "dev.rule")
    _keys(cfg["nesting"], ["rule"], "nesting")
    _text(cfg["nesting"]["rule"], "nesting.rule")
    _keys(cfg["excluded"], list(EXCLUDED), "excluded")
    for k in EXCLUDED:
        _text(cfg["excluded"][k], f"excluded.{k}")
    te = cfg["test"]
    _keys(te, ["per_fold", "source"], "test")
    _text(te["source"], "test.source")
    if te["per_fold"] != exp["test_per_fold"]:
        fail("configuration: test.per_fold must equal schema_expected.test_per_fold (the test of the schema, unchanged)")
    seeds = cfg["train_seeds"]
    if seeds != TRAIN_SEEDS:
        fail(f"configuration: train_seeds must be {TRAIN_SEEDS} (fixed by the coordinator)")
    order = [model_name(*x) for x in training_order(exp["k"], seeds)]
    if cfg["training_order"] != order:
        fail("configuration: training_order must list the 36 models in the fixed order (per fold and seed: B then A*; then B')")
    env = cfg["environment"]
    _keys(env, ["opencv", "tensorflow"], "environment")
    _text(env["opencv"], "environment.opencv")
    _text(env["tensorflow"], "environment.tensorflow")
    ar = cfg["architecture"]
    _keys(ar, ["chosen", "name", "path", "sha256"], "architecture")
    if ar["name"] != "LivenessNet" or not _hex(ar["sha256"]):
        fail("configuration: architecture needs name LivenessNet and a SHA-256")
    _text(ar["path"], "architecture.path")
    _text(ar["chosen"], "architecture.chosen")
    if not amended and cfg["training"] != FR.FROZEN_TRAINING:
        fail("configuration: training must be the recipe of the folds run, unchanged (FROZEN_TRAINING of nuaa_folds_run)")
    if amended and cfg["training"] != {**FR.FROZEN_TRAINING, "early_stopping_patience": AMENDMENT_TO}:
        fail(f"configuration: training of an amended configuration must be the recipe of the folds run with only "
             f"early_stopping_patience = {AMENDMENT_TO}")
    th = cfg["threshold"]
    _keys(th, ["also_reported", "criterion", "dev_reuse_declared", "frozen_before_test", "set", "val_loss_check"], "threshold")
    if th["criterion"] != FR.THRESHOLD_CRITERION:
        fail("configuration: threshold.criterion must be the EER criterion of the M1 protocol (as in the folds run)")
    for k in ("set", "also_reported", "frozen_before_test", "dev_reuse_declared", "val_loss_check"):
        _text(th[k], f"threshold.{k}")
    if "logits" not in th["val_loss_check"]:
        fail("configuration: threshold.val_loss_check must declare the check on the logits of the output layer")
    if "common dev" not in th["set"]:
        fail("configuration: threshold.set must declare the common dev of the fold")
    sc = cfg["scoring"]
    _keys(sc, ["batch", "batchnorm", "device", "input", "score"], "scoring")
    if sc["device"] != "CPU" or sc["batch"] != FR.SCORE_BATCH:
        fail(f"configuration: scoring must be on the CPU with batch {FR.SCORE_BATCH}")
    me = cfg["metrics"]
    me_keys = ["aggregation", "auroc", "decision", "keys", "per_subject", "per_unit", "pre_commitment", "strata"]
    _keys(me, me_keys, "metrics")
    if me["strata"] != list(STRATA) or me["decision"] != FR.DECISION_RULE or me["keys"] != list(METRIC_KEYS):
        fail(f"configuration: metrics needs strata {list(STRATA)}, keys {list(METRIC_KEYS)} and the decision rule {FR.DECISION_RULE!r}")
    for k in ("aggregation", "auroc", "per_subject", "per_unit", "pre_commitment"):
        _text(me[k], f"metrics.{k}")
    co = cfg["comparisons"]
    _keys(co, ["list", "pairing", "pooled"], "comparisons")
    want = [{"name": n, "minuend": m, "subtrahend": s, "role": r} for n, m, s, r in COMPARISONS]
    if co["list"] != want:
        fail(f"configuration: comparisons.list must be {json.dumps(want)}")
    _text(co["pairing"], "comparisons.pairing")
    _text(co["pooled"], "comparisons.pooled")
    if cfg["bootstrap"] != {**BOOT, "recipe": BOOT_RECIPE}:
        fail("configuration: bootstrap must be the recipe of the script (1,000 resamples, seed 0, percentiles 2.5-97.5, at least 5 subjects)")
    rr = cfg["reading_rule"]
    _keys(rr, ["primary_comparison", "primary_metrics", "primary_stratum", "rule", "statistical_test", "verdicts"], "reading_rule")
    if (rr["primary_comparison"], rr["primary_stratum"], rr["primary_metrics"]) != (PRIMARY["comparison"], PRIMARY["stratum"], PRIMARY["metrics"]):
        fail(f"configuration: the primary reading must be {PRIMARY['comparison']}, stratum {PRIMARY['stratum']}, metrics {PRIMARY['metrics']}")
    if rr["rule"] != READING_RULE or rr["verdicts"] != list(VERDICTS) or rr["statistical_test"] != "none":
        fail("configuration: reading_rule must be the rule of the script, with the verdicts consistent/inconclusive and no statistical test")
    ra = cfg["reference_a"]
    _keys(ra, ["label", "source", "use"], "reference_a")
    if ra["label"] != REFERENCE_LABEL:
        fail(f"configuration: reference_a.label must be {REFERENCE_LABEL!r}")
    _text(ra["source"], "reference_a.source")
    _text(ra["use"], "reference_a.use")
    bu = cfg["budget"]
    _keys(bu, ["estimate", "final_writes_after_cap", "on_cap_exceeded", "on_failure", "trainings", "voluntary_stop", "wall_time_cap_minutes"],
          "budget")
    if bu["trainings"] != len(ARMS) * exp["k"] * len(seeds):
        fail("configuration: budget.trainings must be 3 arms x k folds x the number of seeds")
    if bu["wall_time_cap_minutes"] != CAP_MINUTES:
        fail(f"configuration: budget.wall_time_cap_minutes must be {CAP_MINUTES} (fixed by the coordinator)")
    for k in ("estimate", "on_cap_exceeded", "final_writes_after_cap", "on_failure", "voluntary_stop"):
        _text(bu[k], f"budget.{k}")
    if "INCOMPLETE" not in bu["on_cap_exceeded"]:
        fail("configuration: budget.on_cap_exceeded must declare the INCOMPLETE outcome")
    if "SIGUSR1" not in bu["voluntary_stop"] or "INCOMPLETE" not in bu["voluntary_stop"]:
        fail("configuration: budget.voluntary_stop must declare SIGUSR1 and the INCOMPLETE outcome")
    if "FAIL" not in bu["on_failure"] or "kept" not in bu["on_failure"]:
        fail("configuration: budget.on_failure must declare the FAIL outcome and the restricted outputs kept")
    dec = cfg["declarations"]
    if not isinstance(dec, dict) or sorted(dec) != sorted(DECLARATION_KEYS):
        fail(f"configuration: declarations must have exactly the keys {', '.join(DECLARATION_KEYS)}")
    for k in DECLARATION_FLAGS:
        if dec[k] is not True:
            fail(f"configuration: declarations.{k} must be true")
    for k in DECLARATION_KEYS:
        if k not in DECLARATION_FLAGS:
            _text(dec[k], f"declarations.{k}")
    pa = cfg["previous_attempt"]
    _keys(pa, list(PREVIOUS_ATTEMPT_KEYS), "previous_attempt")
    for k in PREVIOUS_ATTEMPT_KEYS:
        _text(pa[k], f"previous_attempt.{k}")
    _keys(cfg["outputs"], ["public", "restricted"], "outputs")
    for k in ("public", "restricted"):
        _text(cfg["outputs"][k], f"outputs.{k}")
    if amended:
        am = cfg[AMENDMENT_KEY]
        _keys(am, list(AMENDMENT_KEYS), AMENDMENT_KEY)
        if (am["changed_field"], am["from"], am["to"]) != (AMENDMENT_FIELD, AMENDMENT_FROM, AMENDMENT_TO) \
                or type(am["from"]) is not int or type(am["to"]) is not int:
            fail(f"configuration: amendment must declare changed_field {AMENDMENT_FIELD}, from {AMENDMENT_FROM}, to {AMENDMENT_TO}")
        if not _hex(am["parent_config_sha256"]):
            fail("configuration: amendment.parent_config_sha256 malformed")
        for k in ("authorisation", "note", "time_estimate", "shared_history_check"):
            _text(am[k], f"amendment.{k}")
        if not isinstance(am["agreed_precisions"], list) or len(am["agreed_precisions"]) != 2:
            fail("configuration: amendment.agreed_precisions must list the two agreed precisions")
        for i, v in enumerate(am["agreed_precisions"]):
            _text(v, f"amendment.agreed_precisions[{i}]")
    return cfg, sha256_bytes(data)


def configuration_name(cfg_sha: str):
    """Nome della configurazione nota con questo SHA-256 (consegnata o emendata), None per una configurazione sconosciuta."""
    return next((name for const, name in CONFIGURATION_NAMES if cfg_sha == globals()[const]), None)


def config_difference(parent, child, allowed=(), path: str = ""):
    """Primo percorso (con punti; indici delle liste come numeri) in cui child differisce da parent, ignorando i percorsi
    di allowed; None se uguali. Tipi confrontati in modo stretto (true non è 1, 4 non è 4.0); chiavi mancanti o in più
    contano come differenze; l'ordine delle chiavi non conta."""
    if type(parent) is not type(child):
        return path or "<root>"
    if isinstance(parent, dict):
        for k in list(parent) + [k for k in child if k not in parent]:
            sub = f"{path}.{k}" if path else str(k)
            if sub in allowed:
                continue
            if k not in parent or k not in child:
                return sub
            if (d := config_difference(parent[k], child[k], allowed, sub)) is not None:
                return d
        return None
    if isinstance(parent, list):
        for i in range(max(len(parent), len(child))):
            sub = f"{path}.{i}" if path else str(i)
            if i >= len(parent) or i >= len(child):
                return sub
            if (d := config_difference(parent[i], child[i], allowed, sub)) is not None:
                return d
        return None
    return None if parent == child else (path or "<root>")


def check_amendment(cfg: dict, cfg_sha: str):
    """Configurazione emendata (chiave amendment): SHA-256 uguale a quello fissato (PATIENCE12_CONFIG_SHA256), genitore
    dichiarato = la configurazione consegnata (CONFIG_SHA256), genitore letto dal repository con lo stesso SHA-256 e uguale
    in ogni campo tranne training.early_stopping_patience (4 -> 12) e la chiave amendment. None senza emendamento."""
    am = cfg.get(AMENDMENT_KEY)
    if am is None:
        return None
    if cfg_sha != PATIENCE12_CONFIG_SHA256:
        fail(f"amended configuration: SHA-256 {cfg_sha} differs from the agreed configuration ({PATIENCE12_CONFIG_SHA256[:12]}...)")
    if am["parent_config_sha256"] != CONFIG_SHA256:
        fail(f"amended configuration: parent_config_sha256 differs from the delivered configuration ({CONFIG_SHA256[:12]}...)")
    pp = ROOT / PARENT_CONFIG
    pb = read_regular(pp, "parent configuration")
    ph = sha256_bytes(pb)
    if ph != CONFIG_SHA256:
        fail(f"parent configuration {pp.name}: SHA-256 {ph} differs from the delivered configuration ({CONFIG_SHA256[:12]}...)")
    parent = read_json(pb, f"parent configuration {pp.name}")
    if (d := config_difference(parent, cfg, AMENDMENT_ALLOWED)) is not None:
        fail(f"amended configuration: field {d} differs from the parent configuration (only {AMENDMENT_FIELD} "
             f"{AMENDMENT_FROM} -> {AMENDMENT_TO} and {AMENDMENT_KEY} may differ)")
    pat = (parent.get("training", {}).get("early_stopping_patience"), cfg["training"]["early_stopping_patience"])
    if pat != (AMENDMENT_FROM, AMENDMENT_TO) or any(type(v) is not int for v in pat):     # tipi stretti: 12.0 non è 12
        fail(f"amended configuration: {AMENDMENT_FIELD} must go from {AMENDMENT_FROM} (parent) to {AMENDMENT_TO}")
    return {"parent_config_sha256": ph, "changed_field": AMENDMENT_FIELD,
            "from": AMENDMENT_FROM, "to": AMENDMENT_TO, "other_fields": "equal to the parent configuration"}


def fr_view(cfg: dict) -> dict:
    """La parte della configurazione nella forma attesa dalle funzioni riusate di nuaa_folds_run."""
    return {"manifests": cfg["manifests"], "schema_run": cfg["schema_run"], "expected": cfg["schema_expected"],
            "cache": cfg["cache"], "images": cfg["images"], "environment": cfg["environment"]}


# ============================================================================ ingressi

class Inputs(FR.Inputs):
    """Gli ingressi del run dei fold (manifest M2, schema, cache) più celle, frame estratti e riferimento A."""

    def __init__(self):
        super().__init__()
        self.cells = {}          # (braccio o "dev", fold) -> righe
        self.cells_sha = {}
        self.extracted_keys = []
        self.extracted_rows = []
        self.X_extracted = None
        self.reference = None    # aggregate.json del run dei fold


def read_json(b: bytes, what: str) -> dict:
    try:
        d = json.loads(b.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        fail(f"{what} is not valid JSON: {e}")
    if not isinstance(d, dict):
        fail(f"{what} is not a JSON object")
    return d


def check_schema_config(cfg: dict, base: Path, inp: Inputs) -> None:
    """La configurazione dello schema approvato: SHA-256 fissato nel codice e nella configurazione; seed, criterio della
    soglia, ricetta del training e numero di fold uguali a quelli di questo run."""
    sr = cfg["schema_run"]
    p = rel_path(base, sr["config_path"])
    b = read_regular(p, "schema configuration")
    h = sha256_bytes(b)
    if h != sr["config_sha256"] or h != SCHEMA_CONFIG_SHA256:
        fail(f"schema configuration {p.name}: SHA-256 {h} differs from the approved schema ({SCHEMA_CONFIG_SHA256[:12]}...)")
    sc = read_json(b, f"schema configuration {p.name}")
    if sc.get("train_seeds") != cfg["train_seeds"]:
        fail("schema configuration: train_seeds differ from this run")
    if sc.get("threshold", {}).get("criterion") != cfg["threshold"]["criterion"]:
        fail("schema configuration: threshold criterion differs from this run")
    tr = sc.get("training", {})
    diff = [k for k, v in FR.FROZEN_TRAINING.items() if k != "device" and tr.get(k) != v]
    if diff or not str(tr.get("device", "")).startswith("CPU"):
        fail(f"schema configuration: training fields {diff or ['device']} differ from the recipe of this run")
    if sc.get("folds", {}).get("k") != cfg["schema_expected"]["k"]:
        fail("schema configuration: number of folds differs from this run")
    inp.checks["schema_config"] = {"path": sr["config_path"], "sha256": h, "train_seeds": sc["train_seeds"], "k": sc["folds"]["k"]}


def cell_of(r: dict) -> str:
    return f"{r['session']}/{CLASS_NAME[r['label']]}"


def load_cells_run(cfg: dict, base: Path, inp: Inputs) -> None:
    """Il run delle celle: status.json PASS con la configurazione e i due SHA256SUMS fissati, SHA256SUMS pubblico e
    riservato ancorati, configurazione delle celle (SHA-256 nel codice e nella configurazione) con le stesse allocazioni
    dei bracci; i 16 manifest usati (bracci e dev comune) uguali alle righe del SHA256SUMS riservato. devmix_fold<k>.csv
    non è mai letto (escluso dal coordinatore)."""
    cr = cfg["cells_run"]
    folder = rel_path(base, cr["folder"])
    st = read_json(read_regular(folder / STATUS, "cells run status.json"), "cells run status.json")
    if st.get("status") != "PASS" or st.get("run_id") != cr["run_id"] or st.get("config_sha256") != cr["config_sha256"] \
            or st.get("public_sha256sums") != cr["public_sha256sums_sha256"] or st.get("restricted_sha256sums") != cr["restricted_sha256sums_sha256"]:
        fail(f"cells run {cr['run_id']}: status.json is not PASS with the run id, configuration and SHA256SUMS of this configuration")
    pub = read_regular(folder / PUBLIC / "SHA256SUMS", "cells run public SHA256SUMS")
    if sha256_bytes(pub) != cr["public_sha256sums_sha256"]:
        fail(f"cells run {cr['run_id']}: public SHA256SUMS has SHA-256 {sha256_bytes(pub)}, expected {cr['public_sha256sums_sha256']}")
    sums_b = read_regular(folder / RESTRICTED / "SHA256SUMS", "cells run restricted SHA256SUMS")
    h = sha256_bytes(sums_b)
    if h != cr["restricted_sha256sums_sha256"]:
        fail(f"cells run {cr['run_id']}: restricted SHA256SUMS has SHA-256 {h}, expected {cr['restricted_sha256sums_sha256']}")
    listed = read_sums(sums_b, "cells run restricted SHA256SUMS")
    cp = rel_path(base, cr["config_path"])
    cb = read_regular(cp, "cells configuration")
    ch = sha256_bytes(cb)
    if ch != cr["config_sha256"] or ch != CELLS_CONFIG_SHA256:
        fail(f"cells configuration {cp.name}: SHA-256 {ch} differs from the configuration of the cells run ({CELLS_CONFIG_SHA256[:12]}...)")
    design = read_json(cb, f"cells configuration {cp.name}").get("design", {})
    arms = cfg["arms"]
    want = {"arm_b": {x: design.get("arm_b", {}).get("per_cell") for x in design.get("arm_b", {}).get("cells", [])},
            "arm_a_star": design.get("arm_a_star", {}).get("allocation"), "arm_b_prime": design.get("arm_b_prime", {}).get("allocation")}
    for arm in ARMS:
        if want[arm] != arms[arm]["cells"]:
            fail(f"cells configuration: the cells of {arm} {want[arm]} differ from this configuration {arms[arm]['cells']}")
    dev_want = {x: design.get("dev", {}).get("per_cell") for x in design.get("dev", {}).get("cells", [])}
    if dev_want != cfg["dev"]["cells"]:
        fail("cells configuration: the cells of the common dev differ from this configuration")
    k = cfg["schema_expected"]["k"]
    for fold in range(1, k + 1):
        for unit, fname, role, cells in [(arm, ARM_FILE[arm], ARM_ROLE[arm], arms[arm]["cells"]) for arm in ARMS] + \
                                        [("dev", DEV_FILE, DEV_ROLE, cfg["dev"]["cells"])]:
            name = fname.format(k=fold)
            if name not in listed:
                fail(f"cells run SHA256SUMS: {name} not listed")
            b = read_regular(folder / RESTRICTED / name, f"cell manifest {name}")
            hh = sha256_bytes(b)
            if hh != listed[name]:
                fail(f"cell manifest {name}: SHA-256 {hh} differs from SHA256SUMS ({listed[name]})")
            rows = parse_csv(b, FOLD_COLUMNS, f"cell manifest {name}")
            if any(r["fold"] != str(fold) or r["role"] != role for r in rows):
                fail(f"cell manifest {name}: a row carries another fold or a role other than {role}")
            if len({r["key"] for r in rows}) != len(rows):
                fail(f"cell manifest {name}: duplicated keys")
            got = Counter(cell_of(r) if r["label"] in LABELS and r["session"] in SESSIONS else "?" for r in rows)
            if dict(got) != {x: n for x, n in cells.items()} or len(rows) != sum(cells.values()):
                fail(f"cell manifest {name}: counts per cell {dict(sorted(got.items()))} differ from the configuration {cells}")
            inp.cells[(unit, fold)] = rows
            inp.cells_sha[name] = hh
    inp.checks["cells_run"] = {"run_id": cr["run_id"], "config_sha256": ch, "restricted_sha256sums_sha256": h,
                               "public_sha256sums_sha256": cr["public_sha256sums_sha256"], "files": len(inp.cells_sha),
                               "not_read": "devmix_fold<k>.csv (dev-mix excluded by the coordinator)"}


def check_cells(cfg: dict, inp: Inputs) -> None:
    """Righe delle celle uguali alle liste ufficiali, flag in_cache, annidamento (B = sessioni 01-02 del fitting dello
    schema; A* 01-02 e B' dentro B; B' = A* 01-02; A* 03 dai soggetti di B), dev comune = sessioni 01-02 del dev dello
    schema (nessuna sessione 03), soggetti del fitting, del dev e del test disgiunti."""
    cache_keys = set(inp.cache_index)
    for (unit, fold), rows in inp.cells.items():
        for r in rows:
            o = inp.official.get(r["key"])
            if o is None or any(o[c] != r[c] for c in ("label", "subject", "session", "sha256", "bytes")):
                fail(f"cell manifest {unit} fold {fold}: a key is not in the official lists with the same label, subject, session, SHA-256 and size")
            if r["in_cache"] not in ("0", "1") or (r["in_cache"] == "1") != (r["key"] in cache_keys):
                fail(f"cell manifest {unit} fold {fold}: in_cache does not match the cache manifests")
    nest = {}
    for fold in range(1, cfg["schema_expected"]["k"] + 1):
        keys = {u: {r["key"] for r in inp.cells[(u, fold)]} for u in ARMS + ("dev",)}
        subj = {u: {r["subject"] for r in inp.cells[(u, fold)]} for u in ARMS + ("dev",)}
        a12 = {r["key"] for r in inp.cells[("arm_a_star", fold)] if r["session"] != "03"}
        schema_fit12 = {r["key"] for r in inp.folds[fold]["fitting"] if r["session"] != "03"}
        schema_dev12 = {r["key"] for r in inp.folds[fold]["dev"] if r["session"] != "03"}
        if any(r["session"] == "03" for r in inp.cells[("dev", fold)]):
            fail(f"fold {fold}: the common dev holds session-03 frames (the dev must be sessions 01-02 only)")
        if keys["dev"] != schema_dev12:
            fail(f"fold {fold}: the common dev is not the session 01-02 part of the dev of the fold schema")
        if keys["arm_b"] != schema_fit12:
            fail(f"fold {fold}: arm B is not the session 01-02 part of the fitting set of the fold schema")
        if not a12 <= keys["arm_b"] or not keys["arm_b_prime"] <= keys["arm_b"]:
            fail(f"fold {fold}: the session 01-02 frames of arm A* or arm B' are not nested in arm B")
        if keys["arm_b_prime"] != a12:
            fail(f"fold {fold}: arm B' differs from the session 01-02 part of arm A*")
        if not subj["arm_a_star"] <= subj["arm_b"] or not subj["arm_b_prime"] <= subj["arm_b"]:
            fail(f"fold {fold}: arm A* or arm B' uses subjects outside the S12 subjects of arm B")
        fit_subj = {r["subject"] for r in inp.folds[fold]["fitting"]}
        test_subj = {r["subject"] for r in inp.folds[fold]["test"]}
        dev_subj = {r["subject"] for r in inp.folds[fold]["dev"]}
        for arm in ARMS:
            if not subj[arm] <= fit_subj or subj[arm] & (test_subj | dev_subj) or subj[arm] & subj["dev"]:
                fail(f"fold {fold}: arm {arm} shares a subject with the dev or the test, or uses a subject outside the fitting folds")
        if subj["dev"] & test_subj:
            fail(f"fold {fold}: the common dev shares a subject with the test")
        nest[fold] = {"arm_b_equals_schema_fitting_sessions_01_02": True, "dev_equals_schema_dev_sessions_01_02": True,
                      "a_star_01_02_in_b": True, "b_prime_in_b": True, "b_prime_equals_a_star_01_02": True,
                      "s12_subjects": len(subj["arm_b"])}
    inp.checks["cells"] = {"nesting": nest, "counts": {arm: cfg["arms"][arm]["cells"] for arm in ARMS}, "dev": cfg["dev"]["cells"]}


def load_folds_run(cfg: dict, base: Path, inp: Inputs) -> None:
    """Il run dei fold: status.json PASS con la configurazione e i SHA256SUMS fissati; SHA256SUMS pubblico e riservato
    ancorati; extracted_frames.csv (riservato), aggregate.json (pubblico, riferimento A) e run_record.json (pubblico) uguali
    alle loro righe. Il codice importato (trainer, scorer, writer: SHARED_CODE) e l'architettura devono avere lo SHA-256
    registrato in code_sha256 del run_record.json del run dei fold: la ricetta usata qui è quella che ha prodotto quel run."""
    fr_ = cfg["folds_run"]
    folder = rel_path(base, fr_["folder"])
    st = read_json(read_regular(folder / STATUS, "folds run status.json"), "folds run status.json")
    if st.get("status") != "PASS" or st.get("run_id") != fr_["run_id"] or st.get("config_sha256") != fr_["config_sha256"] \
            or st.get("public_sums_sha256") != fr_["public_sha256sums_sha256"] or st.get("restricted_sums_sha256") != fr_["restricted_sha256sums_sha256"]:
        fail(f"folds run {fr_['run_id']}: status.json is not PASS with the run id, configuration and SHA256SUMS of this configuration")
    sums = {}
    for sub, key in ((PUBLIC, "public_sha256sums_sha256"), (RESTRICTED, "restricted_sha256sums_sha256")):
        b = read_regular(folder / sub / "SHA256SUMS", f"folds run {sub} SHA256SUMS")
        if sha256_bytes(b) != fr_[key]:
            fail(f"folds run {fr_['run_id']}: {sub} SHA256SUMS has SHA-256 {sha256_bytes(b)}, expected {fr_[key]}")
        sums[sub] = read_sums(b, f"folds run {sub} SHA256SUMS")
    files = {}
    for sub, name in ((RESTRICTED, "extracted_frames.csv"), (PUBLIC, "aggregate.json"), (PUBLIC, "run_record.json")):
        if name not in sums[sub]:
            fail(f"folds run {sub} SHA256SUMS: {name} not listed")
        b = read_regular(folder / sub / name, f"folds run {name}")
        if sha256_bytes(b) != sums[sub][name]:
            fail(f"folds run {name}: SHA-256 differs from its SHA256SUMS")
        files[name] = b
    rows = parse_csv(files["extracted_frames.csv"], FR.FRAME_COLUMNS, "folds run extracted_frames.csv")
    keys = [r["sample_id"] for r in rows]
    if keys != sorted(keys) or keys != inp.to_extract:
        fail("folds run extracted_frames.csv: its keys are not the frames outside the cache, in key order")
    for r in rows:
        o = inp.official[r["sample_id"]]
        if (r["subject_id"], r["session_id"], r["label"], r["file_sha256"]) != (o["subject"], o["session"], o["label"], o["sha256"]) \
                or not _hex(r["frame_sha256"]):
            fail("folds run extracted_frames.csv: a row differs from the official lists or has no frame SHA-256")
    inp.extracted_rows = rows
    agg = read_json(files["aggregate.json"], "folds run aggregate.json")
    if agg.get("run_id") != fr_["run_id"] or "pooled_per_seed" not in agg or "bootstrap_per_seed" not in agg:
        fail("folds run aggregate.json: run id or pooled values missing")
    inp.reference = {"run_id": fr_["run_id"], "aggregate_sha256": sums[PUBLIC]["aggregate.json"],
                     "pooled_per_seed": agg["pooled_per_seed"], "bootstrap_per_seed": agg["bootstrap_per_seed"]}
    rec = read_json(files["run_record.json"], "folds run run_record.json")
    code = rec.get("code_sha256")
    if rec.get("run_id") != fr_["run_id"] or not isinstance(code, dict) or SHARED_CODE[0] not in code:
        fail(f"folds run run_record.json: run id or code_sha256 of {SHARED_CODE[0]} missing")
    shared = {f: (sha256_bytes(read_regular(ROOT / f, "code file")), code[f]) for f in SHARED_CODE if f in code}
    if FOLDS_RUN_ARCH_KEY in code:
        shared["architecture"] = (cfg["architecture"]["sha256"], code[FOLDS_RUN_ARCH_KEY])
    bad = [f for f, (here, there) in shared.items() if here != there]
    if bad:
        fail(f"code shared with the folds run differs from its run_record.json (code_sha256): {', '.join(bad)}; the trainer and the "
             "scorer must be those of the folds run")
    inp.checks["folds_run"] = {"run_id": fr_["run_id"], "public_sha256sums_sha256": fr_["public_sha256sums_sha256"],
                               "restricted_sha256sums_sha256": fr_["restricted_sha256sums_sha256"],
                               "extracted_frames_csv_sha256": sums[RESTRICTED]["extracted_frames.csv"],
                               "aggregate_json_sha256": sums[PUBLIC]["aggregate.json"],
                               "run_record_json_sha256": sums[PUBLIC]["run_record.json"],
                               "shared_code_equal_to_folds_run": sorted(shared)}


def load_frames(cfg: dict, base: Path, inp: Inputs) -> None:
    """Il file dei frame estratti dal run dei fold: SHA-256, membri X (uint8) e keys senza pickle, chiavi = frame fuori
    dalla cache in ordine, hash di ogni frame uguale a extracted_frames.csv. I frame restano in memoria per il run."""
    ff = cfg["folds_run"]["frames_file"]
    p = regular_path(rel_path(base, ff["path"]), "extracted frames file")
    h = sha256_file(p)
    if h != ff["sha256"]:
        fail(f"extracted frames file {p.name}: SHA-256 {h} differs from the expected {ff['sha256']}")
    mem = FR.npz_members(p)
    want_x = {"dtype": "|u1", "shape": [ff["n"], *FR.FRAME_SHAPE], "fortran": False}
    if sorted(mem) != ["X", "keys"] or mem["X"] != want_x or mem["keys"]["shape"] != [ff["n"]] or mem["keys"]["dtype"][1] != "U":
        fail(f"extracted frames file {p.name}: members {mem} differ from X {want_x} and keys ({ff['n']} texts)")
    z = np.load(p, allow_pickle=False)
    keys = [str(x) for x in z["keys"]]
    if keys != inp.to_extract:
        fail(f"extracted frames file {p.name}: keys differ from the frames outside the cache (in key order)")
    X = z["X"]
    hashes = FR.frame_hashes(X)
    bad = sum(1 for a, r in zip(hashes, inp.extracted_rows) if a != r["frame_sha256"])
    if bad or len(hashes) != len(inp.extracted_rows):
        fail(f"extracted frames file {p.name}: {bad} frames differ from the frame SHA-256 of extracted_frames.csv")
    inp.extracted_keys, inp.X_extracted = keys, X
    inp.checks["frames_file"] = {"file": p.name, "sha256": h, "frames": len(keys), "frame_hashes": "every frame equal to extracted_frames.csv"}


def check_sources(cfg: dict, inp: Inputs) -> None:
    """Ogni riga di ogni manifest usato (celle, dev comune, test dello schema) trovata esattamente una volta: nella cache
    (posizione dell'ordine di campionamento M2) oppure nel file dei frame estratti, coerente con la colonna in_cache."""
    extracted = set(inp.extracted_keys)
    counts = Counter()
    units = [(f"{u} fold {f}", rows) for (u, f), rows in sorted(inp.cells.items())] + \
            [(f"test fold {f}", inp.folds[f]["test"]) for f in sorted(inp.folds)]
    for what, rows in units:
        for r in rows:
            n = (r["key"] in inp.cache_index) + (r["key"] in extracted)
            if n != 1:
                fail(f"{what}: a frame is found {n} times in the cache and the extracted frames (exactly once required)")
            if (r["in_cache"] == "1") != (r["key"] in inp.cache_index):
                fail(f"{what}: in_cache does not match the source of the frame")
            counts["cache" if r["key"] in inp.cache_index else "extracted"] += 1
    inp.checks["sources"] = {"rule": "every manifest row found exactly once, in the cache or in the extracted frames file",
                             "rows_from_cache": counts["cache"], "rows_from_extracted": counts["extracted"]}


def frozen_code_checks(cfg: dict, base: Path, inp: Inputs) -> None:
    """Architettura congelata (SHA-256) e codice del run. Il rilevatore non serve: nessuna estrazione."""
    ar = cfg["architecture"]
    p = rel_path(base, ar["path"])
    h = sha256_bytes(read_regular(p, "architecture file"))
    if h != ar["sha256"]:
        fail(f"architecture file {p.name}: SHA-256 {h} differs from the configuration")
    inp.code = {"architecture": h, **{f: sha256_bytes(read_regular(ROOT / f, "code file")) for f in CODE_FILES}}
    inp.checks["frozen_code"] = {"architecture": h, "detector": "not used (no extraction)"}


def load_inputs(cfg: dict, base: Path) -> Inputs:
    refused = [v for v in FR.ENV_REFUSED if os.environ.get(v, "").strip()]
    if refused:
        fail(f"environment variables {refused} are set: they would redirect the reference code (unset them)")
    inp = Inputs()
    v = fr_view(cfg)
    FR.load_m2_manifests(v, base, inp)
    FR.load_fold_manifests(v, base, inp)
    check_schema_config(cfg, base, inp)
    FR.load_cache(v, base, inp)
    load_cells_run(cfg, base, inp)
    check_cells(cfg, inp)
    load_folds_run(cfg, base, inp)
    load_frames(cfg, base, inp)
    check_sources(cfg, inp)
    frozen_code_checks(cfg, base, inp)
    inp.environment = FR.environment(v)
    inp.checks["environment"] = {**inp.environment, "tensorflow_state": FR.tf_state()}
    FR.check_originals(v, base, inp)
    return inp


# ============================================================================ training, soglia, punteggi

class Budget:
    """Tetto di tempo: elapsed() dall'avvio come massimo fra orologio monotono e di sistema; check(step) solleva
    CapExceeded (esito INCOMPLETE, non un errore) se il tetto è superato o se un arresto volontario è stato richiesto
    (stop_requested, impostato dal gestore di SIGUSR1: il segnale non interrompe il codice, il run si ferma al controllo
    successivo); lap registra entrambi i tempi di un passo."""

    def __init__(self, cap_minutes: int):
        self.cap = cap_minutes * 60
        self.start = (clock(), time.time())
        self.timings = {}
        self.stop_requested = None

    def elapsed(self) -> float:
        return max(clock() - self.start[0], time.time() - self.start[1])

    def check(self, step: str) -> None:
        if self.stop_requested:
            raise CapExceeded(step, self.elapsed() / 60, self.stop_requested)
        if self.elapsed() > self.cap:
            raise CapExceeded(step, self.elapsed() / 60)

    def mark(self) -> tuple:
        return (clock(), time.time())

    def lap(self, step: str, t0: tuple) -> None:
        self.timings[step] = {"monotonic_seconds": round(clock() - t0[0], 1), "wall_clock_seconds": round(time.time() - t0[1], 1)}


def score_rows(arm: str, fold: int, seed: int, role: str, part: dict, p_att: np.ndarray, thr: dict, card: dict) -> list:
    rows = []
    for i, r in enumerate(part["rows"]):
        v = float(p_att[i])
        rows.append({"arm": arm, "fold": str(fold), "train_seed": str(seed), "role": role, "sample_id": r["key"], "subject_id": r["subject"],
                     "session_id": r["session"], "label": r["label"], "score_attack": repr(v),
                     "prediction_05": "1" if v > FR.FIXED_THRESHOLD else "0", "prediction_dev": "1" if v > thr["value"] else "0",
                     "threshold_dev": repr(float(thr["value"])), "file_sha256": r["sha256"], "checkpoint_sha256": card["weights_sha256"],
                     "threshold_sha256": thr["sha256"]})
    return rows


def rows_text(rows) -> str:
    return csv_text(SCORE_COLUMNS, [[r[c] for c in SCORE_COLUMNS] for r in rows])


def tf_logits(weights_path, weights_sha: str):
    """Logit del checkpoint salvato: ingresso dello strato "output" (sotto-modello fino a quello strato, CPU, batch 16,
    BatchNorm in inferenza) e z = h @ W + b in float64. Serve al controllo del val_loss: Keras 2.15 lo calcola dai logit
    (softmax cross-entropy, senza clip), mentre FR.val_loss limita le probabilità a [1e-7, 1 - 1e-7]."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        import tensorflow as tf
    except ModuleNotFoundError as e:
        fail(f"missing Python package {e.name!r}: scoring needs TensorFlow")
    FR.hide_gpu(tf)
    if sha256_file(weights_path) != weights_sha:
        fail(f"{Path(weights_path).name}: SHA-256 changed just before loading")
    with tf.device("/CPU:0"):
        model = tf.keras.models.load_model(str(weights_path), compile=False)
        out = model.get_layer("output")
        penult = tf.keras.Model(model.input, out.input)
    W, b = (np.asarray(v, dtype=np.float64) for v in out.get_weights())

    def logits(X):
        hs = []
        for i in range(0, len(X), FR.SCORE_BATCH):
            xb = X[i:i + FR.SCORE_BATCH].astype(np.float32) / 255.0
            with tf.device("/CPU:0"):
                hs.append(np.asarray(penult(xb, training=False), dtype=np.float64))
        return np.concatenate(hs) @ W + b if hs else np.zeros((0, 2))
    return logits


def val_loss_logits(z: np.ndarray, y: np.ndarray) -> float:
    """Categorical cross-entropy media dai logit, come Keras 2.15 (softmax_cross_entropy_with_logits, nessun clip):
    logsumexp(z) - z[y], in float64."""
    z = np.asarray(z, dtype=np.float64)
    m = z.max(axis=1)
    lse = m + np.log(np.exp(z - m[:, None]).sum(axis=1))
    return float(np.mean(lse - z[np.arange(len(y)), y]))


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def dev_logits(weights: Path, card: dict, X, p_dev: np.ndarray, name: str) -> np.ndarray:
    """Logit del dev dal checkpoint, controllati: forma N x 2, finiti, softmax uguale alle probabilità dello scorer."""
    z = np.asarray((MAKE_LOGITS or tf_logits)(weights, card["weights_sha256"])(X), dtype=np.float64)
    if z.shape != (len(X), 2) or not np.all(np.isfinite(z)):
        fail(f"{name}: the logits have shape {z.shape} or non-finite values ({len(X)} x 2 expected)")
    gap = float(np.max(np.abs(softmax(z) - p_dev))) if len(X) else 0.0
    if gap > PROB_TOL:
        fail(f"{name}: the softmax of the output-layer logits differs from the scorer probabilities by {gap:.2e}")
    return z


def train_one(cfg: dict, run: Path, store, inp: Inputs, arm: str, fold: int, seed: int, budget: Budget, arch) -> tuple:
    """Un modello: training, scheda, punteggi del dev comune, soglia congelata, punteggi del test; (scheda, soglia, righe
    del test). Il tetto è controllato prima del training, a fine epoca, prima della soglia e prima del test."""
    rdir = run / RESTRICTED
    rdir.mkdir(exist_ok=True)
    name = model_name(arm, fold, seed)
    budget.check(f"train {name}")
    fit, dev = FR.assemble(store, inp.cells[(arm, fold)]), FR.assemble(store, inp.cells[("dev", fold)])
    weights = rdir / f"{name}_LivenessNet.keras"
    t0 = budget.mark()
    card = (TRAINER or FR.default_trainer)({"X_fit": fit["X"], "y_fit": fit["y"], "X_dev": dev["X"], "y_dev": dev["y"]}, cfg["training"],
                                           weights, seed, arch, lambda step: budget.check(f"{step} of {name}"))
    budget.lap(f"train {name}", t0)
    if problems := FR.card_problems(card, cfg["training"], seed, weights):
        fail(f"{name}: " + "; ".join(problems))
    if card["n_fitting"] != cfg["arms"][arm]["total"] or card["n_dev"] != cfg["dev"]["total"]:
        fail(f"{name}: the card reports {card['n_fitting']} fitting and {card['n_dev']} dev frames, expected "
             f"{cfg['arms'][arm]['total']} and {cfg['dev']['total']}")
    card = {**card, "arm": arm, "fold": fold, "fitting_keys_sha256": sha256_bytes("\n".join(fit["keys"]).encode()),
            "dev_keys_sha256": sha256_bytes("\n".join(dev["keys"]).encode())}
    del fit
    gc.collect()
    # punteggi del dev comune con il checkpoint salvato, val_loss ricalcolato, soglia EER scritta e riletta prima del test
    budget.check(f"threshold {name}")
    t0 = budget.mark()
    scorer = (MAKE_SCORER or FR.tf_scorer)(weights, card["weights_sha256"])
    p_dev = FR.probabilities(scorer, dev["X"])
    # il val_loss di Keras (minimo registrato) è calcolato dai logit senza clip: lo stesso calcolo qui; il valore con le
    # probabilità limitate (FR.val_loss) è solo informativo (diverge se un frame del dev ha perdita oltre -ln 1e-7)
    vl = val_loss_logits(dev_logits(weights, card, dev["X"], p_dev, name), dev["y"])
    diff = abs(vl - card["best_val_loss"])
    if not (diff <= FR.VAL_LOSS_TOL[0] or diff <= FR.VAL_LOSS_TOL[1] * card["best_val_loss"]):
        fail(f"{name}: val_loss recomputed on the dev {vl:.6f}, recorded minimum {card['best_val_loss']:.6f}: the checkpoint is not the "
             "one of minimum val_loss")
    card["val_loss_recomputed"] = vl
    card["val_loss_recomputed_clipped"] = FR.val_loss(p_dev, dev["y"])
    card["val_loss_check"] = VAL_LOSS_CHECK
    sel = FR.eer_threshold(dev["y"], p_dev[:, 1])
    thr = {"arm": arm, "fold": fold, "train_seed": seed, "set": cfg["threshold"]["set"], "criterion": cfg["threshold"]["criterion"],
           "value": float(sel["value"]), "fp_rejected_bona_fide": sel["fp"], "fn_accepted_attacks": sel["fn"], "n_attack": sel["n_attack"],
           "n_bona_fide": sel["n_bona_fide"], "objective": sel["objective"],
           "dev_scores_sha256": sha256_bytes("\n".join(repr(float(v)) for v in p_dev[:, 1]).encode()),
           "checkpoint_sha256": card["weights_sha256"], "frozen_at": now()}
    mw.write_all(rdir, {f"{name}_threshold.json": json.dumps(thr, indent=1) + "\n"}, newline="")
    back_bytes = read_regular(rdir / f"{name}_threshold.json", "threshold")
    if json.loads(back_bytes.decode("utf-8")) != thr:
        fail(f"{name}: threshold.json does not read back identical")
    thr["sha256"] = sha256_bytes(back_bytes)         # l'impronta della soglia congelata è quella dei byte riletti
    thr["fp"], thr["fn"] = sel["fp"], sel["fn"]
    mw.write_all(rdir, {f"{name}_dev_scores.csv": rows_text(score_rows(arm, fold, seed, "dev", dev, p_dev[:, 1], thr, card))}, newline="")
    del dev
    budget.lap(f"threshold {name}", t0)
    # punteggi del test, solo ora: i frame del test sono assemblati dopo la soglia
    budget.check(f"score {name}")
    t0 = budget.mark()
    test = FR.assemble(store, inp.folds[fold]["test"])
    p_test = FR.probabilities(scorer, test["X"])
    test_rows = score_rows(arm, fold, seed, "test", test, p_test[:, 1], thr, card)
    card["test_scored_after_threshold"] = now()
    mw.write_all(rdir, {f"{name}_test_scores.csv": rows_text(test_rows)}, newline="")
    mw.write_all(rdir, {f"{name}_card.json": json.dumps(card, indent=1) + "\n"})
    del test, scorer
    gc.collect()
    budget.lap(f"score {name}", t0)
    m = FR.stratum_metrics(test_rows)
    print(f"== {name}: test {m['n_bona_fide']}/{m['n_attack']}, ACER 05 {f4(m['acer_05'])}, ACER dev {f4(m['acer_dev'])} (threshold "
          f"{thr['value']:.6f}), AUROC {f4(m['auroc'])}, peak RSS {peak_rss_mb():.0f} MB", flush=True)
    return card, thr, test_rows


def checkpoint_temps(rdir: Path, name: str) -> list:
    """Temporanei del checkpoint di un modello (.<nome>_LivenessNet.keras.<pid>.tmp.keras, fuori dalla regola owned_name)."""
    try:
        entries = sorted(os.listdir(rdir))
    except OSError:
        return []
    return [e for e in entries if e.startswith(f".{name}_LivenessNet.keras.") and e.endswith(".tmp.keras")]


def discard_model(run: Path, name: str) -> list:
    """I file di un modello interrotto dal tetto o dall'arresto volontario (né salvato né valutato): tolti, temporanei
    compresi. Il trainer vero toglie il proprio temporaneo se il salvataggio si interrompe; la pulizia qui copre un trainer
    che lo lasci (provata dalla suite con un trainer finto)."""
    rdir = run / RESTRICTED
    if not rdir.is_dir():
        return []
    removed = mw.invalidate(rdir, [f"{name}_{x}" for x in PER_MODEL])
    for e in checkpoint_temps(rdir, name):
        (rdir / e).unlink()
        removed.append(e)
    return removed


# ============================================================================ metriche, bootstrap, confronti

def subject_stats(rows) -> dict:
    """Statistiche sufficienti per soggetto di un insieme di righe di punteggio (soggetti in ordine crescente): numeri di
    bona fide e attacchi, errori alle due soglie e la matrice U[i, j] = conteggio di Mann-Whitney fra gli attacchi del
    soggetto i e i bona fide del soggetto j (pareggi 0,5). Un ricampionamento con molteplicità w dà esattamente le
    metriche dei frame uniti: conteggi = w·c, numeratore dell'AUROC = wᵀUw."""
    subjects = sorted({r["subject_id"] for r in rows})
    pos = {s: i for i, s in enumerate(subjects)}
    n = len(subjects)
    st = {k: np.zeros(n, np.int64) for k in ("na", "nb", "acc05", "rej05", "accdev", "rejdev")}
    att, bf = [[] for _ in range(n)], [[] for _ in range(n)]
    for r in rows:
        i, lab = pos[r["subject_id"]], r["label"]
        if lab == "1":
            st["na"][i] += 1
            st["acc05"][i] += r["prediction_05"] == "0"
            st["accdev"][i] += r["prediction_dev"] == "0"
            att[i].append(float(r["score_attack"]))
        else:
            st["nb"][i] += 1
            st["rej05"][i] += r["prediction_05"] == "1"
            st["rejdev"][i] += r["prediction_dev"] == "1"
            bf[i].append(float(r["score_attack"]))
    U = np.zeros((n, n), np.float64)
    bfs = [np.sort(np.asarray(b, float)) for b in bf]
    for i in range(n):
        a = np.asarray(att[i], float)
        for j in range(n):
            if len(a) and len(bfs[j]):
                lo = np.searchsorted(bfs[j], a, side="left").sum()
                hi = np.searchsorted(bfs[j], a, side="right").sum()
                U[i, j] = (int(lo) + int(hi)) / 2.0           # bona fide sotto + metà dei pari, esatto (mezzi interi)
    return {"subjects": subjects, "U": U, **st}


def weighted_metrics(st: dict, w: np.ndarray) -> dict:
    """Metriche dei frame uniti dei soggetti con molteplicità w (stesse operazioni di nuaa_folds_run.stratum_metrics)."""
    na, nb = int(w @ st["na"]), int(w @ st["nb"])
    out = {}
    for name in ("05", "dev"):
        acc, rej = int(w @ st[f"acc{name}"]), int(w @ st[f"rej{name}"])
        apcer = acc / na if na else None
        bpcer = rej / nb if nb else None
        out[f"apcer_{name}"], out[f"bpcer_{name}"] = apcer, bpcer
        out[f"acer_{name}"] = (apcer + bpcer) / 2 if apcer is not None and bpcer is not None else None
    out["auroc"] = float((w @ st["U"] @ w) / (na * nb)) if na and nb else None
    return out


def boot_picks(n: int) -> list:
    """Gli indici dei ricampionamenti: generatore default_rng(BOOT["seed"]) nuovo a ogni chiamata (ripartito per seed,
    confronto o braccio e strato), integers(0, n, n) per ognuno dei BOOT["resamples"] ricampionamenti."""
    rng = np.random.default_rng(BOOT["seed"])
    return [rng.integers(0, n, n) for _ in range(BOOT["resamples"])]


def interval(vals: list) -> dict:
    if not vals:
        return {"low": None, "high": None, "valid_resamples": 0}
    lo, hi = np.percentile(vals, BOOT["percentiles"])
    return {"low": float(lo), "high": float(hi), "valid_resamples": len(vals)}


def arm_bootstrap(rows, keys=METRIC_KEYS) -> dict:
    """Intervallo per soggetto delle metriche pooled di un braccio (ricetta del run dei fold, stessi numeri)."""
    st = subject_stats(rows)
    n = len(st["subjects"])
    out = {"n_subjects": n, "resamples": BOOT["resamples"], "seed": BOOT["seed"], "percentiles": BOOT["percentiles"]}
    if n < BOOT["min_subjects"]:
        out["note"] = f"no interval: fewer than {BOOT['min_subjects']} subjects"
        return {**out, **{k: {"low": None, "high": None, "valid_resamples": 0} for k in keys}}
    vals = {k: [] for k in keys}
    for pick in boot_picks(n):
        m = weighted_metrics(st, np.bincount(pick, minlength=n))
        for k in keys:
            if m[k] is not None:
                vals[k].append(m[k])
    return {**out, **{k: interval(vals[k]) for k in keys}}


def paired_bootstrap(rows_m, rows_s) -> dict:
    """Bootstrap appaiato della differenza minuendo − sottraendo: stessi soggetti, stessi indici per i due bracci e per
    ogni metrica; un ricampionamento senza una classe non conta per le metriche che la richiedono."""
    sm, ss = subject_stats(rows_m), subject_stats(rows_s)
    if sm["subjects"] != ss["subjects"] or any(not np.array_equal(sm[k], ss[k]) for k in ("na", "nb")):
        fail("paired bootstrap: the two arms are not scored on the same subjects and frames")
    n = len(sm["subjects"])
    out = {"n_subjects": n, "resamples": BOOT["resamples"], "seed": BOOT["seed"], "percentiles": BOOT["percentiles"]}
    if n < BOOT["min_subjects"]:
        out["note"] = f"no interval: fewer than {BOOT['min_subjects']} subjects"
        return {**out, **{k: {"low": None, "high": None, "valid_resamples": 0} for k in METRIC_KEYS}}
    vals = {k: [] for k in METRIC_KEYS}
    for pick in boot_picks(n):
        w = np.bincount(pick, minlength=n)
        a, b = weighted_metrics(sm, w), weighted_metrics(ss, w)
        for k in METRIC_KEYS:
            if a[k] is not None and b[k] is not None:
                vals[k].append(a[k] - b[k])
    return {**out, **{k: interval(vals[k]) for k in METRIC_KEYS}}


def difference(a, b):
    return None if a is None or b is None else a - b


def same_frames(rows_m, rows_s) -> None:
    """Appaiamento: gli stessi frame (chiave, etichetta, soggetto, sessione) nello stesso ordine nei due bracci."""
    if [(r["sample_id"], r["label"], r["subject_id"], r["session_id"]) for r in rows_m] != \
            [(r["sample_id"], r["label"], r["subject_id"], r["session_id"]) for r in rows_s]:
        fail("comparison: the two arms are not scored on the same test frames")


def sign_word(metric: str, sign: int) -> str:
    return "higher" if sign > 0 else "lower"


def reading_verdict(comp: tuple, metric: str, per_seed: list) -> dict:
    """La regola di lettura, applicata meccanicamente: per_seed = per ogni seed {difference, low, high} oppure None."""
    name, m, s, _ = comp
    out = {"verdict": "inconclusive", "direction": None}
    if any(e is None for e in per_seed):
        return {**out, "reason": "a seed has no pooled value (incomplete run)"}
    d = [e["difference"] for e in per_seed]
    if any(x is None for x in d):
        return {**out, "reason": "a seed has no difference (a class absent)"}
    if any(e["low"] is None or e["high"] is None for e in per_seed):
        return {**out, "reason": "a seed has no bootstrap interval"}
    signs = {int(np.sign(x)) for x in d}
    if 0 in signs:
        return {**out, "reason": "a seed has a zero difference"}
    if len(signs) != 1:
        return {**out, "reason": "the sign differs across the seeds"}
    sign = signs.pop()
    if not all((e["low"] > 0) if sign > 0 else (e["high"] < 0) for e in per_seed):
        return {**out, "reason": "the bootstrap interval of a seed does not exclude 0 on the side of the difference"}
    return {"verdict": "consistent", "direction": f"{ARM_LABEL[m]} {sign_word(metric, sign)} {METRIC_LABEL[metric]} than {ARM_LABEL[s]}",
            "reason": "same sign for every seed and every interval excludes 0"}


def verdict_text(v: dict) -> str:
    return f"consistent: {v['direction']}" if v["verdict"] == "consistent" else "inconclusive"


def pooled_rows(scores: dict, arm: str, seed: int, k: int):
    """Righe del test dei quattro fold di un braccio e seed, in ordine di chiave; None se un fold manca (run incompleto)."""
    if any((arm, f, seed) not in scores for f in range(1, k + 1)):
        return None
    return sorted((r for f in range(1, k + 1) for r in scores[(arm, f, seed)]), key=lambda r: r["sample_id"])


def aggregate(cfg: dict, scores: dict, cards: dict, thresholds: dict, reference: dict) -> tuple:
    """scores[(braccio, fold, seed)] = righe del test dei modelli completi; (documento pubblico, righe per soggetto)."""
    seeds, k = cfg["train_seeds"], cfg["schema_expected"]["k"]
    per, subject_rows = {}, []
    for (arm, fold, seed), rows in sorted(scores.items()):
        per.setdefault(arm, {}).setdefault(str(fold), {})[str(seed)] = {
            st: (FR.stratum_metrics(FR.rows_of_stratum(rows, st)) if FR.rows_of_stratum(rows, st) else None) for st in STRATA}
        subject_rows += [[arm] + r for r in FR.per_subject_rows(fold, seed, rows)]
    pooled, boot, prow = {}, {}, {}
    for arm in ARMS:
        for seed in seeds:
            rows = pooled_rows(scores, arm, seed, k)
            if rows is None:
                continue
            prow[(arm, seed)] = rows
            pooled.setdefault(arm, {})[str(seed)] = {st: FR.stratum_metrics(FR.rows_of_stratum(rows, st)) for st in STRATA
                                                     if FR.rows_of_stratum(rows, st)}
            boot.setdefault(arm, {})[str(seed)] = {st: arm_bootstrap(FR.rows_of_stratum(rows, st)) for st in pooled[arm][str(seed)]}
    pooled_summary = {}
    for arm in pooled:
        if all(str(s) in pooled[arm] for s in seeds):
            pooled_summary[arm] = {st: {key: FR.summary([pooled[arm][str(s)][st][key] for s in seeds]) for key in METRIC_KEYS}
                                   for st in STRATA if all(st in pooled[arm][str(s)] for s in seeds)}
    comps = {}
    for comp in COMPARISONS:
        name, m, s, role = comp
        c = {"minuend": m, "subtrahend": s, "role": role, "per_fold_seed": {}, "pooled_per_seed": {}, "pooled_summary_over_seeds": {},
             "verdicts": {}}
        for fold in range(1, k + 1):
            for seed in seeds:
                if (m, fold, seed) in scores and (s, fold, seed) in scores:
                    same_frames(scores[(m, fold, seed)], scores[(s, fold, seed)])
                    a, b = per[m][str(fold)][str(seed)], per[s][str(fold)][str(seed)]
                    c["per_fold_seed"].setdefault(str(fold), {})[str(seed)] = {
                        st: ({key: difference(a[st][key], b[st][key]) for key in METRIC_KEYS} if a[st] and b[st] else None) for st in STRATA}
        for seed in seeds:
            if (m, seed) not in prow or (s, seed) not in prow:
                continue
            same_frames(prow[(m, seed)], prow[(s, seed)])
            c["pooled_per_seed"][str(seed)] = {}
            for st in STRATA:
                rm, rs = FR.rows_of_stratum(prow[(m, seed)], st), FR.rows_of_stratum(prow[(s, seed)], st)
                if not rm:
                    continue
                pb = paired_bootstrap(rm, rs)
                a, b = pooled[m][str(seed)][st], pooled[s][str(seed)][st]
                c["pooled_per_seed"][str(seed)][st] = {
                    "n_subjects": pb["n_subjects"], **({"note": pb["note"]} if "note" in pb else {}),
                    **{key: {"difference": difference(a[key], b[key]), "minuend": a[key], "subtrahend": b[key], **pb[key]} for key in METRIC_KEYS}}
        for st in STRATA:
            per_seed = {key: [c["pooled_per_seed"].get(str(seed), {}).get(st, {}).get(key) for seed in seeds] for key in METRIC_KEYS}
            if all(e is not None for es in per_seed.values() for e in es):
                c["pooled_summary_over_seeds"][st] = {key: FR.summary([e["difference"] for e in per_seed[key]]) for key in METRIC_KEYS}
            c["verdicts"][st] = {}
            for key in METRIC_KEYS:
                v = reading_verdict(comp, key, per_seed[key])
                primary = name == PRIMARY["comparison"] and st == PRIMARY["stratum"] and key in PRIMARY["metrics"]
                c["verdicts"][st][key] = {**v, "label": "primary" if primary else "secondary"}
        comps[name] = c
    primary = {key: comps[PRIMARY["comparison"]]["verdicts"][PRIMARY["stratum"]][key] for key in PRIMARY["metrics"]}
    training = {}
    for (arm, fold, seed), card in sorted(cards.items()):
        t = thresholds[(arm, fold, seed)]
        training[model_name(arm, fold, seed)] = {
            "arm": arm, "fold": fold, "seed": seed, "epochs_run": card["epochs_run"], "restored_epoch": card["restored_epoch"],
            "restored_by": card["restored_by"], "best_val_loss": card["best_val_loss"], "train_seconds": card["train_seconds"],
            "n_fitting": card["n_fitting"], "n_dev": card["n_dev"], "weights_sha256": card["weights_sha256"], "dev_threshold": t["value"],
            "dev_threshold_sha256": t["sha256"], "dev_apcer_at_threshold": t["fn"] / t["n_attack"], "dev_bpcer_at_threshold": t["fp"] / t["n_bona_fide"]}
    ref = {"label": REFERENCE_LABEL, "run_id": reference["run_id"], "aggregate_sha256": reference["aggregate_sha256"],
           "pooled_per_seed": {str(s): {st: {key: reference["pooled_per_seed"].get(str(s), {}).get(st, {}).get(key) for key in METRIC_KEYS + ("n", "n_subjects")}
                                        for st in STRATA} for s in seeds},
           "bootstrap_per_seed": {str(s): {st: reference["bootstrap_per_seed"].get(str(s), {}).get(st) for st in STRATA} for s in seeds}}
    doc = {"per_arm_fold_seed": per, "pooled_per_arm_seed": pooled, "bootstrap_per_arm_seed": boot, "pooled_summary_over_seeds": pooled_summary,
           "comparisons": comps, "primary_verdicts": primary, "reading_rule": READING_RULE, "reference_a": ref, "training": training}
    return doc, subject_rows


# ============================================================================ testo pubblico

def table(headers, rows) -> str:
    return FR.table(headers, rows)


def ci(e: dict) -> str:
    return f"[{f4(e['low'])}, {f4(e['high'])}]" if e and e.get("low") is not None else "no interval"


def aggregate_md(cfg: dict, agg: dict, record: dict, run_id: str) -> str:
    seeds, k = cfg["train_seeds"], cfg["schema_expected"]["k"]
    status = record["status"]
    L = [f"# NUAA session-03 ablation: LivenessNet, arms B / A* / B', {k} subject folds x {len(seeds)} seeds (aggregate, no identifiers)", "",
         f"Run `{run_id}` (commit `{record['commit']}`, dirty tree: {'yes' if record['dirty'] else 'no'}), {record['finished']}. "
         f"Outcome: **{status}**, {len(record['completed'])} of {record['trainings']} trainings complete"
         + (f"; stopped at {record['stopped']['step']} ({record['stopped']['elapsed_minutes']:.1f} min, "
            f"{'wall-time cap' if record['stopped']['cause'] == 'cap' else 'stop requested by ' + record['stopped']['cause']}); missing: "
            f"{', '.join(record['missing'])}" if record["stopped"] else "") + ". "
         f"Configuration `{record['config_sha256'][:12]}`"
         + (f" ({record['configuration_name']}: {AMENDMENT_FIELD} {record['amendment']['from']} -> {record['amendment']['to']}, every "
            f"other field equal to the parent configuration `{record['amendment']['parent_config_sha256'][:12]}`)" if record.get("amendment") else "")
         + f"; cells run {cfg['cells_run']['run_id']}, fold schema {cfg['schema_run']['run_id']}, "
         f"frames of {cfg['folds_run']['run_id']}. Every number below is reported whatever its value (pre-commitment). Diagnostic, not "
         "causal; no statistical test.", "",
         "Arms (same fitting subjects S12 per fold): B = sessions 01-02 (" + "/".join(str(v) for v in cfg["arms"]["arm_b"]["cells"].values())
         + f" per cell, {cfg['arms']['arm_b']['total']} frames); A* = sessions 01-03 ("
         + "/".join(str(v) for v in cfg["arms"]["arm_a_star"]["cells"].values()) + f", {cfg['arms']['arm_a_star']['total']}); B' = sessions "
         "01-02 at the per-cell counts of A* (" + "/".join(str(v) for v in cfg["arms"]["arm_b_prime"]["cells"].values())
         + f", {cfg['arms']['arm_b_prime']['total']}). Common dev: sessions 01-02 of the dev-fold subjects ({cfg['dev']['total']} frames), "
         "for early stopping, the learning-rate schedule, the restored epoch and the EER threshold. Test: every frame of the held-out "
         "subjects (the fold schema, unchanged).", "",
         "Decision: attack if score_attack > threshold (tie = bona fide). `05` = the fixed threshold 0.5; `dev` = the EER threshold of the "
         "common dev (M1 criterion), frozen before any test scoring. AUROC: threshold-free (Mann-Whitney, ties 0.5). NA where a class is "
         "absent. Differences are minuend - subtrahend, paired on the same test frames and seeds (nominal pairing: batch order and "
         "augmentation differ with the data).", "",
         "## Reading rule (fixed in the configuration before the run)", "", READING_RULE[0].upper() + READING_RULE[1:] + ".", "",
         "## Primary verdicts (A* - B, stratum 03)", ""]
    pc = agg["comparisons"][PRIMARY["comparison"]]
    rows = []
    for key in PRIMARY["metrics"]:
        v = agg["primary_verdicts"][key]
        per = [pc["pooled_per_seed"].get(str(s), {}).get(PRIMARY["stratum"], {}).get(key) for s in seeds]
        rows.append([METRIC_LABEL[key]] + [f"{f4(e['difference'])} {ci(e)}" if e else "missing" for e in per] + [verdict_text(v), v["reason"]])
    L.append(table(["metric"] + [f"seed {s}: difference [bootstrap]" for s in seeds] + ["verdict", "reason"], rows))
    for name, m, s, role in COMPARISONS:
        c = agg["comparisons"][name]
        L += ["", f"## Comparison {ARM_LABEL[m]} - {ARM_LABEL[s]} ({role}): pooled over the {k} folds, per seed", ""]
        rows = []
        for seed in seeds:
            for st in STRATA:
                e = c["pooled_per_seed"].get(str(seed), {}).get(st)
                if e is None:
                    continue
                rows.append([seed, st, e["n_subjects"]] + [f"{f4(e[key]['difference'])} {ci(e[key])}" for key in METRIC_KEYS])
        L.append(table(["seed", "stratum", "subjects"] + [f"{METRIC_LABEL[key]}: difference [bootstrap]" for key in METRIC_KEYS], rows)
                 if rows else "No pooled difference (incomplete run).\n")
        L += ["", f"Verdicts ({ARM_LABEL[m]} - {ARM_LABEL[s]}; primary only for AUROC and ACER dev in stratum 03 of A* - B)", ""]
        rows = [[st, METRIC_LABEL[key], c["verdicts"][st][key]["label"], verdict_text(c["verdicts"][st][key]), c["verdicts"][st][key]["reason"]]
                for st in STRATA for key in METRIC_KEYS]
        L.append(table(["stratum", "metric", "label", "verdict", "reason"], rows))
        L += ["", "Median and range over the seeds of the pooled differences", ""]
        rows = [[st] + [f"{f4(v[key]['median'])} ({f4(v[key]['min'])} to {f4(v[key]['max'])})" for key in METRIC_KEYS]
                for st, v in c["pooled_summary_over_seeds"].items()]
        L.append(table(["stratum"] + [METRIC_LABEL[key] for key in METRIC_KEYS], rows) if rows else "Not available (incomplete run).\n")
        L += ["", f"Per fold and seed ({ARM_LABEL[m]} - {ARM_LABEL[s]})", ""]
        rows = []
        for fold in range(1, k + 1):
            for seed in seeds:
                d = c["per_fold_seed"].get(str(fold), {}).get(str(seed))
                if d is None:
                    continue
                for st in STRATA:
                    if d[st] is not None:
                        rows.append([fold, seed, st] + [f4(d[st][key]) for key in METRIC_KEYS])
        L.append(table(["fold", "seed", "stratum"] + [METRIC_LABEL[key] for key in METRIC_KEYS], rows) if rows else "Not available.\n")
    L += ["", "## Per arm: pooled over the folds, per seed (bootstrap over subjects)", ""]
    rows = []
    for arm in ARMS:
        for seed in seeds:
            for st in STRATA:
                mm = agg["pooled_per_arm_seed"].get(arm, {}).get(str(seed), {}).get(st)
                if mm is None:
                    continue
                b = agg["bootstrap_per_arm_seed"][arm][str(seed)][st]
                rows.append([ARM_LABEL[arm], seed, st, mm["n_bona_fide"], mm["n_attack"], mm["n_subjects"], f4(mm["bpcer_05"]), f4(mm["apcer_05"]),
                             f"{f4(mm['acer_05'])} {ci(b['acer_05'])}", f4(mm["bpcer_dev"]), f4(mm["apcer_dev"]), f"{f4(mm['acer_dev'])} {ci(b['acer_dev'])}",
                             f"{f4(mm['auroc'])} {ci(b['auroc'])}"])
    L.append(table(["arm", "seed", "stratum", "n bona fide", "n attack", "subjects", "BPCER 05", "APCER 05", "ACER 05 [bootstrap]", "BPCER dev",
                    "APCER dev", "ACER dev [bootstrap]", "AUROC [bootstrap]"], rows) if rows else "Not available.\n")
    L += ["", "## Per arm, fold and seed (held-out subjects, by session stratum)", ""]
    rows = []
    for arm in ARMS:
        for fold in range(1, k + 1):
            for seed in seeds:
                d = agg["per_arm_fold_seed"].get(arm, {}).get(str(fold), {}).get(str(seed))
                if d is None:
                    continue
                for st in STRATA:
                    mm = d[st]
                    if mm is None:
                        continue
                    rows.append([ARM_LABEL[arm], fold, seed, st, mm["n_bona_fide"], mm["n_attack"], mm["n_subjects"], f4(mm["bpcer_05"]), f4(mm["apcer_05"]),
                                 f4(mm["acer_05"]), f4(mm["bpcer_dev"]), f4(mm["apcer_dev"]), f4(mm["acer_dev"]), f4(mm["auroc"])])
    L.append(table(["arm", "fold", "seed", "stratum", "n bona fide", "n attack", "subjects", "BPCER 05", "APCER 05", "ACER 05", "BPCER dev",
                    "APCER dev", "ACER dev", "AUROC"], rows) if rows else "Not available.\n")
    ref = agg["reference_a"]
    L += ["", f"## Reference A ({ref['label']})", "",
          f"Run {ref['run_id']} (aggregate.json `{ref['aggregate_sha256'][:12]}`): 350 frames per cell over six cells, all eight fitting "
          "subjects, dev with session 03; pooled values per seed copied unchanged; not compared with a bootstrap.", ""]
    rows = []
    for seed in seeds:
        for st in STRATA:
            mm, b = ref["pooled_per_seed"][str(seed)][st], ref["bootstrap_per_seed"][str(seed)][st] or {}
            rows.append([seed, st, f"{f4(mm['acer_05'])} {ci(b.get('acer_05'))}", f"{f4(mm['acer_dev'])} {ci(b.get('acer_dev'))}",
                         f"{f4(mm['auroc'])} {ci(b.get('auroc'))}"])
    L.append(table(["seed", "stratum", "ACER 05 [bootstrap]", "ACER dev [bootstrap]", "AUROC [bootstrap]"], rows))
    L += ["", "## Training cards and dev thresholds", ""]
    rows = [[name, t["n_fitting"], t["n_dev"], t["epochs_run"], t["restored_epoch"], t["restored_by"], f"{t['best_val_loss']:.4f}",
             f"{t['train_seconds']:.0f}", f"{t['dev_threshold']:.6f}", f4(t["dev_bpcer_at_threshold"]), f4(t["dev_apcer_at_threshold"])]
            for name, t in agg["training"].items()]
    L.append(table(["model", "n fitting", "n dev", "epochs", "restored epoch", "restored by", "min val_loss", "seconds", "dev threshold",
                    "dev BPCER at threshold", "dev APCER at threshold"], rows) if rows else "No complete model.\n")
    L += ["", "## Steps, budget, environment and inputs", "",
          table(["step", "seconds (wall clock)", "seconds (monotonic)"],
                [[s, f"{v['wall_clock_seconds']:.0f}", f"{v['monotonic_seconds']:.0f}"] for s, v in record["timings"].items()]),
          f"Wall time {record['wall_seconds']:.0f} s of the cap {cfg['budget']['wall_time_cap_minutes'] * 60} s (checked before every training, "
          "at the end of every epoch, before every threshold and every test scoring; the aggregation and the final writes run after the "
          f"last model, also after the cap; cap already exceeded before them: {'yes' if record['final_writes_after_cap'] else 'no'}). "
          f"Environment: {json.dumps(record['environment'], sort_keys=True)}.", "",
          "Inputs (SHA-256): cells run restricted SHA256SUMS `" + cfg["cells_run"]["restricted_sha256sums_sha256"] + "`, fold schema "
          "restricted SHA256SUMS `" + cfg["schema_run"]["restricted_sha256sums_sha256"] + "`, cache `" + cfg["cache"]["sha256"] + "`, "
          "extracted frames `" + cfg["folds_run"]["frames_file"]["sha256"] + "`, architecture `" + cfg["architecture"]["sha256"] + "`.", "",
          "## Limits", "",
          f"Bootstrap: {BOOT_RECIPE}. " + cfg["declarations"]["limits"], ""]
    return "\n".join(L)


# ============================================================================ run

def repo_state(root: Path) -> dict:
    if REPO_STATE is not None:
        return REPO_STATE(root)
    try:
        c = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], cwd=root, capture_output=True, text=True, timeout=30)
        s = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        fail(f"git not available: {type(e).__name__}")
    if c.returncode or s.returncode or not c.stdout.strip():
        fail("git state not readable: the run id needs the commit")
    return {"commit": c.stdout.strip(), "dirty": bool(s.stdout.strip())}


def run_id_for(date: str, state: dict) -> str:
    if not re.fullmatch(r"\d{8}", date or ""):
        fail("--date must be YYYYMMDD")
    rid = f"{date}-SESSABLRUN-{state['commit']}" + ("-dev" if state["dirty"] else "")
    if not RUN_ID_RE.match(rid):
        fail(f"run id {rid} does not follow YYYYMMDD-SESSABLRUN-<commit>[-dev]")
    return rid


def all_models(cfg: dict) -> list:
    return [model_name(*x) for x in training_order(cfg["schema_expected"]["k"], cfg["train_seeds"])]


def restricted_outputs(cfg: dict) -> tuple:
    return tuple(f"{n}_{x}" for n in all_models(cfg) for x in PER_MODEL) + RESTRICTED_FIXED


def protected_extra(root: Path) -> list:
    return [root / x for x in EXTRA_PROTECTED]


def input_paths(cfg: dict, base: Path, cfg_p: Path) -> dict:
    out = {"the configuration": cfg_p}
    out.update({f"manifest {m}": rel_path(base, cfg["manifests"]["folder"]) / cfg["manifests"]["files"][m]["file"] for m in FR.MANIFESTS})
    out["the schema folder"] = rel_path(base, cfg["schema_run"]["folder"])
    out["the schema configuration"] = rel_path(base, cfg["schema_run"]["config_path"])
    out["the cells run folder"] = rel_path(base, cfg["cells_run"]["folder"])
    out["the cells configuration"] = rel_path(base, cfg["cells_run"]["config_path"])
    out["the folds run folder"] = rel_path(base, cfg["folds_run"]["folder"])
    out["the extracted frames file"] = rel_path(base, cfg["folds_run"]["frames_file"]["path"])
    out["the cache"] = rel_path(base, cfg["cache"]["path"])
    out["the images folder"] = rel_path(base, cfg["images"]["folder"])
    out["the architecture file"] = rel_path(base, cfg["architecture"]["path"])
    return out


def run_dir_problem(run: Path, root: Path, inputs: dict, cfg: dict, out_root: Path = None) -> str:
    """Motivo per cui la cartella del run non è ammessa ("" se lo è), prima di qualunque modifica."""
    if (out_root if out_root is not None else run.parent).is_symlink():
        return "the output root is a symbolic link"
    for label, p in inputs.items():
        if pg.within(p, run) or (Path(p).is_dir() and pg.within(run, p)):
            return f"{label} {Path(p).name} is inside the run folder, or contains it: inputs and outputs must be separate"
    msg = mw.output_problem(run, root, pg.BASE_PROTECTED, [root / DEFAULT_OUT_ROOT], protected_extra(root), inputs, [STATUS, LOG],
                            [run / PUBLIC, run / RESTRICTED])
    if msg:
        return msg
    for sub, names in ((PUBLIC, PUBLIC_OUTPUTS), (RESTRICTED, restricted_outputs(cfg))):
        if msg := mw.output_problem(run / sub, root, pg.BASE_PROTECTED, [root / DEFAULT_OUT_ROOT], protected_extra(root), inputs, list(names)):
            return msg
    allowed = {"": [STATUS, LOG, PUBLIC, RESTRICTED], PUBLIC: PUBLIC_OUTPUTS, RESTRICTED: restricted_outputs(cfg)}
    for sub, names in allowed.items():
        d = run / sub if sub else run
        if d.is_dir():
            for e in sorted(os.listdir(d)):
                if not any(mw.owned_name(e, n) for n in names):
                    return f"{(Path(sub) / e).as_posix()} in the run folder is not an output of this command: refused, nothing removed"
    return ""


def invalidate_outputs(run: Path, cfg: dict) -> list:
    left = []
    for sub, names in ((PUBLIC, PUBLIC_OUTPUTS), (RESTRICTED, restricted_outputs(cfg))):
        d = run / sub
        if d.is_dir():
            left += [f"{sub}/{n}" for n in mw.invalidate_each(d, list(names), None, FAIL_DOC)]
    return left


def resolve(a) -> tuple:
    base = Path(a.input_root).resolve() if a.input_root else ROOT
    cfg_p = Path(a.config) if Path(a.config).is_absolute() else ROOT / a.config
    return base, cfg_p


def cmd_check_inputs(a) -> int:
    base, cfg_p = resolve(a)
    cfg, cfg_sha = load_config(cfg_p)
    amendment = check_amendment(cfg, cfg_sha)
    inp = load_inputs(cfg, base)
    src = inp.checks["sources"]
    if amendment:
        print(f"AMENDMENT: configuration {configuration_name(cfg_sha)} {cfg_sha[:12]}, parent {amendment['parent_config_sha256'][:12]} "
              f"equal in every field except {AMENDMENT_FIELD} ({AMENDMENT_FROM} -> {AMENDMENT_TO}) and {AMENDMENT_KEY}")
    print(f"CHECK-INPUTS: configuration {cfg_sha[:12]}, {len(FR.MANIFESTS)} M2 manifests, {len(FR.ROLES) * len(inp.folds)} fold manifests of "
          f"{cfg['schema_run']['run_id']}, {len(inp.cells_sha)} cell manifests of {cfg['cells_run']['run_id']} (counts, nesting, disjoint "
          f"subjects; devmix not read), cache {len(inp.cache_index)} frames linked, extracted frames {len(inp.extracted_keys)} with every "
          f"frame SHA-256 equal to {cfg['folds_run']['run_id']}, manifest rows from the cache {src['rows_from_cache']} and from the "
          f"extracted file {src['rows_from_extracted']} (each exactly once), {inp.checks['originals']['located_and_hashed']} originals, "
          f"architecture verified, OpenCV {inp.environment['opencv']}, TensorFlow {inp.environment['tensorflow_installed']} installed and "
          "not imported; nothing written")
    print("RESULT: PASS (check-inputs)")
    return 0


def signal_handlers(budget: Budget = None):
    """SIGTERM (e SIGHUP, se non è già ignorato) fermano il run come un errore: FAIL (come il run dei fold; i modelli già
    completi restano, senza verdetto). SIGUSR1 è l'arresto volontario, trattato come il tetto: il gestore registra solo la
    richiesta (budget.stop_requested) e il run si ferma al controllo successivo con esito INCOMPLETE."""
    def handler(signum, frame):
        raise Failure(f"signal {signal.Signals(signum).name} received: run stopped")

    def stop(signum, frame):              # nessuna stampa qui: il gestore può interrompere una scrittura su stdout
        if budget is not None and not budget.stop_requested:
            budget.stop_requested = signal.Signals(signum).name
    sigs = [signal.SIGTERM]
    if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:
        sigs.append(signal.SIGHUP)
    try:
        for sig in sigs:
            signal.signal(sig, handler)
        if budget is not None:
            signal.signal(signal.SIGUSR1, stop)
    except (ValueError, OSError):      # non nel thread principale (test): nessun gestore
        pass


FAIL_KEPT_NOTE = "public removed; restricted outputs of {n} complete models kept, not aggregated; no verdict"


def keep_complete_models(run: Path, cfg: dict, completed: list) -> str:
    """FAIL dopo modelli completi (errore, SIGTERM, SIGHUP, Ctrl-C, rifiuto della scansione pubblica): il pubblico,
    per_subject.csv, i file e i temporanei del modello in corso sono tolti; i cinque file riservati di ogni modello completo
    restano, con un SHA256SUMS riservato solo su quelli (restituisce il suo SHA-256). Nessun aggregato, nessun verdetto.
    Se un file di un modello completo manca o qualcosa non si toglie, solleva Failure: il chiamante toglie tutto."""
    rdir = run / RESTRICTED
    keep = [f"{n}_{x}" for n in completed for x in PER_MODEL]
    left = []
    if (run / PUBLIC).is_dir():
        left += [f"{PUBLIC}/{n}" for n in mw.invalidate_each(run / PUBLIC, list(PUBLIC_OUTPUTS), None, FAIL_DOC)]
    left += [f"{RESTRICTED}/{n}" for n in mw.invalidate_each(rdir, [n for n in restricted_outputs(cfg) if n not in keep], None, FAIL_DOC)]
    for n in all_models(cfg):
        for e in checkpoint_temps(rdir, n):
            (rdir / e).unlink()
    for n in keep:                                   # temporanei di write_all lasciati accanto a un file conservato
        for q in mw.temp_names(rdir, n):
            q.unlink()
    if left:
        fail(f"outputs that could not be removed: {', '.join(left)}")
    bad = [n for n in keep if (rdir / n).is_symlink() or not (rdir / n).is_file()]
    if bad:
        fail(f"files of complete models missing: {', '.join(bad)}")
    extra = sorted(e for e in os.listdir(rdir) if e not in keep)
    if extra:
        fail(f"restricted folder holds other files: {', '.join(extra)}")
    sums = "".join(f"{sha256_file(rdir / n)}  {n}\n" for n in sorted(keep))
    mw.write_all(rdir, {"SHA256SUMS": sums})
    return sha256_bytes(sums.encode("utf-8"))


def cmd_run(a) -> int:
    if not a.i_am_authorized:
        fail("run trains and scores on the restricted NUAA data: pass --i-am-authorized (the coordinator authorised the run on "
             "5 November 2026, and the verification run with early_stopping_patience 12 on 7 November 2026)")
    base, cfg_p = resolve(a)
    cfg, cfg_sha = load_config(cfg_p)
    amendment = check_amendment(cfg, cfg_sha)
    state = repo_state(ROOT)
    rid = run_id_for(a.date or time.strftime("%Y%m%d"), state)
    out_root = Path(a.out_root) if Path(a.out_root).is_absolute() else ROOT / a.out_root
    run = out_root / rid
    inputs = input_paths(cfg, base, cfg_p)
    if (run / STATUS).exists() or (run / STATUS).is_symlink():
        fail(f"the run folder {rid} already holds {STATUS}: every attempt keeps its outcome (archive that folder first)")
    if msg := run_dir_problem(run, ROOT, inputs, cfg, out_root):
        fail(f"refusing the run folder: {msg}")
    run.mkdir(parents=True, exist_ok=True)
    head = {"run_id": rid, "commit": state["commit"], "dirty": state["dirty"], "config_sha256": cfg_sha,
            "delivered_configuration": configuration_name(cfg_sha) is not None, "configuration_name": configuration_name(cfg_sha),
            "command": "run"}
    if amendment:
        head["amendment"] = amendment
    log = None
    budget = Budget(cfg["budget"]["wall_time_cap_minutes"])
    signal_handlers(budget)
    completed = []
    try:
        FR.write_status(run, {**head, "status": "RUNNING", "passed": False, "started": now(), "pid": os.getpid(),
                              "note": "RUNNING without a live process means a failed run; SIGUSR1 to this pid stops it as INCOMPLETE"})
        log = FR.open_log(run)
        left = invalidate_outputs(run, cfg)
        if left:
            fail(f"previous outputs could not be removed: {', '.join(left)}")
        print(f"== NUAA session ablation run {rid}: configuration {cfg_sha[:12]} (delivered: {head['delivered_configuration']}, "
              f"{head['configuration_name']}), cap "
              f"{cfg['budget']['wall_time_cap_minutes']} min, {len(all_models(cfg))} trainings, pid {os.getpid()}", flush=True)
        t0 = budget.mark()
        inp = load_inputs(cfg, base)
        budget.lap("check-inputs", t0)
        print(f"   check-inputs PASS: {len(inp.cells_sha)} cell manifests, {len(inp.cache_index)} cache frames, {len(inp.extracted_keys)} "
              f"extracted frames, peak RSS {peak_rss_mb():.0f} MB", flush=True)
        store = FR.FrameStore(inp, inp.X_extracted, inp.extracted_keys)
        ap = rel_path(base, cfg["architecture"]["path"])
        if sha256_file(ap) != cfg["architecture"]["sha256"]:
            fail(f"architecture file {ap.name}: SHA-256 changed just before loading")
        arch = (ARCH_LOADER or FR.default_arch_loader)(ap)
        cards, thresholds, scores, stopped = {}, {}, {}, None
        for arm, fold, seed in training_order(cfg["schema_expected"]["k"], cfg["train_seeds"]):
            name = model_name(arm, fold, seed)
            try:
                card, thr, rows = train_one(cfg, run, store, inp, arm, fold, seed, budget, arch)
            except CapExceeded as e:
                removed = discard_model(run, name)
                stopped = {"model": name, "step": e.step, "cause": e.cause, "elapsed_minutes": round(e.minutes, 1), "removed_files": removed}
                print(f"== {'CAP' if e.cause == 'cap' else 'STOP'}: {e}; {name} interrupted, its files removed ({len(removed)}); no further "
                      "training", flush=True)
                break
            cards[(arm, fold, seed)], thresholds[(arm, fold, seed)], scores[(arm, fold, seed)] = card, thr, rows
            completed.append(name)
        missing = [n for n in all_models(cfg) if n not in completed]
        over_cap = budget.elapsed() > budget.cap           # misurato prima delle scritture finali, qualunque sia la causa dell'arresto
        t0 = budget.mark()
        agg, subject_rows = aggregate(cfg, scores, cards, thresholds, inp.reference)
        budget.lap("aggregate", t0)
        status = "PASS" if not missing else "INCOMPLETE"
        record = {**head, "status": status, "finished": now(), "wall_seconds": round(budget.elapsed(), 1), "timings": budget.timings,
                  "trainings": len(all_models(cfg)), "training_order": all_models(cfg), "completed": completed, "missing": missing,
                  "stopped": stopped, "cap_minutes": cfg["budget"]["wall_time_cap_minutes"],
                  "final_writes_after_cap": over_cap, "checks": inp.checks, "environment": inp.environment,
                  "code_sha256": inp.code, "cells_manifests_sha256": inp.cells_sha, "fold_manifests_sha256": inp.fold_sha,
                  "train_seeds": cfg["train_seeds"], "bootstrap": {**BOOT, "recipe": BOOT_RECIPE}}
        agg_doc = {**head, "status": status, "finished": record["finished"], "completed": len(completed), "trainings": record["trainings"],
                   "missing": missing, "bootstrap": record["bootstrap"], "decision": FR.DECISION_RULE, "strata": list(STRATA), **agg}
        public = {"aggregate.json": json.dumps(agg_doc, indent=1) + "\n", "run_record.json": json.dumps(record, indent=1) + "\n"}
        public["aggregate.md"] = aggregate_md(cfg, agg, record, rid)
        for pname, text in public.items():
            if probs := FR.public_problems(text, inp, pname.endswith(".md")):
                fail(f"public file {pname} refused: " + "; ".join(probs))
        public["SHA256SUMS"] = FR.sums_text(public)
        rdir = run / RESTRICTED
        mw.write_all(rdir, {"per_subject.csv": csv_text(SUBJECT_COLUMNS, subject_rows),
                            "public_SHA256SUMS.sha256": sha256_bytes(public["SHA256SUMS"].encode("utf-8")) + "  public/SHA256SUMS\n"},
                     newline="")
        names = [f"{n}_{x}" for n in completed for x in PER_MODEL] + ["per_subject.csv", "public_SHA256SUMS.sha256"]
        missing_files = [n for n in names if not (rdir / n).is_file()]
        if missing_files:
            fail(f"restricted outputs missing: {missing_files}")
        extra = sorted(e for e in os.listdir(rdir) if e not in names and e != "SHA256SUMS")
        if extra:
            fail(f"restricted folder holds files that are not outputs of complete models: {extra}")
        mw.write_all(rdir, {"SHA256SUMS": "".join(f"{sha256_file(rdir / n)}  {n}\n" for n in sorted(names))})
        mw.write_all(run / PUBLIC, public)
        for key, v in agg["primary_verdicts"].items():
            print(f"primary {METRIC_LABEL[key]} (A* - B, stratum 03): {verdict_text(v)} ({v['reason']})")
        print(f"run folder: {run.name} (public/ and restricted/), {len(completed)}/{len(all_models(cfg))} trainings, wall {record['wall_seconds']:.0f} s, "
              f"aggregate {budget.timings['aggregate']['wall_clock_seconds']:.1f} s, peak RSS {peak_rss_mb():.0f} MB")
        print(f"RESULT: {status} (run)")
        FR.close_log(log)
        log = None
        FR.write_status(run, {**head, "status": status, "passed": status == "PASS", "finished": record["finished"],
                              "wall_seconds": record["wall_seconds"], "completed": completed, "missing": missing, "stopped": stopped,
                              "final_writes_after_cap": over_cap, "public_sums_sha256": sha256_bytes(public["SHA256SUMS"].encode("utf-8")),
                              "restricted_sums_sha256": sha256_bytes((rdir / "SHA256SUMS").read_bytes()),
                              "run_log_sha256": sha256_file(run / LOG)})
    except BaseException as e:
        reason = str(e) if isinstance(e, Failure) else f"unexpected error {type(e).__name__}: {e}"
        kept, kept_sums = list(completed), None
        if kept:
            try:
                kept_sums = keep_complete_models(run, cfg, kept)
            except BaseException as e2:  # noqa: BLE001  (la conservazione non riesce: tutto tolto, come nel run dei fold)
                print(f"ERROR: the outputs of the complete models cannot be kept ({e2}): every output removed", file=sys.stderr, flush=True)
                kept = []
        left = [] if kept else invalidate_outputs(run, cfg)
        outputs = FAIL_KEPT_NOTE.format(n=len(kept)) if kept else ("removed" if not left else f"left behind: {', '.join(left)}")
        tail = f"ERROR: {reason}\nFAIL outputs: {outputs}\nRESULT: FAIL (run)\n"
        if log is not None:
            log[0].write(tail)
            FR.close_log(log)
        elif (run / LOG).is_file() and not (run / LOG).is_symlink():      # errore dopo la chiusura del log (status finale)
            with open(run / LOG, "a", encoding="utf-8") as fh:
                fh.write(tail)
        try:
            FR.write_status(run, {**head, "status": "FAIL", "passed": False, "reason": reason, "wall_seconds": round(budget.elapsed(), 1),
                                  "outputs": outputs, "kept_complete_models": kept,
                                  "restricted_sums_sha256": kept_sums or "none",
                                  "run_log_sha256": sha256_file(run / LOG) if (run / LOG).is_file() else "none"})
        except Exception:  # noqa: BLE001
            print("ERROR: status.json could not be written", file=sys.stderr)
        raise
    return 0 if status == "PASS" else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="NUAA session-03 ablation: LivenessNet arms B, A* and B', 4 folds x 3 seeds on the CPU")
    ap.add_argument("cmd", choices=["check-inputs", "run"])
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="configuration JSON (relative to the repository root)")
    ap.add_argument("--input-root", default=None, help="base folder of the input paths of the configuration (default: repository root)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="folder that receives <run-id>/ (default results/sessionablationrun; another folder for tests and dry runs only)")
    ap.add_argument("--date", default=None, help="YYYYMMDD of the run id (default: today)")
    ap.add_argument("--i-am-authorized", action="store_true", help="required by run: training and inference on the restricted NUAA data")
    a = ap.parse_args(argv)
    try:
        return cmd_check_inputs(a) if a.cmd == "check-inputs" else cmd_run(a)
    except Failure as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(f"RESULT: FAIL ({a.cmd})")
        return 1
    except KeyboardInterrupt:
        print(f"RESULT: FAIL ({a.cmd}, interrupted)")
        return 130
    except SystemExit as e:                     # fail() di m2_writers: già stampato
        print(f"RESULT: FAIL ({a.cmd})")
        return e.code if isinstance(e.code, int) and e.code else 1
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: unexpected error {type(e).__name__}: {e}", file=sys.stderr)
        print(f"RESULT: FAIL ({a.cmd})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
