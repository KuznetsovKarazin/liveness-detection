"""
Confronto prima/dopo sul test NUAA della ristima delle statistiche delle BatchNormalization della PR #27 (incarico del
coordinatore sulla scheda C2, 16/11/2026): gli stessi 36 checkpoint del run 20261106-SESSABLRUN-d999458 (pazienza 4),
nessun training, lo stimatore della PR #27 invariato (scripts/nuaa_bn_recalibration.py importato, SHA-256 fissato nel
codice e nella configurazione), soglia fissa 0,5. Estensione diagnostica post hoc su un test già analizzato, non una
validazione indipendente.

Fasi, in un solo processo e in quest'ordine (ognuna con il tempo impiegato):
  0. check-inputs   configurazione a insieme chiuso (SHA-256 della nota: null rifiutato dal run); configurazione della
                    PR #27 con lo SHA-256 fissato, validata dal suo runner, e sezioni condivise (run genitore,
                    architettura, BatchNormalization, stimatore, valutazione, criterio, gruppi, sette tolleranze) uguali
                    valore per valore, tranne le chiavi solo descrittive di SHARED_REWRITTEN (parent_run.used,
                    evaluation.probability), riscritte con il testo vero per questo run; codice importato uguale agli SHA-256 fissati e a code_sha256 del run_record del run
                    BNRECAL; ingressi del runner genitore come li carica nuaa_bn_recalibration più lo schema dei fold
                    (FR.load_fold_manifests, SR.check_schema_config, SR.check_cells, SR.check_sources anche sul test);
                    run genitore ancorato da BR.load_parent; ogni <modello>_test_scores.csv del genitore uguale alla sua
                    riga del SHA256SUMS riservato (solo SHA-256, contenuto non decodificato); run BNRECAL PASS con i due
                    SHA256SUMS fissati, results.csv pubblico (gruppi uguali, modelli con un indicatore del dev peggiorato
                    uguali alla configurazione), _scores.csv e _bn_statistics.json di ogni modello uguali al SHA256SUMS
                    riservato; frame del test per fold uguali a comparison.test_frames_per_fold (nessun fold vuoto).
                    Nulla scritto, TensorFlow non importato.
  1. baseline       per ognuno dei 36 modelli, PRIMA di qualunque ristima: SHA-256 verificato subito prima di ogni
                    caricamento; (a) dev: i controlli della PR #27 (BR.baseline_checks, stesse tolleranze); (b) test con
                    le statistiche originali: i frame del test del fold assemblati come nel run genitore
                    (FR.assemble sulle righe di fold<k>_test.csv, in ordine di chiave), probabilità d'attacco del
                    valutatore di BR contro il <modello>_test_scores.csv del genitore (stesse chiavi, etichette, soggetti,
                    sessioni, SHA-256 dei file, stesso numero e stesso ordine: nessun riallineamento; scarto massimo e
                    decisioni discordanti a 0,5 con le tolleranze della configurazione; softmax dei logit). Se un solo
                    controllo di un solo modello non passa: public/baseline.csv, status BASELINE_FAILED (codice 3), nulla
                    calcolato dopo. Nessuna metrica del test scritta o nel log in questa fase: BR.evaluate calcola in
                    memoria anche accuratezza e AUROC del test, e quel record è scartato (restano solo probabilità e
                    logit); sul test si usa solo il valutatore di BR, lo scorer del genitore (FR.tf_scorer) solo sul dev.
  2. recalibration  solo con la baseline superata per tutti i 36, per tutti i 36 PRIMA di qualunque punteggio del test con
                    statistiche ristimate: ristima con BR.recalibrate su copia in memoria (BR.tensor_problems: solo i 10
                    tensori mobili cambiano; BR.consistency), poi i controlli che sia la stessa ristima della PR #27:
                    statistiche installate float32 contro quelle del _bn_statistics.json del run BNRECAL (tolleranza della
                    configurazione, scarto massimo e uguaglianza esatta riportati) e probabilità del dev dopo contro le
                    righe statistics = re-estimated del _scores.csv del run BNRECAL. Se un controllo non passa:
                    public/recalibration_check.csv, status RECALIBRATION_MISMATCH (codice 4), nessun punteggio del test con
                    statistiche ristimate calcolato o scritto. Le statistiche verificate restano in memoria.
  3a. scoring       solo con le fasi 1 e 2 superate per tutti i 36, per TUTTI i 36 prima di qualunque metrica del test:
                    copia ricostruita dalle statistiche verificate (di nuovo solo i 10 tensori mobili diversi), gli stessi
                    frame del test (riassemblati, stesse chiavi) valutati, controllo del softmax; probabilità e logit in
                    memoria, nessun file scritto, nel log solo i tempi. Poi la riverifica degli ingressi (SHA-256 di ogni
                    .keras, del SHA256SUMS riservato del genitore e dei due SHA256SUMS del run BNRECAL).
  3b. metrics       solo dopo 3a e la riverifica: metriche per modello × ambito (test, session_03) × statistiche
                    (original, re-estimated) con FR.stratum_metrics (APCER, BPCER, ACER alla soglia fissa 0,5 con la
                    regola del progetto, AUROC di Mann-Whitney; NA se una classe manca), e dagli score salvati del
                    genitore come controllo riportato; differenze dopo − prima, indicatori di peggioramento; per soggetto
                    nel riservato; file riservati per modello; righe delle metriche nel log.
  4. aggregate      riepiloghi appaiati per gruppo e per braccio con il bootstrap per soggetto del progetto
                    (paired_group_bootstrap: SR.boot_picks sui soggetti tenuti fuori uniti, SR.subject_stats,
                    SR.weighted_metrics, SR.interval) e la regola di lettura applicata meccanicamente; uscite.
Esiti: PASS (0), FAIL (1: ogni uscita tolta; restano status.json con la ragione e run.log con le righe stampate fino
all'errore, che contengono metriche del test solo se l'errore avviene nella fase 3b o nelle scritture finali), INCOMPLETE
(2: tetto o SIGUSR1 nelle fasi 1, 2 o 3a: sono pubblicati i controlli completati, baseline.csv e recalibration_check.csv,
con run_record e SHA256SUMS; nessuna metrica del test calcolata, nel log o scritta, nessun file per frame del test con
statistiche ristimate), BASELINE_FAILED (3), RECALIBRATION_MISMATCH (4). La fase 3b è tutto o niente: non consulta il
tetto né la richiesta di arresto (pochi secondi di NumPy); un SIGUSR1 arrivato durante la 3b non è consultato, la 3b e
le scritture finali si completano. Albero git non
pulito o cartella del run esistente: rifiuto prima di scrivere.

Uscite (results/bntest/YYYYMMDD-BNTEST-<commit>; --out-root diverso solo per i test):
  status.json, run.log
  public/      results.csv, baseline.csv, recalibration_check.csv, summary.csv, aggregate.json, aggregate.md,
               differences.png, run_record.json, SHA256SUMS (nessun identificativo di soggetto o di frame)
  restricted/  <braccio>_fold<k>_seed<s>_test_scores.csv (per frame, statistics original e re-estimated, con i logit; le
               colonne prediction_dev e threshold_dev riportano la soglia dev congelata del genitore come nei suoi file e
               non entrano in nessuna metrica di questo run), _bn_statistics.json (formato di BR.stats_json),
               per_subject.csv, run_record.json, public_SHA256SUMS.sha256, SHA256SUMS

  .venv/bin/python -I -B scripts/nuaa_bn_test.py check-inputs
  .venv/bin/python -I -B scripts/nuaa_bn_test.py run --i-am-authorized
  .venv/bin/python -I -B scripts/nuaa_bn_test.py figure --run-folder <cartella del run> --output <file .png nuovo>

Riuso per importazione, nessuna copia: nuaa_bn_recalibration (BR: configurazione della PR #27, load_parent,
load_folds_frames_rows, frames, load_model, check_architecture, recalibrate, snapshot_weights, moving_keys,
tensor_problems, consistency, change_summary, evaluate, baseline_checks, score_rows, stats_json, cell_text),
nuaa_session_ablation_run (SR: caricatori, Budget, segnali, score_rows, subject_stats, weighted_metrics, interval, BOOT),
nuaa_folds_run (FR: manifest, schema dei fold, cache, FrameStore/assemble, tf_scorer, probabilities, stratum_metrics,
rows_of_stratum, FIXED_THRESHOLD, DECISION_RULE, public_problems, stato e log), m2_writers e path_guard.
Unico elemento nuovo dichiarato: nel bootstrap la statistica di un riepilogo (media sui modelli della differenza appaiata
dopo − prima) e la regola che scarta un ricampionamento in cui la metrica di un modello qualsiasi non è definita; il
ricampionamento è quello del progetto, invariato (con un solo modello coincide con SR.paired_bootstrap).
Dipendenze: NumPy; TensorFlow 2.15 (fasi 1-3, solo CPU); matplotlib (solo la figura, backend Agg).
"""
import argparse
import gc
import importlib.metadata
import io
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import m2_writers as mw  # noqa: E402
import nuaa_bn_recalibration as BR  # noqa: E402
import nuaa_folds_run as FR  # noqa: E402
import nuaa_session_ablation_run as SR  # noqa: E402
import path_guard as pg  # noqa: E402

Failure, fail = FR.Failure, FR.fail
sha256_bytes, sha256_file, read_regular, rel_path, parse_csv, csv_text = (FR.sha256_bytes, FR.sha256_file, FR.read_regular,
                                                                          FR.rel_path, FR.parse_csv, FR.csv_text)
now, f4 = FR.now, FR.f4

DEFAULT_CONFIG = "delivery/bntest/nuaa_bn_test.json"
DEFAULT_OUT_ROOT = "results/bntest"
CONFIG_SCHEMA = "nuaa-bn-test-config/1"
RUN_ID_RE = re.compile(r"^\d{8}-BNTEST-[0-9a-f]{7,40}$")
# lo stimatore della PR #27: il file unito nel fork (research/nuaa-saturation-control/livenesslab/scripts/) e usato dal run
# BNRECAL; un file diverso è rifiutato anche se la configurazione lo dichiarasse
ESTIMATOR_FILE = "scripts/nuaa_bn_recalibration.py"
ESTIMATOR_SHA256 = "8c228ef400f5ef4337612477b23113fb6b8cac5364a5ff026e2c0ac8096734f5"
CODE_FILES = (ESTIMATOR_FILE,) + SR.CODE_FILES          # gli stessi file di BR.code_files(), cioè del run BNRECAL
TOP_KEYS = ("schema", "task", "authorisation", "declarations", "parent_run", "bnrecal_run", "code", "architecture", "batchnorm",
            "estimator", "evaluation", "recovery_criterion", "groups", "dev_worsened", "tolerances", "stop_rules", "comparison",
            "bootstrap", "reading_rule", "outputs", "budget", "environment", "note_path", "note_sha256", "test")
# sezioni che devono essere uguali a quelle della configurazione della PR #27 (validate dal suo runner)
SHARED = ("parent_run", "architecture", "batchnorm", "estimator", "evaluation", "recovery_criterion", "groups")
# chiavi solo descrittive delle sezioni condivise, false per questo run se copiate dalla PR #27: riscritte qui con il
# testo vero per questo run; ogni altra chiave delle sezioni condivise è uguale alla PR #27 per regola
SHARED_REWRITTEN = {"parent_run": ("used",), "evaluation": ("probability",)}
NEW_TOLERANCES = ("test_probability_max_abs_diff", "test_decisions_discordant_at_0_5", "recalibration_statistics_rel",
                  "recalibration_statistics_abs", "recalibration_dev_probability_max_abs_diff",
                  "recalibration_dev_decisions_discordant_at_0_5")
TOLERANCE_KEYS = BR.TOLERANCE_KEYS + NEW_TOLERANCES
COUNT_TOLERANCES = ("dev_decisions_discordant_at_0_5", "test_decisions_discordant_at_0_5", "recalibration_dev_decisions_discordant_at_0_5")
DECLARATION_KEYS = ("post_hoc", "threshold", "procedure", "cause", "worsening")
SCOPES = ("test", "session_03")
STRATUM = {"test": "all", "session_03": "03"}            # ambito -> strato di FR.rows_of_stratum
STATISTICS = ("original", "re-estimated")
METRICS = ("apcer_05", "bpcer_05", "acer_05", "auroc")
METRIC_TEXT = {"apcer_05": "APCER at the fixed threshold 0.5", "bpcer_05": "BPCER at the fixed threshold 0.5",
               "acer_05": "ACER at the fixed threshold 0.5", "auroc": "AUROC (threshold-free)"}
# indicatore di peggioramento per metrica: (nome, segno della differenza dopo − prima che peggiora)
WORSE = {"acer_05": ("acer_higher", 1), "apcer_05": ("apcer_higher", 1), "bpcer_05": ("bpcer_higher", 1), "auroc": ("auroc_lower", -1)}
FLAGS = tuple(WORSE[k][0] for k in ("acer_05", "apcer_05", "bpcer_05", "auroc"))
GROUP_SUMMARIES = ("ten", "control", "apart", "all")
PRIMARY = {"summary": "ten", "scope": "test", "metric": "acer_05"}
CASES = ("below", "contains", "above", "not_available")
READING_PLACEHOLDERS = ("{who}", "{scope}", "{metric}", "{difference}", "{low}", "{high}")
COUNT_PLACEHOLDERS = ("{lower}", "{equal}", "{higher}", "{n}")
BANNED = ("attributable", "a reading against", "fails", "penalise", "caused by", "because", "proves", "demonstrates")
STATUS_CODE = {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2, "BASELINE_FAILED": 3, "RECALIBRATION_MISMATCH": 4}
STATUS, LOG, PUBLIC, RESTRICTED = FR.STATUS, FR.LOG, "public", "restricted"
PUBLIC_OUTPUTS = ("results.csv", "baseline.csv", "recalibration_check.csv", "summary.csv", "aggregate.json", "aggregate.md",
                  "differences.png", "run_record.json", "SHA256SUMS")
PER_MODEL = ("test_scores.csv", "bn_statistics.json")
RESTRICTED_FIXED = ("per_subject.csv", "run_record.json", "public_SHA256SUMS.sha256", "SHA256SUMS")
TEST_SCORE_COLUMNS = BR.SCORE_COLUMNS
SUBJECT_COLUMNS = ("arm", "fold", "seed", "subject_id", "scope", "statistics", "n_bona_fide", "n_attack", "rejected_bona_fide_05",
                   "accepted_attacks_05", "apcer_05", "bpcer_05", "acer_05", "auroc")
EXTRA_PROTECTED = BR.EXTRA_PROTECTED + ("results/bnrecal",)
FAIL_DOC = FR.FAIL_DOC
FIGURE = "differences.png"
TEST_READ = ("this run reads the test: in phase 0 it opens and validates the fold schema (fold<k>_fitting.csv, fold<k>_dev.csv and "
             "fold<k>_test.csv, as the parent run did) and hashes every <model>_test_scores.csv of the parent run against its "
             "restricted SHA256SUMS without parsing it; in phase 1 it assembles the test frames of each fold (every frame of the "
             "held-out subjects, from the cache and the extracted frames, in key order, as the parent run did), scores them with "
             "each checkpoint and its original statistics with the evaluator of nuaa_bn_recalibration (the parent's own scorer is "
             "used only on the dev) and parses the parent's <model>_test_scores.csv to compare the per-frame attack probabilities; "
             "the evaluator also computes a test accuracy and AUROC in memory, which are discarded: no test metric is written or "
             "logged in phase 1; only if phases 1 and 2 pass for all 36 models, phase 3a reassembles the same test frames and "
             "scores them with the re-estimated statistics for all 36, the 36 .keras files, the parent's restricted SHA256SUMS and the two "
             "SHA256SUMS of the BNRECAL run are re-hashed, and only then phase 3b computes "
             "the test metrics (APCER, BPCER and ACER at the fixed threshold 0.5, AUROC) before and after and, as a reported "
             "check, from the parent's stored scores; the M2 manifests official_test and cnn_test are read to index the cache; "
             "the public aggregate of the parent run, which holds test metrics already published, is read only as bytes for its "
             "SHA-256 and never parsed; no public file of the folds run is opened")

# punti di iniezione (solo i test): stato git, scorer del genitore (il caricamento dei modelli è BR.LOAD_MODEL)
REPO_STATE = None       # (root) -> {"commit", "dirty"} (default: SR.repo_state)
MAKE_SCORER = None      # (weights_path, sha256) -> funzione(X uint8) -> probabilità N x 2 (default: FR.tf_scorer)


# ============================================================================ configurazione

def _text(v, what: str) -> None:
    if not isinstance(v, str) or not v.strip():
        fail(f"configuration: {what} must be a non-empty text")


def _keys(d, keys, what: str) -> None:
    if not isinstance(d, dict) or sorted(d) != sorted(keys):
        fail(f"configuration: {what} needs exactly {', '.join(keys)}")


def _number(v, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0:
        fail(f"configuration: {what} must be a finite positive number")
    return float(v)


def load_config(p: Path, base: Path) -> tuple:
    """(configurazione, SHA-256 dei byte letti, configurazione della PR #27). Insiemi chiusi e tipi controllati qui, prima
    di leggere qualunque dato; le sezioni condivise con la PR #27 sono validate dal suo runner (BR.load_config) e devono
    essere uguali a quelle della sua configurazione, letta con lo SHA-256 fissato."""
    data = read_regular(p, "configuration")
    cfg = SR.read_json(data, f"configuration {p.name}")
    if cfg.get("schema") != CONFIG_SCHEMA:
        fail(f"configuration {p.name}: schema must be {CONFIG_SCHEMA!r}")
    if sorted(cfg) != sorted(TOP_KEYS):
        fail(f"configuration: fields must be exactly {', '.join(TOP_KEYS)} (missing {sorted(set(TOP_KEYS) - set(cfg))}, "
             f"unknown {sorted(set(cfg) - set(TOP_KEYS))})")
    for k in ("task", "authorisation", "note_path"):
        _text(cfg[k], k)
    if cfg["test"] != TEST_READ:
        fail("configuration: test must state exactly what is read of the test (TEST_READ of the script)")
    if cfg["note_sha256"] is not None and not SR._hex(cfg["note_sha256"]):
        fail("configuration: note_sha256 must be null or a SHA-256")
    de = cfg["declarations"]
    _keys(de, DECLARATION_KEYS, "declarations")
    for k in DECLARATION_KEYS:
        _text(de[k], f"declarations.{k}")
    if "post hoc" not in de["post_hoc"] or "not an independent validation" not in de["post_hoc"]:
        fail("configuration: declarations.post_hoc must say post hoc and not an independent validation")
    if "0.5" not in de["threshold"]:
        fail("configuration: declarations.threshold must state the fixed threshold 0.5")
    br = cfg["bnrecal_run"]
    _keys(br, ["config_path", "config_sha256", "folder", "models", "public_sha256sums_sha256", "restricted_sha256sums_sha256", "run_id",
               "used"], "bnrecal_run")
    for k in ("config_path", "folder", "used"):
        _text(br[k], f"bnrecal_run.{k}")
    if not isinstance(br["run_id"], str) or not BR.RUN_ID_RE.match(br["run_id"]):
        fail("configuration: bnrecal_run.run_id must be a BNRECAL run id")
    if not all(SR._hex(br[k]) for k in ("config_sha256", "public_sha256sums_sha256", "restricted_sha256sums_sha256")):
        fail("configuration: bnrecal_run SHA-256 values malformed")
    co = cfg["code"]
    _keys(co, list(CODE_FILES), "code")
    if not all(SR._hex(v) for v in co.values()):
        fail("configuration: code SHA-256 values malformed")
    if co[ESTIMATOR_FILE] != ESTIMATOR_SHA256:
        fail(f"configuration: code pins {ESTIMATOR_FILE} to {co[ESTIMATOR_FILE][:12]}..., not the estimator of PR #27 "
             f"({ESTIMATOR_SHA256[:12]}...)")
    tol = cfg["tolerances"]
    _keys(tol, list(TOLERANCE_KEYS), "tolerances")
    for k in NEW_TOLERANCES:
        _keys(tol[k], ["source", "value"], f"tolerances.{k}")
        _text(tol[k]["source"], f"tolerances.{k}.source")
        if k in COUNT_TOLERANCES:
            if not FR.is_count(tol[k]["value"]):
                fail(f"configuration: tolerances.{k}.value must be a non-negative integer")
        else:
            _number(tol[k]["value"], f"tolerances.{k}.value")
    sr_ = cfg["stop_rules"]
    _keys(sr_, ["baseline", "fail", "recalibration"], "stop_rules")
    if not isinstance(sr_["fail"], str) or "FAIL" not in sr_["fail"] or "status.json" not in sr_["fail"] or "run.log" not in sr_["fail"]:
        fail("configuration: stop_rules.fail must state what status.json and run.log hold after a FAIL")
    if not isinstance(sr_["baseline"], str) or "BASELINE_FAILED" not in sr_["baseline"]:
        fail("configuration: stop_rules.baseline must declare the BASELINE_FAILED outcome")
    if not isinstance(sr_["recalibration"], str) or "RECALIBRATION_MISMATCH" not in sr_["recalibration"]:
        fail("configuration: stop_rules.recalibration must declare the RECALIBRATION_MISMATCH outcome")
    dw = cfg["dev_worsened"]
    _keys(dw, ["definition", "expected", "source"], "dev_worsened")
    _text(dw["definition"], "dev_worsened.definition")
    _text(dw["source"], "dev_worsened.source")
    BR._names(dw["expected"], "dev_worsened.expected")
    cp = cfg["comparison"]
    _keys(cp, ["auroc", "decision", "frames", "metrics", "per_subject", "scopes", "statistics", "test_frames_per_fold", "threshold",
               "undefined", "worsening"], "comparison")
    if not isinstance(cp["test_frames_per_fold"], list) or not cp["test_frames_per_fold"] \
            or not all(FR.is_count(x) and x > 0 for x in cp["test_frames_per_fold"]):
        fail("configuration: comparison.test_frames_per_fold must list a positive number of test frames per fold")
    if cp["scopes"] != list(SCOPES) or cp["statistics"] != list(STATISTICS) or cp["metrics"] != list(METRICS):
        fail(f"configuration: comparison must have scopes {list(SCOPES)}, statistics {list(STATISTICS)} and metrics {list(METRICS)}")
    if cp["threshold"] != FR.FIXED_THRESHOLD or isinstance(cp["threshold"], bool) or cp["decision"] != FR.DECISION_RULE:
        fail(f"configuration: comparison.threshold must be {FR.FIXED_THRESHOLD} and comparison.decision {FR.DECISION_RULE!r} "
             "(the rule of the project)")
    for k in ("auroc", "frames", "per_subject", "undefined", "worsening"):
        _text(cp[k], f"comparison.{k}")
    bo = cfg["bootstrap"]
    _keys(bo, ["combination", "min_subjects", "percentiles", "recipe", "resamples", "seed", "summaries"], "bootstrap")
    if {k: bo[k] for k in SR.BOOT} != SR.BOOT:
        fail(f"configuration: bootstrap resamples, seed, percentiles and min_subjects must be those of the project ({SR.BOOT})")
    if bo["summaries"] != {"groups": list(GROUP_SUMMARIES), "arms": list(SR.ARMS)}:
        fail("configuration: bootstrap.summaries must list the groups ten, control, apart, all and the three arms")
    _text(bo["recipe"], "bootstrap.recipe")
    _text(bo["combination"], "bootstrap.combination")
    rr = cfg["reading_rule"]
    _keys(rr, ["count", "limits", "primary", "secondary", *CASES], "reading_rule")
    for k in rr:
        _text(rr[k], f"reading_rule.{k}")
    for k in CASES[:3]:
        if any(ph not in rr[k] for ph in READING_PLACEHOLDERS):
            fail(f"configuration: reading_rule.{k} must contain the placeholders {', '.join(READING_PLACEHOLDERS)}")
    if any(ph not in rr["not_available"] for ph in ("{who}", "{scope}", "{metric}", "{reason}")):
        fail("configuration: reading_rule.not_available must contain the placeholders {who}, {scope}, {metric} and {reason}")
    if any(ph not in rr["count"] for ph in COUNT_PLACEHOLDERS):
        fail(f"configuration: reading_rule.count must contain the placeholders {', '.join(COUNT_PLACEHOLDERS)}")
    banned = [w for w in BANNED if w in json.dumps([rr, de]).lower()]
    if banned:
        fail(f"configuration: the reading rule and the declarations must not use causal or conclusive wording ({', '.join(banned)})")
    ou = cfg["outputs"]
    _keys(ou, ["public", "restricted"], "outputs")
    if ou["public"] != list(PUBLIC_OUTPUTS):
        fail(f"configuration: outputs.public must be {list(PUBLIC_OUTPUTS)}")
    if ou["restricted"] != [f"<model>_{x}" for x in PER_MODEL] + list(RESTRICTED_FIXED):
        fail("configuration: outputs.restricted must list the per-model files and the fixed restricted files of this script")
    bu = cfg["budget"]
    _keys(bu, ["estimate", "estimate_minutes", "on_cap_exceeded", "upper_estimate_minutes", "voluntary_stop", "wall_time_cap_minutes"], "budget")
    if not FR.is_count(bu["wall_time_cap_minutes"]) or not 1 <= bu["wall_time_cap_minutes"] <= 240:
        fail("configuration: budget.wall_time_cap_minutes must be a positive integer, at most 240 (the coordinator's limit)")
    _number(bu["estimate_minutes"], "budget.estimate_minutes")
    _number(bu["upper_estimate_minutes"], "budget.upper_estimate_minutes")
    if not bu["estimate_minutes"] <= bu["upper_estimate_minutes"] < bu["wall_time_cap_minutes"]:
        fail("configuration: budget needs estimate_minutes <= upper_estimate_minutes < wall_time_cap_minutes")
    for k in ("estimate", "on_cap_exceeded", "voluntary_stop"):
        _text(bu[k], f"budget.{k}")
    if "INCOMPLETE" not in bu["on_cap_exceeded"] or "SIGUSR1" not in bu["voluntary_stop"]:
        fail("configuration: budget must declare the INCOMPLETE outcome and the SIGUSR1 stop")
    _keys(cfg["environment"], ["matplotlib", "tensorflow"], "environment")
    _text(cfg["environment"]["tensorflow"], "environment.tensorflow")
    _text(cfg["environment"]["matplotlib"], "environment.matplotlib")
    # la configurazione della PR #27: SHA-256 fissato, validata dal suo runner, sezioni condivise uguali
    bp = rel_path(base, br["config_path"])
    bb = read_regular(bp, "PR #27 configuration")
    if sha256_bytes(bb) != br["config_sha256"]:
        fail(f"PR #27 configuration {bp.name}: SHA-256 {sha256_bytes(bb)} differs from bnrecal_run.config_sha256")
    bn_cfg, _ = BR.load_config(bp)
    diff = [k for k in SHARED if shared_part(cfg, k) != shared_part(bn_cfg, k) or not isinstance(cfg[k], dict)
            or sorted(cfg[k]) != sorted(bn_cfg[k])]
    if diff:
        fail(f"configuration: {', '.join(diff)} differ from the PR #27 configuration (the estimator and every shared setting must be unchanged)")
    for s, keys in SHARED_REWRITTEN.items():
        for k in keys:
            _text(cfg[s].get(k), f"{s}.{k}")
    tdiff = [k for k in BR.TOLERANCE_KEYS if tol[k] != bn_cfg["tolerances"][k]]
    if tdiff:
        fail(f"configuration: tolerances {', '.join(tdiff)} differ from the PR #27 configuration")
    if bn_cfg["environment"]["tensorflow"] != cfg["environment"]["tensorflow"]:
        fail("configuration: environment.tensorflow differs from the PR #27 configuration")
    return cfg, sha256_bytes(data), bn_cfg


def shared_part(cfg: dict, section: str) -> dict:
    """Una sezione condivisa senza le chiavi descrittive riscritte (SHARED_REWRITTEN)."""
    return {k: v for k, v in cfg[section].items() if k not in SHARED_REWRITTEN.get(section, ())}


def tol(cfg: dict, key: str):
    return cfg["tolerances"][key]["value"]


# ============================================================================ ingressi (fase 0)

class Inputs(BR.Inputs):
    """Gli ingressi di BR (run genitore, celle, cache, frame estratti) più lo schema dei fold, i punteggi del test del
    genitore (percorso e SHA-256) e il run BNRECAL (statistiche installate e punteggi del dev dopo)."""

    def __init__(self):
        super().__init__()
        self.bn_cfg = None
        self.test_files = {}      # nome -> (percorso, SHA-256)
        self.bnrecal = {}         # nome -> {"dev_after": array, "stats": {strato: (media32, varianza32)}}
        self.bnrecal_sums = {}
        self.dev_worsened = []


def check_code(cfg: dict) -> dict:
    """Ogni file importato uguale allo SHA-256 della configurazione; lo stimatore anche alla costante del codice."""
    got = {f: sha256_bytes(read_regular(ROOT / f, "code file")) for f in CODE_FILES}
    bad = [f for f in CODE_FILES if got[f] != cfg["code"][f]]
    if bad:
        fail(f"code files {bad} differ from the SHA-256 pinned in the configuration")
    if got[ESTIMATOR_FILE] != ESTIMATOR_SHA256:
        fail(f"{ESTIMATOR_FILE}: SHA-256 {got[ESTIMATOR_FILE]} is not the estimator of PR #27")
    return got


def load_parent_test_files(cfg: dict, base: Path, inp: Inputs) -> None:
    """I <modello>_test_scores.csv del genitore: ognuno elencato nel SHA256SUMS riservato ancorato e con lo stesso SHA-256.
    Solo i byte per l'impronta: il contenuto è decodificato nella fase 1."""
    pr = cfg["parent_run"]
    run = rel_path(base, pr["folder"])
    b = read_regular(run / RESTRICTED / "SHA256SUMS", "parent restricted SHA256SUMS")
    if sha256_bytes(b) != pr["restricted_sha256sums_sha256"]:
        fail("parent restricted SHA256SUMS: SHA-256 differs from the configuration")
    sums = FR.read_sums(b, "parent restricted SHA256SUMS")
    for name in inp.order:
        fn = f"{name}_test_scores.csv"
        if fn not in sums:
            fail(f"parent restricted SHA256SUMS: {fn} not listed")
        p = run / RESTRICTED / fn
        h = sha256_file(FR.regular_path(p, f"parent {fn}"))
        if h != sums[fn]:
            fail(f"parent {fn}: SHA-256 {h} differs from the parent restricted SHA256SUMS")
        inp.test_files[name] = (p, h)
    inp.checks["parent_run"] = {**inp.checks["parent_run"], "not_opened": "none: the <model>_test_scores.csv files are hashed in phase 0 "
                                "and parsed in phase 1",
                                "test_scores": "every <model>_test_scores.csv equal to its line of the restricted SHA256SUMS"}


def load_bnrecal(cfg: dict, base: Path, inp: Inputs) -> None:
    """Il run BNRECAL della PR #27: status PASS con run id, configurazione e SHA256SUMS fissati; results.csv pubblico
    (gruppi uguali a quelli derivati qui, modelli con un indicatore del dev peggiorato uguali alla configurazione),
    run_record.json pubblico (code_sha256 uguale al codice fissato qui); per modello _scores.csv (righe del dev con
    statistics = re-estimated) e _bn_statistics.json (statistiche installate float32) uguali al SHA256SUMS riservato."""
    br, bn_cfg = cfg["bnrecal_run"], inp.bn_cfg
    run = rel_path(base, br["folder"])
    st = SR.read_json(read_regular(run / STATUS, "BNRECAL status.json"), "BNRECAL status.json")
    if st.get("status") != "PASS" or st.get("run_id") != br["run_id"] or st.get("config_sha256") != br["config_sha256"] \
            or st.get("public_sums_sha256") != br["public_sha256sums_sha256"] or st.get("restricted_sums_sha256") != br["restricted_sha256sums_sha256"]:
        fail(f"BNRECAL run {br['run_id']}: status.json is not PASS with the run id, configuration and SHA256SUMS of this configuration")
    if st.get("completed") != inp.order or st.get("baseline_failed") != [] or len(inp.order) != br["models"]:
        fail(f"BNRECAL run {br['run_id']}: the completed models differ from the parent's training order")
    sums = {}
    for sub, key in ((PUBLIC, "public_sha256sums_sha256"), (RESTRICTED, "restricted_sha256sums_sha256")):
        b = read_regular(run / sub / "SHA256SUMS", f"BNRECAL {sub} SHA256SUMS")
        if sha256_bytes(b) != br[key]:
            fail(f"BNRECAL run {br['run_id']}: {sub} SHA256SUMS has SHA-256 {sha256_bytes(b)}, expected {br[key]}")
        sums[sub] = FR.read_sums(b, f"BNRECAL {sub} SHA256SUMS")

    def read_listed(sub: str, fn: str) -> bytes:
        if fn not in sums[sub]:
            fail(f"BNRECAL {sub} SHA256SUMS: {fn} not listed")
        data = read_regular(run / sub / fn, f"BNRECAL {fn}")
        if sha256_bytes(data) != sums[sub][fn]:
            fail(f"BNRECAL {fn}: SHA-256 differs from its SHA256SUMS")
        return data
    if read_listed(RESTRICTED, "public_SHA256SUMS.sha256").decode("utf-8", "replace") != f"{br['public_sha256sums_sha256']}  public/SHA256SUMS\n":
        fail("BNRECAL restricted public_SHA256SUMS.sha256 does not name the public SHA256SUMS of this configuration")
    rec = SR.read_json(read_listed(PUBLIC, "run_record.json"), "BNRECAL run_record.json")
    code = rec.get("code_sha256")
    if not isinstance(code, dict) or any(code.get(f) != cfg["code"][f] for f in CODE_FILES) \
            or code.get("architecture") != cfg["architecture"]["sha256"]:
        fail("BNRECAL run_record.json: code_sha256 differs from the code pinned in this configuration (the code of PR #27 must be unchanged)")
    rows = parse_csv(read_listed(PUBLIC, "results.csv"), BR.result_columns(bn_cfg), "BNRECAL results.csv")
    by = {SR.model_name(r["arm"], r["fold"], r["seed"]): r for r in rows}
    if sorted(by) != sorted(inp.order) or len(rows) != len(inp.order):
        fail("BNRECAL results.csv: the models differ from the parent's 36 models")
    if any(by[n]["group"] != inp.groups[n] for n in inp.order):
        fail("BNRECAL results.csv: a group differs from the group derived from the parent cards")
    flags = [f for f in BR.FLAGS if f != "gained_criterion"]
    if any(by[n][f] not in ("0", "1") for n in inp.order for f in flags):
        fail("BNRECAL results.csv: a worsening flag is not 0 or 1")
    worsened = [n for n in inp.order if any(by[n][f] == "1" for f in flags)]
    if sorted(worsened) != sorted(cfg["dev_worsened"]["expected"]):
        fail(f"BNRECAL results.csv: the models with a worsened dev indicator {worsened} differ from the configuration")
    inp.dev_worsened = worsened
    bn = cfg["batchnorm"]["layers"]
    for name in inp.order:
        m = inp.parent[name]
        srows = parse_csv(read_listed(RESTRICTED, f"{name}_scores.csv"), BR.SCORE_COLUMNS, f"BNRECAL {name}_scores.csv")
        dev = [r for r in srows if r["role"] == "dev" and r["statistics"] == BR.STATISTICS[1]]
        want = sorted(r["key"] for r in BR.role_rows(inp, "dev", m["arm"], m["fold"]))
        if [r["sample_id"] for r in dev] != want or any((r["arm"], r["fold"], r["train_seed"], r["checkpoint_sha256"]) !=
                                                        (m["arm"], str(m["fold"]), str(m["seed"]), m["weights_sha256"]) for r in dev):
            fail(f"BNRECAL {name}_scores.csv: the re-estimated dev rows are not the dev of this model in key order")
        try:
            p = np.array([float(r["score_attack"]) for r in dev], np.float64)
        except ValueError:
            fail(f"BNRECAL {name}_scores.csv: score_attack not numeric")
        if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
            fail(f"BNRECAL {name}_scores.csv: score_attack outside [0, 1]")
        js = SR.read_json(read_listed(RESTRICTED, f"{name}_bn_statistics.json"), f"BNRECAL {name}_bn_statistics.json")
        if js.get("weights_sha256") != m["weights_sha256"] or js.get("estimator") != cfg["estimator"] or sorted(js.get("layers", {})) != sorted(bn):
            fail(f"BNRECAL {name}_bn_statistics.json: checkpoint, estimator or layers differ")
        stats = {}
        for n in bn:
            try:
                a = np.array(js["layers"][n]["installed_moving_mean_float32"], np.float32)
                v = np.array(js["layers"][n]["installed_moving_variance_float32"], np.float32)
            except (KeyError, TypeError, ValueError):
                fail(f"BNRECAL {name}_bn_statistics.json: installed statistics of {n} missing or not numeric")
            if a.ndim != 1 or a.shape != v.shape or not (np.all(np.isfinite(a)) and np.all(np.isfinite(v))):
                fail(f"BNRECAL {name}_bn_statistics.json: installed statistics of {n} malformed or not finite")
            stats[n] = (a, v)
        inp.bnrecal[name] = {"dev_after": p, "stats": stats}
    inp.bnrecal_sums = {PUBLIC: br["public_sha256sums_sha256"], RESTRICTED: br["restricted_sha256sums_sha256"]}
    inp.checks["bnrecal_run"] = {"run_id": br["run_id"], "config_sha256": br["config_sha256"],
                                 "public_sha256sums_sha256": br["public_sha256sums_sha256"],
                                 "restricted_sha256sums_sha256": br["restricted_sha256sums_sha256"],
                                 "files": "results.csv and run_record.json (public); per model _scores.csv and _bn_statistics.json "
                                          "(restricted), each equal to its SHA256SUMS line",
                                 "code": "code_sha256 of its run_record.json equal to the pinned code", "dev_worsened": len(worsened)}


def load_inputs(cfg: dict, bn_cfg: dict, base: Path, need_note: bool = True) -> Inputs:
    """Fase 0. Nessun modello caricato, TensorFlow non importato, nulla scritto."""
    refused = [v for v in FR.ENV_REFUSED if os.environ.get(v, "").strip()]
    if refused:
        fail(f"environment variables {refused} are set: they would redirect the reference code (unset them)")
    inp = Inputs()
    inp.bn_cfg = bn_cfg
    if cfg["note_sha256"] is None:
        if need_note:
            fail("configuration: note_sha256 is null: the NOTE.md must be frozen and its SHA-256 written before the run")
    else:
        h = sha256_bytes(read_regular(rel_path(ROOT, cfg["note_path"]), "note"))
        if h != cfg["note_sha256"]:
            fail(f"note {Path(cfg['note_path']).name}: SHA-256 {h} differs from note_sha256 of the configuration")
        inp.note_sha = h
    code = check_code(cfg)
    pr = cfg["parent_run"]
    cp = rel_path(base, pr["config_path"])
    if sha256_bytes(read_regular(cp, "parent configuration")) != pr["config_sha256"]:
        fail(f"parent configuration {cp.name}: SHA-256 differs from the configuration ({pr['config_sha256'][:12]}...)")
    pcfg, _ = SR.load_config(cp)
    v = SR.fr_view(pcfg)
    FR.load_m2_manifests(v, base, inp)
    FR.load_fold_manifests(v, base, inp)                 # lo schema dei fold, test compreso, come nel run genitore
    SR.check_schema_config(pcfg, base, inp)
    FR.load_cache(v, base, inp)
    SR.load_cells_run(pcfg, base, inp)
    SR.check_cells(pcfg, inp)
    BR.load_folds_frames_rows(pcfg, base, inp)           # del run dei fold nessun file pubblico (come la PR #27)
    SR.load_frames(pcfg, base, inp)
    SR.check_sources(pcfg, inp)                          # anche le righe del test dei fold
    if sorted(inp.folds) != list(range(1, pcfg["schema_expected"]["k"] + 1)):
        fail("the fold schema was not loaded for every fold")
    got = [len(inp.folds[f]["test"]) for f in sorted(inp.folds)]
    if got != cfg["comparison"]["test_frames_per_fold"] or any(n == 0 for n in got):
        fail(f"the test frames per fold {got} differ from comparison.test_frames_per_fold {cfg['comparison']['test_frames_per_fold']} "
             "(or a fold has no test frame)")
    BR.load_parent(cfg, base, inp)
    load_parent_test_files(cfg, base, inp)
    load_bnrecal(cfg, base, inp)
    ap = rel_path(base, cfg["architecture"]["path"])
    h = sha256_bytes(read_regular(ap, "architecture file"))
    if h != cfg["architecture"]["sha256"]:
        fail(f"architecture file {ap.name}: SHA-256 {h} differs from the configuration")
    inp.code = {"architecture": h, **code, "scripts/nuaa_bn_test.py": sha256_bytes(read_regular(Path(__file__), "code file"))}
    inp.environment = environment()
    if inp.environment["tensorflow_installed"] != cfg["environment"]["tensorflow"]:
        fail(f"TensorFlow {inp.environment['tensorflow_installed']} installed, configuration {cfg['environment']['tensorflow']} "
             "(checked without importing it)")
    if inp.environment["matplotlib"] != cfg["environment"]["matplotlib"]:
        fail(f"matplotlib {inp.environment['matplotlib']} installed, configuration {cfg['environment']['matplotlib']}")
    inp.checks["environment"] = {**inp.environment, "tensorflow_state": "imported later (phases 1 to 3)" if "tensorflow" in sys.modules
                                 else FR.tf_state()}
    inp.checks["test"] = TEST_READ
    inp.checks["code"] = {"pinned": list(CODE_FILES), "estimator": f"{ESTIMATOR_FILE} {ESTIMATOR_SHA256}"}
    return inp


def environment() -> dict:
    out = FR.environment()
    try:
        out["matplotlib"] = importlib.metadata.version("matplotlib")
    except importlib.metadata.PackageNotFoundError:
        out["matplotlib"] = "absent"
    return out


# ============================================================================ baseline (fase 1)

def stored_test(inp: Inputs, name: str, part: dict) -> tuple:
    """Le righe del <modello>_test_scores.csv del genitore (SHA-256 riverificato) confrontate con i frame assemblati:
    (punteggi salvati o None, problema in parole senza identificativi o "")."""
    m = inp.parent[name]
    p, h = inp.test_files[name]
    data = read_regular(p, f"parent {p.name}")
    if sha256_bytes(data) != h:
        fail(f"parent {p.name}: SHA-256 changed since phase 0")
    rows = parse_csv(data, SR.SCORE_COLUMNS, f"parent {p.name}")
    probs = []
    if len(rows) != len(part["rows"]):
        return None, f"{len(rows)} stored rows for {len(part['rows'])} assembled frames"
    if [r["sample_id"] for r in rows] != part["keys"]:
        probs.append("sample ids or their order differ from the assembled frames")
    elif any((r["label"], r["subject_id"], r["session_id"], r["file_sha256"]) != (q["label"], q["subject"], q["session"], q["sha256"])
             for r, q in zip(rows, part["rows"])):
        probs.append("label, subject, session or file SHA-256 differ from the fold manifest")
    if any((r["arm"], r["fold"], r["train_seed"], r["role"], r["checkpoint_sha256"], r["threshold_sha256"]) !=
           (m["arm"], str(m["fold"]), str(m["seed"]), "test", m["weights_sha256"], m["threshold"]["sha256"]) for r in rows):
        probs.append("rows of another model, role, checkpoint or threshold")
    try:
        s = np.array([float(r["score_attack"]) for r in rows], np.float64)
    except ValueError:
        fail(f"parent {p.name}: score_attack not numeric")
    if not np.all(np.isfinite(s)) or np.any(s < 0) or np.any(s > 1):
        fail(f"parent {p.name}: score_attack outside [0, 1]")
    if any(r["prediction_05"] != ("1" if v > FR.FIXED_THRESHOLD else "0") for r, v in zip(rows, s)):
        probs.append("stored prediction_05 differs from the stored score under the decision rule")
    return (None if probs else s), "; ".join(probs)


def test_checks(cfg: dict, p_eval: np.ndarray, gap: float, stored, problem: str, n: int) -> dict:
    """Controlli del test con le statistiche originali: frame allineati, scarto massimo e decisioni discordanti a 0,5
    (regola del progetto: attacco se score > 0,5) contro i punteggi salvati, softmax dei logit."""
    pt, dt, sg = tol(cfg, "test_probability_max_abs_diff"), tol(cfg, "test_decisions_discordant_at_0_5"), tol(cfg, "softmax_of_logits_max_abs_diff")
    c = {"n_test": n, "test_frames_pass": stored is not None, "test_frames_problem": problem or "none", "test_prob_tolerance": pt,
         "test_discordant_tolerance": dt, "test_softmax_gap": gap, "test_softmax_pass": gap <= sg}
    if stored is None:
        c.update({"test_prob_max_abs_diff": None, "test_discordant_05": None, "test_prob_pass": False})
    else:
        d = float(np.max(np.abs(p_eval - stored))) if len(stored) else 0.0
        dis = int(np.sum((p_eval > FR.FIXED_THRESHOLD) != (stored > FR.FIXED_THRESHOLD)))
        c.update({"test_prob_max_abs_diff": d, "test_discordant_05": dis, "test_prob_pass": d <= pt and dis <= dt})
    c["test_pass"] = bool(c["test_frames_pass"] and c["test_prob_pass"] and c["test_softmax_pass"])
    return c


DEV_COLUMNS = BR.BASELINE_COLUMNS[4:-4]                 # i controlli del dev della PR #27, senza il fitting
BASELINE_COLUMNS = ("arm", "fold", "seed", "n_dev") + DEV_COLUMNS + ("dev_pass", "n_test", "test_frames_pass", "test_frames_problem",
                                                                     "test_prob_max_abs_diff", "test_discordant_05", "test_prob_tolerance",
                                                                     "test_discordant_tolerance", "test_prob_pass", "test_softmax_gap",
                                                                     "test_softmax_pass", "test_pass", "baseline_pass")


def baseline_csv(inp: Inputs, base: dict) -> str:
    rows = []
    for n in inp.order:
        if n in base:
            m = inp.parent[n]
            c = {**base[n]["checks"], "arm": m["arm"], "fold": m["fold"], "seed": m["seed"], "n_dev": m["card"]["n_dev"]}
            rows.append([BR.cell_text(c[k]) for k in BASELINE_COLUMNS])
    return csv_text(BASELINE_COLUMNS, rows)


def baseline_phase(cfg: dict, inp: Inputs, store, budget, base: dict) -> None:
    """Fase 1: tutti i modelli prima di qualunque ristima (base: nome -> controlli, probabilità e logit del test)."""
    batch = cfg["evaluation"]["batch"]
    for name in inp.order:
        budget.check(f"baseline {name}")
        t0 = time.time()
        m = inp.parent[name]
        dev = BR.frames(store, inp, "dev", m["arm"], m["fold"])
        if dev["keys"] != sorted(r["key"] for r in BR.role_rows(inp, "dev", m["arm"], m["fold"])):
            fail(f"{name}: the dev frames are not in the key order of the stored scores")
        BR.tf_module().keras.backend.clear_session()
        scorer = (MAKE_SCORER or FR.tf_scorer)(m["weights"], m["weights_sha256"])
        p_parent = FR.probabilities(scorer, dev["X"])[:, 1]
        del scorer
        model = BR.load_model(m["weights"], m["weights_sha256"])
        BR.check_architecture(model, cfg, name)
        ev_dev, _ = BR.evaluate(model, dev, batch, f"{name} dev")
        dc = BR.baseline_checks(cfg, p_parent, ev_dev, m["stored"], m["card"])
        dc["dev_pass"] = dc.pop("baseline_pass")
        del dev
        test = FR.assemble(store, inp.folds[m["fold"]]["test"])
        if not test["keys"]:
            fail(f"{name}: the test set of fold {m['fold']} is empty")
        ev_test, _ = BR.evaluate(model, test, batch, f"{name} test")
        stored, problem = stored_test(inp, name, test)
        tc = test_checks(cfg, ev_test["p"][:, 1], ev_test["softmax_gap"], stored, problem, len(test["keys"]))
        checks = {**dc, **tc, "baseline_pass": bool(dc["dev_pass"] and tc["test_pass"])}
        base[name] = {"checks": checks, "p": ev_test["p"], "z": ev_test["z"], "keys_sha256": sha256_bytes("\n".join(test["keys"]).encode()),
                      "stored": stored, "seconds": time.time() - t0}
        del model, test, ev_test
        gc.collect()
        print(f"   baseline {name}: {'pass' if checks['baseline_pass'] else 'DISCREPANCY'} (dev probability "
              f"{dc['prob_parent_max_abs_diff']:.1e}/{dc['prob_eval_max_abs_diff']:.1e}, val_loss diff {dc['val_loss_diff']:.1e}; test "
              f"{tc['n_test']} frames, probability "
              f"{'NA' if tc['test_prob_max_abs_diff'] is None else format(tc['test_prob_max_abs_diff'], '.1e')}, discordant "
              f"{'NA' if tc['test_discordant_05'] is None else tc['test_discordant_05']}), {base[name]['seconds']:.1f} s", flush=True)


# ============================================================================ ristima e controllo contro la PR #27 (fase 2)

def recal_check(cfg: dict, stats: dict, ref: dict, p_dev: np.ndarray, ref_dev: np.ndarray) -> dict:
    """Che la ristima sia quella della PR #27: statistiche installate float32 contro quelle del run BNRECAL (fuori se
    |qui − PR #27| > rel·|PR #27| + abs, NaN compreso), uguaglianza esatta riportata; probabilità del dev dopo contro i
    punteggi salvati del run BNRECAL (scarto massimo, decisioni discordanti a 0,5)."""
    rel, abs_ = tol(cfg, "recalibration_statistics_rel"), tol(cfg, "recalibration_statistics_abs")
    dm = dv = 0.0
    out_n, identical = 0, True
    for n, s in stats.items():
        rm, rv = ref[n]
        if s["mean32"].shape != rm.shape or s["var32"].shape != rv.shape:
            out_n += int(s["mean32"].size + s["var32"].size)
            identical = False
            continue
        for mine, theirs in ((s["mean32"], rm), (s["var32"], rv)):
            d = np.abs(mine.astype(np.float64) - theirs.astype(np.float64))
            out_n += int(np.sum(~(d <= rel * np.abs(theirs.astype(np.float64)) + abs_)))
            identical = identical and mine.astype(np.float32).tobytes() == theirs.astype(np.float32).tobytes()
        dm = max(dm, float(np.max(np.abs(s["mean32"].astype(np.float64) - rm.astype(np.float64)))))
        dv = max(dv, float(np.max(np.abs(s["var32"].astype(np.float64) - rv.astype(np.float64)))))
    pt, dt = tol(cfg, "recalibration_dev_probability_max_abs_diff"), tol(cfg, "recalibration_dev_decisions_discordant_at_0_5")
    if p_dev.shape != ref_dev.shape:
        dp, dis = None, None
    else:
        dp = float(np.max(np.abs(p_dev - ref_dev))) if len(ref_dev) else 0.0
        dis = int(np.sum((p_dev > FR.FIXED_THRESHOLD) != (ref_dev > FR.FIXED_THRESHOLD)))
    c = {"statistics_max_abs_diff_mean": dm, "statistics_max_abs_diff_variance": dv, "statistics_outside_tolerance": out_n,
         "statistics_identical": identical, "statistics_rel_tolerance": rel, "statistics_abs_tolerance": abs_,
         "statistics_pass": out_n == 0, "dev_prob_max_abs_diff": dp, "dev_discordant_05": dis, "dev_prob_tolerance": pt,
         "dev_discordant_tolerance": dt, "dev_pass": dp is not None and dp <= pt and dis <= dt}
    c["recalibration_pass"] = bool(c["statistics_pass"] and c["dev_pass"])
    return c


RECAL_COLUMNS = ("arm", "fold", "seed", "group", "statistics_max_abs_diff_mean", "statistics_max_abs_diff_variance",
                 "statistics_outside_tolerance", "statistics_identical", "statistics_rel_tolerance", "statistics_abs_tolerance",
                 "statistics_pass", "dev_prob_max_abs_diff", "dev_discordant_05", "dev_prob_tolerance", "dev_discordant_tolerance", "dev_pass",
                 "consistency_channels_outside", "recalibration_pass")


def recalibration_csv(inp: Inputs, recal: dict) -> str:
    rows = []
    for n in inp.order:
        if n in recal:
            m = inp.parent[n]
            c = {**recal[n]["check"], "arm": m["arm"], "fold": m["fold"], "seed": m["seed"], "group": inp.groups[n],
                 "consistency_channels_outside": sum(x["channels_outside_tolerance"] for x in recal[n]["cons"].values())}
            rows.append([BR.cell_text(c[k]) for k in RECAL_COLUMNS])
    return csv_text(RECAL_COLUMNS, rows)


def recalibration_phase(cfg: dict, inp: Inputs, store, budget, recal: dict) -> None:
    """Fase 2: ristima con lo stimatore della PR #27 su copia, controlli di BR (tensori, coerenza, softmax) e controllo
    contro il run BNRECAL; nessun frame del test assemblato qui. recal: nome -> statistiche verificate e controlli."""
    tf = BR.tf_module()
    batch, bn = cfg["evaluation"]["batch"], cfg["batchnorm"]["layers"]
    rel, abs_ = tol(cfg, "consistency_rel"), tol(cfg, "consistency_abs")
    sg = tol(cfg, "softmax_of_logits_max_abs_diff")
    for name in inp.order:
        budget.check(f"recalibration {name}")
        t0 = time.time()
        m = inp.parent[name]
        fit, dev = BR.frames(store, inp, "fitting", m["arm"], m["fold"]), BR.frames(store, inp, "dev", m["arm"], m["fold"])
        tf.keras.backend.clear_session()
        orig = BR.load_model(m["weights"], m["weights_sha256"])
        BR.check_architecture(orig, cfg, name)
        copy = tf.keras.models.clone_model(orig)
        copy.set_weights(orig.get_weights())
        snap = BR.snapshot_weights(orig)
        if BR.snapshot_weights(copy) != snap:
            fail(f"{name}: the in-memory copy differs from the original before the recalibration")
        stats = BR.recalibrate(copy, fit, bn, batch, budget, name)
        if probs := BR.tensor_problems(snap, BR.snapshot_weights(copy), BR.moving_keys(copy, bn), stats):
            fail(f"{name}: " + "; ".join(probs))
        if BR.snapshot_weights(orig) != snap:
            fail(f"{name}: the original model changed during the recalibration")
        fit_after, final = BR.evaluate(copy, fit, batch, f"{name} fitting after", layers=bn)
        cons = BR.consistency(stats, final, rel, abs_)
        if bad := {n: c["channels_outside_tolerance"] for n, c in cons.items() if c["channels_outside_tolerance"]}:
            fail(f"{name}: consistency check failed: the inputs of {sorted(bad)} recomputed with the recalibrated copy do not have the "
                 f"installed moments within rel {rel!r} + abs {abs_!r} ({bad} channel statistics outside)")
        if any(c["count"] != stats[n]["count"] for n, c in cons.items()):
            fail(f"{name}: the consistency pass counted a different number of values per channel")
        dev_after, _ = BR.evaluate(copy, dev, batch, f"{name} dev after")
        for rec, what in ((fit_after, "fitting"), (dev_after, "dev")):
            if rec["softmax_gap"] > sg:
                fail(f"{name}: after the recalibration the softmax of the logits differs from the {what} probabilities by {rec['softmax_gap']:.2e}")
        check = recal_check(cfg, stats, inp.bnrecal[name]["stats"], dev_after["p"][:, 1], inp.bnrecal[name]["dev_after"])
        changes = {n: BR.change_summary(stats[n], cfg["batchnorm"]["epsilon"]) for n in bn}
        recal[name] = {"installed": {n: (stats[n]["mean32"].copy(), stats[n]["var32"].copy()) for n in bn}, "check": check, "cons": cons,
                       "json": BR.stats_json(name, m, stats, cons, changes, cfg), "seconds": time.time() - t0}
        del orig, copy, fit, dev, final
        gc.collect()
        print(f"   recalibration {name}: {'same as PR #27' if check['recalibration_pass'] else 'MISMATCH'} (statistics "
              f"{'identical' if check['statistics_identical'] else 'max diff ' + format(max(check['statistics_max_abs_diff_mean'], check['statistics_max_abs_diff_variance']), '.1e')}"
              f", dev probability {'NA' if check['dev_prob_max_abs_diff'] is None else format(check['dev_prob_max_abs_diff'], '.1e')}), "
              f"{recal[name]['seconds']:.1f} s", flush=True)


# ============================================================================ confronto sul test (fase 3)

def scope_rows(rows, scope: str) -> list:
    return FR.rows_of_stratum(rows, STRATUM[scope])


def scope_metrics(rows) -> dict:
    """Metriche di un ambito con la funzione del genitore (FR.stratum_metrics); un ambito senza frame ha conteggi 0 e
    metriche non definite (None), come una classe assente."""
    return FR.stratum_metrics(rows)


def difference(a, b):
    return None if a is None or b is None else a - b


def flag_of(metric: str, before, after):
    """Indicatore di peggioramento (qualunque variazione in quella direzione conta); None se la metrica non è definita."""
    if before is None or after is None:
        return None
    sign = WORSE[metric][1]
    return bool(after > before) if sign > 0 else bool(after < before)


def result_row(name: str, m: dict, group: str, worsened: bool, scope: str, before: dict, after: dict) -> dict:
    if (before["n_bona_fide"], before["n_attack"], before["n_subjects"]) != (after["n_bona_fide"], after["n_attack"], after["n_subjects"]):
        fail(f"{name} {scope}: before and after are not on the same frames")
    r = {"arm": m["arm"], "fold": m["fold"], "seed": m["seed"], "group": group, "dev_worsened": worsened, "scope": scope,
         "n_subjects": before["n_subjects"], "n_bona_fide": before["n_bona_fide"], "n_attack": before["n_attack"]}
    for k in METRICS:
        r[f"{k}_before"], r[f"{k}_after"] = before[k], after[k]
        r[f"{k}_difference"] = difference(after[k], before[k])
        r[WORSE[k][0]] = flag_of(k, before[k], after[k])
    return r


RESULT_COLUMNS = ("arm", "fold", "seed", "group", "dev_worsened", "scope", "n_subjects", "n_bona_fide", "n_attack") + \
    tuple(f"{k}_{s}" for k in METRICS for s in ("before", "after", "difference")) + FLAGS


def results_csv(rows: list) -> str:
    return csv_text(RESULT_COLUMNS, [[BR.cell_text(r[c]) for c in RESULT_COLUMNS] for r in rows])


def subject_rows(m: dict, scope: str, statistics: str, rows) -> list:
    """Righe per soggetto (riservate): conteggi, errori a 0,5 e AUROC con FR.stratum_metrics sui frame del soggetto."""
    out = []
    for subj in sorted({r["subject_id"] for r in rows}):
        x = FR.stratum_metrics([r for r in rows if r["subject_id"] == subj])
        out.append([m["arm"], m["fold"], m["seed"], subj, scope, statistics, x["n_bona_fide"], x["n_attack"], x["rejected_bona_fide_05"],
                    x["accepted_attacks_05"], FR.fmt(x["apcer_05"]), FR.fmt(x["bpcer_05"]), FR.fmt(x["acer_05"]), FR.fmt(x["auroc"])])
    return out


def install(copy, installed: dict) -> dict:
    """Le statistiche verificate nella fase 2 assegnate alla copia; restituisce il formato di BR.tensor_problems."""
    out = {}
    for n, (mean32, var32) in installed.items():
        layer = copy.get_layer(n)
        layer.moving_mean.assign(mean32)
        layer.moving_variance.assign(var32)
        out[n] = {"mean32": mean32, "var32": var32}
    return out


def score_phase(cfg: dict, inp: Inputs, store, budget, base: dict, recal: dict, after: dict) -> None:
    """Fase 3a, per tutti i modelli prima di qualunque metrica del test: copia ricostruita dalle statistiche verificate
    (solo i 10 tensori mobili diversi), gli stessi frame del test, probabilità e logit dopo con il controllo del softmax,
    tenuti in memoria (after: nome -> {"p", "z"}). Nessuna metrica calcolata, nessun file scritto, nel log solo tempi."""
    tf = BR.tf_module()
    batch, bn = cfg["evaluation"]["batch"], cfg["batchnorm"]["layers"]
    sg = tol(cfg, "softmax_of_logits_max_abs_diff")
    for name in inp.order:
        budget.check(f"comparison scoring {name}")
        t0 = time.time()
        m = inp.parent[name]
        test = FR.assemble(store, inp.folds[m["fold"]]["test"])
        if sha256_bytes("\n".join(test["keys"]).encode()) != base[name]["keys_sha256"]:
            fail(f"{name}: the test frames differ from those of the baseline")
        tf.keras.backend.clear_session()
        orig = BR.load_model(m["weights"], m["weights_sha256"])
        BR.check_architecture(orig, cfg, name)
        copy = tf.keras.models.clone_model(orig)
        copy.set_weights(orig.get_weights())
        snap = BR.snapshot_weights(orig)
        stats = install(copy, recal[name]["installed"])
        if probs := BR.tensor_problems(snap, BR.snapshot_weights(copy), BR.moving_keys(copy, bn), stats):
            fail(f"{name}: rebuilt copy: " + "; ".join(probs))
        rec, _ = BR.evaluate(copy, test, batch, f"{name} test after")
        if rec["softmax_gap"] > sg:
            fail(f"{name}: after the recalibration the softmax of the logits differs from the test probabilities by {rec['softmax_gap']:.2e}")
        after[name] = {"p": rec["p"], "z": rec["z"], "seconds": time.time() - t0}
        del orig, copy, test, rec
        gc.collect()
        print(f"   scored {name} with the re-estimated statistics (rebuilt copy verified), {after[name]['seconds']:.1f} s", flush=True)


def metrics_phase(inp: Inputs, base: dict, recal: dict, after: dict, run: Path, out: dict) -> None:
    """Fase 3b, solo dopo la fase 3a per i modelli completati e la riverifica degli ingressi: righe di punteggio (dalle
    righe del manifest del test in ordine di chiave, senza riassemblare i frame), metriche prima e dopo e dagli score
    salvati del genitore (controllo), righe pubbliche e per soggetto, statistiche per il bootstrap, file riservati."""
    for name in inp.order:
        if name not in after:
            continue
        m = inp.parent[name]
        part = {"rows": sorted(inp.folds[m["fold"]]["test"], key=lambda r: r["key"])}
        rb = BR.score_rows(m, "test", part, {"p": base[name]["p"], "z": base[name]["z"]}, STATISTICS[0])
        ra = BR.score_rows(m, "test", part, after[name], STATISTICS[1])
        rs = SR.score_rows(m["arm"], m["fold"], m["seed"], "test", part, base[name]["stored"], m["threshold"], m["card"])
        for scope in SCOPES:
            xb, xa, xs = scope_rows(rb, scope), scope_rows(ra, scope), scope_rows(rs, scope)
            mb, ma, ms = scope_metrics(xb), scope_metrics(xa), scope_metrics(xs)
            out["rows"].append(result_row(name, m, inp.groups[name], name in inp.dev_worsened, scope, mb, ma))
            out["stored_check"].append({"model": name, "scope": scope, "auroc_diff": None if mb["auroc"] is None or ms["auroc"] is None else
                                        abs(mb["auroc"] - ms["auroc"]),
                                        "metrics_05_equal": all(mb[k] == ms[k] for k in ("apcer_05", "bpcer_05", "acer_05"))})
            out["subjects"] += subject_rows(m, scope, STATISTICS[0], xb) + subject_rows(m, scope, STATISTICS[1], xa)
            sb, sa = (SR.subject_stats(x) for x in (xb, xa))
            if sb["subjects"] != sa["subjects"] or any(not np.array_equal(sb[k], sa[k]) for k in ("na", "nb")):
                fail(f"{name} {scope}: before and after are not on the same subjects and frames")
            out["stats"][(name, scope)] = (sb, sa)
        mw.write_all(run / RESTRICTED, {f"{name}_test_scores.csv": csv_text(TEST_SCORE_COLUMNS, [[x[c] for c in TEST_SCORE_COLUMNS] for x in rb + ra]),
                                        f"{name}_bn_statistics.json": recal[name]["json"]}, newline="")
        out["completed"].append(name)
        r = out["rows"][-2]
        print(f"== {name} ({inp.groups[name]}): test ACER 0.5 {f4(r['acer_05_before'])} -> {f4(r['acer_05_after'])}, AUROC "
              f"{f4(r['auroc_before'])} -> {f4(r['auroc_after'])}", flush=True)
        del rb, ra, rs


# ============================================================================ riepiloghi appaiati e bootstrap per soggetto

def paired_group_bootstrap(models: list) -> dict:
    """models = [(fold, statistiche per soggetto prima, dopo)] dei modelli di un riepilogo in un ambito. Ricampionamento
    del progetto invariato: SR.boot_picks(n) sui soggetti tenuti fuori dei fold del riepilogo nell'ambito, uniti e in
    ordine crescente di identificativo (n = 16 sul test intero con tutti i fold), come SR.BOOT_RECIPE. A ogni
    ricampionamento ogni modello usa le molteplicità dei soggetti del proprio fold (SR.weighted_metrics: conteggi per
    soggetto e conteggi di Mann-Whitney a coppie), gli stessi per prima e dopo, per ogni modello e ogni metrica. Elemento
    nuovo dichiarato: la statistica del ricampionamento è la media sui modelli di (dopo − prima), e un ricampionamento in
    cui la metrica di un modello qualsiasi (prima o dopo) non è definita non conta per quella metrica. Percentili con
    SR.interval; nessun intervallo con meno di SR.BOOT["min_subjects"] soggetti in tutto."""
    subj = {}
    for fold, sb, sa in models:
        if sb["subjects"] != sa["subjects"] or any(not np.array_equal(sb[k], sa[k]) for k in ("na", "nb")):
            fail("paired bootstrap: before and after are not on the same subjects and frames")
        if subj.setdefault(fold, sb["subjects"]) != sb["subjects"]:
            fail(f"paired bootstrap: two models of fold {fold} are not scored on the same subjects")
    pooled = sorted({s for v in subj.values() for s in v})
    if len(pooled) != sum(len(v) for v in subj.values()):
        fail("paired bootstrap: a held-out subject appears in two folds")
    pos = {s: i for i, s in enumerate(pooled)}
    index = {f: np.array([pos[s] for s in v], np.int64) for f, v in subj.items()}
    bf = {s for fold, sb, _ in models for s, nb in zip(sb["subjects"], sb["nb"]) if nb > 0}
    at = {s for fold, sb, _ in models for s, na in zip(sb["subjects"], sb["na"]) if na > 0}
    n = len(pooled)
    out = {"n_subjects": n, "n_subjects_bona_fide": len(bf), "n_subjects_attack": len(at), "n_folds": len(subj),
           "resamples": SR.BOOT["resamples"], "seed": SR.BOOT["seed"], "percentiles": SR.BOOT["percentiles"]}
    if n < SR.BOOT["min_subjects"]:
        out["note"] = f"no interval: fewer than {SR.BOOT['min_subjects']} subjects"
        return {**out, **{k: {"low": None, "high": None, "valid_resamples": 0} for k in METRICS}}
    vals = {k: [] for k in METRICS}
    for pick in SR.boot_picks(n):
        w = np.bincount(pick, minlength=n)
        diffs = {k: [] for k in METRICS}
        for fold, sb, sa in models:
            wf = w[index[fold]]
            a, b = SR.weighted_metrics(sa, wf), SR.weighted_metrics(sb, wf)
            for k in METRICS:
                diffs[k].append(None if a[k] is None or b[k] is None else a[k] - b[k])
        for k in METRICS:
            if all(d is not None for d in diffs[k]):
                vals[k].append(float(np.mean(diffs[k])))
    return {**out, **{k: SR.interval(vals[k]) for k in METRICS}}


def reading_case(diff, e: dict) -> str:
    """Le tre letture pre-scritte (intervallo tutto sotto zero, che contiene lo zero, tutto sopra), o non disponibile."""
    if diff is None or e.get("low") is None or e.get("high") is None:
        return "not_available"
    if e["high"] < 0:
        return "below"
    if e["low"] > 0:
        return "above"
    return "contains"


def summary_members(rows: list, kind: str, label: str, scope: str) -> list:
    key = "group" if kind == "group" else "arm"
    return [r for r in rows if r["scope"] == scope and (label == "all" or r[key] == label)]


def summaries(cfg: dict, rows: list, stats: dict) -> list:
    """Riepiloghi appaiati per gruppo e per braccio, ambito e metrica, con la regola di lettura applicata."""
    rr = cfg["reading_rule"]
    out = []
    for kind, labels in (("group", GROUP_SUMMARIES), ("arm", SR.ARMS)):
        for label in labels:
            for scope in SCOPES:
                mem = summary_members(rows, kind, label, scope)
                boot = paired_group_bootstrap([(r["fold"], *stats[(SR.model_name(r["arm"], r["fold"], r["seed"]), scope)]) for r in mem])
                for k in METRICS:
                    d = [r[f"{k}_difference"] for r in mem]
                    defined = [x for x in d if x is not None]
                    full = bool(mem) and len(defined) == len(mem)
                    mb = float(np.mean([r[f"{k}_before"] for r in mem])) if full else None
                    ma = float(np.mean([r[f"{k}_after"] for r in mem])) if full else None
                    md = float(np.mean(defined)) if full else None
                    e = boot[k]
                    case = reading_case(md, e)
                    primary = (label, scope, k) == (PRIMARY["summary"], PRIMARY["scope"], PRIMARY["metric"]) and kind == "group"
                    who = f"{'group' if kind == 'group' else 'arm'} {label} ({len(mem)} models)"
                    if case == "not_available":
                        reason = ("the metric is not defined for at least one model" if not full else boot.get("note", "no interval"))
                        text = rr["not_available"].format(who=who, scope=scope, metric=METRIC_TEXT[k], reason=reason)
                    else:
                        text = rr[case].format(who=who, scope=scope, metric=METRIC_TEXT[k], difference=f4(md), low=f4(e["low"]), high=f4(e["high"]))
                    out.append({"summary_kind": kind, "summary": label, "scope": scope, "metric": k, "label": "primary" if primary else "secondary",
                                "n_models": len(mem), "n_folds": boot["n_folds"], "n_subjects": boot["n_subjects"],
                                "n_subjects_bona_fide": boot["n_subjects_bona_fide"], "n_subjects_attack": boot["n_subjects_attack"], "mean_before": mb,
                                "mean_after": ma, "mean_difference": md, "low": e["low"], "high": e["high"], "valid_resamples": e["valid_resamples"],
                                "resamples": boot["resamples"], "reading": case, "models_lower": sum(1 for x in defined if x < 0),
                                "models_equal": sum(1 for x in defined if x == 0), "models_higher": sum(1 for x in defined if x > 0),
                                "models_undefined": len(d) - len(defined), "note": boot.get("note", "none"), "statement": text})
    return out


SUMMARY_COLUMNS = ("summary_kind", "summary", "scope", "metric", "label", "n_models", "n_folds", "n_subjects", "n_subjects_bona_fide",
                   "n_subjects_attack", "mean_before", "mean_after",
                   "mean_difference", "low", "high", "valid_resamples", "resamples", "reading", "models_lower", "models_equal", "models_higher",
                   "models_undefined", "note")


def summary_csv(summ: list) -> str:
    return csv_text(SUMMARY_COLUMNS, [[BR.cell_text(s[c]) for c in SUMMARY_COLUMNS] for s in summ])


def name_of(r: dict) -> str:
    return SR.model_name(r["arm"], r["fold"], r["seed"])


def reading(cfg: dict, rows: list, summ: list) -> dict:
    """La regola di lettura della configurazione applicata meccanicamente: primaria (i dieci, test, ACER a 0,5) con il
    conteggio semplice dei dieci; secondarie descrittive con la stessa formulazione; ogni peggioramento elencato."""
    rr = cfg["reading_rule"]
    p = next(s for s in summ if s["label"] == "primary")
    ten = [r for r in rows if r["group"] == "ten" and r["scope"] == "test"]
    d = [r["acer_05_difference"] for r in ten]
    cnt = {"lower": sum(1 for x in d if x is not None and x < 0), "equal": sum(1 for x in d if x is not None and x == 0),
           "higher": sum(1 for x in d if x is not None and x > 0), "n": len(ten)}
    worse, undefined = [], []
    for r in rows:
        for k in METRICS:
            f = WORSE[k][0]
            if r[f] is None:
                undefined.append({"model": name_of(r), "group": r["group"], "scope": r["scope"], "metric": k})
            elif r[f]:
                worse.append({"model": name_of(r), "group": r["group"], "dev_worsened": r["dev_worsened"], "scope": r["scope"], "flag": f,
                              "before": r[f"{k}_before"], "after": r[f"{k}_after"], "difference": r[f"{k}_difference"]})
    return {"primary": {"rule": rr["primary"], "case": p["reading"], "statement": p["statement"], "mean_difference": p["mean_difference"],
                        "low": p["low"], "high": p["high"], "count": cnt, "count_statement": rr["count"].format(**cnt),
                        "models_higher": [name_of(r) for r in ten if r["acer_05_difference"] is not None and r["acer_05_difference"] > 0]},
            "secondary": {"rule": rr["secondary"], "statements": [{k: s[k] for k in ("summary_kind", "summary", "scope", "metric", "reading",
                                                                                    "statement")} for s in summ if s["label"] == "secondary"]},
            "worsening": {"rule": cfg["comparison"]["worsening"], "cases": worse,
                          "counts": {f"{f} ({scope})": sum(1 for w in worse if w["flag"] == f and w["scope"] == scope) for scope in SCOPES for f in FLAGS}},
            "undefined": undefined, "limits": rr["limits"]}


# ============================================================================ testi pubblici

def table(headers, rows) -> str:
    return SR.table(headers, rows)


def ci(s: dict) -> str:
    return f"[{f4(s['low'])}, {f4(s['high'])}]" if s.get("low") is not None else "no interval"


def aggregate_md(cfg: dict, doc: dict, rows: list, summ: list) -> str:
    de = cfg["declarations"]
    L = [f"# Before/after on the NUAA test: BatchNormalization statistics re-estimated as in PR #27 ({doc['run_id']})", "",
         f"Status: **{doc['status']}**. Parent run `{doc['parent_run_id']}`; BNRECAL run `{doc['bnrecal_run_id']}`; configuration "
         f"`{doc['config_sha256'][:12]}`; note `{(doc['note_sha256'] or 'none')[:12]}`; estimator `{ESTIMATOR_FILE}` "
         f"`{ESTIMATOR_SHA256[:12]}`. No training.", "",
         "## Declarations (fixed before the run)", "", *[f"- {de[k]}" for k in DECLARATION_KEYS], "",
         f"What is read of the test: {doc['test_read']}.", ""]
    a = doc.get("reading")
    if a is None:
        L += [f"Reading not applied: {doc['reading_not_applied']}", ""]
    else:
        p = a["primary"]
        L += ["## Primary reading (fixed before the run, applied by the script)", "", p["rule"], "", f"**{p['statement']}**", "",
              p["count_statement"] + (f" Higher after: {', '.join(p['models_higher'])}." if p["models_higher"] else ""), "",
              "## Paired summaries (mean over the models of after - before; subject bootstrap, held-out subjects pooled over the folds)", "", a["secondary"]["rule"], "",
              table(["summary", "scope", "metric", "label", "models", "subjects", "mean before", "mean after", "mean difference [interval]",
                     "reading", "lower / equal / higher / NA"],
                    [[f"{s['summary_kind']} {s['summary']}", s["scope"], METRIC_TEXT[s["metric"]], s["label"], s["n_models"], s["n_subjects"],
                      f4(s["mean_before"]), f4(s["mean_after"]), f"{f4(s['mean_difference'])} {ci(s)}", s["reading"],
                      f"{s['models_lower']} / {s['models_equal']} / {s['models_higher']} / {s['models_undefined']}"] for s in summ]), "",
              "## Worsening (every case listed)", "", a["worsening"]["rule"], "",
              table(["flag (scope)", "models"], [[k, v] for k, v in a["worsening"]["counts"].items()]), ""]
        L += [f"- {w['model']} ({w['group']}{', dev indicator worsened in PR #27' if w['dev_worsened'] else ''}), {w['scope']}: {w['flag']} "
              f"{f4(w['before'])} -> {f4(w['after'])} (difference {w['difference']:.6g})" for w in a["worsening"]["cases"]] or ["- none"]
        L += ["", "Metrics not defined (a class absent in the scope): " + (", ".join(f"{u['model']} {u['scope']} {u['metric']}" for u in a["undefined"])
                                                                          or "none") + ".", "", "## Limits", "", a["limits"], ""]
    if rows:
        L += ["## All models, before -> after", "",
              table(["model", "group", "dev worsened (PR #27)", "scope", "n bona fide / attack", "APCER 0.5", "BPCER 0.5", "ACER 0.5", "AUROC"],
                    [[name_of(r), r["group"], int(r["dev_worsened"]), r["scope"], f"{r['n_bona_fide']} / {r['n_attack']}"] +
                     [f"{f4(r[f'{k}_before'])} -> {f4(r[f'{k}_after'])}" for k in METRICS] for r in rows]), ""]
    t = doc["timings"]
    L += ["## Time", "", f"Wall time {t['wall_seconds']:.0f} s ({t['wall_seconds'] / 60:.1f} min), process CPU time {t['cpu_seconds']:.0f} s, "
          f"against the note's estimate of about {t['estimate_central_minutes']:g} minutes (upper estimate about "
          f"{t['estimate_upper_minutes']:g}); wall-time cap {t['cap_minutes']} minutes.", "",
          table(["phase", "seconds"], [[k, f"{v:.1f}"] for k, v in t["phases"].items()]), ""]
    return "\n".join(L)


# ============================================================================ figura

def error_bar(mean: float, low: float, high: float) -> list:
    """yerr di matplotlib per una media e il suo intervallo: [[distanza in basso], [distanza in alto]]."""
    return [[mean - low], [high - mean]]


def figure_png(results_text: str, summary_text: str) -> bytes:
    """differences.png dai soli results.csv e summary.csv pubblici: per metrica e ambito le differenze dopo − prima di
    ogni modello (colore = braccio, forma = gruppo), la linea dello zero, a destra le medie dei gruppi con il loro
    intervallo; i valori non definiti sono contati nel riquadro. Agg, nessun metadato variabile: byte deterministici
    per la stessa versione di matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rows = parse_csv(results_text.encode("utf-8"), RESULT_COLUMNS, "results.csv")
    summ = parse_csv(summary_text.encode("utf-8"), SUMMARY_COLUMNS, "summary.csv")
    color = {"arm_b": "#2a78d6", "arm_a_star": "#eb6834", "arm_b_prime": "#1baf7a"}
    marker = {"ten": "s", "control": "o", "apart": "D"}
    arm_label = {"arm_b": "B", "arm_a_star": "A*", "arm_b_prime": "B'"}
    order = sorted({(r["arm"], int(r["fold"]), int(r["seed"])) for r in rows}, key=lambda x: (SR.ARMS.index(x[0]), x[1], x[2]))
    xpos, x = {}, 0
    for i, key in enumerate(order):
        if i and key[0] != order[i - 1][0]:
            x += 1.5
        xpos[key] = x
        x += 1
    xs = x + 3.0                       # le medie dei gruppi, separate dai modelli
    step = 4.0
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": "#8a8a85", "axes.linewidth": 0.8}):
        fig, axes = plt.subplots(len(SCOPES), len(METRICS), figsize=(16, 7.5), dpi=100, squeeze=False)
        for i, scope in enumerate(SCOPES):
            for j, k in enumerate(METRICS):
                ax = axes[i][j]
                ax.axhline(0.0, color="#4a4a48", linewidth=0.8, zorder=1)
                na = 0
                for r in rows:
                    if r["scope"] != scope:
                        continue
                    v = r[f"{k}_difference"]
                    if v == FR.NA:
                        na += 1
                        continue
                    ax.scatter([xpos[(r["arm"], int(r["fold"]), int(r["seed"]))]], [float(v)], s=22, marker=marker[r["group"]],
                               color=color[r["arm"]], edgecolors="white", linewidths=0.5, zorder=3)
                ticks, labels = [], []
                for g_i, g in enumerate(GROUP_SUMMARIES):
                    s = next((s for s in summ if s["summary_kind"] == "group" and s["summary"] == g and s["scope"] == scope and s["metric"] == k), None)
                    gx = xs + step * g_i
                    ticks.append(gx)
                    labels.append({"all": "all 36"}.get(g, g))
                    if s is None or s["mean_difference"] == FR.NA:
                        continue
                    md = float(s["mean_difference"])
                    if s["low"] != FR.NA:
                        lo, hi = float(s["low"]), float(s["high"])
                        ax.errorbar([gx], [md], yerr=error_bar(md, lo, hi), fmt="_", color="#2b2b29", markersize=9, capsize=3,
                                    linewidth=1.0, zorder=4)
                    else:
                        ax.scatter([gx], [md], marker="_", s=60, color="#2b2b29", zorder=4)
                arm_ticks = []
                for arm in SR.ARMS:
                    pos = [xpos[key] for key in order if key[0] == arm]
                    if pos:
                        arm_ticks.append(((min(pos) + max(pos)) / 2, arm_label[arm]))
                ax.set_xticks([t for t, _ in arm_ticks] + ticks)
                ax.set_xticklabels([lab for _, lab in arm_ticks] + labels, rotation=45, ha="right")
                ax.axvline(xs - 1.5, color="#c3c2b7", linewidth=0.8, zorder=1)
                ax.set_xlim(-1, xs + step * (len(GROUP_SUMMARIES) - 1) + 2)
                ax.set_title(f"{METRIC_TEXT[k]}, scope {scope}", fontsize=8.5)
                if j == 0:
                    ax.set_ylabel("after - before")
                ax.grid(axis="y", color="#e4e3df", linewidth=0.6, zorder=0)
                ax.text(0.99, 0.02, f"not defined: {na}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#5a5a56")
        handles = [Line2D([0], [0], marker="o", linestyle="", color=color[a], label=f"arm {arm_label[a]}") for a in SR.ARMS] + \
                  [Line2D([0], [0], marker=marker[g], linestyle="", color="#5a5a56", label=f"group {g}") for g in marker] + \
                  [Line2D([0], [0], marker="_", linestyle="-", color="#2b2b29", label="right of the line: group mean, subject-bootstrap interval")]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.02))
        fig.text(0.5, 0.985, "After - before (re-estimated minus original statistics) per checkpoint, fixed threshold 0.5 for "
                 "APCER, BPCER and ACER", ha="center", va="top", fontsize=9)
        fig.text(0.5, 0.004, "Post hoc diagnostic comparison on a test already analysed; not an independent validation. Generated from "
                 "the public results.csv and summary.csv.", ha="center", va="bottom", fontsize=7.5, color="#5a5a56")
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", metadata={"Software": None})
        plt.close(fig)
    return buf.getvalue()


def write_binary(out: Path, name: str, data: bytes) -> None:
    """Scrittura atomica di un file binario (temporaneo nella stessa cartella, poi os.replace), come mw.write_all."""
    out.mkdir(parents=True, exist_ok=True)
    q = out / name
    if q.is_symlink() or (q.exists() and not q.is_file()):
        fail(f"{name} in the output folder is a link or not a regular file: refused")
    fd, tp = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=out)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tp, q)
    except BaseException:
        Path(tp).unlink(missing_ok=True)
        raise


# ============================================================================ cartella del run

def repo_state(root: Path) -> dict:
    return REPO_STATE(root) if REPO_STATE is not None else SR.repo_state(root)


def run_id_for(date: str, state: dict) -> str:
    if not re.fullmatch(r"\d{8}", date or ""):
        fail("--date must be YYYYMMDD")
    rid = f"{date}-BNTEST-{state['commit']}"
    if not RUN_ID_RE.match(rid):
        fail(f"run id {rid} does not follow YYYYMMDD-BNTEST-<commit>")
    return rid


def restricted_outputs(order) -> tuple:
    return tuple(f"{n}_{x}" for n in order for x in PER_MODEL) + RESTRICTED_FIXED


def input_paths(cfg: dict, pcfg: dict, base: Path, cfg_p: Path) -> dict:
    out = BR.input_paths(cfg, pcfg, base, cfg_p)
    out["the note"] = rel_path(ROOT, cfg["note_path"])
    out["the schema folder"] = rel_path(base, pcfg["schema_run"]["folder"])
    out["the schema configuration"] = rel_path(base, pcfg["schema_run"]["config_path"])
    out["the BNRECAL run folder"] = rel_path(base, cfg["bnrecal_run"]["folder"])
    out["the PR #27 configuration"] = rel_path(base, cfg["bnrecal_run"]["config_path"])
    return out


def run_dir_problem(run: Path, root: Path, inputs: dict, order, out_root: Path) -> str:
    """Motivo per cui la cartella del run non è ammessa ("" se lo è), prima di qualunque modifica."""
    if out_root.is_symlink():
        return "the output root is a symbolic link"
    for label, p in inputs.items():
        if pg.within(p, run) or (Path(p).is_dir() and pg.within(run, p)):
            return f"{label} {Path(p).name} is inside the run folder, or contains it: inputs and outputs must be separate"
    extra = [root / x for x in EXTRA_PROTECTED]
    msg = mw.output_problem(run, root, pg.BASE_PROTECTED, [root / DEFAULT_OUT_ROOT], extra, inputs, [STATUS, LOG], [run / PUBLIC, run / RESTRICTED])
    for sub, names in ((PUBLIC, PUBLIC_OUTPUTS), (RESTRICTED, restricted_outputs(order))):
        msg = msg or mw.output_problem(run / sub, root, pg.BASE_PROTECTED, [root / DEFAULT_OUT_ROOT], extra, inputs, list(names))
    return msg


def invalidate_outputs(run: Path, order) -> list:
    left = []
    for sub, names in ((PUBLIC, PUBLIC_OUTPUTS), (RESTRICTED, restricted_outputs(order))):
        d = run / sub
        if d.is_dir():
            left += [f"{sub}/{n}" for n in mw.invalidate_each(d, list(names), None, FAIL_DOC)]
    return left


def resolve(a) -> tuple:
    base = Path(a.input_root).resolve() if a.input_root else ROOT
    cfg_p = Path(a.config) if Path(a.config).is_absolute() else ROOT / a.config
    return base, cfg_p


def write_public(run: Path, inp: Inputs, public: dict) -> str:
    """Scansione di ogni testo pubblico (FR.public_problems), la figura binaria scritta per prima, poi i testi e
    SHA256SUMS in un solo gruppo atomico; restituisce lo SHA-256 del SHA256SUMS pubblico."""
    for pname, text in public.items():
        if isinstance(text, str) and (probs := FR.public_problems(text, inp, pname.endswith(".md"))):
            fail(f"public file {pname} refused: " + "; ".join(probs))
    sums = "".join(f"{sha256_bytes(v if isinstance(v, bytes) else v.encode('utf-8'))}  {n}\n" for n, v in sorted(public.items()))
    for n, v in public.items():
        if isinstance(v, bytes):
            write_binary(run / PUBLIC, n, v)
    mw.write_all(run / PUBLIC, {**{n: v for n, v in public.items() if isinstance(v, str)}, "SHA256SUMS": sums})
    return sha256_bytes(sums.encode("utf-8"))


def write_restricted_sums(run: Path, names: list) -> str:
    rdir = run / RESTRICTED
    missing = [n for n in names if not (rdir / n).is_file()]
    if missing:
        fail(f"restricted outputs missing: {missing}")
    extra = sorted(e for e in os.listdir(rdir) if e not in names and e != "SHA256SUMS")
    if extra:
        fail(f"restricted folder holds files that are not outputs of this run: {extra}")
    sums = "".join(f"{sha256_file(rdir / n)}  {n}\n" for n in sorted(names))
    mw.write_all(rdir, {"SHA256SUMS": sums})
    return sha256_bytes(sums.encode("utf-8"))


def final_checks(cfg: dict, inp: Inputs, base_dir: Path) -> dict:
    """Alla fine: SHA-256 di ogni .keras del genitore, del suo SHA256SUMS riservato e dei due SHA256SUMS del run BNRECAL."""
    changed = [n for n in inp.order if sha256_file(inp.parent[n]["weights"]) != inp.parent[n]["weights_sha256"]]
    if changed:
        fail(f"parent checkpoints changed during the run: {changed}")
    run = rel_path(base_dir, cfg["parent_run"]["folder"])
    if sha256_bytes(read_regular(run / RESTRICTED / "SHA256SUMS", "parent restricted SHA256SUMS")) != inp.parent_sums_sha:
        fail("the parent restricted SHA256SUMS changed during the run")
    br = rel_path(base_dir, cfg["bnrecal_run"]["folder"])
    for sub in (PUBLIC, RESTRICTED):
        if sha256_bytes(read_regular(br / sub / "SHA256SUMS", f"BNRECAL {sub} SHA256SUMS")) != inp.bnrecal_sums[sub]:
            fail(f"the BNRECAL {sub} SHA256SUMS changed during the run")
    return {"parent_checkpoints_unchanged": len(inp.order), "parent_restricted_sha256sums_unchanged": True,
            "bnrecal_sha256sums_unchanged": True}


def cpu_seconds() -> float:
    """Tempo CPU del processo su tutti i thread (os.times: utente + sistema), registrato, non limitato."""
    t = os.times()
    return round(t.user + t.system, 1)


# ============================================================================ comandi

def cmd_check_inputs(a) -> int:
    base, cfg_p = resolve(a)
    cfg, cfg_sha, bn_cfg = load_config(cfg_p, base)
    inp = load_inputs(cfg, bn_cfg, base, need_note=False)
    g = inp.checks["groups"]
    print(f"CHECK-INPUTS: configuration {cfg_sha[:12]}, note {inp.note_sha[:12] if inp.note_sha else 'null (not yet frozen)'}, PR #27 configuration "
          f"{cfg['bnrecal_run']['config_sha256'][:12]} with every shared section equal except the rewritten descriptive keys "
          f"{', '.join(f'{s}.{k}' for s, ks in SHARED_REWRITTEN.items() for k in ks)}, code pinned ({len(CODE_FILES)} files, estimator "
          f"{ESTIMATOR_SHA256[:12]}), parent run {cfg['parent_run']['run_id']} PASS ({len(inp.order)} models: card, dev scores, threshold, "
          f"weights and test scores equal to its restricted SHA256SUMS; test scores hashed only, not parsed), fold schema with "
          f"{sum(len(inp.folds[f]['test']) for f in inp.folds)} test rows, BNRECAL run {cfg['bnrecal_run']['run_id']} PASS (statistics and "
          f"dev scores of {len(inp.bnrecal)} models), groups: {len(g['ten'])} degenerate, {g['control']} controls, {len(g['apart'])} apart; "
          f"{len(inp.dev_worsened)} models with a worsened dev indicator in PR #27; TensorFlow {inp.environment['tensorflow_installed']} "
          "installed and not imported; nothing written")
    print("RESULT: PASS (check-inputs)")
    return 0


def cmd_figure(a) -> int:
    folder = Path(a.run_folder)
    out = Path(a.output)
    if os.path.lexists(out):
        fail(f"{out.name} already exists: the figure is written only to a new file")
    res = read_regular(folder / PUBLIC / "results.csv", "results.csv").decode("utf-8")
    summ = read_regular(folder / PUBLIC / "summary.csv", "summary.csv").decode("utf-8")
    data = figure_png(res, summ)
    write_binary(out.parent, out.name, data)
    print(f"figure: {out.name}, SHA-256 {sha256_bytes(data)}")
    print("RESULT: PASS (figure)")
    return 0


def cmd_run(a) -> int:
    if not a.i_am_authorized:
        fail("run loads the restricted checkpoints and scores the NUAA test: pass --i-am-authorized (control assigned by the coordinator "
             "on 16 November 2026, after the note, the configuration and the code are published in the pull request)")
    base_dir, cfg_p = resolve(a)
    cfg, cfg_sha, bn_cfg = load_config(cfg_p, base_dir)
    if cfg["note_sha256"] is None:
        fail("configuration: note_sha256 is null: the NOTE.md must be frozen and its SHA-256 written before the run")
    state = repo_state(ROOT)
    if state["dirty"]:
        fail("the git work tree is not clean: the run needs a committed state (commit or stash the changes first)")
    rid = run_id_for(a.date or time.strftime("%Y%m%d"), state)
    out_root = Path(a.out_root) if Path(a.out_root).is_absolute() else ROOT / a.out_root
    run = out_root / rid
    pcfg, _ = SR.load_config(rel_path(base_dir, cfg["parent_run"]["config_path"]))
    order = [SR.model_name(*x) for x in SR.training_order(pcfg["schema_expected"]["k"], pcfg["train_seeds"])]
    if os.path.lexists(run):
        fail(f"the run folder {rid} already exists: every attempt keeps its outcome (archive it or use another commit)")
    if msg := run_dir_problem(run, ROOT, input_paths(cfg, pcfg, base_dir, cfg_p), order, out_root):
        fail(f"refusing the run folder: {msg}")
    run.mkdir(parents=True)
    head = {"run_id": rid, "commit": state["commit"], "dirty": False, "config_sha256": cfg_sha, "note_sha256": cfg["note_sha256"],
            "parent_run_id": cfg["parent_run"]["run_id"], "bnrecal_run_id": cfg["bnrecal_run"]["run_id"], "command": "run"}
    budget = SR.Budget(cfg["budget"]["wall_time_cap_minutes"])
    SR.signal_handlers(budget)
    log, phases, base, recal, after, stopped = None, {}, {}, {}, {}, None
    out = {"rows": [], "subjects": [], "stats": {}, "completed": [], "stored_check": []}
    try:
        FR.write_status(run, {**head, "status": "RUNNING", "passed": False, "started": now(), "pid": os.getpid(),
                              "note": "RUNNING without a live process means a failed run; SIGUSR1 to this pid stops it as INCOMPLETE"})
        log = FR.open_log(run)
        print(f"== NUAA BatchNormalization before/after on the test {rid}: configuration {cfg_sha[:12]}, note {cfg['note_sha256'][:12]}, "
              f"parent {cfg['parent_run']['run_id']}, BNRECAL {cfg['bnrecal_run']['run_id']}, cap {cfg['budget']['wall_time_cap_minutes']} min, "
              f"pid {os.getpid()}", flush=True)
        t = time.time()
        inp = load_inputs(cfg, bn_cfg, base_dir)
        phases["phase 0 inputs"] = time.time() - t
        store = FR.FrameStore(inp, inp.X_extracted, inp.extracted_keys)
        print(f"   phase 0 PASS: {len(inp.order)} models, fold schema loaded, BNRECAL anchored, peak RSS {SR.peak_rss_mb():.0f} MB", flush=True)
        status, phase = "PASS", "baseline"
        try:
            t = time.time()
            baseline_phase(cfg, inp, store, budget, base)
            phases["phase 1 baseline"] = time.time() - t
            failed = [n for n in inp.order if not base[n]["checks"]["baseline_pass"]]
            if failed:
                status = "BASELINE_FAILED"
                print(f"== BASELINE_FAILED: {len(failed)} models do not reproduce the stored dev or test values ({', '.join(failed)}); "
                      "nothing is re-estimated (stop rule)", flush=True)
            else:
                phase = "recalibration"
                t = time.time()
                recalibration_phase(cfg, inp, store, budget, recal)
                phases["phase 2 recalibration"] = time.time() - t
                mismatch = [n for n in inp.order if not recal[n]["check"]["recalibration_pass"]]
                if mismatch:
                    status = "RECALIBRATION_MISMATCH"
                    print(f"== RECALIBRATION_MISMATCH: {len(mismatch)} models differ from the re-estimation of PR #27 ({', '.join(mismatch)}); "
                          "no test score with re-estimated statistics is computed (stop rule)", flush=True)
                else:
                    phase = "comparison"
                    t = time.time()
                    score_phase(cfg, inp, store, budget, base, recal, after)
                    phases["phase 3a scoring with the re-estimated statistics"] = time.time() - t
        except SR.CapExceeded as e:
            status = "INCOMPLETE"
            stopped = {"phase": phase, "step": e.step, "cause": e.cause, "elapsed_minutes": round(e.minutes, 1)}
            phases[f"{phase} (stopped)"] = time.time() - t
            print(f"== {'CAP' if e.cause == 'cap' else 'STOP'}: {e}; no further model; an INCOMPLETE run writes no per-model file and "
                  "no test metric", flush=True)
        t = time.time()
        # riverifica degli ingressi PRIMA di qualunque metrica del test (fase 3b)
        checks = {**inp.checks, "final": final_checks(cfg, inp, base_dir)}
        phases["input re-verification"] = time.time() - t
        t = time.time()
        if phase == "comparison" and status == "PASS":       # 3b tutto o niente: mai con INCOMPLETE, mai su un sottoinsieme
            metrics_phase(inp, base, recal, after, run, out)
        phases["phase 3b metrics"] = time.time() - t
        t = time.time()
        rows = out["rows"]
        summ, read = [], None
        if status == "PASS":
            summ = summaries(cfg, rows, out["stats"])
            read = reading(cfg, rows, summ)
        if out["stored_check"]:
            sc = out["stored_check"]
            checks["before_metrics_against_stored_scores"] = {
                "metrics_05_equal": sum(1 for x in sc if x["metrics_05_equal"]), "of": len(sc),
                "auroc_max_abs_diff": max((x["auroc_diff"] for x in sc if x["auroc_diff"] is not None), default=None)}
        phases["summaries, final checks and writes"] = time.time() - t
        wall = budget.elapsed()
        timings = {"wall_seconds": round(wall, 1), "cpu_seconds": cpu_seconds(), "phases": {k: round(v, 1) for k, v in phases.items()},
                   "estimate_central_minutes": cfg["budget"]["estimate_minutes"], "estimate_upper_minutes": cfg["budget"]["upper_estimate_minutes"],
                   "cap_minutes": cfg["budget"]["wall_time_cap_minutes"],
                   "per_model": {n: {"baseline": round(base[n]["seconds"], 1), **({"recalibration": round(recal[n]["seconds"], 1)} if n in recal else {}),
                                     **({"comparison_scoring": round(after[n]["seconds"], 1)} if n in after else {})} for n in base}}
        record = {**head, "status": status, "finished": now(), "wall_seconds": timings["wall_seconds"],
                  "process_cpu_seconds": timings["cpu_seconds"], "timings": timings, "stopped": stopped, "models": len(inp.order),
                  "baseline_failed": [n for n in base if not base[n]["checks"]["baseline_pass"]],
                  "recalibration_mismatch": [n for n in recal if not recal[n]["check"]["recalibration_pass"]],
                  "statistics_identical_to_pr27": sum(1 for n in recal if recal[n]["check"]["statistics_identical"]),
                  "completed": out["completed"], "tolerances": cfg["tolerances"], "stop_rules": cfg["stop_rules"], "estimator": cfg["estimator"],
                  "evaluation": cfg["evaluation"], "comparison": cfg["comparison"], "bootstrap": cfg["bootstrap"], "checks": checks,
                  "environment": inp.environment, "device": "CPU (GPU hidden)", "code_sha256": inp.code, "cells_manifests_sha256": inp.cells_sha,
                  "fold_manifests_sha256": inp.fold_sha, "dev_worsened": inp.dev_worsened, "test_read": TEST_READ,
                  "post_hoc": cfg["declarations"]["post_hoc"], "declarations": cfg["declarations"],
                  "parent_public_aggregate_sha256": inp.parent_aggregate_sha}
        public = {"baseline.csv": baseline_csv(inp, base), "run_record.json": json.dumps(record, indent=1) + "\n"}
        if recal:
            public["recalibration_check.csv"] = recalibration_csv(inp, recal)
        if phase == "comparison" and status == "PASS":
            doc = {**head, "status": status, "models": len(inp.order), "completed": len(out["completed"]), "timings": timings,
                   "groups": inp.checks["groups"], "dev_worsened": inp.dev_worsened, "declarations": cfg["declarations"], "reading": read,
                   "reading_not_applied": None if read else f"run {status}: {len(out['completed'])} of {len(inp.order)} models complete",
                   "summaries": summ, "test_read": TEST_READ, "bootstrap": cfg["bootstrap"],
                   "before_metrics_against_stored_scores": checks.get("before_metrics_against_stored_scores")}
            public["results.csv"] = results_csv(rows)
            public["aggregate.json"] = json.dumps(doc, indent=1) + "\n"
            public["aggregate.md"] = aggregate_md(cfg, doc, rows, summ)
            if status == "PASS":
                public["summary.csv"] = summary_csv(summ)
                public[FIGURE] = figure_png(public["results.csv"], public["summary.csv"])
        pub_sha = write_public(run, inp, public)
        rsums = None
        if out["completed"]:
            mw.write_all(run / RESTRICTED, {"per_subject.csv": csv_text(SUBJECT_COLUMNS, out["subjects"]), "run_record.json": public["run_record.json"],
                                            "public_SHA256SUMS.sha256": pub_sha + "  public/SHA256SUMS\n"}, newline="")
            rsums = write_restricted_sums(run, [f"{n}_{x}" for n in out["completed"] for x in PER_MODEL] + list(RESTRICTED_FIXED[:-1]))
        if read:
            print(f"primary: {read['primary']['statement']}")
            print(f"count: {read['primary']['count_statement']}")
        print(f"run folder: {run.name}, wall {wall:.0f} s ({wall / 60:.1f} min; estimate about {cfg['budget']['estimate_minutes']:g}, upper "
              f"about {cfg['budget']['upper_estimate_minutes']:g}), process CPU {cpu_seconds():.0f} s, peak RSS {SR.peak_rss_mb():.0f} MB")
        print(f"RESULT: {status} (run)")
        FR.close_log(log)
        log = None
        FR.write_status(run, {**head, "status": status, "passed": status == "PASS", "finished": record["finished"], "wall_seconds": round(wall, 1),
                              "stopped": stopped, "baseline_failed": record["baseline_failed"], "recalibration_mismatch": record["recalibration_mismatch"],
                              "completed": out["completed"], "public_sums_sha256": pub_sha, "restricted_sums_sha256": rsums or "none",
                              "run_log_sha256": sha256_file(run / LOG)})
    except BaseException as e:
        reason = str(e) if isinstance(e, Failure) else f"unexpected error {type(e).__name__}: {e}"
        left = invalidate_outputs(run, order)
        outputs = "removed" if not left else f"left behind: {', '.join(left)}"
        tail = f"ERROR: {reason}\nFAIL outputs: {outputs}\nRESULT: FAIL (run)\n"
        if log is not None:
            log[0].write(tail)
            FR.close_log(log)
        elif (run / LOG).is_file() and not (run / LOG).is_symlink():
            with open(run / LOG, "a", encoding="utf-8") as fh:
                fh.write(tail)
        try:
            FR.write_status(run, {**head, "status": "FAIL", "passed": False, "reason": reason, "outputs": outputs,
                                  "run_log_sha256": sha256_file(run / LOG) if (run / LOG).is_file() else "none"})
        except Exception:  # noqa: BLE001
            print("ERROR: status.json could not be written", file=sys.stderr)
        raise
    return STATUS_CODE[status]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="NUAA LivenessNet: the 36 patience-4 checkpoints scored on the test before and after the "
                                             "BatchNormalization re-estimation of PR #27 (CPU, no training, fixed threshold 0.5)")
    ap.add_argument("cmd", choices=["check-inputs", "run", "figure"])
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="configuration JSON (relative to the repository root)")
    ap.add_argument("--input-root", default=None, help="base folder of the input paths of the configuration (default: repository root)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="folder that receives <run-id>/ (default results/bntest; another folder for tests only)")
    ap.add_argument("--date", default=None, help="YYYYMMDD of the run id (default: today)")
    ap.add_argument("--i-am-authorized", action="store_true", help="required by run: inference on the restricted checkpoints and frames, test included")
    ap.add_argument("--run-folder", default=None, help="figure: the run folder whose public results.csv and summary.csv are read")
    ap.add_argument("--output", default=None, help="figure: the new PNG file to write")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "figure":
            if not a.run_folder or not a.output:
                fail("figure needs --run-folder and --output")
            return cmd_figure(a)
        return cmd_check_inputs(a) if a.cmd == "check-inputs" else cmd_run(a)
    except Failure as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(f"RESULT: FAIL ({a.cmd})")
        return 1
    except KeyboardInterrupt:
        print(f"RESULT: FAIL ({a.cmd}, interrupted)")
        return 130
    except SystemExit as e:
        print(f"RESULT: FAIL ({a.cmd})")
        return e.code if isinstance(e.code, int) and e.code else 1
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: unexpected error {type(e).__name__}: {e}", file=sys.stderr)
        print(f"RESULT: FAIL ({a.cmd})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
