"""
Valutazione sul test NUAA dei 48 modelli nuovi del confronto LivenessNet / ResNet-18 (preregistrazione definitiva della PR
#29, delivery/pad-arch-comparison/PROTOCOL.md §3, §6, §7, §8, §11): il codice di valutazione chiesto dal coordinatore il
19/11/2026 («Pubblichi il codice di valutazione prima di leggere il test»), da pubblicare prima di qualunque lettura del
test. Legge le cartelle dei due run di scripts/pad_arch_run.py (nuaa_patience_4 e nuaa_patience_12, 24 modelli ciascuno),
assembla una sola volta i frame del test di ciascun fold, li valuta con i 48 checkpoint nei due stati della BN e calcola
tutto ciò che PROTOCOL.md §6 definisce, solo attraverso le funzioni congelate di pad_arch_definitions (D):
  - l'analisi primaria (D.primary_analysis: patience 4, soglia 0,5; i due divari di ACER con BN ordinaria e ristimata,
    il contrasto Δ(ristimata) − Δ(ordinaria) con l'intervallo per soggetto e la lettura, i valori per seed a parte);
  - le 8 condizioni (D.compare_condition: Δ = ACER(ResNet-18) − ACER(LivenessNet) con APCER e BPCER per unità, pareggi
    come frazioni esatte, medie, conteggi dei segni, bootstrap per soggetto, per seed, lettura per seed);
  - il cambio di segno fra condizioni con il contrasto appaiato (D.compare_conditions) sulle 12 coppie che differiscono
    in un solo fattore (scelta dell'autore);
  - la sensibilità ai soggetti (D.leave_one_subject_out) per condizione;
  - i conteggi «one-class» sul dev (D.one_class_run, D.one_class_counts), confrontati con public/one_class.csv dei run, e a
    parte il conteggio «degenere» di prima (nuaa_bn_recalibration.group_of, regola del gruppo «ten» della PR #27).

Controllo chiesto dal coordinatore (19/11/2026, C2): «Per l'analisi primaria aggiunga un controllo che la interrompa se la
soglia effettiva è diversa da 0,5 per una delle reti o se manca una delle 12 combinazioni di fold e seed in uno dei due
stati BN.» Applicato in due punti (scelta dell'autore, come):
  (a) check_plan, prima che sia usata qualunque lista del test o assemblato qualunque frame del test: sul piano di
      valutazione (le voci rete × fold × seed × stato della BN × regola della soglia costruite dalle cartelle dei run
      validate), per patience 4 e ciascuno dei due stati della BN, esattamente le 12 combinazioni fold × seed per ciascuna
      rete (nessuna mancante, duplicata o in più) e la soglia che sarà applicata nelle condizioni a soglia fissa uguale
      a 0,5 per ciascuna rete;
  (b) check_units, subito prima di D.primary_analysis, sulle unità passate a D.primary_analysis (gli stessi oggetti): le
      stesse due condizioni sui valori effettivi. La soglia effettiva è il campo «threshold» di ogni rete di ogni unità,
      cioè il valore che D.prepare_units passa a D.subject_counts per derivare le decisioni che entrano nelle metriche:
      una sola fonte per la decisione e per il controllo (il valore viene dal piano, a sua volta da
      D.condition_threshold, mai da una costante riletta).
Se un controllo non passa l'analisi primaria non è calcolata e il comando termina FAIL senza uscite di risultati. La
completezza delle 12 combinazioni vale per tutte le 8 condizioni e la soglia 0,5 per le quattro a soglia fissa (scelta
dell'autore); la soglia EER non ha controllo di valore (è per modello e per stato).

Sequenza di evaluate (un solo processo, in quest'ordine):
  0. prima di creare la cartella: configurazione a insieme chiuso; file fissati (PROTOCOL.md e plan/run_plan.csv della
     preregistrazione, uguali alle righe del SHA256SUMS del pacchetto; configurazione dei run, validata da
     pad_arch_run.load_config, e ciò che pad_arch_run.load_pins legge; codice); run_plan.csv uguale al file rigenerato;
     tetto ricavato dal piano; variabili dei thread uguali al record della prova (pad_arch_run.thread_environment_rule);
     regola della macchina (pad_arch_run.machine_rule); radice d'uscita;
     le due cartelle dei run (load_run: PASS; ufficiali e in results/archrun per una valutazione ufficiale, né ufficiali
     né in results/archrun per una non ufficiale, che inoltre non usa il repository come base degli ingressi: unofficial_inputs_rule,
     prima di leggere qualunque file; configurazione e codice fissati,
     ogni file uguale al suo SHA256SUMS, 24 modelli completi con i sei file); albero git pulito; regola dei tentativi.
  1. status.json RUNNING e run.log; ingressi (pad_arch_comparison.load_inputs: nessuno schema dei fold); i file del dev
     dei 48 modelli contro i manifest (ID, soggetti, etichette, sessioni, checkpoint); soglie (D.condition_threshold) e
     piano di valutazione; controllo (a); conteggi one-class sul dev.
  2. TensorFlow importato e subito pad_arch_comparison.set_threads (10 e 10), confronto con la prova ufficiale
     (pad_arch_run.threads_rule, registrato), GPU nascosta; per ognuno dei 48 modelli, PRIMA di qualunque lista o frame
     del test: checkpoint ricaricato con lo SHA-256 verificato subito prima (nuaa_bn_recalibration.load_model) e
     controllato (pad_arch_run.check_loaded), dev valutato nello stato ordinario; copia in memoria con le statistiche
     ristimate salvate dal run reinstallate (nuaa_bn_test.install; nuaa_bn_recalibration.tensor_problems: solo i tensori
     mobili diversi e uguali alle statistiche salvate), dev valutato nello stato ristimato; ciascuno stato confrontato con
     i punteggi del dev salvati dal run (pad_arch_run.evaluator_agreement: scarto <= 1e-5, nessuna decisione discordante
     a 0,5 e alla soglia EER; tolleranze della configurazione della PR #27, quelle del runner). Un modello fuori: esito
     MISMATCH, il test non è letto.
  3. solo dopo le 48 riproduzioni: riverifica (final_checks: questa configurazione, la configurazione dei run e i suoi file
     fissati, il codice, PROTOCOL.md e run_plan.csv, e di ogni cartella dei run status.json, run.log, i due SHA256SUMS e
     ogni file elencato; gli ingressi di pad_arch_comparison.load_inputs, già in memoria, non sono riletti; un FAIL che non
     ha bisogno del test viene prima del test); poi schema dei fold (come
     nuaa_bn_test.load_inputs: FR.load_fold_manifests con tutti i fold 1..k, SR.check_schema_config, SR.check_sources); numero dei frame del test per fold uguale al piano; test disgiunto dal
     fitting e dal dev del fold; per ciascun fold i frame del test assemblati una volta (FR.assemble, in ordine di chiave)
     e valutati dai suoi 12 modelli (due run × tre seed × due reti) nei due stati della BN (solo probabilità e logit:
     nessuna metrica del test calcolata qui).
  4. di nuovo la stessa riverifica e la scansione del run.log scritto fin qui (tutto ciò che può fallire senza i
     numeri, prima dei numeri); unità delle 8 condizioni con l'ID di ogni frame; controllo (b); D.primary_analysis; le
     altre condizioni (stesso controllo di completezza: una secondaria incompleta scarta anche l'analisi primaria già
     calcolata); confronti fra condizioni; sensibilità ai soggetti; scansione dei testi pubblici; uscite. Nessun numero
     dell'analisi va nel log: dopo le uscite il run.log dice solo che sono scritte; «analysis computed» e il RESULT sono
     stampati solo dopo lo status.json finale.
Tetto: le ore di tempo reale della configurazione (3,5), ricavate dal piano con la sua regola (cap_from_plan), controllate
(cap.checks) prima della riproduzione del dev di ogni modello, prima della riverifica che precede le liste del test, prima
dell'assemblaggio del test di ogni fold, prima della valutazione del test di ogni modello e prima dell'analisi (l'analisi,
solo NumPy, non è interrotta); SIGUSR1 ferma la valutazione al controllo successivo; al tetto, o con SIGUSR1:
INCOMPLETE, nessun risultato. Nessuna ripresa: un nuovo tentativo ripete tutto in una cartella nuova.
Esiti: PASS (0); FAIL (1: nessuna uscita di risultati, tutto o niente: dopo la lettura del test anche i punteggi riservati
del test sono tolti; restano status.json con la ragione e con ciò che è stato scritto e tolto, e run.log, che non contiene
metriche del test); INCOMPLETE (2);
MISMATCH (4: riproduzione del dev non riuscita per almeno un modello, documentata per modello in status.json; il test non
è letto).

Uscite (results/archeval/YYYYMMDD-ARCHEVAL-<commit>[-r<k>]): status.json, run.log; public/ results.json, report.md,
conditions.csv, units.csv, reproduction.csv, SHA256SUMS (nessun identificativo di frame o di soggetto: conteggi e
SHA-256); restricted/ p4|p12_<rete>_fold<k>_seed<s>_test_scores.csv (per frame, con sample_id, nei due stati della BN),
loso_values.csv (per condizione e soggetto escluso: valore, segno, denominatori), SHA256SUMS.

  .venv/bin/python -I -B scripts/pad_arch_eval.py check-inputs [--run-p4 results/archrun/<id> --run-p12 results/archrun/<id>]
  nohup caffeinate -dims .venv/bin/python -I -B scripts/pad_arch_eval.py evaluate --run-p4 results/archrun/<id> --run-p12 results/archrun/<id> \
      --i-am-authorized > <log fuori dal repository> 2>&1 &
(nohup senza redirezione scrive nohup.out nella radice del repository e l'albero non è più pulito; caffeinate impedisce
che la macchina dorma: il tetto conta il tempo reale.)

Riuso per importazione (una sola definizione per regola): pad_arch_definitions (D), pad_arch_run (R: configurazione e
file fissati dei run, regola della macchina e dei thread, check_loaded, evaluator_agreement, unit_frames, radice
d'uscita, percorsi nel comando, status.json, pulizia dopo un errore, riverifica finale), nuaa_bn_test (BT.install),
nuaa_bn_recalibration (BR: caricamento, forward, istantanee dei pesi, tensor_problems), pad_arch_comparison (PAC:
ingressi, thread, macchina, piano, scansione delle uscite), nuaa_folds_run (FR), nuaa_session_ablation_run (SR),
m2_writers e path_guard. Dipendenze: NumPy; TensorFlow 2.15 (solo evaluate, solo CPU). Formati: configurazione
pad-arch-eval-config/1, risultati pad-arch-eval/1.
"""
import argparse
import gc
import json
import math
import os
import re
import shlex
import sys
import time
import warnings
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import m2_writers as mw  # noqa: E402
import nuaa_bn_recalibration as BR  # noqa: E402
import nuaa_bn_test as BT  # noqa: E402
import nuaa_folds_run as FR  # noqa: E402
import nuaa_session_ablation_run as SR  # noqa: E402
import pad_arch_comparison as PAC  # noqa: E402
import pad_arch_definitions as D  # noqa: E402
import pad_arch_models as AM  # noqa: E402
import pad_arch_run as R  # noqa: E402
import path_guard as pg  # noqa: E402

Failure, fail = FR.Failure, FR.fail
sha256_bytes, sha256_file, read_regular, rel_path, csv_text, parse_csv = (FR.sha256_bytes, FR.sha256_file, FR.read_regular,
                                                                          FR.rel_path, FR.csv_text, FR.parse_csv)
now = FR.now

DEFAULT_CONFIG = "delivery/archeval/pad_arch_eval.json"
DEFAULT_OUT_ROOT = "results/archeval"
RUNS_ROOT = R.DEFAULT_OUT_ROOT                     # results/archrun: le cartelle dei run ufficiali
CONFIG_SCHEMA = "pad-arch-eval-config/1"
EVAL_SCHEMA = "pad-arch-eval/1"
EVAL_ID_RE = re.compile(r"^\d{8}-ARCHEVAL-[0-9a-f]{7,40}(?:-r([2-9]|[1-9]\d+))?$")
LOOKS_LIKE_EVAL = re.compile(r"^\d{8}-ARCHEVAL-", re.IGNORECASE)
THIS_FILE = "scripts/pad_arch_eval.py"
RUN_FILE = R.THIS_FILE                             # scripts/pad_arch_run.py
CODE_FILES = R.CODE_FILES + (RUN_FILE,)
RUN_OPTION = {"nuaa_patience_4": "run_p4", "nuaa_patience_12": "run_p12"}
PACKAGE_ENTRIES = {"protocol": "PROTOCOL.md", "run_plan": "plan/run_plan.csv"}
TOP_KEYS = ("schema", "task", "authorisation", "operations", "preregistration", "run_configuration", "code", "runs", "run_checks",
            "primary_check", "condition_checks", "bn_states", "dev_reproduction", "thresholds", "test_scoring", "comparisons",
            "conclusions", "cap", "after_stop", "attempts", "machine_rule", "late_failures", "degenerate_apart", "outputs", "report",
            "choices_of_the_author", "test")
CHOICE_KEYS = ("primary_check_where", "condition_checks", "bn_state_at_test", "dev_reproduction", "thresholds_source", "missing_class",
               "test_scoring", "pairs", "conclusions", "cap", "after_stop", "attempts", "official_runs", "public_subject_sensitivity",
               "fail_after_test_read", "secondary_incomplete", "re_verification_before_test", "degenerate_apart", "outputs")
STATUS_CODE = {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2, "MISMATCH": 4}
STATUS, LOG, PUBLIC, RESTRICTED = FR.STATUS, FR.LOG, "public", "restricted"
PUBLIC_OUTPUTS = ("results.json", "report.md", "conditions.csv", "units.csv", "reproduction.csv", "SHA256SUMS")
TEST_SUFFIX = "_test_scores.csv"
RESTRICTED_FIXED = ("loso_values.csv", "SHA256SUMS")
CAP_UNIT = R.CAP_UNIT
DATASET, ARM = R.DATASET, R.ARM
PRIMARY = (D.PRIMARY_ORDINARY, D.PRIMARY_RE_ESTIMATED)
REQUIRED_THRESHOLD = 0.5                           # il valore che il coordinatore chiede per l'analisi primaria
TEST_SCORE_COLUMNS = ("run", "network", "fold", "train_seed", "patience", "bn_state", "role", D.FRAME_ID_FIELD, "subject_id", "session_id",
                      "label", "score_attack", "logit_bona_fide", "logit_attack", "prediction_05", "threshold_dev_eer", "prediction_dev_eer",
                      "file_sha256", "checkpoint_sha256")
CONDITION_COLUMNS = ("patience", "bn_state", "threshold", "label", "scope", "units", "n_subjects", "mean_delta_acer", "low_acer", "high_acer",
                     "valid_resamples_acer", "mean_delta_apcer", "low_apcer", "high_apcer", "mean_delta_bpcer", "low_bpcer", "high_bpcer",
                     "resnet18_lower", "livenessnet_lower", "ties", "undefined", "sign", "sign_reason")
UNIT_COLUMNS = (("patience", "bn_state", "threshold", "fold", "seed", "n_frames", "frame_ids_sha256")
                + tuple(f"{n.lower().replace('-', '')}_{k}" for n in AM.NETWORKS for k in ("threshold", "apcer", "bpcer", "acer"))
                + ("delta_apcer", "delta_bpcer", "delta_acer", "delta_acer_value", "tie"))
REPRODUCTION_COLUMNS = ("run", "network", "fold", "seed", "bn_state", "n_dev", "max_abs_diff", "discordant_05", "discordant_eer", "tolerance",
                        "discordant_tolerance", "softmax_gap", "pass")
LOSO_COLUMNS = ("patience", "bn_state", "threshold", "excluded_subject", "mean_delta_acer", "sign", "sign_differs", "n_attack_total",
                "n_bona_fide_total")
DEGENERATE_COLUMNS = ("network", "patience", "degenerate", "trainings")
CONCLUSIONS_RE = re.compile(r'Conclusions, written before the test: if the interval excludes zero, "([^"]+)"; otherwise "([^"]+)"')
CONCLUSION_ALTERNATIVE = "lower [higher]"
EXTRA_PROTECTED = R.EXTRA_PROTECTED + (RUNS_ROOT,)
OFFICIAL_RUNS_DIR = ROOT / RUNS_ROOT                # la radice dei run ufficiali, riconosciuta per identità dei file
TEST_SENTENCE = "the NUAA test is read once, by evaluate, after every check that does not need it"
TEST_READ = (
    f"{TEST_SENTENCE}. What each command reads. check-inputs: this configuration; PROTOCOL.md and plan/run_plan.csv of the "
    "pre-registration (each equal to its line of the SHA256SUMS of the delivered package); the configuration of the two runs "
    "(validated by pad_arch_run.load_config) and what pad_arch_run.load_pins reads with it (the configuration of the "
    "pre-registration, the code, the status.json and trial.json of the official trial and the package SHA256SUMS, run_order.csv and "
    "run_caps.csv, the configurations of PRs #27 and #28); the code files pinned here and this file; what "
    "pad_arch_comparison.load_inputs reads (its TEST_READ: the five M2 manifests, read to index the cache, the cache and the "
    "extracted frames file, loaded whole, which also hold frames of the held-out subjects, the cells run, the folds run without "
    "its public files, the LivenessNet architecture file, the status.json and restricted SHA256SUMS of the two accepted runs and one "
    "training card); with --run-p4 and --run-p12, of each run folder: status.json, run.log (bytes, for its SHA-256), the public "
    "SHA256SUMS and every file it lists (summary.json and one_class.csv parsed, the others as bytes), the restricted SHA256SUMS and every file it "
    "lists (the checkpoints and the phase-1 dev scores as bytes, for the SHA-256; per model the card, the BN statistics and the two "
    "dev score files parsed); no fold schema, no test frame, TensorFlow not imported, nothing written. evaluate, in this order: the "
    "same files; the dev frames of each fold assembled (no other role) and every checkpoint loaded and scored on the dev in both BN "
    "states; only after the 48 dev reproductions have passed and after a re-verification (this configuration; the configuration of the runs "
    "and the files it pins, through pad_arch_run.final_checks; the code; PROTOCOL.md and run_plan.csv; of each run folder status.json, "
    "run.log, the two SHA256SUMS and every file they list; the inputs of pad_arch_comparison.load_inputs, already in memory, are not read "
    "again), the fold schema of the accepted folds run (fold<k>_fitting.csv, "
    "fold<k>_dev.csv and fold<k>_test.csv, loaded and checked as nuaa_bn_test.load_inputs does: every fold 1..k, the approved schema "
    "configuration, every row found once in the cache or in the extracted frames) and then, fold by fold, the test "
    "frames of the fold assembled once (every frame of its held-out subjects, from the cache and the extracted frames, in key "
    "order) and scored by the 12 models of the fold in the two BN states (attack probabilities and logits only: no test metric is "
    "computed while scoring); then the same re-verification is repeated, the checks requested by the coordinator are applied to the units, "
    "and only then the test metrics are computed, the primary analysis first. No aggregate.json, run_record.json or "
    "<model>_test_scores.csv of any earlier run is opened")

# punti di iniezione (solo i test)
SCORE_MODEL = None      # (ctx, rec, parts) -> {stato: {ruolo: {"p", "z", "softmax_gap"}}} (default: score_model)
PLAN_HOOK = None        # (piano) -> piano: alterazioni del piano dopo la sua costruzione (default: nessuna)
UNITS_HOOK = None       # (condizioni) -> condizioni: alterazioni delle unità prima dei controlli (default: nessuna)


# ============================================================================ configurazione

def condition_pairs() -> list:
    """Le coppie di condizioni dei confronti secondari (scelta dell'autore): le 12 coppie che differiscono in un solo
    fattore, nell'ordine stato della BN, soglia, patience; in ogni coppia la prima condizione ha il secondo livello del
    fattore (ristimata, EER del dev, patience 12) e la seconda il primo: contrasto = Δ(prima) − Δ(seconda), il verso del
    contrasto primario. La coppia della BN a patience 4 e soglia 0,5 è il contrasto primario."""
    levels = (D.PATIENCES, D.BN_STATES, D.THRESHOLD_RULES)
    out = []
    for k in (1, 2, 0):
        for c in D.CONDITIONS:
            if c[k] == levels[k][0]:
                a = tuple(levels[k][1] if i == k else v for i, v in enumerate(c))
                out.append((a, c))
    return out


def _pairs_json() -> list:
    return [[list(a), list(b)] for a, b in condition_pairs()]


def load_config(p: Path) -> tuple:
    """(configurazione, SHA-256 dei byte letti). Insieme chiuso a ogni livello e valori fissati controllati qui, prima di
    leggere qualunque altro file; nessun valore predefinito."""
    data = read_regular(p, "configuration")
    cfg = SR.read_json(data, f"configuration {p.name}")
    if cfg.get("schema") != CONFIG_SCHEMA:
        fail(f"configuration {p.name}: schema must be {CONFIG_SCHEMA!r}")
    if sorted(cfg) != sorted(TOP_KEYS):
        fail(f"configuration: fields must be exactly {', '.join(TOP_KEYS)} (missing {sorted(set(TOP_KEYS) - set(cfg))}, "
             f"unknown {sorted(set(cfg) - set(TOP_KEYS))})")
    for k in ("task", "authorisation", "operations", "run_checks", "report", "thresholds", "late_failures"):
        R._text(cfg[k], k)
    if "published" not in cfg["authorisation"] or "before the test" not in cfg["authorisation"]:
        fail("configuration: authorisation must say that this code is published before the test is read")
    if "nohup" not in cfg["operations"] or "commit" not in cfg["operations"]:
        fail("configuration: operations must state the commit before the command and the launch with nohup")
    if cfg["test"] != TEST_READ:
        fail("configuration: test must state exactly what is read (TEST_READ of the script)")
    pr = cfg["preregistration"]
    R._keys(pr, list(PACKAGE_ENTRIES), "preregistration")
    for k in PACKAGE_ENTRIES:
        R._pin(pr[k], f"preregistration.{k}")
    R._pin(cfg["run_configuration"], "run_configuration")
    R._keys(cfg["code"], list(CODE_FILES), "code")
    if not all(SR._hex(v) for v in cfg["code"].values()):
        fail("configuration: code SHA-256 values malformed")
    if cfg["runs"] != list(R.RUNS):
        fail(f"configuration: runs must be {list(R.RUNS)} (the two runs of the plan, in its order)")
    if "SHA256SUMS" not in cfg["run_checks"] or "PASS" not in cfg["run_checks"]:
        fail("configuration: run_checks must state the checks of the run folders (status PASS, every file against its SHA256SUMS)")
    pc = cfg["primary_check"]
    R._keys(pc, ["effective_threshold", "requested", "where"], "primary_check")
    R._text(pc["requested"], "primary_check.requested")
    if "requested by the coordinator" not in pc["requested"]:
        fail("configuration: primary_check.requested must quote the check requested by the coordinator")
    R._author(pc["where"], "primary_check.where")
    if "before any test frame" not in pc["where"] or "primary_analysis" not in pc["where"]:
        fail("configuration: primary_check.where must state the two places of the check (before any test frame, before primary_analysis)")
    R._author(pc["effective_threshold"], "primary_check.effective_threshold")
    R._author(cfg["condition_checks"], "condition_checks")
    bs = cfg["bn_states"]
    R._keys(bs, ["ordinary", "re_estimated"], "bn_states")
    R._text(bs["ordinary"], "bn_states.ordinary")
    R._author(bs["re_estimated"], "bn_states.re_estimated")
    if "re-installed" not in bs["re_estimated"]:
        fail("configuration: bn_states.re_estimated must state that the statistics saved by the run are re-installed")
    R._author(cfg["dev_reproduction"], "dev_reproduction")
    if "1e-5" not in cfg["dev_reproduction"] or "MISMATCH" not in cfg["dev_reproduction"]:
        fail("configuration: dev_reproduction must state the tolerance and the MISMATCH outcome")
    R._author(cfg["test_scoring"], "test_scoring")
    co = cfg["comparisons"]
    R._keys(co, ["pairs", "rule"], "comparisons")
    if co["pairs"] != _pairs_json():
        fail("configuration: comparisons.pairs must be the pairs of condition_pairs (one factor changed, the second level first)")
    R._author(co["rule"], "comparisons.rule")
    cc = cfg["conclusions"]
    R._keys(cc, ["rule", "source"], "conclusions")
    R._text(cc["source"], "conclusions.source")
    R._author(cc["rule"], "conclusions.rule")
    cp = cfg["cap"]
    R._keys(cp, ["checks", "derivation", "hours", "on_cap", "unit"], "cap")
    if isinstance(cp["hours"], bool) or not isinstance(cp["hours"], (int, float)) or not math.isfinite(cp["hours"]) or cp["hours"] <= 0:
        fail("configuration: cap.hours must be a finite positive number")
    if cp["unit"] != CAP_UNIT:
        fail(f"configuration: cap.unit must be {CAP_UNIT!r}")
    R._author(cp["derivation"], "cap.derivation")
    R._text(cp["checks"], "cap.checks")
    R._text(cp["on_cap"], "cap.on_cap")
    if "INCOMPLETE" not in cp["on_cap"]:
        fail("configuration: cap.on_cap must say that the evaluation at its cap stops as INCOMPLETE")
    R._author(cfg["after_stop"], "after_stop")
    if "no automatic resume" not in cfg["after_stop"] or "test_read_started" not in cfg["after_stop"] or "C2 card" not in cfg["after_stop"]:
        fail("configuration: after_stop must state that there is no automatic resume and that a new attempt after test_read_started is a "
             "second reading of the test, started only after the outcome has been reported on the C2 card")
    R._author(cfg["degenerate_apart"], "degenerate_apart")
    if "run.log" not in cfg["late_failures"]:
        fail("configuration: late_failures must state what remains on disk, run.log included")
    R._author(cfg["attempts"], "attempts")
    if "--unofficial-out-root" not in cfg["attempts"]:
        fail("configuration: attempts must state the rule of the attempts and of the unofficial output root")
    R._text(cfg["machine_rule"], "machine_rule")
    if "threads" not in cfg["machine_rule"] or "before any" not in cfg["machine_rule"]:
        fail("configuration: machine_rule must state the threads and that a difference stops the work before any evaluation")
    ou = cfg["outputs"]
    R._keys(ou, ["public", "restricted", "run"], "outputs")
    if ou["public"] != list(PUBLIC_OUTPUTS) or ou["restricted"] != [f"<run>_<model>{TEST_SUFFIX}"] + list(RESTRICTED_FIXED) \
            or ou["run"] != [STATUS, LOG]:
        fail("configuration: outputs must list the public, restricted and run files of the script")
    ch = cfg["choices_of_the_author"]
    R._keys(ch, list(CHOICE_KEYS), "choices_of_the_author")
    for k in CHOICE_KEYS:
        R._text(ch[k], f"choices_of_the_author.{k}")
    return cfg, sha256_bytes(data)


# ============================================================================ file fissati

class Pins:
    """Ciò che la configurazione fissa, letto e controllato: configurazione dei run (rcfg, rcfg_sha, rcfg_p) e i suoi
    file fissati (rpins di pad_arch_run.load_pins: preregistrazione, prova, file del piano, PR #27), codice, PROTOCOL.md
    (con le due conclusioni del §6), run_plan.csv, tetto ricavato dal piano, SHA-256 dei file letti (files)."""


def check_code(cfg: dict) -> dict:
    """Ogni file di CODE_FILES uguale allo SHA-256 fissato; più lo SHA-256 attuale di questo file."""
    got = {f: sha256_bytes(read_regular(ROOT / f, "code file")) for f in CODE_FILES}
    bad = [f for f in CODE_FILES if got[f] != cfg["code"][f]]
    if bad:
        fail(f"code files {bad} differ from the SHA-256 pinned in the configuration")
    return {**got, THIS_FILE: sha256_bytes(read_regular(Path(__file__), "code file"))}


def plan_text(cfg_pr: dict, trial: dict) -> str:
    """run_plan.csv rigenerato dalle funzioni della preregistrazione (PAC.plan_rows con il trial.json, PAC.cell)."""
    return csv_text(PAC.PLAN_COLUMNS, [[PAC.cell(r, k) for k in PAC.PLAN_COLUMNS] for r in PAC.plan_rows(cfg_pr, trial)])


def cap_from_plan(cfg_pr: dict, trial: dict) -> dict:
    """Il tetto della valutazione (scelta dell'autore) con la regola dei tetti del piano (plan.caps: massimo × wall_multiplier,
    arrotondato per eccesso al multiplo di round_up_hours, la stessa formula di PAC.run_caps) applicata alla stima della
    valutazione del piano: la somma, senza arrotondamenti, delle ore di valutazione (evaluation_wall_hours di PAC.plan_rows)
    delle righe NUAA, che il piano conta per tutti i 48 addestramenti (ristima della BN, dev e test nei due stati)."""
    ca = cfg_pr["plan"]["caps"]
    rows = [r for r in PAC.plan_rows(cfg_pr, trial) if r["dataset"] == cfg_pr["plan"]["datasets"][0]["name"]]
    if any(r["evaluation_wall_hours"] is None for r in rows) or sum(r["new_trainings"] for r in rows) != 48:
        fail("plan: the evaluation hours of the 48 NUAA trainings are not all estimated")
    est = sum(r["evaluation_wall_hours"] for r in rows)
    cap = math.ceil(round(est * ca["wall_multiplier"] / ca["round_up_hours"], 9)) * ca["round_up_hours"]
    return {"evaluation_hours_estimate": est, "wall_multiplier": ca["wall_multiplier"], "round_up_hours": ca["round_up_hours"], "cap_hours": cap,
            "rows": [{"network": r["network"], "patience": r["patience"], "evaluation_wall_hours": r["evaluation_wall_hours"]} for r in rows]}


def protocol_conclusions(text: str) -> dict:
    """Le due conclusioni scritte prima del test in PROTOCOL.md §6, citate dal file fissato (spazi normalizzati), non
    riscritte: la prima con l'alternativa «lower [higher]», che deve comparire una volta sola."""
    m = CONCLUSIONS_RE.search(" ".join(text.split()))
    if not m or m.group(1).count(CONCLUSION_ALTERNATIVE) != 1:
        fail("PROTOCOL.md: the two conclusions of §6 written before the test are not found in the expected form")
    return {"excludes_zero": m.group(1), "otherwise": m.group(2)}


def conclusion_for(conclusions: dict, reading: str) -> str:
    """La conclusione di PROTOCOL.md §6 per la lettura del contrasto primario (D.contrast_reading): intervallo tutto
    sotto zero -> la prima con «lower», tutto sopra -> la prima con «higher», altrimenti la seconda."""
    if reading == D.CONTRAST_LOWER:
        return conclusions["excludes_zero"].replace(CONCLUSION_ALTERNATIVE, "lower")
    if reading == D.CONTRAST_HIGHER:
        return conclusions["excludes_zero"].replace(CONCLUSION_ALTERNATIVE, "higher")
    if reading == D.CONTRAST_UNRESOLVED:
        return conclusions["otherwise"]
    fail(f"unknown reading of the primary contrast {reading!r}")


def load_pins(cfg: dict, base: Path) -> Pins:
    """I file fissati: configurazione dei run (SHA-256, R.load_config) e quelli che R.load_pins legge con essa; codice
    (anche uguale a quello fissato nella configurazione dei run); PROTOCOL.md e run_plan.csv con il loro SHA-256 e uguali
    alle righe del SHA256SUMS del pacchetto consegnato (fissato dalla configurazione dei run); run_plan.csv uguale al file
    rigenerato; tetto uguale a cap_from_plan; le due conclusioni del §6 trovate."""
    pins = Pins()
    pins.rcfg_p, rb = R.read_pinned(base, cfg["run_configuration"], "run configuration")
    pins.rcfg, pins.rcfg_sha = R.load_config(pins.rcfg_p)
    pins.rpins = R.load_pins(pins.rcfg, base)
    pins.cfg_pr, pins.pcfg, pins.trial, pins.bn_cfg = pins.rpins.cfg_pr, pins.rpins.pcfg, pins.rpins.trial, pins.rpins.bn_cfg
    pins.code = check_code(cfg)
    bad = [f for f in R.CODE_FILES if cfg["code"][f] != pins.rcfg["code"][f]] + \
        ([RUN_FILE] if cfg["code"][RUN_FILE] != pins.rpins.code[RUN_FILE] else [])
    if bad:
        fail(f"code files {bad}: the SHA-256 pinned here differ from the code of the run configuration")
    pins.run_code = {**{f: pins.rcfg["code"][f] for f in R.CODE_FILES}, RUN_FILE: cfg["code"][RUN_FILE]}
    _, sb = R.read_pinned(base, pins.rcfg["trial_record"]["package_sums"], "package SHA256SUMS")
    sums = FR.read_sums(sb, "package SHA256SUMS")
    pins.files = {"run_configuration": pins.rcfg_sha, **pins.rpins.files}
    texts = {}
    for k, entry in PACKAGE_ENTRIES.items():
        _, b = R.read_pinned(base, cfg["preregistration"][k], f"pre-registration {entry}")
        if sums.get(entry) != cfg["preregistration"][k]["sha256"]:
            fail(f"pre-registration {entry}: SHA-256 differs from its line of the delivered package SHA256SUMS")
        texts[k] = b.decode("utf-8", "replace")
        pins.files[k] = cfg["preregistration"][k]["sha256"]
    if texts["run_plan"] != plan_text(pins.cfg_pr, pins.trial):
        fail("run_plan.csv differs from the plan regenerated from the pre-registration and the trial (PAC.plan_rows)")
    pins.cap = cap_from_plan(pins.cfg_pr, pins.trial)
    if cfg["cap"]["hours"] != pins.cap["cap_hours"]:
        fail(f"cap.hours {cfg['cap']['hours']} differs from the cap derived from the plan ({pins.cap['cap_hours']})")
    pins.conclusions = protocol_conclusions(texts["protocol"])
    nuaa = pins.cfg_pr["plan"]["datasets"][0]
    pins.folds, pins.seeds = list(nuaa["splits"]), list(nuaa["seeds"])
    if tuple(pins.seeds) != D.PLAN_SEEDS or pins.folds != list(range(1, len(pins.folds) + 1)):
        fail("pre-registration: the seeds or folds of the NUAA plan differ from the definitions")
    pins.test_per_fold = list(nuaa["frames_per_unit"]["test_per_split"])
    return pins


# ============================================================================ cartelle dei run

def _float32_list(v, what: str) -> np.ndarray:
    """Una lista di statistiche installate: numeri finiti già rappresentabili esattamente in float32."""
    if not isinstance(v, list) or not v or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in v):
        fail(f"{what}: a non-empty list of numbers is required")
    a = np.array(v, np.float64)
    if not np.all(np.isfinite(a)) or not np.array_equal(a.astype(np.float32).astype(np.float64), a):
        fail(f"{what}: values not finite or not float32 values")
    return a.astype(np.float32)


def load_run(folder: Path, run: str, pins: Pins, mode: str) -> dict:
    """Una cartella di run di pad_arch_run.py, controllata prima di ogni uso: nome = run id del run giusto; status.json
    PASS, completo (nessun arresto, nessun modello mancante, accordo dei valutatori per tutti), con la configurazione e
    il codice fissati qui, albero pulito, regola della macchina e dei thread rispettata (10 e 10); ufficiale (e sotto
    results/archrun) per una valutazione ufficiale (mode «official»); per una valutazione non ufficiale (mode «unofficial»,
    i test del comando) né un run ufficiale né una cartella dentro results/archrun (identità dei file; i limiti sono
    dichiarati in unofficial_inputs_rule); check-inputs (mode «inputs») accetta
    entrambi; run.log, SHA256SUMS pubblico e riservato con gli SHA-256 dello stato;
    ogni file elencato uguale alla sua riga (i checkpoint compresi), esattamente le uscite pubbliche del runner e i sei
    file di ciascuno dei 24 modelli; summary.json coerente; per modello scheda (checkpoint, rete, fold, seed, patience,
    posizione; epoca ripristinata, accuratezza e val_loss del dev al checkpoint, per il conteggio «degenere») e statistiche ristimate (ordine dei 20 o 5 strati, valori float32 finiti e varianze non negative, accordo dei
    valutatori). Restituisce i record dei modelli nell'ordine del piano; i file del dev sono letti dopo, con i manifest."""
    m = R.RUN_ID_RE.match(folder.name)
    if not m or m.group(1) != str(R.RUNS[run]):
        fail(f"--{RUN_OPTION[run].replace('_', '-')}: {folder.name} is not the folder of a run {run} (YYYYMMDD-ARCHRUN-{R.RUN_TAG[run]}-<commit>)")
    if not folder.is_dir() or folder.is_symlink():
        fail(f"run folder {folder.name} missing, not a folder or a symbolic link")
    sb = read_regular(folder / STATUS, f"{folder.name} status.json")
    st = SR.read_json(sb, f"{folder.name} status.json")
    what = f"run {folder.name}"
    if st.get("status") != "PASS" or st.get("passed") is not True or st.get("run_id") != folder.name or st.get("run") != run:
        fail(f"{what}: status.json is not a PASS of {run} with this run id")
    if st.get("config_sha256") != pins.rcfg_sha:
        fail(f"{what}: produced with another configuration of the runs (config_sha256 differs from the pinned run configuration)")
    if st.get("code_sha256") != pins.run_code:
        fail(f"{what}: produced with other code (code_sha256 differs from the code pinned here)")
    if st.get("dirty") is not False:
        fail(f"{what}: produced from a work tree that was not clean")
    inside = pg.within(folder, OFFICIAL_RUNS_DIR)
    if mode == "official" and (st.get("official") is not True or not inside):
        fail(f"{what}: an official evaluation needs an official run in {RUNS_ROOT}")
    if mode == "unofficial" and st.get("official") is not False:
        fail(f"{what}: an official run is never read by an unofficial evaluation (--unofficial-out-root)")
    if mode == "unofficial" and inside:
        fail(f"{what}: a folder in {RUNS_ROOT} is never read by an unofficial evaluation (--unofficial-out-root)")
    if mode not in ("official", "unofficial", "inputs"):
        fail(f"unknown mode {mode!r} of the run folders")
    if st.get("stopped") is not None or st.get("missing_phase_1") != [] or st.get("evaluator_mismatches") != []:
        fail(f"{what}: stopped, with a model missing or with an evaluator mismatch")
    positions = pins.rpins.positions[run]
    names = [PAC.training_name(p["network"], p["fold"], p["seed"]) for p in positions]
    if st.get("complete_phase_1") != names or st.get("complete_phase_2") != names:
        fail(f"{what}: the complete models differ from the {len(names)} positions of the plan")
    if (st.get("machine_rule") or {}).get("equal") is not True or (st.get("threads_rule") or {}).get("equal") is not True:
        fail(f"{what}: the machine and thread rule was not met")
    th = (st.get("threads") or {}).get("effective")
    want = {k: pins.cfg_pr["training"]["threads"][k] for k in PAC.THREAD_KEYS}
    if th != want:
        fail(f"{what}: effective threads {th}, the pre-registration requires {want}")
    if sha256_file(FR.regular_path(folder / LOG, f"{folder.name} run.log")) != st.get("run_log_sha256"):
        fail(f"{what}: run.log differs from run_log_sha256 of its status.json")
    sums = {}
    for sub, key, want_files in ((PUBLIC, "public_sha256sums_sha256", [n for n in R.PUBLIC_OUTPUTS if n != "SHA256SUMS"]),
                                 (RESTRICTED, "restricted_sha256sums_sha256", [f for n in names for f in R.model_files(n)])):
        b = read_regular(folder / sub / "SHA256SUMS", f"{folder.name} {sub} SHA256SUMS")
        if sha256_bytes(b) != st.get(key):
            fail(f"{what}: the {sub} SHA256SUMS differs from {key} of its status.json")
        sums[sub] = FR.read_sums(b, f"{folder.name} {sub} SHA256SUMS")
        if sorted(sums[sub]) != sorted(want_files):
            fail(f"{what}: the {sub} SHA256SUMS does not list exactly the {len(want_files)} {sub} files of the runner")
        for fn, h in sums[sub].items():
            if sha256_file(FR.regular_path(folder / sub / fn, f"{folder.name} {fn}")) != h:
                fail(f"{what}: {sub}/{fn} differs from its SHA256SUMS line")
    summ = SR.read_json(read_regular(folder / PUBLIC / "summary.json", "summary.json"), f"{folder.name} summary.json")
    if (summ.get("schema"), summ.get("status"), summ.get("run_id"), summ.get("patience"), summ.get("complete_phase_2"), summ.get("code_sha256")) != \
            (R.RUN_SCHEMA, "PASS", folder.name, R.RUNS[run], len(names), pins.run_code) \
            or (summ.get("inputs_sha256") or {}).get("configuration") != pins.rcfg_sha:
        fail(f"{what}: summary.json is not the PASS summary of this run, configuration and code")
    orders = pins.rcfg["estimator"]["orders"]
    recs = []
    for p, name in zip(positions, names):
        rd = folder / RESTRICTED
        card = SR.read_json(read_regular(rd / f"{name}_card.json", f"{name}_card.json"), f"{folder.name} {name}_card.json")
        wsha = sums[RESTRICTED][f"{name}.keras"]
        if (card.get("weights_sha256"), card.get("network"), card.get("fold"), card.get("train_seed"), card.get("early_stopping_patience"),
                card.get("run"), card.get("position")) != (wsha, p["network"], p["fold"], p["seed"], R.RUNS[run], run, p["position"]):
            fail(f"{what}: {name}_card.json differs from its position, patience or checkpoint")
        if not FR.is_count(card.get("restored_epoch")) or not all(D.finite_real(card.get(k)) for k in ("val_acc_at_best", "best_val_loss")):
            fail(f"{what}: {name}_card.json lacks restored_epoch, val_acc_at_best or best_val_loss")
        sd = SR.read_json(read_regular(rd / f"{name}_bn_statistics.json", f"{name}_bn_statistics.json"),
                          f"{folder.name} {name}_bn_statistics.json")
        order = orders[p["network"]]
        if (sd.get("model"), sd.get("network"), sd.get("weights_sha256"), sd.get("order")) != (name, p["network"], wsha, order) \
                or list(sd.get("layers") or {}) != order:
            fail(f"{what}: {name}_bn_statistics.json differs from the model, its checkpoint or the order of its BN layers")
        if (sd.get("evaluator_agreement") or {}).get("pass") is not True:
            fail(f"{what}: {name}_bn_statistics.json records no agreement of the evaluators")
        installed = {}
        for n in order:
            lay = sd["layers"][n]
            mean = _float32_list(lay.get("installed_moving_mean_float32"), f"{what}: {name} {n} mean")
            var = _float32_list(lay.get("installed_moving_variance_float32"), f"{what}: {name} {n} variance")
            if mean.shape != var.shape or not np.all(var >= 0):
                fail(f"{what}: {name} {n}: installed mean and variance of different lengths, or a negative variance")
            installed[n] = (mean, var)
        recs.append({"run": run, "tag": R.RUN_TAG[run], "patience": R.RUNS[run], "position": p["position"], "network": p["network"],
                     "fold": p["fold"], "seed": p["seed"], "name": name, "key": f"{R.RUN_TAG[run]}_{name}", "weights": rd / f"{name}.keras",
                     "weights_sha256": wsha, "order": order, "installed": installed,
                     "card": {k: card[k] for k in ("restored_epoch", "val_acc_at_best", "best_val_loss")},
                     "dev_files": {s: (rd / f"{name}_dev_{s}.csv", sums[RESTRICTED][f"{name}_dev_{s}.csv"]) for s in D.BN_STATES}})
    return {"run": run, "folder": folder, "run_id": folder.name, "one_class_csv": read_regular(folder / PUBLIC / "one_class.csv", "one_class.csv"), "status_sha256": sha256_bytes(sb), "official": st.get("official") is True,
            "sums": sums, "run_log_sha256": st["run_log_sha256"], "public_sums_sha256": st["public_sha256sums_sha256"], "restricted_sums_sha256": st["restricted_sha256sums_sha256"],
            "estimator_check": (st.get("estimator_check") or {}).get("run_id"), "models": recs}


def recheck_run(r: dict) -> None:
    """Riverifica di una cartella di run (prima del test e prima dei risultati): status.json, run.log, i due SHA256SUMS e
    ogni file elencato con gli stessi SHA-256 dell'inizio."""
    folder = r["folder"]
    if sha256_bytes(read_regular(folder / STATUS, "status.json")) != r["status_sha256"]:
        fail(f"run {r['run_id']}: status.json changed during the evaluation")
    if sha256_file(FR.regular_path(folder / LOG, "run.log")) != r["run_log_sha256"]:
        fail(f"run {r['run_id']}: run.log changed during the evaluation")
    for sub in (PUBLIC, RESTRICTED):
        if FR.read_sums(read_regular(folder / sub / "SHA256SUMS", "SHA256SUMS"), "SHA256SUMS") != r["sums"][sub]:
            fail(f"run {r['run_id']}: the {sub} SHA256SUMS changed during the evaluation")
        for fn, h in r["sums"][sub].items():
            if sha256_file(FR.regular_path(folder / sub / fn, fn)) != h:
                fail(f"run {r['run_id']}: {sub}/{fn} changed during the evaluation")


def load_dev(rec: dict, inp) -> None:
    """I due file del dev di un modello (SHA-256 riverificato) contro i manifest: righe del dev comune del fold in ordine
    di chiave (ID, soggetto, sessione, etichetta, SHA-256 del file), rete, fold, seed, stato e checkpoint giusti; punteggi
    e logit finiti; prediction_05 con la regola del progetto (D.attack_decisions). Soglie dal file stesso con le funzioni
    della preregistrazione: D.condition_threshold("fixed_0.5") e D.condition_threshold("dev_eer") sui punteggi salvati
    (scelta dell'autore: le soglie del run, fissate prima del test), controllate contro threshold_dev_eer e
    prediction_dev_eer di ogni riga. Un dev senza una classe -> Failure (scelta dell'autore: nessuna soglia EER)."""
    rows = sorted(BR.role_rows(inp, "dev", ARM, rec["fold"]), key=lambda r: r["key"])
    rec["dev"], rec["thresholds"] = {}, {}
    for state in D.BN_STATES:
        p, h = rec["dev_files"][state]
        b = read_regular(p, p.name)
        if sha256_bytes(b) != h:
            fail(f"{rec['key']}: {p.name} changed since its SHA256SUMS was checked")
        got = parse_csv(b, R.SCORE_COLUMNS, f"{rec['key']} {p.name}")
        what = f"{rec['key']} dev {state}"
        if [g[D.FRAME_ID_FIELD] for g in got] != [r["key"] for r in rows]:
            fail(f"{what}: the frame identifiers are not those of the dev of fold {rec['fold']} in key order")
        if any((g["subject_id"], g["session_id"], g["label"], g["file_sha256"]) != (r["subject"], r["session"], r["label"], r["sha256"])
               for g, r in zip(got, rows)):
            fail(f"{what}: subject, session, label or file SHA-256 differ from the manifest")
        if any((g["network"], g["fold"], g["train_seed"], g["bn_state"], g["role"], g["checkpoint_sha256"]) !=
               (rec["network"], str(rec["fold"]), str(rec["seed"]), state, "dev", rec["weights_sha256"]) for g in got):
            fail(f"{what}: rows of another model, state, role or checkpoint")
        try:
            s = np.array([float(g["score_attack"]) for g in got], np.float64)
            z = np.array([[float(g["logit_bona_fide"]), float(g["logit_attack"])] for g in got], np.float64)
        except ValueError:
            fail(f"{what}: scores or logits not numeric")
        y, s = D.labels_scores(np.array([int(r["label"]) for r in rows]), s, what)
        if not np.all(np.isfinite(z)):
            fail(f"{what}: non-finite logits")
        if any(g["prediction_05"] != ("1" if d else "0") for g, d in zip(got, D.attack_decisions(s, FR.FIXED_THRESHOLD))):
            fail(f"{what}: prediction_05 differs from the rule of the project on the saved scores")
        if len(set(y.tolist())) != 2:
            fail(f"{what}: the dev lacks a class: no EER threshold (the evaluation needs both classes in every dev)")
        thr = {rule: D.condition_threshold(rule, y, s) for rule in D.THRESHOLD_RULES}
        if any(g["threshold_dev_eer"] != repr(thr["dev_eer"]) for g in got) or \
                any(g["prediction_dev_eer"] != ("1" if d else "0") for g, d in zip(got, D.attack_decisions(s, thr["dev_eer"]))):
            fail(f"{what}: threshold_dev_eer or prediction_dev_eer differ from the EER threshold of the saved scores")
        rec["dev"][state] = {"y": y, "s": s, "z": z, "subject": np.array([r["subject"] for r in rows]), "ids": [r["key"] for r in rows]}
        for rule in D.THRESHOLD_RULES:
            rec["thresholds"][(state, rule)] = thr[rule]


# ============================================================================ piano di valutazione e controlli chiesti

def evaluation_plan(recs: list) -> list:
    """Il piano di valutazione: una voce per modello × stato della BN × regola della soglia, con la soglia che sarà
    applicata (dal record del modello, cioè da D.condition_threshold). La voce è la fonte della soglia delle unità."""
    return [{"key": r["key"], "patience": r["patience"], "network": r["network"], "fold": r["fold"], "seed": r["seed"], "bn_state": s,
             "rule": t, "threshold": r["thresholds"][(s, t)]} for r in recs for s in D.BN_STATES for t in D.THRESHOLD_RULES]


def condition_label(cond: tuple) -> str:
    return "primary analysis" if cond in PRIMARY else f"condition {cond}"


def combination_problems(cond: tuple, combos: list, folds, seeds) -> list:
    """Le 12 combinazioni fold × seed di una condizione per ciascuna rete: mancanti, duplicate e in più, per rete, con
    lo stato della BN della condizione nel testo (nessun ID)."""
    got = Counter(combos)
    want = {(n, f, s) for n in AM.NETWORKS for f in folds for s in seeds}
    out = []
    for n in AM.NETWORKS:
        miss = sorted((f, s) for (m, f, s) in want - set(got) if m == n)
        dup = sorted((f, s) for (m, f, s), k in got.items() if m == n and k > 1)
        extra = sorted(((f, s) for (m, f, s) in set(got) - want if m == n), key=str)
        for label, v in (("missing", miss), ("duplicated", dup), ("not in the plan", extra)):
            if v:
                out.append(f"{len(v)} of the {len(folds) * len(seeds)} fold-seed combinations {label} for {n} in the {cond[1]} BN state "
                           f"(patience {cond[0]}, threshold {cond[2]}): " + ", ".join(f"fold {f} seed {s}" for f, s in v))
    other = sorted({m for (m, _, _) in got} - set(AM.NETWORKS))
    if other:
        out.append(f"networks outside the plan in the {cond[1]} BN state (patience {cond[0]}, threshold {cond[2]}): {other}")
    return out


def threshold_problems(cond: tuple, pairs: list) -> list:
    """Nelle condizioni a soglia fissa, la soglia di ogni rete deve essere esattamente 0,5 (un numero, non un booleano;
    un NaN o un altro valore -> problema), per rete, con il numero delle voci fuori."""
    if cond[2] != "fixed_0.5":
        return []
    out = []
    for n in AM.NETWORKS:
        bad = [t for m, t in pairs if m == n and (isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, float, np.integer, np.floating))
                                                 or not t == REQUIRED_THRESHOLD)]
        if bad:
            out.append(f"the effective threshold of {n} is {bad[0]!r}, not {REQUIRED_THRESHOLD}, in the {cond[1]} BN state (patience {cond[0]}, "
                       f"threshold {cond[2]}; {len(bad)} entries)")
    return out


def check_plan(plan: list, folds, seeds) -> dict:
    """(a) Il controllo chiesto dal coordinatore, prima di qualunque lista o frame del test, sul piano di valutazione: per
    ogni condizione (le due primarie per prime) le 12 combinazioni fold × seed per ciascuna rete (nessuna mancante,
    duplicata o in più) e, nelle condizioni a soglia fissa, la soglia che sarà applicata uguale a 0,5 per ciascuna rete.
    Il primo difetto ferma il comando (Failure: «primary analysis stopped before the test» per le due primarie).
    Restituisce il riepilogo dei controlli superati."""
    out = {}
    for cond in list(PRIMARY) + [c for c in D.CONDITIONS if c not in PRIMARY]:
        entries = [e for e in plan if (e["patience"], e["bn_state"], e["rule"]) == cond]
        probs = combination_problems(cond, [(e["network"], e["fold"], e["seed"]) for e in entries], folds, seeds)
        probs += threshold_problems(cond, [(e["network"], e["threshold"]) for e in entries])
        if probs:
            fail(f"{condition_label(cond)} stopped before the test: " + "; ".join(probs))
        out[cond] = {"combinations": len(entries) // len(AM.NETWORKS), "threshold": REQUIRED_THRESHOLD if cond[2] == "fixed_0.5" else "per model (dev EER)"}
    return out


def check_units(cond: tuple, units: list, folds, seeds) -> dict:
    """(b) Il controllo chiesto dal coordinatore subito prima del calcolo, sulle unità che saranno passate a D (gli stessi
    oggetti): unità del dataset NUAA; le 12 combinazioni fold × seed per ciascuna rete; nelle condizioni a soglia fissa la
    soglia effettiva di ciascuna rete, cioè il campo «threshold» che D.prepare_units usa per le decisioni, esattamente 0,5.
    Poi D.prepare_units sulle stesse unità deve dare esattamente le 12 unità. Failure («primary analysis stopped» per le
    due primarie), nessun ID nel messaggio."""
    combos, pairs = [], []
    for u in units:
        key = tuple(u.get("unit") or ())
        if len(key) != 3 or key[0] != DATASET:
            fail(f"{condition_label(cond)} stopped: a unit is not a ({DATASET}, fold, seed) unit")
        nets = u.get("networks") or {}
        for n in sorted(nets):
            combos.append((n, key[1], key[2]))
            pairs.append((n, (nets[n] or {}).get("threshold") if isinstance(nets[n], dict) else None))
    probs = combination_problems(cond, combos, folds, seeds) + threshold_problems(cond, pairs)
    if probs:
        fail(f"{condition_label(cond)} stopped: " + "; ".join(probs))
    prepared = D.prepare_units(units)
    if sorted(pu["unit"] for pu in prepared) != sorted((DATASET, f, s) for f in folds for s in seeds):
        fail(f"{condition_label(cond)} stopped: the prepared units are not the {len(folds) * len(seeds)} fold-seed units")
    return {"units": len(prepared), "combinations_per_network": len(combos) // len(AM.NETWORKS),
            "effective_threshold": {n: sorted({float(t) for m, t in pairs if m == n}) for n in AM.NETWORKS} if cond[2] == "fixed_0.5"
            else "per model (dev EER)"}


# ============================================================================ valutazione dei modelli (TensorFlow)

def forward_scores(model, X, batch: int, sg: float, what: str) -> dict:
    """Probabilità d'attacco e logit con nuaa_bn_recalibration.forward (il passaggio di BR.evaluate, senza le etichette:
    nessuna metrica calcolata), con il controllo del softmax dei logit della PR #27."""
    f = BR.forward(model, X, batch, scores=True)
    p = FR.probabilities(lambda _x: f["p"], X)
    z = np.asarray(f["z"], np.float64)
    if z.shape != (len(X), 2) or not np.all(np.isfinite(z)):
        fail(f"{what}: the logits have shape {z.shape} or non-finite values")
    gap = float(np.max(np.abs(SR.softmax(z) - p))) if len(X) else 0.0
    if gap > sg:
        fail(f"{what}: the softmax of the logits differs from the probabilities by {gap:.2e}")
    return {"p": p[:, 1].copy(), "z": z, "softmax_gap": gap}


def score_model(ctx, rec: dict, parts: dict) -> dict:
    """Un modello nei due stati della BN sui frame dati (parts: ruolo -> frame assemblati): checkpoint ricaricato con lo
    SHA-256 verificato subito prima (BR.load_model) e controllato (R.check_loaded); stato ordinario; copia in memoria
    (clone_model + set_weights, uguale sui byte) con le statistiche ristimate salvate dal run reinstallate (BT.install) e
    controllate (BR.tensor_problems: solo moving_mean e moving_variance diversi, uguali sui byte ai valori salvati); stato
    ristimato; originale invariato. Ogni difetto -> Failure."""
    tf = BR.tf_module()
    tf.keras.backend.clear_session()
    name, order = rec["key"], rec["order"]
    orig = BR.load_model(rec["weights"], rec["weights_sha256"])
    R.check_loaded(orig, ctx.cfg_pr, rec["network"])
    out = {s: {} for s in D.BN_STATES}
    for role, part in parts.items():
        out["ordinary"][role] = forward_scores(orig, part["X"], ctx.batch, ctx.sg, f"{name} {role} ordinary")
    snap = BR.snapshot_weights(orig)
    copy = tf.keras.models.clone_model(orig)
    copy.set_weights(orig.get_weights())
    if BR.snapshot_weights(copy) != snap:
        fail(f"{name}: the in-memory copy differs from the checkpoint before the statistics are installed")
    stats = BT.install(copy, rec["installed"])
    if probs := BR.tensor_problems(snap, BR.snapshot_weights(copy), BR.moving_keys(copy, order), stats):
        fail(f"{name}: re-estimated copy: " + "; ".join(probs))
    for role, part in parts.items():
        out["re-estimated"][role] = forward_scores(copy, part["X"], ctx.batch, ctx.sg, f"{name} {role} re-estimated")
    if BR.snapshot_weights(orig) != snap:
        fail(f"{name}: the checkpoint model changed while the copy was scored")
    del orig, copy
    gc.collect()
    return out


def reproduce_dev(ctx, recs: list) -> list:
    """Fase 2: ogni modello sul dev del suo fold nei due stati della BN, prima di qualunque lista o frame del test,
    confrontato con i punteggi salvati dal run (R.evaluator_agreement, tolleranze della configurazione della PR #27: scarto
    <= 1e-5, nessuna decisione discordante a 0,5 e alla soglia EER dello stato). Tutti i modelli sono valutati (il primo
    disaccordo non ferma il ciclo: la tabella documenta tutti). Restituisce le righe della riproduzione."""
    dev_parts, rows = {}, []
    for rec in recs:
        ctx.budget.check(f"dev reproduction {rec['key']}")
        if rec["fold"] not in dev_parts:
            dev_parts[rec["fold"]] = R.unit_frames(ctx.store, ctx.inp, "dev", rec["fold"])
        dev = dev_parts[rec["fold"]]
        if dev["keys"] != rec["dev"]["ordinary"]["ids"]:
            fail(f"{rec['key']}: the assembled dev frames are not those of the saved dev scores, in the same order")
        res = (SCORE_MODEL or score_model)(ctx, rec, {"dev": dev})
        for state in D.BN_STATES:
            got = res[state]["dev"]
            ag = R.evaluator_agreement(rec["dev"][state]["s"], got["p"], rec["thresholds"][(state, "dev_eer")], ctx.bn_cfg)
            rows.append({"run": rec["run"], "network": rec["network"], "fold": rec["fold"], "seed": rec["seed"], "bn_state": state,
                         "n_dev": len(got["p"]), "max_abs_diff": ag["max_abs_diff"], "discordant_05": ag["discordant_05"],
                         "discordant_eer": ag["discordant_eer"], "tolerance": ag["tolerance"], "discordant_tolerance": ag["discordant_tolerance"],
                         "softmax_gap": got["softmax_gap"], "pass": ag["pass"], "key": rec["key"]})
        print(f"   dev reproduced {rec['key']}: " + ", ".join(f"{r['bn_state']} {'yes' if r['pass'] else 'NO'}" for r in rows[-2:]), flush=True)
    return rows


# ============================================================================ test

def load_test_lists(ctx) -> dict:
    """Lo schema dei fold come lo carica nuaa_bn_test.load_inputs (FR.load_fold_manifests, SR.check_schema_config,
    SR.check_sources anche sulle righe del test, tutti i fold 1..k), solo dopo le riproduzioni del dev e la riverifica; per fold: numero dei frame del test
    uguale al piano; test disgiunto per chiave e per soggetto dal fitting del braccio B e dal dev del fold."""
    inp, pcfg = ctx.inp, ctx.pcfg
    FR.load_fold_manifests(SR.fr_view(pcfg), ctx.base, inp)
    if sorted(inp.folds) != list(range(1, pcfg["schema_expected"]["k"] + 1)):
        fail(f"the fold schema was not loaded for every fold 1..{pcfg['schema_expected']['k']} (as nuaa_bn_test.load_inputs requires)")
    SR.check_schema_config(pcfg, ctx.base, inp)
    SR.check_sources(pcfg, inp)
    out = {}
    for i, f in enumerate(ctx.folds):
        rows = inp.folds[f]["test"]
        if len(rows) != ctx.test_per_fold[i]:
            fail(f"fold {f}: {len(rows)} test frames, the pre-registration plans {ctx.test_per_fold[i]}")
        other = BR.role_rows(inp, "fitting", ARM, f) + BR.role_rows(inp, "dev", ARM, f)
        if {r["key"] for r in rows} & {r["key"] for r in other} or {r["subject"] for r in rows} & {r["subject"] for r in other}:
            fail(f"fold {f}: the test shares a frame or a subject with the fitting or the dev of the fold")
        out[f] = rows
    return out


def score_test(ctx, recs: list, lists: dict) -> dict:
    """Fase 3: per fold, i frame del test assemblati una volta (FR.assemble, ordine di chiave) e valutati dai modelli del
    fold nei due stati (score_model); solo probabilità e logit, in memoria. Restituisce per fold le righe e gli array."""
    meta = {}
    for f in ctx.folds:
        ctx.budget.check(f"test assembly fold {f}")
        test = FR.assemble(ctx.store, lists[f])
        if not test["keys"] or test["keys"] != sorted(r["key"] for r in lists[f]):
            fail(f"fold {f}: the assembled test frames are empty or not in key order")
        meta[f] = {"rows": test["rows"], "ids": list(test["keys"]), "y": test["y"], "subject": test["s"]}
        for rec in [r for r in recs if r["fold"] == f]:
            ctx.budget.check(f"test scoring {rec['key']}")
            res = (SCORE_MODEL or score_model)(ctx, rec, {"test": test})
            rec["test"] = {s: res[s]["test"] for s in D.BN_STATES}
            for s in D.BN_STATES:
                if len(rec["test"][s]["p"]) != len(test["keys"]):
                    fail(f"{rec['key']}: {len(rec['test'][s]['p'])} test scores for {len(test['keys'])} frames")
            print(f"   test scored {rec['key']} (two BN states)", flush=True)
        del test
        gc.collect()
    return meta


def build_conditions(plan: list, recs: list, meta: dict) -> dict:
    """Le unità delle 8 condizioni per D, dal piano già controllato (a): per ogni voce (modello × stato × regola)
    un'unità (NUAA, fold, seed) della sua condizione con, per la rete della voce, etichette, punteggi d'attacco dello
    stato, soggetti, l'ID di ogni frame (sample_id) e la soglia della voce (la soglia effettiva: D.prepare_units la usa
    per le decisioni). Una voce ripetuta resta visibile al controllo (b) come combinazione duplicata."""
    by_key = {r["key"]: r for r in recs}
    conds = {c: [] for c in D.CONDITIONS}
    index = {}
    for e in plan:
        r = by_key[e["key"]]
        m = meta[r["fold"]]
        cond = (e["patience"], e["bn_state"], e["rule"])
        net = {"y": m["y"], "s": r["test"][e["bn_state"]]["p"], "subject": m["subject"], "threshold": e["threshold"],
               D.FRAME_ID_FIELD: list(m["ids"])}
        u = index.get((cond, r["fold"], r["seed"]))
        if u is None or r["network"] in u["networks"]:
            u = {"unit": (DATASET, r["fold"], r["seed"]), "networks": {}}
            index[(cond, r["fold"], r["seed"])] = u
            conds.setdefault(cond, []).append(u)
        u["networks"][r["network"]] = net
    return conds


# ============================================================================ uscite

def exact(v):
    """Una frazione esatta come testo «p/q» (None -> NA)."""
    if v is None:
        return "NA"
    v = Fraction(v)
    return f"{v.numerator}/{v.denominator}"


def jsonable(v):
    """Valori per JSON: frazioni come {"exact": "p/q", "value": float}; tuple come liste; chiavi come testo; NumPy in Python."""
    if isinstance(v, Fraction):
        return {"exact": exact(v), "value": float(v)}
    if isinstance(v, dict):
        return {(k if isinstance(k, str) else str(list(k)) if isinstance(k, tuple) else str(k)): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return [jsonable(x) for x in v.tolist()]
    return v


def cell(v) -> str:
    if v is None:
        return "NA"
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, Fraction):
        return exact(v)
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    return str(v)


def public_condition(c: dict) -> dict:
    """La parte pubblicabile di compare_condition: nessun ID (righe per unità con numero dei frame e impronta)."""
    seeds = {s: {k: v[k] for k in ("mean_delta", "interval", "acer_counts", "sign", "sign_reason")} | {"n_subjects": v["bootstrap"]["n_subjects"]}
             for s, v in c["per_seed"].items()}
    return {**D.gap_summary(c), "per_seed": seeds, "per_seed_reading": c["per_seed_reading"], "rows": c["rows"]}


def loso_totals(x: dict) -> tuple:
    """Denominatori di un'esclusione sommati sulle unità: (attacchi, bona fide)."""
    return sum(d["n_attack"] for d in x["denominators"]), sum(d["n_bona_fide"] for d in x["denominators"])


def public_loso(lo: dict) -> dict:
    """La parte pubblica della sensibilità ai soggetti, nella forma pubblica di N1 (PR #10, solo aggregati): media di ΔACER
    con tutti i soggetti, minimo, massimo, esclusioni che cambiano il segno, NA, numero delle esclusioni e intervalli dei
    denominatori per classe (sommati sulle unità). Nessun valore per esclusione: quelli, con il soggetto, stanno nel
    riservato (loso_values.csv)."""
    tot = [loso_totals(x) for x in lo["values"]]
    return {k: lo[k] for k in ("full_mean_delta_acer", "full_sign", "min", "max", "n_sign_differs", "n_not_available", "n_excluded")} | \
        {"n_attack_after_exclusion": [min(a for a, _ in tot), max(a for a, _ in tot)] if tot else None,
         "n_bona_fide_after_exclusion": [min(b for _, b in tot), max(b for _, b in tot)] if tot else None}


def condition_rows(cond: tuple, c: dict) -> list:
    rows = []
    for scope, v in [("all seeds", c)] + [(f"seed {s}", c["per_seed"][s]) for s in sorted(c["per_seed"])]:
        iv, md, ac = v["interval"], v["mean_delta"], v["acer_counts"]
        rows.append({"patience": cond[0], "bn_state": cond[1], "threshold": cond[2], "label": "primary" if cond in PRIMARY else "secondary",
                     "scope": scope, "units": ac["units"], "n_subjects": v["bootstrap"]["n_subjects"], "mean_delta_acer": md["acer"],
                     "low_acer": iv["acer"]["low"], "high_acer": iv["acer"]["high"], "valid_resamples_acer": iv["acer"]["valid_resamples"],
                     "mean_delta_apcer": md["apcer"], "low_apcer": iv["apcer"]["low"], "high_apcer": iv["apcer"]["high"],
                     "mean_delta_bpcer": md["bpcer"], "low_bpcer": iv["bpcer"]["low"], "high_bpcer": iv["bpcer"]["high"],
                     "resnet18_lower": ac["resnet18_lower"], "livenessnet_lower": ac["livenessnet_lower"], "ties": ac["ties"],
                     "undefined": ac["undefined"], "sign": v["sign"], "sign_reason": v["sign_reason"]})
    return rows


def unit_rows(cond: tuple, c: dict, conds: dict) -> list:
    thr = {(tuple(u["unit"]), n): u["networks"][n]["threshold"] for u in conds[cond] for n in AM.NETWORKS}
    out = []
    for r in c["rows"]:
        v = {"patience": cond[0], "bn_state": cond[1], "threshold": cond[2], "fold": r["unit"][1], "seed": r["unit"][2], "n_frames": r["n_frames"],
             "frame_ids_sha256": r["frame_ids_sha256"], "delta_apcer": r["delta"]["apcer"], "delta_bpcer": r["delta"]["bpcer"],
             "delta_acer": r["delta"]["acer"], "delta_acer_value": None if r["delta"]["acer"] is None else float(r["delta"]["acer"]), "tie": r["tie"]}
        for n in AM.NETWORKS:
            p = n.lower().replace("-", "")
            v[f"{p}_threshold"] = float(thr[(tuple(r["unit"]), n)])
            for k in ("apcer", "bpcer", "acer"):
                v[f"{p}_{k}"] = r["metrics"][n][k]
        out.append(v)
    return out


def test_score_rows(rec: dict, meta: dict) -> list:
    """Le righe per frame del test di un modello nei due stati (riservate), con le decisioni della regola del progetto
    alle soglie del piano."""
    m, out = meta[rec["fold"]], []
    for s in D.BN_STATES:
        p, z = rec["test"][s]["p"], rec["test"][s]["z"]
        t05, teer = rec["thresholds"][(s, "fixed_0.5")], rec["thresholds"][(s, "dev_eer")]
        d05, deer = D.attack_decisions(p, t05), D.attack_decisions(p, teer)
        for i, r in enumerate(m["rows"]):
            out.append([rec["run"], rec["network"], str(rec["fold"]), str(rec["seed"]), str(rec["patience"]), s, "test", r["key"], r["subject"],
                        r["session"], r["label"], repr(float(p[i])), repr(float(z[i][0])), repr(float(z[i][1])), "1" if d05[i] else "0",
                        repr(float(teer)), "1" if deer[i] else "0", r["sha256"], rec["weights_sha256"]])
    return out


def fmt(v, nd: int = 4) -> str:
    return "NA" if v is None else f"{float(v):.{nd}f}"


def ci(iv: dict) -> str:
    return "no interval" if iv.get("low") is None else f"[{fmt(iv['low'])}, {fmt(iv['high'])}]"


def report_md(doc: dict) -> str:
    p = doc["primary_analysis"]
    go, gr, c = p["gap_ordinary"], p["gap_re_estimated"], p["contrast"]
    L = [f"# NUAA test: LivenessNet and ResNet-18 ({doc['run_id']})", ""]
    if not doc["official"]:
        L += ["**Not an official evaluation**: written outside results/archeval with --unofficial-out-root (a test of the command).", ""]
    L += [f"Status: **{doc['status']}**. Runs `{doc['runs']['nuaa_patience_4']['run_id']}` (patience 4) and "
          f"`{doc['runs']['nuaa_patience_12']['run_id']}` (patience 12); 48 models, {len(doc['models'])} evaluated; every model enters, "
          "one-class models included; nothing excluded after the test is seen.", "",
          "## Primary analysis (requested by the coordinator: patience 4, threshold 0.5)", "",
          "ACER, APCER and BPCER are proportions between 0 and 1 (not percentages). Gap = ACER(ResNet-18) - ACER(LivenessNet), mean over "
          "the 12 fold-seed units, with the subject-bootstrap interval (percentiles 2.5-97.5 of 1,000 resamples, seed 0, of the "
          f"{go['n_subjects']} test subjects pooled over the 4 folds; in parentheses the valid resamples, those in which the metric is "
          "defined in every unit); contrast = gap(re-estimated BN) - gap(ordinary BN).", "",
          "| | mean | subject interval (valid resamples) | units: ResNet-18 lower / LivenessNet lower / ties |", "|---|---|---|---|",
          f"| ACER gap, ordinary BN | {fmt(go['mean_delta']['acer'])} | {ci(go['interval']['acer'])} ({go['interval']['acer']['valid_resamples']}) | "
          f"{go['acer_counts']['resnet18_lower']} / {go['acer_counts']['livenessnet_lower']} / {go['acer_counts']['ties']} |",
          f"| ACER gap, re-estimated BN | {fmt(gr['mean_delta']['acer'])} | {ci(gr['interval']['acer'])} ({gr['interval']['acer']['valid_resamples']}) | "
          f"{gr['acer_counts']['resnet18_lower']} / {gr['acer_counts']['livenessnet_lower']} / {gr['acer_counts']['ties']} |",
          f"| contrast | {fmt(c['mean']['acer'])} | {ci(c['interval']['acer'])} ({c['interval']['acer']['valid_resamples']}) | "
          f"units whose gap changes sign: {c['units_changing_sign']} of {c['units']} |", "",
          f"Reading of the contrast (a choice of the author, pad_arch_definitions.contrast_reading): **{c['reading']}** ({c['reason']}).", "",
          f"Conclusion (written before the test; quoted from PROTOCOL.md §6, with the alternative resolved): \"{p['conclusion']}\"", "",
          "### Per seed (reported apart; not part of the reading)", "",
          f"Each seed has 4 units (one per fold) and the bootstrap of each seed uses the same {go['n_subjects']} test subjects.", "",
          "| seed | ACER gap, ordinary [interval] | ACER gap, re-estimated [interval] | contrast [interval] | contrast reading |", "|---|---|---|---|---|"]
    for s, v in p["per_seed"].items():
        L.append(f"| {s} | {fmt(v['gap_ordinary']['mean_delta']['acer'])} {ci(v['gap_ordinary']['interval']['acer'])} | "
                 f"{fmt(v['gap_re_estimated']['mean_delta']['acer'])} {ci(v['gap_re_estimated']['interval']['acer'])} | "
                 f"{fmt(v['contrast']['mean']['acer'])} {ci(v['contrast']['interval']['acer'])} | {v['contrast']['reading']} |")
    ch = doc["checks_before_primary_analysis"]
    L += ["", "Checks requested by the coordinator, passed before the primary analysis: (a) on the evaluation plan, before any test list "
          "or frame was used: the 12 fold-seed combinations for each network in both BN states and the threshold to be applied 0.5 for "
          "both networks; (b) on the units passed to pad_arch_definitions.primary_analysis: "
          + "; ".join(f"{k}: {v['units']} units, effective threshold {v['effective_threshold']}" for k, v in ch["units"].items()
                      if k in [str(list(c)) for c in PRIMARY])
          + f". The same completeness check passed on the other {len(ch['units']) - len(PRIMARY)} conditions (choice of the author).", "",
          "## Primary analysis: APCER and BPCER", "", "| | APCER gap [interval] | BPCER gap [interval] |", "|---|---|---|"]
    for label, g in (("ordinary BN", go), ("re-estimated BN", gr)):
        L.append(f"| {label} | {fmt(g['mean_delta']['apcer'])} {ci(g['interval']['apcer'])} | {fmt(g['mean_delta']['bpcer'])} {ci(g['interval']['bpcer'])} |")
    L.append(f"| contrast | {fmt(c['mean']['apcer'])} {ci(c['interval']['apcer'])} | {fmt(c['mean']['bpcer'])} {ci(c['interval']['bpcer'])} |")
    L += ["", "## The eight conditions (secondary except the two primary ones; descriptive)", "",
          "| patience | BN | threshold | mean ACER gap [interval] | ResNet-18 lower / LivenessNet lower / ties | sign | per-seed reading |",
          "|---|---|---|---|---|---|---|"]
    for x in doc["conditions"]:
        k = x["condition"]
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {fmt(x['mean_delta']['acer'])} {ci(x['interval']['acer'])} | {x['acer_counts']['resnet18_lower']} / "
                 f"{x['acer_counts']['livenessnet_lower']} / {x['acer_counts']['ties']} | {x['sign']} | {x['per_seed_reading']['verdict']} |")
    L += ["", "## Between conditions (one factor changed; contrast = gap(first) - gap(second))", "",
          "| first | second | signs | sign change | point-estimate inversion | ACER contrast [interval] | units changing sign |", "|---|---|---|---|---|---|---|"]
    for x in doc["comparisons"]:
        L.append(f"| {x['first']} | {x['second']} | {x['sign_a']} / {x['sign_b']} | {'yes' if x['sign_change'] else 'no'} | "
                 f"{'yes' if x['point_estimate_inversion'] else 'no'} | {fmt(x['contrast_mean']['acer'])} {ci(x['contrast_interval']['acer'])} | "
                 f"{x['units_changing_sign']} of {x['units']} |")
    L += ["", "## Subject sensitivity (mean ACER gap excluding one test subject at a time)", "",
          "Aggregates only, in the public form of N1 (PR #10); the value of each exclusion is in the restricted outputs.", "",
          "| condition | with all subjects | minimum | maximum | exclusions changing the sign | NA | n attack after exclusion | "
          "n bona fide after exclusion |", "|---|---|---|---|---|---|---|---|"]
    rng = lambda v: "NA" if not v else f"{v[0]}-{v[1]}"  # noqa: E731
    for x in doc["subject_sensitivity"]:
        L.append(f"| {x['condition']} | {fmt(x['full_mean_delta_acer'])} | {fmt(x['min'])} | {fmt(x['max'])} | {x['n_sign_differs']} of "
                 f"{x['n_excluded']} | {x['n_not_available']} | {rng(x['n_attack_after_exclusion'])} | {rng(x['n_bona_fide_after_exclusion'])} |")
    L += ["", "## One-class runs at threshold 0.5 on the dev (definition (a))", "", "| network | patience | BN state | count |", "|---|---|---|---|"]
    for o in doc["one_class"]:
        L.append(f"| {o['network']} | {o['patience']} | {o['bn_state']} | {o['text']} (all bona fide {o['all_bona_fide']}, all attack "
                 f"{o['all_attack']}, not evaluable {o['not_evaluable']}) |")
    dg = doc["degenerate_apart"]
    L += ["", "## The earlier \"degenerate\" count, apart (PROTOCOL.md §6)", "", dg["definition"], "", "| network | patience | degenerate |", "|---|---|---|"]
    L += [f"| {x['network']} | {x['patience']} | {x['degenerate']} of {x['trainings']} |" for x in dg["counts"]]
    rp = doc["dev_reproduction"]
    L += ["", f"Dev reproduced before the test: {rp['passed']} of {rp['checked']} model-state pairs within {rp['tolerance']} with no "
          f"discordant decision (largest difference {fmt(rp['largest_difference'], 10)}); per model: reproduction.csv.", "",
          f"Test frames per fold: {', '.join(str(x) for x in doc['test_frames_per_fold'])}; public outputs hold only counts and the SHA-256 "
          "of the sorted frame identifiers.", "",
          f"Started {doc['started']}, finished {doc['finished']}; elapsed time {doc['wall_seconds']:.0f} s of a cap of {doc['cap_hours']:.2f} "
          f"{doc['cap_unit']}; process CPU time (all threads) {doc['process_cpu_seconds']:.0f} s.", "",
          PAC.threads_text(doc["threads"]), "", f"What was read of the test: {doc['test_read']}.", "", f"Command: `{doc['command']}`", ""]
    return "\n".join(L)


# ============================================================================ cartella, tentativi, comando

def previous_attempts(out_root: Path) -> list:
    """Le valutazioni nella radice d'uscita (qualunque data e commit): run id, numero del tentativo e stato (la regola di
    pad_arch_run.previous_attempts con il run id ARCHEVAL)."""
    out = []
    if not out_root.is_dir():
        return out
    for e in sorted(os.listdir(out_root)):
        if not LOOKS_LIKE_EVAL.match(e):
            continue
        m = EVAL_ID_RE.match(e)
        if not m:
            fail(f"output root: the entry {e!r} looks like an evaluation run id but does not have the expected form "
                 "(YYYYMMDD-ARCHEVAL-<commit>, suffix -r<k> with k >= 2): rename or move it")
        try:
            st = json.loads((out_root / e / STATUS).read_text(encoding="utf-8")).get("status", "unreadable")
        except (OSError, ValueError):
            st = "unreadable"
        out.append({"run_id": e, "attempt": int(m.group(1)) if m.group(1) else 1, "status": st})
    return out


def attempt_rule(out_root: Path, attempt: int) -> dict:
    prev = previous_attempts(out_root)
    need = max((p["attempt"] for p in prev), default=0) + 1
    if attempt != need:
        fail(f"--attempt {attempt} refused: {len(prev)} earlier attempts of this evaluation in the output root "
             f"({', '.join(p['run_id'] + ' ' + p['status'] for p in prev) or 'none'}); this attempt must be --attempt {need}")
    return {"attempt": attempt, "previous": prev}


def run_id_for(date: str, state: dict, attempt: int) -> str:
    if not re.fullmatch(r"\d{8}", date or ""):
        fail("--date must be YYYYMMDD")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        fail("--attempt must be a positive integer")
    rid = f"{date}-ARCHEVAL-{state['commit']}" + ("" if attempt == 1 else f"-r{attempt}")
    if not EVAL_ID_RE.match(rid):
        fail(f"run id {rid} malformed")
    return rid


def command_text(a) -> str:
    """Il comando registrato dai valori interpretati (come pad_arch_run.command_text): tutti gli argomenti, percorsi dentro
    il repository relativi alla sua radice, fuori dal repository con il solo ultimo componente."""
    parts = [a.cmd, "--config", R.path_text(R.folder_arg(a.config))]
    if a.input_root is not None:
        parts += ["--input-root", R.path_text(Path(a.input_root).resolve())]
    for opt in ("run_p4", "run_p12"):
        if getattr(a, opt) is not None:
            parts += [f"--{opt.replace('_', '-')}", R.path_text(R.folder_arg(getattr(a, opt)))]
    if a.cmd == "evaluate":
        parts += ["--out-root", R.path_text(R.folder_arg(a.out_root))]
        if a.unofficial_out_root:
            parts.append("--unofficial-out-root")
        if a.date is not None:
            parts += ["--date", a.date]
        parts += ["--attempt", str(a.attempt)]
        if a.i_am_authorized:
            parts.append("--i-am-authorized")
    return "python -I -B scripts/pad_arch_eval.py " + " ".join(shlex.quote(x) for x in parts)


def run_dir_problem(run: Path, out_root: Path, inputs: dict) -> str:
    if os.path.lexists(run):
        return f"the evaluation folder {run.name} already exists: every attempt keeps its outcome"
    if out_root.is_symlink():
        return "the output root is a symbolic link"
    for label, p in inputs.items():
        if pg.within(p, run) or (Path(p).is_dir() and pg.within(run, p)):
            return f"{label} {Path(p).name} is inside the evaluation folder, or contains it: inputs and outputs must be separate"
    return mw.output_problem(run, ROOT, pg.BASE_PROTECTED, [ROOT / DEFAULT_OUT_ROOT], [ROOT / x for x in EXTRA_PROTECTED], inputs,
                             [STATUS, LOG], [run / PUBLIC, run / RESTRICTED])


def load_runs(a, pins: Pins, mode: str) -> dict:
    out = {}
    for run, opt in RUN_OPTION.items():
        v = getattr(a, opt)
        if v is None:
            fail(f"--{opt.replace('_', '-')} is required: the folder of the PASS of run {run}")
        out[run] = load_run(R.folder_arg(v), run, pins, mode)
    if out["nuaa_patience_4"]["folder"].resolve() == out["nuaa_patience_12"]["folder"].resolve():
        fail("--run-p4 and --run-p12 name the same folder")
    return out


class Ctx:
    """Lo stato di una valutazione in corso."""


def unofficial_inputs_rule(base: Path, official: bool) -> None:
    """Una valutazione non ufficiale (--unofficial-out-root, i test del comando) non usa gli ingressi del repository: una
    base degli ingressi che è, per identità dei file (path_guard.within nelle due direzioni, come root_kind), la radice del
    repository è rifiutata prima di leggere qualunque file; con il controllo dei run, una valutazione non ufficiale non
    legge un run ufficiale né una cartella della radice ufficiale results/archrun. Limiti dichiarati, non riconosciuti:
    chi copia fuori dal repository una cartella di run ufficiale e i dati riservati, e cambia «official» nello status.json
    della copia (nessun SHA256SUMS copre status.json); una base degli ingressi fuori dal repository che contiene gli
    ingressi veri (copiati, o raggiunti con collegamenti simbolici: read_regular controlla con lstat solo l'ultimo
    componente), con run non ufficiali fatti su di essi."""
    if not official and pg.within(base, ROOT) and pg.within(ROOT, base):
        fail("an unofficial evaluation (--unofficial-out-root) does not use the inputs of the repository: pass --input-root with inputs "
             "outside it")


def prepare(a) -> tuple:
    base, cfg_p = R.resolve(a)
    official = R.root_kind(R.folder_arg(a.out_root), ROOT, ROOT / DEFAULT_OUT_ROOT, a.unofficial_out_root)
    unofficial_inputs_rule(base, official)
    cfg, cfg_sha = load_config(cfg_p)
    command = command_text(a)
    if probs := PAC.output_problems(command, FR.Inputs()):
        fail("the recorded command holds text that a publishable output must not hold: " + "; ".join(probs))
    pins = load_pins(cfg, base)
    env = R.thread_environment_rule(pins.cfg_pr, pins.trial)     # variabili dei thread diverse dalla prova: rifiuto
    mrule = R.machine_rule(pins.trial)
    runs = load_runs(a, pins, "official" if official else "unofficial")
    state = R.repo_state(ROOT)
    if state["dirty"]:
        fail("the git work tree is not clean: the evaluation needs a committed state (commit the changes first; results/ is versioned)")
    out_root = R.folder_arg(a.out_root)
    attempt = attempt_rule(out_root, a.attempt)
    rid = run_id_for(a.date or time.strftime("%Y%m%d"), state, a.attempt)
    run = out_root / rid
    inputs = R.input_paths(pins.rcfg, pins.rpins, base, pins.rcfg_p)
    inputs["the configuration of this command"] = cfg_p
    for k in PACKAGE_ENTRIES:
        inputs[f"the pre-registration file {k}"] = rel_path(base, cfg["preregistration"][k]["path"])
    for r in runs.values():
        inputs[f"the run folder {r['run']}"] = r["folder"]
    if msg := run_dir_problem(run, out_root, inputs):
        fail(f"refusing the evaluation folder: {msg}")
    return base, cfg_p, cfg, cfg_sha, pins, env, mrule, official, runs, state, rid, run, attempt, command


def before_test(ctx, runs: dict) -> tuple:
    """Fase 1 (nessuna lista né frame del test): file del dev, soglie, piano, controllo (a), conteggi one-class."""
    recs = [m for r in R.RUNS for m in runs[r]["models"]]
    for rec in recs:
        load_dev(rec, ctx.inp)
    plan = evaluation_plan(recs)
    if PLAN_HOOK is not None:
        plan = PLAN_HOOK(plan)
    checked = check_plan(plan, ctx.folds, ctx.seeds)
    oc = D.one_class_counts([{"network": r["network"], "patience": r["patience"], "bn_state": s,
                              "result": D.one_class_run(r["dev"][s]["y"], r["dev"][s]["s"])} for r in recs for s in D.BN_STATES])
    check_one_class(oc, runs)
    return recs, plan, checked, oc


def check_one_class(oc: list, runs: dict) -> None:
    """I conteggi one-class ricalcolati qui dai file del dev uguali, cella per cella, a public/one_class.csv di ciascun run
    (stesse funzioni, stessi punteggi: una differenza vuol dire ingressi diversi)."""
    for r in runs.values():
        rows = parse_csv(r["one_class_csv"], R.ONE_CLASS_COLUMNS, f"{r['run_id']} one_class.csv")
        theirs = sorted(tuple(x[k] for k in R.ONE_CLASS_COLUMNS) for x in rows)
        mine = sorted(tuple(R.cell(o[k]) for k in R.ONE_CLASS_COLUMNS) for o in oc if o["patience"] == R.RUNS[r["run"]])
        if theirs != mine:
            fail(f"one-class counts of {r['run']} differ from public/one_class.csv of run {r['run_id']}")


def degenerate_counts(recs: list, bn_cfg: dict) -> list:
    """Il conteggio «degenere» di prima, a parte (PROTOCOL.md §6), sulle schede dei 48 modelli nuovi (solo fitting e dev):
    la regola accettata del gruppo «ten» della PR #27 (nuaa_bn_recalibration.group_of con la sezione groups della sua
    configurazione fissata: epoca ripristinata fra restored_epochs e accuratezza del dev al checkpoint uguale a
    dev_accuracy entro within), per rete × patience."""
    out = []
    for n in AM.NETWORKS:
        for p in D.PATIENCES:
            rs = [r for r in recs if (r["network"], r["patience"]) == (n, p)]
            out.append({"network": n, "patience": p, "degenerate": sum(1 for r in rs if BR.group_of(r["card"], bn_cfg) == "ten"), "trainings": len(rs)})
    return out


def scan_log(run: Path, log, inp) -> None:
    """Il run.log scritto fin qui (svuotato su disco) passato dalla scansione delle uscite pubblicabili, prima di calcolare i
    numeri dell'analisi: un difetto che non dipende dai numeri ferma il comando prima che esistano."""
    log[0].flush()
    if probs := PAC.output_problems((run / LOG).read_text(encoding="utf-8", errors="replace"), inp):
        fail(f"{LOG} holds text that a publishable output must not hold: " + "; ".join(probs))


def final_checks(ctx, cfg_p: Path, cfg_sha: str, runs: dict) -> dict:
    """Subito prima delle liste del test e di nuovo prima dei risultati: questa configurazione, la configurazione dei run e i suoi file fissati (R.final_checks), il
    codice, PROTOCOL.md e run_plan.csv, le due cartelle dei run (recheck_run) con gli stessi SHA-256 dell'inizio."""
    if sha256_bytes(read_regular(cfg_p, "configuration")) != cfg_sha:
        fail("the configuration changed during the evaluation")
    if check_code(ctx.cfg) != ctx.pins.code:
        fail("the code changed during the evaluation")
    R.final_checks(ctx.pins.rcfg, ctx.pins.rcfg_p, ctx.pins.rcfg_sha, ctx.base, ctx.pins.rpins)
    for k in PACKAGE_ENTRIES:
        R.read_pinned(ctx.base, ctx.cfg["preregistration"][k], "pre-registration file")
    for r in runs.values():
        recheck_run(r)
    return {"configuration_unchanged": True, "code_unchanged": len(ctx.pins.code), "run_configuration_and_pinned_files_unchanged": True,
            "run_folders_unchanged": {r["run_id"]: len(r["sums"][PUBLIC]) + len(r["sums"][RESTRICTED]) for r in runs.values()}}


# ============================================================================ analisi

def analyse(ctx, plan: list, recs: list, meta: dict) -> dict:
    """Fase 4, dopo la riverifica: unità delle 8 condizioni; controllo (b) e analisi primaria; le altre condizioni con lo
    stesso controllo; confronti fra condizioni; sensibilità ai soggetti. Tutto con le funzioni di D."""
    conds = build_conditions(plan, recs, meta)
    if UNITS_HOOK is not None:
        conds = UNITS_HOOK(conds)
    checked = {}
    for cond in PRIMARY:
        checked[str(list(cond))] = check_units(cond, conds[cond], ctx.folds, ctx.seeds)
    primary = D.primary_analysis({cond: conds[cond] for cond in PRIMARY})
    for cond in D.CONDITIONS:
        if cond not in PRIMARY:
            checked[str(list(cond))] = check_units(cond, conds[cond], ctx.folds, ctx.seeds)
    results = {cond: D.compare_condition(conds[cond]) for cond in D.CONDITIONS}
    comparisons = [{"first": list(x), "second": list(y), **D.compare_conditions(results[x], results[y])} for x, y in condition_pairs()]
    loso = {cond: D.leave_one_subject_out(conds[cond]) for cond in D.CONDITIONS}
    return {"conds": conds, "checked": checked, "primary": primary, "results": results, "comparisons": comparisons, "loso": loso}


# ============================================================================ comandi

def cmd_check_inputs(a) -> int:
    base, cfg_p = R.resolve(a)
    cfg, cfg_sha = load_config(cfg_p)
    pins = load_pins(cfg, base)
    env = R.thread_environment_rule(pins.cfg_pr, pins.trial)
    R.machine_rule(pins.trial)
    given = [getattr(a, o) is not None for o in RUN_OPTION.values()]
    if any(given) and not all(given):
        fail("check-inputs needs both --run-p4 and --run-p12, or neither")
    runs = load_runs(a, pins, "inputs") if all(given) else None
    inp = PAC.load_inputs(pins.cfg_pr, pins.pcfg, base)
    text = "run folders not given"
    if runs:
        ctx = Ctx()
        ctx.inp, ctx.folds, ctx.seeds = inp, pins.folds, pins.seeds
        recs, plan, checked, _ = before_test(ctx, runs)
        text = (f"run folders {', '.join(r['run_id'] + (' official' if r['official'] else ' not official') for r in runs.values())} PASS, every "
                f"file equal to its SHA256SUMS; {len(recs)} models with both dev files equal to the manifests; check (a) passed on "
                f"{len(plan)} plan entries ({len(checked)} conditions)")
    PAC.assert_no_test(inp)
    tfs = FR.tf_state()
    print(f"CHECK-INPUTS: configuration {cfg_sha[:12]}; run configuration {pins.rcfg_sha[:12]}; PROTOCOL.md and run_plan.csv equal to the "
          f"package; code pinned ({len(CODE_FILES)} files); cap {pins.cap['cap_hours']:.2f} h from the plan "
          f"({pins.cap['evaluation_hours_estimate']:.4f} h estimated); machine model and versions equal to the trial; environment variables "
          + ", ".join(f"{k} {'absent' if v is None else repr(v)}" for k, v in env.items())
          + f"; inputs of the pre-registration PASS; {text}; no fold schema, no test frame; TensorFlow {tfs}; nothing written")
    print("RESULT: PASS (check-inputs)")
    return 0


def cmd_evaluate(a) -> int:
    if not a.i_am_authorized:
        fail("evaluate reads the NUAA test with the 48 checkpoints: pass --i-am-authorized (only after this code is published)")
    base, cfg_p, cfg, cfg_sha, pins, env, mrule, official, runs, state, rid, run, attempt, command = prepare(a)
    run.mkdir(parents=True)
    scan = FR.Inputs()
    head = {"run_id": rid, "commit": state["commit"], "dirty": False, "config_sha256": cfg_sha, "command": "evaluate", "command_line": command,
            "official": official, "attempt": attempt,
            "runs": {k: {"run_id": r["run_id"], "official": r["official"], "status_sha256": r["status_sha256"],
                         "public_sha256sums_sha256": r["public_sums_sha256"], "restricted_sha256sums_sha256": r["restricted_sums_sha256"]}
                     for k, r in runs.items()}}
    cap_h = cfg["cap"]["hours"]
    budget = SR.Budget(cap_h * 60)
    SR.signal_handlers(budget)
    started, c0 = now(), PAC.cpu_seconds()
    log, saved_warning = None, warnings.showwarning
    threads = {"requested": dict(pins.trial["threads"]["requested"]), "effective": None, "set_by_this_runner": False, "environment": env,
               "logical_cpus": os.cpu_count()}
    extra = {"threads": threads, "machine_rule": mrule, "cap_hours": cap_h, "cap_unit": CAP_UNIT, "test_read_started": False}
    rdir = run / RESTRICTED
    written, pub_written = [], False
    try:
        R.write_status(run, {**head, "status": "RUNNING", "passed": False, "started": started, "pid": os.getpid(), **extra,
                             "note": "RUNNING without a live process means a failed evaluation; SIGUSR1 stops it as INCOMPLETE"}, scan)
        log = FR.open_log(run)
        warnings.showwarning = PAC.show_warning
        print(f"== evaluation {rid} (attempt {attempt['attempt']}{'' if official else ', NOT OFFICIAL'}): configuration {cfg_sha[:12]}, "
              f"cap {cap_h:.2f} {CAP_UNIT}, pid {os.getpid()}", flush=True)
        ctx = Ctx()
        ctx.cfg, ctx.pins, ctx.base, ctx.cfg_pr, ctx.pcfg, ctx.bn_cfg, ctx.budget = cfg, pins, base, pins.cfg_pr, pins.pcfg, pins.bn_cfg, budget
        ctx.folds, ctx.seeds, ctx.test_per_fold = pins.folds, pins.seeds, pins.test_per_fold
        ctx.batch, ctx.sg = pins.rcfg["estimator"]["batch"], BR.tol(pins.bn_cfg, "softmax_of_logits_max_abs_diff")
        inp = PAC.load_inputs(pins.cfg_pr, pins.pcfg, base)
        scan = ctx.inp = inp
        recs, plan, checked_plan, one_class = before_test(ctx, runs)
        degenerate = degenerate_counts(recs, pins.bn_cfg)
        print(f"   check (a) passed before the test: {len(plan)} plan entries, 12 fold-seed combinations per network in each BN state, "
              "threshold 0.5 in the fixed-threshold conditions", flush=True)
        ctx.store = FR.FrameStore(inp, inp.X_extracted, inp.extracted_keys)
        tf = R.tf_module()
        threads = PAC.set_threads(tf, pins.cfg_pr)            # subito dopo l'import, prima di qualunque operazione
        extra["threads"] = threads
        extra["threads_rule"] = R.threads_rule(pins.trial, threads)
        R.require_threads(extra["threads_rule"])
        print(f"   {PAC.threads_text(threads)}", flush=True)
        FR.hide_gpu(tf)
        ctx.tf = tf
        status, stopped, analysis, meta, repro = "PASS", None, None, None, []
        try:
            repro = reproduce_dev(ctx, recs)
            if any(not r["pass"] for r in repro):
                status = "MISMATCH"
            else:
                PAC.check_threads(tf, pins.cfg_pr, "after the dev reproduction")
                budget.check("re-verification before the test")
                extra["checks_before_test"] = final_checks(ctx, cfg_p, cfg_sha, runs)     # un FAIL che non ha bisogno del test viene prima
                extra["test_read_started"] = True
                lists = load_test_lists(ctx)
                meta = score_test(ctx, recs, lists)
                PAC.check_threads(tf, pins.cfg_pr, "after the test scoring")
                budget.check("analysis")
        except SR.CapExceeded as e:
            status = "INCOMPLETE"
            stopped = {"step": e.step, "cause": e.cause, "elapsed_minutes": round(e.minutes, 1)}
            print(f"== {'CAP' if e.cause == 'cap' else 'STOP'}: {e}; no result is computed", flush=True)
        extra["dev_reproduction"] = [{k: v for k, v in r.items() if k != "key"} for r in repro]
        pub_sha = res_sha = None
        checks = None
        if status == "PASS":
            # tutto ciò che può fallire senza i numeri, prima dei numeri: riverifica degli ingressi e il log fino a qui
            checks = final_checks(ctx, cfg_p, cfg_sha, runs)
            scan_log(run, log, inp)
            analysis = analyse(ctx, plan, recs, meta)
        wall, cpu_total, finished = budget.elapsed(), round(PAC.cpu_seconds() - c0, 1), now()
        if status == "PASS":
            pa = analysis["primary"]
            pj = jsonable(pa)
            primary = {"requested": cfg["primary_check"]["requested"], **{k: pj[k] for k in ("conditions", "definition", "gap_ordinary", "gap_re_estimated",
                                                                                          "contrast")},
                       "conclusion": conclusion_for(pins.conclusions, pa["contrast"]["reading"]), "per_seed": pj["per_seed"], "rows": pj["rows"]}
            rp = {"checked": len(repro), "passed": sum(r["pass"] for r in repro), "tolerance": repro[0]["tolerance"] if repro else None,
                  "largest_difference": max((r["max_abs_diff"] for r in repro), default=None)}
            doc = {"schema": EVAL_SCHEMA, "run_id": rid, "status": status, "official": official, "primary_analysis": primary,
                   "checks_before_primary_analysis": {"plan": jsonable({str(list(k)): v for k, v in checked_plan.items()}),
                                                      "units": analysis["checked"]},
                   **head, "started": started, "finished": finished, "wall_seconds": round(wall, 1),
                   "process_cpu_seconds": cpu_total, "cap_hours": cap_h, "cap_unit": CAP_UNIT, "cap_derivation": pins.cap,
                   "conditions": [{"condition": list(c), "label": "primary" if c in PRIMARY else "secondary",
                                   **jsonable(public_condition(analysis["results"][c]))} for c in D.CONDITIONS],
                   "comparisons": jsonable(analysis["comparisons"]),
                   "subject_sensitivity": [{"condition": list(c), **jsonable(public_loso(analysis["loso"][c]))} for c in D.CONDITIONS],
                   "one_class": one_class, "degenerate_apart": {"definition": cfg["degenerate_apart"], "rule": pins.bn_cfg["groups"]["degenerate"],
                                                                "counts": degenerate},
                   "dev_reproduction": rp,
                   "models": [{"run": r["run"], "position": r["position"], "network": r["network"], "fold": r["fold"], "seed": r["seed"],
                               "weights_sha256": r["weights_sha256"], "dev_eer_threshold": {s: r["thresholds"][(s, "dev_eer")] for s in D.BN_STATES}}
                              for r in recs],
                   "test_frames_per_fold": [len(meta[f]["ids"]) for f in ctx.folds],
                   "test_frame_ids_sha256": {str(f): D.ids_sha256(meta[f]["ids"]) for f in ctx.folds},
                   "machine": (R.MACHINE or PAC.machine)(), "environment": (R.ENVIRONMENT or FR.environment)(), "machine_rule": mrule,
                   "threads": threads, "threads_rule": extra["threads_rule"], "checks_before_test": extra["checks_before_test"], "checks": checks,
                   "code_sha256": pins.code, "inputs_sha256": {**pins.files, "configuration": cfg_sha}, "peak_rss_mb": round(SR.peak_rss_mb(), 1),
                   "test_read": TEST_READ, "command": command}
            conds = analysis["conds"]
            texts = {"results.json": PAC.json_text(doc, "results.json"), "report.md": report_md(doc),
                     "conditions.csv": csv_text(CONDITION_COLUMNS, [[cell(x[k]) for k in CONDITION_COLUMNS]
                                                                    for c in D.CONDITIONS for x in condition_rows(c, analysis["results"][c])]),
                     "units.csv": csv_text(UNIT_COLUMNS, [[cell(x[k]) for k in UNIT_COLUMNS] for c in D.CONDITIONS
                                                          for x in unit_rows(c, analysis["results"][c], conds)]),
                     "reproduction.csv": csv_text(REPRODUCTION_COLUMNS, [[cell(r[k]) for k in REPRODUCTION_COLUMNS] for r in repro])}
            for n, t in texts.items():
                if probs := PAC.output_problems(t, inp):
                    fail(f"{n} refused: " + "; ".join(probs))
            rtexts = {f"{r['key']}{TEST_SUFFIX}": csv_text(TEST_SCORE_COLUMNS, test_score_rows(r, meta)) for r in recs}
            rtexts["loso_values.csv"] = csv_text(LOSO_COLUMNS, [[c[0], c[1], c[2], x["excluded_subject"], cell(x["mean_delta_acer"]), cell(x["sign"]),
                                                                 cell(x["sign_differs"]), *loso_totals(x)]
                                                                for c in D.CONDITIONS for x in analysis["loso"][c]["values"]])
            rfiles = {**rtexts, "SHA256SUMS": FR.sums_text(rtexts)}
            mw.write_all(rdir, rfiles, newline="")
            written = list(rfiles)
            res_sha = sha256_bytes(rfiles["SHA256SUMS"].encode("utf-8"))
            pfiles = {**texts, "SHA256SUMS": FR.sums_text(texts)}
            mw.write_all(run / PUBLIC, pfiles, newline="")
            pub_written = True
            pub_sha = sha256_bytes(pfiles["SHA256SUMS"].encode("utf-8"))
            print("   results written to public/ and restricted/ (no number of the analysis in this log); the final scan and status follow", flush=True)
        elif status == "MISMATCH":
            print(f"== MISMATCH: the dev scores of {sum(not r['pass'] for r in repro)} model-state pairs are not reproduced; the test is not read",
                  flush=True)
        print(f"evaluation folder: {run.name}, elapsed {wall:.0f} s ({wall / 3600:.2f} h of the {cap_h:.2f}-hour cap), process CPU {cpu_total:.0f} s",
              flush=True)
        warnings.showwarning = saved_warning
        FR.close_log(log)
        log = None
        if probs := PAC.output_problems((run / LOG).read_text(encoding="utf-8", errors="replace"), inp):
            fail(f"{LOG} holds text that a publishable output must not hold: " + "; ".join(probs))
        R.write_status(run, {**head, "status": status, "passed": status == "PASS", "started": started, "finished": finished,
                             "wall_seconds": round(wall, 1), "process_cpu_seconds": cpu_total, **extra, "stopped": stopped, "checks": checks,
                             "machine": (R.MACHINE or PAC.machine)(), "code_sha256": pins.code, "inputs_sha256": {**pins.files, "configuration": cfg_sha},
                             "public_sha256sums_sha256": pub_sha, "restricted_sha256sums_sha256": res_sha,
                             "run_log_sha256": sha256_file(run / LOG)}, scan)
    except BaseException as e:
        warnings.showwarning = saved_warning
        left = mw.invalidate_each(rdir, written, None, FR.FAIL_DOC) if rdir.is_dir() and written else []
        extra["restricted"] = ("no restricted result file was written" if not written else
                               f"{len(written)} restricted result files were written and then removed after the failure" if not left
                               else f"restricted files left behind: {', '.join(left)}")
        extra["public"] = ("the public result files were written and then removed after the failure" if pub_written
                           else "no public result file was written")
        R.fail_cleanup(run, list(PUBLIC_OUTPUTS), log, head, started, budget, c0, extra, e, scan)
        raise
    if status == "PASS":
        print(f"analysis computed; report: {run.name}/public/report.md")
    print(f"RESULT: {status} (evaluate)")
    return STATUS_CODE[status]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="The evaluation on the NUAA test of the 48 models of the LivenessNet / ResNet-18 comparison "
                                             "(two BN states, eight conditions, primary analysis) and the check of its inputs")
    ap.add_argument("cmd", choices=["check-inputs", "evaluate"])
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="configuration JSON (relative to the repository root)")
    ap.add_argument("--input-root", default=None, help="base folder of the input paths of the configuration (default: repository root)")
    ap.add_argument("--run-p4", default=None, help="the folder of the PASS of run nuaa_patience_4 (relative to the repository root)")
    ap.add_argument("--run-p12", default=None, help="the folder of the PASS of run nuaa_patience_12 (relative to the repository root)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="folder that receives <run-id>/ (official: results/archeval; another folder "
                                                                 "outside the repository only with --unofficial-out-root)")
    ap.add_argument("--unofficial-out-root", action="store_true", help="accept an --out-root other than results/archeval; the evaluation is "
                                                                      "recorded as not official (tests of the command)")
    ap.add_argument("--date", default=None, help="YYYYMMDD of the run id (default: today)")
    ap.add_argument("--attempt", type=int, default=1, help="attempt number: the last attempt in the output root + 1 (1 if none)")
    ap.add_argument("--i-am-authorized", action="store_true", help="required by evaluate")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return {"check-inputs": cmd_check_inputs, "evaluate": cmd_evaluate}[a.cmd](a)
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
