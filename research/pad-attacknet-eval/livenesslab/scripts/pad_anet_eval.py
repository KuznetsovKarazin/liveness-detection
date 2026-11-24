"""
Valutazione sul test NUAA del confronto C5 (scheda C5 del coordinatore, 21/11/2026; PROTOCOL.md della PR A, PR #34, con
la copia emendata della PR B, PR #37): i modelli nuovi delle tre AttackNet dei sei run di scripts/pad_anet_run.py e i 48
riferimenti di C2 (LivenessNet e ResNet-18, i punteggi di test per frame già calcolati dalla valutazione C2
20261121-ARCHEVAL-332fbdf), con le definizioni congelate della PR A (scripts/pad_anet_definitions.py, DEF): analisi
primaria (patience 4, soglia 0,5: variazione di ACER per rete, tre contrasti AttackNet meno LivenessNet sulle stesse 12
coppie fold-seed e sugli stessi soggetti, intervalli bootstrap per soggetto individuali), condizioni descrittive,
graduatorie, denominatori, confronto descrittivo con ResNet-18, guardiano per rete, conteggi a classe unica. Da
pubblicare, e collegare sulla scheda C5, prima che il test dei modelli nuovi sia letto.

Sottocomandi:
- check-inputs: configurazione, file fissati (configurazione della PR A con ciò che fissa, configurazione dei run della
  PR B con ciò che fissa, copia emendata del protocollo, file pubblici della valutazione C2), i sei run (se registrati
  nella configurazione: ultimo tentativo, PASS, file elencati), i 48 file riservati di C2 contro il SHA256SUMS
  ancorato e la riproduzione esatta di C2 dai 48 file (D.primary_analysis, D.compare_condition, units.csv). Non importa
  TensorFlow, non assembla frame, non scrive nulla. Con i run «pending» si ferma dopo la riproduzione di C2 e lo dice.
- evaluate (una sola volta, --i-am-authorized): nell'ordine
  0. prima di creare la cartella: configurazione a insieme chiuso, agganci tutti None nei comandi ufficiali, file
     fissati, regola della macchina e dei thread contro la prova C5, radice d'uscita, i sei run registrati (ultimo
     tentativo del run nella radice dei run, PASS con la fase 1 completa per le 12 posizioni, ogni file uguale al suo
     SHA256SUMS), albero pulito, regola dei tentativi;
  1. status.json RUNNING e run.log; ingressi; file del dev dei modelli valutabili contro i manifest, soglie (0,5 e EER
     del dev, dai punteggi salvati) e piano di valutazione; i 48 file di C2 e la riproduzione esatta di C2 (cancello:
     una differenza -> FAIL prima di qualunque frame del test);
  2. TensorFlow (10 + 10 thread, solo CPU), per ogni modello valutabile il dev riprodotto nei due stati della BN contro
     i punteggi salvati dal run (MISMATCH: il test non è letto);
  3. riverifica, schema dei fold, test di ogni fold assemblato una volta e valutato dai suoi modelli nei due stati;
     punteggi di test non finiti: esito NON_FINITE_TEST con i conteggi, nessuna analisi (PROTOCOL.md §6 (n));
  4. riverifica, scansione del log, unità delle 8 condizioni (le AttackNet nuove e i 48 riferimenti di C2 sugli stessi
     frame), DEF.primary_analysis, DEF.descriptive_analysis, controllo che il confronto descrittivo di LivenessNet con
     ResNet-18 sia l'opposto esatto dei divari e del contrasto di C2, uscite. Nessuna esclusione dopo il test.
Esiti: PASS 0, FAIL 1, INCOMPLETE 2, MISMATCH 4, NON_FINITE_TEST 6.

Uscite (results/aneteval/YYYYMMDD-ANETEVAL-<commit>[-r<k>]): status.json, run.log; public/ results.json, report.md,
table.csv (le cinque reti, analisi primaria), conditions.csv (le 8 condizioni), units.csv, reproduction.csv, SHA256SUMS
(conteggi e SHA-256, nessun identificativo di soggetto o di frame); restricted/ <run>_<rete>_fold<k>_seed<s>_test_scores.csv
(per frame, nei due stati), SHA256SUMS.

  .venv/bin/python -I -B scripts/pad_anet_eval.py check-inputs
  nohup caffeinate -dims .venv/bin/python -I -B scripts/pad_anet_eval.py evaluate --attempt 1 --i-am-authorized > <log fuori dal repository> 2>&1 &

Riuso per importazione: pad_anet_definitions (DEF, l'analisi), pad_arch_definitions (D), pad_arch_eval (E: lettura e
controllo del dev, piano, schema e assemblaggio del test, unità delle condizioni, righe per frame, forme pubbliche),
pad_anet_run (NR: configurazione e file fissati dei run, regola della macchina), pad_anet_comparison (C: configurazione
della PR A, registro), pad_arch_run (R), nuaa_bn_test, nuaa_bn_recalibration, pad_arch_comparison. Copie dichiarate con
sostituzioni, confrontate con il sorgente di C2 dalla suite: score_model (registro della PR A), reproduce_dev e
score_test (l'aggancio di questo modulo). Formati: configurazione pad-anet-eval-config/1, risultati pad-anet-eval/1.
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
import pad_anet_comparison as C  # noqa: E402
import pad_anet_definitions as DEF  # noqa: E402
import pad_anet_models as AN  # noqa: E402
import pad_anet_run as NR  # noqa: E402
import pad_arch_comparison as PAC  # noqa: E402
import pad_arch_definitions as D  # noqa: E402
import pad_arch_eval as E  # noqa: E402
import pad_arch_run as R  # noqa: E402
import path_guard as pg  # noqa: E402

Failure, fail = FR.Failure, FR.fail
sha256_bytes, sha256_file, read_regular, rel_path, csv_text, parse_csv = (FR.sha256_bytes, FR.sha256_file, FR.read_regular,
                                                                          FR.rel_path, FR.csv_text, FR.parse_csv)
now = FR.now

DEFAULT_CONFIG = "delivery/aneteval/pad_anet_eval.json"
DEFAULT_OUT_ROOT = "results/aneteval"
RUNS_ROOT = NR.DEFAULT_OUT_ROOT                      # results/anetrun
CONFIG_SCHEMA = "pad-anet-eval-config/1"
EVAL_SCHEMA = "pad-anet-eval/1"
EVAL_ID_RE = re.compile(r"^\d{8}-ANETEVAL-[0-9a-f]{7,40}(?:-r([2-9]|[1-9]\d+))?$")
LOOKS_LIKE_EVAL = re.compile(r"^\d{8}-ANETEVAL-", re.IGNORECASE)
THIS_FILE = "scripts/pad_anet_eval.py"
CODE_FILES = NR.CODE_FILES + (NR.THIS_FILE, E.THIS_FILE)
RUNS_PENDING = "RUNS_PENDING"
RECORD_FILES = ("status.json", "run.log", "public/", "conditions.txt")       # i record pubblicati di ogni comando dei run (come C2, più le condizioni)
TOP_KEYS = ("schema", "task", "authorisation", "operations", "comparison", "run_configuration", "protocol", "c2_evaluation", "code", "runs",
            "run_records",
            "attempt_rule", "after_stop", "dev_reproduction", "c2_reproduction", "test", "non_finite_test", "analysis", "cap", "outputs",
            "choices_of_the_author", "test_read")
CHOICE_KEYS = ("thresholds_source", "c2_reproduction_gate", "c2_units_bytes", "non_finite_test_outcome", "reused_thresholds", "units_order",
               "cap", "attempts", "outputs", "runs_record", "second_reading", "test_one_class", "run_records", "launch_conditions")
STATUS_CODE = {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2, "MISMATCH": 4, "NON_FINITE_TEST": 6}
STATUS, LOG, PUBLIC, RESTRICTED = FR.STATUS, FR.LOG, "public", "restricted"
PUBLIC_OUTPUTS = ("results.json", "report.md", "table.csv", "conditions.csv", "units.csv", "reproduction.csv", "SHA256SUMS")
TEST_SUFFIX = E.TEST_SUFFIX
CAP_UNIT = PAC.RUN_CAP_UNIT
DATASET = E.DATASET
ARM = R.ARM
C2_TAG = {"nuaa_patience_4": "p4", "nuaa_patience_12": "p12"}
TABLE_COLUMNS = ("network", "bn_state", "units", "apcer_mean", "bpcer_mean", "acer_mean", "n_attack", "n_bona_fide", "n_subjects", "computed")
NEW_CONDITION_COLUMNS = ("condition", "quantity", "network", "computed", "acer_mean", "acer_low", "acer_high", "acer_valid_resamples",
                         "apcer_mean", "apcer_low", "apcer_high", "bpcer_mean", "bpcer_low", "bpcer_high")
NEW_UNIT_COLUMNS = ("patience", "bn_state", "threshold", "fold", "seed", "n_frames", "frame_ids_sha256", "network", "threshold_value", "apcer",
                    "bpcer", "acer")
EXTRA_PROTECTED = NR.EXTRA_PROTECTED + (RUNS_ROOT, "results/archeval")
OFFICIAL_RUNS_DIR = ROOT / RUNS_ROOT
TEST_SENTENCE = "the NUAA test of the new models is read once, by evaluate, after every check that does not need it"
TEST_READ = (
    f"{TEST_SENTENCE}. check-inputs: this configuration; the PR A configuration and what it pins; the configuration of the runs "
    "(pad_anet_run.load_config and load_pins: the PR A package, the trial, the plan, the PR #27 and #28 configurations); the amended "
    "protocol of the PR B package; the public status.json, results.json and units.csv of the C2 evaluation 20261121-ARCHEVAL-332fbdf and "
    "its restricted SHA256SUMS with the 48 <p4|p12>_<network>_fold<k>_seed<s>_test_scores.csv it lists (the C2 test scores, already "
    "published as aggregates: read to reproduce C2 exactly); when the runs are recorded, their status.json, run.log, SHA256SUMS and "
    "the files they list. evaluate, in addition: what pad_arch_comparison.load_inputs reads (manifests, cache, extracted frames, cells "
    "and folds runs); the dev scores and BN statistics of each evaluable new model; its checkpoint; after the dev reproduction of every "
    "model, the fold schema and the test frames of each fold, assembled once")

# punti di iniezione (solo i test)
REPO_STATE = None       # (root) -> {"commit", "dirty"} (default: SR.repo_state)
SCORE_MODEL = None      # (ctx, rec, parts) -> {stato: {ruolo: {"p", "z", "softmax_gap"}}} (default: score_model)
MY_HOOKS = ("REPO_STATE", "SCORE_MODEL")
HOOKS = {**NR.HOOKS, "pad_arch_eval": ("SCORE_MODEL", "PLAN_HOOK", "UNITS_HOOK"), "pad_anet_eval": MY_HOOKS}
PINNED_VALUES = NR.PINNED_VALUES


# ============================================================================ configurazione

def _text(v, what):
    NR._text(v, what)


def _keys(d, keys, what):
    NR._keys(d, keys, what)


def _pin(e, what):
    NR._pin(e, what)


def run_record_keys() -> list:
    return ["run_id", "status_sha256", "public_sha256sums_sha256", "restricted_sha256sums_sha256"]


def load_config(p: Path) -> tuple:
    """(configurazione, SHA-256). Insieme chiuso; i sei run sono «RUNS_PENDING» prima dei run oppure, dopo, sei record
    (run id del run giusto, SHA-256 di status.json e dei due SHA256SUMS), scritti da make_config dalle cartelle dei run."""
    b = read_regular(p, "configuration")
    cfg = SR.read_json(b, "configuration")
    _keys(cfg, TOP_KEYS, "the top level")
    if cfg["schema"] != CONFIG_SCHEMA:
        fail(f"configuration: schema must be {CONFIG_SCHEMA}")
    for k in ("task", "authorisation", "operations", "attempt_rule", "test_read"):
        _text(cfg[k], k)
    if TEST_SENTENCE not in cfg["test_read"]:
        fail("configuration: test_read must state when the test of the new models is read")
    for k in ("comparison", "run_configuration", "protocol"):
        _pin(cfg[k], k)
    ce = cfg["c2_evaluation"]
    _keys(ce, ["run_id", "units", "restricted_folder", "restricted_sha256sums_sha256"], "c2_evaluation")
    if ce["run_id"] != "20261121-ARCHEVAL-332fbdf" or not SR._hex(ce["restricted_sha256sums_sha256"]):
        fail("configuration: c2_evaluation must name 20261121-ARCHEVAL-332fbdf and the SHA-256 of its restricted SHA256SUMS")
    _pin(ce["units"], "c2_evaluation.units")
    _text(ce["restricted_folder"], "c2_evaluation.restricted_folder")
    if not isinstance(cfg["code"], dict) or sorted(cfg["code"]) != sorted(CODE_FILES) or not all(SR._hex(v) for v in cfg["code"].values()):
        fail(f"configuration: code must pin exactly {', '.join(CODE_FILES)} by SHA-256")
    runs = cfg["runs"]
    if runs != RUNS_PENDING:
        _keys(runs, list(NR.RUNS), "runs")
        for r, e in runs.items():
            _keys(e, run_record_keys(), f"runs.{r}")
            m = NR.RUN_ID_RE.match(e["run_id"]) if isinstance(e["run_id"], str) else None
            if not m or f"{m.group(1)}-p{m.group(2)}" != NR.RUN_TAG[r]:
                fail(f"configuration: runs.{r}.run_id must be a run id of {r}")
            if not all(SR._hex(e[k]) for k in run_record_keys()[1:]):
                fail(f"configuration: runs.{r} SHA-256 malformed")
    for k in ("dev_reproduction", "c2_reproduction", "test", "non_finite_test", "analysis"):
        _text(cfg[k], k)
    rr = cfg["run_records"]
    _keys(rr, ["folder", "files", "rule"], "run_records")
    _text(rr["folder"], "run_records.folder")
    if rr["files"] != list(RECORD_FILES):
        fail(f"configuration: run_records.files must be {list(RECORD_FILES)}")
    _text(rr["rule"], "run_records.rule")
    NR._author(cfg["after_stop"], "after_stop")
    if "no automatic resume" not in cfg["after_stop"] or "test_read_started" not in cfg["after_stop"] or "second reading" not in cfg["after_stop"] \
            or "C5 card" not in cfg["after_stop"]:
        fail("configuration: after_stop must state that there is no automatic resume and that a new attempt after test_read_started is a "
             "second reading of the test, started only after the outcome has been reported on the C5 card")
    if "restricted" not in cfg["non_finite_test"]:
        fail("configuration: non_finite_test must state what is kept in restricted/")
    cap = cfg["cap"]
    _keys(cap, ["hours", "unit", "rule"], "cap")
    if not isinstance(cap["hours"], (int, float)) or isinstance(cap["hours"], bool) or not math.isfinite(cap["hours"]) or cap["hours"] <= 0 \
            or cap["unit"] != CAP_UNIT:
        fail(f"configuration: cap.hours must be a positive number of {CAP_UNIT}")
    _text(cap["rule"], "cap.rule")
    out = cfg["outputs"]
    _keys(out, ["run", "public", "restricted"], "outputs")
    if out["run"] != [STATUS, LOG] or out["public"] != list(PUBLIC_OUTPUTS) or out["restricted"] != [f"<run>_<model>{TEST_SUFFIX}", "SHA256SUMS"]:
        fail("configuration: outputs differ from the outputs of this command")
    _keys(cfg["choices_of_the_author"], CHOICE_KEYS, "choices_of_the_author")
    for k, v in cfg["choices_of_the_author"].items():
        NR._author(v, f"choices_of_the_author.{k}")
    return cfg, sha256_bytes(b)


class Pins:
    """Ciò che la configurazione fissa, letto e controllato."""


def check_code(cfg: dict) -> dict:
    got = {f: sha256_bytes(read_regular(ROOT / f, "code file")) for f in CODE_FILES}
    bad = [f for f in CODE_FILES if got[f] != cfg["code"][f]]
    if bad:
        fail(f"code files {bad} differ from the SHA-256 pinned in the configuration")
    return {**got, THIS_FILE: sha256_bytes(read_regular(Path(__file__), "code file"))}


def load_pins(cfg: dict, base: Path) -> Pins:
    """I file fissati: configurazione dei run (pad_anet_run.load_config e load_pins: configurazione della PR A, prova,
    piano, PR #27 e #28), la stessa configurazione della PR A, la copia emendata del protocollo, il codice (anche quello
    fissato dalla configurazione dei run), i file pubblici della valutazione C2 (results.json e status.json dalla
    configurazione della PR A, units.csv qui)."""
    pins = Pins()
    pins.files = {}
    rp, _ = R.read_pinned(base, cfg["run_configuration"], "configuration of the runs")
    pins.rcfg_p = rp
    pins.rcfg, pins.rcfg_sha = NR.load_config(rp)
    pins.rpins = NR.load_pins(pins.rcfg, base)
    if pins.rcfg["comparison"] != cfg["comparison"]:
        fail("comparison: the PR A configuration pinned here differs from the one of the configuration of the runs")
    pins.cfg_c5, pins.cfg_pr, pins.pcfg, pins.bn_cfg, pins.trial = (pins.rpins.cfg_c5, pins.rpins.cfg_pr, pins.rpins.pcfg, pins.rpins.bn_cfg,
                                                                     pins.rpins.trial)
    pins.code = check_code(cfg)
    run_code = NR.check_code(pins.rcfg)
    if {k: v for k, v in pins.code.items() if k in run_code} != run_code:
        fail("code: the files pinned by the configuration of the runs differ from those pinned here")
    pins.run_code = run_code
    _, pb = R.read_pinned(base, cfg["protocol"], "amended protocol")
    pins.protocol = pb.decode("utf-8")
    ce, cr = cfg["c2_evaluation"], pins.cfg_c5["c2_results"]
    if ce["run_id"] != cr["evaluation_run_id"] or ce["restricted_sha256sums_sha256"] != cr["restricted_sha256sums_sha256"]:
        fail("c2_evaluation: run id or restricted SHA256SUMS differ from the PR A configuration")
    _, rb = R.read_pinned(base, cr["results"], "C2 results.json")
    pins.c2_results = SR.read_json(rb, "C2 results.json")
    _, pins.c2_units = R.read_pinned(base, ce["units"], "C2 units.csv")
    pins.c2_cells = C.load_c2_results(pins.cfg_c5, base)
    pins.files.update({"run_configuration": pins.rcfg_sha, "comparison": cfg["comparison"]["sha256"], "protocol": cfg["protocol"]["sha256"],
                       "c2_units": ce["units"]["sha256"], "c2_results": cr["results"]["sha256"]})
    pins.folds, pins.seeds = list(DEF.FOLDS), list(DEF.PLAN_SEEDS)
    pins.test_per_fold = pins.pcfg["schema_expected"]["test_per_fold"]
    return pins


def check_hooks(official: bool) -> dict:
    mods = {m: sys.modules[__name__] if m == "pad_anet_eval" else sys.modules[m] for m in HOOKS}
    setv = sorted(f"{m}.{k}" for m, names in HOOKS.items() for k in names if getattr(mods[m], k) is not None)
    changed = sorted(f"{m}.{k}" for (m, k), v in PINNED_VALUES.items() if getattr(sys.modules[m], k) is not v)
    if official and (setv or changed):
        fail(f"test hooks are set or test values changed ({', '.join(setv + changed)}): the official commands run with every hook None")
    return {"official_configuration": official, "set": setv, "changed": changed}


def is_official_config(cfg_p: Path) -> bool:
    try:
        return os.path.samefile(cfg_p, ROOT / DEFAULT_CONFIG)
    except OSError:
        return False


# ============================================================================ i 48 riferimenti di C2 e la riproduzione di C2

def load_c2_scores(cfg: dict, pins: Pins, base: Path) -> dict:
    """I 48 file riservati dei punteggi di test di C2 (uno per cella riusata di plan/reference_cells.csv), ciascuno elencato
    nel SHA256SUMS riservato della valutazione C2 (il cui SHA-256 è quello fissato) e uguale alla sua riga; per file: righe
    della rete, del fold, del seed e della patience della cella, del checkpoint fissato, nei due stati della BN, stesse
    righe nei due stati; punteggi finiti; prediction_05 con la regola del progetto. (patience, rete, fold, seed) ->
    {stato: {"y", "s", "subject", "ids", "threshold_eer"}}."""
    folder = rel_path(base, cfg["c2_evaluation"]["restricted_folder"])
    sb = read_regular(folder / "SHA256SUMS", "C2 restricted SHA256SUMS")
    if sha256_bytes(sb) != cfg["c2_evaluation"]["restricted_sha256sums_sha256"]:
        fail("C2 restricted SHA256SUMS: SHA-256 differs from the configuration")
    sums = FR.read_sums(sb, "C2 restricted SHA256SUMS")
    out = {}
    for c in pins.c2_cells:
        fn = c["test_scores_file"]
        if fn not in sums:
            fail(f"C2 restricted SHA256SUMS does not list {fn}")
        b = read_regular(folder / fn, fn)
        if sha256_bytes(b) != sums[fn]:
            fail(f"C2 {fn}: SHA-256 differs from the restricted SHA256SUMS")
        rows = parse_csv(b, E.TEST_SCORE_COLUMNS, fn)
        rec = {}
        for st in D.BN_STATES:
            got = [g for g in rows if g["bn_state"] == st]
            if not got or any((g["network"], g["fold"], g["train_seed"], g["patience"], g["role"], g["checkpoint_sha256"]) !=
                              (c["network"], str(c["fold"]), str(c["seed"]), str(c["patience"]), "test", c["checkpoint_sha256"]) for g in got):
                fail(f"C2 {fn}: rows of another cell, role or checkpoint, or no row with {st} BN")
            try:
                s = np.array([float(g["score_attack"]) for g in got], np.float64)
                thr = {float(g["threshold_dev_eer"]) for g in got}
            except ValueError:
                fail(f"C2 {fn}: scores or thresholds not numeric")
            y, s = D.labels_scores(np.array([int(g["label"]) for g in got]), s, f"C2 {fn} {st}")
            if any(g["prediction_05"] != ("1" if d else "0") for g, d in zip(got, D.attack_decisions(s, FR.FIXED_THRESHOLD))):
                fail(f"C2 {fn}: prediction_05 differs from the rule of the project")
            if len(thr) != 1:
                fail(f"C2 {fn}: more than one EER threshold with {st} BN")
            t = D.check_threshold(thr.pop(), f"C2 {fn}")
            if any(g["prediction_dev_eer"] != ("1" if d else "0") for g, d in zip(got, D.attack_decisions(s, t))):
                fail(f"C2 {fn}: prediction_dev_eer differs from the rule of the project at its threshold")
            rec[st] = {"y": y, "s": s, "subject": np.array([g["subject_id"] for g in got]), "ids": [g[D.FRAME_ID_FIELD] for g in got],
                       "threshold_eer": t}
        if rec["ordinary"]["ids"] != rec["re-estimated"]["ids"]:
            fail(f"C2 {fn}: the two BN states are not on the same frames in the same order")
        out[(c["patience"], c["network"], c["fold"], c["seed"])] = rec
    return out


def c2_conditions(c2: dict) -> dict:
    """Le unità delle 8 condizioni con le sole due reti di C2 (la forma di pad_arch_eval.build_conditions), dai file:
    soglia 0,5 o la soglia EER del dev salvata con i punteggi."""
    conds = {c: [] for c in D.CONDITIONS}
    for (p, st, rule) in D.CONDITIONS:
        for f in DEF.FOLDS:
            for s in DEF.PLAN_SEEDS:
                nets = {}
                for n in DEF.REUSED:
                    r = c2[(p, n, f, s)][st]
                    nets[n] = {"y": r["y"], "s": r["s"], "subject": r["subject"],
                               "threshold": float(FR.FIXED_THRESHOLD) if rule == "fixed_0.5" else r["threshold_eer"], D.FRAME_ID_FIELD: list(r["ids"])}
                conds[(p, st, rule)].append({"unit": (DATASET, f, s), "networks": nets})
    return conds


def c2_ids_check(c2: dict, lists: dict) -> None:
    """F4: prima di assemblare il test, gli ID dei frame di ogni file di C2 uguali alle chiavi del test del suo fold, in
    ordine di chiave (le reti nuove saranno valutate sugli stessi frame)."""
    for (p, n, f, s), rec in c2.items():
        keys = sorted(r["key"] for r in lists[f])
        if rec["ordinary"]["ids"] != keys:
            fail(f"C2 reference {n} fold {f} seed {s} patience {p}: its frame identifiers are not the test keys of fold {f} in key order")


def c2_thresholds_check(pins: Pins, c2: dict) -> int:
    """F5 (PROTOCOL.md §4: le soglie EER dei riferimenti dal results.json di C2, models[].dev_eer_threshold): la soglia
    EER di ogni file di C2, per stato della BN, uguale a quella del suo modello in results.json."""
    by = {}
    for m in pins.c2_results.get("models") or []:
        p = 4 if m.get("run") == "nuaa_patience_4" else 12
        by[(p, m.get("network"), m.get("fold"), m.get("seed"))] = m.get("dev_eer_threshold")
    for key, rec in c2.items():
        t = by.get(key)
        if not isinstance(t, dict) or any(t.get(st) != rec[st]["threshold_eer"] for st in D.BN_STATES):
            fail(f"C2 reference {key}: the dev EER threshold of its file differs from models[].dev_eer_threshold of the C2 results.json")
    return len(c2)


def c2_opposite_before_test(pins: Pins, c2: dict) -> dict:
    """B4: l'opposto esatto di C2 sulle sole unità di C2, prima di qualunque lettura del test nuovo: DEF.primary_analysis
    con LivenessNet e ResNet-18 soli (le AttackNet «not computed»: il confronto descrittivo di LivenessNet con ResNet-18
    dipende solo dalle due reti di C2 e dagli stessi soggetti), poi c2_opposite_check. Una differenza -> Failure."""
    conds = c2_conditions(c2)
    pa = DEF.primary_analysis({c: conds[c] for c in E.PRIMARY})
    out = c2_opposite_check(pa, pins)
    if not all(out.values()):
        fail(f"the descriptive comparison of LivenessNet with ResNet-18 on the C2 units is not the exact opposite of C2: {out}")
    return out


def reproduce_c2(pins: Pins, c2: dict) -> dict:
    """Il cancello prima del test nuovo (piano §D): dai 48 file l'analisi primaria (D.primary_analysis), le 8 condizioni
    (D.compare_condition) e units.csv di C2, confrontati con public/results.json e units.csv della valutazione C2 per
    uguaglianza esatta (frazioni e float degli stessi calcoli: +0,0998 [0,0256; 0,1632], 926 ricampionamenti validi). Una
    differenza -> Failure."""
    conds = c2_conditions(c2)
    pa = E.jsonable(D.primary_analysis({c: conds[c] for c in E.PRIMARY}))
    want = pins.c2_results["primary_analysis"]
    bad = [k for k in ("conditions", "definition", "gap_ordinary", "gap_re_estimated", "contrast", "per_seed", "rows") if pa.get(k) != want.get(k)]
    if bad:
        fail(f"C2 reproduction: the primary analysis differs from the C2 results.json in {', '.join(bad)}")
    results = {c: D.compare_condition(conds[c]) for c in D.CONDITIONS}
    pub = {str(list(c)): E.jsonable(E.public_condition(results[c])) for c in D.CONDITIONS}
    theirs = {str(x["condition"]): {k: v for k, v in x.items() if k not in ("condition", "label")} for x in pins.c2_results["conditions"]}
    if pub != theirs:
        fail("C2 reproduction: the 8 conditions differ from the C2 results.json")
    units = csv_text(E.UNIT_COLUMNS, [[E.cell(x[k]) for k in E.UNIT_COLUMNS] for c in D.CONDITIONS for x in E.unit_rows(c, results[c], conds)])
    if units.encode("utf-8") != pins.c2_units:
        fail("C2 reproduction: units.csv differs from the C2 units.csv")
    ct = want["contrast"]
    return {"primary_contrast": {"mean": ct.get("mean"), "interval": ct.get("interval")}, "conditions": len(results),
            "units_csv_sha256": sha256_bytes(pins.c2_units), "exact": True}


# ============================================================================ i sei run

def latest_attempt(run: str, root: Path) -> str:
    """L'ultimo tentativo di un run nella radice dei run (pad_anet_run.previous_attempts): l'unico valutato (emendamento
    di PROTOCOL.md §6 (k)/(n))."""
    prev = NR.previous_attempts(root, "run", run)
    if not prev:
        fail(f"run {run}: no attempt in the runs root")
    return max(prev, key=lambda p: p["attempt"])["run_id"]


def load_run(folder: Path, run: str, pins: Pins, record: dict, mode: str) -> dict:
    """Una cartella di run di pad_anet_run.py (la forma di pad_arch_run, copia di pad_arch_eval.load_run per le AttackNet):
    run id e SHA-256 registrati nella configurazione; status.json PASS, con la configurazione e il codice dei run, albero
    pulito, nessun arresto, fase 1 completa per le 12 posizioni (non valutabili compresi), fase 2 per i valutabili, nessun
    disaccordo, regola della macchina e dei thread; ufficiale e in results/anetrun per una valutazione ufficiale; i due
    SHA256SUMS e ogni file; per modello valutabile scheda, checkpoint, statistiche installate e i file del dev; per
    modello non valutabile il motivo registrato."""
    what = f"run {folder.name}"
    if folder.name != record["run_id"]:
        fail(f"{what}: the configuration records {record['run_id']} for {run}")
    if not folder.is_dir() or folder.is_symlink():
        fail(f"{what}: missing, not a folder or a symbolic link")
    sb = read_regular(folder / STATUS, f"{what} status.json")
    if sha256_bytes(sb) != record["status_sha256"]:
        fail(f"{what}: status.json differs from the SHA-256 recorded in the configuration")
    st = SR.read_json(sb, f"{what} status.json")
    if st.get("status") != "PASS" or st.get("passed") is not True or st.get("run_id") != folder.name or st.get("run") != run:
        fail(f"{what}: status.json is not a PASS of {run}")
    if st.get("config_sha256") != pins.rcfg_sha or st.get("code_sha256") != pins.run_code or st.get("dirty") is not False:
        fail(f"{what}: produced with another configuration or code of the runs, or from a work tree that was not clean")
    inside = pg.within(folder, OFFICIAL_RUNS_DIR)
    if mode == "official" and (st.get("official") is not True or not inside):
        fail(f"{what}: an official evaluation needs an official run in {RUNS_ROOT}")
    if mode == "unofficial" and (st.get("official") is not False or inside):
        fail(f"{what}: an official run, or a folder in {RUNS_ROOT}, is never read by an unofficial evaluation")
    positions = pins.rpins.positions[run]
    names = [PAC.training_name(p["network"], p["fold"], p["seed"]) for p in positions]
    ne = {x["name"]: x["reason"] for x in st.get("not_evaluable") or []}
    if st.get("stopped") is not None or st.get("missing_phase_1") != [] or st.get("evaluator_mismatches") != [] \
            or st.get("complete_phase_1") != names:
        fail(f"{what}: stopped, or phase 1 not complete for the {len(names)} positions, or an evaluator mismatch")
    if st.get("complete_phase_2") != [n for n in names if n not in ne] or any(n not in names for n in ne):
        fail(f"{what}: the models of phase 2 are not the evaluable ones")
    if (st.get("machine_rule") or {}).get("equal") is not True or (st.get("threads_rule") or {}).get("equal") is not True:
        fail(f"{what}: the machine and thread rule was not met")
    if sha256_file(FR.regular_path(folder / LOG, "run.log")) != st.get("run_log_sha256"):
        fail(f"{what}: run.log differs from run_log_sha256 of its status.json")
    sums = {}
    for sub, key, rkey in ((PUBLIC, "public_sha256sums_sha256", "public_sha256sums_sha256"),
                           (RESTRICTED, "restricted_sha256sums_sha256", "restricted_sha256sums_sha256")):
        b = read_regular(folder / sub / "SHA256SUMS", f"{what} {sub} SHA256SUMS")
        if sha256_bytes(b) != st.get(key) or sha256_bytes(b) != record[rkey]:
            fail(f"{what}: the {sub} SHA256SUMS differs from its status.json or from the configuration")
        sums[sub] = FR.read_sums(b, f"{what} {sub} SHA256SUMS")
        for fn, h in sums[sub].items():
            if sha256_file(FR.regular_path(folder / sub / fn, fn)) != h:
                fail(f"{what}: {sub}/{fn} differs from its SHA256SUMS line")
    if sorted(sums[PUBLIC]) != sorted(n for n in NR.PUBLIC_OUTPUTS if n != "SHA256SUMS"):
        fail(f"{what}: the public SHA256SUMS does not list exactly the public files of the runner")
    recs, not_eval = [], []
    rd = folder / RESTRICTED
    for p, name in zip(positions, names):
        if f"{name}_card.json" not in sums[RESTRICTED]:
            fail(f"{what}: {name}_card.json not listed")
        card = SR.read_json(read_regular(rd / f"{name}_card.json", "card"), f"{name}_card.json")
        if (card.get("network"), card.get("fold"), card.get("train_seed"), card.get("early_stopping_patience"), card.get("run"),
                card.get("position")) != (p["network"], p["fold"], p["seed"], p["patience"], run, p["position"]):
            fail(f"{what}: {name}_card.json differs from its position")
        if name in ne:
            not_eval.append({"run": run, "name": name, "network": p["network"], "fold": p["fold"], "seed": p["seed"], "patience": p["patience"],
                             "reason": ne[name]})
            continue
        if any(f not in sums[RESTRICTED] for f in R.model_files(name)):
            fail(f"{what}: the files of {name} are not all listed")
        wsha = sums[RESTRICTED][f"{name}.keras"]
        if card.get("weights_sha256") != wsha:
            fail(f"{what}: {name}_card.json differs from its checkpoint")
        sd = SR.read_json(read_regular(rd / f"{name}_bn_statistics.json", "statistics"), f"{name}_bn_statistics.json")
        order = pins.rcfg["estimator"]["orders"][p["network"]]
        if (sd.get("model"), sd.get("network"), sd.get("weights_sha256"), sd.get("order")) != (name, p["network"], wsha, order) \
                or list(sd.get("layers") or {}) != order or (sd.get("evaluator_agreement") or {}).get("pass") is not True:
            fail(f"{what}: {name}_bn_statistics.json differs from the model, its checkpoint or its order, or records no agreement")
        installed = {}
        for n in order:
            lay = sd["layers"][n]
            mean = E._float32_list(lay.get("installed_moving_mean_float32"), f"{what}: {name} {n} mean")
            var = E._float32_list(lay.get("installed_moving_variance_float32"), f"{what}: {name} {n} variance")
            if mean.shape != var.shape or not np.all(var >= 0):
                fail(f"{what}: {name} {n}: installed mean and variance of different lengths, or a negative variance")
            installed[n] = (mean, var)
        recs.append({"run": run, "tag": NR.RUN_TAG[run], "patience": p["patience"], "position": p["position"], "network": p["network"],
                     "fold": p["fold"], "seed": p["seed"], "name": name, "key": f"{run}_{name}", "weights": rd / f"{name}.keras",
                     "weights_sha256": wsha, "order": order, "installed": installed,
                     "dev_files": {s: (rd / f"{name}_dev_{s}.csv", sums[RESTRICTED][f"{name}_dev_{s}.csv"]) for s in D.BN_STATES}})
    return {"run": run, "folder": folder, "run_id": folder.name, "status_sha256": sha256_bytes(sb), "official": st.get("official") is True,
            "sums": sums, "run_log_sha256": st["run_log_sha256"], "models": recs, "not_evaluable": not_eval}


def load_runs(cfg: dict, pins: Pins, mode: str, runs_root: Path) -> dict:
    """I sei run registrati nella configurazione, ciascuno l'ultimo tentativo del suo run nella radice dei run."""
    if cfg["runs"] == RUNS_PENDING:
        fail("the six runs are not recorded in the configuration yet (runs: RUNS_PENDING): regenerate it with make_config.py after the runs")
    out = {}
    for r in NR.RUNS:
        rec = cfg["runs"][r]
        last = latest_attempt(r, runs_root)
        if last != rec["run_id"]:
            fail(f"run {r}: the configuration records {rec['run_id']}, the last attempt in the runs root is {last} (only the last attempt is evaluated)")
        out[r] = load_run(runs_root / rec["run_id"], r, pins, rec, mode)
    return out


def record_folders(runs_root: Path) -> list:
    """Le cartelle dei comandi dei run da pubblicare: ogni tentativo di ciascuno dei sei run nella radice dei run
    (l'emendamento di PROTOCOL.md §6: i tentativi precedenti si conservano) e ogni record di check-estimator."""
    out = [runs_root / p["run_id"] for r in NR.RUNS for p in NR.previous_attempts(runs_root, "run", r)]
    out += [runs_root / p["run_id"] for p in NR.previous_attempts(runs_root, "estimator")]
    return sorted(out, key=lambda f: f.name)


def record_files(folder: Path) -> list:
    """I file pubblicabili di una cartella di comando dei run: status.json, run.log e ogni file di public/ (nulla di
    restricted/), con i percorsi relativi alla cartella."""
    names = [STATUS, LOG] + [f"{PUBLIC}/{n}" for n in sorted(os.listdir(folder / PUBLIC))] if (folder / PUBLIC).is_dir() else [STATUS, LOG]
    return [n for n in names if (folder / n).is_file()]


def check_run_copies(cfg: dict, base: Path, runs_root: Path, inp=None) -> dict:
    """Le copie dei record dei run nel pacchetto (run_records.folder/<run id>/): per ogni cartella di record_folders le
    stesse cartelle, e per ciascuna status.json, run.log e public/ identici byte per byte ai file della radice dei run,
    con in più conditions.txt (il file delle condizioni dello script di lancio); nessun'altra cartella o file; nessun
    testo vietato nelle copie (PAC.output_problems). Senza inp (check-inputs, che non carica i manifest) la scansione
    trova percorsi e testi personali ma non gli identificativi di soggetto o di frame dei manifest; con inp (evaluate,
    dopo il caricamento dei manifest) anche quelli. Per run.log e public/ la scansione è ridondante (il runner li ha già
    scansionati con gli ingressi reali); conditions.txt contiene solo le uscite di check-machine e check-inputs.
    Restituisce cartelle e file confrontati."""
    pkg = rel_path(base, cfg["run_records"]["folder"])
    want = record_folders(runs_root)
    have = sorted(e for e in os.listdir(pkg)) if pkg.is_dir() else []
    if have != [f.name for f in want]:
        fail(f"run records: the package holds {have}, the runs root {[f.name for f in want]} (every attempt and every check-estimator)")
    n = 0
    for f in want:
        names = record_files(f)
        got = sorted(str(p.relative_to(pkg / f.name)) for p in (pkg / f.name).rglob("*") if p.is_file())
        if got != sorted(names + ["conditions.txt"]):
            fail(f"run records {f.name}: the copy holds {got}, expected {sorted(names + ['conditions.txt'])}")
        for nm in names:
            if read_regular(pkg / f.name / nm, nm) != read_regular(f / nm, nm):
                fail(f"run records {f.name}: {nm} differs from the file of the runs root")
            n += 1
        for nm in names + ["conditions.txt"]:
            if probs := PAC.output_problems(read_regular(pkg / f.name / nm, nm).decode("utf-8", "replace"), inp or FR.Inputs()):
                fail(f"run records {f.name}: {nm} holds text that a publishable output must not hold: " + "; ".join(probs))
    return {"folders": len(want), "files_compared": n}


def recheck_run(r: dict) -> None:
    E.recheck_run(r)


# ============================================================================ riproduzione del dev e test (copie dichiarate)

class Ctx:
    """Lo stato della valutazione in corso."""


def score_model(ctx, rec: dict, parts: dict) -> dict:
    """Copia dichiarata di pad_arch_eval.score_model: il registro della PR A (pad_anet_comparison.check_loaded) al posto di
    pad_arch_run.check_loaded; tutto il resto uguale."""
    tf = BR.tf_module()
    tf.keras.backend.clear_session()
    name, order = rec["key"], rec["order"]
    orig = BR.load_model(rec["weights"], rec["weights_sha256"])
    C.check_loaded(orig, ctx.cfg_c5, rec["network"])
    out = {s: {} for s in D.BN_STATES}
    for role, part in parts.items():
        out["ordinary"][role] = E.forward_scores(orig, part["X"], ctx.batch, ctx.sg, f"{name} {role} ordinary")
    snap = BR.snapshot_weights(orig)
    copy = tf.keras.models.clone_model(orig)
    copy.set_weights(orig.get_weights())
    if BR.snapshot_weights(copy) != snap:
        fail(f"{name}: the in-memory copy differs from the checkpoint before the statistics are installed")
    stats = BT.install(copy, rec["installed"])
    if probs := BR.tensor_problems(snap, BR.snapshot_weights(copy), BR.moving_keys(copy, order), stats):
        fail(f"{name}: re-estimated copy: " + "; ".join(probs))
    for role, part in parts.items():
        out["re-estimated"][role] = E.forward_scores(copy, part["X"], ctx.batch, ctx.sg, f"{name} {role} re-estimated")
    if BR.snapshot_weights(orig) != snap:
        fail(f"{name}: the checkpoint model changed while the copy was scored")
    del orig, copy
    gc.collect()
    return out


def reproduce_dev(ctx, recs: list) -> list:
    """Copia dichiarata di pad_arch_eval.reproduce_dev con l'aggancio di questo modulo."""
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


def score_test(ctx, recs: list, lists: dict) -> dict:
    """Copia dichiarata di pad_arch_eval.score_test con l'aggancio di questo modulo, e i punteggi di test non finiti
    contati (regola (n): «a model with non-finite test scores in either BN state is reported with the count of such
    frames; the evaluation stops before the primary analysis»): forward_scores li rifiuterebbe, quindi qui si usa
    test_scores, che li conta. Restituisce (meta, non finiti)."""
    meta, non_finite = {}, []
    for f in ctx.folds:
        ctx.budget.check(f"test assembly fold {f}")
        test = FR.assemble(ctx.store, lists[f])
        if not test["keys"] or test["keys"] != sorted(r["key"] for r in lists[f]):
            fail(f"fold {f}: the assembled test frames are empty or not in key order")
        meta[f] = {"rows": test["rows"], "ids": list(test["keys"]), "y": test["y"], "subject": test["s"]}
        for rec in [r for r in recs if r["fold"] == f]:
            ctx.budget.check(f"test scoring {rec['key']}")
            res = (SCORE_MODEL or test_scores)(ctx, rec, {"test": test})
            rec["test"] = {s: res[s]["test"] for s in D.BN_STATES}
            for s in D.BN_STATES:
                if len(rec["test"][s]["p"]) != len(test["keys"]):
                    fail(f"{rec['key']}: {len(rec['test'][s]['p'])} test scores for {len(test['keys'])} frames")
                bad = rec["test"][s].get("non_finite")
                if bad is None:
                    bad = int(np.sum(~np.isfinite(np.asarray(rec["test"][s]["p"], np.float64))))
                if bad:
                    non_finite.append({"model": rec["key"], "bn_state": s, "non_finite_frames": bad})
            # i punteggi riservati scritti appena calcolati: dopo la lettura del test restano anche con FAIL o INCOMPLETE
            mw.write_all(ctx.rdir, {f"{rec['key']}{TEST_SUFFIX}": csv_text(E.TEST_SCORE_COLUMNS, E.test_score_rows(rec, meta))}, newline="")
            print(f"   test scored {rec['key']} (two BN states)", flush=True)
        del test
        gc.collect()
    return meta, non_finite


def test_scores(ctx, rec: dict, parts: dict) -> dict:
    """score_model con i punteggi non finiti contati invece che rifiutati (solo per il test): stessa sequenza, il
    passaggio di BR.forward senza il rifiuto di forward_scores; probabilità e logit non finiti contati per frame."""
    tf = BR.tf_module()
    tf.keras.backend.clear_session()
    name, order = rec["key"], rec["order"]
    orig = BR.load_model(rec["weights"], rec["weights_sha256"])
    C.check_loaded(orig, ctx.cfg_c5, rec["network"])

    def fwd(model, X, what):
        f = BR.forward(model, X, ctx.batch, scores=True)
        p, z = np.asarray(f["p"], np.float64), np.asarray(f["z"], np.float64)
        if p.shape != (len(X), 2) or z.shape != (len(X), 2):
            fail(f"{what}: probabilities or logits of the wrong shape")
        bad = ~(np.all(np.isfinite(p), axis=1) & np.all(np.isfinite(z), axis=1))
        ok = ~bad
        if np.any(p[ok] < 0) or np.any(p[ok] > 1):
            fail(f"{what}: finite probabilities outside [0, 1]")
        gap = float(np.max(np.abs(SR.softmax(z[ok]) - p[ok]))) if ok.any() else 0.0
        if gap > ctx.sg:
            fail(f"{what}: the softmax of the logits differs from the probabilities by {gap:.2e}")
        return {"p": p[:, 1].copy(), "z": z, "softmax_gap": gap, "non_finite": int(bad.sum())}
    out = {s: {} for s in D.BN_STATES}
    for role, part in parts.items():
        out["ordinary"][role] = fwd(orig, part["X"], f"{name} {role} ordinary")
    snap = BR.snapshot_weights(orig)
    copy = tf.keras.models.clone_model(orig)
    copy.set_weights(orig.get_weights())
    if BR.snapshot_weights(copy) != snap:
        fail(f"{name}: the in-memory copy differs from the checkpoint before the statistics are installed")
    stats = BT.install(copy, rec["installed"])
    if probs := BR.tensor_problems(snap, BR.snapshot_weights(copy), BR.moving_keys(copy, order), stats):
        fail(f"{name}: re-estimated copy: " + "; ".join(probs))
    for role, part in parts.items():
        out["re-estimated"][role] = fwd(copy, part["X"], f"{name} {role} re-estimated")
    if BR.snapshot_weights(orig) != snap:
        fail(f"{name}: the checkpoint model changed while the copy was scored")
    del orig, copy
    gc.collect()
    return out


# ============================================================================ condizioni e analisi

def build_conditions(plan: list, recs: list, meta: dict, c2: dict) -> dict:
    """Le unità delle 8 condizioni per DEF: per ogni unità (NUAA, fold, seed) LivenessNet e ResNet-18 dai 48 file di C2
    (soglia 0,5 o la soglia EER del dev di C2) e ogni AttackNet valutabile dal piano (pad_arch_eval.build_conditions). Le
    reti di un'unità devono essere sugli stessi frame del test, nello stesso ordine: lo controlla DEF.prepare."""
    new = E.build_conditions(plan, recs, meta)
    ref = c2_conditions(c2)
    conds = {}
    for c in D.CONDITIONS:
        by = {tuple(u["unit"]): u for u in ref[c]}
        for u in new.get(c, []):
            key = tuple(u["unit"])
            if key not in by:
                fail(f"condition {c}: unit {key} of the new models has no C2 reference")
            for n, v in u["networks"].items():
                if n in by[key]["networks"]:
                    fail(f"condition {c}: {n} twice in unit {key}")
                by[key]["networks"][n] = v
        conds[c] = [by[k] for k in sorted(by, key=lambda k: (k[1], k[2]))]
    return conds


def reasons_of(runs: dict, patience: int) -> dict:
    """Il motivo per cui un'AttackNet ha meno di 12 coppie valutabili a una patience, dai suoi modelli non valutabili in
    quella patience (il testo «not computed» del guardiano)."""
    out = {}
    for r in runs.values():
        for x in r["not_evaluable"]:
            if x["patience"] == patience:
                out.setdefault(x["network"], []).append(f"{x['run']}/{x['name']}: {x['reason']}")
    return {n: "; ".join(v) for n, v in out.items()}


def descriptive_analysis(conds: dict, runs: dict) -> dict:
    """Copia dichiarata di pad_anet_definitions.descriptive_analysis con una sola sostituzione: i motivi dei modelli non
    valutabili della patience di ciascuna coppia di condizioni (reasons_of(runs, p)) al posto di un solo testo per tutte
    le coppie. Tutto il resto uguale (pair_analysis, rankings, le 8 condizioni obbligatorie)."""
    DEF._check_conditions(conds)
    missing = [c for c in DEF.CONDITIONS if c not in conds]
    if missing:
        fail(f"descriptive analysis: the conditions {missing} are missing (all 8 are required)")
    pairs = {}
    for p, t in ((12, DEF.PRIMARY_THRESHOLD), (4, "dev_eer"), (12, "dev_eer")):
        r = DEF.pair_analysis(conds, p, t, reasons_of(runs, p), readings=False)
        r.pop("_prepared")
        pairs[f"patience_{p}_{t}"] = r
    ranks = {}
    for c in DEF.CONDITIONS:
        units = sorted(conds[c], key=lambda u: (u["unit"][1], u["unit"][2]))
        ranks["/".join(map(str, c))] = DEF.rankings(DEF.prepare(units, DEF.REUSED))
    return {"pairs": pairs, "rankings": ranks, "intervals": DEF.INTERVALS_NOTE, "note": "descriptive: no reading"}


def test_one_class_counts(conds: dict) -> list:
    """PROTOCOL.md §6 (f), sul test, come conteggio descrittivo: per rete × patience × stato della BN × soglia (0,5 e la
    soglia EER del dev del fold), i modelli che decidono una sola classe per tutti i frame del test (DEF.test_one_class
    sulle unità delle 8 condizioni, con la soglia effettiva di ciascuna rete). Non entra in nessuna lettura."""
    cells = {}
    for (p, st, rule) in DEF.CONDITIONS:
        for u in conds[(p, st, rule)]:
            for n, v in u["networks"].items():
                r = DEF.test_one_class(v["y"], v["s"], v["threshold"])
                c = cells.setdefault((n, p, st, rule), {"models": 0, "one_class": 0, "all_bona_fide": 0, "all_attack": 0})
                c["models"] += 1
                if r["one_class"]:
                    c["one_class"] += 1
                    c["all_bona_fide" if r["class"] == "bona fide" else "all_attack"] += 1
    return [{"network": k[0], "patience": k[1], "bn_state": k[2], "threshold": k[3], **v, "text": f"{v['one_class']} of {v['models']}"}
            for k, v in sorted(cells.items(), key=lambda kv: (DEF.NETWORKS.index(kv[0][0]), kv[0][1], D.BN_STATES.index(kv[0][2]), kv[0][3]))]


def c2_opposite_check(primary: dict, pins: Pins) -> dict:
    """PROTOCOL.md §6 (m): il confronto descrittivo di LivenessNet con ResNet-18 dell'analisi primaria è l'opposto esatto
    dei divari e del contrasto di C2 (results.json della valutazione C2): per ACER, APCER e BPCER media negata, estremi
    negati e scambiati, stessi ricampionamenti validi. Restituisce i confronti; chi chiama fallisce se uno è falso."""
    want = pins.c2_results["primary_analysis"]
    got = primary["descriptive_vs_resnet18"][DEF.REFERENCE]
    out = {}
    for mine, theirs, mkey in (("gap_ordinary", "gap_ordinary", "mean_delta"), ("gap_re_estimated", "gap_re_estimated", "mean_delta"),
                               ("variation_difference", "contrast", "mean")):
        a, b = got[mine], want[theirs]
        ok = True
        for m in DEF.METRICS:
            ia, ib = a["interval"][m], b["interval"][m]
            ok &= (a["mean"][m] == -b[mkey][m] and ia["low"] == -ib["high"] and ia["high"] == -ib["low"]
                   and ia["valid_resamples"] == ib["valid_resamples"])
        out[mine] = bool(ok)
    return out


def analyse(ctx, plan: list, recs: list, meta: dict, c2: dict, runs: dict) -> dict:
    conds = build_conditions(plan, recs, meta, c2)
    primary = DEF.primary_analysis({c: conds[c] for c in E.PRIMARY}, reasons_of(runs, DEF.PRIMARY_PATIENCE))
    descriptive = descriptive_analysis(conds, runs)
    return {"conds": conds, "primary": primary, "descriptive": descriptive, "test_one_class": test_one_class_counts(conds)}


def table_rows(conds: dict) -> list:
    """La tabella delle cinque reti nella condizione primaria (patience 4, soglia 0,5), nei due stati della BN: media sulle
    unità delle frazioni per unità di APCER, BPCER e ACER (DEF.exact_mean, scelta dell'autore: la media che entra nelle
    variazioni), con unità e denominatori (DEF.denominators); una rete assente in qualche unità (AttackNet con modelli
    non valutabili) ha le medie sulle sue unità e il numero delle unità."""
    rows = []
    for c in E.PRIMARY:
        pus = DEF.prepare(sorted(conds[c], key=lambda u: (u["unit"][1], u["unit"][2])), DEF.REUSED)
        for n in DEF.NETWORKS:
            mine = [pu for pu in pus if n in pu["counts"]]
            vals = {m: [DEF._rates(pu, n)[m] for pu in mine] for m in DEF.METRICS}
            mean = {m: (None if not vals[m] or any(v is None for v in vals[m]) else DEF.exact_mean(vals[m])) for m in DEF.METRICS}
            den = DEF.denominators(mine) if mine else None             # le unità della rete (F2)
            rows.append({"network": n, "bn_state": c[1], "units": len(mine), "apcer_mean": mean["apcer"], "bpcer_mean": mean["bpcer"],
                         "acer_mean": mean["acer"], "n_attack": den and den["attack_frames_summed_over_units"],
                         "n_bona_fide": den and den["bona_fide_frames_summed_over_units"], "n_subjects": den and den["subjects"],
                         "computed": len(mine) == len(pus)})
    return rows


# ============================================================================ cartella, tentativi, comando

def repo_state(root: Path) -> dict:
    return REPO_STATE(root) if REPO_STATE is not None else SR.repo_state(root)


def previous_attempts(out_root: Path) -> list:
    out = []
    if not out_root.is_dir():
        return out
    for e in sorted(os.listdir(out_root)):
        if not LOOKS_LIKE_EVAL.match(e):
            continue
        m = EVAL_ID_RE.match(e)
        if not m:
            fail(f"output root: the entry {e!r} looks like an evaluation run id without its form: rename or move it")
        try:
            doc = json.loads((out_root / e / STATUS).read_text(encoding="utf-8"))
            st, read = doc.get("status", "unreadable"), doc.get("test_read_started")
        except (OSError, ValueError):
            st, read = "unreadable", None
        # dalla parte sicura conta come lettura del test anche un tentativo illeggibile, uno rimasto RUNNING (un arresto
        # brusco: batteria, chiusura forzata, memoria) e uno con restricted/ non vuota
        rdir = out_root / e / RESTRICTED
        restricted = rdir.is_dir() and any(rdir.iterdir())
        out.append({"run_id": e, "attempt": int(m.group(1)) if m.group(1) else 1, "status": st,
                    "test_read_started": read is not False or st == "RUNNING" or restricted})
    return out


def attempt_rule(out_root: Path, attempt, second_reading: bool = False) -> dict:
    """La regola dei tentativi (come C2): --attempt obbligatorio, l'ultimo tentativo + 1. Un nuovo tentativo dopo un
    tentativo che ha cominciato a leggere il test (test_read_started vero, oppure stato illeggibile, rimasto RUNNING o
    con restricted/ non vuota: previous_attempts) è una seconda lettura del
    test: rifiutato salvo --second-reading, da passare solo dopo che l'esito del tentativo precedente è stato riportato
    sulla scheda C5, e registrato in status.json e nei risultati; --second-reading senza una lettura precedente è
    rifiutato."""
    if attempt is None:
        fail("--attempt N is required (1 for the first attempt)")
    prev = previous_attempts(out_root)
    need = max((p["attempt"] for p in prev), default=0) + 1
    if attempt != need:
        fail(f"--attempt {attempt} refused: earlier attempts {', '.join(p['run_id'] + ' ' + p['status'] for p in prev) or 'none'}; "
             f"this attempt must be --attempt {need}")
    read = [p["run_id"] for p in prev if p["test_read_started"]]
    if read and not second_reading:
        fail(f"--attempt {attempt} refused: the earlier attempt {', '.join(read)} started reading the test; a new attempt is a second reading "
             "of the test: pass --second-reading only after the outcome has been reported on the C5 card")
    if second_reading and not read:
        fail("--second-reading refused: no earlier attempt started reading the test")
    return {"attempt": attempt, "previous": prev, "second_reading": bool(second_reading), "earlier_readings": read}


def command_text(a) -> str:
    parts = [a.cmd, "--config", R.path_text(NR.folder_arg(a.config))]
    if a.input_root is not None:
        parts += ["--input-root", R.path_text(Path(a.input_root).resolve())]
    parts += ["--out-root", R.path_text(NR.folder_arg(a.out_root)), "--runs-root", R.path_text(NR.folder_arg(a.runs_root))]
    if a.unofficial_out_root:
        parts.append("--unofficial-out-root")
    if a.attempt is not None:
        parts += ["--attempt", str(a.attempt)]
    if a.second_reading:
        parts.append("--second-reading")
    if a.i_am_authorized:
        parts.append("--i-am-authorized")
    return "python -I -B scripts/pad_anet_eval.py " + " ".join(shlex.quote(x) for x in parts)


def official_root(a) -> bool:
    out_root, official = NR.folder_arg(a.out_root), ROOT / DEFAULT_OUT_ROOT
    if pg.within(out_root, official) and pg.within(official, out_root):
        if a.unofficial_out_root:
            fail(f"--unofficial-out-root with the official output root {DEFAULT_OUT_ROOT}: refused")
        return True
    if pg.within(out_root, ROOT) or pg.within(ROOT, out_root):
        fail(f"an output root other than {DEFAULT_OUT_ROOT} inside the repository (or containing it) is refused")
    if not a.unofficial_out_root:
        fail(f"--out-root other than {DEFAULT_OUT_ROOT} needs --unofficial-out-root: the evaluation is then recorded as not official")
    return False


def run_dir_problem(run: Path, out_root: Path, inputs: dict) -> str:
    if os.path.lexists(run):
        return f"the evaluation folder {run.name} already exists: every attempt keeps its outcome"
    if out_root.is_symlink():
        return "the output root is a symbolic link"
    for label, p in inputs.items():
        if pg.within(p, run) or (Path(p).is_dir() and pg.within(run, p)):
            return f"{label} {Path(p).name} is inside the evaluation folder, or contains it"
    return mw.output_problem(run, ROOT, pg.BASE_PROTECTED, [ROOT / DEFAULT_OUT_ROOT], [ROOT / x for x in EXTRA_PROTECTED], inputs,
                             [STATUS, LOG], [run / PUBLIC, run / RESTRICTED])


def final_checks(ctx, cfg_p: Path, cfg_sha: str, runs: dict) -> dict:
    """Prima delle liste del test e prima dei risultati: questa configurazione, il codice, i file fissati dei run
    (pad_anet_run.final_checks) e ogni cartella dei run."""
    if sha256_bytes(read_regular(cfg_p, "configuration")) != cfg_sha:
        fail("the configuration changed during the evaluation")
    if check_code(ctx.cfg) != ctx.pins.code:
        fail("the code changed during the evaluation")
    NR.final_checks(ctx.pins.rcfg, ctx.pins.rcfg_p, ctx.pins.rcfg_sha, ctx.base, ctx.pins.rpins)
    for k in ("protocol", "comparison"):
        R.read_pinned(ctx.base, ctx.cfg[k], k)
    for r in runs.values():
        recheck_run(r)
    return {"configuration_unchanged": True, "code_unchanged": len(ctx.pins.code), "runs_unchanged": len(runs)}


def prepare(a) -> tuple:
    base, cfg_p = NR.resolve(a)
    cfg, cfg_sha = load_config(cfg_p)
    official = official_root(a)
    hooks = check_hooks(official or is_official_config(cfg_p))
    command = command_text(a)
    if probs := PAC.output_problems(command, FR.Inputs()):
        fail("the recorded command holds text that a publishable output must not hold: " + "; ".join(probs))
    pins = load_pins(cfg, base)
    env, mrule = NR.machine_rules(pins.rpins)
    runs = load_runs(cfg, pins, "official" if official else "unofficial", NR.folder_arg(a.runs_root))
    state = repo_state(ROOT)
    if state["dirty"]:
        fail("the git work tree is not clean: the evaluation needs a committed state")
    out_root = NR.folder_arg(a.out_root)
    attempt = attempt_rule(out_root, a.attempt, a.second_reading)
    rid = f"{time.strftime('%Y%m%d')}-ANETEVAL-{state['commit']}" + ("" if a.attempt == 1 else f"-r{a.attempt}")
    if not EVAL_ID_RE.match(rid):
        fail(f"run id {rid} malformed")
    run = out_root / rid
    inputs = {"the configuration": cfg_p, "the runs root": NR.folder_arg(a.runs_root),
              "the C2 restricted folder": rel_path(base, cfg["c2_evaluation"]["restricted_folder"])}
    if msg := run_dir_problem(run, out_root, inputs):
        fail(f"refusing the evaluation folder: {msg}")
    return base, cfg_p, cfg, cfg_sha, pins, env, mrule, official, runs, state, rid, run, attempt, command, hooks


# ============================================================================ uscite

def report_md(doc: dict) -> str:
    pa = doc["primary_analysis"]
    L = [f"# Evaluation {doc['run_id']} of the C5 comparison on the NUAA test", ""]
    if not doc["official"]:
        L += [f"**Not an official evaluation**: written outside {DEFAULT_OUT_ROOT} with --unofficial-out-root.", ""]
    L += [f"Status: **{doc['status']}**. Primary analysis: patience 4, threshold 0.5; {pa['intervals']}", "",
          "## Contrasts (variation of the AttackNet minus that of LivenessNet, ACER)", "",
          "| AttackNet | computed | mean | 95% interval | valid resamples | reading |", "|---|---|---|---|---|---|"]
    for k, c in pa["contrasts"].items():
        if not c.get("computed"):
            L.append(f"| {k} | no | | | | {c.get('text')} |")
            continue
        iv = c["interval"]["acer"]
        L.append(f"| {k} | yes | {DEF.pp_text(c['mean']['acer'])} | [{DEF.pp_text(iv['low'])}, {DEF.pp_text(iv['high'])}] | "
                 f"{iv['valid_resamples']} | {c.get('reading')} |")
    L += ["", "## Variations of ACER (re-estimated minus ordinary BN)", "", "| network | computed | mean | 95% interval |", "|---|---|---|---|"]
    for n, v in pa["variations"].items():
        if not v.get("computed"):
            L.append(f"| {n} | no | | {v.get('text')} |")
            continue
        iv = v["interval"]["acer"]
        L.append(f"| {n} | yes | {DEF.pp_text(v['mean']['acer'])} | [{DEF.pp_text(iv['low'])}, {DEF.pp_text(iv['high'])}] |")
    L += ["", f"C2 reproduced exactly before the test of the new models: {doc['c2_reproduction']['exact']}. Not evaluable models: "
          f"{len(doc['not_evaluable'])}.", "", f"Command: `{doc['command']}`", ""]
    return "\n".join(L)


def write_public(run: Path, inp, texts: dict) -> str:
    return R.write_public(run, inp, texts)


# ============================================================================ comandi

def cmd_check_inputs(a) -> int:
    base, cfg_p = NR.resolve(a)
    cfg, cfg_sha = load_config(cfg_p)
    hooks = check_hooks(is_official_config(cfg_p))
    pins = load_pins(cfg, base)
    env, mrule = NR.machine_rules(pins.rpins)
    c2 = load_c2_scores(cfg, pins, base)
    rep = reproduce_c2(pins, c2)
    c2_thresholds_check(pins, c2)
    c2_opposite_before_test(pins, c2)
    runs_text = "runs pending (recorded in the configuration after the six runs)"
    if cfg["runs"] != RUNS_PENDING:
        runs = load_runs(cfg, pins, "inputs", NR.folder_arg(a.runs_root))
        ne = sum(len(r["not_evaluable"]) for r in runs.values())
        copies = check_run_copies(cfg, base, NR.folder_arg(a.runs_root))
        runs_text = (f"six runs PASS, last attempts, {sum(len(r['models']) for r in runs.values())} evaluable models, {ne} not evaluable; "
                     f"the records of {copies['folders']} commands in the package equal to the runs root ({copies['files_compared']} files)")
    print(f"CHECK-INPUTS: configuration {cfg_sha[:12]}; configuration of the runs {pins.rcfg_sha[:12]}; PR A configuration "
          f"{pins.rpins.sha_c5[:12]}; code pinned ({len(CODE_FILES)} files); hooks "
          f"{'none set' if not hooks['set'] and not hooks['changed'] else ', '.join(hooks['set'] + hooks['changed'])}; machine and versions "
          f"equal to the C5 trial; 48 C2 test score files equal to the anchored restricted SHA256SUMS; C2 reproduced exactly (primary "
          f"analysis, {rep['conditions']} conditions, units.csv; the 48 EER thresholds equal to results.json; the opposite of C2 on the C2 units); {runs_text}; no frame assembled; TensorFlow {FR.tf_state()}; nothing written")
    print("RESULT: PASS (check-inputs)")
    return 0


def cmd_evaluate(a) -> int:
    if not a.i_am_authorized:
        fail("evaluate reads the NUAA test of the new models: pass --i-am-authorized (only after this code is published and linked on the C5 card)")
    base, cfg_p, cfg, cfg_sha, pins, env, mrule, official, runs, state, rid, run, attempt, command, hooks = prepare(a)
    run.mkdir(parents=True)
    scan = FR.Inputs()
    head = {"run_id": rid, "commit": state["commit"], "dirty": False, "config_sha256": cfg_sha, "command": "evaluate", "command_line": command,
            "official": official, "attempt": attempt, "second_reading": attempt["second_reading"], "hooks": hooks,
            "runs": {k: {"run_id": r["run_id"], "official": r["official"], "status_sha256": r["status_sha256"]} for k, r in runs.items()}}
    cap_h = cfg["cap"]["hours"]
    budget = SR.Budget(cap_h * 60)
    SR.signal_handlers(budget)
    started, c0 = now(), PAC.cpu_seconds()
    log, saved_warning = None, warnings.showwarning
    threads = {"requested": dict(pins.trial["threads"]["requested"]), "effective": None, "set_by_this_runner": False, "environment": env,
               "logical_cpus": os.cpu_count()}
    extra = {"threads": threads, "machine_rule": mrule, "cap_hours": cap_h, "cap_unit": CAP_UNIT, "test_read_started": False}
    rdir = run / RESTRICTED

    def running(scan) -> None:
        R.write_status(run, {**head, "status": "RUNNING", "passed": False, "started": started, "pid": os.getpid(), **extra,
                             "note": "RUNNING without a live process means a failed evaluation; SIGUSR1 stops it as INCOMPLETE"}, scan)
    try:
        running(scan)
        log = FR.open_log(run)
        warnings.showwarning = PAC.show_warning
        print(f"== evaluation {rid} (attempt {attempt['attempt']}{'' if official else ', NOT OFFICIAL'}): configuration {cfg_sha[:12]}, "
              f"cap {cap_h:.2f} {CAP_UNIT}, pid {os.getpid()}", flush=True)
        ctx = Ctx()
        ctx.cfg, ctx.pins, ctx.base, ctx.cfg_pr, ctx.cfg_c5, ctx.pcfg, ctx.bn_cfg, ctx.budget = (cfg, pins, base, pins.cfg_pr, pins.cfg_c5, pins.pcfg,
                                                                                                pins.bn_cfg, budget)
        ctx.folds, ctx.seeds, ctx.test_per_fold = pins.folds, pins.seeds, pins.test_per_fold
        ctx.batch, ctx.sg = pins.rcfg["estimator"]["batch"], BR.tol(pins.bn_cfg, "softmax_of_logits_max_abs_diff")
        inp = PAC.load_inputs(pins.cfg_pr, pins.pcfg, base)
        scan = ctx.inp = inp
        # le copie dei record dei run nel pacchetto, come in check-inputs, con la scansione sui manifest caricati
        extra["run_records"] = check_run_copies(cfg, base, NR.folder_arg(a.runs_root), inp)
        recs = [m for r in NR.RUNS for m in runs[r]["models"]]
        not_eval = [x for r in NR.RUNS for x in runs[r]["not_evaluable"]]
        for rec in recs:
            E.load_dev(rec, inp)
        plan = E.evaluation_plan(recs)
        c2 = load_c2_scores(cfg, pins, base)
        c2_rep = reproduce_c2(pins, c2)
        c2_rep["thresholds_checked"] = c2_thresholds_check(pins, c2)
        c2_rep["opposite_before_test"] = c2_opposite_before_test(pins, c2)
        print(f"   C2 reproduced exactly from the 48 reused files ({c2_rep['conditions']} conditions, units.csv): before any test frame of the "
              "new models", flush=True)
        ctx.store = FR.FrameStore(inp, inp.X_extracted, inp.extracted_keys)
        tf = R.tf_module()
        threads = PAC.set_threads(tf, pins.cfg_pr)
        extra["threads"] = threads
        extra["threads_rule"] = R.threads_rule(pins.trial, threads)
        R.require_threads(extra["threads_rule"])
        print(f"   {PAC.threads_text(threads)}", flush=True)
        FR.hide_gpu(tf)
        ctx.tf = tf
        status, stopped, analysis, meta, repro, non_finite = "PASS", None, None, None, [], []
        try:
            repro = reproduce_dev(ctx, recs)
            if any(not r["pass"] for r in repro):
                status = "MISMATCH"
            else:
                PAC.check_threads(tf, pins.cfg_pr, "after the dev reproduction")
                budget.check("re-verification before the test")
                extra["checks_before_test"] = final_checks(ctx, cfg_p, cfg_sha, runs)
                extra["test_read_started"] = True
                # su disco prima di qualsiasi lettura del test (schema dei fold compreso): un arresto brusco da qui in poi
                # lascia un tentativo RUNNING con test_read_started vero, cioè una lettura (previous_attempts)
                running(inp)
                rdir.mkdir()
                ctx.rdir = rdir
                lists = E.load_test_lists(ctx)
                c2_ids_check(c2, lists)
                meta, non_finite = score_test(ctx, recs, lists)
                PAC.check_threads(tf, pins.cfg_pr, "after the test scoring")
                if non_finite:
                    status = "NON_FINITE_TEST"
                budget.check("analysis")
        except SR.CapExceeded as e:
            status = "INCOMPLETE"
            stopped = {"step": e.step, "cause": e.cause, "elapsed_minutes": round(e.minutes, 1)}
            print(f"== {'CAP' if e.cause == 'cap' else 'STOP'}: {e}; no result is computed", flush=True)
        extra["dev_reproduction"] = [{k: v for k, v in r.items() if k != "key"} for r in repro]
        extra["non_finite_test"] = non_finite
        pub_sha = res_sha = None
        if extra["test_read_started"]:
            res_sha = R.write_restricted_sums(rdir)           # PASS, NON_FINITE_TEST e INCOMPLETE dopo la lettura: i punteggi restano
        if status == "PASS":
            checks = final_checks(ctx, cfg_p, cfg_sha, runs)
            E.scan_log(run, log, inp)
            analysis = analyse(ctx, plan, recs, meta, c2, runs)
            pa = analysis["primary"]
            opposite = c2_opposite_check(pa, pins)
            if not all(opposite.values()):
                fail(f"the descriptive comparison of LivenessNet with ResNet-18 is not the exact opposite of C2: {opposite}")
            pj = E.jsonable({k: v for k, v in pa.items()})
            dj = E.jsonable(analysis["descriptive"])
            wall, cpu_total, finished = budget.elapsed(), round(PAC.cpu_seconds() - c0, 1), now()
            doc = {"schema": EVAL_SCHEMA, **head, "status": status, "started": started, "finished": finished, "wall_seconds": round(wall, 1),
                   "process_cpu_seconds": cpu_total, "primary_analysis": pj, "descriptive": dj, "c2_reproduction": c2_rep,
                   "c2_opposite": opposite, "not_evaluable": not_eval, "test_one_class": E.jsonable(analysis["test_one_class"]),
                   "dev_reproduction": {"checked": len(repro), "passed": sum(r["pass"] for r in repro)},
                   "models": [{"run": r["run"], "network": r["network"], "fold": r["fold"], "seed": r["seed"], "weights_sha256": r["weights_sha256"],
                               "dev_eer_threshold": {s: r["thresholds"][(s, "dev_eer")] for s in D.BN_STATES}} for r in recs],
                   "test_frames_per_fold": [len(meta[f]["ids"]) for f in ctx.folds],
                   "test_frame_ids_sha256": {str(f): D.ids_sha256(meta[f]["ids"]) for f in ctx.folds},
                   "machine": (R.MACHINE or PAC.machine)(), "environment": (R.ENVIRONMENT or FR.environment)(), "machine_rule": mrule,
                   "threads": threads, "threads_rule": extra["threads_rule"], "checks": checks, "code_sha256": pins.code,
                   "inputs_sha256": {**pins.files, "configuration": cfg_sha}, "test_read": TEST_READ, "command": command}
            texts = {"results.json": PAC.json_text(doc, "results.json"), "report.md": report_md(doc),
                     "table.csv": csv_text(TABLE_COLUMNS, [[E.cell(x[k]) for k in TABLE_COLUMNS] for x in table_rows(analysis["conds"])]),
                     "conditions.csv": csv_text(NEW_CONDITION_COLUMNS, condition_rows(pa, analysis["descriptive"])),
                     "units.csv": csv_text(NEW_UNIT_COLUMNS, unit_rows(analysis["conds"])),
                     "reproduction.csv": csv_text(E.REPRODUCTION_COLUMNS, [[E.cell(r[k]) for k in E.REPRODUCTION_COLUMNS] for r in repro])}
            pub_sha = write_public(run, inp, texts)
        else:
            wall, cpu_total, finished = budget.elapsed(), round(PAC.cpu_seconds() - c0, 1), now()
        print(f"evaluation folder: {run.name}, wall {wall:.0f} s, process CPU {cpu_total:.0f} s")
        sys.stdout.flush()
        log[0].flush()
        if probs := PAC.output_problems((run / LOG).read_text(encoding="utf-8", errors="replace"), inp):
            fail(f"{LOG} holds text that a publishable output must not hold: " + "; ".join(probs))
        print(f"RESULT: {status} (evaluate)")
        warnings.showwarning = saved_warning
        FR.close_log(log)
        log = None
        R.write_status(run, {**head, "status": status, "passed": status == "PASS", "started": started, "finished": finished,
                             "wall_seconds": round(wall, 1), "process_cpu_seconds": cpu_total, **extra, "stopped": stopped,
                             "not_evaluable": not_eval, "c2_reproduction": c2_rep, "public_sha256sums_sha256": pub_sha,
                             "restricted_sha256sums_sha256": res_sha, "run_log_sha256": sha256_file(run / LOG)}, scan)
    except BaseException as e:
        warnings.showwarning = saved_warning
        if (run / RESTRICTED).is_dir():
            if extra.get("test_read_started"):
                # dopo la lettura del test i punteggi riservati già scritti restano, con il loro SHA256SUMS (B1: un nuovo
                # tentativo sarebbe una seconda lettura)
                extra["restricted_sha256sums_sha256"] = R.write_restricted_sums(run / RESTRICTED)
            else:
                mw.invalidate(run / RESTRICTED, sorted(os.listdir(run / RESTRICTED)))
        R.fail_cleanup(run, list(PUBLIC_OUTPUTS), log, head, started, budget, c0, extra, e, scan)
        raise
    return STATUS_CODE[status]


def condition_rows(pa: dict, desc: dict) -> list:
    """conditions.csv: per condizione (primaria e le tre coppie descrittive) variazioni, contrasti e confronto descrittivo
    con ResNet-18, con media e intervallo di ACER, APCER e BPCER."""
    out = []

    def add(label, kind, n, q):
        if not q.get("computed", True) or "mean" not in q:
            out.append([label, kind, n, "0"] + ["NA"] * (len(NEW_CONDITION_COLUMNS) - 4))
            return
        iv = q["interval"]
        out.append([label, kind, n, "1", E.cell(q["mean"]["acer"]), E.cell(iv["acer"]["low"]), E.cell(iv["acer"]["high"]),
                    E.cell(iv["acer"]["valid_resamples"]), E.cell(q["mean"]["apcer"]), E.cell(iv["apcer"]["low"]), E.cell(iv["apcer"]["high"]),
                    E.cell(q["mean"]["bpcer"]), E.cell(iv["bpcer"]["low"]), E.cell(iv["bpcer"]["high"])])
    for label, res in [("patience_4_fixed_0.5 (primary)", pa)] + list(desc["pairs"].items()):
        for n, v in res["variations"].items():
            add(label, "variation", n, v)
        for k, v in res["contrasts"].items():
            add(label, "contrast", k, v)
    return out


def unit_rows(conds: dict) -> list:
    """units.csv: per condizione, unità e rete presente, soglia effettiva e metriche (frazioni esatte), con il numero dei
    frame e l'impronta dei loro ID; nessun ID."""
    rows = []
    for c in D.CONDITIONS:
        pus = DEF.prepare(sorted(conds[c], key=lambda u: (u["unit"][1], u["unit"][2])), DEF.REUSED)
        for pu in pus:
            for n in pu["networks"]:
                m = DEF._rates(pu, n)
                rows.append([str(c[0]), c[1], c[2], str(pu["unit"][1]), str(pu["unit"][2]), str(pu["n_frames"]), pu["frame_ids_sha256"], n,
                             E.cell(pu["thresholds"][n]), E.cell(m.get("apcer")), E.cell(m.get("bpcer")), E.cell(m.get("acer"))])
    return rows


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="The evaluation of the C5 comparison on the NUAA test (the new AttackNet models and the 48 C2 references)")
    ap.add_argument("cmd", choices=["check-inputs", "evaluate"])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--input-root", default=None)
    ap.add_argument("--runs-root", default=RUNS_ROOT, help=f"folder of the runs (official: {RUNS_ROOT})")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--unofficial-out-root", action="store_true")
    ap.add_argument("--attempt", type=int, default=None)
    ap.add_argument("--second-reading", action="store_true", help="a new attempt after an attempt that started reading the test: only after "
                                                                  "the outcome has been reported on the C5 card (recorded)")
    ap.add_argument("--i-am-authorized", action="store_true")
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
