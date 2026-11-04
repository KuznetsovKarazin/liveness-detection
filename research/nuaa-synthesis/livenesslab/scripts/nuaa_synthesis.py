"""
Sintesi NUAA (C1, M1, N1, N2, confronto del ritaglio, run a fold di soggetti): ricostruisce la tabella unica dei risultati accettati leggendo
soltanto gli aggregati pubblici già uniti in research/pad-2026 (nessun dato riservato, nessuna inferenza, nessun numero
nuovo). Ogni cella della tabella viene da un file accettato, citato riga per riga con il suo SHA-256 al momento della
costruzione; i testi descrittivi (conteggi dei manifest, soggetti) vengono dall'aggregato pubblico dei manifest M2.

Sorgenti (percorsi relativi alla cartella research/ del repository del gruppo):
  c1-livenesslab/tables/nuaa_table.md                      tabella C1 (soglia 0,5): righe delle quattro CNN NUAA, delle
                                                           16 CNN addestrate su altri dataset e degli altri analizzatori
  m1-metrics/tables/nuaa_test_dev_threshold_table.md       tabella M1: le quattro CNN NUAA a 0,5 e alla soglia dev, con
                                                           conteggi, soglie e AUC del test
  m2-nuaa-manifests/nuaa_manifests_aggregate.json          M2: conteggi e soggetti dei manifest NUAA
  n1-nuaa-stability/tables/aggregate.json                  N1 parte 2: LivenessNet, tre seed, test 3.000 e C1 300
  n1-nuaa-stability/tables/subject_sensitivity/n1_subject_sensitivity_public.json
                                                           N1 parte 1: sensibilità ai soggetti del risultato M1 (con i
                                                           conteggi degli errori a 0,5 e alla soglia M1)
  n2-nuaa-check/tables/sessions/aggregate.json             N2 parte 1: i checkpoint N1 sulle sessioni 01-02 e 03
  n2-nuaa-check/tables/combined/aggregate.json             N2 parte 2: il modello combinato storico sul test N1
  c1-crop-check/tables/aggregate.json                      confronto del ritaglio su AttackNet V2.1 e le 300 di C1
  nuaa-subject-folds/tables/fold_schema_public.json        schema dei fold approvato (PR #16): conteggi per fold,
                                                           ruolo e cella classe x sessione
  nuaa-folds-run/tables/aggregate.json                     run a fold (PR #18): LivenessNet, 4 fold x 3 seed, strati
                                                           01/02/03/all, soglie 0,5 e dev, aggregato per seed, bootstrap
  nuaa-folds-run/tables/run_record.json                    run a fold: record del run (run id, commit, configurazione,
                                                           controlli degli input, tempi)
  nuaa-folds-n1-compare/tables/aggregate.json              confronto descrittivo con N1 sugli stessi 3.000 frame
                                                           (run ufficiale: albero pulito, run id senza -dev)

Sottocomandi:
  build --research <research/> --out <cartella>   scrive summary_table.{md,csv,json}, summary_table_compact.md,
                                                  cross_dataset_table.md, context_rows_table.md,
                                                  subject_sensitivity_table.md, folds_run_table.md,
                                                  n1_same_ids_table.md (UTF-8, fine riga LF)
  check --research <research/> --out <cartella>   ricostruisce in memoria e confronta byte per byte con i file di --out;
                                                  esce con 1 se un file manca o differisce; con --synthesis <SYNTHESIS.md>
                                                  esige anche che ogni riga della tabella compatta stia nel documento
Entrambi eseguono i controlli di coerenza fra le sorgenti e si fermano con errore se uno non regge: ogni aggregato con
passed vero e albero pulito (il file dei soggetti N1, che non ha passed: albero pulito e tabella M1 riprodotta); il
suffisso di ogni run id uguale al commit registrato; nessuna riga duplicata nelle tabelle Markdown; in ogni riga,
tassi uguali ai conteggi a quattro decimali (righe M1 e righe lette dai JSON); le righe C1 della tabella M1 uguali
alla tabella C1 (tassi, AUC, denominatori); i conteggi M1 uguali a quelli del file dei soggetti N1 (a 0,5 e alla soglia
M1) e l'ACER alla soglia M1 uguale; nessuna riga M1 con soglia dichiarata non informativa; denominatori delle righe
uguali ai conteggi dei manifest M2 (c1, cnn_test) e dell'obiettivo N2 (491), e quest'ultimo uguale alle esclusioni
registrate da M2; soggetti di fitting, early stopping e training uguali fra M2 e i testi; i seed di N2 uguali a quelli
di N1, con gli stessi pesi, e la sessione 03 di N2 uguale al test N1; sottoinsiemi N2 con gli stessi denominatori per
ogni seed e somme coerenti (fitting + validazione = 491; sessione 03 fitting + validazione = sessione 03 dei 9); il
riferimento N1 del combinato uguale a N1; il riferimento C1 del confronto del ritaglio uguale alla riga C1 (conteggi,
tassi, run) e la riga dell'estrattore C1 uguale al riferimento a quattro decimali, con gli stessi denominatori; soglia
0,5 in N1; 16 righe cross-dataset e tutte le righe di contesto nella tabella C1; soglie M1 del file dei soggetti uguali
alla tabella M1. Run a fold: aggregato, record e schema da albero pulito, record e aggregato dello stesso run
(run id, commit, configurazione, bootstrap) e della configurazione consegnata; schema del record uguale allo schema
approvato (run id, SHA-256 della configurazione, totali di test per fold); denominatori di ogni fold e strato uguali ai
conteggi dello schema; somma dei quattro fold di test uguale alle liste ufficiali M2 (12.614 frame, 16 soggetti), ai
file originali verificati dal run e a cache + frame estratti; strato «all» uguale alla somma delle sessioni; aggregato
per seed uguale alla somma dei quattro fold; mediane e intervalli sui seed uguali ai valori per seed; bootstrap con i
parametri dichiarati; 12 modelli con 2.100 + 600 frame, epoca ripristinata valida e tempo nel record; tempo totale entro
il tetto. Confronto con N1: run ufficiale della configurazione consegnata; ingressi uguali a M2, al run a fold, a N1
e allo schema; valori N1 uguali alle righe N1 della tabella; conteggi dei fold entro la sessione 03 aggregata del run a
fold; differenze, decisioni appaiate, conteggi per soggetto, mediane, bootstrap e valori di controllo coerenti. Il
controllo copre le celle delle tabelle; i numeri della prosa del documento restano da verificare a
mano contro le tabelle e le fonti.

Dipendenze: solo la libreria standard.
"""
import argparse
import csv
import hashlib
import io
import json
import re
import statistics
import sys
from pathlib import Path

SCHEMA = "nuaa-synthesis/1"

SOURCES = {
    "c1_table": "c1-livenesslab/tables/nuaa_table.md",
    "m1_table": "m1-metrics/tables/nuaa_test_dev_threshold_table.md",
    "m2_manifests": "m2-nuaa-manifests/nuaa_manifests_aggregate.json",
    "n1_aggregate": "n1-nuaa-stability/tables/aggregate.json",
    "n1_subjects": "n1-nuaa-stability/tables/subject_sensitivity/n1_subject_sensitivity_public.json",
    "n2_sessions": "n2-nuaa-check/tables/sessions/aggregate.json",
    "n2_combined": "n2-nuaa-check/tables/combined/aggregate.json",
    "crop": "c1-crop-check/tables/aggregate.json",
    "folds_schema": "nuaa-subject-folds/tables/fold_schema_public.json",
    "folds_aggregate": "nuaa-folds-run/tables/aggregate.json",
    "folds_record": "nuaa-folds-run/tables/run_record.json",
    "folds_n1_compare": "nuaa-folds-n1-compare/tables/aggregate.json",
}

NUAA_CNNS = ("livenessnet__nuaa", "attacknet_v1__nuaa", "attacknet_v2_1__nuaa", "attacknet_v2_2__nuaa")
CNN_NAMES = {
    "livenessnet__nuaa": "LivenessNet",
    "attacknet_v1__nuaa": "AttackNet V1",
    "attacknet_v2_1__nuaa": "AttackNet V2.1",
    "attacknet_v2_2__nuaa": "AttackNet V2.2",
}
CROSS_SOURCES = {"casia_fasd": "CASIA-FASD", "celeba_spoof": "CelebA-Spoof", "synthaspoof": "SynthASpoof"}
ARCH_NAMES = {"livenessnet": "LivenessNet", "attacknet_v1": "AttackNet V1", "attacknet_v2_1": "AttackNet V2.1",
              "attacknet_v2_2": "AttackNet V2.2"}
# "trained on" of the context rows: "no NUAA training in our experiment" is what the C1 run establishes; whether
# NUAA appears in the pretraining data of a third-party model or encoder is not verified here (provenance check).
TRAINED_PRETRAINED = "no NUAA training in our experiment; third-party pretraining, NUAA in its data not verified"
TRAINED_PROBE = ("probe: all 3,000 sampled NUAA training images (dev included); encoder: third-party p"
                 "retraining, NUAA in its data not verified")
TRAINED_SVM = "SVM: all 3,000 sampled NUAA training images (dev included); no pre-trained component"
TRAINED_RULE = "no training (hand-crafted rule)"
CONTEXT_ROWS = {
    "minifasnet": ("MiniFASNet (Silent-Face, pre-trained)", TRAINED_PRETRAINED),
    "clip_zeroshot": ("CLIP ViT-B/32 zero-shot", TRAINED_PRETRAINED),
    "clip_probe": ("linear probe on CLIP ViT-B/32", TRAINED_PROBE),
    "dinov2_probe": ("linear probe on DINOv2-small", TRAINED_PROBE),
    "depth": ("Depth Anything V2 (depth statistics)", TRAINED_PRETRAINED),
    "lbp": ("LBP + SVM", TRAINED_SVM),
    "fourier": ("Fourier spectrum rule", TRAINED_RULE),
    "ida": ("IDA features + SVM", TRAINED_SVM),
    "dog": ("DoG features + SVM", TRAINED_SVM),
    "iqa": ("IQA features + SVM", TRAINED_SVM),
}

C1_RUN = "20260928-C1-seed42-a80f6f9"
M1_RUN = "20261003-M1-seed42-e67ecc7"

EXTRACTOR_C1 = "c1_a80f6f9"
EXTRACTOR_CACHE = "cache_0ac7143"
THR_FIXED = "fixed a priori (C1)"
PREP_REF = "detector crop (margin 0.15, square; whole frame without a face), 256x256 INTER_AREA, BGR to RGB, reference enhancement, /255 in [0, 1]"
PREP_HIST = "historical chain of the combined model: BGR to RGB, 256x256 Lanczos, reference enhancement, (x/255 - 0.5) * 2 in [-1, 1]"
TRAIN_COMB = ("3DMAD, CSMAD, Replay-Attack, MSSpoof, Our Dataset (historical code; NUAA not among them; "
              "provenance of the training images not verified here)")
TRANSFER_SESSION_ALL = "session transfer (01-02 to 03; the training subjects and the test-only subjects)"
TRANSFER_SESSION_9 = "session transfer (01-02 to 03; the training subjects only)"
TRANSFER_SAME = "same sessions as training (01-02), fitting and early-stopping subjects, frames never used"
TRANSFER_DATASET = ("dataset transfer (no NUAA training in our experiment: NUAA is not among the training datasets "
                    "named in the historical code; the provenance of its training images is not verified here)")
VARIANTS = {"whole_image": "whole image (no face detection)", "face_crop": "face crop (current detector, margin 0.15, square; whole frame without a face)"}
EXTRACTOR_FOLDS = EXTRACTOR_CACHE + " (cache frames; the other frames extracted once with the frozen copy)"
EXTRACTOR_LABELS = {EXTRACTOR_CACHE: "cache_0ac7143 (the training-cache extractor)",
                    EXTRACTOR_C1: "c1_a80f6f9 (the C1 extractor; box 1 px narrower and shorter)"}

COLUMNS = ["row", "block", "model", "training_data", "early_stopping", "test_set", "unit", "extractor", "preprocessing",
           "threshold", "threshold_origin", "n_bona_fide", "n_attack", "rejected_bona_fide", "accepted_attacks", "apcer",
           "bpcer", "acer", "auroc", "threshold_dev", "rejected_bona_fide_dev", "accepted_attacks_dev", "apcer_dev",
           "bpcer_dev", "acer_dev", "bootstrap", "run_id", "transfer", "source"]
# Colonne riempite soltanto dalle righe del run a fold (soglia dev accanto a 0,5, bootstrap per le righe aggregate);
# vuote nelle altre righe («—» nelle tabelle Markdown).
FOLDS_ONLY_COLUMNS = ("threshold_dev", "rejected_bona_fide_dev", "accepted_attacks_dev", "apcer_dev", "bpcer_dev",
                      "acer_dev", "bootstrap")
METRIC_FIELDS = ("n_bona_fide", "n_attack", "rejected_bona_fide", "accepted_attacks", "apcer", "bpcer", "acer", "auroc")
RATES = ("apcer", "bpcer", "acer", "auroc")


class SynthesisError(Exception):
    pass


def fail(msg: str) -> None:
    raise SynthesisError(msg)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def f4(v) -> str:
    if v is None:
        return "NA"
    return "%.4f" % float(v)


def as_int(v, where: str) -> int:
    try:
        return int(str(v))
    except ValueError:
        fail("%s: %r is not an integer count" % (where, v))


# --- lettura delle sorgenti ---------------------------------------------------------------------------------------

def md_tables(text: str) -> list:
    """Ogni tabella Markdown del testo come (intestazione, righe); una riga con un numero diverso di celle resta grezza."""
    tables, header, rows = [], None, []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s[1:-1].split("|")]
            if header is None:
                header = cells
            elif all(re.fullmatch(r":?-+:?", c) for c in cells):
                continue
            else:
                rows.append(dict(zip(header, cells)) if len(cells) == len(header) else {"_raw": cells, header[0]: cells[0]})
        else:
            if header is not None:
                tables.append((header, rows))
            header, rows = None, []
    if header is not None:
        tables.append((header, rows))
    return tables


def keyed_rows(rows: list, key: str, what: str) -> dict:
    out = {}
    for r in rows:
        if "_raw" in r:
            continue  # righe con meno celle (C1: «not evaluated» / «no score»)
        if r[key] in out:
            fail("%s: duplicate row %s" % (what, r[key]))
        out[r[key]] = r
    return out


def read_c1_table(p: Path) -> dict:
    tables = md_tables(p.read_text(encoding="utf-8"))
    if not tables:
        fail("C1 table: no Markdown table found in %s" % p)
    header, rows = tables[0]
    need = ["analyzer", "n", "n_bona_fide", "n_attack", "apcer", "bpcer", "acer", "eer", "auc"]
    missing = [c for c in need if c not in header]
    if missing:
        fail("C1 table: missing columns %s" % missing)
    return keyed_rows(rows, "analyzer", "C1 table")


M1_COLUMNS = ["analyzer", "n_bona_fide", "n_attack", "APCER@0.5", "BPCER@0.5", "ACER@0.5", "accepted attacks@0.5",
              "rejected bona fide@0.5", "dev threshold", "APCER@dev", "BPCER@dev", "ACER@dev", "accepted attacks@dev",
              "rejected bona fide@dev", "test EER", "test AUC", "operational dev EER", "dev AUC", "dev flag"]


def read_m1_table(p: Path) -> dict:
    tables = md_tables(p.read_text(encoding="utf-8"))
    if not tables:
        fail("M1 table: no Markdown table found in %s" % p)
    header, rows = tables[0]  # la prima tabella è quella principale; la seconda è la sensibilità senza i soggetti dev
    missing = [c for c in M1_COLUMNS if c not in header]
    if missing:
        fail("M1 table: missing columns %s" % missing)
    out = keyed_rows(rows, "analyzer", "M1 table")
    for a in NUAA_CNNS:
        if a not in out:
            fail("M1 table: row %s missing" % a)
    return out


def read_json(p: Path, must: list, clean: bool = True) -> dict:
    with open(p, encoding="utf-8") as fh:
        d = json.load(fh)
    if not isinstance(d, dict):
        fail("%s: not a JSON object" % p.name)
    for k in must:
        if k not in d:
            fail("%s: field %r missing" % (p.name, k))
    if clean:
        if d.get("passed") is not True:
            fail("%s: passed is not true" % p.name)
        for k in ("dirty", "dirty_at_check"):
            if d.get(k) is True:
                fail("%s: %s is true (aggregate from a modified tree)" % (p.name, k))
    return d


def metric_row(m, where: str) -> dict:
    """Le otto metriche di una riga JSON, con i tassi verificati contro i conteggi a quattro decimali (NA se la classe manca)."""
    if not isinstance(m, dict):
        fail("%s: metrics are not an object" % where)
    missing = [k for k in METRIC_FIELDS if k not in m]
    if missing:
        fail("%s: metric fields missing %s" % (where, missing))
    r = {k: m[k] for k in METRIC_FIELDS}
    for k in ("n_bona_fide", "n_attack", "rejected_bona_fide", "accepted_attacks"):
        if not isinstance(r[k], int) or isinstance(r[k], bool) or r[k] < 0:
            fail("%s: %s is not a count" % (where, k))
    apcer = r["accepted_attacks"] / r["n_attack"] if r["n_attack"] else None
    bpcer = r["rejected_bona_fide"] / r["n_bona_fide"] if r["n_bona_fide"] else None
    acer = (apcer + bpcer) / 2 if apcer is not None and bpcer is not None else None
    for k, v in (("apcer", apcer), ("bpcer", bpcer), ("acer", acer)):
        if f4(r[k]) != f4(v):
            fail("%s: %s %s is not the rate of the counts (%s)" % (where, k, f4(r[k]), f4(v)))
    if (r["n_attack"] == 0 or r["n_bona_fide"] == 0) != (r["auroc"] is None):
        fail("%s: auroc must be NA exactly when a class is absent" % where)
    return r


def run_id_check(d: dict, where: str) -> None:
    """Il suffisso del run id (commit breve) deve coincidere con il commit registrato nell'aggregato."""
    suffix = str(d.get("run_id", "")).split("-")[-1].replace("-trial", "")
    if not suffix or not str(d.get("commit", "")).startswith(suffix):
        fail("%s: run id %r does not end with the recorded commit %r" % (where, d.get("run_id"), d.get("commit")))


def seed_block(d: dict, seeds: list, where: str) -> dict:
    out = {}
    for s in seeds:
        if s not in d.get("seeds", {}):
            fail("%s: seed %s of train_seeds is missing in seeds" % (where, s))
        out[s] = d["seeds"][s]
    return out


# --- costruzione delle righe --------------------------------------------------------------------------------------

def row(**kw) -> dict:
    r = {c: "" for c in COLUMNS}
    r["preprocessing"] = PREP_REF
    r["unit"] = "image (frame)"
    unknown = set(kw) - set(COLUMNS)
    if unknown:
        fail("row: unknown columns %s" % sorted(unknown))
    r.update(kw)
    return r


def manifest_counts(m2: dict) -> dict:
    out = {}
    for name in ("official_train", "official_test", "cnn_train", "cnn_es", "cnn_test", "m1_dev", "c1"):
        d = m2["manifests"].get(name)
        if not isinstance(d, dict):
            fail("M2 manifests: %s missing" % name)
        for k in ("n", "n_bona_fide", "n_attack", "n_subjects"):
            if k not in d:
                fail("M2 manifests: %s has no field %s" % (name, k))
        if d["n"] != d["n_bona_fide"] + d["n_attack"]:
            fail("M2 manifests: %s n != n_bona_fide + n_attack" % name)
        out[name] = {k: d[k] for k in ("n", "n_bona_fide", "n_attack", "n_subjects")}
    return out


def fmt(n) -> str:
    return "{:,}".format(int(n))


def texts_of(mc: dict, target: dict) -> dict:
    """Testi descrittivi delle righe, costruiti dai conteggi letti (M2 e obiettivo N2)."""
    return {
        "train": "NUAA official train, sessions 01-02: %s fitting frames (%d subjects)" % (fmt(mc["cnn_train"]["n"]), mc["cnn_train"]["n_subjects"]),
        "es": "%s frames of %d other subjects, sessions 01-02 (early stopping)" % (fmt(mc["cnn_es"]["n"]), mc["cnn_es"]["n_subjects"]),
        "test_c1": "C1 %s (%s + %s), session 03, %d subjects" % (fmt(mc["c1"]["n"]), fmt(mc["c1"]["n_bona_fide"]), fmt(mc["c1"]["n_attack"]), mc["c1"]["n_subjects"]),
        "test_3000": "nuaa_cnn_test %s (%s + %s), session 03, %d subjects" % (fmt(mc["cnn_test"]["n"]), fmt(mc["cnn_test"]["n_bona_fide"]), fmt(mc["cnn_test"]["n_attack"]), mc["cnn_test"]["n_subjects"]),
        "target_491": "%s frames of the official training list outside fitting and early stopping (%s + %s), sessions 01-02, %d subjects" % (fmt(target["rows"]), fmt(target["n_bona_fide"]), fmt(target["n_attack"]), target["n_subjects"]),
        "thr_dev": "EER on the source dev (%s frames of the %d early-stopping subjects, sessions 01-02)" % (fmt(mc["m1_dev"]["n"]), mc["m1_dev"]["n_subjects"]),
    }


def check_counts(m: dict, n_bf: int, n_att: int, where: str, problems: list) -> None:
    if (m["n_bona_fide"], m["n_attack"]) != (n_bf, n_att):
        problems.append("%s: denominators %s/%s differ from the manifest %s/%s" % (where, m["n_bona_fide"], m["n_attack"], n_bf, n_att))


# --- run a fold di soggetti (PR #18) e confronto con N1 ------------------------------------------------------------

FOLDS_STRATA = ("01", "02", "03", "all")
SESSION_STRATA = ("01", "02", "03")
COUNT_FIELDS = ("n_bona_fide", "n_attack", "rejected_bona_fide", "accepted_attacks")
FOLDS_SUMMARY_KEYS = ("apcer_05", "bpcer_05", "acer_05", "apcer_dev", "bpcer_dev", "acer_dev", "auroc")
FOLDS_BLOCK = "Folds run"
COMPARE_BLOCK = "Folds run vs N1"
TRANSFER_FOLDS = ("subject-held-out, sessions represented (held-out subjects never used for fitting or dev; "
                  "all three sessions in training)")
TRANSFER_FOLDS_N1 = TRANSFER_FOLDS + "; restricted to the 3,000 N1 test frames (session 03)"
THR_FOLDS_ORIGIN = ("0.5 fixed a priori (configuration); dev threshold: EER on the dev fold only (M1 criterion), "
                    "written and hashed before any test scoring, one per model")
THR_DEV_POOLED = "4 fold values"


def folds_metric(m, where: str) -> dict:
    """Uno strato del run a fold come due righe di metric_row (soglia 0,5 e soglia dev), con gli stessi denominatori."""
    if not isinstance(m, dict):
        fail("%s: metrics are not an object" % where)
    out = {}
    for t, label in (("05", "0.5"), ("dev", "dev")):
        keys = {"rejected_bona_fide": "rejected_bona_fide_" + t, "accepted_attacks": "accepted_attacks_" + t,
                "apcer": "apcer_" + t, "bpcer": "bpcer_" + t, "acer": "acer_" + t}
        missing = [v for v in ["n_bona_fide", "n_attack", "auroc"] + list(keys.values()) if v not in m]
        if missing:
            fail("%s: metric fields missing %s" % (where, missing))
        r = {"n_bona_fide": m["n_bona_fide"], "n_attack": m["n_attack"], "auroc": m["auroc"]}
        r.update({k: m[v] for k, v in keys.items()})
        out[t] = metric_row(r, "%s at %s" % (where, label))
    if "n" in m and m["n"] != m["n_bona_fide"] + m["n_attack"]:
        fail("%s: n is not n_bona_fide + n_attack" % where)
    return out


def counts_of(m: dict) -> dict:
    return {f: m[f] for f in COUNT_FIELDS}


def summed(parts: list) -> dict:
    return {f: sum(p[f] for p in parts) for f in COUNT_FIELDS}


def dig(d, *keys):
    """Valore annidato o None se un livello manca (per i controlli che segnalano un problema invece di fermarsi)."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def iv(b: dict, key: str) -> str:
    return "%s-%s" % (f4(b[key]["low"]), f4(b[key]["high"]))


def folds_build(agg: dict, rec: dict, sch: dict, mc: dict, problems: list) -> dict:
    """Run a fold (PR #18): righe fold x seed (strato «all») e righe aggregate per seed, testi, annesso e controlli.

    I problemi di coerenza vanno in problems (build si ferma alla fine); una struttura mancante ferma subito."""
    for d, where in ((agg, "folds aggregate"), (rec, "folds run record"), (sch, "fold schema")):
        if d.get("dirty") is not False:
            fail("%s: dirty is not false" % where)
        run_id_check(d, where)
    for d, where in ((agg, "folds aggregate"), (rec, "folds run record")):
        if d.get("delivered_configuration") is not True or d.get("command") != "run":
            fail("%s: not a run of the delivered configuration (delivered_configuration %r, command %r)"
                 % (where, d.get("delivered_configuration"), d.get("command")))
    for k in ("run_id", "commit", "config_sha256", "finished", "bootstrap"):
        if agg[k] != rec[k]:
            problems.append("folds run record: %s %r differs from the aggregate %r" % (k, rec[k], agg[k]))
    seeds = [str(s) for s in rec["train_seeds"]]
    if [str(s) for s in sch["train_seeds"]] != seeds:  # tutto il resto dipende dai seed: fermarsi subito
        fail("folds run record: seeds %s differ from the fold schema %s" % (seeds, sch["train_seeds"]))
    K = sch["folds"]["k"]
    folds = [str(k) for k in range(1, K + 1)]
    if rec["folds"] != K or rec["models"] != K * len(seeds):
        problems.append("folds run record: %r folds and %r models, the schema gives %d and %d"
                        % (rec["folds"], rec["models"], K, K * len(seeds)))
    if (dig(rec, "schema_run", "run_id"), dig(rec, "schema_run", "config_sha256")) != (sch["run_id"], sch["config_sha256"]):
        problems.append("folds run record: schema run %r / configuration %r differ from the fold schema %r / %r"
                        % (dig(rec, "schema_run", "run_id"), dig(rec, "schema_run", "config_sha256"), sch["run_id"], sch["config_sha256"]))
    if dig(rec, "checks", "schema_config", "sha256") != sch["config_sha256"]:
        problems.append("folds run record: checked schema configuration differs from the fold schema")
    if list(agg["strata"]) != list(FOLDS_STRATA):
        fail("folds aggregate: strata %r, expected %r" % (agg["strata"], list(FOLDS_STRATA)))
    if sorted(agg["per_fold_seed"]) != folds:
        fail("folds aggregate: folds %r, the schema has %r" % (sorted(agg["per_fold_seed"]), folds))

    # Schema: denominatori di test per fold e strato, dimensioni di fitting e dev, ruoli dei fold
    expected, test_subjects, roles_of = {}, {}, {}
    sizes = {"fitting frames": set(), "dev frames": set(), "fitting subjects": set(), "dev subjects": set()}
    for k in folds:
        pf = sch["per_fold"].get(k)
        if not isinstance(pf, dict):
            fail("fold schema: fold %s missing" % k)
        roles = pf["roles"]
        c = roles["test"]["counts"]
        e = {s: (c["%s/bona_fide" % s]["frames"], c["%s/attack" % s]["frames"]) for s in SESSION_STRATA}
        e["all"] = (sum(e[s][0] for s in SESSION_STRATA), sum(e[s][1] for s in SESSION_STRATA))
        if c["total"]["frames"] != sum(e["all"]):
            problems.append("fold schema: fold %s test total %s is not the sum of its cells %s" % (k, c["total"]["frames"], sum(e["all"])))
        expected[k], test_subjects[k] = e, roles["test"]["n_subjects"]
        roles_of[k] = (pf["dev_fold"], pf["fitting_folds"])
        sizes["fitting frames"].add(roles["fitting"]["counts"]["total"]["frames"])
        sizes["dev frames"].add(roles["dev"]["counts"]["total"]["frames"])
        sizes["fitting subjects"].add(roles["fitting"]["n_subjects"])
        sizes["dev subjects"].add(roles["dev"]["n_subjects"])
    for what, v in sizes.items():
        if len(v) != 1:
            problems.append("fold schema: %s differ between folds %s" % (what, sorted(v)))
    fit_n, dev_n = min(sizes["fitting frames"]), min(sizes["dev frames"])
    fit_subj, dev_subj = min(sizes["fitting subjects"]), min(sizes["dev subjects"])
    totals = [sum(expected[k]["all"]) for k in folds]
    if dig(rec, "checks", "schema", "test_per_fold") != totals:
        problems.append("folds run record: test rows per fold %r differ from the fold schema %r"
                        % (dig(rec, "checks", "schema", "test_per_fold"), totals))
    off_bf = mc["official_train"]["n_bona_fide"] + mc["official_test"]["n_bona_fide"]
    off_att = mc["official_train"]["n_attack"] + mc["official_test"]["n_attack"]
    sum_bf, sum_att = sum(expected[k]["all"][0] for k in folds), sum(expected[k]["all"][1] for k in folds)
    if (sum_bf, sum_att) != (off_bf, off_att):
        problems.append("fold schema: the %d test folds (%s + %s) do not add up to the official lists of M2 (%s + %s)"
                        % (K, sum_bf, sum_att, off_bf, off_att))
    if sum(test_subjects.values()) != mc["official_test"]["n_subjects"]:
        problems.append("fold schema: %d held-out subjects, the official test lists have %d"
                        % (sum(test_subjects.values()), mc["official_test"]["n_subjects"]))
    if dig(rec, "checks", "originals", "located_and_hashed") != off_bf + off_att:
        problems.append("folds run record: %r originals verified, the official lists have %d"
                        % (dig(rec, "checks", "originals", "located_and_hashed"), off_bf + off_att))
    cc = rec["cache_check"]
    if cc["train"]["different"] or cc["test"]["different"]:
        problems.append("folds run record: cache frames not reproduced bit for bit")
    cache_frames = cc["train"]["equal"] + cc["test"]["equal"]
    if rec["extraction"]["n"] + cache_frames != off_bf + off_att:
        problems.append("folds run record: %s extracted + %s cache frames is not %d"
                        % (rec["extraction"]["n"], cache_frames, off_bf + off_att))
    cap = rec["checks"]["schema_config"]["wall_time_cap_minutes"] * 60
    if not rec["wall_seconds"] <= cap:
        problems.append("folds run record: wall time %s s above the cap %s s" % (rec["wall_seconds"], cap))

    # Per fold, seed e strato: metriche, denominatori dello schema, «all» = somma delle sessioni
    per = {}
    for k in folds:
        for s in seeds:
            blk = agg["per_fold_seed"][k].get(s)
            if not isinstance(blk, dict):
                fail("folds aggregate: fold %s seed %s missing" % (k, s))
            ms = {st: folds_metric(blk.get(st), "folds fold %s seed %s stratum %s" % (k, s, st)) for st in FOLDS_STRATA}
            for st in FOLDS_STRATA:
                got = (ms[st]["05"]["n_bona_fide"], ms[st]["05"]["n_attack"])
                if got != expected[k][st]:
                    problems.append("folds fold %s seed %s stratum %s: denominators %s/%s differ from the fold schema %s/%s"
                                    % (k, s, st, got[0], got[1], expected[k][st][0], expected[k][st][1]))
            for t in ("05", "dev"):
                if summed([ms[st][t] for st in SESSION_STRATA]) != counts_of(ms["all"][t]):
                    problems.append("folds fold %s seed %s: stratum all at %s is not the sum of the session strata" % (k, s, t))
            if blk["all"].get("n_subjects") != test_subjects[k]:
                problems.append("folds fold %s seed %s: %r subjects, the fold schema has %r" % (k, s, blk["all"].get("n_subjects"), test_subjects[k]))
            per[(k, s)] = ms

    # Aggregato per seed: somma dei quattro fold
    pooled = {}
    for s in seeds:
        blk = agg["pooled_per_seed"].get(s)
        if not isinstance(blk, dict):
            fail("folds aggregate: pooled seed %s missing" % s)
        for st in FOLDS_STRATA:
            m = folds_metric(blk.get(st), "folds pooled seed %s stratum %s" % (s, st))
            for t in ("05", "dev"):
                if counts_of(m[t]) != summed([per[(k, s)][st][t] for k in folds]):
                    problems.append("folds pooled seed %s stratum %s at %s: counts are not the sum of the %d folds" % (s, st, t, K))
            pooled[(s, st)] = m
        if blk["all"].get("n_subjects") != sum(test_subjects.values()):
            problems.append("folds pooled seed %s: %r subjects, expected %d" % (s, blk["all"].get("n_subjects"), sum(test_subjects.values())))

    # Mediane e intervalli sui seed: uguali ai valori per seed
    def check_summary(summary, values_of, where):
        for key in FOLDS_SUMMARY_KEYS:
            e = dig(summary, key)
            if not isinstance(e, dict):
                problems.append("%s: %s missing" % (where, key))
                continue
            vals = values_of(key)
            if e.get("values") != vals:
                problems.append("%s %s: values %r differ from the per-seed rows %r" % (where, key, e.get("values"), vals))
            elif None not in vals and (e.get("median"), e.get("min"), e.get("max")) != (statistics.median(vals), min(vals), max(vals)):
                problems.append("%s %s: median or range is not that of the values" % (where, key))
    for k in folds:
        for st in FOLDS_STRATA:
            check_summary(dig(agg, "per_fold_summary_over_seeds", k, st),
                          lambda key: [agg["per_fold_seed"][k][s][st][key] for s in seeds], "folds summary fold %s stratum %s" % (k, st))
    for st in FOLDS_STRATA:
        check_summary(dig(agg, "pooled_summary_over_seeds", st),
                      lambda key: [agg["pooled_per_seed"][s][st][key] for s in seeds], "folds pooled summary stratum %s" % st)

    # Bootstrap sui soggetti (aggregato per seed)
    bs = agg["bootstrap"]
    for k in ("resamples", "seed", "percentiles", "min_subjects"):
        if k not in bs:
            fail("folds aggregate: bootstrap field %s missing" % k)
    boot = {}
    for s in seeds:
        for st in FOLDS_STRATA:
            b = dig(agg, "bootstrap_per_seed", s, st)
            n_sub = agg["pooled_per_seed"][s][st].get("n_subjects")
            if not isinstance(n_sub, int) or n_sub < bs["min_subjects"]:
                continue  # nessun intervallo con meno soggetti del minimo dichiarato
            if not isinstance(b, dict):
                problems.append("folds bootstrap seed %s stratum %s missing" % (s, st))
                continue
            if (b.get("n_subjects"), b.get("resamples"), b.get("seed"), b.get("percentiles")) != (n_sub, bs["resamples"], bs["seed"], bs["percentiles"]):
                problems.append("folds bootstrap seed %s stratum %s: subjects or parameters differ from the declared ones" % (s, st))
            ok = True
            for key in ("acer_05", "acer_dev", "auroc"):
                x = b.get(key)
                if not isinstance(x, dict) or not isinstance(x.get("low"), (int, float)) or not isinstance(x.get("high"), (int, float)) \
                        or x["low"] > x["high"]:
                    problems.append("folds bootstrap seed %s stratum %s: %s interval missing or reversed" % (s, st, key))
                    ok = False
            if ok:
                boot[(s, st)] = b

    # Modelli: 12 schede con le dimensioni dello schema, epoca ripristinata valida, tempo nel record
    names = ["fold%s_seed%s" % (k, s) for k in folds for s in seeds]
    training = agg["training"]
    if sorted(training) != sorted(names):
        problems.append("folds aggregate: models %s, expected %s" % (sorted(training), sorted(names)))
    for name in names:
        tr = training.get(name)
        if not isinstance(tr, dict):
            fail("folds aggregate: training of %s missing" % name)
        for f in ("epochs_run", "restored_epoch", "restored_by", "best_val_loss", "train_seconds", "n_fitting", "n_dev",
                  "weights_sha256", "dev_threshold", "dev_apcer_at_threshold", "dev_bpcer_at_threshold"):
            if f not in tr:
                fail("folds aggregate: training field %s of %s missing" % (f, name))
        if (tr["n_fitting"], tr["n_dev"]) != (fit_n, dev_n):
            problems.append("folds %s: %s fitting / %s dev frames, the fold schema has %s / %s" % (name, tr["n_fitting"], tr["n_dev"], fit_n, dev_n))
        if not 1 <= tr["restored_epoch"] <= tr["epochs_run"]:
            problems.append("folds %s: restored epoch %s outside 1..%s" % (name, tr["restored_epoch"], tr["epochs_run"]))
        if "train " + name not in rec["timings"]:
            problems.append("folds run record: no timing for train %s" % name)

    # Testi, righe della tabella unica, annesso
    T = {"folds_train": "NUAA, all three sessions (01-03): %s fitting frames of the %d subjects of the two fitting folds, "
                        "%d per class x session cell (deterministic spread rule)"
                        % (fmt(fit_n), fit_subj, sch["sampling"]["fitting_per_cell"]),
         "folds_dev": "dev fold (the next held-out fold): %s frames of its %d subjects, all three sessions, %d per class x "
                      "session cell (early stopping, learning-rate schedule, EER threshold)"
                      % (fmt(dev_n), dev_subj, sch["sampling"]["dev_per_cell"]),
         "folds_test_all": "the %d held-out folds pooled: all %s official NUAA frames (%s + %s), %d subjects, sessions 01-03, "
                           "each frame scored by the model of its fold" % (K, fmt(off_bf + off_att), fmt(off_bf), fmt(off_att),
                                                                           sum(test_subjects.values())),
         "folds_legend": "fitting %s + dev %s frames of %d + %d subjects outside the held-out fold (the two fitting folds and "
                         "the dev fold), all three sessions, %d + %d per class x session cell"
                         % (fmt(fit_n), fmt(dev_n), fit_subj, dev_subj, sch["sampling"]["fitting_per_cell"], sch["sampling"]["dev_per_cell"])}
    for k in folds:
        T["folds_test_%s" % k] = ("held-out fold %s: all %s frames (%s + %s) of its %d subjects, sessions 01-03"
                                  % (k, fmt(sum(expected[k]["all"])), fmt(expected[k]["all"][0]), fmt(expected[k]["all"][1]), test_subjects[k]))
    def metric_cells(m):
        a, d = m["05"], m["dev"]
        return dict(n_bona_fide=a["n_bona_fide"], n_attack=a["n_attack"], rejected_bona_fide=a["rejected_bona_fide"],
                    accepted_attacks=a["accepted_attacks"], apcer=f4(a["apcer"]), bpcer=f4(a["bpcer"]), acer=f4(a["acer"]),
                    auroc=f4(a["auroc"]), rejected_bona_fide_dev=d["rejected_bona_fide"], accepted_attacks_dev=d["accepted_attacks"],
                    apcer_dev=f4(d["apcer"]), bpcer_dev=f4(d["bpcer"]), acer_dev=f4(d["acer"]))
    rows = []
    for k in folds:
        dev_fold, fit_folds = roles_of[k]
        for s in seeds:
            tr = training["fold%s_seed%s" % (k, s)]
            rows.append(row(row="FOLDS-f%s-seed%s" % (k, s), block=FOLDS_BLOCK,
                            model="LivenessNet, new CPU training, held-out fold %s, seed %s (fitting folds %s, dev fold %s; "
                                  "restored epoch %d of %d, %s; min val_loss %.6f)"
                                  % (k, s, " and ".join(str(x) for x in fit_folds), dev_fold, tr["restored_epoch"],
                                     tr["epochs_run"], tr["restored_by"], tr["best_val_loss"]),
                            training_data=T["folds_train"], early_stopping=T["folds_dev"], test_set=T["folds_test_%s" % k],
                            extractor=EXTRACTOR_FOLDS, threshold="0.5", threshold_origin=THR_FOLDS_ORIGIN,
                            threshold_dev="%.6f" % tr["dev_threshold"], run_id=agg["run_id"], transfer=TRANSFER_FOLDS,
                            **metric_cells(per[(k, s)]["all"])))
    for s in seeds:
        b = boot.get((s, "all"))
        rows.append(row(row="FOLDS-pooled-seed%s" % s, block=FOLDS_BLOCK,
                        model="LivenessNet, the %d fold models of seed %s, pooled over the held-out folds" % (K, s),
                        training_data=T["folds_train"], early_stopping=T["folds_dev"], test_set=T["folds_test_all"],
                        extractor=EXTRACTOR_FOLDS, threshold="0.5", threshold_origin=THR_FOLDS_ORIGIN,
                        threshold_dev=THR_DEV_POOLED, run_id=agg["run_id"], transfer=TRANSFER_FOLDS,
                        bootstrap=("ACER 0.5 %s; ACER dev %s; AUROC %s" % (iv(b, "acer_05"), iv(b, "acer_dev"), iv(b, "auroc"))
                                   if b else "NA"),
                        **metric_cells(pooled[(s, "all")])))

    def stratum_cells(m):
        a, d = m["05"], m["dev"]
        return ["%s / %s" % (a["rejected_bona_fide"], a["accepted_attacks"]), f4(a["bpcer"]), f4(a["apcer"]), f4(a["acer"]),
                "%s / %s" % (d["rejected_bona_fide"], d["accepted_attacks"]), f4(d["bpcer"]), f4(d["apcer"]), f4(d["acer"])]

    def summary_cells(e):
        return ["%s (%s-%s)" % (f4(e[key]["median"]), f4(e[key]["min"]), f4(e[key]["max"])) for key in ("acer_05", "acer_dev", "auroc")]
    annex = {"per_fold_seed": [], "per_fold_summary": [], "pooled": [], "pooled_summary": [], "training": []}
    for k in folds:
        for s in seeds:
            for st in FOLDS_STRATA:
                m, raw = per[(k, s)][st], agg["per_fold_seed"][k][s][st]
                annex["per_fold_seed"].append([k, s, st, m["05"]["n_bona_fide"], m["05"]["n_attack"], raw.get("n_subjects", "NA")]
                                              + stratum_cells(m) + [f4(m["05"]["auroc"])])
    for k in folds:
        for st in FOLDS_STRATA:
            annex["per_fold_summary"].append([k, st] + summary_cells(agg["per_fold_summary_over_seeds"][k][st]))
    for s in seeds:
        for st in FOLDS_STRATA:
            m, raw, b = pooled[(s, st)], agg["pooled_per_seed"][s][st], boot.get((s, st))
            a, d = m["05"], m["dev"]
            annex["pooled"].append([s, st, a["n_bona_fide"], a["n_attack"], raw.get("n_subjects", "NA"),
                                    "%s / %s" % (a["rejected_bona_fide"], a["accepted_attacks"]), f4(a["bpcer"]), f4(a["apcer"]),
                                    "%s [%s]" % (f4(a["acer"]), iv(b, "acer_05") if b else "NA"),
                                    "%s / %s" % (d["rejected_bona_fide"], d["accepted_attacks"]), f4(d["bpcer"]), f4(d["apcer"]),
                                    "%s [%s]" % (f4(d["acer"]), iv(b, "acer_dev") if b else "NA"),
                                    "%s [%s]" % (f4(a["auroc"]), iv(b, "auroc") if b else "NA")])
    for st in FOLDS_STRATA:
        annex["pooled_summary"].append([st] + summary_cells(agg["pooled_summary_over_seeds"][st]))
    for name in names:
        tr = training[name]
        annex["training"].append([name, tr["n_fitting"], tr["n_dev"], tr["epochs_run"], tr["restored_epoch"], tr["restored_by"],
                                  "%.4f" % tr["best_val_loss"], "%.0f" % tr["train_seconds"], "%.6f" % tr["dev_threshold"],
                                  f4(tr["dev_bpcer_at_threshold"]), f4(tr["dev_apcer_at_threshold"])])
    secs = [training[n]["train_seconds"] for n in names]
    info = {"run_id": agg["run_id"], "commit": agg["commit"], "config_sha256": agg["config_sha256"], "finished": agg["finished"],
            "schema_run": sch["run_id"], "schema_config_sha256": sch["config_sha256"], "folds": K, "seeds": seeds,
            "models": len(names), "train_seconds_min": min(secs), "train_seconds_max": max(secs),
            "wall_seconds": rec["wall_seconds"], "wall_cap_seconds": cap, "cache_frames": cache_frames,
            "extracted_frames": rec["extraction"]["n"], "test_frames_per_fold": totals, "bootstrap": bs}
    return {"rows": rows, "texts": T, "annex": annex, "info": info, "seeds": seeds, "metric_cells": metric_cells, "pooled": pooled}


COMPARE_FIELDS = ["run_id", "commit", "dirty", "config_sha256", "delivered_configuration", "command", "finished", "declarations",
                  "decision", "bootstrap", "inputs", "per_seed", "summary_over_seeds", "bootstrap_per_seed", "control_values"]
COMPARE_SUMMARY_KEYS = ("folds_bpcer_05", "folds_apcer_05", "folds_acer_05", "folds_bpcer_dev", "folds_apcer_dev", "folds_acer_dev",
                        "folds_auroc", "n1_bpcer_05", "n1_apcer_05", "n1_acer_05", "n1_auroc", "diff_acer_05", "diff_acer_dev",
                        "diff_auroc")
COMPARE_BOOT_KEYS = ("folds_acer_05", "folds_acer_dev", "folds_auroc", "n1_acer_05", "n1_auroc", "diff_acer_05", "diff_acer_dev",
                     "diff_auroc")
COMPARE_DECLARATIONS = ("descriptive", "post_hoc", "no_training", "no_inference", "no_new_threshold")


def compare_trial_check(aggregate: dict) -> None:
    """Il confronto deve venire da un run ufficiale: albero pulito e run id senza «-dev»."""
    issues = []
    if aggregate.get("dirty") is not False:
        issues.append("folds vs N1: dirty is %r (trial run from a modified tree)" % aggregate.get("dirty"))
    if str(aggregate.get("run_id", "")).endswith("-dev"):
        issues.append("folds vs N1: run id %r ends with -dev (trial run)" % aggregate.get("run_id"))
    if issues:
        fail("; ".join(issues) + "; the official run must replace it")


def compare_rows(aggregate: dict, ctx: dict, problems: list) -> dict:
    """Confronto descrittivo con N1 sugli stessi 3.000 identificativi (research/nuaa-folds-n1-compare, aggregato pubblico).

    Righe FOLDSN1-seedNN (i modelli a fold sui 3.000, a 0,5 e alla soglia dev); le righe N1 della tabella unica fanno da
    confronto e non sono duplicate. Controlli: run della configurazione consegnata; ingressi (manifest cnn_test di M2,
    run a fold, N1, schema dei fold); N1 del confronto uguale alle righe N1 (conteggi e tassi a quattro decimali, AUROC);
    fold del confronto contenuti nell'aggregato per seed della sessione 03 del run a fold (sottoinsieme: coerenti, non
    uguali); differenze = fold meno N1; decisioni appaiate e conteggi per soggetto coerenti con i conteggi; mediane e
    intervalli uguali ai valori per seed; bootstrap con i parametri dichiarati; valori di controllo del coordinatore
    riprodotti entro le tolleranze. ctx: n1, n1_seeds, mc, m2, folds, fsch, T, src."""
    folds, mc = ctx["folds"], ctx["mc"]
    seeds = folds["seeds"]
    if aggregate.get("delivered_configuration") is not True or aggregate.get("command") != "run":
        fail("folds vs N1: not a run of the delivered configuration (delivered_configuration %r, command %r)"
             % (aggregate.get("delivered_configuration"), aggregate.get("command")))
    for k in COMPARE_DECLARATIONS:
        if dig(aggregate, "declarations", k) is not True:
            problems.append("folds vs N1: declaration %s is not true" % k)
    inp = aggregate["inputs"]
    man = mc["cnn_test"]
    want = {"sha256": ctx["m2"]["manifests"]["cnn_test"].get("sha256"), "rows": man["n"], "n_bona_fide": man["n_bona_fide"],
            "n_attack": man["n_attack"], "n_subjects": man["n_subjects"], "schema_config_sha256": ctx["fsch"]["config_sha256"]}
    got = {k: dig(inp, "manifest", k) for k in want}
    if got != want:
        problems.append("folds vs N1: manifest inputs %r differ from M2 and the fold schema %r" % (got, want))
    if dig(inp, "folds_run", "run_id") != folds["info"]["run_id"]:
        problems.append("folds vs N1: folds run %r is not %r" % (dig(inp, "folds_run", "run_id"), folds["info"]["run_id"]))
    if dig(inp, "n1_run", "run_id") != ctx["n1"]["run_id"]:
        problems.append("folds vs N1: N1 run %r is not %r" % (dig(inp, "n1_run", "run_id"), ctx["n1"]["run_id"]))
    total = sum(folds["info"]["test_frames_per_fold"])
    for s in seeds:
        if dig(inp, "folds_run", "rows_per_seed", s) != total or dig(inp, "folds_run", "cnn_test_rows_per_seed", s) != man["n"] \
                or dig(inp, "n1_run", "rows_per_seed", s) != man["n"]:
            problems.append("folds vs N1 seed %s: score rows differ from the folds run (%d) or the manifest (%d)" % (s, total, man["n"]))
    tspf = {k: v["n_subjects"] for k, v in ((k, ctx["fsch"]["per_fold"][k]["roles"]["test"]) for k in ctx["fsch"]["per_fold"])}
    if dig(inp, "folds_run", "test_subjects_per_fold") != tspf:
        problems.append("folds vs N1: test subjects per fold differ from the fold schema")
    if {k: dig(aggregate, "bootstrap", k) for k in folds["info"]["bootstrap"]} != folds["info"]["bootstrap"] \
            or not isinstance(dig(aggregate, "bootstrap", "recipe"), str):
        problems.append("folds vs N1: bootstrap parameters differ from the folds run, or the recipe is not declared")

    rows, per = [], {}
    for s in seeds:
        blk = aggregate["per_seed"].get(s)
        if not isinstance(blk, dict):
            fail("folds vs N1: seed %s missing" % s)
        fm = folds_metric(blk.get("folds"), "folds vs N1 seed %s folds" % s)
        n1raw = blk.get("n1")
        if not isinstance(n1raw, dict):
            fail("folds vs N1 seed %s: n1 missing" % s)
        n1m = metric_row({"n_bona_fide": n1raw.get("n_bona_fide"), "n_attack": n1raw.get("n_attack"),
                          "rejected_bona_fide": n1raw.get("rejected_bona_fide_05"), "accepted_attacks": n1raw.get("accepted_attacks_05"),
                          "apcer": n1raw.get("apcer_05"), "bpcer": n1raw.get("bpcer_05"), "acer": n1raw.get("acer_05"),
                          "auroc": n1raw.get("auroc")}, "folds vs N1 seed %s n1" % s)
        for m, where in ((fm["05"], "folds"), (n1m, "N1")):
            check_counts(m, man["n_bona_fide"], man["n_attack"], "folds vs N1 seed %s %s" % (s, where), problems)
        # N1 del confronto = riga N1 della tabella unica (stessi 3.000 frame)
        ref = ctx["n1_rows"].get(s)
        if ref is None or [n1m[k] for k in COUNT_FIELDS] != [ref[k] for k in COUNT_FIELDS] \
                or [f4(n1m[k]) for k in RATES] != [ref[k] for k in RATES]:
            problems.append("folds vs N1 seed %s: the N1 metrics differ from the N1 row of the unified table" % s)
        # sottoinsieme della sessione 03 del run a fold: conteggi non superiori all'aggregato per seed (insieme diverso)
        pooled03 = folds["pooled"][(s, "03")]
        for t in ("05", "dev"):
            if any(fm[t][k] > pooled03[t][k] for k in COUNT_FIELDS):
                problems.append("folds vs N1 seed %s at %s: counts exceed the pooled session 03 of the folds run" % (s, t))
        d = blk.get("difference_folds_minus_n1") or {}
        for key, value in (("diff_acer_05", fm["05"]["acer"] - n1m["acer"]), ("diff_acer_dev", fm["dev"]["acer"] - n1m["acer"]),
                           ("diff_auroc", fm["05"]["auroc"] - n1m["auroc"])):
            if not isinstance(d.get(key), (int, float)) or abs(d[key] - value) > 1e-9:
                problems.append("folds vs N1 seed %s: %s %r is not folds minus N1 (%s)" % (s, key, d.get(key), f4(value)))
        pdec = blk.get("paired_decisions") or {}
        for t in ("05", "dev"):
            pt = pdec.get(t)
            if not isinstance(pt, dict):
                problems.append("folds vs N1 seed %s: paired decisions at %s missing" % (s, t))
                continue
            changed = 0
            for cls, n_key, err_key in (("bona_fide", "n_bona_fide", "rejected_bona_fide"), ("attack", "n_attack", "accepted_attacks")):
                c = pt.get(cls) or {}
                vals = [c.get(k) for k in ("both_correct", "n1_only_correct", "folds_only_correct", "both_wrong")]
                if not all(isinstance(v, int) for v in vals):
                    problems.append("folds vs N1 seed %s at %s: paired decisions of %s incomplete" % (s, t, cls))
                    continue
                bc, n1o, fo, bw = vals
                if bc + n1o + fo + bw != fm["05"][n_key] or bc + fo != fm[t][n_key] - fm[t][err_key] \
                        or bc + n1o != n1m[n_key] - n1m[err_key]:
                    problems.append("folds vs N1 seed %s at %s: paired decisions of %s do not match the counts" % (s, t, cls))
                changed += n1o + fo
            if pt.get("decisions_changed") != changed or pt.get("n") != man["n"]:
                problems.append("folds vs N1 seed %s at %s: decisions changed %r, expected %d" % (s, t, pt.get("decisions_changed"), changed))
        psc = blk.get("per_subject_counts") or {}
        if psc.get("n_subjects") != man["n_subjects"]:
            problems.append("folds vs N1 seed %s: %r subjects, the manifest has %d" % (s, psc.get("n_subjects"), man["n_subjects"]))
        for t in ("05", "dev"):
            for measure in ("error_rate", "acer"):
                c = dig(psc, "comparison", t, measure) or {}
                parts = [c.get(k) for k in ("folds_lower", "equal", "folds_higher", "not_defined")]
                undefined = 0 if measure == "error_rate" else man["n_subjects"] - (psc.get("with_both_classes") or 0)
                if not all(isinstance(v, int) for v in parts) or sum(parts) != man["n_subjects"] or parts[3] != undefined:
                    problems.append("folds vs N1 seed %s at %s: subject counts of %s do not add up" % (s, t, measure))
        if blk.get("test_subjects_per_fold") != tspf:
            problems.append("folds vs N1 seed %s: test subjects per fold differ from the fold schema" % s)
        b = dig(aggregate, "bootstrap_per_seed", s)
        bs = aggregate["bootstrap"]
        if not isinstance(b, dict) or (b.get("n_subjects"), b.get("resamples"), b.get("seed"), b.get("percentiles")) != \
                (man["n_subjects"], bs.get("resamples"), bs.get("seed"), bs.get("percentiles")):
            problems.append("folds vs N1 bootstrap seed %s: missing, or subjects or parameters differ from the declared ones" % s)
            b = None
        else:
            for key in COMPARE_BOOT_KEYS:
                x = b.get(key)
                if not isinstance(x, dict) or not isinstance(x.get("low"), (int, float)) or not isinstance(x.get("high"), (int, float)) \
                        or x["low"] > x["high"]:
                    problems.append("folds vs N1 bootstrap seed %s: %s interval missing or reversed" % (s, key))
                    b = None
                    break
        per[s] = {"folds": fm, "n1": n1m, "diff": d, "paired": pdec, "subjects": psc, "boot": b}
        rows.append(row(row="FOLDSN1-seed%s" % s, block=COMPARE_BLOCK,
                        model="LivenessNet, the %d fold models of seed %s, on the N1 test frames (each frame scored by the model "
                              "of its fold)" % (folds["info"]["folds"], s),
                        training_data=ctx["T"]["folds_train"], early_stopping=ctx["T"]["folds_dev"], test_set=ctx["T"]["test_3000"],
                        extractor=EXTRACTOR_FOLDS, threshold="0.5", threshold_origin=THR_FOLDS_ORIGIN,
                        threshold_dev=THR_DEV_POOLED, run_id=aggregate["run_id"], transfer=TRANSFER_FOLDS_N1,
                        bootstrap=("ACER 0.5 %s; ACER dev %s; AUROC %s" % (iv(b, "folds_acer_05"), iv(b, "folds_acer_dev"),
                                                                         iv(b, "folds_auroc")) if b else "NA"),
                        source=ctx["src"], **folds["metric_cells"](fm)))

    # Mediane e intervalli sui seed
    def value_of(s, key):
        side, rest = key.split("_", 1)
        if side == "diff":
            return dig(aggregate, "per_seed", s, "difference_folds_minus_n1", key)
        return dig(aggregate, "per_seed", s, side, rest)
    for key in COMPARE_SUMMARY_KEYS:
        e = dig(aggregate, "summary_over_seeds", key)
        vals = [value_of(s, key) for s in seeds]
        if not isinstance(e, dict) or e.get("values") != vals:
            problems.append("folds vs N1 summary %s: missing or values differ from the per-seed values" % key)
        elif None not in vals and (e.get("median"), e.get("min"), e.get("max")) != (statistics.median(vals), min(vals), max(vals)):
            problems.append("folds vs N1 summary %s: median or range is not that of the values" % key)

    # Valori di controllo del coordinatore: calcolati = valori del run; accordo alle cifre date dal coordinatore
    # (ACER in percentuale con due decimali, AUROC con cinque), per arrotondamento, senza tolleranze
    cv = aggregate["control_values"]
    ctrl_rows = []
    for key, ckey, side, digits, scale, unit in (("acer_05", "acer_05_percent", "05", 2, 100, " %"),
                                                  ("acer_dev", "acer_dev_percent", "dev", 2, 100, " %"),
                                                  ("auroc", "auroc", "05", 5, 1, "")):
        ctrl, comp = dig(cv, "control", ckey), dig(cv, "computed", key)
        mine = [per[s]["folds"][side]["acer" if key != "auroc" else "auroc"] for s in seeds]
        if not isinstance(ctrl, list) or not isinstance(comp, list) or len(ctrl) != len(seeds) or len(comp) != len(seeds):
            problems.append("folds vs N1: control values of %s missing" % key)
            continue
        if any(abs(a - b) > 1e-12 for a, b in zip(comp, mine)):
            problems.append("folds vs N1: computed control values of %s differ from the per-seed values" % key)
        if any(round(scale * b, digits) != round(a, digits) for a, b in zip(ctrl, comp)):
            problems.append("folds vs N1: control values of %s not reproduced at the digits given by the coordinator" % key)
        for s, a, b in zip(seeds, ctrl, comp):
            ctrl_rows.append([key, s, ("%%.%df" % digits) % a + unit, "%.5f" % b])
    if cv.get("reproduced") is not True or not isinstance(cv.get("comparison"), str):
        problems.append("folds vs N1: control values not declared reproduced, or the comparison rule missing")

    # Annesso (copiato dall'aggregato)
    annex = {"per_seed": [], "differences": [], "summary": [], "paired": [], "subjects": [], "control": ctrl_rows}
    for s in seeds:
        p, b = per[s], per[s]["boot"]
        bi = (lambda key: "[%s, %s]" % (f4(b[key]["low"]), f4(b[key]["high"]))) if b else (lambda key: "[NA]")
        for label, m, ab, au in (("folds, 0.5", p["folds"]["05"], "folds_acer_05", bi("folds_auroc")),
                                 ("folds, dev", p["folds"]["dev"], "folds_acer_dev", "as above"),
                                 ("N1, 0.5", p["n1"], "n1_acer_05", bi("n1_auroc"))):
            annex["per_seed"].append([s, label, "%s / %s" % (m["rejected_bona_fide"], m["accepted_attacks"]), f4(m["bpcer"]),
                                      f4(m["apcer"]), "%s %s" % (f4(m["acer"]), bi(ab)),
                                      "%s %s" % (f4(m["auroc"]), au) if au != "as above" else au])
        annex["differences"].append([s] + ["%s %s" % (f4(p["diff"][k]), bi(k)) for k in ("diff_acer_05", "diff_acer_dev", "diff_auroc")])
        for t, tl in (("05", "0.5"), ("dev", "dev")):
            pt = p["paired"][t]
            annex["paired"].append([s, tl] + [" / ".join(str(pt[cls][k]) for k in ("both_correct", "n1_only_correct",
                                                                                  "folds_only_correct", "both_wrong"))
                                              for cls in ("bona_fide", "attack")] + [pt["decisions_changed"]])
            for measure, ml in (("error_rate", "error rate"), ("acer", "acer")):
                c = p["subjects"]["comparison"][t][measure]
                annex["subjects"].append([s, tl, ml, c["folds_lower"], c["equal"], c["folds_higher"], c["not_defined"]])
    labels = {"folds_acer_05": "folds ACER 0.5", "folds_acer_dev": "folds ACER dev", "folds_auroc": "folds AUROC",
              "n1_acer_05": "N1 ACER 0.5", "n1_auroc": "N1 AUROC", "diff_acer_05": "difference ACER 0.5",
              "diff_acer_dev": "difference ACER dev vs N1 0.5", "diff_auroc": "difference AUROC"}
    for key, label in labels.items():
        e = aggregate["summary_over_seeds"][key]
        annex["summary"].append([label, f4(e["median"]), f4(e["min"]), f4(e["max"])])
    info = {"run_id": aggregate["run_id"], "commit": aggregate["commit"], "dirty": aggregate["dirty"],
            "config_sha256": aggregate["config_sha256"], "finished": aggregate["finished"],
            "declarations": aggregate["declarations"], "with_both_classes": per[seeds[0]]["subjects"].get("with_both_classes"),
            "control_source": cv.get("source"), "control_comparison": cv.get("comparison"),
            "bootstrap_recipe": aggregate["bootstrap"].get("recipe"),
            "n1_auroc": [per[s]["n1"]["auroc"] for s in seeds]}
    return {"rows": rows, "annex": annex, "info": info}


def build(research: Path) -> dict:
    paths = {k: research / v for k, v in SOURCES.items()}
    for k, p in paths.items():
        if not p.is_file():
            fail("source %s missing: %s" % (k, p))
    hashes = {k: sha256_file(p) for k, p in paths.items()}

    c1 = read_c1_table(paths["c1_table"])
    m1 = read_m1_table(paths["m1_table"])
    m2 = read_json(paths["m2_manifests"], ["manifests", "passed", "fitted_subjects", "early_stopping_subjects", "exclusions", "commit"])
    n1 = read_json(paths["n1_aggregate"], ["run_id", "commit", "seeds", "train_seeds", "threshold"])
    n1s = read_json(paths["n1_subjects"], ["run_id", "commit", "analyzers", "accepted_m1_table_reproduced"], clean=False)
    if n1s.get("dirty") is True:
        fail("%s: dirty is true" % paths["n1_subjects"].name)
    n2s = read_json(paths["n2_sessions"], ["run_id", "commit", "seeds", "train_seeds", "n1_study", "target"])
    n2c = read_json(paths["n2_combined"], ["run_id", "commit", "variants", "n1_reference", "model"])
    crop = read_json(paths["crop"], ["run_id", "commit", "metrics", "paired", "c1_reference", "weights"])
    # Il run a fold e lo schema non hanno il campo passed: albero pulito e configurazione consegnata li controlla folds_build
    fsch = read_json(paths["folds_schema"], ["run_id", "commit", "dirty", "config_sha256", "folds", "per_fold", "sampling",
                                             "train_seeds"], clean=False)
    fagg = read_json(paths["folds_aggregate"], ["run_id", "commit", "dirty", "config_sha256", "delivered_configuration", "command",
                                                "finished", "bootstrap", "strata", "per_fold_seed", "per_fold_summary_over_seeds",
                                                "pooled_per_seed", "pooled_summary_over_seeds", "bootstrap_per_seed", "training"],
                     clean=False)
    frec = read_json(paths["folds_record"], ["run_id", "commit", "dirty", "config_sha256", "delivered_configuration", "command",
                                             "finished", "wall_seconds", "timings", "checks", "cache_check", "extraction",
                                             "schema_run", "train_seeds", "folds", "models", "bootstrap"], clean=False)
    for d, where in ((n1, "N1"), (n1s, "N1 subjects"), (n2s, "N2 sessions"), (n2c, "N2 combined"), (crop, "crop")):
        run_id_check(d, where)

    mc = manifest_counts(m2)
    target = n2s["target"]
    for k in ("rows", "n_bona_fide", "n_attack", "n_subjects"):
        if k not in target:
            fail("N2 sessions: target has no field %s" % k)
    excl = m2["exclusions"].get("official_train_not_sampled") if isinstance(m2["exclusions"], dict) else None
    if not isinstance(excl, dict):
        fail("M2 manifests: exclusions.official_train_not_sampled missing")
    problems = []
    if (target["rows"], target["n_bona_fide"], target["n_attack"], target["n_subjects"]) != \
            (excl.get("n"), excl.get("n_bona_fide"), excl.get("n_attack"), excl.get("n_subjects")):
        problems.append("N2 sessions: target (%s) differs from the M2 exclusions official_train_not_sampled (%s)"
                        % ((target["rows"], target["n_bona_fide"], target["n_attack"], target["n_subjects"]),
                           (excl.get("n"), excl.get("n_bona_fide"), excl.get("n_attack"), excl.get("n_subjects"))))
    if mc["cnn_train"]["n_subjects"] != m2["fitted_subjects"].get("n"):
        problems.append("M2 manifests: cnn_train subjects %s differ from fitted_subjects %s" % (mc["cnn_train"]["n_subjects"], m2["fitted_subjects"].get("n")))
    if mc["cnn_es"]["n_subjects"] != m2["early_stopping_subjects"].get("n"):
        problems.append("M2 manifests: cnn_es subjects %s differ from early_stopping_subjects %s" % (mc["cnn_es"]["n_subjects"], m2["early_stopping_subjects"].get("n")))
    if mc["cnn_train"]["n_subjects"] + mc["cnn_es"]["n_subjects"] != mc["official_train"]["n_subjects"] or target["n_subjects"] != mc["official_train"]["n_subjects"]:
        problems.append("M2 manifests: fitting + early-stopping subjects (%s + %s) and the N2 target subjects (%s) must equal the official training subjects (%s)"
                        % (mc["cnn_train"]["n_subjects"], mc["cnn_es"]["n_subjects"], target["n_subjects"], mc["official_train"]["n_subjects"]))
    if problems:
        fail("consistency problems:\n  - " + "\n  - ".join(problems))
    T = texts_of(mc, target)
    rows = []
    src = lambda k: "%s (%s)" % (SOURCES[k], hashes[k][:12])

    # C1 e M1: le quattro CNN NUAA sulle 300, a 0,5 e alla soglia dev
    for a in NUAA_CNNS:
        r1, rm = c1.get(a), m1[a]
        if r1 is None:
            problems.append("C1 table: row %s missing" % a)
            continue
        for k1, km in (("apcer", "APCER@0.5"), ("bpcer", "BPCER@0.5"), ("acer", "ACER@0.5"), ("auc", "test AUC")):
            if r1[k1] != rm[km]:
                problems.append("C1 row %s: %s %s differs from M1 table %s %s" % (a, k1, r1[k1], km, rm[km]))
        if (r1["n_bona_fide"], r1["n_attack"]) != (rm["n_bona_fide"], rm["n_attack"]):
            problems.append("C1 row %s: denominators differ between the C1 and the M1 table" % a)
        if rm["dev flag"] not in ("", "-"):
            problems.append("M1 row %s: dev flag %r (threshold declared non-informative)" % (a, rm["dev flag"]))
        n_bf, n_att = as_int(rm["n_bona_fide"], "M1 " + a), as_int(rm["n_attack"], "M1 " + a)
        if (n_bf, n_att) != (mc["c1"]["n_bona_fide"], mc["c1"]["n_attack"]):
            problems.append("M1 row %s: denominators %d/%d differ from the c1 manifest" % (a, n_bf, n_att))
        counts = {}
        for suffix in ("0.5", "dev"):
            rej = as_int(rm["rejected bona fide@" + suffix], "M1 " + a)
            acc = as_int(rm["accepted attacks@" + suffix], "M1 " + a)
            counts[suffix] = (rej, acc)
            for rate, value in (("APCER", acc / n_att), ("BPCER", rej / n_bf), ("ACER", (acc / n_att + rej / n_bf) / 2)):
                if f4(value) != rm["%s@%s" % (rate, suffix)]:
                    problems.append("M1 row %s: %s@%s %s is not the rate of the counts (%s)" % (a, rate, suffix, rm["%s@%s" % (rate, suffix)], f4(value)))
        sub = n1s["analyzers"].get(a)
        if not isinstance(sub, dict) or "full" not in sub or "leave_one_subject_out" not in sub:
            problems.append("N1 subjects: analyzer %s missing or incomplete" % a)
        else:
            for suffix, key in (("0.5", "at_0_5"), ("dev", "at_m1")):
                fa, fr = sub["full"][key].get("fa"), sub["full"][key].get("fr")
                if (fr, fa) != counts[suffix]:
                    problems.append("N1 subjects: %s counts at %s (fr %s, fa %s) differ from the M1 table %s" % (a, key, fr, fa, counts[suffix]))
            if f4(sub["full"]["at_m1"]["acer"]) != rm["ACER@dev"]:
                problems.append("N1 subjects: %s ACER at the M1 threshold %s differs from the M1 table %s" % (a, sub["full"]["at_m1"]["acer"], rm["ACER@dev"]))
            if "%.6f" % float(sub["m1_threshold"]) != rm["dev threshold"]:
                problems.append("N1 subjects: M1 threshold of %s differs from the M1 table" % a)
        rej, acc = counts["0.5"]
        rows.append(row(row="C1-%s" % a, block="C1", model="%s (C1 checkpoint of 2026-09-12)" % CNN_NAMES[a],
                        training_data=T["train"], early_stopping=T["es"], test_set=T["test_c1"],
                        extractor=EXTRACTOR_C1, threshold="0.5", threshold_origin=THR_FIXED,
                        n_bona_fide=n_bf, n_attack=n_att, rejected_bona_fide=rej, accepted_attacks=acc,
                        apcer=rm["APCER@0.5"], bpcer=rm["BPCER@0.5"], acer=rm["ACER@0.5"], auroc=rm["test AUC"],
                        run_id=C1_RUN, transfer=TRANSFER_SESSION_ALL, source=src("m1_table") + "; " + src("c1_table")))
        rej, acc = counts["dev"]
        rows.append(row(row="M1-%s" % a, block="M1", model="%s (C1 checkpoint of 2026-09-12)" % CNN_NAMES[a],
                        training_data=T["train"], early_stopping=T["es"], test_set=T["test_c1"],
                        extractor=EXTRACTOR_C1, threshold=rm["dev threshold"],
                        threshold_origin=T["thr_dev"] + "; dev AUC %s, operational dev EER %s" % (rm["dev AUC"], rm["operational dev EER"]),
                        n_bona_fide=n_bf, n_attack=n_att, rejected_bona_fide=rej, accepted_attacks=acc,
                        apcer=rm["APCER@dev"], bpcer=rm["BPCER@dev"], acer=rm["ACER@dev"], auroc=rm["test AUC"],
                        run_id=M1_RUN, transfer=TRANSFER_SESSION_ALL, source=src("m1_table")))
    rows.sort(key=lambda r: (r["block"] != "C1", NUAA_CNNS.index(r["row"].split("-", 1)[1])))

    # N1 parte 2: LivenessNet, tre seed, test 3.000
    seeds = [str(s) for s in n1["train_seeds"]]
    n1_seeds = seed_block(n1, seeds, "N1")
    if f4(n1["threshold"]) != "0.5000":
        problems.append("N1: threshold is %r, expected 0.5" % n1["threshold"])
    for s in seeds:
        m = metric_row(n1_seeds[s].get("metrics", {}).get("test_3000"), "N1 seed %s test_3000" % s)
        check_counts(m, mc["cnn_test"]["n_bona_fide"], mc["cnn_test"]["n_attack"], "N1 seed %s" % s, problems)
        tr = n1_seeds[s].get("training", {})
        for k in ("restored_epoch", "epochs_run", "best_val_loss", "weights_sha256"):
            if k not in tr:
                fail("N1 seed %s: training field %s missing" % (s, k))
        rows.append(row(row="N1-seed%s" % s, block="N1",
                        model="LivenessNet, new training, seed %s (CPU; restored epoch %s of %s, min val_loss %.6f)"
                              % (s, tr["restored_epoch"], tr["epochs_run"], tr["best_val_loss"]),
                        training_data=T["train"] + " (cache frames)", early_stopping=T["es"], test_set=T["test_3000"],
                        extractor=EXTRACTOR_CACHE + " (cache frames)", threshold="0.5",
                        threshold_origin="fixed a priori (configuration)",
                        n_bona_fide=m["n_bona_fide"], n_attack=m["n_attack"], rejected_bona_fide=m["rejected_bona_fide"],
                        accepted_attacks=m["accepted_attacks"], apcer=f4(m["apcer"]), bpcer=f4(m["bpcer"]),
                        acer=f4(m["acer"]), auroc=f4(m["auroc"]), run_id=n1["run_id"], transfer=TRANSFER_SESSION_ALL,
                        source=src("n1_aggregate")))

    # N2 parte 1: gli stessi checkpoint sui 491 delle sessioni 01-02 e sulla sessione 03 dei 9 soggetti
    if n2s["n1_study"] != n1["run_id"]:
        problems.append("N2 sessions: n1_study %r is not the N1 run %r" % (n2s["n1_study"], n1["run_id"]))
    if [str(s) for s in n2s["train_seeds"]] != seeds:
        problems.append("N2 sessions: seeds differ from N1")
    n2_seeds = seed_block(n2s, seeds, "N2 sessions")
    N2_SUBSETS = ("sessions_01_02_491", "sessions_01_02_fitting_subjects", "sessions_01_02_validation_subjects",
                  "session_03_9_subjects", "session_03_fitting_subjects", "session_03_validation_subjects", "session_03_3000")
    denominators = {}
    for s in seeds:
        ms = n2_seeds[s].get("metrics", {})
        sub = {k: metric_row(ms.get(k), "N2 seed %s %s" % (s, k)) for k in N2_SUBSETS}
        for k, m in sub.items():
            denominators.setdefault(k, (m["n_bona_fide"], m["n_attack"]))
            if denominators[k] != (m["n_bona_fide"], m["n_attack"]):
                problems.append("N2 sessions: denominators of %s differ between seeds" % k)
        for whole, parts in (("sessions_01_02_491", ("sessions_01_02_fitting_subjects", "sessions_01_02_validation_subjects")),
                             ("session_03_9_subjects", ("session_03_fitting_subjects", "session_03_validation_subjects"))):
            for field in ("n_bona_fide", "n_attack", "rejected_bona_fide", "accepted_attacks"):
                if sum(sub[k][field] for k in parts) != sub[whole][field]:
                    problems.append("N2 sessions seed %s: %s of %s is not the sum of its subject groups" % (s, field, whole))
        a, b, c = sub["sessions_01_02_491"], sub["session_03_9_subjects"], sub["session_03_3000"]
        ref = metric_row(n1_seeds[s]["metrics"]["test_3000"], "N1")
        if c != ref:
            problems.append("N2 sessions seed %s: session_03_3000 differs from N1 test_3000" % s)
        if n2_seeds[s].get("weights_sha256") != n1_seeds[s]["training"]["weights_sha256"]:
            problems.append("N2 sessions seed %s: weights SHA-256 differs from N1" % s)
        check_counts(a, target["n_bona_fide"], target["n_attack"], "N2 seed %s 491" % s, problems)
        T.setdefault("s03_9", "session 03 test frames of the same %d training subjects (%s + %s), inside nuaa_cnn_test"
                     % (target["n_subjects"], fmt(b["n_bona_fide"]), fmt(b["n_attack"])))
        rows.append(row(row="N2S-01-02-seed%s" % s, block="N2 sessions",
                        model="LivenessNet, N1 checkpoint seed %s" % s, training_data=T["train"] + " (cache frames)",
                        early_stopping=T["es"], test_set=T["target_491"],
                        extractor=EXTRACTOR_CACHE + " (frozen copy)", threshold="0.5",
                        threshold_origin="fixed a priori (N1)", n_bona_fide=a["n_bona_fide"], n_attack=a["n_attack"],
                        rejected_bona_fide=a["rejected_bona_fide"], accepted_attacks=a["accepted_attacks"],
                        apcer=f4(a["apcer"]), bpcer=f4(a["bpcer"]), acer=f4(a["acer"]), auroc=f4(a["auroc"]),
                        run_id=n2s["run_id"], transfer=TRANSFER_SAME, source=src("n2_sessions")))
        rows.append(row(row="N2S-03-seed%s" % s, block="N2 sessions",
                        model="LivenessNet, N1 checkpoint seed %s" % s, training_data=T["train"] + " (cache frames)",
                        early_stopping=T["es"],
                        test_set=T["s03_9"],
                        extractor=EXTRACTOR_CACHE + " (cache frames)", threshold="0.5",
                        threshold_origin="fixed a priori (N1)", n_bona_fide=b["n_bona_fide"], n_attack=b["n_attack"],
                        rejected_bona_fide=b["rejected_bona_fide"], accepted_attacks=b["accepted_attacks"],
                        apcer=f4(b["apcer"]), bpcer=f4(b["bpcer"]), acer=f4(b["acer"]), auroc=f4(b["auroc"]),
                        run_id=n2s["run_id"], transfer=TRANSFER_SESSION_9, source=src("n2_sessions")))

    # N2 parte 2: il combinato storico sul test N1, due varianti
    for s in seeds:
        ref = n2c["n1_reference"].get("seeds", {}).get(s, {}).get("test_3000")
        if not isinstance(ref, dict):
            problems.append("N2 combined: N1 reference for seed %s missing" % s)
            continue
        mine = metric_row(n1_seeds[s]["metrics"]["test_3000"], "N1")
        if {k: ref.get(k) for k in mine} != mine:
            problems.append("N2 combined: N1 reference seed %s differs from N1 test_3000" % s)
    if not isinstance(n2c["model"], dict) or "sha256" not in n2c["model"]:
        fail("N2 combined: model sha256 missing")
    for v, label in VARIANTS.items():
        if v not in n2c["variants"]:
            problems.append("N2 combined: variant %s missing" % v)
            continue
        m = metric_row(n2c["variants"][v].get("metrics", {}).get("test_3000"), "N2 combined %s" % v)
        check_counts(m, mc["cnn_test"]["n_bona_fide"], mc["cnn_test"]["n_attack"], "N2 combined %s" % v, problems)
        rows.append(row(row="N2C-%s" % v, block="N2 combined",
                        model="AttackNet V2.2 combined (coordinator's checkpoint, SHA-256 %s…)" % n2c["model"]["sha256"][:8],
                        training_data=TRAIN_COMB, early_stopping="historical (not reproduced here)", test_set=T["test_3000"],
                        extractor=label, preprocessing=PREP_HIST, threshold="0.5",
                        threshold_origin="fixed before the test (historical argmax rule)", n_bona_fide=m["n_bona_fide"],
                        n_attack=m["n_attack"], rejected_bona_fide=m["rejected_bona_fide"],
                        accepted_attacks=m["accepted_attacks"], apcer=f4(m["apcer"]), bpcer=f4(m["bpcer"]),
                        acer=f4(m["acer"]), auroc=f4(m["auroc"]), run_id=n2c["run_id"], transfer=TRANSFER_DATASET,
                        source=src("n2_combined")))

    # Confronto del ritaglio: AttackNet V2.1, due estrattori, le 300 di C1
    cref = metric_row(crop["c1_reference"].get("metrics"), "crop c1_reference")
    rm = m1["attacknet_v2_1__nuaa"]
    if (str(cref["n_bona_fide"]), str(cref["n_attack"]), str(cref["rejected_bona_fide"]), str(cref["accepted_attacks"])) != \
            (rm["n_bona_fide"], rm["n_attack"], rm["rejected bona fide@0.5"], rm["accepted attacks@0.5"]):
        problems.append("crop: C1 reference counts differ from the M1/C1 table")
    for k, km in (("apcer", "APCER@0.5"), ("bpcer", "BPCER@0.5"), ("acer", "ACER@0.5"), ("auroc", "test AUC")):
        if f4(cref[k]) != rm[km]:
            problems.append("crop: C1 reference %s %s differs from the table %s" % (k, f4(cref[k]), rm[km]))
    if crop["c1_reference"].get("run_id") != C1_RUN:
        problems.append("crop: c1_reference run %r is not %s" % (crop["c1_reference"].get("run_id"), C1_RUN))
    crop_metrics = {}
    for e, label in EXTRACTOR_LABELS.items():
        if e not in crop["metrics"]:
            problems.append("crop: extractor %s missing" % e)
            continue
        m = metric_row(crop["metrics"][e], "crop %s" % e)
        crop_metrics[e] = m
        check_counts(m, cref["n_bona_fide"], cref["n_attack"], "crop %s" % e, problems)
        rows.append(row(row="CROP-%s" % e, block="C1 crop check",
                        model="AttackNet V2.1 (C1 checkpoint of 2026-09-12), CPU batch 16", training_data=T["train"],
                        early_stopping=T["es"], test_set=T["test_c1"], extractor=label,
                        threshold="0.5", threshold_origin=THR_FIXED, n_bona_fide=m["n_bona_fide"], n_attack=m["n_attack"],
                        rejected_bona_fide=m["rejected_bona_fide"], accepted_attacks=m["accepted_attacks"],
                        apcer=f4(m["apcer"]), bpcer=f4(m["bpcer"]), acer=f4(m["acer"]), auroc=f4(m["auroc"]),
                        run_id=crop["run_id"], transfer=TRANSFER_SESSION_ALL, source=src("crop")))
    if EXTRACTOR_C1 in crop_metrics:
        mc1 = crop_metrics[EXTRACTOR_C1]
        if {k: f4(mc1[k]) for k in RATES} != {k: f4(cref[k]) for k in RATES} or \
                (mc1["rejected_bona_fide"], mc1["accepted_attacks"]) != (cref["rejected_bona_fide"], cref["accepted_attacks"]):
            problems.append("crop: the C1 extractor row differs from the C1 reference (counts or rates at four decimals)")
    paired = crop["paired"]
    for k in ("n", "max_abs_diff", "median_abs_diff", "mean_abs_diff", "mean_signed_diff_cache_minus_c1", "abs_diff_bins",
              "decision_changes_c1_to_cache", "frames_equal"):
        if k not in paired:
            fail("crop: paired field %s missing" % k)

    # Annessi: righe cross-dataset e di contesto della tabella C1
    cross = []
    for name, r in c1.items():
        mm = re.fullmatch(r"(livenessnet|attacknet_v1|attacknet_v2_1|attacknet_v2_2)__(casia_fasd|celeba_spoof|synthaspoof)(-pooled)?", name)
        if not mm:
            continue
        cross.append({"analyzer": name, "model": ARCH_NAMES[mm.group(1)], "source_dataset": CROSS_SOURCES[mm.group(2)],
                      "protocol": ("pooled 80/20 (random over images)" if mm.group(3)
                                   else "own split by synthetic identity" if mm.group(2) == "synthaspoof"
                                   else "official split of the source dataset"),
                      "n_bona_fide": r["n_bona_fide"], "n_attack": r["n_attack"], "apcer": r["apcer"], "bpcer": r["bpcer"],
                      "acer": r["acer"], "eer": r["eer"], "auroc": r["auc"]})
    if len(cross) != 16:
        problems.append("C1 table: %d cross-dataset CNN rows found, expected 16" % len(cross))
    context = []
    for name, (label, trained) in CONTEXT_ROWS.items():
        r = c1.get(name)
        if r is None:
            problems.append("C1 table: context row %s missing" % name)
            continue
        context.append({"analyzer": name, "model": label, "trained_on": trained, "n_bona_fide": r["n_bona_fide"],
                        "n_attack": r["n_attack"], "apcer": r["apcer"], "bpcer": r["bpcer"], "acer": r["acer"],
                        "eer": r["eer"], "auroc": r["auc"]})

    # Sensibilità ai soggetti (N1 parte 1)
    subj = []
    if n1s["accepted_m1_table_reproduced"].get("result") is not True:
        problems.append("N1 subjects: the accepted M1 table is not reproduced")
    for a in NUAA_CNNS:
        d = n1s["analyzers"].get(a)
        if not isinstance(d, dict) or "leave_one_subject_out" not in d:
            continue  # già segnalato sopra
        lo = d["leave_one_subject_out"]
        for k in ("delta_min", "delta_max", "n_sign_changes", "n_exclusions", "n_na"):
            if k not in lo:
                fail("N1 subjects: %s leave_one_subject_out has no field %s" % (a, k))
        subj.append({"analyzer": a, "model": CNN_NAMES[a], "m1_threshold": m1[a]["dev threshold"],
                     "acer_at_0_5": d["full"]["at_0_5"]["acer"], "acer_at_m1": d["full"]["at_m1"]["acer"],
                     "delta_acer": d["full"]["delta_acer"], "loso_delta_min": lo["delta_min"],
                     "loso_delta_max": lo["delta_max"], "sign_changes": "%d/%d" % (lo["n_sign_changes"], lo["n_exclusions"]),
                     "n_na": lo["n_na"]})

    # Run a fold di soggetti (PR #18): 12 righe fold x seed e 3 righe aggregate per seed, annesso
    folds = folds_build(fagg, frec, fsch, mc, problems)
    T.update(folds["texts"])
    folds_src = "%s; %s; %s" % (src("folds_aggregate"), src("folds_record"), src("folds_schema"))
    for r in folds["rows"]:
        r["source"] = folds_src
    rows.extend(folds["rows"])

    # Confronto con N1 sugli stessi 3.000 identificativi (righe FOLDSN1; le righe N1 restano quelle già in tabella)
    cmp_agg = read_json(paths["folds_n1_compare"], COMPARE_FIELDS, clean=False)
    compare_trial_check(cmp_agg)
    run_id_check(cmp_agg, "folds vs N1")
    n1_rows = {r["row"].split("seed")[1]: r for r in rows if r["block"] == "N1"}
    compare = compare_rows(cmp_agg, {"n1": n1, "n1_seeds": n1_seeds, "n1_rows": n1_rows, "mc": mc, "m2": m2, "folds": folds,
                                     "fsch": fsch, "T": T, "src": src("folds_n1_compare")}, problems)
    rows.extend(compare["rows"])
    sources = {k: {"path": SOURCES[k], "sha256": hashes[k]} for k in SOURCES}

    if problems:
        fail("consistency problems:\n  - " + "\n  - ".join(problems))

    doc = {
        "schema": SCHEMA,
        "what": "unified table of the accepted NUAA results (C1, M1, N1, N2, C1 crop check, subject-held-out folds run), rebuilt from the public aggregates of research/pad-2026; no restricted data, no inference, no new number",
        "sources": sources,
        "runs": dict({"C1": C1_RUN, "M1": M1_RUN, "N1": n1["run_id"], "N1_subjects": n1s["run_id"], "N2_sessions": n2s["run_id"],
                      "N2_combined": n2c["run_id"], "C1_crop_check": crop["run_id"], "Fold_schema": fsch["run_id"],
                      "Folds_run": fagg["run_id"], "Folds_vs_N1": cmp_agg["run_id"]}),
        "decision": "attack if score_attack > threshold (tie = bona fide); label 0 bona fide, 1 attack; score = probability of attack",
        "texts": T,
        "manifest_counts": mc,
        "columns": COLUMNS,
        "rows": rows,
        "cross_dataset_rows": cross,
        "context_rows": context,
        "subject_sensitivity": subj,
        "crop_paired": {k: paired[k] for k in ("n", "max_abs_diff", "median_abs_diff", "mean_abs_diff",
                                                "mean_signed_diff_cache_minus_c1", "abs_diff_bins",
                                                "decision_changes_c1_to_cache", "frames_equal")},
        "n2_sessions_subsets": {k: {s: metric_row(n2_seeds[s]["metrics"].get(k), k) for s in seeds}
                                for k in ("sessions_01_02_fitting_subjects", "sessions_01_02_validation_subjects",
                                          "session_03_fitting_subjects", "session_03_3000")},
        "n1_c1_subset": {s: metric_row(n1_seeds[s]["metrics"].get("c1_300"), "N1 c1_300") for s in seeds},
        "folds_run": dict(folds["info"], annex=folds["annex"]),
        "folds_n1_compare": dict(compare["info"], annex=compare["annex"]),
    }
    return doc


# --- scrittura ----------------------------------------------------------------------------------------------------

def md_table(cols: list, rows: list, labels: dict = None, empty: str = "") -> str:
    """Tabella Markdown; con empty, le celle vuote (colonne che non si applicano alla riga) diventano quel segno."""
    labels = labels or {}
    head = "| " + " | ".join(labels.get(c, c) for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join(str(r.get(c, "")) if str(r.get(c, "")) != "" else empty for c in cols) + " |\n"
                   for r in rows)
    return head + body


def summary_md(doc: dict) -> str:
    cols = ["row", "block", "model", "training_data", "early_stopping", "test_set", "unit", "extractor", "preprocessing",
            "threshold", "threshold_origin", "n_bona_fide", "n_attack", "rejected_bona_fide", "accepted_attacks", "apcer",
            "bpcer", "acer", "auroc", "threshold_dev", "rejected_bona_fide_dev", "accepted_attacks_dev", "apcer_dev", "bpcer_dev",
            "acer_dev", "bootstrap", "run_id", "transfer"]
    labels = {"n_bona_fide": "n bona fide", "n_attack": "n attack", "rejected_bona_fide": "rejected bona fide",
              "accepted_attacks": "accepted attacks", "apcer": "APCER", "bpcer": "BPCER", "acer": "ACER", "auroc": "AUROC",
              "run_id": "run", "training_data": "training data (fitting)", "early_stopping": "early stopping",
              "test_set": "test set", "threshold_origin": "threshold origin", "transfer": "same session or transfer",
              "threshold_dev": "dev threshold", "rejected_bona_fide_dev": "rejected bona fide (dev)",
              "accepted_attacks_dev": "accepted attacks (dev)", "apcer_dev": "APCER dev", "bpcer_dev": "BPCER dev",
              "acer_dev": "ACER dev", "bootstrap": "bootstrap over subjects, percentiles 2.5-97.5"}
    out = io.StringIO()
    out.write("# Unified table of the accepted NUAA results (C1, M1, N1, N2, C1 crop check, subject-held-out folds run)\n\n")
    out.write("Built by `nuaa_synthesis.py build` from the public aggregates merged in `research/pad-2026` "
              "(sources and SHA-256 in `summary_table.json`); no restricted data, no inference, no new number. "
              "%s. APCER = accepted attacks / attacks, BPCER = rejected bona fide / bona fide, ACER = their mean, "
              "AUROC threshold-free (the column named `auc` in the C1 table and `test AUC` in the M1 table: the same Mann-Whitney statistic). "
              "Rates as printed in the C1 and M1 tables (four decimals; M1 thresholds six) or rounded to four decimals "
              "from the exact values of the JSON aggregates (N1, N2, crop check, folds run). "
              "The columns from `dev threshold` to `bootstrap` apply to the folds-run rows only (the dev threshold next to "
              "0.5; the bootstrap, 1,000 resamples of the held-out subjects, for the rows pooled over the four folds); "
              "\u2014 elsewhere. "
              "Runs: %s.\n\n" % (doc["decision"], ", ".join("%s `%s`" % (k, v) for k, v in doc["runs"].items())))
    out.write(md_table(cols, doc["rows"], labels, empty="\u2014"))
    out.write("\nSource of every row: `summary_table.csv` (column `source`) and `summary_table.json`.\n")
    return out.getvalue()


SHORT_BLOCK = {"C1": "C1", "M1": "M1", "N1": "N1", "N2 sessions": "N2-S", "N2 combined": "N2-C", "C1 crop check": "CROP",
               FOLDS_BLOCK: "FOLDS", COMPARE_BLOCK: "FOLDS-N1"}
TRANSFER_CODES = {TRANSFER_SAME: "same", TRANSFER_SESSION_ALL: "session", TRANSFER_SESSION_9: "session/9",
                  TRANSFER_DATASET: "dataset", TRANSFER_FOLDS: "subject-held-out (sessions represented)",
                  TRANSFER_FOLDS_N1: "subject-held-out, N1 test frames"}


def short_model(r: dict) -> str:
    """Nome breve del modello per la tabella compatta (il testo completo resta nella tabella estesa)."""
    m = r["model"]
    if r["block"] in ("C1", "M1"):
        return m.split(" (")[0] + " (C1 ckpt 12/09)"
    if r["block"] == "N1":
        return "LivenessNet, new CPU training, seed " + r["row"].split("seed")[1]
    if r["block"] == "N2 sessions":
        return "LivenessNet, N1 ckpt seed " + r["row"].split("seed")[1]
    if r["block"] == "N2 combined":
        return "AttackNet V2.2 combined (coordinator's ckpt)"
    if r["block"] == "C1 crop check":
        return "AttackNet V2.1 (C1 ckpt 12/09), CPU batch 16"
    if r["block"] == FOLDS_BLOCK:
        mm = re.fullmatch(r"FOLDS-(?:f(\d+)|pooled)-seed(\d+)", r["row"])
        if mm and mm.group(1):
            return "LivenessNet, fold %s, seed %s" % (mm.group(1), mm.group(2))
        if mm:
            return "LivenessNet, 4 fold models, seed %s (pooled)" % mm.group(2)
    if r["block"] == COMPARE_BLOCK:
        return "LivenessNet, 4 fold models, seed %s (N1 test frames)" % r["row"].split("seed")[1]
    fail("compact table: block %r" % r["block"])


def compact_legend(doc: dict) -> dict:
    T = doc["texts"]
    folds = sorted(int(k[len("folds_test_"):]) for k in T if re.fullmatch(r"folds_test_\d+", k))
    test = {"S-300": T["test_c1"], "S-3000": T["test_3000"], "S-491": T["target_491"], "S-03/9": T["s03_9"]}
    test.update({"S-FOLD-%d" % k: T["folds_test_%d" % k] for k in folds})
    test["S-FOLD-ALL"] = T["folds_test_all"]
    return {
        "training": {"T-NUAA": T["train"] + "; early stopping on " + T["es"],
                     "T-COMB": TRAIN_COMB + "; historical early stopping",
                     "T-FOLD": T["folds_legend"] + "; the dev fold is the next held-out fold and drives early stopping, the "
                               "learning-rate schedule and the EER threshold; frames drawn with the deterministic spread rule"},
        "test": test,
        "extractor": {"E-C1": "c1_a80f6f9: the C1 evaluation extractor (RetinaFace box 1 px narrower and shorter than the cache)",
                      "E-CACHE": "cache_0ac7143: the training-cache extractor (cache frames, or the frozen copy on new frames)",
                      "E-WHOLE": VARIANTS["whole_image"], "E-CROP": VARIANTS["face_crop"]},
        "preprocessing": {"P-REF": PREP_REF, "P-HIST": PREP_HIST},
        "threshold": {"0.5 (fixed)": "fixed a priori (C1, N1, N2, crop check, folds run)",
                      "(dev EER)": "M1: " + T["thr_dev"] + ", frozen before any test score was read",
                      "(EER, fold dev)": "folds run, column \"dev thr.\": " + THR_FOLDS_ORIGIN.split("; dev threshold: ")[1]
                                         + " (\"" + THR_DEV_POOLED + "\" in the pooled rows: each frame at the threshold of its fold model)"},
        # il codice del confronto con N1 compare solo quando ci sono le sue righe (sorgente facoltativa)
        "transfer": {v: k for k, v in TRANSFER_CODES.items()
                     if k != TRANSFER_FOLDS_N1 or any(r["transfer"] == k for r in doc["rows"])},
    }


def compact_rows(doc: dict) -> list:
    T = doc["texts"]
    out = []
    tests = {v: k for k, v in compact_legend(doc)["test"].items()}
    for r in doc["rows"]:
        training = ("T-COMB" if r["training_data"] == TRAIN_COMB else "T-FOLD" if r["training_data"] == T.get("folds_train")
                    else "T-NUAA")
        test = tests.get(r["test_set"])
        if test is None:
            fail("compact table: test set %r" % r["test_set"])
        e = r["extractor"]
        extractor = ("E-C1" if e.startswith(EXTRACTOR_C1) else "E-CACHE" if e.startswith(EXTRACTOR_CACHE)
                     else "E-WHOLE" if e == VARIANTS["whole_image"] else "E-CROP" if e == VARIANTS["face_crop"]
                     else fail("compact table: extractor %r" % e))
        transfer = TRANSFER_CODES[r["transfer"]]
        thr = r["threshold"] + (" (dev EER)" if r["block"] == "M1" else " (fixed)")
        prep = {PREP_REF: "P-REF", PREP_HIST: "P-HIST"}[r["preprocessing"]]
        dev = r["threshold_dev"] != ""
        boot = r["bootstrap"]
        if boot.startswith("ACER 0.5 "):  # «ACER 0.5 a-b; ACER dev c-d; AUROC e-f» -> «a-b; c-d; e-f»
            boot = "; ".join(part.split(" ")[-1] for part in boot.split("; "))
        out.append({"block": SHORT_BLOCK[r["block"]], "model": short_model(r), "training": training, "test": test,
                    "extractor": extractor, "preprocessing": prep, "threshold": thr,
                    "errors": "%s / %s" % (r["rejected_bona_fide"], r["accepted_attacks"]),
                    "apcer": r["apcer"], "bpcer": r["bpcer"], "acer": r["acer"], "auroc": r["auroc"],
                    "threshold_dev": (r["threshold_dev"] + " (EER, fold dev)") if dev else "",
                    "errors_dev": "%s / %s" % (r["rejected_bona_fide_dev"], r["accepted_attacks_dev"]) if dev else "",
                    "apcer_dev": r["apcer_dev"], "bpcer_dev": r["bpcer_dev"], "acer_dev": r["acer_dev"], "bootstrap": boot,
                    "run_id": r["run_id"], "transfer": transfer})
    return out


def compact_md(doc: dict) -> str:
    cols = ["block", "model", "training", "test", "extractor", "preprocessing", "threshold", "errors", "apcer", "bpcer", "acer",
            "auroc", "threshold_dev", "errors_dev", "apcer_dev", "bpcer_dev", "acer_dev", "bootstrap", "run_id", "transfer"]
    labels = {"errors": "rejected bf / accepted att.", "apcer": "APCER", "bpcer": "BPCER", "acer": "ACER",
              "auroc": "AUROC", "run_id": "run", "transfer": "transfer", "test": "test set", "preprocessing": "prep.",
              "threshold_dev": "dev thr.", "errors_dev": "rejected bf / accepted att. (dev)", "apcer_dev": "APCER dev",
              "bpcer_dev": "BPCER dev", "acer_dev": "ACER dev", "bootstrap": "bootstrap (ACER 0.5; ACER dev; AUROC)"}
    out = io.StringIO()
    out.write("# Unified table, compact form (codes in the legend below; full text, row ids and sources in `summary_table.md`)\n\n")
    out.write(md_table(cols, compact_rows(doc), labels, empty="\u2014"))
    out.write("\nLegend. Blocks: C1, M1, N1 (part 2), N2-S (N2 part 1, sessions), N2-C (N2 part 2, combined model), CROP (C1 crop check), "
              "FOLDS (subject-held-out folds run: one row per held-out fold and seed, stratum all, and one row per seed pooled over the four folds); "
              "ckpt = checkpoint; \"rejected bf / accepted att.\" = rejected bona fide / accepted attacks, the numerators of BPCER and APCER. "
              "The columns from \"dev thr.\" to \"bootstrap\" apply to the FOLDS rows only (\u2014 elsewhere): the errors and rates "
              "at the dev threshold of each model, and for the pooled rows the 2.5-97.5 percentiles of 1,000 bootstrap resamples "
              "of the held-out subjects (seed 0) for ACER at 0.5, ACER at the dev threshold and AUROC. "
              "Unit = image (frame) in every row; %s; APCER = accepted attacks / attacks, BPCER = rejected bona fide / bona fide, "
              "ACER = their mean, AUROC threshold-free (`auc` in the C1 table, `test AUC` in the M1 table).\n\n" % doc["decision"])
    for group, items in compact_legend(doc).items():
        for code, text in items.items():
            out.write("- `%s` (%s): %s\n" % (code, group, text))
    out.write("\nRuns: %s.\n" % ", ".join("%s `%s`" % (k, v) for k, v in doc["runs"].items()))
    return out.getvalue()


def cross_md(doc: dict) -> str:
    cols = ["analyzer", "model", "source_dataset", "protocol", "n_bona_fide", "n_attack", "apcer", "bpcer", "acer", "eer", "auroc"]
    labels = {"source_dataset": "trained on", "n_bona_fide": "n bona fide", "n_attack": "n attack", "apcer": "APCER",
              "bpcer": "BPCER", "acer": "ACER", "eer": "EER", "auroc": "AUROC"}
    return ("# Annex: the 16 cross-dataset CNN rows of the C1 table on the same 300 NUAA images (threshold 0.5, run `%s`)\n\n"
            "Checkpoints of 2026-09-12 (CASIA-FASD, CelebA-Spoof) and 2026-09-25 (SynthASpoof) applied to NUAA without any "
            "adaptation (dataset transfer, fixed threshold 0.5, no source-dev threshold). Source: `%s`.\n\n" % (C1_RUN, SOURCES["c1_table"])
            + md_table(cols, doc["cross_dataset_rows"], labels))


def context_md(doc: dict) -> str:
    cols = ["analyzer", "model", "trained_on", "n_bona_fide", "n_attack", "apcer", "bpcer", "acer", "eer", "auroc"]
    labels = {"trained_on": "trained on", "n_bona_fide": "n bona fide", "n_attack": "n attack", "apcer": "APCER",
              "bpcer": "BPCER", "acer": "ACER", "eer": "EER", "auroc": "AUROC"}
    return ("# Annex: the other analyzers of the C1 table on the same 300 NUAA images (threshold 0.5, run `%s`)\n\n"
            "Context only: third-party pre-trained models with no NUAA training in our experiment (whether NUAA appears in "
            "their pretraining data is not verified here: it would need a provenance check; \"zero-shot\" names the C1 analyzer, "
            "classification by text prompts with no classifier fitted on NUAA, and says nothing about the pretraining data), "
            "hand-crafted rules, linear probes "
            "on third-party encoders (same caveat for the encoders) and SVMs on hand-crafted features, the probes and the SVMs "
            "fitted on all 3,000 sampled NUAA training images (the dev images included; no early stopping involved). "
            "Source: `%s`.\n\n" % (C1_RUN, SOURCES["c1_table"])
            + md_table(cols, doc["context_rows"], labels))


def subjects_md(doc: dict) -> str:
    cols = ["model", "m1_threshold", "acer_at_0_5", "acer_at_m1", "delta_acer", "loso_delta_min", "loso_delta_max", "sign_changes", "n_na"]
    labels = {"m1_threshold": "M1 threshold", "acer_at_0_5": "ACER@0.5", "acer_at_m1": "ACER@M1",
              "delta_acer": "ΔACER (M1 − 0.5)", "loso_delta_min": "leave-one-subject-out ΔACER min",
              "loso_delta_max": "max", "sign_changes": "sign changes", "n_na": "NA"}
    return ("# Annex: sensitivity of the M1 result to the 16 test subjects (N1 part 1, run `%s`)\n\n"
            "Existing scores only; one subject at a time excluded from the computation; thresholds unchanged. "
            "Source: `%s`.\n\n" % (doc["runs"]["N1_subjects"], SOURCES["n1_subjects"])
            + md_table(cols, doc["subject_sensitivity"], labels))


def folds_md(doc: dict) -> str:
    """Annesso del run a fold: tutti gli strati per fold e seed, mediane e intervalli, aggregato per seed, modelli."""
    f, a = doc["folds_run"], doc["folds_run"]["annex"]

    def table(head, rows):
        return ("| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n"
                + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows))
    stratum = ["rejected bf / accepted att. (05)", "BPCER 05", "APCER 05", "ACER 05", "rejected bf / accepted att. (dev)",
               "BPCER dev", "APCER dev", "ACER dev"]
    summary = ["ACER 05: median (min-max)", "ACER dev: median (min-max)", "AUROC: median (min-max)"]
    b = f["bootstrap"]
    return ("# Annex: subject-held-out folds run, LivenessNet, %d subject folds x %d seeds (run `%s`)\n\n"
            "Copied from the public aggregate of the run (`%s`, every count and rate checked by `nuaa_synthesis.py`: "
            "denominators against the fold schema `%s`, stratum all against the sum of the sessions, pooled rows against "
            "the sum of the folds, medians and ranges against the per-seed values). Training on the CPU with the N1 recipe; "
            "`05` = the fixed threshold 0.5, `dev` = the EER threshold of the dev fold only, frozen before any test scoring; "
            "AUROC threshold-free; %s; NA where a class is absent. Frames of one subject and session are correlated: n counts "
            "frames, not independent units. Bootstrap: %d resamples with replacement of the held-out subjects of the stratum "
            "(pooled over the folds, per seed), seed %d, percentiles %s-%s, no interval with fewer than %d subjects. The pooled "
            "rows are scored by %d models with %d dev thresholds.\n\n"
            % (f["folds"], len(f["seeds"]), f["run_id"], SOURCES["folds_aggregate"], f["schema_run"], doc["decision"],
               b["resamples"], b["seed"], b["percentiles"][0], b["percentiles"][1], b["min_subjects"], f["folds"], f["folds"])
            + "## Per fold and seed (held-out subjects, by session stratum)\n\n"
            + table(["fold", "seed", "stratum", "n bona fide", "n attack", "subjects"] + stratum + ["AUROC"], a["per_fold_seed"])
            + "\n## Per fold: median and range over the seeds\n\n"
            + table(["fold", "stratum"] + summary, a["per_fold_summary"])
            + "\n## Pooled over the %d folds, per seed (every subject held out once)\n\n" % f["folds"]
            + table(["seed", "stratum", "n bona fide", "n attack", "subjects", "rejected bf / accepted att. (05)", "BPCER 05",
                     "APCER 05", "ACER 05 [bootstrap]", "rejected bf / accepted att. (dev)", "BPCER dev", "APCER dev",
                     "ACER dev [bootstrap]", "AUROC [bootstrap]"], a["pooled"])
            + "\n## Pooled: median and range over the seeds\n\n"
            + table(["stratum"] + summary, a["pooled_summary"])
            + "\n## Models: training and dev thresholds\n\n"
            + table(["model", "n fitting", "n dev", "epochs", "restored epoch", "restored by", "min val_loss", "seconds",
                     "dev threshold", "dev BPCER at threshold", "dev APCER at threshold"], a["training"]))


def compare_md(doc: dict) -> str:
    """Annesso del confronto con N1 sugli stessi 3.000 frame: copiato dall'aggregato pubblico del confronto."""
    c, a = doc["folds_n1_compare"], doc["folds_n1_compare"]["annex"]

    def table(head, rows):
        return ("| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n"
                + "".join("| " + " | ".join(str(x) for x in r) + " |\n" for r in rows))
    b = doc["folds_run"]["bootstrap"]
    return ("# Annex: folds run versus N1 on the same 3,000 frames (descriptive, run `%s`)\n\n"
            "Copied from the public aggregate of the comparison (`%s`, checked by `nuaa_synthesis.py`: N1 values equal to the "
            "N1 rows of the unified table, folds counts inside the pooled session 03 of the folds run, differences = folds "
            "minus N1, paired decisions and subject counts consistent with the counts, medians and ranges equal to the "
            "per-seed values, control values reproduced). The 3,000 frames are those of `nuaa_cnn_test` (1,500 bona fide, "
            "1,500 attacks, 16 subjects, session 03), on which N1 was tested; they are a subset of the pooled session 03 of "
            "the folds run, so the folds values here differ from the pooled rows of `folds_run_table.md`. Descriptive and "
            "post hoc: existing scores and thresholds already frozen; no training, no inference, no new threshold. N1 has "
            "only the fixed threshold 0.5: the dev values compare the folds run at its frozen dev thresholds with N1 at "
            "0.5. %s. The AUROC of the folds run pools the scores of four models (one per fold). Training roles differ: "
            "N1 was trained on 9 of the 16 subjects (sessions 01-02); in the folds run no frame is scored by a model that "
            "saw its subject.%s Bootstrap over the 16 subjects, paired: %d resamples, seed %d, percentiles %s-%s; recipe: %s "
            "Indicative; no causal reading.\n\n"
            % (c["run_id"], SOURCES["folds_n1_compare"], doc["decision"],
               (" N1 AUROC is below 0.5 on these frames (%s-%s, as published by N1); the AUROC difference includes that."
                % (f4(min(c["n1_auroc"])), f4(max(c["n1_auroc"])))) if max(c["n1_auroc"]) < 0.5 else "",
               b["resamples"], b["seed"], b["percentiles"][0], b["percentiles"][1], c["bootstrap_recipe"].rstrip(".") + ".")
            + "## Per seed\n\n"
            + table(["seed", "experiment", "rej. bf / acc. att.", "BPCER", "APCER", "ACER [bootstrap]", "AUROC [bootstrap]"], a["per_seed"])
            + "\n## Differences, folds minus N1 (paired bootstrap over subjects)\n\n"
            + table(["seed", "ACER folds 0.5 - N1 0.5", "ACER folds dev - N1 0.5", "AUROC folds - N1"], a["differences"])
            + "\n## Median and range over the three seeds\n\n"
            + table(["measure", "median", "min", "max"], a["summary"])
            + "\n## Paired decisions per frame\n\n"
            "Per class: frames correct in both experiments, only in N1, only in the folds run, in neither; N1 always at 0.5. "
            "Changed decisions = frames of either class on which the two decisions differ, one count per seed and "
            "threshold.\n\n"
            + table(["seed", "folds at", "bona fide: both ok / N1 only ok / folds only ok / both wrong", "attack: same cells",
                     "changed decisions (both classes)"], a["paired"])
            + "\n## Subjects (counts only)\n\n"
            "Subjects where the folds run is lower, equal or higher than N1 at 0.5; error rate = wrong frames / frames, ACER "
            "only for the %s subjects with both classes.\n\n" % c["with_both_classes"]
            + table(["seed", "folds at", "measure", "folds lower", "equal", "folds higher", "not defined"], a["subjects"])
            + "\n## Control values of the coordinator\n\n"
            + table(["measure", "seed", "control (as given)", "computed"], a["control"])
            + "\nComparison: %s.\n" % c["control_comparison"].rstrip("."))


def csv_text(doc: dict) -> str:
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in doc["rows"]:
        w.writerow(r)
    return out.getvalue()


def outputs(doc: dict) -> dict:
    return {
        "summary_table.md": summary_md(doc),
        "summary_table.csv": csv_text(doc),
        "summary_table.json": json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
        "summary_table_compact.md": compact_md(doc),
        "cross_dataset_table.md": cross_md(doc),
        "context_rows_table.md": context_md(doc),
        "subject_sensitivity_table.md": subjects_md(doc),
        "folds_run_table.md": folds_md(doc),
        "n1_same_ids_table.md": compare_md(doc),
    }


def cmd_build(a) -> int:
    doc = build(Path(a.research))
    files = outputs(doc)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        with open(out / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    print("BUILD: %d rows, %d cross-dataset rows, %d context rows, %d subject rows, %d folds-run stratum rows -> %s" %
          (len(doc["rows"]), len(doc["cross_dataset_rows"]), len(doc["context_rows"]), len(doc["subject_sensitivity"]),
           len(doc["folds_run"]["annex"]["per_fold_seed"]), out))
    print("RESULT: PASS")
    return 0


def cmd_check(a) -> int:
    doc = build(Path(a.research))
    files = outputs(doc)
    out = Path(a.out)
    bad = []
    for name, text in files.items():
        p = out / name
        if not p.is_file():
            bad.append("%s: missing" % name)
        elif p.read_bytes() != text.encode("utf-8"):
            bad.append("%s: differs from the rebuilt table (byte comparison)" % name)
    if a.synthesis:
        text = Path(a.synthesis).read_text(encoding="utf-8")
        lines = [l for l in compact_md(doc).splitlines() if l.startswith("|")]
        present = set(text.splitlines())
        missing = [l for l in lines if l not in present]
        if missing:
            bad.append("%s: %d of %d lines of the compact table are not in the document verbatim (first: %s)"
                       % (Path(a.synthesis).name, len(missing), len(lines), missing[0][:60]))
    for b in bad:
        print("PROBLEM: " + b)
    print("CHECK: %d files%s, %d problems" % (len(files), " + synthesis" if a.synthesis else "", len(bad)))
    print("RESULT: %s" % ("PASS" if not bad else "FAIL"))
    return 0 if not bad else 1


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("build", "check"):
        s = sub.add_parser(name)
        s.add_argument("--research", required=True, help="the research/ folder of the group repository")
        s.add_argument("--out", required=True, help="the tables folder")
    sub.choices["check"].add_argument("--synthesis", default=None,
                                      help="SYNTHESIS.md: every line of the compact table must appear in it verbatim")
    return p


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    try:
        return cmd_build(a) if a.cmd == "build" else cmd_check(a)
    except SynthesisError as e:
        print("ERROR: %s" % e)
    except (KeyError, TypeError, ValueError, OSError) as e:
        print("ERROR: a source has an unexpected structure or cannot be read: %s: %s" % (type(e).__name__, e))
    print("RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
