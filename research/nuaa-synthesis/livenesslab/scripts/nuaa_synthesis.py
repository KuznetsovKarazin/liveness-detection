"""
Sintesi NUAA (C1, M1, N1, N2, confronto del ritaglio, run a fold di soggetti, run di ablazione delle sessioni con pazienza
4 e 12 e loro confronto): ricostruisce la tabella unica dei risultati accettati leggendo
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
  nuaa-session-ablation-run/tables/aggregate.json, run_record.json, SHA256SUMS e run/status.json
                                                           run di ablazione delle sessioni (PR #21, pazienza 4): bracci
                                                           B, A*, B' x 4 fold x 3 seed, strati, aggregati per braccio e
                                                           seed, bootstrap, confronti e verdetti della regola di lettura
  nuaa-patience-verification/tables/aggregate.json, run_record.json, SHA256SUMS e run/status.json
                                                           run di verifica con pazienza 12 (PR #23), stesso disegno, con
                                                           il blocco amendment
  nuaa-patience-verification/comparison/aggregate.json, comparison.csv, curves.csv, SHA256SUMS e
  nuaa-patience-verification/comparison_run/status.json   confronto delle due run (una riga per modello, curve per
                                                           epoca, criteri, verdetto, sezioni a-e)

Sottocomandi:
  build --research <research/> --out <cartella>   scrive summary_table.{md,csv,json}, summary_table_compact.md,
                                                  cross_dataset_table.md, context_rows_table.md,
                                                  subject_sensitivity_table.md, folds_run_table.md,
                                                  n1_same_ids_table.md, patience_table.md,
                                                  session_ablation_table.md, patience_table_compact.md
                                                  (UTF-8, fine riga LF)
  check --research <research/> --out <cartella>   ricostruisce in memoria e confronta byte per byte con i file di --out;
                                                  esce con 1 se un file manca o differisce; con --synthesis <SYNTHESIS.md>
                                                  esige anche che ogni riga della tabella compatta e ogni riga
                                                  (tabella e righe di riepilogo) della tabella compatta della
                                                  pazienza stiano nel documento
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
fold; differenze, decisioni appaiate, conteggi per soggetto, mediane, bootstrap e valori di controllo coerenti. Run di
ablazione e run con pazienza 12: aggregato, record e status.json da albero pulito, PASS e passed vero, run id che finisce
con il commit e senza -dev, configurazione consegnata, gli stessi run id, commit, configurazione e amendment nei tre file;
status.json -> SHA256SUMS -> aggregato e record; 36 modelli completi (3 bracci x 4 fold x 3 seed), nessuno mancante,
nessun arresto, tempo entro il tetto; schema dei fold e run a fold di riferimento (run id e SHA-256 di aggregato e
record), bootstrap uguali; celle di fitting e dev del record uguali nelle due run e alle dimensioni di ogni scheda;
tassi = conteggi, denominatori dello schema, «all» = somma delle sessioni, aggregati = somma dei fold, mediane sui seed;
differenze dei confronti = minuendo meno sottraendo (per modello e aggregate); i verdetti (84 per run) riprodotti dalla
regola di lettura applicata alle differenze e agli intervalli per seed, verdetti primari uguali; amendment della run con
pazienza 12 = configurazione della run madre, campo training.early_stopping_patience con le pazienze degli ingressi del
confronto. Confronto delle due run: run ufficiale con status PASS, ancore (status.json -> SHA256SUMS -> aggregato, CSV,
curve), dichiarazioni; ingressi uguali alle due run (run id, commit, configurazione, stato, SHA-256 di status.json e
SHA256SUMS, modelli, pazienza); ogni riga del CSV ricostruita dalle curve con i criteri dell'aggregato (epoca
ripristinata = argmin della val_loss, replay dell'arresto precoce, degenere, prima ripresa, epoche a 0,5, lr finale,
accuratezza dev al checkpoint entro la tolleranza) e uguale alle righe JSON e alle schede delle due run; storie
condivise identiche; le dieci = i degeneri con pazienza 4; verdetto positivo riprodotto; mediane per gruppo; sezione
(d) = verdetti e mediane delle due run e numero dei verdetti cambiati; tempi entro il tetto e uguali ai record; stesso
checkpoint -> stessi valori di test e stessa soglia dev. Nessuna tabella Markdown generata con una riga duplicata. Il
controllo copre le celle delle tabelle; i numeri della prosa del documento restano da verificare a mano contro le
tabelle e le fonti.

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
    "sessabl_aggregate": "nuaa-session-ablation-run/tables/aggregate.json",
    "sessabl_record": "nuaa-session-ablation-run/tables/run_record.json",
    "sessabl_sums": "nuaa-session-ablation-run/tables/SHA256SUMS",
    "sessabl_status": "nuaa-session-ablation-run/run/status.json",
    "patience_aggregate": "nuaa-patience-verification/tables/aggregate.json",
    "patience_record": "nuaa-patience-verification/tables/run_record.json",
    "patience_sums": "nuaa-patience-verification/tables/SHA256SUMS",
    "patience_status": "nuaa-patience-verification/run/status.json",
    "patver_aggregate": "nuaa-patience-verification/comparison/aggregate.json",
    "patver_comparison": "nuaa-patience-verification/comparison/comparison.csv",
    "patver_curves": "nuaa-patience-verification/comparison/curves.csv",
    "patver_sums": "nuaa-patience-verification/comparison/SHA256SUMS",
    "patver_status": "nuaa-patience-verification/comparison_run/status.json",
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
    # expected, test_subjects e pooled_subjects servono ai controlli del run di ablazione (stessi frame di test)
    pooled_subjects = {st: agg["pooled_per_seed"][seeds[0]][st].get("n_subjects") for st in FOLDS_STRATA}
    return {"rows": rows, "texts": T, "annex": annex, "info": info, "seeds": seeds, "metric_cells": metric_cells, "pooled": pooled,
            "folds": folds, "expected": expected, "test_subjects": test_subjects, "pooled_subjects": pooled_subjects}


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


# --- run di ablazione delle sessioni (PR #21) e verifica della pazienza 12 (PR #23) -----------------------------------

ABL_ARMS = ("arm_b", "arm_a_star", "arm_b_prime")
ARM_LABELS = {"arm_b": "B", "arm_a_star": "A*", "arm_b_prime": "B'"}
ARM_IDS = {"arm_b": "B", "arm_a_star": "Astar", "arm_b_prime": "Bprime"}
ARM_TEXTS = {"arm_b": "arm B, without session 03: sessions 01-02 of the fitting subjects with frames in sessions 01-02",
             "arm_a_star": "arm A*, with session 03: sessions 01-03 of the same fitting subjects",
             "arm_b_prime": "arm B', without session 03, at the per-cell counts of A*: sessions 01-02 of the same fitting subjects"}
ABL_COMPARISONS = (("a_star_minus_b", "arm_a_star", "arm_b", "primary"), ("b_prime_minus_b", "arm_b_prime", "arm_b", "secondary"),
                   ("a_star_minus_b_prime", "arm_a_star", "arm_b_prime", "secondary"))
COMPARISON_LABELS = {"a_star_minus_b": "A* - B", "b_prime_minus_b": "B' - B", "a_star_minus_b_prime": "A* - B'"}
PRIMARY_STRATUM, PRIMARY_METRICS = "03", ("auroc", "acer_dev")
# Motivi della regola di lettura nell'ordine in cui la regola li applica; il verdetto si riproduce dalle differenze per seed
RULE_REASONS = {"zero": "a seed has a zero difference", "sign": "the sign differs across the seeds",
                "interval": "the bootstrap interval of a seed does not exclude 0 on the side of the difference",
                "consistent": "same sign for every seed and every interval excludes 0"}
# Nomi delle metriche nella direzione di un verdetto «consistent»: METRIC_LABEL del runner dell'ablazione
# (nuaa_session_ablation_run.py), per esempio «A* lower BPCER (0.5) than B»
METRIC_LABEL = {"apcer_05": "APCER (0.5)", "bpcer_05": "BPCER (0.5)", "acer_05": "ACER (0.5)", "apcer_dev": "APCER (dev threshold)",
                "bpcer_dev": "BPCER (dev threshold)", "acer_dev": "ACER (dev threshold)", "auroc": "AUROC"}
# Soglie delle righe di riepilogo della pazienza: le stesse di CONCLUSION.md della verifica della pazienza («five of the ten
# recover at epoch 10 or later», «two of the ten keep a dev accuracy below 0.9»); conteggi descrittivi, non metriche
LATE_RECOVERY_EPOCH = 10
# Larghezza massima di una riga di riepilogo (trattino compreso): le righe stanno alla lettera in SYNTHESIS.md, a 120 colonne
SUMMARY_WIDTH = 120
LOW_DEV_ACCURACY = 0.9
ABL_BLOCK = "Session ablation run, patience %d"
TRANSFER_ABL = ("subject-held-out, session ablation (held-out subjects never used for fitting or dev; sessions in training "
                "as in the arm; dev without session 03)")
THR_ABL_ORIGIN = ("0.5 fixed a priori (configuration); dev threshold: EER on the common dev only (M1 criterion), written and "
                  "hashed before any test scoring, one per model")
ABL_AGG_FIELDS = ["run_id", "commit", "dirty", "config_sha256", "delivered_configuration", "command", "status", "finished",
                  "completed", "trainings", "missing", "bootstrap", "strata", "per_arm_fold_seed", "pooled_per_arm_seed",
                  "bootstrap_per_arm_seed", "pooled_summary_over_seeds", "comparisons", "primary_verdicts", "reading_rule",
                  "reference_a", "training"]
ABL_REC_FIELDS = ["run_id", "commit", "dirty", "config_sha256", "delivered_configuration", "command", "status", "finished",
                  "wall_seconds", "timings", "trainings", "training_order", "completed", "missing", "stopped", "cap_minutes",
                  "final_writes_after_cap", "checks", "train_seeds", "bootstrap"]
ABL_STATUS_FIELDS = ["run_id", "commit", "dirty", "config_sha256", "delivered_configuration", "command", "status", "passed",
                     "finished", "wall_seconds", "completed", "missing", "stopped", "final_writes_after_cap",
                     "public_sums_sha256", "restricted_sums_sha256"]
ABL_TRAINING_FIELDS = ("arm", "fold", "seed", "epochs_run", "restored_epoch", "restored_by", "best_val_loss", "train_seconds",
                       "n_fitting", "n_dev", "dev_threshold")
PATVER_FIELDS = ["run_id", "commit", "dirty", "config_sha256", "delivered_configuration", "criteria", "recipe",
                 "original_degenerate", "models_without_recovery_parent", "inputs", "verdict", "sections", "rows", "declarations"]
PATVER_STATUS_FIELDS = ["run_id", "commit", "dirty", "config_sha256", "delivered_configuration", "command", "status", "passed",
                        "verdict", "verification_status", "public_sums_sha256"]
PATVER_DECLARATIONS = ("no_training", "no_inference", "no_test_scores_read", "test_from_public_aggregate_only",
                       "test_not_used_to_judge_the_recipe", "reading_rule_fixed_before_the_run")
SIDE_FIELDS = ("epochs_run", "restored_epoch", "degenerate", "recovered", "first_recovery", "best_val_loss",
               "dev_acc_at_checkpoint", "dev_auroc_at_checkpoint", "epochs_at_half", "lr_last")
CMP_COLUMNS = (["arm", "fold", "seed", "in_ten", "new_status"] + ["parent_" + f for f in SIDE_FIELDS] + ["new_" + f for f in SIDE_FIELDS]
               + ["shared_epochs", "shared_identical"] + ["max_abs_diff_shared_" + f for f in ("loss", "accuracy", "val_loss",
                                                                                                "val_accuracy", "lr")])
CURVE_COLUMNS = ["arm", "fold", "seed", "in_ten", "epoch", "loss", "accuracy", "val_loss", "val_accuracy", "lr", "is_restored",
                 "saturated", "shared_with_parent", "degenerate", "recovered_by_epoch"]
HISTORY_FIELDS = ("loss", "accuracy", "val_loss", "val_accuracy", "lr")
# campi delle righe JSON del confronto (parent/new) e colonne corrispondenti del CSV
ROW_SIDE_FIELDS = {"epochs_run": "epochs_run", "restored_epoch": "restored_epoch", "degenerate": "degenerate",
                   "recovered": "recovered", "first_recovery": "first_recovery", "best_val_loss": "best_val_loss",
                   "dev_acc": "dev_acc_at_checkpoint", "dev_auroc": "dev_auroc_at_checkpoint", "epochs_at_half": "epochs_at_half",
                   "lr_last": "lr_last"}
SUMMARY_GROUPS = ("the ten", "the other 26", "all 36")
SUMMARY_MEASURES = {"first_recovery": "first_recovery", "restored_epoch": "restored_epoch", "best_val_loss": "best_val_loss",
                    "dev_acc": "dev_acc_at_checkpoint", "dev_auroc": "dev_auroc_at_checkpoint", "epochs_at_half": "epochs_at_half",
                    "lr_last": "lr_last"}


def model_name(arm: str, k: str, s: str) -> str:
    return "%s_fold%s_seed%s" % (arm, k, s)


def model_label(arm: str, k: str, s: str) -> str:
    return "%s fold %s seed %s" % (ARM_LABELS[arm], k, s)


def read_sums(p: Path, where: str) -> dict:
    """Un file SHA256SUMS («<sha256>  <file>» per riga) come {file: sha256}; righe malformate o ripetute fermano tutto."""
    out = {}
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        mm = re.fullmatch(r"([0-9a-f]{64})  (\S+)", line)
        if not mm:
            fail("%s line %d: not '<sha256>  <file>'" % (where, i))
        if mm.group(2) in out:
            fail("%s: %s listed twice" % (where, mm.group(2)))
        out[mm.group(2)] = mm.group(1)
    return out


def read_csv_rows(p: Path, columns: list, where: str) -> list:
    with open(p, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != columns:
            fail("%s: columns %s, expected %s" % (where, reader.fieldnames, columns))
        rows = list(reader)
    for i, r in enumerate(rows, 2):
        if None in r or any(v is None for v in r.values()):
            fail("%s line %d: wrong number of cells" % (where, i))
    return rows


def csv_int(v: str, where: str, empty_ok: bool = False):
    if v == "" and empty_ok:
        return None
    if not re.fullmatch(r"-?\d+", v):
        fail("%s: %r is not an integer" % (where, v))
    return int(v)


def csv_flag(v: str, where: str) -> bool:
    if v not in ("0", "1"):
        fail("%s: %r is not 0 or 1" % (where, v))
    return v == "1"


def csv_float(v: str, where: str) -> float:
    try:
        x = float(v)
    except ValueError:
        fail("%s: %r is not a number" % (where, v))
    if x != x or x in (float("inf"), float("-inf")):
        fail("%s: %r is not a finite number" % (where, v))
    return x


def rule_reason(ds: list):
    """Regola di lettura (configurazione del run): «consistent» solo con lo stesso segno non nullo per i tre seed e ogni
    intervallo bootstrap che esclude 0 dal lato di quel segno; altrimenti il motivo dell'«inconclusive»."""
    if any(not isinstance(d, dict) or not all(isinstance(d.get(k), (int, float)) and not isinstance(d.get(k), bool)
                                              for k in ("difference", "low", "high")) for d in ds):
        return "missing"
    diffs = [d["difference"] for d in ds]
    if any(x == 0 for x in diffs):
        return "zero"
    if not (all(x > 0 for x in diffs) or all(x < 0 for x in diffs)):
        return "sign"
    if not all(d["low"] > 0 if d["difference"] > 0 else d["high"] < 0 for d in ds):
        return "interval"
    return "consistent"


def ablation_run(where: str, agg: dict, rec: dict, status: dict, sums: dict, H: dict, keys: dict, folds: dict, fsch: dict,
                 problems: list) -> dict:
    """Un run del disegno di ablazione delle sessioni (bracci B, A*, B' x 4 fold x 3 seed): identità e albero pulito,
    ancore (status.json -> SHA256SUMS -> aggregato e record), 36 modelli completi entro il tetto, schema dei fold e run a
    fold di riferimento, celle di fitting e dev, metriche per modello e aggregate (tassi = conteggi, denominatori dello
    schema, «all» = somma delle sessioni, aggregato = somma dei fold), mediane, bootstrap, differenze dei confronti e
    verdetti della regola di lettura riprodotti dalle differenze per seed."""
    for d, w in ((agg, "aggregate"), (rec, "run record"), (status, "status")):
        if d.get("dirty") is not False:
            fail("%s %s: dirty is not false" % (where, w))
        if str(d.get("run_id", "")).endswith("-dev"):
            fail("%s %s: run id %r ends with -dev (trial run)" % (where, w, d.get("run_id")))
        run_id_check(d, "%s %s" % (where, w))
        if d.get("delivered_configuration") is not True or d.get("command") != "run":
            fail("%s %s: not a run of the delivered configuration (delivered_configuration %r, command %r)"
                 % (where, w, d.get("delivered_configuration"), d.get("command")))
        if d.get("status") != "PASS":
            fail("%s %s: status %r is not PASS" % (where, w, d.get("status")))
    if status.get("passed") is not True:
        fail("%s status: passed is not true" % where)
    for k in ("run_id", "commit", "config_sha256", "finished", "configuration_name", "amendment"):
        for d, w in ((rec, "run record"), (status, "status")):
            if d.get(k) != agg.get(k):
                problems.append("%s %s: %s %r differs from the aggregate %r" % (where, w, k, d.get(k), agg.get(k)))
    for k in ("wall_seconds", "completed", "missing", "stopped", "final_writes_after_cap"):
        if status.get(k) != rec.get(k):
            problems.append("%s status: %s differs from the run record" % (where, k))
    if status.get("public_sums_sha256") != H[keys["sums"]]:
        problems.append("%s status: public_sums_sha256 is not the SHA-256 of %s" % (where, SOURCES[keys["sums"]]))
    for name, key in (("aggregate.json", keys["aggregate"]), ("run_record.json", keys["record"])):
        if sums.get(name) != H[key]:
            problems.append("%s: %s in SHA256SUMS is not the SHA-256 of %s" % (where, name, SOURCES[key]))

    seeds, fl = folds["seeds"], folds["folds"]
    if [str(s) for s in rec["train_seeds"]] != seeds:
        fail("%s run record: seeds %s differ from the folds run %s" % (where, rec["train_seeds"], seeds))
    names = [model_name(a, k, s) for a in ABL_ARMS for k in fl for s in seeds]
    if not isinstance(agg["training"], dict) or not isinstance(rec["completed"], list) or not isinstance(rec["training_order"], list):
        fail("%s: training, completed or training_order has an unexpected structure" % where)
    if sorted(agg["training"]) != sorted(names) or len(rec["completed"]) != len(names) or sorted(rec["completed"]) != sorted(names) \
            or sorted(rec["training_order"]) != sorted(names):
        problems.append("%s: the models of the aggregate, completed and training_order are not the %d of %d arms x %d folds x %d seeds"
                        % (where, len(names), len(ABL_ARMS), len(fl), len(seeds)))
    if (agg["completed"], agg["trainings"], rec["trainings"], agg["missing"], rec["missing"]) != (len(names), len(names), len(names), [], []):
        problems.append("%s: %r of %r trainings complete, missing %r; expected %d of %d, none missing"
                        % (where, agg["completed"], agg["trainings"], agg["missing"], len(names), len(names)))
    if rec["stopped"] is not None or rec["final_writes_after_cap"] is not False:
        problems.append("%s run record: stopped %r, final writes after the cap %r" % (where, rec["stopped"], rec["final_writes_after_cap"]))
    cap = rec["cap_minutes"] * 60
    if not rec["wall_seconds"] <= cap:
        problems.append("%s run record: wall time %s s above the cap %s s" % (where, rec["wall_seconds"], cap))

    bs = agg["bootstrap"]
    if bs != rec["bootstrap"]:
        problems.append("%s run record: bootstrap differs from the aggregate" % where)
    params = ("resamples", "seed", "percentiles", "min_subjects")
    if not isinstance(bs, dict) or {k: bs.get(k) for k in params} != {k: folds["info"]["bootstrap"].get(k) for k in params}:
        fail("%s aggregate: bootstrap parameters differ from the folds run" % where)
    checks = rec["checks"]
    if (dig(checks, "schema", "run_id"), dig(checks, "schema", "test_per_fold"), dig(checks, "schema_config", "sha256")) != \
            (fsch["run_id"], folds["info"]["test_frames_per_fold"], fsch["config_sha256"]):
        problems.append("%s run record: fold schema (run id, test rows per fold, configuration SHA-256) differs from the fold schema" % where)
    if (dig(checks, "folds_run", "run_id"), dig(checks, "folds_run", "aggregate_json_sha256"),
            dig(checks, "folds_run", "run_record_json_sha256")) != (folds["info"]["run_id"], H["folds_aggregate"], H["folds_record"]):
        problems.append("%s run record: folds run (run id, aggregate and record SHA-256) differs from the folds run read here" % where)
    if (dig(agg, "reference_a", "run_id"), dig(agg, "reference_a", "aggregate_sha256")) != (folds["info"]["run_id"], H["folds_aggregate"]):
        problems.append("%s aggregate: reference A (run id, aggregate SHA-256) differs from the folds run read here" % where)

    # Celle di fitting per braccio e del dev comune (record), dimensioni di ogni scheda
    cells, dev_cells = dig(checks, "cells", "counts"), dig(checks, "cells", "dev")
    if not isinstance(cells, dict) or sorted(cells) != sorted(ABL_ARMS) or not isinstance(dev_cells, dict) or not dev_cells:
        fail("%s run record: fitting cells per arm or dev cells missing" % where)
    for c in [cells[a] for a in ABL_ARMS] + [dev_cells]:
        if not isinstance(c, dict) or not c or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in c.values()):
            fail("%s run record: a cell count is not a positive integer" % where)
    n_fit = {a: sum(cells[a].values()) for a in ABL_ARMS}
    n_dev = sum(dev_cells.values())
    training = agg["training"]
    for a in ABL_ARMS:
        for k in fl:
            for s in seeds:
                name = model_name(a, k, s)
                tr = training.get(name)
                if not isinstance(tr, dict):
                    fail("%s aggregate: training of %s missing" % (where, name))
                for f in ABL_TRAINING_FIELDS:
                    if f not in tr:
                        fail("%s aggregate: training field %s of %s missing" % (where, f, name))
                if (tr["arm"], str(tr["fold"]), str(tr["seed"])) != (a, k, s):
                    problems.append("%s %s: arm, fold or seed of the card differ from its name" % (where, name))
                if (tr["n_fitting"], tr["n_dev"]) != (n_fit[a], n_dev):
                    problems.append("%s %s: %s fitting / %s dev frames, the cells of the record give %s / %s"
                                    % (where, name, tr["n_fitting"], tr["n_dev"], n_fit[a], n_dev))
                if not 1 <= tr["restored_epoch"] <= tr["epochs_run"]:
                    problems.append("%s %s: restored epoch %s outside 1..%s" % (where, name, tr["restored_epoch"], tr["epochs_run"]))
                if "train " + name not in rec["timings"]:
                    problems.append("%s run record: no timing for train %s" % (where, name))

    # Metriche per modello e strato; aggregato per braccio e seed
    if list(agg["strata"]) != list(FOLDS_STRATA):
        fail("%s aggregate: strata %r, expected %r" % (where, agg["strata"], list(FOLDS_STRATA)))
    per, raw = {}, {}
    for a in ABL_ARMS:
        for k in fl:
            for s in seeds:
                blk = dig(agg, "per_arm_fold_seed", a, k, s)
                if not isinstance(blk, dict):
                    fail("%s aggregate: %s missing" % (where, model_label(a, k, s)))
                lab = "%s %s" % (where, model_label(a, k, s))
                ms = {st: folds_metric(blk.get(st), "%s stratum %s" % (lab, st)) for st in FOLDS_STRATA}
                for st in FOLDS_STRATA:
                    got = (ms[st]["05"]["n_bona_fide"], ms[st]["05"]["n_attack"])
                    if got != folds["expected"][k][st]:
                        problems.append("%s stratum %s: denominators %s/%s differ from the fold schema %s/%s"
                                        % (lab, st, got[0], got[1], folds["expected"][k][st][0], folds["expected"][k][st][1]))
                for t in ("05", "dev"):
                    if summed([ms[st][t] for st in SESSION_STRATA]) != counts_of(ms["all"][t]):
                        problems.append("%s: stratum all at %s is not the sum of the session strata" % (lab, t))
                if blk["all"].get("n_subjects") != folds["test_subjects"][k]:
                    problems.append("%s: %r subjects, the fold schema has %r" % (lab, blk["all"].get("n_subjects"), folds["test_subjects"][k]))
                per[(a, k, s)], raw[(a, k, s)] = ms, blk
    pooled, praw = {}, {}
    for a in ABL_ARMS:
        for s in seeds:
            for st in FOLDS_STRATA:
                blk = dig(agg, "pooled_per_arm_seed", a, s, st)
                lab = "%s %s seed %s stratum %s" % (where, ARM_LABELS[a], s, st)
                m = folds_metric(blk, "%s pooled" % lab)
                for t in ("05", "dev"):
                    if counts_of(m[t]) != summed([per[(a, k, s)][st][t] for k in fl]):
                        problems.append("%s at %s: pooled counts are not the sum of the %d folds" % (lab, t, len(fl)))
                if blk.get("n_subjects") != folds["pooled_subjects"][st]:
                    problems.append("%s: %r pooled subjects, the folds run has %r" % (lab, blk.get("n_subjects"), folds["pooled_subjects"][st]))
                pooled[(a, s, st)], praw[(a, s, st)] = m, blk
    for a in ABL_ARMS:
        for st in FOLDS_STRATA:
            for key in FOLDS_SUMMARY_KEYS:
                e, vals = dig(agg, "pooled_summary_over_seeds", a, st, key), [praw[(a, s, st)][key] for s in seeds]
                if not isinstance(e, dict) or e.get("values") != vals:
                    problems.append("%s summary %s stratum %s %s: missing or values differ from the per-seed rows" % (where, ARM_LABELS[a], st, key))
                elif None not in vals and (e.get("median"), e.get("min"), e.get("max")) != (statistics.median(vals), min(vals), max(vals)):
                    problems.append("%s summary %s stratum %s %s: median or range is not that of the values" % (where, ARM_LABELS[a], st, key))
    boot = {}
    for a in ABL_ARMS:
        for s in seeds:
            for st in FOLDS_STRATA:
                b, n_sub = dig(agg, "bootstrap_per_arm_seed", a, s, st), praw[(a, s, st)].get("n_subjects")
                lab = "%s bootstrap %s seed %s stratum %s" % (where, ARM_LABELS[a], s, st)
                if not isinstance(n_sub, int) or n_sub < bs["min_subjects"]:
                    continue
                if not isinstance(b, dict):
                    problems.append("%s missing" % lab)
                    continue
                if (b.get("n_subjects"), b.get("resamples"), b.get("seed"), b.get("percentiles")) != (n_sub, bs["resamples"], bs["seed"], bs["percentiles"]):
                    problems.append("%s: subjects or parameters differ from the declared ones" % lab)
                ok = True
                for key in ("acer_05", "acer_dev", "auroc"):
                    x = b.get(key)
                    if not isinstance(x, dict) or not isinstance(x.get("low"), (int, float)) or not isinstance(x.get("high"), (int, float)) \
                            or x["low"] > x["high"]:
                        problems.append("%s: %s interval missing or reversed" % (lab, key))
                        ok = False
                if ok:
                    boot[(a, s, st)] = b

    # Confronti: differenze per modello e aggregate, mediane, verdetti della regola riprodotti
    comps = agg["comparisons"]
    if not isinstance(comps, dict) or sorted(comps) != sorted(c[0] for c in ABL_COMPARISONS):
        fail("%s aggregate: comparisons %r, expected %r" % (where, sorted(comps) if isinstance(comps, dict) else comps,
                                                            sorted(c[0] for c in ABL_COMPARISONS)))
    verdicts = {}
    for cname, mi, su, role in ABL_COMPARISONS:
        c = comps[cname]
        lab = "%s comparison %s" % (where, COMPARISON_LABELS[cname])
        if (c.get("minuend"), c.get("subtrahend"), c.get("role")) != (mi, su, role):
            problems.append("%s: minuend, subtrahend or role differ from the protocol" % lab)
        bad = [model_label(mi, k, s) + " " + st + " " + m for k in fl for s in seeds for st in FOLDS_STRATA for m in FOLDS_SUMMARY_KEYS
               if dig(c, "per_fold_seed", k, s, st, m) != (None if raw[(mi, k, s)][st][m] is None or raw[(su, k, s)][st][m] is None
                                                            else raw[(mi, k, s)][st][m] - raw[(su, k, s)][st][m])]
        if bad:
            problems.append("%s: %d per-model differences are not minuend minus subtrahend (first: %s)" % (lab, len(bad), bad[0]))
        for s in seeds:
            for st in FOLDS_STRATA:
                p = dig(c, "pooled_per_seed", s, st)
                if not isinstance(p, dict):
                    fail("%s: pooled seed %s stratum %s missing" % (lab, s, st))
                if p.get("n_subjects") != praw[(mi, s, st)].get("n_subjects"):
                    problems.append("%s seed %s stratum %s: subjects differ from the pooled rows" % (lab, s, st))
                for m in FOLDS_SUMMARY_KEYS:
                    x = p.get(m)
                    want = (praw[(mi, s, st)][m], praw[(su, s, st)][m])
                    if not isinstance(x, dict) or (x.get("minuend"), x.get("subtrahend")) != want or \
                            (None not in want and x.get("difference") != want[0] - want[1]):
                        problems.append("%s seed %s stratum %s %s: the difference is not that of the pooled rows" % (lab, s, st, m))
                    elif isinstance(x.get("low"), (int, float)) and isinstance(x.get("high"), (int, float)) and x["low"] > x["high"]:
                        problems.append("%s seed %s stratum %s %s: interval reversed" % (lab, s, st, m))
        for st in FOLDS_STRATA:
            for m in FOLDS_SUMMARY_KEYS:
                ds = [dig(c, "pooled_per_seed", s, st, m) for s in seeds]
                vals = [d.get("difference") if isinstance(d, dict) else None for d in ds]
                e = dig(c, "pooled_summary_over_seeds", st, m)
                if not isinstance(e, dict) or e.get("values") != vals:
                    problems.append("%s summary stratum %s %s: missing or values differ from the per-seed differences" % (lab, st, m))
                elif None not in vals and (e.get("median"), e.get("min"), e.get("max")) != (statistics.median(vals), min(vals), max(vals)):
                    problems.append("%s summary stratum %s %s: median or range is not that of the values" % (lab, st, m))
                v = dig(c, "verdicts", st, m)
                if not isinstance(v, dict):
                    fail("%s: verdict of stratum %s %s missing" % (lab, st, m))
                reason = rule_reason(ds)
                label = "primary" if (cname, st) == ("a_star_minus_b", PRIMARY_STRATUM) and m in PRIMARY_METRICS else "secondary"
                want = "consistent" if reason == "consistent" else "inconclusive"
                ok = v.get("verdict") == want and v.get("label") == label and \
                    (reason == "missing" or v.get("reason") == RULE_REASONS[reason])
                if want == "consistent":
                    sign = "higher" if vals[0] > 0 else "lower"
                    ok = ok and v.get("direction") == "%s %s %s than %s" % (ARM_LABELS[mi], sign, METRIC_LABEL[m], ARM_LABELS[su])
                else:
                    ok = ok and v.get("direction") is None
                if not ok:
                    problems.append("%s stratum %s %s: verdict %r (%r) is not the reading rule applied to the per-seed differences (%s, %s)"
                                    % (lab, st, m, v.get("verdict"), v.get("reason"), want, label))
                verdicts[(cname, st, m)] = {"verdict": v.get("verdict"), "direction": v.get("direction"), "reason": v.get("reason"),
                                            "label": v.get("label"), "median": e.get("median") if isinstance(e, dict) else None,
                                            "per_seed": [(d.get("difference"), d.get("low"), d.get("high")) if isinstance(d, dict)
                                                         else (None, None, None) for d in ds]}
    pv = agg["primary_verdicts"]
    if not isinstance(pv, dict) or sorted(pv) != sorted(PRIMARY_METRICS) or any(
            {k: dig(pv, m, k) for k in ("verdict", "direction", "reason", "label")}
            != {k: verdicts[("a_star_minus_b", PRIMARY_STRATUM, m)][k] for k in ("verdict", "direction", "reason", "label")} for m in PRIMARY_METRICS):
        problems.append("%s aggregate: primary verdicts differ from the verdicts of A* - B in stratum %s" % (where, PRIMARY_STRATUM))
    if not isinstance(agg["reading_rule"], str) or not agg["reading_rule"]:
        problems.append("%s aggregate: reading rule missing" % where)
    return {"names": names, "per": per, "raw": raw, "pooled": pooled, "praw": praw, "boot": boot, "verdicts": verdicts,
            "training": training, "cells": cells, "dev_cells": dev_cells, "n_fit": n_fit, "n_dev": n_dev, "agg": agg, "rec": rec,
            "cap": cap}


def patience_compare(cmp: dict, cstatus: dict, csums: dict, crows: list, curves: list, runs: dict, H: dict, statuses: dict,
                     folds: dict, problems: list) -> dict:
    """Confronto delle due run (pazienza 4 e 12, PR #23): run ufficiale e ancore, ingressi uguali alle due run lette qui,
    ogni riga del CSV ricostruita dalle curve per epoca con i criteri dell'aggregato (epoca ripristinata = argmin della
    val_loss, replay dell'arresto precoce, degenere, prima ripresa, epoche a 0,5, lr finale), righe JSON e schede delle
    due run uguali al CSV, storie condivise identiche, verdetto riprodotto, mediane per gruppo, sezione (d) uguale ai
    verdetti delle due run, tempi entro il tetto."""
    parent, new = runs["parent"], runs["new"]
    where = "patience comparison"
    for d, w in ((cmp, "aggregate"), (cstatus, "status")):
        if d.get("dirty") is not False:
            fail("%s %s: dirty is not false" % (where, w))
        if str(d.get("run_id", "")).endswith("-dev"):
            fail("%s %s: run id %r ends with -dev (trial run)" % (where, w, d.get("run_id")))
        run_id_check(d, "%s %s" % (where, w))
        if d.get("delivered_configuration") is not True:
            fail("%s %s: not a run of the delivered configuration" % (where, w))
    if cstatus.get("passed") is not True or cstatus.get("status") != "PASS" or cstatus.get("command") != "run":
        fail("%s status: passed %r, status %r, command %r" % (where, cstatus.get("passed"), cstatus.get("status"), cstatus.get("command")))
    for k in ("run_id", "commit", "config_sha256", "verdict"):
        if cstatus.get(k) != cmp.get(k):
            problems.append("%s status: %s %r differs from the aggregate %r" % (where, k, cstatus.get(k), cmp.get(k)))
    if cstatus.get("verification_status") != new["agg"]["status"] or cstatus.get("verification_status") != "PASS":
        problems.append("%s status: verification_status %r is not the status of the patience-verification run %r, PASS"
                        % (where, cstatus.get("verification_status"), new["agg"]["status"]))
    if cstatus.get("public_sums_sha256") != H["patver_sums"]:
        problems.append("%s status: public_sums_sha256 is not the SHA-256 of %s" % (where, SOURCES["patver_sums"]))
    for name, key in (("aggregate.json", "patver_aggregate"), ("comparison.csv", "patver_comparison"), ("curves.csv", "patver_curves")):
        if csums.get(name) != H[key]:
            problems.append("%s: %s in SHA256SUMS is not the SHA-256 of %s" % (where, name, SOURCES[key]))
    for k in PATVER_DECLARATIONS:
        if dig(cmp, "declarations", k) is not True:
            problems.append("%s: declaration %s is not true" % (where, k))

    # Ingressi: le due run lette qui (run id, commit, configurazione, stato, ancore, pazienza, modelli)
    patience = {}
    for side, key, run in (("parent", "parent_run", parent), ("new", "verification_run", new)):
        inp = dig(cmp, "inputs", key)
        if not isinstance(inp, dict):
            fail("%s: inputs.%s missing" % (where, key))
        a, st_ = run["agg"], statuses[side]
        want = {"run_id": a["run_id"], "commit": a["commit"], "config_sha256": a["config_sha256"], "status": "PASS",
                "status_sha256": H[run["keys"]["status"]], "public_sums_sha256": H[run["keys"]["sums"]],
                "restricted_sums_sha256": st_.get("restricted_sums_sha256"), "complete_models": len(run["names"]), "missing_models": 0}
        got = {k: inp.get(k) for k in want}
        if got != want:
            problems.append("%s: inputs.%s %r differ from the run read here %r"
                            % (where, key, {k: v for k, v in got.items() if v != want[k]}, {k: v for k, v in want.items() if v != got[k]}))
        p = inp.get("early_stopping_patience")
        if not isinstance(p, int) or isinstance(p, bool) or p < 1:
            fail("%s: inputs.%s.early_stopping_patience %r is not a positive integer" % (where, key, p))
        patience[side] = p
    am = new["agg"].get("amendment")
    if not isinstance(am, dict) or (am.get("parent_config_sha256"), am.get("changed_field"), am.get("from"), am.get("to")) != \
            (parent["agg"]["config_sha256"], "training.early_stopping_patience", patience["parent"], patience["new"]):
        problems.append("%s: amendment of the patience-%s run %r is not parent %s, training.early_stopping_patience %s -> %s"
                        % (where, patience["new"], am, parent["agg"]["config_sha256"][:12], patience["parent"], patience["new"]))
    if parent["agg"].get("amendment") is not None:
        problems.append("%s: the parent run carries an amendment" % where)
    if patience["parent"] == patience["new"]:
        problems.append("%s: the two runs have the same patience %s" % (where, patience["new"]))
    if parent["agg"]["config_sha256"] == new["agg"]["config_sha256"]:
        problems.append("%s: the two runs have the same configuration" % where)
    max_epochs = dig(cmp, "recipe", "max_epochs")
    if not isinstance(max_epochs, int) or isinstance(max_epochs, bool) or max_epochs < 1:
        fail("%s: recipe.max_epochs %r is not a positive integer" % (where, max_epochs))
    cr = cmp["criteria"]
    try:
        rest_set, acc_half, tol_deg = cr["degenerate"]["restored_epochs"], cr["degenerate"]["val_acc_at_best"], cr["degenerate"]["tolerance"]
        sat_acc, sat_tol = cr["saturation"]["val_accuracy"], cr["saturation"]["tolerance"]
        rec_acc, rec_loss = cr["recovery"]["val_accuracy_min"], cr["recovery"]["val_loss_max"]
        ckpt_tol = cr["checkpoint_agreement_tolerance"]
        crit_text = (cr["degenerate"]["text"], cr["recovery"]["text"])
    except (KeyError, TypeError):
        fail("%s: criteria incomplete" % where)

    # Righe del CSV nell'ordine dei modelli; curve per modello, epoche 1..n
    names = parent["names"]
    order = [(a, k, s) for a in ABL_ARMS for k in folds["folds"] for s in folds["seeds"]]
    if len(crows) != len(order):
        fail("%s: comparison.csv has %d rows, expected %d" % (where, len(crows), len(order)))
    hist = {}
    for i, c in enumerate(curves, 2):
        lab = "curves.csv line %d" % i
        key = (c["arm"], c["fold"], c["seed"])
        e = {"epoch": csv_int(c["epoch"], lab), "in_ten": csv_flag(c["in_ten"], lab)}
        e.update({f: csv_float(c[f], lab) for f in HISTORY_FIELDS})
        e.update({f: csv_flag(c[f], lab) for f in ("is_restored", "saturated", "shared_with_parent", "degenerate", "recovered_by_epoch")})
        hist.setdefault(key, []).append(e)
    if sorted(hist) != sorted(order):
        fail("%s: the models of curves.csv are not the %d models" % (where, len(order)))

    def side(h, n, p, lab):
        """Una run di un modello dalle sue prime n epoche: criteri dell'aggregato e replay dell'arresto precoce."""
        vl = [e["val_loss"] for e in h[:n]]
        restored = vl.index(min(vl)) + 1
        if n != min(max_epochs, restored + p):
            problems.append("%s: %d epochs run, early stopping with patience %d and best epoch %d gives %d"
                            % (lab, n, p, restored, min(max_epochs, restored + p)))
        holds = [e["val_accuracy"] >= rec_acc and e["val_loss"] < rec_loss for e in h[:n]]
        first = holds.index(True) + 1 if any(holds) else None
        return {"epochs_run": n, "restored_epoch": restored, "best_val_loss": min(vl), "first_recovery": first,
                "recovered": first is not None, "holding_last": holds[-1],
                "degenerate": restored in rest_set and abs(h[restored - 1]["val_accuracy"] - acc_half) <= tol_deg,
                "epochs_at_half": sum(abs(e["val_accuracy"] - sat_acc) <= sat_tol for e in h[:n]),
                "lr_last": h[n - 1]["lr"], "val_acc_restored": h[restored - 1]["val_accuracy"], "holds": holds}

    models = []
    for i, ((a, k, s), c) in enumerate(zip(order, crows), 2):
        lab = "comparison.csv %s" % model_label(a, k, s)
        if (c["arm"], c["fold"], c["seed"]) != (a, k, s):
            fail("%s: line %d is %s fold %s seed %s, expected the models in arm, fold, seed order" % (where, i, c["arm"], c["fold"], c["seed"]))
        row = {"arm": a, "fold": k, "seed": s, "in_ten": csv_flag(c["in_ten"], lab), "new_status": c["new_status"],
               "shared_epochs": csv_int(c["shared_epochs"], lab), "shared_identical": csv_flag(c["shared_identical"], lab),
               "max_abs_diff": {f: csv_float(c["max_abs_diff_shared_" + f], lab) for f in HISTORY_FIELDS}}
        for sd in ("parent", "new"):
            row[sd] = {"epochs_run": csv_int(c[sd + "_epochs_run"], lab), "restored_epoch": csv_int(c[sd + "_restored_epoch"], lab),
                       "degenerate": csv_flag(c[sd + "_degenerate"], lab), "recovered": csv_flag(c[sd + "_recovered"], lab),
                       "first_recovery": csv_int(c[sd + "_first_recovery"], lab, empty_ok=True),
                       "best_val_loss": csv_float(c[sd + "_best_val_loss"], lab),
                       "dev_acc_at_checkpoint": csv_float(c[sd + "_dev_acc_at_checkpoint"], lab),
                       "dev_auroc_at_checkpoint": csv_float(c[sd + "_dev_auroc_at_checkpoint"], lab),
                       "epochs_at_half": csv_int(c[sd + "_epochs_at_half"], lab), "lr_last": csv_float(c[sd + "_lr_last"], lab)}
        h = sorted(hist[(a, k, s)], key=lambda e: e["epoch"])
        if [e["epoch"] for e in h] != list(range(1, len(h) + 1)) or len(h) != row["new"]["epochs_run"]:
            problems.append("%s: curves.csv epochs are not 1..%d" % (lab, row["new"]["epochs_run"]))
            continue
        if row["new_status"] != "complete":
            problems.append("%s: status %r in the patience-%s run" % (lab, row["new_status"], patience["new"]))
        if not 1 <= row["parent"]["epochs_run"] <= len(h):
            fail("%s: %s epochs in the patience-%s run, %d in the curves" % (lab, row["parent"]["epochs_run"], patience["parent"], len(h)))
        der = {"parent": side(h, row["parent"]["epochs_run"], patience["parent"], "%s patience %d" % (lab, patience["parent"])),
               "new": side(h, len(h), patience["new"], "%s patience %d" % (lab, patience["new"]))}
        for sd, p in (("parent", patience["parent"]), ("new", patience["new"])):
            x, d = row[sd], der[sd]
            diff = [f for f in ("restored_epoch", "best_val_loss", "degenerate", "recovered", "first_recovery", "epochs_at_half", "lr_last")
                    if x[f] != d[f]]
            if diff:
                problems.append("%s patience %d: %s differ from the curves with the criteria of the aggregate" % (lab, p, ", ".join(diff)))
            if abs(x["dev_acc_at_checkpoint"] - d["val_acc_restored"]) > ckpt_tol:
                problems.append("%s patience %d: dev accuracy at the checkpoint differs from val_accuracy of the restored epoch" % (lab, p))
            tr = runs[sd]["training"][model_name(a, k, s)]
            if (tr["epochs_run"], tr["restored_epoch"], tr["best_val_loss"]) != (x["epochs_run"], x["restored_epoch"], x["best_val_loss"]):
                problems.append("%s patience %d: epochs, restored epoch or best val_loss differ from the training card of the run" % (lab, p))
        if row["shared_epochs"] != row["parent"]["epochs_run"] or not row["shared_identical"] or any(v != 0 for v in row["max_abs_diff"].values()):
            problems.append("%s: shared history not identical (%s shared epochs of %s)" % (lab, row["shared_epochs"], row["parent"]["epochs_run"]))
        for e in h:
            want = (e["epoch"] <= row["shared_epochs"], e["epoch"] == der["new"]["restored_epoch"],
                    abs(e["val_accuracy"] - sat_acc) <= sat_tol, der["new"]["holds"][e["epoch"] - 1], row["new"]["degenerate"], row["in_ten"])
            if (e["shared_with_parent"], e["is_restored"], e["saturated"], e["recovered_by_epoch"], e["degenerate"], e["in_ten"]) != want:
                problems.append("%s epoch %d: flags of curves.csv differ from the values" % (lab, e["epoch"]))
                break
        row["derived"] = der
        models.append(row)
    if len(models) != len(order):
        fail("consistency problems:\n  - " + "\n  - ".join(problems))

    # Le dieci: degeneri della run con pazienza 4 = in_ten = original_degenerate dell'aggregato
    labels = [model_label(r["arm"], r["fold"], r["seed"]) for r in models]
    ten = [l for l, r in zip(labels, models) if r["in_ten"]]
    if [l for l, r in zip(labels, models) if r["parent"]["degenerate"]] != ten:
        problems.append("%s: the models marked in_ten are not the degenerate models of the patience-%d run" % (where, patience["parent"]))
    if not isinstance(cmp["original_degenerate"], list) or sorted(cmp["original_degenerate"]) != sorted(ten):
        problems.append("%s: original_degenerate differs from the models marked in_ten" % where)
    if cmp["models_without_recovery_parent"] != [l for l, r in zip(labels, models) if not r["in_ten"] and not r["parent"]["recovered"]]:
        problems.append("%s: models_without_recovery_parent differs from the rows" % where)

    # Righe JSON = CSV
    jrows = cmp["rows"]
    if not isinstance(jrows, list) or len(jrows) != len(models):
        fail("%s: aggregate rows are not %d" % (where, len(models)))
    for l, r, j in zip(labels, models, jrows):
        ok = isinstance(j, dict) and j.get("model") == l and j.get("in_ten") is r["in_ten"] and j.get("new_status") == r["new_status"] \
            and j.get("shared_epochs") == r["shared_epochs"] and j.get("shared_identical") is r["shared_identical"] \
            and j.get("max_abs_diff_shared") == r["max_abs_diff"]
        for sd in ("parent", "new"):
            ok = ok and all(dig(j, sd, jf) == r[sd][cf] for jf, cf in ROW_SIDE_FIELDS.items())
        if not ok:
            problems.append("%s: aggregate row %s differs from comparison.csv" % (where, l))

    # (a) storie condivise, (b) verdetto, (c) gruppi
    sa, sb, sc = dig(cmp, "sections", "a"), dig(cmp, "sections", "b"), dig(cmp, "sections", "c")
    if not all(isinstance(x, dict) for x in (sa, sb, sc)) or not isinstance(dig(cmp, "sections", "d"), dict) \
            or not isinstance(dig(cmp, "sections", "e"), dict):
        fail("%s: sections a-e missing" % where)
    n_ident = sum(r["shared_identical"] for r in models)
    if (sa.get("models_compared"), sa.get("identical"), sa.get("valid"), sa.get("largest_difference")) != (len(models), n_ident, True, None) \
            or sa.get("max_abs_difference_per_field") != {f: 0.0 for f in HISTORY_FIELDS}:
        problems.append("%s: section a (shared histories) differs from the rows: %d of %d identical" % (where, n_ident, len(models)))
    new_deg = [l for l, r in zip(labels, models) if r["new"]["degenerate"]]
    ten_rec = [l for l, r in zip(labels, models) if r["in_ten"] and r["new"]["recovered"]]
    verdict = "positive" if len(ten_rec) == len(ten) and not new_deg else "negative"
    if (cmp["verdict"], sb.get("verdict"), cstatus.get("verdict")) != (verdict, verdict, verdict) or verdict != "positive":
        problems.append("%s: verdict %r (section b %r, status %r); the rows give %r" % (where, cmp["verdict"], sb.get("verdict"),
                                                                                       cstatus.get("verdict"), verdict))
    want_b = {"ten": ten, "ten_recovered": len(ten_rec), "ten_complete": len(ten), "ten_degenerate": [l for l in ten if l in new_deg],
              "ten_without_recovery": [l for l in ten if l not in ten_rec], "degenerate_among_complete": new_deg,
              "without_recovery_outside_ten": [l for l, r in zip(labels, models) if not r["in_ten"] and not r["new"]["recovered"]],
              "without_recovery_parent": [l for l, r in zip(labels, models) if not r["parent"]["recovered"]],
              "ten_first_recovery": {l: r["new"]["first_recovery"] for l, r in zip(labels, models) if r["in_ten"]},
              "verification_status": new["agg"]["status"], "missing_models": []}
    got_b = {k: sb.get(k) for k in want_b}
    got_b["ten"] = sorted(got_b["ten"]) if isinstance(got_b["ten"], list) else got_b["ten"]
    got_b["without_recovery_parent"] = sorted(got_b["without_recovery_parent"]) if isinstance(got_b["without_recovery_parent"], list) else None
    want_b["ten"], want_b["without_recovery_parent"] = sorted(want_b["ten"]), sorted(want_b["without_recovery_parent"])
    if got_b != want_b:
        problems.append("%s: section b differs from the rows in %s" % (where, sorted(k for k in want_b if got_b[k] != want_b[k])))
    rc = sc.get("recovered") or {}
    if (rc.get("parent"), rc.get("new"), rc.get("new_complete")) != (sum(r["parent"]["recovered"] for r in models),
                                                                      sum(r["new"]["recovered"] for r in models), len(models)):
        problems.append("%s: section c recovered counts differ from the rows" % where)
    groups = {"the ten": [r for r in models if r["in_ten"]], "the other 26": [r for r in models if not r["in_ten"]], "all 36": models}
    if len(groups["the ten"]) != 10 or len(models) != 36:
        problems.append("%s: %d models in the ten and %d in all; the groups of section c are the ten, the other 26 and all 36"
                        % (where, len(groups["the ten"]), len(models)))
    summ = sc.get("summary")
    want_s = []
    for g in SUMMARY_GROUPS:
        for sd in ("parent", "new"):
            e = {"group": g, "side": sd}
            for jm, cf in SUMMARY_MEASURES.items():
                vals = [r[sd][cf] for r in groups[g] if r[sd][cf] is not None]
                e[jm] = {"models": len(vals), "median": statistics.median(vals) if vals else None,
                         "min": min(vals) if vals else None, "max": max(vals) if vals else None}
            want_s.append(e)
    if summ != want_s:
        problems.append("%s: section c medians per group differ from the rows" % where)

    # (d) test: verdetti delle due run uguali agli aggregati; i 7 (o quanti siano) che cambiano
    sd_ = cmp["sections"]["d"]
    vp, vn = parent["verdicts"], new["verdicts"]
    keys_ = [(c[0], st, m) for c in ABL_COMPARISONS for st in FOLDS_STRATA for m in FOLDS_SUMMARY_KEYS]
    changed = [k for k in keys_ if (vp[k]["verdict"], vp[k]["direction"]) != (vn[k]["verdict"], vn[k]["direction"])]
    drows = sd_.get("rows")
    want_d = [{"comparison": COMPARISON_LABELS[c], "stratum": st, "metric": m,
               "parent": {"verdict": vp[(c, st, m)]["verdict"], "direction": vp[(c, st, m)]["direction"],
                          "reason": vp[(c, st, m)]["reason"], "median": vp[(c, st, m)]["median"]},
               "new": {"verdict": vn[(c, st, m)]["verdict"], "direction": vn[(c, st, m)]["direction"],
                       "reason": vn[(c, st, m)]["reason"], "median": vn[(c, st, m)]["median"]}} for c, st, m in keys_]
    if drows != want_d:
        problems.append("%s: section d rows differ from the verdicts and medians of the two aggregates" % where)
    want_p = [{"metric": m, "parent": {k: vp[("a_star_minus_b", PRIMARY_STRATUM, m)][k] for k in ("verdict", "direction", "reason")},
               "new": {k: vn[("a_star_minus_b", PRIMARY_STRATUM, m)][k] for k in ("verdict", "direction", "reason")}} for m in PRIMARY_METRICS]
    if sd_.get("primary") != want_p:
        problems.append("%s: section d primary verdicts differ from the two aggregates" % where)
    if sd_.get("verdicts_changed") != len(changed):
        problems.append("%s: section d declares %r verdicts changed, the two aggregates give %d" % (where, sd_.get("verdicts_changed"), len(changed)))
    if not isinstance(sd_.get("label"), str) or sd_.get("reported") is not True:
        problems.append("%s: section d label missing or not reported" % where)

    # (e) tempi: entro il tetto, uguali ai record; epoche e secondi per braccio uguali alle schede
    se = cmp["sections"]["e"]
    if (se.get("new_wall_seconds"), se.get("parent_wall_seconds"), se.get("cap_seconds"), se.get("within_cap"),
            se.get("final_writes_after_cap"), se.get("stopped")) != (new["rec"]["wall_seconds"], parent["rec"]["wall_seconds"],
                                                                      new["cap"], True, False, None):
        problems.append("%s: section e (wall times, cap, within the cap, stop) differs from the run records" % where)
    want_e = []
    for sd in ("parent", "new"):
        for a in ABL_ARMS:
            cards = [runs[sd]["training"][model_name(a, k, s)] for k in folds["folds"] for s in folds["seeds"]]
            want_e.append((sd, ARM_LABELS[a], len(cards), sum(t["epochs_run"] for t in cards), sum(t["train_seconds"] for t in cards)))
    got_e = [(x.get("run"), x.get("arm"), x.get("models"), x.get("epochs"), x.get("train_seconds")) if isinstance(x, dict) else None
             for x in se.get("per_arm") or []]
    if len(got_e) != len(want_e) or any(g is None or g[:4] != w[:4] or not isinstance(g[4], (int, float)) or abs(g[4] - w[4]) > 1e-6
                                        for g, w in zip(got_e, want_e)):
        problems.append("%s: section e per arm (models, epochs, train seconds) differs from the training cards" % where)

    # Modelli con lo stesso checkpoint: valori di test e soglia dev identici nelle due run
    same_ckpt = [r for r in models if r["parent"]["restored_epoch"] == r["new"]["restored_epoch"]]
    for r in same_ckpt:
        a, k, s = r["arm"], r["fold"], r["seed"]
        if parent["raw"][(a, k, s)] != new["raw"][(a, k, s)] or \
                parent["training"][model_name(a, k, s)]["dev_threshold"] != new["training"][model_name(a, k, s)]["dev_threshold"]:
            problems.append("%s %s: same checkpoint in both runs, but test values or dev threshold differ" % (where, model_label(a, k, s)))
    return {"models": models, "labels": labels, "ten": ten, "patience": patience, "changed": changed, "max_epochs": max_epochs,
            "criteria_text": crit_text, "rec_criterion": (rec_acc, rec_loss), "d_label": sd_["label"], "same_ckpt": len(same_ckpt),
            "verdict": verdict}


def span(vals: list) -> str:
    """«a-b», o «a» se i limiti coincidono."""
    return ("%d" % min(vals)) if min(vals) == max(vals) else "%d-%d" % (min(vals), max(vals))


def fmt_lr(v: float) -> str:
    """Learning rate in decimali fissi senza zeri finali (0.0005, 0.000125, 0.0000625)."""
    return ("%.10f" % v).rstrip("0").rstrip(".")


def ablation_rows(run: dict, patience: int, folds: dict, T: dict, src: str) -> list:
    """Righe della tabella unica: per braccio e seed, i 4 modelli dei fold aggregati (strato all), soglie 0,5 e dev, bootstrap."""
    rows = []
    for a in ABL_ARMS:
        for s in folds["seeds"]:
            cards = [run["training"][model_name(a, k, s)] for k in folds["folds"]]
            b = run["boot"].get((a, s, "all"))
            rows.append(row(row="SESSABL-P%d-%s-seed%s" % (patience, ARM_IDS[a], s), block=ABL_BLOCK % patience,
                            model="LivenessNet, %s, the %d fold models of seed %s pooled over the held-out folds; early-stopping "
                                  "patience %d (epochs run %s, restored epochs %s)"
                                  % (ARM_TEXTS[a].split(":")[0], len(cards), s, patience, span([t["epochs_run"] for t in cards]),
                                     span([t["restored_epoch"] for t in cards])),
                            training_data=T["abl_train_" + a], early_stopping=T["abl_dev_p%d" % patience],
                            test_set=T["folds_test_all"], extractor=EXTRACTOR_FOLDS, threshold="0.5", threshold_origin=THR_ABL_ORIGIN,
                            threshold_dev=THR_DEV_POOLED, run_id=run["agg"]["run_id"], transfer=TRANSFER_ABL,
                            bootstrap=("ACER 0.5 %s; ACER dev %s; AUROC %s" % (iv(b, "acer_05"), iv(b, "acer_dev"), iv(b, "auroc"))
                                       if b else "NA"),
                            source=src, **folds["metric_cells"](run["pooled"][(a, s, "all")])))
    return rows


def abl_texts(run: dict, patience: dict, max_epochs: int) -> dict:
    """Testi delle righe di ablazione, costruiti dalle celle del record (uguali nelle due run: controllato da build)."""
    def cell(c):
        return c.replace("bona_fide", "bf").replace("attack", "att")
    T = {}
    for a in ABL_ARMS:
        c = run["cells"][a]
        T["abl_train_" + a] = ("NUAA, %s: %s frames per fold, %s per class x session cell (%s; deterministic spread rule)"
                               % (ARM_TEXTS[a], fmt(run["n_fit"][a]), "/".join(str(v) for v in c.values()),
                                  ", ".join(cell(x) for x in c)))
    dv = run["dev_cells"]
    per_cell = ("%d" % next(iter(dv.values()))) if len(set(dv.values())) == 1 else "/".join(str(v) for v in dv.values())
    for side, p in patience.items():
        T["abl_dev_p%d" % p] = ("common dev of the fold (sessions 01-02 of the subjects of the next held-out fold, no session 03): "
                                "%s frames, %s per class x session cell (%s); early stopping with patience %d, at most %d epochs, "
                                "minimum dev val_loss restored; learning-rate schedule; EER threshold"
                                % (fmt(run["n_dev"]), per_cell, ", ".join(cell(x) for x in dv), p, max_epochs))
    T["abl_dev_legend"] = ("common dev of the fold (sessions 01-02 of the subjects of the next held-out fold, no session 03, %s "
                           "frames) for early stopping, learning-rate schedule and EER threshold; patience %s"
                           % (fmt(run["n_dev"]), " and ".join("%d in the SESSABL-P%d rows" % (p, p) for p in patience.values())))
    return T


def yn(v) -> str:
    return "yes" if v else "no"


def ablation_annex(runs: dict, pc: dict, folds: dict) -> dict:
    """Annessi: tabella della pazienza (36 modelli, train/dev, poi test) e tabella dell'ablazione (aggregati e verdetti)."""
    P4, P12 = pc["patience"]["parent"], pc["patience"]["new"]
    seeds, fl = folds["seeds"], folds["folds"]
    # Tabella 1 della pazienza: train/dev per modello
    t1, t2 = [], []
    for r in pc["models"]:
        a, k, s = r["arm"], r["fold"], r["seed"]
        cells = [ARM_LABELS[a], k, s, yn(r["in_ten"])]
        for sd in ("parent", "new"):
            x, d = r[sd], r["derived"][sd]
            cells += [x["epochs_run"], x["restored_epoch"], yn(x["degenerate"]),
                      "—" if x["first_recovery"] is None else x["first_recovery"], yn(d["holding_last"]), "%.4f" % x["best_val_loss"],
                      "%.4f" % x["dev_acc_at_checkpoint"], "%.4f" % x["dev_auroc_at_checkpoint"], x["epochs_at_half"], fmt_lr(x["lr_last"])]
        cells += [r["shared_epochs"], yn(r["shared_identical"]), yn(r["parent"]["restored_epoch"] != r["new"]["restored_epoch"])]
        t1.append(cells)
        tc = [ARM_LABELS[a], k, s, yn(r["in_ten"]), yn(r["parent"]["restored_epoch"] != r["new"]["restored_epoch"])]
        for sd in ("parent", "new"):
            for st in ("03", "all"):
                m = runs[sd]["raw"][(a, k, s)][st]
                tc += [f4(m["acer_05"]), f4(m["acer_dev"]), f4(m["auroc"])]
        t2.append(tc)

    # Righe di riepilogo calcolate dalle righe
    M = pc["models"]
    ten = [r for r in M if r["in_ten"]]
    changed = [r for r in M if r["parent"]["restored_epoch"] != r["new"]["restored_epoch"]]
    n = len(M)

    def cnt(rs, f):
        return sum(1 for r in rs if f(r))

    def rng(vals, form="%g"):
        return "%s (%s-%s)" % (form % statistics.median(vals), form % min(vals), form % max(vals))
    # Una riga per fatto, al più SUMMARY_WIDTH caratteri con il trattino (le righe stanno alla lettera in SYNTHESIS.md)
    def by_run(head, f):
        return ["%s with patience %d: %s." % (head, p, f(sd)) for sd, p in (("parent", P4), ("new", P12))]
    lines1 = ["Models: %d (%d arms x %d folds x %d seeds); complete: %d with patience %d, %d with patience %d."
              % (n, len(ABL_ARMS), len(fl), len(seeds), n, P4, cnt(M, lambda r: r["new_status"] == "complete"), P12),
              "In the ten original degenerate trainings: %d." % len(ten)]
    lines1 += by_run("Degenerate", lambda sd: "%d of %d (%d of the ten)" % (cnt(M, lambda r: r[sd]["degenerate"]), n,
                                                                             cnt(ten, lambda r: r[sd]["degenerate"])))
    lines1 += ["Shared histories identical: %d of %d (shared epochs = epochs run with patience %d in %d of %d)."
               % (cnt(M, lambda r: r["shared_identical"]), n, P4, cnt(M, lambda r: r["shared_epochs"] == r["parent"]["epochs_run"]), n)]
    lines1 += by_run("Epochs run", lambda sd: "%s, %d of %d models at %d" % (
        span([r[sd]["epochs_run"] for r in M]), cnt(M, lambda r: r[sd]["epochs_run"] == pc["max_epochs"]), n, pc["max_epochs"]))
    lines1 += by_run("With a recovery (an epoch meeting the recovery criterion)", lambda sd: "%d of %d (%d of the ten)" % (
        cnt(M, lambda r: r[sd]["recovered"]), n, cnt(ten, lambda r: r[sd]["recovered"])))
    lines1 += by_run("Recovery holds at the last epoch", lambda sd: "%d of %d (%d of the ten)" % (
        cnt(M, lambda r: r["derived"][sd]["holding_last"]), n, cnt(ten, lambda r: r["derived"][sd]["holding_last"])))
    lines1 += ["First recovery of the ten with patience %d: median (min-max) %s."
               % (P12, rng([r["new"]["first_recovery"] for r in ten if r["new"]["first_recovery"] is not None])),
               "First recovery of the ten at epoch %d or later with patience %d: %d of the ten (cut-off of CONCLUSION.md)."
               % (LATE_RECOVERY_EPOCH, P12, cnt(ten, lambda r: r["new"]["first_recovery"] is not None
                                               and r["new"]["first_recovery"] >= LATE_RECOVERY_EPOCH)),
               "Restored epoch changed: %d of %d (%d of the ten, %d of the other %d)."
               % (len(changed), n, cnt(changed, lambda r: r["in_ten"]), cnt(changed, lambda r: not r["in_ten"]), n - len(ten)),
               "Stopped before epoch %d with patience %d: %d models, %d outside the ten, of which %d changed restored epoch."
               % (pc["max_epochs"], P4, cnt(M, lambda r: r["parent"]["epochs_run"] < pc["max_epochs"]),
                  cnt(M, lambda r: r["parent"]["epochs_run"] < pc["max_epochs"] and not r["in_ten"]),
                  cnt(changed, lambda r: r["parent"]["epochs_run"] < pc["max_epochs"] and not r["in_ten"]))]
    for group, rows_ in (("the ten", ten), ("the other %d" % (n - len(ten)), [r for r in M if not r["in_ten"]])):
        lines1 += by_run("Dev accuracy at the checkpoint, %s, median (min-max)" % group,
                         lambda sd: rng([r[sd]["dev_acc_at_checkpoint"] for r in rows_], "%.4f"))
    for group, rows_ in (("the ten", ten), ("the other %d" % (n - len(ten)), [r for r in M if not r["in_ten"]])):
        lines1.append("Dev accuracy at the checkpoint below %g with patience %d, %s: %d (cut-off of CONCLUSION.md)."
                      % (LOW_DEV_ACCURACY, P12, group, cnt(rows_, lambda r: r["new"]["dev_acc_at_checkpoint"] < LOW_DEV_ACCURACY)))
    lines2 = ["Same test values and dev threshold in both runs (checked): all %d models whose restored epoch did not change."
              % (n - len(changed)),
              "The differences below come from the %d models whose restored epoch changed (\"changed\" below)." % len(changed)]
    for st in ("03", "all"):
        for key, label in (("acer_05", "ACER at 0.5"), ("acer_dev", "ACER at the dev threshold"), ("auroc", "AUROC")):
            def cmpv(r):
                x, y = runs["parent"]["raw"][(r["arm"], r["fold"], r["seed"])][st][key], runs["new"]["raw"][(r["arm"], r["fold"], r["seed"])][st][key]
                return None if x is None or y is None else (y > x) - (y < x)
            for sign, word in ((-1, "lower"), (0, "equal"), (1, "higher")):
                lines2.append("Stratum %s, %s, patience %d vs %d: %s in %d of %d (%d of the %d changed, %d of the ten)."
                              % (st, label, P12, P4, word, cnt(M, lambda r: cmpv(r) == sign), n, cnt(changed, lambda r: cmpv(r) == sign),
                                 len(changed), cnt(ten, lambda r: cmpv(r) == sign)))
            na = cnt(M, lambda r: cmpv(r) is None)
            if na:
                lines2.append("Stratum %s, %s, patience %d vs %d: NA in %d of %d (a class absent)." % (st, label, P12, P4, na, n))
    for l in lines1 + lines2:
        if len("- " + l) > SUMMARY_WIDTH:
            fail("summary line longer than %d characters: %s" % (SUMMARY_WIDTH, l))
    # Tabella dell'ablazione: aggregati per braccio, seed, strato e pazienza; mediane; verdetti affiancati
    pooled_rows, summary_rows, verdict_rows = [], [], []
    for a in ABL_ARMS:
        for s in seeds:
            for st in FOLDS_STRATA:
                for sd, p in (("parent", P4), ("new", P12)):
                    run = runs[sd]
                    m, b, nsub = run["pooled"][(a, s, st)], run["boot"].get((a, s, st)), run["praw"][(a, s, st)].get("n_subjects")
                    x, d = m["05"], m["dev"]
                    pooled_rows.append([ARM_LABELS[a], s, st, p, x["n_bona_fide"], x["n_attack"], nsub,
                                        "%s / %s" % (x["rejected_bona_fide"], x["accepted_attacks"]), f4(x["bpcer"]), f4(x["apcer"]),
                                        "%s [%s]" % (f4(x["acer"]), iv(b, "acer_05") if b else "NA"),
                                        "%s / %s" % (d["rejected_bona_fide"], d["accepted_attacks"]), f4(d["bpcer"]), f4(d["apcer"]),
                                        "%s [%s]" % (f4(d["acer"]), iv(b, "acer_dev") if b else "NA"),
                                        "%s [%s]" % (f4(x["auroc"]), iv(b, "auroc") if b else "NA")])
    for a in ABL_ARMS:
        for st in FOLDS_STRATA:
            for sd, p in (("parent", P4), ("new", P12)):
                e = runs[sd]["agg"]["pooled_summary_over_seeds"][a][st]
                summary_rows.append([ARM_LABELS[a], st, p] + ["%s (%s-%s)" % (f4(e[k]["median"]), f4(e[k]["min"]), f4(e[k]["max"]))
                                                                for k in ("acer_05", "acer_dev", "auroc")])

    def vtext(v):
        return "%s (%s)" % (v["verdict"], v["direction"] if v["direction"] else v["reason"])

    def stext(v):
        return "; ".join("%s [%s, %s]" % (f4(d), f4(lo), f4(hi)) for d, lo, hi in v["per_seed"])
    counts = {sd: {} for sd in ("parent", "new")}
    for c, st, m in [(c[0], st, m) for c in ABL_COMPARISONS for st in FOLDS_STRATA for m in FOLDS_SUMMARY_KEYS]:
        vp, vn = runs["parent"]["verdicts"][(c, st, m)], runs["new"]["verdicts"][(c, st, m)]
        verdict_rows.append([COMPARISON_LABELS[c], st, m, vp["label"], vtext(vp), stext(vp), f4(vp["median"]), vtext(vn), stext(vn),
                             f4(vn["median"]), "yes" if (c, st, m) in pc["changed"] else "—"])
        for sd, v in (("parent", vp), ("new", vn)):
            counts[sd][v["verdict"]] = counts[sd].get(v["verdict"], 0) + 1
    nv = len(verdict_rows)
    lines3 = ["Verdicts: %d per run (%d comparisons x %d strata x %d metrics); patience %d: %s; patience %d: %s."
              % (nv, len(ABL_COMPARISONS), len(FOLDS_STRATA), len(FOLDS_SUMMARY_KEYS), P4,
                 ", ".join("%s %d" % (k, v) for k, v in sorted(counts["parent"].items())), P12,
                 ", ".join("%s %d" % (k, v) for k, v in sorted(counts["new"].items())))]
    for sd, p in (("parent", P4), ("new", P12)):
        pv = [runs[sd]["verdicts"][("a_star_minus_b", PRIMARY_STRATUM, m)]["verdict"] for m in PRIMARY_METRICS]
        lines3.append("Primary verdicts (A* - B, stratum %s: %s) with patience %d: %s."
                      % (PRIMARY_STRATUM, ", ".join(PRIMARY_METRICS), p, ", ".join(pv)))
    lines3.append("Verdicts that differ between the two runs: %d of %d: %s." % (
        len(pc["changed"]), nv, "; ".join("%s stratum %s %s (%s -> %s)" % (COMPARISON_LABELS[c], st, m, runs["parent"]["verdicts"][(c, st, m)]["verdict"],
                                                                           runs["new"]["verdicts"][(c, st, m)]["verdict"]) for c, st, m in pc["changed"])))
    compact = [["%s fold %s seed %s" % (c[0], c[1], c[2]), "yes" if c[3] == "yes" else "—", "%s → %s" % (c[5], c[15]),
                "%s → %s" % (c[7], c[17]), "%s → %s" % (c[8], c[18]), "%s → %s" % (c[10], c[20]), "yes" if c[26] == "yes" else "—"]
               for c in t1]
    return {"patience_train_dev": t1, "patience_test": t2, "patience_lines": lines1, "test_lines": lines2, "patience_compact": compact,
            "pooled": pooled_rows, "pooled_summary": summary_rows, "verdicts": verdict_rows, "verdict_lines": lines3}


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

    # Run di ablazione delle sessioni (PR #21, pazienza 4) e run di verifica con pazienza 12 (PR #23), confronto delle due
    runs, statuses = {}, {}
    for side, pre in (("parent", "sessabl"), ("new", "patience")):
        keys = {"aggregate": pre + "_aggregate", "record": pre + "_record", "sums": pre + "_sums", "status": pre + "_status"}
        agg_ = read_json(paths[keys["aggregate"]], ABL_AGG_FIELDS, clean=False)
        rec_ = read_json(paths[keys["record"]], ABL_REC_FIELDS, clean=False)
        st_ = read_json(paths[keys["status"]], ABL_STATUS_FIELDS, clean=False)
        sums_ = read_sums(paths[keys["sums"]], SOURCES[keys["sums"]])
        runs[side] = ablation_run("%s run" % ("session ablation" if side == "parent" else "patience verification"), agg_, rec_, st_,
                                  sums_, hashes, keys, folds, fsch, problems)
        runs[side]["keys"], statuses[side] = keys, st_
    if runs["parent"]["agg"]["run_id"] == runs["new"]["agg"]["run_id"]:
        fail("session ablation and patience verification: the same run %s" % runs["new"]["agg"]["run_id"])
    if runs["parent"]["agg"]["reading_rule"] != runs["new"]["agg"]["reading_rule"]:
        problems.append("patience verification run: reading rule differs from the session ablation run")
    for k in ("cells", "dev_cells", "n_fit", "n_dev"):
        if runs["parent"][k] != runs["new"][k]:
            problems.append("patience verification run: %s differ from the session ablation run (same 36 models expected)" % k)
    pv_agg = read_json(paths["patver_aggregate"], PATVER_FIELDS, clean=False)
    pv_status = read_json(paths["patver_status"], PATVER_STATUS_FIELDS, clean=False)
    pv_sums = read_sums(paths["patver_sums"], SOURCES["patver_sums"])
    pv_rows = read_csv_rows(paths["patver_comparison"], CMP_COLUMNS, SOURCES["patver_comparison"])
    pv_curves = read_csv_rows(paths["patver_curves"], CURVE_COLUMNS, SOURCES["patver_curves"])
    pc = patience_compare(pv_agg, pv_status, pv_sums, pv_rows, pv_curves, runs, hashes, statuses, folds, problems)
    T.update(abl_texts(runs["parent"], pc["patience"], pc["max_epochs"]))
    for side, pre in (("parent", "sessabl"), ("new", "patience")):
        rows.extend(ablation_rows(runs[side], pc["patience"][side], folds, T, "%s; %s; %s" % (src(pre + "_aggregate"), src(pre + "_record"),
                                                                                             src("folds_schema"))))
    if problems:  # gli annessi leggono strutture che i controlli sopra hanno trovato incoerenti: fermarsi prima
        fail("consistency problems:\n  - " + "\n  - ".join(problems))
    abl_annex = ablation_annex(runs, pc, folds)
    sources = {k: {"path": SOURCES[k], "sha256": hashes[k]} for k in SOURCES}

    if problems:
        fail("consistency problems:\n  - " + "\n  - ".join(problems))

    doc = {
        "schema": SCHEMA,
        "what": "unified table of the accepted NUAA results (C1, M1, N1, N2, C1 crop check, subject-held-out folds run, session ablation run with early-stopping patience 4 and 12), rebuilt from the public aggregates of research/pad-2026; no restricted data, no inference, no new number",
        "sources": sources,
        "runs": dict({"C1": C1_RUN, "M1": M1_RUN, "N1": n1["run_id"], "N1_subjects": n1s["run_id"], "N2_sessions": n2s["run_id"],
                      "N2_combined": n2c["run_id"], "C1_crop_check": crop["run_id"], "Fold_schema": fsch["run_id"],
                      "Folds_run": fagg["run_id"], "Folds_vs_N1": cmp_agg["run_id"],
                      "Session_ablation_run": runs["parent"]["agg"]["run_id"], "Patience_12_run": runs["new"]["agg"]["run_id"],
                      "Patience_comparison": pv_agg["run_id"]}),
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
        "session_ablation": {
            "runs": {side: {"run_id": runs[side]["agg"]["run_id"], "commit": runs[side]["agg"]["commit"],
                            "config_sha256": runs[side]["agg"]["config_sha256"], "finished": runs[side]["agg"]["finished"],
                            "early_stopping_patience": pc["patience"][side], "wall_seconds": runs[side]["rec"]["wall_seconds"],
                            "wall_cap_seconds": runs[side]["cap"], "models": len(runs[side]["names"]),
                            "amendment": runs[side]["agg"].get("amendment")} for side in ("parent", "new")},
            "comparison_run": {"run_id": pv_agg["run_id"], "commit": pv_agg["commit"], "config_sha256": pv_agg["config_sha256"],
                               "verdict": pc["verdict"], "max_epochs": pc["max_epochs"], "criteria": list(pc["criteria_text"]),
                               "test_section_label": pc["d_label"]},
            "bootstrap": runs["parent"]["agg"]["bootstrap"],
            "reading_rule": runs["parent"]["agg"]["reading_rule"],
            "verdicts_changed": [list(k) for k in pc["changed"]],
            "annex": abl_annex,
        },
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
    out.write("# Unified table of the accepted NUAA results (C1, M1, N1, N2, C1 crop check, subject-held-out folds run, "
              "session ablation run with early-stopping patience %s)\n\n" % " and ".join(
                  str(doc["session_ablation"]["runs"][sd]["early_stopping_patience"]) for sd in ("parent", "new")))
    out.write("Built by `nuaa_synthesis.py build` from the public aggregates merged in `research/pad-2026` "
              "(sources and SHA-256 in `summary_table.json`); no restricted data, no inference, no new number. "
              "%s. APCER = accepted attacks / attacks, BPCER = rejected bona fide / bona fide, ACER = their mean, "
              "AUROC threshold-free (the column named `auc` in the C1 table and `test AUC` in the M1 table: the same Mann-Whitney statistic). "
              "Rates as printed in the C1 and M1 tables (four decimals; M1 thresholds six) or rounded to four decimals "
              "from the exact values of the JSON aggregates (N1, N2, crop check, folds run, session ablation runs). "
              "The columns from `dev threshold` to `bootstrap` apply to the folds-run and session-ablation rows only (the dev "
              "threshold next to 0.5; the bootstrap, 1,000 resamples of the held-out subjects, for the rows pooled over the four "
              "folds); \u2014 elsewhere. "
              "Runs: %s.\n\n" % (doc["decision"], ", ".join("%s `%s`" % (k, v) for k, v in doc["runs"].items())))
    out.write(md_table(cols, doc["rows"], labels, empty="\u2014"))
    out.write("\nSource of every row: `summary_table.csv` (column `source`) and `summary_table.json`.\n")
    return out.getvalue()


SHORT_BLOCK = {"C1": "C1", "M1": "M1", "N1": "N1", "N2 sessions": "N2-S", "N2 combined": "N2-C", "C1 crop check": "CROP",
               FOLDS_BLOCK: "FOLDS", COMPARE_BLOCK: "FOLDS-N1"}
TRANSFER_CODES = {TRANSFER_SAME: "same", TRANSFER_SESSION_ALL: "session", TRANSFER_SESSION_9: "session/9",
                  TRANSFER_DATASET: "dataset", TRANSFER_FOLDS: "subject-held-out (sessions represented)",
                  TRANSFER_FOLDS_N1: "subject-held-out, N1 test frames", TRANSFER_ABL: "subject-held-out (session ablation)"}


def short_block(block: str) -> str:
    """Codice breve del blocco; i blocchi dell'ablazione portano la pazienza («Session ablation run, patience 4» -> SESSABL-P4)."""
    mm = re.fullmatch(ABL_BLOCK.replace("%d", r"(\d+)"), block)
    return "SESSABL-P" + mm.group(1) if mm else SHORT_BLOCK[block]


def is_abl(r: dict) -> bool:
    return re.fullmatch(ABL_BLOCK.replace("%d", r"\d+"), r["block"]) is not None


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
    if is_abl(r):
        mm = re.fullmatch(r"SESSABL-P\d+-(\w+)-seed(\d+)", r["row"])
        arm = {v: k for k, v in ARM_IDS.items()}.get(mm.group(1)) if mm else None
        if arm:
            return "LivenessNet, arm %s, 4 fold models, seed %s (pooled)" % (ARM_LABELS[arm], mm.group(2))
    fail("compact table: block %r" % r["block"])


def compact_legend(doc: dict) -> dict:
    T = doc["texts"]
    folds = sorted(int(k[len("folds_test_"):]) for k in T if re.fullmatch(r"folds_test_\d+", k))
    test = {"S-300": T["test_c1"], "S-3000": T["test_3000"], "S-491": T["target_491"], "S-03/9": T["s03_9"]}
    test.update({"S-FOLD-%d" % k: T["folds_test_%d" % k] for k in folds})
    test["S-FOLD-ALL"] = T["folds_test_all"]
    return {
        "training": dict({"T-NUAA": T["train"] + "; early stopping on " + T["es"],
                          "T-COMB": TRAIN_COMB + "; historical early stopping",
                          "T-FOLD": T["folds_legend"] + "; the dev fold is the next held-out fold and drives early stopping, the "
                                    "learning-rate schedule and the EER threshold; frames drawn with the deterministic spread rule"},
                         **{"T-ABL-" + ARM_LABELS[a]: T["abl_train_" + a] + "; " + T["abl_dev_legend"] for a in ABL_ARMS}),
        "test": test,
        "extractor": {"E-C1": "c1_a80f6f9: the C1 evaluation extractor (RetinaFace box 1 px narrower and shorter than the cache)",
                      "E-CACHE": "cache_0ac7143: the training-cache extractor (cache frames, or the frozen copy on new frames)",
                      "E-WHOLE": VARIANTS["whole_image"], "E-CROP": VARIANTS["face_crop"]},
        "preprocessing": {"P-REF": PREP_REF, "P-HIST": PREP_HIST},
        "threshold": {"0.5 (fixed)": "fixed a priori (C1, N1, N2, crop check, folds run, session ablation runs)",
                      "(dev EER)": "M1: " + T["thr_dev"] + ", frozen before any test score was read",
                      "(EER, fold dev)": "folds run, column \"dev thr.\": " + THR_FOLDS_ORIGIN.split("; dev threshold: ")[1]
                                         + " (\"" + THR_DEV_POOLED + "\" in the pooled rows: each frame at the threshold of its fold model)",
                      "(EER, common dev)": "session ablation runs, column \"dev thr.\": " + THR_ABL_ORIGIN.split("; dev threshold: ")[1]
                                           + " (\"" + THR_DEV_POOLED + "\": each frame at the threshold of its fold model)"},
        # il codice del confronto con N1 compare solo quando ci sono le sue righe (sorgente facoltativa)
        "transfer": {v: k for k, v in TRANSFER_CODES.items()
                     if k != TRANSFER_FOLDS_N1 or any(r["transfer"] == k for r in doc["rows"])},
    }


def compact_rows(doc: dict) -> list:
    T = doc["texts"]
    out = []
    tests = {v: k for k, v in compact_legend(doc)["test"].items()}
    for r in doc["rows"]:
        abl_train = {T["abl_train_" + a]: "T-ABL-" + ARM_LABELS[a] for a in ABL_ARMS}
        training = ("T-COMB" if r["training_data"] == TRAIN_COMB else "T-FOLD" if r["training_data"] == T.get("folds_train")
                    else abl_train[r["training_data"]] if r["training_data"] in abl_train else "T-NUAA")
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
        out.append({"block": short_block(r["block"]), "model": short_model(r), "training": training, "test": test,
                    "extractor": extractor, "preprocessing": prep, "threshold": thr,
                    "errors": "%s / %s" % (r["rejected_bona_fide"], r["accepted_attacks"]),
                    "apcer": r["apcer"], "bpcer": r["bpcer"], "acer": r["acer"], "auroc": r["auroc"],
                    "threshold_dev": (r["threshold_dev"] + (" (EER, common dev)" if is_abl(r) else " (EER, fold dev)")) if dev else "",
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
              "FOLDS (subject-held-out folds run: one row per held-out fold and seed, stratum all, and one row per seed pooled over the four folds), "
              "FOLDS-N1 (the fold models on the 3,000 N1 test frames), %s (session ablation run and its verification run with "
              "another early-stopping patience: one row per arm and seed pooled over the four folds, stratum all; every model, "
              "stratum and verdict in `session_ablation_table.md` and `patience_table.md`); "
              "ckpt = checkpoint; \"rejected bf / accepted att.\" = rejected bona fide / accepted attacks, the numerators of BPCER and APCER. "
              "The columns from \"dev thr.\" to \"bootstrap\" apply to the FOLDS, FOLDS-N1 and SESSABL rows only (\u2014 elsewhere): the errors and rates "
              "at the dev threshold of each model, and for the pooled rows the 2.5-97.5 percentiles of 1,000 bootstrap resamples "
              "of the held-out subjects (seed 0) for ACER at 0.5, ACER at the dev threshold and AUROC. "
              "Unit = image (frame) in every row; %s; APCER = accepted attacks / attacks, BPCER = rejected bona fide / bona fide, "
              "ACER = their mean, AUROC threshold-free (`auc` in the C1 table, `test AUC` in the M1 table).\n\n"
              % (" and ".join("SESSABL-P%d" % doc["session_ablation"]["runs"][sd]["early_stopping_patience"] for sd in ("parent", "new")),
                 doc["decision"]))
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


def plain_table(head: list, rows: list) -> str:
    return ("| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n"
            + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows))


def patience_md(doc: dict) -> str:
    """Annesso della pazienza: i 36 modelli, train/dev con le due pazienze affiancate, poi il test per modello."""
    sa, a = doc["session_ablation"], doc["session_ablation"]["annex"]
    rp, rn, c = sa["runs"]["parent"], sa["runs"]["new"], sa["comparison_run"]
    P4, P12 = rp["early_stopping_patience"], rn["early_stopping_patience"]
    side = ["epochs run", "restored epoch", "degenerate", "first recovery", "recovery holds at last epoch", "best val_loss",
            "dev acc at ckpt", "dev AUROC at ckpt", "epochs at 0.5", "last lr"]
    test = ["ACER 0.5", "ACER dev", "AUROC"]
    return ("# Annex: early-stopping patience %d and %d, the %d LivenessNet models of the session ablation run (runs `%s` and `%s`, "
            "comparison `%s`)\n\n"
            "Train and dev from the public comparison of the two runs (`%s`, one row per model, and `%s`, one row per epoch of the "
            "patience-%d run), checked by `nuaa_synthesis.py`: every value of the comparison except the dev AUROC at the checkpoint "
            "(copied; checked against the JSON rows and the group medians) rebuilt from the epoch curves with the "
            "criteria of the comparison aggregate (restored epoch = first epoch of minimum dev val_loss; epochs run = the early-stopping "
            "replay with the patience of the run, at most %d epochs; degenerate: %s; recovery: %s), equal to the JSON rows of the "
            "comparison and to the training cards of the two public aggregates; the first epochs of the patience-%d run are those "
            "of the patience-%d run (shared histories, identical: the largest absolute differences of the comparison are required "
            "to be 0), so the patience-%d columns are read from the same curves. "
            "\"Recovery holds at last epoch\" = the recovery criterion met at the last epoch run (epoch %d with patience %d). "
            "Checkpoint = restored epoch; dev = the common dev of the fold (sessions 01-02). \"Restored epoch changed\" compares the "
            "restored epochs of the two runs only: with the same restored epoch the weights and threshold file hashes still differ "
            "in every model; identity is shown by identical test values and dev threshold (checked). Comparison verdict: %s.\n\n"
            "## Train and dev, per model (patience %d, then patience %d)\n\n"
            % (P4, P12, rp["models"], rp["run_id"], rn["run_id"], c["run_id"], SOURCES["patver_comparison"], SOURCES["patver_curves"],
               P12, c["max_epochs"], c["criteria"][0], c["criteria"][1], P12, P4, P4, c["max_epochs"], P12, c["verdict"], P4, P12)
            + plain_table(["arm", "fold", "seed", "in the ten"] + ["%s (p%d)" % (h, P4) for h in side] + ["%s (p%d)" % (h, P12) for h in side]
                          + ["shared epochs", "shared identical", "restored epoch changed"], a["patience_train_dev"])
            + "\n" + "".join("- %s\n" % l for l in a["patience_lines"])
            + "\n## Test, per model (%s; patience %d, then patience %d)\n\n"
              "From the public `aggregate.json` of the two runs (`%s`, `%s`): every count and rate checked against the fold schema "
              "and summed into the pooled rows of `session_ablation_table.md`. Strata 03 and all of the held-out subjects of the "
              "fold; 0.5 = the fixed threshold, dev = the EER threshold of the common dev of the model.\n\n"
            % (c["test_section_label"], P4, P12, SOURCES["sessabl_aggregate"], SOURCES["patience_aggregate"])
            + plain_table(["arm", "fold", "seed", "in the ten", "restored epoch changed"]
                          + ["%s %s (p%d)" % (st, h, p) for p in (P4, P12) for st in ("03", "all") for h in test], a["patience_test"])
            + "\n" + "".join("- %s\n" % l for l in a["test_lines"]))


PATIENCE_COMPACT_HEAD = ["model", "in the ten", "restored epoch p%d → p%d", "first recovery p%d → p%d",
                         "recovery holds at last epoch p%d → p%d", "dev acc at ckpt p%d → p%d", "restored epoch changed"]


def patience_compact_md(doc: dict) -> str:
    """Tabella compatta dei 36 modelli (per SYNTHESIS.md): celle prese dalle righe di patience_table.md, poi le stesse
    righe di riepilogo della tabella train/dev."""
    sa, a = doc["session_ablation"], doc["session_ablation"]["annex"]
    P4, P12 = sa["runs"]["parent"]["early_stopping_patience"], sa["runs"]["new"]["early_stopping_patience"]
    head = [h % (P4, P12) if "%d" in h else h for h in PATIENCE_COMPACT_HEAD]
    return ("# Early-stopping patience %d and %d, the %d models in compact form (all columns, sources and checks in "
            "`patience_table.md`)\n\n" % (P4, P12, sa["runs"]["parent"]["models"])
            + plain_table(head, a["patience_compact"])
            + "\nLegend. p%d = run `%s` (early-stopping patience %d), p%d = run `%s` (patience %d); in the ten = one of the ten "
              "degenerate trainings of the p%d run (— = not among the ten); first recovery = first epoch meeting the recovery criterion (— when none); "
              "recovery holds at last epoch = that criterion met at the last epoch run; dev acc at ckpt = dev accuracy at the "
              "restored epoch; restored epoch changed = the two restored epochs differ (— when equal). Train and dev only; the "
              "test per model is in `patience_table.md`.\n\n"
            % (P4, sa["runs"]["parent"]["run_id"], P4, P12, sa["runs"]["new"]["run_id"], P12, P4)
            + "".join("- %s\n" % l for l in a["patience_lines"]))


def synthesis_lines(doc: dict) -> dict:
    """Le righe che SYNTHESIS.md deve contenere alla lettera: la tabella compatta unica (righe di tabella) e la tabella
    compatta della pazienza (righe di tabella e righe di riepilogo)."""
    return {"compact table": [l for l in compact_md(doc).splitlines() if l.startswith("|")],
            "patience compact table": [l for l in patience_compact_md(doc).splitlines() if l.startswith("|") or l.startswith("- ")]}


def ablation_md(doc: dict) -> str:
    """Annesso dell'ablazione: aggregati per braccio, seed e strato nelle due run, mediane, verdetti affiancati."""
    sa, a = doc["session_ablation"], doc["session_ablation"]["annex"]
    rp, rn, b = sa["runs"]["parent"], sa["runs"]["new"], sa["bootstrap"]
    P4, P12 = rp["early_stopping_patience"], rn["early_stopping_patience"]
    summary = ["ACER 05: median (min-max)", "ACER dev: median (min-max)", "AUROC: median (min-max)"]
    return ("# Annex: session ablation run, arms B, A* and B' pooled over the 4 subject folds, early-stopping patience %d and %d "
            "(runs `%s` and `%s`)\n\n"
            "Copied from the public aggregates of the two runs (`%s`, `%s`; every count and rate checked by `nuaa_synthesis.py`: "
            "denominators against the fold schema, stratum all against the sum of the sessions, pooled rows against the sum of "
            "the folds, medians against the per-seed values, differences against the pooled rows, every verdict against the "
            "reading rule applied to the per-seed differences and intervals). The two runs train the same 36 models with the same "
            "cell counts and seeds; per the amendment, only the patience changes (%d -> %d). Test values: %s. %s; `05` = the fixed threshold 0.5, `dev` = the EER threshold of the common "
            "dev, one per model (pooled rows: each frame at the threshold of its fold model). Bootstrap: %d resamples with "
            "replacement of the held-out subjects of the stratum, seed %d, percentiles %s-%s, no interval with fewer than %d "
            "subjects. Frames of one subject and session are correlated: n counts frames, not independent units.\n\n"
            "## Pooled over the 4 folds, per arm, seed, stratum and patience\n\n"
            % (P4, P12, rp["run_id"], rn["run_id"], SOURCES["sessabl_aggregate"], SOURCES["patience_aggregate"], P4, P12,
               sa["comparison_run"]["test_section_label"], doc["decision"], b["resamples"], b["seed"], b["percentiles"][0],
               b["percentiles"][1], b["min_subjects"])
            + plain_table(["arm", "seed", "stratum", "patience", "n bona fide", "n attack", "subjects", "rejected bf / accepted att. (05)",
                           "BPCER 05", "APCER 05", "ACER 05 [bootstrap]", "rejected bf / accepted att. (dev)", "BPCER dev", "APCER dev",
                           "ACER dev [bootstrap]", "AUROC [bootstrap]"], a["pooled"])
            + "\n## Pooled: median and range over the seeds\n\n"
            + plain_table(["arm", "stratum", "patience"] + summary, a["pooled_summary"])
            + "\n## Verdicts of the reading rule, patience %d and patience %d\n\n"
              "Reading rule (copied from the aggregates): %s. Per seed (%s): the pooled difference (minuend - subtrahend) with its "
              "paired bootstrap interval [2.5, 97.5]; median over the seeds; differs = verdict or direction not the same in the "
              "two runs.\n\n"
            % (P4, P12, sa["reading_rule"], ", ".join(doc["folds_run"]["seeds"]))
            + plain_table(["comparison", "stratum", "metric", "label", "verdict (p%d)" % P4, "per seed (p%d)" % P4, "median (p%d)" % P4,
                           "verdict (p%d)" % P12, "per seed (p%d)" % P12, "median (p%d)" % P12, "differs"], a["verdicts"])
            + "\n" + "".join("- %s\n" % l for l in a["verdict_lines"]))


def duplicate_rows(files: dict) -> list:
    """Righe duplicate dentro una stessa tabella Markdown di un file generato (ogni riga di una tabella deve essere unica)."""
    out = []
    for name, text in files.items():
        if not name.endswith(".md"):
            continue
        seen = set()
        for line in text.splitlines():
            if not line.startswith("|"):
                seen = set()
                continue
            if line in seen:
                out.append("%s: duplicated row %s" % (name, line[:80]))
            seen.add(line)
    return out


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
        "patience_table.md": patience_md(doc),
        "session_ablation_table.md": ablation_md(doc),
        "patience_table_compact.md": patience_compact_md(doc),
    }


def checked_outputs(doc: dict) -> dict:
    """I file generati, fermandosi se una tabella Markdown ha una riga duplicata."""
    files = outputs(doc)
    dup = duplicate_rows(files)
    if dup:
        fail("duplicated Markdown rows:\n  - " + "\n  - ".join(dup))
    return files


def cmd_build(a) -> int:
    doc = build(Path(a.research))
    files = checked_outputs(doc)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        with open(out / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    print("BUILD: %d rows, %d cross-dataset rows, %d context rows, %d subject rows, %d folds-run stratum rows, %d patience model rows, "
          "%d session-ablation pooled rows, %d verdict rows -> %s" %
          (len(doc["rows"]), len(doc["cross_dataset_rows"]), len(doc["context_rows"]), len(doc["subject_sensitivity"]),
           len(doc["folds_run"]["annex"]["per_fold_seed"]), len(doc["session_ablation"]["annex"]["patience_train_dev"]),
           len(doc["session_ablation"]["annex"]["pooled"]), len(doc["session_ablation"]["annex"]["verdicts"]), out))
    print("RESULT: PASS")
    return 0


def cmd_check(a) -> int:
    doc = build(Path(a.research))
    files = checked_outputs(doc)
    out = Path(a.out)
    bad = []
    for name, text in files.items():
        p = out / name
        if not p.is_file():
            bad.append("%s: missing" % name)
        elif p.read_bytes() != text.encode("utf-8"):
            bad.append("%s: differs from the rebuilt table (byte comparison)" % name)
    if a.synthesis:
        present = set(Path(a.synthesis).read_text(encoding="utf-8").splitlines())
        for what, lines in synthesis_lines(doc).items():
            missing = [l for l in lines if l not in present]
            if missing:
                bad.append("%s: %d of %d lines of the %s are not in the document verbatim (first: %s)"
                           % (Path(a.synthesis).name, len(missing), len(lines), what, missing[0][:60]))
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
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, OSError, csv.Error) as e:
        print("ERROR: a source has an unexpected structure or cannot be read: %s: %s" % (type(e).__name__, e))
    print("RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
