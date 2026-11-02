"""
Sintesi NUAA (C1, M1, N1, N2, confronto del ritaglio): ricostruisce la tabella unica dei risultati accettati leggendo
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

Sottocomandi:
  build --research <research/> --out <cartella>   scrive summary_table.{md,csv,json}, summary_table_compact.md,
                                                  cross_dataset_table.md, context_rows_table.md,
                                                  subject_sensitivity_table.md (UTF-8, fine riga LF)
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
alla tabella M1. Il controllo copre le celle delle tabelle; i numeri della prosa del documento restano da verificare a
mano contro le tabelle e le fonti.

Dipendenze: solo la libreria standard.
"""
import argparse
import csv
import hashlib
import io
import json
import re
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
CONTEXT_ROWS = {
    "minifasnet": ("MiniFASNet (Silent-Face, pre-trained)", "not trained on NUAA"),
    "clip_zeroshot": ("CLIP ViT-B/32 zero-shot", "not trained on NUAA"),
    "clip_probe": ("linear probe on CLIP ViT-B/32", "all 3,000 sampled NUAA training images (dev included)"),
    "dinov2_probe": ("linear probe on DINOv2-small", "all 3,000 sampled NUAA training images (dev included)"),
    "depth": ("Depth Anything V2 (depth statistics)", "not trained on NUAA"),
    "lbp": ("LBP + SVM", "all 3,000 sampled NUAA training images (dev included)"),
    "fourier": ("Fourier spectrum rule", "not trained on NUAA"),
    "ida": ("IDA features + SVM", "all 3,000 sampled NUAA training images (dev included)"),
    "dog": ("DoG features + SVM", "all 3,000 sampled NUAA training images (dev included)"),
    "iqa": ("IQA features + SVM", "all 3,000 sampled NUAA training images (dev included)"),
}

C1_RUN = "20260928-C1-seed42-a80f6f9"
M1_RUN = "20261003-M1-seed42-e67ecc7"

EXTRACTOR_C1 = "c1_a80f6f9"
EXTRACTOR_CACHE = "cache_0ac7143"
THR_FIXED = "fixed a priori (C1)"
PREP_REF = "detector crop (margin 0.15, square; whole frame without a face), 256x256 INTER_AREA, BGR to RGB, reference enhancement, /255 in [0, 1]"
PREP_HIST = "historical chain of the combined model: BGR to RGB, 256x256 Lanczos, reference enhancement, (x/255 - 0.5) * 2 in [-1, 1]"
TRAIN_COMB = "3DMAD, CSMAD, Replay-Attack, MSSpoof, Our Dataset (historical code; no NUAA)"
TRANSFER_SESSION_ALL = "session transfer (01-02 to 03; the training subjects and the test-only subjects)"
TRANSFER_SESSION_9 = "session transfer (01-02 to 03; the training subjects only)"
TRANSFER_SAME = "same sessions as training (01-02), fitting and early-stopping subjects, frames never used"
TRANSFER_DATASET = "dataset transfer (zero-shot: model never trained on NUAA)"
VARIANTS = {"whole_image": "whole image (no face detection)", "face_crop": "face crop (current detector, margin 0.15, square; whole frame without a face)"}
EXTRACTOR_LABELS = {EXTRACTOR_CACHE: "cache_0ac7143 (the training-cache extractor)",
                    EXTRACTOR_C1: "c1_a80f6f9 (the C1 extractor; box 1 px narrower and shorter)"}

COLUMNS = ["row", "block", "model", "training_data", "early_stopping", "test_set", "unit", "extractor", "preprocessing",
           "threshold", "threshold_origin", "n_bona_fide", "n_attack", "rejected_bona_fide", "accepted_attacks", "apcer",
           "bpcer", "acer", "auroc", "run_id", "transfer", "source"]
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

    if problems:
        fail("consistency problems:\n  - " + "\n  - ".join(problems))

    doc = {
        "schema": SCHEMA,
        "what": "unified table of the accepted NUAA results (C1, M1, N1, N2, C1 crop check), rebuilt from the public aggregates of research/pad-2026; no restricted data, no inference, no new number",
        "sources": {k: {"path": SOURCES[k], "sha256": hashes[k]} for k in SOURCES},
        "runs": {"C1": C1_RUN, "M1": M1_RUN, "N1": n1["run_id"], "N1_subjects": n1s["run_id"], "N2_sessions": n2s["run_id"],
                 "N2_combined": n2c["run_id"], "C1_crop_check": crop["run_id"]},
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
    }
    return doc


# --- scrittura ----------------------------------------------------------------------------------------------------

def md_table(cols: list, rows: list, labels: dict = None) -> str:
    labels = labels or {}
    head = "| " + " | ".join(labels.get(c, c) for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n" for r in rows)
    return head + body


def summary_md(doc: dict) -> str:
    cols = ["row", "block", "model", "training_data", "early_stopping", "test_set", "unit", "extractor", "preprocessing",
            "threshold", "threshold_origin", "n_bona_fide", "n_attack", "rejected_bona_fide", "accepted_attacks", "apcer",
            "bpcer", "acer", "auroc", "run_id", "transfer"]
    labels = {"n_bona_fide": "n bona fide", "n_attack": "n attack", "rejected_bona_fide": "rejected bona fide",
              "accepted_attacks": "accepted attacks", "apcer": "APCER", "bpcer": "BPCER", "acer": "ACER", "auroc": "AUROC",
              "run_id": "run", "training_data": "training data (fitting)", "early_stopping": "early stopping",
              "test_set": "test set", "threshold_origin": "threshold origin", "transfer": "same session or transfer"}
    out = io.StringIO()
    out.write("# Unified table of the accepted NUAA results (C1, M1, N1, N2, C1 crop check)\n\n")
    out.write("Built by `nuaa_synthesis.py build` from the public aggregates merged in `research/pad-2026` "
              "(sources and SHA-256 in `summary_table.json`); no restricted data, no inference, no new number. "
              "%s. APCER = accepted attacks / attacks, BPCER = rejected bona fide / bona fide, ACER = their mean, "
              "AUROC threshold-free (the column named `auc` in the C1 table and `test AUC` in the M1 table: the same Mann-Whitney statistic). "
              "Rates as printed in the C1 and M1 tables (four decimals; M1 thresholds six) or rounded to four decimals "
              "from the exact values of the JSON aggregates (N1, N2, crop check). "
              "Runs: %s.\n\n" % (doc["decision"], ", ".join("%s `%s`" % (k, v) for k, v in doc["runs"].items())))
    out.write(md_table(cols, doc["rows"], labels))
    out.write("\nSource of every row: `summary_table.csv` (column `source`) and `summary_table.json`.\n")
    return out.getvalue()


SHORT_BLOCK = {"C1": "C1", "M1": "M1", "N1": "N1", "N2 sessions": "N2-S", "N2 combined": "N2-C", "C1 crop check": "CROP"}


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
    fail("compact table: block %r" % r["block"])


def compact_legend(doc: dict) -> dict:
    T = doc["texts"]
    return {
        "training": {"T-NUAA": T["train"] + "; early stopping on " + T["es"],
                     "T-COMB": TRAIN_COMB + "; historical early stopping"},
        "test": {"S-300": T["test_c1"], "S-3000": T["test_3000"], "S-491": T["target_491"],
                 "S-03/9": T["s03_9"]},
        "extractor": {"E-C1": "c1_a80f6f9: the C1 evaluation extractor (RetinaFace box 1 px narrower and shorter than the cache)",
                      "E-CACHE": "cache_0ac7143: the training-cache extractor (cache frames, or the frozen copy on new frames)",
                      "E-WHOLE": VARIANTS["whole_image"], "E-CROP": VARIANTS["face_crop"]},
        "preprocessing": {"P-REF": PREP_REF, "P-HIST": PREP_HIST},
        "transfer": {"same": TRANSFER_SAME, "session": TRANSFER_SESSION_ALL, "session/9": TRANSFER_SESSION_9,
                     "dataset": TRANSFER_DATASET},
    }


def compact_rows(doc: dict) -> list:
    T = doc["texts"]
    out = []
    for r in doc["rows"]:
        training = "T-COMB" if r["training_data"] == TRAIN_COMB else "T-NUAA"
        test = {T["test_c1"]: "S-300", T["test_3000"]: "S-3000", T["target_491"]: "S-491", T["s03_9"]: "S-03/9"}.get(r["test_set"])
        if test is None:
            fail("compact table: test set %r" % r["test_set"])
        e = r["extractor"]
        extractor = ("E-C1" if e.startswith(EXTRACTOR_C1) else "E-CACHE" if e.startswith(EXTRACTOR_CACHE)
                     else "E-WHOLE" if e == VARIANTS["whole_image"] else "E-CROP" if e == VARIANTS["face_crop"]
                     else fail("compact table: extractor %r" % e))
        transfer = {TRANSFER_SAME: "same", TRANSFER_SESSION_ALL: "session", TRANSFER_SESSION_9: "session/9", TRANSFER_DATASET: "dataset"}[r["transfer"]]
        thr = r["threshold"] + (" (dev EER)" if r["block"] == "M1" else " (fixed)")
        prep = {PREP_REF: "P-REF", PREP_HIST: "P-HIST"}[r["preprocessing"]]
        out.append({"block": SHORT_BLOCK[r["block"]], "model": short_model(r), "training": training, "test": test,
                    "extractor": extractor, "preprocessing": prep, "threshold": thr,
                    "errors": "%s / %s" % (r["rejected_bona_fide"], r["accepted_attacks"]),
                    "apcer": r["apcer"], "bpcer": r["bpcer"], "acer": r["acer"], "auroc": r["auroc"],
                    "run_id": r["run_id"], "transfer": transfer})
    return out


def compact_md(doc: dict) -> str:
    cols = ["block", "model", "training", "test", "extractor", "preprocessing", "threshold", "errors", "apcer", "bpcer", "acer", "auroc", "run_id", "transfer"]
    labels = {"errors": "rejected bf / accepted att.", "apcer": "APCER", "bpcer": "BPCER", "acer": "ACER",
              "auroc": "AUROC", "run_id": "run", "transfer": "transfer", "test": "test set", "preprocessing": "prep."}
    out = io.StringIO()
    out.write("# Unified table, compact form (codes in the legend below; full text, row ids and sources in `summary_table.md`)\n\n")
    out.write(md_table(cols, compact_rows(doc), labels))
    out.write("\nLegend. Blocks: C1, M1, N1 (part 2), N2-S (N2 part 1, sessions), N2-C (N2 part 2, combined model), CROP (C1 crop check); "
              "ckpt = checkpoint; \"rejected bf / accepted att.\" = rejected bona fide / accepted attacks, the numerators of BPCER and APCER. "
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
            "Context only: pre-trained models never trained on NUAA, hand-crafted rules, and classifiers fitted on all 3,000 "
            "sampled NUAA training images (the dev images included; no early stopping involved). Source: `%s`.\n\n" % (C1_RUN, SOURCES["c1_table"])
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
    }


def cmd_build(a) -> int:
    doc = build(Path(a.research))
    files = outputs(doc)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        with open(out / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    print("BUILD: %d rows, %d cross-dataset rows, %d context rows, %d subject rows -> %s" %
          (len(doc["rows"]), len(doc["cross_dataset_rows"]), len(doc["context_rows"]), len(doc["subject_sensitivity"]), out))
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
