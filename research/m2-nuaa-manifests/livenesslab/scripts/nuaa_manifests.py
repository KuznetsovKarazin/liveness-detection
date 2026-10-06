"""
Manifest versionati di NUAA e controllo delle sovrapposizioni fra gli split (M2, prima tranche).

Sottocomandi:
  build          legge le quattro liste ufficiali dell'archivio (raw/{client,imposter}_{train,test}_raw.txt), verifica che
                 coincidano con le cartelle preparate (data/NUAA/images/{bonafide,attack}_{training,validation}) e che i
                 file siano quelli estratti dall'archivio, calcola SHA-256 e byte di ogni immagine (sola lettura) e scrive
                 i manifest per immagine in <out>/manifests/:
                   nuaa_official_train.csv  train ufficiale (3.491)          nuaa_official_test.csv  test ufficiale (9.123)
                   nuaa_cnn_train.csv       train delle CNN official (2.507) nuaa_cnn_early_stopping.csv  validazione (493)
                   nuaa_cnn_test.csv        test delle CNN official (3.000)  nuaa_m1_dev.csv         dev di M1 (574)
                   nuaa_c1_v2.csv           le 300 di C1, versione 2 (sessione corretta: quinto campo del nome)
                 Colonne: key (real/<nome> o attack/<nome>), label (0 bona fide, 1 attacco), subject, glasses, pos, session,
                 pic (campi del nome NUAA), official_split (train/test dalle liste ufficiali), sha256, bytes.
                 Ricostruzione dello split delle CNN con le funzioni di select_threshold.py (stesso campionamento
                 `spread(..., 1500)` e stesso GroupShuffleSplit di train_cnn.py), verificata sui conteggi delle schede dei
                 checkpoint e sugli array dei soggetti e delle label della cache di addestramento (s_train/y_train,
                 s_test/y_test di data/processed/nuaa_1500.npz); i frame della cache (X_train, X_test) sono solo hashati
                 per cercare duplicati. Collega ogni manifest ai checkpoint delle quattro CNN NUAA official
                 (SHA-256 dell'.h5 dalla scheda, ricalcolato sul file). Scrive <out>/nuaa_manifests_summary.json (privato,
                 con i conteggi per soggetto) e <out>/nuaa_manifests_aggregate.json (pubblicabile: nessun identificativo di
                 soggetto, nessun dato per immagine, controllato con la scansione del contenuto vietato).
                 Codice 1 se un controllo fallisce (i file vengono scritti comunque, con passed = false, per l'ispezione).
  check-overlap  legge un insieme di manifest e le regole dichiarate (default delivery/m2/nuaa_overlap_rules.json) e
                 confronta tutte le coppie per chiave, hash del contenuto, soggetto e (soggetto, sessione), più i duplicati
                 interni per chiave e per hash e le regole di derivazione (partizione e campionamento `spread`, chiusura
                 per soggetto). Scrive <out>/overlap_report.json e .md (riservati) e overlap_report_public.json e .md.
                 Codice 1 se una regola "must" non è rispettata; le regole "report" sono solo riportate. Con --self-test
                 ripete il controllo su copie guaste dei manifest (righe e SHA-256 ridichiarati) e pretende per ognuna
                 esattamente le regole attese fallite (<out>/overlap_self_test.json e .md).
  recount        ricalcola dai soli manifest i conteggi dell'aggregato pubblico (per manifest, esclusioni, quota di
                 test da soggetti assenti dal train, riuso della validazione) e li confronta con l'aggregato: codice 1
                 se differiscono. Non servono immagini né pesi.
  reconstruct    ricostruisce dal solo nuaa_official_train.csv lo split delle CNN (stesso `spread` e stesso
                 GroupShuffleSplit) e il dev di M1, e li confronta con cnn_train, cnn_es e m1_dev (serve scikit-learn,
                 non servono immagini).
  compare-c1     confronta il manifest C1 versione 1 (quello del pacchetto C1) con la versione 2: stesse chiavi, hash,
                 byte, soggetti ed etichette per tutte le righe; la colonna session della v1 coincide con glasses.
  report         rapporto breve in inglese (conteggi, sovrapposizioni, esclusioni, informazioni non ricostruibili,
                 dichiarazioni) dal modello delivery/m2/report.md.tmpl: <out>/nuaa_m2_report.md (privato, può nominare
                 i soggetti) e <out>/nuaa_m2_report_public.md (senza identificativi di soggetto, controllato).

Nessuna inferenza, nessun addestramento, nessun modello caricato: elenchi e hash dei file e lettura della cache di
addestramento (array dei soggetti, delle label e dei frame; i frame sono solo hashati). Non scrive mai in results/c1,
results/eval, results/m1, models o data. Dipendenze: Python 3.11 e numpy (tutti i sottocomandi, tramite
select_threshold.py); scikit-learn per build e reconstruct.

Uso tipico:
  python scripts/nuaa_manifests.py build
  python scripts/nuaa_manifests.py check-overlap
  python scripts/nuaa_manifests.py report
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import numpy as np  # noqa: E402
    import select_threshold as st  # noqa: E402  (aggiunge anche src/ al percorso)
except ModuleNotFoundError as _e:   # dipendenza mancante: messaggio chiaro, mai un traceback
    print(f"ERROR: missing Python package {_e.name!r}: install the requirements (requirements.txt: numpy for every "
          "subcommand, scikit-learn for build and reconstruct)", file=sys.stderr)
    sys.exit(1)
from package_rules import C1_MANIFEST_SHA256, FORBIDDEN_TEXT, IP_ALLOW, IPV4, MARKERS  # noqa: E402
from select_threshold import (CLASS_DIRS, IMG_EXT, PER_CLASS, fail, load_json, now, relpath, repo_state,  # noqa: E402
                              sha256, spread)
from tesi_app.paths import DATA, RESULTS, WEIGHTS  # noqa: E402


M2_DIR = RESULTS / "m2"
NUAA = DATA / "NUAA" / "images"
RAW_DIR = DATA / "raw"
NUAA_RAW = RAW_DIR / "nuaa" / "raw"
ARCHIVE = RAW_DIR / "nuaaaa.tar.gz"
HF_METADATA = RAW_DIR / ".cache" / "huggingface" / "download" / "nuaaaa.tar.gz.metadata"
MIRROR = "akahana/anti-spoofing-nuaaaa"
C1_MANIFEST = RESULTS / "c1" / "nuaa_manifest.csv"
C1_CONFIG = RESULTS / "c1" / "nuaa_config.json"
C1_EVAL = DATA / "eval" / "nuaa"
M1_DEV_MANIFEST = RESULTS / "m1" / "nuaa_dev_manifest.csv"
PROCESSED_NPZ = DATA / "processed" / f"nuaa_{PER_CLASS}.npz"
RULES = ROOT / "delivery" / "m2" / "nuaa_overlap_rules.json"
REPORT_TMPL = ROOT / "delivery" / "m2" / "report.md.tmpl"
ARCHS = ["LivenessNet", "AttackNetV1", "AttackNetV2_1", "AttackNetV2_2"]
C1_PER_CLASS = 150                                  # 150 + 150 di C1 (prepare_datasets.py, spread sulle liste ufficiali)
MANIFEST_SCHEMA = "nuaa-m2/1"

# liste ufficiali dell'archivio: (split, label) -> file; cartella dell'archivio per label; cartella preparata per split
OFFICIAL_LISTS = {("train", 0): "client_train_raw.txt", ("train", 1): "imposter_train_raw.txt",
                  ("test", 0): "client_test_raw.txt", ("test", 1): "imposter_test_raw.txt"}
RAW_CLASS_DIR = {0: "ClientRaw", 1: "ImposterRaw"}
SPLIT_TAG = {"train": "training", "test": "validation"}
DIR_OF_LABEL = {lab: cls for cls, lab, _ in CLASS_DIRS}        # 0 -> bonafide, 1 -> attack
KEY_DIR_OF_LABEL = {lab: kd for _, lab, kd in CLASS_DIRS}      # 0 -> real, 1 -> attack

COLUMNS = ["key", "label", "subject", "glasses", "pos", "session", "pic", "official_split", "sha256", "bytes"]
# dichiarazioni sulla fonte (stesso testo in naming_schema.md, README.md e report.md.tmpl)
FIELD_ORDER_SOURCE = ("`ID_glasses_pos_session_picNo`, as stated on the M2 task card (3 October 2026), where the coordinator took it "
                      "from the README of the original release; that README is not in the mirror and was not available to us. The "
                      "reference publication of the database is Tan et al., ECCV 2010. On the data we verified the structure (five "
                      "numeric fields), that ID equals the subject folder, and that the session field separates the official splits "
                      "(01–02 train, 03 test); the meaning of the glasses and pos codes is not verified and they are carried as opaque codes.")
SPLIT_LISTS_SOURCE = ("the mirror's lists carry the path prefix `/kaggle/input/nuaaaa/raw/`, added by the mirror author: they are the "
                      "mirror's version of the official lists. Only file names are compared. We did not verify their equivalence with "
                      "the lists of the original release; the coordinator rebuilt the 300 C1 images from the official source with identical "
                      "hashes (M1 card, 2 October 2026). `build` accepts only the mirror's list format.")
SUBJECTS_SOURCE = ("the 4-digit IDs of the official folders and file names. No identity is inferred from image content or from groups of "
                   "images. Treating the same ID in ClientRaw and ImposterRaw as the same person follows the naming of the release and is "
                   "not verified on the images.")
# nei documenti: "- Field order: " + FIELD_ORDER_SOURCE, "- Split lists: " + SPLIT_LISTS_SOURCE, "- Subjects: " + SUBJECTS_SOURCE
SESSION_SPLIT = {"01": "train", "02": "train", "03": "test"}   # quinto campo -> split ufficiale (verificato da build sulle liste)
MAX_EPOCHS = 12                                     # limite di epoche dell'addestramento del 12/09 (C1 checkpoints/PROVENANCE.md)
# uso delle immagini di validazione nell'addestramento (scripts/train_cnn.py, train): nessun aggiornamento dei pesi;
# la loro val_loss guida EarlyStopping (patience 4, restore_best_weights) e ReduceLROnPlateau (factor 0.5, patience 2)
VALIDATION_USE = ("Validation images (`cnn_es`, 493): never used for gradient updates; their `val_loss` controlled early stopping "
                  "(patience 4, best weights restored), hence which epoch's weights were saved, and the learning-rate reduction "
                  "(ReduceLROnPlateau, factor 0.5, patience 2)")
# manifest scritti da build: nome logico -> (file, descrizione)
MANIFESTS = {
    "official_train": ("nuaa_official_train.csv", "official training split (all images of the client/imposter train lists)"),
    "official_test": ("nuaa_official_test.csv", "official test split (all images of the client/imposter test lists)"),
    "cnn_train": ("nuaa_cnn_train.csv", "images that fitted the weights of the four NUAA official CNNs"),
    "cnn_es": ("nuaa_cnn_early_stopping.csv", "validation images of the four NUAA official CNNs (early stopping, best-epoch selection and learning-rate reduction on val_loss)"),
    "cnn_test": ("nuaa_cnn_test.csv", "test images of the checkpoint cards (spread(..., 1500) per class of the official test split)"),
    "m1_dev": ("nuaa_m1_dev.csv", "M1 source-dev set: all official training images of the two validation subjects of the CNN training"),
    "c1": ("nuaa_c1_v2.csv", "the 300 C1 test images, manifest version 2 (corrected session field)"),
}

# nome su disco: <soggetto>_<ID>_<glasses>_<pos>_<session>_<picNo>.jpg (prefisso della cartella + nome originale)
NAME_RE = re.compile(r"^(\d{4})_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d+)\.jpg$")
# riga delle liste ufficiali: .../raw/{ClientRaw,ImposterRaw}/<ID>/<ID>_<glasses>_<pos>_<session>_<picNo>.jpg
LIST_RE = re.compile(r"^/kaggle/input/nuaaaa/raw/(ClientRaw|ImposterRaw)/(\d{4})/((\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d+)\.jpg)$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


# ----------------------------------------------------------------------------- nomi NUAA

def parse_name(name: str) -> dict:
    """Campi di un nome su disco <soggetto>_<ID>_<glasses>_<pos>_<session>_<picNo>.jpg; ValueError se il nome non ha
    questa forma o se il prefisso della cartella (soggetto) è diverso dall'ID del nome originale."""
    m = NAME_RE.match(name or "")
    if not m:
        raise ValueError(f"not a NUAA file name of the form <subject>_<ID>_<glasses>_<pos>_<session>_<picNo>.jpg: {name!r}")
    subject, ident, glasses, pos, session, pic = m.groups()
    if subject != ident:
        raise ValueError(f"folder prefix {subject} differs from the ID field {ident}: {name!r}")
    return {"subject": subject, "glasses": glasses, "pos": pos, "session": session, "pic": pic}


def parse_list_line(line: str):
    """Riga di una lista ufficiale -> (cartella dell'archivio, nome su disco). ValueError se la riga non ha la forma
    attesa o se la cartella del soggetto è diversa dall'ID del nome."""
    m = LIST_RE.match(line.strip())
    if not m:
        raise ValueError(f"unexpected line in an official list: {line.strip()[:120]!r}")
    raw_dir, folder, orig, ident = m.group(1), m.group(2), m.group(3), m.group(4)
    if folder != ident:
        raise ValueError(f"folder {folder} differs from the ID field {ident}: {line.strip()[:120]!r}")
    return raw_dir, f"{folder}_{orig}"


def make_key(label: int, name: str) -> str:
    return f"{KEY_DIR_OF_LABEL[int(label)]}/{name}"


def name_of(key: str) -> str:
    return key.split("/", 1)[1]


def read_official_list(p: Path, label: int):
    """Nomi su disco di una lista ufficiale, nell'ordine della lista; righe vuote ignorate. Errore pulito su righe
    malformate, cartella dell'archivio diversa dalla classe attesa o righe duplicate."""
    if not p.is_file():
        fail(f"missing official list {relpath(p)}")
    names, problems = [], []
    for i, line in enumerate(p.read_text(encoding="utf-8", errors="strict").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw_dir, name = parse_list_line(line)
        except ValueError as e:
            problems.append(f"line {i}: {e}"); continue
        if raw_dir != RAW_CLASS_DIR[label]:
            problems.append(f"line {i}: folder {raw_dir}, expected {RAW_CLASS_DIR[label]}"); continue
        names.append(name)
    dup = [n for n, c in Counter(names).items() if c > 1]
    if dup:
        problems.append(f"{len(dup)} duplicated entries (e.g. {sorted(dup)[:3]})")
    if problems:
        fail(f"{relpath(p)}: " + "; ".join(problems[:5]))
    return names


# ----------------------------------------------------------------------------- correzione della sessione di C1

def c1_correction(v1_rows):
    """Manifest C1 versione 2 dalla versione 1 (results/c1/nuaa_manifest.csv, congelata): i campi si rileggono dal nome
    con parse_name. Nella versione 1 la colonna `session` era il terzo campo del nome (glasses), non il quinto.
    Restituisce (righe v2, statistiche); le righe v2 mantengono key, label, sha256 e bytes della v1."""
    rows, stats = [], {"n": 0, "v1_session_equals_glasses": 0, "v1_session_equals_session": 0,
                       "v1_session_values": Counter(), "session_values": Counter(), "problems": []}
    for r in v1_rows:
        key = r.get("file", "")
        if st.key_problem(key):
            stats["problems"].append(f"invalid C1 key {key!r}"); continue
        try:
            f = parse_name(name_of(key))
        except ValueError as e:
            stats["problems"].append(str(e)); continue
        if r.get("subject") != f["subject"]:
            stats["problems"].append(f"{key}: subject {r.get('subject')!r} differs from the name ({f['subject']})")
        if r.get("label") != str(st.KEY_LABEL[key.split("/", 1)[0]]):
            stats["problems"].append(f"{key}: label {r.get('label')!r} not consistent with the key folder")
        stats["n"] += 1
        stats["v1_session_equals_glasses"] += r.get("session") == f["glasses"]
        stats["v1_session_equals_session"] += r.get("session") == f["session"]
        stats["v1_session_values"][r.get("session")] += 1
        stats["session_values"][f["session"]] += 1
        rows.append({"key": key, "label": int(r["label"]) if r.get("label") in ("0", "1") else r.get("label"), **f,
                     "official_split": "test", "sha256": r.get("sha256"), "bytes": r.get("bytes")})
    stats["v1_session_values"] = dict(sorted(stats["v1_session_values"].items()))
    stats["session_values"] = dict(sorted(stats["session_values"].items()))
    return rows, stats


# ----------------------------------------------------------------------------- manifest: scrittura, lettura, conteggi

def write_manifest(p: Path, rows) -> str:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["key"]))
    return sha256(p)


def manifest_problems(rows) -> list:
    """Controlli di un manifest per il confronto: chiave valida e nome NUAA, e ogni campo uguale a quello derivato dalla
    chiave (label dal prefisso real/attack; subject, glasses, pos, session, pic dal nome; official_split dalla sessione:
    01/02 train, 03 test), confrontato come testo esatto (spazi, zeri iniziali persi e valori vuoti sono incoerenze);
    sha256 di 64 caratteri esadecimali minuscoli, bytes intero. Una riga sbagliata ferma anche le regole "report"."""
    bad = defaultdict(list)
    for r in rows:
        k = r.get("key")
        if st.key_problem(k):
            bad["invalid keys (expected real/<name> or attack/<name>)"].append(str(k)); continue
        try:
            f = parse_name(name_of(k))
        except ValueError:
            bad["keys that are not NUAA file names <subject>_<ID>_<glasses>_<pos>_<session>_<picNo>.jpg"].append(k); continue
        expected = {"label": str(st.KEY_LABEL[k.split("/", 1)[0]]), **f, "official_split": SESSION_SPLIT.get(f["session"], "")}
        for c, v in expected.items():
            if r.get(c) != v:
                bad[f"{c} values not consistent with the key"].append(k)
        if not SHA_RE.match(r.get("sha256") or ""):
            bad["malformed sha256 values"].append(k)
        if not re.fullmatch(r"\d+", r.get("bytes") or ""):
            bad["malformed bytes values"].append(k)
    return [f"{len(v)} {what} (e.g. {v[:3]})" for what, v in bad.items()]


def read_csv(p: Path, columns):
    """Righe di un CSV letto come utf-8-sig (un BOM iniziale è tollerato); errore pulito se il file manca o non si
    legge, se l'intestazione ha colonne duplicate o mancanti, o se una riga ha un numero di campi diverso."""
    p = Path(p)
    if not p.is_file():
        fail(f"missing {relpath(p)}")
    try:
        with p.open(newline="", encoding="utf-8-sig") as f:
            rd = csv.reader(f)
            header = next(rd, [])
            body = list(rd)
    except (csv.Error, UnicodeDecodeError) as e:
        fail(f"{relpath(p)}: unreadable CSV ({e})")
    dup = sorted(c for c, n in Counter(header).items() if n > 1)
    if dup:
        fail(f"{relpath(p)}: duplicated columns in the header {dup}")
    missing = [c for c in columns if c not in header]
    if missing:
        fail(f"{relpath(p)}: missing columns {missing}")
    ragged = [i + 2 for i, r in enumerate(body) if len(r) != len(header)]
    if ragged:
        fail(f"{relpath(p)}: rows with a wrong number of fields (lines {ragged[:3]})")
    return [dict(zip(header, r)) for r in body]


def load_manifest(p: Path):
    """Righe di un manifest per check-overlap; errore pulito se mancano colonne o se le righe non sono valide."""
    rows = read_csv(p, COLUMNS)
    problems = manifest_problems(rows)
    if problems:
        fail(f"{relpath(p)}: " + "; ".join(problems))
    return rows


def counts(rows, private: bool) -> dict:
    """Conteggi per classe, sessione, glasses e pos e numero di soggetti; per soggetto solo nella versione privata."""
    def by(field):
        d = defaultdict(lambda: {"bona_fide": 0, "attack": 0})
        for r in rows:
            d[str(r[field])]["attack" if str(r["label"]) == "1" else "bona_fide"] += 1
        return {k: d[k] for k in sorted(d)}
    out = {"n": len(rows), "n_bona_fide": sum(1 for r in rows if str(r["label"]) == "0"),
           "n_attack": sum(1 for r in rows if str(r["label"]) == "1"),
           "n_subjects": len({r["subject"] for r in rows}),
           "n_subjects_bona_fide": len({r["subject"] for r in rows if str(r["label"]) == "0"}),
           "n_subjects_attack": len({r["subject"] for r in rows if str(r["label"]) == "1"}),
           "by_session": by("session"), "by_glasses": by("glasses"), "by_pos": by("pos")}
    if private:
        out["by_subject"] = by("subject")
        out["by_subject_session"] = {f"{s}/{x}": v for (s, x), v in sorted(
            Counter((r["subject"], r["session"]) for r in rows).items())}
    return out


RECOUNT_MANIFESTS = ("official_train", "official_test", "cnn_train", "cnn_es", "cnn_test", "m1_dev", "c1")


def recount_block(m: dict) -> dict:
    """Conteggi dell'aggregato che si ricalcolano dai soli manifest (nome logico -> righe): per manifest, esclusioni del
    campionamento, quota di test da soggetti assenti dal train ufficiale (confondente classe-novità del soggetto),
    riuso della validazione e righe di C1 dai soggetti di validazione. Nessun identificativo di soggetto."""
    tr_subj = {r["subject"] for r in m["official_train"]}
    novelty = {}
    for n in ("official_test", "cnn_test", "c1"):
        novelty[n] = {}
        for lab, cls in (("0", "bona_fide"), ("1", "attack")):
            rr = [r for r in m[n] if str(r["label"]) == lab]
            novelty[n][cls] = {"n": len(rr), "from_subjects_absent_from_official_train": sum(r["subject"] not in tr_subj for r in rr)}
    sampled = {r["key"] for r in m["cnn_train"]} | {r["key"] for r in m["cnn_es"]}
    test_keys = {r["key"] for r in m["cnn_test"]}
    es_keys, es_subj = {r["key"] for r in m["cnn_es"]}, {r["subject"] for r in m["cnn_es"]}
    c1_val = [r for r in m["c1"] if r["subject"] in es_subj]
    block = {
        "manifest_counts": {n: counts(m[n], private=False) for n in RECOUNT_MANIFESTS},
        "exclusions": {"official_train_not_sampled": counts([r for r in m["official_train"] if r["key"] not in sampled], private=False),
                       "official_test_not_sampled": counts([r for r in m["official_test"] if r["key"] not in test_keys], private=False)},
        "test_from_subjects_absent_from_official_train": novelty,
        "declaration_counts": {"m1_dev_rows": len(m["m1_dev"]), "m1_dev_rows_that_are_validation_images": sum(r["key"] in es_keys for r in m["m1_dev"]),
                               "c1_rows_from_validation_subjects": len(c1_val),
                               "c1_rows_from_validation_subjects_bona_fide": sum(str(r["label"]) == "0" for r in c1_val),
                               "c1_rows_from_validation_subjects_attack": sum(str(r["label"]) == "1" for r in c1_val)},
    }
    return json.loads(json.dumps(block))          # stessa forma dopo la scrittura in JSON (chiavi testuali)


# ----------------------------------------------------------------------------- scansione del contenuto pubblico

def public_problems(text: str, subject_ids, markdown: bool = False) -> list:
    """Contenuto vietato in un file pubblico, con le regole di package_rules.py (uguali a export_m1.py): percorsi
    assoluti, nomi dell'operatore, credenziali, riferimenti a strumenti di IA, indirizzi IPv4 non locali, segni di
    lavoro non finito (solo nei .md), e identificativi di soggetto isolati (non dentro un hash o un numero)."""
    bad = []
    for label, rx in FORBIDDEN_TEXT:
        if m := rx.search(text):
            bad.append(f"{label}: {m.group(0)[:40]!r}")
    for ip in (m.group(0) for m in IPV4.finditer(text)):
        if ip not in IP_ALLOW:
            bad.append(f"IP address: {ip}"); break
    if markdown and (m := MARKERS.search(text)):
        bad.append(f"unfinished-work marker: {m.group(0)!r}")
    for s in sorted(set(subject_ids)):
        if re.search(rf"(?<![0-9A-Za-z]){re.escape(s)}(?![0-9A-Za-z])", text):
            bad.append(f"subject id {s}")
    return bad


# ----------------------------------------------------------------------------- build

def list_disk(tag: str, cls: str):
    d = NUAA / f"{cls}_{tag}"
    if not d.is_dir():
        fail(f"missing folder {relpath(d)}")
    return d, sorted(p.name for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


def same_file(a: Path, b: Path) -> bool:
    sa, sb = a.stat(), b.stat()
    return (sa.st_ino, sa.st_dev) == (sb.st_ino, sb.st_dev)


def hf_metadata():
    """Revisione (prima riga) e etag (seconda riga) del file di metadati della cache di Hugging Face; None se manca."""
    if not HF_METADATA.is_file():
        return None
    lines = HF_METADATA.read_text(encoding="utf-8").splitlines()
    return {"revision": lines[0].strip() if lines else None, "etag": lines[1].strip() if len(lines) > 1 else None}


def frame_hashes(arr) -> list:
    """SHA-256 di ogni frame derivato di un array della cache (una riga = un'immagine pre-elaborata, così come salvata:
    byte in ordine C con forma e tipo del frame in testa, quindi frame uguali solo se identici pixel per pixel)."""
    head = f"{arr.dtype.str}{arr.shape[1:]}".encode()
    return [hashlib.sha256(head + np.ascontiguousarray(x).tobytes()).hexdigest() for x in arr]


def derived_frame_check(npz: Path, rec) -> dict:
    """Duplicati fra i frame derivati della cache di addestramento: interni a X_train e a X_test, fra train e test, e
    fra le immagini che hanno stimato i pesi e quelle di validazione (indici della ricostruzione). Solo lettura degli
    array, nessun modello: i frame sono legati ai file solo tramite l'ordine di campionamento (gruppi e label)."""
    z = np.load(npz)
    h = {}
    for split in ("train", "test"):
        arr = z[f"X_{split}"]
        h[split] = frame_hashes(arr)
        shape = list(arr.shape)
        del arr
        h[split + "_shape"] = shape
    out = {"hash": "SHA-256 of each preprocessed frame as stored in the cache (dtype, shape, C-order bytes)"}
    for split in ("train", "test"):
        c = Counter(h[split])
        dup = {v: n for v, n in c.items() if n > 1}
        out[split] = {"n_frames": len(h[split]), "shape": h[split + "_shape"], "distinct": len(c),
                      "duplicated_values": len(dup), "duplicated_rows": sum(dup.values())}
    out["train_test_shared"] = len(set(h["train"]) & set(h["test"]))
    fit = {h["train"][i] for i in rec["train_idx"]}
    val = {h["train"][i] for i in rec["val_idx"]}
    out["fit_vs_validation_shared"] = len(fit & val)
    out["linkage"] = ("frames are linked to the image files by sampling order, subject and label arrays (s_*, y_*), "
                      "not by per-image path or hash: the cache stores no file names")
    return out


def cmd_build(a) -> None:
    out = Path(a.out)
    for forbidden in (RESULTS / "c1", RESULTS / "eval", RESULTS / "m1", WEIGHTS, DATA):
        if st.is_under(out, forbidden):
            fail(f"output folder {relpath(out)} is inside {relpath(forbidden)}: refused (read-only area)")
    checks, problems = {}, []

    def check(name, ok, msg=None):
        checks[name] = bool(ok)
        if not ok:
            problems.append(f"{name}: {msg}" if msg else name)
        return ok

    # 1. fonte: archivio del mirror, revisione e etag della cache di Hugging Face
    print("== source", flush=True)
    meta = hf_metadata()
    archive_sha = sha256(ARCHIVE) if ARCHIVE.is_file() else None
    source = {"dataset": "NUAA Photograph Imposter Database (Tan, Li, Liu, Jiang, ECCV 2010)",
              "mirror": f"Hugging Face dataset {MIRROR}", "revision": meta["revision"] if meta else None,
              "archive": ARCHIVE.name, "archive_sha256": archive_sha,
              "archive_sha256_equals_hf_etag": (meta is not None and archive_sha is not None and meta["etag"] == archive_sha),
              "official_readme_in_mirror": False, "field_order_source": FIELD_ORDER_SOURCE, "split_lists_source": SPLIT_LISTS_SOURCE,
              "subjects_source": SUBJECTS_SOURCE,
              "naming_schema": "delivery/m2/naming_schema.md (field order from the M2 task card, 3 October 2026; the README of the original release was not available)"}
    check("archive_present", archive_sha is not None, f"missing {relpath(ARCHIVE)}")
    check("archive_matches_hf_metadata", source["archive_sha256_equals_hf_etag"],
          "archive SHA-256 differs from the etag of the Hugging Face cache metadata (or metadata missing)")
    print(f"   revision {source['revision']}, archive sha256 {archive_sha}", flush=True)

    # 2. liste ufficiali = cartelle preparate; file preparati = file estratti dall'archivio
    print("== split lists (mirror copy) vs prepared folders", flush=True)
    lists, list_names = {}, {}
    for (split, lab), fname in OFFICIAL_LISTS.items():
        p = NUAA_RAW / fname
        names = read_official_list(p, lab)
        list_names[(split, lab)] = names
        d, disk = list_disk(SPLIT_TAG[split], DIR_OF_LABEL[lab])
        missing, extra = sorted(set(names) - set(disk)), sorted(set(disk) - set(names))
        lists[fname] = {"sha256": sha256(p), "n_entries": len(names), "split": split, "label": lab,
                        "prepared_folder": d.name, "n_prepared": len(disk), "equal_to_prepared_folder": not missing and not extra}
        check(f"list_equals_folder:{fname}", not missing and not extra,
              f"{len(missing)} listed images missing from {d.name}, {len(extra)} images of {d.name} not listed (e.g. {(missing + extra)[:3]})")
    raw_check = {"hard_linked": 0, "identical_by_hash": 0, "different": 0, "raw_missing": 0}
    rows = {}            # chiave -> riga del manifest
    for (split, lab), names in list_names.items():
        d = NUAA / f"{DIR_OF_LABEL[lab]}_{SPLIT_TAG[split]}"
        for name in names:
            p = d / name
            if not p.is_file():
                continue
            f = parse_name(name)
            raw = NUAA_RAW / RAW_CLASS_DIR[lab] / f["subject"] / name.split("_", 1)[1]
            h = sha256(p)
            if not raw.is_file():
                raw_check["raw_missing"] += 1
            elif same_file(p, raw):
                raw_check["hard_linked"] += 1
            elif sha256(raw) == h:
                raw_check["identical_by_hash"] += 1
            else:
                raw_check["different"] += 1
            key = make_key(lab, name)
            rows[key] = {"key": key, "label": lab, **f, "official_split": split, "sha256": h, "bytes": p.stat().st_size}
    check("prepared_images_equal_archive_files", raw_check["different"] == 0 and raw_check["raw_missing"] == 0,
          f"prepared images vs extracted archive files: {raw_check}")
    official = {s: [r for r in rows.values() if r["official_split"] == s] for s in ("train", "test")}
    check("official_train_count", len(official["train"]) == 3491, f"{len(official['train'])} images, expected 3491")
    check("official_test_count", len(official["test"]) == st.EXPECTED_TEST, f"{len(official['test'])} images, expected {st.EXPECTED_TEST}")
    print(f"   official train {len(official['train'])}, test {len(official['test'])}; prepared vs archive {raw_check}", flush=True)

    # 3. split delle CNN official: ricostruzione con le funzioni di select_threshold.py e cache di addestramento
    print("== reconstruction of the CNN split (scripts/select_threshold.py reconstruct_split)", flush=True)
    rec = st.reconstruct_split(NUAA)
    st.check_reconstruction(rec)      # conteggi delle schede (2.507/493, 1.500 per classe), altrimenti errore
    keys_sampled = [st.cache_key(p) for p in rec["files"]]
    cnn_train = [keys_sampled[i] for i in rec["train_idx"]]
    cnn_es = [keys_sampled[i] for i in rec["val_idx"]]
    test_files = []
    for cls, lab, _ in CLASS_DIRS:
        d = NUAA / f"{cls}_validation"
        test_files += spread([p for p in d.iterdir() if p.suffix.lower() in IMG_EXT], PER_CLASS)
    cnn_test = [st.cache_key(p) for p in test_files]
    test_groups = np.array([st.subject_of(p.name) for p in test_files])
    test_y = np.array([0 if p.parent.name.startswith("bonafide") else 1 for p in test_files], np.int64)
    cache = {"file": st.rel(PROCESSED_NPZ), "present": PROCESSED_NPZ.is_file()}
    if PROCESSED_NPZ.is_file():
        z = np.load(PROCESSED_NPZ)      # solo gli array dei soggetti e delle label: le immagini non vengono lette
        cache["members"] = sorted(z.files)
        cache["matches_training_cache_groups"] = bool("s_train" in z.files and np.array_equal(z["s_train"], rec["groups"])
                                                      and np.array_equal(z["y_train"], rec["y"]))
        cache["matches_test_cache_groups"] = bool("s_test" in z.files and np.array_equal(z["s_test"], test_groups)
                                                  and np.array_equal(z["y_test"], test_y))
        cache["n_train"], cache["n_test"] = int(len(z["y_train"])), int(len(z["y_test"]))
        cache["sha256"] = sha256(PROCESSED_NPZ)
        stt = PROCESSED_NPZ.stat()
        import time as _t
        cache["file_times"] = {"created": _t.strftime("%Y-%m-%d %H:%M:%S", _t.localtime(getattr(stt, "st_birthtime", stt.st_mtime))),
                               "modified": _t.strftime("%Y-%m-%d %H:%M:%S", _t.localtime(stt.st_mtime)),
                               "note": "file-system times of the local copy (metadata, not evidence)"}
        check("training_cache_groups", cache["matches_training_cache_groups"], "s_train/y_train differ from the reconstruction")
        check("test_cache_groups", cache["matches_test_cache_groups"], "s_test/y_test differ from spread(..., 1500) of the test folders")
    else:
        check("training_cache_present", False, f"missing {relpath(PROCESSED_NPZ)}: the reconstruction cannot be checked against the training cache")
    cache["linkage"] = ("by subject and label arrays in sampling order (group counts and order), not by per-image path or hash: "
                        "the cache stores preprocessed arrays and subject prefixes only")
    # frame derivati (immagini pre-elaborate della cache): duplicati interni e incroci, per hash degli array
    derived = None
    if PROCESSED_NPZ.is_file():
        derived = derived_frame_check(PROCESSED_NPZ, rec)
        check("derived_frames_no_duplicates", derived["train"]["duplicated_values"] == 0 and derived["test"]["duplicated_values"] == 0
              and derived["train_test_shared"] == 0 and derived["fit_vs_validation_shared"] == 0,
              f"duplicated derived frames in {st.rel(PROCESSED_NPZ)}: {derived}")
        print(f"   derived frames: train {derived['train']['distinct']}/{derived['train']['n_frames']} distinct, test "
              f"{derived['test']['distinct']}/{derived['test']['n_frames']} distinct, train/test shared {derived['train_test_shared']}, "
              f"fit/validation shared {derived['fit_vs_validation_shared']}", flush=True)
    cache["derived_frames"] = derived
    print(f"   train {len(cnn_train)}, early stopping {len(cnn_es)}, test {len(cnn_test)}; "
          f"training cache groups {cache.get('matches_training_cache_groups')}, test cache groups {cache.get('matches_test_cache_groups')}", flush=True)

    # 4. dev di M1: tutte le immagini del train ufficiale dei soggetti dell'early stopping
    dev_subjects = set(rec["val_subjects"])
    m1_dev = [r["key"] for r in official["train"] if r["subject"] in dev_subjects]
    m1_check = {"file": st.rel(M1_DEV_MANIFEST), "present": M1_DEV_MANIFEST.is_file()}
    if M1_DEV_MANIFEST.is_file():
        m1 = st.read_manifest(M1_DEV_MANIFEST, st.DEV_COLUMNS)
        m1_check["sha256"] = sha256(M1_DEV_MANIFEST)
        m1_check["same_keys"] = {r["file"] for r in m1} == set(m1_dev)
        m1_check["same_hashes"] = all(rows.get(r["file"], {}).get("sha256") == r["sha256"] for r in m1)
        m1_check["in_training_sample_equals_early_stopping"] = {r["file"] for r in m1 if r["in_training_sample"] == "1"} == set(cnn_es)
        check("m1_dev_manifest_consistent", m1_check["same_keys"] and m1_check["same_hashes"]
              and m1_check["in_training_sample_equals_early_stopping"], f"results/m1 dev manifest differs: {m1_check}")

    # 5. C1: versione 1 congelata, versione 2 con la sessione corretta, verificata sul disco e sulla regola di campionamento
    print("== C1 manifest version 2", flush=True)
    c1_v1 = st.read_manifest(C1_MANIFEST, ["file", "label", "subject", "session", "sha256", "bytes"])
    c1_v1_sha = sha256(C1_MANIFEST)
    c1_rows, c1_stats = c1_correction(c1_v1)
    check("c1_v1_is_delivered_manifest", c1_v1_sha == C1_MANIFEST_SHA256,
          f"{relpath(C1_MANIFEST)} SHA-256 {c1_v1_sha} differs from the C1 delivery ({C1_MANIFEST_SHA256})")
    check("c1_v1_rows_valid", not c1_stats["problems"], "; ".join(c1_stats["problems"][:3]))
    c1_disk = [r["key"] for r in c1_rows if rows.get(r["key"], {}).get("sha256") != r["sha256"]
               or str(rows.get(r["key"], {}).get("bytes")) != str(r["bytes"])]
    check("c1_rows_match_official_test_images", not c1_disk, f"{len(c1_disk)} C1 rows differ from the official test image (e.g. {c1_disk[:3]})")
    c1_rule = []
    for lab in (0, 1):
        c1_rule += [make_key(lab, n) for n in spread(list_names[("test", lab)], C1_PER_CLASS)]
    check("c1_equals_sampling_rule", set(c1_rule) == {r["key"] for r in c1_rows},
          "C1 keys differ from spread(official test list, 150) per class (prepare_datasets.py)")
    eval_mismatch = None
    if C1_EVAL.is_dir():
        eval_mismatch = 0
        for r in c1_rows:
            q = C1_EVAL / r["key"]
            eval_mismatch += (not q.is_file()) or sha256(q) != r["sha256"]
        check("c1_eval_folder_matches", eval_mismatch == 0, f"{eval_mismatch} images of {relpath(C1_EVAL)} differ from the C1 manifest")
    c1_rows = [rows[r["key"]] for r in c1_rows if r["key"] in rows]      # stessi campi delle altre righe (v2 = campi dal nome)
    c1_dev_attacks = [r for r in c1_rows if r["subject"] in dev_subjects]
    c1_info = {"v1": {"file": st.rel(C1_MANIFEST), "sha256": c1_v1_sha, "frozen": True,
                      "registered_sha256_at_c1_delivery": C1_MANIFEST_SHA256,
                      "session_column": "3rd field of the file name on disk, i.e. the glasses field (2nd field of the original name), "
                                        "parsed by make_c1_manifest.py with ^(\\d{4})_\\d{4}_(\\d{2})_",
                      "session_values": c1_stats["v1_session_values"]},
               "n": c1_stats["n"], "v1_session_equals_glasses": c1_stats["v1_session_equals_glasses"],
               "v1_session_equals_official_session": c1_stats["v1_session_equals_session"],
               "official_session_values": c1_stats["session_values"],
               "equals_sampling_rule": checks.get("c1_equals_sampling_rule"),
               "eval_folder_mismatches": eval_mismatch,
               "n_from_dev_subjects": len(c1_dev_attacks),
               "n_from_dev_subjects_by_class": dict(Counter("attack" if r["label"] else "bona_fide" for r in c1_dev_attacks)),
               "sessions_from_dev_subjects": sorted({r["session"] for r in c1_dev_attacks})}
    print(f"   v1 session = glasses for {c1_stats['v1_session_equals_glasses']}/{c1_stats['n']}; official session {c1_stats['session_values']}; "
          f"{len(c1_dev_attacks)} C1 images from the early-stopping subjects", flush=True)

    # 6. manifest per immagine
    sets = {"official_train": [r["key"] for r in official["train"]], "official_test": [r["key"] for r in official["test"]],
            "cnn_train": cnn_train, "cnn_es": cnn_es, "cnn_test": cnn_test, "m1_dev": m1_dev, "c1": [r["key"] for r in c1_rows]}
    unknown = sorted(k for ks in sets.values() for k in ks if k not in rows)
    check("all_keys_in_official_lists", not unknown, f"{len(unknown)} keys not in the official lists (e.g. {unknown[:3]})")
    man_dir = out / "manifests"
    manifests, manifests_public = {}, {}
    for name, keys in sets.items():
        fname, desc = MANIFESTS[name]
        mrows = [rows[k] for k in keys if k in rows]
        h = write_manifest(man_dir / fname, mrows)
        base = {"file": f"manifests/{fname}", "description": desc, "sha256": h}
        manifests[name] = {**base, **counts(mrows, private=True)}
        manifests_public[name] = {**base, **counts(mrows, private=False)}
        print(f"   {fname}: {len(mrows)} rows, sha256 {h}", flush=True)

    # i manifest scritti devono essere quelli dichiarati (righe e SHA-256) nel file delle regole versionato
    declared = (load_json(RULES).get("manifests") or {}) if RULES.is_file() else {}
    differ = sorted(n for n, m in manifests.items() if not isinstance(declared.get(n), dict)
                    or declared[n].get("expected_sha256") != m["sha256"] or declared[n].get("expected_rows") != m["n"])
    check("manifests_equal_declared_in_rules", RULES.is_file() and not differ,
          f"manifests differ from {st.rel(RULES)} (rows or SHA-256): {differ}; update the declared values only after review")

    # 7. collegamento ai checkpoint delle quattro CNN official
    print("== checkpoint linkage", flush=True)
    c1_cfg = load_json(C1_CONFIG) if C1_CONFIG.is_file() else {}
    c1_ckpt = {c.get("card"): c for c in c1_cfg.get("checkpoints", []) if isinstance(c, dict)}
    expected_counts = {"n_train": len(cnn_train), "n_val": len(cnn_es), "n_test": len(cnn_test)}
    checkpoints = {}
    for arch in ARCHS:
        card_p = WEIGHTS / f"{arch}__nuaa.json"
        card = load_json(card_p)
        h5, keras = WEIGHTS / f"{arch}__nuaa.h5", WEIGHTS / f"{arch}__nuaa.keras"
        h5_sha = sha256(h5) if h5.is_file() else None
        test = card.get("test") or {}
        entry = {"card": card_p.name, "weights_file": h5.name, "weights_sha256": card.get("weights_sha256"),
                 "weights_sha256_verified": h5_sha is not None and h5_sha == card.get("weights_sha256"),
                 "c1_config_sha256_equal": (c1_ckpt.get(card_p.name, {}).get("sha256") == card.get("weights_sha256")) if c1_ckpt else None,
                 "keras_copy": {"file": keras.name, "sha256": sha256(keras)} if keras.is_file() else None,
                 "protocol": card.get("protocol"),
                 "val_split": {"per soggetto": "by subject", "casuale": "random"}.get(card.get("val_split"), card.get("val_split")),
                 "val_split_note": "English translation of the value recorded in the card", "saved": card.get("saved"),
                 "epochs_run": card.get("epochs_run"),
                 "early_stopping_fired": isinstance(card.get("epochs_run"), int) and card["epochs_run"] < MAX_EPOCHS,
                 "weights_epoch": ("best val_loss epoch (early stopping fired, restore_best_weights)"
                                   if isinstance(card.get("epochs_run"), int) and card["epochs_run"] < MAX_EPOCHS
                                   else f"last epoch (epoch limit {MAX_EPOCHS} reached, C1 checkpoints/PROVENANCE.md)"),
                 "card_counts": {k: card.get(k) for k in expected_counts},
                 "card_test_by_class": {"n_bona_fide": test.get("n_real"), "n_attack": test.get("n_attack")},
                 "manifests": {"n_train": "cnn_train", "n_val": "cnn_es", "n_test": "cnn_test"}}
        entry["counts_match_manifests"] = (entry["card_counts"] == expected_counts
                                           and test.get("n_real") == manifests["cnn_test"]["n_bona_fide"]
                                           and test.get("n_attack") == manifests["cnn_test"]["n_attack"])
        check(f"weights_sha256:{arch}", entry["weights_sha256_verified"], f"{h5.name} missing or different from the card")
        check(f"card_counts:{arch}", entry["counts_match_manifests"], f"card {entry['card_counts']} vs manifests {expected_counts}")
        check(f"protocol_official:{arch}", card.get("protocol") == "official" and card.get("dataset") == "nuaa", "not a NUAA official card")
        if entry["c1_config_sha256_equal"] is not None:
            check(f"c1_config_checkpoint:{arch}", entry["c1_config_sha256_equal"], "SHA-256 differs from results/c1/nuaa_config.json")
        checkpoints[arch] = entry
        print(f"   {arch}: {h5.name} sha256 verified {entry['weights_sha256_verified']}, counts match {entry['counts_match_manifests']}", flush=True)

    # 8. esclusioni: immagini ufficiali non campionate da spread(..., 1500)
    sampled = set(cnn_train) | set(cnn_es)
    excl_train = [r for r in official["train"] if r["key"] not in sampled]
    excl_test = [r for r in official["test"] if r["key"] not in set(cnn_test)]
    exclusions = {"official_train_not_sampled": counts(excl_train, private=True),
                  "official_test_not_sampled": counts(excl_test, private=True),
                  "rule": "spread(sorted file list, 1500) per class (scripts/train_cnn.py build_arrays); the split lists are otherwise complete"}

    # S6: fra le coppie dei soggetti del train, quante danno esattamente il numero di immagini di validazione delle schede
    from itertools import combinations
    per_subject = Counter(rec["groups"].tolist())
    n_val_card = len(cnn_es)
    pairs = [(x, y) for x, y in combinations(sorted(per_subject), 2) if per_subject[x] + per_subject[y] == n_val_card]
    pair_check = {"n_training_subjects": len(per_subject), "n_held_out_subjects": len(rec["val_subjects"]),
                  "n_pairs": len(list(combinations(per_subject, 2))), "n_pairs_with_card_n_val": len(pairs),
                  "unique_pair_is_the_reconstructed_one": len(pairs) == 1 and list(pairs[0]) == rec["val_subjects"],
                  "note": "independently of the scikit-learn version, only one pair of training subjects gives the validation count of the cards"}
    check("unique_validation_pair", pair_check["unique_pair_is_the_reconstructed_one"],
          f"{len(pairs)} pairs of training subjects give {n_val_card} validation images")
    reconstruction = {"function": "scripts/select_threshold.py reconstruct_split / check_reconstruction (copies of scripts/train_cnn.py build_arrays and train)",
                      "sampling": f"spread(sorted file list, {PER_CLASS}) per class, bona fide first",
                      "split": f"GroupShuffleSplit(n_splits=1, test_size=0.2, random_state={st.SEED}) on the subject prefix",
                      "n_sampled": int(len(rec["y"])), "n_train": len(cnn_train), "n_val": len(cnn_es),
                      "matches_checkpoint_cards": True, "training_cache": cache, "validation_pair_uniqueness": pair_check,
                      "original_file_list_recorded": False,
                      "note": "the training run did not save its file list; the list is rebuilt from the sorted directory listing and "
                              "verified against the card counts and the subject/label arrays of the training cache"}
    passed = not problems
    common = {"schema": MANIFEST_SCHEMA, "source": source, "official_lists": lists, "prepared_vs_archive": raw_check,
              "reconstruction": reconstruction, "checkpoints": checkpoints, "c1_correction": c1_info,
              "m1_dev_cross_check": m1_check, "declarations": {
                  "early_stopping_reuse": f"{len(set(cnn_es) & set(m1_dev))} of the {len(m1_dev)} M1 dev images are the validation "
                                          "images (`cnn_es`) of the four CNNs",
                  "validation_use": VALIDATION_USE,
                  "validation_use_source": "scripts/train_cnn.py (train: callbacks EarlyStopping and ReduceLROnPlateau on val_loss); "
                                           "research/c1-livenesslab/checkpoints/PROVENANCE.md (the 12/09 weights come from an earlier version of "
                                           "the script with the same hyper-parameters; it restored the best weights only when early stopping fired)",
                  "c1_from_dev_subjects": f"{len(c1_dev_attacks)} C1 test images come from the two validation subjects of the CNN training "
                                          f"(session {', '.join(c1_info['sessions_from_dev_subjects']) or '-'}; the dev images of those subjects are sessions "
                                          f"{'/'.join(sorted({rows[k]['session'] for k in m1_dev}))})"},
              "checks": checks, "passed": passed, "problems": problems, **repo_state(), "created": now()}
    summary = {**common, "manifests": manifests, "exclusions": exclusions,
               "early_stopping_subjects": rec["val_subjects"], "fitted_subjects": rec["fit_subjects"],
               "c1_from_dev_subjects_ids": sorted({r["subject"] for r in c1_dev_attacks}), "private": True}
    exclusions_public = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("by_subject")}
                         for k, v in exclusions.items() if isinstance(v, dict)}
    aggregate = {**common, "manifests": manifests_public, "recount": recount_block({n: [rows[k] for k in sets[n] if k in rows] for n in RECOUNT_MANIFESTS}),
                 "exclusions": {**exclusions_public, "rule": exclusions["rule"]},
                 "early_stopping_subjects": {"n": len(rec["val_subjects"])}, "fitted_subjects": {"n": len(rec["fit_subjects"])},
                 "private": False,
                 "note": "aggregate counts only: per-image manifests and per-subject counts are restricted (summary JSON in the private folder)"}
    out.mkdir(parents=True, exist_ok=True)
    summ_p, agg_p = out / "nuaa_manifests_summary.json", out / "nuaa_manifests_aggregate.json"
    summ_p.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    agg_text = json.dumps(aggregate, indent=1) + "\n"
    subject_ids = {r["subject"] for r in rows.values()}
    bad = public_problems(agg_text, subject_ids)
    if bad:
        problems.append("public aggregate contains forbidden content: " + "; ".join(bad))
        aggregate["passed"] = False
        aggregate["problems"] = problems
        agg_text = json.dumps(aggregate, indent=1) + "\n"
    agg_p.write_text(agg_text, encoding="utf-8")
    print(f"== {relpath(summ_p)} (private)\n== {relpath(agg_p)} (public aggregate, scan: {'clean' if not bad else 'FORBIDDEN CONTENT'})", flush=True)
    if problems:
        fail("build checks failed: " + "; ".join(problems))
    print(f"== build checks passed ({len(checks)} checks)", flush=True)


# ----------------------------------------------------------------------------- check-overlap

DIMENSIONS = ("key", "hash", "subject", "subject_session")
KINDS = ("disjoint", "subset", "intersection", "no_internal_duplicates", "spread_partition", "equals_spread", "subject_closure")
DERIVATIONS = ("spread_partition", "equals_spread", "subject_closure")   # regole di derivazione, sempre per chiave
SEVERITIES = ("must", "report")


def values(rows, dim: str):
    """Valore di ogni riga nella dimensione dim (key, hash, subject, subject_session)."""
    if dim == "key":
        return [r["key"] for r in rows]
    if dim == "hash":
        return [r["sha256"] for r in rows]
    if dim == "subject":
        return [r["subject"] for r in rows]
    if dim == "subject_session":
        return [(r["subject"], r["session"]) for r in rows]
    raise ValueError(f"unknown dimension {dim!r}")


def pair_overlap(a_rows, b_rows, dim: str) -> dict:
    """Valori distinti in comune fra due manifest e righe di ciascuno che li hanno."""
    va, vb = values(a_rows, dim), values(b_rows, dim)
    shared = set(va) & set(vb)
    return {"n_shared": len(shared), "rows_a": sum(v in shared for v in va), "rows_b": sum(v in shared for v in vb),
            "_shared": shared}


def internal_duplicates(rows, dim: str) -> dict:
    c = Counter(values(rows, dim))
    dup = {v: n for v, n in c.items() if n > 1}
    return {"n_values": len(dup), "n_rows": sum(dup.values()), "_values": set(dup)}


def as_list(x):
    return x if isinstance(x, list) else [x]


def rule_problems(doc: dict, names) -> list:
    """Struttura del file delle regole: manifest dichiarati, regole con id univoco, tipo, severità, dimensioni e
    manifest noti; `expected` intero solo per le regole intersection (obbligatorio)."""
    problems = []
    if not isinstance(doc.get("manifests"), dict) or not doc["manifests"]:
        problems.append("'manifests' must be a non-empty object {name: {file, expected_rows, expected_sha256}}")
    else:
        for n, m in doc["manifests"].items():
            if not isinstance(m, dict) or not isinstance(m.get("file"), str) or not m.get("file") \
                    or "/" in m["file"] or "\\" in m["file"] or m["file"].startswith("."):
                problems.append(f"manifest {n!r}: needs 'file' (a plain file name in the manifests folder)"); continue
            if type(m.get("expected_rows")) is not int or m["expected_rows"] < 1:
                problems.append(f"manifest {n!r}: 'expected_rows' must be an integer >= 1 (an empty manifest would make the rules vacuous)")
            if not SHA_RE.match(str(m.get("expected_sha256") or "")):
                problems.append(f"manifest {n!r}: 'expected_sha256' must be a SHA-256 (64 lower-case hexadecimal characters)")
    rules = doc.get("rules")
    if not isinstance(rules, list) or not rules:
        return problems + ["'rules' must be a non-empty list"]
    ids = [r.get("id") for r in rules if isinstance(r, dict)]
    for i, n in Counter(ids).items():
        if n > 1:
            problems.append(f"duplicated rule id {i!r}")
    for r in rules:
        if not isinstance(r, dict):
            problems.append(f"rule {r!r} is not an object"); continue
        rid = r.get("id")
        if not isinstance(rid, str) or not rid:
            problems.append(f"rule without an id: {r}")
        if r.get("kind") not in KINDS:
            problems.append(f"{rid}: kind {r.get('kind')!r} not in {KINDS}")
        if r.get("severity") not in SEVERITIES:
            problems.append(f"{rid}: severity {r.get('severity')!r} not in {SEVERITIES}")
        by = r.get("by")
        if not isinstance(by, list) or not by or any(d not in DIMENSIONS for d in by):
            problems.append(f"{rid}: 'by' must be a non-empty list of {DIMENSIONS}")
        if r.get("kind") in DERIVATIONS:
            problems += derivation_problems(r, rid, names)
            continue
        sides = ("manifests",) if r.get("kind") == "no_internal_duplicates" else ("a", "b")
        for s in sides:
            v = r.get(s)
            vals = names if v == "*" else as_list(v)
            if not vals or any(n not in names for n in vals):
                problems.append(f"{rid}: {s} = {v!r} names unknown manifests (declared: {sorted(names)})")
        if sides == ("a", "b") and set(names if r.get("a") == "*" else as_list(r.get("a"))) & set(names if r.get("b") == "*" else as_list(r.get("b"))):
            problems.append(f"{rid}: the same manifest is on both sides (a and b must differ)")
        if r.get("kind") == "intersection" and (type(r.get("expected")) is not int or r["expected"] < 0):
            problems.append(f"{rid}: intersection rules need an integer 'expected' >= 0")
        if r.get("kind") != "intersection" and "expected" in r:
            problems.append(f"{rid}: 'expected' is only allowed in intersection rules")
    return problems


def derivation_problems(r: dict, rid, names) -> list:
    """Campi delle regole di derivazione: spread_partition (source, parts, per_class), equals_spread (a, source,
    per_class), subject_closure (a, source, subjects_of); sempre by = ["key"], manifest noti e diversi dalla sorgente."""
    problems = []
    if r.get("by") != ["key"]:
        problems.append(f"{rid}: derivation rules compare keys only ('by': ['key'])")
    fields = {"spread_partition": ("source", "parts"), "equals_spread": ("a", "source"), "subject_closure": ("a", "source", "subjects_of")}[r["kind"]]
    for f in fields:
        v = r.get(f)
        vals = as_list(v) if f == "parts" else [v]
        if not vals or any(not isinstance(n, str) or n not in names for n in vals):
            problems.append(f"{rid}: {f} = {v!r} names unknown manifests (declared: {sorted(names)})")
    if r["kind"] != "spread_partition" and isinstance(r.get("a"), list):
        problems.append(f"{rid}: 'a' must be a single manifest")
    targets = as_list(r.get("parts")) if r["kind"] == "spread_partition" else [r.get("a")]
    if r.get("source") in targets or (r["kind"] == "subject_closure" and r.get("subjects_of") == r.get("a")):
        problems.append(f"{rid}: the derived manifest cannot be its own source")
    if r["kind"] == "spread_partition" and len(set(as_list(r.get("parts")))) != len(as_list(r.get("parts"))):
        problems.append(f"{rid}: 'parts' lists the same manifest twice")
    if r["kind"] in ("spread_partition", "equals_spread") and (type(r.get("per_class")) is not int or r["per_class"] < 1):
        problems.append(f"{rid}: 'per_class' must be an integer >= 1")
    if "expected" in r:
        problems.append(f"{rid}: 'expected' is only allowed in intersection rules")
    return problems


def spread_keys(rows, per_class: int) -> list:
    """Chiavi di `spread(elenco ordinato dei nomi, per_class)` per classe, bona fide e poi attacchi: lo stesso
    campionamento di train_cnn.py (cartelle) e di prepare_datasets.py (liste ufficiali), a parità di ordinamento."""
    out = []
    for lab in ("0", "1"):
        names = [name_of(r["key"]) for r in rows if str(r["label"]) == lab]
        out += [make_key(int(lab), n) for n in spread(names, per_class)]
    return out


def derivation_check(r: dict, mans: dict) -> list:
    """Verifiche di una regola di derivazione: il manifest (o l'unione delle parti) deve coincidere, come multinsieme
    di chiavi, con quello derivato dalla sorgente; le parti di una partizione devono essere disgiunte."""
    src = mans[r["source"]]
    if r["kind"] == "spread_partition":
        expected = Counter(spread_keys(src, r["per_class"]))
        actual = Counter(k for n in as_list(r["parts"]) for k in values(mans[n], "key"))
        target = " + ".join(as_list(r["parts"]))
    elif r["kind"] == "equals_spread":
        expected, actual, target = Counter(spread_keys(src, r["per_class"])), Counter(values(mans[r["a"]], "key")), r["a"]
    else:
        subj = {x["subject"] for x in mans[r["subjects_of"]]}
        expected = Counter(x["key"] for x in src if x["subject"] in subj)
        actual, target = Counter(values(mans[r["a"]], "key")), r["a"]
    missing, extra = expected - actual, actual - expected
    c = {"a": target, "b": r["source"], "by": "key", "derivation": r["kind"], "n_expected": sum(expected.values()),
         "n_rows": sum(actual.values()), "n_missing": sum(missing.values()), "n_extra": sum(extra.values())}
    ok = not missing and not extra
    c["outcome"] = "pass" if ok else ("FAIL" if r["severity"] == "must" else "reported")
    if not ok and r["severity"] == "must":
        c["examples"] = examples_of(set(missing) | set(extra))
    checks = [c]
    if r["kind"] == "spread_partition":
        parts = as_list(r["parts"])
        shared = set()
        for i, an in enumerate(parts):
            for bn in parts[i + 1:]:
                shared |= set(values(mans[an], "key")) & set(values(mans[bn], "key"))
        d = {"a": " + ".join(parts), "b": "-", "by": "key", "derivation": "parts_disjoint", "n_shared": len(shared)}
        d["outcome"] = "pass" if not shared else ("FAIL" if r["severity"] == "must" else "reported")
        if shared and r["severity"] == "must":
            d["examples"] = examples_of(shared)
        checks.append(d)
    return checks


def examples_of(shared, k=3):
    return [("/".join(v) if isinstance(v, tuple) else v) for v in sorted(shared)[:k]]


def evaluate_rules(mans: dict, rules: list) -> list:
    """Verdetto di ogni regola, espansa sulle coppie (a, b) e sulle dimensioni. Per ogni verifica: conteggi, esito
    (pass, FAIL, reported) ed esempi solo per le verifiche "must" fallite."""
    verdicts = []
    for r in rules:
        must = r["severity"] == "must"
        checks = []
        if r["kind"] in DERIVATIONS:
            checks = derivation_check(r, mans)
        elif r["kind"] == "no_internal_duplicates":
            names = list(mans) if r["manifests"] == "*" else as_list(r["manifests"])
            for n, dim in product(names, r["by"]):
                d = internal_duplicates(mans[n], dim)
                ok = d["n_values"] == 0
                c = {"manifest": n, "by": dim, "duplicated_values": d["n_values"], "rows": d["n_rows"]}
                c["outcome"] = "pass" if ok else ("FAIL" if must else "reported")
                if not ok and must:
                    c["examples"] = examples_of(d["_values"])
                checks.append(c)
        else:
            for an, bn, dim in product(as_list(r["a"]), as_list(r["b"]), r["by"]):
                if an == bn:
                    continue
                o = pair_overlap(mans[an], mans[bn], dim)
                c = {"a": an, "b": bn, "by": dim, "n_shared": o["n_shared"], "rows_a": o["rows_a"], "rows_b": o["rows_b"]}
                bad = set()
                if r["kind"] == "disjoint":
                    ok = o["n_shared"] == 0
                    bad = o["_shared"]
                elif r["kind"] == "subset":
                    missing = set(values(mans[an], dim)) - set(values(mans[bn], dim))
                    c["n_not_in_b"] = len(missing)
                    ok = not missing
                    bad = missing
                else:   # intersection con numero dichiarato di valori in comune
                    c["expected"] = r["expected"]
                    ok = o["n_shared"] == r["expected"]
                c["outcome"] = "pass" if ok else ("FAIL" if must else "reported")
                if not ok and must and bad:
                    c["examples"] = examples_of(bad)
                checks.append(c)
        # nessuna regola passa a vuoto: un manifest senza righe o una regola senza verifiche è un fallimento
        if r["kind"] in DERIVATIONS:
            involved = [n for f in ("a", "parts", "source", "subjects_of") if f in r for n in as_list(r[f])]
        elif r["kind"] == "no_internal_duplicates":
            involved = list(mans) if r.get("manifests") == "*" else as_list(r["manifests"])
        else:
            involved = as_list(r["a"]) + as_list(r["b"])
        empty = sorted(n for n in involved if not mans[n])
        if empty or not checks:
            checks.append({"by": "-", "outcome": "FAIL", "vacuous": True, "empty_manifests": empty})
        failed = any(c["outcome"] == "FAIL" for c in checks)
        if must or failed:      # una regola "report" fallisce solo se è vuota
            verdict = "FAIL" if failed else "pass"
        else:
            verdict = "reported" if any(c["outcome"] == "reported" for c in checks) else "pass"
        verdicts.append({"id": r["id"], "kind": r["kind"], "severity": r["severity"], "by": r["by"],
                         "description": r.get("description", ""), "verdict": verdict, "checks": checks})
    return verdicts


def overlap_matrix(mans: dict) -> dict:
    """Matrice a coppie (tutte le coppie non ordinate) per le quattro dimensioni, più i duplicati interni."""
    names = list(mans)
    pairs = {}
    for i, an in enumerate(names):
        for bn in names[i + 1:]:
            pairs[f"{an}|{bn}"] = {dim: {k: v for k, v in pair_overlap(mans[an], mans[bn], dim).items() if not k.startswith("_")}
                                   for dim in DIMENSIONS}
    internal = {n: {dim: {k: v for k, v in internal_duplicates(mans[n], dim).items() if not k.startswith("_")}
                    for dim in ("key", "hash")} for n in names}
    distinct = {n: {dim: len(set(values(mans[n], dim))) for dim in DIMENSIONS} | {"rows": len(mans[n])} for n in names}
    return {"manifests": names, "distinct": distinct, "pairs": pairs, "internal_duplicates": internal}


def plural(n: int, word: str) -> str:
    """Numero con il sostantivo inglese al singolare o al plurale ("1 rule", "2 rules")."""
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def overlap_markdown(rep: dict) -> str:
    names = rep["matrix"]["manifests"]
    L = ["# NUAA overlap report", "",
         f"Rules: `{rep['rules_file']}` (SHA-256 `{rep['rules_sha256']}`). Generated by `scripts/nuaa_manifests.py check-overlap`, "
         f"commit `{rep['commit']}`{' (working tree modified)' if rep['dirty'] else ''}, {rep['checked']}.", "",
         f"**Result: {'PASS' if rep['passed'] else 'FAIL'}** ({plural(rep['n_must_failed'], 'must-hold rule')} failed, "
         f"{plural(rep['n_reported'], 'rule')} with reported overlaps).", "",
         "Duplicates are exact (file SHA-256 or identical key); near-duplicates (consecutive frames, the same printed photograph) "
         "are not searched. This check does not establish absence of leakage.", "", "## Manifests", "",
         "| name | file | rows | keys | hashes | subjects | (subject, session) | SHA-256 |", "|---|---|---|---|---|---|---|---|"]
    for n in names:
        d, m = rep["matrix"]["distinct"][n], rep["manifests"][n]
        L.append(f"| {n} | `{m['file']}` | {d['rows']} | {d['key']} | {d['hash']} | {d['subject']} | {d['subject_session']} | `{m['sha256'][:16]}…` |")
    L += ["", "## Rule verdicts", "", "| rule | kind | severity | by | verdict | description |", "|---|---|---|---|---|---|"]
    for v in rep["rules"]:
        L.append(f"| {v['id']} | {v['kind']} | {v['severity']} | {', '.join(v['by'])} | **{v['verdict']}** | {v['description']} |")
    L += ["", "Checks of the rules with a non-zero count, and every derivation check:", ""]
    any_nz = False
    for v in rep["rules"]:
        for c in v["checks"]:
            n = c.get("n_shared", c.get("duplicated_values", 0)) or c.get("n_not_in_b", 0)
            if n or c["outcome"] != "pass" or "derivation" in c:
                any_nz = True
                if c.get("vacuous"):
                    L.append(f"- {v['id']}: vacuous rule (empty manifests: {', '.join(c['empty_manifests']) or 'none'}) → FAIL")
                    continue
                if "derivation" in c:
                    detail = (f"shared {c['n_shared']}" if c["derivation"] == "parts_disjoint" else
                              f"rows {c['n_rows']}, derived {c['n_expected']}, missing {c['n_missing']}, extra {c['n_extra']}")
                    L.append(f"- {v['id']} ({c['a']} from {c['b']}, {c['derivation']}): {detail} → {c['outcome']}"
                             f"{' (e.g. ' + ', '.join(map(str, c['examples'])) + ')' if c.get('examples') else ''}")
                    continue
                where = c.get("manifest") or f"{c['a']} vs {c['b']}"
                extra = f", expected {c['expected']}" if "expected" in c else ""
                L.append(f"- {v['id']} ({where}, by {c['by']}): shared {c.get('n_shared', c.get('duplicated_values'))}"
                         f", rows {c.get('rows_a', c.get('rows'))}/{c.get('rows_b', '-')}{extra}"
                         f"{', not in b ' + str(c['n_not_in_b']) if 'n_not_in_b' in c else ''} → {c['outcome']}"
                         f"{' (e.g. ' + ', '.join(map(str, c['examples'])) + ')' if c.get('examples') else ''}")
    if not any_nz:
        L.append("- none")
    for dim in DIMENSIONS:
        L += ["", f"## Pairwise matrix: shared {dim.replace('_', ', ')} values (rows of the row manifest / rows of the column manifest)", "",
              "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for an in names:
            cells = []
            for bn in names:
                if an == bn:
                    cells.append(f"({rep['matrix']['distinct'][an][dim]})")
                    continue
                k = f"{an}|{bn}" if f"{an}|{bn}" in rep["matrix"]["pairs"] else f"{bn}|{an}"
                o = rep["matrix"]["pairs"][k][dim]
                ra, rb = (o["rows_a"], o["rows_b"]) if k.startswith(an + "|") else (o["rows_b"], o["rows_a"])
                cells.append(f"{o['n_shared']} ({ra}/{rb})" if o["n_shared"] else "0")
            L.append(f"| **{an}** | " + " | ".join(cells) + " |")
    L += ["", "Diagonal: number of distinct values in the manifest.", "", "## Internal duplicates", "",
          "| manifest | duplicated keys | duplicated hashes (rows) |", "|---|---|---|"]
    for n in names:
        d = rep["matrix"]["internal_duplicates"][n]
        L.append(f"| {n} | {d['key']['n_values']} | {d['hash']['n_values']} ({d['hash']['n_rows']}) |")
    return "\n".join(L) + "\n"


SELF_TEST_NAMES = ("official_train", "official_test", "cnn_train", "cnn_es", "cnn_test", "m1_dev", "c1")


def self_test_faults(m: dict) -> list:
    """Guasti del self-test sui manifest reali: (id, descrizione, manifest modificati, regole che devono fallire, ed
    esattamente quelle). Le righe scelte sono le prime in ordine di chiave che rispettano la condizione."""
    es_keys = {r["key"] for r in m["cnn_es"]}
    c1_keys = {r["key"] for r in m["c1"]}
    dev_row = next(r for r in m["m1_dev"] if r["key"] not in es_keys)
    test_row = next(r for r in m["cnn_test"] if r["key"] not in c1_keys)
    return [
        ("F1-dev-row-removed", "one M1 dev row that is not a validation image removed from the dev",
         {"m1_dev": [r for r in m["m1_dev"] if r is not dev_row]}, {"R12-dev-subject-closure"}),
        ("F2-train-row-removed", "first CNN train row removed", {"cnn_train": m["cnn_train"][1:]}, {"R11-cnn-split-partition"}),
        ("F3-validation-row-in-train", "first validation row added to the CNN train",
         {"cnn_train": m["cnn_train"] + [m["cnn_es"][0]]}, {"R02-train-vs-es", "R03-train-vs-dev", "R11-cnn-split-partition"}),
        ("F4-train-row-in-c1", "first CNN train row added to C1", {"c1": m["c1"] + [m["cnn_train"][0]]},
         {"R04-train-side-vs-test-side", "R09-test-chain", "R10-c1-in-cnn-test", "R14-c1-spread"}),
        ("F5-train-hash-under-test-name", "a CNN test row outside C1 gets the SHA-256 of the first CNN train image",
         {"cnn_test": [dict(r, sha256=m["cnn_train"][0]["sha256"]) if r is test_row else r for r in m["cnn_test"]]},
         {"R04-train-side-vs-test-side", "R09-test-chain"}),
        ("F6-duplicated-row", "first CNN test row duplicated", {"cnn_test": m["cnn_test"] + [m["cnn_test"][0]]},
         {"R01-internal-duplicates", "R13-cnn-test-spread"}),
        ("F7-c1-row-removed-from-official-test", "first C1 row removed from the official test",
         {"official_test": [r for r in m["official_test"] if r["key"] != m["c1"][0]["key"]]},
         {"R09-test-chain", "R13-cnn-test-spread", "R14-c1-spread"}),
    ]


def write_rows(p: Path, rows) -> None:
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n", extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def run_self_test(doc: dict, rules_p: Path, man_dir: Path, mans: dict, out: Path) -> bool:
    """Ripete check-overlap (sottoprocesso, stesso script) su copie guaste dei manifest con righe e SHA-256 ridichiarati
    in una copia delle regole: ogni guasto deve far fallire esattamente le regole attese, con codice 1."""
    import subprocess
    import tempfile
    missing = [n for n in SELF_TEST_NAMES if n not in mans]
    if missing:
        fail(f"--self-test needs the NUAA manifests {list(SELF_TEST_NAMES)}; missing {missing}")
    results = []
    with tempfile.TemporaryDirectory(prefix="m2-self-test-") as tmp:
        for fid, desc, changed, expected in self_test_faults(mans):
            d = Path(tmp) / fid
            d.mkdir()
            rules = json.loads(json.dumps(doc))
            for n, m in rules["manifests"].items():
                q = d / m["file"]
                if n in changed:
                    write_rows(q, changed[n])
                    m["expected_rows"], m["expected_sha256"] = len(changed[n]), sha256(q)
                else:
                    q.write_bytes((man_dir / m["file"]).read_bytes())
            (d / "rules.json").write_text(json.dumps(rules, indent=1), encoding="utf-8")
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "check-overlap", "--rules", str(d / "rules.json"),
                                "--manifests-dir", str(d), "--out", str(d / "out")], capture_output=True, text=True)
            rep_p = d / "out" / "overlap_report.json"
            failing = sorted(v["id"] for v in json.loads(rep_p.read_text())["rules"] if v["verdict"] == "FAIL") if rep_p.exists() else []
            ok = r.returncode == 1 and set(failing) == expected
            results.append({"fault": fid, "description": desc, "changed_manifests": sorted(changed), "expected_failing_rules": sorted(expected),
                            "failing_rules": failing, "exit_code": r.returncode, "detected_as_expected": ok})
            print(f"   self-test {fid:<38} {'ok' if ok else 'NOT AS EXPECTED'} (failing: {', '.join(failing) or 'none'})", flush=True)
    passed = all(x["detected_as_expected"] for x in results)
    st_rep = {"rules_file": st.rel(rules_p), "rules_sha256": sha256(rules_p), "faults": results, "n_faults": len(results),
              "n_detected_as_expected": sum(x["detected_as_expected"] for x in results), "passed": passed,
              "note": "each faulty copy of the real manifests is checked with its rows and SHA-256 re-declared; it must fail exactly the expected rules",
              **repo_state(), "checked": now()}
    (out / "overlap_self_test.json").write_text(json.dumps(st_rep, indent=1) + "\n", encoding="utf-8")
    L = ["# NUAA overlap self-test", "", f"**Result: {'PASS' if passed else 'FAIL'}** ({st_rep['n_detected_as_expected']} of "
         f"{len(results)} faulty copies rejected by exactly the expected rules).", "",
         "| fault | change | expected failing rules | failing rules | exit code | as expected |", "|---|---|---|---|---|---|"]
    L += [f"| {x['fault']} | {x['description']} | {', '.join(x['expected_failing_rules'])} | {', '.join(x['failing_rules']) or 'none'} | "
          f"{x['exit_code']} | {'yes' if x['detected_as_expected'] else 'NO'} |" for x in results]
    (out / "overlap_self_test.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return passed


def cmd_check_overlap(a) -> None:
    rules_p = Path(a.rules)
    doc = load_json(rules_p)
    names = list((doc.get("manifests") or {}).keys()) if isinstance(doc.get("manifests"), dict) else []
    problems = rule_problems(doc, names)
    if problems:
        fail(f"{relpath(rules_p)}: " + "; ".join(problems))
    man_dir = Path(a.manifests_dir)
    mans, info, mismatch = {}, {}, []
    for n, m in doc["manifests"].items():
        p = man_dir / m["file"]
        mans[n] = load_manifest(p)
        info[n] = {"file": m["file"], "sha256": sha256(p), "rows": len(mans[n])}
        # ogni manifest deve essere quello dichiarato: righe e SHA-256 attesi (un manifest troncato non passa a vuoto)
        if info[n]["rows"] != m["expected_rows"]:
            mismatch.append(f"{n}: {info[n]['rows']} rows, expected {m['expected_rows']}")
        if info[n]["sha256"] != m["expected_sha256"]:
            mismatch.append(f"{n}: SHA-256 {info[n]['sha256'][:16]}… differs from the declared {m['expected_sha256'][:16]}…")
    if mismatch:
        fail(f"manifests differ from those declared in {relpath(rules_p)}: " + "; ".join(mismatch))
    verdicts = evaluate_rules(mans, doc["rules"])
    n_fail = sum(v["verdict"] == "FAIL" for v in verdicts)
    # percorsi relativi al repository, oppure il solo nome fuori dal repository: nessun percorso personale nel rapporto
    rep = {"rules_file": st.rel(rules_p), "rules_sha256": sha256(rules_p), "manifests_dir": st.rel(man_dir), "manifests": info,
           "rules": verdicts, "matrix": overlap_matrix(mans), "n_must_failed": n_fail,
           "n_reported": sum(v["verdict"] == "reported" for v in verdicts), "passed": n_fail == 0,
           "note": "restricted version: example keys are listed for failed must-hold checks; shared subjects are counts, never ids",
           **repo_state(), "checked": now()}
    # versione pubblica: stessi conteggi e verdetti, senza esempi di chiavi; controllata con la scansione del contenuto vietato
    pub = json.loads(json.dumps(rep))
    for v in pub["rules"]:
        for c in v["checks"]:
            c.pop("examples", None)
    pub["note"] = "public version: counts and verdicts only, no keys and no subject identifiers"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "overlap_report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    (out / "overlap_report.md").write_text(overlap_markdown(rep), encoding="utf-8")
    subject_ids = {r["subject"] for rows in mans.values() for r in rows}
    pub_json, pub_md = json.dumps(pub, indent=1) + "\n", overlap_markdown(pub)
    bad = public_problems(pub_json, subject_ids) + public_problems(pub_md, subject_ids, markdown=True)
    if bad:
        for f in ("overlap_report_public.json", "overlap_report_public.md"):
            (out / f).unlink(missing_ok=True)
        fail("public overlap report contains forbidden content (not written): " + "; ".join(sorted(set(bad))))
    (out / "overlap_report_public.json").write_text(pub_json, encoding="utf-8")
    (out / "overlap_report_public.md").write_text(pub_md, encoding="utf-8")
    for v in verdicts:
        print(f"   {v['id']:<28} {v['severity']:<7} {v['verdict']}", flush=True)
    print(f"== {relpath(out / 'overlap_report.json')} and .md (restricted), overlap_report_public.json and .md (public)", flush=True)
    if n_fail:
        fail(f"{n_fail} must-hold overlap rules failed: " + ", ".join(v["id"] for v in verdicts if v["verdict"] == "FAIL"))
    print(f"== overlap checks passed ({plural(len(verdicts), 'rule')}, {rep['n_reported']} with reported overlaps)", flush=True)
    if getattr(a, "self_test", False):
        print("== self-test on faulty copies of the manifests", flush=True)
        if not run_self_test(doc, rules_p, man_dir, mans, out):
            fail("self-test failed: a faulty copy was not rejected by exactly the expected rules (overlap_self_test.md)")
        print(f"== self-test passed: {relpath(out / 'overlap_self_test.json')} and .md", flush=True)


# ----------------------------------------------------------------------------- recount, reconstruct, compare-c1

def load_named_manifests(rules_p: Path, man_dir: Path) -> dict:
    """Manifest nominati nel file delle regole, verificati su righe e SHA-256 dichiarati (come check-overlap)."""
    doc = load_json(rules_p)
    problems = rule_problems(doc, list((doc.get("manifests") or {}).keys()) if isinstance(doc.get("manifests"), dict) else [])
    if problems:
        fail(f"{relpath(rules_p)}: " + "; ".join(problems))
    mans, mismatch = {}, []
    for n, m in doc["manifests"].items():
        p = man_dir / m["file"]
        mans[n] = load_manifest(p)
        if len(mans[n]) != m["expected_rows"] or sha256(p) != m["expected_sha256"]:
            mismatch.append(n)
    if mismatch:
        fail(f"manifests differ from those declared in {relpath(rules_p)} (rows or SHA-256): {mismatch}")
    return mans


def cmd_recount(a) -> None:
    """I conteggi dell'aggregato ricalcolati dai soli manifest devono coincidere con quelli scritti da build."""
    mans = load_named_manifests(Path(a.rules), Path(a.manifests_dir))
    missing = [n for n in RECOUNT_MANIFESTS if n not in mans]
    if missing:
        fail(f"recount needs the manifests {list(RECOUNT_MANIFESTS)}; missing {missing}")
    agg = load_json(Path(a.aggregate))
    fresh = recount_block(mans)
    problems = []
    for k, v in fresh.items():
        if (agg.get("recount") or {}).get(k) != v:
            problems.append(f"recount.{k} differs from the aggregate")
    doc = load_json(Path(a.rules))
    for n in RECOUNT_MANIFESTS:
        am = (agg.get("manifests") or {}).get(n, {})
        if am.get("sha256") != doc["manifests"][n]["expected_sha256"]:
            problems.append(f"manifests.{n}.sha256 differs from the declared manifest")
        if {k: am.get(k) for k in fresh["manifest_counts"][n]} != fresh["manifest_counts"][n]:
            problems.append(f"manifests.{n} counts differ from the recount")
    nov = fresh["test_from_subjects_absent_from_official_train"]
    for n, d in nov.items():
        print(f"   {n}: bona fide {d['bona_fide']['from_subjects_absent_from_official_train']:,}/{d['bona_fide']['n']:,}, "
              f"attacks {d['attack']['from_subjects_absent_from_official_train']:,}/{d['attack']['n']:,} from subjects absent from the official train", flush=True)
    dc = fresh["declaration_counts"]
    print(f"   M1 dev rows that are validation images: {dc['m1_dev_rows_that_are_validation_images']}/{dc['m1_dev_rows']}; "
          f"C1 rows from the validation subjects: {dc['c1_rows_from_validation_subjects']}", flush=True)
    if problems:
        fail("recount differs from the aggregate: " + "; ".join(problems))
    print(f"== recount equal to {st.rel(Path(a.aggregate))} ({len(RECOUNT_MANIFESTS)} manifests, exclusions, test novelty, declarations)", flush=True)


def reconstruct_from_manifest(train_rows):
    """Split delle CNN dal solo manifest del train ufficiale: `spread(nomi ordinati, 1500)` per classe (bona fide e poi
    attacchi, come build_arrays) e GroupShuffleSplit sui soggetti, come select_threshold.reconstruct_split."""
    from sklearn.model_selection import GroupShuffleSplit
    keys = spread_keys(train_rows, PER_CLASS)
    subj = {r["key"]: r["subject"] for r in train_rows}
    y = np.array([st.KEY_LABEL[k.split("/", 1)[0]] for k in keys], np.int64)
    groups = np.array([subj[k] for k in keys])
    tr_i, va_i = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=st.SEED).split(np.zeros(len(y)), y, groups))
    val_subjects = sorted(set(groups[va_i].tolist()))
    dev = [r["key"] for r in train_rows if r["subject"] in set(val_subjects)]
    return [keys[i] for i in tr_i], [keys[i] for i in va_i], dev


def cmd_reconstruct(a) -> None:
    mans = load_named_manifests(Path(a.rules), Path(a.manifests_dir))
    tr, va, dev = reconstruct_from_manifest(mans["official_train"])
    problems = []
    for name, keys in (("cnn_train", tr), ("cnn_es", va), ("m1_dev", dev)):
        have = sorted(r["key"] for r in mans[name])
        same = sorted(keys) == have
        print(f"   {name}: rebuilt {len(keys):,}, manifest {len(have):,}, identical: {'yes' if same else 'NO'}", flush=True)
        if not same:
            problems.append(name)
    if problems:
        fail(f"the split rebuilt from the official train manifest differs from {problems}")
    print("== CNN split and M1 dev rebuilt from nuaa_official_train.csv alone: identical (no image read)", flush=True)


def cmd_compare_c1(a) -> None:
    """C1 v1 (consegnato con C1, colonna `file`) contro v2 (colonna `key`): stesse chiavi, SHA-256, byte, soggetti ed
    etichette; la colonna session della v1 coincide con glasses della v2; sessione della v2 = 03."""
    v1 = read_csv(Path(a.v1), ["file", "label", "subject", "session", "sha256", "bytes"])
    v2 = {r["key"]: r for r in load_manifest(Path(a.v2))}
    print(f"   C1 v1 {st.rel(Path(a.v1))} SHA-256 {sha256(Path(a.v1))} "
          f"({'the C1 delivery' if sha256(Path(a.v1)) == C1_MANIFEST_SHA256 else 'NOT the C1 delivery'})", flush=True)
    problems, same, glasses = [], 0, 0
    if len(v1) != len(v2) or {r["file"] for r in v1} != set(v2):
        problems.append(f"different keys: v1 {len(v1)} rows, v2 {len(v2)} rows")
    for r in v1:
        x = v2.get(r["file"])
        if x is None:
            continue
        if all(r[c] == x[c] for c in ("label", "subject", "sha256", "bytes")):
            same += 1
        glasses += r["session"] == x["glasses"]
    sessions = Counter(x["session"] for x in v2.values())
    print(f"   rows equal by key, SHA-256, bytes, subject and label: {same}/{len(v1)}; v1 session = v2 glasses: {glasses}/{len(v1)}; "
          f"v2 sessions: {dict(sessions)}", flush=True)
    if same != len(v1) or glasses != len(v1):
        problems.append("rows differ")
    if sha256(Path(a.v1)) != C1_MANIFEST_SHA256:
        problems.append("the v1 file is not the manifest delivered with C1")
    if problems:
        fail("C1 v1/v2 comparison failed: " + "; ".join(problems))
    print("== C1 manifest version 2 = version 1 with the session field re-parsed", flush=True)


# ----------------------------------------------------------------------------- report

def pct_list(d: dict) -> str:
    return ", ".join(f"{k}: {v['bona_fide']:,} + {v['attack']:,}" for k, v in d.items())


def derived_text(d) -> str:
    """Frase del rapporto sui frame derivati della cache (o il motivo per cui il controllo non c'è)."""
    if not d:
        return "- not checked: the training cache is missing"
    t, e = d["train"], d["test"]
    return (f"- train frames: {t['n_frames']:,} ({t['distinct']:,} distinct, {t['duplicated_values']} duplicated values)\n"
            f"- test frames: {e['n_frames']:,} ({e['distinct']:,} distinct, {e['duplicated_values']} duplicated values)\n"
            f"- frames shared between train and test: {d['train_test_shared']}; between the fitting and the validation images "
            f"of the training split: {d['fit_vs_validation_shared']}")


def report_fields(summ: dict, ov: dict, private: bool, self_test: dict = None, agg: dict = None) -> dict:
    m = summ["manifests"]
    dev_ids = f" ({' and '.join(summ.get('early_stopping_subjects') or [])})" if private else ""
    yes = {True: "yes", False: "NO", None: "not checked"}
    rows = []
    for n in MANIFESTS:
        x = m[n]
        rows.append(f"| {n} | `{x['file']}` | {x['n']:,} | {x['n_bona_fide']:,} | {x['n_attack']:,} | {x['n_subjects']} | "
                    f"{pct_list(x['by_session'])} | `{x['sha256'][:16]}…` |")
    ck = []
    for arch, c in summ["checkpoints"].items():
        cc = c["card_counts"]
        ck.append(f"| {arch} | `{c['weights_file']}` | `{c['weights_sha256'][:16]}…` | {'yes' if c['weights_sha256_verified'] else 'NO'} | "
                  f"{cc['n_train']:,}/{cc['n_val']:,}/{cc['n_test']:,} | {'yes' if c['counts_match_manifests'] else 'NO'} | "
                  f"{c['epochs_run']}: {c['weights_epoch']} |")
    verdicts = [f"| {v['id']} | {v['severity']} | {', '.join(v['by'])} | **{v['verdict']}** | {v['description']} |" for v in ov["rules"]]
    reported = []
    for v in ov["rules"]:
        if v["verdict"] != "reported":
            continue
        for c in v["checks"]:
            if c["outcome"] != "reported":
                continue
            if "manifest" in c:
                reported.append(f"- {v['id']}: {c['manifest']}: {c['duplicated_values']} duplicated {c['by']} values ({c['rows']} rows)")
            else:
                reported.append(f"- {v['id']}: {c['a']} vs {c['b']}: {plural(c.get('n_shared', 0), 'shared ' + c['by'].replace('_', ', ') + ' value')} "
                                f"({c.get('rows_a', 0):,} rows of {c['a']}, {c.get('rows_b', 0):,} rows of {c['b']})")
    ex = summ["exclusions"]
    et, es_ = ex["official_train_not_sampled"], ex["official_test_not_sampled"]
    ex_lines = [f"- official train not sampled: {et['n']:,} images ({et['n_bona_fide']:,} bona fide, {et['n_attack']:,} attacks; "
                f"by session {pct_list(et['by_session'])})",
                f"- official test not sampled: {es_['n']:,} images ({es_['n_bona_fide']:,} bona fide, {es_['n_attack']:,} attacks; "
                f"by session {pct_list(es_['by_session'])})"]
    if private:
        ex_lines.append(f"- per subject (bona fide + attack), train: {pct_list(et['by_subject'])}")
        ex_lines.append(f"- per subject (bona fide + attack), test: {pct_list(es_['by_subject'])}")
    c1 = summ["c1_correction"]
    rc = summ["reconstruction"]
    nov = ((agg or {}).get("recount") or {}).get("test_from_subjects_absent_from_official_train") or {}
    def share(d):
        return f"{d['from_subjects_absent_from_official_train']:,} / {d['n']:,} ({100 * d['from_subjects_absent_from_official_train'] / d['n']:.1f} %)" if d.get("n") else "n/a"
    novelty_rows = "\n".join(f"| {n} | {share(d['bona_fide'])} | {share(d['attack'])} |" for n, d in nov.items()) or "| not computed | | |"
    pc = rc.get("validation_pair_uniqueness") or {}
    pair_line = (f"Independently of the scikit-learn version, among the {pc['n_pairs']} pairs of the {pc['n_training_subjects']} training "
                 f"subjects only {'one gives' if pc['n_pairs_with_card_n_val'] == 1 else str(pc['n_pairs_with_card_n_val']) + ' give'} "
                 f"{n_val:,} validation images, the count recorded in the cards; the reconstruction is also checked against the subject "
                 "arrays of the training cache." if pc and (n_val := m["cnn_es"]["n"]) else "")
    if self_test:
        self_test_line = (f"{'PASS' if self_test.get('passed') else 'FAIL'}, {self_test.get('n_detected_as_expected')} of {self_test.get('n_faults')} "
                          "faulty copies rejected by exactly the expected rules (" + "; ".join(
                              f"{x['fault']}: {', '.join(x['failing_rules']) or 'none'}" for x in self_test.get("faults", [])) + ").")
    else:
        self_test_line = "not run (`check-overlap --self-test` writes `overlap_self_test.json`)."
    return {
        "commit": summ.get("commit"), "created": summ.get("created"), "revision": summ["source"]["revision"],
        "archive_sha256": summ["source"]["archive_sha256"], "mirror": summ["source"]["mirror"],
        "lists": "\n".join(f"| `{k}` | {v['split']} | {v['n_entries']} | {'yes' if v['equal_to_prepared_folder'] else 'NO'} | `{v['sha256'][:16]}…` |"
                           for k, v in summ["official_lists"].items()),
        "manifest_rows": "\n".join(rows), "checkpoint_rows": "\n".join(ck), "verdict_rows": "\n".join(verdicts),
        "overlap_result": "PASS" if ov["passed"] else "FAIL", "n_rules": len(ov["rules"]), "n_must_failed": plural(ov["n_must_failed"], "must-hold rule") + " failed",
        "reported_overlaps": "\n".join(reported) or "- none",
        "exclusions": "\n".join(ex_lines),
        "train_cache_groups": yes[rc["training_cache"].get("matches_training_cache_groups")],
        "test_cache_groups": yes[rc["training_cache"].get("matches_test_cache_groups")],
        "cache_n_train": f"{rc['training_cache'].get('n_train', 0):,}", "cache_n_test": f"{rc['training_cache'].get('n_test', 0):,}",
        "dev_sessions": " and ".join(m["m1_dev"]["by_session"]),
        "c1_n": c1["n"], "c1_glasses": c1["v1_session_equals_glasses"], "c1_v1_values": ", ".join(f"{k}: {v}" for k, v in c1["v1"]["session_values"].items()),
        "c1_sessions": ", ".join(f"{k}: {v}" for k, v in c1["official_session_values"].items()),
        "c1_v1_sha": c1["v1"]["sha256"], "c1_v2_sha": m["c1"]["sha256"],
        "es_reuse": summ["declarations"]["early_stopping_reuse"],
        "validation_use": summ["declarations"]["validation_use"],
        "derived": derived_text(rc["training_cache"].get("derived_frames")), "novelty_rows": novelty_rows, "pair_line": pair_line,
        "self_test_line": self_test_line,
        "overlap_sentence": ("No image is shared, by key or by content hash, between any training-side manifest (official train, CNN train, "
                             "validation, M1 dev) and any test-side manifest (official test, CNN test, C1), and no (subject, session) pair is "
                             "shared. Subjects are shared: NUAA is not subject-disjoint between the official train and test splits."
                             if ov["passed"] else "**Must-hold rules failed: see the verdicts above and the overlap report.**"),
        "c1_dev_n": c1["n_from_dev_subjects"], "c1_dev_by_class": ", ".join(f"{v} {'attacks' if k == 'attack' else 'bona fide'}" for k, v in c1["n_from_dev_subjects_by_class"].items()) or "none",
        "c1_dev_sessions": ", ".join(c1["sessions_from_dev_subjects"]) or "-", "dev_subjects": dev_ids,
        "build_passed": "yes" if summ.get("passed") else "NO", "n_checks": len(summ.get("checks", {})),
        "visibility": ("Restricted version: it may name subjects." if private
                       else "Public version: no subject identifiers and no per-image data."),
    }


def cmd_report(a) -> None:
    out = Path(a.out)
    summ = load_json(out / "nuaa_manifests_summary.json")
    agg = load_json(out / "nuaa_manifests_aggregate.json")
    ov = load_json(out / "overlap_report.json")
    # aggregato pubblico e riepilogo privato devono certificare gli stessi manifest
    if {n: v["sha256"] for n, v in agg["manifests"].items()} != {n: v["sha256"] for n, v in summ["manifests"].items()}:
        fail("nuaa_manifests_aggregate.json and nuaa_manifests_summary.json list different manifests: rerun build")
    # il rapporto delle sovrapposizioni deve riguardare gli stessi manifest del riepilogo (stesso SHA-256)
    by_file = {v["file"].split("/")[-1]: v["sha256"] for v in summ["manifests"].values()}
    stale = [n for n, v in ov["manifests"].items() if by_file.get(v["file"]) != v["sha256"]]
    if stale:
        fail(f"overlap_report.json refers to manifests that differ from the summary: {stale}; rerun check-overlap")
    st_p = out / "overlap_self_test.json"
    self_test = load_json(st_p) if st_p.is_file() else None
    if self_test is not None and self_test.get("rules_sha256") != ov.get("rules_sha256"):
        fail("overlap_self_test.json was written with different rules than overlap_report.json: rerun check-overlap --self-test")
    tmpl = Path(a.template).read_text(encoding="utf-8")
    subject_ids = set()
    for v in summ["manifests"].values():
        subject_ids |= set(v.get("by_subject", {}))
    outputs = {}
    for private, fname in ((True, "nuaa_m2_report.md"), (False, "nuaa_m2_report_public.md")):
        txt = tmpl.format(**report_fields(summ, ov, private, self_test, agg))
        if not private:
            bad = public_problems(txt, subject_ids, markdown=True)
            if bad:
                fail("public report contains forbidden content: " + "; ".join(bad))
        (out / fname).write_text(txt, encoding="utf-8")
        outputs[fname] = sha256(out / fname)
    for f, h in outputs.items():
        print(f"== {relpath(out / f)} sha256 {h}", flush=True)


# ----------------------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build", help="manifest per immagine, riepilogo privato e aggregato pubblico (solo elenchi e hash)")
    p.add_argument("--out", default=str(M2_DIR), help="cartella di uscita (default results/m2; altra cartella per le prove)")
    p = sub.add_parser("check-overlap", help="matrice delle sovrapposizioni e verdetto delle regole (codice 1 se una regola must fallisce)")
    p.add_argument("--rules", default=str(RULES), help="file delle regole (default delivery/m2/nuaa_overlap_rules.json)")
    p.add_argument("--manifests-dir", default=str(M2_DIR / "manifests"), help="cartella dei manifest nominati nelle regole")
    p.add_argument("--out", default=str(M2_DIR), help="cartella del rapporto (overlap_report.json e .md)")
    p.add_argument("--self-test", action="store_true", help="ripete il controllo su copie guaste dei manifest reali (codice 1 se un guasto non è rilevato come atteso)")
    p = sub.add_parser("recount", help="ricalcola dai soli manifest i conteggi dell'aggregato e li confronta (codice 1 se differiscono)")
    p.add_argument("--rules", default=str(RULES), help="file delle regole (manifest dichiarati)")
    p.add_argument("--manifests-dir", default=str(M2_DIR / "manifests"), help="cartella dei manifest")
    p.add_argument("--aggregate", default=str(M2_DIR / "nuaa_manifests_aggregate.json"), help="aggregato pubblico da verificare")
    p = sub.add_parser("reconstruct", help="ricostruisce lo split delle CNN e il dev dal solo manifest del train ufficiale (serve scikit-learn)")
    p.add_argument("--rules", default=str(RULES), help="file delle regole (manifest dichiarati)")
    p.add_argument("--manifests-dir", default=str(M2_DIR / "manifests"), help="cartella dei manifest")
    p = sub.add_parser("compare-c1", help="confronta il manifest C1 versione 1 (pacchetto C1) con la versione 2")
    p.add_argument("--v1", default=str(C1_MANIFEST), help="manifest C1 versione 1 (default results/c1/nuaa_manifest.csv)")
    p.add_argument("--v2", default=str(M2_DIR / "manifests" / "nuaa_c1_v2.csv"), help="manifest C1 versione 2")
    p = sub.add_parser("report", help="rapporto breve privato e pubblico dal modello delivery/m2/report.md.tmpl")
    p.add_argument("--out", default=str(M2_DIR), help="cartella con riepilogo, aggregato e rapporto delle sovrapposizioni")
    p.add_argument("--template", default=str(REPORT_TMPL), help="modello del rapporto")
    a = ap.parse_args()
    {"build": cmd_build, "check-overlap": cmd_check_overlap, "recount": cmd_recount, "reconstruct": cmd_reconstruct,
     "compare-c1": cmd_compare_c1, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
