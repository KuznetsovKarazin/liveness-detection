"""
Protocollo della soglia source-dev su NUAA (M1): dev set, controlli di separazione, punteggi del dev, scelta della soglia
con il criterio dichiarato in metrics.md ("Threshold selection protocol") e applicazione ai 300 di test.

Sottocomandi:
  build-dev          dev set = tutte le immagini del train ufficiale (bonafide_training, attack_training) dei soggetti
                     tenuti fuori per la validazione quando sono state addestrate le CNN NUAA "official" (stesso elenco,
                     stesso campionamento `spread(..., 1500)`, stesso `GroupShuffleSplit` di scripts/train_cnn.py).
                     Si ferma (codice 1) se la ricostruzione non dà 2.507 immagini di training e 493 di validazione.
                     Scrive m1/nuaa_dev_manifest.csv e m1/nuaa_dev_summary.json nella cartella dei risultati
                     (LIVENESSLAB_RESULTS_DIR, default results/; solo lettura e hash dei file).
  check-separation   controlli automatici (codice 1 se uno fallisce): il manifest coincide con l'elenco del dev ricostruito
                     (stesse chiavi, etichetta coerente con la cartella real/attack, soggetto = prefisso del nome,
                     in_training_sample come nella ricostruzione, byte e SHA-256 uguali al file, chiavi nella forma
                     real/<nome> o attack/<nome>) e i conteggi coincidono con il riepilogo di build-dev; nessuna immagine
                     del dev nel manifest C1 o nel test ufficiale (per SHA-256; il test ufficiale deve contare 9.123
                     immagini), nessuna immagine del dev con sessione 03 nel nome, nessun soggetto del dev tra quelli usati
                     per i pesi, nessun duplicato del dev tra le immagini di quei soggetti. Riporta il limite noto di NUAA
                     (soggetti del dev presenti anche nel test) e commit e stato del repository.
  score-dev          punteggi degli analizzatori ammessi sulle immagini del dev, in una cache SEPARATA
                     (eval/nuaa_dev.json della cartella dei risultati, mai nuaa.json). È nuova inferenza su immagini: parte solo con
                     --i-am-authorized; --dry-run mostra cosa farebbe senza toccare nulla. Prima dell'inferenza verifica
                     che il file dei pesi di ogni analizzatore esista e abbia lo SHA-256 registrato come impronta nella
                     cache C1 (riconosciuta dal suo SHA-256; con --dry-run l'esito è solo riportato), poi rifà
                     check-separation e verifica per hash ogni immagine collegata. Dopo l'inferenza le impronte della
                     cache del dev devono coincidere con i pesi verificati.
  select             legge una cache del dev, sceglie la soglia con la stessa regola con cui si applica (attacco se
                     s > soglia; candidati = punteggi distinti del dev più un valore sotto il minimo; criterio eer:
                     minimo di |fp·n_a − fn·n_b|, a pari merito la soglia più alta; alternativa --criterion apcer10:
                     la soglia più alta con APCER <= 0,10 con la regola >), scrive threshold.json, lo rilegge dal disco e
                     ricalcola i conteggi con > (errore se differiscono). Rifiuta una cache che contenga immagini del test (nome del test ufficiale, sessione 03
                     nel nome, chiave del manifest C1), analizzatori non ammessi o voci non valide.
  apply              verifica threshold.json (criterio, valori in [0, 1], synthetic vero o falso, i quattro analizzatori
                     ammessi, valori uguali a un ricalcolo di select sulla cache del dev a cui rimanda), controlla lo SHA-256
                     della cache C1 (atteso: --expected-c1-cache-sha256, default il valore registrato alla consegna di C1;
                     il valore si normalizza togliendo gli spazi esterni e portando a minuscole, poi deve essere di 64
                     caratteri esadecimali, altrimenti errore prima di qualsiasi passo, anche in synthetic-trial),
                     applica la soglia ai 300 di test della cache C1 (sola lettura) e scrive la tabella alla soglia 0,5 e
                     alla soglia dev affiancate, con lo SHA-256 di threshold.json, la segnalazione "non-informative" (AUC del
                     dev < 0,5 oppure errore bilanciato del dev alla soglia > 0,5, con il motivo) e un blocco di sensibilità senza le immagini dei
                     soggetti del dev (nessun identificativo di soggetto nella tabella).
  synthetic-trial    prova di select + apply su una cache dev SINTETICA e deterministica (seed 42, Beta(2,5) per i bona
                     fide e Beta(5,2) per gli attacchi), in una cartella fuori dai risultati: non sono risultati. Se le
                     cartelle NUAA (data/NUAA/images) mancano, le immagini del test si escludono solo con la regola del nome
                     (quinto campo 03) e con il manifest C1, e l'uscita lo dice.

Uso tipico:
  python scripts/select_threshold.py build-dev
  python scripts/select_threshold.py check-separation
  python scripts/select_threshold.py score-dev --dry-run
  python scripts/select_threshold.py select [--criterion apcer10]
  python scripts/select_threshold.py apply
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from tesi_app.paths import DATA, RESULTS  # noqa: E402

SEED = 42
PER_CLASS = 1500                                   # --per-class dell'addestramento delle CNN NUAA official
EXPECTED_TRAIN, EXPECTED_VAL = 2507, 493           # n_train e n_val delle schede models/weights/*__nuaa.json
EXPECTED_TEST = 9123                               # immagini del test ufficiale NUAA (*_validation), come in nuaa_dev_separation.json
TEST_SESSION = "03"                                # quinto campo del nome NUAA nel test ufficiale
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
NUAA = DATA / "NUAA" / "images"
M1_DIR = RESULTS / "m1"
C1_MANIFEST = RESULTS / "c1" / "nuaa_manifest.csv"
C1_CACHE = RESULTS / "eval" / "nuaa.json"
# SHA-256 della cache C1 dei 300 di test, come registrato alla consegna di C1: SHA256SUMS della cartella riservata
# 02_Experiments/C1/20260928-C1-seed42-a80f6f9 (voce nuaa_score_cache.json) e run report C1. Costante nel codice
# versionato perché è la fonte più robusta: results/ ed export/ non sono versionati e starebbero accanto alla cache
# stessa, quindi un file modificato insieme alla cache non potrebbe smentirla. Un'altra cache (prove sintetiche) si
# dichiara esplicitamente con --expected-c1-cache-sha256.
C1_CACHE_SHA256 = "9b2b63b467e5e413c1fd5051e7c1c478f482301dc17070d60950d2e1daa34e65"
DEV_ID = "nuaa_dev"
DEV_CACHE = RESULTS / "eval" / f"{DEV_ID}.json"
DEV_ROOT = DATA / "m1_dev"                         # radice dati alternativa: fuori da data/eval, invisibile all'app web
PROCESSED_NPZ = DATA / "processed" / f"nuaa_{PER_CLASS}.npz"
# unici analizzatori con NUAA come dataset sorgente e validazione per soggetto: il dev non ha mai toccato i pesi
ELIGIBLE = ["livenessnet__nuaa", "attacknet_v1__nuaa", "attacknet_v2_1__nuaa", "attacknet_v2_2__nuaa"]
CRITERIA = ("eer", "apcer10")
# addestrati su NUAA ma su tutte le 3.000 immagini campionate del train ufficiale (n_train 3000 nelle schede): il dev
# comprende immagini usate per stimare il classificatore, quindi niente soglia source-dev
EXCLUDED_NUAA_TRAINED = {a: f"trained on all {2 * PER_CLASS} sampled images of the NUAA training split (models/weights/{a}_clf.json: n_train 3000); "
                            "the dev images were used to fit it" for a in ("lbp", "dog", "iqa", "ida", "clip_probe", "dinov2_probe")}
EXCLUDED_POOLED = {"nuaa-pooled CNNs (livenessnet__nuaa-pooled, attacknet_v1__nuaa-pooled, attacknet_v2_1__nuaa-pooled, attacknet_v2_2__nuaa-pooled)":
                   "trained on NUAA with the 80/20 per-image pooled split of the reference code, which includes the official test split: "
                   "excluded from the NUAA evaluation and from the source-dev threshold"}
CLASS_DIRS = (("bonafide", 0, "real"), ("attack", 1, "attack"))   # cartella NUAA, label, cartella nella chiave della cache
KEY_LABEL = {key_dir: lab for _, lab, key_dir in CLASS_DIRS}       # real -> 0, attack -> 1
DEV_COLUMNS = ["file", "label", "subject", "in_training_sample", "sha256", "bytes"]
KEY_RE = re.compile(r"^(real|attack)/[^/\\]+$")    # chiave della cache: una sola cartella, niente sottocartelle né ..
DEV_SUMMARY = M1_DIR / "nuaa_dev_summary.json"
DEV_SEPARATION = M1_DIR / "nuaa_dev_separation.json"
DEV_PROVENANCE = M1_DIR / "nuaa_dev_cache_provenance.json"


# ----------------------------------------------------------------------------- utilità

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args) -> str:
    """Uscita di un comando git; "n/d" fuori da un repository o se git non risponde (come eval_dataset.py)."""
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10)
    except Exception:  # noqa: BLE001
        return "n/d"
    return r.stdout.strip() if r.returncode == 0 else "n/d"


def repo_state() -> dict:
    """Commit e stato del repository; fuori da un repository commit "n/d" e dirty false (non "modificato")."""
    commit = git("rev-parse", "HEAD")
    if not commit or commit == "n/d":
        return {"commit": "n/d", "dirty": False}
    status = git("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) and status != "n/d"}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def rel(p: Path) -> str:
    """Percorso relativo al repository quando possibile (niente percorsi personali nei file scritti)."""
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return Path(p).name


def is_under(p: Path, base: Path) -> bool:
    try:
        Path(p).resolve().relative_to(Path(base).resolve())
        return True
    except ValueError:
        return False


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def nuaa_where() -> str:
    """Cartella delle immagini NUAA da citare nei messaggi, senza percorsi personali."""
    return rel(NUAA) if is_under(NUAA, ROOT) else "NUAA/images under LIVENESSLAB_DATA_DIR"


def note(msg: str) -> None:
    print(f"NOTE: {msg}", flush=True)


def spread(items, k):
    """Copia di scripts/train_cnn.py: k elementi a passo costante dall'elenco ORDINATO."""
    items = sorted(items)
    if k is None or k >= len(items):
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def subject_of(name: str) -> str:
    """Soggetto NUAA = prefisso a 4 cifre del nome (come subject_of di train_cnn.py)."""
    return name.split("_")[0]


def session_of(name: str) -> str:
    """Quinto campo del nome NUAA (SSSS_SSSS_xx_xx_03_n.jpg → 03): nel train vale 01 o 02, nel test 03."""
    parts = name.split("_")
    return parts[4] if len(parts) > 5 else ""


def list_split(tag: str, base: Path = None):
    """Immagini di uno split ufficiale NUAA: [(Path, label, cartella della chiave)], stesse estensioni di train_cnn.py."""
    base = base or NUAA
    out = []
    for cls, lab, key_dir in CLASS_DIRS:
        d = base / f"{cls}_{tag}"
        if not d.is_dir():
            fail(f"missing folder {rel(d)}")
        out += [(p, lab, key_dir) for p in d.iterdir() if p.suffix.lower() in IMG_EXT]
    return out


def split_present(tag: str, base: Path = None) -> bool:
    """True se entrambe le cartelle dello split esistono, False se mancano entrambe; errore se ne manca una sola."""
    base = base or NUAA
    there = [(base / f"{cls}_{tag}").is_dir() for cls, _, _ in CLASS_DIRS]
    if any(there) and not all(there):
        fail(f"only part of the NUAA {tag} split is present in {rel(base)}: both bonafide_{tag} and attack_{tag} are needed")
    return all(there)


def reconstruct_split(base: Path = None):
    """Ricostruisce lo split di validazione dell'addestramento delle CNN NUAA official, nello stesso ordine di
    build_arrays/train: bona fide poi attacchi, ciascuno `spread(iterdir, 1500)`; GroupShuffleSplit sui soggetti."""
    from sklearn.model_selection import GroupShuffleSplit
    base = base or NUAA
    files, y = [], []
    for cls, lab, _ in CLASS_DIRS:
        d = base / f"{cls}_training"
        if not d.is_dir():
            fail(f"missing folder {rel(d)}")
        fs = spread([p for p in d.iterdir() if p.suffix.lower() in IMG_EXT], PER_CLASS)
        files += fs; y += [lab] * len(fs)
    y = np.array(y, np.int64)
    groups = np.array([subject_of(p.name) for p in files])
    tr_i, va_i = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(np.zeros(len(y)), y, groups))
    return {"files": files, "y": y, "groups": groups, "train_idx": tr_i, "val_idx": va_i,
            "val_subjects": sorted(set(groups[va_i].tolist())), "fit_subjects": sorted(set(groups[tr_i].tolist()))}


def check_reconstruction(rec) -> None:
    """La ricostruzione deve dare i conteggi delle schede dei pesi, altrimenti il dev non è quello dell'addestramento."""
    per_class = Counter(rec["y"].tolist())
    if (per_class[0] != PER_CLASS or per_class[1] != PER_CLASS or len(rec["train_idx"]) != EXPECTED_TRAIN
            or len(rec["val_idx"]) != EXPECTED_VAL):
        fail(f"reconstruction does not match the weight cards (expected {EXPECTED_TRAIN}/{EXPECTED_VAL} with {PER_CLASS} per class, "
             f"got {len(rec['train_idx'])}/{len(rec['val_idx'])})")


def cache_key(p: Path) -> str:
    """Chiave con la cartella: bona fide e attacchi hanno nomi di file in comune."""
    return ("real/" if p.parent.name.startswith("bonafide") else "attack/") + p.name


class _DuplicateKey(Exception):
    pass


def _no_duplicates(pairs):
    seen = set()
    for k, _ in pairs:
        if k in seen:
            raise _DuplicateKey(k)
        seen.add(k)
    return dict(pairs)


def _no_constant(c):
    raise ValueError(f"non-finite number {c}")


def load_json(p, what: str = None) -> dict:
    """JSON letto in modo stretto: file mancante, JSON non valido, chiavi duplicate, NaN/Infinity o radice che non è un
    oggetto fermano lo script con un errore pulito (mai un traceback)."""
    p = Path(p)
    what = what or relpath(p)
    if not p.is_file():
        fail(f"missing {what}")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates, parse_constant=_no_constant)
    except _DuplicateKey as e:
        fail(f"{what}: duplicated JSON key {e.args[0]!r}: refused")
    except (ValueError, UnicodeDecodeError) as e:
        fail(f"{what}: invalid JSON ({e})")
    if not isinstance(doc, dict):
        fail(f"{what}: the JSON root is not an object")
    return doc


def read_manifest(p: Path, columns=None):
    """Righe di un manifest CSV; file mancante, CSV illeggibile o colonne mancanti → errore pulito."""
    p = Path(p)
    if not p.is_file():
        fail(f"missing {relpath(p)}")
    try:
        with p.open(newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            rows = list(rd)
            cols = rd.fieldnames or []
    except (csv.Error, UnicodeDecodeError) as e:
        fail(f"{relpath(p)}: unreadable CSV ({e})")
    missing = [c for c in (columns or []) if c not in cols]
    if missing:
        fail(f"{relpath(p)}: missing columns {missing}")
    bad = [i + 2 for i, r in enumerate(rows) if None in r or any(v is None for v in r.values())]
    if bad:
        fail(f"{relpath(p)}: rows with a wrong number of fields (lines {bad[:3]})")
    return rows


def key_problem(key) -> str:
    """Motivo per cui una chiave non è valida (stringa vuota se è valida): real/<nome> o attack/<nome>, niente ..,
    niente sottocartelle, estensione d'immagine."""
    if not isinstance(key, str) or not KEY_RE.match(key):
        return "not of the form real/<name> or attack/<name>"
    name = key.split("/", 1)[1]
    if name in (".", "..") or name.startswith("."):
        return "hidden or relative name"
    if Path(name).suffix.lower() not in IMG_EXT:
        return "not an image file name"
    return ""


def source_path(row, base: Path = None) -> Path:
    """File originale di una riga del manifest del dev (la chiave è real/… o attack/…, già validata)."""
    base = base or NUAA
    key_dir, name = row["file"].split("/", 1)
    cls = "bonafide" if key_dir == "real" else "attack"
    return base / f"{cls}_training" / name


def relpath(p) -> str:
    """Percorso relativo alla cartella del repository (anche con ..), per comandi e piani senza percorsi personali."""
    q = Path(p).resolve()
    return os.path.relpath(q, ROOT) if is_under(q, ROOT) else str(q)


def examples(items, k=3):
    return sorted(items)[:k]


def manifest_row_problems(man) -> list:
    """Controlli strutturali del manifest del dev, senza leggere il disco: chiavi valide e non duplicate, etichetta
    coerente con la cartella real/attack, soggetto = prefisso del nome, campi numerici ben formati."""
    problems = []
    keys = [r.get("file") for r in man]
    dup = [k for k, n in Counter(keys).items() if n > 1]
    if dup:
        problems.append(f"{len(dup)} duplicated keys in the dev manifest (e.g. {examples(map(str, dup))})")
    bad_key = [str(k) for k in keys if key_problem(k)]
    if bad_key:
        problems.append(f"{len(bad_key)} invalid keys in the dev manifest (e.g. {examples(bad_key)}): expected real/<name> or attack/<name>")
    bad_label, bad_subj, bad_its, bad_sha, bad_bytes = [], [], [], [], []
    for r in man:
        k = r.get("file")
        if key_problem(k):
            continue
        key_dir, name = k.split("/", 1)
        if r.get("label") not in ("0", "1") or int(r["label"]) != KEY_LABEL[key_dir]:
            bad_label.append(k)
        if r.get("subject") != subject_of(name) or not re.fullmatch(r"\d{4}", r.get("subject") or ""):
            bad_subj.append(k)
        if r.get("in_training_sample") not in ("0", "1"):
            bad_its.append(k)
        if not re.fullmatch(r"[0-9a-f]{64}", r.get("sha256") or ""):
            bad_sha.append(k)
        if not re.fullmatch(r"\d+", r.get("bytes") or ""):
            bad_bytes.append(k)
    for lst, what in ((bad_label, "labels not consistent with the real/attack folder (real = 0, attack = 1)"),
                      (bad_subj, "subjects different from the 4-digit prefix of the name"),
                      (bad_its, "in_training_sample values other than 0/1"),
                      (bad_sha, "malformed sha256 values"), (bad_bytes, "malformed bytes values")):
        if lst:
            problems.append(f"{len(lst)} {what} (e.g. {examples(lst)})")
    return problems


def manifest_counts(man) -> dict:
    return {"n": len(man), "n_bona_fide": sum(1 for r in man if r.get("label") == "0"),
            "n_attack": sum(1 for r in man if r.get("label") == "1"),
            "n_in_training_sample": sum(1 for r in man if r.get("in_training_sample") == "1")}


def summary_problems(man, summ: dict, man_sha: str) -> list:
    """Il riepilogo di build-dev deve certificare questo manifest: hash, conteggi e soggetti uguali."""
    problems = []
    if summ.get("manifest_sha256") != man_sha:
        problems.append(f"dev manifest sha256 {man_sha} differs from the summary ({summ.get('manifest_sha256')})")
    for k, v in manifest_counts(man).items():
        if summ.get(k) != v or isinstance(summ.get(k), bool):
            problems.append(f"{k}: manifest {v}, summary {summ.get(k)!r}")
    subjects = sorted({r.get("subject") for r in man})
    if summ.get("validation_subjects") != subjects:
        problems.append(f"dev manifest subjects {subjects} differ from the summary validation_subjects {summ.get('validation_subjects')!r}")
    return problems


def manifest_disk_problems(man, rec, base: Path = None) -> list:
    """Il manifest deve coincidere con l'elenco del dev ricostruito dal disco: stesse chiavi (tutte le immagini del train
    ufficiale dei soggetti di validazione), etichetta dalla cartella, in_training_sample come nella ricostruzione,
    byte e SHA-256 uguali al file. Le righe con chiave non valida sono già segnalate da manifest_row_problems."""
    base = base or NUAA
    problems = []
    val_subjects = set(rec["val_subjects"])
    expected = {f"{key_dir}/{p.name}": (lab, p) for p, lab, key_dir in list_split("training", base) if subject_of(p.name) in val_subjects}
    sampled = {cache_key(p) for p in rec["files"]}
    have = {r["file"] for r in man if not key_problem(r.get("file"))}
    missing, extra = sorted(set(expected) - have), sorted(have - set(expected))
    if missing:
        problems.append(f"{len(missing)} images of the validation subjects missing from the dev manifest (e.g. {missing[:3]})")
    if extra:
        problems.append(f"{len(extra)} dev manifest keys that are not training-split images of the validation subjects (e.g. {extra[:3]})")
    wrong_lab, wrong_its, wrong_bytes, wrong_sha = [], [], [], []
    for r in man:
        k = r.get("file")
        if key_problem(k) or k not in expected:
            continue
        lab, p = expected[k]
        if r.get("label") != str(lab):
            wrong_lab.append(k)
        if r.get("in_training_sample") != str(int(k in sampled)):
            wrong_its.append(k)
        if r.get("bytes") != str(p.stat().st_size):
            wrong_bytes.append(k)
        if r.get("sha256") != sha256(p):
            wrong_sha.append(k)
    for lst, what in ((wrong_lab, "labels different from the folder on disk"),
                      (wrong_its, "in_training_sample values different from the reconstruction of the training sample"),
                      (wrong_bytes, "bytes values different from the file size"),
                      (wrong_sha, "sha256 values different from the file on disk")):
        if lst:
            problems.append(f"{len(lst)} {what} (e.g. {examples(lst)})")
    return problems


def test_split_names(allow_missing: bool = False):
    """Nomi dei file del test ufficiale NUAA (*_validation): solo elenco, nessuna lettura delle immagini. Il numero deve
    essere 9.123. Se le cartelle mancano entrambe e allow_missing è vero restituisce None (vale solo la regola del nome)."""
    if not split_present("validation"):
        if allow_missing:
            return None
        fail(f"missing folders {nuaa_where()}/{{bonafide,attack}}_validation: the official test split is needed to refuse test images")
    names = [p.name for p, _, _ in list_split("validation")]
    if len(names) != EXPECTED_TEST:
        fail(f"the official NUAA test split lists {len(names)} images, expected {EXPECTED_TEST}: data folder incomplete or different")
    return set(names)


def check_not_test(keys, what: str, c1_manifest: Path = C1_MANIFEST, allow_missing: bool = False) -> None:
    """Nessuna chiave con il nome di un file del test ufficiale NUAA, con sessione 03 nel nome (regola del nome, vale
    anche senza le cartelle NUAA) o presente nel manifest C1."""
    names = test_split_names(allow_missing)
    if names is None:
        note(f"NUAA official test folders not found in {nuaa_where()}: test images in the {what} refused by the name rule "
             f"(5th field {TEST_SESSION}) and by the C1 manifest only")
        names = set()
    c1_keys = {r["file"] for r in read_manifest(c1_manifest, ["file"])}
    hit = sorted(k for k in keys if k.split("/")[-1] in names or session_of(k.split("/")[-1]) == TEST_SESSION or k in c1_keys)
    if hit:
        fail(f"{what} contains {len(hit)} images of the official test split or of the C1 manifest (e.g. {hit[:3]}): refused")


def check_entries(aid: str, per, what: str = "cache") -> None:
    """Voci di un analizzatore: oggetto {chiave: {"y": 0/1 intero, "s": numero finito in [0, 1]}}; altrimenti errore pulito."""
    if not isinstance(per, dict):
        fail(f"{what}: scores of {aid} are not an object")
    bad_key = [str(k) for k in per if key_problem(k)]
    if bad_key:
        fail(f"{what}: {aid} has {len(bad_key)} invalid keys (e.g. {examples(bad_key)})")
    not_obj = [k for k, v in per.items() if not isinstance(v, dict)]
    if not_obj:
        fail(f"{what}: {aid} has {len(not_obj)} entries that are not objects (e.g. {examples(not_obj)})")
    bad_y = [k for k, v in per.items() if type(v.get("y")) is not int or v["y"] not in (0, 1)]
    if bad_y:
        fail(f"{what}: {aid} has {len(bad_y)} labels that are not the integers 0 or 1 (e.g. {examples(bad_y)})")
    bad = [k for k, v in per.items()
           if not isinstance(v.get("s"), (int, float)) or isinstance(v.get("s"), bool) or not math.isfinite(v["s"]) or not 0.0 <= v["s"] <= 1.0]
    if bad:
        n_err = sum(1 for k in bad if per[k].get("e"))
        fail(f"{what}: {aid}: {len(bad)} images without a valid score in [0, 1] (e.g. {examples(bad)}, analyzer errors among them: {n_err})")


def check_dev_cache(cache: dict, what: str) -> None:
    """Struttura della cache del dev: solo analizzatori ammessi, voci valide, impronte come oggetto."""
    scores = cache.get("scores")
    if not isinstance(scores, dict) or not scores:
        fail(f"{what}: no 'scores' object")
    bad = sorted(a for a in scores if a not in ELIGIBLE)
    if bad:
        fail(f"{what}: analyzers not eligible for a NUAA source-dev threshold: {bad} (eligible: {ELIGIBLE})")
    for aid, per in scores.items():
        check_entries(aid, per, what)
    if not isinstance(cache.get("fingerprints", {}), dict):
        fail(f"{what}: 'fingerprints' is not an object")
    if "synthetic" in cache and not isinstance(cache["synthetic"], bool):
        fail(f"{what}: 'synthetic' must be true or false")


def verify_dev_manifest(man_p: Path, summary_p: Path, separation_p: Path, allow_missing: bool = False):
    """Il manifest deve essere quello prodotto da build-dev e controllato da check-separation (hash uguali, passed = true
    e nessun problema), avere gli stessi conteggi del riepilogo, coincidere con l'elenco ricostruito dal disco (se le
    cartelle NUAA ci sono) e nessuna sua chiave può essere un'immagine del test. Restituisce (SHA-256, righe)."""
    for p in (man_p, summary_p, separation_p):
        if not Path(p).exists():
            fail(f"missing {relpath(p)}: run build-dev and check-separation --report {relpath(separation_p)} first")
    man = read_manifest(man_p, DEV_COLUMNS)
    problems = manifest_row_problems(man)
    if problems:
        fail(f"{relpath(man_p)}: " + "; ".join(problems))
    h = sha256(man_p)
    summ = load_json(summary_p)
    sep = load_json(separation_p)
    problems = summary_problems(man, summ, h)
    if problems:
        fail(f"{relpath(summary_p)} does not certify this manifest: " + "; ".join(problems))
    if sep.get("dev_manifest_sha256") != h or sep.get("passed") is not True or sep.get("problems") != []:
        fail(f"{relpath(separation_p)} does not certify this manifest (hash differs, passed is not true or problems is not empty)")
    if split_present("training"):
        rec = reconstruct_split()
        check_reconstruction(rec)
        problems = manifest_disk_problems(man, rec)
        if rec["val_subjects"] != summ.get("validation_subjects"):
            problems.append(f"validation subjects of the reconstruction {rec['val_subjects']} differ from the summary")
        if problems:
            fail(f"{relpath(man_p)} differs from the dev set rebuilt from {nuaa_where()}: " + "; ".join(problems))
    elif allow_missing:
        note(f"NUAA training folders not found in {nuaa_where()}: dev manifest checked against the summary and the separation report only")
    else:
        fail(f"missing folders {nuaa_where()}/{{bonafide,attack}}_training: the dev manifest cannot be checked against the images")
    check_not_test([r["file"] for r in man], "dev manifest", allow_missing=allow_missing)
    return h, man


# ----------------------------------------------------------------------------- build-dev

def cmd_build_dev(a) -> None:
    out_dir = Path(a.out_dir)
    rec = reconstruct_split()
    n_tr, n_va = len(rec["train_idx"]), len(rec["val_idx"])
    per_class = Counter(rec["y"].tolist())
    print(f"== reconstruction: {len(rec['y'])} sampled images (bona fide {per_class[0]}, attack {per_class[1]}), "
          f"train {n_tr}, validation {n_va}", flush=True)
    print(f"   validation subjects: {rec['val_subjects']}; fitted subjects: {rec['fit_subjects']}", flush=True)
    if per_class[0] != PER_CLASS or per_class[1] != PER_CLASS or n_tr != EXPECTED_TRAIN or n_va != EXPECTED_VAL:
        fail(f"reconstruction does not match the weight cards (expected {EXPECTED_TRAIN}/{EXPECTED_VAL} with {PER_CLASS} per class): dev set NOT built")
    # prova indipendente: la cache .npz dell'addestramento conserva i soggetti e le label nell'ordine usato dallo split
    npz_check = None
    if PROCESSED_NPZ.exists():
        z = np.load(PROCESSED_NPZ)
        if "s_train" in z.files:
            npz_check = bool(np.array_equal(z["s_train"], rec["groups"]) and np.array_equal(z["y_train"], rec["y"]))
            print(f"   groups and labels equal to {rel(PROCESSED_NPZ)} (s_train, y_train): {npz_check}", flush=True)
            if not npz_check:
                fail("subject/label order differs from the training cache: dev set NOT built")
    sampled = {cache_key(p) for p in rec["files"]}
    val_keys = {cache_key(rec["files"][i]) for i in rec["val_idx"]}
    dev_subjects = set(rec["val_subjects"])
    rows = []
    for p, lab, key_dir in list_split("training"):
        if subject_of(p.name) not in dev_subjects:
            continue
        rows.append({"file": f"{key_dir}/{p.name}", "label": lab, "subject": subject_of(p.name),
                     "in_training_sample": int(f"{key_dir}/{p.name}" in sampled), "sha256": sha256(p), "bytes": p.stat().st_size})
    rows.sort(key=lambda r: r["file"])
    n_in_sample = sum(r["in_training_sample"] for r in rows)
    if n_in_sample != EXPECTED_VAL or {r["file"] for r in rows if r["in_training_sample"]} != val_keys:
        fail(f"dev images in the training sample ({n_in_sample}) differ from the validation images ({EXPECTED_VAL})")
    out_dir.mkdir(parents=True, exist_ok=True)
    man_p = out_dir / "nuaa_dev_manifest.csv"
    with man_p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DEV_COLUMNS, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    by = defaultdict(lambda: {"bona_fide": 0, "attack": 0})
    for r in rows:
        by[r["subject"]]["attack" if r["label"] else "bona_fide"] += 1
    summary = {
        "dataset": "NUAA Photograph Imposter Database, official training split (bonafide_training, attack_training)",
        "protocol": "metrics.md, Threshold selection protocol, point 1",
        "reconstruction": {"script": "scripts/train_cnn.py (build_arrays, train)", "per_class": PER_CLASS, "sampling": "spread over the sorted file list",
                           "split": f"GroupShuffleSplit(n_splits=1, test_size=0.2, random_state={SEED}) on the 4-digit subject prefix",
                           "n_sampled": int(len(rec["y"])), "n_train": n_tr, "n_val": n_va,
                           "matches_weight_cards": True, "matches_training_cache_groups": npz_check},
        "validation_subjects": rec["val_subjects"], "fitted_subjects": rec["fit_subjects"],
        "n": len(rows), "n_bona_fide": sum(1 for r in rows if r["label"] == 0), "n_attack": sum(1 for r in rows if r["label"] == 1),
        "n_in_training_sample": n_in_sample, "n_not_in_training_sample": len(rows) - n_in_sample,
        "by_subject": {k: by[k] for k in sorted(by)},
        "manifest": man_p.name, "manifest_sha256": sha256(man_p),
        "note": "in_training_sample = 1: among the 3000 images sampled for training, i.e. the validation images used only for early "
                "stopping; 0: never read by the training. Unit = image.",
        "commit": repo_state()["commit"], "created": now(),
    }
    (out_dir / "nuaa_dev_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"== dev set: {summary['n']} images (bona fide {summary['n_bona_fide']}, attack {summary['n_attack']}), "
          f"{n_in_sample} of them were validation images of the training", flush=True)
    print(f"   {rel(man_p)} sha256 {summary['manifest_sha256']}", flush=True)
    print(f"   {rel(out_dir / 'nuaa_dev_summary.json')}", flush=True)


# ----------------------------------------------------------------------------- check-separation

def cmd_check_separation(a) -> None:
    man_p = Path(a.manifest)
    if not man_p.exists():
        fail(f"missing {relpath(man_p)}: run build-dev first")
    summary_p = Path(a.summary) if a.summary else man_p.parent / "nuaa_dev_summary.json"
    man = read_manifest(man_p, DEV_COLUMNS)
    man_sha = sha256(man_p)
    problems, report = [], {}
    # 0. struttura del manifest, riepilogo di build-dev, corrispondenza con l'elenco ricostruito e con i file su disco
    row_problems = manifest_row_problems(man)
    report["manifest_rows_valid"] = not row_problems
    problems += row_problems
    summ = load_json(summary_p)
    summ_problems = summary_problems(man, summ, man_sha)
    report["summary"] = {"file": summary_p.name, "matches_manifest": not summ_problems, "manifest_counts": manifest_counts(man)}
    problems += summ_problems
    rec = reconstruct_split()
    check_reconstruction(rec)
    disk_problems = manifest_disk_problems(man, rec)
    report["manifest_matches_disk"] = not disk_problems
    problems += disk_problems
    valid = [r for r in man if not key_problem(r.get("file"))]
    dev_hash = {r["sha256"]: r["file"] for r in valid}
    dev_subjects = sorted({r["subject"] for r in valid})
    # nessun duplicato dentro il dev (bona fide e attacchi hanno nomi in comune: conta il contenuto)
    report["dev_internal_duplicates"] = len(valid) - len(dev_hash)
    if len(valid) != len(dev_hash):
        problems.append(f"{len(valid) - len(dev_hash)} duplicated images inside the dev set")
    # 1. nessuna immagine del dev nel manifest C1 (hash e chiave)
    c1 = read_manifest(a.c1_manifest, ["file", "subject", "sha256"])
    hit_c1 = sorted(dev_hash[r["sha256"]] for r in c1 if r["sha256"] in dev_hash)
    key_c1 = sorted({r["file"] for r in c1} & {r["file"] for r in man})
    report["c1_manifest"] = {"n_c1": len(c1), "hash_overlap": len(hit_c1), "key_overlap": len(key_c1)}
    if hit_c1 or key_c1:
        problems.append(f"dev images in the C1 manifest: {len(hit_c1)} by hash, {len(key_c1)} by key")
    # 2. nessuna immagine del dev uguale (per hash) a un'immagine del test ufficiale, né con la sessione del test nel nome
    test = list_split("validation")
    hit_test = [p.name for p, _, _ in test if sha256(p) in dev_hash]
    test_names = {p.name for p, _, _ in test}
    name_hit = sorted(r["file"] for r in valid if r["file"].split("/", 1)[1] in test_names
                      or session_of(r["file"].split("/", 1)[1]) == TEST_SESSION)
    report["official_test_split"] = {"n_hashed": len(test), "expected": EXPECTED_TEST, "hash_overlap": len(hit_test),
                                     "name_or_session_overlap": len(name_hit)}
    if len(test) != EXPECTED_TEST:
        problems.append(f"the official test split lists {len(test)} images, expected {EXPECTED_TEST}")
    if hit_test:
        problems.append(f"{len(hit_test)} official test images identical to dev images (e.g. {hit_test[:3]})")
    if name_hit:
        problems.append(f"{len(name_hit)} dev images with a test-split name or session {TEST_SESSION} (e.g. {name_hit[:3]})")
    # 3. nessun soggetto del dev tra quelli usati per stimare i pesi, e nessun duplicato del dev tra le loro immagini
    bad_subj = sorted(set(dev_subjects) & set(rec["fit_subjects"]))
    report["subjects"] = {"dev": dev_subjects, "validation_subjects_of_training": rec["val_subjects"], "fitted_subjects": rec["fit_subjects"],
                          "dev_equals_validation_subjects": dev_subjects == rec["val_subjects"], "overlap_with_fitted": bad_subj}
    if bad_subj or dev_subjects != rec["val_subjects"]:
        problems.append(f"dev subjects {dev_subjects} vs validation subjects {rec['val_subjects']}, overlap with fitted subjects {bad_subj}")
    fitted = [p for p, _, _ in list_split("training") if subject_of(p.name) in set(rec["fit_subjects"])]
    dup = [p.name for p in fitted if sha256(p) in dev_hash]
    report["fitted_subjects_images"] = {"n_hashed": len(fitted), "hash_overlap": len(dup)}
    if dup:
        problems.append(f"{len(dup)} images of fitted subjects identical to dev images (e.g. {dup[:3]})")
    # limite noto di NUAA (informativo, non è un errore): i soggetti del dev compaiono anche nel test ufficiale
    test_subj = defaultdict(set); dev_sess = defaultdict(set)
    for p, _, _ in test:
        test_subj[subject_of(p.name)].add(session_of(p.name))
    for r in valid:
        dev_sess[r["subject"]].add(session_of(r["file"].split("/", 1)[1]))
    c1_subjects = sorted({r["subject"] for r in c1})
    report["known_limit"] = {
        "dev_subjects_in_official_test": {s: sorted(test_subj[s]) for s in dev_subjects if s in test_subj},
        "dev_subjects_in_c1_sample": sorted(set(dev_subjects) & set(c1_subjects)),
        "dev_sessions_by_subject": {s: sorted(v) for s, v in sorted(dev_sess.items())},
        "session_field": "5th field of the NUAA file name (training split 01/02, test split 03); the `session` column of the C1 manifest is the 3rd field",
        "text": "NUAA is not subject-disjoint between the official train and test splits: the dev subjects also appear in the test split, "
                "with a different value of the 5th name field (capture session). Separation from the test set is by official split and session, "
                "and by image hash.",
    }
    same_sess = {s: sorted(dev_sess[s] & test_subj.get(s, set())) for s in dev_subjects if dev_sess[s] & test_subj.get(s, set())}
    report["known_limit"]["dev_test_same_session"] = same_sess
    if same_sess:
        problems.append(f"dev subjects share the capture session with the official test split: {same_sess}")
    report["passed"] = not problems
    report["problems"] = problems
    report["dev_manifest_sha256"] = man_sha
    report.update(repo_state())
    report["checked"] = now()
    report_p = Path(a.report) if a.report else man_p.parent / "nuaa_dev_separation.json"
    report_p.parent.mkdir(parents=True, exist_ok=True)
    report_p.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1), flush=True)
    if problems:
        fail("separation checks failed: " + "; ".join(problems))
    print("== separation checks passed", flush=True)


# ----------------------------------------------------------------------------- score-dev

def clear_link_dirs(link_dir: Path) -> int:
    """Svuota <link_dir>/real e <link_dir>/attack (create se mancano) togliendo SOLO collegamenti simbolici. Se trova un
    file regolare o una cartella si ferma con errore prima di togliere qualsiasi cosa: una radice dati sbagliata
    (--dev-root) non può mai portare a cancellare immagini vere. Restituisce il numero di collegamenti tolti."""
    dirs = [link_dir / key_dir for key_dir in ("real", "attack")]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        not_links = sorted(p.name for p in d.iterdir() if not p.is_symlink())
        if not_links:
            fail(f"{relpath(d)} contains {len(not_links)} entries that are not symbolic links (e.g. {not_links[:3]}): "
                 "nothing removed; check --dev-root and clean the folder by hand")
    n = 0
    for d in dirs:
        for p in d.iterdir():
            p.unlink(); n += 1
    return n


def registered_weights_file(aid: str):
    """File dei pesi che l'analizzatore `aid` userebbe (stessa funzione dell'applicazione, nessun modello caricato),
    oppure una stringa con il motivo se l'analizzatore non è registrato o il file manca (nessun ripiego sulla scheda JSON)."""
    import tesi_app.analyzers  # noqa: F401  registra gli analizzatori
    from tesi_app.analyzers.kuznetsov import weights_file
    from tesi_app.core import registry
    if aid not in registry.ids():
        return "analyzer not registered"
    key = getattr(registry.get(aid), "weights_key", None)
    f = weights_file(key) if key else None
    return f if f is not None else f"weights file for {key} not found"


def weight_problems(ids, c1_cache: Path = None, expected_c1_sha: str = C1_CACHE_SHA256, resolve=registered_weights_file):
    """Controllo dei pesi prima dell'inferenza: per ogni analizzatore il file dei pesi deve esistere e il suo SHA-256
    deve coincidere con l'impronta registrata nella cache C1, a sua volta riconosciuta dal suo SHA-256.
    Restituisce (problemi, pesi) con pesi = {id: {"weights_file", "sha256"}}; solo lettura."""
    c1_cache = Path(c1_cache) if c1_cache is not None else C1_CACHE
    problems, weights = [], {}
    if not c1_cache.exists():
        return [f"{relpath(c1_cache)} missing: the weights cannot be compared with C1"], weights
    if sha256(c1_cache) != expected_c1_sha:
        return [f"{relpath(c1_cache)} differs from the C1 delivery (SHA-256 {expected_c1_sha[:12]}… expected)"], weights
    fps = load_json(c1_cache).get("fingerprints", {})
    fps = fps if isinstance(fps, dict) else {}
    for aid in ids:
        f = resolve(aid)
        if not isinstance(f, Path):
            problems.append(f"{aid}: {f}"); continue
        h = sha256(f)
        weights[aid] = {"weights_file": f.name, "sha256": h}
        if not isinstance(fps.get(aid), str) or h != fps[aid]:
            problems.append(f"{aid}: SHA-256 of {f.name} ({h[:12]}…) differs from the C1 fingerprint ({str(fps.get(aid))[:12]}…)")
    return problems, weights


def cmd_score_dev(a) -> None:
    ids = a.analyzers or ELIGIBLE
    bad = [i for i in ids if i not in ELIGIBLE]
    if bad:
        fail(f"not eligible for a NUAA source-dev threshold: {bad} (eligible: {ELIGIBLE})")
    man_p, summ_p = Path(a.manifest), Path(a.summary)
    for p in (man_p, summ_p):
        if not p.exists():
            fail(f"missing {relpath(p)}: run build-dev first")
    man = read_manifest(man_p, DEV_COLUMNS)
    # chiavi validate prima di qualsiasi collegamento: niente .., niente sottocartelle, etichette coerenti
    problems = manifest_row_problems(man)
    if problems:
        fail(f"{relpath(man_p)}: " + "; ".join(problems))
    man_sha = sha256(man_p)
    problems = summary_problems(man, load_json(summ_p), man_sha)
    if problems:
        fail(f"{relpath(summ_p)} does not certify this manifest: " + "; ".join(problems))
    n_bf = sum(1 for r in man if r["label"] == "0")
    dev_root = Path(a.dev_root)
    link_dir = dev_root / "eval" / DEV_ID
    # argomenti relativi e cwd = cartella del repository: la riga di comando finisce nell'intestazione della tabella
    cmd = ["scripts/eval_dataset.py", "--dataset", DEV_ID, "--analyzers", *ids, "--force", "--lang", "en",
           "--export", relpath(M1_DIR / "nuaa_dev_scores.csv"), "--table", relpath(M1_DIR / "nuaa_dev_table.md")]
    py = relpath(sys.executable) if is_under(sys.executable, ROOT) else "python"
    plan = {"analyzers": ids, "n_images": len(man), "n_bona_fide": n_bf, "n_attack": len(man) - n_bf,
            "dev_manifest": relpath(man_p), "dev_manifest_sha256": man_sha,
            "image_links": relpath(link_dir) + "/{real,attack} (symbolic links to data/NUAA/images/*_training, recreated at each run; "
                           "outside data/eval: not listed by the web app)",
            "cache": relpath(DEV_CACHE) + (" (EXISTS: the run will refuse to start)" if DEV_CACHE.exists() else " (new)"),
            "provenance": relpath(DEV_PROVENANCE),
            "command": f"cd <repository> && LIVENESSLAB_DATA_DIR={relpath(dev_root)} {py} " + " ".join(cmd),
            "never_written": [relpath(C1_CACHE), relpath(RESULTS / "c1")]}
    print("== score-dev plan:\n" + json.dumps(plan, indent=1), flush=True)
    # pesi verificati prima di qualunque inferenza: stessi file, byte per byte, delle impronte di C1. Con --dry-run l'esito
    # è solo riportato (il pacchetto esportato non contiene i pesi); senza --dry-run un problema ferma tutto.
    w_problems, weights = weight_problems(ids)
    if w_problems and a.dry_run:
        note("weight checks would fail here (the run stops on them before inference): " + "; ".join(w_problems))
    elif w_problems:
        fail("weight checks failed, dev scoring not started: " + "; ".join(w_problems))
    else:
        print(f"== weight checks passed: {len(weights)} weights files identical to the C1 fingerprints", flush=True)
    if a.dry_run:
        print("== dry run: nothing executed", flush=True)
        return
    if not a.i_am_authorized:
        print("ERROR: scoring the dev images is new inference on images and needs explicit authorisation: "
              "rerun with --i-am-authorized (or --dry-run to see the plan)", file=sys.stderr, flush=True)
        sys.exit(2)
    if DEV_CACHE.resolve() == C1_CACHE.resolve():
        fail("dev cache path equals the C1 cache")
    if DEV_CACHE.exists():
        fail(f"{relpath(DEV_CACHE)} already exists: move or delete it by hand (e.g. rename it to nuaa_dev.old.json) and rerun, "
             "so that every dev score comes from this run")
    # controlli di separazione rifatti prima dell'inferenza (scrivono anche il rapporto richiesto da select)
    rc = subprocess.run([sys.executable, "scripts/select_threshold.py", "check-separation", "--manifest", relpath(man_p),
                         "--summary", relpath(summ_p), "--report", relpath(DEV_SEPARATION)], cwd=ROOT).returncode
    if rc != 0:
        fail("separation checks failed: dev scoring not started")
    # collegamenti sempre ricreati da zero, solo dentro link_dir, poi verificati per hash sul file a cui puntano
    link_root = link_dir.resolve()
    outside = [r["file"] for r in man if not is_under(link_root / r["file"], link_root)
               or not is_under(source_path(r), NUAA)]
    if outside:
        fail(f"{len(outside)} dev manifest keys would point outside the link folder or the NUAA training split (e.g. {outside[:3]})")
    clear_link_dirs(link_dir)
    for r in man:
        os.symlink(source_path(r).resolve(), link_dir / r["file"])
    wrong = [r["file"] for r in man if sha256((link_dir / r["file"]).resolve()) != r["sha256"]]
    if wrong:
        fail(f"{len(wrong)} linked images differ from the manifest hash (e.g. {wrong[:3]}): dev scoring not started")
    (link_dir / "info.json").write_text(json.dumps({"name": "NUAA source-dev (M1 threshold selection)", "source": "NUAA official training split, validation subjects",
                                                    "note": f"dev manifest sha256 {man_sha}"}, indent=1), encoding="utf-8")
    env = dict(os.environ, LIVENESSLAB_DATA_DIR=relpath(dev_root))
    rc = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env).returncode
    if rc != 0:
        fail(f"eval_dataset.py exited with {rc}")
    cache = load_json(DEV_CACHE)
    check_dev_cache(cache, relpath(DEV_CACHE))
    wanted = {r["file"] for r in man}
    moved = [aid for aid in ids if cache.get("fingerprints", {}).get(aid) != weights[aid]["sha256"]]
    if moved:
        fail(f"fingerprints of the dev cache differ from the weights checked before inference: {moved}")
    for aid in ids:
        per = cache["scores"].get(aid, {})
        if set(per) != wanted:
            fail(f"{aid}: dev cache keys differ from the manifest ({len(per)} vs {len(wanted)})")
    prov = {"cache": relpath(DEV_CACHE), "cache_sha256": sha256(DEV_CACHE), "dev_manifest_sha256": man_sha, "analyzers": ids,
            "fingerprints": {aid: cache.get("fingerprints", {}).get(aid) for aid in ids},
            "weights_checked_before_inference": weights,
            "command": plan["command"], **repo_state(), "date": now()}
    DEV_PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    DEV_PROVENANCE.write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8")
    print(f"== dev scores in {relpath(DEV_CACHE)}, provenance in {relpath(DEV_PROVENANCE)}; next: select_threshold.py select", flush=True)


# ----------------------------------------------------------------------------- select

def threshold_counts(y, s, value):
    """Conteggi alla soglia con la regola operativa (attacco se s > value, pareggio = bona fide)."""
    att, bf = s[y == 1], s[y == 0]
    accepted = int(np.sum(att <= value)); rejected = int(np.sum(bf > value))
    apcer = accepted / len(att) if len(att) else None
    bpcer = rejected / len(bf) if len(bf) else None
    return {"n": int(len(y)), "n_bona_fide": int(len(bf)), "n_attack": int(len(att)),
            "accepted_attacks": accepted, "rejected_bona_fide": rejected, "apcer": apcer, "bpcer": bpcer,
            "acer": (apcer + bpcer) / 2 if apcer is not None and bpcer is not None else None,
            "ties_at_value": int(np.sum(s == value))}


SELECTION_RULE = ("candidates = distinct dev scores plus one value below the minimum; attack if score > value; "
                  "eer: min |fp*n_a - fn*n_b|, ties → highest value; apcer10: max value with APCER <= 0.10")
CRITERION_NOTE = {
    "eer": "eer: balance of APCER and BPCER on the dev scores under the operational rule (attack if score > value): "
           "candidate minimising |fp*n_a - fn*n_b|, ties broken towards the highest value",
    "apcer10": "apcer10: APCER <= 0.10 under the operational rule (attack if score > value) on the dev scores, lowest BPCER, "
               "ties broken towards the highest value (the largest feasible candidate); not the quantile threshold of the "
               "BPCER@APCER10 metric",
}


def candidate_thresholds(s):
    """Soglie candidate, in ordine decrescente: ogni punteggio distinto del dev (di entrambe le classi) più un valore
    appena sotto il minimo, nextafter(min, -inf), solo se resta >= 0 (i valori di threshold.json stanno in [0, 1]):
    con il minimo uguale a 0.0 il candidato sotto il minimo non è rappresentabile e si omette."""
    vals = sorted({float(v) for v in s}, reverse=True)
    low = float(np.nextafter(vals[-1], -np.inf))
    if low >= 0.0:
        vals.append(low)
    return vals


def select_threshold(y, s, criterion: str) -> dict:
    """Sceglie la soglia sui punteggi del dev con la stessa regola con cui si applica (attacco se s > t, pareggio = bona
    fide). fn(t) = attacchi con s <= t (accettati), fp(t) = bona fide con s > t (rifiutati), conteggi interi.
    eer: minimo di |fp·n_a − fn·n_b| (interi Python, esatto), a pari merito la soglia più alta.
    apcer10: tra i candidati con 10·fn <= n_a (APCER <= 0,10 con la regola >), minimo di fp, a pari merito la più alta.
    Restituisce {"value", "fp", "fn", "n_attack", "n_bona_fide"}; None se nessun candidato soddisfa il vincolo di apcer10
    (possibile solo con punteggi di attacco uguali a 0.0, dove il candidato sotto il minimo manca)."""
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    att, bf = np.sort(s[y == 1]), np.sort(s[y == 0])
    n_a, n_b = int(len(att)), int(len(bf))
    if n_a == 0 or n_b == 0:
        raise ValueError("both classes are required to select a threshold")
    best = None
    # ordine decrescente: a pari merito resta il primo trovato, cioè la soglia più alta
    for t in candidate_thresholds(s):
        fn = int(np.searchsorted(att, t, side="right"))
        fp = n_b - int(np.searchsorted(bf, t, side="right"))
        if criterion == "eer":
            key = abs(fp * n_a - fn * n_b)
        elif criterion == "apcer10":
            if 10 * fn > n_a:
                continue
            key = fp
        else:
            raise ValueError(f"unknown criterion {criterion!r}")
        if best is None or key < best[0]:
            best = (key, t, fp, fn)
    if best is None:
        return None
    return {"value": best[1], "fp": best[2], "fn": best[3], "n_attack": n_a, "n_bona_fide": n_b}


def dev_arrays(cache: dict, man_lab: dict, aid: str):
    """Etichette e punteggi del dev di un analizzatore (chiavi in ordine), dopo i controlli contro il manifest."""
    per = cache.get("scores", {}).get(aid)
    if not per:
        fail(f"{aid}: no dev scores in the cache")
    missing = sorted(set(man_lab) - set(per))
    if missing:
        fail(f"{aid}: {len(missing)} dev images without score (e.g. {missing[:3]})")
    extra = sorted(set(per) - set(man_lab))
    if extra:
        fail(f"{aid}: {len(extra)} scored images outside the dev manifest (e.g. {extra[:3]})")
    wrong = [k for k, v in per.items() if v["y"] != man_lab[k]]
    if wrong:
        fail(f"{aid}: {len(wrong)} labels differ from the dev manifest (e.g. {examples(wrong)})")
    keys = sorted(per)
    y = np.array([man_lab[k] for k in keys]); s = np.array([float(per[k]["s"]) for k in keys])
    if not (np.any(y == 0) and np.any(y == 1)):
        fail(f"{aid}: the dev scores contain a single class: no threshold can be selected")
    return y, s


def compute_entries(cache: dict, man_lab: dict, ids, criterion: str, verbose: bool = True) -> dict:
    """Soglie di select per ogni analizzatore (criterio dichiarato), con le voci di threshold.json. Usata da select e,
    in memoria, da apply per rifiutare un threshold.json modificato a mano."""
    from tesi_app.evaluation import compute_metrics
    if criterion not in CRITERIA:
        fail(f"criterion {criterion!r} is not one of {list(CRITERIA)}")
    entries = {}
    for aid in ids:
        y, s = dev_arrays(cache, man_lab, aid)
        m = compute_metrics(y, s)
        sel = select_threshold(y, s, criterion)
        if sel is None:
            fail(f"{aid}: no candidate threshold gives APCER <= 0.10 under the operational rule "
                 "(attack scores equal to 0.0: the candidate below the minimum is not representable)")
        value = float(sel["value"])
        op = threshold_counts(y, s, value)
        if (op["accepted_attacks"], op["rejected_bona_fide"]) != (sel["fn"], sel["fp"]):
            fail(f"{aid}: counts at the selected value {value!r} differ from the selection")
        entries[aid] = {
            "criterion": criterion, "value": value, "decision_rule": "attack if score > value (tie = bona fide)",
            "selection_rule": SELECTION_RULE,
            "fingerprint": cache.get("fingerprints", {}).get(aid),
            "dev": {"at_value_operational_rule_gt": op,
                    # errore bilanciato (APCER + BPCER)/2 alla soglia scelta, con la regola operativa: è l'EER
                    # operativo con il criterio eer, l'ACER alla soglia con apcer10 (etichetta: balanced_error_label)
                    "dev_balanced_error_at_value": op["acer"],
                    # EER della ROC del modulo delle metriche (regola inclusiva >=): solo informativo, non è la base della scelta
                    "roc_eer_information_only": m.get("eer"),
                    "auc": m.get("auc"),
                    "bpcer_at_apcer10_metric": m.get("bpcer_at_apcer10")},
            "note": CRITERION_NOTE[criterion],
        }
        if verbose:
            print(f"== {aid}: {criterion} threshold {value!r} | at threshold (score > value) APCER {op['apcer']} "
                  f"({op['accepted_attacks']}/{op['n_attack']}) BPCER {op['bpcer']} ({op['rejected_bona_fide']}/{op['n_bona_fide']}), "
                  f"{balanced_error_label(criterion)} {op['acer']}, ties {op['ties_at_value']} | ROC EER (information only) {m.get('eer')}", flush=True)
    return entries


def check_round_trip(out_p: Path, cache: dict, man_lab: dict, entries: dict) -> None:
    """Rilegge threshold.json dal disco (andata e ritorno JSON) e ricalcola i conteggi con la regola > per ogni
    analizzatore: devono coincidere con quelli della selezione, altrimenti errore."""
    doc = load_json(out_p)
    for aid, e in entries.items():
        de = (doc.get("analyzers") or {}).get(aid) or {}
        v = de.get("value")
        if not isinstance(v, float) or v != e["value"]:
            fail(f"{out_p}: {aid}: value {v!r} read back differs from the selected value {e['value']!r}")
        y, s = dev_arrays(cache, man_lab, aid)
        got = threshold_counts(y, s, v)
        want = e["dev"]["at_value_operational_rule_gt"]
        if got != want or (de.get("dev") or {}).get("at_value_operational_rule_gt") != want:
            fail(f"{out_p}: {aid}: counts recomputed with score > value after the JSON round trip differ from the selection")
    print(f"== {out_p}: counts of {len(entries)} analyzers recomputed with score > value after the JSON round trip: "
          "equal to the selection (counts verified after the JSON round trip)", flush=True)


def cmd_select(a) -> None:
    cache_p, out_p = Path(a.cache), Path(a.out)
    if cache_p.resolve() == C1_CACHE.resolve():
        fail("the C1 test cache can never be used to select the threshold")
    cache = load_json(cache_p)
    check_dev_cache(cache, relpath(cache_p))
    synthetic = bool(a.synthetic or cache.get("synthetic") is True)
    if synthetic and is_under(out_p, RESULTS):
        fail(f"a threshold from synthetic scores cannot be written under {rel(RESULTS)}: choose --out elsewhere")
    # il test non entra mai nella selezione: né nella cache né nel manifest (nomi del test ufficiale, sessione 03, chiavi C1)
    all_keys = {k for per in cache["scores"].values() for k in per}
    check_not_test(all_keys, "cache", Path(a.c1_manifest), allow_missing=synthetic)
    man_sha, man = verify_dev_manifest(Path(a.manifest), Path(a.summary), Path(a.separation), allow_missing=synthetic)
    if not synthetic:
        # una cache reale vale solo con la provenienza scritta da score-dev per questa cache e questo manifest
        prov_p = Path(a.provenance)
        if not prov_p.exists():
            fail(f"missing {relpath(prov_p)}: a non-synthetic dev cache must come from score-dev")
        prov = load_json(prov_p)
        if prov.get("cache_sha256") != sha256(cache_p) or prov.get("dev_manifest_sha256") != man_sha:
            fail(f"{relpath(prov_p)} does not match this cache and dev manifest")
    man_lab = {r["file"]: int(r["label"]) for r in man}
    extra = sorted(all_keys - set(man_lab))
    if extra:
        fail(f"cache contains {len(extra)} images outside the dev manifest (e.g. {extra[:3]}): refused")
    ids = a.analyzers or ELIGIBLE
    bad = [i for i in ids if i not in ELIGIBLE]
    if bad:
        fail(f"not eligible for a NUAA source-dev threshold: {bad}")
    entries = compute_entries(cache, man_lab, ids, a.criterion)
    doc = {
        "synthetic": synthetic,
        "protocol": "metrics.md, Threshold selection protocol",
        "criterion": a.criterion,
        "dev_set": "NUAA official training split, validation subjects of the NUAA official CNN training",
        "dev_manifest_sha256": man_sha,
        "dev_cache": rel(cache_p), "dev_cache_sha256": sha256(cache_p),
        "analyzers": entries,
        "not_eligible": {**EXCLUDED_NUAA_TRAINED, **EXCLUDED_POOLED,
                         "other analyzers": "source dataset is not NUAA: reported at the fixed threshold 0.5 only"},
        "select_threshold_py_sha256": sha256(Path(__file__).resolve()),
        "evaluation_py_sha256": sha256(ROOT / "src" / "tesi_app" / "evaluation.py"),
        **repo_state(),
        "date": now(),
    }
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    check_round_trip(out_p, cache, man_lab, entries)
    print(f"== written {out_p}" + (" (SYNTHETIC)" if synthetic else ""), flush=True)


# ----------------------------------------------------------------------------- apply

def fmt(v):
    if v is None:
        return "—"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def balanced_error_label(criterion) -> str:
    """Nome di (APCER + BPCER)/2 alla soglia scelta: EER operativo con il criterio eer (soglia di bilanciamento),
    ACER alla soglia con apcer10 (la soglia non bilancia APCER e BPCER, quindi non è un EER)."""
    return "operational dev EER" if criterion == "eer" else "dev ACER at the threshold"


def dev_flag(e) -> str:
    """Segnalazione di una riga di apply: "non-informative (motivi)" se l'AUC del dev è sotto 0,5 oppure l'errore
    bilanciato (APCER + BPCER)/2 alla soglia scelta, con la regola >, è sopra 0,5 (dev_balanced_error_at_value: EER
    operativo con eer, ACER alla soglia con apcer10; mai l'EER della ROC). Oltre 0,5 la soglia fa peggio, in media sulle
    due classi, della regola che accetta tutto (APCER 1, BPCER 0). Nessuna inversione dei punteggi: la soglia si applica
    comunque. Stringa vuota se nessuna condizione vale."""
    dev = e.get("dev", {}) if isinstance(e, dict) else {}
    auc, bal = dev.get("auc"), dev.get("dev_balanced_error_at_value")
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    label = balanced_error_label(e.get("criterion") if isinstance(e, dict) else None)
    reasons = ([] if not (num(auc) and auc < 0.5) else ["dev AUC < 0.5"]) + ([] if not (num(bal) and bal > 0.5) else [f"{label} > 0.5"])
    return f"non-informative ({', '.join(reasons)})" if reasons else ""


def number_word(n: int) -> str:
    return {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}.get(n, str(n))


def check_threshold_doc(thr: dict, what: str) -> None:
    """threshold.json ben formato: criterio ammesso, synthetic vero o falso, i quattro analizzatori ammessi, ogni valore
    finito in [0, 1] con lo stesso criterio del documento."""
    if thr.get("criterion") not in CRITERIA:
        fail(f"{what}: criterion {thr.get('criterion')!r} is not one of {list(CRITERIA)}")
    if "synthetic" not in thr or not isinstance(thr["synthetic"], bool):
        fail(f"{what}: 'synthetic' must be present and be true or false")
    for k in ("dev_manifest_sha256", "dev_cache_sha256", "dev_cache"):
        if not isinstance(thr.get(k), str) or not thr[k]:
            fail(f"{what}: missing or invalid '{k}'")
    an = thr.get("analyzers")
    if not isinstance(an, dict):
        fail(f"{what}: 'analyzers' is not an object")
    if set(an) != set(ELIGIBLE):
        fail(f"{what}: analyzers {sorted(an)} differ from the eligible set {sorted(ELIGIBLE)} (all four are required)")
    for aid, e in an.items():
        if not isinstance(e, dict):
            fail(f"{what}: entry of {aid} is not an object")
        v = e.get("value")
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or not 0.0 <= v <= 1.0:
            fail(f"{what}: {aid}: value {v!r} is not a finite number in [0, 1]")
        if e.get("criterion") != thr["criterion"]:
            fail(f"{what}: {aid}: criterion {e.get('criterion')!r} differs from the document criterion {thr['criterion']!r}")


def resolve_dev_cache(thr: dict, thr_p: Path, override) -> Path:
    """Cache del dev a cui rimanda threshold.json: --dev-cache, oppure il percorso registrato (relativo al repository),
    oppure lo stesso nome accanto a threshold.json o nella cartella eval dei risultati; l'hash deve coincidere."""
    if override:
        cands = [Path(override)]
    else:
        name = Path(thr["dev_cache"]).name
        cands = [ROOT / thr["dev_cache"], Path(thr_p).parent / name, RESULTS / "eval" / name]
    for c in cands:
        if c.is_file() and sha256(c) == thr["dev_cache_sha256"]:
            return c
    fail(f"dev cache of threshold.json not found with sha256 {thr['dev_cache_sha256']} (looked in {[relpath(c) for c in cands]}); "
         "pass --dev-cache")


def cmd_apply(a) -> None:
    from tesi_app.evaluation import THRESHOLD, compute_metrics
    thr_p = Path(a.threshold)
    thr = load_json(thr_p)
    check_threshold_doc(thr, relpath(thr_p))
    thr_sha = sha256(thr_p)
    synthetic = thr["synthetic"]
    out_dir = Path(a.out_dir)
    if synthetic and is_under(out_dir, RESULTS):
        fail(f"tables from a synthetic threshold cannot be written under {rel(RESULTS)}: choose --out-dir elsewhere")
    man_p = Path(a.manifest)
    summary_p = Path(a.summary) if a.summary else man_p.parent / "nuaa_dev_summary.json"
    separation_p = Path(a.separation) if a.separation else man_p.parent / "nuaa_dev_separation.json"
    man_sha, man = verify_dev_manifest(man_p, summary_p, separation_p, allow_missing=synthetic)
    if man_sha != thr["dev_manifest_sha256"]:
        fail("threshold.json was selected on a different dev manifest")
    # ricalcolo di select in memoria sulla cache del dev registrata: un valore modificato a mano viene rifiutato
    cache_p = resolve_dev_cache(thr, thr_p, a.dev_cache)
    cache = load_json(cache_p)
    check_dev_cache(cache, relpath(cache_p))
    if cache.get("synthetic") is True and not synthetic:
        fail("threshold.json says synthetic false but its dev cache is synthetic")
    if not synthetic:
        # come in select: una cache reale vale solo con la provenienza scritta da score-dev
        prov = load_json(a.provenance, f"{relpath(a.provenance)} (provenance of the dev cache written by score-dev)")
        if prov.get("cache_sha256") != sha256(cache_p) or prov.get("dev_manifest_sha256") != man_sha:
            fail(f"{relpath(a.provenance)} does not match the dev cache and dev manifest of threshold.json")
    man_lab = {r["file"]: int(r["label"]) for r in man}
    recomputed = json.loads(json.dumps(compute_entries(cache, man_lab, ELIGIBLE, thr["criterion"], verbose=False)))
    differ = sorted(aid for aid in ELIGIBLE if recomputed[aid] != thr["analyzers"][aid])
    if differ:
        fail(f"threshold.json differs from a recomputation of select on its dev cache for {differ} "
             "(value or dev figures edited by hand?): refused")
    # cache C1 dei 300 di test: deve essere quella consegnata con C1 (SHA-256 atteso), prima di leggerne un punteggio
    c1_sha = sha256(a.c1_cache) if Path(a.c1_cache).is_file() else None
    expected_c1 = (a.expected_c1_cache_sha256 or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_c1):
        fail(f"--expected-c1-cache-sha256 {a.expected_c1_cache_sha256!r} is not a SHA-256 (64 hexadecimal characters)")
    if c1_sha is None:
        fail(f"missing {relpath(a.c1_cache)}")
    if c1_sha != expected_c1:
        fail(f"{relpath(a.c1_cache)}: SHA-256 {c1_sha} differs from the expected C1 cache {expected_c1} "
             "(--expected-c1-cache-sha256; default: the value recorded at the C1 delivery): refused")
    c1 = load_json(a.c1_cache)                                              # sola lettura
    c1_man = read_manifest(a.c1_manifest, ["file", "label", "subject"])
    c1_lab = {r["file"]: int(r["label"]) for r in c1_man}
    if len(c1_lab) != len(c1_man):
        fail(f"{relpath(a.c1_manifest)}: duplicated keys")
    c1_subject = {r["file"]: r["subject"] for r in c1_man}
    dev_subjects = sorted({r["subject"] for r in man})
    rows, sens_rows, per_image = [], [], []
    for aid in ELIGIBLE:
        e = thr["analyzers"][aid]
        per = c1.get("scores", {}).get(aid, {})
        if not isinstance(per, dict) or set(per) != set(c1_lab):
            fail(f"{aid}: C1 cache keys differ from the C1 manifest")
        if c1.get("fingerprints", {}).get(aid) != e.get("fingerprint"):
            fail(f"{aid}: fingerprint of the C1 scores differs from the one of the dev scores (different model)")
        check_entries(aid, per, relpath(a.c1_cache))
        if any(per[k]["y"] != c1_lab[k] for k in per):
            fail(f"{aid}: C1 cache labels differ from the C1 manifest")
        keys = sorted(per)
        y = np.array([c1_lab[k] for k in keys]); s = np.array([float(per[k]["s"]) for k in keys])
        base, dev = compute_metrics(y, s, THRESHOLD), compute_metrics(y, s, e["value"])
        cb, cd = threshold_counts(y, s, THRESHOLD), threshold_counts(y, s, e["value"])
        for k in ("apcer", "bpcer"):
            if not (base[k] == cb[k] and dev[k] == cd[k]):         # stesso conteggio del modulo delle metriche
                fail(f"{aid}: {k} of the metric module differs from the direct count")
        rows.append((aid, e, base, cb, cd))
        # sensibilità: senza le immagini dei soggetti del dev (identità condivisa, clarification (c) di metrics.md)
        keep = np.array([c1_subject[k] not in dev_subjects for k in keys])
        ys, ss = y[keep], s[keep]
        sens_rows.append((aid, e, compute_metrics(ys, ss, THRESHOLD), threshold_counts(ys, ss, THRESHOLD), threshold_counts(ys, ss, e["value"]),
                          int(np.sum((~keep) & (y == 0))), int(np.sum((~keep) & (y == 1)))))
        for k, yy, ss_ in zip(keys, y, s):
            per_image.append([aid, k, int(yy), repr(float(ss_)), int(ss_ > THRESHOLD), e["value"], int(ss_ > e["value"])])
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "SYNTHETIC — not a result. " if synthetic else ""

    def dev_auc(e):
        v = e.get("dev", {}).get("auc")
        return fmt(float(v)) if isinstance(v, (int, float)) else "—"

    head_cols = ("| analyzer | n | n_bona_fide | n_attack | APCER@0.5 | BPCER@0.5 | ACER@0.5 | accepted attacks@0.5 | rejected bona fide@0.5 | "
                 "dev threshold | APCER@dev | BPCER@dev | ACER@dev | accepted attacks@dev | rejected bona fide@dev | test EER | test AUC | " + balanced_error_label(thr["criterion"]) + " | dev AUC | dev flag |")
    sep_cols = "|---|" + "---|" * 19

    def line(aid, e, base, cb, cd):
        dev_eer = e.get("dev", {}).get("dev_balanced_error_at_value")
        return (f"| {aid} | {cb['n']} | {cb['n_bona_fide']} | {cb['n_attack']} | {fmt(cb['apcer'])} | {fmt(cb['bpcer'])} | {fmt(cb['acer'])} | "
                f"{cb['accepted_attacks']} | {cb['rejected_bona_fide']} | {e['value']:.6g} | {fmt(cd['apcer'])} | {fmt(cd['bpcer'])} | {fmt(cd['acer'])} | "
                f"{cd['accepted_attacks']} | {cd['rejected_bona_fide']} | {fmt(base.get('eer'))} | {fmt(base.get('auc'))} | "
                f"{fmt(float(dev_eer)) if isinstance(dev_eer, (int, float)) else '—'} | {dev_auc(e)} | {dev_flag(e)} |")

    md = [f"# NUAA test (300 images, C1 cache): fixed threshold 0.5 vs source-dev threshold",
          "",
          f"{tag}Criterion `{thr['criterion']}`; threshold.json sha256 `{thr_sha}`, commit `{thr.get('commit')}`, date {thr.get('date')}; "
          f"dev manifest sha256 `{thr.get('dev_manifest_sha256')}`; dev cache sha256 `{thr.get('dev_cache_sha256')}`; C1 cache sha256 `{c1_sha}` (equal to the expected value). "
          "Decision: attack if score > threshold (tie = bona fide). The source-dev threshold was chosen on the dev set only, with the same rule, and applied "
          "unchanged (its values were recomputed from the dev cache before this table was written). NUAA is not subject-disjoint between "
          "official train and test (dev subjects appear in the test split, different capture session).",
          "",
          "No score inversion is applied: when the dev AUC is below 0.5 the scores are used as they are. The threshold is applied in every "
          f"case; a row whose dev AUC is below 0.5 or whose {balanced_error_label(thr['criterion'])} is above 0.5 is flagged `non-informative` "
          f"in the `dev flag` column, with the reason (`dev AUC < 0.5`, `{balanced_error_label(thr['criterion'])} > 0.5`, or both); above 0.5 the "
          "threshold does worse, averaged over the two classes, than accepting every sample.",
          "",
          "Columns `@0.5`: fixed threshold 0.5; columns `@dev`: source-dev threshold; `test EER` and `test AUC` do not depend on the threshold; "
          f"`{balanced_error_label(thr['criterion'])}` is (APCER + BPCER)/2 on the dev set at the selected threshold with the rule "
          "score > threshold (`dev_balanced_error_at_value` of threshold.json: the operational dev EER with criterion `eer`, the dev ACER at "
          "the threshold with criterion `apcer10`; the ROC EER of the dev scores is recorded in threshold.json for information only); "
          "`dev AUC` is that of threshold.json.",
          "",
          head_cols, sep_cols]
    md += [line(*r) for r in rows]
    md += ["", "Other analyzers have no NUAA source-dev threshold and are reported at 0.5 only (C1 table): " +
           "; ".join(f"{k}: {v}" for k, v in thr.get("not_eligible", {}).items())]
    n_bf_out, n_att_out = sens_rows[0][5], sens_rows[0][6]
    md += ["", "## Sensitivity (not the main result): test images of the dev subjects removed", "",
           f"{tag}Same thresholds and rule as above, on the test images whose subject is not one of the {number_word(len(dev_subjects))} dev subjects: "
           f"{n_att_out} attacks and {n_bf_out} bona fide removed. The dev subjects appear among the test images with a different capture "
           "session; this block shows how much the table above depends on those images. It is a sensitivity check, not a replacement of the table above.",
           "", head_cols, sep_cols]
    md += [line(aid, e, base, cb, cd) for aid, e, base, cb, cd, _, _ in sens_rows]
    md_p, csv_p = out_dir / "nuaa_test_dev_threshold_table.md", out_dir / "nuaa_test_dev_threshold_scores.csv"
    md_p.write_text("\n".join(md) + "\n", encoding="utf-8")
    with csv_p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["analyzer", "file", "label", "score_attack", "prediction_at_0_5", "dev_threshold", "prediction_at_dev_threshold"])
        w.writerows(per_image)
    print("\n".join(md), flush=True)
    print(f"== written {md_p} and {csv_p}" + (" (SYNTHETIC)" if synthetic else ""), flush=True)


# ----------------------------------------------------------------------------- prova sintetica

SYNTHETIC_README = (
    "# Synthetic trial of the threshold flow: NOT results\n\n"
    "These files are NOT results. They show that `select_threshold.py select` and `apply` run end to end: the dev scores "
    "are synthetic (drawn deterministically, seed {seed}: bona fide ~ Beta(2, 5), attacks ~ Beta(5, 2), one value per "
    "image of the dev manifest and eligible analyzer, analyzers in the order {analyzers}, images in the order of the manifest "
    "keys), never computed on any image. The threshold in `threshold.json` (`\"synthetic\": true`) and the test table at "
    "that threshold are therefore meaningless as measurements. The 300 test scores are the real C1 cache, read only.\n\n"
    "- synthetic dev cache SHA-256: `{cache_sha}` ({n_scores} scores; rebuilt identically by the command below)\n"
    "- dev manifest SHA-256: `{man_sha}`\n"
    "- test images refused by: {test_rule}\n"
    "- reproduce: `python scripts/select_threshold.py synthetic-trial --out-dir <folder outside results/>`\n")


def synthetic_dev_cache(man, c1_cache: dict, seed: int = SEED) -> dict:
    """Cache dev fittizia e deterministica: per ogni analizzatore ammesso (ordine di ELIGIBLE) e ogni immagine del
    manifest (chiavi in ordine), bona fide ~ Beta(2, 5) e attacchi ~ Beta(5, 2) con np.random.default_rng(seed).
    Impronte copiate dalla cache C1, così `apply` accetta la soglia sugli stessi modelli. Mai inferenza."""
    rng = np.random.default_rng(seed)
    rows = sorted((r["file"], int(r["label"])) for r in man)
    scores = {}
    for aid in ELIGIBLE:
        scores[aid] = {k: {"y": y, "s": float(rng.beta(5, 2) if y == 1 else rng.beta(2, 5))} for k, y in rows}
    fps = c1_cache.get("fingerprints", {})
    return {"dataset": DEV_ID, "synthetic": True, "seed": seed,
            "note": "SYNTHETIC dev scores for the trial of the threshold flow: not computed on any image, not results",
            "scores": scores, "fingerprints": {aid: fps.get(aid) for aid in ELIGIBLE}}


def cmd_synthetic_trial(a) -> None:
    """Prova del flusso select + apply su una cache dev sintetica, in una cartella fuori dai risultati."""
    out_dir = Path(a.out_dir)
    if is_under(out_dir, RESULTS):
        fail(f"the synthetic trial cannot be written under {rel(RESULTS)}: choose --out-dir elsewhere")
    for p in (a.manifest, a.summary, a.separation, a.c1_cache, a.c1_manifest):
        if not Path(p).exists():
            fail(f"missing {relpath(p)}: needed by the synthetic trial")
    if split_present("validation") and split_present("training"):
        test_rule = f"official test split names ({EXPECTED_TEST} images), session {TEST_SESSION} in the name, C1 manifest; dev manifest checked against the images"
    else:
        test_rule = (f"session {TEST_SESSION} in the name and C1 manifest only (NUAA image folders not found); "
                     "dev manifest checked against the summary and the separation report only")
        note(f"NUAA image folders not found in {nuaa_where()}: the trial refuses test images by the name rule (5th field {TEST_SESSION}) "
             "and the C1 manifest only, and checks the dev manifest against the summary and the separation report only")
    out_dir.mkdir(parents=True, exist_ok=True)
    man = read_manifest(a.manifest, DEV_COLUMNS)
    problems = manifest_row_problems(man)
    if problems:
        fail(f"{relpath(a.manifest)}: " + "; ".join(problems))
    cache = synthetic_dev_cache(man, load_json(a.c1_cache), a.seed)
    cache_p = out_dir / "nuaa_dev_synthetic_cache.json"
    cache_p.write_text(json.dumps(cache, indent=1) + "\n", encoding="utf-8")
    me = str(Path(__file__).resolve())
    steps = [[sys.executable, me, "select", "--cache", str(cache_p), "--manifest", str(a.manifest), "--summary", str(a.summary),
              "--separation", str(a.separation), "--c1-manifest", str(a.c1_manifest), "--criterion", a.criterion,
              "--out", str(out_dir / "threshold.json"), "--synthetic"],
             [sys.executable, me, "apply", "--threshold", str(out_dir / "threshold.json"), "--c1-cache", str(a.c1_cache),
              "--c1-manifest", str(a.c1_manifest), "--manifest", str(a.manifest), "--summary", str(a.summary),
              "--separation", str(a.separation), "--dev-cache", str(cache_p), "--out-dir", str(out_dir),
              "--expected-c1-cache-sha256", a.expected_c1_cache_sha256]]
    for cmd in steps:
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc != 0:
            fail(f"synthetic trial: `{cmd[2]}` exited with {rc}")
    (out_dir / "README.md").write_text(SYNTHETIC_README.format(
        seed=a.seed, analyzers=", ".join(ELIGIBLE), cache_sha=sha256(cache_p),
        n_scores=sum(len(v) for v in cache["scores"].values()), man_sha=sha256(a.manifest), test_rule=test_rule), encoding="utf-8")
    print(f"== synthetic trial in {out_dir} (NOT results)", flush=True)


# ----------------------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build-dev", help="costruisce il manifest del dev NUAA (solo elenco dei file e hash)")
    p.add_argument("--out-dir", default=str(M1_DIR), help="cartella di uscita (default results/m1; altra cartella per le prove)")
    p = sub.add_parser("check-separation", help="controlli del manifest e di separazione automatici (codice 1 se uno fallisce)")
    p.add_argument("--manifest", default=str(M1_DIR / "nuaa_dev_manifest.csv"), help="manifest del dev")
    p.add_argument("--summary", default=None, help="riepilogo di build-dev (default nuaa_dev_summary.json accanto al manifest)")
    p.add_argument("--c1-manifest", default=str(C1_MANIFEST), help="manifest dei 300 di test di C1")
    p.add_argument("--report", default=None, help="rapporto JSON (default nuaa_dev_separation.json accanto al manifest)")
    p = sub.add_parser("score-dev", help="punteggi delle immagini del dev (nuova inferenza: serve --i-am-authorized)")
    p.add_argument("--manifest", default=str(M1_DIR / "nuaa_dev_manifest.csv"), help="manifest del dev")
    p.add_argument("--summary", default=str(DEV_SUMMARY), help="riepilogo del dev scritto da build-dev (obbligatorio)")
    p.add_argument("--analyzers", nargs="*", default=None, help=f"sottoinsieme di {ELIGIBLE}")
    p.add_argument("--dev-root", default=str(DEV_ROOT), help="radice dati alternativa con eval/nuaa_dev (collegamenti alle immagini)")
    p.add_argument("--dry-run", action="store_true", help="mostra il piano senza eseguire nulla")
    p.add_argument("--i-am-authorized", action="store_true", help="autorizzazione esplicita all'inferenza sulle immagini del dev")
    p = sub.add_parser("select", help="sceglie la soglia sui punteggi del dev e scrive threshold.json")
    p.add_argument("--cache", default=str(DEV_CACHE), help="cache dei punteggi del dev")
    p.add_argument("--manifest", default=str(M1_DIR / "nuaa_dev_manifest.csv"), help="manifest del dev")
    p.add_argument("--summary", default=str(DEV_SUMMARY), help="riepilogo del dev (hash e conteggi del manifest)")
    p.add_argument("--separation", default=str(DEV_SEPARATION), help="rapporto dei controlli di separazione (passed = true, problems vuoto)")
    p.add_argument("--provenance", default=str(DEV_PROVENANCE), help="provenienza della cache scritta da score-dev (obbligatoria se non sintetica)")
    p.add_argument("--c1-manifest", default=str(C1_MANIFEST), help="manifest dei 300 di test di C1")
    p.add_argument("--criterion", choices=CRITERIA, default="eer", help="criterio: eer (principale, bilanciamento di APCER e BPCER con la regola >) o apcer10 (alternativa, APCER <= 0,10 con la regola >)")
    p.add_argument("--analyzers", nargs="*", default=None, help=f"sottoinsieme di {ELIGIBLE} (apply richiede tutti e quattro)")
    p.add_argument("--out", default=str(M1_DIR / "threshold.json"), help="file threshold.json da scrivere")
    p.add_argument("--synthetic", action="store_true",
                   help="marca l'uscita come ottenuta da punteggi sintetici (senza cartelle NUAA vale solo la regola del nome)")
    p = sub.add_parser("apply", help="verifica threshold.json e lo applica ai 300 di test della cache C1 (sola lettura)")
    p.add_argument("--threshold", default=str(M1_DIR / "threshold.json"), help="threshold.json prodotto da select")
    p.add_argument("--c1-cache", default=str(C1_CACHE), help="cache dei punteggi di C1 (sola lettura)")
    p.add_argument("--c1-manifest", default=str(C1_MANIFEST), help="manifest dei 300 di test di C1")
    p.add_argument("--manifest", default=str(M1_DIR / "nuaa_dev_manifest.csv"), help="manifest del dev (obbligatorio)")
    p.add_argument("--summary", default=None, help="riepilogo del dev (default nuaa_dev_summary.json accanto al manifest)")
    p.add_argument("--separation", default=None, help="rapporto di separazione (default nuaa_dev_separation.json accanto al manifest)")
    p.add_argument("--provenance", default=str(DEV_PROVENANCE), help="provenienza della cache del dev (obbligatoria se non sintetica)")
    p.add_argument("--dev-cache", default=None,
                   help="cache del dev su cui ricalcolare select (default: quella registrata in threshold.json, cercata per hash)")
    p.add_argument("--expected-c1-cache-sha256", default=C1_CACHE_SHA256,
                   help="SHA-256 atteso della cache C1 (default: il valore registrato alla consegna di C1, "
                        "SHA256SUMS della cartella riservata del run C1); un'altra cache viene rifiutata; 64 caratteri esadecimali "
                        "(spazi esterni tolti, maiuscole accettate), un valore vuoto o malformato è un errore")
    p.add_argument("--out-dir", default=str(M1_DIR), help="cartella di uscita delle tabelle")
    p = sub.add_parser("synthetic-trial",
                       help="prova di select + apply su punteggi dev SINTETICI (seed fisso), fuori dai risultati; senza le cartelle "
                            "NUAA le immagini del test si escludono solo con la regola del nome (sessione 03) e il manifest C1")
    p.add_argument("--out-dir", required=True, help="cartella di uscita, fuori dalla cartella dei risultati")
    p.add_argument("--manifest", default=str(M1_DIR / "nuaa_dev_manifest.csv"), help="manifest del dev")
    p.add_argument("--summary", default=str(DEV_SUMMARY), help="riepilogo del dev (hash e conteggi del manifest)")
    p.add_argument("--separation", default=str(DEV_SEPARATION), help="rapporto dei controlli di separazione (passed = true, problems vuoto)")
    p.add_argument("--c1-cache", default=str(C1_CACHE), help="cache dei punteggi di C1 (sola lettura: impronte e 300 di test)")
    p.add_argument("--c1-manifest", default=str(C1_MANIFEST), help="manifest dei 300 di test di C1")
    p.add_argument("--criterion", choices=CRITERIA, default="eer", help="criterio da provare (default eer)")
    p.add_argument("--seed", type=int, default=SEED, help=f"seme dei punteggi sintetici (default {SEED})")
    p.add_argument("--expected-c1-cache-sha256", default=C1_CACHE_SHA256,
                   help="SHA-256 atteso della cache C1, passato ad apply (default: il valore registrato alla consegna di C1); "
                        "un valore vuoto o malformato è un errore prima di qualsiasi passo")
    a = ap.parse_args()
    # SHA-256 atteso della cache C1 (apply, synthetic-trial): normalizzato (spazi esterni tolti, minuscole) e verificato
    # prima di qualsiasi passo; un valore vuoto, di soli spazi o malformato è un errore con codice 1
    if getattr(a, "expected_c1_cache_sha256", None) is not None:
        norm = a.expected_c1_cache_sha256.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", norm):
            fail(f"--expected-c1-cache-sha256 {a.expected_c1_cache_sha256!r} is not a SHA-256 (64 hexadecimal characters after "
                 "removing surrounding whitespace; upper case is accepted and lowered): nothing run")
        a.expected_c1_cache_sha256 = norm
    {"build-dev": cmd_build_dev, "check-separation": cmd_check_separation, "score-dev": cmd_score_dev,
     "select": cmd_select, "apply": cmd_apply, "synthetic-trial": cmd_synthetic_trial}[a.cmd](a)


if __name__ == "__main__":
    main()
